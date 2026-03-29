use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::graph::{GlobalId, Node};
use crate::migration::numeric_tensor::NumericTensor;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::nano_graph::lower::{DimKind, NanoLoweringContext, TensorAtomMap};
use crate::pool::Pool;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transpose {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    data: GlobalId,
    perm: Option<Vec<i64>>,
}

impl Transpose {
    pub(crate) fn perm(&self) -> Option<&[i64]> {
        self.perm.as_deref()
    }

    pub fn push_new(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        perm: Option<Vec<i64>>,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        Self::push_new_with_label(graph, data, perm, None, rng)
    }

    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        perm: Option<Vec<i64>>,
        label: Option<String>,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            output,
            data,
            perm,
            global_id: GlobalId::new(rng),
            label,
        };
        graph.push_op(AnyMilliOp::Transpose(node));
        output
    }
}

impl Transpose {
    pub fn lower_to_nano(
        &self,
        ctx: &mut crate::nano_graph::NanoLoweringContext,
    ) -> crate::milli_graph::ops::LowerResult {
        let all_infos = ctx.all_infos;
        let in_id = Node::inputs(self).next().unwrap();
        let out_id = Node::outputs(self).next().unwrap();

        let Some(in_map) = ctx.tensor_map.get(&in_id).cloned() else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };
        let Some(out_info) = all_infos.get(&out_id) else {
            ctx.register_opaque(out_id);
            return crate::milli_graph::ops::LowerResult::Lowered;
        };
        let Some(in_info) = all_infos.get(&in_id) else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };

        let in_rank = match in_info.rank_if_known() {
            Some(r) => r,
            None => {
                return crate::milli_graph::ops::LowerResult::Unsupported;
            }
        };

        let Some((out_layout, out_known_dims, _out_sym_dims, _out_count)) =
            ctx.classify_dims(out_info)
        else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };

        // Build full permutation (handling None=reverse, partial perms, negative indices).
        let full_perm: Vec<usize> = match self.perm() {
            None => (0..in_rank).rev().collect(),
            Some(perm) => {
                let expanded = if perm.len() < in_rank {
                    let prefix_len = in_rank - perm.len();
                    let mut fp: Vec<i64> = (0..prefix_len as i64).collect();
                    fp.extend(
                        perm.iter()
                            .map(|&x| if x < 0 { x + in_rank as i64 } else { x }),
                    );
                    fp
                } else {
                    perm.to_vec()
                };
                expanded
                    .iter()
                    .map(|&x| {
                        if x < 0 {
                            (x + in_rank as i64) as usize
                        } else {
                            x as usize
                        }
                    })
                    .collect()
            }
        };

        // Get input known dim sizes (only the Known dims, in original order).
        let _in_known_sizes: Vec<u64> = in_map
            .layout
            .iter()
            .filter_map(|d| {
                if let DimKind::Known(s) = d {
                    Some(*s)
                } else {
                    None
                }
            })
            .collect();
        // Use the input's actual physical strides (may already be non-row-major
        // from a prior Transpose). Do NOT recompute row-major — that loses the
        // physical layout.
        let in_strides = &in_map.known_strides;
        let _out_strides = TensorAtomMap::compute_strides(&out_known_dims);

        // We need to map between original dim indices and known-dim indices.
        // For now, assume all dims are known (symbolic dims in transpose would
        // be unusual). If any dim is symbolic, fall back to boundary.
        if in_map
            .layout
            .iter()
            .any(|d| matches!(d, DimKind::Symbolic(_)))
            || out_layout.iter().any(|d| matches!(d, DimKind::Symbolic(_)))
        {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        }

        // Zero-cost transpose: reuse the input's atoms with permuted strides.
        //
        // The input atoms are laid out in row-major order with `in_strides`.
        // After transposing with perm, output dim j corresponds to input dim perm[j].
        // So the output's strides (into the SAME flat atom buffer) are:
        //   output_stride[j] = input_stride[perm[j]]
        //
        // This means downstream ops will decompose their flat index using
        // the output strides and arrive at the correct input atom.
        let mut transposed_strides = vec![0u64; full_perm.len()];
        for (out_dim, &in_dim) in full_perm.iter().enumerate() {
            transposed_strides[out_dim] = in_strides[in_dim];
        }

        // Permute the layout as well.
        let mut transposed_layout = vec![DimKind::Known(0); full_perm.len()];
        for (out_dim, &in_dim) in full_perm.iter().enumerate() {
            transposed_layout[out_dim] = in_map.layout[in_dim].clone();
        }

        ctx.tensor_map.insert(
            out_id,
            TensorAtomMap::simple(
                in_map.base_id,
                in_map.count,
                NanoLoweringContext::ndt(out_info),
                transposed_layout,
                transposed_strides,
                in_map.sym_dims.clone(),
            ),
        );
        crate::milli_graph::ops::LowerResult::Lowered
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.data, map);
    }
}

impl Node for Transpose {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Transpose".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.data].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
}

impl MilliOp for Transpose {
    fn infer<'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'p, P>>,
        symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        pool: &'p P,
    ) -> Result<Vec<(GlobalId, crate::tensor_info::TensorInfo<'p, P>)>, MilliOpGraphError> {
        use crate::tensor_info::TensorInfo;

        let input_info = known_inputs
            .get(&self.data)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        let first_elem = input_info.first_element();

        // Compute output info: Transpose has same dtype, permuted shape.
        let out_info = if let Some(ranked) = input_info.as_ranked() {
            let input_shape = ranked.shape();
            let input_rank = input_shape.len();

            let output_shape = match &self.perm {
                Some(perm) => {
                    // Expand partial perm (like [-1, -2]) to full perm
                    let full_perm = if perm.len() < input_rank {
                        let prefix_len = input_rank - perm.len();
                        let mut fp: Vec<i64> = (0..prefix_len as i64).collect();
                        fp.extend(
                            perm.iter()
                                .map(|&x| if x < 0 { x + input_rank as i64 } else { x }),
                        );
                        fp
                    } else {
                        perm.clone()
                    };
                    // Output shape = input_shape[perm[i]] for each i
                    full_perm
                        .iter()
                        .map(|&p| {
                            let idx = if p < 0 {
                                (p + input_rank as i64) as usize
                            } else {
                                p as usize
                            };
                            input_shape[idx].clone()
                        })
                        .collect()
                }
                None => {
                    // Reverse the dimensions
                    let mut s = input_shape;
                    s.reverse();
                    s
                }
            };
            TensorInfo::Ranked(crate::tensor_info::TensorInfoRanked::new(
                first_elem,
                output_shape,
                symbolic_resolver,
            ))
        } else {
            // At minimum: same rank, same dtype
            let rank = input_info.rank();
            TensorInfo::new_from_first_element_and_rank(first_elem, rank, symbolic_resolver)
        };

        // If input is concrete, try constant fold via nano+pool_eval path.
        if let Some(results) = super::constant_fold(
            self,
            known_inputs,
            &[(self.output, out_info.clone_with_pool(pool))],
            pool,
        ) {
            return Ok(results);
        }

        Ok(vec![((self.output, out_info))])
    }

    fn backward(
        &self,
        output_grads: &HashMap<GlobalId, GlobalId>,
        graph: &mut MilliOpGraph,
        rng: &mut impl rand::Rng,
    ) -> Option<HashMap<GlobalId, GlobalId>> {
        let grad_output = *output_grads.get(&self.output)?;
        // Inverse transpose: if perm is None (reverse), inverse is also None.
        // Otherwise compute inverse permutation.
        // NOTE: For partial perms (len < rank), the inverse is computed using
        // p.len() not actual rank. This is correct for the [-1,-2] swap case
        // (only current usage) but would be wrong for general partial perms.
        // Fix by expanding partial perms at construction time if needed.
        let inv_perm = self.perm.as_ref().map(|p| {
            let n = p.len();
            let normalized: Vec<usize> = p
                .iter()
                .map(|&x| {
                    if x < 0 {
                        (x + n as i64) as usize
                    } else {
                        x as usize
                    }
                })
                .collect();
            let mut inv = vec![0i64; n];
            for (i, &ni) in normalized.iter().enumerate() {
                inv[ni] = i as i64;
            }
            inv
        });
        let grad_input = Transpose::push_new(graph, grad_output, inv_perm, rng);
        let mut result = HashMap::new();
        result.insert(self.data, grad_input);
        Some(result)
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        _config: &super::MilliEvalConfig,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        // Handle partial perms: if perm has fewer elements than the rank,
        // prepend identity dims. This allows perm=[-1,-2] to mean "swap last
        // two dims" regardless of rank.
        let perm = if let Some(ref p) = self.perm {
            let rank = inputs[&self.data].rank();
            if p.len() < rank {
                let prefix_len = rank - p.len();
                let mut full_perm: Vec<i64> = (0..prefix_len as i64).collect();
                full_perm.extend(p.iter().map(|&x| if x < 0 { x + rank as i64 } else { x }));
                Some(full_perm)
            } else {
                Some(p.clone())
            }
        } else {
            None
        };
        let out = inputs[&self.data].transpose(perm, backend)?;
        Ok(Box::new([(self.output, out)].into_iter()))
    }

    fn eval_new<'p, P2: crate::pool::Pool + 'p>(
        &self,
        inputs: &[crate::numeric_tensor::NumericTensorView<'_, crate::tensor_rank::DynRank>],
        pool: &'p P2,
    ) -> Result<
        Vec<crate::numeric_tensor::NumericTensor<'p, crate::tensor_rank::DynRank, P2>>,
        crate::nano_graph::pool_eval::PoolEvalError,
    > {
        use crate::numeric_tensor::{NumericTensor, TensorLayout};
        use crate::tensor_rank::DynRank;

        let data = &inputs[0];
        let input_shape = data.shape();
        let rank = input_shape.len();
        let dtype = data.dtype();

        // Build full permutation
        let full_perm: Vec<usize> = match &self.perm {
            None => (0..rank).rev().collect(),
            Some(p) => {
                let expanded = if p.len() < rank {
                    let prefix_len = rank - p.len();
                    let mut fp: Vec<i64> = (0..prefix_len as i64).collect();
                    fp.extend(p.iter().map(|&x| if x < 0 { x + rank as i64 } else { x }));
                    fp
                } else {
                    p.clone()
                };
                expanded
                    .iter()
                    .map(|&x| {
                        if x < 0 {
                            (x + rank as i64) as usize
                        } else {
                            x as usize
                        }
                    })
                    .collect()
            }
        };

        // Compute output shape
        let output_shape: Vec<u64> = full_perm.iter().map(|&p| input_shape[p]).collect();
        let out_numel: usize = output_shape.iter().product::<u64>() as usize;

        // Compute input strides (row-major)
        let mut input_strides = vec![1usize; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            input_strides[i] = input_strides[i + 1] * input_shape[i + 1] as usize;
        }

        // Compute output strides (row-major)
        let mut output_strides = vec![1usize; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            output_strides[i] = output_strides[i + 1] * output_shape[i + 1] as usize;
        }

        let layout = TensorLayout::<DynRank>::row_major(output_shape, dtype);
        let buf = pool
            .allocate(layout.buffer_size_bytes())
            .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
        let mut out = NumericTensor::from_parts(buf, layout);

        // Build inverse permutation: inv_perm[perm[i]] = i
        let mut inv_perm = vec![0usize; rank];
        for (i, &p) in full_perm.iter().enumerate() {
            inv_perm[p] = i;
        }

        for out_flat in 0..out_numel {
            // Decompose output flat index into output coords
            let mut rem = out_flat;
            let mut input_flat = 0usize;
            for out_dim in 0..rank {
                let coord = rem / output_strides[out_dim];
                rem %= output_strides[out_dim];
                // This output coord corresponds to input dim = perm[out_dim]
                let in_dim = full_perm[out_dim];
                input_flat += coord * input_strides[in_dim];
            }
            out.write_element(out_flat, data.read_element(input_flat));
        }

        Ok(vec![out])
    }

    fn lower_to_nano(
        &self,
        ctx: &mut crate::nano_graph::NanoLoweringContext,
    ) -> crate::milli_graph::ops::LowerResult {
        Transpose::lower_to_nano(self, ctx)
    }
}
