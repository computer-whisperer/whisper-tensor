use rand::Rng;
use crate::graph::{GlobalId, Node};
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
    pub fn lower_to_nano<'p, P: crate::pool::Pool + 'p>(
        &self,
        ctx: &mut crate::nano_graph::NanoLoweringContext<'_, 'p, P>,
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

        // Zero-cost transpose: reuse the input's atoms with permuted strides.
        //
        // The input atoms are laid out in row-major order with `in_strides`.
        // After transposing with perm, output dim j corresponds to input dim perm[j].
        // So the output's strides (into the SAME flat atom buffer) are:
        //   output_stride[j] = input_stride[perm[j]]
        //
        // This means downstream ops will decompose their flat index using
        // the output strides and arrive at the correct input atom.
        // Permute the layout (which includes strides) according to the transpose perm.
        let mut transposed_layout: Vec<DimKind> = vec![DimKind::Known { size: 0, stride: 0 }; full_perm.len()];
        for (out_dim, &in_dim) in full_perm.iter().enumerate() {
            transposed_layout[out_dim] = in_map.dims[in_dim].clone();
        }

        let out_dt = NanoLoweringContext::ndt(out_info);

        if !in_map.segments.is_empty() {
            // Segmented input (from Concat): propagate segments with permuted strides.
            let permuted_segments: Vec<crate::nano_graph::lower::ConcatSegment> = in_map
                .segments
                .iter()
                .map(|seg| {
                    let mut new_strides = vec![0u64; full_perm.len()];
                    for (out_dim, &in_dim) in full_perm.iter().enumerate() {
                        new_strides[out_dim] = seg.known_strides[in_dim];
                    }
                    crate::nano_graph::lower::ConcatSegment {
                        concat_dim: full_perm
                            .iter()
                            .position(|&d| d == seg.concat_dim)
                            .expect("concat_dim must appear in transpose permutation"),
                        start: seg.start,
                        size: seg.size,
                        base_id: seg.base_id,
                        known_strides: new_strides,
                    }
                })
                .collect();
            ctx.tensor_map.insert(
                out_id,
                TensorAtomMap::segmented(
                    in_map.count,
                    out_dt,
                    transposed_layout,
                    permuted_segments,
                ),
            );
        } else {
            ctx.tensor_map.insert(
                out_id,
                TensorAtomMap::simple(
                    in_map.base_id,
                    in_map.count,
                    out_dt,
                    transposed_layout,
                ),
            );
        }
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
    fn infer<'a, 'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'a, 'p, P>>,
        rng: &mut impl Rng,
        pool: &'p P,
    ) -> Result<Vec<(GlobalId, crate::tensor_info::TensorInfo<'a, 'p, P>)>, MilliOpGraphError>
    where
        'p: 'a,
    {
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
                rng,
            ))
        } else {
            // At minimum: same rank, same dtype
            let rank = input_info.rank();
            TensorInfo::new_from_first_element_and_rank(first_elem, rank, rng)
        };

        // If input is concrete, try constant fold via nano+pool_eval path.
        let numel = input_info.as_concrete().map(|c| c.numel()).unwrap_or(0);
        let t0 = std::time::Instant::now();
        let fold_result = super::constant_fold(
            self,
            known_inputs,
            &[(self.output, out_info.clone_with_pool(pool))],
            pool,
        );
        let dt = t0.elapsed();
        if dt.as_millis() > 10 {
            eprintln!(
                "    [transpose infer] constant_fold numel={numel}: {:.0}ms",
                dt.as_secs_f64() * 1e3
            );
        }
        if let Some(results) = fold_result {
            return Ok(results);
        }

        Ok(vec![(self.output, out_info)])
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

        let out_layout = TensorLayout::<DynRank>::row_major(output_shape.clone(), dtype);
        let in_layout = data.layout().clone();

        let out = NumericTensor::<DynRank, P2>::from_fn(output_shape, dtype, pool, |out_flat| {
            let out_coords = out_layout.flat_to_coords(out_flat);
            // Output dim d has coord for input dim perm[d]
            let mut in_coords = vec![0usize; rank];
            for (out_dim, &coord) in out_coords.iter().enumerate() {
                in_coords[full_perm[out_dim]] = coord;
            }
            data.read_element(in_layout.coords_to_flat(&in_coords))
        })
        .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;

        Ok(vec![out])
    }

    fn lower_to_nano<'p, P: crate::pool::Pool + 'p>(
        &self,
        ctx: &mut crate::nano_graph::NanoLoweringContext<'_, 'p, P>,
    ) -> crate::milli_graph::ops::LowerResult {
        Transpose::lower_to_nano(self, ctx)
    }
}
