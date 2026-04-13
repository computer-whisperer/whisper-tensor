use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::nano_graph::lower::{ConcatSegment, DimKind, TensorAtomMap};
use crate::nano_graph::pattern::AtomId;
use crate::pool::Pool;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Concat {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    inputs: Vec<GlobalId>,
    axis: i64,
}

impl Concat {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        inputs: Vec<GlobalId>,
        axis: i64,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new_with_label(graph, inputs, axis, None, rng)
    }

    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        inputs: Vec<GlobalId>,
        axis: i64,
        label: Option<String>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            label,
            output,
            inputs,
            axis,
        };
        graph.push_op(AnyMilliOp::Concat(node));
        output
    }
}

impl Concat {
    pub(crate) fn axis(&self) -> i64 {
        self.axis
    }

    pub(crate) fn concat_inputs(&self) -> &[GlobalId] {
        &self.inputs
    }

    pub fn lower_to_nano<'p, P: crate::pool::Pool + 'p>(
        &self,
        ctx: &mut crate::nano_graph::NanoLoweringContext<'_, 'p, P>,
    ) -> crate::milli_graph::ops::LowerResult {
        let all_infos = ctx.all_infos;
        let axis_raw = self.axis();
        let out_id = Node::outputs(self).next().unwrap();
        let input_ids = self.concat_inputs();

        let Some(out_info) = all_infos.get(&out_id) else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };

        let Some(out_dims) = ctx.classify_dims(out_info) else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };
        let out_count: u64 = out_dims.iter().filter_map(|d| match d { DimKind::Known { size, .. } => Some(*size), _ => None }).product::<u64>().max(1);
        let out_known_dims: Vec<u64> = out_dims.iter().filter_map(|d| match d { DimKind::Known { size, .. } => Some(*size), _ => None }).collect();

        // Normalize axis.
        let rank = out_dims.len();
        let axis = if axis_raw < 0 {
            (axis_raw + rank as i64) as usize
        } else {
            axis_raw as usize
        };

        // Concat axis must be a known dim.
        if axis >= rank || !matches!(out_dims[axis], DimKind::Known { .. }) {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        }

        // Known-dim index of the concat axis.
        let concat_known_idx = out_dims[..=axis]
            .iter()
            .filter(|d| matches!(d, DimKind::Known { .. }))
            .count()
            - 1;

        // Gather input maps and their concat-axis sizes.
        let mut input_maps = Vec::with_capacity(input_ids.len());
        let mut concat_dim_sizes = Vec::with_capacity(input_ids.len());
        for &inp_id in input_ids {
            let Some(inp_map) = ctx.tensor_map.get(&inp_id).cloned() else {
                return crate::milli_graph::ops::LowerResult::Unsupported;
            };
            let inp_known: Vec<u64> = inp_map
                .dims
                .iter()
                .filter_map(|d| {
                    if let DimKind::Known { size: s, .. } = d {
                        Some(*s)
                    } else {
                        None
                    }
                })
                .collect();
            if inp_known.len() != out_known_dims.len() {
                return crate::milli_graph::ops::LowerResult::Unsupported;
            }
            concat_dim_sizes.push(inp_known[concat_known_idx]);
            input_maps.push(inp_map);
        }

        // Zero-cost concat: check if all inputs have row-major strides and
        // are laid out contiguously along the concat axis in atom space.
        // If so, the output is just a wider view of the same atoms.
        let ref_strides = input_maps[0].known_strides();
        let concat_stride = ref_strides[concat_known_idx];
        let inp0_known: Vec<u64> = input_maps[0]
            .dims
            .iter()
            .filter_map(|d| {
                if let DimKind::Known { size: s, .. } = d {
                    Some(*s)
                } else {
                    None
                }
            })
            .collect();
        let inp0_rowmajor = TensorAtomMap::compute_strides(&inp0_known);

        if *ref_strides == inp0_rowmajor {
            // Inputs have row-major strides. Check contiguity.
            let mut contiguous = true;
            let mut expected_base = input_maps[0].base_id;
            for (i, inp_map) in input_maps.iter().enumerate() {
                if inp_map.known_strides() != ref_strides || inp_map.base_id != expected_base {
                    contiguous = false;
                    break;
                }
                expected_base = AtomId(expected_base.0 + concat_dim_sizes[i] * concat_stride);
            }
            if contiguous {
                ctx.tensor_map.insert(
                    out_id,
                    TensorAtomMap::simple(
                        input_maps[0].base_id,
                        out_count,
                        crate::nano_graph::NanoLoweringContext::ndt(out_info),
                        out_dims,
                    ),
                );
                return crate::milli_graph::ops::LowerResult::Lowered;
            }
        }

        // Non-contiguous concat: build segments.
        // If any input is itself segmented (from a previous concat), expand
        // its inner segments into the outer segment list so that
        // atom_id_for_element can resolve all atoms in a single level.
        let out_dt = crate::nano_graph::NanoLoweringContext::ndt(out_info);
        let mut segments = Vec::new();
        let mut cum = 0u64;
        for (i, inp_map) in input_maps.iter().enumerate() {
            if !inp_map.segments.is_empty() {
                // Expand inner segments, shifting their concat-dim starts.
                // Only valid when inner and outer concat are on the same axis.
                if inp_map.segments[0].concat_dim != concat_known_idx {
                    return crate::milli_graph::ops::LowerResult::Unsupported;
                }
                for inner_seg in &inp_map.segments {
                    segments.push(ConcatSegment {
                        concat_dim: concat_known_idx,
                        start: cum + inner_seg.start,
                        size: inner_seg.size,
                        base_id: inner_seg.base_id,
                        known_strides: inner_seg.known_strides.clone(),
                    });
                }
            } else {
                segments.push(ConcatSegment {
                    concat_dim: concat_known_idx,
                    start: cum,
                    size: concat_dim_sizes[i],
                    base_id: inp_map.base_id,
                    known_strides: inp_map.known_strides(),
                });
            }
            cum += concat_dim_sizes[i];
        }

        ctx.tensor_map.insert(
            out_id,
            TensorAtomMap::segmented(out_count, out_dt, out_dims, segments),
        );
        crate::milli_graph::ops::LowerResult::Lowered
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        for id in &mut self.inputs {
            super::remap(id, map);
        }
    }
}

impl crate::graph::Node for Concat {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Concat".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(self.inputs.clone().into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
}

impl MilliOp for Concat {
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

        // Collect input infos.
        let mut input_infos = Vec::new();
        for id in &self.inputs {
            let info = known_inputs
                .get(id)
                .ok_or(MilliOpGraphError::UnableToInfer)?;
            input_infos.push(info);
        }

        // Build output hint for constant folding.
        let out_dtype = input_infos[0].dtype();
        let rank = input_infos
            .iter()
            .filter_map(|info| info.rank_if_known())
            .next();
        let output_hint = if let Some(rank) = rank {
            use crate::scalar_info::ScalarInfoTyped;
            use crate::symbolic_scalar::SymbolicScalarTyped;

            let axis = if self.axis < 0 {
                (self.axis + rank as i64) as usize
            } else {
                self.axis as usize
            };
            let mut out_dims: Vec<ScalarInfoTyped<u64>> = Vec::with_capacity(rank);
            for d in 0..rank {
                if d == axis {
                    let mut total: Option<u64> = Some(0);
                    for info in &input_infos {
                        if let Some(dim_val) = info.dim_if_known(d) {
                            total = total.map(|t| t + dim_val);
                        } else {
                            total = None;
                            break;
                        }
                    }
                    match total {
                        Some(v) => out_dims.push(ScalarInfoTyped::Numeric(v)),
                        None => out_dims.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(
                            rng,
                        ))),
                    }
                } else {
                    // Non-concat dim: propagate from inputs. Prefer concrete,
                    // then reuse an existing symbolic scalar so that the same
                    // GraphConstant is shared between input and output TAMIs.
                    if let Some(v) = input_infos.iter().filter_map(|info| info.dim_if_known(d)).next() {
                        out_dims.push(ScalarInfoTyped::Numeric(v));
                    } else if let Some(sym) = input_infos.iter().filter_map(|info| {
                        match info.dim_scalar(d)? {
                            ScalarInfoTyped::Symbolic(s) => Some(s),
                            _ => None,
                        }
                    }).next() {
                        out_dims.push(ScalarInfoTyped::Symbolic(sym));
                    } else {
                        out_dims.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(rng)));
                    }
                }
            }
            TensorInfo::from_dtype_and_shape_scalars(out_dtype, &out_dims)
        } else {
            TensorInfo::from_dtype_and_shape_scalars(out_dtype, &[])
        };

        // If all inputs are concrete, try constant fold via nano+pool_eval path.
        if let Some(results) =
            super::constant_fold(self, known_inputs, &[(self.output, output_hint)], pool)
        {
            return Ok(results);
        }

        // Find the rank from any input that has a known rank.
        let rank = input_infos
            .iter()
            .filter_map(|info| info.rank_if_known())
            .next();

        if let Some(rank) = rank {
            use crate::scalar_info::ScalarInfoTyped;
            use crate::symbolic_scalar::SymbolicScalarTyped;

            let axis = if self.axis < 0 {
                (self.axis + rank as i64) as usize
            } else {
                self.axis as usize
            };

            // Build output dims: take from first input that has shape, sum along concat axis.
            let mut out_dims: Vec<ScalarInfoTyped<u64>> = Vec::with_capacity(rank);
            for d in 0..rank {
                if d == axis {
                    // Sum the concat axis dims across all inputs.
                    let mut total: Option<u64> = Some(0);
                    for info in &input_infos {
                        if let Some(dim_val) = info.dim_if_known(d) {
                            total = total.map(|t| t + dim_val);
                        } else {
                            total = None;
                            break;
                        }
                    }
                    match total {
                        Some(v) => out_dims.push(ScalarInfoTyped::Numeric(v)),
                        None => out_dims.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(
                            rng,
                        ))),
                    }
                } else {
                    // Non-concat axis: propagate from inputs.
                    if let Some(v) = input_infos.iter().filter_map(|info| info.dim_if_known(d)).next() {
                        out_dims.push(ScalarInfoTyped::Numeric(v));
                    } else if let Some(sym) = input_infos.iter().filter_map(|info| {
                        match info.dim_scalar(d)? {
                            ScalarInfoTyped::Symbolic(s) => Some(s),
                            _ => None,
                        }
                    }).next() {
                        out_dims.push(ScalarInfoTyped::Symbolic(sym));
                    } else {
                        out_dims.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(rng)));
                    }
                }
            }

            let out_info = TensorInfo::from_dtype_and_shape_scalars(out_dtype, &out_dims);
            Ok(vec![(self.output, out_info)])
        } else {
            // No input has known rank — fall back to Minimal.
            let first_elem = crate::scalar_info::ScalarInfo::Symbolic(
                crate::symbolic_scalar::SymbolicScalar::new(out_dtype, rng),
            );
            let out_rank = input_infos[0].rank();
            let out_info = TensorInfo::new_from_first_element_and_rank(
                first_elem,
                out_rank,
                rng,
            );
            Ok(vec![(self.output, out_info)])
        }
    }

    fn backward(
        &self,
        output_grads: &HashMap<GlobalId, GlobalId>,
        graph: &mut MilliOpGraph,
        rng: &mut impl Rng,
    ) -> Option<HashMap<GlobalId, GlobalId>> {
        let grad_output = *output_grads.get(&self.output)?;
        let n = self.inputs.len();

        // Build split sizes by gathering each input's dim along the concat axis
        let axis_idx = super::Constant::from_vec(graph, vec![self.axis], rng);
        let mut size_tensors = Vec::new();
        for &input_id in &self.inputs {
            let shape = super::Shape::push_new(graph, input_id, rng);
            let size = super::Gather::push_new(graph, shape, axis_idx, 0, rng);
            size_tensors.push(size);
        }
        let split_sizes = super::Concat::push_new(graph, size_tensors, 0, rng);

        // Split the gradient along the same axis
        let mut result = HashMap::new();
        for (i, &input_id) in self.inputs.iter().enumerate() {
            let grad_i = super::Split::push_new(
                graph,
                grad_output,
                Some(split_sizes),
                self.axis,
                Some(n),
                i,
                rng,
            );
            result
                .entry(input_id)
                .and_modify(|existing: &mut GlobalId| {
                    *existing = super::SimpleBinary::add(graph, *existing, grad_i, rng);
                })
                .or_insert(grad_i);
        }
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

        if inputs.is_empty() {
            return Err(crate::nano_graph::pool_eval::PoolEvalError::Unsupported(
                "Concat: no inputs".to_string(),
            ));
        }

        let dtype = inputs[0].dtype();
        let rank = inputs[0].shape().len();
        let axis = if self.axis < 0 {
            (self.axis + rank as i64) as usize
        } else {
            self.axis as usize
        };

        // Compute output shape: same as first input except concat axis is sum
        let mut output_shape: Vec<u64> = inputs[0].shape().clone();
        for inp in inputs.iter().skip(1) {
            output_shape[axis] += inp.shape()[axis];
        }

        let layout = TensorLayout::<DynRank>::row_major(output_shape.clone(), dtype);
        let buf = pool
            .allocate(layout.buffer_size_bytes())
            .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
        let mut out = NumericTensor::from_parts(buf, layout);

        // Compute output strides
        let mut out_strides = vec![1usize; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            out_strides[i] = out_strides[i + 1] * output_shape[i + 1] as usize;
        }

        // Copy each input into the right slice of the output
        let mut axis_offset = 0usize;
        for inp in inputs.iter() {
            let inp_shape = inp.shape();
            let inp_numel = inp.numel();
            let mut inp_strides = vec![1usize; rank];
            for i in (0..rank.saturating_sub(1)).rev() {
                inp_strides[i] = inp_strides[i + 1] * inp_shape[i + 1] as usize;
            }

            for inp_flat in 0..inp_numel {
                let mut rem = inp_flat;
                let mut out_flat = 0usize;
                for d in 0..rank {
                    let coord = rem / inp_strides[d];
                    rem %= inp_strides[d];
                    let out_coord = if d == axis {
                        coord + axis_offset
                    } else {
                        coord
                    };
                    out_flat += out_coord * out_strides[d];
                }
                out.write_element(out_flat, inp.read_element(inp_flat));
            }
            axis_offset += inp_shape[axis] as usize;
        }

        Ok(vec![out])
    }

    fn lower_to_nano<'p, P: crate::pool::Pool + 'p>(
        &self,
        ctx: &mut crate::nano_graph::NanoLoweringContext<'_, 'p, P>,
    ) -> crate::milli_graph::ops::LowerResult {
        Concat::lower_to_nano(self, ctx)
    }
}
