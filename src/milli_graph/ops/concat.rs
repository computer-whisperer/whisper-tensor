use crate::pool::Pool;
use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp, MilliOpTensorIDOrLiteral};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::nano_graph::lower::{ConcatSegment, DimKind, TensorAtomMap};
use crate::nano_graph::pattern::AtomId;
use crate::migration::numeric_tensor::NumericTensor;
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

    pub fn lower_to_nano(&self, ctx: &mut crate::nano_graph::NanoLoweringContext) -> crate::milli_graph::ops::LowerResult {
        let all_infos = ctx.all_infos;
        let axis_raw = self.axis();
        let out_id = Node::outputs(self).next().unwrap();
        let input_ids = self.concat_inputs();

        let Some(out_info) = all_infos.get(&out_id) else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };

        let Some((out_layout, out_known_dims, out_sym_dims, out_count)) =
            ctx.classify_dims(out_info)
        else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };
        let out_count = out_count.max(1);

        // Normalize axis.
        let rank = out_layout.len();
        let axis = if axis_raw < 0 {
            (axis_raw + rank as i64) as usize
        } else {
            axis_raw as usize
        };

        // Concat axis must be a known dim.
        if axis >= rank || !matches!(out_layout[axis], DimKind::Known(_)) {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        }

        // Known-dim index of the concat axis.
        let concat_known_idx = out_layout[..=axis]
            .iter()
            .filter(|d| matches!(d, DimKind::Known(_)))
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
            if inp_known.len() != out_known_dims.len() {
                return crate::milli_graph::ops::LowerResult::Unsupported;
            }
            concat_dim_sizes.push(inp_known[concat_known_idx]);
            input_maps.push(inp_map);
        }

        // Zero-cost concat: check if all inputs have row-major strides and
        // are laid out contiguously along the concat axis in atom space.
        // If so, the output is just a wider view of the same atoms.
        let ref_strides = &input_maps[0].known_strides;
        let concat_stride = ref_strides[concat_known_idx];
        let inp0_known: Vec<u64> = input_maps[0]
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
        let inp0_rowmajor = TensorAtomMap::compute_strides(&inp0_known);

        if *ref_strides == inp0_rowmajor {
            // Inputs have row-major strides. Check contiguity.
            let mut contiguous = true;
            let mut expected_base = input_maps[0].base_id;
            for (i, inp_map) in input_maps.iter().enumerate() {
                if inp_map.known_strides != *ref_strides || inp_map.base_id != expected_base {
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
                        out_layout,
                        TensorAtomMap::compute_strides(&out_known_dims),
                        out_sym_dims,
                    ),
                );
            return crate::milli_graph::ops::LowerResult::Lowered;
            }
        }

        // Non-contiguous concat: zero-cost segmented view.
        // Each input becomes a segment with its own base_id and strides.
        let mut segments = Vec::with_capacity(input_maps.len());
        let mut cum = 0u64;
        for (i, inp_map) in input_maps.iter().enumerate() {
            segments.push(ConcatSegment {
                concat_dim: concat_known_idx,
                start: cum,
                size: concat_dim_sizes[i],
                base_id: inp_map.base_id,
                known_strides: inp_map.known_strides.clone(),
            });
            cum += concat_dim_sizes[i];
        }

        ctx.tensor_map.insert(
            out_id,
            TensorAtomMap::segmented(
                out_count,
                crate::nano_graph::NanoLoweringContext::ndt(out_info),
                out_layout,
                out_sym_dims,
                segments,
            ),
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
    fn infer<'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'p, P>>,
        symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        pool: &'p P,
    ) -> Result<
        Vec<(GlobalId, crate::tensor_info::TensorInfo<'p, P>)>,
        MilliOpGraphError,
    > {
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
                        None => out_dims.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver))),
                    }
                } else {
                    let known_dim = input_infos
                        .iter()
                        .filter_map(|info| info.dim_if_known(d))
                        .next();
                    match known_dim {
                        Some(v) => out_dims.push(ScalarInfoTyped::Numeric(v)),
                        None => out_dims.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver))),
                    }
                }
            }
            TensorInfo::from_dtype_and_shape_scalars(out_dtype, &out_dims)
        } else {
            TensorInfo::from_dtype_and_shape_scalars(out_dtype, &[])
        };

        // If all inputs are concrete, try constant fold via nano+pool_eval path.
        if let Some(results) = super::constant_fold(self, known_inputs, &[(self.output, output_hint)], pool) {
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
                            symbolic_resolver,
                        ))),
                    }
                } else {
                    // Non-concat axis: take from any input that has a known dim.
                    let known_dim = input_infos
                        .iter()
                        .filter_map(|info| info.dim_if_known(d))
                        .next();
                    match known_dim {
                        Some(v) => out_dims.push(ScalarInfoTyped::Numeric(v)),
                        None => out_dims.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(
                            symbolic_resolver,
                        ))),
                    }
                }
            }

            let out_info = TensorInfo::from_dtype_and_shape_scalars(out_dtype, &out_dims);
            Ok(vec![((self.output, out_info))])
        } else {
            // No input has known rank — fall back to Minimal.
            let first_elem = crate::scalar_info::ScalarInfo::Symbolic(
                crate::symbolic_scalar::SymbolicScalar::new(out_dtype, symbolic_resolver),
            );
            let out_rank = input_infos[0].rank();
            let out_info = TensorInfo::new_from_first_element_and_rank(
                first_elem,
                out_rank,
                symbolic_resolver,
            );
            Ok(vec![((self.output, out_info))])
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
        let axis_idx = super::Constant::push_new(
            graph,
            crate::backends::ndarray_backend::NDArrayNumericTensor::<DynRank>::from_vec_shape(
                vec![self.axis],
                &vec![1],
            )
            .unwrap(),
            rng,
        );
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
                Some(MilliOpTensorIDOrLiteral::TensorID(split_sizes)),
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

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        _config: &super::MilliEvalConfig,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        let mut resolved_inputs = vec![];
        for input in &self.inputs {
            resolved_inputs.push(&inputs[input]);
        }
        let axis = if self.axis < 0 {
            resolved_inputs[0].shape().len() as i64 + self.axis
        } else {
            self.axis
        } as usize;
        let out = NumericTensor::<DynRank>::concat(resolved_inputs.as_slice(), axis, backend)?;
        Ok(Box::new([(self.output, out)].into_iter()))
    }

    fn eval_new<'p, P2: crate::pool::Pool + 'p>(
        &self,
        inputs: &[crate::numeric_tensor::NumericTensorView<'_, crate::tensor_rank::DynRank>],
        pool: &'p P2,
    ) -> Result<Vec<crate::numeric_tensor::NumericTensor<'p, crate::tensor_rank::DynRank, P2>>, crate::nano_graph::pool_eval::PoolEvalError> {
        use crate::numeric_tensor::{NumericTensor, TensorLayout};
        use crate::tensor_rank::DynRank;

        if inputs.is_empty() {
            return Err(crate::nano_graph::pool_eval::PoolEvalError::Unsupported("Concat: no inputs".to_string()));
        }

        let dtype = inputs[0].dtype();
        let rank = inputs[0].shape().len();
        let axis = if self.axis < 0 { (self.axis + rank as i64) as usize } else { self.axis as usize };

        // Compute output shape: same as first input except concat axis is sum
        let mut output_shape: Vec<u64> = inputs[0].shape().clone();
        for inp in inputs.iter().skip(1) {
            output_shape[axis] += inp.shape()[axis];
        }

        let layout = TensorLayout::<DynRank>::row_major(output_shape.clone(), dtype);
        let buf = pool.allocate(layout.buffer_size_bytes())
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
                    let out_coord = if d == axis { coord + axis_offset } else { coord };
                    out_flat += out_coord * out_strides[d];
                }
                out.write_element(out_flat, inp.read_element(inp_flat));
            }
            axis_offset += inp_shape[axis] as usize;
        }

        Ok(vec![out])
    }

    fn lower_to_nano(&self, ctx: &mut crate::nano_graph::NanoLoweringContext) -> crate::milli_graph::ops::LowerResult {
        Concat::lower_to_nano(self, ctx)
    }
}
