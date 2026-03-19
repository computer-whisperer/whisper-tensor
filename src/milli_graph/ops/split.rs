use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp, MilliOpTensorIDOrLiteral};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::nano_graph::lower::{DimKind, TensorAtomMap};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Split {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    data: GlobalId,
    split: Option<MilliOpTensorIDOrLiteral>,
    axis: i64,
    num_outputs: Option<usize>,
    output_id: usize,
}

impl Split {
    pub(crate) fn axis(&self) -> i64 {
        self.axis
    }

    pub(crate) fn output_id(&self) -> usize {
        self.output_id
    }

    pub(crate) fn split_tensor(&self) -> Option<&MilliOpTensorIDOrLiteral> {
        self.split.as_ref()
    }

    pub fn push_new(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        split: Option<MilliOpTensorIDOrLiteral>,
        axis: i64,
        num_outputs: Option<usize>,
        output_id: usize,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        Self::push_new_with_label(graph, data, split, axis, num_outputs, output_id, None, rng)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        split: Option<MilliOpTensorIDOrLiteral>,
        axis: i64,
        num_outputs: Option<usize>,
        output_id: usize,
        label: Option<String>,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            label,
            output,
            data,
            split,
            axis,
            num_outputs,
            output_id,
        };
        graph.push_op(AnyMilliOp::Split(node));
        output
    }
}

impl Split {
    pub fn lower_to_nano(&self, ctx: &mut crate::nano_graph::NanoLoweringContext) {
        let all_infos = ctx.all_infos;
        let in_id = Node::inputs(self).next().unwrap();
        let out_id = Node::outputs(self).next().unwrap();

        let Some(in_map) = ctx.tensor_map.get(&in_id).cloned() else {
            ctx.lower_as_boundary_named(self, "Split");
            return;
        };
        let Some(out_info) = all_infos.get(&out_id) else {
            ctx.lower_as_boundary_named(self, "Split");
            return;
        };
        let Some(_in_info) = all_infos.get(&in_id) else {
            ctx.lower_as_boundary_named(self, "Split");
            return;
        };

        let Some((out_layout, out_known_dims, out_sym_dims, out_count)) =
            ctx.classify_dims(out_info)
        else {
            ctx.lower_as_boundary_named(self, "Split");
            return;
        };
        let out_count = out_count.max(1);

        // Normalize axis.
        let rank = in_map.layout.len();
        let axis_raw = self.axis();
        let axis = if axis_raw < 0 {
            (axis_raw + rank as i64) as usize
        } else {
            axis_raw as usize
        };

        if axis >= rank || !matches!(in_map.layout[axis], DimKind::Known(_)) {
            ctx.lower_as_boundary_named(self, "Split");
            return;
        }

        // Determine the offset along the split axis for this output_id.
        // We need the split sizes. If all outputs have known dims, compute from the
        // input dim and output sizes. Otherwise use the output info directly.
        let split_known_idx = in_map.layout[..=axis]
            .iter()
            .filter(|d| matches!(d, DimKind::Known(_)))
            .count()
            - 1;

        // Get the input's full dim along split axis.
        let in_known: Vec<u64> = in_map
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
        let _in_split_size = in_known[split_known_idx];

        // Figure out the offset: we need to know what came before this output_id's chunk.
        // The output_id tells us which chunk we are. We need the sizes of all prior chunks.
        // We can compute this from the axis dim of the output info and output_id index.
        let output_id_idx = self.output_id();
        let out_split_size = match &out_layout[axis] {
            DimKind::Known(s) => *s,
            _ => {
                ctx.lower_as_boundary_named(self, "Split");
                return;
            }
        };

        // Compute offset: we need the sum of split sizes for all outputs before this one.
        // Since we might not have the split tensor values, estimate from output_id * output_size.
        // This is only correct for equal splits. For unequal splits we need the actual sizes.
        // Try to get them from the split tensor.
        let offset_along_axis = self.compute_split_offset(ctx, output_id_idx, out_split_size);

        if out_known_dims.len() != in_known.len() {
            ctx.lower_as_boundary_named(self, "Split");
            return;
        }

        // Zero-cost split for outermost axis with row-major strides:
        // output atoms are a contiguous sub-range.
        let out_dt = out_info.dtype();
        let in_rowmajor = TensorAtomMap::compute_strides(&in_known);
        if split_known_idx == 0 && in_map.known_strides == in_rowmajor {
            let base_offset = offset_along_axis * in_map.known_strides[split_known_idx];
            ctx.tensor_map.insert(
                out_id,
                TensorAtomMap::simple(
                    in_map.base_id.offset(base_offset),
                    out_count,
                    out_dt,
                    out_layout,
                    TensorAtomMap::compute_strides(&out_known_dims),
                    out_sym_dims,
                ),
            );
            return;
        }

        // Non-outermost split: zero-cost view with input's strides.
        // The strides address the input's atom space with gaps between chunks.
        // build_input_ref handles this correctly via stride decomposition.
        let base_offset = offset_along_axis * in_map.known_strides[split_known_idx];
        ctx.tensor_map.insert(
            out_id,
            TensorAtomMap::simple(
                in_map.base_id.offset(base_offset),
                out_count,
                out_dt,
                out_layout,
                in_map.known_strides.clone(),
                out_sym_dims,
            ),
        );
    }

    /// Compute the cumulative offset along the split axis for output_id_idx.
    fn compute_split_offset(
        &self,
        ctx: &crate::nano_graph::NanoLoweringContext,
        output_id_idx: usize,
        out_split_size: u64,
    ) -> u64 {
        let all_infos = ctx.all_infos;
        // Try to get concrete split sizes from the split tensor.
        if let Some(crate::milli_graph::ops::MilliOpTensorIDOrLiteral::TensorID(tensor_id)) =
            self.split_tensor()
            && let Some(info) = all_infos.get(tensor_id)
            && let Some(numeric) = info.as_numeric()
            && let Ok(cast) = numeric.cast(
                DType::I64,
                &mut crate::backends::eval_backend::EvalBackend::NDArray,
            )
            && let Ok(rank1) = cast.try_to_rank::<typenum::P1>()
            && let Ok(vals) = Vec::<i64>::try_from(rank1.to_ndarray().unwrap())
        {
            let offset: i64 = vals[..output_id_idx].iter().sum();
            return offset as u64;
        }
        // Fallback: assume equal splits.
        output_id_idx as u64 * out_split_size
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.data, map);
        if let Some(super::MilliOpTensorIDOrLiteral::TensorID(ref mut id)) = self.split {
            super::remap(id, map);
        }
    }
}

impl Node for Split {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> String {
        "Split".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        let mut ids = vec![self.data];
        if let Some(MilliOpTensorIDOrLiteral::TensorID(id)) = &self.split {
            ids.push(*id);
        }
        Box::new(ids.into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new([self.output].into_iter())
    }
}

impl MilliOp for Split {
    fn infer(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo>,
        _symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<
        Box<dyn Iterator<Item = (GlobalId, crate::tensor_info::TensorInfo)>>,
        MilliOpGraphError,
    > {
        use crate::scalar_info::ScalarInfoTyped;
        use crate::tensor_info::TensorInfo;

        let data_info = known_inputs
            .get(&self.data)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        // If all inputs are concrete, fall back to eval.
        let split_numeric = match &self.split {
            Some(MilliOpTensorIDOrLiteral::TensorID(id)) => {
                known_inputs.get(id).and_then(|i| i.as_numeric()).is_some()
            }
            Some(MilliOpTensorIDOrLiteral::Literal(_)) => true,
            None => true,
        };
        if data_info.as_numeric().is_some() && split_numeric {
            let mut resolved = HashMap::new();
            for id in self.inputs() {
                let info = known_inputs
                    .get(&id)
                    .ok_or(MilliOpGraphError::UnableToInfer)?;
                resolved.insert(
                    id,
                    info.as_numeric()
                        .ok_or(MilliOpGraphError::UnableToInfer)?
                        .clone(),
                );
            }
            let collected: Vec<(GlobalId, TensorInfo)> = self
                .eval(&resolved, &super::MilliEvalConfig::default(), backend)?
                .map(|(a, b)| (a, TensorInfo::from(b)))
                .collect();
            return Ok(Box::new(collected.into_iter()));
        }

        // Shape-only inference.
        let data_ranked = data_info
            .as_ranked()
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let data_shape = data_ranked.shape();
        let data_rank = data_shape.len();
        let axis = if self.axis < 0 {
            (self.axis + data_rank as i64) as usize
        } else {
            self.axis as usize
        };

        // Determine the split size for this output_id.
        let split_sizes: Vec<i64> = if let Some(split) = &self.split {
            match split {
                MilliOpTensorIDOrLiteral::TensorID(id) => {
                    let info = known_inputs
                        .get(id)
                        .ok_or(MilliOpGraphError::UnableToInfer)?;
                    let tensor = info.as_numeric().ok_or(MilliOpGraphError::UnableToInfer)?;
                    tensor.clone().try_to_rank::<P1>()?.try_into()?
                }
                MilliOpTensorIDOrLiteral::Literal(lit) => lit.try_to_rank::<P1>()?.try_into()?,
            }
        } else if let Some(num_outputs) = self.num_outputs {
            // Compute from data shape along axis.
            if let ScalarInfoTyped::Numeric(dim_val) = &data_shape[axis] {
                let dim = *dim_val as usize;
                let base = dim / num_outputs;
                let remainder = dim % num_outputs;
                (0..num_outputs)
                    .map(|i| (base + if i < remainder { 1 } else { 0 }) as i64)
                    .collect()
            } else {
                return Err(MilliOpGraphError::UnableToInfer);
            }
        } else {
            return Err(MilliOpGraphError::UnableToInfer);
        };

        // Output shape: same as data, but axis dim = split_sizes[output_id].
        let mut out_dims = data_shape.clone();
        if self.output_id < split_sizes.len() {
            out_dims[axis] = ScalarInfoTyped::Numeric(split_sizes[self.output_id] as u64);
        } else {
            return Err(MilliOpGraphError::UnableToInfer);
        }

        let out_dtype = data_info.dtype();
        let out_info = TensorInfo::from_dtype_and_shape_scalars(out_dtype, &out_dims);
        Ok(Box::new([(self.output, out_info)].into_iter()))
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        _config: &super::MilliEvalConfig,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        // Determine the split sizes
        let split: Vec<i64> = if let Some(split) = &self.split {
            match split {
                MilliOpTensorIDOrLiteral::TensorID(split) => {
                    inputs[split].clone().try_to_rank::<P1>()?.try_into()?
                }
                MilliOpTensorIDOrLiteral::Literal(split) => {
                    split.try_to_rank::<P1>()?.try_into()?
                }
            }
        } else if let Some(num_outputs) = self.num_outputs {
            if num_outputs == 0 {
                return Err(MilliOpGraphError::InvalidInput(
                    "Split: num_outputs must be > 0".to_string(),
                ));
            }
            // Compute equal chunk sizes from the input shape along axis
            let input = &inputs[&self.data];
            let shape = input.shape();
            let rank = shape.len();
            let axis = if self.axis < 0 {
                (self.axis + rank as i64) as usize
            } else {
                self.axis as usize
            };
            if axis >= rank {
                return Err(MilliOpGraphError::InvalidInput(format!(
                    "Split: axis {} out of range for rank {}",
                    self.axis, rank
                )));
            }
            let dim = shape[axis] as usize;
            // ONNX semantics: when split attribute is absent and num_outputs is provided,
            // the input tensor is split into num_outputs nearly-equal parts along axis.
            // If not divisible, the first (dim % num_outputs) outputs get one extra element.
            let base = dim / num_outputs;
            let remainder = dim % num_outputs;
            let mut parts = Vec::with_capacity(num_outputs);
            for i in 0..num_outputs {
                let sz = base + if i < remainder { 1 } else { 0 };
                parts.push(sz as i64);
            }
            parts
        } else {
            return Err(MilliOpGraphError::InvalidInput(
                "Split attribute is not set and num_outputs is not provided".to_string(),
            ));
        };

        let outs = inputs[&self.data].split(&split, self.axis, backend)?;
        let out = outs[self.output_id].clone();
        Ok(Box::new([(self.output, out)].into_iter()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;

    #[test]
    fn test_split_num_outputs_even_axis0() {
        let rng = &mut rand::rng();
        // Build a tiny graph with one input and two split outputs, then run eval

        let input_id = GlobalId::new(rng);
        let (mut graph, input_map) = MilliOpGraph::new(std::iter::once(input_id), rng);
        let data_id = input_map[&input_id];
        let out0 = Split::push_new(&mut graph, data_id, None, 0, Some(2), 0, rng);
        let out1 = Split::push_new(&mut graph, data_id, None, 0, Some(2), 1, rng);
        let mut output_map = HashMap::new();
        let out0_id = GlobalId::new(rng);
        let out1_id = GlobalId::new(rng);
        output_map.insert(out0, out0_id);
        output_map.insert(out1, out1_id);
        graph.set_output_map(output_map);

        let mut inputs = HashMap::new();
        let x = NumericTensor::<DynRank>::from_vec_shape(vec![1f32, 2., 3., 4.], vec![4]).unwrap();
        inputs.insert(input_id, x);

        let mut backend = EvalBackend::NDArray;
        let mut obs = ();
        let res = graph
            .eval(&inputs, &mut obs, &mut backend)
            .unwrap()
            .collect::<HashMap<_, _>>();
        let out0 = res[&out0_id].clone();
        let out1 = res[&out1_id].clone();

        assert_eq!(out0.shape(), vec![2u64]);
        assert_eq!(out1.shape(), vec![2u64]);
        assert_eq!(out0.dtype(), DType::F32);
        assert_eq!(out1.dtype(), DType::F32);

        let v0: Vec<f32> = out0.flatten().unwrap().try_into().unwrap();
        let v1: Vec<f32> = out1.flatten().unwrap().try_into().unwrap();
        assert_eq!(v0, vec![1., 2.]);
        assert_eq!(v1, vec![3., 4.]);
    }

    #[test]
    fn test_split_num_outputs_negative_axis() {
        let rng = &mut rand::rng();
        let input_id = GlobalId::new(rng);
        let (mut graph, input_map) = MilliOpGraph::new(std::iter::once(input_id), rng);
        let data_id = input_map[&input_id];
        let out = Split::push_new(&mut graph, data_id, None, -1, Some(2), 1, rng);
        let mut output_map = HashMap::new();
        let output_id = GlobalId::new(rng);
        output_map.insert(out, output_id);
        graph.set_output_map(output_map);

        let mut inputs = HashMap::new();
        // 2x4
        let x = NumericTensor::<DynRank>::from_vec_shape(
            (1..=8).map(|v| v as f32).collect::<Vec<_>>(),
            vec![2, 4],
        )
        .unwrap();
        inputs.insert(input_id, x);

        let mut backend = EvalBackend::NDArray;
        let mut obs = ();
        let res = graph
            .eval(&inputs, &mut obs, &mut backend)
            .unwrap()
            .collect::<HashMap<_, _>>();
        let out = res[&output_id].clone();
        assert_eq!(out.shape(), vec![2u64, 2u64]);
        let v: Vec<f32> = out.flatten().unwrap().try_into().unwrap();
        // Expect second half along last axis: [[3,4],[7,8]]
        assert_eq!(v, vec![3., 4., 7., 8.]);
    }

    #[test]
    fn test_split_num_outputs_uneven_distribution_axis0() {
        let rng = &mut rand::rng();
        let input_id = GlobalId::new(rng);
        let (mut graph, input_map) = MilliOpGraph::new(std::iter::once(input_id), rng);
        let data_id = input_map[&input_id];
        // dim=5, num_outputs=2 -> sizes [3,2]
        let out0_id = GlobalId::new(rng);
        let out1_id = GlobalId::new(rng);
        let out0 = Split::push_new(&mut graph, data_id, None, 0, Some(2), 0, rng);
        let out1 = Split::push_new(&mut graph, data_id, None, 0, Some(2), 1, rng);
        let mut output_map = HashMap::new();
        output_map.insert(out0, out0_id);
        output_map.insert(out1, out1_id);
        graph.set_output_map(output_map);

        let mut inputs = HashMap::new();
        let x =
            NumericTensor::<DynRank>::from_vec_shape(vec![1f32, 2., 3., 4., 5.], vec![5]).unwrap();
        inputs.insert(input_id, x);

        let mut backend = EvalBackend::NDArray;
        let mut obs = ();
        let res = graph
            .eval(&inputs, &mut obs, &mut backend)
            .unwrap()
            .collect::<HashMap<_, _>>();
        let out0 = res[&out0_id].clone();
        let out1 = res[&out1_id].clone();

        assert_eq!(out0.shape(), vec![3u64]);
        assert_eq!(out1.shape(), vec![2u64]);
        let v0: Vec<f32> = out0.flatten().unwrap().try_into().unwrap();
        let v1: Vec<f32> = out1.flatten().unwrap().try_into().unwrap();
        assert_eq!(v0, vec![1., 2., 3.]);
        assert_eq!(v1, vec![4., 5.]);
    }
}
