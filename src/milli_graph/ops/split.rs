use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp, MilliOpTensorIDOrLiteral};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Split {
    global_id: GlobalId,
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
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
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
                .eval(&resolved, backend)?
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
