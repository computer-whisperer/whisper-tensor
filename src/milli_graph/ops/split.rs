use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp, MilliOpTensorIDOrLiteral};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;
use crate::graph::Node;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpSplit {
    output: MilliOpGraphTensorId,
    data: MilliOpGraphTensorId,
    split: Option<MilliOpTensorIDOrLiteral>,
    axis: i64,
    num_outputs: Option<usize>,
    output_id: usize,
}

impl MilliOpSplit {
    pub fn new<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        data: MilliOpGraphTensorId,
        split: Option<MilliOpTensorIDOrLiteral>,
        axis: i64,
        num_outputs: Option<usize>,
        output_id: usize,
    ) -> MilliOpGraphTensorId {
        let output = graph.get_new_tensor_id();
        let node = Self {
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

impl Node<MilliOpGraphTensorId> for MilliOpSplit {
    fn inputs(&self) -> impl Iterator<Item=MilliOpGraphTensorId> { vec![self.data].into_iter() }
    fn outputs(&self) -> impl Iterator<Item=MilliOpGraphTensorId> { vec![self.output].into_iter() }
}

impl MilliOp for MilliOpSplit {
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<impl Iterator<Item=(MilliOpGraphTensorId, NumericTensor<DynRank>)>, MilliOpGraphError> {
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
        Ok([(self.output, out)].into_iter())
    }

    fn get_name(&self) -> String {
        "Split".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;

    #[test]
    fn test_split_num_outputs_even_axis0() {
        // Build a tiny graph with one input and two split outputs, then run eval
        let (mut graph, input_map) = MilliOpGraph::new(&[MilliOpGraphTensorId { inner: 0 }]);
        let data_id = input_map[&MilliOpGraphTensorId { inner: 0 }];
        let out0 = MilliOpSplit::new(&mut graph, data_id, None, 0, Some(2), 0);
        let out1 = MilliOpSplit::new(&mut graph, data_id, None, 0, Some(2), 1);
        let mut output_map = HashMap::new();
        output_map.insert(out0, MilliOpGraphTensorId { inner: 1 });
        output_map.insert(out1, MilliOpGraphTensorId { inner: 2 });
        graph.set_output_map(output_map);

        let mut inputs = HashMap::new();
        let x = NumericTensor::<DynRank>::from_vec_shape(vec![1f32, 2., 3., 4.], vec![4]).unwrap();
        inputs.insert(MilliOpGraphTensorId { inner: 0 }, x);

        let mut backend = EvalBackend::NDArray;
        let mut obs = ();
        let res = graph.eval(&inputs, &mut obs, &mut backend).unwrap();
        let out0 = res[&MilliOpGraphTensorId { inner: 1 }].clone();
        let out1 = res[&MilliOpGraphTensorId { inner: 2 }].clone();

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
        let (mut graph, input_map) = MilliOpGraph::new(&[MilliOpGraphTensorId { inner: 0 }]);
        let data_id = input_map[&MilliOpGraphTensorId { inner: 0 }];
        let out = MilliOpSplit::new(&mut graph, data_id, None, -1, Some(2), 1);
        let mut output_map = HashMap::new();
        output_map.insert(out, MilliOpGraphTensorId { inner: 1 });
        graph.set_output_map(output_map);

        let mut inputs = HashMap::new();
        // 2x4
        let x = NumericTensor::<DynRank>::from_vec_shape(
            (1..=8).map(|v| v as f32).collect::<Vec<_>>(),
            vec![2, 4],
        )
        .unwrap();
        inputs.insert(MilliOpGraphTensorId { inner: 0 }, x);

        let mut backend = EvalBackend::NDArray;
        let mut obs = ();
        let res = graph.eval(&inputs, &mut obs, &mut backend).unwrap();
        let out = res[&MilliOpGraphTensorId { inner: 1 }].clone();
        assert_eq!(out.shape(), vec![2u64, 2u64]);
        let v: Vec<f32> = out.flatten().unwrap().try_into().unwrap();
        // Expect second half along last axis: [[3,4],[7,8]]
        assert_eq!(v, vec![3., 4., 7., 8.]);
    }

    #[test]
    fn test_split_num_outputs_uneven_distribution_axis0() {
        let (mut graph, input_map) = MilliOpGraph::new(&[MilliOpGraphTensorId { inner: 0 }]);
        let data_id = input_map[&MilliOpGraphTensorId { inner: 0 }];
        // dim=5, num_outputs=2 -> sizes [3,2]
        let out0 = MilliOpSplit::new(&mut graph, data_id, None, 0, Some(2), 0);
        let out1 = MilliOpSplit::new(&mut graph, data_id, None, 0, Some(2), 1);
        let mut output_map = HashMap::new();
        output_map.insert(out0, MilliOpGraphTensorId { inner: 1 });
        output_map.insert(out1, MilliOpGraphTensorId { inner: 2 });
        graph.set_output_map(output_map);

        let mut inputs = HashMap::new();
        let x =
            NumericTensor::<DynRank>::from_vec_shape(vec![1f32, 2., 3., 4., 5.], vec![5]).unwrap();
        inputs.insert(MilliOpGraphTensorId { inner: 0 }, x);

        let mut backend = EvalBackend::NDArray;
        let mut obs = ();
        let res = graph.eval(&inputs, &mut obs, &mut backend).unwrap();
        let out0 = res[&MilliOpGraphTensorId { inner: 1 }].clone();
        let out1 = res[&MilliOpGraphTensorId { inner: 2 }].clone();

        assert_eq!(out0.shape(), vec![3u64]);
        assert_eq!(out1.shape(), vec![2u64]);
        let v0: Vec<f32> = out0.flatten().unwrap().try_into().unwrap();
        let v1: Vec<f32> = out1.flatten().unwrap().try_into().unwrap();
        assert_eq!(v0, vec![1., 2., 3.]);
        assert_eq!(v1, vec![4., 5.]);
    }
}
