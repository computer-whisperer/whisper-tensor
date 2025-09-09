use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;
use crate::graph::Node;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shape {
    output: MilliOpGraphTensorId,
    input: MilliOpGraphTensorId,
}

impl Shape {
    pub fn new<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, input: MilliOpGraphTensorId) -> MilliOpGraphTensorId {
        let output = graph.get_new_tensor_id();
        let node = Self { output, input };
        graph.push_op(AnyMilliOp::Shape(node));
        output
    }
}

impl MilliOp for Shape {
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        _backend: &mut EvalBackend,
    ) -> Result<impl Iterator<Item=(MilliOpGraphTensorId, NumericTensor<DynRank>)>, MilliOpGraphError> {
        let output_shape = inputs[&self.input]
            .shape()
            .into_iter()
            .map(|x| x as i64)
            .collect::<Vec<_>>();
        let out: NumericTensor<DynRank> = NDArrayNumericTensor::<P1>::from(output_shape)
            .to_dyn()
            .into();
        Ok([(self.output, out)].into_iter())
    }

    fn get_name(&self) -> String {
        "Shape".to_string()
    }
}

impl Node<MilliOpGraphTensorId> for Shape {
    fn inputs(&self) -> impl Iterator<Item=MilliOpGraphTensorId> { vec![self.input].into_iter() }
    fn outputs(&self) -> impl Iterator<Item=MilliOpGraphTensorId> { vec![self.output].into_iter() }
}
