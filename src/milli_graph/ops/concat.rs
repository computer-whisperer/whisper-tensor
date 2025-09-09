use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Concat {
    output: MilliOpGraphTensorId,
    inputs: Vec<MilliOpGraphTensorId>,
    axis: i64,
}

impl Concat {
    pub fn new<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, inputs: Vec<MilliOpGraphTensorId>, axis: i64) -> MilliOpGraphTensorId {
        let output = graph.get_new_tensor_id();
        let node = Self { output, inputs, axis };
        graph.push_op(AnyMilliOp::Concat(node));
        output
    }
}

impl crate::graph::Node<MilliOpGraphTensorId> for Concat {
    fn inputs(&self) -> impl Iterator<Item=MilliOpGraphTensorId> { self.inputs.clone().into_iter() }
    fn outputs(&self) -> impl Iterator<Item=MilliOpGraphTensorId> { vec![self.output].into_iter() }
}

impl MilliOp for Concat {
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<impl Iterator<Item=(MilliOpGraphTensorId, NumericTensor<DynRank>)>, MilliOpGraphError> {
        let mut resolved_inputs = vec![];
        for input in &self.inputs {
            resolved_inputs.push(&inputs[input]);
        }
        let axis = if self.axis < 0 {
            resolved_inputs[0].shape().len() as i64 + self.axis
        } else {
            self.axis
        } as usize;
        let out = NumericTensor::<DynRank>::concat(
            resolved_inputs.as_slice(),
            axis,
            backend,
        )?;
        Ok([(self.output, out)].into_iter())
    }

    fn get_name(&self) -> String {
        "Concat".to_string()
    }
}
