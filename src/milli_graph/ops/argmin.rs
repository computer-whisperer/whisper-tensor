use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError, MilliOpGraphNodeId, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::graph::Node;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpArgMin {
    output: MilliOpGraphTensorId,
    input: MilliOpGraphTensorId,
    axis: i64,
    keepdims: bool,
    select_last_index: bool,
}

impl MilliOpArgMin {
    pub fn new(
        graph: &mut MilliOpGraph<MilliOpGraphTensorId>,
        input: MilliOpGraphTensorId,
        axis: i64,
        keepdims: bool,
        select_last_index: bool,
    ) -> MilliOpGraphNodeId {
        let node = Self {
            output: graph.get_new_tensor_id(),
            input,
            axis,
            keepdims,
            select_last_index,
        };
        graph.push_op(AnyMilliOp::ArgMin(node))
    }
}

impl MilliOp for MilliOpArgMin {
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<impl Iterator<Item=(MilliOpGraphTensorId, NumericTensor<DynRank>)>, MilliOpGraphError>{
        let input = inputs[&self.input].clone();
        let axis = if self.axis < 0 {
            input.shape().len() as i64 + self.axis
        } else {
            self.axis
        } as usize;
        let min = input.argmin(axis, self.keepdims, self.select_last_index, backend)?;
        Ok([(self.output, min)].iter())
    }

    fn get_name(&self) -> String {
        "ArgMin".to_string()
    }
}


impl Node<MilliOpGraphTensorId> for MilliOpArgMin {
    fn inputs(&self) -> impl Iterator<Item=MilliOpGraphTensorId> {
        [self.input].iter().cloned()
    }
    fn outputs(&self) -> impl Iterator<Item=MilliOpGraphTensorId>{
        [self.output].iter().cloned()
    }
}
