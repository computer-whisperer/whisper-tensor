use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::graph::Node;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArgMax {
    output: MilliOpGraphTensorId,
    input: MilliOpGraphTensorId,
    axis: i64,
    keepdims: bool,
    select_last_index: bool,
}

impl ArgMax {
    pub fn push_new<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        input: MilliOpGraphTensorId,
        axis: i64,
        keepdims: bool,
        select_last_index: bool,
    ) -> MilliOpGraphTensorId {
        let output = graph.get_new_tensor_id();
        let node = Self {
            output,
            input,
            axis,
            keepdims,
            select_last_index,
        };
        graph.push_op(AnyMilliOp::ArgMax(node));
        output
    }
}

impl MilliOp for ArgMax {
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<
        Box<dyn Iterator<Item = (MilliOpGraphTensorId, NumericTensor<DynRank>)>>,
        MilliOpGraphError,
    > {
        let input = inputs[&self.input].clone();
        let axis = if self.axis < 0 {
            input.shape().len() as i64 + self.axis
        } else {
            self.axis
        } as usize;
        let max = input.argmax(axis, self.keepdims, self.select_last_index, backend)?;
        Ok(Box::new([(self.output, max)].into_iter()))
    }
}

impl Node<MilliOpGraphTensorId> for ArgMax {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "ArgMax".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = MilliOpGraphTensorId>> {
        Box::new(vec![self.input].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = MilliOpGraphTensorId>> {
        Box::new(vec![self.output].into_iter())
    }
}
