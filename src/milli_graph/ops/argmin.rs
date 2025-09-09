use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::graph::Node;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArgMin {
    output: MilliOpGraphTensorId,
    input: MilliOpGraphTensorId,
    axis: i64,
    keepdims: bool,
    select_last_index: bool,
}

impl ArgMin {
    pub fn new<T: std::hash::Hash + Clone + Eq>(
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
        graph.push_op(AnyMilliOp::ArgMin(node));
        output
    }
}

impl MilliOp for ArgMin {
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<
        impl Iterator<Item = (MilliOpGraphTensorId, NumericTensor<DynRank>)>,
        MilliOpGraphError,
    > {
        let input = inputs[&self.input].clone();
        let axis = if self.axis < 0 {
            input.shape().len() as i64 + self.axis
        } else {
            self.axis
        } as usize;
        let min = input.argmin(axis, self.keepdims, self.select_last_index, backend)?;
        Ok([(self.output, min)].into_iter())
    }

    fn get_name(&self) -> String {
        "ArgMin".to_string()
    }
}

impl Node<MilliOpGraphTensorId> for ArgMin {
    fn inputs(&self) -> impl Iterator<Item = MilliOpGraphTensorId> {
        vec![self.input].into_iter()
    }
    fn outputs(&self) -> impl Iterator<Item = MilliOpGraphTensorId> {
        vec![self.output].into_iter()
    }
}
