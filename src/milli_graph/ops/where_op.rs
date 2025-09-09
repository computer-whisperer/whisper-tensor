use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::graph::Node;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Where {
    output: MilliOpGraphTensorId,
    condition: MilliOpGraphTensorId,
    x: MilliOpGraphTensorId,
    y: MilliOpGraphTensorId,
}

impl Where {
    pub fn new<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, condition: MilliOpGraphTensorId, x: MilliOpGraphTensorId, y: MilliOpGraphTensorId) -> MilliOpGraphTensorId {
        let output = graph.get_new_tensor_id();
        let node = Self { output, condition, x, y };
        graph.push_op(AnyMilliOp::Where(node));
        output
    }
}

impl Node<MilliOpGraphTensorId> for Where {
    fn inputs(&self) -> impl Iterator<Item = MilliOpGraphTensorId> { vec![self.condition, self.x, self.y].into_iter() }
    fn outputs(&self) -> impl Iterator<Item = MilliOpGraphTensorId> { vec![self.output].into_iter() }
}

impl MilliOp for Where {
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<impl Iterator<Item=(MilliOpGraphTensorId, NumericTensor<DynRank>)>, MilliOpGraphError> {
        let out = inputs[&self.condition].where_op(&inputs[&self.x], &inputs[&self.y], backend)?;
        Ok([(self.output, out)].into_iter())
    }

    fn get_name(&self) -> String {
        "Where".to_string()
    }
}
