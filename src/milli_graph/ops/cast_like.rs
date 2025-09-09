use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CastLike {
    output: MilliOpGraphTensorId,
    data: MilliOpGraphTensorId,
    target_type: MilliOpGraphTensorId,
}

impl CastLike {
    pub fn push_new<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        data: MilliOpGraphTensorId,
        target_type: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        let output = graph.get_new_tensor_id();
        let node = Self {
            output,
            data,
            target_type,
        };
        graph.push_op(AnyMilliOp::CastLike(node));
        output
    }
}

impl crate::graph::Node<MilliOpGraphTensorId> for CastLike {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "CastLike".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = MilliOpGraphTensorId>> {
        Box::new(vec![self.data, self.target_type].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = MilliOpGraphTensorId>> {
        Box::new(vec![self.output].into_iter())
    }
}

impl MilliOp for CastLike {
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<
        Box<dyn Iterator<Item = (MilliOpGraphTensorId, NumericTensor<DynRank>)>>,
        MilliOpGraphError,
    > {
        let out = inputs[&self.data].cast(inputs[&self.target_type].dtype(), backend)?;
        Ok(Box::new([(self.output, out)].into_iter()))
    }
}
