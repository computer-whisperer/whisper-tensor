use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::graph::Node;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transpose {
    output: MilliOpGraphTensorId,
    data: MilliOpGraphTensorId,
    perm: Option<Vec<i64>>,
}

impl Transpose {
    pub fn push_new<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        data: MilliOpGraphTensorId,
        perm: Option<Vec<i64>>,
    ) -> MilliOpGraphTensorId {
        let output = graph.get_new_tensor_id();
        let node = Self { output, data, perm };
        graph.push_op(AnyMilliOp::Transpose(node));
        output
    }
}

impl Node<MilliOpGraphTensorId> for Transpose {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "Transpose".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = MilliOpGraphTensorId>> {
        Box::new(vec![self.data].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = MilliOpGraphTensorId>> {
        Box::new(vec![self.output].into_iter())
    }
}

impl MilliOp for Transpose {
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<
        Box<dyn Iterator<Item = (MilliOpGraphTensorId, NumericTensor<DynRank>)>>,
        MilliOpGraphError,
    > {
        let out = inputs[&self.data].transpose(self.perm.clone(), backend)?;
        Ok(Box::new([(self.output, out)].into_iter()))
    }
}
