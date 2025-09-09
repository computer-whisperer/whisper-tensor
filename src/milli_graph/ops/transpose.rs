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
    pub fn new<T: std::hash::Hash + Clone + Eq>(
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
    fn inputs(&self) -> impl Iterator<Item = MilliOpGraphTensorId> {
        vec![self.data].into_iter()
    }
    fn outputs(&self) -> impl Iterator<Item = MilliOpGraphTensorId> {
        vec![self.output].into_iter()
    }
}

impl MilliOp for Transpose {
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<
        impl Iterator<Item = (MilliOpGraphTensorId, NumericTensor<DynRank>)>,
        MilliOpGraphError,
    > {
        let out = inputs[&self.data].transpose(self.perm.clone(), backend)?;
        Ok([(self.output, out)].into_iter())
    }

    fn get_name(&self) -> String {
        "Transpose".to_string()
    }
}
