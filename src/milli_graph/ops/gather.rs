use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::graph::Node;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gather {
    output: MilliOpGraphTensorId,
    data: MilliOpGraphTensorId,
    indices: MilliOpGraphTensorId,
    axis: i64,
}

impl Gather {
    pub fn push_new<T: std::hash::Hash + Clone + Eq + 'static>(
        graph: &mut MilliOpGraph<T>,
        data: MilliOpGraphTensorId,
        indices: MilliOpGraphTensorId,
        axis: i64,
    ) -> MilliOpGraphTensorId {
        let output = graph.get_new_tensor_id();
        let node = Self {
            output,
            data,
            indices,
            axis,
        };
        graph.push_op(AnyMilliOp::Gather(node));
        output
    }
}

impl Node<MilliOpGraphTensorId> for Gather {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "Gather".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = MilliOpGraphTensorId>> {
        Box::new(vec![self.data, self.indices].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = MilliOpGraphTensorId>> {
        Box::new(vec![self.output].into_iter())
    }
}

impl MilliOp for Gather {
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<
        Box<dyn Iterator<Item = (MilliOpGraphTensorId, NumericTensor<DynRank>)>>,
        MilliOpGraphError,
    > {
        let out = NumericTensor::<DynRank>::gather(
            &inputs[&self.data],
            &inputs[&self.indices],
            self.axis,
            backend,
        )?;
        Ok(Box::new([(self.output, out)].into_iter()))
    }
}
