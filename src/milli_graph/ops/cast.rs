use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cast {
    output: MilliOpGraphTensorId,
    data: MilliOpGraphTensorId,
    dtype: DType,
}

impl Cast {
    pub fn push_new<T: std::hash::Hash + Clone + Eq + 'static>(
        graph: &mut MilliOpGraph<T>,
        data: MilliOpGraphTensorId,
        dtype: DType,
    ) -> MilliOpGraphTensorId {
        let output = graph.get_new_tensor_id();
        let node = Self {
            output,
            data,
            dtype,
        };
        graph.push_op(AnyMilliOp::Cast(node));
        output
    }
}

impl crate::graph::Node<MilliOpGraphTensorId> for Cast {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "Cast".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = MilliOpGraphTensorId>> {
        Box::new(vec![self.data].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = MilliOpGraphTensorId>> {
        Box::new(vec![self.output].into_iter())
    }
}

impl MilliOp for Cast {
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<
        Box<dyn Iterator<Item = (MilliOpGraphTensorId, NumericTensor<DynRank>)>>,
        MilliOpGraphError,
    > {
        let out = inputs[&self.data].cast(self.dtype, backend)?;
        Ok(Box::new([(self.output, out)].into_iter()))
    }
}
