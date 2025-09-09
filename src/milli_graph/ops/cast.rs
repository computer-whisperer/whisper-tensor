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
    pub fn new<T: std::hash::Hash + Clone + Eq>(
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
    fn inputs(&self) -> impl Iterator<Item = MilliOpGraphTensorId> {
        vec![self.data].into_iter()
    }
    fn outputs(&self) -> impl Iterator<Item = MilliOpGraphTensorId> {
        vec![self.output].into_iter()
    }
}

impl MilliOp for Cast {
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<
        impl Iterator<Item = (MilliOpGraphTensorId, NumericTensor<DynRank>)>,
        MilliOpGraphError,
    > {
        let out = inputs[&self.data].cast(self.dtype, backend)?;
        Ok([(self.output, out)].into_iter())
    }

    fn get_name(&self) -> String {
        "Cast".to_string()
    }
}
