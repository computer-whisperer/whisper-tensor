use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_scalar::NumericScalarType;
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CumSum {
    output: MilliOpGraphTensorId,
    input: MilliOpGraphTensorId,
    axis: MilliOpGraphTensorId,
    exclusive: bool,
    reverse: bool,
}

impl CumSum {
    pub fn push_new<T: std::hash::Hash + Clone + Eq + 'static>(
        graph: &mut MilliOpGraph<T>,
        input: MilliOpGraphTensorId,
        axis: MilliOpGraphTensorId,
        exclusive: bool,
        reverse: bool,
    ) -> MilliOpGraphTensorId {
        let output = graph.get_new_tensor_id();
        let node = Self {
            output,
            input,
            axis,
            exclusive,
            reverse,
        };
        graph.push_op(AnyMilliOp::CumSum(node));
        output
    }
}

use crate::graph::Node;

impl Node<MilliOpGraphTensorId> for CumSum {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "CumSum".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = MilliOpGraphTensorId>> {
        Box::new(vec![self.input, self.axis].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = MilliOpGraphTensorId>> {
        Box::new(vec![self.output].into_iter())
    }
}

impl MilliOp for CumSum {
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<
        Box<dyn Iterator<Item = (MilliOpGraphTensorId, NumericTensor<DynRank>)>>,
        MilliOpGraphError,
    > {
        let data = &inputs[&self.input];
        let axis = i64::cast_from_numeric_scalar(&inputs[&self.axis].first_element());
        let out = data.cumsum(Some(axis as isize), self.exclusive, self.reverse, backend)?;
        Ok(Box::new([(self.output, out)].into_iter()))
    }
}
