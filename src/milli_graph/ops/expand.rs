use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Expand {
    output: MilliOpGraphTensorId,
    input: MilliOpGraphTensorId,
    shape: MilliOpGraphTensorId,
}

impl Expand {
    pub fn push_new<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        input: MilliOpGraphTensorId,
        shape: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        let output = graph.get_new_tensor_id();
        let node = Self {
            output,
            input,
            shape,
        };
        graph.push_op(AnyMilliOp::Expand(node));
        output
    }
}

impl crate::graph::Node<MilliOpGraphTensorId> for Expand {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "Expand".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = MilliOpGraphTensorId>> {
        Box::new(vec![self.input, self.shape].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = MilliOpGraphTensorId>> {
        Box::new(vec![self.output].into_iter())
    }
}

impl MilliOp for Expand {
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        _backend: &mut EvalBackend,
    ) -> Result<
        Box<dyn Iterator<Item = (MilliOpGraphTensorId, NumericTensor<DynRank>)>>,
        MilliOpGraphError,
    > {
        let shape: Vec<i64> = inputs[&self.shape].try_to_rank::<P1>()?.try_into()?;
        let shape_u = shape.iter().map(|x| *x as u64).collect::<Vec<u64>>();
        let mut x = inputs[&self.input].clone();
        while x.rank() < shape_u.len() {
            x = x.unsqueeze(0)?;
        }
        let shape_u = shape_u
            .iter()
            .zip(x.shape().iter())
            .map(|(a, b)| std::cmp::max(*a, *b))
            .collect::<Vec<u64>>();
        let out = x.expand(&shape_u)?;
        Ok(Box::new([(self.output, out)].into_iter()))
    }
}
