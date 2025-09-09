use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Range {
    output: MilliOpGraphTensorId,
    start: MilliOpGraphTensorId,
    end: MilliOpGraphTensorId,
    delta: MilliOpGraphTensorId,
}

impl Range {
    pub fn new<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        start: MilliOpGraphTensorId,
        end: MilliOpGraphTensorId,
        delta: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        let output = graph.get_new_tensor_id();
        let node = Self {
            output,
            start,
            end,
            delta,
        };
        graph.push_op(AnyMilliOp::Range(node));
        output
    }
}

impl crate::graph::Node<MilliOpGraphTensorId> for Range {
    fn inputs(&self) -> impl Iterator<Item = MilliOpGraphTensorId> {
        vec![self.start, self.end, self.delta].into_iter()
    }
    fn outputs(&self) -> impl Iterator<Item = MilliOpGraphTensorId> {
        vec![self.output].into_iter()
    }
}

impl MilliOp for Range {
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<
        impl Iterator<Item = (MilliOpGraphTensorId, NumericTensor<DynRank>)>,
        MilliOpGraphError,
    > {
        let out: NumericTensor<DynRank> = NumericTensor::<P1>::range(
            inputs[&self.start].first_element(),
            inputs[&self.end].first_element(),
            inputs[&self.delta].first_element(),
            backend,
        )?
        .to_dyn_rank();
        Ok([(self.output, out)].into_iter())
    }

    fn get_name(&self) -> String {
        "Range".to_string()
    }
}
