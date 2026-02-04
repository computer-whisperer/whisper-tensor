use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArgMin {
    global_id: GlobalId,
    output: GlobalId,
    input: GlobalId,
    axis: i64,
    keepdims: bool,
    select_last_index: bool,
}

impl ArgMin {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        input: GlobalId,
        axis: i64,
        keepdims: bool,
        select_last_index: bool,
        rng: &mut impl rand::Rng
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            output,
            input,
            axis,
            keepdims,
            select_last_index,
        };
        graph.push_op(AnyMilliOp::ArgMin(node));
        output
    }
}

impl MilliOp for ArgMin {
    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<
        Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>,
        MilliOpGraphError,
    > {
        let input = inputs[&self.input].clone();
        let axis = if self.axis < 0 {
            input.shape().len() as i64 + self.axis
        } else {
            self.axis
        } as usize;
        let min = input.argmin(axis, self.keepdims, self.select_last_index, backend)?;
        Ok(Box::new([(self.output, min)].into_iter()))
    }
}

impl Node for ArgMin {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "ArgMin".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.input].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
}
