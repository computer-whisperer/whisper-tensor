use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::MilliOpGraphError;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::numeric_scalar::NumericScalarType;
use crate::numeric_tensor::NumericTensor;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CumSum {
    global_id: GlobalId,
    output: GlobalId,
    input: GlobalId,
    axis: GlobalId,
    exclusive: bool,
    reverse: bool,
}

impl CumSum {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        input: GlobalId,
        axis: GlobalId,
        exclusive: bool,
        reverse: bool,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
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

impl CumSum {
    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.input, map);
        super::remap(&mut self.axis, map);
    }
}

impl Node for CumSum {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "CumSum".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.input, self.axis].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
}

impl MilliOp for CumSum {
    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        let data = &inputs[&self.input];
        let axis = i64::cast_from_numeric_scalar(&inputs[&self.axis].first_element());
        let out = data.cumsum(Some(axis as isize), self.exclusive, self.reverse, backend)?;
        Ok(Box::new([(self.output, out)].into_iter()))
    }
}
