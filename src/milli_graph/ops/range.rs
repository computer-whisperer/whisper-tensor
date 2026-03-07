use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::graph::GlobalId;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::numeric_tensor::NumericTensor;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Range {
    global_id: GlobalId,
    output: GlobalId,
    start: GlobalId,
    end: GlobalId,
    delta: GlobalId,
}

impl Range {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        start: GlobalId,
        end: GlobalId,
        delta: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            output,
            start,
            end,
            delta,
        };
        graph.push_op(AnyMilliOp::Range(node));
        output
    }
}

impl Range {
    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.start, map);
        super::remap(&mut self.end, map);
        super::remap(&mut self.delta, map);
    }
}

impl crate::graph::Node for Range {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Range".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.start, self.end, self.delta].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
}

impl MilliOp for Range {
    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        let out: NumericTensor<DynRank> = NumericTensor::<P1>::range(
            inputs[&self.start].first_element(),
            inputs[&self.end].first_element(),
            inputs[&self.delta].first_element(),
            backend,
        )?
        .to_dyn_rank();
        Ok(Box::new([(self.output, out)].into_iter()))
    }
}
