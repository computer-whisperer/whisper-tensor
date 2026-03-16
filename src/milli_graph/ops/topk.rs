use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::numeric_tensor::NumericTensor;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopK {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    input: GlobalId,
    k: GlobalId,
    output_values: GlobalId,
    output_indices: GlobalId,
    axis: i64,
    largest: bool,
    sorted: bool,
}

impl TopK {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        input: GlobalId,
        k: GlobalId,
        axis: i64,
        largest: bool,
        sorted: bool,
        rng: &mut impl Rng,
    ) -> (GlobalId, GlobalId) {
        Self::push_new_with_label(graph, input, k, axis, largest, sorted, None, rng)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        input: GlobalId,
        k: GlobalId,
        axis: i64,
        largest: bool,
        sorted: bool,
        label: Option<String>,
        rng: &mut impl Rng,
    ) -> (GlobalId, GlobalId) {
        let output_values = graph.get_new_tensor_id(rng);
        let output_indices = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            label,
            input,
            k,
            output_values,
            output_indices,
            axis,
            largest,
            sorted,
        };
        graph.push_op(AnyMilliOp::TopK(node));
        (output_values, output_indices)
    }
}

impl TopK {
    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.input, map);
        super::remap(&mut self.k, map);
        super::remap(&mut self.output_values, map);
        super::remap(&mut self.output_indices, map);
    }
}

impl Node for TopK {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "TopK".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.input, self.k].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output_values, self.output_indices].into_iter())
    }
}

impl MilliOp for TopK {
    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        let (values, indices) = NumericTensor::<DynRank>::topk(
            &inputs[&self.input],
            &inputs[&self.k],
            self.axis,
            self.largest,
            self.sorted,
            backend,
        )?;
        Ok(Box::new(
            [(self.output_values, values), (self.output_indices, indices)].into_iter(),
        ))
    }
}
