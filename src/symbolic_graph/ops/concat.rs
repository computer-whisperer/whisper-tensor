use crate::graph::{GlobalId, Node};
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::*;
use crate::onnx;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::ONNXDecodingError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rand::Rng;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConcatOperation {
    global_id: GlobalId,
    axis: i64,
    inputs: Vec<GlobalId>,
    output: GlobalId,
}

impl ConcatOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Concat"));
        }
        let mut axis = 0;
        for attribute in attributes {
            if attribute.name == "axis" {
                axis = attribute.i;
            }
        }

        Ok(Self {
            global_id: GlobalId::new(rng),
            axis,
            inputs: inputs
                .iter()
                .map(|x| x.ok_or(ONNXDecodingError::InvalidOperatorInputs("Concat")))
                .collect::<Result<_, _>>()?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Concat"))?,
        })
    }
}

impl Node for ConcatOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Concat".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(self.inputs.clone().into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for ConcatOperation {
    fn get_milli_op_graph(&self, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let mut milli_inputs = vec![];
        for input in &self.inputs {
            milli_inputs.push(input_map[input]);
        }
        let out = Concat::push_new(&mut graph, milli_inputs, self.axis, rng);
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
