use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::Transpose;
use crate::onnx;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{ONNXDecodingError, query_attribute_ints};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rand::Rng;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransposeOperation {
    global_id: GlobalId,
    input: GlobalId,
    output: GlobalId,
    perm: Option<Vec<i64>>,
}

impl TransposeOperation {
    pub fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Transpose"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Transpose"));
        }
        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Transpose"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Transpose"))?,
            perm: query_attribute_ints(attributes, "perm"),
        })
    }
}

impl Node for TransposeOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Transpose".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}
impl Operation for TransposeOperation {
    fn parameters(&self) -> Vec<Property> {
        let mut params = Vec::new();
        if let Some(perm) = &self.perm {
            params.push(Property::new("perm", PropertyValue::IntList(perm.clone())));
        }
        params
    }

    fn get_milli_op_graph(&self, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let out = Transpose::push_new(&mut graph, input_map[&self.input], self.perm.clone(), rng);
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
