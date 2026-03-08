use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::*;
use crate::onnx;
use crate::symbolic_graph::ONNXDecodingError;
use crate::symbolic_graph::ops::Operation;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TopKOperation {
    global_id: GlobalId,
    input: GlobalId,
    k: GlobalId,
    output_values: GlobalId,
    output_indices: GlobalId,
    axis: i64,
    largest: bool,
    sorted: bool,
}

impl TopKOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() < 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("TopK"));
        }
        if outputs.len() < 2 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("TopK"));
        }
        let mut axis = -1i64;
        let mut largest = true;
        let mut sorted = true;
        for attribute in attributes {
            match attribute.name.as_str() {
                "axis" => axis = attribute.i,
                "largest" => largest = attribute.i != 0,
                "sorted" => sorted = attribute.i != 0,
                _ => {}
            }
        }
        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("TopK"))?,
            k: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("TopK"))?,
            output_values: outputs[0]
                .ok_or(ONNXDecodingError::InvalidOperatorOutputs("TopK"))?,
            output_indices: outputs[1]
                .ok_or(ONNXDecodingError::InvalidOperatorOutputs("TopK"))?,
            axis,
            largest,
            sorted,
        })
    }
}

impl Node for TopKOperation {
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

impl Operation for TopKOperation {
    fn parameters(&self) -> Vec<Property> {
        vec![
            Property::new("axis", PropertyValue::Int(self.axis)),
            Property::new("largest", PropertyValue::Int(self.largest as i64)),
            Property::new("sorted", PropertyValue::Int(self.sorted as i64)),
        ]
    }

    fn get_milli_op_graph(&self, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let (out_values, out_indices) = TopK::push_new(
            &mut graph,
            input_map[&self.input],
            input_map[&self.k],
            self.axis,
            self.largest,
            self.sorted,
            rng,
        );
        let mut output_map = HashMap::new();
        output_map.insert(out_values, self.output_values);
        output_map.insert(out_indices, self.output_indices);
        graph.set_output_map(output_map);
        graph
    }
}
