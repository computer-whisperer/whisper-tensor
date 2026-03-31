use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::{MilliLoweringContext, MilliOpGraph};
use crate::onnx::AttributeProto;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{ONNXDecodingError, query_attribute_int};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// ONNX ReverseSequence operator.
///
/// Reverses variable-length slices along a time axis, independently for each
/// element along a batch axis.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReverseSequenceOperation {
    global_id: GlobalId,
    input: GlobalId,
    sequence_lens: GlobalId,
    output: GlobalId,
    batch_axis: i64,
    time_axis: i64,
}

impl ReverseSequenceOperation {
    pub fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() < 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("ReverseSequence"));
        }
        if outputs.is_empty() {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("ReverseSequence"));
        }

        let batch_axis = query_attribute_int(attributes, "batch_axis").unwrap_or(1);
        let time_axis = query_attribute_int(attributes, "time_axis").unwrap_or(0);

        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("ReverseSequence"))?,
            sequence_lens: inputs[1]
                .ok_or(ONNXDecodingError::InvalidOperatorInputs("ReverseSequence"))?,
            output: outputs[0]
                .ok_or(ONNXDecodingError::InvalidOperatorOutputs("ReverseSequence"))?,
            batch_axis,
            time_axis,
        })
    }
}

impl Node for ReverseSequenceOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "ReverseSequence".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        Box::new([self.input, self.sequence_lens].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        Box::new([self.output].into_iter())
    }
}

impl Operation for ReverseSequenceOperation {
    fn parameters(&self) -> Vec<Property> {
        vec![
            Property::new("batch_axis", PropertyValue::Int(self.batch_axis)),
            Property::new("time_axis", PropertyValue::Int(self.time_axis)),
        ]
    }

    fn is_differentiable(&self) -> bool {
        false
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let out = crate::milli_graph::ops::ReverseSequence::push_new(
            &mut graph,
            input_map[&self.input],
            input_map[&self.sequence_lens],
            self.batch_axis,
            self.time_axis,
            rng,
        );
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
