use crate::graph::Node;
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::*;
use crate::onnx;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{ONNXDecodingError, SymbolicGraphTensorId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GatherOperation {
    input: SymbolicGraphTensorId,
    indices: SymbolicGraphTensorId,
    output: SymbolicGraphTensorId,
    axis: i64,
}

impl GatherOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Gather"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Gather"));
        }
        let mut axis = 0;
        for attribute in attributes {
            if attribute.name.as_str() == "axis" {
                axis = attribute.i;
            }
        }
        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Gather"))?,
            indices: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Gather"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Gather"))?,
            axis,
        })
    }
}

impl Node<SymbolicGraphTensorId> for GatherOperation {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "Gather".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new(std::iter::once(self.input).chain(std::iter::once(self.indices)))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for GatherOperation {
    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs());
        let out = Gather::push_new(
            &mut graph,
            input_map[&self.input],
            input_map[&self.indices],
            self.axis,
        );
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
