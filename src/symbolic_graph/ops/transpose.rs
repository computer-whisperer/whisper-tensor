use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::{AnyMilliOp, MilliOpTranspose};
use crate::onnx;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{ONNXDecodingError, SymbolicGraphTensorId, query_attribute_ints};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransposeOperation {
    input: SymbolicGraphTensorId,
    output: SymbolicGraphTensorId,
    perm: Option<Vec<i64>>,
}

impl TransposeOperation {
    pub fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Transpose"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Transpose"));
        }
        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Transpose"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Transpose"))?,
            perm: query_attribute_ints(attributes, "perm"),
        })
    }
}

impl Operation for TransposeOperation {
    fn get_op_type_name(&self) -> String {
        "Transpose".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.input]
    }
    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let out = graph.push_op(AnyMilliOp::Transpose(MilliOpTranspose::new(
            input_map[&self.input],
            self.perm.clone(),
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
