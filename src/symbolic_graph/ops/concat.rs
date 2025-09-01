use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::*;
use crate::onnx;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{ONNXDecodingError, SymbolicGraphTensorId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConcatOperation {
    axis: i64,
    inputs: Vec<SymbolicGraphTensorId>,
    output: SymbolicGraphTensorId,
}

impl ConcatOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        attributes: &[onnx::AttributeProto],
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
            axis,
            inputs: inputs
                .iter()
                .map(|x| x.ok_or(ONNXDecodingError::InvalidOperatorInputs("Concat")))
                .collect::<Result<_, _>>()?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Concat"))?,
        })
    }
}

impl Operation for ConcatOperation {
    fn get_op_type_name(&self) -> String {
        "Concat".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        self.inputs.clone()
    }
    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let mut milli_inputs = vec![];
        for input in &self.inputs {
            milli_inputs.push(input_map[input]);
        }
        let out = graph.push_op(AnyMilliOp::Concat(MilliOpConcat::new(
            milli_inputs,
            self.axis,
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
