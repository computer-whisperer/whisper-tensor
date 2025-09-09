use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::*;
use crate::onnx;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{
    ONNXDecodingError, SymbolicGraphTensorId, query_attribute_int, query_attribute_ints,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SplitOperation {
    axis: Option<i64>,
    num_outputs: Option<i64>,
    input: SymbolicGraphTensorId,
    split: Option<SymbolicGraphTensorId>,
    split_attribute: Option<Vec<i64>>,
    outputs: Vec<SymbolicGraphTensorId>,
}

impl SplitOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.is_empty() || inputs.len() > 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Split"));
        }

        let axis = query_attribute_int(attributes, "axis");
        let num_outputs = query_attribute_int(attributes, "num_outputs");
        let split_attribute = query_attribute_ints(attributes, "split");

        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Split"))?,
            split: if inputs.len() > 1 {
                Some(inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Split"))?)
            } else {
                None
            },
            outputs: outputs
                .iter()
                .map(|x| x.ok_or(ONNXDecodingError::InvalidOperatorOutputs("Split")))
                .collect::<Result<_, _>>()?,
            split_attribute,
            axis,
            num_outputs,
        })
    }
}

impl Operation for SplitOperation {
    fn get_op_type_name(&self) -> String {
        "Split".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        if let Some(split) = self.split {
            vec![self.input, split]
        } else {
            vec![self.input]
        }
    }
    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        self.outputs.clone()
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());

        let mut output_map = HashMap::new();

        let split = if let Some(split) = self.split {
            Some(MilliOpTensorIDOrLiteral::TensorID(input_map[&split]))
        } else {
            self.split_attribute.as_ref().map(|split| {
                MilliOpTensorIDOrLiteral::Literal(
                    NDArrayNumericTensor::from_vec(split.clone()).to_dyn(),
                )
            })
        };

        for (output_id, output_tensor_id) in self.outputs.iter().enumerate() {
            let out = MilliOpSplit::new(
                &mut graph,
                input_map[&self.input],
                split.clone(),
                self.axis.unwrap_or_default(),
                self.num_outputs.map(|x| x as usize),
                output_id,
            );

            output_map.insert(out, *output_tensor_id);
        }
        graph.set_output_map(output_map);
        graph
    }
}
