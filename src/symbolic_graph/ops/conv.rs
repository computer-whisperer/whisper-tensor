use crate::milli_graph::MilliOpGraph;
use crate::onnx;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{
    ONNXDecodingError, SymbolicGraphTensorId, query_attribute_int, query_attribute_ints,
    query_attribute_string,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) enum ConvOperationAutoPad {
    NotSet,
    SameUpper,
    SameLower,
    Valid,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConvOperation {
    input: SymbolicGraphTensorId,
    output: SymbolicGraphTensorId,
    weight: SymbolicGraphTensorId,
    bias: Option<SymbolicGraphTensorId>,
    auto_pad: ConvOperationAutoPad,
    dilations: Vec<i64>,
    group: i64,
    kernel_shape: Vec<i64>,
    pads: Vec<i64>,
    strides: Vec<i64>,
}

impl ConvOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() < 2 || inputs.len() > 3 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Conv"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Conv"));
        }

        let auto_pad_str = query_attribute_string(attributes, "auto_pad");
        let auto_pad = match auto_pad_str {
            Some(x) => match x.to_lowercase().as_str() {
                "notset" => ConvOperationAutoPad::NotSet,
                "same_upper" => ConvOperationAutoPad::SameUpper,
                "same_lower" => ConvOperationAutoPad::SameLower,
                "valid" => ConvOperationAutoPad::Valid,
                _ => ConvOperationAutoPad::NotSet,
            },
            _ => ConvOperationAutoPad::NotSet,
        };

        let dilations = query_attribute_ints(attributes, "dilations").unwrap_or_default();
        let group = query_attribute_int(attributes, "group").unwrap_or(1);
        let kernel_shape = query_attribute_ints(attributes, "kernel_shape").unwrap_or_default();
        let pads = query_attribute_ints(attributes, "pads").unwrap_or_default();
        let strides = query_attribute_ints(attributes, "strides").unwrap_or_default();

        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Conv"))?,
            weight: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Conv"))?,
            bias: if inputs.len() > 2 {
                Some(inputs[2].ok_or(ONNXDecodingError::InvalidOperatorInputs("Conv"))?)
            } else {
                None
            },
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Conv"))?,
            auto_pad,
            dilations,
            group,
            kernel_shape,
            pads,
            strides,
        })
    }
}

impl Operation for ConvOperation {
    fn get_op_type_name(&self) -> String {
        "Conv".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        if let Some(bias) = self.bias {
            vec![self.input, self.weight, bias]
        } else {
            vec![self.input, self.weight]
        }
    }

    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        unimplemented!();
    }
}
