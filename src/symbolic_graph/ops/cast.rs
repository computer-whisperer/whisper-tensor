use crate::dtype::DType;
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::*;
use crate::onnx;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{ONNXDecodingError, SymbolicGraphTensorId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CastLikeOperation {
    input: SymbolicGraphTensorId,
    target_type: SymbolicGraphTensorId,
    output: SymbolicGraphTensorId,
}

impl CastLikeOperation {
    pub fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        _attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("CastLike"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("CastLike"));
        }
        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("CastLike"))?,
            target_type: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("CastLike"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("CastLike"))?,
        })
    }
}

impl Operation for CastLikeOperation {
    fn get_op_type_name(&self) -> String {
        "Cast Like".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.input, self.target_type]
    }
    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let out = graph.push_op(AnyMilliOp::CastLike(CastLike::new(
            input_map[&self.input],
            input_map[&self.target_type],
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CastOperation {
    input: SymbolicGraphTensorId,
    output: SymbolicGraphTensorId,
    to: DType,
}

impl CastOperation {
    pub fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Cast"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Cast"));
        }
        let to_i = attributes
            .iter()
            .find(|a| a.name == "to")
            .ok_or(ONNXDecodingError::MissingAttribute(
                "Cast".to_string(),
                "to".to_string(),
            ))?
            .i as i32;
        let to_datatype = onnx::tensor_proto::DataType::try_from(to_i)
            .map_err(|x| ONNXDecodingError::ProtobufDecodeError(x.into()))?;
        let to = DType::try_from(to_datatype)?;
        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Cast"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Cast"))?,
            to,
        })
    }
}

impl Operation for CastOperation {
    fn get_op_type_name(&self) -> String {
        "Cast".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.input]
    }
    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let out = graph.push_op(AnyMilliOp::Cast(Cast::new(
            input_map[&self.input],
            self.to,
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
