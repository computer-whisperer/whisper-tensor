use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::*;
use crate::numeric_scalar::NumericScalar;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{
    ONNXDecodingError, SymbolicGraphTensorId, query_attribute_float, query_attribute_floats,
    query_attribute_int, query_attribute_ints, query_attribute_tensor,
};
use crate::{DynRank, onnx};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConstantOfShapeOperation {
    value: NumericScalar,
    input: SymbolicGraphTensorId,
    output: SymbolicGraphTensorId,
}

impl ConstantOfShapeOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("ConstantOfShape"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("ConstantOfShape"));
        }

        let value = query_attribute_tensor(attributes, "value")
            .map(|x| x.first_element())
            .unwrap_or(NumericScalar::F32(0.0));

        Ok(Self {
            value,
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("ConstantOfShape"))?,
            output: outputs[0]
                .ok_or(ONNXDecodingError::InvalidOperatorOutputs("ConstantOfShape"))?,
        })
    }
}

impl Operation for ConstantOfShapeOperation {
    fn get_op_type_name(&self) -> String {
        "Constant of Shape".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.input]
    }
    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let out = graph.push_op(AnyMilliOp::ConstantOfShape(ConstantOfShape::new(
            self.value.clone(),
            input_map[&self.input],
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConstantOperation {
    pub value: NDArrayNumericTensor<DynRank>,
    output: SymbolicGraphTensorId,
}

impl ConstantOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if !inputs.is_empty() {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Constant"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Constant"));
        }

        let value = if let Some(tensor) = query_attribute_tensor(attributes, "value") {
            tensor
        } else if let Some(value_float) = query_attribute_float(attributes, "value_float") {
            NDArrayNumericTensor::from(vec![value_float]).try_to_rank()?
        } else if let Some(value_floats) = query_attribute_floats(attributes, "value_floats") {
            NDArrayNumericTensor::from(value_floats).try_to_rank()?
        } else if let Some(value_int) = query_attribute_int(attributes, "value_int") {
            NDArrayNumericTensor::from(vec![value_int]).try_to_rank()?
        } else if let Some(value_ints) = query_attribute_ints(attributes, "value_ints") {
            NDArrayNumericTensor::from(value_ints).try_to_rank()?
        } else {
            Err(ONNXDecodingError::MissingAttribute(
                "Constant".to_string(),
                "value".to_string(),
            ))?
        };

        Ok(Self {
            value,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Constant"))?,
        })
    }
}

impl Operation for ConstantOperation {
    fn get_op_type_name(&self) -> String {
        "Constant".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![]
    }

    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, _input_map) = MilliOpGraph::new(&self.get_inputs());

        let out = graph.push_op(AnyMilliOp::Constant(Constant::new(
            self.value.clone(),
        )));

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
