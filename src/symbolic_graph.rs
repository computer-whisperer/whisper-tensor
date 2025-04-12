use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::HashMap;
use prost::Message;

pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum DType {
    F32,
}

impl DType {
    fn from_onnx(onnx_dtype: onnx::tensor_proto::DataType) -> Result<Self, ONNXDecodingError> {
        match onnx_dtype {
            onnx::tensor_proto::DataType::Float => Ok(DType::F32),
            x => Err(ONNXDecodingError::UnsupportedDTypeError(x)),
        }
    }
}

#[derive(Debug, thiserror::Error)]
enum SymbolicGraphError {
    #[error("Invalid operator inputs")]
    InvalidOperatorInputs,
    #[error("Invalid operator outputs")]
    InvalidOperatorOutputs,
}

#[derive(Debug, thiserror::Error)]
pub enum ONNXDecodingError {
    #[error("Error constructing graph node {0}, {1}")]
    GraphConstructionError(String, SymbolicGraphError),
    #[error("Missing field \"{0}\"")]
    MissingField(&'static str),
    #[error("Unsupported Tensor Type: {0:?}")]
    UnsupportedTypeValue(onnx::type_proto::Value),
    #[error("Protobuf decoding error")]
    ProtobufDecodeError(#[from] anyhow::Error),
    #[error("Unsupported DType {0:?}")]
    UnsupportedDTypeError(onnx::tensor_proto::DataType),
    #[error("Unsupported ONNX type {0}")]
    UnsupportedONNXType(String),
    #[error("Negative dimension")]
    NegativeDimensionError,
    #[error("Unknown tensor name \"{0}\"")]
    UnknownTensorName(String),
}

type UnknownDimensionId = usize;
type TensorId = usize;

trait OperationData: Any {
    fn get_inputs(&self) -> Vec<TensorId>;
    fn get_outputs(&self) -> Vec<TensorId>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct BinaryOpData {
    a: TensorId,
    b: TensorId,
    output: TensorId,
}

impl BinaryOpData {
    fn new(inputs: Vec<TensorId>, outputs: Vec<TensorId>) -> Result<Self, SymbolicGraphError> {
        if inputs.len() != 2 {
            return Err(SymbolicGraphError::InvalidOperatorInputs);
        }
        if outputs.len() != 1 {
            return Err(SymbolicGraphError::InvalidOperatorOutputs);
        }
        Ok(BinaryOpData {
            a: inputs[0],
            b: inputs[1],
            output: outputs[0],
        })
    }


}

impl OperationData for BinaryOpData {
    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.a, self.b]
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct UnaryOpData {
    input: TensorId,
    output: TensorId,
}

impl UnaryOpData {
    fn new(inputs: Vec<TensorId>, outputs: Vec<TensorId>) -> Result<Self, SymbolicGraphError> {
        if inputs.len() != 1 {
            return Err(SymbolicGraphError::InvalidOperatorInputs);
        }
        if outputs.len() != 1 {
            return Err(SymbolicGraphError::InvalidOperatorOutputs);
        }
        Ok(Self {
            input: inputs[0],
            output: outputs[0],
        })
    }
}

impl OperationData for UnaryOpData {
    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input]
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
enum OperationType {
    Add(BinaryOpData),
    Sub(BinaryOpData),
    Mul(BinaryOpData),
    Div(BinaryOpData),
    MatMul(BinaryOpData),
    Relu(UnaryOpData),
    Sigmoid(UnaryOpData),
    Tanh(UnaryOpData),
}

impl OperationType {
    fn as_dyn(&self) -> &dyn OperationData {
        match self {
            OperationType::Add(op) => op,
            OperationType::Sub(op) => op,
            OperationType::Mul(op) => op,
            OperationType::Div(op) => op,
            OperationType::MatMul(op) => op,
            OperationType::Relu(op) => op,
            OperationType::Sigmoid(op) => op,
            OperationType::Tanh(op) => op,
        }
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        match self {
            OperationType::Add(op) => op.get_inputs(),
            OperationType::Sub(op) => op.get_inputs(),
            OperationType::Mul(op) => op.get_inputs(),
            OperationType::Div(op) => op.get_inputs(),
            OperationType::MatMul(op) => op.get_inputs(),
            OperationType::Relu(op) => op.get_inputs(),
            OperationType::Sigmoid(op) => op.get_inputs(),
            OperationType::Tanh(op) => op.get_inputs(),
        }
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        match self {
            OperationType::Add(op) => op.get_outputs(),
            OperationType::Sub(op) => op.get_outputs(),
            OperationType::Mul(op) => op.get_outputs(),
            OperationType::Div(op) => op.get_outputs(),
            OperationType::MatMul(op) => op.get_outputs(),
            OperationType::Relu(op) => op.get_outputs(),
            OperationType::Sigmoid(op) => op.get_outputs(),
            OperationType::Tanh(op) => op.get_outputs(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Operation {
    op: OperationType,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
enum Dimension {
    Known(usize),
    Unknown(UnknownDimensionId),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct TensorInfo {
    onnx_name: Option<String>,
    dtype: DType,
    shape: Vec<Dimension>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SymbolicGraph {
    unknown_dimensions: Vec<String>,
    tensors: Vec<TensorInfo>,
    operations: Vec<Operation>,
}

impl SymbolicGraph {
    pub fn from_onnx_bytes(onnx_bytes: &[u8]) -> Result<Self, ONNXDecodingError> {
        let model = onnx::ModelProto::decode(onnx_bytes).map_err(|x| ONNXDecodingError::ProtobufDecodeError(anyhow::Error::from(x)))?;

        Self::from_onnx_graph(&model.graph.ok_or(ONNXDecodingError::MissingField("graph"))?)
    }

    pub fn from_onnx_graph(onnx_graph: &onnx::GraphProto) -> Result<Self, ONNXDecodingError> {

        let mut tensors = vec![];
        let mut tensors_by_name: HashMap<String, TensorId> = HashMap::new();

        let mut unknown_dimensions: Vec<String> = vec![];
        let mut unknown_dimensions_by_name: HashMap<&str, UnknownDimensionId> = HashMap::new();

        for tensor in &onnx_graph.value_info {
            let onnx_tensor_type = tensor.r#type.as_ref().ok_or(ONNXDecodingError::MissingField("value_info.type"))?;
            let onnx_tensor_type_value = onnx_tensor_type
                .value
                .as_ref()
                .ok_or(ONNXDecodingError::MissingField("value_info.type.value"))?;
            let onnx_tensor_type_value_inner =
                if let onnx::type_proto::Value::TensorType(tensor_type) = onnx_tensor_type_value {
                    tensor_type
                } else {
                    return Err(ONNXDecodingError::UnsupportedTypeValue(onnx_tensor_type_value.clone()));
                };

            let onnx_shape = onnx_tensor_type_value_inner
                .shape
                .as_ref()
                .ok_or(ONNXDecodingError::MissingField("tensor_type.shape"))?;

            let dtype = DType::from_onnx(
                onnx::tensor_proto::DataType::try_from(onnx_tensor_type_value_inner.elem_type)
                    .map_err(|x| ONNXDecodingError::ProtobufDecodeError(anyhow::Error::from(x)))?,
            )?;

            let mut dimensions = vec![];
            for dim in &onnx_shape.dim {
                let value = dim.value.as_ref().ok_or(ONNXDecodingError::MissingField("dim.value"))?;
                dimensions.push(match value {
                    onnx::tensor_shape_proto::dimension::Value::DimValue(x) => Dimension::Known(if *x <= 0 {
                        return Err(ONNXDecodingError::NegativeDimensionError);
                    } else {
                        *x as usize
                    }),
                    onnx::tensor_shape_proto::dimension::Value::DimParam(x) => {
                        let unknown_dim_id =
                            if let Some(x) = unknown_dimensions_by_name.get(x.as_str()) {
                                *x
                            } else {
                                let new_id = unknown_dimensions.len();
                                unknown_dimensions_by_name.insert(x.as_str(), new_id);
                                unknown_dimensions.push(x.clone());

                                new_id
                            };
                        Dimension::Unknown(unknown_dim_id)
                    }
                });
            }

            let name = tensor.name.clone();

            let tensor_id = tensors.len();
            tensors_by_name.insert(name.clone(), tensor_id);
            tensors.push(TensorInfo {
                onnx_name: Some(name),
                dtype,
                shape: dimensions,
            })
        }

        let mut operations = vec![];

        for node in &onnx_graph.node {
            let mut input_tensors = vec![];
            for input in &node.input {
                input_tensors.push(*tensors_by_name.get(input.as_str()).ok_or(ONNXDecodingError::UnknownTensorName(input.clone()))?);
            }

            let mut output_tensors = vec![];
            for output in &node.output {
                output_tensors.push(*tensors_by_name.get(output.as_str()).ok_or(ONNXDecodingError::UnknownTensorName(output.clone()))?);
            }

            let op_type = match node.op_type.as_str() {
                "Add" => OperationType::Add(BinaryOpData::new(input_tensors, output_tensors).map_err(|x| ONNXDecodingError::GraphConstructionError(node.name.clone(), x))?),
                "Sub" => OperationType::Sub(BinaryOpData::new(input_tensors, output_tensors).map_err(|x| ONNXDecodingError::GraphConstructionError(node.name.clone(), x))?),
                "Mul" => OperationType::Mul(BinaryOpData::new(input_tensors, output_tensors).map_err(|x| ONNXDecodingError::GraphConstructionError(node.name.clone(), x))?),
                "Div" => OperationType::Div(BinaryOpData::new(input_tensors, output_tensors).map_err(|x| ONNXDecodingError::GraphConstructionError(node.name.clone(), x))?),
                "MatMul" => OperationType::MatMul(BinaryOpData::new(input_tensors, output_tensors).map_err(|x| ONNXDecodingError::GraphConstructionError(node.name.clone(), x))?),
                "Relu" => OperationType::Relu(UnaryOpData::new(input_tensors, output_tensors).map_err(|x| ONNXDecodingError::GraphConstructionError(node.name.clone(), x))?),
                "Sigmoid" => OperationType::Sigmoid(UnaryOpData::new(input_tensors, output_tensors).map_err(|x| ONNXDecodingError::GraphConstructionError(node.name.clone(), x))?),
                "Tanh" => OperationType::Tanh(UnaryOpData::new(input_tensors, output_tensors).map_err(|x| ONNXDecodingError::GraphConstructionError(node.name.clone(), x))?),
                x => Err(ONNXDecodingError::UnsupportedONNXType(x.to_string()))?,
            };

            let op = Operation{
                op: op_type
            };

            operations.push(op);
        }

        Ok(Self {
            unknown_dimensions,
            tensors,
            operations
        })
    }
}
