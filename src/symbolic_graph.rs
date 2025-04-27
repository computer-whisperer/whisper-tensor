use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::HashMap;
use prost::Message;
use crate::dtype::DType;
use crate::native_numeric_tensor::NativeNumericTensor;
use crate::onnx;

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
    UnsupportedDType(onnx::tensor_proto::DataType),
    #[error("Unsupported ONNX type {0}")]
    UnsupportedONNXType(String),
    #[error("Negative dimension")]
    NegativeDimensionError,
    #[error("Unknown tensor name \"{0}\"")]
    UnknownTensorName(String),
    #[error("Missing expected attribute \"{1}\" for op {0}")]
    MissingAttribute(String, String),
    #[error(transparent)]
    DTypeError(#[from] crate::dtype::DTypeError),
    #[error("Unsupported ONNX: {0}")]
    UnsupportedONNX(String)
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
    fn from_onnx(inputs: Vec<TensorId>, outputs: Vec<TensorId>) -> Result<Self, SymbolicGraphError> {
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
    fn from_onnx(inputs: Vec<TensorId>, outputs: Vec<TensorId>) -> Result<Self, SymbolicGraphError> {
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
struct CumSumData {
    input: TensorId,
    output: TensorId,
    axis: TensorId,
}

impl CumSumData {
    fn from_onnx(inputs: Vec<TensorId>, outputs: Vec<TensorId>) -> Result<Self, SymbolicGraphError> {
        if inputs.len() != 2 {
            return Err(SymbolicGraphError::InvalidOperatorInputs);
        }
        if outputs.len() != 1 {
            return Err(SymbolicGraphError::InvalidOperatorOutputs);
        }
        Ok(Self {
            input: inputs[0],
            axis: inputs[1],
            output: outputs[0],
        })
    }
}
impl OperationData for CumSumData {
    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input, self.axis]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct LpNormalizationData {
    input: TensorId,
    output: TensorId,
    axis: i64,
    p: i64,
}

impl LpNormalizationData {
    pub fn from_onnx(inputs: Vec<TensorId>, outputs: Vec<TensorId>, attributes: &[onnx::AttributeProto]) -> Result<Self, SymbolicGraphError> {
        if inputs.len() != 1 {
            return Err(SymbolicGraphError::InvalidOperatorInputs);
        }
        if outputs.len() != 1 {
            return Err(SymbolicGraphError::InvalidOperatorOutputs);
        }
        let mut axis = -1;
        let mut p = 2;
        for attr in attributes{
            match attr.name.as_str() {
                "axis" => axis = attr.i,
                "p" => p = attr.i,
                _ => {}
            }
        }
        Ok(Self {
            input: inputs[0],
            output: outputs[0],
            axis,
            p,
        })
    }
}

impl OperationData for LpNormalizationData {
    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct GroupNormalizationData {
    input: TensorId,
    scale: TensorId,
    bias: TensorId,
    output: TensorId,
    epsilon: f32,
    num_groups: i64
}

impl GroupNormalizationData {
    pub fn from_onnx(inputs: Vec<TensorId>, outputs: Vec<TensorId>, attributes: &[onnx::AttributeProto]) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 3 {
            return Err(ONNXDecodingError::GraphConstructionError("GroupNormalization".to_string(), SymbolicGraphError::InvalidOperatorInputs));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::GraphConstructionError("GroupNormalization".to_string(), SymbolicGraphError::InvalidOperatorOutputs));
        }
        let mut epsilon = 1e-5;
        let mut num_groups = None;
        for attr in attributes{
            match attr.name.as_str() {
                "epsilon" => epsilon = attr.f,
                "num_groups" => num_groups = Some(attr.i),
                _ => {}
            }
        }
        let num_groups = num_groups.ok_or(ONNXDecodingError::MissingAttribute("GroupNormalization".to_string(), "num_groups".to_string()))?;
        Ok(Self {
            input: inputs[0],
            scale: inputs[1],
            bias: inputs[2],
            output: outputs[0],
            epsilon,
            num_groups
        })
    }
}

impl OperationData for GroupNormalizationData {
    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input, self.scale, self.bias]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct SqueezeData {
    input: TensorId,
    axes: TensorId,
    output: TensorId,
}

impl SqueezeData {
    pub fn from_onnx(inputs: Vec<TensorId>, outputs: Vec<TensorId>) -> Result<Self, SymbolicGraphError> {
        if inputs.len() != 2 {
            return Err(SymbolicGraphError::InvalidOperatorInputs);
        }
        if outputs.len() != 1 {
            return Err(SymbolicGraphError::InvalidOperatorOutputs);
        }
        Ok(Self {
            input: inputs[0],
            axes: inputs[1],
            output: outputs[0],
        })
    }
}

impl OperationData for SqueezeData {
    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input, self.axes]
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct TransposeData {
    input: TensorId,
    output: TensorId,
    perm: Vec<i64>,
}

impl TransposeData {
    pub fn from_onnx(inputs: Vec<TensorId>, outputs: Vec<TensorId>, attributes: &[onnx::AttributeProto]) -> Result<Self, SymbolicGraphError> {
        if inputs.len() != 1 {
            return Err(SymbolicGraphError::InvalidOperatorInputs);
        }
        if outputs.len() != 1 {
            return Err(SymbolicGraphError::InvalidOperatorOutputs);
        }
        let mut perm = vec![];
        for attr in attributes{
            match attr.name.as_str() {
                "perm" => {perm = attr.ints.clone()},
                _ => {}
            }
        }
        Ok(Self {
            input: inputs[0],
            output: outputs[0],
            perm
        })
    }
}

impl OperationData for TransposeData {
    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct ReshapeData {
    input: TensorId,
    shape: TensorId,
    output: TensorId,
}

impl ReshapeData {
    pub fn from_onnx(inputs: Vec<TensorId>, outputs: Vec<TensorId>, _attributes: &[onnx::AttributeProto]) -> Result<Self, SymbolicGraphError> {
        if inputs.len() != 2 {
            return Err(SymbolicGraphError::InvalidOperatorInputs);
        }
        if outputs.len() != 1 {
            return Err(SymbolicGraphError::InvalidOperatorOutputs);
        }
        Ok(Self {
            input: inputs[0],
            shape: inputs[1],
            output: outputs[0],
        })
    }
}

impl OperationData for ReshapeData {
    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input, self.shape]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct CastData {
    input: TensorId,
    output: TensorId,
    to: DType,
}

impl CastData {
    pub fn from_onnx(inputs: Vec<TensorId>, outputs: Vec<TensorId>, attributes: &[onnx::AttributeProto]) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::GraphConstructionError("Cast".to_string(), SymbolicGraphError::InvalidOperatorInputs));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::GraphConstructionError("Cast".to_string(), SymbolicGraphError::InvalidOperatorOutputs));
        }
        let to_i = attributes.iter().find(|a| a.name == "to")
            .ok_or(ONNXDecodingError::MissingAttribute("Cast".to_string(), "to".to_string()))?.i as i32;
        let to_datatype = onnx::tensor_proto::DataType::try_from(to_i).map_err(|x| ONNXDecodingError::ProtobufDecodeError(x.into()))?;
        let to = DType::try_from(to_datatype)?;
        Ok(Self {
            input: inputs[0],
            output: outputs[0],
            to,
        })
    }
}

impl OperationData for CastData {
    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct LayerNormalizationData {
    input: TensorId,
    scale: TensorId,
    bias: Option<TensorId>,
    output: TensorId,
    axis: i64,
    epsilon: f32
}

impl LayerNormalizationData {
    fn from_onnx(inputs: &[TensorId], outputs: &[TensorId], attributes: &[onnx::AttributeProto]) -> Result<Self, ONNXDecodingError> {
        if inputs.len() < 2 || inputs.len() > 3  {
            return Err(ONNXDecodingError::GraphConstructionError("LayerNormalization".to_string(), SymbolicGraphError::InvalidOperatorInputs));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::GraphConstructionError("LayerNormalization".to_string(), SymbolicGraphError::InvalidOperatorOutputs));
        }
        let mut axis = -1;
        let mut epsilon = 1e-5;
        for attribute in attributes {
            match attribute.name.as_str() {
                "axis" => {
                    axis = attribute.i;
                },
                "epsilon" => {
                    epsilon = attribute.f;
                },
                _ => {}
            }
        }
        Ok(Self {
            input: inputs[0],
            scale: inputs[1],
            bias: if inputs.len() == 3 { Some(inputs[2]) } else { None },
            output: outputs[0],
            axis,
            epsilon
        })
    }
}

impl OperationData for LayerNormalizationData {
    fn get_inputs(&self) -> Vec<TensorId> {
        let mut v = vec![self.input, self.scale];
        if let Some(bias) = self.bias {
            v.push(bias);
        }
        v
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct GatherData {
    input: TensorId,
    indices: TensorId,
    output: TensorId,
    axis: i64
}

impl GatherData {
    fn from_onnx(inputs: &[TensorId], outputs: &[TensorId], attributes: &[onnx::AttributeProto]) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 2 {
            return Err(ONNXDecodingError::GraphConstructionError("Gather".to_string(), SymbolicGraphError::InvalidOperatorInputs));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::GraphConstructionError("Gather".to_string(), SymbolicGraphError::InvalidOperatorOutputs));
        }
        let mut axis = 0;
        for attribute in attributes {
            match attribute.name.as_str() {
                "axis" => {
                    axis = attribute.i;
                },
                _ => {}
            }
        }
        Ok(Self {
            input: inputs[0],
            indices: inputs[1],
            output: outputs[0],
            axis
        })
    }
}

impl OperationData for GatherData {
    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input, self.indices]
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
    Exp(UnaryOpData),
    Neg(UnaryOpData),
    Softplus(UnaryOpData),
    Cast(CastData),
    Squeeze(SqueezeData),
    Unsqueeze(SqueezeData),
    Transpose(TransposeData),
    Reshape(ReshapeData),
    CumSum(CumSumData),
    Gather(GatherData),
    LpNormalization(LpNormalizationData),
    GroupNormalization(GroupNormalizationData),
    LayerNormalization(LayerNormalizationData)
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
            OperationType::Exp(op) => op,
            OperationType::Neg(op) => op,
            OperationType::Cast(op) => op,
            OperationType::Softplus(op) => op,
            OperationType::Squeeze(op) => op,
            OperationType::Unsqueeze(op) => op,
            OperationType::Transpose(op) => op,
            OperationType::Reshape(op) => op,
            OperationType::CumSum(op) => op,
            OperationType::Gather(op) => op,
            OperationType::LpNormalization(op) => op,
            OperationType::GroupNormalization(op) => op,
            OperationType::LayerNormalization(op) => op,
        }
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        self.as_dyn().get_inputs()
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        self.as_dyn().get_outputs()
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
enum TensorType {
    Input,
    Output,
    Intermediate,
    Constant(NativeNumericTensor),
    Weight
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct TensorInfo {
    onnx_name: Option<String>,
    dtype: DType,
    shape: Vec<Dimension>,
    tensor_type: TensorType
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

        let all_tensor_infos = {
            let mut v = vec![];
            v.extend(onnx_graph.input.iter());
            v.extend(onnx_graph.output.iter());
            v.extend(onnx_graph.value_info.iter());
            v
        };

        for tensor in all_tensor_infos.into_iter() {
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

            let dtype = DType::try_from(
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
            let tensor_type = {
                let mut v = TensorType::Intermediate;
                if onnx_graph.input.iter().any(|x| x.name == name) {
                    v = TensorType::Input;
                }
                if onnx_graph.output.iter().any(|x| x.name == name) {
                    v = TensorType::Output;
                }
                v
            };

            let tensor_id = tensors.len();
            tensors_by_name.insert(name.clone(), tensor_id);
            tensors.push(TensorInfo {
                onnx_name: Some(name),
                dtype,
                shape: dimensions,
                tensor_type
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

            match node.op_type.as_str() {
                "Add" => {
                    operations.push(
                        Operation{
                            op: OperationType::Add(BinaryOpData::from_onnx(input_tensors, output_tensors)
                                .map_err(|x| ONNXDecodingError::GraphConstructionError(node.name.clone(), x))?
                            )});
                },
                "Sub" => {
                    operations.push(
                        Operation{
                            op: OperationType::Sub(BinaryOpData::from_onnx(input_tensors, output_tensors)
                                .map_err(|x| ONNXDecodingError::GraphConstructionError(node.name.clone(), x))?
                            )});
                }
                "Mul" => {
                    operations.push(
                        Operation{
                            op: OperationType::Mul(BinaryOpData::from_onnx(input_tensors, output_tensors)
                                .map_err(|x| ONNXDecodingError::GraphConstructionError(node.name.clone(), x))?
                            )});
                }
                "Div" => {
                    operations.push(
                        Operation{
                            op: OperationType::Div(BinaryOpData::from_onnx(input_tensors, output_tensors)
                                .map_err(|x| ONNXDecodingError::GraphConstructionError(node.name.clone(), x))?
                            )});
                }
                "MatMul" => {
                    operations.push(
                        Operation{
                            op: OperationType::MatMul(BinaryOpData::from_onnx(input_tensors, output_tensors)
                                .map_err(|x| ONNXDecodingError::GraphConstructionError(node.name.clone(), x))?
                            )});
                }
                "Relu" => {
                    operations.push(
                        Operation{
                            op: OperationType::Relu(UnaryOpData::from_onnx(input_tensors, output_tensors)
                                .map_err(|x| ONNXDecodingError::GraphConstructionError(node.name.clone(), x))?
                            )});
                }
                "Sigmoid" => {
                    operations.push(
                        Operation{
                            op: OperationType::Sigmoid(UnaryOpData::from_onnx(input_tensors, output_tensors)
                                .map_err(|x| ONNXDecodingError::GraphConstructionError(node.name.clone(), x))?
                            )});
                }
                "Tanh" => {
                    operations.push(
                        Operation{
                            op: OperationType::Tanh(UnaryOpData::from_onnx(input_tensors, output_tensors)
                                .map_err(|x| ONNXDecodingError::GraphConstructionError(node.name.clone(), x))?
                            )});
                }
                "Exp" => {
                    operations.push(
                        Operation{
                            op: OperationType::Exp(UnaryOpData::from_onnx(input_tensors, output_tensors)
                                .map_err(|x| ONNXDecodingError::GraphConstructionError(node.name.clone(), x))?
                            )});
                }
                "Softplus" => {
                    operations.push(
                        Operation{
                            op: OperationType::Softplus(UnaryOpData::from_onnx(input_tensors, output_tensors)
                                .map_err(|x| ONNXDecodingError::GraphConstructionError(node.name.clone(), x))?
                            )});
                }
                "Neg" => {
                    operations.push(
                        Operation{
                            op: OperationType::Neg(UnaryOpData::from_onnx(input_tensors, output_tensors)
                                .map_err(|x| ONNXDecodingError::GraphConstructionError(node.name.clone(), x))?
                            )});
                }
                "LpNormalization" => {
                    operations.push(
                        Operation{
                            op: OperationType::LpNormalization(LpNormalizationData::from_onnx(input_tensors, output_tensors, &node.attribute)
                                .map_err(|x| ONNXDecodingError::GraphConstructionError(node.name.clone(), x))?
                            )});
                }
                "GroupNormalization" => {
                    operations.push(
                        Operation{
                            op: OperationType::GroupNormalization(GroupNormalizationData::from_onnx(input_tensors, output_tensors, &node.attribute)?
                            )});
                }
                "LayerNormalization" => {
                    operations.push(
                        Operation{
                            op: OperationType::LayerNormalization(LayerNormalizationData::from_onnx(&input_tensors, &output_tensors, &node.attribute)?)
                        }
                    );
                }
                "CumSum" => {
                    operations.push(
                        Operation{
                            op: OperationType::CumSum(CumSumData::from_onnx(input_tensors, output_tensors)
                                .map_err(|x| ONNXDecodingError::GraphConstructionError(node.name.clone(), x))?
                            )});
                }
                "Gather" => {
                    operations.push(
                        Operation{
                            op: OperationType::Gather(GatherData::from_onnx(&input_tensors, &output_tensors, &node.attribute)?
                            )});
                }
                "Cast" => {
                    operations.push(
                        Operation{
                            op: OperationType::Cast(CastData::from_onnx(input_tensors, output_tensors, &node.attribute)?)
                            });
                }
                "Squeeze" => {
                    operations.push(
                        Operation{
                            op: OperationType::Squeeze(SqueezeData::from_onnx(input_tensors, output_tensors)
                                .map_err(|x| ONNXDecodingError::GraphConstructionError(node.name.clone(), x))?
                            )});
                }
                "Unsqueeze" => {
                    operations.push(
                        Operation{
                            op: OperationType::Unsqueeze(SqueezeData::from_onnx(input_tensors, output_tensors)
                                .map_err(|x| ONNXDecodingError::GraphConstructionError(node.name.clone(), x))?
                            )});
                }
                "Transpose" => {
                    operations.push(
                        Operation{
                            op: OperationType::Transpose(TransposeData::from_onnx(input_tensors, output_tensors, &node.attribute)
                                .map_err(|x| ONNXDecodingError::GraphConstructionError(node.name.clone(), x))?
                            )});
                }
                "Reshape" => {
                    operations.push(
                        Operation{
                            op: OperationType::Reshape(ReshapeData::from_onnx(input_tensors, output_tensors, &node.attribute)
                                .map_err(|x| ONNXDecodingError::GraphConstructionError(node.name.clone(), x))?
                            )});
                }
                "Constant" => {
                    let tensor =  {
                        let mut tensor: Option<onnx::TensorProto> = None;
                        for att in &node.attribute {
                            if att.name == "value" && att.r#type == onnx::attribute_proto::AttributeType::Tensor as i32 {
                                if let Some(t) = &att.t {
                                    tensor = Some(t.clone());
                                }
                            }
                        }
                        if let Some(tensor) = tensor {
                            tensor
                        } else {
                            Err(ONNXDecodingError::MissingAttribute("value".to_string(), node.op_type.clone()))?
                        }
                    };

                    let dtype = DType::try_from(
                        onnx::tensor_proto::DataType::try_from(tensor.data_type)
                            .map_err(|x| ONNXDecodingError::ProtobufDecodeError(anyhow::Error::from(x)))?,
                    )?;

                    let constant_value = if tensor.raw_data.len() > 0 {
                        NativeNumericTensor::from_raw_data(&tensor.raw_data, dtype, tensor.dims.iter().map(|x| *x as usize).collect())
                    } else {
                        Err(ONNXDecodingError::UnsupportedONNX("No raw data field!".to_string()))?
                    };

                    tensors[output_tensors[0]].tensor_type = TensorType::Constant(
                        constant_value
                    );
                }
                x => Err(ONNXDecodingError::UnsupportedONNXType(x.to_string()))?
            }
        }

        Ok(Self {
            unknown_dimensions,
            tensors,
            operations
        })
    }
}
