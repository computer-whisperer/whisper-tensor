pub mod ops;
mod milli_ops;

use std::collections::HashMap;
use prost::Message;
use serde::{Deserialize, Serialize};
use crate::dtype::DType;
use crate::ndarray_backend::{NDArrayNumericTensor, NDArrayNumericTensorError};
use crate::ndarray_backend::conversions::FromVecShape;
use crate::onnx;
use crate::symbolic_graph::ops::AnyOperation;

#[derive(Debug, thiserror::Error)]
pub enum SymbolicGraphError {
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
    #[error(transparent)]
    NDArrayNumericTensorError(#[from] NDArrayNumericTensorError),
    #[error("Unsupported ONNX: {0}")]
    UnsupportedONNX(String)
}

pub type UnknownDimensionId = usize;
pub type TensorId = usize;
pub type OperationId = usize;

fn query_attribute_float(attributes: &[onnx::AttributeProto], name: &str) -> Option<f32> {
    for attr in attributes{
        if attr.name == name {
            if attr.r#type == onnx::attribute_proto::AttributeType::Float as i32 {
                return Some(attr.f)
            }
        }
    }
    None
}

fn query_attribute_int(attributes: &[onnx::AttributeProto], name: &str) -> Option<i64> {
    for attr in attributes{
        if attr.name == name {
            if attr.r#type == onnx::attribute_proto::AttributeType::Int as i32 {
                return Some(attr.i)
            }
        }
    }
    None
}

fn query_attribute_ints(attributes: &[onnx::AttributeProto], name: &str) -> Option<Vec<i64>> {
    for attr in attributes{
        if attr.name == name {
            if attr.r#type == onnx::attribute_proto::AttributeType::Ints as i32 {
                return Some(attr.ints.clone())
            }
        }
    }
    None
}

fn query_attribute_bool(attributes: &[onnx::AttributeProto], name: &str) -> Option<bool> {
    for attr in attributes{
        if attr.name == name {
            if attr.r#type == onnx::attribute_proto::AttributeType::Int as i32 {
                return Some(attr.i != 0)
            }
        }
    }
    None
}





#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Dimension {
    Known(usize),
    Unknown(UnknownDimensionId),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
enum TensorType {
    Input,
    Output,
    Intermediate,
    Constant(NDArrayNumericTensor),
    Weight
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TensorInfo {
    onnx_name: Option<String>,
    dtype: Option<DType>,
    shape: Option<Vec<Dimension>>,
    tensor_type: TensorType
}

impl TensorInfo {
    pub fn shape(&self) -> Option<Vec<Dimension>> {
       self.shape.clone()
    }
    
    pub fn dtype(&self) -> Option<DType> {
        self.dtype
    }
    
    pub fn name(&self) -> Option<String> { self.onnx_name.clone() }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphOperation {
    pub name: Option<String>,
    pub op: AnyOperation
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SymbolicGraph {
    unknown_dimensions: HashMap<UnknownDimensionId, String>,
    tensors: HashMap<TensorId, TensorInfo>,
    operations: HashMap<OperationId, GraphOperation>
}

impl SymbolicGraph {
    pub fn new() -> Self {
        Self {
            unknown_dimensions: HashMap::new(),
            tensors: HashMap::new(),
            operations: HashMap::new()
        }
    }


    pub fn get_tensors_by_name(&self) -> HashMap<String, TensorId> {
        let mut tensors_by_name = HashMap::new();
        for (id, tensor) in &self.tensors {
            if let Some(name) = &tensor.onnx_name {
                tensors_by_name.insert(name.clone(), *id);
            }
        }
        tensors_by_name
    }

    pub fn get_unknown_dimensions_by_name(&self) -> HashMap<String, TensorId> {
        let mut unknown_dimensions_by_name = HashMap::new();
        for (id, name) in &self.unknown_dimensions {
            unknown_dimensions_by_name.insert(name.clone(), *id);
        }
        unknown_dimensions_by_name
    }

    pub fn get_operations(&self) -> &HashMap<OperationId, GraphOperation> {
        &self.operations
    }

    pub fn get_outputs(&self) -> Vec<TensorId> {
        let mut results = Vec::new();
        for (id, info) in &self.tensors {
            if let TensorType::Output = info.tensor_type {
                results.push(*id);
            }
        }
        results
    }

    pub fn get_inputs(&self) -> Vec<TensorId> {
        let mut results = Vec::new();
        for (id, info) in &self.tensors {
            if let TensorType::Input = info.tensor_type {
                results.push(*id);
            }
        }
        results
    }

    pub fn get_tensor_name(&self, tensor_id: TensorId) -> Option<&str> {
        if let Some(tensor) = self.tensors.get(&tensor_id) {
            tensor.onnx_name.as_deref()
        }
        else {
            None
        }
    }
    
    pub fn get_tensor_info(&self, tensor_id: TensorId) -> Option<&TensorInfo> {
        self.tensors.get(&tensor_id)
    }
    
    pub fn get_initialized_tensors(&self) -> HashMap<TensorId, NDArrayNumericTensor> {
        let mut out = HashMap::new();
        
        for (key, tensor) in &self.tensors {
            if let TensorType::Constant(x) = &tensor.tensor_type {
                out.insert(*key, x.clone());
            }
        }
        
        out
    }
}

impl TryFrom<&onnx::TensorProto> for NDArrayNumericTensor {
    type Error = ONNXDecodingError;

    fn try_from(tensor: &onnx::TensorProto) -> Result<Self, Self::Error> {
        let dtype = DType::try_from(
            onnx::tensor_proto::DataType::try_from(tensor.data_type)
                .map_err(|x| ONNXDecodingError::ProtobufDecodeError(anyhow::Error::from(x)))?,
        )?;
        
        let shape = tensor.dims.iter().map(|x| *x as usize).collect();

        let out = if !tensor.raw_data.is_empty() {
            NDArrayNumericTensor::from_raw_data(&tensor.raw_data, dtype, shape)?
        } 
        else if !tensor.float_data.is_empty() {
            match dtype {
                DType::F32 => {NDArrayNumericTensor::from_vec_shape(tensor.float_data.clone(), shape)?}
                _ => Err(ONNXDecodingError::UnsupportedONNX("Unsupported dtype in float_data field!".to_string()))?,
            }
        }
        else if !tensor.double_data.is_empty() {
            match dtype {
                DType::F64 => {NDArrayNumericTensor::from_vec_shape(tensor.double_data.clone(), shape)?}
                _ => Err(ONNXDecodingError::UnsupportedONNX("Unsupported dtype in double_data field!".to_string()))?,
            }        
        }
        else if !tensor.int32_data.is_empty() {
            match dtype {
                DType::I32 => NDArrayNumericTensor::from_vec_shape(tensor.int32_data.clone(), shape)?,
                DType::U16 => NDArrayNumericTensor::from_vec_shape(tensor.int32_data.iter().map(|x| *x as u16).collect::<Vec<_>>(), shape)?,
                DType::I16 => NDArrayNumericTensor::from_vec_shape(tensor.int32_data.iter().map(|x| *x as i16).collect::<Vec<_>>(), shape)?,
                DType::U8 => NDArrayNumericTensor::from_vec_shape(tensor.int32_data.iter().map(|x| *x as u8).collect::<Vec<_>>(), shape)?,
                DType::I8 => NDArrayNumericTensor::from_vec_shape(tensor.int32_data.iter().map(|x| *x as i8).collect::<Vec<_>>(), shape)?,
                _ => Err(ONNXDecodingError::UnsupportedONNX("Unsupported dtype in int32_data field!".to_string()))?
            }
        }
        else if !tensor.int64_data.is_empty() {
            match dtype {
                DType::I64 => NDArrayNumericTensor::from_vec_shape(tensor.int64_data.clone(), shape)?,
                _ => Err(ONNXDecodingError::UnsupportedONNX("Unsupported dtype in int32_data field!".to_string()))?
            }
        }
        else if !tensor.uint64_data.is_empty() {
            match dtype {
                DType::U64 => NDArrayNumericTensor::from_vec_shape(tensor.uint64_data.clone(), shape)?,
                DType::U32 => NDArrayNumericTensor::from_vec_shape(tensor.uint64_data.iter().map(|x| *x as u32).collect::<Vec<_>>(), shape)?,
                _ => Err(ONNXDecodingError::UnsupportedONNX("Unsupported dtype in int32_data field!".to_string()))?
            }
        }
        else {
            Err(ONNXDecodingError::UnsupportedONNX("No raw data field!".to_string()))?
        };
        assert_eq!(out.dtype(), dtype);
        Ok(out)
    }
}

pub struct SymbolicGraphMutator {
    graph: SymbolicGraph,
    tensors_by_name: HashMap<String, TensorId>,
    unknown_dimensions_by_name: HashMap<String, UnknownDimensionId>,
    next_tensor_id: TensorId,
    next_unknown_dimension_id: UnknownDimensionId,
    next_operation_id: OperationId,
}

impl SymbolicGraphMutator {
    pub fn new() -> Self {
        Self::from_graph(SymbolicGraph::new())
    }

    pub fn get_inner(self) -> SymbolicGraph {
        self.graph
    }

    pub fn from_graph(graph: SymbolicGraph) -> Self {
        let next_tensor_id = graph.tensors.keys().max().map(|x| x + 1).unwrap_or(0);
        let next_unknown_dimension_id = graph.unknown_dimensions.keys().max().map(|x| x + 1).unwrap_or(0);
        let next_operation_id = graph.operations.keys().max().map(|x| x + 1).unwrap_or(0);
        Self {
            tensors_by_name: graph.get_tensors_by_name(),
            graph,
            next_tensor_id,
            next_unknown_dimension_id,
            unknown_dimensions_by_name: HashMap::new(),
            next_operation_id,
        }
    }

    pub fn from_onnx_bytes(onnx_bytes: &[u8]) -> Result<Self, ONNXDecodingError> {
        let model = onnx::ModelProto::decode(onnx_bytes).map_err(|x| ONNXDecodingError::ProtobufDecodeError(anyhow::Error::from(x)))?;
        Self::from_onnx_model_proto(model)
    }

    pub fn new_unknown_tensor(&mut self, name: &str, tensor_type: TensorType) -> TensorId {
        let tensor_id = self.next_tensor_id;
        self.next_tensor_id += 1;
        let tensor = TensorInfo {
            onnx_name: Some(name.to_string()),
            tensor_type,
            dtype: None,
            shape: None
        };
        self.graph.tensors.insert(tensor_id, tensor);
        self.tensors_by_name.insert(name.to_string(), tensor_id);
        tensor_id
    }

    pub fn new_tensor_from_tensor_info(&mut self, tensor_info: &onnx::ValueInfoProto, tensor_type: TensorType) -> Result<(), ONNXDecodingError> {
        let tensor_id = self.next_tensor_id;
        self.next_tensor_id += 1;
        let onnx_tensor_type = tensor_info.r#type.as_ref().ok_or(ONNXDecodingError::MissingField("value_info.type"))?;
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
                        if let Some(x) = self.unknown_dimensions_by_name.get(x.as_str()) {
                            *x
                        } else {
                            let new_id = self.next_unknown_dimension_id;
                            self.next_unknown_dimension_id += 1;
                            self.unknown_dimensions_by_name.insert(x.clone(), new_id);
                            self.graph.unknown_dimensions.insert(new_id, x.clone());

                            new_id
                        };
                    Dimension::Unknown(unknown_dim_id)
                }
            });
        }

        let name = tensor_info.name.clone();

        self.tensors_by_name.insert(name.clone(), tensor_id);
        self.graph.tensors.insert(tensor_id,
                                  TensorInfo {
                                      onnx_name: Some(name),
                                      dtype: Some(dtype),
                                      shape: Some(dimensions),
                                      tensor_type
                                  });
        Ok(())
    }

    pub fn new_constant_tensor(&mut self, value: NDArrayNumericTensor, name: Option<String>) -> TensorId {
        let tensor_id = self.next_tensor_id;
        self.next_tensor_id += 1;

        let mut shape = Vec::new();
        for s in value.shape() {
            shape.push(Dimension::Known(*s))
        }

        self.graph.tensors.insert(
            tensor_id,
            TensorInfo {
                onnx_name: name.clone(),
                dtype: Some(value.dtype()),
                shape: Some(shape),
                tensor_type: TensorType::Constant(
                    value
                )
            }
        );
        if let Some(name) = name {
            self.tensors_by_name.insert(name, tensor_id);
        }

        tensor_id
    }

    pub fn new_node_from_onnx_node(&mut self, core_opset_version: usize, onnx_node: &onnx::NodeProto) -> Result<(), ONNXDecodingError> {
        let mut input_tensors = vec![];
        for input in &onnx_node.input {
            input_tensors.push(if let Some(tensor_id) = self.tensors_by_name.get(input) {
                *tensor_id
            } else {
                self.new_unknown_tensor(input, TensorType::Intermediate)
            });
        }

        let mut output_tensors = vec![];
        for output in &onnx_node.output {
            output_tensors.push(if let Some(tensor_id) = self.tensors_by_name.get(output) {
                *tensor_id
            } else {
                self.new_unknown_tensor(output, TensorType::Intermediate)
            });
        }
        
        let name = if onnx_node.name.is_empty() {
            None
        } else {
            Some(onnx_node.name.clone())
        };
        
        let new_op = match onnx_node.op_type.as_str() {
            "Add" => {
                Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(input_tensors, output_tensors, ops::WhichBinaryOperation::Add)
                    .map_err(|x| ONNXDecodingError::GraphConstructionError(onnx_node.name.clone(), x))?
                ))
            },
            "Sub" => {
                Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(input_tensors, output_tensors, ops::WhichBinaryOperation::Sub)
                    .map_err(|x| ONNXDecodingError::GraphConstructionError(onnx_node.name.clone(), x))?
                ))
            }
            "Mul" => {
                Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(input_tensors, output_tensors, ops::WhichBinaryOperation::Mul)
                    .map_err(|x| ONNXDecodingError::GraphConstructionError(onnx_node.name.clone(), x))?
                ))
            }
            "Div" => {
                Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(input_tensors, output_tensors, ops::WhichBinaryOperation::Div)
                    .map_err(|x| ONNXDecodingError::GraphConstructionError(onnx_node.name.clone(), x))?
                ))
            }
            "MatMul" => {
                Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(input_tensors, output_tensors, ops::WhichBinaryOperation::MatMul)
                    .map_err(|x| ONNXDecodingError::GraphConstructionError(onnx_node.name.clone(), x))?
                ))
            }
            "Relu" => {
                Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(input_tensors, output_tensors, ops::WhichUnaryOperation::Relu)
                    .map_err(|x| ONNXDecodingError::GraphConstructionError(onnx_node.name.clone(), x))?
                ))
            }
            "Sigmoid" => {
                Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(input_tensors, output_tensors, ops::WhichUnaryOperation::Sigmoid)
                    .map_err(|x| ONNXDecodingError::GraphConstructionError(onnx_node.name.clone(), x))?
                ))
            }
            "Tanh" => {
                Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(input_tensors, output_tensors, ops::WhichUnaryOperation::Tanh)
                    .map_err(|x| ONNXDecodingError::GraphConstructionError(onnx_node.name.clone(), x))?
                ))
            }
            "Exp" => {
                Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(input_tensors, output_tensors, ops::WhichUnaryOperation::Exp)
                    .map_err(|x| ONNXDecodingError::GraphConstructionError(onnx_node.name.clone(), x))?
                ))
            }
            "Sqrt" => {
                Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(input_tensors, output_tensors, ops::WhichUnaryOperation::Sqrt)
                    .map_err(|x| ONNXDecodingError::GraphConstructionError(onnx_node.name.clone(), x))?
                ))
            }
            "Softplus" => {
                Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(input_tensors, output_tensors, ops::WhichUnaryOperation::Softplus)
                    .map_err(|x| ONNXDecodingError::GraphConstructionError(onnx_node.name.clone(), x))?
                ))
            }
            "Neg" => {
                Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(input_tensors, output_tensors, ops::WhichUnaryOperation::Neg)
                    .map_err(|x| ONNXDecodingError::GraphConstructionError(onnx_node.name.clone(), x))?
                ))
            }
            "NonZero" => {
                Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(input_tensors, output_tensors, ops::WhichUnaryOperation::NonZero)
                    .map_err(|x| ONNXDecodingError::GraphConstructionError(onnx_node.name.clone(), x))?
                ))
            }
            "LpNormalization" => {
                Some(AnyOperation::LpNormalization(ops::LpNormalizationOperation::from_onnx(input_tensors, output_tensors, &onnx_node.attribute)
                    .map_err(|x| ONNXDecodingError::GraphConstructionError(onnx_node.name.clone(), x))?
                ))
            }
            "GroupNormalization" => {
                Some(AnyOperation::GroupNormalization(ops::GroupNormalizationOperation::from_onnx(input_tensors, output_tensors, &onnx_node.attribute)?))
            }
            "LayerNormalization" => {
                Some(AnyOperation::LayerNormalization(ops::LayerNormalizationOperation::from_onnx(&input_tensors, &output_tensors, &onnx_node.attribute)?))
            }
            "CumSum" => {
                Some(AnyOperation::CumSum(ops::CumSumOperation::from_onnx(input_tensors, output_tensors)
                    .map_err(|x| ONNXDecodingError::GraphConstructionError(onnx_node.name.clone(), x))?
                ))
            }
            "Gather" => {
                Some(AnyOperation::Gather(ops::GatherOperation::from_onnx(&input_tensors, &output_tensors, &onnx_node.attribute)?))
            }
            "Cast" => {
                Some(AnyOperation::Cast(ops::CastOperation::from_onnx(input_tensors, output_tensors, &onnx_node.attribute)?))
            }
            "Pow" => {
                Some(AnyOperation::Pow(ops::PowOperation::from_onnx(&input_tensors, &output_tensors, &onnx_node.attribute)?))
            }
            "ReduceMean" => {
                if core_opset_version >= 18 {
                    Some(AnyOperation::ReduceMean(ops::ReduceMeanOperation::from_onnx(&input_tensors, &output_tensors, &onnx_node.attribute)?))
                } else {
                    // Adapt for older opset versions without "axes" input
                    let mut axes = Vec::new();
                    for attr in &onnx_node.attribute {
                        if attr.name == "axes" {
                            axes = attr.ints.clone()
                        }
                    }
                    let axes_tensor_id = self.new_constant_tensor(NDArrayNumericTensor::from(axes), None);
                    input_tensors.push(axes_tensor_id);
                    Some(AnyOperation::ReduceMean(ops::ReduceMeanOperation::from_onnx(&input_tensors, &output_tensors, &onnx_node.attribute)?))
                }
            }
            "ReduceSum" => {
                if core_opset_version >= 18 {
                    Some(AnyOperation::ReduceSum(ops::ReduceSumOperation::from_onnx(&input_tensors, &output_tensors, &onnx_node.attribute)?))
                } else {
                    // Adapt for older opset versions without "axes" input
                    let mut axes = Vec::new();
                    for attr in &onnx_node.attribute {
                        if attr.name == "axes" {
                            axes = attr.ints.clone()
                        }
                    }
                    let axes_tensor_id = self.new_constant_tensor(NDArrayNumericTensor::from(axes), None);
                    input_tensors.push(axes_tensor_id);
                    Some(AnyOperation::ReduceSum(ops::ReduceSumOperation::from_onnx(&input_tensors, &output_tensors, &onnx_node.attribute)?))
                }
            }
            "Squeeze" => {
                if core_opset_version >= 21 {
                    Some(AnyOperation::Squeeze(ops::SqueezeOperation::from_onnx(input_tensors, output_tensors)
                        .map_err(|x| ONNXDecodingError::GraphConstructionError(onnx_node.name.clone(), x))?
                    ))
                }
                else {
                    // Adapt for older opset versions without "axes" input
                    let mut axes = Vec::new();
                    for attr in &onnx_node.attribute {
                        if attr.name == "axes" {
                            axes = attr.ints.clone()
                        }
                    }
                    let axes_tensor_id = self.new_constant_tensor(NDArrayNumericTensor::from(axes), None);
                    input_tensors.push(axes_tensor_id);
                    Some(AnyOperation::Squeeze(ops::SqueezeOperation::from_onnx(input_tensors, output_tensors)
                        .map_err(|x| ONNXDecodingError::GraphConstructionError(onnx_node.name.clone(), x))?
                    ))
                }
            }
            "Unsqueeze" => {
                if core_opset_version >= 21 {
                    Some(AnyOperation::Unsqueeze(ops::UnsqueezeOperation::from_onnx(input_tensors, output_tensors)
                        .map_err(|x| ONNXDecodingError::GraphConstructionError(onnx_node.name.clone(), x))?
                    ))
                }
                else {
                    // Adapt for older opset version without axes tensor
                    let mut axes = Vec::new();
                    for attr in &onnx_node.attribute {
                        if attr.name == "axes" {
                            axes = attr.ints.clone()
                        }
                    }
                    let axes_tensor_id = self.new_constant_tensor(NDArrayNumericTensor::from(axes), None);
                    input_tensors.push(axes_tensor_id);
                    Some(AnyOperation::Unsqueeze(ops::UnsqueezeOperation::from_onnx(input_tensors, output_tensors)
                        .map_err(|x| ONNXDecodingError::GraphConstructionError(onnx_node.name.clone(), x))?
                    ))
                }
            }
            "Transpose" => {
                Some(AnyOperation::Transpose(ops::TransposeOperation::from_onnx(input_tensors, output_tensors, &onnx_node.attribute)
                    .map_err(|x| ONNXDecodingError::GraphConstructionError(onnx_node.name.clone(), x))?
                ))
            }
            "Reshape" => {
                Some(AnyOperation::Reshape(ops::ReshapeOperation::from_onnx(input_tensors, output_tensors, &onnx_node.attribute)
                    .map_err(|x| ONNXDecodingError::GraphConstructionError(onnx_node.name.clone(), x))?
                ))
            }
            "Shape" => {
                Some(AnyOperation::Shape(ops::ShapeOperation::from_onnx(&input_tensors, &output_tensors, &onnx_node.attribute)?))
            }
            "Concat" => {
                Some(AnyOperation::Concat(ops::ConcatOperation::from_onnx(&input_tensors, &output_tensors, &onnx_node.attribute)?))
            }
            "ConstantOfShape" => {
                Some(AnyOperation::ConstantOfShape(ops::ConstantOfShapeOperation::from_onnx(&input_tensors, &output_tensors, &onnx_node.attribute)?))
            }
            "Gemm" => {
                Some(AnyOperation::Gemm(ops::GemmOperation::from_onnx(&input_tensors, &output_tensors, &onnx_node.attribute)?))
            }
            "Slice" => {
                Some(AnyOperation::Slice(ops::SliceOperation::from_onnx(&input_tensors, &output_tensors, &onnx_node.attribute)?))
            }
            "Where" => {
                Some(AnyOperation::Where(ops::WhereOperation::from_onnx(&input_tensors, &output_tensors, &onnx_node.attribute)?))
            }
            "Softmax" => {
                Some(AnyOperation::Softmax(ops::SoftmaxOperation::from_onnx(&input_tensors, &output_tensors, &onnx_node.attribute)?))
            }
            "Split" => {
                Some(AnyOperation::Split(ops::SplitOperation::from_onnx(&input_tensors, &output_tensors, &onnx_node.attribute)?))
            }
            "Constant" => {
                let tensor =  {
                    let mut tensor: Option<onnx::TensorProto> = None;
                    for att in &onnx_node.attribute {
                        if att.name == "value" && att.r#type == onnx::attribute_proto::AttributeType::Tensor as i32 {
                            if let Some(t) = &att.t {
                                tensor = Some(t.clone());
                            }
                        }
                    }
                    if let Some(tensor) = tensor {
                        tensor
                    } else {
                        Err(ONNXDecodingError::MissingAttribute("value".to_string(), onnx_node.op_type.clone()))?
                    }
                };

                let numeric_tensor = NDArrayNumericTensor::try_from(&tensor)?;
                self.graph.tensors.get_mut(&output_tensors[0]).unwrap().tensor_type = TensorType::Constant(numeric_tensor);

                None
            }
            x => Err(ONNXDecodingError::UnsupportedONNXType(x.to_string()))?
        };

        
        if let Some(op) = new_op {
            
            let op = GraphOperation {
                name,
                op
            };
            
            let operation_id = self.next_operation_id;
            self.next_operation_id += 1;
            self.graph.operations.insert(operation_id, op);
        }

        Ok(())
    }

    pub fn from_onnx_model_proto(model_proto: onnx::ModelProto) -> Result<Self, ONNXDecodingError> {
        let mut core_opset_version = 0;
        for opset_proto in model_proto.opset_import {
            if opset_proto.domain.is_empty() {
                core_opset_version = opset_proto.version as usize;
            }
        }

        let onnx_graph = model_proto.graph.ok_or(ONNXDecodingError::MissingField("graph"))?;
        let mut graph_mutator = Self::new();

        for t in onnx_graph.input.iter() {
            graph_mutator.new_tensor_from_tensor_info(t, TensorType::Input)?;
        }
        for t in onnx_graph.output.iter() {
            graph_mutator.new_tensor_from_tensor_info(t, TensorType::Output)?;
        }
        for t in onnx_graph.value_info.iter() {
            graph_mutator.new_tensor_from_tensor_info(t, TensorType::Intermediate)?;
        }

        for t in onnx_graph.initializer {
            let numeric_tensor = NDArrayNumericTensor::try_from(&t)?;
            if let Some(x) = graph_mutator.tensors_by_name.get(&t.name) {
                graph_mutator.graph.tensors.get_mut(x).unwrap().tensor_type = TensorType::Constant(numeric_tensor);
            } else {
                graph_mutator.new_constant_tensor(numeric_tensor.clone(), Some(t.name));
            }
        }
        
        for node in &onnx_graph.node {
            graph_mutator.new_node_from_onnx_node(core_opset_version, node)?;
        }
        


        Ok(graph_mutator)
    }
}