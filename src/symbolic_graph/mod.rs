pub mod ops;
pub mod tensor_store;

use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::{NDArrayNumericTensor, NDArrayNumericTensorError};
use crate::dtype::DType;
use crate::numeric_scalar::NumericScalar;
use crate::numeric_tensor::NumericTensor;
use crate::scalar_info::ScalarInfoTyped;
use crate::symbolic_graph::ops::{AnyOperation, EvalError, Operation};
use crate::symbolic_graph::tensor_store::{StoredTensor, TensorStore, TensorStoreTensorId};
use crate::symbolic_scalar::{SymbolicResolver, SymbolicScalar, SymbolicScalarTyped};
use crate::tensor_rank::DynRank;
use crate::{TrigOp, onnx};
use prost::Message;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, thiserror::Error)]
pub enum ONNXDecodingError {
    #[error("Invalid operator inputs")]
    InvalidOperatorInputs(&'static str),
    #[error("Invalid operator outputs")]
    InvalidOperatorOutputs(&'static str),
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
    UnsupportedONNX(String),
}

pub type SymbolicGraphTensorId = usize;
pub type SymbolicGraphOperationId = usize;

fn query_attribute_float(attributes: &[onnx::AttributeProto], name: &str) -> Option<f32> {
    for attr in attributes {
        if attr.name == name && attr.r#type == onnx::attribute_proto::AttributeType::Float as i32 {
            return Some(attr.f);
        }
    }
    None
}

fn query_attribute_floats(attributes: &[onnx::AttributeProto], name: &str) -> Option<Vec<f32>> {
    for attr in attributes {
        if attr.name == name && attr.r#type == onnx::attribute_proto::AttributeType::Floats as i32 {
            return Some(attr.floats.clone());
        }
    }
    None
}

fn query_attribute_int(attributes: &[onnx::AttributeProto], name: &str) -> Option<i64> {
    for attr in attributes {
        if attr.name == name && attr.r#type == onnx::attribute_proto::AttributeType::Int as i32 {
            return Some(attr.i);
        }
    }
    None
}

fn query_attribute_ints(attributes: &[onnx::AttributeProto], name: &str) -> Option<Vec<i64>> {
    for attr in attributes {
        if attr.name == name && attr.r#type == onnx::attribute_proto::AttributeType::Ints as i32 {
            return Some(attr.ints.clone());
        }
    }
    None
}

fn query_attribute_string(attributes: &[onnx::AttributeProto], name: &str) -> Option<String> {
    for attr in attributes {
        if attr.name == name && attr.r#type == onnx::attribute_proto::AttributeType::String as i32 {
            return Some(String::from_utf8(attr.s.clone()).unwrap());
        }
    }
    None
}

fn query_attribute_bool(attributes: &[onnx::AttributeProto], name: &str) -> Option<bool> {
    for attr in attributes {
        if attr.name == name && attr.r#type == onnx::attribute_proto::AttributeType::Int as i32 {
            return Some(attr.i != 0);
        }
    }
    None
}

fn query_attribute_tensor(
    attributes: &[onnx::AttributeProto],
    name: &str,
) -> Option<NDArrayNumericTensor<DynRank>> {
    for attr in attributes {
        if attr.name == name
            && attr.r#type == onnx::attribute_proto::AttributeType::Tensor as i32
            && let Some(tensor_proto) = &attr.t
        {
            return NDArrayNumericTensor::try_from(tensor_proto).ok();
        }
    }
    None
}

fn query_attribute_graph<'a>(
    attributes: &'a [onnx::AttributeProto],
    name: &str,
) -> Option<&'a onnx::GraphProto> {
    for attr in attributes {
        if attr.name == name && attr.r#type == onnx::attribute_proto::AttributeType::Graph as i32 {
            return attr.g.as_ref();
        }
    }
    None
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum StoredOrNotTensor {
    Stored(TensorStoreTensorId),
    NotStored(NDArrayNumericTensor<DynRank>),
}

impl StoredOrNotTensor {
    pub fn get_tensor(&self, tensor_store: &TensorStore) -> NumericTensor<DynRank> {
        match self {
            StoredOrNotTensor::Stored(id) => tensor_store.get_tensor(*id).unwrap().to_numeric(),
            StoredOrNotTensor::NotStored(tensor) => NumericTensor::NDArray(tensor.clone()),
        }
    }
    pub fn shape(&self, tensor_store: &TensorStore) -> Vec<u64> {
        match self {
            StoredOrNotTensor::Stored(id) => tensor_store.get_tensor(*id).unwrap().shape(),
            StoredOrNotTensor::NotStored(tensor) => tensor.shape(),
        }
    }

    pub fn dtype(&self, tensor_store: &TensorStore) -> DType {
        match self {
            StoredOrNotTensor::Stored(id) => tensor_store.get_tensor(*id).unwrap().dtype(),
            StoredOrNotTensor::NotStored(tensor) => tensor.dtype(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TensorType {
    Input(Option<StoredOrNotTensor>),
    Output,
    Intermediate,
    Constant(StoredOrNotTensor),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ONNXTensorInfo {
    pub onnx_name: Option<String>,
    pub dtype: Option<DType>,
    pub shape: Option<Vec<ScalarInfoTyped<u64>>>,
    pub tensor_type: TensorType,
}

impl ONNXTensorInfo {
    pub fn shape(&self) -> Option<Vec<ScalarInfoTyped<u64>>> {
        self.shape.clone()
    }

    pub fn dtype(&self) -> Option<DType> {
        self.dtype
    }

    pub fn name(&self) -> Option<String> {
        self.onnx_name.clone()
    }
}

pub fn check_tensor_matches(
    tensor: &NumericTensor<DynRank>,
    tensor_info: &ONNXTensorInfo,
) -> Result<(), EvalError> {
    if let Some(shape) = tensor_info.shape() {
        if shape.len() != tensor.shape().len() {
            Err(EvalError::UnexpectedRank(shape.len(), tensor.shape().len()))?;
        }
        for (a, b) in shape.iter().zip(tensor.shape()) {
            if let ScalarInfoTyped::Numeric(a) = a
                && *a != b
            {
                Err(EvalError::UnexpectedDimension(*a, b, tensor.shape()))?
            }
        }
    }
    if let Some(dtype) = tensor_info.dtype() {
        let tensor_dtype = tensor.dtype();
        if dtype != tensor_dtype {
            Err(EvalError::UnexpectedDType(dtype, tensor_dtype))?
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphOperation {
    pub name: Option<String>,
    pub op: AnyOperation,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SymbolicGraphInner {
    tensors: HashMap<SymbolicGraphTensorId, ONNXTensorInfo>,
    ordered_outputs: Vec<SymbolicGraphTensorId>,
    ordered_inputs: Vec<SymbolicGraphTensorId>,
    operations: HashMap<SymbolicGraphOperationId, GraphOperation>,
}

impl SymbolicGraphInner {
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            ordered_outputs: Vec::new(),
            ordered_inputs: Vec::new(),
            operations: HashMap::new(),
        }
    }

    pub fn get_tensors_by_name(&self) -> HashMap<String, SymbolicGraphTensorId> {
        let mut tensors_by_name = HashMap::new();
        for (id, tensor) in &self.tensors {
            if let Some(name) = &tensor.onnx_name {
                tensors_by_name.insert(name.clone(), *id);
            }
        }
        tensors_by_name
    }

    pub fn get_operations(&self) -> &HashMap<SymbolicGraphOperationId, GraphOperation> {
        &self.operations
    }

    pub fn get_tensors(&self) -> &HashMap<SymbolicGraphTensorId, ONNXTensorInfo> {
        &self.tensors
    }

    pub fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        let mut results = Vec::new();
        for (id, info) in &self.tensors {
            if let TensorType::Output = info.tensor_type {
                results.push(*id);
            }
        }
        results
    }

    pub fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        let mut results = Vec::new();
        for (id, info) in &self.tensors {
            if let TensorType::Input(_) = info.tensor_type {
                results.push(*id);
            }
        }
        results
    }

    pub fn get_foreign_tensor_ids(&self) -> HashSet<SymbolicGraphTensorId> {
        let mut results = HashSet::new();
        for op in self.operations.values() {
            let inputs = op.op.get_inputs();
            for input in inputs {
                if !self.tensors.contains_key(&input) {
                    results.insert(input);
                }
            }
        }
        results
    }

    pub fn get_tensor_name(&self, tensor_id: SymbolicGraphTensorId) -> Option<&str> {
        if let Some(tensor) = self.tensors.get(&tensor_id) {
            tensor.onnx_name.as_deref()
        } else {
            None
        }
    }

    pub fn get_tensor_info(&self, tensor_id: SymbolicGraphTensorId) -> Option<&ONNXTensorInfo> {
        self.tensors.get(&tensor_id)
    }

    pub fn get_initialized_tensors(
        &self,
        tensor_store: &TensorStore,
    ) -> HashMap<SymbolicGraphTensorId, NumericTensor<DynRank>> {
        let mut out = HashMap::new();

        for (key, tensor) in &self.tensors {
            match &tensor.tensor_type {
                TensorType::Constant(x) => {
                    out.insert(*key, x.get_tensor(tensor_store));
                }
                TensorType::Input(Some(x)) => {
                    out.insert(*key, x.get_tensor(tensor_store));
                }
                _ => {}
            }
        }

        out
    }

    fn populate(
        &mut self,
        graph_mutator: &mut SymbolicGraphMutator,
        onnx_graph: &onnx::GraphProto,
        core_opset_version: usize,
    ) -> Result<(), ONNXDecodingError> {
        for t in onnx_graph.input.iter() {
            let tensor_id =
                graph_mutator.new_tensor_from_tensor_info(self, t, TensorType::Input(None))?;
            self.ordered_inputs.push(tensor_id);
        }
        for t in onnx_graph.output.iter() {
            let tensor_id =
                graph_mutator.new_tensor_from_tensor_info(self, t, TensorType::Output)?;
            self.ordered_outputs.push(tensor_id);
        }
        for t in onnx_graph.value_info.iter() {
            graph_mutator.new_tensor_from_tensor_info(self, t, TensorType::Intermediate)?;
        }

        for t in &onnx_graph.initializer {
            let numeric_tensor = NDArrayNumericTensor::try_from(t)?;
            let tensor = if numeric_tensor.num_elements() > 100 {
                // Larger tensors go into tensor store
                let id = graph_mutator.tensor_store.add_tensor(StoredTensor::Numeric(
                    NumericTensor::NDArray(numeric_tensor),
                ));
                StoredOrNotTensor::Stored(id)
            } else {
                StoredOrNotTensor::NotStored(numeric_tensor)
            };

            if let Some(x) = graph_mutator.tensors_by_name.get(&t.name) {
                let tensor_type = &mut self.tensors.get_mut(x).unwrap().tensor_type;
                match tensor_type.clone() {
                    TensorType::Input(_) => {
                        *tensor_type = TensorType::Input(Some(tensor));
                    }
                    TensorType::Constant(_) => {
                        *tensor_type = TensorType::Constant(tensor);
                    }
                    _ => {
                        *tensor_type = TensorType::Constant(tensor);
                    }
                }
            } else {
                match tensor {
                    StoredOrNotTensor::Stored(x) => {
                        graph_mutator.new_stored_tensor(self, x, Some(t.name.clone()));
                    }
                    StoredOrNotTensor::NotStored(x) => {
                        graph_mutator.new_constant_tensor(self, x, Some(t.name.clone()));
                    }
                }
            }
        }

        for node in &onnx_graph.node {
            graph_mutator.new_node_from_onnx_node(self, core_opset_version, node)?;
        }

        Ok(())
    }

    fn eval(
        &self,
        inputs: &HashMap<SymbolicGraphTensorId, NumericTensor<DynRank>>,
        eval_backend: &mut EvalBackend,
    ) -> Result<HashMap<SymbolicGraphTensorId, NumericTensor<DynRank>>, EvalError> {
        let mut active_tensors: HashMap<SymbolicGraphTensorId, NumericTensor<DynRank>> =
            inputs.clone();

        let ops = self.get_operations();
        let mut remaining_ops_to_complete: Vec<SymbolicGraphOperationId> =
            ops.keys().copied().collect();
        let mut total_ops_completed: Vec<SymbolicGraphOperationId> = vec![];
        loop {
            let mut ops_completed_now = vec![];

            for op_id in &remaining_ops_to_complete {
                let GraphOperation { name: _, op } = ops.get(op_id).unwrap();
                let input_ids = op.get_inputs();
                let mut input_values = HashMap::new();
                // Collect all inputs, abort if we can't do this one yet
                let mut failed_to_fetch = false;
                for tensor_id in &input_ids {
                    if let Some(value) = active_tensors.get(tensor_id) {
                        // Validate shape and dtype
                        if let Some(tensor_info) = self.get_tensor_info(*tensor_id) {
                            check_tensor_matches(value, tensor_info)?;
                        }
                        input_values.insert(*tensor_id, value.clone());
                    } else {
                        // Can't do this one yet
                        failed_to_fetch = true;
                        continue;
                    }
                }
                if failed_to_fetch {
                    continue;
                }
                let outputs = op.eval(eval_backend, &input_values)?;
                for (tensor_id, value) in outputs {
                    //assert_eq!(value.has_nan().unwrap(), false);

                    // Validate shape and dtype
                    if let Some(tensor_info) = self.get_tensor_info(tensor_id) {
                        check_tensor_matches(&value, tensor_info)?;
                    }
                    active_tensors.insert(tensor_id, value);
                }
                ops_completed_now.push(*op_id)
            }
            remaining_ops_to_complete.retain(|&x| !ops_completed_now.contains(&x));
            if ops_completed_now.is_empty() {
                // Hopefully we are done now
                break;
            }

            total_ops_completed.extend(ops_completed_now);
        }

        Ok(active_tensors)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SymbolicGraph {
    unknown_dimensions: HashMap<String, SymbolicScalar>,
    pub inner_graph: SymbolicGraphInner,
}

impl SymbolicGraph {
    pub fn new() -> Self {
        Self {
            unknown_dimensions: HashMap::new(),
            inner_graph: SymbolicGraphInner::new(),
        }
    }

    pub fn get_tensors_by_name(&self) -> HashMap<String, SymbolicGraphTensorId> {
        self.inner_graph.get_tensors_by_name()
    }

    pub fn get_unknown_dimensions_by_name(&self) -> &HashMap<String, SymbolicScalar> {
        &self.unknown_dimensions
    }

    pub fn get_operations(&self) -> &HashMap<SymbolicGraphOperationId, GraphOperation> {
        self.inner_graph.get_operations()
    }

    pub fn get_tensors(&self) -> &HashMap<SymbolicGraphTensorId, ONNXTensorInfo> {
        self.inner_graph.get_tensors()
    }

    pub fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        self.inner_graph.get_outputs()
    }

    pub fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        self.inner_graph.get_inputs()
    }

    pub fn get_tensor_name(&self, tensor_id: SymbolicGraphTensorId) -> Option<&str> {
        self.inner_graph.get_tensor_name(tensor_id)
    }

    pub fn get_tensor_info(&self, tensor_id: SymbolicGraphTensorId) -> Option<&ONNXTensorInfo> {
        self.inner_graph.get_tensor_info(tensor_id)
    }

    pub fn get_initialized_tensors(
        &self,
        tensor_store: &TensorStore,
    ) -> HashMap<SymbolicGraphTensorId, NumericTensor<DynRank>> {
        self.inner_graph.get_initialized_tensors(tensor_store)
    }
}

impl TryFrom<&onnx::TensorProto> for NDArrayNumericTensor<DynRank> {
    type Error = ONNXDecodingError;

    fn try_from(tensor: &onnx::TensorProto) -> Result<Self, Self::Error> {
        let dtype = DType::try_from(
            onnx::tensor_proto::DataType::try_from(tensor.data_type)
                .map_err(|x| ONNXDecodingError::ProtobufDecodeError(anyhow::Error::from(x)))?,
        )?;

        let shape = tensor.dims.iter().map(|x| *x as u64).collect();

        let out = if !tensor.raw_data.is_empty() {
            NDArrayNumericTensor::from_raw_data(&tensor.raw_data, dtype, shape)?
        } else if !tensor.float_data.is_empty() {
            match dtype {
                DType::F32 => {
                    NDArrayNumericTensor::from_vec_shape(tensor.float_data.clone(), &shape)?
                }
                _ => Err(ONNXDecodingError::UnsupportedONNX(
                    "Unsupported dtype in float_data field!".to_string(),
                ))?,
            }
        } else if !tensor.double_data.is_empty() {
            match dtype {
                DType::F64 => {
                    NDArrayNumericTensor::from_vec_shape(tensor.double_data.clone(), &shape)?
                }
                _ => Err(ONNXDecodingError::UnsupportedONNX(
                    "Unsupported dtype in double_data field!".to_string(),
                ))?,
            }
        } else if !tensor.int32_data.is_empty() {
            match dtype {
                DType::I32 => {
                    NDArrayNumericTensor::from_vec_shape(tensor.int32_data.clone(), &shape)?
                }
                DType::U16 => NDArrayNumericTensor::from_vec_shape(
                    tensor
                        .int32_data
                        .iter()
                        .map(|x| *x as u16)
                        .collect::<Vec<_>>(),
                    &shape,
                )?,
                DType::I16 => NDArrayNumericTensor::from_vec_shape(
                    tensor
                        .int32_data
                        .iter()
                        .map(|x| *x as i16)
                        .collect::<Vec<_>>(),
                    &shape,
                )?,
                DType::U8 => NDArrayNumericTensor::from_vec_shape(
                    tensor
                        .int32_data
                        .iter()
                        .map(|x| *x as u8)
                        .collect::<Vec<_>>(),
                    &shape,
                )?,
                DType::I8 => NDArrayNumericTensor::from_vec_shape(
                    tensor
                        .int32_data
                        .iter()
                        .map(|x| *x as i8)
                        .collect::<Vec<_>>(),
                    &shape,
                )?,
                _ => Err(ONNXDecodingError::UnsupportedONNX(
                    "Unsupported dtype in int32_data field!".to_string(),
                ))?,
            }
        } else if !tensor.int64_data.is_empty() {
            match dtype {
                DType::I64 => {
                    NDArrayNumericTensor::from_vec_shape(tensor.int64_data.clone(), &shape)?
                }
                _ => Err(ONNXDecodingError::UnsupportedONNX(
                    "Unsupported dtype in int32_data field!".to_string(),
                ))?,
            }
        } else if !tensor.uint64_data.is_empty() {
            match dtype {
                DType::U64 => {
                    NDArrayNumericTensor::from_vec_shape(tensor.uint64_data.clone(), &shape)?
                }
                DType::U32 => NDArrayNumericTensor::from_vec_shape(
                    tensor
                        .uint64_data
                        .iter()
                        .map(|x| *x as u32)
                        .collect::<Vec<_>>(),
                    &shape,
                )?,
                _ => Err(ONNXDecodingError::UnsupportedONNX(
                    "Unsupported dtype in int32_data field!".to_string(),
                ))?,
            }
        } else if !tensor.string_data.is_empty() {
            match dtype {
                DType::STRING => {
                    let strings = tensor
                        .string_data
                        .iter()
                        .map(|x| String::from_utf8(x.clone()).unwrap())
                        .collect::<Vec<_>>();
                    NDArrayNumericTensor::from_vec_shape(strings, &shape)?
                }
                _ => Err(ONNXDecodingError::UnsupportedONNX(
                    "Unsupported dtype in string field!".to_string(),
                ))?,
            }
        } else {
            NDArrayNumericTensor::fill(NumericScalar::zero_of(dtype), &shape)?
        };
        assert_eq!(out.dtype(), dtype);
        Ok(out)
    }
}

pub struct SymbolicGraphMutator {
    graph: Option<SymbolicGraph>,
    tensors_by_name: HashMap<String, SymbolicGraphTensorId>,
    unknown_dimensions_by_name: HashMap<String, SymbolicScalarTyped<u64>>,
    next_tensor_id: SymbolicGraphTensorId,
    symbolic_resolver: SymbolicResolver,
    next_operation_id: SymbolicGraphOperationId,
    tensor_store: TensorStore,
}

impl SymbolicGraphMutator {
    pub fn new() -> Self {
        Self::from_graph(SymbolicGraph::new(), TensorStore::new())
    }

    pub fn get_inner(self) -> (SymbolicGraph, TensorStore) {
        (self.graph.unwrap(), self.tensor_store)
    }

    pub fn from_graph(graph: SymbolicGraph, tensor_store: TensorStore) -> Self {
        let next_tensor_id = graph
            .inner_graph
            .tensors
            .keys()
            .max()
            .map(|x| x + 1)
            .unwrap_or(0);
        let mut dimension_resolver = SymbolicResolver::new();
        for dim in graph.unknown_dimensions.values() {
            dimension_resolver.update_last_assigned(dim.clone())
        }
        let next_operation_id = graph
            .inner_graph
            .operations
            .keys()
            .max()
            .map(|x| x + 1)
            .unwrap_or(0);
        Self {
            tensors_by_name: graph.get_tensors_by_name(),
            graph: Some(graph),
            next_tensor_id,
            symbolic_resolver: dimension_resolver,
            unknown_dimensions_by_name: HashMap::new(),
            next_operation_id,
            tensor_store,
        }
    }

    pub fn from_onnx_bytes(onnx_bytes: &[u8]) -> Result<Self, ONNXDecodingError> {
        let model = onnx::ModelProto::decode(onnx_bytes)
            .map_err(|x| ONNXDecodingError::ProtobufDecodeError(anyhow::Error::from(x)))?;
        Self::from_onnx_model_proto(model)
    }

    pub(crate) fn new_unknown_tensor(
        &mut self,
        inner_graph: &mut SymbolicGraphInner,
        name: &str,
        tensor_type: TensorType,
    ) -> SymbolicGraphTensorId {
        let tensor_id = self.next_tensor_id;
        self.next_tensor_id += 1;
        let tensor = ONNXTensorInfo {
            onnx_name: Some(name.to_string()),
            tensor_type,
            dtype: None,
            shape: None,
        };
        inner_graph.tensors.insert(tensor_id, tensor);
        self.tensors_by_name.insert(name.to_string(), tensor_id);
        tensor_id
    }

    pub(crate) fn new_tensor_from_tensor_info(
        &mut self,
        inner_graph: &mut SymbolicGraphInner,
        tensor_info: &onnx::ValueInfoProto,
        tensor_type: TensorType,
    ) -> Result<SymbolicGraphTensorId, ONNXDecodingError> {
        let tensor_id = self.next_tensor_id;
        self.next_tensor_id += 1;
        let onnx_tensor_type = tensor_info
            .r#type
            .as_ref()
            .ok_or(ONNXDecodingError::MissingField("value_info.type"))?;
        let onnx_tensor_type_value = onnx_tensor_type
            .value
            .as_ref()
            .ok_or(ONNXDecodingError::MissingField("value_info.type.value"))?;
        let onnx_tensor_type_value_inner =
            if let onnx::type_proto::Value::TensorType(tensor_type) = onnx_tensor_type_value {
                tensor_type
            } else {
                return Err(ONNXDecodingError::UnsupportedTypeValue(
                    onnx_tensor_type_value.clone(),
                ));
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
            let value = dim
                .value
                .as_ref()
                .ok_or(ONNXDecodingError::MissingField("dim.value"))?;
            dimensions.push(match value {
                onnx::tensor_shape_proto::dimension::Value::DimValue(x) => {
                    ScalarInfoTyped::Numeric(if *x < 0 {
                        return Err(ONNXDecodingError::NegativeDimensionError);
                    } else {
                        *x as u64
                    })
                }
                onnx::tensor_shape_proto::dimension::Value::DimParam(x) => {
                    let unknown_dim_id =
                        if let Some(x) = self.unknown_dimensions_by_name.get(x.as_str()) {
                            x.clone()
                        } else {
                            let new_dim = SymbolicScalarTyped::new(&mut self.symbolic_resolver);
                            self.unknown_dimensions_by_name
                                .insert(x.clone(), new_dim.clone());
                            new_dim
                        };
                    ScalarInfoTyped::Symbolic(unknown_dim_id)
                }
            });
        }

        let name = tensor_info.name.clone();

        self.tensors_by_name.insert(name.clone(), tensor_id);
        inner_graph.tensors.insert(
            tensor_id,
            ONNXTensorInfo {
                onnx_name: Some(name),
                dtype: Some(dtype),
                shape: Some(dimensions),
                tensor_type,
            },
        );
        Ok(tensor_id)
    }

    pub(crate) fn new_constant_tensor(
        &mut self,
        inner_graph: &mut SymbolicGraphInner,
        value: NDArrayNumericTensor<DynRank>,
        name: Option<String>,
    ) -> SymbolicGraphTensorId {
        let tensor_id = self.next_tensor_id;
        self.next_tensor_id += 1;

        let mut shape = Vec::new();
        for s in value.shape() {
            shape.push(ScalarInfoTyped::Numeric(s))
        }

        inner_graph.tensors.insert(
            tensor_id,
            ONNXTensorInfo {
                onnx_name: name.clone(),
                dtype: Some(value.dtype()),
                shape: Some(shape),
                tensor_type: TensorType::Constant(StoredOrNotTensor::NotStored(value)),
            },
        );
        if let Some(name) = name {
            self.tensors_by_name.insert(name, tensor_id);
        }

        tensor_id
    }

    pub(crate) fn new_stored_tensor(
        &mut self,
        inner_graph: &mut SymbolicGraphInner,
        id: TensorStoreTensorId,
        name: Option<String>,
    ) -> SymbolicGraphTensorId {
        let tensor_id = self.next_tensor_id;
        self.next_tensor_id += 1;

        let tensor_ref = self.tensor_store.get_tensor(id).unwrap();

        let mut shape = Vec::new();
        for s in tensor_ref.shape() {
            shape.push(ScalarInfoTyped::Numeric(s))
        }

        inner_graph.tensors.insert(
            tensor_id,
            ONNXTensorInfo {
                onnx_name: name.clone(),
                dtype: Some(tensor_ref.dtype()),
                shape: Some(shape),
                tensor_type: TensorType::Constant(StoredOrNotTensor::Stored(id)),
            },
        );
        if let Some(name) = name {
            self.tensors_by_name.insert(name, tensor_id);
        }

        tensor_id
    }

    pub(crate) fn new_node_from_onnx_node(
        &mut self,
        inner_graph: &mut SymbolicGraphInner,
        core_opset_version: usize,
        onnx_node: &onnx::NodeProto,
    ) -> Result<(), ONNXDecodingError> {
        let mut input_tensors = vec![];
        for input in &onnx_node.input {
            input_tensors.push(if input.is_empty() {
                None
            } else if let Some(tensor_id) = self.tensors_by_name.get(input) {
                Some(*tensor_id)
            } else {
                Some(self.new_unknown_tensor(inner_graph, input, TensorType::Intermediate))
            });
        }

        let mut output_tensors = vec![];
        for output in &onnx_node.output {
            output_tensors.push(if output.is_empty() {
                None
            } else if let Some(tensor_id) = self.tensors_by_name.get(output) {
                Some(*tensor_id)
            } else {
                Some(self.new_unknown_tensor(inner_graph, output, TensorType::Intermediate))
            });
        }

        let name = if onnx_node.name.is_empty() {
            None
        } else {
            Some(onnx_node.name.clone())
        };

        let new_op = match onnx_node.op_type.as_str() {
            "Max" => Some(AnyOperation::Max(ops::MaxOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
            )?)),
            "Min" => Some(AnyOperation::Min(ops::MinOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
            )?)),
            "Add" => Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichBinaryOperation::Add,
            )?)),
            "Mod" => Some(AnyOperation::Modulo(ops::ModuloOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
            )?)),
            "Sub" => Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichBinaryOperation::Sub,
            )?)),
            "Mul" => Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichBinaryOperation::Mul,
            )?)),
            "Div" => Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichBinaryOperation::Div,
            )?)),
            "MatMul" => Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichBinaryOperation::MatMul,
            )?)),
            "Equal" => Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichBinaryOperation::Equal,
            )?)),
            "Greater" => Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichBinaryOperation::Greater,
            )?)),
            "GreaterOrEqual" => Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichBinaryOperation::GreaterOrEqual,
            )?)),
            "Less" => Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichBinaryOperation::Less,
            )?)),
            "LessOrEqual" => Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichBinaryOperation::LessOrEqual,
            )?)),
            "And" => Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichBinaryOperation::And,
            )?)),
            "Or" => Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichBinaryOperation::Or,
            )?)),
            "Xor" => Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichBinaryOperation::Xor,
            )?)),
            "BitwiseAnd" => Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichBinaryOperation::BitwiseAnd,
            )?)),
            "BitwiseOr" => Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichBinaryOperation::BitwiseOr,
            )?)),
            "BitwiseXor" => Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichBinaryOperation::BitwiseXor,
            )?)),
            "Relu" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Relu,
            )?)),
            "Sigmoid" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Sigmoid,
            )?)),
            "Asin" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Trig(TrigOp::Asin),
            )?)),
            "Asinh" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Trig(TrigOp::Asinh),
            )?)),
            "Acos" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Trig(TrigOp::Acos),
            )?)),
            "Acosh" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Trig(TrigOp::Acosh),
            )?)),
            "Atan" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Trig(TrigOp::Atan),
            )?)),
            "Atanh" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Trig(TrigOp::Atanh),
            )?)),
            "Sin" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Trig(TrigOp::Sin),
            )?)),
            "Sinh" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Trig(TrigOp::Sinh),
            )?)),
            "Cos" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Trig(TrigOp::Cos),
            )?)),
            "Cosh" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Trig(TrigOp::Cosh),
            )?)),
            "Tan" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Trig(TrigOp::Tan),
            )?)),
            "Tanh" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Trig(TrigOp::Tanh),
            )?)),
            "Exp" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Exp,
            )?)),
            "Log" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Log,
            )?)),
            "Sqrt" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Sqrt,
            )?)),
            "BitwiseNot" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::BitwiseNot,
            )?)),
            "Not" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Not,
            )?)),
            "Softplus" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Softplus,
            )?)),
            "Neg" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Neg,
            )?)),
            "Abs" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Abs,
            )?)),
            "Sign" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Sign,
            )?)),
            "IsNaN" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::IsNan,
            )?)),
            "Erf" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Erf,
            )?)),
            "IsInf" => Some(AnyOperation::IsInf(ops::IsInfOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
            )?)),
            "Floor" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Floor,
            )?)),
            "Ceil" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Ceil,
            )?)),
            "Round" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Round,
            )?)),
            "Clip" => Some(AnyOperation::Clip(ops::ClipOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
            )?)),
            "Reciprocal" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Reciprocal,
            )?)),
            "Size" => Some(AnyOperation::Size(ops::SizeOperation::from_onnx(
                &input_tensors,
                &output_tensors,
            )?)),
            "NonZero" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::NonZero,
            )?)),
            "LpNormalization" => Some(AnyOperation::LpNormalization(
                ops::LpNormalizationOperation::from_onnx(
                    &input_tensors,
                    &output_tensors,
                    &onnx_node.attribute,
                )?,
            )),
            "GroupNormalization" => Some(AnyOperation::GroupNormalization(
                ops::GroupNormalizationOperation::from_onnx(
                    &input_tensors,
                    &output_tensors,
                    &onnx_node.attribute,
                )?,
            )),
            "LayerNormalization" => Some(AnyOperation::LayerNormalization(
                ops::LayerNormalizationOperation::from_onnx(
                    &input_tensors,
                    &output_tensors,
                    &onnx_node.attribute,
                )?,
            )),
            "CumSum" => Some(AnyOperation::CumSum(ops::CumSumOperation::from_onnx(
                &input_tensors,
                &output_tensors,
            )?)),
            "Gather" => Some(AnyOperation::Gather(ops::GatherOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
            )?)),
            "Cast" => Some(AnyOperation::Cast(ops::CastOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
            )?)),
            "CastLike" => Some(AnyOperation::CastLike(ops::CastLikeOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
            )?)),
            "Pow" => Some(AnyOperation::Pow(ops::PowOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
            )?)),
            "ReduceMean" => Some(AnyOperation::ReduceMean(
                ops::ReduceMeanOperation::from_onnx(
                    &input_tensors,
                    &output_tensors,
                    &onnx_node.attribute,
                )?,
            )),
            "ReduceSum" => Some(AnyOperation::ReduceSum(ops::ReduceSumOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
            )?)),
            "ReduceProd" => Some(AnyOperation::ReduceProd(
                ops::ReduceProdOperation::from_onnx(
                    &input_tensors,
                    &output_tensors,
                    &onnx_node.attribute,
                )?,
            )),
            "ReduceMin" => Some(AnyOperation::ReduceMin(ops::ReduceMinOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
            )?)),
            "ReduceMax" => Some(AnyOperation::ReduceMax(ops::ReduceMaxOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
            )?)),
            "Flatten" => Some(AnyOperation::Flatten(ops::FlattenOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
            )?)),
            "Expand" => Some(AnyOperation::Expand(ops::ExpandOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
            )?)),
            "Squeeze" => Some(AnyOperation::Squeeze(ops::SqueezeOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
            )?)),
            "Unsqueeze" => Some(AnyOperation::Unsqueeze(ops::UnsqueezeOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
            )?)),
            "Transpose" => Some(AnyOperation::Transpose(ops::TransposeOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
            )?)),
            "Reshape" => Some(AnyOperation::Reshape(ops::ReshapeOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
            )?)),
            "Shape" => Some(AnyOperation::Shape(ops::ShapeOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
            )?)),
            "Range" => Some(AnyOperation::Range(ops::RangeOperation::from_onnx(
                &input_tensors,
                &output_tensors,
            )?)),
            "Concat" => Some(AnyOperation::Concat(ops::ConcatOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
            )?)),
            "ConstantOfShape" => Some(AnyOperation::ConstantOfShape(
                ops::ConstantOfShapeOperation::from_onnx(
                    &input_tensors,
                    &output_tensors,
                    &onnx_node.attribute,
                )?,
            )),
            "Gemm" => Some(AnyOperation::Gemm(ops::GemmOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
            )?)),
            "Slice" => Some(AnyOperation::Slice(ops::SliceOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
            )?)),
            "Where" => Some(AnyOperation::Where(ops::WhereOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
            )?)),
            "Softmax" => Some(AnyOperation::Softmax(ops::SoftmaxOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
            )?)),
            "LogSoftmax" => Some(AnyOperation::LogSoftmax(
                ops::LogSoftmaxOperation::from_onnx(
                    &input_tensors,
                    &output_tensors,
                    &onnx_node.attribute,
                )?,
            )),
            "Split" => Some(AnyOperation::Split(ops::SplitOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
            )?)),
            "Constant" => Some(AnyOperation::Constant(ops::ConstantOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
            )?)),
            "Identity" => Some(AnyOperation::Identity(ops::IdentityOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
            )?)),
            "Conv" => Some(AnyOperation::Conv(ops::ConvOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
            )?)),
            "InstanceNormalization" => Some(AnyOperation::InstanceNormalization(
                ops::InstanceNormalizationOperation::from_onnx(
                    &input_tensors,
                    &output_tensors,
                    &onnx_node.attribute,
                )?,
            )),
            "Resize" => Some(AnyOperation::Resize(ops::ResizeOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
            )?)),
            "Pad" => Some(AnyOperation::Pad(ops::PadOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
            )?)),
            "RandomNormalLike" => Some(AnyOperation::RandomNormalLike(
                ops::RandomNormalLikeOperation::from_onnx(
                    &input_tensors,
                    &output_tensors,
                    &onnx_node.attribute,
                )?,
            )),
            "ArgMax" => Some(AnyOperation::ArgMax(ops::ArgMaxOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
            )?)),
            "ArgMin" => Some(AnyOperation::ArgMin(ops::ArgMinOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
            )?)),
            "If" => Some(AnyOperation::If(ops::IfOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                self,
                core_opset_version,
            )?)),
            "Scan" => Some(AnyOperation::Scan(ops::ScanOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                self,
                core_opset_version,
            )?)),
            x => Err(ONNXDecodingError::UnsupportedONNXType(x.to_string()))?,
        };

        if let Some(op) = new_op {
            let op = GraphOperation { name, op };

            let operation_id = self.next_operation_id;
            self.next_operation_id += 1;
            inner_graph.operations.insert(operation_id, op);
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

        let onnx_graph = model_proto
            .graph
            .ok_or(ONNXDecodingError::MissingField("graph"))?;
        let mut graph_mutator = Self::new();
        let SymbolicGraph {
            mut inner_graph,
            unknown_dimensions,
        } = graph_mutator.graph.take().unwrap();

        inner_graph.populate(&mut graph_mutator, &onnx_graph, core_opset_version)?;

        graph_mutator.graph = Some(SymbolicGraph {
            inner_graph,
            unknown_dimensions,
        });

        Ok(graph_mutator)
    }
}

impl Default for SymbolicGraphMutator {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for SymbolicGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub enum SymbolicGraphTensorPath {
    Tensor(SymbolicGraphTensorId),
}

#[derive(Clone, Debug, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub enum SymbolicGraphNodePath {
    Node(SymbolicGraphOperationId),
}

#[derive(Clone, Debug, Default)]
pub struct SymbolicGraphTelemetryRequest {
    pub subscribed_tensors: HashSet<SymbolicGraphTensorPath>,
}

#[derive(Clone, Debug, Default)]
pub struct SymbolicGraphTelemetryResponse {
    pub subscribed_tensors: HashMap<SymbolicGraphTensorPath, NDArrayNumericTensor<DynRank>>,
}
