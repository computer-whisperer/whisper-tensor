pub mod observer;
pub mod ops;
pub mod tensor_store;

use crate::backends::ModelLoadedTensorCache;
use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::{NDArrayNumericTensor, NDArrayNumericTensorError};
use crate::dtype::DType;
use crate::graph::{
    GlobalId, Graph, Link, LinkCategory, LinkMetadata, Node, NodeMetadata, Property,
};
use crate::numeric_scalar::NumericScalar;
use crate::numeric_tensor::NumericTensor;
use crate::scalar_info::ScalarInfoTyped;
use crate::symbolic_graph::ops::{AnyOperation, EvalError, Operation};
use crate::symbolic_graph::tensor_store::{StoredTensor, TensorStore, TensorStoreTensorId};
use crate::symbolic_scalar::{SymbolicResolver, SymbolicScalar, SymbolicScalarTyped};
use crate::tensor_rank::DynRank;
use crate::{TrigOp, onnx};
use prost::Message;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, thiserror::Error)]
pub enum ONNXDecodingError {
    #[error("Invalid operator inputs: {0}")]
    InvalidOperatorInputs(&'static str),
    #[error("Invalid operator outputs: {0}")]
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
    global_id: GlobalId,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SymbolicGraph {
    global_id: GlobalId,
    unknown_dimensions: HashMap<String, SymbolicScalar>,
    tensors: HashMap<GlobalId, ONNXTensorInfo>,
    ordered_outputs: Vec<GlobalId>,
    ordered_inputs: Vec<GlobalId>,
    operations: HashMap<GlobalId, GraphOperation>,
}

impl SymbolicGraph {
    pub fn new(rng: &mut impl Rng) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            unknown_dimensions: HashMap::new(),
            tensors: HashMap::new(),
            ordered_outputs: Vec::new(),
            ordered_inputs: Vec::new(),
            operations: HashMap::new(),
        }
    }

    pub fn get_unknown_dimensions_by_name(&self) -> &HashMap<String, SymbolicScalar> {
        &self.unknown_dimensions
    }

    pub fn get_tensors_by_name(&self) -> HashMap<String, GlobalId> {
        let mut tensors_by_name = HashMap::new();
        for (id, tensor) in &self.tensors {
            if let Some(name) = &tensor.onnx_name {
                tensors_by_name.insert(name.clone(), *id);
            }
        }
        tensors_by_name
    }

    pub fn get_operations(&self) -> &HashMap<GlobalId, GraphOperation> {
        &self.operations
    }

    pub fn get_tensors(&self) -> &HashMap<GlobalId, ONNXTensorInfo> {
        &self.tensors
    }

    pub fn get_outputs(&self) -> Vec<GlobalId> {
        let mut results = Vec::new();
        for (id, info) in &self.tensors {
            if let TensorType::Output = info.tensor_type {
                results.push(*id);
            }
        }
        results
    }

    pub fn get_inputs(&self) -> Vec<GlobalId> {
        let mut results = Vec::new();
        for (id, info) in &self.tensors {
            if let TensorType::Input(_) = info.tensor_type {
                results.push(*id);
            }
        }
        results
    }

    pub fn get_foreign_tensor_ids(&self) -> HashSet<GlobalId> {
        let mut results = HashSet::new();
        for op in self.operations.values() {
            let inputs = op.op.inputs();
            for input in inputs {
                if !self.tensors.contains_key(&input) {
                    results.insert(input);
                }
            }
        }
        results
    }

    pub fn get_tensor_name(&self, tensor_id: GlobalId) -> Option<&str> {
        if let Some(tensor) = self.tensors.get(&tensor_id) {
            tensor.onnx_name.as_deref()
        } else {
            None
        }
    }

    pub fn get_tensor_info(&self, tensor_id: GlobalId) -> Option<&ONNXTensorInfo> {
        self.tensors.get(&tensor_id)
    }

    pub fn get_initialized_tensors_cached(
        &self,
        tensor_store: &TensorStore,
        loaded_tensor_cache: &mut ModelLoadedTensorCache,
        eval_backend: &mut EvalBackend,
    ) -> HashMap<GlobalId, NumericTensor<DynRank>> {
        let mut out = HashMap::new();

        for (key, tensor) in &self.tensors {
            if let Some(x) = loaded_tensor_cache.tensors.get(key) {
                out.insert(*key, x.clone());
            } else {
                let tensor = match &tensor.tensor_type {
                    TensorType::Constant(x) => Some(x.get_tensor(tensor_store)),
                    TensorType::Input(Some(x)) => Some(x.get_tensor(tensor_store)),
                    _ => None,
                };
                if let Some(tensor) = tensor {
                    let loaded_tensor = eval_backend.to_native_type(&tensor);
                    loaded_tensor_cache
                        .tensors
                        .insert(*key, loaded_tensor.clone());
                    out.insert(*key, loaded_tensor);
                }
            }
        }

        out
    }

    pub fn get_initialized_tensors(
        &self,
        tensor_store: &TensorStore,
    ) -> HashMap<GlobalId, NumericTensor<DynRank>> {
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
        rng: &mut impl Rng,
        base_dir: Option<&std::path::Path>,
    ) -> Result<(), ONNXDecodingError> {
        for t in onnx_graph.input.iter() {
            let tensor_id =
                graph_mutator.new_tensor_from_tensor_info(self, t, TensorType::Input(None), rng)?;
            self.ordered_inputs.push(tensor_id);
        }
        for t in onnx_graph.output.iter() {
            let tensor_id =
                graph_mutator.new_tensor_from_tensor_info(self, t, TensorType::Output, rng)?;
            self.ordered_outputs.push(tensor_id);
        }
        for t in onnx_graph.value_info.iter() {
            graph_mutator.new_tensor_from_tensor_info(self, t, TensorType::Intermediate, rng)?;
        }

        let resolve_path = |location: &str| -> String {
            if let Some(dir) = base_dir {
                let p = std::path::Path::new(location);
                if p.is_relative() {
                    return dir.join(p).to_string_lossy().into_owned();
                }
            }
            location.to_string()
        };

        for t in &onnx_graph.initializer {
            // Detect external data reference
            let uses_external = t.data_location
                == onnx::tensor_proto::DataLocation::External as i32
                || !t.external_data.is_empty();
            let tensor: StoredOrNotTensor = if uses_external {
                // Parse keys: location, offset, length
                let mut map = std::collections::HashMap::new();
                for kv in &t.external_data {
                    map.insert(kv.key.clone(), kv.value.clone());
                }
                if let Some(fmt) = map.get("format") {
                    if fmt == "pth" {
                        if let (Some(location), Some(tensor_name)) =
                            (map.get("location"), map.get("tensor_name"))
                        {
                            let dtype = DType::try_from(
                                onnx::tensor_proto::DataType::try_from(t.data_type).map_err(
                                    |x| {
                                        ONNXDecodingError::ProtobufDecodeError(anyhow::Error::from(
                                            x,
                                        ))
                                    },
                                )?,
                            )?;
                            let shape = t.dims.iter().map(|x| *x as u64).collect::<Vec<_>>();
                            let id =
                                graph_mutator
                                    .tensor_store
                                    .add_tensor(StoredTensor::ExternalPth {
                                        path: resolve_path(location),
                                        tensor_name: tensor_name.clone(),
                                        dtype,
                                        shape: shape.clone(),
                                    });
                            StoredOrNotTensor::Stored(id)
                        } else {
                            // Missing required keys for pth; fallback to eager load
                            let numeric_tensor = NDArrayNumericTensor::try_from(t)?;
                            if numeric_tensor.num_elements() > 100 {
                                let id = graph_mutator.tensor_store.add_tensor(
                                    StoredTensor::Numeric(NumericTensor::NDArray(numeric_tensor)),
                                );
                                StoredOrNotTensor::Stored(id)
                            } else {
                                StoredOrNotTensor::NotStored(numeric_tensor)
                            }
                        }
                    } else {
                        // Not pth; try generic external binary with offset/length
                        if let (Some(location), Some(offset), Some(length)) =
                            (map.get("location"), map.get("offset"), map.get("length"))
                        {
                            let dtype = DType::try_from(
                                onnx::tensor_proto::DataType::try_from(t.data_type).map_err(
                                    |x| {
                                        ONNXDecodingError::ProtobufDecodeError(anyhow::Error::from(
                                            x,
                                        ))
                                    },
                                )?,
                            )?;
                            let shape = t.dims.iter().map(|x| *x as u64).collect::<Vec<_>>();
                            let id = graph_mutator.tensor_store.add_tensor(
                                StoredTensor::ExternalBinary {
                                    path: resolve_path(location),
                                    offset: offset.parse().unwrap_or(0usize),
                                    length: length.parse().unwrap_or(0usize),
                                    dtype,
                                    shape: shape.clone(),
                                },
                            );
                            StoredOrNotTensor::Stored(id)
                        } else {
                            // Fallback to eager load when essential keys missing
                            let numeric_tensor = NDArrayNumericTensor::try_from(t)?;
                            if numeric_tensor.num_elements() > 100 {
                                let id = graph_mutator.tensor_store.add_tensor(
                                    StoredTensor::Numeric(NumericTensor::NDArray(numeric_tensor)),
                                );
                                StoredOrNotTensor::Stored(id)
                            } else {
                                StoredOrNotTensor::NotStored(numeric_tensor)
                            }
                        }
                    }
                } else if let (Some(location), Some(offset), Some(length)) =
                    (map.get("location"), map.get("offset"), map.get("length"))
                {
                    let dtype = DType::try_from(
                        onnx::tensor_proto::DataType::try_from(t.data_type).map_err(|x| {
                            ONNXDecodingError::ProtobufDecodeError(anyhow::Error::from(x))
                        })?,
                    )?;
                    let shape = t.dims.iter().map(|x| *x as u64).collect::<Vec<_>>();
                    let id = graph_mutator
                        .tensor_store
                        .add_tensor(StoredTensor::ExternalBinary {
                            path: resolve_path(location),
                            offset: offset.parse().unwrap_or(0usize),
                            length: length.parse().unwrap_or(0usize),
                            dtype,
                            shape: shape.clone(),
                        });
                    StoredOrNotTensor::Stored(id)
                } else {
                    // Fallback to eager load when essential keys missing
                    let numeric_tensor = NDArrayNumericTensor::try_from(t)?;
                    if numeric_tensor.num_elements() > 100 {
                        let id = graph_mutator.tensor_store.add_tensor(StoredTensor::Numeric(
                            NumericTensor::NDArray(numeric_tensor),
                        ));
                        StoredOrNotTensor::Stored(id)
                    } else {
                        StoredOrNotTensor::NotStored(numeric_tensor)
                    }
                }
            } else {
                let numeric_tensor = NDArrayNumericTensor::try_from(t)?;
                if numeric_tensor.num_elements() > 100 {
                    let id = graph_mutator.tensor_store.add_tensor(StoredTensor::Numeric(
                        NumericTensor::NDArray(numeric_tensor),
                    ));
                    StoredOrNotTensor::Stored(id)
                } else {
                    StoredOrNotTensor::NotStored(numeric_tensor)
                }
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
                        graph_mutator.new_stored_tensor(self, x, Some(t.name.clone()), rng);
                    }
                    StoredOrNotTensor::NotStored(x) => {
                        graph_mutator.new_constant_tensor(self, x, Some(t.name.clone()), rng);
                    }
                }
            }
        }

        for node in &onnx_graph.node {
            graph_mutator.new_node_from_onnx_node(self, core_opset_version, node, rng)?;
        }

        Ok(())
    }

    /// Topological sort of operations using Kahn's algorithm.
    /// Returns op IDs in dependency order.
    pub fn topological_order_vec(&self) -> Vec<GlobalId> {
        use crate::graph::Node;

        // Build tensor_producer: tensor_id → producing op_id
        let mut tensor_producer: HashMap<GlobalId, GlobalId> = HashMap::new();
        for (op_id, graph_op) in &self.operations {
            for output_id in graph_op.op.outputs() {
                tensor_producer.insert(output_id, *op_id);
            }
        }

        // Build in-degree per op (count of distinct producer ops for its inputs)
        let mut in_degree: HashMap<GlobalId, usize> = HashMap::new();
        let mut dependents: HashMap<GlobalId, Vec<GlobalId>> = HashMap::new();

        for (op_id, graph_op) in &self.operations {
            // Collect distinct producer ops for this op's inputs
            let mut producer_ops: HashSet<GlobalId> = HashSet::new();
            for input_id in graph_op.op.inputs() {
                if let Some(&producer_op) = tensor_producer.get(&input_id) {
                    producer_ops.insert(producer_op);
                }
            }
            in_degree.insert(*op_id, producer_ops.len());
            for &producer_op in &producer_ops {
                dependents.entry(producer_op).or_default().push(*op_id);
            }
        }

        // Seed queue with in-degree 0 ops
        let mut queue: std::collections::VecDeque<GlobalId> = std::collections::VecDeque::new();
        for (op_id, &deg) in &in_degree {
            if deg == 0 {
                queue.push_back(*op_id);
            }
        }

        // Drain queue
        let mut result = Vec::with_capacity(self.operations.len());
        while let Some(op_id) = queue.pop_front() {
            result.push(op_id);
            if let Some(deps) = dependents.get(&op_id) {
                for &dep_id in deps {
                    let deg = in_degree.get_mut(&dep_id).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(dep_id);
                    }
                }
            }
        }

        assert_eq!(
            result.len(),
            self.operations.len(),
            "Topological sort did not visit all operations — possible cycle"
        );

        result
    }

    /// Flatten this SymbolicGraph into a single combined MilliOpGraph.
    pub fn generate_milli_graph(&self, rng: &mut impl Rng) -> crate::milli_graph::MilliOpGraph {
        use crate::graph::{Graph, Node};
        use crate::milli_graph::{MilliOpGraph, MilliOpGroup, MilliOpPhase};

        let mut combined = MilliOpGraph::new_empty(rng);
        let mut sym_to_combined: HashMap<GlobalId, GlobalId> = HashMap::new();

        // 1. Map ordered_inputs
        for &input_id in &self.ordered_inputs {
            let internal = combined.add_input_with_id(input_id, rng);
            sym_to_combined.insert(input_id, internal);
        }

        // 2. Map standalone constants
        for const_id in self.constant_link_ids() {
            let internal = combined.add_input_with_id(const_id, rng);
            sym_to_combined.insert(const_id, internal);
        }

        // 3. Walk ops in topological order, merge each with a group
        for op_id in self.topological_order_vec() {
            let graph_op = &self.operations[&op_id];
            let label = graph_op
                .name
                .clone()
                .unwrap_or_else(|| graph_op.op.op_kind());
            let group = MilliOpGroup {
                id: GlobalId::new(rng),
                source_op: Some(op_id),
                source_graph: Some(self.global_id),
                phase: MilliOpPhase::Forward,
                label: Some(label),
                ..Default::default()
            };
            let group_id = combined.create_group(group);

            let op_graph = graph_op.op.get_milli_op_graph(rng);
            combined.merge_graph(op_graph, &mut sym_to_combined, rng, Some(group_id));
        }

        // 4. Map outputs
        for &output_id in &self.ordered_outputs {
            combined.add_output(sym_to_combined[&output_id], output_id);
        }

        combined
    }

    /// Generate a MilliOpGraph with optional backward pass and optimizer.
    ///
    /// With `options.backward = None`, equivalent to `generate_milli_graph`.
    /// With backward options, composes loss graph, generates backward pass,
    /// and populates TrainingMetadata.
    pub fn generate_milli_graph_with_options(
        &self,
        options: &crate::milli_graph::MilliGraphGenOptions,
        rng: &mut impl Rng,
    ) -> Result<crate::milli_graph::MilliOpGraph, crate::milli_graph::MilliGraphGenError> {
        use crate::graph::{Graph, Node};
        use crate::milli_graph::{
            BackwardGenContext, LossInputSource, MilliGraphGenError, MilliOpGraph, MilliOpGroup,
            MilliOpPhase, TrainingMetadata, generate_milli_backward,
        };
        use crate::tensor_info::TensorInfo;

        if options.optimizer.is_some() && options.backward.is_none() {
            return Err(MilliGraphGenError::OptimizerWithoutBackward);
        }

        let mut combined = MilliOpGraph::new_empty(rng);
        let mut sym_to_combined: HashMap<GlobalId, GlobalId> = HashMap::new();

        // 1a. Map graph inputs
        for &input_id in &self.ordered_inputs {
            let internal = combined.add_input_with_id(input_id, rng);
            sym_to_combined.insert(input_id, internal);
        }

        // 1b. Map standalone constants
        for const_id in self.constant_link_ids() {
            let internal = combined.add_input_with_id(const_id, rng);
            sym_to_combined.insert(const_id, internal);
        }

        // 2. Generate forward pass — walk ops in topological order
        let topo_order = self.topological_order_vec();
        for &op_id in &topo_order {
            let graph_op = &self.operations[&op_id];
            let label = graph_op
                .name
                .clone()
                .unwrap_or_else(|| graph_op.op.op_kind());
            let group = MilliOpGroup {
                id: GlobalId::new(rng),
                source_op: Some(op_id),
                source_graph: Some(self.global_id),
                phase: MilliOpPhase::Forward,
                label: Some(label),
                ..Default::default()
            };
            let group_id = combined.create_group(group);

            let op_graph = graph_op.op.get_milli_op_graph(rng);
            combined.merge_graph(op_graph, &mut sym_to_combined, rng, Some(group_id));
        }

        // 3. Compose loss graph and generate backward (if training)
        if let Some(ref backward_opts) = options.backward {
            let loss_group = combined.create_group(MilliOpGroup {
                id: GlobalId::new(rng),
                phase: MilliOpPhase::Loss,
                label: Some("loss".into()),
                ..Default::default()
            });

            // Wire loss graph inputs to forward outputs / external inputs
            let mut loss_wiring: HashMap<GlobalId, GlobalId> = HashMap::new();
            let mut external_inputs: Vec<GlobalId> = Vec::new();
            for wire in &backward_opts.loss_wiring {
                let combined_id = match &wire.source {
                    LossInputSource::ForwardOutput(sym_id) => sym_to_combined[sym_id],
                    LossInputSource::ExternalInput { .. } => {
                        let ext_id = combined.add_input(rng);
                        external_inputs.push(ext_id);
                        ext_id
                    }
                };
                loss_wiring.insert(wire.loss_input, combined_id);
            }
            combined.merge_graph(
                backward_opts.loss_graph.clone(),
                &mut loss_wiring,
                rng,
                Some(loss_group),
            );
            let loss_tensor = loss_wiring[&backward_opts.loss_output];

            // 4. Generate backward pass

            // 4a. Backward through loss graph (milli-op level differentiation)
            let loss_grad = combined.add_scalar_one(rng);
            let mut grad_map: HashMap<GlobalId, GlobalId> = HashMap::new();
            grad_map.insert(loss_tensor, loss_grad);

            let loss_grads = generate_milli_backward(&mut combined, loss_group, &grad_map, rng);
            grad_map.extend(loss_grads);

            // 4b. Backward through SymbolicGraph ops (Operation-level)
            for &op_id in topo_order.iter().rev() {
                let graph_op = &self.operations[&op_id];

                // Collect output gradients for this op
                let output_grads: HashMap<GlobalId, GlobalId> = graph_op
                    .op
                    .outputs()
                    .filter_map(|out_id| {
                        let combined_id = sym_to_combined.get(&out_id)?;
                        grad_map.get(combined_id).map(|g| (*combined_id, *g))
                    })
                    .collect();

                if output_grads.is_empty() {
                    continue;
                }
                if backward_opts.stop_gradients.contains(&op_id) {
                    continue;
                }

                // Build tensor shape map from ONNXTensorInfo
                let tensor_shapes: HashMap<GlobalId, TensorInfo> = graph_op
                    .op
                    .inputs()
                    .chain(graph_op.op.outputs())
                    .filter_map(|id| {
                        let info = self.get_tensor_info(id)?;
                        let shape = info.shape.as_ref()?;
                        Some((id, TensorInfo::from_shape_scalars(shape)))
                    })
                    .collect();

                let ctx = BackwardGenContext {
                    output_grads,
                    forward_inputs: graph_op
                        .op
                        .inputs()
                        .map(|id| sym_to_combined[&id])
                        .collect(),
                    forward_outputs: graph_op
                        .op
                        .outputs()
                        .map(|id| sym_to_combined[&id])
                        .collect(),
                    tensor_shapes,
                };

                if let Some(backward_result) = graph_op.op.get_backward_milli_ops(&ctx, rng) {
                    let bwd_group = combined.create_group(MilliOpGroup {
                        id: GlobalId::new(rng),
                        source_op: Some(op_id),
                        source_graph: Some(self.global_id),
                        phase: MilliOpPhase::Backward,
                        backward_of: Some(op_id),
                        label: Some(format!("{}_backward", graph_op.op.op_kind())),
                        ..Default::default()
                    });

                    // Build wiring map: combined-space identity mappings
                    let mut bwd_wiring: HashMap<GlobalId, GlobalId> = HashMap::new();
                    for &combined_id in sym_to_combined.values() {
                        bwd_wiring.insert(combined_id, combined_id);
                    }
                    for &grad_id in grad_map.values() {
                        bwd_wiring.insert(grad_id, grad_id);
                    }
                    combined.merge_graph(
                        backward_result.graph,
                        &mut bwd_wiring,
                        rng,
                        Some(bwd_group),
                    );

                    // Accumulate gradients for each forward input
                    for fwd_input_id in &backward_result.differentiable_inputs {
                        let remapped_grad = bwd_wiring[fwd_input_id];
                        grad_map
                            .entry(*fwd_input_id)
                            .and_modify(|existing| {
                                let sum = crate::milli_graph::ops::SimpleBinary::add(
                                    &mut combined,
                                    *existing,
                                    remapped_grad,
                                    rng,
                                );
                                *existing = sum;
                            })
                            .or_insert(remapped_grad);
                    }
                }
            }

            // Record training metadata
            let mut training_meta = TrainingMetadata {
                loss: Some(loss_tensor),
                external_inputs,
                ..TrainingMetadata::default()
            };
            for param in &backward_opts.trainable_params {
                let combined_param = sym_to_combined[param];
                if let Some(&grad) = grad_map.get(&combined_param) {
                    training_meta.param_to_grad.insert(combined_param, grad);
                }
            }

            // 5. Generate optimizer (if requested)
            if let Some(ref optim_opts) = options.optimizer {
                crate::milli_graph::generate_optimizer_ops(
                    &mut combined,
                    &mut training_meta,
                    optim_opts,
                    rng,
                );
            }

            combined.training_metadata = Some(training_meta.clone());

            // Set outputs: loss + forward outputs + updated params + optimizer state
            let mut output_ids: Vec<(GlobalId, GlobalId)> = vec![];
            output_ids.push((loss_tensor, loss_tensor));
            // Include forward outputs
            for &output_id in &self.ordered_outputs {
                let combined_id = sym_to_combined[&output_id];
                output_ids.push((combined_id, output_id));
            }
            // Include gradients as outputs
            for param in &backward_opts.trainable_params {
                if let Some(&grad) = grad_map.get(&sym_to_combined[param]) {
                    output_ids.push((grad, grad));
                }
            }
            // Include updated parameters as outputs
            for &new_param in training_meta.param_to_new_param.values() {
                output_ids.push((new_param, new_param));
            }
            // Include optimizer state outputs
            for &state_out in training_meta.optimizer_state_outputs.values() {
                output_ids.push((state_out, state_out));
            }
            // Include global state outputs (e.g., updated timestep)
            for &global_out in training_meta.global_state_outputs.values() {
                output_ids.push((global_out, global_out));
            }
            combined.set_output_map(output_ids);
        } else {
            // Forward-only: map outputs normally
            for &output_id in &self.ordered_outputs {
                combined.add_output(sym_to_combined[&output_id], output_id);
            }
        }

        Ok(combined)
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        eval_backend: &mut EvalBackend,
    ) -> Result<HashMap<GlobalId, NumericTensor<DynRank>>, EvalError> {
        let mut active_tensors: HashMap<GlobalId, NumericTensor<DynRank>> = inputs.clone();

        let ops = self.get_operations();
        let mut remaining_ops_to_complete: Vec<GlobalId> = ops.keys().copied().collect();
        let mut total_ops_completed: Vec<GlobalId> = vec![];
        loop {
            let mut ops_completed_now = vec![];

            for op_id in &remaining_ops_to_complete {
                let GraphOperation { name: _, op } = ops.get(op_id).unwrap();
                let input_ids = op.inputs();
                let mut input_values = HashMap::new();
                // Collect all inputs, abort if we can't do this one yet
                let mut failed_to_fetch = false;
                for tensor_id in input_ids {
                    if let Some(value) = active_tensors.get(&tensor_id) {
                        // Validate shape and dtype
                        if let Some(tensor_info) = self.get_tensor_info(tensor_id) {
                            check_tensor_matches(value, tensor_info)?;
                        }
                        input_values.insert(tensor_id, value.clone());
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
    tensors_by_name: HashMap<String, GlobalId>,
    unknown_dimensions_by_name: HashMap<String, SymbolicScalarTyped<u64>>,
    symbolic_resolver: SymbolicResolver,
    tensor_store: TensorStore,
}

impl SymbolicGraphMutator {
    pub fn new(rng: &mut impl Rng) -> Self {
        Self::from_graph(SymbolicGraph::new(rng), TensorStore::new())
    }

    pub fn get_inner(self) -> (SymbolicGraph, TensorStore) {
        (self.graph.unwrap(), self.tensor_store)
    }

    pub fn from_graph(graph: SymbolicGraph, tensor_store: TensorStore) -> Self {
        let mut dimension_resolver = SymbolicResolver::new();
        for dim in graph.unknown_dimensions.values() {
            dimension_resolver.update_last_assigned(dim.clone())
        }
        Self {
            tensors_by_name: graph.get_tensors_by_name(),
            graph: Some(graph),
            symbolic_resolver: dimension_resolver,
            unknown_dimensions_by_name: HashMap::new(),
            tensor_store,
        }
    }

    pub fn from_onnx_bytes(
        onnx_bytes: &[u8],
        rng: &mut impl Rng,
        base_dir: Option<&std::path::Path>,
    ) -> Result<Self, ONNXDecodingError> {
        let model = onnx::ModelProto::decode(onnx_bytes)
            .map_err(|x| ONNXDecodingError::ProtobufDecodeError(anyhow::Error::from(x)))?;
        Self::from_onnx_model_proto(model, rng, base_dir)
    }

    pub(crate) fn new_unknown_tensor(
        &mut self,
        inner_graph: &mut SymbolicGraph,
        name: &str,
        tensor_type: TensorType,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let global_id = GlobalId::new(rng);
        let tensor = ONNXTensorInfo {
            onnx_name: Some(name.to_string()),
            tensor_type,
            dtype: None,
            shape: None,
            global_id,
        };
        inner_graph.tensors.insert(global_id, tensor);
        self.tensors_by_name.insert(name.to_string(), global_id);
        global_id
    }

    pub(crate) fn new_tensor_from_tensor_info(
        &mut self,
        inner_graph: &mut SymbolicGraph,
        tensor_info: &onnx::ValueInfoProto,
        tensor_type: TensorType,
        rng: &mut impl Rng,
    ) -> Result<GlobalId, ONNXDecodingError> {
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

        let global_id = GlobalId::new(rng);
        self.tensors_by_name.insert(name.clone(), global_id);
        inner_graph.tensors.insert(
            global_id,
            ONNXTensorInfo {
                onnx_name: Some(name),
                dtype: Some(dtype),
                shape: Some(dimensions),
                tensor_type,
                global_id,
            },
        );
        Ok(global_id)
    }

    pub(crate) fn new_constant_tensor(
        &mut self,
        inner_graph: &mut SymbolicGraph,
        value: NDArrayNumericTensor<DynRank>,
        name: Option<String>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let mut shape = Vec::new();
        for s in value.shape() {
            shape.push(ScalarInfoTyped::Numeric(s))
        }

        let global_id = GlobalId::new(rng);
        inner_graph.tensors.insert(
            global_id,
            ONNXTensorInfo {
                onnx_name: name.clone(),
                dtype: Some(value.dtype()),
                shape: Some(shape),
                tensor_type: TensorType::Constant(StoredOrNotTensor::NotStored(value)),
                global_id,
            },
        );
        if let Some(name) = name {
            self.tensors_by_name.insert(name, global_id);
        }

        global_id
    }

    pub(crate) fn new_stored_tensor(
        &mut self,
        inner_graph: &mut SymbolicGraph,
        id: TensorStoreTensorId,
        name: Option<String>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let tensor_ref = self.tensor_store.get_tensor(id).unwrap();

        let mut shape = Vec::new();
        for s in tensor_ref.shape() {
            shape.push(ScalarInfoTyped::Numeric(s))
        }

        let global_id = GlobalId::new(rng);
        inner_graph.tensors.insert(
            global_id,
            ONNXTensorInfo {
                onnx_name: name.clone(),
                dtype: Some(tensor_ref.dtype()),
                shape: Some(shape),
                tensor_type: TensorType::Constant(StoredOrNotTensor::Stored(id)),
                global_id,
            },
        );
        if let Some(name) = name {
            self.tensors_by_name.insert(name, global_id);
        }

        global_id
    }

    pub(crate) fn new_node_from_onnx_node(
        &mut self,
        inner_graph: &mut SymbolicGraph,
        core_opset_version: usize,
        onnx_node: &onnx::NodeProto,
        rng: &mut impl Rng,
    ) -> Result<(), ONNXDecodingError> {
        let mut input_tensors = vec![];
        for input in &onnx_node.input {
            input_tensors.push(if input.is_empty() {
                None
            } else if let Some(tensor_id) = self.tensors_by_name.get(input) {
                Some(*tensor_id)
            } else {
                Some(self.new_unknown_tensor(inner_graph, input, TensorType::Intermediate, rng))
            });
        }

        let mut output_tensors = vec![];
        for output in &onnx_node.output {
            output_tensors.push(if output.is_empty() {
                None
            } else if let Some(tensor_id) = self.tensors_by_name.get(output) {
                Some(*tensor_id)
            } else {
                Some(self.new_unknown_tensor(inner_graph, output, TensorType::Intermediate, rng))
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
                rng,
            )?)),
            "Min" => Some(AnyOperation::Min(ops::MinOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                rng,
            )?)),
            "Add" => Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichBinaryOperation::Add,
                rng,
            )?)),
            "Mod" => Some(AnyOperation::Modulo(ops::ModuloOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                rng,
            )?)),
            "Sub" => Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichBinaryOperation::Sub,
                rng,
            )?)),
            "Mul" => Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichBinaryOperation::Mul,
                rng,
            )?)),
            "Div" => Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichBinaryOperation::Div,
                rng,
            )?)),
            "MatMul" => Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichBinaryOperation::MatMul,
                rng,
            )?)),
            "Equal" => Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichBinaryOperation::Equal,
                rng,
            )?)),
            "Greater" => Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichBinaryOperation::Greater,
                rng,
            )?)),
            "GreaterOrEqual" => Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichBinaryOperation::GreaterOrEqual,
                rng,
            )?)),
            "Less" => Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichBinaryOperation::Less,
                rng,
            )?)),
            "LessOrEqual" => Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichBinaryOperation::LessOrEqual,
                rng,
            )?)),
            "And" => Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichBinaryOperation::And,
                rng,
            )?)),
            "Or" => Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichBinaryOperation::Or,
                rng,
            )?)),
            "Xor" => Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichBinaryOperation::Xor,
                rng,
            )?)),
            "BitwiseAnd" => Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichBinaryOperation::BitwiseAnd,
                rng,
            )?)),
            "BitwiseOr" => Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichBinaryOperation::BitwiseOr,
                rng,
            )?)),
            "BitwiseXor" => Some(AnyOperation::Binary(ops::BinaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichBinaryOperation::BitwiseXor,
                rng,
            )?)),
            "Relu" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Relu,
                rng,
            )?)),
            "Sigmoid" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Sigmoid,
                rng,
            )?)),
            "Asin" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Trig(TrigOp::Asin),
                rng,
            )?)),
            "Asinh" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Trig(TrigOp::Asinh),
                rng,
            )?)),
            "Acos" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Trig(TrigOp::Acos),
                rng,
            )?)),
            "Acosh" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Trig(TrigOp::Acosh),
                rng,
            )?)),
            "Atan" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Trig(TrigOp::Atan),
                rng,
            )?)),
            "Atanh" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Trig(TrigOp::Atanh),
                rng,
            )?)),
            "Sin" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Trig(TrigOp::Sin),
                rng,
            )?)),
            "Sinh" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Trig(TrigOp::Sinh),
                rng,
            )?)),
            "Cos" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Trig(TrigOp::Cos),
                rng,
            )?)),
            "Cosh" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Trig(TrigOp::Cosh),
                rng,
            )?)),
            "Tan" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Trig(TrigOp::Tan),
                rng,
            )?)),
            "Tanh" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Trig(TrigOp::Tanh),
                rng,
            )?)),
            "Exp" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Exp,
                rng,
            )?)),
            "Log" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Log,
                rng,
            )?)),
            "Sqrt" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Sqrt,
                rng,
            )?)),
            "BitwiseNot" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::BitwiseNot,
                rng,
            )?)),
            "Not" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Not,
                rng,
            )?)),
            "Softplus" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Softplus,
                rng,
            )?)),
            "Neg" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Neg,
                rng,
            )?)),
            "Abs" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Abs,
                rng,
            )?)),
            "Sign" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Sign,
                rng,
            )?)),
            "IsNaN" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::IsNan,
                rng,
            )?)),
            "Erf" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Erf,
                rng,
            )?)),
            "IsInf" => Some(AnyOperation::IsInf(ops::IsInfOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                rng,
            )?)),
            "Floor" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Floor,
                rng,
            )?)),
            "Ceil" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Ceil,
                rng,
            )?)),
            "Round" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Round,
                rng,
            )?)),
            "Clip" => Some(AnyOperation::Clip(ops::ClipOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                rng,
            )?)),
            "Reciprocal" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::Reciprocal,
                rng,
            )?)),
            "Size" => Some(AnyOperation::Size(ops::SizeOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                rng,
            )?)),
            "NonZero" => Some(AnyOperation::Unary(ops::UnaryOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                ops::WhichUnaryOperation::NonZero,
                rng,
            )?)),
            "LpNormalization" => Some(AnyOperation::LpNormalization(
                ops::LpNormalizationOperation::from_onnx(
                    &input_tensors,
                    &output_tensors,
                    &onnx_node.attribute,
                    rng,
                )?,
            )),
            "GroupNormalization" => Some(AnyOperation::GroupNormalization(
                ops::GroupNormalizationOperation::from_onnx(
                    &input_tensors,
                    &output_tensors,
                    &onnx_node.attribute,
                    rng,
                )?,
            )),
            "RMSNormalization" => Some(AnyOperation::RMSNormalization(
                ops::RMSNormalizationOperation::from_onnx(
                    &input_tensors,
                    &output_tensors,
                    &onnx_node.attribute,
                    rng,
                )?,
            )),
            "LayerNormalization" => Some(AnyOperation::LayerNormalization(
                ops::LayerNormalizationOperation::from_onnx(
                    &input_tensors,
                    &output_tensors,
                    &onnx_node.attribute,
                    rng,
                )?,
            )),
            "CumSum" => Some(AnyOperation::CumSum(ops::CumSumOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                rng,
            )?)),
            "Gather" => Some(AnyOperation::Gather(ops::GatherOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                rng,
            )?)),
            "Cast" => Some(AnyOperation::Cast(ops::CastOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                rng,
            )?)),
            "CastLike" => Some(AnyOperation::CastLike(ops::CastLikeOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                rng,
            )?)),
            "Pow" => Some(AnyOperation::Pow(ops::PowOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                rng,
            )?)),
            "ReduceMean" => Some(AnyOperation::ReduceMean(
                ops::ReduceMeanOperation::from_onnx(
                    &input_tensors,
                    &output_tensors,
                    &onnx_node.attribute,
                    rng,
                )?,
            )),
            "ReduceSum" => Some(AnyOperation::ReduceSum(ops::ReduceSumOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                rng,
            )?)),
            "ReduceProd" => Some(AnyOperation::ReduceProd(
                ops::ReduceProdOperation::from_onnx(
                    &input_tensors,
                    &output_tensors,
                    &onnx_node.attribute,
                    rng,
                )?,
            )),
            "ReduceMin" => Some(AnyOperation::ReduceMin(ops::ReduceMinOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                rng,
            )?)),
            "ReduceMax" => Some(AnyOperation::ReduceMax(ops::ReduceMaxOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                rng,
            )?)),
            "Flatten" => Some(AnyOperation::Flatten(ops::FlattenOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                rng,
            )?)),
            "Expand" => Some(AnyOperation::Expand(ops::ExpandOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                rng,
            )?)),
            "Squeeze" => Some(AnyOperation::Squeeze(ops::SqueezeOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                rng,
            )?)),
            "Unsqueeze" => Some(AnyOperation::Unsqueeze(ops::UnsqueezeOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                rng,
            )?)),
            "Transpose" => Some(AnyOperation::Transpose(ops::TransposeOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                rng,
            )?)),
            "Reshape" => Some(AnyOperation::Reshape(ops::ReshapeOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                rng,
            )?)),
            "Shape" => Some(AnyOperation::Shape(ops::ShapeOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                rng,
            )?)),
            "Range" => Some(AnyOperation::Range(ops::RangeOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                rng,
            )?)),
            "Concat" => Some(AnyOperation::Concat(ops::ConcatOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                rng,
            )?)),
            "ConstantOfShape" => Some(AnyOperation::ConstantOfShape(
                ops::ConstantOfShapeOperation::from_onnx(
                    &input_tensors,
                    &output_tensors,
                    &onnx_node.attribute,
                    rng,
                )?,
            )),
            "Gemm" => Some(AnyOperation::Gemm(ops::GemmOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                rng,
            )?)),
            "Slice" => Some(AnyOperation::Slice(ops::SliceOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                rng,
            )?)),
            "Where" => Some(AnyOperation::Where(ops::WhereOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                rng,
            )?)),
            "Softmax" => Some(AnyOperation::Softmax(ops::SoftmaxOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                rng,
            )?)),
            "LogSoftmax" => Some(AnyOperation::LogSoftmax(
                ops::LogSoftmaxOperation::from_onnx(
                    &input_tensors,
                    &output_tensors,
                    &onnx_node.attribute,
                    rng,
                )?,
            )),
            "Split" => Some(AnyOperation::Split(ops::SplitOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                rng,
            )?)),
            "Constant" => Some(AnyOperation::Constant(ops::ConstantOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                rng,
            )?)),
            "Identity" => Some(AnyOperation::Identity(ops::IdentityOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                rng,
            )?)),
            "Conv" => Some(AnyOperation::Conv(ops::ConvOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                rng,
            )?)),
            "InstanceNormalization" => Some(AnyOperation::InstanceNormalization(
                ops::InstanceNormalizationOperation::from_onnx(
                    &input_tensors,
                    &output_tensors,
                    &onnx_node.attribute,
                    rng,
                )?,
            )),
            "Resize" => Some(AnyOperation::Resize(ops::ResizeOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                rng,
            )?)),
            "Pad" => Some(AnyOperation::Pad(ops::PadOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                rng,
            )?)),
            "RandomNormalLike" => Some(AnyOperation::RandomNormalLike(
                ops::RandomNormalLikeOperation::from_onnx(
                    &input_tensors,
                    &output_tensors,
                    &onnx_node.attribute,
                    rng,
                )?,
            )),
            "ArgMax" => Some(AnyOperation::ArgMax(ops::ArgMaxOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                rng,
            )?)),
            "ArgMin" => Some(AnyOperation::ArgMin(ops::ArgMinOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                rng,
            )?)),
            "RotaryEmbedding" => Some(AnyOperation::RotaryEmbedding(
                ops::RotaryEmbeddingOperation::from_onnx(
                    &input_tensors,
                    &output_tensors,
                    &onnx_node.attribute,
                    rng,
                )?,
            )),
            "If" => Some(AnyOperation::If(ops::IfOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                self,
                core_opset_version,
                rng,
            )?)),
            "Scan" => Some(AnyOperation::Scan(ops::ScanOperation::from_onnx(
                &input_tensors,
                &output_tensors,
                &onnx_node.attribute,
                self,
                core_opset_version,
                rng,
            )?)),
            x => Err(ONNXDecodingError::UnsupportedONNXType(x.to_string()))?,
        };

        if let Some(op) = new_op {
            let op = GraphOperation { name, op };

            inner_graph.operations.insert(op.global_id(), op);
        }

        Ok(())
    }

    pub fn from_onnx_model_proto(
        model_proto: onnx::ModelProto,
        rng: &mut impl Rng,
        base_dir: Option<&std::path::Path>,
    ) -> Result<Self, ONNXDecodingError> {
        let mut core_opset_version = 0;
        for opset_proto in model_proto.opset_import {
            if opset_proto.domain.is_empty() {
                core_opset_version = opset_proto.version as usize;
            }
        }

        let onnx_graph = model_proto
            .graph
            .ok_or(ONNXDecodingError::MissingField("graph"))?;
        let mut graph_mutator = Self::new(rng);
        let mut inner_graph = graph_mutator.graph.take().unwrap();

        inner_graph.populate(
            &mut graph_mutator,
            &onnx_graph,
            core_opset_version,
            rng,
            base_dir,
        )?;
        graph_mutator.graph = Some(inner_graph);

        Ok(graph_mutator)
    }
}

impl Graph for SymbolicGraph {
    type Error = ();
    type AnyNode = GraphOperation;
    type AnyLink = ONNXTensorInfo;

    fn global_id(&self) -> GlobalId {
        self.global_id
    }

    fn node_ids(&self) -> impl Iterator<Item = GlobalId> {
        self.operations.keys().cloned()
    }

    fn inner_link_ids(&self) -> impl Iterator<Item = GlobalId> {
        self.tensors.keys().cloned()
    }

    fn get_node_by_id(&self, id: &GlobalId) -> Option<&Self::AnyNode> {
        self.operations.get(id)
    }

    fn get_link_by_id(&self, id: &GlobalId) -> Option<&Self::AnyLink> {
        self.tensors.get(id)
    }

    fn input_link_ids(&self) -> impl Iterator<Item = (GlobalId, GlobalId)> {
        self.tensors.iter().filter_map(|(id, info)| {
            if let TensorType::Input(_) = info.tensor_type {
                Some((*id, *id))
            } else {
                None
            }
        })
    }

    fn output_link_ids(&self) -> impl Iterator<Item = (GlobalId, GlobalId)> {
        self.tensors.iter().filter_map(|(id, info)| {
            if let TensorType::Output = info.tensor_type {
                Some((*id, *id))
            } else {
                None
            }
        })
    }

    fn constant_link_ids(&self) -> impl Iterator<Item = GlobalId> {
        self.tensors.iter().filter_map(|(id, info)| {
            if let TensorType::Constant(_) = info.tensor_type {
                Some(*id)
            } else {
                None
            }
        })
    }

    fn topological_order(&self) -> Option<Box<dyn Iterator<Item = GlobalId>>> {
        Some(Box::new(self.topological_order_vec().into_iter()))
    }
}

impl Link for ONNXTensorInfo {
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn label(&self) -> Option<String> {
        self.onnx_name.clone()
    }
}

impl LinkMetadata for ONNXTensorInfo {
    fn dtype(&self) -> Option<DType> {
        self.dtype
    }

    fn shape(&self) -> Option<Vec<ScalarInfoTyped<u64>>> {
        self.shape.clone()
    }

    fn category(&self) -> Option<LinkCategory> {
        Some(match &self.tensor_type {
            TensorType::Input(_) => LinkCategory::Input,
            TensorType::Output => LinkCategory::Output,
            TensorType::Intermediate => LinkCategory::Intermediate,
            TensorType::Constant(_) => LinkCategory::Constant,
        })
    }
}

impl Node for GraphOperation {
    type OpKind = String;

    fn op_kind(&self) -> Self::OpKind {
        self.op.op_kind()
    }

    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        self.op.inputs()
    }

    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        self.op.outputs()
    }

    fn global_id(&self) -> GlobalId {
        self.op.global_id()
    }

    fn label(&self) -> Option<String> {
        self.name.clone()
    }
}

impl NodeMetadata for GraphOperation {
    fn parameters(&self) -> Vec<Property> {
        self.op.parameters()
    }

    fn has_subgraph(&self) -> bool {
        !self.op.get_sub_graphs().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::eval_backend::EvalBackend;

    /// Helper: load an ONNX node test, evaluate via SymbolicGraph::eval and
    /// via generate_milli_graph, assert outputs match.
    fn test_equivalence_for_onnx_dir(dir: &str) {
        let dir_path = std::path::Path::new(dir);
        let model_path = dir_path.join("model.onnx");
        if !model_path.exists() {
            panic!("Model not found: {}", model_path.display());
        }

        let model_bytes = std::fs::read(&model_path).unwrap();
        let rng = &mut rand::rng();
        let (graph, tensor_store) = SymbolicGraphMutator::from_onnx_bytes(&model_bytes, rng, None)
            .unwrap()
            .get_inner();

        // Load test data set 0
        let test_data_dir = dir_path.join("test_data_set_0");
        let mut user_inputs: HashMap<
            String,
            crate::numeric_tensor::NumericTensor<crate::tensor_rank::DynRank>,
        > = HashMap::new();
        for entry in std::fs::read_dir(&test_data_dir).unwrap() {
            let entry = entry.unwrap();
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("input_") && name.ends_with(".pb") {
                let data = std::fs::read(entry.path()).unwrap();
                let tensor_proto = crate::onnx::TensorProto::decode(data.as_slice()).unwrap();
                let tensor = crate::backends::ndarray_backend::NDArrayNumericTensor::<
                    crate::tensor_rank::DynRank,
                >::try_from(&tensor_proto)
                .unwrap();
                let idx: usize = name
                    .strip_prefix("input_")
                    .unwrap()
                    .strip_suffix(".pb")
                    .unwrap()
                    .parse()
                    .unwrap();
                // Map by input ordering
                let input_id = graph.ordered_inputs[idx];
                let input_name = graph.get_tensor_name(input_id).unwrap().to_string();
                user_inputs.insert(input_name, tensor.into());
            }
        }

        // Build full inputs (initialized tensors + user inputs)
        let initialized_tensors = graph.get_initialized_tensors(&tensor_store);
        let mut all_inputs: HashMap<
            GlobalId,
            crate::numeric_tensor::NumericTensor<crate::tensor_rank::DynRank>,
        > = initialized_tensors;
        let tensors_by_name = graph.get_tensors_by_name();
        for (name, tensor) in &user_inputs {
            let tensor_id = tensors_by_name[name];
            all_inputs.insert(tensor_id, tensor.clone());
        }

        // Evaluate via SymbolicGraph::eval
        let mut backend = EvalBackend::NDArray;
        let symbolic_result = graph.eval(&all_inputs, &mut backend).unwrap();

        // Evaluate via generate_milli_graph
        let combined = graph.generate_milli_graph(rng);
        let milli_result: HashMap<_, _> = combined
            .eval(&all_inputs, &mut (), &mut backend)
            .unwrap()
            .collect();

        // Compare outputs
        for &output_id in &graph.ordered_outputs {
            let sym_tensor = &symbolic_result[&output_id];
            let milli_tensor = &milli_result[&output_id];

            assert_eq!(
                sym_tensor.shape(),
                milli_tensor.shape(),
                "Shape mismatch for output {:?}",
                output_id
            );
            assert_eq!(
                sym_tensor.dtype(),
                milli_tensor.dtype(),
                "DType mismatch for output {:?}",
                output_id
            );

            let sym_values: Vec<f64> = sym_tensor
                .cast(crate::dtype::DType::F64, &mut EvalBackend::NDArray)
                .unwrap()
                .to_ndarray()
                .unwrap()
                .flatten()
                .try_to_vec()
                .unwrap();
            let milli_values: Vec<f64> = milli_tensor
                .cast(crate::dtype::DType::F64, &mut EvalBackend::NDArray)
                .unwrap()
                .to_ndarray()
                .unwrap()
                .flatten()
                .try_to_vec()
                .unwrap();

            for (i, (s, m)) in sym_values.iter().zip(milli_values.iter()).enumerate() {
                let diff = (s - m).abs();
                let tol = 1e-5 + 1e-3 * s.abs();
                assert!(
                    diff <= tol,
                    "Value mismatch at index {} for output {:?}: symbolic={}, milli={}",
                    i,
                    output_id,
                    s,
                    m
                );
            }
        }
    }

    #[test]
    fn test_topological_order_basic() {
        // Build a small symbolic graph and verify topological order
        // We'll use the actual ONNX add test which is simple
        let dir = "libs/onnx/onnx/backend/test/data/node/test_add";
        if !std::path::Path::new(dir).join("model.onnx").exists() {
            eprintln!("Skipping test_topological_order_basic: ONNX test data not found");
            return;
        }
        let model_bytes = std::fs::read(std::path::Path::new(dir).join("model.onnx")).unwrap();
        let rng = &mut rand::rng();
        let (graph, _) = SymbolicGraphMutator::from_onnx_bytes(&model_bytes, rng, None)
            .unwrap()
            .get_inner();

        let topo = graph.topological_order_vec();
        // Should have exactly the number of operations
        assert_eq!(topo.len(), graph.operations.len());

        // Every op should appear exactly once
        let topo_set: HashSet<_> = topo.iter().copied().collect();
        assert_eq!(topo_set.len(), topo.len());
        for op_id in graph.operations.keys() {
            assert!(topo_set.contains(op_id));
        }
    }

    #[test]
    fn test_generate_milli_graph_add() {
        let dir = "libs/onnx/onnx/backend/test/data/node/test_add";
        if !std::path::Path::new(dir).join("model.onnx").exists() {
            eprintln!("Skipping test: ONNX test data not found");
            return;
        }
        test_equivalence_for_onnx_dir(dir);
    }

    #[test]
    fn test_generate_milli_graph_relu() {
        let dir = "libs/onnx/onnx/backend/test/data/node/test_relu";
        if !std::path::Path::new(dir).join("model.onnx").exists() {
            eprintln!("Skipping test: ONNX test data not found");
            return;
        }
        test_equivalence_for_onnx_dir(dir);
    }

    #[test]
    fn test_generate_milli_graph_matmul_2d() {
        let dir = "libs/onnx/onnx/backend/test/data/node/test_matmul_2d";
        if !std::path::Path::new(dir).join("model.onnx").exists() {
            eprintln!("Skipping test: ONNX test data not found");
            return;
        }
        test_equivalence_for_onnx_dir(dir);
    }

    #[test]
    fn test_generate_milli_graph_reshape() {
        let dir = "libs/onnx/onnx/backend/test/data/node/test_reshape_extended_dims";
        if !std::path::Path::new(dir).join("model.onnx").exists() {
            eprintln!("Skipping test: ONNX test data not found");
            return;
        }
        test_equivalence_for_onnx_dir(dir);
    }

    #[test]
    fn test_generate_milli_graph_with_backward_plumbing() {
        // Test that backward orchestration runs without panic.
        // No ops implement backward() yet, so no actual gradients are produced,
        // but the loss graph is composed, generate_milli_backward runs, and
        // TrainingMetadata is populated.
        use crate::milli_graph::{
            BackwardGenOptions, LossInputSource, LossWiring, MilliGraphGenOptions, MilliOpGraph,
        };

        let dir = "libs/onnx/onnx/backend/test/data/node/test_add";
        if !std::path::Path::new(dir).join("model.onnx").exists() {
            eprintln!("Skipping test: ONNX test data not found");
            return;
        }

        let model_bytes = std::fs::read(std::path::Path::new(dir).join("model.onnx")).unwrap();
        let rng = &mut rand::rng();
        let (graph, _tensor_store) = SymbolicGraphMutator::from_onnx_bytes(&model_bytes, rng, None)
            .unwrap()
            .get_inner();

        // Create a simple MSE loss graph
        let (loss_graph, loss_info) = MilliOpGraph::mse_loss(rng);

        // The add model has inputs and outputs. Wire the first output to loss predictions.
        let forward_output = graph.ordered_outputs[0];

        let options = MilliGraphGenOptions {
            backward: Some(BackwardGenOptions {
                loss_graph,
                loss_wiring: vec![
                    LossWiring {
                        loss_input: loss_info.predictions_input,
                        source: LossInputSource::ForwardOutput(forward_output),
                    },
                    LossWiring {
                        loss_input: loss_info.targets_input,
                        source: LossInputSource::ExternalInput {
                            name: "targets".into(),
                        },
                    },
                ],
                loss_output: loss_info.loss_output,
                trainable_params: graph.ordered_inputs.clone(),
                stop_gradients: std::collections::HashSet::new(),
            }),
            optimizer: None,
        };

        let result = graph.generate_milli_graph_with_options(&options, rng);
        let combined = result.unwrap();

        // Verify TrainingMetadata was populated
        let meta = combined.training_metadata.as_ref().unwrap();
        assert!(meta.loss.is_some());
        assert_eq!(meta.external_inputs.len(), 1); // targets

        // The default get_backward_milli_ops() should produce gradients for
        // both inputs of the Add operation.
        // param_to_grad keys are combined-space IDs, so check count matches.
        assert_eq!(
            meta.param_to_grad.len(),
            graph.ordered_inputs.len(),
            "Expected gradient for each trainable param. Got {} gradients for {} params",
            meta.param_to_grad.len(),
            graph.ordered_inputs.len(),
        );
    }
}
