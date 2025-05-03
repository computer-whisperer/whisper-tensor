pub mod operators;
pub mod weights;
pub mod tensor;
mod node;
pub mod pytorch;

use std::collections::{HashMap, HashSet};
use std::path::{PathBuf};
use std::sync::Arc;
use tensor::*;
use node::*;
use serde::{Deserialize, Serialize};
use crate::onnx::StringStringEntryProto;
use crate::weights::{WeightExternalOutputManager};

pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Bad input shape: {0}")]
    InputShapeError(Shape),
    #[error("Shape mismatch: {0} != {1}")]
    ShapeMismatchError(Shape, Shape),
    #[error("Incompatible shapes: {0}, {1}")]
    IncompatibleShapeError(Shape, Shape),
    #[error("DType mismatch: {0} != {1}")]
    DTypeMismatchError(DType, DType),
    #[error("Invalid input")]
    InvalidInputError,
    #[error("Invalid dtype")]
    UnsupportedDTypeError,
    #[error("Name conflict: {0}")]
    NameConflictError(String),
    #[error("No such tensor: {0}")]
    NoSuchTensorError(String),
    #[error("Unresolved dimension")]
    UnresolvedDimensionError,
    #[error("Invalid dtype")]
    InvalidDTypeError(DType),
    #[error("Cannot resolve data")]
    CannotResolveDataError,
    #[error(transparent)]
    IoError(#[from] std::io::Error),
    #[error(transparent)]
    CandleCoreError(#[from] candle_core::Error),
    #[error(transparent)]
    SafeTensorError(#[from] safetensors::SafeTensorError),
    #[error("Other error")]
    OtherError,
    #[error(transparent)]
    SerdeJsonError(#[from] serde_json::Error)
}

pub enum WeightStorageStrategy {
    None,
    BinFile(PathBuf),
    EmbeddedData
}

impl WeightStorageStrategy {
    fn get_manager<'a>(&'a self) -> Result<Box<dyn WeightExternalOutputManager<'a> + 'a>, Error> {
        match self {
            WeightStorageStrategy::None => Ok(Box::new(weights::NullOutputManager::new())),
            WeightStorageStrategy::BinFile(path) => Ok(Box::new(weights::BinOutputManager::<'a>::new(path))),
            WeightStorageStrategy::EmbeddedData => Ok(Box::new(weights::EmbeddedOutputManager::<'a>::new()))
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelInputType {
    TokenID(usize), // Tokenizer ID number
    PreviousInternal(usize)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputMetadata {
    pub model_input_type: ModelInputType
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelOutputType {
    TokenID(usize), // Tokenizer ID number
    NextInternal(usize)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputMetadata {
    pub model_output_type: ModelOutputType
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TokenizerInfo {
    HFTokenizer(String),
    RWKVWorld
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub tokenizer_infos: Vec<TokenizerInfo>,
    pub max_token_batch: Option<usize>
}

pub fn build_proto(
    inputs: &[(Arc<dyn Tensor>, Option<InputMetadata>)],
    outputs: &[(String, Arc<dyn Tensor>, Option<OutputMetadata>)],
    weight_storage: WeightStorageStrategy,
    model_metadata: Option<ModelMetadata>
) -> Result<onnx::ModelProto, Error> {
    
    // Get all nodes in graph
    let mut nodes = HashSet::new();
    for (_, tensor, _) in outputs {
        tensor.get_nodes(&mut nodes);
    }
    
    // Get requested node names
    let mut chosen_node_names: HashSet<String> = HashSet::new();
    let mut node_names: HashMap<&dyn Node, String> = HashMap::new();
    for node in &nodes {
        let base_name = node.get_name().unwrap_or(node.get_onnx_type());
        // Make up a new name based on the onnx type
        let mut i = 0;
        loop {
            let name = if i == 0 {
                base_name.to_string()
            } else {
                format!("{}_{}", base_name, i)
            };
            if !chosen_node_names.contains(&name) {
                node_names.insert(*node, name.clone());
                chosen_node_names.insert(name);
                break;
            }
            i += 1;
        }
    }
    println!("Found {} nodes in graph", nodes.len());

    // Get all tensors in graph
    let mut tensors = HashSet::new();
    for (_, tensor, _) in outputs {
        tensors.insert(tensor.as_ref());
        tensor.get_sub_tensors(&mut tensors);
    }
    println!("Found {} tensors in graph", tensors.len());

    // Assign names to all tensors in graph
    let mut chosen_tensor_names: HashSet<String> = HashSet::new();
    let mut tensor_names: HashMap<&dyn Tensor, String> = HashMap::new();

    // Assign requested names
    for tensor in &tensors { 
        if let Some(name) = tensor.get_name() {
            let name = name.to_string();
            if chosen_tensor_names.contains(&name) {
                return Err(Error::NameConflictError(name));
            }
            chosen_tensor_names.insert(name.clone());
            tensor_names.insert(*tensor, name);
        }
    }
    for (name, tensor, _) in outputs {
        chosen_tensor_names.insert(name.clone());
        tensor_names.insert(tensor.as_ref(), name.clone());
    }
    // Assign remaining names
    let mut next_tensor_id = 0;
    for tensor in &tensors {
        if !tensor_names.contains_key(tensor) {
            let name = loop {
                let name = format!("tensor_{}", next_tensor_id);
                if !chosen_tensor_names.contains(&name) {
                    break name;
                }
                next_tensor_id += 1;
            };
            
            tensor_names.insert(*tensor, name.clone());
            chosen_tensor_names.insert(name);
            next_tensor_id += 1;
        }
    }

    // Gather tensor weights
    let mut data_manager = weight_storage.get_manager()?;
    for tensor in &tensors {
        tensor.gather_weights(data_manager.as_mut());
    }
    data_manager.finalize_tensor_data();
    
    // Find tensors that are not input or output, and add initializer sections
    let mut tensors_to_enumerate = vec![];
    for tensor in &tensors {
        // Check if tensor is not in inputs and outputs
        if !inputs.iter().any(|(t, _)| (t.as_ref() as &dyn Tensor) == *tensor) && !outputs.iter().any(|(_, t, _)| t.as_ref() == *tensor) {
            tensors_to_enumerate.push(*tensor);
        }
    }
    
    // Generate initializer blocks
    let mut initializers = vec![];
    for tensor in tensors {
        if let Some(initializer) = tensor.get_initializer(tensor_names[&tensor].clone(), data_manager.as_mut())? {
            initializers.push(initializer);
        }
    }
    
    // Order nodes so that dependencies are before dependents
    let sorted_nodes = {
        let mut provided_tensors: HashSet<&dyn Tensor> = HashSet::new();
        let mut remaining_nodes: Vec<_> = nodes.drain().collect();
        let mut sorted_nodes = vec![];
        while !remaining_nodes.is_empty() {
            let mut new_remaining_nodes = vec![];
            for node in remaining_nodes {
                let mut all_dependencies_provided = true;
                for dependency in node.get_input_tensors() {
                    if !provided_tensors.contains(&dependency) && !dependency.is_input() {
                        all_dependencies_provided = false;
                        break;
                    }
                }
                if all_dependencies_provided {
                    sorted_nodes.push(node);
                    for output in node.get_output_tensors() {
                        provided_tensors.insert(output); 
                    }
                }
                else {
                    new_remaining_nodes.push(node);
                }
            }
            remaining_nodes = new_remaining_nodes;
        }
        sorted_nodes
    };
    
    let graph_inputs = inputs.iter().map(|(tensor, metadata)| {
        let json_str = serde_json::to_string(&metadata).unwrap();
        tensor.to_value_info_proto(tensor_names[&(tensor.as_ref())].clone(), vec![("whisper_tensor_metadata".to_string(), json_str)])
    }).collect();
    let graph_outputs = outputs.iter().map(|(name, tensor, metadata)| {
        let json_str = serde_json::to_string(&metadata).unwrap();
        tensor.to_value_info_proto(name.to_string(), vec![("whisper_tensor_metadata".to_string(), json_str)])
    }).collect();
    
    
    let graph = onnx::GraphProto {
        name: "model".to_string(),
        node: sorted_nodes.iter().map(|node| node.to_node_proto(node_names.get(node).map(|name| name.clone()), &tensor_names)).collect(),
        initializer: initializers,
        doc_string: String::new(),
        input: graph_inputs,
        output: graph_outputs,
        value_info: tensors_to_enumerate.iter().map(|tensor| tensor.to_value_info_proto(tensor_names.get(tensor).unwrap().to_string(), vec![])).collect(),
        metadata_props: vec![],
        .. Default::default()
    };
    
    let model_metadata = if let Some(metadata) = model_metadata {
        let json_str = serde_json::to_string(&metadata)?;
        vec![StringStringEntryProto{key: "whisper_tensor_metadata".to_string(), value: json_str}]
    } else {
        vec![]
    };

    let own_version = env!("CARGO_PKG_VERSION").to_string();
    Ok(onnx::ModelProto {
        ir_version: onnx::Version::IrVersion2024325 as i64,
        opset_import: vec![onnx::OperatorSetIdProto{version: 22, domain: "".to_string()}],
        producer_version: own_version,
        domain: String::new(),
        model_version: 0,
        doc_string: String::new(),
        graph: Some(graph),
        metadata_props: model_metadata,
        training_info: vec![],
        functions: vec![],
        .. Default::default()
    })
}