pub mod operators;
pub mod weights;
pub mod tensor;
mod node;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tensor::*;
use node::*;

pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}



#[derive(Debug)]
pub enum Error {
    ShapeMismatchError,
    DTypeMismatchError,
    InvalidInputError,
    UnsupportedDtypeError
}

fn validate_elementwise_inputs(inputs: &[Arc<dyn Tensor>]) -> Result<(), Error> {
    for input in inputs {
        if input.dtype() != inputs[0].dtype() {
            return Err(Error::DTypeMismatchError);
        }
        if input.shape() != inputs[0].shape() {
            return Err(Error::ShapeMismatchError);
        }
    };
    Ok(())
}



pub fn build_proto(
    inputs: &[(&str, Arc<InputTensor>)],
    outputs: &[(&str, Arc<dyn Tensor>)]
) -> onnx::ModelProto {
    
    // Get all nodes in graph
    let mut nodes = HashSet::new();
    for (_, tensor) in outputs {
        nodes.extend(tensor.get_nodes());
    }
    
    // Get all tensors in graph
    let mut tensors = HashSet::new();
    for (_, tensor) in outputs {
        for tensor in tensor.get_sub_tensors() {
            tensors.insert(tensor);
            tensors.extend(tensor.get_sub_tensors());
        }
    }
    
    // Assign names to all tensors in graph
    let mut tensor_names = HashMap::new();
    for (name, tensor) in inputs {
        tensor_names.insert(tensor.as_ref() as &dyn Tensor, name.to_string());
    }
    for (name, tensor) in outputs {
        tensor_names.insert(tensor.as_ref(), name.to_string());
    }
    let mut next_tensor_id = 0;
    let mut tensors_to_enumerate = vec![];
    let mut initializers = vec![];
    for tensor in tensors {
        if !tensor_names.contains_key(&tensor) {
            tensor_names.insert(tensor, format!("tensor_{}", next_tensor_id));
            next_tensor_id += 1;
        }
        // Check if tensor is not in inputs and outputs
        if !inputs.iter().any(|(_, t)| (t.as_ref() as &dyn Tensor) == tensor) && !outputs.iter().any(|(_, t)| t.as_ref() == tensor) {
            tensors_to_enumerate.push(tensor);
        }
        if let Some(initializer) = tensor.get_initializer(tensor_names[&tensor].clone()) {
            initializers.push(initializer);
        }
    }
    
    let graph = onnx::GraphProto {
        name: String::new(),
        node: nodes.iter().map(|node| node.to_node_proto(None, &tensor_names)).collect(),
        initializer: initializers,
        doc_string: String::new(),
        input: inputs.iter().map(|(name, tensor)| tensor.to_value_info_proto(name.to_string())).collect(),
        output: outputs.iter().map(|(name, tensor)| tensor.to_value_info_proto(name.to_string())).collect(),
        value_info: tensors_to_enumerate.iter().map(|tensor| tensor.to_value_info_proto(tensor_names.get(tensor).unwrap().to_string())).collect(),
        metadata_props: vec![],
        .. Default::default()
    };

    let own_version = env!("CARGO_PKG_VERSION").to_string();
    onnx::ModelProto {
        ir_version: onnx::Version::IrVersion2024325 as i64,
        opset_import: vec![],
        producer_version: own_version,
        domain: String::from("com.rwkv"),
        model_version: 0,
        doc_string: String::new(),
        graph: Some(graph),
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![],
        .. Default::default()
    }
}