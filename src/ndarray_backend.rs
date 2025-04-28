use std::collections::HashMap;
use tracing_subscriber::Layer;
use crate::native_numeric_tensor::NativeNumericTensor;
use crate::RuntimeError;
use crate::symbolic_graph::{OperationId, SymbolicGraph, TensorId};
use crate::symbolic_graph::ops::Operation;

pub struct NDArrayRuntime {
    model: SymbolicGraph
}

impl NDArrayRuntime {
    pub fn new(model: SymbolicGraph) -> Self {
        Self {
            model
        }
    }

    pub fn run(&mut self, inputs: HashMap<String, NativeNumericTensor>) -> Result<HashMap<String, NativeNumericTensor>, RuntimeError>{
        let mut active_tensors: HashMap<TensorId, NativeNumericTensor> = self.model.get_initialized_tensors().clone();
        let tensors_by_name = self.model.get_tensors_by_name();
        for (name, tensor) in inputs {
            if let Some(tensor_id) = tensors_by_name.get(&name) {
                active_tensors.insert(*tensor_id, tensor);
            }
        }
        
        let ops = self.model.get_operations();
        let mut total_ops_completed: Vec<OperationId> = vec![];
        loop {
            let mut ops_completed_now = vec![];
            
            for (op_id, op) in ops {
                let input_ids = op.get_inputs();
                let mut input_values = Vec::new();
                // Collect all inputs, abort if we can't do this one yet
                for tensor_id in input_ids {
                    if let Some(x) = active_tensors.get(&tensor_id) {
                        input_values.push(x)
                    }
                    else {
                        // Can't do this one yet
                        continue
                    }
                }
                let outputs = op.exec_native(&input_values);
                let output_ids = op.get_outputs();
                for (id, value) in output_ids.iter().zip(outputs) {
                    active_tensors.insert(*id, value);
                }
                ops_completed_now.push(op_id)
            }
            if ops_completed_now.is_empty() {
                // Hopefully we are done now
                break;
            }
            
            total_ops_completed.extend(ops_completed_now);
        }

        
        let mut output_tensors = HashMap::new();
        let model_output_ids = self.model.get_outputs();
        for id in model_output_ids {
            if let Some(name)  = self.model.get_tensor_name(id) {
                let tensor = active_tensors.get(&id).unwrap();
                output_tensors.insert(name.to_string(), tensor.clone());
            }
        }
        
        Ok(output_tensors)
    }
}