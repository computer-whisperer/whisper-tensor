use std::collections::HashMap;
use crate::dtype::{DType, DTypeError};
use crate::numeric_tensor::NumericTensor;
use crate::symbolic_graph::{GraphOperation, OperationId, SymbolicGraph, TensorId, ONNXTensorInfo};
use crate::symbolic_graph::ops::{EvalError, Operation};
use crate::symbolic_graph::scalar_info::ScalarInfoTyped;
use crate::symbolic_graph::tensor_store::TensorStore;
use crate::tensor_rank::DynRank;

#[derive(Debug, Clone)]
pub enum EvalBackend {
    #[cfg(feature = "candle")]
    Candle(candle_core::Device),
    NDArray,
}

impl core::fmt::Display for EvalBackend {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EvalRuntimeError {
    #[error("Disabled Eval Backend: {0}")]
    DisabledBackend(EvalBackend),
    #[error(transparent)]
    DTypeError(#[from] DTypeError),
    #[error("Unexpected shape: expected {0:?}, got {1:?} in shape {2:?}")]
    UnexpectedDimension(u64, u64, Vec<u64>),
    #[error("Unexpected rank: expected {0}, got {1}")]
    UnexpectedRank(usize, usize),
    #[error("Unexpected dtype: expected {0}, got {1}")]
    UnexpectedDType(DType, DType),
    #[error("Missing input tensor: {0} {1:?} {2:?}")]
    MissingInputTensor(String, Option<DType>, Option<Vec<usize>>),
    #[error("Eval Error: {0:?} {1}")]
    EvalError(Option<String>, EvalError)
}

pub struct EvalRuntime {
    eval_backend: EvalBackend,
    pub(crate) tensor_store: TensorStore,
    model: SymbolicGraph
}

fn check_tensor_matches(tensor: &NumericTensor<DynRank>, tensor_info: &ONNXTensorInfo) -> Result<(), EvalRuntimeError> {
    if let Some(shape) = tensor_info.shape() {
        if shape.len() != tensor.shape().len() {
            Err(EvalRuntimeError::UnexpectedRank(shape.len(), tensor.shape().len()))?;
        }
        for (a, b) in shape.iter().zip(tensor.shape()) {
            match a {
                ScalarInfoTyped::Numeric(a) => if *a != b as u64 {
                    Err(EvalRuntimeError::UnexpectedDimension(*a, b as u64, tensor.shape()))?
                },
                _ => {}
            }
        }
    }
    if let Some(dtype) = tensor_info.dtype() {
        let tensor_dtype =  tensor.dtype();
        if dtype != tensor_dtype {
            Err(EvalRuntimeError::UnexpectedDType(dtype, tensor_dtype))?
        }
    }
    Ok(())
}

impl EvalRuntime {
    pub fn new(model: SymbolicGraph, tensor_store: TensorStore, eval_backend: EvalBackend) -> Result<Self, EvalRuntimeError> {
        match &eval_backend {
            // Always available
            EvalBackend::NDArray => {}
            #[cfg(feature = "candle")]
            EvalBackend::Candle(_) => {}
            _ => return Err(EvalRuntimeError::DisabledBackend(eval_backend))
        }
        Ok(Self {
            model,
            eval_backend,
            tensor_store
        })
    }
    
    pub fn get_num_ops(&self) -> usize {
        self.model.get_operations().len()
    }

    pub fn get_input_tensor_info(&self) -> Result<HashMap<String, (DType, Vec<Option<u64>>)>, EvalRuntimeError> {
        let input_ids = self.model.get_inputs();
        let mut results = HashMap::new();
        for tensor_id in input_ids {
            if let Some(tensor_info) = self.model.get_tensor_info(tensor_id) {
                if let (Some(dtype), Some(name), Some(shape)) = (tensor_info.dtype(), tensor_info.name(), tensor_info.shape()) {
                    let shape: Vec<_> = shape.iter().map(|x| match x {
                        ScalarInfoTyped::Numeric(a) => Some(*a),
                        _ => None
                    }).collect();
                    results.insert(name, (dtype, shape));
                }
            }
        }
        Ok(results)
    }

    pub fn get_eval_backend(&self) -> &EvalBackend {
        &self.eval_backend
    }

    pub fn run(&mut self, inputs: HashMap<String, NumericTensor<DynRank>>) -> Result<HashMap<String, NumericTensor<DynRank>>, EvalRuntimeError>{
        let initialized_tensors = self.model.get_initialized_tensors(&self.tensor_store);
        let mut active_tensors: HashMap<TensorId, NumericTensor<DynRank>> = HashMap::new();
        for (tensor_id, tensor) in initialized_tensors {
            active_tensors.insert(tensor_id, tensor.into());
        }
        let tensors_by_name = self.model.get_tensors_by_name();
        for (name, tensor) in inputs {
            if let Some(tensor_id) = tensors_by_name.get(&name) {
                active_tensors.insert(*tensor_id, tensor);
            }
        }

        let ops = self.model.get_operations();
        let mut remaining_ops_to_complete: Vec<OperationId> = ops.keys().map(|op_id| *op_id).collect();
        let mut total_ops_completed: Vec<OperationId> = vec![];
        loop {
            let mut ops_completed_now = vec![];

            for op_id in &remaining_ops_to_complete {
                let GraphOperation{ name, op } = ops.get(op_id).unwrap();
                let input_ids = op.get_inputs();
                let mut input_values = HashMap::new();
                // Collect all inputs, abort if we can't do this one yet
                let mut failed_to_fetch = false;
                for tensor_id in &input_ids {
                    if let Some(value) = active_tensors.get(&tensor_id) {
                        // Validate shape and dtype
                        let tensor_info = self.model.get_tensor_info(*tensor_id).unwrap();
                        check_tensor_matches(value, tensor_info)?;
                        input_values.insert(*tensor_id, value.clone());
                    }
                    else {
                        // Can't do this one yet
                        failed_to_fetch = true;
                        continue
                    }
                }
                if failed_to_fetch {
                    continue
                }
                let outputs = op.eval(&self.eval_backend, &input_values).map_err(|x| EvalRuntimeError::EvalError(name.clone(), x))?;
                for (tensor_id, value) in outputs {
                    //assert_eq!(value.has_nan().unwrap(), false);
                    
                    // Validate shape and dtype
                    let tensor_info = self.model.get_tensor_info(tensor_id).unwrap();
                    check_tensor_matches(&value, tensor_info)?;
                    
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


        let mut output_tensors = HashMap::new();
        let model_output_ids = self.model.get_outputs();
        for id in model_output_ids {
            if let Some(name)  = self.model.get_tensor_name(id) {
                if let Some(tensor) = active_tensors.get(&id) {
                    output_tensors.insert(name.to_string(), tensor.clone());
                }
            }
        }

        Ok(output_tensors)
    }
    
    pub fn get_symbolic_graph(&self) -> &SymbolicGraph {
        &self.model
    }
}