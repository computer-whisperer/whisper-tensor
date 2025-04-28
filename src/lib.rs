use std::collections::HashMap;
use crate::dtype::DType;
use crate::ndarray_backend::NDArrayRuntime;
use crate::numeric_tensor::{NumericTensor, NumericTensorError};
use crate::ort_backend::ORTNumericTensor;
use crate::sampler::SamplerError;
use crate::symbolic_graph::{ONNXDecodingError, SymbolicGraphMutator};

mod vulkan_context;
pub mod symbolic_graph;
pub mod numeric_tensor;
pub mod native_numeric_tensor;
pub mod dtype;
pub mod sampler;
pub mod language_model;
pub mod tokenizer;
pub mod ndarray_backend;

#[cfg(feature = "ort")]
pub mod ort_backend;
#[cfg(feature = "onnx-reference")]
mod onnx_reference_backend;
#[cfg(feature = "candle")]
mod candle_backend;

pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

#[derive(Debug, Clone)]
pub enum Backend {
    ONNXReference,
    ORT,
    Candle,
    NDArray,
}

impl core::fmt::Display for Backend {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self)
    }
}


pub enum RuntimeModel {
    #[cfg(feature = "onnx-reference")]
    ONNXReference(onnx_reference_backend::ONNXReferenceBackend),
    #[cfg(feature = "ort")]
    ORT(ort::session::Session),
    #[cfg(feature = "candle")]
    Candle(candle_onnx::onnx::ModelProto),
    NDArray(NDArrayRuntime)
}

impl From<RuntimeModel> for Backend {
    fn from(value: RuntimeModel) -> Self {
        match value {
            #[cfg(feature = "onnx-reference")]
            RuntimeModel::ONNXReference(_) => Backend::ONNXReference,
            #[cfg(feature = "ort")]
            RuntimeModel::ORT(_) => Backend::ORT,
            #[cfg(feature = "candle")]
            RuntimeModel::Candle(_) => Backend::Candle,
            RuntimeModel::NDArray(_) => Backend::NDArray,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("Disabled runtime {0:?}")]
    DisabledBackend(Backend),
    #[cfg(feature = "onnx-reference")]
    #[error(transparent)]
    ONNXReference(#[from] onnx_reference_backend::ONNXReferenceError),
    #[cfg(feature = "ort")]
    #[error(transparent)]
    ORT(#[from] ort::Error),
    #[cfg(feature = "candle")]
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
    #[error("Invalid tensor backend for runtime")]
    InvalidTensorBackend,
    #[error("Invalid tensor dtype for backend")]
    BackendUnsupportedTensorDType,
    #[error(transparent)]
    NumericTensorError(#[from] NumericTensorError),
    #[error(transparent)]
    SamplerError(#[from] SamplerError),
    #[error("Missing input {0}")]
    MissingInput(String),
    #[error(transparent)]
    ONNXDecodingError(#[from] ONNXDecodingError),
    #[error(transparent)]
    Other(#[from] anyhow::Error)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RuntimeEnvironment {
    pub cuda: bool
}

impl RuntimeModel {
    pub fn load_onnx(onnx_data: &[u8], runtime: Backend, runtime_environment: RuntimeEnvironment) -> Result<Self, RuntimeError> {
        match runtime {
            #[cfg(feature = "onnx-reference")]
            Backend::ONNXReference => {
                let backend = onnx_reference_backend::ONNXReferenceBackend::new(onnx_data)?;
                Ok(RuntimeModel::ONNXReference(backend))
            }
            #[cfg(feature = "ort")]
            Backend::ORT => {
                let mut builder = ort::session::Session::builder()?;
                builder = builder.with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?;
                builder = builder.with_intra_threads(4)?;
                if runtime_environment.cuda {
                    builder = builder.with_execution_providers([ort::execution_providers::CUDAExecutionProvider::default().build()])?;
                }
                Ok(RuntimeModel::ORT(builder.commit_from_memory(&onnx_data)?))
            }
            #[cfg(feature = "candle")]
            Backend::Candle => {
                // Emit to some temp file because bs
                let temp_file = tempfile::NamedTempFile::new().map_err(|e| anyhow::anyhow!(e))?;
                std::fs::write(temp_file.path(), onnx_data).map_err(|e| anyhow::anyhow!(e))?;
                Ok(RuntimeModel::Candle(candle_onnx::read_file(temp_file.path()).map_err(|e| anyhow::anyhow!(e))?))
            }
            Backend::NDArray => {
                let model = SymbolicGraphMutator::from_onnx_bytes(onnx_data)?.get_inner();
                Ok(RuntimeModel::NDArray(NDArrayRuntime::new(model)))
            }
            _ => {
                Err(RuntimeError::DisabledBackend(runtime))
            }
        }
    }

    pub fn run(&mut self, inputs: HashMap<String, NumericTensor>) -> Result<HashMap<String, NumericTensor>, RuntimeError> {
        match self {
            #[cfg(feature = "onnx-reference")]
            RuntimeModel::ONNXReference(session) => {
                let mut converted_inputs = HashMap::new();
                for (name, tensor) in inputs {
                    converted_inputs.insert(name, tensor.try_into()?);
                }
                let res = session.run(converted_inputs)?;
                let mut output_tensors = HashMap::new();
                for (name, tensor) in res {
                    output_tensors.insert(name, NumericTensor::from(tensor));
                }
                Ok(output_tensors)
            }
            #[cfg(feature = "ort")]
            RuntimeModel::ORT(session) => {
                // Run the session
                let mut converted_inputs: HashMap<String, ort::value::DynValue> = HashMap::new();
                for (key, tensor) in inputs.into_iter() {
                    converted_inputs.insert(key.to_string(), ORTNumericTensor::try_from(tensor)?.0);
                }
                let res = session.run(converted_inputs)?;
                let mut output_tensors = HashMap::new();
                for (key, value) in res.into_iter() {
                    output_tensors.insert(key.to_string(), NumericTensor::from(ORTNumericTensor(value)));
                }
                Ok(output_tensors)
            }
            #[cfg(feature = "candle")]
            RuntimeModel::Candle(model) => {
                let mut converted_inputs = HashMap::new();
                for (key, tensor) in inputs {
                    converted_inputs.insert(key.to_string(), candle_core::Tensor::try_from(tensor)?);
                }
                let res = candle_onnx::simple_eval(model, converted_inputs)?;
                let mut output_tensors = HashMap::new();
                for (key, tensor) in res {
                    output_tensors.insert(key, NumericTensor::from(tensor));
                }
                Ok(output_tensors)
            }
            RuntimeModel::NDArray(runtime) => {
                let mut converted_inputs = HashMap::new();
                for (key, tensor) in inputs {
                    converted_inputs.insert(key.to_string(), tensor.try_into()?);
                }
                let res = runtime.run(converted_inputs)?;
                let mut output_tensors = HashMap::new();
                for (key, tensor) in res {
                    output_tensors.insert(key, NumericTensor::from(tensor));
                }
                Ok(output_tensors)
            }
            _ => {
                todo!()
            }
        }
    }
    
    pub fn get_input_tensor_info(&self) -> Result<HashMap<String, (DType, Vec<Option<usize>>)>, RuntimeError> {
        match self {
            #[cfg(feature = "onnx-reference")]
            RuntimeModel::ONNXReference(session) => {
                Ok(session.get_input_tensor_info()?)
            }
            _ => {
                todo!()
            }
        }
    }
}

