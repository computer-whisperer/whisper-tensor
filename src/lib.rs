use std::collections::HashMap;
use crate::dtype::DType;
use crate::native_numeric_tensor::NativeNumericTensor;
use crate::numeric_tensor::{NumericTensor, NumericTensorError};
use crate::sampler::SamplerError;

mod vulkan_context;
mod symbolic_graph;
pub mod numeric_tensor;
pub mod native_numeric_tensor;
pub mod dtype;
pub mod sampler;
pub mod language_model;
pub mod tokenizer;

pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

#[cfg(feature = "onnx-reference")]
mod onnx_reference_backend;

#[derive(Debug, Clone)]
pub enum Backend {
    ONNXReference,
    ORT,
    Candle,
    Kyanite,
    WONNX
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
    #[cfg(feature = "kyanite")]
    Kyanite(),
    #[cfg(feature = "wonnx")]
    WONNX()
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
            #[cfg(feature = "kyanite")]
            RuntimeModel::Kyanite() => Backend::Kyanite,
            #[cfg(feature = "wonnx")]
            RuntimeModel::WONNX() => Backend::WONNX
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
    #[cfg(feature = "kyanite")]
    #[error(transparent)]
    KyaniteONNX(#[from] kn_graph::onnx::result::OnnxError),
    #[cfg(feature = "wonnx")]
    #[error(transparent)]
    WONNX(#[from] wonnx::SessionError),
    #[error("Invalid tensor backend for runtime")]
    InvalidTensorBackend,
    #[error("Invalid tensor dtype for backend")]
    BackendUnsupportedTensorDType,
    #[error(transparent)]
    NumericTensorError(#[from] NumericTensorError),
    #[error(transparent)]
    SamplerError(#[from] SamplerError),
    #[error(transparent)]
    Other(#[from] anyhow::Error)
}

impl RuntimeModel {
    pub fn load_onnx(onnx_data: &[u8], runtime: Backend) -> Result<Self, RuntimeError> {
        match runtime {
            #[cfg(feature = "onnx-reference")]
            Backend::ONNXReference => {
                let backend = onnx_reference_backend::ONNXReferenceBackend::new(onnx_data)?;
                Ok(RuntimeModel::ONNXReference(backend))
            }
            #[cfg(feature = "ort")]
            Backend::ORT => {
                Ok(RuntimeModel::ORT(ort::session::Session::builder()?
                    .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
                    .with_intra_threads(4)?
                    .commit_from_memory(&onnx_data)?))
            }
            #[cfg(feature = "candle")]
            Backend::Candle => {
                // Emit to some temp file because bs
                let temp_file = tempfile::NamedTempFile::new().map_err(|e| anyhow::anyhow!(e))?;
                std::fs::write(temp_file.path(), onnx_data).map_err(|e| anyhow::anyhow!(e))?;
                Ok(RuntimeModel::Candle(candle_onnx::read_file(temp_file.path()).map_err(|e| anyhow::anyhow!(e))?))
            }
            #[cfg(feature = "kyanite")]
            Backend::Kyanite => {
                let graph = kn_graph::onnx::load_graph_from_onnx_bytes(onnx_data)?;
                Ok(RuntimeModel::Kyanite())
            }
            #[cfg(feature = "wonnx")]
            Backend::WONNX => {
                let session = futures::executor::block_on(wonnx::Session::from_bytes(onnx_data))?;
                Ok(RuntimeModel::WONNX())
            }
            _ => {
                Err(RuntimeError::DisabledBackend(runtime))
            }
        }
    }

    pub fn run(&self, inputs: HashMap<String, NumericTensor>) -> Result<HashMap<String, NumericTensor>, RuntimeError> {
        match self {
            #[cfg(feature = "onnx-reference")]
            RuntimeModel::ONNXReference(session) => {
                let mut converted_inputs = HashMap::new();
                for (name, tensor) in inputs {
                    match tensor {
                        NumericTensor::Native(x) => {
                            converted_inputs.insert(name, session.load_tensor(&x)?);
                        }
                        NumericTensor::ONNXReference(x) => {
                            converted_inputs.insert(name, x);
                        }
                        _ => {
                            Err(RuntimeError::InvalidTensorBackend)?;
                        }
                    }
                }
                let res = session.run(converted_inputs)?;
                let mut output_tensors = HashMap::new();
                for (name, tensor) in res {
                    output_tensors.insert(name, NumericTensor::ONNXReference(tensor));
                }
                Ok(output_tensors)
            }
            #[cfg(feature = "ort")]
            RuntimeModel::ORT(session) => {
                // Run the session
                todo!()
            }
            #[cfg(feature = "candle")]
            RuntimeModel::Candle(model) => {
                let mut converted_inputs = HashMap::new();
                for (name, tensor) in inputs {
                    match tensor {
                        NumericTensor::Native(x) => {
                            converted_inputs.insert(name, x.try_into()?);
                        }
                        NumericTensor::Candle(x) => {
                            converted_inputs.insert(name, x);
                        }
                        _ => {
                            Err(RuntimeError::InvalidTensorBackend)?;
                        }
                    }
                }
                let res = candle_onnx::simple_eval(model, converted_inputs)?;
                let mut output_tensors = HashMap::new();
                for (name, tensor) in res {
                    output_tensors.insert(name, NumericTensor::Candle(tensor));
                }
                Ok(output_tensors)
            }
            _ => {
                todo!()
            }
        }
    }
    
    pub fn load_tensor(&self, tensor: &NativeNumericTensor) -> Result<NumericTensor, RuntimeError> {
        match self {
            #[cfg(feature = "onnx-reference")]
            RuntimeModel::ONNXReference(session) => {
                Ok(NumericTensor::ONNXReference(session.load_tensor(tensor)?))
            }
            #[cfg(feature = "ort")]
            RuntimeModel::ORT(session) => {
                // Load the tensor
                todo!()
            }
            #[cfg(feature = "candle")]
            RuntimeModel::Candle(model) => {
                // Load the tensor
                todo!()
            }
            #[cfg(feature = "kyanite")]
            RuntimeModel::Kyanite() => {
                // Load the tensor
                todo!()
            }
            #[cfg(feature = "wonnx")]
            RuntimeModel::WONNX() => {
                // Load the tensor
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
            #[cfg(feature = "ort")]
            RuntimeModel::ORT(session) => {
                // Load the tensor
                todo!()
            }
            #[cfg(feature = "candle")]
            RuntimeModel::Candle(model) => {
                // Load the tensor
                todo!()
            }
            #[cfg(feature = "kyanite")]
            RuntimeModel::Kyanite() => {
                // Load the tensor
                todo!()
            }
            #[cfg(feature = "wonnx")]
            RuntimeModel::WONNX() => {
                // Load the tensor
                todo!()
            }
        }
    }
}

