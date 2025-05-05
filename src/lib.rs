use std::collections::HashMap;
use num_traits::Float;
use num_traits::real::Real;
use ort::session::Input;
use prost::{DecodeError, Message};
use serde::{Deserialize, Serialize};
use onnx_graph::{InputMetadata, ModelMetadata, OutputMetadata};
use crate::dtype::{DType, DTypeError};
use crate::eval_backend::{EvalBackend, EvalRuntime, EvalRuntimeError};
use crate::ndarray_backend::NDArrayNumericTensorError;
use crate::numeric_tensor::{NumericTensor, NumericTensorError};
use crate::onnx::{ModelProto, StringStringEntryProto};
use crate::ort_backend::ORTNumericTensor;
use crate::sampler::SamplerError;
use crate::symbolic_graph::{ONNXDecodingError, SymbolicGraphMutator};
use crate::symbolic_graph::ops::EvalError;

mod vulkan_context;
pub mod symbolic_graph;
pub mod numeric_tensor;
pub mod dtype;
pub mod sampler;
pub mod language_model;
pub mod tokenizer;
pub mod eval_backend;
mod ndarray_backend;
mod onnx_testing;
pub mod numeric_scalar;

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
pub enum RuntimeBackend {
    ONNXReference,
    ORT,
    Candle,
    Eval(EvalBackend),
}

impl core::fmt::Display for RuntimeBackend {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub struct RuntimeModel {
    inner: RuntimeModelInner,
    model_metadata: Option<ModelMetadata>,
    model_inputs: HashMap<String, Option<InputMetadata>>,
    model_outputs: HashMap<String, Option<OutputMetadata>>,
}

enum RuntimeModelInner {
    #[cfg(feature = "onnx-reference")]
    ONNXReference(onnx_reference_backend::ONNXReferenceBackend),
    #[cfg(feature = "ort")]
    ORT(ort::session::Session),
    #[cfg(feature = "candle")]
    Candle(candle_onnx::onnx::ModelProto),
    Eval(EvalRuntime)
}

impl From<RuntimeModelInner> for RuntimeBackend {
    fn from(value: RuntimeModelInner) -> Self {
        match value {
            #[cfg(feature = "onnx-reference")]
            RuntimeModelInner::ONNXReference(_) => RuntimeBackend::ONNXReference,
            #[cfg(feature = "ort")]
            RuntimeModelInner::ORT(_) => RuntimeBackend::ORT,
            #[cfg(feature = "candle")]
            RuntimeModelInner::Candle(_) => RuntimeBackend::Candle,
            RuntimeModelInner::Eval(x) => RuntimeBackend::Eval(x.get_eval_backend().clone()),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("Disabled runtime {0:?}")]
    DisabledBackend(RuntimeBackend),
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
    NDArrayNumericTensorError(#[from] NDArrayNumericTensorError),
    #[error(transparent)]
    SamplerError(#[from] SamplerError),
    #[error("Missing input {0}")]
    MissingInput(String),
    #[error(transparent)]
    ONNXDecodingError(#[from] ONNXDecodingError),
    #[error(transparent)]
    ExecNativeError(#[from] EvalError),
    #[error(transparent)]
    EvalRuntimeError(#[from] EvalRuntimeError),
    #[error(transparent)]
    DTypeError(#[from] DTypeError),
    #[error("Other error: {0}")]
    OtherAnyhow(#[from] anyhow::Error),
    #[error(transparent)]
    DecodeError(#[from] DecodeError),
    #[error(transparent)]
    SerdeJSONError(#[from] serde_json::Error)
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub enum TrigOp {
    Asin,
    Asinh,
    Acos,
    Acosh,
    Atan,
    Atanh,
    Sin,
    Sinh,
    Cos,
    Cosh,
    Tan,
    Tanh
}

impl TrigOp {
    fn apply<F: Float>(&self, x: F) -> F {
        match self {
            TrigOp::Asin => x.asin(),
            TrigOp::Asinh => x.asinh(),
            TrigOp::Acos => x.acos(),
            TrigOp::Acosh => x.acosh(),
            TrigOp::Atan => x.atan(),
            TrigOp::Atanh => x.atanh(),
            TrigOp::Sin => x.sin(),
            TrigOp::Sinh => x.sinh(),
            TrigOp::Cos => x.cos(),
            TrigOp::Cosh => x.cosh(),
            TrigOp::Tan => x.tan(),
            TrigOp::Tanh => x.tanh()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RuntimeEnvironment {
    pub cuda: bool
}

impl RuntimeModel {
    pub fn load_onnx(onnx_data: &[u8], runtime: RuntimeBackend, runtime_environment: RuntimeEnvironment) -> Result<Self, RuntimeError> {
        let inner = match runtime {
            #[cfg(feature = "onnx-reference")]
            RuntimeBackend::ONNXReference => {
                let backend = onnx_reference_backend::ONNXReferenceBackend::new(onnx_data)?;
                RuntimeModelInner::ONNXReference(backend)
            }
            #[cfg(feature = "ort")]
            RuntimeBackend::ORT => {
                let mut builder = ort::session::Session::builder()?;
                builder = builder.with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?;
                builder = builder.with_intra_threads(4)?;
                if runtime_environment.cuda {
                    builder = builder.with_execution_providers([ort::execution_providers::CUDAExecutionProvider::default().build()])?;
                }
                RuntimeModelInner::ORT(builder.commit_from_memory(&onnx_data)?)
            }
            #[cfg(feature = "candle")]
            RuntimeBackend::Candle => {
                // Emit to some temp file because bs
                let temp_file = tempfile::NamedTempFile::new().map_err(|e| anyhow::anyhow!(e))?;
                std::fs::write(temp_file.path(), onnx_data).map_err(|e| anyhow::anyhow!(e))?;
                RuntimeModelInner::Candle(candle_onnx::read_file(temp_file.path()).map_err(|e| anyhow::anyhow!(e))?)
            }
            RuntimeBackend::Eval(eval_backend) => {
                let model = SymbolicGraphMutator::from_onnx_bytes(onnx_data)?.get_inner();
                RuntimeModelInner::Eval(EvalRuntime::new(model, eval_backend)?)
            }
            _ => {
                Err(RuntimeError::DisabledBackend(runtime))?
            }
        };
        let (model_meta, inputs, outputs) = match &inner {
            _ => {
                // Decode anew and extract
                let model_info = ModelProto::decode(onnx_data)?;
                let mut model_meta = None;
                for StringStringEntryProto{key, value} in model_info.metadata_props {
                    if key == "whisper_tensor_metadata" {
                        let model_metadata: ModelMetadata = serde_json::from_str(&value)?;
                        model_meta = Some(model_metadata);
                    }
                }
                let mut inputs = HashMap::new();
                let mut outputs = HashMap::new();
                if let Some(graph) = model_info.graph {
                    for input in graph.input {
                        if !input.name.is_empty() {
                            let mut meta = None;
                            for StringStringEntryProto{key, value} in input.metadata_props {
                                if key == "whisper_tensor_metadata" {
                                    meta = Some(serde_json::from_str(&value)?);
                                }
                            }
                            inputs.insert(input.name.clone(), meta);
                        }
                    }
                    for output in graph.output {
                        if !output.name.is_empty() {
                            let mut meta = None;
                            for StringStringEntryProto{key, value} in output.metadata_props {
                                if key == "whisper_tensor_metadata" {
                                    meta = Some(serde_json::from_str(&value)?);
                                }
                            }
                            outputs.insert(output.name.clone(), meta);
                        }
                    }
                }
                (model_meta, inputs, outputs)
            }
        };
        
        Ok(Self{
            inner,
            model_metadata: model_meta,
            model_inputs: inputs,
            model_outputs: outputs
        })
    }

    pub fn run(&mut self, inputs: HashMap<String, NumericTensor>) -> Result<HashMap<String, NumericTensor>, RuntimeError> {
        match &mut self.inner {
            #[cfg(feature = "onnx-reference")]
            RuntimeModelInner::ONNXReference(session) => {
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
            RuntimeModelInner::ORT(session) => {
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
            RuntimeModelInner::Candle(model) => {
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
            RuntimeModelInner::Eval(runtime) => {
                Ok(runtime.run(inputs)?)
            }
            _ => {
                todo!()
            }
        }
    }
    
    pub fn get_input_tensor_info(&self) -> Result<HashMap<String, (DType, Vec<Option<u64>>)>, RuntimeError> {
        match &self.inner {
            #[cfg(feature = "onnx-reference")]
            RuntimeModelInner::ONNXReference(session) => {
                Ok(session.get_input_tensor_info()?)
            }
            RuntimeModelInner::Eval(runtime) => {
                Ok(runtime.get_input_tensor_info()?)
            }
            _ => {
                todo!()
            }
        }
    }
}

