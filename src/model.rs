use std::collections::HashMap;
use prost::{DecodeError, Message};
use whisper_tensor_import::onnx_graph::{InputMetadata, ModelMetadata, OutputMetadata};
use crate::numeric_tensor::{NumericTensor, NumericTensorError};
use crate::backends::{eval_backend};
use crate::backends::eval_backend::{EvalBackend, EvalRuntimeError};
use crate::dtype::DType;
use crate::onnx::{ModelProto, StringStringEntryProto};

#[cfg(feature = "onnx-reference")]
use crate::backends::onnx_reference_backend::{self, ONNXReferenceTensor};

use crate::symbolic_graph::{ ONNXDecodingError, SymbolicGraph, SymbolicGraphMutator};
use crate::symbolic_graph::tensor_store::TensorStore;

#[cfg(feature = "ort")]
use crate::backends::ort_backend::ORTNumericTensor;
use crate::DynRank;
use crate::scalar_info::ScalarInfoTyped;

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error(transparent)]
    NumericTensorError(#[from] NumericTensorError),
    #[error(transparent)]
    ONNXDecodingError(#[from] ONNXDecodingError),
    #[error(transparent)]
    EvalRuntimeError(#[from] EvalRuntimeError),
    #[error(transparent)]
    SerdeJSONError(#[from] serde_json::Error),
    #[cfg(feature = "onnx-reference")]
    #[error(transparent)]
    ONNXReference(#[from] onnx_reference_backend::ONNXReferenceError),
    #[cfg(feature = "candle")]
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
    #[cfg(feature = "ort")]
    #[error(transparent)]
    ORT(#[from] ort::Error),
    #[error(transparent)]
    DecodeError(#[from] DecodeError),
    #[error("Unconfigured Backend")]
    UnconfiguredBackend
}

pub enum ModelExecutionRuntime {
    ONNXReference,
    ORT,
    Candle,
    Eval(EvalBackend),
}

pub struct Model {
    graph: SymbolicGraph,
    tensor_store: TensorStore,
    onnx_data: Vec<u8>,
    model_metadata: Option<ModelMetadata>,
    pub model_inputs: HashMap<String, Option<InputMetadata>>,
    pub model_outputs: HashMap<String, Option<OutputMetadata>>,
    // Runtimes
    #[cfg(feature = "onnx-reference")]
    onnx_reference_backend: Option<onnx_reference_backend::ONNXReferenceBackend>,
    #[cfg(feature = "ort")]
    ort_session: Option<ort::session::Session>,
    #[cfg(feature = "candle")]
    candle_proto: Option<candle_onnx::onnx::ModelProto>
}


impl Model {
    pub fn new_from_onnx(onnx_data: &[u8]) -> Result<Self, ModelError> {
        let model_info = ModelProto::decode(onnx_data)?;
        let mut model_metadata = None;
        for StringStringEntryProto{key, value} in model_info.metadata_props {
            if key == "whisper_tensor_metadata" {
                let v: ModelMetadata = serde_json::from_str(&value)?;
                model_metadata = Some(v);
            }
        }
        let mut model_inputs = HashMap::new();
        let mut model_outputs = HashMap::new();
        if let Some(graph) = model_info.graph {
            for input in graph.input {
                if !input.name.is_empty() {
                    let mut meta = None;
                    for StringStringEntryProto{key, value} in input.metadata_props {
                        if key == "whisper_tensor_metadata" {
                            meta = Some(serde_json::from_str(&value)?);
                        }
                    }
                    model_inputs.insert(input.name.clone(), meta);
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
                    model_outputs.insert(output.name.clone(), meta);
                }
            }
        }
        let (symbolic_graph, tensor_store) = SymbolicGraphMutator::from_onnx_bytes(onnx_data)?.get_inner();
        Ok(Self {
            graph: symbolic_graph,
            tensor_store,
            onnx_data: onnx_data.to_vec(),
            model_metadata,
            model_inputs,
            model_outputs,
            #[cfg(feature = "onnx-reference")]
            onnx_reference_backend: None,
            #[cfg(feature = "ort")]
            ort_session: None,
            #[cfg(feature = "candle")]
            candle_proto: None
        })
    }

    #[cfg(feature = "ort")]
    pub fn setup_ort_backend(&mut self) -> Result<(), ort::Error> {
        let mut builder = ort::session::Session::builder()?;
        builder = builder.with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?;
        builder = builder.with_intra_threads(4)?;
        //builder = builder.with_execution_providers([ort::execution_providers::CUDAExecutionProvider::default().build()])?;
        self.ort_session = Some(builder.commit_from_memory(&self.onnx_data)?);
        Ok(())
    }

    #[cfg(feature = "candle")]
    pub fn setup_candle_backend(&mut self) -> Result<(), anyhow::Error> {
        // Emit to some temp file because bs
        let temp_file = tempfile::NamedTempFile::new().map_err(|e| anyhow::anyhow!(e))?;
        std::fs::write(temp_file.path(), &self.onnx_data).map_err(|e| anyhow::anyhow!(e))?;
        self.candle_proto = Some(candle_onnx::read_file(temp_file.path()).map_err(|e| anyhow::anyhow!(e))?);
        Ok(())
    }

    pub fn run(&self, inputs: HashMap<String, NumericTensor<DynRank>>, selected_runtime: &mut ModelExecutionRuntime) -> Result<HashMap<String, NumericTensor<DynRank>>, ModelError> {
        Ok(match selected_runtime {
            #[cfg(feature = "onnx-reference")]
            ModelExecutionRuntime::ONNXReference => {
                if self.onnx_reference_backend.is_none() {
                    Err(ModelError::UnconfiguredBackend)?;
                }
                let session = self.onnx_reference_backend.as_ref().unwrap();
                let mut converted_inputs = HashMap::new();
                for (name, tensor) in inputs {
                    converted_inputs.insert(name, ONNXReferenceTensor::try_from(tensor)?);
                }
                let res = session.run(converted_inputs)?;
                let mut output_tensors = HashMap::new();
                for (name, tensor) in res {
                    output_tensors.insert(name, NumericTensor::from(tensor));
                }
                output_tensors
            }
            /*
            #[cfg(feature = "ort")]
            ModelExecutionRuntime::ORT => {
                if self.ort_session.is_none() {
                    Err(ModelError::UnconfiguredBackend)?;
                }
                let session = self.ort_session.as_mut().unwrap();
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
                output_tensors
            }*/
            #[cfg(feature = "candle")]
            ModelExecutionRuntime::Candle => {
                if self.candle_proto.is_none() {
                    Err(ModelError::UnconfiguredBackend)?;
                }
                let candle_proto = self.candle_proto.as_ref().unwrap();
                let mut converted_inputs = HashMap::new();
                for (key, tensor) in inputs {
                    converted_inputs.insert(key.to_string(), candle_core::Tensor::try_from(tensor)?);
                }
                let res = candle_onnx::simple_eval(candle_proto, converted_inputs)?;
                let mut output_tensors = HashMap::new();
                for (key, tensor) in res {
                    output_tensors.insert(key, NumericTensor::from(tensor));
                }
                output_tensors
            }
            ModelExecutionRuntime::Eval(eval_backend) => {
                eval_backend::run(&self.graph, &self.tensor_store, eval_backend, inputs)?
            }
            _ => {panic!("Unsupported backend")}
        })
    }

    pub fn eval(&self, inputs: HashMap<String, NumericTensor<DynRank>>, eval_backend: &mut EvalBackend) -> Result<HashMap<String, NumericTensor<DynRank>>, ModelError> {
        Ok(eval_backend::run(&self.graph, &self.tensor_store, eval_backend, inputs)?)
    }

    pub fn get_symbolic_graph(&self) -> &SymbolicGraph {
        &self.graph
    }
    
    pub fn get_tensor_store(&self) -> &TensorStore {
        &self.tensor_store
    }
    
    pub fn get_model_metadata(&self) -> Option<&ModelMetadata> {
        self.model_metadata.as_ref()
    }

    pub fn get_input_tensor_info(&self) -> Result<HashMap<String, (DType, Vec<Option<u64>>)>, EvalRuntimeError> {
        let input_ids = self.graph.get_inputs();
        let mut results = HashMap::new();
        for tensor_id in input_ids {
            if let Some(tensor_info) = self.graph.get_tensor_info(tensor_id) {
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

}