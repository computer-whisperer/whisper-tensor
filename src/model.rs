use std::collections::HashMap;

use crate::backends::eval_backend;
use crate::backends::eval_backend::{EvalBackend, EvalRuntimeError};
use crate::dtype::DType;
use crate::numeric_tensor::{NumericTensor, NumericTensorError};
use crate::onnx::{ModelProto, StringStringEntryProto};
use prost::{DecodeError, Message};
use whisper_tensor_import::onnx_graph::{InputMetadata, ModelMetadata, OutputMetadata};

#[cfg(feature = "onnx-reference")]
use crate::backends::onnx_reference_backend::{self, ONNXReferenceTensor};

use crate::symbolic_graph::tensor_store::TensorStore;
use crate::symbolic_graph::{ONNXDecodingError, SymbolicGraph, SymbolicGraphMutator};

//#[cfg(feature = "ort")]
//use crate::backends::ort_backend::ORTNumericTensor;
use crate::DynRank;
use crate::interfaces::TextInferenceTokensInLogitOutInterface;
use crate::scalar_info::ScalarInfoTyped;
use crate::symbolic_graph::observer::SymbolicGraphObserver;

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
    #[error(transparent)]
    DecodeError(#[from] DecodeError),
    #[error("Unconfigured Backend")]
    UnconfiguredBackend,
}

#[allow(clippy::large_enum_variant)]
pub enum ModelExecutionRuntime<'a> {
    ONNXReference,
    ORT,
    Candle,
    Eval(EvalBackend<'a>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelID {
    pub name: String,
}

pub struct Model {
    id: ModelID,
    graph: SymbolicGraph,
    tensor_store: TensorStore,
    onnx_data: Vec<u8>,
    model_metadata: Option<ModelMetadata>,
    pub model_inputs: HashMap<String, Option<InputMetadata>>,
    pub model_outputs: HashMap<String, Option<OutputMetadata>>,
    pub text_inference_tokens_in_logits_out_interface:
        Option<TextInferenceTokensInLogitOutInterface>,
    // Runtimes
    #[cfg(feature = "onnx-reference")]
    onnx_reference_backend: Option<onnx_reference_backend::ONNXReferenceBackend>,
    #[cfg(feature = "candle")]
    candle_proto: Option<candle_onnx::onnx::ModelProto>,
}

impl Model {
    pub fn get_id(&self) -> &ModelID {
        &self.id
    }

    pub fn new_from_onnx(onnx_data: &[u8]) -> Result<Self, ModelError> {
        let model_info = ModelProto::decode(onnx_data)?;
        let mut model_metadata = None;
        for StringStringEntryProto { key, value } in model_info.metadata_props {
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
                    for StringStringEntryProto { key, value } in input.metadata_props {
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
                    for StringStringEntryProto { key, value } in output.metadata_props {
                        if key == "whisper_tensor_metadata" {
                            meta = Some(serde_json::from_str(&value)?);
                        }
                    }
                    model_outputs.insert(output.name.clone(), meta);
                }
            }
        }

        let (symbolic_graph, tensor_store) =
            SymbolicGraphMutator::from_onnx_bytes(onnx_data)?.get_inner();

        let text_inference_tokens_in_logits_out_interface = {
            if let Some(meta) = model_metadata.as_ref() {
                TextInferenceTokensInLogitOutInterface::try_from_onnx_metadata(
                    meta,
                    &model_inputs,
                    &model_outputs,
                    &symbolic_graph,
                )
                .ok()
            } else {
                None
            }
        };

        Ok(Self {
            id: ModelID {
                name: "TEST".to_string(),
            },
            graph: symbolic_graph,
            tensor_store,
            onnx_data: onnx_data.to_vec(),
            model_metadata,
            model_inputs,
            model_outputs,
            text_inference_tokens_in_logits_out_interface,
            #[cfg(feature = "onnx-reference")]
            onnx_reference_backend: None,
            #[cfg(feature = "candle")]
            candle_proto: None,
        })
    }

    #[cfg(feature = "candle")]
    pub fn setup_candle_backend(&mut self) -> Result<(), anyhow::Error> {
        // Emit to some temp file because bs
        let temp_file = tempfile::NamedTempFile::new().map_err(|e| anyhow::anyhow!(e))?;
        std::fs::write(temp_file.path(), &self.onnx_data).map_err(|e| anyhow::anyhow!(e))?;
        self.candle_proto =
            Some(candle_onnx::read_file(temp_file.path()).map_err(|e| anyhow::anyhow!(e))?);
        Ok(())
    }

    pub fn run(
        &self,
        inputs: HashMap<String, NumericTensor<DynRank>>,
        observer: &mut impl SymbolicGraphObserver,
        selected_runtime: &mut ModelExecutionRuntime,
    ) -> Result<HashMap<String, NumericTensor<DynRank>>, ModelError> {
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
            #[cfg(feature = "candle")]
            ModelExecutionRuntime::Candle => {
                if self.candle_proto.is_none() {
                    Err(ModelError::UnconfiguredBackend)?;
                }
                let candle_proto = self.candle_proto.as_ref().unwrap();
                let mut converted_inputs = HashMap::new();
                for (key, tensor) in inputs {
                    converted_inputs
                        .insert(key.to_string(), candle_core::Tensor::try_from(tensor)?);
                }
                let res = candle_onnx::simple_eval(candle_proto, converted_inputs)?;
                let mut output_tensors = HashMap::new();
                for (key, tensor) in res {
                    output_tensors.insert(key, NumericTensor::from(tensor));
                }
                output_tensors
            }
            ModelExecutionRuntime::Eval(eval_backend) => eval_backend::run(
                &self.graph,
                &self.tensor_store,
                eval_backend,
                observer,
                inputs,
            )?,
            _ => {
                panic!("Unsupported backend")
            }
        })
    }

    pub fn eval<T: SymbolicGraphObserver>(
        &self,
        inputs: HashMap<String, NumericTensor<DynRank>>,
        observer: &mut T,
        eval_backend: &mut EvalBackend,
    ) -> Result<HashMap<String, NumericTensor<DynRank>>, ModelError> {
        Ok(eval_backend::run(
            &self.graph,
            &self.tensor_store,
            eval_backend,
            observer,
            inputs,
        )?)
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

    #[allow(clippy::type_complexity)]
    pub fn get_input_tensor_info(
        &self,
    ) -> Result<HashMap<String, (DType, Vec<Option<u64>>)>, EvalRuntimeError> {
        let input_ids = self.graph.get_inputs();
        let mut results = HashMap::new();
        for tensor_id in input_ids {
            if let Some(tensor_info) = self.graph.get_tensor_info(tensor_id)
                && let (Some(dtype), Some(name), Some(shape)) =
                    (tensor_info.dtype(), tensor_info.name(), tensor_info.shape())
            {
                let shape: Vec<_> = shape
                    .iter()
                    .map(|x| match x {
                        ScalarInfoTyped::Numeric(a) => Some(*a),
                        _ => None,
                    })
                    .collect();
                results.insert(name.clone(), (dtype, shape));
            }
        }
        Ok(results)
    }
}
