use std::collections::HashMap;

use crate::backends::eval_backend::{EvalBackend, EvalRuntimeError};
use crate::backends::{ModelLoadedTensorCache, eval_backend};
use crate::dtype::DType;
use crate::numeric_tensor::{NumericTensor, NumericTensorError};
use prost::DecodeError;
use rand::Rng;

use crate::symbolic_graph::tensor_store::TensorStore;
use crate::symbolic_graph::{ONNXDecodingError, SymbolicGraph, SymbolicGraphMutator, TensorType};

use crate::DynRank;
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
    #[allow(dead_code)]
    onnx_data: Vec<u8>,
}

impl Model {
    pub fn get_id(&self) -> &ModelID {
        &self.id
    }

    pub fn new_from_graph(
        name: impl Into<String>,
        graph: SymbolicGraph,
        tensor_store: TensorStore,
    ) -> Self {
        Self {
            id: ModelID {
                name: name.into(),
            },
            graph,
            tensor_store,
            onnx_data: Vec::new(),
        }
    }

    pub fn new_from_onnx(
        onnx_data: &[u8],
        rng: &mut impl Rng,
        base_dir: Option<&std::path::Path>,
    ) -> Result<Self, ModelError> {
        let (symbolic_graph, tensor_store) =
            SymbolicGraphMutator::from_onnx_bytes(onnx_data, rng, base_dir)?.get_inner();

        Ok(Self {
            id: ModelID {
                name: "TEST".to_string(),
            },
            graph: symbolic_graph,
            tensor_store,
            onnx_data: onnx_data.to_vec(),
        })
    }

    pub fn load_tensors(
        &self,
        cache: &mut ModelLoadedTensorCache,
        eval_backend: &mut EvalBackend,
    ) -> Result<(), ModelError> {
        for (key, tensor) in self.graph.get_tensors() {
            if !cache.tensors.contains_key(key) {
                let tensor = match &tensor.tensor_type {
                    TensorType::Constant(x) => Some(x.get_tensor(&self.tensor_store)),
                    TensorType::Input(Some(x)) => Some(x.get_tensor(&self.tensor_store)),
                    _ => None,
                };
                if let Some(tensor) = tensor {
                    cache
                        .tensors
                        .insert(*key, eval_backend.to_native_type(&tensor));
                }
            }
        }
        Ok(())
    }

    pub fn run(
        &self,
        inputs: HashMap<String, NumericTensor<DynRank>>,
        observer: &mut impl SymbolicGraphObserver,
        selected_runtime: &mut ModelExecutionRuntime,
    ) -> Result<HashMap<String, NumericTensor<DynRank>>, ModelError> {
        Ok(match selected_runtime {
            ModelExecutionRuntime::Eval(eval_backend) => eval_backend::run(
                &self.graph,
                &self.tensor_store,
                None,
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
        tensor_cache: Option<&mut ModelLoadedTensorCache>,
        eval_backend: &mut EvalBackend,
    ) -> Result<HashMap<String, NumericTensor<DynRank>>, ModelError> {
        Ok(eval_backend::run(
            &self.graph,
            &self.tensor_store,
            tensor_cache,
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
