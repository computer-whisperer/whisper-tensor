use std::collections::HashMap;
use crate::eval_backend::EvalBackend;
use crate::numeric_tensor::NumericTensor;
use crate::{onnx_reference_backend, DynRank};
use crate::symbolic_graph::{ONNXDecodingError, SymbolicGraph, SymbolicGraphMutator};
use crate::symbolic_graph::tensor_store::TensorStore;

#[derive(Debug, thiserror::Error)]
enum Error {
    #[error(transparent)]
    ONNXDecodingError(#[from] ONNXDecodingError)
}

pub enum SelectedModelExecutionRuntime {
    ONNXReference,
    ORT,
    Candle,
    Eval(EvalBackend),
}

struct Model {
    graph: SymbolicGraph,
    tensor_store: TensorStore,
    onnx_data: Vec<u8>,
    // Runtimes
    #[cfg(feature = "onnx-reference")]
    onnx_reference_backend: Option<onnx_reference_backend::ONNXReferenceBackend>,
    #[cfg(feature = "ort")]
    ort_session: Option<ort::session::Session>,
    #[cfg(feature = "candle")]
    candle_proto: Option<candle_onnx::onnx::ModelProto>
}

impl Model {
    fn new_from_onnx(onnx_data: &[u8]) -> Result<Self, Error> {
        let (symbolic_graph, tensor_store) = SymbolicGraphMutator::from_onnx_bytes(onnx_data)?.get_inner();
        Ok(Self {
            graph: symbolic_graph,
            tensor_store,
            onnx_data: onnx_data.to_vec(),
            #[cfg(feature = "onnx-reference")]
            onnx_reference_backend: None,
            #[cfg(feature = "ort")]
            ort_session: None,
            #[cfg(feature = "candle")]
            candle_proto: None
        })
    }

    pub fn run(&mut self, inputs: HashMap<String, NumericTensor<DynRank>>, selected_runtime: SelectedModelExecutionRuntime) -> Result<HashMap<String, NumericTensor<DynRank>>, Error> {
        match selected_runtime {
            w
        }
    }
}