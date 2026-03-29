use std::collections::HashMap;

use crate::backends::eval_backend::EvalRuntimeError;
use crate::dtype::DType;
use crate::migration::numeric_tensor::NumericTensorError;
use crate::numeric_tensor::NumericTensorView;
use crate::pool::Pool;
use crate::symbolic_graph::ops::EvalError;
use crate::tensor_rank::DynRank;
use prost::DecodeError;
use rand::Rng;

use crate::symbolic_graph::tensor_store::TensorStore;
use crate::symbolic_graph::{ONNXDecodingError, SymbolicGraph, SymbolicGraphMutator};

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
    DecodeError(#[from] DecodeError),
    #[error("Unconfigured Backend")]
    UnconfiguredBackend,
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
            id: ModelID { name: name.into() },
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

    pub fn get_symbolic_graph(&self) -> &SymbolicGraph {
        &self.graph
    }

    pub fn get_tensor_store(&self) -> &TensorStore {
        &self.tensor_store
    }

    /// Run the model through the pool-based eval pipeline.
    ///
    /// Accepts named inputs as `NumericTensorView`s, maps them to GlobalIds,
    /// runs `pool_eval_with_store`, and maps output GlobalIds back to names.
    pub fn eval_pool<'p, P: Pool + 'p>(
        &self,
        inputs: HashMap<String, &NumericTensorView<'_, DynRank>>,
        pool: &'p P,
    ) -> Result<HashMap<String, crate::numeric_tensor::NumericTensor<'p, DynRank, P>>, EvalError>
    {
        let tensors_by_name = self.graph.get_tensors_by_name();

        // Map String names → GlobalIds for inputs.
        let mut id_inputs = HashMap::new();
        for (name, view) in inputs {
            if let Some(&id) = tensors_by_name.get(&name) {
                id_inputs.insert(id, view);
            }
        }

        let id_outputs = self
            .graph
            .pool_eval_with_store(&id_inputs, &self.tensor_store, pool)?;

        // Map GlobalIds → String names for outputs.
        let id_to_name: HashMap<_, _> = tensors_by_name
            .iter()
            .map(|(name, &id)| (id, name.as_str()))
            .collect();

        let mut named_outputs = HashMap::new();
        for (id, tensor) in id_outputs {
            if let Some(&name) = id_to_name.get(&id) {
                named_outputs.insert(name.to_string(), tensor);
            }
        }
        Ok(named_outputs)
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
