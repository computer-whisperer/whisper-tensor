use std::collections::HashMap;

use crate::backends::eval_backend::EvalRuntimeError;
use crate::numeric_dtype::ONNXDType;
use crate::numeric_tensor::NumericTensorView;
use crate::pool::Pool;
use crate::symbolic_graph::ops::EvalError;
use crate::tensor_rank::DynRank;
use prost::DecodeError;
use rand::Rng;

use crate::graph::GlobalId;
use crate::symbolic_graph::tensor_store::TensorStore;
use crate::symbolic_graph::{ONNXDecodingError, SymbolicGraph, SymbolicGraphMutator};

use crate::scalar_info::ScalarInfoTyped;

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
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

        let all_tensors = self
            .graph
            .pool_eval_with_store(&id_inputs, &self.tensor_store, pool)?;

        // Only return declared output tensors, not all intermediates.
        let output_ids: std::collections::HashSet<GlobalId> =
            self.graph.get_outputs().into_iter().collect();

        // Map GlobalIds → String names for outputs.
        let id_to_name: HashMap<_, _> = tensors_by_name
            .iter()
            .map(|(name, &id)| (id, name.as_str()))
            .collect();

        let mut named_outputs = HashMap::new();
        for (id, tensor) in all_tensors {
            if output_ids.contains(&id)
                && let Some(&name) = id_to_name.get(&id)
            {
                named_outputs.insert(name.to_string(), tensor);
            }
        }
        Ok(named_outputs)
    }

    /// Evaluate with ONNXTensor inputs/outputs (supports string tensors).
    pub fn eval_pool_onnx<'p, P: Pool + 'p>(
        &self,
        inputs: HashMap<String, crate::numeric_dtype::ONNXTensorView<'_>>,
        pool: &'p P,
    ) -> Result<HashMap<String, crate::numeric_dtype::ONNXTensor<'p, P>>, EvalError> {
        use crate::numeric_dtype::ONNXTensorView;
        let tensors_by_name = self.graph.get_tensors_by_name();

        // Map String names → GlobalIds.
        let mut id_inputs: HashMap<GlobalId, ONNXTensorView<'_>> = HashMap::new();
        for (name, view) in inputs {
            if let Some(&id) = tensors_by_name.get(&name) {
                id_inputs.insert(id, view);
            }
        }

        // Load numeric constants from tensor_store.
        let bridged = self.graph.load_numeric_constants(&self.tensor_store);
        let bridged_views: Vec<_> = bridged.iter().map(|(id, t)| (*id, t.view())).collect();
        for (id, view) in &bridged_views {
            id_inputs.entry(*id).or_insert_with(|| {
                ONNXTensorView::Numeric(crate::numeric_tensor::NumericTensorView::new(
                    view.data(),
                    view.layout().clone(),
                ))
            });
        }

        let all_tensors = self.graph.eval_pool_onnx(&id_inputs, pool, &mut ())?;

        let output_ids: std::collections::HashSet<GlobalId> =
            self.graph.get_outputs().into_iter().collect();
        let id_to_name: HashMap<_, _> = tensors_by_name
            .iter()
            .map(|(name, &id)| (id, name.as_str()))
            .collect();

        let mut named_outputs = HashMap::new();
        for (id, tensor) in all_tensors {
            if output_ids.contains(&id)
                && let Some(&name) = id_to_name.get(&id)
            {
                named_outputs.insert(name.to_string(), tensor);
            }
        }
        Ok(named_outputs)
    }

    #[allow(clippy::type_complexity)]
    pub fn get_input_tensor_info(
        &self,
    ) -> Result<HashMap<String, (ONNXDType, Vec<Option<u64>>)>, EvalRuntimeError> {
        let input_ids = self.graph.get_inputs();
        let mut results = HashMap::new();
        for tensor_id in input_ids {
            if let Some(tensor_info) = self.graph.get_tensor_info(tensor_id)
                && let (Some(onnx_dtype), Some(name), Some(shape)) =
                    (tensor_info.dtype(), tensor_info.name(), tensor_info.shape())
            {
                let shape: Vec<_> = shape
                    .iter()
                    .map(|x| match x {
                        ScalarInfoTyped::Numeric(a) => Some(*a),
                        _ => None,
                    })
                    .collect();
                results.insert(name.clone(), (onnx_dtype, shape));
            }
        }
        Ok(results)
    }
}
