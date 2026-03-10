use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DTypeError;
use crate::graph::{GlobalId, Graph, Link, Node};
use crate::milli_graph::observer::MilliOpGraphObserver;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::numeric_tensor::NumericTensor;
use crate::tensor_info::{TensorInfo, TensorInfoError};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::time::Instant;

pub mod observer;
pub mod ops;
pub(crate) mod ops_helpers;

#[derive(Debug, thiserror::Error)]
pub enum MilliOpGraphError {
    #[error(transparent)]
    NumericTensorError(#[from] crate::numeric_tensor::NumericTensorError),
    #[error(transparent)]
    NDArrayNumericTensorError(#[from] crate::backends::ndarray_backend::NDArrayNumericTensorError),
    #[error("Unimplemented milli operator: {0}")]
    UnimplementedOperatorError(String),
    #[error("Invalid input for operation {0}")]
    InvalidInput(String),
    #[error(transparent)]
    DTypeError(#[from] DTypeError),
    #[error(transparent)]
    TensorInfoError(#[from] TensorInfoError),
    #[error("Unable to do any type if inference")]
    UnableToInfer,
}

/// Training phase a group of milli-ops belongs to.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum MilliOpPhase {
    Forward,
    Loss,
    Backward,
    Optimizer,
    Custom,
}

/// A group of milli-ops that originated from a single source operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpGroup {
    pub id: GlobalId,
    pub source_op: Option<GlobalId>,
    pub source_graph: Option<GlobalId>,
    pub phase: MilliOpPhase,
    pub label: Option<String>,
    /// If this is a backward group, which forward group it differentiates.
    pub backward_of: Option<GlobalId>,
    /// If this is an optimizer group, which parameter it updates.
    pub optimizes_param: Option<GlobalId>,
}

impl Default for MilliOpGroup {
    fn default() -> Self {
        Self {
            id: GlobalId(0),
            source_op: None,
            source_graph: None,
            phase: MilliOpPhase::Forward,
            label: None,
            backward_of: None,
            optimizes_param: None,
        }
    }
}

/// Metadata about the training structure of a combined milli-op graph.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingMetadata {
    pub loss: Option<GlobalId>,
    pub param_to_grad: HashMap<GlobalId, GlobalId>,
    pub param_to_new_param: HashMap<GlobalId, GlobalId>,
    /// (param_id, state_name) → input state tensor, e.g. (W_id, "m") → m_W_input_id
    pub optimizer_state_inputs: HashMap<(GlobalId, String), GlobalId>,
    /// (param_id, state_name) → output state tensor (after update)
    pub optimizer_state_outputs: HashMap<(GlobalId, String), GlobalId>,
    /// Global optimizer state inputs, e.g. "timestep" → t_input_id
    pub global_state_inputs: HashMap<String, GlobalId>,
    /// Global optimizer state outputs
    pub global_state_outputs: HashMap<String, GlobalId>,
    pub external_inputs: Vec<GlobalId>,

    /// Maps external (symbolic-space) parameter IDs to updated-parameter output IDs.
    ///
    /// This is the caller-facing version of `param_to_new_param` (which uses
    /// combined-space IDs). Use this in the training loop to feed updated
    /// parameters back:
    ///
    /// ```ignore
    /// for (&ext_param, &new_param_output) in &meta.param_updates {
    ///     params.insert(ext_param, results[&new_param_output].clone());
    /// }
    /// ```
    ///
    /// Only populated when the graph was generated via
    /// `SymbolicGraph::generate_milli_graph_with_options`.
    pub param_updates: HashMap<GlobalId, GlobalId>,

    /// Like `optimizer_state_inputs`/`optimizer_state_outputs`, but keyed by
    /// external (symbolic-space) parameter IDs instead of combined-space IDs.
    ///
    /// Only populated via `SymbolicGraph::generate_milli_graph_with_options`.
    pub state_updates: HashMap<(GlobalId, String), StateUpdate>,
}

/// Per-parameter optimizer state entry for the external-facing API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateUpdate {
    /// Feed this tensor as an input each step (the current state value).
    pub input: GlobalId,
    /// Read this tensor from outputs each step (the updated state value).
    pub output: GlobalId,
}

/// Semantic role of a tensor within the training graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TensorRole {
    Loss,
    DataInput {
        name: String,
    },
    Parameter,
    Gradient {
        of_param: GlobalId,
    },
    UpdatedParameter {
        of_param: GlobalId,
    },
    OptimizerState {
        for_param: GlobalId,
        state_name: String,
    },
    UpdatedOptimizerState {
        for_param: GlobalId,
        state_name: String,
    },
}

/// Metadata about a loss graph's inputs/outputs.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LossGraphInfo {
    /// Input tensor for model predictions (logits).
    pub predictions_input: GlobalId,
    /// Input tensor for ground truth targets.
    pub targets_input: GlobalId,
    /// Output tensor containing the scalar loss.
    pub loss_output: GlobalId,
}

// --- Backward generation types ---

/// Options for generating a MilliOpGraph from a SymbolicGraph.
pub struct MilliGraphGenOptions {
    /// Generate backward pass for training.
    pub backward: Option<BackwardGenOptions>,
    /// Generate optimizer updates (requires backward).
    pub optimizer: Option<OptimizerGenOptions>,
}

/// Options for backward pass generation.
pub struct BackwardGenOptions {
    /// Loss computation graph to compose into the training graph.
    pub loss_graph: MilliOpGraph,
    /// How to wire loss_graph inputs to forward outputs or external inputs.
    pub loss_wiring: Vec<LossWiring>,
    /// Which output of loss_graph is the scalar loss to differentiate.
    pub loss_output: GlobalId,
    /// Parameters to compute gradients for.
    pub trainable_params: Vec<GlobalId>,
    /// Ops where gradient flow stops (e.g., for freezing layers).
    pub stop_gradients: HashSet<GlobalId>,
}

/// Specifies how a loss graph input gets wired.
pub struct LossWiring {
    /// Input tensor ID within the loss_graph.
    pub loss_input: GlobalId,
    /// Where this input comes from.
    pub source: LossInputSource,
}

/// Source for a loss graph input.
pub enum LossInputSource {
    /// Wire to this output tensor of the forward SymbolicGraph.
    ForwardOutput(GlobalId),
    /// Create as new external input to the training graph (e.g., labels).
    ExternalInput { name: String },
}

/// Placeholder for optimizer generation options (Phase 8).
pub struct OptimizerGenOptions {
    pub kind: OptimizerKind,
}

/// Optimizer algorithm selection.
pub enum OptimizerKind {
    SGD {
        lr: f32,
    },
    SGDMomentum {
        lr: f32,
        momentum: f32,
        nesterov: bool,
    },
    Adam {
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    },
    AdamW {
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    },
}

/// Error type for milli graph generation.
#[derive(Debug, thiserror::Error)]
pub enum MilliGraphGenError {
    #[error("Backward generation requires at least one trainable parameter")]
    NoTrainableParams,
    #[error("Loss wiring references unknown tensor")]
    InvalidLossWiring,
    #[error("Optimizer requires backward pass")]
    OptimizerWithoutBackward,
}

/// Context provided to `Operation::get_backward_milli_ops()`.
///
/// All tensor IDs are in combined-graph space. The backward implementation
/// uses these IDs as external input keys when constructing its MilliOpGraph,
/// so merge_graph can wire them automatically via sym_to_combined.
pub struct BackwardGenContext {
    /// Maps forward output tensor IDs to their gradient tensor IDs
    /// (both in combined-graph space).
    pub output_grads: HashMap<GlobalId, GlobalId>,
    /// Forward input tensor IDs (in combined-graph space),
    /// ordered to match Operation::inputs().
    pub forward_inputs: Vec<GlobalId>,
    /// Forward output tensor IDs (in combined-graph space),
    /// ordered to match Operation::outputs().
    pub forward_outputs: Vec<GlobalId>,
    /// Shape information for forward tensors (from SymbolicGraph TensorInfo).
    /// Keyed by SymbolicGraph tensor ID. Used for broadcast analysis.
    pub tensor_shapes: HashMap<GlobalId, TensorInfo>,
}

/// Result of backward op generation from `Operation::get_backward_milli_ops()`.
///
/// The backward graph's input_map external keys are combined-graph-space IDs
/// (from BackwardGenContext), so merge_graph wires them automatically.
/// The graph's output_map maps each gradient tensor to the corresponding
/// forward input ID — after merge_graph, the caller finds gradients via
/// `bwd_wiring[fwd_input_id]`.
pub struct BackwardGenResult {
    /// Self-contained backward computation graph.
    pub graph: MilliOpGraph,
    /// Which forward inputs (combined-graph-space) have gradients computed.
    /// Each must appear as an external key in graph.output_map.
    pub differentiable_inputs: Vec<GlobalId>,
}

#[derive(Debug, Clone, Copy, Hash, Ord, PartialOrd, Eq, PartialEq, Serialize, Deserialize)]
pub struct MilliOpGraphTensor {
    global_id: GlobalId,
    pub source_tensor: Option<GlobalId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpGraph {
    global_id: GlobalId,
    pub input_map: HashMap<GlobalId, GlobalId>,
    pub input_ordering: Vec<GlobalId>,
    pub output_map: Option<HashMap<GlobalId, GlobalId>>,
    pub output_ordering: Option<Vec<GlobalId>>,
    ops: HashMap<GlobalId, AnyMilliOp>,
    op_ordering: Vec<GlobalId>,
    tensors: HashMap<GlobalId, MilliOpGraphTensor>,
    groups: HashMap<GlobalId, MilliOpGroup>,
    op_to_group: HashMap<GlobalId, GlobalId>,
    default_group: Option<GlobalId>,
    pub training_metadata: Option<TrainingMetadata>,
}

impl MilliOpGraph {
    pub fn new(
        inputs: impl IntoIterator<Item = GlobalId>,
        rng: &mut impl Rng,
    ) -> (Self, HashMap<GlobalId, GlobalId>) {
        let mut input_map = HashMap::new();
        let mut input_ordering = Vec::new();
        let mut tensors = HashMap::new();
        for input in inputs {
            input_ordering.push(input);
            let global_id = GlobalId::new(rng);
            input_map.insert(input, global_id);
            tensors.insert(
                global_id,
                MilliOpGraphTensor {
                    global_id,
                    source_tensor: None,
                },
            );
        }
        (
            Self {
                global_id: GlobalId::new(rng),
                tensors,
                input_ordering,
                output_ordering: None,
                input_map: input_map.clone(),
                ops: HashMap::new(),
                output_map: None,
                op_ordering: vec![],
                groups: HashMap::new(),
                op_to_group: HashMap::new(),
                default_group: None,
                training_metadata: None,
            },
            input_map,
        )
    }

    pub fn new_empty(rng: &mut impl Rng) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            tensors: HashMap::new(),
            input_ordering: Vec::new(),
            output_ordering: None,
            input_map: HashMap::new(),
            ops: HashMap::new(),
            output_map: None,
            op_ordering: Vec::new(),
            groups: HashMap::new(),
            op_to_group: HashMap::new(),
            default_group: None,
            training_metadata: None,
        }
    }

    /// Add an input with a known external ID. Returns the internal tensor ID.
    pub fn add_input_with_id(&mut self, external_id: GlobalId, rng: &mut impl Rng) -> GlobalId {
        let internal_id = GlobalId::new(rng);
        self.input_map.insert(external_id, internal_id);
        self.input_ordering.push(external_id);
        self.tensors.insert(
            internal_id,
            MilliOpGraphTensor {
                global_id: internal_id,
                source_tensor: Some(external_id),
            },
        );
        internal_id
    }

    /// Mark an internal tensor as an output with a given external ID.
    pub fn add_output(&mut self, internal_id: GlobalId, external_id: GlobalId) {
        let output_map = self.output_map.get_or_insert_with(HashMap::new);
        output_map.insert(internal_id, external_id);
        let output_ordering = self.output_ordering.get_or_insert_with(Vec::new);
        output_ordering.push(external_id);
    }

    /// Merge another MilliOpGraph's ops into self.
    /// `sym_to_combined` maps symbolic-graph tensor IDs to combined-graph internal tensor IDs.
    /// Updated in-place: output tensors from `other` are added to the mapping.
    /// If `group_id` is `Some`, merged ops are assigned to that group; otherwise uses `default_group`.
    pub fn merge_graph(
        &mut self,
        other: MilliOpGraph,
        sym_to_combined: &mut HashMap<GlobalId, GlobalId>,
        rng: &mut impl Rng,
        group_id: Option<GlobalId>,
    ) {
        // Build tensor_map: other's internal IDs → self's internal IDs
        let mut tensor_map: HashMap<GlobalId, GlobalId> = HashMap::new();

        // Map other's inputs: other_internal → sym_to_combined[external]
        for (external, other_internal) in &other.input_map {
            let combined_id = sym_to_combined[external];
            tensor_map.insert(*other_internal, combined_id);
        }

        // For each non-input tensor in other, create a fresh ID in self
        let input_internals: HashSet<GlobalId> = other.input_map.values().copied().collect();
        for (&tensor_id, tensor) in &other.tensors {
            if !input_internals.contains(&tensor_id) {
                let fresh_id = self.get_new_tensor_id(rng);
                // Propagate source_tensor from the other graph's tensor
                if let Some(src) = tensor.source_tensor
                    && let Some(t) = self.tensors.get_mut(&fresh_id)
                {
                    t.source_tensor = Some(src);
                }
                tensor_map.insert(tensor_id, fresh_id);
            }
        }

        // Clone + remap each op, push to self
        for op_id in &other.op_ordering {
            let mut op = other.ops[op_id].clone();
            op.remap_tensors(&tensor_map, rng);
            if let Some(gid) = group_id {
                self.push_op_in_group(op, gid);
            } else {
                self.push_op(op);
            }
        }

        // Map other's outputs back to sym_to_combined
        if let Some(output_map) = &other.output_map {
            for (other_internal, external) in output_map {
                sym_to_combined.insert(*external, tensor_map[other_internal]);
            }
        }
    }

    pub fn get_inputs(&self) -> Vec<GlobalId> {
        self.input_ordering.clone()
    }

    pub fn get_all_tensors(&self) -> HashSet<GlobalId> {
        let mut result = HashSet::new();
        for v in self.input_map.values() {
            result.insert(*v);
        }
        for op in self.ops.values() {
            for tid in op.outputs() {
                result.insert(tid);
            }
        }
        result
    }

    pub fn get_outputs(&self) -> Vec<GlobalId> {
        if let Some(x) = &self.output_ordering {
            x.clone()
        } else if let Some(x) = &self.output_map {
            x.values().cloned().collect()
        } else {
            vec![]
        }
    }

    pub fn set_output_map(&mut self, output_map: impl IntoIterator<Item = (GlobalId, GlobalId)>) {
        assert!(self.output_map.is_none());
        self.output_map = Some(output_map.into_iter().collect());
    }

    pub fn set_output_map_ordered(
        &mut self,
        output_map: HashMap<GlobalId, GlobalId>,
        output_ordering: Vec<GlobalId>,
    ) {
        assert!(self.output_map.is_none());
        assert!(self.output_ordering.is_none());
        self.output_map = Some(output_map);
        self.output_ordering = Some(output_ordering);
    }

    pub fn get_new_tensor_id(&mut self, rng: &mut impl Rng) -> GlobalId {
        let global_id = GlobalId::new(rng);
        self.tensors.insert(
            global_id,
            MilliOpGraphTensor {
                global_id,
                source_tensor: None,
            },
        );
        global_id
    }

    pub fn push_op(&mut self, op: AnyMilliOp) -> GlobalId {
        let id = op.global_id();
        self.ops.insert(id, op);
        self.op_ordering.push(id);
        if let Some(gid) = self.default_group {
            self.op_to_group.insert(id, gid);
        }
        id
    }

    pub fn push_op_in_group(&mut self, op: AnyMilliOp, group_id: GlobalId) -> GlobalId {
        let id = op.global_id();
        self.ops.insert(id, op);
        self.op_ordering.push(id);
        self.op_to_group.insert(id, group_id);
        id
    }

    // --- Group management ---

    pub fn create_group(&mut self, group: MilliOpGroup) -> GlobalId {
        let id = group.id;
        self.groups.insert(id, group);
        id
    }

    pub fn set_default_group(&mut self, group_id: Option<GlobalId>) {
        self.default_group = group_id;
    }

    pub fn get_group(&self, id: GlobalId) -> Option<&MilliOpGroup> {
        self.groups.get(&id)
    }

    pub fn get_group_mut(&mut self, id: GlobalId) -> Option<&mut MilliOpGroup> {
        self.groups.get_mut(&id)
    }

    pub fn ops_in_group(&self, group_id: GlobalId) -> impl Iterator<Item = GlobalId> + '_ {
        self.op_to_group
            .iter()
            .filter(move |&(_, &gid)| gid == group_id)
            .map(|(&op_id, _)| op_id)
    }

    pub fn groups_by_phase(&self, phase: MilliOpPhase) -> impl Iterator<Item = GlobalId> + '_ {
        self.groups
            .iter()
            .filter(move |(_, g)| g.phase == phase)
            .map(|(&id, _)| id)
    }

    // --- Input convenience ---

    /// Add an input where external == internal ID (useful for loss graph building).
    pub fn add_input(&mut self, rng: &mut impl Rng) -> GlobalId {
        let id = GlobalId::new(rng);
        self.input_map.insert(id, id);
        self.input_ordering.push(id);
        self.tensors.insert(
            id,
            MilliOpGraphTensor {
                global_id: id,
                source_tensor: None,
            },
        );
        id
    }

    /// Set outputs where external == internal IDs.
    pub fn set_outputs(&mut self, internal_ids: Vec<GlobalId>) {
        let mut output_map = HashMap::new();
        let mut output_ordering = Vec::new();
        for id in internal_ids {
            output_map.insert(id, id);
            output_ordering.push(id);
        }
        self.output_map = Some(output_map);
        self.output_ordering = Some(output_ordering);
    }

    // --- Tensor role queries ---

    pub fn tensor_role(&self, id: GlobalId) -> Option<TensorRole> {
        let meta = self.training_metadata.as_ref()?;
        if meta.loss == Some(id) {
            return Some(TensorRole::Loss);
        }
        for (&param, &grad) in &meta.param_to_grad {
            if id == param {
                return Some(TensorRole::Parameter);
            }
            if id == grad {
                return Some(TensorRole::Gradient { of_param: param });
            }
        }
        for (&param, &new_param) in &meta.param_to_new_param {
            if id == new_param {
                return Some(TensorRole::UpdatedParameter { of_param: param });
            }
        }
        for ((param_id, state_name), &tensor_id) in &meta.optimizer_state_inputs {
            if tensor_id == id {
                return Some(TensorRole::OptimizerState {
                    for_param: *param_id,
                    state_name: state_name.clone(),
                });
            }
        }
        for ((param_id, state_name), &tensor_id) in &meta.optimizer_state_outputs {
            if tensor_id == id {
                return Some(TensorRole::UpdatedOptimizerState {
                    for_param: *param_id,
                    state_name: state_name.clone(),
                });
            }
        }
        None
    }

    pub fn gradient_of(&self, param: GlobalId) -> Option<GlobalId> {
        self.training_metadata
            .as_ref()?
            .param_to_grad
            .get(&param)
            .copied()
    }

    pub fn param_of_gradient(&self, grad: GlobalId) -> Option<GlobalId> {
        let meta = self.training_metadata.as_ref()?;
        meta.param_to_grad
            .iter()
            .find(|&(_, &g)| g == grad)
            .map(|(&p, _)| p)
    }

    // --- Utility ---

    /// Push a scalar `1.0f32` constant op. Useful as backward pass loss gradient seed.
    pub fn add_scalar_one(&mut self, rng: &mut impl Rng) -> GlobalId {
        use crate::backends::ndarray_backend::NDArrayNumericTensor;
        let data = NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![1.0f32], &vec![1]).unwrap();
        ops::Constant::push_new(self, data, rng)
    }

    // --- Loss graph helpers ---

    /// Cross-entropy loss for classification.
    /// Inputs: logits [batch, classes], targets [batch, classes] (one-hot).
    /// Output: scalar loss.
    ///
    /// Uses numerically stable log-softmax: log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
    pub fn cross_entropy_loss(rng: &mut impl Rng) -> (Self, LossGraphInfo) {
        let mut graph = Self::new_empty(rng);
        let logits = graph.add_input(rng);
        let targets = graph.add_input(rng);

        // Axis constant for class dimension (-1)
        let axis = ops::Constant::new_scalar(&mut graph, -1i64, rng);

        // Numerically stable log_softmax:
        //   shifted = logits - max(logits, axis=-1, keepdims=true)
        let max_logits = ops::ReduceMax::push_new(&mut graph, logits, Some(axis), true, false, rng);
        let shifted = ops::SimpleBinary::sub(&mut graph, logits, max_logits, rng);
        //   log_sum_exp = log(sum(exp(shifted), axis=-1, keepdims=true))
        let exp_shifted = ops::SimpleUnaryOp::exp(&mut graph, shifted, rng);
        let sum_exp =
            ops::ReduceSum::push_new(&mut graph, exp_shifted, Some(axis), true, false, rng);
        let log_sum_exp = ops::SimpleUnaryOp::ln(&mut graph, sum_exp, rng);
        //   log_probs = shifted - log_sum_exp
        let log_probs = ops::SimpleBinary::sub(&mut graph, shifted, log_sum_exp, rng);

        // Negative log likelihood: -mean(sum(targets * log_probs, axis=-1))
        let nll = ops::SimpleBinary::mul(&mut graph, targets, log_probs, rng);
        let sum = ops::ReduceSum::push_new(&mut graph, nll, Some(axis), false, false, rng);
        let neg = ops::SimpleUnaryOp::neg(&mut graph, sum, rng);
        // Mean over batch (reduce all remaining axes)
        let loss = ops::ReduceMean::push_new(&mut graph, neg, None, false, false, rng);

        graph.set_outputs(vec![loss]);

        (
            graph,
            LossGraphInfo {
                predictions_input: logits,
                targets_input: targets,
                loss_output: loss,
            },
        )
    }

    /// Mean squared error loss for regression.
    /// Inputs: predictions [batch, ...], targets [batch, ...] (same shape).
    /// Output: scalar loss.
    pub fn mse_loss(rng: &mut impl Rng) -> (Self, LossGraphInfo) {
        let mut graph = Self::new_empty(rng);
        let predictions = graph.add_input(rng);
        let targets = graph.add_input(rng);

        let diff = ops::SimpleBinary::sub(&mut graph, predictions, targets, rng);
        let sq = ops::SimpleBinary::mul(&mut graph, diff, diff, rng);
        let loss = ops::ReduceMean::push_new(&mut graph, sq, None, false, false, rng);

        graph.set_outputs(vec![loss]);

        (
            graph,
            LossGraphInfo {
                predictions_input: predictions,
                targets_input: targets,
                loss_output: loss,
            },
        )
    }

    /// L1 / Mean Absolute Error loss.
    /// Inputs: predictions [batch, ...], targets [batch, ...] (same shape).
    /// Output: scalar loss.
    pub fn l1_loss(rng: &mut impl Rng) -> (Self, LossGraphInfo) {
        let mut graph = Self::new_empty(rng);
        let predictions = graph.add_input(rng);
        let targets = graph.add_input(rng);

        let diff = ops::SimpleBinary::sub(&mut graph, predictions, targets, rng);
        let abs_diff = ops::SimpleUnaryOp::abs(&mut graph, diff, rng);
        let loss = ops::ReduceMean::push_new(&mut graph, abs_diff, None, false, false, rng);

        graph.set_outputs(vec![loss]);

        (
            graph,
            LossGraphInfo {
                predictions_input: predictions,
                targets_input: targets,
                loss_output: loss,
            },
        )
    }

    #[allow(clippy::type_complexity)]
    pub fn eval<T: MilliOpGraphObserver>(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        observer: &mut T,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        assert!(self.output_map.is_some());

        let mut intermediate_values = HashMap::new();
        for (tensor_id, tensor_value) in inputs {
            if let Some(&internal_id) = self.input_map.get(tensor_id) {
                intermediate_values.insert(internal_id, tensor_value.clone());
            }
        }

        for op_id in &self.op_ordering {
            let op = &self.ops[op_id];
            let start_instant = Instant::now();
            let out_vec: Vec<_> = op.eval(&intermediate_values, backend)?.collect();
            let end_instant = Instant::now();
            observer.on_node_executed(&[op.global_id()], start_instant, end_instant, backend);
            for (tensor_id, value) in out_vec {
                observer.on_tensor_assigned(
                    &[self.tensors[&tensor_id].global_id()],
                    &value,
                    backend,
                );
                intermediate_values.insert(tensor_id, value);
            }
        }

        let mut outputs = HashMap::new();
        for (a, b) in self.output_map.as_ref().unwrap() {
            outputs.insert(*b, intermediate_values[a].clone());
        }

        Ok(Box::new(outputs.into_iter()))
    }
}

impl Graph for MilliOpGraph {
    type Error = ();
    type AnyNode = AnyMilliOp;
    type AnyLink = MilliOpGraphTensor;

    fn global_id(&self) -> GlobalId {
        self.global_id
    }

    fn node_ids(&self) -> impl Iterator<Item = GlobalId> {
        self.ops.keys().cloned()
    }

    fn inner_link_ids(&self) -> impl Iterator<Item = GlobalId> {
        self.get_all_tensors().into_iter()
    }

    fn get_node_by_id(&self, id: &GlobalId) -> Option<&Self::AnyNode> {
        self.ops.get(id)
    }

    fn get_link_by_id(&self, id: &GlobalId) -> Option<&Self::AnyLink> {
        self.tensors.get(id)
    }

    fn input_link_ids(&self) -> impl Iterator<Item = (GlobalId, GlobalId)> {
        self.input_ordering.iter().map(|x| (*x, self.input_map[x]))
    }

    fn constant_link_ids(&self) -> impl Iterator<Item = GlobalId> {
        core::iter::empty()
    }

    fn output_link_ids(&self) -> impl Iterator<Item = (GlobalId, GlobalId)> {
        let mut output = vec![];
        if let Some(ordering) = &self.output_ordering {
            let map = self.output_map.as_ref().unwrap();
            output.extend(ordering.iter().cloned().map(move |x| {
                let tid = map
                    .iter()
                    .find(|(_, id)| **id == x)
                    .map(|(tid, _)| *tid)
                    .expect("output id not found in map");
                (x, tid)
            }))
        } else {
            output.extend(
                self.output_map
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|(tid, id)| (*id, *tid)),
            )
        }
        output.into_iter()
    }
}

// --- Broadcast Analysis ---

/// Computed broadcast information for backward generation.
#[derive(Clone, Debug)]
pub struct BroadcastAnalysis {
    /// Axes where input A broadcasts (A's dim is 1 or rank-padded, B's is not).
    pub a_broadcast_axes: Vec<i64>,
    /// Axes where input B broadcasts.
    pub b_broadcast_axes: Vec<i64>,
    /// Number of dims A was left-padded (for rank alignment).
    pub a_rank_padding: usize,
    /// Number of dims B was left-padded.
    pub b_rank_padding: usize,
}

/// Analyze shapes to determine which axes were broadcast.
///
/// Returns `None` if rank is unknown for either input.
/// Used at generation time to insert correct ReduceSum ops in backward pass.
pub fn analyze_broadcast(a_shape: &TensorInfo, b_shape: &TensorInfo) -> Option<BroadcastAnalysis> {
    let a_rank = a_shape.rank_if_known()?;
    let b_rank = b_shape.rank_if_known()?;
    let target_rank = a_rank.max(b_rank);

    let a_padding = target_rank - a_rank;
    let b_padding = target_rank - b_rank;

    let mut a_broadcast = vec![];
    let mut b_broadcast = vec![];

    for i in 0..target_rank {
        let a_dim = if i < a_padding {
            Some(1) // Left-padded with 1
        } else {
            a_shape.dim_if_known(i - a_padding)
        };

        let b_dim = if i < b_padding {
            Some(1)
        } else {
            b_shape.dim_if_known(i - b_padding)
        };

        match (a_dim, b_dim) {
            (Some(1), Some(d)) if d != 1 => a_broadcast.push(i as i64),
            (Some(1), None) => a_broadcast.push(i as i64), // 1 vs symbolic → assume broadcast
            (Some(d), Some(1)) if d != 1 => b_broadcast.push(i as i64),
            (None, Some(1)) => b_broadcast.push(i as i64),
            _ => {} // Same size or both symbolic (assume equal)
        }
    }

    Some(BroadcastAnalysis {
        a_broadcast_axes: a_broadcast,
        b_broadcast_axes: b_broadcast,
        a_rank_padding: a_padding,
        b_rank_padding: b_padding,
    })
}

// --- Milli-op level backward ---

/// Differentiate through a section of a MilliOpGraph (e.g., the loss group).
///
/// Walks ops in reverse order within the group, calling each op's `backward()`.
/// Accumulates gradients for fan-out (multiple consumers of the same tensor).
/// Returns a gradient map: tensor_id → gradient_tensor_id.
pub fn generate_milli_backward(
    graph: &mut MilliOpGraph,
    group_id: GlobalId,
    initial_grad_map: &HashMap<GlobalId, GlobalId>,
    rng: &mut impl Rng,
) -> HashMap<GlobalId, GlobalId> {
    let mut grad_map = initial_grad_map.clone();

    // Collect ops in this group, sorted by op_ordering position, reversed
    let mut ordered: Vec<(usize, GlobalId)> = graph
        .op_to_group
        .iter()
        .filter(|&(_, &gid)| gid == group_id)
        .filter_map(|(&op_id, _)| {
            graph
                .op_ordering
                .iter()
                .position(|id| *id == op_id)
                .map(|pos| (pos, op_id))
        })
        .collect();
    ordered.sort_by_key(|&(pos, _)| pos);
    ordered.reverse();

    for (_, op_id) in ordered {
        let op = graph.ops[&op_id].clone();

        // Collect output gradients for this op
        let output_grads: HashMap<GlobalId, GlobalId> = op
            .outputs()
            .filter_map(|out_id| grad_map.get(&out_id).map(|&g| (out_id, g)))
            .collect();

        if output_grads.is_empty() {
            continue;
        }

        // Generate backward ops (added directly to the graph)
        if let Some(input_grads) = op.backward(&output_grads, graph, rng) {
            // Accumulate gradients (handles fan-out)
            for (input_id, grad_id) in input_grads {
                grad_map
                    .entry(input_id)
                    .and_modify(|existing| {
                        let sum = ops::SimpleBinary::add(graph, *existing, grad_id, rng);
                        *existing = sum;
                    })
                    .or_insert(grad_id);
            }
        }
    }

    grad_map
}

/// Generate optimizer update ops for all trainable parameters.
///
/// For each parameter with a gradient, generates the appropriate update ops
/// (SGD, Adam, etc.) and populates TrainingMetadata with state mappings.
pub fn generate_optimizer_ops(
    graph: &mut MilliOpGraph,
    training_meta: &mut TrainingMetadata,
    options: &OptimizerGenOptions,
    rng: &mut impl Rng,
) {
    // Shared constants group (for Adam: β^t computations)
    let shared_group = graph.create_group(MilliOpGroup {
        id: GlobalId::new(rng),
        phase: MilliOpPhase::Optimizer,
        label: Some("optimizer_shared".into()),
        ..Default::default()
    });

    // Generate shared state for Adam/AdamW
    let shared = match &options.kind {
        OptimizerKind::Adam { beta1, beta2, .. } | OptimizerKind::AdamW { beta1, beta2, .. } => {
            graph.set_default_group(Some(shared_group));

            // t input (timestep)
            let t_in = graph.add_input(rng);
            training_meta
                .global_state_inputs
                .insert("timestep".into(), t_in);

            let one_i = ops::Constant::new_scalar(graph, 1i64, rng);
            let t_new = ops::SimpleBinary::add(graph, t_in, one_i, rng);
            // Cast to float for pow
            let t_float = ops::Cast::push_new(graph, t_new, crate::dtype::DType::F32, rng);
            training_meta
                .global_state_outputs
                .insert("timestep".into(), t_new);

            let beta1_c = ops::Constant::new_scalar(graph, *beta1, rng);
            let beta2_c = ops::Constant::new_scalar(graph, *beta2, rng);
            let one_f = ops::Constant::new_scalar(graph, 1.0f32, rng);

            // β₁^t, β₂^t
            let beta1_t = ops::Pow::push_new(graph, beta1_c, t_float, rng);
            let beta2_t = ops::Pow::push_new(graph, beta2_c, t_float, rng);
            // 1 - β₁^t, 1 - β₂^t
            let one_minus_beta1_t = ops::SimpleBinary::sub(graph, one_f, beta1_t, rng);
            let one_minus_beta2_t = ops::SimpleBinary::sub(graph, one_f, beta2_t, rng);
            // 1 - β₁, 1 - β₂ (constant, shared across params)
            let one_minus_beta1 = ops::SimpleBinary::sub(graph, one_f, beta1_c, rng);
            let one_minus_beta2 = ops::SimpleBinary::sub(graph, one_f, beta2_c, rng);

            graph.set_default_group(None);
            Some(AdamShared {
                beta1_c,
                beta2_c,
                one_minus_beta1,
                one_minus_beta2,
                one_minus_beta1_t,
                one_minus_beta2_t,
            })
        }
        _ => None,
    };

    // Generate per-parameter update ops
    let params: Vec<(GlobalId, GlobalId)> = training_meta
        .param_to_grad
        .iter()
        .map(|(&p, &g)| (p, g))
        .collect();

    for (param_id, grad_id) in params {
        let param_group = graph.create_group(MilliOpGroup {
            id: GlobalId::new(rng),
            phase: MilliOpPhase::Optimizer,
            label: Some(format!("optimizer_param_{:?}", param_id)),
            optimizes_param: Some(param_id),
            ..Default::default()
        });
        graph.set_default_group(Some(param_group));

        let new_param = match &options.kind {
            OptimizerKind::SGD { lr } => {
                // new_param = param - lr * grad
                let lr_c = ops::Constant::new_scalar(graph, *lr, rng);
                let scaled_grad = ops::SimpleBinary::mul(graph, lr_c, grad_id, rng);
                ops::SimpleBinary::sub(graph, param_id, scaled_grad, rng)
            }
            OptimizerKind::SGDMomentum {
                lr,
                momentum,
                nesterov,
            } => {
                // v_new = momentum * v + grad
                // if nesterov: new_param = param - lr * (momentum * v_new + grad)
                // else:        new_param = param - lr * v_new
                let v_in = graph.add_input(rng);
                training_meta
                    .optimizer_state_inputs
                    .insert((param_id, "velocity".into()), v_in);

                let mom_c = ops::Constant::new_scalar(graph, *momentum, rng);
                let lr_c = ops::Constant::new_scalar(graph, *lr, rng);

                let mom_v = ops::SimpleBinary::mul(graph, mom_c, v_in, rng);
                let v_new = ops::SimpleBinary::add(graph, mom_v, grad_id, rng);

                training_meta
                    .optimizer_state_outputs
                    .insert((param_id, "velocity".into()), v_new);

                let update = if *nesterov {
                    let mom_v_new = ops::SimpleBinary::mul(graph, mom_c, v_new, rng);
                    ops::SimpleBinary::add(graph, mom_v_new, grad_id, rng)
                } else {
                    v_new
                };

                let scaled = ops::SimpleBinary::mul(graph, lr_c, update, rng);
                ops::SimpleBinary::sub(graph, param_id, scaled, rng)
            }
            OptimizerKind::Adam {
                lr,
                epsilon,
                weight_decay,
                ..
            }
            | OptimizerKind::AdamW {
                lr,
                epsilon,
                weight_decay,
                ..
            } => {
                let shared = shared.as_ref().unwrap();
                let is_adamw = matches!(options.kind, OptimizerKind::AdamW { .. });

                // State inputs
                let m_in = graph.add_input(rng);
                let v_in = graph.add_input(rng);
                training_meta
                    .optimizer_state_inputs
                    .insert((param_id, "m".into()), m_in);
                training_meta
                    .optimizer_state_inputs
                    .insert((param_id, "v".into()), v_in);

                let lr_c = ops::Constant::new_scalar(graph, *lr, rng);
                let eps_c = ops::Constant::new_scalar(graph, *epsilon, rng);

                // Apply L2 weight decay to gradient (Adam only, not AdamW)
                let effective_grad = if !is_adamw && *weight_decay != 0.0 {
                    let wd_c = ops::Constant::new_scalar(graph, *weight_decay, rng);
                    let wd_param = ops::SimpleBinary::mul(graph, wd_c, param_id, rng);
                    ops::SimpleBinary::add(graph, grad_id, wd_param, rng)
                } else {
                    grad_id
                };

                // m_new = β₁ * m + (1 - β₁) * grad
                let beta1_m = ops::SimpleBinary::mul(graph, shared.beta1_c, m_in, rng);
                let one_minus_beta1_grad =
                    ops::SimpleBinary::mul(graph, shared.one_minus_beta1, effective_grad, rng);
                let m_new = ops::SimpleBinary::add(graph, beta1_m, one_minus_beta1_grad, rng);

                // v_new = β₂ * v + (1 - β₂) * grad²
                let beta2_v = ops::SimpleBinary::mul(graph, shared.beta2_c, v_in, rng);
                let grad_sq = ops::SimpleBinary::mul(graph, effective_grad, effective_grad, rng);
                let one_minus_beta2_gradsq =
                    ops::SimpleBinary::mul(graph, shared.one_minus_beta2, grad_sq, rng);
                let v_new = ops::SimpleBinary::add(graph, beta2_v, one_minus_beta2_gradsq, rng);

                training_meta
                    .optimizer_state_outputs
                    .insert((param_id, "m".into()), m_new);
                training_meta
                    .optimizer_state_outputs
                    .insert((param_id, "v".into()), v_new);

                // Bias-corrected estimates
                // m_hat = m_new / (1 - β₁^t)
                let m_hat = ops::SimpleBinary::div(graph, m_new, shared.one_minus_beta1_t, rng);
                // v_hat = v_new / (1 - β₂^t)
                let v_hat = ops::SimpleBinary::div(graph, v_new, shared.one_minus_beta2_t, rng);

                // update = lr * m_hat / (sqrt(v_hat) + ε)
                let sqrt_v = ops::SimpleUnaryOp::sqrt(graph, v_hat, rng);
                let denom = ops::SimpleBinary::add(graph, sqrt_v, eps_c, rng);
                let step = ops::SimpleBinary::div(graph, m_hat, denom, rng);
                let scaled_step = ops::SimpleBinary::mul(graph, lr_c, step, rng);

                let mut updated = ops::SimpleBinary::sub(graph, param_id, scaled_step, rng);

                // AdamW: decoupled weight decay
                if is_adamw && *weight_decay != 0.0 {
                    let wd_c = ops::Constant::new_scalar(graph, *weight_decay, rng);
                    let lr_wd = ops::SimpleBinary::mul(graph, lr_c, wd_c, rng);
                    let decay = ops::SimpleBinary::mul(graph, lr_wd, param_id, rng);
                    updated = ops::SimpleBinary::sub(graph, updated, decay, rng);
                }

                updated
            }
        };

        training_meta.param_to_new_param.insert(param_id, new_param);
        graph.set_default_group(None);
    }
}

/// Shared tensors for Adam/AdamW optimizer (bias correction terms).
struct AdamShared {
    beta1_c: GlobalId,
    beta2_c: GlobalId,
    one_minus_beta1: GlobalId,
    one_minus_beta2: GlobalId,
    one_minus_beta1_t: GlobalId,
    one_minus_beta2_t: GlobalId,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::eval_backend::EvalBackend;
    use crate::backends::ndarray_backend::NDArrayNumericTensor;
    use crate::milli_graph::ops::{Constant, SimpleBinary};

    #[test]
    fn test_merge_graph_two_small_graphs() {
        // Graph A: computes add(x, y)
        // Graph B: computes add(z, const(10))
        // Merge: x and z are the same external tensor; B depends on A's output.

        let rng = &mut rand::rng();

        // External tensor IDs (symbolic graph IDs)
        let ext_x = GlobalId::new(rng);
        let ext_y = GlobalId::new(rng);
        let ext_sum_xy = GlobalId::new(rng); // output of graph A

        // Build graph A: add(x, y) -> sum_xy
        let (mut graph_a, a_input_map) = MilliOpGraph::new([ext_x, ext_y], rng);
        let a_x = a_input_map[&ext_x];
        let a_y = a_input_map[&ext_y];
        let a_out = SimpleBinary::add(&mut graph_a, a_x, a_y, rng);
        let mut a_output_map = HashMap::new();
        a_output_map.insert(a_out, ext_sum_xy);
        graph_a.set_output_map(a_output_map);

        // Build graph B: add(sum_xy, const(10)) -> final_out
        let ext_final = GlobalId::new(rng);
        let (mut graph_b, b_input_map) = MilliOpGraph::new([ext_sum_xy], rng);
        let b_in = b_input_map[&ext_sum_xy];
        let b_const = Constant::push_new(
            &mut graph_b,
            NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![10.0f32], &vec![1]).unwrap(),
            rng,
        );
        let b_out = SimpleBinary::add(&mut graph_b, b_in, b_const, rng);
        let mut b_output_map = HashMap::new();
        b_output_map.insert(b_out, ext_final);
        graph_b.set_output_map(b_output_map);

        // Now merge into a combined graph
        let mut combined = MilliOpGraph::new_empty(rng);
        let mut sym_to_combined: HashMap<GlobalId, GlobalId> = HashMap::new();

        // Add external inputs
        let c_x = combined.add_input_with_id(ext_x, rng);
        sym_to_combined.insert(ext_x, c_x);
        let c_y = combined.add_input_with_id(ext_y, rng);
        sym_to_combined.insert(ext_y, c_y);

        // Merge graph A
        combined.merge_graph(graph_a, &mut sym_to_combined, rng, None);

        // After merging A, ext_sum_xy should be in sym_to_combined
        assert!(sym_to_combined.contains_key(&ext_sum_xy));

        // Merge graph B
        combined.merge_graph(graph_b, &mut sym_to_combined, rng, None);

        // After merging B, ext_final should be in sym_to_combined
        assert!(sym_to_combined.contains_key(&ext_final));

        // Set output
        combined.add_output(sym_to_combined[&ext_final], ext_final);

        // Evaluate: x=3, y=5 → add(3,5)=8 → add(8,10)=18
        let mut inputs = HashMap::new();
        inputs.insert(
            ext_x,
            NumericTensor::<DynRank>::from_vec_shape(vec![3.0f32], vec![1]).unwrap(),
        );
        inputs.insert(
            ext_y,
            NumericTensor::<DynRank>::from_vec_shape(vec![5.0f32], vec![1]).unwrap(),
        );

        let mut backend = EvalBackend::NDArray;
        let results: HashMap<_, _> = combined
            .eval(&inputs, &mut (), &mut backend)
            .unwrap()
            .collect();
        let result = &results[&ext_final];
        let values: Vec<f32> = result.flatten().unwrap().try_into().unwrap();
        assert_eq!(values, vec![18.0f32]);
    }

    #[test]
    fn test_group_management() {
        let rng = &mut rand::rng();
        let mut graph = MilliOpGraph::new_empty(rng);

        // Create two groups in different phases
        let g1 = graph.create_group(MilliOpGroup {
            id: GlobalId::new(rng),
            phase: MilliOpPhase::Forward,
            label: Some("conv1".into()),
            ..Default::default()
        });
        let g2 = graph.create_group(MilliOpGroup {
            id: GlobalId::new(rng),
            phase: MilliOpPhase::Backward,
            label: Some("conv1_grad".into()),
            ..Default::default()
        });

        // Push ops into specific groups
        let x = graph.add_input(rng);
        let y = graph.add_input(rng);
        let out1 = SimpleBinary::add(&mut graph, x, y, rng);
        // Manually assign the add op to g1
        // (push_op was already called by SimpleBinary::add, so we re-assign)
        let op_id_1 = graph.op_ordering[0];
        graph.op_to_group.insert(op_id_1, g1);

        // Push another op into g2 via push_op_in_group
        let const_tensor = Constant::push_new(
            &mut graph,
            NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![2.0f32], &vec![1]).unwrap(),
            rng,
        );
        // The constant op was pushed via push_op, manually assign to g2
        let op_id_2 = graph.op_ordering[1];
        graph.op_to_group.insert(op_id_2, g2);

        // Verify ops_in_group
        let g1_ops: Vec<_> = graph.ops_in_group(g1).collect();
        assert_eq!(g1_ops.len(), 1);
        assert_eq!(g1_ops[0], op_id_1);

        let g2_ops: Vec<_> = graph.ops_in_group(g2).collect();
        assert_eq!(g2_ops.len(), 1);
        assert_eq!(g2_ops[0], op_id_2);

        // Verify groups_by_phase
        let forward_groups: Vec<_> = graph.groups_by_phase(MilliOpPhase::Forward).collect();
        assert_eq!(forward_groups.len(), 1);
        assert_eq!(forward_groups[0], g1);

        let backward_groups: Vec<_> = graph.groups_by_phase(MilliOpPhase::Backward).collect();
        assert_eq!(backward_groups.len(), 1);
        assert_eq!(backward_groups[0], g2);

        // Verify group metadata
        assert_eq!(graph.get_group(g1).unwrap().label.as_deref(), Some("conv1"));
        assert_eq!(
            graph.get_group(g2).unwrap().label.as_deref(),
            Some("conv1_grad")
        );

        // Suppress unused warnings
        let _ = (out1, const_tensor);
    }

    #[test]
    fn test_tensor_role() {
        let rng = &mut rand::rng();
        let mut graph = MilliOpGraph::new_empty(rng);

        let loss_id = GlobalId::new(rng);
        let param_id = GlobalId::new(rng);
        let grad_id = GlobalId::new(rng);
        let new_param_id = GlobalId::new(rng);

        graph.training_metadata = Some(TrainingMetadata {
            loss: Some(loss_id),
            param_to_grad: HashMap::from([(param_id, grad_id)]),
            param_to_new_param: HashMap::from([(param_id, new_param_id)]),
            ..Default::default()
        });

        // Loss role
        assert!(matches!(graph.tensor_role(loss_id), Some(TensorRole::Loss)));

        // Parameter role
        assert!(matches!(
            graph.tensor_role(param_id),
            Some(TensorRole::Parameter)
        ));

        // Gradient role
        match graph.tensor_role(grad_id) {
            Some(TensorRole::Gradient { of_param }) => assert_eq!(of_param, param_id),
            other => panic!("expected Gradient, got {:?}", other),
        }

        // Updated parameter role
        match graph.tensor_role(new_param_id) {
            Some(TensorRole::UpdatedParameter { of_param }) => assert_eq!(of_param, param_id),
            other => panic!("expected UpdatedParameter, got {:?}", other),
        }

        // gradient_of / param_of_gradient
        assert_eq!(graph.gradient_of(param_id), Some(grad_id));
        assert_eq!(graph.param_of_gradient(grad_id), Some(param_id));

        // Unknown tensor
        let unknown = GlobalId::new(rng);
        assert!(graph.tensor_role(unknown).is_none());
    }

    /// Helper: evaluate a loss graph with the given prediction and target tensors.
    fn eval_loss(
        graph: &MilliOpGraph,
        loss_info: &LossGraphInfo,
        predictions: NumericTensor<DynRank>,
        targets: NumericTensor<DynRank>,
    ) -> f32 {
        let mut inputs = HashMap::new();
        inputs.insert(loss_info.predictions_input, predictions);
        inputs.insert(loss_info.targets_input, targets);
        let mut backend = EvalBackend::NDArray;
        let results: HashMap<_, _> = graph
            .eval(&inputs, &mut (), &mut backend)
            .unwrap()
            .collect();
        let result = &results[&loss_info.loss_output];
        let values: Vec<f32> = result.flatten().unwrap().try_into().unwrap();
        assert_eq!(values.len(), 1);
        values[0]
    }

    #[test]
    fn test_cross_entropy_loss() {
        let rng = &mut rand::rng();
        let (graph, info) = MilliOpGraph::cross_entropy_loss(rng);

        // batch=1, 3 classes. logits=[0, 0, 100], target=[0, 0, 1] (class 2)
        // Correct class has overwhelming logit → loss ≈ 0
        let logits =
            NumericTensor::<DynRank>::from_vec_shape(vec![0.0f32, 0.0, 100.0], vec![1, 3]).unwrap();
        let targets =
            NumericTensor::<DynRank>::from_vec_shape(vec![0.0f32, 0.0, 1.0], vec![1, 3]).unwrap();
        let loss = eval_loss(&graph, &info, logits, targets);
        assert!(loss.abs() < 1e-4, "expected ~0, got {}", loss);

        // Wrong class: logits=[100, 0, 0], target=[0, 0, 1] (class 2)
        // Correct class has logit 0, dominant class has logit 100 → loss ≈ 100
        let logits2 =
            NumericTensor::<DynRank>::from_vec_shape(vec![100.0f32, 0.0, 0.0], vec![1, 3]).unwrap();
        let targets2 =
            NumericTensor::<DynRank>::from_vec_shape(vec![0.0f32, 0.0, 1.0], vec![1, 3]).unwrap();
        let loss2 = eval_loss(&graph, &info, logits2, targets2);
        assert!(loss2 > 90.0, "expected ~100, got {}", loss2);
    }

    #[test]
    fn test_mse_loss() {
        let rng = &mut rand::rng();
        let (graph, info) = MilliOpGraph::mse_loss(rng);

        // predictions=[1,2,3], targets=[1,2,3] → MSE = 0
        let preds =
            NumericTensor::<DynRank>::from_vec_shape(vec![1.0f32, 2.0, 3.0], vec![1, 3]).unwrap();
        let targets =
            NumericTensor::<DynRank>::from_vec_shape(vec![1.0f32, 2.0, 3.0], vec![1, 3]).unwrap();
        let loss = eval_loss(&graph, &info, preds, targets);
        assert!(loss.abs() < 1e-6, "expected 0, got {}", loss);

        // predictions=[1,2,3], targets=[4,5,6] → diffs=[3,3,3], sq=[9,9,9], mean=9
        let preds2 =
            NumericTensor::<DynRank>::from_vec_shape(vec![1.0f32, 2.0, 3.0], vec![1, 3]).unwrap();
        let targets2 =
            NumericTensor::<DynRank>::from_vec_shape(vec![4.0f32, 5.0, 6.0], vec![1, 3]).unwrap();
        let loss2 = eval_loss(&graph, &info, preds2, targets2);
        assert!((loss2 - 9.0).abs() < 1e-5, "expected 9, got {}", loss2);
    }

    #[test]
    fn test_l1_loss() {
        let rng = &mut rand::rng();
        let (graph, info) = MilliOpGraph::l1_loss(rng);

        // predictions=[1,2,3], targets=[1,2,3] → L1 = 0
        let preds =
            NumericTensor::<DynRank>::from_vec_shape(vec![1.0f32, 2.0, 3.0], vec![1, 3]).unwrap();
        let targets =
            NumericTensor::<DynRank>::from_vec_shape(vec![1.0f32, 2.0, 3.0], vec![1, 3]).unwrap();
        let loss = eval_loss(&graph, &info, preds, targets);
        assert!(loss.abs() < 1e-6, "expected 0, got {}", loss);

        // predictions=[1,2,3], targets=[4,6,9] → diffs=[3,4,6], mean=13/3≈4.333
        let preds2 =
            NumericTensor::<DynRank>::from_vec_shape(vec![1.0f32, 2.0, 3.0], vec![1, 3]).unwrap();
        let targets2 =
            NumericTensor::<DynRank>::from_vec_shape(vec![4.0f32, 6.0, 9.0], vec![1, 3]).unwrap();
        let loss2 = eval_loss(&graph, &info, preds2, targets2);
        let expected = 13.0f32 / 3.0;
        assert!(
            (loss2 - expected).abs() < 1e-5,
            "expected {}, got {}",
            expected,
            loss2
        );
    }

    #[test]
    fn test_broadcast_analysis_same_shape() {
        // Both [batch, hidden] — no broadcasting
        use crate::tensor_info::TensorInfo;
        let a = TensorInfo::from_shape_u64(&[32, 256]);
        let b = TensorInfo::from_shape_u64(&[32, 256]);
        let result = analyze_broadcast(&a, &b).unwrap();
        assert!(result.a_broadcast_axes.is_empty());
        assert!(result.b_broadcast_axes.is_empty());
        assert_eq!(result.a_rank_padding, 0);
        assert_eq!(result.b_rank_padding, 0);
    }

    #[test]
    fn test_broadcast_analysis_bias_add() {
        // a=[batch, hidden], b=[1, hidden] — b broadcasts axis 0
        use crate::tensor_info::TensorInfo;
        let a = TensorInfo::from_shape_u64(&[32, 256]);
        let b = TensorInfo::from_shape_u64(&[1, 256]);
        let result = analyze_broadcast(&a, &b).unwrap();
        assert!(result.a_broadcast_axes.is_empty());
        assert_eq!(result.b_broadcast_axes, vec![0]);
    }

    #[test]
    fn test_broadcast_analysis_rank_mismatch() {
        // a=[batch, seq, hidden], b=[hidden] — b left-padded to [1, 1, hidden]
        use crate::tensor_info::TensorInfo;
        let a = TensorInfo::from_shape_u64(&[32, 128, 256]);
        let b = TensorInfo::from_shape_u64(&[256]);
        let result = analyze_broadcast(&a, &b).unwrap();
        assert!(result.a_broadcast_axes.is_empty());
        assert_eq!(result.b_broadcast_axes, vec![0, 1]);
        assert_eq!(result.a_rank_padding, 0);
        assert_eq!(result.b_rank_padding, 2);
    }

    #[test]
    fn test_broadcast_analysis_both_broadcast() {
        // a=[batch, 1, hidden], b=[1, seq, hidden]
        use crate::tensor_info::TensorInfo;
        let a = TensorInfo::from_shape_u64(&[32, 1, 256]);
        let b = TensorInfo::from_shape_u64(&[1, 128, 256]);
        let result = analyze_broadcast(&a, &b).unwrap();
        assert_eq!(result.a_broadcast_axes, vec![1]); // a's dim 1 is 1
        assert_eq!(result.b_broadcast_axes, vec![0]); // b's dim 0 is 1
    }

    #[test]
    fn test_broadcast_analysis_scalar() {
        // a=[batch, hidden], b=[] (scalar) — b broadcasts all axes
        use crate::tensor_info::TensorInfo;
        let a = TensorInfo::from_shape_u64(&[32, 256]);
        let b = TensorInfo::from_shape_u64(&[]);
        let result = analyze_broadcast(&a, &b).unwrap();
        assert!(result.a_broadcast_axes.is_empty());
        // scalar left-padded to [1, 1], both broadcast
        assert_eq!(result.b_broadcast_axes, vec![0, 1]);
        assert_eq!(result.b_rank_padding, 2);
    }

    // ============================================================
    // Finite difference gradient tests for Phase 5 backward ops
    // ============================================================

    use crate::milli_graph::ops::{ClampMin, MatMul, ReduceMean, ReduceSum, SimpleUnaryOp};

    /// Helper: build a graph that applies `build_fn` to inputs, reduces to scalar via ReduceSum,
    /// then run forward to get the scalar output. Returns the scalar value.
    fn eval_scalar_graph(
        build_fn: &dyn Fn(&mut MilliOpGraph, &[GlobalId], &mut wyrand::WyRand) -> GlobalId,
        input_values: &[Vec<f32>],
        input_shapes: &[Vec<usize>],
    ) -> f32 {
        let rng = &mut wyrand::WyRand::new(42);
        let mut graph = MilliOpGraph::new_empty(rng);

        // Create input tensors
        let mut ext_ids = Vec::new();
        let mut int_ids = Vec::new();
        for _ in 0..input_values.len() {
            let ext = GlobalId::new(rng);
            let int = graph.add_input_with_id(ext, rng);
            ext_ids.push(ext);
            int_ids.push(int);
        }

        // Build the op under test
        let result = build_fn(&mut graph, &int_ids, rng);

        // Reduce to scalar
        let scalar = ReduceSum::push_new(&mut graph, result, None, false, false, rng);

        // Set output
        let ext_out = GlobalId::new(rng);
        graph.add_output(scalar, ext_out);

        // Prepare inputs
        let mut inputs = HashMap::new();
        for (i, ext) in ext_ids.iter().enumerate() {
            let shape_u64: Vec<u64> = input_shapes[i].iter().map(|&s| s as u64).collect();
            let t = NumericTensor::<DynRank>::from_vec_shape(
                input_values[i].clone(),
                input_shapes[i].clone(),
            )
            .unwrap();
            inputs.insert(*ext, t);
        }

        let mut backend = EvalBackend::NDArray;
        let results: HashMap<_, _> = graph
            .eval(&inputs, &mut (), &mut backend)
            .unwrap()
            .collect();
        let val: Vec<f32> = results[&ext_out].flatten().unwrap().try_into().unwrap();
        val[0]
    }

    /// Finite difference gradient check for a unary op.
    /// Computes analytic gradient via backward, compares to (f(x+eps) - f(x-eps)) / (2*eps).
    fn check_unary_backward(
        build_fn: impl Fn(&mut MilliOpGraph, GlobalId, &mut wyrand::WyRand) -> GlobalId,
        input_values: Vec<f32>,
        input_shape: Vec<usize>,
        eps: f32,
        tol: f32,
    ) {
        let rng = &mut wyrand::WyRand::new(42);

        // Build graph: input -> op -> reduce_sum -> scalar
        let mut graph = MilliOpGraph::new_empty(rng);
        let ext_in = GlobalId::new(rng);
        let int_in = graph.add_input_with_id(ext_in, rng);

        // Create a group for the ops so generate_milli_backward can find them
        let group = graph.create_group(MilliOpGroup {
            id: GlobalId::new(rng),
            phase: MilliOpPhase::Forward,
            ..Default::default()
        });
        graph.set_default_group(Some(group));

        let op_out = build_fn(&mut graph, int_in, rng);

        // ReduceSum is outside the group — backward won't try to differentiate through it
        graph.set_default_group(None);
        let scalar = ReduceSum::push_new(&mut graph, op_out, None, false, false, rng);

        // Set output
        let ext_out = GlobalId::new(rng);
        graph.add_output(scalar, ext_out);

        // Seed gradient at op_out with shape-matching ones
        let ones_data = NDArrayNumericTensor::<DynRank>::from_vec_shape(
            vec![1.0f32; input_values.len()],
            &input_shape.iter().map(|&s| s as u64).collect::<Vec<_>>(),
        )
        .unwrap();
        let ones = Constant::push_new(&mut graph, ones_data, rng);
        let mut grad_map = HashMap::new();
        grad_map.insert(op_out, ones);
        let grads = generate_milli_backward(&mut graph, group, &grad_map, rng);

        // The gradient of the input should be in grads
        let grad_tensor_id = grads
            .get(&int_in)
            .unwrap_or_else(|| panic!("No gradient produced for input"));

        // Also expose gradient as output
        let ext_grad = GlobalId::new(rng);
        graph.add_output(*grad_tensor_id, ext_grad);

        // Evaluate to get analytic gradient
        let shape_u64: Vec<u64> = input_shape.iter().map(|&s| s as u64).collect();
        let input_tensor =
            NumericTensor::<DynRank>::from_vec_shape(input_values.clone(), input_shape.clone())
                .unwrap();
        let mut inputs = HashMap::new();
        inputs.insert(ext_in, input_tensor);

        let mut backend = EvalBackend::NDArray;
        let results: HashMap<_, _> = graph
            .eval(&inputs, &mut (), &mut backend)
            .unwrap()
            .collect();
        let analytic_grad: Vec<f32> = results[&ext_grad].flatten().unwrap().try_into().unwrap();

        // Finite difference check for each element
        for i in 0..input_values.len() {
            let mut plus = input_values.clone();
            plus[i] += eps;
            let mut minus = input_values.clone();
            minus[i] -= eps;

            let f_plus = eval_scalar_graph(
                &|g, ids, r| {
                    let out = build_fn(g, ids[0], r);
                    out
                },
                &[plus],
                &[input_shape.clone()],
            );
            let f_minus = eval_scalar_graph(
                &|g, ids, r| {
                    let out = build_fn(g, ids[0], r);
                    out
                },
                &[minus],
                &[input_shape.clone()],
            );

            let numerical_grad = (f_plus - f_minus) / (2.0 * eps);
            let diff = (analytic_grad[i] - numerical_grad).abs();
            let scale = analytic_grad[i].abs().max(numerical_grad.abs()).max(1e-7);
            assert!(
                diff / scale < tol,
                "Gradient mismatch at element {}: analytic={}, numerical={}, rel_diff={}",
                i,
                analytic_grad[i],
                numerical_grad,
                diff / scale
            );
        }
    }

    /// Finite difference gradient check for a binary op.
    fn check_binary_backward(
        build_fn: impl Fn(&mut MilliOpGraph, GlobalId, GlobalId, &mut wyrand::WyRand) -> GlobalId,
        a_values: Vec<f32>,
        b_values: Vec<f32>,
        shape: Vec<usize>,
        eps: f32,
        tol: f32,
    ) {
        let rng = &mut wyrand::WyRand::new(42);

        let mut graph = MilliOpGraph::new_empty(rng);
        let ext_a = GlobalId::new(rng);
        let ext_b = GlobalId::new(rng);
        let int_a = graph.add_input_with_id(ext_a, rng);
        let int_b = graph.add_input_with_id(ext_b, rng);

        let group = graph.create_group(MilliOpGroup {
            id: GlobalId::new(rng),
            phase: MilliOpPhase::Forward,
            ..Default::default()
        });
        graph.set_default_group(Some(group));

        let op_out = build_fn(&mut graph, int_a, int_b, rng);

        // ReduceSum outside the group
        graph.set_default_group(None);
        let scalar = ReduceSum::push_new(&mut graph, op_out, None, false, false, rng);

        let ext_out = GlobalId::new(rng);
        graph.add_output(scalar, ext_out);

        // Seed gradient at op_out with shape-matching ones
        let ones_data = NDArrayNumericTensor::<DynRank>::from_vec_shape(
            vec![1.0f32; a_values.len()],
            &shape.iter().map(|&s| s as u64).collect::<Vec<_>>(),
        )
        .unwrap();
        let ones = Constant::push_new(&mut graph, ones_data, rng);
        let mut grad_map = HashMap::new();
        grad_map.insert(op_out, ones);
        let grads = generate_milli_backward(&mut graph, group, &grad_map, rng);

        let grad_a_id = grads.get(&int_a).expect("No gradient for input a");
        let grad_b_id = grads.get(&int_b).expect("No gradient for input b");

        let ext_grad_a = GlobalId::new(rng);
        graph.add_output(*grad_a_id, ext_grad_a);
        // If both grads point to same tensor, reuse the same external output ID
        let ext_grad_b = if grad_a_id == grad_b_id {
            ext_grad_a
        } else {
            let id = GlobalId::new(rng);
            graph.add_output(*grad_b_id, id);
            id
        };

        let input_a =
            NumericTensor::<DynRank>::from_vec_shape(a_values.clone(), shape.clone()).unwrap();
        let input_b =
            NumericTensor::<DynRank>::from_vec_shape(b_values.clone(), shape.clone()).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert(ext_a, input_a);
        inputs.insert(ext_b, input_b);

        let mut backend = EvalBackend::NDArray;
        let results: HashMap<_, _> = graph
            .eval(&inputs, &mut (), &mut backend)
            .unwrap()
            .collect();
        let analytic_grad_a: Vec<f32> = results[&ext_grad_a].flatten().unwrap().try_into().unwrap();
        let analytic_grad_b: Vec<f32> = results[&ext_grad_b].flatten().unwrap().try_into().unwrap();

        // Check gradients w.r.t. a
        for i in 0..a_values.len() {
            let mut plus = a_values.clone();
            plus[i] += eps;
            let mut minus = a_values.clone();
            minus[i] -= eps;

            let f_plus = eval_scalar_graph(
                &|g, ids, r| build_fn(g, ids[0], ids[1], r),
                &[plus, b_values.clone()],
                &[shape.clone(), shape.clone()],
            );
            let f_minus = eval_scalar_graph(
                &|g, ids, r| build_fn(g, ids[0], ids[1], r),
                &[minus, b_values.clone()],
                &[shape.clone(), shape.clone()],
            );

            let numerical = (f_plus - f_minus) / (2.0 * eps);
            let diff = (analytic_grad_a[i] - numerical).abs();
            let scale = analytic_grad_a[i].abs().max(numerical.abs()).max(1e-7);
            assert!(
                diff / scale < tol,
                "Grad a mismatch at {}: analytic={}, numerical={}, rel_diff={}",
                i,
                analytic_grad_a[i],
                numerical,
                diff / scale
            );
        }

        // Check gradients w.r.t. b
        for i in 0..b_values.len() {
            let mut plus = b_values.clone();
            plus[i] += eps;
            let mut minus = b_values.clone();
            minus[i] -= eps;

            let f_plus = eval_scalar_graph(
                &|g, ids, r| build_fn(g, ids[0], ids[1], r),
                &[a_values.clone(), plus],
                &[shape.clone(), shape.clone()],
            );
            let f_minus = eval_scalar_graph(
                &|g, ids, r| build_fn(g, ids[0], ids[1], r),
                &[a_values.clone(), minus],
                &[shape.clone(), shape.clone()],
            );

            let numerical = (f_plus - f_minus) / (2.0 * eps);
            let diff = (analytic_grad_b[i] - numerical).abs();
            let scale = analytic_grad_b[i].abs().max(numerical.abs()).max(1e-7);
            assert!(
                diff / scale < tol,
                "Grad b mismatch at {}: analytic={}, numerical={}, rel_diff={}",
                i,
                analytic_grad_b[i],
                numerical,
                diff / scale
            );
        }
    }

    #[test]
    fn test_backward_neg() {
        check_unary_backward(
            SimpleUnaryOp::neg,
            vec![1.0, -2.0, 3.0, -0.5],
            vec![2, 2],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_exp() {
        check_unary_backward(
            SimpleUnaryOp::exp,
            vec![0.5, -0.3, 1.0, 0.1],
            vec![2, 2],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_ln() {
        // Use positive values only
        check_unary_backward(
            SimpleUnaryOp::ln,
            vec![1.0, 2.0, 0.5, 3.0],
            vec![2, 2],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_sqrt() {
        check_unary_backward(
            SimpleUnaryOp::sqrt,
            vec![1.0, 4.0, 0.25, 9.0],
            vec![2, 2],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_reciprocal() {
        check_unary_backward(
            SimpleUnaryOp::reciprocal,
            vec![1.0, 2.0, 0.5, -3.0],
            vec![2, 2],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_add() {
        check_binary_backward(
            SimpleBinary::add,
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![2, 2],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_sub() {
        check_binary_backward(
            SimpleBinary::sub,
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![2, 2],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_mul() {
        check_binary_backward(
            SimpleBinary::mul,
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![2, 2],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_div() {
        // Avoid zero in denominator
        check_binary_backward(
            SimpleBinary::div,
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 2.0, 7.0, 3.0],
            vec![2, 2],
            1e-3,
            1e-2,
        );
    }

    /// General finite difference checker for N-input graphs.
    /// `build_fn` receives N internal tensor IDs and builds the computation, returning the final tensor.
    /// Checks analytic gradients (via generate_milli_backward) against numerical finite differences.
    fn check_general_backward(
        build_fn: impl Fn(&mut MilliOpGraph, &[GlobalId], &mut wyrand::WyRand) -> GlobalId,
        input_values: &[Vec<f32>],
        input_shapes: &[Vec<usize>],
        eps: f32,
        tol: f32,
    ) {
        let rng = &mut wyrand::WyRand::new(42);

        let n = input_values.len();
        let mut graph = MilliOpGraph::new_empty(rng);

        let mut ext_ids = Vec::new();
        let mut int_ids = Vec::new();
        for _ in 0..n {
            let ext = GlobalId::new(rng);
            let int = graph.add_input_with_id(ext, rng);
            ext_ids.push(ext);
            int_ids.push(int);
        }

        let group = graph.create_group(MilliOpGroup {
            id: GlobalId::new(rng),
            phase: MilliOpPhase::Forward,
            ..Default::default()
        });
        graph.set_default_group(Some(group));

        let op_out = build_fn(&mut graph, &int_ids, rng);

        graph.set_default_group(None);
        let scalar = ReduceSum::push_new(&mut graph, op_out, None, false, false, rng);

        let ext_out = GlobalId::new(rng);
        graph.add_output(scalar, ext_out);

        // Run a forward-only eval to determine op_out's shape for the gradient seed
        let output_shape = {
            let mut fwd_graph = MilliOpGraph::new_empty(&mut wyrand::WyRand::new(99));
            let rng2 = &mut wyrand::WyRand::new(99);
            let mut fwd_ext = Vec::new();
            let mut fwd_int = Vec::new();
            for _ in 0..n {
                let ext = GlobalId::new(rng2);
                let int = fwd_graph.add_input_with_id(ext, rng2);
                fwd_ext.push(ext);
                fwd_int.push(int);
            }
            let fwd_out = build_fn(&mut fwd_graph, &fwd_int, rng2);
            let fwd_ext_out = GlobalId::new(rng2);
            fwd_graph.add_output(fwd_out, fwd_ext_out);
            let mut fwd_inputs = HashMap::new();
            for (i, ext) in fwd_ext.iter().enumerate() {
                let t = NumericTensor::<DynRank>::from_vec_shape(
                    input_values[i].clone(),
                    input_shapes[i].clone(),
                )
                .unwrap();
                fwd_inputs.insert(*ext, t);
            }
            let mut backend = EvalBackend::NDArray;
            let fwd_results: HashMap<_, _> = fwd_graph
                .eval(&fwd_inputs, &mut (), &mut backend)
                .unwrap()
                .collect();
            fwd_results[&fwd_ext_out].shape().to_vec()
        };

        // Seed gradient with shape-matching ones
        let total_elems: usize = output_shape.iter().map(|&s| s as usize).product();
        let ones_data = NDArrayNumericTensor::<DynRank>::from_vec_shape(
            vec![1.0f32; total_elems.max(1)],
            &output_shape,
        )
        .unwrap();
        let ones = Constant::push_new(&mut graph, ones_data, rng);
        let mut grad_map = HashMap::new();
        grad_map.insert(op_out, ones);
        let grads = generate_milli_backward(&mut graph, group, &grad_map, rng);

        // Expose gradients as outputs
        let mut ext_grad_ids = Vec::new();
        let mut already_output: HashMap<GlobalId, GlobalId> = HashMap::new();
        for i in 0..n {
            if let Some(&grad_id) = grads.get(&int_ids[i]) {
                // Handle same gradient tensor mapped to multiple outputs
                let ext = if let Some(&existing_ext) = already_output.get(&grad_id) {
                    existing_ext
                } else {
                    let ext = GlobalId::new(rng);
                    graph.add_output(grad_id, ext);
                    already_output.insert(grad_id, ext);
                    ext
                };
                ext_grad_ids.push(Some(ext));
            } else {
                ext_grad_ids.push(None);
            }
        }

        // Evaluate
        let mut inputs = HashMap::new();
        for (i, ext) in ext_ids.iter().enumerate() {
            let t = NumericTensor::<DynRank>::from_vec_shape(
                input_values[i].clone(),
                input_shapes[i].clone(),
            )
            .unwrap();
            inputs.insert(*ext, t);
        }

        let mut backend = EvalBackend::NDArray;
        let results: HashMap<_, _> = graph
            .eval(&inputs, &mut (), &mut backend)
            .unwrap()
            .collect();

        // Check each input's gradient
        for input_idx in 0..n {
            let ext_grad = match ext_grad_ids[input_idx] {
                Some(id) => id,
                None => continue,
            };
            let analytic: Vec<f32> = results[&ext_grad].flatten().unwrap().try_into().unwrap();

            for elem in 0..input_values[input_idx].len() {
                let mut plus_vals: Vec<Vec<f32>> = input_values.to_vec();
                plus_vals[input_idx][elem] += eps;
                let mut minus_vals: Vec<Vec<f32>> = input_values.to_vec();
                minus_vals[input_idx][elem] -= eps;

                let f_plus = eval_scalar_graph(&build_fn, &plus_vals, input_shapes);
                let f_minus = eval_scalar_graph(&build_fn, &minus_vals, input_shapes);

                let numerical = (f_plus - f_minus) / (2.0 * eps);
                let diff = (analytic[elem] - numerical).abs();
                let scale = analytic[elem].abs().max(numerical.abs()).max(1e-7);
                assert!(
                    diff / scale < tol,
                    "Input {} elem {}: analytic={}, numerical={}, rel_diff={}",
                    input_idx,
                    elem,
                    analytic[elem],
                    numerical,
                    diff / scale
                );
            }
        }
    }

    #[test]
    fn test_backward_chain_exp_neg() {
        // f(x) = sum(exp(neg(x))) = sum(exp(-x))
        // d/dx = -exp(-x)
        check_general_backward(
            |g, ids, r| {
                let neg = SimpleUnaryOp::neg(g, ids[0], r);
                SimpleUnaryOp::exp(g, neg, r)
            },
            &[vec![0.5, -0.3, 1.0, 0.1]],
            &[vec![2, 2]],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_chain_sqrt_mul() {
        // f(a, b) = sum(sqrt(a * b))
        check_general_backward(
            |g, ids, r| {
                let prod = SimpleBinary::mul(g, ids[0], ids[1], r);
                SimpleUnaryOp::sqrt(g, prod, r)
            },
            &[vec![1.0, 2.0, 3.0, 4.0], vec![4.0, 3.0, 2.0, 1.0]],
            &[vec![2, 2], vec![2, 2]],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_fanout_x_times_x() {
        // f(x) = sum(x * x) — same tensor as both inputs to mul
        // d/dx_i = 2 * x_i
        check_general_backward(
            |g, ids, r| SimpleBinary::mul(g, ids[0], ids[0], r),
            &[vec![1.0, 2.0, 3.0, 4.0]],
            &[vec![2, 2]],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_fanout_x_plus_x() {
        // f(x) = sum(x + x) — same tensor as both inputs to add
        // d/dx_i = 2
        check_general_backward(
            |g, ids, r| SimpleBinary::add(g, ids[0], ids[0], r),
            &[vec![1.0, -2.0, 3.0, -0.5]],
            &[vec![2, 2]],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_multi_consumer() {
        // f(x) = sum(exp(x) + neg(x)) — x fans out to two different ops
        // d/dx_i = exp(x_i) - 1
        check_general_backward(
            |g, ids, r| {
                let e = SimpleUnaryOp::exp(g, ids[0], r);
                let n = SimpleUnaryOp::neg(g, ids[0], r);
                SimpleBinary::add(g, e, n, r)
            },
            &[vec![0.5, -0.3, 1.0, 0.1]],
            &[vec![2, 2]],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_composition_mul_add_sub() {
        // f(a, b) = sum((a + b) * (a - b)) = sum(a^2 - b^2)
        // d/da_i = 2*a_i, d/db_i = -2*b_i
        check_general_backward(
            |g, ids, r| {
                let sum = SimpleBinary::add(g, ids[0], ids[1], r);
                let diff = SimpleBinary::sub(g, ids[0], ids[1], r);
                SimpleBinary::mul(g, sum, diff, r)
            },
            &[vec![1.0, 2.0, 3.0, 4.0], vec![0.5, 1.5, 2.5, 3.5]],
            &[vec![2, 2], vec![2, 2]],
            1e-3,
            1e-2,
        );
    }

    // ============================================================
    // Phase 6: Core Neural Net backward ops
    // ============================================================

    #[test]
    fn test_backward_matmul() {
        // f(A, B) = sum(A @ B), A=[2,3], B=[3,2]
        check_general_backward(
            |g, ids, r| MatMul::push_new(g, ids[0], ids[1], r),
            &[
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], // A [2,3]
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], // B [3,2]
            ],
            &[vec![2, 3], vec![3, 2]],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_clampmin_relu() {
        // f(x) = sum(clamp_min(x, 0.0)) — this is ReLU
        // d/dx_i = 1 if x_i >= 0, 0 otherwise
        check_general_backward(
            |g, ids, r| ClampMin::push_new(g, ids[0], 0.0, r),
            &[vec![1.0, -2.0, 3.0, -0.5]],
            &[vec![2, 2]],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_tanh() {
        // f(x) = sum(tanh(x))
        // d/dx_i = 1 - tanh(x_i)^2
        check_general_backward(
            |g, ids, r| SimpleUnaryOp::trig(g, ids[0], crate::TrigOp::Tanh, r),
            &[vec![0.5, -0.3, 1.0, 0.1]],
            &[vec![2, 2]],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_reduce_sum_all() {
        // f(x) = reduce_sum(x * x, all_axes)
        // This tests ReduceSum backward with axes=None
        // d/dx_i = 2*x_i (from the mul backward), ReduceSum just broadcasts
        check_general_backward(
            |g, ids, r| {
                let sq = SimpleBinary::mul(g, ids[0], ids[0], r);
                ReduceSum::push_new(g, sq, None, false, false, r)
            },
            &[vec![1.0, 2.0, 3.0, 4.0]],
            &[vec![2, 2]],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_reduce_sum_axis() {
        // f(x) = sum(reduce_sum(x, axis=1))
        // x=[2,3], reduce along axis 1 gives [2], then sum gives scalar
        // d/dx_ij = 1 for all i,j
        check_general_backward(
            |g, ids, r| {
                let axis = ops::Constant::new_scalar(g, 1i64, r);
                ReduceSum::push_new(g, ids[0], Some(axis), false, false, r)
            },
            &[vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
            &[vec![2, 3]],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_reduce_mean_all() {
        // f(x) = reduce_mean(x * x, all_axes)
        // d/dx_i = 2*x_i / n where n=4
        check_general_backward(
            |g, ids, r| {
                let sq = SimpleBinary::mul(g, ids[0], ids[0], r);
                ReduceMean::push_new(g, sq, None, false, false, r)
            },
            &[vec![1.0, 2.0, 3.0, 4.0]],
            &[vec![2, 2]],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_reduce_mean_axis() {
        // f(x) = sum(reduce_mean(x, axis=1))
        // x=[2,3], mean along axis 1 gives [2], then sum gives scalar
        // d/dx_ij = 1/3 for all i,j
        check_general_backward(
            |g, ids, r| {
                let axis = ops::Constant::new_scalar(g, 1i64, r);
                ReduceMean::push_new(g, ids[0], Some(axis), false, false, r)
            },
            &[vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
            &[vec![2, 3]],
            1e-3,
            1e-2,
        );
    }

    // ---- Phase 7: Shape ops backward ----

    #[test]
    fn test_backward_reshape() {
        // f(x) = sum(reshape(x, [3, 2])) where x is [2, 3]
        // Gradient flows through reshape; d/dx_i = 1 for all i
        check_general_backward(
            |g, ids, r| {
                let shape = ops::Constant::push_new(
                    g,
                    NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![3i64, 2], &vec![2])
                        .unwrap(),
                    r,
                );
                ops::Reshape::push_new(g, ids[0], shape, false, r)
            },
            &[vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
            &[vec![2, 3]],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_reshape_chain() {
        // f(x) = sum(reshape(x * 2, [6])) where x is [2, 3]
        // d/dx_i = 2 for all i
        check_general_backward(
            |g, ids, r| {
                let two = ops::Constant::new_scalar(g, 2.0f32, r);
                let scaled = SimpleBinary::mul(g, ids[0], two, r);
                let shape = ops::Constant::push_new(
                    g,
                    NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![6i64], &vec![1]).unwrap(),
                    r,
                );
                ops::Reshape::push_new(g, scaled, shape, false, r)
            },
            &[vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
            &[vec![2, 3]],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_transpose() {
        // f(x) = sum(transpose(x, perm=[1,0])) where x is [2, 3]
        // Gradient flows through transpose; d/dx_i = 1 for all i
        check_general_backward(
            |g, ids, r| ops::Transpose::push_new(g, ids[0], Some(vec![1, 0]), r),
            &[vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
            &[vec![2, 3]],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_transpose_3d() {
        // f(x) = sum(transpose(x, perm=[2,0,1])) where x is [2, 3, 4]
        check_general_backward(
            |g, ids, r| ops::Transpose::push_new(g, ids[0], Some(vec![2, 0, 1]), r),
            &[(1..=24).map(|x| x as f32).collect()],
            &[vec![2, 3, 4]],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_transpose_reverse() {
        // f(x) = sum(transpose(x, None)) — reverses all dims
        check_general_backward(
            |g, ids, r| ops::Transpose::push_new(g, ids[0], None, r),
            &[vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
            &[vec![2, 3]],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_squeeze() {
        // f(x) = sum(squeeze(x, axes=[1])) where x is [2, 1, 3]
        check_general_backward(
            |g, ids, r| {
                let axes = ops::Constant::push_new(
                    g,
                    NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![1i64], &vec![1]).unwrap(),
                    r,
                );
                ops::Squeeze::push_new(g, ids[0], axes, r)
            },
            &[vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
            &[vec![2, 1, 3]],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_unsqueeze() {
        // f(x) = sum(unsqueeze(x, axes=[1])) where x is [2, 3]
        check_general_backward(
            |g, ids, r| {
                let axes = ops::Constant::push_new(
                    g,
                    NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![1i64], &vec![1]).unwrap(),
                    r,
                );
                ops::Unsqueeze::push_new(g, ids[0], axes, r)
            },
            &[vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
            &[vec![2, 3]],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_squeeze_unsqueeze_roundtrip() {
        // f(x) = sum(squeeze(unsqueeze(x * 2, axes=[0]), axes=[0]))
        // Should be identity on shape, gradient = 2
        check_general_backward(
            |g, ids, r| {
                let two = ops::Constant::new_scalar(g, 2.0f32, r);
                let scaled = SimpleBinary::mul(g, ids[0], two, r);
                let axes = ops::Constant::push_new(
                    g,
                    NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![0i64], &vec![1]).unwrap(),
                    r,
                );
                let unsq = ops::Unsqueeze::push_new(g, scaled, axes, r);
                ops::Squeeze::push_new(g, unsq, axes, r)
            },
            &[vec![1.0, 2.0, 3.0, 4.0]],
            &[vec![2, 2]],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_concat() {
        // f(a, b) = sum(concat([a, b], axis=0))
        // a=[2,3], b=[3,3] -> output=[5,3]
        // d/da_i = 1, d/db_i = 1
        check_general_backward(
            |g, ids, r| ops::Concat::push_new(g, vec![ids[0], ids[1]], 0, r),
            &[
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            ],
            &[vec![2, 3], vec![3, 3]],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_concat_axis1() {
        // f(a, b) = sum(concat([a, b], axis=1))
        // a=[2,3], b=[2,2] -> output=[2,5]
        check_general_backward(
            |g, ids, r| ops::Concat::push_new(g, vec![ids[0], ids[1]], 1, r),
            &[
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                vec![7.0, 8.0, 9.0, 10.0],
            ],
            &[vec![2, 3], vec![2, 2]],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_concat_same_input() {
        // f(x) = sum(concat([x, x], axis=0))
        // x=[2,3], output=[4,3], d/dx_i = 2 (gradient from both branches)
        check_general_backward(
            |g, ids, r| ops::Concat::push_new(g, vec![ids[0], ids[0]], 0, r),
            &[vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
            &[vec![2, 3]],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_concat_with_computation() {
        // f(a, b) = sum(concat([a * 2, b * 3], axis=0))
        // d/da_i = 2, d/db_i = 3
        check_general_backward(
            |g, ids, r| {
                let two = ops::Constant::new_scalar(g, 2.0f32, r);
                let three = ops::Constant::new_scalar(g, 3.0f32, r);
                let a2 = SimpleBinary::mul(g, ids[0], two, r);
                let b3 = SimpleBinary::mul(g, ids[1], three, r);
                ops::Concat::push_new(g, vec![a2, b3], 0, r)
            },
            &[vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]],
            &[vec![2, 2], vec![2, 2]],
            1e-3,
            1e-2,
        );
    }

    // ---- Broadcast backward tests ----

    #[test]
    fn test_backward_add_broadcast_scalar() {
        // a=[2,3] + b=[1] → grad_a should be [2,3], grad_b should be scalar sum
        check_general_backward(
            |graph, inputs, rng| SimpleBinary::add(graph, inputs[0], inputs[1], rng),
            &[vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![10.0]],
            &[vec![2, 3], vec![1]],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_mul_broadcast_row() {
        // a=[3,4] * b=[1,4] → grad_b must reduce axis 0
        check_general_backward(
            |graph, inputs, rng| SimpleBinary::mul(graph, inputs[0], inputs[1], rng),
            &[
                vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                ],
                vec![0.5, 1.0, 1.5, 2.0],
            ],
            &[vec![3, 4], vec![1, 4]],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_div_broadcast_col() {
        // a=[3,1] / b=[3,4] → grad_a must reduce axis 1
        check_general_backward(
            |graph, inputs, rng| SimpleBinary::div(graph, inputs[0], inputs[1], rng),
            &[
                vec![1.0, 2.0, 3.0],
                vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                ],
            ],
            &[vec![3, 1], vec![3, 4]],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_sub_broadcast_rank() {
        // a=[2,3] - b=[3] → grad_b must reduce axis 0 (rank padding)
        check_general_backward(
            |graph, inputs, rng| SimpleBinary::sub(graph, inputs[0], inputs[1], rng),
            &[vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![10.0, 20.0, 30.0]],
            &[vec![2, 3], vec![3]],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_matmul_broadcast_batch() {
        // A=[2,3,4] @ B=[4,2] → output=[2,3,2], grad_B must sum out batch dim
        check_general_backward(
            |graph, inputs, rng| ops::MatMul::push_new(graph, inputs[0], inputs[1], rng),
            &[
                (1..=24).map(|x| x as f32 * 0.1).collect(),
                vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            ],
            &[vec![2, 3, 4], vec![4, 2]],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_conv_no_bias() {
        // Conv2d: input=[1,1,4,4], weight=[1,1,3,3], no bias, stride=1, pad=0
        // 16 input elements, 9 weight elements
        let input_data: Vec<f32> = (1..=16).map(|x| x as f32 * 0.1).collect();
        let weight_data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        check_general_backward(
            |graph, inputs, rng| {
                ops::Conv::push_new(
                    graph,
                    inputs[0],
                    inputs[1],
                    None,
                    ops::ConvAutoPad::NotSet,
                    vec![1, 1],
                    1,
                    vec![3, 3],
                    vec![0, 0, 0, 0],
                    vec![1, 1],
                    rng,
                )
            },
            &[input_data, weight_data],
            &[vec![1, 1, 4, 4], vec![1, 1, 3, 3]],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_conv_with_bias() {
        // Conv2d with bias: input=[1,1,4,4], weight=[2,1,3,3], bias=[2]
        let input_data: Vec<f32> = (1..=16).map(|x| x as f32 * 0.1).collect();
        let weight_data: Vec<f32> = (1..=18).map(|x| x as f32 * 0.05).collect();
        let bias_data: Vec<f32> = vec![0.1, -0.2];
        check_general_backward(
            |graph, inputs, rng| {
                ops::Conv::push_new(
                    graph,
                    inputs[0],
                    inputs[1],
                    Some(inputs[2]),
                    ops::ConvAutoPad::NotSet,
                    vec![1, 1],
                    1,
                    vec![3, 3],
                    vec![0, 0, 0, 0],
                    vec![1, 1],
                    rng,
                )
            },
            &[input_data, weight_data, bias_data],
            &[vec![1, 1, 4, 4], vec![2, 1, 3, 3], vec![2]],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_conv_with_padding() {
        // Conv2d with padding: input=[1,1,3,3], weight=[1,1,3,3], pad=1
        let input_data: Vec<f32> = (1..=9).map(|x| x as f32 * 0.1).collect();
        let weight_data: Vec<f32> = vec![0.1, 0.2, 0.1, 0.0, 0.5, 0.0, 0.1, 0.2, 0.1];
        check_general_backward(
            |graph, inputs, rng| {
                ops::Conv::push_new(
                    graph,
                    inputs[0],
                    inputs[1],
                    None,
                    ops::ConvAutoPad::NotSet,
                    vec![1, 1],
                    1,
                    vec![3, 3],
                    vec![1, 1, 1, 1],
                    vec![1, 1],
                    rng,
                )
            },
            &[input_data, weight_data],
            &[vec![1, 1, 3, 3], vec![1, 1, 3, 3]],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_conv_stride2() {
        // Conv2d with stride=2: input=[1,1,6,6], weight=[1,1,3,3]
        let input_data: Vec<f32> = (1..=36).map(|x| x as f32 * 0.05).collect();
        let weight_data: Vec<f32> = vec![0.1, 0.2, 0.1, 0.3, 0.0, -0.1, 0.1, 0.2, 0.1];
        check_general_backward(
            |graph, inputs, rng| {
                ops::Conv::push_new(
                    graph,
                    inputs[0],
                    inputs[1],
                    None,
                    ops::ConvAutoPad::NotSet,
                    vec![1, 1],
                    1,
                    vec![3, 3],
                    vec![0, 0, 0, 0],
                    vec![2, 2],
                    rng,
                )
            },
            &[input_data, weight_data],
            &[vec![1, 1, 6, 6], vec![1, 1, 3, 3]],
            1e-3,
            1e-2,
        );
    }

    #[test]
    fn test_backward_gather_axis0() {
        // Embedding lookup: data=[4,3] (4 embeddings of dim 3), indices=[2] (look up 2 rows)
        // f(data) = sum(gather(data, indices, axis=0))
        // Only rows selected by indices get gradient = 1
        let data: Vec<f32> = (1..=12).map(|x| x as f32 * 0.1).collect();
        // indices are not differentiable, but check_general_backward requires all
        // inputs to be f32. We'll use a custom test instead.
        let rng = &mut wyrand::WyRand::new(42);
        let mut graph = MilliOpGraph::new_empty(rng);

        let data_ext = GlobalId::new(rng);
        let data_int = graph.add_input_with_id(data_ext, rng);

        // Create indices as a constant (not a differentiable input)
        let idx_tensor =
            NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![1i64, 3], &vec![2u64]).unwrap();
        let idx_id = ops::Constant::push_new(&mut graph, idx_tensor, rng);

        let group = graph.create_group(MilliOpGroup {
            id: GlobalId::new(rng),
            phase: MilliOpPhase::Forward,
            ..Default::default()
        });
        graph.set_default_group(Some(group));

        let gathered = ops::Gather::push_new(&mut graph, data_int, idx_id, 0, rng);

        graph.set_default_group(None);
        let scalar = ReduceSum::push_new(&mut graph, gathered, None, false, false, rng);
        let out_ext = GlobalId::new(rng);
        graph.add_output(scalar, out_ext);

        // Shape of gathered output: [2, 3]
        let ones_data =
            NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![1.0f32; 6], &vec![2u64, 3])
                .unwrap();
        let ones = ops::Constant::push_new(&mut graph, ones_data, rng);
        let mut grad_map = HashMap::new();
        grad_map.insert(gathered, ones);
        let grads = generate_milli_backward(&mut graph, group, &grad_map, rng);

        let grad_data_id = grads[&data_int];
        let grad_ext = GlobalId::new(rng);
        graph.add_output(grad_data_id, grad_ext);

        let mut inputs = HashMap::new();
        inputs.insert(
            data_ext,
            NumericTensor::<DynRank>::from_vec_shape(data.clone(), vec![4, 3]).unwrap(),
        );

        let mut backend = EvalBackend::NDArray;
        let results: HashMap<_, _> = graph
            .eval(&inputs, &mut (), &mut backend)
            .unwrap()
            .collect();

        let grad: Vec<f32> = results[&grad_ext].flatten().unwrap().try_into().unwrap();
        // Rows 1 and 3 were gathered, so they get gradient 1.0. Rows 0 and 2 get 0.
        assert_eq!(grad.len(), 12);
        // Row 0: [0, 0, 0]
        assert_eq!(&grad[0..3], &[0.0, 0.0, 0.0]);
        // Row 1: [1, 1, 1]
        assert_eq!(&grad[3..6], &[1.0, 1.0, 1.0]);
        // Row 2: [0, 0, 0]
        assert_eq!(&grad[6..9], &[0.0, 0.0, 0.0]);
        // Row 3: [1, 1, 1]
        assert_eq!(&grad[9..12], &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_backward_gather_duplicate_indices() {
        // Same index gathered twice → gradients should accumulate
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2,3]
        let rng = &mut wyrand::WyRand::new(42);
        let mut graph = MilliOpGraph::new_empty(rng);

        let data_ext = GlobalId::new(rng);
        let data_int = graph.add_input_with_id(data_ext, rng);

        // indices = [0, 0, 1, 0] — row 0 gathered 3 times, row 1 once
        let idx_tensor =
            NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![0i64, 0, 1, 0], &vec![4u64])
                .unwrap();
        let idx_id = ops::Constant::push_new(&mut graph, idx_tensor, rng);

        let group = graph.create_group(MilliOpGroup {
            id: GlobalId::new(rng),
            phase: MilliOpPhase::Forward,
            ..Default::default()
        });
        graph.set_default_group(Some(group));

        let gathered = ops::Gather::push_new(&mut graph, data_int, idx_id, 0, rng);

        graph.set_default_group(None);
        let scalar = ReduceSum::push_new(&mut graph, gathered, None, false, false, rng);
        let out_ext = GlobalId::new(rng);
        graph.add_output(scalar, out_ext);

        // gathered shape: [4, 3]
        let ones_data =
            NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![1.0f32; 12], &vec![4u64, 3])
                .unwrap();
        let ones = ops::Constant::push_new(&mut graph, ones_data, rng);
        let mut grad_map = HashMap::new();
        grad_map.insert(gathered, ones);
        let grads = generate_milli_backward(&mut graph, group, &grad_map, rng);

        let grad_data_id = grads[&data_int];
        let grad_ext = GlobalId::new(rng);
        graph.add_output(grad_data_id, grad_ext);

        let mut inputs = HashMap::new();
        inputs.insert(
            data_ext,
            NumericTensor::<DynRank>::from_vec_shape(data, vec![2, 3]).unwrap(),
        );

        let mut backend = EvalBackend::NDArray;
        let results: HashMap<_, _> = graph
            .eval(&inputs, &mut (), &mut backend)
            .unwrap()
            .collect();

        let grad: Vec<f32> = results[&grad_ext].flatten().unwrap().try_into().unwrap();
        // Row 0 gathered 3 times → gradient 3.0 per element
        // Row 1 gathered 1 time → gradient 1.0 per element
        assert_eq!(&grad[0..3], &[3.0, 3.0, 3.0]);
        assert_eq!(&grad[3..6], &[1.0, 1.0, 1.0]);
    }

    // ---- Phase 8: Optimizer tests ----

    /// Helper to build a simple "param * 2 -> reduce_sum -> loss" graph
    /// with backward pass already generated. Returns (graph, param_internal, loss, training_meta).
    fn build_simple_training_graph(
        rng: &mut wyrand::WyRand,
    ) -> (MilliOpGraph, GlobalId, GlobalId, GlobalId, TrainingMetadata) {
        let mut graph = MilliOpGraph::new_empty(rng);
        let param_ext = GlobalId::new(rng);
        let param = graph.add_input_with_id(param_ext, rng);

        // Forward group: param * 2
        let group = graph.create_group(MilliOpGroup {
            id: GlobalId::new(rng),
            phase: MilliOpPhase::Forward,
            ..Default::default()
        });
        graph.set_default_group(Some(group));
        let two = ops::Constant::new_scalar(&mut graph, 2.0f32, rng);
        let scaled = SimpleBinary::mul(&mut graph, param, two, rng);
        graph.set_default_group(None);

        // Loss: reduce_sum (outside the group)
        let loss = ops::ReduceSum::push_new(&mut graph, scaled, None, false, false, rng);

        // Seed gradient on the forward output (scaled), not on loss.
        // Expand scalar 1.0 to match scaled's shape dynamically.
        let ones_scalar = ops::Constant::push_new(
            &mut graph,
            NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![1.0f32], &vec![1]).unwrap(),
            rng,
        );
        let scaled_shape = ops::Shape::push_new(&mut graph, scaled, rng);
        let ones = ops::Expand::push_new(&mut graph, ones_scalar, scaled_shape, rng);
        let mut grad_map = HashMap::new();
        grad_map.insert(scaled, ones);
        let grads = generate_milli_backward(&mut graph, group, &grad_map, rng);

        let mut training_meta = TrainingMetadata::default();
        training_meta.loss = Some(loss);
        training_meta.param_to_grad.insert(param, grads[&param]);

        (graph, param_ext, param, loss, training_meta)
    }

    #[test]
    fn test_optimizer_sgd() {
        // output = sum(param * 2), grad(param) = 2
        // SGD: new_param = param - lr * grad = param - 0.1 * 2 = param - 0.2
        let rng = &mut wyrand::WyRand::new(42);
        let (mut graph, param_ext, param, loss, mut training_meta) =
            build_simple_training_graph(rng);

        generate_optimizer_ops(
            &mut graph,
            &mut training_meta,
            &OptimizerGenOptions {
                kind: OptimizerKind::SGD { lr: 0.1 },
            },
            rng,
        );

        // Set up outputs
        let loss_ext = GlobalId::new(rng);
        graph.add_output(loss, loss_ext);
        let new_param_id = training_meta.param_to_new_param[&param];
        let new_param_ext = GlobalId::new(rng);
        graph.add_output(new_param_id, new_param_ext);

        // Evaluate
        let param_val =
            NumericTensor::<DynRank>::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert(param_ext, param_val);

        let mut backend = EvalBackend::NDArray;
        let results: HashMap<_, _> = graph
            .eval(&inputs, &mut (), &mut backend)
            .unwrap()
            .collect();

        let new_param: Vec<f32> = results[&new_param_ext]
            .flatten()
            .unwrap()
            .try_into()
            .unwrap();
        // new_param = [1, 2, 3, 4] - 0.1 * 2 = [0.8, 1.8, 2.8, 3.8]
        for (i, &v) in new_param.iter().enumerate() {
            let expected = (i + 1) as f32 - 0.2;
            assert!(
                (v - expected).abs() < 1e-5,
                "elem {}: got {}, expected {}",
                i,
                v,
                expected
            );
        }
    }

    #[test]
    fn test_optimizer_adam() {
        let rng = &mut wyrand::WyRand::new(42);
        let (mut graph, param_ext, param, loss, mut training_meta) =
            build_simple_training_graph(rng);

        let lr = 0.001;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let epsilon = 1e-8;
        generate_optimizer_ops(
            &mut graph,
            &mut training_meta,
            &OptimizerGenOptions {
                kind: OptimizerKind::Adam {
                    lr,
                    beta1,
                    beta2,
                    epsilon,
                    weight_decay: 0.0,
                },
            },
            rng,
        );

        // Set up outputs
        let loss_ext = GlobalId::new(rng);
        graph.add_output(loss, loss_ext);
        let new_param_id = training_meta.param_to_new_param[&param];
        let new_param_ext = GlobalId::new(rng);
        graph.add_output(new_param_id, new_param_ext);
        let m_out_id = training_meta.optimizer_state_outputs[&(param, "m".into())];
        let m_out_ext = GlobalId::new(rng);
        graph.add_output(m_out_id, m_out_ext);
        let v_out_id = training_meta.optimizer_state_outputs[&(param, "v".into())];
        let v_out_ext = GlobalId::new(rng);
        graph.add_output(v_out_id, v_out_ext);
        let t_out_id = training_meta.global_state_outputs["timestep"];
        let t_out_ext = GlobalId::new(rng);
        graph.add_output(t_out_id, t_out_ext);

        // Inputs: param, m, v, t
        let param_val =
            NumericTensor::<DynRank>::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let m_in_id = training_meta.optimizer_state_inputs[&(param, "m".into())];
        let v_in_id = training_meta.optimizer_state_inputs[&(param, "v".into())];
        let t_in_id = training_meta.global_state_inputs["timestep"];
        let zeros = NumericTensor::<DynRank>::from_vec_shape(vec![0.0f32; 4], vec![4]).unwrap();
        let t_zero = NumericTensor::<DynRank>::from_vec_shape(vec![0i64], vec![1]).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(param_ext, param_val);
        inputs.insert(m_in_id, zeros.clone());
        inputs.insert(v_in_id, zeros);
        inputs.insert(t_in_id, t_zero);

        let mut backend = EvalBackend::NDArray;
        let results: HashMap<_, _> = graph
            .eval(&inputs, &mut (), &mut backend)
            .unwrap()
            .collect();

        // Verify manually: grad = 2 for all elements
        // t_new = 1
        // m_new = 0.9*0 + 0.1*2 = 0.2
        // v_new = 0.999*0 + 0.001*4 = 0.004
        // m_hat = 0.2 / (1 - 0.9^1) = 0.2 / 0.1 = 2.0
        // v_hat = 0.004 / (1 - 0.999^1) = 0.004 / 0.001 = 4.0
        // step = lr * m_hat / (sqrt(v_hat) + eps) = 0.001 * 2.0 / (2.0 + 1e-8) ≈ 0.001
        // new_param = param - step ≈ param - 0.001

        let new_param: Vec<f32> = results[&new_param_ext]
            .flatten()
            .unwrap()
            .try_into()
            .unwrap();
        let expected_step = lr * 2.0 / (4.0f32.sqrt() + epsilon);
        for (i, &v) in new_param.iter().enumerate() {
            let expected = (i + 1) as f32 - expected_step;
            assert!(
                (v - expected).abs() < 1e-5,
                "elem {}: got {}, expected {}",
                i,
                v,
                expected,
            );
        }

        // Verify m and v outputs
        let m_out: Vec<f32> = results[&m_out_ext].flatten().unwrap().try_into().unwrap();
        let v_out: Vec<f32> = results[&v_out_ext].flatten().unwrap().try_into().unwrap();
        for &m in &m_out {
            assert!((m - 0.2).abs() < 1e-5, "m: got {}, expected 0.2", m);
        }
        for &v in &v_out {
            assert!((v - 0.004).abs() < 1e-5, "v: got {}, expected 0.004", v);
        }
    }

    #[test]
    fn test_optimizer_sgd_momentum() {
        let rng = &mut wyrand::WyRand::new(42);
        let (mut graph, param_ext, param, loss, mut training_meta) =
            build_simple_training_graph(rng);

        let lr = 0.1;
        let momentum = 0.9;
        generate_optimizer_ops(
            &mut graph,
            &mut training_meta,
            &OptimizerGenOptions {
                kind: OptimizerKind::SGDMomentum {
                    lr,
                    momentum,
                    nesterov: false,
                },
            },
            rng,
        );

        let loss_ext = GlobalId::new(rng);
        graph.add_output(loss, loss_ext);
        let new_param_id = training_meta.param_to_new_param[&param];
        let new_param_ext = GlobalId::new(rng);
        graph.add_output(new_param_id, new_param_ext);

        let param_val =
            NumericTensor::<DynRank>::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let v_in_id = training_meta.optimizer_state_inputs[&(param, "velocity".into())];
        let v_zeros = NumericTensor::<DynRank>::from_vec_shape(vec![0.0f32; 4], vec![4]).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(param_ext, param_val);
        inputs.insert(v_in_id, v_zeros);

        let mut backend = EvalBackend::NDArray;
        let results: HashMap<_, _> = graph
            .eval(&inputs, &mut (), &mut backend)
            .unwrap()
            .collect();

        // grad = 2, v_old = 0
        // v_new = 0.9 * 0 + 2 = 2
        // new_param = param - 0.1 * 2 = param - 0.2
        let new_param: Vec<f32> = results[&new_param_ext]
            .flatten()
            .unwrap()
            .try_into()
            .unwrap();
        for (i, &v) in new_param.iter().enumerate() {
            let expected = (i + 1) as f32 - 0.2;
            assert!(
                (v - expected).abs() < 1e-5,
                "elem {}: got {}, expected {}",
                i,
                v,
                expected
            );
        }
    }

    #[test]
    fn test_optimizer_sgd_nesterov() {
        let rng = &mut wyrand::WyRand::new(42);
        let (mut graph, param_ext, param, loss, mut training_meta) =
            build_simple_training_graph(rng);

        let lr = 0.1;
        let momentum = 0.9;
        generate_optimizer_ops(
            &mut graph,
            &mut training_meta,
            &OptimizerGenOptions {
                kind: OptimizerKind::SGDMomentum {
                    lr,
                    momentum,
                    nesterov: true,
                },
            },
            rng,
        );

        let loss_ext = GlobalId::new(rng);
        graph.add_output(loss, loss_ext);
        let new_param_id = training_meta.param_to_new_param[&param];
        let new_param_ext = GlobalId::new(rng);
        graph.add_output(new_param_id, new_param_ext);

        let param_val =
            NumericTensor::<DynRank>::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let v_in_id = training_meta.optimizer_state_inputs[&(param, "velocity".into())];
        let v_zeros = NumericTensor::<DynRank>::from_vec_shape(vec![0.0f32; 4], vec![4]).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(param_ext, param_val);
        inputs.insert(v_in_id, v_zeros);

        let mut backend = EvalBackend::NDArray;
        let results: HashMap<_, _> = graph
            .eval(&inputs, &mut (), &mut backend)
            .unwrap()
            .collect();

        // grad = 2, v_old = 0
        // v_new = 0.9 * 0 + 2 = 2
        // nesterov update = momentum * v_new + grad = 0.9 * 2 + 2 = 3.8
        // new_param = param - 0.1 * 3.8 = param - 0.38
        let new_param: Vec<f32> = results[&new_param_ext]
            .flatten()
            .unwrap()
            .try_into()
            .unwrap();
        for (i, &v) in new_param.iter().enumerate() {
            let expected = (i + 1) as f32 - 0.38;
            assert!(
                (v - expected).abs() < 1e-5,
                "elem {}: got {}, expected {}",
                i,
                v,
                expected
            );
        }
    }

    #[test]
    fn test_optimizer_adamw() {
        let rng = &mut wyrand::WyRand::new(42);
        let (mut graph, param_ext, param, loss, mut training_meta) =
            build_simple_training_graph(rng);

        let lr = 0.001f32;
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let epsilon = 1e-8f32;
        let weight_decay = 0.01f32;
        generate_optimizer_ops(
            &mut graph,
            &mut training_meta,
            &OptimizerGenOptions {
                kind: OptimizerKind::AdamW {
                    lr,
                    beta1,
                    beta2,
                    epsilon,
                    weight_decay,
                },
            },
            rng,
        );

        let loss_ext = GlobalId::new(rng);
        graph.add_output(loss, loss_ext);
        let new_param_id = training_meta.param_to_new_param[&param];
        let new_param_ext = GlobalId::new(rng);
        graph.add_output(new_param_id, new_param_ext);

        let param_vals = vec![1.0f32, 2.0, 3.0, 4.0];
        let param_val =
            NumericTensor::<DynRank>::from_vec_shape(param_vals.clone(), vec![4]).unwrap();
        let m_in_id = training_meta.optimizer_state_inputs[&(param, "m".into())];
        let v_in_id = training_meta.optimizer_state_inputs[&(param, "v".into())];
        let t_in_id = training_meta.global_state_inputs["timestep"];
        let zeros = NumericTensor::<DynRank>::from_vec_shape(vec![0.0f32; 4], vec![4]).unwrap();
        let t_zero = NumericTensor::<DynRank>::from_vec_shape(vec![0i64], vec![1]).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(param_ext, param_val);
        inputs.insert(m_in_id, zeros.clone());
        inputs.insert(v_in_id, zeros);
        inputs.insert(t_in_id, t_zero);

        let mut backend = EvalBackend::NDArray;
        let results: HashMap<_, _> = graph
            .eval(&inputs, &mut (), &mut backend)
            .unwrap()
            .collect();

        // grad = 2 (unmodified for AdamW, no L2 on grad)
        // m_new = 0.1 * 2 = 0.2, v_new = 0.001 * 4 = 0.004
        // m_hat = 0.2/0.1 = 2.0, v_hat = 0.004/0.001 = 4.0
        // adam_step = lr * m_hat / (sqrt(v_hat) + eps) = 0.001 * 2.0 / 2.0 ≈ 0.001
        // decay_step = lr * weight_decay * param
        // new_param = param - adam_step - decay_step
        let adam_step = lr * 2.0 / (4.0f32.sqrt() + epsilon);
        let new_param: Vec<f32> = results[&new_param_ext]
            .flatten()
            .unwrap()
            .try_into()
            .unwrap();
        for (i, &v) in new_param.iter().enumerate() {
            let p = param_vals[i];
            let decay_step = lr * weight_decay * p;
            let expected = p - adam_step - decay_step;
            assert!(
                (v - expected).abs() < 1e-5,
                "elem {}: got {}, expected {}",
                i,
                v,
                expected,
            );
        }
    }

    #[test]
    fn test_optimizer_adam_two_steps() {
        // Run Adam for 2 steps, verify state carries over correctly
        let rng = &mut wyrand::WyRand::new(42);
        let (mut graph, param_ext, param, loss, mut training_meta) =
            build_simple_training_graph(rng);

        let lr = 0.001f32;
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let epsilon = 1e-8f32;
        generate_optimizer_ops(
            &mut graph,
            &mut training_meta,
            &OptimizerGenOptions {
                kind: OptimizerKind::Adam {
                    lr,
                    beta1,
                    beta2,
                    epsilon,
                    weight_decay: 0.0,
                },
            },
            rng,
        );

        let loss_ext = GlobalId::new(rng);
        graph.add_output(loss, loss_ext);
        let new_param_id = training_meta.param_to_new_param[&param];
        let new_param_ext = GlobalId::new(rng);
        graph.add_output(new_param_id, new_param_ext);
        let m_out_id = training_meta.optimizer_state_outputs[&(param, "m".into())];
        let m_out_ext = GlobalId::new(rng);
        graph.add_output(m_out_id, m_out_ext);
        let v_out_id = training_meta.optimizer_state_outputs[&(param, "v".into())];
        let v_out_ext = GlobalId::new(rng);
        graph.add_output(v_out_id, v_out_ext);
        let t_out_id = training_meta.global_state_outputs["timestep"];
        let t_out_ext = GlobalId::new(rng);
        graph.add_output(t_out_id, t_out_ext);

        let m_in_id = training_meta.optimizer_state_inputs[&(param, "m".into())];
        let v_in_id = training_meta.optimizer_state_inputs[&(param, "v".into())];
        let t_in_id = training_meta.global_state_inputs["timestep"];

        let grad = 2.0f32; // constant grad since f(x) = sum(x * 2)
        let mut m = 0.0f32;
        let mut v = 0.0f32;
        let mut p = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut t = 0i64;

        for step in 0..2 {
            let mut inputs = HashMap::new();
            inputs.insert(
                param_ext,
                NumericTensor::<DynRank>::from_vec_shape(p.clone(), vec![4]).unwrap(),
            );
            inputs.insert(
                m_in_id,
                NumericTensor::<DynRank>::from_vec_shape(vec![m; 4], vec![4]).unwrap(),
            );
            inputs.insert(
                v_in_id,
                NumericTensor::<DynRank>::from_vec_shape(vec![v; 4], vec![4]).unwrap(),
            );
            inputs.insert(
                t_in_id,
                NumericTensor::<DynRank>::from_vec_shape(vec![t], vec![1]).unwrap(),
            );

            let mut backend = EvalBackend::NDArray;
            let results: HashMap<_, _> = graph
                .eval(&inputs, &mut (), &mut backend)
                .unwrap()
                .collect();

            // Compute expected values
            t += 1;
            m = beta1 * m + (1.0 - beta1) * grad;
            v = beta2 * v + (1.0 - beta2) * grad * grad;
            let m_hat = m / (1.0 - beta1.powi(t as i32));
            let v_hat = v / (1.0 - beta2.powi(t as i32));
            let adam_step = lr * m_hat / (v_hat.sqrt() + epsilon);
            for x in &mut p {
                *x -= adam_step;
            }

            let actual_param: Vec<f32> = results[&new_param_ext]
                .flatten()
                .unwrap()
                .try_into()
                .unwrap();
            for (i, &actual) in actual_param.iter().enumerate() {
                assert!(
                    (actual - p[i]).abs() < 1e-4,
                    "step {} elem {}: got {}, expected {}",
                    step,
                    i,
                    actual,
                    p[i],
                );
            }

            // Carry state forward
            let m_out: Vec<f32> = results[&m_out_ext].flatten().unwrap().try_into().unwrap();
            let v_out: Vec<f32> = results[&v_out_ext].flatten().unwrap().try_into().unwrap();
            assert!(
                (m_out[0] - m).abs() < 1e-5,
                "step {} m: got {}, expected {}",
                step,
                m_out[0],
                m
            );
            assert!(
                (v_out[0] - v).abs() < 1e-5,
                "step {} v: got {}, expected {}",
                step,
                v_out[0],
                v
            );
        }
    }

    // ---- Phase 9: End-to-end training tests ----

    /// Build a training graph for linear regression: y = matmul(x, W)
    /// with MSE loss: loss = mean((y - targets)^2)
    ///
    /// Returns (graph, x_ext, w_ext, targets_ext, loss_ext, new_w_ext, training_meta)
    fn build_linear_regression_graph(
        lr: f32,
        rng: &mut wyrand::WyRand,
    ) -> (
        MilliOpGraph,
        GlobalId,
        GlobalId,
        GlobalId, // x, w, targets external IDs
        GlobalId,
        GlobalId, // loss, new_w external IDs
        TrainingMetadata,
    ) {
        let mut graph = MilliOpGraph::new_empty(rng);

        // Inputs: x [batch, features], W [features, 1], targets [batch, 1]
        let x_ext = GlobalId::new(rng);
        let w_ext = GlobalId::new(rng);
        let targets_ext = GlobalId::new(rng);
        let x = graph.add_input_with_id(x_ext, rng);
        let w = graph.add_input_with_id(w_ext, rng);
        let targets = graph.add_input_with_id(targets_ext, rng);

        // Forward group: y = matmul(x, W)
        let fwd_group = graph.create_group(MilliOpGroup {
            id: GlobalId::new(rng),
            phase: MilliOpPhase::Forward,
            label: Some("forward".into()),
            ..Default::default()
        });
        graph.set_default_group(Some(fwd_group));
        let y = ops::MatMul::push_new(&mut graph, x, w, rng);
        graph.set_default_group(None);

        // Loss group: mse = mean((y - targets)^2)
        let loss_group = graph.create_group(MilliOpGroup {
            id: GlobalId::new(rng),
            phase: MilliOpPhase::Loss,
            label: Some("mse_loss".into()),
            ..Default::default()
        });
        graph.set_default_group(Some(loss_group));
        let diff = SimpleBinary::sub(&mut graph, y, targets, rng);
        let diff_sq = SimpleBinary::mul(&mut graph, diff, diff, rng);
        let loss = ops::ReduceMean::push_new(&mut graph, diff_sq, None, false, false, rng);
        graph.set_default_group(None);

        // Backward: seed with ones on loss, backward through loss group then forward group
        let ones = ops::Constant::push_new(
            &mut graph,
            NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![1.0f32], &vec![1]).unwrap(),
            rng,
        );
        let mut grad_map = HashMap::new();
        grad_map.insert(loss, ones);

        // Backward through loss group
        let loss_grads = generate_milli_backward(&mut graph, loss_group, &grad_map, rng);
        grad_map.extend(loss_grads);

        // Backward through forward group
        let fwd_grads = generate_milli_backward(&mut graph, fwd_group, &grad_map, rng);
        grad_map.extend(fwd_grads);

        // Training metadata
        let mut training_meta = TrainingMetadata::default();
        training_meta.loss = Some(loss);
        training_meta.param_to_grad.insert(w, grad_map[&w]);

        // Optimizer
        generate_optimizer_ops(
            &mut graph,
            &mut training_meta,
            &OptimizerGenOptions {
                kind: OptimizerKind::SGD { lr },
            },
            rng,
        );

        // Outputs
        let loss_ext = GlobalId::new(rng);
        graph.add_output(loss, loss_ext);
        let new_w = training_meta.param_to_new_param[&w];
        let new_w_ext = GlobalId::new(rng);
        graph.add_output(new_w, new_w_ext);

        (
            graph,
            x_ext,
            w_ext,
            targets_ext,
            loss_ext,
            new_w_ext,
            training_meta,
        )
    }

    #[test]
    fn test_e2e_linear_regression_sgd() {
        // Train y = 3*x1 - 2*x2 using SGD
        // Starting from W = [0, 0], should converge to [3, -2]
        let rng = &mut wyrand::WyRand::new(123);
        let (graph, x_ext, w_ext, targets_ext, loss_ext, new_w_ext, _meta) =
            build_linear_regression_graph(0.01, rng);

        // Training data: x = [[1,0],[0,1],[1,1],[2,1]]
        // true W = [3, -2], so targets = [3, -2, 1, 4]
        let x_data = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0];
        let targets_data = vec![3.0f32, -2.0, 1.0, 4.0];

        let mut w_vals = vec![0.0f32, 0.0]; // start from zeros
        let mut losses = Vec::new();

        for _step in 0..500 {
            let mut inputs = HashMap::new();
            inputs.insert(
                x_ext,
                NumericTensor::<DynRank>::from_vec_shape(x_data.clone(), vec![4, 2]).unwrap(),
            );
            inputs.insert(
                w_ext,
                NumericTensor::<DynRank>::from_vec_shape(w_vals.clone(), vec![2, 1]).unwrap(),
            );
            inputs.insert(
                targets_ext,
                NumericTensor::<DynRank>::from_vec_shape(targets_data.clone(), vec![4, 1]).unwrap(),
            );

            let mut backend = EvalBackend::NDArray;
            let results: HashMap<_, _> = graph
                .eval(&inputs, &mut (), &mut backend)
                .unwrap()
                .collect();

            let loss_val: Vec<f32> = results[&loss_ext].flatten().unwrap().try_into().unwrap();
            losses.push(loss_val[0]);

            let new_w: Vec<f32> = results[&new_w_ext].flatten().unwrap().try_into().unwrap();
            w_vals = new_w;
        }

        // Loss should decrease significantly
        assert!(
            losses.last().unwrap() < &(losses[0] * 0.05),
            "Loss didn't decrease enough: first={}, last={}",
            losses[0],
            losses.last().unwrap(),
        );

        // W should converge near [3, -2] (SGD converges slowly, allow 0.2 tolerance)
        assert!(
            (w_vals[0] - 3.0).abs() < 0.2,
            "W[0] should be ~3.0, got {}",
            w_vals[0],
        );
        assert!(
            (w_vals[1] - (-2.0)).abs() < 0.2,
            "W[1] should be ~-2.0, got {}",
            w_vals[1],
        );
    }

    #[test]
    fn test_e2e_linear_regression_adam() {
        // Same as above but with Adam — should converge faster
        let rng = &mut wyrand::WyRand::new(456);

        let mut graph = MilliOpGraph::new_empty(rng);
        let x_ext = GlobalId::new(rng);
        let w_ext = GlobalId::new(rng);
        let targets_ext = GlobalId::new(rng);
        let x = graph.add_input_with_id(x_ext, rng);
        let w = graph.add_input_with_id(w_ext, rng);
        let targets = graph.add_input_with_id(targets_ext, rng);

        let fwd_group = graph.create_group(MilliOpGroup {
            id: GlobalId::new(rng),
            phase: MilliOpPhase::Forward,
            ..Default::default()
        });
        graph.set_default_group(Some(fwd_group));
        let y = ops::MatMul::push_new(&mut graph, x, w, rng);
        graph.set_default_group(None);

        let loss_group = graph.create_group(MilliOpGroup {
            id: GlobalId::new(rng),
            phase: MilliOpPhase::Loss,
            ..Default::default()
        });
        graph.set_default_group(Some(loss_group));
        let diff = SimpleBinary::sub(&mut graph, y, targets, rng);
        let diff_sq = SimpleBinary::mul(&mut graph, diff, diff, rng);
        let loss = ops::ReduceMean::push_new(&mut graph, diff_sq, None, false, false, rng);
        graph.set_default_group(None);

        let ones = ops::Constant::push_new(
            &mut graph,
            NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![1.0f32], &vec![1]).unwrap(),
            rng,
        );
        let mut grad_map = HashMap::new();
        grad_map.insert(loss, ones);
        let loss_grads = generate_milli_backward(&mut graph, loss_group, &grad_map, rng);
        grad_map.extend(loss_grads);
        let fwd_grads = generate_milli_backward(&mut graph, fwd_group, &grad_map, rng);
        grad_map.extend(fwd_grads);

        let mut training_meta = TrainingMetadata::default();
        training_meta.loss = Some(loss);
        training_meta.param_to_grad.insert(w, grad_map[&w]);

        generate_optimizer_ops(
            &mut graph,
            &mut training_meta,
            &OptimizerGenOptions {
                kind: OptimizerKind::Adam {
                    lr: 0.1,
                    beta1: 0.9,
                    beta2: 0.999,
                    epsilon: 1e-8,
                    weight_decay: 0.0,
                },
            },
            rng,
        );

        let loss_ext = GlobalId::new(rng);
        graph.add_output(loss, loss_ext);
        let new_w = training_meta.param_to_new_param[&w];
        let new_w_ext = GlobalId::new(rng);
        graph.add_output(new_w, new_w_ext);
        let m_out_id = training_meta.optimizer_state_outputs[&(w, "m".into())];
        let m_out_ext = GlobalId::new(rng);
        graph.add_output(m_out_id, m_out_ext);
        let v_out_id = training_meta.optimizer_state_outputs[&(w, "v".into())];
        let v_out_ext = GlobalId::new(rng);
        graph.add_output(v_out_id, v_out_ext);
        let t_out_id = training_meta.global_state_outputs["timestep"];
        let t_out_ext = GlobalId::new(rng);
        graph.add_output(t_out_id, t_out_ext);

        let m_in_id = training_meta.optimizer_state_inputs[&(w, "m".into())];
        let v_in_id = training_meta.optimizer_state_inputs[&(w, "v".into())];
        let t_in_id = training_meta.global_state_inputs["timestep"];

        let x_data = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0];
        let targets_data = vec![3.0f32, -2.0, 1.0, 4.0];

        let mut w_vals = vec![0.0f32, 0.0];
        let mut m_vals = vec![0.0f32, 0.0];
        let mut v_vals = vec![0.0f32, 0.0];
        let mut t_val = 0i64;
        let mut losses = Vec::new();

        for _step in 0..100 {
            let mut inputs = HashMap::new();
            inputs.insert(
                x_ext,
                NumericTensor::<DynRank>::from_vec_shape(x_data.clone(), vec![4, 2]).unwrap(),
            );
            inputs.insert(
                w_ext,
                NumericTensor::<DynRank>::from_vec_shape(w_vals.clone(), vec![2, 1]).unwrap(),
            );
            inputs.insert(
                targets_ext,
                NumericTensor::<DynRank>::from_vec_shape(targets_data.clone(), vec![4, 1]).unwrap(),
            );
            inputs.insert(
                m_in_id,
                NumericTensor::<DynRank>::from_vec_shape(m_vals.clone(), vec![2, 1]).unwrap(),
            );
            inputs.insert(
                v_in_id,
                NumericTensor::<DynRank>::from_vec_shape(v_vals.clone(), vec![2, 1]).unwrap(),
            );
            inputs.insert(
                t_in_id,
                NumericTensor::<DynRank>::from_vec_shape(vec![t_val], vec![1]).unwrap(),
            );

            let mut backend = EvalBackend::NDArray;
            let results: HashMap<_, _> = graph
                .eval(&inputs, &mut (), &mut backend)
                .unwrap()
                .collect();

            let loss_val: Vec<f32> = results[&loss_ext].flatten().unwrap().try_into().unwrap();
            losses.push(loss_val[0]);

            w_vals = results[&new_w_ext].flatten().unwrap().try_into().unwrap();
            m_vals = results[&m_out_ext].flatten().unwrap().try_into().unwrap();
            v_vals = results[&v_out_ext].flatten().unwrap().try_into().unwrap();
            let t_out_val: Vec<i64> = results[&t_out_ext].flatten().unwrap().try_into().unwrap();
            t_val = t_out_val[0];
        }

        // Adam with lr=0.1 should converge quickly
        assert!(
            *losses.last().unwrap() < 0.01,
            "Loss should be near 0, got {}",
            losses.last().unwrap(),
        );
        assert!(
            (w_vals[0] - 3.0).abs() < 0.15,
            "W[0] should be ~3.0, got {}",
            w_vals[0],
        );
        assert!(
            (w_vals[1] - (-2.0)).abs() < 0.15,
            "W[1] should be ~-2.0, got {}",
            w_vals[1],
        );
    }

    #[test]
    fn test_e2e_mlp_xor() {
        // 2-layer MLP solving XOR: x -> matmul(x, W1) -> clamp_min(0) -> matmul(_, W2) -> y
        // XOR: [0,0]->0, [0,1]->1, [1,0]->1, [1,1]->0
        let rng = &mut wyrand::WyRand::new(789);

        let mut graph = MilliOpGraph::new_empty(rng);
        let x_ext = GlobalId::new(rng);
        let w1_ext = GlobalId::new(rng);
        let w2_ext = GlobalId::new(rng);
        let targets_ext = GlobalId::new(rng);
        let x = graph.add_input_with_id(x_ext, rng);
        let w1 = graph.add_input_with_id(w1_ext, rng);
        let w2 = graph.add_input_with_id(w2_ext, rng);
        let targets = graph.add_input_with_id(targets_ext, rng);

        // Forward: h = relu(matmul(x, W1)), y = matmul(h, W2)
        // Using 8 hidden units for more robust XOR learning
        let fwd_group = graph.create_group(MilliOpGroup {
            id: GlobalId::new(rng),
            phase: MilliOpPhase::Forward,
            ..Default::default()
        });
        graph.set_default_group(Some(fwd_group));
        let h_pre = ops::MatMul::push_new(&mut graph, x, w1, rng);
        let h = ops::ClampMin::push_new(&mut graph, h_pre, 0.0, rng);
        let y = ops::MatMul::push_new(&mut graph, h, w2, rng);
        graph.set_default_group(None);

        // Loss: MSE
        let loss_group = graph.create_group(MilliOpGroup {
            id: GlobalId::new(rng),
            phase: MilliOpPhase::Loss,
            ..Default::default()
        });
        graph.set_default_group(Some(loss_group));
        let diff = SimpleBinary::sub(&mut graph, y, targets, rng);
        let diff_sq = SimpleBinary::mul(&mut graph, diff, diff, rng);
        let loss = ops::ReduceMean::push_new(&mut graph, diff_sq, None, false, false, rng);
        graph.set_default_group(None);

        // Backward
        let ones = ops::Constant::push_new(
            &mut graph,
            NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![1.0f32], &vec![1]).unwrap(),
            rng,
        );
        let mut grad_map = HashMap::new();
        grad_map.insert(loss, ones);
        let loss_grads = generate_milli_backward(&mut graph, loss_group, &grad_map, rng);
        grad_map.extend(loss_grads);
        let fwd_grads = generate_milli_backward(&mut graph, fwd_group, &grad_map, rng);
        grad_map.extend(fwd_grads);

        let mut training_meta = TrainingMetadata::default();
        training_meta.loss = Some(loss);
        training_meta.param_to_grad.insert(w1, grad_map[&w1]);
        training_meta.param_to_grad.insert(w2, grad_map[&w2]);

        generate_optimizer_ops(
            &mut graph,
            &mut training_meta,
            &OptimizerGenOptions {
                kind: OptimizerKind::Adam {
                    lr: 0.05,
                    beta1: 0.9,
                    beta2: 0.999,
                    epsilon: 1e-8,
                    weight_decay: 0.0,
                },
            },
            rng,
        );

        // Outputs
        let loss_ext = GlobalId::new(rng);
        graph.add_output(loss, loss_ext);
        let new_w1 = training_meta.param_to_new_param[&w1];
        let new_w1_ext = GlobalId::new(rng);
        graph.add_output(new_w1, new_w1_ext);
        let new_w2 = training_meta.param_to_new_param[&w2];
        let new_w2_ext = GlobalId::new(rng);
        graph.add_output(new_w2, new_w2_ext);

        // Optimizer state outputs (W1)
        let m1_out = training_meta.optimizer_state_outputs[&(w1, "m".into())];
        let m1_out_ext = GlobalId::new(rng);
        graph.add_output(m1_out, m1_out_ext);
        let v1_out = training_meta.optimizer_state_outputs[&(w1, "v".into())];
        let v1_out_ext = GlobalId::new(rng);
        graph.add_output(v1_out, v1_out_ext);
        // Optimizer state outputs (W2)
        let m2_out = training_meta.optimizer_state_outputs[&(w2, "m".into())];
        let m2_out_ext = GlobalId::new(rng);
        graph.add_output(m2_out, m2_out_ext);
        let v2_out = training_meta.optimizer_state_outputs[&(w2, "v".into())];
        let v2_out_ext = GlobalId::new(rng);
        graph.add_output(v2_out, v2_out_ext);
        // Global state
        let t_out = training_meta.global_state_outputs["timestep"];
        let t_out_ext = GlobalId::new(rng);
        graph.add_output(t_out, t_out_ext);

        let m1_in = training_meta.optimizer_state_inputs[&(w1, "m".into())];
        let v1_in = training_meta.optimizer_state_inputs[&(w1, "v".into())];
        let m2_in = training_meta.optimizer_state_inputs[&(w2, "m".into())];
        let v2_in = training_meta.optimizer_state_inputs[&(w2, "v".into())];
        let t_in = training_meta.global_state_inputs["timestep"];

        // XOR data
        let x_data = vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
        let targets_data = vec![0.0f32, 1.0, 1.0, 0.0];

        // Initialize weights with small random values (deterministic)
        // W1: [2, 8] (2 inputs, 8 hidden), W2: [8, 1]
        let mut w_rng = wyrand::WyRand::new(42);
        let mut w1_vals: Vec<f32> = (0..16)
            .map(|_| (rand::Rng::random::<f32>(&mut w_rng) - 0.5) * 1.0)
            .collect();
        let mut w2_vals: Vec<f32> = (0..8)
            .map(|_| (rand::Rng::random::<f32>(&mut w_rng) - 0.5) * 1.0)
            .collect();

        let mut m1_vals = vec![0.0f32; 16];
        let mut v1_vals = vec![0.0f32; 16];
        let mut m2_vals = vec![0.0f32; 8];
        let mut v2_vals = vec![0.0f32; 8];
        let mut t_val = 0i64;
        let mut losses = Vec::new();

        for _step in 0..1000 {
            let mut inputs = HashMap::new();
            inputs.insert(
                x_ext,
                NumericTensor::<DynRank>::from_vec_shape(x_data.clone(), vec![4, 2]).unwrap(),
            );
            inputs.insert(
                w1_ext,
                NumericTensor::<DynRank>::from_vec_shape(w1_vals.clone(), vec![2, 8]).unwrap(),
            );
            inputs.insert(
                w2_ext,
                NumericTensor::<DynRank>::from_vec_shape(w2_vals.clone(), vec![8, 1]).unwrap(),
            );
            inputs.insert(
                targets_ext,
                NumericTensor::<DynRank>::from_vec_shape(targets_data.clone(), vec![4, 1]).unwrap(),
            );
            inputs.insert(
                m1_in,
                NumericTensor::<DynRank>::from_vec_shape(m1_vals.clone(), vec![2, 8]).unwrap(),
            );
            inputs.insert(
                v1_in,
                NumericTensor::<DynRank>::from_vec_shape(v1_vals.clone(), vec![2, 8]).unwrap(),
            );
            inputs.insert(
                m2_in,
                NumericTensor::<DynRank>::from_vec_shape(m2_vals.clone(), vec![8, 1]).unwrap(),
            );
            inputs.insert(
                v2_in,
                NumericTensor::<DynRank>::from_vec_shape(v2_vals.clone(), vec![8, 1]).unwrap(),
            );
            inputs.insert(
                t_in,
                NumericTensor::<DynRank>::from_vec_shape(vec![t_val], vec![1]).unwrap(),
            );

            let mut backend = EvalBackend::NDArray;
            let results: HashMap<_, _> = graph
                .eval(&inputs, &mut (), &mut backend)
                .unwrap()
                .collect();

            let loss_val: Vec<f32> = results[&loss_ext].flatten().unwrap().try_into().unwrap();
            losses.push(loss_val[0]);

            // Update state
            w1_vals = results[&new_w1_ext].flatten().unwrap().try_into().unwrap();
            w2_vals = results[&new_w2_ext].flatten().unwrap().try_into().unwrap();
            m1_vals = results[&m1_out_ext].flatten().unwrap().try_into().unwrap();
            v1_vals = results[&v1_out_ext].flatten().unwrap().try_into().unwrap();
            m2_vals = results[&m2_out_ext].flatten().unwrap().try_into().unwrap();
            v2_vals = results[&v2_out_ext].flatten().unwrap().try_into().unwrap();
            let t_out_val: Vec<i64> = results[&t_out_ext].flatten().unwrap().try_into().unwrap();
            t_val = t_out_val[0];
        }

        // MLP should learn XOR — loss should be very small
        let final_loss = *losses.last().unwrap();
        assert!(
            final_loss < 0.05,
            "XOR MLP didn't converge: final loss = {} (expected < 0.05). \
             Loss trajectory: first={}, mid={}, last={}",
            final_loss,
            losses[0],
            losses[losses.len() / 2],
            final_loss,
        );

        // Verify: loss decreased significantly from start
        assert!(
            final_loss < losses[0] * 0.1,
            "Loss didn't decrease enough: {} -> {}",
            losses[0],
            final_loss,
        );
    }
}

impl crate::graph::Link for MilliOpGraphTensor {
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

impl crate::graph::LinkMetadata for MilliOpGraphTensor {
    // MilliOpGraph tensors don't have rich metadata like dtype/shape at this level
}
