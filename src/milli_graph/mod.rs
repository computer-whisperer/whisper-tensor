use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DTypeError;
use crate::graph::{GlobalId, Graph, Link, Node};
use crate::milli_graph::observer::MilliOpGraphObserver;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::numeric_tensor::NumericTensor;
use crate::tensor_info::{TensorInfo, TensorInfoError};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::time::Instant;
use rand::Rng;

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
    pub optimizer_state_inputs: Vec<GlobalId>,
    pub optimizer_state_outputs: Vec<GlobalId>,
    pub global_state_inputs: Vec<GlobalId>,
    pub global_state_outputs: Vec<GlobalId>,
    pub external_inputs: Vec<GlobalId>,
}

/// Semantic role of a tensor within the training graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TensorRole {
    Loss,
    DataInput { name: String },
    Parameter,
    Gradient { of_param: GlobalId },
    UpdatedParameter { of_param: GlobalId },
    OptimizerState { for_param: GlobalId, state_name: String },
    UpdatedOptimizerState { for_param: GlobalId, state_name: String },
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
    SGD { lr: f32 },
    SGDMomentum { lr: f32, momentum: f32, nesterov: bool },
    Adam { lr: f32, beta1: f32, beta2: f32, epsilon: f32, weight_decay: f32 },
    AdamW { lr: f32, beta1: f32, beta2: f32, epsilon: f32, weight_decay: f32 },
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
    pub fn new(inputs: impl IntoIterator<Item = GlobalId>, rng: &mut impl Rng) -> (Self, HashMap<GlobalId, GlobalId>) {
        let mut input_map = HashMap::new();
        let mut input_ordering = Vec::new();
        let mut tensors = HashMap::new();
        for input in inputs {
            input_ordering.push(input.clone());
            let global_id = GlobalId::new(rng);
            input_map.insert(input.clone(), global_id);
            tensors.insert(global_id, MilliOpGraphTensor { global_id, source_tensor: None });
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
        self.tensors.insert(internal_id, MilliOpGraphTensor { global_id: internal_id, source_tensor: Some(external_id) });
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
                if let Some(src) = tensor.source_tensor {
                    if let Some(t) = self.tensors.get_mut(&fresh_id) {
                        t.source_tensor = Some(src);
                    }
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

    pub fn set_output_map(
        &mut self,
        output_map: impl IntoIterator<Item = (GlobalId, GlobalId)>,
    ) {
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
        self.tensors.insert(global_id, MilliOpGraphTensor { global_id, source_tensor: None });
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
        self.op_to_group.iter()
            .filter(move |&(_, &gid)| gid == group_id)
            .map(|(&op_id, _)| op_id)
    }

    pub fn groups_by_phase(&self, phase: MilliOpPhase) -> impl Iterator<Item = GlobalId> + '_ {
        self.groups.iter()
            .filter(move |(_, g)| g.phase == phase)
            .map(|(&id, _)| id)
    }

    // --- Input convenience ---

    /// Add an input where external == internal ID (useful for loss graph building).
    pub fn add_input(&mut self, rng: &mut impl Rng) -> GlobalId {
        let id = GlobalId::new(rng);
        self.input_map.insert(id, id);
        self.input_ordering.push(id);
        self.tensors.insert(id, MilliOpGraphTensor { global_id: id, source_tensor: None });
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
        None
    }

    pub fn gradient_of(&self, param: GlobalId) -> Option<GlobalId> {
        self.training_metadata.as_ref()?.param_to_grad.get(&param).copied()
    }

    pub fn param_of_gradient(&self, grad: GlobalId) -> Option<GlobalId> {
        let meta = self.training_metadata.as_ref()?;
        meta.param_to_grad.iter()
            .find(|&(_, &g)| g == grad)
            .map(|(&p, _)| p)
    }

    // --- Utility ---

    /// Push a scalar `1.0f32` constant op. Useful for backward pass seed.
    pub fn add_constant_ones_like(&mut self, _tensor: GlobalId, rng: &mut impl Rng) -> GlobalId {
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
        let sum_exp = ops::ReduceSum::push_new(&mut graph, exp_shifted, Some(axis), true, false, rng);
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

        (graph, LossGraphInfo {
            predictions_input: logits,
            targets_input: targets,
            loss_output: loss,
        })
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

        (graph, LossGraphInfo {
            predictions_input: predictions,
            targets_input: targets,
            loss_output: loss,
        })
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

        (graph, LossGraphInfo {
            predictions_input: predictions,
            targets_input: targets,
            loss_output: loss,
        })
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn eval<T: MilliOpGraphObserver>(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        observer: &mut T,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError> {
        assert!(self.output_map.is_some());

        let mut intermediate_values = HashMap::new();
        for (tensor_id, tensor_value) in inputs {
            intermediate_values.insert(self.input_map[tensor_id], tensor_value.clone());
        }

        for op_id in &self.op_ordering {
            let op = &self.ops[op_id];
            let start_instant = Instant::now();
            let out_vec: Vec<_> = op.eval(&intermediate_values, backend)?.collect();
            let end_instant = Instant::now();
            observer.on_node_executed(
                &[op.global_id()],
                start_instant,
                end_instant,
                backend,
            );
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
            outputs.insert(b.clone(), intermediate_values[a].clone());
        }

        Ok(Box::new(outputs.into_iter()))
    }
}

impl Graph for MilliOpGraph
{
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
        self.input_ordering
            .iter()
            .map(|x| (x.clone(), self.input_map[x]))
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
                    .map(|(tid, id)| (id.clone(), *tid)),
            )
        }
        output.into_iter()
    }
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
        let (mut graph_a, a_input_map) =
            MilliOpGraph::new([ext_x, ext_y], rng);
        let a_x = a_input_map[&ext_x];
        let a_y = a_input_map[&ext_y];
        let a_out = SimpleBinary::add(&mut graph_a, a_x, a_y, rng);
        let mut a_output_map = HashMap::new();
        a_output_map.insert(a_out, ext_sum_xy);
        graph_a.set_output_map(a_output_map);

        // Build graph B: add(sum_xy, const(10)) -> final_out
        let ext_final = GlobalId::new(rng);
        let (mut graph_b, b_input_map) =
            MilliOpGraph::new([ext_sum_xy], rng);
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
        let results: HashMap<_, _> = combined.eval(&inputs, &mut (), &mut backend).unwrap().collect();
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
        assert_eq!(graph.get_group(g2).unwrap().label.as_deref(), Some("conv1_grad"));

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
        assert!(matches!(graph.tensor_role(param_id), Some(TensorRole::Parameter)));

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
        let results: HashMap<_, _> = graph.eval(&inputs, &mut (), &mut backend).unwrap().collect();
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
        let logits = NumericTensor::<DynRank>::from_vec_shape(
            vec![0.0f32, 0.0, 100.0], vec![1, 3],
        ).unwrap();
        let targets = NumericTensor::<DynRank>::from_vec_shape(
            vec![0.0f32, 0.0, 1.0], vec![1, 3],
        ).unwrap();
        let loss = eval_loss(&graph, &info, logits, targets);
        assert!(loss.abs() < 1e-4, "expected ~0, got {}", loss);

        // Wrong class: logits=[100, 0, 0], target=[0, 0, 1] (class 2)
        // Correct class has logit 0, dominant class has logit 100 → loss ≈ 100
        let logits2 = NumericTensor::<DynRank>::from_vec_shape(
            vec![100.0f32, 0.0, 0.0], vec![1, 3],
        ).unwrap();
        let targets2 = NumericTensor::<DynRank>::from_vec_shape(
            vec![0.0f32, 0.0, 1.0], vec![1, 3],
        ).unwrap();
        let loss2 = eval_loss(&graph, &info, logits2, targets2);
        assert!(loss2 > 90.0, "expected ~100, got {}", loss2);
    }

    #[test]
    fn test_mse_loss() {
        let rng = &mut rand::rng();
        let (graph, info) = MilliOpGraph::mse_loss(rng);

        // predictions=[1,2,3], targets=[1,2,3] → MSE = 0
        let preds = NumericTensor::<DynRank>::from_vec_shape(
            vec![1.0f32, 2.0, 3.0], vec![1, 3],
        ).unwrap();
        let targets = NumericTensor::<DynRank>::from_vec_shape(
            vec![1.0f32, 2.0, 3.0], vec![1, 3],
        ).unwrap();
        let loss = eval_loss(&graph, &info, preds, targets);
        assert!(loss.abs() < 1e-6, "expected 0, got {}", loss);

        // predictions=[1,2,3], targets=[4,5,6] → diffs=[3,3,3], sq=[9,9,9], mean=9
        let preds2 = NumericTensor::<DynRank>::from_vec_shape(
            vec![1.0f32, 2.0, 3.0], vec![1, 3],
        ).unwrap();
        let targets2 = NumericTensor::<DynRank>::from_vec_shape(
            vec![4.0f32, 5.0, 6.0], vec![1, 3],
        ).unwrap();
        let loss2 = eval_loss(&graph, &info, preds2, targets2);
        assert!((loss2 - 9.0).abs() < 1e-5, "expected 9, got {}", loss2);
    }

    #[test]
    fn test_l1_loss() {
        let rng = &mut rand::rng();
        let (graph, info) = MilliOpGraph::l1_loss(rng);

        // predictions=[1,2,3], targets=[1,2,3] → L1 = 0
        let preds = NumericTensor::<DynRank>::from_vec_shape(
            vec![1.0f32, 2.0, 3.0], vec![1, 3],
        ).unwrap();
        let targets = NumericTensor::<DynRank>::from_vec_shape(
            vec![1.0f32, 2.0, 3.0], vec![1, 3],
        ).unwrap();
        let loss = eval_loss(&graph, &info, preds, targets);
        assert!(loss.abs() < 1e-6, "expected 0, got {}", loss);

        // predictions=[1,2,3], targets=[4,6,9] → diffs=[3,4,6], mean=13/3≈4.333
        let preds2 = NumericTensor::<DynRank>::from_vec_shape(
            vec![1.0f32, 2.0, 3.0], vec![1, 3],
        ).unwrap();
        let targets2 = NumericTensor::<DynRank>::from_vec_shape(
            vec![4.0f32, 6.0, 9.0], vec![1, 3],
        ).unwrap();
        let loss2 = eval_loss(&graph, &info, preds2, targets2);
        let expected = 13.0f32 / 3.0;
        assert!((loss2 - expected).abs() < 1e-5, "expected {}, got {}", expected, loss2);
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
