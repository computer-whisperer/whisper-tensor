# Training Support Design Document

## Overview

This document outlines the design for adding training (backpropagation) support to whisper-tensor. The core approach is to **extend MilliOpGraph** to serve as the unified training IR, with SymbolicGraph gaining a `generate_milli_graph()` method that can optionally include backward pass and optimizer operations.

---

## Design Philosophy

1. **MilliOpGraph as the training IR** - All training computation (forward, backward, optimizer) lives in a single MilliOpGraph
2. **SymbolicGraph stays clean** - High-level source representation, compiles down to MilliOpGraph; no loss baked in
3. **Loss as composable graph** - Loss computation is a separate MilliOpGraph, composed during training graph generation
4. **Editable after generation** - Users can surgically modify the MilliOpGraph for custom backward behavior
5. **Source linkage preserved** - Groups track which SymbolicGraph ops they came from
6. **Backend agnostic** - Pure symbolic differentiation, no dependency on framework autograd
7. **Minimal SuperGraph changes** - Training uses existing `SuperGraphNodeMilliOpGraph`; no new node types needed

---

## Architecture

### Compilation Flow

```
SymbolicGraph (ONNX-level, forward only)
    │
    │ generate_milli_graph(MilliGraphGenOptions)
    │   ├── loss_graph: MilliOpGraph (composed in)
    │   ├── backward options
    │   └── optimizer options
    ▼
MilliOpGraph (expanded, editable)
    ├── Forward groups (from symbolic ops)
    ├── Loss groups (from loss_graph)
    ├── Backward groups (auto-generated, editable)
    └── Optimizer groups (SGD, Adam, etc.)
    │
    │ [optional editing]
    ▼
MilliOpGraph (customized)
    │
    │ eval() via SuperGraphNodeMilliOpGraph
    ▼
Backend execution
```

### Integration with SuperGraph

**No changes needed to SuperGraph.** The existing `SuperGraphNodeMilliOpGraph` already:
- Embeds a MilliOpGraph directly
- Executes it with inputs from SuperGraphData
- Returns outputs to SuperGraphData

For training, we simply:
1. Generate a training-enabled MilliOpGraph (with backward + optimizer)
2. Wrap it in `SuperGraphNodeMilliOpGraph`
3. Execute normally

The `TrainingMetadata` on the MilliOpGraph tells the caller how to interpret inputs/outputs (which are params, which are optimizer state, etc.), but execution is identical to inference

---

## MilliOpGraph Extensions

### Node Grouping with Source Linkage

```rust
/// A group of related MilliOps with metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MilliOpGroup {
    pub id: GlobalId,
    /// Which SymbolicGraph op this came from (if any)
    pub source_op: Option<GlobalId>,
    /// Which SymbolicGraph this came from
    pub source_graph: Option<GlobalId>,
    /// Phase in the training pipeline
    pub phase: MilliOpPhase,
    /// Human-readable label for UI
    pub label: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MilliOpPhase {
    Forward,
    Backward,
    Optimizer,
    Custom,
}
```

### Extended MilliOpGraph Structure

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpGraph {
    // Existing fields
    global_id: GlobalId,
    pub input_map: HashMap<GlobalId, GlobalId>,
    pub input_ordering: Vec<GlobalId>,
    pub output_map: Option<HashMap<GlobalId, GlobalId>>,
    pub output_ordering: Option<Vec<GlobalId>>,
    ops: HashMap<GlobalId, AnyMilliOp>,
    op_ordering: Vec<GlobalId>,
    tensors: HashMap<GlobalId, MilliOpGraphTensor>,

    // New: grouping and source linkage
    groups: HashMap<GlobalId, MilliOpGroup>,
    op_to_group: HashMap<GlobalId, GlobalId>,

    // New: training metadata (optional)
    training_metadata: Option<TrainingMetadata>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingMetadata {
    /// Parameter tensor → Gradient tensor
    pub param_to_grad: HashMap<GlobalId, GlobalId>,
    /// Parameter tensor → New parameter tensor (post-optimizer)
    pub param_to_new_param: HashMap<GlobalId, GlobalId>,
    /// Optimizer state: (param, state_name) → tensor
    pub optimizer_state: HashMap<(GlobalId, String), GlobalId>,
    /// New optimizer state: (param, state_name) → new tensor
    pub new_optimizer_state: HashMap<(GlobalId, String), GlobalId>,
    /// Global optimizer state (timestep, etc.)
    pub global_state: HashMap<String, GlobalId>,
    pub new_global_state: HashMap<String, GlobalId>,
    /// Loss tensor
    pub loss: Option<GlobalId>,
}
```

### Extended Tensor Metadata

```rust
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MilliOpGraphTensor {
    global_id: GlobalId,
    /// Link back to source SymbolicGraph tensor (if any)
    pub source_tensor: Option<GlobalId>,
    /// Tensor role in training
    pub role: TensorRole,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorRole {
    /// Regular intermediate tensor
    Intermediate,
    /// A trainable parameter
    Parameter,
    /// Gradient of a parameter
    Gradient { of_param: GlobalId },
    /// Optimizer state (momentum, variance, etc.)
    OptimizerState { for_param: GlobalId, name: &'static str },
    /// Data input (changes each batch)
    DataInput,
    /// Loss value
    Loss,
}
```

---

## SymbolicGraph Generation Method

```rust
impl SymbolicGraph {
    /// Generate a MilliOpGraph from this SymbolicGraph
    ///
    /// With no options, generates forward-only graph (equivalent to current behavior).
    /// With backward options, includes gradient computation.
    /// With optimizer options, includes parameter updates.
    pub fn generate_milli_graph(
        &self,
        options: &MilliGraphGenOptions,
        rng: &mut impl Rng,
    ) -> Result<MilliOpGraph, MilliGraphGenError> {
        let mut graph = MilliOpGraph::new_with_groups(rng);
        let mut tensor_map = HashMap::new();  // SymbolicGraph tensor → MilliOpGraph tensor

        // 1. Generate forward pass
        for op in self.topological_order() {
            let group_id = graph.create_group(MilliOpGroup {
                source_op: Some(op.global_id()),
                source_graph: Some(self.global_id()),
                phase: MilliOpPhase::Forward,
                label: Some(op.op_kind()),
                ..Default::default()
            }, rng);

            let forward_milli = op.get_milli_op_graph(rng);
            graph.merge_graph(forward_milli, group_id, &mut tensor_map);
        }

        // 2. Generate backward pass (if requested)
        if let Some(ref backward_opts) = options.backward {
            // Initialize loss gradient to 1.0
            let loss_tensor = tensor_map[&backward_opts.loss];
            let loss_grad = graph.add_constant_ones_like(loss_tensor, rng);
            let mut grad_map = HashMap::new();
            grad_map.insert(loss_tensor, loss_grad);

            // Walk operations in reverse
            for op in self.topological_order().rev() {
                // Skip if no outputs have gradients
                let output_grads: Vec<_> = op.outputs()
                    .filter_map(|out_id| {
                        let milli_id = tensor_map.get(&out_id)?;
                        grad_map.get(milli_id).map(|g| (*milli_id, *g))
                    })
                    .collect();

                if output_grads.is_empty() {
                    continue;
                }

                // Skip stop-gradient boundaries
                if backward_opts.stop_gradients.contains(&op.global_id()) {
                    continue;
                }

                let ctx = BackwardGenContext {
                    output_grads: output_grads.into_iter().collect(),
                    forward_inputs: op.inputs().map(|id| tensor_map[&id]).collect(),
                    forward_outputs: op.outputs().map(|id| tensor_map[&id]).collect(),
                };

                if let Some(backward_milli) = op.get_backward_milli_ops(&ctx, rng) {
                    let group_id = graph.create_group(MilliOpGroup {
                        source_op: Some(op.global_id()),
                        source_graph: Some(self.global_id()),
                        phase: MilliOpPhase::Backward,
                        label: Some(format!("{}_backward", op.op_kind())),
                        ..Default::default()
                    }, rng);

                    // Merge backward ops and accumulate gradients
                    for (input_id, grad_id) in backward_milli.input_grads {
                        let orig_id = /* reverse lookup from tensor_map */;
                        grad_map.entry(input_id)
                            .and_modify(|existing| {
                                // Accumulate: existing += grad
                                let sum = graph.add_op_in_group(
                                    SimpleBinary::add(existing, grad_id),
                                    group_id
                                );
                                *existing = sum;
                            })
                            .or_insert(grad_id);
                    }
                }
            }

            // Record param → grad mapping
            let mut training_meta = TrainingMetadata::default();
            training_meta.loss = Some(loss_tensor);
            for param in &backward_opts.trainable_params {
                if let Some(&grad) = grad_map.get(&tensor_map[param]) {
                    training_meta.param_to_grad.insert(tensor_map[param], grad);
                }
            }

            // 3. Generate optimizer (if requested)
            if let Some(ref optim_opts) = options.optimizer {
                generate_optimizer_ops(
                    &mut graph,
                    &mut training_meta,
                    optim_opts,
                    &tensor_map,
                    rng,
                )?;
            }

            graph.training_metadata = Some(training_meta);
        }

        Ok(graph)
    }
}
```

### Generation Options

```rust
pub struct MilliGraphGenOptions {
    /// Generate backward pass for training
    pub backward: Option<BackwardGenOptions>,
    /// Generate optimizer updates (requires backward)
    pub optimizer: Option<OptimizerGenOptions>,
}

pub struct BackwardGenOptions {
    /// Loss computation graph to compose into the training graph
    pub loss_graph: MilliOpGraph,
    /// How to wire loss_graph inputs to forward outputs or external inputs
    pub loss_wiring: Vec<LossWiring>,
    /// Which output of loss_graph is the scalar loss to differentiate
    pub loss_output: GlobalId,
    /// Parameters to compute gradients for
    pub trainable_params: Vec<GlobalId>,
    /// Tensors where gradient flow stops (e.g., for freezing layers)
    pub stop_gradients: HashSet<GlobalId>,
}

/// Specifies how a loss graph input gets wired
pub struct LossWiring {
    /// Input tensor ID within the loss_graph
    pub loss_input: GlobalId,
    /// Where this input comes from
    pub source: LossInputSource,
}

pub enum LossInputSource {
    /// Wire to this output tensor of the forward SymbolicGraph
    ForwardOutput(GlobalId),
    /// Create as new external input to the training graph (e.g., labels)
    ExternalInput { name: String },
}

pub struct OptimizerGenOptions {
    pub kind: OptimizerKind,
    /// Per-parameter overrides
    pub param_configs: HashMap<GlobalId, ParamOptimizerConfig>,
}

pub enum OptimizerKind {
    SGD { lr: f32 },
    SGDMomentum { lr: f32, momentum: f32, nesterov: bool },
    Adam { lr: f32, beta1: f32, beta2: f32, epsilon: f32, weight_decay: f32 },
    AdamW { lr: f32, beta1: f32, beta2: f32, epsilon: f32, weight_decay: f32 },
    RMSprop { lr: f32, alpha: f32, epsilon: f32 },
}

pub struct ParamOptimizerConfig {
    /// Override optimizer for this param
    pub optimizer: Option<OptimizerKind>,
    /// Freeze this parameter (no updates)
    pub frozen: bool,
    /// Custom learning rate multiplier
    pub lr_scale: Option<f32>,
}
```

---

## Loss Graph Composition

The loss function is provided as a separate MilliOpGraph that gets composed into the training graph during generation. This keeps the forward SymbolicGraph clean and reusable for inference.

### Data Flow

```
Forward SymbolicGraph:
  inputs: [x]
  params: [W, b]
  outputs: [logits]

Loss MilliOpGraph:
  inputs: [predictions, targets]
  outputs: [loss_scalar]

loss_wiring:
  - predictions → ForwardOutput(logits)
  - targets → ExternalInput("targets")

Generated Training MilliOpGraph:
  inputs: [x, W, b, targets, m_W, v_W, m_b, v_b, t]
  outputs: [loss_scalar, W_new, b_new, m_W_new, v_W_new, m_b_new, v_b_new, t_new]

  groups:
    [Forward: expanded from SymbolicGraph ops]
    [Loss: from loss_graph, wired to forward outputs]
    [Backward: auto-generated from forward + loss]
    [Optimizer: from optimizer config]
```

### Loss Graph Helpers

Common loss functions provided as helpers:

```rust
impl MilliOpGraph {
    /// Cross-entropy loss for classification
    /// Input: logits [batch, classes], targets [batch] (class indices) or [batch, classes] (one-hot)
    /// Output: scalar loss
    pub fn cross_entropy_loss(rng: &mut impl Rng) -> (Self, LossGraphInfo) {
        let mut graph = MilliOpGraph::new_empty(rng);
        let logits = graph.add_input(rng);
        let targets = graph.add_input(rng);

        // log_softmax for numerical stability
        let log_probs = ops::LogSoftmax::push_new(&mut graph, logits, -1, rng);
        // negative log likelihood
        let nll = ops::SimpleBinary::mul(&mut graph, targets, log_probs, rng);
        let sum = ops::ReduceSum::push_new(&mut graph, nll, &[-1], false, rng);
        let neg = ops::SimpleUnary::neg(&mut graph, sum, rng);
        let loss = ops::ReduceMean::push_new(&mut graph, neg, &[], false, rng);

        graph.set_outputs(vec![loss]);

        (graph, LossGraphInfo {
            predictions_input: logits,
            targets_input: targets,
            loss_output: loss,
        })
    }

    /// Mean squared error loss for regression
    /// Input: predictions [batch, ...], targets [batch, ...] (same shape)
    /// Output: scalar loss
    pub fn mse_loss(rng: &mut impl Rng) -> (Self, LossGraphInfo) {
        let mut graph = MilliOpGraph::new_empty(rng);
        let predictions = graph.add_input(rng);
        let targets = graph.add_input(rng);

        let diff = ops::SimpleBinary::sub(&mut graph, predictions, targets, rng);
        let sq = ops::SimpleBinary::mul(&mut graph, diff, diff, rng);
        let loss = ops::ReduceMean::push_new(&mut graph, sq, &[], false, rng);

        graph.set_outputs(vec![loss]);

        (graph, LossGraphInfo {
            predictions_input: predictions,
            targets_input: targets,
            loss_output: loss,
        })
    }

    /// L1 / Mean Absolute Error loss
    pub fn l1_loss(rng: &mut impl Rng) -> (Self, LossGraphInfo);

    /// Binary cross-entropy with logits
    pub fn bce_with_logits_loss(rng: &mut impl Rng) -> (Self, LossGraphInfo);

    /// Huber loss (smooth L1)
    pub fn huber_loss(delta: f32, rng: &mut impl Rng) -> (Self, LossGraphInfo);
}

/// Metadata about a loss graph's inputs/outputs
pub struct LossGraphInfo {
    /// Input tensor for model predictions
    pub predictions_input: GlobalId,
    /// Input tensor for ground truth targets
    pub targets_input: GlobalId,
    /// Output tensor containing the scalar loss
    pub loss_output: GlobalId,
}
```

### Usage Example

```rust
// Load forward model
let forward_model = SymbolicGraph::from_onnx("classifier.onnx")?;
let logits_output = forward_model.get_output_by_name("logits")?;
let param_ids = forward_model.get_trainable_params();

// Create loss graph
let (loss_graph, loss_info) = MilliOpGraph::cross_entropy_loss(&mut rng);

// Generate training graph
let training_graph = forward_model.generate_milli_graph(&MilliGraphGenOptions {
    backward: Some(BackwardGenOptions {
        loss_graph,
        loss_wiring: vec![
            LossWiring {
                loss_input: loss_info.predictions_input,
                source: LossInputSource::ForwardOutput(logits_output),
            },
            LossWiring {
                loss_input: loss_info.targets_input,
                source: LossInputSource::ExternalInput { name: "labels".into() },
            },
        ],
        loss_output: loss_info.loss_output,
        trainable_params: param_ids,
        stop_gradients: HashSet::new(),
    }),
    optimizer: Some(OptimizerGenOptions {
        kind: OptimizerKind::Adam {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
        },
        param_configs: HashMap::new(),
    }),
}, &mut rng)?;

// Optional: edit the training graph for custom behavior
// training_graph.insert_after_tensor(some_grad_tensor, gradient_clip_op)?;

// Wrap in SuperGraph node for execution
let training_node = SuperGraphNodeMilliOpGraph::new(training_graph, &mut rng);
```
```

---

## Optimizer Graph Generation

### Adam Example

For a single parameter W with Adam optimizer:

```
Inputs needed:
  W         - parameter tensor
  grad_W    - gradient tensor
  m         - first moment state
  v         - second moment state
  t         - timestep (global)
  lr, β₁, β₂, ε - hyperparameters

Generated ops:
  [Group: shared_optimizer_constants]
    t_new = t + 1
    β₁^t = pow(β₁, t_new)
    β₂^t = pow(β₂, t_new)
    1 - β₁^t = sub(1, β₁^t)
    1 - β₂^t = sub(1, β₂^t)

  [Group: W_optimizer]
    # First moment
    m_scaled = mul(m, β₁)
    grad_scaled_m = mul(grad_W, 1 - β₁)
    m_new = add(m_scaled, grad_scaled_m)

    # Second moment
    grad_sq = mul(grad_W, grad_W)
    v_scaled = mul(v, β₂)
    grad_scaled_v = mul(grad_sq, 1 - β₂)
    v_new = add(v_scaled, grad_scaled_v)

    # Bias correction
    m_hat = div(m_new, 1 - β₁^t)
    v_hat = div(v_new, 1 - β₂^t)

    # Update
    v_hat_sqrt = sqrt(v_hat)
    denom = add(v_hat_sqrt, ε)
    update = div(m_hat, denom)
    scaled_update = mul(update, lr)
    W_new = sub(W, scaled_update)

Outputs:
  W_new, m_new, v_new, t_new
```

### Shared Computation

Multiple parameters share:
- Timestep increment (`t_new = t + 1`)
- Power computations (`β₁^t`, `β₂^t`)
- Bias correction denominators

These go in a single `shared_optimizer_constants` group, referenced by all parameter optimizer groups.

---

## Operation Trait Extension

```rust
pub trait Operation: Node {
    // Existing
    fn eval(&self, backend: &mut EvalBackend, inputs: &HashMap<GlobalId, NumericTensor<DynRank>>)
        -> OperationEvalRet;
    fn get_milli_op_graph(&self, rng: &mut impl Rng) -> MilliOpGraph;
    fn parameters(&self) -> Vec<Property>;

    // New: backward graph generation
    fn get_backward_milli_ops(
        &self,
        ctx: &BackwardGenContext,
        rng: &mut impl Rng,
    ) -> Option<BackwardGenResult> {
        None  // Default: not differentiable
    }

    fn is_differentiable(&self) -> bool {
        false
    }
}

pub struct BackwardGenContext {
    /// Output tensor → its gradient tensor (both in milli graph ID space)
    pub output_grads: HashMap<GlobalId, GlobalId>,
    /// Forward input tensor IDs (in milli graph ID space)
    pub forward_inputs: Vec<GlobalId>,
    /// Forward output tensor IDs (in milli graph ID space)
    pub forward_outputs: Vec<GlobalId>,
}

pub struct BackwardGenResult {
    /// The backward ops (will be merged into main graph)
    pub ops: Vec<AnyMilliOp>,
    /// Input tensor → gradient tensor mapping
    pub input_grads: HashMap<GlobalId, GlobalId>,
}
```

---

## MilliOpGraph Editing API

```rust
impl MilliOpGraph {
    // === Group management ===

    pub fn create_group(&mut self, group: MilliOpGroup, rng: &mut impl Rng) -> GlobalId;
    pub fn get_group(&self, group_id: GlobalId) -> Option<&MilliOpGroup>;
    pub fn get_group_mut(&mut self, group_id: GlobalId) -> Option<&mut MilliOpGroup>;
    pub fn ops_in_group(&self, group_id: GlobalId) -> impl Iterator<Item = GlobalId> + '_;
    pub fn groups_by_phase(&self, phase: MilliOpPhase) -> impl Iterator<Item = GlobalId> + '_;
    pub fn groups_by_source_op(&self, source_op: GlobalId) -> impl Iterator<Item = GlobalId> + '_;

    // === Surgical editing ===

    /// Replace an op with another (must have same inputs/outputs signature)
    pub fn replace_op(&mut self, old: GlobalId, new: AnyMilliOp) -> Result<(), EditError>;

    /// Insert an op after a tensor, rewiring downstream consumers
    pub fn insert_after_tensor(&mut self, tensor: GlobalId, op: AnyMilliOp) -> Result<GlobalId, EditError>;

    /// Delete a group and all its ops (must not have external dependents)
    pub fn delete_group(&mut self, group: GlobalId) -> Result<(), EditError>;

    /// Rewire a tensor reference (all uses of `from` become `to`)
    pub fn rewire_tensor(&mut self, from: GlobalId, to: GlobalId) -> Result<(), EditError>;

    /// Add an op to an existing group
    pub fn add_op_to_group(&mut self, op: AnyMilliOp, group: GlobalId) -> GlobalId;

    // === Merging ===

    /// Merge another MilliOpGraph into this one, assigning to a group
    pub fn merge_graph(
        &mut self,
        other: MilliOpGraph,
        group: GlobalId,
        tensor_map: &mut HashMap<GlobalId, GlobalId>,
    );

    // === Queries ===

    pub fn tensor_role(&self, tensor: GlobalId) -> Option<TensorRole>;
    pub fn gradient_of(&self, param: GlobalId) -> Option<GlobalId>;
    pub fn param_of_gradient(&self, grad: GlobalId) -> Option<GlobalId>;

    // === Validation ===

    pub fn validate(&self) -> Result<(), ValidationError>;
    pub fn validate_training(&self) -> Result<(), TrainingValidationError>;
}
```

---

## Training Loop Pattern

Since training is just MilliOpGraph execution, we use existing infrastructure:

```rust
// 1. Load forward model
let forward_model = SymbolicGraph::from_onnx("model.onnx")?;
let param_ids = forward_model.get_trainable_params();

// 2. Create loss graph
let (loss_graph, loss_info) = MilliOpGraph::cross_entropy_loss(&mut rng);

// 3. Generate training graph
let training_graph = forward_model.generate_milli_graph(&MilliGraphGenOptions {
    backward: Some(BackwardGenOptions {
        loss_graph,
        loss_wiring: vec![
            LossWiring {
                loss_input: loss_info.predictions_input,
                source: LossInputSource::ForwardOutput(forward_model.output("logits")),
            },
            LossWiring {
                loss_input: loss_info.targets_input,
                source: LossInputSource::ExternalInput { name: "labels".into() },
            },
        ],
        loss_output: loss_info.loss_output,
        trainable_params: param_ids.clone(),
        stop_gradients: HashSet::new(),
    }),
    optimizer: Some(OptimizerGenOptions {
        kind: OptimizerKind::Adam {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
        },
        param_configs: HashMap::new(),
    }),
}, &mut rng)?;

// 4. Optional: edit the training graph for custom behavior
// training_graph.insert_after_tensor(some_grad, gradient_clip_op)?;

// 5. Wrap in SuperGraph for execution (uses existing node type)
let mut builder = SuperGraphBuilder::new();
let training_node = SuperGraphNodeMilliOpGraph::new(training_graph.clone(), &mut rng);
builder.add_node(training_node.to_any());
// ... add other nodes (tokenizer, data preprocessing, etc.) ...
let super_graph = builder.build(&mut rng, &inputs, &outputs);

// 6. Initialize state
let meta = training_graph.training_metadata.as_ref().unwrap();
let mut params: HashMap<GlobalId, NumericTensor<DynRank>> = load_initial_params();
let mut optim_state: HashMap<GlobalId, NumericTensor<DynRank>> = meta.initialize_optimizer_state();

// 7. Training loop
for epoch in 0..num_epochs {
    for (batch_inputs, batch_labels) in data_loader {
        // Prepare inputs
        let mut data = SuperGraphData::new();

        // Data inputs
        data.tensors.insert(input_link, batch_inputs);
        data.tensors.insert(labels_link, batch_labels);

        // Current params and optimizer state
        for (param_id, tensor) in &params {
            data.tensors.insert(SuperGraphLinkTensor(*param_id), tensor.clone());
        }
        for (state_id, tensor) in &optim_state {
            data.tensors.insert(SuperGraphLinkTensor(*state_id), tensor.clone());
        }

        // Execute training step
        let outputs = super_graph.run(data, &mut context)?;

        // Extract updated params and state for next iteration
        for (old_param, new_param) in &meta.param_to_new_param {
            params.insert(*old_param, outputs.tensors[&SuperGraphLinkTensor(*new_param)].clone());
        }
        for ((param, name), new_state) in &meta.new_optimizer_state {
            optim_state.insert(*new_state, outputs.tensors[&SuperGraphLinkTensor(*new_state)].clone());
        }

        // Log progress
        let loss = outputs.tensors[&SuperGraphLinkTensor(meta.loss.unwrap())].to_scalar::<f32>()?;
        println!("Epoch {}, Loss: {:.4}", epoch, loss);
    }
}
```

---

## UI Considerations

### Graph Explorer Extensions

1. **Phase coloring**
   - Forward ops: blue
   - Backward ops: orange
   - Optimizer ops: green
   - Custom ops: purple

2. **Group display modes**
   - Collapsed: show group as single node
   - Expanded: show all ops in group
   - Source-linked: highlight corresponding SymbolicGraph op

3. **Training-specific views**
   - Gradient flow visualization
   - Parameter → Gradient → Update chain
   - Optimizer state inspection

4. **Editing mode**
   - Select ops/groups
   - Delete, replace, rewire
   - Insert custom ops
   - Validate after edits

---

## Implementation Phases

### Phase 1: Forward-Only Graph Generation

The first implementation phase focuses on `SymbolicGraph::generate_milli_graph()` for forward pass only. This validates the merge infrastructure and provides a foundation for backward generation.

**Core implementation:**

```rust
impl SymbolicGraph {
    pub fn generate_milli_graph(&self, rng: &mut impl Rng) -> MilliOpGraph {
        let mut combined = MilliOpGraph::new_empty(rng);
        // SymbolicGraph tensor ID → MilliOpGraph tensor ID
        let mut tensor_map: HashMap<GlobalId, GlobalId> = HashMap::new();

        // Map graph inputs
        for input_id in &self.ordered_inputs {
            let internal_id = combined.add_input_with_external_id(*input_id, rng);
            tensor_map.insert(*input_id, internal_id);
        }

        // Walk forward ops in topological order
        for op in self.topological_order() {
            let op_graph = op.get_milli_op_graph(rng);
            combined.merge_graph(op_graph, &mut tensor_map, rng);
            // merge_graph updates tensor_map with new output mappings
        }

        // Map graph outputs
        for output_id in &self.ordered_outputs {
            combined.add_output(tensor_map[output_id], *output_id);
        }

        combined
    }
}
```

**Required infrastructure:**

- [ ] `MilliOpGraph::new_empty(rng)` - create empty graph for building programmatically
- [ ] `MilliOpGraph::add_input_with_external_id(external_id, rng)` - add input with known external ID, returns internal ID
- [ ] `MilliOpGraph::merge_graph(other, tensor_map, rng)` - merge ops from another graph, remapping tensor IDs via tensor_map, updating tensor_map with new outputs
- [ ] `MilliOpGraph::add_output(internal_id, external_id)` - mark tensor as output
- [ ] `SymbolicGraph::topological_order()` - iterate ops in dependency order (check if exists)
- [ ] `SymbolicGraph::generate_milli_graph(rng)` - the main method

**Testing:**

- [ ] Generate MilliOpGraph from simple SymbolicGraph (single op)
- [ ] Generate from multi-op graph, verify tensor wiring
- [ ] **Equivalence test**: Generated MilliOpGraph should produce identical outputs to direct SymbolicGraph evaluation for same inputs

### Phase 2: Grouping and Source Linkage

- [ ] Add `MilliOpGroup` struct and group management to MilliOpGraph
- [ ] Add extended `MilliOpGraphTensor` with source linkage
- [ ] Update `generate_milli_graph()` to create groups per source op
- [ ] Add `TrainingMetadata` struct (empty for now)
- [ ] Unit tests for group operations

### Phase 3: Loss Graph Helpers

- [ ] Add `MilliOpGraph::add_input()` for building graphs without external IDs
- [ ] Implement `cross_entropy_loss()`
- [ ] Implement `mse_loss()`
- [ ] Implement `l1_loss()`
- [ ] Tests for loss graph construction

### Phase 4: Backward Generation Infrastructure

- [ ] Add `BroadcastAnalysis` struct and `analyze_broadcast()` function
- [ ] Add `get_backward_milli_ops()` to Operation trait with default impl
- [ ] Add generation options struct with backward/loss config
- [ ] Add loss graph composition logic to `generate_milli_graph()`
- [ ] Add backward pass generation (reverse walk, gradient accumulation)
- [ ] Tests for broadcast analysis (verify correct reduce axes computed)

### Phase 5: Backward Ops - Simple

- [ ] Implement backward for: Identity, Neg
- [ ] Implement backward for: Add, Sub, Mul, Div
- [ ] Implement backward for: Exp, Log, Sqrt, Reciprocal
- [ ] Finite difference tests for each

### Phase 6: Backward Ops - Core Neural Net

- [ ] Implement backward for: MatMul, Gemm
- [ ] Implement backward for: Relu, Sigmoid, Tanh
- [ ] Implement backward for: Softmax, LogSoftmax
- [ ] Implement backward for: ReduceSum, ReduceMean
- [ ] Finite difference tests for each

### Phase 7: Backward Ops - Shape Operations

- [ ] Implement backward for: Reshape, Flatten
- [ ] Implement backward for: Transpose
- [ ] Implement backward for: Squeeze, Unsqueeze
- [ ] Implement backward for: Concat, Split
- [ ] Implement backward for: Slice
- [ ] Implement backward for: Expand
- [ ] Finite difference tests for each

### Phase 8: Optimizers

- [ ] Implement SGD optimizer generation
- [ ] Implement SGD + Momentum
- [ ] Implement Adam/AdamW
- [ ] Shared constant optimization (β^t computations)
- [ ] Per-parameter config support
- [ ] `TrainingMetadata::initialize_optimizer_state()` helper

### Phase 9: End-to-End Testing

- [ ] Linear regression training example
- [ ] MLP on synthetic data
- [ ] Small model training test (verify convergence)

### Phase 10: Complex Ops (as needed)

- [ ] Normalization layers backward (LayerNorm, RMSNorm)
- [ ] Conv backward
- [ ] Gather backward (scatter-add)
- [ ] Pow, Where, Clip, Pad backward

### Phase 11: UI & Polish
- [ ] Graph explorer: phase coloring for training graphs
- [ ] Graph explorer: group collapse/expand
- [ ] Graph explorer: source linkage visualization
- [ ] Documentation and examples

---

## Detailed Data Structure Mockups

### MilliOpGraph Core Extensions

```rust
/// Extended MilliOpGraph with grouping and training support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpGraph {
    // === Existing fields ===
    global_id: GlobalId,
    ops: HashMap<GlobalId, AnyMilliOp>,
    op_ordering: Vec<GlobalId>,
    tensors: HashMap<GlobalId, MilliOpGraphTensor>,

    // Input/output mappings (external ID → internal ID)
    pub input_map: HashMap<GlobalId, GlobalId>,
    pub input_ordering: Vec<GlobalId>,
    pub output_map: Option<HashMap<GlobalId, GlobalId>>,
    pub output_ordering: Option<Vec<GlobalId>>,

    // === New: Grouping ===
    groups: HashMap<GlobalId, MilliOpGroup>,
    op_to_group: HashMap<GlobalId, GlobalId>,
    default_group: Option<GlobalId>,  // for ops added without explicit group

    // === New: Training metadata (present when generated with backward/optimizer) ===
    pub training_metadata: Option<TrainingMetadata>,
}

/// A group of related ops with provenance metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MilliOpGroup {
    pub id: GlobalId,

    // === Source provenance ===
    /// The SymbolicGraph this group came from (if any)
    pub source_graph: Option<GlobalId>,
    /// The SymbolicGraph Operation this group came from (if any)
    pub source_op: Option<GlobalId>,

    // === Classification ===
    pub phase: MilliOpPhase,

    // === Display ===
    pub label: Option<String>,

    // === Relationships ===
    /// For backward groups: the forward group this is the gradient of
    pub backward_of: Option<GlobalId>,
    /// For optimizer groups: the parameter this updates
    pub optimizes_param: Option<GlobalId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MilliOpPhase {
    Forward,
    Loss,
    Backward,
    Optimizer,
    Custom,
}

/// Extended tensor with source tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpGraphTensor {
    pub global_id: GlobalId,

    /// Link to source tensor in SymbolicGraph (if this came from expansion)
    pub source_tensor: Option<GlobalId>,

    /// Semantic role (for training graphs)
    pub role: Option<TensorRole>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TensorRole {
    /// Regular computation intermediate
    Intermediate,
    /// Model input (data)
    DataInput { name: String },
    /// Trainable parameter
    Parameter { name: Option<String> },
    /// Gradient of a parameter
    Gradient { of_param: GlobalId },
    /// Optimizer state (momentum, variance, etc.)
    OptimizerState {
        for_param: GlobalId,
        state_name: String,  // "m", "v", "velocity", etc.
    },
    /// The loss scalar
    Loss,
}
```

### Training Metadata

```rust
/// Metadata for training graphs - describes the semantic meaning of tensors
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TrainingMetadata {
    // === Loss ===
    /// The scalar loss tensor (output of loss computation)
    pub loss: Option<GlobalId>,

    // === Parameters ===
    /// Original parameter tensor → gradient tensor
    pub param_to_grad: HashMap<GlobalId, GlobalId>,
    /// Original parameter tensor → updated parameter tensor (post-optimizer)
    pub param_to_new_param: HashMap<GlobalId, GlobalId>,

    // === Optimizer State ===
    /// (param_id, state_name) → input state tensor
    /// e.g., (W_id, "m") → m_W_input_id
    pub optimizer_state_inputs: HashMap<(GlobalId, String), GlobalId>,
    /// (param_id, state_name) → output state tensor (after update)
    pub optimizer_state_outputs: HashMap<(GlobalId, String), GlobalId>,

    // === Global State ===
    /// Global optimizer state inputs (e.g., "timestep" → t_input_id)
    pub global_state_inputs: HashMap<String, GlobalId>,
    /// Global optimizer state outputs
    pub global_state_outputs: HashMap<String, GlobalId>,

    // === External Inputs ===
    /// Named external inputs (e.g., "labels" → labels_tensor_id)
    pub external_inputs: HashMap<String, GlobalId>,
}

impl TrainingMetadata {
    /// Create initial optimizer state tensors (zeros for m, v; 0 for timestep)
    pub fn initialize_optimizer_state<B: Backend>(
        &self,
        param_tensors: &HashMap<GlobalId, NumericTensor<DynRank>>,
        backend: &mut B,
    ) -> HashMap<GlobalId, NumericTensor<DynRank>> {
        let mut state = HashMap::new();

        // Per-parameter state (m, v for Adam)
        for ((param_id, state_name), state_tensor_id) in &self.optimizer_state_inputs {
            let param_shape = param_tensors[param_id].shape();
            let zeros = NumericTensor::zeros(param_shape, DType::F32, backend);
            state.insert(*state_tensor_id, zeros);
        }

        // Global state (timestep = 0)
        for (name, tensor_id) in &self.global_state_inputs {
            if name == "timestep" {
                state.insert(*tensor_id, NumericTensor::scalar(0i64));
            }
        }

        state
    }

    /// Extract updated parameters from execution outputs
    pub fn extract_new_params(
        &self,
        outputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
    ) -> HashMap<GlobalId, NumericTensor<DynRank>> {
        self.param_to_new_param
            .iter()
            .map(|(old, new)| (*old, outputs[new].clone()))
            .collect()
    }

    /// Extract updated optimizer state from execution outputs
    pub fn extract_new_state(
        &self,
        outputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
    ) -> HashMap<GlobalId, NumericTensor<DynRank>> {
        let mut state = HashMap::new();

        for ((param_id, state_name), old_id) in &self.optimizer_state_inputs {
            let new_id = &self.optimizer_state_outputs[&(*param_id, state_name.clone())];
            state.insert(*old_id, outputs[new_id].clone());
        }

        for (name, old_id) in &self.global_state_inputs {
            let new_id = &self.global_state_outputs[name];
            state.insert(*old_id, outputs[new_id].clone());
        }

        state
    }
}
```

### MilliOpGraph Construction & Manipulation

```rust
impl MilliOpGraph {
    // === Constructors ===

    /// Create empty graph for programmatic construction
    pub fn new_empty(rng: &mut impl Rng) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            ops: HashMap::new(),
            op_ordering: Vec::new(),
            tensors: HashMap::new(),
            input_map: HashMap::new(),
            input_ordering: Vec::new(),
            output_map: None,
            output_ordering: None,
            groups: HashMap::new(),
            op_to_group: HashMap::new(),
            default_group: None,
            training_metadata: None,
        }
    }

    /// Create from inputs (existing pattern, preserved)
    pub fn new(
        inputs: impl IntoIterator<Item = GlobalId>,
        rng: &mut impl Rng,
    ) -> (Self, HashMap<GlobalId, GlobalId>) {
        // ... existing implementation ...
    }

    // === Input/Output Management ===

    /// Add an input tensor, returns internal tensor ID
    pub fn add_input(&mut self, rng: &mut impl Rng) -> GlobalId {
        let external_id = GlobalId::new(rng);
        let internal_id = GlobalId::new(rng);
        self.input_map.insert(external_id, internal_id);
        self.input_ordering.push(external_id);
        self.tensors.insert(internal_id, MilliOpGraphTensor {
            global_id: internal_id,
            source_tensor: None,
            role: None,
        });
        internal_id
    }

    /// Add an input tensor with explicit external ID
    pub fn add_input_with_id(
        &mut self,
        external_id: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let internal_id = GlobalId::new(rng);
        self.input_map.insert(external_id, internal_id);
        self.input_ordering.push(external_id);
        self.tensors.insert(internal_id, MilliOpGraphTensor {
            global_id: internal_id,
            source_tensor: Some(external_id),
            role: None,
        });
        internal_id
    }

    /// Mark a tensor as an output
    pub fn add_output(&mut self, internal_id: GlobalId, external_id: GlobalId) {
        if self.output_map.is_none() {
            self.output_map = Some(HashMap::new());
            self.output_ordering = Some(Vec::new());
        }
        self.output_map.as_mut().unwrap().insert(internal_id, external_id);
        self.output_ordering.as_mut().unwrap().push(external_id);
    }

    // === Group Management ===

    /// Create a new group, returns group ID
    pub fn create_group(&mut self, group: MilliOpGroup) -> GlobalId {
        let id = group.id;
        self.groups.insert(id, group);
        id
    }

    /// Set the default group for new ops
    pub fn set_default_group(&mut self, group_id: Option<GlobalId>) {
        self.default_group = group_id;
    }

    /// Get group by ID
    pub fn get_group(&self, id: GlobalId) -> Option<&MilliOpGroup> {
        self.groups.get(&id)
    }

    /// Get mutable group by ID
    pub fn get_group_mut(&mut self, id: GlobalId) -> Option<&mut MilliOpGroup> {
        self.groups.get_mut(&id)
    }

    /// Iterate ops in a group
    pub fn ops_in_group(&self, group_id: GlobalId) -> impl Iterator<Item = GlobalId> + '_ {
        self.op_to_group
            .iter()
            .filter(move |(_, g)| **g == group_id)
            .map(|(op, _)| *op)
    }

    /// Find groups by phase
    pub fn groups_by_phase(&self, phase: MilliOpPhase) -> impl Iterator<Item = GlobalId> + '_ {
        self.groups
            .iter()
            .filter(move |(_, g)| g.phase == phase)
            .map(|(id, _)| *id)
    }

    // === Op Management ===

    /// Add an op to the graph (uses default group if set)
    pub fn push_op(&mut self, op: AnyMilliOp) -> GlobalId {
        let id = op.global_id();
        if let Some(group_id) = self.default_group {
            self.op_to_group.insert(id, group_id);
        }
        self.ops.insert(id, op);
        self.op_ordering.push(id);
        id
    }

    /// Add an op to a specific group
    pub fn push_op_in_group(&mut self, op: AnyMilliOp, group_id: GlobalId) -> GlobalId {
        let id = op.global_id();
        self.op_to_group.insert(id, group_id);
        self.ops.insert(id, op);
        self.op_ordering.push(id);
        id
    }

    // === Graph Composition ===

    /// Merge another graph into this one, remapping IDs
    /// Returns mapping from other's tensor IDs to this graph's tensor IDs
    pub fn merge_graph(
        &mut self,
        other: MilliOpGraph,
        group_id: GlobalId,
        input_wiring: &HashMap<GlobalId, GlobalId>,  // other's input → this graph's tensor
        rng: &mut impl Rng,
    ) -> MergeResult {
        let mut tensor_remap: HashMap<GlobalId, GlobalId> = HashMap::new();

        // Map other's inputs to existing tensors via wiring
        for (other_external, other_internal) in &other.input_map {
            if let Some(&this_tensor) = input_wiring.get(other_external) {
                tensor_remap.insert(*other_internal, this_tensor);
            } else {
                // Input not wired - create as new input to this graph
                let new_id = self.add_input_with_id(*other_external, rng);
                tensor_remap.insert(*other_internal, new_id);
            }
        }

        // Remap and add ops
        for op_id in &other.op_ordering {
            let op = other.ops[op_id].clone();
            let remapped_op = op.remap_tensors(&tensor_remap, rng);

            // Register new output tensors
            for output in remapped_op.outputs() {
                if !tensor_remap.contains_key(&output) {
                    let new_id = GlobalId::new(rng);
                    tensor_remap.insert(output, new_id);
                    self.tensors.insert(new_id, MilliOpGraphTensor {
                        global_id: new_id,
                        source_tensor: other.tensors.get(&output)
                            .and_then(|t| t.source_tensor),
                        role: other.tensors.get(&output)
                            .and_then(|t| t.role.clone()),
                    });
                }
            }

            self.push_op_in_group(remapped_op, group_id);
        }

        // Map outputs
        let output_map: HashMap<GlobalId, GlobalId> = other.output_map
            .map(|om| {
                om.into_iter()
                    .map(|(internal, external)| (external, tensor_remap[&internal]))
                    .collect()
            })
            .unwrap_or_default();

        MergeResult { tensor_remap, output_map }
    }
}

pub struct MergeResult {
    /// Mapping from merged graph's tensor IDs to this graph's tensor IDs
    pub tensor_remap: HashMap<GlobalId, GlobalId>,
    /// Mapping from merged graph's external output IDs to this graph's internal tensor IDs
    pub output_map: HashMap<GlobalId, GlobalId>,
}
```

### Editing API

```rust
impl MilliOpGraph {
    // === Surgical Editing ===

    /// Replace an op with another (must produce same outputs)
    pub fn replace_op(
        &mut self,
        old_id: GlobalId,
        new_op: AnyMilliOp,
    ) -> Result<(), EditError> {
        let old_op = self.ops.get(&old_id)
            .ok_or(EditError::OpNotFound(old_id))?;

        // Verify outputs match
        let old_outputs: Vec<_> = old_op.outputs().collect();
        let new_outputs: Vec<_> = new_op.outputs().collect();
        if old_outputs != new_outputs {
            return Err(EditError::OutputMismatch);
        }

        // Preserve group membership
        let group = self.op_to_group.get(&old_id).cloned();

        // Replace in ordering
        let pos = self.op_ordering.iter().position(|&id| id == old_id)
            .ok_or(EditError::OpNotFound(old_id))?;

        self.ops.remove(&old_id);
        self.op_to_group.remove(&old_id);

        let new_id = new_op.global_id();
        self.ops.insert(new_id, new_op);
        self.op_ordering[pos] = new_id;

        if let Some(g) = group {
            self.op_to_group.insert(new_id, g);
        }

        Ok(())
    }

    /// Insert an op that transforms a tensor, rewiring all downstream uses
    /// Returns the new output tensor ID
    pub fn insert_after_tensor(
        &mut self,
        tensor_id: GlobalId,
        op_fn: impl FnOnce(GlobalId, &mut impl Rng) -> (AnyMilliOp, GlobalId),
        rng: &mut impl Rng,
    ) -> Result<GlobalId, EditError> {
        let (new_op, new_output) = op_fn(tensor_id, rng);

        // Find where to insert (after the op that produces tensor_id)
        let producer_idx = self.op_ordering.iter().position(|&op_id| {
            self.ops[&op_id].outputs().any(|out| out == tensor_id)
        });

        let insert_pos = producer_idx.map(|i| i + 1).unwrap_or(0);

        // Rewire downstream ops to use new_output instead of tensor_id
        for op_id in &self.op_ordering[insert_pos..] {
            if let Some(op) = self.ops.get_mut(op_id) {
                *op = op.clone().remap_input(tensor_id, new_output);
            }
        }

        // Insert the new op
        let new_op_id = new_op.global_id();
        self.ops.insert(new_op_id, new_op);
        self.op_ordering.insert(insert_pos, new_op_id);

        // Add the new tensor
        self.tensors.insert(new_output, MilliOpGraphTensor {
            global_id: new_output,
            source_tensor: None,
            role: None,
        });

        Ok(new_output)
    }

    /// Delete a group and all its ops
    /// Fails if any op in the group has outputs used outside the group
    pub fn delete_group(&mut self, group_id: GlobalId) -> Result<(), EditError> {
        let ops_in_group: Vec<_> = self.ops_in_group(group_id).collect();

        // Collect all outputs produced by this group
        let group_outputs: HashSet<GlobalId> = ops_in_group.iter()
            .flat_map(|op_id| self.ops[op_id].outputs())
            .collect();

        // Check for external dependencies
        for op_id in &self.op_ordering {
            if ops_in_group.contains(op_id) {
                continue;
            }
            for input in self.ops[op_id].inputs() {
                if group_outputs.contains(&input) {
                    return Err(EditError::HasExternalDependents(group_id));
                }
            }
        }

        // Safe to delete
        for op_id in ops_in_group {
            self.ops.remove(&op_id);
            self.op_to_group.remove(&op_id);
            self.op_ordering.retain(|&id| id != op_id);
        }
        self.groups.remove(&group_id);

        Ok(())
    }

    /// Rewire all uses of one tensor to another
    pub fn rewire_tensor(
        &mut self,
        from: GlobalId,
        to: GlobalId,
    ) -> Result<(), EditError> {
        if !self.tensors.contains_key(&to) {
            return Err(EditError::TensorNotFound(to));
        }

        for op in self.ops.values_mut() {
            *op = op.clone().remap_input(from, to);
        }

        // Update output map if needed
        if let Some(ref mut output_map) = self.output_map {
            for (internal, _external) in output_map.iter_mut() {
                if *internal == from {
                    *internal = to;
                }
            }
        }

        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EditError {
    #[error("Op not found: {0:?}")]
    OpNotFound(GlobalId),
    #[error("Tensor not found: {0:?}")]
    TensorNotFound(GlobalId),
    #[error("Output tensors do not match")]
    OutputMismatch,
    #[error("Group {0:?} has ops with outputs used outside the group")]
    HasExternalDependents(GlobalId),
}
```

### AnyMilliOp Extensions for Editing

```rust
impl AnyMilliOp {
    /// Create a copy with tensor IDs remapped
    pub fn remap_tensors(
        &self,
        tensor_map: &HashMap<GlobalId, GlobalId>,
        rng: &mut impl Rng,
    ) -> Self {
        // Each variant needs to remap its input/output tensor IDs
        // This would be implemented via the delegate! macro pattern
        match self {
            AnyMilliOp::SimpleBinary(op) => {
                AnyMilliOp::SimpleBinary(op.remap_tensors(tensor_map, rng))
            }
            // ... other variants ...
        }
    }

    /// Create a copy with a single input remapped
    pub fn remap_input(&self, from: GlobalId, to: GlobalId) -> Self {
        let mut map = HashMap::new();
        map.insert(from, to);
        // For outputs, map to self (no change)
        for out in self.outputs() {
            map.insert(out, out);
        }
        self.remap_tensors(&map, &mut NoopRng)  // Don't need rng for identity output mapping
    }
}
```

### Backward Generation Structures

```rust
/// Context provided to Operation::get_backward_milli_ops()
pub struct BackwardGenContext {
    /// Maps forward output tensor IDs to their gradient tensor IDs
    /// (both IDs are in the combined training MilliOpGraph space)
    pub output_grads: HashMap<GlobalId, GlobalId>,

    /// Forward input tensor IDs (in combined graph space)
    /// Ordered to match Operation::inputs()
    pub forward_inputs: Vec<GlobalId>,

    /// Forward output tensor IDs (in combined graph space)
    /// Ordered to match Operation::outputs()
    pub forward_outputs: Vec<GlobalId>,
}

/// Result of backward op generation
pub struct BackwardGenResult {
    /// The MilliOpGraph containing backward computation
    pub graph: MilliOpGraph,

    /// Maps forward input tensor IDs to their gradient tensor IDs
    /// (forward input ID → gradient tensor ID within `graph`)
    pub input_grads: HashMap<GlobalId, GlobalId>,

    /// Any additional inputs the backward graph needs
    /// (e.g., forward input values for Mul backward)
    /// Maps: what the backward graph calls it → what to wire it to
    pub additional_inputs: Vec<BackwardInput>,
}

pub struct BackwardInput {
    /// Tensor ID within the backward graph that needs wiring
    pub graph_input: GlobalId,
    /// What this should be connected to
    pub source: BackwardInputSource,
}

pub enum BackwardInputSource {
    /// A forward tensor value (needed for ops like Mul, MatMul)
    ForwardTensor(GlobalId),
    /// An upstream gradient
    UpstreamGrad(GlobalId),
}
```

### Example: Mul Backward Implementation

```rust
impl BinaryOperation {
    fn get_backward_milli_ops(
        &self,
        ctx: &BackwardGenContext,
        rng: &mut impl Rng,
    ) -> Option<BackwardGenResult> {
        match self.which {
            WhichBinaryOperation::Mul => {
                // d/da(a * b) = dout * b
                // d/db(a * b) = dout * a

                let mut graph = MilliOpGraph::new_empty(rng);

                // Inputs we need:
                let dout = graph.add_input(rng);      // upstream gradient
                let a_fwd = graph.add_input(rng);     // forward value of a
                let b_fwd = graph.add_input(rng);     // forward value of b

                // Compute gradients
                let da = ops::SimpleBinary::mul(&mut graph, dout, b_fwd, rng);
                let db = ops::SimpleBinary::mul(&mut graph, dout, a_fwd, rng);

                // Set outputs
                graph.add_output(da, self.a);  // gradient for input a
                graph.add_output(db, self.b);  // gradient for input b

                Some(BackwardGenResult {
                    graph,
                    input_grads: [
                        (self.a, da),
                        (self.b, db),
                    ].into_iter().collect(),
                    additional_inputs: vec![
                        BackwardInput {
                            graph_input: dout,
                            source: BackwardInputSource::UpstreamGrad(self.output),
                        },
                        BackwardInput {
                            graph_input: a_fwd,
                            source: BackwardInputSource::ForwardTensor(ctx.forward_inputs[0]),
                        },
                        BackwardInput {
                            graph_input: b_fwd,
                            source: BackwardInputSource::ForwardTensor(ctx.forward_inputs[1]),
                        },
                    ],
                })
            }

            WhichBinaryOperation::Add => {
                // d/da(a + b) = dout
                // d/db(a + b) = dout
                // Simpler: both grads are just dout (no additional forward values needed)

                let mut graph = MilliOpGraph::new_empty(rng);
                let dout = graph.add_input(rng);

                // da = dout, db = dout (identity, but may need broadcast reduction)
                // For simplicity, assuming shapes match; real impl handles broadcasting

                graph.add_output(dout, self.a);
                graph.add_output(dout, self.b);

                Some(BackwardGenResult {
                    graph,
                    input_grads: [
                        (self.a, dout),
                        (self.b, dout),
                    ].into_iter().collect(),
                    additional_inputs: vec![
                        BackwardInput {
                            graph_input: dout,
                            source: BackwardInputSource::UpstreamGrad(self.output),
                        },
                    ],
                })
            }

            // Non-differentiable ops
            WhichBinaryOperation::Equal |
            WhichBinaryOperation::Greater |
            WhichBinaryOperation::Less |
            WhichBinaryOperation::And |
            WhichBinaryOperation::Or => None,

            // ... other ops ...
        }
    }
}
```

---

## Constraints & Assumptions

### Fixed Rank at Generation Time

The generated MilliOpGraph assumes the rank structure of the source SymbolicGraph.

- **Variable dimension sizes**: Work fine (batch size, sequence length flow dynamically via `Shape` ops)
- **Variable rank**: Would require generating different MilliOpGraphs

This is acceptable because:
1. Training graphs are generated for a specific model architecture
2. Model rank structure is fixed by the ONNX/model definition
3. The common variable dimensions (batch, sequence length) are size-variable, not rank-variable

### Dynamic Shape Handling

Both SymbolicGraph and MilliOpGraph are shape-agnostic at construction time:

- **SymbolicGraph**: `shape: Option<Vec<ScalarInfoTyped<u64>>>` allows symbolic dimensions
- **MilliOpGraph**: Ops take shapes as tensor inputs (`GlobalId`), resolved at eval time

For backward ops that need shape information (ReduceSum, Reshape, etc.), we use dynamic shape computation:

```rust
// Example: ReduceSum backward needs to expand gradient to input shape
let x_shape = Shape::push_new(&mut graph, x_forward, rng);  // capture shape dynamically
let dx = Expand::push_new(&mut graph, dy, x_shape, rng);    // use it
```

This means **no shape inference is required at generation time**.

---

## Broadcast-Aware Backward Generation

### Background: The Broadcasting Problem

When forward ops broadcast (e.g., adding a bias `[1, 10]` to activations `[batch, 10]`), the backward pass needs to reduce gradients back to the original shape.

- **Forward pass**: Binary ops handle broadcasting implicitly in the backend (no Expand ops - that would explode memory)
- **Backward challenge**: We need to know which axes were broadcast to generate the correct ReduceSum ops

### Solution: Compute Broadcast Axes at Generation Time

**Key insight**: At SymbolicGraph level, we have access to `TensorInfo` which tracks:
- Tensor rank (known for most models)
- Dimension sizes: can be `Numeric(u64)` or `Symbolic(SymbolicScalar)` for variable dims
- Critically: dimensions that are `1` are definitively broadcast candidates

**The approach**:

1. **Forward pass unchanged**: Keep implicit broadcasting - no Expand ops inserted
2. **At generation time**: Analyze `TensorInfo` shapes to determine which axes broadcast
3. **Generate backward ops**: Use computed broadcast info to insert correct ReduceSum axes

```rust
// Example: a[batch, 1, hidden] + b[1, seq, hidden] → c[batch, seq, hidden]
//
// Forward (unchanged - implicit broadcasting):
//   output = SimpleBinary::add(a, b)
//
// At generation time, we compute:
//   a_broadcast_axes = [1]   // a's dim 1 is 1, b's dim 1 is seq (not 1)
//   b_broadcast_axes = [0]   // b's dim 0 is 1, a's dim 0 is batch (not 1)
//
// Generated backward ops:
//   da = reduce_sum(dout, axes=[1], keepdims=true)
//   db = reduce_sum(dout, axes=[0], keepdims=true)
```

### Handling Variable Dimensions

Variable dimensions (batch size, sequence length) work fine:

1. **Rank is known**: Most ONNX models have fixed tensor ranks
2. **Broadcast dims are explicitly size-1**: A dimension that broadcasts is `Numeric(1)`, not symbolic
3. **Symbolic dims don't broadcast against each other**: Two symbolic dims are assumed equal

```rust
// Example: bias [hidden] + activations [batch, seq, hidden]
// TensorInfo for bias:     rank=1, shape=[Numeric(256)]
// TensorInfo for activ:    rank=3, shape=[Symbolic(?), Symbolic(?), Numeric(256)]
//
// At generation time, we determine:
//   - bias has lower rank → conceptually left-padded to [1, 1, hidden]
//   - axes 0, 1 are broadcast axes for bias (1 vs symbolic)
//
// Generated backward for bias:
//   d_bias = reduce_sum(dout, axes=[0, 1], keepdims=false)
```

### Implementation

```rust
/// Computed broadcast information for backward generation
#[derive(Clone, Debug)]
pub struct BroadcastAnalysis {
    /// Axes where input A broadcasts (A's dim is 1, B's is not)
    pub a_broadcast_axes: Vec<i64>,
    /// Axes where input B broadcasts
    pub b_broadcast_axes: Vec<i64>,
    /// Number of dims A was left-padded (for rank alignment)
    pub a_rank_padding: usize,
    /// Number of dims B was left-padded
    pub b_rank_padding: usize,
}

/// Analyze shapes to determine broadcast axes
fn analyze_broadcast(
    a_shape: &TensorInfo,
    b_shape: &TensorInfo,
) -> Option<BroadcastAnalysis> {
    // Require known rank for both
    let a_rank = a_shape.rank_if_known()?;
    let b_rank = b_shape.rank_if_known()?;
    let target_rank = a_rank.max(b_rank);

    let a_padding = target_rank - a_rank;
    let b_padding = target_rank - b_rank;

    let mut a_broadcast = vec![];
    let mut b_broadcast = vec![];

    for i in 0..target_rank {
        let a_dim = if i < a_padding {
            Some(1)  // Left-padded with 1
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
            (Some(1), None) => a_broadcast.push(i as i64),  // 1 vs symbolic → assume broadcast
            (Some(d), Some(1)) if d != 1 => b_broadcast.push(i as i64),
            (None, Some(1)) => b_broadcast.push(i as i64),
            _ => {}  // Same size or both symbolic (assume equal)
        }
    }

    Some(BroadcastAnalysis {
        a_broadcast_axes: a_broadcast,
        b_broadcast_axes: b_broadcast,
        a_rank_padding: a_padding,
        b_rank_padding: b_padding,
    })
}
```

### Backward Generation with Broadcast Analysis

```rust
impl BinaryOperation {
    fn get_backward_milli_ops(
        &self,
        ctx: &BackwardGenContext,
        rng: &mut impl Rng,
    ) -> Option<BackwardGenResult> {
        // Get shape info from SymbolicGraph context
        let a_shape = ctx.tensor_info(self.a);
        let b_shape = ctx.tensor_info(self.b);
        let broadcast = analyze_broadcast(&a_shape, &b_shape)?;

        match self.which {
            WhichBinaryOperation::Add => {
                // d/da(a + b) = dout (reduced if a was broadcast)
                // d/db(a + b) = dout (reduced if b was broadcast)
                let dout = ctx.output_grads[&self.output];

                let da = if broadcast.a_broadcast_axes.is_empty() {
                    dout
                } else {
                    ReduceSum::push_new(&mut graph, dout, &broadcast.a_broadcast_axes, true, rng)
                };

                let db = if broadcast.b_broadcast_axes.is_empty() {
                    dout
                } else {
                    ReduceSum::push_new(&mut graph, dout, &broadcast.b_broadcast_axes, true, rng)
                };

                // Handle rank padding: squeeze leading dims if needed
                let da = squeeze_leading(&mut graph, da, broadcast.a_rank_padding, rng);
                let db = squeeze_leading(&mut graph, db, broadcast.b_rank_padding, rng);

                // ... return result
            }

            WhichBinaryOperation::Mul => {
                // d/da(a * b) = dout * b (reduced if a was broadcast)
                // d/db(a * b) = dout * a (reduced if b was broadcast)
                let dout = ctx.output_grads[&self.output];
                let a_fwd = ctx.forward_inputs[0];
                let b_fwd = ctx.forward_inputs[1];

                // Compute unreduced gradients
                let da_full = SimpleBinary::mul(&mut graph, dout, b_fwd, rng);
                let db_full = SimpleBinary::mul(&mut graph, dout, a_fwd, rng);

                // Reduce along broadcast axes
                let da = reduce_and_squeeze(&mut graph, da_full, &broadcast.a_broadcast_axes,
                                            broadcast.a_rank_padding, rng);
                let db = reduce_and_squeeze(&mut graph, db_full, &broadcast.b_broadcast_axes,
                                            broadcast.b_rank_padding, rng);

                // ... return result
            }

            // Non-differentiable ops return None
            WhichBinaryOperation::Equal | WhichBinaryOperation::Greater => None,
        }
    }
}
```

### Edge Cases

1. **Unknown rank** (`TensorInfo::Minimal`): Cannot determine broadcast axes statically.
   - **Decision**: Require known rank at generation time. This is acceptable because training models have fixed architecture.

2. **Both dimensions symbolic and non-1**: Cannot determine which broadcasts.
   - **Solution**: Assume equal (no broadcasting). This is correct for well-formed models where symbolic dims represent the same runtime value.

3. **One dim is 1, other is symbolic**: Assume the symbolic dim is larger (broadcasting occurs).
   - This handles the common case of bias terms with explicit size-1 dims.

### Benefits

1. **No memory overhead**: Forward pass unchanged, no Expand ops
2. **Clean backward generation**: Broadcast analysis gives exact reduce axes
3. **Variable dims work**: Symbolic dims handled correctly (only concrete 1s broadcast)
4. **Deterministic**: Analysis is purely structural, no runtime decisions

## Open Questions & Future Considerations

### Gradient Checkpointing

For memory-efficient training of large models, we may want to:
1. Mark certain tensors as "checkpoint boundaries"
2. During backward, recompute forward values from checkpoints instead of storing all intermediates

This could be a post-generation transformation on the MilliOpGraph:
```rust
impl MilliOpGraph {
    /// Transform graph to use gradient checkpointing
    pub fn apply_checkpointing(
        &mut self,
        checkpoint_tensors: &[GlobalId],
    ) -> Result<(), CheckpointError>;
}
```

### Second-Order Gradients

For some advanced use cases (meta-learning, Hessian computation), we'd want gradients of gradients. The current design doesn't preclude this - you could:
1. Generate a training graph
2. Treat it as a "forward" graph
3. Generate backward pass of the backward pass

But the implementation complexity is high. Defer unless needed.

### Scan/Loop Backward

The `ScanOperation` contains a subgraph that executes multiple times. Its backward pass requires:
1. Running the forward subgraph to get intermediates (or recomputing from checkpoints)
2. Running backward through each iteration in reverse order
3. Accumulating gradients across iterations

This is complex but follows from the existing patterns. Mark as Phase 9 work.

---

## Testing Strategy

### Finite Difference Verification

```rust
fn test_backward_finite_diff<F>(
    forward_fn: F,
    inputs: &[NumericTensor<DynRank>],
    epsilon: f64,
    tolerance: f64,
) where F: Fn(&[NumericTensor<DynRank>]) -> NumericTensor<DynRank>
{
    // 1. Forward pass
    let output = forward_fn(inputs);

    // 2. Random upstream gradient
    let dout = random_tensor_like(&output);

    // 3. Analytic backward
    let analytic_grads = backward_fn(inputs, &dout);

    // 4. Numeric gradient via finite differences
    for (input_idx, input) in inputs.iter().enumerate() {
        for elem_idx in 0..input.numel() {
            let mut inputs_plus = inputs.to_vec();
            let mut inputs_minus = inputs.to_vec();

            inputs_plus[input_idx] = perturb(&input, elem_idx, epsilon);
            inputs_minus[input_idx] = perturb(&input, elem_idx, -epsilon);

            let out_plus = forward_fn(&inputs_plus);
            let out_minus = forward_fn(&inputs_minus);

            let numeric_grad = dot(&dout, &(out_plus - out_minus)) / (2.0 * epsilon);
            let analytic_grad = analytic_grads[input_idx].get(elem_idx);

            assert!(
                (numeric_grad - analytic_grad).abs() < tolerance,
                "Gradient mismatch at input {}, element {}: numeric={}, analytic={}",
                input_idx, elem_idx, numeric_grad, analytic_grad
            );
        }
    }
}
```

---

## Appendix: Operation Backward Reference

**Note on Broadcasting**: Binary ops use implicit broadcasting in the forward pass (no Expand ops). At generation time, we analyze `TensorInfo` shapes to determine which axes broadcast. Backward ops include `reduce_sum` along those axes to restore the original shape. See "Broadcast-Aware Backward Generation" section.

| Operation | Backward Formula | Complexity |
|-----------|-----------------|------------|
| `Identity` | `da = dout` | Simple |
| `Neg` | `da = -dout` | Simple |
| `Expand` | `da = reduce_sum(dout, expanded_axes)` | Simple |
| `Add` | `da = reduce_sum(dout, a_broadcast_axes)` | Simple |
| `Sub` | `da = reduce_sum(dout, a_broadcast_axes), db = reduce_sum(-dout, b_broadcast_axes)` | Simple |
| `Mul` | `da = reduce_sum(dout * b, a_broadcast_axes), db = reduce_sum(dout * a, b_broadcast_axes)` | Simple |
| `Div` | `da = dout / b, db = -dout * a / b²` | Medium |
| `MatMul` | `dA = dout @ Bᵀ, dB = Aᵀ @ dout` | Medium |
| `Exp` | `da = dout * exp(a)` | Simple |
| `Log` | `da = dout / a` | Simple |
| `Sqrt` | `da = dout / (2 * sqrt(a))` | Simple |
| `Relu` | `da = dout * (a > 0)` | Simple |
| `Sigmoid` | `da = dout * σ * (1 - σ)` | Simple |
| `Tanh` | `da = dout * (1 - tanh²)` | Simple |
| `Softmax` | `da = out * (dout - sum(dout * out))` | Hard |
| `ReduceSum` | `da = broadcast(dout)` | Medium |
| `ReduceMean` | `da = broadcast(dout / n)` | Medium |
| `Reshape` | `da = reshape(dout, input_shape)` | Simple |
| `Transpose` | `da = transpose(dout, inverse_perm)` | Simple |
| `Concat` | `da_i = slice(dout, ...)` | Medium |
| `Slice` | `da = pad(dout, ...)` | Medium |

---

*Document created: 2026-01-31*
*Last updated: 2026-02-03*
*Status: Design phase - ready for Phase 1 implementation*
