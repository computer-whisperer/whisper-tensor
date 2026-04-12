# MilliOpGraph Subsystem

The MilliOpGraph is the tensor-level intermediate representation in
Whisper Tensor.  It sits between the ONNX-based SymbolicGraph and the
scalar NanoGraph, reducing 60+ ONNX operations to ~40 primitives with
explicit data flow via GlobalId references.

## Purpose

1. **Simplified operation set** — fewer ops than ONNX, each with
   precise dtype and broadcasting semantics.
2. **Explicit data flow** — every tensor dependency is a GlobalId
   edge, no implicit state.
3. **Serializable** — full serde support for saving/loading graphs.
4. **Observable** — execution hooks for tracing and debugging.
5. **Lowering target** — each op implements `lower_to_nano()` to
   decompose into scalar nano-ops, or falls back to `eval_new()` for
   ops that cannot be expressed analytically.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MilliOpGraph                             │
├─────────────────────────────────────────────────────────────────┤
│  input_map: HashMap<GlobalId, GlobalId>    (external → internal)│
│  input_ordering: Vec<GlobalId>             (ordered inputs)     │
│  output_map: HashMap<GlobalId, GlobalId>   (internal → external)│
│  output_ordering: Vec<GlobalId>            (ordered outputs)    │
├─────────────────────────────────────────────────────────────────┤
│  ops: HashMap<GlobalId, AnyMilliOp>        (all operations)     │
│  op_ordering: Vec<GlobalId>                (topological order)  │
│  tensors: HashMap<GlobalId, MilliOpGraphTensor>  (all tensors)  │
└─────────────────────────────────────────────────────────────────┘
```

- **Explicit topological ordering** — `op_ordering` stores execution
  order, no runtime sorting needed.
- **Input/output mapping** — external tensor IDs map to internal IDs,
  enabling graph composition and isolation.
- **Random GlobalIds** — all IDs are randomly generated u64, unique
  across the system.

---

## Execution Model

The primary execution path is:

```
MilliOpGraph
  → infer() (shape/dtype inference)
  → lower_to_nano() (decompose to scalar NanoGraph)
  → pool_eval or compiled eval (execute the NanoGraph)
```

Ops that cannot decompose into nano-ops (data-dependent output
shapes, sequential dependencies, external library calls) fall back to
`eval_new()`, which runs as an opaque node within the nano evaluation.

A legacy `eval()` path exists for backward compatibility but is not
the primary execution model.

---

## Core Traits

### MilliOp

The primary interface for all operations (`src/milli_graph/ops/mod.rs`):

```rust
pub trait MilliOp: Node<OpKind = String> {
    /// Shape/dtype inference from known inputs.
    fn infer<'a, 'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, TensorInfo<'a, 'p, P>>,
        symbolic_resolver: &mut SymbolicResolver,
        pool: &'p P,
    ) -> Result<Vec<(GlobalId, TensorInfo<'a, 'p, P>)>, MilliOpGraphError>;

    /// Generate backward ops for autodiff.
    fn backward(
        &self,
        output_grads: &HashMap<GlobalId, GlobalId>,
        graph: &mut MilliOpGraph,
        rng: &mut impl Rng,
    ) -> Option<HashMap<GlobalId, GlobalId>>;

    /// Last-resort evaluation for ops that cannot be expressed as
    /// nano-op lowerings.  Do NOT use as a performance shortcut.
    fn eval_new<'p, P: Pool + 'p>(
        &self,
        inputs: &[NumericTensorView<'_, DynRank>],
        pool: &'p P,
    ) -> Result<Vec<NumericTensor<'p, DynRank, P>>, PoolEvalError>;

    /// Decompose into scalar nano-ops.  Returns Lowered on success,
    /// Unsupported to fall through to eval_new.
    fn lower_to_nano<'p, P: Pool + 'p>(
        &self,
        ctx: &mut NanoLoweringContext<'_, 'p, P>,
    ) -> LowerResult;
}
```

### MilliOpGraphObserver

Execution observation hooks (`src/milli_graph/observer.rs`):

```rust
pub trait MilliOpGraphObserver {
    fn on_tensor_assigned(&mut self, tensor_path: &[GlobalId],
                          tensor: &NumericTensorView<'_, DynRank>);
    fn on_node_executed(&mut self, node_path: &[GlobalId],
                        start: Instant, end: Instant);
    fn should_cancel(&mut self) -> bool;
}
```

A no-op `impl MilliOpGraphObserver for ()` is provided.

---

## Operation Categories

### Constants

| Op | Description |
|----|-------------|
| `Constant` | Fixed tensor value embedded in the graph |
| `ConstantOfShape` | Scalar fill with runtime-determined shape |

### Binary

All binary ops support NumPy-style multidirectional broadcasting.

| Category | Operations |
|----------|------------|
| Arithmetic | Add, Sub, Mul, Div, Modulo |
| Logical | And, Or, Xor |
| Bitwise | BitwiseAnd, BitwiseOr, BitwiseXor |
| Comparison | Equal, Greater, GreaterOrEqual, Less, LessOrEqual |
| Element-wise | Max, Min |

Additional: `MatMul` (with automatic accumulation dtype), `Pow`.

### Unary

| Category | Operations |
|----------|------------|
| Arithmetic | Neg, Abs, Sign, Reciprocal |
| Exponential | Exp, Ln, Sqrt |
| Rounding | Floor, Ceil, Round |
| Logical | Not, BitwiseNot |
| Trigonometric | Sin, Cos, Tan, etc. |
| Special | IsNan, IsInf, Erf |

Additional: `ClampMin`.

### View / Layout

| Op | Description |
|----|-------------|
| Shape | Returns tensor shape as 1D tensor |
| Reshape | Reshape with -1 inference |
| Transpose | Permute dimensions |
| Squeeze | Remove size-1 dimensions |
| Unsqueeze | Insert size-1 dimensions |
| Expand | Broadcast to larger shape |

### Indexing

| Op | Description |
|----|-------------|
| Slice | Contiguous sub-tensor extraction |
| Gather | Index along an axis |
| Concat | Concatenate along an axis |
| Split | Split into multiple outputs |

### Reductions

All support axis specification, keepdims, and
noop\_with\_empty\_axes.  BF16/F16 inputs accumulate in F32.

| Op | Description |
|----|-------------|
| ReduceSum | Sum along axes |
| ReduceMean | Mean along axes |
| ReduceMax | Max along axes |
| ReduceMin | Min along axes |
| ReduceProd | Product along axes |

### Type

| Op | Description |
|----|-------------|
| Cast | Convert to specified dtype |
| CastLike | Convert to match another tensor's dtype |

### Conditional

| Op | Description |
|----|-------------|
| Where | Element-wise ternary select |

### Convolution

| Op | Description |
|----|-------------|
| Conv | N-D convolution (forward) with optional bias |
| ConvInputGrad | Backward: gradient w.r.t. input |
| ConvWeightGrad | Backward: gradient w.r.t. weight |
| ConvBiasGrad | Backward: gradient w.r.t. bias |

### Padding & Resize

| Op | Description |
|----|-------------|
| Pad | Constant, reflect, edge, or wrap mode |
| Resize | Spatial interpolation (nearest, linear, cubic) |

### Misc

| Op | Description |
|----|-------------|
| Range | Generate start-to-end sequence |
| CumSum | Cumulative sum along axis |
| NonZero | Indices of non-zero elements |
| ArgMax / ArgMin | Index of extremum along axis |
| TopK | Top-K values and indices |
| SumTo | Un-broadcast reduction to target shape |
| RandomNormalLike | Random normal matching input shape |
| GatherGrad | Backward for Gather (scatter-add) |

---

## Nano Lowering

Each op implements `lower_to_nano()` to decompose into the scalar
NanoGraph representation.  The lowering maps tensor dimensions into
two categories:

- **Known dimensions** — concrete sizes, expanded into separate atoms
  (each atom computes one scalar per known-dim position).
- **Symbolic dimensions** — runtime-unknown sizes (batch, seq\_len),
  carried as `sym_dims` on each atom group.  See
  `docs/symbolic_dims_nano.md` for the full specification.

Ops that return `LowerResult::Unsupported` run via `eval_new()` as
opaque nodes within the nano evaluation.  The design intent is that
all ops should eventually have full analytical lowerings — `eval_new`
is a last resort for ops whose semantics cannot be captured by the
ScalarOp vocabulary.

---

## Graph Construction

```rust
// Create graph with external inputs
let (mut graph, input_map) = MilliOpGraph::new(input_ids, &mut rng);
let a = input_map[&external_a];
let b = input_map[&external_b];

// Build computation
let sum = SimpleBinary::add(&mut graph, a, b, &mut rng);
let product = MatMul::push_new(&mut graph, sum, weights, &mut rng);
let mean = ReduceMean::push_new(&mut graph, product, Some(axes), true, false, &mut rng);

// Set outputs
graph.set_output_map([(mean, external_output)]);
```

---

## Broadcasting

NumPy-style multidirectional broadcasting via
`infer_multidirectional_broadcasting_shape`:

1. Output rank = max of input ranks
2. Per dimension (right-aligned): if both equal, use that; if one is
   1, use the other; if both differ and neither is 1, error.

Handles both concrete and symbolic dimensions via
`ScalarInfoTyped<u64>`.

---

## File Structure

```
src/milli_graph/
├── mod.rs              # MilliOpGraph struct, execution logic
├── observer.rs         # MilliOpGraphObserver trait
├── ops_helpers.rs      # Graph-building utility functions
├── validate_infer.rs   # Shape inference validation
└── ops/
    ├── mod.rs          # AnyMilliOp enum, MilliOp trait, broadcasting
    ├── binary.rs       # SimpleBinary, MatMul, Pow
    ├── unary.rs        # SimpleUnaryOp, ClampMin
    ├── cast.rs         # Cast
    ├── cast_like.rs    # CastLike
    ├── concat.rs       # Concat
    ├── constant.rs     # Constant, ConstantOfShape
    ├── conv.rs         # Conv, ConvInputGrad, ConvWeightGrad, ConvBiasGrad
    ├── expand.rs       # Expand
    ├── gather.rs       # Gather, GatherGrad
    ├── pad.rs          # Pad
    ├── reduce_*.rs     # ReduceSum, ReduceMean, ReduceMax, ReduceMin, ReduceProd
    ├── reshape.rs      # Reshape
    ├── shape.rs        # Shape
    ├── slice.rs        # Slice
    ├── split.rs        # Split
    ├── squeeze.rs      # Squeeze
    ├── transpose.rs    # Transpose
    ├── unsqueeze.rs    # Unsqueeze
    ├── where_op.rs     # Where
    └── ...             # Range, CumSum, NonZero, ArgMax, ArgMin, TopK, etc.
```

---

## Integration Points

### With SymbolicGraph

MilliOpGraph is generated from SymbolicGraph during compilation.
Complex ONNX operations are decomposed into simpler primitives.

### With SuperGraph

SuperGraph uses MilliOpGraph for model execution nodes, mixing
high-level orchestration (tokenization, caching) with tensor
computation.

### With NanoGraph

Each op's `lower_to_nano()` produces scalar atom groups with explicit
addressing (InputRef), dtype handling, and symbolic dimension
propagation.  The NanoGraph is the target for both interpreted
evaluation (pool\_eval) and compiled evaluation (x86\_jit).
