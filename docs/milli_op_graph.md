# MilliOpGraph Subsystem

The MilliOpGraph is a core intermediate representation (IR) layer in Whisper Tensor that provides a simplified, executable computation graph. It sits between the high-level ONNX-based SymbolicGraph and the low-level backend execution, serving as a common target for graph optimization and execution.

## Purpose and Design Goals

The MilliOpGraph serves several key purposes:

1. **Simplified Operation Set**: Reduces the 60+ ONNX operations to ~30 primitive operations, making backend implementation more tractable
2. **Explicit Data Flow**: All tensor dependencies are explicitly tracked via GlobalId references
3. **Serializable**: Full serde support enables saving/loading computation graphs
4. **Observable**: Built-in observer pattern for execution tracing and debugging
5. **Backend-Agnostic**: Operations delegate to the `EvalBackend` abstraction

---

## Architecture Overview

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

### Key Design Decisions

- **Explicit Topological Ordering**: Operations are stored in `op_ordering` in execution order, eliminating the need for runtime topological sorting
- **Input/Output Mapping**: External tensor IDs are mapped to internal IDs, enabling graph composition and isolation
- **Random GlobalIds**: All IDs are randomly generated 64-bit integers, ensuring uniqueness across the system

---

## Core Data Structures

### MilliOpGraph

The main graph container, located in `src/milli_graph/mod.rs`:

```rust
pub struct MilliOpGraph {
    global_id: GlobalId,                              // Unique graph identifier
    pub input_map: HashMap<GlobalId, GlobalId>,       // External → internal tensor IDs
    pub input_ordering: Vec<GlobalId>,                // Ordered list of graph inputs
    pub output_map: Option<HashMap<GlobalId, GlobalId>>, // Internal → external IDs
    pub output_ordering: Option<Vec<GlobalId>>,       // Ordered list of graph outputs
    ops: HashMap<GlobalId, AnyMilliOp>,               // All operations by ID
    op_ordering: Vec<GlobalId>,                       // Execution order
    tensors: HashMap<GlobalId, MilliOpGraphTensor>,   // All tensor nodes
}
```

### MilliOpGraphTensor

A minimal tensor node representation:

```rust
pub struct MilliOpGraphTensor {
    global_id: GlobalId,  // Unique tensor identifier
}
```

Tensors in MilliOpGraph are lightweight link nodes - the actual tensor data flows through during execution, not stored in the graph structure itself.

### AnyMilliOp

An enum that wraps all supported operation types:

```rust
pub enum AnyMilliOp {
    // Constants
    Constant(Constant),
    ConstantOfShape(ConstantOfShape),
    
    // Binary Operations
    SimpleBinary(SimpleBinary),   // Add, Sub, Mul, Div, comparisons, etc.
    MatMul(MatMul),
    Pow(Pow),
    
    // Unary Operations
    SimpleUnary(SimpleUnaryOp),   // Neg, Abs, Exp, Sqrt, trig, etc.
    ClampMin(ClampMin),
    
    // Shape Operations
    Shape(Shape),
    Reshape(Reshape),
    Transpose(Transpose),
    Squeeze(Squeeze),
    Unsqueeze(Unsqueeze),
    Expand(Expand),
    
    // Indexing Operations
    Slice(Slice),
    Gather(Gather),
    
    // Reduction Operations
    ReduceSum(ReduceSum),
    ReduceMean(ReduceMean),
    ReduceMax(ReduceMax),
    ReduceMin(ReduceMin),
    ReduceProd(ReduceProd),
    
    // Type Operations
    Cast(Cast),
    CastLike(CastLike),
    
    // Multi-tensor Operations
    Concat(Concat),
    Split(Split),
    Where(Where),
    
    // Sequence Operations
    Range(Range),
    CumSum(CumSum),
    NonZero(NonZero),
    ArgMax(ArgMax),
    ArgMin(ArgMin),
}
```

---

## Operation Categories

### Constants

| Operation | Description |
|-----------|-------------|
| `Constant` | Embeds a fixed tensor value directly in the graph |
| `ConstantOfShape` | Creates a tensor filled with a scalar value, shape determined at runtime |

### Binary Operations (SimpleBinary)

All binary operations support broadcasting. The `WhichSimpleBinaryOp` enum specifies the operation:

| Category | Operations |
|----------|------------|
| Arithmetic | `Add`, `Sub`, `Mul`, `Div`, `Modulo` |
| Logical | `And`, `Or`, `Xor` |
| Bitwise | `BitwiseAnd`, `BitwiseOr`, `BitwiseXor` |
| Comparison | `Equal`, `Greater`, `GreaterOrEqual`, `Less`, `LessOrEqual` |
| Element-wise | `Max`, `Min` |

Additional binary ops:
- `MatMul` - Matrix multiplication with automatic accumulation dtype handling
- `Pow` - Element-wise power operation

### Unary Operations (SimpleUnaryOp)

The `WhichSimpleUnaryOp` enum includes:

| Category | Operations |
|----------|------------|
| Arithmetic | `Neg`, `Abs`, `Sign`, `Reciprocal` |
| Exponential | `Exp`, `Ln`, `Sqrt` |
| Rounding | `Floor`, `Ceil`, `Round` |
| Logical | `Not`, `BitwiseNot` |
| Trigonometric | `Trig(TrigOp)` - sin, cos, tan, etc. |
| Special | `IsNan`, `IsInf`, `Erf` |

Additional unary op:
- `ClampMin` - Clamps values to a minimum threshold

### Shape Operations

| Operation | Description |
|-----------|-------------|
| `Shape` | Returns tensor shape as a 1D tensor |
| `Reshape` | Reshapes tensor, supports -1 dimension inference |
| `Transpose` | Permutes tensor dimensions |
| `Squeeze` | Removes dimensions of size 1 |
| `Unsqueeze` | Inserts dimensions of size 1 |
| `Expand` | Broadcasts tensor to larger shape |

### Indexing Operations

| Operation | Description |
|-----------|-------------|
| `Slice` | Extracts a contiguous sub-tensor |
| `Gather` | Gathers elements along an axis using indices |

### Reduction Operations

All reductions support:
- Axis specification (optional, defaults to all axes)
- `keepdims` flag to preserve reduced dimensions
- `noop_with_empty_axes` flag for identity behavior

| Operation | Description |
|-----------|-------------|
| `ReduceSum` | Sum elements along axes |
| `ReduceMean` | Average elements along axes |
| `ReduceMax` | Maximum element along axes |
| `ReduceMin` | Minimum element along axes |
| `ReduceProd` | Product of elements along axes |

**Precision Handling**: BF16/F16 inputs are automatically cast to F32 for accumulation, then cast back to the original dtype for output.

### Type Operations

| Operation | Description |
|-----------|-------------|
| `Cast` | Convert tensor to specified dtype |
| `CastLike` | Convert tensor to match another tensor's dtype |

### Multi-tensor Operations

| Operation | Description |
|-----------|-------------|
| `Concat` | Concatenate tensors along an axis |
| `Split` | Split tensor into multiple outputs |
| `Where` | Conditional selection between two tensors |

### Sequence Operations

| Operation | Description |
|-----------|-------------|
| `Range` | Generate sequence from start to end with step |
| `CumSum` | Cumulative sum along an axis |
| `NonZero` | Returns indices of non-zero elements |
| `ArgMax` | Index of maximum value along axis |
| `ArgMin` | Index of minimum value along axis |

---

## Core Traits

### MilliOp Trait

The primary interface for all operations (`src/milli_graph/ops/mod.rs`):

```rust
pub trait MilliOp: Node {
    /// Type inference from known inputs (optional override)
    fn infer(
        &self,
        known_inputs: &HashMap<GlobalId, TensorInfo>,
        symbolic_resolver: &mut SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, TensorInfo)>>, MilliOpGraphError>;
    
    /// Execute the operation
    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> EvalResult;
}
```

The default `infer` implementation attempts to run `eval` with concrete values to determine output types. Operations can override this for symbolic inference.

### Node Trait

Graph connectivity interface (from `src/graph.rs`):

```rust
pub trait Node {
    type OpKind: AsRef<str> + Clone + Debug;
    
    fn global_id(&self) -> GlobalId;
    fn op_kind(&self) -> Self::OpKind;
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_>;
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_>;
}
```

### Link Trait

Tensor node interface (from `src/graph.rs`):

```rust
pub trait Link {
    fn global_id(&self) -> GlobalId;
    fn label(&self) -> Option<String> { None }
}
```

### MilliOpGraphObserver Trait

Execution observation hooks (`src/milli_graph/observer.rs`):

```rust
pub trait MilliOpGraphObserver {
    fn on_tensor_assigned(
        &mut self,
        tensor_path: &[GlobalId],
        tensor: &NumericTensor<DynRank>,
        backend: &mut EvalBackend,
    );
    
    fn on_node_executed(
        &mut self,
        node_path: &[GlobalId],
        start_instant: Instant,
        end_instant: Instant,
        backend: &mut EvalBackend,
    );
}
```

A no-op implementation `impl MilliOpGraphObserver for ()` is provided for when observation is not needed.

---

## Graph Construction

### Creating a New Graph

```rust
let input_ids = vec![external_input_1, external_input_2];
let (mut graph, input_map) = MilliOpGraph::new(input_ids, &mut rng);

// input_map contains: external_id → internal_id mappings
let internal_a = input_map[&external_input_1];
let internal_b = input_map[&external_input_2];
```

### Adding Operations

Operations provide builder methods that:
1. Allocate a new output tensor ID
2. Create the operation node
3. Add it to the graph's operation ordering
4. Return the output tensor ID

Example pattern:

```rust
// Add two tensors
let sum = SimpleBinary::add(&mut graph, tensor_a, tensor_b, &mut rng);

// Matrix multiplication
let product = MatMul::push_new(&mut graph, matrix_a, matrix_b, &mut rng);

// Reshape with inferred dimension
let reshaped = Reshape::push_new(&mut graph, data, shape_tensor, false, &mut rng);

// Reduction
let mean = ReduceMean::push_new(&mut graph, data, Some(axes), true, false, &mut rng);
```

### Setting Output Mapping

```rust
// Simple output mapping
graph.set_output_map([
    (internal_output_1, external_output_1),
    (internal_output_2, external_output_2),
]);

// With explicit ordering
graph.set_output_map_ordered(output_map, output_ordering);
```

---

## Graph Execution

### Execution Flow

The `eval` method executes the graph:

```rust
pub fn eval<T: MilliOpGraphObserver>(
    &self,
    inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
    observer: &mut T,
    backend: &mut EvalBackend,
) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
```

**Execution Steps:**

1. **Input Mapping**: Map external input tensors to internal IDs via `input_map`

2. **Sequential Execution**: Iterate through `op_ordering` in topological order:
   - Resolve input tensors from `intermediate_values`
   - Call the operation's `eval` method
   - Record timing information
   - Notify observer via `on_node_executed`
   - Store output tensors in `intermediate_values`
   - Notify observer via `on_tensor_assigned`

3. **Output Mapping**: Map internal output tensors back to external IDs via `output_map`

### Example Usage

```rust
let mut inputs = HashMap::new();
inputs.insert(external_input_id, input_tensor);

let results = graph.eval(&inputs, &mut (), &mut backend)?;

for (output_id, tensor) in results {
    println!("Output {}: shape {:?}", output_id, tensor.shape());
}
```

---

## Helper Utilities

The `ops_helpers.rs` module provides common graph-building patterns:

### rank

Computes tensor rank (number of dimensions):

```rust
pub fn rank(graph: &mut MilliOpGraph, tensor: GlobalId, rng: &mut impl Rng) -> GlobalId
```

Implementation: `Shape(Shape(tensor))` - shape of shape gives rank.

### scalar_const

Creates a scalar constant tensor:

```rust
pub fn scalar_const<T: NDArrayNumericTensorType>(
    graph: &mut MilliOpGraph,
    value: T,
    rng: &mut impl Rng,
) -> GlobalId
```

### resolve_axes

Normalizes negative axes to positive indices:

```rust
pub fn resolve_axes(
    graph: &mut MilliOpGraph,
    axes: GlobalId,
    tensor: GlobalId,
    rng: &mut impl Rng,
) -> GlobalId
```

Implementation: `(axes + rank) % rank` - handles negative indexing like Python/NumPy.

---

## Broadcasting

MilliOpGraph follows NumPy-style broadcasting rules. The `infer_multidirectional_broadcasting_shape` function in `ops/mod.rs` handles symbolic shape inference:

1. Determine output rank as max of input ranks
2. For each dimension (right-aligned):
   - If both dimensions are 1, output is 1
   - If one dimension is 1, use the other
   - If both are equal, use that value
   - If both are different and neither is 1, error

The function handles both concrete and symbolic dimensions via `ScalarInfoTyped<u64>`.

---

## Error Handling

```rust
pub enum MilliOpGraphError {
    NumericTensorError(NumericTensorError),
    NDArrayNumericTensorError(NDArrayNumericTensorError),
    UnimplementedOperatorError(String),
    InvalidInput(String),
    DTypeError(DTypeError),
    TensorInfoError(TensorInfoError),
    UnableToInfer,
}
```

All errors use `thiserror` for transparent error conversion.

---

## Serialization

The entire graph structure is serializable via serde:

```rust
#[derive(Serialize, Deserialize)]
pub struct MilliOpGraph { ... }

#[derive(Serialize, Deserialize)]  
pub enum AnyMilliOp { ... }
```

This enables:
- Saving/loading computation graphs to disk
- Transmitting graphs over network
- Caching compiled graph structures

---

## Integration Points

### With SymbolicGraph

MilliOpGraph is typically generated from SymbolicGraph during compilation. Complex ONNX operations are decomposed into simpler MilliOp primitives.

### With SuperGraph

The SuperGraph layer uses MilliOpGraph for direct execution nodes via `SuperGraphNodeModelExecution`. This enables mixing high-level operations (tokenization, caching) with low-level tensor computation.

### With Backends

All operations delegate to `EvalBackend` for actual computation. The backend abstraction allows:
- NDArray (CPU reference implementation)
- Vulkan (GPU compute)
- Future backends (Candle, TCH)

### With NumericTensor

Operations work with `NumericTensor<DynRank>`:
- Dynamic rank tensors (rank determined at runtime)
- Backend-agnostic tensor wrapper
- Automatic dtype conversion and broadcasting

---

## File Structure

```
src/milli_graph/
├── mod.rs              # MilliOpGraph struct, execution logic
├── observer.rs         # MilliOpGraphObserver trait
├── ops_helpers.rs      # Graph-building utility functions
└── ops/
    ├── mod.rs          # AnyMilliOp enum, MilliOp trait, broadcasting
    ├── argmax.rs       # ArgMax operation
    ├── argmin.rs       # ArgMin operation
    ├── binary.rs       # SimpleBinary, MatMul, Pow
    ├── cast.rs         # Cast operation
    ├── cast_like.rs    # CastLike operation
    ├── concat.rs       # Concat operation
    ├── constant.rs     # Constant, ConstantOfShape
    ├── cumsum.rs       # CumSum operation
    ├── expand.rs       # Expand operation
    ├── gather.rs       # Gather operation
    ├── nonzero.rs      # NonZero operation
    ├── range.rs        # Range operation
    ├── reduce_max.rs   # ReduceMax operation
    ├── reduce_mean.rs  # ReduceMean operation
    ├── reduce_min.rs   # ReduceMin operation
    ├── reduce_prod.rs  # ReduceProd operation
    ├── reduce_sum.rs   # ReduceSum operation
    ├── reshape.rs      # Reshape operation
    ├── shape.rs        # Shape operation
    ├── slice.rs        # Slice operation
    ├── split.rs        # Split operation
    ├── squeeze.rs      # Squeeze operation
    ├── transpose.rs    # Transpose operation
    ├── unary.rs        # SimpleUnaryOp, ClampMin
    ├── unsqueeze.rs    # Unsqueeze operation
    └── where_op.rs     # Where operation
```

---

## Implementation Patterns

### Standard Operation Structure

Each operation follows a consistent pattern:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SomeOp {
    global_id: GlobalId,      // Unique operation ID
    output: GlobalId,         // Output tensor ID
    input: GlobalId,          // Input tensor ID(s)
    // Operation-specific parameters...
}

impl SomeOp {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        input: GlobalId,
        // Operation-specific parameters...
        rng: &mut impl Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            output,
            input,
            // ...
        };
        graph.push_op(AnyMilliOp::SomeOp(node));
        output
    }
}

impl Node for SomeOp {
    type OpKind = String;
    fn global_id(&self) -> GlobalId { self.global_id }
    fn op_kind(&self) -> String { "SomeOp".to_string() }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new([self.input].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new([self.output].into_iter())
    }
}

impl MilliOp for SomeOp {
    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> EvalResult {
        let input = &inputs[&self.input];
        let output = input.some_operation(backend)?;
        Ok(Box::new([(self.output, output)].into_iter()))
    }
}
```

### Macro-Based Delegation

The `AnyMilliOp` enum uses a `delegate!` macro to forward trait methods to the inner operation type, avoiding boilerplate for each variant:

```rust
macro_rules! delegate {
    ($name:ident($($arg:ident: $ty:ty),*) -> $ret:ty) => {
        fn $name(&self, $($arg: $ty),*) -> $ret {
            match self {
                AnyMilliOp::Constant(x) => x.$name($($arg),*),
                AnyMilliOp::SimpleBinary(x) => x.$name($($arg),*),
                // ... all variants
            }
        }
    }
}
```
