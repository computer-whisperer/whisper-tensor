# SymbolicGraph Subsystem

The SymbolicGraph is the high-level intermediate representation (IR) layer in Whisper Tensor that provides a direct mapping to ONNX computation graphs. It sits above the low-level MilliOpGraph layer, serving as the entry point for model import and the primary interface for graph manipulation.

## Purpose and Design Goals

The SymbolicGraph serves several key purposes:

1. **ONNX Compatibility**: Direct 1:1 mapping with ONNX operations and semantics, enabling straightforward model import
2. **Symbolic Shape Handling**: Supports unknown dimensions (e.g., batch size) via symbolic scalars
3. **Lazy Tensor Loading**: Efficient memory management through deferred loading of large constants
4. **Sub-Graph Support**: First-class support for control flow operations (If, Scan) with nested graphs
5. **Type-Safe Validation**: Runtime validation of tensor shapes and dtypes against declared specifications

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        SymbolicGraph                            │
├─────────────────────────────────────────────────────────────────┤
│  unknown_dimensions: HashMap<String, SymbolicScalar>            │
│  tensors: HashMap<GlobalId, ONNXTensorInfo>                     │
│  ordered_inputs: Vec<GlobalId>                                  │
│  ordered_outputs: Vec<GlobalId>                                 │
├─────────────────────────────────────────────────────────────────┤
│  operations: HashMap<GlobalId, GraphOperation>                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     TensorStore (external)                      │
├─────────────────────────────────────────────────────────────────┤
│  tensors: HashMap<TensorStoreTensorId, StoredTensor>            │
│    - Numeric (in-memory)                                        │
│    - ExternalBinary (file offset/length)                        │
│    - ExternalPth (PyTorch .pth files)                           │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

- **ONNX-First Design**: Operations retain ONNX semantics and naming, simplifying import and debugging
- **Symbolic Dimensions**: Unknown dimensions use `SymbolicScalar` values that can be resolved at runtime
- **Lazy Loading**: Large tensors (>100 elements) can be stored externally and loaded on-demand
- **Two-Layer Execution**: Operations implement `get_milli_op_graph()` to bridge to the lower-level execution layer

---

## Core Data Structures

### SymbolicGraph

The main graph container, located in `src/symbolic_graph/mod.rs`:

```rust
pub struct SymbolicGraph {
    global_id: GlobalId,                                    // Unique graph identifier
    unknown_dimensions: HashMap<String, SymbolicScalar>,    // Named symbolic dimensions
    tensors: HashMap<GlobalId, ONNXTensorInfo>,             // All tensor metadata
    ordered_inputs: Vec<GlobalId>,                          // Ordered list of graph inputs
    ordered_outputs: Vec<GlobalId>,                         // Ordered list of graph outputs
    operations: HashMap<GlobalId, GraphOperation>,          // All operations by ID
}
```

### ONNXTensorInfo

Tensor metadata representation:

```rust
pub struct ONNXTensorInfo {
    global_id: GlobalId,                        // Unique tensor identifier
    pub onnx_name: Option<String>,              // Original ONNX name
    pub dtype: Option<DType>,                   // Data type (may be unknown)
    pub shape: Option<Vec<ScalarInfoTyped<u64>>>, // Shape (may contain symbolic dims)
    pub tensor_type: TensorType,                // Input/Output/Intermediate/Constant
}

pub enum TensorType {
    Input(Option<StoredOrNotTensor>),    // Graph input, optionally with default value
    Output,                               // Graph output
    Intermediate,                         // Internal tensor between operations
    Constant(StoredOrNotTensor),          // Constant value (inline or stored)
}
```

### StoredOrNotTensor

Represents constant tensor data with lazy loading support:

```rust
pub enum StoredOrNotTensor {
    NotStored(NDArrayNumericTensor<DynRank>),  // Small tensor stored inline
    Stored(TensorStoreTensorId),               // Reference to TensorStore entry
}
```

### GraphOperation

Wrapper for operations with optional naming:

```rust
pub struct GraphOperation {
    pub name: Option<String>,    // Original ONNX node name
    pub op: AnyOperation,        // The actual operation
}
```

### SymbolicGraphMutator

Builder for constructing graphs during ONNX import:

```rust
pub struct SymbolicGraphMutator {
    graph: Option<SymbolicGraph>,
    tensors_by_name: HashMap<String, GlobalId>,
    unknown_dimensions_by_name: HashMap<String, SymbolicScalarTyped<u64>>,
    symbolic_resolver: SymbolicResolver,
    tensor_store: TensorStore,
}
```

---

## Symbolic Dimensions

SymbolicGraph supports unknown dimensions through the symbolic scalar system (from `src/symbolic_scalar.rs`):

### SymbolicScalar

```rust
pub struct SymbolicScalarTyped<T> {
    _phantom_type: PhantomData<T>,
    offset: i64,          // Constant offset added to symbol
    symbol_idx: usize,    // Index into resolver's symbol table
}

pub struct SymbolicResolver {
    next_symbolic_id: usize,  // Counter for generating unique symbol IDs
}
```

**Purpose**: Allows shape inference to proceed even when some dimensions are unknown. For example, a batch dimension might be symbolic until runtime:

```rust
// Shape might be [batch_size, 512, 768] where batch_size is symbolic
let shape = vec![
    ScalarInfoTyped::Symbolic(batch_size_symbol),
    ScalarInfoTyped::Concrete(512),
    ScalarInfoTyped::Concrete(768),
];
```

---

## Tensor Storage

The `TensorStore` (in `src/symbolic_graph/tensor_store.rs`) provides efficient storage for constant tensors:

```rust
pub struct TensorStore {
    next_tensor_id: TensorStoreTensorId,
    tensors: HashMap<TensorStoreTensorId, StoredTensor>,
}

pub enum StoredTensor {
    // Tensor loaded into memory
    Numeric(NumericTensor<DynRank>),
    
    // Lazy-load from binary file at specific offset
    ExternalBinary {
        path: PathBuf,
        offset: u64,
        length: u64,
        dtype: DType,
        shape: Vec<usize>,
    },
    
    // Lazy-load from PyTorch .pth file (requires candle feature)
    ExternalPth {
        path: PathBuf,
        tensor_name: String,
        dtype: DType,
        shape: Vec<usize>,
    },
}
```

**Storage Policy**: Tensors with >100 elements are stored in TensorStore rather than inline, reducing memory pressure during graph construction.

---

## Operation System

### AnyOperation Enum

All 40+ operation types are unified via `AnyOperation` (`src/symbolic_graph/ops/mod.rs`):

```rust
pub enum AnyOperation {
    // Arithmetic
    Unary(UnaryOperation),
    Binary(BinaryOperation),
    
    // Type Operations
    Cast(CastOperation),
    
    // Shape Operations
    Reshape(ReshapeOperation),
    Squeeze(SqueezeOperation),
    Unsqueeze(UnsqueezeOperation),
    Transpose(TransposeOperation),
    Shape(ShapeOperation),
    Size(SizeOperation),
    
    // Indexing
    Slice(SliceOperation),
    Gather(GatherOperation),
    Split(SplitOperation),
    Concat(ConcatOperation),
    
    // Reductions
    ReduceMean(ReduceMeanOperation),
    ReduceSum(ReduceSumOperation),
    ReduceProd(ReduceProdOperation),
    ReduceMin(ReduceMinOperation),
    ReduceMax(ReduceMaxOperation),
    CumSum(CumSumOperation),
    
    // Neural Network
    Conv(ConvOperation),
    Gemm(GemmOperation),
    LayerNormalization(LayerNormalizationOperation),
    GroupNormalization(GroupNormalizationOperation),
    RMSNormalization(RMSNormalizationOperation),
    RotaryEmbedding(RotaryEmbeddingOperation),
    
    // Control Flow
    If(IfOperation),
    Scan(ScanOperation),
    
    // Constants
    Constant(ConstantOperation),
    ConstantOfShape(ConstantOfShapeOperation),
    
    // Misc
    Where(WhereOperation),
    Pad(PadOperation),
    Resize(ResizeOperation),
    Range(RangeOperation),
    // ... and more
}
```

### Operation Trait

The core interface for all operations:

```rust
pub trait Operation: Node {
    /// Execute this operation
    fn eval(
        &self,
        backend: &mut EvalBackend,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
    ) -> OperationEvalRet;
    
    /// Convert to low-level MilliOpGraph for execution
    fn get_milli_op_graph(&self, rng: &mut impl Rng) -> MilliOpGraph;
    
    /// Return nested sub-graphs (for If, Scan operations)
    fn get_sub_graphs(&self) -> Vec<&SymbolicGraph> {
        vec![]  // Default: no sub-graphs
    }
}
```

**Default Implementation**: The default `eval` creates a MilliOpGraph and delegates execution:

```rust
fn eval(&self, backend: &mut EvalBackend, inputs: &HashMap<GlobalId, NumericTensor<DynRank>>) -> OperationEvalRet {
    let milli_graph = self.get_milli_op_graph(&mut rng);
    milli_graph.eval(inputs, &mut (), backend)
}
```

---

## Operation Categories

### Unary Operations (`ops/unary.rs`)

| Category | Operations |
|----------|------------|
| Activation | `Relu`, `Sigmoid`, `Softplus`, `Softmax`, `LogSoftmax` |
| Arithmetic | `Neg`, `Abs`, `Sign`, `Reciprocal` |
| Exponential | `Exp`, `Log`, `Sqrt` |
| Rounding | `Floor`, `Ceil`, `Round` |
| Trigonometric | `Sin`, `Sinh`, `Cos`, `Cosh`, `Tan`, `Tanh`, `Asin`, `Acos`, `Atan`, etc. |
| Logical | `Not`, `BitwiseNot` |
| Special | `IsNan`, `Erf`, `NonZero` |

### Binary Operations (`ops/binary.rs`)

| Category | Operations |
|----------|------------|
| Arithmetic | `Add`, `Sub`, `Mul`, `Div`, `Mod`, `Pow` |
| Matrix | `MatMul`, `Gemm` |
| Comparison | `Equal`, `Greater`, `GreaterOrEqual`, `Less`, `LessOrEqual` |
| Logical | `And`, `Or`, `Xor` |
| Bitwise | `BitwiseAnd`, `BitwiseOr`, `BitwiseXor` |
| Element-wise | `Max`, `Min` |
| Index | `ArgMax`, `ArgMin` |

### Shape Operations

| Operation | File | Description |
|-----------|------|-------------|
| `Reshape` | `reshape.rs` | Change tensor shape, supports -1 inference |
| `Squeeze` | `reshape.rs` | Remove dimensions of size 1 |
| `Unsqueeze` | `reshape.rs` | Insert dimensions of size 1 |
| `Flatten` | `reshape.rs` | Collapse dimensions to 1D/2D |
| `Transpose` | `transpose.rs` | Permute tensor dimensions |
| `Shape` | `shape.rs` | Return tensor shape as 1D tensor |
| `Size` | `shape.rs` | Return total element count |
| `Expand` | `misc.rs` | Broadcast tensor to larger shape |

### Indexing Operations

| Operation | File | Description |
|-----------|------|-------------|
| `Slice` | `slice.rs` | Extract contiguous sub-tensor |
| `Gather` | `gather.rs` | Index selection along axis |
| `Split` | `split.rs` | Divide tensor into parts |
| `Concat` | `concat.rs` | Join tensors along axis |

### Reduction Operations (`ops/reduce.rs`)

| Operation | Description |
|-----------|-------------|
| `ReduceMean` | Average along axes |
| `ReduceSum` | Sum along axes |
| `ReduceProd` | Product along axes |
| `ReduceMin` | Minimum along axes |
| `ReduceMax` | Maximum along axes |
| `CumSum` | Cumulative sum along axis |

### Normalization Operations (`ops/normalization.rs`)

| Operation | Description |
|-----------|-------------|
| `LayerNormalization` | Per-sample normalization |
| `GroupNormalization` | Group-wise normalization |
| `RMSNormalization` | Root mean square normalization |
| `LpNormalization` | Lp norm normalization |
| `InstanceNormalization` | Per-instance normalization |

### Type Operations (`ops/cast.rs`)

| Operation | Description |
|-----------|-------------|
| `Cast` | Convert tensor to specified dtype |
| `CastLike` | Convert tensor to match another's dtype |

### Constant Operations (`ops/constant.rs`)

| Operation | Description |
|-----------|-------------|
| `Constant` | Embed fixed tensor value |
| `ConstantOfShape` | Create tensor filled with scalar, shape from input |

### Control Flow Operations

#### If Operation (`ops/misc.rs`)

```rust
pub struct IfOperation {
    global_id: GlobalId,
    outputs: Vec<GlobalId>,
    condition: GlobalId,
    then_branch: SymbolicGraph,  // Nested graph for true case
    else_branch: SymbolicGraph,  // Nested graph for false case
}
```

Executes one of two sub-graphs based on boolean condition.

#### Scan Operation (`ops/scan.rs`)

```rust
pub struct ScanOperation {
    scan_inputs: Vec<Option<GlobalId>>,      // Inputs iterated over
    state_inputs: Vec<Option<GlobalId>>,     // Initial loop state
    scan_outputs: Vec<Option<GlobalId>>,     // Accumulated outputs
    state_outputs: Vec<Option<GlobalId>>,    // Final state
    body: SymbolicGraph,                      // Loop body graph
    scan_input_axes: Option<Vec<i64>>,       // Axes to scan
}
```

Iterates over input tensor axis, executing body graph each iteration with state passing.

### Other Operations

| Operation | File | Description |
|-----------|------|-------------|
| `Conv` | `conv.rs` | Convolution operation |
| `Resize` | `resize.rs` | Tensor resizing with interpolation |
| `Pad` | `misc.rs` | Add padding to tensor |
| `Where` | `misc.rs` | Conditional element selection |
| `Range` | `misc.rs` | Generate numeric sequence |
| `Clip` | `misc.rs` | Clamp values to range |
| `IsInf` | `misc.rs` | Check for infinity |
| `RandomNormalLike` | `misc.rs` | Random normal tensor |
| `RotaryEmbedding` | `rotary_embedding.rs` | Rotary position embeddings |

---

## ONNX Import

### Import Flow

```
ONNX Protobuf Bytes
        │
        ▼
SymbolicGraphMutator::from_onnx_bytes()
        │
        ├─► Decode protobuf via prost
        │
        ├─► Create tensors from ValueInfoProto
        │
        ├─► Parse initializers (constants)
        │
        ├─► For each ONNX node:
        │       │
        │       ├─► Match operation type string
        │       │
        │       ├─► Call Operation::from_onnx()
        │       │
        │       └─► Register in operations map
        │
        └─► Return (SymbolicGraph, TensorStore)
```

### Attribute Parsing Helpers

The module provides utilities for extracting ONNX attributes:

```rust
fn query_attribute_float(attributes: &[AttributeProto], name: &str) -> Option<f32>
fn query_attribute_floats(attributes: &[AttributeProto], name: &str) -> Option<Vec<f32>>
fn query_attribute_int(attributes: &[AttributeProto], name: &str) -> Option<i64>
fn query_attribute_ints(attributes: &[AttributeProto], name: &str) -> Option<Vec<i64>>
fn query_attribute_string(attributes: &[AttributeProto], name: &str) -> Option<String>
fn query_attribute_tensor(attributes: &[AttributeProto], name: &str) -> Option<NDArrayNumericTensor<DynRank>>
fn query_attribute_graph(attributes: &[AttributeProto], name: &str) -> Option<&GraphProto>
```

### Example: Parsing a Binary Operation

```rust
"Add" => Some(AnyOperation::Binary(BinaryOperation::from_onnx(
    &input_tensors,
    &output_tensors,
    WhichBinaryOperation::Add,
    rng,
)?)),
```

Each operation's `from_onnx()` method handles its specific ONNX attributes and creates the appropriate structure.

---

## Graph Execution

### Execution Flow

```rust
pub fn eval(
    &self,
    inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
    eval_backend: &mut EvalBackend,
) -> Result<HashMap<GlobalId, NumericTensor<DynRank>>, EvalError>
```

**Algorithm:**

1. **Initialize**: Load input tensors and constants into `active_tensors`
2. **Iterate**: Loop until all operations complete:
   - Find operations with all inputs available
   - Execute operation via `op.eval()`
   - Validate outputs against declared tensor info
   - Store outputs in `active_tensors`
3. **Return**: Final `active_tensors` map containing outputs

### Tensor Validation

During execution, each tensor is validated:

```rust
// Check rank matches
if declared_shape.len() != actual_shape.len() {
    return Err(EvalError::UnexpectedRank(...));
}

// Check dimensions match (concrete or symbolic)
for (declared, actual) in declared_shape.iter().zip(actual_shape) {
    match declared {
        ScalarInfoTyped::Concrete(expected) => {
            if expected != actual {
                return Err(EvalError::UnexpectedDimension(...));
            }
        }
        ScalarInfoTyped::Symbolic(_) => {
            // Symbolic dimensions always match
        }
    }
}
```

---

## Observer Pattern

The `SymbolicGraphObserver` trait (`src/symbolic_graph/observer.rs`) enables execution monitoring:

```rust
pub trait SymbolicGraphObserver {
    fn on_op_executed(
        &mut self,
        node_path: &[GlobalId],
        start_instant: Instant,
        end_instant: Instant,
        backend: &mut EvalBackend,
    );
    
    fn on_tensor_assigned(
        &mut self,
        tensor_path: &[GlobalId],
        tensor: &NumericTensor<DynRank>,
        backend: &mut EvalBackend,
    );
}

// No-op implementation for when observation is not needed
impl SymbolicGraphObserver for () {
    fn on_op_executed(...) {}
    fn on_tensor_assigned(...) {}
}
```

---

## Core Traits

### Node Trait

Graph connectivity interface (from `src/graph.rs`):

```rust
pub trait Node {
    type OpKind: AsRef<str> + Clone + Debug;
    
    fn global_id(&self) -> GlobalId;
    fn op_kind(&self) -> Self::OpKind;
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_>;
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_>;
    fn label(&self) -> Option<String> { None }
}
```

### Link Trait

Tensor node interface:

```rust
pub trait Link {
    fn global_id(&self) -> GlobalId;
    fn label(&self) -> Option<String> { None }
}
```

### Graph Trait

Shared interface for all graph types:

```rust
pub trait Graph {
    type AnyNode: Node;
    type AnyLink: Link;
    
    fn global_id(&self) -> GlobalId;
    fn node_ids(&self) -> impl Iterator<Item = GlobalId>;
    fn inner_link_ids(&self) -> impl Iterator<Item = GlobalId>;
    fn get_node_by_id(&self, id: &GlobalId) -> Option<&Self::AnyNode>;
    fn get_link_by_id(&self, id: &GlobalId) -> Option<&Self::AnyLink>;
    fn input_link_ids(&self) -> impl Iterator<Item = (GlobalId, GlobalId)>;
    fn output_link_ids(&self) -> impl Iterator<Item = (GlobalId, GlobalId)>;
    fn constant_link_ids(&self) -> impl Iterator<Item = GlobalId>;
}
```

---

## Error Handling

### ONNXDecodingError

Errors during ONNX import:

```rust
pub enum ONNXDecodingError {
    InvalidOperatorInputs(&'static str),
    InvalidOperatorOutputs(&'static str),
    MissingField(&'static str),
    UnsupportedTypeValue(onnx::type_proto::Value),
    ProtobufDecodeError(#[from] anyhow::Error),
    UnsupportedDType(onnx::tensor_proto::DataType),
    UnsupportedONNXType(String),
    NegativeDimensionError,
    UnknownTensorName(String),
    MissingAttribute(String, String),
    DTypeError(#[from] DTypeError),
    NDArrayNumericTensorError(#[from] NDArrayNumericTensorError),
    UnsupportedONNX(String),
}
```

### EvalError

Errors during graph execution:

```rust
pub enum EvalError {
    UnexpectedDType(DType, DType),
    UnimplementedOperatorError(String),
    InvalidInput(String),
    UnexpectedDimension(u64, u64, Vec<u64>),
    UnexpectedRank(usize, usize),
    MissingInputTensor(String, Option<DType>, Option<Vec<usize>>),
    // ...
}
```

---

## Integration Points

### With MilliOpGraph

Each symbolic operation implements `get_milli_op_graph()` to bridge to the lower execution layer:

```rust
impl Operation for UnaryOperation {
    fn get_milli_op_graph(&self, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let a = input_map[&self.input];
        
        let out_tid = match &self.which {
            WhichUnaryOperation::Relu => 
                milli_graph::ops::ClampMin::push_new(&mut graph, a, 0.0, rng),
            WhichUnaryOperation::Exp => 
                milli_graph::ops::SimpleUnaryOp::exp(&mut graph, a, rng),
            // ... more mappings
        };
        
        let mut output_map = HashMap::new();
        output_map.insert(out_tid, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
```

**Execution Hierarchy:**
```
SymbolicGraph::eval()
        │
        ▼ (for each operation)
Operation::eval()
        │
        ▼
Operation::get_milli_op_graph()
        │
        ▼
MilliOpGraph::eval()
        │
        ▼
EvalBackend (actual computation)
```

### With SuperGraph

The SuperGraph layer can reference SymbolicGraph as a building block for higher-level system composition, mixing model execution with other operations like tokenization and caching.

### With Backends

All tensor computation ultimately delegates to `EvalBackend`:
- **NDArray**: CPU reference implementation
- **Vulkan**: GPU compute backend
- Future backends can be added via the abstraction

---

## Serialization

All core types support serde serialization:

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SymbolicGraph { ... }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphOperation { ... }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AnyOperation { ... }
```

This enables saving/loading graphs for caching, debugging, and transfer.

---

## File Structure

```
src/symbolic_graph/
├── mod.rs              # SymbolicGraph, SymbolicGraphMutator, ONNX import
├── observer.rs         # SymbolicGraphObserver trait
├── tensor_store.rs     # TensorStore for lazy tensor loading
└── ops/
    ├── mod.rs          # AnyOperation enum, Operation trait
    ├── binary.rs       # Binary operations (Add, MatMul, etc.)
    ├── cast.rs         # Cast, CastLike
    ├── concat.rs       # Concat operation
    ├── constant.rs     # Constant, ConstantOfShape
    ├── conv.rs         # Convolution
    ├── gather.rs       # Gather operation
    ├── misc.rs         # If, Where, Pad, Range, Expand, etc.
    ├── normalization.rs # LayerNorm, GroupNorm, RMSNorm, etc.
    ├── reduce.rs       # ReduceMean, ReduceSum, CumSum, etc.
    ├── reshape.rs      # Reshape, Squeeze, Unsqueeze, Flatten
    ├── resize.rs       # Resize operation
    ├── rotary_embedding.rs # Rotary position embeddings
    ├── scan.rs         # Scan loop operation
    ├── shape.rs        # Shape, Size operations
    ├── slice.rs        # Slice operation
    ├── split.rs        # Split operation
    ├── transpose.rs    # Transpose operation
    └── unary.rs        # Unary operations (Relu, Exp, Trig, etc.)
```

Related files:
```
src/
├── symbolic_scalar.rs  # SymbolicScalar, SymbolicResolver
└── graph.rs            # Node, Link, Graph traits
```

---

## Implementation Patterns

### Standard Operation Structure

Each operation follows a consistent pattern:

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SomeOperation {
    global_id: GlobalId,         // Unique operation ID
    pub input: GlobalId,         // Input tensor ID
    pub output: GlobalId,        // Output tensor ID
    // Operation-specific parameters...
}

impl SomeOperation {
    pub fn from_onnx(
        inputs: &[GlobalId],
        outputs: &[GlobalId],
        attributes: &[AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        // Parse ONNX attributes
        // Validate inputs/outputs
        // Construct operation
    }
}

impl Node for SomeOperation {
    type OpKind = String;
    
    fn global_id(&self) -> GlobalId { self.global_id }
    fn op_kind(&self) -> String { "SomeOperation".to_string() }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new([self.input].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new([self.output].into_iter())
    }
}

impl Operation for SomeOperation {
    fn get_milli_op_graph(&self, rng: &mut impl Rng) -> MilliOpGraph {
        // Build equivalent MilliOpGraph
    }
}
```

### Macro-Based Delegation

The `AnyOperation` enum uses delegation macros to forward trait methods:

```rust
macro_rules! delegate {
    ($name:ident($($arg:ident: $ty:ty),*) -> $ret:ty) => {
        fn $name(&self, $($arg: $ty),*) -> $ret {
            match self {
                AnyOperation::Unary(x) => x.$name($($arg),*),
                AnyOperation::Binary(x) => x.$name($($arg),*),
                // ... all variants
            }
        }
    }
}
```

---

## Summary

The SymbolicGraph subsystem provides:

- **Complete ONNX Support**: 40+ operation types covering neural network, arithmetic, and control flow operations
- **Symbolic Shape Handling**: Unknown dimensions tracked via SymbolicScalar for dynamic batch sizes
- **Lazy Tensor Loading**: Efficient memory management through TensorStore with multiple storage backends
- **Two-Layer Architecture**: High-level ONNX operations bridge to low-level MilliOpGraph for execution
- **Control Flow**: Full support for conditional (If) and iterative (Scan) operations with nested graphs
- **Type Safety**: Runtime validation of shapes and dtypes against declared specifications
- **Serialization**: Full serde support for graph persistence
- **Extensibility**: Clean trait-based design enables new operations and backends
