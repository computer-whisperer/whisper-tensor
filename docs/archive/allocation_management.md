# Allocation Management Migration

## Motivation

whisper-tensor currently uses the system heap for all allocations with no tracking, budgeting, or pool management. Large ML models routinely OOM the host. Long-term, whisper-tensor-server needs to:

- Limit its own RAM usage
- Know which high-level tasks (model loads, inference requests) own which buffers
- Make resource contention explicit rather than implicit
- Free intermediates incrementally as ops complete, not at the end of a request

This migration also restructures the tensor type system and execution model to
align with whisper-tensor's future direction as an immediate-mode CPU tensor
system with managed memory.

## Architecture Overview

### What Changes

| Before | After |
|--------|-------|
| `NumericTensor` is a multi-backend dispatch enum (NDArray, Vulkan, Candle, TCH, Packed) | `NumericTensor` is a struct: bytes in a managed pool |
| `NDArrayNumericTensor` has 18 dtype variants, each holding `ArcArray<T>` | One struct with dtype-erased buffer + `NumericDType` metadata |
| `PackedTensor` is a separate type for quantized data | Subsumed into `NumericTensor` — block-quantized is just a layout variant |
| `DType` is a flat enum with 18+ variants | `NumericDType` structured as `Float(FloatType) \| SignedInt(IntType) \| UnsignedInt(IntType) \| Bool` |
| Tensor methods implement operations (add, matmul, cast, etc.) | Tensors are pure data; all operations go through `MilliOpGraph` |
| ndarray is the core computation engine | Removed — nano-op eval for v1, compiler for performance |
| Vulkan, Candle, TCH backends are tensor variants | Dropped entirely |
| Heap allocation via Vec/Arc with no tracking | Pool-managed buffers with lifetime enforcement and budget control |

### What Stays

- `MilliOpGraph` and its op definitions
- Nano-op lowering and evaluation
- `SymbolicGraph` (ONNX import layer)
- `SuperGraph` (multi-model orchestration)
- Model loading pipeline (tensor_store, GGUF, safetensors)
- The compiler (already targets raw buffers)

## Design

### Core Trait: `Pool`

A `Pool` is a general-purpose allocation facility — not tensor-specific. Tensors
and any other large buffer consumers all allocate through pools.

```rust
trait Pool {
    type Buffer<'a>: Deref<Target = [u8]> + DerefMut where Self: 'a;

    fn allocate(&self, size: usize) -> Result<Self::Buffer<'_>, AllocationError>;
    fn bytes_in_use(&self) -> usize;
    fn budget(&self) -> Option<usize>;
}
```

**Key properties:**
- `Buffer` is a GAT — its lifetime is tied to the pool via `&self` borrow
- `Buffer::drop()` returns memory to the pool incrementally (pool uses interior
  mutability for its freelist)
- The pool owns the backing memory; buffers borrow from it
- Compile-time enforcement: tensors cannot outlive their pool

### Concrete Pool Implementations

```
TrackedPool    — freelist-based, incremental free, usage tracking per tag
SystemPool     — passthrough to global allocator (backward compat / tests)
```

No type-erased `DynPool` is needed. Every collection of tensors is homogeneous
in pool type — model weights all come from the model pool, request intermediates
all come from the request pool. The eval loop takes these as separate inputs
rather than mixing pool types in one HashMap.

**Design constraint:** the eval loop is generic over a single `P: Pool` type.
Model pools and request pools must be the same pool *type* (e.g., both
`TrackedPool`) but can be different *instances* with different budgets and
lifetimes. This is deliberate — it avoids type-level complexity while still
allowing per-model and per-request budget control. A `DynPool` type-erased
wrapper can be built later if heterogeneous pool types are ever needed.

Pool implementations must be `Send + Sync` — model pools are shared across
concurrent request-handling threads via `spawn_blocking`.

### NumericTensor: Pure Data

`NumericTensor` is a struct, not an enum. It represents "an n-dimensional
indexable set of numbers in a CPU buffer that we manage." No backend dispatch,
no operation methods. The tensor is the noun — it has no verbs.

Shape is always a root-level property (always meaningful regardless of storage).
DType is folded into the layout (it describes storage, not the tensor's
identity).

```rust
/// General tensor — any storage format. Pool-managed buffer.
struct NumericTensor<'a, R: Rank, P: Pool> {
    buffer: P::Buffer<'a>,
    shape: R::KnownDims,
    layout: BufferLayout<R>,
}

/// How elements are stored in the buffer.
struct BufferLayout<R: Rank> {
    format: StorageFormat,
    strides: R::KnownDims,    // in bits, always positive — no negative stride support
}

/// The storage format of elements in a buffer.
enum StorageFormat {
    /// Uniform-stride elements of a single numeric type.
    /// Strides (in BufferLayout) fully describe the element spacing.
    Element {
        dtype: NumericDType,
    },
    /// Block-quantized with per-block scales/offsets (GGUF Q4_K, etc.)
    BlockQuantized(PackedFormat),
}
```

This replaces:
- `NDArrayNumericTensor` (18 `ArcArray<T>` variants → one dtype-erased buffer)
- `PackedTensor` (subsumed — block-quantized is a `StorageFormat` variant)
- The multi-backend dispatch enum

### NumericScalarView: The Fundamental Scalar Primitive

All typed numeric operations are defined on `NumericScalarView` — a view into
scalar data at an arbitrary bit offset within a byte buffer.

```rust
/// View into scalar data at an arbitrary bit offset.
/// The fundamental unit of typed numeric operations.
struct NumericScalarView<'a> {
    data: &'a [u8],
    bit_offset: usize,
    dtype: NumericDType,
}

struct NumericScalarViewMut<'a> {
    data: &'a mut [u8],
    bit_offset: usize,
    dtype: NumericDType,
}

/// Owned convenience wrapper — small local buffer + view into it.
struct NumericScalar {
    bits: [u8; 8],         // enough for any scalar up to 64 bits
    dtype: NumericDType,
}

impl NumericScalar {
    fn view(&self) -> NumericScalarView<'_> {
        NumericScalarView { data: &self.bits, bit_offset: 0, dtype: self.dtype }
    }
}
```

Public operations live on `NumericScalar`, taking and returning owned values
(copying ~12 bytes is essentially free). Internally, they delegate to
`NumericScalarView` where the bit-manipulation logic lives:

```rust
impl NumericScalar {
    fn add(&self, other: &NumericScalar) -> NumericScalar {
        let mut result = NumericScalar::zero(self.dtype);
        self.view().add(&other.view(), &mut result.view_mut());
        result
    }
    fn cast_to(&self, target: NumericDType) -> NumericScalar { ... }
    fn to_f64(&self) -> f64 { self.view().to_f64() }
}
```

`NumericScalarView` / `NumericScalarViewMut` are also public — useful for
in-place operations on tensor buffers without round-tripping through owned
scalars. But the primary API for most callers is owned `NumericScalar` in/out.
The view types are where the bit-manipulation logic (software float arithmetic,
bit-level interpretation) lives in one place.

**Key properties:**

- **Works on any dtype** — known types (f32, f64, bf16, ...) fast-path to
  native Rust operations. Unknown/novel float formats fall back to software
  arithmetic that interprets bits according to the exponent/mantissa spec.
  Add a new ML float format → define the spec → operations just work.

- **Tensor element access** — `NumericTensor::read_element()` returns an owned
  `NumericScalar`. The copy (~12 bytes) is trivially cheap. Works for all
  storage formats including block-quantized.

- **Connects to bit-stride layout** — iterating a tensor's elements is stepping
  through the buffer by `stride` bits per element.

- **Better packing** — `NumericScalar` is `[u8; 8]` + dtype (~12 bytes) vs the
  current ~24-byte enum. Nano-op eval buffers shrink significantly.

This replaces the current 18-variant `NumericScalar` enum. The variants become
dtype metadata rather than Rust type-level dispatch.

### NumericTensorView: Zero-Copy Tensor Borrowing

Same pattern as NumericScalar/NumericScalarView — the owned type wraps a view
type, complex logic lives on the view, owned type delegates.

```rust
/// View into tensor data — borrows a byte slice, no pool involvement.
/// Enables zero-copy reshape/transpose via stride manipulation.
struct NumericTensorView<'a, R: Rank> {
    data: &'a [u8],
    shape: R::KnownDims,
    layout: BufferLayout<R>,
}

/// Owned tensor — pool-managed buffer.
struct NumericTensor<'a, R: Rank, P: Pool> {
    buffer: P::Buffer<'a>,
    shape: R::KnownDims,
    layout: BufferLayout<R>,
}

impl<'a, R: Rank, P: Pool> NumericTensor<'a, R, P> {
    fn view(&self) -> NumericTensorView<'_, R> {
        NumericTensorView {
            data: &self.buffer,
            shape: self.shape.clone(),
            layout: self.layout.clone(),
        }
    }
}
```

`NumericTensorView` borrows `&[u8]` — no pool, no lifetime coupling to a pool
type. This has several important consequences:

- **Zero-copy reshape/transpose** — produce new views with different
  shape/strides pointing at the same data. Not implemented in v1, but the
  type structure supports it without rework.

- **Op eval takes `NumericTensorView`** — this erases the pool lifetime at the
  computation boundary. Ops don't care whether their inputs come from a model
  pool, a request pool, or anywhere else. Solves the mixed-lifetime problem
  in the eval loop: model weights (`'model`) and request intermediates
  (`'req`) are both passed as `NumericTensorView<'_>` to ops.

- **Direct mmap access** — a `NumericTensorView` can point at raw mmap'd file
  storage (safetensors, GGUF). `read_element()` reads directly from the
  mmap'd bytes — including dequantizing block-quantized weights on the fly.
  No pool allocation needed for model weights that are only read, not
  mutated. The OS pages data in on demand.

### Element Access

Element access lives on `NumericTensorView`. `NumericTensor` delegates to its
view. All access returns owned `NumericScalar` — the copy is trivially cheap.

```rust
impl<'a, R: Rank> NumericTensorView<'a, R> {
    /// Read a single element by flat index, returning owned scalar.
    /// Works for all storage formats including block-quantized.
    fn read_element(&self, flat_index: usize) -> NumericScalar {
        match &self.layout.format {
            StorageFormat::Element { dtype } => {
                let bit_offset = self.compute_bit_offset(flat_index);
                read_scalar_at(self.data, bit_offset, dtype)
            }
            StorageFormat::BlockQuantized(fmt) => {
                let block_size = fmt.block_size();
                let block_idx = flat_index / block_size;
                let element_in_block = flat_index % block_size;
                dequantize_block_element(self.data, fmt, block_idx, element_in_block)
            }
        }
    }
}

impl<'a, R: Rank, P: Pool> NumericTensor<'a, R, P> {
    fn read_element(&self, flat_index: usize) -> NumericScalar {
        self.view().read_element(flat_index)
    }
}
```

Batched typed access (reading contiguous runs of elements as typed slices,
skipping per-element dispatch) is a **compiler optimization**, not part of the
v1 correctness path.

### DType Rework

The current `DType` enum conflates ONNX interop naming, numeric type semantics,
and storage layout. The rework separates these cleanly.

#### Layer 1: ONNX Interop

```rust
/// What ONNX model files give us. Bridges the gap between ONNX's type system
/// and our internal numeric type system.
enum ONNXDType {
    Numeric(NumericDType),
    String,
}
```

STRING exists only because ONNX insists on it. It is not a numeric type and
does not participate in the pool-managed tensor system.

Note: the current `NumericScalar` enum has a `STRING(String)` variant used by
`ScalarOp::Literal` in the nano-graph. The new `NumericScalar` (backed by
`[u8; 8]` + `NumericDType`) cannot hold strings. `ScalarOp::Literal` will need
a separate `LiteralValue` enum that can hold both `NumericScalar` and `String`,
or STRING literal support can be dropped if no nano-lowering path produces
string literals (to be verified during Phase 1).

#### Layer 2: Numeric Type Semantics

The semantic meaning of a value — independent of how it's stored in memory.

```rust
/// What a numeric value IS — its mathematical interpretation.
enum NumericDType {
    UnsignedInt(IntType),
    SignedInt(IntType),
    Float(FloatType),
    Bool,
}

/// An integer type, parameterized by bit width.
struct IntType {
    bits: u8,  // 4, 8, 16, 32, 64
}

/// A floating-point type, parameterized by exponent and mantissa width
/// plus semantic variant for formats that share the same bit layout.
/// Total bits = 1 (sign) + exponent_bits + mantissa_bits.
struct FloatType {
    exponent_bits: u8,
    mantissa_bits: u8,
    semantics: FloatSemantics,
}

/// Distinguishes float formats that share the same exponent/mantissa
/// bit widths but differ in NaN, Inf, or negative-zero behavior.
enum FloatSemantics {
    /// Standard IEEE 754 — NaN, Inf, negative zero all present.
    /// Covers F64, F32, F16, BF16, F8E5M2.
    IEEE,
    /// Finite, no infinities, special NaN encoding.
    /// Covers F8E4M3FN, F4E2M1.
    FN,
    /// Finite, no negative zero, unsigned zero, different NaN.
    /// Covers F8E4M3FNUZ, F8E5M2FNUZ.
    FNUZ,
}
```

Named constants for common types:
```rust
impl FloatType {
    const F64: Self     = FloatType { exponent_bits: 11, mantissa_bits: 52, semantics: FloatSemantics::IEEE };
    const F32: Self     = FloatType { exponent_bits: 8,  mantissa_bits: 23, semantics: FloatSemantics::IEEE };
    const F16: Self     = FloatType { exponent_bits: 5,  mantissa_bits: 10, semantics: FloatSemantics::IEEE };
    const BF16: Self    = FloatType { exponent_bits: 8,  mantissa_bits: 7,  semantics: FloatSemantics::IEEE };
    const F8E4M3FN: Self = FloatType { exponent_bits: 4, mantissa_bits: 3,  semantics: FloatSemantics::FN };
    const F8E5M2: Self  = FloatType { exponent_bits: 5,  mantissa_bits: 2,  semantics: FloatSemantics::IEEE };
    const F4E2M1: Self  = FloatType { exponent_bits: 2,  mantissa_bits: 1,  semantics: FloatSemantics::FN };
}

impl IntType {
    const BITS_4: Self  = IntType { bits: 4  };
    const BITS_8: Self  = IntType { bits: 8  };
    const BITS_16: Self = IntType { bits: 16 };
    const BITS_32: Self = IntType { bits: 32 };
    const BITS_64: Self = IntType { bits: 64 };
}
```

#### Layer 3: Storage Layout (BufferLayout)

How values are packed into a buffer — separate from what the values mean.
Strides are part of the layout, measured in **bits**. This unifies byte-aligned
and sub-byte layouts under a single indexing model.

| dtype | semantic bits | innermost stride (bits) | meaning |
|-------|--------------|------------------------|---------|
| F32   | 32           | 32                     | standard float32 |
| U8    | 8            | 8                      | standard byte |
| U4    | 4            | 4                      | two per byte, packed |
| U4    | 4            | 8                      | one per byte, padded |
| Bool  | 1            | 1                      | bit-packed |
| Bool  | 1            | 8                      | one per byte |

For `StorageFormat::Element`, strides fully describe the element spacing.
No separate `bit_stride` field is needed — the innermost stride *is* the
element spacing.

Block quantization (GGUF Q4_K, etc.) uses `StorageFormat::BlockQuantized`
with per-block scales/offsets. Strides are not directly meaningful for block-
quantized layouts — per-element access uses `read_element()` which
dequantizes on the fly.

#### Dispatch to Rust Types

Dtype-to-Rust-type dispatch happens inside `NumericScalarView` operations
(fast paths for known types like f32, f64, etc. — software fallback for
others). This is internal to the scalar view implementation, not a public
macro or dispatch table. The compiler generates native typed code directly
and does not need runtime dispatch.

### Execution Model: MilliOpGraph Is the Only Verb

Tensors are pure data. All computation goes through `MilliOpGraph`.

**Before:** tensor methods implement operations
```rust
// caller reaches into tensor and calls methods directly
let c = a.add(&b)?;
let d = c.matmul(&weights)?;
```

**After:** build a graph, eval it
```rust
// build a MilliOpGraph describing the computation
let graph = MilliOpGraph::new();
let a_id = graph.input("a", a.shape(), a.dtype());
let b_id = graph.input("b", b.shape(), b.dtype());
let c_id = graph.add_op(MilliOp::Add, &[a_id, b_id]);
let d_id = graph.add_op(MilliOp::MatMul, &[c_id, weight_id]);
graph.mark_output(d_id);

// eval — all allocation goes through the pool
let results = graph.eval(pool, &inputs)?;
```

**Why this works for v1 with minimal code:**

`MilliOp::eval()` already has nano-op lowering. Nano-ops already know how to do
elementwise operations on raw buffers. So the initial implementation:

1. `NumericTensor` struct — tiny (buffer + metadata)
2. `MilliOp::eval()` lowers to nano-ops, evals them on the pool-backed buffer
3. Delete the ~2500 lines of operation implementations on `NDArrayNumericTensor`
4. Delete ndarray, Vulkan, Candle, TCH backends

v1 is slower (everything is scalar nano-ops) but correct and minimal.
Performance recovery comes from maturing the compiler (Phase 6), which already
operates on MilliOpGraphs and produces optimized native code. No intermediate
hand-optimized dispatch layer needed.

The tensor type never accumulates operation methods. Optimizations live in the
graph evaluation layer.

**Tensor-level ops may exist as private implementation details** of the
MilliOp/nano-op eval path, but they are not part of `NumericTensor`'s public
API. The intended interface for all tensor operations is `MilliOpGraph`.

### Nano-Op Eval: V1 Adaptation

The current nano-op eval system was designed as a compiler backend, not as the
primary execution path. Two issues need addressing:

#### Issue 1: NumericScalar Intermediate Overhead (tolerable for v1)

Nano-op eval uses `Vec<NumericScalar>` per group for intermediates. Each
`NumericScalar` is a ~24-byte enum even for u8 values. For a 4096-element group,
that's ~96KB instead of 4KB.

**Why tolerable:** The high-level eval path goes op-by-op at the symbolic graph
level, doing a lower() + eval() per op. Each nano-op eval context is scoped to
one symbolic op's worth of computation. The NumericScalar overhead is bounded
per-op, not per-model. Wasteful but won't destroy RAM.

**Future fix (Phase 6):** Replace `Vec<NumericScalar>` with typed pool-backed
`NumericTensor` buffers. Nano-op groups operate on contiguous typed slices
instead of boxed scalars.

#### Issue 2: Packed Tensor Indexing (must fix for v1)

Nano-op eval currently only handles `NDArrayNumericTensor` inputs. PackedTensor
(block-quantized) has no per-element access — the only path is `dequantize()`
which materializes the entire decompressed tensor.

**Fix:** `NumericTensor::read_element()` provides uniform per-element access
regardless of storage format (see "Element Access" section above). Nano-op eval
uses this method to populate its scalar input buffers, replacing the current
`populate_from_tensor()` which pattern-matches on NDArray dtype variants.

The per-element dispatch cost (matching on `StorageFormat` per read) is
negligible — nano-op eval is already doing per-element `NumericScalar`
operations everywhere. Batched typed access is a compiler optimization concern.

### Pool Scoping / Hierarchy

Different tensor lifetimes need different pool scopes:

| Scope | Lifetime | Contents |
|-------|----------|----------|
| Server pool | `'static` or top-level `main` scope | Budget ceiling for the whole process |
| Model pool | Lives as long as model is loaded | Model weights, constants |
| Request pool | Scoped to one inference request | Intermediate activations |

The server pool is the parent allocator. Model and request pools sub-allocate
from it, enforcing per-model and per-request budgets.

**Model weight cache:** `ModelLoadedTensorCache` currently uses
`Arc<Mutex<HashMap<LoadedModelId, Cache>>>`. With pool-scoped tensors, the model
pool must outlive all requests using that model. The scheduler extracts a
`&'model` pool reference when dispatching a request.

**spawn_blocking boundary:** The request pool is created and moved into the
blocking task closure. All intermediates allocate from it. When the closure
returns, the pool drops and all intermediate memory is freed.

### Eval Loop: Separate Inputs by Pool

The eval loop does NOT mix tensors from different pools in one collection.
Instead, model weights and request intermediates are passed separately:

```rust
fn eval<'model, 'req, P: Pool>(
    &self,
    model_weights: &HashMap<GlobalId, &'model NumericTensor<'model, DynRank, P>>,
    request_pool: &'req P,
    inputs: &HashMap<GlobalId, NumericTensor<'req, DynRank, P>>,
) -> Result<HashMap<GlobalId, NumericTensor<'req, DynRank, P>>>
```

Model weights are borrowed (`&'model`), request intermediates are owned by the
request pool (`'req`). No type erasure needed, no mixed-pool collections. The
model pool must outlive the request (which is structurally guaranteed by the
server's pool hierarchy).

### Serialization Model

The serialization boundary and the pool-tracking boundary are the same line.
Small constants that live inline in graphs don't need pool tracking. Large
weights that need tracking are external references, never serialized inline.

**SystemPool for inline constants:**

`SystemPool` is a stateless passthrough to the global allocator. Its buffer type
is essentially `Vec<u8>` — owned, `'static`, trivially serializable. Graph
structures that embed tensor data (e.g., `Constant` op, `ConstantOfShape`
value) use `NumericTensor<'static, DynRank, SystemPool>`.

```rust
// Constant op — small tensor data embedded in the graph
struct Constant {
    data: NumericTensor<'static, DynRank, SystemPool>,  // serde-compatible
}

// Serialization: just write the buffer bytes + shape + dtype metadata
// Deserialization: allocate from SystemPool (i.e., normal heap alloc)
```

**TrackedPool for runtime tensors:**

Request intermediates and model weights loaded at runtime use `TrackedPool`.
These are never serialized — they exist only during computation.

**TensorStore for large weights:**

Large model weights are already modeled as external references in `TensorStore`
(file path + offset + length + dtype + shape). They are inputs to `MilliOpGraph`,
not embedded constants. This existing system is unchanged.

**At eval time, `NumericTensorView` erases the distinction:**

- `Constant` op produces a `NumericTensorView` from its `SystemPool` data
- Model weights produce a `NumericTensorView` from `TrackedPool` (or mmap)
- Request intermediates produce a `NumericTensorView` from `TrackedPool`
- Op eval sees `NumericTensorView` for all of them — pool type is invisible

**Protocol serialization (WebSocket/CBOR):**

Tensors sent to clients go through manual serialization at the protocol
boundary: read buffer bytes + shape + dtype → CBOR. This doesn't need serde
derive on the tensor type — it's explicit serialization of the view's data.

### Clone Elimination

The current codebase clones `NumericTensor` ~210 times. The pool migration
removes Arc from the picture — tensors are pool-borrowed, not refcounted.

**New pattern (reference-based, separate pool scopes):**

Model weights are borrowed from the model pool. Request intermediates are
allocated from the request pool and owned by the eval loop's local storage.
When a tensor's use count hits zero, dropping it returns the buffer to the
request pool via `Drop`.

The existing `tensor_uses_left` tracking in `eval_backend.rs` already counts
remaining consumers.

### Testing Architecture

The current test suite has 63 NumericTensor integration tests (per-op, expanded
across backends), 74 NanoGraph lowering tests (`check_integrity()` framework),
and 3 MilliOp unit tests (Split only). The migration pivots from testing tensor
methods to testing MilliOpGraph evaluation.

#### Principle: Separate Data from Evaluation

Test cases are split into two independent parts:

1. **Named datasets on disk** — input tensors + expected output tensors.
   Pure data, no graph definitions, no evaluation logic.
2. **Graph construction in Rust code** — `build_test_set()` synthesizes
   MilliOpGraphs and associates them with their named datasets.

This separation allows the same data to be verified from two independent
directions.

#### Named Datasets

```
test_data/
  add_f32/
    input_a.bin + metadata
    input_b.bin + metadata
    expected_output.bin + metadata
  matmul_2x3_bf16/
    ...
  softmax_3x4_f32/
    ...
```

Each dataset is a named collection of tensors (inputs + expected outputs)
with shape, dtype, and raw bytes. Format TBD — likely a simple binary format
with a small metadata header, or CBOR.

#### Rust Side: Graph Construction + Multi-Mode Eval

```rust
struct TestCase {
    name: &'static str,
    graph: MilliOpGraph,
    inputs: HashMap<GlobalId, NumericTensor<...>>,
    expected: HashMap<GlobalId, NumericTensor<...>>,
    tolerance: Tolerance,  // dtype-appropriate (F32: 1e-5, BF16: looser, etc.)
}

fn build_test_set() -> Vec<TestCase> {
    vec![
        TestCase {
            name: "add_f32",
            graph: build_add_graph(NumericDType::float(FloatType::F32)),
            // loads inputs/expected from test_data/add_f32/
            ...
        },
        // ...
    ]
}
```

The same test set runs through every evaluation mode:

```rust
#[test]
fn test_via_nano_eval() {
    for case in build_test_set() {
        let result = case.graph.eval_nano(pool, &case.inputs);
        assert_approx_eq(&result, &case.expected, &case.tolerance);
    }
}

#[test]
fn test_via_compiled() {
    for case in build_test_set() {
        let result = case.graph.eval_compiled(pool, &case.inputs);
        assert_approx_eq(&result, &case.expected, &case.tolerance);
    }
}

// Future: test_via_compiled_cached, etc.
```

Adding a new eval mode means adding one test function. Adding a new op means
adding one entry to `build_test_set()`. The two dimensions are independent.

#### PyTorch Side: Independent Verification

PyTorch scripts construct equivalent operations and verify the same named
datasets. PyTorch is the source of truth for "what is the correct output."

```python
def test_add_f32():
    a = load_tensor("test_data/add_f32/input_a")
    b = load_tensor("test_data/add_f32/input_b")
    expected = load_tensor("test_data/add_f32/expected_output")
    result = a + b
    torch.testing.assert_close(result, expected)

def test_matmul_2x3_bf16():
    a = load_tensor("test_data/matmul_2x3_bf16/input_a")
    b = load_tensor("test_data/matmul_2x3_bf16/input_b")
    expected = load_tensor("test_data/matmul_2x3_bf16/expected_output")
    result = torch.matmul(a, b)
    torch.testing.assert_close(result, expected)
```

The named dataset is the contract. Rust builds graphs and evals against the
data. PyTorch builds equivalent ops and verifies the data is correct. Neither
needs to know about the other's implementation.

#### What This Catches

- **Rust-only bug:** all Rust eval modes agree with each other but differ from
  PyTorch → PyTorch tests fail → dataset is wrong or Rust has a bug.
- **Eval-mode-specific bug:** nano-op eval and compiled eval produce different
  results → cross-mode Rust tests catch it.
- **Regression:** optimized fast path produces different results from scalar
  fallback → cross-mode tests catch it.

## Migration Phases

### Migration Strategy: Gradual Replacement

`NDArrayNumericTensor` and the old `NumericTensor` enum are too load-bearing
to replace in one shot. The strategy is:

1. **Build new types alongside old ones** — no deletions until the new types
   are proven and all call sites have been migrated.
2. **Tested conversion bridges** between old and new formats — `from_ndarray()`
   and `to_ndarray()` methods allow incremental migration of call sites while
   keeping all existing tests green.
3. **Free deletions first** — rip out backends that nothing depends on (Vulkan,
   Candle, TCH) before touching the core types.
4. **Migrate call sites gradually** — convert one subsystem at a time (model
   loading → eval pipeline → server), verifying tests at each step.
5. **Delete old types last** — only after zero remaining references.

### Phase 0: Free Deletions + Foundation

Things we can do immediately that don't break anything:

- [ ] Remove Vulkan backend (`src/backends/vulkan_backend/`)
- [ ] Remove Candle backend (`src/backends/candle_backend/`)
- [ ] Remove TCH backend (`src/backends/tch_backend/`)
- [ ] Remove ONNX reference backend
- [ ] Clean up feature flags and conditional compilation for removed backends
- [ ] Define `Pool` trait with GAT `Buffer<'a>`
- [ ] Implement `SystemPool` (passthrough to global allocator)
- [ ] Implement `TrackedPool` (freelist, usage tracking, budget enforcement)
- [ ] Unit tests for pool allocation, deallocation, budget enforcement, accounting

Backend removal is safe because these are behind feature flags and not used
in the default build path. Pool types are purely additive — nothing depends
on them yet.

### Phase 1: New Type Definitions (Additive Only)

Build all new types alongside the old ones. Nothing is deleted or replaced.
Old code continues to work unchanged.

- [ ] Define `NumericDType`, `IntType`, `FloatType`, `FloatSemantics` structs
- [ ] Define `ONNXDType` enum bridging ONNX ↔ `NumericDType`
- [ ] Define `BufferLayout` with `StorageFormat` and bit-strides
- [ ] Define new `NumericScalar` (`[u8; 8]` + `NumericDType`)
- [ ] Define `NumericScalarView` / `NumericScalarViewMut`
- [ ] Implement scalar operations on `NumericScalarView` (arithmetic, cast, comparison)
  - Fast paths for known native types (f32, f64, f16, bf16, ints)
  - Software fallback for arbitrary float formats (from exponent/mantissa spec)
- [ ] Define `NumericTensorView<'a, R>` struct (byte slice + shape + layout)
- [ ] Define `NumericTensor<'a, R, P>` struct (pool buffer + shape + layout)
- [ ] Implement `read_element()` on `NumericTensorView` for all storage formats
- [ ] Implement **conversion bridges**:
  - `OldNumericTensor → new NumericTensor` (wraps existing data in SystemPool buffer)
  - `new NumericTensor → OldNumericTensor` (extracts to NDArrayNumericTensor)
  - `old DType ↔ NumericDType`
  - `old NumericScalar ↔ new NumericScalar`
- [ ] Unit tests for all new types and conversions

At this point both type systems coexist. Existing tests still pass using old
types. New types can be exercised through conversion bridges.

### Phase 2: Test Infrastructure

- [ ] Define named dataset format (binary + metadata or CBOR)
- [ ] Build dataset read/write utilities (Rust + Python)
- [ ] Port existing NumericTensor test cases to named datasets
- [ ] Create `build_test_set()` producing MilliOpGraph + dataset associations
- [ ] Create PyTorch verification scripts for all named datasets
- [ ] Verify datasets pass both Rust (current eval) and PyTorch before proceeding

This phase establishes the safety net before the destructive phases begin.
The existing NumericTensor tests continue to pass — the new test
infrastructure runs alongside them, not instead of them.

### Phase 3: Nano-Op Eval on New Types

Prove that every MilliOpGraph that currently evals through NDArrayNumericTensor
can eval flawlessly through nano-op lowering + eval using the new types.
This is the critical proof point before the bulk migration begins.

- [ ] Pivot nano-op eval internals to use new `NumericScalar` / `NumericTensorView`
- [ ] Convert at boundaries: old `NumericTensor` → new types at nano-op entry,
      new types → old `NumericTensor` at nano-op exit
- [ ] Build a MilliOp-level entry point that lowers any MilliOp to nano-ops
      and evals using the new-type nano-op path
- [ ] Run the full named dataset test suite (Phase 2) through this path
- [ ] Run existing nano-graph lowering tests (`check_integrity()`) through this path
- [ ] Identify and fix any ops that don't lower/eval correctly
- [ ] Verify all 74 existing lowering tests + all named dataset tests pass

At this point: the new types are proven correct for all supported ops, nano-op
eval is the validated fallback path, and we have confidence to start replacing
call sites. The old types still exist and the rest of the codebase is untouched.

### Phase 4: Gradual Call Site Migration

Migrate subsystems one at a time from old types to new types. Conversion
bridges allow partially-migrated code to interoperate. Tests stay green
throughout.

**3a: Nano-op eval**
- [ ] Adapt `populate_from_tensor()` to accept `NumericTensorView`
- [ ] Adapt nano-op eval output to construct pool-backed `NumericTensor`
- [ ] Add pool parameter to nano-op eval entry point
- [ ] Verify via test infrastructure (nano-op eval mode)

**3b: MilliOp layer**
- [ ] Add pool parameter to `MilliOp::eval()` and `MilliOp::infer()` signatures
- [ ] Update `TensorInfo` to use new tensor types (concrete values use `SystemPool`)
- [ ] Migrate `Constant` op data to `NumericTensor<'static, DynRank, SystemPool>`
- [ ] Wire `MilliOp::eval()` to lower-to-nano and eval on pool-backed buffers
- [ ] Migrate milli_graph eval loop to use `NumericTensorView` for op inputs
- [ ] Conversion bridges at subsystem boundaries (old NumericTensor ↔ new)

**3c: Model loading**
- [ ] Update tensor_store to produce new `NumericTensor` types
- [ ] Update GGUF loader to produce new types
- [ ] Update safetensors loader (can use mmap → `NumericTensorView` directly)
- [ ] Update ONNX import path to produce `ONNXDType` → `NumericDType`
- [ ] Subsume `PackedTensor` into `NumericTensor` (block-quantized = StorageFormat)

**3d: Symbolic graph + SuperGraph**
- [ ] Add pool parameter to `Operation::eval()` trait
- [ ] Rework `eval_backend::run()` — simplifies significantly with single backend
- [ ] Thread pool through `SuperGraph::eval()` and `SuperGraphContext`
- [ ] Migrate `ModelLoadedTensorCache` to new types
- [ ] Eliminate tensor cloning in eval paths (reference-based ownership)

Each sub-phase can be a separate PR/commit. Conversion bridges handle the
boundaries between migrated and unmigrated code.

### Phase 5: Old Type Deletion

Only after all call sites are migrated and tests pass:

- [ ] Remove conversion bridges (no longer needed)
- [ ] Delete `NDArrayNumericTensor` and its ~2500 lines of operation implementations
- [ ] Delete old `NumericTensor` enum
- [ ] Delete old `DType` enum
- [ ] Delete old `NumericScalar` enum
- [ ] Delete old `PackedTensor` type
- [ ] Remove ndarray as core dependency
- [ ] Migrate `DTypeOfPrimitive` trait to new type system (or remove)
- [ ] Remove dead imports, unused conversions, orphaned test helpers

### Phase 6: Server Integration

- [ ] Create server-level pool with configurable budget
- [ ] Create per-model pools on model load, destroy on model unload
- [ ] Create per-request pools in scheduler, move into `spawn_blocking` closure
- [ ] Add pool usage reporting to scheduler metrics / observer
- [ ] Handle allocation failure gracefully (refuse request, not abort)

### Phase 7: Compiler Maturation

- [ ] Mature the compilation backend (already operates on MilliOpGraphs)
- [ ] Implement compilation caching — same graph shape → reuse compiled code
- [ ] Improve codegen quality (vectorization, fused ops, BLAS calls from codegen)
- [ ] Profile and optimize pool allocator (slab sizes, fragmentation)

No separate "BLAS dispatch" or "hand-optimized op" layer. The compiler already
takes MilliOpGraphs and produces optimized code. Performance recovery comes from
maturing that path, not from bolting on bespoke optimizations that would be
thrown away when the compiler catches up. The test infrastructure (Phases 2-3)
validates that compiled eval matches nano-op eval across all test cases.

## Key Risks and Open Questions

1. **GAT ergonomics.** GATs are stable since Rust 1.65, but some patterns
   (returning `Buffer<'_>` from trait methods, storing in structs) can hit
   lifetime inference limitations. May need explicit lifetime annotations in
   more places than expected.

2. **Alignment guarantees.** Pool allocations must be aligned to at least 16
   bytes (covers all primitive types up to f128). The typed view path requires
   correct alignment.

3. **STRING dtype.** Moves to `ONNXDType::String`, outside the numeric type
   system entirely. STRING tensors are rare in inference (mostly ONNX metadata).

4. **Serialization.** Resolved — see "Serialization Model" section.

5. **v1 performance.** Scalar nano-op eval is significantly slower than
   optimized code. This is acceptable for proving the architecture — Phase 6
   matures the compiler for performance recovery. Models used for testing
   during migration should be small enough that nano-op eval is tolerable.

6. **Block quantization in BufferLayout.** How exactly block-quantized formats
   (with per-block scales/offsets) are represented in `BufferLayout` needs
   design. May keep `PackedFormat` as a layout variant initially.

7. **MilliOpGraph construction ergonomics.** If MilliOpGraph is the only
   interface for tensor ops, building small ad-hoc graphs needs to be low-
   friction. May want builder helpers or convenience functions that construct
   and immediately eval single-op graphs.

8. **Compilation caching.** Repeated eval of the same MilliOpGraph shape
   should reuse compiled code. Cache keying (graph topology + dtypes + shapes)
   and invalidation need design. The server already has some caching
   infrastructure (`SuperGraphCache`) that may be adaptable.
