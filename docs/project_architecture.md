# Whisper Tensor - Project Architecture

## Overview

Whisper Tensor is a **correctness-first machine learning runtime** written in Rust. It serves as a reliable oracle for model execution with transparent intermediate representations (IRs), precise dtype/shape semantics, and comprehensive introspection tooling.

### Design Philosophy

- **Transparency-first**: All computation graphs are fully inspectable and debuggable
- **Correctness over performance**: Prioritizes correctness as the foundation, with performance as a secondary goal
- **ONNX-native**: Uses ONNX as the canonical input format
- **Cross-platform execution**: Supports multiple backends with per-op fallback capabilities

### Supported Models

- GPT-2 (transformers)
- RWKV (RNN-based language models)
- LLaMA variants (via import pipeline)
- General ONNX models (400+ conformance tests passing)

---

## Workspace Structure

```
whisper-tensor/
├── src/                         # Core library (whisper-tensor crate)
├── crates/
│   ├── whisper-tensor-import/   # Model loading & ONNX conversion
│   ├── whisper-tensor-server/   # Inference server & WebSocket API
│   └── whisper-tensor-webui/    # WASM-based graph explorer
└── docs/
```

---

## Core Crates

### `whisper-tensor` (Main Library)

The central computation engine containing the graph representations and execution logic.

**Key Modules:**

| Module | Responsibility |
|--------|----------------|
| `symbolic_graph/` | ONNX ingestion and symbolic computation graph |
| `milli_graph/` | Lower-level simplified operation graph |
| `super_graph/` | High-level job interfaces for model usage patterns |
| `compiler/` | Compilation from symbolic graphs to executable programs |
| `backends/` | Pluggable execution backends (NDArray, Vulkan, etc.) |
| `dtype/` | Comprehensive dtype system (F64, F32, BF16, F16, integers, etc.) |
| `tensor_info/` | Tensor metadata, shape, and type information |
| `numeric_tensor/` | Runtime tensor representation with multi-backend support |
| `graph/` | Core graph abstraction traits (Link, Node, Graph) |
| `symbolic_scalar/` | Symbolic dimension/value handling for dynamic shapes |
| `interfaces/` | High-level APIs for specific use cases |

### `whisper-tensor-import`

Converts raw model weights and architectures into canonical ONNX format.

**Capabilities:**
- Load `.safetensors`, `.pth`, and HuggingFace directory models
- Architecture-specific converters (GPT-2, LLaMA, RWKV)
- Weight storage strategies: Embedded, Binary file, or Origin reference
- Metadata injection for tokenizer info and I/O descriptions

### `whisper-tensor-server`

WebSocket-based inference server for model execution and WebUI communication.

**Features:**
- Model lifecycle management (load/unload/compile)
- SuperGraph execution via WebSocket protocol
- Tensor introspection and streaming
- HTTP server for WebUI static assets

### `whisper-tensor-webui`

Interactive graph visualization compiled to WebAssembly.

**Features:**
- Real-time graph navigation and visualization
- Tensor value inspection during execution
- Model architecture exploration
- LLM-specific inference interface

---

## Graph Layers

Whisper Tensor uses a layered graph architecture for progressive lowering and optimization:

```
┌─────────────────────────────────────────────────────┐
│  SuperGraph                                         │
│  High-level job patterns (tokenize → infer → decode)│
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  SymbolicGraph                                      │
│  Full ONNX semantics (60+ ops, symbolic dimensions) │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  MilliOpGraph                                       │
│  Simplified primitives (40+ ops, shape inference)   │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  Backend Execution                                  │
│  NDArray (CPU) / Vulkan (GPU) / Candle / TCH       │
└─────────────────────────────────────────────────────┘
```

### SymbolicGraph

The ONNX-level representation preserving full operator semantics.

- Supports opsets 10-23
- Mixed symbolic + numeric dimensions
- TensorStore for weight management
- 60+ ONNX operations

### MilliOpGraph

A simplified intermediate representation with decomposed operations.

- Complex ONNX ops decomposed into primitives
- Pre-execution shape inference
- Constant folding and dead code elimination
- ~40 simplified operations (matmul, unary, binary, reshape, slice, etc.)

### SuperGraph

High-level job adapters for specific usage patterns.

**Node Types:**
- `ModelExecution` - Invoke compiled models
- `MilliOpGraph` - Direct milli-op execution
- `TokenizerEncode/Decode` - Text processing
- `RNNCacheRead/Write` - State management for RNNs
- `Scan` - Sequence operations

**Link Types:**
- `Tensor` - Numeric tensor data
- `String` - Text data
- `Hash` - Cache keys
- `Tokenizer` - Tokenizer instances
- `TensorMap` - Model weight maps

---

## Execution Backends

### NDArray (CPU Reference)

- Primary reference implementation using `ndarray` crate
- Multi-threaded via rayon
- Full operation coverage
- Used for correctness validation

### Vulkan (GPU Compute)

- Production GPU backend via `vulkano`
- SPIR-V shader compilation
- Per-op fallback to NDArray for unimplemented kernels
- Currently supports ~20 operations (expanding)

### Optional Backends

- **Candle** - Hugging Face's lightweight ML runtime
- **TCH** - PyTorch bindings via libtorch

---

## Key Abstractions

### GlobalId System

Every graph element (nodes, links, tensors) has a unique `GlobalId` (random u64). This enables:
- Deterministic serialization
- Cross-layer element tracking
- WebUI introspection

### Graph Traits

```rust
pub trait Link { fn global_id(&self) -> GlobalId; }

pub trait Node {
    type OpKind;
    fn global_id(&self) -> GlobalId;
    fn op_kind(&self) -> OpKind;
    fn inputs(&self) -> impl Iterator<Item = GlobalId>;
    fn outputs(&self) -> impl Iterator<Item = GlobalId>;
}

pub trait Graph {
    type AnyNode;
    type AnyLink;
    fn global_id(&self) -> GlobalId;
}
```

### Rank Type System

Compile-time and runtime rank handling:

```rust
pub trait Rank {
    type NDArrayDim: Dimension;
    type UnknownDims;  // symbolic dimensions
    type KnownDims;    // concrete dimensions
}
```

Supports both static ranks (`Rank0` through `Rank6`) and dynamic (`DynRank`).

### DType System

```rust
pub enum DType {
    F64, F32, BF16, F16,
    I64, I32, I16, I8,
    U64, U32, U16, U8,
    BOOL, STRING
}
```

Backend-agnostic with conversion bridges to all supported backends.

### NumericTensor

Backend-agnostic tensor wrapper:

```rust
pub enum NumericTensor<R: Rank> {
    NDArray(NDArrayNumericTensor<R>),
    Vulkan(VulkanNumericTensor<R>),
    Candle(CandleNumericTensor<R>),
    TCH(TCHNumericTensor<R>),
}
```

---

## Data Flow

### Model Loading Pipeline

```
Model File (.onnx, .safetensors, directory)
         │
         ▼ (whisper-tensor-import)
    ONNX ModelProto
         │
         ▼ Model::new_from_onnx
    SymbolicGraph + TensorStore
         │
         ▼ (optional) Compiler::build_program
    CompiledProgram
```

### Server Execution Flow

```
WebUI (Browser/WASM)
         │
         ▼ WebSocket
    Server (Tokio async)
         │
         ├─ Model Cache
         ├─ Compiled Programs
         │
         ▼
    SuperGraph Execution
         │
         ▼
    Backend Dispatch (NDArray/Vulkan)
         │
         ▼
    Tensor Results → WebSocket → WebUI
```

---

## Observer Pattern

Execution tracing and introspection via callbacks:

```rust
pub trait SymbolicGraphObserver {
    fn on_op_executed(&mut self, node_path: &str, start: Instant, end: Instant, backend: &str);
    fn on_tensor_assigned(&mut self, tensor_path: &str, tensor: &NumericTensor, backend: &str);
}
```

Enables the WebUI to display real-time execution progress and tensor values.

---

## External Dependencies

| Category | Dependencies |
|----------|-------------|
| GPU Compute | vulkano (Vulkan 1.x) |
| ML Frameworks | candle (optional), tch (optional) |
| Tensor Ops | ndarray, half (f16/bf16) |
| Serialization | prost (protobuf), serde, ciborium (CBOR) |
| Model Formats | ONNX, SafeTensors |
| Async Runtime | tokio (server only) |
| Web Framework | axum (HTTP/WebSocket) |
| GUI | egui, eframe (WASM) |
| Observability | tracing |

---

## Current Status

**Working:**
- Full end-to-end GPT-2 and RWKV execution
- NDArray and Vulkan backends with cross-backend fallback
- 400+ ONNX conformance tests passing
- WebUI for graph exploration and tensor introspection
- Model compilation and caching
- Mixed symbolic/numeric dimension handling
- Low-precision dtype support (F16, BF16)

**In Development:**
- Expanded Vulkan operation coverage
- Additional backend integrations
- Advanced graph optimizations and operator fusion
