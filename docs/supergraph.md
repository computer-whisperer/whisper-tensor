# SuperGraph Subsystem

The SuperGraph is a high-level job orchestration layer in Whisper Tensor that abstracts over the low-level graph representations (SymbolicGraph and MilliOpGraph) to provide practical, domain-specific execution patterns for LLM inference and model composition.

## Purpose and Design Goals

The SuperGraph serves several key purposes:

1. **Abstraction Bridging**: Bridges the semantic gap between low-level tensor operations and high-level use cases like LLM inference pipelines
2. **Heterogeneous Computation**: Supports mixing different operation types (model execution, tokenization, caching, loops) in a unified graph
3. **State Management**: Manages stateful operations like RNN caches that need to persist across inference steps
4. **Practical Interfaces**: Provides concrete interfaces for common patterns (e.g., `TextInferenceTokensInLogitOutInterface`)
5. **Multi-Backend Support**: Enables seamless switching between direct interpretation and compiled execution

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         SuperGraph                              │
├─────────────────────────────────────────────────────────────────┤
│  input_links: HashSet<SuperGraphAnyLink>    (graph inputs)      │
│  output_links: HashSet<SuperGraphAnyLink>   (graph outputs)     │
├─────────────────────────────────────────────────────────────────┤
│  nodes: HashMap<GlobalId, SuperGraphAnyNode>                    │
│  links_by_global_id: HashMap<GlobalId, SuperGraphAnyLink>       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SuperGraphData                             │
├─────────────────────────────────────────────────────────────────┤
│  tensors: HashMap<SuperGraphLinkTensor, NumericTensor>          │
│  strings: HashMap<SuperGraphLinkString, String>                 │
│  tokenizers: HashMap<SuperGraphLinkTokenizer, AnyTokenizer>     │
│  tensor_maps: HashMap<SuperGraphLinkTensorMap, &TensorStore>    │
│  hashes: HashMap<SuperGraphLinkHash, SuperGraphHash>            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SuperGraphContext                           │
├─────────────────────────────────────────────────────────────────┤
│  symbolic_graphs: Vec<&SymbolicGraph>   (model graphs)          │
│  compiled_models: Option<Vec<(Model, CompiledProgram)>>         │
│  eval_backend: &mut EvalBackend         (computation backend)   │
│  caches: Option<&mut SuperGraphCache>   (RNN state cache)       │
│  super_graph_tensor_cache: &mut SuperGraphTensorCache           │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

- **Polymorphic Links**: Five distinct link types support different data flowing through the graph (tensors, strings, tokenizers, tensor maps, hashes)
- **Polymorphic Nodes**: Eight node types cover model execution, tokenization, caching, and control flow
- **Data-Driven Execution**: Greedy scheduling executes any node whose inputs are ready
- **Compositional Nesting**: Scan nodes contain nested SuperGraphs for loop execution

---

## Core Data Structures

### SuperGraph

The main graph container, located in `src/super_graph/mod.rs`:

```rust
pub struct SuperGraph {
    global_id: GlobalId,                                // Unique graph identifier
    pub input_links: HashSet<SuperGraphAnyLink>,        // External input points
    pub output_links: HashSet<SuperGraphAnyLink>,       // External output points
    pub nodes: HashMap<GlobalId, SuperGraphAnyNode>,    // All computation nodes
    pub links_by_global_id: HashMap<GlobalId, SuperGraphAnyLink>, // Index for lookups
}
```

### SuperGraphAnyLink

Polymorphic link types supporting heterogeneous data flow:

```rust
pub enum SuperGraphAnyLink {
    Tensor(SuperGraphLinkTensor),      // Numeric tensor data
    String(SuperGraphLinkString),      // Text data
    TensorMap(SuperGraphLinkTensorMap), // Model weight maps
    Tokenizer(SuperGraphLinkTokenizer), // Tokenizer instances
    Hash(SuperGraphLinkHash),          // Cache keys (u64)
}
```

Each link type is a newtype wrapper around `GlobalId`:

```rust
pub struct SuperGraphLinkTensor(GlobalId);
pub struct SuperGraphLinkString(GlobalId);
pub struct SuperGraphLinkTensorMap(GlobalId);
pub struct SuperGraphLinkTokenizer(GlobalId);
pub struct SuperGraphLinkHash(GlobalId);
```

### Link Wrappers

For operations that transform links:

```rust
// Maps input → output (simple pass-through or transformation)
pub struct SuperGraphLinkDouble {
    pub from: SuperGraphAnyLink,
    pub to: SuperGraphAnyLink,
}

// Maps initial state → input → output (for loops with state)
pub struct SuperGraphLinkTriple {
    pub initial_value: SuperGraphAnyLink,  // Initial state before loop
    pub input: SuperGraphAnyLink,          // State input to loop body
    pub output: SuperGraphAnyLink,         // State output from loop body
}
```

### SuperGraphAnyNode

Eight node variants for different operation categories:

```rust
pub enum SuperGraphAnyNode {
    // Model Execution
    ModelExecution(SuperGraphNodeModelExecution),  // Execute SymbolicGraph
    MilliOpGraph(SuperGraphNodeMilliOpGraph),      // Execute MilliOpGraph directly
    
    // Tokenization
    TokenizerLoad(SuperGraphNodeTokenizerLoad),    // Load tokenizer instance
    TokenizerEncode(SuperGraphNodeTokenizerEncode), // Text → tokens
    TokenizerDecode(SuperGraphNodeTokenizerDecode), // Tokens → text
    
    // Control Flow
    Scan(SuperGraphNodeScan),                      // Loop over sequence
    
    // Caching
    RNNCacheRead(SuperGraphNodeRNNCacheRead),      // Lookup cache state
    RNNCacheWrite(SuperGraphNodeRNNCacheWrite),    // Store cache state
}
```

### SuperGraphData

Runtime container for all data flowing through execution:

```rust
pub struct SuperGraphData<'models> {
    pub tensors: HashMap<SuperGraphLinkTensor, NumericTensor<DynRank>>,
    pub strings: HashMap<SuperGraphLinkString, String>,
    pub tokenizers: HashMap<SuperGraphLinkTokenizer, AnyTokenizer>,
    pub tensor_maps: HashMap<SuperGraphLinkTensorMap, &'models TensorStore>,
    pub hashes: HashMap<SuperGraphLinkHash, SuperGraphHash>,
}
```

### SuperGraphContext

Bundles all runtime resources needed during execution:

```rust
pub struct SuperGraphContext<'short, 'model, 'c, 'd, T: SuperGraphObserver> {
    pub observer: &'short mut T,                    // Execution observer
    pub eval_backend: &'c mut EvalBackend<'d>,      // Computation backend
    pub caches: Option<&'short mut SuperGraphCache>, // RNN state cache
    pub super_graph_tensor_cache: &'short mut SuperGraphTensorCache<'model>,
    pub use_compiled_models: bool,                  // Use compiled or interpreted
    pub symbolic_graphs: Vec<&'model SymbolicGraph>, // Referenced symbolic graphs
    pub compiled_models: Option<Vec<(&'model Model, &'short CompiledProgram)>>,
}
```

### SuperGraphBuilder

Mutable builder for graph construction:

```rust
pub struct SuperGraphBuilder {
    nodes: HashMap<GlobalId, SuperGraphAnyNode>,
}
```

**Builder Methods:**

| Method | Description |
|--------|-------------|
| `add_node(node)` | Add a node, returns its GlobalId |
| `new_tensor_link(rng)` | Create a new tensor link |
| `new_model_link(rng)` | Create a new tensor map link |
| `new_tokenizer_link(rng)` | Create a new tokenizer link |
| `new_string_link(rng)` | Create a new string link |
| `new_hash_link(rng)` | Create a new hash link |
| `build(rng, inputs, outputs)` | Validate and produce immutable SuperGraph |

---

## Node Types

### SuperGraphNodeModelExecution

Executes a SymbolicGraph (the primary model execution node):

```rust
pub struct SuperGraphNodeModelExecution {
    global_id: GlobalId,
    tensor_map: SuperGraphLinkTensorMap,              // Model weights
    symbolic_graph_id: usize,                         // Index into context.symbolic_graphs
    tensor_inputs: Vec<(SuperGraphLinkTensor, String)>, // (link, ONNX input name)
    tensor_outputs: Vec<(String, SuperGraphLinkTensor)>, // (ONNX output name, link)
}
```

**Execution Behavior:**
- Maps SuperGraph tensor links to ONNX tensor names
- Executes either `SymbolicGraph.eval()` or `CompiledProgram.run()` based on context flags
- Maps outputs back to SuperGraph tensor links

### SuperGraphNodeMilliOpGraph

Executes a MilliOpGraph directly (for tensor manipulation):

```rust
pub struct SuperGraphNodeMilliOpGraph {
    global_id: GlobalId,
    milli_op_graph: MilliOpGraph,
    pub inputs: Vec<SuperGraphLinkDouble>,   // Input mappings
    pub outputs: Vec<SuperGraphLinkDouble>,  // Output mappings
}
```

**Use Cases:**
- Post-processing (slicing logits, reshaping)
- Pre-processing (shape calculations)
- Custom tensor operations not requiring full model execution

### SuperGraphNodeTokenizerLoad

Loads a tokenizer instance:

```rust
pub struct SuperGraphNodeTokenizerLoad {
    global_id: GlobalId,
    tokenizer_path: PathBuf,
    output: SuperGraphLinkTokenizer,
}
```

### SuperGraphNodeTokenizerEncode

Encodes text to tokens:

```rust
pub struct SuperGraphNodeTokenizerEncode {
    global_id: GlobalId,
    tokenizer: SuperGraphLinkTokenizer,
    text: SuperGraphLinkString,
    output: SuperGraphLinkTensor,
}
```

### SuperGraphNodeTokenizerDecode

Decodes tokens to text:

```rust
pub struct SuperGraphNodeTokenizerDecode {
    global_id: GlobalId,
    tokenizer: SuperGraphLinkTokenizer,
    tokens: SuperGraphLinkTensor,
    output: SuperGraphLinkString,
}
```

### SuperGraphNodeScan

Implements ONNX Scan semantics (loop over sequence):

```rust
pub struct SuperGraphNodeScan {
    global_id: GlobalId,
    inner_graph: SuperGraph,                               // Loop body graph
    iteration_count: SuperGraphLinkTensor,                 // Number of iterations
    simple_inputs: Vec<SuperGraphLinkDouble>,              // Pass-through inputs
    state_links: Vec<SuperGraphLinkTriple>,                // State flowing through loop
    scan_inputs: Vec<(SuperGraphLinkTensor, SuperGraphLinkTensor, u32)>,  // (outer, inner, axis)
    scan_outputs: Vec<(SuperGraphLinkTensor, SuperGraphLinkTensor, u32)>, // (inner, outer, axis)
}
```

**Loop Execution:**
1. For each iteration i in 0..iteration_count:
   - Slice scan_inputs at index i along specified axis
   - Execute inner_graph with sliced inputs + current state
   - Accumulate scan_outputs
   - Update state for next iteration
2. Concatenate accumulated outputs along scan axis

### SuperGraphNodeRNNCacheRead

Looks up cached RNN state:

```rust
pub struct SuperGraphNodeRNNCacheRead {
    global_id: GlobalId,
    hash_input: SuperGraphLinkHash,                      // Cache key
    tokens_input: SuperGraphLinkTensor,                  // Token sequence
    tokens_output: SuperGraphLinkTensor,                 // Remaining tokens after cache hit
    state_outputs: Vec<SuperGraphLinkDouble>,            // Retrieved state tensors
    default_state_inputs: Vec<SuperGraphLinkDouble>,     // Default state if cache miss
}
```

**Cache Lookup Algorithm:**
```
For i = len(tokens) down to 1:
    if cache[key][tokens[0:i]] exists:
        return (tokens[i:], cached_state)
return (tokens, default_state)
```

### SuperGraphNodeRNNCacheWrite

Stores RNN state in cache:

```rust
pub struct SuperGraphNodeRNNCacheWrite {
    global_id: GlobalId,
    hash_input: SuperGraphLinkHash,         // Cache key
    tokens_input: SuperGraphLinkTensor,     // Full token sequence
    state_inputs: Vec<SuperGraphLinkDouble>, // State to cache
}
```

---

## Graph Execution

### Execution Algorithm

```rust
pub fn eval<T: SuperGraphObserver>(
    &self,
    data: &mut SuperGraphData,
    context: &mut SuperGraphContext<T>,
) -> Result<(), SuperGraphEvalError>
```

**Algorithm (Data-Driven Scheduling):**

```
remaining_ops = all_nodes
data = input_data

loop:
    for each op in remaining_ops:
        if all_inputs_ready(op, data):
            op.eval(data, context)
            remove op from remaining_ops
            break  // Restart loop to find next ready op
    if no op executed:
        break  // All operations complete
```

**Characteristics:**
- Greedy data-driven scheduling
- No explicit topological sort (relies on builder validation)
- Sequential execution (individual ops may parallelize internally via EvalBackend)

### Graph Validation (SuperGraphBuilder::build)

1. Validate all output links are sourced exactly once
2. Validate all internal links (node outputs) are sourced exactly once
3. Validate all internal sinks (node inputs) have sources
4. Validate input links are sourced once
5. Build link index for fast lookup

---

## Observer Pattern

The `SuperGraphObserver` trait enables execution monitoring:

```rust
pub trait SuperGraphObserver {
    fn on_node_executed(
        &mut self,
        node_path: &[GlobalId],
        start_instant: Instant,
        end_instant: Instant,
    );
    
    fn on_tensor_assigned(
        &mut self,
        tensor_link: &SuperGraphLinkTensor,
        tensor: &NumericTensor<DynRank>,
    );
    
    fn on_string_assigned(
        &mut self,
        string_link: &SuperGraphLinkString,
        value: &str,
    );
}

// No-op implementation for when observation is not needed
impl SuperGraphObserver for () {
    fn on_node_executed(...) {}
    fn on_tensor_assigned(...) {}
    fn on_string_assigned(...) {}
}
```

---

## Caching System

### SuperGraphCache

Stores RNN state keyed by hash and token sequence:

```rust
pub struct SuperGraphCache {
    cache: HashMap<u64, HashMap<Vec<u32>, HashMap<String, NumericTensor<DynRank>>>>,
}
```

**Structure:** `cache[hash_key][token_sequence][state_name] → tensor`

**Purpose:** For RNN models, caches intermediate states so subsequent inference on extended prompts can resume from cached state rather than recomputing.

### SuperGraphTensorCache

Caches loaded model weights:

```rust
pub struct SuperGraphTensorCache<'model> {
    tensors: HashMap<TensorStoreTensorId, NumericTensor<DynRank>>,
    tensor_store: &'model TensorStore,
}
```

---

## High-Level Interfaces

### TextInferenceTokensInLogitOutInterface

The primary interface for LLM text inference:

```rust
pub struct TextInferenceTokensInLogitOutInterface {
    pub super_graph: SuperGraph,
    pub tensor_map_link: SuperGraphLinkTensorMap,
    pub token_context_link: SuperGraphLinkTensor,
    pub logit_output_link: SuperGraphLinkTensor,
    pub hash_link: SuperGraphLinkHash,
}
```

**Construction:** `TextInferenceTokensInLogitOutInterface::try_from_onnx_metadata()` introspects the SymbolicGraph to:
- Identify input/output tensor names and types
- Extract metadata (tokenizer info, batch info)
- Build preprocessing/postprocessing MilliOpGraphs
- Handle RNN state chains automatically

**Usage:**

```rust
let interface = TextInferenceTokensInLogitOutInterface::try_from_onnx_metadata(
    &model.model_metadata,
    &model.model_inputs,
    &model.model_outputs,
    &model.graph,
    &mut rng,
)?;

let response = interface.run_string_in_string_out(
    &model,
    compiled_model.as_ref(),
    "input text",
    &mut tokenizer_cache,
    &mut tensor_cache,
    &mut super_graph_caches,
    &mut backend,
)?;
```

---

## Integration with Compilation Pipeline

### Pipeline Architecture

```
ONNX File
    │
    ▼
Model::new_from_onnx()
    ├── SymbolicGraphMutator::from_onnx_bytes()
    ├── Creates SymbolicGraph (ONNX IR)
    └── Creates TensorStore (weights)
    │
    ▼
TextInferenceTokensInLogitOutInterface::try_from_onnx_metadata()
    │   Constructs SuperGraph for text inference pattern
    │
    ▼
SuperGraph
    ├── SuperGraphNodeTokenizerLoad
    ├── SuperGraphNodeTokenizerEncode
    ├── SuperGraphNodeRNNCacheRead (RNN mode)
    ├── SuperGraphNodeScan (RNN mode)
    │   └── SuperGraphNodeModelExecution
    │       └── Uses SymbolicGraph or CompiledProgram
    │           └── Internally generates MilliOpGraph
    ├── SuperGraphNodeRNNCacheWrite (RNN mode)
    └── SuperGraphNodeTokenizerDecode
```

### Integration Points

| Component | Integration |
|-----------|-------------|
| **SymbolicGraph** | Referenced by `SuperGraphNodeModelExecution`, executed via `eval()` |
| **MilliOpGraph** | Wrapped by `SuperGraphNodeMilliOpGraph`, used internally by SymbolicGraph |
| **TensorStore** | Passed via `SuperGraphLinkTensorMap`, provides model weights |
| **CompiledProgram** | Alternative execution path when `use_compiled_models` is true |
| **EvalBackend** | Passed through context, performs actual computation |

---

## Example: Simple Text Inference Graph

```
SuperGraph {
  nodes: [
    SuperGraphNodeTokenizerLoad
      output: tokenizer_link
    
    SuperGraphNodeTokenizerEncode
      inputs: [tokenizer_link, text_input_link]
      output: tokens
    
    SuperGraphNodeModelExecution
      inputs: [tokens, model_link]
      output: logits
    
    SuperGraphNodeMilliOpGraph (sampling)
      inputs: [logits]
      output: chosen_token
    
    SuperGraphNodeTokenizerDecode
      inputs: [tokenizer_link, chosen_token]
      output: text_output
  ],
  
  input_links: {model_link, text_input_link},
  output_links: {text_output}
}
```

**Data Flow:**
```
Model weights → model_link ──────────────────────────────→ ModelExecution
Input text → text_input_link → TokenizerEncode → tokens → ModelExecution → logits → Sampling → token → TokenizerDecode → output text
                                     ↑                                                                        ↑
                            TokenizerLoad → tokenizer_link ──────────────────────────────────────────────────┘
```

---

## Example: RNN-Mode Inference with Caching

```
SuperGraph {
  nodes: [
    SuperGraphNodeRNNCacheRead
      inputs: [cache_key, token_context]
      outputs: [post_cache_tokens, state_outputs[]]
    
    SuperGraphNodeMilliOpGraph (shape)
      inputs: [post_cache_tokens]
      output: loop_count
    
    SuperGraphNodeScan {
      inner_graph: SuperGraph {
        nodes: [
          SuperGraphNodeMilliOpGraph (input processing),
          SuperGraphNodeModelExecution,
          SuperGraphNodeMilliOpGraph (output processing)
        ]
      },
      inputs: [model_link, post_cache_tokens, state_inputs[]]
      outputs: [logit_output, state_outputs[]]
    }
    
    SuperGraphNodeRNNCacheWrite
      inputs: [cache_key, token_context, state_inputs[]]
  ],
  
  input_links: {cache_key, model_link, token_context},
  output_links: {logit_output}
}
```

---

## Error Handling

```rust
pub enum SuperGraphEvalError {
    MilliOpGraphError(MilliOpGraphError),
    SymbolicGraphError(EvalError),
    TokenizerError(String),
    CacheMiss,
    MissingInput(GlobalId),
    // ...
}
```

---

## Serialization

All core types support serde serialization:

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperGraph { ... }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SuperGraphAnyNode { ... }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SuperGraphAnyLink { ... }
```

This enables:
- Saving/loading graphs to disk
- Transmitting graphs over network (server/client)
- Caching compiled graph structures

---

## File Structure

```
src/super_graph/
├── mod.rs          # SuperGraph struct, builder, execution algorithm
├── nodes.rs        # All 8 node types
├── links.rs        # 5 link types and double/triple wrappers
├── data.rs         # SuperGraphData container
├── cache.rs        # SuperGraphCache and SuperGraphTensorCache
└── observer.rs     # SuperGraphObserver trait

src/
└── interfaces.rs   # TextInferenceTokensInLogitOutInterface
```

Related files:
```
examples/
├── super_graph_test.rs    # Simple end-to-end example
├── super_graph_test_2.rs  # High-level interface usage
└── super_graph_test_3.rs  # Vulkan backend example

crates/whisper-tensor-server/src/
├── scheduler.rs    # Server-side execution with reporting
└── lib.rs          # SuperGraphRequest/Response types
```

---

## Summary

The SuperGraph subsystem provides:

- **Semantic Bridging**: Connects low-level tensor operations to high-level inference patterns
- **Heterogeneous Computation**: Unified graph for tensors, text, tokenizers, and caching operations
- **Compositional Execution**: Nested graphs via Scan nodes for loop execution
- **State Management**: RNN caching for efficient token-by-token inference
- **Multi-Backend Support**: Seamless switching between interpreted and compiled execution
- **Full Observability**: Observer pattern for execution monitoring and debugging
- **Serialization**: Complete serde support for graph persistence and transport

The SuperGraph represents the "application layer" of Whisper Tensor's IR hierarchy - high enough to express practical inference patterns, but low enough to maintain direct control over computation and enable efficient execution.
