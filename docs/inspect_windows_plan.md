# Inspect Windows Improvement Plan

## Current State Analysis

### What the Old Version Had (pre-global_id)

**Operation Inspect Windows:**
- ONNX name, op type, execution duration
- Interactive input/output tensor grids with dtype, shape
- Hover highlighting back to graph
- Inline constant value display

**Tensor Inspect Windows:**
- Tensor metadata (name, dtype, shape, category)
- Stored tensor fetching with loading state
- Live tensor subscriptions
- Dual tensor views (stored vs subscribed)

### What the New Version Has (global_id)

**Node Inspect Windows:**
- Basic operation type and label
- GlobalId display
- Execution duration
- Simple collapsible inputs/outputs (just IDs)
- Subgraph navigation

**Link Inspect Windows:**
- Label and ID
- Type-specific info (dtype/shape for SymbolicGraph, link type for SuperGraph)
- Subscribe/unsubscribe
- Stored value fetching
- Tensor viewing

---

## Complete Window Specifications

### Node Inspect Window - Full Design

```
┌─────────────────────────────────────────────────────────────────┐
│ LayerNorm: layer_norm_1                                     [X] │
├─────────────────────────────────────────────────────────────────┤
│ HEADER SECTION                                                  │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ encoder → block_0 → layer_norm_1                            │ │
│ │ Last Execution: 0.032ms                                     │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ OPERATION PARAMETERS (collapsible, for ops with params)         │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ epsilon: 1e-5                                               │ │
│ │ axis:    -1                                                 │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ INPUTS (3)                                                      │
│ ┌────┬────────────────────┬─────────┬──────────────┬──────────┐ │
│ │ #  │ Name               │ DType   │ Shape        │          │ │
│ ├────┼────────────────────┼─────────┼──────────────┼──────────┤ │
│ │ 0  │ hidden_states      │ f32     │ (1, 128, 768)│  [Open]  │ │
│ │ 1  │ gamma              │ f32     │ (768,)       │  [Open]  │ │
│ │ 2  │ beta               │ f32     │ (768,)       │  [Open]  │ │
│ └────┴────────────────────┴─────────┴──────────────┴──────────┘ │
│ (hover row = highlight in graph, click = select, dbl-click/Open │
│  = open link inspect window)                                    │
│                                                                 │
│ OUTPUTS (1)                                                     │
│ ┌────┬────────────────────┬─────────┬──────────────┬──────────┐ │
│ │ #  │ Name               │ DType   │ Shape        │          │ │
│ ├────┼────────────────────┼─────────┼──────────────┼──────────┤ │
│ │ 0  │ normalized_output  │ f32     │ (1, 128, 768)│  [Open]  │ │
│ └────┴────────────────────┴─────────┴──────────────┴──────────┘ │
│                                                                 │
│ SUBGRAPH (if applicable)                                        │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Contains subgraph with 24 operations                        │ │
│ │                                    [Navigate to Subgraph]   │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ CONSTANT VALUE (only for Constant nodes)                        │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ [Full tensor_view widget here]                              │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ ▶ Debug Info (collapsed by default)                             │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ GlobalId:  147                                              │ │
│ │ Full Path: [1, 42, 147]                                     │ │
│ │ [Copy ID] [Copy Path]                                       │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ [Locate in Graph]                                               │
└─────────────────────────────────────────────────────────────────┘
```

**Node Window Data Structure:**
```rust
pub(crate) struct InspectWindowGraphNode {
    pub(crate) path: Vec<GlobalId>,

    // For constant nodes - display inline value
    pub(crate) value_view_state: TensorViewState,

    // UI state
    pub(crate) show_op_params: bool,
    pub(crate) inputs_expanded: bool,
    pub(crate) outputs_expanded: bool,
    pub(crate) show_debug_info: bool,  // collapsed by default
}
```

---

### Link Inspect Window - Full Design

```
┌─────────────────────────────────────────────────────────────────┐
│ Tensor: hidden_states                                       [X] │
├─────────────────────────────────────────────────────────────────┤
│ HEADER SECTION                                                  │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ encoder → block_0 → hidden_states                           │ │
│ │ Category: Intermediate tensor                               │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ TYPE INFO                                                       │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ DType:          Float32                                     │ │
│ │ Symbolic Shape: (batch, seq_len, hidden_dim)                │ │
│ │ Concrete Shape: (1, 128, 768)  [from last execution]        │ │
│ │ Total Elements: 98,304                                      │ │
│ │ Memory Size:    384 KB                                      │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ CONNECTIVITY                                                    │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Source:                                                     │ │
│ │   Add: residual_add  →  output[0]              [Open Node]  │ │
│ │                                                             │ │
│ │ Destinations (2):                                           │ │
│ │   LayerNorm: layer_norm_1  ←  input[0]         [Open Node]  │ │
│ │   Attention: self_attn     ←  input[0]         [Open Node]  │ │
│ └─────────────────────────────────────────────────────────────┘ │
│ (hover = highlight node, click = select, Open = open window)    │
│                                                                 │
│ LIVE VALUE                                                      │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ ● Subscribed                              [Unsubscribe]     │ │
│ │                                                             │ │
│ │ -- OR --                                                    │ │
│ │                                                             │ │
│ │ ○ Not subscribed    [Subscribe for Live Updates]            │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ TENSOR VALUE                                                    │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Showing: [●Live] [○Stored]                                  │ │
│ │                                                             │ │
│ │ Statistics:                                                 │ │
│ │   Min: -2.341   Max: 3.892   Mean: 0.003   Std: 0.912      │ │
│ │   NaN: 0        Inf: 0       Zeros: 1,247                   │ │
│ │                                                             │ │
│ │ ┌─────────────────────────────────────────────────────────┐ │ │
│ │ │ [Full tensor_view widget]                               │ │ │
│ │ │                                                         │ │ │
│ │ │  Axes: Row=[dim 1] Col=[dim 2]  Fixed: dim 0 = 0        │ │ │
│ │ │                                                         │ │ │
│ │ │  ┌─────┬─────┬─────┬─────┬─────┐                        │ │ │
│ │ │  │     │ [0] │ [1] │ [2] │ ... │                        │ │ │
│ │ │  ├─────┼─────┼─────┼─────┼─────┤                        │ │ │
│ │ │  │ [0] │ 0.12│-0.34│ 0.56│ ... │                        │ │ │
│ │ │  │ [1] │ 0.78│ 0.91│-0.23│ ... │                        │ │ │
│ │ │  └─────┴─────┴─────┴─────┴─────┘                        │ │ │
│ │ │                                                         │ │ │
│ │ │  Page 1/128  [Prev] [Next]  Rows: 10  Cols: 10          │ │ │
│ │ └─────────────────────────────────────────────────────────┘ │ │
│ │                                                             │ │
│ │ -- OR (if no value yet) --                                  │ │
│ │                                                             │ │
│ │ No value available.                                         │ │
│ │ [Fetch Stored Value]  (for constants/stored inputs)         │ │
│ │ -- or --                                                    │ │
│ │ Subscribe to see live values during execution.              │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ ▶ Debug Info (collapsed by default)                             │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ GlobalId:  145                                              │ │
│ │ Full Path: [1, 42, 145]                                     │ │
│ │ [Copy ID] [Copy Path]                                       │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ [Locate in Graph] [Copy Value as JSON]                          │
└─────────────────────────────────────────────────────────────────┘
```

**Link Window Data Structure:**
```rust
pub(crate) struct InspectWindowGraphLink {
    pub(crate) path: Vec<GlobalId>,

    // Stored tensor fetching
    pub(crate) stored_value_requested: Option<TensorStoreTensorId>,
    pub(crate) stored_value: Option<Result<NDArrayNumericTensor<DynRank>, String>>,

    // Tensor view states
    pub(crate) value_view_state: TensorViewState,        // For stored values
    pub(crate) subscribed_view_state: TensorViewState,   // For live values

    // Cached statistics (computed when tensor loads)
    pub(crate) cached_stats: Option<TensorStats>,

    // UI state
    pub(crate) show_connectivity: bool,
    pub(crate) show_debug_info: bool,  // collapsed by default
    pub(crate) value_source_preference: ValueSourcePreference,
}

pub(crate) struct TensorStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std: f64,
    pub nan_count: usize,
    pub inf_count: usize,
    pub zero_count: usize,
}

pub(crate) enum ValueSourcePreference {
    Live,    // Prefer subscribed/live value
    Stored,  // Prefer stored value
}
```

---

## Graph Type-Specific Information

### For SymbolicGraph

**Node-specific:**
- ONNX operation name
- Operation parameters (epsilon, axis, kernel_size, etc.)
- For Constant ops: inline tensor value

**Link-specific:**
- DType, Shape (symbolic and concrete)
- TensorType category (Input, Output, Intermediate, Constant)
- StoredOrNot status

### For SuperGraph

**Node-specific:**
- Node variant type (ModelExecution, MilliOpGraph, Loop, Conditional, etc.)
- For ModelExecution: which model ID
- For Loop: iteration count, condition

**Link-specific:**
- Link variant type (Tensor, String, TensorMap, Tokenizer, Hash)
- For Tensor links: dtype, shape
- For TensorMap: weight count, total size

### For MilliOpGraph

**Node-specific:**
- Milli-op type
- Kernel info if applicable

**Link-specific:**
- DType, Shape
- Buffer allocation info

---

## Interaction Patterns

### From Node Window → Graph
| Action | Target | Result |
|--------|--------|--------|
| Hover input/output row | Link in graph | Highlight link |
| Click input/output row | Link in graph | Select link |
| Double-click row / [Open] | Link window | Open new window |
| [Locate in Graph] | Node in graph | Pan view to center on node |

### From Link Window → Graph
| Action | Target | Result |
|--------|--------|--------|
| Hover source/dest row | Node in graph | Highlight node |
| Click source/dest row | Node in graph | Select node |
| [Open Node] button | Node window | Open new window |
| [Locate in Graph] | Link in graph | Pan view to center on link |

### From Window → Window
| Source | Action | Result |
|--------|--------|--------|
| Node window | Click [Open] on I/O | Open Link window |
| Link window | Click [Open Node] | Open Node window |

---

## Fixes Applied

### Double-click on link nodes now works
**File:** `graph_explorer/mod.rs` (lines 931-946)

The match statement for double-click handling was missing cases for:
- `ConstantLinkNode`
- `ConnectionByNameSrc`
- `ConnectionByNameDest`

These now properly open link inspect windows.

---

## Implementation Phases

### Phase 1: Core Structure
1. Update `InspectWindowGraphNode` struct with new fields
2. Update `InspectWindowGraphLink` struct with new fields
3. Add `TensorStats` computation helper
4. Implement basic expanded layouts for both window types

### Phase 2: Interactive Tables
1. Create `render_link_row()` helper for node windows
2. Create `render_node_row()` helper for link windows
3. Wire up hover → `next_explorer_hovered`
4. Wire up click → `explorer_selection`
5. Wire up [Open] buttons → spawn new windows

### Phase 3: Type-Specific Info
1. Add SymbolicGraph-specific node details (op params)
2. Add SymbolicGraph-specific link details (tensor category)
3. Add SuperGraph-specific node details (node variant)
4. Add SuperGraph-specific link details (link variant)
5. Inline constant display for Constant nodes

### Phase 4: Tensor Value Display
1. Tensor statistics computation and display
2. Value source toggle (subscribed vs stored)
3. Memory size calculation
4. Symbolic vs concrete shape display

### Phase 5: Actions & Polish
1. [Copy ID], [Copy Path] buttons
2. [Locate in Graph] functionality
3. Collapsible sections with persistent state
4. Execution timing history (avg over N runs)

---

## Helper Functions

```rust
/// Render an interactive row for a link (used in node windows)
/// Shows: index, label/name, dtype, shape, [Open] button
fn render_link_row(
    ui: &mut egui::Ui,
    index: usize,
    link_id: GlobalId,
    graph: &dyn GraphDyn,
    parent_path: &[GlobalId],
) -> LinkRowResponse {
    // Returns: hovered, clicked, open_requested
}

/// Render an interactive row for a node (used in link windows)
/// Shows: "OpType: label" with port info, [Open] button
fn render_node_row(
    ui: &mut egui::Ui,
    node_id: GlobalId,
    port_info: &str,  // e.g., "input[0]" or "output[0]"
    graph: &dyn GraphDyn,
    parent_path: &[GlobalId],
) -> NodeRowResponse {
    // Returns: hovered, clicked, open_requested
}

/// Get link metadata in a uniform way across graph types
fn get_link_metadata(graph: &dyn GraphDyn, link_id: GlobalId) -> LinkMetadata {
    // label, dtype, shape, category, etc.
}

/// Get node metadata in a uniform way across graph types
fn get_node_metadata(graph: &dyn GraphDyn, node_id: GlobalId) -> NodeMetadata {
    // op_kind, label, params, etc.
}

/// Compute tensor statistics
fn compute_tensor_stats(tensor: &NDArrayNumericTensor<DynRank>) -> TensorStats {
    // min, max, mean, std, nan/inf/zero counts
}

/// Format a path for breadcrumb display using labels, not IDs
fn format_path_breadcrumb(path: &[GlobalId], graphs: ...) -> String {
    // "encoder → block_0 → layer_norm_1"
    // Falls back to index-based names if no label available
}
```

---

## Notes

- Both window types should feel like first-class citizens with equal depth of information
- The key differentiator: nodes have I/O relationships, links have source/dest relationships
- Tensor viewing is link-centric (links ARE tensors), but nodes can show inline values for constants
- All interactive elements should have consistent hover/click/open patterns
- Stats computation should be lazy (only when tensor is loaded) and cached

### Design Philosophy: User-Facing vs Implementation Details

**Prominent (always visible):**
- Labels/names (ONNX names, user-defined labels)
- Operation types
- Tensor metadata (dtype, shape)
- Connectivity (what connects to what)
- Values and statistics

**Hidden by default (in "Debug Info" section):**
- GlobalIds
- Full path arrays
- Internal identifiers
- Copy ID/Path buttons

The UI should speak in terms of the *graph structure* (operations, tensors, data flow),
not the *memory representation* (GlobalIds, path vectors). Users care about "the LayerNorm
that normalizes hidden_states", not "GlobalId(147)".
