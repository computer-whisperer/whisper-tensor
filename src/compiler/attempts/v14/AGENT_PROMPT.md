# Partitioner Agent Prompt — v14

This document is the prompt template for multi-agent partitioner attempts.
Each agent receives this plus a unique creative direction section.

---

## The Problem

You are building a scheduler for a dataflow graph that must execute
efficiently across multiple processor cores. This is the core scheduling
problem in any parallel compiler: given a DAG of operations with data
dependencies, partition the work across N execution lanes (threads/cores)
while respecting dependencies and maximizing throughput.

The input is a **NanoGraph** — a compressed scalar DAG where all tensor
operations have been dissolved into individual scalar computations grouped
by structural regularity. The output is an **ExecutionPlan**: a sequence
of phases separated by barrier sync points, where each phase contains
independent spans (one per lane) that execute in parallel.

### The execution model

- **Lanes** are persistent threads, each pinned to a core. A lane runs
  through all phases of the model, hitting barrier sync points between them.
- **Phases** are separated by barriers. Within a phase, all lanes execute
  independently — no cross-lane communication.
- **Spans** are one lane's work within one phase. Each span is a
  self-contained NanoGraph fragment that can be compiled and executed
  independently, given its declared input data.

The key scheduling tension: fewer barriers means less synchronization
overhead, but requires more work to be truly independent within each phase.
More barriers means easier independence but more sync cost and less
opportunity for cross-op fusion.

### What makes this hard

The NanoGraph for GPT-2 has **45,921 groups representing 8.1 billion
scalar atoms**. The dominant structure is matmuls (~73 of them), each
decomposed into M independent row computations. The rows share weight
data but produce independent outputs — this is the primary source of
within-phase parallelism.

### Scale considerations

The primary constraint is **memory**, not compute. Modern CPUs can scan
billions of values quickly if the working set fits in cache and nothing
gets written out to lower memory tiers. What kills you is allocating
per-atom data structures — a `HashMap<AtomId, ...>` with 8.1B entries
is ~200GB.

Rules of thumb for 45K groups / 8.1B atoms:
- **O(groups)** or **O(groups²)**: fine, groups are the natural unit
- **O(groups · log(groups))**: fine
- **O(atoms)** read-only scan: can work if the per-atom expression is
  trivial (a few comparisons, no allocation) and the data stays in cache.
  Scanning 8.1B atoms at L1 speed ≈ a few seconds.
- **O(atoms) with allocation**: will OOM. Don't build `Vec<T>` or
  `HashMap` indexed by atom ID at full scale.

The test suite uses small graphs where everything is fast. Scale issues
only appear on real models. Design your data structures around groups
and ranges, but don't be afraid to do a linear scan over atoms for
analysis if the per-element work is trivially cheap.

## Your Task

Implement a partitioner in a single file:
`src/compiler/attempts/v14/partitioner.rs`

The core function:

```rust
pub fn plan(
    graph: &NanoGraph,
    num_lanes: usize,
    input_tensors: &[InputTensor],
    output_atom_ids: &[AtomId],
) -> Vec<Phase>
```

The caller wraps your `Vec<Phase>` into the full `ExecutionPlan` with
metadata. Your job is the scheduling: where barriers go, which groups
each lane executes, and how to split or duplicate groups for balance
and independence.

## The NanoGraph

A NanoGraph is a compressed scalar DAG. Every tensor operation has been
dissolved into individual scalar atoms. Atoms are grouped into `AtomGroup`s
for compression — structurally identical atoms doing the same operation
with regular addressing patterns.

**AtomGroups are compression artifacts, not semantic boundaries.** The
scheduler is free to split groups, duplicate groups, or rearrange them
as needed.

### Key Types

```rust
struct AtomId(pub u64);  // unique scalar atom identifier
impl AtomId { pub fn offset(self, n: u64) -> Self; }

struct AtomRange { pub base: AtomId, pub count: u64, pub dtype: DType }

struct SymDim(pub u16);  // symbolic runtime dimension (batch, seq_len)

// How atoms in a group address their source atoms
enum InputRef {
    Broadcast(AtomId),           // all atoms read same source
    Affine { base: AtomId, stride: i64 },  // atom i reads base + stride*i
    StridedBroadcast { base: AtomId, stride: i64, repeat: u64 },
    Modular { base: AtomId, stride: i64, modulus: u64 },
    Explicit(Vec<AtomId>),       // arbitrary per-atom (rare)
}
impl InputRef {
    pub fn resolve(&self, i: u64) -> AtomId;
    pub fn distinct_sources(&self, count: u64) -> usize;
}

enum ScalarOp {
    Literal(NumericScalar),      // constant, no inputs
    Identity,                    // dtype cast, 1 input
    Binary { op, compute_dtype },// 2 inputs
    Unary { op, compute_dtype }, // 1 input
    Select,                      // 3 inputs: [cond, x, y]
    Reduce { kind, reduce_count: u64, reduce_stride: i64, compute_dtype },
    IndirectLoad { table_base: AtomId },
}

struct AtomGroup {
    pub base_id: AtomId,     // first atom ID in this group
    pub count: u64,          // number of atoms
    pub atom_offset: u64,    // nonzero for split groups
    pub output_dtype: DType,
    pub op: ScalarOp,
    pub sym_dims: Vec<SymDim>,
    pub inputs: Vec<InputRef>,  // how atoms address their sources
}

struct InputTensor {
    pub tensor_id: GlobalId,
    pub base_id: AtomId,
    pub count: u64,
    pub dtype: DType,
}
```

### NanoGraph API

```rust
impl NanoGraph {
    // Query
    pub fn groups(&self) -> &[AtomGroup];
    pub fn num_groups(&self) -> usize;
    pub fn num_atoms(&self) -> u64;
    pub fn group_of(&self, id: AtomId) -> Option<&AtomGroup>;
    pub fn find_group_idx(&self, id: AtomId) -> Option<usize>;
    pub fn input_tensors(&self) -> &[InputTensor];
    pub fn contains_atom(&self, id: AtomId) -> bool;

    // Analysis
    pub fn liveness(&self) -> Vec<GroupUseCount>;
    pub fn collect_all_producer_indices(
        &self, group: &AtomGroup, gi: usize, out: &mut HashSet<usize>
    );
    pub fn validate(&self) -> Vec<String>;

    // Construction (sequential ID allocation)
    pub fn new() -> Self;
    pub fn push_group(&mut self, count: u64, output_dtype: DType, op: ScalarOp,
                       sym_dims: Vec<SymDim>, inputs: Vec<InputRef>) -> AtomId;
    pub fn add_input_tensor(&mut self, tensor_id: GlobalId,
                             count: u64, dtype: DType) -> AtomId;
    pub fn alloc_placeholder(&mut self, count: u64, output_dtype: DType) -> AtomId;
    pub fn fill_placeholder(&mut self, base_id: AtomId, count: u64,
                             output_dtype: DType, op: ScalarOp,
                             sym_dims: Vec<SymDim>, inputs: Vec<InputRef>);

    // Construction (specific ID placement — for building span NanoGraphs)
    pub fn insert_group_at(&mut self, base_id: AtomId, count: u64,
                            atom_offset: u64, output_dtype: DType, op: ScalarOp,
                            sym_dims: Vec<SymDim>, inputs: Vec<InputRef>);
    pub fn insert_input_tensor_at(&mut self, base_id: AtomId, tensor_id: GlobalId,
                                   count: u64, dtype: DType);

    // Fields
    pub sym_dim_names: HashMap<String, SymDim>,
    pub sym_dim_bounds: HashMap<SymDim, u64>,
    pub outputs: Vec<AtomId>,
}
```

**Span NanoGraph construction:** Use `insert_group_at` and
`insert_input_tensor_at` to place groups and input ranges at specific
atom IDs in span NanoGraphs. These insert into the internal RangeMap
at arbitrary positions without requiring sequential allocation.

## Output Types

```rust
pub struct Phase {
    pub spans: Vec<Span>,  // one per lane
}

pub struct Span {
    pub graph: NanoGraph,        // fragment using main graph's atom ID space
    pub inputs: Vec<AtomRange>,  // reads from shared value store
    pub outputs: Vec<AtomRange>, // writes to shared value store
}
```

**Atom ID invariant:** Span NanoGraphs use the same atom ID space as the
main graph. Atom X in a span is the same atom as atom X in the main graph.
No remapping tables.

## Structural Invariants

1. **Within a phase, spans are independent.** No span reads atoms produced
   by another span in the same phase.
2. **Each span's NanoGraph is self-contained.** Every InputRef resolves to
   a group within the span or a declared entry in `span.inputs`.
3. **All model output atoms are produced by some span.**
4. **Groups within each span are in valid topological order.**

## GPT-2 Profile (concrete target)

- 45,921 groups, 8.1B atoms
- 21,801 Mul + 21,630 ReduceSum (matmuls dominate)
- 2,082 Literal (weights/constants)
- ~400 other compute ops, 12 Identity, 1 IndirectLoad
- 10 transformer layers, ~73 matmuls
- Elementwise ops: single groups with count=49,152
- Matmul structure: M Mul groups (count=K*N) + M ReduceSum groups per matmul;
  rows are independent, share weight inputs
- Inter-layer pinch points: only ~768 residual values live between layers

**Performance target: < 30 seconds on GPT-2 (45K groups, 8 lanes).**
Faster is better, but correctness matters more than speed.

## Module Setup

```rust
#![allow(clippy::all, dead_code, unreachable_patterns)]

use std::collections::{HashMap, HashSet, BTreeSet, BTreeMap};

use crate::nano_graph::{
    AtomGroup, AtomId, AtomRange, InputRef, NanoGraph, ScalarOp, SymDim,
};
use crate::nano_graph::pattern::InputTensor;
use crate::graph::GlobalId;
use crate::dtype::DType;

use super::types::{Phase, Span};
```

Add `pub mod partitioner;` to `src/compiler/attempts/v14/mod.rs`.

The file must compile. Run `cargo check -p whisper-tensor` to verify.
Write unit tests in a `#[cfg(test)] mod tests` block at the bottom.

## Independence Requirement

**Your implementation must be completely self-contained.** Do not call,
import, or depend on any code in `src/compiler/attempts/v13_claude/` or
any other attempt directory. Do not reuse planners, helpers, or utilities
from prior attempts. Everything you need is in the NanoGraph API and the
v14 types module. Build your solution from scratch.

The only imports from the compiler module should be `super::types::{Phase, Span}`.
Everything else comes from `crate::nano_graph::*`, `crate::graph::*`,
and `crate::dtype::*`.

## Gen-1 Results on GPT-2 (47,887 groups, 8 lanes)

Four prior attempts were tested. Here are their results:

| Attempt | Phases | Validation | Cross-lane violations | Imbalance | Time |
|---------|--------|------------|----------------------|-----------|------|
| A (pinch-point liveness) | 27 | 0 errors | 0 | 694M x | <5min |
| B (wavefront/depth) | 173 | 0 errors | 0 | 11.3x | <5min |
| C (greedy lane sim) | 175 | 0 errors | 0 | 76M x | 250ms |
| D (hierarchical super-groups) | 2 | 0 errors | 24 | 37K x | <5min |

Key observations:
- **B had the best overall result**: 0 violations with 11.3x imbalance.
  Its depth-based wavefront approach gives provable independence for free.
  Too many phases (173) — could be reduced by more aggressive merging.
- **A and C achieved 0 violations but catastrophic imbalance** — most work
  ended up on lane 0. The lane assignment heuristics failed to distribute
  matmul rows across lanes.
- **D had only 2 phases** (not enough barriers) and 24 cross-lane violations.
- **The dependency DAG build** (using `collect_all_producer_indices`) takes
  ~140ms for 48K groups — fast and not a bottleneck.
- **Span NanoGraph construction** takes ~80ms per partitioner — also fast.
- **The hard problem is lane assignment + balance**, not phase detection.

## Creative Direction

{{CREATIVE_DIRECTION}}
