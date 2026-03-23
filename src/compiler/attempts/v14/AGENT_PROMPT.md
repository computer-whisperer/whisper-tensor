# Partitioner Agent Prompt — v14 gen-2

This document is the prompt template for multi-agent partitioner attempts.
Each agent receives this plus a unique creative direction section.

---

## The Problem

You are building a scheduler for a dataflow graph that must execute
efficiently across multiple processor cores. Given a DAG of scalar
operations with data dependencies, partition the work across N execution
lanes while respecting dependencies and maximizing throughput.

The input is a **NanoGraph** — a compressed scalar DAG where all tensor
operations have been dissolved into individual scalar atoms grouped by
structural regularity. The output is a sequence of **Phases** separated
by barrier sync points, where each phase contains independent **Spans**
(one per lane) that execute in parallel.

## CRITICAL: Groups Must Be Split

**This is the single most important requirement.** AtomGroups are a
compression artifact — they are NOT execution boundaries. A group of
49,152 atoms running on a single lane while 7 other lanes sit idle is a
failure, not a valid plan.

The partitioner MUST split groups across lanes. A group with count=N
should become N/num_lanes pieces, each assigned to a different lane.
The infrastructure for this is fully built and tested:

- `atom_offset` on AtomGroup: the second half of a split group uses
  `atom_offset = split_point` so InputRef resolution still works correctly
- `insert_group_at(base_id, count, atom_offset, ...)`: places a split
  fragment in a span NanoGraph at the correct atom IDs
- The eval loop uses `i + group.atom_offset` for all InputRef resolution
- The JIT codegen loops from `atom_offset..atom_offset+count`
- The value store handles split outputs: if group [0..1000) is split
  into spans producing [0..500) and [500..1000), downstream phases find
  both entries via overlap-based range matching

**Every prior attempt disabled splitting and produced plans where serial
chains of operations each occupied an entire phase on a single lane.**
This is the primary failure mode to avoid.

### How splitting works

To split group `g` (base_id=B, count=N) across K lanes:

```rust
let chunk = N / K;
for lane in 0..K {
    let start = lane as u64 * chunk;
    let count = if lane == K-1 { N - start } else { chunk };
    // In lane's span NanoGraph:
    span_graph.insert_group_at(
        AtomId(B + start),  // base_id of this fragment
        count,               // atoms in this fragment
        start,               // atom_offset for InputRef resolution
        g.output_dtype,
        g.op.clone(),
        g.sym_dims.clone(),
        g.inputs.clone(),    // inputs stay the same — atom_offset handles it
    );
    // Declare output range for this fragment:
    span.outputs.push(AtomRange { base: AtomId(B + start), count, dtype: g.output_dtype });
}
```

The inputs vector is the SAME for all fragments. The `atom_offset` parameter
tells the eval/codegen to resolve `input.resolve(i + atom_offset)` instead of
`input.resolve(i)`, which produces the correct source atom IDs for each fragment.

### When NOT to split

- **Literal groups**: all atoms have the same value. Duplicate into each
  span rather than splitting — each lane needs the full literal.
- **Reduce groups** reading from split sources: a ReduceSum over atoms
  [0..K) cannot be split if K is the reduction dimension. The reduce
  itself must stay whole. But the *output* of the reduce (which is
  typically small) can be broadcast or duplicated for downstream consumers.
- **Very small groups** (count < num_lanes): not worth splitting.

### What MUST be split

- **Elementwise ops** (Binary, Unary, Select, Identity) with large count:
  these are embarrassingly parallel. Split across all lanes.
- **MatMul Mul groups** (count = M*K with StridedBroadcast input): each
  row's K products are independent. Split by rows across lanes.
- **MatMul ReduceSum groups** (count = M): each output element is an
  independent reduction. Split across lanes.
- **IndirectLoad groups**: each lookup is independent. Split across lanes.

## Gen-1 Postmortem: What Went Wrong

Eight partitioner attempts were built and tested. ALL of them failed to
split groups. The "best" result (partitioner B) produced 65 phases for
GPT-2 where the vast majority were single-lane:

```
Phase  3: 1 group, 49152 atoms, ALL on lane 7 (Sub)
Phase  4: 2 groups, 49153 atoms, ALL on lane 6 (Pow)
Phase  7: 16 groups, 26.6M atoms, ALL on lane 4 (MatMul+Add)
Phase 12: 38 groups, 52.2M atoms, ALL on lane 7 (MatMul+pointwise chain)
Phase 63: 3692 groups, 1.27B atoms, ALL on lane 7 (MatMul+Mul+Add+Pow chain)
```

The result: JIT execution took 68s vs 13.5s for the single-threaded
ndarray interpreter. The 8-lane "parallel" plan was 5x SLOWER than
sequential because:

1. **No group splitting** — each group was assigned whole to one lane
2. **Serial chains became single-lane phases** — a chain of ops where
   each depends on the previous got assigned to one lane per phase,
   with 7 lanes idle
3. **Lane jumping** — the chain bounced between lanes across phases
   (lane 7 → lane 6 → lane 4 → lane 0 → ...) creating unnecessary
   barriers with no parallelism benefit

The root cause in every attempt was treating groups as atomic scheduling
units. The "split large groups" step was either not implemented or was
explicitly disabled with comments like "split outputs create a mismatch
with how later phases look up data in the value store" — which was false.
The store handles split outputs correctly.

## The Execution Model

- **Lanes** are persistent threads. A lane runs through all phases,
  hitting barrier sync points between them.
- **Phases** are separated by barriers. Within a phase, all lanes
  execute independently — no cross-lane communication.
- **Spans** are one lane's work within one phase. Each span is a
  self-contained NanoGraph fragment compiled and executed independently.

### Cache affinity (lane pinning)

A lane's cache is persistent across phases. If lane 0 handles atoms
[0..6144) of a MatMul output, it should handle the same atom range of
the downstream Add, Sub, Div, etc. This keeps data hot in L1/L2.

**Lane assignment IS tiling.** Lane 0 gets the first 1/8 of every
splittable group across the entire model. This is not an optimization —
it's the fundamental scheduling strategy.

### Where barriers go

Barriers are needed when a downstream op reads atoms produced by
multiple lanes. The canonical example: a ReduceSum that contracts
across a dimension where the source data was split across lanes.

Barriers are NOT needed between every op in a serial chain. If
Sub→Pow→ReduceMean→Sqrt→Div is split identically across 8 lanes
(each lane gets the same 1/8 slice), the chain runs within each lane
with zero barriers.

## The NanoGraph

A NanoGraph is a compressed scalar DAG. Every tensor operation has been
dissolved into scalar atoms, grouped into `AtomGroup`s for compression.

**AtomGroups are compression artifacts, not semantic boundaries.** The
scheduler splits, duplicates, or rearranges groups freely.

### Key Types

```rust
struct AtomId(pub u64);
struct AtomRange { pub base: AtomId, pub count: u64, pub dtype: DType }
struct SymDim(pub u16);

enum InputRef {
    Broadcast(AtomId),
    Affine { base: AtomId, stride: i64 },
    StridedBroadcast { base: AtomId, stride: i64, repeat: u64 },
    Modular { base: AtomId, stride: i64, modulus: u64 },
    Explicit(Vec<AtomId>),
}
impl InputRef {
    pub fn resolve(&self, i: u64) -> AtomId;
}

enum ScalarOp {
    Literal(NumericScalar),
    Identity,
    Binary { op, compute_dtype },
    Unary { op, compute_dtype },
    Select,
    Reduce { kind, reduce_count: u64, reduce_stride: i64, compute_dtype },
    IndirectLoad { table_base: AtomId },
}

struct AtomGroup {
    pub base_id: AtomId,
    pub count: u64,
    pub atom_offset: u64,    // nonzero for split groups
    pub output_dtype: DType,
    pub op: ScalarOp,
    pub sym_dims: Vec<SymDim>,
    pub inputs: Vec<InputRef>,
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
    pub fn push_group(...) -> AtomId;
    pub fn add_input_tensor(...) -> AtomId;

    // Construction (specific ID placement — for span NanoGraphs)
    pub fn insert_group_at(&mut self, base_id: AtomId, count: u64,
                            atom_offset: u64, output_dtype: DType, op: ScalarOp,
                            sym_dims: Vec<SymDim>, inputs: Vec<InputRef>);
    pub fn insert_input_tensor_at(&mut self, base_id: AtomId, tensor_id: GlobalId,
                                   count: u64, dtype: DType);

    pub sym_dim_names: HashMap<String, SymDim>,
    pub sym_dim_bounds: HashMap<SymDim, u64>,
    pub outputs: Vec<AtomId>,
}
```

## Output Types

```rust
pub struct Phase {
    pub spans: Vec<Span>,  // one per lane
}

pub struct Span {
    pub graph: NanoGraph,
    pub inputs: Vec<AtomRange>,
    pub outputs: Vec<AtomRange>,
}
```

**Atom ID invariant:** Span NanoGraphs use the same atom ID space as the
main graph. Atom X in a span is the same atom as atom X in the main graph.

## Structural Invariants

1. **Within a phase, spans are independent.** No span reads atoms produced
   by another span in the same phase.
2. **Each span's NanoGraph is self-contained.** Every InputRef resolves to
   a group within the span or a declared entry in `span.inputs`.
3. **All model output atoms are produced by some span.**
4. **Groups within each span are in valid topological order.**

## Your Task

Implement a partitioner in a single file:
`src/compiler/attempts/v14/partitioner_X.rs` (where X is your letter)

```rust
pub fn plan(
    graph: &NanoGraph,
    num_lanes: usize,
    input_tensors: &[InputTensor],
    output_atom_ids: &[AtomId],
) -> Vec<Phase>
```

Add `pub mod partitioner_X;` to `src/compiler/attempts/v14/mod.rs`.

The file must compile. Run `cargo check -p whisper-tensor` to verify.
Write unit tests in a `#[cfg(test)] mod tests` block at the bottom.

### What a correct plan looks like

For a serial chain Sub(49152) → Pow(49152) → ReduceMean → Sqrt → Div(49152)
with 8 lanes:

- **ONE phase** (not five separate phases)
- Each lane gets 1/8 of the Sub, Pow, and Div groups
- The ReduceMean stays whole (it's a reduction) — or gets duplicated
  into each lane if the downstream ops need its output
- Sqrt similarly stays whole or gets duplicated
- Zero barriers needed because each lane's slice is independent

For a MatMul (M=768, K=768, producing M ReduceSum groups of count=1 each):
- The M ReduceSum groups are independent — split across lanes
- The M Mul groups (each count=K) are independent — split across lanes
- Each lane gets M/8 rows of both Mul and ReduceSum
- Weights (Literal groups) are duplicated into each lane's span

### Quality metrics

1. **Utilization**: all lanes should have work in every phase. A phase
   where only 1 of 8 lanes is active wastes 87.5% of available compute.
2. **Balance**: within each phase, lane atom counts should be within 2x.
3. **Few phases**: fewer barriers = less sync overhead. A serial chain
   of elementwise ops split across lanes needs ZERO barriers between them.
4. **Lane affinity**: the same atom range slice should stay on the same
   lane across consecutive operations (cache locality).

## GPT-2 Profile (concrete target)

- 98,567 groups, 8.1B atoms (after lowering with full weight data)
- MatMul dominates: 43,136 groups, 7.9B atoms
- 52,552 Gather groups (embedding lookups, 1 atom each)
- Elementwise ops: Mul(182g), Add(148g), Pow(37g), Reshape(48g), etc.
- 10 transformer layers, ~73 matmuls
- Elementwise ops typically have count=49,152 (= 4×4×4×768/batch dims)
- Inter-layer pinch points: residual stream (~768 values)

**Performance target: < 30 seconds on GPT-2 (98K groups, 8 lanes).**

## Scale Considerations

- **O(groups)** or **O(groups²)**: fine (98K groups)
- **O(atoms)** with allocation: will OOM (8.1B atoms × 8 bytes = 65GB)
- Work at group granularity for scheduling decisions. Only touch atoms
  when computing split points within a group.

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

## Independence Requirement

**Your implementation must be completely self-contained.** Do not call
or depend on any code in other partitioner files or attempt directories.
Everything you need is in the NanoGraph API and the v14 types module.

The only imports from the compiler module should be `super::types::{Phase, Span}`.

## Creative Direction

{{CREATIVE_DIRECTION}}
