# Compiler Problem Shape

This document captures the design reasoning for the whisper-tensor compiler.
It exists so that future attempts (human or AI) don't have to re-derive these
ideas from scratch.

## What We're Compiling

The NanoGraph (`nano_graph::pattern`) is a compressed scalar DAG. All tensor
operations from the milli-op graph have been dissolved into individual scalar
atoms — each atom computes one value from one or more input values. Known
dimensions (weight shapes, hidden sizes) are fully expanded; unknown dimensions
(batch, seq_len) are symbolic (`SymDim`).

Compression is achieved by grouping structurally identical atoms into
`AtomGroup`s. **AtomGroups are a compression artifact.** They exist because
emitting every atom individually would require terabytes of RAM. The grouping
is determined by what the milli→nano lowering chose to packetize, not by any
semantic boundary the compiler should respect. The compiler must be able to
collect atoms from across multiple groups, or from partial groups, when
deciding how to partition work.

### Key types

- `AtomId(u64)` — unique scalar atom identifier. Contiguous within a group.
- `AtomGroup` — compression packet: `base_id`, `count`, `op: ScalarOp`,
  `sym_dims`, `reduce_dims`, `inputs: Vec<InputRef>`.
- `InputRef` — how atoms in a group address their source atoms:
  - `Broadcast(AtomId)` — all atoms read the same source (shared input).
  - `Affine { base, stride }` — atom `i` reads `base + stride * i`.
  - `StridedBroadcast { base, stride, repeat }` — atom `i` reads
    `base + stride * (i / repeat)`. Each block of `repeat` atoms shares
    one source. Used for merged matmul Mul groups.
  - `Modular { base, stride, modulus }` — atom `i` reads
    `base + stride * (i % modulus)`. Cyclic/tiling access for broadcasts
    along batch dimensions.
  - `SymAffine { base, stride_i, stride_k }` — `base + stride_i*i + stride_k*k`,
    only used for truly symbolic (runtime-unknown) reductions.
  - `Explicit(Vec<AtomId>)` — arbitrary per-atom mapping. Rare after lowering
    improvements; only used for irregular patterns that no other mode captures.
- `ScalarOp` — the operation each atom performs:
  - `Literal(NumericScalar)` — constant value.
  - `Identity { compute_dtype, output_dtype }` — dtype cast (only emitted when
    dtype actually changes; view ops like Transpose/Split/Slice are zero-cost).
  - `Binary { op, compute_dtype, output_dtype }` — Add/Sub/Mul/Div/Max/Min/
    Mod/Pow plus comparison (Equal/Greater/Less/etc) and logical (And/Or/Xor).
  - `Unary { op, compute_dtype, output_dtype }` — Neg/Abs/Exp/Ln/Sqrt/
    Reciprocal/Tanh/Floor/Ceil.
  - `Select { compute_dtype, output_dtype }` — ternary condition ? x : y.
  - `ReduceSum { reduce_count, reduce_stride, compute_dtype, output_dtype }` —
    sum over a known number of steps. Input is Affine; the op iterates
    k=0..reduce_count reading at offset k*reduce_stride.
  - `ReduceMax { reduce_count, reduce_stride, compute_dtype, output_dtype }` —
    max over a known number of steps. Same addressing as ReduceSum.
  - `IndirectLoad { table_base, output_dtype }` — runtime-indexed table lookup
    (Gather/embedding). One input (the computed index).
- `SymDim` — symbolic runtime dimension (batch, seq_len). Rare in fully
  concrete models like GPT-2.

### What tensor dissolution gives us

Intermediate tensors and their layouts have been completely dissolved. In a
tensor-level IR, a matmul produces an [M, N] tensor with a committed memory
layout. In the NanoGraph, there is no intermediate tensor — just atoms that
produce values and other atoms that consume them.

This means the *order of values in memory is a free variable*. The compiler
chooses how to arrange intermediates (or whether they exist in memory at all
vs. being register-transient). No layout has been imposed from above. The only
fixed layouts are external inputs (arriving from the outside world) and final
outputs (which must match what the consumer expects).

### View ops are zero-cost

Reshape, Transpose, Split, Slice, and Concat do not produce atoms. They
register views into existing atoms with adjusted strides:
- **Reshape/Squeeze/Unsqueeze**: re-register with same base_id, new layout.
- **Transpose**: same base_id, permuted strides.
- **Split**: base_id + offset, input's strides (may be non-contiguous).
- **Slice**: base_id + offset, strides scaled by step.
- **Concat**: segmented view with multiple (base_id, strides) segments.

The downstream `build_input_ref` handles non-row-major and segmented strides
correctly when computing InputRef for consumers.

## GPT-2 NanoGraph Profile (the concrete target)

GPT-2 (10-layer, 768-dim) with input [4, 4, 4] produces:

- **45,921 groups**, 8.1B atoms, 0 symbolic groups
- Lowering time: 0.27s (fast)

Op breakdown:
- 21,801 Mul (Binary) — matmul element products
- 21,630 ReduceSum — matmul contractions (reduce_count=K, reduce_stride=N)
- 2,082 Literal — weights, constants
- 12 Identity — legitimate dtype casts only
- ~400 other compute (Add, Div, Sub, Pow, Sqrt, Exp, Tanh, Select, ReduceMax)
- 1 IndirectLoad — embedding lookup

InputRef distribution:
- 43,949 Affine — dominant, used for elementwise and reduce inputs
- 21,642 StridedBroadcast — merged matmul Mul groups
- 283 Broadcast — scalar broadcasts
- 99 Modular — cyclic/tiling broadcasts
- 13 Explicit — irreducible irregular patterns (0.7 MB total)
- 0 SymAffine — all reductions use known parameters

### Matmul structure after lowering

For C[M,N] = A[M,K] @ B[K,N]:
- **M Mul groups**, each count=K*N:
  - Input 0: `StridedBroadcast { base: A[m,0], stride: 1, repeat: N }`
  - Input 1: `Affine { base: B[0,0], stride: 1 }`
- **M ReduceSum groups**, each count=N:
  - Input: `Affine { base: mul_group_m, stride: 1 }`
  - `reduce_count: K, reduce_stride: N`

Total per matmul: 2M groups. GPT-2 has ~73 matmuls contributing ~21K Mul + ~21K ReduceSum groups.

## The Three Hardware Constraints

1. **Multiple ALUs.** We have more than one execution unit, so we must schedule
   ops in parallel.

2. **Finite fast storage.** Registers and caches are limited. We must decide
   when to shuffle data between memory tiers, and which ops to co-locate so
   their intermediates stay in fast storage.

3. **Limited instruction memory.** Instructions themselves consume RAM/cache.
   We can't emit one instruction per scalar atom — we must express repeated
   computation as SIMD operations and loops.

## Memory Bandwidth Is the Primary Constraint

Most compute platforms are memory-bandwidth limited. The optimization target:
**minimize the number of times data crosses a cache boundary.** Every load/store
across the boundary to the next memory tier is the dominant cost.

## Execution Model: Lanes, Phases, and Barriers

### The lane model

The execution unit is a **lane** — a persistent thread pinned to a core (or
warp on GPU). A lane runs through the entire model computation, hitting
**barrier sync points** between phases. The lane's cache is persistent
across phases.

```
Lane 0 (core 0): phase_0_work → barrier → phase_1_work → barrier → phase_2_work → ...
Lane 1 (core 1): phase_0_work → barrier → phase_1_work → barrier → phase_2_work → ...
...
Lane N (core N): phase_N_work → barrier → phase_N_work → barrier → ...
```

A **phase** is the work between two consecutive barriers. Within a phase,
all lanes execute independently — no cross-lane communication. At the
barrier, all lanes synchronize and their outputs become visible to all lanes
in the next phase.

### Why barriers align with matmul reductions

In a sequential matmul chain (the critical path through a transformer),
matmul N+1's every output depends on ALL outputs from matmul N (because
the reduction reads the full output vector). This is a mandatory sync
point — you can't start any row of matmul N+1 until ALL rows of matmul N
are done.

Barriers naturally align with these reduction boundaries. Between barriers,
the matmul rows are independent and distribute across lanes.

### Cyclic dependencies are managed, not forbidden

Unlike the earlier "acyclic kernel DAG" model, this model allows cyclic
dependencies between lanes. Lane 0's phase 2 output might be read by
lane 1 in phase 3, while lane 1's phase 2 output is read by lane 0 in
phase 3. This is fine — the barrier ensures both lanes' phase 2 outputs
are visible before any lane starts phase 3.

The constraint is: **within a phase, lanes are independent.** All cross-lane
communication happens through the barrier (via the shared values buffer).

### Cache affinity across phases (lane pinning)

A lane's cache is persistent across phases. Lane 0 in phase 3 inherits
whatever is hot in L1/L2 from lane 0's phases 0-2. The partitioner should
exploit this:

- **Consistent row assignment.** If lane 0 handles rows 0-95 of matmul 1,
  it should handle rows 0-95 of matmul 2, the same slice of LayerNorm,
  etc. This keeps the row slice hot across phases.

- **Cache-aware cost model.** When evaluating whether to assign a group to
  a lane's phase N, the partitioner should consider what's already in that
  lane's cache from phases 0..N-1. Values already hot are "free" to read.

- **Lane assignment IS tiling.** The partitioner doesn't separately decide
  tiling and kernel assignment — they're the same decision. Lane 0 gets
  row slice [0, 96) across ALL matmuls in ALL layers.

### Execution plan format

```rust
struct ExecutionPlan {
    num_lanes: usize,
    phases: Vec<Phase>,
}
struct Phase {
    lane_work: Vec<Vec<usize>>,  // lane_idx -> group indices for this phase
}
```

The codegen emits one function per lane — the lane function contains all
phases with barrier calls between them:

```rust
fn lane_0(values: *mut f32, barriers: &[AtomicBarrier]) {
    // Phase 0: Q matmul rows 0-95
    ... compute ...
    barrier_wait(&barriers[0]);
    // Phase 1: attention head 0
    ... compute ...
    barrier_wait(&barriers[1]);
    // ...continues through all phases/layers...
}
```

### Inter-barrier spans as compilation units

Each span (one lane's work in one phase) is compiled as a standalone code
block within the lane function. For CPU targets, each span is a sequence
of compute loops. The barrier call is a simple function call that blocks
until all lanes arrive.

This enables work-stealing: if a lane finishes its span early, the runtime
can optionally assign it another lane's span from the same phase. But the
primary model is pinned execution for cache affinity.

### The partitioner's job

The partitioner produces an `ExecutionPlan`: how many lanes, where the
barriers go, and which groups each lane executes in each phase.

1. **Identify barrier positions.** Find the matmul reduction boundaries on
   the critical path. These are mandatory sync points.

2. **Assign groups to lanes across all phases.** For each phase, distribute
   the independent work (matmul rows, attention heads, elementwise slices)
   across lanes. The same lane should get a consistent "slice" across phases
   to maximize cache reuse.

3. **Balance work per phase.** At each barrier, all lanes wait for the
   slowest lane. Minimize waiting by balancing work across lanes within
   each phase.

4. **Track lane cache contents.** When choosing assignments, consider what
   each lane already has hot from previous phases. Prefer assignments that
   reuse cached data over assignments that require new loads.

5. **Handle parallel matmul groups.** Q/K/V projections in attention are
   three independent matmuls that branch from the same source and
   reconverge. These can be interleaved across lanes (lane 0 does Q rows
   0-95 AND K rows 0-95).

### Requirements

1. **Within each phase, lanes are independent.** No lane reads another
   lane's current-phase output. All cross-lane communication goes through
   barriers.

2. **Balanced work per phase.** Max lane work / min lane work < 2x.

3. **Cache-consistent lane assignment.** Each lane's data slice should be
   stable across phases (same rows, same heads).

4. **Reasonable number of lanes.** Match available hardware parallelism
   (4-16 for CPU, 32+ for GPU).

5. **Few barriers.** Each barrier is a sync cost. Minimize the number of
   phases while respecting the mandatory matmul reduction boundaries.

### Previous partitioning attempts and what went wrong

**nano_part_b** (Broadcast source analysis): Detected matmul row structure
and split rows into kernels. Result on GPT-2: star topology with circular
deps — 35 small kernels feeding one mega-kernel (79% of atoms), with the
mega-kernel also feeding back into them. Fundamental problem: split within
matmuls across layers without understanding sequential structure.

**nano_part_topo/merge/live** (contiguous topo range approaches): Guaranteed
acyclicity by restricting kernels to contiguous ranges of topologically-
ordered groups. This was **a wrong constraint**: it forces purely sequential
execution with zero within-phase parallelism. A matmul's rows are independent
and should run in parallel — but they're interleaved in topo order with the
matmul's shared inputs, so they can't be separated by contiguous ranges.

### What a good partitioner must do

The partitioner assigns groups to kernels. Groups from different parts of
the topo order can be in the same kernel — this is NOT limited to contiguous
ranges.

**Acyclicity constraint:** The kernel dependency graph must be a DAG. This
means: if kernel A contains a group that reads from a group in kernel B,
and kernel B contains a group that reads from a group in kernel A, that's
a cycle and is forbidden. The partitioner must check and enforce this.
Note: acyclicity does NOT require contiguous topo ranges. Two parallel
kernels can have interleaved group indices as long as neither reads from
the other (both read from a third earlier kernel).

**Parallelism:** The whole point of partitioning is to enable parallel
execution. Within a single matmul, different output rows are independent
(they read from the same weight data but produce independent outputs).
These should be in separate kernels that can execute simultaneously.
A partition that produces only sequential kernels defeats the purpose.

**The two-level structure of transformer models:** GPT-2 (and transformers
in general) has two levels of structure:

1. **Sequential phases.** Layer 1 must complete before layer 2 starts.
   Between layers, only the residual stream is live. These phase boundaries
   have minimal cross-kernel traffic and are natural places for kernel
   boundaries.

2. **Parallel work within phases.** Within a single layer, the Q/K/V
   matmul projections produce independent row groups. The attention score
   computation across heads is independent. The FFN's output rows are
   independent. These should be split across parallel kernels.

A good partitioner combines both: phase-level sequential boundaries
(between layers) with within-phase parallelism (split matmul rows).

Note: the earlier "acyclic kernel DAG" requirement is superseded by the
lane+barrier model. Cyclic dependencies between lanes are fine — the
barriers manage synchronization. The constraint is independence WITHIN
a phase, not across phases.

### GPT-2 structure for reference

- 45,921 groups, 8.1B atoms
- 10 transformer layers, sequential pipeline
- Each layer: LayerNorm → attention (Q/K/V projections, scores, softmax,
  output projection) → residual → LayerNorm → FFN (up + down) → residual
- Matmul structure: M Mul groups (StridedBroadcast) + M ReduceSum groups
  per matmul. Rows are independent. ~73 matmuls total.
- Inter-layer pinch points: only residual stream (~768 values) is live
- Within a matmul: rows share weight data but produce independent outputs

### Suggested approach (not prescriptive)

One possible strategy:
1. Detect sequential phase boundaries (live set analysis — where does the
   set of "values produced but not yet consumed" narrow?)
2. Within each phase, identify independent sub-DAGs (connected components
   after removing shared inputs like weight literals)
3. Assign independent sub-DAGs to parallel kernels
4. Verify acyclicity of the resulting kernel dependency graph
5. Balance by splitting large kernels or merging small ones

## Key Principles

- **AtomGroups are compression, not semantics.** The compiler reads through
  them but partitions atoms freely across group boundaries.
- **Tensor layouts are dissolved.** Intermediate data layout is a compiler
  output, not an input.
- **Memory bandwidth dominates.** Minimize cache-boundary crossings.
- **Cache is a cost, not a hard constraint.** Exceeding cache incurs streaming
  penalties but doesn't prohibit a kernel.
- **Shared data footprint drives grouping.** Atoms that share inputs/outputs
  belong together to amortize memory traffic.
- **Barriers manage cross-lane sync.** Lanes are independent within phases;
  barriers separate phases where cross-lane data is needed.
- **Streaming dimensions reduce liveness.** Reduction dims are processed one
  step at a time, not materialized fully.
- **No tensor op pattern matching.** One general mechanism for discovering
  data sharing structure in the scalar DAG.
