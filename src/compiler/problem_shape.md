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

- `AtomId(u32)` — unique scalar atom identifier. Contiguous within a group.
- `AtomGroup` — compression packet: `base_id`, `count`, `op: ScalarOp`,
  `sym_dims`, `reduce_dims`, `inputs: Vec<InputRef>`.
- `InputRef` — how atoms in a group address their source atoms:
  - `Broadcast(AtomId)` — all atoms read the same source (shared input).
  - `Affine { base, stride }` — atom `i` reads `base + stride * i`.
  - `SymAffine { base, stride_i, stride_k }` — `base + stride_i*i + stride_k*k`,
    used for contractions where sources vary with both atom offset and reduction
    iteration.
  - `Explicit(Vec<AtomId>)` — arbitrary per-atom mapping.
- `ScalarOp` — the operation each atom performs (Literal, Identity, Binary,
  Unary, Select, ReduceSum, ReduceMax), each carrying its own compute/output
  dtype.
- `SymDim` — symbolic runtime dimension. Groups may iterate over sym_dims
  (producing a value per sym_dim element) and/or reduce over reduce_dims.

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

The ordering of numbers within a tensor is itself information. Dissolving that
information makes the resulting computation fundamentally more compressible —
the compiler can choose whatever arrangement minimizes memory traffic for the
target hardware.

## The Three Hardware Constraints

We have a giant set of scalar ops to execute. Three properties of real compute
hardware complicate this:

1. **Multiple ALUs.** We have more than one execution unit, so we must schedule
   ops in parallel.

2. **Finite fast storage.** Registers and caches are limited. We must decide
   when to shuffle data between memory tiers, and which ops to co-locate in
   time and space so their intermediates stay in fast storage.

3. **Limited instruction memory.** Instructions themselves consume
   RAM/cache. We can't emit one instruction per scalar atom — we must express
   repeated computation as SIMD operations and loops.

These three constraints are coupled. The loop/SIMD structure determines
instruction footprint, register pressure, and available parallelism
simultaneously.

## Memory Bandwidth Is the Primary Constraint

Most serious compute platforms (GPUs, Tenstorrent cards, and even modern CPUs)
are memory-bandwidth limited. ALU utilization is a secondary concern. This
means the compiler is fundamentally a **data locality scheduling problem**.

The optimization target: **minimize the number of times data crosses a cache
boundary.** Every load and store across the boundary to the next memory tier is
the dominant cost. Intermediates that stay in fast storage are essentially free.

## The Cache Bucket Model

The tightest region of fast memory around an execution lane defines the
fundamental "bucket" of computation:

- **CPU:** L1 cache per core (32-64 KB typical)
- **GPU:** shared memory / L1 near a warp (~64-128 KB)
- **Tenstorrent:** 1 MB SRAM per tensix core

These buckets define how "wide" a subgraph the hardware can hold at once.

### Cache as cost, not hard constraint

The cache budget is NOT a hard constraint on kernel size. A kernel does not
need all its values to fit in cache simultaneously — it executes ops over
time, and values enter and leave the live set as they are produced and
consumed. A chain A → B → C → D has peak liveness of ~2 values regardless
of chain length, because each intermediate is consumed immediately.

A kernel that temporarily exceeds cache simply pays a streaming cost — values
are paged to/from the next memory tier. This is a cost to minimize, not a
hard wall. The alternative (fragmenting large streaming workloads into many
tiny kernels) would be worse due to kernel launch overhead and forced
materialization at every boundary.

What matters is **total memory traffic**: the sum of all loads and stores
across cache boundaries during the kernel's execution. This depends on:
1. The kernel's external I/O (inputs from other kernels + outputs to other
   kernels) — this is the floor, unavoidable regardless of scheduling.
2. The peak liveness during execution — if this exceeds cache, excess values
   are evicted and potentially reloaded, adding penalty traffic.
3. The execution order — a good schedule keeps producers near their consumers,
   minimizing peak liveness. A bad schedule creates long live ranges that
   thrash the cache.

## Three Separable Subproblems

The scheduling problem decomposes into three largely independent levels:

### 1. Kernel Formation (which ops go in which kernel)

A **kernel** is a chunk of compiled code for a group of execution units that
share a cache level — a cluster of CPU cores sharing L2, a set of GPU warps
sharing shared memory, a group of tensix cores sharing SRAM.

The grouping criterion is **shared data footprint**: atoms that read
overlapping inputs and contribute to related outputs belong in the same kernel,
because co-scheduling them means loading shared data once and keeping
intermediates cache-resident.

The kernel boundary decision is cost-based: **does splitting here reduce
total memory traffic enough to justify the overhead of a separate kernel?**
A kernel that streams through a large dataset, temporarily exceeding cache,
may be better than fragmenting into many small kernels that each force
materialization at their boundaries.

Evaluating candidate kernels requires a cheap cost heuristic — estimating
memory traffic without fully scheduling the kernel. The structural properties
of the addressing modes (Broadcast, Affine, SymAffine) enable this: they
tell us which values are shared, which are streamed sequentially, and which
dimensions are reduced (and therefore streamed one step at a time).

### 2. Intra-Kernel Organization (how ops are arranged within a kernel)

Within a kernel, the compute group has internal memory tiering (registers, L1,
L2/shared). Scheduling decisions here determine:

- Which ops share registers vs. L1 vs. the kernel-level shared cache
- SIMD grouping and instruction ordering
- Data layout of intermediates within the kernel's cache

This is where the actual instruction sequence gets determined.

### 3. Inter-Kernel Runtime Scheduling (when and where kernels execute)

At execution time, a scheduler dispatches compiled kernels across available
hardware. This level handles:

- Heterogeneous compute (CPU vs GPU vs accelerator)
- Dependency ordering between kernels
- Load balancing across execution units
- Multi-graph workload management

Making this a runtime decision (rather than compile-time) gives flexibility for
varying hardware configurations and concurrent workloads.

## The Matmul Pattern (and why we don't pattern-match it)

A matmul C = A × B dissolves into: orthogonal broadcasts of A and B elements
into elementwise multiplications into sum reductions. In the NanoGraph:

- Multiply atoms broadcast A[i,k] across all columns j (via `Broadcast`)
- Multiply atoms access B[k,j] sequentially (via `Affine`)
- ReduceSum atoms accumulate across k (via `SymAffine`)

The reason tiles are so attractive for matmul: they balance the natural memory
bandwidth demands of the broadcast-followed-by-reduction pattern. A tile loads
a subblock of A and B once, computes all the multiply-accumulate ops that share
those inputs, and writes back the output subblock.

**The compiler should not recognize "this is a matmul."** Instead, it should
discover, from the DAG's input-sharing structure, that a set of atoms all
broadcast the same input values and feed into shared reductions. These atoms
belong together because co-scheduling them amortizes the cost of loading the
shared inputs. The tile shape falls out of "how large can this group get before
its working set exceeds the cache budget?"

This same logic handles matmul, convolution, attention, and any other operation
that dissolves into broadcast-multiply-reduce patterns. The compiler doesn't
need a catalog of tensor op strategies — it needs one general mechanism for
discovering shared data footprints in the scalar DAG.

## What Previous Attempts Got Wrong

Prior compiler attempts (v10, v11, early v12) failed architecturally by:

1. **Treating AtomGroups as kernel boundaries.** This leaked the lowering's
   compression decisions directly into the execution plan. A matmul lowered
   with one AtomGroup per output row produced one kernel per row — no loop
   recovery, no tiling, just a scalar interpreter compiled to native code.

2. **Pattern-matching tensor ops** instead of analyzing the scalar DAG's
   structure. This recreates the combinatorial explosion the NanoGraph was
   designed to eliminate.

3. **Assuming intermediate tensors exist in memory** with fixed layouts. The
   whole point of dissolving tensors is that intermediates can be
   register-transient or laid out however the compiler chooses.

4. **Treating loop recovery as an afterthought** rather than the central
   design problem. The NanoGraph's group structure implicitly encodes loop
   structure (structurally identical groups with contiguous atom ranges = an
   expanded loop). But the compiler shouldn't even think in terms of "recovering
   loops from groups" — it should think in terms of "which atoms share data
   footprints and should be computed together," which is a more general
   framing that subsumes loop recovery.

## Kernel Cost Estimation

Evaluating partition quality requires estimating each kernel's memory traffic
without fully scheduling it. The cost model uses structural analysis of the
addressing modes:

### Peak liveness by pattern

The peak number of simultaneously live values during execution depends on
the DAG structure within the kernel:

- **Chain** (A → B → C → D): peak ≈ 2. Each value consumed immediately.
- **Elementwise over N**: peak ≈ streamable. Process in chunks.
- **Reduction fan-in** (K inputs → 1 accumulator): peak ≈ 2. Stream inputs
  past the accumulator one at a time.
- **Broadcast fan-out** (1 → N consumers): peak ≈ chunk_size + 1. Process
  consumers in chunks while the broadcast source stays live.
- **Matmul tile** (broadcast × affine → reduce): peak ≈ N_tile (accumulators)
  + N_tile (affine input row) + 1 (broadcast value). Streams over K.

### Streaming recognition

The key to accurate estimation: **SymAffine inputs with reduce_dims indicate
a streamed dimension.** The K values aren't all live simultaneously — they're
processed one step at a time during reduction. Only one k-step's worth of
intermediate values is live at any moment.

Similarly, Affine stride-1 inputs can be processed in streaming chunks —
the full range doesn't need to be resident, just the current chunk.

### Cost formula

For a candidate kernel:
- `external_io`: count of external inputs + external outputs (floor cost)
- `peak_liveness`: estimated peak simultaneously-live values
- If `peak_liveness ≤ cache_capacity`: cost ≈ `external_io`
- If `peak_liveness > cache_capacity`: cost ≈ `external_io + spill_penalty`
  where spill_penalty grows with the excess (values evicted and reloaded)

The ratio `total_compute_ops / total_memory_traffic` is the **arithmetic
intensity** — higher is better. The partitioner should maximize this.

## Summary of Key Principles

- **AtomGroups are compression, not semantics.** The compiler reads through
  them but partitions atoms freely across group boundaries.
- **Tensor layouts are dissolved.** Intermediate data layout is a compiler
  output, not an input.
- **Memory bandwidth dominates.** Minimize cache-boundary crossings.
- **Cache is a cost, not a hard constraint.** Exceeding cache incurs streaming
  penalties but doesn't prohibit a kernel. Fragmentation into many tiny
  kernels can be worse than moderate cache pressure.
- **Shared data footprint drives grouping.** Atoms that share inputs/outputs
  belong together to amortize memory traffic.
- **Streaming dimensions reduce liveness.** Reduction dims (SymAffine) and
  sequential access (Affine) are processed one step/chunk at a time, not
  materialized fully.
- **Three separable problems:** kernel formation, intra-kernel organization,
  inter-kernel runtime scheduling.
- **No tensor op pattern matching.** One general mechanism for discovering
  data sharing structure in the scalar DAG.
