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

## Kernel Formation: The Current Problem

A **kernel** is a chunk of compiled code that shares a cache level. The
partitioner assigns groups to kernels.

### Requirements for a good partition

1. **No circular dependencies.** If kernel A reads from kernel B, then B must
   not read from A. The kernel dependency graph must be a DAG. This is required
   for correct sequential execution and for parallel scheduling.

2. **Balanced work.** No single kernel should dominate. For GPT-2, the current
   partitioner puts 79% of atoms in one mega-kernel.

3. **Minimal cross-kernel traffic.** Data that crosses a kernel boundary must
   be materialized to memory. Atoms that share inputs should be in the same
   kernel to amortize load costs.

4. **Sufficient parallelism.** Independent kernels can execute simultaneously.
   The partition should expose enough independent kernels to utilize available
   hardware parallelism.

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

**Concrete requirements:**

1. **No circular kernel dependencies.** The kernel dependency graph is a DAG.
2. **Expose parallelism.** Multiple independent kernels per sequential phase.
3. **Balanced work.** No kernel should have more than ~10-20% of total work.
4. **Minimal cross-kernel traffic.** Groups sharing data belong together.
5. **Reasonable kernel count.** 20-100 kernels for GPT-2 (45.9K groups).

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
- **No circular kernel dependencies.** The kernel DAG must be acyclic.
- **Streaming dimensions reduce liveness.** Reduction dims are processed one
  step at a time, not materialized fully.
- **No tensor op pattern matching.** One general mechanism for discovering
  data sharing structure in the scalar DAG.
