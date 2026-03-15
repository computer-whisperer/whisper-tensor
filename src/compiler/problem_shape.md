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

### What the current partitioner (nano_part_b) does wrong on GPT-2

nano_part_b uses Broadcast source analysis to detect "row slices" within
individual matmuls and separates them into independent kernels. This works
for toy examples but fails at model scale:

1. **Star topology with circular deps.** It peels off matmul rows as
   independent kernels (35 small kernels), but dumps the entire sequential
   backbone (LayerNorm, Softmax, attention glue, most matmuls) into one
   mega-kernel. The mega-kernel depends on all small kernels AND they all
   depend on it.

2. **No sequential structure awareness.** GPT-2 is a pipeline: layer 1's
   output feeds layer 2's input. The partitioner mixes groups from different
   layers in the same kernel. It should respect the sequential data flow.

3. **Interleaved group ranges.** A kernel's groups span nearly the entire
   index range (e.g., [17..30947] out of 46K total). This means the
   partitioner is splitting by matmul row (groups scattered across layers),
   not by computation phase.

### What a good partitioner should do

The key insight: the NanoGraph is a DAG with clear topological structure.
Groups earlier in topo order produce values consumed by later groups.

A good partitioner should:

1. **Respect the DAG structure.** Never create circular kernel dependencies.
   If group A (topo order before B) is in kernel 1, and group B is in
   kernel 2, then kernel 2 may depend on kernel 1 but not vice versa.

2. **Find natural phase boundaries.** In a sequential model, there are points
   where the live set narrows — between transformer layers, after attention
   completes, etc. These are natural kernel boundaries because the cross-
   kernel traffic is minimal (just the residual stream).

3. **Parallelize within phases.** Within a single matmul or attention
   computation, row-level parallelism exists. Split these into parallel
   kernels that execute simultaneously.

4. **Consider data sharing.** Groups that share Broadcast or StridedBroadcast
   inputs benefit from being in the same kernel (shared data loaded once).

5. **Produce reasonably sized kernels.** Each kernel should be large enough
   to amortize overhead but small enough to compile efficiently. For
   Cranelift JIT: each kernel should be <100K groups and <1GB of code+data.

### Practical constraints for GPT-2

- 45,921 groups total
- Sequential pipeline of 10 transformer layers
- Each layer has: LayerNorm → attention (Q/K/V projections, scores, softmax,
  output projection) → residual → LayerNorm → FFN (up + down projection) →
  residual
- The matmul structure is visible in the group DAG: Mul groups with
  StridedBroadcast inputs feeding ReduceSum groups
- The inter-layer "pinch points" where only the residual stream is live
  are the ideal kernel boundaries

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
