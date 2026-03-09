# compiler/ — Volatile Sandbox

This directory is a **move-fast-and-break-things** zone. It will be rewritten
from scratch multiple times as we figure out the right approach.

The rest of the whisper-tensor codebase was carefully built and should be
treated with care. This module is the exception — it exists for rapid
experimentation with compilation strategies.

## Architecture (current iteration)

The compiler takes a `MilliOpGraph` and produces native code via Cranelift.

```
MilliOpGraph
    |
    v
NanoOp stream (scalar expansion of every milli op)
    |
    v
Cranelift IR (direct lowering, no optimization yet)
    |
    v
Native machine code (JIT compiled)
```

### Nano Ops

Nano ops are near-scalar operations. Each milli op decomposes into a stream of
nano ops — one per output element. When all dimensions are known at compile
time, this is fully unrolled. (Dynamic dimensions will come later.)

### Future: Crystal Growth

The current v1 lowers nano ops directly to Cranelift without optimization.
The planned approach is "crystal growth" — a re-vectorization pass where
recognized patterns absorb compatible nano ops from a working buffer, like
molecules joining a crystal lattice. This will enable cross-op fusion,
vectorization, and eventually multi-stream scheduling (CPU threads, GPU
warps, etc.).

### Nano Op Identity

Every nano op will eventually carry a deterministic ID derived from its
milli op's GlobalId + element predicate. This enables downstream consumers
to declare dependencies on specific scalar values and look up which output
crystal is responsible for them. Deferred to the crystal iteration.

## Directional Intent (v6+)

The long-term target is not a catalog of hard-coded kernel recognizers.
The compiler should recover high-performance loop schedules from generic
nano-op dataflow and access patterns.

Desired properties:

- **BLAS-like single-thread performance patterns** from recovered schedules
  (tiling, reorder, vector-friendly inner loops, register blocking).
- **No explicit MatMul special-case recognizer** in the synthesis core.
  Contraction-like behavior should emerge from generic dependence + access
  analysis and legal loop transforms.
- **Transformation-driven synthesis**: recover loops, analyze legality,
  apply reorder/split/tile/fuse/vectorization transforms, then pick schedules
  via a cost model.
- **Keep frontend stable**: milli-op -> nano-op lowering stays shared and
  iterable; current experiments may still `collect()` while schedule synthesis
  evolves.
- **Future-proof to heterogeneous backends**: the same recovered schedule
  concepts should later map to GPU workgroups/warps, not just CPU loops.

In short: derive optimized execution structure from the whitewashed nano-op
set itself, rather than routing through per-op handcrafted happy paths.

## Active Experiments

- `common/v1_frontend`: shared nano-op IR + iterator-based lowering frontend.
- `v1_scalar_crystal`: nano-op expansion + crystal loop detection.
- `v2_fusion`: direct kernel planning + elementwise fusion + matmul kernel.
- `v3_nano_fusion`: v1 nano/crystal pipeline with post-crystal loop fusion.
- `v4_pool_growth`: recover/fuse crystals from unordered nano-op pools.
- `v5_typed_synthesis`: dtype-aware schedules + software bf16 matmul kernel.
- `v6_schedule_synthesis`: generic schedule-intent recovery with no explicit matmul match,
  including affine access-role recovery and ranked schedule-candidate synthesis
  for additive-reduction families, plus initial Cranelift lowering for recovered
  reduction schedules. Schedule synthesis now runs as a streaming two-pass
  nano-op iterator (use-count pass + grouping pass) instead of collecting full
  ordered/whitewashed nano-op vectors. Current safety cap is
  `8_000_000` estimated nano ops; override with
  `WT_V6_MAX_MATERIALIZED_NANO_OPS=<n>` for intentional stress runs.
- `v7_parallel_crystal`: parallel-ready execution planning that converts recovered
  additive-reduction loops into explicit non-overlapping output tile tasks,
  plus a configurable task executor (single-thread + loop-parallel CPU mode)
  for correctness benchmarking and an
  initial Cranelift emitter for planned rank-2 `sum_k(a*b)` tile kernels.
  Current codegen supports multiple planned reduction loops, affine pointwise
  loop emission from recovered canonical terms (including unary math calls),
  dependency-ordered full-graph emission, and a reduction-only compile mode.
  Executor parallel mode currently uses per-loop barriers and row-block task
  partitioning; next steps are persistent workers/work-stealing, stronger
  vectorization, and broader reduction term coverage.
