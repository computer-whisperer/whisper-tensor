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

## Active Experiments

- `common/v1_frontend`: shared nano-op IR + iterator-based lowering frontend.
- `v1_scalar_crystal`: nano-op expansion + crystal loop detection.
- `v2_fusion`: direct kernel planning + elementwise fusion + matmul kernel.
- `v3_nano_fusion`: v1 nano/crystal pipeline with post-crystal loop fusion.
- `v4_pool_growth`: recover/fuse crystals from unordered nano-op pools.
- `v5_typed_synthesis`: dtype-aware schedules + software bf16 matmul kernel.
