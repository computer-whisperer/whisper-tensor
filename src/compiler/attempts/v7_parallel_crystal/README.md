# v7_parallel_crystal

Seventh attempt: turn recovered loops into explicit tile tasks, then support
both CPU task execution and Cranelift emission from those planned tasks.

## Goal

- Decouple recovery from execution strategy.
- Provide a path to single-thread, multi-thread, and JIT backends from the
  same recovered schedule/task plan.

## Structure

- `planner.rs`:
  - task planning for additive-reduction loops,
  - direct large-graph fallback (`WT_V7_DIRECT_FALLBACK_ESTIMATE`) for rank-2
    matmul artifacts when v6 synthesis would be too large.
- `executor.rs`:
  - serial and loop-parallel task execution,
  - rank-2 fast path for `sum_k(a*b)` reductions.
- `codegen.rs`:
  - dependency-ordered loop compilation,
  - reduction + pointwise emission,
  - packed-B-panel rank-2 reduction kernel path.

## What Worked

- Full plan/execute/compile pipeline for recovered loops.
- Multi-loop coverage including pointwise + reduction sequencing.
- Large jump over v2 for the current 1024 matmul benchmark in both task and
  Cranelift paths.

## Limitations

- Still far behind interpreter/BLAS baseline on large matmul.
- Planner/synthesis overhead on complex graphs remains high.
- Task scheduling is still static (no persistent workers/work stealing yet).

## Result Snapshot (2026-03-09, local machine)

Command: `WT_TEST7_DIM=1024 target/release/examples/compiler_test`

- Test 7 (`[1024x1024] x [1024x1024]`):
  - Interpreter: `8.171319ms`
  - v2 fusion: `531.317367ms`
  - v7 tasks: `178.566736ms`
  - v7 tasks mt: `27.90274ms`
  - v7 cranelift: `174.547157ms`
  - `v7 cranelift vs v2`: `3.04x`

Command: `cargo run --release --example compiler_residual_block_bench --no-default-features --features cranelift`

- Residual gated block:
  - plan: `11.775941049s` (`tasks=15`, `planned_loops=3`)
  - compile (strict): `4.424811ms`
  - execute (strict): `347.43us` (`0.34x` vs interpreter)
  - max diff: `6.08e-6`
