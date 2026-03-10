# v4_pool_growth

Fourth attempt: recover loop crystals from a whitewashed, unordered nano-op pool.

## Goal

- Test whether performant structure can be recovered after discarding original
  execution order hints.

## Structure

- `growth.rs`:
  - rebuilds expressions from SSA defs,
  - groups structurally identical stores,
  - recognizes matmul-style reductions from nano expressions,
  - topo-orders and fuses recovered loops.
- `codegen.rs`: wrapper for
  `expand -> whitewash -> grow_from_pool -> compile`.

## What Worked

- Recovered meaningful loop groups from unordered nano ops.
- Recovered matmul crystals without relying on milli-op labels in the grouping
  core.

## Limitations

- Matmul codegen path not wired in this attempt (`MatMulNotYetSupported`).
- Non-matmul elementwise performance is still behind v2.

## Result Snapshot (2026-03-09, local machine)

Command: `WT_TEST7_DIM=1024 target/release/examples/compiler_test`

- Test 2b (arithmetic chain):
  - v4: `15.239us`
  - v2: `7.389us` (v4 is `0.48x` of v2)

Command: `cargo run --release --example compiler_residual_block_bench --no-default-features --features cranelift`

- Residual gated block:
  - unsupported: `v4 matmul codegen not wired yet; recovered 3 matmul crystal groups`.
