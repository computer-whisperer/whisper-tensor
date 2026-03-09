# v3_nano_fusion

Third attempt: keep v1's nano-op + crystal pipeline, then fuse adjacent crystals
when the consumer can reuse producer values directly.

## Goal

- Preserve the nano-op-first idea while reducing redundant loads/stores.

## Structure

- `fusion.rs`: crystal loop absorption/fusion and fusion stats.
- `codegen.rs`: pipeline wrapper
  (`expand -> crystallize -> fuse -> v1 codegen`).

## What Worked

- Demonstrated post-crystallization loop absorption.
- Reduced reloads in simple elementwise chains.

## Limitations

- Not robust on complex real-world graphs.
- Performance did not catch v2 fusion on current benchmarks.
- Still inherits v1 fragility in backend execution.

## Result Snapshot (2026-03-09, local machine)

Command: `WT_TEST7_DIM=1024 target/release/examples/compiler_test`

- Test 2b (arithmetic chain):
  - v3: `15.213us`
  - v2: `7.389us` (v3 is `0.49x` of v2)

Command: `cargo run --release --example compiler_residual_block_bench --no-default-features --features cranelift`

- Residual gated block:
  - v3: unsupported (panic during compile/execute).
