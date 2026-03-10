# v2_fusion

Second attempt: skip nano-op recovery and build fused loop kernels directly from
the milli graph.

## Goal

- Improve stability and speed with explicit kernel planning.
- Fuse adjacent elementwise ops and add direct GEMM emission.

## Structure

- `kernel.rs`: kernel IR (`Elementwise`, `Gemm`) and stats.
- `planner.rs`: graph walk + fusion grouping + GEMM kernel planning.
- `codegen.rs`: Cranelift emitter for elementwise loops and GEMM micro-kernels.

## What Worked

- Most robust early attempt on mixed graphs.
- Strong elementwise arithmetic performance for small tensors.
- Clean "planner -> kernel IR -> codegen" architecture.

## Limitations

- Matmul path is still far from BLAS/interpreter kernels on large sizes.
- Kernel strategy is mostly template-driven and less recovery-driven.

## Result Snapshot (2026-03-09, local machine)

Command: `WT_TEST7_DIM=1024 target/release/examples/compiler_test`

- Test 2b (arithmetic chain):
  - Interpreter: `12.033us`
  - v2: `7.389us` (`1.63x` vs interpreter)
- Test 5 (matmul MLP stress):
  - Interpreter: `37.119us`
  - v2: `116.501us` (`0.32x`)
- Test 7 (`1024x1024` matmul):
  - Interpreter: `8.171319ms`
  - v2: `531.317367ms`

Command: `cargo run --release --example compiler_residual_block_bench --no-default-features --features cranelift`

- Residual gated block:
  - compile: `1.441791ms`
  - execute: `352.302us` (`0.33x` vs interpreter `116.469us`)
  - max diff: `9.54e-5`
