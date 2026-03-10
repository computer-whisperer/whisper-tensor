# v1_scalar_crystal

First attempt: expand milli ops to scalar NanoOps, then recover loops ("crystals")
from repeated index patterns.

## Goal

- Prove the end-to-end path:
  milli graph -> nano ops -> recovered loops -> Cranelift JIT.

## Structure

- `nano_op.rs`: re-export of shared frontend nano-op expansion.
- `crystal.rs`: loop recovery from repeating nano-op templates.
- `codegen.rs`: Cranelift lowering for scalar and crystallized loops.

## What Worked

- Correct scalar/loop execution for small elementwise workloads.
- Established core data model used by later attempts.

## Limitations

- Fragile on larger/complex graphs.
- Performance is behind the interpreter baseline.
- Panics on some realistic graphs (see residual benchmark snapshot below).

## Result Snapshot (2026-03-09, local machine)

Command: `WT_TEST7_DIM=1024 target/release/examples/compiler_test`

- Test 2 (`[32,64]` elementwise MLP):
  - Interpreter: `55.893us`
  - v1: `60.68us` (`0.92x` vs interpreter)
- Test 2b (arithmetic chain):
  - Interpreter: `12.033us`
  - v1: `17.259us` (`0.70x`)
- Test 2c (`[256,256]` elementwise MLP):
  - Interpreter: `1.701822ms`
  - v1: `2.19845ms` (`0.77x`)

Command: `cargo run --release --example compiler_residual_block_bench --no-default-features --features cranelift`

- Residual gated block:
  - v1: unsupported (panic during compile/execute).
