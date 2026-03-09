# v6_schedule_synthesis

Sixth attempt: generic schedule-intent recovery from nano-op pools without
explicit matmul recognizers in the synthesis core.

## Goal

- Recover pointwise/reduction loop structure from access behavior and expression
  shape, not from hard-coded op-name templates.

## Structure

- `synthesis.rs`:
  - streaming two-pass nano-op analysis (use-count pass + grouping pass),
  - loop intent classification (`Pointwise`, `AdditiveReduction`, `Unknown`),
  - affine access-role recovery and schedule candidate generation.
- `codegen.rs`:
  - Cranelift codegen focused on supported recovered reduction kernels.

## What Worked

- Strong schedule introspection: grouped families + loop intent metadata.
- Streaming synthesis architecture avoids materializing giant nano-op vectors.

## Limitations

- Current codegen entrypoint expects a narrow supported shape
  (single supported reduction kernel), so many realistic graphs are
  synthesis-success / codegen-unsupported.
- Build/synthesis cost is still high on larger composite graphs.

## Result Snapshot (2026-03-09, local machine)

Command: `cargo run --release --example compiler_residual_block_bench --no-default-features --features cranelift`

- Residual gated block:
  - synthesis build: `12.275888362s`
  - recovered schedule families: `15` (`pointwise=12`, `reductions=3`, `unknown=0`)
  - compile: unsupported
    - `v6: no supported single additive-reduction loop recovered for codegen`
