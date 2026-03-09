# v5_typed_synthesis

Fifth attempt: typed kernel synthesis on top of recovered crystals, with a
software BF16 matmul path (no BLAS dependency).

## Goal

- Prove dtype-aware synthesis and software fallback kernels (especially BF16).

## Structure

- `synth.rs`:
  - typed planning from pool-recovered matmul crystals,
  - blocking selection from a hardware profile,
  - BF16 strategies:
    - `OnTheFlyConvert`
    - `PackBPanelF32`,
  - software execution path `BF16 x BF16 -> F32`.

## What Worked

- Practical software BF16 kernel without hardware BF16 assumptions.
- Panel-packed B strategy measurably reduces conversion overhead.

## Limitations

- Scope is mainly matmul kernel synthesis/execution.
- Not yet integrated as a full graph-wide general scheduler.

## Result Snapshot (2026-03-09, local machine)

Command: `WT_TEST7_DIM=1024 target/release/examples/compiler_test`

- Test 6 (`[96x128] x [128x96]`, BF16 synthesis):
  - `OnTheFlyConvert`: `499.893us` (diff `0.00e0`)
  - `PackBPanelF32`: `434.347us` (diff `0.00e0`)
  - Packed-B reduced runtime vs on-the-fly conversion in this run.
