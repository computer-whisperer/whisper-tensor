# x86_jit rewrite plan

> **Status:** Active plan for replacing the cranelift backend with a
> direct x86-64 dynasm-rs JIT. Supersedes the previous `X86_JIT_DESIGN.md`
> (the original fast-paths-first plan that produced ~7300 lines of
> spaghetti) and `X86_JIT_REWRITE.md` (the first coverage-first rewrite
> attempt that was patched too many times). Both are deleted from the
> tree; their history is in git.

## Reading order

This document only covers things the contract and codec documents
don't. Read those first:

1. **`docs/dtype_contract.md`** — what NanoGraph requires of any
   runtime. The semantic specification.
2. **`src/compiler/attempts/v14/x86_jit_codec.md`** — how this JIT
   realizes the contract on x86-64. Compute representations, the
   codec layers, the narrow-in-register primitive, the no-op
   compression principle, the BF16 fusion walkthrough.
3. **This document** — file structure, phase plan, validation
   harness, the deletion of cranelift, open questions.

If you find yourself confused about a *what* question
(what value should op X produce?), it belongs in the contract.
If you find yourself confused about a *how* question (what asm
should we emit for op X on hardware Y?), it belongs in the codec
doc. This document only handles project structure and process.

## 1. Goal

Replace the cranelift backend completely. After this plan completes:
- The cranelift dependency, the `cranelift` and `x86_compile` features,
  and `src/compiler/attempts/v14/codegen.rs` are all deleted.
- The x86-64 build's only compiled-eval backend is the new `x86_jit/`.
- That backend handles every NanoGraph the type system can express,
  including arbitrary `FloatType` configurations, sub-byte values, and
  bit-strided storage.
- The pool-based slow path (`nano_graph::pool_eval`) remains as the
  semantic reference for testing.

## 2. Non-goals

- **Per-span compile-time targets.** The original plan chased
  microsecond-per-span numbers. The rewrite chases *correctness on
  every NanoGraph the type system can express*. Compile time should
  remain orders of magnitude faster than cranelift but is not the
  primary metric.
- **SIMD vectorization.** Scalar code only. SIMD is a future
  optimization that can layer on top of the scalar architecture.
- **Auto-fusion analysis or IR-level optimization.** Fusion is a phase
  6 optimization, ported from cranelift's existing chain-builder.
- **Aarch64, Windows ABI, MSVC.** x86-64 SystemV (Linux/macOS) only.
  Future architectures get their own backend directories with the
  same layer structure.
- **Replacing `compute_layout` (slab allocation with liveness),
  `EmbeddedTables`, or the executor interface.** These stay; only the
  per-span codegen changes. The bit-aware rewrite of `BufferLayout` /
  `SlotInfo` in phase 1 is a refactor, not a redesign.
- **Quantized format support (`SimpleBlockQuant`, `KQuant`).** Out of
  scope for this rewrite. The dequant path runs before nano-op
  evaluation today; the JIT only sees `ElementStrided` slots. Future
  work may bring quantized formats into the JIT directly.

## 3. Architecture summary

Three layers, each independent of the layers above. Full description
in `x86_jit_codec.md`; here's the one-paragraph version.

**Codec layer** (`x86_jit/codec/`): three sub-modules covering bit
I/O, format conversion, and in-register precision narrowing. Each is
parameterized by `FloatType` / `IntType` properties (not named
constants) and emits zero instructions for cases where the dtype
matches the compute representation exactly. This is the layer that
realizes the contract from `docs/dtype_contract.md`.

**Op layer** (`x86_jit/ops/`): arithmetic ScalarOps emitted against
canonical compute slots (A/B/C → fixed registers). Knows nothing
about storage dtypes. Three implementations per op: F32, F64, Int.

**Orchestration layer** (`x86_jit/orch/`): the only layer that
composes. Resolves InputRef addresses, calls the codec to load/store,
calls op emitters to compute, and decides loop structure (single
group, reduce outer/inner, fused chain). This is where every
"what to emit when" decision lives.

A fourth flat-file layer (`x86_jit/mod.rs`, `prologue.rs`,
`support.rs`) handles the public API, ABI, and reject list.

## 4. Layer independence: the load-bearing structural principle

This is what the original x86_jit got wrong and what this rewrite
must get right. **Every layer is independent of the layers above
it.** Specifically:

- `codec/bit_io.rs` calls nothing in this crate except low-level
  dynasm helpers. Knows nothing about dtypes, ops, or loops.
- `codec/format.rs` calls `bit_io` and the dynasm helpers. Knows
  about dtypes but not about ops or loops.
- `codec/precision.rs` calls only dynasm helpers. Pure register-to-
  register. Knows about dtypes but not about ops, memory, or loops.
- `ops/float.rs` and `ops/int.rs` call only dynasm helpers. Operate
  on canonical compute slots. Know about compute reprs but not
  about dtypes, addresses, or loops.
- `orch/address.rs`, `orch/group.rs`, `orch/reduce.rs`, `orch/chain.rs`
  are the only files that import from the codec and ops layers. They
  compose those primitives into complete span code.

The test for whether a file is in the right place: **can I delete
the layers above it without changes?** If `codec/format.rs` can
compile and pass its unit tests with `ops/` and `orch/` deleted,
the layer separation is correct.

This independence is what makes future changes tractable:
- Adding a new ScalarOp: edit `ops/float.rs` or `ops/int.rs`. Done.
- Adding a new exotic FloatType: zero changes (codec dispatches on
  properties, not constants).
- Adding fusion: edit `orch/chain.rs`. Done.
- Adding SIMD: a new sibling directory `x86_jit/simd/` mirrors the
  scalar structure with vector versions; `orch/` chooses between
  scalar and SIMD per-group. The scalar layers are untouched.

## 5. File structure

```
src/compiler/attempts/v14/
├── X86_JIT_PLAN.md          # This file
├── x86_jit_codec.md         # Codec implementation strategy
│
├── codegen.rs               # Cranelift backend (deleted in phase 5)
├── layout.rs                # Bit-aware BufferLayout (created in phase 1)
│
├── executor.rs              # Unchanged
├── partitioner_m.rs         # Unchanged
├── plan.rs                  # Unchanged
├── report.rs                # Unchanged
├── types.rs                 # Unchanged
├── mod.rs                   # Updated to point at the new x86_jit/ module
│
└── x86_jit/
    ├── mod.rs               # Public API: X86JitSpan, executor glue
    ├── prologue.rs          # ABI: prologue, epilogue, register layout
    ├── support.rs           # check_supported (reject list)
    │
    ├── codec/
    │   ├── mod.rs           # Re-exports
    │   ├── bit_io.rs        # Read/write N bits at any bit offset
    │   ├── format.rs        # Raw bits ↔ compute repr (decode/encode)
    │   └── precision.rs     # narrow_to: in-register precision narrowing
    │
    ├── ops/
    │   ├── mod.rs           # Re-exports
    │   ├── float.rs         # Float arithmetic on compute slots
    │   └── int.rs           # Integer arithmetic on compute slots
    │
    ├── orch/
    │   ├── mod.rs           # Re-exports
    │   ├── address.rs       # compute_address: InputRef → bit offset
    │   ├── group.rs         # Emit a single group (loop or unrolled)
    │   ├── reduce.rs        # Emit a reduce (outer + inner loop)
    │   └── chain.rs         # Emit a fused chain (phase 6)
    │
    └── tests/
        ├── ab_harness.rs    # Compare x86_jit output to pool_eval
        ├── codec_bit_io.rs  # Phase 2: bit_io tests
        ├── codec_roundtrip.rs # Phase 2: format + precision roundtrip
        ├── identity_cast.rs # Phase 2: Identity and Cast across all dtypes
        ├── ops.rs           # Phase 3: every op × every compute_dtype
        ├── reduce.rs        # Phase 3: reduce per-step quantization
        ├── indirect_load.rs # Phase 3: IndirectLoad
        └── e2e.rs           # Phase 4: end-to-end model validation
```

The directory has 14 source files plus tests. No file should exceed
2000 lines; if a file approaches the limit, split it before reaching
it. The old `x86_jit.rs` was 7309 lines as a single file — that
structural failure is what the directory layout exists to prevent.

## 6. Phase plan

Each phase ends with a clean commit and a test gate. **No phase
moves forward until the gate passes.**

### Phase 0: Tear-down + scaffold + close contract gaps

**Build:**
- Delete `src/compiler/attempts/v14/x86_jit.rs` entirely.
- Delete the test code in that file (the A/B harness is the only
  salvageable bit; recreate it in `x86_jit/tests/ab_harness.rs`
  comparing against `pool_eval` instead of cranelift).
- Create the `x86_jit/` directory with the layout above. All files
  are stubs containing only the minimum needed to compile.
- `mod.rs` defines `X86JitSpan` with `compile_empty` and a
  `CompiledSpanFn` impl. `compile` returns `Err("x86_jit: rewrite in
  progress")` for any non-empty graph.
- Wire `compiled_eval.rs::compile_one_span_native` to dispatch to the
  new `x86_jit::X86JitSpan` (replacing the old import). Cranelift
  remains the fallback for everything during phases 0–4.
- Update `mod.rs` (the v14 module) to re-export the new module.

**Close contract gaps in the reference implementation:**

The dtype contract identifies known bugs in `scalar_ops` and
`pool_eval`. These need to be fixed BEFORE phase 2 because phase 2's
A/B harness uses pool_eval as the reference.

**Already done before phase 0 begins:**
- Float `IMod` semantic committed to Euclidean. `scalar_ops::modulo`
  now provides `float_imod`, `pool_eval` dispatches to it for
  `ScalarBinOp::IMod` on floats, and the milli-op lowering honors the
  ONNX `fmod` attribute uniformly for floats and integers. ONNX `Mod`
  tests including `test_mod_mixed_sign_float{16,32,64}` pass.
- Logical truthiness: `pool_eval` uses an `is_truthy(raw, dtype)`
  helper based on `decode_to_f64(raw) != 0.0` for And/Or/Xor and the
  float `Not` unary op. `-0.0` is now correctly falsy and NaN is
  correctly truthy across all logical paths.
- Signed `BitShiftRight`: `scalar_ops::bitwise` now has explicit
  `signed_shift_right` (arithmetic) and `unsigned_shift_right`
  (logical) functions, plus a unified `shift_left`. `pool_eval`
  dispatches to the right one per signedness.

**Remaining for phase 0:**
- Verify and document `scalar_ops::min` / `max` NaN handling. Update
  `dtype_contract.md` §5.3 with the verified semantic.
- Verify `signed_pow` semantics for negative exponents.
- Verify `Cast` saturating semantics for FN target types.

These fixes happen in `src/scalar_ops/` and `src/nano_graph/pool_eval.rs`,
not in `x86_jit/`. They are part of phase 0 because the rewrite has no
correct reference to test against without them.

**Gate:**
- `cargo build --features x86_compile` succeeds.
- `./scripts/check-all.sh` passes.
- All existing tests pass (because x86_jit refuses every non-empty
  span and falls back to cranelift; the only x86_jit-emitted spans are
  empty ones).
- Conformance gaps in §8 of `dtype_contract.md` items 1, 2, 3 are fixed
  (the clear bugs); items 4, 5, 6 are verified and documented.

### Phase 1: Bit-aware layout

**Build:**
- **Step 1.A**: Extract `BufferLayout`, `SlotInfo`, `compute_layout`,
  `EmbeddedTables`, `read_buffer_to_output`, `write_store_slice_to_buffer`,
  `populate_literals`, and related layout machinery from
  `codegen.rs` into a new `src/compiler/attempts/v14/layout.rs`.
  No behavioral changes. Both `codegen.rs` and `x86_jit/` import from
  `layout.rs`.
- **Step 1.B**: Rewrite `SlotInfo` to be bit-addressed:
  ```rust
  pub struct SlotInfo {
      pub atom_base: AtomId,
      pub count: u64,
      pub bit_offset: u64,    // first element's bit position from buffer base
      pub bit_stride: u64,    // bits between consecutive elements
      pub elem_bits: u64,     // = dtype.total_bits()
      pub dtype: NumericDType,
  }
  ```
  Helpers `byte_offset()`, `bit_in_byte()`, `is_byte_aligned()` are
  derived. The slab allocator (`FreeList`) tracks bits internally;
  per-slot start positions are byte-aligned in phase 1 (bit-packing
  is a phase 6 optimization).
- Update `compute_layout` to compute slot positions in bits.
  Element-bit-aligned slot starts; the buffer's total bit size rounds
  up to a byte boundary.
- Update `write_store_slice_to_buffer` and `read_buffer_to_output` to
  handle bit-strided source `TensorLayout::ElementStrided` (using the
  source's `offset_bits` and `strides`). For sub-byte source dtypes
  the marshalling code does the bit math.
- Update `populate_literals` to use the bit-aware write path.
- Update the cranelift codegen in `codegen.rs` to consume the new
  `SlotInfo`. For byte-aligned slots (every test case in the current
  codebase), the cranelift code can convert `bit_offset / 8` and use
  byte arithmetic internally. For sub-byte / non-aligned slots,
  cranelift bails with an explicit error — no test exercises this
  today, and cranelift will be deleted in phase 5 anyway.

**Gate:**
- All existing tests pass byte-for-byte against the pre-rewrite
  output. The slab allocator may pack identically (since slot starts
  remain byte-aligned), so the buffer offsets shouldn't change for
  any byte-aligned dtype.
- New layout-level unit tests cover:
  - Bit-strided `TensorLayout::ElementStrided` input → buffer write →
    buffer read → output. Round-trip is bit-equal.
  - `populate_literals` with sub-byte dtypes (I4, U4, F4E2M1, Bool):
    write a literal, read it back via the bit-aware read path, assert
    bit-equality with the source value.
  - `compute_layout` for a graph containing sub-byte slots: the
    resulting `bit_offset` / `elem_bits` are correct.
- `./scripts/check-all.sh` passes.

### Phase 2: Codec foundations + Identity / Cast

**Build:**
- `codec/bit_io.rs`:
  - `emit_load_bits(addr_reg, n_bits) → raw bits in rax`
  - `emit_store_bits(addr_reg, n_bits, src_reg)`
  - General bit-extraction / bit-write at any bit offset, no
    byte-aligned short-circuit yet.
- `codec/format.rs`:
  - `emit_decode(dtype, raw_gp_reg, compute_repr, dst_slot)` for
    every `NumericDType` parameterization. Floats specialized on
    `(e_bits, m_bits, has_inf, has_nan)`. Ints on `(bits, signed)`.
    Native types emit zero or one instruction; non-native types emit
    the general inline decoder.
  - `emit_encode(dtype, src_slot, compute_repr, dst_gp_reg)` symmetric.
- `codec/precision.rs`:
  - `emit_narrow_to(dtype, slot, compute_repr)`. No-op when dtype
    matches compute repr exactly. Otherwise emits the in-register
    rounding sequence per the recipe in `x86_jit_codec.md` §3.4 / §3.5.
- `orch/address.rs`:
  - `compute_address(input: &InputRef, layout, atom_offset, iter_var,
    addr_reg) → (NumericDType, u64 elem_bits)`. Materializes the
    bit offset into `addr_reg`. Handles all `InputRef` variants
    (Broadcast, Strided 1D / 2D / N-D, Explicit single, Explicit
    multi). General N-D path; no special-case fast paths.
- `orch/group.rs`:
  - `emit_group(group, layout, ...)` → emit a single-group loop or
    unrolled body for `Identity` / `Cast` / `Literal` / `LiteralSpan`.
- `prologue.rs`: ABI + register layout. Reserve registers per the
  compute slot table, the address-resolution scratch, and the codec
  scratch. Document spill slots.
- `support.rs`: accept `Identity`, `Cast`, `Literal`, `LiteralSpan`.
  Reject everything else with a clear message.

**Gate:**

Phase 2 has three layers of tests, smallest first.

**Layer 1: codec unit tests** (in `x86_jit/tests/`).

A/B against `pool_eval` for Identity and Cast on:

1. **Every named `NumericDType`** as both source and destination
   dtype, all cross combinations (~400 cells in the matrix).
2. **Every `InputRef` variant**: Broadcast, Strided nd=1, nd=2, nd=3,
   nd=4, Explicit single, Explicit multi.
3. **Bit alignments**: for every dtype, run a load/store roundtrip at
   bit offsets 0, 1, 2, ..., 7 within a byte. For dtypes with
   `total_bits ≥ 8`, this exercises the spanning-byte case.
4. **Single-element groups (`count=1`) and counted-loop groups
   (`count > 1`)**, both `IterVar::ConstantIndex` and
   `IterVar::OuterLoop` paths.

Plus codec roundtrip tests:

5. **For every `FloatType`**: ~1000 representative values (normal,
   ±0, denormal, ±inf when applicable, NaN when applicable, max/min
   finite, rounding edges) → `emit_decode` → `emit_encode` → assert
   bit-equality with the input (or with `cast_raw` for cases where
   re-encoding loses precision).
6. **For every `IntType` with both signednesses**: full range (or
   sampled), positive, negative, boundary values, sign-extension cases.
7. **`narrow_to` roundtrip**: for every `(compute_repr, target_dtype)`
   pair where target fits in compute repr, generate values, narrow,
   re-narrow, assert idempotent. Compare against
   `cast_raw(value, target_dtype, target_dtype)`.

**Layer 2: `src/test_set/` cast and elementwise subsets.** Run the
existing `test_set::cast` and `test_set::elementwise` cases through
the new x86_jit (a new `run_case_via_x86_jit` runner) and assert
byte equality with the existing pool_eval runners. Identity/Cast/
Literal coverage in those cases is the integration test.

**Layer 3:** RWKV-0.1B runs end-to-end with x86_jit handling
Identity/Cast/Literal and cranelift handling everything else. No
regressions vs the current cranelift-only baseline. (RWKV-0.1B is
fast enough on most backends to use as a phase 2 smoke test.)

100% of layers 1, 2, and 3 pass. The conformance test categories in
`docs/dtype_contract.md` §7 are satisfied for Identity and Cast.

### Phase 3: All ScalarOps

**Build:**
- `ops/float.rs`: emit_binop_float, emit_unop_float for every
  `ScalarBinOp` and `ScalarUnaryOp`, dispatched on `compute_repr`
  (F32 / F64). Read inputs from canonical slots A/B/C, write
  output to A.
- `ops/int.rs`: emit_binop_int, emit_unop_int for the integer
  paths. Wrapping arithmetic per `dtype_contract.md` §5.2.
- `orch/group.rs`: extend to handle Binary, Unary, Select.
- `orch/reduce.rs`: emit reduce as outer atom loop (when count > 1)
  + inner k loop. Per-iteration: load via codec, fold via op layer,
  `narrow_to(compute_dtype)` after the fold to enforce per-step
  quantization.
- `orch/group.rs`: extend to handle IndirectLoad. Calls compute_address
  on the index input, multiplies and adds the table base, calls
  codec to load + decode.
- `support.rs`: accept all ScalarOps except `OpaqueOutput`.

**Gate:**

Phase 3 gates progressively from unit to integration to end-to-end.

**Layer 1: op unit tests** (in `x86_jit/tests/`). A/B against
`pool_eval`:

1. **Every `ScalarBinOp` × every `compute_dtype` × representative
   input shapes**. Each combination at least 100 random inputs
   covering normal/edge/special values per dtype.
2. **Every `ScalarUnaryOp` × every `compute_dtype`**. Same.
3. **Select × every `output_dtype`** with cond as float, int, and bool;
   covering truthy/falsy/-0/NaN cond values.
4. **Reduce × every `(kind, compute_dtype, output_dtype)` triple**.
   Specifically including BF16 / F16 / F8x reduces of 100+ elements
   to validate per-step quantization against pool_eval (which uses
   per-iteration `cast_raw` to enforce the same contract).
5. **IndirectLoad × {table dtype, index dtype, output dtype}**
   cross product, with random and edge index values.

**Layer 2: full `src/test_set/`.** All cases (cast, composite, conv,
dtype_discipline, elementwise, matmul, opaque_ops, pad, reduce,
structural) pass via the x86_jit runner with byte-equal output to
the pool_eval runners. This is the primary correctness gate at the
graph-pattern level.

**Layer 3: ONNX op tests** (`tests/onnx_testing.rs`). The full
`pool_test_*` suite passes via x86_jit. These are the contract
conformance tests at the standard-spec level.

**Layer 4: RWKV-0.1B end-to-end.** Runs with x86_jit handling
**every** span and zero cranelift fallbacks. The `x86_jit_stats`
coverage report shows 100% accept.

100% of all four layers pass. Any `pool_eval` mismatch is a phase 3
bug to fix before moving on.

### Phase 4: Full coverage validation

**Build:**
- A new env var `X86_JIT_VALIDATE=1` that runs every span through
  *both* x86_jit and pool_eval, comparing buffer outputs bit-by-bit.
  Mismatches print the span's group structure and the differing
  values for debugging. (This validation harness already exists for
  cranelift; port it to use pool_eval as the reference.)
- Run RWKV-0.1B with `X86_JIT_VALIDATE=1` and full performance
  measurement. RWKV-0.1B is the primary end-to-end correctness +
  performance test. Fix any divergences.
- Run other small models the test set covers (GPT-2 already validated
  bit-perfect via `project_nano_gpt2_bitperfect.md`). Fix divergences.
- Tighten `support.rs` to reject only `OpaqueOutput` and graph-shape
  errors. Anything else that rejects is a phase 4 bug.

**Gate:**
- RWKV-0.1B runs end-to-end with `X86_JIT_VALIDATE=1` and produces
  zero buffer mismatches against pool_eval.
- GPT-2 runs successfully under the new backend, bit-equal to its
  established baseline.
- `support.rs::check_supported` rejects only `OpaqueOutput`.
- The `x86_jit_stats` coverage report shows 100% accept on every test
  model.
- RWKV-0.1B execution time is recorded as the baseline performance
  for phase 6 to optimize against.

### Phase 5: Delete cranelift

**Build:**
- Delete `src/compiler/attempts/v14/codegen.rs`.
- Delete `cranelift-*` crates from `Cargo.toml` `[dependencies]`.
- Delete the `cranelift` and `x86_compile` features from
  `Cargo.toml`. The new backend is unconditional on x86-64 builds.
- Update `compiled_eval.rs::compile_one_span_native` to call
  `X86JitSpan::compile` directly with no fallback.
- Update CLI feature wiring: the `wt` binary gets x86_jit support
  unconditionally on x86-64.
- Capture pool_eval reference outputs as snapshot tests for the new
  backend (this is what the unit tests gate against from now on).
- Delete `compile_span_validated` and any other cranelift-specific
  validation infrastructure.
- Update memory files / docs that reference the cranelift backend.

**Gate:**
- `cargo build` (no features) succeeds.
- `cargo build --all-features` succeeds and produces no
  cranelift-related warnings.
- All tests pass.
- `./scripts/check-all.sh` passes.
- RWKV-0.1B runs unchanged in execution behavior.

### Phase 6: Optimization

Everything in phases 0–5 is correctness work. Phase 6 is the only
phase that adds optimization. Each optimization must:

1. Be a *bypass* of the default path, not a parallel path. Disabling
   the optimization (via env var) leaves a fully correct backend.
2. Be A/B validated bit-equal against the unoptimized form on a test
   matrix of inputs.
3. Have a measured wall-time improvement on RWKV-0.1B. Optimizations
   that save < 1% should be reverted as not worth the maintenance.

**Approximate order** (re-prioritize after profiling):

1. **Profile** RWKV-0.1B with the naive backend. Identify the top
   hot spans and the per-instruction cost breakdown.
2. **Byte-aligned short-circuits in `bit_io.rs` / `format.rs`**: when
   the slot's `bit_offset`, `bit_stride`, and `elem_bits` are all
   statically multiples of 8, replace the bit-extraction path with a
   single `movss` / `movsxd` / etc. The most-impactful optimization
   for typical models because almost everything is byte-aligned.
3. **Direct displacement folding for `IterVar::ConstantIndex`**: skip
   `compute_address` entirely when the iteration index is a constant.
4. **1D affine SIB encoding**: when the address is 1D affine and
   byte-aligned, emit `[r12 + r13*scale + base]` directly in the load
   instead of materializing the bit offset into a register.
5. **Broadcast hoisting**: when an input's address is constant per
   loop iteration, hoist the address computation out of the loop.
6. **Address caching**: when the same `InputRef` is used by multiple
   ops in the same iteration, materialize once and reuse.
7. **Fused chains** (`orch/chain.rs`): port from cranelift's
   `build_fusion_chains` + `emit_chain` + `emit_group_body_forwarded`,
   with `narrow_to` between consecutive groups per the contract.
8. **Reduce-fold inlining**: port from cranelift's `inlinable[gi]`
   handling. The producer expression is evaluated inside the reduce's
   inner k loop, with no buffer materialization for the producer.
9. **2D Strided power-of-2 modulus shortcut**: use shr/and instead
   of div/mod when the inner dim is a power of 2.
10. **F16C inline F16 conversions**: when the build target enables F16C,
    use `vcvtph2ps` / `vcvtps2ph` instead of the inline software codec.
11. **Bit-packed slab allocator**: when sub-byte slots have many
    consecutive small elements, pack them into shared bytes in the
    working buffer to reduce memory traffic.

Each optimization gets its own commit with profile measurements
attached.

## 7. Validation strategy

### 7.1 What we validate against

- **Phases 0–4**: validate against `pool_eval`, NOT against cranelift.
  The contract identifies cranelift as having known bugs in its
  fusion path (`emit_store_load_roundtrip`); using it as a reference
  would let those bugs propagate. Phase 0 fixes pool_eval's known
  bugs so it can be a correct reference.
- **Phase 5+**: validate against snapshot outputs captured from
  pool_eval during phase 4. These snapshots are the fossil record
  and live in `x86_jit/tests/snapshots/`.
- **Phase 6**: each optimization additionally validates the optimized
  output against the unoptimized output (with the optimization env
  var disabled) for byte equality.

### 7.2 Test corpora (in order of how big a hammer they are)

Each later corpus is a proper superset of the earlier ones in coverage
intent. Bugs caught later usually mean we missed a case in an earlier
corpus and should add a regression test there.

1. **Codec unit tests** (phase 2 onward, in `x86_jit/tests/`):
   `codec_bit_io.rs`, `codec_roundtrip.rs`, `identity_cast.rs`. Test
   the codec layer functions in isolation, without going through ops
   or orchestration. Per-dtype, per-bit-alignment, per-InputRef-variant.
2. **Op unit tests** (phase 3 onward, in `x86_jit/tests/`): `ops.rs`,
   `reduce.rs`, `indirect_load.rs`. Build a one- or two-group graph
   for each `(ScalarOp, compute_dtype)` combination and A/B against
   pool_eval.
3. **`src/test_set/`** (phases 2–4): the existing test set (cast,
   composite, conv, dtype_discipline, elementwise, matmul, opaque_ops,
   pad, reduce, structural). Tighter and faster than full models, but
   exercises real graph patterns end-to-end. Run via the existing
   `test_all_cases_via_pool_eval` and `test_all_cases_via_graph_pool_eval`
   harnesses, with a new `test_all_cases_via_x86_jit` runner once the
   backend handles enough ops. **This is the primary phase 2 / 3 gate
   alongside the unit tests.**
4. **ONNX op tests** (phases 3–4, in `tests/onnx_testing.rs`): the
   ONNX standard test set, runnable via the `pool_test_*` harness.
   Already passes for pool_eval; gate is "passes for x86_jit too."
5. **End-to-end model tests** (phase 4 onward): RWKV-0.1B and other
   small models with `X86_JIT_VALIDATE=1`. RWKV-0.1B specifically
   should be reasonably fast on most backends and is the primary
   end-to-end correctness + performance test. Larger model loaders
   (`tests/llama3_model_loading.rs` etc.) are smoke tests for
   loading, not performance tests.
6. **Optimization equivalence tests** (phase 6 onward): each fast
   path has a test that runs the same span with and without the
   optimization and asserts byte equality.

**Start small.** Phase 2 can run codec unit tests + the cast and
elementwise subsets of `src/test_set/` to validate Identity/Cast.
Phase 3 expands to the full test_set and the op unit tests. Phase 4
adds ONNX op tests and the model end-to-end runs.

### 7.3 Conformance with `dtype_contract.md` §7

The conformance test categories in §7 of the contract map to:
- §7.1 cast roundtrip → covered by `codec_roundtrip.rs` (phase 2)
- §7.2 per-op roundtrip → covered by `ops.rs` and `src/test_set/` (phase 3)
- §7.3 reduce per-step quantization → covered by `reduce.rs` and
  the `src/test_set/reduce.rs` cases (phase 3)
- §7.4 optimization equivalence → covered by phase 6 tests

Phase 4 doesn't complete until all four categories are passing for
the new x86_jit backend on the full named-NumericDType matrix.

## 8. Compile-time vs execute-time

The original cranelift backend was slow at compile time (~14ms per
span × 1880 spans = 26 seconds for RWKV-0.1B) AND wrong on certain
fusion paths. The original x86_jit attempt fixed compile time but
had the wrong architecture.

This rewrite optimizes for **correctness first, execute-time second,
compile-time third**. Concretely:

- **Compile time**: should remain orders of magnitude faster than
  cranelift. The new backend doesn't build an SSA IR, doesn't run a
  register allocator, and doesn't run multi-pass optimizations. Per-
  group emission is essentially a single pass. We accept whatever
  compile time results from honest implementation of the contract;
  we don't chase microsecond targets.
- **Execute time**: the primary metric. The architecture's no-op
  compression property (per `x86_jit_codec.md` §4.1) ensures that
  aligned F32 chains compile to bare loads/stores/ALU with no
  overhead. Optimizations in phase 6 add SIMD-free improvements
  ordered by profiled impact.
- **Correctness**: non-negotiable. Bit-identical output to pool_eval
  for every span the rewrite supports. The phase gates enforce this.

If a compile-time-vs-execute-time tradeoff arises in implementation
(e.g., "should we cache an address materialization across iterations
even though it complicates the orchestration?"), the answer is
"do whatever produces faster execute time, as long as compile time
stays well below cranelift's baseline."

## 9. What this plan rules out

The following are explicitly forbidden. Each rules out a class of
mistakes the previous attempts made.

1. **Adding ScalarOp support outside the address / codec / op /
   orchestration pipeline.** No bespoke per-op load/store paths.
2. **Hardcoding `NumericDType::F32` (or any other named constant) in
   codec helpers.** Dispatch is on `FloatType` properties (`e_bits`,
   `m_bits`, `has_inf`, `has_nan`) and `IntType` width/signedness as
   data. The codec must work for any future exotic type without code
   changes.
3. **Byte addressing in any default path.** All addressing is in bits.
   Byte-aligned shortcuts are phase 6 optimizations that bypass the
   bit path. The general path emits bit arithmetic.
4. **Extern "C" trampolines for type conversion.** The codec is
   inline bit manipulation. (Trampolines for transcendental math fns
   like `expf` / `logf` / `sin` remain — reimplementing libm in
   inline asm is out of scope.)
5. **Two parallel code paths that compute the same thing
   differently at different optimization levels.** A fast path is a
   bypass that produces the *same* result as the default path.
6. **`check_supported` growing in phase 4 or later.** It only shrinks.
7. **Files growing past 2000 lines.** Split before reaching the limit.
8. **Cross-layer imports.** `codec/` doesn't import from `ops/`.
   `ops/` doesn't import from `orch/`. Only `orch/` composes.
9. **Validation against the cranelift backend during phases 0–4.**
   The reference is pool_eval (after its bugs are fixed in phase 0).
10. **Skipping phase 0 contract-gap fixes.** Phase 2's A/B harness
    cannot work without a correct reference; the gaps in the contract
    document §8 must be closed before phase 2 starts.

## 10. Open questions

These are flagged for resolution as the rewrite progresses. None
block phase 0.

1. **Fixed register layout vs. minimal register tracking in
   orchestration.** The compute slot model assigns A→xmm0 (or rax),
   B→xmm1 (rcx), C→xmm2 (rdx) statically. For fused chains in phase
   6, we may need a slightly more flexible model where forwarded
   intermediates don't always live in slot A. **Recommendation:**
   start with strict static slots in phases 2–5; introduce minimal
   register tracking only if profiling phase 6 shows a meaningful
   win. Document any new mechanism in `x86_jit_codec.md`.

2. **Reduce inner-k loop register reservation.** The outer loop uses
   r13 (atom index) and r14 (atom end). The reduce inner loop needs
   its own counter (e.g., r11) and a separate address-compute scratch
   (e.g., r15) so the inner load doesn't clobber the outer state.
   The exact split should be locked in `prologue.rs` before phase 3
   starts. The codec scratch register budget needs to fit alongside
   this.

3. **Codec scratch register budget.** The general float decode needs
   ~6 GP scratch registers; the general float encode needs ~7 (with
   the rounding bias). The bit-extraction load itself uses
   rax/rsi/rcx/rdi. Together that's tight against the existing op
   pipeline. Phase 0's `prologue.rs` should reserve enough spill
   slots to handle the worst case.

4. **F16C runtime detection.** Should we runtime-detect CPUID and
   pick between inline F16C and the general inline codec, or
   static-pick at compile time? **Recommendation:** static-pick for
   now (build flag). Revisit in phase 6 if profiling shows F16-heavy
   workloads warrant runtime detection.

5. **Sub-byte input/output marshalling.** `write_store_slice_to_buffer`
   and `read_buffer_to_output` currently take a `StoreSlice` /
   `SpanOutput` described in bytes. To handle bit-strided source /
   destination tensors, these structs may need bit-aware variants.
   Resolve in phase 1.

6. **Validation against pool_eval performance.** Phase 4's
   `X86_JIT_VALIDATE=1` runs every span through both backends. This
   may be slow on RWKV-0.1B. If it's prohibitively slow, we may need
   a sampling mode that validates a random subset of spans per run.

These resolve naturally as the code is written; none require
upfront design decisions.
