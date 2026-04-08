# X86 JIT — direct x86-64 codegen via dynasm-rs

## Why

The Cranelift codegen path is overkill for what we emit. Per the
RWKV-0.1B compile profile (`COMPILE_PROFILE=1`):

| stage | time | notes |
|---|---|---|
| `compute_layout` | 0.48 s | (after Win A) |
| `ir_build` | 1.94 s | building Cranelift IR |
| `cranelift_def` | 24.2 s | machine codegen, regalloc, sched |
| `cranelift_finalize` | 0.02 s | |
| **`compile_phases` total** | **26.8 s** | 1880 spans, ~14 ms each |

We emit straight-line scalar loops:
- per-iteration: a few loads, an arithmetic op, a store
- counted loop scaffolding (`iadd_imm` / `icmp` / `brif`)
- occasional `callq` to extern math fns (expf, logf, tanhf, etc.)

We use **none** of Cranelift's interesting capabilities — instruction
selection is 1:1, no GVN/LICM/inlining/scheduling/vectorization. The
mapping from our IR to x86-64 is mechanical.

A direct emitter via `dynasm-rs` should compile each span in
*microseconds* instead of milliseconds. Target: drop the 24 s of
`cranelift_def` to <0.5 s.

## Goals

1. Drop-in replacement for `JitCompiledSpan` in
   `compile_nano_graph` — same `CompiledSpanFn` trait, same
   `BufferLayout`, same `EmbeddedTables`.
2. Sub-millisecond compile time per span (target).
3. Match Cranelift exec performance within ±10% on the RWKV-0.1B
   workload.
4. x86-64 SystemV (Linux/Mac) only initially.
5. Validation harness that A/Bs every span against Cranelift output
   bytewise during development.
6. **Become the only compiled-eval backend.** Cranelift is a
   transitional fallback during rollout — once op coverage is
   complete and validated, the cranelift dependency, feature, and
   `JitCompiledSpan` type all get deleted.

## Non-goals

- aarch64, Windows ABI, MSVC — follow-ups behind separate features.
- IR-level optimization (GVN, LICM, vectorization).
- Auto-vectorization. We emit scalar code.
- Replacing the existing `BufferLayout`, `compute_layout`,
  fusion-chain detection, or reduce-fold inlining. Those layers
  stay; only the per-span codegen changes.

## Scope of op coverage

**Op coverage** (`ScalarOp` variants):

- `Literal` / `LiteralSpan` — no codegen (values populated by
  `populate_literals` outside the JIT path)
- `Identity`, `Cast` (both saturating and non-saturating)
- `Binary` — all `ScalarBinOp` variants
- `Unary` — all `ScalarUnaryOp` variants (math fns reach extern C)
- `Select` (3-input ternary)
- `Reduce { kind: Sum | Max | Min | Prod }`
- `IndirectLoad { table_base }` (embedding gather)
- `OpaqueOutput` — N/A; opaque-op spans route to `PoolEvalSpan`
  before reaching codegen.

**InputRef variants**:
- `Strided` 1D affine (`dim_strides=[s]`, `dim_shape=[MAX]`)
- `Strided` 2D modular and 2D strided-broadcast
- `Strided` general N-D
- `Broadcast` (single atom)
- `Explicit` (per-atom array via embedded table)

The rollout below introduces these incrementally; spans that contain
unsupported variants fall back to Cranelift until each phase lands.

## Dtype handling — parameterized over `NumericDType`

The existing `NumericDType` already encodes everything we need:

```rust
enum NumericDType {
    Float(FloatType),       // { exponent_bits, mantissa_bits, has_infinity, has_nan }
    SignedInt(IntType),     // { bits }
    UnsignedInt(IntType),   // { bits }
    Bool,
}
```

The codegen dispatches on the *shape* of the dtype, not the named
variant. We have three concentric paths:

### 1. Native fast path (F32, F64; I8/16/32/64; U8/16/32/64; Bool)

Direct SSE/integer instructions, parameterized by element width:

- **F32**: `movss` / `addss` / `mulss` / … on xmm regs
- **F64**: `movsd` / `addsd` / `mulsd` / … on xmm regs
- **Integers**: `mov`/`add`/`imul`/`cmp` etc. The operand size is
  picked from `IntType::bits` (8 → `al/cl`, 16 → `ax/cx`, 32 →
  `eax/ecx`, 64 → `rax/rcx`); sign-vs-zero extension on load picks
  `movsx` vs `movzx`.
- **Bool**: `movzx` to 32-bit, `test`/`setcc` for compares.

This is one parameterized emission path per *category* (float-32,
float-64, int-of-width-N), not one per named dtype. Adding `I16` is
free once the int path exists.

### 2. Half-width float fast path (F16, BF16)

Stored at native width, computed in F32. The expand/narrow pair is
the only special handling required:

- **BF16 → F32**: `movzx eax, [mem]; shl eax, 16; movd xmm0, eax`
  (3 instructions; the bit pattern is the high half of an F32).
- **F32 → BF16**: round + truncate, `movd eax, xmm0; shr eax, 16;
  mov [mem], ax` (RTNE: add 0x7fff + ((bits>>16) & 1), then shift).
- **F16 → F32**: bit-twiddle expansion (~10 instructions), or use
  `vcvtph2ps` if F16C is available at runtime. Default to the
  software path for portability.
- **F32 → F16**: software round-to-nearest-even (~15 instructions),
  or `vcvtps2ph` if F16C present.

After expand/narrow, all arithmetic uses the F32 native path. No
duplicated op-emission code.

The F16C feature check is a one-time runtime cpuid probe; we set a
global flag and emit the fast path when set. F16C is widely
supported (Intel ≥ Ivy Bridge 2012, AMD ≥ Bulldozer 2011) so the
software fallback is mostly for portability rather than performance.

### 3. Inline encode/decode for exotic floats (F8E4M3FN, F8E5M2,
###    F4E2M1, F6E3M2, F6E2M3, plus any future arbitrary FloatType)

These are tiny — at most 8 bits — and the encode/decode logic is
pure bit shuffling. We emit it inline; no FFI. There are no slow
paths in this backend.

After load/decode the value lives in F32 (or F64 if compute_dtype
is F64). Compute and store reuse the native fast path.

#### Decode (narrow → F32)

For any format with `total_bits ≤ 12`, we precompute a lookup table
at codegen time and embed it via the existing `EmbeddedTables`
mechanism. Indexed by raw bits, value is the F32 representation:

```asm
movzx  eax, byte [r12 + atom_addr]            ; raw narrow bits
mov    edx, dword [r12 + decode_table + eax*4]  ; lookup
movd   xmm0, edx                                ; → F32
```

3 instructions, no branches, all special cases (zero, ±inf, NaN,
denormal) baked into the table. The table is built at codegen time
by calling the existing `FloatType::decode_to_f64` (then casting to
f32) over the full 2^total_bits domain. Tables are deduplicated
across spans by `(exp_bits, mant_bits, has_infinity, has_nan)`.

For F8 formats: 256 × 4 B = 1 KB per format. F4/F6: ≤256 B. The
total embedded-table footprint per format is negligible.

If we ever add a >12-bit exotic float (none planned), the codegen
falls back to the inline bit-manipulation decode below.

#### Encode (F32 → narrow)

Inline parameterized bit math. Single emitter takes
`(exp_bits, mant_bits, has_infinity, has_nan)` and produces the
correct instruction sequence. Sketch for the normal case:

```asm
movd    eax, xmm0                  ; F32 bits
mov     ecx, eax
shr     ecx, 31
shl     ecx, total_bits - 1        ; sign bit, pre-positioned
mov     edx, eax
shr     edx, 23
and     edx, 0xff                  ; F32 biased exponent
sub     edx, 127 - bias            ; rebias to narrow
; … overflow / underflow / RTNE branches …
shl     edx, mant_bits             ; exponent in narrow position
mov     esi, eax
shr     esi, 23 - mant_bits        ; mantissa, shifted + rounded
and     esi, (1 << mant_bits) - 1
or      ecx, edx
or      ecx, esi
mov     [r12 + out_addr], cl       ; store narrow bits
```

Special-case branches: overflow → emit NaN (`has_nan`) or saturate
to max-finite, underflow → zero or denormal, NaN input → propagate
NaN. The emitter tracks which branches are needed from the
`FloatType` flags — formats without infinity skip the inf-overflow
branch, formats without NaN skip the NaN-input branch, etc.

Round-to-nearest-even takes ~6 extra instructions (round bit +
sticky OR + add-and-mask). The whole encoder per-format lands at
30–50 instructions.

#### Why not extern calls

The earlier draft had `jit_decode_float` / `jit_encode_float`
extern C wrappers. Rejected: call overhead (~20+ cycles for
call+ret + xmm0 save/restore around the call) is much worse than
the table lookup or inline math, and inlining keeps everything in
one register file. The conversion library in `conversions.rs`
stays as the source of truth for table contents and as the
correctness oracle for unit tests.

#### Adding a new exotic FloatType

Zero codegen changes if `total_bits ≤ 12`: the decode-table builder
walks the new format automatically and the parameterized encoder
already handles arbitrary `(exp_bits, mant_bits, has_*)`. The unit
test matrix needs the new format added.

### 4. Sub-byte integers (I4, U4)

Two-per-byte packing. Load: `mov al, [mem]; shr al, 4*offset; and
al, 0x0f`; sign-extend if signed (`cbw`/`movsx`). Store similarly
masked. Address arithmetic divides by 2 for the byte offset.

Sub-byte handling is added in the same phase as the int width
generalization — once the int path is parameterized over `bits`,
sub-byte is just `bits < 8`.

### Compute dtype selection

`ScalarOp::Binary { compute_dtype }` and friends already specify
the compute dtype. The codegen just trusts it:

- compute=F32 or compute=F64 → native float path
- compute=any int → int path of that width
- compute=Bool → 32-bit int path with set/cmp

Storage dtypes can be different from compute dtypes — the existing
`emit_cast_to_output` pattern stays. Cast is just a load+store
through the pipeline (decode → cvt → encode).

## Function signature & calling convention (System V)

```
extern "C" fn span_main(buffer: *mut u8) -> ()
```

- `rdi` = buffer pointer (single arg)
- No return value
- Function preamble saves callee-saved regs we use (r12, r13)
- Function epilogue restores and `ret`

```
prologue:
  push rbp
  mov  rbp, rsp
  push r12              ; we use r12 as buffer ptr alias
  push r13              ; we use r13 as loop var alias
  sub  rsp, FRAME_BYTES ; spill space + 16-byte alignment
  mov  r12, rdi         ; r12 = buffer

epilogue:
  add  rsp, FRAME_BYTES
  pop  r13
  pop  r12
  pop  rbp
  ret
```

`r12` and `r13` are callee-saved per System V, so any extern math call
inside the function leaves them intact — no need to spill across calls.

## Memory model

The buffer is a flat byte array. Every input/output/intermediate atom
lives at a fixed byte offset chosen by `BufferLayout` (already
existing). Loads and stores look like:

```
movss xmm0, DWORD [r12 + offset_const + index_reg*elem_bytes]
movss DWORD [r12 + offset_const + index_reg*elem_bytes], xmm0
```

For groups with `count == 1`, no index register — direct displacement.

For atom-offset that's compile-time known but ≥ 2GiB, we materialize
the offset in a scratch register first (mov rax, imm64; lea …).

## Register strategy: hard-coded, no real allocator

For Tier 1 we hard-allocate:

| reg | role |
|---|---|
| `rdi` | function arg (overwritten by `mov r12, rdi` then unused) |
| `r12` | buffer pointer (callee-saved by us) |
| `r13` | loop variable `i` (callee-saved by us) |
| `xmm0` | result accumulator (and 1st binop arg) |
| `xmm1` | 2nd binop arg, or temporary |
| `xmm2` | 3rd value (Select cond/x/y) or reduce-loop scratch |
| `rax` | scratch GP / int load destination |
| `rcx` | scratch GP / int 2nd operand |
| `rdx` | scratch GP / int 3rd operand (Select) |
| `r8`–`r11` | spill scratch when needed |

Each group body has a tiny SSA tree (≤4 simultaneous live values for
Tier 1 ops). Hard-coding registers per op variant covers all cases.
**There is no register allocator.** When we need to preserve `xmm0`
across an extern call (math fn), we save to a fixed stack slot at
`[rsp + 0]` and reload after.

This is intentionally suboptimal — measured against ~85 ms/span
Cranelift baseline, we have a *lot* of headroom for codegen waste, as
long as we keep memory traffic similar to the Cranelift output.

## Math function calls

Math fns (`expf`, `logf`, `tanhf`, …) are extern C functions linked
into the host binary. We declare them once (statically) and embed
their absolute address as an imm64 at every call site:

```
sub  rsp, 8               ; align stack to 16 before call
movss [rsp], xmm0         ; save accumulator if needed
movss xmm0, xmm_arg       ; arg to xmm0
mov  rax, FN_ADDR_IMM64
call rax
movss xmm0, ...           ; result already in xmm0; restore other regs if needed
add  rsp, 8
```

Function addresses come from `extern "C"` wrappers like the existing
`jit_expf`/`jit_logf` in `codegen.rs`. We hand them to dynasm as raw
`u64` immediates — no PLT, no relocation logic.

Stack alignment: System V requires `rsp ≡ 0 (mod 16)` *immediately
before* a `call`. Our prologue pushes 4 × 8 bytes (rbp, r12, r13,
return-addr) so rsp ≡ 8 (mod 16) at function entry's `mov r12, rdi`
point. We allocate a `FRAME_BYTES` that keeps it aligned, then
sub another 8 before any `call` to land on alignment.

## Loop emission

For groups with `count > 1`:

```
  mov  r13, atom_offset             ; i = atom_offset
  mov  rax, atom_offset + count     ; end
.loop_top:
  cmp  r13, rax
  jge  .loop_done
  ; ── body ── (uses r12, r13)
  inc  r13
  jmp  .loop_top
.loop_done:
```

For groups with `count == 1`, no loop — emit body once with
`i = atom_offset` baked into displacements.

dynasm-rs handles labels via `=>label`-style local refs.

## Module structure

```
src/compiler/attempts/v14/
├── codegen.rs          (existing — Cranelift, transitional fallback)
├── x86_jit.rs          (new — dynasm-rs emitter)
└── ...
```

Cargo feature: **`x86_compile`**, separate from `cranelift`. The two
coexist during rollout. Once `x86_compile` covers everything, the
`cranelift` feature, dependencies, and `codegen.rs` all get
deleted in one commit.

Public surface in `x86_jit.rs`:

```rust
pub struct X86JitSpan {
    code: dynasmrt::ExecutableBuffer,
    entry: dynasmrt::AssemblyOffset,
    layout: BufferLayout,
    literal_template: Vec<u8>,
    output_ranges: Vec<AtomRange>,
}

impl X86JitSpan {
    pub fn compile(
        graph: &NanoGraph<'static, SystemPool>,
        output_ranges: &[AtomRange],
    ) -> Result<Self, String>;
}

impl CompiledSpanFn for X86JitSpan {
    fn execute(&self, inputs: &[StoreSlice<'_>], outputs: &mut [SpanOutput<'_>]) {
        // Identical to JitCompiledSpan::execute — clone literal template,
        // populate inputs, transmute self.code.ptr(self.entry) to fn,
        // call it, extract outputs. Reuses BufferLayout helpers.
    }
}
```

Wiring in `compile_nano_graph` (`super_graph/compiled_eval.rs`)
during rollout (both features enabled):

```rust
match (force_pool_eval, has_opaque, x86_jit_enabled) {
    (true, _, _) | (_, true, _) => /* PoolEvalSpan */,
    (_, _, true) => X86JitSpan::compile(&span.graph, &span.outputs)
        .map(box_compiled)
        .or_else(|_| {
            if std::env::var("X86_JIT_STRICT").is_ok() {
                Err(/* hard error so we never miss a coverage gap */)
            } else {
                JitCompiledSpan::compile(&span.graph, &span.outputs)
                    .map(box_compiled)
            }
        }),
    _ => JitCompiledSpan::compile(&span.graph, &span.outputs).map(box_compiled),
}
```

Selection knobs:
- `X86_JIT=1` env — dev opt-in, then becomes default once stable.
- `X86_JIT_STRICT=1` env — error instead of falling back to
  Cranelift on unsupported ops. Used in CI to catch coverage
  regressions.
- `X86_JIT_VALIDATE=1` env — see validation section.

## Validation strategy

`X86_JIT_VALIDATE=1` mode (mirrors existing `FUSION_VALIDATE`):

1. For each span, compile via both backends.
2. Run both on a small synthetic input buffer.
3. Compare output bytes; on first divergence, dump:
   - the span graph (`.text_report`)
   - the Cranelift output offset+bytes
   - the X86JitSpan output offset+bytes
   - first 64 bytes of each
4. Panic with the location.

Slow but catches encoder bugs immediately. Default off.

Per-op unit tests: each `x86_jit_emit_*` function gets a focused
test that builds a 1-group `NanoGraph`, runs through both backends,
asserts byte equality. Reuses the same harness as the existing
`codegen.rs` tests.

For dtype coverage: the test matrix iterates over every variant in
`NumericDType` (including the exotic float types) per supported op,
and checks both backends produce identical bytes. The conversion
slow path's correctness is anchored by `conversions.rs`'s existing
test suite.

## Open risks

1. **dynasm-rs maturity**: stable but not heavily maintained. Last
   release Q1 2025. If it ships a regression we'd be on our own.
   Mitigation: pin a known-good version.

2. **Stack alignment bugs**: System V's 16-byte alignment requirement
   at calls is easy to get wrong. Math functions called with
   misaligned stack will SIGSEGV on systems where they use SSE on the
   stack. Mitigation: validation harness exercises every math fn
   variant on day 1.

3. **dynasm doesn't know about our `unsafe`**: the produced
   `ExecutableBuffer` can be transmuted to a `fn` pointer, but the
   pointer is only valid as long as the buffer lives. Same situation
   as `JITModule`. We store both in `SimpleJitSpan` and the lifetimes
   work out.

4. **General N-D Strided InputRef encoding**: this is the
   trickiest part. dim_strides[]/dim_shape[] decomposition needs
   modular arithmetic per loop iteration. We can defer the general
   path to Tier 2 and fall back to Cranelift for any group with N-D
   Strided inputs initially.

5. **Reduce-fold inlining**: the existing inlining mechanism
   (`inlinable[gi]`, `inlines_producer[ci]`) re-emits a producer's
   body inside its consumer's k-loop. Replicating this in SimpleJit
   adds complexity. **Defer to Tier 2** — initial SimpleJit spans
   that contain inlined groups fall back to Cranelift, then we add
   inlining once everything else works.

6. **Embedded tables**: `EmbeddedTables` (lookup tables for
   `IndirectLoad` and `Explicit` InputRefs) lives outside codegen
   already — reuses unchanged.

7. **Steady-state perf**: the hard-coded reg strategy will produce
   *worse* code than Cranelift on dense fusion chains. Tier 1 target
   is "within 30% of Cranelift exec time"; if we miss, we either add
   a real per-group register pool or revisit. The current
   `COMPILE_FAST=1` measurement showed 14% exec slowdown vs
   `opt_level=speed`, so a hand-rolled emitter should land in the
   same ballpark.

## Rollout plan

Numbering refers to **commits / PRs**, not days. Each phase is
verified by `X86_JIT_VALIDATE=1` against Cranelift on the relevant
test suite before moving on.

### P0 — Plumbing
- Add `dynasmrt = "<latest>"` to Cargo.toml under a new
  `x86_compile` feature.
- Create `src/compiler/attempts/v14/x86_jit.rs` with the
  `X86JitSpan` skeleton.
- `compile()` returns `Err("unsupported")` for any non-empty graph.
- Empty span (`graph.num_groups() == 0`) returns a no-op function:
  prologue → ret → epilogue. Validates ABI, executable memory,
  function ptr transmute end-to-end.
- Wire `X86_JIT=1` selection into `compile_nano_graph` with
  fallback to Cranelift on `Err` (and `X86_JIT_STRICT=1` to error
  instead).
- Test: empty span runs without crashing.

### P1 — F32 happy path
- Hand-emit one Binary Add F32 (count=1) end-to-end first to anchor
  the ABI/encoding pieces.
- Counted-loop scaffold using r13.
- All `ScalarBinOp` arithmetic + comparison variants for
  compute_dtype F32.
- All pure-arithmetic and math-fn Unary ops for F32 (exercises the
  extern-call + stack-alignment path).
- Identity / Cast within F32-only.
- Select with F32 inputs.
- 1D affine `InputRef::Strided` + `InputRef::Broadcast`. Other
  InputRef variants fall back to Cranelift.
- Validation: A/B against Cranelift on every existing F32 unit test
  in `codegen.rs`.

### P2 — Width-parameterized integers + F64
- Add the parameterized integer emission path (F32 has shown the
  pattern). One emission function takes `(bits, signed)` and emits
  the right-width `mov`/`add`/`imul`/`cmp`/`movsx`/`movzx` etc.
- Falls out: I8, I16, I32, I64, U8, U16, U32, U64, Bool — all
  covered by the same code path.
- Sub-byte ints (I4, U4) added in the same phase via the `bits<8`
  branch (shift+mask load/store).
- F64 fast path: parameterized over `addsd`/`mulsd`/etc. — same
  shape as F32 with different operand size.
- Cast across all native dtype pairs (saturating + non-saturating).
- All `ScalarBinOp`/`ScalarUnaryOp` variants we deferred from P1
  (And/Or/Xor, Bitwise*, Mod/IMod, IsNan/IsInf, Erf, Tan, etc.).

### P3 — BF16 / F16 fast paths
- BF16 expand/narrow: 3-instruction `shl 16` for load,
  RTNE+truncate for store.
- F16 expand/narrow: software path first (~10/15 instructions for
  load/store), then optional `vcvtph2ps`/`vcvtps2ph` fast path
  guarded by a one-time F16C cpuid probe.
- Compute path is unchanged — these go through the F32 compute
  pipeline. Only the load/store edges change.
- Validation: full A/B sweep on every BF16/F16 unit test in the
  existing codegen + dtype_discipline tests.

### P4 — Reduce, IndirectLoad, 2D Strided InputRefs
- Reduce loop with accumulator init (Sum=0, Max=−inf, Min=+inf,
  Prod=1) for all native dtypes.
- IndirectLoad: load index from input, index into `table_base`.
- 2D modular Strided (`(i % m) * stride`).
- 2D strided-broadcast Strided (`(i / r) * stride`).
- `InputRef::Explicit` via embedded table (already populated
  outside codegen).

### P5 — End-to-end RWKV-0.1B validation
- Run with `X86_JIT=1 X86_JIT_VALIDATE=1` on the RWKV-0.1B
  generation path. Fix any divergences.
- Profile with `COMPILE_PROFILE=1` — confirm `compile_phases`
  drops to **<1 s** for the supported subset.
- Measure steady-state exec — confirm within ±30% of Cranelift
  (target ±10%).
- Catalogue any ops/InputRefs that still fall back. Either land
  them or note them as known gaps for P6.

### P6 — Long tail
- General N-D Strided InputRef (the full
  `dim_strides[]`/`dim_shape[]` decomposition with modular
  arithmetic per iteration). Tricky enough to deserve its own
  phase.
- Reduce-fold inlining (replicate the existing
  `inlinable[gi]`/`inlines_producer[ci]` mechanism in
  `x86_jit.rs`).
- Anything else that fell back during P5.

### P7 — Inline encode/decode for exotic floats
- **Decode**: build per-format precomputed tables at codegen time
  (call `FloatType::decode_to_f64` over the full domain, cast to
  f32). Embed via `EmbeddedTables`. Deduplicate by `FloatType`
  identity across spans. Codegen emits the 3-instruction table
  lookup.
- **Encode**: parameterized bit-manipulation emitter taking
  `(exp_bits, mant_bits, has_infinity, has_nan)`. Sign / exponent
  rebias / mantissa RTNE / overflow → NaN-or-saturate / underflow
  → zero-or-denormal. ~30–50 instructions per call site.
- Codegen detects `Float(ft)` where `ft` is none of
  F32/F64/F16/BF16 and routes to the inline emitter for
  load (decode) and store (encode). Compute uses F32 (or F64)
  native between them.
- Validation: round-trip every exotic format
  (F8E4M3FN, F8E5M2, F4E2M1, F6E3M2, F6E2M3) through Add/Mul/Cast,
  bytewise A/B against `conversions.rs`'s scalar oracle (which is
  the source of truth that the codegen-time table builder uses
  internally).
- After this lands, **`X86_JIT_STRICT=1` should pass** on
  RWKV-0.1B and any model in the test suite — every `NumericDType`
  has a code path.

### P8 — Make default + drop Cranelift
- Flip default codegen to `X86JitSpan` on x86-64 builds with
  `x86_compile` enabled.
- Add a benchmark gate that fails CI on >10% exec regression vs the
  Cranelift baseline.
- Run all integration tests with `X86_JIT_STRICT=1`.
- One commit deletes:
  - the `cranelift` feature
  - all `cranelift-*` dependencies
  - `src/compiler/attempts/v14/codegen.rs`
  - `JitCompiledSpan`, `compile_span`, `compile_span_validated`,
    `MathFuncs`, `register_math_symbols`, `declare_math_funcs`
  - the `FUSION_VALIDATE` path (the fusion logic itself stays —
    it lives in `x86_jit.rs` now)
- Remaining "compile" surface: `X86JitSpan` and `PoolEvalSpan`
  (the latter still handles opaque ops).

## Decisions confirmed

1. **Feature**: `x86_compile` (new feature, separate from
   `cranelift`).
2. **Crate**: latest `dynasmrt` from crates.io. No version pin.
3. **Cranelift fate**: temporary fallback during P0–P7, deleted in
   P8.
4. **Dtype handling**: parameterized over `NumericDType`. Native
   fast path for F32/F64 + integer widths. Half-width fast path
   for BF16/F16. **Inline** encode/decode for any other
   `Float(ft)` — decode via codegen-time precomputed table
   (`total_bits ≤ 12`) plus parameterized inline bit-manip
   encoder. **No FFI; no extern decode/encode wrappers.** Adding a
   new exotic FloatType requires no codegen changes.
5. **Validation env-var**: `X86_JIT_VALIDATE=1`.
6. **Selection env-vars**: `X86_JIT=1`, `X86_JIT_STRICT=1`.
