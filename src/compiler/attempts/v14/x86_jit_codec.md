# x86_jit codec implementation strategy

> **Status:** Implementation strategy notes for the x86-64 JIT backend's
> realization of the NanoGraph dtype contract. The contract itself is
> specified in `docs/dtype_contract.md` and is runtime-agnostic; this
> document covers how *this specific JIT* satisfies it on x86-64 SystemV.
>
> Read `docs/dtype_contract.md` first. This document assumes you know
> what NanoGraph requires; it explains how the x86_jit backend produces
> those values.

## 1. Compute representation

The dtype contract specifies what value each ScalarOp produces, but it
doesn't say what hardware representation a runtime uses internally. The
x86_jit backend uses one of three hardware "compute representations" for
every value it touches in registers:

| compute repr | hardware              | holds                          |
|--------------|------------------------|--------------------------------|
| `F32`        | `xmm` (low 32 bits, scalar single)  | any FloatType that fits in F32 |
| `F64`        | `xmm` (low 64 bits, scalar double)  | F64, or any FloatType that doesn't fit in F32 |
| `Int`        | `r__` (full 64 bits, sign- or zero-extended) | any IntType, Bool |

A `FloatType` "fits in F32" when `exponent_bits ≤ 8` AND
`mantissa_bits ≤ 23`. All currently named sub-F64 floats (BF16, F16, F8x,
F4x, F6x) satisfy both. A hypothetical `FloatType` with 9-bit exponent or
24+ bit mantissa would require F64 compute repr.

The compute repr is chosen per ScalarOp based on the op's `compute_dtype`,
not per operand. When inputs have different storage dtypes than the
compute_dtype, they are decoded into the chosen compute repr at load
time. When the compute_dtype's natural compute repr differs from the
storage dtype, that's normal — e.g., a BF16 storage slot loaded into an
F32 compute slot.

The contract requires that the result of each op be at compute_dtype
precision. When the chosen compute repr is wider than compute_dtype, the
runtime must apply **precision narrowing** (§3) after each op to enforce
the contract.

## 2. Compute slots

Operators in the x86_jit backend reference values by *compute slot*, not
by physical register. A compute slot is one of three abstract names —
`A`, `B`, `C` — that maps to a fixed physical register based on the
compute repr in use:

| slot | F32     | F64     | Int  |
|------|---------|---------|------|
| A    | xmm0    | xmm0    | rax  |
| B    | xmm1    | xmm1    | rcx  |
| C    | xmm2    | xmm2    | rdx  |

This is the only place in the codegen where physical register names
appear. Op emitters take inputs in canonical slots and write outputs to
canonical slots; they never see physical names directly. The dynasm-rs
"literal register name in macro" tax is paid once per slot/dtype
combination in the load/store helpers, not multiplied across every op.

For Reduce, the inner-k loop uses the same slot conventions but operates
on a separate accumulator state. See §6.

## 3. The narrow-in-register primitive

The contract (`docs/dtype_contract.md` §5) requires that the result of an
op at `compute_dtype = D` be at D's precision. When the x86_jit backend
holds the result in a compute repr wider than D (the common case for
sub-F32 dtypes), it must apply **precision narrowing** to enforce the
contract.

`narrow(value_in_compute_repr, target_dtype)` produces the same bit
pattern that `decode(encode(value, target_dtype), target_dtype)` would,
but stays in the compute repr. It is the in-register equivalent of a
store-and-load roundtrip through a buffer of `target_dtype`.

### 3.1 When narrow is required

- **Between fused producer and consumer in a chain** when the
  intermediate's dtype is narrower than the chosen compute repr. See §5.
- **After each iteration of a Reduce** when `compute_dtype` is narrower
  than the compute repr. Otherwise the accumulator drifts toward
  full-compute-repr precision, breaking the per-step quantization
  contract (`docs/dtype_contract.md` §4.7).
- **Before format encoding to a wider storage dtype** — but this case
  is handled implicitly by the encode itself; see §3.3.

### 3.2 When narrow is a no-op

`narrow` emits zero instructions when the compute repr exactly matches
the target dtype:
- F32 compute repr, target F32 → no-op
- F64 compute repr, target F64 → no-op
- Int compute repr, target I64 or U64 → no-op

This is the load-bearing property that makes the abstraction free for
common cases. A chain of aligned F32 ops compiles to nothing more than
the underlying loads, ALU instructions, and stores — no narrow
instructions, no codec instructions, no extra arithmetic. The only
overhead the architecture introduces over hand-written x86 is the cost
of the orchestration layer, which is per-group not per-instruction.

### 3.3 When narrow is implicit in encode

When a value is being stored to a buffer of dtype `D`, the format
encode (§4) inherently rounds to D's precision. An explicit
`narrow_to(D)` immediately before the encode is redundant — the encode
does the same RTNE math as part of producing the raw bits.

The orchestration layer is responsible for omitting the redundant
narrow when followed by an encode. (It is not a *correctness* problem
to emit it — `narrow(narrow(v, D), D) == narrow(v, D)` because narrow
is idempotent. It is only a wasted instruction.)

### 3.4 Float narrow algorithm

For target `FloatType { exp_bits, mant_bits, has_inf, has_nan }` in
F32 compute repr:

The general algorithm extracts sign / biased_exp / mantissa from the
F32 bit pattern, performs RTNE rounding to `mant_bits`, handles overflow
per `has_inf`, propagates NaN per `has_nan`, and reassembles a new F32
bit pattern that represents the same numeric value at the target
precision. The mantissa beyond `mant_bits` is zeroed; the exponent may
adjust by one if the rounding carries.

Constants known at JIT-build time (`mant_bits`, `exp_bits`, `has_inf`,
`has_nan`) fold into immediates. Branches whose conditions are known
at codegen time (e.g., the inf-check when `has_inf = false`) are elided.

Worst case: ~15-25 instructions for a non-trivial narrow. Best case:
zero instructions (when the target is F32 itself, or any `FloatType`
that's bit-equivalent to F32 — though no current named type satisfies
this beyond F32).

Specific recipe for **F32 → BF16 precision in F32 register**:
```
; xmm0 holds the f32 value to narrow to BF16 precision
movd      eax, xmm0              ; raw bits → eax
mov       ecx, eax
shr       ecx, 16
and       ecx, 1                 ; ecx = lsb of upper-half (round-half-to-even)
add       ecx, 0x7fff            ; bias for RTNE
add       eax, ecx               ; round
and       eax, 0xffff_0000       ; zero low 16 mantissa bits
movd      xmm0, eax              ; back to xmm
```

Roughly 8 instructions, all on integer side. This is the textbook
F32→BF16 RTNE round-trip applied in registers without touching memory.

### 3.5 Int narrow algorithm

For target `IntType { bits }` with signedness `signed`:
- **`bits == 64`**: no-op.
- **`bits == 32`** and signed: `movsxd rax, eax` (sign-extend low 32).
- **`bits == 32`** and unsigned: `mov eax, eax` (zero-extend low 32 by
  writing the low 32-bit reg, which automatically zeroes the high half
  on x86-64).
- **`bits == 16`** and signed: `movsx rax, ax`.
- **`bits == 16`** and unsigned: `movzx rax, ax`.
- **`bits == 8`** and signed: `movsx rax, al`.
- **`bits == 8`** and unsigned: `movzx rax, al`.
- **`bits` other widths**: AND with `(1 << bits) - 1` mask, then sign-
  extend (signed) by shifting left by `64 - bits` and arithmetic-right-
  shifting back. Two instructions.
- **Bool**: `test rax, rax; setne al; movzx rax, al`. Three instructions.

All cases are at most three instructions. Most common cases are one.

## 4. The codec layers (bit I/O, format, narrow)

Per `docs/dtype_contract.md` and the discussion in this document, the
codec splits cleanly into three independent capabilities. The x86_jit
backend implements each in its own module:

- **`bit_io.rs`** — read/write N bits at a bit address from/to a
  register. Knows nothing about dtypes. Output of a load is "raw bits in
  a 64-bit GP register, right-justified, masked." Input of a store is
  the same. Handles arbitrary bit alignment per `docs/dtype_contract.md`
  §2.4.

- **`format.rs`** — convert between raw bits in a 64-bit GP register and
  a value in the chosen compute repr. Two operations:
  - `decode(dtype, raw_gp_reg) → compute_slot`: emit the inline bit
    manipulation that converts raw bits to a real-number-equivalent
    f32/f64 in xmm or i64 in gp. Specialized at JIT-build time on
    `FloatType` properties / `IntType` width.
  - `encode(compute_slot, dtype, raw_gp_reg)`: symmetric inverse,
    including the RTNE rounding when narrowing.

  Native types short-circuit: F32 → `movss` (effectively a no-op since
  the f32 value IS the raw bits), BF16 → `shl 16; movd`, etc. Non-
  native types use the general inline encoder/decoder.

- **`precision.rs`** — `narrow_to(dtype, compute_slot)`: the in-
  register precision narrowing primitive from §3 of this document.
  Pure register-to-register, dtype-aware. Per §3.2, this emits zero
  instructions in the common F32-on-F32 / F64-on-F64 / I64-on-Int cases.

The orchestration layer (§5) calls these in the right order to satisfy
the dtype contract for each ScalarOp.

### 4.1 No-op compression: how aligned F32 chains drop to bare ALU

This is the architecture's load-bearing property. When the dtype is F32,
the storage is byte-aligned, and the compute repr is F32:

- `bit_io::load_bits` → recognized as a single `movss xmm, [r12+addr]`
  (the byte-aligned short-circuit replaces the general bit extraction).
- `format::decode(F32, ...)` → no-op. The f32 in xmm IS the raw bits.
  Zero instructions emitted.
- `format::encode(F32, ...)` → no-op. Same reason.
- `precision::narrow_to(F32, ...)` → no-op (§3.2). Zero instructions.
- `bit_io::store_bits` → `movss [r12+addr], xmm`.

So a single-op F32 group emits exactly:
```
movss xmm0, [r12 + addr_a]
movss xmm1, [r12 + addr_b]
addss xmm0, xmm1
movss [r12 + addr_out], xmm0
```

That is the *full* output, with no overhead from the codec abstraction.
The architecture lets this drop out automatically because every codec
function knows how to be a no-op when its work isn't required.

### 4.2 No-op compression: BF16 vs F32

For comparison, a BF16 single-op group with the same shape:
```
movzx eax, WORD [r12 + addr_a]
shl   eax, 16
movd  xmm0, eax                  ; format::decode(BF16) into xmm0
movzx eax, WORD [r12 + addr_b]
shl   eax, 16
movd  xmm1, eax                  ; format::decode(BF16) into xmm1
addss xmm0, xmm1                 ; op (in F32 compute repr, BF16 inputs)
; precision::narrow_to(BF16) — only needed if there's a successor that reads
;                              this in the same chain; for an unfused single
;                              op, the encode below does the rounding.
movd  eax, xmm0                  ; format::encode(BF16): bits → eax
mov   ecx, eax
shr   ecx, 16
and   ecx, 1
add   ecx, 0x7fff
add   eax, ecx
shr   eax, 16                    ; eax holds BF16 raw bits in low 16
mov   WORD [r12 + addr_out], ax  ; bit_io::store_bits
```

The codec instructions appear here because BF16 isn't bit-equivalent to
F32. They're not "overhead from the abstraction"; they're the actual
work the dtype contract requires for BF16.

## 5. Fusion and the dtype contract

The dtype contract (`docs/dtype_contract.md`) specifies what each op
must produce. It does not require runtimes to materialize intermediates
to memory — fusion is permitted, as long as fused intermediates have
the same value the unfused version would produce.

In the x86_jit backend, fusion is implemented in the loop layer
(`chain.rs`, phase 6 of the rewrite). The mechanism is:

1. The orchestration layer recognizes a chain of ScalarOps where each
   group's output is consumed only by the next group, the shapes match,
   and the load/store pattern is compatible.
2. Instead of emitting a separate loop per group, it emits one loop
   containing all the group bodies.
3. Between consecutive groups in the loop body, the producer's compute
   slot value is forwarded directly to the consumer (no buffer
   roundtrip).
4. **Before the consumer reads the forwarded value, the orchestration
   layer emits `precision::narrow_to(producer.output_dtype)` on the
   forwarded slot.** This enforces the dtype contract on the
   intermediate.
5. The producer's encode/store pair is omitted (since nothing reads it
   from memory).
6. The consumer's load/decode pair is replaced by a direct read of the
   forwarded slot (no memory access).

Step 4 is the critical one. Without it, the fused chain would silently
use full-compute-repr precision for the intermediate, which is more
precise than the contract permits. The result of the fused chain would
differ from the unfused chain, violating the optimization-equivalence
clause of `docs/dtype_contract.md` §7.

### 5.1 Worked example: BF16 add then BF16 mul, fused

Source NanoGraph:
- Group 1: `Add(BF16, BF16) → BF16` with `compute_dtype = BF16`
- Group 2: `Mul(BF16, BF16) → BF16` with `compute_dtype = BF16`
  (Group 2's first input is Group 1's output)

Both compute_dtypes are BF16, so the chosen compute repr is F32 for
both. The fused emission:

```
; --- Group 1: a + b → intermediate ---
; Load a (BF16) into compute slot A
movzx eax, WORD [r12 + addr_a]
shl   eax, 16
movd  xmm0, eax                  ; xmm0 = f32 representation of a (BF16-precision)

; Load b (BF16) into compute slot B
movzx eax, WORD [r12 + addr_b]
shl   eax, 16
movd  xmm1, eax                  ; xmm1 = f32 representation of b (BF16-precision)

; Op: addss is exact at f32 precision. Both operands are BF16-precision values
; held in f32, so the sum has at most 8 mantissa bits — fits exactly in f32.
addss xmm0, xmm1                 ; xmm0 = a + b at full f32 precision

; precision::narrow_to(BF16) — enforce the BF16 contract on the intermediate
movd  eax, xmm0
mov   ecx, eax
shr   ecx, 16
and   ecx, 1
add   ecx, 0x7fff
add   eax, ecx
and   eax, 0xffff_0000
movd  xmm0, eax                  ; xmm0 = BF16-rounded sum, still in f32 representation

; --- (No store; intermediate is forwarded in-register) ---
; --- (No load; consumer reads xmm0 directly) ---

; --- Group 2: intermediate * c → output ---
; Load c (BF16) into compute slot B
movzx eax, WORD [r12 + addr_c]
shl   eax, 16
movd  xmm1, eax                  ; xmm1 = f32 representation of c (BF16-precision)

; Op: mulss is exact for these operands' precision
mulss xmm0, xmm1                 ; xmm0 = (BF16-rounded sum) * c at full f32 precision

; format::encode(BF16) implicitly narrows; no separate narrow_to needed
movd  eax, xmm0
mov   ecx, eax
shr   ecx, 16
and   ecx, 1
add   ecx, 0x7fff
add   eax, ecx
shr   eax, 16                    ; eax holds BF16 raw bits in low 16

; bit_io::store_bits
mov   WORD [r12 + addr_out], ax
```

The narrow between the addss and the second load is what enforces the
contract. Without it, the multiply would see the full f32 sum (more
precise than BF16), and the result would differ from the unfused version
where Group 1's output goes through a real BF16 store-and-load.

### 5.2 Fusion across multi-step chains

A longer chain (e.g., LayerNorm-style sequence: subtract mean, square,
divide, add) would have a `narrow_to` between each pair of consecutive
operations whose `output_dtype` is narrower than the compute repr. If
all intermediates have the same dtype, the narrow is the same instruction
sequence each time and can potentially be hoisted into a helper, but the
default is to emit it inline at each fusion boundary.

### 5.3 When fusion is forbidden

The orchestration layer must NOT fuse across:
- A reduce boundary (the reduce loop is its own structure)
- An IndirectLoad (data-dependent address breaks the fusion shape)
- A group whose output is read by multiple consumers (forwarding the
  intermediate to one consumer doesn't help the others)
- Any case where the load/store pattern has cross-iteration dependencies

These restrictions are inherited from the cranelift backend's existing
`build_fusion_chains` analysis, which the x86_jit backend will port over
in phase 6 of the rewrite.

## 6. Reduce per-step quantization

Reduce ops require the accumulator to be at `compute_dtype` precision
after each iteration (`docs/dtype_contract.md` §4.7). The x86_jit
implementation emits a `precision::narrow_to(compute_dtype)` after each
fold step inside the inner k-loop.

For `compute_dtype = BF16` with F32 compute repr:

```
; Init accumulator
xorps xmm0, xmm0                 ; xmm0 = +0 (BF16 zero is +0 in f32 too — no-op narrow)

; Inner k loop
; ... (loop setup) ...
loop_top:
  ; Load src[k] into xmm1 (with format decode for BF16)
  movzx eax, WORD [r12 + reduce_addr_k]
  shl   eax, 16
  movd  xmm1, eax

  ; Fold
  addss xmm0, xmm1               ; acc += src[k] at f32 precision

  ; precision::narrow_to(BF16) on the accumulator — enforce per-step contract
  movd  eax, xmm0
  mov   ecx, eax
  shr   ecx, 16
  and   ecx, 1
  add   ecx, 0x7fff
  add   eax, ecx
  and   eax, 0xffff_0000
  movd  xmm0, eax                ; xmm0 = BF16-rounded acc

  ; Continue loop
  ; ... (k++, branch) ...
```

For `compute_dtype = F32` with F32 compute repr, the `narrow_to(F32)` is
a no-op and zero instructions are emitted between the addss and the next
iteration. The reduce loop is exactly:
```
loop_top:
  movss xmm1, [r12 + reduce_addr_k]
  addss xmm0, xmm1
  ; ... (k++, branch) ...
```

This is the desired property: F32 reduces are tight, BF16 reduces pay
the narrow cost per iteration as the contract demands.

## 7. Codegen obligations (this backend)

The x86_jit backend MUST:

1. **Produce bit-identical results to the reference implementation**
   (`scalar_ops::*` through `pool_eval::eval_binop` /
   `pool_eval::eval_unaryop`) for every supported NanoGraph. The
   conformance tests in `docs/dtype_contract.md` §7 are the gate.

2. **Choose a compute repr at least as wide as `compute_dtype`** for
   every op. F32 compute repr for any FloatType that fits in F32; F64
   otherwise. I64 (sign-extended) for any IntType.

3. **Apply `narrow_to` after every op whose result will be observed at
   compute_dtype precision** when the compute repr is wider than
   compute_dtype. "Observed" means: forwarded to a fused successor, or
   used as a Reduce accumulator in the next iteration. Encoding to
   memory implicitly narrows (§3.3) and does not need a separate narrow.

4. **Dispatch on `FloatType` and `IntType` properties, not on named
   constants.** A new `FloatType` constant added to `numeric_dtype/mod.rs`
   must work in this backend with no x86_jit changes. The format codec
   and the narrow primitive both consume `(e_bits, m_bits, has_inf,
   has_nan)` as compile-time parameters and emit specialized inline
   code.

5. **Make no-op codec calls emit zero instructions.** A chain of aligned
   F32 ops must compile to bare loads/stores/ALU with no overhead from
   the codec abstraction. This is enforced by §3.2 (narrow no-op cases),
   §4 (codec function fast paths), and the byte-aligned short-circuit
   in `bit_io.rs`.

6. **Support arbitrary bit alignment in `bit_io.rs`.** The default load
   path emits bit extraction at any bit offset; the byte-aligned case is
   a fast path on top. Sub-byte values (I4, U4, F4E2M1, Bool packed
   2/byte etc.) are first-class.

7. **Reject unsupported graph constructs explicitly** in `support.rs`
   rather than producing wrong code or panicking at runtime. The
   reject list should shrink to near-empty as the rewrite progresses;
   per the rewrite plan, anything in the reject list at the end of
   phase 4 is a bug to fix.

These obligations are how the x86_jit backend complies with the contract
in `docs/dtype_contract.md`. Future backends (Vulkan, ARM, etc.) will
have analogous obligations adapted to their hardware models.
