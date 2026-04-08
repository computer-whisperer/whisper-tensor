# NanoGraph dtype contract

> **Status:** Authoritative semantic specification of NanoGraph dtype
> behavior. Any runtime that evaluates a NanoGraph (the slow nano evaluator,
> any compiled-eval JIT backend, future Vulkan/Metal/CUDA backends) must
> produce bit-identical results matching this contract for any well-formed
> nano graph. Where the existing code disagrees with this document, **this
> document is correct and the code is a bug.** Known disagreements are
> listed in §8.
>
> This document defines *what* a runtime must produce. It says nothing
> about *how* a particular runtime achieves it. For implementation strategy
> in the x86-64 JIT backend, see
> `src/compiler/attempts/v14/x86_jit_codec.md`.

## 1. Purpose

Whisper-tensor is built to faithfully reproduce the *exact* numerical
behavior of arbitrary models, including models trained on hardware whose
numeric formats differ from any standard hardware available at inference
time. A model trained on BF16 hardware must produce BF16-equivalent outputs
when inferred on any backend; a model trained on F8E4M3FN must produce
F8E4M3FN-equivalent outputs.

The graph format (NanoGraph) explicitly specifies the dtype at every step.
The runtime's job is to evaluate that graph in a way that produces the
values the dtype specification demands. Backends may use any hardware
shortcuts that *provably* yield the contracted result; backends may not
silently use higher precision, different rounding, or different overflow
behavior than the contract specifies.

The contract says nothing about *how* a runtime computes the result. A
backend may use software emulation, hardware ops, SIMD, GPU shaders, or
any other mechanism — as long as the resulting bits match.

## 2. The numeric type model

### 2.1 NumericDType

A `NumericDType` is one of:
- `Float(FloatType)` — IEEE-style binary floating point
- `SignedInt(IntType)` — two's complement signed integer
- `UnsignedInt(IntType)` — unsigned integer
- `Bool` — single boolean (logically 1 bit; the value is `0` for false
  and `1` for true)

### 2.2 FloatType

```rust
struct FloatType {
    exponent_bits: u8,
    mantissa_bits: u8,
    has_infinity: bool,
    has_nan: bool,
}
```

A `FloatType` is parameterized by:
- `exponent_bits`: number of bits in the biased exponent field (1..=11)
- `mantissa_bits`: number of bits in the explicit mantissa (1..=52)
- `has_infinity`: whether `biased_exp = max, mantissa = 0` decodes to ±∞
- `has_nan`: whether the type can represent NaN

The total bit width is `1 + exponent_bits + mantissa_bits`, capped at 64.

The bias is always `2^(exponent_bits - 1) - 1` (standard IEEE formula).
Encoding rules follow IEEE 754:
- A normal number is `(-1)^sign * (1 + mantissa/2^m) * 2^(biased_exp - bias)`
- A subnormal number (`biased_exp = 0`, `mantissa != 0`) is
  `(-1)^sign * (mantissa/2^m) * 2^(1 - bias)`
- Zero is `biased_exp = 0, mantissa = 0` (signed: distinct +0 and -0)
- The maximum biased exponent is reserved for inf/NaN if `has_infinity`
  and/or `has_nan` are true; otherwise it's a normal exponent value
- "FN" types (no infinity, single NaN at all-ones mantissa) follow the
  F8E4M3FN convention: `biased_exp = max` is a normal exponent EXCEPT for
  the all-ones-mantissa pattern which is the single NaN

The named float types are *examples*; any `(e, m, has_inf, has_nan)`
configuration within the bounds is a valid `FloatType`. A runtime that
handles the named constants but not arbitrary configurations is incomplete.

| Name      | e_bits | m_bits | has_inf | has_nan | Bias | Notes |
|-----------|--------|--------|---------|---------|------|-------|
| F64       | 11     | 52     | true    | true    | 1023 | IEEE binary64 |
| F32       | 8      | 23     | true    | true    | 127  | IEEE binary32 |
| BF16      | 8      | 7      | true    | true    | 127  | F32-shifted |
| F16       | 5      | 10     | true    | true    | 15   | IEEE binary16 |
| F8E5M2    | 5      | 2      | true    | true    | 15   | OCP F8 |
| F8E4M3FN  | 4      | 3      | false   | true    | 7    | OCP F8 (FN) |
| F4E2M1    | 2      | 1      | false   | false   | 1    | OCP F4 |
| F6E3M2    | 3      | 2      | false   | false   | 3    | OCP F6 |
| F6E2M3    | 2      | 3      | false   | false   | 1    | OCP F6 |

### 2.3 IntType

```rust
struct IntType {
    bits: u8,
}
```

`IntType` is parameterized by bit width `1..=64`. The named widths (4, 8,
16, 32, 64) are common cases; arbitrary widths are valid. Combined with
the `SignedInt` / `UnsignedInt` arms of `NumericDType`, every integer
type can be expressed.

Signed integers use two's complement. Range:
- `SignedInt(bits)`: `[-2^(bits-1), 2^(bits-1) - 1]`
- `UnsignedInt(bits)`: `[0, 2^bits - 1]`

### 2.4 Storage

Raw bits of a value of dtype `T` are the low `T.total_bits()` bits of a
`u64`. Higher bits are zero. Storage in a buffer may use any bit alignment
and any bit stride; the slot's element width is `T.total_bits()`.

There is **no** rule that values must be byte-aligned. Sub-byte values
(4-bit ints, 4-bit floats, Bool) may be packed into bytes, and bit strides
may not be multiples of 8. A conformant runtime must extract exactly
`total_bits()` bits at the specified bit position regardless of byte
alignment.

## 3. Reference operations on raw bits

These pure functions are the canonical semantic reference. They operate
on raw bits in a `u64` plus a dtype descriptor. All runtimes must produce
results consistent with these. The reference implementation lives in
`src/numeric_dtype/conversions.rs` and `src/scalar_ops/`.

### 3.1 Decode

`decode(raw: u64, dtype: NumericDType) -> RealValue`

Returns the abstract real-number value the bits represent. For floats this
is a real number, possibly ±∞ or NaN. For ints it's an integer in the
type's range. For Bool it's a boolean.

The reference implementation is `NumericDType::decode_to_f64` (and the
underlying `FloatType::decode_f64` / `IntType::decode_signed` /
`IntType::decode_unsigned`).

For every supported `FloatType`, decode is exact: every representable bit
pattern decodes to its exact mathematical value (with NaN representing
"any NaN" — distinct NaN bit patterns are not distinguished after decode).
Integers decode exactly via sign- or zero-extension.

### 3.2 Encode

`encode(value: RealValue, dtype: NumericDType) -> u64`

Returns the raw bits of the dtype value closest to the given real value,
under the dtype's rounding and overflow rules:

- **Float encode**: round-to-nearest-even (RTNE) to the dtype's precision,
  with overflow producing `+∞` if `has_infinity` else `+max_finite`, and
  the symmetric behavior for negatives. NaN encodes to the dtype's
  canonical NaN if `has_nan` else to `+0`.
- **Signed integer encode**: truncate toward zero (for float input),
  saturate to the type's range. NaN → `0`. (This matches Rust's `as`
  cast semantics for f64→signed.)
- **Unsigned integer encode**: truncate toward zero, saturate. Negative
  values → `0`. NaN → `0`.
- **Bool encode**: `false` if value is `0` (or `-0` decoded as 0);
  `true` otherwise (including NaN, since "is not zero" is true for NaN).

The reference implementation is `NumericDType::encode_from_f64` (and the
underlying `FloatType::encode_f64`).

### 3.3 Cast

`cast(raw: u64, src: NumericDType, dst: NumericDType) -> u64`

Returns the dst-encoded raw bits of the value `decode(raw, src)`, encoded
via `encode(_, dst)`. Equivalently:

```
cast(raw, src, dst) = encode(decode(raw, src), dst)
```

When `src == dst`, cast is the identity function (the bits pass through
unchanged). When `src != dst`, cast goes through the abstract real value.

Special cases the contract guarantees:
- **Float → float, src precision ≥ dst precision**: result is the dst
  value closest to the src value under RTNE. Overflow produces inf or
  saturates to max_finite per `dst.has_infinity`.
- **Float → float, src range ≤ dst range and src precision ≤ dst
  precision**: exact (no information loss).
- **Float → int**: truncate toward zero, saturate to int range. NaN → 0.
- **Int → float**: RTNE round to dst precision.
- **Int → int (widening)**: sign-extend (signed src) or zero-extend
  (unsigned src), then re-encode in dst's width.
- **Int → int (narrowing)**: saturate to dst range.
- **Bool → numeric**: `false → 0`, `true → 1` in the dst's encoding.
- **Numeric → Bool**: `false` if real value is exactly 0 (including -0);
  `true` otherwise (including NaN).

The reference implementation is `NumericDType::cast_raw`.

## 4. ScalarOp contracts

A `NanoGraph` is composed of `AtomGroup`s, each of which has an
`output_dtype` and a `ScalarOp`. The ScalarOp specifies what computation
each atom in the group performs. The `output_dtype` specifies the dtype of
the produced atoms.

The contract for each ScalarOp is given as a function from inputs to
output. Runtimes must produce the specified output bits exactly,
regardless of how they internally compute the result.

### 4.1 Literal / LiteralSpan

```rust
ScalarOp::Literal(NumericScalar)
ScalarOp::LiteralSpan(NumericTensor)
```

Each atom in the group produces a constant value:
- `Literal(s)`: every atom produces `cast(s.raw_bits, s.dtype, output_dtype)`.
- `LiteralSpan(t)`: atom `i` produces `cast(t.element_at(i).raw_bits,
  t.dtype, output_dtype)`.

### 4.2 Identity

```rust
ScalarOp::Identity
```

One input. Each atom produces:
```
cast(input.raw_bits, input.slot_dtype, output_dtype)
```

The cast is the **non-saturating** form (overflow on float dtypes with
`has_infinity` produces inf, not max_finite).

### 4.3 Cast

```rust
ScalarOp::Cast { saturating: bool }
```

One input. Each atom produces:
```
let raw = cast(input.raw_bits, input.slot_dtype, output_dtype);
if saturating { saturate_inf(raw, output_dtype) } else { raw }
```

When `saturating = true`, any infinity in the result is replaced with
`±max_finite`. This matters only when `output_dtype.has_infinity = true`;
for FN types the encode already saturates and the flag is a no-op.

ONNX `Cast` defaults to `saturating = true` for float8 targets and
`saturating = false` otherwise.

### 4.4 Binary

```rust
ScalarOp::Binary {
    op: ScalarBinOp,
    compute_dtype: NumericDType,
}
```

Two inputs. Each atom produces:
```
let a = cast(input_a.raw_bits, input_a.slot_dtype, compute_dtype);
let b = cast(input_b.raw_bits, input_b.slot_dtype, compute_dtype);
let result = apply_binop(op, a, b, compute_dtype);
cast(result, compute_dtype, output_dtype)
```

The op runs **at compute_dtype precision**. Inputs are cast down (or up)
to compute_dtype before the op. The result is at compute_dtype precision
(by definition of `apply_binop`), then cast to output_dtype.

`compute_dtype` and `output_dtype` may differ. A common pattern: BF16
inputs and output, but `compute_dtype = F32` (cast inputs up to F32 first,
op in F32, cast result back to BF16).

`apply_binop` is defined per op in §5.

### 4.5 Unary

```rust
ScalarOp::Unary {
    op: ScalarUnaryOp,
    compute_dtype: NumericDType,
}
```

One input. Each atom produces:
```
let x = cast(input.raw_bits, input.slot_dtype, compute_dtype);
let result = apply_unop(op, x, compute_dtype);
cast(result, compute_dtype, output_dtype)
```

Same precision discipline as Binary.

### 4.6 Select

```rust
ScalarOp::Select
```

Three inputs: `[cond, x, y]`. Each atom produces:
```
let is_true = is_truthy(decode(cond.raw_bits, cond.slot_dtype));
let chosen = if is_true { x } else { y };
cast(chosen.raw_bits, chosen.slot_dtype, output_dtype)
```

`is_truthy` for the decoded value:
- **Float**: `false` iff the value is exactly `0.0` (or `-0.0`); NaN is
  **truthy** (matches "not equal to zero" semantics, in which any
  comparison with NaN is non-equal).
- **Integer**: `value != 0`.
- **Bool**: the bool value itself.

There is no `compute_dtype` on Select. The chosen branch's value is cast
directly from its slot dtype to output_dtype.

### 4.7 Reduce

```rust
ScalarOp::Reduce {
    kind: ReduceKind,           // Sum / Prod / Max / Min
    reduce_count: u64,
    reduce_stride: i64,
    compute_dtype: NumericDType,
}
```

One input. The reduction iterates `k = 0 .. reduce_count`, reading the
input atom at offset `base + k * reduce_stride` from the resolved base.
The accumulator is initialized per kind:

| kind | initial value (in compute_dtype) |
|------|----------------------------------|
| Sum  | `+0`                             |
| Prod | `+1`                             |
| Max  | `-∞` (float with `has_infinity`) or `-max_finite` (FN float) or `MIN` (int) |
| Min  | `+∞` (float with `has_infinity`) or `+max_finite` (FN float) or `MAX` (int) |

Each iteration:
```
let v = cast(input_k.raw_bits, input_k.slot_dtype, compute_dtype);
let binop = match kind { Sum → Add, Prod → Mul, Max → Max, Min → Min };
acc = apply_binop(binop, acc, v, compute_dtype);
```

**Critical: the accumulator is at compute_dtype precision after each
iteration.** A Reduce with `compute_dtype = BF16` must produce the same
per-step rounding as if each accumulation step were stored to and reloaded
from a BF16 buffer. Runtimes may not "improve" precision by accumulating
in a wider type — the per-step quantization IS the contract.

After all iterations, the result is cast from compute_dtype to
output_dtype.

### 4.8 IndirectLoad

```rust
ScalarOp::IndirectLoad { table_base: AtomId }
```

One input (the index). Each atom produces:
```
let idx = decode(index.raw_bits, index.slot_dtype).to_int();
let table_atom = AtomId(table_base.0 + idx);
let val = lookup(table_atom);
cast(val.raw_bits, val.slot_dtype, output_dtype)
```

The index input must decode to a non-negative integer (the contract
permits any numeric dtype as the index, but the value must be a
representable atom offset). Behavior for out-of-bounds indices is
unspecified.

### 4.9 OpaqueOutput

```rust
ScalarOp::OpaqueOutput { opaque_idx: usize, output_idx: usize }
```

The output of an opaque sub-graph. The runtime invokes the opaque op
(once, caching its outputs across all atom groups that read from it) and
reads this group's portion. The opaque op has its own dtype contract and
is beyond the scope of this document.

## 5. Per-op semantics

This section defines `apply_binop(op, a, b, compute_dtype)` and
`apply_unop(op, x, compute_dtype)` for every variant.

### 5.1 Float arithmetic (Add, Sub, Mul, Div)

For `compute_dtype = Float(ft)`:
```
apply_binop(op, a, b, ft) = encode(decode(a, ft) op decode(b, ft), ft)
```

That is: decode both operands to abstract real values, perform the
real-number operation, and RTNE-round to ft's precision.

**Division by zero** follows IEEE: `positive/0 = +∞`, `negative/0 = -∞`,
`0/0 = NaN`. For ft without infinity, the encode rounds to ±max_finite.
For ft without NaN, `0/0` encodes to `+0`.

**NaN propagation** follows IEEE: any op with NaN input produces NaN. For
ft without NaN, the encode produces `+0`.

### 5.2 Integer arithmetic (Add, Sub, Mul, Div) — wrapping

Integer arithmetic uses **wrapping** semantics on overflow.

- Signed: result is `(decode_signed(a) op decode_signed(b))` reduced
  modulo `2^bits` and re-encoded. `MIN / -1` wraps to `MIN`.
- Unsigned: result is `(decode_unsigned(a) op decode_unsigned(b))`
  reduced modulo `2^bits`.
- **Division by zero returns 0** for both signed and unsigned (not
  undefined behavior).

A future ScalarOp parameter may specify saturating arithmetic; until then
all integer arithmetic is wrapping.

### 5.3 Min and Max

`Min(a, b) = if a < b then a else b`
`Max(a, b) = if a > b then a else b`

For floats, the contract is **IEEE 754-2008 minNum / maxNum** semantics:
NaN-skipping. Specifically:

- If exactly one operand is NaN, the result is the **non-NaN** operand.
- If both operands are NaN, the result is NaN.
- Otherwise, the result is the lesser (Min) or greater (Max) operand by
  IEEE ordered comparison.
- `Min(+0, -0)` and `Max(+0, -0)` may return either zero (the contract
  does not distinguish), matching `f64::min` / `f64::max`.

This matches Rust's `f32::min` / `f32::max` / `f64::min` / `f64::max`,
which is what the reference implementation in `scalar_ops::min` /
`scalar_ops::max` uses. **A backend that propagates NaN unconditionally
(IEEE 754-2019 minimum/maximum semantics) is non-compliant** and must
emit additional code to filter NaN.

For ints, the comparison uses signed or unsigned ordering per the
compute_dtype.

### 5.4 Modulo

Two distinct ops:

- **`Mod`**: C-style remainder (truncated). Result sign matches the
  dividend.
  - For floats: `a - trunc(a/b) * b` (Rust's `%`, equivalent to `fmod`).
  - For signed ints: `a % b` with truncated division.
  - For unsigned ints: `a % b` (no sign issue).
  - Division by zero → 0 for ints; NaN for floats.

- **`IMod`**: Mathematical (Euclidean / floored) modulo. Result sign
  matches the divisor; result is in `[0, |b|)` for positive `b` and in
  `(b, 0]` for negative `b`. **This is the same Euclidean form for all
  numeric dtypes — float and integer.** Matches the ONNX `Mod` op with
  `fmod=0` (the default).
  - For signed ints: `let r = a % b; if r != 0 && (r ^ b) < 0 { r + b }
    else { r }`.
  - For unsigned ints: same as `Mod` (no negatives).
  - For floats: `let r = a % b; if r != 0 && sign(r) != sign(b) { r + b }
    else { r }`. The reference implementation is
    `scalar_ops::modulo::float_imod`.
  - Division by zero → 0 for ints; NaN for floats.

### 5.5 Pow

`Pow(a, b) = a^b`, evaluated as a real number, then encoded in
compute_dtype.

For ints, the contract is:

- If `b >= 0`: compute `a^b` with **saturating** integer arithmetic
  (overflow saturates to the type's MAX/MIN). For unsigned bases the
  result is in `[0, MAX]`; for signed bases the sign of the result is
  `sign(a)^b`.
- If `b < 0` (signed compute_dtype only): the result is **`0`**. This
  reflects the truncation of the mathematical value `1 / a^|b|`, which
  is `0` for any `|a| ≥ 2`. The contract extends this to all integer
  bases, including `±1` and `0`, for definedness — practically, integer
  Pow with negative exponents is meaningless and this rule simply gives
  it a single, testable value.
- Pathologically large exponents (`b > u32::MAX`) are clamped to
  `u32::MAX` before the saturating power loop. Because integer Pow
  saturates within `O(log_2 |MAX|)` iterations of repeated squaring for
  any `|base| ≥ 2`, and because `|base| ∈ {0, 1}` give a fixed point at
  the first iteration, clamping is observationally equivalent to
  computing the full exponent.

The reference implementation is `scalar_ops::pow::signed_pow` /
`scalar_ops::pow::unsigned_pow`.

For floats, follows IEEE's `pow` semantics including special cases:
- `pow(1, _) = 1`, `pow(_, 0) = 1` (including `pow(0, 0) = 1`)
- `pow(NaN, _) = NaN`, `pow(_, NaN) = NaN`
- Other special cases per IEEE 754-2008 §9.2.1

### 5.6 Comparisons (Equal, Less, LessOrEqual, Greater, GreaterOrEqual)

Comparison ops produce a boolean result encoded into compute_dtype as
`0` (false) or `1` (true), which is then cast to output_dtype by the
ScalarOp dispatch (§4.4 / §4.5).

- **Float comparisons** use IEEE ordered semantics:
  - `Equal(NaN, _) = false`, `Equal(_, NaN) = false`, `Equal(NaN, NaN)
    = false`.
  - `Less(NaN, _) = false`, `Greater(NaN, _) = false`, etc.
  - `Equal(+0, -0) = true` (signed zeros compare equal).
- **Integer comparisons** use signed or unsigned ordering after sign- or
  zero-extension.
- **Bool comparisons**: `Equal` is bool equality; other comparisons are
  unspecified (don't generate them).

The encoded result `1` in compute_dtype is `encode(1.0, compute_dtype)`
for floats and integer `1` for ints. The result `0` is `encode(0.0,
compute_dtype)`.

### 5.7 Logical (And, Or, Xor, Not)

Logical ops treat their operands as truthy/falsy via decoded value:

- **Float operand**: truthy iff the decoded real value is not exactly
  `0.0` (treats `-0.0` as falsy because `-0.0 == 0.0`). NaN is **truthy**.
- **Integer operand**: truthy iff value `!= 0`.
- **Bool operand**: the bool itself.

Then the boolean op is applied and the result is encoded into compute_dtype
as `0` or `1` (same encoding rule as comparisons in §5.6).

`Not` is unary: `Not(x) = !is_truthy(x)`.

This must be consistent with `Select`'s truthiness rule (§4.6). The
current `scalar_ops::logical` tests raw bits instead of decoded values
and is therefore inconsistent with `Select` for `-0.0`. See §8.

### 5.8 Bitwise (BitwiseAnd, BitwiseOr, BitwiseXor, BitShiftLeft, BitShiftRight, BitwiseNot)

Bitwise ops are **integer-only**. Calling them with a float `compute_dtype`
is a graph error.

The op operates on the raw integer bits, masked to `it.bits` width:
```
result = op(a & mask, b & mask) & mask
where mask = (1 << it.bits) - 1
```

`BitShiftLeft` shifts in zeros from the right. `BitShiftRight` is
**arithmetic** for signed ints (sign-extends from the left) and **logical**
for unsigned ints (shifts in zeros).

Shift count is taken modulo the type width. `BitShiftLeft(x, n)` where
`n >= bits` is undefined; runtimes should reject during graph validation.

`BitwiseNot(x) = !x & mask`.

### 5.9 Unary float ops (Exp, Ln, Sqrt, Sin, Cos, Tan, Tanh, Asin, Acos, Atan, Sinh, Cosh, Asinh, Acosh, Atanh, Erf, Log1p)

Each unary float op evaluates its real-valued counterpart and RTNE-encodes
to compute_dtype. The reference is `scalar_ops::*::float_*` which uses
Rust's libm-bound implementations.

Backends may use hardware approximations only when those approximations
are bit-equal to the libm result for the compute_dtype. For most
transcendentals this means calling libm via a trampoline; only `sqrt` is
generally bit-equal between hardware and libm.

### 5.10 Other unary (Neg, Abs, Sign, Reciprocal, Floor, Ceil, Round, IsNan, IsInf)

- **`Neg(x) = -x`**. Sign flip on floats (preserves NaN); two's complement
  negation on ints (`Neg(MIN) = MIN` under wrapping).
- **`Abs(x) = |x|`**. Float Abs of NaN is NaN. Int `Abs(MIN) = MIN` under
  wrapping.
- **`Sign(x)`**: `-1` if `x < 0`, `0` if `x == 0` (or `-0`), `+1` if
  `x > 0`. For floats, `Sign(NaN) = NaN`.
- **`Reciprocal(x) = 1 / x`**. Float-only.
- **`Floor`, `Ceil`**: float-only. Round toward `-∞` / `+∞` respectively.
- **`Round`**: float-only. Round-half-away-from-zero (matches ONNX
  convention and Rust `f32::round` / `f64::round`).
- **`IsNan(x)`**: returns `1` if `x` is a NaN under its dtype's encoding
  rules, `0` otherwise. Always `0` for integers and Bool.
- **`IsInf { positive: bool, negative: bool }`**: returns `1` if `x` is
  the matching infinity. Always `0` for integers, Bool, and floats
  without infinity.

## 6. Special-value handling

### 6.1 Signed zero

Floats distinguish `+0` and `-0`. The contract preserves this distinction
through arithmetic per IEEE 754:
- `+0 + +0 = +0`, `+0 + -0 = +0`, `-0 + -0 = -0`
- `-0 * x = -0` for `x > 0`, `+0` for `x < 0`
- `Equal(+0, -0) = true`
- `is_truthy(+0) = false`, `is_truthy(-0) = false`
- `Sign(+0) = 0`, `Sign(-0) = 0`

### 6.2 Infinity

Floats with `has_infinity = true` follow IEEE inf semantics:
- `inf + finite = inf`, `inf + inf = inf`, `inf - inf = NaN`
- `inf * 0 = NaN`, `inf * inf = inf`, `inf * finite = inf`
- `finite / 0 = ±inf`, `inf / finite = ±inf`, `inf / inf = NaN`

For floats with `has_infinity = false`, any operation that would produce
inf instead produces `±max_finite` via the encode's overflow handling.

### 6.3 NaN

Floats with `has_nan = true` follow IEEE NaN propagation:
- Any arithmetic op with a NaN input produces NaN.
- Comparisons with NaN produce `false` (except `!=`, but NanoGraph does
  not provide a `NotEqual` op directly).
- `is_truthy(NaN) = true` (since "not equal to zero" is true for NaN).
- `Min`/`Max` NaN handling: IEEE 754-2008 minNum / maxNum (NaN-skipping)
  per §5.3.

For floats with `has_nan = false`, any operation that would produce NaN
instead produces `+0` via the encode's NaN handling (see §3.2).

### 6.4 Subnormals (denormals)

Floats represent subnormals when `biased_exp = 0, mantissa != 0`. The
contract requires runtimes to handle subnormals correctly per IEEE 754
gradual underflow. Runtimes may not flush subnormals to zero except where
the dtype's encode rules already do so.

## 7. Conformance testing

Any runtime that claims to evaluate a NanoGraph must pass the following
tests, all run for **every** named `NumericDType` constant (and for
representative custom `(e, m, has_inf, has_nan)` configurations of
`FloatType` and arbitrary widths of `IntType`):

1. **Cast roundtrip**: for each `(src_dtype, dst_dtype)` pair, generate
   ~1000 representative source values (normal, ±0, denormal, ±inf, NaN,
   max/min finite, rounding edges), compute the cast via the runtime and
   via the reference (`NumericDType::cast_raw`), and assert bit-equality.

2. **Per-op roundtrip**: for each `ScalarBinOp` and `ScalarUnaryOp`, for
   each compute_dtype, run the op via the runtime and via the reference
   (`scalar_ops::*` through `pool_eval::eval_binop` /
   `pool_eval::eval_unaryop`) on representative input pairs, and assert
   bit-equality.

3. **Reduce per-step quantization**: for compute_dtype with narrower
   precision than the natural compute repr (e.g., BF16 with f32 compute),
   run a Reduce of 1024 elements with values designed to exercise
   quantization (sums where intermediate precision matters) and assert
   that the runtime's result matches the unfused per-iteration
   quantization reference.

4. **Optimization equivalence**: for each optimization a runtime
   implements (fusion, vectorization, hardware shortcuts), run the same
   graph with and without the optimization and assert bit-equality.

These tests are the *minimum*; runtimes should add tests for any
optimizations to confirm they don't break the contract.

## 8. Current implementation gaps

Known places where the existing reference code disagrees with this
document, listed so they can be fixed (and so backend authors know not
to copy the bug):

1. **(Resolved.)** `scalar_ops::logical`'s functions still treat their
   inputs as raw-bit truthiness, but their *callers* in `pool_eval` no
   longer pass raw float bits directly. `pool_eval` now uses an
   `is_truthy(raw, dtype)` helper based on `decode_to_f64(raw) != 0.0`
   for `And` / `Or` / `Xor` (and the corresponding Bool/Bitwise variants),
   so `-0.0` is correctly falsy and NaN is correctly truthy. The float
   `Not` unary op was also fixed to use the same truthiness rule (it
   previously special-cased NaN as falsy). The `scalar_ops::logical`
   module is preserved as a low-level helper that operates on
   pre-truthified 0/1 values, per its existing documentation.

2. **(Resolved.)** Float `IMod` previously lumped with `Mod` in
   `pool_eval::eval_binop` (using truncated semantics for both), and the
   `milli_graph::SimpleBinary::Modulo` lowering forced `fmod=true` for
   floats regardless of the requested attribute. Both are now fixed:
   `scalar_ops::modulo::float_imod` provides the Euclidean form,
   `pool_eval` dispatches `IMod` to it, and the lowering honors the
   `fmod` attribute uniformly for floats and integers per §5.4.

3. **(Resolved.)** `BitShiftRight` for signed ints previously used the
   `bitwise_op_int(a, b, &it, |x, y| x >> y)` form, where `>>` on `u64`
   is a logical shift — incorrect for signed types. `pool_eval` now
   dispatches `BitShiftRight` to `scalar_ops::bitwise::signed_shift_right`
   for the signed arm and `unsigned_shift_right` for the unsigned arm.
   `scalar_ops::bitwise` has new `shift_left`, `signed_shift_right`,
   and `unsigned_shift_right` functions with conformance tests covering
   the signed-vs-unsigned divergence on the high bit, count-mod-width
   semantics, and narrow types.

4. **(Resolved.)** `scalar_ops::min` / `max` NaN handling is now
   formally specified in §5.3 as IEEE 754-2008 minNum / maxNum
   (NaN-skipping). The reference implementations `float_min` /
   `float_max` use Rust's `f32::min` / `f64::min`, which match this
   semantic. Tests in `scalar_ops::min` / `scalar_ops::max` cover NaN
   in either argument position, both-NaN, and the BF16 fast path.
   Backends that propagate NaN unconditionally (the IEEE 754-2019
   minimum/maximum form) are non-compliant and must filter explicitly.

5. **(Resolved.)** `signed_pow` previously decoded the exponent via
   `decode_unsigned`, silently reinterpreting negative exponents as
   huge positive values. Fixed in `scalar_ops::pow::signed_pow`: the
   exponent is now decoded as signed, and `b < 0` returns `0` per the
   §5.5 contract (extended to give negative integer Pow a single,
   testable value rather than leaving it undefined). Both signed and
   unsigned Pow now clamp exponents larger than `u32::MAX` instead of
   truncating, which is observationally equivalent due to saturation
   in the power loop. Tests cover negative exponents on positive,
   negative, and `±1` bases, narrow types, and pathologically large
   exponents.

6. **(Resolved.)** `Cast { saturating }` for FN target types is now
   verified by tests in `numeric_dtype::conversions::tests`. Two tests
   prove (a) `saturate_inf` is a bit-for-bit no-op on any `cast_raw`
   result targeting an FN type, and (b) `cast_raw` to an FN target
   never decodes to ±∞ for any input. Both properties are checked
   across all named FN-shaped float types (`F8E4M3FN`, `F4E2M1`,
   `F6E3M2`, `F6E2M3`) and a custom `(e=4, m=2, has_inf=false)`
   configuration to confirm the property is structural, not specific
   to the named constants.

7. **`encode_intermediate` Bool target NaN handling is wrong.** In
   `numeric_dtype::conversions::encode_intermediate`, the `Bool` arm
   uses `f != 0.0 && !f.is_nan()`, which decodes NaN as `false`. This
   contradicts §3.2 / §3.3 / §4.6 / §5.7 which all say NaN is truthy
   ("not equal to zero" is true for NaN). The same bug exists in the
   `encode_from_f64` function for the `Bool` target. The standalone
   `pool_eval::is_truthy` helper added during the logical-truthiness
   fix already routes around this for binary/unary/select dispatch,
   but raw `cast_raw(_, Bool)` and `encode_from_f64(_, Bool)` still
   carry the bug. Fix is straightforward (`f != 0.0`), but deferred
   to a follow-up so this phase 0 verification stays focused.

8. **Runtimes that hardcode named NumericDType constants are
   non-compliant.** A compliant runtime must dispatch on `FloatType`
   *properties* (`e_bits`, `m_bits`, `has_inf`, `has_nan`) and
   `IntType.bits`/signedness, not on the named constants. The current
   cranelift backend's `emit_store_load_roundtrip` is an example of a
   non-compliant pattern: it matches on `NumericDType::BF16`,
   `NumericDType::F16`, etc., and silently produces wrong results for
   any unnamed type — including a known bug where `F16` is a pass-through
   (no actual narrowing) and all F8/F4/F6 types fall through to a no-op
   default arm.

These gaps are the "this document is correct, the code is a bug" cases.
Fixing them in `scalar_ops` and `pool_eval` is required so that any new
runtime can use them as a correct reference.
