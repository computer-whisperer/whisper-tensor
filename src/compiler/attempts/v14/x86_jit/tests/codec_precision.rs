//! Unit tests for [`super::super::codec::precision`].
//!
//! `narrow_to` is contract-equivalent to `decode(encode(value,
//! target), target)` — and that's how we test it. The JIT loads
//! the input via [`emit_decode`] of the source dtype, calls
//! `emit_narrow_to(target)`, then writes the result back via
//! [`emit_encode`] of the source dtype. The reference is the chained
//! cast `cast_raw(cast_raw(input, source, target), target, source)`.
//!
//! Test ABI: `extern "C" fn(u64) -> u64`
//!   - `rdi` — source dtype raw bits in
//!   - `rax` — source dtype raw bits out
//!
//! For float tests, source = F32, so the value passes through xmm0
//! between decode and encode. For int tests, source = I64, so the
//! value passes through rax (which is also raw_reg for both decode
//! and encode).
//!
//! Register conventions match `tests/codec_format.rs`:
//!   - r8/r9 for explicit codec scratches
//!   - rcx/rdx implicitly clobbered

use super::super::codec::format::{CodecSlot, CodecTables, ComputeRepr, emit_decode, emit_encode};
use super::super::codec::precision::emit_narrow_to;
use super::jit_harness::JitFn;
use crate::numeric_dtype::NumericDType;

const RAW_IN: u8 = 7; // rdi
const RAW_OUT: u8 = 0; // rax
const SCRATCH_GP: u8 = 8; // r8
const SCRATCH_GP2: u8 = 9; // r9
const RAW_TEMP: u8 = 10; // r10 — narrow_to's intermediate raw bits
const FLT_SLOT: u8 = 0; // xmm0
const FLT_SCRATCH: u8 = 1; // xmm1

/// A built JIT plus the codec tables it holds pointers into.
struct JitWithTables {
    jit: JitFn,
    _tables: CodecTables,
}

/// Build a JIT that decodes the source dtype's raw bits, applies
/// `narrow_to(target)`, then re-encodes back to the source dtype.
/// `source` and `target` must use the same compute repr.
fn build_narrow_jit(source: NumericDType, target: NumericDType) -> JitWithTables {
    let src_repr = ComputeRepr::for_dtype(source);
    let tgt_repr = ComputeRepr::for_dtype(target);
    assert_eq!(
        src_repr, tgt_repr,
        "narrow_to test requires source and target to share a compute repr"
    );

    let mut tables = CodecTables::new();
    let jit = JitFn::build(|asm| {
        let slot = match src_repr {
            ComputeRepr::F32 | ComputeRepr::F64 => CodecSlot::Xmm(FLT_SLOT),
            ComputeRepr::Int => CodecSlot::Gp(RAW_OUT),
        };
        // Decode: source raw bits → compute slot
        emit_decode(
            asm,
            source,
            RAW_IN,
            slot,
            SCRATCH_GP,
            FLT_SCRATCH,
            &mut tables,
        )
        .expect("source decode emit");
        // Narrow: compute slot mutated in place
        emit_narrow_to(
            asm,
            target,
            slot,
            RAW_TEMP,
            SCRATCH_GP,
            SCRATCH_GP2,
            FLT_SCRATCH,
            &mut tables,
        )
        .expect("narrow_to emit");
        // Encode: compute slot → source raw bits in RAW_OUT
        emit_encode(
            asm,
            source,
            slot,
            RAW_OUT,
            SCRATCH_GP,
            SCRATCH_GP2,
            FLT_SCRATCH,
        )
        .expect("source encode emit");
    });
    JitWithTables {
        jit,
        _tables: tables,
    }
}

fn run(jit: &JitWithTables, raw_in: u64) -> u64 {
    let f: extern "C" fn(u64) -> u64 = unsafe { std::mem::transmute(jit.jit.ptr()) };
    f(raw_in)
}

/// Compute the reference value: chain `source → target → source`
/// through `cast_raw`. This is the contract-equivalent of the JIT's
/// `decode(source) → narrow(target) → encode(source)`.
fn reference(source: NumericDType, target: NumericDType, raw_in: u64) -> u64 {
    let through = source.cast_raw(raw_in, target);
    target.cast_raw(through, source)
}

/// Run the JIT and reference for every input bit pattern in `inputs`
/// and assert agreement.
fn assert_narrow(
    source: NumericDType,
    target: NumericDType,
    inputs: impl IntoIterator<Item = u64>,
) {
    let jit = build_narrow_jit(source, target);
    for raw_in in inputs {
        let want = reference(source, target, raw_in);
        let got = run(&jit, raw_in);
        assert_eq!(
            got, want,
            "narrow {source}→{target}: input 0x{raw_in:x} → got 0x{got:x} want 0x{want:x}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// Float narrow_to: F32 source through every smaller float dtype
// ─────────────────────────────────────────────────────────────────────

/// Representative F32 inputs covering normal, subnormal, ±0, ±inf,
/// NaN, and rounding edges.
fn f32_narrow_samples() -> Vec<u64> {
    let mut samples = Vec::new();
    // Walk every F32 exponent with a few mantissas + both signs.
    for biased_exp in 0u64..=0xff {
        for mant in [0u64, 1, 0x100, 0x400000, 0x7ffffe, 0x7fffff] {
            for sign in 0..2u64 {
                samples.push((sign << 31) | (biased_exp << 23) | mant);
            }
        }
    }
    // Handful of explicit constants.
    samples.extend([
        0x3f800000, // 1.0
        0xbf800000, // -1.0
        0x40490fdb, // pi
        0x7f7fffff, // F32::MAX
        0xff7fffff, // -F32::MAX
        0x00800000, // smallest normal
        0x007fffff, // largest subnormal
        0x00000001, // smallest subnormal
        0x7f800000, // +inf
        0xff800000, // -inf
        0x7fc00000, // canonical NaN
    ]);
    samples
}

#[test]
fn narrow_f32_to_f32_is_noop() {
    assert_narrow(NumericDType::F32, NumericDType::F32, f32_narrow_samples());
}

#[test]
fn narrow_f32_to_bf16() {
    assert_narrow(NumericDType::F32, NumericDType::BF16, f32_narrow_samples());
}

#[test]
fn narrow_f32_to_f16() {
    assert_narrow(NumericDType::F32, NumericDType::F16, f32_narrow_samples());
}

#[test]
fn narrow_f32_to_f8e5m2() {
    assert_narrow(
        NumericDType::F32,
        NumericDType::F8E5M2,
        f32_narrow_samples(),
    );
}

#[test]
fn narrow_f32_to_f8e4m3fn() {
    assert_narrow(
        NumericDType::F32,
        NumericDType::F8E4M3FN,
        f32_narrow_samples(),
    );
}

#[test]
fn narrow_f32_to_f4e2m1() {
    assert_narrow(
        NumericDType::F32,
        NumericDType::F4E2M1,
        f32_narrow_samples(),
    );
}

#[test]
fn narrow_f32_to_f6e3m2() {
    assert_narrow(
        NumericDType::F32,
        NumericDType::F6E3M2,
        f32_narrow_samples(),
    );
}

#[test]
fn narrow_f32_to_f6e2m3() {
    assert_narrow(
        NumericDType::F32,
        NumericDType::F6E2M3,
        f32_narrow_samples(),
    );
}

// ─────────────────────────────────────────────────────────────────────
// Int narrow_to
// ─────────────────────────────────────────────────────────────────────

fn int_narrow_samples() -> Vec<u64> {
    let mut samples = Vec::new();
    // Boundary values for various widths.
    samples.extend([
        0u64,
        1,
        2,
        7,
        8,
        15,
        16,
        127,
        128,
        255,
        256,
        0x7fff,
        0x8000,
        0xffff,
        0x10000,
        0x7fffffff,
        0x80000000,
        0xffffffff,
        0x100000000,
        i64::MAX as u64,
        (i64::MIN as u64),
        u64::MAX,
        u64::MAX - 1,
    ]);
    // A handful of negative-as-i64 values.
    for v in [-1i64, -2, -7, -8, -127, -128, -129, -32768, -32769] {
        samples.push(v as u64);
    }
    samples
}

#[test]
fn narrow_i64_to_i64_is_noop() {
    assert_narrow(NumericDType::I64, NumericDType::I64, int_narrow_samples());
}

#[test]
fn narrow_i64_to_u64() {
    // Not a no-op: signed i64 → u64 must clamp negatives to 0
    // (matching `IntType::clamp_unsigned` / `cast_raw`).
    assert_narrow(NumericDType::I64, NumericDType::U64, int_narrow_samples());
}

#[test]
fn narrow_i64_to_i32() {
    assert_narrow(NumericDType::I64, NumericDType::I32, int_narrow_samples());
}

#[test]
fn narrow_i64_to_u32() {
    assert_narrow(NumericDType::I64, NumericDType::U32, int_narrow_samples());
}

#[test]
fn narrow_i64_to_i16() {
    assert_narrow(NumericDType::I64, NumericDType::I16, int_narrow_samples());
}

#[test]
fn narrow_i64_to_i8() {
    assert_narrow(NumericDType::I64, NumericDType::I8, int_narrow_samples());
}

#[test]
fn narrow_i64_to_u8() {
    assert_narrow(NumericDType::I64, NumericDType::U8, int_narrow_samples());
}

#[test]
fn narrow_i64_to_i4() {
    assert_narrow(NumericDType::I64, NumericDType::I4, int_narrow_samples());
}

#[test]
fn narrow_i64_to_bool() {
    assert_narrow(NumericDType::I64, NumericDType::BOOL, int_narrow_samples());
}
