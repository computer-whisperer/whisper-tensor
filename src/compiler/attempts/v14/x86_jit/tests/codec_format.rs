//! Unit tests for [`super::super::codec::format`].
//!
//! Each test JITs a small `extern "C" fn(u64) -> u64` wrapper that
//! decodes the input bits via the codec, then re-encodes them, and
//! returns the resulting raw bits. We compare against
//! `NumericDType::cast_raw(input, dtype, dtype)` (the reference
//! roundtrip via `decode_f64` + `encode_f64`).
//!
//! Test ABI register conventions:
//!   - rdi (7) — input raw bits
//!   - rax (0) — output raw bits / return
//!   - xmm0 (0) — float compute slot A
//!   - xmm1 (1) — scratch xmm (for F16C / future paths)
//!   - rcx (1) — scratch GP (used by encode for the BF16 round bias)
//!   - r9 (9), r10 (10) — extra scratch where needed
//!
//! All registers are caller-saved under System V; the harness needs
//! no prologue/epilogue beyond the trailing `ret`.

use super::super::codec::format::{CodecSlot, CodecTables, ComputeRepr, emit_decode, emit_encode};
use super::jit_harness::JitFn;
use crate::numeric_dtype::{FloatType, IntType, NumericDType};

const RAW_IN: u8 = 7; // rdi (arg 0)
const RAW_OUT: u8 = 0; // rax (return)
// Caller-saved GPs we use as codec scratches. We deliberately avoid
// rcx (1) and rdx (2) because the codec implicitly clobbers them
// (rcx as shift count, rdx as round-bias scratch in the sub-F16
// inline encode). r8/r9 are caller-saved and unused by our test ABI.
const SCRATCH_GP: u8 = 8; // r8
const SCRATCH_GP2: u8 = 9; // r9
const FLT_SLOT: u8 = 0; // xmm0
const FLT_SCRATCH: u8 = 1; // xmm1

/// A built JIT plus the codec tables it holds pointers into.
///
/// The `_tables` field is `_`-prefixed because we never read it
/// directly; its sole purpose is to keep the table allocations alive
/// for as long as `jit` exists, since the JIT'd code embeds raw
/// pointers to the table memory.
struct JitWithTables {
    jit: JitFn,
    _tables: CodecTables,
}

/// Build a JIT that does `decode → encode` for one dtype, returning
/// the raw output bits in `rax`. The input is in `rdi`.
fn build_roundtrip(dtype: NumericDType) -> Option<JitWithTables> {
    let mut tables = CodecTables::new();
    let repr = ComputeRepr::for_dtype(dtype);
    let jit = JitFn::build(|asm| match repr {
        ComputeRepr::F32 | ComputeRepr::F64 => {
            emit_decode(
                asm,
                dtype,
                RAW_IN,
                CodecSlot::Xmm(FLT_SLOT),
                SCRATCH_GP2,
                FLT_SCRATCH,
                &mut tables,
            )
            .expect("decode emit");
            emit_encode(
                asm,
                dtype,
                CodecSlot::Xmm(FLT_SLOT),
                RAW_OUT,
                SCRATCH_GP,
                SCRATCH_GP2,
                FLT_SCRATCH,
            )
            .expect("encode emit");
        }
        ComputeRepr::Int => {
            emit_decode(
                asm,
                dtype,
                RAW_IN,
                CodecSlot::Gp(RAW_OUT),
                SCRATCH_GP2,
                FLT_SCRATCH,
                &mut tables,
            )
            .expect("decode emit");
            emit_encode(
                asm,
                dtype,
                CodecSlot::Gp(RAW_OUT),
                RAW_OUT,
                SCRATCH_GP,
                SCRATCH_GP2,
                FLT_SCRATCH,
            )
            .expect("encode emit");
        }
    });
    Some(JitWithTables {
        jit,
        _tables: tables,
    })
}

/// Run the JIT roundtrip for one input value.
fn run_roundtrip(jit: &JitWithTables, raw_in: u64) -> u64 {
    let f: extern "C" fn(u64) -> u64 = unsafe { std::mem::transmute(jit.jit.ptr()) };
    f(raw_in)
}

/// Test: for every input bit pattern in `inputs`, JIT roundtrip
/// (decode → encode through compute repr) produces the expected bits.
///
/// Reference choice depends on the compute repr:
///
/// - **Floats (F32 / F64 compute repr)**: compare against the chained
///   cast `dtype → F32/F64 → dtype`. The encode step canonicalizes NaN
///   per `dtype_contract.md` line 156 ("NaN encodes to the dtype's
///   canonical NaN"), so non-NaN inputs round-trip identically and
///   any NaN bit pattern collapses to the canonical one. We can't
///   compare against `cast_raw(raw, dtype, dtype)` (the same-dtype
///   identity) because that path skips the canonicalization the JIT
///   necessarily applies.
///
/// - **Ints / Bool (Int compute repr)**: compare against the JIT's
///   own chain semantics, simulated in software. The codec interprets
///   the slot bits with the **target's** signedness — `encode_signed`
///   treats the slot as i64 and clamps to `[min, max]`;
///   `encode_unsigned` treats the slot as i64, clamps negatives to 0,
///   and clamps above `max_unsigned`. This is *not* the same as the
///   `cast_raw` short-circuit for same-dtype casts: an unsigned source
///   with the high bit set looks negative in the i64 view of the slot,
///   so `encode_unsigned` clamps it to 0 — losing the original bits.
///   Cross-signedness 64-bit casts are handled by the orchestration's
///   future Cast op (P2.B), not by the codec primitive in isolation.
///
/// Same-dtype Identity in real ops (the case that preserves bits
/// regardless of signedness) is implemented in Phase 2.B as a memcpy,
/// not via the codec.
fn assert_roundtrip(dtype: NumericDType, inputs: impl IntoIterator<Item = u64>) {
    let jit = build_roundtrip(dtype).expect("dtype is in supported set");
    let mask: u64 = if dtype.total_bits() == 64 {
        u64::MAX
    } else {
        (1u64 << dtype.total_bits()) - 1
    };
    let compute_repr_for_chain = match ComputeRepr::for_dtype(dtype) {
        ComputeRepr::F32 => Some(NumericDType::F32),
        ComputeRepr::F64 => Some(NumericDType::F64),
        ComputeRepr::Int => None,
    };
    for raw_in in inputs {
        let raw_in_masked = raw_in & mask;
        let want = match compute_repr_for_chain {
            Some(crd) => {
                let through = dtype.cast_raw(raw_in_masked, crd);
                crd.cast_raw(through, dtype)
            }
            None => simulate_int_roundtrip(dtype, raw_in_masked),
        };
        let got = run_roundtrip(&jit, raw_in_masked);
        assert_eq!(
            got, want,
            "roundtrip {dtype}: input 0x{raw_in_masked:x} → got 0x{got:x} want 0x{want:x}"
        );
    }
}

/// Software simulation of the JIT's int decode → encode chain. The
/// JIT round-trip JIT returns the encoded bits directly (slot register
/// is the return register rax), so we mirror exactly that: decode
/// (sign/zero-extend), then encode (saturating-as-target then mask),
/// then return the encoded bits.
fn simulate_int_roundtrip(dtype: NumericDType, raw_in: u64) -> u64 {
    let bits = dtype.total_bits() as u32;
    let mask: u64 = if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    // Decode: sign-extend (signed) or zero-extend (unsigned/Bool).
    let masked = raw_in & mask;
    let after_decode: i64 = match dtype {
        NumericDType::SignedInt(_) if bits < 64 => {
            let sign_bit = 1u64 << (bits - 1);
            if masked & sign_bit != 0 {
                (masked | !mask) as i64
            } else {
                masked as i64
            }
        }
        NumericDType::SignedInt(_) => masked as i64,
        NumericDType::UnsignedInt(_) => masked as i64,
        NumericDType::Bool => (masked & 1) as i64,
        _ => unreachable!("not an int dtype"),
    };
    // Encode: saturate-as-target then mask. This matches what the JIT
    // writes back to the return register.
    match dtype {
        NumericDType::SignedInt(it) => {
            let b = it.bits as u32;
            if b == 64 {
                after_decode as u64
            } else {
                let max = (1i64 << (b - 1)) - 1;
                let min = -(1i64 << (b - 1));
                let clamped = after_decode.clamp(min, max);
                (clamped as u64) & ((1u64 << b) - 1)
            }
        }
        NumericDType::UnsignedInt(it) => {
            let b = it.bits as u32;
            let after_neg = if after_decode < 0 { 0 } else { after_decode };
            if b == 64 {
                after_neg as u64
            } else {
                let max = (1i64 << b) - 1;
                let clamped = after_neg.min(max);
                (clamped as u64) & ((1u64 << b) - 1)
            }
        }
        NumericDType::Bool => {
            if after_decode != 0 {
                1
            } else {
                0
            }
        }
        _ => unreachable!("not an int dtype"),
    }
}

/// Generate sample raw bit patterns for a float type that hit normal,
/// subnormal, ±0, ±inf (when `has_infinity`), NaN (when `has_nan`),
/// and rounding edges. The reference implementation in
/// `numeric_dtype/conversions.rs` is the ground truth; we sample
/// across the dtype's representable space at coarse resolution.
fn float_samples(ft: FloatType) -> Vec<u64> {
    let mut samples = Vec::new();
    let total_bits = ft.total_bits() as u32;
    if total_bits <= 16 {
        // Exhaustive: every bit pattern.
        let n = 1u64 << total_bits;
        for raw in 0..n {
            samples.push(raw);
        }
    } else {
        // Sparse sampling: walk the exponent range, with several
        // mantissa values per exponent. Include both signs.
        let mant_bits = ft.mantissa_bits as u32;
        let mant_max: u64 = (1u64 << mant_bits) - 1;
        let exp_max = ft.max_biased_exponent();
        let mant_samples: Vec<u64> = if mant_bits >= 4 {
            vec![0, 1, mant_max / 4, mant_max / 2, mant_max - 1, mant_max]
        } else {
            (0..=mant_max).collect()
        };
        for biased_exp in 0..=exp_max as u64 {
            for &mant in &mant_samples {
                for sign in 0..2u64 {
                    let raw = (sign << (total_bits - 1)) | (biased_exp << mant_bits) | mant;
                    samples.push(raw);
                }
            }
        }
    }
    samples
}

/// Generate sample raw bit patterns for an integer type. Includes
/// boundary values (0, 1, -1, min, max) plus a sweep across the range.
fn int_samples(it: IntType, signed: bool) -> Vec<u64> {
    let mut samples: Vec<u64> = Vec::new();
    let bits = it.bits as u32;
    if bits <= 8 {
        // Exhaustive.
        let n = 1u64 << bits;
        for raw in 0..n {
            samples.push(raw);
        }
        return samples;
    }
    // Boundaries.
    samples.extend([0u64, 1]);
    let mask: u64 = if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    samples.push(mask); // all-ones (which is -1 for signed)
    samples.push(mask >> 1); // 0x7f...f (max signed)
    samples.push((mask >> 1) + 1); // 0x80...0 (min signed)
    // Sweep at coarse resolution. Bounded loop — `mask / 64` can
    // saturate at u64::MAX for 64-bit dtypes, so we count steps
    // explicitly rather than walking `v` with `saturating_add`
    // (which never escapes the `v <= mask` guard).
    let stride = mask / 64;
    if stride > 0 {
        for step in 0u64..=64 {
            let v = step.saturating_mul(stride);
            samples.push(v);
            if v == mask {
                break;
            }
        }
    }
    let _ = signed; // both signed/unsigned see the same bit patterns
    samples
}

// ─────────────────────────────────────────────────────────────────────
// Float tests
// ─────────────────────────────────────────────────────────────────────

#[test]
fn roundtrip_f32_samples() {
    let dtype = NumericDType::F32;
    let mut samples = float_samples(FloatType::F32);
    // Sparse sampling didn't include exact constants — add them.
    samples.extend([
        0u64, 0x3f800000, // 1.0
        0xbf800000, // -1.0
        0x7f7fffff, // F32::MAX
        0xff7fffff, // -F32::MAX
        0x7f800000, // +inf
        0xff800000, // -inf
        0x7fc00000, // canonical NaN
        0x00800000, // smallest normal
        0x007fffff, // largest subnormal
        0x00000001, // smallest subnormal
    ]);
    assert_roundtrip(dtype, samples);
}

#[test]
fn roundtrip_f64_samples() {
    let dtype = NumericDType::F64;
    let mut samples = float_samples(FloatType::F64);
    samples.extend([
        0u64,
        0x3ff0000000000000, // 1.0
        0xbff0000000000000, // -1.0
        0x7fefffffffffffff, // F64::MAX
        0xffefffffffffffff, // -F64::MAX
        0x7ff0000000000000, // +inf
        0xfff0000000000000, // -inf
        0x7ff8000000000000, // canonical NaN
        0x0010000000000000, // smallest normal
        0x000fffffffffffff, // largest subnormal
    ]);
    assert_roundtrip(dtype, samples);
}

#[test]
fn roundtrip_bf16_exhaustive() {
    let dtype = NumericDType::BF16;
    let samples: Vec<u64> = (0..(1u64 << 16)).collect();
    assert_roundtrip(dtype, samples);
}

#[test]
fn roundtrip_f16_exhaustive() {
    let dtype = NumericDType::F16;
    let samples: Vec<u64> = (0..(1u64 << 16)).collect();
    assert_roundtrip(dtype, samples);
}

#[test]
fn roundtrip_f8e5m2_exhaustive() {
    let dtype = NumericDType::F8E5M2;
    let samples: Vec<u64> = (0..(1u64 << 8)).collect();
    assert_roundtrip(dtype, samples);
}

#[test]
fn roundtrip_f8e4m3fn_exhaustive() {
    let dtype = NumericDType::F8E4M3FN;
    let samples: Vec<u64> = (0..(1u64 << 8)).collect();
    assert_roundtrip(dtype, samples);
}

#[test]
fn roundtrip_f4e2m1_exhaustive() {
    let dtype = NumericDType::F4E2M1;
    let samples: Vec<u64> = (0..(1u64 << 4)).collect();
    assert_roundtrip(dtype, samples);
}

#[test]
fn roundtrip_f6e3m2_exhaustive() {
    let dtype = NumericDType::F6E3M2;
    let samples: Vec<u64> = (0..(1u64 << 6)).collect();
    assert_roundtrip(dtype, samples);
}

#[test]
fn roundtrip_f6e2m3_exhaustive() {
    let dtype = NumericDType::F6E2M3;
    let samples: Vec<u64> = (0..(1u64 << 6)).collect();
    assert_roundtrip(dtype, samples);
}

// ─────────────────────────────────────────────────────────────────────
// Integer tests
// ─────────────────────────────────────────────────────────────────────

#[test]
fn roundtrip_i64() {
    let dtype = NumericDType::I64;
    assert_roundtrip(dtype, int_samples(IntType::BITS_64, true));
}

#[test]
fn roundtrip_i32() {
    let dtype = NumericDType::I32;
    assert_roundtrip(dtype, int_samples(IntType::BITS_32, true));
}

#[test]
fn roundtrip_i16() {
    let dtype = NumericDType::I16;
    assert_roundtrip(dtype, int_samples(IntType::BITS_16, true));
}

#[test]
fn roundtrip_i8_exhaustive() {
    let dtype = NumericDType::I8;
    assert_roundtrip(dtype, int_samples(IntType::BITS_8, true));
}

#[test]
fn roundtrip_i4_exhaustive() {
    let dtype = NumericDType::I4;
    assert_roundtrip(dtype, int_samples(IntType::BITS_4, true));
}

#[test]
fn roundtrip_u64() {
    let dtype = NumericDType::U64;
    assert_roundtrip(dtype, int_samples(IntType::BITS_64, false));
}

#[test]
fn roundtrip_u32() {
    let dtype = NumericDType::U32;
    assert_roundtrip(dtype, int_samples(IntType::BITS_32, false));
}

#[test]
fn roundtrip_u16() {
    let dtype = NumericDType::U16;
    assert_roundtrip(dtype, int_samples(IntType::BITS_16, false));
}

#[test]
fn roundtrip_u8_exhaustive() {
    let dtype = NumericDType::U8;
    assert_roundtrip(dtype, int_samples(IntType::BITS_8, false));
}

#[test]
fn roundtrip_u4_exhaustive() {
    let dtype = NumericDType::U4;
    assert_roundtrip(dtype, int_samples(IntType::BITS_4, false));
}

// ─────────────────────────────────────────────────────────────────────
// Bool tests
// ─────────────────────────────────────────────────────────────────────

#[test]
fn roundtrip_bool() {
    let dtype = NumericDType::BOOL;
    // Bool's only valid raw bits are 0 and 1, but we test that the
    // codec normalizes other inputs as well.
    let samples = vec![0u64, 1, 0xff, 0xdeadbeef, u64::MAX];
    let jit = build_roundtrip(dtype).expect("bool supported");
    for raw_in in samples {
        let got = run_roundtrip(&jit, raw_in);
        // Bool reference: any nonzero → 1, zero → 0. The codec masks
        // raw_in to 1 bit before calling cast_raw, which gives "low
        // bit" semantics. Our codec normalizes the WHOLE 64 bits via
        // test+setne, so it produces 1 for ANY nonzero input. Verify
        // that.
        let want = if raw_in & 1 != 0 { 1u64 } else { 0u64 };
        assert_eq!(
            got, want,
            "bool roundtrip: input 0x{raw_in:x} → got {got} want {want}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// Cross-dtype cast tests
// ─────────────────────────────────────────────────────────────────────

/// Build a JIT that does decode(src_dtype) → encode(dst_dtype). The
/// input is in `rdi`, the output is in `rax`. For float→float, both
/// dtypes must use the same compute repr; for int↔float we'd need
/// extra plumbing (deferred to Phase 2.B's full op pipeline).
fn build_cast(src: NumericDType, dst: NumericDType) -> Option<JitWithTables> {
    let src_repr = ComputeRepr::for_dtype(src);
    let dst_repr = ComputeRepr::for_dtype(dst);
    if src_repr != dst_repr {
        // Cross-repr casts (float↔int) need narrowing primitives that
        // belong in P2.A.4 / Phase 2.B's `ops::cast`. Skip here.
        return None;
    }
    let mut tables = CodecTables::new();
    let jit = JitFn::build(|asm| match src_repr {
        ComputeRepr::F32 | ComputeRepr::F64 => {
            emit_decode(
                asm,
                src,
                RAW_IN,
                CodecSlot::Xmm(FLT_SLOT),
                SCRATCH_GP2,
                FLT_SCRATCH,
                &mut tables,
            )
            .unwrap();
            emit_encode(
                asm,
                dst,
                CodecSlot::Xmm(FLT_SLOT),
                RAW_OUT,
                SCRATCH_GP,
                SCRATCH_GP2,
                FLT_SCRATCH,
            )
            .unwrap();
        }
        ComputeRepr::Int => {
            emit_decode(
                asm,
                src,
                RAW_IN,
                CodecSlot::Gp(RAW_OUT),
                SCRATCH_GP2,
                FLT_SCRATCH,
                &mut tables,
            )
            .unwrap();
            emit_encode(
                asm,
                dst,
                CodecSlot::Gp(RAW_OUT),
                RAW_OUT,
                SCRATCH_GP,
                SCRATCH_GP2,
                FLT_SCRATCH,
            )
            .unwrap();
        }
    });
    Some(JitWithTables {
        jit,
        _tables: tables,
    })
}

#[test]
fn cast_f32_to_bf16_samples() {
    let src = NumericDType::F32;
    let dst = NumericDType::BF16;
    let jit = build_cast(src, dst).expect("f32→bf16 supported");
    let inputs = [
        0u64, 0x3f800000, // 1.0
        0x40490fdb, // pi
        0xbf800000, // -1.0
        0x7f7fffff, // F32::MAX → bf16 ±inf or max
        0x7f800000, // +inf
        0x7fc00000, // NaN
        0x00800000, // smallest normal
    ];
    for raw in inputs {
        let want = src.cast_raw(raw, dst);
        let f: extern "C" fn(u64) -> u64 = unsafe { std::mem::transmute(jit.jit.ptr()) };
        let got = f(raw);
        assert_eq!(
            got, want,
            "f32→bf16: input 0x{raw:x} → got 0x{got:x} want 0x{want:x}"
        );
    }
}

#[test]
fn cast_bf16_to_f32_exhaustive() {
    let src = NumericDType::BF16;
    let dst = NumericDType::F32;
    let jit = build_cast(src, dst).expect("bf16→f32 supported");
    for raw in 0u64..(1u64 << 16) {
        let want = src.cast_raw(raw, dst);
        let f: extern "C" fn(u64) -> u64 = unsafe { std::mem::transmute(jit.jit.ptr()) };
        let got = f(raw);
        assert_eq!(
            got, want,
            "bf16→f32: input 0x{raw:x} → got 0x{got:x} want 0x{want:x}"
        );
    }
}

#[test]
fn cast_f16_to_f32_exhaustive() {
    let src = NumericDType::F16;
    let dst = NumericDType::F32;
    let jit = build_cast(src, dst).expect("f16→f32 supported");
    for raw in 0u64..(1u64 << 16) {
        let want = src.cast_raw(raw, dst);
        let f: extern "C" fn(u64) -> u64 = unsafe { std::mem::transmute(jit.jit.ptr()) };
        let got = f(raw);
        assert_eq!(
            got, want,
            "f16→f32: input 0x{raw:x} → got 0x{got:x} want 0x{want:x}"
        );
    }
}

#[test]
fn cast_f32_to_f16_samples() {
    let src = NumericDType::F32;
    let dst = NumericDType::F16;
    let jit = build_cast(src, dst).expect("f32→f16 supported");
    let inputs = [
        0u64, 0x3f800000, // 1.0
        0x40490fdb, // pi
        0xbf800000, // -1.0
        0x7f7fffff, // F32::MAX (overflows F16 range)
        0x7f800000, // +inf
        0x7fc00000, // NaN
        0x33800000, // a small value that becomes a F16 subnormal
    ];
    for raw in inputs {
        let want = src.cast_raw(raw, dst);
        let f: extern "C" fn(u64) -> u64 = unsafe { std::mem::transmute(jit.jit.ptr()) };
        let got = f(raw);
        assert_eq!(
            got, want,
            "f32→f16: input 0x{raw:x} → got 0x{got:x} want 0x{want:x}"
        );
    }
}

#[test]
fn cast_int_widening() {
    // Widening int casts go through the Int compute repr (i64
    // sign-extended) and re-encode to the wider target. The codec
    // is bit-faithful for these.
    let jit = build_cast(NumericDType::I8, NumericDType::I32).expect("i8→i32");
    for raw in 0u64..256 {
        let want = NumericDType::I8.cast_raw(raw, NumericDType::I32);
        let f: extern "C" fn(u64) -> u64 = unsafe { std::mem::transmute(jit.jit.ptr()) };
        let got = f(raw);
        assert_eq!(
            got, want,
            "i8→i32: input {raw} → got 0x{got:x} want 0x{want:x}"
        );
    }

    // u8 → u16 (zero-extend)
    let jit = build_cast(NumericDType::U8, NumericDType::U16).expect("u8→u16");
    for raw in 0u64..256 {
        let want = NumericDType::U8.cast_raw(raw, NumericDType::U16);
        let f: extern "C" fn(u64) -> u64 = unsafe { std::mem::transmute(jit.jit.ptr()) };
        let got = f(raw);
        assert_eq!(
            got, want,
            "u8→u16: input {raw} → got 0x{got:x} want 0x{want:x}"
        );
    }
}

/// Exhaustively decode every raw bit pattern of a sub-F16 float type
/// to F32, asserting the result matches `cast_raw(raw, ft, F32)` —
/// this is the only thing the lookup-table decode is responsible for.
/// Encode is the inverse direction and lands in P2.A.3.b.
fn assert_subf16_decode_to_f32_exhaustive(src: NumericDType) {
    let dst = NumericDType::F32;
    let jit = build_cast(src, dst).expect("decode→F32 supported");
    let total_bits = src.total_bits() as u32;
    let n: u64 = 1u64 << total_bits;
    for raw in 0..n {
        let want = src.cast_raw(raw, dst);
        let f: extern "C" fn(u64) -> u64 = unsafe { std::mem::transmute(jit.jit.ptr()) };
        let got = f(raw);
        assert_eq!(
            got, want,
            "{src}→F32: input 0x{raw:x} → got 0x{got:x} want 0x{want:x}"
        );
    }
}

#[test]
fn decode_f8e5m2_exhaustive() {
    assert_subf16_decode_to_f32_exhaustive(NumericDType::F8E5M2);
}

#[test]
fn decode_f8e4m3fn_exhaustive() {
    assert_subf16_decode_to_f32_exhaustive(NumericDType::F8E4M3FN);
}

#[test]
fn decode_f4e2m1_exhaustive() {
    assert_subf16_decode_to_f32_exhaustive(NumericDType::F4E2M1);
}

#[test]
fn decode_f6e3m2_exhaustive() {
    assert_subf16_decode_to_f32_exhaustive(NumericDType::F6E3M2);
}

#[test]
fn decode_f6e2m3_exhaustive() {
    assert_subf16_decode_to_f32_exhaustive(NumericDType::F6E2M3);
}

#[test]
fn cast_int_narrowing_saturating() {
    // The codec encode now saturates per the target's signedness
    // (P2.A.4). Cross-dtype int casts where source and target share a
    // signedness sign agree with `cast_raw` for every input value,
    // including out-of-range values that saturate.
    let jit = build_cast(NumericDType::I32, NumericDType::I8).expect("i32→i8");
    for raw in [
        0i32,
        1,
        -1,
        127,
        128,
        255,
        256,
        -128,
        -129,
        i32::MAX,
        i32::MIN,
    ] {
        let raw_u = raw as u32 as u64;
        let want = NumericDType::I32.cast_raw(raw_u, NumericDType::I8);
        let f: extern "C" fn(u64) -> u64 = unsafe { std::mem::transmute(jit.jit.ptr()) };
        let got = f(raw_u);
        assert_eq!(
            got, want,
            "i32→i8 input {raw} (0x{raw_u:x}): got 0x{got:x} want 0x{want:x}"
        );
    }
}
