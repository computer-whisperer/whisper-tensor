//! Fast-path decode/encode helpers for F32, BF16, and F16.
//!
//! These bypass the generic FloatType software engine and operate in f32
//! precision, avoiding the f64 round-trip that the generic path requires.
//! Returns `None` for other float types (F64, F8, F4, etc.) so the caller
//! falls back to the generic path at full precision.

use crate::numeric_dtype::FloatType;

/// If `ft` is F32, BF16, or F16, decode raw bits to f32 directly.
#[inline(always)]
pub fn decode_f32(raw: u64, ft: &FloatType) -> Option<f32> {
    match (ft.exponent_bits, ft.mantissa_bits) {
        (8, 23) => Some(f32::from_bits(raw as u32)),
        (8, 7) => Some(half::bf16::from_bits(raw as u16).to_f32()),
        (5, 10) => Some(half::f16::from_bits(raw as u16).to_f32()),
        _ => None,
    }
}

/// If `ft` is F32, BF16, or F16, encode an f32 value to raw bits directly.
#[inline(always)]
pub fn encode_f32(value: f32, ft: &FloatType) -> Option<u64> {
    match (ft.exponent_bits, ft.mantissa_bits) {
        (8, 23) => Some(value.to_bits() as u64),
        (8, 7) => Some(half::bf16::from_f32(value).to_bits() as u64),
        (5, 10) => Some(half::f16::from_f32(value).to_bits() as u64),
        _ => None,
    }
}

/// Apply a binary f32 operation. Returns `None` if `ft` isn't F32/BF16/F16.
#[inline(always)]
pub fn binary_f32(a: u64, b: u64, ft: &FloatType, op: fn(f32, f32) -> f32) -> Option<u64> {
    let va = decode_f32(a, ft)?;
    let vb = decode_f32(b, ft)?;
    encode_f32(op(va, vb), ft)
}

/// Apply a unary f32 operation. Returns `None` if `ft` isn't F32/BF16/F16.
#[inline(always)]
pub fn unary_f32(raw: u64, ft: &FloatType, op: fn(f32) -> f32) -> Option<u64> {
    let v = decode_f32(raw, ft)?;
    encode_f32(op(v), ft)
}
