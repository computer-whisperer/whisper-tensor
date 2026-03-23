//! Unary negation.

use crate::numeric_dtype::{FloatType, IntType};

/// IEEE float negation: flips the sign bit.
/// Correct for all values including NaN, Inf, and zero (preserves payload).
pub fn float_neg(raw: u64, ft: &FloatType) -> u64 {
    let sign_pos = ft.total_bits() as u64 - 1;
    raw ^ (1u64 << sign_pos)
}

/// Signed integer negation with wrapping overflow.
/// `-MIN` wraps back to `MIN` (2's complement behavior).
pub fn signed_neg_wrapping(raw: u64, it: &IntType) -> u64 {
    let v = it.decode_signed(raw);
    // In i128, negation of i64::MIN gives i64::MAX+1. Wrapping means
    // we take the low bits: reinterpret as the original width.
    let negated = v.wrapping_neg();
    // Truncate to type width (wrapping = keep low bits, not clamp)
    let mask = if it.bits >= 64 {
        u64::MAX
    } else {
        (1u64 << it.bits) - 1
    };
    (negated as i64 as u64) & mask
}

/// Signed integer negation with saturating overflow.
/// `-MIN` saturates to `MAX`.
pub fn signed_neg_saturating(raw: u64, it: &IntType) -> u64 {
    let v = it.decode_signed(raw);
    it.encode_signed(v.wrapping_neg())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric_dtype::FloatType;

    // -- Float negation --

    #[test]
    fn float_neg_f64() {
        let ft = FloatType::F64;

        let one = ft.encode_f64(1.0);
        assert_eq!(ft.decode_f64(float_neg(one, &ft)), -1.0);

        let neg_pi = ft.encode_f64(-3.14);
        assert_eq!(ft.decode_f64(float_neg(neg_pi, &ft)), 3.14);

        // 0.0 → -0.0
        let zero = ft.encode_f64(0.0);
        let result = ft.decode_f64(float_neg(zero, &ft));
        assert!(result == 0.0 && result.is_sign_negative());

        // -0.0 → 0.0
        let nz = ft.encode_f64(-0.0);
        let result = ft.decode_f64(float_neg(nz, &ft));
        assert!(result == 0.0 && result.is_sign_positive());

        // NaN stays NaN
        let nan = ft.encode_f64(f64::NAN);
        assert!(ft.decode_f64(float_neg(nan, &ft)).is_nan());

        // Inf → -Inf
        let inf = ft.encode_f64(f64::INFINITY);
        assert_eq!(ft.decode_f64(float_neg(inf, &ft)), f64::NEG_INFINITY);
    }

    #[test]
    fn float_neg_bf16() {
        let ft = FloatType::BF16;
        let v = ft.encode_f64(3.0);
        let result = ft.decode_f64(float_neg(v, &ft));
        assert!((result - (-3.0)).abs() < 0.1);
    }

    #[test]
    fn float_neg_f4e2m1() {
        let ft = FloatType::F4E2M1;
        // 6.0 (0b0111) → -6.0 (0b1111)
        let v = ft.encode_f64(6.0);
        assert_eq!(v, 0b0111);
        let nv = float_neg(v, &ft);
        assert_eq!(nv, 0b1111);
        assert_eq!(ft.decode_f64(nv), -6.0);
    }

    // -- Signed int negation (wrapping) --

    #[test]
    fn signed_neg_wrapping_basic() {
        let it = IntType::BITS_32;
        let v = it.encode_signed(42);
        assert_eq!(it.decode_signed(signed_neg_wrapping(v, &it)), -42);

        let v = it.encode_signed(-1);
        assert_eq!(it.decode_signed(signed_neg_wrapping(v, &it)), 1);

        let v = it.encode_signed(0);
        assert_eq!(it.decode_signed(signed_neg_wrapping(v, &it)), 0);
    }

    #[test]
    fn signed_neg_wrapping_min_wraps() {
        let it = IntType::BITS_32;
        // -MIN wraps back to MIN in 32-bit 2's complement
        let v = it.encode_signed(i32::MIN as i128);
        let result = it.decode_signed(signed_neg_wrapping(v, &it));
        assert_eq!(result, i32::MIN as i128);
    }

    #[test]
    fn signed_neg_wrapping_i64_min() {
        let it = IntType::BITS_64;
        let v = it.encode_signed(i64::MIN as i128);
        let result = it.decode_signed(signed_neg_wrapping(v, &it));
        assert_eq!(result, i64::MIN as i128);
    }

    // -- Signed int negation (saturating) --

    #[test]
    fn signed_neg_saturating_basic() {
        let it = IntType::BITS_32;
        let v = it.encode_signed(42);
        assert_eq!(it.decode_signed(signed_neg_saturating(v, &it)), -42);
    }

    #[test]
    fn signed_neg_saturating_min_saturates() {
        let it = IntType::BITS_32;
        let v = it.encode_signed(i32::MIN as i128);
        let result = it.decode_signed(signed_neg_saturating(v, &it));
        assert_eq!(result, i32::MAX as i128);
    }

    #[test]
    fn signed_neg_saturating_i64_min() {
        let it = IntType::BITS_64;
        let v = it.encode_signed(i64::MIN as i128);
        let result = it.decode_signed(signed_neg_saturating(v, &it));
        assert_eq!(result, i64::MAX as i128);
    }
}
