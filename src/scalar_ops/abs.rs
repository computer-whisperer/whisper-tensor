//! Unary absolute value.

use crate::numeric_dtype::{FloatType, IntType};

/// IEEE float absolute value: clears the sign bit.
/// Correct for all values including NaN, Inf, and zero.
pub fn float_abs(raw: u64, ft: &FloatType) -> u64 {
    let sign_pos = ft.total_bits() as u64 - 1;
    raw & !(1u64 << sign_pos)
}

/// Signed integer absolute value with wrapping overflow.
/// `abs(MIN)` wraps back to `MIN`.
pub fn signed_abs_wrapping(raw: u64, it: &IntType) -> u64 {
    let v = it.decode_signed(raw);
    let result = v.wrapping_neg();
    if v < 0 {
        let mask = if it.bits >= 64 {
            u64::MAX
        } else {
            (1u64 << it.bits) - 1
        };
        (result as i64 as u64) & mask
    } else {
        raw
    }
}

/// Signed integer absolute value with saturating overflow.
/// `abs(MIN)` saturates to `MAX`.
pub fn signed_abs_saturating(raw: u64, it: &IntType) -> u64 {
    let v = it.decode_signed(raw);
    if v < 0 {
        it.encode_signed(v.wrapping_neg())
    } else {
        raw
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Float abs --

    #[test]
    fn float_abs_f64() {
        let ft = FloatType::F64;

        let neg = ft.encode_f64(-3.14);
        assert_eq!(ft.decode_f64(float_abs(neg, &ft)), 3.14);

        let pos = ft.encode_f64(2.71);
        assert_eq!(ft.decode_f64(float_abs(pos, &ft)), 2.71);

        let zero = ft.encode_f64(0.0);
        let result = ft.decode_f64(float_abs(zero, &ft));
        assert!(result == 0.0 && result.is_sign_positive());

        let neg_zero = ft.encode_f64(-0.0);
        let result = ft.decode_f64(float_abs(neg_zero, &ft));
        assert!(result == 0.0 && result.is_sign_positive());
    }

    #[test]
    fn float_abs_nan() {
        let ft = FloatType::F64;
        let nan = ft.encode_f64(f64::NAN);
        assert!(ft.decode_f64(float_abs(nan, &ft)).is_nan());
    }

    #[test]
    fn float_abs_inf() {
        let ft = FloatType::F64;
        let neg_inf = ft.encode_f64(f64::NEG_INFINITY);
        assert_eq!(ft.decode_f64(float_abs(neg_inf, &ft)), f64::INFINITY);

        let inf = ft.encode_f64(f64::INFINITY);
        assert_eq!(ft.decode_f64(float_abs(inf, &ft)), f64::INFINITY);
    }

    #[test]
    fn float_abs_bf16() {
        let ft = FloatType::BF16;
        let v = ft.encode_f64(-3.0);
        assert!((ft.decode_f64(float_abs(v, &ft)) - 3.0).abs() < 0.1);
    }

    #[test]
    fn float_abs_f4e2m1() {
        let ft = FloatType::F4E2M1;
        let v = ft.encode_f64(-6.0);
        assert_eq!(ft.decode_f64(float_abs(v, &ft)), 6.0);
    }

    // -- Signed abs (wrapping) --

    #[test]
    fn signed_abs_wrapping_basic() {
        let it = IntType::BITS_32;
        let v = it.encode_signed(-42);
        assert_eq!(it.decode_signed(signed_abs_wrapping(v, &it)), 42);

        let v = it.encode_signed(42);
        assert_eq!(it.decode_signed(signed_abs_wrapping(v, &it)), 42);

        let v = it.encode_signed(0);
        assert_eq!(it.decode_signed(signed_abs_wrapping(v, &it)), 0);
    }

    #[test]
    fn signed_abs_wrapping_min_wraps() {
        let it = IntType::BITS_8;
        let v = it.encode_signed(-128);
        let result = it.decode_signed(signed_abs_wrapping(v, &it));
        assert_eq!(result, -128); // wraps back to MIN
    }

    // -- Signed abs (saturating) --

    #[test]
    fn signed_abs_saturating_basic() {
        let it = IntType::BITS_32;
        let v = it.encode_signed(-42);
        assert_eq!(it.decode_signed(signed_abs_saturating(v, &it)), 42);
    }

    #[test]
    fn signed_abs_saturating_min_saturates() {
        let it = IntType::BITS_8;
        let v = it.encode_signed(-128);
        let result = it.decode_signed(signed_abs_saturating(v, &it));
        assert_eq!(result, 127); // saturates to MAX
    }
}
