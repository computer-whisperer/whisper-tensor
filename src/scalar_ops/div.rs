//! Binary division.

use crate::numeric_dtype::{FloatType, IntType};

/// IEEE float division. Result encoded back to the same FloatType.
pub fn float_div(a: u64, b: u64, ft: &FloatType) -> u64 {
    ft.encode_f64(ft.decode_f64(a) / ft.decode_f64(b))
}

/// Signed integer division with wrapping overflow (truncate toward zero).
/// Division by zero returns 0. MIN / -1 wraps back to MIN.
pub fn signed_div_wrapping(a: u64, b: u64, it: &IntType) -> u64 {
    let va = it.decode_signed(a);
    let vb = it.decode_signed(b);
    if vb == 0 {
        return 0;
    }
    // In i128, MIN/-1 doesn't overflow, but we need wrapping at the type width.
    let quot = va / vb;
    let mask = if it.bits >= 64 {
        u64::MAX
    } else {
        (1u64 << it.bits) - 1
    };
    (quot as i64 as u64) & mask
}

/// Signed integer division with saturating overflow (truncate toward zero).
/// Division by zero returns 0. MIN / -1 saturates to MAX.
pub fn signed_div_saturating(a: u64, b: u64, it: &IntType) -> u64 {
    let va = it.decode_signed(a);
    let vb = it.decode_signed(b);
    if vb == 0 {
        return 0;
    }
    it.encode_signed(va / vb)
}

/// Unsigned integer division with wrapping overflow (truncate toward zero).
/// Division by zero returns 0.
pub fn unsigned_div_wrapping(a: u64, b: u64, it: &IntType) -> u64 {
    let va = it.decode_unsigned(a);
    let vb = it.decode_unsigned(b);
    if vb == 0 {
        return 0;
    }
    let quot = va / vb;
    if it.bits >= 64 {
        quot as u64
    } else {
        (quot as u64) & ((1u64 << it.bits) - 1)
    }
}

/// Unsigned integer division with saturating overflow (truncate toward zero).
/// Division by zero returns 0.
pub fn unsigned_div_saturating(a: u64, b: u64, it: &IntType) -> u64 {
    let va = it.decode_unsigned(a);
    let vb = it.decode_unsigned(b);
    if vb == 0 {
        return 0;
    }
    it.encode_unsigned(va / vb)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Float division --

    #[test]
    fn float_div_f64() {
        let ft = FloatType::F64;
        let a = ft.encode_f64(10.0);
        let b = ft.encode_f64(4.0);
        assert_eq!(ft.decode_f64(float_div(a, b, &ft)), 2.5);
    }

    #[test]
    fn float_div_by_zero() {
        let ft = FloatType::F64;
        let a = ft.encode_f64(1.0);
        let b = ft.encode_f64(0.0);
        assert_eq!(ft.decode_f64(float_div(a, b, &ft)), f64::INFINITY);

        let a = ft.encode_f64(-1.0);
        assert_eq!(ft.decode_f64(float_div(a, b, &ft)), f64::NEG_INFINITY);

        let a = ft.encode_f64(0.0);
        assert!(ft.decode_f64(float_div(a, b, &ft)).is_nan());
    }

    #[test]
    fn float_div_nan() {
        let ft = FloatType::F64;
        let nan = ft.encode_f64(f64::NAN);
        let one = ft.encode_f64(1.0);
        assert!(ft.decode_f64(float_div(nan, one, &ft)).is_nan());
    }

    #[test]
    fn float_div_f32() {
        let ft = FloatType::F32;
        let a = ft.encode_f64(7.0);
        let b = ft.encode_f64(2.0);
        assert!((ft.decode_f64(float_div(a, b, &ft)) - 3.5).abs() < 1e-6);
    }

    // -- Signed integer division (wrapping) --

    #[test]
    fn signed_div_wrapping_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_signed(20);
        let b = it.encode_signed(3);
        assert_eq!(it.decode_signed(signed_div_wrapping(a, b, &it)), 6);

        let a = it.encode_signed(-20);
        let b = it.encode_signed(3);
        assert_eq!(it.decode_signed(signed_div_wrapping(a, b, &it)), -6);
    }

    #[test]
    fn signed_div_wrapping_by_zero() {
        let it = IntType::BITS_32;
        let a = it.encode_signed(42);
        let b = it.encode_signed(0);
        assert_eq!(signed_div_wrapping(a, b, &it), 0);
    }

    #[test]
    fn signed_div_wrapping_min_by_neg1() {
        let it = IntType::BITS_8;
        // -128 / -1 = 128 → wraps in i8 to -128
        let a = it.encode_signed(-128);
        let b = it.encode_signed(-1);
        let result = it.decode_signed(signed_div_wrapping(a, b, &it));
        assert_eq!(result, -128);
    }

    // -- Signed integer division (saturating) --

    #[test]
    fn signed_div_saturating_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_signed(20);
        let b = it.encode_signed(3);
        assert_eq!(it.decode_signed(signed_div_saturating(a, b, &it)), 6);
    }

    #[test]
    fn signed_div_saturating_by_zero() {
        let it = IntType::BITS_32;
        let a = it.encode_signed(42);
        let b = it.encode_signed(0);
        assert_eq!(signed_div_saturating(a, b, &it), 0);
    }

    #[test]
    fn signed_div_saturating_min_by_neg1() {
        let it = IntType::BITS_8;
        // -128 / -1 = 128 → saturates to 127
        let a = it.encode_signed(-128);
        let b = it.encode_signed(-1);
        let result = it.decode_signed(signed_div_saturating(a, b, &it));
        assert_eq!(result, 127);
    }

    // -- Unsigned integer division --

    #[test]
    fn unsigned_div_wrapping_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_unsigned(200);
        let b = it.encode_unsigned(7);
        assert_eq!(it.decode_unsigned(unsigned_div_wrapping(a, b, &it)), 28);
    }

    #[test]
    fn unsigned_div_wrapping_by_zero() {
        let it = IntType::BITS_32;
        let a = it.encode_unsigned(42);
        let b = it.encode_unsigned(0);
        assert_eq!(unsigned_div_wrapping(a, b, &it), 0);
    }

    #[test]
    fn unsigned_div_saturating_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_unsigned(200);
        let b = it.encode_unsigned(7);
        assert_eq!(it.decode_unsigned(unsigned_div_saturating(a, b, &it)), 28);
    }

    #[test]
    fn unsigned_div_saturating_by_zero() {
        let it = IntType::BITS_32;
        let a = it.encode_unsigned(42);
        let b = it.encode_unsigned(0);
        assert_eq!(unsigned_div_saturating(a, b, &it), 0);
    }
}
