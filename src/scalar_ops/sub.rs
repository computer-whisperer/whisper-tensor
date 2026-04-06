//! Binary subtraction.

use crate::numeric_dtype::{FloatType, IntType};

/// IEEE float subtraction. Result encoded back to the same FloatType.
pub fn float_sub(a: u64, b: u64, ft: &FloatType) -> u64 {
    if let Some(r) = super::fast::binary_f32(a, b, ft, std::ops::Sub::sub) {
        return r;
    }
    ft.encode_f64(ft.decode_f64(a) - ft.decode_f64(b))
}

/// Signed integer subtraction with wrapping overflow.
/// Computes in i128, truncates to type width (keeps low bits).
pub fn signed_sub_wrapping(a: u64, b: u64, it: &IntType) -> u64 {
    let va = it.decode_signed(a);
    let vb = it.decode_signed(b);
    let diff = va.wrapping_sub(vb);
    let mask = if it.bits >= 64 {
        u64::MAX
    } else {
        (1u64 << it.bits) - 1
    };
    (diff as i64 as u64) & mask
}

/// Signed integer subtraction with saturating overflow.
/// Computes in i128, clamps to type range.
pub fn signed_sub_saturating(a: u64, b: u64, it: &IntType) -> u64 {
    let va = it.decode_signed(a);
    let vb = it.decode_signed(b);
    it.encode_signed(va - vb)
}

/// Unsigned integer subtraction with wrapping overflow.
/// Computes in u128, truncates to type width.
pub fn unsigned_sub_wrapping(a: u64, b: u64, it: &IntType) -> u64 {
    let va = it.decode_unsigned(a);
    let vb = it.decode_unsigned(b);
    let diff = va.wrapping_sub(vb);
    if it.bits >= 64 {
        diff as u64
    } else {
        (diff as u64) & ((1u64 << it.bits) - 1)
    }
}

/// Unsigned integer subtraction with saturating overflow.
/// Computes in u128, clamps to type range (min 0).
pub fn unsigned_sub_saturating(a: u64, b: u64, it: &IntType) -> u64 {
    let va = it.decode_unsigned(a);
    let vb = it.decode_unsigned(b);
    if vb > va {
        0
    } else {
        it.encode_unsigned(va - vb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Float subtraction --

    #[test]
    fn float_sub_f64() {
        let ft = FloatType::F64;
        let a = ft.encode_f64(5.0);
        let b = ft.encode_f64(3.0);
        assert_eq!(ft.decode_f64(float_sub(a, b, &ft)), 2.0);
    }

    #[test]
    fn float_sub_f32() {
        let ft = FloatType::F32;
        let a = ft.encode_f64(10.0);
        let b = ft.encode_f64(3.0);
        assert!((ft.decode_f64(float_sub(a, b, &ft)) - 7.0).abs() < 1e-6);
    }

    #[test]
    fn float_sub_inf() {
        let ft = FloatType::F32;
        let inf = ft.encode_f64(f64::INFINITY);
        let one = ft.encode_f64(1.0);
        assert!(ft.decode_f64(float_sub(inf, one, &ft)).is_infinite());

        // inf - inf = NaN
        assert!(ft.decode_f64(float_sub(inf, inf, &ft)).is_nan());
    }

    #[test]
    fn float_sub_nan() {
        let ft = FloatType::F64;
        let nan = ft.encode_f64(f64::NAN);
        let one = ft.encode_f64(1.0);
        assert!(ft.decode_f64(float_sub(nan, one, &ft)).is_nan());
    }

    #[test]
    fn float_sub_zero() {
        let ft = FloatType::F64;
        let a = ft.encode_f64(0.0);
        let b = ft.encode_f64(0.0);
        assert_eq!(ft.decode_f64(float_sub(a, b, &ft)), 0.0);
    }

    // -- Signed integer subtraction (wrapping) --

    #[test]
    fn signed_sub_wrapping_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_signed(10);
        let b = it.encode_signed(3);
        assert_eq!(it.decode_signed(signed_sub_wrapping(a, b, &it)), 7);

        let a = it.encode_signed(-5);
        let b = it.encode_signed(3);
        assert_eq!(it.decode_signed(signed_sub_wrapping(a, b, &it)), -8);
    }

    #[test]
    fn signed_sub_wrapping_overflow() {
        let it = IntType::BITS_8;
        // -100 - 100 = -200 → wraps in i8 to 56
        let a = it.encode_signed(-100);
        let b = it.encode_signed(100);
        let result = it.decode_signed(signed_sub_wrapping(a, b, &it));
        assert_eq!(result, (-200i16 as i8) as i128); // 56
    }

    // -- Signed integer subtraction (saturating) --

    #[test]
    fn signed_sub_saturating_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_signed(10);
        let b = it.encode_signed(3);
        assert_eq!(it.decode_signed(signed_sub_saturating(a, b, &it)), 7);
    }

    #[test]
    fn signed_sub_saturating_overflow() {
        let it = IntType::BITS_8;
        // -100 - 100 = -200 → saturates to -128
        let a = it.encode_signed(-100);
        let b = it.encode_signed(100);
        assert_eq!(it.decode_signed(signed_sub_saturating(a, b, &it)), -128);
    }

    // -- Unsigned integer subtraction (wrapping) --

    #[test]
    fn unsigned_sub_wrapping_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_unsigned(200);
        let b = it.encode_unsigned(100);
        assert_eq!(it.decode_unsigned(unsigned_sub_wrapping(a, b, &it)), 100);
    }

    #[test]
    fn unsigned_sub_wrapping_underflow() {
        let it = IntType::BITS_8;
        // 10 - 20 wraps in u8 to 246
        let a = it.encode_unsigned(10);
        let b = it.encode_unsigned(20);
        let result = it.decode_unsigned(unsigned_sub_wrapping(a, b, &it));
        assert_eq!(result, 246);
    }

    // -- Unsigned integer subtraction (saturating) --

    #[test]
    fn unsigned_sub_saturating_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_unsigned(200);
        let b = it.encode_unsigned(100);
        assert_eq!(it.decode_unsigned(unsigned_sub_saturating(a, b, &it)), 100);
    }

    #[test]
    fn unsigned_sub_saturating_underflow() {
        let it = IntType::BITS_8;
        // 10 - 20 saturates to 0
        let a = it.encode_unsigned(10);
        let b = it.encode_unsigned(20);
        assert_eq!(it.decode_unsigned(unsigned_sub_saturating(a, b, &it)), 0);
    }
}
