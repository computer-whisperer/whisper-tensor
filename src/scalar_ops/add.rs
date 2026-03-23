//! Binary addition.

use crate::numeric_dtype::{FloatType, IntType};

/// IEEE float addition. Result encoded back to the same FloatType.
pub fn float_add(a: u64, b: u64, ft: &FloatType) -> u64 {
    ft.encode_f64(ft.decode_f64(a) + ft.decode_f64(b))
}

/// Signed integer addition with wrapping overflow.
/// Computes in i128, truncates to type width (keeps low bits).
pub fn signed_add_wrapping(a: u64, b: u64, it: &IntType) -> u64 {
    let va = it.decode_signed(a);
    let vb = it.decode_signed(b);
    let sum = va.wrapping_add(vb);
    let mask = if it.bits >= 64 {
        u64::MAX
    } else {
        (1u64 << it.bits) - 1
    };
    (sum as i64 as u64) & mask
}

/// Signed integer addition with saturating overflow.
/// Computes in i128, clamps to type range.
pub fn signed_add_saturating(a: u64, b: u64, it: &IntType) -> u64 {
    let va = it.decode_signed(a);
    let vb = it.decode_signed(b);
    it.encode_signed(va + vb)
}

/// Unsigned integer addition with wrapping overflow.
/// Computes in u128, truncates to type width.
pub fn unsigned_add_wrapping(a: u64, b: u64, it: &IntType) -> u64 {
    let va = it.decode_unsigned(a);
    let vb = it.decode_unsigned(b);
    let sum = va.wrapping_add(vb);
    if it.bits >= 64 {
        sum as u64
    } else {
        (sum as u64) & ((1u64 << it.bits) - 1)
    }
}

/// Unsigned integer addition with saturating overflow.
/// Computes in u128, clamps to type range.
pub fn unsigned_add_saturating(a: u64, b: u64, it: &IntType) -> u64 {
    let va = it.decode_unsigned(a);
    let vb = it.decode_unsigned(b);
    it.encode_unsigned(va + vb)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Float addition --

    #[test]
    fn float_add_f64() {
        let ft = FloatType::F64;
        let a = ft.encode_f64(1.5);
        let b = ft.encode_f64(2.5);
        assert_eq!(ft.decode_f64(float_add(a, b, &ft)), 4.0);
    }

    #[test]
    fn float_add_f32() {
        let ft = FloatType::F32;
        let a = ft.encode_f64(1.0);
        let b = ft.encode_f64(2.0);
        assert!((ft.decode_f64(float_add(a, b, &ft)) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn float_add_bf16() {
        let ft = FloatType::BF16;
        let a = ft.encode_f64(1.0);
        let b = ft.encode_f64(0.5);
        assert!((ft.decode_f64(float_add(a, b, &ft)) - 1.5).abs() < 0.01);
    }

    #[test]
    fn float_add_inf() {
        let ft = FloatType::F32;
        let inf = ft.encode_f64(f64::INFINITY);
        let one = ft.encode_f64(1.0);
        assert!(ft.decode_f64(float_add(inf, one, &ft)).is_infinite());

        let neg_inf = ft.encode_f64(f64::NEG_INFINITY);
        assert!(ft.decode_f64(float_add(inf, neg_inf, &ft)).is_nan());
    }

    #[test]
    fn float_add_f4e2m1() {
        let ft = FloatType::F4E2M1;
        // 1.0 + 0.5 = 1.5
        let a = ft.encode_f64(1.0);
        let b = ft.encode_f64(0.5);
        assert_eq!(ft.decode_f64(float_add(a, b, &ft)), 1.5);

        // 4.0 + 3.0 = 7.0 → saturates to 6.0 (max F4E2M1)
        let a = ft.encode_f64(4.0);
        let b = ft.encode_f64(3.0);
        assert_eq!(ft.decode_f64(float_add(a, b, &ft)), 6.0);
    }

    // -- Signed integer addition (wrapping) --

    #[test]
    fn signed_add_wrapping_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_signed(10);
        let b = it.encode_signed(20);
        assert_eq!(it.decode_signed(signed_add_wrapping(a, b, &it)), 30);

        let a = it.encode_signed(-5);
        let b = it.encode_signed(3);
        assert_eq!(it.decode_signed(signed_add_wrapping(a, b, &it)), -2);
    }

    #[test]
    fn signed_add_wrapping_overflow() {
        let it = IntType::BITS_8;
        // 100 + 100 = 200 → wraps in i8 to -56
        let a = it.encode_signed(100);
        let b = it.encode_signed(100);
        let result = it.decode_signed(signed_add_wrapping(a, b, &it));
        assert_eq!(result, (200i16 as i8) as i128); // -56
    }

    #[test]
    fn signed_add_wrapping_i64_large() {
        let it = IntType::BITS_64;
        // Large values that would lose precision through f64
        let a = it.encode_signed(i64::MAX as i128 - 1);
        let b = it.encode_signed(1);
        assert_eq!(it.decode_signed(signed_add_wrapping(a, b, &it)), i64::MAX as i128);
    }

    // -- Signed integer addition (saturating) --

    #[test]
    fn signed_add_saturating_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_signed(10);
        let b = it.encode_signed(20);
        assert_eq!(it.decode_signed(signed_add_saturating(a, b, &it)), 30);
    }

    #[test]
    fn signed_add_saturating_overflow() {
        let it = IntType::BITS_8;
        // 100 + 100 = 200 → saturates to 127 (i8::MAX)
        let a = it.encode_signed(100);
        let b = it.encode_signed(100);
        assert_eq!(it.decode_signed(signed_add_saturating(a, b, &it)), 127);
    }

    // -- Unsigned integer addition (wrapping) --

    #[test]
    fn unsigned_add_wrapping_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_unsigned(100);
        let b = it.encode_unsigned(200);
        assert_eq!(it.decode_unsigned(unsigned_add_wrapping(a, b, &it)), 300);
    }

    #[test]
    fn unsigned_add_wrapping_overflow() {
        let it = IntType::BITS_8;
        // 200 + 200 = 400 → wraps in u8 to 144
        let a = it.encode_unsigned(200);
        let b = it.encode_unsigned(200);
        let result = it.decode_unsigned(unsigned_add_wrapping(a, b, &it));
        assert_eq!(result, (400u16 as u8) as u128); // 144
    }

    // -- Unsigned integer addition (saturating) --

    #[test]
    fn unsigned_add_saturating_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_unsigned(100);
        let b = it.encode_unsigned(200);
        assert_eq!(it.decode_unsigned(unsigned_add_saturating(a, b, &it)), 300);
    }

    #[test]
    fn unsigned_add_saturating_overflow() {
        let it = IntType::BITS_8;
        // 200 + 200 = 400 → saturates to 255
        let a = it.encode_unsigned(200);
        let b = it.encode_unsigned(200);
        assert_eq!(it.decode_unsigned(unsigned_add_saturating(a, b, &it)), 255);
    }
}
