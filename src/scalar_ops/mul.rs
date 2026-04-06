//! Binary multiplication.

use crate::numeric_dtype::{FloatType, IntType};

/// IEEE float multiplication. Result encoded back to the same FloatType.
pub fn float_mul(a: u64, b: u64, ft: &FloatType) -> u64 {
    if let Some(r) = super::fast::binary_f32(a, b, ft, std::ops::Mul::mul) {
        return r;
    }
    ft.encode_f64(ft.decode_f64(a) * ft.decode_f64(b))
}

/// Signed integer multiplication with wrapping overflow.
/// Computes in i128, truncates to type width.
pub fn signed_mul_wrapping(a: u64, b: u64, it: &IntType) -> u64 {
    let va = it.decode_signed(a);
    let vb = it.decode_signed(b);
    let product = va.wrapping_mul(vb);
    let mask = if it.bits >= 64 {
        u64::MAX
    } else {
        (1u64 << it.bits) - 1
    };
    (product as i64 as u64) & mask
}

/// Signed integer multiplication with saturating overflow.
/// Computes in i128, clamps to type range.
pub fn signed_mul_saturating(a: u64, b: u64, it: &IntType) -> u64 {
    let va = it.decode_signed(a);
    let vb = it.decode_signed(b);
    it.encode_signed(va * vb)
}

/// Unsigned integer multiplication with wrapping overflow.
/// Computes in u128, truncates to type width.
pub fn unsigned_mul_wrapping(a: u64, b: u64, it: &IntType) -> u64 {
    let va = it.decode_unsigned(a);
    let vb = it.decode_unsigned(b);
    let product = va.wrapping_mul(vb);
    if it.bits >= 64 {
        product as u64
    } else {
        (product as u64) & ((1u64 << it.bits) - 1)
    }
}

/// Unsigned integer multiplication with saturating overflow.
/// Computes in u128, clamps to type range.
pub fn unsigned_mul_saturating(a: u64, b: u64, it: &IntType) -> u64 {
    let va = it.decode_unsigned(a);
    let vb = it.decode_unsigned(b);
    it.encode_unsigned(va * vb)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Float multiplication --

    #[test]
    fn float_mul_f64() {
        let ft = FloatType::F64;
        let a = ft.encode_f64(3.0);
        let b = ft.encode_f64(4.0);
        assert_eq!(ft.decode_f64(float_mul(a, b, &ft)), 12.0);
    }

    #[test]
    fn float_mul_f32() {
        let ft = FloatType::F32;
        let a = ft.encode_f64(2.5);
        let b = ft.encode_f64(4.0);
        assert!((ft.decode_f64(float_mul(a, b, &ft)) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn float_mul_zero_times_inf() {
        let ft = FloatType::F32;
        let zero = ft.encode_f64(0.0);
        let inf = ft.encode_f64(f64::INFINITY);
        assert!(ft.decode_f64(float_mul(zero, inf, &ft)).is_nan());
    }

    #[test]
    fn float_mul_bf16() {
        let ft = FloatType::BF16;
        let a = ft.encode_f64(2.0);
        let b = ft.encode_f64(3.0);
        assert!((ft.decode_f64(float_mul(a, b, &ft)) - 6.0).abs() < 0.1);
    }

    // -- Signed integer multiplication --

    #[test]
    fn signed_mul_wrapping_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_signed(7);
        let b = it.encode_signed(6);
        assert_eq!(it.decode_signed(signed_mul_wrapping(a, b, &it)), 42);

        let a = it.encode_signed(-3);
        let b = it.encode_signed(4);
        assert_eq!(it.decode_signed(signed_mul_wrapping(a, b, &it)), -12);
    }

    #[test]
    fn signed_mul_wrapping_overflow() {
        let it = IntType::BITS_8;
        // 16 * 16 = 256 → wraps in i8 to 0
        let a = it.encode_signed(16);
        let b = it.encode_signed(16);
        let result = it.decode_signed(signed_mul_wrapping(a, b, &it));
        assert_eq!(result, (256i16 as i8) as i128); // 0
    }

    #[test]
    fn signed_mul_saturating_overflow() {
        let it = IntType::BITS_32;
        let a = it.encode_signed(i32::MAX as i128);
        let b = it.encode_signed(2);
        assert_eq!(
            it.decode_signed(signed_mul_saturating(a, b, &it)),
            i32::MAX as i128
        );
    }

    #[test]
    fn signed_mul_i64_large() {
        let it = IntType::BITS_64;
        let a = it.encode_signed(1_000_000_007);
        let b = it.encode_signed(2);
        assert_eq!(
            it.decode_signed(signed_mul_wrapping(a, b, &it)),
            2_000_000_014
        );
    }

    // -- Unsigned integer multiplication --

    #[test]
    fn unsigned_mul_wrapping_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_unsigned(100);
        let b = it.encode_unsigned(200);
        assert_eq!(it.decode_unsigned(unsigned_mul_wrapping(a, b, &it)), 20000);
    }

    #[test]
    fn unsigned_mul_wrapping_overflow() {
        let it = IntType::BITS_8;
        // 20 * 20 = 400 → wraps in u8 to 144
        let a = it.encode_unsigned(20);
        let b = it.encode_unsigned(20);
        assert_eq!(
            it.decode_unsigned(unsigned_mul_wrapping(a, b, &it)),
            (400u16 as u8) as u128
        );
    }

    #[test]
    fn unsigned_mul_saturating_overflow() {
        let it = IntType::BITS_8;
        let a = it.encode_unsigned(200);
        let b = it.encode_unsigned(200);
        assert_eq!(it.decode_unsigned(unsigned_mul_saturating(a, b, &it)), 255);
    }
}
