//! Binary power.

use crate::numeric_dtype::{FloatType, IntType};

/// IEEE float power: a^b via powf.
pub fn float_pow(a: u64, b: u64, ft: &FloatType) -> u64 {
    if let Some(r) = super::fast::binary_f32(a, b, ft, f32::powf) {
        return r;
    }
    ft.encode_f64(ft.decode_f64(a).powf(ft.decode_f64(b)))
}

/// Signed integer power (saturating). Exponent is decoded as unsigned.
/// Negative base with even exponent gives positive result, odd gives negative.
pub fn signed_pow(a: u64, b: u64, it: &IntType) -> u64 {
    let base = it.decode_signed(a);
    let exp = it.decode_unsigned(b) as u32;
    let result = checked_signed_pow(base, exp);
    it.encode_signed(result)
}

/// Unsigned integer power (saturating).
pub fn unsigned_pow(a: u64, b: u64, it: &IntType) -> u64 {
    let base = it.decode_unsigned(a);
    let exp = it.decode_unsigned(b) as u32;
    let result = checked_unsigned_pow(base, exp);
    it.encode_unsigned(result)
}

/// Compute base^exp in i128, allowing encode_signed to handle saturation.
fn checked_signed_pow(mut base: i128, mut exp: u32) -> i128 {
    if exp == 0 {
        return 1;
    }
    let mut result: i128 = 1;
    loop {
        if exp & 1 == 1 {
            result = result.saturating_mul(base);
        }
        exp >>= 1;
        if exp == 0 {
            break;
        }
        base = base.saturating_mul(base);
    }
    result
}

/// Compute base^exp in u128, allowing encode_unsigned to handle saturation.
fn checked_unsigned_pow(mut base: u128, mut exp: u32) -> u128 {
    if exp == 0 {
        return 1;
    }
    let mut result: u128 = 1;
    loop {
        if exp & 1 == 1 {
            result = result.saturating_mul(base);
        }
        exp >>= 1;
        if exp == 0 {
            break;
        }
        base = base.saturating_mul(base);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Float power --

    #[test]
    fn float_pow_f64() {
        let ft = FloatType::F64;
        let a = ft.encode_f64(2.0);
        let b = ft.encode_f64(10.0);
        assert_eq!(ft.decode_f64(float_pow(a, b, &ft)), 1024.0);
    }

    #[test]
    fn float_pow_fractional() {
        let ft = FloatType::F64;
        let a = ft.encode_f64(4.0);
        let b = ft.encode_f64(0.5);
        assert!((ft.decode_f64(float_pow(a, b, &ft)) - 2.0).abs() < 1e-15);
    }

    #[test]
    fn float_pow_zero_exp() {
        let ft = FloatType::F64;
        let a = ft.encode_f64(42.0);
        let b = ft.encode_f64(0.0);
        assert_eq!(ft.decode_f64(float_pow(a, b, &ft)), 1.0);
    }

    #[test]
    fn float_pow_nan() {
        let ft = FloatType::F64;
        let nan = ft.encode_f64(f64::NAN);
        let one = ft.encode_f64(1.0);
        assert!(ft.decode_f64(float_pow(nan, one, &ft)).is_nan());
    }

    #[test]
    fn float_pow_negative_base() {
        let ft = FloatType::F64;
        let a = ft.encode_f64(-2.0);
        let b = ft.encode_f64(3.0);
        assert_eq!(ft.decode_f64(float_pow(a, b, &ft)), -8.0);
    }

    // -- Signed integer power --

    #[test]
    fn signed_pow_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_signed(3);
        let b = it.encode_unsigned(4);
        assert_eq!(it.decode_signed(signed_pow(a, b, &it)), 81);
    }

    #[test]
    fn signed_pow_negative_base() {
        let it = IntType::BITS_32;
        let a = it.encode_signed(-2);
        let b = it.encode_unsigned(3);
        assert_eq!(it.decode_signed(signed_pow(a, b, &it)), -8);

        let b = it.encode_unsigned(4);
        assert_eq!(it.decode_signed(signed_pow(a, b, &it)), 16);
    }

    #[test]
    fn signed_pow_zero_exp() {
        let it = IntType::BITS_32;
        let a = it.encode_signed(42);
        let b = it.encode_unsigned(0);
        assert_eq!(it.decode_signed(signed_pow(a, b, &it)), 1);
    }

    #[test]
    fn signed_pow_saturates() {
        let it = IntType::BITS_8;
        // 10^3 = 1000 → saturates to 127
        let a = it.encode_signed(10);
        let b = it.encode_unsigned(3);
        assert_eq!(it.decode_signed(signed_pow(a, b, &it)), 127);
    }

    // -- Unsigned integer power --

    #[test]
    fn unsigned_pow_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_unsigned(2);
        let b = it.encode_unsigned(10);
        assert_eq!(it.decode_unsigned(unsigned_pow(a, b, &it)), 1024);
    }

    #[test]
    fn unsigned_pow_zero_exp() {
        let it = IntType::BITS_32;
        let a = it.encode_unsigned(42);
        let b = it.encode_unsigned(0);
        assert_eq!(it.decode_unsigned(unsigned_pow(a, b, &it)), 1);
    }

    #[test]
    fn unsigned_pow_saturates() {
        let it = IntType::BITS_8;
        // 10^3 = 1000 → saturates to 255
        let a = it.encode_unsigned(10);
        let b = it.encode_unsigned(3);
        assert_eq!(it.decode_unsigned(unsigned_pow(a, b, &it)), 255);
    }
}
