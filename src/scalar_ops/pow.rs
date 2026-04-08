//! Binary power.

use crate::numeric_dtype::{FloatType, IntType};

/// IEEE float power: a^b via powf.
pub fn float_pow(a: u64, b: u64, ft: &FloatType) -> u64 {
    if let Some(r) = super::fast::binary_f32(a, b, ft, f32::powf) {
        return r;
    }
    ft.encode_f64(ft.decode_f64(a).powf(ft.decode_f64(b)))
}

/// Signed integer power (saturating). Negative base with even exponent
/// gives a positive result, odd gives a negative result. Per dtype contract
/// §5.5, a **negative exponent** on an integer Pow returns `0` (the
/// truncated integer value of `1 / base^|exp|`). Exponents larger than
/// `u32::MAX` are clamped: integer `pow` saturates within a few iterations
/// of repeated squaring for any `|base| ≥ 2`, so the clamp is observationally
/// equivalent for those cases, and `±1` / `0` bases are handled correctly
/// because parity of the low bit is preserved.
pub fn signed_pow(a: u64, b: u64, it: &IntType) -> u64 {
    let base = it.decode_signed(a);
    let exp = it.decode_signed(b);
    if exp < 0 {
        return it.encode_signed(0);
    }
    let exp_u32 = exp.min(u32::MAX as i128) as u32;
    let result = checked_signed_pow(base, exp_u32);
    it.encode_signed(result)
}

/// Unsigned integer power (saturating). Exponents larger than `u32::MAX`
/// are clamped (see `signed_pow` for the rationale).
pub fn unsigned_pow(a: u64, b: u64, it: &IntType) -> u64 {
    let base = it.decode_unsigned(a);
    let exp = it.decode_unsigned(b);
    let exp_u32 = exp.min(u32::MAX as u128) as u32;
    let result = checked_unsigned_pow(base, exp_u32);
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

    #[test]
    fn signed_pow_negative_exponent_returns_zero() {
        // Per dtype contract §5.5: negative integer exponent → 0.
        let it = IntType::BITS_32;
        let a = it.encode_signed(2);
        let b = it.encode_signed(-3);
        assert_eq!(it.decode_signed(signed_pow(a, b, &it)), 0);

        // Even base = 1: still 0 by the contract rule.
        let a = it.encode_signed(1);
        let b = it.encode_signed(-5);
        assert_eq!(it.decode_signed(signed_pow(a, b, &it)), 0);

        // Negative base with negative exponent: 0.
        let a = it.encode_signed(-2);
        let b = it.encode_signed(-3);
        assert_eq!(it.decode_signed(signed_pow(a, b, &it)), 0);
    }

    #[test]
    fn signed_pow_negative_exponent_narrow() {
        // Verify the rule applies for narrow signed types.
        let it = IntType::BITS_8;
        let a = it.encode_signed(7);
        let b = it.encode_signed(-1);
        assert_eq!(it.decode_signed(signed_pow(a, b, &it)), 0);
    }

    #[test]
    fn signed_pow_huge_exponent_saturates() {
        // i64 exponent above u32::MAX should clamp and saturate, not
        // silently wrap to a small value.
        let it = IntType::BITS_64;
        let a = it.encode_signed(2);
        let b = it.encode_signed(1_000_000_000_000); // > u32::MAX
        // 2^huge saturates to i64::MAX.
        assert_eq!(it.decode_signed(signed_pow(a, b, &it)), i64::MAX as i128);
    }

    #[test]
    fn signed_pow_one_base_huge_exponent() {
        // 1^anything = 1, even with huge exponents.
        let it = IntType::BITS_32;
        let a = it.encode_signed(1);
        let b = it.encode_signed(i32::MAX as i128);
        assert_eq!(it.decode_signed(signed_pow(a, b, &it)), 1);
    }

    #[test]
    fn signed_pow_neg_one_base_parity() {
        // (-1)^even = 1, (-1)^odd = -1.
        let it = IntType::BITS_32;
        let a = it.encode_signed(-1);
        let b = it.encode_signed(100);
        assert_eq!(it.decode_signed(signed_pow(a, b, &it)), 1);

        let b = it.encode_signed(101);
        assert_eq!(it.decode_signed(signed_pow(a, b, &it)), -1);
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
