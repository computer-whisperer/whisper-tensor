//! Binary maximum.

use crate::numeric_dtype::{FloatType, IntType};

/// IEEE 754-2008 maxNum: NaN-skipping maximum. If exactly one operand is
/// NaN, returns the non-NaN operand. If both are NaN, returns NaN. This
/// matches Rust's `f32::max` / `f64::max` and the dtype contract §5.3.
pub fn float_max(a: u64, b: u64, ft: &FloatType) -> u64 {
    if let Some(r) = super::fast::binary_f32(a, b, ft, f32::max) {
        return r;
    }
    let va = ft.decode_f64(a);
    let vb = ft.decode_f64(b);
    ft.encode_f64(va.max(vb))
}

/// Signed integer maximum.
pub fn signed_max(a: u64, b: u64, it: &IntType) -> u64 {
    let va = it.decode_signed(a);
    let vb = it.decode_signed(b);
    if va >= vb { a } else { b }
}

/// Unsigned integer maximum.
pub fn unsigned_max(a: u64, b: u64, it: &IntType) -> u64 {
    let va = it.decode_unsigned(a);
    let vb = it.decode_unsigned(b);
    if va >= vb { a } else { b }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Float max --

    #[test]
    fn float_max_f64() {
        let ft = FloatType::F64;
        let a = ft.encode_f64(3.0);
        let b = ft.encode_f64(5.0);
        assert_eq!(ft.decode_f64(float_max(a, b, &ft)), 5.0);
    }

    #[test]
    fn float_max_negative() {
        let ft = FloatType::F64;
        let a = ft.encode_f64(-1.0);
        let b = ft.encode_f64(1.0);
        assert_eq!(ft.decode_f64(float_max(a, b, &ft)), 1.0);
    }

    #[test]
    fn float_max_nan() {
        let ft = FloatType::F64;
        let nan = ft.encode_f64(f64::NAN);
        let one = ft.encode_f64(1.0);
        // f64::max follows IEEE 754-2008 maxNum: max(NaN, x) = x
        // Symmetric for both argument positions.
        assert_eq!(ft.decode_f64(float_max(nan, one, &ft)), 1.0);
        assert_eq!(ft.decode_f64(float_max(one, nan, &ft)), 1.0);
    }

    #[test]
    fn float_max_both_nan() {
        // max(NaN, NaN) = NaN.
        let ft = FloatType::F64;
        let nan = ft.encode_f64(f64::NAN);
        assert!(ft.decode_f64(float_max(nan, nan, &ft)).is_nan());
    }

    #[test]
    fn float_max_nan_bf16() {
        // Verify the fast-path covers BF16 too.
        let ft = FloatType::BF16;
        let nan = ft.encode_f64(f64::NAN);
        let one = ft.encode_f64(1.0);
        assert_eq!(ft.decode_f64(float_max(nan, one, &ft)), 1.0);
        assert_eq!(ft.decode_f64(float_max(one, nan, &ft)), 1.0);
    }

    #[test]
    fn float_max_inf() {
        let ft = FloatType::F64;
        let inf = ft.encode_f64(f64::INFINITY);
        let one = ft.encode_f64(1.0);
        assert_eq!(ft.decode_f64(float_max(inf, one, &ft)), f64::INFINITY);

        let neg_inf = ft.encode_f64(f64::NEG_INFINITY);
        assert_eq!(ft.decode_f64(float_max(neg_inf, one, &ft)), 1.0);
    }

    #[test]
    fn float_max_equal() {
        let ft = FloatType::F64;
        let a = ft.encode_f64(42.0);
        let b = ft.encode_f64(42.0);
        assert_eq!(ft.decode_f64(float_max(a, b, &ft)), 42.0);
    }

    // -- Signed integer max --

    #[test]
    fn signed_max_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_signed(10);
        let b = it.encode_signed(-5);
        assert_eq!(it.decode_signed(signed_max(a, b, &it)), 10);
    }

    #[test]
    fn signed_max_equal() {
        let it = IntType::BITS_32;
        let a = it.encode_signed(7);
        let b = it.encode_signed(7);
        assert_eq!(it.decode_signed(signed_max(a, b, &it)), 7);
    }

    // -- Unsigned integer max --

    #[test]
    fn unsigned_max_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_unsigned(100);
        let b = it.encode_unsigned(200);
        assert_eq!(it.decode_unsigned(unsigned_max(a, b, &it)), 200);
    }

    #[test]
    fn unsigned_max_zero() {
        let it = IntType::BITS_32;
        let a = it.encode_unsigned(42);
        let b = it.encode_unsigned(0);
        assert_eq!(it.decode_unsigned(unsigned_max(a, b, &it)), 42);
    }
}
