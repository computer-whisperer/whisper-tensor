//! Binary minimum.

use crate::numeric_dtype::{FloatType, IntType};

/// IEEE 754-2008 minNum: NaN-skipping minimum. If exactly one operand is
/// NaN, returns the non-NaN operand. If both are NaN, returns NaN. This
/// matches Rust's `f32::min` / `f64::min` and the dtype contract §5.3.
pub fn float_min(a: u64, b: u64, ft: &FloatType) -> u64 {
    if let Some(r) = super::fast::binary_f32(a, b, ft, f32::min) {
        return r;
    }
    let va = ft.decode_f64(a);
    let vb = ft.decode_f64(b);
    ft.encode_f64(va.min(vb))
}

/// Signed integer minimum.
pub fn signed_min(a: u64, b: u64, it: &IntType) -> u64 {
    let va = it.decode_signed(a);
    let vb = it.decode_signed(b);
    if va <= vb { a } else { b }
}

/// Unsigned integer minimum.
pub fn unsigned_min(a: u64, b: u64, it: &IntType) -> u64 {
    let va = it.decode_unsigned(a);
    let vb = it.decode_unsigned(b);
    if va <= vb { a } else { b }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Float min --

    #[test]
    fn float_min_f64() {
        let ft = FloatType::F64;
        let a = ft.encode_f64(3.0);
        let b = ft.encode_f64(5.0);
        assert_eq!(ft.decode_f64(float_min(a, b, &ft)), 3.0);
    }

    #[test]
    fn float_min_negative() {
        let ft = FloatType::F64;
        let a = ft.encode_f64(-1.0);
        let b = ft.encode_f64(1.0);
        assert_eq!(ft.decode_f64(float_min(a, b, &ft)), -1.0);
    }

    #[test]
    fn float_min_nan() {
        let ft = FloatType::F64;
        let nan = ft.encode_f64(f64::NAN);
        let one = ft.encode_f64(1.0);
        // f64::min follows IEEE 754-2008 minNum: min(NaN, x) = x
        // Symmetric for both argument positions.
        assert_eq!(ft.decode_f64(float_min(nan, one, &ft)), 1.0);
        assert_eq!(ft.decode_f64(float_min(one, nan, &ft)), 1.0);
    }

    #[test]
    fn float_min_both_nan() {
        // min(NaN, NaN) = NaN.
        let ft = FloatType::F64;
        let nan = ft.encode_f64(f64::NAN);
        assert!(ft.decode_f64(float_min(nan, nan, &ft)).is_nan());
    }

    #[test]
    fn float_min_nan_bf16() {
        // Verify the fast-path covers BF16 too.
        let ft = FloatType::BF16;
        let nan = ft.encode_f64(f64::NAN);
        let one = ft.encode_f64(1.0);
        assert_eq!(ft.decode_f64(float_min(nan, one, &ft)), 1.0);
        assert_eq!(ft.decode_f64(float_min(one, nan, &ft)), 1.0);
    }

    #[test]
    fn float_min_inf() {
        let ft = FloatType::F64;
        let inf = ft.encode_f64(f64::INFINITY);
        let one = ft.encode_f64(1.0);
        assert_eq!(ft.decode_f64(float_min(inf, one, &ft)), 1.0);

        let neg_inf = ft.encode_f64(f64::NEG_INFINITY);
        assert_eq!(
            ft.decode_f64(float_min(neg_inf, one, &ft)),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn float_min_equal() {
        let ft = FloatType::F64;
        let a = ft.encode_f64(42.0);
        let b = ft.encode_f64(42.0);
        assert_eq!(ft.decode_f64(float_min(a, b, &ft)), 42.0);
    }

    // -- Signed integer min --

    #[test]
    fn signed_min_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_signed(10);
        let b = it.encode_signed(-5);
        assert_eq!(it.decode_signed(signed_min(a, b, &it)), -5);
    }

    #[test]
    fn signed_min_equal() {
        let it = IntType::BITS_32;
        let a = it.encode_signed(7);
        let b = it.encode_signed(7);
        assert_eq!(it.decode_signed(signed_min(a, b, &it)), 7);
    }

    // -- Unsigned integer min --

    #[test]
    fn unsigned_min_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_unsigned(100);
        let b = it.encode_unsigned(200);
        assert_eq!(it.decode_unsigned(unsigned_min(a, b, &it)), 100);
    }

    #[test]
    fn unsigned_min_zero() {
        let it = IntType::BITS_32;
        let a = it.encode_unsigned(42);
        let b = it.encode_unsigned(0);
        assert_eq!(it.decode_unsigned(unsigned_min(a, b, &it)), 0);
    }
}
