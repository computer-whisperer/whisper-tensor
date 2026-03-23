//! Unary sign function.

use crate::numeric_dtype::{FloatType, IntType};

/// Float sign: returns -1.0, 0.0, or 1.0 (NaN returns NaN).
pub fn float_sign(raw: u64, ft: &FloatType) -> u64 {
    let v = ft.decode_f64(raw);
    let s = if v.is_nan() {
        f64::NAN
    } else if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    };
    ft.encode_f64(s)
}

/// Signed integer sign: returns -1, 0, or 1.
pub fn signed_sign(raw: u64, it: &IntType) -> u64 {
    let v = it.decode_signed(raw);
    let s = if v > 0 { 1 } else if v < 0 { -1 } else { 0 };
    it.encode_signed(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Float sign --

    #[test]
    fn float_sign_f64() {
        let ft = FloatType::F64;

        assert_eq!(ft.decode_f64(float_sign(ft.encode_f64(42.0), &ft)), 1.0);
        assert_eq!(ft.decode_f64(float_sign(ft.encode_f64(-42.0), &ft)), -1.0);
        assert_eq!(ft.decode_f64(float_sign(ft.encode_f64(0.0), &ft)), 0.0);
    }

    #[test]
    fn float_sign_special_values() {
        let ft = FloatType::F64;
        assert!(ft.decode_f64(float_sign(ft.encode_f64(f64::NAN), &ft)).is_nan());
        assert_eq!(ft.decode_f64(float_sign(ft.encode_f64(f64::INFINITY), &ft)), 1.0);
        assert_eq!(ft.decode_f64(float_sign(ft.encode_f64(f64::NEG_INFINITY), &ft)), -1.0);
    }

    #[test]
    fn float_sign_bf16() {
        let ft = FloatType::BF16;
        assert_eq!(ft.decode_f64(float_sign(ft.encode_f64(3.0), &ft)), 1.0);
        assert_eq!(ft.decode_f64(float_sign(ft.encode_f64(-3.0), &ft)), -1.0);
    }

    // -- Signed integer sign --

    #[test]
    fn signed_sign_basic() {
        let it = IntType::BITS_32;
        assert_eq!(it.decode_signed(signed_sign(it.encode_signed(42), &it)), 1);
        assert_eq!(it.decode_signed(signed_sign(it.encode_signed(-42), &it)), -1);
        assert_eq!(it.decode_signed(signed_sign(it.encode_signed(0), &it)), 0);
    }

    #[test]
    fn signed_sign_extremes() {
        let it = IntType::BITS_8;
        assert_eq!(it.decode_signed(signed_sign(it.encode_signed(127), &it)), 1);
        assert_eq!(it.decode_signed(signed_sign(it.encode_signed(-128), &it)), -1);
    }
}
