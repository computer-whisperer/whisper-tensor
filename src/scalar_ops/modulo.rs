//! Binary modulo/remainder.

use crate::numeric_dtype::{FloatType, IntType};

/// IEEE float remainder (Rust `%` on f64).
pub fn float_mod(a: u64, b: u64, ft: &FloatType) -> u64 {
    ft.encode_f64(ft.decode_f64(a) % ft.decode_f64(b))
}

/// Signed integer remainder (truncated division). Division by zero returns 0.
/// Result sign matches the dividend (Rust semantics).
pub fn signed_mod(a: u64, b: u64, it: &IntType) -> u64 {
    let va = it.decode_signed(a);
    let vb = it.decode_signed(b);
    if vb == 0 {
        return 0;
    }
    it.encode_signed(va % vb)
}

/// Signed integer mathematical modulo (result sign matches divisor).
/// ONNX `Mod` with `fmod=0` on integer types.
/// Division by zero returns 0.
pub fn signed_imod(a: u64, b: u64, it: &IntType) -> u64 {
    let va = it.decode_signed(a);
    let vb = it.decode_signed(b);
    if vb == 0 {
        return 0;
    }
    let rem = va % vb;
    let result = if rem != 0 && (rem ^ vb) < 0 {
        rem + vb
    } else {
        rem
    };
    it.encode_signed(result)
}

/// Unsigned integer remainder. Division by zero returns 0.
pub fn unsigned_mod(a: u64, b: u64, it: &IntType) -> u64 {
    let va = it.decode_unsigned(a);
    let vb = it.decode_unsigned(b);
    if vb == 0 {
        return 0;
    }
    it.encode_unsigned(va % vb)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Float mod --

    #[test]
    fn float_mod_f64() {
        let ft = FloatType::F64;
        let a = ft.encode_f64(7.5);
        let b = ft.encode_f64(2.0);
        assert_eq!(ft.decode_f64(float_mod(a, b, &ft)), 1.5);
    }

    #[test]
    fn float_mod_negative() {
        let ft = FloatType::F64;
        let a = ft.encode_f64(-7.5);
        let b = ft.encode_f64(2.0);
        assert_eq!(ft.decode_f64(float_mod(a, b, &ft)), -1.5);
    }

    #[test]
    fn float_mod_by_zero() {
        let ft = FloatType::F64;
        let a = ft.encode_f64(1.0);
        let b = ft.encode_f64(0.0);
        assert!(ft.decode_f64(float_mod(a, b, &ft)).is_nan());
    }

    #[test]
    fn float_mod_nan() {
        let ft = FloatType::F64;
        let nan = ft.encode_f64(f64::NAN);
        let one = ft.encode_f64(1.0);
        assert!(ft.decode_f64(float_mod(nan, one, &ft)).is_nan());
    }

    // -- Signed integer mod --

    #[test]
    fn signed_mod_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_signed(17);
        let b = it.encode_signed(5);
        assert_eq!(it.decode_signed(signed_mod(a, b, &it)), 2);
    }

    #[test]
    fn signed_mod_negative_dividend() {
        let it = IntType::BITS_32;
        let a = it.encode_signed(-17);
        let b = it.encode_signed(5);
        assert_eq!(it.decode_signed(signed_mod(a, b, &it)), -2);
    }

    #[test]
    fn signed_mod_by_zero() {
        let it = IntType::BITS_32;
        let a = it.encode_signed(42);
        let b = it.encode_signed(0);
        assert_eq!(signed_mod(a, b, &it), 0);
    }

    // -- Unsigned integer mod --

    #[test]
    fn unsigned_mod_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_unsigned(17);
        let b = it.encode_unsigned(5);
        assert_eq!(it.decode_unsigned(unsigned_mod(a, b, &it)), 2);
    }

    #[test]
    fn unsigned_mod_by_zero() {
        let it = IntType::BITS_32;
        let a = it.encode_unsigned(42);
        let b = it.encode_unsigned(0);
        assert_eq!(unsigned_mod(a, b, &it), 0);
    }

    #[test]
    fn unsigned_mod_exact() {
        let it = IntType::BITS_8;
        let a = it.encode_unsigned(200);
        let b = it.encode_unsigned(50);
        assert_eq!(it.decode_unsigned(unsigned_mod(a, b, &it)), 0);
    }
}
