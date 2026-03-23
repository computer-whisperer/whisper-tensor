//! Binary addition.

use crate::numeric_dtype::NumericDType;
use super::{OpError, check_dtype_match};

/// Add two scalar values of the same dtype.
///
/// - Float: IEEE addition via f64 intermediate, result encoded back.
/// - SignedInt: wrapping addition in i128, clamped to type range.
/// - UnsignedInt: wrapping addition in u128, clamped to type range.
/// - Bool: unsupported (use bitwise OR if needed).
pub fn add(a: u64, b: u64, dtype: NumericDType) -> Result<u64, OpError> {
    check_dtype_match("add", dtype, dtype)?; // self-consistency (dtype is shared)
    match dtype {
        NumericDType::Float(ft) => {
            let va = ft.decode_f64(a);
            let vb = ft.decode_f64(b);
            Ok(ft.encode_f64(va + vb))
        }
        NumericDType::SignedInt(it) => {
            let va = it.decode_signed(a);
            let vb = it.decode_signed(b);
            // Wrapping add in full i128, then clamp to type range
            Ok(it.encode_signed(va.wrapping_add(vb)))
        }
        NumericDType::UnsignedInt(it) => {
            let va = it.decode_unsigned(a);
            let vb = it.decode_unsigned(b);
            Ok(it.encode_unsigned(va.wrapping_add(vb)))
        }
        NumericDType::Bool => {
            Err(OpError::UnsupportedDType { op: "add", dtype })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric_dtype::{FloatType, IntType};

    // -- Float addition --

    #[test]
    fn add_f64() {
        let ft = FloatType::F64;
        let dt = NumericDType::Float(ft);

        let a = ft.encode_f64(1.5);
        let b = ft.encode_f64(2.5);
        assert_eq!(ft.decode_f64(add(a, b, dt).unwrap()), 4.0);
    }

    #[test]
    fn add_f32() {
        let ft = FloatType::F32;
        let dt = NumericDType::Float(ft);

        let a = ft.encode_f64(1.0);
        let b = ft.encode_f64(2.0);
        let result = ft.decode_f64(add(a, b, dt).unwrap());
        assert!((result - 3.0).abs() < 1e-6);
    }

    #[test]
    fn add_bf16() {
        let ft = FloatType::BF16;
        let dt = NumericDType::Float(ft);

        let a = ft.encode_f64(1.0);
        let b = ft.encode_f64(0.5);
        let result = ft.decode_f64(add(a, b, dt).unwrap());
        assert!((result - 1.5).abs() < 0.01);
    }

    #[test]
    fn add_f32_inf_handling() {
        let ft = FloatType::F32;
        let dt = NumericDType::Float(ft);

        // Inf + 1 = Inf
        let inf = ft.encode_f64(f64::INFINITY);
        let one = ft.encode_f64(1.0);
        assert!(ft.decode_f64(add(inf, one, dt).unwrap()).is_infinite());

        // Inf + (-Inf) = NaN
        let neg_inf = ft.encode_f64(f64::NEG_INFINITY);
        assert!(ft.decode_f64(add(inf, neg_inf, dt).unwrap()).is_nan());
    }

    #[test]
    fn add_f4e2m1() {
        let ft = FloatType::F4E2M1;
        let dt = NumericDType::Float(ft);

        // 1.0 + 0.5 = 1.5
        let a = ft.encode_f64(1.0);
        let b = ft.encode_f64(0.5);
        let result = ft.decode_f64(add(a, b, dt).unwrap());
        assert_eq!(result, 1.5);

        // 4.0 + 3.0 = 7.0 → rounds/clamps to 6.0 (max F4E2M1 value)
        let a = ft.encode_f64(4.0);
        let b = ft.encode_f64(3.0);
        let result = ft.decode_f64(add(a, b, dt).unwrap());
        assert_eq!(result, 6.0); // saturates to max
    }

    // -- Integer addition --

    #[test]
    fn add_i32() {
        let it = IntType::BITS_32;
        let dt = NumericDType::SignedInt(it);

        let a = it.encode_signed(10);
        let b = it.encode_signed(20);
        assert_eq!(it.decode_signed(add(a, b, dt).unwrap()), 30);

        // Negative
        let a = it.encode_signed(-5);
        let b = it.encode_signed(3);
        assert_eq!(it.decode_signed(add(a, b, dt).unwrap()), -2);
    }

    #[test]
    fn add_i64() {
        let it = IntType::BITS_64;
        let dt = NumericDType::SignedInt(it);

        // Large values that would lose precision through f64
        let a = it.encode_signed(i64::MAX as i128 - 1);
        let b = it.encode_signed(1);
        assert_eq!(it.decode_signed(add(a, b, dt).unwrap()), i64::MAX as i128);
    }

    #[test]
    fn add_u32() {
        let it = IntType::BITS_32;
        let dt = NumericDType::UnsignedInt(it);

        let a = it.encode_unsigned(100);
        let b = it.encode_unsigned(200);
        assert_eq!(it.decode_unsigned(add(a, b, dt).unwrap()), 300);
    }

    #[test]
    fn add_u8_overflow_wraps() {
        let it = IntType::BITS_8;
        let dt = NumericDType::UnsignedInt(it);

        // 200 + 200 = 400 → clamped to 255 by encode_unsigned
        let a = it.encode_unsigned(200);
        let b = it.encode_unsigned(200);
        let result = it.decode_unsigned(add(a, b, dt).unwrap());
        assert_eq!(result, 255); // clamped
    }

    // -- Error cases --

    #[test]
    fn add_bool_errors() {
        assert!(add(1, 1, NumericDType::Bool).is_err());
    }
}
