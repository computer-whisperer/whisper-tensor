//! Binary multiplication.

use crate::numeric_dtype::NumericDType;
use super::{OpError, check_dtype_match};

/// Multiply two scalar values of the same dtype.
///
/// - Float: IEEE multiplication via f64 intermediate, result encoded back.
/// - SignedInt: multiplication in i128, clamped to type range.
/// - UnsignedInt: multiplication in u128, clamped to type range.
/// - Bool: unsupported.
pub fn mul(a: u64, b: u64, dtype: NumericDType) -> Result<u64, OpError> {
    check_dtype_match("mul", dtype, dtype)?;
    match dtype {
        NumericDType::Float(ft) => {
            let va = ft.decode_f64(a);
            let vb = ft.decode_f64(b);
            Ok(ft.encode_f64(va * vb))
        }
        NumericDType::SignedInt(it) => {
            let va = it.decode_signed(a);
            let vb = it.decode_signed(b);
            // i128 * i128 won't overflow for values that fit in i64
            Ok(it.encode_signed(va * vb))
        }
        NumericDType::UnsignedInt(it) => {
            let va = it.decode_unsigned(a);
            let vb = it.decode_unsigned(b);
            Ok(it.encode_unsigned(va * vb))
        }
        NumericDType::Bool => {
            Err(OpError::UnsupportedDType { op: "mul", dtype })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric_dtype::{FloatType, IntType};

    // -- Float multiplication --

    #[test]
    fn mul_f64() {
        let ft = FloatType::F64;
        let dt = NumericDType::Float(ft);

        let a = ft.encode_f64(3.0);
        let b = ft.encode_f64(4.0);
        assert_eq!(ft.decode_f64(mul(a, b, dt).unwrap()), 12.0);
    }

    #[test]
    fn mul_f32() {
        let ft = FloatType::F32;
        let dt = NumericDType::Float(ft);

        let a = ft.encode_f64(2.5);
        let b = ft.encode_f64(4.0);
        let result = ft.decode_f64(mul(a, b, dt).unwrap());
        assert!((result - 10.0).abs() < 1e-6);
    }

    #[test]
    fn mul_f32_zero_times_inf() {
        let ft = FloatType::F32;
        let dt = NumericDType::Float(ft);

        // 0 * Inf = NaN (IEEE)
        let zero = ft.encode_f64(0.0);
        let inf = ft.encode_f64(f64::INFINITY);
        assert!(ft.decode_f64(mul(zero, inf, dt).unwrap()).is_nan());
    }

    #[test]
    fn mul_bf16() {
        let ft = FloatType::BF16;
        let dt = NumericDType::Float(ft);

        let a = ft.encode_f64(2.0);
        let b = ft.encode_f64(3.0);
        let result = ft.decode_f64(mul(a, b, dt).unwrap());
        assert!((result - 6.0).abs() < 0.1);
    }

    // -- Integer multiplication --

    #[test]
    fn mul_i32() {
        let it = IntType::BITS_32;
        let dt = NumericDType::SignedInt(it);

        let a = it.encode_signed(7);
        let b = it.encode_signed(6);
        assert_eq!(it.decode_signed(mul(a, b, dt).unwrap()), 42);

        let a = it.encode_signed(-3);
        let b = it.encode_signed(4);
        assert_eq!(it.decode_signed(mul(a, b, dt).unwrap()), -12);
    }

    #[test]
    fn mul_i64_large() {
        let it = IntType::BITS_64;
        let dt = NumericDType::SignedInt(it);

        // Values that would lose precision through f64
        let a = it.encode_signed(1_000_000_007);
        let b = it.encode_signed(2);
        assert_eq!(it.decode_signed(mul(a, b, dt).unwrap()), 2_000_000_014);
    }

    #[test]
    fn mul_u32() {
        let it = IntType::BITS_32;
        let dt = NumericDType::UnsignedInt(it);

        let a = it.encode_unsigned(100);
        let b = it.encode_unsigned(200);
        assert_eq!(it.decode_unsigned(mul(a, b, dt).unwrap()), 20000);
    }

    #[test]
    fn mul_i32_overflow_clamps() {
        let it = IntType::BITS_32;
        let dt = NumericDType::SignedInt(it);

        // i32::MAX * 2 → clamped to i32::MAX by encode_signed
        let a = it.encode_signed(i32::MAX as i128);
        let b = it.encode_signed(2);
        let result = it.decode_signed(mul(a, b, dt).unwrap());
        assert_eq!(result, i32::MAX as i128);
    }

    // -- Error cases --

    #[test]
    fn mul_bool_errors() {
        assert!(mul(1, 1, NumericDType::Bool).is_err());
    }
}
