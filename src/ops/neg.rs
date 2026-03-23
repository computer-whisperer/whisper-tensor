//! Unary negation.

use crate::numeric_dtype::NumericDType;
use super::OpError;

/// Negate a scalar value.
///
/// - Float: IEEE negation (flips sign bit, preserves NaN/Inf).
/// - SignedInt: arithmetic negation in i128, saturating to type range (e.g. -MIN = MAX).
/// - UnsignedInt / Bool: unsupported.
pub fn neg(raw: u64, dtype: NumericDType) -> Result<u64, OpError> {
    match dtype {
        NumericDType::Float(ft) => {
            // IEEE negation: flip the sign bit. Correct for all values
            // including NaN, Inf, and zero (preserves payload, flips sign).
            let sign_pos = ft.total_bits() as u64 - 1;
            Ok(raw ^ (1u64 << sign_pos))
        }
        NumericDType::SignedInt(it) => {
            let v = it.decode_signed(raw);
            // Wrapping negation: -MIN overflows back to MIN for 2's complement
            Ok(it.encode_signed(v.wrapping_neg()))
        }
        NumericDType::UnsignedInt(_) | NumericDType::Bool => {
            Err(OpError::UnsupportedDType { op: "neg", dtype })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric_dtype::{FloatType, IntType};

    // -- Float negation --

    #[test]
    fn neg_f64() {
        let ft = FloatType::F64;
        let dt = NumericDType::Float(ft);

        // 1.0 → -1.0
        let one = ft.encode_f64(1.0);
        let neg_one = neg(one, dt).unwrap();
        assert_eq!(ft.decode_f64(neg_one), -1.0);

        // -3.14 → 3.14
        let v = ft.encode_f64(-3.14);
        assert_eq!(ft.decode_f64(neg(v, dt).unwrap()), 3.14);

        // 0.0 → -0.0
        let zero = ft.encode_f64(0.0);
        let neg_zero = neg(zero, dt).unwrap();
        let result = ft.decode_f64(neg_zero);
        assert!(result == 0.0 && result.is_sign_negative());

        // -0.0 → 0.0
        let nz = ft.encode_f64(-0.0);
        let pz = neg(nz, dt).unwrap();
        let result = ft.decode_f64(pz);
        assert!(result == 0.0 && result.is_sign_positive());

        // NaN stays NaN (sign flips but value is still NaN)
        let nan = ft.encode_f64(f64::NAN);
        assert!(ft.decode_f64(neg(nan, dt).unwrap()).is_nan());

        // Inf → -Inf
        let inf = ft.encode_f64(f64::INFINITY);
        assert_eq!(ft.decode_f64(neg(inf, dt).unwrap()), f64::NEG_INFINITY);
    }

    #[test]
    fn neg_bf16() {
        let ft = FloatType::BF16;
        let dt = NumericDType::Float(ft);

        let v = ft.encode_f64(3.0);
        let result = ft.decode_f64(neg(v, dt).unwrap());
        assert!((result - (-3.0)).abs() < 0.1);
    }

    #[test]
    fn neg_f4e2m1() {
        let ft = FloatType::F4E2M1;
        let dt = NumericDType::Float(ft);

        // 6.0 (0b0111) → -6.0 (0b1111)
        let v = ft.encode_f64(6.0);
        assert_eq!(v, 0b0111);
        let nv = neg(v, dt).unwrap();
        assert_eq!(nv, 0b1111);
        assert_eq!(ft.decode_f64(nv), -6.0);
    }

    // -- Signed int negation --

    #[test]
    fn neg_i32() {
        let it = IntType::BITS_32;
        let dt = NumericDType::SignedInt(it);

        let v = it.encode_signed(42);
        assert_eq!(it.decode_signed(neg(v, dt).unwrap()), -42);

        let v = it.encode_signed(-1);
        assert_eq!(it.decode_signed(neg(v, dt).unwrap()), 1);

        // 0 → 0
        let v = it.encode_signed(0);
        assert_eq!(it.decode_signed(neg(v, dt).unwrap()), 0);
    }

    #[test]
    fn neg_i64_min_saturates() {
        let it = IntType::BITS_64;
        let dt = NumericDType::SignedInt(it);

        // -(i64::MIN) = 2^63, which overflows i64. i128 negation gives
        // the correct mathematical result, then encode_signed clamps to i64::MAX.
        let v = it.encode_signed(i64::MIN as i128);
        let result = it.decode_signed(neg(v, dt).unwrap());
        assert_eq!(result, i64::MAX as i128);
    }

    // -- Unsupported dtypes --

    #[test]
    fn neg_unsigned_errors() {
        let dt = NumericDType::UnsignedInt(IntType::BITS_32);
        assert!(neg(0, dt).is_err());
    }

    #[test]
    fn neg_bool_errors() {
        assert!(neg(0, NumericDType::Bool).is_err());
    }
}
