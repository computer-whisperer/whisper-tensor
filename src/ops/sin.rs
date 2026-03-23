//! Unary sine.

use crate::numeric_dtype::NumericDType;
use super::OpError;

/// Compute the sine of a scalar value.
///
/// - Float: `sin(x)` computed in f64, result encoded back to the source dtype.
/// - Integer / Bool: unsupported (caller should cast to float first).
pub fn sin(raw: u64, dtype: NumericDType) -> Result<u64, OpError> {
    match dtype {
        NumericDType::Float(ft) => {
            let v = ft.decode_f64(raw);
            Ok(ft.encode_f64(v.sin()))
        }
        NumericDType::SignedInt(_)
        | NumericDType::UnsignedInt(_)
        | NumericDType::Bool => {
            Err(OpError::UnsupportedDType { op: "sin", dtype })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric_dtype::FloatType;

    #[test]
    fn sin_f64() {
        let ft = FloatType::F64;
        let dt = NumericDType::Float(ft);

        // sin(0) = 0
        let zero = ft.encode_f64(0.0);
        assert_eq!(ft.decode_f64(sin(zero, dt).unwrap()), 0.0);

        // sin(π/2) = 1
        let half_pi = ft.encode_f64(std::f64::consts::FRAC_PI_2);
        let result = ft.decode_f64(sin(half_pi, dt).unwrap());
        assert!((result - 1.0).abs() < 1e-15);

        // sin(π) ≈ 0
        let pi = ft.encode_f64(std::f64::consts::PI);
        let result = ft.decode_f64(sin(pi, dt).unwrap());
        assert!(result.abs() < 1e-15);

        // sin(-π/2) = -1
        let neg_half_pi = ft.encode_f64(-std::f64::consts::FRAC_PI_2);
        let result = ft.decode_f64(sin(neg_half_pi, dt).unwrap());
        assert!((result - (-1.0)).abs() < 1e-15);
    }

    #[test]
    fn sin_f32() {
        let ft = FloatType::F32;
        let dt = NumericDType::Float(ft);

        let half_pi = ft.encode_f64(std::f64::consts::FRAC_PI_2);
        let result = ft.decode_f64(sin(half_pi, dt).unwrap());
        // F32 precision: sin is computed in f64 then encoded to f32
        assert!((result - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sin_bf16() {
        let ft = FloatType::BF16;
        let dt = NumericDType::Float(ft);

        let half_pi = ft.encode_f64(std::f64::consts::FRAC_PI_2);
        let result = ft.decode_f64(sin(half_pi, dt).unwrap());
        assert!((result - 1.0).abs() < 0.01);
    }

    #[test]
    fn sin_f64_special_values() {
        let ft = FloatType::F64;
        let dt = NumericDType::Float(ft);

        // sin(NaN) = NaN
        let nan = ft.encode_f64(f64::NAN);
        assert!(ft.decode_f64(sin(nan, dt).unwrap()).is_nan());

        // sin(Inf) = NaN
        let inf = ft.encode_f64(f64::INFINITY);
        assert!(ft.decode_f64(sin(inf, dt).unwrap()).is_nan());
    }

    #[test]
    fn sin_f4e2m1() {
        let ft = FloatType::F4E2M1;
        let dt = NumericDType::Float(ft);

        // sin(0) = 0
        let zero = ft.encode_f64(0.0);
        assert_eq!(ft.decode_f64(sin(zero, dt).unwrap()), 0.0);

        // sin(1.0) ≈ 0.841 → encodes to nearest F4E2M1 value (1.0)
        let one = ft.encode_f64(1.0);
        let result = ft.decode_f64(sin(one, dt).unwrap());
        // F4E2M1 can only represent 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
        // sin(1.0) ≈ 0.841 → nearest is 1.0
        assert_eq!(result, 1.0);
    }

    // -- Error cases --

    #[test]
    fn sin_int_errors() {
        use crate::numeric_dtype::IntType;
        assert!(sin(0, NumericDType::SignedInt(IntType::BITS_32)).is_err());
        assert!(sin(0, NumericDType::UnsignedInt(IntType::BITS_32)).is_err());
    }

    #[test]
    fn sin_bool_errors() {
        assert!(sin(0, NumericDType::Bool).is_err());
    }
}
