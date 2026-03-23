//! Unary sine.

use crate::numeric_dtype::FloatType;

/// Compute the sine of a float value.
/// Decoded to f64, computed, encoded back to the source FloatType.
pub fn float_sin(raw: u64, ft: &FloatType) -> u64 {
    ft.encode_f64(ft.decode_f64(raw).sin())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_sin_f64() {
        let ft = FloatType::F64;

        assert_eq!(ft.decode_f64(float_sin(ft.encode_f64(0.0), &ft)), 0.0);

        let result = ft.decode_f64(float_sin(ft.encode_f64(std::f64::consts::FRAC_PI_2), &ft));
        assert!((result - 1.0).abs() < 1e-15);

        let result = ft.decode_f64(float_sin(ft.encode_f64(std::f64::consts::PI), &ft));
        assert!(result.abs() < 1e-15);

        let result = ft.decode_f64(float_sin(ft.encode_f64(-std::f64::consts::FRAC_PI_2), &ft));
        assert!((result - (-1.0)).abs() < 1e-15);
    }

    #[test]
    fn float_sin_f32() {
        let ft = FloatType::F32;
        let result = ft.decode_f64(float_sin(ft.encode_f64(std::f64::consts::FRAC_PI_2), &ft));
        assert!((result - 1.0).abs() < 1e-6);
    }

    #[test]
    fn float_sin_bf16() {
        let ft = FloatType::BF16;
        let result = ft.decode_f64(float_sin(ft.encode_f64(std::f64::consts::FRAC_PI_2), &ft));
        assert!((result - 1.0).abs() < 0.01);
    }

    #[test]
    fn float_sin_special_values() {
        let ft = FloatType::F64;
        assert!(ft.decode_f64(float_sin(ft.encode_f64(f64::NAN), &ft)).is_nan());
        assert!(ft.decode_f64(float_sin(ft.encode_f64(f64::INFINITY), &ft)).is_nan());
    }

    #[test]
    fn float_sin_f4e2m1() {
        let ft = FloatType::F4E2M1;
        assert_eq!(ft.decode_f64(float_sin(ft.encode_f64(0.0), &ft)), 0.0);
        // sin(1.0) ≈ 0.841 → nearest F4E2M1 value is 1.0
        assert_eq!(ft.decode_f64(float_sin(ft.encode_f64(1.0), &ft)), 1.0);
    }
}
