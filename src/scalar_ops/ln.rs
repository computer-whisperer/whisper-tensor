//! Unary natural logarithm.

use crate::numeric_dtype::FloatType;

/// Compute the natural logarithm (ln) of a float value.
pub fn float_ln(raw: u64, ft: &FloatType) -> u64 {
    ft.encode_f64(ft.decode_f64(raw).ln())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_ln_f64() {
        let ft = FloatType::F64;

        // ln(1) = 0
        assert_eq!(ft.decode_f64(float_ln(ft.encode_f64(1.0), &ft)), 0.0);

        // ln(e) = 1
        let result = ft.decode_f64(float_ln(ft.encode_f64(std::f64::consts::E), &ft));
        assert!((result - 1.0).abs() < 1e-15);

        // ln(e^2) = 2
        let result = ft.decode_f64(float_ln(ft.encode_f64(std::f64::consts::E * std::f64::consts::E), &ft));
        assert!((result - 2.0).abs() < 1e-14);
    }

    #[test]
    fn float_ln_f32() {
        let ft = FloatType::F32;
        let result = ft.decode_f64(float_ln(ft.encode_f64(1.0), &ft));
        assert!(result.abs() < 1e-6);
    }

    #[test]
    fn float_ln_special_values() {
        let ft = FloatType::F64;

        // ln(0) = -inf
        assert_eq!(ft.decode_f64(float_ln(ft.encode_f64(0.0), &ft)), f64::NEG_INFINITY);

        // ln(-1) = NaN
        assert!(ft.decode_f64(float_ln(ft.encode_f64(-1.0), &ft)).is_nan());

        // ln(NaN) = NaN
        assert!(ft.decode_f64(float_ln(ft.encode_f64(f64::NAN), &ft)).is_nan());

        // ln(inf) = inf
        assert_eq!(ft.decode_f64(float_ln(ft.encode_f64(f64::INFINITY), &ft)), f64::INFINITY);
    }

    #[test]
    fn float_ln_bf16() {
        let ft = FloatType::BF16;
        let result = ft.decode_f64(float_ln(ft.encode_f64(1.0), &ft));
        assert!(result.abs() < 0.01);
    }
}
