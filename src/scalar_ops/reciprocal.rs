//! Unary reciprocal (1/x).

use crate::numeric_dtype::FloatType;

/// Compute 1/x for a float value.
pub fn float_reciprocal(raw: u64, ft: &FloatType) -> u64 {
    ft.encode_f64(1.0 / ft.decode_f64(raw))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_reciprocal_f64() {
        let ft = FloatType::F64;

        assert_eq!(
            ft.decode_f64(float_reciprocal(ft.encode_f64(2.0), &ft)),
            0.5
        );
        assert_eq!(
            ft.decode_f64(float_reciprocal(ft.encode_f64(1.0), &ft)),
            1.0
        );
        assert_eq!(
            ft.decode_f64(float_reciprocal(ft.encode_f64(-4.0), &ft)),
            -0.25
        );
    }

    #[test]
    fn float_reciprocal_f32() {
        let ft = FloatType::F32;
        let result = ft.decode_f64(float_reciprocal(ft.encode_f64(5.0), &ft));
        assert!((result - 0.2).abs() < 1e-6);
    }

    #[test]
    fn float_reciprocal_zero() {
        let ft = FloatType::F64;
        assert_eq!(
            ft.decode_f64(float_reciprocal(ft.encode_f64(0.0), &ft)),
            f64::INFINITY
        );
        assert_eq!(
            ft.decode_f64(float_reciprocal(ft.encode_f64(-0.0), &ft)),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn float_reciprocal_special_values() {
        let ft = FloatType::F64;

        assert!(
            ft.decode_f64(float_reciprocal(ft.encode_f64(f64::NAN), &ft))
                .is_nan()
        );
        assert_eq!(
            ft.decode_f64(float_reciprocal(ft.encode_f64(f64::INFINITY), &ft)),
            0.0
        );
    }

    #[test]
    fn float_reciprocal_bf16() {
        let ft = FloatType::BF16;
        let result = ft.decode_f64(float_reciprocal(ft.encode_f64(2.0), &ft));
        assert!((result - 0.5).abs() < 0.01);
    }
}
