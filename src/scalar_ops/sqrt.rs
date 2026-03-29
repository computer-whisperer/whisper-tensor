//! Unary square root.

use crate::numeric_dtype::FloatType;

/// Compute the square root of a float value.
pub fn float_sqrt(raw: u64, ft: &FloatType) -> u64 {
    ft.encode_f64(ft.decode_f64(raw).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_sqrt_f64() {
        let ft = FloatType::F64;

        assert_eq!(ft.decode_f64(float_sqrt(ft.encode_f64(4.0), &ft)), 2.0);
        assert_eq!(ft.decode_f64(float_sqrt(ft.encode_f64(0.0), &ft)), 0.0);
        assert_eq!(ft.decode_f64(float_sqrt(ft.encode_f64(1.0), &ft)), 1.0);

        let result = ft.decode_f64(float_sqrt(ft.encode_f64(2.0), &ft));
        assert!((result - std::f64::consts::SQRT_2).abs() < 1e-15);
    }

    #[test]
    fn float_sqrt_f32() {
        let ft = FloatType::F32;
        let result = ft.decode_f64(float_sqrt(ft.encode_f64(9.0), &ft));
        assert!((result - 3.0).abs() < 1e-6);
    }

    #[test]
    fn float_sqrt_special_values() {
        let ft = FloatType::F64;

        // sqrt(-1) = NaN
        assert!(ft.decode_f64(float_sqrt(ft.encode_f64(-1.0), &ft)).is_nan());

        // sqrt(NaN) = NaN
        assert!(
            ft.decode_f64(float_sqrt(ft.encode_f64(f64::NAN), &ft))
                .is_nan()
        );

        // sqrt(inf) = inf
        assert_eq!(
            ft.decode_f64(float_sqrt(ft.encode_f64(f64::INFINITY), &ft)),
            f64::INFINITY
        );
    }

    #[test]
    fn float_sqrt_bf16() {
        let ft = FloatType::BF16;
        let result = ft.decode_f64(float_sqrt(ft.encode_f64(4.0), &ft));
        assert!((result - 2.0).abs() < 0.1);
    }
}
