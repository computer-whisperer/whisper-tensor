//! Unary ceiling.

use crate::numeric_dtype::FloatType;

/// Compute the ceiling of a float value.
pub fn float_ceil(raw: u64, ft: &FloatType) -> u64 {
    ft.encode_f64(ft.decode_f64(raw).ceil())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_ceil_f64() {
        let ft = FloatType::F64;

        assert_eq!(ft.decode_f64(float_ceil(ft.encode_f64(2.3), &ft)), 3.0);
        assert_eq!(ft.decode_f64(float_ceil(ft.encode_f64(-2.7), &ft)), -2.0);
        assert_eq!(ft.decode_f64(float_ceil(ft.encode_f64(3.0), &ft)), 3.0);
        assert_eq!(ft.decode_f64(float_ceil(ft.encode_f64(0.0), &ft)), 0.0);
    }

    #[test]
    fn float_ceil_f32() {
        let ft = FloatType::F32;
        assert_eq!(ft.decode_f64(float_ceil(ft.encode_f64(2.1), &ft)), 3.0);
    }

    #[test]
    fn float_ceil_special_values() {
        let ft = FloatType::F64;
        assert!(
            ft.decode_f64(float_ceil(ft.encode_f64(f64::NAN), &ft))
                .is_nan()
        );
        assert_eq!(
            ft.decode_f64(float_ceil(ft.encode_f64(f64::INFINITY), &ft)),
            f64::INFINITY
        );
        assert_eq!(
            ft.decode_f64(float_ceil(ft.encode_f64(f64::NEG_INFINITY), &ft)),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn float_ceil_bf16() {
        let ft = FloatType::BF16;
        let result = ft.decode_f64(float_ceil(ft.encode_f64(2.5), &ft));
        assert!((result - 3.0).abs() < 0.1);
    }
}
