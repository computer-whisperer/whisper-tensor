//! Unary hyperbolic tangent — re-exports from [`trig`](super::trig).

pub use super::trig::float_tanh;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric_dtype::FloatType;

    #[test]
    fn float_tanh_f64() {
        let ft = FloatType::F64;

        assert_eq!(ft.decode_f64(float_tanh(ft.encode_f64(0.0), &ft)), 0.0);

        // tanh(large) → 1.0
        let result = ft.decode_f64(float_tanh(ft.encode_f64(100.0), &ft));
        assert!((result - 1.0).abs() < 1e-15);

        // tanh(-large) → -1.0
        let result = ft.decode_f64(float_tanh(ft.encode_f64(-100.0), &ft));
        assert!((result - (-1.0)).abs() < 1e-15);
    }

    #[test]
    fn float_tanh_f32() {
        let ft = FloatType::F32;
        let result = ft.decode_f64(float_tanh(ft.encode_f64(0.0), &ft));
        assert!(result.abs() < 1e-6);
    }

    #[test]
    fn float_tanh_special_values() {
        let ft = FloatType::F64;
        assert!(
            ft.decode_f64(float_tanh(ft.encode_f64(f64::NAN), &ft))
                .is_nan()
        );
        assert_eq!(
            ft.decode_f64(float_tanh(ft.encode_f64(f64::INFINITY), &ft)),
            1.0
        );
        assert_eq!(
            ft.decode_f64(float_tanh(ft.encode_f64(f64::NEG_INFINITY), &ft)),
            -1.0
        );
    }

    #[test]
    fn float_tanh_bf16() {
        let ft = FloatType::BF16;
        let result = ft.decode_f64(float_tanh(ft.encode_f64(0.0), &ft));
        assert!(result.abs() < 0.01);
    }
}
