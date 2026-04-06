//! Unary exponential (e^x).

use crate::numeric_dtype::FloatType;

/// Compute e^x for a float value.
pub fn float_exp(raw: u64, ft: &FloatType) -> u64 {
    if let Some(r) = super::fast::unary_f32(raw, ft, f32::exp) {
        return r;
    }
    ft.encode_f64(ft.decode_f64(raw).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_exp_f64() {
        let ft = FloatType::F64;

        // e^0 = 1
        assert_eq!(ft.decode_f64(float_exp(ft.encode_f64(0.0), &ft)), 1.0);

        // e^1 ≈ 2.718
        let result = ft.decode_f64(float_exp(ft.encode_f64(1.0), &ft));
        assert!((result - std::f64::consts::E).abs() < 1e-15);

        // e^(-large) ≈ 0
        let result = ft.decode_f64(float_exp(ft.encode_f64(-100.0), &ft));
        assert!(result >= 0.0 && result < 1e-40);
    }

    #[test]
    fn float_exp_f32() {
        let ft = FloatType::F32;
        let result = ft.decode_f64(float_exp(ft.encode_f64(1.0), &ft));
        assert!((result - std::f64::consts::E).abs() < 1e-6);
    }

    #[test]
    fn float_exp_special_values() {
        let ft = FloatType::F64;
        assert!(
            ft.decode_f64(float_exp(ft.encode_f64(f64::NAN), &ft))
                .is_nan()
        );
        assert_eq!(
            ft.decode_f64(float_exp(ft.encode_f64(f64::INFINITY), &ft)),
            f64::INFINITY
        );
        assert_eq!(
            ft.decode_f64(float_exp(ft.encode_f64(f64::NEG_INFINITY), &ft)),
            0.0
        );
    }

    #[test]
    fn float_exp_bf16() {
        let ft = FloatType::BF16;
        let result = ft.decode_f64(float_exp(ft.encode_f64(0.0), &ft));
        assert!((result - 1.0).abs() < 0.01);
    }
}
