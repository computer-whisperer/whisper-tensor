//! Trigonometric and hyperbolic functions.

use crate::numeric_dtype::FloatType;

fn trig_unary(raw: u64, ft: &FloatType, f32_op: fn(f32) -> f32, f64_op: fn(f64) -> f64) -> u64 {
    if let Some(r) = super::fast::unary_f32(raw, ft, f32_op) {
        return r;
    }
    ft.encode_f64(f64_op(ft.decode_f64(raw)))
}

pub fn float_sin(raw: u64, ft: &FloatType) -> u64 {
    trig_unary(raw, ft, f32::sin, f64::sin)
}
pub fn float_cos(raw: u64, ft: &FloatType) -> u64 {
    trig_unary(raw, ft, f32::cos, f64::cos)
}
pub fn float_tan(raw: u64, ft: &FloatType) -> u64 {
    trig_unary(raw, ft, f32::tan, f64::tan)
}
pub fn float_asin(raw: u64, ft: &FloatType) -> u64 {
    trig_unary(raw, ft, f32::asin, f64::asin)
}
pub fn float_acos(raw: u64, ft: &FloatType) -> u64 {
    trig_unary(raw, ft, f32::acos, f64::acos)
}
pub fn float_atan(raw: u64, ft: &FloatType) -> u64 {
    trig_unary(raw, ft, f32::atan, f64::atan)
}
pub fn float_sinh(raw: u64, ft: &FloatType) -> u64 {
    trig_unary(raw, ft, f32::sinh, f64::sinh)
}
pub fn float_cosh(raw: u64, ft: &FloatType) -> u64 {
    trig_unary(raw, ft, f32::cosh, f64::cosh)
}
pub fn float_tanh(raw: u64, ft: &FloatType) -> u64 {
    trig_unary(raw, ft, f32::tanh, f64::tanh)
}
pub fn float_asinh(raw: u64, ft: &FloatType) -> u64 {
    trig_unary(raw, ft, f32::asinh, f64::asinh)
}
pub fn float_acosh(raw: u64, ft: &FloatType) -> u64 {
    trig_unary(raw, ft, f32::acosh, f64::acosh)
}
pub fn float_atanh(raw: u64, ft: &FloatType) -> u64 {
    trig_unary(raw, ft, f32::atanh, f64::atanh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    // -- sin --

    #[test]
    fn sin_f64() {
        let ft = FloatType::F64;
        assert_eq!(ft.decode_f64(float_sin(ft.encode_f64(0.0), &ft)), 0.0);

        let result = ft.decode_f64(float_sin(ft.encode_f64(FRAC_PI_2), &ft));
        assert!((result - 1.0).abs() < 1e-15);

        let result = ft.decode_f64(float_sin(ft.encode_f64(PI), &ft));
        assert!(result.abs() < 1e-15);
    }

    #[test]
    fn sin_special() {
        let ft = FloatType::F64;
        assert!(
            ft.decode_f64(float_sin(ft.encode_f64(f64::NAN), &ft))
                .is_nan()
        );
        assert!(
            ft.decode_f64(float_sin(ft.encode_f64(f64::INFINITY), &ft))
                .is_nan()
        );
    }

    // -- cos --

    #[test]
    fn cos_f64() {
        let ft = FloatType::F64;
        assert_eq!(ft.decode_f64(float_cos(ft.encode_f64(0.0), &ft)), 1.0);

        let result = ft.decode_f64(float_cos(ft.encode_f64(FRAC_PI_2), &ft));
        assert!(result.abs() < 1e-15);

        let result = ft.decode_f64(float_cos(ft.encode_f64(PI), &ft));
        assert!((result - (-1.0)).abs() < 1e-15);
    }

    #[test]
    fn cos_special() {
        let ft = FloatType::F64;
        assert!(
            ft.decode_f64(float_cos(ft.encode_f64(f64::NAN), &ft))
                .is_nan()
        );
        assert!(
            ft.decode_f64(float_cos(ft.encode_f64(f64::INFINITY), &ft))
                .is_nan()
        );
    }

    // -- tan --

    #[test]
    fn tan_f64() {
        let ft = FloatType::F64;
        assert_eq!(ft.decode_f64(float_tan(ft.encode_f64(0.0), &ft)), 0.0);

        let result = ft.decode_f64(float_tan(ft.encode_f64(FRAC_PI_4), &ft));
        assert!((result - 1.0).abs() < 1e-14);
    }

    #[test]
    fn tan_special() {
        let ft = FloatType::F64;
        assert!(
            ft.decode_f64(float_tan(ft.encode_f64(f64::NAN), &ft))
                .is_nan()
        );
    }

    // -- asin --

    #[test]
    fn asin_f64() {
        let ft = FloatType::F64;
        assert_eq!(ft.decode_f64(float_asin(ft.encode_f64(0.0), &ft)), 0.0);

        let result = ft.decode_f64(float_asin(ft.encode_f64(1.0), &ft));
        assert!((result - FRAC_PI_2).abs() < 1e-15);

        // asin(2.0) = NaN (out of domain)
        assert!(ft.decode_f64(float_asin(ft.encode_f64(2.0), &ft)).is_nan());
    }

    // -- acos --

    #[test]
    fn acos_f64() {
        let ft = FloatType::F64;

        let result = ft.decode_f64(float_acos(ft.encode_f64(1.0), &ft));
        assert!(result.abs() < 1e-15);

        let result = ft.decode_f64(float_acos(ft.encode_f64(0.0), &ft));
        assert!((result - FRAC_PI_2).abs() < 1e-15);

        // acos(2.0) = NaN
        assert!(ft.decode_f64(float_acos(ft.encode_f64(2.0), &ft)).is_nan());
    }

    // -- atan --

    #[test]
    fn atan_f64() {
        let ft = FloatType::F64;
        assert_eq!(ft.decode_f64(float_atan(ft.encode_f64(0.0), &ft)), 0.0);

        let result = ft.decode_f64(float_atan(ft.encode_f64(1.0), &ft));
        assert!((result - FRAC_PI_4).abs() < 1e-15);

        // atan(inf) = pi/2
        let result = ft.decode_f64(float_atan(ft.encode_f64(f64::INFINITY), &ft));
        assert!((result - FRAC_PI_2).abs() < 1e-15);
    }

    // -- sinh --

    #[test]
    fn sinh_f64() {
        let ft = FloatType::F64;
        assert_eq!(ft.decode_f64(float_sinh(ft.encode_f64(0.0), &ft)), 0.0);

        let result = ft.decode_f64(float_sinh(ft.encode_f64(1.0), &ft));
        assert!((result - 1.0_f64.sinh()).abs() < 1e-15);
    }

    #[test]
    fn sinh_special() {
        let ft = FloatType::F64;
        assert!(
            ft.decode_f64(float_sinh(ft.encode_f64(f64::NAN), &ft))
                .is_nan()
        );
        assert_eq!(
            ft.decode_f64(float_sinh(ft.encode_f64(f64::INFINITY), &ft)),
            f64::INFINITY
        );
    }

    // -- cosh --

    #[test]
    fn cosh_f64() {
        let ft = FloatType::F64;
        assert_eq!(ft.decode_f64(float_cosh(ft.encode_f64(0.0), &ft)), 1.0);

        let result = ft.decode_f64(float_cosh(ft.encode_f64(1.0), &ft));
        assert!((result - 1.0_f64.cosh()).abs() < 1e-15);
    }

    #[test]
    fn cosh_special() {
        let ft = FloatType::F64;
        assert!(
            ft.decode_f64(float_cosh(ft.encode_f64(f64::NAN), &ft))
                .is_nan()
        );
        assert_eq!(
            ft.decode_f64(float_cosh(ft.encode_f64(f64::INFINITY), &ft)),
            f64::INFINITY
        );
    }

    // -- tanh --

    #[test]
    fn tanh_f64() {
        let ft = FloatType::F64;
        assert_eq!(ft.decode_f64(float_tanh(ft.encode_f64(0.0), &ft)), 0.0);

        let result = ft.decode_f64(float_tanh(ft.encode_f64(100.0), &ft));
        assert!((result - 1.0).abs() < 1e-15);

        let result = ft.decode_f64(float_tanh(ft.encode_f64(-100.0), &ft));
        assert!((result - (-1.0)).abs() < 1e-15);
    }

    #[test]
    fn tanh_special() {
        let ft = FloatType::F64;
        assert!(
            ft.decode_f64(float_tanh(ft.encode_f64(f64::NAN), &ft))
                .is_nan()
        );
        assert_eq!(
            ft.decode_f64(float_tanh(ft.encode_f64(f64::INFINITY), &ft)),
            1.0
        );
    }

    // -- Cross-format tests --

    #[test]
    fn trig_f32() {
        let ft = FloatType::F32;
        let result = ft.decode_f64(float_cos(ft.encode_f64(0.0), &ft));
        assert!((result - 1.0).abs() < 1e-6);

        let result = ft.decode_f64(float_tan(ft.encode_f64(0.0), &ft));
        assert!(result.abs() < 1e-6);
    }

    #[test]
    fn trig_bf16() {
        let ft = FloatType::BF16;
        let result = ft.decode_f64(float_sin(ft.encode_f64(0.0), &ft));
        assert!(result.abs() < 0.01);

        let result = ft.decode_f64(float_cos(ft.encode_f64(0.0), &ft));
        assert!((result - 1.0).abs() < 0.01);
    }
}
