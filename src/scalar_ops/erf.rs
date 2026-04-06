//! Error function (erf).

use crate::numeric_dtype::FloatType;

/// Compute the error function using the Abramowitz & Stegun approximation.
/// Maximum error ≈ 1.5e-7.
pub fn float_erf(raw: u64, ft: &FloatType) -> u64 {
    if let Some(r) = super::fast::unary_f32(raw, ft, erf_approx_f32) {
        return r;
    }
    ft.encode_f64(erf_approx_f64(ft.decode_f64(raw)))
}

/// Abramowitz & Stegun formula 7.1.26 approximation of erf(x) in f64.
fn erf_approx_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    let sign: f64 = if x < 0.0 { -1.0 } else { 1.0 };
    let a = x.abs();
    const P: f64 = 0.3275911;
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    let t = 1.0 / (1.0 + P * a);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    let poly = A1 * t + A2 * t2 + A3 * t3 + A4 * t4 + A5 * t5;
    sign * (1.0 - poly * (-a * a).exp())
}

/// Abramowitz & Stegun formula 7.1.26 approximation of erf(x) in f32.
#[allow(clippy::excessive_precision)]
fn erf_approx_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }

    if x == 0.0 {
        return 0.0;
    }

    let sign: f32 = if x < 0.0 { -1.0 } else { 1.0 };
    let a = x.abs();

    const P: f32 = 0.327_591_1;
    const A1: f32 = 0.254_829_59;
    const A2: f32 = -0.284_496_74;
    const A3: f32 = 1.421_413_8;
    const A4: f32 = -1.453_152_1;
    const A5: f32 = 1.061_405_4;

    let t = 1.0f32 / (1.0 + P * a);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;

    let poly = A1 * t + A2 * t2 + A3 * t3 + A4 * t4 + A5 * t5;
    sign * (1.0 - poly * (-a * a).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_erf_f64() {
        let ft = FloatType::F64;

        // erf(0) = 0
        assert_eq!(ft.decode_f64(float_erf(ft.encode_f64(0.0), &ft)), 0.0);

        // erf(1) ≈ 0.8427
        let result = ft.decode_f64(float_erf(ft.encode_f64(1.0), &ft));
        assert!((result - 0.8427007929).abs() < 1e-6);

        // erf(-1) ≈ -0.8427
        let result = ft.decode_f64(float_erf(ft.encode_f64(-1.0), &ft));
        assert!((result - (-0.8427007929)).abs() < 1e-6);

        // erf(large) → 1.0
        let result = ft.decode_f64(float_erf(ft.encode_f64(10.0), &ft));
        assert!((result - 1.0).abs() < 1e-10);

        // erf(-large) → -1.0
        let result = ft.decode_f64(float_erf(ft.encode_f64(-10.0), &ft));
        assert!((result - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn float_erf_f32() {
        let ft = FloatType::F32;
        let result = ft.decode_f64(float_erf(ft.encode_f64(1.0), &ft));
        assert!((result - 0.8427).abs() < 0.001);
    }

    #[test]
    fn float_erf_special_values() {
        let ft = FloatType::F64;
        assert!(
            ft.decode_f64(float_erf(ft.encode_f64(f64::NAN), &ft))
                .is_nan()
        );

        // erf(inf) = 1.0
        let result = ft.decode_f64(float_erf(ft.encode_f64(f64::INFINITY), &ft));
        assert_eq!(result, 1.0);

        // erf(-inf) = -1.0
        let result = ft.decode_f64(float_erf(ft.encode_f64(f64::NEG_INFINITY), &ft));
        assert_eq!(result, -1.0);
    }

    #[test]
    fn float_erf_bf16() {
        let ft = FloatType::BF16;
        let result = ft.decode_f64(float_erf(ft.encode_f64(0.0), &ft));
        assert!(result.abs() < 0.01);
    }
}
