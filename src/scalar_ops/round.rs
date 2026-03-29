//! Unary rounding (round half to even / banker's rounding).

use crate::numeric_dtype::FloatType;

/// Round a float to the nearest integer, with ties rounding to even.
pub fn float_round(raw: u64, ft: &FloatType) -> u64 {
    let v = ft.decode_f64(raw);
    ft.encode_f64(round_half_to_even(v))
}

/// Round-half-to-even (banker's rounding), matching ONNX Round semantics.
fn round_half_to_even(v: f64) -> f64 {
    let rounded = v.round();
    // Check if we're exactly at .5 — the tie case
    let frac = v - v.floor();
    if (frac - 0.5).abs() < 1e-15 {
        // Round to even
        let floor = v.floor();
        if floor as i64 % 2 == 0 {
            floor
        } else {
            v.ceil()
        }
    } else {
        rounded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_round_f64() {
        let ft = FloatType::F64;

        assert_eq!(ft.decode_f64(float_round(ft.encode_f64(2.3), &ft)), 2.0);
        assert_eq!(ft.decode_f64(float_round(ft.encode_f64(2.7), &ft)), 3.0);
        assert_eq!(ft.decode_f64(float_round(ft.encode_f64(-2.3), &ft)), -2.0);
        assert_eq!(ft.decode_f64(float_round(ft.encode_f64(-2.7), &ft)), -3.0);
        assert_eq!(ft.decode_f64(float_round(ft.encode_f64(0.0), &ft)), 0.0);
    }

    #[test]
    fn float_round_half_to_even() {
        let ft = FloatType::F64;

        // 0.5 → 0 (even)
        assert_eq!(ft.decode_f64(float_round(ft.encode_f64(0.5), &ft)), 0.0);
        // 1.5 → 2 (even)
        assert_eq!(ft.decode_f64(float_round(ft.encode_f64(1.5), &ft)), 2.0);
        // 2.5 → 2 (even)
        assert_eq!(ft.decode_f64(float_round(ft.encode_f64(2.5), &ft)), 2.0);
        // 3.5 → 4 (even)
        assert_eq!(ft.decode_f64(float_round(ft.encode_f64(3.5), &ft)), 4.0);
        // -0.5 → 0 (even)
        assert_eq!(ft.decode_f64(float_round(ft.encode_f64(-0.5), &ft)), 0.0);
        // -1.5 → -2 (even)
        assert_eq!(ft.decode_f64(float_round(ft.encode_f64(-1.5), &ft)), -2.0);
    }

    #[test]
    fn float_round_f32() {
        let ft = FloatType::F32;
        assert_eq!(ft.decode_f64(float_round(ft.encode_f64(2.7), &ft)), 3.0);
    }

    #[test]
    fn float_round_special_values() {
        let ft = FloatType::F64;
        assert!(
            ft.decode_f64(float_round(ft.encode_f64(f64::NAN), &ft))
                .is_nan()
        );
        assert_eq!(
            ft.decode_f64(float_round(ft.encode_f64(f64::INFINITY), &ft)),
            f64::INFINITY
        );
        assert_eq!(
            ft.decode_f64(float_round(ft.encode_f64(f64::NEG_INFINITY), &ft)),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn float_round_bf16() {
        let ft = FloatType::BF16;
        let result = ft.decode_f64(float_round(ft.encode_f64(2.7), &ft));
        assert!((result - 3.0).abs() < 0.1);
    }
}
