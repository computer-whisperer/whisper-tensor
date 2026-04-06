//! Float NaN predicate.

use crate::numeric_dtype::FloatType;

/// Returns 1 if the float value is NaN, 0 otherwise.
pub fn float_is_nan(raw: u64, ft: &FloatType) -> u64 {
    if let Some(v) = super::fast::decode_f32(raw, ft) {
        return v.is_nan() as u64;
    }
    ft.decode_f64(raw).is_nan() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_is_nan_f64() {
        let ft = FloatType::F64;

        assert_eq!(float_is_nan(ft.encode_f64(f64::NAN), &ft), 1);
        assert_eq!(float_is_nan(ft.encode_f64(0.0), &ft), 0);
        assert_eq!(float_is_nan(ft.encode_f64(1.0), &ft), 0);
        assert_eq!(float_is_nan(ft.encode_f64(f64::INFINITY), &ft), 0);
        assert_eq!(float_is_nan(ft.encode_f64(f64::NEG_INFINITY), &ft), 0);
    }

    #[test]
    fn float_is_nan_f32() {
        let ft = FloatType::F32;
        assert_eq!(float_is_nan(ft.encode_f64(f64::NAN), &ft), 1);
        assert_eq!(float_is_nan(ft.encode_f64(42.0), &ft), 0);
    }

    #[test]
    fn float_is_nan_bf16() {
        let ft = FloatType::BF16;
        assert_eq!(float_is_nan(ft.encode_f64(f64::NAN), &ft), 1);
        assert_eq!(float_is_nan(ft.encode_f64(0.0), &ft), 0);
    }

    #[test]
    fn float_is_nan_f4e2m1() {
        // F4E2M1 has no NaN
        let ft = FloatType::F4E2M1;
        // encode_f64(NAN) returns 0 for no-NaN formats
        assert_eq!(float_is_nan(ft.encode_f64(f64::NAN), &ft), 0);
        assert_eq!(float_is_nan(ft.encode_f64(1.0), &ft), 0);
    }
}
