//! Float infinity predicate.

use crate::numeric_dtype::FloatType;

/// Returns 1 if the float value is an infinity matching the requested signs.
/// `detect_positive`: include +Inf. `detect_negative`: include -Inf.
pub fn float_is_inf(raw: u64, ft: &FloatType, detect_positive: bool, detect_negative: bool) -> u64 {
    let v = ft.decode_f64(raw);
    let hit =
        (detect_positive && v == f64::INFINITY) || (detect_negative && v == f64::NEG_INFINITY);
    hit as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_is_inf_both() {
        let ft = FloatType::F64;

        assert_eq!(
            float_is_inf(ft.encode_f64(f64::INFINITY), &ft, true, true),
            1
        );
        assert_eq!(
            float_is_inf(ft.encode_f64(f64::NEG_INFINITY), &ft, true, true),
            1
        );
        assert_eq!(float_is_inf(ft.encode_f64(0.0), &ft, true, true), 0);
        assert_eq!(float_is_inf(ft.encode_f64(f64::NAN), &ft, true, true), 0);
    }

    #[test]
    fn float_is_inf_positive_only() {
        let ft = FloatType::F64;

        assert_eq!(
            float_is_inf(ft.encode_f64(f64::INFINITY), &ft, true, false),
            1
        );
        assert_eq!(
            float_is_inf(ft.encode_f64(f64::NEG_INFINITY), &ft, true, false),
            0
        );
    }

    #[test]
    fn float_is_inf_negative_only() {
        let ft = FloatType::F64;

        assert_eq!(
            float_is_inf(ft.encode_f64(f64::INFINITY), &ft, false, true),
            0
        );
        assert_eq!(
            float_is_inf(ft.encode_f64(f64::NEG_INFINITY), &ft, false, true),
            1
        );
    }

    #[test]
    fn float_is_inf_neither() {
        let ft = FloatType::F64;

        assert_eq!(
            float_is_inf(ft.encode_f64(f64::INFINITY), &ft, false, false),
            0
        );
        assert_eq!(
            float_is_inf(ft.encode_f64(f64::NEG_INFINITY), &ft, false, false),
            0
        );
    }

    #[test]
    fn float_is_inf_f32() {
        let ft = FloatType::F32;
        assert_eq!(
            float_is_inf(ft.encode_f64(f64::INFINITY), &ft, true, true),
            1
        );
        assert_eq!(float_is_inf(ft.encode_f64(42.0), &ft, true, true), 0);
    }

    #[test]
    fn float_is_inf_f4e2m1() {
        // F4E2M1 has no infinity
        let ft = FloatType::F4E2M1;
        // encode_f64(INFINITY) saturates to max finite for no-infinity formats
        assert_eq!(
            float_is_inf(ft.encode_f64(f64::INFINITY), &ft, true, true),
            0
        );
    }
}
