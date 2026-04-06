//! Comparison operations.
//!
//! All return u64 where 1 = true, 0 = false (Bool raw bits).

use crate::numeric_dtype::{FloatType, IntType};

// -- Float comparisons --

pub fn float_equal(a: u64, b: u64, ft: &FloatType) -> u64 {
    if let (Some(va), Some(vb)) = (
        super::fast::decode_f32(a, ft),
        super::fast::decode_f32(b, ft),
    ) {
        return (va == vb) as u64;
    }
    (ft.decode_f64(a) == ft.decode_f64(b)) as u64
}

pub fn float_greater(a: u64, b: u64, ft: &FloatType) -> u64 {
    if let (Some(va), Some(vb)) = (
        super::fast::decode_f32(a, ft),
        super::fast::decode_f32(b, ft),
    ) {
        return (va > vb) as u64;
    }
    (ft.decode_f64(a) > ft.decode_f64(b)) as u64
}

pub fn float_greater_or_equal(a: u64, b: u64, ft: &FloatType) -> u64 {
    if let (Some(va), Some(vb)) = (
        super::fast::decode_f32(a, ft),
        super::fast::decode_f32(b, ft),
    ) {
        return (va >= vb) as u64;
    }
    (ft.decode_f64(a) >= ft.decode_f64(b)) as u64
}

pub fn float_less(a: u64, b: u64, ft: &FloatType) -> u64 {
    if let (Some(va), Some(vb)) = (
        super::fast::decode_f32(a, ft),
        super::fast::decode_f32(b, ft),
    ) {
        return (va < vb) as u64;
    }
    (ft.decode_f64(a) < ft.decode_f64(b)) as u64
}

pub fn float_less_or_equal(a: u64, b: u64, ft: &FloatType) -> u64 {
    if let (Some(va), Some(vb)) = (
        super::fast::decode_f32(a, ft),
        super::fast::decode_f32(b, ft),
    ) {
        return (va <= vb) as u64;
    }
    (ft.decode_f64(a) <= ft.decode_f64(b)) as u64
}

// -- Signed integer comparisons --

pub fn signed_equal(a: u64, b: u64, it: &IntType) -> u64 {
    (it.decode_signed(a) == it.decode_signed(b)) as u64
}

pub fn signed_greater(a: u64, b: u64, it: &IntType) -> u64 {
    (it.decode_signed(a) > it.decode_signed(b)) as u64
}

pub fn signed_greater_or_equal(a: u64, b: u64, it: &IntType) -> u64 {
    (it.decode_signed(a) >= it.decode_signed(b)) as u64
}

pub fn signed_less(a: u64, b: u64, it: &IntType) -> u64 {
    (it.decode_signed(a) < it.decode_signed(b)) as u64
}

pub fn signed_less_or_equal(a: u64, b: u64, it: &IntType) -> u64 {
    (it.decode_signed(a) <= it.decode_signed(b)) as u64
}

// -- Unsigned integer comparisons --

pub fn unsigned_equal(a: u64, b: u64, it: &IntType) -> u64 {
    (it.decode_unsigned(a) == it.decode_unsigned(b)) as u64
}

pub fn unsigned_greater(a: u64, b: u64, it: &IntType) -> u64 {
    (it.decode_unsigned(a) > it.decode_unsigned(b)) as u64
}

pub fn unsigned_greater_or_equal(a: u64, b: u64, it: &IntType) -> u64 {
    (it.decode_unsigned(a) >= it.decode_unsigned(b)) as u64
}

pub fn unsigned_less(a: u64, b: u64, it: &IntType) -> u64 {
    (it.decode_unsigned(a) < it.decode_unsigned(b)) as u64
}

pub fn unsigned_less_or_equal(a: u64, b: u64, it: &IntType) -> u64 {
    (it.decode_unsigned(a) <= it.decode_unsigned(b)) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Float comparisons --

    #[test]
    fn float_equal_basic() {
        let ft = FloatType::F64;
        let a = ft.encode_f64(1.0);
        let b = ft.encode_f64(1.0);
        let c = ft.encode_f64(2.0);
        assert_eq!(float_equal(a, b, &ft), 1);
        assert_eq!(float_equal(a, c, &ft), 0);
    }

    #[test]
    fn float_equal_nan() {
        let ft = FloatType::F64;
        let nan = ft.encode_f64(f64::NAN);
        let one = ft.encode_f64(1.0);
        // NaN != NaN
        assert_eq!(float_equal(nan, nan, &ft), 0);
        assert_eq!(float_equal(nan, one, &ft), 0);
    }

    #[test]
    fn float_greater_basic() {
        let ft = FloatType::F64;
        let a = ft.encode_f64(3.0);
        let b = ft.encode_f64(2.0);
        assert_eq!(float_greater(a, b, &ft), 1);
        assert_eq!(float_greater(b, a, &ft), 0);
        assert_eq!(float_greater(a, a, &ft), 0);
    }

    #[test]
    fn float_greater_or_equal_basic() {
        let ft = FloatType::F64;
        let a = ft.encode_f64(3.0);
        let b = ft.encode_f64(3.0);
        let c = ft.encode_f64(2.0);
        assert_eq!(float_greater_or_equal(a, b, &ft), 1);
        assert_eq!(float_greater_or_equal(a, c, &ft), 1);
        assert_eq!(float_greater_or_equal(c, a, &ft), 0);
    }

    #[test]
    fn float_less_basic() {
        let ft = FloatType::F64;
        let a = ft.encode_f64(2.0);
        let b = ft.encode_f64(3.0);
        assert_eq!(float_less(a, b, &ft), 1);
        assert_eq!(float_less(b, a, &ft), 0);
    }

    #[test]
    fn float_less_or_equal_basic() {
        let ft = FloatType::F64;
        let a = ft.encode_f64(2.0);
        let b = ft.encode_f64(2.0);
        assert_eq!(float_less_or_equal(a, b, &ft), 1);
    }

    #[test]
    fn float_cmp_nan_is_false() {
        let ft = FloatType::F64;
        let nan = ft.encode_f64(f64::NAN);
        let one = ft.encode_f64(1.0);
        // All comparisons with NaN return false
        assert_eq!(float_greater(nan, one, &ft), 0);
        assert_eq!(float_greater_or_equal(nan, one, &ft), 0);
        assert_eq!(float_less(nan, one, &ft), 0);
        assert_eq!(float_less_or_equal(nan, one, &ft), 0);
    }

    #[test]
    fn float_cmp_inf() {
        let ft = FloatType::F64;
        let inf = ft.encode_f64(f64::INFINITY);
        let one = ft.encode_f64(1.0);
        assert_eq!(float_greater(inf, one, &ft), 1);
        assert_eq!(float_less(one, inf, &ft), 1);
    }

    // -- Signed integer comparisons --

    #[test]
    fn signed_equal_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_signed(5);
        let b = it.encode_signed(5);
        let c = it.encode_signed(-5);
        assert_eq!(signed_equal(a, b, &it), 1);
        assert_eq!(signed_equal(a, c, &it), 0);
    }

    #[test]
    fn signed_greater_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_signed(5);
        let b = it.encode_signed(-5);
        assert_eq!(signed_greater(a, b, &it), 1);
        assert_eq!(signed_greater(b, a, &it), 0);
    }

    #[test]
    fn signed_less_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_signed(-10);
        let b = it.encode_signed(10);
        assert_eq!(signed_less(a, b, &it), 1);
        assert_eq!(signed_less(b, a, &it), 0);
    }

    #[test]
    fn signed_greater_or_equal_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_signed(5);
        let b = it.encode_signed(5);
        assert_eq!(signed_greater_or_equal(a, b, &it), 1);
    }

    #[test]
    fn signed_less_or_equal_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_signed(5);
        let b = it.encode_signed(5);
        assert_eq!(signed_less_or_equal(a, b, &it), 1);
    }

    // -- Unsigned integer comparisons --

    #[test]
    fn unsigned_equal_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_unsigned(100);
        let b = it.encode_unsigned(100);
        let c = it.encode_unsigned(200);
        assert_eq!(unsigned_equal(a, b, &it), 1);
        assert_eq!(unsigned_equal(a, c, &it), 0);
    }

    #[test]
    fn unsigned_greater_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_unsigned(200);
        let b = it.encode_unsigned(100);
        assert_eq!(unsigned_greater(a, b, &it), 1);
        assert_eq!(unsigned_greater(b, a, &it), 0);
    }

    #[test]
    fn unsigned_less_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_unsigned(50);
        let b = it.encode_unsigned(100);
        assert_eq!(unsigned_less(a, b, &it), 1);
        assert_eq!(unsigned_less(b, a, &it), 0);
    }

    #[test]
    fn unsigned_greater_or_equal_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_unsigned(100);
        let b = it.encode_unsigned(100);
        assert_eq!(unsigned_greater_or_equal(a, b, &it), 1);
    }

    #[test]
    fn unsigned_less_or_equal_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_unsigned(100);
        let b = it.encode_unsigned(100);
        assert_eq!(unsigned_less_or_equal(a, b, &it), 1);
    }
}
