//! Binary modulo/remainder.

use crate::numeric_dtype::{FloatType, IntType};

/// IEEE float remainder (Rust `%` on f64) — truncated division, sign of
/// result matches the dividend. Equivalent to C `fmod`.
///
/// Used for `ScalarBinOp::Mod` and for `IMod` on FloatTypes ONLY when the
/// caller has explicitly opted out of mathematical modulo. Per
/// `docs/dtype_contract.md` §5.4, the contract for `IMod` is Euclidean
/// — see `float_imod` below.
pub fn float_mod(a: u64, b: u64, ft: &FloatType) -> u64 {
    if let Some(r) = super::fast::binary_f32(a, b, ft, std::ops::Rem::rem) {
        return r;
    }
    ft.encode_f64(ft.decode_f64(a) % ft.decode_f64(b))
}

/// IEEE float mathematical modulo (Euclidean) — sign of result matches
/// the divisor. Matches `signed_imod` for signed integers and the ONNX
/// `Mod` op with `fmod=0` (the default).
///
/// `fmod_then_adjust(a, b) = let r = a % b; if r != 0 && sign(r) != sign(b)
/// then r + b else r`. Returns `NaN` for division by zero.
pub fn float_imod(a: u64, b: u64, ft: &FloatType) -> u64 {
    let af = ft.decode_f64(a);
    let bf = ft.decode_f64(b);
    let rem = af % bf;
    let result = if rem != 0.0 && rem.is_sign_negative() != bf.is_sign_negative() {
        rem + bf
    } else {
        rem
    };
    ft.encode_f64(result)
}

/// Signed integer remainder (truncated division). Division by zero returns 0.
/// Result sign matches the dividend (Rust semantics).
pub fn signed_mod(a: u64, b: u64, it: &IntType) -> u64 {
    let va = it.decode_signed(a);
    let vb = it.decode_signed(b);
    if vb == 0 {
        return 0;
    }
    it.encode_signed(va % vb)
}

/// Signed integer mathematical modulo (result sign matches divisor).
/// ONNX `Mod` with `fmod=0` on integer types.
/// Division by zero returns 0.
pub fn signed_imod(a: u64, b: u64, it: &IntType) -> u64 {
    let va = it.decode_signed(a);
    let vb = it.decode_signed(b);
    if vb == 0 {
        return 0;
    }
    let rem = va % vb;
    let result = if rem != 0 && (rem ^ vb) < 0 {
        rem + vb
    } else {
        rem
    };
    it.encode_signed(result)
}

/// Unsigned integer remainder. Division by zero returns 0.
pub fn unsigned_mod(a: u64, b: u64, it: &IntType) -> u64 {
    let va = it.decode_unsigned(a);
    let vb = it.decode_unsigned(b);
    if vb == 0 {
        return 0;
    }
    it.encode_unsigned(va % vb)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Float mod --

    #[test]
    fn float_mod_f64() {
        let ft = FloatType::F64;
        let a = ft.encode_f64(7.5);
        let b = ft.encode_f64(2.0);
        assert_eq!(ft.decode_f64(float_mod(a, b, &ft)), 1.5);
    }

    #[test]
    fn float_mod_negative() {
        let ft = FloatType::F64;
        let a = ft.encode_f64(-7.5);
        let b = ft.encode_f64(2.0);
        assert_eq!(ft.decode_f64(float_mod(a, b, &ft)), -1.5);
    }

    #[test]
    fn float_mod_by_zero() {
        let ft = FloatType::F64;
        let a = ft.encode_f64(1.0);
        let b = ft.encode_f64(0.0);
        assert!(ft.decode_f64(float_mod(a, b, &ft)).is_nan());
    }

    #[test]
    fn float_mod_nan() {
        let ft = FloatType::F64;
        let nan = ft.encode_f64(f64::NAN);
        let one = ft.encode_f64(1.0);
        assert!(ft.decode_f64(float_mod(nan, one, &ft)).is_nan());
    }

    // -- Float imod (Euclidean / mathematical modulo) --

    #[test]
    fn float_imod_positive_divisor() {
        let ft = FloatType::F64;
        // 7.5 mod 2.0 = 1.5 (no sign correction needed)
        let a = ft.encode_f64(7.5);
        let b = ft.encode_f64(2.0);
        assert_eq!(ft.decode_f64(float_imod(a, b, &ft)), 1.5);

        // -7.5 mod 2.0 = 0.5 (Euclidean: -7.5 = -4*2.0 + 0.5)
        let a = ft.encode_f64(-7.5);
        let b = ft.encode_f64(2.0);
        assert_eq!(ft.decode_f64(float_imod(a, b, &ft)), 0.5);
    }

    #[test]
    fn float_imod_negative_divisor() {
        let ft = FloatType::F64;
        // 7.5 mod -2.0 = -0.5 (Euclidean: 7.5 = -4*-2.0 + -0.5)
        let a = ft.encode_f64(7.5);
        let b = ft.encode_f64(-2.0);
        assert_eq!(ft.decode_f64(float_imod(a, b, &ft)), -0.5);

        // -7.5 mod -2.0 = -1.5 (Euclidean: -7.5 = 3*-2.0 + -1.5)
        let a = ft.encode_f64(-7.5);
        let b = ft.encode_f64(-2.0);
        assert_eq!(ft.decode_f64(float_imod(a, b, &ft)), -1.5);
    }

    #[test]
    fn float_imod_exact() {
        let ft = FloatType::F64;
        // 6.0 mod 2.0 = 0 (no sign correction even though sign matches)
        let a = ft.encode_f64(6.0);
        let b = ft.encode_f64(2.0);
        assert_eq!(ft.decode_f64(float_imod(a, b, &ft)), 0.0);
    }

    #[test]
    fn float_imod_matches_signed_imod_pattern() {
        // For each (a, b) integer pair, float_imod and signed_imod should
        // produce equivalent (real-number-equal) results.
        let ft = FloatType::F64;
        let it = IntType::BITS_32;
        for &(a, b) in &[(7, 3), (-7, 3), (7, -3), (-7, -3), (10, 5), (-10, 5)] {
            let fa = ft.encode_f64(a as f64);
            let fb = ft.encode_f64(b as f64);
            let fr = ft.decode_f64(float_imod(fa, fb, &ft));

            let ia = it.encode_signed(a);
            let ib = it.encode_signed(b);
            let ir = it.decode_signed(signed_imod(ia, ib, &it)) as f64;

            assert_eq!(fr, ir, "mismatch for {a} mod {b}: float={fr}, int={ir}");
        }
    }

    // -- Signed integer mod --

    #[test]
    fn signed_mod_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_signed(17);
        let b = it.encode_signed(5);
        assert_eq!(it.decode_signed(signed_mod(a, b, &it)), 2);
    }

    #[test]
    fn signed_mod_negative_dividend() {
        let it = IntType::BITS_32;
        let a = it.encode_signed(-17);
        let b = it.encode_signed(5);
        assert_eq!(it.decode_signed(signed_mod(a, b, &it)), -2);
    }

    #[test]
    fn signed_mod_by_zero() {
        let it = IntType::BITS_32;
        let a = it.encode_signed(42);
        let b = it.encode_signed(0);
        assert_eq!(signed_mod(a, b, &it), 0);
    }

    // -- Unsigned integer mod --

    #[test]
    fn unsigned_mod_basic() {
        let it = IntType::BITS_32;
        let a = it.encode_unsigned(17);
        let b = it.encode_unsigned(5);
        assert_eq!(it.decode_unsigned(unsigned_mod(a, b, &it)), 2);
    }

    #[test]
    fn unsigned_mod_by_zero() {
        let it = IntType::BITS_32;
        let a = it.encode_unsigned(42);
        let b = it.encode_unsigned(0);
        assert_eq!(unsigned_mod(a, b, &it), 0);
    }

    #[test]
    fn unsigned_mod_exact() {
        let it = IntType::BITS_8;
        let a = it.encode_unsigned(200);
        let b = it.encode_unsigned(50);
        assert_eq!(it.decode_unsigned(unsigned_mod(a, b, &it)), 0);
    }
}
