//! Software cast operator: convert any [`NumericScalar`] to any [`NumericDType`].
//!
//! The implementation is fully software-defined — it interprets the
//! [`FloatType`] fields (exponent_bits, mantissa_bits, semantics) directly
//! rather than dispatching on known type names. Any `FloatType` within the
//! supported bounds ([`FloatType::is_supported`]) works correctly.
//!
//! Float conversions implement IEEE 754 round-to-nearest-even.

use crate::numeric_dtype::{FloatSemantics, FloatType, IntType, NumericDType};

use super::NumericScalar;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

impl NumericScalar {
    /// Cast this scalar to a different dtype.
    ///
    /// Implements IEEE 754 round-to-nearest-even for float conversions.
    /// Integer conversions use Rust's saturating/truncating semantics.
    pub fn cast_to(&self, target: NumericDType) -> NumericScalar {
        if self.dtype == target {
            return *self;
        }
        cast_scalar(self.bits, self.dtype, target)
    }

    // -- Safe to_* methods: cast then read bits --

    /// Convert to f64 regardless of source dtype.
    pub fn to_f64(&self) -> f64 {
        if self.dtype == NumericDType::F64 {
            return f64::from_le_bytes(self.bits);
        }
        f64::from_le_bytes(self.cast_to(NumericDType::F64).bits)
    }

    /// Convert to f32 regardless of source dtype.
    pub fn to_f32(&self) -> f32 {
        if self.dtype == NumericDType::F32 {
            return f32::from_le_bytes([self.bits[0], self.bits[1], self.bits[2], self.bits[3]]);
        }
        let cast = self.cast_to(NumericDType::F32);
        f32::from_le_bytes([cast.bits[0], cast.bits[1], cast.bits[2], cast.bits[3]])
    }

    /// Convert to i64 regardless of source dtype.
    pub fn to_i64(&self) -> i64 {
        if self.dtype == NumericDType::I64 {
            return i64::from_le_bytes(self.bits);
        }
        i64::from_le_bytes(self.cast_to(NumericDType::I64).bits)
    }

    /// Convert to bool (nonzero = true).
    pub fn is_nonzero(&self) -> bool {
        if self.dtype == NumericDType::BOOL {
            return self.bits[0] != 0;
        }
        self.cast_to(NumericDType::BOOL).bits[0] != 0
    }
}

// ---------------------------------------------------------------------------
// Software cast engine
// ---------------------------------------------------------------------------

/// Cast raw bits from one dtype to another.
fn cast_scalar(bits: [u8; 8], src: NumericDType, dst: NumericDType) -> NumericScalar {
    // Step 1: Decode source to a universal intermediate representation.
    let intermediate = decode_to_intermediate(bits, src);

    // Step 2: Encode the intermediate into the target dtype.
    encode_from_intermediate(intermediate, dst)
}

/// Universal intermediate value. Either an f64 (for float sources) or
/// i128 (for integer sources — wide enough for any 64-bit int without loss).
#[derive(Debug, Clone, Copy)]
enum Intermediate {
    Float(f64),
    Int(i128),
    Bool(bool),
}

/// Decode raw bits + dtype into the universal intermediate.
fn decode_to_intermediate(bits: [u8; 8], dtype: NumericDType) -> Intermediate {
    match dtype {
        NumericDType::Float(ft) => Intermediate::Float(software_float_decode(bits, &ft)),
        NumericDType::SignedInt(it) => Intermediate::Int(decode_signed_int(bits, &it)),
        NumericDType::UnsignedInt(it) => Intermediate::Int(decode_unsigned_int(bits, &it) as i128),
        NumericDType::Bool => Intermediate::Bool(bits[0] != 0),
    }
}

/// Encode the universal intermediate into a target dtype.
fn encode_from_intermediate(value: Intermediate, dtype: NumericDType) -> NumericScalar {
    match dtype {
        NumericDType::Float(ft) => {
            let f = match value {
                Intermediate::Float(f) => f,
                Intermediate::Int(i) => i as f64,
                Intermediate::Bool(b) => if b { 1.0 } else { 0.0 },
            };
            let bits = software_float_encode(f, &ft);
            NumericScalar { bits, dtype }
        }
        NumericDType::SignedInt(it) => {
            let i = match value {
                Intermediate::Float(f) => float_to_signed_int(f, &it),
                Intermediate::Int(i) => truncate_signed(i, &it),
                Intermediate::Bool(b) => if b { 1 } else { 0 },
            };
            let bits = encode_signed_int(i, &it);
            NumericScalar { bits, dtype }
        }
        NumericDType::UnsignedInt(it) => {
            let u = match value {
                Intermediate::Float(f) => float_to_unsigned_int(f, &it),
                Intermediate::Int(i) => truncate_unsigned(i, &it),
                Intermediate::Bool(b) => if b { 1 } else { 0 },
            };
            let bits = encode_unsigned_int(u, &it);
            NumericScalar { bits, dtype }
        }
        NumericDType::Bool => {
            let b = match value {
                Intermediate::Float(f) => f != 0.0 && !f.is_nan(),
                Intermediate::Int(i) => i != 0,
                Intermediate::Bool(b) => b,
            };
            let mut bits = [0u8; 8];
            bits[0] = b as u8;
            NumericScalar { bits, dtype }
        }
    }
}

// ---------------------------------------------------------------------------
// Software float decode: raw bits → f64
// ---------------------------------------------------------------------------

/// Decode a float value from raw bits according to the FloatType spec.
/// Works for ANY supported FloatType configuration, not just known constants.
fn software_float_decode(bits: [u8; 8], ft: &FloatType) -> f64 {
    let total = ft.total_bits();
    let raw = u64_from_le_bytes_masked(bits, total);

    let mant_bits = ft.mantissa_bits as u32;
    let exp_bits = ft.exponent_bits as u32;

    // Extract fields
    let sign_bit = (raw >> (total as u32 - 1)) & 1;
    let exp_mask = (1u64 << exp_bits) - 1;
    let mant_mask = (1u64 << mant_bits) - 1;
    let biased_exp = ((raw >> mant_bits) & exp_mask) as u32;
    let raw_mant = raw & mant_mask;

    let bias = ft.bias();
    let sign = if sign_bit == 1 { -1.0f64 } else { 1.0f64 };

    let max_exp = ft.max_biased_exponent();

    if biased_exp == max_exp {
        // All exponent bits set — special values
        match ft.semantics {
            FloatSemantics::IEEE => {
                if raw_mant == 0 {
                    sign * f64::INFINITY
                } else {
                    f64::NAN
                }
            }
            FloatSemantics::FN => {
                // FN: all-ones exponent + all-ones mantissa = NaN, else normal
                if raw_mant == mant_mask {
                    f64::NAN
                } else {
                    // Normal number with max exponent
                    let mant_val = 1.0 + (raw_mant as f64) / ((1u64 << mant_bits) as f64);
                    sign * mant_val * f64::exp2((biased_exp as i32 - bias) as f64)
                }
            }
            FloatSemantics::FNUZ => {
                // FNUZ: all-ones exponent = NaN (any mantissa)
                f64::NAN
            }
        }
    } else if biased_exp == 0 {
        // Subnormal or zero
        match ft.semantics {
            FloatSemantics::IEEE | FloatSemantics::FN => {
                if raw_mant == 0 {
                    // Zero (preserves sign for IEEE/FN)
                    sign * 0.0
                } else {
                    // Subnormal: no implicit leading 1, exponent is 1 - bias
                    let mant_val = (raw_mant as f64) / ((1u64 << mant_bits) as f64);
                    sign * mant_val * f64::exp2((1 - bias) as f64)
                }
            }
            FloatSemantics::FNUZ => {
                if raw_mant == 0 && sign_bit == 0 {
                    0.0 // Positive zero only (FNUZ has no negative zero)
                } else if raw_mant == 0 && sign_bit == 1 {
                    // FNUZ: negative zero bit pattern is NaN
                    f64::NAN
                } else {
                    // Subnormal
                    let mant_val = (raw_mant as f64) / ((1u64 << mant_bits) as f64);
                    sign * mant_val * f64::exp2((1 - bias) as f64)
                }
            }
        }
    } else {
        // Normal number
        let mant_val = 1.0 + (raw_mant as f64) / ((1u64 << mant_bits) as f64);
        sign * mant_val * f64::exp2((biased_exp as i32 - bias) as f64)
    }
}

// ---------------------------------------------------------------------------
// Software float encode: f64 → raw bits
// ---------------------------------------------------------------------------

/// Encode an f64 value into raw bits for the target FloatType.
/// Implements IEEE 754 round-to-nearest-even.
fn software_float_encode(value: f64, ft: &FloatType) -> [u8; 8] {
    let total = ft.total_bits();
    let mant_bits = ft.mantissa_bits as u32;
    let exp_bits = ft.exponent_bits as u32;
    let bias = ft.bias();
    let max_exp = ft.max_biased_exponent();

    // Handle special values
    if value.is_nan() {
        return encode_nan(ft);
    }

    let sign_bit: u64 = if value.is_sign_negative() { 1 } else { 0 };
    let abs_val = value.abs();

    if abs_val == 0.0 {
        return encode_zero(sign_bit, ft);
    }

    if value.is_infinite() {
        return encode_infinity(sign_bit, ft);
    }

    // Decompose the absolute value: abs_val = mantissa * 2^exponent
    // where 1.0 <= mantissa < 2.0 (for normal numbers)
    let (frac, raw_exp) = frexp_f64(abs_val);
    // frac is in [0.5, 1.0), raw_exp is such that abs_val = frac * 2^raw_exp
    // Normalize to [1.0, 2.0): mantissa = frac * 2, exponent = raw_exp - 1
    let true_exp = raw_exp - 1; // unbiased exponent

    // Check if the value fits as a normal number
    let min_normal_exp = 1 - bias; // minimum unbiased exponent for normal numbers
    let max_normal_exp = (max_exp as i32 - 1) - bias; // maximum unbiased exponent

    if true_exp > max_normal_exp {
        // Overflow — need to check if rounding could bring it back to max finite
        return encode_overflow(sign_bit, ft);
    }

    if true_exp >= min_normal_exp {
        // Normal number
        encode_normal(sign_bit, frac * 2.0 - 1.0, true_exp, ft)
    } else {
        // Subnormal region
        encode_subnormal(sign_bit, abs_val, ft)
    }
}

/// Encode NaN in the target format.
fn encode_nan(ft: &FloatType) -> [u8; 8] {
    let mant_bits = ft.mantissa_bits as u32;
    let exp_bits = ft.exponent_bits as u32;
    let max_exp = ft.max_biased_exponent();

    let (exp_val, mant_val) = match ft.semantics {
        FloatSemantics::IEEE => {
            // Quiet NaN: all exponent bits set, MSB of mantissa set
            (max_exp as u64, 1u64 << (mant_bits - 1))
        }
        FloatSemantics::FN => {
            // FN NaN: all exponent bits set, all mantissa bits set
            let mant_mask = (1u64 << mant_bits) - 1;
            (max_exp as u64, mant_mask)
        }
        FloatSemantics::FNUZ => {
            // FNUZ NaN: negative zero pattern (sign=1, exp=0, mant=0)
            // Encoded specially below
            let raw = 1u64 << (ft.total_bits() as u32 - 1); // just the sign bit
            return u64_to_le_bytes_masked(raw, ft.total_bits());
        }
    };

    let raw = (exp_val << mant_bits) | mant_val;
    u64_to_le_bytes_masked(raw, ft.total_bits())
}

/// Encode zero in the target format.
fn encode_zero(sign_bit: u64, ft: &FloatType) -> [u8; 8] {
    match ft.semantics {
        FloatSemantics::IEEE | FloatSemantics::FN => {
            let raw = sign_bit << (ft.total_bits() as u32 - 1);
            u64_to_le_bytes_masked(raw, ft.total_bits())
        }
        FloatSemantics::FNUZ => {
            // FNUZ: only positive zero exists
            u64_to_le_bytes_masked(0, ft.total_bits())
        }
    }
}

/// Encode infinity in the target format.
fn encode_infinity(sign_bit: u64, ft: &FloatType) -> [u8; 8] {
    match ft.semantics {
        FloatSemantics::IEEE => {
            let max_exp = ft.max_biased_exponent() as u64;
            let raw = (sign_bit << (ft.total_bits() as u32 - 1))
                | (max_exp << ft.mantissa_bits as u32);
            u64_to_le_bytes_masked(raw, ft.total_bits())
        }
        FloatSemantics::FN | FloatSemantics::FNUZ => {
            // No infinities — saturate to max finite value
            encode_max_finite(sign_bit, ft)
        }
    }
}

/// Encode the maximum finite value.
fn encode_max_finite(sign_bit: u64, ft: &FloatType) -> [u8; 8] {
    let mant_bits = ft.mantissa_bits as u32;
    let max_exp = ft.max_biased_exponent() as u64;
    let mant_mask = (1u64 << mant_bits) - 1;

    let (exp_val, mant_val) = match ft.semantics {
        FloatSemantics::IEEE => {
            // Max normal: exp = max_exp - 1, mantissa = all ones
            (max_exp - 1, mant_mask)
        }
        FloatSemantics::FN => {
            // FN: max_exp with all-ones mantissa is NaN, so max finite is
            // max_exp with mantissa = all-ones - 1
            (max_exp, mant_mask - 1)
        }
        FloatSemantics::FNUZ => {
            // FNUZ: all max_exp values are NaN, so max finite is
            // (max_exp - 1) with all-ones mantissa
            (max_exp - 1, mant_mask)
        }
    };

    let raw = (sign_bit << (ft.total_bits() as u32 - 1))
        | (exp_val << mant_bits)
        | mant_val;
    u64_to_le_bytes_masked(raw, ft.total_bits())
}

/// Handle overflow: value too large for the format.
fn encode_overflow(sign_bit: u64, ft: &FloatType) -> [u8; 8] {
    match ft.semantics {
        FloatSemantics::IEEE => encode_infinity(sign_bit, ft),
        FloatSemantics::FN | FloatSemantics::FNUZ => encode_max_finite(sign_bit, ft),
    }
}

/// Encode a normal number. `frac_part` is the fractional mantissa (0.0 to <1.0),
/// `true_exp` is the unbiased exponent.
fn encode_normal(sign_bit: u64, frac_part: f64, true_exp: i32, ft: &FloatType) -> [u8; 8] {
    let mant_bits = ft.mantissa_bits as u32;
    let bias = ft.bias();
    let biased_exp = (true_exp + bias) as u64;

    // Scale the fractional part to integer mantissa bits
    let scale = (1u64 << mant_bits) as f64;
    let scaled = frac_part * scale;

    // Round-to-nearest-even
    let mant_int = round_to_nearest_even(scaled);

    // Check for mantissa overflow (rounding carried into exponent)
    if mant_int >= (1u64 << mant_bits) {
        let new_exp = biased_exp + 1;
        let max_exp = ft.max_biased_exponent() as u64;
        if new_exp >= max_exp {
            return encode_overflow(sign_bit, ft);
        }
        // Mantissa becomes 0 (the carry absorbed the leading 1)
        let raw = (sign_bit << (ft.total_bits() as u32 - 1))
            | (new_exp << mant_bits);
        return u64_to_le_bytes_masked(raw, ft.total_bits());
    }

    let raw = (sign_bit << (ft.total_bits() as u32 - 1))
        | (biased_exp << mant_bits)
        | mant_int;
    u64_to_le_bytes_masked(raw, ft.total_bits())
}

/// Encode a subnormal number.
fn encode_subnormal(sign_bit: u64, abs_val: f64, ft: &FloatType) -> [u8; 8] {
    let mant_bits = ft.mantissa_bits as u32;
    let bias = ft.bias();

    // Subnormal: biased exponent = 0, mantissa encodes value directly
    // value = mantissa / 2^mant_bits * 2^(1-bias)
    // → mantissa = value / 2^(1-bias) * 2^mant_bits
    let scale = f64::exp2((mant_bits as i32 + bias - 1) as f64);
    let scaled = abs_val * scale;

    let mant_int = round_to_nearest_even(scaled);

    if mant_int >= (1u64 << mant_bits) {
        // Rounded up to the smallest normal number
        let raw = (sign_bit << (ft.total_bits() as u32 - 1))
            | (1u64 << mant_bits); // biased_exp = 1, mant = 0
        return u64_to_le_bytes_masked(raw, ft.total_bits());
    }

    if mant_int == 0 {
        // Underflow to zero
        return encode_zero(sign_bit, ft);
    }

    let raw = (sign_bit << (ft.total_bits() as u32 - 1)) | mant_int;
    u64_to_le_bytes_masked(raw, ft.total_bits())
}

/// IEEE 754 round-to-nearest-even.
fn round_to_nearest_even(value: f64) -> u64 {
    let floor = value.floor();
    let frac = value - floor;
    let floor_u64 = floor as u64;

    if frac > 0.5 {
        floor_u64 + 1
    } else if frac < 0.5 {
        floor_u64
    } else {
        // Exactly 0.5 — round to even
        if floor_u64 % 2 == 0 {
            floor_u64
        } else {
            floor_u64 + 1
        }
    }
}

// ---------------------------------------------------------------------------
// Integer decode/encode
// ---------------------------------------------------------------------------

fn decode_signed_int(bits: [u8; 8], it: &IntType) -> i128 {
    match it.bits {
        64 => i64::from_le_bytes(bits) as i128,
        32 => i32::from_le_bytes([bits[0], bits[1], bits[2], bits[3]]) as i128,
        16 => i16::from_le_bytes([bits[0], bits[1]]) as i128,
        8 => (bits[0] as i8) as i128,
        4 => {
            let nibble = bits[0] & 0x0F;
            let signed = if nibble & 0x08 != 0 {
                (nibble | 0xF0) as i8
            } else {
                nibble as i8
            };
            signed as i128
        }
        b => {
            // Generic: read b bits, sign-extend from bit (b-1)
            let raw = u64_from_le_bytes_masked(bits, b);
            let sign_bit = 1u64 << (b - 1);
            if raw & sign_bit != 0 {
                // Negative: sign-extend
                let mask = !((1u64 << b) - 1);
                (raw | mask) as i64 as i128
            } else {
                raw as i128
            }
        }
    }
}

fn decode_unsigned_int(bits: [u8; 8], it: &IntType) -> u128 {
    match it.bits {
        64 => u64::from_le_bytes(bits) as u128,
        32 => u32::from_le_bytes([bits[0], bits[1], bits[2], bits[3]]) as u128,
        16 => u16::from_le_bytes([bits[0], bits[1]]) as u128,
        8 => bits[0] as u128,
        4 => (bits[0] & 0x0F) as u128,
        b => u64_from_le_bytes_masked(bits, b) as u128,
    }
}

fn encode_signed_int(value: i128, it: &IntType) -> [u8; 8] {
    let clamped = clamp_signed(value, it);
    let mut bits = [0u8; 8];
    let bytes = (clamped as i64).to_le_bytes();
    let nbytes = ((it.bits as usize) + 7) / 8;
    bits[..nbytes.min(8)].copy_from_slice(&bytes[..nbytes.min(8)]);
    // Mask off bits beyond the type width for sub-byte types
    if it.bits < 8 {
        bits[0] &= (1u8 << it.bits) - 1;
    }
    bits
}

fn encode_unsigned_int(value: u128, it: &IntType) -> [u8; 8] {
    let clamped = clamp_unsigned(value, it);
    let mut bits = [0u8; 8];
    let bytes = (clamped as u64).to_le_bytes();
    let nbytes = ((it.bits as usize) + 7) / 8;
    bits[..nbytes.min(8)].copy_from_slice(&bytes[..nbytes.min(8)]);
    if it.bits < 8 {
        bits[0] &= (1u8 << it.bits) - 1;
    }
    bits
}

/// Clamp i128 to the range of a signed integer with `it.bits` bits.
fn clamp_signed(value: i128, it: &IntType) -> i128 {
    let min = -(1i128 << (it.bits - 1));
    let max = (1i128 << (it.bits - 1)) - 1;
    value.clamp(min, max)
}

/// Clamp u128 to the range of an unsigned integer with `it.bits` bits.
fn clamp_unsigned(value: u128, it: &IntType) -> u128 {
    let max = if it.bits >= 64 {
        u64::MAX as u128
    } else {
        (1u128 << it.bits) - 1
    };
    value.min(max)
}

/// Convert f64 to signed int with saturation (Rust `as` semantics).
fn float_to_signed_int(f: f64, it: &IntType) -> i128 {
    if f.is_nan() {
        return 0;
    }
    let min = -(1i128 << (it.bits - 1));
    let max = (1i128 << (it.bits - 1)) - 1;
    let truncated = f.trunc() as i128;
    truncated.clamp(min, max)
}

/// Convert f64 to unsigned int with saturation.
fn float_to_unsigned_int(f: f64, it: &IntType) -> u128 {
    if f.is_nan() || f < 0.0 {
        return 0;
    }
    let max = if it.bits >= 64 {
        u64::MAX as u128
    } else {
        (1u128 << it.bits) - 1
    };
    let truncated = f.trunc() as u128;
    truncated.min(max)
}

/// Truncate i128 to fit in a signed int (keep low bits).
fn truncate_signed(value: i128, it: &IntType) -> i128 {
    clamp_signed(value, it)
}

/// Truncate i128 to fit in an unsigned int.
fn truncate_unsigned(value: i128, it: &IntType) -> u128 {
    if value < 0 {
        0
    } else {
        clamp_unsigned(value as u128, it)
    }
}

// ---------------------------------------------------------------------------
// Bit-level helpers
// ---------------------------------------------------------------------------

/// Read up to 64 bits from LE bytes, masking to `total_bits`.
fn u64_from_le_bytes_masked(bits: [u8; 8], total_bits: u8) -> u64 {
    let raw = u64::from_le_bytes(bits);
    if total_bits >= 64 {
        raw
    } else {
        raw & ((1u64 << total_bits) - 1)
    }
}

/// Write a u64 value to LE bytes, only setting the low `total_bits` bits.
fn u64_to_le_bytes_masked(value: u64, total_bits: u8) -> [u8; 8] {
    let masked = if total_bits >= 64 {
        value
    } else {
        value & ((1u64 << total_bits) - 1)
    };
    let mut result = [0u8; 8];
    let nbytes = ((total_bits as usize) + 7) / 8;
    result[..nbytes].copy_from_slice(&masked.to_le_bytes()[..nbytes]);
    result
}

/// Decompose f64 into (fraction, exponent) where value = fraction * 2^exponent,
/// 0.5 <= fraction < 1.0 for nonzero values.
fn frexp_f64(value: f64) -> (f64, i32) {
    if value == 0.0 {
        return (0.0, 0);
    }
    let bits = value.to_bits();
    let biased_exp = ((bits >> 52) & 0x7FF) as i32;
    let mantissa_bits = bits & ((1u64 << 52) - 1);

    if biased_exp == 0 {
        // Subnormal — normalize
        let normalized = value * f64::exp2(64.0);
        let (frac, exp) = frexp_f64(normalized);
        return (frac, exp - 64);
    }

    let exp = biased_exp - 1022; // 1023 (bias) - 1 (because fraction is [0.5, 1.0))
    let frac_bits = (0x3FE0_0000_0000_0000u64) | mantissa_bits; // exponent = -1
    let frac = f64::from_bits(frac_bits);

    (frac, exp)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use half::{bf16, f16};
    use float8::{F8E4M3, F8E5M2};

    /// Assert two NumericScalars have identical bits.
    macro_rules! assert_bits {
        ($actual:expr, $expected:expr, $msg:expr) => {
            assert_eq!(
                $actual.as_le_bytes(),
                $expected.as_le_bytes(),
                "{}: actual {:?} != expected {:?}",
                $msg, $actual, $expected
            );
        };
    }

    // ===================================================================
    // Float → Float: verify against native Rust / half crate conversions
    // ===================================================================

    #[test]
    fn f64_to_f32() {
        let values: &[f64] = &[
            0.0, -0.0, 1.0, -1.0, 0.5, 3.14,
            f64::INFINITY, f64::NEG_INFINITY, f64::NAN,
            1e-45, 1e38, -1e38,
            // Values that require rounding in f32
            1.0000001192092896, // 1.0 + f32::EPSILON
            std::f64::consts::PI,
        ];
        for &v in values {
            let src = NumericScalar::from_f64(v);
            let dst = src.cast_to(NumericDType::F32);
            let expected = NumericScalar::from_f32(v as f32);
            assert_bits!(dst, expected, format!("f64({v}) → f32"));
        }
    }

    #[test]
    fn f32_to_f64() {
        let values: &[f32] = &[
            0.0, -0.0, 1.0, -1.0, 3.14, f32::INFINITY, f32::NEG_INFINITY,
            f32::NAN, f32::MIN_POSITIVE, f32::MAX, f32::MIN,
            f32::EPSILON, 1e-38,
        ];
        for &v in values {
            let src = NumericScalar::from_f32(v);
            let dst = src.cast_to(NumericDType::F64);
            let expected = NumericScalar::from_f64(v as f64);
            assert_bits!(dst, expected, format!("f32({v}) → f64"));
        }
    }

    #[test]
    fn f32_to_bf16() {
        let values: &[f32] = &[
            0.0, -0.0, 1.0, -1.0, 3.14, 0.5, 0.1,
            f32::INFINITY, f32::NEG_INFINITY, f32::NAN,
            f32::MAX, f32::MIN, f32::MIN_POSITIVE,
            // Round-to-nearest-even edge cases
            1.0, 1.5, 2.5, 3.5, 4.5,
            // BF16 has 7-bit mantissa, so precision loss starts early
            256.0, 256.5, 257.0,
        ];
        for &v in values {
            let src = NumericScalar::from_f32(v);
            let dst = src.cast_to(NumericDType::BF16);
            let expected = NumericScalar::from_bf16(bf16::from_f32(v));
            assert_bits!(dst, expected, format!("f32({v}) → bf16"));
        }
    }

    #[test]
    fn bf16_to_f32() {
        let values: &[f32] = &[0.0, 1.0, -1.0, 3.14, 100.0, -0.5];
        for &v in values {
            let bf = bf16::from_f32(v);
            let src = NumericScalar::from_bf16(bf);
            let dst = src.cast_to(NumericDType::F32);
            let expected = NumericScalar::from_f32(bf.to_f32());
            assert_bits!(dst, expected, format!("bf16({bf}) → f32"));
        }
    }

    #[test]
    fn f32_to_f16() {
        let values: &[f32] = &[
            0.0, -0.0, 1.0, -1.0, 3.14, 0.5, 0.1,
            f32::INFINITY, f32::NEG_INFINITY, f32::NAN,
            // F16 max is ~65504
            65504.0, 65536.0, // overflow
            // Subnormal region
            5.96e-8, 1e-7,
            // Round-to-nearest-even
            1.0, 1.5, 2.5, 3.5,
        ];
        for &v in values {
            let src = NumericScalar::from_f32(v);
            let dst = src.cast_to(NumericDType::F16);
            let expected = NumericScalar::from_f16(f16::from_f32(v));
            assert_bits!(dst, expected, format!("f32({v}) → f16"));
        }
    }

    #[test]
    fn f16_to_f32() {
        let values: &[f32] = &[0.0, 1.0, -1.0, 0.5, 100.0, 65504.0];
        for &v in values {
            let hf = f16::from_f32(v);
            let src = NumericScalar::from_f16(hf);
            let dst = src.cast_to(NumericDType::F32);
            let expected = NumericScalar::from_f32(hf.to_f32());
            assert_bits!(dst, expected, format!("f16({hf}) → f32"));
        }
    }

    #[test]
    fn f32_to_f8e4m3fn() {
        let values: &[f32] = &[
            0.0, 1.0, -1.0, 0.5, 1.5, 3.0, -3.0,
            // F8E4M3FN max is 448
            448.0, 500.0, // overflow → 448 (FN saturates)
        ];
        for &v in values {
            let src = NumericScalar::from_f32(v);
            let dst = src.cast_to(NumericDType::F8E4M3FN);
            let expected = NumericScalar::from_f8e4m3fn(F8E4M3::from(v));
            assert_bits!(dst, expected, format!("f32({v}) → f8e4m3fn"));
        }
    }

    #[test]
    fn f32_to_f8e5m2() {
        // Note: the float8 crate saturates inf→max_finite for E5M2, which
        // contradicts IEEE semantics. Our software implementation follows
        // IEEE (inf→inf). We test non-special values against the crate,
        // and special values against our IEEE-correct encoding.
        let finite_values: &[f32] = &[0.0, 1.0, -1.0, 0.5, 2.0, 4.0];
        for &v in finite_values {
            let src = NumericScalar::from_f32(v);
            let dst = src.cast_to(NumericDType::F8E5M2);
            let expected = NumericScalar::from_f8e5m2(F8E5M2::from(v));
            assert_bits!(dst, expected, format!("f32({v}) → f8e5m2"));
        }

        // IEEE-correct: infinity encodes as exp=all-ones, mant=0
        let inf_src = NumericScalar::from_f32(f32::INFINITY);
        let inf_dst = inf_src.cast_to(NumericDType::F8E5M2);
        assert_eq!(inf_dst.bits[0], 0x7C, "f8e5m2 +inf should be 0x7C (IEEE)");

        let ninf_src = NumericScalar::from_f32(f32::NEG_INFINITY);
        let ninf_dst = ninf_src.cast_to(NumericDType::F8E5M2);
        assert_eq!(ninf_dst.bits[0], 0xFC, "f8e5m2 -inf should be 0xFC (IEEE)");
    }

    // ===================================================================
    // Float → Int: truncate toward zero, saturate on overflow
    // ===================================================================

    #[test]
    fn f32_to_i32() {
        let cases: &[(f32, i32)] = &[
            (0.0, 0), (1.0, 1), (-1.0, -1),
            (3.7, 3), (-3.7, -3),  // truncate toward zero
            (f32::NAN, 0),         // NaN → 0
            (f32::INFINITY, i32::MAX),
            (f32::NEG_INFINITY, i32::MIN),
            (2147483648.0, i32::MAX),  // overflow
            (-2147483904.0, i32::MIN), // underflow
        ];
        for &(f, expected) in cases {
            let src = NumericScalar::from_f32(f);
            let dst = src.cast_to(NumericDType::I32);
            let exp_scalar = NumericScalar::from_i32(expected);
            assert_bits!(dst, exp_scalar, format!("f32({f}) → i32"));
        }
    }

    #[test]
    fn f64_to_i64() {
        let cases: &[(f64, i64)] = &[
            (0.0, 0), (1.0, 1), (-1.0, -1),
            (42.9, 42), (-42.9, -42),
            (f64::NAN, 0),
            (f64::INFINITY, i64::MAX),
            (f64::NEG_INFINITY, i64::MIN),
        ];
        for &(f, expected) in cases {
            let src = NumericScalar::from_f64(f);
            let dst = src.cast_to(NumericDType::I64);
            let exp_scalar = NumericScalar::from_i64(expected);
            assert_bits!(dst, exp_scalar, format!("f64({f}) → i64"));
        }
    }

    #[test]
    fn f32_to_u8() {
        let cases: &[(f32, u8)] = &[
            (0.0, 0), (1.0, 1), (255.0, 255),
            (256.0, 255), // saturate
            (-1.0, 0),    // clamp negative → 0
            (f32::NAN, 0),
        ];
        for &(f, expected) in cases {
            let src = NumericScalar::from_f32(f);
            let dst = src.cast_to(NumericDType::U8);
            let exp_scalar = NumericScalar::from_u8(expected);
            assert_bits!(dst, exp_scalar, format!("f32({f}) → u8"));
        }
    }

    // ===================================================================
    // Int → Float: exact or round-to-nearest-even
    // ===================================================================

    #[test]
    fn i32_to_f32() {
        let values: &[i32] = &[
            0, 1, -1, i32::MAX, i32::MIN, 42, -42,
            // Large values that require rounding in f32
            16777217, // 2^24 + 1 — can't be represented exactly in f32
        ];
        for &v in values {
            let src = NumericScalar::from_i32(v);
            let dst = src.cast_to(NumericDType::F32);
            let expected = NumericScalar::from_f32(v as f32);
            assert_bits!(dst, expected, format!("i32({v}) → f32"));
        }
    }

    #[test]
    fn i64_to_f64() {
        let values: &[i64] = &[
            0, 1, -1, i64::MAX, i64::MIN, 42, -999999999,
        ];
        for &v in values {
            let src = NumericScalar::from_i64(v);
            let dst = src.cast_to(NumericDType::F64);
            let expected = NumericScalar::from_f64(v as f64);
            assert_bits!(dst, expected, format!("i64({v}) → f64"));
        }
    }

    #[test]
    fn u8_to_bf16() {
        for v in 0..=255u8 {
            let src = NumericScalar::from_u8(v);
            let dst = src.cast_to(NumericDType::BF16);
            let expected = NumericScalar::from_bf16(bf16::from_f32(v as f32));
            assert_bits!(dst, expected, format!("u8({v}) → bf16"));
        }
    }

    // ===================================================================
    // Int → Int
    // ===================================================================

    #[test]
    fn i32_to_i64() {
        let values: &[i32] = &[0, 1, -1, i32::MAX, i32::MIN, 42];
        for &v in values {
            let src = NumericScalar::from_i32(v);
            let dst = src.cast_to(NumericDType::I64);
            let expected = NumericScalar::from_i64(v as i64);
            assert_bits!(dst, expected, format!("i32({v}) → i64"));
        }
    }

    #[test]
    fn i64_to_i32_saturating() {
        let cases: &[(i64, i32)] = &[
            (0, 0), (1, 1), (-1, -1),
            (i32::MAX as i64, i32::MAX),
            (i32::MIN as i64, i32::MIN),
            (i64::MAX, i32::MAX),  // saturate
            (i64::MIN, i32::MIN),  // saturate
        ];
        for &(v, expected) in cases {
            let src = NumericScalar::from_i64(v);
            let dst = src.cast_to(NumericDType::I32);
            let exp_scalar = NumericScalar::from_i32(expected);
            assert_bits!(dst, exp_scalar, format!("i64({v}) → i32"));
        }
    }

    #[test]
    fn u32_to_i32_saturating() {
        let cases: &[(u32, i32)] = &[
            (0, 0), (1, 1),
            (i32::MAX as u32, i32::MAX),
            (u32::MAX, i32::MAX), // saturate
        ];
        for &(v, expected) in cases {
            let src = NumericScalar::from_u32(v);
            let dst = src.cast_to(NumericDType::I32);
            let exp_scalar = NumericScalar::from_i32(expected);
            assert_bits!(dst, exp_scalar, format!("u32({v}) → i32"));
        }
    }

    // ===================================================================
    // Bool conversions
    // ===================================================================

    #[test]
    fn bool_to_f32() {
        let t = NumericScalar::from_bool(true).cast_to(NumericDType::F32);
        let f = NumericScalar::from_bool(false).cast_to(NumericDType::F32);
        assert_bits!(t, NumericScalar::from_f32(1.0), "true → f32");
        assert_bits!(f, NumericScalar::from_f32(0.0), "false → f32");
    }

    #[test]
    fn f32_to_bool() {
        let cases: &[(f32, bool)] = &[
            (0.0, false), (1.0, true), (-1.0, true), (0.5, true),
            (f32::NAN, false), // NaN → false
        ];
        for &(v, expected) in cases {
            let src = NumericScalar::from_f32(v);
            let dst = src.cast_to(NumericDType::BOOL);
            let exp = NumericScalar::from_bool(expected);
            assert_bits!(dst, exp, format!("f32({v}) → bool"));
        }
    }

    #[test]
    fn i32_to_bool() {
        assert_eq!(NumericScalar::from_i32(0).cast_to(NumericDType::BOOL), NumericScalar::from_bool(false));
        assert_eq!(NumericScalar::from_i32(1).cast_to(NumericDType::BOOL), NumericScalar::from_bool(true));
        assert_eq!(NumericScalar::from_i32(-1).cast_to(NumericDType::BOOL), NumericScalar::from_bool(true));
    }

    // ===================================================================
    // Identity cast
    // ===================================================================

    #[test]
    fn identity_cast_preserves_bits() {
        let scalars = [
            NumericScalar::from_f32(3.14),
            NumericScalar::from_i64(i64::MAX),
            NumericScalar::from_bf16(bf16::from_f32(1.5)),
            NumericScalar::from_bool(true),
            NumericScalar::from_u8(42),
        ];
        for s in &scalars {
            let cast = s.cast_to(s.dtype());
            assert_eq!(*s, cast, "identity cast should preserve bits");
        }
    }

    // ===================================================================
    // Round-to-nearest-even edge cases (critical for IEEE compliance)
    // ===================================================================

    #[test]
    fn rte_f64_to_f32() {
        // These values are exactly at the midpoint between two f32 values.
        // IEEE RTE should round to the one with an even least significant bit.

        // 1.0 + 0.5 * f32::EPSILON = midpoint between 1.0 and 1.0+EPSILON
        // Should round to 1.0 (even)
        let mid_low = 1.0f64 + 0.5 * f32::EPSILON as f64;
        let src = NumericScalar::from_f64(mid_low);
        let dst = src.cast_to(NumericDType::F32);
        let expected = NumericScalar::from_f32(mid_low as f32);
        assert_bits!(dst, expected, "RTE: midpoint rounds to even (down)");

        // 1.0 + 1.5 * f32::EPSILON = midpoint between 1.0+EPSILON and 1.0+2*EPSILON
        // Should round to 1.0+2*EPSILON (even)
        let mid_high = 1.0f64 + 1.5 * f32::EPSILON as f64;
        let src = NumericScalar::from_f64(mid_high);
        let dst = src.cast_to(NumericDType::F32);
        let expected = NumericScalar::from_f32(mid_high as f32);
        assert_bits!(dst, expected, "RTE: midpoint rounds to even (up)");
    }

    // ===================================================================
    // Safe to_* methods via cast
    // ===================================================================

    #[test]
    fn to_f64_from_various() {
        assert_eq!(NumericScalar::from_f32(3.14f32).to_f64(), 3.14f32 as f64);
        assert_eq!(NumericScalar::from_i32(42).to_f64(), 42.0);
        assert_eq!(NumericScalar::from_bool(true).to_f64(), 1.0);
    }

    #[test]
    fn to_f32_from_various() {
        assert_eq!(NumericScalar::from_f64(3.14).to_f32(), 3.14f64 as f32);
        assert_eq!(NumericScalar::from_i32(42).to_f32(), 42.0f32);
    }

    #[test]
    fn to_i64_from_various() {
        assert_eq!(NumericScalar::from_f64(42.9).to_i64(), 42);
        assert_eq!(NumericScalar::from_i32(-5).to_i64(), -5);
        assert_eq!(NumericScalar::from_bool(true).to_i64(), 1);
    }

    #[test]
    fn is_nonzero_from_various() {
        assert!(!NumericScalar::from_f32(0.0).is_nonzero());
        assert!(NumericScalar::from_f32(1.0).is_nonzero());
        assert!(!NumericScalar::from_i32(0).is_nonzero());
        assert!(NumericScalar::from_i32(-1).is_nonzero());
        assert!(!NumericScalar::from_bool(false).is_nonzero());
        assert!(NumericScalar::from_bool(true).is_nonzero());
    }
}
