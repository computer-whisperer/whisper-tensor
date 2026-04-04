//! Software conversion engine for numeric types.
//!
//! All conversion logic lives on the dtype types themselves:
//! - [`FloatType::decode_f64`] / [`FloatType::encode_f64`] — software float engine
//! - [`NumericDType::decode_to_f64`] / [`NumericDType::encode_from_f64`] — universal
//! - [`NumericDType::cast_raw`] — cast raw bits from one dtype to another
//!
//! These are pure functions: `(raw_bits, dtype) → value`. No scalar or tensor
//! wrapper needed.

use super::{FloatType, IntType, NumericDType};

// ===========================================================================
// FloatType: software decode/encode
// ===========================================================================

impl FloatType {
    /// Decode raw bits into an f64 value according to this float type's spec.
    ///
    /// The input `raw` should contain the float's bits in the low
    /// `total_bits()` positions. See [`FloatType`] docs for encoding rules.
    pub fn decode_f64(&self, raw: u64) -> f64 {
        debug_assert!(self.is_supported(), "FloatType {:?} is not supported", self);
        let mant_bits = self.mantissa_bits as u32;
        let exp_mask = (1u64 << self.exponent_bits as u32) - 1;
        let mant_mask = (1u64 << mant_bits) - 1;

        let sign_bit = (raw >> (self.total_bits() as u32 - 1)) & 1;
        let biased_exp = ((raw >> mant_bits) & exp_mask) as u32;
        let raw_mant = raw & mant_mask;

        let bias = self.bias();
        let sign = if sign_bit == 1 { -1.0f64 } else { 1.0f64 };
        let max_exp = self.max_biased_exponent();

        if biased_exp == max_exp {
            if self.has_infinity && raw_mant == 0 {
                return sign * f64::INFINITY;
            }
            if self.has_nan {
                let is_nan = if self.has_infinity {
                    raw_mant != 0
                } else {
                    raw_mant == mant_mask
                };
                if is_nan {
                    return f64::NAN;
                }
            }
            let mant_val = 1.0 + (raw_mant as f64) / ((1u64 << mant_bits) as f64);
            sign * mant_val * f64::exp2((biased_exp as i32 - bias) as f64)
        } else if biased_exp == 0 {
            if raw_mant == 0 {
                sign * 0.0
            } else {
                let mant_val = (raw_mant as f64) / ((1u64 << mant_bits) as f64);
                sign * mant_val * f64::exp2((1 - bias) as f64)
            }
        } else {
            let mant_val = 1.0 + (raw_mant as f64) / ((1u64 << mant_bits) as f64);
            sign * mant_val * f64::exp2((biased_exp as i32 - bias) as f64)
        }
    }

    /// Encode an f64 value into raw bits for this float type.
    /// Implements IEEE 754 round-to-nearest-even.
    pub fn encode_f64(&self, value: f64) -> u64 {
        debug_assert!(self.is_supported(), "FloatType {:?} is not supported", self);
        let max_exp = self.max_biased_exponent();

        if value.is_nan() {
            return if self.has_nan {
                self.encode_nan()
            } else {
                self.encode_zero(0)
            };
        }

        let sign_bit: u64 = if value.is_sign_negative() { 1 } else { 0 };
        let abs_val = value.abs();

        if abs_val == 0.0 {
            return self.encode_zero(sign_bit);
        }
        if value.is_infinite() {
            return self.encode_infinity(sign_bit);
        }

        let (frac, raw_exp) = frexp_f64(abs_val);
        let true_exp = raw_exp - 1;
        let bias = self.bias();
        let min_normal_exp = 1 - bias;
        let max_usable_biased = if self.has_infinity {
            max_exp as i32 - 1
        } else {
            max_exp as i32
        };
        let max_normal_exp = max_usable_biased - bias;

        if true_exp > max_normal_exp {
            return self.encode_overflow(sign_bit);
        }

        if true_exp >= min_normal_exp {
            self.encode_normal(sign_bit, frac * 2.0 - 1.0, true_exp)
        } else {
            self.encode_subnormal(sign_bit, abs_val)
        }
    }

    // -- Encode helpers (private) --

    fn encode_nan(&self) -> u64 {
        let mant_bits = self.mantissa_bits as u32;
        let max_exp = self.max_biased_exponent() as u64;
        if self.has_infinity {
            (max_exp << mant_bits) | (1u64 << (mant_bits - 1))
        } else {
            let mant_mask = (1u64 << mant_bits) - 1;
            (max_exp << mant_bits) | mant_mask
        }
    }

    fn encode_zero(&self, sign_bit: u64) -> u64 {
        sign_bit << (self.total_bits() as u32 - 1)
    }

    fn encode_infinity(&self, sign_bit: u64) -> u64 {
        if self.has_infinity {
            let max_exp = self.max_biased_exponent() as u64;
            (sign_bit << (self.total_bits() as u32 - 1)) | (max_exp << self.mantissa_bits as u32)
        } else {
            self.encode_max_finite(sign_bit)
        }
    }

    pub(crate) fn encode_max_finite(&self, sign_bit: u64) -> u64 {
        let mant_bits = self.mantissa_bits as u32;
        let max_exp = self.max_biased_exponent() as u64;
        let mant_mask = (1u64 << mant_bits) - 1;

        let (exp_val, mant_val) = if self.has_infinity {
            (max_exp - 1, mant_mask)
        } else if self.has_nan {
            (max_exp, mant_mask - 1)
        } else {
            (max_exp, mant_mask)
        };

        (sign_bit << (self.total_bits() as u32 - 1)) | (exp_val << mant_bits) | mant_val
    }

    fn encode_overflow(&self, sign_bit: u64) -> u64 {
        if self.has_infinity {
            self.encode_infinity(sign_bit)
        } else {
            self.encode_max_finite(sign_bit)
        }
    }

    fn encode_normal(&self, sign_bit: u64, frac_part: f64, true_exp: i32) -> u64 {
        let mant_bits = self.mantissa_bits as u32;
        let bias = self.bias();
        let biased_exp = (true_exp + bias) as u64;

        let scale = (1u64 << mant_bits) as f64;
        let scaled = frac_part * scale;
        let mant_int = round_to_nearest_even(scaled);

        if mant_int >= (1u64 << mant_bits) {
            let new_exp = biased_exp + 1;
            let max_biased = self.max_biased_exponent() as u64;
            if self.has_infinity && new_exp >= max_biased {
                return self.encode_overflow(sign_bit);
            }
            if new_exp > max_biased {
                return self.encode_overflow(sign_bit);
            }
            return (sign_bit << (self.total_bits() as u32 - 1)) | (new_exp << mant_bits);
        }

        // FN at max exponent: mant=all-ones is NaN, clamp
        let mant_mask = (1u64 << mant_bits) - 1;
        let final_mant = if self.has_nan
            && !self.has_infinity
            && biased_exp == self.max_biased_exponent() as u64
            && mant_int == mant_mask
        {
            mant_mask - 1
        } else {
            mant_int
        };

        (sign_bit << (self.total_bits() as u32 - 1)) | (biased_exp << mant_bits) | final_mant
    }

    fn encode_subnormal(&self, sign_bit: u64, abs_val: f64) -> u64 {
        let mant_bits = self.mantissa_bits as u32;
        let bias = self.bias();

        let scale = f64::exp2((mant_bits as i32 + bias - 1) as f64);
        let scaled = abs_val * scale;
        let mant_int = round_to_nearest_even(scaled);

        if mant_int >= (1u64 << mant_bits) {
            return (sign_bit << (self.total_bits() as u32 - 1)) | (1u64 << mant_bits);
        }
        if mant_int == 0 {
            return self.encode_zero(sign_bit);
        }

        (sign_bit << (self.total_bits() as u32 - 1)) | mant_int
    }
}

// ===========================================================================
// IntType: decode/encode helpers
// ===========================================================================

impl IntType {
    /// Decode raw bits as a signed integer, with sign extension.
    pub fn decode_signed(&self, raw: u64) -> i128 {
        if self.bits >= 64 {
            return raw as i64 as i128;
        }
        let sign_bit = 1u64 << (self.bits - 1);
        if raw & sign_bit != 0 {
            let mask = !((1u64 << self.bits) - 1);
            (raw | mask) as i64 as i128
        } else {
            raw as i128
        }
    }

    /// Decode raw bits as an unsigned integer.
    pub fn decode_unsigned(&self, raw: u64) -> u128 {
        if self.bits >= 64 {
            raw as u128
        } else {
            (raw & ((1u64 << self.bits) - 1)) as u128
        }
    }

    /// Encode a signed integer value, clamping to this type's range.
    pub fn encode_signed(&self, value: i128) -> u64 {
        let clamped = self.clamp_signed(value);
        let raw = clamped as i64 as u64;
        if self.bits >= 64 {
            raw
        } else {
            raw & ((1u64 << self.bits) - 1)
        }
    }

    /// Encode an unsigned integer value, clamping to this type's range.
    pub fn encode_unsigned(&self, value: u128) -> u64 {
        self.clamp_unsigned(value) as u64
    }

    /// Clamp a signed value to this type's range.
    pub fn clamp_signed(&self, value: i128) -> i128 {
        let min = -(1i128 << (self.bits - 1));
        let max = (1i128 << (self.bits - 1)) - 1;
        value.clamp(min, max)
    }

    /// Clamp an unsigned value to this type's range.
    pub fn clamp_unsigned(&self, value: u128) -> u128 {
        let max = if self.bits >= 64 {
            u64::MAX as u128
        } else {
            (1u128 << self.bits) - 1
        };
        value.min(max)
    }

    /// Convert f64 to signed integer with saturation.
    /// NaN → 0, ±Inf → min/max.
    pub fn float_to_signed(&self, f: f64) -> i128 {
        if f.is_nan() {
            return 0;
        }
        let min = -(1i128 << (self.bits - 1));
        let max = (1i128 << (self.bits - 1)) - 1;
        // Rust guarantees saturating float-to-int casts (since 1.45)
        (f.trunc() as i128).clamp(min, max)
    }

    /// Convert f64 to unsigned integer with saturation.
    /// NaN or negative → 0.
    pub fn float_to_unsigned(&self, f: f64) -> u128 {
        if f.is_nan() || f < 0.0 {
            return 0;
        }
        let max = if self.bits >= 64 {
            u64::MAX as u128
        } else {
            (1u128 << self.bits) - 1
        };
        (f.trunc() as u128).min(max)
    }
}

// ===========================================================================
// NumericDType: universal decode/encode/cast
// ===========================================================================

/// Universal intermediate value for type conversion.
/// f64 for floats, i128 for integers (wide enough for any 64-bit int).
#[derive(Debug, Clone, Copy)]
enum Intermediate {
    Float(f64),
    Int(i128),
    Bool(bool),
}

impl NumericDType {
    /// Decode raw bits as f64 according to this dtype.
    pub fn decode_to_f64(&self, raw: u64) -> f64 {
        match self {
            NumericDType::Float(ft) => ft.decode_f64(raw),
            NumericDType::SignedInt(it) => it.decode_signed(raw) as f64,
            NumericDType::UnsignedInt(it) => it.decode_unsigned(raw) as f64,
            NumericDType::Bool => {
                if raw != 0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    /// Encode an f64 value as raw bits in this dtype.
    pub fn encode_from_f64(&self, value: f64) -> u64 {
        match self {
            NumericDType::Float(ft) => ft.encode_f64(value),
            NumericDType::SignedInt(it) => it.encode_signed(it.float_to_signed(value)),
            NumericDType::UnsignedInt(it) => it.encode_unsigned(it.float_to_unsigned(value)),
            NumericDType::Bool => (value != 0.0 && !value.is_nan()) as u64,
        }
    }

    /// Cast raw bits from this dtype to a target dtype.
    /// Returns the raw bits in the target dtype's encoding.
    pub fn cast_raw(&self, raw: u64, target: NumericDType) -> u64 {
        if *self == target {
            return raw;
        }
        let intermediate = match self {
            NumericDType::Float(ft) => Intermediate::Float(ft.decode_f64(raw)),
            NumericDType::SignedInt(it) => Intermediate::Int(it.decode_signed(raw)),
            NumericDType::UnsignedInt(it) => Intermediate::Int(it.decode_unsigned(raw) as i128),
            NumericDType::Bool => Intermediate::Bool(raw != 0),
        };
        encode_intermediate(intermediate, target)
    }

    /// If the raw value decodes to ±inf for this dtype, replace it with
    /// ±max_finite. Non-float or non-infinite values pass through unchanged.
    pub fn saturate_inf(&self, raw: u64) -> u64 {
        match self {
            NumericDType::Float(ft) => {
                let val = ft.decode_f64(raw);
                if val.is_infinite() {
                    let sign_bit: u64 = if val.is_sign_negative() { 1 } else { 0 };
                    ft.encode_max_finite(sign_bit)
                } else {
                    raw
                }
            }
            _ => raw,
        }
    }
}

/// Encode an intermediate value into raw bits for a target dtype.
fn encode_intermediate(value: Intermediate, dtype: NumericDType) -> u64 {
    match dtype {
        NumericDType::Float(ft) => {
            let f = match value {
                Intermediate::Float(f) => f,
                Intermediate::Int(i) => i as f64,
                Intermediate::Bool(b) => {
                    if b {
                        1.0
                    } else {
                        0.0
                    }
                }
            };
            ft.encode_f64(f)
        }
        NumericDType::SignedInt(it) => {
            let i = match value {
                Intermediate::Float(f) => it.float_to_signed(f),
                Intermediate::Int(i) => it.clamp_signed(i),
                Intermediate::Bool(b) => {
                    if b {
                        1
                    } else {
                        0
                    }
                }
            };
            it.encode_signed(i)
        }
        NumericDType::UnsignedInt(it) => {
            let u = match value {
                Intermediate::Float(f) => it.float_to_unsigned(f),
                Intermediate::Int(i) => {
                    if i < 0 {
                        0
                    } else {
                        it.clamp_unsigned(i as u128)
                    }
                }
                Intermediate::Bool(b) => {
                    if b {
                        1
                    } else {
                        0
                    }
                }
            };
            it.encode_unsigned(u)
        }
        NumericDType::Bool => {
            let b = match value {
                Intermediate::Float(f) => f != 0.0 && !f.is_nan(),
                Intermediate::Int(i) => i != 0,
                Intermediate::Bool(b) => b,
            };
            b as u64
        }
    }
}

// ===========================================================================
// Internal helpers
// ===========================================================================

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
        if floor_u64.is_multiple_of(2) {
            floor_u64
        } else {
            floor_u64 + 1
        }
    }
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
        let normalized = value * f64::exp2(64.0);
        let (frac, exp) = frexp_f64(normalized);
        return (frac, exp - 64);
    }

    let exp = biased_exp - 1022;
    let frac_bits = 0x3FE0_0000_0000_0000u64 | mantissa_bits;
    (f64::from_bits(frac_bits), exp)
}
