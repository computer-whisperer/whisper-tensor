//! Software cast operator: convert any [`NumericScalar`] to any [`NumericDType`].
//!
//! The implementation is fully software-defined — it interprets the
//! [`FloatType`] fields (exponent_bits, mantissa_bits, semantics) directly
//! rather than dispatching on known type names. Any `FloatType` within the
//! supported bounds ([`FloatType::is_supported`]) works correctly.
//!
//! Float conversions implement IEEE 754 round-to-nearest-even.
//!
//! All conversion logic operates on [`NumericScalarView`] / [`NumericScalarViewMut`]
//! and handles arbitrary bit offsets within byte slices. [`NumericScalar`] defers
//! to the view path.

use crate::numeric_dtype::{FloatType, IntType, NumericDType};

use super::{NumericScalar, NumericScalarView, NumericScalarViewMut};

// ---------------------------------------------------------------------------
// Raw bit access: read/write u64 from/to byte slices at arbitrary bit offsets
// ---------------------------------------------------------------------------

/// Read up to 64 bits from a byte slice starting at `bit_offset`.
/// Returns the value as a u64 with only the low `total_bits` bits set.
fn read_raw_bits(data: &[u8], bit_offset: usize, total_bits: u8) -> u64 {
    if total_bits == 0 {
        return 0;
    }

    let byte_off = bit_offset / 8;
    let bit_shift = (bit_offset % 8) as u32;

    // Read up to 9 bytes (64 bits + up to 7 bit shift can span 9 bytes)
    let bytes_needed = ((bit_shift as usize + total_bits as usize) + 7) / 8;
    let mut buf = [0u8; 9];
    let available = data.len().saturating_sub(byte_off);
    let to_copy = bytes_needed.min(available).min(9);
    buf[..to_copy].copy_from_slice(&data[byte_off..byte_off + to_copy]);

    // Assemble as a wide integer and shift
    // For simplicity, read as u128 from the 9 bytes, shift, mask
    let mut wide: u128 = 0;
    for (i, &b) in buf.iter().enumerate() {
        wide |= (b as u128) << (i * 8);
    }
    let shifted = (wide >> bit_shift) as u64;

    if total_bits >= 64 {
        shifted
    } else {
        shifted & ((1u64 << total_bits) - 1)
    }
}

/// Write up to 64 bits into a byte slice at `bit_offset`.
/// Only modifies the `total_bits` bits starting at `bit_offset`; other bits
/// in affected bytes are preserved.
fn write_raw_bits(data: &mut [u8], bit_offset: usize, total_bits: u8, value: u64) {
    if total_bits == 0 {
        return;
    }

    let byte_off = bit_offset / 8;
    let bit_shift = (bit_offset % 8) as u32;

    let masked_value = if total_bits >= 64 {
        value
    } else {
        value & ((1u64 << total_bits) - 1)
    };

    let bytes_needed = ((bit_shift as usize + total_bits as usize) + 7) / 8;
    let available = data.len().saturating_sub(byte_off);
    let to_touch = bytes_needed.min(available).min(9);

    // Read existing bytes into a wide buffer
    let mut buf = [0u8; 9];
    buf[..to_touch].copy_from_slice(&data[byte_off..byte_off + to_touch]);

    let mut wide: u128 = 0;
    for (i, &b) in buf.iter().enumerate() {
        wide |= (b as u128) << (i * 8);
    }

    // Create a mask for the bits we're writing
    let bit_mask: u128 = if total_bits >= 64 {
        (u64::MAX as u128) << bit_shift
    } else {
        (((1u128 << total_bits) - 1) << bit_shift)
    };

    // Clear target bits, set new value
    wide = (wide & !bit_mask) | ((masked_value as u128) << bit_shift);

    // Write back
    for i in 0..to_touch {
        data[byte_off + i] = (wide >> (i * 8)) as u8;
    }
}

// ---------------------------------------------------------------------------
// NumericScalarView — conversion methods
// ---------------------------------------------------------------------------

impl<'a> NumericScalarView<'a> {
    /// Read the raw bits of this scalar as a u64.
    pub fn read_raw(&self) -> u64 {
        read_raw_bits(self.data, self.bit_offset, self.dtype.total_bits())
    }

    /// Cast this viewed value to a different dtype, returning an owned scalar.
    pub fn cast_to(&self, target: NumericDType) -> NumericScalar {
        if self.dtype == target {
            return self.to_owned_scalar();
        }
        let raw = self.read_raw();
        let intermediate = decode_to_intermediate(raw, self.dtype);
        encode_from_intermediate(intermediate, target)
    }

    /// Read as an owned NumericScalar with the same dtype.
    pub fn to_owned_scalar(&self) -> NumericScalar {
        let raw = self.read_raw();
        let mut bits = [0u8; 8];
        let nbytes = self.dtype.bytes_per_element();
        bits[..nbytes].copy_from_slice(&raw.to_le_bytes()[..nbytes]);
        NumericScalar {
            bits,
            dtype: self.dtype,
        }
    }

    /// Convert to f64 regardless of source dtype.
    pub fn to_f64(&self) -> f64 {
        let raw = self.read_raw();
        match self.dtype {
            NumericDType::Float(ft) => software_float_decode(raw, &ft),
            NumericDType::SignedInt(it) => decode_signed_int(raw, &it) as f64,
            NumericDType::UnsignedInt(it) => decode_unsigned_int(raw, &it) as f64,
            NumericDType::Bool => if raw != 0 { 1.0 } else { 0.0 },
        }
    }

    /// Convert to f32 regardless of source dtype.
    pub fn to_f32(&self) -> f32 {
        self.to_f64() as f32
    }

    /// Convert to i64 regardless of source dtype.
    pub fn to_i64(&self) -> i64 {
        let raw = self.read_raw();
        match self.dtype {
            NumericDType::Float(ft) => {
                let f = software_float_decode(raw, &ft);
                float_to_signed_int(f, &IntType::BITS_64) as i64
            }
            NumericDType::SignedInt(it) => decode_signed_int(raw, &it) as i64,
            NumericDType::UnsignedInt(it) => {
                let u = decode_unsigned_int(raw, &it);
                u.min(i64::MAX as u128) as i64
            }
            NumericDType::Bool => if raw != 0 { 1 } else { 0 },
        }
    }

    /// Convert to bool (nonzero = true).
    pub fn is_nonzero(&self) -> bool {
        let raw = self.read_raw();
        match self.dtype {
            NumericDType::Float(ft) => {
                let f = software_float_decode(raw, &ft);
                f != 0.0 && !f.is_nan()
            }
            _ => raw != 0,
        }
    }
}

// ---------------------------------------------------------------------------
// NumericScalarViewMut — write methods
// ---------------------------------------------------------------------------

impl<'a> NumericScalarViewMut<'a> {
    /// Write the raw bits of a scalar value at this view's offset.
    /// The scalar's dtype must match this view's dtype.
    pub fn write_scalar(&mut self, scalar: &NumericScalar) {
        debug_assert_eq!(self.dtype, scalar.dtype);
        let raw = if scalar.dtype.total_bits() >= 64 {
            u64::from_le_bytes(scalar.bits)
        } else {
            u64::from_le_bytes(scalar.bits) & ((1u64 << scalar.dtype.total_bits()) - 1)
        };
        write_raw_bits(self.data, self.bit_offset, self.dtype.total_bits(), raw);
    }

    /// Cast a scalar to this view's dtype and write it.
    pub fn write_cast(&mut self, scalar: &NumericScalar) {
        let cast = scalar.cast_to(self.dtype);
        self.write_scalar(&cast);
    }

    /// Write an f64 value, converting to this view's dtype.
    pub fn write_f64(&mut self, value: f64) {
        let scalar = NumericScalar::from_f64(value).cast_to(self.dtype);
        self.write_scalar(&scalar);
    }

    /// Write raw bits directly.
    pub fn write_raw(&mut self, value: u64) {
        write_raw_bits(self.data, self.bit_offset, self.dtype.total_bits(), value);
    }
}

// ---------------------------------------------------------------------------
// NumericScalar — defers to view for all conversions
// ---------------------------------------------------------------------------

impl NumericScalar {
    /// Cast this scalar to a different dtype.
    pub fn cast_to(&self, target: NumericDType) -> NumericScalar {
        self.view().cast_to(target)
    }

    /// Convert to f64 regardless of source dtype.
    pub fn to_f64(&self) -> f64 {
        self.view().to_f64()
    }

    /// Convert to f32 regardless of source dtype.
    pub fn to_f32(&self) -> f32 {
        self.view().to_f32()
    }

    /// Convert to i64 regardless of source dtype.
    pub fn to_i64(&self) -> i64 {
        self.view().to_i64()
    }

    /// Convert to bool (nonzero = true).
    pub fn is_nonzero(&self) -> bool {
        self.view().is_nonzero()
    }
}

// ---------------------------------------------------------------------------
// Software cast internals
// ---------------------------------------------------------------------------

/// Universal intermediate value. Either an f64 (for float sources) or
/// i128 (for integer sources — wide enough for any 64-bit int without loss).
#[derive(Debug, Clone, Copy)]
enum Intermediate {
    Float(f64),
    Int(i128),
    Bool(bool),
}

/// Decode raw bits + dtype into the universal intermediate.
fn decode_to_intermediate(raw: u64, dtype: NumericDType) -> Intermediate {
    match dtype {
        NumericDType::Float(ft) => Intermediate::Float(software_float_decode(raw, &ft)),
        NumericDType::SignedInt(it) => Intermediate::Int(decode_signed_int(raw, &it)),
        NumericDType::UnsignedInt(it) => Intermediate::Int(decode_unsigned_int(raw, &it) as i128),
        NumericDType::Bool => Intermediate::Bool(raw != 0),
    }
}

/// Encode the universal intermediate into a target dtype.
fn encode_from_intermediate(value: Intermediate, dtype: NumericDType) -> NumericScalar {
    let raw: u64 = match dtype {
        NumericDType::Float(ft) => {
            let f = match value {
                Intermediate::Float(f) => f,
                Intermediate::Int(i) => i as f64,
                Intermediate::Bool(b) => if b { 1.0 } else { 0.0 },
            };
            software_float_encode(f, &ft)
        }
        NumericDType::SignedInt(it) => {
            let i = match value {
                Intermediate::Float(f) => float_to_signed_int(f, &it),
                Intermediate::Int(i) => clamp_signed(i, &it),
                Intermediate::Bool(b) => if b { 1 } else { 0 },
            };
            encode_signed_int(i, &it)
        }
        NumericDType::UnsignedInt(it) => {
            let u = match value {
                Intermediate::Float(f) => float_to_unsigned_int(f, &it),
                Intermediate::Int(i) => if i < 0 { 0 } else { clamp_unsigned(i as u128, &it) },
                Intermediate::Bool(b) => if b { 1 } else { 0 },
            };
            encode_unsigned_int(u, &it)
        }
        NumericDType::Bool => {
            let b = match value {
                Intermediate::Float(f) => f != 0.0 && !f.is_nan(),
                Intermediate::Int(i) => i != 0,
                Intermediate::Bool(b) => b,
            };
            b as u64
        }
    };

    let mut bits = [0u8; 8];
    let nbytes = dtype.bytes_per_element();
    bits[..nbytes].copy_from_slice(&raw.to_le_bytes()[..nbytes]);
    NumericScalar { bits, dtype }
}

// ---------------------------------------------------------------------------
// Software float decode: raw u64 bits → f64
// ---------------------------------------------------------------------------

/// Decode a float value from raw bits according to the FloatType spec.
///
/// See [`FloatType`] docs for the encoding rules driven by `has_infinity`
/// and `has_nan`.
fn software_float_decode(raw: u64, ft: &FloatType) -> f64 {
    let mant_bits = ft.mantissa_bits as u32;
    let exp_mask = (1u64 << ft.exponent_bits as u32) - 1;
    let mant_mask = (1u64 << mant_bits) - 1;

    let sign_bit = (raw >> (ft.total_bits() as u32 - 1)) & 1;
    let biased_exp = ((raw >> mant_bits) & exp_mask) as u32;
    let raw_mant = raw & mant_mask;

    let bias = ft.bias();
    let sign = if sign_bit == 1 { -1.0f64 } else { 1.0f64 };
    let max_exp = ft.max_biased_exponent();

    if biased_exp == max_exp {
        // Max exponent row — check for special values
        if ft.has_infinity && raw_mant == 0 {
            return sign * f64::INFINITY;
        }
        if ft.has_nan {
            let is_nan = if ft.has_infinity {
                // IEEE: all nonzero mantissa at max exp are NaN
                raw_mant != 0
            } else {
                // FN: only mant=all-ones is NaN
                raw_mant == mant_mask
            };
            if is_nan {
                return f64::NAN;
            }
        }
        // Normal number at max exponent
        let mant_val = 1.0 + (raw_mant as f64) / ((1u64 << mant_bits) as f64);
        sign * mant_val * f64::exp2((biased_exp as i32 - bias) as f64)
    } else if biased_exp == 0 {
        // Zero or subnormal
        if raw_mant == 0 {
            sign * 0.0
        } else {
            let mant_val = (raw_mant as f64) / ((1u64 << mant_bits) as f64);
            sign * mant_val * f64::exp2((1 - bias) as f64)
        }
    } else {
        // Normal number
        let mant_val = 1.0 + (raw_mant as f64) / ((1u64 << mant_bits) as f64);
        sign * mant_val * f64::exp2((biased_exp as i32 - bias) as f64)
    }
}

// ---------------------------------------------------------------------------
// Software float encode: f64 → raw u64 bits
// ---------------------------------------------------------------------------

/// Encode an f64 value into raw bits for the target FloatType.
fn software_float_encode(value: f64, ft: &FloatType) -> u64 {
    let mant_bits = ft.mantissa_bits as u32;
    let max_exp = ft.max_biased_exponent();

    if value.is_nan() {
        if ft.has_nan {
            return encode_nan_raw(ft);
        } else {
            // Format has no NaN — encode as zero
            return encode_zero_raw(0, ft);
        }
    }

    let sign_bit: u64 = if value.is_sign_negative() { 1 } else { 0 };
    let abs_val = value.abs();

    if abs_val == 0.0 {
        return encode_zero_raw(sign_bit, ft);
    }

    if value.is_infinite() {
        return encode_infinity_raw(sign_bit, ft);
    }

    let (frac, raw_exp) = frexp_f64(abs_val);
    let true_exp = raw_exp - 1;
    let bias = ft.bias();
    let min_normal_exp = 1 - bias;
    // For formats with reserved max exponent (has_infinity or has_nan),
    // max_exp-1 is the highest usable biased exponent for normal numbers.
    // For formats with no reserved patterns, max_exp itself is usable.
    let max_usable_biased = if ft.has_infinity || ft.has_nan {
        max_exp as i32 - 1
    } else {
        max_exp as i32
    };
    let max_normal_exp = max_usable_biased - bias;

    if true_exp > max_normal_exp {
        return encode_overflow_raw(sign_bit, ft);
    }

    if true_exp >= min_normal_exp {
        encode_normal_raw(sign_bit, frac * 2.0 - 1.0, true_exp, ft)
    } else {
        encode_subnormal_raw(sign_bit, abs_val, ft)
    }
}

/// Encode NaN. Returns canonical quiet NaN for the format.
/// Panics if `!ft.has_nan` (format has no NaN representation).
fn encode_nan_raw(ft: &FloatType) -> u64 {
    assert!(ft.has_nan, "cannot encode NaN in a format without NaN");
    let mant_bits = ft.mantissa_bits as u32;
    let max_exp = ft.max_biased_exponent() as u64;
    if ft.has_infinity {
        // IEEE: quiet NaN = max exp + MSB of mantissa set
        (max_exp << mant_bits) | (1u64 << (mant_bits - 1))
    } else {
        // FN: NaN = max exp + all-ones mantissa
        let mant_mask = (1u64 << mant_bits) - 1;
        (max_exp << mant_bits) | mant_mask
    }
}

fn encode_zero_raw(sign_bit: u64, ft: &FloatType) -> u64 {
    // Negative zero always exists in FloatType formats.
    sign_bit << (ft.total_bits() as u32 - 1)
}

fn encode_infinity_raw(sign_bit: u64, ft: &FloatType) -> u64 {
    if ft.has_infinity {
        let max_exp = ft.max_biased_exponent() as u64;
        (sign_bit << (ft.total_bits() as u32 - 1)) | (max_exp << ft.mantissa_bits as u32)
    } else {
        // No infinity — saturate to max finite value
        encode_max_finite_raw(sign_bit, ft)
    }
}

/// Encode the maximum finite value for this format.
fn encode_max_finite_raw(sign_bit: u64, ft: &FloatType) -> u64 {
    let mant_bits = ft.mantissa_bits as u32;
    let max_exp = ft.max_biased_exponent() as u64;
    let mant_mask = (1u64 << mant_bits) - 1;

    let (exp_val, mant_val) = if ft.has_infinity {
        // IEEE: max finite = (max_exp - 1) with all-ones mantissa
        (max_exp - 1, mant_mask)
    } else if ft.has_nan {
        // FN: max exp row is valid except mant=all-ones (NaN).
        // Max finite = max_exp with (all-ones - 1) mantissa.
        (max_exp, mant_mask - 1)
    } else {
        // No inf, no NaN: entire max exp row is valid.
        // Max finite = max_exp with all-ones mantissa.
        (max_exp, mant_mask)
    };

    (sign_bit << (ft.total_bits() as u32 - 1)) | (exp_val << mant_bits) | mant_val
}

fn encode_overflow_raw(sign_bit: u64, ft: &FloatType) -> u64 {
    if ft.has_infinity {
        encode_infinity_raw(sign_bit, ft)
    } else {
        encode_max_finite_raw(sign_bit, ft)
    }
}

fn encode_normal_raw(sign_bit: u64, frac_part: f64, true_exp: i32, ft: &FloatType) -> u64 {
    let mant_bits = ft.mantissa_bits as u32;
    let bias = ft.bias();
    let biased_exp = (true_exp + bias) as u64;

    let scale = (1u64 << mant_bits) as f64;
    let scaled = frac_part * scale;
    let mant_int = round_to_nearest_even(scaled);

    if mant_int >= (1u64 << mant_bits) {
        let new_exp = biased_exp + 1;
        let max_usable = if ft.has_infinity || ft.has_nan {
            ft.max_biased_exponent() as u64 - 1
        } else {
            ft.max_biased_exponent() as u64
        };
        if new_exp > max_usable {
            return encode_overflow_raw(sign_bit, ft);
        }
        return (sign_bit << (ft.total_bits() as u32 - 1)) | (new_exp << mant_bits);
    }

    (sign_bit << (ft.total_bits() as u32 - 1)) | (biased_exp << mant_bits) | mant_int
}

fn encode_subnormal_raw(sign_bit: u64, abs_val: f64, ft: &FloatType) -> u64 {
    let mant_bits = ft.mantissa_bits as u32;
    let bias = ft.bias();

    let scale = f64::exp2((mant_bits as i32 + bias - 1) as f64);
    let scaled = abs_val * scale;
    let mant_int = round_to_nearest_even(scaled);

    if mant_int >= (1u64 << mant_bits) {
        return (sign_bit << (ft.total_bits() as u32 - 1)) | (1u64 << mant_bits);
    }

    if mant_int == 0 {
        return encode_zero_raw(sign_bit, ft);
    }

    (sign_bit << (ft.total_bits() as u32 - 1)) | mant_int
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
        if floor_u64 % 2 == 0 { floor_u64 } else { floor_u64 + 1 }
    }
}

// ---------------------------------------------------------------------------
// Integer decode/encode (from raw u64 bits)
// ---------------------------------------------------------------------------

fn decode_signed_int(raw: u64, it: &IntType) -> i128 {
    let bits = it.bits;
    if bits >= 64 {
        return raw as i64 as i128;
    }
    let sign_bit = 1u64 << (bits - 1);
    if raw & sign_bit != 0 {
        let mask = !((1u64 << bits) - 1);
        (raw | mask) as i64 as i128
    } else {
        raw as i128
    }
}

fn decode_unsigned_int(raw: u64, it: &IntType) -> u128 {
    if it.bits >= 64 {
        raw as u128
    } else {
        (raw & ((1u64 << it.bits) - 1)) as u128
    }
}

fn encode_signed_int(value: i128, it: &IntType) -> u64 {
    let clamped = clamp_signed(value, it);
    let raw = clamped as i64 as u64;
    if it.bits >= 64 { raw } else { raw & ((1u64 << it.bits) - 1) }
}

fn encode_unsigned_int(value: u128, it: &IntType) -> u64 {
    let clamped = clamp_unsigned(value, it);
    clamped as u64
}

fn clamp_signed(value: i128, it: &IntType) -> i128 {
    let min = -(1i128 << (it.bits - 1));
    let max = (1i128 << (it.bits - 1)) - 1;
    value.clamp(min, max)
}

fn clamp_unsigned(value: u128, it: &IntType) -> u128 {
    let max = if it.bits >= 64 { u64::MAX as u128 } else { (1u128 << it.bits) - 1 };
    value.min(max)
}

fn float_to_signed_int(f: f64, it: &IntType) -> i128 {
    if f.is_nan() { return 0; }
    let min = -(1i128 << (it.bits - 1));
    let max = (1i128 << (it.bits - 1)) - 1;
    (f.trunc() as i128).clamp(min, max)
}

fn float_to_unsigned_int(f: f64, it: &IntType) -> u128 {
    if f.is_nan() || f < 0.0 { return 0; }
    let max = if it.bits >= 64 { u64::MAX as u128 } else { (1u128 << it.bits) - 1 };
    (f.trunc() as u128).min(max)
}

/// Decompose f64 into (fraction, exponent) where value = fraction * 2^exponent,
/// 0.5 <= fraction < 1.0 for nonzero values.
fn frexp_f64(value: f64) -> (f64, i32) {
    if value == 0.0 { return (0.0, 0); }
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use half::{bf16, f16};
    use float8::{F8E4M3, F8E5M2};

    macro_rules! assert_bits {
        ($actual:expr, $expected:expr, $msg:expr) => {
            assert_eq!(
                $actual.as_le_bytes(), $expected.as_le_bytes(),
                "{}: actual {:?} != expected {:?}", $msg, $actual, $expected
            );
        };
    }

    // ===================================================================
    // Raw bit access tests
    // ===================================================================

    #[test]
    fn read_write_raw_bits_byte_aligned() {
        let data = [0x12u8, 0x34, 0x56, 0x78];
        assert_eq!(read_raw_bits(&data, 0, 32), 0x78563412);
        assert_eq!(read_raw_bits(&data, 0, 16), 0x3412);
        assert_eq!(read_raw_bits(&data, 16, 16), 0x7856);
        assert_eq!(read_raw_bits(&data, 0, 8), 0x12);
        assert_eq!(read_raw_bits(&data, 8, 8), 0x34);
    }

    #[test]
    fn read_raw_bits_sub_byte() {
        let data = [0b1010_0110u8];
        // Read 4 bits at bit offset 0: low nibble = 0110 = 6
        assert_eq!(read_raw_bits(&data, 0, 4), 0b0110);
        // Read 4 bits at bit offset 4: high nibble = 1010 = 10
        assert_eq!(read_raw_bits(&data, 4, 4), 0b1010);
        // Read 1 bit at various offsets
        assert_eq!(read_raw_bits(&data, 0, 1), 0); // bit 0 = 0
        assert_eq!(read_raw_bits(&data, 1, 1), 1); // bit 1 = 1
        assert_eq!(read_raw_bits(&data, 5, 1), 1); // bit 5 = 1
    }

    #[test]
    fn write_raw_bits_sub_byte_preserves_neighbors() {
        let mut data = [0xFFu8];
        // Write 0 into bits 2..5 (4 bits starting at offset 2)
        write_raw_bits(&mut data, 2, 4, 0b0000);
        // bits: 11_0000_11 = 0b1100_0011 = 0xC3
        assert_eq!(data[0], 0b1100_0011);
    }

    #[test]
    fn write_raw_bits_spanning_bytes() {
        let mut data = [0u8; 4];
        // Write 0xABCD at bit offset 4 (spans bytes 0..2)
        write_raw_bits(&mut data, 4, 16, 0xABCD);
        // Byte 0: low nibble 0, high nibble = low nibble of 0xABCD = 0xD
        // Byte 1: 0xBC
        // Byte 2: low nibble = 0xA, high nibble = 0
        assert_eq!(data[0], 0xD0);
        assert_eq!(data[1], 0xBC);
        assert_eq!(data[2], 0x0A);
    }

    #[test]
    fn read_write_raw_roundtrip_at_various_offsets() {
        for bit_offset in [0, 1, 3, 4, 7, 8, 12, 16] {
            for total_bits in [1, 4, 8, 16, 32] {
                let mut data = [0u8; 16];
                let value = 0xDEAD_BEEFu64 & ((1u64 << total_bits.min(64)) - 1);
                write_raw_bits(&mut data, bit_offset, total_bits, value);
                let readback = read_raw_bits(&data, bit_offset, total_bits);
                assert_eq!(
                    readback, value,
                    "roundtrip failed: offset={bit_offset}, bits={total_bits}, value=0x{value:X}"
                );
            }
        }
    }

    // ===================================================================
    // View conversion tests
    // ===================================================================

    #[test]
    fn view_to_f64_byte_aligned_f32() {
        let mut buf = [0u8; 8];
        buf[..4].copy_from_slice(&3.14f32.to_le_bytes());
        let view = NumericScalarView { data: &buf, bit_offset: 0, dtype: NumericDType::F32 };
        let f = view.to_f64();
        assert!((f - 3.14f32 as f64).abs() < 1e-6);
    }

    #[test]
    fn view_to_f64_at_offset() {
        // Two f32 values packed contiguously
        let mut buf = [0u8; 8];
        buf[..4].copy_from_slice(&1.5f32.to_le_bytes());
        buf[4..8].copy_from_slice(&2.5f32.to_le_bytes());

        let v0 = NumericScalarView { data: &buf, bit_offset: 0, dtype: NumericDType::F32 };
        let v1 = NumericScalarView { data: &buf, bit_offset: 32, dtype: NumericDType::F32 };
        assert!((v0.to_f64() - 1.5).abs() < 1e-6);
        assert!((v1.to_f64() - 2.5).abs() < 1e-6);
    }

    #[test]
    fn view_cast_bf16_to_f32() {
        let bf = bf16::from_f32(3.14);
        let mut buf = [0u8; 2];
        buf.copy_from_slice(&bf.to_le_bytes());
        let view = NumericScalarView { data: &buf, bit_offset: 0, dtype: NumericDType::BF16 };
        let result = view.cast_to(NumericDType::F32);
        let expected = NumericScalar::from_f32(bf.to_f32());
        assert_bits!(result, expected, "view bf16 → f32");
    }

    #[test]
    fn view_mut_write_f64() {
        let mut buf = [0u8; 4];
        let mut view = NumericScalarViewMut { data: &mut buf, bit_offset: 0, dtype: NumericDType::F32 };
        view.write_f64(3.14);
        // Should have written 3.14 as f32
        let written = f32::from_le_bytes(buf);
        assert!((written - 3.14f32 as f32).abs() < 0.01);
    }

    #[test]
    fn view_mut_write_at_bit_offset_bool() {
        let mut buf = [0u8; 1];
        // Write true at bit 3
        let mut view = NumericScalarViewMut { data: &mut buf, bit_offset: 3, dtype: NumericDType::BOOL };
        let scalar = NumericScalar::from_bool(true);
        view.write_scalar(&scalar);
        assert_eq!(buf[0], 0b0000_1000);

        // Read it back
        let rview = NumericScalarView { data: &buf, bit_offset: 3, dtype: NumericDType::BOOL };
        assert!(rview.is_nonzero());
        assert!(!NumericScalarView { data: &buf, bit_offset: 0, dtype: NumericDType::BOOL }.is_nonzero());
    }

    #[test]
    fn view_mut_write_cast() {
        // Write an f64 value into a bf16 slot
        let mut buf = [0u8; 2];
        let mut view = NumericScalarViewMut { data: &mut buf, bit_offset: 0, dtype: NumericDType::BF16 };
        let src = NumericScalar::from_f64(3.14);
        view.write_cast(&src);

        // Read back and verify it matches bf16::from_f64(3.14)
        let rview = NumericScalarView { data: &buf, bit_offset: 0, dtype: NumericDType::BF16 };
        let result = rview.to_owned_scalar();
        let expected = NumericScalar::from_bf16(bf16::from_f64(3.14));
        assert_bits!(result, expected, "write_cast f64→bf16");
    }

    #[test]
    fn view_sub_byte_u4_packed() {
        // Pack two u4 values into one byte: low nibble = 5, high nibble = 12
        let mut buf = [0u8; 1];
        {
            let mut v0 = NumericScalarViewMut { data: &mut buf, bit_offset: 0, dtype: NumericDType::U4 };
            v0.write_scalar(&NumericScalar::from_u4(arbitrary_int::u4::new(5)));
        }
        {
            let mut v1 = NumericScalarViewMut { data: &mut buf, bit_offset: 4, dtype: NumericDType::U4 };
            v1.write_scalar(&NumericScalar::from_u4(arbitrary_int::u4::new(12)));
        }
        assert_eq!(buf[0], 0xC5); // high=12=0xC, low=5=0x5

        // Read back
        let r0 = NumericScalarView { data: &buf, bit_offset: 0, dtype: NumericDType::U4 };
        let r1 = NumericScalarView { data: &buf, bit_offset: 4, dtype: NumericDType::U4 };
        assert_eq!(r0.to_owned_scalar(), NumericScalar::from_u4(arbitrary_int::u4::new(5)));
        assert_eq!(r1.to_owned_scalar(), NumericScalar::from_u4(arbitrary_int::u4::new(12)));
    }

    // ===================================================================
    // NumericScalar cast tests (defer to view — same results as before)
    // ===================================================================

    #[test]
    fn f64_to_f32() {
        for &v in &[0.0f64, -0.0, 1.0, -1.0, 0.5, 3.14, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 1e-45, 1e38, -1e38, std::f64::consts::PI] {
            let dst = NumericScalar::from_f64(v).cast_to(NumericDType::F32);
            assert_bits!(dst, NumericScalar::from_f32(v as f32), format!("f64({v}) → f32"));
        }
    }

    #[test]
    fn f32_to_f64() {
        for &v in &[0.0f32, -0.0, 1.0, -1.0, 3.14, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, f32::MIN_POSITIVE, f32::MAX, f32::EPSILON] {
            let dst = NumericScalar::from_f32(v).cast_to(NumericDType::F64);
            assert_bits!(dst, NumericScalar::from_f64(v as f64), format!("f32({v}) → f64"));
        }
    }

    #[test]
    fn f32_to_bf16() {
        for &v in &[0.0f32, -0.0, 1.0, -1.0, 3.14, 0.5, 0.1, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, f32::MAX, f32::MIN, f32::MIN_POSITIVE, 1.5, 2.5, 3.5, 256.0, 256.5, 257.0] {
            let dst = NumericScalar::from_f32(v).cast_to(NumericDType::BF16);
            assert_bits!(dst, NumericScalar::from_bf16(bf16::from_f32(v)), format!("f32({v}) → bf16"));
        }
    }

    #[test]
    fn bf16_to_f32() {
        for &v in &[0.0f32, 1.0, -1.0, 3.14, 100.0, -0.5] {
            let bf = bf16::from_f32(v);
            let dst = NumericScalar::from_bf16(bf).cast_to(NumericDType::F32);
            assert_bits!(dst, NumericScalar::from_f32(bf.to_f32()), format!("bf16({bf}) → f32"));
        }
    }

    #[test]
    fn f32_to_f16() {
        for &v in &[0.0f32, -0.0, 1.0, -1.0, 3.14, 0.5, 0.1, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 65504.0, 65536.0, 5.96e-8, 1e-7, 1.5, 2.5, 3.5] {
            let dst = NumericScalar::from_f32(v).cast_to(NumericDType::F16);
            assert_bits!(dst, NumericScalar::from_f16(f16::from_f32(v)), format!("f32({v}) → f16"));
        }
    }

    #[test]
    fn f16_to_f32() {
        for &v in &[0.0f32, 1.0, -1.0, 0.5, 100.0, 65504.0] {
            let hf = f16::from_f32(v);
            let dst = NumericScalar::from_f16(hf).cast_to(NumericDType::F32);
            assert_bits!(dst, NumericScalar::from_f32(hf.to_f32()), format!("f16({hf}) → f32"));
        }
    }

    #[test]
    fn f32_to_f8e4m3fn() {
        for &v in &[0.0f32, 1.0, -1.0, 0.5, 1.5, 3.0, -3.0, 448.0, 500.0] {
            let dst = NumericScalar::from_f32(v).cast_to(NumericDType::F8E4M3FN);
            assert_bits!(dst, NumericScalar::from_f8e4m3fn(F8E4M3::from(v)), format!("f32({v}) → f8e4m3fn"));
        }
    }

    #[test]
    fn f32_to_f8e5m2_finite() {
        for &v in &[0.0f32, 1.0, -1.0, 0.5, 2.0, 4.0] {
            let dst = NumericScalar::from_f32(v).cast_to(NumericDType::F8E5M2);
            assert_bits!(dst, NumericScalar::from_f8e5m2(F8E5M2::from(v)), format!("f32({v}) → f8e5m2"));
        }
        // IEEE infinity encoding
        assert_eq!(NumericScalar::from_f32(f32::INFINITY).cast_to(NumericDType::F8E5M2).bits[0], 0x7C);
        assert_eq!(NumericScalar::from_f32(f32::NEG_INFINITY).cast_to(NumericDType::F8E5M2).bits[0], 0xFC);
    }

    #[test]
    fn f32_to_i32() {
        for &(f, expected) in &[(0.0f32, 0i32), (1.0, 1), (-1.0, -1), (3.7, 3), (-3.7, -3), (f32::NAN, 0), (f32::INFINITY, i32::MAX), (f32::NEG_INFINITY, i32::MIN)] {
            let dst = NumericScalar::from_f32(f).cast_to(NumericDType::I32);
            assert_bits!(dst, NumericScalar::from_i32(expected), format!("f32({f}) → i32"));
        }
    }

    #[test]
    fn f64_to_i64() {
        for &(f, expected) in &[(0.0f64, 0i64), (1.0, 1), (-1.0, -1), (42.9, 42), (-42.9, -42), (f64::NAN, 0), (f64::INFINITY, i64::MAX), (f64::NEG_INFINITY, i64::MIN)] {
            let dst = NumericScalar::from_f64(f).cast_to(NumericDType::I64);
            assert_bits!(dst, NumericScalar::from_i64(expected), format!("f64({f}) → i64"));
        }
    }

    #[test]
    fn i32_to_f32() {
        for &v in &[0i32, 1, -1, i32::MAX, i32::MIN, 42, -42, 16777217] {
            let dst = NumericScalar::from_i32(v).cast_to(NumericDType::F32);
            assert_bits!(dst, NumericScalar::from_f32(v as f32), format!("i32({v}) → f32"));
        }
    }

    #[test]
    fn i64_to_f64() {
        for &v in &[0i64, 1, -1, i64::MAX, i64::MIN, 42, -999999999] {
            let dst = NumericScalar::from_i64(v).cast_to(NumericDType::F64);
            assert_bits!(dst, NumericScalar::from_f64(v as f64), format!("i64({v}) → f64"));
        }
    }

    #[test]
    fn u8_to_bf16() {
        for v in 0..=255u8 {
            let dst = NumericScalar::from_u8(v).cast_to(NumericDType::BF16);
            assert_bits!(dst, NumericScalar::from_bf16(bf16::from_f32(v as f32)), format!("u8({v}) → bf16"));
        }
    }

    #[test]
    fn i32_to_i64() {
        for &v in &[0i32, 1, -1, i32::MAX, i32::MIN] {
            let dst = NumericScalar::from_i32(v).cast_to(NumericDType::I64);
            assert_bits!(dst, NumericScalar::from_i64(v as i64), format!("i32({v}) → i64"));
        }
    }

    #[test]
    fn i64_to_i32_saturating() {
        for &(v, expected) in &[(0i64, 0i32), (1, 1), (-1, -1), (i64::MAX, i32::MAX), (i64::MIN, i32::MIN)] {
            let dst = NumericScalar::from_i64(v).cast_to(NumericDType::I32);
            assert_bits!(dst, NumericScalar::from_i32(expected), format!("i64({v}) → i32"));
        }
    }

    #[test]
    fn bool_conversions() {
        assert_bits!(NumericScalar::from_bool(true).cast_to(NumericDType::F32), NumericScalar::from_f32(1.0), "true→f32");
        assert_bits!(NumericScalar::from_bool(false).cast_to(NumericDType::F32), NumericScalar::from_f32(0.0), "false→f32");
        assert_bits!(NumericScalar::from_f32(0.0).cast_to(NumericDType::BOOL), NumericScalar::from_bool(false), "0.0→bool");
        assert_bits!(NumericScalar::from_f32(1.0).cast_to(NumericDType::BOOL), NumericScalar::from_bool(true), "1.0→bool");
        assert_bits!(NumericScalar::from_i32(-1).cast_to(NumericDType::BOOL), NumericScalar::from_bool(true), "-1→bool");
    }

    #[test]
    fn identity_cast() {
        for s in [NumericScalar::from_f32(3.14), NumericScalar::from_i64(i64::MAX), NumericScalar::from_bool(true), NumericScalar::from_u8(42)] {
            assert_eq!(s.cast_to(s.dtype()), s);
        }
    }

    #[test]
    fn rte_f64_to_f32() {
        let mid_low = 1.0f64 + 0.5 * f32::EPSILON as f64;
        let mid_high = 1.0f64 + 1.5 * f32::EPSILON as f64;
        assert_bits!(NumericScalar::from_f64(mid_low).cast_to(NumericDType::F32), NumericScalar::from_f32(mid_low as f32), "RTE down");
        assert_bits!(NumericScalar::from_f64(mid_high).cast_to(NumericDType::F32), NumericScalar::from_f32(mid_high as f32), "RTE up");
    }

    #[test]
    fn to_f64_from_various() {
        assert_eq!(NumericScalar::from_f32(3.14f32).to_f64(), 3.14f32 as f64);
        assert_eq!(NumericScalar::from_i32(42).to_f64(), 42.0);
        assert_eq!(NumericScalar::from_bool(true).to_f64(), 1.0);
    }

    #[test]
    fn to_i64_from_various() {
        assert_eq!(NumericScalar::from_f64(42.9).to_i64(), 42);
        assert_eq!(NumericScalar::from_i32(-5).to_i64(), -5);
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
