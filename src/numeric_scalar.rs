//! Dtype-erased numeric scalar with bit-level storage.
//!
//! [`NumericScalar`] stores any numeric value in `[u8; 8]` + [`NumericDType`].
//! Operations dispatch on dtype at runtime, with fast paths for standard types.
//!
//! [`NumericScalarView`] / [`NumericScalarViewMut`] provide zero-copy access
//! into byte buffers — the primitive used for tensor element access.

use std::fmt;

use arbitrary_int::{i4, u4};
use float8::{F8E4M3, F8E5M2};
use half::{bf16, f16};

use crate::numeric_dtype::{FloatType, IntType, NumericDType};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Owned numeric scalar — `[u8; 8]` buffer + dtype metadata.
///
/// 12 bytes total. Trivially cheap to copy. All operations take and return
/// owned values.
#[derive(Copy, Clone)]
pub struct NumericScalar {
    /// Little-endian bits. Only the first `dtype.total_bits() / 8` bytes
    /// (rounded up) are meaningful; the rest are zero-padded.
    bits: [u8; 8],
    dtype: NumericDType,
}

/// Immutable view into scalar data within a byte buffer.
#[derive(Copy, Clone)]
pub struct NumericScalarView<'a> {
    /// The byte buffer containing this scalar's data.
    pub data: &'a [u8],
    /// Bit offset from the start of `data` to this scalar's first bit.
    pub bit_offset: usize,
    pub dtype: NumericDType,
}

/// Mutable view into scalar data within a byte buffer.
pub struct NumericScalarViewMut<'a> {
    pub data: &'a mut [u8],
    pub bit_offset: usize,
    pub dtype: NumericDType,
}

// ---------------------------------------------------------------------------
// NumericScalar — constructors and accessors
// ---------------------------------------------------------------------------

impl NumericScalar {
    pub fn dtype(&self) -> NumericDType {
        self.dtype
    }

    /// View into this scalar's bits.
    pub fn view(&self) -> NumericScalarView<'_> {
        NumericScalarView {
            data: &self.bits,
            bit_offset: 0,
            dtype: self.dtype,
        }
    }

    /// Mutable view into this scalar's bits.
    pub fn view_mut(&mut self) -> NumericScalarViewMut<'_> {
        NumericScalarViewMut {
            data: &mut self.bits,
            bit_offset: 0,
            dtype: self.dtype,
        }
    }

    /// Zero value for the given dtype.
    pub fn zero(dtype: NumericDType) -> Self {
        Self {
            bits: [0u8; 8],
            dtype,
        }
    }

    /// Construct from an f64 value, converting to the target dtype.
    pub fn from_f64(value: f64, dtype: NumericDType) -> Self {
        let mut s = Self::zero(dtype);
        s.view_mut().write_f64(value);
        s
    }

    /// Construct from an f32 value, converting to the target dtype.
    pub fn from_f32(value: f32, dtype: NumericDType) -> Self {
        Self::from_f64(value as f64, dtype)
    }

    /// Construct from an i64 value, converting to the target dtype.
    pub fn from_i64(value: i64, dtype: NumericDType) -> Self {
        Self::from_f64(value as f64, dtype)
    }

    /// Construct from raw little-endian bytes + dtype. Only reads the bytes
    /// needed for the dtype; remaining bytes are ignored.
    pub fn from_le_bytes(bytes: &[u8], dtype: NumericDType) -> Self {
        let nbytes = dtype.bytes_per_element();
        let mut bits = [0u8; 8];
        bits[..nbytes].copy_from_slice(&bytes[..nbytes]);
        Self { bits, dtype }
    }

    /// Read as f64 (universal numeric conversion).
    pub fn to_f64(&self) -> f64 {
        self.view().to_f64()
    }

    /// Read as f32.
    pub fn to_f32(&self) -> f32 {
        self.to_f64() as f32
    }

    /// Read as i64.
    pub fn to_i64(&self) -> i64 {
        self.view().to_i64()
    }

    /// Read as bool (nonzero = true).
    pub fn is_nonzero(&self) -> bool {
        self.view().is_nonzero()
    }

    /// Cast to another dtype.
    pub fn cast_to(&self, target: NumericDType) -> Self {
        if self.dtype == target {
            return *self;
        }
        Self::from_f64(self.to_f64(), target)
    }

    /// Negative infinity for the given floating-point dtype.
    pub fn neg_infinity(dtype: NumericDType) -> Self {
        Self::from_f64(f64::NEG_INFINITY, dtype)
    }

    /// Positive infinity for the given floating-point dtype.
    pub fn infinity(dtype: NumericDType) -> Self {
        Self::from_f64(f64::INFINITY, dtype)
    }

    /// Access the raw little-endian bits.
    pub fn as_le_bytes(&self) -> &[u8] {
        &self.bits[..self.dtype.bytes_per_element()]
    }
}

// ---------------------------------------------------------------------------
// NumericScalarView — reading values
// ---------------------------------------------------------------------------

impl<'a> NumericScalarView<'a> {
    /// Read the value as f64.
    pub fn to_f64(&self) -> f64 {
        read_as_f64(self.data, self.bit_offset, self.dtype)
    }

    /// Read the value as i64.
    pub fn to_i64(&self) -> i64 {
        self.to_f64() as i64
    }

    /// Check if the value is nonzero.
    pub fn is_nonzero(&self) -> bool {
        // For most types, any nonzero bit pattern is nonzero.
        // Special case: floats where -0.0 should be considered zero.
        let f = self.to_f64();
        f != 0.0
    }

    /// Read into an owned NumericScalar.
    pub fn to_owned(&self) -> NumericScalar {
        let mut s = NumericScalar::zero(self.dtype);
        let nbytes = self.dtype.bytes_per_element();
        let byte_offset = self.bit_offset / 8;

        if self.bit_offset % 8 == 0 && self.dtype.total_bits() >= 8 {
            // Byte-aligned, byte-sized — fast path
            s.bits[..nbytes].copy_from_slice(&self.data[byte_offset..byte_offset + nbytes]);
        } else {
            // Sub-byte or unaligned — go through f64 roundtrip
            s = NumericScalar::from_f64(self.to_f64(), self.dtype);
        }
        s
    }
}

// ---------------------------------------------------------------------------
// NumericScalarViewMut — writing values
// ---------------------------------------------------------------------------

impl<'a> NumericScalarViewMut<'a> {
    /// Write an f64 value, converting to this view's dtype.
    pub fn write_f64(&mut self, value: f64) {
        write_f64_as(self.data, self.bit_offset, self.dtype, value);
    }

    /// Write from an owned NumericScalar. The scalar must have the same dtype.
    pub fn write_scalar(&mut self, scalar: &NumericScalar) {
        debug_assert_eq!(self.dtype, scalar.dtype);
        let nbytes = self.dtype.bytes_per_element();
        let byte_offset = self.bit_offset / 8;

        if self.bit_offset % 8 == 0 && self.dtype.total_bits() >= 8 {
            self.data[byte_offset..byte_offset + nbytes]
                .copy_from_slice(&scalar.bits[..nbytes]);
        } else {
            self.write_f64(scalar.to_f64());
        }
    }
}

// ---------------------------------------------------------------------------
// Arithmetic — binary ops
// ---------------------------------------------------------------------------

/// Apply a binary op, dispatching on dtype for fast native-type paths.
macro_rules! binary_op {
    ($name:ident, $op_f64:expr, $op_f32:expr, $op_i64:expr) => {
        impl NumericScalar {
            pub fn $name(&self, other: &Self) -> Self {
                debug_assert_eq!(self.dtype, other.dtype);
                let dtype = self.dtype;
                match dtype {
                    NumericDType::Float(ft) if ft == FloatType::F64 => {
                        let a = f64::from_le_bytes(self.bits);
                        let b = f64::from_le_bytes(other.bits);
                        let r: f64 = $op_f64(a, b);
                        Self { bits: r.to_le_bytes(), dtype }
                    }
                    NumericDType::Float(ft) if ft == FloatType::F32 => {
                        let a = f32::from_le_bytes(read4(&self.bits));
                        let b = f32::from_le_bytes(read4(&other.bits));
                        let r: f32 = $op_f32(a, b);
                        let mut bits = [0u8; 8];
                        bits[..4].copy_from_slice(&r.to_le_bytes());
                        Self { bits, dtype }
                    }
                    NumericDType::Float(ft) if ft == FloatType::BF16 => {
                        let a = bf16::from_le_bytes(read2(&self.bits));
                        let b = bf16::from_le_bytes(read2(&other.bits));
                        let r = $op_f32(a.to_f32(), b.to_f32());
                        let mut bits = [0u8; 8];
                        bits[..2].copy_from_slice(&bf16::from_f32(r).to_le_bytes());
                        Self { bits, dtype }
                    }
                    NumericDType::Float(ft) if ft == FloatType::F16 => {
                        let a = f16::from_le_bytes(read2(&self.bits));
                        let b = f16::from_le_bytes(read2(&other.bits));
                        let r = $op_f32(a.to_f32(), b.to_f32());
                        let mut bits = [0u8; 8];
                        bits[..2].copy_from_slice(&f16::from_f32(r).to_le_bytes());
                        Self { bits, dtype }
                    }
                    NumericDType::SignedInt(it) if it == IntType::BITS_64 => {
                        let a = i64::from_le_bytes(self.bits);
                        let b = i64::from_le_bytes(other.bits);
                        let r: i64 = $op_i64(a, b);
                        Self { bits: r.to_le_bytes(), dtype }
                    }
                    NumericDType::SignedInt(it) if it == IntType::BITS_32 => {
                        let a = i32::from_le_bytes(read4(&self.bits));
                        let b = i32::from_le_bytes(read4(&other.bits));
                        let r = $op_i64(a as i64, b as i64) as i32;
                        let mut bits = [0u8; 8];
                        bits[..4].copy_from_slice(&r.to_le_bytes());
                        Self { bits, dtype }
                    }
                    NumericDType::UnsignedInt(it) if it == IntType::BITS_64 => {
                        let a = u64::from_le_bytes(self.bits);
                        let b = u64::from_le_bytes(other.bits);
                        let r = $op_i64(a as i64, b as i64) as u64;
                        Self { bits: r.to_le_bytes(), dtype }
                    }
                    NumericDType::UnsignedInt(it) if it == IntType::BITS_32 => {
                        let a = u32::from_le_bytes(read4(&self.bits));
                        let b = u32::from_le_bytes(read4(&other.bits));
                        let r = $op_i64(a as i64, b as i64) as u32;
                        let mut bits = [0u8; 8];
                        bits[..4].copy_from_slice(&r.to_le_bytes());
                        Self { bits, dtype }
                    }
                    // Fallback: go through f64
                    _ => {
                        let a = self.to_f64();
                        let b = other.to_f64();
                        Self::from_f64($op_f64(a, b), dtype)
                    }
                }
            }
        }
    };
}

binary_op!(add, |a: f64, b: f64| a + b, |a: f32, b: f32| a + b, |a: i64, b: i64| a.wrapping_add(b));
binary_op!(sub, |a: f64, b: f64| a - b, |a: f32, b: f32| a - b, |a: i64, b: i64| a.wrapping_sub(b));
binary_op!(mul, |a: f64, b: f64| a * b, |a: f32, b: f32| a * b, |a: i64, b: i64| a.wrapping_mul(b));
binary_op!(div, |a: f64, b: f64| a / b, |a: f32, b: f32| a / b, |a: i64, b: i64| if b == 0 { 0 } else { a / b });
binary_op!(modulo, |a: f64, b: f64| a % b, |a: f32, b: f32| a % b, |a: i64, b: i64| if b == 0 { 0 } else { a % b });
binary_op!(pow, |a: f64, b: f64| a.powf(b), |a: f32, b: f32| a.powf(b), |a: i64, b: i64| a.wrapping_pow(b as u32));
binary_op!(scalar_max, |a: f64, b: f64| a.max(b), |a: f32, b: f32| a.max(b), |a: i64, b: i64| a.max(b));
binary_op!(scalar_min, |a: f64, b: f64| a.min(b), |a: f32, b: f32| a.min(b), |a: i64, b: i64| a.min(b));

// ---------------------------------------------------------------------------
// Arithmetic — unary ops
// ---------------------------------------------------------------------------

/// Apply a unary op with f64/f32/i64 fast paths.
macro_rules! unary_op {
    ($name:ident, $op_f64:expr, $op_f32:expr, $op_i64:expr) => {
        impl NumericScalar {
            pub fn $name(&self) -> Self {
                let dtype = self.dtype;
                match dtype {
                    NumericDType::Float(ft) if ft == FloatType::F64 => {
                        let v = f64::from_le_bytes(self.bits);
                        let r: f64 = $op_f64(v);
                        Self { bits: r.to_le_bytes(), dtype }
                    }
                    NumericDType::Float(ft) if ft == FloatType::F32 => {
                        let v = f32::from_le_bytes(read4(&self.bits));
                        let r: f32 = $op_f32(v);
                        let mut bits = [0u8; 8];
                        bits[..4].copy_from_slice(&r.to_le_bytes());
                        Self { bits, dtype }
                    }
                    _ => {
                        let v = self.to_f64();
                        Self::from_f64($op_f64(v), dtype)
                    }
                }
            }
        }
    };
}

unary_op!(neg, |v: f64| -v, |v: f32| -v, |v: i64| v.wrapping_neg());
unary_op!(abs, |v: f64| v.abs(), |v: f32| v.abs(), |v: i64| v.wrapping_abs());
unary_op!(exp, |v: f64| v.exp(), |v: f32| v.exp(), |_v: i64| panic!("exp on int"));
unary_op!(ln, |v: f64| v.ln(), |v: f32| v.ln(), |_v: i64| panic!("ln on int"));
unary_op!(sqrt, |v: f64| v.sqrt(), |v: f32| v.sqrt(), |_v: i64| panic!("sqrt on int"));
unary_op!(ceil, |v: f64| v.ceil(), |v: f32| v.ceil(), |v: i64| v);
unary_op!(floor, |v: f64| v.floor(), |v: f32| v.floor(), |v: i64| v);
unary_op!(round, |v: f64| v.round(), |v: f32| v.round(), |v: i64| v);
unary_op!(recip, |v: f64| 1.0 / v, |v: f32| 1.0 / v, |_v: i64| panic!("recip on int"));
unary_op!(tanh, |v: f64| v.tanh(), |v: f32| v.tanh(), |_v: i64| panic!("tanh on int"));

impl NumericScalar {
    /// Signum: -1, 0, or 1.
    pub fn sign(&self) -> Self {
        let v = self.to_f64();
        let s = if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            0.0
        };
        Self::from_f64(s, self.dtype)
    }

    /// Logical NOT (for bool), bitwise NOT for integers.
    pub fn not(&self) -> Self {
        match self.dtype {
            NumericDType::Bool => {
                Self::from_f64(if self.is_nonzero() { 0.0 } else { 1.0 }, self.dtype)
            }
            _ => {
                // Flip all bits in the value bytes
                let nbytes = self.dtype.bytes_per_element();
                let mut bits = self.bits;
                for b in &mut bits[..nbytes] {
                    *b = !*b;
                }
                Self {
                    bits,
                    dtype: self.dtype,
                }
            }
        }
    }

    /// Is the value NaN?
    pub fn is_nan(&self) -> Self {
        let v = self.to_f64();
        Self::from_f64(if v.is_nan() { 1.0 } else { 0.0 }, NumericDType::BOOL)
    }

    /// Trig operations.
    pub fn trig(&self, op: TrigOp) -> Self {
        let v = self.to_f64();
        let r = match op {
            TrigOp::Sin => v.sin(),
            TrigOp::Cos => v.cos(),
            TrigOp::Tan => v.tan(),
            TrigOp::Asin => v.asin(),
            TrigOp::Acos => v.acos(),
            TrigOp::Atan => v.atan(),
            TrigOp::Sinh => v.sinh(),
            TrigOp::Cosh => v.cosh(),
        };
        Self::from_f64(r, self.dtype)
    }

    /// Error function (erf).
    pub fn erf(&self) -> Self {
        let x = self.to_f64() as f32;
        // Horner approximation (Abramowitz & Stegun 7.1.26)
        let sign = x.signum();
        let x = x.abs();
        let t = 1.0 / (1.0 + 0.3275911 * x);
        let y = 1.0
            - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
                + 0.254829592)
                * t
                * (-x * x).exp();
        Self::from_f64((sign * y) as f64, self.dtype)
    }

    /// Clamp to minimum value.
    pub fn clamp_min(&self, min: f32) -> Self {
        let v = self.to_f64();
        Self::from_f64(v.max(min as f64), self.dtype)
    }

    /// Is infinity check.
    pub fn is_inf(&self, detect_positive: bool, detect_negative: bool) -> Self {
        let v = self.to_f64();
        let is = (detect_positive && v == f64::INFINITY)
            || (detect_negative && v == f64::NEG_INFINITY);
        Self::from_f64(if is { 1.0 } else { 0.0 }, NumericDType::BOOL)
    }

    /// Convert to bytes (for serialization). Returns the meaningful bytes only.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.as_le_bytes().to_vec()
    }
}

/// Trigonometric operation selector.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TrigOp {
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
}

// ---------------------------------------------------------------------------
// Conversion bridge: legacy NumericScalar ↔ new NumericScalar
// ---------------------------------------------------------------------------

use crate::migration::numeric_scalar::NumericScalar as LegacyNumericScalar;

impl NumericScalar {
    /// Convert from the legacy NumericScalar enum.
    pub fn from_legacy(legacy: &LegacyNumericScalar) -> Self {
        let dtype = match NumericDType::from_legacy(legacy.dtype()) {
            Some(dt) => dt,
            None => panic!("Cannot convert legacy dtype {:?} to NumericDType", legacy.dtype()),
        };
        Self::from_f64(legacy.to_f64(), dtype)
    }

    /// Convert to the legacy NumericScalar by going through f64.
    pub fn into_legacy(self) -> LegacyNumericScalar {
        let legacy_dtype = self.dtype.to_legacy();
        let f64_scalar = LegacyNumericScalar::F64(self.to_f64());
        f64_scalar.cast_to(legacy_dtype)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers: reading typed values from byte buffers
// ---------------------------------------------------------------------------

fn read2(bytes: &[u8]) -> [u8; 2] {
    [bytes[0], bytes[1]]
}

fn read4(bytes: &[u8]) -> [u8; 4] {
    [bytes[0], bytes[1], bytes[2], bytes[3]]
}

/// Read a scalar from a byte buffer as f64.
fn read_as_f64(data: &[u8], bit_offset: usize, dtype: NumericDType) -> f64 {
    let byte_off = bit_offset / 8;
    let bit_rem = bit_offset % 8;

    match dtype {
        NumericDType::Float(ft) if ft == FloatType::F64 => {
            f64::from_le_bytes(data[byte_off..byte_off + 8].try_into().unwrap())
        }
        NumericDType::Float(ft) if ft == FloatType::F32 => {
            f32::from_le_bytes(data[byte_off..byte_off + 4].try_into().unwrap()) as f64
        }
        NumericDType::Float(ft) if ft == FloatType::BF16 => {
            bf16::from_le_bytes(data[byte_off..byte_off + 2].try_into().unwrap()).to_f64()
        }
        NumericDType::Float(ft) if ft == FloatType::F16 => {
            f16::from_le_bytes(data[byte_off..byte_off + 2].try_into().unwrap()).to_f64()
        }
        NumericDType::Float(ft) if ft == FloatType::F8E4M3FN => {
            F8E4M3::from_bits(data[byte_off]).to_f64()
        }
        NumericDType::Float(ft) if ft == FloatType::F8E5M2 => {
            F8E5M2::from_bits(data[byte_off]).to_f64()
        }
        NumericDType::Float(ft) if ft == FloatType::F4E2M1 => {
            // 4-bit float: extract nibble at bit offset
            let byte = data[byte_off];
            let nibble = if bit_rem == 0 {
                byte & 0x0F
            } else {
                (byte >> bit_rem) & 0x0F
            };
            // F4E2M1: sign(1) + exp(2) + mantissa(1) = 4 bits
            // Interpret as a very small float. For now, simple linear mapping.
            // TODO: proper F4E2M1 decode
            let sign = if nibble & 0x08 != 0 { -1.0 } else { 1.0 };
            let exp = ((nibble >> 1) & 0x03) as i32;
            let mant = (nibble & 0x01) as f64;
            if exp == 0 {
                sign * (mant / 2.0) // subnormal
            } else {
                sign * (1.0 + mant) * (2.0f64).powi(exp - 1)
            }
        }
        NumericDType::SignedInt(it) if it == IntType::BITS_64 => {
            i64::from_le_bytes(data[byte_off..byte_off + 8].try_into().unwrap()) as f64
        }
        NumericDType::SignedInt(it) if it == IntType::BITS_32 => {
            i32::from_le_bytes(data[byte_off..byte_off + 4].try_into().unwrap()) as f64
        }
        NumericDType::SignedInt(it) if it == IntType::BITS_16 => {
            i16::from_le_bytes(data[byte_off..byte_off + 2].try_into().unwrap()) as f64
        }
        NumericDType::SignedInt(it) if it == IntType::BITS_8 => data[byte_off] as i8 as f64,
        NumericDType::SignedInt(it) if it == IntType::BITS_4 => {
            let byte = data[byte_off];
            let nibble = if bit_rem == 0 {
                byte & 0x0F
            } else {
                (byte >> bit_rem) & 0x0F
            };
            i4::new(nibble as i8).value() as f64
        }
        NumericDType::UnsignedInt(it) if it == IntType::BITS_64 => {
            u64::from_le_bytes(data[byte_off..byte_off + 8].try_into().unwrap()) as f64
        }
        NumericDType::UnsignedInt(it) if it == IntType::BITS_32 => {
            u32::from_le_bytes(data[byte_off..byte_off + 4].try_into().unwrap()) as f64
        }
        NumericDType::UnsignedInt(it) if it == IntType::BITS_16 => {
            u16::from_le_bytes(data[byte_off..byte_off + 2].try_into().unwrap()) as f64
        }
        NumericDType::UnsignedInt(it) if it == IntType::BITS_8 => data[byte_off] as f64,
        NumericDType::UnsignedInt(it) if it == IntType::BITS_4 => {
            let byte = data[byte_off];
            let nibble = if bit_rem == 0 {
                byte & 0x0F
            } else {
                (byte >> bit_rem) & 0x0F
            };
            u4::new(nibble).value() as f64
        }
        NumericDType::Bool => {
            let byte = data[byte_off];
            let bit = (byte >> bit_rem) & 1;
            bit as f64
        }
        // Fallback for non-standard types
        _ => {
            panic!(
                "read_as_f64: unsupported dtype {:?} (add a fast path or software decode)",
                dtype
            );
        }
    }
}

/// Write an f64 value into a byte buffer, converting to the target dtype.
fn write_f64_as(data: &mut [u8], bit_offset: usize, dtype: NumericDType, value: f64) {
    let byte_off = bit_offset / 8;
    let bit_rem = bit_offset % 8;

    match dtype {
        NumericDType::Float(ft) if ft == FloatType::F64 => {
            data[byte_off..byte_off + 8].copy_from_slice(&value.to_le_bytes());
        }
        NumericDType::Float(ft) if ft == FloatType::F32 => {
            data[byte_off..byte_off + 4].copy_from_slice(&(value as f32).to_le_bytes());
        }
        NumericDType::Float(ft) if ft == FloatType::BF16 => {
            data[byte_off..byte_off + 2]
                .copy_from_slice(&bf16::from_f64(value).to_le_bytes());
        }
        NumericDType::Float(ft) if ft == FloatType::F16 => {
            data[byte_off..byte_off + 2]
                .copy_from_slice(&f16::from_f64(value).to_le_bytes());
        }
        NumericDType::Float(ft) if ft == FloatType::F8E4M3FN => {
            data[byte_off] = F8E4M3::from(value as f32).to_bits();
        }
        NumericDType::Float(ft) if ft == FloatType::F8E5M2 => {
            data[byte_off] = F8E5M2::from(value as f32).to_bits();
        }
        NumericDType::SignedInt(it) if it == IntType::BITS_64 => {
            data[byte_off..byte_off + 8].copy_from_slice(&(value as i64).to_le_bytes());
        }
        NumericDType::SignedInt(it) if it == IntType::BITS_32 => {
            data[byte_off..byte_off + 4].copy_from_slice(&(value as i32).to_le_bytes());
        }
        NumericDType::SignedInt(it) if it == IntType::BITS_16 => {
            data[byte_off..byte_off + 2].copy_from_slice(&(value as i16).to_le_bytes());
        }
        NumericDType::SignedInt(it) if it == IntType::BITS_8 => {
            data[byte_off] = value as i8 as u8;
        }
        NumericDType::SignedInt(it) if it == IntType::BITS_4 => {
            let nibble = (i4::new(value as i8).value() as u8) & 0x0F;
            let mask = !(0x0F << bit_rem);
            data[byte_off] = (data[byte_off] & mask) | (nibble << bit_rem);
        }
        NumericDType::UnsignedInt(it) if it == IntType::BITS_64 => {
            data[byte_off..byte_off + 8].copy_from_slice(&(value as u64).to_le_bytes());
        }
        NumericDType::UnsignedInt(it) if it == IntType::BITS_32 => {
            data[byte_off..byte_off + 4].copy_from_slice(&(value as u32).to_le_bytes());
        }
        NumericDType::UnsignedInt(it) if it == IntType::BITS_16 => {
            data[byte_off..byte_off + 2].copy_from_slice(&(value as u16).to_le_bytes());
        }
        NumericDType::UnsignedInt(it) if it == IntType::BITS_8 => {
            data[byte_off] = value as u8;
        }
        NumericDType::UnsignedInt(it) if it == IntType::BITS_4 => {
            let nibble = u4::new(value as u8).value() & 0x0F;
            let mask = !(0x0F << bit_rem);
            data[byte_off] = (data[byte_off] & mask) | (nibble << bit_rem);
        }
        NumericDType::Bool => {
            let bit = if value != 0.0 { 1u8 } else { 0u8 };
            let mask = !(1u8 << bit_rem);
            data[byte_off] = (data[byte_off] & mask) | (bit << bit_rem);
        }
        _ => {
            panic!("write_f64_as: unsupported dtype {:?}", dtype);
        }
    }
}

// ---------------------------------------------------------------------------
// Debug / Display
// ---------------------------------------------------------------------------

impl fmt::Debug for NumericScalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NumericScalar({}: {:.6})", self.dtype, self.to_f64())
    }
}

impl fmt::Display for NumericScalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.dtype {
            NumericDType::Bool => {
                write!(f, "{}", self.is_nonzero())
            }
            NumericDType::SignedInt(_) | NumericDType::UnsignedInt(_) => {
                write!(f, "{}", self.to_i64())
            }
            _ => write!(f, "{}", self.to_f64()),
        }
    }
}

impl PartialEq for NumericScalar {
    fn eq(&self, other: &Self) -> bool {
        self.dtype == other.dtype && self.to_f64() == other.to_f64()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric_dtype::NumericDType;

    #[test]
    fn zero_values() {
        for dtype in [
            NumericDType::F64,
            NumericDType::F32,
            NumericDType::BF16,
            NumericDType::I32,
            NumericDType::U8,
            NumericDType::BOOL,
        ] {
            let z = NumericScalar::zero(dtype);
            assert_eq!(z.to_f64(), 0.0, "zero for {dtype}");
        }
    }

    #[test]
    fn from_f64_roundtrip() {
        let cases: &[(NumericDType, f64, f64)] = &[
            (NumericDType::F64, 3.14159, 3.14159),
            (NumericDType::F32, 3.14159, 3.141590118408203), // f32 precision
            (NumericDType::I32, 42.7, 42.0),                 // truncated
            (NumericDType::U8, 200.0, 200.0),
            (NumericDType::BOOL, 1.0, 1.0),
            (NumericDType::BOOL, 0.0, 0.0),
        ];
        for &(dtype, input, expected) in cases {
            let s = NumericScalar::from_f64(input, dtype);
            let actual = s.to_f64();
            assert!(
                (actual - expected).abs() < 1e-10,
                "from_f64 roundtrip for {dtype}: input={input}, expected={expected}, got={actual}"
            );
        }
    }

    #[test]
    fn bf16_roundtrip() {
        let s = NumericScalar::from_f64(1.5, NumericDType::BF16);
        assert!((s.to_f64() - 1.5).abs() < 0.01);
    }

    #[test]
    fn f16_roundtrip() {
        let s = NumericScalar::from_f64(1.5, NumericDType::F16);
        assert!((s.to_f64() - 1.5).abs() < 0.01);
    }

    #[test]
    fn add_f32() {
        let a = NumericScalar::from_f64(2.5, NumericDType::F32);
        let b = NumericScalar::from_f64(3.5, NumericDType::F32);
        let c = a.add(&b);
        assert!((c.to_f64() - 6.0).abs() < 1e-6);
    }

    #[test]
    fn add_i32() {
        let a = NumericScalar::from_f64(10.0, NumericDType::I32);
        let b = NumericScalar::from_f64(20.0, NumericDType::I32);
        let c = a.add(&b);
        assert_eq!(c.to_i64(), 30);
    }

    #[test]
    fn sub_mul_div() {
        let a = NumericScalar::from_f64(10.0, NumericDType::F64);
        let b = NumericScalar::from_f64(3.0, NumericDType::F64);
        assert!((a.sub(&b).to_f64() - 7.0).abs() < 1e-10);
        assert!((a.mul(&b).to_f64() - 30.0).abs() < 1e-10);
        assert!((a.div(&b).to_f64() - 10.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn neg_and_abs() {
        let a = NumericScalar::from_f64(-5.0, NumericDType::F32);
        assert!((a.neg().to_f64() - 5.0).abs() < 1e-6);
        assert!((a.abs().to_f64() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn cast_f32_to_i32() {
        let a = NumericScalar::from_f64(42.7, NumericDType::F32);
        let b = a.cast_to(NumericDType::I32);
        assert_eq!(b.to_i64(), 42);
        assert_eq!(b.dtype(), NumericDType::I32);
    }

    #[test]
    fn is_nonzero() {
        assert!(!NumericScalar::zero(NumericDType::F32).is_nonzero());
        assert!(NumericScalar::from_f64(1.0, NumericDType::F32).is_nonzero());
        assert!(!NumericScalar::zero(NumericDType::BOOL).is_nonzero());
        assert!(NumericScalar::from_f64(1.0, NumericDType::BOOL).is_nonzero());
    }

    #[test]
    fn view_read_from_buffer() {
        // Write two f32 values into a buffer, read them back via views
        let mut buf = [0u8; 8];
        buf[..4].copy_from_slice(&(1.5f32).to_le_bytes());
        buf[4..8].copy_from_slice(&(2.5f32).to_le_bytes());

        let v0 = NumericScalarView {
            data: &buf,
            bit_offset: 0,
            dtype: NumericDType::F32,
        };
        let v1 = NumericScalarView {
            data: &buf,
            bit_offset: 32,
            dtype: NumericDType::F32,
        };
        assert!((v0.to_f64() - 1.5).abs() < 1e-6);
        assert!((v1.to_f64() - 2.5).abs() < 1e-6);
    }

    #[test]
    fn view_mut_write_to_buffer() {
        let mut buf = [0u8; 4];
        let mut view = NumericScalarViewMut {
            data: &mut buf,
            bit_offset: 0,
            dtype: NumericDType::F32,
        };
        view.write_f64(3.14);
        let readback = f32::from_le_bytes(buf);
        assert!((readback - 3.14).abs() < 0.01);
    }

    #[test]
    fn bool_bit_level() {
        let mut buf = [0u8; 1];
        // Write true at bit 3
        let mut view = NumericScalarViewMut {
            data: &mut buf,
            bit_offset: 3,
            dtype: NumericDType::BOOL,
        };
        view.write_f64(1.0);
        assert_eq!(buf[0], 0b0000_1000);

        // Read it back
        let view = NumericScalarView {
            data: &buf,
            bit_offset: 3,
            dtype: NumericDType::BOOL,
        };
        assert_eq!(view.to_f64(), 1.0);
    }

    #[test]
    fn legacy_conversion_roundtrip() {
        let legacy = LegacyNumericScalar::F32(3.14);
        let new = NumericScalar::from_legacy(&legacy);
        assert!((new.to_f64() - 3.14).abs() < 0.01);
        assert_eq!(new.dtype(), NumericDType::F32);

        let back = new.into_legacy();
        assert!((back.to_f64() - 3.14).abs() < 0.01);
    }

    #[test]
    fn size_of_numeric_scalar() {
        // [u8; 8] + NumericDType (enum with u8 fields) = 11 bytes, no padding needed
        assert!(std::mem::size_of::<NumericScalar>() <= 12);
    }
}
