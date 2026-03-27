//! Dtype-erased numeric scalar with bit-level storage.
//!
//! [`NumericScalar`] stores any numeric value in `[u8; 8]` + [`NumericDType`].
//! [`NumericScalarView`] / [`NumericScalarViewMut`] provide zero-copy access
//! into byte buffers at arbitrary bit offsets.

pub mod conversions;
pub mod core_conversions;

use crate::numeric_dtype::NumericDType;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Owned numeric scalar — `[u8; 8]` buffer + dtype metadata.
///
/// Stores raw little-endian bits for any numeric type up to 64 bits.
/// Only the first `dtype.bytes_per_element()` bytes are meaningful;
/// the rest are zero-padded.
#[derive(Copy, Clone, serde::Serialize, serde::Deserialize)]
pub struct NumericScalar {
    pub(crate) bits: [u8; 8],
    pub(crate) dtype: NumericDType,
}

/// Immutable view into scalar data within a byte buffer.
#[derive(Copy, Clone)]
pub struct NumericScalarView<'a> {
    pub data: &'a [u8],
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
// Basic accessors
// ---------------------------------------------------------------------------

impl NumericScalar {
    /// The dtype of this scalar.
    pub fn dtype(&self) -> NumericDType {
        self.dtype
    }

    /// Raw little-endian bits (full 8-byte buffer).
    pub fn raw_bits(&self) -> &[u8; 8] {
        &self.bits
    }

    /// The meaningful bytes for this dtype (little-endian).
    pub fn as_le_bytes(&self) -> &[u8] {
        &self.bits[..self.dtype.bytes_per_element()]
    }

    /// Zero value for the given dtype.
    pub fn zero(dtype: NumericDType) -> Self {
        Self {
            bits: [0u8; 8],
            dtype,
        }
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
}

// ---------------------------------------------------------------------------
// Scalar arithmetic — convenience wrappers around scalar_ops
// ---------------------------------------------------------------------------

impl NumericScalar {
    /// Construct from raw bits and dtype.
    pub fn from_raw_bits(raw: u64, dtype: NumericDType) -> Self {
        let mut bits = [0u8; 8];
        let n = dtype.bytes_per_element().min(8);
        bits[..n].copy_from_slice(&raw.to_le_bytes()[..n]);
        Self { bits, dtype }
    }

    /// Read the raw u64 bits of this scalar.
    pub fn raw(&self) -> u64 {
        self.view().read_raw()
    }

    /// Positive infinity for float types, max value for integer types.
    /// Useful as initial value for ReduceMin.
    pub fn max_sentinel(dtype: NumericDType) -> Self {
        match dtype {
            NumericDType::Float(ft) => Self::from_raw_bits(ft.encode_f64(f64::INFINITY), dtype),
            NumericDType::SignedInt(it) => {
                // e.g. i8 max = 127 = 2^(bits-1) - 1
                let max_val = if it.bits >= 64 { i64::MAX as i128 } else { (1i128 << (it.bits - 1)) - 1 };
                Self::from_raw_bits(it.encode_signed(max_val), dtype)
            }
            NumericDType::UnsignedInt(it) => {
                let max_val = if it.bits >= 64 { u64::MAX as u128 } else { (1u128 << it.bits) - 1 };
                Self::from_raw_bits(it.encode_unsigned(max_val), dtype)
            }
            NumericDType::Bool => Self::from_raw_bits(1, dtype),
        }
    }

    /// Negative infinity for float types, min value for integer types.
    /// Useful as initial value for ReduceMax.
    pub fn min_sentinel(dtype: NumericDType) -> Self {
        match dtype {
            NumericDType::Float(ft) => Self::from_raw_bits(ft.encode_f64(f64::NEG_INFINITY), dtype),
            NumericDType::SignedInt(it) => {
                // e.g. i8 min = -128 = -(2^(bits-1))
                let min_val = -(1i128 << (it.bits - 1));
                Self::from_raw_bits(it.encode_signed(min_val), dtype)
            }
            NumericDType::UnsignedInt(_) => Self::from_raw_bits(0, dtype),
            NumericDType::Bool => Self::from_raw_bits(0, dtype),
        }
    }

    /// `self + other` in the scalar's dtype.
    pub fn add(self, other: Self) -> Self {
        use crate::scalar_ops::add;
        let (a, b) = (self.raw(), other.raw());
        let r = match self.dtype {
            NumericDType::Float(ft) => add::float_add(a, b, &ft),
            NumericDType::SignedInt(it) => add::signed_add_wrapping(a, b, &it),
            NumericDType::UnsignedInt(it) => add::unsigned_add_wrapping(a, b, &it),
            NumericDType::Bool => a | b,
        };
        Self::from_raw_bits(r, self.dtype)
    }

    /// `self - other` in the scalar's dtype.
    pub fn sub(self, other: Self) -> Self {
        use crate::scalar_ops::sub;
        let (a, b) = (self.raw(), other.raw());
        let r = match self.dtype {
            NumericDType::Float(ft) => sub::float_sub(a, b, &ft),
            NumericDType::SignedInt(it) => sub::signed_sub_wrapping(a, b, &it),
            NumericDType::UnsignedInt(it) => sub::unsigned_sub_wrapping(a, b, &it),
            NumericDType::Bool => a & !b,
        };
        Self::from_raw_bits(r, self.dtype)
    }

    /// `self * other` in the scalar's dtype.
    pub fn mul(self, other: Self) -> Self {
        use crate::scalar_ops::mul;
        let (a, b) = (self.raw(), other.raw());
        let r = match self.dtype {
            NumericDType::Float(ft) => mul::float_mul(a, b, &ft),
            NumericDType::SignedInt(it) => mul::signed_mul_wrapping(a, b, &it),
            NumericDType::UnsignedInt(it) => mul::unsigned_mul_wrapping(a, b, &it),
            NumericDType::Bool => a & b,
        };
        Self::from_raw_bits(r, self.dtype)
    }

    /// `self / other` in the scalar's dtype.
    pub fn div(self, other: Self) -> Self {
        use crate::scalar_ops::div;
        let (a, b) = (self.raw(), other.raw());
        let r = match self.dtype {
            NumericDType::Float(ft) => div::float_div(a, b, &ft),
            NumericDType::SignedInt(it) => div::signed_div_wrapping(a, b, &it),
            NumericDType::UnsignedInt(it) => div::unsigned_div_wrapping(a, b, &it),
            NumericDType::Bool => a, // x / true = x
        };
        Self::from_raw_bits(r, self.dtype)
    }

    /// `max(self, other)` in the scalar's dtype.
    pub fn max(self, other: Self) -> Self {
        use crate::scalar_ops::max;
        let (a, b) = (self.raw(), other.raw());
        let r = match self.dtype {
            NumericDType::Float(ft) => max::float_max(a, b, &ft),
            NumericDType::SignedInt(it) => max::signed_max(a, b, &it),
            NumericDType::UnsignedInt(it) => max::unsigned_max(a, b, &it),
            NumericDType::Bool => a | b,
        };
        Self::from_raw_bits(r, self.dtype)
    }

    /// `min(self, other)` in the scalar's dtype.
    pub fn min(self, other: Self) -> Self {
        use crate::scalar_ops::min;
        let (a, b) = (self.raw(), other.raw());
        let r = match self.dtype {
            NumericDType::Float(ft) => min::float_min(a, b, &ft),
            NumericDType::SignedInt(it) => min::signed_min(a, b, &it),
            NumericDType::UnsignedInt(it) => min::unsigned_min(a, b, &it),
            NumericDType::Bool => a & b,
        };
        Self::from_raw_bits(r, self.dtype)
    }

    /// `self > other` — returns true/false as a Rust bool.
    pub fn gt(self, other: Self) -> bool {
        use crate::scalar_ops::cmp;
        let (a, b) = (self.raw(), other.raw());
        match self.dtype {
            NumericDType::Float(ft) => cmp::float_greater(a, b, &ft) != 0,
            NumericDType::SignedInt(it) => cmp::signed_greater(a, b, &it) != 0,
            NumericDType::UnsignedInt(it) => cmp::unsigned_greater(a, b, &it) != 0,
            NumericDType::Bool => a > b,
        }
    }

    /// `self < other` — returns true/false as a Rust bool.
    pub fn lt(self, other: Self) -> bool {
        use crate::scalar_ops::cmp;
        let (a, b) = (self.raw(), other.raw());
        match self.dtype {
            NumericDType::Float(ft) => cmp::float_less(a, b, &ft) != 0,
            NumericDType::SignedInt(it) => cmp::signed_less(a, b, &it) != 0,
            NumericDType::UnsignedInt(it) => cmp::unsigned_less(a, b, &it) != 0,
            NumericDType::Bool => a < b,
        }
    }
}

// ---------------------------------------------------------------------------
// Debug / Display
// ---------------------------------------------------------------------------

impl std::fmt::Debug for NumericScalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NumericScalar({}: 0x", self.dtype)?;
        for b in self.as_le_bytes().iter().rev() {
            write!(f, "{b:02x}")?;
        }
        write!(f, ")")
    }
}

impl std::fmt::Display for NumericScalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Use the software decode path for display
        match self.dtype {
            NumericDType::Float(_) => write!(f, "{}", self.to_f64()),
            NumericDType::SignedInt(_) => write!(f, "{}", self.to_i64()),
            NumericDType::UnsignedInt(_) => {
                // Read raw bits and display as unsigned
                let raw = self.view().read_raw();
                write!(f, "{raw}")
            }
            NumericDType::Bool => write!(f, "{}", self.is_nonzero()),
        }
    }
}

impl PartialEq for NumericScalar {
    fn eq(&self, other: &Self) -> bool {
        self.dtype == other.dtype && self.bits == other.bits
    }
}

impl Eq for NumericScalar {}

impl std::hash::Hash for NumericScalar {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.dtype.hash(state);
        self.bits.hash(state);
    }
}
