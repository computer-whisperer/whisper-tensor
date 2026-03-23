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
#[derive(Copy, Clone)]
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
