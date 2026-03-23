//! Structured numeric type system.
//!
//! Separates numeric type semantics from ONNX interop and storage layout.
//!
//! - [`NumericDType`] — what a numeric value IS (its mathematical interpretation)
//! - [`FloatType`] / [`IntType`] — parameterized by bit layout, not named variants
//! - [`ONNXDType`] — bridges ONNX's type system to ours (includes String)

use std::fmt;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// FloatSemantics
// ---------------------------------------------------------------------------

/// Distinguishes float formats that share the same exponent/mantissa bit widths
/// but differ in NaN, Inf, or negative-zero behavior.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FloatSemantics {
    /// Standard IEEE 754 — NaN, Inf, negative zero all present.
    /// Covers F64, F32, F16, BF16, F8E5M2.
    IEEE,
    /// Finite, no infinities, special NaN encoding.
    /// Covers F8E4M3FN, F4E2M1.
    FN,
    /// Finite, no negative zero, unsigned zero, different NaN.
    /// Covers F8E4M3FNUZ, F8E5M2FNUZ.
    FNUZ,
}

// ---------------------------------------------------------------------------
// FloatType
// ---------------------------------------------------------------------------

/// A floating-point type, parameterized by exponent and mantissa width.
///
/// Total bits = 1 (sign) + exponent_bits + mantissa_bits.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FloatType {
    pub exponent_bits: u8,
    pub mantissa_bits: u8,
    pub semantics: FloatSemantics,
}

impl FloatType {
    pub const F64: Self = FloatType {
        exponent_bits: 11,
        mantissa_bits: 52,
        semantics: FloatSemantics::IEEE,
    };
    pub const F32: Self = FloatType {
        exponent_bits: 8,
        mantissa_bits: 23,
        semantics: FloatSemantics::IEEE,
    };
    pub const F16: Self = FloatType {
        exponent_bits: 5,
        mantissa_bits: 10,
        semantics: FloatSemantics::IEEE,
    };
    pub const BF16: Self = FloatType {
        exponent_bits: 8,
        mantissa_bits: 7,
        semantics: FloatSemantics::IEEE,
    };
    pub const F8E4M3FN: Self = FloatType {
        exponent_bits: 4,
        mantissa_bits: 3,
        semantics: FloatSemantics::FN,
    };
    pub const F8E5M2: Self = FloatType {
        exponent_bits: 5,
        mantissa_bits: 2,
        semantics: FloatSemantics::IEEE,
    };
    pub const F4E2M1: Self = FloatType {
        exponent_bits: 2,
        mantissa_bits: 1,
        semantics: FloatSemantics::FN,
    };

    /// Total bits for one value: 1 (sign) + exponent + mantissa.
    pub const fn total_bits(&self) -> u8 {
        1 + self.exponent_bits + self.mantissa_bits
    }
}

impl fmt::Display for FloatType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Use well-known names for standard types
        match *self {
            Self::F64 => write!(f, "F64"),
            Self::F32 => write!(f, "F32"),
            Self::F16 => write!(f, "F16"),
            Self::BF16 => write!(f, "BF16"),
            Self::F8E4M3FN => write!(f, "F8E4M3FN"),
            Self::F8E5M2 => write!(f, "F8E5M2"),
            Self::F4E2M1 => write!(f, "F4E2M1"),
            _ => write!(
                f,
                "Float(e{}m{}{:?})",
                self.exponent_bits, self.mantissa_bits, self.semantics
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// IntType
// ---------------------------------------------------------------------------

/// An integer type, parameterized by bit width.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IntType {
    pub bits: u8,
}

impl IntType {
    pub const BITS_4: Self = IntType { bits: 4 };
    pub const BITS_8: Self = IntType { bits: 8 };
    pub const BITS_16: Self = IntType { bits: 16 };
    pub const BITS_32: Self = IntType { bits: 32 };
    pub const BITS_64: Self = IntType { bits: 64 };
}

impl fmt::Display for IntType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.bits)
    }
}

// ---------------------------------------------------------------------------
// NumericDType
// ---------------------------------------------------------------------------

/// What a numeric value IS — its mathematical interpretation.
///
/// Independent of storage layout. A `NumericDType` tells you how to interpret
/// bits, not how they're packed in a buffer.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NumericDType {
    Float(FloatType),
    SignedInt(IntType),
    UnsignedInt(IntType),
    Bool,
}

impl NumericDType {
    // Convenience constructors for common types.
    pub const F64: Self = NumericDType::Float(FloatType::F64);
    pub const F32: Self = NumericDType::Float(FloatType::F32);
    pub const F16: Self = NumericDType::Float(FloatType::F16);
    pub const BF16: Self = NumericDType::Float(FloatType::BF16);
    pub const F8E4M3FN: Self = NumericDType::Float(FloatType::F8E4M3FN);
    pub const F8E5M2: Self = NumericDType::Float(FloatType::F8E5M2);
    pub const F4E2M1: Self = NumericDType::Float(FloatType::F4E2M1);

    pub const I64: Self = NumericDType::SignedInt(IntType::BITS_64);
    pub const I32: Self = NumericDType::SignedInt(IntType::BITS_32);
    pub const I16: Self = NumericDType::SignedInt(IntType::BITS_16);
    pub const I8: Self = NumericDType::SignedInt(IntType::BITS_8);
    pub const I4: Self = NumericDType::SignedInt(IntType::BITS_4);

    pub const U64: Self = NumericDType::UnsignedInt(IntType::BITS_64);
    pub const U32: Self = NumericDType::UnsignedInt(IntType::BITS_32);
    pub const U16: Self = NumericDType::UnsignedInt(IntType::BITS_16);
    pub const U8: Self = NumericDType::UnsignedInt(IntType::BITS_8);
    pub const U4: Self = NumericDType::UnsignedInt(IntType::BITS_4);

    pub const BOOL: Self = NumericDType::Bool;

    /// Total bits for one element of this type.
    pub const fn total_bits(&self) -> u8 {
        match self {
            NumericDType::Float(ft) => ft.total_bits(),
            NumericDType::SignedInt(it) | NumericDType::UnsignedInt(it) => it.bits,
            NumericDType::Bool => 1,
        }
    }

    /// Bytes per element when stored one-per-slot (byte-aligned).
    /// Sub-byte types (U4, I4, Bool, F4E2M1) round up to 1 byte.
    pub const fn bytes_per_element(&self) -> usize {
        let bits = self.total_bits() as usize;
        (bits + 7) / 8
    }

    /// Whether this is a floating-point type.
    pub const fn is_float(&self) -> bool {
        matches!(self, NumericDType::Float(_))
    }

    /// Whether this is an integer type (signed or unsigned).
    pub const fn is_integer(&self) -> bool {
        matches!(
            self,
            NumericDType::SignedInt(_) | NumericDType::UnsignedInt(_)
        )
    }
}

impl fmt::Display for NumericDType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NumericDType::Float(ft) => write!(f, "{ft}"),
            NumericDType::SignedInt(it) => write!(f, "I{it}"),
            NumericDType::UnsignedInt(it) => write!(f, "U{it}"),
            NumericDType::Bool => write!(f, "Bool"),
        }
    }
}

// ---------------------------------------------------------------------------
// ONNXDType
// ---------------------------------------------------------------------------

/// What ONNX model files give us.
///
/// Bridges ONNX's type system to our internal numeric type system.
/// STRING exists only because ONNX insists on it — it is not a numeric type
/// and does not participate in the pool-managed tensor system.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ONNXDType {
    Numeric(NumericDType),
    String,
}

impl ONNXDType {
    /// Get the numeric dtype, or `None` if this is a string type.
    pub fn as_numeric(&self) -> Option<NumericDType> {
        match self {
            ONNXDType::Numeric(dt) => Some(*dt),
            ONNXDType::String => None,
        }
    }
}

impl fmt::Display for ONNXDType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ONNXDType::Numeric(dt) => write!(f, "{dt}"),
            ONNXDType::String => write!(f, "String"),
        }
    }
}

impl From<NumericDType> for ONNXDType {
    fn from(dt: NumericDType) -> Self {
        ONNXDType::Numeric(dt)
    }
}

// ---------------------------------------------------------------------------
// Conversion from legacy DType
// ---------------------------------------------------------------------------

use crate::dtype::DType;

impl NumericDType {
    /// Convert from the legacy `DType` enum.
    ///
    /// Returns `None` for `DType::STRING` and `DType::Packed(_)` which have
    /// no direct `NumericDType` representation.
    pub fn from_legacy(dt: DType) -> Option<Self> {
        Some(match dt {
            DType::F64 => Self::F64,
            DType::F32 => Self::F32,
            DType::BF16 => Self::BF16,
            DType::F16 => Self::F16,
            DType::F8E4M3FN => Self::F8E4M3FN,
            DType::F8E5M2 => Self::F8E5M2,
            DType::F4E2M1 => Self::F4E2M1,
            DType::I64 => Self::I64,
            DType::I32 => Self::I32,
            DType::I16 => Self::I16,
            DType::I8 => Self::I8,
            DType::I4 => Self::I4,
            DType::U64 => Self::U64,
            DType::U32 => Self::U32,
            DType::U16 => Self::U16,
            DType::U8 => Self::U8,
            DType::U4 => Self::U4,
            DType::BOOL => Self::BOOL,
            DType::STRING | DType::Packed(_) => return None,
        })
    }

    /// Convert back to the legacy `DType` enum.
    pub fn to_legacy(self) -> DType {
        match self {
            Self::F64 => DType::F64,
            Self::F32 => DType::F32,
            Self::BF16 => DType::BF16,
            Self::F16 => DType::F16,
            Self::F8E4M3FN => DType::F8E4M3FN,
            Self::F8E5M2 => DType::F8E5M2,
            Self::F4E2M1 => DType::F4E2M1,
            Self::I64 => DType::I64,
            Self::I32 => DType::I32,
            Self::I16 => DType::I16,
            Self::I8 => DType::I8,
            Self::I4 => DType::I4,
            Self::U64 => DType::U64,
            Self::U32 => DType::U32,
            Self::U16 => DType::U16,
            Self::U8 => DType::U8,
            Self::U4 => DType::U4,
            Self::BOOL => DType::BOOL,
            // Non-standard float types have no legacy equivalent; convert to
            // the closest named type by bit width, or fall back to F32.
            NumericDType::Float(ft) => match ft.total_bits() {
                64 => DType::F64,
                32 => DType::F32,
                16 => DType::F16,
                _ => DType::F32,
            },
            NumericDType::SignedInt(it) => match it.bits {
                64 => DType::I64,
                32 => DType::I32,
                16 => DType::I16,
                8 => DType::I8,
                4 => DType::I4,
                _ => DType::I32,
            },
            NumericDType::UnsignedInt(it) => match it.bits {
                64 => DType::U64,
                32 => DType::U32,
                16 => DType::U16,
                8 => DType::U8,
                4 => DType::U4,
                _ => DType::U32,
            },
        }
    }
}

impl ONNXDType {
    /// Convert from the legacy `DType` enum.
    pub fn from_legacy(dt: DType) -> Self {
        match dt {
            DType::STRING => ONNXDType::String,
            other => match NumericDType::from_legacy(other) {
                Some(ndt) => ONNXDType::Numeric(ndt),
                // Packed types don't have an ONNXDType equivalent.
                // This shouldn't happen in practice — Packed comes from GGUF, not ONNX.
                None => panic!("DType::Packed has no ONNXDType representation"),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_type_total_bits() {
        assert_eq!(FloatType::F64.total_bits(), 64);
        assert_eq!(FloatType::F32.total_bits(), 32);
        assert_eq!(FloatType::F16.total_bits(), 16);
        assert_eq!(FloatType::BF16.total_bits(), 16);
        assert_eq!(FloatType::F8E4M3FN.total_bits(), 8);
        assert_eq!(FloatType::F8E5M2.total_bits(), 8);
        assert_eq!(FloatType::F4E2M1.total_bits(), 4);
    }

    #[test]
    fn numeric_dtype_total_bits() {
        assert_eq!(NumericDType::F32.total_bits(), 32);
        assert_eq!(NumericDType::I64.total_bits(), 64);
        assert_eq!(NumericDType::U8.total_bits(), 8);
        assert_eq!(NumericDType::U4.total_bits(), 4);
        assert_eq!(NumericDType::BOOL.total_bits(), 1);
    }

    #[test]
    fn numeric_dtype_bytes_per_element() {
        assert_eq!(NumericDType::F64.bytes_per_element(), 8);
        assert_eq!(NumericDType::F32.bytes_per_element(), 4);
        assert_eq!(NumericDType::F16.bytes_per_element(), 2);
        assert_eq!(NumericDType::BF16.bytes_per_element(), 2);
        assert_eq!(NumericDType::I8.bytes_per_element(), 1);
        assert_eq!(NumericDType::U4.bytes_per_element(), 1);
        assert_eq!(NumericDType::BOOL.bytes_per_element(), 1);
    }

    #[test]
    fn display_standard_types() {
        assert_eq!(NumericDType::F32.to_string(), "F32");
        assert_eq!(NumericDType::BF16.to_string(), "BF16");
        assert_eq!(NumericDType::I64.to_string(), "I64");
        assert_eq!(NumericDType::U8.to_string(), "U8");
        assert_eq!(NumericDType::BOOL.to_string(), "Bool");
    }

    #[test]
    fn display_onnx_dtype() {
        assert_eq!(ONNXDType::Numeric(NumericDType::F32).to_string(), "F32");
        assert_eq!(ONNXDType::String.to_string(), "String");
    }

    #[test]
    fn legacy_roundtrip() {
        let types = [
            DType::F64,
            DType::F32,
            DType::BF16,
            DType::F16,
            DType::F8E4M3FN,
            DType::F8E5M2,
            DType::F4E2M1,
            DType::I64,
            DType::I32,
            DType::I16,
            DType::I8,
            DType::I4,
            DType::U64,
            DType::U32,
            DType::U16,
            DType::U8,
            DType::U4,
            DType::BOOL,
        ];
        for dt in types {
            let ndt = NumericDType::from_legacy(dt).unwrap();
            let back = ndt.to_legacy();
            assert_eq!(dt, back, "roundtrip failed for {dt}");
        }
    }

    #[test]
    fn legacy_string_and_packed_return_none() {
        assert!(NumericDType::from_legacy(DType::STRING).is_none());
        use crate::packed_format::PackedFormat;
        assert!(NumericDType::from_legacy(DType::Packed(PackedFormat::Q4_0)).is_none());
    }

    #[test]
    fn onnx_dtype_from_legacy() {
        assert_eq!(
            ONNXDType::from_legacy(DType::F32),
            ONNXDType::Numeric(NumericDType::F32)
        );
        assert_eq!(ONNXDType::from_legacy(DType::STRING), ONNXDType::String);
    }

    #[test]
    fn is_float_and_is_integer() {
        assert!(NumericDType::F32.is_float());
        assert!(!NumericDType::F32.is_integer());
        assert!(NumericDType::I32.is_integer());
        assert!(!NumericDType::I32.is_float());
        assert!(NumericDType::U8.is_integer());
        assert!(!NumericDType::BOOL.is_float());
        assert!(!NumericDType::BOOL.is_integer());
    }

    #[test]
    fn equality_by_structure() {
        // Two FloatTypes with same fields are equal, even if not a named constant
        let custom = FloatType {
            exponent_bits: 8,
            mantissa_bits: 23,
            semantics: FloatSemantics::IEEE,
        };
        assert_eq!(custom, FloatType::F32);
        assert_eq!(NumericDType::Float(custom), NumericDType::F32);
    }
}
