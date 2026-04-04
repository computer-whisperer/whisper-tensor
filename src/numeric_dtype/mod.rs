//! Structured numeric type system.
//!
//! Separates numeric type semantics from ONNX interop and storage layout.
//!
//! - [`NumericDType`] — what a numeric value IS (its mathematical interpretation)
//! - [`FloatType`] / [`IntType`] — parameterized by bit layout, not named variants
//! - [`ONNXDType`] — bridges ONNX's type system to ours (includes String)
//!
//! The [`conversions`] submodule provides the software conversion engine:
//! decode/encode raw bits, cast between dtypes — all as pure functions on
//! the dtype types themselves.

pub mod conversions;

use std::fmt;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// FloatType
// ---------------------------------------------------------------------------

/// An IEEE-like floating-point type, fully described by four properties.
///
/// # Bit layout
///
/// Every value is encoded as: `[sign: 1 bit] [exponent: exponent_bits] [mantissa: mantissa_bits]`
///
/// # Value interpretation
///
/// Given raw fields `(sign_bit, biased_exp, raw_mant)`:
///
/// **Normal numbers** (`0 < biased_exp < max_biased_exp`, plus max-exp
/// patterns not reserved for inf/NaN):
/// ```text
/// value = (-1)^sign * (1 + raw_mant / 2^mantissa_bits) * 2^(biased_exp - bias)
/// ```
///
/// **Subnormal numbers** (`biased_exp == 0, raw_mant != 0`):
/// ```text
/// value = (-1)^sign * (raw_mant / 2^mantissa_bits) * 2^(1 - bias)
/// ```
///
/// **Zero** (`biased_exp == 0, raw_mant == 0`): `±0.0` (both signs valid;
/// negative zero always exists).
///
/// **Bias**: always `2^(exponent_bits - 1) - 1` (standard IEEE formula).
///
/// # Max-exponent encoding (`biased_exp == max_biased_exp`)
///
/// The `has_infinity` and `has_nan` flags uniquely determine how the
/// max-exponent row is partitioned:
///
/// | `has_infinity` | `has_nan` | `mant == 0` | `0 < mant < all-ones` | `mant == all-ones` |
/// |----------------|-----------|-------------|-----------------------|--------------------|
/// | true           | true      | ±Infinity   | NaN                   | NaN                |
/// | true           | false     | ±Infinity   | normal                | normal             |
/// | false          | true      | normal      | normal                | NaN                |
/// | false          | false     | normal      | normal                | normal             |
///
/// When `has_nan=true && has_infinity=true` (IEEE): all nonzero mantissa
/// values at max exponent are NaN, `mant=0` is infinity.
///
/// When `has_nan=true && has_infinity=false` (FN): only `mant=all-ones`
/// at max exponent is NaN; all other max-exponent patterns (including
/// `mant=0`) are normal numbers.
///
/// When `has_nan=false`: the entire max-exponent row is normal numbers.
/// No NaN representation exists in the format.
///
/// # Formats not covered
///
/// FNUZ formats (non-standard bias, negative-zero-as-NaN) and signless
/// formats (E8M0) do not fit this decomposition and belong in separate
/// [`NumericDType`] arms if needed.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FloatType {
    pub exponent_bits: u8,
    pub mantissa_bits: u8,
    /// Whether `biased_exp=max, mant=0` encodes ±infinity.
    /// If false, that bit pattern is a normal number.
    pub has_infinity: bool,
    /// Whether NaN exists in this format. The NaN encoding is determined
    /// jointly with `has_infinity` — see the table in the struct docs.
    pub has_nan: bool,
}

impl FloatType {
    // -- Standard IEEE types --

    pub const F64: Self = FloatType {
        exponent_bits: 11,
        mantissa_bits: 52,
        has_infinity: true,
        has_nan: true,
    };
    pub const F32: Self = FloatType {
        exponent_bits: 8,
        mantissa_bits: 23,
        has_infinity: true,
        has_nan: true,
    };
    pub const F16: Self = FloatType {
        exponent_bits: 5,
        mantissa_bits: 10,
        has_infinity: true,
        has_nan: true,
    };
    pub const BF16: Self = FloatType {
        exponent_bits: 8,
        mantissa_bits: 7,
        has_infinity: true,
        has_nan: true,
    };
    pub const F8E5M2: Self = FloatType {
        exponent_bits: 5,
        mantissa_bits: 2,
        has_infinity: true,
        has_nan: true,
    };

    // -- FN types (no infinity, single NaN) --

    pub const F8E4M3FN: Self = FloatType {
        exponent_bits: 4,
        mantissa_bits: 3,
        has_infinity: false,
        has_nan: true,
    };

    // -- No-special-value types (no infinity, no NaN) --

    pub const F4E2M1: Self = FloatType {
        exponent_bits: 2,
        mantissa_bits: 1,
        has_infinity: false,
        has_nan: false,
    };
    pub const F6E3M2: Self = FloatType {
        exponent_bits: 3,
        mantissa_bits: 2,
        has_infinity: false,
        has_nan: false,
    };
    pub const F6E2M3: Self = FloatType {
        exponent_bits: 2,
        mantissa_bits: 3,
        has_infinity: false,
        has_nan: false,
    };

    // -- Bounds --

    /// Maximum exponent bits supported. Bounded by f64's 11-bit exponent,
    /// which we use as the intermediate representation for software arithmetic.
    pub const MAX_EXPONENT_BITS: u8 = 11;

    /// Maximum mantissa bits supported. Bounded by f64's 52-bit mantissa.
    pub const MAX_MANTISSA_BITS: u8 = 52;

    // -- Derived properties --

    /// Total bits for one value: 1 (sign) + exponent + mantissa.
    pub const fn total_bits(&self) -> u8 {
        1 + self.exponent_bits + self.mantissa_bits
    }

    /// The exponent bias: `2^(exponent_bits - 1) - 1`.
    pub const fn bias(&self) -> i32 {
        (1 << (self.exponent_bits - 1)) - 1
    }

    /// Maximum biased exponent value (all exponent bits set).
    pub const fn max_biased_exponent(&self) -> u32 {
        (1u32 << self.exponent_bits) - 1
    }

    /// Whether this configuration is within the supported bounds.
    pub const fn is_supported(&self) -> bool {
        self.exponent_bits >= 1
            && self.exponent_bits <= Self::MAX_EXPONENT_BITS
            && self.mantissa_bits >= 1
            && self.mantissa_bits <= Self::MAX_MANTISSA_BITS
            && self.total_bits() <= 64
    }
}

impl fmt::Display for FloatType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::F64 => write!(f, "F64"),
            Self::F32 => write!(f, "F32"),
            Self::F16 => write!(f, "F16"),
            Self::BF16 => write!(f, "BF16"),
            Self::F8E4M3FN => write!(f, "F8E4M3FN"),
            Self::F8E5M2 => write!(f, "F8E5M2"),
            Self::F4E2M1 => write!(f, "F4E2M1"),
            Self::F6E3M2 => write!(f, "F6E3M2"),
            Self::F6E2M3 => write!(f, "F6E2M3"),
            _ => write!(
                f,
                "Float(e{}m{}{}{})",
                self.exponent_bits,
                self.mantissa_bits,
                if self.has_infinity { "" } else { "_noinf" },
                if self.has_nan { "" } else { "_nonan" },
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

    /// Maximum bit width supported for integer types.
    pub const MAX_BITS: u8 = 64;

    /// Whether this configuration is within the supported bounds.
    pub const fn is_supported(&self) -> bool {
        self.bits >= 1 && self.bits <= Self::MAX_BITS
    }
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
    pub const F6E3M2: Self = NumericDType::Float(FloatType::F6E3M2);
    pub const F6E2M3: Self = NumericDType::Float(FloatType::F6E2M3);

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
        bits.div_ceil(8)
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

    /// Get the numeric dtype, or panic with a message if this is not numeric.
    pub fn expect_numeric(&self, msg: &str) -> NumericDType {
        self.as_numeric()
            .unwrap_or_else(|| panic!("{msg}: expected numeric dtype, got {self}"))
    }

    /// Parse from an ONNX protobuf DataType enum value (i32).
    pub fn from_onnx_i32(value: i32) -> Result<Self, ONNXDTypeError> {
        use crate::onnx::tensor_proto::DataType;
        let dt = DataType::try_from(value)
            .map_err(|_| ONNXDTypeError::UnsupportedONNXDataType(value))?;
        Self::from_onnx_proto(dt)
    }

    /// Parse from an ONNX protobuf DataType enum.
    pub fn from_onnx_proto(
        dt: crate::onnx::tensor_proto::DataType,
    ) -> Result<Self, ONNXDTypeError> {
        use crate::onnx::tensor_proto::DataType;
        Ok(match dt {
            DataType::Double => ONNXDType::Numeric(NumericDType::F64),
            DataType::Float => ONNXDType::Numeric(NumericDType::F32),
            DataType::Bfloat16 => ONNXDType::Numeric(NumericDType::BF16),
            DataType::Float16 => ONNXDType::Numeric(NumericDType::F16),
            DataType::Float8e4m3fn => ONNXDType::Numeric(NumericDType::F8E4M3FN),
            DataType::Float8e5m2 => ONNXDType::Numeric(NumericDType::F8E5M2),
            DataType::Float4e2m1 => ONNXDType::Numeric(NumericDType::F4E2M1),
            DataType::Int64 => ONNXDType::Numeric(NumericDType::I64),
            DataType::Int32 => ONNXDType::Numeric(NumericDType::I32),
            DataType::Int16 => ONNXDType::Numeric(NumericDType::I16),
            DataType::Int8 => ONNXDType::Numeric(NumericDType::I8),
            DataType::Int4 => ONNXDType::Numeric(NumericDType::I4),
            DataType::Uint64 => ONNXDType::Numeric(NumericDType::U64),
            DataType::Uint32 => ONNXDType::Numeric(NumericDType::U32),
            DataType::Uint16 => ONNXDType::Numeric(NumericDType::U16),
            DataType::Uint8 => ONNXDType::Numeric(NumericDType::U8),
            DataType::Uint4 => ONNXDType::Numeric(NumericDType::U4),
            DataType::Bool => ONNXDType::Numeric(NumericDType::BOOL),
            DataType::String => ONNXDType::String,
            other => return Err(ONNXDTypeError::UnsupportedONNXDataType(other as i32)),
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ONNXDTypeError {
    #[error("Unsupported ONNX DataType value: {0}")]
    UnsupportedONNXDataType(i32),
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
// ONNXTensor — symbolic-graph-level tensor that can hold string or numeric data
// ---------------------------------------------------------------------------

/// Tensor at the ONNX / symbolic-graph level.
///
/// Wraps either a pool-backed `NumericTensor` (the common case) or a string
/// tensor (rare — used by Equal, StringNormalizer, etc.). Milli-op lowering
/// only accepts the Numeric arm; string ops must be handled eagerly at the
/// symbolic eval level.
pub enum ONNXTensor<'p, P: crate::pool::Pool + 'p> {
    Numeric(crate::numeric_tensor::NumericTensor<'p, crate::tensor_rank::DynRank, P>),
    String {
        shape: Vec<u64>,
        data: Vec<std::string::String>,
    },
}

impl<'p, P: crate::pool::Pool + 'p> ONNXTensor<'p, P> {
    /// Get the ONNX-level dtype.
    pub fn onnx_dtype(&self) -> ONNXDType {
        match self {
            ONNXTensor::Numeric(t) => ONNXDType::Numeric(t.dtype()),
            ONNXTensor::String { .. } => ONNXDType::String,
        }
    }

    /// Unwrap as a numeric tensor, or return an error.
    pub fn as_numeric(
        &self,
    ) -> Result<&crate::numeric_tensor::NumericTensor<'p, crate::tensor_rank::DynRank, P>, String>
    {
        match self {
            ONNXTensor::Numeric(t) => Ok(t),
            ONNXTensor::String { .. } => Err("expected numeric tensor, got string".into()),
        }
    }

    /// Unwrap into a numeric tensor, or return an error.
    pub fn into_numeric(
        self,
    ) -> Result<crate::numeric_tensor::NumericTensor<'p, crate::tensor_rank::DynRank, P>, String>
    {
        match self {
            ONNXTensor::Numeric(t) => Ok(t),
            ONNXTensor::String { .. } => Err("expected numeric tensor, got string".into()),
        }
    }

    pub fn shape(&self) -> &[u64] {
        match self {
            ONNXTensor::Numeric(t) => t.shape(),
            ONNXTensor::String { shape, .. } => shape,
        }
    }
}

impl<'p, P: crate::pool::Pool + 'p> ONNXTensor<'p, P> {
    /// Borrow as a pool-erased view.
    pub fn view(&self) -> ONNXTensorView<'_> {
        match self {
            ONNXTensor::Numeric(t) => ONNXTensorView::Numeric(t.view()),
            ONNXTensor::String { shape, data } => ONNXTensorView::String { shape, data },
        }
    }
}

impl<'p, P: crate::pool::Pool + 'p>
    From<crate::numeric_tensor::NumericTensor<'p, crate::tensor_rank::DynRank, P>>
    for ONNXTensor<'p, P>
{
    fn from(t: crate::numeric_tensor::NumericTensor<'p, crate::tensor_rank::DynRank, P>) -> Self {
        ONNXTensor::Numeric(t)
    }
}

/// Borrowed view of an ONNXTensor. Pool-erased — can reference data from any pool.
///
/// Parallels `NumericTensorView` but also handles string data.
pub enum ONNXTensorView<'a> {
    Numeric(crate::numeric_tensor::NumericTensorView<'a, crate::tensor_rank::DynRank>),
    String {
        shape: &'a [u64],
        data: &'a [std::string::String],
    },
}

impl<'a> ONNXTensorView<'a> {
    pub fn onnx_dtype(&self) -> ONNXDType {
        match self {
            ONNXTensorView::Numeric(v) => ONNXDType::Numeric(v.dtype()),
            ONNXTensorView::String { .. } => ONNXDType::String,
        }
    }

    pub fn as_numeric(
        &self,
    ) -> Result<&crate::numeric_tensor::NumericTensorView<'a, crate::tensor_rank::DynRank>, String>
    {
        match self {
            ONNXTensorView::Numeric(v) => Ok(v),
            ONNXTensorView::String { .. } => Err("expected numeric tensor, got string".into()),
        }
    }

    pub fn shape(&self) -> &[u64] {
        match self {
            ONNXTensorView::Numeric(v) => v.shape(),
            ONNXTensorView::String { shape, .. } => shape,
        }
    }
}

// ---------------------------------------------------------------------------
// NumericPrimitive — map Rust types to NumericDType / NumericScalar
// ---------------------------------------------------------------------------

use crate::numeric_scalar::NumericScalar;

/// Trait implemented by Rust primitive types that have a corresponding `NumericDType`.
///
/// Provides the bridge between typed Rust values and the dtype-erased `NumericScalar`
/// used throughout the tensor system.
pub trait NumericPrimitive: Copy + Clone + PartialEq + 'static {
    const NUMERIC_DTYPE: NumericDType;
    fn to_scalar(self) -> NumericScalar;
    fn from_scalar(s: &NumericScalar) -> Self;
}

impl NumericPrimitive for f64 {
    const NUMERIC_DTYPE: NumericDType = NumericDType::F64;
    fn to_scalar(self) -> NumericScalar {
        NumericScalar::from_f64(self)
    }
    fn from_scalar(s: &NumericScalar) -> Self {
        s.to_f64()
    }
}
impl NumericPrimitive for f32 {
    const NUMERIC_DTYPE: NumericDType = NumericDType::F32;
    fn to_scalar(self) -> NumericScalar {
        NumericScalar::from_f32(self)
    }
    fn from_scalar(s: &NumericScalar) -> Self {
        s.to_f32()
    }
}
impl NumericPrimitive for half::bf16 {
    const NUMERIC_DTYPE: NumericDType = NumericDType::BF16;
    fn to_scalar(self) -> NumericScalar {
        NumericScalar::from_bf16(self)
    }
    fn from_scalar(s: &NumericScalar) -> Self {
        half::bf16::from_bits(u16::from_le_bytes(s.raw_bits()[..2].try_into().unwrap()))
    }
}
impl NumericPrimitive for half::f16 {
    const NUMERIC_DTYPE: NumericDType = NumericDType::F16;
    fn to_scalar(self) -> NumericScalar {
        NumericScalar::from_f16(self)
    }
    fn from_scalar(s: &NumericScalar) -> Self {
        half::f16::from_bits(u16::from_le_bytes(s.raw_bits()[..2].try_into().unwrap()))
    }
}
impl NumericPrimitive for float8::F8E4M3 {
    const NUMERIC_DTYPE: NumericDType = NumericDType::F8E4M3FN;
    fn to_scalar(self) -> NumericScalar {
        NumericScalar::from_f8e4m3fn(self)
    }
    fn from_scalar(s: &NumericScalar) -> Self {
        float8::F8E4M3::from_bits(s.raw_bits()[0])
    }
}
impl NumericPrimitive for float8::F8E5M2 {
    const NUMERIC_DTYPE: NumericDType = NumericDType::F8E5M2;
    fn to_scalar(self) -> NumericScalar {
        NumericScalar::from_f8e5m2(self)
    }
    fn from_scalar(s: &NumericScalar) -> Self {
        float8::F8E5M2::from_bits(s.raw_bits()[0])
    }
}
impl NumericPrimitive for i64 {
    const NUMERIC_DTYPE: NumericDType = NumericDType::I64;
    fn to_scalar(self) -> NumericScalar {
        NumericScalar::from_i64(self)
    }
    fn from_scalar(s: &NumericScalar) -> Self {
        s.to_i64()
    }
}
impl NumericPrimitive for u64 {
    const NUMERIC_DTYPE: NumericDType = NumericDType::U64;
    fn to_scalar(self) -> NumericScalar {
        NumericScalar::from_u64(self)
    }
    fn from_scalar(s: &NumericScalar) -> Self {
        u64::from_le_bytes(s.raw_bits()[..8].try_into().unwrap())
    }
}
impl NumericPrimitive for i32 {
    const NUMERIC_DTYPE: NumericDType = NumericDType::I32;
    fn to_scalar(self) -> NumericScalar {
        NumericScalar::from_i32(self)
    }
    fn from_scalar(s: &NumericScalar) -> Self {
        i32::from_le_bytes(s.raw_bits()[..4].try_into().unwrap())
    }
}
impl NumericPrimitive for u32 {
    const NUMERIC_DTYPE: NumericDType = NumericDType::U32;
    fn to_scalar(self) -> NumericScalar {
        NumericScalar::from_u32(self)
    }
    fn from_scalar(s: &NumericScalar) -> Self {
        u32::from_le_bytes(s.raw_bits()[..4].try_into().unwrap())
    }
}
impl NumericPrimitive for i16 {
    const NUMERIC_DTYPE: NumericDType = NumericDType::I16;
    fn to_scalar(self) -> NumericScalar {
        NumericScalar::from_i16(self)
    }
    fn from_scalar(s: &NumericScalar) -> Self {
        i16::from_le_bytes(s.raw_bits()[..2].try_into().unwrap())
    }
}
impl NumericPrimitive for u16 {
    const NUMERIC_DTYPE: NumericDType = NumericDType::U16;
    fn to_scalar(self) -> NumericScalar {
        NumericScalar::from_u16(self)
    }
    fn from_scalar(s: &NumericScalar) -> Self {
        u16::from_le_bytes(s.raw_bits()[..2].try_into().unwrap())
    }
}
impl NumericPrimitive for i8 {
    const NUMERIC_DTYPE: NumericDType = NumericDType::I8;
    fn to_scalar(self) -> NumericScalar {
        NumericScalar::from_i8(self)
    }
    fn from_scalar(s: &NumericScalar) -> Self {
        s.raw_bits()[0] as i8
    }
}
impl NumericPrimitive for u8 {
    const NUMERIC_DTYPE: NumericDType = NumericDType::U8;
    fn to_scalar(self) -> NumericScalar {
        NumericScalar::from_u8(self)
    }
    fn from_scalar(s: &NumericScalar) -> Self {
        s.raw_bits()[0]
    }
}
impl NumericPrimitive for arbitrary_int::i4 {
    const NUMERIC_DTYPE: NumericDType = NumericDType::I4;
    fn to_scalar(self) -> NumericScalar {
        NumericScalar::from_i4(self)
    }
    fn from_scalar(s: &NumericScalar) -> Self {
        arbitrary_int::i4::new((s.raw_bits()[0] & 0x0F) as i8)
    }
}
impl NumericPrimitive for arbitrary_int::u4 {
    const NUMERIC_DTYPE: NumericDType = NumericDType::U4;
    fn to_scalar(self) -> NumericScalar {
        NumericScalar::from_u4(self)
    }
    fn from_scalar(s: &NumericScalar) -> Self {
        arbitrary_int::u4::new(s.raw_bits()[0] & 0x0F)
    }
}
impl NumericPrimitive for bool {
    const NUMERIC_DTYPE: NumericDType = NumericDType::BOOL;
    fn to_scalar(self) -> NumericScalar {
        NumericScalar::from_bool(self)
    }
    fn from_scalar(s: &NumericScalar) -> Self {
        s.raw_bits()[0] != 0
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
            DType::STRING => return None,
        })
    }

    /// Convert back to the legacy `DType` enum.
    ///
    /// Panics if this type has no legacy equivalent (e.g. F6E3M2, F6E2M3,
    /// or custom FloatType/IntType configurations).
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
            other => panic!("NumericDType {other} has no legacy DType equivalent"),
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
                None => panic!("DType::Packed has no ONNXDType representation"),
            },
        }
    }

    /// Convert back to the legacy `DType` enum.
    ///
    /// Panics if the inner `NumericDType` has no legacy equivalent.
    pub fn to_legacy(self) -> DType {
        match self {
            ONNXDType::String => DType::STRING,
            ONNXDType::Numeric(ndt) => ndt.to_legacy(),
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
        assert_eq!(FloatType::F6E3M2.total_bits(), 6);
        assert_eq!(FloatType::F6E2M3.total_bits(), 6);
    }

    #[test]
    fn float_type_bias() {
        assert_eq!(FloatType::F64.bias(), 1023);
        assert_eq!(FloatType::F32.bias(), 127);
        assert_eq!(FloatType::F16.bias(), 15);
        assert_eq!(FloatType::BF16.bias(), 127);
        assert_eq!(FloatType::F8E5M2.bias(), 15);
        assert_eq!(FloatType::F8E4M3FN.bias(), 7);
        assert_eq!(FloatType::F4E2M1.bias(), 1);
        assert_eq!(FloatType::F6E3M2.bias(), 3);
        assert_eq!(FloatType::F6E2M3.bias(), 1);
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
        assert_eq!(NumericDType::F6E3M2.to_string(), "F6E3M2");
        assert_eq!(NumericDType::F6E2M3.to_string(), "F6E2M3");
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
    fn legacy_string_returns_none() {
        assert!(NumericDType::from_legacy(DType::STRING).is_none());
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
        let custom = FloatType {
            exponent_bits: 8,
            mantissa_bits: 23,
            has_infinity: true,
            has_nan: true,
        };
        assert_eq!(custom, FloatType::F32);
        assert_eq!(NumericDType::Float(custom), NumericDType::F32);
    }

    #[test]
    fn is_supported() {
        assert!(FloatType::F32.is_supported());
        assert!(FloatType::F4E2M1.is_supported());
        assert!(FloatType::F6E3M2.is_supported());
        // mantissa_bits=0 is unsupported
        assert!(
            !(FloatType {
                exponent_bits: 8,
                mantissa_bits: 0,
                has_infinity: false,
                has_nan: false
            })
            .is_supported()
        );
        // exponent_bits > 11 is unsupported
        assert!(
            !(FloatType {
                exponent_bits: 12,
                mantissa_bits: 4,
                has_infinity: true,
                has_nan: true
            })
            .is_supported()
        );
    }
}
