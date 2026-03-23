use crate::onnx;
use crate::migration::packed_format::PackedFormat;
use arbitrary_int::{i4, u4};
use float8::{F8E4M3, F8E5M2};
use half::{bf16, f16};
use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum DTypeError {
    #[error("The backend does not support the dtype {0}")]
    DTypeNotSupportedByBackend(DType),
    #[error("The onnx dtype {0:?} is not supported")]
    UnsupportedONNXDtype(onnx::tensor_proto::DataType),
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum DType {
    // Floating-point (descending precision)
    F64,
    F32,
    BF16,
    F16,
    F8E4M3FN,
    F8E5M2,
    F4E2M1,
    // Signed integers (descending width)
    I64,
    I32,
    I16,
    I8,
    I4,
    // Unsigned integers (descending width)
    U64,
    U32,
    U16,
    U8,
    U4,
    // Other
    BOOL,
    STRING,
    // Block-quantized
    Packed(PackedFormat),
}

impl DType {
    /// Bytes per element when stored in an element-addressable (one element per slot) layout.
    /// Returns None for STRING and Packed types which don't have fixed-size addressable elements.
    /// Note: 4-bit types occupy one byte per element in addressable storage, even though
    /// their logical bit-width is 4. Packed serialization formats (e.g. ONNX raw_data)
    /// may store two 4-bit values per byte, but that is a serialization concern.
    pub fn bytes_per_element(&self) -> Option<usize> {
        match self {
            DType::F64 => Some(8),
            DType::F32 => Some(4),
            DType::BF16 => Some(2),
            DType::F16 => Some(2),
            DType::F8E4M3FN => Some(1),
            DType::F8E5M2 => Some(1),
            DType::F4E2M1 => Some(1),
            DType::I64 => Some(8),
            DType::I32 => Some(4),
            DType::I16 => Some(2),
            DType::I8 => Some(1),
            DType::I4 => Some(1),
            DType::U64 => Some(8),
            DType::U32 => Some(4),
            DType::U16 => Some(2),
            DType::U8 => Some(1),
            DType::U4 => Some(1),
            DType::BOOL => Some(1),
            DType::STRING => None,
            DType::Packed(_) => None,
        }
    }
}

impl DType {
    /// Returns the packed format if this is a Packed dtype, None otherwise.
    pub fn packed_format(&self) -> Option<PackedFormat> {
        match self {
            DType::Packed(fmt) => Some(*fmt),
            _ => None,
        }
    }

    /// Returns true if this is a packed (block-quantized) type.
    pub fn is_packed(&self) -> bool {
        matches!(self, DType::Packed(_))
    }
}

impl TryFrom<onnx::tensor_proto::DataType> for DType {
    type Error = DTypeError;
    fn try_from(onnx_dtype: onnx::tensor_proto::DataType) -> Result<Self, DTypeError> {
        Ok(match onnx_dtype {
            onnx::tensor_proto::DataType::Double => DType::F64,
            onnx::tensor_proto::DataType::Float => DType::F32,
            onnx::tensor_proto::DataType::Bfloat16 => DType::BF16,
            onnx::tensor_proto::DataType::Float16 => DType::F16,
            onnx::tensor_proto::DataType::Float8e4m3fn => DType::F8E4M3FN,
            onnx::tensor_proto::DataType::Float8e5m2 => DType::F8E5M2,
            onnx::tensor_proto::DataType::Float4e2m1 => DType::F4E2M1,
            onnx::tensor_proto::DataType::Int64 => DType::I64,
            onnx::tensor_proto::DataType::Int32 => DType::I32,
            onnx::tensor_proto::DataType::Int16 => DType::I16,
            onnx::tensor_proto::DataType::Int8 => DType::I8,
            onnx::tensor_proto::DataType::Int4 => DType::I4,
            onnx::tensor_proto::DataType::Uint64 => DType::U64,
            onnx::tensor_proto::DataType::Uint32 => DType::U32,
            onnx::tensor_proto::DataType::Uint16 => DType::U16,
            onnx::tensor_proto::DataType::Uint8 => DType::U8,
            onnx::tensor_proto::DataType::Uint4 => DType::U4,
            onnx::tensor_proto::DataType::Bool => DType::BOOL,
            onnx::tensor_proto::DataType::String => DType::STRING,
            _ => Err(DTypeError::UnsupportedONNXDtype(onnx_dtype))?,
        })
    }
}

impl From<DType> for onnx::tensor_proto::DataType {
    fn from(dtype: DType) -> Self {
        match dtype {
            DType::F64 => onnx::tensor_proto::DataType::Double,
            DType::F32 => onnx::tensor_proto::DataType::Float,
            DType::BF16 => onnx::tensor_proto::DataType::Bfloat16,
            DType::F16 => onnx::tensor_proto::DataType::Float16,
            DType::F8E4M3FN => onnx::tensor_proto::DataType::Float8e4m3fn,
            DType::F8E5M2 => onnx::tensor_proto::DataType::Float8e5m2,
            DType::F4E2M1 => onnx::tensor_proto::DataType::Float4e2m1,
            DType::I64 => onnx::tensor_proto::DataType::Int64,
            DType::I32 => onnx::tensor_proto::DataType::Int32,
            DType::I16 => onnx::tensor_proto::DataType::Int16,
            DType::I8 => onnx::tensor_proto::DataType::Int8,
            DType::I4 => onnx::tensor_proto::DataType::Int4,
            DType::U64 => onnx::tensor_proto::DataType::Uint64,
            DType::U32 => onnx::tensor_proto::DataType::Uint32,
            DType::U16 => onnx::tensor_proto::DataType::Uint16,
            DType::U8 => onnx::tensor_proto::DataType::Uint8,
            DType::U4 => onnx::tensor_proto::DataType::Uint4,
            DType::BOOL => onnx::tensor_proto::DataType::Bool,
            DType::STRING => onnx::tensor_proto::DataType::String,
            DType::Packed(fmt) => panic!("Packed format {fmt} has no ONNX DataType equivalent"),
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DType::F64 => write!(f, "Float64"),
            DType::F32 => write!(f, "Float32"),
            DType::BF16 => write!(f, "BFloat16"),
            DType::F16 => write!(f, "Float16"),
            DType::F8E4M3FN => write!(f, "Float8E4M3FN"),
            DType::F8E5M2 => write!(f, "Float8E5M2"),
            DType::F4E2M1 => write!(f, "Float4E2M1"),
            DType::I64 => write!(f, "Int64"),
            DType::I32 => write!(f, "Int32"),
            DType::I16 => write!(f, "Int16"),
            DType::I8 => write!(f, "Int8"),
            DType::I4 => write!(f, "Int4"),
            DType::U64 => write!(f, "UInt64"),
            DType::U32 => write!(f, "UInt32"),
            DType::U16 => write!(f, "UInt16"),
            DType::U8 => write!(f, "UInt8"),
            DType::U4 => write!(f, "UInt4"),
            DType::BOOL => write!(f, "Bool"),
            DType::STRING => write!(f, "String"),
            DType::Packed(fmt) => write!(f, "Packed({fmt})"),
        }
    }
}

pub trait DTypeOfPrimitive {
    const DTYPE: DType;
}

impl DTypeOfPrimitive for f64 {
    const DTYPE: DType = DType::F64;
}
impl DTypeOfPrimitive for f32 {
    const DTYPE: DType = DType::F32;
}
impl DTypeOfPrimitive for bf16 {
    const DTYPE: DType = DType::BF16;
}
impl DTypeOfPrimitive for f16 {
    const DTYPE: DType = DType::F16;
}
impl DTypeOfPrimitive for F8E4M3 {
    const DTYPE: DType = DType::F8E4M3FN;
}
impl DTypeOfPrimitive for F8E5M2 {
    const DTYPE: DType = DType::F8E5M2;
}
impl DTypeOfPrimitive for i64 {
    const DTYPE: DType = DType::I64;
}
impl DTypeOfPrimitive for u64 {
    const DTYPE: DType = DType::U64;
}
impl DTypeOfPrimitive for i32 {
    const DTYPE: DType = DType::I32;
}
impl DTypeOfPrimitive for u32 {
    const DTYPE: DType = DType::U32;
}
impl DTypeOfPrimitive for i16 {
    const DTYPE: DType = DType::I16;
}
impl DTypeOfPrimitive for u16 {
    const DTYPE: DType = DType::U16;
}
impl DTypeOfPrimitive for i8 {
    const DTYPE: DType = DType::I8;
}
impl DTypeOfPrimitive for u8 {
    const DTYPE: DType = DType::U8;
}
impl DTypeOfPrimitive for i4 {
    const DTYPE: DType = DType::I4;
}
impl DTypeOfPrimitive for u4 {
    const DTYPE: DType = DType::U4;
}
impl DTypeOfPrimitive for bool {
    const DTYPE: DType = DType::BOOL;
}
impl DTypeOfPrimitive for String {
    const DTYPE: DType = DType::STRING;
}
