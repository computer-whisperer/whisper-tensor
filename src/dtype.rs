use half::{bf16, f16};
use serde::{Deserialize, Serialize};
use crate::{onnx};

#[derive(Debug, thiserror::Error)]
pub enum DTypeError {
    #[error("The backend does not support the dtype {0}")]
    DTypeNotSupportedByBackend(DType),
    #[error("The onnx dtype {0:?} is not supported")]
    UnsupportedONNXDtype(onnx::tensor_proto::DataType),
    #[cfg(feature = "ort")]
    #[error("The ort dtype {0:?} is not supported")]
    UnsupportedORTDtype(ort::tensor::TensorElementType)
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum DType {
    F64,
    F32,
    BF16,
    F16,
    U64,
    I64,
    U32,
    I32,
    U16,
    I16,
    U8,
    I8,
    BOOL
}

impl DType {
    pub fn size(&self) -> usize {
        match self {
            DType::F64 => 8,
            DType::F32 => 4,
            DType::BF16 => 2,
            DType::F16 => 2,
            DType::U64 => 8,
            DType::I64 => 8,
            DType::U32 => 4,
            DType::I32 => 4,
            DType::U16 => 2,
            DType::I16 => 2,
            DType::U8 => 1,
            DType::I8 => 1,
            DType::BOOL => 1
        }
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
            onnx::tensor_proto::DataType::Int64 => DType::I64,
            onnx::tensor_proto::DataType::Int32 => DType::I32,
            onnx::tensor_proto::DataType::Uint64 => DType::U64,
            onnx::tensor_proto::DataType::Uint32 => DType::U32,
            onnx::tensor_proto::DataType::Uint16 => DType::U16,
            onnx::tensor_proto::DataType::Int16 => DType::I16,
            onnx::tensor_proto::DataType::Uint8 => DType::U8,
            onnx::tensor_proto::DataType::Int8 => DType::I8,
            onnx::tensor_proto::DataType::Bool => DType::BOOL,
            _ => Err(DTypeError::UnsupportedONNXDtype(onnx_dtype))?
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
            DType::I64 => onnx::tensor_proto::DataType::Int64,
            DType::I32 => onnx::tensor_proto::DataType::Int32,
            DType::U64 => onnx::tensor_proto::DataType::Uint64,
            DType::U32 => onnx::tensor_proto::DataType::Uint32,
            DType::U16 => onnx::tensor_proto::DataType::Uint16,
            DType::I16 => onnx::tensor_proto::DataType::Int16,
            DType::U8 => onnx::tensor_proto::DataType::Uint8,
            DType::I8 => onnx::tensor_proto::DataType::Int8,
            DType::BOOL => onnx::tensor_proto::DataType::Bool
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
            DType::I64 => write!(f, "Int64"),
            DType::I32 => write!(f, "Int32"),
            DType::U64 => write!(f, "UInt64"),
            DType::U32 => write!(f, "UInt32"),
            DType::I16 => write!(f, "Int16"),
            DType::U16 => write!(f, "UInt16"),
            DType::U8 => write!(f, "UInt8"),
            DType::I8 => write!(f, "Int8"),
            DType::BOOL => write!(f, "Bool")
        }
    }
}

#[cfg(feature = "candle")]
impl TryFrom<DType> for candle_core::DType {
    type Error = DTypeError;
    fn try_from(value: DType) -> Result<Self, Self::Error> {
        Ok(match value {
            DType::F64 => candle_core::DType::F64,
            DType::F32 => candle_core::DType::F32,
            DType::BF16 => candle_core::DType::BF16,
            DType::F16 => candle_core::DType::F16,
            DType::I64 => candle_core::DType::I64,
            DType::U32 => candle_core::DType::U32,
            DType::U8 => candle_core::DType::U8,
            _ => Err(DTypeError::DTypeNotSupportedByBackend(value))?
        })
    }
}

#[cfg(feature = "candle")]
impl From<candle_core::DType> for DType {
    fn from(value: candle_core::DType) -> Self {
        match value {
            candle_core::DType::F64 => DType::F64,
            candle_core::DType::F32 => DType::F32,
            candle_core::DType::BF16 => DType::BF16,
            candle_core::DType::F16 => DType::F16,
            candle_core::DType::I64 => DType::I64,
            candle_core::DType::U8 => DType::U8,
            candle_core::DType::U32 => DType::U32
        }
    }
}

pub trait DTypeOfPrimitive {
    const DTYPE: DType;
}

impl DTypeOfPrimitive for f64 { const DTYPE: DType = DType::F64; }
impl DTypeOfPrimitive for f32 { const DTYPE: DType = DType::F32; }
impl DTypeOfPrimitive for bf16 { const DTYPE: DType = DType::BF16; }
impl DTypeOfPrimitive for f16 { const DTYPE: DType = DType::F16; }
impl DTypeOfPrimitive for i64 { const DTYPE: DType = DType::I64; }
impl DTypeOfPrimitive for u64 { const DTYPE: DType = DType::U64; }
impl DTypeOfPrimitive for i32 { const DTYPE: DType = DType::I32; }
impl DTypeOfPrimitive for u32 { const DTYPE: DType = DType::U32; }
impl DTypeOfPrimitive for i16 { const DTYPE: DType = DType::I16; }
impl DTypeOfPrimitive for u16 { const DTYPE: DType = DType::U16; }
impl DTypeOfPrimitive for i8 { const DTYPE: DType = DType::I8; }
impl DTypeOfPrimitive for u8 { const DTYPE: DType = DType::U8; }
impl DTypeOfPrimitive for bool { const DTYPE: DType = DType::BOOL; }
