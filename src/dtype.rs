use serde::{Deserialize, Serialize};
use crate::{Backend, onnx};

#[derive(Debug, thiserror::Error)]
pub enum DTypeError {
    #[error("The backend {0} does not support the dtype {1}")]
    DTypeNotSupportedByBackend(DType, Backend),
    #[error("The onnx dtype {0:?} is not supported")]
    UnsupportedONNXDtype(onnx::tensor_proto::DataType)
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum DType {
    F64,
    F32,
    BF16,
    F16,
    I64,
    I32,
    U64,
    U32,
    U16,
    U8
}

impl TryFrom<onnx::tensor_proto::DataType> for DType {
    type Error = DTypeError;
    fn try_from(onnx_dtype: onnx::tensor_proto::DataType) -> Result<Self, DTypeError> {
        match onnx_dtype {
            onnx::tensor_proto::DataType::Double => Ok(DType::F64),
            onnx::tensor_proto::DataType::Float => Ok(DType::F32),
            onnx::tensor_proto::DataType::Bfloat16 => Ok(DType::BF16),
            onnx::tensor_proto::DataType::Float16 => Ok(DType::F16),
            onnx::tensor_proto::DataType::Int64 => Ok(DType::I64),
            onnx::tensor_proto::DataType::Int32 => Ok(DType::I32),
            onnx::tensor_proto::DataType::Uint64 => Ok(DType::U64),
            onnx::tensor_proto::DataType::Uint32 => Ok(DType::U32),
            onnx::tensor_proto::DataType::Uint16 => Ok(DType::U16),
            _ => Err(DTypeError::UnsupportedONNXDtype(onnx_dtype))
        }
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
            DType::U8 => onnx::tensor_proto::DataType::Uint8,
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
            DType::U16 => write!(f, "UInt16"),
            DType::U8 => write!(f, "UInt8"),
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
            _ => Err(DTypeError::DTypeNotSupportedByBackend(value, Backend::Candle))?
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