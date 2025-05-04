use half::{bf16, f16};
use serde::{Deserialize, Serialize};
use crate::dtype::DType;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NumericScalar {
    F64(f64),
    F32(f32),
    BF16(bf16),
    F16(f16),
    U64(u64),
    I64(i64),
    U32(u32),
    I32(i32),
    U16(u16),
    I16(i16),
    U8(u8),
    I8(i8),
    BOOL(bool)
}

impl NumericScalar {
    pub fn dtype(&self) -> DType {
        match self {
            NumericScalar::F64(_) => DType::F64,
            NumericScalar::F32(_) => DType::F32,
            NumericScalar::BF16(_) => DType::BF16,
            NumericScalar::F16(_) => DType::F16,
            NumericScalar::U64(_) => DType::U64,
            NumericScalar::I64(_) => DType::I64,
            NumericScalar::U32(_) => DType::U32,
            NumericScalar::I32(_) => DType::I32,
            NumericScalar::U16(_) => DType::U16,
            NumericScalar::I16(_) => DType::I16,
            NumericScalar::U8(_) => DType::U8,
            NumericScalar::I8(_) => DType::I8,
            NumericScalar::BOOL(_) => DType::BOOL,
        }
    }
    
    pub fn zero_of(dtype: DType) -> Self {
        match dtype {
            DType::F64 => NumericScalar::F64(0.0),
            DType::F32 => NumericScalar::F32(0.0),
            DType::BF16 => NumericScalar::BF16(bf16::ZERO),
            DType::F16 => NumericScalar::F16(f16::ZERO),
            DType::U64 => NumericScalar::U64(0),
            DType::I64 => NumericScalar::I64(0),
            DType::U32 => NumericScalar::U32(0),
            DType::I32 => NumericScalar::I32(0),
            DType::U16 => NumericScalar::U16(0),
            DType::I16 => NumericScalar::I16(0),
            DType::U8 => NumericScalar::U8(0),
            DType::I8 => NumericScalar::I8(0),
            DType::BOOL => NumericScalar::BOOL(false),
        }
    }
}

impl From<f64> for NumericScalar {
    fn from(value: f64) -> Self {
        NumericScalar::F64(value)
    }
}

impl From<f32> for NumericScalar {
    fn from(value: f32) -> Self {
        NumericScalar::F32(value)
    }
}

impl From<bf16> for NumericScalar {
    fn from(value: bf16) -> Self {
        NumericScalar::BF16(value)
    }
}

impl From<f16> for NumericScalar {
    fn from(value: f16) -> Self {
        NumericScalar::F16(value)
    }
}

impl From<u64> for NumericScalar {
    fn from(value: u64) -> Self {
        NumericScalar::U64(value)
    }
}

impl From<i64> for NumericScalar {
    fn from(value: i64) -> Self {
        NumericScalar::I64(value)
    }
}

impl From<u32> for NumericScalar {
    fn from(value: u32) -> Self {
        NumericScalar::U32(value)
    }
}

impl From<i32> for NumericScalar {
    fn from(value: i32) -> Self {
        NumericScalar::I32(value)
    }
}

impl From<u16> for NumericScalar {
    fn from(value: u16) -> Self {
        NumericScalar::U16(value)
    }
}

impl From<i16> for NumericScalar {
    fn from(value: i16) -> Self {
        NumericScalar::I16(value)
    }
}

impl From<u8> for NumericScalar {
    fn from(value: u8) -> Self {
        NumericScalar::U8(value)
    }
}

impl From<i8> for NumericScalar {
    fn from(value: i8) -> Self {
        NumericScalar::I8(value)
    }
}
