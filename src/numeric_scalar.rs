use half::{bf16, f16};
use num_traits::Float;
use serde::{Deserialize, Serialize};
use crate::dtype::{DType, DTypeOfPrimitive};
use crate::TrigOp;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    BOOL(bool),
    STRING(String)
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
            NumericScalar::STRING(_) => DType::STRING
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
            DType::STRING => NumericScalar::STRING(String::new()),
        }
    }
    
    pub fn neg(&self) -> Self {
        match self {
            NumericScalar::F64(x) => Self::F64(-x),
            NumericScalar::F32(x) => Self::F32(-x),
            NumericScalar::BF16(x) => Self::BF16(-x),
            NumericScalar::F16(x) => Self::F16(-x),
            NumericScalar::I64(x) => Self::I64(-x),
            NumericScalar::I32(x) => Self::I32(-x),
            NumericScalar::I16(x) => Self::I16(-x),
            NumericScalar::I8(x) => Self::I8(-x),
            _ => panic!("Cannot negate this type"),
        }
    }

    pub fn not(&self) -> Self {
        match self {
            NumericScalar::BOOL(x) => Self::BOOL(!x),
            _ => panic!("Cannot not this type"),
        }
    }

    pub fn bitwise_not(&self) -> Self {
        match self {
            NumericScalar::I64(x) => Self::I64(!x),
            NumericScalar::U64(x) => Self::U64(!x),
            NumericScalar::I32(x) => Self::I32(!x),
            NumericScalar::U32(x) => Self::U32(!x),
            NumericScalar::I16(x) => Self::I16(!x),
            NumericScalar::U16(x) => Self::U16(!x),
            NumericScalar::I8(x) => Self::I8(!x),
            NumericScalar::U8(x) => Self::U8(!x),
            _ => panic!("Cannot bitwise not this type"),
        }
    }

    pub fn abs(&self) -> Self {
        match self {
            NumericScalar::F64(x) => Self::F64(x.abs()),
            NumericScalar::F32(x) => Self::F32(x.abs()),
            NumericScalar::BF16(x) => Self::BF16(x.abs()),
            NumericScalar::F16(x) => Self::F16(x.abs()),
            NumericScalar::I64(x) => Self::I64(x.abs()),
            NumericScalar::U64(x) => Self::U64(*x),
            NumericScalar::I32(x) => Self::I32(x.abs()),
            NumericScalar::U32(x) => Self::U32(*x),
            NumericScalar::I16(x) => Self::I16(x.abs()),
            NumericScalar::U16(x) => Self::U16(*x),
            NumericScalar::I8(x) => Self::I8(-x),
            NumericScalar::U8(x) => Self::U8(*x),
            _ => panic!("Cannot abs this type"),
        }
    }

    pub fn exp(&self) -> Self {
        match self {
            NumericScalar::F64(x) => Self::F64(x.exp()),
            NumericScalar::F32(x) => Self::F32(x.exp()),
            NumericScalar::BF16(x) => Self::BF16(x.exp()),
            NumericScalar::F16(x) => Self::F16(x.exp()),
            _ => panic!("Cannot exp this type"),
        }
    }

    pub fn ceil(&self) -> Self {
        match self {
            NumericScalar::F64(x) => Self::F64(x.ceil()),
            NumericScalar::F32(x) => Self::F32(x.ceil()),
            NumericScalar::BF16(x) => Self::BF16(x.ceil()),
            NumericScalar::F16(x) => Self::F16(x.ceil()),
            _ => panic!("Cannot ceil this type"),
        }
    }

    pub fn floor(&self) -> Self {
        match self {
            NumericScalar::F64(x) => Self::F64(x.floor()),
            NumericScalar::F32(x) => Self::F32(x.floor()),
            NumericScalar::BF16(x) => Self::BF16(x.floor()),
            NumericScalar::F16(x) => Self::F16(x.floor()),
            _ => panic!("Cannot floor this type"),
        }
    }

    pub fn round(&self) -> Self {
        match self {
            NumericScalar::F64(x) => Self::F64(x.round()),
            NumericScalar::F32(x) => Self::F32(x.round()),
            NumericScalar::BF16(x) => Self::BF16(x.round()),
            NumericScalar::F16(x) => Self::F16(x.round()),
            _ => panic!("Cannot round this type"),
        }
    }

    pub fn sign(&self) -> Self {
        match self {
            NumericScalar::F64(x) => Self::F64(if *x == 0.0 { 0.0 } else { x.signum() }),
            NumericScalar::F32(x) => Self::F32(if *x == 0.0 { 0.0 } else { x.signum() }),
            NumericScalar::BF16(x) => Self::BF16(if x.to_f32() == 0.0 { bf16::from_f32(0.0) } else { x.signum() }),
            NumericScalar::F16(x) => Self::F16(if x.to_f32() == 0.0 { f16::from_f32(0.0) } else { x.signum() }),
            NumericScalar::U64(x) => Self::U64(if *x == 0 { 0 } else { *x }),
            NumericScalar::U32(x) => Self::U32(if *x == 0 { 0 } else { *x }),
            NumericScalar::U16(x) => Self::U16(if *x == 0 { 0 } else { *x }),
            NumericScalar::U8(x) => Self::U8(if *x == 0 { 0 } else { *x }),
            NumericScalar::I64(x) => Self::I64(if *x == 0 { 0 } else { x.signum() }),
            NumericScalar::I32(x) => Self::I32(if *x == 0 { 0 } else { x.signum() }),
            NumericScalar::I16(x) => Self::I16(if *x == 0 { 0 } else { x.signum() }),
            NumericScalar::I8(x) => Self::I8(if *x == 0 { 0 } else { x.signum() }),
            _ => panic!("Cannot sign this type"),
        }
    }

    pub fn is_inf(&self, detect_positive: bool, detect_negative: bool) -> Self {
        match self {
            NumericScalar::F64(x) => Self::BOOL(x.is_infinite() && ((detect_positive && x.is_sign_positive()) || (detect_negative && x.is_sign_negative()))),
            NumericScalar::F32(x) => Self::BOOL(x.is_infinite() && ((detect_positive && x.is_sign_positive()) || (detect_negative && x.is_sign_negative()))),
            NumericScalar::BF16(x) => Self::BOOL(x.is_infinite() && ((detect_positive && x.is_sign_positive()) || (detect_negative && x.is_sign_negative()))),
            NumericScalar::F16(x) => Self::BOOL(x.is_infinite() && ((detect_positive && x.is_sign_positive()) || (detect_negative && x.is_sign_negative()))),
            _ => panic!("Cannot is_inf this type"),
        }
    }

    pub fn is_nan(&self) -> Self {
        match self {
            NumericScalar::F64(x) => Self::BOOL(x.is_nan()),
            NumericScalar::F32(x) => Self::BOOL(x.is_nan()),
            NumericScalar::BF16(x) => Self::BOOL(x.is_nan()),
            NumericScalar::F16(x) => Self::BOOL(x.is_nan()),
            _ => panic!("Cannot is_nan this type"),
        }
    }

    pub fn erf(&self) -> Self {
        unimplemented!();
    }

    pub fn ln(&self) -> Self {
        match self {
            NumericScalar::F64(x) => Self::F64(x.ln()),
            NumericScalar::F32(x) => Self::F32(x.ln()),
            NumericScalar::BF16(x) => Self::BF16(x.ln()),
            NumericScalar::F16(x) => Self::F16(x.ln()),
            _ => panic!("Cannot ln this type"),
        }
    }

    pub fn trig(&self, trig_op: TrigOp) -> Self {
        match self {
            NumericScalar::F64(x) => Self::F64(trig_op.apply(*x)),
            NumericScalar::F32(x) => Self::F32(trig_op.apply(*x)),
            NumericScalar::BF16(x) => Self::BF16(trig_op.apply(*x)),
            NumericScalar::F16(x) => Self::F16(trig_op.apply(*x)),
            _ => panic!("Cannot trig this type"),
        }
    }

    pub fn sqrt(&self) -> Self {
        match self {
            NumericScalar::F64(x) => Self::F64(x.sqrt()),
            NumericScalar::F32(x) => Self::F32(x.sqrt()),
            NumericScalar::BF16(x) => Self::BF16(x.sqrt()),
            NumericScalar::F16(x) => Self::F16(x.sqrt()),
            _ => panic!("Cannot ln this type"),
        }
    }

    pub fn recip(&self) -> Self {
        match self {
            NumericScalar::F64(x) => Self::F64(x.recip()),
            NumericScalar::F32(x) => Self::F32(x.recip()),
            NumericScalar::BF16(x) => Self::BF16(x.recip()),
            NumericScalar::F16(x) => Self::F16(x.recip()),
            _ => panic!("Cannot recip this type"),
        }
    }

    pub fn clamp_min(&self, v: f32) -> Self {
        match self {
            NumericScalar::F64(x) => Self::F64(x.max(v as f64)),
            NumericScalar::F32(x) => Self::F32(x.max(v)),
            NumericScalar::BF16(x) => Self::BF16(x.max(bf16::from_f32(v))),
            NumericScalar::F16(x) => Self::F16(x.max(f16::from_f32(v))),
            _ => panic!("Cannot clamp_min this type"),
        }
    }
    
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            NumericScalar::F64(x) => x.to_le_bytes().to_vec(),
            NumericScalar::F32(x) => x.to_le_bytes().to_vec(),
            NumericScalar::BF16(x) => x.to_le_bytes().to_vec(),
            NumericScalar::F16(x) => x.to_le_bytes().to_vec(),
            NumericScalar::U64(x) => x.to_le_bytes().to_vec(),
            NumericScalar::I64(x) => x.to_le_bytes().to_vec(),
            NumericScalar::U32(x) => x.to_le_bytes().to_vec(),
            NumericScalar::I32(x) => x.to_le_bytes().to_vec(),
            NumericScalar::U16(x) => x.to_le_bytes().to_vec(),
            NumericScalar::I16(x) => x.to_le_bytes().to_vec(),
            NumericScalar::U8(x) => x.to_le_bytes().to_vec(),
            NumericScalar::I8(x) => x.to_le_bytes().to_vec(),
            NumericScalar::BOOL(x) => vec![*x as u8],
            NumericScalar::STRING(_) => panic!()
        }
    }
}

pub trait NumericScalarType: DTypeOfPrimitive {
    fn to_numeric_scalar(self) -> NumericScalar;
    fn cast_from_numeric_scalar(value: &NumericScalar) -> Self;
}

impl NumericScalarType for f64 {
    fn to_numeric_scalar(self) -> NumericScalar {
        NumericScalar::F64(self)
    }
    fn cast_from_numeric_scalar(value: &NumericScalar) -> Self {
        match value {
            NumericScalar::F64(v) => *v,
            NumericScalar::F32(v) => *v as f64,
            NumericScalar::BF16(v) => v.to_f64(),
            NumericScalar::F16(v) => v.to_f64(),
            NumericScalar::U64(v) => *v as f64,
            NumericScalar::I64(v) => *v as f64,
            NumericScalar::U32(v) => *v as f64,
            NumericScalar::I32(v) => *v as f64,
            NumericScalar::U16(v) => *v as f64,
            NumericScalar::I16(v) => *v as f64,
            NumericScalar::U8(v) => *v as f64,
            NumericScalar::I8(v) => *v as f64,
            _ => panic!("Cannot cast from {:?} to f64", value),
        }
    }
}

impl NumericScalarType for f32 {
    fn to_numeric_scalar(self) -> NumericScalar {
        NumericScalar::F32(self)
    }
    fn cast_from_numeric_scalar(value: &NumericScalar) -> Self {
        match value {
            NumericScalar::F64(v) => *v as f32,
            NumericScalar::F32(v) => *v as f32,
            NumericScalar::BF16(v) => v.to_f32(),
            NumericScalar::F16(v) => v.to_f32(),
            NumericScalar::U64(v) => *v as f32,
            NumericScalar::I64(v) => *v as f32,
            NumericScalar::U32(v) => *v as f32,
            NumericScalar::I32(v) => *v as f32,
            NumericScalar::U16(v) => *v as f32,
            NumericScalar::I16(v) => *v as f32,
            NumericScalar::U8(v) => *v as f32,
            NumericScalar::I8(v) => *v as f32,
            _ => panic!("Cannot cast from {:?} to f32", value),
        }
    }
}

impl NumericScalarType for bf16 {
    fn to_numeric_scalar(self) -> NumericScalar {
        NumericScalar::BF16(self)
    }
    fn cast_from_numeric_scalar(value: &NumericScalar) -> Self {
        match value {
            NumericScalar::F64(v) => bf16::from_f64(*v),
            NumericScalar::F32(v) => bf16::from_f32(*v),
            NumericScalar::BF16(v) => *v,
            NumericScalar::F16(v) => bf16::from_f32(v.to_f32()),
            NumericScalar::U64(v) => bf16::from_f32(*v as f32),
            NumericScalar::I64(v) => bf16::from_f32(*v as f32),
            NumericScalar::U32(v) => bf16::from_f32(*v as f32),
            NumericScalar::I32(v) => bf16::from_f32(*v as f32),
            NumericScalar::U16(v) => bf16::from_f32(*v as f32),
            NumericScalar::I16(v) => bf16::from_f32(*v as f32),
            NumericScalar::U8(v) => bf16::from_f32(*v as f32),
            NumericScalar::I8(v) => bf16::from_f32(*v as f32),
            _ => panic!("Cannot cast from {:?} to bf16", value),
        }
    }
}

impl NumericScalarType for f16 {
    fn to_numeric_scalar(self) -> NumericScalar {
        NumericScalar::F16(self)
    }
    fn cast_from_numeric_scalar(value: &NumericScalar) -> Self {
        match value {
            NumericScalar::F64(v) => f16::from_f64(*v),
            NumericScalar::F32(v) => f16::from_f32(*v),
            NumericScalar::BF16(v) => f16::from_f32(v.to_f32()),
            NumericScalar::F16(v) => *v,
            NumericScalar::U64(v) => f16::from_f32(*v as f32),
            NumericScalar::I64(v) => f16::from_f32(*v as f32),
            NumericScalar::U32(v) => f16::from_f32(*v as f32),
            NumericScalar::I32(v) => f16::from_f32(*v as f32),
            NumericScalar::U16(v) => f16::from_f32(*v as f32),
            NumericScalar::I16(v) => f16::from_f32(*v as f32),
            NumericScalar::U8(v) => f16::from_f32(*v as f32),
            NumericScalar::I8(v) => f16::from_f32(*v as f32),
            _ => panic!("Cannot cast from {:?} to f16", value),
        }
    }
}

impl NumericScalarType for u64 {
    fn to_numeric_scalar(self) -> NumericScalar {
        NumericScalar::U64(self)
    }
    fn cast_from_numeric_scalar(value: &NumericScalar) -> Self {
        match value {
            NumericScalar::F64(v) => *v as u64,
            NumericScalar::F32(v) => *v as u64,
            NumericScalar::BF16(v) => v.to_f32() as u64,
            NumericScalar::F16(v) => v.to_f32() as u64,
            NumericScalar::U64(v) => *v,
            NumericScalar::I64(v) => *v as u64,
            NumericScalar::U32(v) => *v as u64,
            NumericScalar::I32(v) => *v as u64,
            NumericScalar::U16(v) => *v as u64,
            NumericScalar::I16(v) => *v as u64,
            NumericScalar::U8(v) => *v as u64,
            NumericScalar::I8(v) => *v as u64,
            _ => panic!("Cannot cast from {:?} to u64", value),
        }
    }
}

impl NumericScalarType for i64 {
    fn to_numeric_scalar(self) -> NumericScalar {
        NumericScalar::I64(self)
    }
    fn cast_from_numeric_scalar(value: &NumericScalar) -> Self {
        match value {
            NumericScalar::F64(v) => *v as i64,
            NumericScalar::F32(v) => *v as i64,
            NumericScalar::BF16(v) => v.to_f32() as i64,
            NumericScalar::F16(v) => v.to_f32() as i64,
            NumericScalar::U64(v) => *v as i64,
            NumericScalar::I64(v) => *v as i64,
            NumericScalar::U32(v) => *v as i64,
            NumericScalar::I32(v) => *v as i64,
            NumericScalar::U16(v) => *v as i64,
            NumericScalar::I16(v) => *v as i64,
            NumericScalar::U8(v) => *v as i64,
            NumericScalar::I8(v) => *v as i64,
            _ => panic!("Cannot cast from {:?} to i64", value),
        }
    }
}


impl NumericScalarType for u32 {
    fn to_numeric_scalar(self) -> NumericScalar {
        NumericScalar::U32(self)
    }
    fn cast_from_numeric_scalar(value: &NumericScalar) -> Self {
        match value {
            NumericScalar::F64(v) => *v as u32,
            NumericScalar::F32(v) => *v as u32,
            NumericScalar::BF16(v) => v.to_f32() as u32,
            NumericScalar::F16(v) => v.to_f32() as u32,
            NumericScalar::U64(v) => *v as u32,
            NumericScalar::I64(v) => *v as u32,
            NumericScalar::U32(v) => *v as u32,
            NumericScalar::I32(v) => *v as u32,
            NumericScalar::U16(v) => *v as u32,
            NumericScalar::I16(v) => *v as u32,
            NumericScalar::U8(v) => *v as u32,
            NumericScalar::I8(v) => *v as u32,
            _ => panic!("Cannot cast from {:?} to u32", value),
        }
    }
}

impl NumericScalarType for i32 {
    fn to_numeric_scalar(self) -> NumericScalar {
        NumericScalar::I32(self)
    }
    fn cast_from_numeric_scalar(value: &NumericScalar) -> Self {
        match value {
            NumericScalar::F64(v) => *v as i32,
            NumericScalar::F32(v) => *v as i32,
            NumericScalar::BF16(v) => v.to_f32() as i32,
            NumericScalar::F16(v) => v.to_f32() as i32,
            NumericScalar::U64(v) => *v as i32,
            NumericScalar::I64(v) => *v as i32,
            NumericScalar::U32(v) => *v as i32,
            NumericScalar::I32(v) => *v as i32,
            NumericScalar::U16(v) => *v as i32,
            NumericScalar::I16(v) => *v as i32,
            NumericScalar::U8(v) => *v as i32,
            NumericScalar::I8(v) => *v as i32,
            _ => panic!("Cannot cast from {:?} to i32", value),
        }
    }
}

impl NumericScalarType for u16 {
    fn to_numeric_scalar(self) -> NumericScalar {
        NumericScalar::U16(self)
    }
    fn cast_from_numeric_scalar(value: &NumericScalar) -> Self {
        match value {
            NumericScalar::F64(v) => *v as u16,
            NumericScalar::F32(v) => *v as u16,
            NumericScalar::BF16(v) => v.to_f32() as u16,
            NumericScalar::F16(v) => v.to_f32() as u16,
            NumericScalar::U64(v) => *v as u16,
            NumericScalar::I64(v) => *v as u16,
            NumericScalar::U32(v) => *v as u16,
            NumericScalar::I32(v) => *v as u16,
            NumericScalar::U16(v) => *v as u16,
            NumericScalar::I16(v) => *v as u16,
            NumericScalar::U8(v) => *v as u16,
            NumericScalar::I8(v) => *v as u16,
            _ => panic!("Cannot cast from {:?} to u16", value),
        }
    }
}

impl NumericScalarType for i16 {
    fn to_numeric_scalar(self) -> NumericScalar {
        NumericScalar::I16(self)
    }
    fn cast_from_numeric_scalar(value: &NumericScalar) -> Self {
        match value {
            NumericScalar::F64(v) => *v as i16,
            NumericScalar::F32(v) => *v as i16,
            NumericScalar::BF16(v) => v.to_f32() as i16,
            NumericScalar::F16(v) => v.to_f32() as i16,
            NumericScalar::U64(v) => *v as i16,
            NumericScalar::I64(v) => *v as i16,
            NumericScalar::U32(v) => *v as i16,
            NumericScalar::I32(v) => *v as i16,
            NumericScalar::U16(v) => *v as i16,
            NumericScalar::I16(v) => *v as i16,
            NumericScalar::U8(v) => *v as i16,
            NumericScalar::I8(v) => *v as i16,
            _ => panic!("Cannot cast from {:?} to i16", value),
        }
    }
}

impl NumericScalarType for u8 {
    fn to_numeric_scalar(self) -> NumericScalar {
        NumericScalar::U8(self)
    }
    fn cast_from_numeric_scalar(value: &NumericScalar) -> Self {
        match value {
            NumericScalar::F64(v) => *v as u8,
            NumericScalar::F32(v) => *v as u8,
            NumericScalar::BF16(v) => v.to_f32() as u8,
            NumericScalar::F16(v) => v.to_f32() as u8,
            NumericScalar::U64(v) => *v as u8,
            NumericScalar::I64(v) => *v as u8,
            NumericScalar::U32(v) => *v as u8,
            NumericScalar::I32(v) => *v as u8,
            NumericScalar::U16(v) => *v as u8,
            NumericScalar::I16(v) => *v as u8,
            NumericScalar::U8(v) => *v as u8,
            NumericScalar::I8(v) => *v as u8,
            _ => panic!("Cannot cast from {:?} to u8", value),
        }
    }
}

impl NumericScalarType for i8 {
    fn to_numeric_scalar(self) -> NumericScalar {
        NumericScalar::I8(self)
    }
    fn cast_from_numeric_scalar(value: &NumericScalar) -> Self {
        match value {
            NumericScalar::F64(v) => *v as i8,
            NumericScalar::F32(v) => *v as i8,
            NumericScalar::BF16(v) => v.to_f32() as i8,
            NumericScalar::F16(v) => v.to_f32() as i8,
            NumericScalar::U64(v) => *v as i8,
            NumericScalar::I64(v) => *v as i8,
            NumericScalar::U32(v) => *v as i8,
            NumericScalar::I32(v) => *v as i8,
            NumericScalar::U16(v) => *v as i8,
            NumericScalar::I16(v) => *v as i8,
            NumericScalar::U8(v) => *v as i8,
            NumericScalar::I8(v) => *v as i8,
            _ => panic!("Cannot cast from {:?} to i8", value),
        }
    }
}

impl NumericScalarType for bool {
    fn to_numeric_scalar(self) -> NumericScalar {
        NumericScalar::BOOL(self)
    }
    fn cast_from_numeric_scalar(value: &NumericScalar) -> Self {
        match value {
            NumericScalar::BOOL(v) => *v,
            _ => panic!("Cannot cast from {:?} to bool", value),
        }
    }
}

impl<T: NumericScalarType> From<T> for NumericScalar {
    fn from(value: T) -> Self {
        value.to_numeric_scalar()
    }
}

impl From<NumericScalar> for bool {
    fn from(value: NumericScalar) -> Self {
        Self::cast_from_numeric_scalar(&value)
    }
}

impl From<NumericScalar> for i64 {
    fn from(value: NumericScalar) -> Self {
        Self::cast_from_numeric_scalar(&value)
    }
}