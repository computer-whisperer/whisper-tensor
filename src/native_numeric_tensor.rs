use std::ops::Range;
use half::{bf16, f16};
use ndarray::{ArcArray,  Ix1, IxDyn, SliceInfo, SliceInfoElem};
use serde::{Deserialize, Serialize};
use crate::dtype::{DType};

#[derive(Debug, thiserror::Error)]
pub enum NativeNumericTensorError {
    #[error("Requested dtype {0}, but had dtype {1}")]
    WrongDTypeError(DType, DType),
    #[error("Cannot reshape tensor from {0:?} to {1:?}")]
    InvalidReshapeError(Vec<usize>, Vec<usize>),
    #[error(transparent)]
    ShapeError(#[from] ndarray::ShapeError)
}


#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NativeNumericTensor {
    F32(ArcArray<f32, IxDyn>),
    F64(ArcArray<f64, IxDyn>),
    F16(ArcArray<f16, IxDyn>),
    BF16(ArcArray<bf16, IxDyn>),
    U32(ArcArray<u32, IxDyn>),
    I32(ArcArray<i32, IxDyn>),
    U64(ArcArray<u64, IxDyn>),
    I64(ArcArray<i64, IxDyn>),
    U16(ArcArray<u16, IxDyn>),
    U8(ArcArray<u8, IxDyn>),
}

enum CmpType {
    Min,
    Max,
}

impl NativeNumericTensor {

    pub fn dtype(&self) -> DType {
        match self{
            NativeNumericTensor::F32(_) => DType::F32,
            NativeNumericTensor::F64(_) => DType::F64,
            NativeNumericTensor::F16(_) => DType::F16,
            NativeNumericTensor::BF16(_) => DType::BF16,
            NativeNumericTensor::U32(_) => DType::U32,
            NativeNumericTensor::I32(_) => DType::I32,
            NativeNumericTensor::U64(_) => DType::U64,
            NativeNumericTensor::I64(_) => DType::I64,
            NativeNumericTensor::U16(_) => DType::U16,
            NativeNumericTensor::U8(_) => DType::U8,
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            NativeNumericTensor::F32(x) => x.shape(),
            NativeNumericTensor::F64(x) => x.shape(),
            NativeNumericTensor::F16(x) => x.shape(),
            NativeNumericTensor::BF16(x) => x.shape(),
            NativeNumericTensor::U32(x) => x.shape(),
            NativeNumericTensor::I32(x) => x.shape(),
            NativeNumericTensor::U64(x) => x.shape(),
            NativeNumericTensor::I64(x) => x.shape(),
            NativeNumericTensor::U16(x) => x.shape(),
            NativeNumericTensor::U8(x) => x.shape()
        }
    }

    pub fn from_raw_data(data: &[u8], dtype: DType, shape: Vec<usize>) -> Result<Self, NativeNumericTensorError> {
        Ok(match dtype {
            DType::F64 => {
                let data: Vec<_> = data.chunks_exact(8).map(|x| f64::from_le_bytes(x.try_into().unwrap())).collect();
                NativeNumericTensor::F64(ArcArray::<_, IxDyn>::from_shape_vec(shape.clone(), data)?)
            },
            DType::F32 => {
                let data = data.chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect();
                NativeNumericTensor::F32(ArcArray::<_, IxDyn>::from_shape_vec(shape, data)?)
            },
            DType::BF16 => {
                let data = data.chunks_exact(2).map(|x| half::bf16::from_bits(u16::from_le_bytes(x.try_into().unwrap()))).collect();
                NativeNumericTensor::BF16(ArcArray::<_, IxDyn>::from_shape_vec(shape, data)?)
            }
            DType::F16 => {
                let data = data.chunks_exact(2).map(|x| half::f16::from_bits(u16::from_le_bytes(x.try_into().unwrap()))).collect();
                NativeNumericTensor::F16(ArcArray::<_, IxDyn>::from_shape_vec(shape, data)?)
            }
            DType::I64 => {
                let data = data.chunks_exact(8).map(|x| i64::from_le_bytes(x.try_into().unwrap())).collect();
                NativeNumericTensor::I64(ArcArray::<_, IxDyn>::from_shape_vec(shape, data)?)
            },
            DType::I32 => {
                let data = data.chunks_exact(4).map(|x| i32::from_le_bytes(x.try_into().unwrap())).collect();
                NativeNumericTensor::I32(ArcArray::<_, IxDyn>::from_shape_vec(shape, data)?)
            },
            DType::U64 => {
                let data = data.chunks_exact(8).map(|x| u64::from_le_bytes(x.try_into().unwrap())).collect();
                NativeNumericTensor::U64(ArcArray::<_, IxDyn>::from_shape_vec(shape, data)?)
            },
            DType::U32 => {
                let data = data.chunks_exact(4).map(|x| u32::from_le_bytes(x.try_into().unwrap())).collect();
                NativeNumericTensor::U32(ArcArray::<_, IxDyn>::from_shape_vec(shape, data)?)
            },
            DType::U16 => {
                let data = data.chunks_exact(2).map(|x| u16::from_le_bytes(x.try_into().unwrap())).collect();
                NativeNumericTensor::U16(ArcArray::<_, IxDyn>::from_shape_vec(shape, data)?)
            }
            DType::U8 => {
                NativeNumericTensor::U8(ArcArray::<_, IxDyn>::from_shape_vec(shape, data.to_vec())?)
            }
        })
    }

    pub fn argmax(&self, axis: usize) -> (Self, Self) {
        todo!()
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, NativeNumericTensorError> {
        match self {
            NativeNumericTensor::F32(x) => Ok(NativeNumericTensor::F32(x.clone().into_shape_with_order(new_shape)?)),
            NativeNumericTensor::F64(x) => Ok(NativeNumericTensor::F64(x.clone().into_shape_with_order(new_shape)?)),
            NativeNumericTensor::F16(x) => Ok(NativeNumericTensor::F16(x.clone().into_shape_with_order(new_shape)?)),
            NativeNumericTensor::BF16(x) => Ok(NativeNumericTensor::BF16(x.clone().into_shape_with_order(new_shape)?)),
            NativeNumericTensor::U32(x) => Ok(NativeNumericTensor::U32(x.clone().into_shape_with_order(new_shape)?)),
            NativeNumericTensor::I32(x) => Ok(NativeNumericTensor::I32(x.clone().into_shape_with_order(new_shape)?)),
            NativeNumericTensor::U64(x) => Ok(NativeNumericTensor::U64(x.clone().into_shape_with_order(new_shape)?)),
            NativeNumericTensor::I64(x) => Ok(NativeNumericTensor::I64(x.clone().into_shape_with_order(new_shape)?)),
            NativeNumericTensor::U16(x) => Ok(NativeNumericTensor::U16(x.clone().into_shape_with_order(new_shape)?)),
            NativeNumericTensor::U8(x) => Ok(NativeNumericTensor::U8(x.clone().into_shape_with_order(new_shape)?)),
        }
    }

    pub fn unsqueeze(&self, p0: usize) -> Result<Self, NativeNumericTensorError> {
        let mut s = self.shape().to_vec();
        s.insert(p0, 1);
        self.reshape(s)
    }

    pub fn squeeze(&self, p0: usize) -> Result<Self, NativeNumericTensorError> {
        let mut s = self.shape().to_vec();
        s.remove(p0);
        self.reshape(s)
    }

    pub fn slice(&self, indices: &[Range<usize>]) -> Result<Self, NativeNumericTensorError> {
        let s: SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn> = SliceInfo::try_from(indices.iter().map(|x| SliceInfoElem::from(x.clone())).collect::<Vec<_>>())?;
        Ok(match self {
            NativeNumericTensor::F32(x) => {NativeNumericTensor::F32(x.slice(s).to_shared())}
            NativeNumericTensor::F64(x) => {NativeNumericTensor::F64(x.slice(s).to_shared())}
            NativeNumericTensor::F16(x) => {NativeNumericTensor::F16(x.slice(s).to_shared())}
            NativeNumericTensor::BF16(x) => {NativeNumericTensor::BF16(x.slice(s).to_shared())}
            NativeNumericTensor::U32(x) => {NativeNumericTensor::U32(x.slice(s).to_shared())}
            NativeNumericTensor::I32(x) => {NativeNumericTensor::I32(x.slice(s).to_shared())}
            NativeNumericTensor::U64(x) => {NativeNumericTensor::U64(x.slice(s).to_shared())}
            NativeNumericTensor::I64(x) => {NativeNumericTensor::I64(x.slice(s).to_shared())}
            NativeNumericTensor::U16(x) => {NativeNumericTensor::U16(x.slice(s).to_shared())}
            NativeNumericTensor::U8(x) => {NativeNumericTensor::U8(x.slice(s).to_shared())}
        })
    }
}

pub trait FromVecShape<T>: Sized {
    fn from_vec_shape(v: Vec<T>, s: Vec<usize>) -> Result<Self, NativeNumericTensorError>;
}

impl FromVecShape<f32> for NativeNumericTensor {
    fn from_vec_shape(v: Vec<f32>, s: Vec<usize>) -> Result<Self, NativeNumericTensorError> {
        Ok(NativeNumericTensor::F32(ArcArray::<_, IxDyn>::from_shape_vec(s, v)?))
    }
}

impl FromVecShape<f64> for NativeNumericTensor {
    fn from_vec_shape(v: Vec<f64>, s: Vec<usize>) -> Result<Self, NativeNumericTensorError> {
        Ok(NativeNumericTensor::F64(ArcArray::<_, IxDyn>::from_shape_vec(s, v)?))
    }
}

impl FromVecShape<u32> for NativeNumericTensor {
    fn from_vec_shape(v: Vec<u32>, s: Vec<usize>) -> Result<Self, NativeNumericTensorError> {
        Ok(NativeNumericTensor::U32(ArcArray::<_, IxDyn>::from_shape_vec(s, v)?))
    }
}

impl FromVecShape<u64> for NativeNumericTensor {
    fn from_vec_shape(v: Vec<u64>, s: Vec<usize>) -> Result<Self, NativeNumericTensorError> {
        Ok(NativeNumericTensor::U64(ArcArray::<_, IxDyn>::from_shape_vec(s, v)?))
    }
}

impl From<Vec::<f32>> for NativeNumericTensor {
    fn from(v: Vec::<f32>) -> Self {
        NativeNumericTensor::F32(ArcArray::from_vec(v).into_dyn())
    }
}

impl From<Vec::<f64>> for NativeNumericTensor {
    fn from(v: Vec::<f64>) -> Self {
        NativeNumericTensor::F64(ArcArray::from_vec(v).into_dyn())
    }
}

impl From<Vec::<half::f16>> for NativeNumericTensor {
    fn from(v: Vec::<half::f16>) -> Self {
        NativeNumericTensor::F16(ArcArray::from_vec(v).into_dyn())
    }
}

impl From<Vec::<half::bf16>> for NativeNumericTensor {
    fn from(v: Vec::<half::bf16>) -> Self {
        NativeNumericTensor::BF16(ArcArray::from_vec(v).into_dyn())
    }
}

impl From<Vec::<u32>> for NativeNumericTensor {
    fn from(v: Vec::<u32>) -> Self {
        NativeNumericTensor::U32(ArcArray::from_vec(v).into_dyn())
    }
}

impl From<Vec::<i32>> for NativeNumericTensor {
    fn from(v: Vec::<i32>) -> Self {
        NativeNumericTensor::I32(ArcArray::from_vec(v).into_dyn())
    }
}

impl From<Vec::<u64>> for NativeNumericTensor {
    fn from(v: Vec::<u64>) -> Self {
        NativeNumericTensor::U64(ArcArray::from_vec(v).into_dyn())
    }
}

impl From<Vec::<i64>> for NativeNumericTensor {
    fn from(v: Vec::<i64>) -> Self {
        NativeNumericTensor::I64(ArcArray::from_vec(v).into_dyn())
    }
}

impl From<Vec::<u16>> for NativeNumericTensor {
    fn from(v: Vec::<u16>) -> Self {
        NativeNumericTensor::U16(ArcArray::from_vec(v).into_dyn())
    }
}

impl From<Vec::<u8>> for NativeNumericTensor {
    fn from(v: Vec::<u8>) -> Self {
        NativeNumericTensor::U8(ArcArray::from_vec(v).into_dyn())
    }
}

impl core::fmt::Display for NativeNumericTensor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub trait TryToSlice<T> {
    fn try_to_slice(&self) -> Result<&[T], NativeNumericTensorError>;
}

impl TryFrom<NativeNumericTensor> for Vec<u32>
{
    type Error = NativeNumericTensorError;
    fn try_from(value: NativeNumericTensor) -> Result<Vec<u32>, NativeNumericTensorError> {
        assert_eq!(value.shape().len(), 1);
        if let NativeNumericTensor::U32(x) = value {
            Ok(x.into_dimensionality::<Ix1>()?.to_vec())
        }
        else {
            Err(NativeNumericTensorError::WrongDTypeError(DType::U32, value.dtype()))
        }
    }
}

impl TryFrom<NativeNumericTensor> for Vec<f32>
{
    type Error = NativeNumericTensorError;
    fn try_from(value: NativeNumericTensor) -> Result<Vec<f32>, NativeNumericTensorError> {
        assert_eq!(value.shape().len(), 1);
        if let NativeNumericTensor::F32(x) = value {
            Ok(x.into_dimensionality::<Ix1>()?.to_vec())
        }
        else {
            Err(NativeNumericTensorError::WrongDTypeError(DType::F32, value.dtype()))
        }
    }
}

impl TryToSlice<u32> for NativeNumericTensor {
    fn try_to_slice(&self) -> Result<&[u32], NativeNumericTensorError> {
        assert_eq!(self.shape().len(), 1);
        if let NativeNumericTensor::U32(x) = &self {
            Ok(x.as_slice_memory_order().unwrap())
        }
        else {
            Err(NativeNumericTensorError::WrongDTypeError(DType::U32, self.dtype()))
        }
    }
}

impl TryToSlice<f32> for NativeNumericTensor {
    fn try_to_slice(&self) -> Result<&[f32], NativeNumericTensorError> {
        assert_eq!(self.shape().len(), 1);
        if let NativeNumericTensor::F32(x) = &self {
            Ok(x.as_slice_memory_order().unwrap())
        }
        else {
            Err(NativeNumericTensorError::WrongDTypeError(DType::F32, self.dtype()))
        }
    }
}

