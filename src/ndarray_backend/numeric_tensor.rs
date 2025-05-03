use std::ops::Range;
use half::{bf16, f16};
use ndarray::{ArcArray, IxDyn, SliceInfo, SliceInfoElem};
use num_traits::Pow;
use serde::{Deserialize, Serialize};
use crate::dtype::DType;
use crate::ndarray_backend::conversions::{FromScalarShape, FromVecShape, TryToSlice};
use super::ops;
use super::ops::{NativeNumericTensorBinaryOperation, NativeNumericTensorUnaryOperation, ReduceOp};

#[derive(Debug, thiserror::Error)]
pub enum NDArrayNumericTensorError {
    #[error("Requested dtype {0}, but had dtype {1}")]
    WrongDTypeError(DType, DType),
    #[error("Cannot reshape tensor from {0:?} to {1:?}")]
    InvalidReshapeError(Vec<usize>, Vec<usize>),
    #[error("Unsupported operation {0} for dtypes {1:?}")]
    UnsupportedOperationForDTypes(String, Vec<DType>),
    #[error("Cannot cast from {0} to {1}")]
    InvalidCastOperation(DType, DType),
    #[error(transparent)]
    NDArrayOperationError(#[from] ops::NDArrayOperationError),
    #[error(transparent)]
    ShapeError(#[from] ndarray::ShapeError),
    #[error("other error")]
    OtherError
}


#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NDArrayNumericTensor {
    F64(ArcArray<f64, IxDyn>),
    F32(ArcArray<f32, IxDyn>),
    BF16(ArcArray<bf16, IxDyn>),
    F16(ArcArray<f16, IxDyn>),
    U64(ArcArray<u64, IxDyn>),
    I64(ArcArray<i64, IxDyn>),
    U32(ArcArray<u32, IxDyn>),
    I32(ArcArray<i32, IxDyn>),
    U16(ArcArray<u16, IxDyn>),
    I16(ArcArray<i16, IxDyn>),
    U8(ArcArray<u8, IxDyn>),
    I8(ArcArray<i8, IxDyn>),
    BOOL(ArcArray<bool, IxDyn>),
}



impl NDArrayNumericTensor {

    pub fn rank(&self) -> usize {
        self.shape().len()
    }

    pub fn dtype(&self) -> DType {
        match self{
            NDArrayNumericTensor::F64(_) => DType::F64,
            NDArrayNumericTensor::F32(_) => DType::F32,
            NDArrayNumericTensor::BF16(_) => DType::BF16,
            NDArrayNumericTensor::F16(_) => DType::F16,
            NDArrayNumericTensor::U64(_) => DType::U64,
            NDArrayNumericTensor::I64(_) => DType::I64,
            NDArrayNumericTensor::U32(_) => DType::U32,
            NDArrayNumericTensor::I32(_) => DType::I32,
            NDArrayNumericTensor::U16(_) => DType::U16,
            NDArrayNumericTensor::I16(_) => DType::I16,
            NDArrayNumericTensor::U8(_) => DType::U8,
            NDArrayNumericTensor::I8(_) => DType::I8,
            NDArrayNumericTensor::BOOL(_) => DType::BOOL
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            NDArrayNumericTensor::F32(x) => x.shape(),
            NDArrayNumericTensor::F64(x) => x.shape(),
            NDArrayNumericTensor::F16(x) => x.shape(),
            NDArrayNumericTensor::BF16(x) => x.shape(),
            NDArrayNumericTensor::U32(x) => x.shape(),
            NDArrayNumericTensor::I32(x) => x.shape(),
            NDArrayNumericTensor::U64(x) => x.shape(),
            NDArrayNumericTensor::I64(x) => x.shape(),
            NDArrayNumericTensor::U16(x) => x.shape(),
            NDArrayNumericTensor::I16(x) => x.shape(),
            NDArrayNumericTensor::U8(x) => x.shape(),
            NDArrayNumericTensor::I8(x) => x.shape(),
            NDArrayNumericTensor::BOOL(x) => x.shape()
        }
    }

    pub fn from_raw_data(data: &[u8], dtype: DType, shape: Vec<usize>) -> Result<Self, NDArrayNumericTensorError> {
        Ok(match dtype {
            DType::F64 => {
                let data: Vec<_> = data.chunks_exact(8).map(|x| f64::from_le_bytes(x.try_into().unwrap())).collect();
                NDArrayNumericTensor::F64(ArcArray::<_, IxDyn>::from_shape_vec(shape.clone(), data)?)
            },
            DType::F32 => {
                let data = data.chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect();
                NDArrayNumericTensor::F32(ArcArray::<_, IxDyn>::from_shape_vec(shape, data)?)
            },
            DType::BF16 => {
                let data = data.chunks_exact(2).map(|x| bf16::from_bits(u16::from_le_bytes(x.try_into().unwrap()))).collect();
                NDArrayNumericTensor::BF16(ArcArray::<_, IxDyn>::from_shape_vec(shape, data)?)
            }
            DType::F16 => {
                let data = data.chunks_exact(2).map(|x| f16::from_bits(u16::from_le_bytes(x.try_into().unwrap()))).collect();
                NDArrayNumericTensor::F16(ArcArray::<_, IxDyn>::from_shape_vec(shape, data)?)
            }
            DType::I64 => {
                let data = data.chunks_exact(8).map(|x| i64::from_le_bytes(x.try_into().unwrap())).collect();
                NDArrayNumericTensor::I64(ArcArray::<_, IxDyn>::from_shape_vec(shape, data)?)
            },
            DType::I32 => {
                let data = data.chunks_exact(4).map(|x| i32::from_le_bytes(x.try_into().unwrap())).collect();
                NDArrayNumericTensor::I32(ArcArray::<_, IxDyn>::from_shape_vec(shape, data)?)
            },
            DType::U64 => {
                let data = data.chunks_exact(8).map(|x| u64::from_le_bytes(x.try_into().unwrap())).collect();
                NDArrayNumericTensor::U64(ArcArray::<_, IxDyn>::from_shape_vec(shape, data)?)
            },
            DType::U32 => {
                let data = data.chunks_exact(4).map(|x| u32::from_le_bytes(x.try_into().unwrap())).collect();
                NDArrayNumericTensor::U32(ArcArray::<_, IxDyn>::from_shape_vec(shape, data)?)
            },
            DType::I16 => {
                let data = data.chunks_exact(2).map(|x| i16::from_le_bytes(x.try_into().unwrap())).collect();
                NDArrayNumericTensor::I16(ArcArray::<_, IxDyn>::from_shape_vec(shape, data)?)
            }
            DType::U16 => {
                let data = data.chunks_exact(2).map(|x| u16::from_le_bytes(x.try_into().unwrap())).collect();
                NDArrayNumericTensor::U16(ArcArray::<_, IxDyn>::from_shape_vec(shape, data)?)
            }
            DType::U8 => {
                NDArrayNumericTensor::U8(ArcArray::<_, IxDyn>::from_shape_vec(shape, data.to_vec())?)
            }
            DType::I8 => {
                let data = data.into_iter().map(|x| *x as i8).collect();
                NDArrayNumericTensor::I8(ArcArray::<_, IxDyn>::from_shape_vec(shape, data)?)
            }
            DType::BOOL => {
                let data = data.into_iter().map(|x| *x != 0).collect();
                NDArrayNumericTensor::BOOL(ArcArray::<_, IxDyn>::from_shape_vec(shape, data)?)
            }
        })
    }

    pub fn from_f32_data(data: &[f32], dtype: DType, shape: Vec<usize>) -> Result<Self, NDArrayNumericTensorError> {
        match dtype {
            DType::F32 => Ok(NDArrayNumericTensor::F32(ArcArray::<_, IxDyn>::from_shape_vec(shape, data.to_vec())?)),
            _ => Err(NDArrayNumericTensorError::OtherError)
        }
    }

    pub fn from_f64_data(data: &[f64], dtype: DType, shape: Vec<usize>) -> Result<Self, NDArrayNumericTensorError> {
        match dtype {
            DType::F64 => Ok(NDArrayNumericTensor::F64(ArcArray::<_, IxDyn>::from_shape_vec(shape, data.to_vec())?)),
            _ => Err(NDArrayNumericTensorError::OtherError)
        }
    }

    pub fn argmax(&self, axis: usize) -> (Self, Self) {
        todo!()
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, NDArrayNumericTensorError> {
        match self {
            NDArrayNumericTensor::F32(x) => Ok(NDArrayNumericTensor::F32(ops::reshape(x.clone(), &new_shape)?)),
            NDArrayNumericTensor::F64(x) => Ok(NDArrayNumericTensor::F64(ops::reshape(x.clone(), &new_shape)?)),
            NDArrayNumericTensor::F16(x) => Ok(NDArrayNumericTensor::F16(ops::reshape(x.clone(), &new_shape)?)),
            NDArrayNumericTensor::BF16(x) => Ok(NDArrayNumericTensor::BF16(ops::reshape(x.clone(), &new_shape)?)),
            NDArrayNumericTensor::U32(x) => Ok(NDArrayNumericTensor::U32(ops::reshape(x.clone(), &new_shape)?)),
            NDArrayNumericTensor::I32(x) => Ok(NDArrayNumericTensor::I32(ops::reshape(x.clone(), &new_shape)?)),
            NDArrayNumericTensor::U64(x) => Ok(NDArrayNumericTensor::U64(ops::reshape(x.clone(), &new_shape)?)),
            NDArrayNumericTensor::I64(x) => Ok(NDArrayNumericTensor::I64(ops::reshape(x.clone(), &new_shape)?)),
            NDArrayNumericTensor::U16(x) => Ok(NDArrayNumericTensor::U16(ops::reshape(x.clone(), &new_shape)?)),
            NDArrayNumericTensor::I16(x) => Ok(NDArrayNumericTensor::I16(ops::reshape(x.clone(), &new_shape)?)),
            NDArrayNumericTensor::U8(x) => Ok(NDArrayNumericTensor::U8(ops::reshape(x.clone(), &new_shape)?)),
            NDArrayNumericTensor::I8(x) => Ok(NDArrayNumericTensor::I8(ops::reshape(x.clone(), &new_shape)?)),
            NDArrayNumericTensor::BOOL(x) => Ok(NDArrayNumericTensor::BOOL(ops::reshape(x.clone(), &new_shape)?)),
        }
    }

    pub fn unsqueeze(&self, p0: usize) -> Result<Self, NDArrayNumericTensorError> {
        let mut s = self.shape().to_vec();
        s.insert(p0, 1);
        self.reshape(s)
    }

    pub fn squeeze(&self, p0: usize) -> Result<Self, NDArrayNumericTensorError> {
        let mut s = self.shape().to_vec();
        s.remove(p0);
        self.reshape(s)
    }

    pub fn flatten(&self) -> Result<Self, NDArrayNumericTensorError> {
        Ok(match self {
            NDArrayNumericTensor::F32(x) => {NDArrayNumericTensor::F32(x.flatten().to_shared().into_dyn())},
            NDArrayNumericTensor::I64(x) => {NDArrayNumericTensor::I64(x.flatten().to_shared().into_dyn())},
            NDArrayNumericTensor::F64(x) => {NDArrayNumericTensor::F64(x.flatten().to_shared().into_dyn())},
            NDArrayNumericTensor::BF16(x) => {NDArrayNumericTensor::BF16(x.flatten().to_shared().into_dyn())},
            NDArrayNumericTensor::F16(x) => {NDArrayNumericTensor::F16(x.flatten().to_shared().into_dyn())},
            NDArrayNumericTensor::U64(x) => {NDArrayNumericTensor::U64(x.flatten().to_shared().into_dyn())},
            NDArrayNumericTensor::U32(x) => {NDArrayNumericTensor::U32(x.flatten().to_shared().into_dyn())},
            NDArrayNumericTensor::I32(x) => {NDArrayNumericTensor::I32(x.flatten().to_shared().into_dyn())},
            NDArrayNumericTensor::U16(x) => {NDArrayNumericTensor::U16(x.flatten().to_shared().into_dyn())},
            NDArrayNumericTensor::I16(x) => {NDArrayNumericTensor::I16(x.flatten().to_shared().into_dyn())},
            NDArrayNumericTensor::U8(x) => {NDArrayNumericTensor::U8(x.flatten().to_shared().into_dyn())},
            NDArrayNumericTensor::I8(x) => {NDArrayNumericTensor::I8(x.flatten().to_shared().into_dyn())},
            NDArrayNumericTensor::BOOL(x) => {NDArrayNumericTensor::BOOL(x.flatten().to_shared().into_dyn())},
        })
    }

    pub fn slice(&self, indices: &[Range<usize>]) -> Result<Self, NDArrayNumericTensorError> {
        let s: SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn> = SliceInfo::try_from(indices.iter().map(|x| SliceInfoElem::from(x.clone())).collect::<Vec<_>>())?;
        Ok(match self {
            NDArrayNumericTensor::F32(x) => { NDArrayNumericTensor::F32(x.slice(s).to_shared())}
            NDArrayNumericTensor::F64(x) => { NDArrayNumericTensor::F64(x.slice(s).to_shared())}
            NDArrayNumericTensor::F16(x) => { NDArrayNumericTensor::F16(x.slice(s).to_shared())}
            NDArrayNumericTensor::BF16(x) => { NDArrayNumericTensor::BF16(x.slice(s).to_shared())}
            NDArrayNumericTensor::U32(x) => { NDArrayNumericTensor::U32(x.slice(s).to_shared())}
            NDArrayNumericTensor::I32(x) => { NDArrayNumericTensor::I32(x.slice(s).to_shared())}
            NDArrayNumericTensor::U64(x) => { NDArrayNumericTensor::U64(x.slice(s).to_shared())}
            NDArrayNumericTensor::I64(x) => { NDArrayNumericTensor::I64(x.slice(s).to_shared())}
            NDArrayNumericTensor::U16(x) => { NDArrayNumericTensor::U16(x.slice(s).to_shared())}
            NDArrayNumericTensor::I16(x) => { NDArrayNumericTensor::I16(x.slice(s).to_shared())}
            NDArrayNumericTensor::U8(x) => { NDArrayNumericTensor::U8(x.slice(s).to_shared())}
            NDArrayNumericTensor::I8(x) => { NDArrayNumericTensor::I8(x.slice(s).to_shared())}
            NDArrayNumericTensor::BOOL(x) => { NDArrayNumericTensor::BOOL(x.slice(s).to_shared())}
        })
    }

    fn try_unary_op(a: &NDArrayNumericTensor, op: NativeNumericTensorUnaryOperation) -> Result<NDArrayNumericTensor, NDArrayNumericTensorError> {
        let v = match a {
            NDArrayNumericTensor::F32(a) => {
                Some(NDArrayNumericTensor::F32(op.apply(a.clone())?))
            }
            NDArrayNumericTensor::F64(a) => {
                Some(NDArrayNumericTensor::F64(op.apply(a.clone())?))
            }
            NDArrayNumericTensor::F16(a) => {
                Some(NDArrayNumericTensor::F16(op.apply(a.clone())?))
            }
            NDArrayNumericTensor::BF16(a) => {
                Some(NDArrayNumericTensor::BF16(op.apply(a.clone())?))
            }
            _ => None,
        };
        v.ok_or(NDArrayNumericTensorError::UnsupportedOperationForDTypes(op.to_string(), vec![a.dtype()]))
    }

    fn try_binary_op(a: &NDArrayNumericTensor, b: &NDArrayNumericTensor, op: NativeNumericTensorBinaryOperation) -> Result<NDArrayNumericTensor, NDArrayNumericTensorError> {
        let result = match a {
            NDArrayNumericTensor::F32(a) => {
                if let NDArrayNumericTensor::F32(b) = b {
                    Some(NDArrayNumericTensor::F32(op.apply(a.clone(), b.clone())?))
                } else {
                    None
                }
            }
            NDArrayNumericTensor::F64(a) => {
                if let NDArrayNumericTensor::F64(b) = b {
                    Some(NDArrayNumericTensor::F64(op.apply(a.clone(), b.clone())?))
                } else {
                    None
                }
            }
            NDArrayNumericTensor::F16(a) => {
                if let NDArrayNumericTensor::F16(b) = b {
                    Some(NDArrayNumericTensor::F16(op.apply(a.clone(), b.clone())?))
                } else {
                    None
                }
            }
            NDArrayNumericTensor::BF16(a) => {
                if let NDArrayNumericTensor::BF16(b) = b {
                    Some(NDArrayNumericTensor::BF16(op.apply(a.clone(), b.clone())?))
                } else {
                    None
                }
            }
            NDArrayNumericTensor::U32(a) => {
                if let NDArrayNumericTensor::U32(b) = b {
                    Some(NDArrayNumericTensor::U32(op.apply(a.clone(), b.clone())?))
                } else {
                    None
                }
            }
            NDArrayNumericTensor::I32(a) => {
                if let NDArrayNumericTensor::I32(b) = b {
                    Some(NDArrayNumericTensor::I32(op.apply(a.clone(), b.clone())?))
                } else {
                    None
                }
            }
            NDArrayNumericTensor::U64(a) => {
                if let NDArrayNumericTensor::U64(b) = b {
                    Some(NDArrayNumericTensor::U64(op.apply(a.clone(), b.clone())?))
                } else {
                    None
                }
            }
            NDArrayNumericTensor::I64(a) => {
                if let NDArrayNumericTensor::I64(b) = b {
                    Some(NDArrayNumericTensor::I64(op.apply(a.clone(), b.clone())?))
                } else {
                    None
                }
            }
            NDArrayNumericTensor::U16(a) => {
                if let NDArrayNumericTensor::U16(b) = b {
                    Some(NDArrayNumericTensor::U16(op.apply(a.clone(), b.clone())?))
                } else {
                    None
                }
            }
            NDArrayNumericTensor::I16(a) => {
                if let NDArrayNumericTensor::I16(b) = b {
                    Some(NDArrayNumericTensor::I16(op.apply(a.clone(), b.clone())?))
                } else {
                    None
                }
            }
            NDArrayNumericTensor::U8(a) => {
                if let NDArrayNumericTensor::U8(b) = b {
                    Some(NDArrayNumericTensor::U8(op.apply(a.clone(), b.clone())?))
                } else {
                    None
                }
            }
            NDArrayNumericTensor::I8(a) => {
                if let NDArrayNumericTensor::I8(b) = b {
                    Some(NDArrayNumericTensor::I8(op.apply(a.clone(), b.clone())?))
                } else {
                    None
                }
            }
            NDArrayNumericTensor::BOOL(_) => {
                None
            }
        };
        if let Some(result) = result {
            Ok(result)
        } else {
            Err(NDArrayNumericTensorError::UnsupportedOperationForDTypes(op.to_string(), vec![a.dtype(), b.dtype()]))
        }
    }

    pub fn add(a: &NDArrayNumericTensor, b: &NDArrayNumericTensor) -> Result<NDArrayNumericTensor, NDArrayNumericTensorError> {
        Self::try_binary_op(a, b, NativeNumericTensorBinaryOperation::Add)
    }

    pub fn add_f32(a: &NDArrayNumericTensor, b: f32) -> Result<NDArrayNumericTensor, NDArrayNumericTensorError> {
        Ok(match a {
            NDArrayNumericTensor::F64(x) => NDArrayNumericTensor::F64(x.map(|x| *x + b as f64).to_shared()),
            NDArrayNumericTensor::F32(x) => NDArrayNumericTensor::F32(x.map(|x| *x + b).to_shared()),
            NDArrayNumericTensor::BF16(x) => NDArrayNumericTensor::BF16(x.map(|x| bf16::from_f32(x.to_f32() + b)).to_shared()),
            NDArrayNumericTensor::F16(x) => NDArrayNumericTensor::F16(x.map(|x| f16::from_f32(x.to_f32() + b)).to_shared()),
            _ => return Err(NDArrayNumericTensorError::UnsupportedOperationForDTypes("add_f32".to_string(), vec![a.dtype()])),
        })
    }

    pub fn mul_f32(a: &NDArrayNumericTensor, b: f32) -> Result<NDArrayNumericTensor, NDArrayNumericTensorError> {
        Ok(match a {
            NDArrayNumericTensor::F64(x) => NDArrayNumericTensor::F64(x.map(|x| *x * b as f64).to_shared()),
            NDArrayNumericTensor::F32(x) => NDArrayNumericTensor::F32(x.map(|x| *x * b).to_shared()),
            NDArrayNumericTensor::BF16(x) => NDArrayNumericTensor::BF16(x.map(|x| bf16::from_f32(x.to_f32() * b)).to_shared()),
            NDArrayNumericTensor::F16(x) => NDArrayNumericTensor::F16(x.map(|x| f16::from_f32(x.to_f32() * b)).to_shared()),
            _ => return Err(NDArrayNumericTensorError::UnsupportedOperationForDTypes("mul_f32".to_string(), vec![a.dtype()])),
        })
    }

    pub fn div_f32(a: &NDArrayNumericTensor, b: f32) -> Result<NDArrayNumericTensor, NDArrayNumericTensorError> {
        Ok(match a {
            NDArrayNumericTensor::F64(x) => NDArrayNumericTensor::F64(x.map(|x| *x / b as f64).to_shared()),
            NDArrayNumericTensor::F32(x) => NDArrayNumericTensor::F32(x.map(|x| *x / b).to_shared()),
            NDArrayNumericTensor::BF16(x) => NDArrayNumericTensor::BF16(x.map(|x| bf16::from_f32(x.to_f32() / b)).to_shared()),
            NDArrayNumericTensor::F16(x) => NDArrayNumericTensor::F16(x.map(|x| f16::from_f32(x.to_f32() / b)).to_shared()),
            _ => return Err(NDArrayNumericTensorError::UnsupportedOperationForDTypes("add_f32".to_string(), vec![a.dtype()])),
        })
    }

    pub fn sub(a: &NDArrayNumericTensor, b: &NDArrayNumericTensor) -> Result<NDArrayNumericTensor, NDArrayNumericTensorError> {
        Self::try_binary_op(a, b, NativeNumericTensorBinaryOperation::Sub)
    }

    pub fn mul(a: &NDArrayNumericTensor, b: &NDArrayNumericTensor) -> Result<NDArrayNumericTensor, NDArrayNumericTensorError> {
        Self::try_binary_op(a, b, NativeNumericTensorBinaryOperation::Mul)
    }

    pub fn matmul(a: &NDArrayNumericTensor, b: &NDArrayNumericTensor) -> Result<NDArrayNumericTensor, NDArrayNumericTensorError> {
        Ok(match (a, b) {
            (NDArrayNumericTensor::F32(a), NDArrayNumericTensor::F32(b)) => NDArrayNumericTensor::F32(ops::matmul(a, b)?),
            (NDArrayNumericTensor::F64(a), NDArrayNumericTensor::F64(b)) => NDArrayNumericTensor::F64(ops::matmul(a, b)?),
            (NDArrayNumericTensor::BF16(a), NDArrayNumericTensor::BF16(b)) => NDArrayNumericTensor::BF16(ops::matmul(a, b)?),
            (NDArrayNumericTensor::F16(a), NDArrayNumericTensor::F16(b)) => NDArrayNumericTensor::F16(ops::matmul(a, b)?),
            (NDArrayNumericTensor::U64(a), NDArrayNumericTensor::U64(b)) => NDArrayNumericTensor::U64(ops::matmul(a, b)?),
            (NDArrayNumericTensor::I64(a), NDArrayNumericTensor::I64(b)) => NDArrayNumericTensor::I64(ops::matmul(a, b)?),
            (NDArrayNumericTensor::U32(a), NDArrayNumericTensor::U32(b)) => NDArrayNumericTensor::U32(ops::matmul(a, b)?),
            (NDArrayNumericTensor::I32(a), NDArrayNumericTensor::I32(b)) => NDArrayNumericTensor::I32(ops::matmul(a, b)?),
            _ => return Err(NDArrayNumericTensorError::UnsupportedOperationForDTypes("matmul".to_string(), vec![a.dtype(), b.dtype()])),
        })
    }

    pub fn div(a: &NDArrayNumericTensor, b: &NDArrayNumericTensor) -> Result<NDArrayNumericTensor, NDArrayNumericTensorError> {
        Self::try_binary_op(a, b, NativeNumericTensorBinaryOperation::Div)
    }

    pub fn neg(self) -> Result<Self, NDArrayNumericTensorError> {
        Ok(match self {
            NDArrayNumericTensor::F32(x) => NDArrayNumericTensor::F32(x.map(|x| -x).to_shared()),
            NDArrayNumericTensor::F64(x) => NDArrayNumericTensor::F64(x.map(|x| -x).to_shared()),
            NDArrayNumericTensor::BF16(x) => NDArrayNumericTensor::BF16(x.map(|x| -x).to_shared()),
            NDArrayNumericTensor::F16(x) => NDArrayNumericTensor::F16(x.map(|x| -x).to_shared()),
            NDArrayNumericTensor::I64(x) => NDArrayNumericTensor::I64(x.map(|x| -x).to_shared()),
            NDArrayNumericTensor::I32(x) => NDArrayNumericTensor::I32(x.map(|x| -x).to_shared()),
            _ => return Err(NDArrayNumericTensorError::UnsupportedOperationForDTypes("neg".to_string(), vec![self.dtype()])),
        })
    }

    pub fn abs(self) -> Result<Self, NDArrayNumericTensorError> {
        Ok(match self {
            NDArrayNumericTensor::F32(x) => NDArrayNumericTensor::F32(x.abs().to_shared()),
            NDArrayNumericTensor::F64(x) => NDArrayNumericTensor::F64(x.abs().to_shared()),
            NDArrayNumericTensor::BF16(x) => NDArrayNumericTensor::BF16(x.abs().to_shared()),
            NDArrayNumericTensor::F16(x) => NDArrayNumericTensor::F16(x.abs().to_shared()),
            NDArrayNumericTensor::I64(x) => NDArrayNumericTensor::I64(x.map(|x| x.abs()).to_shared()),
            NDArrayNumericTensor::U64(x) => NDArrayNumericTensor::U64(x),
            NDArrayNumericTensor::I32(x) => NDArrayNumericTensor::I32(x.map(|x| x.abs()).to_shared()),
            NDArrayNumericTensor::U32(x) => NDArrayNumericTensor::U32(x),
            NDArrayNumericTensor::I16(x) => NDArrayNumericTensor::I16(x.map(|x| x.abs()).to_shared()),
            NDArrayNumericTensor::U16(x) => NDArrayNumericTensor::U16(x),
            NDArrayNumericTensor::I8(x) => NDArrayNumericTensor::I8(x.map(|x| x.abs()).to_shared()),
            NDArrayNumericTensor::U8(x) => NDArrayNumericTensor::U8(x),
            _ => return Err(NDArrayNumericTensorError::UnsupportedOperationForDTypes("abs".to_string(), vec![self.dtype()])),
        })
    }

    pub fn clamp_min(self, min: f32) -> Result<Self, NDArrayNumericTensorError> {
        Ok(match self {
            NDArrayNumericTensor::F32(x) => NDArrayNumericTensor::F32(x.map(|x| x.max(min)).to_shared()),
            NDArrayNumericTensor::F64(x) => NDArrayNumericTensor::F64(x.map(|x| x.max(min as f64)).to_shared()),
            NDArrayNumericTensor::BF16(x) => NDArrayNumericTensor::BF16(x.map(|x| x.max(bf16::from_f32(min))).to_shared()),
            NDArrayNumericTensor::F16(x) => NDArrayNumericTensor::F16(x.map(|x| x.max(f16::from_f32(min))).to_shared()),
            NDArrayNumericTensor::I64(x) => NDArrayNumericTensor::I64(x.map(|x| (*x).max(min as i64)).to_shared()),
            NDArrayNumericTensor::I32(x) => NDArrayNumericTensor::I32(x.map(|x| (*x).max(min as i32)).to_shared()),
            NDArrayNumericTensor::I16(x) => NDArrayNumericTensor::I16(x.map(|x| (*x).max(min as i16)).to_shared()),
            NDArrayNumericTensor::I8(x) => NDArrayNumericTensor::I8(x.map(|x| (*x).max(min as i8)).to_shared()),
            _ => return Err(NDArrayNumericTensorError::UnsupportedOperationForDTypes("clamp_min".to_string(), vec![self.dtype()])),
        })
    }

    pub fn sigmoid(&self) -> Result<Self, NDArrayNumericTensorError> {
        Self::try_unary_op(self, NativeNumericTensorUnaryOperation::Sigmoid)
    }

    pub fn relu(&self) -> Result<Self, NDArrayNumericTensorError> {
        Ok(match self {
            NDArrayNumericTensor::F32(x) => NDArrayNumericTensor::F32(x.map(|x| x.max(0.0)).to_shared()),
            NDArrayNumericTensor::F64(x) => NDArrayNumericTensor::F64(x.map(|x| x.max(0.0)).to_shared()),
            NDArrayNumericTensor::BF16(x) => NDArrayNumericTensor::BF16(x.map(|x| x.max(bf16::ZERO)).to_shared()),
            NDArrayNumericTensor::F16(x) => NDArrayNumericTensor::F16(x.map(|x| x.max(f16::ZERO)).to_shared()),
            NDArrayNumericTensor::I64(x) => NDArrayNumericTensor::I64(x.map(|x| (*x).max(0i64)).to_shared()),
            NDArrayNumericTensor::I32(x) => NDArrayNumericTensor::I32(x.map(|x| (*x).max(0i32)).to_shared()),
            NDArrayNumericTensor::I16(x) => NDArrayNumericTensor::I16(x.map(|x| (*x).max(0i16)).to_shared()),
            NDArrayNumericTensor::I8(x) => NDArrayNumericTensor::I8(x.map(|x| (*x).max(0i8)).to_shared()),
            _ => return Err(NDArrayNumericTensorError::UnsupportedOperationForDTypes("relu".to_string(), vec![self.dtype()])),
        })
    }

    pub fn tanh(&self) -> Result<Self, NDArrayNumericTensorError> {
        Self::try_unary_op(self, NativeNumericTensorUnaryOperation::Tanh)
    }

    pub fn softplus(&self) -> Result<Self, NDArrayNumericTensorError> {
        Self::try_unary_op(self, NativeNumericTensorUnaryOperation::Softplus)
    }

    pub fn exp(&self) -> Result<Self, NDArrayNumericTensorError> {
        Self::try_unary_op(self, NativeNumericTensorUnaryOperation::Exp)
    }

    pub fn log(&self) -> Result<Self, NDArrayNumericTensorError> {
        Self::try_unary_op(self, NativeNumericTensorUnaryOperation::Log)
    }

    pub fn pow(&self, input_exponent: &NDArrayNumericTensor) -> Result<Self, NDArrayNumericTensorError> {
        Ok(match input_exponent {
            NDArrayNumericTensor::F32(exponent) => {
                match self {
                    NDArrayNumericTensor::F32(x) => NDArrayNumericTensor::F32(ops::pow(x.clone(), exponent.clone())?),
                    NDArrayNumericTensor::F64(x) => NDArrayNumericTensor::F64(ops::pow(x.clone(), exponent.clone())?),
                    NDArrayNumericTensor::F16(x) => NDArrayNumericTensor::F16(ops::pow(x.map(|x| (*x).to_f32()).to_shared(), exponent.clone())?.map(|x| f16::from_f32(*x)).to_shared()),
                    NDArrayNumericTensor::BF16(x) => NDArrayNumericTensor::BF16(ops::pow(x.map(|x| (*x).to_f32()).to_shared(), exponent.clone())?.map(|x| bf16::from_f32(*x)).to_shared()),
                    _ => Err(NDArrayNumericTensorError::UnsupportedOperationForDTypes("pow".to_string(), vec![self.dtype(), input_exponent.dtype()]))?,
                }
            },
            NDArrayNumericTensor::F64(exponent) => {
                match self {
                    NDArrayNumericTensor::F32(x) => NDArrayNumericTensor::F32(ops::pow(x.map(|x| *x as f64).to_shared(), exponent.clone())?.map(|x| *x as f32).to_shared()),
                    NDArrayNumericTensor::F64(x) => NDArrayNumericTensor::F64(ops::pow(x.clone(), exponent.clone())?),
                    NDArrayNumericTensor::F16(x) => NDArrayNumericTensor::F16(ops::pow(x.map(|x| (*x).to_f64()).to_shared(), exponent.clone())?.map(|x| f16::from_f64(*x)).to_shared()),
                    NDArrayNumericTensor::BF16(x) => NDArrayNumericTensor::BF16(ops::pow(x.map(|x| (*x).to_f64()).to_shared(), exponent.clone())?.map(|x| bf16::from_f64(*x)).to_shared()),
                    _ => Err(NDArrayNumericTensorError::UnsupportedOperationForDTypes("pow".to_string(), vec![self.dtype(), input_exponent.dtype()]))?,
                }
            },
            NDArrayNumericTensor::F16(exponent) => {
                let exponent = exponent.map(|x| (*x).to_f32()).to_shared();
                match self {
                    NDArrayNumericTensor::F32(x) => NDArrayNumericTensor::F32(ops::pow(x.map(|x| *x as f64).to_shared(), exponent.clone())?.map(|x| *x as f32).to_shared()),
                    NDArrayNumericTensor::F64(x) => NDArrayNumericTensor::F64(ops::pow(x.clone(), exponent.clone())?),
                    NDArrayNumericTensor::F16(x) => NDArrayNumericTensor::F16(ops::pow(x.map(|x| (*x).to_f64()).to_shared(), exponent.clone())?.map(|x| f16::from_f64(*x)).to_shared()),
                    NDArrayNumericTensor::BF16(x) => NDArrayNumericTensor::BF16(ops::pow(x.map(|x| (*x).to_f64()).to_shared(), exponent.clone())?.map(|x| bf16::from_f64(*x)).to_shared()),
                    _ => Err(NDArrayNumericTensorError::UnsupportedOperationForDTypes("pow".to_string(), vec![self.dtype(), input_exponent.dtype()]))?,
                }
            },
            NDArrayNumericTensor::BF16(exponent) => {
                let exponent = exponent.map(|x| (*x).to_f32()).to_shared();
                match self {
                    NDArrayNumericTensor::F32(x) => NDArrayNumericTensor::F32(ops::pow(x.map(|x| *x as f64).to_shared(), exponent.clone())?.map(|x| *x as f32).to_shared()),
                    NDArrayNumericTensor::F64(x) => NDArrayNumericTensor::F64(ops::pow(x.clone(), exponent.clone())?),
                    NDArrayNumericTensor::F16(x) => NDArrayNumericTensor::F16(ops::pow(x.map(|x| (*x).to_f64()).to_shared(), exponent.clone())?.map(|x| f16::from_f64(*x)).to_shared()),
                    NDArrayNumericTensor::BF16(x) => NDArrayNumericTensor::BF16(ops::pow(x.map(|x| (*x).to_f64()).to_shared(), exponent.clone())?.map(|x| bf16::from_f64(*x)).to_shared()),
                    _ => Err(NDArrayNumericTensorError::UnsupportedOperationForDTypes("pow".to_string(), vec![self.dtype(), input_exponent.dtype()]))?,
                }
            },
            NDArrayNumericTensor::U64(exponent) => {
                let exponent = exponent.map(|x| (*x) as i32).to_shared();
                match self {
                    NDArrayNumericTensor::F32(x) => NDArrayNumericTensor::F32(ops::pow(x.clone(), exponent.clone())?),
                    NDArrayNumericTensor::F64(x) => NDArrayNumericTensor::F64(ops::pow(x.clone(), exponent.clone())?),
                    NDArrayNumericTensor::F16(x) => NDArrayNumericTensor::F16(ops::pow(x.map(|x| (*x).to_f32()).to_shared(), exponent.clone())?.map(|x| f16::from_f32(*x)).to_shared()),
                    NDArrayNumericTensor::BF16(x) => NDArrayNumericTensor::BF16(ops::pow(x.map(|x| (*x).to_f32()).to_shared(), exponent.clone())?.map(|x| bf16::from_f32(*x)).to_shared()),
                    _ => Err(NDArrayNumericTensorError::UnsupportedOperationForDTypes("pow".to_string(), vec![self.dtype(), input_exponent.dtype()]))?,
                }
            },
            NDArrayNumericTensor::I64(exponent) => {
                let exponent = exponent.map(|x| (*x) as i32).to_shared();
                match self {
                    NDArrayNumericTensor::F32(x) => NDArrayNumericTensor::F32(ops::pow(x.clone(), exponent.clone())?),
                    NDArrayNumericTensor::F64(x) => NDArrayNumericTensor::F64(ops::pow(x.clone(), exponent.clone())?),
                    NDArrayNumericTensor::F16(x) => NDArrayNumericTensor::F16(ops::pow(x.map(|x| (*x).to_f32()).to_shared(), exponent.clone())?.map(|x| f16::from_f32(*x)).to_shared()),
                    NDArrayNumericTensor::BF16(x) => NDArrayNumericTensor::BF16(ops::pow(x.map(|x| (*x).to_f32()).to_shared(), exponent.clone())?.map(|x| bf16::from_f32(*x)).to_shared()),
                    _ => Err(NDArrayNumericTensorError::UnsupportedOperationForDTypes("pow".to_string(), vec![self.dtype(), input_exponent.dtype()]))?,
                }
            },
            NDArrayNumericTensor::U32(exponent) => {
                let exponent = exponent.map(|x| (*x) as i32).to_shared();
                match self {
                    NDArrayNumericTensor::F32(x) => NDArrayNumericTensor::F32(ops::pow(x.clone(), exponent.clone())?),
                    NDArrayNumericTensor::F64(x) => NDArrayNumericTensor::F64(ops::pow(x.clone(), exponent.clone())?),
                    NDArrayNumericTensor::F16(x) => NDArrayNumericTensor::F16(ops::pow(x.map(|x| (*x).to_f32()).to_shared(), exponent.clone())?.map(|x| f16::from_f32(*x)).to_shared()),
                    NDArrayNumericTensor::BF16(x) => NDArrayNumericTensor::BF16(ops::pow(x.map(|x| (*x).to_f32()).to_shared(), exponent.clone())?.map(|x| bf16::from_f32(*x)).to_shared()),
                    _ => Err(NDArrayNumericTensorError::UnsupportedOperationForDTypes("pow".to_string(), vec![self.dtype(), input_exponent.dtype()]))?,
                }
            },
            NDArrayNumericTensor::I32(exponent) => {
                match self {
                    NDArrayNumericTensor::F32(x) => NDArrayNumericTensor::F32(ops::pow(x.clone(), exponent.clone())?),
                    NDArrayNumericTensor::F64(x) => NDArrayNumericTensor::F64(ops::pow(x.clone(), exponent.clone())?),
                    NDArrayNumericTensor::F16(x) => NDArrayNumericTensor::F16(ops::pow(x.map(|x| (*x).to_f32()).to_shared(), exponent.clone())?.map(|x| f16::from_f32(*x)).to_shared()),
                    NDArrayNumericTensor::BF16(x) => NDArrayNumericTensor::BF16(ops::pow(x.map(|x| (*x).to_f32()).to_shared(), exponent.clone())?.map(|x| bf16::from_f32(*x)).to_shared()),
                    _ => Err(NDArrayNumericTensorError::UnsupportedOperationForDTypes("pow".to_string(), vec![self.dtype(), input_exponent.dtype()]))?,
                }
            },
            NDArrayNumericTensor::U16(exponent) => {
                match self {
                    NDArrayNumericTensor::F32(x) => NDArrayNumericTensor::F32(ops::pow(x.clone(), exponent.clone())?),
                    NDArrayNumericTensor::F64(x) => NDArrayNumericTensor::F64(ops::pow(x.clone(), exponent.clone())?),
                    NDArrayNumericTensor::F16(x) => NDArrayNumericTensor::F16(ops::pow(x.map(|x| (*x).to_f32()).to_shared(), exponent.clone())?.map(|x| f16::from_f32(*x)).to_shared()),
                    NDArrayNumericTensor::BF16(x) => NDArrayNumericTensor::BF16(ops::pow(x.map(|x| (*x).to_f32()).to_shared(), exponent.clone())?.map(|x| bf16::from_f32(*x)).to_shared()),
                    _ => Err(NDArrayNumericTensorError::UnsupportedOperationForDTypes("pow".to_string(), vec![self.dtype(), input_exponent.dtype()]))?,
                }
            },
            NDArrayNumericTensor::I16(exponent) => {
                match self {
                    NDArrayNumericTensor::F32(x) => NDArrayNumericTensor::F32(ops::pow(x.clone(), exponent.clone())?),
                    NDArrayNumericTensor::F64(x) => NDArrayNumericTensor::F64(ops::pow(x.clone(), exponent.clone())?),
                    NDArrayNumericTensor::F16(x) => NDArrayNumericTensor::F16(ops::pow(x.map(|x| (*x).to_f32()).to_shared(), exponent.clone())?.map(|x| f16::from_f32(*x)).to_shared()),
                    NDArrayNumericTensor::BF16(x) => NDArrayNumericTensor::BF16(ops::pow(x.map(|x| (*x).to_f32()).to_shared(), exponent.clone())?.map(|x| bf16::from_f32(*x)).to_shared()),
                    _ => Err(NDArrayNumericTensorError::UnsupportedOperationForDTypes("pow".to_string(), vec![self.dtype(), input_exponent.dtype()]))?,
                }
            },
            NDArrayNumericTensor::U8(exponent) => {
                match self {
                    NDArrayNumericTensor::F32(x) => NDArrayNumericTensor::F32(ops::pow(x.clone(), exponent.clone())?),
                    NDArrayNumericTensor::F64(x) => NDArrayNumericTensor::F64(ops::pow(x.clone(), exponent.clone())?),
                    NDArrayNumericTensor::F16(x) => NDArrayNumericTensor::F16(ops::pow(x.map(|x| (*x).to_f32()).to_shared(), exponent.clone())?.map(|x| f16::from_f32(*x)).to_shared()),
                    NDArrayNumericTensor::BF16(x) => NDArrayNumericTensor::BF16(ops::pow(x.map(|x| (*x).to_f32()).to_shared(), exponent.clone())?.map(|x| bf16::from_f32(*x)).to_shared()),
                    _ => Err(NDArrayNumericTensorError::UnsupportedOperationForDTypes("pow".to_string(), vec![self.dtype(), input_exponent.dtype()]))?,
                }
            },
            NDArrayNumericTensor::I8(exponent) => {
                match self {
                    NDArrayNumericTensor::F32(x) => NDArrayNumericTensor::F32(ops::pow(x.clone(), exponent.clone())?),
                    NDArrayNumericTensor::F64(x) => NDArrayNumericTensor::F64(ops::pow(x.clone(), exponent.clone())?),
                    NDArrayNumericTensor::F16(x) => NDArrayNumericTensor::F16(ops::pow(x.map(|x| (*x).to_f32()).to_shared(), exponent.clone())?.map(|x| f16::from_f32(*x)).to_shared()),
                    NDArrayNumericTensor::BF16(x) => NDArrayNumericTensor::BF16(ops::pow(x.map(|x| (*x).to_f32()).to_shared(), exponent.clone())?.map(|x| bf16::from_f32(*x)).to_shared()),
                    _ => Err(NDArrayNumericTensorError::UnsupportedOperationForDTypes("pow".to_string(), vec![self.dtype(), input_exponent.dtype()]))?,
                }
            },
            _ => Err(NDArrayNumericTensorError::UnsupportedOperationForDTypes("pow".to_string(), vec![self.dtype(), input_exponent.dtype()]))?,
        })
    }

    pub fn nonzero(&self) -> Result<Self, NDArrayNumericTensorError> {
        Ok(match self {
            NDArrayNumericTensor::F32(x) => NDArrayNumericTensor::I64(ops::nonzero(x.clone())?),
            NDArrayNumericTensor::F64(x) => NDArrayNumericTensor::I64(ops::nonzero(x.clone())?),
            NDArrayNumericTensor::F16(x) => NDArrayNumericTensor::I64(ops::nonzero(x.clone())?),
            NDArrayNumericTensor::BF16(x) => NDArrayNumericTensor::I64(ops::nonzero(x.clone())?),
            NDArrayNumericTensor::U32(x) => NDArrayNumericTensor::I64(ops::nonzero(x.clone())?),
            NDArrayNumericTensor::I32(x) => NDArrayNumericTensor::I64(ops::nonzero(x.clone())?),
            NDArrayNumericTensor::U64(x) => NDArrayNumericTensor::I64(ops::nonzero(x.clone())?),
            NDArrayNumericTensor::I64(x) => NDArrayNumericTensor::I64(ops::nonzero(x.clone())?),
            NDArrayNumericTensor::U16(x) => NDArrayNumericTensor::I64(ops::nonzero(x.clone())?),
            NDArrayNumericTensor::I16(x) => NDArrayNumericTensor::I64(ops::nonzero(x.clone())?),
            NDArrayNumericTensor::I8(x) => NDArrayNumericTensor::I64(ops::nonzero(x.clone())?),
            NDArrayNumericTensor::U8(x) => NDArrayNumericTensor::I64(ops::nonzero(x.clone())?),
            NDArrayNumericTensor::BOOL(x) => NDArrayNumericTensor::I64(ops::nonzero(x.clone())?),
        })
    }

    pub fn sqrt(&self) -> Result<Self, NDArrayNumericTensorError> {
        Ok(match self {
            NDArrayNumericTensor::F32(x) => NDArrayNumericTensor::F32(x.sqrt().to_shared()),
            NDArrayNumericTensor::F64(x) => NDArrayNumericTensor::F64(x.sqrt().to_shared()),
            NDArrayNumericTensor::BF16(x) => NDArrayNumericTensor::BF16(x.sqrt().to_shared()),
            NDArrayNumericTensor::F16(x) => NDArrayNumericTensor::F16(x.sqrt().to_shared()),
            _ => Err(NDArrayNumericTensorError::UnsupportedOperationForDTypes("sqrt".to_string(), vec![self.dtype()]))?
        })
    }

    pub fn reciprocal(&self) -> Result<Self, NDArrayNumericTensorError> {
        Ok(match self {
            NDArrayNumericTensor::F32(x) => NDArrayNumericTensor::F32(x.recip().to_shared()),
            NDArrayNumericTensor::F64(x) => NDArrayNumericTensor::F64(x.recip().to_shared()),
            NDArrayNumericTensor::BF16(x) => NDArrayNumericTensor::BF16(x.recip().to_shared()),
            NDArrayNumericTensor::F16(x) => NDArrayNumericTensor::F16(x.recip().to_shared()),
            _ => Err(NDArrayNumericTensorError::UnsupportedOperationForDTypes("recip".to_string(), vec![self.dtype()]))?
        })
    }
    
    pub fn transpose(&self, axes: Option<Vec<i64>>) -> Result<Self, NDArrayNumericTensorError> {
        let v = match self {
            NDArrayNumericTensor::F32(a) => {
                Some(NDArrayNumericTensor::F32(ops::transpose(a.clone(), axes)?))
            }
            NDArrayNumericTensor::F64(a) => {
                Some(NDArrayNumericTensor::F64(ops::transpose(a.clone(), axes)?))
            }
            NDArrayNumericTensor::F16(a) => {
                Some(NDArrayNumericTensor::F16(ops::transpose(a.clone(), axes)?))
            }
            NDArrayNumericTensor::BF16(a) => {
                Some(NDArrayNumericTensor::BF16(ops::transpose(a.clone(), axes)?))
            }
            NDArrayNumericTensor::U32(a) => {
                Some(NDArrayNumericTensor::U32(ops::transpose(a.clone(), axes)?))
            }
            NDArrayNumericTensor::I32(a) => {
                Some(NDArrayNumericTensor::I32(ops::transpose(a.clone(), axes)?))
            }
            NDArrayNumericTensor::U64(a) => {
                Some(NDArrayNumericTensor::U64(ops::transpose(a.clone(), axes)?))
            }
            NDArrayNumericTensor::I64(a) => {
                Some(NDArrayNumericTensor::I64(ops::transpose(a.clone(), axes)?))
            }
            NDArrayNumericTensor::U16(a) => {
                Some(NDArrayNumericTensor::U16(ops::transpose(a.clone(), axes)?))
            }
            NDArrayNumericTensor::I16(a) => {
                Some(NDArrayNumericTensor::I16(ops::transpose(a.clone(), axes)?))
            }
            NDArrayNumericTensor::I8(a) => {
                Some(NDArrayNumericTensor::I8(ops::transpose(a.clone(), axes)?))
            }
            NDArrayNumericTensor::U8(a) => {
                Some(NDArrayNumericTensor::U8(ops::transpose(a.clone(), axes)?))
            }
            NDArrayNumericTensor::BOOL(a) => {
                Some(NDArrayNumericTensor::BOOL(ops::transpose(a.clone(), axes)?))
            }
        };
        v.ok_or(NDArrayNumericTensorError::UnsupportedOperationForDTypes("Transpose".to_string(), vec![self.dtype()]))
    }

    pub fn gather(data: &NDArrayNumericTensor, indices: &NDArrayNumericTensor, axis: i64) -> Result<NDArrayNumericTensor, NDArrayNumericTensorError> {
        let indices = if let NDArrayNumericTensor::I64(x) = indices.cast(DType::I64)? {x} else {unreachable!()};
        let axis = (if axis < 0 {axis + data.rank() as i64} else {axis}) as usize;
        Ok(match data {
            NDArrayNumericTensor::F32(x) => NDArrayNumericTensor::F32(ops::gather(axis, x.clone(), indices)?),
            NDArrayNumericTensor::F64(x) => NDArrayNumericTensor::F64(ops::gather(axis, x.clone(), indices)?),
            NDArrayNumericTensor::F16(x) => NDArrayNumericTensor::F16(ops::gather(axis, x.clone(), indices)?),
            NDArrayNumericTensor::BF16(x) => NDArrayNumericTensor::BF16(ops::gather(axis, x.clone(), indices)?),
            NDArrayNumericTensor::U32(x) => NDArrayNumericTensor::U32(ops::gather(axis, x.clone(), indices)?),
            NDArrayNumericTensor::I32(x) => NDArrayNumericTensor::I32(ops::gather(axis, x.clone(), indices)?),
            NDArrayNumericTensor::U64(x) => NDArrayNumericTensor::U64(ops::gather(axis, x.clone(), indices)?),
            NDArrayNumericTensor::I64(x) => NDArrayNumericTensor::I64(ops::gather(axis, x.clone(), indices)?),
            NDArrayNumericTensor::U16(x) => NDArrayNumericTensor::U16(ops::gather(axis, x.clone(), indices)?),
            NDArrayNumericTensor::I16(x) => NDArrayNumericTensor::I16(ops::gather(axis, x.clone(), indices)?),
            NDArrayNumericTensor::U8(x) => NDArrayNumericTensor::U8(ops::gather(axis, x.clone(), indices)?),
            NDArrayNumericTensor::I8(x) => NDArrayNumericTensor::I8(ops::gather(axis, x.clone(), indices)?),
            NDArrayNumericTensor::BOOL(x) => NDArrayNumericTensor::BOOL(ops::gather(axis, x.clone(), indices)?),
        })
    }

    pub fn cast(&self, dtype: DType) -> Result<NDArrayNumericTensor, NDArrayNumericTensorError> {
        match self {
            NDArrayNumericTensor::F32(x) => {
                match dtype {
                    DType::F32 => Ok(NDArrayNumericTensor::F32(x.clone())),
                    DType::F64 => Ok(NDArrayNumericTensor::F64(x.map(|x| *x as f64).to_shared())),
                    DType::F16 => Ok(NDArrayNumericTensor::F16(x.map(|x| f16::from_f32(*x)).to_shared())),
                    DType::BF16 => Ok(NDArrayNumericTensor::BF16(x.map(|x| bf16::from_f32(*x)).to_shared())),
                    DType::U32 => Ok(NDArrayNumericTensor::U32(x.map(|x| *x as u32).to_shared())),
                    DType::I64 => Ok(NDArrayNumericTensor::I64(x.map(|x| *x as i64).to_shared())),
                    DType::I32 => Ok(NDArrayNumericTensor::I32(x.map(|x| *x as i32).to_shared())),
                    DType::U64 => Ok(NDArrayNumericTensor::U64(x.map(|x| *x as u64).to_shared())),
                    DType::I16 => Ok(NDArrayNumericTensor::I16(x.map(|x| *x as i16).to_shared())),
                    DType::U16 => Ok(NDArrayNumericTensor::U16(x.map(|x| *x as u16).to_shared())),
                    DType::I8 => Ok(NDArrayNumericTensor::I8(x.map(|x| *x as i8).to_shared())),
                    DType::U8 => Ok(NDArrayNumericTensor::U8(x.map(|x| *x as u8).to_shared())),
                    _ => Err(NDArrayNumericTensorError::InvalidCastOperation(self.dtype(), dtype)),
                }
            }
            NDArrayNumericTensor::F64(x) => {
                match dtype {
                    DType::F32 => Ok(NDArrayNumericTensor::F32(x.map(|x| *x as f32).to_shared())),
                    DType::F64 => Ok(NDArrayNumericTensor::F64(x.clone())),
                    DType::F16 => Ok(NDArrayNumericTensor::F16(x.map(|x| f16::from_f64(*x)).to_shared())),
                    DType::BF16 => Ok(NDArrayNumericTensor::BF16(x.map(|x| bf16::from_f64(*x)).to_shared())),
                    DType::U32 => Ok(NDArrayNumericTensor::U32(x.map(|x| *x as u32).to_shared())),
                    DType::I64 => Ok(NDArrayNumericTensor::I64(x.map(|x| *x as i64).to_shared())),
                    DType::I32 => Ok(NDArrayNumericTensor::I32(x.map(|x| *x as i32).to_shared())),
                    DType::U64 => Ok(NDArrayNumericTensor::U64(x.map(|x| *x as u64).to_shared())),
                    DType::U16 => Ok(NDArrayNumericTensor::U16(x.map(|x| *x as u16).to_shared())),
                    DType::I16 => Ok(NDArrayNumericTensor::I16(x.map(|x| *x as i16).to_shared())),
                    DType::I8 => Ok(NDArrayNumericTensor::I8(x.map(|x| *x as i8).to_shared())),
                    DType::U8 => Ok(NDArrayNumericTensor::U8(x.map(|x| *x as u8).to_shared())),
                    _ => Err(NDArrayNumericTensorError::InvalidCastOperation(self.dtype(), dtype)),
                }
            }
            NDArrayNumericTensor::F16(x) => {
                match dtype {
                    DType::F32 => Ok(NDArrayNumericTensor::F32(x.map(|x| x.to_f32()).to_shared())),
                    DType::F64 => Ok(NDArrayNumericTensor::F64(x.map(|x| x.to_f64()).to_shared())),
                    DType::F16 => Ok(NDArrayNumericTensor::F16(x.clone())),
                    DType::BF16 => Ok(NDArrayNumericTensor::BF16(x.map(|x| bf16::from_f32(x.to_f32())).to_shared())),
                    DType::U32 => Ok(NDArrayNumericTensor::U32(x.map(|x| x.to_f32() as u32).to_shared())),
                    DType::I64 => Ok(NDArrayNumericTensor::I64(x.map(|x| x.to_f32() as i64).to_shared())),
                    DType::I32 => Ok(NDArrayNumericTensor::I32(x.map(|x| x.to_f32() as i32).to_shared())),
                    DType::U64 => Ok(NDArrayNumericTensor::U64(x.map(|x| x.to_f32() as u64).to_shared())),
                    DType::U16 => Ok(NDArrayNumericTensor::U16(x.map(|x| x.to_f32() as u16).to_shared())),
                    DType::I16 => Ok(NDArrayNumericTensor::I16(x.map(|x| x.to_f32() as i16).to_shared())),
                    DType::I8 => Ok(NDArrayNumericTensor::I8(x.map(|x| x.to_f32() as i8).to_shared())),
                    DType::U8 => Ok(NDArrayNumericTensor::U8(x.map(|x| x.to_f32() as u8).to_shared())),
                    _ => Err(NDArrayNumericTensorError::InvalidCastOperation(self.dtype(), dtype)),
                }
            }
            NDArrayNumericTensor::BF16(x) => {
                match dtype {
                    DType::F32 => Ok(NDArrayNumericTensor::F32(x.map(|x| x.to_f32()).to_shared())),
                    DType::F64 => Ok(NDArrayNumericTensor::F64(x.map(|x| x.to_f64()).to_shared())),
                    DType::F16 => Ok(NDArrayNumericTensor::F16(x.map(|x| f16::from_f32(x.to_f32())).to_shared())),
                    DType::BF16 => Ok(NDArrayNumericTensor::BF16(x.clone())),
                    DType::U32 => Ok(NDArrayNumericTensor::U32(x.map(|x| x.to_f32() as u32).to_shared())),
                    DType::I64 => Ok(NDArrayNumericTensor::I64(x.map(|x| x.to_f32() as i64).to_shared())),
                    DType::I32 => Ok(NDArrayNumericTensor::I32(x.map(|x| x.to_f32() as i32).to_shared())),
                    DType::U64 => Ok(NDArrayNumericTensor::U64(x.map(|x| x.to_f32() as u64).to_shared())),
                    DType::U16 => Ok(NDArrayNumericTensor::U16(x.map(|x| x.to_f32() as u16).to_shared())),
                    DType::I16 => Ok(NDArrayNumericTensor::I16(x.map(|x| x.to_f32() as i16).to_shared())),
                    DType::I8 => Ok(NDArrayNumericTensor::I8(x.map(|x| x.to_f32() as i8).to_shared())),
                    DType::U8 => Ok(NDArrayNumericTensor::U8(x.map(|x| x.to_f32() as u8).to_shared())),
                    _ => Err(NDArrayNumericTensorError::InvalidCastOperation(self.dtype(), dtype)),
                }
            }
            NDArrayNumericTensor::U32(x) => {
                match dtype {
                    DType::F32 => Ok(NDArrayNumericTensor::F32(x.map(|x| *x as f32).to_shared())),
                    DType::F64 => Ok(NDArrayNumericTensor::F64(x.map(|x| *x as f64).to_shared())),
                    DType::F16 => Ok(NDArrayNumericTensor::F16(x.map(|x| f16::from_f32(*x as f32)).to_shared())),
                    DType::BF16 => Ok(NDArrayNumericTensor::BF16(x.map(|x| bf16::from_f32(*x as f32)).to_shared())),
                    DType::U32 => Ok(NDArrayNumericTensor::U32(x.clone())),
                    DType::I64 => Ok(NDArrayNumericTensor::I64(x.map(|x| *x as i64).to_shared())),
                    DType::I32 => Ok(NDArrayNumericTensor::I32(x.map(|x| *x as i32).to_shared())),
                    DType::U64 => Ok(NDArrayNumericTensor::U64(x.map(|x| *x as u64).to_shared())),
                    DType::U16 => Ok(NDArrayNumericTensor::U16(x.map(|x| *x as u16).to_shared())),
                    DType::I16 => Ok(NDArrayNumericTensor::I16(x.map(|x| *x as i16).to_shared())),
                    DType::I8 => Ok(NDArrayNumericTensor::I8(x.map(|x| *x as i8).to_shared())),
                    DType::U8 => Ok(NDArrayNumericTensor::U8(x.map(|x| *x as u8).to_shared())),
                    DType::BOOL => Ok(NDArrayNumericTensor::BOOL(x.map(|x| *x != 0).to_shared())),
                }
            }
            NDArrayNumericTensor::I32(x) => {
                match dtype {
                    DType::F32 => Ok(NDArrayNumericTensor::F32(x.map(|x| *x as f32).to_shared())),
                    DType::F64 => Ok(NDArrayNumericTensor::F64(x.map(|x| *x as f64).to_shared())),
                    DType::F16 => Ok(NDArrayNumericTensor::F16(x.map(|x| f16::from_f32(*x as f32)).to_shared())),
                    DType::BF16 => Ok(NDArrayNumericTensor::BF16(x.map(|x| bf16::from_f32(*x as f32)).to_shared())),
                    DType::U32 => Ok(NDArrayNumericTensor::U32(x.map(|x| *x as u32).to_shared())),
                    DType::I32 => Ok(NDArrayNumericTensor::I32(x.clone())),
                    DType::I64 => Ok(NDArrayNumericTensor::I64(x.map(|x| *x as i64).to_shared())),
                    DType::U64 => Ok(NDArrayNumericTensor::U64(x.map(|x| *x as u64).to_shared())),
                    DType::U16 => Ok(NDArrayNumericTensor::U16(x.map(|x| *x as u16).to_shared())),
                    DType::I16 => Ok(NDArrayNumericTensor::I16(x.map(|x| *x as i16).to_shared())),
                    DType::I8 => Ok(NDArrayNumericTensor::I8(x.map(|x| *x as i8).to_shared())),
                    DType::U8 => Ok(NDArrayNumericTensor::U8(x.map(|x| *x as u8).to_shared())),
                    DType::BOOL => Ok(NDArrayNumericTensor::BOOL(x.map(|x| *x != 0).to_shared())),
                }
            }
            NDArrayNumericTensor::U64(x) => {
                match dtype {
                    DType::F32 => Ok(NDArrayNumericTensor::F32(x.map(|x| *x as f32).to_shared())),
                    DType::F64 => Ok(NDArrayNumericTensor::F64(x.map(|x| *x as f64).to_shared())),
                    DType::F16 => Ok(NDArrayNumericTensor::F16(x.map(|x| f16::from_f32(*x as f32)).to_shared())),
                    DType::BF16 => Ok(NDArrayNumericTensor::BF16(x.map(|x| bf16::from_f32(*x as f32)).to_shared())),
                    DType::U32 => Ok(NDArrayNumericTensor::U32(x.map(|x| *x as u32).to_shared())),
                    DType::I32 => Ok(NDArrayNumericTensor::I32(x.map(|x| *x as i32).to_shared())),
                    DType::U64 => Ok(NDArrayNumericTensor::U64(x.clone())),
                    DType::I64 => Ok(NDArrayNumericTensor::I64(x.map(|x| *x as i64).to_shared())),
                    DType::U16 => Ok(NDArrayNumericTensor::U16(x.map(|x| *x as u16).to_shared())),
                    DType::I16 => Ok(NDArrayNumericTensor::I16(x.map(|x| *x as i16).to_shared())),
                    DType::I8 => Ok(NDArrayNumericTensor::I8(x.map(|x| *x as i8).to_shared())),
                    DType::U8 => Ok(NDArrayNumericTensor::U8(x.map(|x| *x as u8).to_shared())),
                    DType::BOOL => Ok(NDArrayNumericTensor::BOOL(x.map(|x| *x != 0).to_shared())),
                }
            }
            NDArrayNumericTensor::I64(x) => {
                match dtype {
                    DType::F32 => Ok(NDArrayNumericTensor::F32(x.map(|x| *x as f32).to_shared())),
                    DType::F64 => Ok(NDArrayNumericTensor::F64(x.map(|x| *x as f64).to_shared())),
                    DType::F16 => Ok(NDArrayNumericTensor::F16(x.map(|x| f16::from_f32(*x as f32)).to_shared())),
                    DType::BF16 => Ok(NDArrayNumericTensor::BF16(x.map(|x| bf16::from_f32(*x as f32)).to_shared())),
                    DType::U32 => Ok(NDArrayNumericTensor::U32(x.map(|x| *x as u32).to_shared())),
                    DType::I32 => Ok(NDArrayNumericTensor::I32(x.map(|x| *x as i32).to_shared())),
                    DType::U64 => Ok(NDArrayNumericTensor::U64(x.map(|x| *x as u64).to_shared())),
                    DType::I64 => Ok(NDArrayNumericTensor::I64(x.clone())),
                    DType::U16 => Ok(NDArrayNumericTensor::U16(x.map(|x| *x as u16).to_shared())),
                    DType::I16 => Ok(NDArrayNumericTensor::I16(x.map(|x| *x as i16).to_shared())),
                    DType::I8 => Ok(NDArrayNumericTensor::I8(x.map(|x| *x as i8).to_shared())),
                    DType::U8 => Ok(NDArrayNumericTensor::U8(x.map(|x| *x as u8).to_shared())),
                    DType::BOOL => Ok(NDArrayNumericTensor::BOOL(x.map(|x| *x != 0).to_shared())),
                }
            }
            NDArrayNumericTensor::U16(x) => {
                match dtype {
                    DType::F32 => Ok(NDArrayNumericTensor::F32(x.map(|x| *x as f32).to_shared())),
                    DType::F64 => Ok(NDArrayNumericTensor::F64(x.map(|x| *x as f64).to_shared())),
                    DType::F16 => Ok(NDArrayNumericTensor::F16(x.map(|x| f16::from_f32(*x as f32)).to_shared())),
                    DType::BF16 => Ok(NDArrayNumericTensor::BF16(x.map(|x| bf16::from_f32(*x as f32)).to_shared())),
                    DType::U32 => Ok(NDArrayNumericTensor::U32(x.map(|x| *x as u32).to_shared())),
                    DType::I32 => Ok(NDArrayNumericTensor::I32(x.map(|x| *x as i32).to_shared())),
                    DType::U64 => Ok(NDArrayNumericTensor::U64(x.map(|x| *x as u64).to_shared())),
                    DType::I64 => Ok(NDArrayNumericTensor::I64(x.map(|x| *x as i64).to_shared())),
                    DType::I8 => Ok(NDArrayNumericTensor::I8(x.map(|x| *x as i8).to_shared())),
                    DType::U16 => Ok(NDArrayNumericTensor::U16(x.clone())),
                    DType::I16 => Ok(NDArrayNumericTensor::I16(x.map(|x| *x as i16).to_shared())),
                    DType::U8 => Ok(NDArrayNumericTensor::U8(x.map(|x| *x as u8).to_shared())),
                    DType::BOOL => Ok(NDArrayNumericTensor::BOOL(x.map(|x| *x != 0).to_shared())),
                }
            }
            NDArrayNumericTensor::I16(x) => {
                match dtype {
                    DType::F32 => Ok(NDArrayNumericTensor::F32(x.map(|x| *x as f32).to_shared())),
                    DType::F64 => Ok(NDArrayNumericTensor::F64(x.map(|x| *x as f64).to_shared())),
                    DType::F16 => Ok(NDArrayNumericTensor::F16(x.map(|x| f16::from_f32(*x as f32)).to_shared())),
                    DType::BF16 => Ok(NDArrayNumericTensor::BF16(x.map(|x| bf16::from_f32(*x as f32)).to_shared())),
                    DType::U32 => Ok(NDArrayNumericTensor::U32(x.map(|x| *x as u32).to_shared())),
                    DType::I32 => Ok(NDArrayNumericTensor::I32(x.map(|x| *x as i32).to_shared())),
                    DType::U64 => Ok(NDArrayNumericTensor::U64(x.map(|x| *x as u64).to_shared())),
                    DType::I64 => Ok(NDArrayNumericTensor::I64(x.map(|x| *x as i64).to_shared())),
                    DType::I8 => Ok(NDArrayNumericTensor::I8(x.map(|x| *x as i8).to_shared())),
                    DType::U16 => Ok(NDArrayNumericTensor::U16(x.map(|x| *x as u16).to_shared())),
                    DType::I16 => Ok(NDArrayNumericTensor::I16(x.clone())),
                    DType::U8 => Ok(NDArrayNumericTensor::U8(x.map(|x| *x as u8).to_shared())),
                    DType::BOOL => Ok(NDArrayNumericTensor::BOOL(x.map(|x| *x != 0).to_shared())),
                }
            }
            NDArrayNumericTensor::I8(x) => {
                match dtype {
                    DType::F32 => Ok(NDArrayNumericTensor::F32(x.map(|x| *x as f32).to_shared())),
                    DType::F64 => Ok(NDArrayNumericTensor::F64(x.map(|x| *x as f64).to_shared())),
                    DType::F16 => Ok(NDArrayNumericTensor::F16(x.map(|x| f16::from_f32(*x as f32)).to_shared())),
                    DType::BF16 => Ok(NDArrayNumericTensor::BF16(x.map(|x| bf16::from_f32(*x as f32)).to_shared())),
                    DType::U32 => Ok(NDArrayNumericTensor::U32(x.map(|x| *x as u32).to_shared())),
                    DType::I32 => Ok(NDArrayNumericTensor::I32(x.map(|x| *x as i32).to_shared())),
                    DType::U64 => Ok(NDArrayNumericTensor::U64(x.map(|x| *x as u64).to_shared())),
                    DType::I64 => Ok(NDArrayNumericTensor::I64(x.map(|x| *x as i64).to_shared())),
                    DType::U16 => Ok(NDArrayNumericTensor::U16(x.map(|x| *x as u16).to_shared())),
                    DType::I16 => Ok(NDArrayNumericTensor::I16(x.map(|x| *x as i16).to_shared())),
                    DType::U8 => Ok(NDArrayNumericTensor::I8(x.clone())),
                    DType::I8 => Ok(NDArrayNumericTensor::U8(x.map(|x| *x as u8).to_shared())),
                    DType::BOOL => Ok(NDArrayNumericTensor::BOOL(x.map(|x| *x != 0).to_shared())),
                }
            }
            NDArrayNumericTensor::U8(x) => {
                match dtype {
                    DType::F32 => Ok(NDArrayNumericTensor::F32(x.map(|x| *x as f32).to_shared())),
                    DType::F64 => Ok(NDArrayNumericTensor::F64(x.map(|x| *x as f64).to_shared())),
                    DType::F16 => Ok(NDArrayNumericTensor::F16(x.map(|x| f16::from_f32(*x as f32)).to_shared())),
                    DType::BF16 => Ok(NDArrayNumericTensor::BF16(x.map(|x| bf16::from_f32(*x as f32)).to_shared())),
                    DType::U32 => Ok(NDArrayNumericTensor::U32(x.map(|x| *x as u32).to_shared())),
                    DType::I32 => Ok(NDArrayNumericTensor::I32(x.map(|x| *x as i32).to_shared())),
                    DType::U64 => Ok(NDArrayNumericTensor::U64(x.map(|x| *x as u64).to_shared())),
                    DType::I64 => Ok(NDArrayNumericTensor::I64(x.map(|x| *x as i64).to_shared())),
                    DType::U16 => Ok(NDArrayNumericTensor::U16(x.map(|x| *x as u16).to_shared())),
                    DType::I16 => Ok(NDArrayNumericTensor::I16(x.map(|x| *x as i16).to_shared())),
                    DType::I8 => Ok(NDArrayNumericTensor::I8(x.map(|x| *x as i8).to_shared())),
                    DType::U8 => Ok(NDArrayNumericTensor::U8(x.clone())),
                    DType::BOOL => Ok(NDArrayNumericTensor::BOOL(x.map(|x| *x != 0).to_shared())),
                }
            }
            NDArrayNumericTensor::BOOL(x) => {
                match dtype {
                    DType::U32 => Ok(NDArrayNumericTensor::U32(x.map(|x| *x as u32).to_shared())),
                    DType::I32 => Ok(NDArrayNumericTensor::I32(x.map(|x| *x as i32).to_shared())),
                    DType::U64 => Ok(NDArrayNumericTensor::U64(x.map(|x| *x as u64).to_shared())),
                    DType::I64 => Ok(NDArrayNumericTensor::I64(x.map(|x| *x as i64).to_shared())),
                    DType::U16 => Ok(NDArrayNumericTensor::U16(x.map(|x| *x as u16).to_shared())),
                    DType::I16 => Ok(NDArrayNumericTensor::I16(x.map(|x| *x as i16).to_shared())),
                    DType::I8 => Ok(NDArrayNumericTensor::I8(x.map(|x| *x as i8).to_shared())),
                    DType::U8 => Ok(NDArrayNumericTensor::U8(x.map(|x| *x as u8).to_shared())),
                    DType::BOOL => Ok(NDArrayNumericTensor::BOOL(x.clone())),
                    _ => Err(NDArrayNumericTensorError::InvalidCastOperation(self.dtype(), dtype)),
                }
            }
        }
    }
    
    pub fn group_norm(&self, scale: &NDArrayNumericTensor, bias: &NDArrayNumericTensor, num_groups: usize, epsilon: f64) -> Result<Self, NDArrayNumericTensorError> {
        match (self, scale, bias) {
            (NDArrayNumericTensor::F32(x), NDArrayNumericTensor::F32(scale), NDArrayNumericTensor::F32(bias)) => 
                Ok(NDArrayNumericTensor::F32(ops::group_normalization(x, scale, bias, num_groups, epsilon)?)),
            (NDArrayNumericTensor::F64(x), NDArrayNumericTensor::F64(scale), NDArrayNumericTensor::F64(bias)) =>
                Ok(NDArrayNumericTensor::F64(ops::group_normalization(x, scale, bias, num_groups, epsilon)?)),
            (NDArrayNumericTensor::BF16(x), NDArrayNumericTensor::BF16(scale), NDArrayNumericTensor::BF16(bias)) =>
                Ok(NDArrayNumericTensor::BF16(ops::group_normalization(x, scale, bias, num_groups, epsilon)?)),
            (NDArrayNumericTensor::F16(x), NDArrayNumericTensor::F16(scale), NDArrayNumericTensor::F16(bias)) =>
                Ok(NDArrayNumericTensor::F16(ops::group_normalization(x, scale, bias, num_groups, epsilon)?)),
            _ => Err(NDArrayNumericTensorError::UnsupportedOperationForDTypes("GroupNorm".to_string(), vec![self.dtype(), scale.dtype(), bias.dtype()]))?
        }
    }

    pub fn concat(tensors: &[&NDArrayNumericTensor], axis: usize) -> Result<Self, NDArrayNumericTensorError> {
        match tensors[0] {
            NDArrayNumericTensor::F32(_) => {
                let mut selected_arrays = vec![];
                for t in tensors{
                    if let NDArrayNumericTensor::F32(x) = t {
                        selected_arrays.push(x.clone());
                    } else {
                        return Err(NDArrayNumericTensorError::InvalidCastOperation(t.dtype(), DType::F32));
                    }
                }
                Ok(NDArrayNumericTensor::F32(ops::concat(axis, &selected_arrays)?))
            }
            NDArrayNumericTensor::F64(_) => {
                let mut selected_arrays = vec![];
                for t in tensors{
                    if let NDArrayNumericTensor::F64(x) = t {
                        selected_arrays.push(x.clone());
                    } else {
                        return Err(NDArrayNumericTensorError::InvalidCastOperation(t.dtype(), DType::F64));
                    }
                }
                Ok(NDArrayNumericTensor::F64(ops::concat(axis, &selected_arrays)?))
            }
            NDArrayNumericTensor::BF16(_) => {
                let mut selected_arrays = vec![];
                for t in tensors{
                    if let NDArrayNumericTensor::BF16(x) = t {
                        selected_arrays.push(x.clone());
                    } else {
                        return Err(NDArrayNumericTensorError::InvalidCastOperation(t.dtype(), DType::BF16));
                    }
                }
                Ok(NDArrayNumericTensor::BF16(ops::concat(axis, &selected_arrays)?))
            }
            NDArrayNumericTensor::F16(_) => {
                let mut selected_arrays = vec![];
                for t in tensors{
                    if let NDArrayNumericTensor::F16(x) = t {
                        selected_arrays.push(x.clone());
                    } else {
                        return Err(NDArrayNumericTensorError::InvalidCastOperation(t.dtype(), DType::F16));
                    }
                }
                Ok(NDArrayNumericTensor::F16(ops::concat(axis, &selected_arrays)?))
            }
            NDArrayNumericTensor::U64(_) => {
                let mut selected_arrays = vec![];
                for t in tensors{
                    if let NDArrayNumericTensor::U64(x) = t {
                        selected_arrays.push(x.clone());
                    } else {
                        return Err(NDArrayNumericTensorError::InvalidCastOperation(t.dtype(), DType::U64));
                    }
                }
                Ok(NDArrayNumericTensor::U64(ops::concat(axis, &selected_arrays)?))
            }
            NDArrayNumericTensor::I64(_) => {
                let mut selected_arrays = vec![];
                for t in tensors{
                    if let NDArrayNumericTensor::I64(x) = t {
                        selected_arrays.push(x.clone());
                    } else {
                        return Err(NDArrayNumericTensorError::InvalidCastOperation(t.dtype(), DType::I64));
                    }
                }
                Ok(NDArrayNumericTensor::I64(ops::concat(axis, &selected_arrays)?))
            }
            NDArrayNumericTensor::U32(_) => {
                let mut selected_arrays = vec![];
                for t in tensors{
                    if let NDArrayNumericTensor::U32(x) = t {
                        selected_arrays.push(x.clone());
                    } else {
                        return Err(NDArrayNumericTensorError::InvalidCastOperation(t.dtype(), DType::U32));
                    }
                }
                Ok(NDArrayNumericTensor::U32(ops::concat(axis, &selected_arrays)?))
            }
            NDArrayNumericTensor::I32(_) => {
                let mut selected_arrays = vec![];
                for t in tensors {
                    if let NDArrayNumericTensor::I32(x) = t {
                        selected_arrays.push(x.clone());
                    } else {
                        return Err(NDArrayNumericTensorError::InvalidCastOperation(t.dtype(), DType::I32));
                    }
                }
                Ok(NDArrayNumericTensor::I32(ops::concat(axis, &selected_arrays)?))
            }
            NDArrayNumericTensor::U16(_) => {
                let mut selected_arrays = vec![];
                for t in tensors {
                    if let NDArrayNumericTensor::U16(x) = t {
                        selected_arrays.push(x.clone());
                    } else {
                        return Err(NDArrayNumericTensorError::InvalidCastOperation(t.dtype(), DType::U16));
                    }
                }
                Ok(NDArrayNumericTensor::U16(ops::concat(axis, &selected_arrays)?))
            }
            NDArrayNumericTensor::I16(_) => {
                let mut selected_arrays = vec![];
                for t in tensors {
                    if let NDArrayNumericTensor::I16(x) = t {
                        selected_arrays.push(x.clone());
                    } else {
                        return Err(NDArrayNumericTensorError::InvalidCastOperation(t.dtype(), DType::U16));
                    }
                }
                Ok(NDArrayNumericTensor::I16(ops::concat(axis, &selected_arrays)?))
            }
            NDArrayNumericTensor::I8(_) => {
                let mut selected_arrays = vec![];
                for t in tensors {
                    if let NDArrayNumericTensor::I8(x) = t {
                        selected_arrays.push(x.clone());
                    } else {
                        return Err(NDArrayNumericTensorError::InvalidCastOperation(t.dtype(), DType::I8));
                    }
                }
                Ok(NDArrayNumericTensor::I8(ops::concat(axis, &selected_arrays)?))
            }
            NDArrayNumericTensor::U8(_) => {
                let mut selected_arrays = vec![];
                for t in tensors {
                    if let NDArrayNumericTensor::U8(x) = t {
                        selected_arrays.push(x.clone());
                    } else {
                        return Err(NDArrayNumericTensorError::InvalidCastOperation(t.dtype(), DType::U8));
                    }
                }
                Ok(NDArrayNumericTensor::U8(ops::concat(axis, &selected_arrays)?))
            }
            NDArrayNumericTensor::BOOL(_) => {
                let mut selected_arrays = vec![];
                for t in tensors {
                    if let NDArrayNumericTensor::BOOL(x) = t {
                        selected_arrays.push(x.clone());
                    } else {
                        return Err(NDArrayNumericTensorError::InvalidCastOperation(t.dtype(), DType::BOOL));
                    }
                }
                Ok(NDArrayNumericTensor::BOOL(ops::concat(axis, &selected_arrays)?))
            }
        }
    }
    
    pub(crate) fn reduce_op(&self, axes: Option<Vec<i64>>, keepdims: bool, op: ReduceOp)  -> Result<NDArrayNumericTensor, NDArrayNumericTensorError> {
        Ok(match self {
            NDArrayNumericTensor::F32(x) => NDArrayNumericTensor::F32(op.apply(x.clone(), axes, keepdims)?),
            NDArrayNumericTensor::F64(x) => NDArrayNumericTensor::F64(op.apply(x.clone(), axes, keepdims)?),
            NDArrayNumericTensor::F16(x) => NDArrayNumericTensor::F16(op.apply(x.clone(), axes, keepdims)?),
            NDArrayNumericTensor::BF16(x) => NDArrayNumericTensor::BF16(op.apply(x.clone(), axes, keepdims)?),
            NDArrayNumericTensor::U32(x) => NDArrayNumericTensor::U32(op.apply(x.clone(), axes, keepdims)?),
            NDArrayNumericTensor::I32(x) => NDArrayNumericTensor::I32(op.apply(x.clone(), axes, keepdims)?),
            NDArrayNumericTensor::U64(x) => NDArrayNumericTensor::U64(op.apply(x.clone(), axes, keepdims)?),
            NDArrayNumericTensor::I64(x) => NDArrayNumericTensor::I64(op.apply(x.clone(), axes, keepdims)?),
            NDArrayNumericTensor::U16(x) => NDArrayNumericTensor::U16(op.apply(x.clone(), axes, keepdims)?),
            NDArrayNumericTensor::I16(x) => NDArrayNumericTensor::I16(op.apply(x.clone(), axes, keepdims)?),
            NDArrayNumericTensor::I8(x) => NDArrayNumericTensor::I8(op.apply(x.clone(), axes, keepdims)?),
            NDArrayNumericTensor::U8(x) => NDArrayNumericTensor::U8(op.apply(x.clone(), axes, keepdims)?),
            _ => Err(NDArrayNumericTensorError::UnsupportedOperationForDTypes("ReduceMean".to_string(), vec![self.dtype()]))?
        })
    }
    
    pub fn reduce_mean(&self, axes: Option<Vec<i64>>, keepdims: bool) -> Result<NDArrayNumericTensor, NDArrayNumericTensorError> {
        self.reduce_op(axes, keepdims, ReduceOp::Mean)
    }

    pub fn reduce_sum(&self, axes: Option<Vec<i64>>, keepdims: bool) -> Result<NDArrayNumericTensor, NDArrayNumericTensorError> {
        self.reduce_op(axes, keepdims, ReduceOp::Sum)
    }

    pub fn gemm(a: &NDArrayNumericTensor, b: &NDArrayNumericTensor, c: Option<&NDArrayNumericTensor>, alpha: f32, beta: f32, transa: bool, transb: bool) -> Result<NDArrayNumericTensor, NDArrayNumericTensorError> {
        Ok(if let Some(c) = c {
            match (a, b, c) {
                (NDArrayNumericTensor::F32(a), NDArrayNumericTensor::F32(b), NDArrayNumericTensor::F32(c)) =>
                    NDArrayNumericTensor::F32(ops::gemm(a.clone(), b.clone(), Some(c.clone()), alpha, beta, transa, transb)?),
                (NDArrayNumericTensor::F64(a), NDArrayNumericTensor::F64(b), NDArrayNumericTensor::F64(c)) =>
                    NDArrayNumericTensor::F64(ops::gemm(a.clone(), b.clone(), Some(c.clone()), alpha.into(), beta.into(), transa, transb)?),
                (NDArrayNumericTensor::BF16(a), NDArrayNumericTensor::BF16(b), NDArrayNumericTensor::BF16(c)) =>
                    NDArrayNumericTensor::BF16(ops::gemm(a.clone(), b.clone(), Some(c.clone()), bf16::from_f32(alpha), bf16::from_f32(beta), transa, transb)?),
                (NDArrayNumericTensor::F16(a), NDArrayNumericTensor::F16(b), NDArrayNumericTensor::F16(c)) =>
                    NDArrayNumericTensor::F16(ops::gemm(a.clone(), b.clone(), Some(c.clone()), f16::from_f32(alpha), f16::from_f32(beta), transa, transb)?),
                (NDArrayNumericTensor::U64(a), NDArrayNumericTensor::U64(b), NDArrayNumericTensor::U64(c)) =>
                    NDArrayNumericTensor::U64(ops::gemm(a.clone(), b.clone(), Some(c.clone()), alpha as u64, beta as u64, transa, transb)?),
                (NDArrayNumericTensor::I64(a), NDArrayNumericTensor::I64(b), NDArrayNumericTensor::I64(c)) =>
                    NDArrayNumericTensor::I64(ops::gemm(a.clone(), b.clone(), Some(c.clone()), alpha as i64, beta as i64, transa, transb)?),
                (NDArrayNumericTensor::U32(a), NDArrayNumericTensor::U32(b), NDArrayNumericTensor::U32(c)) =>
                    NDArrayNumericTensor::U32(ops::gemm(a.clone(), b.clone(), Some(c.clone()), alpha as u32, beta as u32, transa, transb)?),
                (NDArrayNumericTensor::I32(a), NDArrayNumericTensor::I32(b), NDArrayNumericTensor::I32(c)) =>
                    NDArrayNumericTensor::I32(ops::gemm(a.clone(), b.clone(), Some(c.clone()), alpha as i32, beta as i32, transa, transb)?),
                _ => Err(NDArrayNumericTensorError::UnsupportedOperationForDTypes("Gemm".to_string(), vec![a.dtype(), b.dtype(), c.dtype()]))?
            }
        } else {
            match (a, b) {
                (NDArrayNumericTensor::F32(a), NDArrayNumericTensor::F32(b)) =>
                    NDArrayNumericTensor::F32(ops::gemm(a.clone(), b.clone(), None, alpha, beta, transa, transb)?),
                (NDArrayNumericTensor::F64(a), NDArrayNumericTensor::F64(b)) =>
                    NDArrayNumericTensor::F64(ops::gemm(a.clone(), b.clone(), None, alpha as f64, beta as f64, transa, transb)?),
                (NDArrayNumericTensor::BF16(a), NDArrayNumericTensor::BF16(b)) =>
                    NDArrayNumericTensor::BF16(ops::gemm(a.clone(), b.clone(), None, bf16::from_f32(alpha), bf16::from_f32(beta), transa, transb)?),
                (NDArrayNumericTensor::F16(a), NDArrayNumericTensor::F16(b)) =>
                    NDArrayNumericTensor::F16(ops::gemm(a.clone(), b.clone(), None, f16::from_f32(alpha), f16::from_f32(beta), transa, transb)?),
                (NDArrayNumericTensor::U64(a), NDArrayNumericTensor::U64(b)) =>
                    NDArrayNumericTensor::U64(ops::gemm(a.clone(), b.clone(), None, alpha as u64, beta as u64, transa, transb)?),
                (NDArrayNumericTensor::I64(a), NDArrayNumericTensor::I64(b)) =>
                    NDArrayNumericTensor::I64(ops::gemm(a.clone(), b.clone(), None, alpha as i64, beta as i64, transa, transb)?),
                (NDArrayNumericTensor::U32(a), NDArrayNumericTensor::U32(b)) =>
                    NDArrayNumericTensor::U32(ops::gemm(a.clone(), b.clone(), None, alpha as u32, beta as u32, transa, transb)?),
                (NDArrayNumericTensor::I32(a), NDArrayNumericTensor::I32(b)) =>
                    NDArrayNumericTensor::I32(ops::gemm(a.clone(), b.clone(), None, alpha as i32, beta as i32, transa, transb)?),
                _ => Err(NDArrayNumericTensorError::UnsupportedOperationForDTypes("Gemm".to_string(), vec![a.dtype(), b.dtype()]))?
            }
        })
    }

    pub fn split(&self, split: &[i64], axis: i64) -> Result<Vec<NDArrayNumericTensor>, NDArrayNumericTensorError> {
        let axis = Some(axis);
        let split = Some(split);
        Ok(match self {
            NDArrayNumericTensor::F32(x) => ops::split(x.clone(), axis, split)?.into_iter().map(|x| NDArrayNumericTensor::F32(x)).collect(),
            NDArrayNumericTensor::F64(x) => ops::split(x.clone(), axis, split)?.into_iter().map(|x| NDArrayNumericTensor::F64(x)).collect(),
            NDArrayNumericTensor::F16(x) => ops::split(x.clone(), axis, split)?.into_iter().map(|x| NDArrayNumericTensor::F16(x)).collect(),
            NDArrayNumericTensor::BF16(x) => ops::split(x.clone(), axis, split)?.into_iter().map(|x| NDArrayNumericTensor::BF16(x)).collect(),
            NDArrayNumericTensor::U32(x) => ops::split(x.clone(), axis, split)?.into_iter().map(|x| NDArrayNumericTensor::U32(x)).collect(),
            NDArrayNumericTensor::I32(x) => ops::split(x.clone(), axis, split)?.into_iter().map(|x| NDArrayNumericTensor::I32(x)).collect(),
            NDArrayNumericTensor::U64(x) => ops::split(x.clone(), axis, split)?.into_iter().map(|x| NDArrayNumericTensor::U64(x)).collect(),
            NDArrayNumericTensor::I64(x) => ops::split(x.clone(), axis, split)?.into_iter().map(|x| NDArrayNumericTensor::I64(x)).collect(),
            NDArrayNumericTensor::U16(x) => ops::split(x.clone(), axis, split)?.into_iter().map(|x| NDArrayNumericTensor::U16(x)).collect(),
            NDArrayNumericTensor::I16(x) => ops::split(x.clone(), axis, split)?.into_iter().map(|x| NDArrayNumericTensor::I16(x)).collect(),
            NDArrayNumericTensor::I8(x) => ops::split(x.clone(), axis, split)?.into_iter().map(|x| NDArrayNumericTensor::I8(x)).collect(),
            NDArrayNumericTensor::U8(x) => ops::split(x.clone(), axis, split)?.into_iter().map(|x| NDArrayNumericTensor::U8(x)).collect(),
            NDArrayNumericTensor::BOOL(x) => ops::split(x.clone(), axis, split)?.into_iter().map(|x| NDArrayNumericTensor::BOOL(x)).collect(),
        })
    }

    pub fn where_op(&self, a: &NDArrayNumericTensor, b: &NDArrayNumericTensor) -> Result<NDArrayNumericTensor, NDArrayNumericTensorError> {
        let cond = if let NDArrayNumericTensor::BOOL(x) = self { 
            x 
        } 
        else { 
            return Err(NDArrayNumericTensorError::UnsupportedOperationForDTypes("where".to_string(), vec![self.dtype(), a.dtype(), b.dtype()])) 
        };
        let cond = cond.clone();
        
        Ok(match (a, b) {
            (NDArrayNumericTensor::F64(a), NDArrayNumericTensor::F64(b)) => NDArrayNumericTensor::F64(ops::where_op(cond, a.clone(), b.clone())?),
            (NDArrayNumericTensor::F32(a), NDArrayNumericTensor::F32(b)) => NDArrayNumericTensor::F32(ops::where_op(cond, a.clone(), b.clone())?),
            (NDArrayNumericTensor::I64(a), NDArrayNumericTensor::I64(b)) => NDArrayNumericTensor::I64(ops::where_op(cond, a.clone(), b.clone())?),
            (NDArrayNumericTensor::I32(a), NDArrayNumericTensor::I32(b)) => NDArrayNumericTensor::I32(ops::where_op(cond, a.clone(), b.clone())?),
            (NDArrayNumericTensor::I16(a), NDArrayNumericTensor::I16(b)) => NDArrayNumericTensor::I16(ops::where_op(cond, a.clone(), b.clone())?),
            (NDArrayNumericTensor::I8(a), NDArrayNumericTensor::I8(b)) => NDArrayNumericTensor::I8(ops::where_op(cond, a.clone(), b.clone())?),
            (NDArrayNumericTensor::U8(a), NDArrayNumericTensor::U8(b)) => NDArrayNumericTensor::U8(ops::where_op(cond, a.clone(), b.clone())?),
            _ => return Err(NDArrayNumericTensorError::UnsupportedOperationForDTypes("where".to_string(), vec![self.dtype(), a.dtype(), b.dtype()]))
        })
    }
}

