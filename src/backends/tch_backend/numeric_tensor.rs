use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::dtype::{DType, DTypeError};
use crate::tensor_rank::{DimContainer, DynRank, Rank};
use std::marker::PhantomData;
use tch::{Kind, Tensor};

#[derive(Debug, thiserror::Error)]
pub enum TCHNumericTensorError {
    #[error(transparent)]
    DTypeError(#[from] DTypeError),
    #[error(transparent)]
    TCHError(#[from] tch::TchError),
    #[error("Unsupported DType")]
    UnsupportedDtype,
}

#[derive(Debug)]
pub struct TCHNumericTensor<R: Rank> {
    tensor: Tensor,
    phantom_data: PhantomData<R>,
}

impl<R: Rank> TCHNumericTensor<R> {
    pub fn dtype(&self) -> DType {
        self.tensor.kind().try_into().unwrap()
    }
    pub fn to_ndarray(&self) -> Result<NDArrayNumericTensor<R>, TCHNumericTensorError> {
        let kind = self.tensor.kind();
        match kind {
            Kind::Double => {
                let ndw: ndarray::ArrayD<f64> = (&self.tensor).try_into()?;
                Ok(NDArrayNumericTensor::<DynRank>::F64(ndw.to_shared())
                    .try_to_rank()
                    .unwrap())
            }
            Kind::Float => {
                let ndw: ndarray::ArrayD<f32> = (&self.tensor).try_into()?;
                Ok(NDArrayNumericTensor::<DynRank>::F32(ndw.to_shared())
                    .try_to_rank()
                    .unwrap())
            }
            Kind::BFloat16 => {
                let tensor_f32 = self.tensor.to_kind(Kind::Float);
                let ndw: ndarray::ArrayD<f32> = (&tensor_f32).try_into()?;
                let ndarray_tensor = NDArrayNumericTensor::<DynRank>::F32(ndw.to_shared());
                let bf_tensor = ndarray_tensor.cast(DType::BF16).unwrap();
                Ok(bf_tensor.try_to_rank().unwrap())
            }
            Kind::Half => {
                let ndw: ndarray::ArrayD<half::f16> = (&self.tensor).try_into()?;
                Ok(NDArrayNumericTensor::<DynRank>::F16(ndw.to_shared())
                    .try_to_rank()
                    .unwrap())
            }
            Kind::Int64 => {
                let ndw: ndarray::ArrayD<i64> = (&self.tensor).try_into()?;
                Ok(NDArrayNumericTensor::<DynRank>::I64(ndw.to_shared())
                    .try_to_rank()
                    .unwrap())
            }
            Kind::Int => {
                let ndw: ndarray::ArrayD<i32> = (&self.tensor).try_into()?;
                Ok(NDArrayNumericTensor::<DynRank>::I32(ndw.to_shared())
                    .try_to_rank()
                    .unwrap())
            }
            Kind::Int16 => {
                let ndw: ndarray::ArrayD<i16> = (&self.tensor).try_into()?;
                Ok(NDArrayNumericTensor::<DynRank>::I16(ndw.to_shared())
                    .try_to_rank()
                    .unwrap())
            }
            Kind::Int8 => {
                let ndw: ndarray::ArrayD<i8> = (&self.tensor).try_into()?;
                Ok(NDArrayNumericTensor::<DynRank>::I8(ndw.to_shared())
                    .try_to_rank()
                    .unwrap())
            }
            Kind::Uint8 => {
                let ndw: ndarray::ArrayD<u8> = (&self.tensor).try_into()?;
                Ok(NDArrayNumericTensor::<DynRank>::U8(ndw.to_shared())
                    .try_to_rank()
                    .unwrap())
            }
            _ => Err(DTypeError::UnsupportedTCHDType(kind).into()),
        }
    }

    pub fn from_ndarray(
        orig_value: NDArrayNumericTensor<R>,
    ) -> Result<Self, TCHNumericTensorError> {
        match orig_value {
            NDArrayNumericTensor::F64(value) => Ok(Self {
                tensor: Tensor::try_from(value)?,
                phantom_data: PhantomData::<R>,
            }),
            NDArrayNumericTensor::F32(value) => Ok(Self {
                tensor: Tensor::try_from(value)?,
                phantom_data: PhantomData::<R>,
            }),
            NDArrayNumericTensor::BF16(_) => {
                let value_f32 = orig_value.cast(DType::F32).unwrap();
                let value_f32_inner = match value_f32 {
                    NDArrayNumericTensor::F32(value_f32_inner) => value_f32_inner,
                    _ => unreachable!(),
                };
                let tensor_f32 = Tensor::try_from(value_f32_inner)?;
                let tensor_bf16 = tensor_f32.to_kind(Kind::BFloat16);
                Ok(Self {
                    tensor: tensor_bf16,
                    phantom_data: PhantomData::<R>,
                })
            }
            NDArrayNumericTensor::F16(value) => Ok(Self {
                tensor: Tensor::try_from(value)?,
                phantom_data: PhantomData::<R>,
            }),
            NDArrayNumericTensor::I64(value) => Ok(Self {
                tensor: Tensor::try_from(value)?,
                phantom_data: PhantomData::<R>,
            }),
            NDArrayNumericTensor::I32(value) => Ok(Self {
                tensor: Tensor::try_from(value)?,
                phantom_data: PhantomData::<R>,
            }),
            NDArrayNumericTensor::I16(value) => Ok(Self {
                tensor: Tensor::try_from(value)?,
                phantom_data: PhantomData::<R>,
            }),
            NDArrayNumericTensor::U8(value) => Ok(Self {
                tensor: Tensor::try_from(value)?,
                phantom_data: PhantomData::<R>,
            }),
            NDArrayNumericTensor::I8(value) => Ok(Self {
                tensor: Tensor::try_from(value)?,
                phantom_data: PhantomData::<R>,
            }),
            NDArrayNumericTensor::BOOL(value) => Ok(Self {
                tensor: Tensor::try_from(value)?,
                phantom_data: PhantomData::<R>,
            }),
            _ => Err(DTypeError::DTypeNotSupportedByBackend(orig_value.dtype()).into()),
        }
    }

    pub fn shape(&self) -> R::KnownDims {
        let v = self
            .tensor
            .size()
            .iter()
            .map(|x| *x as u64)
            .collect::<Vec<u64>>();
        R::KnownDims::try_from_slice(v.as_slice()).unwrap()
    }

    pub fn rank(&self) -> usize {
        self.tensor.dim()
    }

    pub fn add(a: &Self, b: &Self) -> Result<Self, TCHNumericTensorError> {
        Ok(Self {
            tensor: &a.tensor + &b.tensor,
            phantom_data: PhantomData::<R>,
        })
    }

    pub fn sub(a: &Self, b: &Self) -> Result<Self, TCHNumericTensorError> {
        Ok(Self {
            tensor: &a.tensor - &b.tensor,
            phantom_data: PhantomData::<R>,
        })
    }

    pub fn mul(a: &Self, b: &Self) -> Result<Self, TCHNumericTensorError> {
        Ok(Self {
            tensor: &a.tensor * &b.tensor,
            phantom_data: PhantomData::<R>,
        })
    }

    pub fn div(a: &Self, b: &Self) -> Result<Self, TCHNumericTensorError> {
        Ok(Self {
            tensor: &a.tensor / &b.tensor,
            phantom_data: PhantomData::<R>,
        })
    }

    pub fn exp(&self) -> Result<Self, TCHNumericTensorError> {
        Ok(Self {
            tensor: self.tensor.exp(),
            phantom_data: PhantomData::<R>,
        })
    }

    pub fn ln(&self) -> Result<Self, TCHNumericTensorError> {
        Ok(Self {
            tensor: self.tensor.log(),
            phantom_data: PhantomData::<R>,
        })
    }

    pub fn abs(&self) -> Result<Self, TCHNumericTensorError> {
        Ok(Self {
            tensor: self.tensor.abs(),
            phantom_data: PhantomData::<R>,
        })
    }

    pub fn floor(&self) -> Result<Self, TCHNumericTensorError> {
        Ok(Self {
            tensor: self.tensor.floor(),
            phantom_data: PhantomData::<R>,
        })
    }

    pub fn ceil(&self) -> Result<Self, TCHNumericTensorError> {
        Ok(Self {
            tensor: self.tensor.ceil(),
            phantom_data: PhantomData::<R>,
        })
    }

    pub fn round(&self) -> Result<Self, TCHNumericTensorError> {
        Ok(Self {
            tensor: self.tensor.round(),
            phantom_data: PhantomData::<R>,
        })
    }

    pub fn cumsum(&self, axis: i64) -> Result<Self, TCHNumericTensorError> {
        let kind = self.tensor.kind();
        Ok(Self {
            tensor: self.tensor.cumsum(axis, kind),
            phantom_data: PhantomData::<R>,
        })
    }

    pub fn reduce_sum(&self, axes: &[i64], keepdims: bool) -> Result<Self, TCHNumericTensorError> {
        let kind = self.tensor.kind();
        Ok(Self {
            tensor: self.tensor.sum_dim_intlist(axes, keepdims, kind),
            phantom_data: PhantomData::<R>,
        })
    }

    pub fn reduce_mean(&self, axes: &[i64], keepdims: bool) -> Result<Self, TCHNumericTensorError> {
        // Implement mean as sum / count to avoid dtype surprises.
        let sizes = self.tensor.size();
        let mut count: i64 = 1;
        for &ax in axes {
            let idx = if ax < 0 {
                (sizes.len() as i64 + ax) as usize
            } else {
                ax as usize
            };
            count *= sizes[idx];
        }
        let sum = self
            .tensor
            .sum_dim_intlist(axes, keepdims, self.tensor.kind());
        let denom = Tensor::from(count as f64).to_kind(self.tensor.kind());
        Ok(Self {
            tensor: &sum / &denom,
            phantom_data: PhantomData::<R>,
        })
    }

    pub fn matmul(
        a: &Self,
        b: &Self,
        accumulate_dtype: Option<DType>,
    ) -> Result<Self, TCHNumericTensorError> {
        match (a.dtype(), b.dtype(), accumulate_dtype) {
            (DType::F64, DType::F64, Some(DType::F64))
            | (DType::F32, DType::F32, Some(DType::F32))
            | (DType::BF16, DType::BF16, Some(DType::F32))
            | (DType::F16, DType::F16, Some(DType::F32))
            | (DType::F32, DType::F32, None)
            | (DType::F64, DType::F64, None) => {
                // This works
            }
            _ => {
                // This doesnt
                return Err(TCHNumericTensorError::UnsupportedDtype);
            }
        }
        let res_tensor = (a.tensor).matmul(&b.tensor);
        Ok(Self {
            tensor: res_tensor,
            phantom_data: PhantomData::<R>,
        })
    }

    pub fn to_string(&self) -> Result<String, TCHNumericTensorError> {
        Ok(self.tensor.to_string(10)?)
    }
}

impl TCHNumericTensor<DynRank> {}

impl<R: Rank> Clone for TCHNumericTensor<R> {
    fn clone(&self) -> Self {
        Self {
            tensor: self.tensor.shallow_clone(),
            phantom_data: PhantomData::<R>,
        }
    }
}
