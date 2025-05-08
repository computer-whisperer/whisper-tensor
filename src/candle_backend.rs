use candle_core::Device;
use crate::RuntimeBackend;
use crate::dtype::DTypeError;
use crate::ndarray_backend::numeric_tensor::{NDArrayNumericTensor};
use crate::numeric_tensor::{NumericTensor, NumericTensorError};
use crate::tensor_rank::{Rank};

pub(crate) fn load_to_device<R: Rank>(value: &NDArrayNumericTensor<R>, device: &Device) -> Result<candle_core::Tensor, NumericTensorError> {
    Ok(match &value {
        NDArrayNumericTensor::F32(x) => {
            let x = x.as_slice_memory_order().unwrap();
            candle_core::Tensor::from_slice(x, value.shape(), device)?
        }
        NDArrayNumericTensor::F64(x) => {
            let x = x.as_slice_memory_order().unwrap();
            candle_core::Tensor::from_slice(x, value.shape(), device)?
        }
        NDArrayNumericTensor::F16(x) => {
            let x = x.as_slice_memory_order().unwrap();
            candle_core::Tensor::from_slice(x, value.shape(), device)?
        }
        NDArrayNumericTensor::BF16(x) => {
            let x = x.as_slice_memory_order().unwrap();
            candle_core::Tensor::from_slice(x, value.shape(), device)?
        }
        NDArrayNumericTensor::U32(x) => {
            let x = x.as_slice_memory_order().unwrap();
            candle_core::Tensor::from_slice(x, value.shape(), device)?
        }
        NDArrayNumericTensor::I64(x) => {
            let x = x.as_slice_memory_order().unwrap();
            candle_core::Tensor::from_slice(x, value.shape(), device)?
        }
        _ => {
            Err(DTypeError::DTypeNotSupportedByBackend(value.dtype(), RuntimeBackend::Candle))?
        }
    })
}

impl<R: Rank> TryFrom<&NDArrayNumericTensor<R>> for candle_core::Tensor {
    type Error = NumericTensorError;
    fn try_from(value: &NDArrayNumericTensor<R>) -> Result<Self, Self::Error> {
        load_to_device(value, &candle_core::Device::Cpu)
    }
}


impl<R: Rank> TryFrom<NDArrayNumericTensor<R>> for candle_core::Tensor {
    type Error = NumericTensorError;
    fn try_from(value: NDArrayNumericTensor<R>) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl<R: Rank> TryFrom<&candle_core::Tensor> for NDArrayNumericTensor<R> {
    type Error = NumericTensorError;
    fn try_from(value: &candle_core::Tensor) -> Result<Self, Self::Error> {
        let shape = value.shape().dims();
        // I don't think there is any way for this to fail, so unwrap
        let tensor_flat = value.flatten_all().unwrap();
        Ok(match value.dtype() {
            candle_core::DType::F64 => {
                let v = tensor_flat.to_vec1::<f64>()?;
                NDArrayNumericTensor::try_from_vec_shape(v, shape)?
            },
            candle_core::DType::U8 => {
                let v = tensor_flat.to_vec1::<u8>()?;
                NDArrayNumericTensor::try_from_vec_shape(v, shape)?
            }
            candle_core::DType::U32 => {
                let v = tensor_flat.to_vec1::<u32>()?;
                NDArrayNumericTensor::try_from_vec_shape(v, shape)?
            }
            candle_core::DType::I64 => {
                let v = tensor_flat.to_vec1::<i64>()?;
                NDArrayNumericTensor::try_from_vec_shape(v, shape)?
            }
            candle_core::DType::BF16 => {
                let v = tensor_flat.to_vec1::<half::bf16>()?;
                NDArrayNumericTensor::try_from_vec_shape(v, shape)?
            }
            candle_core::DType::F16 => {
                let v = tensor_flat.to_vec1::<half::f16>()?;
                NDArrayNumericTensor::try_from_vec_shape(v, shape)?
            }
            candle_core::DType::F32 => {
                let v = tensor_flat.to_vec1::<f32>()?;
                NDArrayNumericTensor::try_from_vec_shape(v, shape)?
            }
        })
    }
}

impl<R: Rank> TryFrom<candle_core::Tensor> for NDArrayNumericTensor<R> {
    type Error = NumericTensorError;
    fn try_from(value: candle_core::Tensor) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl<R: Rank> TryFrom<NumericTensor<R>> for candle_core::Tensor {
    type Error = NumericTensorError;
    fn try_from(value: NumericTensor<R>) -> Result<Self, Self::Error> {
        Ok(match value {
            NumericTensor::Candle(x) => x,
            _ => candle_core::Tensor::try_from(value.to_ndarray()?)?
        })
    }
}

impl<R: Rank> TryFrom<&NumericTensor<R>> for candle_core::Tensor {
    type Error = NumericTensorError;
    fn try_from(value: &NumericTensor<R>) -> Result<Self, Self::Error> {
        Ok(match value {
            NumericTensor::Candle(x) => x.clone(),
            _ => candle_core::Tensor::try_from(value.to_ndarray()?)?
        })
    }
}

impl<R: Rank> From<candle_core::Tensor> for NumericTensor<R> {
    fn from(value: candle_core::Tensor) -> Self {
        Self::Candle(value)
    }
}

impl<R: Rank> From<&candle_core::Tensor> for NumericTensor<R> {
    fn from(value: &candle_core::Tensor) -> Self {
        Self::Candle(value.clone())
    }
}