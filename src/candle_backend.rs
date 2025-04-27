use ndarray::ArcArray;
use crate::Backend;
use crate::dtype::DTypeError;
use crate::native_numeric_tensor::{NativeNumericTensor, NativeNumericTensorError};
use crate::numeric_tensor::{NumericTensor, NumericTensorError};

impl TryFrom<&NativeNumericTensor> for candle_core::Tensor {
    type Error = NumericTensorError;
    fn try_from(value: &NativeNumericTensor) -> Result<Self, Self::Error> {
        Ok(match &value {
            NativeNumericTensor::F32(x) => {
                let x = x.as_slice_memory_order().unwrap();
                candle_core::Tensor::from_slice(x, value.shape(), &candle_core::Device::Cpu)?
            }
            NativeNumericTensor::F64(x) => {
                let x = x.as_slice_memory_order().unwrap();
                candle_core::Tensor::from_slice(x, value.shape(), &candle_core::Device::Cpu)?
            }
            NativeNumericTensor::F16(x) => {
                let x = x.as_slice_memory_order().unwrap();
                candle_core::Tensor::from_slice(x, value.shape(), &candle_core::Device::Cpu)?
            }
            NativeNumericTensor::BF16(x) => {
                let x = x.as_slice_memory_order().unwrap();
                candle_core::Tensor::from_slice(x, value.shape(), &candle_core::Device::Cpu)?
            }
            NativeNumericTensor::U32(x) => {
                let x = x.as_slice_memory_order().unwrap();
                candle_core::Tensor::from_slice(x, value.shape(), &candle_core::Device::Cpu)?
            }
            NativeNumericTensor::I64(x) => {
                let x = x.as_slice_memory_order().unwrap();
                candle_core::Tensor::from_slice(x, value.shape(), &candle_core::Device::Cpu)?
            }
            _ => {
                Err(DTypeError::DTypeNotSupportedByBackend(value.dtype(), Backend::Candle))?
            }
        })
    }
}


impl TryFrom<NativeNumericTensor> for candle_core::Tensor {
    type Error = NumericTensorError;
    fn try_from(value: NativeNumericTensor) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl TryFrom<&candle_core::Tensor> for NativeNumericTensor {
    type Error = NumericTensorError;
    fn try_from(value: &candle_core::Tensor) -> Result<Self, Self::Error> {
        let shape = value.shape().dims();
        // I don't think there is any way for this to fail, so unwrap
        let tensor_flat = value.flatten_all().unwrap();
        Ok(match value.dtype() {
            candle_core::DType::F64 => {
                let v = tensor_flat.to_vec1::<f64>()?;
                NativeNumericTensor::F64(ArcArray::from_shape_vec(shape, v).map_err(NativeNumericTensorError::ShapeError)?)
            },
            candle_core::DType::U8 => {
                let v = tensor_flat.to_vec1::<u8>()?;
                NativeNumericTensor::U8(ArcArray::from_shape_vec(shape, v).map_err(NativeNumericTensorError::ShapeError)?)
            }
            candle_core::DType::U32 => {
                let v = tensor_flat.to_vec1::<u32>()?;
                NativeNumericTensor::U32(ArcArray::from_shape_vec(shape, v).map_err(NativeNumericTensorError::ShapeError)?)
            }
            candle_core::DType::I64 => {
                let v = tensor_flat.to_vec1::<i64>()?;
                NativeNumericTensor::I64(ArcArray::from_shape_vec(shape, v).map_err(NativeNumericTensorError::ShapeError)?)
            }
            candle_core::DType::BF16 => {
                let v = tensor_flat.to_vec1::<half::bf16>()?;
                NativeNumericTensor::BF16(ArcArray::from_shape_vec(shape, v).map_err(NativeNumericTensorError::ShapeError)?)
            }
            candle_core::DType::F16 => {
                let v = tensor_flat.to_vec1::<half::f16>()?;
                NativeNumericTensor::F16(ArcArray::from_shape_vec(shape, v).map_err(NativeNumericTensorError::ShapeError)?)
            }
            candle_core::DType::F32 => {
                let v = tensor_flat.to_vec1::<f32>()?;
                NativeNumericTensor::F32(ArcArray::from_shape_vec(shape, v).map_err(NativeNumericTensorError::ShapeError)?)
            }
        })
    }
}

impl TryFrom<candle_core::Tensor> for NativeNumericTensor {
    type Error = NumericTensorError;
    fn try_from(value: candle_core::Tensor) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl TryFrom<NumericTensor> for candle_core::Tensor {
    type Error = NumericTensorError;
    fn try_from(value: NumericTensor) -> Result<Self, Self::Error> {
        Ok(match value {
            NumericTensor::Candle(x) => x,
            _ => candle_core::Tensor::try_from(NativeNumericTensor::try_from(value)?)?
        })
    }
}

impl TryFrom<&NumericTensor> for candle_core::Tensor {
    type Error = NumericTensorError;
    fn try_from(value: &NumericTensor) -> Result<Self, Self::Error> {
        Ok(match value {
            NumericTensor::Candle(x) => x.clone(),
            _ => candle_core::Tensor::try_from(NativeNumericTensor::try_from(value)?)?
        })
    }
}

impl From<candle_core::Tensor> for NumericTensor {
    fn from(value: candle_core::Tensor) -> Self {
        Self::Candle(value)
    }
}

impl From<&candle_core::Tensor> for NumericTensor {
    fn from(value: &candle_core::Tensor) -> Self {
        Self::Candle(value.clone())
    }
}