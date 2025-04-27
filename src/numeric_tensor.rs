use std::fmt::Formatter;
use std::ops::Range;
use crate::dtype::{DType, DTypeError};
use crate::native_numeric_tensor::{FromVecShape, NativeNumericTensor, NativeNumericTensorError};
use crate::ort_backend;

#[derive(Debug, thiserror::Error)]
pub enum NumericTensorError {
    #[error(transparent)]
    DTypeError(#[from] crate::dtype::DTypeError),
    #[error(transparent)]
    NativeNumericTensorError(#[from] NativeNumericTensorError),
    #[cfg(feature = "candle")]
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
    #[cfg(feature = "onnx-reference")]
    #[error(transparent)]
    ONNXReference(#[from] crate::onnx_reference_backend::ONNXReferenceError),
    #[cfg(feature = "ort")]
    #[error(transparent)]
    ORT(#[from] ort::Error),
}

#[derive(Debug, Clone)]
pub enum NumericTensor {
    Native(NativeNumericTensor),
    #[cfg(feature = "onnx-reference")]
    ONNXReference(crate::onnx_reference_backend::ONNXReferenceTensor),
    #[cfg(feature = "candle")]
    Candle(candle_core::Tensor),
    #[cfg(feature = "ort")]
    ORT(ort_backend::ORTNumericTensor),
}

impl NumericTensor {
    pub fn dtype(&self) -> Result<DType, DTypeError> {
        Ok(match self {
            NumericTensor::Native(x) => x.dtype(),
            #[cfg(feature = "onnx-reference")]
            NumericTensor::ONNXReference(x) => x.dtype(),
            #[cfg(feature = "candle")]
            NumericTensor::Candle(x) => x.dtype().into(),
            #[cfg(feature = "ort")]
            NumericTensor::ORT(x) => x.dtype()?,
        })
    }

    pub fn shape(&self) -> Vec<usize> {
        match self {
            NumericTensor::Native(x) => x.shape().to_vec(),
            #[cfg(feature = "onnx-reference")]
            NumericTensor::ONNXReference(x) => x.shape(),
            #[cfg(feature = "candle")]
            NumericTensor::Candle(x) => x.shape().dims().to_vec(),
            #[cfg(feature = "ort")]
            NumericTensor::ORT(x) => x.shape(),
        }
    }
    
    pub fn num_elements(&self) -> usize {
        self.shape().iter().product()
    }

    
    pub fn from_vec_shape<T>(v: Vec<T>, shape: Vec<usize>) -> Result<NumericTensor, NumericTensorError>
       where NativeNumericTensor: FromVecShape<T>
    {
        Ok(NumericTensor::Native(NativeNumericTensor::from_vec_shape(v, shape)?))
    }

    pub fn from_vec1<T>(v: Vec<T>) -> NumericTensor
    where NativeNumericTensor: From<Vec<T>>
    {
        NumericTensor::Native(NativeNumericTensor::from(v))
    }

    pub fn concat(tensors: &[&NumericTensor], axis: usize) -> Result<Self, NumericTensorError> {
        todo!();
    }
    
    pub fn slice(&self, indices: &[Range<usize>]) -> Result<Self, NumericTensorError> {
        match self {
            _ => {
                Ok(NumericTensor::Native(NativeNumericTensor::try_from(self)?.slice(indices)?))
            }
        }
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, NumericTensorError> {
        match self {
            _ => {
                Ok(NumericTensor::Native(NativeNumericTensor::try_from(self)?.reshape(new_shape)?))
            }
        }
    }
    
    pub fn unsqueeze(&self, axis: usize) -> Result<Self, NumericTensorError> {
        match self {
            _ => {
                Ok(NumericTensor::Native(NativeNumericTensor::try_from(self)?.unsqueeze(axis)?))
            }
        }
    }
    
    pub fn squeeze(&self, axis: usize) -> Result<Self, NumericTensorError> {
        match self {
            _ => {
                Ok(NumericTensor::Native(NativeNumericTensor::try_from(self)?.squeeze(axis)?))
            }
        }
    }
}

impl TryFrom<&NumericTensor> for NativeNumericTensor {
    type Error = NumericTensorError;
    fn try_from(value: &NumericTensor) -> Result<Self, Self::Error> {
        match value {
            NumericTensor::Native(x) => Ok(x.clone()),
            #[cfg(feature = "onnx-reference")]
            NumericTensor::ONNXReference(x) => Ok(x.try_into()?),
            #[cfg(feature = "candle")]
            NumericTensor::Candle(x) => Ok(x.try_into()?),
            #[cfg(feature = "ort")]
            NumericTensor::ORT(x) => Ok(x.try_into()?),
        }
    }
}

impl TryFrom<NumericTensor> for NativeNumericTensor {
    type Error = NumericTensorError;
    fn try_from(value: NumericTensor) -> Result<Self, Self::Error> {
        match value {
            NumericTensor::Native(x) => Ok(x),
            #[cfg(feature = "onnx-reference")]
            NumericTensor::ONNXReference(x) => Ok(x.try_into()?),
            #[cfg(feature = "candle")]
            NumericTensor::Candle(x) => Ok((&x).try_into()?),
            #[cfg(feature = "ort")]
            NumericTensor::ORT(x) => Ok((&x).try_into()?),
        }
    }
}

impl core::fmt::Display for NumericTensor {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        NativeNumericTensor::try_from(self).map_err(|_| std::fmt::Error)?.fmt(f)
    }
}

impl From<NativeNumericTensor> for NumericTensor {
    fn from(x: NativeNumericTensor) -> Self {
        NumericTensor::Native(x)
    }
}

impl TryFrom<NumericTensor> for Vec<u32>
{
    type Error = NumericTensorError;

    fn try_from(value: NumericTensor) -> Result<Self, Self::Error> {
        match value {
            _ => {
                Ok(NativeNumericTensor::try_from(value)?.try_into()?)
            }
        }
    }
}

impl TryFrom<NumericTensor> for Vec<f32>
{
    type Error = NumericTensorError;

    fn try_from(value: NumericTensor) -> Result<Self, Self::Error> {
        match value {
            _ => {
                Ok(NativeNumericTensor::try_from(value)?.try_into()?)
            }
        }
    }
}