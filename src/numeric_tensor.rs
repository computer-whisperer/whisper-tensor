use std::fmt::Formatter;
use std::ops::Range;

use crate::dtype::{DType, DTypeError};
use crate::eval_backend::EvalBackend;
use crate::ndarray_backend::{NDArrayNumericTensor, NDArrayNumericTensorError};
use crate::ndarray_backend::conversions::FromVecShape;

#[cfg(feature = "candle")]
use crate::candle_backend;

#[cfg(feature = "ort")]
use crate::ort_backend;

#[derive(Debug, thiserror::Error)]
pub enum NumericTensorError {
    #[error(transparent)]
    DTypeError(#[from] DTypeError),
    #[error(transparent)]
    NDArrayNumericTensorError(#[from] NDArrayNumericTensorError),
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
    NDArray(NDArrayNumericTensor),
    #[cfg(feature = "onnx-reference")]
    ONNXReference(crate::onnx_reference_backend::ONNXReferenceTensor),
    #[cfg(feature = "candle")]
    Candle(candle_core::Tensor),
    #[cfg(feature = "ort")]
    ORT(ort_backend::ORTNumericTensor),
}

impl NumericTensor {
    #[cfg(feature = "candle")]
    pub fn to_candle(&self, device: &candle_core::Device) -> Result<candle_core::Tensor, NumericTensorError> {
        if let NumericTensor::Candle(x) = self {
            Ok(x.to_device(device)?)
        }
        else {
            Ok(candle_backend::load_to_device(&NDArrayNumericTensor::try_from(self)?, device)?)
        }
    }
    
    pub fn dtype(&self) -> Result<DType, DTypeError> {
        Ok(match self {
            NumericTensor::NDArray(x) => x.dtype(),
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
            NumericTensor::NDArray(x) => x.shape().to_vec(),
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
       where NDArrayNumericTensor: FromVecShape<T>
    {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::from_vec_shape(v, shape)?))
    }

    pub fn from_vec1<T>(v: Vec<T>) -> NumericTensor
    where NDArrayNumericTensor: From<Vec<T>>
    {
        NumericTensor::NDArray(NDArrayNumericTensor::from(v))
    }

    pub fn concat(tensors: &[&NumericTensor], axis: usize, backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(match backend {
            _ => { 
                let ndarrays: Vec<NDArrayNumericTensor> = tensors.iter().map(|x| NDArrayNumericTensor::try_from(*x)).collect::<Result<Vec<NDArrayNumericTensor>, NumericTensorError>>()?;
                let ndarrays_ref: Vec<&NDArrayNumericTensor> = ndarrays.iter().collect();
                NumericTensor::NDArray(NDArrayNumericTensor::concat(&ndarrays_ref, axis)?)
            }
        })
    }
    
    pub fn slice(&self, indices: &[Range<usize>], _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::try_from(self)?.slice(indices)?))
    }

    pub fn reshape(&self, new_shape: Vec<usize>, backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle(self.to_candle(device)?.reshape(new_shape)?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::try_from(self)?.reshape(new_shape)?))
    }
    
    pub fn unsqueeze(&self, axis: usize, backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle(self.to_candle(device)?.unsqueeze(axis)?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::try_from(self)?.unsqueeze(axis)?))
    }
    
    pub fn squeeze(&self, axis: usize, backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle(self.to_candle(device)?.squeeze(axis)?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::try_from(self)?.squeeze(axis)?))
    }
    
    pub fn add(a: &NumericTensor, b: &NumericTensor, backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle((a.to_candle(device)?+b.to_candle(device)?)?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::add(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn sub(a: &NumericTensor, b: &NumericTensor, backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle((a.to_candle(device)?-b.to_candle(device)?)?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::sub(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn div(a: &NumericTensor, b: &NumericTensor, backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle((a.to_candle(device)?/b.to_candle(device)?)?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::div(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn mul(a: &NumericTensor, b: &NumericTensor, backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle((a.to_candle(device)?*b.to_candle(device)?)?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::mul(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn matmul(a: &NumericTensor, b: &NumericTensor, backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle((a.to_candle(device)?.broadcast_matmul(&b.to_candle(device)?))?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::matmul(&a.try_into()?, &b.try_into()?)?))
    }
    
    pub fn neg(&self, backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle(self.to_candle(device)?.neg()?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::try_from(self)?.neg()?))
    }

    pub fn relu(&self, backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle(self.to_candle(device)?.relu()?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::try_from(self)?.relu()?))
    }

    pub fn exp(&self, backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle(self.to_candle(device)?.exp()?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::try_from(self)?.exp()?))
    }

    pub fn sigmoid(&self, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::try_from(self)?.sigmoid()?))
    }

    pub fn tanh(&self, backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle(self.to_candle(device)?.tanh()?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::try_from(self)?.tanh()?))
    }

    pub fn softplus(&self, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::try_from(self)?.softplus()?))
    }

    pub fn nonzero(&self, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::try_from(self)?.nonzero()?))
    }

    pub fn sqrt(&self, backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle(self.to_candle(device)?.sqrt()?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::try_from(self)?.sqrt()?))
    }

    pub fn pow(&self, exponent: &NumericTensor, backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle(self.to_candle(device)?.pow(&exponent.to_candle(device)?)?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::try_from(self)?.pow(&NDArrayNumericTensor::try_from(self)?)?))
    }
    
    pub fn cast(&self, dtype: DType, backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle(self.to_candle(device)?.to_dtype(dtype.try_into()?)?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::try_from(self)?.cast(dtype)?))
    }
    
    pub fn transpose(&self, axes: Option<Vec<i64>>, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::try_from(self)?.transpose(axes)?))
    }

    pub fn gather(data: &NumericTensor, indices: &NumericTensor, axis: i64, _backend: &EvalBackend) -> Result<NumericTensor, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::gather(&data.try_into()?, &indices.try_into()?, axis)?), )
    }

    pub fn reduce_mean(&self, axes: Option<Vec<i64>>, keepdims: bool, _backend: &EvalBackend) -> Result<NumericTensor, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::try_from(self)?.reduce_mean(axes, keepdims)?))
    }

    pub fn reduce_sum(&self, axes: Option<Vec<i64>>, keepdims: bool, _backend: &EvalBackend) -> Result<NumericTensor, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::try_from(self)?.reduce_sum(axes, keepdims)?))
    }
    
    pub fn gemm(a: &NumericTensor, b: &NumericTensor, c: Option<&NumericTensor>, alpha: f32, beta: f32, trans_a: bool, trans_b: bool, backend: &EvalBackend) -> Result<NumericTensor, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::gemm(&a.try_into()?, &b.try_into()?, c.map(|x| NDArrayNumericTensor::try_from(x)).transpose()?.as_ref(), alpha, beta, trans_a, trans_b)?), )
    }
    
    pub fn split(&self, split: &[i64], axis: i64, backend: &EvalBackend) -> Result<Vec<NumericTensor>, NumericTensorError> {
        Ok(match backend {
            _ => {
                let splits = NDArrayNumericTensor::try_from(self)?.split(split, axis)?;
                let mut out = Vec::new();
                for split in splits {
                    out.push(NumericTensor::NDArray(split));
                }
                out
            }
        })
    }
    
    pub fn where_op(&self, a: &NumericTensor, b: &NumericTensor, backend: &EvalBackend) -> Result<NumericTensor, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::try_from(self)?.where_op(&a.try_into()?, &b.try_into()?)?))
    }
}

impl<T> TryFrom<NumericTensor> for Vec<T> 
where
    Vec<T>: TryFrom<NDArrayNumericTensor>,
    <Vec<T> as TryFrom<NDArrayNumericTensor>>::Error: Into<NumericTensorError>,
{
    type Error = NumericTensorError;
    fn try_from(value: NumericTensor) -> Result<Self, Self::Error> {
        let ndarray_tensor = NDArrayNumericTensor::try_from(value)?;
        Ok(Self::try_from(ndarray_tensor).map_err(|e| e.into())?)
    }
}

impl TryFrom<&NumericTensor> for NDArrayNumericTensor {
    type Error = NumericTensorError;
    fn try_from(value: &NumericTensor) -> Result<Self, Self::Error> {
        match value {
            NumericTensor::NDArray(x) => Ok(x.clone()),
            #[cfg(feature = "onnx-reference")]
            NumericTensor::ONNXReference(x) => Ok(x.try_into()?),
            #[cfg(feature = "candle")]
            NumericTensor::Candle(x) => Ok(x.try_into()?),
            #[cfg(feature = "ort")]
            NumericTensor::ORT(x) => Ok(x.try_into()?),
        }
    }
}

impl TryFrom<NumericTensor> for NDArrayNumericTensor {
    type Error = NumericTensorError;
    fn try_from(value: NumericTensor) -> Result<Self, Self::Error> {
        match value {
            NumericTensor::NDArray(x) => Ok(x),
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
        NDArrayNumericTensor::try_from(self).map_err(|_| std::fmt::Error)?.fmt(f)
    }
}

impl From<NDArrayNumericTensor> for NumericTensor {
    fn from(x: NDArrayNumericTensor) -> Self {
        NumericTensor::NDArray(x)
    }
}