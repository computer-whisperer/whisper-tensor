use std::fmt::{Debug, Formatter};
use std::ops::Range;
use futures::StreamExt;
use num_traits::real::Real;
use ort::operator::kernel::Kernel;
use tracing_subscriber::filter::FilterExt;
use typenum::P1;
use crate::dtype::{DType, DTypeError};
use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::{NDArrayNumericTensor, NDArrayNumericTensorError};

#[cfg(feature = "candle")]
use crate::backends::candle_backend;
use crate::backends::ndarray_backend::conversions::NDArrayNumericTensorType;
use crate::numeric_scalar::NumericScalar;
use crate::numeric_tensor_typed::NumericTensorTyped;
#[cfg(feature = "ort")]
use crate::backends::ort_backend;
use crate::tensor_rank::{DimContainer, DynRank, Rank};
use crate::TrigOp;
#[cfg(feature = "vulkan")]
use crate::backends::vulkan_backend::tensor::VulkanTensor;
#[cfg(feature = "vulkan")]
use crate::backends::vulkan_backend::{VulkanError, VulkanImmediateExecutor};

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
    ONNXReference(#[from] crate::backends::onnx_reference_backend::ONNXReferenceError),
    #[cfg(feature = "ort")]
    #[error(transparent)]
    ORT(#[from] ort::Error),
    #[cfg(feature = "vulkan")]
    #[error(transparent)]
    Vulkan(#[from] VulkanError),
}

#[derive(Debug, Clone)]
pub enum NumericTensor<R: Rank> {
    NDArray(NDArrayNumericTensor<R>),
    #[cfg(feature = "onnx-reference")]
    ONNXReference(crate::backends::onnx_reference_backend::ONNXReferenceTensor),
    #[cfg(feature = "candle")]
    Candle(candle_core::Tensor),
    #[cfg(feature = "ort")]
    ORT(ort_backend::ORTNumericTensor),
    #[cfg(feature = "vulkan")]
    Vulkan(VulkanTensor<R>)
}

impl<R: Rank> NumericTensor<R> {

    pub fn to_ndarray(&self) -> Result<NDArrayNumericTensor<R>, NumericTensorError> {
        match self {
            NumericTensor::NDArray(x) => Ok(x.clone()),
            #[cfg(feature = "onnx-reference")]
            NumericTensor::ONNXReference(x) => Ok(x.to_ndarray()?),
            #[cfg(feature = "candle")]
            NumericTensor::Candle(x) => Ok(x.try_into()?),
            #[cfg(feature = "ort")]
            NumericTensor::ORT(x) => Ok(x.try_into()?),
            #[cfg(feature = "vulkan")]
            NumericTensor::Vulkan(x) => Ok(x.to_ndarray()),
        }
    }

    pub fn into_ndarray(self) -> Result<NDArrayNumericTensor<R>, NumericTensorError> {
        match self {
            NumericTensor::NDArray(x) => Ok(x.clone()),
            #[cfg(feature = "onnx-reference")]
            NumericTensor::ONNXReference(x) => Ok(x.to_ndarray()?),
            #[cfg(feature = "candle")]
            NumericTensor::Candle(x) => Ok(x.try_into()?),
            #[cfg(feature = "ort")]
            NumericTensor::ORT(x) => Ok(x.try_into()?),
            #[cfg(feature = "vulkan")]
            NumericTensor::Vulkan(x) => Ok(x.to_ndarray()),
        }
    }

    #[cfg(feature = "candle")]
    pub fn to_candle(&self, device: &candle_core::Device) -> Result<candle_core::Tensor, NumericTensorError> {
        if let NumericTensor::Candle(x) = self {
            Ok(x.to_device(device)?)
        }
        else {
            Ok(candle_backend::load_to_device(&self.to_ndarray()?, device)?)
        }
    }

    #[cfg(feature = "vulkan")]
    pub fn to_vulkan(&self, vulkan_immediate_executor: &mut VulkanImmediateExecutor) -> Result<VulkanTensor<R>, NumericTensorError> {
        if let NumericTensor::Vulkan(x) = self {
            Ok(x.clone())
        }
        else {
            Ok(VulkanTensor::from_ndarray(self.to_ndarray()?, vulkan_immediate_executor)?)
        }
    }


    pub fn reshape(&self, new_shape: R::KnownDims, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(self.to_vulkan(executor)?.reshape(new_shape, executor)?))
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.reshape(&new_shape)?))
    }
    
    pub fn get(&self, index: &R::KnownDims) -> Option<NumericScalar> {
        self.to_ndarray().unwrap().get(index)
    }

    pub fn try_to_rank<R1: Rank>(&self) -> Result<NumericTensor<R1>, NumericTensorError> {
        Ok(NumericTensor::<R1>::NDArray(self.to_ndarray()?.try_to_rank()?))
    }
    
    pub fn try_to_type<T: NDArrayNumericTensorType>(&self) -> Result<NumericTensorTyped<T, R>, NumericTensorError> {
        Ok(NumericTensorTyped::<T, R>::NDArray(self.to_ndarray()?.as_inner()?.clone()))
    }
    
    pub fn to_dyn_rank(&self) -> NumericTensor<DynRank> {
        NumericTensor::NDArray(self.to_ndarray().unwrap().to_dyn())
    }

    pub fn dtype(&self) -> DType {
        match self {
            NumericTensor::NDArray(x) => x.dtype(),
            #[cfg(feature = "onnx-reference")]
            NumericTensor::ONNXReference(x) => x.dtype(),
            #[cfg(feature = "candle")]
            NumericTensor::Candle(x) => x.dtype().into(),
            #[cfg(feature = "ort")]
            NumericTensor::ORT(x) => x.dtype().unwrap(),
            #[cfg(feature = "vulkan")]
            NumericTensor::Vulkan(x) => x.dtype(),
        }
    }

    pub fn shape(&self) -> R::KnownDims {
        match self {
            NumericTensor::NDArray(x) => x.shape(),
            #[cfg(feature = "onnx-reference")]
            NumericTensor::ONNXReference(x) => {
                let s = x.shape();
                let s2 = s.iter().map(|x| *x as u64).collect::<Vec<_>>();
                R::KnownDims::try_from_slice(s2.as_slice()).unwrap()
            },
            #[cfg(feature = "candle")]
            NumericTensor::Candle(x) => {
                let s = x.shape().dims().to_vec();
                let s2 = s.iter().map(|x| *x as u64).collect::<Vec<_>>();
                R::KnownDims::try_from_slice(s2.as_slice()).unwrap()
            },
            #[cfg(feature = "ort")]
            NumericTensor::ORT(x) => {
                let s = x.shape();
                let s2 = s.iter().map(|x| *x as u64).collect::<Vec<_>>();
                R::KnownDims::try_from_slice(s2.as_slice()).unwrap()
            },
            #[cfg(feature = "vulkan")]
            NumericTensor::Vulkan(x) => x.shape().clone(),
        }
    }

    pub fn rank(&self) -> usize {
        match self {
            NumericTensor::NDArray(x) => x.rank(),
            #[cfg(feature = "onnx-reference")]
            NumericTensor::ONNXReference(x) => x.rank(),
            #[cfg(feature = "candle")]
            NumericTensor::Candle(x) => x.rank(),
            #[cfg(feature = "ort")]
            NumericTensor::ORT(x) => x.rank(),
            #[cfg(feature = "vulkan")]
            NumericTensor::Vulkan(x) => x.rank(),
        }
    }
    
    pub fn num_elements(&self) -> u64 {
        self.shape().as_slice().iter().product()
    }

    pub fn from_vec_shape<T>(v: Vec<T>, shape: Vec<usize>) -> Result<Self, NumericTensorError>
    where
        T: NDArrayNumericTensorType
    {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::<R>::try_from_vec_shape(v, &shape)?))
    }

    pub fn neg(&self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle(self.to_candle(device)?.neg()?))
        }
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(self.to_vulkan(executor)?.neg(executor)?))
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.neg()?))
    }

    pub fn exp(&self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle(self.to_candle(device)?.exp()?))
        }
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(self.to_vulkan(executor)?.exp(executor)?))
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.exp()?))
    }

    pub fn ln(&self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(self.to_vulkan(executor)?.ln(executor)?))
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.ln()?))
    }

    pub fn floor(&self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(self.to_vulkan(executor)?.floor(executor)?))
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.floor()?))
    }

    pub fn ceil(&self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(self.to_vulkan(executor)?.ceil(executor)?))
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.ceil()?))
    }

    pub fn round(&self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(self.to_vulkan(executor)?.round(executor)?))
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.round()?))
    }

    pub fn erf(&self, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(self.to_ndarray()?.erf()?))
    }

    pub fn abs(&self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle(self.to_candle(device)?.abs()?))
        }
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(self.to_vulkan(executor)?.abs(executor)?))
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.abs()?))
    }

    pub fn clamp_min(&self, min: f32, _backend: &EvalBackend) -> Result<Self, NumericTensorError>  {
        Ok(NumericTensor::NDArray(self.to_ndarray()?.clamp_min(min)?))
    }

    pub fn trig(&self, op: TrigOp, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(self.to_vulkan(executor)?.trig(op, executor)?))
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.trig(op)?))
    }

    pub fn reciprocal(&self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(self.to_vulkan(executor)?.reciprocal(executor)?))
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.reciprocal()?))
    }

    pub fn sqrt(&self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle(self.to_candle(device)?.sqrt()?))
        }
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(self.to_vulkan(executor)?.sqrt(executor)?))
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.sqrt()?))
    }

    pub fn not(&self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(self.to_vulkan(executor)?.not(executor)?))
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.not()?))
    }

    pub fn bitwise_not(&self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(self.to_vulkan(executor)?.bitwise_not(executor)?))
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.bitwise_not()?))
    }

    pub fn first_element(&self) -> NumericScalar {
        self.to_ndarray().unwrap().first_element()
    }
}

impl NumericTensor<DynRank> {

    pub fn concat(tensors: &[&Self], axis: usize, backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(match backend {
            _ => {
                let ndarrays: Vec<NDArrayNumericTensor<DynRank>> =
                    tensors.iter().map(|x| NDArrayNumericTensor::<DynRank>::try_from(*x)).collect::<Result<Vec<NDArrayNumericTensor<DynRank>>, NumericTensorError>>()?;
                let ndarrays_ref: Vec<&NDArrayNumericTensor<DynRank>> = ndarrays.iter().collect();
                NumericTensor::NDArray(NDArrayNumericTensor::<DynRank>::concat(&ndarrays_ref, axis)?)
            }
        })
    }

    pub fn slice(&self, indices: &[Range<u64>], _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(match self {
            NumericTensor::Vulkan(tensor) => {
                NumericTensor::Vulkan(tensor.slice(indices)?)
            }
            _ => {
                let indices = indices.into_iter().map(|x| x.start as usize .. x.end as usize ).collect::<Vec<_>>();
                NumericTensor::NDArray(NDArrayNumericTensor::<DynRank>::try_from(self)?.slice(indices.as_slice())?)
            }
        })
    }

    pub fn unsqueeze(&self, axis: usize) -> Result<Self, NumericTensorError> {
        Ok(match self {
            NumericTensor::Candle(x) => {
                NumericTensor::Candle(x.unsqueeze(axis)?)
            }
            NumericTensor::Vulkan(x) => {
                NumericTensor::Vulkan(x.unsqueeze(axis)?)
            }
            _ => {
                NumericTensor::NDArray(NDArrayNumericTensor::<DynRank>::try_from(self)?.unsqueeze(axis)?)
            }
        })
    }

    pub fn squeeze(&self, axis: usize) -> Result<Self, NumericTensorError> {
        Ok(match self {
            NumericTensor::Candle(x) => {
                NumericTensor::Candle(x.squeeze(axis)?)
            }
            NumericTensor::Vulkan(x) => {
                NumericTensor::Vulkan(x.squeeze(axis)?)
            }
            _ => {
                NumericTensor::NDArray(NDArrayNumericTensor::<DynRank>::try_from(self)?.squeeze(axis)?)
            }
        })
    }

    pub fn add(a: &Self, b: &Self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle(a.to_candle(device)?.broadcast_add(&b.to_candle(device)?)?))
        }
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(VulkanTensor::add(
                &a.to_vulkan(executor)?,
                &b.to_vulkan(executor)?,
                executor
            )?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::add(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn sub(a: &Self, b: &Self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle(a.to_candle(device)?.broadcast_sub(&b.to_candle(device)?)?))
        }
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(VulkanTensor::sub(
                &a.to_vulkan(executor)?,
                &b.to_vulkan(executor)?,
                executor
            )?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::sub(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn div(a: &Self, b: &Self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle(a.to_candle(device)?.broadcast_div(&b.to_candle(device)?)?))
        }
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(VulkanTensor::div(
                &a.to_vulkan(executor)?,
                &b.to_vulkan(executor)?,
                executor
            )?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::div(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn mul(a: &Self, b: &Self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle(a.to_candle(device)?.broadcast_mul(&b.to_candle(device)?)?))
        }
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(VulkanTensor::mul(
                &a.to_vulkan(executor)?,
                &b.to_vulkan(executor)?,
                executor
            )?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::mul(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn fmod(a: &Self, b: &Self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(VulkanTensor::fmod(
                &a.to_vulkan(executor)?,
                &b.to_vulkan(executor)?,
                executor
            )?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::fmod(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn imod(a: &Self, b: &Self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(VulkanTensor::imod(
                &a.to_vulkan(executor)?,
                &b.to_vulkan(executor)?,
                executor
            )?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::imod(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn matmul(a: &Self, b: &Self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle((a.to_candle(device)?.broadcast_matmul(&b.to_candle(device)?))?))
        }
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(VulkanTensor::matmul(
                &a.to_vulkan(executor)?,
                &b.to_vulkan(executor)?,
                executor
            )?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::matmul(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn and(a: &Self, b: &Self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(VulkanTensor::and(
                &a.to_vulkan(executor)?,
                &b.to_vulkan(executor)?,
                executor
            )?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::and(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn or(a: &Self, b: &Self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(VulkanTensor::or(
                &a.to_vulkan(executor)?,
                &b.to_vulkan(executor)?,
                executor
            )?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::or(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn xor(a: &Self, b: &Self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(VulkanTensor::xor(
                &a.to_vulkan(executor)?,
                &b.to_vulkan(executor)?,
                executor
            )?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::xor(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn bitwise_and(a: &Self, b: &Self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(VulkanTensor::bitwise_and(
                &a.to_vulkan(executor)?,
                &b.to_vulkan(executor)?,
                executor
            )?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::bitwise_and(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn bitwise_or(a: &Self, b: &Self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(VulkanTensor::bitwise_or(
                &a.to_vulkan(executor)?,
                &b.to_vulkan(executor)?,
                executor
            )?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::bitwise_or(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn bitwise_xor(a: &Self, b: &Self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(VulkanTensor::bitwise_xor(
                &a.to_vulkan(executor)?,
                &b.to_vulkan(executor)?,
                executor
            )?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::bitwise_xor(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn max(a: &Self, b: &Self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(VulkanTensor::max(
                &a.to_vulkan(executor)?,
                &b.to_vulkan(executor)?,
                executor
            )?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::max(&a.try_into()?, &b.try_into()?)?))
    }
    
    pub fn min(a: &Self, b: &Self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(VulkanTensor::min(
                &a.to_vulkan(executor)?,
                &b.to_vulkan(executor)?,
                executor
            )?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::min(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn equal(a: &Self, b: &Self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(VulkanTensor::equal(
                &a.to_vulkan(executor)?,
                &b.to_vulkan(executor)?,
                executor
            )?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::equal(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn greater(a: &Self, b: &Self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(VulkanTensor::greater(
                &a.to_vulkan(executor)?,
                &b.to_vulkan(executor)?,
                executor
            )?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::greater(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn greater_or_equal(a: &Self, b: &Self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(VulkanTensor::greater_or_equal(
                &a.to_vulkan(executor)?,
                &b.to_vulkan(executor)?,
                executor
            )?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::greater_or_equal(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn less(a: &Self, b: &Self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(VulkanTensor::less(
                &a.to_vulkan(executor)?,
                &b.to_vulkan(executor)?,
                executor
            )?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::less(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn less_or_equal(a: &Self, b: &Self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(VulkanTensor::less_or_equal(
                &a.to_vulkan(executor)?,
                &b.to_vulkan(executor)?,
                executor
            )?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::less_or_equal(&a.try_into()?, &b.try_into()?)?))
    }
    
    pub fn nonzero(&self, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::<DynRank>::try_from(self)?.nonzero()?))
    }

    pub fn pow(&self, exponent: &Self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle(self.to_candle(device)?.broadcast_pow(&exponent.to_candle(device)?)?))
        }
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(self.to_vulkan(executor)?.pow(&exponent.to_vulkan(executor)?, executor)?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::<DynRank>::try_from(self)?.pow(&NDArrayNumericTensor::<DynRank>::try_from(exponent)?)?))
    }

    pub fn transpose(&self, axes: Option<Vec<i64>>, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(match self {
            NumericTensor::Vulkan(tensor) => {
                let axes = match axes {
                    Some(axes) => {
                        axes.iter().map(|x| *x as usize).collect::<Vec<_>>()
                    }
                    None => {
                        (0..tensor.shape().len()).rev().collect::<Vec<_>>()
                    }
                };
                NumericTensor::Vulkan(tensor.transpose(&axes)?)
            }
            _ => {
                NumericTensor::NDArray(NDArrayNumericTensor::<DynRank>::try_from(self)?.transpose(axes)?)
            }
        })
    }
    
    pub fn is_nan(&self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(self.to_vulkan(executor)?.is_nan(executor)?))
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.is_nan()?))
    }

    pub fn is_inf(&self, detect_positive: bool, detect_negative: bool) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(self.to_ndarray()?.is_inf(detect_positive, detect_negative)?))
    }

    pub fn sign(&self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(self.to_vulkan(executor)?.sign(executor)?))
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.sign()?))
    }
    
    pub fn has_nan(&self, backend: &mut EvalBackend) -> Result<bool, NumericTensorError> {
        let is_nan = if let Ok(is_nan) = self.is_nan(backend) {
            is_nan
        } else {
            return Ok(false)
        };
        let values = is_nan.flatten()?.to_ndarray()?.try_to_vec()?;
        Ok(values.iter().any(|v| *v))
    }

    pub fn gather(data: &Self, indices: &Self, axis: i64, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::<DynRank>::gather(&data.try_into()?, &indices.try_into()?, axis)?), )
    }

    pub fn reduce_mean(&self, axes: Vec<usize>, keepdims: bool, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::<DynRank>::try_from(self)?.reduce_mean(axes, keepdims)?))
    }

    pub fn reduce_sum(&self, axes: Vec<usize>, keepdims: bool, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::<DynRank>::try_from(self)?.reduce_sum(axes, keepdims)?))
    }

    pub fn reduce_min(&self, axes: Vec<usize>, keepdims: bool, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::<DynRank>::try_from(self)?.reduce_min(axes, keepdims)?))
    }

    pub fn reduce_max(&self, axes: Vec<usize>, keepdims: bool, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::<DynRank>::try_from(self)?.reduce_max(axes, keepdims)?))
    }

    pub fn reduce_prod(&self, axes: Vec<usize>, keepdims: bool, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::<DynRank>::try_from(self)?.reduce_prod(axes, keepdims)?))
    }

    pub fn gemm(a: &Self, b: &Self, c: Option<&Self>, alpha: f32, beta: f32, trans_a: bool, trans_b: bool, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::<DynRank>::gemm(&a.try_into()?, &b.try_into()?, c.map(|x| NDArrayNumericTensor::<DynRank>::try_from(x)).transpose()?.as_ref(), alpha, beta, trans_a, trans_b)?), )
    }

    pub fn split(&self, split: &[i64], axis: i64, backend: &EvalBackend) -> Result<Vec<Self>, NumericTensorError> {
        Ok(match backend {
            _ => {
                let splits = NDArrayNumericTensor::<DynRank>::try_from(self)?.split(split, axis)?;
                let mut out = Vec::new();
                for split in splits {
                    out.push(NumericTensor::NDArray(split));
                }
                out
            }
        })
    }

    pub fn where_op(&self, a: &Self, b: &Self, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::<DynRank>::try_from(self)?.where_op(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn cast(&self, dtype: DType, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle(self.to_candle(device)?.to_dtype(dtype.try_into()?)?))
        }
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = backend {
            return Ok(NumericTensor::Vulkan(self.to_vulkan(executor)?.cast(executor, dtype)?))
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.cast(dtype)?))
    }

    pub fn flatten(&self) -> Result<NumericTensor<P1>, NumericTensorError> {
        Ok(NumericTensor::NDArray(self.to_ndarray()?.flatten()))
    }
    
    pub fn expand(&self, shape: &[u64]) -> Result<NumericTensor<DynRank>, NumericTensorError> {
        Ok(NumericTensor::NDArray(self.to_ndarray()?.expand(shape)?))
    }
}

impl NumericTensor<P1> {
    pub fn from_vec<T>(v: Vec<T>) -> Self
    where
        T: NDArrayNumericTensorType
    {
        NumericTensor::NDArray(NDArrayNumericTensor::from_vec(v))
    }

    pub fn range(start: NumericScalar, end: NumericScalar, step: NumericScalar, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::range(start, end, step)?))
    }
}

impl<T: NDArrayNumericTensorType> TryFrom<NumericTensor<P1>> for Vec<T>
{
    type Error = NumericTensorError;
    fn try_from(value: NumericTensor<P1>) -> Result<Self, Self::Error> {
        Ok( value.to_ndarray()?.try_to_vec()?)
    }
}

impl TryFrom<&NumericTensor<DynRank>> for NDArrayNumericTensor<DynRank> {
    type Error = NumericTensorError;
    fn try_from(value: &NumericTensor<DynRank>) -> Result<Self, Self::Error> {
        value.to_ndarray()
    }
}

impl TryFrom<NumericTensor<DynRank>> for NDArrayNumericTensor<DynRank> {
    type Error = NumericTensorError;
    fn try_from(value: NumericTensor<DynRank>) -> Result<Self, Self::Error> {
        value.into_ndarray()
    }
}

impl core::fmt::Display for NumericTensor<DynRank> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.to_ndarray().map_err(|_| std::fmt::Error)?, f)
    }
}

impl From<NDArrayNumericTensor<DynRank>> for NumericTensor<DynRank> {
    fn from(x: NDArrayNumericTensor<DynRank>) -> Self {
        NumericTensor::NDArray(x)
    }
}