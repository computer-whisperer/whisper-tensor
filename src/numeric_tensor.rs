use std::fmt::{Debug, Formatter};
use std::ops::Range;
use typenum::P1;
use crate::dtype::{DType, DTypeError};
use crate::eval_backend::EvalBackend;
use crate::ndarray_backend::{NDArrayNumericTensor, NDArrayNumericTensorError};

#[cfg(feature = "candle")]
use crate::candle_backend;
use crate::ndarray_backend::conversions::NDArrayNumericTensorType;
use crate::numeric_scalar::NumericScalar;
use crate::numeric_tensor_typed::NumericTensorTyped;
#[cfg(feature = "ort")]
use crate::ort_backend;
use crate::tensor_rank::{DimContainer, DynRank, Rank};
use crate::TrigOp;

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
pub enum NumericTensor<R: Rank> {
    NDArray(NDArrayNumericTensor<R>),
    #[cfg(feature = "onnx-reference")]
    ONNXReference(crate::onnx_reference_backend::ONNXReferenceTensor),
    #[cfg(feature = "candle")]
    Candle(candle_core::Tensor),
    #[cfg(feature = "ort")]
    ORT(ort_backend::ORTNumericTensor),
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
    
    pub fn reshape(&self, new_shape: R::KnownDims) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let NumericTensor::Candle(tensor) = self {
            return Ok(NumericTensor::Candle(tensor.reshape(new_shape.as_slice().iter().map(|x| *x as usize).collect::<Vec<_>>())?))
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

    pub fn neg(&self, backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle(self.to_candle(device)?.neg()?))
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.neg()?))
    }

    pub fn relu(&self, backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle(self.to_candle(device)?.relu()?))
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.relu()?))
    }

    pub fn exp(&self, backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle(self.to_candle(device)?.exp()?))
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.exp()?))
    }

    pub fn ln(&self, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(self.to_ndarray()?.ln()?))
    }

    pub fn floor(&self, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(self.to_ndarray()?.floor()?))
    }

    pub fn ceil(&self, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(self.to_ndarray()?.ceil()?))
    }

    pub fn round(&self, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(self.to_ndarray()?.round()?))
    }

    pub fn erf(&self, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(self.to_ndarray()?.erf()?))
    }


    pub fn abs(&self, backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle(self.to_candle(device)?.abs()?))
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.abs()?))
    }

    pub fn clamp_min(&self, min: f32, _backend: &EvalBackend) -> Result<Self, NumericTensorError>  {
        Ok(NumericTensor::NDArray(self.to_ndarray()?.clamp_min(min)?))
    }

    pub fn sigmoid(&self, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(self.to_ndarray()?.sigmoid()?))
    }

    pub fn trig(&self, op: TrigOp, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(self.to_ndarray()?.trig(op)?))
    }

    pub fn softplus(&self, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(self.to_ndarray()?.softplus()?))
    }

    pub fn reciprocal(&self, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(self.to_ndarray()?.reciprocal()?))
    }

    pub fn sqrt(&self, backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle(self.to_candle(device)?.sqrt()?))
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.sqrt()?))
    }

    pub fn not(&self, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(self.to_ndarray()?.not()?))
    }

    pub fn bitwise_not(&self, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
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
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::<DynRank>::try_from(self)?.slice(indices.into_iter().map(|x| x.start as usize .. x.end as usize ).collect::<Vec<_>>().as_slice())?))
    }

    pub fn unsqueeze(&self, axis: usize, backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle(self.to_candle(device)?.unsqueeze(axis)?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::<DynRank>::try_from(self)?.unsqueeze(axis)?))
    }

    pub fn squeeze(&self, axis: usize, backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle(self.to_candle(device)?.squeeze(axis)?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::<DynRank>::try_from(self)?.squeeze(axis)?))
    }

    pub fn add(a: &Self, b: &Self, backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle(a.to_candle(device)?.broadcast_add(&b.to_candle(device)?)?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::add(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn sub(a: &Self, b: &Self, backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle(a.to_candle(device)?.broadcast_sub(&b.to_candle(device)?)?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::sub(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn div(a: &Self, b: &Self, backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle(a.to_candle(device)?.broadcast_div(&b.to_candle(device)?)?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::div(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn mul(a: &Self, b: &Self, backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle(a.to_candle(device)?.broadcast_mul(&b.to_candle(device)?)?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::mul(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn fmod(a: &Self, b: &Self, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::fmod(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn imod(a: &Self, b: &Self, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::imod(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn matmul(a: &Self, b: &Self, backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle((a.to_candle(device)?.broadcast_matmul(&b.to_candle(device)?))?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::matmul(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn and(a: &Self, b: &Self, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::and(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn or(a: &Self, b: &Self, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::or(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn xor(a: &Self, b: &Self, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::xor(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn bitwise_and(a: &Self, b: &Self, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::bitwise_and(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn bitwise_or(a: &Self, b: &Self, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::bitwise_or(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn bitwise_xor(a: &Self, b: &Self, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::bitwise_xor(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn max(a: &Self, b: &Self, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::max(&a.try_into()?, &b.try_into()?)?))
    }
    
    pub fn min(a: &Self, b: &Self, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::min(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn equal(a: &Self, b: &Self, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::equal(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn greater(a: &Self, b: &Self, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::greater(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn greater_or_equal(a: &Self, b: &Self, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::greater_or_equal(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn less(a: &Self, b: &Self, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::less(&a.try_into()?, &b.try_into()?)?))
    }

    pub fn less_or_equal(a: &Self, b: &Self, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::less_or_equal(&a.try_into()?, &b.try_into()?)?))
    }
    
    pub fn nonzero(&self, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::<DynRank>::try_from(self)?.nonzero()?))
    }

    pub fn pow(&self, exponent: &Self, backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle(self.to_candle(device)?.broadcast_pow(&exponent.to_candle(device)?)?))
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::<DynRank>::try_from(self)?.pow(&NDArrayNumericTensor::<DynRank>::try_from(exponent)?)?))
    }

    pub fn transpose(&self, axes: Option<Vec<i64>>, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::<DynRank>::try_from(self)?.transpose(axes)?))
    }
    
    pub fn is_nan(&self) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(self.to_ndarray()?.is_nan()?))
    }

    pub fn is_inf(&self, detect_positive: bool, detect_negative: bool) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(self.to_ndarray()?.is_inf(detect_positive, detect_negative)?))
    }

    pub fn sign(&self) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(self.to_ndarray()?.sign()?))
    }
    
    pub fn has_nan(&self) -> Result<bool, NumericTensorError> {
        let is_nan = if let Ok(is_nan) = self.is_nan() {
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

    pub fn cast(&self, dtype: DType, backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "candle")]
        if let EvalBackend::Candle(device) = backend {
            return Ok(NumericTensor::Candle(self.to_candle(device)?.to_dtype(dtype.try_into()?)?))
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