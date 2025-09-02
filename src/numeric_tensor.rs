use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::{NDArrayNumericTensor, NDArrayNumericTensorError};
use crate::dtype::{DType, DTypeError};
use std::fmt::{Debug, Formatter};
use std::ops::{Neg, Range};
use typenum::P1;

use crate::TrigOp;
#[cfg(feature = "candle")]
use crate::backends::candle_backend;
use crate::backends::ndarray_backend::conversions::NDArrayNumericTensorType;
#[cfg(feature = "tch")]
use crate::backends::tch_backend::{self, TCHNumericTensor};
#[cfg(feature = "vulkan")]
use crate::backends::vulkan_backend::tensor::VulkanTensor;
#[cfg(feature = "vulkan")]
use crate::backends::vulkan_backend::{VulkanError, VulkanImmediateExecutor};
use crate::numeric_scalar::NumericScalar;
use crate::numeric_tensor_typed::NumericTensorTyped;
use crate::tensor_rank::{DimContainer, DynRank, Rank};

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
    #[cfg(feature = "vulkan")]
    #[error(transparent)]
    Vulkan(#[from] VulkanError),
    #[cfg(feature = "tch")]
    #[error(transparent)]
    TCH(#[from] tch_backend::TCHNumericTensorError),
}

#[derive(Debug, Clone)]
pub enum NumericTensor<R: Rank> {
    NDArray(NDArrayNumericTensor<R>),
    #[cfg(feature = "onnx-reference")]
    ONNXReference(crate::backends::onnx_reference_backend::ONNXReferenceTensor),
    #[cfg(feature = "candle")]
    Candle(candle_core::Tensor),
    #[cfg(feature = "vulkan")]
    Vulkan(VulkanTensor<R>),
    #[cfg(feature = "tch")]
    TCH(tch_backend::TCHNumericTensor<R>),
}

impl<R: Rank> NumericTensor<R> {
    /// Convert this tensor to the NDArray backend without changing shape, dtype, or values.
    ///
    /// - If the tensor is already NDArray, this clones the underlying handle.
    /// - Otherwise it performs a zero-copy view when supported by the backend, or a value-preserving copy.
    /// - Errors if the target backend cannot represent the dtype/shape.
    ///
    /// Returns an NDArrayNumericTensor that contains the same data and metadata.
    pub fn to_ndarray(&self) -> Result<NDArrayNumericTensor<R>, NumericTensorError> {
        match self {
            NumericTensor::NDArray(x) => Ok(x.clone()),
            #[cfg(feature = "onnx-reference")]
            NumericTensor::ONNXReference(x) => Ok(x.to_ndarray()?),
            #[cfg(feature = "candle")]
            NumericTensor::Candle(x) => Ok(x.try_into()?),
            #[cfg(feature = "vulkan")]
            NumericTensor::Vulkan(x) => Ok(x.to_ndarray()),
            #[cfg(feature = "tch")]
            NumericTensor::TCH(x) => Ok(x.to_ndarray()?),
        }
    }

    /// Consume this tensor and convert it into the NDArray backend without changing shape, dtype, or values.
    ///
    /// - If already NDArray, returns the inner handle (clone of pointer/rc if applicable).
    /// - Otherwise performs a zero-copy view when supported or a value-preserving copy.
    /// - Errors if the target backend cannot represent the dtype/shape.
    pub fn into_ndarray(self) -> Result<NDArrayNumericTensor<R>, NumericTensorError> {
        match self {
            NumericTensor::NDArray(x) => Ok(x.clone()),
            #[cfg(feature = "onnx-reference")]
            NumericTensor::ONNXReference(x) => Ok(x.to_ndarray()?),
            #[cfg(feature = "candle")]
            NumericTensor::Candle(x) => Ok(x.try_into()?),
            #[cfg(feature = "vulkan")]
            NumericTensor::Vulkan(x) => Ok(x.to_ndarray()),
            #[cfg(feature = "tch")]
            NumericTensor::TCH(x) => Ok(x.to_ndarray()?),
        }
    }

    #[cfg(feature = "candle")]
    /// Convert or move this tensor to a Candle tensor on the given device.
    ///
    /// - If already Candle, this will move/copy to the target device if needed.
    /// - Otherwise converts from the current backend to Candle, preserving dtype and shape.
    pub fn to_candle(
        &self,
        device: &candle_core::Device,
    ) -> Result<candle_core::Tensor, NumericTensorError> {
        if let NumericTensor::Candle(x) = self {
            Ok(x.to_device(device)?)
        } else {
            Ok(candle_backend::load_to_device(&self.to_ndarray()?, device)?)
        }
    }

    #[cfg(feature = "vulkan")]
    /// Convert this tensor to a Vulkan backend tensor using the provided immediate executor.
    ///
    /// - If already Vulkan, returns a clone.
    /// - Otherwise uploads/converts from host/backend to Vulkan, preserving dtype and shape.
    pub fn to_vulkan(
        &self,
        vulkan_immediate_executor: &mut VulkanImmediateExecutor,
    ) -> Result<VulkanTensor<R>, NumericTensorError> {
        if let NumericTensor::Vulkan(x) = self {
            Ok(x.clone())
        } else {
            Ok(VulkanTensor::from_ndarray(
                self.to_ndarray()?,
                vulkan_immediate_executor,
            )?)
        }
    }

    #[cfg(feature = "tch")]
    /// Convert this tensor to the tch backend tensor, preserving shape and dtype.
    /// If already tch, returns a clone of the handle.
    pub fn to_tch(&self) -> TCHNumericTensor<R> {
        if let NumericTensor::TCH(x) = self {
            x.clone()
        } else {
            TCHNumericTensor::<R>::from_ndarray(self.to_ndarray().unwrap()).unwrap()
        }
    }

    /// Return a view or copy of this tensor with a new shape.
    ///
    /// - new_shape must have the same number of elements as the original shape.
    /// - Implementation may choose the most efficient path (view or copy) while preserving values.
    /// - Errors if reshape is not possible with given shape.
    pub fn reshape(
        &self,
        new_shape: R::KnownDims,
        _backend: &mut EvalBackend,
    ) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = _backend {
            return Ok(NumericTensor::Vulkan(
                self.to_vulkan(executor)?.reshape(new_shape, executor)?,
            ));
        }
        Ok(NumericTensor::NDArray(
            self.to_ndarray()?.reshape(&new_shape)?,
        ))
    }

    /// Get the scalar value at the given multi-dimensional index. Returns None if out of bounds.
    pub fn get(&self, index: &R::KnownDims) -> Option<NumericScalar> {
        self.to_ndarray().unwrap().get(index)
    }

    /// Attempt to reinterpret this tensor as another rank type without changing data.
    /// Fails if the current shape is incompatible with the target rank type.
    pub fn try_to_rank<R1: Rank>(&self) -> Result<NumericTensor<R1>, NumericTensorError> {
        Ok(NumericTensor::<R1>::NDArray(
            self.to_ndarray()?.try_to_rank()?,
        ))
    }

    /// View this tensor as a statically-typed NumericTensorTyped without changing underlying data.
    /// Errors if the underlying dtype cannot be viewed as the target T.
    pub fn try_to_type<T: NDArrayNumericTensorType>(
        &self,
    ) -> Result<NumericTensorTyped<T, R>, NumericTensorError> {
        Ok(NumericTensorTyped::<T, R>::NDArray(
            self.to_ndarray()?.as_inner()?.clone(),
        ))
    }

    /// Convert to a dynamically-ranked tensor view (DynRank), preserving data and shape.
    pub fn to_dyn_rank(&self) -> NumericTensor<DynRank> {
        NumericTensor::NDArray(self.to_ndarray().unwrap().to_dyn())
    }

    /// Return the element dtype of this tensor.
    pub fn dtype(&self) -> DType {
        match self {
            NumericTensor::NDArray(x) => x.dtype(),
            #[cfg(feature = "onnx-reference")]
            NumericTensor::ONNXReference(x) => x.dtype(),
            #[cfg(feature = "candle")]
            NumericTensor::Candle(x) => x.dtype().into(),
            #[cfg(feature = "vulkan")]
            NumericTensor::Vulkan(x) => x.dtype(),
            #[cfg(feature = "tch")]
            NumericTensor::TCH(x) => x.dtype(),
        }
    }

    /// Return the shape of this tensor as rank-typed dimensions.
    pub fn shape(&self) -> R::KnownDims {
        match self {
            NumericTensor::NDArray(x) => x.shape(),
            #[cfg(feature = "onnx-reference")]
            NumericTensor::ONNXReference(x) => {
                let s = x.shape();
                let s2 = s.iter().map(|x| *x as u64).collect::<Vec<_>>();
                R::KnownDims::try_from_slice(s2.as_slice()).unwrap()
            }
            #[cfg(feature = "candle")]
            NumericTensor::Candle(x) => {
                let s = x.shape().dims().to_vec();
                let s2 = s.iter().map(|x| *x as u64).collect::<Vec<_>>();
                R::KnownDims::try_from_slice(s2.as_slice()).unwrap()
            }
            #[cfg(feature = "vulkan")]
            NumericTensor::Vulkan(x) => x.shape().clone(),
            #[cfg(feature = "tch")]
            NumericTensor::TCH(x) => x.shape(),
        }
    }

    /// Return the number of dimensions (rank) of this tensor.
    pub fn rank(&self) -> usize {
        match self {
            NumericTensor::NDArray(x) => x.rank(),
            #[cfg(feature = "onnx-reference")]
            NumericTensor::ONNXReference(x) => x.rank(),
            #[cfg(feature = "candle")]
            NumericTensor::Candle(x) => x.rank(),
            #[cfg(feature = "vulkan")]
            NumericTensor::Vulkan(x) => x.rank(),
            #[cfg(feature = "tch")]
            NumericTensor::TCH(x) => x.rank(),
        }
    }

    /// Return the total number of elements in this tensor (product of shape dims).
    pub fn num_elements(&self) -> u64 {
        self.shape().as_slice().iter().product()
    }

    /// Create a tensor from a flat vector and explicit shape. Length must match the product of shape.
    pub fn from_vec_shape<T>(v: Vec<T>, shape: Vec<usize>) -> Result<Self, NumericTensorError>
    where
        T: NDArrayNumericTensorType,
    {
        Ok(NumericTensor::NDArray(
            NDArrayNumericTensor::<R>::try_from_vec_shape(v, &shape)?,
        ))
    }

    /// Element-wise negation (x -> -x).
    pub fn neg(&self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(self.dtype()) {
            #[cfg(feature = "candle")]
            if let EvalBackend::Candle(device) = backend {
                return Ok(NumericTensor::Candle(self.to_candle(device)?.neg()?));
            }
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(
                    self.to_vulkan(executor)?.neg(executor)?,
                ));
            }
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.neg()?))
    }

    /// Element-wise natural exponential.
    pub fn exp(&self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(self.dtype()) {
            #[cfg(feature = "candle")]
            if let EvalBackend::Candle(device) = backend {
                return Ok(NumericTensor::Candle(self.to_candle(device)?.exp()?));
            }
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(
                    self.to_vulkan(executor)?.exp(executor)?,
                ));
            }
            #[cfg(feature = "tch")]
            if let EvalBackend::TCH = backend {
                return Ok(NumericTensor::TCH(self.to_tch().exp()?));
            }
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.exp()?))
    }

    /// Element-wise natural logarithm.
    pub fn ln(&self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(self.dtype()) {
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(
                    self.to_vulkan(executor)?.ln(executor)?,
                ));
            }
            #[cfg(feature = "tch")]
            if let EvalBackend::TCH = backend {
                return Ok(NumericTensor::TCH(self.to_tch().ln()?));
            }
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.ln()?))
    }

    /// Create a tensor of ones with the same shape and dtype as self.
    pub fn ones_like(&self, _backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(self.to_ndarray()?.ones_like()?))
    }

    /// Element-wise floor (round toward negative infinity).
    pub fn floor(&self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(self.dtype()) {
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(
                    self.to_vulkan(executor)?.floor(executor)?,
                ));
            }
            #[cfg(feature = "tch")]
            if let EvalBackend::TCH = backend {
                return Ok(NumericTensor::TCH(self.to_tch().floor()?));
            }
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.floor()?))
    }

    /// Element-wise ceil (round toward positive infinity).
    pub fn ceil(&self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(self.dtype()) {
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(
                    self.to_vulkan(executor)?.ceil(executor)?,
                ));
            }
            #[cfg(feature = "tch")]
            if let EvalBackend::TCH = backend {
                return Ok(NumericTensor::TCH(self.to_tch().ceil()?));
            }
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.ceil()?))
    }

    /// Element-wise round to nearest integer. Ties are resolved to the nearest even integer.
    pub fn round(&self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(self.dtype()) {
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(
                    self.to_vulkan(executor)?.round(executor)?,
                ));
            }
            #[cfg(feature = "tch")]
            if let EvalBackend::TCH = backend {
                return Ok(NumericTensor::TCH(self.to_tch().round()?));
            }
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.round()?))
    }

    /// Element-wise Gauss error function.
    pub fn erf(&self, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(self.to_ndarray()?.erf()?))
    }

    /// Element-wise absolute value.
    pub fn abs(&self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(self.dtype()) {
            #[cfg(feature = "candle")]
            if let EvalBackend::Candle(device) = backend {
                return Ok(NumericTensor::Candle(self.to_candle(device)?.abs()?));
            }
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(
                    self.to_vulkan(executor)?.abs(executor)?,
                ));
            }
            #[cfg(feature = "tch")]
            if let EvalBackend::TCH = backend {
                return Ok(NumericTensor::TCH(self.to_tch().abs()?));
            }
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.abs()?))
    }

    /// Clamp values below min up to min (element-wise).
    pub fn clamp_min(&self, min: f32, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(self.to_ndarray()?.clamp_min(min)?))
    }

    /// Apply a trigonometric operation element-wise (sin, cos, tan, etc.), as specified by op.
    pub fn trig(&self, op: TrigOp, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(self.dtype()) {
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(
                    self.to_vulkan(executor)?.trig(op, executor)?,
                ));
            }
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.trig(op)?))
    }

    /// Element-wise reciprocal (1/x).
    pub fn reciprocal(&self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(self.dtype()) {
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(
                    self.to_vulkan(executor)?.reciprocal(executor)?,
                ));
            }
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.reciprocal()?))
    }

    /// Element-wise square root.
    pub fn sqrt(&self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(self.dtype()) {
            #[cfg(feature = "candle")]
            if let EvalBackend::Candle(device) = backend {
                return Ok(NumericTensor::Candle(self.to_candle(device)?.sqrt()?));
            }
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(
                    self.to_vulkan(executor)?.sqrt(executor)?,
                ));
            }
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.sqrt()?))
    }

    /// Logical NOT for boolean tensors (element-wise). For non-boolean tensors, elements are compared to zero and the result is logical negation of that predicate.
    pub fn not(&self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(self.dtype()) {
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(
                    self.to_vulkan(executor)?.not(executor)?,
                ));
            }
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.not()?))
    }

    /// Bitwise NOT for integer tensors (element-wise).
    pub fn bitwise_not(&self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(self.dtype()) {
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(
                    self.to_vulkan(executor)?.bitwise_not(executor)?,
                ));
            }
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.bitwise_not()?))
    }

    /// Return the first element (in row-major order) as a scalar.
    pub fn first_element(&self) -> NumericScalar {
        self.to_ndarray().unwrap().first_element()
    }
}

impl NumericTensor<DynRank> {
    /// Cumulative sum along an axis.
    ///
    /// - axis: None means flatten then cumsum; Some(i) reduces along that axis.
    /// - exclusive: if true, uses exclusive cumsum (shifted right, first zero).
    /// - reverse: if true, computes cumsum in reverse order.
    pub fn cumsum(
        &self,
        axis: Option<isize>,
        exclusive: bool,
        reverse: bool,
        backend: &mut EvalBackend,
    ) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(self.dtype()) {
            #[cfg(feature = "tch")]
            if let EvalBackend::TCH = backend {
                let ax = axis.unwrap_or(0) as i64;
                return Ok(NumericTensor::TCH(
                    self.to_tch().cumsum_full(ax, exclusive, reverse)?,
                ));
            }
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                match self
                    .to_vulkan(executor)?
                    .cumsum(axis, exclusive, reverse, executor)
                {
                    Ok(vk) => return Ok(NumericTensor::Vulkan(vk)),
                    Err(VulkanError::UnsupportedByBackendError) => {
                        // Fallback to NDArray for unsupported dtypes on Vulkan
                    }
                    Err(e) => return Err(e.into()),
                }
            }
        }
        Ok(NumericTensor::NDArray(
            self.to_ndarray()?.cumsum(axis, exclusive, reverse)?,
        ))
    }

    /// Concatenate tensors along a given axis.
    ///
    /// - All tensors must have the same shape on non-concatenated axes.
    /// - axis is zero-based.
    /// - Returns a new tensor whose size along axis is the sum of inputs.
    pub fn concat(
        tensors: &[&Self],
        axis: usize,
        _backend: &EvalBackend,
    ) -> Result<Self, NumericTensorError> {
        Ok({
            let ndarrays: Vec<NDArrayNumericTensor<DynRank>> = tensors
                .iter()
                .map(|x| NDArrayNumericTensor::<DynRank>::try_from(*x))
                .collect::<Result<Vec<NDArrayNumericTensor<DynRank>>, NumericTensorError>>()?;
            let ndarrays_ref: Vec<&NDArrayNumericTensor<DynRank>> = ndarrays.iter().collect();
            NumericTensor::NDArray(NDArrayNumericTensor::<DynRank>::concat(
                &ndarrays_ref,
                axis,
            )?)
        })
    }

    /// Indices of maximum values along an axis.
    ///
    /// - axis: zero-based axis to reduce over.
    /// - keepdims: if true, retains reduced dimension with size 1.
    /// - select_last_index: if true, returns the last index of the max; else first.
    pub fn argmax(
        &self,
        axis: usize,
        keepdims: bool,
        select_last_index: bool,
        _backend: &mut EvalBackend,
    ) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(self.to_ndarray()?.argmax(
            axis,
            keepdims,
            select_last_index,
        )?))
    }

    /// Indices of minimum values along an axis. See argmax for semantics of keepdims and select_last_index.
    pub fn argmin(
        &self,
        axis: usize,
        keepdims: bool,
        select_last_index: bool,
        _backend: &mut EvalBackend,
    ) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(self.to_ndarray()?.argmin(
            axis,
            keepdims,
            select_last_index,
        )?))
    }

    /// Slice ranges for each axis, start inclusive and end exclusive.
    ///
    /// - indices length must equal rank; use 0..dim to keep an axis unchanged.
    /// - Returns a view or a copy depending on the underlying representation.
    pub fn slice(
        &self,
        indices: &[Range<u64>],
        _backend: &EvalBackend,
    ) -> Result<Self, NumericTensorError> {
        Ok(match self {
            #[cfg(feature = "vulkan")]
            NumericTensor::Vulkan(tensor) => NumericTensor::Vulkan(tensor.slice(indices)?),
            _ => {
                let indices = indices
                    .iter()
                    .map(|x| x.start as usize..x.end as usize)
                    .collect::<Vec<_>>();
                NumericTensor::NDArray(
                    NDArrayNumericTensor::<DynRank>::try_from(self)?.slice(indices.as_slice())?,
                )
            }
        })
    }

    /// Insert a size-1 dimension at the given axis.
    ///
    /// - axis is zero-based and may be equal to rank to append a new trailing dim.
    /// - Errors if axis > rank.
    pub fn unsqueeze(&self, axis: usize) -> Result<Self, NumericTensorError> {
        Ok(match self {
            #[cfg(feature = "candle")]
            NumericTensor::Candle(x) => NumericTensor::Candle(x.unsqueeze(axis)?),
            #[cfg(feature = "vulkan")]
            NumericTensor::Vulkan(x) => NumericTensor::Vulkan(x.unsqueeze(axis)?),
            _ => NumericTensor::NDArray(
                NDArrayNumericTensor::<DynRank>::try_from(self)?.unsqueeze(axis)?,
            ),
        })
    }

    /// Remove a dimension of size 1 at the given axis.
    ///
    /// - Errors if the axis is out of bounds or the dimension is not 1.
    pub fn squeeze(&self, axis: usize) -> Result<Self, NumericTensorError> {
        Ok(match self {
            #[cfg(feature = "candle")]
            NumericTensor::Candle(x) => NumericTensor::Candle(x.squeeze(axis)?),
            #[cfg(feature = "vulkan")]
            NumericTensor::Vulkan(x) => NumericTensor::Vulkan(x.squeeze(axis)?),
            _ => NumericTensor::NDArray(
                NDArrayNumericTensor::<DynRank>::try_from(self)?.squeeze(axis)?,
            ),
        })
    }

    /// Element-wise addition with NumPy-style broadcasting.
    ///
    /// - Shapes must be broadcast-compatible.
    /// - DType promotion follows standard broadcasting and type-promotion rules; result dtype is the common type of the inputs.
    /// - Errors if dtypes are unsupported or broadcasting fails.
    pub fn add(a: &Self, b: &Self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(a.dtype()) && backend.supports_dtype(b.dtype()) {
            #[cfg(feature = "candle")]
            if let EvalBackend::Candle(device) = backend {
                return Ok(NumericTensor::Candle(
                    a.to_candle(device)?.broadcast_add(&b.to_candle(device)?)?,
                ));
            }
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(VulkanTensor::add(
                    &a.to_vulkan(executor)?,
                    &b.to_vulkan(executor)?,
                    executor,
                )?));
            }
            #[cfg(feature = "tch")]
            if let EvalBackend::TCH = backend {
                return Ok(NumericTensor::TCH(TCHNumericTensor::add(
                    &a.to_tch(),
                    &b.to_tch(),
                )?));
            }
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::add(
            &a.try_into()?,
            &b.try_into()?,
        )?))
    }

    /// Element-wise subtraction with NumPy/ONNX-style broadcasting: a - b.
    ///
    /// - Shapes must be broadcast-compatible.
    /// - DType promotion follows standard type-promotion rules to a common result type.
    /// - Behavior with NaN/Inf follows IEEE-754: NaN propagates; Inf - Inf = NaN of corresponding sign rules.
    pub fn sub(a: &Self, b: &Self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(a.dtype()) && backend.supports_dtype(b.dtype()) {
            #[cfg(feature = "candle")]
            if let EvalBackend::Candle(device) = backend {
                return Ok(NumericTensor::Candle(
                    a.to_candle(device)?.broadcast_sub(&b.to_candle(device)?)?,
                ));
            }
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(VulkanTensor::sub(
                    &a.to_vulkan(executor)?,
                    &b.to_vulkan(executor)?,
                    executor,
                )?));
            }
            #[cfg(feature = "tch")]
            if let EvalBackend::TCH = backend {
                return Ok(NumericTensor::TCH(TCHNumericTensor::sub(
                    &a.to_tch(),
                    &b.to_tch(),
                )?));
            }
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::sub(
            &a.try_into()?,
            &b.try_into()?,
        )?))
    }

    /// Element-wise division with NumPy/ONNX-style broadcasting: a / b.
    ///
    /// ONNX semantics (explicit):
    /// - Broadcasting applies across leading dimensions.
    /// - Result dtype is the common promoted type of inputs.
    /// - Integer division performs floor toward zero (truncating) for signed integers; for unsigned, standard truncating division.
    /// - Division by zero:
    ///   - For floating inputs: x/0 -> +/-inf depending on sign; 0/0 -> NaN.
    ///   - For integer inputs: behavior is to raise error in typical runtimes; if supported here, results are implementation-defined and should not be relied upon.
    pub fn div(a: &Self, b: &Self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(a.dtype()) && backend.supports_dtype(b.dtype()) {
            #[cfg(feature = "candle")]
            if let EvalBackend::Candle(device) = backend {
                return Ok(NumericTensor::Candle(
                    a.to_candle(device)?.broadcast_div(&b.to_candle(device)?)?,
                ));
            }
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(VulkanTensor::div(
                    &a.to_vulkan(executor)?,
                    &b.to_vulkan(executor)?,
                    executor,
                )?));
            }
            #[cfg(feature = "tch")]
            if let EvalBackend::TCH = backend {
                return Ok(NumericTensor::TCH(TCHNumericTensor::div(
                    &a.to_tch(),
                    &b.to_tch(),
                )?));
            }
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::div(
            &a.try_into()?,
            &b.try_into()?,
        )?))
    }

    /// Element-wise multiplication with NumPy/ONNX-style broadcasting: a * b.
    ///
    /// - Shapes must be broadcast-compatible.
    /// - DType promotion follows standard rules; result is the common type.
    /// - Overflows in integer types wrap per two's-complement arithmetic.
    pub fn mul(a: &Self, b: &Self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(a.dtype()) && backend.supports_dtype(b.dtype()) {
            #[cfg(feature = "candle")]
            if let EvalBackend::Candle(device) = backend {
                return Ok(NumericTensor::Candle(
                    a.to_candle(device)?.broadcast_mul(&b.to_candle(device)?)?,
                ));
            }
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(VulkanTensor::mul(
                    &a.to_vulkan(executor)?,
                    &b.to_vulkan(executor)?,
                    executor,
                )?));
            }
            #[cfg(feature = "tch")]
            if let EvalBackend::TCH = backend {
                return Ok(NumericTensor::TCH(TCHNumericTensor::mul(
                    &a.to_tch(),
                    &b.to_tch(),
                )?));
            }
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::mul(
            &a.try_into()?,
            &b.try_into()?,
        )?))
    }

    /// Floating-point remainder (fmod) with broadcasting: remainder of a divided by b.
    ///
    /// ONNX-aligned semantics:
    /// - For floating types, result has the same sign as a and magnitude less than |b|.
    /// - For integer inputs, values are computed in floating domain when necessary to match ONNX behavior, then cast back as needed by the implementation.
    /// - Division by zero yields NaN for floating inputs. For integer inputs, results are undefined and may error.
    pub fn fmod(a: &Self, b: &Self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(a.dtype()) && backend.supports_dtype(b.dtype()) {
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(VulkanTensor::fmod(
                    &a.to_vulkan(executor)?,
                    &b.to_vulkan(executor)?,
                    executor,
                )?));
            }
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::fmod(
            &a.try_into()?,
            &b.try_into()?,
        )?))
    }

    /// Integer modulo with broadcasting: a % b using truncation toward zero for the quotient (ONNX semantics).
    ///
    /// - The sign of the result follows ONNX: r = a - trunc(a/b)*b, so r has the same sign as a and |r| < |b| when b != 0.
    /// - Division by zero is an error.
    /// - Mixed dtypes are promoted to an integer common type when possible; otherwise integers may be promoted to float and the result cast as needed by the implementation.
    pub fn imod(a: &Self, b: &Self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(a.dtype()) && backend.supports_dtype(b.dtype()) {
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(VulkanTensor::imod(
                    &a.to_vulkan(executor)?,
                    &b.to_vulkan(executor)?,
                    executor,
                )?));
            }
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::imod(
            &a.try_into()?,
            &b.try_into()?,
        )?))
    }

    /// Matrix multiplication / batched matmul following ONNX semantics.
    ///
    /// - Supports broadcasting of leading batch dims.
    /// - accumulate_dtype selects accumulator precision when supported; None uses default.
    /// - Errors if inner dimensions are incompatible after broadcasting.
    pub fn matmul(
        a: &Self,
        b: &Self,
        accumulate_dtype: Option<DType>,
        backend: &mut EvalBackend,
    ) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(a.dtype()) && backend.supports_dtype(b.dtype()) {
            #[cfg(feature = "candle")]
            if let EvalBackend::Candle(device) = backend {
                return Ok(NumericTensor::Candle(
                    (a.to_candle(device)?.broadcast_matmul(&b.to_candle(device)?))?,
                ));
            }
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(VulkanTensor::matmul(
                    &a.to_vulkan(executor)?,
                    &b.to_vulkan(executor)?,
                    accumulate_dtype,
                    executor,
                )?));
            }
            #[cfg(feature = "tch")]
            if let EvalBackend::TCH = backend {
                return Ok(NumericTensor::TCH(TCHNumericTensor::matmul(
                    &a.to_tch(),
                    &b.to_tch(),
                    accumulate_dtype,
                )?));
            }
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::matmul(
            &a.try_into()?,
            &b.try_into()?,
            accumulate_dtype,
        )?))
    }

    /// Element-wise logical AND with broadcasting over boolean tensors.
    ///
    /// - Non-boolean inputs are first compared to zero to obtain a boolean view.
    /// - Broadcasting rules apply.
    pub fn and(a: &Self, b: &Self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(a.dtype()) && backend.supports_dtype(b.dtype()) {
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(VulkanTensor::and(
                    &a.to_vulkan(executor)?,
                    &b.to_vulkan(executor)?,
                    executor,
                )?));
            }
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::and(
            &a.try_into()?,
            &b.try_into()?,
        )?))
    }

    /// Element-wise logical OR with broadcasting over boolean tensors.
    ///
    /// - Non-boolean inputs are first compared to zero to obtain a boolean view.
    /// - Broadcasting rules apply.
    pub fn or(a: &Self, b: &Self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(a.dtype()) && backend.supports_dtype(b.dtype()) {
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(VulkanTensor::or(
                    &a.to_vulkan(executor)?,
                    &b.to_vulkan(executor)?,
                    executor,
                )?));
            }
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::or(
            &a.try_into()?,
            &b.try_into()?,
        )?))
    }

    /// Element-wise logical XOR with broadcasting over boolean tensors.
    ///
    /// - Non-boolean inputs are first compared to zero to obtain a boolean view.
    /// - Broadcasting rules apply.
    pub fn xor(a: &Self, b: &Self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(a.dtype()) && backend.supports_dtype(b.dtype()) {
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(VulkanTensor::xor(
                    &a.to_vulkan(executor)?,
                    &b.to_vulkan(executor)?,
                    executor,
                )?));
            }
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::xor(
            &a.try_into()?,
            &b.try_into()?,
        )?))
    }

    /// Element-wise bitwise AND on integer tensors with broadcasting. For boolean tensors, use `and`.
    ///
    /// - Inputs must be integer dtypes (signed or unsigned). Mixed integer widths are promoted to a common integer type.
    /// - Broadcasting rules apply.
    pub fn bitwise_and(
        a: &Self,
        b: &Self,
        backend: &mut EvalBackend,
    ) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(a.dtype()) && backend.supports_dtype(b.dtype()) {
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(VulkanTensor::bitwise_and(
                    &a.to_vulkan(executor)?,
                    &b.to_vulkan(executor)?,
                    executor,
                )?));
            }
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::bitwise_and(
            &a.try_into()?,
            &b.try_into()?,
        )?))
    }

    /// Element-wise bitwise OR on integer tensors with broadcasting. For boolean tensors, use `or`.
    ///
    /// - Inputs must be integer dtypes (signed or unsigned). Mixed integer widths are promoted to a common integer type.
    /// - Broadcasting rules apply.
    pub fn bitwise_or(
        a: &Self,
        b: &Self,
        backend: &mut EvalBackend,
    ) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(a.dtype()) && backend.supports_dtype(b.dtype()) {
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(VulkanTensor::bitwise_or(
                    &a.to_vulkan(executor)?,
                    &b.to_vulkan(executor)?,
                    executor,
                )?));
            }
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::bitwise_or(
            &a.try_into()?,
            &b.try_into()?,
        )?))
    }

    /// Element-wise bitwise XOR on integer tensors with broadcasting. For boolean tensors, use `xor`.
    ///
    /// - Inputs must be integer dtypes (signed or unsigned). Mixed integer widths are promoted to a common integer type.
    /// - Broadcasting rules apply.
    pub fn bitwise_xor(
        a: &Self,
        b: &Self,
        backend: &mut EvalBackend,
    ) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(a.dtype()) && backend.supports_dtype(b.dtype()) {
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(VulkanTensor::bitwise_xor(
                    &a.to_vulkan(executor)?,
                    &b.to_vulkan(executor)?,
                    executor,
                )?));
            }
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::bitwise_xor(
            &a.try_into()?,
            &b.try_into()?,
        )?))
    }

    /// Element-wise maximum with broadcasting.
    ///
    /// ONNX semantics:
    /// - If either operand is NaN at a position, the result is NaN (NaN propagates).
    /// - Inputs are cast to a common type when necessary.
    pub fn max(a: &Self, b: &Self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(a.dtype()) && backend.supports_dtype(b.dtype()) {
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(VulkanTensor::max(
                    &a.to_vulkan(executor)?,
                    &b.to_vulkan(executor)?,
                    executor,
                )?));
            }
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::max(
            &a.try_into()?,
            &b.try_into()?,
        )?))
    }

    /// Element-wise minimum with broadcasting.
    ///
    /// ONNX semantics:
    /// - If either operand is NaN at a position, the result is NaN (NaN propagates).
    /// - Inputs are cast to a common type when necessary.
    pub fn min(a: &Self, b: &Self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(a.dtype()) && backend.supports_dtype(b.dtype()) {
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(VulkanTensor::min(
                    &a.to_vulkan(executor)?,
                    &b.to_vulkan(executor)?,
                    executor,
                )?));
            }
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::min(
            &a.try_into()?,
            &b.try_into()?,
        )?))
    }

    /// Element-wise equality comparison (a == b) with broadcasting.
    ///
    /// ONNX semantics:
    /// - Returns a boolean tensor.
    /// - NaN is not equal to anything, including NaN.
    /// - Broadcasting and type promotion apply to compare values in a common type.
    pub fn equal(
        a: &Self,
        b: &Self,
        backend: &mut EvalBackend,
    ) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(a.dtype()) && backend.supports_dtype(b.dtype()) {
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(VulkanTensor::equal(
                    &a.to_vulkan(executor)?,
                    &b.to_vulkan(executor)?,
                    executor,
                )?));
            }
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::equal(
            &a.try_into()?,
            &b.try_into()?,
        )?))
    }

    /// Element-wise greater-than comparison (a > b) with broadcasting.
    ///
    /// ONNX semantics:
    /// - Returns a boolean tensor.
    /// - Any comparison involving NaN yields false.
    pub fn greater(
        a: &Self,
        b: &Self,
        backend: &mut EvalBackend,
    ) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(a.dtype()) && backend.supports_dtype(b.dtype()) {
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(VulkanTensor::greater(
                    &a.to_vulkan(executor)?,
                    &b.to_vulkan(executor)?,
                    executor,
                )?));
            }
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::greater(
            &a.try_into()?,
            &b.try_into()?,
        )?))
    }

    /// Element-wise greater-or-equal comparison (a >= b) with broadcasting.
    ///
    /// ONNX semantics:
    /// - Returns a boolean tensor.
    /// - Any comparison involving NaN yields false.
    pub fn greater_or_equal(
        a: &Self,
        b: &Self,
        backend: &mut EvalBackend,
    ) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(a.dtype()) && backend.supports_dtype(b.dtype()) {
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(VulkanTensor::greater_or_equal(
                    &a.to_vulkan(executor)?,
                    &b.to_vulkan(executor)?,
                    executor,
                )?));
            }
        }
        Ok(NumericTensor::NDArray(
            NDArrayNumericTensor::greater_or_equal(&a.try_into()?, &b.try_into()?)?,
        ))
    }

    /// Element-wise less-than comparison (a < b) with broadcasting.
    ///
    /// ONNX semantics:
    /// - Returns a boolean tensor.
    /// - Any comparison involving NaN yields false.
    pub fn less(a: &Self, b: &Self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(a.dtype()) && backend.supports_dtype(b.dtype()) {
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(VulkanTensor::less(
                    &a.to_vulkan(executor)?,
                    &b.to_vulkan(executor)?,
                    executor,
                )?));
            }
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::less(
            &a.try_into()?,
            &b.try_into()?,
        )?))
    }

    /// Element-wise less-or-equal comparison (a <= b) with broadcasting.
    ///
    /// ONNX semantics:
    /// - Returns a boolean tensor.
    /// - Any comparison involving NaN yields false.
    pub fn less_or_equal(
        a: &Self,
        b: &Self,
        backend: &mut EvalBackend,
    ) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(a.dtype()) && backend.supports_dtype(b.dtype()) {
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(VulkanTensor::less_or_equal(
                    &a.to_vulkan(executor)?,
                    &b.to_vulkan(executor)?,
                    executor,
                )?));
            }
        }
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::less_or_equal(
            &a.try_into()?,
            &b.try_into()?,
        )?))
    }

    /// Return indices of non-zero elements (like NumPy nonzero) as a 2-D tensor of shape [ndim, count].
    pub fn nonzero(&self, _backend: &EvalBackend) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(
            NDArrayNumericTensor::<DynRank>::try_from(self)?.nonzero()?,
        ))
    }

    /// Element-wise power: self.pow(exponent)
    ///
    /// - Broadcasting is supported between base and exponent.
    /// - Integer bases or exponents are supported; when necessary, values are promoted to a sufficient floating type to perform the operation.
    pub fn pow(
        &self,
        exponent: &Self,
        backend: &mut EvalBackend,
    ) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(self.dtype()) {
            #[cfg(feature = "candle")]
            if let EvalBackend::Candle(device) = backend {
                return Ok(NumericTensor::Candle(
                    self.to_candle(device)?
                        .broadcast_pow(&exponent.to_candle(device)?)?,
                ));
            }
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(
                    self.to_vulkan(executor)?
                        .pow(&exponent.to_vulkan(executor)?, executor)?,
                ));
            }
        }

        Ok(NumericTensor::NDArray(
            NDArrayNumericTensor::<DynRank>::try_from(self)?.pow(&NDArrayNumericTensor::<
                DynRank,
            >::try_from(
                exponent
            )?)?,
        ))
    }

    /// Permute the axes of the tensor.
    ///
    /// - If axes is None, the order is fully reversed.
    /// - If axes is Some, it must be a permutation of 0..rank.
    /// - Returns a view or a copy depending on the underlying representation.
    pub fn transpose(
        &self,
        axes: Option<Vec<i64>>,
        _backend: &EvalBackend,
    ) -> Result<Self, NumericTensorError> {
        Ok(match self {
            #[cfg(feature = "vulkan")]
            NumericTensor::Vulkan(tensor) => {
                let axes = match axes {
                    Some(axes) => axes.iter().map(|x| *x as usize).collect::<Vec<_>>(),
                    None => (0..tensor.shape().len()).rev().collect::<Vec<_>>(),
                };
                NumericTensor::Vulkan(tensor.transpose(&axes)?)
            }
            _ => NumericTensor::NDArray(
                NDArrayNumericTensor::<DynRank>::try_from(self)?.transpose(axes)?,
            ),
        })
    }

    /// Return a boolean tensor marking NaNs in self (element-wise).
    pub fn is_nan(&self, backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(self.dtype()) {
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(
                    self.to_vulkan(executor)?.is_nan(executor)?,
                ));
            }
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.is_nan()?))
    }

    /// Return a boolean tensor marking +inf and/or -inf values in self (element-wise).
    /// - detect_positive: include +inf
    /// - detect_negative: include -inf
    pub fn is_inf(
        &self,
        detect_positive: bool,
        detect_negative: bool,
    ) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(
            self.to_ndarray()?
                .is_inf(detect_positive, detect_negative)?,
        ))
    }

    /// Element-wise sign: returns -1, 0, or 1 for negative, zero, positive. For floating tensors, returns -1.0, 0.0, or 1.0.
    pub fn sign(&self, _backend: &mut EvalBackend) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = _backend {
            return Ok(NumericTensor::Vulkan(
                self.to_vulkan(executor)?.sign(executor)?,
            ));
        }
        Ok(NumericTensor::NDArray(self.to_ndarray()?.sign()?))
    }

    /// Return true if any element is NaN. If the is_nan operation is unavailable, returns false.
    pub fn has_nan(&self, backend: &mut EvalBackend) -> Result<bool, NumericTensorError> {
        let is_nan = if let Ok(is_nan) = self.is_nan(backend) {
            is_nan
        } else {
            return Ok(false);
        };
        let values = is_nan.flatten()?.to_ndarray()?.try_to_vec()?;
        Ok(values.iter().any(|v| *v))
    }

    /// Gather elements from data along an axis using indices (ONNX Gather).
    ///
    /// - axis is zero-based and can be negative like ONNX (negative counts from end).
    /// - indices values select along the given axis; other axes are broadcast.
    /// - Errors if indices are out of bounds or shapes are incompatible.
    pub fn gather(
        data: &Self,
        indices: &Self,
        axis: i64,
        _backend: &mut EvalBackend,
    ) -> Result<Self, NumericTensorError> {
        #[cfg(feature = "vulkan")]
        if let EvalBackend::Vulkan(executor) = _backend {
            return Ok(NumericTensor::Vulkan(VulkanTensor::gather(
                &data.to_vulkan(executor)?,
                &indices.to_vulkan(executor)?,
                axis,
                executor,
            )?));
        }
        Ok(NumericTensor::NDArray(
            NDArrayNumericTensor::<DynRank>::gather(&data.try_into()?, &indices.try_into()?, axis)?,
        ))
    }

    /// Mean-reduction over the specified axes. Returns a tensor of the same dtype as the input unless a wider accumulation type is explicitly requested by the operation.
    pub fn reduce_mean(
        &self,
        axes: Vec<usize>,
        keepdims: bool,
        _backend: &EvalBackend,
    ) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(
            NDArrayNumericTensor::<DynRank>::try_from(self)?.reduce_mean(axes, keepdims)?,
        ))
    }

    /// Sum-reduction over the specified axes.
    ///
    /// - axes are zero-based dimensions to reduce; must be valid for this tensor's rank.
    /// - If keepdims is true, reduced axes are kept with size 1; otherwise they are removed.
    /// - Returns a tensor with the same dtype; integer and floating types keep their type.
    pub fn reduce_sum(
        &self,
        axes: Vec<usize>,
        keepdims: bool,
        _backend: &EvalBackend,
    ) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(
            NDArrayNumericTensor::<DynRank>::try_from(self)?.reduce_sum(axes, keepdims)?,
        ))
    }

    /// Min-reduction over the specified axes.
    pub fn reduce_min(
        &self,
        axes: Vec<usize>,
        keepdims: bool,
        _backend: &EvalBackend,
    ) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(
            NDArrayNumericTensor::<DynRank>::try_from(self)?.reduce_min(axes, keepdims)?,
        ))
    }

    /// Max-reduction over the specified axes.
    pub fn reduce_max(
        &self,
        axes: Vec<usize>,
        keepdims: bool,
        _backend: &EvalBackend,
    ) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(
            NDArrayNumericTensor::<DynRank>::try_from(self)?.reduce_max(axes, keepdims)?,
        ))
    }

    /// Product-reduction over the specified axes.
    pub fn reduce_prod(
        &self,
        axes: Vec<usize>,
        keepdims: bool,
        _backend: &EvalBackend,
    ) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(
            NDArrayNumericTensor::<DynRank>::try_from(self)?.reduce_prod(axes, keepdims)?,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    /// General matrix multiply: alpha * (A op) @ (B op) + beta * C
    ///
    /// - trans_a/trans_b control whether A/B are transposed before matmul.
    /// - C is optional; if provided, must be broadcastable to the resulting matmul shape.
    /// - Accumulation is performed in the specified accumulate_dtype when provided; otherwise the operation uses a reasonable default for numerical stability.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        a: &Self,
        b: &Self,
        c: Option<&Self>,
        alpha: f32,
        beta: f32,
        trans_a: bool,
        trans_b: bool,
        _backend: &EvalBackend,
    ) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(
            NDArrayNumericTensor::<DynRank>::gemm(
                &a.try_into()?,
                &b.try_into()?,
                c.map(NDArrayNumericTensor::<DynRank>::try_from)
                    .transpose()?
                    .as_ref(),
                alpha,
                beta,
                trans_a,
                trans_b,
            )?,
        ))
    }

    /// Split the tensor into multiple chunks along axis with given sizes in split.
    /// The sum of split must equal the size of the axis.
    pub fn split(
        &self,
        split: &[i64],
        axis: i64,
        _backend: &EvalBackend,
    ) -> Result<Vec<Self>, NumericTensorError> {
        Ok({
            let splits = NDArrayNumericTensor::<DynRank>::try_from(self)?.split(split, axis)?;
            let mut out = Vec::new();
            for split in splits {
                out.push(NumericTensor::NDArray(split));
            }
            out
        })
    }

    /// Element-wise select: returns a where self (condition) is true, otherwise b.
    ///
    /// - self must be a boolean tensor broadcastable to a and b.
    /// - a and b must be broadcast-compatible and castable to a common dtype if needed.
    pub fn where_op(
        &self,
        a: &Self,
        b: &Self,
        _backend: &EvalBackend,
    ) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(
            NDArrayNumericTensor::<DynRank>::try_from(self)?
                .where_op(&a.try_into()?, &b.try_into()?)?,
        ))
    }

    /// Cast this tensor to a different dtype, preserving shape and values where representable.
    ///
    /// - Errors if values are out of range for the target type or the cast is not supported.
    pub fn cast(
        &self,
        dtype: DType,
        backend: &mut EvalBackend,
    ) -> Result<Self, NumericTensorError> {
        if backend.supports_dtype(self.dtype()) && backend.supports_dtype(dtype) {
            #[cfg(feature = "candle")]
            if let EvalBackend::Candle(device) = backend {
                return Ok(NumericTensor::Candle(
                    self.to_candle(device)?.to_dtype(dtype.try_into()?)?,
                ));
            }
            #[cfg(feature = "vulkan")]
            if let EvalBackend::Vulkan(executor) = backend {
                return Ok(NumericTensor::Vulkan(
                    self.to_vulkan(executor)?.cast(executor, dtype)?,
                ));
            }
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
    /// Construct a rank-1 tensor from a vector (shape [len]).
    pub fn from_vec<T>(v: Vec<T>) -> Self
    where
        T: NDArrayNumericTensorType,
    {
        NumericTensor::NDArray(NDArrayNumericTensor::from_vec(v))
    }

    pub fn range(
        start: NumericScalar,
        end: NumericScalar,
        step: NumericScalar,
        _backend: &EvalBackend,
    ) -> Result<Self, NumericTensorError> {
        Ok(NumericTensor::NDArray(NDArrayNumericTensor::range(
            start, end, step,
        )?))
    }
}

impl<T: NDArrayNumericTensorType> TryFrom<NumericTensor<P1>> for Vec<T> {
    type Error = NumericTensorError;
    fn try_from(value: NumericTensor<P1>) -> Result<Self, Self::Error> {
        Ok(value.to_ndarray()?.try_to_vec()?)
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
