use ndarray::{ArcArray, Dimension, IntoDimension, Ix1, IxDyn, StrideShape};
use half::{bf16, f16};
use typenum::P1;
use crate::dtype::{DTypeOfPrimitive};
use crate::ndarray_backend::{NDArrayNumericTensor, NDArrayNumericTensorError};
use crate::tensor_rank::{DynRank, Rank};

pub trait NDArrayNumericTensorType: Sized + DTypeOfPrimitive + Clone {
    fn ndarray_numeric_tensor_from_parts(v: Vec<Self>, shape: &[usize]) -> Result<NDArrayNumericTensor<DynRank>, NDArrayNumericTensorError>;
    fn ndarray_numeric_tensor_from_vec(v: Vec<Self>) -> NDArrayNumericTensor<P1>;
    fn ndarray_numeric_tensor_ref_parts<R: Rank>(tensor: &NDArrayNumericTensor<R>) -> Result<(&[Self], &[usize]), NDArrayNumericTensorError>;
    fn ndarray_numeric_tensor_inner<R: Rank>(tensor: &NDArrayNumericTensor<R>) -> Result<&ArcArray<Self, R::NDArrayDim>, NDArrayNumericTensorError>;
    fn ndarray_numeric_tensor_from_ndarray<R: Rank>(value: ArcArray<Self, R::NDArrayDim>) -> NDArrayNumericTensor<R>;
}

impl NDArrayNumericTensor<DynRank> {
    pub(crate) fn from_vec_shape<T: NDArrayNumericTensorType>(v: Vec<T>, shape: &[usize]) -> Result<Self, NDArrayNumericTensorError> {
        T::ndarray_numeric_tensor_from_parts(v, shape)
    }
}

impl<R: Rank> NDArrayNumericTensor<R> {
    pub(crate) fn try_from_vec_shape<T: NDArrayNumericTensorType>(v: Vec<T>, shape: &[usize]) -> Result<Self, NDArrayNumericTensorError> {
        let new_shape = R::try_cast_to_dim(shape)?;
        let new_ndarray: ArcArray<T, R::NDArrayDim> = ArcArray::from_shape_vec(new_shape, v)?;
        Ok(T::ndarray_numeric_tensor_from_ndarray(new_ndarray))
    }
}

impl NDArrayNumericTensor<P1> {
    pub(crate) fn from_vec<T: NDArrayNumericTensorType>(v: Vec<T>) -> Self {
        T::ndarray_numeric_tensor_from_vec(v)
    }
    
    pub(crate) fn try_to_vec<T: NDArrayNumericTensorType>(&self) ->  Result<Vec<T>, NDArrayNumericTensorError> {
        Ok(T::ndarray_numeric_tensor_inner(self)?.to_vec())
    }
}

impl<T: NDArrayNumericTensorType> From<Vec<T>> for NDArrayNumericTensor<P1> {
    fn from(value: Vec<T>) -> Self {
        NDArrayNumericTensor::from_vec(value)
    }
}

impl<R: Rank> NDArrayNumericTensor<R> {
    pub fn as_inner<T: NDArrayNumericTensorType>(&self) -> Result<&ArcArray<T, R::NDArrayDim>, NDArrayNumericTensorError> {
        T::ndarray_numeric_tensor_inner(self)
    }
}

impl<T: NDArrayNumericTensorType> TryFrom<NDArrayNumericTensor<P1>> for Vec<T> {
    type Error = NDArrayNumericTensorError;

    fn try_from(value: NDArrayNumericTensor<P1>) -> Result<Self, Self::Error> {
        value.try_to_vec()
    }
}

#[macro_export]
macro_rules! impl_type_ndarray_backend {
    ($a:ident, $b:ident) => {
        impl NDArrayNumericTensorType for $a
        {
            fn ndarray_numeric_tensor_from_parts(v: Vec<Self>, shape: &[usize]) -> Result<NDArrayNumericTensor<DynRank>, NDArrayNumericTensorError> {
                Ok(NDArrayNumericTensor::$b(ArcArray::from_shape_vec(shape, v)?))
            }
            
            fn ndarray_numeric_tensor_from_vec(v: Vec<Self>) -> NDArrayNumericTensor<P1> {
                NDArrayNumericTensor::$b(ArcArray::from_vec(v))
            }

            fn ndarray_numeric_tensor_ref_parts<R: Rank>(tensor: &NDArrayNumericTensor<R>) -> Result<(&[Self], &[usize]), NDArrayNumericTensorError> {
                if let NDArrayNumericTensor::$b(x) = tensor {
                    Ok((x.as_slice_memory_order().unwrap(), x.shape()))
                }
                else {
                    Err(NDArrayNumericTensorError::WrongDTypeError(Self::DTYPE, tensor.dtype()))
                }
            }

            fn ndarray_numeric_tensor_inner<R: Rank>(tensor: &NDArrayNumericTensor<R>) -> Result<&ArcArray<Self, R::NDArrayDim>, NDArrayNumericTensorError> {
                if let NDArrayNumericTensor::$b(x) = tensor {
                    Ok(x)
                }
                else {
                    Err(NDArrayNumericTensorError::WrongDTypeError(Self::DTYPE, tensor.dtype()))
                }
            }
            
            fn ndarray_numeric_tensor_from_ndarray<R: Rank>(value: ArcArray<Self, R::NDArrayDim>) -> NDArrayNumericTensor<R> {
                NDArrayNumericTensor::<R>::$b(value)
            }
        }
    }
}

impl_type_ndarray_backend!(f64, F64);
impl_type_ndarray_backend!(f32, F32);
impl_type_ndarray_backend!(bf16, BF16);
impl_type_ndarray_backend!(f16, F16);
impl_type_ndarray_backend!(u64, U64);
impl_type_ndarray_backend!(i64, I64);
impl_type_ndarray_backend!(u32, U32);
impl_type_ndarray_backend!(i32, I32);
impl_type_ndarray_backend!(u16, U16);
impl_type_ndarray_backend!(i16, I16);
impl_type_ndarray_backend!(u8, U8);
impl_type_ndarray_backend!(i8, I8);
impl_type_ndarray_backend!(bool, BOOL);


