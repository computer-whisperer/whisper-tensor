use ndarray::{ArcArray};
use half::{bf16, f16};
use crate::dtype::{DTypeOfPrimitive};
use crate::ndarray_backend::{NDArrayNumericTensor, NDArrayNumericTensorError};

pub trait NDArrayNumericTensorType: Sized + DTypeOfPrimitive {
    fn from_parts(v: Vec<Self>, shape: &[usize]) -> Result<NDArrayNumericTensor, NDArrayNumericTensorError>;
    fn ref_parts(tensor: &NDArrayNumericTensor) -> Result<(&[Self], &[usize]), NDArrayNumericTensorError>;
}

impl NDArrayNumericTensor {
    pub(crate) fn from_vec_shape<T: NDArrayNumericTensorType>(v: Vec<T>, shape: &[usize]) -> Result<NDArrayNumericTensor, NDArrayNumericTensorError> {
        T::from_parts(v, shape)
    }
    pub(crate) fn to_1d_vec<T: NDArrayNumericTensorType + Clone>(&self) -> Result<Vec<T>, NDArrayNumericTensorError> {
        let (a, b) = T::ref_parts(self)?;
        assert_eq!(b.len(), 1);
        Ok(a.to_vec())
    }
}

#[macro_export]
macro_rules! impl_type {
    ($a:ident, $b:ident) => {
        impl NDArrayNumericTensorType for $a
        {
            fn from_parts(v: Vec<Self>, shape: &[usize]) -> Result<NDArrayNumericTensor, NDArrayNumericTensorError> {
                Ok(NDArrayNumericTensor::$b(ArcArray::from_shape_vec(shape, v)?))
            }
        
            fn ref_parts(tensor: &NDArrayNumericTensor) -> Result<(&[Self], &[usize]), NDArrayNumericTensorError> {
                if let NDArrayNumericTensor::$b(x) = tensor {
                    Ok((x.as_slice_memory_order().unwrap(), x.shape()))
                }
                else {
                    Err(NDArrayNumericTensorError::WrongDTypeError(Self::DTYPE, tensor.dtype()))
                }
            }
        }
    }
}


impl_type!(f64, F64);
impl_type!(f32, F32);
impl_type!(bf16, BF16);
impl_type!(f16, F16);
impl_type!(u64, U64);
impl_type!(i64, I64);
impl_type!(u32, U32);
impl_type!(i32, I32);
impl_type!(u16, U16);
impl_type!(i16, I16);
impl_type!(u8, U8);
impl_type!(i8, I8);
impl_type!(bool, BOOL);

impl<T: NDArrayNumericTensorType + Clone> TryFrom<&NDArrayNumericTensor> for Vec<T> {
    type Error = NDArrayNumericTensorError;

    fn try_from(value: &NDArrayNumericTensor) -> Result<Self, Self::Error> {
        let (a, b) = T::ref_parts(value)?;
        assert_eq!(b.len(), 1);
        Ok(a.to_vec())
    }
}

impl<T: NDArrayNumericTensorType + Clone> TryFrom<NDArrayNumericTensor> for Vec<T> {
    type Error = NDArrayNumericTensorError;

    fn try_from(value: NDArrayNumericTensor) -> Result<Self, Self::Error> {
        let (a, b) = T::ref_parts(&value)?;
        assert_eq!(b.len(), 1);
        Ok(a.to_vec())
    }
}

impl<T: NDArrayNumericTensorType> From<Vec<T>> for NDArrayNumericTensor {
    fn from(value: Vec<T>) -> Self {
        let len = value.len();
        NDArrayNumericTensor::from_vec_shape(value, &[len]).unwrap()
    }
}
