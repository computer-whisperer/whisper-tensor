use ndarray::{ArcArray, Ix1, IxDyn};
use half::{bf16, f16};
use crate::dtype::DType;
use crate::ndarray_backend::{NDArrayNumericTensor, NDArrayNumericTensorError};
use crate::numeric_tensor::TryToFlatVec;

pub trait FromScalarShape<T>: Sized {
    fn from_scalar_shape(v: T, s: Vec<usize>) -> Result<Self, NDArrayNumericTensorError>;
}

pub trait FromVecShape<T>: Sized {
    fn from_vec_shape(v: Vec<T>, s: Vec<usize>) -> Result<Self, NDArrayNumericTensorError>;
}

pub trait TryToSlice<T> {
    fn try_to_slice(&self) -> Result<&[T], NDArrayNumericTensorError>;
}

impl TryFrom<NDArrayNumericTensor> for Vec<u32>
{
    type Error = NDArrayNumericTensorError;
    fn try_from(value: NDArrayNumericTensor) -> Result<Vec<u32>, NDArrayNumericTensorError> {
        assert_eq!(value.shape().len(), 1);
        if let NDArrayNumericTensor::U32(x) = value {
            Ok(x.into_dimensionality::<Ix1>()?.to_vec())
        }
        else {
            Err(NDArrayNumericTensorError::WrongDTypeError(DType::U32, value.dtype()))
        }
    }
}

impl TryFrom<NDArrayNumericTensor> for Vec<u16>
{
    type Error = NDArrayNumericTensorError;
    fn try_from(value: NDArrayNumericTensor) -> Result<Vec<u16>, NDArrayNumericTensorError> {
        assert_eq!(value.shape().len(), 1);
        if let NDArrayNumericTensor::U16(x) = value {
            Ok(x.into_dimensionality::<Ix1>()?.to_vec())
        }
        else {
            Err(NDArrayNumericTensorError::WrongDTypeError(DType::U16, value.dtype()))
        }
    }
}

impl TryFrom<NDArrayNumericTensor> for Vec<i16>
{
    type Error = NDArrayNumericTensorError;
    fn try_from(value: NDArrayNumericTensor) -> Result<Vec<i16>, NDArrayNumericTensorError> {
        assert_eq!(value.shape().len(), 1);
        if let NDArrayNumericTensor::I16(x) = value {
            Ok(x.into_dimensionality::<Ix1>()?.to_vec())
        }
        else {
            Err(NDArrayNumericTensorError::WrongDTypeError(DType::I16, value.dtype()))
        }
    }
}

impl TryFrom<NDArrayNumericTensor> for Vec<u8>
{
    type Error = NDArrayNumericTensorError;
    fn try_from(value: NDArrayNumericTensor) -> Result<Vec<u8>, NDArrayNumericTensorError> {
        assert_eq!(value.shape().len(), 1);
        if let NDArrayNumericTensor::U8(x) = value {
            Ok(x.into_dimensionality::<Ix1>()?.to_vec())
        }
        else {
            Err(NDArrayNumericTensorError::WrongDTypeError(DType::U8, value.dtype()))
        }
    }
}

impl TryFrom<NDArrayNumericTensor> for Vec<i8>
{
    type Error = NDArrayNumericTensorError;
    fn try_from(value: NDArrayNumericTensor) -> Result<Vec<i8>, NDArrayNumericTensorError> {
        assert_eq!(value.shape().len(), 1);
        if let NDArrayNumericTensor::I8(x) = value {
            Ok(x.into_dimensionality::<Ix1>()?.to_vec())
        }
        else {
            Err(NDArrayNumericTensorError::WrongDTypeError(DType::I8, value.dtype()))
        }
    }
}

impl TryFrom<NDArrayNumericTensor> for Vec<bool>
{
    type Error = NDArrayNumericTensorError;
    fn try_from(value: NDArrayNumericTensor) -> Result<Vec<bool>, NDArrayNumericTensorError> {
        assert_eq!(value.shape().len(), 1);
        if let NDArrayNumericTensor::BOOL(x) = value {
            Ok(x.into_dimensionality::<Ix1>()?.to_vec())
        }
        else {
            Err(NDArrayNumericTensorError::WrongDTypeError(DType::BOOL, value.dtype()))
        }
    }
}

impl TryFrom<NDArrayNumericTensor> for Vec<u64>
{
    type Error = NDArrayNumericTensorError;
    fn try_from(value: NDArrayNumericTensor) -> Result<Vec<u64>, NDArrayNumericTensorError> {
        assert_eq!(value.shape().len(), 1);
        if let NDArrayNumericTensor::U64(x) = value {
            Ok(x.into_dimensionality::<Ix1>()?.to_vec())
        }
        else {
            Err(NDArrayNumericTensorError::WrongDTypeError(DType::U64, value.dtype()))
        }
    }
}

impl TryFrom<NDArrayNumericTensor> for Vec<i64>
{
    type Error = NDArrayNumericTensorError;
    fn try_from(value: NDArrayNumericTensor) -> Result<Vec<i64>, NDArrayNumericTensorError> {
        assert_eq!(value.shape().len(), 1);
        if let NDArrayNumericTensor::I64(x) = value {
            Ok(x.into_dimensionality::<Ix1>()?.to_vec())
        }
        else {
            Err(NDArrayNumericTensorError::WrongDTypeError(DType::I64, value.dtype()))
        }
    }
}

impl TryFrom<NDArrayNumericTensor> for Vec<i32>
{
    type Error = NDArrayNumericTensorError;
    fn try_from(value: NDArrayNumericTensor) -> Result<Vec<i32>, NDArrayNumericTensorError> {
        assert_eq!(value.shape().len(), 1);
        if let NDArrayNumericTensor::I32(x) = value {
            Ok(x.into_dimensionality::<Ix1>()?.to_vec())
        }
        else {
            Err(NDArrayNumericTensorError::WrongDTypeError(DType::I32, value.dtype()))
        }
    }
}

impl TryFrom<NDArrayNumericTensor> for Vec<f32>
{
    type Error = NDArrayNumericTensorError;
    fn try_from(value: NDArrayNumericTensor) -> Result<Vec<f32>, NDArrayNumericTensorError> {
        assert_eq!(value.shape().len(), 1);
        if let NDArrayNumericTensor::F32(x) = value {
            Ok(x.into_dimensionality::<Ix1>()?.to_vec())
        }
        else {
            Err(NDArrayNumericTensorError::WrongDTypeError(DType::F32, value.dtype()))
        }
    }
}

impl TryFrom<NDArrayNumericTensor> for Vec<f64>
{
    type Error = NDArrayNumericTensorError;
    fn try_from(value: NDArrayNumericTensor) -> Result<Vec<f64>, NDArrayNumericTensorError> {
        assert_eq!(value.shape().len(), 1);
        if let NDArrayNumericTensor::F64(x) = value {
            Ok(x.into_dimensionality::<Ix1>()?.to_vec())
        }
        else {
            Err(NDArrayNumericTensorError::WrongDTypeError(DType::F64, value.dtype()))
        }
    }
}

impl TryFrom<NDArrayNumericTensor> for Vec<bf16>
{
    type Error = NDArrayNumericTensorError;
    fn try_from(value: NDArrayNumericTensor) -> Result<Vec<bf16>, NDArrayNumericTensorError> {
        assert_eq!(value.shape().len(), 1);
        if let NDArrayNumericTensor::BF16(x) = value {
            Ok(x.into_dimensionality::<Ix1>()?.to_vec())
        }
        else {
            Err(NDArrayNumericTensorError::WrongDTypeError(DType::BF16, value.dtype()))
        }
    }
}

impl TryFrom<NDArrayNumericTensor> for Vec<f16>
{
    type Error = NDArrayNumericTensorError;
    fn try_from(value: NDArrayNumericTensor) -> Result<Vec<f16>, NDArrayNumericTensorError> {
        assert_eq!(value.shape().len(), 1);
        if let NDArrayNumericTensor::F16(x) = value {
            Ok(x.into_dimensionality::<Ix1>()?.to_vec())
        }
        else {
            Err(NDArrayNumericTensorError::WrongDTypeError(DType::F16, value.dtype()))
        }
    }
}

impl FromScalarShape<f32> for NDArrayNumericTensor {
    fn from_scalar_shape(v: f32, s: Vec<usize>) -> Result<Self, NDArrayNumericTensorError> {
        Ok(NDArrayNumericTensor::F32(ArcArray::<_, IxDyn>::from_elem(s, v)))
    }
}

impl FromScalarShape<f64> for NDArrayNumericTensor {
    fn from_scalar_shape(v: f64, s: Vec<usize>) -> Result<Self, NDArrayNumericTensorError> {
        Ok(NDArrayNumericTensor::F64(ArcArray::<_, IxDyn>::from_elem(s, v)))
    }
}

impl FromScalarShape<bf16> for NDArrayNumericTensor {
    fn from_scalar_shape(v: bf16, s: Vec<usize>) -> Result<Self, NDArrayNumericTensorError> {
        Ok(NDArrayNumericTensor::BF16(ArcArray::<_, IxDyn>::from_elem(s, v)))
    }
}

impl FromScalarShape<f16> for NDArrayNumericTensor {
    fn from_scalar_shape(v: f16, s: Vec<usize>) -> Result<Self, NDArrayNumericTensorError> {
        Ok(NDArrayNumericTensor::F16(ArcArray::<_, IxDyn>::from_elem(s, v)))
    }
}

impl FromScalarShape<u64> for NDArrayNumericTensor {
    fn from_scalar_shape(v: u64, s: Vec<usize>) -> Result<Self, NDArrayNumericTensorError> {
        Ok(NDArrayNumericTensor::U64(ArcArray::<_, IxDyn>::from_elem(s, v)))
    }
}

impl FromScalarShape<i64> for NDArrayNumericTensor {
    fn from_scalar_shape(v: i64, s: Vec<usize>) -> Result<Self, NDArrayNumericTensorError> {
        Ok(NDArrayNumericTensor::I64(ArcArray::<_, IxDyn>::from_elem(s, v)))
    }
}

impl FromScalarShape<u32> for NDArrayNumericTensor {
    fn from_scalar_shape(v: u32, s: Vec<usize>) -> Result<Self, NDArrayNumericTensorError> {
        Ok(NDArrayNumericTensor::U32(ArcArray::<_, IxDyn>::from_elem(s, v)))
    }
}

impl FromScalarShape<u16> for NDArrayNumericTensor {
    fn from_scalar_shape(v: u16, s: Vec<usize>) -> Result<Self, NDArrayNumericTensorError> {
        Ok(NDArrayNumericTensor::U16(ArcArray::<_, IxDyn>::from_elem(s, v)))
    }
}

impl FromScalarShape<i16> for NDArrayNumericTensor {
    fn from_scalar_shape(v: i16, s: Vec<usize>) -> Result<Self, NDArrayNumericTensorError> {
        Ok(NDArrayNumericTensor::I16(ArcArray::<_, IxDyn>::from_elem(s, v)))
    }
}

impl FromScalarShape<i32> for NDArrayNumericTensor {
    fn from_scalar_shape(v: i32, s: Vec<usize>) -> Result<Self, NDArrayNumericTensorError> {
        Ok(NDArrayNumericTensor::I32(ArcArray::<_, IxDyn>::from_elem(s, v)))
    }
}

impl FromScalarShape<u8> for NDArrayNumericTensor {
    fn from_scalar_shape(v: u8, s: Vec<usize>) -> Result<Self, NDArrayNumericTensorError> {
        Ok(NDArrayNumericTensor::U8(ArcArray::<_, IxDyn>::from_elem(s, v)))
    }
}

impl FromScalarShape<i8> for NDArrayNumericTensor {
    fn from_scalar_shape(v: i8, s: Vec<usize>) -> Result<Self, NDArrayNumericTensorError> {
        Ok(NDArrayNumericTensor::I8(ArcArray::<_, IxDyn>::from_elem(s, v)))
    }
}

impl FromScalarShape<bool> for NDArrayNumericTensor {
    fn from_scalar_shape(v: bool, s: Vec<usize>) -> Result<Self, NDArrayNumericTensorError> {
        Ok(NDArrayNumericTensor::BOOL(ArcArray::<_, IxDyn>::from_elem(s, v)))
    }
}

impl FromVecShape<f32> for NDArrayNumericTensor {
    fn from_vec_shape(v: Vec<f32>, s: Vec<usize>) -> Result<Self, NDArrayNumericTensorError> {
        Ok(NDArrayNumericTensor::F32(ArcArray::<_, IxDyn>::from_shape_vec(s, v)?))
    }
}

impl FromVecShape<f64> for NDArrayNumericTensor {
    fn from_vec_shape(v: Vec<f64>, s: Vec<usize>) -> Result<Self, NDArrayNumericTensorError> {
        Ok(NDArrayNumericTensor::F64(ArcArray::<_, IxDyn>::from_shape_vec(s, v)?))
    }
}

impl FromVecShape<u32> for NDArrayNumericTensor {
    fn from_vec_shape(v: Vec<u32>, s: Vec<usize>) -> Result<Self, NDArrayNumericTensorError> {
        Ok(NDArrayNumericTensor::U32(ArcArray::<_, IxDyn>::from_shape_vec(s, v)?))
    }
}

impl FromVecShape<i32> for NDArrayNumericTensor {
    fn from_vec_shape(v: Vec<i32>, s: Vec<usize>) -> Result<Self, NDArrayNumericTensorError> {
        Ok(NDArrayNumericTensor::I32(ArcArray::<_, IxDyn>::from_shape_vec(s, v)?))
    }
}

impl FromVecShape<u16> for NDArrayNumericTensor {
    fn from_vec_shape(v: Vec<u16>, s: Vec<usize>) -> Result<Self, NDArrayNumericTensorError> {
        Ok(NDArrayNumericTensor::U16(ArcArray::<_, IxDyn>::from_shape_vec(s, v)?))
    }
}

impl FromVecShape<i16> for NDArrayNumericTensor {
    fn from_vec_shape(v: Vec<i16>, s: Vec<usize>) -> Result<Self, NDArrayNumericTensorError> {
        Ok(NDArrayNumericTensor::I16(ArcArray::<_, IxDyn>::from_shape_vec(s, v)?))
    }
}

impl FromVecShape<u8> for NDArrayNumericTensor {
    fn from_vec_shape(v: Vec<u8>, s: Vec<usize>) -> Result<Self, NDArrayNumericTensorError> {
        Ok(NDArrayNumericTensor::U8(ArcArray::<_, IxDyn>::from_shape_vec(s, v)?))
    }
}

impl FromVecShape<i8> for NDArrayNumericTensor {
    fn from_vec_shape(v: Vec<i8>, s: Vec<usize>) -> Result<Self, NDArrayNumericTensorError> {
        Ok(NDArrayNumericTensor::I8(ArcArray::<_, IxDyn>::from_shape_vec(s, v)?))
    }
}

impl FromVecShape<u64> for NDArrayNumericTensor {
    fn from_vec_shape(v: Vec<u64>, s: Vec<usize>) -> Result<Self, NDArrayNumericTensorError> {
        Ok(NDArrayNumericTensor::U64(ArcArray::<_, IxDyn>::from_shape_vec(s, v)?))
    }
}

impl FromVecShape<i64> for NDArrayNumericTensor {
    fn from_vec_shape(v: Vec<i64>, s: Vec<usize>) -> Result<Self, NDArrayNumericTensorError> {
        Ok(NDArrayNumericTensor::I64(ArcArray::<_, IxDyn>::from_shape_vec(s, v)?))
    }
}

impl From<Vec::<f32>> for NDArrayNumericTensor {
    fn from(v: Vec::<f32>) -> Self {
        NDArrayNumericTensor::F32(ArcArray::from_vec(v).into_dyn())
    }
}

impl From<Vec::<f64>> for NDArrayNumericTensor {
    fn from(v: Vec::<f64>) -> Self {
        NDArrayNumericTensor::F64(ArcArray::from_vec(v).into_dyn())
    }
}

impl From<Vec::<half::f16>> for NDArrayNumericTensor {
    fn from(v: Vec::<half::f16>) -> Self {
        NDArrayNumericTensor::F16(ArcArray::from_vec(v).into_dyn())
    }
}

impl From<Vec::<half::bf16>> for NDArrayNumericTensor {
    fn from(v: Vec::<half::bf16>) -> Self {
        NDArrayNumericTensor::BF16(ArcArray::from_vec(v).into_dyn())
    }
}

impl From<Vec::<u32>> for NDArrayNumericTensor {
    fn from(v: Vec::<u32>) -> Self {
        NDArrayNumericTensor::U32(ArcArray::from_vec(v).into_dyn())
    }
}

impl From<Vec::<i32>> for NDArrayNumericTensor {
    fn from(v: Vec::<i32>) -> Self {
        NDArrayNumericTensor::I32(ArcArray::from_vec(v).into_dyn())
    }
}

impl From<Vec::<u64>> for NDArrayNumericTensor {
    fn from(v: Vec::<u64>) -> Self {
        NDArrayNumericTensor::U64(ArcArray::from_vec(v).into_dyn())
    }
}

impl From<Vec::<i64>> for NDArrayNumericTensor {
    fn from(v: Vec::<i64>) -> Self {
        NDArrayNumericTensor::I64(ArcArray::from_vec(v).into_dyn())
    }
}

impl From<Vec::<u16>> for NDArrayNumericTensor {
    fn from(v: Vec::<u16>) -> Self {
        NDArrayNumericTensor::U16(ArcArray::from_vec(v).into_dyn())
    }
}

impl From<Vec::<u8>> for NDArrayNumericTensor {
    fn from(v: Vec::<u8>) -> Self {
        NDArrayNumericTensor::U8(ArcArray::from_vec(v).into_dyn())
    }
}

impl core::fmt::Display for NDArrayNumericTensor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self)
    }
}


impl TryToSlice<u32> for NDArrayNumericTensor {
    fn try_to_slice(&self) -> Result<&[u32], NDArrayNumericTensorError> {
        assert_eq!(self.shape().len(), 1);
        if let NDArrayNumericTensor::U32(x) = &self {
            Ok(x.as_slice_memory_order().unwrap())
        }
        else {
            Err(NDArrayNumericTensorError::WrongDTypeError(DType::U32, self.dtype()))
        }
    }
}

impl TryToSlice<i64> for NDArrayNumericTensor {
    fn try_to_slice(&self) -> Result<&[i64], NDArrayNumericTensorError> {
        assert_eq!(self.shape().len(), 1);
        if let NDArrayNumericTensor::I64(x) = &self {
            Ok(x.as_slice_memory_order().unwrap())
        }
        else {
            Err(NDArrayNumericTensorError::WrongDTypeError(DType::I64, self.dtype()))
        }
    }
}

impl TryToSlice<f32> for NDArrayNumericTensor {
    fn try_to_slice(&self) -> Result<&[f32], NDArrayNumericTensorError> {
        assert_eq!(self.shape().len(), 1);
        if let NDArrayNumericTensor::F32(x) = &self {
            Ok(x.as_slice_memory_order().unwrap())
        }
        else {
            Err(NDArrayNumericTensorError::WrongDTypeError(DType::F32, self.dtype()))
        }
    }
}

impl TryToFlatVec<f64> for NDArrayNumericTensor {
    fn try_to_flat_vec(&self) -> Result<Vec<f64>, NDArrayNumericTensorError> {
        if let NDArrayNumericTensor::F64(x) = &self {
            Ok(x.to_owned().as_slice().unwrap().to_vec())
        }
        else {
            Err(NDArrayNumericTensorError::WrongDTypeError(DType::F64, self.dtype()))
        }
    }
}