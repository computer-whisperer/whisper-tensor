use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::dtype::{DType, DTypeError};
use crate::numeric_tensor::{NumericTensor, NumericTensorError};
use crate::tensor_rank::{DynRank, Rank};
use half::{bf16, f16};
use ort::value::DynValue;

#[derive(Debug)]
pub struct ORTNumericTensor(pub(crate) DynValue);

impl From<DType> for ort::tensor::TensorElementType {
    fn from(value: DType) -> Self {
        match value {
            DType::F32 => ort::tensor::TensorElementType::Float32,
            DType::F64 => ort::tensor::TensorElementType::Float64,
            DType::BF16 => ort::tensor::TensorElementType::Bfloat16,
            DType::F16 => ort::tensor::TensorElementType::Float16,
            DType::I64 => ort::tensor::TensorElementType::Int64,
            DType::I32 => ort::tensor::TensorElementType::Int32,
            DType::U64 => ort::tensor::TensorElementType::Uint64,
            DType::U32 => ort::tensor::TensorElementType::Uint32,
            DType::U16 => ort::tensor::TensorElementType::Uint16,
            DType::I16 => ort::tensor::TensorElementType::Int16,
            DType::I8 => ort::tensor::TensorElementType::Int8,
            DType::U8 => ort::tensor::TensorElementType::Uint8,
            DType::BOOL => ort::tensor::TensorElementType::Bool,
            DType::STRING => ort::tensor::TensorElementType::String,
        }
    }
}

impl TryFrom<ort::tensor::TensorElementType> for DType {
    type Error = DTypeError;
    fn try_from(value: ort::tensor::TensorElementType) -> Result<Self, Self::Error> {
        Ok(match value {
            ort::tensor::TensorElementType::Float64 => DType::F64,
            ort::tensor::TensorElementType::Float32 => DType::F32,
            ort::tensor::TensorElementType::Bfloat16 => DType::BF16,
            ort::tensor::TensorElementType::Float16 => DType::F16,
            ort::tensor::TensorElementType::Uint64 => DType::U64,
            ort::tensor::TensorElementType::Int64 => DType::I64,
            ort::tensor::TensorElementType::Uint32 => DType::U32,
            ort::tensor::TensorElementType::Int32 => DType::I32,
            ort::tensor::TensorElementType::Uint16 => DType::U16,
            ort::tensor::TensorElementType::Int16 => DType::I16,
            ort::tensor::TensorElementType::Uint8 => DType::U8,
            ort::tensor::TensorElementType::Int8 => DType::I8,
            _ => Err(DTypeError::UnsupportedORTDtype(value))?,
        })
    }
}

impl ORTNumericTensor {
    pub fn dtype(&self) -> Result<DType, DTypeError> {
        self.0.dtype().tensor_type().unwrap().try_into()
    }

    pub fn rank(&self) -> usize {
        self.0.shape().len()
    }

    pub fn shape(&self) -> Vec<usize> {
        self.0
            .shape()
            .to_vec()
            .iter()
            .map(|x| *x as usize)
            .collect()
    }
}

impl Clone for ORTNumericTensor {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl TryFrom<&NDArrayNumericTensor<DynRank>> for ORTNumericTensor {
    type Error = NumericTensorError;
    fn try_from(value: &NDArrayNumericTensor<DynRank>) -> Result<Self, Self::Error> {
        Ok(ORTNumericTensor(match value {
            NDArrayNumericTensor::F32(x) => ort::value::Value::from_array(x.to_owned())?.into_dyn(),
            NDArrayNumericTensor::F64(x) => ort::value::Value::from_array(x.to_owned())?.into_dyn(),
            NDArrayNumericTensor::U32(x) => ort::value::Value::from_array(x.to_owned())?.into_dyn(),
            NDArrayNumericTensor::I32(x) => ort::value::Value::from_array(x.to_owned())?.into_dyn(),
            NDArrayNumericTensor::U64(x) => ort::value::Value::from_array(x.to_owned())?.into_dyn(),
            NDArrayNumericTensor::I64(x) => ort::value::Value::from_array(x.to_owned())?.into_dyn(),
            NDArrayNumericTensor::U16(x) => ort::value::Value::from_array(x.to_owned())?.into_dyn(),
            NDArrayNumericTensor::I16(x) => ort::value::Value::from_array(x.to_owned())?.into_dyn(),
            NDArrayNumericTensor::U8(x) => ort::value::Value::from_array(x.to_owned())?.into_dyn(),
            NDArrayNumericTensor::I8(x) => ort::value::Value::from_array(x.to_owned())?.into_dyn(),
            NDArrayNumericTensor::F16(x) => ort::value::Value::from_array(x.to_owned())?.into_dyn(),
            NDArrayNumericTensor::BF16(x) => {
                ort::value::Value::from_array(x.to_owned())?.into_dyn()
            }
            NDArrayNumericTensor::BOOL(x) => {
                ort::value::Value::from_array(x.to_owned())?.into_dyn()
            }
            NDArrayNumericTensor::STRING(_) => {
                unimplemented!()
            }
        }))
    }
}

impl TryFrom<NDArrayNumericTensor<DynRank>> for ORTNumericTensor {
    type Error = NumericTensorError;
    fn try_from(value: NDArrayNumericTensor<DynRank>) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl<R: Rank> TryFrom<&ORTNumericTensor> for NDArrayNumericTensor<R> {
    type Error = NumericTensorError;
    fn try_from(value: &ORTNumericTensor) -> Result<Self, Self::Error> {
        let ort_dtype = value.0.dtype().tensor_type().unwrap();
        let dyn_arr = match ort_dtype {
            ort::tensor::TensorElementType::Float32 => NDArrayNumericTensor::<DynRank>::F32(
                value.0.try_extract_array::<f32>()?.to_shared(),
            ),
            ort::tensor::TensorElementType::Float64 => {
                NDArrayNumericTensor::F64(value.0.try_extract_array::<f64>()?.to_shared())
            }
            ort::tensor::TensorElementType::Int32 => {
                NDArrayNumericTensor::I32(value.0.try_extract_array::<i32>()?.to_shared())
            }
            ort::tensor::TensorElementType::Int64 => {
                NDArrayNumericTensor::I64(value.0.try_extract_array::<i64>()?.to_shared())
            }
            ort::tensor::TensorElementType::Uint32 => {
                NDArrayNumericTensor::U32(value.0.try_extract_array::<u32>()?.to_shared())
            }
            ort::tensor::TensorElementType::Uint64 => {
                NDArrayNumericTensor::U64(value.0.try_extract_array::<u64>()?.to_shared())
            }
            ort::tensor::TensorElementType::Int16 => {
                NDArrayNumericTensor::I16(value.0.try_extract_array::<i16>()?.to_shared())
            }
            ort::tensor::TensorElementType::Uint16 => {
                NDArrayNumericTensor::U16(value.0.try_extract_array::<u16>()?.to_shared())
            }
            ort::tensor::TensorElementType::Int8 => {
                NDArrayNumericTensor::I8(value.0.try_extract_array::<i8>()?.to_shared())
            }
            ort::tensor::TensorElementType::Uint8 => {
                NDArrayNumericTensor::U8(value.0.try_extract_array::<u8>()?.to_shared())
            }
            ort::tensor::TensorElementType::Bfloat16 => {
                NDArrayNumericTensor::BF16(value.0.try_extract_array::<bf16>()?.to_shared())
            }
            ort::tensor::TensorElementType::Float16 => {
                NDArrayNumericTensor::F16(value.0.try_extract_array::<f16>()?.to_shared())
            }
            _ => Err(DTypeError::UnsupportedORTDtype(ort_dtype))?,
        };
        Ok(dyn_arr.try_to_rank()?)
    }
}

impl<R: Rank> TryFrom<ORTNumericTensor> for NDArrayNumericTensor<R> {
    type Error = NumericTensorError;
    fn try_from(value: ORTNumericTensor) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl TryFrom<NumericTensor<DynRank>> for ORTNumericTensor {
    type Error = NumericTensorError;
    fn try_from(value: NumericTensor<DynRank>) -> Result<Self, Self::Error> {
        Ok(match value {
            NumericTensor::ORT(x) => x,
            _ => ORTNumericTensor::try_from(NDArrayNumericTensor::try_from(value)?)?,
        })
    }
}

impl<R: Rank> From<ORTNumericTensor> for NumericTensor<R> {
    fn from(value: ORTNumericTensor) -> Self {
        Self::ORT(value)
    }
}
