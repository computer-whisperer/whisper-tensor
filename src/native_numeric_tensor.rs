use std::ops::Range;
use serde::{Deserialize, Serialize};
use crate::dtype::{DType, DTypeError};
use crate::numeric_tensor::{NumericTensor, NumericTensorError};
use crate::RuntimeError;

#[derive(Debug, thiserror::Error)]
pub enum NativeNumericTensorError {
    #[error("Requested dtype {0}, but had dtype {1}")]
    WrongDTypeError(DType, DType),
    #[error("Cannot reshape tensor from {0:?} to {1:?}")]
    InvalidReshapeError(Vec<usize>, Vec<usize>)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NativeNumericTensorInner {
    F32(Vec<f32>),
    F64(Vec<f64>),
    F16(Vec<half::f16>),
    BF16(Vec<half::bf16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    U16(Vec<u16>),
    U8(Vec<u8>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeNumericTensor {
    pub(crate) value: NativeNumericTensorInner,
    pub(crate) shape: Vec<usize>
}

fn primitive_argmax<T>(
    input: &[T],
    dim: usize,
    shape: &[usize],
) -> (Vec<T>, Vec<u32>, Vec<usize>)
where
    T: PartialOrd + Copy,
{
    // --- sanity checks ---
    let total_elems: usize = shape.iter().product();
    assert_eq!(input.len(), total_elems, "input.len()={} but shape {:?} has {} elements", input.len(), shape, total_elems);
    assert!(dim < shape.len(), "dim {} out of bounds for shape {:?}", dim, shape);

    let ndims = shape.len();
    let axis_size = shape[dim];

    // compute strides for the input tensor
    let mut strides = vec![1; ndims];
    for i in (0..ndims - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    // build output shape (drop the `dim` axis)
    let mut out_shape = shape.to_vec();
    out_shape.remove(dim);
    let out_elems: usize = out_shape.iter().product();

    // precompute strides for the output indexing
    let mut out_strides = vec![1; out_shape.len()];
    for j in (0..out_strides.len() - 1).rev() {
        out_strides[j] = out_strides[j + 1] * out_shape[j + 1];
    }

    let mut out_vals = Vec::with_capacity(out_elems);
    let mut out_idxs = Vec::with_capacity(out_elems);

    // for each "collapsed" position
    for flat_idx in 0..out_elems {
        // reconstruct the base offset into input (skipping the reduction dim)
        let mut base = 0;
        for j in 0..out_strides.len() {
            let idx_j = (flat_idx / out_strides[j]) % out_shape[j];
            let orig_dim = if j < dim { j } else { j + 1 };
            base += idx_j * strides[orig_dim];
        }

        // scan along the reduction axis
        let mut best_val = input[base];
        let mut best_idx = 0;
        for step in 1..axis_size {
            let v = input[base + step * strides[dim]];
            if v > best_val {
                best_val = v;
                best_idx = step;
            }
        }

        out_vals.push(best_val);
        out_idxs.push(best_idx as u32);
    }
    (out_vals, out_idxs, out_shape)
}

impl NativeNumericTensor {
    
    pub fn dtype(&self) -> DType {
        match self.value{
            NativeNumericTensorInner::F32(_) => DType::F32,
            NativeNumericTensorInner::F64(_) => DType::F64,
            NativeNumericTensorInner::F16(_) => DType::F16,
            NativeNumericTensorInner::BF16(_) => DType::BF16,
            NativeNumericTensorInner::U32(_) => DType::U32,
            NativeNumericTensorInner::I32(_) => DType::I32,
            NativeNumericTensorInner::U64(_) => DType::U64,
            NativeNumericTensorInner::I64(_) => DType::I64,
            NativeNumericTensorInner::U16(_) => DType::U16,
            NativeNumericTensorInner::U8(_) => DType::U8,
        }
    }
    
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn from_raw_data(data: &[u8], dtype: DType, shape: Vec<usize>) -> Self {
        let value = match dtype {
            DType::F64 => {
                let data = data.chunks_exact(8).map(|x| f64::from_le_bytes(x.try_into().unwrap())).collect();
                NativeNumericTensorInner::F64(data)
            },
            DType::F32 => {
                let data = data.chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect();
                NativeNumericTensorInner::F32(data)
            },
            DType::BF16 => {
                let data = data.chunks_exact(2).map(|x| half::bf16::from_bits(u16::from_le_bytes(x.try_into().unwrap()))).collect();
                NativeNumericTensorInner::BF16(data)
            }
            DType::F16 => {
                let data = data.chunks_exact(2).map(|x| half::f16::from_bits(u16::from_le_bytes(x.try_into().unwrap()))).collect();
                NativeNumericTensorInner::F16(data)
            }
            DType::I64 => {
                let data = data.chunks_exact(8).map(|x| i64::from_le_bytes(x.try_into().unwrap())).collect();
                NativeNumericTensorInner::I64(data)
            },
            DType::I32 => {
                let data = data.chunks_exact(4).map(|x| i32::from_le_bytes(x.try_into().unwrap())).collect();
                NativeNumericTensorInner::I32(data)
            },
            DType::U64 => {
                let data = data.chunks_exact(8).map(|x| u64::from_le_bytes(x.try_into().unwrap())).collect();
                NativeNumericTensorInner::U64(data)
            },
            DType::U32 => {
                let data = data.chunks_exact(4).map(|x| u32::from_le_bytes(x.try_into().unwrap())).collect();
                NativeNumericTensorInner::U32(data)
            },
            DType::U16 => {
                let data = data.chunks_exact(2).map(|x| u16::from_le_bytes(x.try_into().unwrap())).collect();
                NativeNumericTensorInner::U16(data)
            }
            DType::U8 => {
                NativeNumericTensorInner::U8(data.to_vec())
            }
        };
        
        NativeNumericTensor {
            value,
            shape: shape.into(),
        }
    }

    pub fn from_vec<T>(v: Vec<T>, shape: Vec<usize>) -> Self
    where NativeNumericTensorInner: From<Vec<T>>
    {
        assert_eq!(shape.iter().product::<usize>(), v.len());
        Self{
            value: NativeNumericTensorInner::from(v),
            shape: shape.into(),
        }
    }
    
    fn get_index(&self, index: &[usize]) -> usize {
        assert_eq!(index.len(), self.shape.len());
        let mut idx = 0;
        let mut stride = 1;
        for (i, &dim) in self.shape.iter().enumerate().rev() {
            idx += index[i] * stride;
            stride *= dim;
        }
        idx
    }
    
    pub fn argmax(&self, axis: usize) -> (Self, Self) {
        match &self.value {
            NativeNumericTensorInner::F32(x) => {
                let (x, i,  s) = primitive_argmax(x, axis, &self.shape);
                (NativeNumericTensor::from_vec(x, s.clone()), NativeNumericTensor::from_vec(i, s))
            }
            NativeNumericTensorInner::F64(x) => {
                let (x, i,  s) = primitive_argmax(x, axis, &self.shape);
                (NativeNumericTensor::from_vec(x, s.clone()), NativeNumericTensor::from_vec(i, s))
            }
            NativeNumericTensorInner::F16(x) => {
                let (x, i,  s) = primitive_argmax(x, axis, &self.shape);
                (NativeNumericTensor::from_vec(x, s.clone()), NativeNumericTensor::from_vec(i, s))
            }
            NativeNumericTensorInner::BF16(x) => {
                let (x, i,  s) = primitive_argmax(x, axis, &self.shape);
                (NativeNumericTensor::from_vec(x, s.clone()), NativeNumericTensor::from_vec(i, s))
            }
            NativeNumericTensorInner::I32(x) => {
                let (x, i,  s) = primitive_argmax(x, axis, &self.shape);
                (NativeNumericTensor::from_vec(x, s.clone()), NativeNumericTensor::from_vec(i, s))
            }
            NativeNumericTensorInner::I64(x) => {
                let (x, i,  s) = primitive_argmax(x, axis, &self.shape);
                (NativeNumericTensor::from_vec(x, s.clone()), NativeNumericTensor::from_vec(i, s))
            }
            NativeNumericTensorInner::U32(x) => {
                let (x, i,  s) = primitive_argmax(x, axis, &self.shape);
                (NativeNumericTensor::from_vec(x, s.clone()), NativeNumericTensor::from_vec(i, s))
            }
            NativeNumericTensorInner::U64(x) => {
                let (x, i,  s) = primitive_argmax(x, axis, &self.shape);
                (NativeNumericTensor::from_vec(x, s.clone()), NativeNumericTensor::from_vec(i, s))
            }
            NativeNumericTensorInner::U16(x) => {
                let (x, i,  s) = primitive_argmax(x, axis, &self.shape);
                (NativeNumericTensor::from_vec(x, s.clone()), NativeNumericTensor::from_vec(i, s))
            }
            NativeNumericTensorInner::U8(x) => {
                let (x, i,  s) = primitive_argmax(x, axis, &self.shape);
                (NativeNumericTensor::from_vec(x, s.clone()), NativeNumericTensor::from_vec(i, s))
            }
        }
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, NativeNumericTensorError> {
        let orig_num_elements: usize = self.shape.iter().product();
        let new_num_elements: usize = new_shape.iter().product();
        if orig_num_elements == new_num_elements {
            Ok(Self {
                value: self.value.clone(),
                shape: new_shape
            })
        }
        else {
            Err(NativeNumericTensorError::InvalidReshapeError(self.shape.clone(), new_shape))
        }
    }

    pub fn unsqueeze(&self, p0: usize) -> Result<Self, NativeNumericTensorError> {
        let mut s = self.shape.clone();
        s.insert(p0, 1);
        self.reshape(s)
    }

    pub fn squeeze(&self, p0: usize) -> Result<Self, NativeNumericTensorError> {
        let mut s = self.shape.clone();
        s.remove(p0);
        self.reshape(s)
    }
    
    pub fn slice(&self, indices: &[Range<usize>]) -> Result<Self, NativeNumericTensorError> {
        
    }
}

#[cfg(feature = "candle")]
impl TryFrom<NativeNumericTensor> for candle_core::Tensor {
    type Error = RuntimeError;
    fn try_from(value: NativeNumericTensor) -> Result<Self, Self::Error> {
        Ok(match &value.value {
            NativeNumericTensorInner::F32(x) => {
                candle_core::Tensor::from_vec(x.clone(), value.shape.clone(), &candle_core::Device::Cpu)?
            }
            NativeNumericTensorInner::F64(x) => {
                candle_core::Tensor::from_vec(x.clone(), value.shape.clone(), &candle_core::Device::Cpu)?
            }
            NativeNumericTensorInner::F16(x) => {
                candle_core::Tensor::from_vec(x.clone(), value.shape.clone(), &candle_core::Device::Cpu)?
            }
            NativeNumericTensorInner::BF16(x) => {
                candle_core::Tensor::from_vec(x.clone(), value.shape.clone(), &candle_core::Device::Cpu)?
            }
            NativeNumericTensorInner::U32(x) => {
                candle_core::Tensor::from_vec(x.clone(), value.shape.clone(), &candle_core::Device::Cpu)?
            }
            NativeNumericTensorInner::I64(x) => {
                candle_core::Tensor::from_vec(x.clone(), value.shape.clone(), &candle_core::Device::Cpu)?
            }
            _ => {
                Err(RuntimeError::BackendUnsupportedTensorDType)?
            }
        })
    }
}

#[cfg(feature = "candle")]
impl TryFrom<&candle_core::Tensor> for NativeNumericTensor {
    type Error = candle_core::Error;
    fn try_from(value: &candle_core::Tensor) -> Result<Self, Self::Error> {
        let shape = value.shape().dims();
        let tensor_flat = value.flatten_all()?;
        let inner_value = match value.dtype() {
            candle_core::DType::F64 => {
                let v = tensor_flat.to_vec1::<f64>()?;
                NativeNumericTensorInner::F64(v)
            },
            candle_core::DType::U8 => {
                let v = tensor_flat.to_vec1::<u8>()?;
                NativeNumericTensorInner::U8(v)
            }
            candle_core::DType::U32 => {
                let v = tensor_flat.to_vec1::<u32>()?;
                NativeNumericTensorInner::U32(v)
            }
            candle_core::DType::I64 => {
                let v = tensor_flat.to_vec1::<i64>()?;
                NativeNumericTensorInner::I64(v)
            }
            candle_core::DType::BF16 => {
                let v = tensor_flat.to_vec1::<half::bf16>()?;
                NativeNumericTensorInner::BF16(v)
            }
            candle_core::DType::F16 => {
                let v = tensor_flat.to_vec1::<half::f16>()?;
                NativeNumericTensorInner::F16(v)
            }
            candle_core::DType::F32 => {
                let v = tensor_flat.to_vec1::<f32>()?;
                NativeNumericTensorInner::F32(v)
            }
        };
        Ok(NativeNumericTensor {
            shape: shape.to_vec(),
            value: inner_value,
        })
    }
}

impl TryFrom<candle_core::Tensor> for NativeNumericTensor {
    type Error = candle_core::Error;
    fn try_from(value: candle_core::Tensor) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl From<Vec::<f32>> for NativeNumericTensorInner {
    fn from(v: Vec::<f32>) -> Self {
        NativeNumericTensorInner::F32(v)
    }
}

impl From<Vec::<f64>> for NativeNumericTensorInner {
    fn from(v: Vec::<f64>) -> Self {
        NativeNumericTensorInner::F64(v)
    }
}

impl From<Vec::<half::f16>> for NativeNumericTensorInner {
    fn from(v: Vec::<half::f16>) -> Self {
        NativeNumericTensorInner::F16(v)
    }
}

impl From<Vec::<half::bf16>> for NativeNumericTensorInner {
    fn from(v: Vec::<half::bf16>) -> Self {
        NativeNumericTensorInner::BF16(v)
    }
}

impl From<Vec::<u32>> for NativeNumericTensorInner {
    fn from(v: Vec::<u32>) -> Self {
        NativeNumericTensorInner::U32(v)
    }
}

impl From<Vec::<i32>> for NativeNumericTensorInner {
    fn from(v: Vec::<i32>) -> Self {
        NativeNumericTensorInner::I32(v)
    }
}

impl From<Vec::<u64>> for NativeNumericTensorInner {
    fn from(v: Vec::<u64>) -> Self {
        NativeNumericTensorInner::U64(v)
    }
}

impl From<Vec::<i64>> for NativeNumericTensorInner {
    fn from(v: Vec::<i64>) -> Self {
        NativeNumericTensorInner::I64(v)
    }
}

impl From<Vec::<u16>> for NativeNumericTensorInner {
    fn from(v: Vec::<u16>) -> Self {
        NativeNumericTensorInner::U16(v)
    }
}

impl From<Vec::<u8>> for NativeNumericTensorInner {
    fn from(v: Vec::<u8>) -> Self {
        NativeNumericTensorInner::U8(v)
    }
}

impl core::fmt::Display for NativeNumericTensor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub trait TryToSlice<T> {
    fn try_to_slice(&self) -> Result<&[T], NativeNumericTensorError>;
}

impl TryFrom<NativeNumericTensor> for Vec<u32>
{
    type Error = NativeNumericTensorError;
    fn try_from(value: NativeNumericTensor) -> Result<Vec<u32>, NativeNumericTensorError> {
        assert_eq!(value.shape().len(), 1);
        if let NativeNumericTensorInner::U32(x) = value.value {
            Ok(x)
        }
        else {
            Err(NativeNumericTensorError::WrongDTypeError(DType::U32, value.dtype()))
        }
    }
}

impl TryFrom<NativeNumericTensor> for Vec<f32>
{
    type Error = NativeNumericTensorError;
    fn try_from(value: NativeNumericTensor) -> Result<Vec<f32>, NativeNumericTensorError> {
        assert_eq!(value.shape().len(), 1);
        if let NativeNumericTensorInner::F32(x) = value.value {
            Ok(x)
        }
        else {
            Err(NativeNumericTensorError::WrongDTypeError(DType::F32, value.dtype()))
        }
    }
}

impl TryToSlice<u32> for NativeNumericTensor {
    fn try_to_slice(&self) -> Result<&[u32], NativeNumericTensorError> {
        assert_eq!(self.shape().len(), 1);
        if let NativeNumericTensorInner::U32(x) = &self.value {
            Ok(x.as_slice())
        }
        else {
            Err(NativeNumericTensorError::WrongDTypeError(DType::U32, self.dtype()))
        }
    }
}

impl TryToSlice<f32> for NativeNumericTensor {
    fn try_to_slice(&self) -> Result<&[f32], NativeNumericTensorError> {
        assert_eq!(self.shape().len(), 1);
        if let NativeNumericTensorInner::F32(x) = &self.value {
            Ok(x.as_slice())
        }
        else {
            Err(NativeNumericTensorError::WrongDTypeError(DType::F32, self.dtype()))
        }
    }
}

