use crate::dtype::{DType};
use crate::numeric_scalar::{NumericScalarType};
use crate::numeric_tensor::NumericTensor;
use crate::symbolic_graph::scalar_info::{ScalarInfo, ScalarInfoTyped};
use crate::symbolic_graph::symbolic_scalar::{SymbolicScalarTyped};

#[derive(Clone, Debug)]
pub(crate) struct ShapedTensorTyped<T>
where
    T: Clone + PartialEq + Copy + NumericScalarType,
{
    shape: Vec<usize>,
    values: Vec<ScalarInfoTyped<T>>
}

impl<T> ShapedTensorTyped<T>
where
    T: Clone + PartialEq + Copy + NumericScalarType, 
{
    pub(crate) fn new(shape: Vec<usize>, values: Vec<ScalarInfoTyped<T>>) -> Self {
        Self { shape, values }
    }
    
    pub(crate) fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub(crate) fn rank(&self) -> u32 {
        self.shape.len() as u32
    }
    pub(crate) fn first_element(&self) -> &ScalarInfoTyped<T> {
        &self.values[0]
    }
    pub(crate) fn get(&self, index: usize) -> &ScalarInfoTyped<T> {
        &self.values[index]
    }
    pub(crate) fn reshape(&self, new_shape: Vec<usize>) -> Self {
        Self {
            shape: new_shape,
            values: self.values.clone(),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum ShapedTensor {
    F64(ShapedTensorTyped<f64>),
    F32(ShapedTensorTyped<f32>),
    I64(ShapedTensorTyped<i64>),
}

impl ShapedTensor {
    pub(crate) fn dtype(&self) -> DType {
        match self {
            ShapedTensor::F64(_) => DType::F64,
            ShapedTensor::F32(_) => DType::F32,
            ShapedTensor::I64(_) => DType::I64,
        }
    }
    pub(crate) fn shape(&self) -> &[usize] {
        match self {
            ShapedTensor::F64(t) => t.shape(),
            ShapedTensor::F32(t) => t.shape(),
            ShapedTensor::I64(t) => t.shape(),
        }
    }
    pub(crate) fn rank(&self) -> u32 {
        match self {
            ShapedTensor::F64(x) => x.rank(),
            ShapedTensor::F32(x) => x.rank(),
            ShapedTensor::I64(x) => x.rank(),
        }
    }
    pub(crate) fn first_element(&self) -> ScalarInfo {
        match self {
            ShapedTensor::F64(x) => x.first_element().promote(),
            ShapedTensor::F32(x) => x.first_element().promote(),
            ShapedTensor::I64(x) => x.first_element().promote(),
        }
    }
    pub(crate) fn get(&self, idx: usize) -> ScalarInfo {
        match self {
            ShapedTensor::F64(x) => x.get(idx).promote(),
            ShapedTensor::F32(x) => x.get(idx).promote(),
            ShapedTensor::I64(x) => x.get(idx).promote(),
        }
    }
    pub(crate) fn reshape(&self, new_shape: Vec<usize>) -> Self {
        match self {
            ShapedTensor::F64(x) => Self::F64(x.reshape(new_shape)),
            ShapedTensor::F32(x) => Self::F32(x.reshape(new_shape)),
            ShapedTensor::I64(x) => Self::I64(x.reshape(new_shape))
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RankedTensor {
    first_element: ScalarInfo,
    shape: Vec<ScalarInfoTyped<u64>>
}

impl RankedTensor {
    pub(crate) fn new(first_element: ScalarInfo, shape: Vec<ScalarInfoTyped<u64>>) -> Self {
        Self {
            first_element,
            shape
        }
    }
    
    pub(crate) fn shape(&self) -> &[ScalarInfoTyped<u64>] {
        &self.shape
    }
    
    pub(crate) fn dtype(&self) -> DType {
        self.first_element.dtype()
    }
    
    pub(crate) fn first_element(&self) -> &ScalarInfo {
        &self.first_element
    }
}

#[derive(Clone, Debug)]
pub(crate) struct MinimalTensor {
    first_element: ScalarInfo,
    rank: SymbolicScalarTyped<u32>
}

impl MinimalTensor {
    pub(crate) fn new(first_element: ScalarInfo, rank: SymbolicScalarTyped<u32>) -> Self {
        Self {
            first_element,
            rank
        }
    }
    
    pub(crate) fn dtype(&self) -> DType {
        self.first_element.dtype()
    }
    pub(crate) fn first_element(&self) -> &ScalarInfo {
        &self.first_element
    }
    pub(crate) fn rank(&self) -> &SymbolicScalarTyped<u32> {
        &self.rank
    }
}

#[derive(Clone, Debug)]
pub(crate) enum TensorInfo {
    Numeric(NumericTensor),
    Shaped(ShapedTensor),
    Ranked(RankedTensor),
    Minimal(MinimalTensor)
}

impl TensorInfo {
    pub(crate) fn dtype(&self) -> DType {
        match self {
            Self::Numeric(tensor) => tensor.dtype().unwrap(),
            Self::Shaped(tensor) => tensor.dtype(),
            Self::Ranked(x) => x.dtype(),
            Self::Minimal(x) => x.dtype()
        }
    }
    pub(crate) fn shape(&self) -> Option<Vec<ScalarInfoTyped<u64>>> {
        match self {
            Self::Numeric(x) => Some(x.shape().iter().map(|x| ScalarInfoTyped::Numeric(*x as u64)).collect()),
            Self::Shaped(tensor) => Some(tensor.shape().iter().map(|x| ScalarInfoTyped::Numeric(*x as u64)).collect()),
            Self::Ranked(x) => Some(x.shape().to_vec()),
            Self::Minimal(_) => None
        }
    }
    pub(crate) fn rank(&self) -> ScalarInfoTyped<u32> {
        match self {
            Self::Numeric(x) => ScalarInfoTyped::Numeric(x.rank() as u32),
            Self::Shaped(tensor) => ScalarInfoTyped::Numeric(tensor.rank()),
            Self::Ranked(x) => ScalarInfoTyped::Numeric(x.shape.len() as u32),
            Self::Minimal(x) => ScalarInfoTyped::Symbolic(x.rank.clone())
        }
    }
    pub(crate) fn first_element(&self) -> ScalarInfo {
        match self {
            TensorInfo::Numeric(tensor) => ScalarInfo::Numeric(tensor.to_scalar().unwrap()),
            TensorInfo::Shaped(tensor) => tensor.first_element(),
            TensorInfo::Ranked(tensor) => tensor.first_element().clone(),
            TensorInfo::Minimal(tensor) => tensor.first_element().clone()
        }
    }
    
    pub(crate) fn numeric(&self) -> Option<NumericTensor> {
        match self {
            Self::Numeric(x) => Some(x.clone()),
            _ => None
        }
    }
}