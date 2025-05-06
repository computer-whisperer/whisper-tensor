use crate::dtype::{DType};
use crate::ndarray_backend::conversions::NDArrayNumericTensorType;
use crate::numeric_scalar::{NumericScalarType};
use crate::numeric_tensor::NumericTensor;
use crate::symbolic_graph::scalar_info::{ScalarInfo, ScalarInfoTyped};
use crate::symbolic_graph::symbolic_scalar::{SymbolicResolver, SymbolicScalar, SymbolicScalarTyped};

pub(crate) enum TensorInfoTyped<T> {
    Numeric(NumericTensorTyped),
    Shaped(ShapedTensorTyped),
    Ranked(RankedTensorTyped),
    Minimal(MinimalTensorTyped)
}

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
    T: Clone + PartialEq + Copy + NumericScalarType + NDArrayNumericTensorType, 
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

    pub(crate) fn try_upgrade_to_numeric_tensor(&self) -> Option<NumericTensor> {
        let mut are_all_known = true;
        let mut entries = vec![];
        for scalar in &self.values {
            if let ScalarInfoTyped::Numeric(value) = scalar {
                entries.push(*value);
            } else {
                are_all_known = false;
                break;
            }
        }
        if are_all_known {
            Some(NumericTensor::from_vec_shape(entries, self.shape.clone()).unwrap())
        } else {
            None
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
    pub(crate) fn get(&self, index: &[usize]) -> ScalarInfo {
        assert_eq!(index.len(), 1);
        match self {
            ShapedTensor::F64(x) => x.get(index[0]).promote(),
            ShapedTensor::F32(x) => x.get(index[0]).promote(),
            ShapedTensor::I64(x) => x.get(index[0]).promote(),
        }
    }
    pub(crate) fn reshape(&self, new_shape: Vec<usize>) -> Self {
        match self {
            ShapedTensor::F64(x) => Self::F64(x.reshape(new_shape)),
            ShapedTensor::F32(x) => Self::F32(x.reshape(new_shape)),
            ShapedTensor::I64(x) => Self::I64(x.reshape(new_shape))
        }
    }
    pub(crate) fn try_upgrade_to_numeric_tensor(&self) -> Option<NumericTensor> {
        match self {
            ShapedTensor::F64(x) => x.try_upgrade_to_numeric_tensor(),
            ShapedTensor::F32(x) => x.try_upgrade_to_numeric_tensor(),
            ShapedTensor::I64(x) => x.try_upgrade_to_numeric_tensor()
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
    // Resolves and applies any viable promotions of it's internal types
    pub(crate) fn refine(&self) -> Self {
        match self {
            TensorInfo::Shaped(shaped_tensor) => {
                if let Some(x) = shaped_tensor.try_upgrade_to_numeric_tensor() {
                    TensorInfo::Numeric(x)
                } else {
                    self.clone()
                }
            }
            _ => self.clone()
        }
    }
    
    pub(crate) fn new_from_rank_and_first_value(first_value: ScalarInfo, rank: ScalarInfoTyped<u32>, symbolic_resolver: &mut SymbolicResolver) -> Self {
        match rank {
            ScalarInfoTyped::Numeric(x) => {
                let mut shape = vec![];
                for _ in 0..x {
                    shape.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver))); 
                }
                TensorInfo::Ranked(RankedTensor::new(first_value, shape))
            }
            ScalarInfoTyped::Symbolic(x) => {
                TensorInfo::Minimal(MinimalTensor::new(
                    first_value,
                    x
                ))
            }
        }
    }
    
    pub(crate) fn dtype(&self) -> DType {
        match self {
            Self::Numeric(tensor) => tensor.dtype().unwrap(),
            Self::Shaped(tensor) => tensor.dtype(),
            Self::Ranked(x) => x.dtype(),
            Self::Minimal(x) => x.dtype()
        }
    }
    pub(crate) fn shape(&self) -> TensorInfo::<u64> {
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
    
    pub(crate) fn get(&self, index: &[u64], symbolic_resolver: &mut SymbolicResolver) -> ScalarInfo {
        let index_usize = index.iter().map(|x| *x as usize).collect::<Vec<usize>>();
        if index.iter().all(|x| *x == 0) {
            return self.first_element()
        }
        match self {
            TensorInfo::Numeric(tensor) => ScalarInfo::Numeric(tensor.get(&index_usize).unwrap()),
            TensorInfo::Shaped(tensor) => tensor.get(&index_usize).clone(),
            TensorInfo::Ranked(_) => ScalarInfo::Symbolic(SymbolicScalar::new(self.dtype(), symbolic_resolver)),
            TensorInfo::Minimal(_) => ScalarInfo::Symbolic(SymbolicScalar::new(self.dtype(), symbolic_resolver))
        }
    }
}