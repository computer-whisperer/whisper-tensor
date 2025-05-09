use typenum::P1;
use crate::dtype::{DType};
use crate::ndarray_backend::conversions::NDArrayNumericTensorType;
use crate::numeric_scalar::{NumericScalarType};
use crate::numeric_tensor::{NumericTensor, NumericTensorError};
use crate::numeric_tensor_typed::NumericTensorTyped;
use crate::symbolic_graph::scalar_info::{ScalarInfo, ScalarInfoTyped};
use crate::symbolic_graph::symbolic_scalar::{SymbolicResolver, SymbolicScalar, SymbolicScalarTyped};
use crate::tensor_rank::{DimContainer, DynRank, KnownRank, Rank, RankError};

#[derive(Debug, thiserror::Error)]
pub enum TensorInfoError {
    #[error("Cannot cast to rank")]
    CannotConvertToRank,
    #[error("Cannot cast to type")]
    CannotConvertToType,
    #[error(transparent)]
    RankError(#[from] RankError),
    #[error(transparent)]
    NumericTensorError(#[from] NumericTensorError)
}

trait TensorInfoType: Clone + PartialEq + Copy + NumericScalarType + NDArrayNumericTensorType {
    fn typed_shaped_tensor_into_shaped_tensor<R: Rank>(value: ShapedTensorTyped<Self, R>) -> ShapedTensor<R>;
    fn try_shaped_tensor_into_typed_shaped_tensor<R: Rank>(value: &ShapedTensor<R>) -> Result<&ShapedTensorTyped<Self, R>, TensorInfoError>;
}

#[macro_export]
macro_rules! impl_type {
    ($a:ident, $b:ident) => {
        impl TensorInfoType for $a {
            fn typed_shaped_tensor_into_shaped_tensor<R: Rank>(value: ShapedTensorTyped<Self, R>) -> ShapedTensor<R> {
                ShapedTensor::$b(value)
            }
            fn try_shaped_tensor_into_typed_shaped_tensor<R: Rank>(value: &ShapedTensor<R>) -> Result<&ShapedTensorTyped<Self, R>, TensorInfoError> {
                if let ShapedTensor::$b(v) = value {
                    Ok(v)
                } else {
                    Err(TensorInfoError::CannotConvertToType)
                }
            }
        }
    }
}

impl_type!(f64, F64);
impl_type!(f32, F32);
impl_type!(u64, U64);
impl_type!(i64, I64);

#[derive(Clone, Debug)]
pub(crate) struct ShapedTensorTyped<T, R: Rank>
where
    T: Clone + PartialEq + Copy + NumericScalarType,
{
    shape: R::KnownDims,
    values: Vec<ScalarInfoTyped<T>>
}

impl<T, R: Rank> ShapedTensorTyped<T, R>
where
    T: TensorInfoType, 
{
    pub(crate) fn new(shape: R::KnownDims, values: Vec<ScalarInfoTyped<T>>) -> Self {
        Self { shape, values }
    }
    
    pub(crate) fn shape(&self) -> &[u64] {
        self.shape.as_slice()
    }
    pub(crate) fn rank(&self) -> u32 {
        self.shape.len() as u32
    }
    pub(crate) fn first_element(&self) -> &ScalarInfoTyped<T> {
        &self.values[0]
    }
    pub(crate) fn get(&self, index: &R::KnownDims) -> Option<&ScalarInfoTyped<T>> {
        assert_eq!(index.len(), 1);
        self.values.get(index[0] as usize)
    }
    pub(crate) fn reshape(&self, new_shape: R::KnownDims) -> Self {
        Self {
            shape: new_shape,
            values: self.values.clone(),
        }
    }

    pub(crate) fn try_to_rank<R1: Rank>(&self) -> Result<ShapedTensorTyped<T, R1>, TensorInfoError> {
        let slice = self.shape.as_slice();
        let new_shape = R1::KnownDims::try_from_slice(slice)?;
        Ok(ShapedTensorTyped{
            shape: new_shape,
            values: self.values.clone()
        })
    }
    
    pub(crate) fn to_dyn_rank(&self) -> ShapedTensorTyped<T, DynRank> {
        self.try_to_rank().unwrap()
    }
    
    pub(crate) fn to_dyn_type(&self) -> ShapedTensor<R> {
        T::typed_shaped_tensor_into_shaped_tensor(self.clone())
    }

    pub(crate) fn try_upgrade_to_numeric_tensor(&self) -> Option<NumericTensor<R>> {
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
            Some(NumericTensor::from_vec_shape(entries, self.shape.as_slice().into_iter().map(|x| *x as usize).collect()).unwrap())
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum ShapedTensor<R: Rank> {
    F64(ShapedTensorTyped<f64, R>),
    F32(ShapedTensorTyped<f32, R>),
    U64(ShapedTensorTyped<u64, R>),
    I64(ShapedTensorTyped<i64, R>),
}

impl<R: Rank> ShapedTensor<R> {
    pub(crate) fn dtype(&self) -> DType {
        match self {
            ShapedTensor::F64(_) => DType::F64,
            ShapedTensor::F32(_) => DType::F32,
            ShapedTensor::I64(_) => DType::I64,
            ShapedTensor::U64(_) => DType::U64,
        }
    }
    pub(crate) fn shape(&self) -> &[u64] {
        match self {
            ShapedTensor::F64(t) => t.shape(),
            ShapedTensor::F32(t) => t.shape(),
            ShapedTensor::I64(t) => t.shape(),
            ShapedTensor::U64(t) => t.shape(),
        }
    }
    pub(crate) fn rank(&self) -> u32 {
        match self {
            ShapedTensor::F64(x) => x.rank(),
            ShapedTensor::F32(x) => x.rank(),
            ShapedTensor::I64(x) => x.rank(),
            ShapedTensor::U64(x) => x.rank(),
        }
    }
    pub(crate) fn first_element(&self) -> ScalarInfo {
        match self {
            ShapedTensor::F64(x) => x.first_element().promote(),
            ShapedTensor::F32(x) => x.first_element().promote(),
            ShapedTensor::I64(x) => x.first_element().promote(),
            ShapedTensor::U64(x) => x.first_element().promote(),
        }
    }
    pub(crate) fn get(&self, index: &R::KnownDims) -> Option<ScalarInfo> {
        Some(match self {
            ShapedTensor::F64(x) => x.get(index)?.promote(),
            ShapedTensor::F32(x) => x.get(index)?.promote(),
            ShapedTensor::I64(x) => x.get(index)?.promote(),
            ShapedTensor::U64(x) => x.get(index)?.promote(),
        })
    }
    pub(crate) fn reshape(&self, new_shape: R::KnownDims) -> Self {
        match self {
            ShapedTensor::F64(x) => Self::F64(x.reshape(new_shape)),
            ShapedTensor::F32(x) => Self::F32(x.reshape(new_shape)),
            ShapedTensor::I64(x) => Self::I64(x.reshape(new_shape)),
            ShapedTensor::U64(x) => Self::U64(x.reshape(new_shape)),
        }
    }
    pub(crate) fn try_upgrade_to_numeric_tensor(&self) -> Option<NumericTensor<R>> {
        match self {
            ShapedTensor::F64(x) => x.try_upgrade_to_numeric_tensor(),
            ShapedTensor::F32(x) => x.try_upgrade_to_numeric_tensor(),
            ShapedTensor::I64(x) => x.try_upgrade_to_numeric_tensor(),
            ShapedTensor::U64(x) => x.try_upgrade_to_numeric_tensor(),
        }
    }

    pub(crate) fn try_to_rank<R1: Rank>(&self) -> Result<ShapedTensor<R1>, TensorInfoError> {
        Ok(match self {
            ShapedTensor::F64(x) => ShapedTensor::<R1>::F64(x.try_to_rank()?),
            ShapedTensor::F32(x) => ShapedTensor::<R1>::F32(x.try_to_rank()?),
            ShapedTensor::I64(x) => ShapedTensor::<R1>::I64(x.try_to_rank()?),
            ShapedTensor::U64(x) => ShapedTensor::<R1>::U64(x.try_to_rank()?),
        })
    }
    
    pub(crate) fn try_to_type<T: TensorInfoType>(&self) -> Result<ShapedTensorTyped<T, R>, TensorInfoError> {
        Ok(T::try_shaped_tensor_into_typed_shaped_tensor(self)?.clone())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RankedTensorTyped<T, R: Rank>
where
    T: Clone + Copy + PartialEq + NumericScalarType,
{
    first_element: ScalarInfoTyped<T>,
    shape: R::UnknownDims
}

impl<T: Clone + Copy + PartialEq + NumericScalarType, R: Rank> RankedTensorTyped<T, R> {
    pub(crate) fn new(first_element: ScalarInfoTyped<T>, shape: R::UnknownDims) -> Self {
        RankedTensorTyped { 
            first_element,
            shape
        }
    }

    pub(crate) fn try_to_rank<R1: Rank>(&self) -> Result<RankedTensorTyped<T, R1>, TensorInfoError> {
        let slice = self.shape.as_slice();
        let new_shape = R1::UnknownDims::try_from_slice(slice)?;
        Ok(RankedTensorTyped {
            first_element: self.first_element.clone(), 
            shape: new_shape
        })
    }
    
    pub(crate) fn to_dyn_rank(&self) -> RankedTensorTyped<T, DynRank> {
        self.try_to_rank().unwrap()
    }
    
    pub(crate) fn to_dyn_type(&self) -> RankedTensor<R> {
        RankedTensor {
            first_element: self.first_element.to_dyn_type(),
            shape: self.shape.clone()
        }
    }
    
    pub(crate) fn shape(&self) -> &R::UnknownDims {
        &self.shape
    }
    
    pub(crate) fn first_element(&self) -> &ScalarInfoTyped<T> {
        &self.first_element
    }
}


#[derive(Clone, Debug)]
pub(crate) struct RankedTensor<R: Rank> {
    first_element: ScalarInfo,
    shape: R::UnknownDims
}

impl<R: Rank> RankedTensor<R> {
    pub(crate) fn new(first_element: ScalarInfo, shape: R::UnknownDims) -> Self {
        Self {
            first_element,
            shape
        }
    }
    
    pub(crate) fn shape(&self) -> &R::UnknownDims {
        &self.shape
    }
    
    pub(crate) fn dtype(&self) -> DType {
        self.first_element.dtype()
    }
    
    pub(crate) fn first_element(&self) -> &ScalarInfo {
        &self.first_element
    }
    
    pub(crate) fn try_to_rank<R1:Rank>(&self) -> Result<RankedTensor<R1>, TensorInfoError> {
        Ok(RankedTensor::new(self.first_element.clone(), R1::UnknownDims::try_from_slice(self.shape.as_slice())?))
    }
    
    pub(crate) fn try_to_type<T:TensorInfoType>(&self) -> Result<RankedTensorTyped<T, R>, TensorInfoError> {
        Ok(RankedTensorTyped::new(self.first_element.cast(), self.shape.clone()))
    }
}

#[derive(Clone, Debug)]
pub(crate) struct MinimalTensorTyped<T>
where
    T: TensorInfoType,
{
    first_element: ScalarInfoTyped<T>,
    rank: SymbolicScalarTyped<u32>
}

impl <T> MinimalTensorTyped<T>
where
    T: TensorInfoType,
{
    pub(crate) fn to_dyn_type(&self) -> MinimalTensor {
        MinimalTensor {
            first_element: self.first_element.to_dyn_type(),
            rank: self.rank.clone()
        }
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
    pub(crate) fn shape(&self, symbolic_resolver: &mut SymbolicResolver) -> RankedTensorTyped<u64, P1> {
        RankedTensorTyped::new(
            ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)),
            [ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver))]
        )
    }
}

#[derive(Clone, Debug)]
pub(crate) enum TensorInfoTypedShaped<T, R: KnownRank>
where
    T: TensorInfoType,
{
    Numeric(NumericTensorTyped<T, R>),
    Shaped(ShapedTensorTyped<T, R>),
}

#[derive(Clone, Debug)]
pub(crate) enum TensorInfoTypedRanked<T, R: KnownRank>
where
    T: TensorInfoType,
{
    Numeric(NumericTensorTyped<T, R>),
    Shaped(ShapedTensorTyped<T, R>),
    Ranked(RankedTensorTyped<T, R>),
}

impl<T, R: KnownRank> TensorInfoTypedRanked<T, R>
where
    T: TensorInfoType,
{
    pub(crate) fn new_from_scalar_infos(scalar_infos: Vec<ScalarInfoTyped<T>>, shape: R::KnownDims) -> Self {
        Self::Shaped(ShapedTensorTyped{
            shape,
            values: scalar_infos
        })
    }
    
    pub(crate) fn shape(&self) -> TensorInfoTypedRanked<u64, P1> {
        match self {
            TensorInfoTypedRanked::Numeric(numeric) => {
                let shape = numeric.shape();
                let slice = shape.as_slice();
                TensorInfoTypedRanked::Numeric(NumericTensorTyped::from_vec(slice.to_vec()))
            },
            TensorInfoTypedRanked::Shaped(shaped) => {
                let slice = shaped.shape();
                TensorInfoTypedRanked::Numeric(NumericTensorTyped::from_vec(slice.to_vec()))
            },
            TensorInfoTypedRanked::Ranked(ranked) => {
                let slice = ranked.shape().as_slice();
                TensorInfoTypedRanked::Shaped(ShapedTensorTyped::<u64, P1>::new([slice.len() as u64], slice.to_vec()))
            }
        }
    }
    
    pub(crate) fn to_dyn_rank(&self) -> TensorInfoTyped<T> {
        match self {
            TensorInfoTypedRanked::Numeric(numeric) => TensorInfoTyped::Numeric(numeric.to_dyn_rank()),
            TensorInfoTypedRanked::Shaped(shaped) => TensorInfoTyped::Shaped(shaped.to_dyn_rank()),
            TensorInfoTypedRanked::Ranked(ranked) => TensorInfoTyped::Ranked(ranked.to_dyn_rank())
        }
    }
    
    pub(crate) fn get(&self, index: &R::KnownDims, symbolic_resolver: &mut SymbolicResolver) -> Option<ScalarInfoTyped<T>> {
        match self {
            TensorInfoTypedRanked::Numeric(numeric) => Some(ScalarInfoTyped::Numeric(*numeric.get(index)?)),
            TensorInfoTypedRanked::Shaped(shaped) => shaped.get(index).map(|x| x.clone()),
            TensorInfoTypedRanked::Ranked(_) => Some(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)))
        }
    }
}

pub(crate) enum TensorInfoTyped<T>
where
    T: TensorInfoType,
{
    Numeric(NumericTensorTyped<T, DynRank>),
    Shaped(ShapedTensorTyped<T, DynRank>),
    Ranked(RankedTensorTyped<T, DynRank>),
    Minimal(MinimalTensorTyped<T>)
}

impl<T> TensorInfoTyped<T>
where
    T: TensorInfoType,
{
    pub(crate) fn to_dyn_type(&self) -> TensorInfo {
        match self {
            TensorInfoTyped::Numeric(numeric) => TensorInfo::Numeric(numeric.to_dyn_type()),
            TensorInfoTyped::Shaped(shaped) => TensorInfo::Shaped(shaped.to_dyn_type()),
            TensorInfoTyped::Ranked(ranked) => TensorInfo::Ranked(ranked.to_dyn_type()),
            TensorInfoTyped::Minimal(minimal) => TensorInfo::Minimal(minimal.to_dyn_type())
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum TensorInfoRanked<R: KnownRank> {
    Numeric(NumericTensor<R>),
    Shaped(ShapedTensor<R>),
    Ranked(RankedTensor<R>)
}

impl<R: KnownRank> TensorInfoRanked<R> {
    pub(crate) fn try_to_type<T: TensorInfoType>(&self) -> Result<TensorInfoTypedRanked<T, R>, TensorInfoError> {
        match self{
            TensorInfoRanked::Numeric(x) => Ok(TensorInfoTypedRanked::Numeric(x.try_to_type::<T>()?)),
            TensorInfoRanked::Shaped(x) => Ok(TensorInfoTypedRanked::Shaped(x.try_to_type::<T>()?)),
            TensorInfoRanked::Ranked(x) => Ok(TensorInfoTypedRanked::Ranked(x.try_to_type::<T>()?)),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum TensorInfo {
    Numeric(NumericTensor<DynRank>),
    Shaped(ShapedTensor<DynRank>),
    Ranked(RankedTensor<DynRank>),
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

    pub(crate) fn new_from_first_value_and_shape(first_value: ScalarInfo, shape: TensorInfoTypedRanked::<u64, P1>, symbolic_resolver: &mut SymbolicResolver) -> Self {
        match shape {
            TensorInfoTypedRanked::Numeric(x) => {
                let shape = x.to_vec().iter().map(|x| ScalarInfoTyped::Numeric(*x)).collect();
                TensorInfo::Ranked(RankedTensor::new(first_value, shape))
            }
            TensorInfoTypedRanked::Shaped(x) => {
                TensorInfo::Ranked(RankedTensor::new(first_value, x.values))
            }
            TensorInfoTypedRanked::Ranked(_x) => {
                TensorInfo::Minimal(MinimalTensor::new(first_value, SymbolicScalarTyped::new(symbolic_resolver)))
            }
        }
    }
    
    pub(crate) fn dtype(&self) -> DType {
        match self {
            Self::Numeric(tensor) => tensor.dtype(),
            Self::Shaped(tensor) => tensor.dtype(),
            Self::Ranked(x) => x.dtype(),
            Self::Minimal(x) => x.dtype()
        }
    }
    pub(crate) fn shape(&self, symbolic_resolver: &mut SymbolicResolver) -> TensorInfoTypedRanked::<u64, P1> {
        match self {
            Self::Numeric(x) => {
                let shape = x.shape();
                TensorInfoTypedRanked::Numeric(NumericTensorTyped::from_vec(shape.to_vec()))
            },
            Self::Shaped(x) => { 
                let shape = x.shape();
                TensorInfoTypedRanked::Numeric(NumericTensorTyped::from_vec(shape.to_vec()))
            },
            Self::Ranked(x) => {
                let shape = x.shape();
                TensorInfoTypedRanked::Shaped(ShapedTensorTyped::new([shape.len() as u64], shape.clone()))
            },
            Self::Minimal(x) => TensorInfoTypedRanked::Ranked(x.shape(symbolic_resolver))
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
            TensorInfo::Numeric(tensor) => ScalarInfo::Numeric(tensor.first_element().unwrap()),
            TensorInfo::Shaped(tensor) => tensor.first_element(),
            TensorInfo::Ranked(tensor) => tensor.first_element().clone(),
            TensorInfo::Minimal(tensor) => tensor.first_element().clone()
        }
    }
    
    pub(crate) fn get(&self, index: &[u64], symbolic_resolver: &mut SymbolicResolver) -> Option<ScalarInfo> {
        if index.iter().all(|x| *x == 0) {
            return Some(self.first_element())
        }
        match self {
            TensorInfo::Numeric(tensor) => Some(ScalarInfo::Numeric(tensor.get(&index).unwrap()?)),
            TensorInfo::Shaped(tensor) => tensor.get(&index.to_vec()).clone(),
            TensorInfo::Ranked(_) => Some(ScalarInfo::Symbolic(SymbolicScalar::new(self.dtype(), symbolic_resolver))),
            TensorInfo::Minimal(_) => Some(ScalarInfo::Symbolic(SymbolicScalar::new(self.dtype(), symbolic_resolver)))
        }
    }
    
    pub(crate) fn try_to_rank<R: KnownRank>(&self, symbolic_resolver: &mut SymbolicResolver) -> Result<TensorInfoRanked<R>, TensorInfoError> {
        match self {
            TensorInfo::Numeric(x) => Ok(TensorInfoRanked::Numeric(x.try_to_rank()?)),
            TensorInfo::Shaped(x) => Ok(TensorInfoRanked::Shaped(x.try_to_rank()?)),
            TensorInfo::Ranked(x) => Ok(TensorInfoRanked::Ranked(x.try_to_rank()?)),
            TensorInfo::Minimal(x) => {
                // Optimistically cast to new rank
                let mut new_shape = vec![];
                for _ in 0..R::KNOWN_LEN {
                    new_shape.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)));
                }
                let new_shape = R::UnknownDims::try_from_slice(new_shape.as_slice())?;
                Ok(TensorInfoRanked::Ranked(RankedTensor::new(x.first_element.clone(), new_shape)))
            }
        }
    } 
}