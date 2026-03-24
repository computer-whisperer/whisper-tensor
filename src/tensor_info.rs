use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::conversions::NDArrayNumericTensorType;
use crate::migration::numeric_scalar::NumericScalarType;
use crate::migration::numeric_tensor::{NumericTensor, NumericTensorError};
use crate::migration::numeric_tensor_typed::NumericTensorTyped;
use crate::numeric_dtype::NumericDType;
use crate::numeric_scalar::NumericScalar as NewNumericScalar;
use crate::pool::Pool;
use crate::scalar_info::{ScalarInfo, ScalarInfoTyped};
use crate::symbolic_scalar::{SymbolicResolver, SymbolicScalar, SymbolicScalarTyped};
use crate::tensor_rank::{DimContainer, DynRank, KnownRank, Rank, RankError};
use ndarray::{ArcArray, ShapeError};
use std::marker::PhantomData;
use typenum::P1;

#[derive(Debug, thiserror::Error)]
pub enum TensorInfoError {
    #[error(transparent)]
    ShapeError(#[from] ShapeError),
    #[error("Cannot cast to rank")]
    CannotConvertToRank,
    #[error("Cannot cast to type")]
    CannotConvertToType,
    #[error(transparent)]
    RankError(#[from] RankError),
    #[error(transparent)]
    NumericTensorError(#[from] NumericTensorError),
}

/// A tensor with fully known shape and per-element scalar info.
///
/// Dtype-erased: stores `NumericDType` + `Vec<ScalarInfo>` instead of
/// per-Rust-type enum variants. This replaces the old macro-generated
/// `ShapedTensor` enum that had one variant per dtype.
#[derive(Clone, Debug)]
pub struct ShapedTensor<R: Rank> {
    dtype: NumericDType,
    shape: R::KnownDims,
    values: Vec<ScalarInfo>,
}

impl<R: Rank> ShapedTensor<R> {
    pub(crate) fn dtype(&self) -> NumericDType {
        self.dtype
    }

    pub(crate) fn shape(&self) -> &R::KnownDims {
        &self.shape
    }

    pub(crate) fn rank(&self) -> usize {
        self.shape.len()
    }

    pub(crate) fn first_element(&self) -> ScalarInfo {
        self.values[0].clone()
    }

    #[allow(dead_code)]
    pub(crate) fn get(&self, index: &R::KnownDims) -> Option<ScalarInfo> {
        assert_eq!(index.len(), 1);
        self.values.get(index[0] as usize).cloned()
    }

    pub(crate) fn reshape(&self, new_shape: R::KnownDims) -> Self {
        Self {
            dtype: self.dtype,
            shape: new_shape,
            values: self.values.clone(),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn try_upgrade_as_numeric_tensor(&self) -> Option<NumericTensor<R>> {
        // TODO: needs rework — old path went through typed NumericScalar variants
        // to construct NDArray-backed NumericTensor. New path should construct
        // pool-backed NumericTensor directly from ScalarInfo values.
        None
    }

    pub(crate) fn try_to_rank<R1: Rank>(&self) -> Result<ShapedTensor<R1>, TensorInfoError> {
        let new_shape = R1::KnownDims::try_from_slice(self.shape.as_slice())?;
        Ok(ShapedTensor {
            dtype: self.dtype,
            shape: new_shape,
            values: self.values.clone(),
        })
    }

    pub(crate) fn to_dyn_rank(&self) -> ShapedTensor<DynRank> {
        self.try_to_rank().unwrap()
    }

    pub(crate) fn new_symbolic(
        first_element: ScalarInfo,
        shape: R::KnownDims,
        symbolic_resolver: &mut SymbolicResolver,
    ) -> Self {
        let dtype = first_element.dtype();
        let num_values: u64 = shape.as_slice().iter().product();
        let mut values = vec![first_element];
        for _ in 1..num_values {
            values.push(ScalarInfo::Symbolic(SymbolicScalar::new(
                dtype,
                symbolic_resolver,
            )));
        }
        Self {
            dtype,
            shape,
            values,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ShapedTensorTyped<T, R: Rank>
where
    T: Clone + PartialEq + Copy + NumericScalarType,
{
    shape: R::KnownDims,
    values: Vec<ScalarInfoTyped<T>>,
}

impl<T, R: Rank> ShapedTensorTyped<T, R>
where
    T: Clone + Copy + PartialEq + NumericScalarType,
{
    pub(crate) fn new(shape: R::KnownDims, values: Vec<ScalarInfoTyped<T>>) -> Self {
        Self { shape, values }
    }

    pub(crate) fn new_symbolic(
        first_element: ScalarInfoTyped<T>,
        shape: R::KnownDims,
        symbolic_resolver: &mut SymbolicResolver,
    ) -> Self {
        let num_values = shape.as_slice().iter().product();
        let mut values = vec![first_element];
        for _ in 1..num_values {
            values.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(
                symbolic_resolver,
            )));
        }
        Self::new(shape, values)
    }

    pub(crate) fn shape(&self) -> &R::KnownDims {
        &self.shape
    }
    pub(crate) fn rank(&self) -> usize {
        self.shape.len()
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

    pub(crate) fn try_to_rank<R1: Rank>(
        &self,
    ) -> Result<ShapedTensorTyped<T, R1>, TensorInfoError> {
        let slice = self.shape.as_slice();
        let new_shape = R1::KnownDims::try_from_slice(slice)?;
        Ok(ShapedTensorTyped {
            shape: new_shape,
            values: self.values.clone(),
        })
    }

    pub(crate) fn to_dyn_rank(&self) -> ShapedTensorTyped<T, DynRank> {
        self.try_to_rank().unwrap()
    }

    /// Convert typed tensor to dtype-erased ShapedTensor.
    pub(crate) fn to_dyn_type(&self) -> ShapedTensor<R> {
        let values: Vec<ScalarInfo> = self.values.iter().map(|v| v.promote()).collect();
        let dtype = values.first().map(|v| v.dtype()).unwrap_or(NumericDType::F32);
        ShapedTensor {
            dtype,
            shape: self.shape.clone(),
            values,
        }
    }

    pub(crate) fn to_vec(&self) -> Vec<T> {
        self.values
            .iter()
            .filter_map(|v| v.as_numeric().copied())
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct RankedTensorTyped<T, R: Rank>
where
    T: Clone + Copy + PartialEq + NumericScalarType,
{
    first_element: ScalarInfoTyped<T>,
    shape: R::UnknownDims,
}

impl<T: Clone + Copy + PartialEq + NumericScalarType, R: Rank> RankedTensorTyped<T, R> {
    pub(crate) fn new(first_element: ScalarInfoTyped<T>, shape: R::UnknownDims) -> Self {
        RankedTensorTyped {
            first_element,
            shape,
        }
    }

    pub(crate) fn try_to_rank<R1: Rank>(
        &self,
    ) -> Result<RankedTensorTyped<T, R1>, TensorInfoError> {
        let slice = self.shape.as_slice();
        let new_shape = R1::UnknownDims::try_from_slice(slice)?;
        Ok(RankedTensorTyped {
            first_element: self.first_element.clone(),
            shape: new_shape,
        })
    }

    pub(crate) fn to_dyn_rank(&self) -> RankedTensorTyped<T, DynRank> {
        self.try_to_rank().unwrap()
    }

    pub(crate) fn to_dyn_type(&self) -> RankedTensor<R> {
        RankedTensor {
            first_element: self.first_element.to_dyn_type(),
            shape: self.shape.clone(),
        }
    }

    pub(crate) fn shape(&self) -> &R::UnknownDims {
        &self.shape
    }

    #[allow(dead_code)]
    pub(crate) fn rank(&self) -> usize {
        self.shape.len()
    }

    #[allow(dead_code)]
    pub(crate) fn first_element(&self) -> &ScalarInfoTyped<T> {
        &self.first_element
    }
}

#[derive(Clone, Debug)]
pub struct RankedTensor<R: Rank> {
    first_element: ScalarInfo,
    shape: R::UnknownDims,
}

impl<R: Rank> RankedTensor<R> {
    // Don't trust the user to ensure shape is not fully defined (use TensorInfoRanked instead)
    fn new(first_element: ScalarInfo, shape: R::UnknownDims) -> Self {
        Self {
            first_element,
            shape,
        }
    }

    pub(crate) fn shape(&self) -> &R::UnknownDims {
        &self.shape
    }

    pub(crate) fn rank(&self) -> usize {
        self.shape.len()
    }

    pub(crate) fn dtype(&self) -> NumericDType {
        self.first_element.dtype()
    }

    pub(crate) fn first_element(&self) -> &ScalarInfo {
        &self.first_element
    }

    pub(crate) fn try_to_rank<R1: Rank>(&self) -> Result<RankedTensor<R1>, TensorInfoError> {
        Ok(RankedTensor::new(
            self.first_element.clone(),
            R1::UnknownDims::try_from_slice(self.shape.as_slice())?,
        ))
    }

    pub(crate) fn to_dyn_rank(&self) -> RankedTensor<DynRank> {
        self.try_to_rank().unwrap()
    }

    #[allow(dead_code)]
    pub(crate) fn reshape(&self, new_shape: R::UnknownDims) -> Self {
        Self {
            first_element: self.first_element.clone(),
            shape: new_shape,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MinimalTensor {
    first_element: ScalarInfo,
    rank: SymbolicScalarTyped<u32>,
}

impl MinimalTensor {
    pub(crate) fn new(first_element: ScalarInfo, rank: SymbolicScalarTyped<u32>) -> Self {
        Self {
            first_element,
            rank,
        }
    }

    pub(crate) fn dtype(&self) -> NumericDType {
        self.first_element.dtype()
    }
    pub(crate) fn first_element(&self) -> &ScalarInfo {
        &self.first_element
    }
    #[allow(dead_code)]
    pub(crate) fn rank(&self) -> &SymbolicScalarTyped<u32> {
        &self.rank
    }
    pub(crate) fn shape(
        &self,
        symbolic_resolver: &mut SymbolicResolver,
    ) -> RankedTensorTyped<u64, P1> {
        RankedTensorTyped::new(
            ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)),
            [ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(
                symbolic_resolver,
            ))],
        )
    }
}

#[derive(Clone, Debug)]
pub enum TensorInfoTypedShaped<T, R: Rank>
where
    T: Clone + PartialEq + Copy + NumericScalarType + NDArrayNumericTensorType,
{
    Numeric(NumericTensorTyped<T, R>),
    Shaped(ShapedTensorTyped<T, R>),
}

impl<T, R: Rank> TensorInfoTypedShaped<T, R>
where
    T: Clone + PartialEq + Copy + NumericScalarType + NDArrayNumericTensorType,
{
    pub(crate) fn new_from_scalar_infos(
        shape: R::KnownDims,
        values: Vec<ScalarInfoTyped<T>>,
    ) -> Result<Self, TensorInfoError> {
        if values.iter().all(|value| value.is_numeric()) {
            let mut new_values = vec![];
            for value in values {
                if let ScalarInfoTyped::Numeric(value) = value {
                    new_values.push(value);
                } else {
                    unreachable!();
                }
            }
            let ndarray_shape = R::cast_to_ndarray_dim(&shape);
            Ok(Self::Numeric(NumericTensorTyped::NDArray(
                ArcArray::from_shape_vec(ndarray_shape, new_values)?,
            )))
        } else {
            Ok(Self::Shaped(ShapedTensorTyped::new(shape, values)))
        }
    }

    pub(crate) fn shape(&self) -> R::KnownDims {
        match self {
            TensorInfoTypedShaped::Numeric(numeric) => numeric.shape(),
            TensorInfoTypedShaped::Shaped(shaped) => shaped.shape().clone(),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn rank(&self) -> usize {
        match self {
            TensorInfoTypedShaped::Numeric(numeric) => numeric.rank(),
            TensorInfoTypedShaped::Shaped(shaped) => shaped.rank(),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn get(&self, index: &R::KnownDims) -> Option<ScalarInfoTyped<T>> {
        match self {
            TensorInfoTypedShaped::Numeric(numeric) => {
                Some(ScalarInfoTyped::Numeric(*numeric.get(index)?))
            }
            TensorInfoTypedShaped::Shaped(shaped) => Some(shaped.get(index)?.clone()),
        }
    }

    pub(crate) fn to_dyn_type(&self) -> TensorInfoShaped<R> {
        match self {
            TensorInfoTypedShaped::Numeric(numeric) => {
                TensorInfoShaped::Numeric(numeric.to_dyn_type())
            }
            TensorInfoTypedShaped::Shaped(shaped) => {
                TensorInfoShaped::Symbolic(shaped.to_dyn_type())
            }
        }
    }

    pub(crate) fn to_dyn_rank(&self) -> TensorInfoTypedShaped<T, DynRank> {
        match self {
            TensorInfoTypedShaped::Numeric(numeric) => {
                TensorInfoTypedShaped::Numeric(numeric.to_dyn_rank())
            }
            TensorInfoTypedShaped::Shaped(shaped) => {
                TensorInfoTypedShaped::Shaped(shaped.to_dyn_rank())
            }
        }
    }

    pub(crate) fn as_numeric(&self) -> Option<&NumericTensorTyped<T, R>> {
        match self {
            TensorInfoTypedShaped::Numeric(numeric) => Some(numeric),
            TensorInfoTypedShaped::Shaped(_) => None,
        }
    }
}

#[derive(Clone, Debug)]
pub enum TensorInfoTypedRanked<T, R: Rank>
where
    T: Clone + PartialEq + Copy + NumericScalarType + NDArrayNumericTensorType,
{
    Shaped(TensorInfoTypedShaped<T, R>),
    Ranked(RankedTensorTyped<T, R>),
}

impl<T, R: Rank> TensorInfoTypedRanked<T, R>
where
    T: Clone + PartialEq + Copy + NumericScalarType + NDArrayNumericTensorType,
{
    #[allow(dead_code)]
    pub(crate) fn shape(&self) -> R::UnknownDims {
        match self {
            TensorInfoTypedRanked::Shaped(shaped) => R::known_to_unknown_dims(&shaped.shape()),
            TensorInfoTypedRanked::Ranked(ranked) => ranked.shape().clone(),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn rank(&self) -> usize {
        match self {
            TensorInfoTypedRanked::Shaped(shaped) => shaped.rank(),
            TensorInfoTypedRanked::Ranked(ranked) => ranked.rank(),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn get(
        &self,
        index: &R::KnownDims,
        symbolic_resolver: &mut SymbolicResolver,
    ) -> Option<ScalarInfoTyped<T>> {
        match self {
            TensorInfoTypedRanked::Shaped(shaped) => shaped.get(index).clone(),
            TensorInfoTypedRanked::Ranked(_) => Some(ScalarInfoTyped::Symbolic(
                SymbolicScalarTyped::new(symbolic_resolver),
            )),
        }
    }

    pub(crate) fn to_dyn_type(&self) -> TensorInfoRanked<R> {
        match self {
            TensorInfoTypedRanked::Shaped(shaped) => TensorInfoRanked::Shaped(shaped.to_dyn_type()),
            TensorInfoTypedRanked::Ranked(ranked) => TensorInfoRanked::Ranked(ranked.to_dyn_type()),
        }
    }

    pub(crate) fn to_dyn_rank(&self) -> TensorInfoTypedRanked<T, DynRank> {
        match self {
            TensorInfoTypedRanked::Shaped(shaped) => {
                TensorInfoTypedRanked::Shaped(shaped.to_dyn_rank())
            }
            TensorInfoTypedRanked::Ranked(ranked) => {
                TensorInfoTypedRanked::Ranked(ranked.to_dyn_rank())
            }
        }
    }

    pub(crate) fn as_shaped(&self) -> Option<&TensorInfoTypedShaped<T, R>> {
        match self {
            TensorInfoTypedRanked::Shaped(shaped) => Some(shaped),
            TensorInfoTypedRanked::Ranked(_) => None,
        }
    }

    pub(crate) fn as_numeric(&self) -> Option<&NumericTensorTyped<T, R>> {
        match self {
            TensorInfoTypedRanked::Shaped(shaped) => shaped.as_numeric(),
            TensorInfoTypedRanked::Ranked(_) => None,
        }
    }
}

#[derive(Clone, Debug)]
pub enum TensorInfoShaped<R: Rank> {
    Numeric(NumericTensor<R>),
    Symbolic(ShapedTensor<R>),
}

impl<R: Rank> TensorInfoShaped<R> {
    fn dtype(&self) -> NumericDType {
        match self {
            TensorInfoShaped::Numeric(x) => {
                NumericDType::from_legacy(x.dtype())
                    .expect("NumericTensor dtype has no NumericDType equivalent")
            }
            TensorInfoShaped::Symbolic(x) => x.dtype(),
        }
    }

    fn shape(&self) -> R::KnownDims {
        match self {
            TensorInfoShaped::Numeric(x) => x.shape(),
            TensorInfoShaped::Symbolic(x) => x.shape().clone(),
        }
    }

    fn rank(&self) -> usize {
        match self {
            TensorInfoShaped::Numeric(x) => x.rank(),
            TensorInfoShaped::Symbolic(x) => x.rank(),
        }
    }

    fn get(&self, index: &R::KnownDims) -> Option<ScalarInfo> {
        match self {
            TensorInfoShaped::Numeric(x) => x.get(index).map(|old| ScalarInfo::Numeric(crate::nano_graph::lower::legacy_scalar_to_new(&old))),
            TensorInfoShaped::Symbolic(x) => x.get(index),
        }
    }

    pub(crate) fn try_to_rank<R1: Rank>(&self) -> Result<TensorInfoShaped<R1>, TensorInfoError> {
        match self {
            TensorInfoShaped::Numeric(x) => Ok(TensorInfoShaped::Numeric(x.try_to_rank()?)),
            TensorInfoShaped::Symbolic(x) => Ok(TensorInfoShaped::Symbolic(x.try_to_rank()?)),
        }
    }

    pub(crate) fn to_dyn_rank(&self) -> TensorInfoShaped<DynRank> {
        match self {
            TensorInfoShaped::Numeric(x) => TensorInfoShaped::Numeric(x.to_dyn_rank()),
            TensorInfoShaped::Symbolic(x) => TensorInfoShaped::Symbolic(x.to_dyn_rank()),
        }
    }

    pub(crate) fn first_element(&self) -> ScalarInfo {
        match self {
            TensorInfoShaped::Numeric(x) => ScalarInfo::Numeric(crate::nano_graph::lower::legacy_scalar_to_new(&x.first_element())),
            TensorInfoShaped::Symbolic(x) => x.first_element(),
        }
    }

    pub(crate) fn as_numeric(&self) -> Option<&NumericTensor<R>> {
        match self {
            TensorInfoShaped::Numeric(x) => Some(x),
            TensorInfoShaped::Symbolic(_) => None,
        }
    }

    pub(crate) fn reshape(
        &self,
        new_shape: R::KnownDims,
        backend: &mut EvalBackend,
    ) -> Result<Self, TensorInfoError> {
        match self {
            TensorInfoShaped::Numeric(x) => {
                Ok(TensorInfoShaped::Numeric(x.reshape(new_shape, backend)?))
            }
            TensorInfoShaped::Symbolic(x) => Ok(TensorInfoShaped::Symbolic(x.reshape(new_shape))),
        }
    }
}

#[derive(Clone, Debug)]
pub enum TensorInfoRanked<R: Rank> {
    Shaped(TensorInfoShaped<R>),
    Ranked(RankedTensor<R>),
}

impl<R: Rank> TensorInfoRanked<R> {
    #[allow(dead_code)]
    pub(crate) fn new(
        first_element: ScalarInfo,
        shape: R::UnknownDims,
        symbolic_resolver: &mut SymbolicResolver,
    ) -> Self {
        if let Some(shape) = R::try_unknown_to_known_dims(&shape) {
            TensorInfoRanked::Shaped(TensorInfoShaped::Symbolic(ShapedTensor::new_symbolic(
                first_element,
                shape,
                symbolic_resolver,
            )))
        } else {
            TensorInfoRanked::Ranked(RankedTensor::new(first_element, shape))
        }
    }

    pub(crate) fn dtype(&self) -> NumericDType {
        match self {
            TensorInfoRanked::Shaped(x) => x.dtype(),
            TensorInfoRanked::Ranked(x) => x.dtype(),
        }
    }

    pub(crate) fn shape(&self) -> R::UnknownDims {
        match self {
            TensorInfoRanked::Shaped(x) => R::known_to_unknown_dims(&x.shape()),
            TensorInfoRanked::Ranked(x) => x.shape().clone(),
        }
    }

    pub(crate) fn rank(&self) -> usize {
        match self {
            TensorInfoRanked::Shaped(x) => x.rank(),
            TensorInfoRanked::Ranked(x) => x.rank(),
        }
    }

    pub(crate) fn get(
        &self,
        index: &R::KnownDims,
        symbolic_resolver: &mut SymbolicResolver,
    ) -> Option<ScalarInfo> {
        match self {
            TensorInfoRanked::Shaped(x) => x.get(index),
            TensorInfoRanked::Ranked(_x) => Some(ScalarInfo::Symbolic(SymbolicScalar::new(
                self.dtype(),
                symbolic_resolver,
            ))),
        }
    }

    pub(crate) fn try_to_rank<R1: Rank>(&self) -> Result<TensorInfoRanked<R1>, TensorInfoError> {
        match self {
            TensorInfoRanked::Shaped(x) => Ok(TensorInfoRanked::Shaped(x.try_to_rank()?)),
            TensorInfoRanked::Ranked(x) => Ok(TensorInfoRanked::Ranked(x.try_to_rank()?)),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn to_dyn_rank(&self) -> TensorInfoRanked<DynRank> {
        match self {
            TensorInfoRanked::Shaped(x) => TensorInfoRanked::Shaped(x.to_dyn_rank()),
            TensorInfoRanked::Ranked(x) => TensorInfoRanked::Ranked(x.to_dyn_rank()),
        }
    }

    pub(crate) fn first_element(&self) -> ScalarInfo {
        match self {
            TensorInfoRanked::Shaped(x) => x.first_element(),
            TensorInfoRanked::Ranked(x) => x.first_element().clone(),
        }
    }

    pub(crate) fn as_shaped(&self) -> Option<&TensorInfoShaped<R>> {
        match self {
            TensorInfoRanked::Shaped(x) => Some(x),
            TensorInfoRanked::Ranked(_x) => None,
        }
    }

    pub(crate) fn as_numeric(&self) -> Option<&NumericTensor<R>> {
        match self {
            TensorInfoRanked::Shaped(x) => x.as_numeric(),
            TensorInfoRanked::Ranked(_x) => None,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn reshape(
        &self,
        new_shape: R::UnknownDims,
        symbolic_resolver: &mut SymbolicResolver,
        eval_backend: &mut EvalBackend,
    ) -> Result<Self, TensorInfoError> {
        if let Some(new_shape) = R::try_unknown_to_known_dims(&new_shape) {
            match self {
                TensorInfoRanked::Shaped(x) => Ok(TensorInfoRanked::Shaped(
                    x.reshape(new_shape, eval_backend)?,
                )),
                TensorInfoRanked::Ranked(_x) => Ok(TensorInfoRanked::Shaped(
                    TensorInfoShaped::Symbolic(ShapedTensor::new_symbolic(
                        self.first_element(),
                        new_shape,
                        symbolic_resolver,
                    )),
                )),
            }
        } else {
            Ok(TensorInfoRanked::Ranked(RankedTensor::new(
                self.first_element(),
                new_shape,
            )))
        }
    }
}

/// Non-generic inner storage for TensorInfo.
#[derive(Clone, Debug)]
pub(crate) enum TensorInfoData {
    Ranked(TensorInfoRanked<DynRank>),
    Minimal(MinimalTensor),
}

/// Pool-generic tensor info.
///
/// The pool parameter (`P`) is carried through the type system for future use when
/// `TensorInfoShaped::Numeric` switches from legacy `NumericTensor` to pool-backed storage.
/// Currently, `P` is phantom — the internal data is the same regardless of pool type.
#[derive(Debug)]
pub struct TensorInfo<'p, P: Pool + 'p>(pub(crate) TensorInfoData, PhantomData<&'p P>);

// Manual Clone impl: P doesn't need to be Clone since it's only in PhantomData.
impl<'p, P: Pool + 'p> Clone for TensorInfo<'p, P> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), PhantomData)
    }
}

impl<'p, P: Pool + 'p> TensorInfo<'p, P> {
    /// Wrap inner data with pool phantom.
    pub(crate) fn wrap(data: TensorInfoData) -> Self {
        Self(data, PhantomData)
    }

    #[allow(dead_code)]
    pub(crate) fn new_from_first_element_and_rank(
        first_element: ScalarInfo,
        rank: ScalarInfoTyped<u32>,
        symbolic_resolver: &mut SymbolicResolver,
    ) -> Self {
        match rank {
            ScalarInfoTyped::Numeric(x) => {
                let mut new_dims = vec![];
                for _ in 0..x {
                    new_dims.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(
                        symbolic_resolver,
                    )));
                }
                Self::wrap(TensorInfoData::Ranked(TensorInfoRanked::Ranked(RankedTensor::<DynRank>::new(first_element, new_dims))))
            }
            ScalarInfoTyped::Symbolic(x) => Self::wrap(TensorInfoData::Minimal(MinimalTensor::new(first_element, x))),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn new_from_first_element_and_shape(
        first_element: ScalarInfo,
        shape: TensorInfoTypedRanked<u64, P1>,
        symbolic_resolver: &mut SymbolicResolver,
    ) -> Self {
        match shape {
            TensorInfoTypedRanked::Shaped(shape) => match shape {
                TensorInfoTypedShaped::Numeric(shape) => {
                    let shape = shape.to_vec();
                    Self::wrap(TensorInfoData::Ranked(TensorInfoRanked::Shaped(TensorInfoShaped::Symbolic(ShapedTensor::<DynRank>::new_symbolic(
                        first_element,
                        shape,
                        symbolic_resolver,
                    )))))
                }
                TensorInfoTypedShaped::Shaped(shape) => {
                    Self::wrap(TensorInfoData::Ranked(TensorInfoRanked::Ranked(RankedTensor::<DynRank>::new(first_element, shape.values))))
                }
            },
            TensorInfoTypedRanked::Ranked(shape) => Self::new_from_first_element_and_rank(
                first_element,
                shape.shape()[0].cast(),
                symbolic_resolver,
            ),
        }
    }

    pub(crate) fn dtype(&self) -> NumericDType {
        match &self.0 {
            TensorInfoData::Ranked(x) => x.dtype(),
            TensorInfoData::Minimal(x) => x.dtype(),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn shape(
        &self,
        symbolic_resolver: &mut SymbolicResolver,
    ) -> TensorInfoTypedRanked<u64, P1> {
        match &self.0 {
            TensorInfoData::Ranked(x) => {
                let shape = x.shape();
                TensorInfoTypedRanked::Shaped(
                    TensorInfoTypedShaped::new_from_scalar_infos(
                        [shape.len() as u64],
                        shape.clone(),
                    )
                    .unwrap(),
                )
            }
            TensorInfoData::Minimal(x) => TensorInfoTypedRanked::Ranked(x.shape(symbolic_resolver)),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn rank(&self) -> ScalarInfoTyped<u32> {
        match &self.0 {
            TensorInfoData::Ranked(x) => ScalarInfoTyped::Numeric(x.rank() as u32),
            TensorInfoData::Minimal(x) => ScalarInfoTyped::Symbolic(x.rank.clone()),
        }
    }
    pub(crate) fn first_element(&self) -> ScalarInfo {
        match &self.0 {
            TensorInfoData::Ranked(tensor) => tensor.first_element(),
            TensorInfoData::Minimal(tensor) => tensor.first_element().clone(),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn get(
        &self,
        index: &Vec<u64>,
        symbolic_resolver: &mut SymbolicResolver,
    ) -> Option<ScalarInfo> {
        if index.iter().all(|x| *x == 0) {
            return Some(self.first_element());
        }
        match &self.0 {
            TensorInfoData::Ranked(tensor) => tensor.get(index, symbolic_resolver),
            TensorInfoData::Minimal(_) => Some(ScalarInfo::Symbolic(SymbolicScalar::new(
                self.dtype(),
                symbolic_resolver,
            ))),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn as_ranked(&self) -> Option<&TensorInfoRanked<DynRank>> {
        match &self.0 {
            TensorInfoData::Ranked(tensor) => Some(tensor),
            TensorInfoData::Minimal(_) => None,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn as_shaped(&self) -> Option<&TensorInfoShaped<DynRank>> {
        match &self.0 {
            TensorInfoData::Ranked(tensor) => tensor.as_shaped(),
            TensorInfoData::Minimal(_) => None,
        }
    }

    /// Returns the concrete legacy NumericTensor if this TensorInfo holds concrete numeric data.
    /// Returns an owned value (cloned from internal storage).
    pub(crate) fn as_numeric(&self) -> Option<NumericTensor<DynRank>> {
        match &self.0 {
            TensorInfoData::Ranked(tensor) => tensor.as_numeric().cloned(),
            TensorInfoData::Minimal(_) => None,
        }
    }

    /// Backward-compat constructor: wrap a legacy NumericTensor into a TensorInfo.
    /// The pool parameter is for future use when storage switches to pool-backed.
    pub fn from_legacy(tensor: &NumericTensor<DynRank>, _pool: &'p P) -> Self {
        Self::wrap(TensorInfoData::Ranked(TensorInfoRanked::Shaped(TensorInfoShaped::Numeric(
            tensor.to_dyn_rank(),
        ))))
    }

    /// Returns the rank if statically known.
    pub fn rank_if_known(&self) -> Option<usize> {
        match &self.0 {
            TensorInfoData::Ranked(tensor) => Some(tensor.rank()),
            TensorInfoData::Minimal(_) => None,
        }
    }

    /// Returns the size of dimension `i` if statically known as a concrete value.
    /// Returns `None` if the rank is unknown, `i` is out of bounds, or the dim is symbolic.
    pub fn dim_if_known(&self, i: usize) -> Option<u64> {
        match &self.0 {
            TensorInfoData::Ranked(tensor) => {
                let shape = tensor.shape();
                let dim = shape.get(i)?;
                match dim {
                    ScalarInfoTyped::Numeric(v) => Some(*v),
                    ScalarInfoTyped::Symbolic(_) => None,
                }
            }
            TensorInfoData::Minimal(_) => None,
        }
    }

    /// Create a TensorInfo from ScalarInfoTyped<u64> dims (as stored in ONNXTensorInfo).
    /// Dims that are Numeric become known; Symbolic dims remain unknown.
    pub fn from_shape_scalars(shape: &[ScalarInfoTyped<u64>]) -> Self {
        let first_element = ScalarInfo::Numeric(NewNumericScalar::from_f32(0.0));
        Self::wrap(TensorInfoData::Ranked(TensorInfoRanked::Ranked(RankedTensor::new(
            first_element,
            shape.to_vec(),
        ))))
    }

    /// Create a TensorInfo with known dtype and ScalarInfoTyped dims (may be symbolic).
    pub fn from_dtype_and_shape_scalars(dtype: NumericDType, shape: &[ScalarInfoTyped<u64>]) -> Self {
        let first_element = ScalarInfo::Numeric(NewNumericScalar::zero(dtype));
        Self::wrap(TensorInfoData::Ranked(TensorInfoRanked::Ranked(RankedTensor::new(
            first_element,
            shape.to_vec(),
        ))))
    }

    /// Create a TensorInfo with known shape from a u64 slice. Dtype defaults to F32.
    /// Useful for broadcast analysis and tests.
    pub fn from_shape_u64(shape: &[u64]) -> Self {
        let dims: Vec<ScalarInfoTyped<u64>> =
            shape.iter().map(|&v| ScalarInfoTyped::Numeric(v)).collect();
        let first_element = ScalarInfo::Numeric(NewNumericScalar::from_f32(0.0));
        Self::wrap(TensorInfoData::Ranked(TensorInfoRanked::Ranked(RankedTensor::new(
            first_element,
            dims,
        ))))
    }

    /// Create a TensorInfo with known dtype and shape, but no concrete values.
    pub fn from_dtype_and_shape(dtype: NumericDType, shape: &[u64]) -> Self {
        let dims: Vec<ScalarInfoTyped<u64>> =
            shape.iter().map(|&v| ScalarInfoTyped::Numeric(v)).collect();
        let first_element = ScalarInfo::Numeric(NewNumericScalar::zero(dtype));
        Self::wrap(TensorInfoData::Ranked(TensorInfoRanked::Ranked(RankedTensor::new(
            first_element,
            dims,
        ))))
    }

    #[allow(dead_code)]
    pub(crate) fn try_to_rank<R: KnownRank>(
        &self,
        symbolic_resolver: &mut SymbolicResolver,
    ) -> Result<TensorInfoRanked<R>, TensorInfoError> {
        match &self.0 {
            TensorInfoData::Ranked(tensor) => tensor.try_to_rank(),
            TensorInfoData::Minimal(tensor) => {
                // Optimistically cast to new rank
                let mut new_shape = vec![];
                for _ in 0..R::KNOWN_LEN {
                    new_shape.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(
                        symbolic_resolver,
                    )));
                }
                let new_shape = R::UnknownDims::try_from_slice(new_shape.as_slice())?;
                Ok(TensorInfoRanked::Ranked(RankedTensor::new(
                    tensor.first_element.clone(),
                    new_shape,
                )))
            }
        }
    }
}

impl<'p, P: Pool + 'p, R: Rank, T: Clone + PartialEq + Copy + NumericScalarType + NDArrayNumericTensorType> From<TensorInfoTypedShaped<T, R>> for TensorInfo<'p, P> {
    fn from(tensor: TensorInfoTypedShaped<T, R>) -> Self {
        Self::wrap(TensorInfoData::Ranked(TensorInfoRanked::Shaped(tensor.to_dyn_rank().to_dyn_type())))
    }
}

impl<'p, P: Pool + 'p, R: Rank> From<TensorInfoShaped<R>> for TensorInfo<'p, P> {
    fn from(tensor: TensorInfoShaped<R>) -> Self {
        Self::wrap(TensorInfoData::Ranked(TensorInfoRanked::Shaped(tensor.to_dyn_rank())))
    }
}

impl<'p, P: Pool + 'p, R: Rank, T: Clone + PartialEq + Copy + NumericScalarType + NDArrayNumericTensorType> From<TensorInfoTypedRanked<T, R>> for TensorInfo<'p, P> {
    fn from(tensor: TensorInfoTypedRanked<T, R>) -> Self {
        Self::wrap(TensorInfoData::Ranked(tensor.to_dyn_rank().to_dyn_type()))
    }
}

impl<'p, P: Pool + 'p, R: Rank> From<TensorInfoRanked<R>> for TensorInfo<'p, P> {
    fn from(tensor: TensorInfoRanked<R>) -> Self {
        Self::wrap(TensorInfoData::Ranked(tensor.to_dyn_rank()))
    }
}

impl<'p, P: Pool + 'p, R: Rank, T: Clone + PartialEq + Copy + NumericScalarType + NDArrayNumericTensorType> From<NumericTensorTyped<T, R>> for TensorInfo<'p, P> {
    fn from(tensor: NumericTensorTyped<T, R>) -> Self {
        Self::wrap(TensorInfoData::Ranked(TensorInfoRanked::Shaped(TensorInfoShaped::Numeric(
            tensor.to_dyn_rank().to_dyn_type(),
        ))))
    }
}

impl<'p, P: Pool + 'p, R: Rank, T: Clone + PartialEq + Copy + NumericScalarType + NDArrayNumericTensorType> From<ShapedTensorTyped<T, R>> for TensorInfo<'p, P> {
    fn from(tensor: ShapedTensorTyped<T, R>) -> Self {
        Self::wrap(TensorInfoData::Ranked(TensorInfoRanked::Shaped(TensorInfoShaped::Symbolic(
            tensor.to_dyn_rank().to_dyn_type(),
        ))))
    }
}

impl<'p, P: Pool + 'p, R: Rank, T: Clone + PartialEq + Copy + NumericScalarType + NDArrayNumericTensorType> From<RankedTensorTyped<T, R>> for TensorInfo<'p, P> {
    fn from(tensor: RankedTensorTyped<T, R>) -> Self {
        Self::wrap(TensorInfoData::Ranked(TensorInfoRanked::Ranked(tensor.to_dyn_rank().to_dyn_type())))
    }
}

impl<'p, P: Pool + 'p, R: Rank> From<NumericTensor<R>> for TensorInfo<'p, P> {
    fn from(tensor: NumericTensor<R>) -> Self {
        Self::wrap(TensorInfoData::Ranked(TensorInfoRanked::Shaped(TensorInfoShaped::Numeric(
            tensor.to_dyn_rank(),
        ))))
    }
}

impl<'p, P: Pool + 'p, R: Rank> From<ShapedTensor<R>> for TensorInfo<'p, P> {
    fn from(tensor: ShapedTensor<R>) -> Self {
        Self::wrap(TensorInfoData::Ranked(TensorInfoRanked::Shaped(TensorInfoShaped::Symbolic(
            tensor.to_dyn_rank(),
        ))))
    }
}

impl<'p, P: Pool + 'p, R: Rank> From<RankedTensor<R>> for TensorInfo<'p, P> {
    fn from(tensor: RankedTensor<R>) -> Self {
        Self::wrap(TensorInfoData::Ranked(TensorInfoRanked::Ranked(tensor.to_dyn_rank())))
    }
}

impl<'p, P: Pool + 'p> From<MinimalTensor> for TensorInfo<'p, P> {
    fn from(tensor: MinimalTensor) -> Self {
        Self::wrap(TensorInfoData::Minimal(tensor))
    }
}
