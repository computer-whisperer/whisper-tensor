use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::conversions::NDArrayNumericTensorType;
use crate::dtype::DType;
use crate::numeric_scalar::NumericScalarType;
use crate::numeric_tensor::{NumericTensor, NumericTensorError};
use crate::numeric_tensor_typed::NumericTensorTyped;
use crate::scalar_info::{ScalarInfo, ScalarInfoTyped};
use crate::symbolic_scalar::{SymbolicResolver, SymbolicScalar, SymbolicScalarTyped};
use crate::tensor_rank::{DimContainer, DynRank, KnownRank, Rank, RankError};
use half::{bf16, f16};
use ndarray::{ArcArray, ShapeError};
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

pub trait TensorInfoType:
    Clone + PartialEq + Copy + NumericScalarType + NDArrayNumericTensorType
{
    fn typed_shaped_tensor_into_shaped_tensor<R: Rank>(
        value: ShapedTensorTyped<Self, R>,
    ) -> ShapedTensor<R>;
    fn try_shaped_tensor_into_typed_shaped_tensor<R: Rank>(
        value: &ShapedTensor<R>,
    ) -> Result<&ShapedTensorTyped<Self, R>, TensorInfoError>;
}

/// Defines the `ShapedTensor` enum, `TensorInfoType` impls, and all dispatch
/// methods from a single variant list. Each entry is `(rust_type, Variant, DType)`.
macro_rules! define_shaped_tensor {
    ( $(($rust_ty:ident, $variant:ident, $dtype:ident)),+ $(,)? ) => {
        // --- TensorInfoType impls ---
        $(
            impl TensorInfoType for $rust_ty {
                fn typed_shaped_tensor_into_shaped_tensor<R: Rank>(
                    value: ShapedTensorTyped<Self, R>,
                ) -> ShapedTensor<R> {
                    ShapedTensor::$variant(value)
                }
                fn try_shaped_tensor_into_typed_shaped_tensor<R: Rank>(
                    value: &ShapedTensor<R>,
                ) -> Result<&ShapedTensorTyped<Self, R>, TensorInfoError> {
                    if let ShapedTensor::$variant(v) = value {
                        Ok(v)
                    } else {
                        Err(TensorInfoError::CannotConvertToType)
                    }
                }
            }
        )+

        // --- Enum definition ---
        #[derive(Clone, Debug)]
        pub enum ShapedTensor<R: Rank> {
            $( $variant(ShapedTensorTyped<$rust_ty, R>), )+
        }

        // --- Dispatch methods ---
        impl<R: Rank> ShapedTensor<R> {
            pub(crate) fn dtype(&self) -> DType {
                match self {
                    $( ShapedTensor::$variant(_) => DType::$dtype, )+
                }
            }
            pub(crate) fn shape(&self) -> &R::KnownDims {
                match self {
                    $( ShapedTensor::$variant(x) => x.shape(), )+
                }
            }
            pub(crate) fn rank(&self) -> usize {
                match self {
                    $( ShapedTensor::$variant(x) => x.rank(), )+
                }
            }
            pub(crate) fn first_element(&self) -> ScalarInfo {
                match self {
                    $( ShapedTensor::$variant(x) => x.first_element().promote(), )+
                }
            }
            #[allow(dead_code)]
            pub(crate) fn get(&self, index: &R::KnownDims) -> Option<ScalarInfo> {
                Some(match self {
                    $( ShapedTensor::$variant(x) => x.get(index)?.promote(), )+
                })
            }
            pub(crate) fn reshape(&self, new_shape: R::KnownDims) -> Self {
                match self {
                    $( ShapedTensor::$variant(x) => Self::$variant(x.reshape(new_shape)), )+
                }
            }
            #[allow(dead_code)]
            pub(crate) fn try_upgrade_as_numeric_tensor(&self) -> Option<NumericTensor<R>> {
                match self {
                    $( ShapedTensor::$variant(x) => x.try_upgrade_as_numeric_tensor(), )+
                }
            }
            pub(crate) fn try_to_rank<R1: Rank>(&self) -> Result<ShapedTensor<R1>, TensorInfoError> {
                Ok(match self {
                    $( ShapedTensor::$variant(x) => ShapedTensor::<R1>::$variant(x.try_to_rank()?), )+
                })
            }
            pub(crate) fn try_to_type<T: TensorInfoType>(
                &self,
            ) -> Result<ShapedTensorTyped<T, R>, TensorInfoError> {
                Ok(T::try_shaped_tensor_into_typed_shaped_tensor(self)?.clone())
            }
            pub(crate) fn to_dyn_rank(&self) -> ShapedTensor<DynRank> {
                self.try_to_rank().unwrap()
            }
            pub(crate) fn new_symbolic(
                first_element: ScalarInfo,
                shape: R::KnownDims,
                symbolic_resolver: &mut SymbolicResolver,
            ) -> Self {
                match first_element.dtype() {
                    $( DType::$dtype => ShapedTensor::$variant(ShapedTensorTyped::new_symbolic(
                        first_element.cast(),
                        shape,
                        symbolic_resolver,
                    )), )+
                    other => panic!("Unsupported dtype for ShapedTensor: {:?}", other),
                }
            }
        }
    };
}

define_shaped_tensor!(
    (f64,  F64,  F64),
    (f32,  F32,  F32),
    (u64,  U64,  U64),
    (i64,  I64,  I64),
    (bf16, BF16, BF16),
    (f16,  F16,  F16),
    (i32,  I32,  I32),
    (u32,  U32,  U32),
    (i8,   I8,   I8),
    (u8,   U8,   U8),
    (bool, BOOL, BOOL),
);

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
    T: TensorInfoType,
{
    // Don't trust users to check for possible fully-numeric tensors
    fn new(shape: R::KnownDims, values: Vec<ScalarInfoTyped<T>>) -> Self {
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

    pub(crate) fn to_dyn_type(&self) -> ShapedTensor<R> {
        T::typed_shaped_tensor_into_shaped_tensor(self.clone())
    }

    #[allow(dead_code)]
    pub(crate) fn try_upgrade_as_numeric_tensor(&self) -> Option<NumericTensor<R>> {
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
            Some(
                NumericTensor::from_vec_shape(
                    entries,
                    self.shape.as_slice().iter().map(|x| *x as usize).collect(),
                )
                .unwrap(),
            )
        } else {
            None
        }
    }
}

// ShapedTensor enum and impl are generated by define_shaped_tensor! above.

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

    pub(crate) fn dtype(&self) -> DType {
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

    pub(crate) fn try_to_type<T: TensorInfoType>(
        &self,
    ) -> Result<RankedTensorTyped<T, R>, TensorInfoError> {
        Ok(RankedTensorTyped::new(
            self.first_element.cast(),
            self.shape.clone(),
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
pub struct MinimalTensorTyped<T>
where
    T: TensorInfoType,
{
    first_element: ScalarInfoTyped<T>,
    rank: SymbolicScalarTyped<u32>,
}

impl<T> MinimalTensorTyped<T>
where
    T: TensorInfoType,
{
    #[allow(dead_code)]
    pub(crate) fn to_dyn_type(&self) -> MinimalTensor {
        MinimalTensor {
            first_element: self.first_element.to_dyn_type(),
            rank: self.rank.clone(),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn rank(&self) -> &SymbolicScalarTyped<u32> {
        &self.rank
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

    pub(crate) fn dtype(&self) -> DType {
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
    T: TensorInfoType,
{
    Numeric(NumericTensorTyped<T, R>),
    Shaped(ShapedTensorTyped<T, R>),
}

impl<T, R: Rank> TensorInfoTypedShaped<T, R>
where
    T: TensorInfoType,
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
    T: TensorInfoType,
{
    Shaped(TensorInfoTypedShaped<T, R>),
    Ranked(RankedTensorTyped<T, R>),
}

impl<T, R: Rank> TensorInfoTypedRanked<T, R>
where
    T: TensorInfoType,
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

pub enum TensorInfoTyped<T>
where
    T: TensorInfoType,
{
    Ranked(TensorInfoTypedRanked<T, DynRank>),
    Minimal(MinimalTensorTyped<T>),
}

impl<T> TensorInfoTyped<T>
where
    T: TensorInfoType,
{
    #[allow(dead_code)]
    pub(crate) fn rank(&self) -> ScalarInfoTyped<u32> {
        match self {
            TensorInfoTyped::Ranked(ranked) => ScalarInfoTyped::Numeric(ranked.rank() as u32),
            TensorInfoTyped::Minimal(minimal) => ScalarInfoTyped::Symbolic(minimal.rank().clone()),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn to_dyn_type(&self) -> TensorInfo {
        match self {
            TensorInfoTyped::Ranked(ranked) => TensorInfo::Ranked(ranked.to_dyn_type()),
            TensorInfoTyped::Minimal(minimal) => TensorInfo::Minimal(minimal.to_dyn_type()),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn as_ranked(&self) -> Option<&TensorInfoTypedRanked<T, DynRank>> {
        match self {
            TensorInfoTyped::Ranked(tensor) => Some(tensor),
            TensorInfoTyped::Minimal(_) => None,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn as_shaped(&self) -> Option<&TensorInfoTypedShaped<T, DynRank>> {
        match self {
            TensorInfoTyped::Ranked(tensor) => tensor.as_shaped(),
            TensorInfoTyped::Minimal(_) => None,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn as_numeric(&self) -> Option<&NumericTensorTyped<T, DynRank>> {
        match self {
            TensorInfoTyped::Ranked(ranked) => ranked.as_numeric(),
            TensorInfoTyped::Minimal(_) => None,
        }
    }
}

#[derive(Clone, Debug)]
pub enum TensorInfoShaped<R: Rank> {
    Numeric(NumericTensor<R>),
    Symbolic(ShapedTensor<R>),
}

impl<R: Rank> TensorInfoShaped<R> {
    fn dtype(&self) -> DType {
        match self {
            TensorInfoShaped::Numeric(x) => x.dtype(),
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
            TensorInfoShaped::Numeric(x) => x.get(index).map(ScalarInfo::Numeric),
            TensorInfoShaped::Symbolic(x) => x.get(index),
        }
    }

    pub(crate) fn try_to_rank<R1: Rank>(&self) -> Result<TensorInfoShaped<R1>, TensorInfoError> {
        match self {
            TensorInfoShaped::Numeric(x) => Ok(TensorInfoShaped::Numeric(x.try_to_rank()?)),
            TensorInfoShaped::Symbolic(x) => Ok(TensorInfoShaped::Symbolic(x.try_to_rank()?)),
        }
    }

    pub(crate) fn try_to_type<T: TensorInfoType>(
        &self,
    ) -> Result<TensorInfoTypedShaped<T, R>, TensorInfoError> {
        match self {
            TensorInfoShaped::Numeric(x) => Ok(TensorInfoTypedShaped::Numeric(x.try_to_type()?)),
            TensorInfoShaped::Symbolic(x) => Ok(TensorInfoTypedShaped::Shaped(x.try_to_type()?)),
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
            TensorInfoShaped::Numeric(x) => ScalarInfo::Numeric(x.first_element()),
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

    pub(crate) fn dtype(&self) -> DType {
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
    pub(crate) fn try_to_type<T: TensorInfoType>(
        &self,
    ) -> Result<TensorInfoTypedRanked<T, R>, TensorInfoError> {
        match self {
            TensorInfoRanked::Shaped(x) => Ok(TensorInfoTypedRanked::Shaped(x.try_to_type::<T>()?)),
            TensorInfoRanked::Ranked(x) => Ok(TensorInfoTypedRanked::Ranked(x.try_to_type::<T>()?)),
        }
    }

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

#[derive(Clone, Debug)]
pub enum TensorInfo {
    Ranked(TensorInfoRanked<DynRank>),
    Minimal(MinimalTensor),
}

impl TensorInfo {
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
                Self::from(RankedTensor::<DynRank>::new(first_element, new_dims))
            }
            ScalarInfoTyped::Symbolic(x) => Self::from(MinimalTensor::new(first_element, x)),
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
                    Self::from(ShapedTensor::<DynRank>::new_symbolic(
                        first_element,
                        shape,
                        symbolic_resolver,
                    ))
                }
                TensorInfoTypedShaped::Shaped(shape) => {
                    Self::from(RankedTensor::<DynRank>::new(first_element, shape.values))
                }
            },
            TensorInfoTypedRanked::Ranked(shape) => Self::new_from_first_element_and_rank(
                first_element,
                shape.shape()[0].cast(),
                symbolic_resolver,
            ),
        }
    }

    pub(crate) fn dtype(&self) -> DType {
        match self {
            Self::Ranked(x) => x.dtype(),
            Self::Minimal(x) => x.dtype(),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn shape(
        &self,
        symbolic_resolver: &mut SymbolicResolver,
    ) -> TensorInfoTypedRanked<u64, P1> {
        match self {
            Self::Ranked(x) => {
                let shape = x.shape();
                TensorInfoTypedRanked::Shaped(
                    TensorInfoTypedShaped::new_from_scalar_infos(
                        [shape.len() as u64],
                        shape.clone(),
                    )
                    .unwrap(),
                )
            }
            Self::Minimal(x) => TensorInfoTypedRanked::Ranked(x.shape(symbolic_resolver)),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn rank(&self) -> ScalarInfoTyped<u32> {
        match self {
            Self::Ranked(x) => ScalarInfoTyped::Numeric(x.rank() as u32),
            Self::Minimal(x) => ScalarInfoTyped::Symbolic(x.rank.clone()),
        }
    }
    pub(crate) fn first_element(&self) -> ScalarInfo {
        match self {
            TensorInfo::Ranked(tensor) => tensor.first_element(),
            TensorInfo::Minimal(tensor) => tensor.first_element().clone(),
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
        match self {
            TensorInfo::Ranked(tensor) => tensor.get(index, symbolic_resolver),
            TensorInfo::Minimal(_) => Some(ScalarInfo::Symbolic(SymbolicScalar::new(
                self.dtype(),
                symbolic_resolver,
            ))),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn as_ranked(&self) -> Option<&TensorInfoRanked<DynRank>> {
        match self {
            TensorInfo::Ranked(tensor) => Some(tensor),
            TensorInfo::Minimal(_) => None,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn as_shaped(&self) -> Option<&TensorInfoShaped<DynRank>> {
        match self {
            TensorInfo::Ranked(tensor) => tensor.as_shaped(),
            TensorInfo::Minimal(_) => None,
        }
    }

    pub(crate) fn as_numeric(&self) -> Option<&NumericTensor<DynRank>> {
        match self {
            TensorInfo::Ranked(tensor) => tensor.as_numeric(),
            TensorInfo::Minimal(_) => None,
        }
    }

    /// Returns the rank if statically known.
    pub fn rank_if_known(&self) -> Option<usize> {
        match self {
            TensorInfo::Ranked(tensor) => Some(tensor.rank()),
            TensorInfo::Minimal(_) => None,
        }
    }

    /// Returns the size of dimension `i` if statically known as a concrete value.
    /// Returns `None` if the rank is unknown, `i` is out of bounds, or the dim is symbolic.
    pub fn dim_if_known(&self, i: usize) -> Option<u64> {
        match self {
            TensorInfo::Ranked(tensor) => {
                let shape = tensor.shape();
                let dim = shape.get(i)?;
                match dim {
                    ScalarInfoTyped::Numeric(v) => Some(*v),
                    ScalarInfoTyped::Symbolic(_) => None,
                }
            }
            TensorInfo::Minimal(_) => None,
        }
    }

    /// Create a TensorInfo from ScalarInfoTyped<u64> dims (as stored in ONNXTensorInfo).
    /// Dims that are Numeric become known; Symbolic dims remain unknown.
    pub fn from_shape_scalars(shape: &[ScalarInfoTyped<u64>]) -> Self {
        use crate::numeric_scalar::NumericScalar;
        let first_element = ScalarInfo::Numeric(NumericScalar::F32(0.0));
        TensorInfo::Ranked(TensorInfoRanked::Ranked(RankedTensor::new(
            first_element,
            shape.to_vec(),
        )))
    }

    /// Create a TensorInfo with known dtype and ScalarInfoTyped dims (may be symbolic).
    pub fn from_dtype_and_shape_scalars(dtype: DType, shape: &[ScalarInfoTyped<u64>]) -> Self {
        use crate::numeric_scalar::NumericScalar;
        let first_element = ScalarInfo::Numeric(NumericScalar::zero_of(dtype));
        TensorInfo::Ranked(TensorInfoRanked::Ranked(RankedTensor::new(
            first_element,
            shape.to_vec(),
        )))
    }

    /// Create a TensorInfo with known shape from a u64 slice. Dtype defaults to F32.
    /// Useful for broadcast analysis and tests.
    pub fn from_shape_u64(shape: &[u64]) -> Self {
        use crate::numeric_scalar::NumericScalar;
        let dims: Vec<ScalarInfoTyped<u64>> =
            shape.iter().map(|&v| ScalarInfoTyped::Numeric(v)).collect();
        let first_element = ScalarInfo::Numeric(NumericScalar::F32(0.0));
        TensorInfo::Ranked(TensorInfoRanked::Ranked(RankedTensor::new(
            first_element,
            dims,
        )))
    }

    /// Create a TensorInfo with known dtype and shape, but no concrete values.
    pub fn from_dtype_and_shape(dtype: DType, shape: &[u64]) -> Self {
        use crate::numeric_scalar::NumericScalar;
        let dims: Vec<ScalarInfoTyped<u64>> =
            shape.iter().map(|&v| ScalarInfoTyped::Numeric(v)).collect();
        let first_element = ScalarInfo::Numeric(NumericScalar::zero_of(dtype));
        TensorInfo::Ranked(TensorInfoRanked::Ranked(RankedTensor::new(
            first_element,
            dims,
        )))
    }

    /// Extract shape dimensions as nano_graph Dims.
    /// Known concrete dims become `Dim::Known`, symbolic dims become `Dim::Symbolic`.
    /// Returns `None` if rank is unknown (Minimal).
    pub(crate) fn dims_for_nano(&self) -> Option<Vec<crate::nano_graph::Dim>> {
        use crate::nano_graph::Dim;
        match self {
            TensorInfo::Ranked(ranked) => {
                let shape = ranked.shape();
                Some(
                    shape
                        .iter()
                        .map(|d| match d {
                            ScalarInfoTyped::Numeric(v) => Dim::Known(*v),
                            ScalarInfoTyped::Symbolic(s) => Dim::Symbolic(s.symbol_idx() as u16),
                        })
                        .collect(),
                )
            }
            TensorInfo::Minimal(_) => None,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn try_to_rank<R: KnownRank>(
        &self,
        symbolic_resolver: &mut SymbolicResolver,
    ) -> Result<TensorInfoRanked<R>, TensorInfoError> {
        match self {
            TensorInfo::Ranked(tensor) => tensor.try_to_rank(),
            TensorInfo::Minimal(tensor) => {
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

impl<R: Rank, T: TensorInfoType> From<TensorInfoTypedShaped<T, R>> for TensorInfo {
    fn from(tensor: TensorInfoTypedShaped<T, R>) -> Self {
        Self::Ranked(TensorInfoRanked::Shaped(tensor.to_dyn_rank().to_dyn_type()))
    }
}

impl<R: Rank> From<TensorInfoShaped<R>> for TensorInfo {
    fn from(tensor: TensorInfoShaped<R>) -> Self {
        Self::Ranked(TensorInfoRanked::Shaped(tensor.to_dyn_rank()))
    }
}

impl<R: Rank, T: TensorInfoType> From<TensorInfoTypedRanked<T, R>> for TensorInfo {
    fn from(tensor: TensorInfoTypedRanked<T, R>) -> Self {
        Self::Ranked(tensor.to_dyn_rank().to_dyn_type())
    }
}

impl<R: Rank> From<TensorInfoRanked<R>> for TensorInfo {
    fn from(tensor: TensorInfoRanked<R>) -> Self {
        Self::Ranked(tensor.to_dyn_rank())
    }
}

impl<R: Rank, T: TensorInfoType> From<NumericTensorTyped<T, R>> for TensorInfo {
    fn from(tensor: NumericTensorTyped<T, R>) -> Self {
        Self::Ranked(TensorInfoRanked::Shaped(TensorInfoShaped::Numeric(
            tensor.to_dyn_rank().to_dyn_type(),
        )))
    }
}

impl<R: Rank, T: TensorInfoType> From<ShapedTensorTyped<T, R>> for TensorInfo {
    fn from(tensor: ShapedTensorTyped<T, R>) -> Self {
        Self::Ranked(TensorInfoRanked::Shaped(TensorInfoShaped::Symbolic(
            tensor.to_dyn_rank().to_dyn_type(),
        )))
    }
}

impl<R: Rank, T: TensorInfoType> From<RankedTensorTyped<T, R>> for TensorInfo {
    fn from(tensor: RankedTensorTyped<T, R>) -> Self {
        Self::Ranked(TensorInfoRanked::Ranked(tensor.to_dyn_rank().to_dyn_type()))
    }
}

impl<R: Rank> From<NumericTensor<R>> for TensorInfo {
    fn from(tensor: NumericTensor<R>) -> Self {
        Self::Ranked(TensorInfoRanked::Shaped(TensorInfoShaped::Numeric(
            tensor.to_dyn_rank(),
        )))
    }
}

impl<R: Rank> From<ShapedTensor<R>> for TensorInfo {
    fn from(tensor: ShapedTensor<R>) -> Self {
        Self::Ranked(TensorInfoRanked::Shaped(TensorInfoShaped::Symbolic(
            tensor.to_dyn_rank(),
        )))
    }
}

impl<R: Rank> From<RankedTensor<R>> for TensorInfo {
    fn from(tensor: RankedTensor<R>) -> Self {
        Self::Ranked(TensorInfoRanked::Ranked(tensor.to_dyn_rank()))
    }
}

impl From<MinimalTensor> for TensorInfo {
    fn from(tensor: MinimalTensor) -> Self {
        Self::Minimal(tensor)
    }
}
