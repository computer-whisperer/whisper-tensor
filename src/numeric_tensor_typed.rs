use crate::backends::ndarray_backend::conversions::NDArrayNumericTensorType;
use crate::numeric_tensor::NumericTensor;
use crate::tensor_rank::{DimContainer, DynRank, Rank, RankError};
use ndarray::ArcArray;
use typenum::P1;

#[derive(Clone, Debug)]
pub enum NumericTensorTyped<T, R: Rank> {
    NDArray(ArcArray<T, R::NDArrayDim>),
}

#[derive(Debug, thiserror::Error)]
pub enum TypedNumericTensorError {
    #[error(transparent)]
    ShapeError(#[from] ndarray::ShapeError),
    #[error(transparent)]
    RankError(#[from] RankError),
    #[error("other error")]
    OtherError,
}

impl<T: NDArrayNumericTensorType, R: Rank> NumericTensorTyped<T, R> {
    pub fn to_ndarray(&self) -> ArcArray<T, R::NDArrayDim> {
        match self {
            NumericTensorTyped::NDArray(x) => x.clone(),
        }
    }

    pub fn to_dyn_type(&self) -> NumericTensor<R> {
        NumericTensor::NDArray(T::ndarray_numeric_tensor_from_ndarray(self.to_ndarray()))
    }

    pub fn to_dyn_rank(&self) -> NumericTensorTyped<T, DynRank> {
        match self {
            NumericTensorTyped::NDArray(x) => NumericTensorTyped::NDArray(x.clone().into_dyn()),
        }
    }

    pub fn try_to_rank<R1: Rank>(
        &self,
    ) -> Result<NumericTensorTyped<T, R1>, TypedNumericTensorError> {
        Ok(match self {
            NumericTensorTyped::NDArray(x) => {
                NumericTensorTyped::NDArray(x.clone().into_dimensionality()?.to_shared())
            }
        })
    }

    pub fn shape(&self) -> R::KnownDims {
        match self {
            NumericTensorTyped::NDArray(x) => R::KnownDims::try_from_slice(
                x.shape()
                    .iter()
                    .map(|x| *x as u64)
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
            .unwrap(),
        }
    }

    pub fn rank(&self) -> usize {
        match self {
            NumericTensorTyped::NDArray(x) => x.ndim(),
        }
    }

    pub fn get(&self, index: &R::KnownDims) -> Option<&T> {
        let usize_index = index
            .as_slice()
            .iter()
            .map(|x| *x as usize)
            .collect::<Vec<_>>();
        let ndarray_idx = R::try_cast_to_dim(usize_index.as_slice()).unwrap();
        match self {
            NumericTensorTyped::NDArray(x) => x.get(ndarray_idx),
        }
    }
}

impl<T: NDArrayNumericTensorType> NumericTensorTyped<T, P1> {
    pub fn from_vec(v: Vec<T>) -> Self
    where
        T: NDArrayNumericTensorType,
    {
        NumericTensorTyped::NDArray(ArcArray::from_vec(v))
    }

    pub fn to_vec(&self) -> Vec<T> {
        match self {
            NumericTensorTyped::NDArray(x) => x.to_vec(),
        }
    }
}
