use std::fmt::Debug;
use std::ops::{ Index};
use ndarray::{Dimension, Ix1, Ix2};
use serde::{Serialize, Deserialize};
use typenum::{P1, P2};
use crate::symbolic_graph::scalar_info::ScalarInfoTyped;

#[derive(Debug, thiserror::Error)]
pub enum RankError {
    #[error("Cannot cast rank")]
    CannotCastRank
}

pub trait DimContainer<T: Clone> {
    fn as_slice(&self) -> &[T];
    fn try_from_slice(value: &[T]) -> Result<Self, RankError> where Self: Sized;
    fn len(&self) -> usize;
}

impl<T: Clone> DimContainer<T> for Vec<T> {
    fn as_slice(&self) -> &[T] {
        self.as_slice()
    }

    fn try_from_slice(value: &[T]) -> Result<Self, RankError> {
        Ok(value.to_vec())
    }

    fn len(&self) -> usize {
        self.len()
    }
}

impl<T: Clone, const L: usize> DimContainer<T> for [T; L] 
{
    fn as_slice(&self) -> &[T] {
        self.as_slice()
    }
    fn try_from_slice(value: &[T]) -> Result<Self, RankError> {
        if value.len() != L {
            return Err(RankError::CannotCastRank);
        }
        Ok(core::array::from_fn(|x| value[x].clone()))
    }
    fn len(&self) -> usize {
        L
    }
}


pub trait Rank: Debug + Clone {
    type NDArrayDim: Dimension + Serialize + for<'a> Deserialize<'a>;
    const LEN: Option<usize>;
    
    type UnknownDims: Debug + Clone + DimContainer<ScalarInfoTyped<u64>> + Index<usize, Output=ScalarInfoTyped<u64>>;
    type KnownDims: Debug + Clone + DimContainer<u64> + Index<usize, Output=u64>;
    
    fn try_cast_to_dim(dims: &[usize]) -> Result<Self::NDArrayDim, RankError>;
    fn cast_to_ndarray_dim(dims: &Self::KnownDims) -> Self::NDArrayDim;
    fn known_to_unknown_dims(dims: &Self::KnownDims) -> Self::UnknownDims;
    fn try_unknown_to_known_dims(dims: &Self::UnknownDims) -> Option<Self::KnownDims>;
}

pub trait KnownRank: Rank {
    const KNOWN_LEN: usize;
}

#[derive(Debug, Clone)]
pub struct DynRank {}

impl Rank for DynRank {
    type NDArrayDim = ndarray::IxDyn;
    const LEN: Option<usize> = None;
    type UnknownDims = Vec<ScalarInfoTyped<u64>>;
    type KnownDims = Vec<u64>;
    
    fn try_cast_to_dim(dims: &[usize]) -> Result<Self::NDArrayDim, RankError> {
        Ok(ndarray::IxDyn(dims))
    }
    
    fn cast_to_ndarray_dim(dims: &Self::KnownDims) -> Self::NDArrayDim {
        let s = dims.into_iter().map(|x| *x as usize).collect::<Vec<_>>();
        ndarray::IxDyn(s.as_slice())
    }
    
    fn known_to_unknown_dims(dims: &Self::KnownDims) -> Self::UnknownDims {
        dims.into_iter().map(|x| ScalarInfoTyped::Numeric(*x)).collect::<Vec<_>>() 
    
    }
    
    fn try_unknown_to_known_dims(dims: &Self::UnknownDims) -> Option<Self::KnownDims> {
        dims.into_iter().map(|x| x.as_numeric().map(|x| *x)).collect::<Option<Vec<_>>>()
    }
}

impl Rank for P1 {
    type NDArrayDim = Ix1;
    const LEN: Option<usize> = Some(1);
    type UnknownDims = [ScalarInfoTyped<u64>; 1];
    type KnownDims = [u64; 1];
    
    fn try_cast_to_dim(dims: &[usize]) -> Result<Self::NDArrayDim, RankError> {
        Ok(ndarray::Ix1(*dims.get(0).ok_or(RankError::CannotCastRank)?))
    }

    fn cast_to_ndarray_dim(dims: &Self::KnownDims) -> Self::NDArrayDim {
        ndarray::Ix1(dims[0] as usize)
    }

    fn known_to_unknown_dims(dims: &Self::KnownDims) -> Self::UnknownDims {
        [ScalarInfoTyped::Numeric(dims[0])]
    }
    
    fn try_unknown_to_known_dims(dims: &Self::UnknownDims) -> Option<Self::KnownDims> {
        dims[0].as_numeric().map(|x| [*x])
    }
}

impl KnownRank for P1 {
    const KNOWN_LEN: usize = 1;
}

impl Rank for P2 {
    type NDArrayDim = Ix2;
    const LEN: Option<usize> = Some(2);
    type UnknownDims = [ScalarInfoTyped<u64>; 2];
    type KnownDims = [u64; 2];

    fn try_cast_to_dim(dims: &[usize]) -> Result<Self::NDArrayDim, RankError> {
        Ok(ndarray::Ix2(
            *dims.get(0).ok_or(RankError::CannotCastRank)?,
                *dims.get(1).ok_or(RankError::CannotCastRank)?
        ))
    }

    fn cast_to_ndarray_dim(dims: &Self::KnownDims) -> Self::NDArrayDim {
        ndarray::Ix2(dims[0] as usize, dims[1] as usize)
    }
    
    fn known_to_unknown_dims(dims: &Self::KnownDims) -> Self::UnknownDims {
        [ScalarInfoTyped::Numeric(dims[0]), ScalarInfoTyped::Numeric(dims[1])]
    }
    
    fn try_unknown_to_known_dims(dims: &Self::UnknownDims) -> Option<Self::KnownDims> {
        Some([*dims[0].as_numeric()?, *dims[1].as_numeric()?])
    }
}

impl KnownRank for P2 {
    const KNOWN_LEN: usize = 2;
}