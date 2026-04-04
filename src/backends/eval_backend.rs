use crate::dtype::{DType, DTypeError};
use crate::migration::numeric_tensor::NumericTensor;
use crate::symbolic_graph::ops::EvalError;
use crate::tensor_rank::{DynRank, Rank};
use std::marker::PhantomData;

#[derive(Debug)]
#[allow(unused_lifetimes)]
pub enum EvalBackend<'a> {
    NDArray,
    NotUsed(PhantomData<&'a ()>),
}

impl<'a> core::fmt::Display for EvalBackend<'a> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl<'a> EvalBackend<'a> {
    pub fn supports_dtype(&self, _dtype: DType) -> bool {
        matches!(self, EvalBackend::NDArray)
    }

    pub fn to_native_type(&mut self, tensor: &NumericTensor<DynRank>) -> NumericTensor<DynRank> {
        match self {
            EvalBackend::NDArray => tensor.to_ndarray().unwrap().into(),
            _ => {
                unimplemented!()
            }
        }
    }

    pub fn is_on_backend<R: Rank>(&self, tensor: &NumericTensor<R>) -> bool {
        matches!(
            (self, tensor),
            (EvalBackend::NDArray, NumericTensor::NDArray(_))
        )
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EvalRuntimeError {
    #[error(transparent)]
    DTypeError(#[from] DTypeError),
    #[error("Unexpected shape: expected {0:?}, got {1:?} in shape {2:?}")]
    UnexpectedDimension(u64, u64, Vec<u64>),
    #[error("Unexpected rank: expected {0}, got {1}")]
    UnexpectedRank(usize, usize),
    #[error("Unexpected dtype: expected {0}, got {1}")]
    UnexpectedDType(DType, DType),
    #[error("Missing input tensor: {0} {1:?} {2:?}")]
    MissingInputTensor(String, Option<DType>, Option<Vec<usize>>),
    #[error("Eval Error: {0:?} {1}")]
    EvalError(Option<String>, EvalError),
    #[error("Execution cancelled")]
    Cancelled,
}
