mod binary;
mod cast;
mod concat;
mod constant;
mod conv;
mod gather;
mod misc;
mod normalization;
mod reduce;
mod reshape;
mod resize;
mod rotary_embedding;
mod scan;
mod shape;
mod slice;
mod split;
mod transpose;
mod unary;

pub use binary::{
    ArgMaxOperation, ArgMinOperation, BinaryOperation, GemmOperation, MaxOperation, MinOperation,
    ModuloOperation, PowOperation, WhichBinaryOperation,
};
pub use cast::{CastLikeOperation, CastOperation};
pub use concat::ConcatOperation;
pub use constant::{ConstantOfShapeOperation, ConstantOperation};
pub use conv::ConvOperation;
pub use gather::GatherOperation;
pub use misc::{
    ClipOperation, ExpandOperation, IfOperation, PadOperation, RandomNormalLikeOperation,
    RangeOperation, WhereOperation,
};
pub use normalization::{
    GroupNormalizationOperation, InstanceNormalizationOperation, LayerNormalizationOperation,
    LpNormalizationOperation, RMSNormalizationOperation,
};
pub use reduce::{
    CumSumOperation, ReduceMaxOperation, ReduceMeanOperation, ReduceMinOperation,
    ReduceProdOperation, ReduceSumOperation,
};
pub use reshape::{FlattenOperation, ReshapeOperation, SqueezeOperation, UnsqueezeOperation};
pub use resize::ResizeOperation;
pub use rotary_embedding::RotaryEmbeddingOperation;
pub use scan::ScanOperation;
pub use shape::{ShapeOperation, SizeOperation};
pub use slice::SliceOperation;
pub use split::SplitOperation;
pub use transpose::TransposeOperation;
pub use unary::{
    IdentityOperation, IsInfOperation, LogSoftmaxOperation, SoftmaxOperation, UnaryOperation,
    WhichUnaryOperation,
};

use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensorError;
use crate::dtype::{DType, DTypeError};
use crate::graph::Node;
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::numeric_tensor::{NumericTensor, NumericTensorError};
use crate::symbolic_graph::{SymbolicGraphInner, SymbolicGraphTensorId};
use crate::tensor_rank::DynRank;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error(transparent)]
    NDArrayNumericTensorError(#[from] NDArrayNumericTensorError),
    #[error(transparent)]
    NumericTensorError(#[from] NumericTensorError),
    #[error("Unexpected dtype: expected {0}, got {1}")]
    UnexpectedDType(DType, DType),
    #[error("Unimplemented operator: {0}")]
    UnimplementedOperatorError(String),
    #[error(transparent)]
    MilliOpGraphError(#[from] MilliOpGraphError),
    #[error("Invalid input for operation {0}")]
    InvalidInput(String),
    #[error(transparent)]
    DTypeError(#[from] DTypeError),
    #[error("Unexpected shape: expected {0:?}, got {1:?} in shape {2:?}")]
    UnexpectedDimension(u64, u64, Vec<u64>),
    #[error("Unexpected rank: expected {0}, got {1}")]
    UnexpectedRank(usize, usize),
    #[error("Missing input tensor: {0} {1:?} {2:?}")]
    MissingInputTensor(String, Option<DType>, Option<Vec<usize>>),
}

type OperationEvalRet =
    Result<Box<dyn Iterator<Item = (SymbolicGraphTensorId, NumericTensor<DynRank>)>>, EvalError>;
pub trait Operation: Node<SymbolicGraphTensorId> {
    fn eval(
        &self,
        backend: &mut EvalBackend,
        inputs: &HashMap<SymbolicGraphTensorId, NumericTensor<DynRank>>,
    ) -> OperationEvalRet {
        let milli_graph = self.get_milli_op_graph();
        Ok(milli_graph.eval(inputs, &mut (), backend)?)
    }
    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId>;
    fn get_sub_graphs(&self) -> Vec<&SymbolicGraphInner> {
        vec![]
    }
}

#[derive(Clone, Debug, strum_macros::VariantNames, Serialize, Deserialize)]
pub enum AnyOperation {
    Unary(UnaryOperation),
    Binary(BinaryOperation),
    Cast(CastOperation),
    CastLike(CastLikeOperation),
    Squeeze(SqueezeOperation),
    Unsqueeze(UnsqueezeOperation),
    Transpose(TransposeOperation),
    Reshape(ReshapeOperation),
    CumSum(CumSumOperation),
    Gather(GatherOperation),
    LpNormalization(LpNormalizationOperation),
    GroupNormalization(GroupNormalizationOperation),
    LayerNormalization(LayerNormalizationOperation),
    RMSNormalization(RMSNormalizationOperation),
    Shape(ShapeOperation),
    Concat(ConcatOperation),
    ConstantOfShape(ConstantOfShapeOperation),
    ReduceMean(ReduceMeanOperation),
    ReduceSum(ReduceSumOperation),
    ReduceProd(ReduceProdOperation),
    ReduceMin(ReduceMinOperation),
    ReduceMax(ReduceMaxOperation),
    Pow(PowOperation),
    Gemm(GemmOperation),
    Split(SplitOperation),
    Slice(SliceOperation),
    Where(WhereOperation),
    Softmax(SoftmaxOperation),
    LogSoftmax(LogSoftmaxOperation),
    Size(SizeOperation),
    Range(RangeOperation),
    Flatten(FlattenOperation),
    Constant(ConstantOperation),
    Identity(IdentityOperation),
    Clip(ClipOperation),
    IsInf(IsInfOperation),
    Modulo(ModuloOperation),
    Expand(ExpandOperation),
    Conv(ConvOperation),
    InstanceNormalization(InstanceNormalizationOperation),
    Resize(ResizeOperation),
    Pad(PadOperation),
    RandomNormalLike(RandomNormalLikeOperation),
    ArgMax(ArgMaxOperation),
    ArgMin(ArgMinOperation),
    Max(MaxOperation),
    Min(MinOperation),
    If(IfOperation),
    Scan(ScanOperation),
    RotaryEmbedding(RotaryEmbeddingOperation),
}

macro_rules! delegate {
    ($name:ident($($arg:ident: $ty:ty),*) -> $ret:ty) => {
        fn $name(&self, $($arg: $ty),*) -> $ret {
            match self {
            AnyOperation::Unary(x) => x.$name($($arg),*),
            AnyOperation::Binary(x) => x.$name($($arg),*),
            AnyOperation::Cast(x) => x.$name($($arg),*),
            AnyOperation::CastLike(x) => x.$name($($arg),*),
            AnyOperation::Squeeze(x) => x.$name($($arg),*),
            AnyOperation::Unsqueeze(x) => x.$name($($arg),*),
            AnyOperation::Transpose(x) => x.$name($($arg),*),
            AnyOperation::Reshape(x) => x.$name($($arg),*),
            AnyOperation::CumSum(x) => x.$name($($arg),*),
            AnyOperation::Gather(x) => x.$name($($arg),*),
            AnyOperation::LpNormalization(x) => x.$name($($arg),*),
            AnyOperation::GroupNormalization(x) => x.$name($($arg),*),
            AnyOperation::LayerNormalization(x) => x.$name($($arg),*),
            AnyOperation::RMSNormalization(x) => x.$name($($arg),*),
            AnyOperation::Shape(x) => x.$name($($arg),*),
            AnyOperation::Concat(x) => x.$name($($arg),*),
            AnyOperation::ConstantOfShape(x) => x.$name($($arg),*),
            AnyOperation::ReduceMean(x) => x.$name($($arg),*),
            AnyOperation::ReduceSum(x) => x.$name($($arg),*),
            AnyOperation::ReduceProd(x) => x.$name($($arg),*),
            AnyOperation::ReduceMax(x) => x.$name($($arg),*),
            AnyOperation::ReduceMin(x) => x.$name($($arg),*),
            AnyOperation::Pow(x) => x.$name($($arg),*),
            AnyOperation::Gemm(x) => x.$name($($arg),*),
            AnyOperation::Split(x) => x.$name($($arg),*),
            AnyOperation::Slice(x) => x.$name($($arg),*),
            AnyOperation::Where(x) => x.$name($($arg),*),
            AnyOperation::Softmax(x) => x.$name($($arg),*),
            AnyOperation::LogSoftmax(x) => x.$name($($arg),*),
            AnyOperation::Size(x) => x.$name($($arg),*),
            AnyOperation::Range(x) => x.$name($($arg),*),
            AnyOperation::Flatten(x) => x.$name($($arg),*),
            AnyOperation::Constant(x) => x.$name($($arg),*),
            AnyOperation::Identity(x) => x.$name($($arg),*),
            AnyOperation::IsInf(x) => x.$name($($arg),*),
            AnyOperation::Clip(x) => x.$name($($arg),*),
            AnyOperation::Modulo(x) => x.$name($($arg),*),
            AnyOperation::Expand(x) => x.$name($($arg),*),
            AnyOperation::Conv(x) => x.$name($($arg),*),
            AnyOperation::InstanceNormalization(x) => x.$name($($arg),*),
            AnyOperation::Resize(x) => x.$name($($arg),*),
            AnyOperation::Pad(x) => x.$name($($arg),*),
            AnyOperation::RandomNormalLike(x) => x.$name($($arg),*),
            AnyOperation::ArgMax(x) => x.$name($($arg),*),
            AnyOperation::ArgMin(x) => x.$name($($arg),*),
            AnyOperation::Max(x) => x.$name($($arg),*),
            AnyOperation::Min(x) => x.$name($($arg),*),
            AnyOperation::If(x) => x.$name($($arg),*),
            AnyOperation::Scan(x) => x.$name($($arg),*),
            AnyOperation::RotaryEmbedding(x) => x.$name($($arg),*)
                    }
        }
    }
}

impl Node<SymbolicGraphTensorId> for AnyOperation {
    type OpKind = String;

    delegate!(op_kind() -> Self::OpKind);

    delegate!(inputs() -> Box<dyn Iterator<Item=SymbolicGraphTensorId> + '_>);

    delegate!(outputs() -> Box<dyn Iterator<Item=SymbolicGraphTensorId> + '_>);
}

impl Operation for AnyOperation {
    delegate!(eval(
        backend: &mut EvalBackend,
        inputs: &HashMap<SymbolicGraphTensorId, NumericTensor<DynRank>>
    ) -> Result<Box<dyn Iterator<Item=(SymbolicGraphTensorId, NumericTensor<DynRank>)>>, EvalError>);

    delegate!(get_milli_op_graph() -> MilliOpGraph<SymbolicGraphTensorId>);
}
