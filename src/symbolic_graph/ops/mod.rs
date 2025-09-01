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

pub trait Operation {
    /*fn get_op_type_name(&self) -> String {
        type_name_of_val(self).to_string()
    }*/
    fn get_op_type_name(&self) -> String;
    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId>;
    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId>;

    fn eval(
        &self,
        backend: &mut EvalBackend,
        inputs: &HashMap<SymbolicGraphTensorId, NumericTensor<DynRank>>,
    ) -> Result<HashMap<SymbolicGraphTensorId, NumericTensor<DynRank>>, EvalError> {
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

impl AnyOperation {
    fn as_dyn(&self) -> &dyn Operation {
        match self {
            AnyOperation::Unary(op) => op,
            AnyOperation::Binary(op) => op,
            AnyOperation::Cast(op) => op,
            AnyOperation::CastLike(op) => op,
            AnyOperation::Squeeze(op) => op,
            AnyOperation::Unsqueeze(op) => op,
            AnyOperation::Transpose(op) => op,
            AnyOperation::Reshape(op) => op,
            AnyOperation::CumSum(op) => op,
            AnyOperation::Gather(op) => op,
            AnyOperation::LpNormalization(op) => op,
            AnyOperation::GroupNormalization(op) => op,
            AnyOperation::LayerNormalization(op) => op,
            AnyOperation::RMSNormalization(op) => op,
            AnyOperation::Shape(op) => op,
            AnyOperation::Concat(op) => op,
            AnyOperation::ConstantOfShape(op) => op,
            AnyOperation::ReduceMean(op) => op,
            AnyOperation::ReduceSum(op) => op,
            AnyOperation::ReduceProd(op) => op,
            AnyOperation::ReduceMin(op) => op,
            AnyOperation::ReduceMax(op) => op,
            AnyOperation::Pow(op) => op,
            AnyOperation::Gemm(op) => op,
            AnyOperation::Split(op) => op,
            AnyOperation::Slice(op) => op,
            AnyOperation::Where(op) => op,
            AnyOperation::Softmax(op) => op,
            AnyOperation::LogSoftmax(op) => op,
            AnyOperation::Size(op) => op,
            AnyOperation::Range(op) => op,
            AnyOperation::Flatten(op) => op,
            AnyOperation::Constant(op) => op,
            AnyOperation::Identity(op) => op,
            AnyOperation::IsInf(op) => op,
            AnyOperation::Clip(op) => op,
            AnyOperation::Modulo(op) => op,
            AnyOperation::Expand(op) => op,
            AnyOperation::Conv(op) => op,
            AnyOperation::InstanceNormalization(op) => op,
            AnyOperation::Resize(op) => op,
            AnyOperation::Pad(op) => op,
            AnyOperation::RandomNormalLike(op) => op,
            AnyOperation::ArgMax(op) => op,
            AnyOperation::ArgMin(op) => op,
            AnyOperation::Max(op) => op,
            AnyOperation::Min(op) => op,
            AnyOperation::If(op) => op,
            AnyOperation::Scan(op) => op,
            AnyOperation::RotaryEmbedding(op) => op,
        }
    }
}

impl Operation for AnyOperation {
    fn get_op_type_name(&self) -> String {
        self.as_dyn().get_op_type_name()
    }
    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        match self {
            AnyOperation::Unary(op) => op.get_inputs(),
            AnyOperation::Binary(op) => op.get_inputs(),
            AnyOperation::Cast(op) => op.get_inputs(),
            AnyOperation::CastLike(op) => op.get_inputs(),
            AnyOperation::Squeeze(op) => op.get_inputs(),
            AnyOperation::Unsqueeze(op) => op.get_inputs(),
            AnyOperation::Transpose(op) => op.get_inputs(),
            AnyOperation::Reshape(op) => op.get_inputs(),
            AnyOperation::CumSum(op) => op.get_inputs(),
            AnyOperation::Gather(op) => op.get_inputs(),
            AnyOperation::LpNormalization(op) => op.get_inputs(),
            AnyOperation::GroupNormalization(op) => op.get_inputs(),
            AnyOperation::LayerNormalization(op) => op.get_inputs(),
            AnyOperation::RMSNormalization(op) => op.get_inputs(),
            AnyOperation::Shape(op) => op.get_inputs(),
            AnyOperation::Concat(op) => op.get_inputs(),
            AnyOperation::ConstantOfShape(op) => op.get_inputs(),
            AnyOperation::ReduceMean(op) => op.get_inputs(),
            AnyOperation::ReduceSum(op) => op.get_inputs(),
            AnyOperation::ReduceProd(op) => op.get_inputs(),
            AnyOperation::ReduceMin(op) => op.get_inputs(),
            AnyOperation::ReduceMax(op) => op.get_inputs(),
            AnyOperation::Pow(op) => op.get_inputs(),
            AnyOperation::Gemm(op) => op.get_inputs(),
            AnyOperation::Split(op) => op.get_inputs(),
            AnyOperation::Slice(op) => op.get_inputs(),
            AnyOperation::Where(op) => op.get_inputs(),
            AnyOperation::Softmax(op) => op.get_inputs(),
            AnyOperation::LogSoftmax(op) => op.get_inputs(),
            AnyOperation::Size(op) => op.get_inputs(),
            AnyOperation::Range(op) => op.get_inputs(),
            AnyOperation::Flatten(op) => op.get_inputs(),
            AnyOperation::Constant(op) => op.get_inputs(),
            AnyOperation::Identity(op) => op.get_inputs(),
            AnyOperation::IsInf(op) => op.get_inputs(),
            AnyOperation::Clip(op) => op.get_inputs(),
            AnyOperation::Modulo(op) => op.get_inputs(),
            AnyOperation::Expand(op) => op.get_inputs(),
            AnyOperation::Conv(op) => op.get_inputs(),
            AnyOperation::InstanceNormalization(op) => op.get_inputs(),
            AnyOperation::Resize(op) => op.get_inputs(),
            AnyOperation::Pad(op) => op.get_inputs(),
            AnyOperation::RandomNormalLike(op) => op.get_inputs(),
            AnyOperation::ArgMax(op) => op.get_inputs(),
            AnyOperation::ArgMin(op) => op.get_inputs(),
            AnyOperation::Max(op) => op.get_inputs(),
            AnyOperation::Min(op) => op.get_inputs(),
            AnyOperation::If(op) => op.get_inputs(),
            AnyOperation::Scan(op) => op.get_inputs(),
            AnyOperation::RotaryEmbedding(op) => op.get_inputs(),
        }
    }
    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        match self {
            AnyOperation::Unary(op) => op.get_outputs(),
            AnyOperation::Binary(op) => op.get_outputs(),
            AnyOperation::Cast(op) => op.get_outputs(),
            AnyOperation::CastLike(op) => op.get_outputs(),
            AnyOperation::Squeeze(op) => op.get_outputs(),
            AnyOperation::Unsqueeze(op) => op.get_outputs(),
            AnyOperation::Transpose(op) => op.get_outputs(),
            AnyOperation::Reshape(op) => op.get_outputs(),
            AnyOperation::CumSum(op) => op.get_outputs(),
            AnyOperation::Gather(op) => op.get_outputs(),
            AnyOperation::LpNormalization(op) => op.get_outputs(),
            AnyOperation::GroupNormalization(op) => op.get_outputs(),
            AnyOperation::LayerNormalization(op) => op.get_outputs(),
            AnyOperation::RMSNormalization(op) => op.get_outputs(),
            AnyOperation::Shape(op) => op.get_outputs(),
            AnyOperation::Concat(op) => op.get_outputs(),
            AnyOperation::ConstantOfShape(op) => op.get_outputs(),
            AnyOperation::ReduceMean(op) => op.get_outputs(),
            AnyOperation::ReduceSum(op) => op.get_outputs(),
            AnyOperation::ReduceProd(op) => op.get_outputs(),
            AnyOperation::ReduceMin(op) => op.get_outputs(),
            AnyOperation::ReduceMax(op) => op.get_outputs(),
            AnyOperation::Pow(op) => op.get_outputs(),
            AnyOperation::Gemm(op) => op.get_outputs(),
            AnyOperation::Split(op) => op.get_outputs(),
            AnyOperation::Slice(op) => op.get_outputs(),
            AnyOperation::Where(op) => op.get_outputs(),
            AnyOperation::Softmax(op) => op.get_outputs(),
            AnyOperation::LogSoftmax(op) => op.get_outputs(),
            AnyOperation::Size(op) => op.get_outputs(),
            AnyOperation::Range(op) => op.get_outputs(),
            AnyOperation::Flatten(op) => op.get_outputs(),
            AnyOperation::Constant(op) => op.get_outputs(),
            AnyOperation::Identity(op) => op.get_outputs(),
            AnyOperation::IsInf(op) => op.get_outputs(),
            AnyOperation::Clip(op) => op.get_outputs(),
            AnyOperation::Modulo(op) => op.get_outputs(),
            AnyOperation::Expand(op) => op.get_outputs(),
            AnyOperation::Conv(op) => op.get_outputs(),
            AnyOperation::InstanceNormalization(op) => op.get_outputs(),
            AnyOperation::Resize(op) => op.get_outputs(),
            AnyOperation::Pad(op) => op.get_outputs(),
            AnyOperation::RandomNormalLike(op) => op.get_outputs(),
            AnyOperation::ArgMax(op) => op.get_outputs(),
            AnyOperation::ArgMin(op) => op.get_outputs(),
            AnyOperation::Max(op) => op.get_outputs(),
            AnyOperation::Min(op) => op.get_outputs(),
            AnyOperation::If(op) => op.get_outputs(),
            AnyOperation::Scan(op) => op.get_outputs(),
            AnyOperation::RotaryEmbedding(op) => op.get_outputs(),
        }
    }

    fn eval(
        &self,
        backend: &mut EvalBackend,
        inputs: &HashMap<SymbolicGraphTensorId, NumericTensor<DynRank>>,
    ) -> Result<HashMap<SymbolicGraphTensorId, NumericTensor<DynRank>>, EvalError> {
        match self {
            AnyOperation::Unary(op) => op.eval(backend, inputs),
            AnyOperation::Binary(op) => op.eval(backend, inputs),
            AnyOperation::Cast(op) => op.eval(backend, inputs),
            AnyOperation::CastLike(op) => op.eval(backend, inputs),
            AnyOperation::Squeeze(op) => op.eval(backend, inputs),
            AnyOperation::Unsqueeze(op) => op.eval(backend, inputs),
            AnyOperation::Transpose(op) => op.eval(backend, inputs),
            AnyOperation::Reshape(op) => op.eval(backend, inputs),
            AnyOperation::CumSum(op) => op.eval(backend, inputs),
            AnyOperation::Gather(op) => op.eval(backend, inputs),
            AnyOperation::LpNormalization(op) => op.eval(backend, inputs),
            AnyOperation::GroupNormalization(op) => op.eval(backend, inputs),
            AnyOperation::LayerNormalization(op) => op.eval(backend, inputs),
            AnyOperation::RMSNormalization(op) => op.eval(backend, inputs),
            AnyOperation::Shape(op) => op.eval(backend, inputs),
            AnyOperation::Concat(op) => op.eval(backend, inputs),
            AnyOperation::ConstantOfShape(op) => op.eval(backend, inputs),
            AnyOperation::ReduceMean(op) => op.eval(backend, inputs),
            AnyOperation::ReduceSum(op) => op.eval(backend, inputs),
            AnyOperation::ReduceProd(op) => op.eval(backend, inputs),
            AnyOperation::ReduceMax(op) => op.eval(backend, inputs),
            AnyOperation::ReduceMin(op) => op.eval(backend, inputs),
            AnyOperation::Pow(op) => op.eval(backend, inputs),
            AnyOperation::Gemm(op) => op.eval(backend, inputs),
            AnyOperation::Split(op) => op.eval(backend, inputs),
            AnyOperation::Slice(op) => op.eval(backend, inputs),
            AnyOperation::Where(op) => op.eval(backend, inputs),
            AnyOperation::Softmax(op) => op.eval(backend, inputs),
            AnyOperation::LogSoftmax(op) => op.eval(backend, inputs),
            AnyOperation::Size(op) => op.eval(backend, inputs),
            AnyOperation::Range(op) => op.eval(backend, inputs),
            AnyOperation::Flatten(op) => op.eval(backend, inputs),
            AnyOperation::Constant(op) => op.eval(backend, inputs),
            AnyOperation::Identity(op) => op.eval(backend, inputs),
            AnyOperation::IsInf(op) => op.eval(backend, inputs),
            AnyOperation::Clip(op) => op.eval(backend, inputs),
            AnyOperation::Modulo(op) => op.eval(backend, inputs),
            AnyOperation::Expand(op) => op.eval(backend, inputs),
            AnyOperation::Conv(op) => op.eval(backend, inputs),
            AnyOperation::InstanceNormalization(op) => op.eval(backend, inputs),
            AnyOperation::Resize(op) => op.eval(backend, inputs),
            AnyOperation::Pad(op) => op.eval(backend, inputs),
            AnyOperation::RandomNormalLike(op) => op.eval(backend, inputs),
            AnyOperation::ArgMax(op) => op.eval(backend, inputs),
            AnyOperation::ArgMin(op) => op.eval(backend, inputs),
            AnyOperation::Max(op) => op.eval(backend, inputs),
            AnyOperation::Min(op) => op.eval(backend, inputs),
            AnyOperation::If(op) => op.eval(backend, inputs),
            AnyOperation::Scan(op) => op.eval(backend, inputs),
            AnyOperation::RotaryEmbedding(op) => op.eval(backend, inputs),
        }
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        match self {
            AnyOperation::Unary(op) => op.get_milli_op_graph(),
            AnyOperation::Binary(op) => op.get_milli_op_graph(),
            AnyOperation::Cast(op) => op.get_milli_op_graph(),
            AnyOperation::CastLike(op) => op.get_milli_op_graph(),
            AnyOperation::Squeeze(op) => op.get_milli_op_graph(),
            AnyOperation::Unsqueeze(op) => op.get_milli_op_graph(),
            AnyOperation::Transpose(op) => op.get_milli_op_graph(),
            AnyOperation::Reshape(op) => op.get_milli_op_graph(),
            AnyOperation::CumSum(op) => op.get_milli_op_graph(),
            AnyOperation::Gather(op) => op.get_milli_op_graph(),
            AnyOperation::LpNormalization(op) => op.get_milli_op_graph(),
            AnyOperation::GroupNormalization(op) => op.get_milli_op_graph(),
            AnyOperation::LayerNormalization(op) => op.get_milli_op_graph(),
            AnyOperation::RMSNormalization(op) => op.get_milli_op_graph(),
            AnyOperation::Shape(op) => op.get_milli_op_graph(),
            AnyOperation::Concat(op) => op.get_milli_op_graph(),
            AnyOperation::ConstantOfShape(op) => op.get_milli_op_graph(),
            AnyOperation::ReduceMean(op) => op.get_milli_op_graph(),
            AnyOperation::ReduceSum(op) => op.get_milli_op_graph(),
            AnyOperation::ReduceProd(op) => op.get_milli_op_graph(),
            AnyOperation::ReduceMax(op) => op.get_milli_op_graph(),
            AnyOperation::ReduceMin(op) => op.get_milli_op_graph(),
            AnyOperation::Pow(op) => op.get_milli_op_graph(),
            AnyOperation::Gemm(op) => op.get_milli_op_graph(),
            AnyOperation::Split(op) => op.get_milli_op_graph(),
            AnyOperation::Slice(op) => op.get_milli_op_graph(),
            AnyOperation::Where(op) => op.get_milli_op_graph(),
            AnyOperation::Softmax(op) => op.get_milli_op_graph(),
            AnyOperation::LogSoftmax(op) => op.get_milli_op_graph(),
            AnyOperation::Size(op) => op.get_milli_op_graph(),
            AnyOperation::Range(op) => op.get_milli_op_graph(),
            AnyOperation::Flatten(op) => op.get_milli_op_graph(),
            AnyOperation::Constant(op) => op.get_milli_op_graph(),
            AnyOperation::Identity(op) => op.get_milli_op_graph(),
            AnyOperation::IsInf(op) => op.get_milli_op_graph(),
            AnyOperation::Clip(op) => op.get_milli_op_graph(),
            AnyOperation::Modulo(op) => op.get_milli_op_graph(),
            AnyOperation::Expand(op) => op.get_milli_op_graph(),
            AnyOperation::Conv(op) => op.get_milli_op_graph(),
            AnyOperation::InstanceNormalization(op) => op.get_milli_op_graph(),
            AnyOperation::Resize(op) => op.get_milli_op_graph(),
            AnyOperation::Pad(op) => op.get_milli_op_graph(),
            AnyOperation::RandomNormalLike(op) => op.get_milli_op_graph(),
            AnyOperation::ArgMax(op) => op.get_milli_op_graph(),
            AnyOperation::ArgMin(op) => op.get_milli_op_graph(),
            AnyOperation::Max(op) => op.get_milli_op_graph(),
            AnyOperation::Min(op) => op.get_milli_op_graph(),
            AnyOperation::If(op) => op.get_milli_op_graph(),
            AnyOperation::Scan(op) => op.get_milli_op_graph(),
            AnyOperation::RotaryEmbedding(op) => op.get_milli_op_graph(),
        }
    }
}
