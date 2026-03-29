mod binary;
mod cast;
mod concat;
mod constant;
mod conv;
mod conv_transpose;
mod einsum;
mod gather;
mod gather_elements;
mod lrn;
mod lstm;
mod matmul_integer;
mod mel_weight_matrix;
mod misc;
mod nlll;
mod normalization;
mod onehot;
mod pool;
mod quant_matmul;
mod reduce;
mod reshape;
mod resize;
mod reverse_sequence;
mod rnn;
mod rotary_embedding;
mod scan;
mod scatter_elements;
mod scatter_nd;
mod sce;
mod shape;
mod slice;
mod split;
mod stft;
mod topk;
mod transpose;
mod unary;
mod window;

pub use binary::{
    ArgMaxOperation, ArgMinOperation, BinaryOperation, BitShiftOperation, GemmOperation,
    MaxOperation, MinOperation, ModuloOperation, PowOperation, WhichBinaryOperation,
};
pub use cast::{CastLikeOperation, CastOperation};
pub use concat::ConcatOperation;
pub use constant::{ConstantOfShapeOperation, ConstantOperation};
pub use conv::ConvOperation;
pub use conv_transpose::ConvTransposeOperation;
pub use einsum::EinsumOperation;
pub use gather::GatherOperation;
pub use gather_elements::{GatherElementsOperation, GatherNDOperation};
pub use lrn::LrnOperation;
pub use lstm::LstmOperation;
pub use matmul_integer::MatMulIntegerOperation;
pub use mel_weight_matrix::MelWeightMatrixOperation;
pub use misc::{
    ClipOperation, CompressOperation, DepthToSpaceOperation, DropoutOperation, ExpandOperation,
    EyeLikeOperation, GlobalAveragePoolOperation, GlobalMaxPoolOperation, HardmaxOperation,
    IfOperation, MeanOperation, MeanVarianceNormalizationOperation, PadOperation,
    RandomNormalLikeOperation, RangeOperation, ShrinkOperation, SpaceToDepthOperation,
    SumOperation, TileOperation, TriluOperation, WhereOperation,
};
pub use nlll::NegativeLogLikelihoodLossOperation;
pub use normalization::{
    BatchNormalizationOperation, GroupNormalizationOperation, InstanceNormalizationOperation,
    LayerNormalizationOperation, LpNormalizationOperation, RMSNormalizationOperation,
};
pub use onehot::OneHotOperation;
pub use pool::{AveragePoolOperation, LpPoolOperation, MaxPoolOperation};
pub use quant_matmul::QuantMatMulOperation;
pub use reduce::{
    CumSumOperation, ReduceL1Operation, ReduceL2Operation, ReduceLogSumExpOperation,
    ReduceLogSumOperation, ReduceMaxOperation, ReduceMeanOperation, ReduceMinOperation,
    ReduceProdOperation, ReduceSumOperation, ReduceSumSquareOperation,
};
pub use reshape::{FlattenOperation, ReshapeOperation, SqueezeOperation, UnsqueezeOperation};
pub use resize::ResizeOperation;
pub use reverse_sequence::ReverseSequenceOperation;
pub use rnn::{GruOperation, SimpleRnnOperation};
pub use rotary_embedding::RotaryEmbeddingOperation;
pub use scan::ScanOperation;
pub use scatter_elements::ScatterElementsOperation;
pub use scatter_nd::ScatterNDOperation;
pub use sce::SoftmaxCrossEntropyLossOperation;
pub use shape::{ShapeOperation, SizeOperation};
pub use slice::SliceOperation;
pub use split::SplitOperation;
pub use stft::StftOperation;
pub use topk::TopKOperation;
pub use transpose::TransposeOperation;
pub use unary::{
    BiasGeluOperation, CeluOperation, EluOperation, GeluOperation, HardSigmoidOperation,
    HardSwishOperation, IdentityOperation, IsInfOperation, LeakyReluOperation, LogSoftmaxOperation,
    MishOperation, PReluOperation, SeluOperation, SoftmaxOperation, SoftsignOperation,
    ThresholdedReluOperation, UnaryOperation, WhichUnaryOperation,
};
pub use window::{WindowKind, WindowOperation};

use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensorError;
use crate::dtype::{DType, DTypeError};
use crate::graph::{GlobalId, Node, Property};
use crate::migration::numeric_tensor::{NumericTensor, NumericTensorError};
use crate::milli_graph::{MilliLoweringContext, MilliOpGraph, MilliOpGraphError};
use crate::symbolic_graph::SymbolicGraph;
use crate::tensor_rank::DynRank;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wyrand::WyRand;

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
    Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, EvalError>;
pub trait Operation: Node {
    fn eval(
        &self,
        backend: &mut EvalBackend,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
    ) -> OperationEvalRet {
        let tensor_dtypes: HashMap<GlobalId, DType> =
            inputs.iter().map(|(id, t)| (*id, t.dtype())).collect();
        let ctx = MilliLoweringContext::new(tensor_dtypes);
        let mut rng = WyRand::new(Default::default());
        let milli_graph = self.get_milli_op_graph(&ctx, &mut rng);
        Ok(milli_graph.eval(inputs, &mut (), backend)?)
    }

    /// Pool-based evaluation. Default: lower to milli graph → pool_eval.
    ///
    /// Ops with sub-graphs (Scan, If) override this to recursively call
    /// SymbolicGraph::eval_pool on their sub-graphs.
    fn eval_pool<'p, P: crate::pool::Pool + 'p>(
        &self,
        inputs: &HashMap<GlobalId, &crate::numeric_tensor::NumericTensorView<'_, DynRank>>,
        pool: &'p P,
    ) -> Result<HashMap<GlobalId, crate::numeric_tensor::NumericTensor<'p, DynRank, P>>, EvalError>
    {
        let tensor_dtypes: HashMap<GlobalId, DType> = inputs
            .iter()
            .map(|(id, view)| (*id, view.dtype().to_legacy()))
            .collect();
        let ctx = MilliLoweringContext::new(tensor_dtypes);
        let mut rng = WyRand::new(Default::default());
        let milli_graph = self.get_milli_op_graph(&ctx, &mut rng);
        Ok(milli_graph.pool_eval(inputs, pool)?)
    }

    fn get_milli_op_graph(&self, ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph;
    fn get_sub_graphs(&self) -> Vec<&SymbolicGraph> {
        vec![]
    }
    /// Returns introspectable parameters for this operation.
    fn parameters(&self) -> Vec<Property> {
        Vec::new()
    }

    /// Generate a backward computation graph for this operation.
    ///
    /// The default implementation builds the forward milli-op graph, runs
    /// `generate_milli_backward` on it, and returns the result. This works
    /// for any operation whose milli ops implement `backward()`. Operations
    /// can override this for custom backward logic, or return `None` if
    /// not differentiable.
    fn get_backward_milli_ops(
        &self,
        ctx: &crate::milli_graph::BackwardGenContext,
        rng: &mut impl Rng,
    ) -> Option<crate::milli_graph::BackwardGenResult> {
        use crate::milli_graph::{
            BackwardGenResult, MilliOpGraph, MilliOpGroup, MilliOpPhase, generate_milli_backward,
        };

        let lowering_ctx = MilliLoweringContext::empty();
        let fwd = self.get_milli_op_graph(&lowering_ctx, rng);
        let sym_inputs: Vec<GlobalId> = self.inputs().collect();
        let sym_outputs: Vec<GlobalId> = self.outputs().collect();

        // Build workspace graph with combined-space external keys
        let mut workspace = MilliOpGraph::new_empty(rng);
        let mut comb_to_internal: HashMap<GlobalId, GlobalId> = HashMap::new();

        // Add forward input tensors
        for &comb_id in &ctx.forward_inputs {
            let internal = workspace.add_input_with_id(comb_id, rng);
            comb_to_internal.insert(comb_id, internal);
        }
        // Add output gradient tensors
        for &grad_id in ctx.output_grads.values() {
            comb_to_internal
                .entry(grad_id)
                .or_insert_with(|| workspace.add_input_with_id(grad_id, rng));
        }

        // Merge forward ops into workspace, mapping sym IDs → workspace internals
        let mut wiring: HashMap<GlobalId, GlobalId> = HashMap::new();
        for (sym, &comb) in sym_inputs.iter().zip(ctx.forward_inputs.iter()) {
            wiring.insert(*sym, comb_to_internal[&comb]);
        }
        let fwd_group = workspace.create_group(MilliOpGroup {
            id: GlobalId::new(rng),
            phase: MilliOpPhase::Forward,
            ..Default::default()
        });
        workspace.merge_graph(fwd, &mut wiring, rng, Some(fwd_group));

        // Build output grad map in workspace-internal space
        let mut internal_output_grads: HashMap<GlobalId, GlobalId> = HashMap::new();
        for (sym, &comb) in sym_outputs.iter().zip(ctx.forward_outputs.iter()) {
            if let Some(&grad_comb) = ctx.output_grads.get(&comb) {
                let internal_out = wiring[sym];
                let internal_grad = comb_to_internal[&grad_comb];
                internal_output_grads.insert(internal_out, internal_grad);
            }
        }
        if internal_output_grads.is_empty() {
            return None;
        }

        // Generate backward through forward group
        let grads = generate_milli_backward(&mut workspace, fwd_group, &internal_output_grads, rng);

        // Set outputs: gradient for each forward input → combined forward input ID
        let mut differentiable_inputs = Vec::new();
        for &comb_input in ctx.forward_inputs.iter() {
            let internal_input = comb_to_internal[&comb_input];
            if let Some(&grad_internal) = grads.get(&internal_input) {
                workspace.add_output(grad_internal, comb_input);
                differentiable_inputs.push(comb_input);
            }
        }

        if differentiable_inputs.is_empty() {
            return None;
        }

        Some(BackwardGenResult {
            graph: workspace,
            differentiable_inputs,
        })
    }

    /// Whether this operation supports differentiation.
    /// Default: true (the default get_backward_milli_ops uses milli-level backward).
    /// Override to return false for non-differentiable ops.
    fn is_differentiable(&self) -> bool {
        true
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
    AveragePool(AveragePoolOperation),
    MaxPool(MaxPoolOperation),
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
    QuantMatMul(QuantMatMulOperation),
    LeakyRelu(LeakyReluOperation),
    Lstm(LstmOperation),
    ConvTranspose(ConvTransposeOperation),
    Stft(StftOperation),
    ScatterND(ScatterNDOperation),
    GatherElements(GatherElementsOperation),
    GatherND(GatherNDOperation),
    Gelu(GeluOperation),
    BiasGelu(BiasGeluOperation),
    ReduceL2(ReduceL2Operation),
    ReduceL1(ReduceL1Operation),
    ReduceSumSquare(ReduceSumSquareOperation),
    ReduceLogSum(ReduceLogSumOperation),
    ReduceLogSumExp(ReduceLogSumExpOperation),
    TopK(TopKOperation),
    Elu(EluOperation),
    Selu(SeluOperation),
    Celu(CeluOperation),
    HardSigmoid(HardSigmoidOperation),
    HardSwish(HardSwishOperation),
    Mish(MishOperation),
    Softsign(SoftsignOperation),
    ThresholdedRelu(ThresholdedReluOperation),
    PRelu(PReluOperation),
    BatchNormalization(BatchNormalizationOperation),
    Tile(TileOperation),
    Dropout(DropoutOperation),
    GlobalAveragePool(GlobalAveragePoolOperation),
    GlobalMaxPool(GlobalMaxPoolOperation),
    MeanOp(MeanOperation),
    SumOp(SumOperation),
    SpaceToDepth(SpaceToDepthOperation),
    DepthToSpace(DepthToSpaceOperation),
    Trilu(TriluOperation),
    BitShift(BitShiftOperation),
    EyeLike(EyeLikeOperation),
    Shrink(ShrinkOperation),
    Hardmax(HardmaxOperation),
    Compress(CompressOperation),
    ScatterElements(ScatterElementsOperation),
    MeanVarianceNormalization(MeanVarianceNormalizationOperation),
    ReverseSequence(ReverseSequenceOperation),
    NegativeLogLikelihoodLoss(NegativeLogLikelihoodLossOperation),
    Einsum(EinsumOperation),
    SoftmaxCrossEntropyLoss(SoftmaxCrossEntropyLossOperation),
    OneHot(OneHotOperation),
    LpPool(LpPoolOperation),
    Lrn(LrnOperation),
    SimpleRnn(SimpleRnnOperation),
    Gru(GruOperation),
    Window(WindowOperation),
    MatMulInteger(MatMulIntegerOperation),
    MelWeightMatrix(MelWeightMatrixOperation),
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
            AnyOperation::AveragePool(x) => x.$name($($arg),*),
            AnyOperation::MaxPool(x) => x.$name($($arg),*),
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
            AnyOperation::RotaryEmbedding(x) => x.$name($($arg),*),
            AnyOperation::QuantMatMul(x) => x.$name($($arg),*),
            AnyOperation::LeakyRelu(x) => x.$name($($arg),*),
            AnyOperation::Lstm(x) => x.$name($($arg),*),
            AnyOperation::ConvTranspose(x) => x.$name($($arg),*),
            AnyOperation::Stft(x) => x.$name($($arg),*),
            AnyOperation::ScatterND(x) => x.$name($($arg),*),
            AnyOperation::GatherElements(x) => x.$name($($arg),*),
            AnyOperation::GatherND(x) => x.$name($($arg),*),
            AnyOperation::Gelu(x) => x.$name($($arg),*),
            AnyOperation::BiasGelu(x) => x.$name($($arg),*),
            AnyOperation::ReduceL2(x) => x.$name($($arg),*),
            AnyOperation::ReduceL1(x) => x.$name($($arg),*),
            AnyOperation::ReduceSumSquare(x) => x.$name($($arg),*),
            AnyOperation::ReduceLogSum(x) => x.$name($($arg),*),
            AnyOperation::ReduceLogSumExp(x) => x.$name($($arg),*),
            AnyOperation::TopK(x) => x.$name($($arg),*),
            AnyOperation::Elu(x) => x.$name($($arg),*),
            AnyOperation::Selu(x) => x.$name($($arg),*),
            AnyOperation::Celu(x) => x.$name($($arg),*),
            AnyOperation::HardSigmoid(x) => x.$name($($arg),*),
            AnyOperation::HardSwish(x) => x.$name($($arg),*),
            AnyOperation::Mish(x) => x.$name($($arg),*),
            AnyOperation::Softsign(x) => x.$name($($arg),*),
            AnyOperation::ThresholdedRelu(x) => x.$name($($arg),*),
            AnyOperation::PRelu(x) => x.$name($($arg),*),
            AnyOperation::BatchNormalization(x) => x.$name($($arg),*),
            AnyOperation::Tile(x) => x.$name($($arg),*),
            AnyOperation::Dropout(x) => x.$name($($arg),*),
            AnyOperation::GlobalAveragePool(x) => x.$name($($arg),*),
            AnyOperation::GlobalMaxPool(x) => x.$name($($arg),*),
            AnyOperation::MeanOp(x) => x.$name($($arg),*),
            AnyOperation::SumOp(x) => x.$name($($arg),*),
            AnyOperation::SpaceToDepth(x) => x.$name($($arg),*),
            AnyOperation::DepthToSpace(x) => x.$name($($arg),*),
            AnyOperation::Trilu(x) => x.$name($($arg),*),
            AnyOperation::BitShift(x) => x.$name($($arg),*),
            AnyOperation::EyeLike(x) => x.$name($($arg),*),
            AnyOperation::Shrink(x) => x.$name($($arg),*),
            AnyOperation::Hardmax(x) => x.$name($($arg),*),
            AnyOperation::Compress(x) => x.$name($($arg),*),
            AnyOperation::ScatterElements(x) => x.$name($($arg),*),
            AnyOperation::MeanVarianceNormalization(x) => x.$name($($arg),*),
            AnyOperation::ReverseSequence(x) => x.$name($($arg),*),
            AnyOperation::NegativeLogLikelihoodLoss(x) => x.$name($($arg),*),
            AnyOperation::Einsum(x) => x.$name($($arg),*),
            AnyOperation::SoftmaxCrossEntropyLoss(x) => x.$name($($arg),*),
            AnyOperation::OneHot(x) => x.$name($($arg),*),
            AnyOperation::LpPool(x) => x.$name($($arg),*),
            AnyOperation::Lrn(x) => x.$name($($arg),*),
            AnyOperation::SimpleRnn(x) => x.$name($($arg),*),
            AnyOperation::Gru(x) => x.$name($($arg),*),
            AnyOperation::Window(x) => x.$name($($arg),*),
            AnyOperation::MatMulInteger(x) => x.$name($($arg),*),
            AnyOperation::MelWeightMatrix(x) => x.$name($($arg),*)
                    }
        }
    }
}

impl Node for AnyOperation {
    type OpKind = String;

    delegate!(op_kind() -> Self::OpKind);

    delegate!(inputs() -> Box<dyn Iterator<Item=GlobalId> + '_>);

    delegate!(outputs() -> Box<dyn Iterator<Item=GlobalId> + '_>);

    delegate!(global_id() -> GlobalId);
}

impl Operation for AnyOperation {
    delegate!(eval(
        backend: &mut EvalBackend,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>
    ) -> Result<Box<dyn Iterator<Item=(GlobalId, NumericTensor<DynRank>)>>, EvalError>);

    fn eval_pool<'p, P: crate::pool::Pool + 'p>(
        &self,
        inputs: &HashMap<GlobalId, &crate::numeric_tensor::NumericTensorView<'_, DynRank>>,
        pool: &'p P,
    ) -> Result<HashMap<GlobalId, crate::numeric_tensor::NumericTensor<'p, DynRank, P>>, EvalError>
    {
        match self {
            AnyOperation::Unary(x) => x.eval_pool(inputs, pool),
            AnyOperation::Binary(x) => x.eval_pool(inputs, pool),
            AnyOperation::Cast(x) => x.eval_pool(inputs, pool),
            AnyOperation::CastLike(x) => x.eval_pool(inputs, pool),
            AnyOperation::Squeeze(x) => x.eval_pool(inputs, pool),
            AnyOperation::Unsqueeze(x) => x.eval_pool(inputs, pool),
            AnyOperation::Transpose(x) => x.eval_pool(inputs, pool),
            AnyOperation::Reshape(x) => x.eval_pool(inputs, pool),
            AnyOperation::CumSum(x) => x.eval_pool(inputs, pool),
            AnyOperation::Gather(x) => x.eval_pool(inputs, pool),
            AnyOperation::LpNormalization(x) => x.eval_pool(inputs, pool),
            AnyOperation::GroupNormalization(x) => x.eval_pool(inputs, pool),
            AnyOperation::LayerNormalization(x) => x.eval_pool(inputs, pool),
            AnyOperation::RMSNormalization(x) => x.eval_pool(inputs, pool),
            AnyOperation::Shape(x) => x.eval_pool(inputs, pool),
            AnyOperation::Concat(x) => x.eval_pool(inputs, pool),
            AnyOperation::ConstantOfShape(x) => x.eval_pool(inputs, pool),
            AnyOperation::ReduceMean(x) => x.eval_pool(inputs, pool),
            AnyOperation::ReduceSum(x) => x.eval_pool(inputs, pool),
            AnyOperation::ReduceProd(x) => x.eval_pool(inputs, pool),
            AnyOperation::ReduceMax(x) => x.eval_pool(inputs, pool),
            AnyOperation::ReduceMin(x) => x.eval_pool(inputs, pool),
            AnyOperation::Pow(x) => x.eval_pool(inputs, pool),
            AnyOperation::Gemm(x) => x.eval_pool(inputs, pool),
            AnyOperation::Split(x) => x.eval_pool(inputs, pool),
            AnyOperation::Slice(x) => x.eval_pool(inputs, pool),
            AnyOperation::Where(x) => x.eval_pool(inputs, pool),
            AnyOperation::Softmax(x) => x.eval_pool(inputs, pool),
            AnyOperation::LogSoftmax(x) => x.eval_pool(inputs, pool),
            AnyOperation::Size(x) => x.eval_pool(inputs, pool),
            AnyOperation::Range(x) => x.eval_pool(inputs, pool),
            AnyOperation::Flatten(x) => x.eval_pool(inputs, pool),
            AnyOperation::Constant(x) => x.eval_pool(inputs, pool),
            AnyOperation::Identity(x) => x.eval_pool(inputs, pool),
            AnyOperation::IsInf(x) => x.eval_pool(inputs, pool),
            AnyOperation::Clip(x) => x.eval_pool(inputs, pool),
            AnyOperation::Modulo(x) => x.eval_pool(inputs, pool),
            AnyOperation::Expand(x) => x.eval_pool(inputs, pool),
            AnyOperation::Conv(x) => x.eval_pool(inputs, pool),
            AnyOperation::AveragePool(x) => x.eval_pool(inputs, pool),
            AnyOperation::MaxPool(x) => x.eval_pool(inputs, pool),
            AnyOperation::InstanceNormalization(x) => x.eval_pool(inputs, pool),
            AnyOperation::Resize(x) => x.eval_pool(inputs, pool),
            AnyOperation::Pad(x) => x.eval_pool(inputs, pool),
            AnyOperation::RandomNormalLike(x) => x.eval_pool(inputs, pool),
            AnyOperation::ArgMax(x) => x.eval_pool(inputs, pool),
            AnyOperation::ArgMin(x) => x.eval_pool(inputs, pool),
            AnyOperation::Max(x) => x.eval_pool(inputs, pool),
            AnyOperation::Min(x) => x.eval_pool(inputs, pool),
            AnyOperation::If(x) => x.eval_pool(inputs, pool),
            AnyOperation::Scan(x) => x.eval_pool(inputs, pool),
            AnyOperation::RotaryEmbedding(x) => x.eval_pool(inputs, pool),
            AnyOperation::QuantMatMul(x) => x.eval_pool(inputs, pool),
            AnyOperation::LeakyRelu(x) => x.eval_pool(inputs, pool),
            AnyOperation::Lstm(x) => x.eval_pool(inputs, pool),
            AnyOperation::ConvTranspose(x) => x.eval_pool(inputs, pool),
            AnyOperation::Stft(x) => x.eval_pool(inputs, pool),
            AnyOperation::ScatterND(x) => x.eval_pool(inputs, pool),
            AnyOperation::GatherElements(x) => x.eval_pool(inputs, pool),
            AnyOperation::GatherND(x) => x.eval_pool(inputs, pool),
            AnyOperation::Gelu(x) => x.eval_pool(inputs, pool),
            AnyOperation::BiasGelu(x) => x.eval_pool(inputs, pool),
            AnyOperation::ReduceL2(x) => x.eval_pool(inputs, pool),
            AnyOperation::ReduceL1(x) => x.eval_pool(inputs, pool),
            AnyOperation::ReduceSumSquare(x) => x.eval_pool(inputs, pool),
            AnyOperation::ReduceLogSum(x) => x.eval_pool(inputs, pool),
            AnyOperation::ReduceLogSumExp(x) => x.eval_pool(inputs, pool),
            AnyOperation::TopK(x) => x.eval_pool(inputs, pool),
            AnyOperation::Elu(x) => x.eval_pool(inputs, pool),
            AnyOperation::Selu(x) => x.eval_pool(inputs, pool),
            AnyOperation::Celu(x) => x.eval_pool(inputs, pool),
            AnyOperation::HardSigmoid(x) => x.eval_pool(inputs, pool),
            AnyOperation::HardSwish(x) => x.eval_pool(inputs, pool),
            AnyOperation::Mish(x) => x.eval_pool(inputs, pool),
            AnyOperation::Softsign(x) => x.eval_pool(inputs, pool),
            AnyOperation::ThresholdedRelu(x) => x.eval_pool(inputs, pool),
            AnyOperation::PRelu(x) => x.eval_pool(inputs, pool),
            AnyOperation::BatchNormalization(x) => x.eval_pool(inputs, pool),
            AnyOperation::Tile(x) => x.eval_pool(inputs, pool),
            AnyOperation::Dropout(x) => x.eval_pool(inputs, pool),
            AnyOperation::GlobalAveragePool(x) => x.eval_pool(inputs, pool),
            AnyOperation::GlobalMaxPool(x) => x.eval_pool(inputs, pool),
            AnyOperation::MeanOp(x) => x.eval_pool(inputs, pool),
            AnyOperation::SumOp(x) => x.eval_pool(inputs, pool),
            AnyOperation::SpaceToDepth(x) => x.eval_pool(inputs, pool),
            AnyOperation::DepthToSpace(x) => x.eval_pool(inputs, pool),
            AnyOperation::Trilu(x) => x.eval_pool(inputs, pool),
            AnyOperation::BitShift(x) => x.eval_pool(inputs, pool),
            AnyOperation::EyeLike(x) => x.eval_pool(inputs, pool),
            AnyOperation::Shrink(x) => x.eval_pool(inputs, pool),
            AnyOperation::Hardmax(x) => x.eval_pool(inputs, pool),
            AnyOperation::Compress(x) => x.eval_pool(inputs, pool),
            AnyOperation::ScatterElements(x) => x.eval_pool(inputs, pool),
            AnyOperation::MeanVarianceNormalization(x) => x.eval_pool(inputs, pool),
            AnyOperation::ReverseSequence(x) => x.eval_pool(inputs, pool),
            AnyOperation::NegativeLogLikelihoodLoss(x) => x.eval_pool(inputs, pool),
            AnyOperation::Einsum(x) => x.eval_pool(inputs, pool),
            AnyOperation::SoftmaxCrossEntropyLoss(x) => x.eval_pool(inputs, pool),
            AnyOperation::OneHot(x) => x.eval_pool(inputs, pool),
            AnyOperation::LpPool(x) => x.eval_pool(inputs, pool),
            AnyOperation::Lrn(x) => x.eval_pool(inputs, pool),
            AnyOperation::SimpleRnn(x) => x.eval_pool(inputs, pool),
            AnyOperation::Gru(x) => x.eval_pool(inputs, pool),
            AnyOperation::Window(x) => x.eval_pool(inputs, pool),
            AnyOperation::MatMulInteger(x) => x.eval_pool(inputs, pool),
            AnyOperation::MelWeightMatrix(x) => x.eval_pool(inputs, pool),
        }
    }

    delegate!(get_milli_op_graph(ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph);

    delegate!(parameters() -> Vec<Property>);

    delegate!(get_backward_milli_ops(
        ctx: &crate::milli_graph::BackwardGenContext,
        rng: &mut impl Rng
    ) -> Option<crate::milli_graph::BackwardGenResult>);

    delegate!(is_differentiable() -> bool);
}

impl AnyOperation {
    /// Return a clone with input tensor GlobalIds remapped according to the provided map.
    /// Only values that are actual inputs of this operation are remapped; outputs,
    /// the operation's own ID, and non-GlobalId fields remain unchanged.
    ///
    /// Uses serde round-trip. Safe because GlobalIds are random u64 values that won't
    /// collide with small integer parameters (axis, epsilon, etc.).
    pub fn remap_inputs(&self, map: &HashMap<GlobalId, GlobalId>) -> Self {
        // Build replacement map restricted to actual inputs of this operation
        let input_remaps: HashMap<u64, u64> = self
            .inputs()
            .filter_map(|id| map.get(&id).map(|new| (id.0, new.0)))
            .collect();
        if input_remaps.is_empty() {
            return self.clone();
        }
        let mut json = serde_json::to_value(self).unwrap();
        remap_u64s_in_json(&mut json, &input_remaps);
        serde_json::from_value(json).unwrap()
    }
}

/// Recursively walk a JSON value tree, replacing u64 values found in the map.
fn remap_u64s_in_json(value: &mut serde_json::Value, map: &HashMap<u64, u64>) {
    match value {
        serde_json::Value::Number(n) => {
            if let Some(v) = n.as_u64()
                && let Some(&new) = map.get(&v)
            {
                *value = serde_json::Value::Number(new.into());
            }
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                remap_u64s_in_json(item, map);
            }
        }
        serde_json::Value::Object(obj) => {
            for (_, v) in obj {
                remap_u64s_in_json(v, map);
            }
        }
        _ => {}
    }
}

/// Bridge: run a legacy Operation::eval through pool types.
///
/// Converts pool tensor views → legacy NumericTensor, calls op.eval(),
/// converts results back → pool tensors. Temporary compatibility shim
/// for ops that haven't been ported to pool-native eval_pool yet.
/// Bridge: run a legacy Operation::eval through pool types.
///
/// Converts pool tensor views → legacy NumericTensor, calls op.eval(),
/// converts results back → pool tensors. Temporary compatibility shim
/// for ops that haven't been ported to pool-native eval_pool yet.
pub(crate) fn eval_pool_via_legacy<'p, O, P>(
    op: &O,
    inputs: &HashMap<GlobalId, &crate::numeric_tensor::NumericTensorView<'_, DynRank>>,
    pool: &'p P,
) -> Result<HashMap<GlobalId, crate::numeric_tensor::NumericTensor<'p, DynRank, P>>, EvalError>
where
    O: Operation,
    P: crate::pool::Pool + 'p,
{
    use crate::migration::bridge;
    use crate::numeric_tensor::TensorLayout;

    // Convert pool views → legacy tensors.
    let legacy_inputs: HashMap<GlobalId, NumericTensor<DynRank>> = inputs
        .iter()
        .map(|(&id, view)| (id, bridge::view_to_legacy(view)))
        .collect();

    // Run legacy eval.
    let mut backend = EvalBackend::NDArray;
    let legacy_outputs: HashMap<GlobalId, NumericTensor<DynRank>> =
        op.eval(&mut backend, &legacy_inputs)?.collect();

    // Convert legacy results → pool tensors.
    let mut pool_outputs = HashMap::new();
    for (id, legacy_tensor) in &legacy_outputs {
        let new_tensor = bridge::legacy_to_new(legacy_tensor);
        let view = new_tensor.view();
        let layout = TensorLayout::<DynRank>::row_major(view.shape().to_vec(), view.dtype());
        let buf = pool
            .allocate(layout.buffer_size_bytes())
            .map_err(|e| EvalError::InvalidInput(format!("pool allocation: {e}")))?;
        let mut out = crate::numeric_tensor::NumericTensor::from_parts(buf, layout);
        for i in 0..view.numel() {
            out.write_element(i, view.read_element(i));
        }
        pool_outputs.insert(*id, out);
    }
    Ok(pool_outputs)
}
