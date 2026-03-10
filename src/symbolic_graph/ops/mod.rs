mod binary;
mod cast;
mod concat;
mod constant;
mod conv;
mod conv_transpose;
mod gather;
mod gather_elements;
mod lstm;
mod misc;
mod normalization;
mod quant_matmul;
mod reduce;
mod reshape;
mod resize;
mod rotary_embedding;
mod scan;
mod scatter_nd;
mod shape;
mod slice;
mod split;
mod stft;
mod topk;
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
pub use conv_transpose::ConvTransposeOperation;
pub use gather::GatherOperation;
pub use gather_elements::{GatherElementsOperation, GatherNDOperation};
pub use lstm::LstmOperation;
pub use misc::{
    ClipOperation, ExpandOperation, IfOperation, PadOperation, RandomNormalLikeOperation,
    RangeOperation, WhereOperation,
};
pub use normalization::{
    GroupNormalizationOperation, InstanceNormalizationOperation, LayerNormalizationOperation,
    LpNormalizationOperation, RMSNormalizationOperation,
};
pub use quant_matmul::QuantMatMulOperation;
pub use reduce::{
    CumSumOperation, ReduceL2Operation, ReduceMaxOperation, ReduceMeanOperation,
    ReduceMinOperation, ReduceProdOperation, ReduceSumOperation,
};
pub use reshape::{FlattenOperation, ReshapeOperation, SqueezeOperation, UnsqueezeOperation};
pub use resize::ResizeOperation;
pub use rotary_embedding::RotaryEmbeddingOperation;
pub use scan::ScanOperation;
pub use scatter_nd::ScatterNDOperation;
pub use shape::{ShapeOperation, SizeOperation};
pub use slice::SliceOperation;
pub use split::SplitOperation;
pub use stft::StftOperation;
pub use topk::TopKOperation;
pub use transpose::TransposeOperation;
pub use unary::{
    BiasGeluOperation, GeluOperation, IdentityOperation, IsInfOperation, LeakyReluOperation,
    LogSoftmaxOperation, SoftmaxOperation, UnaryOperation, WhichUnaryOperation,
};

use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensorError;
use crate::dtype::{DType, DTypeError};
use crate::graph::{GlobalId, Node, Property};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::numeric_tensor::{NumericTensor, NumericTensorError};
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
        let mut rng = WyRand::new(Default::default());
        let milli_graph = self.get_milli_op_graph(&mut rng);
        Ok(milli_graph.eval(inputs, &mut (), backend)?)
    }
    fn get_milli_op_graph(&self, rng: &mut impl Rng) -> MilliOpGraph;
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

        let fwd = self.get_milli_op_graph(rng);
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
    TopK(TopKOperation),
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
            AnyOperation::TopK(x) => x.$name($($arg),*)
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

    delegate!(get_milli_op_graph(rng: &mut impl Rng) -> MilliOpGraph);

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
