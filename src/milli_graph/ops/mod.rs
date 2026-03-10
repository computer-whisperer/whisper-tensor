mod argmax;
mod argmin;
mod binary;
mod cast;
mod cast_like;
mod concat;
mod constant;
mod conv;
mod cumsum;
mod expand;
mod gather;
mod nonzero;
mod pad;
mod random_normal_like;
mod range;
mod reduce_max;
mod reduce_mean;
mod reduce_min;
mod reduce_prod;
mod reduce_sum;
mod reshape;
mod resize;
mod shape;
mod slice;
mod split;
mod squeeze;
mod sum_to;
mod topk;
mod transpose;
mod unary;
mod unsqueeze;
mod where_op;

pub use argmax::*;
pub use argmin::*;
pub use binary::*;
pub use cast::*;
pub use cast_like::*;
pub use concat::*;
pub use constant::*;
pub use conv::*;
pub use cumsum::*;
pub use expand::*;
pub use gather::*;
pub use nonzero::*;
pub use pad::*;
pub use random_normal_like::*;
pub use range::*;
pub use reduce_max::*;
pub use reduce_mean::*;
pub use reduce_min::*;
pub use reduce_prod::*;
pub use reduce_sum::*;
pub use reshape::*;
pub use resize::*;
pub use shape::*;
pub use slice::*;
pub use split::*;
pub use squeeze::*;
pub use sum_to::*;
pub use topk::*;
pub use transpose::*;
pub use unary::*;
pub use unsqueeze::*;
pub use where_op::*;

use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::graph::{GlobalId, Node, NodeMetadata};
use crate::milli_graph::MilliOpGraphError;
use crate::numeric_tensor::NumericTensor;
use crate::scalar_info::ScalarInfoTyped;
use crate::symbolic_scalar::{SymbolicResolver, SymbolicScalarTyped};
use crate::tensor_info::{TensorInfo, TensorInfoTypedRanked};
use crate::tensor_rank::DynRank;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

pub(crate) fn remap(id: &mut GlobalId, map: &HashMap<GlobalId, GlobalId>) {
    if let Some(&new) = map.get(id) {
        *id = new;
    }
}

pub(crate) fn remap_opt(id: &mut Option<GlobalId>, map: &HashMap<GlobalId, GlobalId>) {
    if let Some(inner) = id {
        remap(inner, map);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MilliOpTensorIDOrLiteral {
    TensorID(GlobalId),
    Literal(NDArrayNumericTensor<DynRank>),
}

pub type EvalResult =
    Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>;
pub trait MilliOp: Node {
    fn infer(
        &self,
        known_inputs: &HashMap<GlobalId, TensorInfo>,
        _symbolic_resolver: &mut SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, TensorInfo)>>, MilliOpGraphError> {
        let mut resolved_inputs = HashMap::new();
        for input in self.inputs() {
            if let Some(tensor_info) = known_inputs.get(&input) {
                if let Some(tensor) = tensor_info.as_numeric() {
                    resolved_inputs.insert(input, tensor.clone());
                } else {
                    return Err(MilliOpGraphError::UnableToInfer);
                }
            } else {
                return Err(MilliOpGraphError::UnableToInfer);
            }
        }

        let collected: Vec<(GlobalId, TensorInfo)> = self
            .eval(&resolved_inputs, backend)?
            .map(|(a, b)| (a, TensorInfo::from(b)))
            .collect();
        Ok(Box::new(collected.into_iter()))
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        _backend: &mut EvalBackend,
    ) -> EvalResult;

    /// Generate backward ops for this milli op.
    /// `output_grads` maps each output tensor ID to its gradient tensor ID.
    /// New gradient ops are added directly to `graph`.
    /// Returns input_id → gradient_id for each differentiable input.
    fn backward(
        &self,
        _output_grads: &HashMap<GlobalId, GlobalId>,
        _graph: &mut crate::milli_graph::MilliOpGraph,
        _rng: &mut impl Rng,
    ) -> Option<HashMap<GlobalId, GlobalId>> {
        None // default: not differentiable
    }
}

#[allow(dead_code)]
fn infer_multidirectional_broadcasting_rank(
    shapes: &[TensorInfoTypedRanked<u64, P1>],
    symbolic_resolver: &mut SymbolicResolver,
) -> Result<ScalarInfoTyped<u32>, MilliOpGraphError> {
    let mut output_rank: Option<usize> = None;
    for shape in shapes {
        match shape {
            TensorInfoTypedRanked::Shaped(x) => {
                let this_rank = x.shape()[0] as usize;
                if let Some(o) = output_rank {
                    output_rank = Some(o.min(this_rank));
                } else {
                    output_rank = Some(this_rank)
                }
            }
            TensorInfoTypedRanked::Ranked(_x) => {
                output_rank = None;
                break;
            }
        }
    }
    match output_rank {
        Some(x) => Ok(ScalarInfoTyped::Numeric(x as u32)),
        None => Ok(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(
            symbolic_resolver,
        ))),
    }
}

#[allow(dead_code)]
fn infer_multidirectional_broadcasting_shape(
    shapes: &[Vec<ScalarInfoTyped<u64>>],
    symbolic_resolver: &mut SymbolicResolver,
) -> Result<Vec<ScalarInfoTyped<u64>>, MilliOpGraphError> {
    if shapes.is_empty() {
        return Err(MilliOpGraphError::InvalidInput(
            "Cannot broadcast empty input".to_string(),
        ));
    }

    let output_rank = shapes.iter().map(|x| x.len()).max().unwrap();

    let mut output_shape = vec![];
    for i in 0..output_rank {
        let mut dim = ScalarInfoTyped::<u64>::Numeric(1);
        for shape in shapes {
            let rank = shape.len();
            let local_i = (i as i64 - output_rank as i64) + rank as i64;
            if local_i < 0 {
                // Infer dim of size 1, and pass
            } else {
                let local_dim = shape[local_i as usize].clone();
                match local_dim {
                    ScalarInfoTyped::Numeric(x) => {
                        if x == 1 {
                            // Do not modify the dimension, pass it through.
                        } else {
                            match dim {
                                ScalarInfoTyped::Numeric(y) => {
                                    if y == 1 || x == y {
                                        dim = ScalarInfoTyped::Numeric(y.max(x));
                                    } else {
                                        return Err(MilliOpGraphError::InvalidInput(
                                            "Cannot broadcast input shape".to_string(),
                                        ));
                                    }
                                }
                                _ => {
                                    // The only way this is valid is if the unknown dim matches the known one, so be optimistic here
                                    dim = local_dim.clone();
                                }
                            }
                        }
                    }
                    _ => {
                        // Incoming dim is unknown
                        match dim {
                            ScalarInfoTyped::Numeric(y) => {
                                if y == 1 {
                                    // Use the existing unknown dim
                                    dim = local_dim.clone();
                                } else {
                                    // The only way this is valid is if the unknown dim matches the known one, so be optimistic here
                                }
                            }
                            _ => {
                                // Two unknown dimensions
                                match dim.try_eq(&local_dim) {
                                    None => {
                                        // Must use new unknown dimension
                                        dim = ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(
                                            symbolic_resolver,
                                        ))
                                    }
                                    Some(is_same) => {
                                        if is_same {
                                            // Ok, use the unknown dim already in there
                                        } else {
                                            // Must use new unknown dimension
                                            dim = ScalarInfoTyped::Symbolic(
                                                SymbolicScalarTyped::new(symbolic_resolver),
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        output_shape.push(dim);
    }
    Ok(output_shape)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnyMilliOp {
    Constant(Constant),
    ConstantOfShape(ConstantOfShape),
    SimpleBinary(SimpleBinary),
    MatMul(MatMul),
    Pow(Pow),
    SimpleUnary(SimpleUnaryOp),
    ClampMin(ClampMin),
    NonZero(NonZero),
    CumSum(CumSum),
    Shape(Shape),
    Reshape(Reshape),
    Slice(Slice),
    ReduceSum(ReduceSum),
    ReduceMin(ReduceMin),
    ReduceMax(ReduceMax),
    ReduceProd(ReduceProd),
    ReduceMean(ReduceMean),
    Cast(Cast),
    CastLike(CastLike),
    Transpose(Transpose),
    Squeeze(Squeeze),
    Unsqueeze(Unsqueeze),
    Gather(Gather),
    GatherGrad(GatherGrad),
    Concat(Concat),
    Split(Split),
    Where(Where),
    Range(Range),
    Expand(Expand),
    SumTo(SumTo),
    ArgMax(ArgMax),
    ArgMin(ArgMin),
    Resize(Resize),
    Conv(Conv),
    ConvInputGrad(ConvInputGrad),
    ConvWeightGrad(ConvWeightGrad),
    ConvBiasGrad(ConvBiasGrad),
    Pad(Pad),
    TopK(TopK),
    RandomNormalLike(RandomNormalLike),
}

impl AnyMilliOp {
    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl Rng) {
        match self {
            AnyMilliOp::Constant(x) => x.remap_tensors(map, rng),
            AnyMilliOp::ConstantOfShape(x) => x.remap_tensors(map, rng),
            AnyMilliOp::SimpleBinary(x) => x.remap_tensors(map, rng),
            AnyMilliOp::MatMul(x) => x.remap_tensors(map, rng),
            AnyMilliOp::Pow(x) => x.remap_tensors(map, rng),
            AnyMilliOp::SimpleUnary(x) => x.remap_tensors(map, rng),
            AnyMilliOp::ClampMin(x) => x.remap_tensors(map, rng),
            AnyMilliOp::NonZero(x) => x.remap_tensors(map, rng),
            AnyMilliOp::CumSum(x) => x.remap_tensors(map, rng),
            AnyMilliOp::Shape(x) => x.remap_tensors(map, rng),
            AnyMilliOp::Reshape(x) => x.remap_tensors(map, rng),
            AnyMilliOp::Slice(x) => x.remap_tensors(map, rng),
            AnyMilliOp::ReduceSum(x) => x.remap_tensors(map, rng),
            AnyMilliOp::ReduceMin(x) => x.remap_tensors(map, rng),
            AnyMilliOp::ReduceMax(x) => x.remap_tensors(map, rng),
            AnyMilliOp::ReduceProd(x) => x.remap_tensors(map, rng),
            AnyMilliOp::ReduceMean(x) => x.remap_tensors(map, rng),
            AnyMilliOp::Cast(x) => x.remap_tensors(map, rng),
            AnyMilliOp::CastLike(x) => x.remap_tensors(map, rng),
            AnyMilliOp::Transpose(x) => x.remap_tensors(map, rng),
            AnyMilliOp::Squeeze(x) => x.remap_tensors(map, rng),
            AnyMilliOp::Unsqueeze(x) => x.remap_tensors(map, rng),
            AnyMilliOp::Gather(x) => x.remap_tensors(map, rng),
            AnyMilliOp::GatherGrad(x) => x.remap_tensors(map, rng),
            AnyMilliOp::Concat(x) => x.remap_tensors(map, rng),
            AnyMilliOp::Split(x) => x.remap_tensors(map, rng),
            AnyMilliOp::Where(x) => x.remap_tensors(map, rng),
            AnyMilliOp::Range(x) => x.remap_tensors(map, rng),
            AnyMilliOp::Expand(x) => x.remap_tensors(map, rng),
            AnyMilliOp::SumTo(x) => x.remap_tensors(map, rng),
            AnyMilliOp::ArgMax(x) => x.remap_tensors(map, rng),
            AnyMilliOp::ArgMin(x) => x.remap_tensors(map, rng),
            AnyMilliOp::Resize(x) => x.remap_tensors(map, rng),
            AnyMilliOp::Conv(x) => x.remap_tensors(map, rng),
            AnyMilliOp::ConvInputGrad(x) => x.remap_tensors(map, rng),
            AnyMilliOp::ConvWeightGrad(x) => x.remap_tensors(map, rng),
            AnyMilliOp::ConvBiasGrad(x) => x.remap_tensors(map, rng),
            AnyMilliOp::Pad(x) => x.remap_tensors(map, rng),
            AnyMilliOp::TopK(x) => x.remap_tensors(map, rng),
            AnyMilliOp::RandomNormalLike(x) => x.remap_tensors(map, rng),
        }
    }
}

macro_rules! delegate {
    ($name:ident($($arg:ident: $ty:ty),*) -> $ret:ty) => {
        fn $name(&self, $($arg: $ty),*) -> $ret {
            match self {
                AnyMilliOp::Constant(x) => x.$name($($arg),*),
                AnyMilliOp::ConstantOfShape(x) => x.$name($($arg),*),
                AnyMilliOp::SimpleBinary(x) => x.$name($($arg),*),
                AnyMilliOp::MatMul(x) => x.$name($($arg),*),
                AnyMilliOp::Pow(x) => x.$name($($arg),*),
                AnyMilliOp::SimpleUnary(x) => x.$name($($arg),*),
                AnyMilliOp::ClampMin(x) => x.$name($($arg),*),
                AnyMilliOp::NonZero(x) => x.$name($($arg),*),
                AnyMilliOp::CumSum(x) => x.$name($($arg),*),
                AnyMilliOp::Shape(x) => x.$name($($arg),*),
                AnyMilliOp::Reshape(x) => x.$name($($arg),*),
                AnyMilliOp::Slice(x) => x.$name($($arg),*),
                AnyMilliOp::ReduceSum(x) => x.$name($($arg),*),
                AnyMilliOp::ReduceMin(x) => x.$name($($arg),*),
                AnyMilliOp::ReduceMax(x) => x.$name($($arg),*),
                AnyMilliOp::ReduceProd(x) => x.$name($($arg),*),
                AnyMilliOp::ReduceMean(x) => x.$name($($arg),*),
                AnyMilliOp::Cast(x) => x.$name($($arg),*),
                AnyMilliOp::CastLike(x) => x.$name($($arg),*),
                AnyMilliOp::Transpose(x) => x.$name($($arg),*),
                AnyMilliOp::Squeeze(x) => x.$name($($arg),*),
                AnyMilliOp::Unsqueeze(x) => x.$name($($arg),*),
                AnyMilliOp::Gather(x) => x.$name($($arg),*),
                AnyMilliOp::GatherGrad(x) => x.$name($($arg),*),
                AnyMilliOp::Concat(x) => x.$name($($arg),*),
                AnyMilliOp::Split(x) => x.$name($($arg),*),
                AnyMilliOp::Where(x) => x.$name($($arg),*),
                AnyMilliOp::Range(x) => x.$name($($arg),*),
                AnyMilliOp::Expand(x) => x.$name($($arg),*),
                AnyMilliOp::SumTo(x) => x.$name($($arg),*),
                AnyMilliOp::ArgMax(x) => x.$name($($arg),*),
                AnyMilliOp::ArgMin(x) => x.$name($($arg),*),
                AnyMilliOp::Resize(x) => x.$name($($arg),*),
                AnyMilliOp::Conv(x) => x.$name($($arg),*),
                AnyMilliOp::ConvInputGrad(x) => x.$name($($arg),*),
                AnyMilliOp::ConvWeightGrad(x) => x.$name($($arg),*),
                AnyMilliOp::ConvBiasGrad(x) => x.$name($($arg),*),
                AnyMilliOp::Pad(x) => x.$name($($arg),*),
                AnyMilliOp::TopK(x) => x.$name($($arg),*),
                AnyMilliOp::RandomNormalLike(x) => x.$name($($arg),*),
            }
        }
    }
}

impl MilliOp for AnyMilliOp {
    delegate!(eval(
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend
    ) -> Result<
        Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>,
        MilliOpGraphError,
    > );

    fn backward(
        &self,
        output_grads: &HashMap<GlobalId, GlobalId>,
        graph: &mut crate::milli_graph::MilliOpGraph,
        rng: &mut impl Rng,
    ) -> Option<HashMap<GlobalId, GlobalId>> {
        match self {
            AnyMilliOp::Constant(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::ConstantOfShape(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::SimpleBinary(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::MatMul(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::Pow(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::SimpleUnary(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::ClampMin(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::NonZero(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::CumSum(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::Shape(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::Reshape(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::Slice(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::ReduceSum(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::ReduceMin(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::ReduceMax(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::ReduceProd(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::ReduceMean(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::Cast(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::CastLike(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::Transpose(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::Squeeze(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::Unsqueeze(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::Gather(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::GatherGrad(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::Concat(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::Split(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::Where(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::Range(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::Expand(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::SumTo(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::ArgMax(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::ArgMin(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::Resize(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::Conv(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::ConvInputGrad(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::ConvWeightGrad(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::ConvBiasGrad(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::Pad(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::RandomNormalLike(x) => x.backward(output_grads, graph, rng),
            AnyMilliOp::TopK(x) => x.backward(output_grads, graph, rng),
        }
    }
}

impl Node for AnyMilliOp {
    type OpKind = String;
    delegate!(op_kind() -> String);
    delegate!(inputs() ->  Box<dyn Iterator<Item = GlobalId> + '_>);
    delegate!(outputs() -> Box<dyn Iterator<Item = GlobalId> + '_>);
    delegate!(global_id() -> GlobalId);
}

impl NodeMetadata for AnyMilliOp {
    // MilliOp nodes currently don't expose parameters via introspection
    // This can be expanded later by adding parameters() to MilliOp trait
}
