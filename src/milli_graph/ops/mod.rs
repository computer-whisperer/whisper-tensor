mod argmax;
mod argmin;
mod binary;
mod cast;
mod cast_like;
mod concat;
mod constant;
mod cumsum;
mod expand;
mod gather;
mod nonzero;
mod range;
mod reduce_max;
mod reduce_mean;
mod reduce_min;
mod reduce_prod;
mod reduce_sum;
mod reshape;
mod shape;
mod slice;
mod split;
mod squeeze;
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
pub use cumsum::*;
pub use expand::*;
pub use gather::*;
pub use nonzero::*;
pub use range::*;
pub use reduce_max::*;
pub use reduce_mean::*;
pub use reduce_min::*;
pub use reduce_prod::*;
pub use reduce_sum::*;
pub use reshape::*;
pub use shape::*;
pub use slice::*;
pub use split::*;
pub use squeeze::*;
pub use transpose::*;
pub use unary::*;
pub use unsqueeze::*;
pub use where_op::*;

use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::graph::Node;
use crate::milli_graph::{MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use crate::scalar_info::ScalarInfoTyped;
use crate::symbolic_scalar::{SymbolicResolver, SymbolicScalarTyped};
use crate::tensor_info::{TensorInfo, TensorInfoTypedRanked};
use crate::tensor_rank::DynRank;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MilliOpTensorIDOrLiteral {
    TensorID(MilliOpGraphTensorId),
    Literal(NDArrayNumericTensor<DynRank>),
}

pub type EvalResult = Result<
    Box<dyn Iterator<Item = (MilliOpGraphTensorId, NumericTensor<DynRank>)>>,
    MilliOpGraphError,
>;
pub trait MilliOp: Node<MilliOpGraphTensorId> {
    fn infer(
        &self,
        known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>,
        _symbolic_resolver: &mut SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (MilliOpGraphTensorId, TensorInfo)>>, MilliOpGraphError>
    {
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

        let collected: Vec<(MilliOpGraphTensorId, TensorInfo)> = self
            .eval(&resolved_inputs, backend)?
            .map(|(a, b)| (a, TensorInfo::from(b)))
            .collect();
        Ok(Box::new(collected.into_iter()))
    }
    //fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, _symbolic_resolver: &mut SymbolicResolver, backend: &mut EvalBackend) -> Result<TensorInfo, MilliOpGraphError>;
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        _backend: &mut EvalBackend,
    ) -> EvalResult;
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
    Concat(Concat),
    Split(Split),
    Where(Where),
    Range(Range),
    Expand(Expand),
    ArgMax(ArgMax),
    ArgMin(ArgMin),
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
                AnyMilliOp::Concat(x) => x.$name($($arg),*),
                AnyMilliOp::Split(x) => x.$name($($arg),*),
                AnyMilliOp::Where(x) => x.$name($($arg),*),
                AnyMilliOp::Range(x) => x.$name($($arg),*),
                AnyMilliOp::Expand(x) => x.$name($($arg),*),
                AnyMilliOp::ArgMax(x) => x.$name($($arg),*),
                AnyMilliOp::ArgMin(x) => x.$name($($arg),*),
            }
        }
    }
}

impl MilliOp for AnyMilliOp {
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<
        Box<dyn Iterator<Item = (MilliOpGraphTensorId, NumericTensor<DynRank>)>>,
        MilliOpGraphError,
    > {
        let collected: Vec<(MilliOpGraphTensorId, NumericTensor<DynRank>)> = match self {
            AnyMilliOp::ArgMax(x) => x.eval(inputs, backend)?.collect(),
            AnyMilliOp::ArgMin(x) => x.eval(inputs, backend)?.collect(),
            AnyMilliOp::ReduceSum(x) => x.eval(inputs, backend)?.collect(),
            AnyMilliOp::ReduceMin(x) => x.eval(inputs, backend)?.collect(),
            AnyMilliOp::ReduceMax(x) => x.eval(inputs, backend)?.collect(),
            AnyMilliOp::ReduceProd(x) => x.eval(inputs, backend)?.collect(),
            AnyMilliOp::ReduceMean(x) => x.eval(inputs, backend)?.collect(),
            AnyMilliOp::Transpose(x) => x.eval(inputs, backend)?.collect(),
            AnyMilliOp::Reshape(x) => x.eval(inputs, backend)?.collect(),
            AnyMilliOp::Squeeze(x) => x.eval(inputs, backend)?.collect(),
            AnyMilliOp::Unsqueeze(x) => x.eval(inputs, backend)?.collect(),
            AnyMilliOp::Cast(x) => x.eval(inputs, backend)?.collect(),
            AnyMilliOp::CastLike(x) => x.eval(inputs, backend)?.collect(),
            AnyMilliOp::Concat(x) => x.eval(inputs, backend)?.collect(),
            AnyMilliOp::Slice(x) => x.eval(inputs, backend)?.collect(),
            AnyMilliOp::Constant(x) => x.eval(inputs, backend)?.collect(),
            AnyMilliOp::ConstantOfShape(x) => x.eval(inputs, backend)?.collect(),
            AnyMilliOp::SimpleBinary(x) => x.eval(inputs, backend)?.collect(),
            AnyMilliOp::MatMul(x) => x.eval(inputs, backend)?.collect(),
            AnyMilliOp::Pow(x) => x.eval(inputs, backend)?.collect(),
            AnyMilliOp::Where(x) => x.eval(inputs, backend)?.collect(),
            AnyMilliOp::Expand(x) => x.eval(inputs, backend)?.collect(),
            AnyMilliOp::Gather(x) => x.eval(inputs, backend)?.collect(),
            AnyMilliOp::CumSum(x) => x.eval(inputs, backend)?.collect(),
            AnyMilliOp::Split(x) => x.eval(inputs, backend)?.collect(),
            AnyMilliOp::Shape(x) => x.eval(inputs, backend)?.collect(),
            AnyMilliOp::Range(x) => x.eval(inputs, backend)?.collect(),
            AnyMilliOp::SimpleUnary(x) => x.eval(inputs, backend)?.collect(),
            AnyMilliOp::ClampMin(x) => x.eval(inputs, backend)?.collect(),
            AnyMilliOp::NonZero(x) => x.eval(inputs, backend)?.collect(),
        };
        Ok(Box::new(collected.into_iter()))
    }
}

impl Node<MilliOpGraphTensorId> for AnyMilliOp {
    type OpKind = String;
    delegate!(op_kind() -> String);
    delegate!(inputs() ->  Box<dyn Iterator<Item = MilliOpGraphTensorId>>);
    delegate!(outputs() -> Box<dyn Iterator<Item = MilliOpGraphTensorId>>);
}
