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

pub trait MilliOp {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId>;

    fn infer(
        &self,
        known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>,
        _symbolic_resolver: &mut SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<TensorInfo, MilliOpGraphError> {
        let mut resolved_inputs = HashMap::new();
        for input in self.get_inputs() {
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

        Ok(TensorInfo::from(self.eval(&resolved_inputs, backend)?))
    }
    //fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, _symbolic_resolver: &mut SymbolicResolver, backend: &mut EvalBackend) -> Result<TensorInfo, MilliOpGraphError>;
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        _backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError>;

    fn get_name(&self) -> String;
}

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
    Constant(MilliOpConstant),
    ConstantOfShape(MilliOpConstantOfShape),
    SimpleBinary(MilliOpSimpleBinary),
    MatMul(MilliOpMatMul),
    Pow(MilliOpPow),
    SimpleUnary(MilliOpSimpleUnary),
    ClampMin(MilliOpClampMin),
    NonZero(MilliOpNonZero),
    CumSum(MilliOpCumSum),
    Shape(MilliOpShape),
    Reshape(MilliOpReshape),
    Slice(MilliOpSlice),
    ReduceSum(MilliOpReduceSum),
    ReduceMin(MilliOpReduceMin),
    ReduceMax(MilliOpReduceMax),
    ReduceProd(MilliOpReduceProd),
    ReduceMean(MilliOpReduceMean),
    Cast(MilliOpCast),
    CastLike(MilliOpCastLike),
    Transpose(MilliOpTranspose),
    Squeeze(MilliOpSqueeze),
    Unsqueeze(MilliOpUnsqueeze),
    Gather(MilliOpGather),
    Concat(MilliOpConcat),
    Split(MilliOpSplit),
    Where(MilliOpWhere),
    Range(MilliOpRange),
    Expand(MilliOpExpand),
    ArgMax(MilliOpArgMax),
    ArgMin(MilliOpArgMin),
}

impl MilliOp for AnyMilliOp {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        match self {
            AnyMilliOp::Constant(x) => x.get_inputs(),
            AnyMilliOp::ConstantOfShape(x) => x.get_inputs(),
            AnyMilliOp::SimpleBinary(x) => x.get_inputs(),
            AnyMilliOp::Pow(x) => x.get_inputs(),
            AnyMilliOp::SimpleUnary(x) => <_ as MilliOp>::get_inputs(x),
            AnyMilliOp::MatMul(x) => x.get_inputs(),
            AnyMilliOp::ClampMin(x) => <_ as MilliOp>::get_inputs(x),
            AnyMilliOp::NonZero(x) => x.get_inputs(),
            AnyMilliOp::CumSum(x) => x.get_inputs(),
            AnyMilliOp::Shape(x) => x.get_inputs(),
            AnyMilliOp::Reshape(x) => x.get_inputs(),
            AnyMilliOp::Slice(x) => x.get_inputs(),
            AnyMilliOp::ReduceSum(x) => x.get_inputs(),
            AnyMilliOp::ReduceMin(x) => x.get_inputs(),
            AnyMilliOp::ReduceMax(x) => x.get_inputs(),
            AnyMilliOp::ReduceProd(x) => x.get_inputs(),
            AnyMilliOp::ReduceMean(x) => x.get_inputs(),
            AnyMilliOp::Cast(x) => x.get_inputs(),
            AnyMilliOp::CastLike(x) => x.get_inputs(),
            AnyMilliOp::Transpose(x) => x.get_inputs(),
            AnyMilliOp::Squeeze(x) => x.get_inputs(),
            AnyMilliOp::Unsqueeze(x) => x.get_inputs(),
            AnyMilliOp::Gather(x) => x.get_inputs(),
            AnyMilliOp::Concat(x) => x.get_inputs(),
            AnyMilliOp::Split(x) => x.get_inputs(),
            AnyMilliOp::Where(x) => x.get_inputs(),
            AnyMilliOp::Range(x) => x.get_inputs(),
            AnyMilliOp::Expand(x) => x.get_inputs(),
            AnyMilliOp::ArgMax(x) => x.get_inputs(),
            AnyMilliOp::ArgMin(x) => x.get_inputs(),
        }
    }

    fn infer(
        &self,
        known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>,
        symbolic_resolver: &mut SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<TensorInfo, MilliOpGraphError> {
        match self {
            AnyMilliOp::Constant(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::ConstantOfShape(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::SimpleBinary(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Pow(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::SimpleUnary(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::MatMul(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::ClampMin(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::NonZero(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::CumSum(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Shape(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Reshape(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Slice(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::ReduceSum(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::ReduceMin(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::ReduceMax(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::ReduceProd(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::ReduceMean(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Cast(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::CastLike(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Transpose(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Squeeze(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Unsqueeze(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Gather(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Concat(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Split(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Where(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Range(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Expand(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::ArgMax(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::ArgMin(x) => x.infer(known_inputs, symbolic_resolver, backend),
        }
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        match self {
            AnyMilliOp::Constant(x) => x.eval(inputs, backend),
            AnyMilliOp::ConstantOfShape(x) => x.eval(inputs, backend),
            AnyMilliOp::SimpleBinary(x) => x.eval(inputs, backend),
            AnyMilliOp::Pow(x) => x.eval(inputs, backend),
            AnyMilliOp::SimpleUnary(x) => <_ as MilliOp>::eval(x, inputs, backend),
            AnyMilliOp::MatMul(x) => x.eval(inputs, backend),
            AnyMilliOp::ClampMin(x) => <_ as MilliOp>::eval(x, inputs, backend),
            AnyMilliOp::NonZero(x) => x.eval(inputs, backend),
            AnyMilliOp::CumSum(x) => x.eval(inputs, backend),
            AnyMilliOp::Shape(x) => x.eval(inputs, backend),
            AnyMilliOp::Reshape(x) => x.eval(inputs, backend),
            AnyMilliOp::Slice(x) => x.eval(inputs, backend),
            AnyMilliOp::ReduceSum(x) => x.eval(inputs, backend),
            AnyMilliOp::ReduceMin(x) => x.eval(inputs, backend),
            AnyMilliOp::ReduceMax(x) => x.eval(inputs, backend),
            AnyMilliOp::ReduceProd(x) => x.eval(inputs, backend),
            AnyMilliOp::ReduceMean(x) => x.eval(inputs, backend),
            AnyMilliOp::Cast(x) => x.eval(inputs, backend),
            AnyMilliOp::CastLike(x) => x.eval(inputs, backend),
            AnyMilliOp::Transpose(x) => x.eval(inputs, backend),
            AnyMilliOp::Squeeze(x) => x.eval(inputs, backend),
            AnyMilliOp::Unsqueeze(x) => x.eval(inputs, backend),
            AnyMilliOp::Gather(x) => x.eval(inputs, backend),
            AnyMilliOp::Concat(x) => x.eval(inputs, backend),
            AnyMilliOp::Split(x) => x.eval(inputs, backend),
            AnyMilliOp::Where(x) => x.eval(inputs, backend),
            AnyMilliOp::Range(x) => x.eval(inputs, backend),
            AnyMilliOp::Expand(x) => x.eval(inputs, backend),
            AnyMilliOp::ArgMax(x) => x.eval(inputs, backend),
            AnyMilliOp::ArgMin(x) => x.eval(inputs, backend),
        }
    }

    fn get_name(&self) -> String {
        match self {
            AnyMilliOp::Constant(x) => x.get_name(),
            AnyMilliOp::ConstantOfShape(x) => x.get_name(),
            AnyMilliOp::SimpleBinary(x) => x.get_name(),
            AnyMilliOp::MatMul(x) => x.get_name(),
            AnyMilliOp::Pow(x) => x.get_name(),
            AnyMilliOp::SimpleUnary(x) => <_ as MilliOp>::get_name(x),
            AnyMilliOp::ClampMin(x) => <_ as MilliOp>::get_name(x),
            AnyMilliOp::NonZero(x) => <_ as MilliOp>::get_name(x),
            AnyMilliOp::CumSum(x) => x.get_name(),
            AnyMilliOp::Shape(x) => x.get_name(),
            AnyMilliOp::Reshape(x) => x.get_name(),
            AnyMilliOp::Slice(x) => x.get_name(),
            AnyMilliOp::ReduceSum(x) => x.get_name(),
            AnyMilliOp::ReduceMin(x) => x.get_name(),
            AnyMilliOp::ReduceMax(x) => x.get_name(),
            AnyMilliOp::ReduceProd(x) => x.get_name(),
            AnyMilliOp::ReduceMean(x) => x.get_name(),
            AnyMilliOp::Cast(x) => x.get_name(),
            AnyMilliOp::CastLike(x) => x.get_name(),
            AnyMilliOp::Transpose(x) => x.get_name(),
            AnyMilliOp::Squeeze(x) => x.get_name(),
            AnyMilliOp::Unsqueeze(x) => x.get_name(),
            AnyMilliOp::Gather(x) => x.get_name(),
            AnyMilliOp::Concat(x) => x.get_name(),
            AnyMilliOp::Split(x) => x.get_name(),
            AnyMilliOp::Where(x) => x.get_name(),
            AnyMilliOp::Range(x) => x.get_name(),
            AnyMilliOp::Expand(x) => x.get_name(),
            AnyMilliOp::ArgMax(x) => x.get_name(),
            AnyMilliOp::ArgMin(x) => x.get_name(),
        }
    }
}
