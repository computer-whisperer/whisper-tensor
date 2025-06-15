use std::collections::HashMap;
use typenum::P1;
use crate::dtype::{DType, DTypeError};
use crate::eval_backend::EvalBackend;
use crate::ndarray_backend::conversions::NDArrayNumericTensorType;
use crate::ndarray_backend::NDArrayNumericTensor;
use crate::numeric_scalar::NumericScalar;
use crate::numeric_tensor::NumericTensor;
use crate::symbolic_graph::{TensorId};
use crate::symbolic_graph::scalar_info::{ScalarInfo, ScalarInfoTyped};
use crate::symbolic_graph::symbolic_scalar::{SymbolicResolver, SymbolicScalar, SymbolicScalarTyped};
use crate::symbolic_graph::tensor_info::{MinimalTensor, TensorInfo, TensorInfoError, TensorInfoRanked, TensorInfoShaped, TensorInfoTypedRanked, TensorInfoTypedShaped};
use crate::tensor_rank::DynRank;
use crate::TrigOp;

#[derive(Debug, thiserror::Error)]
pub enum MilliOpGraphError {
    #[error(transparent)]
    NumericTensorError(#[from] crate::numeric_tensor::NumericTensorError),
    #[error(transparent)]
    NDArrayNumericTensorError(#[from] crate::ndarray_backend::NDArrayNumericTensorError),
    #[error("Unimplemented milli operator: {0}")]
    UnimplementedOperatorError(String),
    #[error("Invalid input for operation {0}")]
    InvalidInput(String),
    #[error(transparent)]
    DTypeError(#[from] DTypeError),
    #[error(transparent)]
    TensorInfoError(#[from] TensorInfoError),
    #[error("Unable to do any type if inference")]
    UnableToInfer
}

#[derive(Debug, Clone, Copy, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub(crate) struct MilliOpGraphTensorId {
    inner: usize,
}

#[derive(Debug, Clone)]
pub(crate) enum MilliOpTensorIDOrLiteral {
    TensorID(MilliOpGraphTensorId),
    Literal(NumericTensor<DynRank>)
}

trait MilliOp {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId>;
    
    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, _symbolic_resolver: &mut SymbolicResolver, backend: &mut EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let mut resolved_inputs = HashMap::new();
        for input in self.get_inputs() {
            if let Some(tensor_info) = known_inputs.get(&input) {
                if let Some(tensor) = tensor_info.as_numeric() {
                    resolved_inputs.insert(input, tensor.clone());
                }
                else {
                    return Err(MilliOpGraphError::UnableToInfer);
                }
            }
            else {
                return Err(MilliOpGraphError::UnableToInfer);
            }
        }

        Ok(TensorInfo::from(self.eval(&resolved_inputs, backend)?))
    }
    //fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, _symbolic_resolver: &mut SymbolicResolver, backend: &mut EvalBackend) -> Result<TensorInfo, MilliOpGraphError>;
    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>, _backend: &mut EvalBackend) -> Result<NumericTensor<DynRank>, MilliOpGraphError>;
}

pub(crate) struct MilliOpConstant {
    data: NumericTensor<DynRank>
}

impl MilliOpConstant {
    pub(crate) fn new(a: NumericTensor<DynRank>) -> Self {
        Self { data: a }
    }

    pub(crate) fn new_scalar<T>(v: T) -> Self
    where
        T: NDArrayNumericTensorType
    {
        Self{
            data: NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![v], &vec![1]).unwrap().into()
        }
    }
}

impl MilliOp for MilliOpConstant {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![]}

    fn infer(&self, _known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, _symbolic_resolver: &mut SymbolicResolver, _backend: &mut EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        Ok(TensorInfo::from(self.data.clone()))
    }
    fn eval(&self, _inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>, _backend: &mut EvalBackend) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        Ok(self.data.clone())
    }
}

pub(crate) struct MilliOpRange {
    start: MilliOpGraphTensorId,
    end: MilliOpGraphTensorId,
    delta: MilliOpGraphTensorId
}

impl MilliOpRange {
    pub(crate) fn new(start: MilliOpGraphTensorId,
                      end: MilliOpGraphTensorId,
                      delta: MilliOpGraphTensorId) -> Self {
        Self { start, end, delta}
    }
}

impl MilliOp for MilliOpRange {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![]}

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, symbolic_resolver: &mut SymbolicResolver, backend: &mut EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let start = &known_inputs[&self.start].first_element();
        let end = &known_inputs[&self.end].first_element();
        let delta = &known_inputs[&self.delta].first_element();
        assert_eq!(start.dtype(), end.dtype());
        assert_eq!(start.dtype(), end.dtype());
        Ok(if let (
            ScalarInfo::Numeric(start),
            ScalarInfo::Numeric(end),
            ScalarInfo::Numeric(delta)) =
            (start, end, delta) {
            // We have enough info, so just resolve it
            TensorInfo::from(NumericTensor::<P1>::range(
                *start,
                *end,
                *delta,
                backend
            )?)
        } 
        else {
            TensorInfo::from(TensorInfoRanked::<DynRank>::new(
                ScalarInfo::Symbolic(SymbolicScalar::new(DType::I64, symbolic_resolver)),
                vec![ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver))],
                symbolic_resolver
            ))
        })
    }

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>, backend: &mut EvalBackend) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        Ok(NumericTensor::<P1>::range(
            inputs[&self.start].first_element(),
            inputs[&self.end].first_element(),
            inputs[&self.delta].first_element(),
            backend
        )?.to_dyn_rank())
    }
}

pub(crate) struct MilliOpExpand {
    input: MilliOpGraphTensorId,
    shape: MilliOpGraphTensorId
}

impl MilliOpExpand {
    pub(crate) fn new(input: MilliOpGraphTensorId, shape: MilliOpGraphTensorId) -> Self{
        Self {input, shape}
    }
}

impl MilliOp for MilliOpExpand {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.input, self.shape]
    }

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>, backend: &mut EvalBackend) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let shape: Vec<i64> = inputs[&self.shape].try_to_rank::<P1>()?.try_into()?;
        let shape_u = shape.iter().map(|x| *x as u64).collect::<Vec<u64>>();
        let mut x = inputs[&self.input].clone();
        while x.rank() < shape_u.len() {
            x = x.unsqueeze(0)?;
        }
        let shape_u = shape_u.iter().zip(x.shape().iter()).map(|(a, b)| std::cmp::max(*a, *b)).collect::<Vec<u64>>();
        Ok(x.expand(&shape_u)?)
    }
}

pub(crate) struct MilliOpConstantOfShape {
    value: NumericScalar,
    shape: MilliOpGraphTensorId
}

impl MilliOpConstantOfShape {
    pub(crate) fn new(value: NumericScalar, shape: MilliOpGraphTensorId) -> Self {
        Self { value, shape }
    }
}

impl MilliOp for MilliOpConstantOfShape {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![]}

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, symbolic_resolver: &mut SymbolicResolver, backend: &mut EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let input = &known_inputs[&self.shape];
        let input = input.try_to_rank::<P1>(symbolic_resolver)?.try_to_type::<u64>()?;

        match input {
            TensorInfoTypedRanked::Shaped(tensor) => {
                match tensor {
                    TensorInfoTypedShaped::Numeric(tensor) => {
                        let inputs = HashMap::from([(self.shape, tensor.to_dyn_rank().to_dyn_type())]);
                        Ok(TensorInfo::from(self.eval(&inputs, backend)?))
                    }
                    TensorInfoTypedShaped::Shaped(tensor) => {
                        let mut new_shape = vec![];
                        for i in 0..tensor.shape()[0] {
                            new_shape.push(tensor.get(&[i]).unwrap().clone());
                        }
                        Ok(TensorInfo::from(TensorInfoRanked::<DynRank>::new(ScalarInfo::Numeric(self.value), new_shape, symbolic_resolver)))
                    }
                }
            }
            TensorInfoTypedRanked::Ranked(tensor) => {
                Ok(TensorInfo::new_from_first_element_and_rank(ScalarInfo::Numeric(self.value), tensor.shape()[0].cast(), symbolic_resolver))
            }
        }
    }

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>, _backend: &mut EvalBackend) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let shape: Vec<i64> = inputs[&self.shape].try_to_rank::<P1>()?.try_into()?;
        let shape_usize = shape.iter().map(|x| *x as u64).collect::<Vec<_>>();
        Ok(NDArrayNumericTensor::<DynRank>::fill(self.value.clone(), &shape_usize)?.into())
    }
}

fn infer_multidirectional_broadcasting_rank(shapes: &[TensorInfoTypedRanked<u64, P1>], symbolic_resolver: &mut SymbolicResolver) -> Result<ScalarInfoTyped<u32>, MilliOpGraphError> {
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
        None => Ok(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver))),
    }
}


fn infer_multidirectional_broadcasting_shape(shapes: &[Vec<ScalarInfoTyped<u64>>], symbolic_resolver: &mut SymbolicResolver) -> Result<Vec<ScalarInfoTyped<u64>>, MilliOpGraphError> {
    if shapes.is_empty() {
        return Err(MilliOpGraphError::InvalidInput("Cannot broadcast empty input".to_string()));
    }

    let output_rank = shapes.into_iter().map(|x| x.len()).max().unwrap();

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
                        }
                        else {
                            match dim {
                                ScalarInfoTyped::Numeric(y) => {
                                    if y == 1 || x == y {
                                        dim = ScalarInfoTyped::Numeric(y.max(x));
                                    } else {
                                        return Err(MilliOpGraphError::InvalidInput("Cannot broadcast input shape".to_string()));
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
                                        dim = ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver))
                                    }
                                    Some(is_same) => {
                                        if is_same {
                                            // Ok, use the unknown dim already in there
                                        }
                                        else {
                                            // Must use new unknown dimension
                                            dim = ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver))
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

enum SimpleBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Modulo(Option<bool>),
    And,
    Or,
    Xor,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    Equal,
    Greater,
    GreaterOrEqual,
    Less,
    LessOrEqual,
    Max,
    Min
}

pub(crate) struct MilliOpSimpleBinary {
    which_op: SimpleBinaryOp,
    a: MilliOpGraphTensorId,
    b: MilliOpGraphTensorId
}


impl MilliOpSimpleBinary {
    pub(crate) fn new(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId, which_op: SimpleBinaryOp) -> Self {
        Self { a, b, which_op }
    }

    pub(crate) fn add(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self { a, b, which_op: SimpleBinaryOp::Add }
    }

    pub(crate) fn sub(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self { a, b, which_op: SimpleBinaryOp::Sub }
    }

    pub(crate) fn mul(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self { a, b, which_op: SimpleBinaryOp::Mul }
    }

    pub(crate) fn div(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self { a, b, which_op: SimpleBinaryOp::Div }
    }

    pub(crate) fn modulo(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId, fmod: Option<bool>) -> Self {
        Self { a, b, which_op: SimpleBinaryOp::Modulo(fmod) }
    }
    pub(crate) fn and(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self { a, b, which_op: SimpleBinaryOp::And }
    }

    pub(crate) fn or(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self { a, b, which_op: SimpleBinaryOp::Or }
    }

    pub(crate) fn xor(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self { a, b, which_op: SimpleBinaryOp::Xor }
    }

    pub(crate) fn bitwise_and(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self { a, b, which_op: SimpleBinaryOp::BitwiseAnd }
    }

    pub(crate) fn bitwise_or(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self { a, b, which_op: SimpleBinaryOp::BitwiseOr }
    }

    pub(crate) fn bitwise_xor(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self { a, b, which_op: SimpleBinaryOp::BitwiseXor }
    }

    pub(crate) fn equal(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self { a, b, which_op: SimpleBinaryOp::Equal }
    }

    pub(crate) fn greater(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self { a, b, which_op: SimpleBinaryOp::Greater }
    }

    pub(crate) fn greater_or_equal(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self { a, b, which_op: SimpleBinaryOp::GreaterOrEqual }
    }

    pub(crate) fn less(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self { a, b, which_op: SimpleBinaryOp::Less }
    }

    pub(crate) fn less_or_equal(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self { a, b, which_op: SimpleBinaryOp::LessOrEqual }
    }

    pub(crate) fn max(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self { a, b, which_op: SimpleBinaryOp::Max }
    }

    pub(crate) fn min(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self { a, b, which_op: SimpleBinaryOp::Min }
    }

}

impl MilliOp for MilliOpSimpleBinary {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.a.clone(), self.b.clone()]}

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, symbolic_resolver: &mut SymbolicResolver, backend: &mut EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let a = &known_inputs[&self.a];
        let b = &known_inputs[&self.b];

        let a_shape = a.shape(symbolic_resolver);
        let b_shape = b.shape(symbolic_resolver);
        let output_rank = infer_multidirectional_broadcasting_rank(&[a_shape.clone(), b_shape.clone()], symbolic_resolver)?;

        if let (Some(a), Some(b)) = (a.as_numeric(), b.as_numeric()) {
            let inputs = HashMap::from([(self.a.clone(), a.clone()), (self.b.clone(), b.clone())]);
            Ok(TensorInfo::from(self.eval(&inputs, backend)?))
        } else if let (Some(a), Some(b)) = (a.as_ranked(), b.as_ranked()) {
            let output_shape = infer_multidirectional_broadcasting_shape(&[a.shape(), b.shape()], symbolic_resolver)?;
            Ok(TensorInfo::from(TensorInfoRanked::<DynRank>::new(
                ScalarInfo::Symbolic(SymbolicScalar::new(a.dtype(), symbolic_resolver)),
                output_shape,
                symbolic_resolver
            )))
        }
        else {
            Ok(TensorInfo::new_from_first_element_and_rank(ScalarInfo::Symbolic(SymbolicScalar::new(a.dtype(), symbolic_resolver)), output_rank, symbolic_resolver))
        }

    }

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>, backend: &mut EvalBackend) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let a = &inputs[&self.a];
        let b = &inputs[&self.b];
        Ok(match self.which_op {
            SimpleBinaryOp::Add => NumericTensor::<DynRank>::add(a, b, backend)?,
            SimpleBinaryOp::Sub => NumericTensor::<DynRank>::sub(a, b, backend)?,
            SimpleBinaryOp::Mul => NumericTensor::<DynRank>::mul(a, b, backend)?,
            SimpleBinaryOp::Div => NumericTensor::<DynRank>::div(a, b, backend)?,
            SimpleBinaryOp::Modulo(fmod) => {
                let is_float = [DType::F64, DType::F32, DType::BF16, DType::F16].contains(&a.dtype());
                let fmod = if is_float {true} else {fmod.unwrap_or(false)};
                if fmod {
                    NumericTensor::<DynRank>::fmod(a, b, backend)?
                }
                else {
                    NumericTensor::<DynRank>::imod(a, b, backend)?
                }
            },
            SimpleBinaryOp::And => NumericTensor::<DynRank>::and(a, b, backend)?,
            SimpleBinaryOp::Or => NumericTensor::<DynRank>::or(a, b, backend)?,
            SimpleBinaryOp::Xor => NumericTensor::<DynRank>::xor(a, b, backend)?,
            SimpleBinaryOp::BitwiseAnd => NumericTensor::<DynRank>::bitwise_and(a, b, backend)?,
            SimpleBinaryOp::BitwiseOr => NumericTensor::<DynRank>::bitwise_or(a, b, backend)?,
            SimpleBinaryOp::BitwiseXor => NumericTensor::<DynRank>::bitwise_xor(a, b, backend)?,
            SimpleBinaryOp::Equal => NumericTensor::<DynRank>::equal(a, b, backend)?,
            SimpleBinaryOp::Greater => NumericTensor::<DynRank>::greater(a, b, backend)?,
            SimpleBinaryOp::GreaterOrEqual => NumericTensor::<DynRank>::greater_or_equal(a, b, backend)?,
            SimpleBinaryOp::Less => NumericTensor::<DynRank>::less(a, b, backend)?,
            SimpleBinaryOp::LessOrEqual => NumericTensor::<DynRank>::less_or_equal(a, b, backend)?,
            SimpleBinaryOp::Max => NumericTensor::<DynRank>::max(a, b, backend)?,
            SimpleBinaryOp::Min => NumericTensor::<DynRank>::min(a, b, backend)?,
        })
    }
}

pub(crate) struct MilliOpPow {
    a: MilliOpGraphTensorId,
    b: MilliOpGraphTensorId
}

impl MilliOpPow {
    pub(crate) fn new(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self { a, b }
    }
}

impl MilliOp for MilliOpPow {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.a, self.b]}

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, symbolic_resolver: &mut SymbolicResolver, backend: &mut EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let a = &known_inputs[&self.a];
        let b = &known_inputs[&self.b];

        let a_shape = a.shape(symbolic_resolver);
        let b_shape = b.shape(symbolic_resolver);
        let output_rank = infer_multidirectional_broadcasting_rank(&[a_shape.clone(), b_shape.clone()], symbolic_resolver)?;

        if let (Some(a), Some(b)) = (a.as_numeric(), b.as_numeric()) {
            let inputs = HashMap::from([(self.a.clone(), a.clone()), (self.b.clone(), b.clone())]);
            Ok(TensorInfo::from(self.eval(&inputs, backend)?))
        } else if let (Some(a), Some(b)) = (a.as_ranked(), b.as_ranked()) {
            let output_shape = infer_multidirectional_broadcasting_shape(&[a.shape(), b.shape()], symbolic_resolver)?;
            Ok(TensorInfo::from(TensorInfoRanked::<DynRank>::new(
                ScalarInfo::Symbolic(SymbolicScalar::new(a.dtype(), symbolic_resolver)),
                output_shape,
                symbolic_resolver
            )))
        }
        else {
            Ok(TensorInfo::new_from_first_element_and_rank(ScalarInfo::Symbolic(SymbolicScalar::new(a.dtype(), symbolic_resolver)), output_rank, symbolic_resolver))
        }
    }

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>, backend: &mut EvalBackend) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        Ok(NumericTensor::<DynRank>::pow(&inputs[&self.a], &inputs[&self.b], backend)?)
    }
}


pub(crate) struct MilliOpMatMul {
    a: MilliOpGraphTensorId,
    b: MilliOpGraphTensorId
}

impl MilliOpMatMul {
    pub(crate) fn new(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self { a, b }
    }
}

impl MilliOp for MilliOpMatMul {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.a, self.b]}

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, symbolic_resolver: &mut SymbolicResolver, backend: &mut EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let a = &known_inputs[&self.a];
        let b = &known_inputs[&self.b];
        if let (Some(a), Some(b)) = (a.as_numeric(), b.as_numeric()) {
            let inputs = HashMap::from([(self.a.clone(), a.clone()), (self.b.clone(), b.clone())]);
            Ok(TensorInfo::from(
                self.eval(&inputs, backend)?
            ))
        }
        else {
            let dtype = a.dtype();
            assert_eq!(b.dtype(), dtype);

            if let (Some(a), Some(b)) = (a.as_ranked(), b.as_ranked()) {
                let shape_a = a.shape().to_vec();
                let shape_b = b.shape().to_vec();
                // Prepend to a if rank 1
                let (mut shape_a, prune_first_after) = if shape_a.len() == 1 {
                    (vec![ScalarInfoTyped::Numeric(1), shape_a[0].clone()], true)
                } else {
                    (shape_a, false)
                };

                // Append to b if rank 1
                let (mut shape_b, prune_last_after) = if shape_b.len() == 1 {
                    (vec![shape_b[0].clone(), ScalarInfoTyped::Numeric(1)], true)
                } else {
                    (shape_b, false)
                };

                // Broadcast both shapes
                while shape_a.len() < shape_b.len() {
                    shape_a.insert(0, ScalarInfoTyped::Numeric(1))
                }
                while shape_b.len() < shape_a.len() {
                    shape_b.insert(0, ScalarInfoTyped::Numeric(1))
                }

                let mut dims_out = vec![];
                for i in 0..shape_a.len()-2 {
                    let dim = match shape_a[i].clone() {
                        ScalarInfoTyped::Numeric(x) => {
                            if x == 1 {
                                // Use the other one
                                shape_b[i].clone()
                            }
                            else {
                                match shape_b[i] {
                                    ScalarInfoTyped::Numeric(y) => {
                                        ScalarInfoTyped::Numeric(x.max(y))
                                    }
                                    _ => {
                                        // Assume it's the known one
                                        ScalarInfoTyped::Numeric(x)
                                    }
                                }
                            }
                        }
                        ScalarInfoTyped::Symbolic(x) => {
                            match shape_b[i].clone() {
                                ScalarInfoTyped::Numeric(y) => {
                                    // Assume it's the known one
                                    ScalarInfoTyped::Numeric(y)
                                }
                                ScalarInfoTyped::Symbolic(y) => {
                                    match x.try_eq(&y) {
                                        None => {
                                            // Can't compare them, must use a new unknown
                                            ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver))
                                        }
                                        Some(is_same) => {
                                            if is_same {
                                                // They are the same dimension, so we can use it
                                                ScalarInfoTyped::Symbolic(x)
                                            }
                                            else {
                                                // They are different dimensions, so we can't use it
                                                ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver))
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    };
                    dims_out.push(dim);
                }
                dims_out.push(shape_a[shape_a.len() - 2].clone());
                dims_out.push(shape_b[shape_b.len() - 1].clone());

                if prune_first_after {
                    dims_out.remove(0);
                }
                if prune_last_after {
                    dims_out.pop();
                }

                Ok(TensorInfo::from(TensorInfoRanked::<DynRank>::new(
                    ScalarInfo::Symbolic(SymbolicScalar::new(dtype, symbolic_resolver)),
                    dims_out,
                    symbolic_resolver
                )))
            } else {
                // One of the input ranks was unknown, must simply pass on the confusion
                Ok(TensorInfo::Minimal(MinimalTensor::new(
                    ScalarInfo::Symbolic(SymbolicScalar::new(dtype, symbolic_resolver)),
                    SymbolicScalarTyped::new(symbolic_resolver)
                )))
            }
        }
    }
    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>, backend: &mut EvalBackend) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        Ok(NumericTensor::<DynRank>::matmul(&inputs[&self.a], &inputs[&self.b], backend)?)
    }
}


trait SimpleUnaryMilliOp {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId>;

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>, backend: &mut EvalBackend) -> Result<NumericTensor<DynRank>, MilliOpGraphError>;

    fn eval_scalar(&self, input: &NumericScalar) -> Result<NumericScalar, MilliOpGraphError>;
}

impl<T: SimpleUnaryMilliOp> MilliOp for T {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        <T as SimpleUnaryMilliOp>::get_inputs(self)
    }

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, symbolic_resolver: &mut SymbolicResolver, backend: &mut EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let input_id = self.get_inputs()[0];
        let input = &known_inputs[&input_id];

        if let Some(input) = input.as_shaped() {
            match input {
                TensorInfoShaped::Numeric(input) => {
                    let inputs = HashMap::from([(input_id, input.clone())]);
                    Ok(TensorInfo::from(self.eval(&inputs, backend)?))
                }
                TensorInfoShaped::Symbolic(input) => {
                    // For now, just decay this to a ranked tensor
                    Ok(TensorInfo::from(
                        TensorInfoRanked::<DynRank>::new(
                            ScalarInfo::Symbolic(SymbolicScalar::new(input.dtype(), symbolic_resolver)),
                            input.shape().iter().map(|x| ScalarInfoTyped::Numeric(*x)).collect::<Vec<_>>(),
                            symbolic_resolver
                        )
                    ))
                }
            }
        } else {
            // For now, just decay this to a ranked tensor
            Ok(TensorInfo::new_from_first_element_and_shape(
                ScalarInfo::Symbolic(SymbolicScalar::new(input.dtype(), symbolic_resolver)),
                input.shape(symbolic_resolver),
                symbolic_resolver
            ))
        }
    }

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>, backend: &mut EvalBackend) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        <T as SimpleUnaryMilliOp>::eval(self, inputs, backend)
    }
}

pub(crate) enum SimpleUnaryOp {
    Neg,
    Abs,
    Exp,
    Ln,
    Sqrt,
    Not,
    Sign,
    BitwiseNot,
    Reciprocal,
    Trig(TrigOp),
    Floor,
    Ceil,
    Round,
    IsNan,
    IsInf { detect_positive: bool, detect_negative: bool },
    Erf,
}

pub(crate) struct MilliOpSimpleUnary {
    input: MilliOpGraphTensorId,
    op: SimpleUnaryOp
}

impl MilliOpSimpleUnary {
    pub(crate) fn new(input: MilliOpGraphTensorId, op: SimpleUnaryOp) -> Self {
        Self {input, op}
    }

    pub(crate) fn neg(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Neg)
    }

    pub(crate) fn abs(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Abs)
    }

    pub(crate) fn exp(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Exp)
    }

    pub(crate) fn ln(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Ln)
    }

    pub(crate) fn sqrt(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Sqrt)
    }

    pub(crate) fn not(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Not)
    }

    pub(crate) fn sign(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Sign)
    }

    pub(crate) fn bitwise_not(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::BitwiseNot)
    }

    pub(crate) fn reciprocal(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Reciprocal)
    }

    pub(crate) fn trig(input: MilliOpGraphTensorId, trig_op: TrigOp) -> Self {
        Self::new(input, SimpleUnaryOp::Trig(trig_op))
    }

    pub(crate) fn floor(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Floor)
    }

    pub(crate) fn ceil(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Ceil)
    }

    pub(crate) fn round(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Round)
    }

    pub(crate) fn is_inf(input: MilliOpGraphTensorId, detect_positive: bool, detect_negative: bool) -> Self {
        Self::new(input, SimpleUnaryOp::IsInf{detect_positive, detect_negative})
    }

    pub(crate) fn is_nan(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::IsNan)
    }

    pub(crate) fn erf(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Erf)
    }
}

impl SimpleUnaryMilliOp for MilliOpSimpleUnary {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.input]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>, backend: &mut EvalBackend) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let input = &inputs[&self.input];
        match self.op {
            SimpleUnaryOp::Neg => Ok(input.neg(backend)?),
            SimpleUnaryOp::Abs => Ok(input.abs(backend)?),
            SimpleUnaryOp::Exp => Ok(input.exp(backend)?),
            SimpleUnaryOp::Ln => Ok(input.ln(backend)?),
            SimpleUnaryOp::Sqrt => Ok(input.sqrt(backend)?),
            SimpleUnaryOp::Not => Ok(input.not(backend)?),
            SimpleUnaryOp::Sign => Ok(input.sign(backend)?),
            SimpleUnaryOp::BitwiseNot => Ok(input.bitwise_not(backend)?),
            SimpleUnaryOp::Reciprocal => Ok(input.reciprocal(backend)?),
            SimpleUnaryOp::Trig(trig_op) => Ok(input.trig(trig_op, backend)?),
            SimpleUnaryOp::Floor => Ok(input.floor(backend)?),
            SimpleUnaryOp::Ceil => Ok(input.ceil(backend)?),
            SimpleUnaryOp::Round => Ok(input.round(backend)?),
            SimpleUnaryOp::IsInf{detect_positive, detect_negative} => Ok(input.is_inf(detect_positive, detect_negative)?),
            SimpleUnaryOp::IsNan => Ok(input.is_nan(backend)?),
            SimpleUnaryOp::Erf => Ok(input.erf(backend)?),
        }
    }

    fn eval_scalar(&self, input: &NumericScalar) -> Result<NumericScalar, MilliOpGraphError> {
        match self.op {
            SimpleUnaryOp::Neg => Ok(input.neg()),
            SimpleUnaryOp::Abs => Ok(input.abs()),
            SimpleUnaryOp::Exp => Ok(input.exp()),
            SimpleUnaryOp::Ln => Ok(input.ln()),
            SimpleUnaryOp::Sqrt => Ok(input.sqrt()),
            SimpleUnaryOp::Not => Ok(input.not()),
            SimpleUnaryOp::Sign => Ok(input.sign()),
            SimpleUnaryOp::BitwiseNot => Ok(input.bitwise_not()),
            SimpleUnaryOp::Reciprocal => Ok(input.recip()),
            SimpleUnaryOp::Trig(trig_op) => Ok(input.trig(trig_op)),
            SimpleUnaryOp::Floor => Ok(input.floor()),
            SimpleUnaryOp::Ceil => Ok(input.ceil()),
            SimpleUnaryOp::Round => Ok(input.round()),
            SimpleUnaryOp::IsInf{detect_positive, detect_negative} => Ok(input.is_inf(detect_positive, detect_negative)),
            SimpleUnaryOp::IsNan => Ok(input.is_nan()),
            SimpleUnaryOp::Erf => Ok(input.erf()),
        }
    }
}

pub(crate) struct MilliOpClampMin {
    input: MilliOpGraphTensorId,
    value: f32
}

impl MilliOpClampMin {
    pub(crate) fn new(a: MilliOpGraphTensorId, value: f32) -> Self {
        Self { input: a, value}
    }
}

impl SimpleUnaryMilliOp for MilliOpClampMin {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.input]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>, backend: &mut EvalBackend) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        Ok(inputs[&self.input].clamp_min(self.value, backend)?)
    }

    fn eval_scalar(&self, input: &NumericScalar) -> Result<NumericScalar, MilliOpGraphError> {
        Ok(input.clamp_min(self.value))
    }
}

pub(crate) struct MilliOpNonZero {
    input: MilliOpGraphTensorId,
}

impl MilliOpNonZero {
    pub(crate) fn new(input: MilliOpGraphTensorId) -> Self {
        Self { input}
    }
}

impl MilliOp for MilliOpNonZero {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.input]}

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, symbolic_resolver: &mut SymbolicResolver, backend: &mut EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let input = &known_inputs[&self.input];
        if let Some(x) = input.as_numeric() {
            let inputs = HashMap::from([(self.input.clone(), x.clone())]);
            Ok(TensorInfo::from(
                self.eval(&inputs, backend)?
            ))
        }
        else {
            // Don't even try shape inference for now
            Ok(TensorInfo::Minimal(MinimalTensor::new(
                ScalarInfo::Symbolic(SymbolicScalar::new(DType::I64, symbolic_resolver)),
                SymbolicScalarTyped::new(symbolic_resolver)
            )))
        }
    }

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>, backend: &mut EvalBackend) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        Ok(inputs[&self.input].nonzero(backend)?)
    }
}

pub(crate) struct MilliOpCumSum {
    a: MilliOpGraphTensorId,
    axis: MilliOpGraphTensorId,
    exclusive: bool,
    reverse: bool,
}

impl MilliOpCumSum {
    pub(crate) fn new(a: MilliOpGraphTensorId, axis: MilliOpGraphTensorId, exclusive: bool, reverse: bool) -> Self {
        Self {a, axis, exclusive, reverse}
    }
}

impl MilliOp for MilliOpCumSum {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.a, self.axis]}

    fn eval(&self, _inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>, _backend: &mut EvalBackend) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        Err(MilliOpGraphError::UnimplementedOperatorError("CumSum".to_string()))?;
        todo!()
    }
}

pub(crate) struct MilliOpShape {
    input: MilliOpGraphTensorId
}

impl MilliOpShape {
    pub(crate) fn new(input: MilliOpGraphTensorId) -> Self {
        Self { input }
    }
}

impl MilliOp for MilliOpShape {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.input]}

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, symbolic_resolver: &mut SymbolicResolver, _backend: &mut EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let input = &known_inputs[&self.input];
        Ok(TensorInfo::from(input.shape(symbolic_resolver)))
    }

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>, _backend: &mut EvalBackend) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let output_shape = inputs[&self.input].shape().into_iter().map(|x| x as i64).collect::<Vec<_>>();
        Ok(NDArrayNumericTensor::<P1>::from(output_shape).to_dyn().into())
    }
}

pub(crate) struct MilliOpReduceSum {
    data: MilliOpGraphTensorId,
    axes: Option<MilliOpGraphTensorId>,
    keepdims: bool,
    noop_with_empty_axes: bool
}

impl MilliOpReduceSum {
    pub(crate) fn new(data: MilliOpGraphTensorId, axes: Option<MilliOpGraphTensorId>, keepdims: bool, noop_with_empty_axes: bool) -> Self {
        Self { data, axes, keepdims, noop_with_empty_axes}
    }
}

impl MilliOp for MilliOpReduceSum {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.data]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>, backend: &mut EvalBackend) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let data = &inputs[&self.data];
        let axes = if let Some(axes) = self.axes {
            Vec::<i64>::try_from(inputs[&axes].try_to_rank::<P1>()?)?
        } else {
            (0i64 .. (data.shape().len() as i64)).into_iter().collect()
        };
        let axes = if axes.len() == 0 {
            if self.noop_with_empty_axes {
                return Ok(data.clone())
            } else {
                (0i64 .. (data.shape().len() as i64)).into_iter().collect::<Vec<_>>()
            }
        } else {
            axes
        };
        let axes = axes.into_iter().map(|x| (if x < 0 {x + data.shape().len() as i64} else {x}) as usize).collect::<Vec<_>>();
        let out = data.reduce_sum(axes, self.keepdims, backend)?;
        Ok(out)
    }
}

pub(crate) struct MilliOpReduceMin {
    data: MilliOpGraphTensorId,
    axes: Option<MilliOpGraphTensorId>,
    keepdims: bool,
    noop_with_empty_axes: bool
}

impl MilliOpReduceMin {
    pub(crate) fn new(data: MilliOpGraphTensorId, axes: Option<MilliOpGraphTensorId>, keepdims: bool, noop_with_empty_axes: bool) -> Self {
        Self { data, axes, keepdims, noop_with_empty_axes}
    }
}

impl MilliOp for MilliOpReduceMin {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.data]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>, backend: &mut EvalBackend) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let data = &inputs[&self.data];
        let axes = if let Some(axes) = self.axes {
            Vec::<i64>::try_from(inputs[&axes].try_to_rank::<P1>()?)?
        } else {
            (0i64 .. (data.shape().len() as i64)).into_iter().collect()
        };
        let axes = if axes.len() == 0 {
            if self.noop_with_empty_axes {
                return Ok(data.clone())
            } else {
                (0i64 .. (data.shape().len() as i64)).into_iter().collect::<Vec<_>>()
            }
        } else {
            axes
        };
        let axes = axes.into_iter().map(|x| (if x < 0 {x + data.shape().len() as i64} else {x}) as usize).collect::<Vec<_>>();
        let out = data.reduce_min(axes, self.keepdims, backend)?;
        Ok(out)
    }
}


pub(crate) struct MilliOpReduceMax {
    data: MilliOpGraphTensorId,
    axes: Option<MilliOpGraphTensorId>,
    keepdims: bool,
    noop_with_empty_axes: bool
}

impl MilliOpReduceMax {
    pub(crate) fn new(data: MilliOpGraphTensorId, axes: Option<MilliOpGraphTensorId>, keepdims: bool, noop_with_empty_axes: bool) -> Self {
        Self { data, axes, keepdims, noop_with_empty_axes}
    }
}

impl MilliOp for MilliOpReduceMax {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.data]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>, backend: &mut EvalBackend) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let data = &inputs[&self.data];
        let axes = if let Some(axes) = self.axes {
            Vec::<i64>::try_from(inputs[&axes].try_to_rank::<P1>()?)?
        } else {
            (0i64 .. (data.shape().len() as i64)).into_iter().collect()
        };
        let axes = if axes.len() == 0 {
            if self.noop_with_empty_axes {
                return Ok(data.clone())
            } else {
                (0i64 .. (data.shape().len() as i64)).into_iter().collect::<Vec<_>>()
            }
        } else {
            axes
        };
        let axes = axes.into_iter().map(|x| (if x < 0 {x + data.shape().len() as i64} else {x}) as usize).collect::<Vec<_>>();
        let out = data.reduce_max(axes, self.keepdims, backend)?;
        Ok(out)
    }
}

pub(crate) struct MilliOpReduceProd {
    data: MilliOpGraphTensorId,
    axes: Option<MilliOpGraphTensorId>,
    keepdims: bool,
    noop_with_empty_axes: bool
}

impl MilliOpReduceProd {
    pub(crate) fn new(data: MilliOpGraphTensorId, axes: Option<MilliOpGraphTensorId>, keepdims: bool, noop_with_empty_axes: bool) -> Self {
        Self { data, axes, keepdims, noop_with_empty_axes}
    }
}

impl MilliOp for MilliOpReduceProd {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.data]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>, backend: &mut EvalBackend) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let data = &inputs[&self.data];
        let axes = if let Some(axes) = self.axes {
            Vec::<i64>::try_from(inputs[&axes].try_to_rank::<P1>()?)?
        } else {
            (0i64 .. (data.shape().len() as i64)).into_iter().collect()
        };
        let axes = if axes.len() == 0 {
            if self.noop_with_empty_axes {
                return Ok(data.clone())
            } else {
                (0i64 .. (data.shape().len() as i64)).into_iter().collect::<Vec<_>>()
            }
        } else {
            axes
        };
        let axes = axes.into_iter().map(|x| (if x < 0 {x + data.shape().len() as i64} else {x}) as usize).collect::<Vec<_>>();
        let out = data.reduce_prod(axes, self.keepdims, backend)?;
        Ok(out)
    }
}


pub(crate) struct MilliOpReduceMean {
    data: MilliOpGraphTensorId,
    axes: Option<MilliOpGraphTensorId>,
    keepdims: bool,
    noop_with_empty_axes: bool
}

impl MilliOpReduceMean {
    pub(crate) fn new(data: MilliOpGraphTensorId, axes: Option<MilliOpGraphTensorId>, keepdims: bool, noop_with_empty_axes: bool) -> Self {
        Self { data, axes, keepdims, noop_with_empty_axes}
    }
}

impl MilliOp for MilliOpReduceMean {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.data]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>, backend: &mut EvalBackend) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let data = &inputs[&self.data];
        let axes = if let Some(axes) = self.axes {
            Vec::<i64>::try_from(inputs[&axes].try_to_rank::<P1>()?)?
        } else {
            (0i64 .. (data.rank() as i64)).into_iter().collect()
        };
        let axes = if axes.len() == 0 {
            if self.noop_with_empty_axes {
                return Ok(data.clone())
            } else {
                (0i64 .. (data.rank() as i64)).into_iter().collect::<Vec<_>>()
            }
        } else {
            axes
        };
        let axes = axes.into_iter().map(|x| (if x < 0 {x + data.rank() as i64} else {x}) as usize).collect::<Vec<_>>();
        let out = data.reduce_mean(axes, self.keepdims, backend)?;
        Ok(out)
    }
}

pub(crate) struct MilliOpSlice {
    data: MilliOpGraphTensorId,
    starts: MilliOpGraphTensorId,
    ends: MilliOpGraphTensorId,
    steps: Option<MilliOpGraphTensorId>,
    axes: Option<MilliOpGraphTensorId>
}

impl MilliOpSlice {
    pub(crate) fn new(    data: MilliOpGraphTensorId,
                          starts: MilliOpGraphTensorId,
                          ends: MilliOpGraphTensorId,
                          steps: Option<MilliOpGraphTensorId>,
                          axes: Option<MilliOpGraphTensorId>) -> Self {
        Self { data, starts, ends, steps, axes}
    }
}

impl MilliOp for MilliOpSlice {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        let mut res = vec![self.data, self.starts, self.ends];
        if let Some(steps) = &self.steps {
            res.push(*steps);
        }
        if let Some(axes) = &self.axes {
            res.push(*axes);
        }
        res
    }

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>, backend: &mut EvalBackend) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let data_input = &inputs[&self.data];
        let input_shape = data_input.shape();
        let input_rank = data_input.rank();
        let axes: Vec<i64> = if let Some(axes) = &self.axes {
            inputs[axes].cast(DType::I64, backend)?.try_to_rank::<P1>()?.try_into()?
        } else {
            (0i64..(input_rank as i64)).into_iter().collect()
        };
        let steps: Vec<i64> = if let Some(steps) = &self.steps {
            inputs[steps].cast(DType::I64, backend)?.try_to_rank::<P1>()?.try_into()?
        } else {
            axes.iter().map(|_| 1).collect()
        };
        let starts: Vec<i64> = inputs[&self.starts].cast(DType::I64, backend)?.try_to_rank::<P1>()?.try_into()?;
        let ends: Vec<i64> = inputs[&self.ends].cast(DType::I64, backend)?.try_to_rank::<P1>()?.try_into()?;
        let mut output_slice = vec![];
        for i in 0..input_rank {
            output_slice.push(0..input_shape[i]);
        }
        for (i, axis) in axes.into_iter().enumerate() {
            let axis = if axis < 0 {
                (input_rank as i64 + axis) as usize
            } else {
                axis as usize
            };
            let step = steps[i];
            if step != 1 {
                return Err(MilliOpGraphError::InvalidInput(format!("Step {} is not supported", step)));
            }

            let start = (if starts[i] < 0 {
                input_shape[axis] as i64 + starts[i]
            } else {
                starts[i]
            }).min(input_shape[axis] as i64).max(0) as u64;

            let end = (if ends[i] < 0 {
                input_shape[axis] as i64 + ends[i]
            } else {
                ends[i]
            }).min(input_shape[axis] as i64).max(0) as u64;
            output_slice[axis] = start..end;
        }
        let output = data_input.slice(&output_slice, backend)?;
        Ok(output)
    }
}

pub(crate) struct MilliOpReshape {
    data: MilliOpGraphTensorId,
    shape: MilliOpGraphTensorId,
    allowzero: bool,
}

impl MilliOpReshape {
    pub(crate) fn new(    data: MilliOpGraphTensorId,
                          shape: MilliOpGraphTensorId,
                          allowzero: bool) -> Self {
        Self {
            data,
            shape,
            allowzero
        }
    }

    fn calculate_new_shape(&self, data_input_shape: &Vec<u64>, shape_input_value: &Vec<i64>) -> Result<Vec<u64>, MilliOpGraphError> {
        let mut new_shape_dims = vec![];
        let mut backfill_dim: Option<usize> = None;
        for i in 0..shape_input_value.len() {
            new_shape_dims.push(if shape_input_value[i] == 0 {
                data_input_shape[i].clone()
            } else if shape_input_value[i] == -1 {
                if backfill_dim.is_some() {
                    // Only one dimension can be inferred
                    Err(MilliOpGraphError::InvalidInput("Reshape".to_string()))?
                }
                backfill_dim = Some(i);
                1
            }
            else if shape_input_value[i] < -1 {
                Err(MilliOpGraphError::InvalidInput("Reshape".to_string()))?
            } else {
                shape_input_value[i] as u64
            });
        }

        // Backfill the inferred dimension
        if let Some(i) = backfill_dim {
            let total_input_size = data_input_shape.iter().product::<u64>();

            // Calculate the current product of the dimensions
            let mut current_product = 1;
            for (j, dim) in new_shape_dims.iter().enumerate() {
                if j != i {
                    current_product *= dim;
                }
            }
            // Calculate the inferred dimension size
            let inferred_size = total_input_size / current_product;
            new_shape_dims[i] = inferred_size;
        }
        let output_shape = new_shape_dims;

        // Verify that the dimensions are compatible
        if output_shape.iter().product::<u64>() != data_input_shape.iter().product::<u64>() {
            Err(MilliOpGraphError::InvalidInput("Reshape".to_string()))?
        }
        
        Ok(output_shape)
    }
}

impl MilliOp for MilliOpReshape {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.data, self.shape]
    }

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, symbolic_resolver: &mut SymbolicResolver, backend: &mut EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let data_input = &known_inputs[&self.data];
        let shape_input = known_inputs[&self.shape].try_to_rank::<P1>(symbolic_resolver)?.try_to_type::<i64>()?;

        if let Some(shape) = shape_input.as_numeric() {
            if let Some(data) = data_input.as_shaped() {
                match data {
                    TensorInfoShaped::Numeric(data) => {
                        let inputs = HashMap::from([(self.shape, shape.to_dyn_rank().to_dyn_type()), (self.data, data.clone())]);
                        Ok(TensorInfo::from(self.eval(&inputs, backend)?))
                    }
                    TensorInfoShaped::Symbolic(data) => {
                        let new_shape = self.calculate_new_shape(data.shape(), &shape.to_vec())?;
                        Ok(TensorInfo::from(data.reshape(new_shape)))
                    }
                }
            }
            else {
                let mut new_shape = vec![];
                for (i, dim) in shape.to_vec().into_iter().enumerate() {
                    if dim > 0 {
                        new_shape.push(ScalarInfoTyped::Numeric(dim as u64));
                    }
                    else if dim == 0 {
                        new_shape.push(data_input.shape(symbolic_resolver).get(&[i as u64], symbolic_resolver).unwrap());
                    }
                    else {
                        new_shape.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)));
                    }
                }
                Ok(TensorInfo::from(TensorInfoRanked::<DynRank>::new(data_input.first_element(), new_shape, symbolic_resolver)))
            }
        }
        else if let Some(shape) = shape_input.as_shaped() {
            let output_rank = shape.shape()[0];
            let mut new_shape = vec![];
            for i in 0..output_rank {
                let dim = shape.get(&[i]).unwrap();
                match dim {
                    ScalarInfoTyped::Numeric(dim) => {
                        if dim > 0 {
                            new_shape.push(ScalarInfoTyped::Numeric(dim as u64));
                        }
                        else if dim == 0 {
                            new_shape.push(data_input.shape(symbolic_resolver).get(&[i as u64], symbolic_resolver).unwrap());
                        }
                        else {
                            new_shape.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)));
                        }
                    }
                    ScalarInfoTyped::Symbolic(_x) => {
                        // Could be negative or zero, so have to use new symbol
                        new_shape.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)));
                    }
                }
            }
            Ok(TensorInfo::from(TensorInfoRanked::<DynRank>::new(data_input.first_element(), new_shape, symbolic_resolver)))
        }
        else {
            // We don't even know the rank of the output
            Ok(TensorInfo::new_from_first_element_and_rank(data_input.first_element(), shape_input.shape()[0].cast(), symbolic_resolver))
        }
    }


    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>, backend: &mut EvalBackend) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let data_input = &inputs[&self.data];
        let shape_input = &inputs[&self.shape];
        let shape_input_value: Vec<i64> = shape_input.cast(DType::I64, backend)?.try_to_rank::<P1>()?.try_into()?;

        let output_shape = self.calculate_new_shape(
            &data_input.shape(),
            &shape_input_value
        )?;
        
        let output_value = data_input.reshape(output_shape, backend)?;

        Ok(output_value)
    }
}

pub(crate) struct MilliOpSqueeze {
    data: MilliOpGraphTensorId,
    axes: MilliOpGraphTensorId,
}

impl MilliOpSqueeze {
    pub(crate) fn new(    data: MilliOpGraphTensorId,
                          axes: MilliOpGraphTensorId) -> Self {
        Self {
            data,
            axes
        }
    }
}

impl MilliOp for MilliOpSqueeze {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.data, self.axes]}

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, symbolic_resolver: &mut SymbolicResolver, backend: &mut EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let data_input = &known_inputs[&self.data];
        let axes_input = &known_inputs[&self.axes].try_to_rank::<P1>(symbolic_resolver)?.try_to_type::<i64>()?;

        if let Some(axes) = axes_input.as_numeric() {
            if let Some(data) = data_input.as_numeric() {
                let inputs = HashMap::from([(self.data, data.clone()), (self.axes, axes.to_dyn_type().to_dyn_rank())]);
                Ok(TensorInfo::from(self.eval(&inputs, backend)?))
            }
            else {
                let axes = axes.to_vec();
                if let Some(data) = data_input.as_ranked() {
                    let mut new_shape = vec![];
                    for i in 0..data.rank() {
                        let mut found = false;
                        for axis in &axes {
                            if axis >= &0 && i == *axis as usize {
                                // Skip dim
                                found = true;
                                break;
                            }
                            else if axis < &0 && i == (data.rank() as i64 + axis) as usize {
                                // Skip dim
                                found = true;
                                break;
                            }
                            else {
                                // keep dim
                            }
                        }
                        if !found {
                            new_shape.push(data.shape()[i].clone());
                        }
                    }
                    Ok(TensorInfo::from(data.reshape(new_shape, symbolic_resolver, backend)?))
                } else {
                    // Data has no defined rank, just decrease the rank by the number of axes.
                    let new_rank = data_input.rank().add_offset(-(axes.len() as i64));
                    Ok(TensorInfo::new_from_first_element_and_rank(data_input.first_element(), new_rank, symbolic_resolver))
                }
            }
        }
        else if let Some(axes) = axes_input.as_shaped() {
            // Data has no defined rank, just decrease the rank by the number of axes.
            let new_rank = data_input.rank().add_offset(-(axes.shape()[0] as i64));
            Ok(TensorInfo::new_from_first_element_and_rank(data_input.first_element(), new_rank, symbolic_resolver))
        }
        else {
            // Can't infer the output shape at all
            Ok(TensorInfo::new_from_first_element_and_rank(data_input.first_element(), ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)), symbolic_resolver))
        }
    }

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>, backend: &mut EvalBackend) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let axes_ndarray = NDArrayNumericTensor::<DynRank>::try_from(inputs[&self.axes].cast(DType::I64, backend)?)?;
        let axes = Vec::<i64>::try_from(axes_ndarray.try_to_rank::<P1>()?)?;
        if axes.len() == 1 {
            let axis = axes[0];
            let input_shape = inputs[&self.data].shape();
            let axis = if axis >= 0 {
                axis as usize
            } else {
                (input_shape.len() as i64 + axis) as usize
            };
            let output = inputs[&self.data].squeeze(axis)?;
            Ok(output)
        } else {
            // Multiple axes (use reshape)
            let input_shape = inputs[&self.data].shape();
            let mut output_shape = Vec::new();
            for i in 0..(input_shape.len() - axes.len()) {
                let mut is_selected = false;
                for axis in &axes {
                    let axis = if *axis < 0 {
                        input_shape.len() as i64 + *axis
                    } else {
                        *axis
                    };
                    if axis == i as i64 {
                        is_selected = true;
                        break;
                    }
                }
                if is_selected {
                    // Skip it
                } else {
                    output_shape.push(input_shape[i]);
                }
            }
            let output = inputs[&self.data].reshape(output_shape, backend)?;
            Ok(output)
        }

    }
}

pub(crate) struct MilliOpUnsqueeze {
    data: MilliOpGraphTensorId,
    axes: MilliOpGraphTensorId,
}

impl MilliOpUnsqueeze {
    pub(crate) fn new(    data: MilliOpGraphTensorId,
                          axes: MilliOpGraphTensorId) -> Self {
        Self {
            data,
            axes
        }
    }
}

impl MilliOp for MilliOpUnsqueeze {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.data, self.axes]}

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, symbolic_resolver: &mut SymbolicResolver, backend: &mut EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let data_input = &known_inputs[&self.data];
        let axes_input = &known_inputs[&self.axes].try_to_rank::<P1>(symbolic_resolver)?.try_to_type::<i64>()?;

        if let Some(axes) = axes_input.as_numeric() {
            if let Some(data) = data_input.as_numeric() {
                let inputs = HashMap::from([(self.data, data.clone()), (self.axes, axes.to_dyn_type().to_dyn_rank())]);
                Ok(TensorInfo::from(self.eval(&inputs, backend)?))
            }
            else {
                let axes = axes.to_vec();
                if let Some(data) = data_input.as_ranked() {
                    let mut new_shape = vec![];
                    let new_rank = data.rank() + axes.len();
                    let mut input_i = 0;
                    for i in 0..new_rank {
                        let mut found = false;
                        for axis in &axes {
                            if axis >= &0 && i == *axis as usize {
                                // Skip dim
                                found = true;
                                break;
                            }
                            else if axis < &0 && i == (new_rank as i64 + axis) as usize {
                                // Skip dim
                                found = true;
                                break;
                            }
                            else {
                                // keep dim
                            }
                        }
                        if found {
                            new_shape.push(ScalarInfoTyped::Numeric(1));
                        }
                        else {
                            new_shape.push(data.shape()[input_i].clone());
                            input_i += 1;
                        }
                    }
                    Ok(TensorInfo::from(data.reshape(new_shape, symbolic_resolver, backend)?))
                } else {
                    // Data has no defined rank, just decrease the rank by the number of axes.
                    let new_rank = data_input.rank().add_offset(axes.len() as i64);
                    Ok(TensorInfo::new_from_first_element_and_rank(data_input.first_element(), new_rank, symbolic_resolver))
                }
            }
        }
        else if let Some(axes) = axes_input.as_shaped() {
            // Data has no defined rank, just decrease the rank by the number of axes.
            let new_rank = data_input.rank().add_offset(axes.shape()[0] as i64);
            Ok(TensorInfo::new_from_first_element_and_rank(data_input.first_element(), new_rank, symbolic_resolver))
        }
        else {
            // Can't infer the output shape at all
            Ok(TensorInfo::new_from_first_element_and_rank(data_input.first_element(), ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)), symbolic_resolver))
        }
    }

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>, backend: &mut EvalBackend) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let axes_ndarray = NDArrayNumericTensor::<DynRank>::try_from(inputs[&self.axes].cast(DType::I64, backend)?)?;
        let axes = Vec::<i64>::try_from(axes_ndarray.try_to_rank::<P1>()?)?;
        if axes.len() == 1 {
            let axis = axes[0];
            let input_shape = inputs[&self.data].shape();
            let axis = if axis >= 0 {
                axis as usize
            } else {
                (input_shape.len() as i64 + axis) as usize
            };
            let output = inputs[&self.data].unsqueeze(axis)?;
            Ok(output)
        }
        else {
            // Multiple axes (use reshape)
            let input_shape = inputs[&self.data].shape();
            let mut output_shape = Vec::new();
            let mut input_index = 0;
            for i in 0..(input_shape.len() + axes.len()) {
                let mut is_selected = false;
                for axis in &axes {
                    let axis = if *axis < 0 {
                        input_shape.len() as i64 + *axis
                    } else {
                        *axis
                    };
                    if axis == i as i64 {
                        is_selected = true;
                        break;
                    }
                }
                if is_selected {
                    output_shape.push(1);
                } else {
                    output_shape.push(input_shape[input_index]);
                    input_index += 1;
                }
            }
            let output = inputs[&self.data].reshape(output_shape, backend)?;
            Ok(output)

        }
    }
}


pub(crate) struct MilliOpCast {
    data: MilliOpGraphTensorId,
    dtype: DType
}

impl MilliOpCast {
    pub(crate) fn new(data: MilliOpGraphTensorId, dtype: DType) -> Self {
        Self {data, dtype}
    }
}

impl MilliOp for MilliOpCast {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.data]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>, backend: &mut EvalBackend) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        Ok(inputs[&self.data].cast(self.dtype, backend)?)
    }
}

pub(crate) struct MilliOpCastLike {
    data: MilliOpGraphTensorId,
    target_type: MilliOpGraphTensorId
}

impl MilliOpCastLike {
    pub(crate) fn new(data: MilliOpGraphTensorId, target_type: MilliOpGraphTensorId) -> Self {
        Self {data, target_type}
    }
}

impl MilliOp for MilliOpCastLike {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.data]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>, backend: &mut EvalBackend) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        Ok(inputs[&self.data].cast(inputs[&self.target_type].dtype(), backend)?)
    }
}

pub(crate) struct MilliOpTranspose {
    data: MilliOpGraphTensorId,
    perm: Option<Vec<i64>>
}

impl MilliOpTranspose {
    pub(crate) fn new(    data: MilliOpGraphTensorId,
                          perm: Option<Vec<i64>>) -> Self {
        Self {data, perm}
    }
}

impl MilliOp for MilliOpTranspose {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.data]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>, backend: &mut EvalBackend) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        Ok(inputs[&self.data].transpose(self.perm.clone(), backend)?)
    }
}

pub(crate) struct MilliOpGather {
    data: MilliOpGraphTensorId,
    indices: MilliOpGraphTensorId,
    axis: i64
}

impl MilliOpGather {
    pub(crate) fn new(    data: MilliOpGraphTensorId,
                          indices: MilliOpGraphTensorId,
                          axis: i64) -> Self {
        Self {data, indices, axis}
    }
}

impl MilliOp for MilliOpGather {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.data]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>, backend: &mut EvalBackend) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        Ok(NumericTensor::<DynRank>::gather(&inputs[&self.data], &inputs[&self.indices], self.axis, backend)?)
    }
}

pub(crate) struct MilliOpConcat {
    inputs: Vec<MilliOpGraphTensorId>,
    axis: i64
}

impl MilliOpConcat {
    pub(crate) fn new(    inputs: Vec<MilliOpGraphTensorId>,
                          axis: i64) -> Self {
        Self {inputs, axis}
    }
}

impl MilliOp for MilliOpConcat {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {self.inputs.clone()}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>, backend: &mut EvalBackend) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let mut resolved_inputs = vec![];
        for input in &self.inputs {
            resolved_inputs.push(&inputs[input]);
        }
        let axis = if self.axis < 0 {
            resolved_inputs[0].shape().len() as i64 + self.axis
        } else {
            self.axis
        } as usize;
        Ok(NumericTensor::<DynRank>::concat(resolved_inputs.as_slice(), axis, backend)?)
    }
}

pub(crate) struct MilliOpSplit {
    data: MilliOpGraphTensorId,
    split: Option<MilliOpTensorIDOrLiteral>,
    axis: i64,
    num_outputs: Option<usize>,
    output_id: usize,
}

impl MilliOpSplit {
    pub(crate) fn new(    data: MilliOpGraphTensorId,
                          split: Option<MilliOpTensorIDOrLiteral>,
                          axis: i64,
                          num_outputs: Option<usize>,
                          output_id: usize
    ) -> Self {
        Self {data, split, axis, num_outputs, output_id}
    }
}

impl MilliOp for MilliOpSplit {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.data]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>, backend: &mut EvalBackend) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let split: Vec<i64> = if let Some(split) = &self.split {
            match split {
                MilliOpTensorIDOrLiteral::TensorID(split) => {inputs[&split].clone().try_to_rank::<P1>()?.try_into()?}
                MilliOpTensorIDOrLiteral::Literal(split) => {split.try_to_rank::<P1>()?.try_into()?}
            }
        } else {
            Err(MilliOpGraphError::InvalidInput("Split attribute is not set, and we do not support num_outputs yet".to_string()))?
        };
        let outs = inputs[&self.data].split(&split, self.axis, backend)?;
        Ok(outs[self.output_id].clone())
    }
}

pub(crate) struct MilliOpWhere {
    condition: MilliOpGraphTensorId,
    x: MilliOpGraphTensorId,
    y: MilliOpGraphTensorId,
}

impl MilliOpWhere {
    pub(crate) fn new(    condition: MilliOpGraphTensorId,
                          x: MilliOpGraphTensorId,
                          y: MilliOpGraphTensorId
    ) -> Self {
        Self{condition, x, y}
    }
}

impl MilliOp for MilliOpWhere {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.condition, self.x, self.y]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>, backend: &mut EvalBackend) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        Ok(inputs[&self.condition].where_op(&inputs[&self.x], &inputs[&self.y], backend)?)
    }
}

pub(crate) enum AnyMilliOp {
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
    Expand(MilliOpExpand)
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
            AnyMilliOp::Expand(x) => x.get_inputs()
        }
    }

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, symbolic_resolver: &mut SymbolicResolver, backend: &mut EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
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
            AnyMilliOp::Squeeze(x) =>  x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Unsqueeze(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Gather(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Concat(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Split(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Where(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Range(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Expand(x) => x.infer(known_inputs, symbolic_resolver, backend),
        }
    }

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>, backend: &mut EvalBackend) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
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
            AnyMilliOp::Squeeze(x) =>  x.eval(inputs, backend),
            AnyMilliOp::Unsqueeze(x) => x.eval(inputs, backend),
            AnyMilliOp::Gather(x) => x.eval(inputs, backend),
            AnyMilliOp::Concat(x) => x.eval(inputs, backend),
            AnyMilliOp::Split(x) => x.eval(inputs, backend),
            AnyMilliOp::Where(x) => x.eval(inputs, backend),
            AnyMilliOp::Range(x) => x.eval(inputs, backend),
            AnyMilliOp::Expand(x) => x.eval(inputs, backend),
        }
    }
}

pub struct MilliOpGraph {
    input_map: HashMap<TensorId, MilliOpGraphTensorId>,
    output_map: Option<HashMap<MilliOpGraphTensorId, TensorId>>,
    ops: HashMap<MilliOpGraphTensorId, AnyMilliOp>,
    next_op_id: usize
}

impl MilliOpGraph {
    pub(crate) fn new(inputs: &[TensorId]) -> (Self, HashMap<TensorId, MilliOpGraphTensorId>) {
        let mut next_op_id = 0;
        let mut input_map = HashMap::new();
        for input in inputs {
            input_map.insert(*input, MilliOpGraphTensorId{inner:next_op_id});
            next_op_id += 1;
        }
        (Self{
            input_map: input_map.clone(),
            ops: HashMap::new(),
            output_map: None,
            next_op_id
        }, input_map)
    }

    pub(crate) fn set_output_map(&mut self, output_map: HashMap<MilliOpGraphTensorId, TensorId>) {
        assert!(self.output_map.is_none());
        self.output_map = Some(output_map)
    }

    pub(crate) fn push_op(&mut self, op: AnyMilliOp) -> MilliOpGraphTensorId {
        let new_tensor_id = MilliOpGraphTensorId{inner:self.next_op_id};
        self.next_op_id += 1;
        self.ops.insert(new_tensor_id, op);
        new_tensor_id
    }

    pub(crate) fn eval(&self, inputs: &HashMap<TensorId, NumericTensor<DynRank>>, backend: &mut EvalBackend) -> Result<HashMap<TensorId, NumericTensor<DynRank>>, MilliOpGraphError> {
        assert!(self.output_map.is_some());

        let op_ids_to_eval: Vec<_> = {
            let mut x = self.ops.keys().collect::<Vec<_>>();
            x.sort();
            x
        };

        let mut intermediate_values = HashMap::new();
        for (tensor_id, tensor_value) in inputs {
            intermediate_values.insert(self.input_map[tensor_id], tensor_value.clone());
        }

        for op_id in op_ids_to_eval {
            let op = &self.ops[op_id];
            let out = op.eval(&intermediate_values, backend)?;
            //assert_eq!(out.has_nan()?, false);
            intermediate_values.insert(*op_id, out);
        }

        let mut outputs = HashMap::new();
        for (a, b) in self.output_map.as_ref().unwrap() {
            outputs.insert(*b, intermediate_values[a].clone());
        }

        Ok(outputs)
    }
}