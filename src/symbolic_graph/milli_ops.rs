use std::collections::HashMap;
use futures::StreamExt;
use onnx_graph::tensor::Tensor;
use crate::dtype::{DType, DTypeError};
use crate::eval_backend::EvalBackend;
use crate::ndarray_backend::conversions::NDArrayNumericTensorType;
use crate::ndarray_backend::NDArrayNumericTensor;
use crate::numeric_scalar::NumericScalar;
use crate::numeric_tensor::NumericTensor;
use crate::symbolic_graph::{TensorId};
use crate::symbolic_graph::scalar_info::{ScalarInfo, ScalarInfoTyped};
use crate::symbolic_graph::symbolic_scalar::{SymbolicResolver, SymbolicScalar, SymbolicScalarTyped};
use crate::symbolic_graph::tensor_info::{MinimalTensor, RankedTensor, ShapedTensor, ShapedTensorTyped, TensorInfo};
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
    #[error("Unable to do any type if inference")]
    UnableToInfer
}

#[derive(Debug, Clone, Copy, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub(crate) struct MilliOpGraphTensorId {
    inner: usize,
}

trait MilliOp {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId>;
    /*
    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, _symbolic_resolver: &mut SymbolicResolver, backend: &EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let mut resolved_inputs = HashMap::new();
        for input in self.get_inputs() {
            if let Some(tensor_info) = known_inputs.get(&input) {
                if let TensorInfo::Numeric(value) = &tensor_info {
                    resolved_inputs.insert(input, value.clone());
                }
                else {
                    return Err(MilliOpGraphError::UnableToInfer);
                }
            }
            else {
                return Err(MilliOpGraphError::UnableToInfer);
            }
        }

        Ok(TensorInfo::Numeric(self.eval(&resolved_inputs, backend)?))
    }*/
    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, _symbolic_resolver: &mut SymbolicResolver, backend: &EvalBackend) -> Result<TensorInfo, MilliOpGraphError>;
    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, _backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError>;
}

pub(crate) struct MilliOpConstant {
    data: NumericTensor
}

impl MilliOpConstant {
    pub(crate) fn new(a: NumericTensor) -> Self {
        Self { data: a }
    }

    pub(crate) fn new_scalar<T>(v: T) -> Self
    where
        T: NDArrayNumericTensorType
    {
        Self{
            data: NDArrayNumericTensor::from_vec_shape(vec![v], &vec![1]).unwrap().into()
        }
    }
}

impl MilliOp for MilliOpConstant {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![]}

    fn infer(&self, _known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, _symbolic_resolver: &mut SymbolicResolver, _backend: &EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        Ok(TensorInfo::Numeric(self.data.clone()))
    }
    fn eval(&self, _inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, _backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
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

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, symbolic_resolver: &mut SymbolicResolver, backend: &EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
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
            TensorInfo::Numeric(NumericTensor::range(
                *start,
                *end,
                *delta,
                backend
            )?)
        } 
        else {
            TensorInfo::Ranked(RankedTensor::new(
                ScalarInfo::Symbolic(SymbolicScalar::new(DType::I64, symbolic_resolver)),
                vec![ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver))]
            ))
        })
    }

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(NumericTensor::range(
            inputs[&self.start].to_scalar()?,
            inputs[&self.end].to_scalar()?,
            inputs[&self.delta].to_scalar()?,
            backend
        )?)
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

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, symbolic_resolver: &mut SymbolicResolver, _backend: &EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let input = &known_inputs[&self.shape];
        match input {
            TensorInfo::Numeric(value) => {
                let shape: Vec<i64> = value.clone().try_into()?;
                let shape_usize = shape.iter().map(|x| *x as usize).collect::<Vec<usize>>();
                let out = NDArrayNumericTensor::fill(self.value.clone(), shape_usize.as_slice())?.into();
                Ok(TensorInfo::Numeric(out))
            }
            TensorInfo::Shaped(shaped_tensor) => {
                assert_eq!(shaped_tensor.dtype(), DType::I64);
                assert_eq!(shaped_tensor.shape().len(), 1);
                let new_rank = shaped_tensor.shape()[0];
                let mut output_shape = vec![];
                if let ShapedTensor::I64(typed_tensor) = shaped_tensor {
                    for i in 0..new_rank {
                        output_shape.push(typed_tensor.get(i).cast::<u64>())
                    }
                } else {
                    panic!("This should not happen")
                }
                Ok(TensorInfo::Ranked(RankedTensor::new(
                    ScalarInfo::Numeric(self.value),
                    output_shape
                )))
            }
            TensorInfo::Ranked(ranked_tensor) => {
                match ranked_tensor.shape()[0].cast::<u32>() {
                    ScalarInfoTyped::Numeric(x) => {
                        let mut new_shape = vec![];
                        new_shape.push(ranked_tensor.first_element().cast::<u64>());
                        for _ in 1..x {
                            new_shape.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)))
                        }
                        Ok(TensorInfo::Ranked(RankedTensor::new(
                            ScalarInfo::Numeric(self.value),
                            new_shape
                        )))
                    }
                    ScalarInfoTyped::Symbolic(x) => {
                        Ok(TensorInfo::Minimal(MinimalTensor::new(
                            ScalarInfo::Numeric(self.value),
                            x
                        )))
                    }
                }
            }
            TensorInfo::Minimal(_minimal_tensor) => {
                Ok(TensorInfo::Minimal(MinimalTensor::new(
                    ScalarInfo::Numeric(self.value),
                    SymbolicScalarTyped::new(symbolic_resolver)
                )))
            }
        }
    }

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, _backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        let shape: Vec<i64> = inputs[&self.shape].clone().try_into()?;
        let shape_usize = shape.iter().map(|x| *x as usize).collect::<Vec<usize>>();
        Ok(NDArrayNumericTensor::fill(self.value.clone(), shape_usize.as_slice())?.into())
    }
}

fn infer_multidirectional_broadcasting(shapes: &[&[ScalarInfoTyped<u64>]], symbolic_resolver: &mut SymbolicResolver) -> Result<Vec<ScalarInfoTyped<u64>>, MilliOpGraphError> {
    if shapes.is_empty() {
        return Err(MilliOpGraphError::InvalidInput("Cannot broadcast empty input".to_string()));
    }
    let output_rank = shapes.iter().map(|x| x.len()).max().unwrap();
    let mut output_shape = vec![];
    for i in 0..output_rank {
        let mut dim = ScalarInfoTyped::<u64>::Numeric(1);
        for shape in shapes {
            let local_i = (i as i64 - output_rank as i64) + shape.len() as i64;
            if local_i < 0 {
                // Infer dim of size 1, and pass
            } else {
                let local_dim = &shape[local_i as usize];
                match local_dim {
                    ScalarInfoTyped::Numeric(x) => {
                        if *x == 1 {
                            // Do not modify the dimension, pass it through.
                        }
                        else {
                            match dim {
                                ScalarInfoTyped::Numeric(y) => {
                                    if y == 1 || *x == y {
                                        dim = ScalarInfoTyped::Numeric(y.max(*x));
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
                                match dim.try_eq(local_dim) {
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
}

impl MilliOp for MilliOpSimpleBinary {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.a.clone(), self.b.clone()]}

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, symbolic_resolver: &mut SymbolicResolver, backend: &EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let a = &known_inputs[&self.a];
        let b = &known_inputs[&self.b];
        if let (TensorInfo::Numeric(a), TensorInfo::Numeric(b)) = (a, b) {
            // Call through to eval
            let inputs = HashMap::from([(self.a.clone(), a.clone()), (self.b.clone(), b.clone())]);
            Ok(TensorInfo::Numeric(
                self.eval(&inputs, backend)?
            ))
        }
        else {
            let dtype = a.dtype();
            assert_eq!(b.dtype(), dtype);
            let shape_a = a.shape();
            let shape_b = b.shape();
            if let (Some(a), Some(b)) = (shape_a, shape_b) {
                // Both ranks are known, can do broadcast inference
                let new_shape = infer_multidirectional_broadcasting(&[a.as_slice(), b.as_slice()], symbolic_resolver)?;
                Ok(TensorInfo::Ranked(RankedTensor::new(
                    ScalarInfo::Symbolic(SymbolicScalar::new(dtype, symbolic_resolver)),
                    new_shape
                )))
            }
            else {
                // Need to know the rank of both inputs to infer the rank of the output
                let rank_a = a.rank();
                let rank_b = b.rank();
                if rank_a.try_eq(&rank_b).unwrap_or(false) {
                    // Same rank, so we can use this for the output rank
                    match rank_a {
                        ScalarInfoTyped::Numeric(_x) => {
                            // Should not happen (already handled by earlier cases)
                            unreachable!()
                        }
                        ScalarInfoTyped::Symbolic(x) => {
                            Ok(TensorInfo::Minimal(MinimalTensor::new(ScalarInfo::Symbolic(SymbolicScalar::new(dtype, symbolic_resolver)), x)))
                        }
                    }
                } else {
                    // Different or unresolvable ranks
                    Ok(TensorInfo::Minimal(MinimalTensor::new(
                        ScalarInfo::Symbolic(SymbolicScalar::new(dtype, symbolic_resolver)),
                        SymbolicScalarTyped::new(symbolic_resolver)
                    )))
                }
            }
        }
    }

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        let a = &inputs[&self.a];
        let b = &inputs[&self.b];
        Ok(match self.which_op {
            SimpleBinaryOp::Add => NumericTensor::add(a, b, backend)?,
            SimpleBinaryOp::Sub => NumericTensor::sub(a, b, backend)?,
            SimpleBinaryOp::Mul => NumericTensor::mul(a, b, backend)?,
            SimpleBinaryOp::Div => NumericTensor::div(a, b, backend)?,
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

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, symbolic_resolver: &mut SymbolicResolver, backend: &EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let a = &known_inputs[&self.a];
        let b = &known_inputs[&self.b];
        if let (TensorInfo::Numeric(a), TensorInfo::Numeric(b)) = (a, b) {
            // Call through to eval
            let inputs = HashMap::from([(self.a.clone(), a.clone()), (self.b.clone(), b.clone())]);
            Ok(TensorInfo::Numeric(
                self.eval(&inputs, backend)?
            ))
        }
        else {
            let dtype = a.dtype();
            let shape_a = a.shape();
            let shape_b = b.shape();
            if let (Some(a), Some(b)) = (shape_a, shape_b) {
                // Both ranks are known, can do broadcast inference
                let new_shape = infer_multidirectional_broadcasting(&[a.as_slice(), b.as_slice()], symbolic_resolver)?;
                Ok(TensorInfo::Ranked(RankedTensor::new(
                    ScalarInfo::Symbolic(SymbolicScalar::new(dtype, symbolic_resolver)),
                    new_shape
                )))
            }
            else {
                // Need to know the rank of both inputs to infer the rank of the output
                let rank_a = a.rank();
                let rank_b = b.rank();
                if rank_a.try_eq(&rank_b).unwrap_or(false) {
                    // Same rank, so we can use this for the output rank
                    match rank_a {
                        ScalarInfoTyped::Numeric(_x) => {
                            // Should not happen (already handled by earlier cases)
                            unreachable!()
                        }
                        ScalarInfoTyped::Symbolic(x) => {
                            Ok(TensorInfo::Minimal(MinimalTensor::new(ScalarInfo::Symbolic(SymbolicScalar::new(dtype, symbolic_resolver)), x)))
                        }
                    }
                } else {
                    // Different or unresolvable ranks
                    Ok(TensorInfo::Minimal(MinimalTensor::new(
                        ScalarInfo::Symbolic(SymbolicScalar::new(dtype, symbolic_resolver)),
                        SymbolicScalarTyped::new(symbolic_resolver)
                    )))
                }
            }
        }
    }

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(NumericTensor::pow(&inputs[&self.a], &inputs[&self.b], backend)?)
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

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, symbolic_resolver: &mut SymbolicResolver, backend: &EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let a = &known_inputs[&self.a];
        let b = &known_inputs[&self.b];
        if let (TensorInfo::Numeric(a), TensorInfo::Numeric(b)) = (a, b) {
            let inputs = HashMap::from([(self.a.clone(), a.clone()), (self.b.clone(), b.clone())]);
            Ok(TensorInfo::Numeric(
                self.eval(&inputs, backend)?
            ))
        }
        else {
            let dtype = a.dtype();
            assert_eq!(b.dtype(), dtype);
            let shape_a = a.shape();
            let shape_b = b.shape();

            if let (Some(shape_a), Some(shape_b)) = (shape_a, shape_b) {
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

                Ok(TensorInfo::Ranked(RankedTensor::new(
                    ScalarInfo::Symbolic(SymbolicScalar::new(dtype, symbolic_resolver)),
                    dims_out
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
    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(NumericTensor::matmul(&inputs[&self.a], &inputs[&self.b], backend)?)
    }
}


trait SimpleUnaryMilliOp {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId>;

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError>;

    fn eval_scalar(&self, input: &NumericScalar) -> Result<NumericScalar, MilliOpGraphError>;
}

impl<T: SimpleUnaryMilliOp> MilliOp for T {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        <T as SimpleUnaryMilliOp>::get_inputs(self)
    }

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, symbolic_resolver: &mut SymbolicResolver, backend: &EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let input_id = self.get_inputs()[0];
        let input = &known_inputs[&input_id];
        match input {
            TensorInfo::Numeric(numeric_tensor) => {
                let inputs = HashMap::from([(input_id, numeric_tensor.clone())]);
                Ok(TensorInfo::Numeric(
                    self.eval(&inputs, backend)?
                ))
            }
            TensorInfo::Shaped(shaped_tensor) => {
                // For now, just decay this to a ranked tensor
                Ok(TensorInfo::Ranked(
                    RankedTensor::new(
                        ScalarInfo::Symbolic(SymbolicScalar::new(shaped_tensor.dtype(), symbolic_resolver)),
                        shaped_tensor.shape().iter().map(|x| ScalarInfoTyped::Numeric(*x as u64)).collect()
                    )
                ))
            }
            TensorInfo::Ranked(ranked_tensor) => {
                let new_first_value = match ranked_tensor.first_element() {
                    ScalarInfo::Numeric(x) => {
                        ScalarInfo::Numeric(self.eval_scalar(x)?)
                    }
                    ScalarInfo::Symbolic(x) => {
                        ScalarInfo::Symbolic(SymbolicScalar::new(x.dtype(), symbolic_resolver))
                    }
                };
                Ok(TensorInfo::Ranked(
                    RankedTensor::new(
                        new_first_value,
                        ranked_tensor.shape().to_vec()
                    )
                ))
            }
            TensorInfo::Minimal(minimal_tensor) => {
                let new_first_value = match minimal_tensor.first_element() {
                    ScalarInfo::Numeric(x) => {
                        ScalarInfo::Numeric(self.eval_scalar(x)?)
                    }
                    ScalarInfo::Symbolic(x) => {
                        ScalarInfo::Symbolic(SymbolicScalar::new(x.dtype(), symbolic_resolver))
                    }
                };
                Ok(TensorInfo::Minimal(MinimalTensor::new(
                    new_first_value,
                    minimal_tensor.rank().clone()
                )))
            }
        }
    }

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        <T as SimpleUnaryMilliOp>::eval(self, inputs, backend)
    }
}

pub(crate) struct MilliOpNeg {
    input: MilliOpGraphTensorId
}

impl MilliOpNeg {
    pub(crate) fn new(input: MilliOpGraphTensorId) -> Self {
        Self {input}
    }
}

impl SimpleUnaryMilliOp for MilliOpNeg {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.input]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        let input = &inputs[&self.input];
        Ok(input.neg(backend)?)
    }

    fn eval_scalar(&self, input: &NumericScalar) -> Result<NumericScalar, MilliOpGraphError> {
        Ok(input.neg())
    }
}

pub(crate) struct MilliOpAbs {
    input: MilliOpGraphTensorId
}

impl MilliOpAbs {
    pub(crate) fn new(input: MilliOpGraphTensorId) -> Self {
        Self {input}
    }
}

impl SimpleUnaryMilliOp for MilliOpAbs {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.input]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(inputs[&self.input].abs(backend)?)
    }

    fn eval_scalar(&self, input: &NumericScalar) -> Result<NumericScalar, MilliOpGraphError> {
        Ok(input.abs())
    }
}

pub(crate) struct MilliOpExp {
    input: MilliOpGraphTensorId
}

impl MilliOpExp {
    pub(crate) fn new(input: MilliOpGraphTensorId) -> Self {
        Self { input }
    }
}

impl SimpleUnaryMilliOp for MilliOpExp {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.input]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(inputs[&self.input].exp(backend)?)
    }

    fn eval_scalar(&self, input: &NumericScalar) -> Result<NumericScalar, MilliOpGraphError> {
        Ok(input.exp())
    }
}

pub(crate) struct MilliOpLog {
    input: MilliOpGraphTensorId
}

impl MilliOpLog {
    pub(crate) fn new(input: MilliOpGraphTensorId) -> Self {
        Self { input }
    }
}

impl SimpleUnaryMilliOp for MilliOpLog {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.input]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(inputs[&self.input].ln(backend)?)
    }

    fn eval_scalar(&self, input: &NumericScalar) -> Result<NumericScalar, MilliOpGraphError> {
        Ok(input.ln())
    }
}


pub(crate) struct MilliOpTrig {
    input: MilliOpGraphTensorId,
    op: TrigOp
}

impl MilliOpTrig {
    pub(crate) fn new(input: MilliOpGraphTensorId, op: TrigOp) -> Self {
        Self { input, op}
    }
}

impl SimpleUnaryMilliOp for MilliOpTrig {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.input]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(inputs[&self.input].trig(self.op, backend)?)
    }

    fn eval_scalar(&self, input: &NumericScalar) -> Result<NumericScalar, MilliOpGraphError> {
        Ok(input.trig(self.op))
    }
}

pub(crate) struct MilliOpSqrt {
    input: MilliOpGraphTensorId
}

impl MilliOpSqrt {
    pub(crate) fn new(input: MilliOpGraphTensorId) -> Self {
        Self { input }
    }
}

impl SimpleUnaryMilliOp for MilliOpSqrt {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.input]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(inputs[&self.input].sqrt(backend)?)
    }

    fn eval_scalar(&self, input: &NumericScalar) -> Result<NumericScalar, MilliOpGraphError> {
        Ok(input.sqrt())
    }
}


pub(crate) struct MilliOpReciprocal {
    input: MilliOpGraphTensorId
}

impl MilliOpReciprocal {
    pub(crate) fn new(input: MilliOpGraphTensorId) -> Self {
        Self { input }
    }
}

impl SimpleUnaryMilliOp for MilliOpReciprocal {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.input]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(inputs[&self.input].reciprocal(backend)?)
    }

    fn eval_scalar(&self, input: &NumericScalar) -> Result<NumericScalar, MilliOpGraphError> {
        Ok(input.recip())
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

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
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
    pub(crate) fn new(a: MilliOpGraphTensorId) -> Self {
        Self { input: a }
    }
}

impl MilliOp for MilliOpNonZero {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.input]}

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, symbolic_resolver: &mut SymbolicResolver, backend: &EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let input = &known_inputs[&self.input];
        if let TensorInfo::Numeric(a) = input {
            Ok(TensorInfo::Numeric(a.nonzero(backend)?))
        }
        else {
            // Don't even try shape inference for now
            Ok(TensorInfo::Minimal(MinimalTensor::new(
                ScalarInfo::Symbolic(SymbolicScalar::new(DType::I64, symbolic_resolver)),
                SymbolicScalarTyped::new(symbolic_resolver)
            )))
        }
    }

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
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

    fn eval(&self, _inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, _backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
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

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, symbolic_resolver: &mut SymbolicResolver, backend: &EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let input = &known_inputs[&self.input];
        match input {
            TensorInfo::Numeric(numeric_tensor) => {
                let inputs = HashMap::from([(self.input, numeric_tensor.clone())]);
                Ok(TensorInfo::Numeric(
                    self.eval(&inputs, backend)?
                ))
            }
            TensorInfo::Shaped(shaped_tensor) => {
                let shape: Vec<_> = shaped_tensor.shape().iter().map(|x| *x as i64).collect();
                Ok(TensorInfo::Numeric(
                    NDArrayNumericTensor::from(shape).into()
                ))
            }
            TensorInfo::Ranked(ranked_tensor) => {
                let shape: Vec<_> = ranked_tensor.shape().iter().map(|x| x.cast::<i64>()).collect();
                Ok(TensorInfo::Shaped(ShapedTensor::I64(
                    ShapedTensorTyped::new(
                        vec![shape.len()],
                        shape
                    )
                )))
            }
            TensorInfo::Minimal(minimal_tensor) => {
                Ok(TensorInfo::Ranked(RankedTensor::new(
                    ScalarInfo::Symbolic(SymbolicScalar::new(DType::I64, symbolic_resolver)),
                    vec![ScalarInfoTyped::Symbolic(minimal_tensor.rank().cast::<u64>())]
                )))
            }
        }
    }

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, _backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        let output_shape = inputs[&self.input].shape().into_iter().map(|x| x as i64).collect::<Vec<_>>();
        Ok(NDArrayNumericTensor::from(output_shape).into())
    }
}

pub(crate) struct MilliOpReduceSum {
    data: MilliOpGraphTensorId,
    axes: Option<MilliOpGraphTensorId>,
    keepdims: bool
}

impl MilliOpReduceSum {
    pub(crate) fn new(data: MilliOpGraphTensorId, axes: Option<MilliOpGraphTensorId>, keepdims: bool) -> Self {
        Self { data, axes, keepdims}
    }
}

impl MilliOp for MilliOpReduceSum {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.data]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        let data = &inputs[&self.data];
        let axes = if let Some(axes) = self.axes {
            Vec::<i64>::try_from(inputs[&axes].clone())?
        } else {
            (0i64 .. (data.shape().len() as i64)).into_iter().collect()
        };
        let out = data.reduce_sum(Some(axes), self.keepdims, backend)?;
        Ok(out)
    }
}

pub(crate) struct MilliOpReduceProd {
    data: MilliOpGraphTensorId,
    axes: Option<MilliOpGraphTensorId>,
    keepdims: bool
}

impl MilliOpReduceProd {
    pub(crate) fn new(data: MilliOpGraphTensorId, axes: Option<MilliOpGraphTensorId>, keepdims: bool) -> Self {
        Self { data, axes, keepdims}
    }
}

impl MilliOp for MilliOpReduceProd {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.data]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        let data = &inputs[&self.data];
        let axes = if let Some(axes) = self.axes {
            Vec::<i64>::try_from(inputs[&axes].clone())?
        } else {
            (0i64 .. (data.shape().len() as i64)).into_iter().collect()
        };
        let out = data.reduce_prod(Some(axes), self.keepdims, backend)?;
        Ok(out)
    }
}


pub(crate) struct MilliOpReduceMean {
    data: MilliOpGraphTensorId,
    axes: Option<MilliOpGraphTensorId>,
    keepdims: bool
}

impl MilliOpReduceMean {
    pub(crate) fn new(data: MilliOpGraphTensorId, axes: Option<MilliOpGraphTensorId>, keepdims: bool) -> Self {
        Self { data, axes, keepdims}
    }
}

impl MilliOp for MilliOpReduceMean {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.data]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        let data = &inputs[&self.data];
        let axes = if let Some(axes) = self.axes {
            Vec::<i64>::try_from(inputs[&axes].clone())?
        } else {
            (0i64 .. (data.shape().len() as i64)).into_iter().collect()
        };
        let out = data.reduce_mean(Some(axes), self.keepdims, backend)?;
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

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, symbolic_resolver: &mut SymbolicResolver, backend: &EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let data_input = &known_inputs[&self.data];
        let start_input = &known_inputs[&self.starts];
        let end_input = &known_inputs[&self.ends];
        let steps_input = self.steps.map(|x| &known_inputs[&x]);
        let axes_input = self.axes.map(|x| &known_inputs[&x]);
        
        let output_rank = data_input.rank();
        
        let output_rank = match output_rank {
            ScalarInfoTyped::Numeric(output_rank) => {Some(output_rank)}
            ScalarInfoTyped::Symbolic(output_rank) => {
                if let Some(axes_input) = axes_input {
                    // Axes are being selected, we cannot infer output rank from the axes
                    None
                } else {
                    // output rank should match input len
                    if let ScalarInfoTyped::Numeric(x) = start_input.shape()[0] {
                        
                    }
                }
            }
        }

        match data_input {
            TensorInfo::Numeric(data_input_numeric) => {
                // We know the start element for certain, and we know the rank for certain
                let is_steps_numeric_or_none = steps_input.map(|x| if let TensorInfo::Numeric(_) = x {true} else {false}).unwrap_or(true);
                let is_axes_numeric_or_none = axes_input.map(|x| if let TensorInfo::Numeric(_) = x {true} else {false}).unwrap_or(true);

                // Try to use eval
                let res = if is_steps_numeric_or_none && is_axes_numeric_or_none {
                    if let (TensorInfo::Numeric(start_input), TensorInfo::Numeric(end_input)) = (start_input, end_input) {
                        let mut inputs = HashMap::from([(self.data, data_input_numeric.clone()), (self.starts, start_input.clone()), (self.ends, end_input.clone())]);
                        if let Some(steps_input_id) = self.steps {
                            if let TensorInfo::Numeric(steps_input) = steps_input.unwrap() {
                                inputs.insert(steps_input_id, steps_input.clone());
                            }
                        }
                        if let Some(axes_input_id) = self.axes {
                            if let TensorInfo::Numeric(axes_input) = axes_input.unwrap() {
                                inputs.insert(axes_input_id, axes_input.clone());
                            }
                        }
                        Some(Ok(TensorInfo::Numeric(self.eval(&inputs, backend)?)))
                    }
                    else {
                        // We do not know the values
                        None
                    }
                }
                else {
                    None
                };
                
                if let Some(res) = res {
                    res
                } else {
                    // We know the values for the data input, but at least one value from the slice is unknown/symbolic. 
                    // This means we cannot fully numerically determine the output shape.
                    // Do our best to figure it out symbolically
                    let mut output_rank = data_input_numeric.rank();
                    let mut output_shape = vec![];
                    
                    if is_axes_numeric_or_none {
                        for i in 0..output_rank {
                            // Determine if this axis is controlled by the slice or not
                            let inner_i = if let Some(axes_input) = axes_input {
                                if let TensorInfo::Numeric(axes_input) = axes_input {
                                    let axes_vec: Vec<i64> = axes_input.to_1d_vec()?;
                                    let mut res = None;
                                    for (j, axis) in axes_vec.iter().enumerate() {
                                        if (*axis as usize) == i {
                                            res = Some(j);
                                            break;
                                        }
                                    }
                                    res
                                } else {
                                    unreachable!()
                                }
                            } else {
                                Some(i)
                            };
                            
                            if let Some(inner_i) = inner_i {
                                // Axis is controlled by the slice, can only resolve if we have a full numeric set
                                let start = start_input.get(&[inner_i as u64], symbolic_resolver).cast::<i64>();
                                let end = end_input.get(&[inner_i as u64], symbolic_resolver).cast::<i64>();
                                let step = steps_input.map(|x| x.get(&[inner_i as u64], symbolic_resolver).cast::<i64>());
                                
                                if let Some(step) = step {
                                    // Give up for now todo: do better
                                    output_shape.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)));
                                } else {
                                    if let (ScalarInfoTyped::Numeric(start), ScalarInfoTyped::Numeric(end)) = (start, end) {
                                        output_shape.push(ScalarInfoTyped::Numeric((start - end).abs() as u64))
                                    }
                                }
                                
                            } else {
                                // Axis is not controlled by the slice
                                output_shape.push(ScalarInfoTyped::Numeric(data_input_numeric.shape()[i] as u64))
                            }
                        }
                    } else {
                        // Impossible to resolve shape, push new symbolic axes
                        for _ in 0..output_rank {
                            output_shape.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)));
                        }
                    }
                    Ok(TensorInfo::Ranked(RankedTensor::new(
                        ScalarInfo::Symbolic(SymbolicScalar::new(data_input.dtype(), symbolic_resolver)),
                        output_shape
                    )))
                }
            }
            TensorInfo::Shaped(data_input_shaped) => {
                
            }
            TensorInfo::Ranked(data_input_ranked) => {

            }
            TensorInfo::Minimal(data_input_minimal) => {
                // todo: in theory we could try to use start/end/steps/axes to figure more stuff out
                Ok(TensorInfo::Minimal(MinimalTensor::new(
                    ScalarInfo::Symbolic(SymbolicScalar::new(data_input_minimal.dtype(), symbolic_resolver)),
                    data_input_minimal.rank().clone()
                )))
            }
        }
    }

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        let data_input = &inputs[&self.data];
        let input_shape = data_input.shape();
        let input_rank = data_input.rank();
        let axes: Vec<i64> = if let Some(axes) = &self.axes {
            inputs[axes].cast(DType::I64, backend)?.try_into()?
        } else {
            (0i64..(input_rank as i64)).into_iter().collect()
        };
        let steps: Vec<i64> = if let Some(steps) = &self.steps {
            inputs[steps].cast(DType::I64, backend)?.try_into()?
        } else {
            axes.iter().map(|_| 1).collect()
        };
        let starts: Vec<i64> = inputs[&self.starts].cast(DType::I64, backend)?.try_into()?;
        let ends: Vec<i64> = inputs[&self.ends].cast(DType::I64, backend)?.try_into()?;
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
            }).min(input_shape[axis] as i64).max(0) as usize;

            let end = (if ends[i] < 0 {
                input_shape[axis] as i64 + ends[i]
            } else {
                ends[i]
            }).min(input_shape[axis] as i64).max(0) as usize;
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

    fn calculate_new_shape(&self, data_input_shape: Vec<u64>, shape_input_value: Vec<i64>) -> Result<Vec<u64>, MilliOpGraphError> {
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

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, symbolic_resolver: &mut SymbolicResolver, backend: &EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let data_input = &known_inputs[&self.data];
        let shape_input = &known_inputs[&self.shape];

        match &shape_input {
            TensorInfo::Numeric(shape_input_numeric_tensor) => {
                let shape_input_value: Vec<i64> = shape_input_numeric_tensor.cast(DType::I64, backend)?.try_into()?;
                match data_input {
                    TensorInfo::Numeric(data_input_numeric_tensor) => {
                        // Just run it straight
                        let inputs = HashMap::from([
                            (self.shape, shape_input_numeric_tensor.clone()), (self.data, data_input_numeric_tensor.clone())]);
                        Ok(TensorInfo::Numeric(self.eval(&inputs, backend)?))
                    }
                    TensorInfo::Shaped(data_input_shaped_tensor) => {
                        let data_shape: Vec<_> = data_input_shaped_tensor.shape().iter().map(|x| *x as u64).collect();
                        let new_shape = self.calculate_new_shape(
                            data_shape, shape_input_value)?;
                        let new_shape_usize: Vec<_> = new_shape.iter().map(|x| *x as usize).collect();
                        Ok(TensorInfo::Shaped(data_input_shaped_tensor.reshape(new_shape_usize)))
                    }
                    TensorInfo::Ranked(data_input_ranked_tensor) => {
                        let mut data_shape = vec![];
                        let mut could_resolve_input_shape = true;
                        for dim in data_input_ranked_tensor.shape() {
                            match dim {
                                ScalarInfoTyped::Numeric(x) => {
                                    data_shape.push(*x);
                                }
                                ScalarInfoTyped::Symbolic(_) => {
                                    could_resolve_input_shape = false;
                                    break;
                                }
                            }
                        }
                        if could_resolve_input_shape {
                            let new_shape = self.calculate_new_shape(
                                data_shape, shape_input_value)?;
                            let new_shape: Vec<_> = new_shape.iter().map(|x| ScalarInfoTyped::Numeric(*x)).collect();
                            Ok(TensorInfo::Ranked(RankedTensor::new(
                                data_input_ranked_tensor.first_element().clone(),
                                new_shape
                            )))
                        }
                        else {
                            // Could not resolve the input shape, so we are stuck with imperfect resolution here
                            let input_shape_info = data_input_ranked_tensor.shape();
                            let mut output_shape = vec![];
                            for (i, value) in shape_input_value.iter().enumerate() {
                                if *value == 0 {
                                    output_shape.push(input_shape_info[i].clone());
                                } else if *value < 0 {
                                    output_shape.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)));
                                } else {
                                    output_shape.push(ScalarInfoTyped::Numeric(*value as u64));
                                }
                            }
                            Ok(TensorInfo::Ranked(RankedTensor::new(
                                data_input_ranked_tensor.first_element().clone(),
                                output_shape
                            )))
                        }
                    }
                    TensorInfo::Minimal(data_input_minimal_tensor) => {
                        // Could not resolve even the input rank, so even worse resolution
                        let mut output_shape = vec![];
                        for (_i, value) in shape_input_value.iter().enumerate() {
                            if *value == 0 {
                                output_shape.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)));
                            } else if *value < 0 {
                                output_shape.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)));
                            } else {
                                output_shape.push(ScalarInfoTyped::Numeric(*value as u64));
                            }
                        }
                        Ok(TensorInfo::Ranked(RankedTensor::new(
                            data_input_minimal_tensor.first_element().clone(),
                            output_shape
                        )))
                    }
                }
            }
            TensorInfo::Shaped(shape_input_shaped_tensor) => {
                let mut shape_input_values = vec![];
                let mut could_resolve_input_shape = true;
                for i in 0..shape_input_shaped_tensor.shape()[0] {
                    let val = shape_input_shaped_tensor.get(i).cast::<i64>();
                    match val {
                        ScalarInfoTyped::Numeric(x) => {
                            shape_input_values.push(x);
                        }
                        ScalarInfoTyped::Symbolic(_x) => {
                            could_resolve_input_shape = false;
                        }
                    }
                }
                match &data_input {
                    TensorInfo::Numeric(data_input_numeric_tensor) => {
                        if could_resolve_input_shape {
                            let data_shape: Vec<_> = data_input_numeric_tensor.shape().iter().map(|x| *x as u64).collect();
                            let new_shape = self.calculate_new_shape(
                                data_shape, shape_input_values)?;
                            let new_shape_usize: Vec<_> = new_shape.iter().map(|x| *x as usize).collect();
                            Ok(TensorInfo::Numeric(data_input_numeric_tensor.reshape(new_shape_usize, backend)?))
                        }
                        else {
                            // At least one dim of the shape input is symbolic, so we can't completely resolve.
                            let mut output_dims = vec![];
                            for i in 0..shape_input_shaped_tensor.shape()[0] {
                                let v = shape_input_shaped_tensor.get(i).cast::<i64>();
                                match v {
                                    ScalarInfoTyped::Numeric(x) => {
                                        if x > 0 {
                                            output_dims.push(v.cast::<u64>());
                                        } else if x == 0 {
                                            output_dims.push(ScalarInfoTyped::Numeric(data_input_numeric_tensor.shape()[i] as u64));
                                        } else {
                                            output_dims.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)));
                                        }
                                    }
                                    ScalarInfoTyped::Symbolic(_) => {
                                        // Must use a new symbol here, since it could be negative or 0 at runtime
                                        output_dims.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)));
                                    }
                                }
                            }
                            Ok(TensorInfo::Ranked(
                                RankedTensor::new(
                                    data_input.first_element(),
                                    output_dims
                                ),
                            ))
                        }
                    }
                    TensorInfo::Shaped(data_input_shaped_tensor) => {
                        if could_resolve_input_shape {
                            let data_shape: Vec<_> = data_input_shaped_tensor.shape().iter().map(|x| *x as u64).collect();
                            let new_shape = self.calculate_new_shape(
                                data_shape, shape_input_values)?;
                            let new_shape_usize: Vec<_> = new_shape.iter().map(|x| *x as usize).collect();
                            Ok(TensorInfo::Shaped(data_input_shaped_tensor.reshape(new_shape_usize)))
                        } else {
                            // At least one dim of the shape input is symbolic, so we can't completely resolve.
                            let mut output_dims = vec![];
                            for i in 0..shape_input_shaped_tensor.shape()[0] {
                                let v = shape_input_shaped_tensor.get(i).cast::<i64>();
                                match v {
                                    ScalarInfoTyped::Numeric(x) => {
                                        if x > 0 {
                                            output_dims.push(v.cast::<u64>());
                                        } else if x == 0 {
                                            output_dims.push(ScalarInfoTyped::Numeric(data_input_shaped_tensor.shape()[i] as u64));
                                        } else {
                                            output_dims.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)));
                                        }
                                    }
                                    ScalarInfoTyped::Symbolic(_) => {
                                        // Must use a new symbol here, since it could be negative or 0 at runtime
                                        output_dims.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)));
                                    }
                                }
                            }
                            Ok(TensorInfo::Ranked(
                                RankedTensor::new(
                                    data_input.first_element(),
                                    output_dims
                                ),
                            ))
                        }
                    }
                    TensorInfo::Ranked(data_input_ranked_tensor) => {
                        // At least one dim of the shape input is symbolic, so we can't completely resolve.
                        let mut output_dims = vec![];
                        for i in 0..shape_input_shaped_tensor.shape()[0] {
                            let v = shape_input_shaped_tensor.get(i).cast::<i64>();
                            match v {
                                ScalarInfoTyped::Numeric(x) => {
                                    if x > 0 {
                                        output_dims.push(v.cast::<u64>());
                                    } else if x == 0 {
                                        output_dims.push(data_input_ranked_tensor.shape()[i].clone());
                                    } else {
                                        output_dims.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)));
                                    }
                                }
                                ScalarInfoTyped::Symbolic(_) => {
                                    // Must use a new symbol here, since it could be negative or 0 at runtime
                                    output_dims.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)));
                                }
                            }
                        }
                        Ok(TensorInfo::Ranked(
                            RankedTensor::new(
                                data_input.first_element(),
                                output_dims
                            ),
                        ))
                    }
                    TensorInfo::Minimal(_) => {
                        // At least one dim of the shape input is symbolic, so we can't completely resolve.
                        let mut output_dims = vec![];
                        for i in 0..shape_input_shaped_tensor.shape()[0] {
                            let v = shape_input_shaped_tensor.get(i).cast::<i64>();
                            match v {
                                ScalarInfoTyped::Numeric(x) => {
                                    if x > 0 {
                                        output_dims.push(v.cast::<u64>());
                                    } else if x == 0 {
                                        output_dims.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)));
                                    } else {
                                        output_dims.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)));
                                    }
                                }
                                ScalarInfoTyped::Symbolic(_) => {
                                    // Must use a new symbol here, since it could be negative or 0 at runtime
                                    output_dims.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)));
                                }
                            }
                        }
                        Ok(TensorInfo::Ranked(
                            RankedTensor::new(
                                data_input.first_element(),
                                output_dims
                            ),
                        ))
                    }
                }
            }
            TensorInfo::Ranked(shape_input_ranked_tensor) => {
                let value = shape_input_ranked_tensor.shape()[0].clone();
                match value {
                    ScalarInfoTyped::Numeric(x) => {
                        // We know the full rank of the output, just not the shape
                        let mut new_shape = vec![];
                        new_shape.push(shape_input_ranked_tensor.first_element().cast::<u64>());
                        for _ in 1..x {
                            new_shape.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)));
                        }
                        Ok(TensorInfo::Ranked(RankedTensor::new(
                            data_input.first_element(),
                            new_shape
                        )))
                    }
                    ScalarInfoTyped::Symbolic(x) => {
                        // Rank is symbolic
                        Ok(TensorInfo::Minimal(MinimalTensor::new(
                            data_input.first_element(),
                            x.cast::<u32>()
                        )))
                    }
                }
            }
            TensorInfo::Minimal(_) => {
                // We know actually nothing about the final shape...
                Ok(TensorInfo::Minimal(MinimalTensor::new(
                    data_input.first_element(),
                    SymbolicScalarTyped::new(symbolic_resolver)
                )))
            }
        }
    }


    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        let data_input = &inputs[&self.data];
        let shape_input = &inputs[&self.shape];
        let data_input_shape: Vec<_> = data_input.shape().iter().map(|x| *x as u64).collect();
        let shape_input_value: Vec<i64> = shape_input.cast(DType::I64, backend)?.try_into()?;

        let output_shape = self.calculate_new_shape(
            data_input_shape,
            shape_input_value
        )?;
        
        let output_shape = output_shape.into_iter().map(|x| x as usize).collect();
        
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

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, symbolic_resolver: &mut SymbolicResolver, backend: &EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let data_input = &known_inputs[&self.data];
        let axes_input = &known_inputs[&self.axes];

        match axes_input {
            TensorInfo::Numeric(axes_input_numeric_tensor) => {
                let axes: Vec<i64> = axes_input_numeric_tensor.to_ndarray()?.try_into()?;
                match data_input {
                    TensorInfo::Numeric(data_input_numeric_tensor) => {
                        // Just run it straight
                        let inputs = HashMap::from([
                            (self.axes, axes_input_numeric_tensor.clone()), (self.data, data_input_numeric_tensor.clone())]);
                        Ok(TensorInfo::Numeric(self.eval(&inputs, backend)?))
                    }
                    TensorInfo::Shaped(data_input_shaped_tensor) => {
                        let old_shape = data_input_shaped_tensor.shape().iter().map(|&x| x as u64).collect::<Vec<_>>();
                        let mut new_shape = vec![];
                        for i in 0..old_shape.len() {
                            let mut do_skip = false;
                            for axis in &axes {
                                let axis = if *axis < 0 {
                                    old_shape.len() as i64 + axis
                                } else {
                                    *axis
                                };
                                if axis == i as i64 {
                                    do_skip = true;
                                    break;
                                }
                            }
                            if do_skip {
                                continue;
                            }
                            new_shape.push(old_shape[i] as usize);
                        }
                        Ok(TensorInfo::Shaped(data_input_shaped_tensor.reshape(new_shape)))
                    }
                    TensorInfo::Ranked(data_input_ranked_tensor) => {
                        let old_shape = data_input_ranked_tensor.shape();
                        let mut new_shape = vec![];
                        for i in 0..old_shape.len() {
                            let mut do_skip = false;
                            for axis in &axes {
                                let axis = if *axis < 0 {
                                    old_shape.len() as i64 + axis
                                } else {
                                    *axis
                                };
                                if axis == i as i64 {
                                    do_skip = true;
                                    break;
                                }
                            }
                            if do_skip {
                                continue;
                            }
                            new_shape.push(old_shape[i].clone());
                        }
                        Ok(TensorInfo::Ranked(RankedTensor::new(
                            data_input_ranked_tensor.first_element().clone(),
                            new_shape
                        )))
                    }
                    TensorInfo::Minimal(data_input_minimal_tensor) => {
                        let old_rank = data_input_minimal_tensor.rank();
                        let new_rank = old_rank.add_offset(axes.len() as i64);
                        Ok(TensorInfo::Minimal(MinimalTensor::new(
                            data_input_minimal_tensor.first_element().clone(),
                            new_rank
                        )))
                    }
                }
            }
            TensorInfo::Shaped(axes_input_shaped_tensor) => {
                // We can assume that not enough values are resolved to allow for any resolution of the output shape, limiting us to only knowing the rank
                let old_rank = data_input.rank();
                let new_rank = old_rank.add_offset(axes_input_shaped_tensor.shape()[0] as i64);
                Ok(TensorInfo::new_from_rank_and_first_value(data_input.first_element().clone(), new_rank, symbolic_resolver))
            }
            TensorInfo::Ranked(axes_input_ranked_tensor) => {
                // We can assume that not enough values are resolved to allow for any resolution of the output shape, limiting us to only knowing the rank
                let old_rank = data_input.rank();
                let new_rank = match &axes_input_ranked_tensor.shape()[0] {
                    ScalarInfoTyped::Numeric(x) => {
                        old_rank.add_offset(*x as i64)
                    }
                    ScalarInfoTyped::Symbolic(_) => {
                        ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver))
                    }
                };
                Ok(TensorInfo::new_from_rank_and_first_value(data_input.first_element().clone(), new_rank, symbolic_resolver))
            }
            TensorInfo::Minimal(_) => {
                // We don't even know the number of indexes we will be removing
                Ok(TensorInfo::Minimal(MinimalTensor::new( data_input.first_element().clone(), SymbolicScalarTyped::new(symbolic_resolver))))
            }
        }
    }

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        let axes_ndarray = NDArrayNumericTensor::try_from(inputs[&self.axes].cast(DType::I64, backend)?)?;
        let axes = Vec::<i64>::try_from(axes_ndarray)?;
        if axes.len() > 1 {
            return Err(MilliOpGraphError::InvalidInput("Unsqueeze".to_string()));
        }
        let axis = axes[0];
        let input_shape = inputs[&self.data].shape();
        let axis = if axis >= 0 {
            axis as usize
        } else {
            (input_shape.len() as i64 + axis) as usize
        };
        let output = inputs[&self.data].squeeze(axis, backend)?;
        Ok(output)
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

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, symbolic_resolver: &mut SymbolicResolver, backend: &EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let data_input = &known_inputs[&self.data];
        let axes_input = &known_inputs[&self.axes];

        match axes_input {
            TensorInfo::Numeric(axes_input_numeric_tensor) => {
                let axes: Vec<i64> = axes_input_numeric_tensor.to_ndarray()?.try_into()?;
                match data_input {
                    TensorInfo::Numeric(data_input_numeric_tensor) => {
                        // Just run it straight
                        let inputs = HashMap::from([
                            (self.axes, axes_input_numeric_tensor.clone()), (self.data, data_input_numeric_tensor.clone())]);
                        Ok(TensorInfo::Numeric(self.eval(&inputs, backend)?))
                    }
                    TensorInfo::Shaped(data_input_shaped_tensor) => {
                        let old_shape = data_input_shaped_tensor.shape().iter().map(|&x| x as u64).collect::<Vec<_>>();
                        let mut new_shape = vec![];
                        let mut old_i = 0;
                        for i in 0..old_shape.len()+axes.len() {
                            let mut insert_here = false;
                            for axis in &axes {
                                let axis = if *axis < 0 {
                                    old_shape.len() as i64 + axis
                                } else {
                                    *axis
                                };
                                if axis == i as i64 {
                                    insert_here = true;
                                    break;
                                }
                            }
                            if insert_here {
                                new_shape.push(1)
                            }
                            else {
                                new_shape.push(old_shape[old_i] as usize);
                                old_i += 1;
                            }
                        }
                        Ok(TensorInfo::Shaped(data_input_shaped_tensor.reshape(new_shape)))
                    }
                    TensorInfo::Ranked(data_input_ranked_tensor) => {
                        let old_shape = data_input_ranked_tensor.shape();
                        let mut new_shape = vec![];
                        let mut old_i = 0;
                        for i in 0..old_shape.len()+axes.len() {
                            let mut insert_here = false;
                            for axis in &axes {
                                let axis = if *axis < 0 {
                                    old_shape.len() as i64 + axis
                                } else {
                                    *axis
                                };
                                if axis == i as i64 {
                                    insert_here = true;
                                    break;
                                }
                            }
                            if insert_here {
                                new_shape.push(ScalarInfoTyped::Numeric(1));
                            }
                            else {
                                new_shape.push(old_shape[old_i].clone());
                                old_i += 1;
                            }
                        }
                        Ok(TensorInfo::Ranked(RankedTensor::new(
                            data_input_ranked_tensor.first_element().clone(),
                            new_shape
                        )))
                    }
                    TensorInfo::Minimal(data_input_minimal_tensor) => {
                        let old_rank = data_input_minimal_tensor.rank();
                        let new_rank = old_rank.add_offset(-(axes.len() as i64));
                        Ok(TensorInfo::Minimal(MinimalTensor::new(
                            data_input_minimal_tensor.first_element().clone(),
                            new_rank
                        )))
                    }
                }
            }
            TensorInfo::Shaped(axes_input_shaped_tensor) => {
                // We can assume that not enough values are resolved to allow for any resolution of the output shape, limiting us to only knowing the rank
                let old_rank = data_input.rank();
                let new_rank = old_rank.add_offset(-(axes_input_shaped_tensor.shape()[0] as i64));
                Ok(TensorInfo::new_from_rank_and_first_value(data_input.first_element().clone(), new_rank, symbolic_resolver))
            }
            TensorInfo::Ranked(axes_input_ranked_tensor) => {
                // We can assume that not enough values are resolved to allow for any resolution of the output shape, limiting us to only knowing the rank
                let old_rank = data_input.rank();
                let new_rank = match &axes_input_ranked_tensor.shape()[0] {
                    ScalarInfoTyped::Numeric(x) => {
                        old_rank.add_offset(-(*x as i64))
                    }
                    ScalarInfoTyped::Symbolic(_) => {
                        ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver))
                    }
                };
                Ok(TensorInfo::new_from_rank_and_first_value(data_input.first_element().clone(), new_rank, symbolic_resolver))
            }
            TensorInfo::Minimal(_) => {
                // We don't even know the number of indexes we will be removing
                Ok(TensorInfo::Minimal(MinimalTensor::new( data_input.first_element().clone(), SymbolicScalarTyped::new(symbolic_resolver))))
            }
        }
    }

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        let axes_ndarray = NDArrayNumericTensor::try_from(inputs[&self.axes].cast(DType::I64, backend)?)?;
        let axes = Vec::<i64>::try_from(axes_ndarray)?;
        if axes.len() > 1 {
            return Err(MilliOpGraphError::InvalidInput("Unsqueeze".to_string()));
        }
        let axis = axes[0];
        let input_shape = inputs[&self.data].shape();
        let axis = if axis >= 0 {
            axis as usize
        } else {
            (input_shape.len() as i64 + axis) as usize
        };
        let output = inputs[&self.data].unsqueeze(axis, backend)?;
        Ok(output)
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

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
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

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(inputs[&self.data].cast(inputs[&self.target_type].dtype()?, backend)?)
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

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
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

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(NumericTensor::gather(&inputs[&self.data], &inputs[&self.indices], self.axis, backend)?)
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

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        let mut resolved_inputs = vec![];
        for input in &self.inputs {
            resolved_inputs.push(&inputs[input]);
        }
        let axis = if self.axis < 0 {
            resolved_inputs[0].shape().len() as i64 + self.axis
        } else {
            self.axis
        } as usize;
        Ok(NumericTensor::concat(resolved_inputs.as_slice(), axis, backend)?)
    }
}

pub(crate) struct MilliOpSplit {
    data: MilliOpGraphTensorId,
    split: Option<MilliOpGraphTensorId>,
    axis: i64,
    num_outputs: Option<usize>,
    output_id: usize,
}

impl MilliOpSplit {
    pub(crate) fn new(    data: MilliOpGraphTensorId,
                          split: Option<MilliOpGraphTensorId>,
                          axis: i64,
                          num_outputs: Option<usize>,
                          output_id: usize
    ) -> Self {
        Self {data, split, axis, num_outputs, output_id}
    }
}

impl MilliOp for MilliOpSplit {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.data]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        let split: Vec<i64> = if let Some(split) = self.split {
            inputs[&split].clone().try_into()?
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

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(inputs[&self.condition].where_op(&inputs[&self.x], &inputs[&self.y], backend)?)
    }
}


pub(crate) enum AnyMilliOp {
    Constant(MilliOpConstant),
    ConstantOfShape(MilliOpConstantOfShape),
    SimpleBinary(MilliOpSimpleBinary),
    MatMul(MilliOpMatMul),
    Pow(MilliOpPow),
    Neg(MilliOpNeg),
    Abs(MilliOpAbs),
    Exp(MilliOpExp),
    Log(MilliOpLog),
    Sqrt(MilliOpSqrt),
    Trig(MilliOpTrig),
    ClampMin(MilliOpClampMin),
    Reciprocal(MilliOpReciprocal),
    NonZero(MilliOpNonZero),
    CumSum(MilliOpCumSum),
    Shape(MilliOpShape),
    Reshape(MilliOpReshape),
    Slice(MilliOpSlice),
    ReduceSum(MilliOpReduceSum),
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
}

impl MilliOp for AnyMilliOp {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        match self {
            AnyMilliOp::Constant(x) => x.get_inputs(),
            AnyMilliOp::ConstantOfShape(x) => x.get_inputs(),
            AnyMilliOp::SimpleBinary(x) => x.get_inputs(),
            AnyMilliOp::Pow(x) => x.get_inputs(),
            AnyMilliOp::Neg(x) => <_ as MilliOp>::get_inputs(x),
            AnyMilliOp::Abs(x) => <_ as MilliOp>::get_inputs(x),
            AnyMilliOp::Exp(x) => <_ as MilliOp>::get_inputs(x),
            AnyMilliOp::Log(x) => <_ as MilliOp>::get_inputs(x),
            AnyMilliOp::Sqrt(x) => <_ as MilliOp>::get_inputs(x),
            AnyMilliOp::Trig(x) => <_ as MilliOp>::get_inputs(x),
            AnyMilliOp::MatMul(x) => x.get_inputs(),
            AnyMilliOp::ClampMin(x) => <_ as MilliOp>::get_inputs(x),
            AnyMilliOp::Reciprocal(x) => <_ as MilliOp>::get_inputs(x),
            AnyMilliOp::NonZero(x) => x.get_inputs(),
            AnyMilliOp::CumSum(x) => x.get_inputs(),
            AnyMilliOp::Shape(x) => x.get_inputs(),
            AnyMilliOp::Reshape(x) => x.get_inputs(),
            AnyMilliOp::Slice(x) => x.get_inputs(),
            AnyMilliOp::ReduceSum(x) => x.get_inputs(),
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
            AnyMilliOp::Range(x) => x.get_inputs()
        }
    }

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, symbolic_resolver: &mut SymbolicResolver, backend: &EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        match self {
            AnyMilliOp::Constant(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::ConstantOfShape(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::SimpleBinary(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Pow(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Neg(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Abs(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Exp(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Log(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Sqrt(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Trig(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::MatMul(x) =>x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::ClampMin(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Reciprocal(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::NonZero(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::CumSum(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Shape(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Reshape(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::Slice(x) => x.infer(known_inputs, symbolic_resolver, backend),
            AnyMilliOp::ReduceSum(x) => x.infer(known_inputs, symbolic_resolver, backend),
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
        }
    }

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        match self {
            AnyMilliOp::Constant(x) => x.eval(inputs, backend),
            AnyMilliOp::ConstantOfShape(x) => x.eval(inputs, backend),
            AnyMilliOp::SimpleBinary(x) => x.eval(inputs, backend),
            AnyMilliOp::Pow(x) => x.eval(inputs, backend),
            AnyMilliOp::Neg(x) => <_ as MilliOp>::eval(x, inputs, backend),
            AnyMilliOp::Abs(x) => <_ as MilliOp>::eval(x, inputs, backend),
            AnyMilliOp::Exp(x) => <_ as MilliOp>::eval(x, inputs, backend),
            AnyMilliOp::Log(x) => <_ as MilliOp>::eval(x, inputs, backend),
            AnyMilliOp::Sqrt(x) => <_ as MilliOp>::eval(x, inputs, backend),
            AnyMilliOp::Trig(x) => <_ as MilliOp>::eval(x, inputs, backend),
            AnyMilliOp::MatMul(x) => x.eval(inputs, backend),
            AnyMilliOp::ClampMin(x) => <_ as MilliOp>::eval(x, inputs, backend),
            AnyMilliOp::Reciprocal(x) => <_ as MilliOp>::eval(x, inputs, backend),
            AnyMilliOp::NonZero(x) => x.eval(inputs, backend),
            AnyMilliOp::CumSum(x) => x.eval(inputs, backend),
            AnyMilliOp::Shape(x) => x.eval(inputs, backend),
            AnyMilliOp::Reshape(x) => x.eval(inputs, backend),
            AnyMilliOp::Slice(x) => x.eval(inputs, backend),
            AnyMilliOp::ReduceSum(x) => x.eval(inputs, backend),
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

    pub(crate) fn eval(&self, inputs: &HashMap<TensorId, NumericTensor>, backend: &EvalBackend) -> Result<HashMap<TensorId, NumericTensor>, MilliOpGraphError> {
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
            intermediate_values.insert(*op_id, self.ops[op_id].eval(&intermediate_values, backend)?);
        }

        let mut outputs = HashMap::new();
        for (a, b) in self.output_map.as_ref().unwrap() {
            outputs.insert(*b, intermediate_values[a].clone());
        }

        Ok(outputs)
    }
}