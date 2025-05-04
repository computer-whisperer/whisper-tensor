use std::collections::HashMap;
use crate::dtype::{DType, DTypeError};
use crate::eval_backend::EvalBackend;
use crate::ndarray_backend::NDArrayNumericTensor;
use crate::numeric_scalar::NumericScalar;
use crate::numeric_tensor::NumericTensor;
use crate::symbolic_graph::{Dimension, DimensionResolver, TensorId};
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

#[derive(Debug, Clone)]
pub(crate) struct TensorMetadata {
    dtype: DType,
    shape: Option<Vec<Dimension>>
}

pub(crate) enum TensorInfo {
    Value(NumericTensor),
    Meta(TensorMetadata),
}

impl TensorInfo {
    fn dtype(&self) -> DType {
        match self {
            Self::Value(tensor) => tensor.dtype().unwrap(),
            Self::Meta(info) => info.dtype
        }
    }
    fn shape(&self) -> Option<Vec<Dimension>> {
        match self {
            TensorInfo::Value(x) => {
                Some(x.shape().iter().map(|x| Dimension::Known(*x)).collect())
            }
            TensorInfo::Meta(x) => {
                x.shape.clone()
            }
        }
    }
}

trait MilliOp {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId>;
    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, _dimension_resolver: &mut DimensionResolver, backend: &EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let mut resolved_inputs = HashMap::new();
        for input in self.get_inputs() {
            if let Some(tensor_info) = known_inputs.get(&input) {
                if let TensorInfo::Value(value) = &tensor_info {
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

        Ok(TensorInfo::Value(self.eval(&resolved_inputs, backend)?))
    }
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
        NDArrayNumericTensor: From<Vec<T>>
    {
        Self{
            data: NDArrayNumericTensor::from(vec![v]).into()
        }
    }
}

impl MilliOp for MilliOpConstant {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![]}

    fn infer(&self, _known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, _dimension_resolver: &mut DimensionResolver, _backend: &EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        Ok(TensorInfo::Value(self.data.clone()))
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

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, dimension_resolver: &mut DimensionResolver, backend: &EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let start = &known_inputs[&self.start];
        let end = &known_inputs[&self.end];
        let delta = &known_inputs[&self.delta];
        assert_eq!(start.dtype(), end.dtype());
        assert_eq!(start.dtype(), end.dtype());
        Ok(if let (
            TensorInfo::Value(start), 
            TensorInfo::Value(end), 
            TensorInfo::Value(delta)) =
            (start, end, delta) {
            TensorInfo::Value(NumericTensor::range(
                start.to_scalar()?,
                end.to_scalar()?,
                delta.to_scalar()?,
                backend
            )?)
        } 
        else {
            TensorInfo::Meta(TensorMetadata{
                dtype: start.dtype(),
                shape: Some(vec![Dimension::Unknown(dimension_resolver.new_unknown())])
            })
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

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, dimension_resolver: &mut DimensionResolver, _backend: &EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        match &known_inputs[&self.shape] {
            TensorInfo::Value(value) => {
                let shape: Vec<i64> = value.clone().try_into()?;
                let shape_usize = shape.iter().map(|x| *x as usize).collect::<Vec<usize>>();
                let out = NDArrayNumericTensor::fill(self.value.clone(), shape_usize.as_slice())?.into();
                Ok(TensorInfo::Value(out))
            }
            TensorInfo::Meta(meta) => {
                let shape = if let Some(shape) = &meta.shape {
                    if let Dimension::Known(dim) = shape[0] {
                        let mut v = Vec::new();
                        v.resize(dim, Dimension::Unknown(dimension_resolver.new_unknown()));
                        Some(v)
                    }
                    else {
                        None
                    }
                }
                else {
                    None
                };
                Ok(TensorInfo::Meta(TensorMetadata { 
                    shape,
                    dtype: meta.dtype
                }))
            }
        }
    }

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, _backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        let shape: Vec<i64> = inputs[&self.shape].clone().try_into()?;
        let shape_usize = shape.iter().map(|x| *x as usize).collect::<Vec<usize>>();
        Ok(NDArrayNumericTensor::fill(self.value.clone(), shape_usize.as_slice())?.into())
    }
}

fn infer_multidirectional_broadcasting(shapes: &[&[Dimension]], dimension_resolver: &mut DimensionResolver) -> Result<Vec<Dimension>, MilliOpGraphError> {
    if shapes.is_empty() {
        return Err(MilliOpGraphError::InvalidInput("Cannot broadcast empty input".to_string()));
    }
    let output_rank = shapes.iter().map(|x| x.len()).max().unwrap();
    let mut output_shape = vec![];
    for i in 0..output_rank {
        let mut dim = Dimension::Known(1);
        for shape in shapes {
            let local_i = (i as i64 - output_rank as i64) + shape.len() as i64;
            if local_i < 0 {
                // Infer dim of size 1, and pass
            } else {
                let local_dim = &shape[local_i as usize];
                match local_dim {
                    Dimension::Known(x) => {
                        if *x == 1 {
                            // Do not modify the dimension, pass it through.
                        }
                        else {
                            match dim {
                                Dimension::Known(y) => {
                                    if y == 1 || *x == y {
                                        dim = Dimension::Known(y.max(*x));
                                    } else {
                                        return Err(MilliOpGraphError::InvalidInput("Cannot broadcast input shape".to_string()));
                                    }
                                }
                                _ => {
                                    // The only way this is valid is if the unknown dim matches the known one, so be optimistic here
                                    dim = *local_dim;
                                }
                            }
                        }
                    }
                    _ => {
                        // Incoming dim is unknown
                        match dim {
                            Dimension::Known(y) => {
                                if y == 1 {
                                    // Use the existing unknown dim
                                    dim = *local_dim;
                                } else {
                                    // The only way this is valid is if the unknown dim matches the known one, so be optimistic here
                                }
                            }
                            _ => {
                                // Two unknown dimensions
                                match dim.try_test_eq(local_dim) {
                                    None => {
                                        // Must use new unknown dimension
                                        dim = Dimension::Unknown(dimension_resolver.new_unknown())
                                    }
                                    Some(is_same) => {
                                        if is_same {
                                            // Ok, use the unknown dim already in there
                                        }
                                        else {
                                            // Must use new unknown dimension
                                            dim = Dimension::Unknown(dimension_resolver.new_unknown())
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

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, dimension_resolver: &mut DimensionResolver, backend: &EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let a = &known_inputs[&self.a];
        let b = &known_inputs[&self.b];
        if let (TensorInfo::Value(a), TensorInfo::Value(b)) = (a, b) {
            Ok(TensorInfo::Value(
                match self.which_op {
                    SimpleBinaryOp::Add => NumericTensor::add(a, b, backend)?,
                    SimpleBinaryOp::Sub => NumericTensor::sub(a, b, backend)?,
                    SimpleBinaryOp::Mul => NumericTensor::mul(a, b, backend)?,
                    SimpleBinaryOp::Div => NumericTensor::div(a, b, backend)?,
                }
            ))
        }
        else {
            let dtype = a.dtype();
            assert_eq!(b.dtype(), dtype);
            let shape_a = a.shape();
            let shape_b = b.shape();
            let shape = if let (Some(a), Some(b)) = (shape_a, shape_b) {
                Some(infer_multidirectional_broadcasting(&[a.as_slice(), b.as_slice()], dimension_resolver)?)
            } else {
                // Need to know the rank of at least one of the inputs
                None
            };

            Ok(TensorInfo::Meta(TensorMetadata{
                dtype,
                shape
            }))
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

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, dimension_resolver: &mut DimensionResolver, backend: &EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let a = &known_inputs[&self.a];
        let b = &known_inputs[&self.b];
        if let (TensorInfo::Value(a), TensorInfo::Value(b)) = (a, b) {
            Ok(TensorInfo::Value(
                NumericTensor::pow(a, b, backend)?,
            ))
        }
        else {
            let dtype = a.dtype();
            let shape_a = a.shape();
            let shape_b = b.shape();
            let shape = if let (Some(a), Some(b)) = (shape_a, shape_b) {
                Some(infer_multidirectional_broadcasting(&[a.as_slice(), b.as_slice()], dimension_resolver)?)
            } else {
                // Need to know the rank of at least one of the inputs
                None
            };

            Ok(TensorInfo::Meta(TensorMetadata{
                dtype,
                shape
            }))
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

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, dimension_resolver: &mut DimensionResolver, backend: &EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let a = &known_inputs[&self.a];
        let b = &known_inputs[&self.b];
        if let (TensorInfo::Value(a), TensorInfo::Value(b)) = (a, b) {
            Ok(TensorInfo::Value(
                NumericTensor::matmul(a, b, backend)?,
            ))
        }
        else {
            let dtype = a.dtype();
            assert_eq!(b.dtype(), dtype);
            let shape_a = a.shape();
            let shape_b = b.shape();

            let shape = if let (Some(shape_a), Some(shape_b)) = (shape_a, shape_b) {
                // Prepend to a if rank 1
                let (mut shape_a, prune_first_after) = if shape_a.len() == 1 {
                    (vec![Dimension::Known(1), shape_a[0]], true)
                } else {
                    (shape_a, false)
                };

                // Append to b if rank 1
                let (mut shape_b, prune_last_after) = if shape_b.len() == 1 {
                    (vec![shape_b[0], Dimension::Known(1)], true)
                } else {
                    (shape_b, false)
                };

                // Broadcast both shapes
                while shape_a.len() < shape_b.len() {
                    shape_a.insert(0, Dimension::Known(1))
                }
                while shape_b.len() < shape_a.len() {
                    shape_b.insert(0, Dimension::Known(1))
                }

                let mut dims_out = vec![];
                for i in 0..shape_a.len()-2 {
                    let dim = match shape_a[i] {
                        Dimension::Known(x) => {
                            if x == 1 {
                                // Use the other one
                                shape_b[i]
                            }
                            else {
                                match shape_b[i] {
                                    Dimension::Known(y) => {
                                        Dimension::Known(x.max(y))
                                    }
                                    Dimension::Unknown(_) => {
                                        // Assume it's the known one
                                        Dimension::Known(x)
                                    }
                                }
                            }
                        }
                        Dimension::Unknown(x) => {
                            match shape_b[i] {
                                Dimension::Known(y) => {
                                    // Assume it's the known one
                                    Dimension::Known(y)
                                }
                                Dimension::Unknown(y) => {
                                    match x.try_test_eq(&y) {
                                        None => {
                                            // Can't compare them, must use a new unknown
                                            Dimension::Unknown(dimension_resolver.new_unknown())
                                        }
                                        Some(is_same) => {
                                            if is_same {
                                                // They are the same dimension, so we can use it
                                                Dimension::Unknown(x)
                                            }
                                            else {
                                                // They are different dimensions, so we can't use it
                                                Dimension::Unknown(dimension_resolver.new_unknown())
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    };
                    dims_out.push(dim);
                }
                dims_out.push(shape_a[shape_a.len() - 2]);
                dims_out.push(shape_b[shape_b.len() - 1]);

                if prune_first_after {
                    dims_out.remove(0);
                }
                if prune_last_after {
                    dims_out.pop();
                }

                Some(dims_out)
            } else {
                None
            };

            Ok(TensorInfo::Meta(TensorMetadata{
                dtype,
                shape
            }))
        }
    }
    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(NumericTensor::matmul(&inputs[&self.a], &inputs[&self.b], backend)?)
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

impl MilliOp for MilliOpNeg {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.input]}

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, _dimension_resolver: &mut DimensionResolver, backend: &EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let input = &known_inputs[&self.input];
        if let TensorInfo::Value(a) = input {
            Ok(TensorInfo::Value(a.neg(backend)?))
        }
        else {
            Ok(TensorInfo::Meta(TensorMetadata{
                dtype: input.dtype(),
                shape: input.shape()
            }))
        }
    }

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        let input = &inputs[&self.input];
        Ok(input.neg(backend)?)
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

impl MilliOp for MilliOpAbs {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.input]}

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, _dimension_resolver: &mut DimensionResolver, backend: &EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let input = &known_inputs[&self.input];
        if let TensorInfo::Value(a) = input {
            Ok(TensorInfo::Value(a.abs(backend)?))
        }
        else {
            Ok(TensorInfo::Meta(TensorMetadata{
                dtype: input.dtype(),
                shape: input.shape()
            }))
        }
    }

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(inputs[&self.input].abs(backend)?)
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

impl MilliOp for MilliOpExp {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.input]}

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, _dimension_resolver: &mut DimensionResolver, backend: &EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let input = &known_inputs[&self.input];
        if let TensorInfo::Value(a) = input {
            Ok(TensorInfo::Value(a.exp(backend)?))
        }
        else {
            Ok(TensorInfo::Meta(TensorMetadata{
                dtype: input.dtype(),
                shape: input.shape()
            }))
        }
    }

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(inputs[&self.input].exp(backend)?)
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

impl MilliOp for MilliOpLog {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.input]}

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, _dimension_resolver: &mut DimensionResolver, backend: &EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let input = &known_inputs[&self.input];
        if let TensorInfo::Value(a) = input {
            Ok(TensorInfo::Value(a.log(backend)?))
        }
        else {
            Ok(TensorInfo::Meta(TensorMetadata{
                dtype: input.dtype(),
                shape: input.shape()
            }))
        }
    }

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(inputs[&self.input].log(backend)?)
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

impl MilliOp for MilliOpTrig {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.input]}

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, _dimension_resolver: &mut DimensionResolver, backend: &EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let input = &known_inputs[&self.input];
        if let TensorInfo::Value(a) = input {
            Ok(TensorInfo::Value(a.trig(self.op, backend)?))
        }
        else {
            Ok(TensorInfo::Meta(TensorMetadata{
                dtype: input.dtype(),
                shape: input.shape()
            }))
        }
    }

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(inputs[&self.input].trig(self.op, backend)?)
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

impl MilliOp for MilliOpSqrt {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.input]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(inputs[&self.input].sqrt(backend)?)
    }

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, _dimension_resolver: &mut DimensionResolver, backend: &EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let input = &known_inputs[&self.input];
        if let TensorInfo::Value(a) = input {
            Ok(TensorInfo::Value(a.sqrt(backend)?))
        }
        else {
            Ok(TensorInfo::Meta(TensorMetadata{
                dtype: input.dtype(),
                shape: input.shape()
            }))
        }
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

impl MilliOp for MilliOpReciprocal {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.input]}

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, _dimension_resolver: &mut DimensionResolver, backend: &EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let input = &known_inputs[&self.input];
        if let TensorInfo::Value(a) = input {
            Ok(TensorInfo::Value(a.reciprocal(backend)?))
        }
        else {
            Ok(TensorInfo::Meta(TensorMetadata{
                dtype: input.dtype(),
                shape: input.shape()
            }))
        }
    }

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(inputs[&self.input].reciprocal(backend)?)
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

impl MilliOp for MilliOpClampMin {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.input]}

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, _dimension_resolver: &mut DimensionResolver, backend: &EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let input = &known_inputs[&self.input];
        if let TensorInfo::Value(a) = input {
            Ok(TensorInfo::Value(a.clamp_min(self.value, backend)?))
        }
        else {
            Ok(TensorInfo::Meta(TensorMetadata{
                dtype: input.dtype(),
                shape: input.shape()
            }))
        }
    }

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(inputs[&self.input].clamp_min(self.value, backend)?)
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

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, _dimension_resolver: &mut DimensionResolver, backend: &EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let input = &known_inputs[&self.input];
        if let TensorInfo::Value(a) = input {
            Ok(TensorInfo::Value(a.nonzero(backend)?))
        }
        else {
            // Don't even try shape inference for now
            Ok(TensorInfo::Meta(TensorMetadata{
                dtype: DType::I64,
                shape: None
            }))
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

    fn infer(&self, known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>, dimension_resolver: &mut DimensionResolver, _backend: &EvalBackend) -> Result<TensorInfo, MilliOpGraphError> {
        let input = &known_inputs[&self.input];
        match input.shape() {
            None => Ok(TensorInfo::Meta(
                TensorMetadata{
                    dtype: DType::I64,
                    shape: Some(vec![Dimension::Unknown(dimension_resolver.new_unknown())])
                }
            )),
            Some(shape) => {
                let mut numeric_shape = vec![];
                let mut cannot_resolve = false;
                for dim in &shape {
                    match dim {
                        Dimension::Unknown(_) => {
                            cannot_resolve = true;
                            break;
                        },
                        Dimension::Known(d) => numeric_shape.push(*d as i64)
                    }
                }
                if cannot_resolve {
                    Ok(TensorInfo::Meta(
                        TensorMetadata{
                            dtype: DType::I64,
                            shape: Some(vec![Dimension::Known(shape.len())])
                        }
                    ))
                }
                else {
                    Ok(TensorInfo::Value(
                        NDArrayNumericTensor::from(numeric_shape).into()
                    ))
                }
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
}

impl MilliOp for MilliOpReshape {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.data, self.shape]
    }

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        let data_input = &inputs[&self.data];
        let shape_input = &inputs[&self.shape];
        let data_input_shape = data_input.shape();
        let shape_input_value: Vec<i64> = shape_input.cast(DType::I64, backend)?.try_into()?;
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
                shape_input_value[i] as usize
            });
        }

        // Backfill the inferred dimension
        if let Some(i) = backfill_dim {
            let total_input_size = data_input.shape().iter().product::<usize>();

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
        if output_shape.iter().product::<usize>() != data_input.shape().iter().product::<usize>() {
            Err(MilliOpGraphError::InvalidInput("Reshape".to_string()))?
        }

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
            AnyMilliOp::Neg(x) => x.get_inputs(),
            AnyMilliOp::Abs(x) => x.get_inputs(),
            AnyMilliOp::Exp(x) => x.get_inputs(),
            AnyMilliOp::Log(x) => x.get_inputs(),
            AnyMilliOp::Sqrt(x) => x.get_inputs(),
            AnyMilliOp::Trig(x) => x.get_inputs(),
            AnyMilliOp::MatMul(x) => x.get_inputs(),
            AnyMilliOp::ClampMin(x) => x.get_inputs(),
            AnyMilliOp::Reciprocal(x) => x.get_inputs(),
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

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        match self {
            AnyMilliOp::Constant(x) => x.eval(inputs, backend),
            AnyMilliOp::ConstantOfShape(x) => x.eval(inputs, backend),
            AnyMilliOp::SimpleBinary(x) => x.eval(inputs, backend),
            AnyMilliOp::Pow(x) => x.eval(inputs, backend),
            AnyMilliOp::Neg(x) => x.eval(inputs, backend),
            AnyMilliOp::Abs(x) => x.eval(inputs, backend),
            AnyMilliOp::Exp(x) => x.eval(inputs, backend),
            AnyMilliOp::Log(x) => x.eval(inputs, backend),
            AnyMilliOp::Sqrt(x) => x.eval(inputs, backend),
            AnyMilliOp::Trig(x) => x.eval(inputs, backend),
            AnyMilliOp::MatMul(x) => x.eval(inputs, backend),
            AnyMilliOp::ClampMin(x) => x.eval(inputs, backend),
            AnyMilliOp::Reciprocal(x) => x.eval(inputs, backend),
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