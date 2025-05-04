use std::collections::HashMap;
use crate::dtype::{DType, DTypeError};
use crate::eval_backend::EvalBackend;
use crate::ndarray_backend::NDArrayNumericTensor;
use crate::numeric_scalar::NumericScalar;
use crate::numeric_tensor::NumericTensor;
use crate::symbolic_graph::{TensorId};
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
    DTypeError(#[from] DTypeError)
}

#[derive(Debug, Clone, Copy, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub(crate) struct MilliOpGraphTensorId {inner: usize}

trait MilliOp {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId>;
    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError>;
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

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, _backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        let shape: Vec<i64> = inputs[&self.shape].clone().try_into()?;
        let shape_usize = shape.iter().map(|x| *x as usize).collect::<Vec<usize>>();
        Ok(NDArrayNumericTensor::fill(self.value.clone(), shape_usize.as_slice())?.into())
    }
}


pub(crate) struct MilliOpAdd {
    a: MilliOpGraphTensorId,
    b: MilliOpGraphTensorId
}

impl MilliOpAdd {
    pub(crate) fn new(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self { a, b }
    }
}

impl MilliOp for MilliOpAdd {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.a, self.b]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(NumericTensor::add(&inputs[&self.a], &inputs[&self.b], backend)?)
    }
}

pub(crate) struct MilliOpSub {
    a: MilliOpGraphTensorId,
    b: MilliOpGraphTensorId
}

impl MilliOpSub {
    pub(crate) fn new(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self { a, b }
    }
}

impl MilliOp for MilliOpSub {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.a, self.b]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(NumericTensor::sub(&inputs[&self.a], &inputs[&self.b], backend)?)
    }
}

pub(crate) struct MilliOpMul {
    a: MilliOpGraphTensorId,
    b: MilliOpGraphTensorId
}

impl MilliOpMul {
    pub(crate) fn new(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self { a, b }
    }
}

impl MilliOp for MilliOpMul {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.a, self.b]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(NumericTensor::mul(&inputs[&self.a], &inputs[&self.b], backend)?)
    }
}

pub(crate) struct MilliOpDiv {
    a: MilliOpGraphTensorId,
    b: MilliOpGraphTensorId
}

impl MilliOpDiv {
    pub(crate) fn new(a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> Self {
        Self { a, b }
    }
}

impl MilliOp for MilliOpDiv {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.a, self.b]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(NumericTensor::div(&inputs[&self.a], &inputs[&self.b], backend)?)
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

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(NumericTensor::matmul(&inputs[&self.a], &inputs[&self.b], backend)?)
    }
}


pub(crate) struct MilliOpNeg {
    a: MilliOpGraphTensorId
}

impl MilliOpNeg {
    pub(crate) fn new(a: MilliOpGraphTensorId) -> Self {
        Self {a}
    }
}

impl MilliOp for MilliOpNeg {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.a]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(inputs[&self.a].neg(backend)?)
    }
}

pub(crate) struct MilliOpAbs {
    a: MilliOpGraphTensorId
}

impl MilliOpAbs {
    pub(crate) fn new(a: MilliOpGraphTensorId) -> Self {
        Self {a}
    }
}

impl MilliOp for MilliOpAbs {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.a]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(inputs[&self.a].abs(backend)?)
    }
}

pub(crate) struct MilliOpExp {
    a: MilliOpGraphTensorId
}

impl MilliOpExp {
    pub(crate) fn new(a: MilliOpGraphTensorId) -> Self {
        Self {a}
    }
}

impl MilliOp for MilliOpExp {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.a]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(inputs[&self.a].exp(backend)?)
    }
}

pub(crate) struct MilliOpLog {
    a: MilliOpGraphTensorId
}

impl MilliOpLog {
    pub(crate) fn new(a: MilliOpGraphTensorId) -> Self {
        Self {a}
    }
}

impl MilliOp for MilliOpLog {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.a]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(inputs[&self.a].log(backend)?)
    }
}


pub(crate) struct MilliOpTrig {
    a: MilliOpGraphTensorId,
    op: TrigOp
}

impl MilliOpTrig {
    pub(crate) fn new(a: MilliOpGraphTensorId, op: TrigOp) -> Self {
        Self {a, op}
    }
}

impl MilliOp for MilliOpTrig {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.a]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(inputs[&self.a].trig(self.op, backend)?)
    }
}

pub(crate) struct MilliOpSqrt {
    a: MilliOpGraphTensorId
}

impl MilliOpSqrt {
    pub(crate) fn new(a: MilliOpGraphTensorId) -> Self {
        Self {a}
    }
}

impl MilliOp for MilliOpSqrt {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.a]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(inputs[&self.a].sqrt(backend)?)
    }
}


pub(crate) struct MilliOpReciprocal {
    a: MilliOpGraphTensorId
}

impl MilliOpReciprocal {
    pub(crate) fn new(a: MilliOpGraphTensorId) -> Self {
        Self {a}
    }
}

impl MilliOp for MilliOpReciprocal {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.a]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(inputs[&self.a].reciprocal(backend)?)
    }
}

pub(crate) struct MilliOpClampMin {
    a: MilliOpGraphTensorId,
    value: f32
}

impl MilliOpClampMin {
    pub(crate) fn new(a: MilliOpGraphTensorId, value: f32) -> Self {
        Self {a, value}
    }
}

impl MilliOp for MilliOpClampMin {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.a]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(inputs[&self.a].clamp_min(self.value, backend)?)
    }
}

pub(crate) struct MilliOpNonZero {
    a: MilliOpGraphTensorId,
}

impl MilliOpNonZero {
    pub(crate) fn new(a: MilliOpGraphTensorId) -> Self {
        Self {a}
    }
}

impl MilliOp for MilliOpNonZero {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.a]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(inputs[&self.a].nonzero(backend)?)
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
    a: MilliOpGraphTensorId
}

impl MilliOpShape {
    pub(crate) fn new(a: MilliOpGraphTensorId) -> Self {
        Self {a}
    }
}

impl MilliOp for MilliOpShape {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.a]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, _backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        let output_shape = inputs[&self.a].shape().into_iter().map(|x| x as i64).collect::<Vec<_>>();
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
    Add(MilliOpAdd),
    Sub(MilliOpSub),
    Mul(MilliOpMul),
    Div(MilliOpDiv),
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
            AnyMilliOp::Add(x) => x.get_inputs(),
            AnyMilliOp::Sub(x) => x.get_inputs(),
            AnyMilliOp::Mul(x) => x.get_inputs(),
            AnyMilliOp::Div(x) => x.get_inputs(),
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
            AnyMilliOp::Add(x) => x.eval(inputs, backend),
            AnyMilliOp::Sub(x) => x.eval(inputs, backend),
            AnyMilliOp::Mul(x) => x.eval(inputs, backend),
            AnyMilliOp::Div(x) => x.eval(inputs, backend),
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