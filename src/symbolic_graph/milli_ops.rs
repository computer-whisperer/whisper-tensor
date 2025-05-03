use std::collections::HashMap;
use crate::eval_backend::EvalBackend;
use crate::ndarray_backend::NDArrayNumericTensor;
use crate::numeric_tensor::NumericTensor;
use crate::symbolic_graph::TensorId;

#[derive(Debug, thiserror::Error)]
pub enum MilliOpGraphError {
    #[error(transparent)]
    NumericTensorError(#[from] crate::numeric_tensor::NumericTensorError),
    #[error("Unimplemented milli operator: {0}")]
    UnimplementedOperatorError(String),
}

#[derive(Debug, Clone, Copy, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub(crate) struct MilliOpGraphTensorId {inner: usize}

trait MilliOp {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId>;
    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError>;
}

pub(crate) struct MilliOpConstant {
    a: NumericTensor
}

impl MilliOpConstant {
    pub(crate) fn new(a: NumericTensor) -> Self {
        Self { a }
    }
}

impl MilliOp for MilliOpConstant {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![]}

    fn eval(&self, _inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, _backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(self.a.clone())
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

pub(crate) struct MilliOpAddScalar {
    a: MilliOpGraphTensorId,
    b: f32
}

impl MilliOpAddScalar {
    pub(crate) fn new(a: MilliOpGraphTensorId, b: f32) -> Self {
        Self { a, b }
    }
}

impl MilliOp for MilliOpAddScalar {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.a]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(NumericTensor::add_f32(&inputs[&self.a], self.b, backend)?)
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

pub(crate) struct MilliOpMulScalar {
    a: MilliOpGraphTensorId,
    b: f32
}

impl MilliOpMulScalar {
    pub(crate) fn new(a: MilliOpGraphTensorId, b: f32) -> Self {
        Self { a, b }
    }
}

impl MilliOp for MilliOpMulScalar {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.a]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(NumericTensor::mul_f32(&inputs[&self.a], self.b, backend)?)
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


pub(crate) struct MilliOpTanh {
    a: MilliOpGraphTensorId
}

impl MilliOpTanh {
    pub(crate) fn new(a: MilliOpGraphTensorId) -> Self {
        Self {a}
    }
}

impl MilliOp for MilliOpTanh {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.a]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        Ok(inputs[&self.a].tanh(backend)?)
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
    a: MilliOpGraphTensorId,
    axes: Option<MilliOpGraphTensorId>,
    keepdims: bool
}

impl MilliOpReduceSum {
    pub(crate) fn new(a: MilliOpGraphTensorId, axis: Option<MilliOpGraphTensorId>, keepdims: bool) -> Self {
        Self {a, axes: axis, keepdims}
    }
}

impl MilliOp for MilliOpReduceSum {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {vec![self.a]}

    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        let data = &inputs[&self.a];
        let axes = if let Some(axes) = self.axes {
            Vec::<i64>::try_from(inputs[&axes].clone())?
        } else {
            (0i64 .. (data.shape().len() as i64)).into_iter().collect()
        };
        let out = data.reduce_sum(Some(axes), self.keepdims, backend)?;
        Ok(out)
    }
}

pub(crate) enum AnyMilliOp {
    Constant(MilliOpConstant),
    Add(MilliOpAdd),
    Sub(MilliOpSub),
    Mul(MilliOpMul),
    Div(MilliOpDiv),
    MatMul(MilliOpMatMul),
    AddScalar(MilliOpAddScalar),
    MulScalar(MilliOpMulScalar),
    Neg(MilliOpNeg),
    Abs(MilliOpAbs),
    Exp(MilliOpExp),
    Log(MilliOpLog),
    Sqrt(MilliOpSqrt),
    Tanh(MilliOpTanh),
    ClampMin(MilliOpClampMin),
    Reciprocal(MilliOpReciprocal),
    NonZero(MilliOpNonZero),
    CumSum(MilliOpCumSum),
    Shape(MilliOpShape),
    ReduceSum(MilliOpReduceSum)
}

impl MilliOp for AnyMilliOp {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        match self {
            AnyMilliOp::Constant(x) => x.get_inputs(),
            AnyMilliOp::Add(x) => x.get_inputs(),
            AnyMilliOp::Sub(x) => x.get_inputs(),
            AnyMilliOp::Mul(x) => x.get_inputs(),
            AnyMilliOp::Div(x) => x.get_inputs(),
            AnyMilliOp::AddScalar(x) => x.get_inputs(),
            AnyMilliOp::MulScalar(x) => x.get_inputs(),
            AnyMilliOp::Neg(x) => x.get_inputs(),
            AnyMilliOp::Abs(x) => x.get_inputs(),
            AnyMilliOp::Exp(x) => x.get_inputs(),
            AnyMilliOp::Log(x) => x.get_inputs(),
            AnyMilliOp::Sqrt(x) => x.get_inputs(),
            AnyMilliOp::Tanh(x) => x.get_inputs(),
            AnyMilliOp::MatMul(x) => x.get_inputs(),
            AnyMilliOp::ClampMin(x) => x.get_inputs(),
            AnyMilliOp::Reciprocal(x) => x.get_inputs(),
            AnyMilliOp::NonZero(x) => x.get_inputs(),
            AnyMilliOp::CumSum(x) => x.get_inputs(),
            AnyMilliOp::Shape(x) => x.get_inputs(),
            AnyMilliOp::ReduceSum(x) => x.get_inputs(),
        }
    }
    
    fn eval(&self, inputs: &HashMap<MilliOpGraphTensorId, NumericTensor>, backend: &EvalBackend) -> Result<NumericTensor, MilliOpGraphError> {
        match self {
            AnyMilliOp::Constant(x) => x.eval(inputs, backend),
            AnyMilliOp::Add(x) => x.eval(inputs, backend),
            AnyMilliOp::Sub(x) => x.eval(inputs, backend),
            AnyMilliOp::Mul(x) => x.eval(inputs, backend),
            AnyMilliOp::Div(x) => x.eval(inputs, backend),
            AnyMilliOp::Neg(x) => x.eval(inputs, backend),
            AnyMilliOp::Abs(x) => x.eval(inputs, backend),
            AnyMilliOp::AddScalar(x) => x.eval(inputs, backend),
            AnyMilliOp::MulScalar(x) => x.eval(inputs, backend),
            AnyMilliOp::Exp(x) => x.eval(inputs, backend),
            AnyMilliOp::Log(x) => x.eval(inputs, backend),
            AnyMilliOp::Sqrt(x) => x.eval(inputs, backend),
            AnyMilliOp::Tanh(x) => x.eval(inputs, backend),
            AnyMilliOp::MatMul(x) => x.eval(inputs, backend),
            AnyMilliOp::ClampMin(x) => x.eval(inputs, backend),
            AnyMilliOp::Reciprocal(x) => x.eval(inputs, backend),
            AnyMilliOp::NonZero(x) => x.eval(inputs, backend),
            AnyMilliOp::CumSum(x) => x.eval(inputs, backend),
            AnyMilliOp::Shape(x) => x.eval(inputs, backend),
            AnyMilliOp::ReduceSum(x) => x.eval(inputs, backend)
        }
    }
}

pub(crate) struct MilliOpGraph {
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