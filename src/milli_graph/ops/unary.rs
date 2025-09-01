use crate::backends::eval_backend::EvalBackend;
use crate::milli_graph::ops::MilliOp;
use crate::milli_graph::{MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_scalar::NumericScalar;
use crate::numeric_tensor::NumericTensor;
use crate::scalar_info::{ScalarInfo, ScalarInfoTyped};
use crate::symbolic_scalar::{SymbolicResolver, SymbolicScalar};
use crate::tensor_info::{TensorInfo, TensorInfoRanked, TensorInfoShaped};
use crate::{DynRank, TrigOp};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

trait SimpleUnaryMilliOp {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId>;

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError>;

    #[allow(dead_code)]
    fn eval_scalar(&self, input: &NumericScalar) -> Result<NumericScalar, MilliOpGraphError>;

    fn get_name(&self) -> String;
}

impl<T: SimpleUnaryMilliOp> MilliOp for T {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        <T as SimpleUnaryMilliOp>::get_inputs(self)
    }

    fn infer(
        &self,
        known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>,
        symbolic_resolver: &mut SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<TensorInfo, MilliOpGraphError> {
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
                    Ok(TensorInfo::from(TensorInfoRanked::<DynRank>::new(
                        ScalarInfo::Symbolic(SymbolicScalar::new(input.dtype(), symbolic_resolver)),
                        input
                            .shape()
                            .iter()
                            .map(|x| ScalarInfoTyped::Numeric(*x))
                            .collect::<Vec<_>>(),
                        symbolic_resolver,
                    )))
                }
            }
        } else {
            // For now, just decay this to a ranked tensor
            Ok(TensorInfo::new_from_first_element_and_shape(
                ScalarInfo::Symbolic(SymbolicScalar::new(input.dtype(), symbolic_resolver)),
                input.shape(symbolic_resolver),
                symbolic_resolver,
            ))
        }
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        <T as SimpleUnaryMilliOp>::eval(self, inputs, backend)
    }

    fn get_name(&self) -> String {
        <T as SimpleUnaryMilliOp>::get_name(self)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    IsInf {
        detect_positive: bool,
        detect_negative: bool,
    },
    Erf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpSimpleUnary {
    input: MilliOpGraphTensorId,
    op: SimpleUnaryOp,
}

impl MilliOpSimpleUnary {
    pub(crate) fn new(input: MilliOpGraphTensorId, op: SimpleUnaryOp) -> Self {
        Self { input, op }
    }

    pub fn neg(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Neg)
    }

    pub fn abs(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Abs)
    }

    pub fn exp(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Exp)
    }

    pub fn ln(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Ln)
    }

    pub fn sqrt(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Sqrt)
    }

    pub fn not(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Not)
    }

    pub fn sign(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Sign)
    }

    pub fn bitwise_not(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::BitwiseNot)
    }

    pub fn reciprocal(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Reciprocal)
    }

    pub fn trig(input: MilliOpGraphTensorId, trig_op: TrigOp) -> Self {
        Self::new(input, SimpleUnaryOp::Trig(trig_op))
    }

    pub fn floor(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Floor)
    }

    pub fn ceil(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Ceil)
    }

    pub fn round(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Round)
    }

    pub fn is_inf(
        input: MilliOpGraphTensorId,
        detect_positive: bool,
        detect_negative: bool,
    ) -> Self {
        Self::new(
            input,
            SimpleUnaryOp::IsInf {
                detect_positive,
                detect_negative,
            },
        )
    }

    pub fn is_nan(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::IsNan)
    }

    pub fn erf(input: MilliOpGraphTensorId) -> Self {
        Self::new(input, SimpleUnaryOp::Erf)
    }
}

impl SimpleUnaryMilliOp for MilliOpSimpleUnary {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.input]
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
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
            SimpleUnaryOp::IsInf {
                detect_positive,
                detect_negative,
            } => Ok(input.is_inf(detect_positive, detect_negative)?),
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
            SimpleUnaryOp::IsInf {
                detect_positive,
                detect_negative,
            } => Ok(input.is_inf(detect_positive, detect_negative)),
            SimpleUnaryOp::IsNan => Ok(input.is_nan()),
            SimpleUnaryOp::Erf => Ok(input.erf()),
        }
    }

    fn get_name(&self) -> String {
        match self.op {
            SimpleUnaryOp::Neg => "Neg",
            SimpleUnaryOp::Abs => "Abs",
            SimpleUnaryOp::Exp => "Exp",
            SimpleUnaryOp::Ln => "Ln",
            SimpleUnaryOp::Sqrt => "Sqrt",
            SimpleUnaryOp::Not => "Not",
            SimpleUnaryOp::Sign => "Sign",
            SimpleUnaryOp::BitwiseNot => "Bitwise Not",
            SimpleUnaryOp::Reciprocal => "Reciprocal",
            SimpleUnaryOp::Trig(trig_op) => trig_op.get_name(),
            SimpleUnaryOp::Floor => "Floor",
            SimpleUnaryOp::Ceil => "Ceil",
            SimpleUnaryOp::Round => "Round",
            SimpleUnaryOp::IsNan => "IsNan",
            SimpleUnaryOp::IsInf { .. } => "IsInf",
            SimpleUnaryOp::Erf => "Erf",
        }
        .to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpClampMin {
    input: MilliOpGraphTensorId,
    value: f32,
}

impl MilliOpClampMin {
    pub fn new(a: MilliOpGraphTensorId, value: f32) -> Self {
        Self { input: a, value }
    }
}

impl SimpleUnaryMilliOp for MilliOpClampMin {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.input]
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        Ok(inputs[&self.input].clamp_min(self.value, backend)?)
    }

    fn eval_scalar(&self, input: &NumericScalar) -> Result<NumericScalar, MilliOpGraphError> {
        Ok(input.clamp_min(self.value))
    }

    fn get_name(&self) -> String {
        "Clamp Min".to_string()
    }
}
