use crate::backends::eval_backend::EvalBackend;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use crate::{DynRank, TrigOp};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::graph::Node;

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
    output: MilliOpGraphTensorId,
    input: MilliOpGraphTensorId,
    op: SimpleUnaryOp,
}

impl MilliOpSimpleUnary {
    fn new_internal<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, input: MilliOpGraphTensorId, op: SimpleUnaryOp) -> MilliOpGraphTensorId {
        let output = graph.get_new_tensor_id();
        let node = Self { output, input, op };
        graph.push_op(AnyMilliOp::SimpleUnary(node));
        output
    }

    pub fn neg<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, input: MilliOpGraphTensorId) -> MilliOpGraphTensorId { Self::new_internal(graph, input, SimpleUnaryOp::Neg) }
    pub fn abs<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, input: MilliOpGraphTensorId) -> MilliOpGraphTensorId { Self::new_internal(graph, input, SimpleUnaryOp::Abs) }
    pub fn exp<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, input: MilliOpGraphTensorId) -> MilliOpGraphTensorId { Self::new_internal(graph, input, SimpleUnaryOp::Exp) }
    pub fn ln<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, input: MilliOpGraphTensorId) -> MilliOpGraphTensorId { Self::new_internal(graph, input, SimpleUnaryOp::Ln) }
    pub fn sqrt<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, input: MilliOpGraphTensorId) -> MilliOpGraphTensorId { Self::new_internal(graph, input, SimpleUnaryOp::Sqrt) }
    pub fn not<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, input: MilliOpGraphTensorId) -> MilliOpGraphTensorId { Self::new_internal(graph, input, SimpleUnaryOp::Not) }
    pub fn sign<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, input: MilliOpGraphTensorId) -> MilliOpGraphTensorId { Self::new_internal(graph, input, SimpleUnaryOp::Sign) }
    pub fn bitwise_not<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, input: MilliOpGraphTensorId) -> MilliOpGraphTensorId { Self::new_internal(graph, input, SimpleUnaryOp::BitwiseNot) }
    pub fn reciprocal<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, input: MilliOpGraphTensorId) -> MilliOpGraphTensorId { Self::new_internal(graph, input, SimpleUnaryOp::Reciprocal) }
    pub fn trig<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, input: MilliOpGraphTensorId, trig_op: TrigOp) -> MilliOpGraphTensorId { Self::new_internal(graph, input, SimpleUnaryOp::Trig(trig_op)) }
    pub fn floor<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, input: MilliOpGraphTensorId) -> MilliOpGraphTensorId { Self::new_internal(graph, input, SimpleUnaryOp::Floor) }
    pub fn ceil<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, input: MilliOpGraphTensorId) -> MilliOpGraphTensorId { Self::new_internal(graph, input, SimpleUnaryOp::Ceil) }
    pub fn round<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, input: MilliOpGraphTensorId) -> MilliOpGraphTensorId { Self::new_internal(graph, input, SimpleUnaryOp::Round) }
    pub fn is_inf<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, input: MilliOpGraphTensorId, detect_positive: bool, detect_negative: bool) -> MilliOpGraphTensorId { Self::new_internal(graph, input, SimpleUnaryOp::IsInf{detect_positive, detect_negative}) }
    pub fn is_nan<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, input: MilliOpGraphTensorId) -> MilliOpGraphTensorId { Self::new_internal(graph, input, SimpleUnaryOp::IsNan) }
    pub fn erf<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, input: MilliOpGraphTensorId) -> MilliOpGraphTensorId { Self::new_internal(graph, input, SimpleUnaryOp::Erf) }
}

impl Node<MilliOpGraphTensorId> for MilliOpSimpleUnary {
    fn inputs(&self) -> impl Iterator<Item=MilliOpGraphTensorId> { vec![self.input].into_iter() }
    fn outputs(&self) -> impl Iterator<Item=MilliOpGraphTensorId> { vec![self.output].into_iter() }
}

impl MilliOp for MilliOpSimpleUnary {
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<impl Iterator<Item=(MilliOpGraphTensorId, NumericTensor<DynRank>)>, MilliOpGraphError> {
        let input = &inputs[&self.input];
        let out = match self.op {
            SimpleUnaryOp::Neg => input.neg(backend)?,
            SimpleUnaryOp::Abs => input.abs(backend)?,
            SimpleUnaryOp::Exp => input.exp(backend)?,
            SimpleUnaryOp::Ln => input.ln(backend)?,
            SimpleUnaryOp::Sqrt => input.sqrt(backend)?,
            SimpleUnaryOp::Not => input.not(backend)?,
            SimpleUnaryOp::Sign => input.sign(backend)?,
            SimpleUnaryOp::BitwiseNot => input.bitwise_not(backend)?,
            SimpleUnaryOp::Reciprocal => input.reciprocal(backend)?,
            SimpleUnaryOp::Trig(trig_op) => input.trig(trig_op, backend)?,
            SimpleUnaryOp::Floor => input.floor(backend)?,
            SimpleUnaryOp::Ceil => input.ceil(backend)?,
            SimpleUnaryOp::Round => input.round(backend)?,
            SimpleUnaryOp::IsInf { detect_positive, detect_negative } => input.is_inf(detect_positive, detect_negative)?,
            SimpleUnaryOp::IsNan => input.is_nan(backend)?,
            SimpleUnaryOp::Erf => input.erf(backend)?,
        };
        Ok([(self.output, out)].into_iter())
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
    output: MilliOpGraphTensorId,
    input: MilliOpGraphTensorId,
    value: f32,
}

impl MilliOpClampMin {
    pub fn new<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, a: MilliOpGraphTensorId, value: f32) -> MilliOpGraphTensorId {
        let output = graph.get_new_tensor_id();
        let node = Self { output, input: a, value };
        graph.push_op(AnyMilliOp::ClampMin(node));
        output
    }
}

impl Node<MilliOpGraphTensorId> for MilliOpClampMin {
    fn inputs(&self) -> impl Iterator<Item=MilliOpGraphTensorId> { vec![self.input].into_iter() }
    fn outputs(&self) -> impl Iterator<Item=MilliOpGraphTensorId> { vec![self.output].into_iter() }
}

impl MilliOp for MilliOpClampMin {
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<impl Iterator<Item=(MilliOpGraphTensorId, NumericTensor<DynRank>)>, MilliOpGraphError> {
        let out = inputs[&self.input].clamp_min(self.value, backend)?;
        Ok([(self.output, out)].into_iter())
    }

    fn get_name(&self) -> String {
        "Clamp Min".to_string()
    }
}
