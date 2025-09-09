use crate::backends::eval_backend::EvalBackend;
use crate::graph::Node;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use crate::{DynRank, TrigOp};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum WhichSimpleUnaryOp {
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
pub struct SimpleUnaryOp {
    output: MilliOpGraphTensorId,
    input: MilliOpGraphTensorId,
    op: WhichSimpleUnaryOp,
}

impl SimpleUnaryOp {
    fn new_internal<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        input: MilliOpGraphTensorId,
        op: WhichSimpleUnaryOp,
    ) -> MilliOpGraphTensorId {
        let output = graph.get_new_tensor_id();
        let node = Self { output, input, op };
        graph.push_op(AnyMilliOp::SimpleUnary(node));
        output
    }

    pub fn neg<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        input: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        Self::new_internal(graph, input, WhichSimpleUnaryOp::Neg)
    }
    pub fn abs<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        input: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        Self::new_internal(graph, input, WhichSimpleUnaryOp::Abs)
    }
    pub fn exp<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        input: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        Self::new_internal(graph, input, WhichSimpleUnaryOp::Exp)
    }
    pub fn ln<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        input: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        Self::new_internal(graph, input, WhichSimpleUnaryOp::Ln)
    }
    pub fn sqrt<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        input: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        Self::new_internal(graph, input, WhichSimpleUnaryOp::Sqrt)
    }
    pub fn not<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        input: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        Self::new_internal(graph, input, WhichSimpleUnaryOp::Not)
    }
    pub fn sign<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        input: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        Self::new_internal(graph, input, WhichSimpleUnaryOp::Sign)
    }
    pub fn bitwise_not<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        input: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        Self::new_internal(graph, input, WhichSimpleUnaryOp::BitwiseNot)
    }
    pub fn reciprocal<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        input: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        Self::new_internal(graph, input, WhichSimpleUnaryOp::Reciprocal)
    }
    pub fn trig<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        input: MilliOpGraphTensorId,
        trig_op: TrigOp,
    ) -> MilliOpGraphTensorId {
        Self::new_internal(graph, input, WhichSimpleUnaryOp::Trig(trig_op))
    }
    pub fn floor<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        input: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        Self::new_internal(graph, input, WhichSimpleUnaryOp::Floor)
    }
    pub fn ceil<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        input: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        Self::new_internal(graph, input, WhichSimpleUnaryOp::Ceil)
    }
    pub fn round<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        input: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        Self::new_internal(graph, input, WhichSimpleUnaryOp::Round)
    }
    pub fn is_inf<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        input: MilliOpGraphTensorId,
        detect_positive: bool,
        detect_negative: bool,
    ) -> MilliOpGraphTensorId {
        Self::new_internal(
            graph,
            input,
            WhichSimpleUnaryOp::IsInf {
                detect_positive,
                detect_negative,
            },
        )
    }
    pub fn is_nan<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        input: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        Self::new_internal(graph, input, WhichSimpleUnaryOp::IsNan)
    }
    pub fn erf<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        input: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        Self::new_internal(graph, input, WhichSimpleUnaryOp::Erf)
    }
}

impl Node<MilliOpGraphTensorId> for SimpleUnaryOp {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        match self.op {
            WhichSimpleUnaryOp::Neg => "Neg",
            WhichSimpleUnaryOp::Abs => "Abs",
            WhichSimpleUnaryOp::Exp => "Exp",
            WhichSimpleUnaryOp::Ln => "Ln",
            WhichSimpleUnaryOp::Sqrt => "Sqrt",
            WhichSimpleUnaryOp::Not => "Not",
            WhichSimpleUnaryOp::Sign => "Sign",
            WhichSimpleUnaryOp::BitwiseNot => "Bitwise Not",
            WhichSimpleUnaryOp::Reciprocal => "Reciprocal",
            WhichSimpleUnaryOp::Trig(trig_op) => trig_op.get_name(),
            WhichSimpleUnaryOp::Floor => "Floor",
            WhichSimpleUnaryOp::Ceil => "Ceil",
            WhichSimpleUnaryOp::Round => "Round",
            WhichSimpleUnaryOp::IsNan => "IsNan",
            WhichSimpleUnaryOp::IsInf { .. } => "IsInf",
            WhichSimpleUnaryOp::Erf => "Erf",
        }
        .to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = MilliOpGraphTensorId>> {
        Box::new(vec![self.input].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = MilliOpGraphTensorId>> {
        Box::new(vec![self.output].into_iter())
    }
}

impl MilliOp for SimpleUnaryOp {
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<
        Box<dyn Iterator<Item = (MilliOpGraphTensorId, NumericTensor<DynRank>)>>,
        MilliOpGraphError,
    > {
        let input = &inputs[&self.input];
        let out = match self.op {
            WhichSimpleUnaryOp::Neg => input.neg(backend)?,
            WhichSimpleUnaryOp::Abs => input.abs(backend)?,
            WhichSimpleUnaryOp::Exp => input.exp(backend)?,
            WhichSimpleUnaryOp::Ln => input.ln(backend)?,
            WhichSimpleUnaryOp::Sqrt => input.sqrt(backend)?,
            WhichSimpleUnaryOp::Not => input.not(backend)?,
            WhichSimpleUnaryOp::Sign => input.sign(backend)?,
            WhichSimpleUnaryOp::BitwiseNot => input.bitwise_not(backend)?,
            WhichSimpleUnaryOp::Reciprocal => input.reciprocal(backend)?,
            WhichSimpleUnaryOp::Trig(trig_op) => input.trig(trig_op, backend)?,
            WhichSimpleUnaryOp::Floor => input.floor(backend)?,
            WhichSimpleUnaryOp::Ceil => input.ceil(backend)?,
            WhichSimpleUnaryOp::Round => input.round(backend)?,
            WhichSimpleUnaryOp::IsInf {
                detect_positive,
                detect_negative,
            } => input.is_inf(detect_positive, detect_negative)?,
            WhichSimpleUnaryOp::IsNan => input.is_nan(backend)?,
            WhichSimpleUnaryOp::Erf => input.erf(backend)?,
        };
        Ok(Box::new([(self.output, out)].into_iter()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClampMin {
    output: MilliOpGraphTensorId,
    input: MilliOpGraphTensorId,
    value: f32,
}

impl ClampMin {
    pub fn push_new<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        a: MilliOpGraphTensorId,
        value: f32,
    ) -> MilliOpGraphTensorId {
        let output = graph.get_new_tensor_id();
        let node = Self {
            output,
            input: a,
            value,
        };
        graph.push_op(AnyMilliOp::ClampMin(node));
        output
    }
}

impl Node<MilliOpGraphTensorId> for ClampMin {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "Clamp Min".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = MilliOpGraphTensorId>> {
        Box::new(vec![self.input].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = MilliOpGraphTensorId>> {
        Box::new(vec![self.output].into_iter())
    }
}

impl MilliOp for ClampMin {
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<
        Box<dyn Iterator<Item = (MilliOpGraphTensorId, NumericTensor<DynRank>)>>,
        MilliOpGraphError,
    > {
        let out = inputs[&self.input].clamp_min(self.value, backend)?;
        Ok(Box::new([(self.output, out)].into_iter()))
    }
}
