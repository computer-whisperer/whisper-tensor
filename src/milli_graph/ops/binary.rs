use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::milli_graph::ops::{
    MilliOp,
};
use crate::milli_graph::{MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum SimpleBinaryOp {
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
    Min,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpSimpleBinary {
    output: MilliOpGraphTensorId,
    which_op: SimpleBinaryOp,
    a: MilliOpGraphTensorId,
    b: MilliOpGraphTensorId,
}

use crate::milli_graph::ops::AnyMilliOp;
use crate::milli_graph::MilliOpGraph;
use crate::graph::Node;

impl MilliOpSimpleBinary {
    fn new<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, a: MilliOpGraphTensorId, b: MilliOpGraphTensorId, which_op: SimpleBinaryOp) -> MilliOpGraphTensorId {
        let output = graph.get_new_tensor_id();
        let node = Self { output, which_op, a, b };
        graph.push_op(AnyMilliOp::SimpleBinary(node));
        output
    }
    pub fn add<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, SimpleBinaryOp::Add)
    }

    pub fn sub<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, SimpleBinaryOp::Sub)
    }

    pub fn mul<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, SimpleBinaryOp::Mul)
    }

    pub fn div<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, SimpleBinaryOp::Div)
    }

    pub fn modulo<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, a: MilliOpGraphTensorId, b: MilliOpGraphTensorId, fmod: Option<bool>) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, SimpleBinaryOp::Modulo(fmod))
    }
    pub fn and<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, SimpleBinaryOp::And)
    }

    pub fn or<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, SimpleBinaryOp::Or)
    }

    pub fn xor<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, SimpleBinaryOp::Xor)
    }

    pub fn bitwise_and<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, SimpleBinaryOp::BitwiseAnd)
    }

    pub fn bitwise_or<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, SimpleBinaryOp::BitwiseOr)
    }

    pub fn bitwise_xor<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, SimpleBinaryOp::BitwiseXor)
    }

    pub fn equal<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, SimpleBinaryOp::Equal)
    }

    pub fn greater<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, SimpleBinaryOp::Greater)
    }

    pub fn greater_or_equal<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, SimpleBinaryOp::GreaterOrEqual)
    }

    pub fn less<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, SimpleBinaryOp::Less)
    }

    pub fn less_or_equal<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, SimpleBinaryOp::LessOrEqual)
    }

    pub fn max<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, SimpleBinaryOp::Max)
    }

    pub fn min<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, SimpleBinaryOp::Min)
    }
}

impl MilliOp for MilliOpSimpleBinary {
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<impl Iterator<Item = (MilliOpGraphTensorId, NumericTensor<DynRank>)>, MilliOpGraphError> {
        let a = &inputs[&self.a];
        let b = &inputs[&self.b];
        let out = match self.which_op {
            SimpleBinaryOp::Add => NumericTensor::<DynRank>::add(a, b, backend)?,
            SimpleBinaryOp::Sub => NumericTensor::<DynRank>::sub(a, b, backend)?,
            SimpleBinaryOp::Mul => NumericTensor::<DynRank>::mul(a, b, backend)?,
            SimpleBinaryOp::Div => NumericTensor::<DynRank>::div(a, b, backend)?,
            SimpleBinaryOp::Modulo(fmod) => {
                let is_float = [DType::F64, DType::F32, DType::BF16, DType::F16].contains(&a.dtype());
                let fmod = if is_float { true } else { fmod.unwrap_or(false) };
                if fmod { NumericTensor::<DynRank>::fmod(a, b, backend)? } else { NumericTensor::<DynRank>::imod(a, b, backend)? }
            }
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
        };
        Ok([(self.output, out)].into_iter())
    }

    fn get_name(&self) -> String {
        match self.which_op {
            SimpleBinaryOp::Add => "Add",
            SimpleBinaryOp::Sub => "Sub",
            SimpleBinaryOp::Mul => "Mul",
            SimpleBinaryOp::Div => "Div",
            SimpleBinaryOp::Modulo(_) => "Modulo",
            SimpleBinaryOp::And => "And",
            SimpleBinaryOp::Or => "Or",
            SimpleBinaryOp::Xor => "Xor",
            SimpleBinaryOp::BitwiseAnd => "Bitwise And",
            SimpleBinaryOp::BitwiseOr => "Bitwise Or",
            SimpleBinaryOp::BitwiseXor => "Bitwise Xor",
            SimpleBinaryOp::Equal => "Equal",
            SimpleBinaryOp::Greater => "Greater",
            SimpleBinaryOp::GreaterOrEqual => "Greater or Equal",
            SimpleBinaryOp::Less => "Less",
            SimpleBinaryOp::LessOrEqual => "Less or Equal",
            SimpleBinaryOp::Max => "Max",
            SimpleBinaryOp::Min => "Min",
        }
        .to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpPow {
    output: MilliOpGraphTensorId,
    a: MilliOpGraphTensorId,
    b: MilliOpGraphTensorId,
}

impl MilliOpPow {
    pub fn new<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> MilliOpGraphTensorId {
        let output = graph.get_new_tensor_id();
        let node = Self { output, a, b };
        graph.push_op(AnyMilliOp::Pow(node));
        output
    }
}

impl MilliOp for MilliOpPow {
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<impl Iterator<Item = (MilliOpGraphTensorId, NumericTensor<DynRank>)>, MilliOpGraphError> {
        let out = NumericTensor::<DynRank>::pow(
            &inputs[&self.a],
            &inputs[&self.b],
            backend,
        )?;
        Ok([(self.output, out)].into_iter())
    }

    fn get_name(&self) -> String {
        "Pow".to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpMatMul {
    output: MilliOpGraphTensorId,
    a: MilliOpGraphTensorId,
    b: MilliOpGraphTensorId,
}

impl MilliOpMatMul {
    pub fn new<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, a: MilliOpGraphTensorId, b: MilliOpGraphTensorId) -> MilliOpGraphTensorId {
        let output = graph.get_new_tensor_id();
        let node = Self { output, a, b };
        graph.push_op(AnyMilliOp::MatMul(node));
        output
    }
}

impl MilliOp for MilliOpMatMul {
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<impl Iterator<Item = (MilliOpGraphTensorId, NumericTensor<DynRank>)>, MilliOpGraphError> {
        let a_input = &inputs[&self.a];
        let b_input = &inputs[&self.b];
        let accumulate_dtype = match a_input.dtype() {
            DType::BF16 | DType::F16 => Some(DType::F32),
            _ => None,
        };
        let out = NumericTensor::<DynRank>::matmul(
            a_input,
            b_input,
            accumulate_dtype,
            backend,
        )?;
        Ok([(self.output, out)].into_iter())
    }

    fn get_name(&self) -> String {
        "MatMul".to_string()
    }
}


impl Node<MilliOpGraphTensorId> for MilliOpSimpleBinary {
    fn inputs(&self) -> impl Iterator<Item = MilliOpGraphTensorId> {
        vec![self.a, self.b].into_iter()
    }
    fn outputs(&self) -> impl Iterator<Item = MilliOpGraphTensorId> {
        vec![self.output].into_iter()
    }
}


impl Node<MilliOpGraphTensorId> for MilliOpPow {
    fn inputs(&self) -> impl Iterator<Item = MilliOpGraphTensorId> { vec![self.a, self.b].into_iter() }
    fn outputs(&self) -> impl Iterator<Item = MilliOpGraphTensorId> { vec![self.output].into_iter() }
}

impl Node<MilliOpGraphTensorId> for MilliOpMatMul {
    fn inputs(&self) -> impl Iterator<Item = MilliOpGraphTensorId> { vec![self.a, self.b].into_iter() }
    fn outputs(&self) -> impl Iterator<Item = MilliOpGraphTensorId> { vec![self.output].into_iter() }
}
