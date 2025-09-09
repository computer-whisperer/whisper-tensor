use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::milli_graph::ops::MilliOp;
use crate::milli_graph::{MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum WhichSimpleBinaryOp {
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
pub struct SimpleBinary {
    output: MilliOpGraphTensorId,
    which_op: WhichSimpleBinaryOp,
    a: MilliOpGraphTensorId,
    b: MilliOpGraphTensorId,
}

use crate::graph::Node;
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::AnyMilliOp;

impl SimpleBinary {
    fn new<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        a: MilliOpGraphTensorId,
        b: MilliOpGraphTensorId,
        which_op: WhichSimpleBinaryOp,
    ) -> MilliOpGraphTensorId {
        let output = graph.get_new_tensor_id();
        let node = Self {
            output,
            which_op,
            a,
            b,
        };
        graph.push_op(AnyMilliOp::SimpleBinary(node));
        output
    }
    pub fn add<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        a: MilliOpGraphTensorId,
        b: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, WhichSimpleBinaryOp::Add)
    }

    pub fn sub<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        a: MilliOpGraphTensorId,
        b: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, WhichSimpleBinaryOp::Sub)
    }

    pub fn mul<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        a: MilliOpGraphTensorId,
        b: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, WhichSimpleBinaryOp::Mul)
    }

    pub fn div<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        a: MilliOpGraphTensorId,
        b: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, WhichSimpleBinaryOp::Div)
    }

    pub fn modulo<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        a: MilliOpGraphTensorId,
        b: MilliOpGraphTensorId,
        fmod: Option<bool>,
    ) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, WhichSimpleBinaryOp::Modulo(fmod))
    }
    pub fn and<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        a: MilliOpGraphTensorId,
        b: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, WhichSimpleBinaryOp::And)
    }

    pub fn or<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        a: MilliOpGraphTensorId,
        b: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, WhichSimpleBinaryOp::Or)
    }

    pub fn xor<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        a: MilliOpGraphTensorId,
        b: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, WhichSimpleBinaryOp::Xor)
    }

    pub fn bitwise_and<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        a: MilliOpGraphTensorId,
        b: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, WhichSimpleBinaryOp::BitwiseAnd)
    }

    pub fn bitwise_or<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        a: MilliOpGraphTensorId,
        b: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, WhichSimpleBinaryOp::BitwiseOr)
    }

    pub fn bitwise_xor<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        a: MilliOpGraphTensorId,
        b: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, WhichSimpleBinaryOp::BitwiseXor)
    }

    pub fn equal<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        a: MilliOpGraphTensorId,
        b: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, WhichSimpleBinaryOp::Equal)
    }

    pub fn greater<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        a: MilliOpGraphTensorId,
        b: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, WhichSimpleBinaryOp::Greater)
    }

    pub fn greater_or_equal<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        a: MilliOpGraphTensorId,
        b: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, WhichSimpleBinaryOp::GreaterOrEqual)
    }

    pub fn less<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        a: MilliOpGraphTensorId,
        b: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, WhichSimpleBinaryOp::Less)
    }

    pub fn less_or_equal<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        a: MilliOpGraphTensorId,
        b: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, WhichSimpleBinaryOp::LessOrEqual)
    }

    pub fn max<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        a: MilliOpGraphTensorId,
        b: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, WhichSimpleBinaryOp::Max)
    }

    pub fn min<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        a: MilliOpGraphTensorId,
        b: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        Self::new(graph, a, b, WhichSimpleBinaryOp::Min)
    }
}

impl MilliOp for SimpleBinary {
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<
        impl Iterator<Item = (MilliOpGraphTensorId, NumericTensor<DynRank>)>,
        MilliOpGraphError,
    > {
        let a = &inputs[&self.a];
        let b = &inputs[&self.b];
        let out = match self.which_op {
            WhichSimpleBinaryOp::Add => NumericTensor::<DynRank>::add(a, b, backend)?,
            WhichSimpleBinaryOp::Sub => NumericTensor::<DynRank>::sub(a, b, backend)?,
            WhichSimpleBinaryOp::Mul => NumericTensor::<DynRank>::mul(a, b, backend)?,
            WhichSimpleBinaryOp::Div => NumericTensor::<DynRank>::div(a, b, backend)?,
            WhichSimpleBinaryOp::Modulo(fmod) => {
                let is_float =
                    [DType::F64, DType::F32, DType::BF16, DType::F16].contains(&a.dtype());
                let fmod = if is_float {
                    true
                } else {
                    fmod.unwrap_or(false)
                };
                if fmod {
                    NumericTensor::<DynRank>::fmod(a, b, backend)?
                } else {
                    NumericTensor::<DynRank>::imod(a, b, backend)?
                }
            }
            WhichSimpleBinaryOp::And => NumericTensor::<DynRank>::and(a, b, backend)?,
            WhichSimpleBinaryOp::Or => NumericTensor::<DynRank>::or(a, b, backend)?,
            WhichSimpleBinaryOp::Xor => NumericTensor::<DynRank>::xor(a, b, backend)?,
            WhichSimpleBinaryOp::BitwiseAnd => {
                NumericTensor::<DynRank>::bitwise_and(a, b, backend)?
            }
            WhichSimpleBinaryOp::BitwiseOr => NumericTensor::<DynRank>::bitwise_or(a, b, backend)?,
            WhichSimpleBinaryOp::BitwiseXor => {
                NumericTensor::<DynRank>::bitwise_xor(a, b, backend)?
            }
            WhichSimpleBinaryOp::Equal => NumericTensor::<DynRank>::equal(a, b, backend)?,
            WhichSimpleBinaryOp::Greater => NumericTensor::<DynRank>::greater(a, b, backend)?,
            WhichSimpleBinaryOp::GreaterOrEqual => {
                NumericTensor::<DynRank>::greater_or_equal(a, b, backend)?
            }
            WhichSimpleBinaryOp::Less => NumericTensor::<DynRank>::less(a, b, backend)?,
            WhichSimpleBinaryOp::LessOrEqual => {
                NumericTensor::<DynRank>::less_or_equal(a, b, backend)?
            }
            WhichSimpleBinaryOp::Max => NumericTensor::<DynRank>::max(a, b, backend)?,
            WhichSimpleBinaryOp::Min => NumericTensor::<DynRank>::min(a, b, backend)?,
        };
        Ok([(self.output, out)].into_iter())
    }

    fn get_name(&self) -> String {
        match self.which_op {
            WhichSimpleBinaryOp::Add => "Add",
            WhichSimpleBinaryOp::Sub => "Sub",
            WhichSimpleBinaryOp::Mul => "Mul",
            WhichSimpleBinaryOp::Div => "Div",
            WhichSimpleBinaryOp::Modulo(_) => "Modulo",
            WhichSimpleBinaryOp::And => "And",
            WhichSimpleBinaryOp::Or => "Or",
            WhichSimpleBinaryOp::Xor => "Xor",
            WhichSimpleBinaryOp::BitwiseAnd => "Bitwise And",
            WhichSimpleBinaryOp::BitwiseOr => "Bitwise Or",
            WhichSimpleBinaryOp::BitwiseXor => "Bitwise Xor",
            WhichSimpleBinaryOp::Equal => "Equal",
            WhichSimpleBinaryOp::Greater => "Greater",
            WhichSimpleBinaryOp::GreaterOrEqual => "Greater or Equal",
            WhichSimpleBinaryOp::Less => "Less",
            WhichSimpleBinaryOp::LessOrEqual => "Less or Equal",
            WhichSimpleBinaryOp::Max => "Max",
            WhichSimpleBinaryOp::Min => "Min",
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
    pub fn new<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        a: MilliOpGraphTensorId,
        b: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
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
    ) -> Result<
        impl Iterator<Item = (MilliOpGraphTensorId, NumericTensor<DynRank>)>,
        MilliOpGraphError,
    > {
        let out = NumericTensor::<DynRank>::pow(&inputs[&self.a], &inputs[&self.b], backend)?;
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
    pub fn new<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        a: MilliOpGraphTensorId,
        b: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
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
    ) -> Result<
        impl Iterator<Item = (MilliOpGraphTensorId, NumericTensor<DynRank>)>,
        MilliOpGraphError,
    > {
        let a_input = &inputs[&self.a];
        let b_input = &inputs[&self.b];
        let accumulate_dtype = match a_input.dtype() {
            DType::BF16 | DType::F16 => Some(DType::F32),
            _ => None,
        };
        let out = NumericTensor::<DynRank>::matmul(a_input, b_input, accumulate_dtype, backend)?;
        Ok([(self.output, out)].into_iter())
    }

    fn get_name(&self) -> String {
        "MatMul".to_string()
    }
}

impl Node<MilliOpGraphTensorId> for SimpleBinary {
    fn inputs(&self) -> impl Iterator<Item = MilliOpGraphTensorId> {
        vec![self.a, self.b].into_iter()
    }
    fn outputs(&self) -> impl Iterator<Item = MilliOpGraphTensorId> {
        vec![self.output].into_iter()
    }
}

impl Node<MilliOpGraphTensorId> for MilliOpPow {
    fn inputs(&self) -> impl Iterator<Item = MilliOpGraphTensorId> {
        vec![self.a, self.b].into_iter()
    }
    fn outputs(&self) -> impl Iterator<Item = MilliOpGraphTensorId> {
        vec![self.output].into_iter()
    }
}

impl Node<MilliOpGraphTensorId> for MilliOpMatMul {
    fn inputs(&self) -> impl Iterator<Item = MilliOpGraphTensorId> {
        vec![self.a, self.b].into_iter()
    }
    fn outputs(&self) -> impl Iterator<Item = MilliOpGraphTensorId> {
        vec![self.output].into_iter()
    }
}
