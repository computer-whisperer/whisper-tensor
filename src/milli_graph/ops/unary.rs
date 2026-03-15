use crate::backends::eval_backend::EvalBackend;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::numeric_tensor::NumericTensor;
use crate::tensor_info::TensorInfo;
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
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    input: GlobalId,
    op: WhichSimpleUnaryOp,
}

impl SimpleUnaryOp {
    pub(crate) fn which_op(&self) -> &WhichSimpleUnaryOp {
        &self.op
    }

    fn new_internal(
        graph: &mut MilliOpGraph,
        input: GlobalId,
        op: WhichSimpleUnaryOp,
        label: Option<String>,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            output,
            input,
            op,
            global_id: GlobalId::new(rng),
            label,
        };
        graph.push_op(AnyMilliOp::SimpleUnary(node));
        output
    }

    pub fn neg(graph: &mut MilliOpGraph, input: GlobalId, rng: &mut impl rand::Rng) -> GlobalId {
        Self::new_internal(graph, input, WhichSimpleUnaryOp::Neg, None, rng)
    }
    pub fn abs(graph: &mut MilliOpGraph, input: GlobalId, rng: &mut impl rand::Rng) -> GlobalId {
        Self::new_internal(graph, input, WhichSimpleUnaryOp::Abs, None, rng)
    }
    pub fn exp(graph: &mut MilliOpGraph, input: GlobalId, rng: &mut impl rand::Rng) -> GlobalId {
        Self::new_internal(graph, input, WhichSimpleUnaryOp::Exp, None, rng)
    }
    pub fn ln(graph: &mut MilliOpGraph, input: GlobalId, rng: &mut impl rand::Rng) -> GlobalId {
        Self::new_internal(graph, input, WhichSimpleUnaryOp::Ln, None, rng)
    }
    pub fn sqrt(graph: &mut MilliOpGraph, input: GlobalId, rng: &mut impl rand::Rng) -> GlobalId {
        Self::new_internal(graph, input, WhichSimpleUnaryOp::Sqrt, None, rng)
    }
    pub fn not(graph: &mut MilliOpGraph, input: GlobalId, rng: &mut impl rand::Rng) -> GlobalId {
        Self::new_internal(graph, input, WhichSimpleUnaryOp::Not, None, rng)
    }
    pub fn sign(graph: &mut MilliOpGraph, input: GlobalId, rng: &mut impl rand::Rng) -> GlobalId {
        Self::new_internal(graph, input, WhichSimpleUnaryOp::Sign, None, rng)
    }
    pub fn bitwise_not(
        graph: &mut MilliOpGraph,
        input: GlobalId,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        Self::new_internal(graph, input, WhichSimpleUnaryOp::BitwiseNot, None, rng)
    }
    pub fn reciprocal(
        graph: &mut MilliOpGraph,
        input: GlobalId,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        Self::new_internal(graph, input, WhichSimpleUnaryOp::Reciprocal, None, rng)
    }
    pub fn trig(
        graph: &mut MilliOpGraph,
        input: GlobalId,
        trig_op: TrigOp,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        Self::new_internal(graph, input, WhichSimpleUnaryOp::Trig(trig_op), None, rng)
    }
    pub fn floor(graph: &mut MilliOpGraph, input: GlobalId, rng: &mut impl rand::Rng) -> GlobalId {
        Self::new_internal(graph, input, WhichSimpleUnaryOp::Floor, None, rng)
    }
    pub fn ceil(graph: &mut MilliOpGraph, input: GlobalId, rng: &mut impl rand::Rng) -> GlobalId {
        Self::new_internal(graph, input, WhichSimpleUnaryOp::Ceil, None, rng)
    }
    pub fn round(graph: &mut MilliOpGraph, input: GlobalId, rng: &mut impl rand::Rng) -> GlobalId {
        Self::new_internal(graph, input, WhichSimpleUnaryOp::Round, None, rng)
    }
    pub fn is_inf(
        graph: &mut MilliOpGraph,
        input: GlobalId,
        detect_positive: bool,
        detect_negative: bool,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        Self::new_internal(
            graph,
            input,
            WhichSimpleUnaryOp::IsInf {
                detect_positive,
                detect_negative,
            },
            None,
            rng,
        )
    }
    pub fn is_nan(graph: &mut MilliOpGraph, input: GlobalId, rng: &mut impl rand::Rng) -> GlobalId {
        Self::new_internal(graph, input, WhichSimpleUnaryOp::IsNan, None, rng)
    }
    pub fn erf(graph: &mut MilliOpGraph, input: GlobalId, rng: &mut impl rand::Rng) -> GlobalId {
        Self::new_internal(graph, input, WhichSimpleUnaryOp::Erf, None, rng)
    }
}

impl SimpleUnaryOp {
    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.input, map);
    }
}

impl Node for SimpleUnaryOp {
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
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.input].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}

impl MilliOp for SimpleUnaryOp {
    fn infer(
        &self,
        known_inputs: &HashMap<GlobalId, TensorInfo>,
        symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, TensorInfo)>>, MilliOpGraphError> {
        let input_info = known_inputs
            .get(&self.input)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        // If input is concrete, fall back to eval.
        if input_info.as_numeric().is_some() {
            let mut resolved = HashMap::new();
            resolved.insert(self.input, input_info.as_numeric().unwrap().clone());
            let collected: Vec<(GlobalId, TensorInfo)> = self
                .eval(&resolved, backend)?
                .map(|(a, b)| (a, TensorInfo::from(b)))
                .collect();
            return Ok(Box::new(collected.into_iter()));
        }

        // Unary ops preserve shape and dtype (except IsNan/IsInf which produce Bool).
        let out_info = match self.op {
            WhichSimpleUnaryOp::IsNan | WhichSimpleUnaryOp::IsInf { .. } => {
                // Output is Bool with same shape. Build new TensorInfo with Bool dtype.
                use crate::scalar_info::ScalarInfo;
                use crate::symbolic_scalar::SymbolicScalar;
                let first_elem = ScalarInfo::Symbolic(SymbolicScalar::new(
                    crate::dtype::DType::BOOL,
                    symbolic_resolver,
                ));
                TensorInfo::new_from_first_element_and_rank(
                    first_elem,
                    input_info.rank(),
                    symbolic_resolver,
                )
            }
            _ => {
                // Same dtype, same shape — clone the input info.
                input_info.clone()
            }
        };
        Ok(Box::new([(self.output, out_info)].into_iter()))
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
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

    fn backward(
        &self,
        output_grads: &HashMap<GlobalId, GlobalId>,
        graph: &mut MilliOpGraph,
        rng: &mut impl rand::Rng,
    ) -> Option<HashMap<GlobalId, GlobalId>> {
        let grad_output = *output_grads.get(&self.output)?;
        let grad_input = match self.op {
            // d/dx(-x) = -1 => grad_input = -grad_output
            WhichSimpleUnaryOp::Neg => SimpleUnaryOp::neg(graph, grad_output, rng),
            // d/dx(exp(x)) = exp(x) => grad_input = grad_output * output
            WhichSimpleUnaryOp::Exp => {
                // output = exp(input), reuse the forward output
                super::SimpleBinary::mul(graph, grad_output, self.output, rng)
            }
            // d/dx(ln(x)) = 1/x => grad_input = grad_output / input
            WhichSimpleUnaryOp::Ln => super::SimpleBinary::div(graph, grad_output, self.input, rng),
            // d/dx(sqrt(x)) = 1/(2*sqrt(x)) => grad_input = grad_output / (2 * output)
            WhichSimpleUnaryOp::Sqrt => {
                let two = super::Constant::new_scalar(graph, 2.0f32, rng);
                let two_output = super::SimpleBinary::mul(graph, two, self.output, rng);
                super::SimpleBinary::div(graph, grad_output, two_output, rng)
            }
            // d/dx(1/x) = -1/x^2 => grad_input = -grad_output * output^2
            // equivalently: -grad_output / (input * input)
            WhichSimpleUnaryOp::Reciprocal => {
                let input_sq = super::SimpleBinary::mul(graph, self.input, self.input, rng);
                let neg_grad = SimpleUnaryOp::neg(graph, grad_output, rng);
                super::SimpleBinary::div(graph, neg_grad, input_sq, rng)
            }
            // d/dx(tanh(x)) = 1 - tanh(x)^2 = 1 - output^2
            WhichSimpleUnaryOp::Trig(crate::TrigOp::Tanh) => {
                let out_sq = super::SimpleBinary::mul(graph, self.output, self.output, rng);
                let one = super::Constant::new_scalar(graph, 1.0f32, rng);
                let one_minus = super::SimpleBinary::sub(graph, one, out_sq, rng);
                super::SimpleBinary::mul(graph, grad_output, one_minus, rng)
            }
            _ => return None,
        };
        let mut result = HashMap::new();
        result.insert(self.input, grad_input);
        Some(result)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClampMin {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    input: GlobalId,
    value: f32,
}

impl ClampMin {
    pub(crate) fn min_val(&self) -> f32 {
        self.value
    }

    pub fn push_new(
        graph: &mut MilliOpGraph,
        a: GlobalId,
        value: f32,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        Self::push_new_with_label(graph, a, value, None, rng)
    }

    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        a: GlobalId,
        value: f32,
        label: Option<String>,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            label,
            output,
            input: a,
            value,
        };
        graph.push_op(AnyMilliOp::ClampMin(node));
        output
    }
}

impl ClampMin {
    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.input, map);
    }
}

impl Node for ClampMin {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Clamp Min".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.input].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
}

impl MilliOp for ClampMin {
    fn infer(
        &self,
        known_inputs: &HashMap<GlobalId, TensorInfo>,
        _symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, TensorInfo)>>, MilliOpGraphError> {
        let input_info = known_inputs
            .get(&self.input)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        if input_info.as_numeric().is_some() {
            let mut resolved = HashMap::new();
            resolved.insert(self.input, input_info.as_numeric().unwrap().clone());
            let collected: Vec<(GlobalId, TensorInfo)> = self
                .eval(&resolved, backend)?
                .map(|(a, b)| (a, TensorInfo::from(b)))
                .collect();
            return Ok(Box::new(collected.into_iter()));
        }

        // ClampMin preserves shape and dtype.
        Ok(Box::new([(self.output, input_info.clone())].into_iter()))
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        let out = inputs[&self.input].clamp_min(self.value, backend)?;
        Ok(Box::new([(self.output, out)].into_iter()))
    }

    fn backward(
        &self,
        output_grads: &HashMap<GlobalId, GlobalId>,
        graph: &mut MilliOpGraph,
        rng: &mut impl rand::Rng,
    ) -> Option<HashMap<GlobalId, GlobalId>> {
        let grad_output = *output_grads.get(&self.output)?;
        // grad_input = grad_output where input >= value, 0 otherwise
        // mask = cast(input >= value, float)
        let threshold = super::Constant::new_scalar(graph, self.value, rng);
        let mask = super::SimpleBinary::greater_or_equal(graph, self.input, threshold, rng);
        let mask_float = super::Cast::push_new(graph, mask, crate::dtype::DType::F32, rng);
        let grad_input = super::SimpleBinary::mul(graph, grad_output, mask_float, rng);
        let mut result = HashMap::new();
        result.insert(self.input, grad_input);
        Some(result)
    }
}
