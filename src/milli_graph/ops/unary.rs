use crate::pool::Pool;
use crate::backends::eval_backend::EvalBackend;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::nano_graph::lower::{NanoLoweringContext, TensorAtomMap};
use crate::nano_graph::ops::{ScalarBinOp, ScalarOp, ScalarUnaryOp};
use crate::nano_graph::pattern::InputRef;
use crate::migration::numeric_tensor::NumericTensor;
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
    /// ln(1 + x) — numerically stable for small x where ln(1+x) ≈ x.
    Log1p,
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
    pub fn log1p(graph: &mut MilliOpGraph, input: GlobalId, rng: &mut impl rand::Rng) -> GlobalId {
        Self::new_internal(graph, input, WhichSimpleUnaryOp::Log1p, None, rng)
    }
}

impl SimpleUnaryOp {
    pub fn lower_to_nano(&self, ctx: &mut crate::nano_graph::NanoLoweringContext) {
        let all_infos = ctx.all_infos;

        let in_id = Node::inputs(self).next().unwrap();
        let out_id = Node::outputs(self).next().unwrap();

        let Some(in_map) = ctx.tensor_map.get(&in_id).cloned() else {
            ctx.lower_as_boundary_named(self, "SimpleUnary");
            return;
        };
        let Some(out_info) = all_infos.get(&out_id) else {
            ctx.lower_as_boundary_named(self, "SimpleUnary");
            return;
        };

        let dt = NanoLoweringContext::ndt(out_info);
        let scalar_op = match self.which_op() {
            WhichSimpleUnaryOp::Neg => ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: dt,
            },
            WhichSimpleUnaryOp::Abs => ScalarOp::Unary {
                op: ScalarUnaryOp::Abs,
                compute_dtype: dt,
            },
            WhichSimpleUnaryOp::Exp => ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: dt,
            },
            WhichSimpleUnaryOp::Ln => ScalarOp::Unary {
                op: ScalarUnaryOp::Ln,
                compute_dtype: dt,
            },
            WhichSimpleUnaryOp::Sqrt => ScalarOp::Unary {
                op: ScalarUnaryOp::Sqrt,
                compute_dtype: dt,
            },
            WhichSimpleUnaryOp::Reciprocal => ScalarOp::Unary {
                op: ScalarUnaryOp::Reciprocal,
                compute_dtype: dt,
            },
            WhichSimpleUnaryOp::Trig(crate::TrigOp::Tanh) => ScalarOp::Unary {
                op: ScalarUnaryOp::Tanh,
                compute_dtype: dt,
            },
            WhichSimpleUnaryOp::Floor => ScalarOp::Unary {
                op: ScalarUnaryOp::Floor,
                compute_dtype: dt,
            },
            WhichSimpleUnaryOp::Ceil => ScalarOp::Unary {
                op: ScalarUnaryOp::Ceil,
                compute_dtype: dt,
            },
            WhichSimpleUnaryOp::Round => ScalarOp::Unary {
                op: ScalarUnaryOp::Round,
                compute_dtype: dt,
            },
            WhichSimpleUnaryOp::Sign => ScalarOp::Unary {
                op: ScalarUnaryOp::Sign,
                compute_dtype: dt,
            },
            WhichSimpleUnaryOp::Not => ScalarOp::Unary {
                op: ScalarUnaryOp::Not,
                compute_dtype: dt,
            },
            WhichSimpleUnaryOp::IsNan => ScalarOp::Unary {
                op: ScalarUnaryOp::IsNan,
                compute_dtype: all_infos.get(&in_id).map(|i| NanoLoweringContext::ndt(i)).unwrap_or(dt),
            },
            WhichSimpleUnaryOp::Erf => ScalarOp::Unary {
                op: ScalarUnaryOp::Erf,
                compute_dtype: dt,
            },
            WhichSimpleUnaryOp::Trig(crate::TrigOp::Sin) => ScalarOp::Unary {
                op: ScalarUnaryOp::Sin,
                compute_dtype: dt,
            },
            WhichSimpleUnaryOp::Trig(crate::TrigOp::Cos) => ScalarOp::Unary {
                op: ScalarUnaryOp::Cos,
                compute_dtype: dt,
            },
            WhichSimpleUnaryOp::IsInf {
                detect_positive,
                detect_negative,
            } => ScalarOp::Unary {
                op: ScalarUnaryOp::IsInf {
                    detect_positive: *detect_positive,
                    detect_negative: *detect_negative,
                },
                compute_dtype: all_infos.get(&in_id).map(|i| NanoLoweringContext::ndt(i)).unwrap_or(dt),
            },
            WhichSimpleUnaryOp::BitwiseNot => ScalarOp::Unary {
                op: ScalarUnaryOp::BitwiseNot,
                compute_dtype: dt,
            },
            WhichSimpleUnaryOp::Log1p => ScalarOp::Unary {
                op: ScalarUnaryOp::Log1p,
                compute_dtype: dt,
            },
            WhichSimpleUnaryOp::Trig(crate::TrigOp::Tan) => ScalarOp::Unary {
                op: ScalarUnaryOp::Tan,
                compute_dtype: dt,
            },
            WhichSimpleUnaryOp::Trig(crate::TrigOp::Asin) => ScalarOp::Unary {
                op: ScalarUnaryOp::Asin,
                compute_dtype: dt,
            },
            WhichSimpleUnaryOp::Trig(crate::TrigOp::Acos) => ScalarOp::Unary {
                op: ScalarUnaryOp::Acos,
                compute_dtype: dt,
            },
            WhichSimpleUnaryOp::Trig(crate::TrigOp::Atan) => ScalarOp::Unary {
                op: ScalarUnaryOp::Atan,
                compute_dtype: dt,
            },
            WhichSimpleUnaryOp::Trig(crate::TrigOp::Sinh) => ScalarOp::Unary {
                op: ScalarUnaryOp::Sinh,
                compute_dtype: dt,
            },
            WhichSimpleUnaryOp::Trig(crate::TrigOp::Cosh) => ScalarOp::Unary {
                op: ScalarUnaryOp::Cosh,
                compute_dtype: dt,
            },
            WhichSimpleUnaryOp::Trig(crate::TrigOp::Asinh) => ScalarOp::Unary {
                op: ScalarUnaryOp::Asinh,
                compute_dtype: dt,
            },
            WhichSimpleUnaryOp::Trig(crate::TrigOp::Acosh) => ScalarOp::Unary {
                op: ScalarUnaryOp::Acosh,
                compute_dtype: dt,
            },
            WhichSimpleUnaryOp::Trig(crate::TrigOp::Atanh) => ScalarOp::Unary {
                op: ScalarUnaryOp::Atanh,
                compute_dtype: dt,
            },
        };

        let known_dims = in_map.known_dims();
        let input_ref = NanoLoweringContext::pointwise_input_ref(&in_map);

        let base_id = ctx.nano.push_group(
            in_map.count,
            dt,
            scalar_op,
            in_map.sym_dims.clone(),
            vec![input_ref],
        );

        ctx.tensor_map.insert(
            out_id,
            TensorAtomMap::simple(
                base_id,
                in_map.count,
                dt,
                in_map.layout.clone(),
                TensorAtomMap::compute_strides(&known_dims),
                in_map.sym_dims.clone(),
            ),
        );
    }

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
            WhichSimpleUnaryOp::Log1p => "Log1p",
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
    fn infer<'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, TensorInfo<'p, P>>,
        symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        pool: &'p P,
    ) -> Result<Vec<(GlobalId, TensorInfo<'p, P>)>, MilliOpGraphError> {
        let input_info = known_inputs
            .get(&self.input)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        // If input is concrete, fall back to eval.
        if input_info.as_numeric().is_some() {
            let mut resolved = HashMap::new();
            resolved.insert(self.input, input_info.as_numeric().unwrap());
            let collected: Vec<(GlobalId, TensorInfo<'p, P>)> = self
                .eval(&resolved, &super::MilliEvalConfig::default(), &mut crate::backends::eval_backend::EvalBackend::NDArray)?
                .map(|(a, b)| (a, TensorInfo::from_legacy(&b, pool)))
                .collect();
            return Ok(collected);
        }

        // Unary ops preserve shape and dtype (except IsNan/IsInf which produce Bool).
        let out_info = match self.op {
            WhichSimpleUnaryOp::IsNan | WhichSimpleUnaryOp::IsInf { .. } => {
                // Output is Bool with same shape. Build new TensorInfo with Bool dtype.
                use crate::scalar_info::ScalarInfo;
                use crate::symbolic_scalar::SymbolicScalar;
                let first_elem = ScalarInfo::Symbolic(SymbolicScalar::new(
                    crate::numeric_dtype::NumericDType::Bool,
                    symbolic_resolver,
                ));
                TensorInfo::new_from_first_element_and_rank(
                    first_elem,
                    input_info.rank(),
                    symbolic_resolver,
                )
            }
            _ => {
                // Same dtype, same shape — reconstruct from first_element + rank.
                TensorInfo::new_from_first_element_and_rank(
                    input_info.first_element(),
                    input_info.rank(),
                    symbolic_resolver,
                )
            }
        };
        Ok(vec![((self.output, out_info))])
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        _config: &super::MilliEvalConfig,
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
            WhichSimpleUnaryOp::Log1p => input.log1p(backend)?,
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
    pub fn lower_to_nano(&self, ctx: &mut crate::nano_graph::NanoLoweringContext) {
        let all_infos = ctx.all_infos;
        let in_id = Node::inputs(self).next().unwrap();
        let out_id = Node::outputs(self).next().unwrap();

        let Some(in_map) = ctx.tensor_map.get(&in_id).cloned() else {
            ctx.lower_as_boundary_named(self, "ClampMin");
            return;
        };
        let Some(out_info) = all_infos.get(&out_id) else {
            ctx.lower_as_boundary_named(self, "ClampMin");
            return;
        };

        let min_val = self.min_val();
        let dt = NanoLoweringContext::ndt(out_info);
        let min_id = ctx.nano.push_atom(
            dt,
            ScalarOp::Literal(crate::numeric_scalar::NumericScalar::from_f32(min_val)),
            vec![],
            vec![],
        );

        let known_dims = in_map.known_dims();
        let input_ref = NanoLoweringContext::pointwise_input_ref(&in_map);

        let base_id = ctx.nano.push_group(
            in_map.count,
            dt,
            ScalarOp::Binary {
                op: ScalarBinOp::Max,
                compute_dtype: dt,
            },
            in_map.sym_dims.clone(),
            vec![input_ref, InputRef::Broadcast(min_id)],
        );

        ctx.tensor_map.insert(
            out_id,
            TensorAtomMap::simple(
                base_id,
                in_map.count,
                dt,
                in_map.layout.clone(),
                TensorAtomMap::compute_strides(&known_dims),
                in_map.sym_dims.clone(),
            ),
        );
    }

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
    fn infer<'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, TensorInfo<'p, P>>,
        symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        pool: &'p P,
    ) -> Result<Vec<(GlobalId, TensorInfo<'p, P>)>, MilliOpGraphError> {
        let input_info = known_inputs
            .get(&self.input)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        if input_info.as_numeric().is_some() {
            let mut resolved = HashMap::new();
            resolved.insert(self.input, input_info.as_numeric().unwrap());
            let collected: Vec<(GlobalId, TensorInfo<'p, P>)> = self
                .eval(&resolved, &super::MilliEvalConfig::default(), &mut crate::backends::eval_backend::EvalBackend::NDArray)?
                .map(|(a, b)| (a, TensorInfo::from_legacy(&b, pool)))
                .collect();
            return Ok(collected);
        }

        // ClampMin preserves shape and dtype.
        let out = TensorInfo::new_from_first_element_and_rank(
            input_info.first_element(),
            input_info.rank(),
            symbolic_resolver,
        );
        Ok(vec![((self.output, out))])
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        _config: &super::MilliEvalConfig,
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
