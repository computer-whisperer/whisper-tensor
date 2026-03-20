use crate::dtype::DType;
use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::{MilliLoweringContext, MilliOpGraph};
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{ONNXDecodingError, query_attribute_float, query_attribute_int};
use crate::{TrigOp, milli_graph, onnx};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, strum_macros::Display, Serialize, Deserialize)]
pub enum WhichUnaryOperation {
    Relu,
    Sigmoid,
    Exp,
    Log,
    Softplus,
    Reciprocal,
    Neg,
    Abs,
    Sign,
    Not,
    NonZero,
    Sqrt,
    BitwiseNot,
    Trig(TrigOp),
    Floor,
    Ceil,
    Round,
    IsNan,
    Erf,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UnaryOperation {
    global_id: GlobalId,
    input: GlobalId,
    output: GlobalId,
    which: WhichUnaryOperation,
}

impl UnaryOperation {
    pub fn new(
        input: GlobalId,
        output: GlobalId,
        which: WhichUnaryOperation,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            input,
            output,
            which,
        }
    }

    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        which: WhichUnaryOperation,
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Unary"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Unary"));
        }

        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Unary"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Unary"))?,
            which,
        })
    }
}

impl Node for UnaryOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        match self.which {
            WhichUnaryOperation::Trig(trig) => trig.to_string(),
            _ => self.which.to_string(),
        }
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}
impl Operation for UnaryOperation {
    fn parameters(&self) -> Vec<Property> {
        vec![Property::new(
            "operation",
            PropertyValue::String(self.which.to_string()),
        )]
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let a = input_map[&self.input];
        if let WhichUnaryOperation::NonZero = &self.which {
            let out_tid = milli_graph::ops::NonZero::push_new(&mut graph, a, rng);
            let mut output_map = HashMap::new();
            output_map.insert(out_tid, self.output);
            graph.set_output_map(output_map);
            return graph;
        }
        let out_tid = match &self.which {
            WhichUnaryOperation::Relu => {
                milli_graph::ops::ClampMin::push_new(&mut graph, a, 0.0, rng)
            }
            WhichUnaryOperation::Sigmoid => {
                let xn = milli_graph::ops::Cast::push_new(&mut graph, a, DType::F32, rng);
                let xn = milli_graph::ops::SimpleUnaryOp::neg(&mut graph, xn, rng);
                let xn = milli_graph::ops::SimpleUnaryOp::exp(&mut graph, xn, rng);
                let c_tid = milli_graph::ops::Constant::new_scalar(&mut graph, 1.0f32, rng);
                let c = milli_graph::ops::CastLike::push_new(&mut graph, c_tid, xn, rng);
                let o = milli_graph::ops::SimpleBinary::add(&mut graph, xn, c, rng);
                let o = milli_graph::ops::SimpleUnaryOp::reciprocal(&mut graph, o, rng);
                milli_graph::ops::CastLike::push_new(&mut graph, o, a, rng)
            }
            WhichUnaryOperation::Exp => milli_graph::ops::SimpleUnaryOp::exp(&mut graph, a, rng),
            WhichUnaryOperation::Log => milli_graph::ops::SimpleUnaryOp::ln(&mut graph, a, rng),
            WhichUnaryOperation::Softplus => {
                let x = milli_graph::ops::SimpleUnaryOp::exp(&mut graph, a, rng);
                let c_tid = milli_graph::ops::Constant::new_scalar(&mut graph, 1.0f32, rng);
                let c = milli_graph::ops::CastLike::push_new(&mut graph, c_tid, x, rng);
                let x = milli_graph::ops::SimpleBinary::add(&mut graph, x, c, rng);
                milli_graph::ops::SimpleUnaryOp::ln(&mut graph, x, rng)
            }
            WhichUnaryOperation::Neg => milli_graph::ops::SimpleUnaryOp::neg(&mut graph, a, rng),
            WhichUnaryOperation::Sqrt => milli_graph::ops::SimpleUnaryOp::sqrt(&mut graph, a, rng),
            WhichUnaryOperation::Abs => milli_graph::ops::SimpleUnaryOp::abs(&mut graph, a, rng),
            WhichUnaryOperation::Trig(trig_op) => {
                milli_graph::ops::SimpleUnaryOp::trig(&mut graph, a, *trig_op, rng)
            }
            WhichUnaryOperation::Reciprocal => {
                milli_graph::ops::SimpleUnaryOp::reciprocal(&mut graph, a, rng)
            }
            WhichUnaryOperation::BitwiseNot => {
                milli_graph::ops::SimpleUnaryOp::bitwise_not(&mut graph, a, rng)
            }
            WhichUnaryOperation::Not => milli_graph::ops::SimpleUnaryOp::not(&mut graph, a, rng),
            WhichUnaryOperation::Sign => milli_graph::ops::SimpleUnaryOp::sign(&mut graph, a, rng),
            WhichUnaryOperation::Floor => {
                milli_graph::ops::SimpleUnaryOp::floor(&mut graph, a, rng)
            }
            WhichUnaryOperation::Ceil => milli_graph::ops::SimpleUnaryOp::ceil(&mut graph, a, rng),
            WhichUnaryOperation::Round => {
                milli_graph::ops::SimpleUnaryOp::round(&mut graph, a, rng)
            }
            WhichUnaryOperation::IsNan => {
                milli_graph::ops::SimpleUnaryOp::is_nan(&mut graph, a, rng)
            }
            WhichUnaryOperation::Erf => milli_graph::ops::SimpleUnaryOp::erf(&mut graph, a, rng),
            WhichUnaryOperation::NonZero => unreachable!(),
        };
        let mut output_map = HashMap::new();
        output_map.insert(out_tid, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SoftmaxOperation {
    global_id: GlobalId,
    axis: Option<i64>,
    input: GlobalId,
    output: GlobalId,
}

impl SoftmaxOperation {
    pub fn new(input: GlobalId, output: GlobalId, axis: Option<i64>, rng: &mut impl Rng) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            axis,
            input,
            output,
        }
    }

    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Softmax"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Softmax"));
        }

        Ok(Self {
            global_id: GlobalId::new(rng),
            axis: query_attribute_int(attributes, "axis"),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Softmax"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Softmax"))?,
        })
    }
}

impl Node for SoftmaxOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Softmax".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}
impl Operation for SoftmaxOperation {
    fn parameters(&self) -> Vec<Property> {
        let mut params = Vec::new();
        if let Some(axis) = self.axis {
            params.push(Property::new("axis", PropertyValue::Int(axis)));
        }
        params
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);

        let axis_tid =
            milli_graph::ops::Constant::new_scalar(&mut graph, self.axis.unwrap_or(-1), rng);
        // Subtract row max before exp to prevent overflow (critical for f16)
        let row_max = milli_graph::ops::ReduceMax::push_new(
            &mut graph,
            input_map[&self.input],
            Some(axis_tid),
            true,
            false,
            rng,
        );
        let shifted =
            milli_graph::ops::SimpleBinary::sub(&mut graph, input_map[&self.input], row_max, rng);
        let e = milli_graph::ops::SimpleUnaryOp::exp(&mut graph, shifted, rng);
        let sum =
            milli_graph::ops::ReduceSum::push_new(&mut graph, e, Some(axis_tid), true, false, rng);
        let out_tid = milli_graph::ops::SimpleBinary::div(&mut graph, e, sum, rng);
        let mut output_map = HashMap::new();
        output_map.insert(out_tid, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LogSoftmaxOperation {
    global_id: GlobalId,
    axis: Option<i64>,
    input: GlobalId,
    output: GlobalId,
}

impl LogSoftmaxOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("LogSoftmax"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("LogSoftmax"));
        }

        Ok(Self {
            global_id: GlobalId::new(rng),
            axis: query_attribute_int(attributes, "axis"),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("LogSoftmax"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("LogSoftmax"))?,
        })
    }
}

impl Node for LogSoftmaxOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "LogSoftmax".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for LogSoftmaxOperation {
    fn parameters(&self) -> Vec<Property> {
        let mut params = Vec::new();
        if let Some(axis) = self.axis {
            params.push(Property::new("axis", PropertyValue::Int(axis)));
        }
        params
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);

        let axis_tid =
            milli_graph::ops::Constant::new_scalar(&mut graph, self.axis.unwrap_or(-1), rng);
        // Subtract row max before exp to prevent overflow (critical for f16)
        let row_max = milli_graph::ops::ReduceMax::push_new(
            &mut graph,
            input_map[&self.input],
            Some(axis_tid),
            true,
            false,
            rng,
        );
        let shifted =
            milli_graph::ops::SimpleBinary::sub(&mut graph, input_map[&self.input], row_max, rng);
        let e_tid = milli_graph::ops::SimpleUnaryOp::exp(&mut graph, shifted, rng);
        let sum_tid = milli_graph::ops::ReduceSum::push_new(
            &mut graph,
            e_tid,
            Some(axis_tid),
            true,
            false,
            rng,
        );
        let log_sum = milli_graph::ops::SimpleUnaryOp::ln(&mut graph, sum_tid, rng);
        // log_softmax(x) = (x - max) - log(sum(exp(x - max)))
        let out_tid = milli_graph::ops::SimpleBinary::sub(&mut graph, shifted, log_sum, rng);
        let mut output_map = HashMap::new();
        output_map.insert(out_tid, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IsInfOperation {
    global_id: GlobalId,
    input: GlobalId,
    output: GlobalId,
    detect_negative: Option<bool>,
    detect_positive: Option<bool>,
}

impl IsInfOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("IsInf"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("IsInf"));
        }

        let detect_negative = query_attribute_int(attributes, "detect_negative").map(|x| x != 0);
        let detect_positive = query_attribute_int(attributes, "detect_positive").map(|x| x != 0);

        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("IsInf"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("IsInf"))?,
            detect_negative,
            detect_positive,
        })
    }
}

impl Node for IsInfOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Is Inf".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}
impl Operation for IsInfOperation {
    fn parameters(&self) -> Vec<Property> {
        let mut params = Vec::new();
        if let Some(detect_negative) = self.detect_negative {
            params.push(Property::new(
                "detect_negative",
                PropertyValue::Bool(detect_negative),
            ));
        }
        if let Some(detect_positive) = self.detect_positive {
            params.push(Property::new(
                "detect_positive",
                PropertyValue::Bool(detect_positive),
            ));
        }
        params
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let input = input_map[&self.input];
        let out_tid = milli_graph::ops::SimpleUnaryOp::is_inf(
            &mut graph,
            input,
            self.detect_positive.unwrap_or(true),
            self.detect_negative.unwrap_or(true),
            rng,
        );
        let mut output_map = HashMap::new();
        output_map.insert(out_tid, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IdentityOperation {
    global_id: GlobalId,
    input: GlobalId,
    output: GlobalId,
}

impl IdentityOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        _attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Identity"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Identity"));
        }

        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Identity"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Identity"))?,
        })
    }
}

impl Node for IdentityOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Identity".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}
impl Operation for IdentityOperation {
    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let input = input_map[&self.input];
        let mut output_map = HashMap::new();
        output_map.insert(input, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LeakyReluOperation {
    global_id: GlobalId,
    input: GlobalId,
    output: GlobalId,
    alpha: f32,
}

impl LeakyReluOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("LeakyRelu"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("LeakyRelu"));
        }

        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("LeakyRelu"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("LeakyRelu"))?,
            alpha: query_attribute_float(attributes, "alpha").unwrap_or(0.01),
        })
    }
}

impl Node for LeakyReluOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "LeakyRelu".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for LeakyReluOperation {
    fn parameters(&self) -> Vec<Property> {
        vec![Property::new(
            "alpha",
            PropertyValue::Float(self.alpha.into()),
        )]
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        // LeakyRelu(x) = max(x, alpha * x)
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let a = input_map[&self.input];
        let alpha = milli_graph::ops::Constant::new_scalar(&mut graph, self.alpha, rng);
        let alpha = milli_graph::ops::CastLike::push_new(&mut graph, alpha, a, rng);
        let alpha_x = milli_graph::ops::SimpleBinary::mul(&mut graph, alpha, a, rng);
        let out_tid = milli_graph::ops::SimpleBinary::max(&mut graph, a, alpha_x, rng);
        let mut output_map = HashMap::new();
        output_map.insert(out_tid, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

/// ONNX Gelu operator.
/// Gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GeluOperation {
    global_id: GlobalId,
    input: GlobalId,
    output: GlobalId,
    approximate: String,
}

impl GeluOperation {
    pub fn new(input: GlobalId, output: GlobalId, rng: &mut impl Rng) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            input,
            output,
            approximate: "none".to_string(),
        }
    }

    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Gelu"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Gelu"));
        }
        let approximate = crate::symbolic_graph::query_attribute_string(attributes, "approximate")
            .unwrap_or_else(|| "none".to_string());
        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Gelu"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Gelu"))?,
            approximate,
        })
    }
}

impl Node for GeluOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Gelu".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for GeluOperation {
    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let x = input_map[&self.input];
        let out_tid = if self.approximate == "tanh" {
            push_gelu_tanh(&mut graph, x, rng)
        } else {
            push_gelu(&mut graph, x, rng)
        };
        let mut output_map = HashMap::new();
        output_map.insert(out_tid, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

/// Build GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2))) in a milli graph.
fn push_gelu(graph: &mut MilliOpGraph, x: GlobalId, rng: &mut impl Rng) -> GlobalId {
    let sqrt2 = milli_graph::ops::Constant::new_scalar(graph, std::f32::consts::SQRT_2, rng);
    let sqrt2 = milli_graph::ops::CastLike::push_new(graph, sqrt2, x, rng);
    let half = milli_graph::ops::Constant::new_scalar(graph, 0.5f32, rng);
    let half = milli_graph::ops::CastLike::push_new(graph, half, x, rng);
    let one = milli_graph::ops::Constant::new_scalar(graph, 1.0f32, rng);
    let one = milli_graph::ops::CastLike::push_new(graph, one, x, rng);

    let x_div_sqrt2 = milli_graph::ops::SimpleBinary::div(graph, x, sqrt2, rng);
    let erf_val = milli_graph::ops::SimpleUnaryOp::erf(graph, x_div_sqrt2, rng);
    let one_plus_erf = milli_graph::ops::SimpleBinary::add(graph, one, erf_val, rng);
    let half_x = milli_graph::ops::SimpleBinary::mul(graph, x, half, rng);
    milli_graph::ops::SimpleBinary::mul(graph, half_x, one_plus_erf, rng)
}

/// Build GELU_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
fn push_gelu_tanh(graph: &mut MilliOpGraph, x: GlobalId, rng: &mut impl Rng) -> GlobalId {
    let half = milli_graph::ops::Constant::new_scalar(graph, 0.5f32, rng);
    let half = milli_graph::ops::CastLike::push_new(graph, half, x, rng);
    let one = milli_graph::ops::Constant::new_scalar(graph, 1.0f32, rng);
    let one = milli_graph::ops::CastLike::push_new(graph, one, x, rng);
    let c = milli_graph::ops::Constant::new_scalar(graph, 0.044715f32, rng);
    let c = milli_graph::ops::CastLike::push_new(graph, c, x, rng);
    // sqrt(2/pi)
    let sqrt_2_pi =
        milli_graph::ops::Constant::new_scalar(graph, (2.0f32 / std::f32::consts::PI).sqrt(), rng);
    let sqrt_2_pi = milli_graph::ops::CastLike::push_new(graph, sqrt_2_pi, x, rng);

    let x_cubed = {
        let x2 = milli_graph::ops::SimpleBinary::mul(graph, x, x, rng);
        milli_graph::ops::SimpleBinary::mul(graph, x2, x, rng)
    };
    let inner = milli_graph::ops::SimpleBinary::mul(graph, c, x_cubed, rng);
    let inner = milli_graph::ops::SimpleBinary::add(graph, x, inner, rng);
    let inner = milli_graph::ops::SimpleBinary::mul(graph, sqrt_2_pi, inner, rng);
    let tanh_val = milli_graph::ops::SimpleUnaryOp::trig(graph, inner, TrigOp::Tanh, rng);
    let one_plus_tanh = milli_graph::ops::SimpleBinary::add(graph, one, tanh_val, rng);
    let half_x = milli_graph::ops::SimpleBinary::mul(graph, half, x, rng);
    milli_graph::ops::SimpleBinary::mul(graph, half_x, one_plus_tanh, rng)
}

/// ONNX Elu: max(0,x) + min(0, alpha*(exp(x)-1))
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EluOperation {
    global_id: GlobalId,
    input: GlobalId,
    output: GlobalId,
    alpha: f32,
}

impl EluOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Elu"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Elu"));
        }
        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Elu"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Elu"))?,
            alpha: query_attribute_float(attributes, "alpha").unwrap_or(1.0),
        })
    }
}

impl Node for EluOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Elu".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for EluOperation {
    fn parameters(&self) -> Vec<Property> {
        vec![Property::new(
            "alpha",
            PropertyValue::Float(self.alpha.into()),
        )]
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        // Elu(x) = max(0, x) + min(0, alpha * (exp(x) - 1))
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let x = input_map[&self.input];
        let zero = milli_graph::ops::Constant::new_scalar(&mut graph, 0.0f32, rng);
        let zero = milli_graph::ops::CastLike::push_new(&mut graph, zero, x, rng);
        let pos = milli_graph::ops::SimpleBinary::max(&mut graph, zero, x, rng);
        let alpha = milli_graph::ops::Constant::new_scalar(&mut graph, self.alpha, rng);
        let alpha = milli_graph::ops::CastLike::push_new(&mut graph, alpha, x, rng);
        let one = milli_graph::ops::Constant::new_scalar(&mut graph, 1.0f32, rng);
        let one = milli_graph::ops::CastLike::push_new(&mut graph, one, x, rng);
        let exp_x = milli_graph::ops::SimpleUnaryOp::exp(&mut graph, x, rng);
        let exp_m1 = milli_graph::ops::SimpleBinary::sub(&mut graph, exp_x, one, rng);
        let alpha_exp_m1 = milli_graph::ops::SimpleBinary::mul(&mut graph, alpha, exp_m1, rng);
        let neg = milli_graph::ops::SimpleBinary::min(&mut graph, zero, alpha_exp_m1, rng);
        let out_tid = milli_graph::ops::SimpleBinary::add(&mut graph, pos, neg, rng);
        let mut output_map = HashMap::new();
        output_map.insert(out_tid, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

/// ONNX Selu: gamma * (alpha * exp(x) - alpha) for x <= 0, gamma * x for x > 0
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SeluOperation {
    global_id: GlobalId,
    input: GlobalId,
    output: GlobalId,
    alpha: f32,
    gamma: f32,
}

impl SeluOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Selu"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Selu"));
        }
        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Selu"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Selu"))?,
            alpha: query_attribute_float(attributes, "alpha")
                .unwrap_or(1.6732632423543772),
            gamma: query_attribute_float(attributes, "gamma")
                .unwrap_or(1.0507009873554805),
        })
    }
}

impl Node for SeluOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Selu".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for SeluOperation {
    fn parameters(&self) -> Vec<Property> {
        vec![
            Property::new("alpha", PropertyValue::Float(self.alpha.into())),
            Property::new("gamma", PropertyValue::Float(self.gamma.into())),
        ]
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        // Selu(x) = gamma * (alpha * exp(x) - alpha) for x <= 0, gamma * x for x > 0
        // = gamma * elu(x, alpha)
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let x = input_map[&self.input];
        let zero = milli_graph::ops::Constant::new_scalar(&mut graph, 0.0f32, rng);
        let zero = milli_graph::ops::CastLike::push_new(&mut graph, zero, x, rng);
        let pos = milli_graph::ops::SimpleBinary::max(&mut graph, zero, x, rng);
        let alpha = milli_graph::ops::Constant::new_scalar(&mut graph, self.alpha, rng);
        let alpha = milli_graph::ops::CastLike::push_new(&mut graph, alpha, x, rng);
        let one = milli_graph::ops::Constant::new_scalar(&mut graph, 1.0f32, rng);
        let one = milli_graph::ops::CastLike::push_new(&mut graph, one, x, rng);
        let exp_x = milli_graph::ops::SimpleUnaryOp::exp(&mut graph, x, rng);
        let exp_m1 = milli_graph::ops::SimpleBinary::sub(&mut graph, exp_x, one, rng);
        let alpha_exp_m1 = milli_graph::ops::SimpleBinary::mul(&mut graph, alpha, exp_m1, rng);
        let neg = milli_graph::ops::SimpleBinary::min(&mut graph, zero, alpha_exp_m1, rng);
        let elu = milli_graph::ops::SimpleBinary::add(&mut graph, pos, neg, rng);
        let gamma = milli_graph::ops::Constant::new_scalar(&mut graph, self.gamma, rng);
        let gamma = milli_graph::ops::CastLike::push_new(&mut graph, gamma, x, rng);
        let out_tid = milli_graph::ops::SimpleBinary::mul(&mut graph, gamma, elu, rng);
        let mut output_map = HashMap::new();
        output_map.insert(out_tid, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

/// ONNX Celu: max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CeluOperation {
    global_id: GlobalId,
    input: GlobalId,
    output: GlobalId,
    alpha: f32,
}

impl CeluOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Celu"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Celu"));
        }
        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Celu"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Celu"))?,
            alpha: query_attribute_float(attributes, "alpha").unwrap_or(1.0),
        })
    }
}

impl Node for CeluOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Celu".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for CeluOperation {
    fn parameters(&self) -> Vec<Property> {
        vec![Property::new(
            "alpha",
            PropertyValue::Float(self.alpha.into()),
        )]
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        // Celu(x) = max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let x = input_map[&self.input];
        let zero = milli_graph::ops::Constant::new_scalar(&mut graph, 0.0f32, rng);
        let zero = milli_graph::ops::CastLike::push_new(&mut graph, zero, x, rng);
        let pos = milli_graph::ops::SimpleBinary::max(&mut graph, zero, x, rng);
        let alpha = milli_graph::ops::Constant::new_scalar(&mut graph, self.alpha, rng);
        let alpha = milli_graph::ops::CastLike::push_new(&mut graph, alpha, x, rng);
        let one = milli_graph::ops::Constant::new_scalar(&mut graph, 1.0f32, rng);
        let one = milli_graph::ops::CastLike::push_new(&mut graph, one, x, rng);
        let x_over_alpha = milli_graph::ops::SimpleBinary::div(&mut graph, x, alpha, rng);
        let exp_val = milli_graph::ops::SimpleUnaryOp::exp(&mut graph, x_over_alpha, rng);
        let exp_m1 = milli_graph::ops::SimpleBinary::sub(&mut graph, exp_val, one, rng);
        let alpha_exp_m1 = milli_graph::ops::SimpleBinary::mul(&mut graph, alpha, exp_m1, rng);
        let neg = milli_graph::ops::SimpleBinary::min(&mut graph, zero, alpha_exp_m1, rng);
        let out_tid = milli_graph::ops::SimpleBinary::add(&mut graph, pos, neg, rng);
        let mut output_map = HashMap::new();
        output_map.insert(out_tid, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

/// ONNX HardSigmoid: max(0, min(1, alpha * x + beta))
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HardSigmoidOperation {
    global_id: GlobalId,
    input: GlobalId,
    output: GlobalId,
    alpha: f32,
    beta: f32,
}

impl HardSigmoidOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("HardSigmoid"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("HardSigmoid"));
        }
        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("HardSigmoid"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("HardSigmoid"))?,
            alpha: query_attribute_float(attributes, "alpha").unwrap_or(0.2),
            beta: query_attribute_float(attributes, "beta").unwrap_or(0.5),
        })
    }
}

impl Node for HardSigmoidOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "HardSigmoid".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for HardSigmoidOperation {
    fn parameters(&self) -> Vec<Property> {
        vec![
            Property::new("alpha", PropertyValue::Float(self.alpha.into())),
            Property::new("beta", PropertyValue::Float(self.beta.into())),
        ]
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        // HardSigmoid(x) = max(0, min(1, alpha * x + beta))
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let x = input_map[&self.input];
        let alpha = milli_graph::ops::Constant::new_scalar(&mut graph, self.alpha, rng);
        let alpha = milli_graph::ops::CastLike::push_new(&mut graph, alpha, x, rng);
        let beta = milli_graph::ops::Constant::new_scalar(&mut graph, self.beta, rng);
        let beta = milli_graph::ops::CastLike::push_new(&mut graph, beta, x, rng);
        let zero = milli_graph::ops::Constant::new_scalar(&mut graph, 0.0f32, rng);
        let zero = milli_graph::ops::CastLike::push_new(&mut graph, zero, x, rng);
        let one = milli_graph::ops::Constant::new_scalar(&mut graph, 1.0f32, rng);
        let one = milli_graph::ops::CastLike::push_new(&mut graph, one, x, rng);
        let ax = milli_graph::ops::SimpleBinary::mul(&mut graph, alpha, x, rng);
        let axb = milli_graph::ops::SimpleBinary::add(&mut graph, ax, beta, rng);
        let clamped_low = milli_graph::ops::SimpleBinary::max(&mut graph, axb, zero, rng);
        let out_tid = milli_graph::ops::SimpleBinary::min(&mut graph, clamped_low, one, rng);
        let mut output_map = HashMap::new();
        output_map.insert(out_tid, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

/// ONNX HardSwish: x * max(0, min(1, x/6 + 0.5))
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HardSwishOperation {
    global_id: GlobalId,
    input: GlobalId,
    output: GlobalId,
}

impl HardSwishOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        _attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("HardSwish"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("HardSwish"));
        }
        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("HardSwish"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("HardSwish"))?,
        })
    }
}

impl Node for HardSwishOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "HardSwish".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for HardSwishOperation {
    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        // HardSwish(x) = x * HardSigmoid(x, alpha=1/6, beta=0.5)
        //              = x * max(0, min(1, x/6 + 0.5))
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let x = input_map[&self.input];
        let sixth = milli_graph::ops::Constant::new_scalar(&mut graph, 1.0f32 / 6.0, rng);
        let sixth = milli_graph::ops::CastLike::push_new(&mut graph, sixth, x, rng);
        let half = milli_graph::ops::Constant::new_scalar(&mut graph, 0.5f32, rng);
        let half = milli_graph::ops::CastLike::push_new(&mut graph, half, x, rng);
        let zero = milli_graph::ops::Constant::new_scalar(&mut graph, 0.0f32, rng);
        let zero = milli_graph::ops::CastLike::push_new(&mut graph, zero, x, rng);
        let one = milli_graph::ops::Constant::new_scalar(&mut graph, 1.0f32, rng);
        let one = milli_graph::ops::CastLike::push_new(&mut graph, one, x, rng);
        let x6 = milli_graph::ops::SimpleBinary::mul(&mut graph, x, sixth, rng);
        let x6h = milli_graph::ops::SimpleBinary::add(&mut graph, x6, half, rng);
        let clamped_low = milli_graph::ops::SimpleBinary::max(&mut graph, x6h, zero, rng);
        let hard_sig = milli_graph::ops::SimpleBinary::min(&mut graph, clamped_low, one, rng);
        let out_tid = milli_graph::ops::SimpleBinary::mul(&mut graph, x, hard_sig, rng);
        let mut output_map = HashMap::new();
        output_map.insert(out_tid, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

/// ONNX Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MishOperation {
    global_id: GlobalId,
    input: GlobalId,
    output: GlobalId,
}

impl MishOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        _attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Mish"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Mish"));
        }
        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Mish"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Mish"))?,
        })
    }
}

impl Node for MishOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Mish".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for MishOperation {
    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        // Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let x = input_map[&self.input];
        let exp_x = milli_graph::ops::SimpleUnaryOp::exp(&mut graph, x, rng);
        let one = milli_graph::ops::Constant::new_scalar(&mut graph, 1.0f32, rng);
        let one = milli_graph::ops::CastLike::push_new(&mut graph, one, x, rng);
        let one_plus_exp = milli_graph::ops::SimpleBinary::add(&mut graph, exp_x, one, rng);
        let softplus = milli_graph::ops::SimpleUnaryOp::ln(&mut graph, one_plus_exp, rng);
        let tanh_sp =
            milli_graph::ops::SimpleUnaryOp::trig(&mut graph, softplus, TrigOp::Tanh, rng);
        let out_tid = milli_graph::ops::SimpleBinary::mul(&mut graph, x, tanh_sp, rng);
        let mut output_map = HashMap::new();
        output_map.insert(out_tid, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

/// ONNX Softsign: x / (1 + |x|)
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SoftsignOperation {
    global_id: GlobalId,
    input: GlobalId,
    output: GlobalId,
}

impl SoftsignOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        _attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Softsign"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Softsign"));
        }
        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Softsign"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Softsign"))?,
        })
    }
}

impl Node for SoftsignOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Softsign".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for SoftsignOperation {
    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        // Softsign(x) = x / (1 + |x|)
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let x = input_map[&self.input];
        let abs_x = milli_graph::ops::SimpleUnaryOp::abs(&mut graph, x, rng);
        let one = milli_graph::ops::Constant::new_scalar(&mut graph, 1.0f32, rng);
        let one = milli_graph::ops::CastLike::push_new(&mut graph, one, x, rng);
        let denom = milli_graph::ops::SimpleBinary::add(&mut graph, one, abs_x, rng);
        let out_tid = milli_graph::ops::SimpleBinary::div(&mut graph, x, denom, rng);
        let mut output_map = HashMap::new();
        output_map.insert(out_tid, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

/// ONNX ThresholdedRelu: x if x > alpha else 0
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThresholdedReluOperation {
    global_id: GlobalId,
    input: GlobalId,
    output: GlobalId,
    alpha: f32,
}

impl ThresholdedReluOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("ThresholdedRelu"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("ThresholdedRelu"));
        }
        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("ThresholdedRelu"))?,
            output: outputs[0]
                .ok_or(ONNXDecodingError::InvalidOperatorOutputs("ThresholdedRelu"))?,
            alpha: query_attribute_float(attributes, "alpha").unwrap_or(1.0),
        })
    }
}

impl Node for ThresholdedReluOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "ThresholdedRelu".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for ThresholdedReluOperation {
    fn parameters(&self) -> Vec<Property> {
        vec![Property::new(
            "alpha",
            PropertyValue::Float(self.alpha.into()),
        )]
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        // ThresholdedRelu(x) = x if x > alpha else 0
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let x = input_map[&self.input];
        let alpha = milli_graph::ops::Constant::new_scalar(&mut graph, self.alpha, rng);
        let alpha = milli_graph::ops::CastLike::push_new(&mut graph, alpha, x, rng);
        let zero = milli_graph::ops::Constant::new_scalar(&mut graph, 0.0f32, rng);
        let zero = milli_graph::ops::CastLike::push_new(&mut graph, zero, x, rng);
        let cond = milli_graph::ops::SimpleBinary::greater(&mut graph, x, alpha, rng);
        let out_tid = milli_graph::ops::Where::push_new(&mut graph, cond, x, zero, rng);
        let mut output_map = HashMap::new();
        output_map.insert(out_tid, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

/// ONNX PRelu: slope * x for x < 0, x for x >= 0
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PReluOperation {
    global_id: GlobalId,
    input: GlobalId,
    slope: GlobalId,
    output: GlobalId,
}

impl PReluOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        _attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("PRelu"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("PRelu"));
        }
        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("PRelu"))?,
            slope: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("PRelu"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("PRelu"))?,
        })
    }
}

impl Node for PReluOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "PRelu".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new([self.input, self.slope].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for PReluOperation {
    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        // PRelu(x, slope) = x if x >= 0, slope * x if x < 0
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let x = input_map[&self.input];
        let slope = input_map[&self.slope];
        let zero = milli_graph::ops::Constant::new_scalar(&mut graph, 0.0f32, rng);
        let zero = milli_graph::ops::CastLike::push_new(&mut graph, zero, x, rng);
        let cond = milli_graph::ops::SimpleBinary::greater_or_equal(&mut graph, x, zero, rng);
        let slope_x = milli_graph::ops::SimpleBinary::mul(&mut graph, slope, x, rng);
        let out_tid = milli_graph::ops::Where::push_new(&mut graph, cond, x, slope_x, rng);
        let mut output_map = HashMap::new();
        output_map.insert(out_tid, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

/// ONNX Runtime contrib op: BiasGelu(x, bias) = Gelu(x + bias).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BiasGeluOperation {
    global_id: GlobalId,
    input: GlobalId,
    bias: GlobalId,
    output: GlobalId,
}

impl BiasGeluOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        _attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("BiasGelu"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("BiasGelu"));
        }
        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("BiasGelu"))?,
            bias: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("BiasGelu"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("BiasGelu"))?,
        })
    }
}

impl Node for BiasGeluOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "BiasGelu".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new([self.input, self.bias].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for BiasGeluOperation {
    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let x = input_map[&self.input];
        let bias = input_map[&self.bias];
        let biased = milli_graph::ops::SimpleBinary::add(&mut graph, x, bias, rng);
        let out_tid = push_gelu(&mut graph, biased, rng);
        let mut output_map = HashMap::new();
        output_map.insert(out_tid, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
