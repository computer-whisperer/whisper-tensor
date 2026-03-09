use crate::dtype::DType;
use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::MilliOpGraph;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{ONNXDecodingError, query_attribute_int};
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

    fn get_milli_op_graph(&self, rng: &mut impl Rng) -> MilliOpGraph {
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

    fn get_milli_op_graph(&self, rng: &mut impl Rng) -> MilliOpGraph {
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

    fn get_milli_op_graph(&self, rng: &mut impl Rng) -> MilliOpGraph {
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

    fn get_milli_op_graph(&self, rng: &mut impl Rng) -> MilliOpGraph {
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
    fn get_milli_op_graph(&self, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let input = input_map[&self.input];
        let mut output_map = HashMap::new();
        output_map.insert(input, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
