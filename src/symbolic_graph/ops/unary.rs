use crate::dtype::DType;
use crate::graph::Node;
use crate::milli_graph::MilliOpGraph;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{ONNXDecodingError, SymbolicGraphTensorId, query_attribute_int};
use crate::{TrigOp, milli_graph, onnx};
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
    input: SymbolicGraphTensorId,
    output: SymbolicGraphTensorId,
    which: WhichUnaryOperation,
}

impl UnaryOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        which: WhichUnaryOperation,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Unary"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Unary"));
        }

        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Unary"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Unary"))?,
            which,
        })
    }
}

impl Node<SymbolicGraphTensorId> for UnaryOperation {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        match self.which {
            WhichUnaryOperation::Trig(trig) => trig.to_string(),
            _ => self.which.to_string(),
        }
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new(std::iter::once(self.output))
    }
}
impl Operation for UnaryOperation {
    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs());
        let a = input_map[&self.input];
        if let WhichUnaryOperation::NonZero = &self.which {
            let out_tid = milli_graph::ops::NonZero::push_new(&mut graph, a);
            let mut output_map = HashMap::new();
            output_map.insert(out_tid, self.output);
            graph.set_output_map(output_map);
            return graph;
        }
        let out_tid = match &self.which {
            WhichUnaryOperation::Relu => milli_graph::ops::ClampMin::push_new(&mut graph, a, 0.0),
            WhichUnaryOperation::Sigmoid => {
                let xn = milli_graph::ops::Cast::push_new(&mut graph, a, DType::F32);
                let xn = milli_graph::ops::SimpleUnaryOp::neg(&mut graph, xn);
                let xn = milli_graph::ops::SimpleUnaryOp::exp(&mut graph, xn);
                let c_tid = milli_graph::ops::Constant::new_scalar(&mut graph, 1.0f32);
                let c = milli_graph::ops::CastLike::push_new(&mut graph, c_tid, xn);
                let o = milli_graph::ops::SimpleBinary::add(&mut graph, xn, c);
                let o = milli_graph::ops::SimpleUnaryOp::reciprocal(&mut graph, o);
                milli_graph::ops::CastLike::push_new(&mut graph, o, a)
            }
            WhichUnaryOperation::Exp => milli_graph::ops::SimpleUnaryOp::exp(&mut graph, a),
            WhichUnaryOperation::Log => milli_graph::ops::SimpleUnaryOp::ln(&mut graph, a),
            WhichUnaryOperation::Softplus => {
                let x = milli_graph::ops::SimpleUnaryOp::exp(&mut graph, a);
                let c_tid = milli_graph::ops::Constant::new_scalar(&mut graph, 1.0f32);
                let c = milli_graph::ops::CastLike::push_new(&mut graph, c_tid, x);
                let x = milli_graph::ops::SimpleBinary::add(&mut graph, x, c);
                milli_graph::ops::SimpleUnaryOp::ln(&mut graph, x)
            }
            WhichUnaryOperation::Neg => milli_graph::ops::SimpleUnaryOp::neg(&mut graph, a),
            WhichUnaryOperation::Sqrt => milli_graph::ops::SimpleUnaryOp::sqrt(&mut graph, a),
            WhichUnaryOperation::Abs => milli_graph::ops::SimpleUnaryOp::abs(&mut graph, a),
            WhichUnaryOperation::Trig(trig_op) => {
                milli_graph::ops::SimpleUnaryOp::trig(&mut graph, a, *trig_op)
            }
            WhichUnaryOperation::Reciprocal => {
                milli_graph::ops::SimpleUnaryOp::reciprocal(&mut graph, a)
            }
            WhichUnaryOperation::BitwiseNot => {
                milli_graph::ops::SimpleUnaryOp::bitwise_not(&mut graph, a)
            }
            WhichUnaryOperation::Not => milli_graph::ops::SimpleUnaryOp::not(&mut graph, a),
            WhichUnaryOperation::Sign => milli_graph::ops::SimpleUnaryOp::sign(&mut graph, a),
            WhichUnaryOperation::Floor => milli_graph::ops::SimpleUnaryOp::floor(&mut graph, a),
            WhichUnaryOperation::Ceil => milli_graph::ops::SimpleUnaryOp::ceil(&mut graph, a),
            WhichUnaryOperation::Round => milli_graph::ops::SimpleUnaryOp::round(&mut graph, a),
            WhichUnaryOperation::IsNan => milli_graph::ops::SimpleUnaryOp::is_nan(&mut graph, a),
            WhichUnaryOperation::Erf => milli_graph::ops::SimpleUnaryOp::erf(&mut graph, a),
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
    axis: Option<i64>,
    input: SymbolicGraphTensorId,
    output: SymbolicGraphTensorId,
}

impl SoftmaxOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Softmax"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Softmax"));
        }

        Ok(Self {
            axis: query_attribute_int(attributes, "axis"),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Softmax"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Softmax"))?,
        })
    }
}

impl Node<SymbolicGraphTensorId> for SoftmaxOperation {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "Softmax".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new(std::iter::once(self.output))
    }
}
impl Operation for SoftmaxOperation {
    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs());

        let e = milli_graph::ops::SimpleUnaryOp::exp(&mut graph, input_map[&self.input]);
        let axis_tid = milli_graph::ops::Constant::new_scalar(&mut graph, self.axis.unwrap_or(-1));
        let sum = milli_graph::ops::ReduceSum::push_new(&mut graph, e, Some(axis_tid), true, false);
        let out_tid = milli_graph::ops::SimpleBinary::div(&mut graph, e, sum);
        let mut output_map = HashMap::new();
        output_map.insert(out_tid, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LogSoftmaxOperation {
    axis: Option<i64>,
    input: SymbolicGraphTensorId,
    output: SymbolicGraphTensorId,
}

impl LogSoftmaxOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("LogSoftmax"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("LogSoftmax"));
        }

        Ok(Self {
            axis: query_attribute_int(attributes, "axis"),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("LogSoftmax"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("LogSoftmax"))?,
        })
    }
}

impl Node<SymbolicGraphTensorId> for LogSoftmaxOperation {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "LogSoftmax".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for LogSoftmaxOperation {
    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs());

        let e_tid = milli_graph::ops::SimpleUnaryOp::exp(&mut graph, input_map[&self.input]);
        let axis_tid = milli_graph::ops::Constant::new_scalar(&mut graph, self.axis.unwrap_or(-1));
        let sum_tid =
            milli_graph::ops::ReduceSum::push_new(&mut graph, e_tid, Some(axis_tid), true, false);
        let softmax_tid = milli_graph::ops::SimpleBinary::div(&mut graph, e_tid, sum_tid);
        let out_tid = milli_graph::ops::SimpleUnaryOp::ln(&mut graph, softmax_tid);
        let mut output_map = HashMap::new();
        output_map.insert(out_tid, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IsInfOperation {
    input: SymbolicGraphTensorId,
    output: SymbolicGraphTensorId,
    detect_negative: Option<bool>,
    detect_positive: Option<bool>,
}

impl IsInfOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        attributes: &[onnx::AttributeProto],
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
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("IsInf"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("IsInf"))?,
            detect_negative,
            detect_positive,
        })
    }
}

impl Node<SymbolicGraphTensorId> for IsInfOperation {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "Is Inf".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new(std::iter::once(self.output))
    }
}
impl Operation for IsInfOperation {
    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs());
        let input = input_map[&self.input];
        let out_tid = milli_graph::ops::SimpleUnaryOp::is_inf(
            &mut graph,
            input,
            self.detect_positive.unwrap_or(true),
            self.detect_negative.unwrap_or(true),
        );
        let mut output_map = HashMap::new();
        output_map.insert(out_tid, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IdentityOperation {
    input: SymbolicGraphTensorId,
    output: SymbolicGraphTensorId,
}

impl IdentityOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        _attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Identity"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Identity"));
        }

        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Identity"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Identity"))?,
        })
    }
}

impl Node<SymbolicGraphTensorId> for IdentityOperation {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "Identity".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new(std::iter::once(self.output))
    }
}
impl Operation for IdentityOperation {
    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs());
        let input = input_map[&self.input];
        let mut output_map = HashMap::new();
        output_map.insert(input, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
