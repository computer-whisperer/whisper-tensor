use crate::dtype::DType;
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::*;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{ONNXDecodingError, SymbolicGraphTensorId, query_attribute_int};
use crate::{TrigOp, onnx};
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

impl Operation for UnaryOperation {
    fn get_op_type_name(&self) -> String {
        match self.which {
            WhichUnaryOperation::Trig(trig) => trig.to_string(),
            _ => self.which.to_string(),
        }
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.input]
    }

    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let a = input_map[&self.input];
        let res = match &self.which {
            WhichUnaryOperation::Relu => AnyMilliOp::ClampMin(MilliOpClampMin::new(a, 0.0)),
            WhichUnaryOperation::Sigmoid => {
                let xn = graph.push_op(AnyMilliOp::Cast(MilliOpCast::new(a, DType::F32)));
                let xn = graph.push_op(AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::neg(xn)));
                let xn = graph.push_op(AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::exp(xn)));
                let c = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(1.0f32)));
                let c = graph.push_op(AnyMilliOp::CastLike(MilliOpCastLike::new(c, xn)));
                let o = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::add(xn, c)));
                let o = graph.push_op(AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::reciprocal(o)));
                AnyMilliOp::CastLike(MilliOpCastLike::new(o, a))
            }
            WhichUnaryOperation::Exp => AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::exp(a)),
            WhichUnaryOperation::Log => AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::ln(a)),
            WhichUnaryOperation::Softplus => {
                let x = graph.push_op(AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::exp(a)));
                let c = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(1.0f32)));
                let c = graph.push_op(AnyMilliOp::CastLike(MilliOpCastLike::new(c, x)));
                let x = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::add(x, c)));
                AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::ln(x))
            }
            WhichUnaryOperation::Neg => AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::neg(a)),
            WhichUnaryOperation::NonZero => AnyMilliOp::NonZero(MilliOpNonZero::new(a)),
            WhichUnaryOperation::Sqrt => AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::sqrt(a)),
            WhichUnaryOperation::Abs => AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::abs(a)),
            WhichUnaryOperation::Trig(trig_op) => {
                AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::trig(a, *trig_op))
            }
            WhichUnaryOperation::Reciprocal => {
                AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::reciprocal(a))
            }
            WhichUnaryOperation::BitwiseNot => {
                AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::bitwise_not(a))
            }
            WhichUnaryOperation::Not => AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::not(a)),
            WhichUnaryOperation::Sign => AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::sign(a)),
            WhichUnaryOperation::Floor => AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::floor(a)),
            WhichUnaryOperation::Ceil => AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::ceil(a)),
            WhichUnaryOperation::Round => AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::round(a)),
            WhichUnaryOperation::IsNan => AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::is_nan(a)),
            WhichUnaryOperation::Erf => AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::erf(a)),
        };
        let mut output_map = HashMap::new();
        let out = graph.push_op(res);
        output_map.insert(out, self.output);
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

impl Operation for SoftmaxOperation {
    fn get_op_type_name(&self) -> String {
        "Softmax".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.input]
    }
    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());

        let e = graph.push_op(AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::exp(
            input_map[&self.input],
        )));
        let axis_const = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(
            self.axis.unwrap_or(-1),
        )));
        let sum = graph.push_op(AnyMilliOp::ReduceSum(MilliOpReduceSum::new(
            e,
            Some(axis_const),
            true,
            false,
        )));
        let out = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::div(e, sum)));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
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

impl Operation for LogSoftmaxOperation {
    fn get_op_type_name(&self) -> String {
        "LogSoftmax".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.input]
    }
    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());

        let e = graph.push_op(AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::exp(
            input_map[&self.input],
        )));
        let axis_const = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(
            self.axis.unwrap_or(-1),
        )));
        let sum = graph.push_op(AnyMilliOp::ReduceSum(MilliOpReduceSum::new(
            e,
            Some(axis_const),
            true,
            false,
        )));
        let softmax = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::div(e, sum)));
        let out = graph.push_op(AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::ln(softmax)));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
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

impl Operation for IsInfOperation {
    fn get_op_type_name(&self) -> String {
        "Is Inf".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.input]
    }

    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let input = input_map[&self.input];
        let output = graph.push_op(AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::is_inf(
            input,
            self.detect_positive.unwrap_or(true),
            self.detect_negative.unwrap_or(true),
        )));
        let mut output_map = HashMap::new();
        output_map.insert(output, self.output);
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

impl Operation for IdentityOperation {
    fn get_op_type_name(&self) -> String {
        "Identity".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.input]
    }

    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let input = input_map[&self.input];
        let mut output_map = HashMap::new();
        output_map.insert(input, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
