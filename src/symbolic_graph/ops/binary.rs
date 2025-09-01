use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::*;
use crate::onnx;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{
    ONNXDecodingError, SymbolicGraphTensorId, query_attribute_bool, query_attribute_float,
    query_attribute_int,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, strum_macros::Display, Serialize, Deserialize)]
pub enum WhichBinaryOperation {
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
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
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BinaryOperation {
    a: SymbolicGraphTensorId,
    b: SymbolicGraphTensorId,
    output: SymbolicGraphTensorId,
    which: WhichBinaryOperation,
}

impl BinaryOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        which: WhichBinaryOperation,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Binary"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Binary"));
        }
        Ok(BinaryOperation {
            a: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Binary"))?,
            b: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Binary"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Binary"))?,
            which,
        })
    }
}

impl Operation for BinaryOperation {
    fn get_op_type_name(&self) -> String {
        self.which.to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.a, self.b]
    }

    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let a = input_map[&self.a];
        let b = input_map[&self.b];
        let res = match self.which {
            WhichBinaryOperation::Add => AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::add(a, b)),
            WhichBinaryOperation::Sub => AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::sub(a, b)),
            WhichBinaryOperation::Mul => AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::mul(a, b)),
            WhichBinaryOperation::Div => AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::div(a, b)),
            WhichBinaryOperation::MatMul => AnyMilliOp::MatMul(MilliOpMatMul::new(a, b)),
            WhichBinaryOperation::And => AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::and(a, b)),
            WhichBinaryOperation::Or => AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::or(a, b)),
            WhichBinaryOperation::Xor => AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::xor(a, b)),
            WhichBinaryOperation::BitwiseAnd => {
                AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::bitwise_and(a, b))
            }
            WhichBinaryOperation::BitwiseOr => {
                AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::bitwise_or(a, b))
            }
            WhichBinaryOperation::BitwiseXor => {
                AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::bitwise_xor(a, b))
            }
            WhichBinaryOperation::Equal => {
                AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::equal(a, b))
            }
            WhichBinaryOperation::Greater => {
                AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::greater(a, b))
            }
            WhichBinaryOperation::GreaterOrEqual => {
                AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::greater_or_equal(a, b))
            }
            WhichBinaryOperation::Less => AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::less(a, b)),
            WhichBinaryOperation::LessOrEqual => {
                AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::less_or_equal(a, b))
            }
        };
        let mut output_map = HashMap::new();
        let out = graph.push_op(res);
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PowOperation {
    input_x: SymbolicGraphTensorId,
    input_y: SymbolicGraphTensorId,
    output: SymbolicGraphTensorId,
}

impl PowOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        _attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Pow"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Pow"));
        }
        Ok(Self {
            input_x: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Pow"))?,
            input_y: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Pow"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Pow"))?,
        })
    }
}

impl Operation for PowOperation {
    fn get_op_type_name(&self) -> String {
        "Pow".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.input_x, self.input_y]
    }
    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let out = graph.push_op(AnyMilliOp::Pow(MilliOpPow::new(
            input_map[&self.input_x],
            input_map[&self.input_y],
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GemmOperation {
    alpha: Option<f32>,
    beta: Option<f32>,
    trans_a: Option<bool>,
    trans_b: Option<bool>,
    input_a: SymbolicGraphTensorId,
    input_b: SymbolicGraphTensorId,
    input_c: Option<SymbolicGraphTensorId>,
    output: SymbolicGraphTensorId,
}

impl GemmOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() < 2 || inputs.len() > 3 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Gemm"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Gemm"));
        }

        let trans_a = query_attribute_bool(attributes, "transA");
        let trans_b = query_attribute_bool(attributes, "transB");
        let alpha = query_attribute_float(attributes, "alpha");
        let beta = query_attribute_float(attributes, "beta");

        Ok(Self {
            trans_a,
            trans_b,
            alpha,
            beta,
            input_a: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Gemm"))?,
            input_b: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Gemm"))?,
            input_c: if inputs.len() > 2 {
                Some(inputs[2].ok_or(ONNXDecodingError::InvalidOperatorInputs("Gemm"))?)
            } else {
                None
            },
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Gemm"))?,
        })
    }
}

impl Operation for GemmOperation {
    fn get_op_type_name(&self) -> String {
        "Gemm".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        if let Some(input_c) = self.input_c {
            vec![self.input_a, self.input_b, input_c]
        } else {
            vec![self.input_a, self.input_b]
        }
    }
    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());

        let a = input_map[&self.input_a];
        let b = input_map[&self.input_b];

        let a = if let Some(trans_a) = self.trans_a {
            if trans_a {
                graph.push_op(AnyMilliOp::Transpose(MilliOpTranspose::new(a, None)))
            } else {
                a
            }
        } else {
            a
        };

        let b = if let Some(trans_b) = self.trans_b {
            if trans_b {
                graph.push_op(AnyMilliOp::Transpose(MilliOpTranspose::new(b, None)))
            } else {
                b
            }
        } else {
            b
        };

        let x = graph.push_op(AnyMilliOp::MatMul(MilliOpMatMul::new(a, b)));

        let x = if let Some(alpha) = self.alpha {
            let alpha_const =
                graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(alpha)));
            let alpha_const =
                graph.push_op(AnyMilliOp::CastLike(MilliOpCastLike::new(alpha_const, x)));
            graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::mul(
                x,
                alpha_const,
            )))
        } else {
            x
        };

        let x = if let Some(c) = self.input_c {
            let c = input_map[&c];
            let c = if let Some(beta) = self.beta {
                let beta_const =
                    graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(beta)));
                let beta_const =
                    graph.push_op(AnyMilliOp::CastLike(MilliOpCastLike::new(beta_const, c)));
                graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::mul(
                    c, beta_const,
                )))
            } else {
                c
            };
            graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::add(x, c)))
        } else {
            x
        };

        let mut output_map = HashMap::new();
        output_map.insert(x, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArgMaxOperation {
    axis: i64,
    keepdims: bool,
    select_last_index: bool,
    input: SymbolicGraphTensorId,
    output: SymbolicGraphTensorId,
}

impl ArgMaxOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("ArgMax"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("ArgMax"));
        }

        let axis = query_attribute_int(attributes, "axis").unwrap_or(0);
        let keepdims = query_attribute_bool(attributes, "keepdims").unwrap_or(true);
        let select_last_index =
            query_attribute_bool(attributes, "select_last_index").unwrap_or(false);

        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("ArgMax"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("ArgMax"))?,
            axis,
            keepdims,
            select_last_index,
        })
    }
}

impl Operation for ArgMaxOperation {
    fn get_op_type_name(&self) -> String {
        "ArgMax".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.input]
    }

    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());

        let x = graph.push_op(AnyMilliOp::ArgMax(MilliOpArgMax::new(
            input_map[&self.input],
            self.axis,
            self.keepdims,
            self.select_last_index,
        )));

        let mut output_map = HashMap::new();
        output_map.insert(x, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArgMinOperation {
    axis: i64,
    keepdims: bool,
    select_last_index: bool,
    input: SymbolicGraphTensorId,
    output: SymbolicGraphTensorId,
}

impl ArgMinOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("ArgMin"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("ArgMin"));
        }

        let axis = query_attribute_int(attributes, "axis").unwrap_or(0);
        let keepdims = query_attribute_bool(attributes, "keepdims").unwrap_or(true);
        let select_last_index =
            query_attribute_bool(attributes, "select_last_index").unwrap_or(false);

        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("ArgMin"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("ArgMin"))?,
            axis,
            keepdims,
            select_last_index,
        })
    }
}

impl Operation for ArgMinOperation {
    fn get_op_type_name(&self) -> String {
        "ArgMin".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.input]
    }

    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());

        let x = graph.push_op(AnyMilliOp::ArgMin(MilliOpArgMin::new(
            input_map[&self.input],
            self.axis,
            self.keepdims,
            self.select_last_index,
        )));

        let mut output_map = HashMap::new();
        output_map.insert(x, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaxOperation {
    inputs: Vec<SymbolicGraphTensorId>,
    output: SymbolicGraphTensorId,
}

impl MaxOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        _attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.is_empty() {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Max"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Max"));
        }

        Ok(Self {
            inputs: inputs
                .iter()
                .map(|x| x.ok_or(ONNXDecodingError::InvalidOperatorInputs("Max")))
                .collect::<Result<_, _>>()?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Max"))?,
        })
    }
}

impl Operation for MaxOperation {
    fn get_op_type_name(&self) -> String {
        "Max".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        self.inputs.clone()
    }

    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let mut x = input_map[&self.inputs[0]];
        for input in &self.inputs[1..] {
            let y = input_map[input];
            x = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::max(x, y)))
        }
        let mut output_map = HashMap::new();
        output_map.insert(x, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MinOperation {
    inputs: Vec<SymbolicGraphTensorId>,
    output: SymbolicGraphTensorId,
}

impl MinOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        _attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.is_empty() {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Min"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Min"));
        }

        Ok(Self {
            inputs: inputs
                .iter()
                .map(|x| x.ok_or(ONNXDecodingError::InvalidOperatorInputs("Min")))
                .collect::<Result<_, _>>()?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Min"))?,
        })
    }
}

impl Operation for MinOperation {
    fn get_op_type_name(&self) -> String {
        "Min".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        self.inputs.clone()
    }

    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let mut x = input_map[&self.inputs[0]];
        for input in &self.inputs[1..] {
            let y = input_map[input];
            x = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::min(x, y)))
        }
        let mut output_map = HashMap::new();
        output_map.insert(x, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModuloOperation {
    a: SymbolicGraphTensorId,
    b: SymbolicGraphTensorId,
    output: SymbolicGraphTensorId,
    fmod: Option<bool>,
}

impl ModuloOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Modulo"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Modulo"));
        }

        let fmod = query_attribute_int(attributes, "fmod").map(|x| x != 0);

        Ok(Self {
            a: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Modulo"))?,
            b: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Modulo"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Modulo"))?,
            fmod,
        })
    }
}

impl Operation for ModuloOperation {
    fn get_op_type_name(&self) -> String {
        "Modulo".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.a, self.b]
    }

    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let a = input_map[&self.a];
        let b = input_map[&self.b];

        let output = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::modulo(
            a, b, self.fmod,
        )));
        let mut output_map = HashMap::new();
        output_map.insert(output, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
