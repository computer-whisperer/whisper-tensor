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
        let out_tid = match self.which {
            WhichBinaryOperation::Add => SimpleBinary::add(&mut graph, a, b),
            WhichBinaryOperation::Sub => SimpleBinary::sub(&mut graph, a, b),
            WhichBinaryOperation::Mul => SimpleBinary::mul(&mut graph, a, b),
            WhichBinaryOperation::Div => SimpleBinary::div(&mut graph, a, b),
            WhichBinaryOperation::MatMul => MatMul::push_new(&mut graph, a, b),
            WhichBinaryOperation::And => SimpleBinary::and(&mut graph, a, b),
            WhichBinaryOperation::Or => SimpleBinary::or(&mut graph, a, b),
            WhichBinaryOperation::Xor => SimpleBinary::xor(&mut graph, a, b),
            WhichBinaryOperation::BitwiseAnd => SimpleBinary::bitwise_and(&mut graph, a, b),
            WhichBinaryOperation::BitwiseOr => SimpleBinary::bitwise_or(&mut graph, a, b),
            WhichBinaryOperation::BitwiseXor => SimpleBinary::bitwise_xor(&mut graph, a, b),
            WhichBinaryOperation::Equal => SimpleBinary::equal(&mut graph, a, b),
            WhichBinaryOperation::Greater => SimpleBinary::greater(&mut graph, a, b),
            WhichBinaryOperation::GreaterOrEqual => {
                SimpleBinary::greater_or_equal(&mut graph, a, b)
            }
            WhichBinaryOperation::Less => SimpleBinary::less(&mut graph, a, b),
            WhichBinaryOperation::LessOrEqual => SimpleBinary::less_or_equal(&mut graph, a, b),
        };
        let mut output_map = HashMap::new();
        output_map.insert(out_tid, self.output);
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
        let out = Pow::push_new(
            &mut graph,
            input_map[&self.input_x],
            input_map[&self.input_y],
        );
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
                Transpose::push_new(&mut graph, a, None)
            } else {
                a
            }
        } else {
            a
        };

        let b = if let Some(trans_b) = self.trans_b {
            if trans_b {
                Transpose::push_new(&mut graph, b, None)
            } else {
                b
            }
        } else {
            b
        };

        let x = MatMul::push_new(&mut graph, a, b);

        let x = if let Some(alpha) = self.alpha {
            let alpha_tid = Constant::new_scalar(&mut graph, alpha);
            let alpha_const = CastLike::push_new(&mut graph, alpha_tid, x);
            SimpleBinary::mul(&mut graph, x, alpha_const)
        } else {
            x
        };

        let x = if let Some(c) = self.input_c {
            let c = input_map[&c];
            let c = if let Some(beta) = self.beta {
                let beta_tid = Constant::new_scalar(&mut graph, beta);
                let beta_const = CastLike::push_new(&mut graph, beta_tid, c);
                SimpleBinary::mul(&mut graph, c, beta_const)
            } else {
                c
            };
            SimpleBinary::add(&mut graph, x, c)
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

        let x = ArgMax::push_new(
            &mut graph,
            input_map[&self.input],
            self.axis,
            self.keepdims,
            self.select_last_index,
        );

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

        let x = ArgMin::push_new(
            &mut graph,
            input_map[&self.input],
            self.axis,
            self.keepdims,
            self.select_last_index,
        );

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
            x = SimpleBinary::max(&mut graph, x, y)
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
            x = SimpleBinary::min(&mut graph, x, y)
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

        let output = SimpleBinary::modulo(&mut graph, a, b, self.fmod);
        let mut output_map = HashMap::new();
        output_map.insert(output, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
