use crate::graph::{GlobalId, Node};
use crate::milli_graph::MilliOpGraph;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{
    ONNXDecodingError, SymbolicGraphTensorId, query_attribute_bool, query_attribute_float,
    query_attribute_int,
};
use crate::{milli_graph, onnx};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rand::Rng;

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
    global_id: GlobalId,
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
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Binary"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Binary"));
        }
        Ok(BinaryOperation {
            global_id: GlobalId::new(rng),
            a: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Binary"))?,
            b: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Binary"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Binary"))?,
            which,
        })
    }
}

impl Node<SymbolicGraphTensorId> for BinaryOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        self.which.to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new([self.a, self.b].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new([self.output].into_iter())
    }
}

impl Operation for BinaryOperation {
    fn get_milli_op_graph(&self, rng: &mut impl Rng) -> MilliOpGraph<SymbolicGraphTensorId> {
        let input_ids: Vec<_> = self.inputs().collect();
        let (mut graph, input_map) = MilliOpGraph::new(input_ids, rng);
        let a = input_map[&self.a];
        let b = input_map[&self.b];
        let out_tid = match self.which {
            WhichBinaryOperation::Add => milli_graph::ops::SimpleBinary::add(&mut graph, a, b, rng),
            WhichBinaryOperation::Sub => milli_graph::ops::SimpleBinary::sub(&mut graph, a, b, rng),
            WhichBinaryOperation::Mul => milli_graph::ops::SimpleBinary::mul(&mut graph, a, b, rng),
            WhichBinaryOperation::Div => milli_graph::ops::SimpleBinary::div(&mut graph, a, b, rng),
            WhichBinaryOperation::MatMul => milli_graph::ops::MatMul::push_new(&mut graph, a, b, rng),
            WhichBinaryOperation::And => milli_graph::ops::SimpleBinary::and(&mut graph, a, b, rng),
            WhichBinaryOperation::Or => milli_graph::ops::SimpleBinary::or(&mut graph, a, b, rng),
            WhichBinaryOperation::Xor => milli_graph::ops::SimpleBinary::xor(&mut graph, a, b, rng),
            WhichBinaryOperation::BitwiseAnd => {
                milli_graph::ops::SimpleBinary::bitwise_and(&mut graph, a, b, rng)
            }
            WhichBinaryOperation::BitwiseOr => {
                milli_graph::ops::SimpleBinary::bitwise_or(&mut graph, a, b, rng)
            }
            WhichBinaryOperation::BitwiseXor => {
                milli_graph::ops::SimpleBinary::bitwise_xor(&mut graph, a, b, rng)
            }
            WhichBinaryOperation::Equal => milli_graph::ops::SimpleBinary::equal(&mut graph, a, b, rng),
            WhichBinaryOperation::Greater => {
                milli_graph::ops::SimpleBinary::greater(&mut graph, a, b, rng)
            }
            WhichBinaryOperation::GreaterOrEqual => {
                milli_graph::ops::SimpleBinary::greater_or_equal(&mut graph, a, b, rng)
            }
            WhichBinaryOperation::Less => milli_graph::ops::SimpleBinary::less(&mut graph, a, b, rng),
            WhichBinaryOperation::LessOrEqual => {
                milli_graph::ops::SimpleBinary::less_or_equal(&mut graph, a, b, rng)
            }
        };
        let mut output_map = HashMap::new();
        output_map.insert(out_tid, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PowOperation {
    global_id: GlobalId,
    input_x: SymbolicGraphTensorId,
    input_y: SymbolicGraphTensorId,
    output: SymbolicGraphTensorId,
}

impl PowOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        _attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Pow"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Pow"));
        }
        Ok(Self {
            global_id: GlobalId::new(rng),
            input_x: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Pow"))?,
            input_y: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Pow"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Pow"))?,
        })
    }
}

impl Node<SymbolicGraphTensorId> for PowOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Pow".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new([self.input_x, self.input_y].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new([self.output].into_iter())
    }
}

impl Operation for PowOperation {
    fn get_milli_op_graph(&self, rng: &mut impl Rng) -> MilliOpGraph<SymbolicGraphTensorId> {
        let input_ids: Vec<_> = self.inputs().collect();
        let (mut graph, input_map) = MilliOpGraph::new(input_ids, rng);
        let out = milli_graph::ops::Pow::push_new(
            &mut graph,
            input_map[&self.input_x],
            input_map[&self.input_y],
            rng
        );
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GemmOperation {
    global_id: GlobalId,
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
        rng: &mut impl Rng,
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
            global_id: GlobalId::new(rng),
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

impl Node<SymbolicGraphTensorId> for GemmOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Gemm".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        let v = if let Some(c) = self.input_c {
            vec![self.input_a, self.input_b, c]
        } else {
            vec![self.input_a, self.input_b]
        };
        Box::new(v.into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new([self.output].into_iter())
    }
}

impl Operation for GemmOperation {
    fn get_milli_op_graph(&self, rng: &mut impl Rng) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);

        let a = input_map[&self.input_a];
        let b = input_map[&self.input_b];

        let a = if let Some(trans_a) = self.trans_a {
            if trans_a {
                milli_graph::ops::Transpose::push_new(&mut graph, a, None, rng)
            } else {
                a
            }
        } else {
            a
        };

        let b = if let Some(trans_b) = self.trans_b {
            if trans_b {
                milli_graph::ops::Transpose::push_new(&mut graph, b, None, rng)
            } else {
                b
            }
        } else {
            b
        };

        let x = milli_graph::ops::MatMul::push_new(&mut graph, a, b, rng);

        let x = if let Some(alpha) = self.alpha {
            let alpha_tid = milli_graph::ops::Constant::new_scalar(&mut graph, alpha, rng);
            let alpha_const = milli_graph::ops::CastLike::push_new(&mut graph, alpha_tid, x, rng);
            milli_graph::ops::SimpleBinary::mul(&mut graph, x, alpha_const, rng)
        } else {
            x
        };

        let x = if let Some(c) = self.input_c {
            let c = input_map[&c];
            let c = if let Some(beta) = self.beta {
                let beta_tid = milli_graph::ops::Constant::new_scalar(&mut graph, beta, rng);
                let beta_const = milli_graph::ops::CastLike::push_new(&mut graph, beta_tid, c, rng);
                milli_graph::ops::SimpleBinary::mul(&mut graph, c, beta_const, rng)
            } else {
                c
            };
            milli_graph::ops::SimpleBinary::add(&mut graph, x, c, rng)
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
    global_id: GlobalId,
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
        rng: &mut impl Rng,
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
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("ArgMax"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("ArgMax"))?,
            axis,
            keepdims,
            select_last_index,
        })
    }
}

impl Node<SymbolicGraphTensorId> for ArgMaxOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "ArgMax".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new([self.input].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new([self.output].into_iter())
    }
}

impl Operation for ArgMaxOperation {
    fn get_milli_op_graph(&self, rng: &mut impl Rng) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);

        let x = milli_graph::ops::ArgMax::push_new(
            &mut graph,
            input_map[&self.input],
            self.axis,
            self.keepdims,
            self.select_last_index,
            rng
        );

        let mut output_map = HashMap::new();
        output_map.insert(x, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArgMinOperation {
    global_id: GlobalId,
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
        rng: &mut impl Rng,
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
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("ArgMin"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("ArgMin"))?,
            axis,
            keepdims,
            select_last_index,
        })
    }
}

impl Node<SymbolicGraphTensorId> for ArgMinOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "ArgMin".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new([self.input].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new([self.output].into_iter())
    }
}

impl Operation for ArgMinOperation {
    fn get_milli_op_graph(&self, rng: &mut impl Rng) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);

        let x = milli_graph::ops::ArgMin::push_new(
            &mut graph,
            input_map[&self.input],
            self.axis,
            self.keepdims,
            self.select_last_index,
            rng
        );

        let mut output_map = HashMap::new();
        output_map.insert(x, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaxOperation {
    global_id: GlobalId,
    inputs: Vec<SymbolicGraphTensorId>,
    output: SymbolicGraphTensorId,
}

impl MaxOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        _attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.is_empty() {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Max"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Max"));
        }

        Ok(Self {
            global_id: GlobalId::new(rng),
            inputs: inputs
                .iter()
                .map(|x| x.ok_or(ONNXDecodingError::InvalidOperatorInputs("Max")))
                .collect::<Result<_, _>>()?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Max"))?,
        })
    }
}

impl Node<SymbolicGraphTensorId> for MaxOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Max".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new(self.inputs.clone().into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new([self.output].into_iter())
    }
}

impl Operation for MaxOperation {
    fn get_milli_op_graph(&self, rng: &mut impl Rng) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let mut x = input_map[&self.inputs[0]];
        for input in &self.inputs[1..] {
            let y = input_map[input];
            x = milli_graph::ops::SimpleBinary::max(&mut graph, x, y, rng)
        }
        let mut output_map = HashMap::new();
        output_map.insert(x, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MinOperation {
    global_id: GlobalId,
    inputs: Vec<SymbolicGraphTensorId>,
    output: SymbolicGraphTensorId,
}

impl MinOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        _attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.is_empty() {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Min"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Min"));
        }

        Ok(Self {
            global_id: GlobalId::new(rng),
            inputs: inputs
                .iter()
                .map(|x| x.ok_or(ONNXDecodingError::InvalidOperatorInputs("Min")))
                .collect::<Result<_, _>>()?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Min"))?,
        })
    }
}

impl Node<SymbolicGraphTensorId> for MinOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Min".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new(self.inputs.clone().into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new([self.output].into_iter())
    }
}

impl Operation for MinOperation {
    fn get_milli_op_graph(&self, rng: &mut impl Rng) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let mut x = input_map[&self.inputs[0]];
        for input in &self.inputs[1..] {
            let y = input_map[input];
            x = milli_graph::ops::SimpleBinary::min(&mut graph, x, y, rng)
        }
        let mut output_map = HashMap::new();
        output_map.insert(x, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModuloOperation {
    global_id: GlobalId,
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
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Modulo"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Modulo"));
        }

        let fmod = query_attribute_int(attributes, "fmod").map(|x| x != 0);

        Ok(Self {
            global_id: GlobalId::new(rng),
            a: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Modulo"))?,
            b: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Modulo"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Modulo"))?,
            fmod,
        })
    }
}

impl Node<SymbolicGraphTensorId> for ModuloOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Modulo".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new([self.a, self.b].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new([self.output].into_iter())
    }
}

impl Operation for ModuloOperation {
    fn get_milli_op_graph(&self, rng: &mut impl Rng) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let a = input_map[&self.a];
        let b = input_map[&self.b];

        let output = milli_graph::ops::SimpleBinary::modulo(&mut graph, a, b, self.fmod, rng);
        let mut output_map = HashMap::new();
        output_map.insert(output, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
