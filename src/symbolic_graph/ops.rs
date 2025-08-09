use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::{NDArrayNumericTensor, NDArrayNumericTensorError};
use crate::dtype::{DType, DTypeError};
use crate::milli_graph::ops::*;
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError, ops_helpers};
use crate::numeric_scalar::NumericScalar;
use crate::numeric_tensor::{NumericTensor, NumericTensorError};
use crate::symbolic_graph::{
    ONNXDecodingError, SymbolicGraphInner, SymbolicGraphMutator, TensorId, query_attribute_bool,
    query_attribute_float, query_attribute_floats, query_attribute_graph, query_attribute_int,
    query_attribute_ints, query_attribute_string, query_attribute_tensor,
};
use crate::tensor_rank::DynRank;
use crate::{TrigOp, onnx};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error(transparent)]
    NDArrayNumericTensorError(#[from] NDArrayNumericTensorError),
    #[error(transparent)]
    NumericTensorError(#[from] NumericTensorError),
    #[error("Unexpected dtype: expected {0}, got {1}")]
    UnexpectedDType(DType, DType),
    #[error("Unimplemented operator: {0}")]
    UnimplementedOperatorError(String),
    #[error(transparent)]
    MilliOpGraphError(#[from] MilliOpGraphError),
    #[error("Invalid input for operation {0}")]
    InvalidInput(String),
    #[error(transparent)]
    DTypeError(#[from] DTypeError),
    #[error("Unexpected shape: expected {0:?}, got {1:?} in shape {2:?}")]
    UnexpectedDimension(u64, u64, Vec<u64>),
    #[error("Unexpected rank: expected {0}, got {1}")]
    UnexpectedRank(usize, usize),
    #[error("Missing input tensor: {0} {1:?} {2:?}")]
    MissingInputTensor(String, Option<DType>, Option<Vec<usize>>),
}

pub trait Operation {
    /*fn get_op_type_name(&self) -> String {
        type_name_of_val(self).to_string()
    }*/
    fn get_op_type_name(&self) -> String;
    fn get_inputs(&self) -> Vec<TensorId>;
    fn get_outputs(&self) -> Vec<TensorId>;

    fn eval(
        &self,
        backend: &mut EvalBackend,
        inputs: &HashMap<TensorId, NumericTensor<DynRank>>,
    ) -> Result<HashMap<TensorId, NumericTensor<DynRank>>, EvalError> {
        let milli_graph = self.get_milli_op_graph();
        Ok(milli_graph.eval(inputs, backend)?)
    }
    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId>;
    fn get_sub_graphs(&self) -> Vec<&SymbolicGraphInner> {
        vec![]
    }
}

#[derive(Clone, Debug, PartialEq, strum_macros::Display, Serialize, Deserialize)]
pub(crate) enum WhichBinaryOperation {
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
    a: TensorId,
    b: TensorId,
    output: TensorId,
    which: WhichBinaryOperation,
}

impl BinaryOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
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

    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.a, self.b]
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
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

#[derive(Clone, Debug, PartialEq, strum_macros::Display, Serialize, Deserialize)]
pub(crate) enum WhichUnaryOperation {
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
    input: TensorId,
    output: TensorId,
    which: WhichUnaryOperation,
}

impl UnaryOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
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

    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input]
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let a = input_map[&self.input];
        let res = match &self.which {
            WhichUnaryOperation::Relu => AnyMilliOp::ClampMin(MilliOpClampMin::new(a, 0.0)),
            WhichUnaryOperation::Sigmoid => {
                let x = graph.push_op(AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::neg(a)));
                let x = graph.push_op(AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::exp(x)));
                let c = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(1.0f32)));
                let c = graph.push_op(AnyMilliOp::CastLike(MilliOpCastLike::new(c, x)));
                let x = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::add(x, c)));
                AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::reciprocal(x))
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
pub struct CumSumOperation {
    input: TensorId,
    output: TensorId,
    axis: TensorId,
    exclusive: bool,
    reverse: bool,
}

impl CumSumOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("CumSum"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("CumSum"));
        }
        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Unary"))?,
            axis: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Unary"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Unary"))?,
            exclusive: false,
            reverse: false,
        })
    }
}
impl Operation for CumSumOperation {
    fn get_op_type_name(&self) -> String {
        "CumSum".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input, self.axis]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let a = input_map[&self.input];
        let b = input_map[&self.axis];

        let out = graph.push_op(AnyMilliOp::CumSum(MilliOpCumSum::new(
            a,
            b,
            self.exclusive,
            self.reverse,
        )));

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LpNormalizationOperation {
    input: TensorId,
    output: TensorId,
    axis: i64,
    p: i64,
}

impl LpNormalizationOperation {
    pub fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("LpNormalization"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("LpNormalization"));
        }
        let mut axis = -1;
        let mut p = 2;
        for attr in attributes {
            match attr.name.as_str() {
                "axis" => axis = attr.i,
                "p" => p = attr.i,
                _ => {}
            }
        }
        match p {
            1 | 2 => {}
            _ => return Err(ONNXDecodingError::InvalidOperatorInputs("LpNormalization")),
        }

        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("LpNormalization"))?,
            output: outputs[0]
                .ok_or(ONNXDecodingError::InvalidOperatorOutputs("LpNormalization"))?,
            axis,
            p,
        })
    }
}

impl Operation for LpNormalizationOperation {
    fn get_op_type_name(&self) -> String {
        "LpNormalization".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let input = input_map[&self.input];

        let x = graph.push_op(AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::abs(input)));

        let x = match self.p {
            1 => x,
            2 => graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::mul(
                input, input,
            ))),
            _ => panic!(),
        };
        let axis_tensor = NDArrayNumericTensor::from(vec![self.axis])
            .try_to_rank::<DynRank>()
            .unwrap();
        let axis = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new(axis_tensor)));
        let x = graph.push_op(AnyMilliOp::ReduceSum(MilliOpReduceSum::new(
            x,
            Some(axis),
            true,
            false,
        )));
        let x = match self.p {
            1 => x,
            2 => graph.push_op(AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::sqrt(x))),
            _ => panic!(),
        };
        let out = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::div(input, x)));

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GroupNormalizationOperation {
    input: TensorId,
    scale: TensorId,
    bias: TensorId,
    output: TensorId,
    epsilon: f32,
    num_groups: usize,
}

impl GroupNormalizationOperation {
    pub fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 3 {
            return Err(ONNXDecodingError::InvalidOperatorInputs(
                "GroupNormalization",
            ));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs(
                "GroupNormalization",
            ));
        }
        let mut epsilon = 1e-5;
        let mut num_groups = None;
        for attr in attributes {
            match attr.name.as_str() {
                "epsilon" => epsilon = attr.f,
                "num_groups" => num_groups = Some(attr.i),
                _ => {}
            }
        }
        let num_groups = num_groups.ok_or(ONNXDecodingError::MissingAttribute(
            "GroupNormalization".to_string(),
            "num_groups".to_string(),
        ))? as usize;
        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                "GroupNormalization",
            ))?,
            scale: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                "GroupNormalization",
            ))?,
            bias: inputs[2].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                "GroupNormalization",
            ))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs(
                "GroupNormalization",
            ))?,
            epsilon,
            num_groups,
        })
    }
}

impl Operation for GroupNormalizationOperation {
    fn get_op_type_name(&self) -> String {
        "GroupNormalization".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input, self.scale, self.bias]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let input = input_map[&self.input];

        let input_shape = graph.push_op(AnyMilliOp::Shape(MilliOpShape::new(input)));
        let num_channels = {
            let starts = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(1i64)));
            let ends = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(2i64)));
            graph.push_op(AnyMilliOp::Slice(MilliOpSlice::new(
                input_shape,
                starts,
                ends,
                None,
                None,
            )))
        };
        let reshaped_input = {
            let new_shape_tensor =
                NDArrayNumericTensor::from(vec![0i64, self.num_groups as i64, -1]);
            let new_shape = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new(
                new_shape_tensor.to_dyn(),
            )));
            graph.push_op(AnyMilliOp::Reshape(MilliOpReshape::new(
                input, new_shape, false,
            )))
        };

        let mean_axis = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(2i64)));
        let mean = graph.push_op(AnyMilliOp::ReduceMean(MilliOpReduceMean::new(
            reshaped_input,
            Some(mean_axis),
            true,
            false,
        )));

        let input = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::sub(
            reshaped_input,
            mean,
        )));

        let variance = {
            let x = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::mul(
                input, input,
            )));
            graph.push_op(AnyMilliOp::ReduceMean(MilliOpReduceMean::new(
                x,
                Some(mean_axis),
                true,
                false,
            )))
        };

        let input_normalized = {
            let epsilon = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(
                self.epsilon,
            )));
            let epsilon = graph.push_op(AnyMilliOp::CastLike(MilliOpCastLike::new(
                epsilon, variance,
            )));
            let var_plus_eps = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::add(
                variance, epsilon,
            )));
            let val = graph.push_op(AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::sqrt(
                var_plus_eps,
            )));
            graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::div(
                input, val,
            )))
        };

        let zero = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(0i64)));
        let neg_one = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(-1i64)));
        let one = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(1i64)));

        let y = {
            let new_shape = graph.push_op(AnyMilliOp::Concat(MilliOpConcat::new(
                vec![zero, num_channels, neg_one],
                0,
            )));
            graph.push_op(AnyMilliOp::Reshape(MilliOpReshape::new(
                input_normalized,
                new_shape,
                false,
            )))
        };

        let y = {
            let scale = graph.push_op(AnyMilliOp::Unsqueeze(MilliOpUnsqueeze::new(
                input_map[&self.scale],
                one,
            )));
            graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::mul(y, scale)))
        };

        let y = {
            let bias = graph.push_op(AnyMilliOp::Unsqueeze(MilliOpUnsqueeze::new(
                input_map[&self.bias],
                one,
            )));
            graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::add(y, bias)))
        };

        let out = graph.push_op(AnyMilliOp::Reshape(MilliOpReshape::new(
            y,
            input_shape,
            false,
        )));

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SqueezeOperation {
    input: TensorId,
    axes: Option<TensorId>,
    axes_attribute: Option<Vec<i64>>,
    output: TensorId,
}

impl SqueezeOperation {
    pub fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.is_empty() || inputs.len() > 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Squeeze"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Squeeze"));
        }
        let axes_attribute = query_attribute_ints(attributes, "axes");
        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Squeeze"))?,
            axes: if inputs.len() > 1 {
                Some(inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Squeeze"))?)
            } else {
                None
            },
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Squeeze"))?,
            axes_attribute,
        })
    }
}

impl Operation for SqueezeOperation {
    fn get_op_type_name(&self) -> String {
        "Squeeze".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        if let Some(axes) = self.axes {
            vec![self.input, axes]
        } else {
            vec![self.input]
        }
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let axes_input = if let Some(axes) = self.axes {
            input_map[&axes]
        } else if let Some(axes) = &self.axes_attribute {
            let axes_tensor = NDArrayNumericTensor::from_vec(axes.clone());
            graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new(
                axes_tensor.to_dyn(),
            )))
        } else {
            panic!();
        };

        let out = graph.push_op(AnyMilliOp::Squeeze(MilliOpSqueeze::new(
            input_map[&self.input],
            axes_input,
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UnsqueezeOperation {
    input: TensorId,
    axes: Option<TensorId>,
    axes_attribute: Option<Vec<i64>>,
    output: TensorId,
}

impl UnsqueezeOperation {
    pub fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.is_empty() || inputs.len() > 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Unsqueeze"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Unsqueeze"));
        }
        let axes_attribute = query_attribute_ints(attributes, "axes");
        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Unsqueeze"))?,
            axes: if inputs.len() > 1 {
                Some(inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Unsqueeze"))?)
            } else {
                None
            },
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Unsqueeze"))?,
            axes_attribute,
        })
    }
}

impl Operation for UnsqueezeOperation {
    fn get_op_type_name(&self) -> String {
        "Unsqueeze".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        if let Some(axes) = self.axes {
            vec![self.input, axes]
        } else {
            vec![self.input]
        }
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let axes_input = if let Some(axes) = self.axes {
            input_map[&axes]
        } else if let Some(axes) = &self.axes_attribute {
            let axes_tensor = NDArrayNumericTensor::from_vec(axes.clone());
            graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new(
                axes_tensor.to_dyn(),
            )))
        } else {
            panic!();
        };

        let out = graph.push_op(AnyMilliOp::Unsqueeze(MilliOpUnsqueeze::new(
            input_map[&self.input],
            axes_input,
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransposeOperation {
    input: TensorId,
    output: TensorId,
    perm: Option<Vec<i64>>,
}

impl TransposeOperation {
    pub fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Transpose"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Transpose"));
        }
        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Transpose"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Transpose"))?,
            perm: query_attribute_ints(attributes, "perm"),
        })
    }
}

impl Operation for TransposeOperation {
    fn get_op_type_name(&self) -> String {
        "Transpose".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let out = graph.push_op(AnyMilliOp::Transpose(MilliOpTranspose::new(
            input_map[&self.input],
            self.perm.clone(),
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReshapeOperation {
    input: TensorId,
    shape: TensorId,
    output: TensorId,
}

impl ReshapeOperation {
    pub fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
        _attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Reshape"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Reshape"));
        }
        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Reshape"))?,
            shape: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Reshape"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Reshape"))?,
        })
    }
}

impl Operation for ReshapeOperation {
    fn get_op_type_name(&self) -> String {
        "Reshape".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input, self.shape]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let out = graph.push_op(AnyMilliOp::Reshape(MilliOpReshape::new(
            input_map[&self.input],
            input_map[&self.shape],
            false,
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CastLikeOperation {
    input: TensorId,
    target_type: TensorId,
    output: TensorId,
}

impl CastLikeOperation {
    pub fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
        _attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("CastLike"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("CastLike"));
        }
        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("CastLike"))?,
            target_type: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("CastLike"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("CastLike"))?,
        })
    }
}

impl Operation for CastLikeOperation {
    fn get_op_type_name(&self) -> String {
        "Cast Like".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input, self.target_type]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let out = graph.push_op(AnyMilliOp::CastLike(MilliOpCastLike::new(
            input_map[&self.input],
            input_map[&self.target_type],
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CastOperation {
    input: TensorId,
    output: TensorId,
    to: DType,
}

impl CastOperation {
    pub fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Cast"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Cast"));
        }
        let to_i = attributes
            .iter()
            .find(|a| a.name == "to")
            .ok_or(ONNXDecodingError::MissingAttribute(
                "Cast".to_string(),
                "to".to_string(),
            ))?
            .i as i32;
        let to_datatype = onnx::tensor_proto::DataType::try_from(to_i)
            .map_err(|x| ONNXDecodingError::ProtobufDecodeError(x.into()))?;
        let to = DType::try_from(to_datatype)?;
        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Cast"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Cast"))?,
            to,
        })
    }
}

impl Operation for CastOperation {
    fn get_op_type_name(&self) -> String {
        "Cast".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let out = graph.push_op(AnyMilliOp::Cast(MilliOpCast::new(
            input_map[&self.input],
            self.to,
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerNormalizationOperation {
    input: TensorId,
    scale: TensorId,
    bias: Option<TensorId>,
    output: TensorId,
    mean_output: Option<TensorId>,
    inv_std_dev_output: Option<TensorId>,
    axis: i64,
    epsilon: f32,
}

impl LayerNormalizationOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() < 2 || inputs.len() > 3 {
            return Err(ONNXDecodingError::InvalidOperatorInputs(
                "LayerNormalization",
            ));
        }
        if outputs.is_empty() || outputs.len() > 3 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs(
                "LayerNormalization",
            ));
        }
        let mut axis = -1;
        let mut epsilon = 1e-5;
        for attribute in attributes {
            match attribute.name.as_str() {
                "axis" => {
                    axis = attribute.i;
                }
                "epsilon" => {
                    epsilon = attribute.f;
                }
                _ => {}
            }
        }
        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                "LayerNormalization",
            ))?,
            scale: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                "LayerNormalization",
            ))?,
            bias: if inputs.len() == 3 {
                Some(inputs[2].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                    "LayerNormalization",
                ))?)
            } else {
                None
            },
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                "LayerNormalization",
            ))?,
            mean_output: if outputs.len() > 1 {
                Some(outputs[1].ok_or(ONNXDecodingError::InvalidOperatorOutputs(
                    "LayerNormalization",
                ))?)
            } else {
                None
            },
            inv_std_dev_output: if outputs.len() > 2 {
                Some(outputs[2].ok_or(ONNXDecodingError::InvalidOperatorOutputs(
                    "LayerNormalization",
                ))?)
            } else {
                None
            },
            axis,
            epsilon,
        })
    }
}

impl Operation for LayerNormalizationOperation {
    fn get_op_type_name(&self) -> String {
        "Layer Normalization".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        let mut v = vec![self.input, self.scale];
        if let Some(bias) = self.bias {
            v.push(bias);
        }
        v
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        let mut res = vec![self.output];
        if let Some(mean_output) = self.mean_output {
            res.push(mean_output);
        }
        if let Some(inv_std_dev_output) = self.inv_std_dev_output {
            res.push(inv_std_dev_output);
        }
        res
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let input_data = input_map[&self.input];
        let input_scale = input_map[&self.scale];

        let axis = ops_helpers::scalar_const(&mut graph, self.axis);
        let axis = ops_helpers::resolve_axes(&mut graph, axis, input_data);

        let normalized_axes = {
            let r = MilliOpRange::new(
                axis,
                ops_helpers::rank(&mut graph, input_data),
                ops_helpers::scalar_const(&mut graph, 1i64),
            );
            graph.push_op(AnyMilliOp::Range(r))
        };

        let mean = graph.push_op(AnyMilliOp::ReduceMean(MilliOpReduceMean::new(
            input_data,
            Some(normalized_axes),
            true,
            false,
        )));

        let d = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::sub(
            input_data, mean,
        )));
        let dd = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::mul(d, d)));
        let variance = graph.push_op(AnyMilliOp::ReduceMean(MilliOpReduceMean::new(
            dd,
            Some(normalized_axes),
            true,
            false,
        )));
        let epsilon = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(
            self.epsilon,
        )));
        let epsilon = graph.push_op(AnyMilliOp::CastLike(MilliOpCastLike::new(
            epsilon, variance,
        )));
        let var_plus_eps = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::add(
            variance, epsilon,
        )));
        let stddev = graph.push_op(AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::sqrt(
            var_plus_eps,
        )));
        let inv_stddev = graph.push_op(AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::reciprocal(
            stddev,
        )));
        let normalized = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::mul(
            d, inv_stddev,
        )));

        let out = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::mul(
            normalized,
            input_scale,
        )));

        let out = if let Some(bias) = self.bias {
            graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::add(
                out,
                input_map[&bias],
            )))
        } else {
            out
        };

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        if let Some(x) = self.mean_output {
            output_map.insert(mean, x);
        }
        if let Some(x) = self.inv_std_dev_output {
            output_map.insert(inv_stddev, x);
        }
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GatherOperation {
    input: TensorId,
    indices: TensorId,
    output: TensorId,
    axis: i64,
}

impl GatherOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Gather"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Gather"));
        }
        let mut axis = 0;
        for attribute in attributes {
            if attribute.name.as_str() == "axis" {
                axis = attribute.i;
            }
        }
        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Gather"))?,
            indices: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Gather"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Gather"))?,
            axis,
        })
    }
}

impl Operation for GatherOperation {
    fn get_op_type_name(&self) -> String {
        "Gather".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input, self.indices]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let out = graph.push_op(AnyMilliOp::Gather(MilliOpGather::new(
            input_map[&self.input],
            input_map[&self.indices],
            self.axis,
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShapeOperation {
    start: Option<i64>,
    end: Option<i64>,
    input: TensorId,
    output: TensorId,
}

impl ShapeOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Shape"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Shape"));
        }
        let mut end = None;
        let mut start = None;
        for attribute in attributes {
            match attribute.name.as_str() {
                "start" => {
                    start = Some(attribute.i);
                }
                "end" => {
                    end = Some(attribute.i);
                }
                _ => {}
            }
        }
        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Shape"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Shape"))?,
            start,
            end,
        })
    }
}

impl Operation for ShapeOperation {
    fn get_op_type_name(&self) -> String {
        "Shape".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let out = graph.push_op(AnyMilliOp::Shape(MilliOpShape::new(input_map[&self.input])));
        let out = if self.start.is_some() || self.end.is_some() {
            let start = ops_helpers::scalar_const(&mut graph, self.start.unwrap_or(0));
            let end = if let Some(end) = self.end {
                ops_helpers::scalar_const(&mut graph, end)
            } else {
                graph.push_op(AnyMilliOp::Shape(MilliOpShape::new(out)))
            };
            graph.push_op(AnyMilliOp::Slice(MilliOpSlice::new(
                out, start, end, None, None,
            )))
        } else {
            out
        };
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConcatOperation {
    axis: i64,
    inputs: Vec<TensorId>,
    output: TensorId,
}

impl ConcatOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Concat"));
        }
        let mut axis = 0;
        for attribute in attributes {
            if attribute.name == "axis" {
                axis = attribute.i;
            }
        }

        Ok(Self {
            axis,
            inputs: inputs
                .iter()
                .map(|x| x.ok_or(ONNXDecodingError::InvalidOperatorInputs("Concat")))
                .collect::<Result<_, _>>()?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Concat"))?,
        })
    }
}

impl Operation for ConcatOperation {
    fn get_op_type_name(&self) -> String {
        "Concat".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        self.inputs.clone()
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let mut milli_inputs = vec![];
        for input in &self.inputs {
            milli_inputs.push(input_map[input]);
        }
        let out = graph.push_op(AnyMilliOp::Concat(MilliOpConcat::new(
            milli_inputs,
            self.axis,
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConstantOfShapeOperation {
    value: NumericScalar,
    input: TensorId,
    output: TensorId,
}

impl ConstantOfShapeOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("ConstantOfShape"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("ConstantOfShape"));
        }

        let value = query_attribute_tensor(attributes, "value")
            .map(|x| x.first_element())
            .unwrap_or(NumericScalar::F32(0.0));

        Ok(Self {
            value,
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("ConstantOfShape"))?,
            output: outputs[0]
                .ok_or(ONNXDecodingError::InvalidOperatorOutputs("ConstantOfShape"))?,
        })
    }
}

impl Operation for ConstantOfShapeOperation {
    fn get_op_type_name(&self) -> String {
        "Constant of Shape".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let out = graph.push_op(AnyMilliOp::ConstantOfShape(MilliOpConstantOfShape::new(
            self.value.clone(),
            input_map[&self.input],
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReduceMeanOperation {
    keepdims: Option<bool>,
    noop_with_empty_axes: Option<bool>,
    input_data: TensorId,
    input_axes: Option<TensorId>,
    axes_attr: Option<Vec<i64>>,
    output: TensorId,
}

impl ReduceMeanOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.is_empty() || inputs.len() > 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("ReduceMean"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("ReduceMean"));
        }

        let axes_attr = query_attribute_ints(attributes, "axes");
        let keepdims = query_attribute_int(attributes, "keepdims").map(|x| x != 0);
        let noop_with_empty_axes =
            query_attribute_int(attributes, "noop_with_empty_axes").map(|x| x != 0);

        Ok(Self {
            keepdims,
            noop_with_empty_axes,
            input_data: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("ReduceMean"))?,
            input_axes: if inputs.len() > 1 {
                Some(inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("ReduceMean"))?)
            } else {
                None
            },
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("ReduceMean"))?,
            axes_attr,
        })
    }
}

impl Operation for ReduceMeanOperation {
    fn get_op_type_name(&self) -> String {
        "Reduce Mean".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        if let Some(input_axes) = self.input_axes {
            vec![self.input_data, input_axes]
        } else {
            vec![self.input_data]
        }
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let axes = if let Some(input_axes) = &self.input_axes {
            Some(input_map[input_axes])
        } else if let Some(axes) = &self.axes_attr {
            let tensor = NDArrayNumericTensor::from(axes.clone());
            Some(graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new(tensor.to_dyn()))))
        } else {
            None
        };
        let out = graph.push_op(AnyMilliOp::ReduceMean(MilliOpReduceMean::new(
            input_map[&self.input_data],
            axes,
            self.keepdims.unwrap_or(true),
            self.noop_with_empty_axes.unwrap_or(false),
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReduceSumOperation {
    keepdims: Option<bool>,
    noop_with_empty_axes: Option<bool>,
    input_data: TensorId,
    input_axes: Option<TensorId>,
    axes_attr: Option<Vec<i64>>,
    output: TensorId,
}

impl ReduceSumOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.is_empty() || inputs.len() > 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("ReduceSum"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("ReduceSum"));
        }

        let axes_attr = query_attribute_ints(attributes, "attr");
        let keepdims = query_attribute_int(attributes, "keepdims").map(|x| x != 0);
        let noop_with_empty_axes =
            query_attribute_int(attributes, "noop_with_empty_axes").map(|x| x != 0);

        Ok(Self {
            keepdims,
            noop_with_empty_axes,
            input_data: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("ReduceSum"))?,
            input_axes: if inputs.len() > 1 {
                Some(inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("ReduceSum"))?)
            } else {
                None
            },
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("ReduceSum"))?,
            axes_attr,
        })
    }
}

impl Operation for ReduceSumOperation {
    fn get_op_type_name(&self) -> String {
        "Reduce Sum".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        if let Some(input_axes) = self.input_axes {
            vec![self.input_data, input_axes]
        } else {
            vec![self.input_data]
        }
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let axes = if let Some(input_axes) = &self.input_axes {
            Some(input_map[input_axes])
        } else if let Some(axes) = &self.axes_attr {
            let tensor = NDArrayNumericTensor::from(axes.clone());
            Some(graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new(tensor.to_dyn()))))
        } else {
            None
        };
        let out = graph.push_op(AnyMilliOp::ReduceSum(MilliOpReduceSum::new(
            input_map[&self.input_data],
            axes,
            self.keepdims.unwrap_or(true),
            self.noop_with_empty_axes.unwrap_or(false),
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReduceMaxOperation {
    keepdims: Option<bool>,
    noop_with_empty_axes: Option<bool>,
    input_data: TensorId,
    input_axes: Option<TensorId>,
    axes_attr: Option<Vec<i64>>,
    output: TensorId,
}

impl ReduceMaxOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.is_empty() || inputs.len() > 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("ReduceMax"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("ReduceMax"));
        }

        let axes_attr = query_attribute_ints(attributes, "attr");
        let keepdims = query_attribute_int(attributes, "keepdims").map(|x| x != 0);
        let noop_with_empty_axes =
            query_attribute_int(attributes, "noop_with_empty_axes").map(|x| x != 0);

        Ok(Self {
            keepdims,
            noop_with_empty_axes,
            input_data: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("ReduceMax"))?,
            input_axes: if inputs.len() > 1 {
                Some(inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("ReduceMax"))?)
            } else {
                None
            },
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("ReduceMax"))?,
            axes_attr,
        })
    }
}

impl Operation for ReduceMaxOperation {
    fn get_op_type_name(&self) -> String {
        "Reduce Max".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        if let Some(input_axes) = self.input_axes {
            vec![self.input_data, input_axes]
        } else {
            vec![self.input_data]
        }
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let axes = if let Some(input_axes) = &self.input_axes {
            Some(input_map[input_axes])
        } else if let Some(axes) = &self.axes_attr {
            let tensor = NDArrayNumericTensor::from(axes.clone());
            Some(graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new(tensor.to_dyn()))))
        } else {
            None
        };
        let out = graph.push_op(AnyMilliOp::ReduceMax(MilliOpReduceMax::new(
            input_map[&self.input_data],
            axes,
            self.keepdims.unwrap_or(true),
            self.noop_with_empty_axes.unwrap_or(false),
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReduceMinOperation {
    keepdims: Option<bool>,
    noop_with_empty_axes: Option<bool>,
    input_data: TensorId,
    input_axes: Option<TensorId>,
    axes_attr: Option<Vec<i64>>,
    output: TensorId,
}

impl ReduceMinOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.is_empty() || inputs.len() > 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("ReduceMin"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("ReduceMin"));
        }

        let axes_attr = query_attribute_ints(attributes, "attr");
        let keepdims = query_attribute_int(attributes, "keepdims").map(|x| x != 0);
        let noop_with_empty_axes =
            query_attribute_int(attributes, "noop_with_empty_axes").map(|x| x != 0);

        Ok(Self {
            keepdims,
            noop_with_empty_axes,
            input_data: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("ReduceMin"))?,
            input_axes: if inputs.len() > 1 {
                Some(inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("ReduceMin"))?)
            } else {
                None
            },
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("ReduceMin"))?,
            axes_attr,
        })
    }
}

impl Operation for ReduceMinOperation {
    fn get_op_type_name(&self) -> String {
        "Reduce Min".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        if let Some(input_axes) = self.input_axes {
            vec![self.input_data, input_axes]
        } else {
            vec![self.input_data]
        }
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let axes = if let Some(input_axes) = &self.input_axes {
            Some(input_map[input_axes])
        } else if let Some(axes) = &self.axes_attr {
            let tensor = NDArrayNumericTensor::from(axes.clone());
            Some(graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new(tensor.to_dyn()))))
        } else {
            None
        };
        let out = graph.push_op(AnyMilliOp::ReduceMin(MilliOpReduceMin::new(
            input_map[&self.input_data],
            axes,
            self.keepdims.unwrap_or(true),
            self.noop_with_empty_axes.unwrap_or(false),
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReduceProdOperation {
    keepdims: Option<bool>,
    noop_with_empty_axes: Option<bool>,
    input_data: TensorId,
    input_axes: Option<TensorId>,
    axes_attr: Option<Vec<i64>>,
    output: TensorId,
}

impl ReduceProdOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.is_empty() || inputs.len() > 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("ReduceProd"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("ReduceProd"));
        }

        let axes_attr = query_attribute_ints(attributes, "attr");
        let keepdims = query_attribute_int(attributes, "keepdims").map(|x| x != 0);
        let noop_with_empty_axes =
            query_attribute_int(attributes, "noop_with_empty_axes").map(|x| x != 0);

        Ok(Self {
            keepdims,
            noop_with_empty_axes,
            input_data: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("ReduceProd"))?,
            input_axes: if inputs.len() > 1 {
                Some(inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("ReduceProd"))?)
            } else {
                None
            },
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("ReduceProd"))?,
            axes_attr,
        })
    }
}

impl Operation for ReduceProdOperation {
    fn get_op_type_name(&self) -> String {
        "Reduce Prod".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        if let Some(input_axes) = self.input_axes {
            vec![self.input_data, input_axes]
        } else {
            vec![self.input_data]
        }
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let axes = if let Some(input_axes) = &self.input_axes {
            Some(input_map[input_axes])
        } else if let Some(axes) = &self.axes_attr {
            let tensor = NDArrayNumericTensor::from(axes.clone());
            Some(graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new(tensor.to_dyn()))))
        } else {
            None
        };
        let out = graph.push_op(AnyMilliOp::ReduceProd(MilliOpReduceProd::new(
            input_map[&self.input_data],
            axes,
            self.keepdims.unwrap_or(true),
            self.noop_with_empty_axes.unwrap_or(false),
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PowOperation {
    input_x: TensorId,
    input_y: TensorId,
    output: TensorId,
}

impl PowOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
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

    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input_x, self.input_y]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
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
    input_a: TensorId,
    input_b: TensorId,
    input_c: Option<TensorId>,
    output: TensorId,
}

impl GemmOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
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

    fn get_inputs(&self) -> Vec<TensorId> {
        if let Some(input_c) = self.input_c {
            vec![self.input_a, self.input_b, input_c]
        } else {
            vec![self.input_a, self.input_b]
        }
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
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
pub struct SplitOperation {
    axis: Option<i64>,
    num_outputs: Option<i64>,
    input: TensorId,
    split: Option<TensorId>,
    split_attribute: Option<Vec<i64>>,
    outputs: Vec<TensorId>,
}

impl SplitOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.is_empty() || inputs.len() > 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Split"));
        }

        let axis = query_attribute_int(attributes, "axis");
        let num_outputs = query_attribute_int(attributes, "num_outputs");
        let split_attribute = query_attribute_ints(attributes, "split");

        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Split"))?,
            split: if inputs.len() > 1 {
                Some(inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Split"))?)
            } else {
                None
            },
            outputs: outputs
                .iter()
                .map(|x| x.ok_or(ONNXDecodingError::InvalidOperatorOutputs("Split")))
                .collect::<Result<_, _>>()?,
            split_attribute,
            axis,
            num_outputs,
        })
    }
}

impl Operation for SplitOperation {
    fn get_op_type_name(&self) -> String {
        "Split".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        if let Some(split) = self.split {
            vec![self.input, split]
        } else {
            vec![self.input]
        }
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        self.outputs.clone()
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());

        let mut output_map = HashMap::new();

        let split = if let Some(split) = self.split {
            Some(MilliOpTensorIDOrLiteral::TensorID(input_map[&split]))
        } else {
            self.split_attribute.as_ref().map(|split| {
                MilliOpTensorIDOrLiteral::Literal(
                    NDArrayNumericTensor::from_vec(split.clone()).to_dyn(),
                )
            })
        };

        for (output_id, output_tensor_id) in self.outputs.iter().enumerate() {
            let out = graph.push_op(AnyMilliOp::Split(MilliOpSplit::new(
                input_map[&self.input],
                split.clone(),
                self.axis.unwrap_or_default(),
                self.num_outputs.map(|x| x as usize),
                output_id,
            )));

            output_map.insert(out, *output_tensor_id);
        }
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SliceOperation {
    data: TensorId,
    starts: TensorId,
    ends: TensorId,
    axes: Option<TensorId>,
    steps: Option<TensorId>,
    output: TensorId,
}

impl SliceOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
        _attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() < 3 || inputs.len() > 5 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Slice"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Slice"));
        }

        Ok(Self {
            data: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Slice"))?,
            starts: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Slice"))?,
            ends: inputs[2].ok_or(ONNXDecodingError::InvalidOperatorInputs("Slice"))?,
            axes: if inputs.len() > 3 {
                Some(inputs[3].ok_or(ONNXDecodingError::InvalidOperatorInputs("Slice"))?)
            } else {
                None
            },
            steps: if inputs.len() > 4 {
                Some(inputs[4].ok_or(ONNXDecodingError::InvalidOperatorInputs("Slice"))?)
            } else {
                None
            },
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Slice"))?,
        })
    }
}

impl Operation for SliceOperation {
    fn get_op_type_name(&self) -> String {
        "Slice".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        if let Some(axes) = self.axes {
            if let Some(steps) = self.steps {
                vec![self.data, self.starts, self.ends, axes, steps]
            } else {
                vec![self.data, self.starts, self.ends, axes]
            }
        } else {
            vec![self.data, self.starts, self.ends]
        }
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let out = graph.push_op(AnyMilliOp::Slice(MilliOpSlice::new(
            input_map[&self.data],
            input_map[&self.starts],
            input_map[&self.ends],
            self.steps.map(|x| input_map[&x]),
            self.axes.map(|x| input_map[&x]),
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WhereOperation {
    condition: TensorId,
    x: TensorId,
    y: TensorId,
    output: TensorId,
}

impl WhereOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
        _attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 3 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Where"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Where"));
        }

        Ok(Self {
            condition: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Where"))?,
            x: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Where"))?,
            y: inputs[2].ok_or(ONNXDecodingError::InvalidOperatorInputs("Where"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Where"))?,
        })
    }
}

impl Operation for WhereOperation {
    fn get_op_type_name(&self) -> String {
        "Where".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.condition, self.x, self.y]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let out = graph.push_op(AnyMilliOp::Where(MilliOpWhere::new(
            input_map[&self.condition],
            input_map[&self.x],
            input_map[&self.y],
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SoftmaxOperation {
    axis: Option<i64>,
    input: TensorId,
    output: TensorId,
}

impl SoftmaxOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
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

    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
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
    input: TensorId,
    output: TensorId,
}

impl LogSoftmaxOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
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

    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
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
pub struct SizeOperation {
    input: TensorId,
    output: TensorId,
}

impl SizeOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Size"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Size"));
        }
        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Size"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Size"))?,
        })
    }
}

impl Operation for SizeOperation {
    fn get_op_type_name(&self) -> String {
        "Size".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());

        let shape = graph.push_op(AnyMilliOp::Shape(MilliOpShape::new(input_map[&self.input])));
        let size = graph.push_op(AnyMilliOp::ReduceProd(MilliOpReduceProd::new(
            shape, None, false, false,
        )));

        let mut output_map = HashMap::new();
        output_map.insert(size, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RangeOperation {
    start: TensorId,
    end: TensorId,
    delta: TensorId,
    output: TensorId,
}

impl RangeOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 3 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Range"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Range"));
        }
        Ok(Self {
            start: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Range"))?,
            end: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Range"))?,
            delta: inputs[2].ok_or(ONNXDecodingError::InvalidOperatorInputs("Range"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Range"))?,
        })
    }
}

impl Operation for RangeOperation {
    fn get_op_type_name(&self) -> String {
        "Range".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.start, self.end, self.delta]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());

        let out = graph.push_op(AnyMilliOp::Range(MilliOpRange::new(
            input_map[&self.start],
            input_map[&self.end],
            input_map[&self.delta],
        )));

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FlattenOperation {
    input: TensorId,
    output: TensorId,
    axis: i64,
}

impl FlattenOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Flatten"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Flatten"));
        }
        let axis = query_attribute_int(attributes, "axis").unwrap_or(1);
        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Flatten"))?,
            axis,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Flatten"))?,
        })
    }
}

impl Operation for FlattenOperation {
    fn get_op_type_name(&self) -> String {
        "Flatten".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input]
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let input = input_map[&self.input];

        let shape_tensor = if self.axis == 0 {
            let output_shape = vec![1i64, -1i64];
            let shape_tensor = NDArrayNumericTensor::from(output_shape);
            graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new(
                shape_tensor.to_dyn(),
            )))
        } else {
            let input_shape = graph.push_op(AnyMilliOp::Shape(MilliOpShape::new(input)));
            let zero_const = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new(
                NDArrayNumericTensor::from_vec_shape(vec![0i64], &vec![1]).unwrap(),
            )));
            let axis_const = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new(
                NDArrayNumericTensor::from_vec_shape(vec![self.axis], &vec![1]).unwrap(),
            )));
            let first_dims = graph.push_op(AnyMilliOp::Slice(MilliOpSlice::new(
                input_shape,
                zero_const,
                axis_const,
                None,
                None,
            )));
            let prod = graph.push_op(AnyMilliOp::ReduceProd(MilliOpReduceProd::new(
                first_dims, None, true, false,
            )));
            let neg_one_const = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new(
                NDArrayNumericTensor::from_vec_shape(vec![-1i64], &vec![1]).unwrap(),
            )));
            graph.push_op(AnyMilliOp::Concat(MilliOpConcat::new(
                vec![prod, neg_one_const],
                0,
            )))
        };

        let out = graph.push_op(AnyMilliOp::Reshape(MilliOpReshape::new(
            input,
            shape_tensor,
            false,
        )));

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConstantOperation {
    pub value: NDArrayNumericTensor<DynRank>,
    output: TensorId,
}

impl ConstantOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if !inputs.is_empty() {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Constant"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Constant"));
        }

        let value = if let Some(tensor) = query_attribute_tensor(attributes, "value") {
            tensor
        } else if let Some(value_float) = query_attribute_float(attributes, "value_float") {
            NDArrayNumericTensor::from(vec![value_float]).try_to_rank()?
        } else if let Some(value_floats) = query_attribute_floats(attributes, "value_floats") {
            NDArrayNumericTensor::from(value_floats).try_to_rank()?
        } else if let Some(value_int) = query_attribute_int(attributes, "value_int") {
            NDArrayNumericTensor::from(vec![value_int]).try_to_rank()?
        } else if let Some(value_ints) = query_attribute_ints(attributes, "value_ints") {
            NDArrayNumericTensor::from(value_ints).try_to_rank()?
        } else {
            Err(ONNXDecodingError::MissingAttribute(
                "Constant".to_string(),
                "value".to_string(),
            ))?
        };

        Ok(Self {
            value,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Constant"))?,
        })
    }
}

impl Operation for ConstantOperation {
    fn get_op_type_name(&self) -> String {
        "Constant".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        vec![]
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        let (mut graph, _input_map) = MilliOpGraph::new(&self.get_inputs());

        let out = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new(
            self.value.clone(),
        )));

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IdentityOperation {
    input: TensorId,
    output: TensorId,
}

impl IdentityOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
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

    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input]
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let input = input_map[&self.input];
        let mut output_map = HashMap::new();
        output_map.insert(input, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IsInfOperation {
    input: TensorId,
    output: TensorId,
    detect_negative: Option<bool>,
    detect_positive: Option<bool>,
}

impl IsInfOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
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

    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input]
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
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
pub struct ModuloOperation {
    a: TensorId,
    b: TensorId,
    output: TensorId,
    fmod: Option<bool>,
}

impl ModuloOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
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

    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.a, self.b]
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClipOperation {
    input: TensorId,
    min: Option<TensorId>,
    max: Option<TensorId>,
    output: TensorId,
}

impl ClipOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
        _attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.is_empty() || inputs.len() > 3 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Clip"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Clip"));
        }

        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Clip"))?,
            min: if inputs.len() > 1 {
                Some(inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Clip"))?)
            } else {
                None
            },
            max: if inputs.len() > 2 {
                Some(inputs[2].ok_or(ONNXDecodingError::InvalidOperatorInputs("Clip"))?)
            } else {
                None
            },
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Clip"))?,
        })
    }
}

impl Operation for ClipOperation {
    fn get_op_type_name(&self) -> String {
        "Clip".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        let mut o = vec![self.input];
        if let Some(min) = self.min {
            o.push(min);
        }
        if let Some(max) = self.max {
            o.push(max);
        }
        o
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let x = input_map[&self.input];
        let x = if let Some(min) = self.min {
            let min = input_map[&min];
            graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::max(x, min)))
        } else {
            x
        };
        let x = if let Some(max) = self.max {
            let max = input_map[&max];
            graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::min(x, max)))
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
pub struct ExpandOperation {
    input: TensorId,
    shape: TensorId,
    output: TensorId,
}

impl ExpandOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
        _attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Expand"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Expand"));
        }

        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Expand"))?,
            shape: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Expand"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Expand"))?,
        })
    }
}

impl Operation for ExpandOperation {
    fn get_op_type_name(&self) -> String {
        "Expand".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input, self.shape]
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());

        let x = graph.push_op(AnyMilliOp::Expand(MilliOpExpand::new(
            input_map[&self.input],
            input_map[&self.shape],
        )));

        let mut output_map = HashMap::new();
        output_map.insert(x, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) enum ConvOperationAutoPad {
    NotSet,
    SameUpper,
    SameLower,
    Valid,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConvOperation {
    input: TensorId,
    output: TensorId,
    weight: TensorId,
    bias: Option<TensorId>,
    auto_pad: ConvOperationAutoPad,
    dilations: Vec<i64>,
    group: i64,
    kernel_shape: Vec<i64>,
    pads: Vec<i64>,
    strides: Vec<i64>,
}

impl ConvOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() < 2 || inputs.len() > 3 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Conv"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Conv"));
        }

        let auto_pad_str = query_attribute_string(attributes, "auto_pad");
        let auto_pad = match auto_pad_str {
            Some(x) => match x.to_lowercase().as_str() {
                "notset" => ConvOperationAutoPad::NotSet,
                "same_upper" => ConvOperationAutoPad::SameUpper,
                "same_lower" => ConvOperationAutoPad::SameLower,
                "valid" => ConvOperationAutoPad::Valid,
                _ => ConvOperationAutoPad::NotSet,
            },
            _ => ConvOperationAutoPad::NotSet,
        };

        let dilations = query_attribute_ints(attributes, "dilations").unwrap_or_default();
        let group = query_attribute_int(attributes, "group").unwrap_or(1);
        let kernel_shape = query_attribute_ints(attributes, "kernel_shape").unwrap_or_default();
        let pads = query_attribute_ints(attributes, "pads").unwrap_or_default();
        let strides = query_attribute_ints(attributes, "strides").unwrap_or_default();

        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Conv"))?,
            weight: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Conv"))?,
            bias: if inputs.len() > 2 {
                Some(inputs[2].ok_or(ONNXDecodingError::InvalidOperatorInputs("Conv"))?)
            } else {
                None
            },
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Conv"))?,
            auto_pad,
            dilations,
            group,
            kernel_shape,
            pads,
            strides,
        })
    }
}

impl Operation for ConvOperation {
    fn get_op_type_name(&self) -> String {
        "Conv".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        if let Some(bias) = self.bias {
            vec![self.input, self.weight, bias]
        } else {
            vec![self.input, self.weight]
        }
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        unimplemented!();
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InstanceNormalizationOperation {
    input: TensorId,
    scale: TensorId,
    bias: TensorId,
    output: TensorId,
    epsilon: Option<f32>,
}

impl InstanceNormalizationOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 3 {
            return Err(ONNXDecodingError::InvalidOperatorInputs(
                "InstanceNormalization",
            ));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs(
                "InstanceNormalization",
            ));
        }
        let epsilon = query_attribute_float(attributes, "epsilon");

        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                "InstanceNormalization",
            ))?,
            scale: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                "InstanceNormalization",
            ))?,
            bias: inputs[2].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                "InstanceNormalization",
            ))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs(
                "InstanceNormalization",
            ))?,
            epsilon,
        })
    }
}

impl Operation for InstanceNormalizationOperation {
    fn get_op_type_name(&self) -> String {
        "Instance Normalization".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input, self.scale, self.bias]
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        unimplemented!();
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) enum ResizeCoordinateTransformationMode {
    HalfPixel,
    HalfPixelSymmetric,
    PytorchHalfPixel,
    AlignCorners,
    Asymmetric,
    TFCropAndResize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) enum ResizeKeepAspectRatioPolicy {
    Stretch,
    NotLarger,
    NotSmaller,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) enum ResizeMode {
    Nearest,
    Linear,
    Cubic,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) enum ResizeNearestMode {
    RoundPreferFloor,
    Ceil,
    Floor,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResizeOperation {
    input: TensorId,
    roi: Option<TensorId>,
    scales: Option<TensorId>,
    sizes: Option<TensorId>,
    output: TensorId,
    antialias: bool,
    axes: Vec<i64>,
    coordinate_transformation_mode: ResizeCoordinateTransformationMode,
    cubic_coeff_a: f32,
    exclude_outside: bool,
    extrapolation_value: f32,
    keep_aspect_ratio_policy: ResizeKeepAspectRatioPolicy,
    mode: ResizeMode,
    nearest_mode: ResizeNearestMode,
}

impl ResizeOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.is_empty() || inputs.len() > 4 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Resize"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Resize"));
        }

        let antialias = query_attribute_bool(attributes, "antialias").unwrap_or_default();
        let axes = query_attribute_ints(attributes, "axes").unwrap_or_default();
        let coordinate_transformation_mode =
            query_attribute_string(attributes, "coordinate_transformation_mode")
                .unwrap_or("half_pixel".to_string());
        let coordinate_transformation_mode = match coordinate_transformation_mode.as_str() {
            "half_pixel_symmetric" => ResizeCoordinateTransformationMode::HalfPixelSymmetric,
            "align_corners" => ResizeCoordinateTransformationMode::AlignCorners,
            "asymmetric" => ResizeCoordinateTransformationMode::Asymmetric,
            "pytorch_half_pixel" => ResizeCoordinateTransformationMode::PytorchHalfPixel,
            "tf_crop_and_resize" => ResizeCoordinateTransformationMode::TFCropAndResize,
            "half_pixel" => ResizeCoordinateTransformationMode::HalfPixel,
            _ => ResizeCoordinateTransformationMode::HalfPixel,
        };
        let cubic_coeff_a = query_attribute_float(attributes, "cubic_coeff_a").unwrap_or(-0.75);
        let exclude_outside = query_attribute_bool(attributes, "exclude_outside").unwrap_or(false);
        let extrapolation_value =
            query_attribute_float(attributes, "extrapolation_value").unwrap_or(0.0);
        let keep_aspect_ratio_policy =
            query_attribute_string(attributes, "keep_aspect_ratio_policy")
                .unwrap_or("stretch".to_string());
        let keep_aspect_ratio_policy = match keep_aspect_ratio_policy.as_str() {
            "stretch" => ResizeKeepAspectRatioPolicy::Stretch,
            "not_larger" => ResizeKeepAspectRatioPolicy::NotLarger,
            "not_smaller" => ResizeKeepAspectRatioPolicy::NotSmaller,
            _ => ResizeKeepAspectRatioPolicy::Stretch,
        };
        let mode = query_attribute_string(attributes, "mode").unwrap_or("nearest".to_string());
        let mode = match mode.as_str() {
            "nearest" => ResizeMode::Nearest,
            "linear" => ResizeMode::Linear,
            "cubic" => ResizeMode::Cubic,
            _ => ResizeMode::Nearest,
        };
        let nearest_mode = query_attribute_string(attributes, "nearest_mode")
            .unwrap_or("round_prefer_floor".to_string());
        let nearest_mode = match nearest_mode.as_str() {
            "round_prefer_floor" => ResizeNearestMode::RoundPreferFloor,
            "floor" => ResizeNearestMode::Floor,
            "ceil" => ResizeNearestMode::Ceil,
            _ => ResizeNearestMode::RoundPreferFloor,
        };

        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Resize"))?,
            roi: if inputs.len() > 1 {
                Some(inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Resize"))?)
            } else {
                None
            },
            scales: if inputs.len() > 2 {
                Some(inputs[2].ok_or(ONNXDecodingError::InvalidOperatorInputs("Resize"))?)
            } else {
                None
            },
            sizes: if inputs.len() > 3 {
                Some(inputs[3].ok_or(ONNXDecodingError::InvalidOperatorInputs("Resize"))?)
            } else {
                None
            },
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Resize"))?,
            antialias,
            axes,
            coordinate_transformation_mode,
            cubic_coeff_a,
            exclude_outside,
            extrapolation_value,
            keep_aspect_ratio_policy,
            mode,
            nearest_mode,
        })
    }
}

impl Operation for ResizeOperation {
    fn get_op_type_name(&self) -> String {
        "Resize".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        let mut ret = vec![self.input];
        if let Some(roi) = &self.roi {
            ret.push(*roi);
        }
        if let Some(scales) = &self.scales {
            ret.push(*scales);
        }
        if let Some(sizes) = &self.sizes {
            ret.push(*sizes)
        }
        ret
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) enum PadMode {
    Constant,
    Reflect,
    Edge,
    Wrap,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PadOperation {
    input: TensorId,
    pads: TensorId,
    constant_value: Option<TensorId>,
    axes: Option<TensorId>,
    mode: PadMode,
    output: TensorId,
}

impl PadOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() < 2 || inputs.len() > 4 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Pad"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Pad"));
        }

        let pad_mode = query_attribute_string(attributes, "mode").unwrap_or("constant".to_string());
        let pad_mode = match pad_mode.as_str() {
            "constant" => PadMode::Constant,
            "reflect" => PadMode::Reflect,
            "edge" => PadMode::Edge,
            "wrap" => PadMode::Wrap,
            _ => PadMode::Constant,
        };

        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Pad"))?,
            pads: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Pad"))?,
            constant_value: if inputs.len() > 2 {
                Some(inputs[2].ok_or(ONNXDecodingError::InvalidOperatorInputs("Pad"))?)
            } else {
                None
            },
            axes: if inputs.len() > 3 {
                Some(inputs[3].ok_or(ONNXDecodingError::InvalidOperatorInputs("Pad"))?)
            } else {
                None
            },
            mode: pad_mode,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Pad"))?,
        })
    }
}

impl Operation for PadOperation {
    fn get_op_type_name(&self) -> String {
        "Pad".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        let mut ret = vec![self.input, self.pads];
        if let Some(constant_value) = self.constant_value {
            ret.push(constant_value);
        }
        if let Some(axes) = self.axes {
            ret.push(axes);
        }
        ret
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RandomNormalLikeOperation {
    input: TensorId,
    output: TensorId,
    dtype: Option<DType>,
    mean: f32,
    scale: f32,
    seed: Option<f32>,
}

impl RandomNormalLikeOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("RandomNormalLike"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs(
                "RandomNormalLike",
            ));
        }

        let dtype = attributes.iter().find(|a| a.name == "dtype");
        let dtype = if let Some(dtype) = dtype {
            let to_datatype = onnx::tensor_proto::DataType::try_from(dtype.i as i32)
                .map_err(|x| ONNXDecodingError::ProtobufDecodeError(x.into()))?;
            Some(DType::try_from(to_datatype)?)
        } else {
            None
        };

        let mean = query_attribute_float(attributes, "mean").unwrap_or(0.0);
        let scale = query_attribute_float(attributes, "scale").unwrap_or(1.0);
        let seed = query_attribute_float(attributes, "seed");

        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("RandomNormalLike"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs(
                "RandomNormalLike",
            ))?,
            dtype,
            mean,
            scale,
            seed,
        })
    }
}

impl Operation for RandomNormalLikeOperation {
    fn get_op_type_name(&self) -> String {
        "Random Normal Like".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input]
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArgMaxOperation {
    axis: i64,
    keepdims: bool,
    select_last_index: bool,
    input: TensorId,
    output: TensorId,
}

impl ArgMaxOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
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

    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input]
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
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
    input: TensorId,
    output: TensorId,
}

impl ArgMinOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
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

    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input]
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
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
    inputs: Vec<TensorId>,
    output: TensorId,
}

impl MaxOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
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

    fn get_inputs(&self) -> Vec<TensorId> {
        self.inputs.clone()
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
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
    inputs: Vec<TensorId>,
    output: TensorId,
}

impl MinOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
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

    fn get_inputs(&self) -> Vec<TensorId> {
        self.inputs.clone()
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IfOperation {
    outputs: Vec<TensorId>,
    condition: TensorId,
    then_branch: SymbolicGraphInner,
    else_branch: SymbolicGraphInner,
}

impl IfOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
        attributes: &[onnx::AttributeProto],
        symbolic_graph_mutator: &mut SymbolicGraphMutator,
        core_opset_version: usize,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("If"));
        }
        if outputs.is_empty() {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("If"));
        }

        let then_branch_graph = query_attribute_graph(attributes, "then_branch")
            .ok_or(ONNXDecodingError::MissingField("then_branch"))?;
        let then_branch_graph = {
            let mut inner_graph = SymbolicGraphInner::new();
            inner_graph.populate(
                symbolic_graph_mutator,
                then_branch_graph,
                core_opset_version,
            )?;
            inner_graph
        };
        let else_branch_graph = query_attribute_graph(attributes, "else_branch")
            .ok_or(ONNXDecodingError::MissingField("else_branch"))?;
        let else_branch_graph = {
            let mut inner_graph = SymbolicGraphInner::new();
            inner_graph.populate(
                symbolic_graph_mutator,
                else_branch_graph,
                core_opset_version,
            )?;
            inner_graph
        };

        Ok(Self {
            condition: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("If"))?,
            outputs: outputs
                .iter()
                .map(|x| x.ok_or(ONNXDecodingError::InvalidOperatorOutputs("Min")))
                .collect::<Result<_, _>>()?,
            then_branch: then_branch_graph,
            else_branch: else_branch_graph,
        })
    }
}

impl Operation for IfOperation {
    fn get_op_type_name(&self) -> String {
        "If".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        let mut inputs_set = HashSet::new();
        inputs_set.insert(self.condition);
        inputs_set.extend(self.then_branch.get_foreign_tensor_ids());
        inputs_set.extend(self.else_branch.get_foreign_tensor_ids());
        let mut inputs_vec: Vec<_> = inputs_set.into_iter().collect();
        inputs_vec.sort(); // Deterministic ordering
        inputs_vec
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        self.outputs.clone()
    }

    fn eval(
        &self,
        backend: &mut EvalBackend,
        inputs: &HashMap<TensorId, NumericTensor<DynRank>>,
    ) -> Result<HashMap<TensorId, NumericTensor<DynRank>>, EvalError> {
        let condition = inputs.get(&self.condition).unwrap();
        let condition: bool = condition.first_element().into();
        let (active_tensors, output_ids) = if condition {
            let tensors = self.then_branch.eval(inputs, backend)?;
            (tensors, &self.then_branch.ordered_outputs)
        } else {
            let tensors = self.else_branch.eval(inputs, backend)?;
            (tensors, &self.else_branch.ordered_outputs)
        };

        // Get all outputs
        let mut outputs = HashMap::new();
        for (to_id, from_id) in self.outputs.iter().zip(output_ids.iter()) {
            outputs.insert(*to_id, active_tensors.get(from_id).unwrap().clone());
        }
        Ok(outputs)
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        todo!()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScanOperation {
    scan_inputs: Vec<Option<TensorId>>,
    state_inputs: Vec<Option<TensorId>>,
    scan_outputs: Vec<Option<TensorId>>,
    state_outputs: Vec<Option<TensorId>>,
    scan_input_axes: Option<Vec<i64>>,
    scan_input_directions: Option<Vec<i64>>,
    scan_output_axes: Option<Vec<i64>>,
    scan_output_directions: Option<Vec<i64>>,
    body: SymbolicGraphInner,
}

impl ScanOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<TensorId>],
        outputs: &[Option<TensorId>],
        attributes: &[onnx::AttributeProto],
        symbolic_graph_mutator: &mut SymbolicGraphMutator,
        core_opset_version: usize,
    ) -> Result<Self, ONNXDecodingError> {
        let body = query_attribute_graph(attributes, "body")
            .ok_or(ONNXDecodingError::MissingField("body"))?;
        let body = {
            let mut inner_graph = SymbolicGraphInner::new();
            inner_graph.populate(symbolic_graph_mutator, body, core_opset_version)?;
            inner_graph
        };

        let scan_inputs_start = if core_opset_version < 9 {
            //1
            panic!("We don't support this version of scan!")
        } else {
            0
        };

        let num_scan_inputs = query_attribute_int(attributes, "num_scan_inputs")
            .ok_or(ONNXDecodingError::MissingField("num_scan_inputs"))?;
        assert!(num_scan_inputs <= inputs.len() as i64);
        assert!(num_scan_inputs >= 1);
        let num_state_tensors = (inputs.len() - scan_inputs_start) - num_scan_inputs as usize;

        let scan_input_axes = query_attribute_ints(attributes, "scan_input_axes");
        let scan_input_directions = query_attribute_ints(attributes, "scan_input_directions");
        let scan_output_axes = query_attribute_ints(attributes, "scan_output_axes");
        let scan_output_directions = query_attribute_ints(attributes, "scan_output_directions");

        let state_inputs = inputs[scan_inputs_start..num_state_tensors].to_vec();
        let scan_inputs = inputs[scan_inputs_start + num_state_tensors..].to_vec();

        let state_outputs = outputs[..num_state_tensors].to_vec();
        let scan_outputs = outputs[num_state_tensors..].to_vec();

        Ok(Self {
            state_inputs,
            scan_inputs,
            state_outputs,
            scan_outputs,
            body,
            scan_input_axes,
            scan_input_directions,
            scan_output_axes,
            scan_output_directions,
        })
    }
}

impl Operation for ScanOperation {
    fn get_op_type_name(&self) -> String {
        "Scan".to_string()
    }

    fn get_inputs(&self) -> Vec<TensorId> {
        let mut inputs_set = HashSet::new();
        inputs_set.extend(self.state_inputs.iter().filter_map(|x| *x));
        inputs_set.extend(self.scan_inputs.iter().filter_map(|x| *x));
        inputs_set.extend(self.body.get_foreign_tensor_ids());
        let mut inputs_vec: Vec<_> = inputs_set.into_iter().collect();
        inputs_vec.sort(); // Deterministic ordering
        inputs_vec
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        let mut outputs = Vec::new();
        outputs.extend(self.state_outputs.iter().filter_map(|x| *x));
        outputs.extend(self.scan_outputs.iter().filter_map(|x| *x));
        outputs
    }

    fn eval(
        &self,
        backend: &mut EvalBackend,
        inputs: &HashMap<TensorId, NumericTensor<DynRank>>,
    ) -> Result<HashMap<TensorId, NumericTensor<DynRank>>, EvalError> {
        let state_inputs: Vec<_> = self
            .state_inputs
            .iter()
            .map(|x| x.map(|x| inputs[&x].clone()))
            .collect();
        let scan_inputs: Vec<_> = self
            .scan_inputs
            .iter()
            .map(|x| inputs[&x.unwrap()].clone())
            .collect();

        let scan_input_axes = if let Some(scan_input_axes) = &self.scan_input_axes {
            let mut output = Vec::new();
            for (i, axis) in scan_input_axes.iter().enumerate() {
                if *axis >= 0 {
                    output.push(*axis as usize);
                } else {
                    output.push((scan_inputs[i].rank() as i64 + *axis) as usize);
                }
                assert!(output[i] < scan_inputs[i].rank());
            }
            output
        } else {
            vec![0; scan_inputs.len()]
        };

        let iter_count = scan_inputs[0].shape()[scan_input_axes[0]];

        assert!(self.scan_input_directions.is_none());
        assert!(self.scan_output_directions.is_none());

        let mut accumulated_scan_outputs = Vec::new();
        for _ in 0..self.scan_outputs.len() {
            accumulated_scan_outputs.push(Vec::new());
        }

        let mut state_tensors = state_inputs;

        for i in 0..iter_count {
            let iter_scan_inputs = {
                let mut iter_scan_inputs = Vec::new();
                for j in 0..scan_inputs.len() {
                    let mut slice_indexes = Vec::new();
                    for k in 0..scan_inputs[j].rank() {
                        if k == scan_input_axes[j] {
                            slice_indexes.push(i..i + 1);
                        } else {
                            slice_indexes.push(0..scan_inputs[j].shape()[k]);
                        }
                    }
                    let sliced = scan_inputs[j].slice(slice_indexes.as_slice(), backend)?;
                    let squeezed = sliced.squeeze(scan_input_axes[j])?;
                    iter_scan_inputs.push(squeezed);
                }
                iter_scan_inputs
            };

            let input_map = {
                let mut input_map = HashMap::new();
                let mut i = 0;
                for input in state_tensors.iter().flatten() {
                    input_map.insert(self.body.ordered_inputs[i], input.clone());
                    i += 1;
                }
                for input in &iter_scan_inputs {
                    input_map.insert(self.body.ordered_inputs[i], input.clone());
                    i += 1;
                }
                for input in self.body.get_foreign_tensor_ids() {
                    input_map.insert(input, inputs[&input].clone());
                }
                input_map
            };

            let eval_outputs = self.body.eval(&input_map, backend)?;

            let temp_outputs = self.body.ordered_outputs[0..state_tensors.len()]
                .iter()
                .map(|x| eval_outputs[x].clone())
                .collect::<Vec<_>>();
            let scan_outputs = self.body.ordered_outputs[state_tensors.len()..]
                .iter()
                .map(|x| eval_outputs[x].clone())
                .collect::<Vec<_>>();

            for (i, output) in scan_outputs.iter().enumerate() {
                accumulated_scan_outputs[i].push(output.clone());
            }
            state_tensors = temp_outputs
                .iter()
                .map(|x| Some(x.clone()))
                .collect::<Vec<_>>();
        }

        // Concatenate the accumulated outputs
        let scan_outputs = {
            let mut scan_outputs: Vec<NumericTensor<DynRank>> = Vec::new();
            for (i, outputs) in accumulated_scan_outputs.iter().enumerate() {
                let concat_dim = if let Some(x) = &self.scan_output_axes {
                    let v = x[i];
                    if v < 0 {
                        (v + (scan_outputs[0].rank() + 1) as i64) as usize
                    } else {
                        v as usize
                    }
                } else {
                    0
                };
                let mut unsqueezed_outputs = Vec::new();
                for output in outputs {
                    unsqueezed_outputs.push(output.unsqueeze(concat_dim)?);
                }
                let mut unsqueezed_outputs_refs = Vec::new();
                for output in &unsqueezed_outputs {
                    unsqueezed_outputs_refs.push(output);
                }
                scan_outputs.push(NumericTensor::concat(
                    unsqueezed_outputs_refs.as_slice(),
                    concat_dim,
                    backend,
                )?);
            }
            scan_outputs
        };

        let mut outputs = HashMap::new();
        for (i, state_tensor) in state_tensors.iter().enumerate() {
            if let Some(x) = self.state_outputs[i] {
                outputs.insert(x, state_tensor.clone().unwrap());
            }
        }
        for (i, output) in scan_outputs.iter().enumerate() {
            if let Some(x) = self.scan_outputs[i] {
                outputs.insert(x, output.clone());
            }
        }

        Ok(outputs)
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        todo!()
    }
}

#[derive(Clone, Debug, strum_macros::VariantNames, Serialize, Deserialize)]
pub enum AnyOperation {
    Unary(UnaryOperation),
    Binary(BinaryOperation),
    Cast(CastOperation),
    CastLike(CastLikeOperation),
    Squeeze(SqueezeOperation),
    Unsqueeze(UnsqueezeOperation),
    Transpose(TransposeOperation),
    Reshape(ReshapeOperation),
    CumSum(CumSumOperation),
    Gather(GatherOperation),
    LpNormalization(LpNormalizationOperation),
    GroupNormalization(GroupNormalizationOperation),
    LayerNormalization(LayerNormalizationOperation),
    Shape(ShapeOperation),
    Concat(ConcatOperation),
    ConstantOfShape(ConstantOfShapeOperation),
    ReduceMean(ReduceMeanOperation),
    ReduceSum(ReduceSumOperation),
    ReduceProd(ReduceProdOperation),
    ReduceMin(ReduceMinOperation),
    ReduceMax(ReduceMaxOperation),
    Pow(PowOperation),
    Gemm(GemmOperation),
    Split(SplitOperation),
    Slice(SliceOperation),
    Where(WhereOperation),
    Softmax(SoftmaxOperation),
    LogSoftmax(LogSoftmaxOperation),
    Size(SizeOperation),
    Range(RangeOperation),
    Flatten(FlattenOperation),
    Constant(ConstantOperation),
    Identity(IdentityOperation),
    Clip(ClipOperation),
    IsInf(IsInfOperation),
    Modulo(ModuloOperation),
    Expand(ExpandOperation),
    Conv(ConvOperation),
    InstanceNormalization(InstanceNormalizationOperation),
    Resize(ResizeOperation),
    Pad(PadOperation),
    RandomNormalLike(RandomNormalLikeOperation),
    ArgMax(ArgMaxOperation),
    ArgMin(ArgMinOperation),
    Max(MaxOperation),
    Min(MinOperation),
    If(IfOperation),
    Scan(ScanOperation),
}

impl AnyOperation {
    fn as_dyn(&self) -> &dyn Operation {
        match self {
            AnyOperation::Unary(op) => op,
            AnyOperation::Binary(op) => op,
            AnyOperation::Cast(op) => op,
            AnyOperation::CastLike(op) => op,
            AnyOperation::Squeeze(op) => op,
            AnyOperation::Unsqueeze(op) => op,
            AnyOperation::Transpose(op) => op,
            AnyOperation::Reshape(op) => op,
            AnyOperation::CumSum(op) => op,
            AnyOperation::Gather(op) => op,
            AnyOperation::LpNormalization(op) => op,
            AnyOperation::GroupNormalization(op) => op,
            AnyOperation::LayerNormalization(op) => op,
            AnyOperation::Shape(op) => op,
            AnyOperation::Concat(op) => op,
            AnyOperation::ConstantOfShape(op) => op,
            AnyOperation::ReduceMean(op) => op,
            AnyOperation::ReduceSum(op) => op,
            AnyOperation::ReduceProd(op) => op,
            AnyOperation::ReduceMin(op) => op,
            AnyOperation::ReduceMax(op) => op,
            AnyOperation::Pow(op) => op,
            AnyOperation::Gemm(op) => op,
            AnyOperation::Split(op) => op,
            AnyOperation::Slice(op) => op,
            AnyOperation::Where(op) => op,
            AnyOperation::Softmax(op) => op,
            AnyOperation::LogSoftmax(op) => op,
            AnyOperation::Size(op) => op,
            AnyOperation::Range(op) => op,
            AnyOperation::Flatten(op) => op,
            AnyOperation::Constant(op) => op,
            AnyOperation::Identity(op) => op,
            AnyOperation::IsInf(op) => op,
            AnyOperation::Clip(op) => op,
            AnyOperation::Modulo(op) => op,
            AnyOperation::Expand(op) => op,
            AnyOperation::Conv(op) => op,
            AnyOperation::InstanceNormalization(op) => op,
            AnyOperation::Resize(op) => op,
            AnyOperation::Pad(op) => op,
            AnyOperation::RandomNormalLike(op) => op,
            AnyOperation::ArgMax(op) => op,
            AnyOperation::ArgMin(op) => op,
            AnyOperation::Max(op) => op,
            AnyOperation::Min(op) => op,
            AnyOperation::If(op) => op,
            AnyOperation::Scan(op) => op,
        }
    }
}

impl Operation for AnyOperation {
    fn get_op_type_name(&self) -> String {
        self.as_dyn().get_op_type_name()
    }
    fn get_inputs(&self) -> Vec<TensorId> {
        match self {
            AnyOperation::Unary(op) => op.get_inputs(),
            AnyOperation::Binary(op) => op.get_inputs(),
            AnyOperation::Cast(op) => op.get_inputs(),
            AnyOperation::CastLike(op) => op.get_inputs(),
            AnyOperation::Squeeze(op) => op.get_inputs(),
            AnyOperation::Unsqueeze(op) => op.get_inputs(),
            AnyOperation::Transpose(op) => op.get_inputs(),
            AnyOperation::Reshape(op) => op.get_inputs(),
            AnyOperation::CumSum(op) => op.get_inputs(),
            AnyOperation::Gather(op) => op.get_inputs(),
            AnyOperation::LpNormalization(op) => op.get_inputs(),
            AnyOperation::GroupNormalization(op) => op.get_inputs(),
            AnyOperation::LayerNormalization(op) => op.get_inputs(),
            AnyOperation::Shape(op) => op.get_inputs(),
            AnyOperation::Concat(op) => op.get_inputs(),
            AnyOperation::ConstantOfShape(op) => op.get_inputs(),
            AnyOperation::ReduceMean(op) => op.get_inputs(),
            AnyOperation::ReduceSum(op) => op.get_inputs(),
            AnyOperation::ReduceProd(op) => op.get_inputs(),
            AnyOperation::ReduceMin(op) => op.get_inputs(),
            AnyOperation::ReduceMax(op) => op.get_inputs(),
            AnyOperation::Pow(op) => op.get_inputs(),
            AnyOperation::Gemm(op) => op.get_inputs(),
            AnyOperation::Split(op) => op.get_inputs(),
            AnyOperation::Slice(op) => op.get_inputs(),
            AnyOperation::Where(op) => op.get_inputs(),
            AnyOperation::Softmax(op) => op.get_inputs(),
            AnyOperation::LogSoftmax(op) => op.get_inputs(),
            AnyOperation::Size(op) => op.get_inputs(),
            AnyOperation::Range(op) => op.get_inputs(),
            AnyOperation::Flatten(op) => op.get_inputs(),
            AnyOperation::Constant(op) => op.get_inputs(),
            AnyOperation::Identity(op) => op.get_inputs(),
            AnyOperation::IsInf(op) => op.get_inputs(),
            AnyOperation::Clip(op) => op.get_inputs(),
            AnyOperation::Modulo(op) => op.get_inputs(),
            AnyOperation::Expand(op) => op.get_inputs(),
            AnyOperation::Conv(op) => op.get_inputs(),
            AnyOperation::InstanceNormalization(op) => op.get_inputs(),
            AnyOperation::Resize(op) => op.get_inputs(),
            AnyOperation::Pad(op) => op.get_inputs(),
            AnyOperation::RandomNormalLike(op) => op.get_inputs(),
            AnyOperation::ArgMax(op) => op.get_inputs(),
            AnyOperation::ArgMin(op) => op.get_inputs(),
            AnyOperation::Max(op) => op.get_inputs(),
            AnyOperation::Min(op) => op.get_inputs(),
            AnyOperation::If(op) => op.get_inputs(),
            AnyOperation::Scan(op) => op.get_inputs(),
        }
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        match self {
            AnyOperation::Unary(op) => op.get_outputs(),
            AnyOperation::Binary(op) => op.get_outputs(),
            AnyOperation::Cast(op) => op.get_outputs(),
            AnyOperation::CastLike(op) => op.get_outputs(),
            AnyOperation::Squeeze(op) => op.get_outputs(),
            AnyOperation::Unsqueeze(op) => op.get_outputs(),
            AnyOperation::Transpose(op) => op.get_outputs(),
            AnyOperation::Reshape(op) => op.get_outputs(),
            AnyOperation::CumSum(op) => op.get_outputs(),
            AnyOperation::Gather(op) => op.get_outputs(),
            AnyOperation::LpNormalization(op) => op.get_outputs(),
            AnyOperation::GroupNormalization(op) => op.get_outputs(),
            AnyOperation::LayerNormalization(op) => op.get_outputs(),
            AnyOperation::Shape(op) => op.get_outputs(),
            AnyOperation::Concat(op) => op.get_outputs(),
            AnyOperation::ConstantOfShape(op) => op.get_outputs(),
            AnyOperation::ReduceMean(op) => op.get_outputs(),
            AnyOperation::ReduceSum(op) => op.get_outputs(),
            AnyOperation::ReduceProd(op) => op.get_outputs(),
            AnyOperation::ReduceMin(op) => op.get_outputs(),
            AnyOperation::ReduceMax(op) => op.get_outputs(),
            AnyOperation::Pow(op) => op.get_outputs(),
            AnyOperation::Gemm(op) => op.get_outputs(),
            AnyOperation::Split(op) => op.get_outputs(),
            AnyOperation::Slice(op) => op.get_outputs(),
            AnyOperation::Where(op) => op.get_outputs(),
            AnyOperation::Softmax(op) => op.get_outputs(),
            AnyOperation::LogSoftmax(op) => op.get_outputs(),
            AnyOperation::Size(op) => op.get_outputs(),
            AnyOperation::Range(op) => op.get_outputs(),
            AnyOperation::Flatten(op) => op.get_outputs(),
            AnyOperation::Constant(op) => op.get_outputs(),
            AnyOperation::Identity(op) => op.get_outputs(),
            AnyOperation::IsInf(op) => op.get_outputs(),
            AnyOperation::Clip(op) => op.get_outputs(),
            AnyOperation::Modulo(op) => op.get_outputs(),
            AnyOperation::Expand(op) => op.get_outputs(),
            AnyOperation::Conv(op) => op.get_outputs(),
            AnyOperation::InstanceNormalization(op) => op.get_outputs(),
            AnyOperation::Resize(op) => op.get_outputs(),
            AnyOperation::Pad(op) => op.get_outputs(),
            AnyOperation::RandomNormalLike(op) => op.get_outputs(),
            AnyOperation::ArgMax(op) => op.get_outputs(),
            AnyOperation::ArgMin(op) => op.get_outputs(),
            AnyOperation::Max(op) => op.get_outputs(),
            AnyOperation::Min(op) => op.get_outputs(),
            AnyOperation::If(op) => op.get_outputs(),
            AnyOperation::Scan(op) => op.get_outputs(),
        }
    }

    fn eval(
        &self,
        backend: &mut EvalBackend,
        inputs: &HashMap<TensorId, NumericTensor<DynRank>>,
    ) -> Result<HashMap<TensorId, NumericTensor<DynRank>>, EvalError> {
        match self {
            AnyOperation::Unary(op) => op.eval(backend, inputs),
            AnyOperation::Binary(op) => op.eval(backend, inputs),
            AnyOperation::Cast(op) => op.eval(backend, inputs),
            AnyOperation::CastLike(op) => op.eval(backend, inputs),
            AnyOperation::Squeeze(op) => op.eval(backend, inputs),
            AnyOperation::Unsqueeze(op) => op.eval(backend, inputs),
            AnyOperation::Transpose(op) => op.eval(backend, inputs),
            AnyOperation::Reshape(op) => op.eval(backend, inputs),
            AnyOperation::CumSum(op) => op.eval(backend, inputs),
            AnyOperation::Gather(op) => op.eval(backend, inputs),
            AnyOperation::LpNormalization(op) => op.eval(backend, inputs),
            AnyOperation::GroupNormalization(op) => op.eval(backend, inputs),
            AnyOperation::LayerNormalization(op) => op.eval(backend, inputs),
            AnyOperation::Shape(op) => op.eval(backend, inputs),
            AnyOperation::Concat(op) => op.eval(backend, inputs),
            AnyOperation::ConstantOfShape(op) => op.eval(backend, inputs),
            AnyOperation::ReduceMean(op) => op.eval(backend, inputs),
            AnyOperation::ReduceSum(op) => op.eval(backend, inputs),
            AnyOperation::ReduceProd(op) => op.eval(backend, inputs),
            AnyOperation::ReduceMax(op) => op.eval(backend, inputs),
            AnyOperation::ReduceMin(op) => op.eval(backend, inputs),
            AnyOperation::Pow(op) => op.eval(backend, inputs),
            AnyOperation::Gemm(op) => op.eval(backend, inputs),
            AnyOperation::Split(op) => op.eval(backend, inputs),
            AnyOperation::Slice(op) => op.eval(backend, inputs),
            AnyOperation::Where(op) => op.eval(backend, inputs),
            AnyOperation::Softmax(op) => op.eval(backend, inputs),
            AnyOperation::LogSoftmax(op) => op.eval(backend, inputs),
            AnyOperation::Size(op) => op.eval(backend, inputs),
            AnyOperation::Range(op) => op.eval(backend, inputs),
            AnyOperation::Flatten(op) => op.eval(backend, inputs),
            AnyOperation::Constant(op) => op.eval(backend, inputs),
            AnyOperation::Identity(op) => op.eval(backend, inputs),
            AnyOperation::IsInf(op) => op.eval(backend, inputs),
            AnyOperation::Clip(op) => op.eval(backend, inputs),
            AnyOperation::Modulo(op) => op.eval(backend, inputs),
            AnyOperation::Expand(op) => op.eval(backend, inputs),
            AnyOperation::Conv(op) => op.eval(backend, inputs),
            AnyOperation::InstanceNormalization(op) => op.eval(backend, inputs),
            AnyOperation::Resize(op) => op.eval(backend, inputs),
            AnyOperation::Pad(op) => op.eval(backend, inputs),
            AnyOperation::RandomNormalLike(op) => op.eval(backend, inputs),
            AnyOperation::ArgMax(op) => op.eval(backend, inputs),
            AnyOperation::ArgMin(op) => op.eval(backend, inputs),
            AnyOperation::Max(op) => op.eval(backend, inputs),
            AnyOperation::Min(op) => op.eval(backend, inputs),
            AnyOperation::If(op) => op.eval(backend, inputs),
            AnyOperation::Scan(op) => op.eval(backend, inputs),
        }
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<TensorId> {
        match self {
            AnyOperation::Unary(op) => op.get_milli_op_graph(),
            AnyOperation::Binary(op) => op.get_milli_op_graph(),
            AnyOperation::Cast(op) => op.get_milli_op_graph(),
            AnyOperation::CastLike(op) => op.get_milli_op_graph(),
            AnyOperation::Squeeze(op) => op.get_milli_op_graph(),
            AnyOperation::Unsqueeze(op) => op.get_milli_op_graph(),
            AnyOperation::Transpose(op) => op.get_milli_op_graph(),
            AnyOperation::Reshape(op) => op.get_milli_op_graph(),
            AnyOperation::CumSum(op) => op.get_milli_op_graph(),
            AnyOperation::Gather(op) => op.get_milli_op_graph(),
            AnyOperation::LpNormalization(op) => op.get_milli_op_graph(),
            AnyOperation::GroupNormalization(op) => op.get_milli_op_graph(),
            AnyOperation::LayerNormalization(op) => op.get_milli_op_graph(),
            AnyOperation::Shape(op) => op.get_milli_op_graph(),
            AnyOperation::Concat(op) => op.get_milli_op_graph(),
            AnyOperation::ConstantOfShape(op) => op.get_milli_op_graph(),
            AnyOperation::ReduceMean(op) => op.get_milli_op_graph(),
            AnyOperation::ReduceSum(op) => op.get_milli_op_graph(),
            AnyOperation::ReduceProd(op) => op.get_milli_op_graph(),
            AnyOperation::ReduceMax(op) => op.get_milli_op_graph(),
            AnyOperation::ReduceMin(op) => op.get_milli_op_graph(),
            AnyOperation::Pow(op) => op.get_milli_op_graph(),
            AnyOperation::Gemm(op) => op.get_milli_op_graph(),
            AnyOperation::Split(op) => op.get_milli_op_graph(),
            AnyOperation::Slice(op) => op.get_milli_op_graph(),
            AnyOperation::Where(op) => op.get_milli_op_graph(),
            AnyOperation::Softmax(op) => op.get_milli_op_graph(),
            AnyOperation::LogSoftmax(op) => op.get_milli_op_graph(),
            AnyOperation::Size(op) => op.get_milli_op_graph(),
            AnyOperation::Range(op) => op.get_milli_op_graph(),
            AnyOperation::Flatten(op) => op.get_milli_op_graph(),
            AnyOperation::Constant(op) => op.get_milli_op_graph(),
            AnyOperation::Identity(op) => op.get_milli_op_graph(),
            AnyOperation::IsInf(op) => op.get_milli_op_graph(),
            AnyOperation::Clip(op) => op.get_milli_op_graph(),
            AnyOperation::Modulo(op) => op.get_milli_op_graph(),
            AnyOperation::Expand(op) => op.get_milli_op_graph(),
            AnyOperation::Conv(op) => op.get_milli_op_graph(),
            AnyOperation::InstanceNormalization(op) => op.get_milli_op_graph(),
            AnyOperation::Resize(op) => op.get_milli_op_graph(),
            AnyOperation::Pad(op) => op.get_milli_op_graph(),
            AnyOperation::RandomNormalLike(op) => op.get_milli_op_graph(),
            AnyOperation::ArgMax(op) => op.get_milli_op_graph(),
            AnyOperation::ArgMin(op) => op.get_milli_op_graph(),
            AnyOperation::Max(op) => op.get_milli_op_graph(),
            AnyOperation::Min(op) => op.get_milli_op_graph(),
            AnyOperation::If(op) => op.get_milli_op_graph(),
            AnyOperation::Scan(op) => op.get_milli_op_graph(),
        }
    }
}
