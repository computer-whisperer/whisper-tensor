use std::collections::HashMap;
use crate::dtype::DType;
use crate::eval_backend::EvalBackend;
use crate::ndarray_backend::{NDArrayNumericTensor, NDArrayNumericTensorError};
use crate::numeric_scalar::NumericScalar;
use crate::numeric_tensor::{NumericTensor, NumericTensorError};
use crate::{onnx, TrigOp};
use crate::symbolic_graph::{milli_ops_helpers, query_attribute_bool, query_attribute_float, query_attribute_floats, query_attribute_int, query_attribute_ints, query_attribute_tensor, ONNXDecodingError, SymbolicGraphError, TensorId};
use crate::symbolic_graph::milli_ops::*;
use crate::tensor_rank::DynRank;

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
}

pub trait Operation {
    fn get_inputs(&self) -> Vec<TensorId>;
    fn get_outputs(&self) -> Vec<TensorId>;
    
    fn eval(&self, backend: &EvalBackend, inputs: &HashMap<TensorId, NumericTensor<DynRank>>) -> Result<HashMap<TensorId, NumericTensor<DynRank>>, EvalError> {
        let milli_graph = self.get_milli_op_graph();
        Ok(milli_graph.eval(inputs, backend)?)
    }
    fn get_milli_op_graph(&self) -> MilliOpGraph;
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum WhichBinaryOperation {
    Add,
    Sub,
    Mul,
    Div,
    MatMul
}

#[derive(Clone, Debug, PartialEq)]
pub struct BinaryOperation {
    a: TensorId,
    b: TensorId,
    output: TensorId,
    which: WhichBinaryOperation
}

impl BinaryOperation {
    pub(crate) fn from_onnx(inputs: Vec<TensorId>, outputs: Vec<TensorId>, which: WhichBinaryOperation) -> Result<Self, SymbolicGraphError> {
        if inputs.len() != 2 {
            return Err(SymbolicGraphError::InvalidOperatorInputs);
        }
        if outputs.len() != 1 {
            return Err(SymbolicGraphError::InvalidOperatorOutputs);
        }
        Ok(BinaryOperation {
            a: inputs[0],
            b: inputs[1],
            output: outputs[0],
            which
        })
    }
}

impl Operation for BinaryOperation {
    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.a, self.b]
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let a = input_map[&self.a];
        let b = input_map[&self.b];
        let res = match self.which {
            WhichBinaryOperation::Add => AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::add(a, b)),
            WhichBinaryOperation::Sub => AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::sub(a, b)),
            WhichBinaryOperation::Mul => AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::mul(a, b)),
            WhichBinaryOperation::Div => AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::div(a, b)),
            WhichBinaryOperation::MatMul => AnyMilliOp::MatMul(MilliOpMatMul::new(a, b)),
        };
        let mut output_map = HashMap::new();
        let out = graph.push_op(res);
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum WhichUnaryOperation {
    Relu,
    Sigmoid,
    Exp,
    Log,
    Softplus,
    Reciprocal,
    Neg,
    Abs,
    NonZero,
    Sqrt,
    Trig(TrigOp)
}

#[derive(Clone, Debug, PartialEq)]
pub struct UnaryOperation {
    input: TensorId,
    output: TensorId,
    which: WhichUnaryOperation
}

impl UnaryOperation {
    pub(crate) fn from_onnx(inputs: Vec<TensorId>, outputs: Vec<TensorId>, which: WhichUnaryOperation) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1  {
            return Err(ONNXDecodingError::GraphConstructionError("Unary".to_string(), SymbolicGraphError::InvalidOperatorInputs));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::GraphConstructionError("Unary".to_string(), SymbolicGraphError::InvalidOperatorOutputs));
        }

        Ok(Self {
            input: inputs[0],
            output: outputs[0],
            which
        })
    }
}

impl Operation for UnaryOperation {
    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input]
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let a = input_map[&self.input];
        let res = match &self.which {
            WhichUnaryOperation::Relu => AnyMilliOp::ClampMin(MilliOpClampMin::new(a, 0.0)),
            WhichUnaryOperation::Sigmoid => {
                let x = graph.push_op(AnyMilliOp::Neg(MilliOpNeg::new(a)));
                let x = graph.push_op(AnyMilliOp::Exp(MilliOpExp::new(x)));
                let c = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(1.0f32)));
                let c = graph.push_op(AnyMilliOp::CastLike(MilliOpCastLike::new(c, x)));
                let x = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::add(x, c)));
                AnyMilliOp::Reciprocal(MilliOpReciprocal::new(x))
            },
            WhichUnaryOperation::Exp => AnyMilliOp::Exp(MilliOpExp::new(a)),
            WhichUnaryOperation::Log => AnyMilliOp::Log(MilliOpLog::new(a)),
            WhichUnaryOperation::Softplus => {
                let x = graph.push_op(AnyMilliOp::Exp(MilliOpExp::new(a)));
                let c = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(1.0f32)));
                let c = graph.push_op(AnyMilliOp::CastLike(MilliOpCastLike::new(c, x)));
                let x = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::add(x, c)));
                AnyMilliOp::Log(MilliOpLog::new(x))
            }
            WhichUnaryOperation::Neg => AnyMilliOp::Neg(MilliOpNeg::new(a)),
            WhichUnaryOperation::NonZero => AnyMilliOp::NonZero(MilliOpNonZero::new(a)),
            WhichUnaryOperation::Sqrt => AnyMilliOp::Sqrt(MilliOpSqrt::new(a)),
            WhichUnaryOperation::Abs => AnyMilliOp::Abs(MilliOpAbs::new(a)),
            WhichUnaryOperation::Trig(trig_op) => AnyMilliOp::Trig(MilliOpTrig::new(a, *trig_op)),
            WhichUnaryOperation::Reciprocal => AnyMilliOp::Reciprocal(MilliOpReciprocal::new(a))
        };
        let mut output_map = HashMap::new();
        let out = graph.push_op(res);
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CumSumOperation {
    input: TensorId,
    output: TensorId,
    axis: TensorId,
    exclusive: bool,
    reverse: bool
}

impl CumSumOperation {
    pub(crate) fn from_onnx(inputs: Vec<TensorId>, outputs: Vec<TensorId>) -> Result<Self, SymbolicGraphError> {
        if inputs.len() != 2 {
            return Err(SymbolicGraphError::InvalidOperatorInputs);
        }
        if outputs.len() != 1 {
            return Err(SymbolicGraphError::InvalidOperatorOutputs);
        }
        Ok(Self {
            input: inputs[0],
            axis: inputs[1],
            output: outputs[0],
            exclusive: false,
            reverse: false
        })
    }
}
impl Operation for CumSumOperation {
    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input, self.axis]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let a = input_map[&self.input];
        let b = input_map[&self.axis];

        let out = graph.push_op(AnyMilliOp::CumSum(MilliOpCumSum::new(a, b, self.exclusive, self.reverse)));

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LpNormalizationOperation {
    input: TensorId,
    output: TensorId,
    axis: i64,
    p: i64,
}

impl LpNormalizationOperation {
    pub fn from_onnx(inputs: Vec<TensorId>, outputs: Vec<TensorId>, attributes: &[onnx::AttributeProto]) -> Result<Self, SymbolicGraphError> {
        if inputs.len() != 1 {
            return Err(SymbolicGraphError::InvalidOperatorInputs);
        }
        if outputs.len() != 1 {
            return Err(SymbolicGraphError::InvalidOperatorOutputs);
        }
        let mut axis = -1;
        let mut p = 2;
        for attr in attributes{
            match attr.name.as_str() {
                "axis" => axis = attr.i,
                "p" => p = attr.i,
                _ => {}
            }
        }
        match p {
            1 | 2 => {},
            _ => return Err(SymbolicGraphError::InvalidOperatorInputs),
        }

        Ok(Self {
            input: inputs[0],
            output: outputs[0],
            axis,
            p,
        })
    }
}

impl Operation for LpNormalizationOperation {
    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let input = input_map[&self.input];

        let x = graph.push_op(AnyMilliOp::Abs(MilliOpAbs::new(input)));

        let x = match self.p{
            1 => x,
            2 => graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::mul(input, input))),
            _ => panic!()
        };
        let axis_tensor = NDArrayNumericTensor::from(vec![self.axis]).try_to_rank::<DynRank>().unwrap();
        let axis = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new(axis_tensor.into())));
        let x = graph.push_op(AnyMilliOp::ReduceSum(MilliOpReduceSum::new(x, Some(axis), true)));
        let x = match self.p{
            1 => x,
            2 => graph.push_op(AnyMilliOp::Sqrt(MilliOpSqrt::new(input))),
            _ => panic!()
        };
        let out = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::div(input, x)));

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GroupNormalizationOperation {
    input: TensorId,
    scale: TensorId,
    bias: TensorId,
    output: TensorId,
    epsilon: f32,
    num_groups: usize
}

impl GroupNormalizationOperation {
    pub fn from_onnx(inputs: Vec<TensorId>, outputs: Vec<TensorId>, attributes: &[onnx::AttributeProto]) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 3 {
            return Err(ONNXDecodingError::GraphConstructionError("GroupNormalization".to_string(), SymbolicGraphError::InvalidOperatorInputs));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::GraphConstructionError("GroupNormalization".to_string(), SymbolicGraphError::InvalidOperatorOutputs));
        }
        let mut epsilon = 1e-5;
        let mut num_groups = None;
        for attr in attributes{
            match attr.name.as_str() {
                "epsilon" => epsilon = attr.f,
                "num_groups" => num_groups = Some(attr.i),
                _ => {}
            }
        }
        let num_groups = num_groups.ok_or(ONNXDecodingError::MissingAttribute("GroupNormalization".to_string(), "num_groups".to_string()))? as usize;
        Ok(Self {
            input: inputs[0],
            scale: inputs[1],
            bias: inputs[2],
            output: outputs[0],
            epsilon,
            num_groups
        })
    }
}

impl Operation for GroupNormalizationOperation {
    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input, self.scale, self.bias]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
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
                None
            )))
        };
        let reshaped_input = {
            let new_shape_tensor = NDArrayNumericTensor::from(vec![0i64, self.num_groups as i64, -1]);
            let new_shape = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new(new_shape_tensor.to_dyn().into())));
            graph.push_op(AnyMilliOp::Reshape(MilliOpReshape::new(input, new_shape, false)))
        };

        let mean_axis = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(2i64)));
        let mean = graph.push_op(AnyMilliOp::ReduceMean(MilliOpReduceMean::new(reshaped_input, Some(mean_axis), true)));

        let input = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::sub(reshaped_input, mean)));

        let variance = {
            let x = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::mul(input, input)));
            graph.push_op(AnyMilliOp::ReduceMean(MilliOpReduceMean::new(x, Some(mean_axis), true)))
        };

        let input_normalized = {
            let epsilon = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(self.epsilon)));
            let var_plus_eps = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::add(variance, epsilon)));
            let val = graph.push_op(AnyMilliOp::Sqrt(MilliOpSqrt::new(var_plus_eps)));
            graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::div(input, val)))
        };

        let zero = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(0i64)));
        let neg_one = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(-1i64)));
        let one = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(1i64)));

        let y = {

            let new_shape = graph.push_op(AnyMilliOp::Concat(MilliOpConcat::new(vec![zero, num_channels, neg_one], 0)));
            graph.push_op(AnyMilliOp::Reshape(MilliOpReshape::new(input_normalized, new_shape, false)))
        };

        let y = {
            let scale = graph.push_op(AnyMilliOp::Unsqueeze(MilliOpUnsqueeze::new(
                input_map[&self.scale],
                one
            )));
            graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::mul(y, scale)))
        };

        let y = {
            let bias = graph.push_op(AnyMilliOp::Unsqueeze(MilliOpUnsqueeze::new(
                input_map[&self.bias],
                one
            )));
            graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::add(y, bias)))
        };

        let out = graph.push_op(AnyMilliOp::Reshape(MilliOpReshape::new(
            y,
            input_shape,
            false
        )));

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SqueezeOperation {
    input: TensorId,
    axes: TensorId,
    output: TensorId,
}

impl SqueezeOperation {
    pub fn from_onnx(inputs: Vec<TensorId>, outputs: Vec<TensorId>) -> Result<Self, SymbolicGraphError> {
        if inputs.len() != 2 {
            return Err(SymbolicGraphError::InvalidOperatorInputs);
        }
        if outputs.len() != 1 {
            return Err(SymbolicGraphError::InvalidOperatorOutputs);
        }
        Ok(Self {
            input: inputs[0],
            axes: inputs[1],
            output: outputs[0],
        })
    }
}

impl Operation for SqueezeOperation {
    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input, self.axes]
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let out = graph.push_op(AnyMilliOp::Squeeze(MilliOpSqueeze::new(
            input_map[&self.input],
            input_map[&self.axes]
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}


#[derive(Clone, Debug, PartialEq)]
pub struct UnsqueezeOperation {
    input: TensorId,
    axes: TensorId,
    output: TensorId,
}

impl UnsqueezeOperation {
    pub fn from_onnx(inputs: Vec<TensorId>, outputs: Vec<TensorId>) -> Result<Self, SymbolicGraphError> {
        if inputs.len() != 2 {
            return Err(SymbolicGraphError::InvalidOperatorInputs);
        }
        if outputs.len() != 1 {
            return Err(SymbolicGraphError::InvalidOperatorOutputs);
        }
        Ok(Self {
            input: inputs[0],
            axes: inputs[1],
            output: outputs[0],
        })
    }
}

impl Operation for UnsqueezeOperation {
    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input, self.axes]
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let out = graph.push_op(AnyMilliOp::Unsqueeze(MilliOpUnsqueeze::new(
            input_map[&self.input],
            input_map[&self.axes]
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TransposeOperation {
    input: TensorId,
    output: TensorId,
    perm: Option<Vec<i64>>,
}

impl TransposeOperation {
    pub fn from_onnx(inputs: Vec<TensorId>, outputs: Vec<TensorId>, attributes: &[onnx::AttributeProto]) -> Result<Self, SymbolicGraphError> {
        if inputs.len() != 1 {
            return Err(SymbolicGraphError::InvalidOperatorInputs);
        }
        if outputs.len() != 1 {
            return Err(SymbolicGraphError::InvalidOperatorOutputs);
        }
        Ok(Self {
            input: inputs[0],
            output: outputs[0],
            perm: query_attribute_ints(attributes, "perm")
        })
    }
}

impl Operation for TransposeOperation {
    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
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

#[derive(Clone, Debug, PartialEq)]
pub struct ReshapeOperation {
    input: TensorId,
    shape: TensorId,
    output: TensorId,
}

impl ReshapeOperation {
    pub fn from_onnx(inputs: Vec<TensorId>, outputs: Vec<TensorId>, _attributes: &[onnx::AttributeProto]) -> Result<Self, SymbolicGraphError> {
        if inputs.len() != 2 {
            return Err(SymbolicGraphError::InvalidOperatorInputs);
        }
        if outputs.len() != 1 {
            return Err(SymbolicGraphError::InvalidOperatorOutputs);
        }
        Ok(Self {
            input: inputs[0],
            shape: inputs[1],
            output: outputs[0],
        })
    }
}

impl Operation for ReshapeOperation {
    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input, self.shape]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let out = graph.push_op(AnyMilliOp::Reshape(MilliOpReshape::new(
            input_map[&self.input],
            input_map[&self.shape],
            false
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CastLikeOperation {
    input: TensorId,
    target_type: TensorId,
    output: TensorId,
}

impl CastLikeOperation {
    pub fn from_onnx(inputs: Vec<TensorId>, outputs: Vec<TensorId>, _attributes: &[onnx::AttributeProto]) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 2 {
            return Err(ONNXDecodingError::GraphConstructionError("CastLike".to_string(), SymbolicGraphError::InvalidOperatorInputs));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::GraphConstructionError("CastLike".to_string(), SymbolicGraphError::InvalidOperatorOutputs));
        }
        Ok(Self {
            input: inputs[0],
            target_type: inputs[1],
            output: outputs[0],
        })
    }
}

impl Operation for CastLikeOperation {
    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input, self.target_type]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let out = graph.push_op(AnyMilliOp::CastLike(MilliOpCastLike::new(
            input_map[&self.input],
            input_map[&self.target_type]
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CastOperation {
    input: TensorId,
    output: TensorId,
    to: DType,
}

impl CastOperation {
    pub fn from_onnx(inputs: Vec<TensorId>, outputs: Vec<TensorId>, attributes: &[onnx::AttributeProto]) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::GraphConstructionError("Cast".to_string(), SymbolicGraphError::InvalidOperatorInputs));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::GraphConstructionError("Cast".to_string(), SymbolicGraphError::InvalidOperatorOutputs));
        }
        let to_i = attributes.iter().find(|a| a.name == "to")
            .ok_or(ONNXDecodingError::MissingAttribute("Cast".to_string(), "to".to_string()))?.i as i32;
        let to_datatype = onnx::tensor_proto::DataType::try_from(to_i).map_err(|x| ONNXDecodingError::ProtobufDecodeError(x.into()))?;
        let to = DType::try_from(to_datatype)?;
        Ok(Self {
            input: inputs[0],
            output: outputs[0],
            to,
        })
    }
}

impl Operation for CastOperation {
    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let out = graph.push_op(AnyMilliOp::Cast(MilliOpCast::new(
            input_map[&self.input],
            self.to
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LayerNormalizationOperation {
    input: TensorId,
    scale: TensorId,
    bias: Option<TensorId>,
    output: TensorId,
    mean_output: Option<TensorId>,
    inv_std_dev_output: Option<TensorId>,
    axis: i64,
    epsilon: f32
}

impl LayerNormalizationOperation {
    pub(crate) fn from_onnx(inputs: &[TensorId], outputs: &[TensorId], attributes: &[onnx::AttributeProto]) -> Result<Self, ONNXDecodingError> {
        if inputs.len() < 2 || inputs.len() > 3  {
            return Err(ONNXDecodingError::GraphConstructionError("LayerNormalization".to_string(), SymbolicGraphError::InvalidOperatorInputs));
        }
        if outputs.len() < 1 || outputs.len() > 3 {
            return Err(ONNXDecodingError::GraphConstructionError("LayerNormalization".to_string(), SymbolicGraphError::InvalidOperatorOutputs));
        }
        let mut axis = -1;
        let mut epsilon = 1e-5;
        for attribute in attributes {
            match attribute.name.as_str() {
                "axis" => {
                    axis = attribute.i;
                },
                "epsilon" => {
                    epsilon = attribute.f;
                },
                _ => {}
            }
        }
        Ok(Self {
            input: inputs[0],
            scale: inputs[1],
            bias: if inputs.len() == 3 { Some(inputs[2]) } else { None },
            output: outputs[0],
            mean_output: if inputs.len() > 1 {Some(outputs[1])} else {None},
            inv_std_dev_output: if inputs.len() > 2 {Some(outputs[2])} else {None},
            axis,
            epsilon
        })
    }
}

impl Operation for LayerNormalizationOperation {
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
    
    fn get_milli_op_graph(&self) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let input_data = input_map[&self.input];
        let input_scale = input_map[&self.scale];

        let normalized_axes = {
            let r =MilliOpRange::new(
                milli_ops_helpers::scalar_const(&mut graph, self.axis),
                milli_ops_helpers::rank(&mut graph, input_data),
                milli_ops_helpers::scalar_const(&mut graph, 1i64),
            );
            graph.push_op(AnyMilliOp::Range(r))
        };

        let mean = graph.push_op(AnyMilliOp::ReduceMean(MilliOpReduceMean::new(
            input_data, Some(normalized_axes), true)));

        let d = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::sub(input_data, mean)));
        let dd = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::mul(d, d)));
        let variance = graph.push_op(AnyMilliOp::ReduceMean(MilliOpReduceMean::new(
            dd, Some(normalized_axes), true)));
        let stddev = graph.push_op(AnyMilliOp::Sqrt(MilliOpSqrt::new(variance)));
        let inv_stddev = graph.push_op(AnyMilliOp::Reciprocal(MilliOpReciprocal::new(stddev)));
        let normalized = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::mul(d, inv_stddev)));

        let out = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::mul(normalized, input_scale)));

        let out = if let Some(bias) = self.bias {
            graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::add(out, input_map[&bias])))
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

#[derive(Clone, Debug, PartialEq)]
pub struct GatherOperation {
    input: TensorId,
    indices: TensorId,
    output: TensorId,
    axis: i64
}

impl GatherOperation {
    pub(crate) fn from_onnx(inputs: &[TensorId], outputs: &[TensorId], attributes: &[onnx::AttributeProto]) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 2 {
            return Err(ONNXDecodingError::GraphConstructionError("Gather".to_string(), SymbolicGraphError::InvalidOperatorInputs));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::GraphConstructionError("Gather".to_string(), SymbolicGraphError::InvalidOperatorOutputs));
        }
        let mut axis = 0;
        for attribute in attributes {
            match attribute.name.as_str() {
                "axis" => {
                    axis = attribute.i;
                },
                _ => {}
            }
        }
        Ok(Self {
            input: inputs[0],
            indices: inputs[1],
            output: outputs[0],
            axis
        })
    }
}

impl Operation for GatherOperation {
    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input, self.indices]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let out = graph.push_op(AnyMilliOp::Gather(MilliOpGather::new(
            input_map[&self.input],
            input_map[&self.indices],
            self.axis
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ShapeOperation {
    start: Option<i64>,
    end: Option<i64>,
    input: TensorId,
    output: TensorId
}

impl ShapeOperation {
    pub(crate) fn from_onnx(inputs: &[TensorId], outputs: &[TensorId], attributes: &[onnx::AttributeProto]) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::GraphConstructionError("Shape".to_string(), SymbolicGraphError::InvalidOperatorInputs));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::GraphConstructionError("Shape".to_string(), SymbolicGraphError::InvalidOperatorOutputs));
        }
        let mut end = None;
        let mut start = None;
        for attribute in attributes {
            match attribute.name.as_str() {
                "start" => {
                    start = Some(attribute.i);
                },
                "end" => {
                    end = Some(attribute.i);
                }
                _ => {}
            }
        }
        Ok(Self {
            input: inputs[0],
            output: outputs[0],
            start,
            end
        })
    }
}

impl Operation for ShapeOperation {
    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let out = graph.push_op(AnyMilliOp::Shape(MilliOpShape::new(
            input_map[&self.input]
        )));
        let out = if self.start.is_some() || self.end.is_some() {
            let start = milli_ops_helpers::scalar_const(&mut graph, self.start.unwrap_or(0));
            let end = if let Some(end) = self.end {
                milli_ops_helpers::scalar_const(&mut graph, end)
            } else {
                graph.push_op(AnyMilliOp::Shape(MilliOpShape::new(out)))
            };
            graph.push_op(AnyMilliOp::Slice(MilliOpSlice::new(
                out,
                start,
                end,
                None,
                None
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

#[derive(Clone, Debug, PartialEq)]
pub struct ConcatOperation {
    axis: i64,
    inputs: Vec<TensorId>,
    output: TensorId
}

impl ConcatOperation {
    pub(crate) fn from_onnx(inputs: &[TensorId], outputs: &[TensorId], attributes: &[onnx::AttributeProto]) -> Result<Self, ONNXDecodingError> {
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::GraphConstructionError("Concat".to_string(), SymbolicGraphError::InvalidOperatorOutputs));
        }
        let mut axis = 0;
        for attribute in attributes {
            if attribute.name == "axis" {
                axis = attribute.i;
            }
        }

        Ok(Self{
            axis,
            inputs: inputs.to_vec(),
            output: outputs[0]
        })
    }
}

impl Operation for ConcatOperation {
    fn get_inputs(&self) -> Vec<TensorId> {
        self.inputs.clone()
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let mut milli_inputs = vec![];
        for input in &self.inputs {
            milli_inputs.push(input_map[input]);
        }
        let out = graph.push_op(AnyMilliOp::Concat(MilliOpConcat::new(
            milli_inputs,
            self.axis
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConstantOfShapeOperation {
    value: NumericScalar,
    input: TensorId,
    output: TensorId
}

impl ConstantOfShapeOperation {
    pub(crate) fn from_onnx(inputs: &[TensorId], outputs: &[TensorId], attributes: &[onnx::AttributeProto]) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::GraphConstructionError("ConstantOfShape".to_string(), SymbolicGraphError::InvalidOperatorInputs));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::GraphConstructionError("ConstantOfShape".to_string(), SymbolicGraphError::InvalidOperatorOutputs));
        }

        let value = query_attribute_tensor(attributes, "value")
            .map(|x| x.first_element())
            .unwrap_or(NumericScalar::F32(0.0));

        Ok(Self{
            value,
            input: inputs[0],
            output: outputs[0]
        })
    }
}

impl Operation for ConstantOfShapeOperation {
    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let out = graph.push_op(AnyMilliOp::ConstantOfShape(MilliOpConstantOfShape::new(
            self.value,
            input_map[&self.input]
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReduceMeanOperation {
    keepdims: Option<i64>,
    noop_with_empty_axes: Option<i64>,
    input_data: TensorId,
    input_axes: Option<TensorId>,
    axes_attr: Option<Vec<i64>>,
    output: TensorId
}

impl ReduceMeanOperation {
    pub(crate) fn from_onnx(inputs: &[TensorId], outputs: &[TensorId], attributes: &[onnx::AttributeProto]) -> Result<Self, ONNXDecodingError> {
        if inputs.len() < 1 || inputs.len() > 2 {
            return Err(ONNXDecodingError::GraphConstructionError("ReduceMean".to_string(), SymbolicGraphError::InvalidOperatorInputs));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::GraphConstructionError("ReduceMean".to_string(), SymbolicGraphError::InvalidOperatorOutputs));
        }

        let axes_attr = query_attribute_ints(attributes, "attr");
        let keepdims = query_attribute_int(attributes, "keepdims");
        let noop_with_empty_axes = query_attribute_int(attributes, "noop_with_empty_axes");


        Ok(
            Self {
                keepdims,
                noop_with_empty_axes,
                input_data: inputs[0],
                input_axes: if inputs.len() > 1 {Some(inputs[1])} else {None},
                output: outputs[0],
                axes_attr
            }
        )
    }
}

impl Operation for ReduceMeanOperation {
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

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let axes = if let Some(input_axes) = &self.input_axes {
            Some(input_map[input_axes])
        } else {
            if let Some(axes) = &self.axes_attr {
                let tensor = NDArrayNumericTensor::from(axes.clone());
                Some(graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new(tensor.to_dyn().into()))))
            } else {
                None
            }
        };
        let out = graph.push_op(AnyMilliOp::ReduceMean(MilliOpReduceMean::new(
            input_map[&self.input_data],
            axes,
            self.keepdims.unwrap_or(1) != 0
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReduceSumOperation {
    keepdims: Option<i64>,
    noop_with_empty_axes: Option<i64>,
    input_data: TensorId,
    input_axes: Option<TensorId>,
    axes_attr: Option<Vec<i64>>,
    output: TensorId
}

impl ReduceSumOperation {
    pub(crate) fn from_onnx(inputs: &[TensorId], outputs: &[TensorId], attributes: &[onnx::AttributeProto]) -> Result<Self, ONNXDecodingError> {
        if inputs.len() < 1 || inputs.len() > 2 {
            return Err(ONNXDecodingError::GraphConstructionError("ReduceSum".to_string(), SymbolicGraphError::InvalidOperatorInputs));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::GraphConstructionError("ReduceSum".to_string(), SymbolicGraphError::InvalidOperatorOutputs));
        }

        let axes_attr = query_attribute_ints(attributes, "attr");
        let keepdims = query_attribute_int(attributes, "keepdims");
        let noop_with_empty_axes = query_attribute_int(attributes, "noop_with_empty_axes");

        Ok(
            Self {
                keepdims,
                noop_with_empty_axes,
                input_data: inputs[0],
                input_axes: if inputs.len() > 1 {Some(inputs[1])} else {None},
                output: outputs[0],
                axes_attr
            }
        )
    }
}

impl Operation for ReduceSumOperation {
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

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let axes = if let Some(input_axes) = &self.input_axes {
            Some(input_map[input_axes])
        } else {
            if let Some(axes) = &self.axes_attr {
                let tensor = NDArrayNumericTensor::from(axes.clone());
                Some(graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new(tensor.to_dyn().into()))))
            } else {
                None
            }
        };
        let out = graph.push_op(AnyMilliOp::ReduceSum(MilliOpReduceSum::new(
            input_map[&self.input_data],
            axes,
            self.keepdims.unwrap_or(1) != 0
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}


#[derive(Clone, Debug, PartialEq)]
pub struct ReduceProdOperation {
    keepdims: Option<i64>,
    noop_with_empty_axes: Option<i64>,
    input_data: TensorId,
    input_axes: Option<TensorId>,
    axes_attr: Option<Vec<i64>>,
    output: TensorId
}

impl ReduceProdOperation {
    pub(crate) fn from_onnx(inputs: &[TensorId], outputs: &[TensorId], attributes: &[onnx::AttributeProto]) -> Result<Self, ONNXDecodingError> {
        if inputs.len() < 1 || inputs.len() > 2 {
            return Err(ONNXDecodingError::GraphConstructionError("ReduceProd".to_string(), SymbolicGraphError::InvalidOperatorInputs));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::GraphConstructionError("ReduceProd".to_string(), SymbolicGraphError::InvalidOperatorOutputs));
        }

        let axes_attr = query_attribute_ints(attributes, "attr");
        let keepdims = query_attribute_int(attributes, "keepdims");
        let noop_with_empty_axes = query_attribute_int(attributes, "noop_with_empty_axes");

        Ok(
            Self {
                keepdims,
                noop_with_empty_axes,
                input_data: inputs[0],
                input_axes: if inputs.len() > 1 {Some(inputs[1])} else {None},
                output: outputs[0],
                axes_attr
            }
        )
    }
}

impl Operation for ReduceProdOperation {
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

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let axes = if let Some(input_axes) = &self.input_axes {
            Some(input_map[input_axes])
        } else {
            if let Some(axes) = &self.axes_attr {
                let tensor = NDArrayNumericTensor::from(axes.clone());
                Some(graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new(tensor.to_dyn().into()))))
            } else {
                None
            }
        };
        let out = graph.push_op(AnyMilliOp::ReduceProd(MilliOpReduceProd::new(
            input_map[&self.input_data],
            axes,
            self.keepdims.unwrap_or(1) != 0
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}



#[derive(Clone, Debug, PartialEq)]
pub struct PowOperation {
    input_x: TensorId,
    input_y: TensorId,
    output: TensorId
}

impl PowOperation {
    pub(crate) fn from_onnx(inputs: &[TensorId], outputs: &[TensorId], _attributes: &[onnx::AttributeProto]) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 2 {
            return Err(ONNXDecodingError::GraphConstructionError("Pow".to_string(), SymbolicGraphError::InvalidOperatorInputs));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::GraphConstructionError("Pow".to_string(), SymbolicGraphError::InvalidOperatorOutputs));
        }
        Ok(Self{
            input_x: inputs[0],
            input_y: inputs[1],
            output: outputs[0]
        })
    }
}

impl Operation for PowOperation {
    fn get_inputs(&self) -> Vec<TensorId> {
        vec!(self.input_x, self.input_y)
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
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

#[derive(Clone, Debug, PartialEq)]
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
    pub(crate) fn from_onnx(inputs: &[TensorId], outputs: &[TensorId], attributes: &[onnx::AttributeProto]) -> Result<Self, ONNXDecodingError> {
        if inputs.len() < 2 || inputs.len() > 3  {
            return Err(ONNXDecodingError::GraphConstructionError("Gemm".to_string(), SymbolicGraphError::InvalidOperatorInputs));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::GraphConstructionError("Gemm".to_string(), SymbolicGraphError::InvalidOperatorOutputs));
        }

        let trans_a = query_attribute_bool(attributes, "transA");
        let trans_b = query_attribute_bool(attributes, "transB");
        let alpha = query_attribute_float(attributes, "alpha");
        let beta = query_attribute_float(attributes, "beta");

        Ok(Self{
            trans_a,
            trans_b,
            alpha,
            beta,
            input_a: inputs[0],
            input_b: inputs[1],
            input_c: if inputs.len() > 2 {Some(inputs[2])} else {None},
            output: outputs[0]
        })
    }
}

impl Operation for GemmOperation {
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

    fn get_milli_op_graph(&self) -> MilliOpGraph {
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
            let alpha_const = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(alpha)));
            let alpha_const = graph.push_op(AnyMilliOp::CastLike(MilliOpCastLike::new(alpha_const, x)));
            graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::mul(x, alpha_const)))
        } else {
            x
        };

        let x = if let Some(c) = self.input_c {
            let c = input_map[&c];
            let c = if let Some(beta) = self.beta {
                let beta_const = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(beta)));
                let beta_const = graph.push_op(AnyMilliOp::CastLike(MilliOpCastLike::new(beta_const, c)));
                graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::mul(c, beta_const)))
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

#[derive(Clone, Debug, PartialEq)]
pub struct SplitOperation {
    axis: Option<i64>,
    num_outputs: Option<i64>,
    input: TensorId,
    split: Option<TensorId>,
    split_attribute: Option<Vec<i64>>,
    outputs: Vec<TensorId>
}

impl SplitOperation {
    pub(crate) fn from_onnx(inputs: &[TensorId], outputs: &[TensorId], attributes: &[onnx::AttributeProto]) -> Result<Self, ONNXDecodingError> {
        if inputs.len() < 1 || inputs.len() > 2  {
            return Err(ONNXDecodingError::GraphConstructionError("Split".to_string(), SymbolicGraphError::InvalidOperatorInputs));
        }

        let axis = query_attribute_int(attributes, "axis");
        let num_outputs = query_attribute_int(attributes, "num_outputs");
        let split_attribute = query_attribute_ints(attributes, "split");

        Ok(Self{
            input: inputs[0],
            split: if inputs.len() > 1 {Some(inputs[1])} else {None},
            outputs: outputs.to_vec(),
            split_attribute,
            axis,
            num_outputs
        })
    }
}

impl Operation for SplitOperation {
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

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());

        let mut output_map = HashMap::new();

        for (output_id, output_tensor_id) in self.outputs.iter().enumerate() {
            let out = graph.push_op(AnyMilliOp::Split(MilliOpSplit::new(
                input_map[&self.input],
                self.split.map(|x| input_map[&x]),
                self.axis.unwrap_or_default(),
                self.num_outputs.map(|x| x as usize),
                output_id
            )));

            output_map.insert(out, output_tensor_id.clone());
        }
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SliceOperation {
    data: TensorId,
    starts: TensorId,
    ends: TensorId,
    axes: Option<TensorId>,
    steps: Option<TensorId>,
    output: TensorId
}

impl SliceOperation {
    pub(crate) fn from_onnx(inputs: &[TensorId], outputs: &[TensorId], _attributes: &[onnx::AttributeProto]) -> Result<Self, ONNXDecodingError> {
        if inputs.len() < 3 || inputs.len() > 5  {
            return Err(ONNXDecodingError::GraphConstructionError("Slice".to_string(), SymbolicGraphError::InvalidOperatorInputs));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::GraphConstructionError("Slice".to_string(), SymbolicGraphError::InvalidOperatorOutputs));
        }

        Ok(Self{
            data: inputs[0],
            starts: inputs[1],
            ends: inputs[2],
            axes: if inputs.len() > 3 {Some(inputs[3])} else {None},
            steps: if inputs.len() > 4 {Some(inputs[4])} else {None},
            output: outputs[0]
        })
    }
}

impl Operation for SliceOperation {
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

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let out = graph.push_op(AnyMilliOp::Slice(MilliOpSlice::new(
            input_map[&self.data],
            input_map[&self.starts],
            input_map[&self.ends],
            self.axes.map(|x| input_map[&x]),
            self.steps.map(|x| input_map[&x])
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct WhereOperation {
    condition: TensorId,
    x: TensorId,
    y: TensorId,
    output: TensorId
}

impl WhereOperation {
    pub(crate) fn from_onnx(inputs: &[TensorId], outputs: &[TensorId], _attributes: &[onnx::AttributeProto]) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 3  {
            return Err(ONNXDecodingError::GraphConstructionError("Where".to_string(), SymbolicGraphError::InvalidOperatorInputs));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::GraphConstructionError("Where".to_string(), SymbolicGraphError::InvalidOperatorOutputs));
        }

        Ok(Self{
            condition: inputs[0],
            x: inputs[1],
            y: inputs[2],
            output: outputs[0]
        })
    }
}

impl Operation for WhereOperation {
    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.condition, self.x, self.y]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let out = graph.push_op(AnyMilliOp::Where(MilliOpWhere::new(
            input_map[&self.condition],
            input_map[&self.x],
            input_map[&self.y]
        )));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SoftmaxOperation {
    axis: Option<i64>,
    input: TensorId,
    output: TensorId
}

impl SoftmaxOperation {
    pub(crate) fn from_onnx(inputs: &[TensorId], outputs: &[TensorId], attributes: &[onnx::AttributeProto]) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1  {
            return Err(ONNXDecodingError::GraphConstructionError("Softmax".to_string(), SymbolicGraphError::InvalidOperatorInputs));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::GraphConstructionError("Softmax".to_string(), SymbolicGraphError::InvalidOperatorOutputs));
        }

        Ok(Self{
            axis: query_attribute_int(attributes, "axis"),
            input: inputs[0],
            output: outputs[0]
        })
    }
}

impl Operation for SoftmaxOperation {
    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());

        let e = graph.push_op(AnyMilliOp::Exp(MilliOpExp::new(input_map[&self.input])));
        let axis_const = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(self.axis.unwrap_or(-1))));
        let sum = graph.push_op(AnyMilliOp::ReduceSum(MilliOpReduceSum::new(e, Some(axis_const), true)));
        let out = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::div(e, sum)));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}


#[derive(Clone, Debug, PartialEq)]
pub struct LogSoftmaxOperation {
    axis: Option<i64>,
    input: TensorId,
    output: TensorId
}

impl LogSoftmaxOperation {
    pub(crate) fn from_onnx(inputs: &[TensorId], outputs: &[TensorId], attributes: &[onnx::AttributeProto]) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1  {
            return Err(ONNXDecodingError::GraphConstructionError("LogSoftmax".to_string(), SymbolicGraphError::InvalidOperatorInputs));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::GraphConstructionError("LogSoftmax".to_string(), SymbolicGraphError::InvalidOperatorOutputs));
        }

        Ok(Self{
            axis: query_attribute_int(attributes, "axis"),
            input: inputs[0],
            output: outputs[0]
        })
    }
}

impl Operation for LogSoftmaxOperation {
    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());

        let e = graph.push_op(AnyMilliOp::Exp(MilliOpExp::new(input_map[&self.input])));
        let axis_const = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(self.axis.unwrap_or(-1))));
        let sum = graph.push_op(AnyMilliOp::ReduceSum(MilliOpReduceSum::new(e, Some(axis_const), true)));
        let softmax = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::div(e, sum)));
        let out = graph.push_op(AnyMilliOp::Log(MilliOpLog::new(softmax)));
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SizeOperation {
    input: TensorId,
    output: TensorId
}

impl SizeOperation {
    pub(crate) fn from_onnx(inputs: &[TensorId], outputs: &[TensorId]) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1  {
            return Err(ONNXDecodingError::GraphConstructionError("Size".to_string(), SymbolicGraphError::InvalidOperatorInputs));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::GraphConstructionError("Size".to_string(), SymbolicGraphError::InvalidOperatorOutputs));
        }
        Ok(Self{
            input: inputs[0],
            output: outputs[0]
        })
    }
}

impl Operation for SizeOperation {
    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());

        let shape = graph.push_op(AnyMilliOp::Shape(MilliOpShape::new(input_map[&self.input])));
        let size = graph.push_op(AnyMilliOp::ReduceProd(MilliOpReduceProd::new(shape, None, false)));

        let mut output_map = HashMap::new();
        output_map.insert(size, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RangeOperation {
    start: TensorId,
    end: TensorId,
    delta: TensorId,
    output: TensorId
}

impl RangeOperation {
    pub(crate) fn from_onnx(inputs: &[TensorId], outputs: &[TensorId]) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 3  {
            return Err(ONNXDecodingError::GraphConstructionError("Range".to_string(), SymbolicGraphError::InvalidOperatorInputs));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::GraphConstructionError("Range".to_string(), SymbolicGraphError::InvalidOperatorOutputs));
        }
        Ok(Self{
            start: inputs[0],
            end: inputs[1],
            delta: inputs[2],
            output: outputs[0]
        })
    }
}

impl Operation for RangeOperation {
    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.start, self.end, self.delta]
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
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

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FlattenOperation {
    input: TensorId,
    output: TensorId,
    axis: i64
}

impl FlattenOperation {
    pub(crate) fn from_onnx(inputs: &[TensorId], outputs: &[TensorId], attributes: &[onnx::AttributeProto]) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1  {
            return Err(ONNXDecodingError::GraphConstructionError("Flatten".to_string(), SymbolicGraphError::InvalidOperatorInputs));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::GraphConstructionError("Flatten".to_string(), SymbolicGraphError::InvalidOperatorOutputs));
        }
        let axis = query_attribute_int(attributes, "axis").unwrap_or(1);
        Ok(Self{
            input: inputs[0],
            axis,
            output: outputs[0]
        })
    }
}

impl Operation for FlattenOperation {
    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input]
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let input = input_map[&self.input];
        
        let mut output_shape = vec![];
        for _ in 0..self.axis {
            output_shape.push(0i64);
        }
        output_shape.push(-1i64);
        let shape_tensor = NDArrayNumericTensor::from(output_shape);
        let shape_constant = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new(shape_tensor.to_dyn().into())));
        let out = graph.push_op(AnyMilliOp::Reshape(MilliOpReshape::new(input, shape_constant, false)));
        
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ConstantOperation {
    value: NDArrayNumericTensor<DynRank>,
    output: TensorId
}

impl ConstantOperation {
    pub(crate) fn from_onnx(inputs: &[TensorId], outputs: &[TensorId], attributes: &[onnx::AttributeProto]) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 0  {
            return Err(ONNXDecodingError::GraphConstructionError("Constant".to_string(), SymbolicGraphError::InvalidOperatorInputs));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::GraphConstructionError("Constant".to_string(), SymbolicGraphError::InvalidOperatorOutputs));
        }
        
        let value = if let Some(tensor) = query_attribute_tensor(attributes, "value") {
            tensor
        } 
        else if let Some(value_float) = query_attribute_float(attributes, "value_float") {
            NDArrayNumericTensor::from(vec![value_float]).try_to_rank()?
        }
        else if let Some(value_floats) = query_attribute_floats(attributes, "value_floats") {
            NDArrayNumericTensor::from(value_floats).try_to_rank()?
        }
        else if let Some(value_int) = query_attribute_int(attributes, "value_int") {
            NDArrayNumericTensor::from(vec![value_int]).try_to_rank()?
        }
        else if let Some(value_ints) = query_attribute_ints(attributes, "value_ints") {
            NDArrayNumericTensor::from(value_ints).try_to_rank()?
        }
        else {
            Err(ONNXDecodingError::MissingAttribute("Constant".to_string(), "value".to_string()))?
        };
        
        Ok(Self{
            value,
            output: outputs[0]
        })
    }
}

impl Operation for ConstantOperation {
    fn get_inputs(&self) -> Vec<TensorId> {
        vec![]
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        let (mut graph, _input_map) = MilliOpGraph::new(&self.get_inputs());

        let out = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new(self.value.clone().into())));

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct IdentityOperation {
    input: TensorId,
    output: TensorId
}

impl IdentityOperation {
    pub(crate) fn from_onnx(inputs: &[TensorId], outputs: &[TensorId], _attributes: &[onnx::AttributeProto]) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1  {
            return Err(ONNXDecodingError::GraphConstructionError("Identity".to_string(), SymbolicGraphError::InvalidOperatorInputs));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::GraphConstructionError("Identity".to_string(), SymbolicGraphError::InvalidOperatorOutputs));
        }

        Ok(Self{
            input: inputs[0],
            output: outputs[0]
        })
    }
}

impl Operation for IdentityOperation {
    fn get_inputs(&self) -> Vec<TensorId> {
        vec![self.input]
    }

    fn get_outputs(&self) -> Vec<TensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let input = input_map[&self.input];
        let mut output_map = HashMap::new();
        output_map.insert(input, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug)]
pub(crate) enum AnyOperation {
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
        }
    }
}

impl Operation for AnyOperation {
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
        }
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
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
        }
    }
}