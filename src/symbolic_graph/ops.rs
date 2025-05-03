use std::collections::HashMap;
use candle_core::cpu::kernels::VecOps;
use half::{bf16, f16};
use serde::{Deserialize, Serialize};
use crate::dtype::DType;
use crate::eval_backend::EvalBackend;
use crate::ndarray_backend::{NDArrayNumericTensor, NDArrayNumericTensorError};
use crate::ndarray_backend::conversions::FromScalarShape;
use crate::numeric_tensor::{NumericTensor, NumericTensorError};
use crate::onnx;
use crate::symbolic_graph::{query_attribute_bool, query_attribute_float, query_attribute_int, query_attribute_ints, ONNXDecodingError, SymbolicGraphError, TensorId};
use crate::symbolic_graph::milli_ops::*;

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
    fn eval_old(&self, backend: &EvalBackend, inputs: &[&NumericTensor]) -> Result<Vec<NumericTensor>, EvalError> {unimplemented!()}

    fn eval(&self, backend: &EvalBackend, inputs: &HashMap<TensorId, NumericTensor>) -> Result<HashMap<TensorId, NumericTensor>, EvalError> {
        let milli_graph = self.get_milli_op_graph();
        Ok(milli_graph.eval(inputs, backend)?)
    }
    fn get_milli_op_graph(&self) -> MilliOpGraph;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) enum WhichBinaryOperation {
    Add,
    Sub,
    Mul,
    Div,
    MatMul
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
            WhichBinaryOperation::Add => AnyMilliOp::Add(MilliOpAdd::new(a, b)),
            WhichBinaryOperation::Sub => AnyMilliOp::Sub(MilliOpSub::new(a, b)),
            WhichBinaryOperation::Mul => AnyMilliOp::Mul(MilliOpMul::new(a, b)),
            WhichBinaryOperation::Div => AnyMilliOp::Div(MilliOpDiv::new(a, b)),
            WhichBinaryOperation::MatMul => AnyMilliOp::MatMul(MilliOpMatMul::new(a, b)),
        };
        let mut output_map = HashMap::new();
        let out = graph.push_op(res);
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) enum WhichUnaryOperation {
    Relu,
    Sigmoid,
    Tanh,
    Exp,
    Softplus,
    Neg,
    NonZero,
    Sqrt,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UnaryOperation {
    input: TensorId,
    output: TensorId,
    which: WhichUnaryOperation
}

impl UnaryOperation {
    pub(crate) fn from_onnx(inputs: Vec<TensorId>, outputs: Vec<TensorId>, which: WhichUnaryOperation) -> Result<Self, SymbolicGraphError> {
        if inputs.len() != 1 {
            return Err(SymbolicGraphError::InvalidOperatorInputs);
        }
        if outputs.len() != 1 {
            return Err(SymbolicGraphError::InvalidOperatorOutputs);
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
        let res = match self.which {
            WhichUnaryOperation::Relu => AnyMilliOp::ClampMin(MilliOpClampMin::new(a, 0.0)),
            WhichUnaryOperation::Sigmoid => {
                let x = graph.push_op(AnyMilliOp::Neg(MilliOpNeg::new(a)));
                let x = graph.push_op(AnyMilliOp::Exp(MilliOpExp::new(x)));
                let x = graph.push_op(AnyMilliOp::AddScalar(MilliOpAddScalar::new(x, 1.0)));
                AnyMilliOp::Reciprocal(MilliOpReciprocal::new(x))
            },
            WhichUnaryOperation::Tanh => AnyMilliOp::Tanh(MilliOpTanh::new(a)),
            WhichUnaryOperation::Exp => AnyMilliOp::Exp(MilliOpExp::new(a)),
            WhichUnaryOperation::Softplus => {
                let x = graph.push_op(AnyMilliOp::Exp(MilliOpExp::new(a)));
                let x = graph.push_op(AnyMilliOp::AddScalar(MilliOpAddScalar::new(x, 1.0)));
                AnyMilliOp::Log(MilliOpLog::new(x))
            }
            WhichUnaryOperation::Neg => AnyMilliOp::Neg(MilliOpNeg::new(a)),
            WhichUnaryOperation::NonZero => AnyMilliOp::NonZero(MilliOpNonZero::new(a)),
            WhichUnaryOperation::Sqrt => AnyMilliOp::Sqrt(MilliOpSqrt::new(a)),
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
    fn eval_old(&self, backend: &EvalBackend, inputs: &[&NumericTensor]) -> Result<Vec<NumericTensor>, EvalError> {
        let input = inputs[0];
        let axis = (if self.axis < 0 {input.shape().len() as i64 + self.axis} else {self.axis}) as i64;
        let x = input.abs(backend)?;
        let x = match self.p{
            1 => x,
            2 => NumericTensor::mul(&x, &x, backend)?,
            _ => Err(EvalError::InvalidInput("p must be either 1 or 2".to_string()))?
        };
        let x = x.reduce_sum(Some(vec![axis]), true, backend)?;
        let x = match self.p{
            1 => x,
            2 => x.sqrt(backend)?,
            _ => Err(EvalError::InvalidInput("p must be either 1 or 2".to_string()))?
        };
        let y = NumericTensor::div(input, &x, backend)?;

        Ok(vec![y])
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let input = input_map[&self.input];

        let x = graph.push_op(AnyMilliOp::Abs(MilliOpAbs::new(input)));

        let x = match self.p{
            1 => x,
            2 => graph.push_op(AnyMilliOp::Mul(MilliOpMul::new(input, input))),
            _ => panic!()
        };
        let axis_tensor = NDArrayNumericTensor::from(vec![self.axis]);
        let axis = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new(axis_tensor.into())));
        let x = graph.push_op(AnyMilliOp::ReduceSum(MilliOpReduceSum::new(x, Some(axis), true)));
        let x = match self.p{
            1 => x,
            2 => graph.push_op(AnyMilliOp::Sqrt(MilliOpSqrt::new(input))),
            _ => panic!()
        };
        let out = graph.push_op(AnyMilliOp::Div(MilliOpDiv::new(input, x)));

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
    fn eval_old(&self, backend: &EvalBackend, inputs: &[&NumericTensor]) -> Result<Vec<NumericTensor>, EvalError> {
        let input = inputs[0];
        let scale = inputs[1];
        let bias = inputs[2];

        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let num_channels = input_shape[1];
        let num_groups = self.num_groups;

        let hidden_size = input_shape[2..].iter().product::<usize>() * num_channels / num_groups;
        let input = input.reshape(vec![batch_size, num_groups, hidden_size], backend)?;

        let mean = NumericTensor::div_f32(&input.clone().reduce_sum(Some(vec![2]), true, backend)?, hidden_size as f32, backend)?;
        let input = NumericTensor::sub(&input, &mean, backend)?;

        let x = NumericTensor::mul(&input, &input, backend)?.reduce_sum(Some(vec![2]), true, backend)?;
        let var =  NumericTensor::div_f32(&x, hidden_size as f32, backend)?;
        let input_normalized = NumericTensor::div(&input, &NumericTensor::add_f32(&var, self.epsilon, backend)?.sqrt(backend)?, backend)?;

        let y = input_normalized.reshape(vec![batch_size, num_channels, input_shape[2..].iter().product::<usize>()], backend)?;
        
        let y = NumericTensor::mul(&y, &scale.unsqueeze(1, backend)?, backend)?;
        let y = NumericTensor::add(&y, &bias.unsqueeze(1, backend)?, backend)?;

        let y = y.reshape(input_shape, backend)?;
        Ok(vec![y])
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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

    fn eval_old(&self, backend: &EvalBackend, inputs: &[&NumericTensor]) -> Result<Vec<NumericTensor>, EvalError> {
        let axes_ndarray = NDArrayNumericTensor::try_from(inputs[1].cast(DType::I64, backend)?)?;
        let axes = Vec::<i64>::try_from(axes_ndarray)?;
        if axes.len() > 1 {
            return Err(EvalError::InvalidInput("Unsqueeze".to_string()));
        }
        let axis = axes[0];
        let input_shape = inputs[0].shape();
        let axis = if axis >= 0 {
            axis as usize
        } else {
            (input_shape.len() as i64 + axis) as usize
        };
        let output = inputs[0].squeeze(axis, backend)?;
        Ok(vec![output])
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        todo!()
    }
}


#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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

    fn eval_old(&self, backend: &EvalBackend, inputs: &[&NumericTensor]) -> Result<Vec<NumericTensor>, EvalError> {
        let axes_ndarray = NDArrayNumericTensor::try_from(inputs[1].cast(DType::I64, backend)?)?;
        let axes = Vec::<i64>::try_from(axes_ndarray)?;
        if axes.len() > 1 {
            return Err(EvalError::InvalidInput("Unsqueeze".to_string()));
        }
        let axis = axes[0];
        let input_shape = inputs[0].shape();
        let axis = if axis >= 0 {
            axis as usize
        } else {
            (input_shape.len() as i64 + axis) as usize
        };
        let output = inputs[0].unsqueeze(axis, backend)?;
        Ok(vec![output])
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
    fn eval_old(&self, backend: &EvalBackend, inputs: &[&NumericTensor]) -> Result<Vec<NumericTensor>, EvalError> {
        Ok(vec![inputs[0].transpose(self.perm.clone(), backend)?])
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
    fn eval_old(&self, backend: &EvalBackend, inputs: &[&NumericTensor]) -> Result<Vec<NumericTensor>, EvalError> {
        let data_input = inputs[0];
        let shape_input = inputs[1];
        let data_input_shape = data_input.shape();
        let shape_input_value: Vec<i64> = shape_input.cast(DType::I64, backend)?.try_into()?;
        let mut new_shape_dims = vec![];
        let mut backfill_dim: Option<usize> = None;
        for i in 0..shape_input_value.len() {
            new_shape_dims.push(if shape_input_value[i] == 0 {
                data_input_shape[i].clone()
            } else if shape_input_value[i] == -1 {
                if backfill_dim.is_some() {
                    // Only one dimension can be inferred
                    Err(EvalError::InvalidInput("Reshape".to_string()))?
                }
                backfill_dim = Some(i);
                1
            }
            else if shape_input_value[i] < -1 {
                Err(EvalError::InvalidInput("Reshape".to_string()))?
            } else {
                shape_input_value[i] as usize
            });
        }

        // Backfill the inferred dimension
        if let Some(i) = backfill_dim {
            let total_input_size = data_input.shape().iter().product::<usize>();

            // Calculate the current product of the dimensions
            let mut current_product = 1;
            for (j, dim) in new_shape_dims.iter().enumerate() {
                if j != i {
                    current_product *= dim;
                }
            }
            // Calculate the inferred dimension size
            let inferred_size = total_input_size / current_product;
            new_shape_dims[i] = inferred_size;
        }
        let output_shape = new_shape_dims;

        // Verify that the dimensions are compatible
        if output_shape.iter().product::<usize>() != data_input.shape().iter().product::<usize>() {
            Err(EvalError::InvalidInput("Reshape".to_string()))?
        }

        let output_value = data_input.reshape(output_shape, backend)?;

        Ok(vec![output_value])
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
    fn eval_old(&self, backend: &EvalBackend, inputs: &[&NumericTensor]) -> Result<Vec<NumericTensor>, EvalError> {
        Ok(vec![inputs[0].cast(self.to, backend)?])
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        todo!()
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
    epsilon: f32
}

impl LayerNormalizationOperation {
    pub(crate) fn from_onnx(inputs: &[TensorId], outputs: &[TensorId], attributes: &[onnx::AttributeProto]) -> Result<Self, ONNXDecodingError> {
        if inputs.len() < 2 || inputs.len() > 3  {
            return Err(ONNXDecodingError::GraphConstructionError("LayerNormalization".to_string(), SymbolicGraphError::InvalidOperatorInputs));
        }
        if outputs.len() != 1 {
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
        vec![self.output]
    }
    fn eval_old(&self, backend: &EvalBackend, inputs: &[&NumericTensor]) -> Result<Vec<NumericTensor>, EvalError> {
        let input = inputs[0];
        let scale = inputs[1];
        let bias = self.bias.map(|b| inputs[2]);

        let axis = if self.axis < 0 { (input.rank() as i64 + self.axis) as usize } else { self.axis as usize };

        let normalized_axes: Vec<_> = (axis..input.rank()).map(|x| x as i64).collect();

        let mean = input.reduce_mean(Some(normalized_axes.clone()), true, backend)?;
        let d = NumericTensor::sub(input, &mean, backend)?;
        let dd = NumericTensor::mul(&d, &d, backend)?;
        let var = dd.reduce_mean(Some(normalized_axes), true, backend)?;
        let var_eps = NumericTensor::add_f32(&var, self.epsilon, backend)?;
        let stddev = var_eps.sqrt(backend)?;
        let inv_stddev = stddev.reciprocal(backend)?;
        let normalized = NumericTensor::mul(&d, &inv_stddev, backend)?;

        let normalized_scaled = NumericTensor::mul(&normalized, scale, backend)?;
        let y = if let Some(bias) = bias {
            NumericTensor::add(&normalized_scaled, bias, backend)?
        } else {
            normalized_scaled
        };
        let mut outputs = vec![y];
        if self.mean_output.is_some() {
            outputs.push(mean);
        }
        if self.inv_std_dev_output.is_some() {
            outputs.push(inv_stddev);
        }
        Ok(outputs)
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
    fn eval_old(&self, backend: &EvalBackend, inputs: &[&NumericTensor]) -> Result<Vec<NumericTensor>, EvalError> {
        Ok(vec![NumericTensor::gather(inputs[0], inputs[1], self.axis, backend)?])
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
    fn eval_old(&self, _backend: &EvalBackend, inputs: &[&NumericTensor]) -> Result<Vec<NumericTensor>, EvalError> {
        let input_shape = inputs[0].shape();
        let mut output_shape = vec![];
        let start = {
            let mut start = 0;
            if let Some(x) = self.start {
                start = x;
            }
            if start < 0 {
                start = start + input_shape.len() as i64
            }
            if start > input_shape.len() as i64 {
                start = input_shape.len() as i64;
            }
            if start < 0 {
                start = 0;
            }
            start as usize
        };
        let end = {
            let mut end = input_shape.len() as i64;
            if let Some(x) = self.end {
                end = x;
            }
            if end < 0 {
                end = end + input_shape.len() as i64
            }
            if end > input_shape.len() as i64 {
                end = input_shape.len() as i64;
            }
            if end < 0 {
                end = 0;
            }
            end as usize
        };
        for i in 0..input_shape.len() {
            if i < start {
                continue;
            }
            if i >= end {
                continue;
            }
            output_shape.push(input_shape[i] as i64);
        }
        let shape_tensor = NDArrayNumericTensor::from(output_shape);
        Ok(vec![shape_tensor.into()])
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
    fn eval_old(&self, backend: &EvalBackend, inputs: &[&NumericTensor]) -> Result<Vec<NumericTensor>, EvalError> {
        let axis = if self.axis < 0 {
            inputs[0].shape().len() as i64 + self.axis
        } else {
            self.axis
        } as usize;
        Ok(vec![NumericTensor::concat(inputs, axis, backend)?])
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConstantOfShapeOperation {
    value: Option<NDArrayNumericTensor>,
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

        let mut value = None;
        for attr in attributes {
            if attr.name == "value" {
                if let Some(tensor_proto) = &attr.t {
                    value = Some(NDArrayNumericTensor::try_from(tensor_proto)?);
                }
            }
        }

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
    fn eval_old(&self, backend: &EvalBackend, inputs: &[&NumericTensor]) -> Result<Vec<NumericTensor>, EvalError> {
        let shape: Vec<i64> = inputs[0].cast(DType::I64, backend)?.try_into()?;
        let shape = shape.iter().map(|x| *x as usize).collect();
        let v = if let Some(value) = &self.value {
            match value.dtype() {
                DType::F64 => NDArrayNumericTensor::from_scalar_shape(Vec::<f64>::try_from(value.clone())?[0], shape)?,
                DType::F32 => NDArrayNumericTensor::from_scalar_shape(Vec::<f32>::try_from(value.clone())?[0], shape)?,
                DType::BF16 => NDArrayNumericTensor::from_scalar_shape(Vec::<bf16>::try_from(value.clone())?[0], shape)?,
                DType::F16 => NDArrayNumericTensor::from_scalar_shape(Vec::<f16>::try_from(value.clone())?[0], shape)?,
                DType::I64 => NDArrayNumericTensor::from_scalar_shape(Vec::<i64>::try_from(value.clone())?[0], shape)?,
                DType::I32 => NDArrayNumericTensor::from_scalar_shape(Vec::<i32>::try_from(value.clone())?[0], shape)?,
                DType::U64 => NDArrayNumericTensor::from_scalar_shape(Vec::<u64>::try_from(value.clone())?[0], shape)?,
                DType::U32 => NDArrayNumericTensor::from_scalar_shape(Vec::<u32>::try_from(value.clone())?[0], shape)?,
                DType::I16 => NDArrayNumericTensor::from_scalar_shape(Vec::<i16>::try_from(value.clone())?[0], shape)?,
                DType::U16 => NDArrayNumericTensor::from_scalar_shape(Vec::<u16>::try_from(value.clone())?[0], shape)?,
                DType::U8 => NDArrayNumericTensor::from_scalar_shape(Vec::<u8>::try_from(value.clone())?[0], shape)?,
                DType::I8 => NDArrayNumericTensor::from_scalar_shape(Vec::<i8>::try_from(value.clone())?[0], shape)?,
                DType::BOOL => NDArrayNumericTensor::from_scalar_shape(Vec::<bool>::try_from(value.clone())?[0], shape)?,
            }
        } else {
            NDArrayNumericTensor::from_scalar_shape(0f32, shape)?
        };
        Ok(vec![v.into()])
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReduceMeanOperation {
    keepdims: Option<i64>,
    noop_with_empty_axes: Option<i64>,
    input_data: TensorId,
    input_axes: Option<TensorId>,
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

        let keepdims = query_attribute_int(attributes, "keepdims");
        let noop_with_empty_axes = query_attribute_int(attributes, "noop_with_empty_axes");


        Ok(
            Self {
                keepdims,
                noop_with_empty_axes,
                input_data: inputs[0],
                input_axes: if inputs.len() > 1 {Some(inputs[1])} else {None},
                output: outputs[0]
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
    fn eval_old(&self, backend: &EvalBackend, inputs: &[&NumericTensor]) -> Result<Vec<NumericTensor>, EvalError> {
        let data_input = inputs[0];
        let axes = if self.input_axes.is_some() {
            Vec::<i64>::try_from(inputs[1].clone())?
        } else {
            (0i64 .. (data_input.shape().len() as i64)).into_iter().collect()
        };
        let keepdims = self.keepdims.unwrap_or(1);
        Ok(vec![inputs[0].reduce_mean(Some(axes), keepdims != 0, backend)?])
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReduceSumOperation {
    keepdims: Option<i64>,
    noop_with_empty_axes: Option<i64>,
    input_data: TensorId,
    input_axes: Option<TensorId>,
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

        let keepdims = query_attribute_int(attributes, "keepdims");
        let noop_with_empty_axes = query_attribute_int(attributes, "noop_with_empty_axes");

        Ok(
            Self {
                keepdims,
                noop_with_empty_axes,
                input_data: inputs[0],
                input_axes: if inputs.len() > 1 {Some(inputs[1])} else {None},
                output: outputs[0]
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

    fn eval_old(&self, backend: &EvalBackend, inputs: &[&NumericTensor]) -> Result<Vec<NumericTensor>, EvalError> {
        let data_input = inputs[0];
        let axes = if self.input_axes.is_some() {
            Vec::<i64>::try_from(inputs[1].clone())?
        } else {
            (0i64 .. (data_input.shape().len() as i64)).into_iter().collect()
        };
        let keepdims = self.keepdims.unwrap_or(1);
        Ok(vec![inputs[0].reduce_sum(Some(axes), keepdims != 0, backend)?])
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        todo!()
    }
}


#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
    fn eval_old(&self, backend: &EvalBackend, inputs: &[&NumericTensor]) -> Result<Vec<NumericTensor>, EvalError> {
        Ok(vec![inputs[0].pow(inputs[1], backend)?])
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        todo!()
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
    fn eval_old(&self, backend: &EvalBackend, inputs: &[&NumericTensor]) -> Result<Vec<NumericTensor>, EvalError> {
        Ok(vec![NumericTensor::gemm(inputs[0], inputs[1], if inputs.len() > 2 {Some(inputs[2])} else {None},
                            self.alpha.unwrap_or(1.0), self.beta.unwrap_or(1.0),
                            self.trans_a.unwrap_or(false), self.trans_b.unwrap_or(false), backend)?])
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
    fn eval_old(&self, backend: &EvalBackend, inputs: &[&NumericTensor]) -> Result<Vec<NumericTensor>, EvalError> {
        let split: Vec<i64> = if let Some(split) = self.split_attribute.clone() {
            split
        }
        else if let Some(_) = self.split {
            inputs[1].clone().try_into()?
        } else {
            Err(EvalError::InvalidInput("Split attribute is not set, and we do not support num_outputs yet".to_string()))?
        };
        Ok(inputs[0].split(&split, self.axis.unwrap_or_default(), backend)?)
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
    fn eval_old(&self, backend: &EvalBackend, inputs: &[&NumericTensor]) -> Result<Vec<NumericTensor>, EvalError> {
        let data_input = inputs[0];
        let axes: Vec<i64> = if self.axes.is_some() {
            inputs[3].cast(DType::I64, backend)?.try_into()?
        } else {
            (0i64..(data_input.shape().len() as i64)).into_iter().collect()
        };
        let steps: Vec<i64> = if self.steps.is_some() {
            inputs[4].cast(DType::I64, backend)?.try_into()?
        } else {
            axes.iter().map(|_| 1).collect()
        };
        let starts: Vec<i64> = inputs[1].cast(DType::I64, backend)?.try_into()?;
        let ends: Vec<i64> = inputs[2].cast(DType::I64, backend)?.try_into()?;
        let mut output_slice = vec![];
        for i in 0..data_input.shape().len() {
            output_slice.push(0..data_input.shape()[i]);
        }
        for (i, axis) in axes.into_iter().enumerate() {
            let axis = if axis < 0 {
                (data_input.shape().len() as i64 + axis) as usize
            } else {
                axis as usize
            };
            let step = steps[i];
            if step != 1 {
                return Err(EvalError::InvalidInput(format!("Step {} is not supported", step)));
            }
            let start = (if starts[i] < 0 { data_input.shape()[axis] as i64 + starts[i] } else { starts[i] }) as usize;
            let end = (if ends[i] < 0 { data_input.shape()[axis] as i64 + ends[i] } else { ends[i] }) as usize;
            let start = start.min(data_input.shape()[axis]);
            let end = end.min(data_input.shape()[axis]);
            output_slice[axis] = start..end;
        }
        let output = data_input.slice(&output_slice, backend)?;
        Ok(vec![output])
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
    fn eval_old(&self, backend: &EvalBackend, inputs: &[&NumericTensor]) -> Result<Vec<NumericTensor>, EvalError> {
        let conditional = inputs[0];
        let a = inputs[1];
        let b = inputs[2];
        Ok(vec![conditional.where_op(a, b, backend)?])
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
    fn eval_old(&self, backend: &EvalBackend, inputs: &[&NumericTensor]) -> Result<Vec<NumericTensor>, EvalError> {
        let e = inputs[0].exp(backend)?;
        let axis = if let Some(axis) = self.axis {
            axis
        } else {
             -1
        };
        let r = e.reduce_sum(Some(vec![axis]), true, backend)?;
        let o = NumericTensor::div(&e, &r, backend)?;
        Ok(vec![o])
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AnyOperation {
    Unary(UnaryOperation),
    Binary(BinaryOperation),
    Cast(CastOperation),
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
    Pow(PowOperation),
    Gemm(GemmOperation),
    Split(SplitOperation),
    Slice(SliceOperation),
    Where(WhereOperation),
    Softmax(SoftmaxOperation)
}

impl AnyOperation {
    fn as_dyn(&self) -> &dyn Operation {
        match self {
            AnyOperation::Unary(op) => op,
            AnyOperation::Binary(op) => op,
            AnyOperation::Cast(op) => op,
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
            AnyOperation::Pow(op) => op,
            AnyOperation::Gemm(op) => op,
            AnyOperation::Split(op) => op,
            AnyOperation::Slice(op) => op,
            AnyOperation::Where(op) => op,
            AnyOperation::Softmax(op) => op
        }
    }
}

impl Operation for AnyOperation {
    fn get_inputs(&self) -> Vec<TensorId> {
        match self {
            AnyOperation::Unary(op) => op.get_inputs(),
            AnyOperation::Binary(op) => op.get_inputs(),
            AnyOperation::Cast(op) => op.get_inputs(),
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
            AnyOperation::Pow(op) => op.get_inputs(),
            AnyOperation::Gemm(op) => op.get_inputs(),
            AnyOperation::Split(op) => op.get_inputs(),
            AnyOperation::Slice(op) => op.get_inputs(),
            AnyOperation::Where(op) => op.get_inputs(),
            AnyOperation::Softmax(op) => op.get_inputs()
        }
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        match self {
            AnyOperation::Unary(op) => op.get_outputs(),
            AnyOperation::Binary(op) => op.get_outputs(),
            AnyOperation::Cast(op) => op.get_outputs(),
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
            AnyOperation::Pow(op) => op.get_outputs(),
            AnyOperation::Gemm(op) => op.get_outputs(),
            AnyOperation::Split(op) => op.get_outputs(),
            AnyOperation::Slice(op) => op.get_outputs(),
            AnyOperation::Where(op) => op.get_outputs(),
            AnyOperation::Softmax(op) => op.get_outputs()
        }
    }
    fn eval_old(&self, backend: &EvalBackend, inputs: &[&NumericTensor]) -> Result<Vec<NumericTensor>, EvalError> {
        self.as_dyn().eval_old(backend, inputs)
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph {
        match self {
            AnyOperation::Unary(op) => op.get_milli_op_graph(),
            AnyOperation::Binary(op) => op.get_milli_op_graph(),
            AnyOperation::Cast(op) => op.get_milli_op_graph(),
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
            AnyOperation::Pow(op) => op.get_milli_op_graph(),
            AnyOperation::Gemm(op) => op.get_milli_op_graph(),
            AnyOperation::Split(op) => op.get_milli_op_graph(),
            AnyOperation::Slice(op) => op.get_milli_op_graph(),
            AnyOperation::Where(op) => op.get_milli_op_graph(),
            AnyOperation::Softmax(op) => op.get_milli_op_graph()
        }
    }
}