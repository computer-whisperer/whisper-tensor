use serde::{Deserialize, Serialize};
use crate::dtype::DType;
use crate::native_numeric_tensor::NativeNumericTensor;
use crate::onnx;
use crate::symbolic_graph::{query_attribute_bool, query_attribute_float, query_attribute_int, ONNXDecodingError, SymbolicGraphError, TensorId};

pub trait Operation {
    fn get_inputs(&self) -> Vec<TensorId>;
    fn get_outputs(&self) -> Vec<TensorId>;
    fn exec_native(&self, inputs: &[&NativeNumericTensor]) -> Vec<NativeNumericTensor>;
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

    fn exec_native(&self, inputs: &[&NativeNumericTensor]) -> Vec<NativeNumericTensor> {
        match self.which {
            WhichBinaryOperation::Add => {}
            _ => todo!()
        }
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

    fn exec_native(&self, inputs: &[&NativeNumericTensor]) -> Vec<NativeNumericTensor> {
        match self.which {
            _ => todo!()
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CumSumOperation {
    input: TensorId,
    output: TensorId,
    axis: TensorId,
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

    fn exec_native(&self, inputs: &[&NativeNumericTensor]) -> Vec<NativeNumericTensor> {
        todo!()
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
    fn exec_native(&self, inputs: &[&NativeNumericTensor]) -> Vec<NativeNumericTensor> {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GroupNormalizationOperation {
    input: TensorId,
    scale: TensorId,
    bias: TensorId,
    output: TensorId,
    epsilon: f32,
    num_groups: i64
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
        let num_groups = num_groups.ok_or(ONNXDecodingError::MissingAttribute("GroupNormalization".to_string(), "num_groups".to_string()))?;
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
    fn exec_native(&self, inputs: &[&NativeNumericTensor]) -> Vec<NativeNumericTensor> {
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

    fn exec_native(&self, inputs: &[&NativeNumericTensor]) -> Vec<NativeNumericTensor> {
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

    fn exec_native(&self, inputs: &[&NativeNumericTensor]) -> Vec<NativeNumericTensor> {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransposeOperation {
    input: TensorId,
    output: TensorId,
    perm: Vec<i64>,
}

impl TransposeOperation {
    pub fn from_onnx(inputs: Vec<TensorId>, outputs: Vec<TensorId>, attributes: &[onnx::AttributeProto]) -> Result<Self, SymbolicGraphError> {
        if inputs.len() != 1 {
            return Err(SymbolicGraphError::InvalidOperatorInputs);
        }
        if outputs.len() != 1 {
            return Err(SymbolicGraphError::InvalidOperatorOutputs);
        }
        let mut perm = vec![];
        for attr in attributes{
            match attr.name.as_str() {
                "perm" => {perm = attr.ints.clone()},
                _ => {}
            }
        }
        Ok(Self {
            input: inputs[0],
            output: outputs[0],
            perm
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
    fn exec_native(&self, inputs: &[&NativeNumericTensor]) -> Vec<NativeNumericTensor> {
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
    fn exec_native(&self, inputs: &[&NativeNumericTensor]) -> Vec<NativeNumericTensor> {
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
    fn exec_native(&self, inputs: &[&NativeNumericTensor]) -> Vec<NativeNumericTensor> {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerNormalizationOperation {
    input: TensorId,
    scale: TensorId,
    bias: Option<TensorId>,
    output: TensorId,
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
    fn exec_native(&self, inputs: &[&NativeNumericTensor]) -> Vec<NativeNumericTensor> {
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
    fn exec_native(&self, inputs: &[&NativeNumericTensor]) -> Vec<NativeNumericTensor> {
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
    fn exec_native(&self, inputs: &[&NativeNumericTensor]) -> Vec<NativeNumericTensor> {
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
    fn exec_native(&self, inputs: &[&NativeNumericTensor]) -> Vec<NativeNumericTensor> {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConstantOfShapeOperation {
    value: Option<NativeNumericTensor>,
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
                    value = Some(NativeNumericTensor::try_from(tensor_proto)?);
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
    fn exec_native(&self, inputs: &[&NativeNumericTensor]) -> Vec<NativeNumericTensor> {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReduceMeanOperation {
    keepdims: i64,
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

        let mut keepdims = 0;
        let mut noop_with_empty_axes = None;
        for attr in attributes {
            match attr.name.as_str() {
                "keepdims" => {
                    keepdims = attr.i;
                }
                "noop_with_empty_axes" => {
                    noop_with_empty_axes = Some(attr.i)
                }
                _ => {}
            }
        }

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
    fn exec_native(&self, inputs: &[&NativeNumericTensor]) -> Vec<NativeNumericTensor> {
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
    pub(crate) fn from_onnx(inputs: &[TensorId], outputs: &[TensorId], attributes: &[onnx::AttributeProto]) -> Result<Self, ONNXDecodingError> {
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
    fn exec_native(&self, inputs: &[&NativeNumericTensor]) -> Vec<NativeNumericTensor> {
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
    fn exec_native(&self, inputs: &[&NativeNumericTensor]) -> Vec<NativeNumericTensor> {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SplitOperation {
    axis: Option<i64>,
    num_outputs: i64,
    input: TensorId,
    split: Option<TensorId>,
    outputs: Vec<TensorId>
}

impl SplitOperation {
    pub(crate) fn from_onnx(inputs: &[TensorId], outputs: &[TensorId], attributes: &[onnx::AttributeProto]) -> Result<Self, ONNXDecodingError> {
        if inputs.len() < 1 || inputs.len() > 2  {
            return Err(ONNXDecodingError::GraphConstructionError("Split".to_string(), SymbolicGraphError::InvalidOperatorInputs));
        }

        let axis = query_attribute_int(attributes, "axis");
        let num_outputs = query_attribute_int(attributes, "num_outputs")
            .ok_or(ONNXDecodingError::MissingAttribute("Split".to_string(), "num_outputs".to_string()))?;

        Ok(Self{
            input: inputs[0],
            split: if inputs.len() > 1 {Some(inputs[1])} else {None},
            outputs: outputs.to_vec(),
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
    fn exec_native(&self, inputs: &[&NativeNumericTensor]) -> Vec<NativeNumericTensor> {
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
    pub(crate) fn from_onnx(inputs: &[TensorId], outputs: &[TensorId], attributes: &[onnx::AttributeProto]) -> Result<Self, ONNXDecodingError> {
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
    fn exec_native(&self, inputs: &[&NativeNumericTensor]) -> Vec<NativeNumericTensor> {
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
    pub(crate) fn from_onnx(inputs: &[TensorId], outputs: &[TensorId], attributes: &[onnx::AttributeProto]) -> Result<Self, ONNXDecodingError> {
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
    fn exec_native(&self, inputs: &[&NativeNumericTensor]) -> Vec<NativeNumericTensor> {
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
    fn exec_native(&self, inputs: &[&NativeNumericTensor]) -> Vec<NativeNumericTensor> {
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
        self.as_dyn().get_inputs()
    }
    fn get_outputs(&self) -> Vec<TensorId> {
        self.as_dyn().get_outputs()
    }
    fn exec_native(&self, inputs: &[&NativeNumericTensor]) -> Vec<NativeNumericTensor> {
        self.as_dyn().exec_native(inputs)
    }
}