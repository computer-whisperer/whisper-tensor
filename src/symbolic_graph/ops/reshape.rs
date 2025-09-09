use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::graph::Node;
use crate::milli_graph::{self, MilliOpGraph};
use crate::onnx;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{
    ONNXDecodingError, SymbolicGraphTensorId, query_attribute_int, query_attribute_ints,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SqueezeOperation {
    input: SymbolicGraphTensorId,
    axes: Option<SymbolicGraphTensorId>,
    axes_attribute: Option<Vec<i64>>,
    output: SymbolicGraphTensorId,
}

impl SqueezeOperation {
    pub fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
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

impl Node<SymbolicGraphTensorId> for SqueezeOperation {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "Squeeze".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        if let Some(axes) = self.axes {
            Box::new([self.input, axes].into_iter())
        } else {
            Box::new(std::iter::once(self.input))
        }
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new(std::iter::once(self.output))
    }
}
impl Operation for SqueezeOperation {
    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs());
        let axes_input = if let Some(axes) = self.axes {
            input_map[&axes]
        } else if let Some(axes) = &self.axes_attribute {
            let axes_tensor = NDArrayNumericTensor::from_vec(axes.clone());
            milli_graph::ops::Constant::push_new(&mut graph, axes_tensor.to_dyn())
        } else {
            panic!();
        };

        let out =
            milli_graph::ops::Squeeze::push_new(&mut graph, input_map[&self.input], axes_input);
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UnsqueezeOperation {
    input: SymbolicGraphTensorId,
    axes: Option<SymbolicGraphTensorId>,
    axes_attribute: Option<Vec<i64>>,
    output: SymbolicGraphTensorId,
}

impl UnsqueezeOperation {
    pub fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
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

impl Node<SymbolicGraphTensorId> for UnsqueezeOperation {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "Unsqueeze".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        if let Some(axes) = self.axes {
            Box::new([self.input, axes].into_iter())
        } else {
            Box::new(std::iter::once(self.input))
        }
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new(std::iter::once(self.output))
    }
}
impl Operation for UnsqueezeOperation {
    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs());
        let axes_input = if let Some(axes) = self.axes {
            input_map[&axes]
        } else if let Some(axes) = &self.axes_attribute {
            let axes_tensor = NDArrayNumericTensor::from_vec(axes.clone());
            milli_graph::ops::Constant::push_new(&mut graph, axes_tensor.to_dyn())
        } else {
            panic!();
        };

        let out =
            milli_graph::ops::Unsqueeze::push_new(&mut graph, input_map[&self.input], axes_input);
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReshapeOperation {
    input: SymbolicGraphTensorId,
    shape: SymbolicGraphTensorId,
    output: SymbolicGraphTensorId,
}

impl ReshapeOperation {
    pub fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
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

impl Node<SymbolicGraphTensorId> for ReshapeOperation {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "Reshape".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new([self.input, self.shape].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for ReshapeOperation {
    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs());
        let out = milli_graph::ops::Reshape::push_new(
            &mut graph,
            input_map[&self.input],
            input_map[&self.shape],
            false,
        );
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FlattenOperation {
    input: SymbolicGraphTensorId,
    output: SymbolicGraphTensorId,
    axis: i64,
}

impl FlattenOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
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

impl Node<SymbolicGraphTensorId> for FlattenOperation {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "Flatten".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for FlattenOperation {
    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs());
        let input = input_map[&self.input];

        let shape_tensor = if self.axis == 0 {
            let output_shape = vec![1i64, -1i64];
            let shape_tensor = NDArrayNumericTensor::from(output_shape);
            milli_graph::ops::Constant::push_new(&mut graph, shape_tensor.to_dyn())
        } else {
            let input_shape = milli_graph::ops::Shape::push_new(&mut graph, input);
            let zero_const = milli_graph::ops::Constant::push_new(
                &mut graph,
                NDArrayNumericTensor::from_vec_shape(vec![0i64], &vec![1]).unwrap(),
            );
            let axis_const = milli_graph::ops::Constant::push_new(
                &mut graph,
                NDArrayNumericTensor::from_vec_shape(vec![self.axis], &vec![1]).unwrap(),
            );
            let first_dims = milli_graph::ops::Slice::push_new(
                &mut graph,
                input_shape,
                zero_const,
                axis_const,
                None,
                None,
            );
            let prod =
                milli_graph::ops::ReduceProd::push_new(&mut graph, first_dims, None, true, false);
            let neg_one_const = milli_graph::ops::Constant::push_new(
                &mut graph,
                NDArrayNumericTensor::from_vec_shape(vec![-1i64], &vec![1]).unwrap(),
            );
            milli_graph::ops::Concat::push_new(&mut graph, vec![prod, neg_one_const], 0)
        };

        let out = milli_graph::ops::Reshape::push_new(&mut graph, input, shape_tensor, false);

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
