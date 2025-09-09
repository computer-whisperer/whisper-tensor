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
pub struct CumSumOperation {
    input: SymbolicGraphTensorId,
    output: SymbolicGraphTensorId,
    axis: SymbolicGraphTensorId,
    exclusive: bool,
    reverse: bool,
}

impl CumSumOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("CumSum"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("CumSum"));
        }
        let exclusive = query_attribute_int(attributes, "exclusive").unwrap_or_default() != 0;
        let reverse = query_attribute_int(attributes, "reverse").unwrap_or_default() != 0;
        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Unary"))?,
            axis: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Unary"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Unary"))?,
            exclusive,
            reverse,
        })
    }
}

impl Node<SymbolicGraphTensorId> for CumSumOperation {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "CumSum".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new([self.input, self.axis].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for CumSumOperation {
    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs());
        let a = input_map[&self.input];
        let b = input_map[&self.axis];

        let out =
            milli_graph::ops::CumSum::push_new(&mut graph, a, b, self.exclusive, self.reverse);

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
    input_data: SymbolicGraphTensorId,
    input_axes: Option<SymbolicGraphTensorId>,
    axes_attr: Option<Vec<i64>>,
    output: SymbolicGraphTensorId,
}

impl ReduceMeanOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
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

impl Node<SymbolicGraphTensorId> for ReduceMeanOperation {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "ReduceMean".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        if let Some(input_axes) = self.input_axes {
            Box::new([self.input_data, input_axes].into_iter())
        } else {
            Box::new(std::iter::once(self.input_data))
        }
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new(std::iter::once(self.output))
    }
}
impl Operation for ReduceMeanOperation {
    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs());
        let axes = if let Some(input_axes) = &self.input_axes {
            Some(input_map[input_axes])
        } else if let Some(axes) = &self.axes_attr {
            let tensor = NDArrayNumericTensor::from(axes.clone());
            let tid = milli_graph::ops::Constant::push_new(&mut graph, tensor.to_dyn());
            Some(tid)
        } else {
            None
        };
        let out = milli_graph::ops::ReduceMean::push_new(
            &mut graph,
            input_map[&self.input_data],
            axes,
            self.keepdims.unwrap_or(true),
            self.noop_with_empty_axes.unwrap_or(false),
        );
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
    input_data: SymbolicGraphTensorId,
    input_axes: Option<SymbolicGraphTensorId>,
    axes_attr: Option<Vec<i64>>,
    output: SymbolicGraphTensorId,
}

impl ReduceSumOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
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

impl Node<SymbolicGraphTensorId> for ReduceSumOperation {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "ReduceSum".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        if let Some(input_axes) = self.input_axes {
            Box::new([self.input_data, input_axes].into_iter())
        } else {
            Box::new(std::iter::once(self.input_data))
        }
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new(std::iter::once(self.output))
    }
}
impl Operation for ReduceSumOperation {
    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs());
        let axes = if let Some(input_axes) = &self.input_axes {
            Some(input_map[input_axes])
        } else if let Some(axes) = &self.axes_attr {
            let tensor = NDArrayNumericTensor::from(axes.clone());
            let tid = milli_graph::ops::Constant::push_new(&mut graph, tensor.to_dyn());
            Some(tid)
        } else {
            None
        };
        let out = milli_graph::ops::ReduceSum::push_new(
            &mut graph,
            input_map[&self.input_data],
            axes,
            self.keepdims.unwrap_or(true),
            self.noop_with_empty_axes.unwrap_or(false),
        );
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
    input_data: SymbolicGraphTensorId,
    input_axes: Option<SymbolicGraphTensorId>,
    axes_attr: Option<Vec<i64>>,
    output: SymbolicGraphTensorId,
}

impl ReduceMaxOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
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

impl Node<SymbolicGraphTensorId> for ReduceMaxOperation {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "ReduceMax".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        if let Some(input_axes) = self.input_axes {
            Box::new([self.input_data, input_axes].into_iter())
        } else {
            Box::new(std::iter::once(self.input_data))
        }
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new(std::iter::once(self.output))
    }
}
impl Operation for ReduceMaxOperation {
    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs());
        let axes = if let Some(input_axes) = &self.input_axes {
            Some(input_map[input_axes])
        } else if let Some(axes) = &self.axes_attr {
            let tensor = NDArrayNumericTensor::from(axes.clone());
            let tid = milli_graph::ops::Constant::push_new(&mut graph, tensor.to_dyn());
            Some(tid)
        } else {
            None
        };
        let out = milli_graph::ops::ReduceMax::push_new(
            &mut graph,
            input_map[&self.input_data],
            axes,
            self.keepdims.unwrap_or(true),
            self.noop_with_empty_axes.unwrap_or(false),
        );
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
    input_data: SymbolicGraphTensorId,
    input_axes: Option<SymbolicGraphTensorId>,
    axes_attr: Option<Vec<i64>>,
    output: SymbolicGraphTensorId,
}

impl ReduceMinOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
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

impl Node<SymbolicGraphTensorId> for ReduceMinOperation {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "ReduceMin".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        if let Some(input_axes) = self.input_axes {
            Box::new([self.input_data, input_axes].into_iter())
        } else {
            Box::new(std::iter::once(self.input_data))
        }
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for ReduceMinOperation {
    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs());
        let axes = if let Some(input_axes) = &self.input_axes {
            Some(input_map[input_axes])
        } else if let Some(axes) = &self.axes_attr {
            let tensor = NDArrayNumericTensor::from(axes.clone());
            let tid = milli_graph::ops::Constant::push_new(&mut graph, tensor.to_dyn());
            Some(tid)
        } else {
            None
        };
        let out = milli_graph::ops::ReduceMin::push_new(
            &mut graph,
            input_map[&self.input_data],
            axes,
            self.keepdims.unwrap_or(true),
            self.noop_with_empty_axes.unwrap_or(false),
        );
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
    input_data: SymbolicGraphTensorId,
    input_axes: Option<SymbolicGraphTensorId>,
    axes_attr: Option<Vec<i64>>,
    output: SymbolicGraphTensorId,
}

impl ReduceProdOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
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

impl Node<SymbolicGraphTensorId> for ReduceProdOperation {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "ReduceProd".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        if let Some(input_axes) = self.input_axes {
            Box::new([self.input_data, input_axes].into_iter())
        } else {
            Box::new(std::iter::once(self.input_data))
        }
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for ReduceProdOperation {
    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs());
        let axes = if let Some(input_axes) = &self.input_axes {
            Some(input_map[input_axes])
        } else if let Some(axes) = &self.axes_attr {
            let tensor = NDArrayNumericTensor::from(axes.clone());
            let tid = milli_graph::ops::Constant::push_new(&mut graph, tensor.to_dyn());
            Some(tid)
        } else {
            None
        };
        let out = milli_graph::ops::ReduceProd::push_new(
            &mut graph,
            input_map[&self.input_data],
            axes,
            self.keepdims.unwrap_or(true),
            self.noop_with_empty_axes.unwrap_or(false),
        );
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
