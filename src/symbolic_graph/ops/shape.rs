use crate::graph::Node;
use crate::milli_graph::{self, MilliOpGraph, ops_helpers};
use crate::onnx;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{ONNXDecodingError, SymbolicGraphTensorId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShapeOperation {
    start: Option<i64>,
    end: Option<i64>,
    input: SymbolicGraphTensorId,
    output: SymbolicGraphTensorId,
}

impl ShapeOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
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

impl Node<SymbolicGraphTensorId> for ShapeOperation {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "Shape".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new(std::iter::once(self.output))
    }
}
impl Operation for ShapeOperation {
    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs());
        let out = milli_graph::ops::Shape::push_new(&mut graph, input_map[&self.input]);
        let out = if self.start.is_some() || self.end.is_some() {
            let start = ops_helpers::scalar_const(&mut graph, self.start.unwrap_or(0));
            let end = if let Some(end) = self.end {
                ops_helpers::scalar_const(&mut graph, end)
            } else {
                milli_graph::ops::Shape::push_new(&mut graph, out)
            };
            milli_graph::ops::Slice::push_new(&mut graph, out, start, end, None, None)
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
pub struct SizeOperation {
    input: SymbolicGraphTensorId,
    output: SymbolicGraphTensorId,
}

impl SizeOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
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

impl Node<SymbolicGraphTensorId> for SizeOperation {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "Size".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for SizeOperation {
    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs());

        let shape_tid = milli_graph::ops::Shape::push_new(&mut graph, input_map[&self.input]);
        let size_tid =
            milli_graph::ops::ReduceProd::push_new(&mut graph, shape_tid, None, false, false);

        let mut output_map = HashMap::new();
        output_map.insert(size_tid, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
