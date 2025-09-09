use crate::milli_graph::ops::*;
use crate::milli_graph::{MilliOpGraph, ops_helpers};
use crate::onnx;
use crate::symbolic_graph::ops::Operation;
use crate::graph::{Graph, Node, InnerGraph};
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

impl Operation for ShapeOperation {
    fn get_op_type_name(&self) -> String {
        "Shape".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.input]
    }
    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let out = Shape::new(&mut graph, input_map[&self.input]);
        let out = if self.start.is_some() || self.end.is_some() {
            let start = ops_helpers::scalar_const(&mut graph, self.start.unwrap_or(0));
            let end = if let Some(end) = self.end {
                ops_helpers::scalar_const(&mut graph, end)
            } else {
                Shape::new(&mut graph, out)
            };
            Slice::new(
                &mut graph, out, start, end, None, None,
            )
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

impl Operation for SizeOperation {
    fn get_op_type_name(&self) -> String {
        "Size".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.input]
    }
    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());

        let shape_tid = Shape::new(&mut graph, input_map[&self.input]);
        let size_node = ReduceProd::new(
            &mut graph, shape_tid, None, false, false,
        );
        let size_tid = match graph.inner(&()).get_node(&size_node) {
            Some(AnyMilliOp::ReduceProd(op)) => op.outputs().next().unwrap(),
            _ => unreachable!(),
        };

        let mut output_map = HashMap::new();
        output_map.insert(size_tid, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
