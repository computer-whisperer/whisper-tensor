use crate::graph::{GlobalId, Node};
use crate::milli_graph::{self, MilliOpGraph, ops_helpers};
use crate::onnx;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::ONNXDecodingError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rand::Rng;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShapeOperation {
    global_id: GlobalId,
    start: Option<i64>,
    end: Option<i64>,
    input: GlobalId,
    output: GlobalId,
}

impl ShapeOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
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
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Shape"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Shape"))?,
            start,
            end,
        })
    }
}

impl Node for ShapeOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Shape".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}
impl Operation for ShapeOperation {
    fn get_milli_op_graph(&self, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let out = milli_graph::ops::Shape::push_new(&mut graph, input_map[&self.input], rng);
        let out = if self.start.is_some() || self.end.is_some() {
            let start = ops_helpers::scalar_const(&mut graph, self.start.unwrap_or(0), rng);
            let end = if let Some(end) = self.end {
                ops_helpers::scalar_const(&mut graph, end, rng)
            } else {
                milli_graph::ops::Shape::push_new(&mut graph, out, rng)
            };
            milli_graph::ops::Slice::push_new(&mut graph, out, start, end, None, None, rng)
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
    global_id: GlobalId,
    input: GlobalId,
    output: GlobalId,
}

impl SizeOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Size"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Size"));
        }
        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Size"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Size"))?,
        })
    }
}

impl Node for SizeOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Size".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for SizeOperation {
    fn get_milli_op_graph(&self, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);

        let shape_tid = milli_graph::ops::Shape::push_new(&mut graph, input_map[&self.input], rng);
        let size_tid =
            milli_graph::ops::ReduceProd::push_new(&mut graph, shape_tid, None, false, false, rng);

        let mut output_map = HashMap::new();
        output_map.insert(size_tid, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
