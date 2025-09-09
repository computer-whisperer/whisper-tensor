use crate::graph::Node;
use crate::milli_graph::{self, MilliOpGraph};
use crate::onnx;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{ONNXDecodingError, SymbolicGraphTensorId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SliceOperation {
    data: SymbolicGraphTensorId,
    starts: SymbolicGraphTensorId,
    ends: SymbolicGraphTensorId,
    axes: Option<SymbolicGraphTensorId>,
    steps: Option<SymbolicGraphTensorId>,
    output: SymbolicGraphTensorId,
}

impl SliceOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
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

impl Node<SymbolicGraphTensorId> for SliceOperation {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "Slice".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        let mut v = vec![self.data, self.starts, self.ends];
        if let Some(axes) = &self.axes {
            v.push(*axes);
        }
        if let Some(steps) = &self.steps {
            v.push(*steps);
        }
        Box::new(v.into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = SymbolicGraphTensorId>> {
        Box::new(std::iter::once(self.output))
    }
}
impl Operation for SliceOperation {
    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs());
        let out = milli_graph::ops::Slice::push_new(
            &mut graph,
            input_map[&self.data],
            input_map[&self.starts],
            input_map[&self.ends],
            self.steps.map(|x| input_map[&x]),
            self.axes.map(|x| input_map[&x]),
        );
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
