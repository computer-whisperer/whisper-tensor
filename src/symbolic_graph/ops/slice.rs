use crate::graph::{GlobalId, Node};
use crate::milli_graph::{self, MilliOpGraph};
use crate::onnx;
use crate::symbolic_graph::ONNXDecodingError;
use crate::symbolic_graph::ops::Operation;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SliceOperation {
    global_id: GlobalId,
    data: GlobalId,
    starts: GlobalId,
    ends: GlobalId,
    axes: Option<GlobalId>,
    steps: Option<GlobalId>,
    output: GlobalId,
}

impl SliceOperation {
    pub fn new_from_parts(
        data: GlobalId,
        starts: GlobalId,
        ends: GlobalId,
        axes: Option<GlobalId>,
        steps: Option<GlobalId>,
        output: GlobalId,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            data,
            starts,
            ends,
            axes,
            steps,
            output,
        }
    }

    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        _attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() < 3 || inputs.len() > 5 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Slice"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Slice"));
        }

        Ok(Self {
            global_id: GlobalId::new(rng),
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

impl Node for SliceOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Slice".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        let mut v = vec![self.data, self.starts, self.ends];
        if let Some(axes) = &self.axes {
            v.push(*axes);
        }
        if let Some(steps) = &self.steps {
            v.push(*steps);
        }
        Box::new(v.into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}
impl Operation for SliceOperation {
    fn get_milli_op_graph(&self, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let out = milli_graph::ops::Slice::push_new(
            &mut graph,
            input_map[&self.data],
            input_map[&self.starts],
            input_map[&self.ends],
            self.steps.map(|x| input_map[&x]),
            self.axes.map(|x| input_map[&x]),
            rng,
        );
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
