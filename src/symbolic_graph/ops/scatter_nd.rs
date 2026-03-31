use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::{MilliLoweringContext, MilliOpGraph};
use crate::onnx;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{ONNXDecodingError, query_attribute_string};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
enum Reduction {
    None,
    Add,
    Mul,
    Max,
    Min,
}

/// ONNX ScatterND operator.
///
/// Inputs: data, indices, updates
/// Output: copy of data with updates scattered at indices
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScatterNDOperation {
    global_id: GlobalId,
    data: GlobalId,
    indices: GlobalId,
    updates: GlobalId,
    output: GlobalId,
    reduction: Reduction,
}

impl ScatterNDOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 3 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("ScatterND"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("ScatterND"));
        }

        let reduction = match query_attribute_string(attributes, "reduction").as_deref() {
            Some("add") => Reduction::Add,
            Some("mul") => Reduction::Mul,
            Some("max") => Reduction::Max,
            Some("min") => Reduction::Min,
            _ => Reduction::None,
        };

        Ok(Self {
            global_id: GlobalId::new(rng),
            data: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("ScatterND"))?,
            indices: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("ScatterND"))?,
            updates: inputs[2].ok_or(ONNXDecodingError::InvalidOperatorInputs("ScatterND"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("ScatterND"))?,
            reduction,
        })
    }
}

impl Node for ScatterNDOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "ScatterND".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.data, self.indices, self.updates].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for ScatterNDOperation {
    fn parameters(&self) -> Vec<Property> {
        vec![Property::new(
            "reduction",
            PropertyValue::String(format!("{:?}", self.reduction)),
        )]
    }

    fn is_differentiable(&self) -> bool {
        false
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let reduction = match self.reduction {
            Reduction::None => crate::milli_graph::ops::ScatterReduction::None,
            Reduction::Add => crate::milli_graph::ops::ScatterReduction::Add,
            Reduction::Mul => crate::milli_graph::ops::ScatterReduction::Mul,
            Reduction::Max => crate::milli_graph::ops::ScatterReduction::Max,
            Reduction::Min => crate::milli_graph::ops::ScatterReduction::Min,
        };
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let out = crate::milli_graph::ops::ScatterND::push_new(
            &mut graph,
            input_map[&self.data],
            input_map[&self.indices],
            input_map[&self.updates],
            reduction,
            rng,
        );
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
