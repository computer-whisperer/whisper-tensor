use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::{MilliLoweringContext, MilliOpGraph};
use crate::onnx::AttributeProto;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{ONNXDecodingError, query_attribute_int};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// ONNX OneHot operator.
///
/// Produces a one-hot tensor. indices[i] -> a vector of length depth with
/// values[1] at position indices[i] and values[0] elsewhere.
///
/// Inputs: indices (any shape), depth (scalar), values [off_value, on_value]
/// Output: shape = indices.shape with a new dim of size depth inserted at axis.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OneHotOperation {
    global_id: GlobalId,
    indices: GlobalId,
    depth: GlobalId,
    values: GlobalId,
    output: GlobalId,
    axis: i64,
}

impl OneHotOperation {
    pub fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() < 3 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("OneHot"));
        }
        if outputs.is_empty() {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("OneHot"));
        }

        let axis = query_attribute_int(attributes, "axis").unwrap_or(-1);

        Ok(Self {
            global_id: GlobalId::new(rng),
            indices: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("OneHot"))?,
            depth: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("OneHot"))?,
            values: inputs[2].ok_or(ONNXDecodingError::InvalidOperatorInputs("OneHot"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("OneHot"))?,
            axis,
        })
    }
}

impl Node for OneHotOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "OneHot".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        Box::new([self.indices, self.depth, self.values].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for OneHotOperation {
    fn parameters(&self) -> Vec<Property> {
        vec![Property::new("axis", PropertyValue::Int(self.axis))]
    }

    fn is_differentiable(&self) -> bool {
        false
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, _rng: &mut impl Rng) -> MilliOpGraph {
        panic!("OneHot uses custom eval, not milli-op decomposition")
    }
}
