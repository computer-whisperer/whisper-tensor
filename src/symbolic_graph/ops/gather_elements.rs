use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::{MilliLoweringContext, MilliOpGraph};
use crate::onnx;
use crate::symbolic_graph::ONNXDecodingError;
use crate::symbolic_graph::ops::Operation;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// ONNX GatherElements operator.
///
/// Gathers values along an axis using element-wise indices.
/// output[i][j][k] = data[index[i][j][k]][j][k]  (for axis=0)
/// output[i][j][k] = data[i][index[i][j][k]][k]  (for axis=1)
/// etc.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GatherElementsOperation {
    global_id: GlobalId,
    data: GlobalId,
    indices: GlobalId,
    output: GlobalId,
    axis: i64,
}

impl GatherElementsOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("GatherElements"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("GatherElements"));
        }
        let mut axis = 0i64;
        for attr in attributes {
            if attr.name == "axis" {
                axis = attr.i;
            }
        }
        Ok(Self {
            global_id: GlobalId::new(rng),
            data: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("GatherElements"))?,
            indices: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("GatherElements"))?,
            output: outputs[0]
                .ok_or(ONNXDecodingError::InvalidOperatorOutputs("GatherElements"))?,
            axis,
        })
    }
}

impl Node for GatherElementsOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "GatherElements".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.data, self.indices].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for GatherElementsOperation {
    fn parameters(&self) -> Vec<Property> {
        vec![Property::new("axis", PropertyValue::Int(self.axis))]
    }

    fn is_differentiable(&self) -> bool {
        false
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let out = crate::milli_graph::ops::GatherElements::push_new(
            &mut graph,
            input_map[&self.data],
            input_map[&self.indices],
            self.axis,
            rng,
        );
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

/// ONNX GatherND operator.
///
/// Gathers slices from data using multi-dimensional indices.
/// batch_dims defaults to 0.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GatherNDOperation {
    global_id: GlobalId,
    data: GlobalId,
    indices: GlobalId,
    output: GlobalId,
    batch_dims: i64,
}

impl GatherNDOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("GatherND"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("GatherND"));
        }
        let mut batch_dims = 0i64;
        for attr in attributes {
            if attr.name == "batch_dims" {
                batch_dims = attr.i;
            }
        }
        Ok(Self {
            global_id: GlobalId::new(rng),
            data: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("GatherND"))?,
            indices: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("GatherND"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("GatherND"))?,
            batch_dims,
        })
    }
}

impl Node for GatherNDOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "GatherND".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.data, self.indices].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for GatherNDOperation {
    fn parameters(&self) -> Vec<Property> {
        vec![Property::new(
            "batch_dims",
            PropertyValue::Int(self.batch_dims),
        )]
    }

    fn is_differentiable(&self) -> bool {
        false
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let out = crate::milli_graph::ops::GatherND::push_new(
            &mut graph,
            input_map[&self.data],
            input_map[&self.indices],
            self.batch_dims,
            rng,
        );
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
