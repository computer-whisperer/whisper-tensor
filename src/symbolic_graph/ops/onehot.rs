use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::ops as milli_ops;
use crate::milli_graph::{MilliLoweringContext, MilliOpGraph, ops_helpers};
use crate::onnx::AttributeProto;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{ONNXDecodingError, query_attribute_int};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        use crate::numeric_dtype::NumericDType;

        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let indices_in = input_map[&self.indices];
        let depth_in = input_map[&self.depth];
        let values_in = input_map[&self.values];

        // Cast indices to I64 (they may be float).
        let indices_i64 = milli_ops::Cast::push_new(&mut graph, indices_in, NumericDType::I64, rng);

        // depth is a scalar — cast to I64, then reshape to [1] for Range.
        let depth_i64 = milli_ops::Cast::push_new(&mut graph, depth_in, NumericDType::I64, rng);
        let shape_1 = milli_ops::Constant::from_vec(&mut graph, vec![1i64], rng);
        let depth_1d = milli_ops::Reshape::push_new(&mut graph, depth_i64, shape_1, false, rng);

        // Handle negative indices: norm = Where(indices < 0, indices + depth, indices).
        let zero = ops_helpers::scalar_const(&mut graph, 0i64, rng);
        let neg_mask = milli_ops::SimpleBinary::less(&mut graph, indices_i64, zero, rng);
        let indices_plus_depth =
            milli_ops::SimpleBinary::add(&mut graph, indices_i64, depth_i64, rng);
        let norm_indices =
            milli_ops::Where::push_new(&mut graph, neg_mask, indices_plus_depth, indices_i64, rng);

        // Unsqueeze indices at axis — adds the depth dimension.
        let axis_const = ops_helpers::scalar_const(&mut graph, self.axis, rng);
        let unsqueezed = milli_ops::Unsqueeze::push_new(&mut graph, norm_indices, axis_const, rng);

        // Range(0, depth, 1) → [depth] I64.
        let one = ops_helpers::scalar_const(&mut graph, 1i64, rng);
        let arange = milli_ops::Range::push_new(&mut graph, zero, depth_1d, one, rng);

        // Reshape range to [1, ..., depth, ..., 1] for broadcasting.
        // output_rank = indices_rank + 1
        let indices_shape = milli_ops::Shape::push_new(&mut graph, indices_i64, rng);
        let indices_rank = milli_ops::Shape::push_new(&mut graph, indices_shape, rng);
        let output_rank = milli_ops::SimpleBinary::add(&mut graph, indices_rank, one, rng);

        // Normalize axis: norm_axis = axis + output_rank if axis < 0, else axis.
        let norm_axis = if self.axis < 0 {
            milli_ops::SimpleBinary::add(&mut graph, axis_const, output_rank, rng)
        } else {
            axis_const
        };
        let norm_axis_plus_one = milli_ops::SimpleBinary::add(&mut graph, norm_axis, one, rng);

        // Build reshape target: [1, ..., 1] with depth at axis position.
        let nd = milli_ops::SimpleBinary::sub(&mut graph, output_rank, one, rng);
        let ones_shape = milli_ops::ConstantOfShape::push_new(
            &mut graph,
            crate::numeric_scalar::NumericScalar::from_i64(1),
            nd,
            rng,
        );
        // before = Slice(ones, 0, norm_axis), after = Slice(ones, norm_axis, nd)
        let before =
            milli_ops::Slice::push_new(&mut graph, ones_shape, zero, norm_axis, None, None, rng);
        let after =
            milli_ops::Slice::push_new(&mut graph, ones_shape, norm_axis, nd, None, None, rng);
        let reshape_target =
            milli_ops::Concat::push_new(&mut graph, vec![before, depth_1d, after], 0, rng);
        let arange_reshaped =
            milli_ops::Reshape::push_new(&mut graph, arange, reshape_target, false, rng);

        // Equal(unsqueezed_indices, reshaped_range) → Bool mask.
        let mask = milli_ops::SimpleBinary::equal(&mut graph, unsqueezed, arange_reshaped, rng);

        // Extract on/off values.
        let zero_idx = ops_helpers::scalar_const(&mut graph, 0i64, rng);
        let one_idx = ops_helpers::scalar_const(&mut graph, 1i64, rng);
        let off_val = milli_ops::Gather::push_new(&mut graph, values_in, zero_idx, 0, rng);
        let on_val = milli_ops::Gather::push_new(&mut graph, values_in, one_idx, 0, rng);

        // Where(mask, on_value, off_value) → output.
        let result = milli_ops::Where::push_new(&mut graph, mask, on_val, off_val, rng);

        let mut output_map = HashMap::new();
        output_map.insert(result, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
