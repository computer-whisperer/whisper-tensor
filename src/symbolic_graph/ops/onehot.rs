use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::dtype::DType;
use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::migration::numeric_tensor::NumericTensor;
use crate::milli_graph::{MilliLoweringContext, MilliOpGraph};
use crate::onnx::AttributeProto;
use crate::symbolic_graph::ops::{EvalError, Operation};
use crate::symbolic_graph::{ONNXDecodingError, query_attribute_int};
use crate::tensor_rank::DynRank;
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

    fn eval(
        &self,
        backend: &mut EvalBackend,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, EvalError> {
        let indices = &inputs[&self.indices];
        let depth_tensor = &inputs[&self.depth];
        let values = &inputs[&self.values];

        // Extract depth scalar
        let depth_vec: Vec<i64> = depth_tensor
            .cast(DType::I64, backend)?
            .to_ndarray()?
            .flatten()
            .try_to_vec()?;
        let depth = depth_vec[0] as usize;

        // Extract off/on values
        let values_vec: Vec<f64> = values
            .cast(DType::F64, backend)?
            .to_ndarray()?
            .flatten()
            .try_to_vec()?;
        let off_value = values_vec[0];
        let on_value = values_vec[1];

        // Get indices as i64
        let idx_vec: Vec<i64> = indices
            .cast(DType::I64, backend)?
            .to_ndarray()?
            .flatten()
            .try_to_vec()?;
        let idx_shape = indices.shape();
        let idx_rank = idx_shape.len();

        // Resolve axis (output rank = idx_rank + 1)
        let output_rank = idx_rank + 1;
        let axis = if self.axis < 0 {
            (output_rank as i64 + self.axis) as usize
        } else {
            self.axis as usize
        };

        // Build output shape: insert depth at axis
        let mut out_shape: Vec<u64> = Vec::with_capacity(output_rank);
        for d in 0..output_rank {
            if d == axis {
                out_shape.push(depth as u64);
            } else {
                let idx_d = if d < axis { d } else { d - 1 };
                out_shape.push(idx_shape[idx_d]);
            }
        }

        // Compute strides for output
        let mut out_strides = vec![1usize; output_rank];
        for d in (0..output_rank - 1).rev() {
            out_strides[d] = out_strides[d + 1] * out_shape[d + 1] as usize;
        }

        // Compute strides for indices
        let mut idx_strides = vec![1usize; idx_rank];
        for d in (0..idx_rank.saturating_sub(1)).rev() {
            idx_strides[d] = idx_strides[d + 1] * idx_shape[d + 1] as usize;
        }

        let total_out: usize = out_shape.iter().product::<u64>() as usize;
        let mut out_data = vec![off_value; total_out];

        for (flat_idx, &raw_idx) in idx_vec.iter().enumerate() {
            let mut idx_val = raw_idx;
            if idx_val < 0 {
                idx_val += depth as i64;
            }
            if idx_val < 0 || idx_val >= depth as i64 {
                continue;
            }

            // Convert flat index in indices to coordinates
            let mut remaining = flat_idx;
            let mut out_flat = 0usize;
            for (d, &stride) in idx_strides.iter().enumerate() {
                let coord = remaining / stride;
                remaining %= stride;
                let out_d = if d < axis { d } else { d + 1 };
                out_flat += coord * out_strides[out_d];
            }
            // Add the depth coordinate
            out_flat += idx_val as usize * out_strides[axis];
            out_data[out_flat] = on_value;
        }

        // Build the output tensor in the values dtype
        let out_nd: NDArrayNumericTensor<DynRank> =
            NDArrayNumericTensor::from_vec_shape(out_data, &out_shape)?;
        let out_tensor = NumericTensor::NDArray(out_nd.to_dyn());
        let out_tensor = out_tensor.cast(values.dtype(), backend)?;

        let mut result = HashMap::new();
        result.insert(self.output, out_tensor);
        Ok(Box::new(result.into_iter()))
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, _rng: &mut impl Rng) -> MilliOpGraph {
        panic!("OneHot uses custom eval, not milli-op decomposition")
    }
}
