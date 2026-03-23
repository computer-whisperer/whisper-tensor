use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::dtype::DType;
use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::{MilliLoweringContext, MilliOpGraph};
use crate::migration::numeric_tensor::NumericTensor;
use crate::onnx;
use crate::symbolic_graph::ops::{EvalError, Operation};
use crate::symbolic_graph::{ONNXDecodingError, query_attribute_int, query_attribute_string};
use crate::tensor_rank::DynRank;
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

/// ONNX ScatterElements operator.
///
/// Inputs: data, indices, updates
/// Output: copy of data with updates scattered at indices along axis
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScatterElementsOperation {
    global_id: GlobalId,
    data: GlobalId,
    indices: GlobalId,
    updates: GlobalId,
    output: GlobalId,
    axis: i64,
    reduction: Reduction,
}

impl ScatterElementsOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 3 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("ScatterElements"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("ScatterElements"));
        }

        let axis = query_attribute_int(attributes, "axis").unwrap_or(0);
        let reduction = match query_attribute_string(attributes, "reduction").as_deref() {
            Some("add") => Reduction::Add,
            Some("mul") => Reduction::Mul,
            Some("max") => Reduction::Max,
            Some("min") => Reduction::Min,
            _ => Reduction::None,
        };

        Ok(Self {
            global_id: GlobalId::new(rng),
            data: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("ScatterElements"))?,
            indices: inputs[1]
                .ok_or(ONNXDecodingError::InvalidOperatorInputs("ScatterElements"))?,
            updates: inputs[2]
                .ok_or(ONNXDecodingError::InvalidOperatorInputs("ScatterElements"))?,
            output: outputs[0]
                .ok_or(ONNXDecodingError::InvalidOperatorOutputs("ScatterElements"))?,
            axis,
            reduction,
        })
    }
}

impl Node for ScatterElementsOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "ScatterElements".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.data, self.indices, self.updates].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for ScatterElementsOperation {
    fn parameters(&self) -> Vec<Property> {
        vec![
            Property::new("axis", PropertyValue::Int(self.axis)),
            Property::new(
                "reduction",
                PropertyValue::String(format!("{:?}", self.reduction)),
            ),
        ]
    }

    fn is_differentiable(&self) -> bool {
        false
    }

    fn eval(
        &self,
        backend: &mut EvalBackend,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, EvalError> {
        let data = &inputs[&self.data];
        let indices = &inputs[&self.indices];
        let updates = &inputs[&self.updates];

        let data_shape: Vec<usize> = data.shape().iter().map(|&v| v as usize).collect();
        let indices_shape: Vec<usize> = indices.shape().iter().map(|&v| v as usize).collect();
        let rank = data_shape.len();

        // Normalize axis
        let axis = if self.axis < 0 {
            (self.axis + rank as i64) as usize
        } else {
            self.axis as usize
        };

        let total_data: usize = data_shape.iter().product();
        let total_indices: usize = indices_shape.iter().product();

        // Get flat data
        let data_f32 = data.cast(DType::F32, backend)?;
        let updates_f32 = updates.cast(DType::F32, backend)?;
        let mut out_data: Vec<f32> = data_f32.to_ndarray()?.flatten().try_into()?;
        let updates_data: Vec<f32> = updates_f32.to_ndarray()?.flatten().try_into()?;
        let indices_cast = indices.cast(DType::I64, backend)?;
        let indices_i64: Vec<i64> = indices_cast.to_ndarray()?.flatten().try_into()?;

        // Compute strides for data
        let mut data_strides = vec![0usize; rank];
        if rank > 0 {
            data_strides[rank - 1] = 1;
            for i in (0..rank - 1).rev() {
                data_strides[i] = data_strides[i + 1] * data_shape[i + 1];
            }
        }

        // Compute strides for indices
        let mut indices_strides = vec![0usize; rank];
        if rank > 0 {
            indices_strides[rank - 1] = 1;
            for i in (0..rank - 1).rev() {
                indices_strides[i] = indices_strides[i + 1] * indices_shape[i + 1];
            }
        }

        for flat_idx in 0..total_indices {
            // Convert flat index to multi-dimensional index in indices tensor
            let mut multi_idx = vec![0usize; rank];
            let mut remaining = flat_idx;
            for d in 0..rank {
                multi_idx[d] = remaining / indices_strides[d];
                remaining %= indices_strides[d];
            }

            // Get index value and handle negative indices
            let mut idx_val = indices_i64[flat_idx];
            if idx_val < 0 {
                idx_val += data_shape[axis] as i64;
            }

            // Build the target position in data: same as multi_idx except on axis dimension
            let mut data_idx = 0usize;
            for d in 0..rank {
                if d == axis {
                    data_idx += idx_val as usize * data_strides[d];
                } else {
                    data_idx += multi_idx[d] * data_strides[d];
                }
            }

            if data_idx < total_data {
                let update_val = updates_data[flat_idx];
                match self.reduction {
                    Reduction::None => out_data[data_idx] = update_val,
                    Reduction::Add => out_data[data_idx] += update_val,
                    Reduction::Mul => out_data[data_idx] *= update_val,
                    Reduction::Max => out_data[data_idx] = out_data[data_idx].max(update_val),
                    Reduction::Min => out_data[data_idx] = out_data[data_idx].min(update_val),
                }
            }
        }

        let out_shape: Vec<u64> = data_shape.iter().map(|&v| v as u64).collect();
        let mut out =
            NumericTensor::NDArray(NDArrayNumericTensor::from_vec_shape(out_data, &out_shape)?);

        let original_dtype = data.dtype();
        if original_dtype != DType::F32 {
            out = out.cast(original_dtype, backend)?;
        }

        let mut result = HashMap::new();
        result.insert(self.output, out);
        Ok(Box::new(result.into_iter()))
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, _rng: &mut impl Rng) -> MilliOpGraph {
        panic!("ScatterElements uses custom eval")
    }
}
