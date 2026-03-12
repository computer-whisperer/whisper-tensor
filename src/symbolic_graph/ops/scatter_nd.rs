use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::dtype::DType;
use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::{MilliLoweringContext, MilliOpGraph};
use crate::numeric_tensor::NumericTensor;
use crate::onnx;
use crate::symbolic_graph::ops::{EvalError, Operation};
use crate::symbolic_graph::{ONNXDecodingError, query_attribute_string};
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
        // k = last dim of indices
        let k = *indices_shape.last().unwrap();
        // batch dims of indices = indices_shape[..q-1]
        let batch_dims = &indices_shape[..indices_shape.len() - 1];
        let num_updates: usize = batch_dims.iter().product();

        // Slice size = product of data_shape[k..]
        let slice_size: usize = data_shape[k..].iter().product();
        let total_data: usize = data_shape.iter().product();

        // Get flat data
        let data_f32 = data.cast(DType::F32, backend)?;
        let updates_f32 = updates.cast(DType::F32, backend)?;
        let mut out_data: Vec<f32> = data_f32.to_ndarray()?.flatten().try_into()?;
        let updates_data: Vec<f32> = updates_f32.to_ndarray()?.flatten().try_into()?;
        let indices_i64: Vec<i64> = indices.to_ndarray()?.flatten().try_into()?;

        // Compute strides for the first k dims of data
        let mut data_strides = vec![0usize; k];
        if k > 0 {
            data_strides[k - 1] = slice_size;
            for i in (0..k - 1).rev() {
                data_strides[i] = data_strides[i + 1] * data_shape[i + 1];
            }
        }

        for u in 0..num_updates {
            // Compute flat offset into data
            let mut offset = 0usize;
            for j in 0..k {
                let mut idx = indices_i64[u * k + j] as isize;
                if idx < 0 {
                    idx += data_shape[j] as isize;
                }
                offset += idx as usize * data_strides[j];
            }

            let update_start = u * slice_size;
            for s in 0..slice_size {
                let data_idx = offset + s;
                if data_idx < total_data {
                    let update_val = updates_data[update_start + s];
                    match self.reduction {
                        Reduction::None => out_data[data_idx] = update_val,
                        Reduction::Add => out_data[data_idx] += update_val,
                        Reduction::Mul => out_data[data_idx] *= update_val,
                        Reduction::Max => out_data[data_idx] = out_data[data_idx].max(update_val),
                        Reduction::Min => out_data[data_idx] = out_data[data_idx].min(update_val),
                    }
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
        panic!("ScatterND uses custom eval")
    }
}
