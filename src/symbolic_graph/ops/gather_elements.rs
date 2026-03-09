use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::dtype::DType;
use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::MilliOpGraph;
use crate::numeric_tensor::NumericTensor;
use crate::onnx;
use crate::symbolic_graph::ONNXDecodingError;
use crate::symbolic_graph::ops::{EvalError, Operation};
use crate::tensor_rank::DynRank;
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
            indices: inputs[1]
                .ok_or(ONNXDecodingError::InvalidOperatorInputs("GatherElements"))?,
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

    fn eval(
        &self,
        backend: &mut EvalBackend,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, EvalError> {
        let data = &inputs[&self.data];
        let indices = &inputs[&self.indices];

        let data_shape: Vec<usize> = data.shape().iter().map(|&v| v as usize).collect();
        let indices_shape: Vec<usize> = indices.shape().iter().map(|&v| v as usize).collect();
        let rank = data_shape.len();

        // Normalize axis
        let axis = if self.axis < 0 {
            (rank as i64 + self.axis) as usize
        } else {
            self.axis as usize
        };

        // Total elements in output = total elements in indices
        let total: usize = indices_shape.iter().product();

        let data_f32 = data.cast(DType::F32, backend)?;
        let data_flat: Vec<f32> = data_f32.to_ndarray()?.flatten().try_into()?;
        let indices_i64: Vec<i64> = indices.to_ndarray()?.flatten().try_into()?;

        // Compute data strides
        let mut data_strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            data_strides[i] = data_strides[i + 1] * data_shape[i + 1];
        }

        // Compute indices strides
        let mut indices_strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            indices_strides[i] = indices_strides[i + 1] * indices_shape[i + 1];
        }

        let mut out = vec![0.0f32; total];

        for flat_idx in 0..total {
            // Convert flat index to multi-dimensional index in indices tensor
            let mut multi_idx = vec![0usize; rank];
            let mut remaining = flat_idx;
            for d in 0..rank {
                multi_idx[d] = remaining / indices_strides[d];
                remaining %= indices_strides[d];
            }

            // The index along the gather axis comes from the indices tensor
            let mut gather_idx = indices_i64[flat_idx];
            if gather_idx < 0 {
                gather_idx += data_shape[axis] as i64;
            }

            // Build the data index: same as multi_idx except at axis dimension
            let mut data_flat_idx = 0usize;
            for d in 0..rank {
                let dim_idx = if d == axis {
                    gather_idx as usize
                } else {
                    multi_idx[d]
                };
                data_flat_idx += dim_idx * data_strides[d];
            }

            out[flat_idx] = data_flat[data_flat_idx];
        }

        let out_shape: Vec<u64> = indices_shape.iter().map(|&v| v as u64).collect();
        let mut result =
            NumericTensor::NDArray(NDArrayNumericTensor::from_vec_shape(out, &out_shape)?);

        let original_dtype = data.dtype();
        if original_dtype != DType::F32 {
            result = result.cast(original_dtype, backend)?;
        }

        let mut map = HashMap::new();
        map.insert(self.output, result);
        Ok(Box::new(map.into_iter()))
    }

    fn get_milli_op_graph(&self, _rng: &mut impl Rng) -> MilliOpGraph {
        panic!("GatherElements uses custom eval")
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

    fn eval(
        &self,
        backend: &mut EvalBackend,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, EvalError> {
        let data = &inputs[&self.data];
        let indices = &inputs[&self.indices];

        let data_shape: Vec<usize> = data.shape().iter().map(|&v| v as usize).collect();
        let indices_shape: Vec<usize> = indices.shape().iter().map(|&v| v as usize).collect();
        let data_rank = data_shape.len();
        let indices_rank = indices_shape.len();
        let batch_dims = self.batch_dims as usize;

        // Last dim of indices = number of index dimensions
        let k = *indices_shape.last().unwrap();

        // Output shape: indices_shape[:-1] + data_shape[batch_dims + k:]
        let mut out_shape: Vec<usize> = Vec::new();
        for &d in &indices_shape[..indices_rank - 1] {
            out_shape.push(d);
        }
        let slice_dims = &data_shape[batch_dims + k..];
        for &d in slice_dims {
            out_shape.push(d);
        }

        let slice_size: usize = slice_dims.iter().product::<usize>().max(1);
        let num_lookups: usize = indices_shape[..indices_rank - 1].iter().product::<usize>().max(1);
        let total_out = out_shape.iter().product::<usize>();

        // Handle empty output
        if total_out == 0 {
            let out_shape_u64: Vec<u64> = out_shape.iter().map(|&v| v as u64).collect();
            let result = NumericTensor::NDArray(NDArrayNumericTensor::from_vec_shape(
                Vec::<f32>::new(),
                &out_shape_u64,
            )?);
            let mut map = HashMap::new();
            map.insert(self.output, result);
            return Ok(Box::new(map.into_iter()));
        }

        let data_f32 = data.cast(DType::F32, backend)?;
        let data_flat: Vec<f32> = data_f32.to_ndarray()?.flatten().try_into()?;
        let indices_i64: Vec<i64> = indices.to_ndarray()?.flatten().try_into()?;

        // Compute data strides
        let mut data_strides = vec![1usize; data_rank];
        for i in (0..data_rank - 1).rev() {
            data_strides[i] = data_strides[i + 1] * data_shape[i + 1];
        }

        // Compute batch strides for indices (batch_dims dimensions)
        let batch_size: usize = data_shape[..batch_dims].iter().product::<usize>().max(1);
        let lookups_per_batch = num_lookups / batch_size.max(1);
        let data_per_batch: usize = data_flat.len() / batch_size.max(1);

        let mut out = vec![0.0f32; total_out];

        for lookup in 0..num_lookups {
            let batch_idx = if batch_dims > 0 {
                lookup / lookups_per_batch
            } else {
                0
            };

            // Read the k index values
            let idx_start = lookup * k;
            let mut data_offset = batch_idx * data_per_batch;

            for j in 0..k {
                let mut idx_val = indices_i64[idx_start + j];
                if idx_val < 0 {
                    idx_val += data_shape[batch_dims + j] as i64;
                }
                data_offset += idx_val as usize * data_strides[batch_dims + j];
            }

            // Copy slice
            let out_start = lookup * slice_size;
            for s in 0..slice_size {
                out[out_start + s] = data_flat[data_offset + s];
            }
        }

        let out_shape_u64: Vec<u64> = out_shape.iter().map(|&v| v as u64).collect();
        let mut result =
            NumericTensor::NDArray(NDArrayNumericTensor::from_vec_shape(out, &out_shape_u64)?);

        let original_dtype = data.dtype();
        if original_dtype != DType::F32 {
            result = result.cast(original_dtype, backend)?;
        }

        let mut map = HashMap::new();
        map.insert(self.output, result);
        Ok(Box::new(map.into_iter()))
    }

    fn get_milli_op_graph(&self, _rng: &mut impl Rng) -> MilliOpGraph {
        panic!("GatherND uses custom eval")
    }
}
