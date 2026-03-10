use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::numeric_tensor::NumericTensor;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gather {
    global_id: GlobalId,
    output: GlobalId,
    data: GlobalId,
    indices: GlobalId,
    axis: i64,
}

impl Gather {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        indices: GlobalId,
        axis: i64,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            output,
            data,
            indices,
            axis,
        };
        graph.push_op(AnyMilliOp::Gather(node));
        output
    }
}

impl Gather {
    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.data, map);
        super::remap(&mut self.indices, map);
    }
}

impl Node for Gather {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Gather".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.data, self.indices].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
}

impl MilliOp for Gather {
    fn backward(
        &self,
        output_grads: &HashMap<GlobalId, GlobalId>,
        graph: &mut MilliOpGraph,
        rng: &mut impl Rng,
    ) -> Option<HashMap<GlobalId, GlobalId>> {
        let grad_output = *output_grads.get(&self.output)?;
        let mut result = HashMap::new();
        // Only the data input is differentiable (not indices)
        let grad_data =
            GatherGrad::push_new(graph, grad_output, self.indices, self.data, self.axis, rng);
        result.insert(self.data, grad_data);
        Some(result)
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        let out = NumericTensor::<DynRank>::gather(
            &inputs[&self.data],
            &inputs[&self.indices],
            self.axis,
            backend,
        )?;
        Ok(Box::new([(self.output, out)].into_iter()))
    }
}

// ---------------------------------------------------------------------------
// GatherGrad: scatter-add grad_output back into data shape
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatherGrad {
    global_id: GlobalId,
    output: GlobalId,
    grad_output: GlobalId,
    indices: GlobalId,
    /// Original data tensor — needed for its shape
    data: GlobalId,
    axis: i64,
}

impl GatherGrad {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        grad_output: GlobalId,
        indices: GlobalId,
        data: GlobalId,
        axis: i64,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            output,
            grad_output,
            indices,
            data,
            axis,
        };
        graph.push_op(AnyMilliOp::GatherGrad(node));
        output
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.grad_output, map);
        super::remap(&mut self.indices, map);
        super::remap(&mut self.data, map);
    }
}

impl Node for GatherGrad {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> String {
        "GatherGrad".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.grad_output, self.indices, self.data].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl MilliOp for GatherGrad {
    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        let grad_out = &inputs[&self.grad_output];
        let indices = &inputs[&self.indices];
        let data = &inputs[&self.data];

        let data_shape: Vec<usize> = data.shape().iter().map(|&x| x as usize).collect();
        let rank = data_shape.len();
        let axis = if self.axis < 0 {
            (self.axis + rank as i64) as usize
        } else {
            self.axis as usize
        };

        let grad_f32 = grad_out.cast(DType::F32, backend)?;
        let grad_data: Vec<f32> = grad_f32.to_ndarray()?.flatten().try_into()?;

        let indices_i64 = indices.cast(DType::I64, backend)?;
        let idx_data: Vec<i64> = indices_i64.to_ndarray()?.flatten().try_into()?;
        let idx_shape: Vec<usize> = indices.shape().iter().map(|&x| x as usize).collect();

        let out_size: usize = data_shape.iter().product();
        let mut result = vec![0.0f32; out_size];

        // Compute strides for the data tensor
        let mut data_strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            data_strides[i] = data_strides[i + 1] * data_shape[i + 1];
        }

        // The grad_output shape is: data_shape[..axis] ++ idx_shape ++ data_shape[axis+1..]
        // We iterate over every element of grad_output, compute which data element it
        // came from, and scatter-add.
        let grad_shape: Vec<usize> = grad_out.shape().iter().map(|&x| x as usize).collect();
        let grad_rank = grad_shape.len();
        let mut grad_strides = vec![1usize; grad_rank];
        for i in (0..grad_rank.saturating_sub(1)).rev() {
            grad_strides[i] = grad_strides[i + 1] * grad_shape[i + 1];
        }

        let prefix_dims = axis;
        let suffix_dims = rank - axis - 1;
        let idx_ndim = idx_shape.len();
        let axis_len = data_shape[axis] as i64;

        for (flat_g, &val) in grad_data.iter().enumerate() {
            if val == 0.0 {
                continue;
            }

            // Decompose flat_g into (prefix_coords, idx_coords, suffix_coords)
            let mut rem = flat_g;

            // Prefix coords (dims 0..axis of data)
            let mut data_flat = 0usize;
            for d in 0..prefix_dims {
                let coord = rem / grad_strides[d];
                rem %= grad_strides[d];
                data_flat += coord * data_strides[d];
            }

            // Index coords (middle dims from indices shape)
            let mut idx_flat = 0usize;
            let mut idx_strides = vec![1usize; idx_ndim];
            for i in (0..idx_ndim.saturating_sub(1)).rev() {
                idx_strides[i] = idx_strides[i + 1] * idx_shape[i + 1];
            }
            for (d, &stride) in idx_strides.iter().enumerate() {
                let grad_dim = prefix_dims + d;
                let coord = rem / grad_strides[grad_dim];
                rem %= grad_strides[grad_dim];
                idx_flat += coord * stride;
            }

            // Suffix coords (dims axis+1.. of data)
            for d in 0..suffix_dims {
                let grad_dim = prefix_dims + idx_ndim + d;
                let coord = rem / grad_strides[grad_dim];
                rem %= grad_strides[grad_dim];
                data_flat += coord * data_strides[axis + 1 + d];
            }

            // Look up the actual index along the gather axis
            let mut idx_val = idx_data[idx_flat];
            if idx_val < 0 {
                idx_val += axis_len;
            }
            data_flat += idx_val as usize * data_strides[axis];

            result[data_flat] += val;
        }

        let result_tensor = NumericTensor::<DynRank>::from_vec_shape(result, data_shape)?;
        Ok(Box::new(std::iter::once((self.output, result_tensor))))
    }
}
