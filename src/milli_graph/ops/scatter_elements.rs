use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::pool::Pool;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::scatter_reduction::ScatterReduction;

/// ScatterElements: scatter updates into data along an axis using element-wise indices.
///
/// Output shape = data shape.
/// Copy data to output, then for each element in indices: write updates[i]
/// to output at the computed coordinates (with optional reduction).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScatterElements {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    data: GlobalId,
    indices: GlobalId,
    updates: GlobalId,
    axis: i64,
    reduction: ScatterReduction,
}

impl ScatterElements {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        indices: GlobalId,
        updates: GlobalId,
        axis: i64,
        reduction: ScatterReduction,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            label: None,
            output,
            data,
            indices,
            updates,
            axis,
            reduction,
        };
        graph.push_op(AnyMilliOp::ScatterElements(node));
        output
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.data, map);
        super::remap(&mut self.indices, map);
        super::remap(&mut self.updates, map);
    }
}

impl Node for ScatterElements {
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

impl MilliOp for ScatterElements {
    fn infer<'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'p, P>>,
        _symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        _pool: &'p P,
    ) -> Result<Vec<(GlobalId, crate::tensor_info::TensorInfo<'p, P>)>, MilliOpGraphError> {
        // Output shape = data shape.
        let data_info = known_inputs
            .get(&self.data)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let ranked = data_info
            .as_ranked()
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let shape = ranked.shape();
        let out_info =
            crate::tensor_info::TensorInfo::from_dtype_and_shape_scalars(data_info.dtype(), &shape);
        Ok(vec![(self.output, out_info)])
    }

    fn eval(
        &self,
        inputs: &HashMap<
            GlobalId,
            crate::migration::numeric_tensor::NumericTensor<crate::tensor_rank::DynRank>,
        >,
        _config: &super::MilliEvalConfig,
        _backend: &mut crate::backends::eval_backend::EvalBackend,
    ) -> super::EvalResult {
        let pool_tensors: Vec<_> = [self.data, self.indices, self.updates]
            .iter()
            .map(|id| {
                crate::nano_graph::lower::legacy_numeric_to_new(
                    &inputs[id],
                    &crate::pool::SystemPool,
                )
            })
            .collect();
        let view_refs: Vec<_> = pool_tensors.iter().map(|t| t.view()).collect();
        let results = self
            .eval_new(&view_refs, &crate::pool::SystemPool)
            .map_err(|e| MilliOpGraphError::InvalidInput(format!("{e}")))?;
        let output = self.output;
        Ok(Box::new(results.into_iter().map(move |t| {
            (output, crate::nano_graph::lower::new_numeric_to_legacy(&t))
        })))
    }

    fn eval_new<'p, P2: Pool + 'p>(
        &self,
        inputs: &[crate::numeric_tensor::NumericTensorView<'_, crate::tensor_rank::DynRank>],
        pool: &'p P2,
    ) -> Result<
        Vec<crate::numeric_tensor::NumericTensor<'p, crate::tensor_rank::DynRank, P2>>,
        crate::nano_graph::pool_eval::PoolEvalError,
    > {
        use crate::numeric_tensor::{NumericTensor, TensorLayout};
        use crate::tensor_rank::DynRank;

        let data = &inputs[0];
        let indices = &inputs[1];
        let updates = &inputs[2];
        let data_shape = data.shape();
        let indices_shape = indices.shape();
        let rank = data_shape.len();
        let dtype = data.dtype();
        let axis = if self.axis < 0 {
            (self.axis + rank as i64) as usize
        } else {
            self.axis as usize
        };

        let total_data: usize = data_shape.iter().product::<u64>() as usize;
        let total_indices: usize = indices_shape.iter().product::<u64>() as usize;

        // Compute strides.
        let mut data_strides = vec![1usize; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            data_strides[i] = data_strides[i + 1] * data_shape[i + 1] as usize;
        }
        let mut indices_strides = vec![1usize; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            indices_strides[i] = indices_strides[i + 1] * indices_shape[i + 1] as usize;
        }

        // Copy data to output.
        let layout = TensorLayout::<DynRank>::row_major(data_shape.to_vec(), dtype);
        let buf = pool
            .allocate(layout.buffer_size_bytes())
            .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
        let mut out = NumericTensor::from_parts(buf, layout);
        for i in 0..total_data {
            out.write_element(i, data.read_element(i));
        }

        // Scatter updates.
        for flat_idx in 0..total_indices {
            let mut multi_idx = vec![0usize; rank];
            let mut remaining = flat_idx;
            for d in 0..rank {
                multi_idx[d] = remaining / indices_strides[d];
                remaining %= indices_strides[d];
            }

            let mut idx_val = indices.read_element(flat_idx).to_i64();
            if idx_val < 0 {
                idx_val += data_shape[axis] as i64;
            }

            let mut data_idx = 0usize;
            for d in 0..rank {
                if d == axis {
                    data_idx += idx_val as usize * data_strides[d];
                } else {
                    data_idx += multi_idx[d] * data_strides[d];
                }
            }

            if data_idx < total_data {
                let update_val = updates.read_element(flat_idx).to_f32();
                let existing = out.read_element(data_idx).to_f32();
                let result = self.reduction.apply(existing, update_val);
                out.write_element(
                    data_idx,
                    crate::numeric_scalar::NumericScalar::from_f32(result).cast_to(dtype),
                );
            }
        }

        Ok(vec![out])
    }
}
