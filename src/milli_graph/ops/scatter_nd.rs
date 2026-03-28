use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::pool::Pool;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::scatter_reduction::ScatterReduction;

/// ScatterND: scatter updates into data using multi-dimensional indices.
///
/// Output shape = data shape.
/// Copy data to output, then for each update: compute offset from K index
/// values and apply update with reduction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScatterND {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    data: GlobalId,
    indices: GlobalId,
    updates: GlobalId,
    reduction: ScatterReduction,
}

impl ScatterND {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        indices: GlobalId,
        updates: GlobalId,
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
            reduction,
        };
        graph.push_op(AnyMilliOp::ScatterND(node));
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

impl Node for ScatterND {
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

impl MilliOp for ScatterND {
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
        let out_info = crate::tensor_info::TensorInfo::from_dtype_and_shape_scalars(
            data_info.dtype(),
            &shape,
        );
        Ok(vec![(self.output, out_info)])
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, crate::migration::numeric_tensor::NumericTensor<crate::tensor_rank::DynRank>>,
        _config: &super::MilliEvalConfig,
        _backend: &mut crate::backends::eval_backend::EvalBackend,
    ) -> super::EvalResult {
        let views: Vec<_> = [self.data, self.indices, self.updates]
            .iter()
            .map(|id| crate::symbolic_graph::SharedPoolTensor::from_legacy(&inputs[id]))
            .collect();
        let view_refs: Vec<_> = views.iter().map(|s| s.0.view()).collect();
        let results = self.eval_new(&view_refs, &crate::pool::SystemPool)
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
        let data_rank = data_shape.len();
        let dtype = data.dtype();

        let k = *indices_shape.last().unwrap() as usize;
        let batch_dims = &indices_shape[..indices_shape.len() - 1];
        let num_updates: usize = batch_dims.iter().product::<u64>().max(1) as usize;
        let slice_size: usize = data_shape[k..].iter().product::<u64>().max(1) as usize;
        let total_data: usize = data_shape.iter().product::<u64>() as usize;

        // Compute strides for the first k dims of data.
        let mut data_strides = vec![0usize; k];
        if k > 0 {
            data_strides[k - 1] = slice_size;
            for i in (0..k.saturating_sub(1)).rev() {
                data_strides[i] = data_strides[i + 1] * data_shape[i + 1] as usize;
            }
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
        for u in 0..num_updates {
            let mut offset = 0usize;
            for j in 0..k {
                let mut idx = indices.read_element(u * k + j).to_i64();
                if idx < 0 {
                    idx += data_shape[j] as i64;
                }
                offset += idx as usize * data_strides[j];
            }

            let update_start = u * slice_size;
            for s in 0..slice_size {
                let data_idx = offset + s;
                if data_idx < total_data {
                    let update_val = updates.read_element(update_start + s).to_f32();
                    let existing = out.read_element(data_idx).to_f32();
                    let result = self.reduction.apply(existing, update_val);
                    out.write_element(
                        data_idx,
                        crate::numeric_scalar::NumericScalar::from_f32(result).cast_to(dtype),
                    );
                }
            }
        }

        Ok(vec![out])
    }
}
