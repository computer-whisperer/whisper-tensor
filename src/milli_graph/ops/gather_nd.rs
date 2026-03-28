use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::pool::Pool;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// GatherND: gathers slices from data using multi-dimensional indices.
///
/// Output shape = indices_shape[:-1] + data_shape[batch_dims + K:]
/// where K = indices_shape[-1].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatherND {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    data: GlobalId,
    indices: GlobalId,
    batch_dims: i64,
}

impl GatherND {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        indices: GlobalId,
        batch_dims: i64,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            label: None,
            output,
            data,
            indices,
            batch_dims,
        };
        graph.push_op(AnyMilliOp::GatherND(node));
        output
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.data, map);
        super::remap(&mut self.indices, map);
    }
}

impl Node for GatherND {
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

impl MilliOp for GatherND {
    fn infer<'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'p, P>>,
        _symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        pool: &'p P,
    ) -> Result<Vec<(GlobalId, crate::tensor_info::TensorInfo<'p, P>)>, MilliOpGraphError> {
        let data_info = known_inputs
            .get(&self.data)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let indices_info = known_inputs
            .get(&self.indices)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        let data_ranked = data_info
            .as_ranked()
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let indices_ranked = indices_info
            .as_ranked()
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        let data_shape = data_ranked.shape();
        let indices_shape = indices_ranked.shape();
        let batch_dims = self.batch_dims as usize;

        // K = indices_shape[-1] (must be known)
        let k = match indices_shape.last() {
            Some(crate::scalar_info::ScalarInfoTyped::Numeric(k)) => *k as usize,
            _ => return Err(MilliOpGraphError::UnableToInfer),
        };

        // Output shape: indices_shape[:-1] + data_shape[batch_dims + k:]
        let mut out_dims = Vec::new();
        for d in &indices_shape[..indices_shape.len() - 1] {
            out_dims.push(d.clone());
        }
        for d in &data_shape[batch_dims + k..] {
            out_dims.push(d.clone());
        }
        if out_dims.is_empty() {
            out_dims.push(crate::scalar_info::ScalarInfoTyped::Numeric(1));
        }

        let out_dtype = data_info.dtype();
        let out_info =
            crate::tensor_info::TensorInfo::from_dtype_and_shape_scalars(out_dtype, &out_dims);

        if let Some(results) = super::constant_fold(
            self,
            known_inputs,
            &[(self.output, out_info.clone_with_pool(pool))],
            pool,
        ) {
            return Ok(results);
        }

        Ok(vec![(self.output, out_info)])
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, crate::migration::numeric_tensor::NumericTensor<crate::tensor_rank::DynRank>>,
        _config: &super::MilliEvalConfig,
        _backend: &mut crate::backends::eval_backend::EvalBackend,
    ) -> super::EvalResult {
        let views: Vec<_> = [self.data, self.indices]
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
        let data_shape = data.shape();
        let indices_shape = indices.shape();
        let data_rank = data_shape.len();
        let indices_rank = indices_shape.len();
        let batch_dims = self.batch_dims as usize;
        let dtype = data.dtype();

        let k = *indices_shape.last().unwrap() as usize;

        // Output shape: indices_shape[:-1] + data_shape[batch_dims + k:]
        let mut out_shape: Vec<u64> = Vec::new();
        for &d in &indices_shape[..indices_rank - 1] {
            out_shape.push(d);
        }
        let slice_dims = &data_shape[batch_dims + k..];
        for &d in slice_dims {
            out_shape.push(d);
        }
        if out_shape.is_empty() {
            out_shape.push(1);
        }

        let slice_size: usize = slice_dims.iter().product::<u64>().max(1) as usize;
        let num_lookups: usize = indices_shape[..indices_rank - 1]
            .iter()
            .product::<u64>()
            .max(1) as usize;
        let total_out: usize = out_shape.iter().product::<u64>() as usize;

        let layout = TensorLayout::<DynRank>::row_major(out_shape, dtype);
        let buf = pool
            .allocate(layout.buffer_size_bytes())
            .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
        let mut out = NumericTensor::from_parts(buf, layout);

        if total_out == 0 {
            return Ok(vec![out]);
        }

        // Compute data strides.
        let mut data_strides = vec![1usize; data_rank];
        for i in (0..data_rank.saturating_sub(1)).rev() {
            data_strides[i] = data_strides[i + 1] * data_shape[i + 1] as usize;
        }

        let batch_size: usize = data_shape[..batch_dims]
            .iter()
            .product::<u64>()
            .max(1) as usize;
        let lookups_per_batch = num_lookups / batch_size.max(1);
        let data_per_batch: usize = data.numel() / batch_size.max(1);

        for lookup in 0..num_lookups {
            let batch_idx = if batch_dims > 0 {
                lookup / lookups_per_batch
            } else {
                0
            };

            let idx_start = lookup * k;
            let mut data_offset = batch_idx * data_per_batch;

            for j in 0..k {
                let mut idx_val = indices.read_element(idx_start + j).to_i64();
                if idx_val < 0 {
                    idx_val += data_shape[batch_dims + j] as i64;
                }
                data_offset += idx_val as usize * data_strides[batch_dims + j];
            }

            let out_start = lookup * slice_size;
            for s in 0..slice_size {
                out.write_element(out_start + s, data.read_element(data_offset + s));
            }
        }

        Ok(vec![out])
    }

    fn lower_to_nano(
        &self,
        _ctx: &mut crate::nano_graph::NanoLoweringContext,
    ) -> super::LowerResult {
        // GatherND nano lowering: only simple case batch_dims=0, K=data_rank (scalar gather).
        // For now, all cases fall through to opaque eval_new.
        super::LowerResult::Unsupported
    }
}
