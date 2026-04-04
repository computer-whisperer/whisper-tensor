use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::pool::Pool;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Reduces `data` via summation so its shape matches `target_shape`.
///
/// This is the standard "un-broadcast" operation used in backward passes:
/// when a binary op broadcasts input A from shape [3,1] to [3,4], the
/// gradient has shape [3,4] but needs to be reduced back to [3,1].
///
/// Handles both rank-padding (fewer dims) and dim=1 broadcasting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SumTo {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    data: GlobalId,
    target_shape: GlobalId,
}

impl SumTo {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        target_shape: GlobalId,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        Self::push_new_with_label(graph, data, target_shape, None, rng)
    }

    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        target_shape: GlobalId,
        label: Option<String>,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            label,
            output,
            data,
            target_shape,
        };
        graph.push_op(AnyMilliOp::SumTo(node));
        output
    }
}

impl SumTo {
    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.data, map);
        super::remap(&mut self.target_shape, map);
    }
}

impl Node for SumTo {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> String {
        "SumTo".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new([self.data, self.target_shape].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new([self.output].into_iter())
    }
}

impl MilliOp for SumTo {
    fn infer<'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'p, P>>,
        symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        _pool: &'p P,
    ) -> Result<Vec<(GlobalId, crate::tensor_info::TensorInfo<'p, P>)>, MilliOpGraphError> {
        use crate::scalar_info::{ScalarInfo, ScalarInfoTyped};
        use crate::symbolic_scalar::SymbolicScalar;
        use crate::tensor_info::TensorInfo;

        let data_info = known_inputs
            .get(&self.data)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let target_info = known_inputs
            .get(&self.target_shape)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let out_dtype = data_info.dtype();

        // If target_shape is concrete, extract its values to determine output shape.
        if let Some(shape_vals) = target_info.to_i64_vec() {
            let dims: Vec<ScalarInfoTyped<u64>> = shape_vals
                .iter()
                .map(|&v| ScalarInfoTyped::Numeric(v as u64))
                .collect();
            return Ok(vec![(
                self.output,
                TensorInfo::from_dtype_and_shape_scalars(out_dtype, &dims),
            )]);
        }

        // target_shape is a 1D tensor whose length = output rank.
        // If we know the length of target_shape, we know the output rank.
        if let Some(out_rank) = target_info.dim_if_known(0) {
            let first = ScalarInfo::Symbolic(SymbolicScalar::new(out_dtype, symbolic_resolver));
            return Ok(vec![(
                self.output,
                TensorInfo::new_from_first_element_and_rank(
                    first,
                    ScalarInfoTyped::Numeric(out_rank as u32),
                    symbolic_resolver,
                ),
            )]);
        }

        Err(MilliOpGraphError::UnableToInfer)
    }

    fn eval_new<'p, P2: crate::pool::Pool + 'p>(
        &self,
        inputs: &[crate::numeric_tensor::NumericTensorView<'_, crate::tensor_rank::DynRank>],
        pool: &'p P2,
    ) -> Result<
        Vec<crate::numeric_tensor::NumericTensor<'p, crate::tensor_rank::DynRank, P2>>,
        crate::nano_graph::pool_eval::PoolEvalError,
    > {
        use crate::numeric_scalar::NumericScalar;
        use crate::numeric_tensor::{NumericTensor, TensorLayout};
        use crate::tensor_rank::DynRank;

        let data = &inputs[0];
        let target_shape_view = &inputs[1];
        let dtype = data.dtype();

        // Extract target shape.
        let target_shape: Vec<u64> = (0..target_shape_view.numel())
            .map(|i| target_shape_view.read_element(i).to_i64() as u64)
            .collect();
        let data_shape = data.shape();
        let data_rank = data_shape.len();
        let target_rank = target_shape.len();

        // Pad target with leading 1s.
        let rank_padding = data_rank.saturating_sub(target_rank);
        let mut padded_target = vec![1u64; rank_padding];
        padded_target.extend(&target_shape);

        // Find reduce axes.
        let reduce_axes: Vec<usize> = (0..data_rank)
            .filter(|&i| padded_target[i] == 1 && data_shape[i] > 1)
            .collect();

        // If no reduction needed, copy data and reshape.
        if reduce_axes.is_empty() {
            let layout = TensorLayout::<DynRank>::row_major(target_shape.clone(), dtype);
            let buf = pool
                .allocate(layout.buffer_size_bytes())
                .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
            let mut out = NumericTensor::from_parts(buf, layout);
            for i in 0..data.numel() {
                out.write_element(i, data.read_element(i));
            }
            return Ok(vec![out]);
        }

        // Reduce sum along reduce_axes with keepdims=true, then reshape.
        // Use padded_target as the intermediate shape (keepdims).
        let keepdims_shape: Vec<u64> = (0..data_rank)
            .map(|i| {
                if reduce_axes.contains(&i) {
                    1
                } else {
                    data_shape[i]
                }
            })
            .collect();
        let out_numel: usize = keepdims_shape.iter().product::<u64>() as usize;

        // Allocate intermediate in keepdims shape.
        let kd_layout = TensorLayout::<DynRank>::row_major(keepdims_shape.clone(), dtype);
        let kd_buf = pool
            .allocate(kd_layout.buffer_size_bytes())
            .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
        let mut kd_out = NumericTensor::from_parts(kd_buf, kd_layout);

        // Initialize to zero.
        for i in 0..out_numel {
            kd_out.write_element(i, NumericScalar::zero(dtype));
        }

        // Strides.
        let in_strides = {
            let mut s = vec![1usize; data_rank];
            for i in (0..data_rank.saturating_sub(1)).rev() {
                s[i] = s[i + 1] * data_shape[i + 1] as usize;
            }
            s
        };
        let out_strides = {
            let mut s = vec![1usize; data_rank];
            for i in (0..data_rank.saturating_sub(1)).rev() {
                s[i] = s[i + 1] * keepdims_shape[i + 1] as usize;
            }
            s
        };

        // Accumulate.
        for flat_in in 0..data.numel() {
            let mut rem = flat_in;
            let mut out_flat = 0usize;
            for i in 0..data_rank {
                let idx = rem / in_strides[i];
                rem %= in_strides[i];
                if !reduce_axes.contains(&i) {
                    out_flat += idx * out_strides[i];
                }
                // For reduce axes, keepdims coord is always 0 (stride * 0 = 0).
            }
            let val = data.read_element(flat_in);
            let cur = kd_out.read_element(out_flat);
            kd_out.write_element(out_flat, cur.add(val));
        }

        // Reshape to target_shape if needed.
        if keepdims_shape.iter().map(|&x| x as u64).collect::<Vec<_>>() != target_shape {
            let layout = TensorLayout::<DynRank>::row_major(target_shape, dtype);
            let buf = pool
                .allocate(layout.buffer_size_bytes())
                .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
            let mut out = NumericTensor::from_parts(buf, layout);
            for i in 0..out_numel {
                out.write_element(i, kd_out.read_element(i));
            }
            Ok(vec![out])
        } else {
            Ok(vec![kd_out])
        }
    }
}
