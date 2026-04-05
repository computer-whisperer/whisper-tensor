use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::pool::Pool;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// ReverseSequence: reverses variable-length slices along a time axis,
/// independently for each element along a batch axis.
///
/// Output shape = input shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReverseSequence {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    input: GlobalId,
    sequence_lens: GlobalId,
    batch_axis: i64,
    time_axis: i64,
}

impl ReverseSequence {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        input: GlobalId,
        sequence_lens: GlobalId,
        batch_axis: i64,
        time_axis: i64,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            label: None,
            output,
            input,
            sequence_lens,
            batch_axis,
            time_axis,
        };
        graph.push_op(AnyMilliOp::ReverseSequence(node));
        output
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.input, map);
        super::remap(&mut self.sequence_lens, map);
    }
}

impl Node for ReverseSequence {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "ReverseSequence".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.input, self.sequence_lens].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl MilliOp for ReverseSequence {
    fn infer<'a, 'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'a, 'p, P>>,
        _symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        _pool: &'p P,
    ) -> Result<Vec<(GlobalId, crate::tensor_info::TensorInfo<'a, 'p, P>)>, MilliOpGraphError>
    where
        'p: 'a,
    {
        // Output shape = input shape.
        let input_info = known_inputs
            .get(&self.input)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let ranked = input_info
            .as_ranked()
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let shape = ranked.shape();
        let out_info = crate::tensor_info::TensorInfo::from_dtype_and_shape_scalars(
            input_info.dtype(),
            &shape,
        );
        Ok(vec![(self.output, out_info)])
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

        let input = &inputs[0];
        let seq_lens = &inputs[1];
        let shape = input.shape();
        let rank = shape.len();
        let dtype = input.dtype();

        let batch_axis = if self.batch_axis < 0 {
            (rank as i64 + self.batch_axis) as usize
        } else {
            self.batch_axis as usize
        };
        let time_axis = if self.time_axis < 0 {
            (rank as i64 + self.time_axis) as usize
        } else {
            self.time_axis as usize
        };

        let total: usize = shape.iter().product::<u64>() as usize;

        // Compute strides.
        let mut strides = vec![1usize; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1] as usize;
        }

        let layout = TensorLayout::<DynRank>::row_major(shape.to_vec(), dtype);
        let buf = pool
            .allocate(layout.buffer_size_bytes())
            .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
        let mut out = NumericTensor::from_parts(buf, layout);

        for flat_idx in 0..total {
            let mut remaining = flat_idx;
            let mut coords = vec![0usize; rank];
            for d in 0..rank {
                coords[d] = remaining / strides[d];
                remaining %= strides[d];
            }

            let batch_idx = coords[batch_axis];
            let time_idx = coords[time_axis];
            let seq_len = seq_lens.read_element(batch_idx).to_i64() as usize;

            let src_time = if time_idx < seq_len {
                seq_len - 1 - time_idx
            } else {
                time_idx
            };

            let mut src_flat = 0;
            for d in 0..rank {
                let c = if d == time_axis { src_time } else { coords[d] };
                src_flat += c * strides[d];
            }

            out.write_element(flat_idx, input.read_element(src_flat));
        }

        Ok(vec![out])
    }
}
