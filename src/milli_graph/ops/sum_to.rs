use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

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
    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        let data = &inputs[&self.data];
        let target_shape_tensor = &inputs[&self.target_shape];
        let target_shape: Vec<i64> = target_shape_tensor
            .clone()
            .try_to_rank::<P1>()?
            .try_into()?;
        let target_shape_usize: Vec<usize> = target_shape.iter().map(|&x| x as usize).collect();

        let data_shape = data.shape();
        let data_rank = data_shape.len();
        let target_rank = target_shape_usize.len();

        // Fast path: shapes already match
        if data_rank == target_rank
            && data_shape
                .iter()
                .zip(target_shape_usize.iter())
                .all(|(&d, &t)| d as usize == t)
        {
            return Ok(Box::new([(self.output, data.clone())].into_iter()));
        }

        // Pad target_shape with leading 1s to match data rank
        let rank_padding = data_rank.saturating_sub(target_rank);
        let mut padded_target = vec![1usize; rank_padding];
        padded_target.extend(&target_shape_usize);

        // Find axes where padded_target[i] == 1 and data_shape[i] > 1
        let mut reduce_axes: Vec<usize> = Vec::new();
        for i in 0..data_rank {
            if padded_target[i] == 1 && data_shape[i] as usize > 1 {
                reduce_axes.push(i);
            }
        }

        // ReduceSum along broadcast axes with keepdims=true (preserves rank)
        let mut result = data.clone();
        if !reduce_axes.is_empty() {
            result = result.reduce_sum(
                reduce_axes,
                true,
                super::AccumulationMode::default(),
                backend,
            )?;
        }

        // Reshape to target shape (removes any rank-padded leading dims)
        let final_shape: Vec<u64> = target_shape_usize.iter().map(|&x| x as u64).collect();
        if result.shape() != final_shape {
            result = result.reshape(final_shape, backend)?;
        }

        Ok(Box::new([(self.output, result)].into_iter()))
    }
}
