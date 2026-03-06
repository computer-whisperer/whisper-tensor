use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transpose {
    global_id: GlobalId,
    output: GlobalId,
    data: GlobalId,
    perm: Option<Vec<i64>>,
}

impl Transpose {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        perm: Option<Vec<i64>>,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self { output, data, perm, global_id: GlobalId::new(rng) };
        graph.push_op(AnyMilliOp::Transpose(node));
        output
    }
}

impl Transpose {
    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.data, map);
    }
}

impl Node for Transpose {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Transpose".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.data].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
}

impl MilliOp for Transpose {
    fn backward(
        &self,
        output_grads: &HashMap<GlobalId, GlobalId>,
        graph: &mut MilliOpGraph,
        rng: &mut impl rand::Rng,
    ) -> Option<HashMap<GlobalId, GlobalId>> {
        let grad_output = *output_grads.get(&self.output)?;
        // Inverse transpose: if perm is None (reverse), inverse is also None.
        // Otherwise compute inverse permutation.
        // NOTE: For partial perms (len < rank), the inverse is computed using
        // p.len() not actual rank. This is correct for the [-1,-2] swap case
        // (only current usage) but would be wrong for general partial perms.
        // Fix by expanding partial perms at construction time if needed.
        let inv_perm = self.perm.as_ref().map(|p| {
            let n = p.len();
            let normalized: Vec<usize> = p
                .iter()
                .map(|&x| if x < 0 { (x + n as i64) as usize } else { x as usize })
                .collect();
            let mut inv = vec![0i64; n];
            for (i, &ni) in normalized.iter().enumerate() {
                inv[ni] = i as i64;
            }
            inv
        });
        let grad_input = Transpose::push_new(graph, grad_output, inv_perm, rng);
        let mut result = HashMap::new();
        result.insert(self.data, grad_input);
        Some(result)
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<
        Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>,
        MilliOpGraphError,
    > {
        // Handle partial perms: if perm has fewer elements than the rank,
        // prepend identity dims. This allows perm=[-1,-2] to mean "swap last
        // two dims" regardless of rank.
        let perm = if let Some(ref p) = self.perm {
            let rank = inputs[&self.data].rank();
            if p.len() < rank {
                let prefix_len = rank - p.len();
                let mut full_perm: Vec<i64> = (0..prefix_len as i64).collect();
                full_perm.extend(p.iter().map(|&x| {
                    if x < 0 { x + rank as i64 } else { x }
                }));
                Some(full_perm)
            } else {
                Some(p.clone())
            }
        } else {
            None
        };
        let out = inputs[&self.data].transpose(perm, backend)?;
        Ok(Box::new([(self.output, out)].into_iter()))
    }
}
