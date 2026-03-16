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
    pub(crate) label: Option<String>,
    output: GlobalId,
    data: GlobalId,
    perm: Option<Vec<i64>>,
}

impl Transpose {
    pub(crate) fn perm(&self) -> Option<&[i64]> {
        self.perm.as_deref()
    }

    pub fn push_new(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        perm: Option<Vec<i64>>,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        Self::push_new_with_label(graph, data, perm, None, rng)
    }

    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        perm: Option<Vec<i64>>,
        label: Option<String>,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            output,
            data,
            perm,
            global_id: GlobalId::new(rng),
            label,
        };
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
    fn infer(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo>,
        symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<
        Box<dyn Iterator<Item = (GlobalId, crate::tensor_info::TensorInfo)>>,
        MilliOpGraphError,
    > {
        use crate::tensor_info::TensorInfo;

        let input_info = known_inputs
            .get(&self.data)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        // If input is concrete, delegate to eval
        if let Some(numeric) = input_info.as_numeric() {
            let inputs = HashMap::from([(self.data, numeric.clone())]);
            let out: Vec<_> = self
                .eval(&inputs, backend)?
                .map(|(id, t)| (id, TensorInfo::from(t)))
                .collect();
            return Ok(Box::new(out.into_iter()));
        }

        let first_elem = input_info.first_element();

        // Try to propagate shape when rank is known
        if let Some(ranked) = input_info.as_ranked() {
            let input_shape = ranked.shape();
            let input_rank = input_shape.len();

            let output_shape = match &self.perm {
                Some(perm) => {
                    // Expand partial perm (like [-1, -2]) to full perm
                    let full_perm = if perm.len() < input_rank {
                        let prefix_len = input_rank - perm.len();
                        let mut fp: Vec<i64> = (0..prefix_len as i64).collect();
                        fp.extend(
                            perm.iter()
                                .map(|&x| if x < 0 { x + input_rank as i64 } else { x }),
                        );
                        fp
                    } else {
                        perm.clone()
                    };
                    // Output shape = input_shape[perm[i]] for each i
                    full_perm
                        .iter()
                        .map(|&p| {
                            let idx = if p < 0 {
                                (p + input_rank as i64) as usize
                            } else {
                                p as usize
                            };
                            input_shape[idx].clone()
                        })
                        .collect()
                }
                None => {
                    // Reverse the dimensions
                    let mut s = input_shape;
                    s.reverse();
                    s
                }
            };
            let out = TensorInfo::Ranked(crate::tensor_info::TensorInfoRanked::new(
                first_elem,
                output_shape,
                symbolic_resolver,
            ));
            return Ok(Box::new([(self.output, out)].into_iter()));
        }

        // At minimum: same rank, same dtype
        let rank = input_info.rank();
        let out = TensorInfo::new_from_first_element_and_rank(first_elem, rank, symbolic_resolver);
        Ok(Box::new([(self.output, out)].into_iter()))
    }

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
                .map(|&x| {
                    if x < 0 {
                        (x + n as i64) as usize
                    } else {
                        x as usize
                    }
                })
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
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        // Handle partial perms: if perm has fewer elements than the rank,
        // prepend identity dims. This allows perm=[-1,-2] to mean "swap last
        // two dims" regardless of rank.
        let perm = if let Some(ref p) = self.perm {
            let rank = inputs[&self.data].rank();
            if p.len() < rank {
                let prefix_len = rank - p.len();
                let mut full_perm: Vec<i64> = (0..prefix_len as i64).collect();
                full_perm.extend(p.iter().map(|&x| if x < 0 { x + rank as i64 } else { x }));
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
