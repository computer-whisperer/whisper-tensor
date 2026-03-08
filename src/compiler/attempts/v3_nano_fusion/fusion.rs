//! Loop-fusion pass over crystallized nano-op loops.
//!
//! This pass keeps the nano-op pipeline from v1, but fuses adjacent
//! crystal loops when the consumer loop reloads values produced by
//! the previous loop with the same per-iteration index mapping.

use crate::compiler::attempts::v1_scalar_crystal::crystal::{CrystalLoop, CrystalOp, LoopBodyOp};
use crate::graph::GlobalId;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FusionStats {
    pub input_ops: usize,
    pub output_ops: usize,
    pub input_loops: usize,
    pub output_loops: usize,
    pub fused_pairs: usize,
    pub eliminated_loads: usize,
}

#[derive(Debug, Clone, Copy)]
struct ProducedValue {
    base_index: usize,
    step: isize,
    value_ref: usize,
}

pub fn fuse_crystal_ops(crystal_ops: &[CrystalOp]) -> Vec<CrystalOp> {
    fuse_with_stats(crystal_ops).0
}

pub fn fuse_with_stats(crystal_ops: &[CrystalOp]) -> (Vec<CrystalOp>, FusionStats) {
    let mut fused = Vec::with_capacity(crystal_ops.len());
    let mut idx = 0;
    let mut fused_pairs = 0;
    let mut eliminated_loads = 0;

    while idx < crystal_ops.len() {
        match &crystal_ops[idx] {
            CrystalOp::Loop(loop_seed) => {
                let mut acc = loop_seed.clone();
                idx += 1;

                while idx < crystal_ops.len() {
                    let next_loop = match &crystal_ops[idx] {
                        CrystalOp::Loop(l) => l,
                        CrystalOp::Scalar(_) => break,
                    };

                    if let Some((merged, replaced_loads)) = try_fuse_loops(&acc, next_loop) {
                        acc = merged;
                        fused_pairs += 1;
                        eliminated_loads += replaced_loads;
                        idx += 1;
                    } else {
                        break;
                    }
                }

                fused.push(CrystalOp::Loop(acc));
            }
            CrystalOp::Scalar(op) => {
                fused.push(CrystalOp::Scalar(op.clone()));
                idx += 1;
            }
        }
    }

    let input_loops = crystal_ops
        .iter()
        .filter(|op| matches!(op, CrystalOp::Loop(_)))
        .count();
    let output_loops = fused
        .iter()
        .filter(|op| matches!(op, CrystalOp::Loop(_)))
        .count();

    let stats = FusionStats {
        input_ops: crystal_ops.len(),
        output_ops: fused.len(),
        input_loops,
        output_loops,
        fused_pairs,
        eliminated_loads,
    };

    (fused, stats)
}

fn try_fuse_loops(lhs: &CrystalLoop, rhs: &CrystalLoop) -> Option<(CrystalLoop, usize)> {
    if lhs.count != rhs.count {
        return None;
    }

    let produced = collect_produced_values(lhs);

    let mut merged_body = lhs.body.clone();
    let mut rhs_to_merged: Vec<Option<usize>> = vec![None; rhs.body.len()];
    let mut replaced_loads = 0usize;

    for (idx, op) in rhs.body.iter().enumerate() {
        let merged_op = match op {
            LoopBodyOp::Load {
                tensor,
                base_index,
                step,
            } => {
                if let Some(value_ref) = find_produced_value(&produced, *tensor, *base_index, *step)
                {
                    rhs_to_merged[idx] = Some(value_ref);
                    replaced_loads += 1;
                    None
                } else {
                    Some(LoopBodyOp::Load {
                        tensor: *tensor,
                        base_index: *base_index,
                        step: *step,
                    })
                }
            }
            LoopBodyOp::Store {
                tensor,
                base_index,
                step,
                value_ref,
            } => Some(LoopBodyOp::Store {
                tensor: *tensor,
                base_index: *base_index,
                step: *step,
                value_ref: remap_ref(&rhs_to_merged, *value_ref)?,
            }),
            LoopBodyOp::BinOp { op, a_ref, b_ref } => Some(LoopBodyOp::BinOp {
                op: *op,
                a_ref: remap_ref(&rhs_to_merged, *a_ref)?,
                b_ref: remap_ref(&rhs_to_merged, *b_ref)?,
            }),
            LoopBodyOp::UnaryOp { op, input_ref } => Some(LoopBodyOp::UnaryOp {
                op: *op,
                input_ref: remap_ref(&rhs_to_merged, *input_ref)?,
            }),
            LoopBodyOp::Literal { value } => Some(LoopBodyOp::Literal { value: *value }),
        };

        if let Some(op) = merged_op {
            let new_idx = merged_body.len();
            merged_body.push(op);
            rhs_to_merged[idx] = Some(new_idx);
        }
    }

    if replaced_loads == 0 {
        return None;
    }

    let mut merged_nano_ops = lhs.nano_ops.clone();
    merged_nano_ops.extend(rhs.nano_ops.iter().cloned());

    Some((
        CrystalLoop {
            nano_ops: merged_nano_ops,
            count: lhs.count,
            body_len: merged_body.len(),
            body: merged_body,
        },
        replaced_loads,
    ))
}

fn remap_ref(remap: &[Option<usize>], old_ref: usize) -> Option<usize> {
    remap.get(old_ref).copied().flatten()
}

fn collect_produced_values(loop_op: &CrystalLoop) -> HashMap<GlobalId, Vec<ProducedValue>> {
    let mut out: HashMap<GlobalId, Vec<ProducedValue>> = HashMap::new();
    for op in &loop_op.body {
        if let LoopBodyOp::Store {
            tensor,
            base_index,
            step,
            value_ref,
        } = op
        {
            out.entry(*tensor).or_default().push(ProducedValue {
                base_index: *base_index,
                step: *step,
                value_ref: *value_ref,
            });
        }
    }
    out
}

fn find_produced_value(
    produced: &HashMap<GlobalId, Vec<ProducedValue>>,
    tensor: GlobalId,
    base_index: usize,
    step: isize,
) -> Option<usize> {
    produced.get(&tensor).and_then(|entries| {
        entries
            .iter()
            .rev()
            .find(|e| e.base_index == base_index && e.step == step)
            .map(|e| e.value_ref)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::attempts::v1_scalar_crystal::crystal;
    use crate::compiler::attempts::v1_scalar_crystal::nano_op::NanoOpExpander;
    use crate::graph::GlobalId;
    use crate::milli_graph::MilliOpGraph;
    use crate::milli_graph::ops::{SimpleBinary, SimpleUnaryOp};

    #[test]
    fn test_fuse_mul_then_neg() {
        let mut rng = wyrand::WyRand::new(42);
        let ext_a = GlobalId::new(&mut rng);
        let ext_b = GlobalId::new(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let a = input_map[&ext_a];
        let b = input_map[&ext_b];
        let mul = SimpleBinary::mul(&mut graph, a, b, &mut rng);
        let neg = SimpleUnaryOp::neg(&mut graph, mul, &mut rng);

        let mut shapes = HashMap::new();
        for &tid in &[a, b, mul, neg] {
            shapes.insert(tid, vec![16]);
        }

        let mut expander = NanoOpExpander::new(shapes);
        let nano_ops = expander.expand(&graph).unwrap();
        let crystal_ops = crystal::crystallize(&nano_ops);

        let (fused, stats) = fuse_with_stats(&crystal_ops);

        assert_eq!(stats.input_loops, 2);
        assert_eq!(stats.output_loops, 1);
        assert_eq!(stats.fused_pairs, 1);
        assert_eq!(stats.eliminated_loads, 1);
        assert_eq!(fused.len(), 1);

        let loop_op = match &fused[0] {
            CrystalOp::Loop(l) => l,
            CrystalOp::Scalar(_) => panic!("expected fused loop"),
        };

        let num_loads = loop_op
            .body
            .iter()
            .filter(|op| matches!(op, LoopBodyOp::Load { .. }))
            .count();
        let num_stores = loop_op
            .body
            .iter()
            .filter(|op| matches!(op, LoopBodyOp::Store { .. }))
            .count();

        assert_eq!(num_loads, 2);
        assert_eq!(num_stores, 2);
        assert_eq!(loop_op.body_len, 6);
    }

    #[test]
    fn test_fuse_three_stage_chain() {
        let mut rng = wyrand::WyRand::new(7);
        let ext_a = GlobalId::new(&mut rng);
        let ext_b = GlobalId::new(&mut rng);
        let ext_c = GlobalId::new(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b, ext_c], &mut rng);
        let a = input_map[&ext_a];
        let b = input_map[&ext_b];
        let c = input_map[&ext_c];
        let mul = SimpleBinary::mul(&mut graph, a, b, &mut rng);
        let add = SimpleBinary::add(&mut graph, mul, c, &mut rng);
        let neg = SimpleUnaryOp::neg(&mut graph, add, &mut rng);

        let mut shapes = HashMap::new();
        for &tid in &[a, b, c, mul, add, neg] {
            shapes.insert(tid, vec![8]);
        }

        let mut expander = NanoOpExpander::new(shapes);
        let nano_ops = expander.expand(&graph).unwrap();
        let crystal_ops = crystal::crystallize(&nano_ops);

        let (fused, stats) = fuse_with_stats(&crystal_ops);

        assert_eq!(stats.input_loops, 3);
        assert_eq!(stats.output_loops, 1);
        assert_eq!(stats.fused_pairs, 2);
        assert_eq!(stats.eliminated_loads, 2);
        assert_eq!(fused.len(), 1);

        let loop_op = match &fused[0] {
            CrystalOp::Loop(l) => l,
            CrystalOp::Scalar(_) => panic!("expected fused loop"),
        };
        assert_eq!(loop_op.body_len, 9);
    }

    #[test]
    fn test_no_fuse_when_loop_counts_differ() {
        let t0 = GlobalId(1);
        let t1 = GlobalId(2);
        let t2 = GlobalId(3);

        let lhs = CrystalLoop {
            nano_ops: Vec::new(),
            count: 4,
            body_len: 2,
            body: vec![
                LoopBodyOp::Load {
                    tensor: t0,
                    base_index: 0,
                    step: 1,
                },
                LoopBodyOp::Store {
                    tensor: t1,
                    base_index: 0,
                    step: 1,
                    value_ref: 0,
                },
            ],
        };
        let rhs = CrystalLoop {
            nano_ops: Vec::new(),
            count: 5,
            body_len: 2,
            body: vec![
                LoopBodyOp::Load {
                    tensor: t1,
                    base_index: 0,
                    step: 1,
                },
                LoopBodyOp::Store {
                    tensor: t2,
                    base_index: 0,
                    step: 1,
                    value_ref: 0,
                },
            ],
        };

        let ops = vec![CrystalOp::Loop(lhs), CrystalOp::Loop(rhs)];
        let (fused, stats) = fuse_with_stats(&ops);

        assert_eq!(fused.len(), 2);
        assert_eq!(stats.fused_pairs, 0);
        assert_eq!(stats.eliminated_loads, 0);
    }
}
