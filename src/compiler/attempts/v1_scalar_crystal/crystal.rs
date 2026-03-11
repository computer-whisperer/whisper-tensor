//! Crystal pass: re-vectorization of nano op streams.
//!
//! Scans the nano op stream for repeating patterns with incrementing indices
//! and compresses them into loop crystals. Each crystal stores the full verbose
//! list of nano ops it absorbed (no ID compression yet).

use super::nano_op::{NanoOp, ScalarBinOp, ScalarUnaryOp};
use crate::graph::GlobalId;
use std::collections::HashMap;

/// A crystallized operation — either a loop or a scalar fallback.
#[derive(Debug, Clone)]
pub enum CrystalOp {
    /// Single scalar nano op (not part of any detected pattern).
    Scalar(NanoOp),
    /// A loop crystal — a repeating pattern detected in the stream.
    Loop(CrystalLoop),
}

/// A loop crystal — a repeating pattern of nano ops.
#[derive(Debug, Clone)]
pub struct CrystalLoop {
    /// The original nano ops this loop absorbed (verbose, uncompressed).
    pub nano_ops: Vec<NanoOp>,
    /// Number of loop iterations.
    pub count: usize,
    /// Number of nano ops per iteration.
    pub body_len: usize,
    /// Loop body template (how to generate code for one iteration).
    pub body: Vec<LoopBodyOp>,
}

/// One operation in the loop body template.
/// References to other body ops use indices into the body array.
#[derive(Debug, Clone)]
pub enum LoopBodyOp {
    /// Load from tensor: element index = base_index + i * step
    Load {
        tensor: GlobalId,
        base_index: usize,
        step: isize,
    },
    /// Store to tensor: element index = base_index + i * step
    Store {
        tensor: GlobalId,
        base_index: usize,
        step: isize,
        /// Index into body array of the value to store.
        value_ref: usize,
    },
    /// Binary scalar op.
    BinOp {
        op: ScalarBinOp,
        /// Index into body array of operand a.
        a_ref: usize,
        /// Index into body array of operand b.
        b_ref: usize,
    },
    /// Unary scalar op.
    UnaryOp {
        op: ScalarUnaryOp,
        /// Index into body array of the input.
        input_ref: usize,
    },
    /// Literal constant (same every iteration).
    Literal { value: f64 },
}

/// Run the crystal pass on a nano op stream.
///
/// Scans for repeating groups with consistent index strides and
/// compresses them into loop crystals. Anything that doesn't match
/// a pattern falls through as a scalar op.
pub fn crystallize(nano_ops: &[NanoOp]) -> Vec<CrystalOp> {
    let mut result = Vec::new();
    let mut pos = 0;

    while pos < nano_ops.len() {
        if let Some((crystal, consumed)) = try_detect_loop(&nano_ops[pos..]) {
            result.push(CrystalOp::Loop(crystal));
            pos += consumed;
        } else {
            result.push(CrystalOp::Scalar(nano_ops[pos].clone()));
            pos += 1;
        }
    }

    result
}

/// Try to detect a repeating loop pattern starting at the given position.
/// Returns the crystal and the number of nano ops consumed.
fn try_detect_loop(ops: &[NanoOp]) -> Option<(CrystalLoop, usize)> {
    // Try candidate body lengths, smallest first.
    // 3 = unary (load, op, store)
    // 4 = binary (load, load, op, store)
    // Could extend to larger bodies for fused patterns later.
    for body_len in [3, 4, 5, 6, 7, 8] {
        if let Some(result) = try_body_length(ops, body_len) {
            return Some(result);
        }
    }
    None
}

/// Try a specific body length and see if the pattern repeats.
fn try_body_length(ops: &[NanoOp], body_len: usize) -> Option<(CrystalLoop, usize)> {
    if ops.len() < body_len * 2 {
        return None;
    }

    let template = &ops[0..body_len];
    let next = &ops[body_len..body_len * 2];

    // Compute index deltas between iteration 0 and iteration 1
    let deltas = compute_deltas(template, next)?;

    // Count how many consecutive iterations match
    let mut count = 2;
    loop {
        let start = count * body_len;
        let end = start + body_len;
        if end > ops.len() {
            break;
        }
        let iteration = &ops[start..end];
        if !check_iteration(template, iteration, &deltas, count) {
            break;
        }
        count += 1;
    }

    // Build the loop body template
    let body = build_loop_body(template, &deltas)?;
    let consumed = count * body_len;

    Some((
        CrystalLoop {
            nano_ops: ops[..consumed].to_vec(),
            count,
            body_len,
            body,
        },
        consumed,
    ))
}

/// Compute per-op index deltas between two consecutive iterations.
/// Returns None if the ops don't have compatible structure.
fn compute_deltas(iter0: &[NanoOp], iter1: &[NanoOp]) -> Option<Vec<isize>> {
    if iter0.len() != iter1.len() {
        return None;
    }

    let mut deltas = Vec::with_capacity(iter0.len());
    for (a, b) in iter0.iter().zip(iter1.iter()) {
        let delta = match (a, b) {
            (
                NanoOp::Load {
                    tensor: t0,
                    flat_index: i0,
                    ..
                },
                NanoOp::Load {
                    tensor: t1,
                    flat_index: i1,
                    ..
                },
            ) => {
                if t0 != t1 {
                    return None;
                }
                *i1 as isize - *i0 as isize
            }
            (
                NanoOp::Store {
                    tensor: t0,
                    flat_index: i0,
                    ..
                },
                NanoOp::Store {
                    tensor: t1,
                    flat_index: i1,
                    ..
                },
            ) => {
                if t0 != t1 {
                    return None;
                }
                *i1 as isize - *i0 as isize
            }
            (NanoOp::BinOp { op: op0, .. }, NanoOp::BinOp { op: op1, .. }) => {
                if op0 != op1 {
                    return None;
                }
                0
            }
            (NanoOp::UnaryOp { op: op0, .. }, NanoOp::UnaryOp { op: op1, .. }) => {
                if op0 != op1 {
                    return None;
                }
                0
            }
            (NanoOp::Literal { value: v0, .. }, NanoOp::Literal { value: v1, .. }) => {
                if v0.to_bits() != v1.to_bits() {
                    return None;
                }
                0
            }
            _ => return None, // type mismatch
        };
        deltas.push(delta);
    }
    Some(deltas)
}

/// Check that iteration N matches the template with expected indices.
fn check_iteration(
    template: &[NanoOp],
    iteration: &[NanoOp],
    deltas: &[isize],
    iter_num: usize,
) -> bool {
    if template.len() != iteration.len() {
        return false;
    }

    for (i, (tmpl, actual)) in template.iter().zip(iteration.iter()).enumerate() {
        let expected_offset = deltas[i] * iter_num as isize;
        let ok = match (tmpl, actual) {
            (
                NanoOp::Load {
                    tensor: t0,
                    flat_index: i0,
                    ..
                },
                NanoOp::Load {
                    tensor: t1,
                    flat_index: i1,
                    ..
                },
            ) => *t0 == *t1 && *i1 as isize == *i0 as isize + expected_offset,
            (
                NanoOp::Store {
                    tensor: t0,
                    flat_index: i0,
                    ..
                },
                NanoOp::Store {
                    tensor: t1,
                    flat_index: i1,
                    ..
                },
            ) => *t0 == *t1 && *i1 as isize == *i0 as isize + expected_offset,
            (NanoOp::BinOp { op: op0, .. }, NanoOp::BinOp { op: op1, .. }) => op0 == op1,
            (NanoOp::UnaryOp { op: op0, .. }, NanoOp::UnaryOp { op: op1, .. }) => op0 == op1,
            (NanoOp::Literal { value: v0, .. }, NanoOp::Literal { value: v1, .. }) => {
                v0.to_bits() == v1.to_bits()
            }
            _ => false,
        };
        if !ok {
            return false;
        }
    }
    true
}

/// Build the loop body template from the first iteration's nano ops.
fn build_loop_body(template: &[NanoOp], deltas: &[isize]) -> Option<Vec<LoopBodyOp>> {
    let mut body = Vec::with_capacity(template.len());
    // Map NanoValue id -> body op index (for resolving references)
    let mut nano_to_body: HashMap<u64, usize> = HashMap::new();

    for (i, op) in template.iter().enumerate() {
        let body_op = match op {
            NanoOp::Load {
                dst,
                tensor,
                flat_index,
            } => {
                nano_to_body.insert(dst.0, i);
                LoopBodyOp::Load {
                    tensor: *tensor,
                    base_index: *flat_index,
                    step: deltas[i],
                }
            }
            NanoOp::Store {
                tensor,
                flat_index,
                src,
            } => {
                let value_ref = *nano_to_body.get(&src.0)?;
                LoopBodyOp::Store {
                    tensor: *tensor,
                    base_index: *flat_index,
                    step: deltas[i],
                    value_ref,
                }
            }
            NanoOp::BinOp { dst, op, a, b } => {
                let a_ref = *nano_to_body.get(&a.0)?;
                let b_ref = *nano_to_body.get(&b.0)?;
                nano_to_body.insert(dst.0, i);
                LoopBodyOp::BinOp {
                    op: *op,
                    a_ref,
                    b_ref,
                }
            }
            NanoOp::UnaryOp { dst, op, input } => {
                let input_ref = *nano_to_body.get(&input.0)?;
                nano_to_body.insert(dst.0, i);
                LoopBodyOp::UnaryOp { op: *op, input_ref }
            }
            NanoOp::Literal { dst, value } => {
                nano_to_body.insert(dst.0, i);
                LoopBodyOp::Literal { value: *value }
            }
        };
        body.push(body_op);
    }
    Some(body)
}

/// Summary statistics from crystallization.
#[derive(Debug)]
pub struct CrystallizeStats {
    pub total_crystal_ops: usize,
    pub num_loops: usize,
    pub num_scalars: usize,
    pub nano_ops_in_loops: usize,
    pub nano_ops_as_scalars: usize,
}

pub fn stats(crystal_ops: &[CrystalOp]) -> CrystallizeStats {
    let mut num_loops = 0;
    let mut num_scalars = 0;
    let mut nano_ops_in_loops = 0;
    let mut nano_ops_as_scalars = 0;

    for op in crystal_ops {
        match op {
            CrystalOp::Scalar(_) => {
                num_scalars += 1;
                nano_ops_as_scalars += 1;
            }
            CrystalOp::Loop(l) => {
                num_loops += 1;
                nano_ops_in_loops += l.nano_ops.len();
            }
        }
    }

    CrystallizeStats {
        total_crystal_ops: crystal_ops.len(),
        num_loops,
        num_scalars,
        nano_ops_in_loops,
        nano_ops_as_scalars,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::nano_op::NanoOpExpander;
    use crate::graph::GlobalId;
    use crate::milli_graph::MilliOpGraph;
    use crate::milli_graph::ops::SimpleBinary;

    #[test]
    fn test_crystallize_contiguous_add() {
        let mut rng = wyrand::WyRand::new(42);
        let ext_a = GlobalId::new(&mut rng);
        let ext_b = GlobalId::new(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let int_a = input_map[&ext_a];
        let int_b = input_map[&ext_b];
        let int_out = SimpleBinary::add(&mut graph, int_a, int_b, &mut rng);

        let mut shapes = HashMap::new();
        shapes.insert(int_a, vec![8]);
        shapes.insert(int_b, vec![8]);
        shapes.insert(int_out, vec![8]);

        let mut expander = NanoOpExpander::new(shapes);
        let nano_ops = expander.expand(&graph).unwrap();
        assert_eq!(nano_ops.len(), 32); // 8 × (load, load, binop, store)

        let crystal_ops = crystallize(&nano_ops);
        let s = stats(&crystal_ops);

        // Should produce 1 loop, 0 scalars
        assert_eq!(s.num_loops, 1);
        assert_eq!(s.num_scalars, 0);
        assert_eq!(s.nano_ops_in_loops, 32);

        // Check loop properties
        if let CrystalOp::Loop(l) = &crystal_ops[0] {
            assert_eq!(l.count, 8);
            assert_eq!(l.body_len, 4);
            assert_eq!(l.body.len(), 4);
            // All strides should be 1 (contiguous)
            for body_op in &l.body {
                match body_op {
                    LoopBodyOp::Load { step, .. } => assert_eq!(*step, 1),
                    LoopBodyOp::Store { step, .. } => assert_eq!(*step, 1),
                    _ => {}
                }
            }
        } else {
            panic!("Expected loop");
        }
    }

    #[test]
    fn test_crystallize_broadcast() {
        let mut rng = wyrand::WyRand::new(42);
        let ext_a = GlobalId::new(&mut rng);
        let ext_b = GlobalId::new(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let int_a = input_map[&ext_a];
        let int_b = input_map[&ext_b];
        let int_out = SimpleBinary::add(&mut graph, int_a, int_b, &mut rng);

        // a=[8], b=[1] (broadcast b)
        let mut shapes = HashMap::new();
        shapes.insert(int_a, vec![8]);
        shapes.insert(int_b, vec![1]);
        shapes.insert(int_out, vec![8]);

        let mut expander = NanoOpExpander::new(shapes);
        let nano_ops = expander.expand(&graph).unwrap();

        let crystal_ops = crystallize(&nano_ops);
        let s = stats(&crystal_ops);

        assert_eq!(s.num_loops, 1);
        assert_eq!(s.num_scalars, 0);

        if let CrystalOp::Loop(l) = &crystal_ops[0] {
            assert_eq!(l.count, 8);
            // Check that b has stride 0 (broadcast)
            if let LoopBodyOp::Load { step, tensor, .. } = &l.body[1]
                && *tensor == int_b
            {
                assert_eq!(*step, 0, "broadcast load should have step=0");
            }
        }
    }

    #[test]
    fn test_crystallize_chain() {
        // Two ops in sequence: mul then neg
        // Should produce two separate loops
        let mut rng = wyrand::WyRand::new(42);
        let ext_a = GlobalId::new(&mut rng);
        let ext_b = GlobalId::new(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let int_a = input_map[&ext_a];
        let int_b = input_map[&ext_b];
        let mul_out = SimpleBinary::mul(&mut graph, int_a, int_b, &mut rng);
        let neg_out = crate::milli_graph::ops::SimpleUnaryOp::neg(&mut graph, mul_out, &mut rng);

        let mut shapes = HashMap::new();
        shapes.insert(int_a, vec![16]);
        shapes.insert(int_b, vec![16]);
        shapes.insert(mul_out, vec![16]);
        shapes.insert(neg_out, vec![16]);

        let mut expander = NanoOpExpander::new(shapes);
        let nano_ops = expander.expand(&graph).unwrap();
        // 16×4 (mul) + 16×3 (neg) = 112
        assert_eq!(nano_ops.len(), 112);

        let crystal_ops = crystallize(&nano_ops);
        let s = stats(&crystal_ops);

        assert_eq!(s.num_loops, 2);
        assert_eq!(s.num_scalars, 0);
        assert_eq!(s.nano_ops_in_loops, 112);
    }
}
