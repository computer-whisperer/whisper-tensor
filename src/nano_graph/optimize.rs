//! Optimization passes on the NanoGraph.
//!
//! Each pass takes a `NanoGraph` by value and returns a new, optimized one.
//! Passes can be composed by chaining: `dce(strength_reduce(cse(graph)))`.

use std::collections::{HashMap, HashSet};

use crate::nano_graph::ops::{LiteralBits, ScalarBinOp, ScalarOp};
use crate::nano_graph::pattern::{BlockId, InputRef, NanoGraph, PatternBlock};

/// Result of running an optimization pass.
pub struct OptResult {
    pub graph: NanoGraph,
    /// Number of blocks removed / replaced by this pass.
    pub eliminated: usize,
}

// ---------------------------------------------------------------------------
// Dead-code elimination
// ---------------------------------------------------------------------------

/// Remove blocks not reachable backward from `output_blocks`.
pub fn dce(graph: NanoGraph) -> OptResult {
    let live = backward_reachable(&graph);
    let before = graph.len();
    let new_graph = rebuild_keeping(&graph, &live);
    let eliminated = before - new_graph.len();
    OptResult {
        graph: new_graph,
        eliminated,
    }
}

/// Compute the set of block ids reachable backward from output blocks.
fn backward_reachable(graph: &NanoGraph) -> HashSet<BlockId> {
    let mut visited = HashSet::new();
    let mut stack: Vec<BlockId> = graph.output_blocks.clone();
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        for producer in graph.producers_of(id) {
            if !visited.contains(&producer) {
                stack.push(producer);
            }
        }
    }
    visited
}

/// Rebuild graph keeping only blocks in `keep`, preserving topological order.
fn rebuild_keeping(old: &NanoGraph, keep: &HashSet<BlockId>) -> NanoGraph {
    let mut new_graph = old.new_with_same_symbolics();

    let mut id_map: HashMap<BlockId, BlockId> = HashMap::new();

    for block in old.blocks() {
        if !keep.contains(&block.id) {
            continue;
        }
        let new_inputs: Vec<InputRef> = block
            .inputs
            .iter()
            .map(|inp| InputRef {
                source_block: id_map[&inp.source_block],
                dim_map: inp.dim_map.clone(),
            })
            .collect();
        let new_id = new_graph.push(
            block.op,
            block.dtype,
            block.dims.clone(),
            new_inputs,
        );
        id_map.insert(block.id, new_id);
    }

    new_graph.output_blocks = old
        .output_blocks
        .iter()
        .filter_map(|id| id_map.get(id).copied())
        .collect();

    new_graph
}

// ---------------------------------------------------------------------------
// Strength reduction
// ---------------------------------------------------------------------------

/// Algebraic simplifications on binary ops with literal operands.
///
/// Rewrites:
/// - `x + 0` or `0 + x` → identity to x
/// - `x * 1` or `1 * x` → identity to x
/// - `x * 0` or `0 * x` → literal 0
/// - `x - 0` → identity to x
/// - `x / 1` → identity to x
pub fn strength_reduce(graph: NanoGraph) -> OptResult {
    let mut new_graph = graph.new_with_same_symbolics();
    let mut id_map: HashMap<BlockId, BlockId> = HashMap::new();
    let mut eliminated = 0usize;

    for block in graph.blocks() {
        let mapped_inputs: Vec<InputRef> = block
            .inputs
            .iter()
            .map(|inp| InputRef {
                source_block: id_map[&inp.source_block],
                dim_map: inp.dim_map.clone(),
            })
            .collect();

        if let Some(replacement) =
            try_strength_reduce(block, &mapped_inputs, &new_graph)
        {
            let new_id = new_graph.push(
                replacement.op,
                replacement.dtype,
                replacement.dims,
                replacement.inputs,
            );
            id_map.insert(block.id, new_id);
            eliminated += 1;
        } else {
            let new_id = new_graph.push(
                block.op,
                block.dtype,
                block.dims.clone(),
                mapped_inputs,
            );
            id_map.insert(block.id, new_id);
        }
    }

    new_graph.output_blocks = graph
        .output_blocks
        .iter()
        .filter_map(|id| id_map.get(id).copied())
        .collect();

    OptResult {
        graph: new_graph,
        eliminated,
    }
}

struct Replacement {
    op: ScalarOp,
    dtype: crate::dtype::DType,
    dims: Vec<crate::nano_graph::pattern::Dim>,
    inputs: Vec<InputRef>,
}

/// Check if a block is a literal with the given f64 value.
fn is_literal_value(graph: &NanoGraph, block_id: BlockId, value: f64) -> bool {
    if let Some(block) = graph.get(block_id)
        && let ScalarOp::Literal(lit) = &block.op
    {
        return lit.as_f64() == value;
    }
    false
}

/// Build an identity replacement that passes through one input unchanged,
/// reusing the original input ref's dim_map to preserve broadcast mappings.
fn identity_passthrough(block: &PatternBlock, original_input: &InputRef) -> Replacement {
    Replacement {
        op: ScalarOp::Identity,
        dtype: block.dtype,
        dims: block.dims.clone(),
        inputs: vec![original_input.clone()],
    }
}

fn try_strength_reduce(
    block: &PatternBlock,
    mapped_inputs: &[InputRef],
    new_graph: &NanoGraph,
) -> Option<Replacement> {
    let ScalarOp::Binary(bin_op) = &block.op else {
        return None;
    };
    if mapped_inputs.len() != 2 {
        return None;
    }

    let lhs_id = mapped_inputs[0].source_block;
    let rhs_id = mapped_inputs[1].source_block;

    match bin_op {
        ScalarBinOp::Add => {
            // x + 0 → x
            if is_literal_value(new_graph, rhs_id, 0.0) {
                return Some(identity_passthrough(block, &mapped_inputs[0]));
            }
            // 0 + x → x
            if is_literal_value(new_graph, lhs_id, 0.0) {
                return Some(identity_passthrough(block, &mapped_inputs[1]));
            }
        }
        ScalarBinOp::Sub => {
            // x - 0 → x
            if is_literal_value(new_graph, rhs_id, 0.0) {
                return Some(identity_passthrough(block, &mapped_inputs[0]));
            }
        }
        ScalarBinOp::Mul => {
            // x * 1 → x
            if is_literal_value(new_graph, rhs_id, 1.0) {
                return Some(identity_passthrough(block, &mapped_inputs[0]));
            }
            // 1 * x → x
            if is_literal_value(new_graph, lhs_id, 1.0) {
                return Some(identity_passthrough(block, &mapped_inputs[1]));
            }
            // x * 0 → literal 0
            if is_literal_value(new_graph, rhs_id, 0.0)
                || is_literal_value(new_graph, lhs_id, 0.0)
            {
                return Some(Replacement {
                    op: ScalarOp::Literal(LiteralBits::f64(0.0)),
                    dtype: block.dtype,
                    dims: block.dims.clone(),
                    inputs: vec![],
                });
            }
        }
        ScalarBinOp::Div => {
            // x / 1 → x
            if is_literal_value(new_graph, rhs_id, 1.0) {
                return Some(identity_passthrough(block, &mapped_inputs[0]));
            }
        }
        _ => {}
    }

    None
}

// ---------------------------------------------------------------------------
// Common subexpression elimination (CSE)
// ---------------------------------------------------------------------------

/// Deduplicate blocks that have the same (op, dtype, dims, inputs).
///
/// Two blocks are considered equivalent if they have the same scalar op,
/// dtype, dimension list, and their inputs point to the same (already-mapped)
/// source blocks with the same dim_maps.
pub fn cse(graph: NanoGraph) -> OptResult {
    let mut new_graph = graph.new_with_same_symbolics();

    let mut id_map: HashMap<BlockId, BlockId> = HashMap::new();
    let mut canonical: HashMap<BlockSignature, BlockId> = HashMap::new();
    let mut eliminated = 0usize;

    for block in graph.blocks() {
        let mapped_inputs: Vec<InputRef> = block
            .inputs
            .iter()
            .map(|inp| InputRef {
                source_block: id_map[&inp.source_block],
                dim_map: inp.dim_map.clone(),
            })
            .collect();

        let sig = BlockSignature {
            op: block.op,
            dtype: block.dtype,
            dims: block.dims.clone(),
            inputs: mapped_inputs.clone(),
        };

        // Don't deduplicate leaf blocks (0 inputs): they represent distinct
        // external inputs/weights that happen to share the same sentinel value.
        if !block.inputs.is_empty()
            && let Some(&existing) = canonical.get(&sig)
        {
            id_map.insert(block.id, existing);
            eliminated += 1;
            continue;
        }
        {
            let new_id = new_graph.push(
                block.op,
                block.dtype,
                block.dims.clone(),
                mapped_inputs,
            );
            id_map.insert(block.id, new_id);
            canonical.insert(sig, new_id);
        }
    }

    new_graph.output_blocks = graph
        .output_blocks
        .iter()
        .filter_map(|id| id_map.get(id).copied())
        .collect();

    OptResult {
        graph: new_graph,
        eliminated,
    }
}

/// Hashable signature for CSE deduplication.
#[derive(Clone, PartialEq, Eq, Hash)]
struct BlockSignature {
    op: ScalarOp,
    dtype: crate::dtype::DType,
    dims: Vec<crate::nano_graph::pattern::Dim>,
    inputs: Vec<InputRef>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::nano_graph::ops::{LiteralBits, ScalarBinOp, ScalarOp, ScalarUnaryOp};
    use crate::nano_graph::pattern::{Dim, InputRef, NanoGraph, SourceDimMap};

    fn literal_block(g: &mut NanoGraph, val: f32, dims: Vec<Dim>) -> BlockId {
        g.push(ScalarOp::Literal(LiteralBits::f32(val)), DType::F32, dims, vec![])
    }

    #[test]
    fn test_dce_removes_unreachable() {
        let mut g = NanoGraph::new();
        let a = literal_block(&mut g, 1.0, vec![Dim::Known(10)]);
        let _dead = literal_block(&mut g, 2.0, vec![Dim::Known(10)]);
        let b = g.push(
            ScalarOp::Unary(ScalarUnaryOp::Neg),
            DType::F32,
            vec![Dim::Known(10)],
            vec![InputRef {
                source_block: a,
                dim_map: vec![SourceDimMap::identity(0, 1)],
            }],
        );
        g.output_blocks = vec![b];

        let result = dce(g);
        assert_eq!(result.eliminated, 1);
        assert_eq!(result.graph.len(), 2); // a and b
        assert!(result.graph.validate().is_empty());
    }

    #[test]
    fn test_strength_reduce_add_zero() {
        let mut g = NanoGraph::new();
        let x = literal_block(&mut g, 5.0, vec![Dim::Known(10)]);
        let zero = literal_block(&mut g, 0.0, vec![Dim::Known(10)]);
        let sum = g.push(
            ScalarOp::Binary(ScalarBinOp::Add),
            DType::F32,
            vec![Dim::Known(10)],
            vec![
                InputRef {
                    source_block: x,
                    dim_map: vec![SourceDimMap::identity(0, 1)],
                },
                InputRef {
                    source_block: zero,
                    dim_map: vec![SourceDimMap::identity(0, 1)],
                },
            ],
        );
        g.output_blocks = vec![sum];

        let result = strength_reduce(g);
        assert_eq!(result.eliminated, 1);
        // The Add should become Identity.
        let out_id = result.graph.output_blocks[0];
        let out_block = result.graph.get(out_id).unwrap();
        assert_eq!(out_block.op, ScalarOp::Identity);
        assert!(result.graph.validate().is_empty());
    }

    #[test]
    fn test_strength_reduce_mul_zero() {
        let mut g = NanoGraph::new();
        let x = literal_block(&mut g, 5.0, vec![Dim::Known(10)]);
        let zero = literal_block(&mut g, 0.0, vec![Dim::Known(10)]);
        let prod = g.push(
            ScalarOp::Binary(ScalarBinOp::Mul),
            DType::F32,
            vec![Dim::Known(10)],
            vec![
                InputRef {
                    source_block: x,
                    dim_map: vec![SourceDimMap::identity(0, 1)],
                },
                InputRef {
                    source_block: zero,
                    dim_map: vec![SourceDimMap::identity(0, 1)],
                },
            ],
        );
        g.output_blocks = vec![prod];

        let result = strength_reduce(g);
        assert_eq!(result.eliminated, 1);
        let out_id = result.graph.output_blocks[0];
        let out_block = result.graph.get(out_id).unwrap();
        assert!(matches!(out_block.op, ScalarOp::Literal(_)));
        assert!(result.graph.validate().is_empty());
    }

    #[test]
    fn test_cse_deduplicates() {
        let mut g = NanoGraph::new();
        let a = literal_block(&mut g, 1.0, vec![Dim::Known(10)]);
        // Two identical neg blocks on the same input.
        let neg1 = g.push(
            ScalarOp::Unary(ScalarUnaryOp::Neg),
            DType::F32,
            vec![Dim::Known(10)],
            vec![InputRef {
                source_block: a,
                dim_map: vec![SourceDimMap::identity(0, 1)],
            }],
        );
        let neg2 = g.push(
            ScalarOp::Unary(ScalarUnaryOp::Neg),
            DType::F32,
            vec![Dim::Known(10)],
            vec![InputRef {
                source_block: a,
                dim_map: vec![SourceDimMap::identity(0, 1)],
            }],
        );
        // Use both outputs so neither is dead.
        let sum = g.push(
            ScalarOp::Binary(ScalarBinOp::Add),
            DType::F32,
            vec![Dim::Known(10)],
            vec![
                InputRef {
                    source_block: neg1,
                    dim_map: vec![SourceDimMap::identity(0, 1)],
                },
                InputRef {
                    source_block: neg2,
                    dim_map: vec![SourceDimMap::identity(0, 1)],
                },
            ],
        );
        g.output_blocks = vec![sum];

        let result = cse(g);
        assert_eq!(result.eliminated, 1);
        assert_eq!(result.graph.len(), 3); // a, one neg, sum
        assert!(result.graph.validate().is_empty());
    }
}
