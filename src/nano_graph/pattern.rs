//! Compressed pattern-block representation of the scalar DAG.
//!
//! A `PatternBlock` represents a family of identical scalar operations
//! parameterized by one or more iteration dimensions. Each dimension has
//! a concrete count or a symbolic variable (for batch/seq_len). The inputs
//! to each instance are described by `InputRef`s — affine functions of the
//! iteration indices that point into other blocks.
//!
//! The full scalar DAG is the union of all pattern blocks. Edges between
//! blocks are implicit in the `InputRef` descriptors.

use crate::dtype::DType;
use crate::nano_graph::ops::ScalarOp;
use std::collections::HashMap;

/// Identifies a pattern block within a `NanoGraph`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

/// A dimension that may be concrete or symbolic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dim {
    /// Known size at graph-build time.
    Known(u64),
    /// Unknown size, resolved at runtime. The u16 is a symbolic variable id
    /// (e.g., 0 = batch, 1 = seq_len). All blocks sharing the same variable
    /// id must agree on the runtime value.
    Symbolic(u16),
}

impl Dim {
    /// Returns the concrete size, or None if symbolic.
    pub fn as_known(&self) -> Option<u64> {
        match self {
            Dim::Known(n) => Some(*n),
            Dim::Symbolic(_) => None,
        }
    }
}

/// Describes how one input to a pattern block is sourced from another block's
/// output space.
///
/// There is one `SourceDimMap` entry per dimension of the *source* block.
/// Each entry is either an affine function of the consumer's iteration
/// indices (`Affine`), or a marker that the consumer's reduce op sweeps
/// over that source dimension internally (`Reduce`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InputRef {
    /// The block that produces the values we consume.
    pub source_block: BlockId,
    /// One entry per dimension of the source block. The meaning of each
    /// entry depends on the variant:
    ///
    /// - `Affine`: the source index on this dim is computed from the
    ///   consumer's iteration indices via an affine expression.
    /// - `Reduce`: the consumer (which must be a reduce op) sweeps over
    ///   the full extent of this source dimension, accumulating results.
    pub dim_map: Vec<SourceDimMap>,
}

/// How one source dimension is accessed by a consumer block.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SourceDimMap {
    /// Source index on this dim is an affine function of the consumer's
    /// iteration indices:
    ///
    ///   `index = base + sum(coeffs[j] * consumer_iter[j])`
    Affine(AffineDimMap),
    /// This dimension is swept by the consumer's reduce operation.
    /// The reduce iterates over the full extent of this source dimension.
    /// Only valid when the consumer block's op is a reduce variant.
    Reduce,
}

impl SourceDimMap {
    /// Shorthand: affine identity on a single consumer dim.
    pub fn identity(consumer_dim: usize, num_consumer_dims: usize) -> Self {
        Self::Affine(AffineDimMap::identity(consumer_dim, num_consumer_dims))
    }

    /// Shorthand: affine constant (broadcast / fixed index).
    pub fn constant(base: i64) -> Self {
        Self::Affine(AffineDimMap::constant(base))
    }

    /// Shorthand: affine strided on a single consumer dim.
    pub fn strided(consumer_dim: usize, stride: i64, num_consumer_dims: usize) -> Self {
        Self::Affine(AffineDimMap::strided(consumer_dim, stride, num_consumer_dims))
    }
}

/// Affine map for one dimension of the source block:
///   index = base + sum(coeffs[j] * consumer_iter[j])
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AffineDimMap {
    pub base: i64,
    pub coeffs: Vec<i64>,
}

impl AffineDimMap {
    /// Constant index (no dependence on consumer iteration).
    pub fn constant(base: i64) -> Self {
        Self {
            base,
            coeffs: vec![],
        }
    }

    /// Simple 1:1 mapping on a single consumer dimension.
    pub fn identity(consumer_dim: usize, num_consumer_dims: usize) -> Self {
        let mut coeffs = vec![0i64; num_consumer_dims];
        coeffs[consumer_dim] = 1;
        Self { base: 0, coeffs }
    }

    /// Strided mapping on a single consumer dimension.
    pub fn strided(consumer_dim: usize, stride: i64, num_consumer_dims: usize) -> Self {
        let mut coeffs = vec![0i64; num_consumer_dims];
        coeffs[consumer_dim] = stride;
        Self { base: 0, coeffs }
    }

    /// Return a copy with the base offset changed.
    pub fn with_base(mut self, base: i64) -> Self {
        self.base = base;
        self
    }

    /// Evaluate this affine map given concrete consumer iteration indices.
    pub fn eval(&self, consumer_iter: &[i64]) -> i64 {
        let mut result = self.base;
        for (c, &idx) in self.coeffs.iter().zip(consumer_iter) {
            result += c * idx;
        }
        result
    }
}

/// A compressed representation of a family of identical scalar ops.
///
/// Each block produces `product(dims)` scalar values — one per point in
/// the iteration space defined by `dims`. The scalar op at each point is
/// the same (`op`), and the inputs are computed by the `InputRef` dim maps.
///
/// For reduce blocks, the block's `dims` describe the *output* iteration
/// space (the space after reduction). The reduction dimensions are implicit —
/// they live in the source block and are marked as `SourceDimMap::Reduce`
/// in the input ref.
#[derive(Debug, Clone)]
pub struct PatternBlock {
    pub id: BlockId,
    /// The scalar operation.
    pub op: ScalarOp,
    /// The dtype this block operates in / produces.
    pub dtype: DType,
    /// Iteration dimensions. The block produces one scalar per point in this
    /// multi-dimensional space.
    pub dims: Vec<Dim>,
    /// Inputs to the operation. The number of inputs must match what `op`
    /// expects (0 for Literal, 1 for Unary/Reduce, 2 for Binary).
    pub inputs: Vec<InputRef>,
}

impl PatternBlock {
    /// Total number of concrete output elements, or None if any dim is symbolic.
    pub fn concrete_count(&self) -> Option<u64> {
        let mut total = 1u64;
        for d in &self.dims {
            total = total.checked_mul(d.as_known()?)?;
        }
        Some(total)
    }

    /// Number of iteration dimensions.
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// For reduce blocks: returns the source dimensions that are reduced over,
    /// by examining the input ref's dim_map for `Reduce` entries.
    /// Returns (source_dim_index, source_dim_size) pairs.
    pub fn reduce_dims(&self, graph: &NanoGraph) -> Vec<(usize, Dim)> {
        if !matches!(self.op, ScalarOp::ReduceSum | ScalarOp::ReduceMax) {
            return vec![];
        }
        let Some(input) = self.inputs.first() else {
            return vec![];
        };
        let Some(source) = graph.get(input.source_block) else {
            return vec![];
        };
        input
            .dim_map
            .iter()
            .enumerate()
            .filter_map(|(i, dm)| match dm {
                SourceDimMap::Reduce => Some((i, source.dims[i])),
                _ => None,
            })
            .collect()
    }
}

/// The compressed scalar DAG for an entire computation.
#[derive(Default)]
pub struct NanoGraph {
    blocks: Vec<PatternBlock>,
    /// BlockId → index in `blocks`.
    index: HashMap<BlockId, usize>,
    next_id: u32,
    /// Named symbolic dimensions (e.g., "batch" → 0, "seq_len" → 1).
    pub symbolic_names: HashMap<String, u16>,
    next_symbolic: u16,
    /// Which blocks are final outputs (model outputs, loss, etc.)
    pub output_blocks: Vec<BlockId>,
}

impl NanoGraph {
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
            index: HashMap::new(),
            next_id: 0,
            symbolic_names: HashMap::new(),
            next_symbolic: 0,
            output_blocks: Vec::new(),
        }
    }

    /// Create a new empty graph that inherits symbolic dimension names.
    pub fn new_with_same_symbolics(&self) -> Self {
        Self {
            blocks: Vec::new(),
            index: HashMap::new(),
            next_id: 0,
            symbolic_names: self.symbolic_names.clone(),
            next_symbolic: self.next_symbolic,
            output_blocks: Vec::new(),
        }
    }

    /// Register or retrieve a symbolic dimension by name.
    pub fn symbolic_dim(&mut self, name: &str) -> Dim {
        if let Some(&id) = self.symbolic_names.get(name) {
            Dim::Symbolic(id)
        } else {
            let id = self.next_symbolic;
            self.next_symbolic += 1;
            self.symbolic_names.insert(name.to_string(), id);
            Dim::Symbolic(id)
        }
    }

    /// Allocate a new block id.
    fn alloc_id(&mut self) -> BlockId {
        let id = BlockId(self.next_id);
        self.next_id = self.next_id.checked_add(1).expect("BlockId overflow");
        id
    }

    /// Add a pattern block to the graph. Returns its id.
    pub fn push(
        &mut self,
        op: ScalarOp,
        dtype: DType,
        dims: Vec<Dim>,
        inputs: Vec<InputRef>,
    ) -> BlockId {
        let id = self.alloc_id();
        let block = PatternBlock {
            id,
            op,
            dtype,
            dims,
            inputs,
        };
        let idx = self.blocks.len();
        self.blocks.push(block);
        self.index.insert(id, idx);
        id
    }

    /// Look up a block by id.
    pub fn get(&self, id: BlockId) -> Option<&PatternBlock> {
        self.index.get(&id).map(|&idx| &self.blocks[idx])
    }

    /// Iterate all blocks in insertion order.
    pub fn blocks(&self) -> &[PatternBlock] {
        &self.blocks
    }

    /// Number of blocks.
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Iterate blocks that directly consume a given block's output.
    pub fn consumers_of(&self, id: BlockId) -> Vec<BlockId> {
        self.blocks
            .iter()
            .filter(|b| b.inputs.iter().any(|inp| inp.source_block == id))
            .map(|b| b.id)
            .collect()
    }

    /// Iterate blocks that a given block directly depends on.
    pub fn producers_of(&self, id: BlockId) -> Vec<BlockId> {
        match self.get(id) {
            Some(block) => block
                .inputs
                .iter()
                .map(|inp| inp.source_block)
                .collect(),
            None => vec![],
        }
    }

    /// Validate structural invariants on all blocks. Returns a list of errors.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        for block in &self.blocks {
            let is_reduce = matches!(block.op, ScalarOp::ReduceSum | ScalarOp::ReduceMax);

            for (inp_idx, input) in block.inputs.iter().enumerate() {
                let Some(source) = self.get(input.source_block) else {
                    errors.push(format!(
                        "Block {:?} input {} references nonexistent block {:?}",
                        block.id, inp_idx, input.source_block
                    ));
                    continue;
                };

                // dim_map length must match source block's dims.
                if input.dim_map.len() != source.dims.len() {
                    errors.push(format!(
                        "Block {:?} input {}: dim_map has {} entries but source {:?} has {} dims",
                        block.id,
                        inp_idx,
                        input.dim_map.len(),
                        input.source_block,
                        source.dims.len(),
                    ));
                }

                let has_reduce = input.dim_map.iter().any(|d| matches!(d, SourceDimMap::Reduce));
                if has_reduce && !is_reduce {
                    errors.push(format!(
                        "Block {:?} input {}: has Reduce dim map but op is {:?} (not a reduce)",
                        block.id, inp_idx, block.op,
                    ));
                }
                if is_reduce && !has_reduce {
                    errors.push(format!(
                        "Block {:?}: reduce op but input {} has no Reduce dims",
                        block.id, inp_idx,
                    ));
                }

                // Affine coeffs length should match consumer's dims.
                for (dim_idx, dm) in input.dim_map.iter().enumerate() {
                    if let SourceDimMap::Affine(aff) = dm
                        && !aff.coeffs.is_empty()
                        && aff.coeffs.len() != block.dims.len()
                    {
                        errors.push(format!(
                            "Block {:?} input {} dim {}: affine has {} coeffs but block has {} dims",
                            block.id,
                            inp_idx,
                            dim_idx,
                            aff.coeffs.len(),
                            block.dims.len(),
                        ));
                    }
                }
            }

            // Input count check.
            let expected_inputs = match block.op {
                ScalarOp::Literal(_) => 0,
                ScalarOp::Unary(_) | ScalarOp::Identity => 1,
                ScalarOp::Binary(_) => 2,
                ScalarOp::ReduceSum | ScalarOp::ReduceMax => 1,
            };
            if block.inputs.len() != expected_inputs {
                errors.push(format!(
                    "Block {:?}: op {:?} expects {} inputs but has {}",
                    block.id,
                    block.op,
                    expected_inputs,
                    block.inputs.len(),
                ));
            }
        }
        errors
    }

    /// Summary stats for debugging.
    pub fn stats(&self) -> NanoGraphStats {
        let mut total_concrete = 0u64;
        let mut symbolic_blocks = 0u64;
        for block in &self.blocks {
            match block.concrete_count() {
                Some(n) => total_concrete = total_concrete.saturating_add(n),
                None => symbolic_blocks += 1,
            }
        }
        NanoGraphStats {
            num_blocks: self.blocks.len() as u64,
            total_concrete_ops: total_concrete,
            symbolic_blocks,
        }
    }
}

#[derive(Debug)]
pub struct NanoGraphStats {
    pub num_blocks: u64,
    pub total_concrete_ops: u64,
    pub symbolic_blocks: u64,
}

impl std::fmt::Display for NanoGraphStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} blocks ({} concrete ops, {} symbolic blocks)",
            self.num_blocks, self.total_concrete_ops, self.symbolic_blocks
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::nano_graph::ops::{LiteralBits, ScalarBinOp, ScalarOp};

    /// Build a tiny graph: c = a + b, elementwise over 1024 elements.
    #[test]
    fn test_elementwise_add() {
        let mut g = NanoGraph::new();

        let a = g.push(
            ScalarOp::Literal(LiteralBits::f32(0.0)),
            DType::F32,
            vec![Dim::Known(1024)],
            vec![],
        );
        let b = g.push(
            ScalarOp::Literal(LiteralBits::f32(0.0)),
            DType::F32,
            vec![Dim::Known(1024)],
            vec![],
        );

        let c = g.push(
            ScalarOp::Binary(ScalarBinOp::Add),
            DType::F32,
            vec![Dim::Known(1024)],
            vec![
                InputRef {
                    source_block: a,
                    dim_map: vec![SourceDimMap::identity(0, 1)],
                },
                InputRef {
                    source_block: b,
                    dim_map: vec![SourceDimMap::identity(0, 1)],
                },
            ],
        );

        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let stats = g.stats();
        assert_eq!(stats.num_blocks, 3);
        assert_eq!(stats.total_concrete_ops, 3 * 1024);

        assert_eq!(g.consumers_of(a), vec![c]);
        assert_eq!(g.producers_of(c), vec![a, b]);
    }

    /// Broadcast: c[i,j] = a[i,j] + b[j], where a is [4,8] and b is [8].
    #[test]
    fn test_broadcast() {
        let mut g = NanoGraph::new();

        let a = g.push(
            ScalarOp::Literal(LiteralBits::f32(0.0)),
            DType::F32,
            vec![Dim::Known(4), Dim::Known(8)],
            vec![],
        );
        let b = g.push(
            ScalarOp::Literal(LiteralBits::f32(0.0)),
            DType::F32,
            vec![Dim::Known(8)],
            vec![],
        );

        let _c = g.push(
            ScalarOp::Binary(ScalarBinOp::Add),
            DType::F32,
            vec![Dim::Known(4), Dim::Known(8)],
            vec![
                InputRef {
                    source_block: a,
                    dim_map: vec![
                        SourceDimMap::identity(0, 2),
                        SourceDimMap::identity(1, 2),
                    ],
                },
                InputRef {
                    source_block: b,
                    // b has 1 dim; consumer has 2. b_dim0 = consumer_dim1.
                    dim_map: vec![SourceDimMap::identity(1, 2)],
                },
            ],
        );

        assert!(g.validate().is_empty(), "{:?}", g.validate());
        assert_eq!(g.len(), 3);
    }

    /// MatMul: C[i,j] = sum_k(A[i,k] * B[k,j])
    /// A is [M,K], B is [K,N], C is [M,N].
    #[test]
    fn test_matmul_pattern() {
        let mut g = NanoGraph::new();
        let (m, k, n) = (4, 3, 5);

        let a = g.push(
            ScalarOp::Literal(LiteralBits::f32(0.0)),
            DType::F32,
            vec![Dim::Known(m), Dim::Known(k)],
            vec![],
        );
        let b = g.push(
            ScalarOp::Literal(LiteralBits::f32(0.0)),
            DType::F32,
            vec![Dim::Known(k), Dim::Known(n)],
            vec![],
        );

        // product[i,j,k] = A[i,k] * B[k,j]
        // Block has 3 dims: [M, N, K].
        let product = g.push(
            ScalarOp::Binary(ScalarBinOp::Mul),
            DType::F32,
            vec![Dim::Known(m), Dim::Known(n), Dim::Known(k)],
            vec![
                InputRef {
                    source_block: a,
                    dim_map: vec![
                        SourceDimMap::identity(0, 3), // A_dim0 (i) = consumer_dim0
                        SourceDimMap::identity(2, 3), // A_dim1 (k) = consumer_dim2
                    ],
                },
                InputRef {
                    source_block: b,
                    dim_map: vec![
                        SourceDimMap::identity(2, 3), // B_dim0 (k) = consumer_dim2
                        SourceDimMap::identity(1, 3), // B_dim1 (j) = consumer_dim1
                    ],
                },
            ],
        );

        // C[i,j] = reduce_sum over k of product[i,j,k]
        // Block has 2 dims: [M, N].
        // Source has 3 dims; dims 0,1 are mapped, dim 2 is Reduce.
        let c = g.push(
            ScalarOp::ReduceSum,
            DType::F32,
            vec![Dim::Known(m), Dim::Known(n)],
            vec![InputRef {
                source_block: product,
                dim_map: vec![
                    SourceDimMap::identity(0, 2), // product_dim0 (i) = consumer_dim0
                    SourceDimMap::identity(1, 2), // product_dim1 (j) = consumer_dim1
                    SourceDimMap::Reduce,         // product_dim2 (k) = reduced
                ],
            }],
        );

        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let stats = g.stats();
        assert_eq!(stats.num_blocks, 4);
        // a: 4*3=12, b: 3*5=15, product: 4*5*3=60, c: 4*5=20
        assert_eq!(stats.total_concrete_ops, 12 + 15 + 60 + 20);

        // Verify reduce_dims reports the K dimension.
        let c_block = g.get(c).unwrap();
        let rdims = c_block.reduce_dims(&g);
        assert_eq!(rdims.len(), 1);
        assert_eq!(rdims[0], (2, Dim::Known(k)));
    }

    /// Symbolic dimension: add over [batch, 1024].
    #[test]
    fn test_symbolic_dim() {
        let mut g = NanoGraph::new();
        let batch = g.symbolic_dim("batch");

        let a = g.push(
            ScalarOp::Literal(LiteralBits::f32(0.0)),
            DType::F32,
            vec![batch, Dim::Known(1024)],
            vec![],
        );
        let b = g.push(
            ScalarOp::Literal(LiteralBits::f32(0.0)),
            DType::F32,
            vec![batch, Dim::Known(1024)],
            vec![],
        );
        let _c = g.push(
            ScalarOp::Binary(ScalarBinOp::Add),
            DType::F32,
            vec![batch, Dim::Known(1024)],
            vec![
                InputRef {
                    source_block: a,
                    dim_map: vec![
                        SourceDimMap::identity(0, 2),
                        SourceDimMap::identity(1, 2),
                    ],
                },
                InputRef {
                    source_block: b,
                    dim_map: vec![
                        SourceDimMap::identity(0, 2),
                        SourceDimMap::identity(1, 2),
                    ],
                },
            ],
        );

        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let stats = g.stats();
        assert_eq!(stats.num_blocks, 3);
        assert_eq!(stats.symbolic_blocks, 3);
    }

    /// Reduction to scalar: sum all elements of a [4, 8] block.
    #[test]
    fn test_reduce_to_scalar() {
        let mut g = NanoGraph::new();

        let a = g.push(
            ScalarOp::Literal(LiteralBits::f32(1.0)),
            DType::F32,
            vec![Dim::Known(4), Dim::Known(8)],
            vec![],
        );

        // Reduce both dims → scalar (empty dims).
        let total = g.push(
            ScalarOp::ReduceSum,
            DType::F32,
            vec![],
            vec![InputRef {
                source_block: a,
                dim_map: vec![SourceDimMap::Reduce, SourceDimMap::Reduce],
            }],
        );

        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let total_block = g.get(total).unwrap();
        assert_eq!(total_block.concrete_count(), Some(1)); // scalar output
        let rdims = total_block.reduce_dims(&g);
        assert_eq!(rdims.len(), 2);
        assert_eq!(rdims[0], (0, Dim::Known(4)));
        assert_eq!(rdims[1], (1, Dim::Known(8)));
    }

    /// Partial reduction: [4, 8] → [4] by reducing dim 1.
    #[test]
    fn test_partial_reduce() {
        let mut g = NanoGraph::new();

        let a = g.push(
            ScalarOp::Literal(LiteralBits::f32(1.0)),
            DType::F32,
            vec![Dim::Known(4), Dim::Known(8)],
            vec![],
        );

        // row_sum[i] = sum_j(a[i, j])
        let row_sum = g.push(
            ScalarOp::ReduceSum,
            DType::F32,
            vec![Dim::Known(4)],
            vec![InputRef {
                source_block: a,
                dim_map: vec![
                    SourceDimMap::identity(0, 1), // a_dim0 = consumer_dim0
                    SourceDimMap::Reduce,         // a_dim1 = reduced
                ],
            }],
        );

        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let block = g.get(row_sum).unwrap();
        assert_eq!(block.concrete_count(), Some(4));
        let rdims = block.reduce_dims(&g);
        assert_eq!(rdims.len(), 1);
        assert_eq!(rdims[0], (1, Dim::Known(8)));
    }

    /// Validation catches mismatched dim_map length.
    #[test]
    fn test_validate_dim_mismatch() {
        let mut g = NanoGraph::new();

        let a = g.push(
            ScalarOp::Literal(LiteralBits::f32(0.0)),
            DType::F32,
            vec![Dim::Known(4), Dim::Known(8)],
            vec![],
        );

        // Wrong: source has 2 dims but we only provide 1 dim_map entry.
        let _bad = g.push(
            ScalarOp::Unary(crate::nano_graph::ScalarUnaryOp::Neg),
            DType::F32,
            vec![Dim::Known(4), Dim::Known(8)],
            vec![InputRef {
                source_block: a,
                dim_map: vec![SourceDimMap::identity(0, 2)], // missing dim 1
            }],
        );

        let errors = g.validate();
        assert!(!errors.is_empty());
        assert!(errors[0].contains("dim_map has 1 entries but source"));
    }

    /// Validation catches Reduce on non-reduce op.
    #[test]
    fn test_validate_reduce_on_non_reduce_op() {
        let mut g = NanoGraph::new();

        let a = g.push(
            ScalarOp::Literal(LiteralBits::f32(0.0)),
            DType::F32,
            vec![Dim::Known(4)],
            vec![],
        );

        let _bad = g.push(
            ScalarOp::Unary(crate::nano_graph::ScalarUnaryOp::Neg),
            DType::F32,
            vec![],
            vec![InputRef {
                source_block: a,
                dim_map: vec![SourceDimMap::Reduce],
            }],
        );

        let errors = g.validate();
        assert!(!errors.is_empty());
        assert!(errors[0].contains("Reduce dim map but op is"));
    }
}
