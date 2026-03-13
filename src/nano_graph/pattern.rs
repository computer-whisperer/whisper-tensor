//! Compressed scalar DAG representation for full-model computation graphs.
//!
//! The nano graph dissolves tensor boundaries into a DAG of scalar operations.
//! Known dimensions (weight shapes, hidden sizes) are fully expanded into
//! individual atoms. Unknown dimensions (batch, seq_len) are represented as
//! symbolic iteration parameters on atom groups.
//!
//! Compression is achieved by grouping structurally identical atoms into
//! `AtomGroup`s. Groups are a convenience — every atom could exist standalone
//! without changing semantics. The grouping never limits what can be expressed.

use crate::nano_graph::ops::ScalarOp;
use std::collections::HashMap;

/// A symbolic runtime dimension (batch, seq_len, etc.).
///
/// Multiple atoms/groups can share the same SymDim, meaning they iterate
/// over the same runtime-variable extent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SymDim(pub u16);

/// Unique identifier for a scalar atom in the graph.
///
/// Within an `AtomGroup`, atoms have contiguous ids from `base_id` to
/// `base_id + count - 1`. The offset within the group determines how
/// `InputRef::Affine` strides are applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AtomId(pub u32);

impl AtomId {
    pub fn offset(self, n: u32) -> Self {
        AtomId(self.0 + n)
    }
}

impl std::fmt::Display for AtomId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "a{}", self.0)
    }
}

/// How an input to an atom group references source atoms.
///
/// Multiple addressing modes coexist in the same graph — this is the key
/// difference from v1 where affine was the only option.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum InputRef {
    /// Every atom in this group reads the same source atom (broadcast).
    Broadcast(AtomId),
    /// Atom at offset `i` in this group reads source atom `base + stride * i`.
    /// Covers elementwise (stride=1), strided views, and reversed access.
    Affine { base: AtomId, stride: i32 },
    /// Arbitrary per-atom source mapping. Used when no regular pattern exists
    /// (e.g., Gather with compile-time-known indices, irregular Concat).
    /// Length must equal the group's `count`.
    Explicit(Vec<AtomId>),
    /// Source depends on both atom offset `i` and sym_dim iteration index `k`.
    /// Resolves to `base + stride_i * i + stride_k * k`.
    /// Used for contractions (MatMul) where each output atom's source
    /// varies with the reduction iteration.
    SymAffine {
        base: AtomId,
        stride_i: i32,
        stride_k: i32,
    },
}

impl InputRef {
    /// Resolve the source atom for the `i`-th atom in the group,
    /// at sym_dim iteration `k` (ignored for non-SymAffine variants).
    pub fn resolve(&self, i: u32, k: u32) -> AtomId {
        match self {
            InputRef::Broadcast(id) => *id,
            InputRef::Affine { base, stride } => {
                AtomId(base.0.wrapping_add((*stride as i64 * i as i64) as u32))
            }
            InputRef::Explicit(ids) => ids[i as usize],
            InputRef::SymAffine {
                base,
                stride_i,
                stride_k,
            } => {
                AtomId(base.0.wrapping_add(
                    (*stride_i as i64 * i as i64 + *stride_k as i64 * k as i64) as u32,
                ))
            }
        }
    }

    /// Number of distinct source atoms referenced (for compression stats).
    pub fn distinct_sources(&self, count: u32) -> usize {
        match self {
            InputRef::Broadcast(_) => 1,
            InputRef::Affine { .. } => count as usize,
            InputRef::Explicit(ids) => {
                let mut seen = ids.clone();
                seen.sort();
                seen.dedup();
                seen.len()
            }
            InputRef::SymAffine { .. } => count as usize, // lower bound; actual depends on k range
        }
    }
}

/// A group of structurally identical scalar atoms.
///
/// All atoms in a group share the same op, symbolic dimensions,
/// and input addressing pattern. The group stores `count` atoms with
/// contiguous `AtomId`s starting at `base_id`.
///
/// Dtype precision semantics (compute_dtype, output_dtype) live on the
/// ScalarOp itself, so each op variant can carry its own behavioral config.
///
/// A group of count=1 is a standalone atom — this is the degenerate case
/// for ops that don't compress (e.g., Gather boundary atoms).
#[derive(Debug, Clone)]
pub struct AtomGroup {
    /// First AtomId in this group.
    pub base_id: AtomId,
    /// Number of atoms in the group.
    pub count: u32,
    /// The scalar operation each atom performs, including dtype precision.
    pub op: ScalarOp,
    /// Symbolic dimensions this group iterates over.
    /// Each atom in the group independently iterates over these dims.
    /// E.g., `[batch, seq_len]` means each atom produces a 2D tile of values.
    pub sym_dims: Vec<SymDim>,
    /// Symbolic dimensions reduced by this op (for ReduceSum/ReduceMax).
    /// The op accumulates over these dims, so the output doesn't have them.
    pub reduce_dims: Vec<SymDim>,
    /// Inputs to the operation. Number must match what `op` expects.
    pub inputs: Vec<InputRef>,
}

impl AtomGroup {
    /// Returns the AtomId range `[base_id, base_id + count)`.
    pub fn atom_ids(&self) -> impl Iterator<Item = AtomId> {
        let base = self.base_id.0;
        let count = self.count;
        (0..count).map(move |i| AtomId(base + i))
    }

    /// Check if an AtomId belongs to this group.
    pub fn contains(&self, id: AtomId) -> bool {
        id.0 >= self.base_id.0 && id.0 < self.base_id.0 + self.count
    }

    /// Offset of an AtomId within this group.
    pub fn offset_of(&self, id: AtomId) -> Option<u32> {
        if self.contains(id) {
            Some(id.0 - self.base_id.0)
        } else {
            None
        }
    }
}

/// The compressed scalar DAG for an entire computation.
#[derive(Default)]
pub struct NanoGraph {
    groups: Vec<AtomGroup>,
    next_atom_id: u32,
    /// Named symbolic dimensions (e.g., "batch" → SymDim(0)).
    pub sym_dim_names: HashMap<String, SymDim>,
    /// Known upper bounds for symbolic dimensions. A SymDim with a known bound
    /// is used for contractions (e.g., MatMul's K dimension) where the extent
    /// is compile-time known but the dim is iterated over during reduction.
    pub sym_dim_bounds: HashMap<SymDim, u64>,
    next_sym_dim: u16,
    /// Which atoms are final outputs of the computation.
    pub outputs: Vec<AtomId>,
}

impl NanoGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register or retrieve a symbolic dimension by name.
    pub fn sym_dim(&mut self, name: &str) -> SymDim {
        if let Some(&sd) = self.sym_dim_names.get(name) {
            sd
        } else {
            let sd = SymDim(self.next_sym_dim);
            self.next_sym_dim += 1;
            self.sym_dim_names.insert(name.to_string(), sd);
            sd
        }
    }

    /// Look up a sym dim by name without creating it.
    pub fn get_sym_dim(&self, name: &str) -> Option<SymDim> {
        self.sym_dim_names.get(name).copied()
    }

    /// Create a symbolic dimension with a known upper bound.
    /// Used for contraction dimensions (MatMul K, reduction axes).
    pub fn bounded_sym_dim(&mut self, name: &str, bound: u64) -> SymDim {
        let sd = self.sym_dim(name);
        self.sym_dim_bounds.insert(sd, bound);
        sd
    }

    /// Allocate `count` contiguous AtomIds. Returns the base id.
    fn alloc_ids(&mut self, count: u32) -> AtomId {
        let base = AtomId(self.next_atom_id);
        self.next_atom_id = self
            .next_atom_id
            .checked_add(count)
            .expect("AtomId overflow");
        base
    }

    /// Add an atom group to the graph. Returns the base AtomId.
    pub fn push_group(
        &mut self,
        count: u32,
        op: ScalarOp,
        sym_dims: Vec<SymDim>,
        reduce_dims: Vec<SymDim>,
        inputs: Vec<InputRef>,
    ) -> AtomId {
        let base_id = self.alloc_ids(count);
        self.groups.push(AtomGroup {
            base_id,
            count,
            op,
            sym_dims,
            reduce_dims,
            inputs,
        });
        base_id
    }

    /// Convenience: push a single-atom group.
    pub fn push_atom(
        &mut self,
        op: ScalarOp,
        sym_dims: Vec<SymDim>,
        reduce_dims: Vec<SymDim>,
        inputs: Vec<InputRef>,
    ) -> AtomId {
        self.push_group(1, op, sym_dims, reduce_dims, inputs)
    }

    /// Find the group index for an AtomId via binary search on group ranges.
    fn find_group_idx(&self, id: AtomId) -> Option<usize> {
        // Groups have contiguous, non-overlapping AtomId ranges in insertion order.
        // Binary search: find the last group whose base_id <= id.
        let idx = self.groups.partition_point(|g| g.base_id.0 <= id.0);
        if idx == 0 {
            return None;
        }
        let gi = idx - 1;
        let group = &self.groups[gi];
        if group.contains(id) { Some(gi) } else { None }
    }

    /// Look up which group an atom belongs to.
    pub fn group_of(&self, id: AtomId) -> Option<&AtomGroup> {
        self.find_group_idx(id).map(|gi| &self.groups[gi])
    }

    /// Look up group and offset for an atom.
    pub fn group_and_offset(&self, id: AtomId) -> Option<(&AtomGroup, u32)> {
        self.find_group_idx(id).map(|gi| {
            let group = &self.groups[gi];
            (group, id.0 - group.base_id.0)
        })
    }

    /// Check if an AtomId exists in any group.
    pub fn contains_atom(&self, id: AtomId) -> bool {
        self.find_group_idx(id).is_some()
    }

    /// Iterate all groups in insertion order.
    pub fn groups(&self) -> &[AtomGroup] {
        &self.groups
    }

    /// Number of groups.
    pub fn num_groups(&self) -> usize {
        self.groups.len()
    }

    /// Total number of atoms.
    pub fn num_atoms(&self) -> u32 {
        self.next_atom_id
    }

    /// Summary stats for debugging.
    pub fn stats(&self) -> NanoGraphStats {
        let mut total_atoms: u64 = 0;
        let mut singleton_groups: u64 = 0;
        let mut symbolic_groups: u64 = 0;
        let mut groups_by_op: HashMap<&'static str, u64> = HashMap::new();

        for group in &self.groups {
            total_atoms += group.count as u64;
            if group.count == 1 {
                singleton_groups += 1;
            }
            if !group.sym_dims.is_empty() {
                symbolic_groups += 1;
            }
            let op_name = match &group.op {
                ScalarOp::Binary { op, .. } => match op {
                    crate::nano_graph::ScalarBinOp::Add => "Add",
                    crate::nano_graph::ScalarBinOp::Sub => "Sub",
                    crate::nano_graph::ScalarBinOp::Mul => "Mul",
                    crate::nano_graph::ScalarBinOp::Div => "Div",
                    crate::nano_graph::ScalarBinOp::Max => "Max",
                    crate::nano_graph::ScalarBinOp::Min => "Min",
                    crate::nano_graph::ScalarBinOp::Mod => "Mod",
                    crate::nano_graph::ScalarBinOp::Pow => "Pow",
                },
                ScalarOp::Unary { op, .. } => match op {
                    crate::nano_graph::ScalarUnaryOp::Neg => "Neg",
                    crate::nano_graph::ScalarUnaryOp::Abs => "Abs",
                    crate::nano_graph::ScalarUnaryOp::Exp => "Exp",
                    crate::nano_graph::ScalarUnaryOp::Ln => "Ln",
                    crate::nano_graph::ScalarUnaryOp::Sqrt => "Sqrt",
                    crate::nano_graph::ScalarUnaryOp::Reciprocal => "Reciprocal",
                    crate::nano_graph::ScalarUnaryOp::Tanh => "Tanh",
                    crate::nano_graph::ScalarUnaryOp::Floor => "Floor",
                    crate::nano_graph::ScalarUnaryOp::Ceil => "Ceil",
                },
                ScalarOp::Identity { .. } => "Identity",
                ScalarOp::Literal(_) => "Literal",
                ScalarOp::Select { .. } => "Select",
                ScalarOp::ReduceSum { .. } => "ReduceSum",
                ScalarOp::ReduceMax { .. } => "ReduceMax",
            };
            *groups_by_op.entry(op_name).or_default() += 1;
        }

        NanoGraphStats {
            num_groups: self.groups.len() as u64,
            total_atoms,
            singleton_groups,
            symbolic_groups,
            groups_by_op,
        }
    }

    /// Validate structural invariants. Returns a list of errors.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();

        for (gi, group) in self.groups.iter().enumerate() {
            // Verify input refs resolve to existing atoms.
            for (inp_idx, input) in group.inputs.iter().enumerate() {
                match input {
                    InputRef::SymAffine { .. } => {
                        // SymAffine depends on k — validate at k=0 only
                        // (full validation would need sym_dim_bounds).
                        for i in 0..group.count {
                            let source = input.resolve(i, 0);
                            if !self.contains_atom(source) {
                                errors.push(format!(
                                    "Group {} (base={}) input {} atom offset {} k=0: references nonexistent atom {}",
                                    gi, group.base_id, inp_idx, i, source
                                ));
                                break;
                            }
                        }
                    }
                    _ => {
                        for i in 0..group.count {
                            let source = input.resolve(i, 0);
                            if !self.contains_atom(source) {
                                errors.push(format!(
                                    "Group {} (base={}) input {} atom offset {}: references nonexistent atom {}",
                                    gi, group.base_id, inp_idx, i, source
                                ));
                                break;
                            }
                        }
                    }
                }

                // Explicit refs must have correct length.
                if let InputRef::Explicit(ids) = input
                    && ids.len() != group.count as usize
                {
                    errors.push(format!(
                        "Group {} input {}: Explicit has {} entries but group has {} atoms",
                        gi,
                        inp_idx,
                        ids.len(),
                        group.count
                    ));
                }
            }

            // Input count check.
            let expected_inputs = match &group.op {
                ScalarOp::Literal(_) => 0,
                ScalarOp::Unary { .. } | ScalarOp::Identity { .. } => 1,
                ScalarOp::Binary { .. } => 2,
                ScalarOp::Select { .. } => 3,
                ScalarOp::ReduceSum { .. } | ScalarOp::ReduceMax { .. } => 1,
            };
            if group.inputs.len() != expected_inputs {
                errors.push(format!(
                    "Group {} (base={}): op {:?} expects {} inputs but has {}",
                    gi,
                    group.base_id,
                    group.op,
                    expected_inputs,
                    group.inputs.len()
                ));
            }
        }

        errors
    }
}

/// Summary statistics for a NanoGraph.
#[derive(Debug)]
pub struct NanoGraphStats {
    pub num_groups: u64,
    pub total_atoms: u64,
    pub singleton_groups: u64,
    pub symbolic_groups: u64,
    pub groups_by_op: HashMap<&'static str, u64>,
}

impl std::fmt::Display for NanoGraphStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} groups, {} atoms ({} singletons, {} symbolic)",
            self.num_groups, self.total_atoms, self.singleton_groups, self.symbolic_groups
        )?;
        if !self.groups_by_op.is_empty() {
            let mut sorted: Vec<_> = self.groups_by_op.iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(a.1));
            write!(f, "\n  Op breakdown:")?;
            for (op, count) in sorted {
                write!(f, "\n    {:>6}x {}", count, op)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::nano_graph::ops::{ScalarBinOp, ScalarOp, ScalarUnaryOp};
    use crate::numeric_scalar::NumericScalar;

    /// Build a tiny graph: c = a + b, elementwise over 1024 atoms.
    #[test]
    fn test_elementwise_add() {
        let mut g = NanoGraph::new();

        let a = g.push_group(
            1024,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );
        let b = g.push_group(
            1024,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );

        let c = g.push_group(
            1024,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine { base: a, stride: 1 },
                InputRef::Affine { base: b, stride: 1 },
            ],
        );

        assert!(g.validate().is_empty(), "{:?}", g.validate());
        let stats = g.stats();
        assert_eq!(stats.num_groups, 3);
        assert_eq!(stats.total_atoms, 3 * 1024);
        assert_eq!(g.num_atoms(), 3 * 1024);
        // c reads from a and b
        assert_eq!(g.group_of(c).unwrap().inputs.len(), 2);
    }

    /// Broadcast: c[i] = a[i] + scalar_b, 1024 atoms.
    #[test]
    fn test_broadcast() {
        let mut g = NanoGraph::new();

        let a = g.push_group(
            1024,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let b = g.push_atom(
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
            vec![],
        );

        let _c = g.push_group(
            1024,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine { base: a, stride: 1 },
                InputRef::Broadcast(b),
            ],
        );

        assert!(g.validate().is_empty(), "{:?}", g.validate());
        assert_eq!(g.num_groups(), 3);
    }

    /// Symbolic dimensions: add over [batch, 1024].
    #[test]
    fn test_symbolic_dim() {
        let mut g = NanoGraph::new();
        let batch = g.sym_dim("batch");

        let a = g.push_group(
            1024,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![batch],
            vec![],
            vec![],
        );
        let b = g.push_group(
            1024,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![batch],
            vec![],
            vec![],
        );
        let _c = g.push_group(
            1024,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![batch],
            vec![],
            vec![
                InputRef::Affine { base: a, stride: 1 },
                InputRef::Affine { base: b, stride: 1 },
            ],
        );

        assert!(g.validate().is_empty(), "{:?}", g.validate());
        let stats = g.stats();
        assert_eq!(stats.symbolic_groups, 3);
    }

    /// Reduction over a symbolic dim.
    #[test]
    fn test_reduce_symbolic() {
        let mut g = NanoGraph::new();
        let seq = g.sym_dim("seq_len");

        let input = g.push_group(
            768,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![seq],
            vec![],
            vec![],
        );

        // Reduce over seq_len: each of 768 hidden atoms accumulates over seq.
        let _reduced = g.push_group(
            768,
            ScalarOp::ReduceSum {
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![], // seq is reduced away
            vec![seq],
            vec![InputRef::Affine {
                base: input,
                stride: 1,
            }],
        );

        assert!(g.validate().is_empty(), "{:?}", g.validate());
        assert_eq!(g.num_groups(), 2);
    }

    /// Singleton atom for boundary ops.
    #[test]
    fn test_singleton_boundary() {
        let mut g = NanoGraph::new();
        let batch = g.sym_dim("batch");
        let seq = g.sym_dim("seq_len");

        // A Gather boundary: single atom, runtime-variable dims.
        let _gather = g.push_atom(
            ScalarOp::Identity {
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![batch, seq],
            vec![],
            vec![], // no inputs tracked (boundary)
        );

        assert_eq!(g.num_groups(), 1);
        assert_eq!(g.num_atoms(), 1);
        let stats = g.stats();
        assert_eq!(stats.singleton_groups, 1);
    }

    /// Explicit input ref for irregular addressing.
    #[test]
    fn test_explicit_input() {
        let mut g = NanoGraph::new();

        // 4 source atoms.
        let src = g.push_group(
            4,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );

        // 3 consumer atoms that pick from source irregularly: [2, 0, 3].
        let _consumer = g.push_group(
            3,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Explicit(vec![
                src.offset(2),
                src.offset(0),
                src.offset(3),
            ])],
        );

        assert!(g.validate().is_empty(), "{:?}", g.validate());
    }

    /// InputRef::resolve correctness.
    #[test]
    fn test_input_ref_resolve() {
        let base = AtomId(100);
        let broadcast = InputRef::Broadcast(AtomId(5));
        assert_eq!(broadcast.resolve(0, 0), AtomId(5));
        assert_eq!(broadcast.resolve(99, 0), AtomId(5));

        let affine = InputRef::Affine { base, stride: 2 };
        assert_eq!(affine.resolve(0, 0), AtomId(100));
        assert_eq!(affine.resolve(1, 0), AtomId(102));
        assert_eq!(affine.resolve(3, 0), AtomId(106));

        let explicit = InputRef::Explicit(vec![AtomId(10), AtomId(20), AtomId(30)]);
        assert_eq!(explicit.resolve(0, 0), AtomId(10));
        assert_eq!(explicit.resolve(1, 0), AtomId(20));
        assert_eq!(explicit.resolve(2, 0), AtomId(30));

        // SymAffine: base=100, stride_i=3, stride_k=10
        let sym = InputRef::SymAffine {
            base: AtomId(100),
            stride_i: 3,
            stride_k: 10,
        };
        assert_eq!(sym.resolve(0, 0), AtomId(100));
        assert_eq!(sym.resolve(1, 0), AtomId(103));
        assert_eq!(sym.resolve(0, 1), AtomId(110));
        assert_eq!(sym.resolve(2, 3), AtomId(100 + 6 + 30)); // 136
    }
}
