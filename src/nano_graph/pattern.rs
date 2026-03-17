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

use crate::dtype::DType;
use crate::graph::GlobalId;
use crate::nano_graph::ops::ScalarOp;
use crate::range_map::RangeMap;
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
pub struct AtomId(pub u64);

impl AtomId {
    pub fn offset(self, n: u64) -> Self {
        AtomId(self.0 + n)
    }
}

impl std::fmt::Display for AtomId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "a{}", self.0)
    }
}

/// A contiguous range of atoms with a known element dtype.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtomRange {
    pub base: AtomId,
    pub count: u64,
    pub dtype: DType,
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
    /// Strided broadcast: each block of `repeat` consecutive atoms shares one
    /// source atom. Atom at offset `i` reads `base + stride * (i / repeat)`.
    /// Used for merged matmul Mul groups where chunks of N atoms broadcast the
    /// same A element.
    StridedBroadcast {
        base: AtomId,
        stride: i64,
        repeat: u64,
    },
    /// Modular/tiling access: atom at offset `i` reads `base + stride * (i % modulus)`.
    /// Used when a smaller tensor tiles/repeats to fill a larger consumer
    /// (e.g., bias broadcast along batch dimension).
    Modular {
        base: AtomId,
        stride: i32,
        modulus: u64,
    },
}

impl InputRef {
    /// Resolve the source atom for the `i`-th atom in the group,
    /// at sym_dim iteration `k` (ignored for non-SymAffine variants).
    pub fn resolve(&self, i: u64, k: u64) -> AtomId {
        match self {
            InputRef::Broadcast(id) => *id,
            InputRef::Affine { base, stride } => {
                AtomId(base.0.wrapping_add((*stride as i64 * i as i64) as u64))
            }
            InputRef::Explicit(ids) => ids[i as usize],
            InputRef::SymAffine {
                base,
                stride_i,
                stride_k,
            } => {
                AtomId(base.0.wrapping_add(
                    (*stride_i as i64 * i as i64 + *stride_k as i64 * k as i64) as u64,
                ))
            }
            InputRef::StridedBroadcast {
                base,
                stride,
                repeat,
            } => {
                let block = i / repeat;
                AtomId(base.0.wrapping_add((*stride * block as i64) as u64))
            }
            InputRef::Modular {
                base,
                stride,
                modulus,
            } => {
                let wrapped = i % modulus;
                AtomId(
                    base.0
                        .wrapping_add((*stride as i64 * wrapped as i64) as u64),
                )
            }
        }
    }

    /// Number of distinct source atoms referenced (for compression stats).
    pub fn distinct_sources(&self, count: u64) -> usize {
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
            InputRef::StridedBroadcast { repeat, .. } => {
                // Each block of `repeat` atoms shares one source.
                count.div_ceil(*repeat) as usize
            }
            InputRef::Modular { modulus, .. } => *modulus as usize,
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
    pub count: u64,
    /// Logical offset for InputRef resolution.
    ///
    /// When a group is split (e.g., for lane partitioning), the second half
    /// needs its InputRefs to resolve as if it were still at the original
    /// position. `atom_offset` is added to the local index `i` before
    /// resolving InputRefs: `input.resolve(i + atom_offset, k)`.
    ///
    /// Zero for non-split groups (the common case). For a group split at
    /// position `s`, the second half has `base_id = original_base + s`,
    /// `count = original_count - s`, and `atom_offset = s`.
    pub atom_offset: u64,
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
    pub fn offset_of(&self, id: AtomId) -> Option<u64> {
        if self.contains(id) {
            Some(id.0 - self.base_id.0)
        } else {
            None
        }
    }
}

/// An external tensor mapped into the graph's AtomId space.
/// The executor uses this to know where to load tensor data.
#[derive(Debug, Clone)]
pub struct InputTensor {
    /// The milli-graph tensor ID.
    pub tensor_id: GlobalId,
    /// First AtomId allocated for this tensor.
    pub base_id: AtomId,
    /// Number of atoms (elements) in the tensor.
    pub count: u64,
    /// Element dtype.
    pub dtype: DType,
}

/// One entry from a liveness scan: an atom range and how many
/// downstream groups consume it.
#[derive(Debug, Clone)]
pub struct GroupUseCount {
    /// First atom ID in this group.
    pub base: AtomId,
    /// Number of atoms.
    pub count: u64,
    /// Element dtype (from the group's op).
    pub dtype: DType,
    /// Number of downstream groups that read atoms from this group.
    /// Zero means this group's output is never consumed (dead code)
    /// or is a final output of the graph.
    pub use_count: u32,
}

/// The compressed scalar DAG for an entire computation.
#[derive(Default, Clone)]
pub struct NanoGraph {
    groups: RangeMap<AtomGroup>,
    next_atom_id: u64,
    /// External input tensors mapped into the AtomId space.
    /// These are NOT groups — they occupy atom IDs that compute groups
    /// reference via InputRefs, but they have no ScalarOp. The executor
    /// fills these ranges from the TensorStore or user-provided data.
    input_ranges: RangeMap<InputTensor>,
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

    /// Reserve an AtomId range for an external input tensor.
    /// Returns the base AtomId. No group is created — the executor fills
    /// these atoms from the TensorStore or user data at runtime.
    pub fn add_input_tensor(&mut self, tensor_id: GlobalId, count: u64, dtype: DType) -> AtomId {
        let base_id = self.alloc_ids(count);
        self.input_ranges.insert(
            base_id.0,
            count,
            InputTensor {
                tensor_id,
                base_id,
                count,
                dtype,
            },
        );
        base_id
    }

    /// Allocate `count` contiguous AtomIds. Returns the base id.
    fn alloc_ids(&mut self, count: u64) -> AtomId {
        let base = AtomId(self.next_atom_id);
        self.next_atom_id = self
            .next_atom_id
            .checked_add(count)
            .expect("AtomId overflow");
        base
    }

    /// Allocate space for a group without filling in its fields yet.
    /// Returns the base AtomId. The group is initially a no-op Literal placeholder.
    /// Call `fill_placeholder` to set the actual op and inputs.
    pub fn alloc_placeholder(&mut self, count: u64) -> AtomId {
        let base_id = self.alloc_ids(count);
        self.groups.insert(
            base_id.0,
            count,
            AtomGroup {
                base_id,
                count,
                atom_offset: 0,
                op: ScalarOp::Literal(crate::numeric_scalar::NumericScalar::F32(0.0)),
                sym_dims: vec![],
                reduce_dims: vec![],
                inputs: vec![],
            },
        );
        base_id
    }

    /// Fill in a previously allocated placeholder group.
    /// The `base_id` must match one returned by `alloc_placeholder`.
    pub fn fill_placeholder(
        &mut self,
        base_id: AtomId,
        count: u64,
        op: ScalarOp,
        sym_dims: Vec<SymDim>,
        reduce_dims: Vec<SymDim>,
        inputs: Vec<InputRef>,
    ) {
        let idx = self
            .groups
            .find_index(base_id.0)
            .expect("fill_placeholder: base_id not found");
        let (_, _, g) = self.groups.get_by_index_mut(idx).unwrap();
        debug_assert_eq!(g.count, count, "fill_placeholder: count mismatch");
        g.op = op;
        g.sym_dims = sym_dims;
        g.reduce_dims = reduce_dims;
        g.inputs = inputs;
    }

    /// Add an atom group to the graph. Returns the base AtomId.
    pub fn push_group(
        &mut self,
        count: u64,
        op: ScalarOp,
        sym_dims: Vec<SymDim>,
        reduce_dims: Vec<SymDim>,
        inputs: Vec<InputRef>,
    ) -> AtomId {
        let base_id = self.alloc_ids(count);
        self.groups.insert(
            base_id.0,
            count,
            AtomGroup {
                base_id,
                count,
                atom_offset: 0,
                op,
                sym_dims,
                reduce_dims,
                inputs,
            },
        );
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

    /// Find the group index for an AtomId. O(log n) via RangeMap.
    pub fn find_group_idx(&self, id: AtomId) -> Option<usize> {
        self.groups.find_index(id.0)
    }

    /// Look up which group an atom belongs to.
    pub fn group_of(&self, id: AtomId) -> Option<&AtomGroup> {
        self.groups.get(id.0).map(|(g, _)| g)
    }

    /// Look up group and offset for an atom.
    pub fn group_and_offset(&self, id: AtomId) -> Option<(&AtomGroup, u64)> {
        self.groups.get(id.0)
    }

    /// Check if an AtomId exists in any group or input tensor range.
    pub fn contains_atom(&self, id: AtomId) -> bool {
        self.groups.contains(id.0) || self.input_ranges.contains(id.0)
    }

    /// Iterate all groups in insertion order.
    pub fn groups(&self) -> &[AtomGroup] {
        self.groups.values()
    }

    /// Access the input tensors.
    pub fn input_tensors(&self) -> &[InputTensor] {
        self.input_ranges.values()
    }

    /// Find which input tensor an AtomId belongs to.
    /// Returns `(index_into_input_tensors, offset_within_tensor)`.
    pub fn find_input_idx(&self, id: AtomId) -> Option<(usize, u64)> {
        self.input_ranges
            .find_index(id.0)
            .map(|idx| (idx, id.0 - self.input_ranges.values()[idx].base_id.0))
    }

    /// Number of groups.
    pub fn num_groups(&self) -> usize {
        self.groups.len()
    }

    /// Total number of atoms (including input tensor ranges).
    pub fn num_atoms(&self) -> u64 {
        self.next_atom_id
    }

    /// Summary stats for debugging.
    pub fn stats(&self) -> NanoGraphStats {
        let mut total_atoms: u64 = 0;
        let mut singleton_groups: u64 = 0;
        let mut symbolic_groups: u64 = 0;
        let mut groups_by_op: HashMap<&'static str, u64> = HashMap::new();

        for group in self.groups.values() {
            total_atoms += group.count;
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
                    crate::nano_graph::ScalarBinOp::Equal => "Equal",
                    crate::nano_graph::ScalarBinOp::Greater => "Greater",
                    crate::nano_graph::ScalarBinOp::GreaterOrEqual => "GreaterOrEqual",
                    crate::nano_graph::ScalarBinOp::Less => "Less",
                    crate::nano_graph::ScalarBinOp::LessOrEqual => "LessOrEqual",
                    crate::nano_graph::ScalarBinOp::And => "And",
                    crate::nano_graph::ScalarBinOp::Or => "Or",
                    crate::nano_graph::ScalarBinOp::Xor => "Xor",
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
                ScalarOp::IndirectLoad { .. } => "IndirectLoad",
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

    /// Compute group-level liveness: for each group, how many downstream
    /// groups consume its atoms.
    ///
    /// Returns one entry per group in insertion order. Runs in
    /// O(num_groups × inputs_per_group), not O(num_atoms).
    pub fn liveness(&self) -> Vec<GroupUseCount> {
        let groups = self.groups.values();
        let n = groups.len();
        let mut use_counts = vec![0u32; n];

        for (gi, group) in groups.iter().enumerate() {
            let mut seen = std::collections::HashSet::<usize>::new();

            for input in &group.inputs {
                self.collect_producer_indices(input, group.count, group.atom_offset, &mut seen);
            }

            // Reduce ops access additional atoms via stride.
            match &group.op {
                ScalarOp::ReduceSum {
                    reduce_count,
                    reduce_stride,
                    ..
                }
                | ScalarOp::ReduceMax {
                    reduce_count,
                    reduce_stride,
                    ..
                } if *reduce_count > 1 && *reduce_stride != 0 => {
                    for input in &group.inputs {
                        let first = input.resolve(group.atom_offset, 0);
                        let last = input.resolve(group.atom_offset + group.count - 1, 0);
                        let end_off = (*reduce_count as i64 - 1) * reduce_stride;
                        let endpoints = [
                            first.0,
                            (first.0 as i64 + end_off) as u64,
                            last.0,
                            (last.0 as i64 + end_off) as u64,
                        ];
                        let lo = *endpoints.iter().min().unwrap();
                        let hi = *endpoints.iter().max().unwrap();
                        self.insert_groups_in_id_range(lo, hi, &mut seen);
                    }
                }
                _ => {}
            }

            // IndirectLoad table reference.
            if let ScalarOp::IndirectLoad { table_base, .. } = &group.op
                && let Some(pi) = self.groups.find_index(table_base.0)
            {
                seen.insert(pi);
            }

            seen.remove(&gi);
            for pi in seen {
                use_counts[pi] += 1;
            }
        }

        groups
            .iter()
            .zip(use_counts)
            .map(|(g, uc)| GroupUseCount {
                base: g.base_id,
                count: g.count,
                dtype: g.op.output_dtype(),
                use_count: uc,
            })
            .collect()
    }

    /// Collect producer group indices for an InputRef (group-level, not per-atom).
    fn collect_producer_indices(
        &self,
        input: &InputRef,
        count: u64,
        atom_offset: u64,
        out: &mut std::collections::HashSet<usize>,
    ) {
        if count == 0 {
            return;
        }
        match input {
            InputRef::Broadcast(base) => {
                if let Some(gi) = self.groups.find_index(base.0) {
                    out.insert(gi);
                }
            }
            InputRef::Affine { .. }
            | InputRef::SymAffine { .. }
            | InputRef::StridedBroadcast { .. } => {
                let first = input.resolve(atom_offset, 0);
                let last = input.resolve(atom_offset + count - 1, 0);
                let lo = first.0.min(last.0);
                let hi = first.0.max(last.0);
                self.insert_groups_in_id_range(lo, hi, out);
            }
            InputRef::Modular {
                base,
                stride,
                modulus,
            } => {
                let a = base.0;
                let b = (base.0 as i64 + *stride as i64 * (*modulus as i64 - 1)) as u64;
                self.insert_groups_in_id_range(a.min(b), a.max(b), out);
            }
            InputRef::Explicit(ids) => {
                let mut prev_gi: Option<usize> = None;
                for id in ids.iter().skip(atom_offset as usize).take(count as usize) {
                    let gi = self.groups.find_index(id.0);
                    if gi != prev_gi {
                        if let Some(g) = gi {
                            out.insert(g);
                        }
                        prev_gi = gi;
                    }
                }
            }
        }
    }

    /// Insert all group indices whose atom ranges overlap [lo, hi].
    fn insert_groups_in_id_range(
        &self,
        lo: u64,
        hi: u64,
        out: &mut std::collections::HashSet<usize>,
    ) {
        let groups = self.groups.values();
        if let Some(first_gi) = self.groups.find_index(lo) {
            out.insert(first_gi);
            for (gi, group) in groups.iter().enumerate().skip(first_gi + 1) {
                if group.base_id.0 > hi {
                    break;
                }
                out.insert(gi);
            }
        } else if let Some(gi) = self.groups.find_index(hi) {
            // lo might be in an input_tensor range; try hi.
            out.insert(gi);
        }
    }

    /// Validate structural invariants. Returns a list of errors.
    /// Validate the NanoGraph structure. Runs in O(groups), not O(atoms).
    ///
    /// Checks:
    /// - All InputRefs resolve to existing atoms (sample-checked: first, last, mid)
    /// - Topological ordering: inputs only reference earlier groups (no self/forward refs)
    /// - Reduce stride ranges stay within earlier groups
    /// - Explicit InputRef length matches group count
    /// - Input count matches op expectation
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        let groups = self.groups.values();

        for (gi, group) in groups.iter().enumerate() {
            for (inp_idx, input) in group.inputs.iter().enumerate() {
                // Determine k values to check for SymAffine.
                let k_vals: Vec<u64> = if matches!(input, InputRef::SymAffine { .. }) {
                    let k_max = group
                        .reduce_dims
                        .iter()
                        .filter_map(|sd| self.sym_dim_bounds.get(sd).copied())
                        .next();
                    let mut v = vec![0u64];
                    if let Some(km) = k_max
                        && km > 1
                    {
                        v.push(km - 1);
                    }
                    v
                } else {
                    vec![0u64]
                };

                // Sample positions: first, last, and middle.
                let sample_positions: Vec<u64> = if group.count <= 3 {
                    (0..group.count).collect()
                } else {
                    vec![0, group.count / 2, group.count - 1]
                };

                // For Explicit, also check unique atoms that might span multiple groups.
                let extra_positions: Vec<u64> = if let InputRef::Explicit(ids) = input {
                    // Find positions where the atom ID jumps (different source groups).
                    let mut extras = Vec::new();
                    let mut prev_gi = None;
                    for (i, id) in ids.iter().enumerate() {
                        let cur_gi = self.groups.find_index(id.0);
                        if cur_gi != prev_gi {
                            extras.push(i as u64);
                            prev_gi = cur_gi;
                        }
                    }
                    extras
                } else {
                    vec![]
                };

                for &k in &k_vals {
                    for &i in sample_positions.iter().chain(extra_positions.iter()) {
                        if i >= group.count {
                            continue;
                        }
                        let source = input.resolve(i + group.atom_offset, k);

                        // Check existence.
                        if !self.contains_atom(source) {
                            errors.push(format!(
                                "Group {} (base={}) input {} atom {} k={}: references nonexistent atom {}",
                                gi, group.base_id, inp_idx, i, k, source
                            ));
                            break;
                        }

                        // Check topological ordering: source must be in an earlier group.
                        if let Some(src_gi) = self.groups.find_index(source.0)
                            && src_gi >= gi
                        {
                            errors.push(format!(
                                "Group {} (base={}) input {} atom {}: references group {} (base={}) — not earlier (self/forward reference)",
                                gi, group.base_id, inp_idx, i,
                                src_gi, groups[src_gi].base_id,
                            ));
                            break;
                        }
                    }
                }

                // For reduce ops, check the reduce-strided access range.
                match &group.op {
                    ScalarOp::ReduceSum {
                        reduce_count,
                        reduce_stride,
                        ..
                    }
                    | ScalarOp::ReduceMax {
                        reduce_count,
                        reduce_stride,
                        ..
                    } if *reduce_count > 1 && *reduce_stride != 0 => {
                        // Check from first and last consumer atom.
                        for &i in &[0u64, group.count.saturating_sub(1)] {
                            let base_atom = input.resolve(i + group.atom_offset, 0);
                            let last_k_atom = AtomId(
                                (base_atom.0 as i64 + (*reduce_count as i64 - 1) * reduce_stride)
                                    as u64,
                            );
                            if !self.contains_atom(last_k_atom) {
                                errors.push(format!(
                                    "Group {} (base={}) input {}: reduce stride from atom {} reaches nonexistent atom {}",
                                    gi, group.base_id, inp_idx, i, last_k_atom,
                                ));
                            } else if let Some(src_gi) = self.groups.find_index(last_k_atom.0)
                                && src_gi >= gi
                            {
                                errors.push(format!(
                                    "Group {} (base={}) input {}: reduce stride reaches group {} which is not earlier",
                                    gi, group.base_id, inp_idx, src_gi,
                                ));
                            }
                        }
                    }
                    _ => {}
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
                ScalarOp::Unary { .. }
                | ScalarOp::Identity { .. }
                | ScalarOp::IndirectLoad { .. } => 1,
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
        // reduce_count=0 signals "driven by SymDim" rather than a known count.
        let _reduced = g.push_group(
            768,
            ScalarOp::ReduceSum {
                reduce_count: 0,
                reduce_stride: 0,
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

        // StridedBroadcast: base=200, stride=1, repeat=4
        // Blocks of 4 atoms share the same source.
        let sb = InputRef::StridedBroadcast {
            base: AtomId(200),
            stride: 1,
            repeat: 4,
        };
        assert_eq!(sb.resolve(0, 0), AtomId(200)); // block 0
        assert_eq!(sb.resolve(1, 0), AtomId(200)); // block 0
        assert_eq!(sb.resolve(3, 0), AtomId(200)); // block 0
        assert_eq!(sb.resolve(4, 0), AtomId(201)); // block 1
        assert_eq!(sb.resolve(7, 0), AtomId(201)); // block 1
        assert_eq!(sb.resolve(8, 0), AtomId(202)); // block 2
    }
}
