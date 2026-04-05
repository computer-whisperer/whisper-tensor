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

use crate::graph::GlobalId;
use crate::nano_graph::ops::ScalarOp;
use crate::numeric_dtype::NumericDType;
use crate::pool::Pool;
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
/// `InputRef::Strided` strides are applied.
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
    pub dtype: NumericDType,
}

/// How an input to an atom group references source atoms.
///
/// Multiple addressing modes coexist in the same graph — this is the key
/// difference from v1 where affine was the only option.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum InputRef {
    /// Every atom in this group reads the same source atom (broadcast).
    Broadcast(AtomId),
    /// N-dimensional strided access. Atom at offset `i` is decomposed into
    /// coordinates via row-major indexing into `dim_shape` (outer-to-inner),
    /// then dotted with `dim_strides`:
    ///
    ///   remaining = i
    ///   for d in (nd-1)..=1:
    ///       coord[d] = remaining % dim_shape[d]
    ///       remaining /= dim_shape[d]
    ///   coord[0] = remaining   // outermost: no modulus
    ///   result = base + sum(coord[d] * dim_strides[d])
    ///
    /// Common patterns via helpers:
    /// - `affine(stride)`:             1D, dim_strides=[stride], dim_shape=[MAX]
    /// - `modular(stride, mod)`:       2D, dim_strides=[0, stride], dim_shape=[MAX, mod]
    /// - `strided_broadcast(stride, repeat)`: 2D, dim_strides=[stride, 0], dim_shape=[MAX, repeat]
    Strided {
        base: AtomId,
        dim_strides: Vec<i64>,
        dim_shape: Vec<u64>,
    },
    /// Arbitrary per-atom source mapping. Used when no regular pattern exists
    /// (e.g., Gather with compile-time-known indices, irregular Concat).
    /// Length must equal the group's `count`.
    Explicit(Vec<AtomId>),
}

impl InputRef {
    /// Linear access: atom i reads base + stride * i.
    pub fn affine(base: AtomId, stride: i64) -> Self {
        InputRef::Strided {
            base,
            dim_strides: vec![stride],
            dim_shape: vec![u64::MAX],
        }
    }

    /// Modular access: atom i reads base + stride * (i % modulus).
    pub fn modular(base: AtomId, stride: i64, modulus: u64) -> Self {
        InputRef::Strided {
            base,
            dim_strides: vec![0, stride],
            dim_shape: vec![u64::MAX, modulus],
        }
    }

    /// Strided broadcast: atom i reads base + stride * (i / repeat).
    pub fn strided_broadcast(base: AtomId, stride: i64, repeat: u64) -> Self {
        InputRef::Strided {
            base,
            dim_strides: vec![stride, 0],
            dim_shape: vec![u64::MAX, repeat],
        }
    }

    /// Resolve the source atom for the `i`-th atom in the group.
    pub fn resolve(&self, i: u64) -> AtomId {
        match self {
            InputRef::Broadcast(id) => *id,
            InputRef::Strided {
                base,
                dim_strides,
                dim_shape,
            } => {
                let nd = dim_strides.len();
                let mut offset = 0i64;
                let mut remaining = i;
                // Inner to outer: decompose flat index into ND coordinates.
                for d in (0..nd).rev() {
                    let coord = if d == 0 {
                        remaining // outermost dim: no modulus
                    } else {
                        let c = remaining % dim_shape[d];
                        remaining /= dim_shape[d];
                        c
                    };
                    offset += coord as i64 * dim_strides[d];
                }
                AtomId(base.0.wrapping_add(offset as u64))
            }
            InputRef::Explicit(ids) => ids[i as usize],
        }
    }

    /// Number of distinct source atoms referenced (for compression stats).
    pub fn distinct_sources(&self, count: u64) -> usize {
        match self {
            InputRef::Broadcast(_) => 1,
            InputRef::Strided {
                dim_strides,
                dim_shape,
                ..
            } => {
                let nd = dim_strides.len();
                if nd == 1 {
                    // Affine
                    count as usize
                } else if nd == 2 && dim_strides[0] == 0 {
                    // Modular
                    dim_shape[1] as usize
                } else if nd == 2 && dim_strides[1] == 0 {
                    // StridedBroadcast
                    count.div_ceil(dim_shape[1]) as usize
                } else {
                    // General ND — estimate from count.
                    count as usize
                }
            }
            InputRef::Explicit(ids) => {
                let mut seen = ids.clone();
                seen.sort();
                seen.dedup();
                seen.len()
            }
        }
    }
}

/// A group of structurally identical scalar atoms.
///
/// All atoms in a group share the same op, symbolic dimensions,
/// and input addressing pattern. The group stores `count` atoms with
/// contiguous `AtomId`s starting at `base_id`.
///
/// A group of count=1 is a standalone atom — this is the degenerate case
/// for ops that don't compress (e.g., Gather boundary atoms).
#[derive(Debug)]
pub struct AtomGroup<'p, P: Pool + 'p = crate::pool::SystemPool> {
    /// First AtomId in this group.
    pub base_id: AtomId,
    /// Number of atoms in the group.
    pub count: u64,
    /// Logical offset for InputRef resolution.
    ///
    /// When a group is split (e.g., for lane partitioning), the second half
    /// needs its InputRefs to resolve as if it were still at the original
    /// position. `atom_offset` is added to the local index `i` before
    /// resolving InputRefs: `input.resolve(i + atom_offset)`.
    ///
    /// Zero for non-split groups (the common case). For a group split at
    /// position `s`, the second half has `base_id = original_base + s`,
    /// `count = original_count - s`, and `atom_offset = s`.
    pub atom_offset: u64,
    /// Element dtype of this group's output.
    pub output_dtype: NumericDType,
    /// The scalar operation each atom performs.
    pub op: ScalarOp<'p, P>,
    /// Symbolic dimensions this group iterates over.
    /// Each atom in the group independently iterates over these dims.
    /// E.g., `[batch, seq_len]` means each atom produces a 2D tile of values.
    pub sym_dims: Vec<SymDim>,
    /// Inputs to the operation. Number must match what `op` expects.
    pub inputs: Vec<InputRef>,
}

impl<'p, P: Pool + 'p> Clone for AtomGroup<'p, P>
where
    P::Buffer<'p>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            base_id: self.base_id,
            count: self.count,
            atom_offset: self.atom_offset,
            output_dtype: self.output_dtype,
            op: self.op.clone(),
            sym_dims: self.sym_dims.clone(),
            inputs: self.inputs.clone(),
        }
    }
}

impl<P: Pool> AtomGroup<'_, P> {
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
    pub dtype: NumericDType,
}

/// One entry from a liveness scan: an atom range and how many
/// downstream groups consume it.
#[derive(Debug, Clone)]
pub struct GroupUseCount {
    /// First atom ID in this group.
    pub base: AtomId,
    /// Number of atoms.
    pub count: u64,
    /// Element dtype (from the group's output_dtype).
    pub dtype: NumericDType,
    /// Number of downstream groups that read atoms from this group.
    /// Zero means this group's output is never consumed (dead code)
    /// or is a final output of the graph.
    pub use_count: u32,
}

/// The compressed scalar DAG for an entire computation.
pub struct NanoGraph<'p, P: Pool + 'p = crate::pool::SystemPool> {
    groups: RangeMap<AtomGroup<'p, P>>,
    next_atom_id: u64,
    /// External input tensors mapped into the AtomId space.
    input_ranges: RangeMap<InputTensor>,
    /// Opaque milli-ops that can't be decomposed into scalar nano-ops.
    /// AtomGroups with `ScalarOp::OpaqueOutput` reference these by index.
    opaque_ops: Vec<super::ops::OpaqueOp>,
    /// Named symbolic dimensions (e.g., "batch" → SymDim(0)).
    pub sym_dim_names: HashMap<String, SymDim>,
    /// Known upper bounds for symbolic dimensions.
    pub sym_dim_bounds: HashMap<SymDim, u64>,
    next_sym_dim: u16,
    /// Which atom ranges are final outputs of the computation.
    pub outputs: Vec<AtomRange>,
}

impl<'p, P: Pool + 'p> Clone for NanoGraph<'p, P>
where
    P::Buffer<'p>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            groups: self.groups.clone(),
            next_atom_id: self.next_atom_id,
            input_ranges: self.input_ranges.clone(),
            opaque_ops: self.opaque_ops.clone(),
            sym_dim_names: self.sym_dim_names.clone(),
            sym_dim_bounds: self.sym_dim_bounds.clone(),
            next_sym_dim: self.next_sym_dim,
            outputs: self.outputs.clone(),
        }
    }
}

impl<P: Pool> Default for NanoGraph<'_, P> {
    fn default() -> Self {
        Self {
            groups: RangeMap::new(),
            next_atom_id: 0,
            input_ranges: RangeMap::new(),
            opaque_ops: Vec::new(),
            sym_dim_names: HashMap::new(),
            sym_dim_bounds: HashMap::new(),
            next_sym_dim: 0,
            outputs: Vec::new(),
        }
    }
}

impl<'p, P: Pool + 'p> NanoGraph<'p, P> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an opaque op and create output AtomGroups for each output mapping.
    /// Returns the atom IDs for each output (in order).
    pub fn push_opaque_op(&mut self, mut op: super::ops::OpaqueOp) -> Vec<AtomId> {
        let opaque_idx = self.opaque_ops.len();
        let mut output_bases = Vec::with_capacity(op.outputs.len());

        for (output_idx, mapping) in op.outputs.iter_mut().enumerate() {
            let base_id = self.alloc_ids(mapping.count);
            mapping.base = base_id;

            // Create an AtomGroup for this output — references the opaque op by index.
            // The group's inputs reference the opaque op's input atom bases for liveness.
            let input_refs: Vec<InputRef> = op
                .inputs
                .iter()
                .map(|inp| InputRef::Broadcast(inp.base))
                .collect();

            self.groups.insert(
                base_id.0,
                mapping.count,
                AtomGroup {
                    base_id,
                    count: mapping.count,
                    atom_offset: 0,
                    output_dtype: mapping.dtype,
                    op: ScalarOp::OpaqueOutput {
                        opaque_idx,
                        output_idx,
                    },
                    sym_dims: vec![],
                    inputs: input_refs,
                },
            );

            output_bases.push(base_id);
        }

        self.opaque_ops.push(op);
        output_bases
    }

    /// Allocate contiguous atoms and create Identity groups that copy from a
    /// non-contiguous TensorAtomMap (segmented or non-row-major strides).
    /// Returns the base AtomId of the new contiguous range.
    pub fn alloc_contiguous_copy(
        &mut self,
        tam: &crate::nano_graph::lower::TensorAtomMap,
        count: u64,
    ) -> AtomId {
        let base = self.alloc_ids(count);
        let dtype = tam.dtype;
        // Build explicit mapping from each element to its source atom.
        let source_atoms: Vec<AtomId> = (0..count).map(|i| tam.atom_id_for_element(i)).collect();
        self.groups.insert(
            base.0,
            count,
            AtomGroup {
                base_id: base,
                count,
                atom_offset: 0,
                output_dtype: dtype,
                op: ScalarOp::Identity,
                sym_dims: vec![],
                inputs: vec![InputRef::Explicit(source_atoms)],
            },
        );
        base
    }

    /// Access the opaque ops list.
    pub fn opaque_ops(&self) -> &[super::ops::OpaqueOp] {
        &self.opaque_ops
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
    pub fn add_input_tensor(
        &mut self,
        tensor_id: GlobalId,
        count: u64,
        dtype: NumericDType,
    ) -> AtomId {
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
    pub fn alloc_placeholder(&mut self, count: u64, output_dtype: NumericDType) -> AtomId {
        let base_id = self.alloc_ids(count);
        self.groups.insert(
            base_id.0,
            count,
            AtomGroup {
                base_id,
                count,
                atom_offset: 0,
                output_dtype,
                op: ScalarOp::Literal(crate::numeric_scalar::NumericScalar::zero(output_dtype)),
                sym_dims: vec![],
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
        output_dtype: NumericDType,
        op: ScalarOp<'p, P>,
        sym_dims: Vec<SymDim>,
        inputs: Vec<InputRef>,
    ) {
        let idx = self
            .groups
            .find_index(base_id.0)
            .expect("fill_placeholder: base_id not found");
        let (_, _, g) = self.groups.get_by_index_mut(idx).unwrap();
        debug_assert_eq!(g.count, count, "fill_placeholder: count mismatch");
        g.output_dtype = output_dtype;
        g.op = op;
        g.sym_dims = sym_dims;
        g.inputs = inputs;
    }

    /// Insert an input tensor at a specific base AtomId.
    ///
    /// Unlike `add_input_tensor`, this does NOT allocate sequential IDs.
    /// Used for constructing span NanoGraphs that preserve the main graph's
    /// atom ID space. The RangeMap handles out-of-order insertion.
    pub fn insert_input_tensor_at(
        &mut self,
        base_id: AtomId,
        tensor_id: GlobalId,
        count: u64,
        dtype: NumericDType,
    ) {
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
        // Ensure next_atom_id stays past this range.
        let end = base_id.0 + count;
        if end > self.next_atom_id {
            self.next_atom_id = end;
        }
    }

    /// Insert a group at a specific base AtomId.
    ///
    /// Unlike `push_group`, this does NOT allocate sequential IDs.
    /// Used for constructing span NanoGraphs that preserve the main graph's
    /// atom ID space. The RangeMap handles out-of-order insertion.
    #[allow(clippy::too_many_arguments)]
    pub fn insert_group_at(
        &mut self,
        base_id: AtomId,
        count: u64,
        atom_offset: u64,
        output_dtype: NumericDType,
        op: ScalarOp<'p, P>,
        sym_dims: Vec<SymDim>,
        inputs: Vec<InputRef>,
    ) {
        self.groups.insert(
            base_id.0,
            count,
            AtomGroup {
                base_id,
                count,
                atom_offset,
                output_dtype,
                op,
                sym_dims,
                inputs,
            },
        );
        // Ensure next_atom_id stays past this range.
        let end = base_id.0 + count;
        if end > self.next_atom_id {
            self.next_atom_id = end;
        }
    }

    /// Add an atom group to the graph. Returns the base AtomId.
    pub fn push_group(
        &mut self,
        count: u64,
        output_dtype: NumericDType,
        op: ScalarOp<'p, P>,
        sym_dims: Vec<SymDim>,
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
                output_dtype,
                op,
                sym_dims,
                inputs,
            },
        );
        base_id
    }

    /// Convenience: push a single-atom group.
    pub fn push_atom(
        &mut self,
        output_dtype: NumericDType,
        op: ScalarOp<'p, P>,
        sym_dims: Vec<SymDim>,
        inputs: Vec<InputRef>,
    ) -> AtomId {
        self.push_group(1, output_dtype, op, sym_dims, inputs)
    }

    /// Find the group index for an AtomId. O(log n) via RangeMap.
    pub fn find_group_idx(&self, id: AtomId) -> Option<usize> {
        self.groups.find_index(id.0)
    }

    /// Look up which group an atom belongs to.
    pub fn group_of(&self, id: AtomId) -> Option<&AtomGroup<'p, P>> {
        self.groups.get(id.0).map(|(g, _)| g)
    }

    /// Look up group and offset for an atom.
    pub fn group_and_offset(&self, id: AtomId) -> Option<(&AtomGroup<'p, P>, u64)> {
        self.groups.get(id.0)
    }

    /// Check if an AtomId exists in any group or input tensor range.
    pub fn contains_atom(&self, id: AtomId) -> bool {
        self.groups.contains(id.0) || self.input_ranges.contains(id.0)
    }

    /// Iterate all groups in insertion order.
    pub fn groups(&self) -> &[AtomGroup<'p, P>] {
        self.groups.values()
    }

    pub fn groups_mut(&mut self) -> &mut [AtomGroup<'p, P>] {
        self.groups.values_mut()
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

    /// Build an `AtomRange` for an atom that belongs to either a group or an
    /// input tensor. Panics if `id` is not found in either.
    pub fn atom_to_range(&self, id: AtomId) -> AtomRange {
        if let Some(gi) = self.find_group_idx(id) {
            let g = &self.groups()[gi];
            AtomRange {
                base: g.base_id,
                count: g.count,
                dtype: g.output_dtype,
            }
        } else if let Some((ti, _)) = self.find_input_idx(id) {
            let it = &self.input_tensors()[ti];
            AtomRange {
                base: it.base_id,
                count: it.count,
                dtype: it.dtype,
            }
        } else {
            panic!(
                "atom_to_range: AtomId({}) not found in groups or inputs",
                id.0
            );
        }
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
        use crate::nano_graph::ops::ReduceKind;

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
                    crate::nano_graph::ScalarBinOp::IMod => "IMod",
                    crate::nano_graph::ScalarBinOp::Pow => "Pow",
                    crate::nano_graph::ScalarBinOp::Equal => "Equal",
                    crate::nano_graph::ScalarBinOp::Greater => "Greater",
                    crate::nano_graph::ScalarBinOp::GreaterOrEqual => "GreaterOrEqual",
                    crate::nano_graph::ScalarBinOp::Less => "Less",
                    crate::nano_graph::ScalarBinOp::LessOrEqual => "LessOrEqual",
                    crate::nano_graph::ScalarBinOp::And => "And",
                    crate::nano_graph::ScalarBinOp::Or => "Or",
                    crate::nano_graph::ScalarBinOp::Xor => "Xor",
                    crate::nano_graph::ScalarBinOp::BitwiseAnd => "BitwiseAnd",
                    crate::nano_graph::ScalarBinOp::BitwiseOr => "BitwiseOr",
                    crate::nano_graph::ScalarBinOp::BitwiseXor => "BitwiseXor",
                    crate::nano_graph::ScalarBinOp::BitShiftLeft => "BitShiftLeft",
                    crate::nano_graph::ScalarBinOp::BitShiftRight => "BitShiftRight",
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
                    crate::nano_graph::ScalarUnaryOp::Round => "Round",
                    crate::nano_graph::ScalarUnaryOp::Sign => "Sign",
                    crate::nano_graph::ScalarUnaryOp::Not => "Not",
                    crate::nano_graph::ScalarUnaryOp::IsNan => "IsNan",
                    crate::nano_graph::ScalarUnaryOp::Erf => "Erf",
                    crate::nano_graph::ScalarUnaryOp::Sin => "Sin",
                    crate::nano_graph::ScalarUnaryOp::Cos => "Cos",
                    crate::nano_graph::ScalarUnaryOp::IsInf { .. } => "IsInf",
                    crate::nano_graph::ScalarUnaryOp::BitwiseNot => "BitwiseNot",
                    crate::nano_graph::ScalarUnaryOp::Log1p => "Log1p",
                    crate::nano_graph::ScalarUnaryOp::Tan => "Tan",
                    crate::nano_graph::ScalarUnaryOp::Asin => "Asin",
                    crate::nano_graph::ScalarUnaryOp::Acos => "Acos",
                    crate::nano_graph::ScalarUnaryOp::Atan => "Atan",
                    crate::nano_graph::ScalarUnaryOp::Sinh => "Sinh",
                    crate::nano_graph::ScalarUnaryOp::Cosh => "Cosh",
                    crate::nano_graph::ScalarUnaryOp::Asinh => "Asinh",
                    crate::nano_graph::ScalarUnaryOp::Acosh => "Acosh",
                    crate::nano_graph::ScalarUnaryOp::Atanh => "Atanh",
                },
                ScalarOp::Identity => "Identity",
                ScalarOp::Cast { .. } => "Cast",
                ScalarOp::Literal(_) => "Literal",
                ScalarOp::Select => "Select",
                ScalarOp::Reduce {
                    kind: ReduceKind::Sum,
                    ..
                } => "ReduceSum",
                ScalarOp::Reduce {
                    kind: ReduceKind::Max,
                    ..
                } => "ReduceMax",
                ScalarOp::Reduce {
                    kind: ReduceKind::Min,
                    ..
                } => "ReduceMin",
                ScalarOp::Reduce {
                    kind: ReduceKind::Prod,
                    ..
                } => "ReduceProd",
                ScalarOp::IndirectLoad { .. } => "IndirectLoad",
                ScalarOp::OpaqueOutput { .. } => "OpaqueOutput",
                ScalarOp::LiteralSpan(_) => "LiteralSpan",
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
            self.collect_all_producer_indices(group, gi, &mut seen);
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
                dtype: g.output_dtype,
                use_count: uc,
            })
            .collect()
    }

    /// Collect all producer group indices for a group: InputRef producers,
    /// reduce-stride ranges, and IndirectLoad table references.
    ///
    /// Removes self-references (the group's own index `gi`).
    /// Used by both `liveness()` and the evaluator.
    pub fn collect_all_producer_indices(
        &self,
        group: &AtomGroup<'p, P>,
        gi: usize,
        out: &mut std::collections::HashSet<usize>,
    ) {
        // OpaqueOutput groups read their inputs' full atom ranges at eval time,
        // not just the broadcast base. Use the actual input mappings.
        if let ScalarOp::OpaqueOutput { opaque_idx, .. } = &group.op {
            let opaque_op = &self.opaque_ops[*opaque_idx];
            for inp in &opaque_op.inputs {
                if inp.count > 0 {
                    let lo = inp.base.0;
                    let hi = lo + inp.count - 1;
                    self.insert_groups_in_id_range(lo, hi, out);
                }
            }
            out.remove(&gi);
            return;
        }

        for input in &group.inputs {
            self.collect_producer_indices(input, group.count, group.atom_offset, out);
        }

        // Reduce ops access additional atoms via stride.
        if let ScalarOp::Reduce {
            reduce_count,
            reduce_stride,
            ..
        } = &group.op
            && *reduce_count > 1
            && *reduce_stride != 0
        {
            for input in &group.inputs {
                let first = input.resolve(group.atom_offset);
                let last = input.resolve(group.atom_offset + group.count - 1);
                let end_off = (*reduce_count as i64 - 1) * reduce_stride;
                let endpoints = [
                    first.0,
                    (first.0 as i64 + end_off) as u64,
                    last.0,
                    (last.0 as i64 + end_off) as u64,
                ];
                let lo = *endpoints.iter().min().unwrap();
                let hi = *endpoints.iter().max().unwrap();
                self.insert_groups_in_id_range(lo, hi, out);
            }
        }

        // IndirectLoad table reference.
        if let ScalarOp::IndirectLoad { table_base, .. } = &group.op
            && let Some(pi) = self.groups.find_index(table_base.0)
        {
            out.insert(pi);
        }

        out.remove(&gi);
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
            InputRef::Strided {
                base,
                dim_strides,
                dim_shape,
            } => {
                // For modular patterns (2D with outer stride=0): the range
                // spans base to base + inner_stride * (modulus - 1).
                let is_modular = dim_strides.len() >= 2
                    && dim_strides[0] == 0
                    && dim_shape.last().copied().unwrap_or(u64::MAX) != u64::MAX;
                if is_modular {
                    let inner_stride = *dim_strides.last().unwrap();
                    let modulus = *dim_shape.last().unwrap();
                    let a = base.0;
                    let b = (base.0 as i64 + inner_stride * (modulus as i64 - 1)) as u64;
                    self.insert_groups_in_id_range(a.min(b), a.max(b), out);
                } else {
                    // General case: resolve first and last to find the range.
                    let first = input.resolve(atom_offset);
                    let last = input.resolve(atom_offset + count - 1);
                    let lo = first.0.min(last.0);
                    let hi = first.0.max(last.0);
                    self.insert_groups_in_id_range(lo, hi, out);
                }
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

                for &i in sample_positions.iter().chain(extra_positions.iter()) {
                    if i >= group.count {
                        continue;
                    }
                    let source = input.resolve(i + group.atom_offset);

                    // Check existence.
                    if !self.contains_atom(source) {
                        errors.push(format!(
                            "Group {} (base={}) input {} atom {}: references nonexistent atom {}",
                            gi, group.base_id, inp_idx, i, source
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

                // For reduce ops, check the reduce-strided access range.
                if let ScalarOp::Reduce {
                    reduce_count,
                    reduce_stride,
                    ..
                } = &group.op
                    && *reduce_count > 1
                    && *reduce_stride != 0
                {
                    // Check from first and last consumer atom.
                    for &i in &[0u64, group.count.saturating_sub(1)] {
                        let base_atom = input.resolve(i + group.atom_offset);
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
                ScalarOp::Literal(_) | ScalarOp::LiteralSpan(_) => 0,
                ScalarOp::Unary { .. }
                | ScalarOp::Identity
                | ScalarOp::Cast { .. }
                | ScalarOp::IndirectLoad { .. } => 1,
                ScalarOp::Binary { .. } => 2,
                ScalarOp::Select => 3,
                ScalarOp::Reduce { .. } => 1,
                // OpaqueOutput inputs are Broadcast refs to opaque op's input bases (for liveness).
                // The count varies — skip the check.
                ScalarOp::OpaqueOutput { .. } => {
                    continue;
                }
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
    use crate::nano_graph::ops::{ReduceKind, ScalarBinOp, ScalarOp, ScalarUnaryOp};
    use crate::numeric_dtype::NumericDType;
    use crate::numeric_scalar::NumericScalar;
    use crate::pool::SystemPool;

    type TestGraph = NanoGraph<'static, SystemPool>;

    /// Build a tiny graph: c = a + b, elementwise over 1024 atoms.
    #[test]
    fn test_elementwise_add() {
        let mut g = TestGraph::new();

        let a = g.push_group(
            1024,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(0.0)),
            vec![],
            vec![],
        );
        let b = g.push_group(
            1024,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(0.0)),
            vec![],
            vec![],
        );

        let c = g.push_group(
            1024,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(a, 1), InputRef::affine(b, 1)],
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
        let mut g = TestGraph::new();

        let a = g.push_group(
            1024,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(1.0)),
            vec![],
            vec![],
        );
        let b = g.push_atom(
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(2.0)),
            vec![],
            vec![],
        );

        let _c = g.push_group(
            1024,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(a, 1), InputRef::Broadcast(b)],
        );

        assert!(g.validate().is_empty(), "{:?}", g.validate());
        assert_eq!(g.num_groups(), 3);
    }

    /// Symbolic dimensions: add over [batch, 1024].
    #[test]
    fn test_symbolic_dim() {
        let mut g = TestGraph::new();
        let batch = g.sym_dim("batch");

        let a = g.push_group(
            1024,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(0.0)),
            vec![batch],
            vec![],
        );
        let b = g.push_group(
            1024,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(0.0)),
            vec![batch],
            vec![],
        );
        let _c = g.push_group(
            1024,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![batch],
            vec![InputRef::affine(a, 1), InputRef::affine(b, 1)],
        );

        assert!(g.validate().is_empty(), "{:?}", g.validate());
        let stats = g.stats();
        assert_eq!(stats.symbolic_groups, 3);
    }

    /// Reduction over a symbolic dim.
    #[test]
    fn test_reduce_symbolic() {
        let mut g = TestGraph::new();
        let seq = g.sym_dim("seq_len");

        let input = g.push_group(
            768,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(1.0)),
            vec![seq],
            vec![],
        );

        // Reduce over seq_len: each of 768 hidden atoms accumulates over seq.
        // reduce_count=0 signals "driven by SymDim" rather than a known count.
        let _reduced = g.push_group(
            768,
            NumericDType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: 0,
                reduce_stride: 0,
                compute_dtype: NumericDType::F32,
            },
            vec![], // seq is reduced away
            vec![InputRef::affine(input, 1)],
        );

        assert!(g.validate().is_empty(), "{:?}", g.validate());
        assert_eq!(g.num_groups(), 2);
    }

    /// Singleton atom for boundary ops.
    #[test]
    fn test_singleton_boundary() {
        let mut g = TestGraph::new();
        let batch = g.sym_dim("batch");
        let seq = g.sym_dim("seq_len");

        // A Gather boundary: single atom, runtime-variable dims.
        let _gather = g.push_atom(
            NumericDType::F32,
            ScalarOp::Identity,
            vec![batch, seq],
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
        let mut g = TestGraph::new();

        // 4 source atoms.
        let src = g.push_group(
            4,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(0.0)),
            vec![],
            vec![],
        );

        // 3 consumer atoms that pick from source irregularly: [2, 0, 3].
        let _consumer = g.push_group(
            3,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
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
        assert_eq!(broadcast.resolve(0), AtomId(5));
        assert_eq!(broadcast.resolve(99), AtomId(5));

        let affine = InputRef::affine(base, 2);
        assert_eq!(affine.resolve(0), AtomId(100));
        assert_eq!(affine.resolve(1), AtomId(102));
        assert_eq!(affine.resolve(3), AtomId(106));

        let explicit = InputRef::Explicit(vec![AtomId(10), AtomId(20), AtomId(30)]);
        assert_eq!(explicit.resolve(0), AtomId(10));
        assert_eq!(explicit.resolve(1), AtomId(20));
        assert_eq!(explicit.resolve(2), AtomId(30));

        // StridedBroadcast: base=200, stride=1, repeat=4
        // Blocks of 4 atoms share the same source.
        let sb = InputRef::strided_broadcast(AtomId(200), 1, 4);
        assert_eq!(sb.resolve(0), AtomId(200)); // block 0
        assert_eq!(sb.resolve(1), AtomId(200)); // block 0
        assert_eq!(sb.resolve(3), AtomId(200)); // block 0
        assert_eq!(sb.resolve(4), AtomId(201)); // block 1
        assert_eq!(sb.resolve(7), AtomId(201)); // block 1
        assert_eq!(sb.resolve(8), AtomId(202)); // block 2
    }
}
