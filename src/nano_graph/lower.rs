//! Lowering from MilliOpGraph to NanoGraph.
//!
//! Walks the MilliOpGraph in topological order, classifying each tensor's
//! dimensions as Known (expanded to atoms) or Symbolic (iteration params).
//! View ops dissolve into addressing changes. Elementwise ops become groups
//! with the same atom count. Ops that don't fit become boundary groups.

use std::collections::HashMap;

use crate::graph::{GlobalId, Graph, Node};
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::AnyMilliOp;
use crate::nano_graph::ops::ScalarOp;
use crate::nano_graph::pattern::{AtomId, InputRef, NanoGraph, SymDim};
use crate::numeric_dtype::NumericDType;
use crate::numeric_scalar::NumericScalar;
use crate::pool::SystemPool;
use crate::tensor_info::TensorInfo;

/// TensorInfo type used during lowering — generic over the pool.
pub type LowerTensorInfo<'a, 'p, P = SystemPool> = TensorInfo<'a, 'p, P>;

/// Wrapper that calls `AnyMilliOp::eval_new` through the `OpaqueEval` trait.
///
/// Used by `lower_default` to make any MilliOp executable through pool_eval
/// without nano decomposition.
struct MilliOpOpaqueEval(crate::milli_graph::ops::AnyMilliOp);

// Safety: AnyMilliOp derives Clone + Serialize + Debug, and all op fields are
// plain data (GlobalId, numeric params). No thread-unsafe state.
unsafe impl Send for MilliOpOpaqueEval {}
unsafe impl Sync for MilliOpOpaqueEval {}

impl crate::nano_graph::ops::OpaqueEval for MilliOpOpaqueEval {
    fn eval(
        &self,
        inputs: &[crate::numeric_tensor::NumericTensorView<'_, crate::tensor_rank::DynRank>],
    ) -> Result<
        Vec<
            crate::numeric_tensor::NumericTensor<
                'static,
                crate::tensor_rank::DynRank,
                crate::pool::SystemPool,
            >,
        >,
        crate::nano_graph::pool_eval::PoolEvalError,
    > {
        use crate::milli_graph::ops::MilliOp;
        static POOL: crate::pool::SystemPool = crate::pool::SystemPool;
        self.0.eval_new(inputs, &POOL)
    }
}

/// Common accessors for reduce ops (ReduceSum, ReduceMax, ReduceMean).
pub trait ReduceAccessors {
    fn axes_tensor(&self) -> Option<GlobalId>;
    fn noop_with_empty_axes(&self) -> bool;
    #[allow(dead_code)]
    fn keepdims(&self) -> bool;
}

impl ReduceAccessors for crate::milli_graph::ops::ReduceSum {
    fn axes_tensor(&self) -> Option<GlobalId> {
        self.axes_tensor()
    }
    fn noop_with_empty_axes(&self) -> bool {
        self.noop_with_empty_axes()
    }
    fn keepdims(&self) -> bool {
        self.keepdims()
    }
}

impl ReduceAccessors for crate::milli_graph::ops::ReduceMax {
    fn axes_tensor(&self) -> Option<GlobalId> {
        self.axes_tensor()
    }
    fn noop_with_empty_axes(&self) -> bool {
        self.noop_with_empty_axes()
    }
    fn keepdims(&self) -> bool {
        self.keepdims()
    }
}

impl ReduceAccessors for crate::milli_graph::ops::ReduceMin {
    fn axes_tensor(&self) -> Option<GlobalId> {
        self.axes_tensor()
    }
    fn noop_with_empty_axes(&self) -> bool {
        self.noop_with_empty_axes()
    }
    fn keepdims(&self) -> bool {
        self.keepdims()
    }
}

impl ReduceAccessors for crate::milli_graph::ops::ReduceProd {
    fn axes_tensor(&self) -> Option<GlobalId> {
        self.axes_tensor()
    }
    fn noop_with_empty_axes(&self) -> bool {
        self.noop_with_empty_axes()
    }
    fn keepdims(&self) -> bool {
        self.keepdims()
    }
}

impl ReduceAccessors for crate::milli_graph::ops::ReduceMean {
    fn axes_tensor(&self) -> Option<GlobalId> {
        self.axes_tensor()
    }
    fn noop_with_empty_axes(&self) -> bool {
        self.noop_with_empty_axes()
    }
    fn keepdims(&self) -> bool {
        self.keepdims()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LowerError {
    #[error("Missing tensor shape for {0}")]
    MissingShape(GlobalId),
    #[error("Missing tensor dtype for {0}")]
    MissingDtype(GlobalId),
    #[error("Missing tensor value for {0} (needed for {1})")]
    MissingValue(GlobalId, &'static str),
    #[error("Missing tensor ref for {0}")]
    MissingRef(GlobalId),
    #[error("Unsupported op: {0}")]
    UnsupportedOp(String),
    #[error("MilliOpGraph error: {0}")]
    MilliGraph(#[from] crate::milli_graph::MilliOpGraphError),
}

/// Result of lowering a MilliOpGraph.
pub struct LowerResult<'a, 'p, P: crate::pool::Pool + 'p = crate::pool::SystemPool> {
    pub graph: NanoGraph<'p, P>,
    /// Ops that could not be lowered (treated as boundary).
    pub unsupported: Vec<(GlobalId, String)>,
    /// Human-readable detail for each unsupported op (input/output shapes).
    pub unsupported_details: Vec<String>,
    /// Mapping from milli tensor GlobalId to nano atom group.
    pub tensor_map: HashMap<GlobalId, TensorAtomMapInfo>,
    /// Provenance: for each nano group index, the (milli_op_id, op_kind) that produced it.
    /// Length equals graph.num_groups(). Used by reporting/visualization.
    pub group_provenance: Vec<(GlobalId, String)>,
    /// Full tensor info map from inference (dtype + shape + concrete values).
    /// Populated by `lower()` so callers don't need to run `infer_all` separately.
    pub all_infos: HashMap<GlobalId, LowerTensorInfo<'a, 'p, P>>,
}

/// Public view of how a milli tensor maps to nano atoms.
#[derive(Debug, Clone)]
pub struct TensorAtomMapInfo {
    pub base_id: AtomId,
    pub count: u64,
    pub dtype: NumericDType,
    pub sym_dims: Vec<SymDim>,
    pub known_strides: Vec<u64>,
    pub known_dims: Vec<u64>,
    /// Concat segments (empty for simple views).
    pub segments: Vec<(usize, u64, u64, AtomId, Vec<u64>)>, // (concat_dim, start, size, base_id, strides)
}

impl TensorAtomMapInfo {
    /// Returns true if this tensor's atoms are contiguous with row-major strides.
    pub fn is_contiguous(&self) -> bool {
        self.segments.is_empty()
            && self.known_strides == TensorAtomMap::compute_strides(&self.known_dims)
    }

    /// Return the atom ranges that compose this tensor.
    ///
    /// For simple contiguous tensors: a single range. For non-contiguous or
    /// segmented tensors, needs the NanoGraph to discover distinct groups.
    pub fn atom_ranges<'p, P: crate::pool::Pool + 'p>(
        &self,
        graph: &super::pattern::NanoGraph<'p, P>,
    ) -> Vec<super::pattern::AtomRange> {
        use super::pattern::AtomRange;
        let is_contiguous = self.segments.is_empty()
            && self.known_strides == TensorAtomMap::compute_strides(&self.known_dims);
        if is_contiguous {
            return vec![AtomRange {
                base: self.base_id,
                count: self.count,
                dtype: self.dtype,
            }];
        }
        // Non-contiguous or segmented: enumerate to find distinct ranges.
        let mut ranges = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for i in 0..self.count {
            let atom = self.atom_id_for_element(i);
            if let Some(gi) = graph.find_group_idx(atom) {
                if seen.insert(gi) {
                    let g = &graph.groups()[gi];
                    ranges.push(AtomRange {
                        base: g.base_id,
                        count: g.count,
                        dtype: g.output_dtype,
                    });
                }
            } else if let Some((ti, _)) = graph.find_input_idx(atom)
                && seen.insert(usize::MAX - ti)
            {
                let it = &graph.input_tensors()[ti];
                ranges.push(AtomRange {
                    base: it.base_id,
                    count: it.count,
                    dtype: it.dtype,
                });
            }
        }
        ranges
    }

    /// Map logical element index to AtomId using strides.
    pub fn atom_id_for_element(&self, flat: u64) -> AtomId {
        let row_major = TensorAtomMap::compute_strides(&self.known_dims);
        let mut indices = vec![0u64; self.known_dims.len()];
        let mut rem = flat;
        for (i, &rm_stride) in row_major.iter().enumerate() {
            if rm_stride > 0 {
                indices[i] = rem / rm_stride;
                rem %= rm_stride;
            }
        }
        if !self.segments.is_empty() {
            let (concat_dim, _, _, _, _) = &self.segments[0];
            let concat_idx = indices[*concat_dim];
            for &(_, start, size, seg_base, ref seg_strides) in &self.segments {
                if concat_idx >= start && concat_idx < start + size {
                    let mut seg_indices = indices.clone();
                    seg_indices[*concat_dim] = concat_idx - start;
                    let mut offset = 0u64;
                    for (i, &stride) in seg_strides.iter().enumerate() {
                        offset += seg_indices[i] * stride;
                    }
                    return seg_base.offset(offset);
                }
            }
        }
        let mut offset = 0u64;
        for (i, &stride) in self.known_strides.iter().enumerate() {
            offset += indices[i] * stride;
        }
        self.base_id.offset(offset)
    }
}

/// How a milli tensor maps to atoms in the nano graph.
///
/// The tensor's dimensions are split into known (expanded to atoms) and
/// symbolic (iteration parameters). Atoms are indexed by their position
/// in the flattened known dims (row-major order).
///
/// For simple tensors (most ops), `base_id` + `known_strides` is sufficient.
/// For concatenated tensors, `segments` describes how the concat axis is
/// split across multiple source atom ranges, each with its own base_id and strides.
#[derive(Debug, Clone)]
pub struct TensorAtomMap {
    /// First atom id in the group (for simple views).
    pub base_id: AtomId,
    /// Total number of atoms (product of known dims).
    pub count: u64,
    /// Data type of this tensor.
    pub dtype: NumericDType,
    /// The full tensor layout: one entry per dim, preserving original order.
    pub layout: Vec<DimKind>,
    /// Physical strides for the known dims into the atom buffer.
    /// May be non-row-major for transposed or strided views.
    pub known_strides: Vec<u64>,
    /// Symbolic dims attached to each atom.
    pub sym_dims: Vec<SymDim>,
    /// For concatenated tensors: segments along a specific known dim.
    /// If empty, this is a simple single-range view.
    pub segments: Vec<ConcatSegment>,
}

/// A segment of a concatenated tensor along one axis.
#[derive(Debug, Clone)]
pub struct ConcatSegment {
    /// Which known dim index the concat is along.
    pub concat_dim: usize,
    /// Starting index along the concat dim for this segment.
    pub start: u64,
    /// Number of elements along the concat dim in this segment.
    pub size: u64,
    /// Base atom ID for this segment's source tensor.
    pub base_id: AtomId,
    /// Physical strides for this segment (into its source atom space).
    pub known_strides: Vec<u64>,
}

/// Classification of one tensor dimension.
#[derive(Debug, Clone)]
pub enum DimKind {
    Known(u64),
    Symbolic(#[allow(dead_code)] SymDim),
}

/// (layout, known_dims, sym_dims, atom_count)
pub type DimClassification = (Vec<DimKind>, Vec<u64>, Vec<SymDim>, u64);

impl TensorAtomMap {
    /// Create a simple (non-segmented) tensor atom map.
    pub fn simple(
        base_id: AtomId,
        count: u64,
        dtype: NumericDType,
        layout: Vec<DimKind>,
        known_strides: Vec<u64>,
        sym_dims: Vec<SymDim>,
    ) -> Self {
        Self {
            base_id,
            count,
            dtype,
            layout,
            known_strides,
            sym_dims,
            segments: vec![],
        }
    }

    /// Create a segmented tensor atom map (for Concat).
    pub fn segmented(
        count: u64,
        dtype: NumericDType,
        layout: Vec<DimKind>,
        sym_dims: Vec<SymDim>,
        segments: Vec<ConcatSegment>,
    ) -> Self {
        let base_id = if segments.is_empty() {
            AtomId(0)
        } else {
            segments[0].base_id
        };
        let known_strides = if segments.is_empty() {
            vec![]
        } else {
            segments[0].known_strides.clone()
        };
        Self {
            base_id,
            count,
            dtype,
            layout,
            known_strides,
            sym_dims,
            segments,
        }
    }

    /// Compute row-major strides from known dim sizes.
    pub fn compute_strides(known_dims: &[u64]) -> Vec<u64> {
        let mut strides = vec![0u64; known_dims.len()];
        if known_dims.is_empty() {
            return strides;
        }
        let mut stride = 1u64;
        for i in (0..known_dims.len()).rev() {
            strides[i] = stride;
            stride *= known_dims[i];
        }
        strides
    }

    /// Returns true if this tensor's atoms are a contiguous range base..base+count
    /// with row-major strides (i.e. atom_id_for_element(i) == base + i for all i).
    pub fn is_contiguous(&self) -> bool {
        if !self.segments.is_empty() {
            return false;
        }
        let known_dims = self.known_dims();
        self.known_strides == Self::compute_strides(&known_dims)
    }

    /// Get the known dim sizes from the layout.
    pub fn known_dims(&self) -> Vec<u64> {
        self.layout
            .iter()
            .filter_map(|d| {
                if let DimKind::Known(s) = d {
                    Some(*s)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Return the atom ranges that compose this tensor.
    ///
    /// For simple contiguous tensors: a single range `[base, base+count)`.
    /// For non-contiguous views (transposed strides): falls back to per-element
    /// enumeration to discover the distinct groups.
    /// For segmented tensors (concat): one range per segment.
    pub fn atom_ranges<'p, P: crate::pool::Pool + 'p>(
        &self,
        graph: &super::pattern::NanoGraph<'p, P>,
    ) -> Vec<super::pattern::AtomRange> {
        use super::pattern::AtomRange;
        if self.segments.is_empty() && self.is_contiguous() {
            // Single contiguous range.
            return vec![AtomRange {
                base: self.base_id,
                count: self.count,
                dtype: self.dtype,
            }];
        }
        // Non-contiguous or segmented: enumerate to find distinct ranges.
        let mut ranges = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for i in 0..self.count {
            let atom = self.atom_id_for_element(i);
            if let Some(gi) = graph.find_group_idx(atom) {
                if seen.insert(gi) {
                    let g = &graph.groups()[gi];
                    ranges.push(AtomRange {
                        base: g.base_id,
                        count: g.count,
                        dtype: g.output_dtype,
                    });
                }
            } else if let Some((ti, _)) = graph.find_input_idx(atom)
                && seen.insert(usize::MAX - ti)
            {
                let it = &graph.input_tensors()[ti];
                ranges.push(AtomRange {
                    base: it.base_id,
                    count: it.count,
                    dtype: it.dtype,
                });
            }
        }
        ranges
    }

    /// Map logical element index to AtomId using strides.
    /// Handles simple views, transposed/strided views, and segmented (concat) views.
    pub fn atom_id_for_element(&self, flat: u64) -> AtomId {
        let known_dims = self.known_dims();
        let row_major = Self::compute_strides(&known_dims);

        // Decompose flat index into per-dim indices using row-major strides.
        let mut indices = vec![0u64; known_dims.len()];
        let mut rem = flat;
        for (i, &rm_stride) in row_major.iter().enumerate() {
            if rm_stride > 0 {
                indices[i] = rem / rm_stride;
                rem %= rm_stride;
            }
        }

        // If segmented (concat), find the right segment for the concat dim index.
        if !self.segments.is_empty() {
            let concat_dim = self.segments[0].concat_dim;
            let concat_idx = indices[concat_dim];
            for seg in &self.segments {
                if concat_idx >= seg.start && concat_idx < seg.start + seg.size {
                    // Remap the concat dim index to be relative to this segment.
                    let mut seg_indices = indices.clone();
                    seg_indices[concat_dim] = concat_idx - seg.start;
                    let mut offset = 0u64;
                    for (i, &stride) in seg.known_strides.iter().enumerate() {
                        offset += seg_indices[i] * stride;
                    }
                    return seg.base_id.offset(offset);
                }
            }
            // Shouldn't happen if segments cover the full concat dim.
            panic!(
                "Concat segment not found for index {} on dim {}",
                concat_idx, concat_dim
            );
        }

        // Simple view: use base_id + physical strides.
        let mut offset = 0u64;
        for (i, &stride) in self.known_strides.iter().enumerate() {
            offset += indices[i] * stride;
        }
        self.base_id.offset(offset)
    }
}

/// Lower a MilliOpGraph into a NanoGraph using partial tensor information.
///
/// Walks ops in topological order, classifying dimensions, and building atom
/// groups. The returned `LowerResult` contains the NanoGraph, a tensor map,
/// and any ops that couldn't be lowered (boundary ops).
pub fn lower<'a, 'p: 'a, P: crate::pool::Pool + 'p>(
    graph: &MilliOpGraph,
    inputs: &HashMap<GlobalId, LowerTensorInfo<'a, 'p, P>>,
    pool: &'p P,
) -> Result<LowerResult<'a, 'p, P>, LowerError> {
    let all_infos = graph.infer_all(inputs, pool)?;
    let mut ctx = NanoLoweringContext::new(&all_infos, pool);

    // Register all tensors that exist before ops run (graph inputs + inferred
    // constants) and are NOT produced by any op.
    // Use register_constant so that tensors with concrete data get LiteralSpan
    // groups (baking values into the graph), while tensors without data fall
    // through to register_input (creating InputTensor slots for runtime).
    let op_outputs: std::collections::HashSet<GlobalId> = graph
        .op_ordering()
        .iter()
        .filter_map(|op_id| graph.get_node_by_id(op_id))
        .flat_map(|op| op.outputs().collect::<Vec<_>>())
        .collect();
    // Copy the shared reference to avoid borrowing ctx during iteration.
    let infos_ref = ctx.all_infos;
    for (id, info) in infos_ref {
        if !op_outputs.contains(id) {
            ctx.register_constant(*id, info);
        }
    }

    // Walk ops in topological order.
    for &op_id in graph.op_ordering() {
        let Some(op) = graph.get_node_by_id(&op_id) else {
            continue;
        };
        ctx.lower_op(op);
    }

    // Validate the graph structure (catch degenerate/self-referencing groups).
    let validation_errors = ctx.nano.validate();
    if !validation_errors.is_empty() {
        eprintln!(
            "  [lower] NanoGraph validation: {} errors",
            validation_errors.len()
        );
        for (i, err) in validation_errors.iter().enumerate() {
            if i < 5 {
                eprintln!("    {}", err);
            }
        }
    }

    // Collect outputs. Try get_outputs() first; if empty, scan all tensors
    // in the tensor_map that aren't consumed by any op (terminal tensors).
    let output_ids = graph.get_outputs();
    if !output_ids.is_empty() {
        for out_id in &output_ids {
            if let Some(tam) = ctx.tensor_map.get(out_id) {
                ctx.nano.outputs.extend(tam.atom_ranges(&ctx.nano));
            }
        }
    }

    let tensor_map: HashMap<GlobalId, TensorAtomMapInfo> = ctx
        .tensor_map
        .iter()
        .map(|(id, tam)| {
            (
                *id,
                TensorAtomMapInfo {
                    base_id: tam.base_id,
                    count: tam.count,
                    dtype: tam.dtype,
                    sym_dims: tam.sym_dims.clone(),
                    known_strides: tam.known_strides.clone(),
                    known_dims: tam.known_dims(),
                    segments: tam
                        .segments
                        .iter()
                        .map(|s| {
                            (
                                s.concat_dim,
                                s.start,
                                s.size,
                                s.base_id,
                                s.known_strides.clone(),
                            )
                        })
                        .collect(),
                },
            )
        })
        .collect();

    Ok(LowerResult {
        graph: ctx.nano,
        unsupported: ctx.unsupported,
        unsupported_details: ctx.unsupported_details,
        tensor_map,
        group_provenance: ctx.group_provenance,
        all_infos,
    })
}

/// Backward-compatible alias for `lower()`. Will be removed once all call
/// sites are migrated.
pub fn lower_with_info<'a, 'p: 'a, P: crate::pool::Pool + 'p>(
    graph: &MilliOpGraph,
    inputs: &HashMap<GlobalId, LowerTensorInfo<'a, 'p, P>>,
    pool: &'p P,
) -> Result<LowerResult<'a, 'p, P>, LowerError> {
    lower(graph, inputs, pool)
}

/// Context for lowering milli ops to nano graph.
///
/// Holds all mutable state needed during lowering. The `all_infos` field
/// is a shared reference to tensor shape information computed by inference,
/// so methods can freely read tensor info without borrowing self.
pub struct NanoLoweringContext<'a, 'p, P: crate::pool::Pool + 'p = SystemPool> {
    pub nano: NanoGraph<'p, P>,
    pub tensor_map: HashMap<GlobalId, TensorAtomMap>,
    pub all_infos: &'a HashMap<GlobalId, LowerTensorInfo<'a, 'p, P>>,
    pub pool: &'p P,
    next_anon_sym: usize,
    pub unsupported: Vec<(GlobalId, String)>,
    pub unsupported_details: Vec<String>,
    /// Provenance: maps each nano group index to the milli op that produced it.
    /// Recorded as (milli_op_id, op_kind_string).
    pub group_provenance: Vec<(GlobalId, String)>,
}

impl<'a, 'p, P: crate::pool::Pool + 'p> NanoLoweringContext<'a, 'p, P> {
    pub fn new(all_infos: &'a HashMap<GlobalId, LowerTensorInfo<'a, 'p, P>>, pool: &'p P) -> Self {
        Self {
            nano: NanoGraph::new(),
            tensor_map: HashMap::new(),
            all_infos,
            pool,
            next_anon_sym: 0,
            unsupported: Vec::new(),
            unsupported_details: Vec::new(),
            group_provenance: Vec::new(),
        }
    }

    /// Get the NumericDType from a LowerTensorInfo.
    pub fn ndt(info: &LowerTensorInfo<'_, 'p, P>) -> NumericDType {
        info.dtype()
    }

    /// Classify tensor dims and return layout info.
    /// Returns None if rank is unknown or atom count overflows u32.
    pub fn classify_dims(
        &mut self,
        info: &LowerTensorInfo<'_, 'p, P>,
    ) -> Option<DimClassification> {
        let rank = info.rank_if_known()?;
        let mut layout = Vec::with_capacity(rank);
        let mut known_dims = Vec::new();
        let mut sym_dims = Vec::new();

        for i in 0..rank {
            if let Some(size) = info.dim_if_known(i) {
                layout.push(DimKind::Known(size));
                known_dims.push(size);
            } else {
                let sd = self.alloc_sym_dim();
                layout.push(DimKind::Symbolic(sd));
                sym_dims.push(sd);
            }
        }

        let count: u64 = known_dims.iter().product();

        Some((layout, known_dims, sym_dims, count))
    }

    pub fn alloc_sym_dim(&mut self) -> SymDim {
        let name = format!("sym_{}", self.next_anon_sym);
        self.next_anon_sym += 1;
        self.nano.sym_dim(&name)
    }

    /// Register a known-value constant as Literal groups.
    ///
    /// Extracts scalar values from the LowerTensorInfo and creates Literal groups.
    /// Runs of identical values are coalesced into single groups. If the
    /// LowerTensorInfo has no numeric data, falls back to `register_input`.
    pub fn register_constant(&mut self, id: GlobalId, info: &LowerTensorInfo<'_, 'p, P>) {
        let Some(concrete) = info.as_concrete() else {
            self.register_input(id, info);
            return;
        };

        let Some((layout, known_dims, sym_dims, count)) = self.classify_dims(info) else {
            self.register_input(id, info);
            return;
        };

        let strides = TensorAtomMap::compute_strides(&known_dims);
        let count = count.max(1) as usize;
        let dt = Self::ndt(info);

        let n_elems = concrete.numel();
        if n_elems == 0 {
            self.register_input(id, info);
            return;
        }

        let n = n_elems.min(count);

        // For single-element constants, use a broadcast Literal (no allocation).
        // For multi-element constants, copy into a pool tensor and emit one
        // LiteralSpan group — O(1) groups instead of O(n).
        if n == 1 {
            let scalar = concrete.read_element(0);
            let base_id =
                self.nano
                    .push_group(1, dt, ScalarOp::Literal(scalar), sym_dims.clone(), vec![]);
            self.tensor_map.insert(
                id,
                TensorAtomMap::simple(base_id, 1, dt, layout, strides, sym_dims),
            );
        } else {
            // Check if all elements are identical — use broadcast Literal.
            let first = concrete.read_element(0);
            let all_same = (1..n).all(|i| concrete.read_element(i) == first);
            if all_same {
                let base_id = self.nano.push_group(
                    n as u64,
                    dt,
                    ScalarOp::Literal(first),
                    sym_dims.clone(),
                    vec![],
                );
                self.tensor_map.insert(
                    id,
                    TensorAtomMap::simple(base_id, n as u64, dt, layout, strides, sym_dims),
                );
            } else {
                // Copy into a 1D pool tensor backing the LiteralSpan.
                use crate::numeric_tensor::TensorLayout;
                let span_layout =
                    TensorLayout::<crate::tensor_rank::DynRank>::row_major(vec![n as u64], dt);
                let cv = concrete.view();
                let tensor = if cv.layout().is_contiguous() && cv.numel() == n {
                    // Fast path: memcpy raw bytes into 1D layout.
                    let size = span_layout.buffer_size_bytes();
                    let mut buf = self
                        .pool
                        .allocate(size)
                        .expect("pool allocation for LiteralSpan");
                    buf[..size].copy_from_slice(&cv.data()[..size]);
                    crate::numeric_tensor::NumericTensor::from_parts(buf, span_layout)
                } else {
                    // Slow path: element-wise copy for non-contiguous views.
                    let buf = self
                        .pool
                        .allocate(span_layout.buffer_size_bytes())
                        .expect("pool allocation for LiteralSpan");
                    let mut tensor =
                        crate::numeric_tensor::NumericTensor::from_parts(buf, span_layout);
                    for i in 0..n {
                        tensor.write_element(i, concrete.read_element(i));
                    }
                    tensor
                };
                let base_id = self.nano.push_group(
                    n as u64,
                    dt,
                    ScalarOp::LiteralSpan(tensor),
                    sym_dims.clone(),
                    vec![],
                );
                self.tensor_map.insert(
                    id,
                    TensorAtomMap::simple(base_id, n as u64, dt, layout, strides, sym_dims),
                );
            }
        }
    }

    /// Register a graph input as leaf atoms.
    ///
    /// Creates Literal groups with placeholder value 0.0. For external inputs
    /// (user-provided tensors), the executor fills in actual values at runtime.
    /// For constant tensors whose values are known at lowering time, the
    /// Literal(0.0) placeholder is similarly overridden by the executor using
    /// the tensor data from `all_infos`.
    pub fn register_input(&mut self, id: GlobalId, info: &LowerTensorInfo<'_, 'p, P>) {
        let Some((layout, known_dims, sym_dims, count)) = self.classify_dims(info) else {
            // Unknown rank — register a single atom.
            let dt = Self::ndt(info);
            let base_id = self.nano.add_input_tensor(id, 1, dt);
            self.tensor_map.insert(
                id,
                TensorAtomMap::simple(base_id, 1, dt, vec![], vec![], vec![]),
            );
            return;
        };

        let strides = TensorAtomMap::compute_strides(&known_dims);
        let count = count.max(1);
        let dt = Self::ndt(info);

        let base_id = self.nano.add_input_tensor(id, count, dt);

        self.tensor_map.insert(
            id,
            TensorAtomMap::simple(base_id, count, dt, layout, strides, sym_dims),
        );
    }

    /// Register a tensor as a boundary (opaque) group.
    /// Boundary atoms are leaves — they use Literal(0) with no inputs.
    pub fn register_boundary(
        &mut self,
        output_id: GlobalId,
        info: &LowerTensorInfo<'_, 'p, P>,
        _op_kind: &str,
    ) {
        let dt = Self::ndt(info);
        let Some((layout, known_dims, sym_dims, count)) = self.classify_dims(info) else {
            let base_id = self.nano.push_atom(
                dt,
                ScalarOp::Literal(NumericScalar::zero(dt)),
                vec![],
                vec![],
            );
            self.tensor_map.insert(
                output_id,
                TensorAtomMap::simple(base_id, 1, dt, vec![], vec![], vec![]),
            );
            return;
        };

        let strides = TensorAtomMap::compute_strides(&known_dims);
        let count = count.max(1);

        let base_id = self.nano.push_group(
            count,
            dt,
            ScalarOp::Literal(NumericScalar::zero(dt)),
            sym_dims.clone(),
            vec![],
        );

        self.tensor_map.insert(
            output_id,
            TensorAtomMap::simple(base_id, count, dt, layout, strides, sym_dims),
        );
    }

    /// Compute an InputRef mapping consumer atoms → producer atoms.
    pub fn compute_input_ref(
        &self,
        consumer: &TensorAtomMap,
        producer: &TensorAtomMap,
        consumer_info: &LowerTensorInfo<'a, 'p, P>,
        producer_info: &LowerTensorInfo<'a, 'p, P>,
    ) -> InputRef {
        // Segmented producers (concat): always build via atom_id_for_element
        // since they can't be expressed as a single Affine/Broadcast pattern.
        if !producer.segments.is_empty() {
            return self.build_segmented_input_ref(
                consumer,
                producer,
                consumer_info,
                producer_info,
            );
        }

        // Same count + same known shape → stride 1.
        if consumer.count == producer.count && consumer.count > 0 {
            let c_known: Vec<u64> = consumer
                .layout
                .iter()
                .filter_map(|d| {
                    if let DimKind::Known(s) = d {
                        Some(*s)
                    } else {
                        None
                    }
                })
                .collect();
            let p_known: Vec<u64> = producer
                .layout
                .iter()
                .filter_map(|d| {
                    if let DimKind::Known(s) = d {
                        Some(*s)
                    } else {
                        None
                    }
                })
                .collect();
            if c_known == p_known && consumer.known_strides == producer.known_strides {
                return InputRef::affine(producer.base_id, 1);
            }
        }

        // Scalar producer → broadcast.
        if producer.count == 1 {
            return InputRef::Broadcast(producer.base_id);
        }

        // Try the analytical N-D Strided builder. Handles broadcast +
        // non-row-major producer strides directly without enumerating atoms.
        if let Some(ir) =
            self.try_build_broadcast_strided(consumer, producer, consumer_info, producer_info)
        {
            return ir;
        }

        // Fallback: build the per-atom Explicit list and run heuristic compress.
        self.build_broadcast_explicit(consumer, producer, consumer_info, producer_info)
    }

    /// Analytically derive an N-D `InputRef::Strided` for a broadcast access.
    ///
    /// Mirrors the per-atom math in `build_broadcast_explicit` but expresses it
    /// directly as `dim_shape` / `dim_strides` against the producer's physical
    /// strides. Returns `None` if any precondition is violated; the caller
    /// falls back to the explicit walk in that case.
    ///
    /// Preconditions:
    ///  - Producer is non-segmented (concat goes through `atom_id_for_element`).
    ///  - Producer's `known_strides` length matches its `known_dims()`.
    ///  - Consumer's `known_strides` length matches its `known_dims()`.
    ///  - Consumer is row-major (because `InputRef::Strided::resolve` decomposes
    ///    `i` row-major over `dim_shape`). The existing explicit walk has the
    ///    same latent assumption — it iterates `c_strides` in declaration order,
    ///    which only correctly inverts row-major layouts. In practice consumers
    ///    of `compute_input_ref` are always freshly-allocated row-major op
    ///    outputs, so this is not a real restriction.
    fn try_build_broadcast_strided(
        &self,
        consumer: &TensorAtomMap,
        producer: &TensorAtomMap,
        consumer_info: &LowerTensorInfo<'a, 'p, P>,
        producer_info: &LowerTensorInfo<'a, 'p, P>,
    ) -> Option<InputRef> {
        if !producer.segments.is_empty() {
            return None;
        }

        let c_known_sizes = consumer.known_dims();
        let p_known_sizes = producer.known_dims();
        let c_strides = &consumer.known_strides;
        let p_strides = &producer.known_strides;

        if c_strides.len() != c_known_sizes.len() {
            return None;
        }
        if p_strides.len() != p_known_sizes.len() {
            return None;
        }

        // Bail out for non-row-major consumers — see the doc comment for why.
        let c_row_major = TensorAtomMap::compute_strides(&c_known_sizes);
        if c_strides != &c_row_major {
            return None;
        }

        let c_rank = consumer_info.rank_if_known().unwrap_or(0);
        let p_rank = producer_info.rank_if_known().unwrap_or(0);

        // Map original-dim positions to indices in the known-dims arrays.
        let c_known_indices: Vec<Option<usize>> = {
            let mut ki = 0;
            consumer
                .layout
                .iter()
                .map(|d| {
                    if matches!(d, DimKind::Known(_)) {
                        let idx = ki;
                        ki += 1;
                        Some(idx)
                    } else {
                        None
                    }
                })
                .collect()
        };
        let p_known_indices: Vec<Option<usize>> = {
            let mut ki = 0;
            producer
                .layout
                .iter()
                .map(|d| {
                    if matches!(d, DimKind::Known(_)) {
                        let idx = ki;
                        ki += 1;
                        Some(idx)
                    } else {
                        None
                    }
                })
                .collect()
        };

        // c_to_p_known[c_ki] = mapped producer known-dim index, if any.
        // Right-aligned broadcasting per ONNX semantics.
        let dim_offset = c_rank.saturating_sub(p_rank);
        let mut c_to_p_known: Vec<Option<usize>> = vec![None; c_known_sizes.len()];
        for (c_orig, c_known_idx) in c_known_indices.iter().enumerate() {
            let Some(c_ki) = *c_known_idx else {
                continue;
            };
            if c_orig < dim_offset {
                continue;
            }
            let p_orig = c_orig - dim_offset;
            if p_orig >= p_rank {
                continue;
            }
            if let Some(p_ki) = p_known_indices[p_orig] {
                c_to_p_known[c_ki] = Some(p_ki);
            }
        }

        // Build dim_shape / dim_strides directly from the consumer's known dims
        // and the (possibly non-row-major) producer physical strides.
        let mut dim_shape: Vec<u64> = Vec::with_capacity(c_known_sizes.len());
        let mut dim_strides: Vec<i64> = Vec::with_capacity(c_known_sizes.len());
        for (c_ki, &p_ki_opt) in c_to_p_known.iter().enumerate() {
            dim_shape.push(c_known_sizes[c_ki]);
            let stride = match p_ki_opt {
                // Producer dim of size 1 → broadcast → stride 0 (coord forced to 0).
                Some(p_ki) if p_known_sizes[p_ki] == 1 => 0i64,
                // Mapped to a real producer dim → use its physical stride verbatim.
                // This is what naturally handles transposed/sliced producers.
                Some(p_ki) => p_strides[p_ki] as i64,
                // Unmapped consumer dim (consumer outranks producer) → broadcast.
                None => 0i64,
            };
            dim_strides.push(stride);
        }

        // Degenerate consumer with no known dims: nothing meaningful to express.
        if dim_shape.is_empty() {
            return None;
        }

        // Squeeze size-1 dims: their coord is always 0, so they contribute
        // nothing to the offset *and* nothing to the row-major iteration order.
        // Removing them reduces nd, helps the codegen take its 1D/2D fast paths,
        // and exposes simple Affine/Modular/StridedBroadcast patterns to the
        // partitioner's pattern matchers.
        let mut sq_shape: Vec<u64> = Vec::with_capacity(dim_shape.len());
        let mut sq_strides: Vec<i64> = Vec::with_capacity(dim_shape.len());
        for d in 0..dim_shape.len() {
            if dim_shape[d] == 1 {
                continue;
            }
            sq_shape.push(dim_shape[d]);
            sq_strides.push(dim_strides[d]);
        }
        // Everything broadcast / scalar producer → fall through to Broadcast.
        if sq_shape.is_empty() {
            return Some(InputRef::Broadcast(producer.base_id));
        }

        // 1D affine fast path: a single non-trivial dim becomes Affine directly.
        if sq_shape.len() == 1 {
            return Some(InputRef::affine(producer.base_id, sq_strides[0]));
        }

        // Recognize a row-major access (`stride[d] = product(shape[d+1..])` and
        // innermost stride 1) as flat Affine stride 1 over the full count. The
        // partitioner's reducers care a lot about this: a "row-major view of
        // self" is just a contiguous read.
        let mut is_rowmajor_unit = sq_strides[sq_strides.len() - 1] == 1;
        if is_rowmajor_unit {
            let mut expected: i64 = 1;
            for d in (0..sq_shape.len()).rev() {
                if sq_strides[d] != expected {
                    is_rowmajor_unit = false;
                    break;
                }
                expected = expected.saturating_mul(sq_shape[d] as i64);
            }
        }
        if is_rowmajor_unit {
            return Some(InputRef::affine(producer.base_id, 1));
        }

        Some(InputRef::Strided {
            base: producer.base_id,
            dim_strides: sq_strides,
            dim_shape: sq_shape,
        })
    }

    /// Build an Explicit InputRef for broadcast patterns.
    fn build_broadcast_explicit(
        &self,
        consumer: &TensorAtomMap,
        producer: &TensorAtomMap,
        consumer_info: &LowerTensorInfo<'a, 'p, P>,
        producer_info: &LowerTensorInfo<'a, 'p, P>,
    ) -> InputRef {
        let c_rank = consumer_info.rank_if_known().unwrap_or(0);
        let p_rank = producer_info.rank_if_known().unwrap_or(0);

        let c_known_sizes: Vec<u64> = consumer
            .layout
            .iter()
            .filter_map(|d| {
                if let DimKind::Known(s) = d {
                    Some(*s)
                } else {
                    None
                }
            })
            .collect();
        let p_known_sizes: Vec<u64> = producer
            .layout
            .iter()
            .filter_map(|d| {
                if let DimKind::Known(s) = d {
                    Some(*s)
                } else {
                    None
                }
            })
            .collect();
        // Use the producer's actual strides (may be non-row-major after Transpose).
        let p_strides = &producer.known_strides;

        // Build mapping from consumer known-dim index to producer known-dim
        // index using right-aligned broadcasting.
        let offset = c_rank.saturating_sub(p_rank);

        // For each original dim, track its known-dim index (if known).
        let c_known_indices: Vec<Option<usize>> = {
            let mut ki = 0;
            consumer
                .layout
                .iter()
                .map(|d| {
                    if matches!(d, DimKind::Known(_)) {
                        let idx = ki;
                        ki += 1;
                        Some(idx)
                    } else {
                        None
                    }
                })
                .collect()
        };
        let p_known_indices: Vec<Option<usize>> = {
            let mut ki = 0;
            producer
                .layout
                .iter()
                .map(|d| {
                    if matches!(d, DimKind::Known(_)) {
                        let idx = ki;
                        ki += 1;
                        Some(idx)
                    } else {
                        None
                    }
                })
                .collect()
        };

        // Map: consumer_known_dim_idx → producer_known_dim_idx (or None if broadcast/mismatch).
        let mut c_to_p_known: Vec<Option<usize>> = vec![None; c_known_sizes.len()];
        for (c_orig, c_known_idx) in c_known_indices.iter().enumerate() {
            let Some(c_ki) = *c_known_idx else {
                continue;
            };
            if c_orig < offset {
                continue;
            }
            let p_orig = c_orig - offset;
            if p_orig >= p_rank {
                continue;
            }
            if let Some(p_ki) = p_known_indices[p_orig] {
                c_to_p_known[c_ki] = Some(p_ki);
            }
        }

        let mut ids = Vec::with_capacity(consumer.count as usize);
        let c_strides = &consumer.known_strides;
        let p_row_major = TensorAtomMap::compute_strides(&p_known_sizes);

        for flat_c in 0..consumer.count {
            // Decompose flat_c into known-dim indices.
            let mut c_indices = vec![0u64; c_known_sizes.len()];
            let mut rem = flat_c;
            for (i, &stride) in c_strides.iter().enumerate() {
                if stride > 0 {
                    c_indices[i] = rem / stride;
                    rem %= stride;
                }
            }

            // Map to producer indices.
            let mut p_indices = vec![0u64; p_known_sizes.len()];
            for (c_ki, &p_ki_opt) in c_to_p_known.iter().enumerate() {
                if let Some(p_ki) = p_ki_opt {
                    if p_known_sizes[p_ki] == 1 {
                        p_indices[p_ki] = 0; // broadcast
                    } else {
                        p_indices[p_ki] = c_indices[c_ki];
                    }
                }
            }

            // Convert producer dim indices to an atom ID.
            // Use physical strides directly to compute offset into the
            // producer's atom space. This correctly handles:
            //  - Row-major producers (physical strides = row-major)
            //  - Transposed producers (physical strides = permuted row-major)
            //  - Non-contiguous slice producers (physical strides from
            //    larger source tensor; offset may exceed `count` but the
            //    resulting AtomId correctly references the source atoms)
            //
            // For segmented (concat) producers, fall back to
            // atom_id_for_element which handles the segment lookup.
            if !producer.segments.is_empty() {
                let mut flat_p = 0u64;
                for (i, &stride) in p_row_major.iter().enumerate() {
                    flat_p += p_indices[i] * stride;
                }
                ids.push(producer.atom_id_for_element(flat_p));
            } else {
                let mut offset = 0u64;
                for (i, &stride) in p_strides.iter().enumerate() {
                    offset += p_indices[i] * stride;
                }
                ids.push(producer.base_id.offset(offset));
            }
        }

        // Try to compress the Explicit table into a simpler InputRef pattern.
        Self::compress_explicit(ids)
    }

    /// Given an Explicit atom ID list, detect if it's actually a simpler pattern.
    pub fn compress_explicit(ids: Vec<AtomId>) -> InputRef {
        if ids.is_empty() {
            return InputRef::Explicit(ids);
        }
        if ids.len() == 1 {
            return InputRef::Broadcast(ids[0]);
        }

        // Check for Broadcast: all same.
        if ids.iter().all(|id| id.0 == ids[0].0) {
            return InputRef::Broadcast(ids[0]);
        }

        // Check for Affine: constant stride.
        let stride = ids[1].0 as i64 - ids[0].0 as i64;
        let is_affine = ids
            .windows(2)
            .all(|w| (w[1].0 as i64 - w[0].0 as i64) == stride);
        if is_affine {
            return InputRef::affine(ids[0], stride);
        }

        // Check for StridedBroadcast: blocks of identical values with regular stride.
        // Find repeat length (how many consecutive ids are the same).
        let mut repeat = 1u64;
        while (repeat as usize) < ids.len() && ids[repeat as usize].0 == ids[0].0 {
            repeat += 1;
        }
        if repeat > 1 && (ids.len() as u64).is_multiple_of(repeat) {
            let num_blocks = ids.len() as u64 / repeat;
            if num_blocks > 1 {
                let block_stride = ids[repeat as usize].0 as i64 - ids[0].0 as i64;
                let is_strided_broadcast = (0..num_blocks).all(|b| {
                    let expected_base = ids[0].0 as i64 + block_stride * b as i64;
                    (0..repeat).all(|r| ids[(b * repeat + r) as usize].0 as i64 == expected_base)
                });
                if is_strided_broadcast {
                    return InputRef::strided_broadcast(ids[0], block_stride, repeat);
                }
            }
        }

        // Check for Modular: ids[i] = ids[i % period] for some period.
        // Try small periods that divide the length.
        let len = ids.len();
        'modular: for period in [
            2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536,
            2048, 2304, 3072, 4096,
        ] {
            if period >= len || !len.is_multiple_of(period) {
                continue;
            }
            // Check if the pattern repeats with this period
            let matches = ids
                .iter()
                .enumerate()
                .all(|(i, id)| id.0 == ids[i % period].0);
            if !matches {
                continue;
            }
            // Found a repeating period — check if the period itself is Affine
            let inner_stride = if period >= 2 {
                ids[1].0 as i64 - ids[0].0 as i64
            } else {
                0
            };
            let inner_is_affine = period < 2
                || ids[..period]
                    .windows(2)
                    .all(|w| (w[1].0 as i64 - w[0].0 as i64) == inner_stride);
            if inner_is_affine {
                return InputRef::modular(ids[0], inner_stride, period as u64);
            }
            break 'modular;
        }

        InputRef::Explicit(ids)
    }

    /// Build InputRef for a segmented (concat) producer.
    /// Uses atom_id_for_element to resolve each consumer element's source,
    /// then compresses the resulting Explicit table.
    fn build_segmented_input_ref(
        &self,
        consumer: &TensorAtomMap,
        producer: &TensorAtomMap,
        consumer_info: &LowerTensorInfo<'a, 'p, P>,
        producer_info: &LowerTensorInfo<'a, 'p, P>,
    ) -> InputRef {
        let c_rank = consumer_info.rank_if_known().unwrap_or(0);
        let p_rank = producer_info.rank_if_known().unwrap_or(0);

        let c_known_sizes: Vec<u64> = consumer
            .layout
            .iter()
            .filter_map(|d| {
                if let DimKind::Known(s) = d {
                    Some(*s)
                } else {
                    None
                }
            })
            .collect();
        let p_known_sizes = producer.known_dims();

        let offset = c_rank.saturating_sub(p_rank);

        let c_known_indices: Vec<Option<usize>> = {
            let mut ki = 0;
            consumer
                .layout
                .iter()
                .map(|d| {
                    if matches!(d, DimKind::Known(_)) {
                        let idx = ki;
                        ki += 1;
                        Some(idx)
                    } else {
                        None
                    }
                })
                .collect()
        };
        let p_known_indices: Vec<Option<usize>> = {
            let mut ki = 0;
            producer
                .layout
                .iter()
                .map(|d| {
                    if matches!(d, DimKind::Known(_)) {
                        let idx = ki;
                        ki += 1;
                        Some(idx)
                    } else {
                        None
                    }
                })
                .collect()
        };

        let mut c_to_p_known: Vec<Option<usize>> = vec![None; c_known_sizes.len()];
        for (c_orig, c_known_idx) in c_known_indices.iter().enumerate() {
            let Some(c_ki) = *c_known_idx else { continue };
            if c_orig < offset {
                continue;
            }
            let p_orig = c_orig - offset;
            if p_orig >= p_rank {
                continue;
            }
            if let Some(p_ki) = p_known_indices[p_orig] {
                c_to_p_known[c_ki] = Some(p_ki);
            }
        }

        let c_strides = &consumer.known_strides;
        let mut ids = Vec::with_capacity(consumer.count as usize);

        for flat_c in 0..consumer.count {
            let mut c_indices = vec![0u64; c_known_sizes.len()];
            let mut rem = flat_c;
            for (i, &stride) in c_strides.iter().enumerate() {
                if stride > 0 {
                    c_indices[i] = rem / stride;
                    rem %= stride;
                }
            }

            let mut p_indices = vec![0u64; p_known_sizes.len()];
            for (c_ki, &p_ki_opt) in c_to_p_known.iter().enumerate() {
                if let Some(p_ki) = p_ki_opt {
                    if p_known_sizes[p_ki] == 1 {
                        p_indices[p_ki] = 0;
                    } else {
                        p_indices[p_ki] = c_indices[c_ki];
                    }
                }
            }

            // Compute flat index in producer's logical space
            let p_rowmajor = TensorAtomMap::compute_strides(&p_known_sizes);
            let mut flat_p = 0u64;
            for (i, &stride) in p_rowmajor.iter().enumerate() {
                flat_p += p_indices[i] * stride;
            }

            ids.push(producer.atom_id_for_element(flat_p));
        }

        Self::compress_explicit(ids)
    }

    pub fn lower_op(&mut self, op: &AnyMilliOp) {
        use crate::milli_graph::ops::LowerResult;

        // If inference already produced concrete values for all outputs,
        // register them directly as constants — no need to lower analytically
        // and re-evaluate through pool_eval.
        let all_numeric = op.outputs().all(|out_id| {
            self.all_infos
                .get(&out_id)
                .is_some_and(|i| i.as_concrete().is_some())
        });
        if all_numeric {
            let groups_before = self.nano.num_groups();
            for out_id in op.outputs() {
                if let Some(info) = self.all_infos.get(&out_id) {
                    self.register_constant(out_id, info);
                }
            }
            let groups_after = self.nano.num_groups();
            let op_id = op.global_id();
            let op_kind = op.op_kind();
            for _ in groups_before..groups_after {
                self.group_provenance.push((op_id, op_kind.clone()));
            }
            return;
        }

        let groups_before = self.nano.num_groups();
        match op.lower_to_nano(self) {
            LowerResult::Lowered => {}
            LowerResult::Unsupported => self.lower_default(op),
        }
        let groups_after = self.nano.num_groups();
        // Record provenance for any new groups.
        let op_id = op.global_id();
        let op_kind = op.op_kind();
        for _ in groups_before..groups_after {
            self.group_provenance.push((op_id, op_kind.clone()));
        }
    }

    /// Default lowering for ops without nano decomposition.
    ///
    /// If all outputs are numeric (constant-folded), registers as constants.
    /// Otherwise, registers an OpaqueOp that calls `eval_new` on the op
    /// through pool_eval. Falls back to boundary if output info is missing.
    pub fn lower_default(&mut self, op: &AnyMilliOp) {
        let all_infos = self.all_infos;
        let op_kind = op.op_kind();

        // If all outputs have concrete values, register as constants (no eval needed).
        let all_numeric = op.outputs().all(|out_id| {
            all_infos
                .get(&out_id)
                .is_some_and(|i| i.as_concrete().is_some())
        });
        if all_numeric {
            for out_id in op.outputs() {
                if let Some(info) = all_infos.get(&out_id) {
                    self.register_constant(out_id, info);
                }
            }
            return;
        }

        // Check all outputs have shape info (needed for opaque op registration).
        let all_outputs_known = op.outputs().all(|out_id| {
            all_infos
                .get(&out_id)
                .is_some_and(|i| i.rank_if_known().is_some())
        });

        if !all_outputs_known {
            // Fall back to boundary if we can't determine output shapes.
            for out_id in op.outputs() {
                if let Some(info) = all_infos.get(&out_id) {
                    self.register_boundary(out_id, info, &op_kind);
                } else {
                    self.register_opaque(out_id);
                }
            }
            self.push_unsupported(op, &op_kind);
            return;
        }

        // Register as an opaque op — pool_eval will call eval_new on the cloned op.
        let input_ids: Vec<GlobalId> = op.inputs().collect();
        let output_ids: Vec<GlobalId> = op.outputs().collect();
        let eval_fn = std::sync::Arc::new(MilliOpOpaqueEval(op.clone()));
        self.register_opaque_op(eval_fn, &op_kind, &input_ids, &output_ids);
    }

    /// Register an opaque milli-op that can't be decomposed into scalar nano-ops.
    ///
    /// Creates OpaqueOutput atom groups for each output and registers them in the
    /// tensor_map. Input tensor mappings are built from the tensor_map entries.
    pub fn register_opaque_op(
        &mut self,
        eval_fn: std::sync::Arc<dyn crate::nano_graph::ops::OpaqueEval>,
        name: &str,
        input_ids: &[GlobalId],
        output_ids: &[GlobalId],
    ) {
        use crate::nano_graph::ops::{OpaqueOp, OpaqueTensorMapping};

        // Build input mappings from tensor_map.
        // Opaque ops expect contiguous atom ranges (base..base+count).
        // If a tensor map is segmented (e.g. from zero-cost Concat) or has
        // non-row-major strides, flatten it into a contiguous range first
        // by inserting identity-copy groups.
        let inputs: Vec<OpaqueTensorMapping> = input_ids
            .iter()
            .filter_map(|&id| {
                let tam = self.tensor_map.get(&id)?;
                let known_dims = tam.known_dims();

                if !tam.segments.is_empty() || !tam.is_contiguous() {
                    // Non-contiguous: allocate contiguous atoms and insert identity copies.
                    let count = tam.count;
                    let dtype = tam.dtype;
                    let base = self.nano.alloc_contiguous_copy(tam, count);
                    Some(OpaqueTensorMapping {
                        base,
                        count,
                        shape: known_dims,
                        dtype,
                    })
                } else {
                    Some(OpaqueTensorMapping {
                        base: tam.base_id,
                        count: tam.count,
                        shape: known_dims,
                        dtype: tam.dtype,
                    })
                }
            })
            .collect();

        // Build output mappings from all_infos (shape/dtype).
        let outputs: Vec<OpaqueTensorMapping> = output_ids
            .iter()
            .filter_map(|&id| {
                let info = self.all_infos.get(&id)?;
                let dtype = info.dtype();
                let shape: Vec<u64> = if let Some(rank) = info.rank_if_known() {
                    (0..rank)
                        .map(|i| info.dim_if_known(i).unwrap_or(1))
                        .collect()
                } else {
                    vec![1]
                };
                let count: u64 = shape.iter().product();
                Some(OpaqueTensorMapping {
                    base: crate::nano_graph::pattern::AtomId(0), // filled by push_opaque_op
                    count,
                    shape,
                    dtype,
                })
            })
            .collect();

        if inputs.len() != input_ids.len() || outputs.len() != output_ids.len() {
            // Some inputs/outputs missing — fall back to boundary.
            for &out_id in output_ids {
                self.register_opaque_id(out_id);
            }
            return;
        }

        let op = OpaqueOp {
            eval_fn,
            inputs,
            outputs,
            name: name.to_string(),
        };

        let output_bases = self.nano.push_opaque_op(op);

        // Register outputs in tensor_map.
        for (i, &out_id) in output_ids.iter().enumerate() {
            let base_id = output_bases[i];
            let info = self.all_infos.get(&out_id).unwrap();
            let dtype = info.dtype();
            let shape: Vec<u64> = if let Some(rank) = info.rank_if_known() {
                (0..rank)
                    .map(|i| info.dim_if_known(i).unwrap_or(1))
                    .collect()
            } else {
                vec![1]
            };
            let strides = TensorAtomMap::compute_strides(&shape);
            let layout: Vec<DimKind> = shape.iter().map(|&d| DimKind::Known(d)).collect();
            let count: u64 = shape.iter().product();
            self.tensor_map.insert(
                out_id,
                TensorAtomMap::simple(base_id, count, dtype, layout, strides, vec![]),
            );
        }
    }

    /// Register a tensor ID as opaque (unknown, single atom).
    fn register_opaque_id(&mut self, id: GlobalId) {
        self.register_opaque(id);
    }

    /// Build an InputRef for a pointwise (element-by-element) read of a tensor.
    ///
    /// For row-major tensors this returns `Affine { base, stride: 1 }`.
    /// For non-row-major tensors (transposed views, non-contiguous slices)
    /// this builds an Explicit mapping so each output atom reads the
    /// correct source atom.
    pub fn pointwise_input_ref(in_map: &TensorAtomMap) -> InputRef {
        let known_dims = in_map.known_dims();
        let row_major = TensorAtomMap::compute_strides(&known_dims);
        if in_map.known_strides == row_major || in_map.count <= 1 {
            InputRef::affine(in_map.base_id, 1)
        } else {
            let mut ids = Vec::with_capacity(in_map.count as usize);
            for flat in 0..in_map.count {
                ids.push(in_map.atom_id_for_element(flat));
            }
            Self::compress_explicit(ids)
        }
    }

    /// Cast / CastLike / other shape-preserving identity ops.
    ///
    /// If input and output dtypes match, this is a zero-cost view (no atoms created).
    /// Otherwise, emits an Identity group that performs the dtype cast.
    pub fn lower_identity_passthrough<T: Node>(
        &mut self,
        op: &T,
    ) -> crate::milli_graph::ops::LowerResult {
        self.lower_passthrough_with_op(op, ScalarOp::Identity)
    }

    /// Like `lower_identity_passthrough`, but uses a `ScalarOp::Cast` with
    /// the given saturation mode.
    pub fn lower_cast_passthrough<T: Node>(
        &mut self,
        op: &T,
        saturating: bool,
    ) -> crate::milli_graph::ops::LowerResult {
        self.lower_passthrough_with_op(op, ScalarOp::Cast { saturating })
    }

    fn lower_passthrough_with_op<T: Node>(
        &mut self,
        op: &T,
        cast_op: ScalarOp<'p, P>,
    ) -> crate::milli_graph::ops::LowerResult {
        let all_infos = self.all_infos;
        let in_id = Node::inputs(op).next().unwrap();
        let out_id = Node::outputs(op).next().unwrap();

        let Some(in_map) = self.tensor_map.get(&in_id).cloned() else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };
        let Some(out_info) = all_infos.get(&out_id) else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };

        let in_dt = all_infos.get(&in_id).map(Self::ndt);
        let out_dt = Self::ndt(out_info);

        // If dtypes match, this is a no-op — just re-register the tensor.
        if in_dt == Some(out_dt) {
            self.tensor_map.insert(out_id, in_map);
            return crate::milli_graph::ops::LowerResult::Lowered;
        }

        // Dtype differs — emit a cast group.
        let known_dims = in_map.known_dims();
        let input_ref = Self::pointwise_input_ref(&in_map);

        let base_id = self.nano.push_group(
            in_map.count,
            out_dt,
            cast_op,
            in_map.sym_dims.clone(),
            vec![input_ref],
        );

        self.tensor_map.insert(
            out_id,
            TensorAtomMap::simple(
                base_id,
                in_map.count,
                out_dt,
                in_map.layout.clone(),
                TensorAtomMap::compute_strides(&known_dims),
                in_map.sym_dims.clone(),
            ),
        );

        crate::milli_graph::ops::LowerResult::Lowered
    }

    /// View op: no compute, just re-register with the new shape.
    pub fn lower_view_op<T: Node>(&mut self, op: &T) -> crate::milli_graph::ops::LowerResult {
        let all_infos = self.all_infos;
        let in_id = Node::inputs(op).next().unwrap();
        let out_id = Node::outputs(op).next().unwrap();

        let Some(in_map) = self.tensor_map.get(&in_id).cloned() else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };
        let Some(out_info) = all_infos.get(&out_id) else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };

        let Some((layout, known_dims, sym_dims, count)) = self.classify_dims(out_info) else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };
        let count = count.max(1);

        if count == in_map.count {
            let in_known = in_map.known_dims();
            let in_rowmajor = TensorAtomMap::compute_strides(&in_known);
            let is_row_major = in_map.known_strides == in_rowmajor
                || in_map.count <= 1
                || in_map.segments.is_empty() && in_map.known_strides.iter().all(|&s| s <= 1);

            if is_row_major && in_map.segments.is_empty() {
                self.tensor_map.insert(
                    out_id,
                    TensorAtomMap::simple(
                        in_map.base_id,
                        count,
                        Self::ndt(out_info),
                        layout,
                        TensorAtomMap::compute_strides(&known_dims),
                        sym_dims,
                    ),
                );
            } else {
                let dt = Self::ndt(out_info);
                let input_ref = Self::pointwise_input_ref(&in_map);
                let base_id = self.nano.push_group(
                    count,
                    dt,
                    ScalarOp::Identity,
                    in_map.sym_dims.clone(),
                    vec![input_ref],
                );
                self.tensor_map.insert(
                    out_id,
                    TensorAtomMap::simple(
                        base_id,
                        count,
                        dt,
                        layout,
                        TensorAtomMap::compute_strides(&known_dims),
                        sym_dims,
                    ),
                );
            }
            crate::milli_graph::ops::LowerResult::Lowered
        } else {
            crate::milli_graph::ops::LowerResult::Unsupported
        }
    }

    /// Extract concrete i64 values from a tensor in all_infos.
    pub fn extract_i64(
        all_infos: &HashMap<GlobalId, LowerTensorInfo<'a, 'p, P>>,
        id: &GlobalId,
    ) -> Option<Vec<i64>> {
        let info = all_infos.get(id)?;
        let concrete = info.as_concrete()?;
        let n = concrete.numel();
        let vals: Vec<i64> = (0..n).map(|i| concrete.read_element(i).to_i64()).collect();
        Some(vals)
    }

    /// Lower ReduceSum or ReduceMax over known axes.
    ///
    /// Returns `Lowered` on success, `Unsupported` if this configuration
    /// can't be decomposed (multi-axis, symbolic axis, etc.).
    pub fn lower_reduce<R, F>(
        &mut self,
        reduce: &R,
        make_reduce_op: F,
    ) -> crate::milli_graph::ops::LowerResult
    where
        R: Node,
        R: ReduceAccessors,
        F: Fn(NumericDType, u64, i64) -> ScalarOp<'p, P>,
    {
        let all_infos = self.all_infos;
        let in_id = Node::inputs(reduce).next().unwrap();
        let out_id = Node::outputs(reduce).next().unwrap();

        let Some(in_map) = self.tensor_map.get(&in_id).cloned() else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };
        let Some(out_info) = all_infos.get(&out_id) else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };

        // Get the concrete reduction axes.
        let axes: Vec<i64> = if let Some(axes_id) = reduce.axes_tensor() {
            let Some(vals) = Self::extract_i64(all_infos, &axes_id) else {
                return crate::milli_graph::ops::LowerResult::Unsupported;
            };
            vals
        } else if reduce.noop_with_empty_axes() {
            // No axes + noop = identity.
            let Some((layout, known_dims, sym_dims, count)) = self.classify_dims(out_info) else {
                return crate::milli_graph::ops::LowerResult::Unsupported;
            };
            let count = count.max(1);
            let dt = Self::ndt(out_info);
            let input_ref = Self::pointwise_input_ref(&in_map);
            let base_id = self.nano.push_group(
                count,
                dt,
                ScalarOp::Identity,
                sym_dims.clone(),
                vec![input_ref],
            );
            self.tensor_map.insert(
                out_id,
                TensorAtomMap::simple(
                    base_id,
                    count,
                    dt,
                    layout,
                    TensorAtomMap::compute_strides(&known_dims),
                    sym_dims,
                ),
            );
            return crate::milli_graph::ops::LowerResult::Lowered;
        } else {
            // No axes tensor and not noop → reduce all axes.
            let in_rank = in_map.layout.len();
            (0..in_rank as i64).collect()
        };

        let in_rank = in_map.layout.len();

        // Normalize axes and check they're all known dims.
        let norm_axes: Vec<usize> = axes
            .iter()
            .map(|&a| {
                if a < 0 {
                    (a + in_rank as i64) as usize
                } else {
                    a as usize
                }
            })
            .collect();

        // Map tensor-axis → known-dim index.
        let axis_to_known_idx: Vec<Option<usize>> = {
            let mut ki = 0;
            in_map
                .layout
                .iter()
                .map(|d| {
                    if matches!(d, DimKind::Known(_)) {
                        let idx = ki;
                        ki += 1;
                        Some(idx)
                    } else {
                        None
                    }
                })
                .collect()
        };

        let in_known: Vec<u64> = in_map
            .layout
            .iter()
            .filter_map(|d| {
                if let DimKind::Known(s) = d {
                    Some(*s)
                } else {
                    None
                }
            })
            .collect();

        // Check all reduction axes are known dims.
        let mut reduce_known_indices = Vec::new();
        for &ax in &norm_axes {
            if ax >= in_rank {
                return crate::milli_graph::ops::LowerResult::Unsupported;
            }
            let Some(ki) = axis_to_known_idx[ax] else {
                return crate::milli_graph::ops::LowerResult::Unsupported;
            };
            reduce_known_indices.push(ki);
        }

        // Compute the reduction extent (product of reduced known dims).
        let reduce_extent: u64 = reduce_known_indices
            .iter()
            .map(|&ki| in_known[ki])
            .product();
        if reduce_extent == 0 {
            // Reducing over 0 elements produces the identity value.
            // This can't be expressed as a Reduce nano-op; fall to opaque.
            return crate::milli_graph::ops::LowerResult::Unsupported;
        }

        // Output known dims = input known dims minus the reduced ones.
        let out_known: Vec<u64> = in_known
            .iter()
            .enumerate()
            .filter(|(i, _)| !reduce_known_indices.contains(i))
            .map(|(_, &v)| v)
            .collect();
        let out_count = out_known.iter().product::<u64>().max(1);

        // For now, handle single-axis reduction (covers most cases).
        if reduce_known_indices.len() != 1 {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        }

        let rki = reduce_known_indices[0];
        let in_strides = &in_map.known_strides;
        let reduce_stride = in_strides[rki] as i64;

        let out_strides_local = TensorAtomMap::compute_strides(&out_known);

        let mut base_ids = Vec::with_capacity(out_count as usize);
        for flat_out in 0..out_count {
            let mut out_indices = vec![0u64; out_known.len()];
            let mut rem = flat_out;
            for (i, &stride) in out_strides_local.iter().enumerate() {
                if stride > 0 {
                    out_indices[i] = rem / stride;
                    rem %= stride;
                }
            }

            let mut in_indices = Vec::with_capacity(in_known.len());
            let mut oi = 0;
            for ki in 0..in_known.len() {
                if ki == rki {
                    in_indices.push(0u64);
                } else {
                    in_indices.push(out_indices[oi]);
                    oi += 1;
                }
            }

            let mut in_flat = 0u64;
            for (i, &stride) in in_strides.iter().enumerate() {
                in_flat += in_indices[i] * stride;
            }
            base_ids.push(in_flat);
        }

        let is_affine = if out_count <= 1 {
            true
        } else {
            let stride = base_ids[1] as i64 - base_ids[0] as i64;
            base_ids
                .windows(2)
                .all(|w| (w[1] as i64 - w[0] as i64) == stride)
        };

        let input_ref = if is_affine && out_count > 0 {
            let stride_i = if out_count > 1 {
                base_ids[1] as i64 - base_ids[0] as i64
            } else {
                1
            };
            InputRef::affine(in_map.base_id.offset(base_ids[0]), stride_i)
        } else if out_count > 0 {
            InputRef::Explicit(
                base_ids
                    .iter()
                    .map(|&offset| in_map.base_id.offset(offset))
                    .collect(),
            )
        } else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };

        let out_dt = Self::ndt(out_info);
        let in_dt = all_infos.get(&in_id).map(Self::ndt).unwrap_or(out_dt);
        let compute_dt = match in_dt {
            NumericDType::BF16 | NumericDType::F16 => NumericDType::F32,
            other => other,
        };

        let Some((out_layout, out_known_dims_full, out_sym_dims, _)) = self.classify_dims(out_info)
        else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };

        let reduce_op = make_reduce_op(compute_dt, reduce_extent, reduce_stride);
        let base_id = self.nano.push_group(
            out_count,
            out_dt,
            reduce_op,
            out_sym_dims.clone(),
            vec![input_ref],
        );

        self.tensor_map.insert(
            out_id,
            TensorAtomMap::simple(
                base_id,
                out_count,
                out_dt,
                out_layout,
                TensorAtomMap::compute_strides(&out_known_dims_full),
                out_sym_dims,
            ),
        );

        crate::milli_graph::ops::LowerResult::Lowered
    }

    fn push_unsupported<T: Node>(&mut self, op: &T, name: &str) {
        let all_infos = self.all_infos;
        let in_shapes: Vec<String> = op
            .inputs()
            .map(|id| Self::fmt_info(all_infos.get(&id)))
            .collect();
        let out_shapes: Vec<String> = op
            .outputs()
            .map(|id| Self::fmt_info(all_infos.get(&id)))
            .collect();
        let detail = format!(
            "{} : ({}) → ({})",
            name,
            in_shapes.join(", "),
            out_shapes.join(", ")
        );
        self.unsupported.push((op.global_id(), name.to_string()));
        self.unsupported_details.push(detail);
    }

    pub fn fmt_info(info: Option<&LowerTensorInfo<'_, 'p, P>>) -> String {
        let Some(info) = info else {
            return "?".to_string();
        };
        let r = info.rank_if_known().unwrap_or(0);
        let prefix = if info.as_concrete().is_some() {
            "N"
        } else {
            "R"
        };
        let dims: Vec<String> = (0..r)
            .map(|i| {
                info.dim_if_known(i)
                    .map(|d| d.to_string())
                    .unwrap_or("?".to_string())
            })
            .collect();
        format!("{}[{}]", prefix, dims.join(","))
    }

    /// Register a tensor with no shape info as a single opaque atom.
    pub fn register_opaque(&mut self, id: GlobalId) {
        if self.tensor_map.contains_key(&id) {
            return;
        }
        let base_id = self.nano.push_atom(
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(0.0)),
            vec![],
            vec![],
        );
        self.tensor_map.insert(
            id,
            TensorAtomMap::simple(base_id, 1, NumericDType::F32, vec![], vec![], vec![]),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DynRank;
    use crate::graph::Graph;
    use crate::milli_graph::MilliOpGraph;
    use crate::milli_graph::ops::MilliOp;
    use crate::milli_graph::ops::constant::ConstantValue;
    use crate::nano_graph::pattern::AtomRange;
    use crate::nano_graph::pool_eval;
    use crate::numeric_tensor::TensorLayout;
    use crate::pool::{Pool, TrackedPool};

    type PoolTensor = crate::numeric_tensor::NumericTensor<'static, DynRank, SystemPool>;

    /// Build a pool-backed tensor from a Vec of typed values and a shape.
    fn pool_tensor<T: ConstantValue>(values: Vec<T>, shape: Vec<u64>) -> PoolTensor {
        let layout = TensorLayout::<DynRank>::row_major(shape, T::dtype());
        let buf = SystemPool
            .allocate(layout.buffer_size_bytes())
            .expect("pool alloc");
        let mut t = crate::numeric_tensor::NumericTensor::from_parts(buf, layout);
        for (i, v) in values.iter().enumerate() {
            t.write_element(i, v.to_scalar());
        }
        t
    }

    /// Build a milli graph, eval through both per-op eval_new and nano pool_eval,
    /// compare results.
    fn check_integrity(
        build_graph: impl FnOnce(
            &mut MilliOpGraph,
            &mut rand::rngs::ThreadRng,
        ) -> (Vec<GlobalId>, Vec<GlobalId>),
        inputs: Vec<PoolTensor>,
    ) {
        let mut rng = rand::rng();
        let (mut milli, _ext_map) = MilliOpGraph::new(std::iter::empty(), &mut rng);

        let (input_ids, output_ids) = build_graph(&mut milli, &mut rng);
        assert_eq!(input_ids.len(), inputs.len());

        // Prepare inputs for lowering info.
        let mut info_inputs: HashMap<GlobalId, LowerTensorInfo> = HashMap::new();
        let mut intermediates: HashMap<GlobalId, PoolTensor> = HashMap::new();
        for (id, tensor) in input_ids.iter().zip(inputs.iter()) {
            info_inputs.insert(*id, LowerTensorInfo::from_view(&tensor.view(), &SystemPool));
            intermediates.insert(*id, tensor.to_tensor(&SystemPool).unwrap());
        }

        // Eval through per-op eval_new (reference implementation).
        for &op_id in milli.op_ordering() {
            let op = milli.get_node_by_id(&op_id).unwrap();
            let op_input_ids: Vec<_> = op.inputs().collect();
            let input_views: Vec<_> = op_input_ids
                .iter()
                .map(|id| intermediates[id].view())
                .collect();
            let results = op.eval_new(&input_views, &SystemPool).unwrap();
            let op_output_ids: Vec<_> = op.outputs().collect();
            for (tid, val) in op_output_ids.into_iter().zip(results) {
                intermediates.insert(tid, val);
            }
        }

        // Lower to NanoGraph.
        let result = lower_with_info(&milli, &info_inputs, &SystemPool).unwrap();
        assert!(
            result.unsupported.is_empty(),
            "Unsupported ops: {:?}",
            result.unsupported_details
        );

        // Map to (TAMI, view) pairs for pool_eval.
        let input_views: Vec<_> = inputs.iter().map(|t| t.view()).collect();
        let eval_input_tamis: Vec<_> = result
            .graph
            .input_tensors()
            .iter()
            .filter_map(|it| {
                let idx = input_ids.iter().position(|&id| id == it.tensor_id)?;
                // Build a contiguous TAMI for the input.
                let tam_info = result
                    .tensor_map
                    .values()
                    .find(|t| t.base_id == it.base_id)?;
                Some((tam_info, idx))
            })
            .collect();
        let eval_inputs: Vec<_> = eval_input_tamis
            .iter()
            .map(|(tam, idx)| (*tam, &input_views[*idx]))
            .collect();

        // Build output TAMIs.
        let output_tamis: Vec<_> = output_ids
            .iter()
            .map(|out_id| {
                let tam = result.tensor_map.get(out_id).unwrap();
                assert!(
                    tam.sym_dims.is_empty(),
                    "Sym dims not yet supported in test"
                );
                tam
            })
            .collect();

        // Eval NanoGraph via pool_eval — returns correctly-shaped tensors.
        let pool = TrackedPool::new(None);
        let nano_results =
            pool_eval::pool_eval(&result.graph, &eval_inputs, &output_tamis, &pool).unwrap();

        // Compare outputs — results are in output_ids order, already shaped.
        for (out_id, nano_tensor) in output_ids.iter().zip(nano_results.iter()) {
            let milli_tensor = &intermediates[out_id];
            let milli_numel = milli_tensor.numel();
            let milli_flat: Vec<f64> = (0..milli_numel)
                .map(|i| milli_tensor.read_element(i).to_f64())
                .collect();
            let tam = result.tensor_map.get(out_id).unwrap();

            let nano_flat: Vec<f64> = (0..nano_tensor.numel())
                .map(|i| nano_tensor.read_element(i).to_f64())
                .collect();

            assert_eq!(
                milli_flat.len(),
                nano_flat.len(),
                "Output {:?}: milli has {} elements, nano has {}",
                out_id,
                milli_flat.len(),
                nano_flat.len()
            );

            // For BOOL outputs, compare as 0/1.
            let is_bool = tam.dtype == NumericDType::BOOL;
            for (i, (m, n)) in milli_flat.iter().zip(nano_flat.iter()).enumerate() {
                let diff = if is_bool {
                    let mb: f64 = if *m != 0.0 { 1.0 } else { 0.0 };
                    let nb: f64 = if *n != 0.0 { 1.0 } else { 0.0 };
                    (mb - nb).abs()
                } else {
                    (m - n).abs()
                };
                let tol = if is_bool {
                    0.0
                } else {
                    1e-4 * m.abs().max(1.0)
                };
                assert!(
                    diff <= tol,
                    "Output {:?} element {}: milli={} nano={} diff={}",
                    out_id,
                    i,
                    m,
                    n,
                    diff
                );
            }
        }
    }

    #[test]
    fn test_add_two_vectors() {
        check_integrity(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = graph.add_input(rng);
                let c = crate::milli_graph::ops::SimpleBinary::add(graph, a, b, rng);
                (vec![a, b], vec![c])
            },
            vec![
                pool_tensor(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]),
                pool_tensor(vec![10.0f32, 20.0, 30.0, 40.0], vec![4]),
            ],
        );
    }

    #[test]
    fn test_mul_add_chain() {
        check_integrity(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = graph.add_input(rng);
                let c = graph.add_input(rng);
                let ab = crate::milli_graph::ops::SimpleBinary::mul(graph, a, b, rng);
                let abc = crate::milli_graph::ops::SimpleBinary::add(graph, ab, c, rng);
                (vec![a, b, c], vec![abc])
            },
            vec![
                pool_tensor(vec![1.0f32, 2.0, 3.0], vec![3]),
                pool_tensor(vec![4.0f32, 5.0, 6.0], vec![3]),
                pool_tensor(vec![0.1f32, 0.2, 0.3], vec![3]),
            ],
        );
    }

    #[test]
    fn test_broadcast_add() {
        // [2,3] + [3] → [2,3]
        check_integrity(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = graph.add_input(rng);
                let c = crate::milli_graph::ops::SimpleBinary::add(graph, a, b, rng);
                (vec![a, b], vec![c])
            },
            vec![
                pool_tensor(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
                pool_tensor(vec![10.0f32, 20.0, 30.0], vec![3]),
            ],
        );
    }

    #[test]
    fn test_unary_exp() {
        check_integrity(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = crate::milli_graph::ops::SimpleUnaryOp::exp(graph, a, rng);
                (vec![a], vec![b])
            },
            vec![pool_tensor(vec![0.0f32, 1.0, -1.0, 0.5], vec![4])],
        );
    }

    #[test]
    fn test_matmul_2d() {
        // [2,3] @ [3,2] → [2,2]
        check_integrity(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = graph.add_input(rng);
                let c = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    graph,
                    a,
                    b,
                    NumericDType::F32,
                    rng,
                );
                (vec![a, b], vec![c])
            },
            vec![
                pool_tensor(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
                pool_tensor(vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2]),
            ],
        );
    }

    #[test]
    fn test_matmul_merged_mul_groups() {
        // Verify that matmul lowering produces M Mul groups (not M*K).
        // MatMul(4, 8, 16): A=[4,8], B=[8,16] -> C=[4,16]
        // Old: 4*8 = 32 Mul groups of 16 atoms each
        // New: 4 Mul groups of 8*16 = 128 atoms each (StridedBroadcast)
        use crate::nano_graph::pattern::InputRef;
        check_integrity(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = graph.add_input(rng);
                let c = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    graph,
                    a,
                    b,
                    NumericDType::F32,
                    rng,
                );
                (vec![a, b], vec![c])
            },
            vec![
                pool_tensor(vec![1.0f32; 4 * 8], vec![4, 8]),
                pool_tensor(vec![1.0f32; 8 * 16], vec![8, 16]),
            ],
        );

        // Also verify group structure directly.
        // Use shape-only (non-concrete) inputs so lowering produces
        // analytical groups rather than short-circuiting to constants.
        let mut rng = rand::rng();
        let (mut milli, _ext_map) =
            crate::milli_graph::MilliOpGraph::new(std::iter::empty(), &mut rng);
        let a_id = milli.add_input(&mut rng);
        let b_id = milli.add_input(&mut rng);
        let _c_id = crate::milli_graph::ops::MatMul::push_new_default_precision(
            &mut milli,
            a_id,
            b_id,
            NumericDType::F32,
            &mut rng,
        );

        let mut info = std::collections::HashMap::new();
        info.insert(
            a_id,
            LowerTensorInfo::from_dtype_and_shape(NumericDType::F32, &[4, 8]),
        );
        info.insert(
            b_id,
            LowerTensorInfo::from_dtype_and_shape(NumericDType::F32, &[8, 16]),
        );

        let result = super::lower_with_info(&milli, &info, &SystemPool).unwrap();
        let graph = &result.graph;

        // Count Mul groups — should be M=4, not M*K=32.
        let mul_groups: Vec<_> = graph
            .groups()
            .iter()
            .filter(|g| {
                matches!(
                    &g.op,
                    crate::nano_graph::ops::ScalarOp::Binary {
                        op: crate::nano_graph::ops::ScalarBinOp::Mul,
                        ..
                    }
                )
            })
            .collect();
        assert_eq!(
            mul_groups.len(),
            4,
            "Expected M=4 merged Mul groups, got {}",
            mul_groups.len()
        );

        // Each merged Mul group should have K*N = 8*16 = 128 atoms.
        for (i, g) in mul_groups.iter().enumerate() {
            assert_eq!(
                g.count, 128,
                "Merged Mul group {} should have K*N=128 atoms, got {}",
                i, g.count
            );
            // Input 0 should be StridedBroadcast with repeat=N=16.
            match &g.inputs[0] {
                InputRef::Strided {
                    dim_strides,
                    dim_shape,
                    ..
                } if dim_strides.len() == 2 && dim_strides[1] == 0 => {
                    assert_eq!(dim_shape[1], 16, "StridedBroadcast repeat should be N=16");
                }
                other => panic!("Expected StridedBroadcast for input 0, got {:?}", other),
            }
            // Input 1 should be Affine with stride=1.
            match &g.inputs[1] {
                InputRef::Strided {
                    dim_strides,
                    dim_shape,
                    ..
                } if dim_strides.len() == 1 && dim_shape[0] == u64::MAX => {
                    assert_eq!(dim_strides[0], 1, "Affine stride should be 1");
                }
                other => panic!("Expected Affine for input 1, got {:?}", other),
            }
        }
    }

    #[test]
    fn test_gather_axis0_embedding() {
        // Gather(data=[4, 3], indices=[2], axis=0) → [2, 3]
        // This is the common embedding lookup pattern.
        // data = [[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]]
        // indices = [1, 3]
        // expected output = [[40, 50, 60], [100, 110, 120]]
        check_integrity(
            |graph, rng| {
                let data = graph.add_input(rng);
                let indices = graph.add_input(rng);
                let out = crate::milli_graph::ops::Gather::push_new(graph, data, indices, 0, rng);
                (vec![data, indices], vec![out])
            },
            vec![
                pool_tensor(
                    vec![
                        10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0,
                        120.0,
                    ],
                    vec![4, 3],
                ),
                pool_tensor(vec![1.0f32, 3.0], vec![2]),
            ],
        );
    }

    #[test]
    fn test_split_zero_cost_axis0() {
        // Split [4, 3] along axis 0 into [2, 3] and [2, 3].
        // Both outputs should be zero-cost views (no Identity groups).
        check_integrity(
            |graph, rng| {
                let data = graph.add_input(rng);
                let out0 =
                    crate::milli_graph::ops::Split::push_new(graph, data, None, 0, Some(2), 0, rng);
                let out1 =
                    crate::milli_graph::ops::Split::push_new(graph, data, None, 0, Some(2), 1, rng);
                (vec![data], vec![out0, out1])
            },
            vec![pool_tensor(
                vec![1.0f32, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
                vec![4, 3],
            )],
        );

        // Verify no Identity groups were created.
        let mut rng = rand::rng();
        let (mut milli, _) = MilliOpGraph::new(std::iter::empty(), &mut rng);
        let data = milli.add_input(&mut rng);
        let _out0 = crate::milli_graph::ops::Split::push_new(
            &mut milli,
            data,
            None,
            0,
            Some(2),
            0,
            &mut rng,
        );
        let _out1 = crate::milli_graph::ops::Split::push_new(
            &mut milli,
            data,
            None,
            0,
            Some(2),
            1,
            &mut rng,
        );
        let tensor = pool_tensor(vec![1.0f32; 12], vec![4, 3]);
        let mut info = std::collections::HashMap::new();
        info.insert(
            data,
            LowerTensorInfo::from_view(&tensor.view(), &SystemPool),
        );
        let result = super::lower_with_info(&milli, &info, &SystemPool).unwrap();
        let identity_count = result
            .graph
            .groups()
            .iter()
            .filter(|g| matches!(&g.op, crate::nano_graph::ops::ScalarOp::Identity))
            .count();
        assert_eq!(
            identity_count, 0,
            "Split should be zero-cost (no Identity groups), got {}",
            identity_count
        );
    }

    #[test]
    fn test_split_zero_cost_axis1() {
        // Split [2, 6] along axis 1 into [2, 3] and [2, 3].
        check_integrity(
            |graph, rng| {
                let data = graph.add_input(rng);
                let out0 =
                    crate::milli_graph::ops::Split::push_new(graph, data, None, 1, Some(2), 0, rng);
                let out1 =
                    crate::milli_graph::ops::Split::push_new(graph, data, None, 1, Some(2), 1, rng);
                (vec![data], vec![out0, out1])
            },
            vec![pool_tensor(
                (1..=12).map(|v| v as f32).collect(),
                vec![2, 6],
            )],
        );
    }

    #[test]
    fn test_split_then_add() {
        // Split [4] into [2] and [2], then add the two halves.
        // Verifies zero-cost split outputs are correctly addressed by downstream ops.
        check_integrity(
            |graph, rng| {
                let data = graph.add_input(rng);
                let out0 =
                    crate::milli_graph::ops::Split::push_new(graph, data, None, 0, Some(2), 0, rng);
                let out1 =
                    crate::milli_graph::ops::Split::push_new(graph, data, None, 0, Some(2), 1, rng);
                let sum = crate::milli_graph::ops::SimpleBinary::add(graph, out0, out1, rng);
                (vec![data], vec![sum])
            },
            vec![pool_tensor(vec![1.0f32, 2., 3., 4.], vec![4])],
        );
    }

    #[test]
    fn test_slice_zero_cost() {
        // Slice [6] with start=1, end=4, step=1 → [3]
        check_integrity(
            |graph, rng| {
                let data = graph.add_input(rng);
                let starts = crate::milli_graph::ops::Constant::from_vec(graph, vec![1i64], rng);
                let ends = crate::milli_graph::ops::Constant::from_vec(graph, vec![4i64], rng);
                let out = crate::milli_graph::ops::Slice::push_new(
                    graph, data, starts, ends, None, None, rng,
                );
                (vec![data], vec![out])
            },
            vec![pool_tensor(vec![10.0f32, 20., 30., 40., 50., 60.], vec![6])],
        );
    }

    #[test]
    fn test_slice_zero_cost_2d() {
        // Slice [3, 4] along axis 0 with start=1, end=3 → [2, 4]
        check_integrity(
            |graph, rng| {
                let data = graph.add_input(rng);
                let starts = crate::milli_graph::ops::Constant::from_vec(graph, vec![1i64], rng);
                let ends = crate::milli_graph::ops::Constant::from_vec(graph, vec![3i64], rng);
                let axes = crate::milli_graph::ops::Constant::from_vec(graph, vec![0i64], rng);
                let out = crate::milli_graph::ops::Slice::push_new(
                    graph,
                    data,
                    starts,
                    ends,
                    None,
                    Some(axes),
                    rng,
                );
                (vec![data], vec![out])
            },
            vec![pool_tensor(
                (1..=12).map(|v| v as f32).collect(),
                vec![3, 4],
            )],
        );
    }

    #[test]
    fn test_slice_then_add() {
        // Slice [4] two ways and add: data[0:2] + data[2:4]
        check_integrity(
            |graph, rng| {
                let data = graph.add_input(rng);
                let s0 = crate::milli_graph::ops::Constant::from_vec(graph, vec![0i64], rng);
                let e0 = crate::milli_graph::ops::Constant::from_vec(graph, vec![2i64], rng);
                let s1 = crate::milli_graph::ops::Constant::from_vec(graph, vec![2i64], rng);
                let e1 = crate::milli_graph::ops::Constant::from_vec(graph, vec![4i64], rng);
                let out0 =
                    crate::milli_graph::ops::Slice::push_new(graph, data, s0, e0, None, None, rng);
                let out1 =
                    crate::milli_graph::ops::Slice::push_new(graph, data, s1, e1, None, None, rng);
                let sum = crate::milli_graph::ops::SimpleBinary::add(graph, out0, out1, rng);
                (vec![data], vec![sum])
            },
            vec![pool_tensor(vec![1.0f32, 2., 3., 4.], vec![4])],
        );
    }

    #[test]
    fn test_slice_zero_cost_inner_axis() {
        // Slice [3, 4] along axis 1 with start=1, end=3 → [3, 2]
        // This is an inner-axis slice that falls back to Explicit, but
        // should still produce correct results.
        check_integrity(
            |graph, rng| {
                let data = graph.add_input(rng);
                let starts = crate::milli_graph::ops::Constant::from_vec(graph, vec![1i64], rng);
                let ends = crate::milli_graph::ops::Constant::from_vec(graph, vec![3i64], rng);
                let axes = crate::milli_graph::ops::Constant::from_vec(graph, vec![1i64], rng);
                let out = crate::milli_graph::ops::Slice::push_new(
                    graph,
                    data,
                    starts,
                    ends,
                    None,
                    Some(axes),
                    rng,
                );
                (vec![data], vec![out])
            },
            vec![pool_tensor(
                (1..=12).map(|v| v as f32).collect(),
                vec![3, 4],
            )],
        );
    }

    #[test]
    fn test_slice_no_identity_groups() {
        // Verify Slice [6] start=1 end=4 creates no Identity groups (zero-cost).
        let mut rng = rand::rng();
        let (mut milli, _) = MilliOpGraph::new(std::iter::empty(), &mut rng);
        let data = milli.add_input(&mut rng);
        let starts = crate::milli_graph::ops::Constant::from_vec(&mut milli, vec![1i64], &mut rng);
        let ends = crate::milli_graph::ops::Constant::from_vec(&mut milli, vec![4i64], &mut rng);
        let _out = crate::milli_graph::ops::Slice::push_new(
            &mut milli, data, starts, ends, None, None, &mut rng,
        );
        let tensor = pool_tensor(vec![1.0f32; 6], vec![6]);
        let mut info = std::collections::HashMap::new();
        info.insert(
            data,
            LowerTensorInfo::from_view(&tensor.view(), &SystemPool),
        );
        let result = super::lower_with_info(&milli, &info, &SystemPool).unwrap();
        let identity_count = result
            .graph
            .groups()
            .iter()
            .filter(|g| matches!(&g.op, crate::nano_graph::ops::ScalarOp::Identity))
            .count();
        assert_eq!(
            identity_count, 0,
            "Slice on outermost axis should be zero-cost (no Identity groups), got {}",
            identity_count
        );
    }

    #[test]
    fn test_concat_zero_cost_contiguous() {
        // Split [6] into [3] and [3], then concat back.
        // The concat should be zero-cost because the split outputs are contiguous.
        check_integrity(
            |graph, rng| {
                let data = graph.add_input(rng);
                let out0 =
                    crate::milli_graph::ops::Split::push_new(graph, data, None, 0, Some(2), 0, rng);
                let out1 =
                    crate::milli_graph::ops::Split::push_new(graph, data, None, 0, Some(2), 1, rng);
                let cat =
                    crate::milli_graph::ops::Concat::push_new(graph, vec![out0, out1], 0, rng);
                (vec![data], vec![cat])
            },
            vec![pool_tensor(vec![10.0f32, 20., 30., 40., 50., 60.], vec![6])],
        );
    }

    #[test]
    fn test_concat_non_contiguous() {
        // Concat two separate inputs — not contiguous, should fall back to Explicit.
        check_integrity(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = graph.add_input(rng);
                let cat = crate::milli_graph::ops::Concat::push_new(graph, vec![a, b], 0, rng);
                (vec![a, b], vec![cat])
            },
            vec![
                pool_tensor(vec![1.0f32, 2., 3.], vec![3]),
                pool_tensor(vec![4.0f32, 5., 6.], vec![3]),
            ],
        );
    }

    #[test]
    fn test_gather_axis0_single_column() {
        // Gather(data=[4], indices=[3], axis=0) → [3]
        // 1D data (d_total=1), 1D indices
        // data = [100, 200, 300, 400]
        // indices = [0, 2, 3]
        // expected output = [100, 300, 400]
        check_integrity(
            |graph, rng| {
                let data = graph.add_input(rng);
                let indices = graph.add_input(rng);
                let out = crate::milli_graph::ops::Gather::push_new(graph, data, indices, 0, rng);
                (vec![data, indices], vec![out])
            },
            vec![
                pool_tensor(vec![100.0f32, 200.0, 300.0, 400.0], vec![4]),
                pool_tensor(vec![0.0f32, 2.0, 3.0], vec![3]),
            ],
        );
    }

    /// Large MatMul to check for accumulation or addressing bugs.
    #[test]
    fn test_three_way_matmul() {
        let m = 4usize;
        let k = 64usize;
        let n = 64usize;
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01 - 1.28).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.007 + 0.5).collect();

        check_integrity(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = graph.add_input(rng);
                let c = crate::milli_graph::ops::MatMul::push_new(
                    graph,
                    a,
                    b,
                    NumericDType::F32,
                    NumericDType::F32,
                    NumericDType::F32,
                    NumericDType::F32,
                    rng,
                );
                (vec![a, b], vec![c])
            },
            vec![
                pool_tensor(a_data, vec![m as u64, k as u64]),
                pool_tensor(b_data, vec![k as u64, n as u64]),
            ],
        );
    }

    /// MatMul -> Add(bias) -> MatMul -> Add(bias) chain.
    #[test]
    fn test_three_way_matmul_chain_with_bias() {
        use crate::milli_graph::ops::{MatMul, SimpleBinary};

        check_integrity(
            |graph, rng| {
                let x = graph.add_input(rng);
                let w1 = graph.add_input(rng);
                let b1 = graph.add_input(rng);
                let w2 = graph.add_input(rng);
                let b2 = graph.add_input(rng);

                let mm1 = MatMul::push_new(
                    graph,
                    x,
                    w1,
                    NumericDType::F32,
                    NumericDType::F32,
                    NumericDType::F32,
                    NumericDType::F32,
                    rng,
                );
                let h = SimpleBinary::add(graph, mm1, b1, rng);
                let mm2 = MatMul::push_new(
                    graph,
                    h,
                    w2,
                    NumericDType::F32,
                    NumericDType::F32,
                    NumericDType::F32,
                    NumericDType::F32,
                    rng,
                );
                let out = SimpleBinary::add(graph, mm2, b2, rng);
                (vec![x, w1, b1, w2, b2], vec![out])
            },
            vec![
                pool_tensor(
                    (0..256).map(|i| (i as f32) * 0.01 - 1.28).collect(),
                    vec![4, 64],
                ),
                pool_tensor(
                    (0..4096).map(|i| (i as f32) * 0.002 - 4.0).collect(),
                    vec![64, 64],
                ),
                pool_tensor((0..64).map(|i| (i as f32) * 0.1 - 3.2).collect(), vec![64]),
                pool_tensor(
                    (0..2048).map(|i| (i as f32) * 0.003 - 3.0).collect(),
                    vec![64, 32],
                ),
                pool_tensor((0..32).map(|i| (i as f32) * 0.05 - 0.8).collect(), vec![32]),
            ],
        );
    }

    /// MatMul -> Transpose -> LayerNorm-like chain.
    #[test]
    fn test_three_way_transpose_layernorm() {
        use crate::milli_graph::ops::{
            Constant, MatMul, Pow, ReduceMean, SimpleBinary, SimpleUnaryOp, Transpose,
        };

        check_integrity(
            |graph, rng| {
                let x = graph.add_input(rng);
                let w = graph.add_input(rng);
                let gamma = graph.add_input(rng);
                let beta = graph.add_input(rng);

                let mm = MatMul::push_new(
                    graph,
                    x,
                    w,
                    NumericDType::F32,
                    NumericDType::F32,
                    NumericDType::F32,
                    NumericDType::F32,
                    rng,
                );
                let transposed = Transpose::push_new(graph, mm, Some(vec![0, 2, 1]), rng);

                let axes_id = Constant::from_vec(graph, vec![-1i64], rng);
                let mean = ReduceMean::push_new(graph, transposed, Some(axes_id), true, false, rng);
                let centered = SimpleBinary::sub(graph, transposed, mean, rng);

                let pow2_id = Constant::from_vec(graph, vec![2.0f32], rng);
                let squared = Pow::push_new(graph, centered, pow2_id, rng);

                let axes_id2 = Constant::from_vec(graph, vec![-1i64], rng);
                let var = ReduceMean::push_new(graph, squared, Some(axes_id2), true, false, rng);

                let eps_id = Constant::from_vec(graph, vec![1e-5f32], rng);
                let var_eps = SimpleBinary::add(graph, var, eps_id, rng);
                let std_dev = SimpleUnaryOp::sqrt(graph, var_eps, rng);
                let normed = SimpleBinary::div(graph, centered, std_dev, rng);
                let scaled = SimpleBinary::mul(graph, normed, gamma, rng);
                let out = SimpleBinary::add(graph, scaled, beta, rng);

                (vec![x, w, gamma, beta], vec![out])
            },
            vec![
                pool_tensor(
                    (0..64).map(|i| (i as f32) * 0.1 - 3.2).collect(),
                    vec![2, 4, 8],
                ),
                pool_tensor(
                    (0..64).map(|i| (i as f32) * 0.02 - 0.64).collect(),
                    vec![8, 8],
                ),
                pool_tensor(vec![1.0f32, 1.1, 0.9, 1.05], vec![4]),
                pool_tensor(vec![0.0f32, 0.1, -0.1, 0.05], vec![4]),
            ],
        );
    }

    // =========================================================================
    // Binary ops — each variant
    // =========================================================================

    #[test]
    fn test_binary_sub() {
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let b = g.add_input(rng);
                let c = crate::milli_graph::ops::SimpleBinary::sub(g, a, b, rng);
                (vec![a, b], vec![c])
            },
            vec![
                pool_tensor(vec![10.0f32, 20.0, 30.0], vec![3]),
                pool_tensor(vec![1.0f32, 2.0, 3.0], vec![3]),
            ],
        );
    }

    #[test]
    fn test_binary_div() {
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let b = g.add_input(rng);
                let c = crate::milli_graph::ops::SimpleBinary::div(g, a, b, rng);
                (vec![a, b], vec![c])
            },
            vec![
                pool_tensor(vec![12.0f32, 20.0, 30.0, 7.0], vec![4]),
                pool_tensor(vec![3.0f32, 4.0, 5.0, 2.0], vec![4]),
            ],
        );
    }

    #[test]
    fn test_binary_max() {
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let b = g.add_input(rng);
                let c = crate::milli_graph::ops::SimpleBinary::max(g, a, b, rng);
                (vec![a, b], vec![c])
            },
            vec![
                pool_tensor(vec![1.0f32, 5.0, 3.0, 8.0], vec![4]),
                pool_tensor(vec![4.0f32, 2.0, 6.0, 1.0], vec![4]),
            ],
        );
    }

    #[test]
    fn test_binary_min() {
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let b = g.add_input(rng);
                let c = crate::milli_graph::ops::SimpleBinary::min(g, a, b, rng);
                (vec![a, b], vec![c])
            },
            vec![
                pool_tensor(vec![1.0f32, 5.0, 3.0, 8.0], vec![4]),
                pool_tensor(vec![4.0f32, 2.0, 6.0, 1.0], vec![4]),
            ],
        );
    }

    #[test]
    fn test_binary_equal() {
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let b = g.add_input(rng);
                let c = crate::milli_graph::ops::SimpleBinary::equal(g, a, b, rng);
                (vec![a, b], vec![c])
            },
            vec![
                pool_tensor(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]),
                pool_tensor(vec![1.0f32, 9.0, 3.0, 0.0], vec![4]),
            ],
        );
    }

    #[test]
    fn test_binary_greater() {
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let b = g.add_input(rng);
                let c = crate::milli_graph::ops::SimpleBinary::greater(g, a, b, rng);
                (vec![a, b], vec![c])
            },
            vec![
                pool_tensor(vec![1.0f32, 5.0, 3.0, 4.0], vec![4]),
                pool_tensor(vec![4.0f32, 2.0, 3.0, 8.0], vec![4]),
            ],
        );
    }

    #[test]
    fn test_binary_less() {
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let b = g.add_input(rng);
                let c = crate::milli_graph::ops::SimpleBinary::less(g, a, b, rng);
                (vec![a, b], vec![c])
            },
            vec![
                pool_tensor(vec![1.0f32, 5.0, 3.0, 4.0], vec![4]),
                pool_tensor(vec![4.0f32, 2.0, 3.0, 8.0], vec![4]),
            ],
        );
    }

    #[test]
    fn test_binary_broadcast_scalar() {
        // scalar + [4] broadcast
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let b = g.add_input(rng);
                let c = crate::milli_graph::ops::SimpleBinary::mul(g, a, b, rng);
                (vec![a, b], vec![c])
            },
            vec![
                pool_tensor(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]),
                pool_tensor(vec![10.0f32], vec![1]),
            ],
        );
    }

    #[test]
    fn test_binary_broadcast_2d() {
        // [3,4] + [4] → broadcast along axis 0
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let b = g.add_input(rng);
                let c = crate::milli_graph::ops::SimpleBinary::add(g, a, b, rng);
                (vec![a, b], vec![c])
            },
            vec![
                pool_tensor((0..12).map(|i| i as f32).collect(), vec![3, 4]),
                pool_tensor(vec![100.0f32, 200.0, 300.0, 400.0], vec![4]),
            ],
        );
    }

    // =========================================================================
    // Unary ops
    // =========================================================================

    #[test]
    fn test_unary_neg() {
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let b = crate::milli_graph::ops::SimpleUnaryOp::neg(g, a, rng);
                (vec![a], vec![b])
            },
            vec![pool_tensor(vec![1.0f32, -2.0, 0.0, 3.5], vec![4])],
        );
    }

    #[test]
    fn test_unary_abs() {
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let b = crate::milli_graph::ops::SimpleUnaryOp::abs(g, a, rng);
                (vec![a], vec![b])
            },
            vec![pool_tensor(vec![-3.0f32, 2.0, -1.0, 0.0], vec![4])],
        );
    }

    #[test]
    fn test_unary_sqrt() {
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let b = crate::milli_graph::ops::SimpleUnaryOp::sqrt(g, a, rng);
                (vec![a], vec![b])
            },
            vec![pool_tensor(vec![4.0f32, 9.0, 16.0, 1.0], vec![4])],
        );
    }

    #[test]
    fn test_unary_reciprocal() {
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let b = crate::milli_graph::ops::SimpleUnaryOp::reciprocal(g, a, rng);
                (vec![a], vec![b])
            },
            vec![pool_tensor(vec![2.0f32, 4.0, 0.5, 1.0], vec![4])],
        );
    }

    #[test]
    fn test_unary_tanh() {
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let b =
                    crate::milli_graph::ops::SimpleUnaryOp::trig(g, a, crate::TrigOp::Tanh, rng);
                (vec![a], vec![b])
            },
            vec![pool_tensor(vec![0.0f32, 1.0, -1.0, 2.0], vec![4])],
        );
    }

    #[test]
    fn test_unary_ln() {
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let b = crate::milli_graph::ops::SimpleUnaryOp::ln(g, a, rng);
                (vec![a], vec![b])
            },
            vec![pool_tensor(vec![1.0f32, 2.0, 0.5, 10.0], vec![4])],
        );
    }

    #[test]
    fn test_unary_floor_ceil() {
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let b = crate::milli_graph::ops::SimpleUnaryOp::floor(g, a, rng);
                let c = crate::milli_graph::ops::SimpleUnaryOp::ceil(g, a, rng);
                (vec![a], vec![b, c])
            },
            vec![pool_tensor(vec![1.5f32, -1.5, 2.0, 0.1], vec![4])],
        );
    }

    // =========================================================================
    // Pow, ClampMin, Where
    // =========================================================================

    #[test]
    fn test_pow() {
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let b = g.add_input(rng);
                let c = crate::milli_graph::ops::Pow::push_new(g, a, b, rng);
                (vec![a, b], vec![c])
            },
            vec![
                pool_tensor(vec![2.0f32, 3.0, 4.0, 5.0], vec![4]),
                pool_tensor(vec![3.0f32, 2.0, 0.5, 1.0], vec![4]),
            ],
        );
    }

    #[test]
    fn test_clamp_min() {
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let b = crate::milli_graph::ops::ClampMin::push_new(g, a, 0.0, rng);
                (vec![a], vec![b])
            },
            vec![pool_tensor(vec![-2.0f32, -0.5, 0.0, 1.0, 3.0], vec![5])],
        );
    }

    #[test]
    fn test_where_op() {
        check_integrity(
            |g, rng| {
                // Where expects BOOL condition. Build via Greater(a, 0).
                let a = g.add_input(rng);
                let x = g.add_input(rng);
                let y = g.add_input(rng);
                let zero = g.add_input(rng);
                let cond = crate::milli_graph::ops::SimpleBinary::greater(g, a, zero, rng);
                let out = crate::milli_graph::ops::Where::push_new(g, cond, x, y, rng);
                (vec![a, x, y, zero], vec![out])
            },
            vec![
                pool_tensor(vec![1.0f32, -1.0, 2.0, -2.0], vec![4]),
                pool_tensor(vec![10.0f32, 20.0, 30.0, 40.0], vec![4]),
                pool_tensor(vec![100.0f32, 200.0, 300.0, 400.0], vec![4]),
                pool_tensor(vec![0.0f32, 0.0, 0.0, 0.0], vec![4]),
            ],
        );
    }

    // =========================================================================
    // View ops: Reshape, Squeeze, Unsqueeze, Expand
    // =========================================================================

    #[test]
    fn test_reshape() {
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let shape = g.add_input(rng);
                let b = crate::milli_graph::ops::Reshape::push_new(g, a, shape, false, rng);
                (vec![a, shape], vec![b])
            },
            vec![
                pool_tensor((0..12).map(|i| i as f32).collect(), vec![3, 4]),
                pool_tensor(vec![2i64, 6], vec![2]),
            ],
        );
    }

    #[test]
    fn test_squeeze() {
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let axes = g.add_input(rng);
                let b = crate::milli_graph::ops::Squeeze::push_new(g, a, axes, rng);
                (vec![a, axes], vec![b])
            },
            vec![
                pool_tensor(vec![1.0f32, 2.0, 3.0], vec![1, 3]),
                pool_tensor(vec![0i64], vec![1]),
            ],
        );
    }

    #[test]
    fn test_unsqueeze() {
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let axes = g.add_input(rng);
                let b = crate::milli_graph::ops::Unsqueeze::push_new(g, a, axes, rng);
                (vec![a, axes], vec![b])
            },
            vec![
                pool_tensor(vec![1.0f32, 2.0, 3.0], vec![3]),
                pool_tensor(vec![0i64], vec![1]),
            ],
        );
    }

    #[test]
    fn test_expand() {
        // [1, 3] → [4, 3]
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let shape = g.add_input(rng);
                let b = crate::milli_graph::ops::Expand::push_new(g, a, shape, rng);
                (vec![a, shape], vec![b])
            },
            vec![
                pool_tensor(vec![1.0f32, 2.0, 3.0], vec![1, 3]),
                pool_tensor(vec![4i64, 3], vec![2]),
            ],
        );
    }

    #[test]
    fn test_transpose_2d() {
        // [3, 4] → [4, 3]
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let b = crate::milli_graph::ops::Transpose::push_new(g, a, Some(vec![1, 0]), rng);
                (vec![a], vec![b])
            },
            vec![pool_tensor((0..12).map(|i| i as f32).collect(), vec![3, 4])],
        );
    }

    #[test]
    fn test_transpose_3d() {
        // [2, 3, 4] → [2, 4, 3]  perm=[0,2,1]
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let b =
                    crate::milli_graph::ops::Transpose::push_new(g, a, Some(vec![0, 2, 1]), rng);
                (vec![a], vec![b])
            },
            vec![pool_tensor(
                (0..24).map(|i| i as f32).collect(),
                vec![2, 3, 4],
            )],
        );
    }

    // =========================================================================
    // Cast / CastLike
    // =========================================================================

    #[test]
    fn test_cast_f32_to_f64() {
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let b = crate::milli_graph::ops::Cast::push_new(g, a, NumericDType::F64, rng);
                (vec![a], vec![b])
            },
            vec![pool_tensor(vec![1.5f32, -2.5, 0.0, 3.14], vec![4])],
        );
    }

    // =========================================================================
    // Reduce ops
    // =========================================================================

    #[test]
    fn test_reduce_sum_axis0() {
        // [3, 4] → reduce sum axis 0 → [4]
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let axes = crate::milli_graph::ops::Constant::from_vec(g, vec![0i64], rng);
                let b = crate::milli_graph::ops::ReduceSum::push_new(
                    g,
                    a,
                    Some(axes),
                    false,
                    false,
                    rng,
                );
                (vec![a], vec![b])
            },
            vec![pool_tensor(
                (1..=12).map(|i| i as f32).collect(),
                vec![3, 4],
            )],
        );
    }

    #[test]
    fn test_reduce_sum_axis1() {
        // [3, 4] → reduce sum axis 1 → [3]
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let axes = crate::milli_graph::ops::Constant::from_vec(g, vec![1i64], rng);
                let b = crate::milli_graph::ops::ReduceSum::push_new(
                    g,
                    a,
                    Some(axes),
                    false,
                    false,
                    rng,
                );
                (vec![a], vec![b])
            },
            vec![pool_tensor(
                (1..=12).map(|i| i as f32).collect(),
                vec![3, 4],
            )],
        );
    }

    #[test]
    fn test_reduce_max_axis1() {
        // [3, 4] → reduce max axis 1 → [3]
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let axes = crate::milli_graph::ops::Constant::from_vec(g, vec![1i64], rng);
                let b = crate::milli_graph::ops::ReduceMax::push_new(
                    g,
                    a,
                    Some(axes),
                    false,
                    false,
                    rng,
                );
                (vec![a], vec![b])
            },
            vec![pool_tensor(
                vec![
                    3.0f32, 1.0, 4.0, 1.5, 9.0, 2.6, 5.3, 5.8, 9.7, 9.3, 2.3, 8.4,
                ],
                vec![3, 4],
            )],
        );
    }

    #[test]
    fn test_reduce_mean_axis1() {
        // [3, 4] → reduce mean axis 1 → [3]
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let axes = crate::milli_graph::ops::Constant::from_vec(g, vec![1i64], rng);
                let b = crate::milli_graph::ops::ReduceMean::push_new(
                    g,
                    a,
                    Some(axes),
                    false,
                    false,
                    rng,
                );
                (vec![a], vec![b])
            },
            vec![pool_tensor(
                (1..=12).map(|i| i as f32).collect(),
                vec![3, 4],
            )],
        );
    }

    // =========================================================================
    // Batched MatMul
    // =========================================================================

    #[test]
    fn test_matmul_batched() {
        // [2, 3, 4] @ [4, 5] → [2, 3, 5]
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let b = g.add_input(rng);
                let c = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    g,
                    a,
                    b,
                    NumericDType::F32,
                    rng,
                );
                (vec![a, b], vec![c])
            },
            vec![
                pool_tensor((0..24).map(|i| (i as f32) * 0.1).collect(), vec![2, 3, 4]),
                pool_tensor((0..20).map(|i| (i as f32) * 0.1).collect(), vec![4, 5]),
            ],
        );
    }

    // =========================================================================
    // Transpose then compute (exercises non-row-major strides propagation)
    // =========================================================================

    #[test]
    fn test_transpose_then_add() {
        // [3,4] transpose to [4,3], then add a [3] bias
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let bias = g.add_input(rng);
                let t = crate::milli_graph::ops::Transpose::push_new(g, a, Some(vec![1, 0]), rng);
                let out = crate::milli_graph::ops::SimpleBinary::add(g, t, bias, rng);
                (vec![a, bias], vec![out])
            },
            vec![
                pool_tensor((0..12).map(|i| i as f32).collect(), vec![3, 4]),
                pool_tensor(vec![100.0f32, 200.0, 300.0], vec![3]),
            ],
        );
    }

    #[test]
    fn test_transpose_then_matmul() {
        // A [3,4] transposed to [4,3], then matmul with B [3,2] → [4,2]
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let b = g.add_input(rng);
                let at = crate::milli_graph::ops::Transpose::push_new(g, a, Some(vec![1, 0]), rng);
                let c = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    g,
                    at,
                    b,
                    NumericDType::F32,
                    rng,
                );
                (vec![a, b], vec![c])
            },
            vec![
                pool_tensor((0..12).map(|i| (i as f32) * 0.1).collect(), vec![3, 4]),
                pool_tensor((0..6).map(|i| (i as f32) * 0.1).collect(), vec![3, 2]),
            ],
        );
    }

    // =========================================================================
    // ConstantOfShape
    // =========================================================================

    #[test]
    fn test_constant_of_shape() {
        use crate::numeric_scalar::NumericScalar;
        // ConstantOfShape(shape=[2,3], val=7.0) → [2,3] of 7s.
        // Then add with an input to force nano eval of the constant.
        check_integrity(
            |g, rng| {
                let shape = g.add_input(rng);
                let x = g.add_input(rng);
                let constant = crate::milli_graph::ops::ConstantOfShape::push_new(
                    g,
                    NumericScalar::from_f32(7.0),
                    shape,
                    rng,
                );
                let out = crate::milli_graph::ops::SimpleBinary::add(g, x, constant, rng);
                (vec![shape, x], vec![out])
            },
            vec![
                pool_tensor(vec![2i64, 3], vec![2]),
                pool_tensor(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
            ],
        );
    }

    // =========================================================================
    // Slice with steps
    // =========================================================================

    #[test]
    fn test_slice_with_step() {
        // [8] slice [0:8:2] → [4]
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let starts = g.add_input(rng);
                let ends = g.add_input(rng);
                let axes = g.add_input(rng);
                let steps = g.add_input(rng);
                let b = crate::milli_graph::ops::Slice::push_new(
                    g,
                    a,
                    starts,
                    ends,
                    Some(steps),
                    Some(axes),
                    rng,
                );
                (vec![a, starts, ends, axes, steps], vec![b])
            },
            vec![
                pool_tensor((0..8).map(|i| (i as f32) * 10.0).collect(), vec![8]),
                pool_tensor(vec![0i64], vec![1]),
                pool_tensor(vec![8i64], vec![1]),
                pool_tensor(vec![0i64], vec![1]),
                pool_tensor(vec![2i64], vec![1]),
            ],
        );
    }

    // =========================================================================
    // Concat then compute
    // =========================================================================

    #[test]
    fn test_concat_then_matmul() {
        // concat([2,3], [2,3], axis=0) → [4,3], then matmul with [3,2]
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let b = g.add_input(rng);
                let w = g.add_input(rng);
                let cat = crate::milli_graph::ops::Concat::push_new(g, vec![a, b], 0, rng);
                let out = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    g,
                    cat,
                    w,
                    NumericDType::F32,
                    rng,
                );
                (vec![a, b, w], vec![out])
            },
            vec![
                pool_tensor((0..6).map(|i| (i as f32) * 0.1).collect(), vec![2, 3]),
                pool_tensor((6..12).map(|i| (i as f32) * 0.1).collect(), vec![2, 3]),
                pool_tensor((0..6).map(|i| (i as f32) * 0.1).collect(), vec![3, 2]),
            ],
        );
    }

    // =========================================================================
    // GPT-2 pattern tests — scaling toward full model coverage
    // =========================================================================

    #[test]
    fn test_gather_2d_indices() {
        // Embedding lookup: data=[10, 4], indices=[2, 3] → [2, 3, 4]
        check_integrity(
            |g, rng| {
                let data = g.add_input(rng);
                let indices = g.add_input(rng);
                let out = crate::milli_graph::ops::Gather::push_new(g, data, indices, 0, rng);
                (vec![data, indices], vec![out])
            },
            vec![
                pool_tensor((0..40).map(|i| (i as f32) * 0.1).collect(), vec![10, 4]),
                pool_tensor(vec![0i64, 3, 7, 1, 5, 9], vec![2, 3]),
            ],
        );
    }

    #[test]
    fn test_gather_then_add() {
        // Embedding lookup + position embedding add: typical GPT-2 input
        // token_embed[10, 8] gathered by indices[2, 3] → [2, 3, 8]
        // pos_embed[3, 8] broadcast-added → [2, 3, 8]
        check_integrity(
            |g, rng| {
                let tok_data = g.add_input(rng);
                let pos_data = g.add_input(rng);
                let indices = g.add_input(rng);
                let tok_emb =
                    crate::milli_graph::ops::Gather::push_new(g, tok_data, indices, 0, rng);
                let out = crate::milli_graph::ops::SimpleBinary::add(g, tok_emb, pos_data, rng);
                (vec![tok_data, pos_data, indices], vec![out])
            },
            vec![
                pool_tensor((0..80).map(|i| (i as f32) * 0.01).collect(), vec![10, 8]),
                pool_tensor((0..24).map(|i| (i as f32) * 0.1).collect(), vec![3, 8]),
                pool_tensor(vec![0i64, 3, 7, 1, 5, 9], vec![2, 3]),
            ],
        );
    }

    #[test]
    fn test_reshape_split_heads() {
        // GPT-2 head split: [2, 4, 12] → reshape to [2, 4, 3, 4]
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let shape = g.add_input(rng);
                let b = crate::milli_graph::ops::Reshape::push_new(g, a, shape, false, rng);
                (vec![a, shape], vec![b])
            },
            vec![
                pool_tensor((0..96).map(|i| (i as f32) * 0.01).collect(), vec![2, 4, 12]),
                pool_tensor(vec![2i64, 4, 3, 4], vec![4]),
            ],
        );
    }

    #[test]
    fn test_transpose_4d_attention() {
        // GPT-2 attention head reorder: [B, S, H, D] → [B, H, S, D]
        // perm = [0, 2, 1, 3]
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let b =
                    crate::milli_graph::ops::Transpose::push_new(g, a, Some(vec![0, 2, 1, 3]), rng);
                (vec![a], vec![b])
            },
            vec![pool_tensor(
                (0..48).map(|i| (i as f32) * 0.1).collect(),
                vec![2, 3, 2, 4], // [B=2, S=3, H=2, D=4]
            )],
        );
    }

    #[test]
    fn test_matmul_4d_batched() {
        // Attention scores: [B, H, S, D] @ [B, H, D, S] → [B, H, S, S]
        // B=1, H=2, S=3, D=4
        check_integrity(
            |g, rng| {
                let q = g.add_input(rng);
                let k = g.add_input(rng);
                let c = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    g,
                    q,
                    k,
                    NumericDType::F32,
                    rng,
                );
                (vec![q, k], vec![c])
            },
            vec![
                pool_tensor(
                    (0..24).map(|i| (i as f32) * 0.1).collect(),
                    vec![1, 2, 3, 4], // Q: [B, H, S, D]
                ),
                pool_tensor(
                    (0..24).map(|i| (i as f32) * 0.1).collect(),
                    vec![1, 2, 4, 3], // K^T: [B, H, D, S]
                ),
            ],
        );
    }

    #[test]
    fn test_reshape_transpose_matmul_chain() {
        // Full head split + transpose + attention score pattern:
        // Q [2, 3, 8] → reshape [2, 3, 2, 4] → transpose [2, 2, 3, 4] → matmul with K^T [2, 2, 4, 3]
        check_integrity(
            |g, rng| {
                let q_flat = g.add_input(rng);
                let q_shape = g.add_input(rng);
                let k_t = g.add_input(rng);
                let q_4d =
                    crate::milli_graph::ops::Reshape::push_new(g, q_flat, q_shape, false, rng);
                let q_perm = crate::milli_graph::ops::Transpose::push_new(
                    g,
                    q_4d,
                    Some(vec![0, 2, 1, 3]),
                    rng,
                );
                let scores = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    g,
                    q_perm,
                    k_t,
                    NumericDType::F32,
                    rng,
                );
                (vec![q_flat, q_shape, k_t], vec![scores])
            },
            vec![
                pool_tensor(
                    (0..48).map(|i| (i as f32) * 0.01).collect(),
                    vec![2, 3, 8], // Q: [B=2, S=3, D=8]
                ),
                pool_tensor(vec![2i64, 3, 2, 4], vec![4]),
                pool_tensor(
                    (0..48).map(|i| (i as f32) * 0.01).collect(),
                    vec![2, 2, 4, 3], // K^T: [B=2, H=2, D=4, S=3]
                ),
            ],
        );
    }

    #[test]
    fn test_softmax_pattern() {
        // Softmax: ReduceMax → Sub → Exp → ReduceSum → Div
        // input [2, 4]
        check_integrity(
            |g, rng| {
                let x = g.add_input(rng);
                let axes = crate::milli_graph::ops::Constant::from_vec(g, vec![1i64], rng);
                let axes2 = crate::milli_graph::ops::Constant::from_vec(g, vec![1i64], rng);

                // max_val = ReduceMax(x, axis=1, keepdims=true)
                let max_val = crate::milli_graph::ops::ReduceMax::push_new(
                    g,
                    x,
                    Some(axes),
                    true,
                    false,
                    rng,
                );
                // shifted = x - max_val
                let shifted = crate::milli_graph::ops::SimpleBinary::sub(g, x, max_val, rng);
                // exp_val = Exp(shifted)
                let exp_val = crate::milli_graph::ops::SimpleUnaryOp::exp(g, shifted, rng);
                // sum_exp = ReduceSum(exp_val, axis=1, keepdims=true)
                let sum_exp = crate::milli_graph::ops::ReduceSum::push_new(
                    g,
                    exp_val,
                    Some(axes2),
                    true,
                    false,
                    rng,
                );
                // result = exp_val / sum_exp
                let result = crate::milli_graph::ops::SimpleBinary::div(g, exp_val, sum_exp, rng);

                (vec![x], vec![result])
            },
            vec![pool_tensor(
                vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 1.0, 0.5, 2.0],
                vec![2, 4],
            )],
        );
    }

    #[test]
    fn test_layernorm_pattern() {
        // LayerNorm: ReduceMean → Sub → Pow(2) → ReduceMean → Add(eps) → Sqrt → Div → Mul(gamma) → Add(beta)
        // input [2, 4], gamma [4], beta [4]
        check_integrity(
            |g, rng| {
                let x = g.add_input(rng);
                let gamma = g.add_input(rng);
                let beta = g.add_input(rng);

                let axes1 = crate::milli_graph::ops::Constant::from_vec(g, vec![-1i64], rng);
                let axes2 = crate::milli_graph::ops::Constant::from_vec(g, vec![-1i64], rng);
                let pow2 = crate::milli_graph::ops::Constant::from_vec(g, vec![2.0f32], rng);
                let eps = crate::milli_graph::ops::Constant::from_vec(g, vec![1e-5f32], rng);

                let mean = crate::milli_graph::ops::ReduceMean::push_new(
                    g,
                    x,
                    Some(axes1),
                    true,
                    false,
                    rng,
                );
                let centered = crate::milli_graph::ops::SimpleBinary::sub(g, x, mean, rng);
                let squared = crate::milli_graph::ops::Pow::push_new(g, centered, pow2, rng);
                let var = crate::milli_graph::ops::ReduceMean::push_new(
                    g,
                    squared,
                    Some(axes2),
                    true,
                    false,
                    rng,
                );
                let var_eps = crate::milli_graph::ops::SimpleBinary::add(g, var, eps, rng);
                let std_dev = crate::milli_graph::ops::SimpleUnaryOp::sqrt(g, var_eps, rng);
                let normed = crate::milli_graph::ops::SimpleBinary::div(g, centered, std_dev, rng);
                let scaled = crate::milli_graph::ops::SimpleBinary::mul(g, normed, gamma, rng);
                let out = crate::milli_graph::ops::SimpleBinary::add(g, scaled, beta, rng);

                (vec![x, gamma, beta], vec![out])
            },
            vec![
                pool_tensor(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 1.0, 0.5, 2.0], vec![2, 4]),
                pool_tensor(vec![1.0f32, 1.0, 1.0, 1.0], vec![4]),
                pool_tensor(vec![0.0f32, 0.0, 0.0, 0.0], vec![4]),
            ],
        );
    }

    #[test]
    fn test_double_transpose_matmul() {
        // Transpose twice then matmul — isolates the K^T double-transpose pattern.
        // K [1, 3, 4] → reshape [1, 3, 2, 2] → transpose [0,2,1,3] → [1, 2, 3, 2]
        //   → transpose [0,1,3,2] → [1, 2, 2, 3]
        // Q [1, 2, 3, 2] (direct input)
        // scores = Q @ K^T = [1, 2, 3, 3]

        check_integrity(
            |g, rng| {
                let k_flat = g.add_input(rng);
                let k_shape = g.add_input(rng);
                let q = g.add_input(rng);

                let k4 = crate::milli_graph::ops::Reshape::push_new(g, k_flat, k_shape, false, rng);
                let kt1 = crate::milli_graph::ops::Transpose::push_new(
                    g,
                    k4,
                    Some(vec![0, 2, 1, 3]),
                    rng,
                );
                let kt2 = crate::milli_graph::ops::Transpose::push_new(
                    g,
                    kt1,
                    Some(vec![0, 1, 3, 2]),
                    rng,
                );
                let scores = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    g,
                    q,
                    kt2,
                    NumericDType::F32,
                    rng,
                );

                (vec![k_flat, k_shape, q], vec![scores])
            },
            vec![
                // K [1, 3, 4]
                pool_tensor((0..12).map(|i| (i as f32) * 0.1).collect(), vec![1, 3, 4]),
                // shape for [1, 3, 2, 2]
                pool_tensor(vec![1i64, 3, 2, 2], vec![4]),
                // Q [1, 2, 3, 2]
                pool_tensor(
                    (0..12).map(|i| (i as f32) * 0.1).collect(),
                    vec![1, 2, 3, 2],
                ),
            ],
        );
    }

    #[test]
    fn test_attention_block_small() {
        // Minimal attention: Q/K/V projections → head split → attention scores → output
        // B=1, S=3, D=4, H=2, D_h=2
        check_integrity(
            |g, rng| {
                let x = g.add_input(rng); // [1, 3, 4]
                let wq = g.add_input(rng); // [4, 4]
                let wk = g.add_input(rng); // [4, 4]
                let wv = g.add_input(rng); // [4, 4]

                // Q = x @ Wq, K = x @ Wk, V = x @ Wv  → all [1, 3, 4]
                let q = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    g,
                    x,
                    wq,
                    NumericDType::F32,
                    rng,
                );
                let k = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    g,
                    x,
                    wk,
                    NumericDType::F32,
                    rng,
                );
                let v = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    g,
                    x,
                    wv,
                    NumericDType::F32,
                    rng,
                );

                // Reshape to [1, 3, 2, 2] for head split
                let shape4d = g.add_input(rng);
                let q4 = crate::milli_graph::ops::Reshape::push_new(g, q, shape4d, false, rng);

                // We need separate shape tensors for K and V since tensor IDs
                // are consumed (each Reshape reads a different shape tensor).
                let shape4d_k = g.add_input(rng);
                let shape4d_v = g.add_input(rng);
                let k4 = crate::milli_graph::ops::Reshape::push_new(g, k, shape4d_k, false, rng);
                let v4 = crate::milli_graph::ops::Reshape::push_new(g, v, shape4d_v, false, rng);

                // Transpose to [1, 2, 3, 2]  (B, H, S, D_h)
                let qt = crate::milli_graph::ops::Transpose::push_new(
                    g,
                    q4,
                    Some(vec![0, 2, 1, 3]),
                    rng,
                );
                let kt = crate::milli_graph::ops::Transpose::push_new(
                    g,
                    k4,
                    Some(vec![0, 2, 1, 3]),
                    rng,
                );
                let vt = crate::milli_graph::ops::Transpose::push_new(
                    g,
                    v4,
                    Some(vec![0, 2, 1, 3]),
                    rng,
                );

                // K^T: [1, 2, 3, 2] → [1, 2, 2, 3]
                let kt_t = crate::milli_graph::ops::Transpose::push_new(
                    g,
                    kt,
                    Some(vec![0, 1, 3, 2]),
                    rng,
                );

                // Attention scores: Q @ K^T = [1, 2, 3, 2] @ [1, 2, 2, 3] → [1, 2, 3, 3]
                let scores = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    g,
                    qt,
                    kt_t,
                    NumericDType::F32,
                    rng,
                );

                // Attention output: scores @ V = [1, 2, 3, 3] @ [1, 2, 3, 2] → [1, 2, 3, 2]
                let attn_out = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    g,
                    scores,
                    vt,
                    NumericDType::F32,
                    rng,
                );

                (
                    vec![x, wq, wk, wv, shape4d, shape4d_k, shape4d_v],
                    vec![scores, attn_out],
                )
            },
            vec![
                // x [1, 3, 4]
                pool_tensor((0..12).map(|i| (i as f32) * 0.1).collect(), vec![1, 3, 4]),
                // Wq [4, 4]
                pool_tensor(
                    (0..16).map(|i| (i as f32) * 0.05 - 0.4).collect(),
                    vec![4, 4],
                ),
                // Wk [4, 4]
                pool_tensor(
                    (0..16).map(|i| (i as f32) * 0.03 + 0.1).collect(),
                    vec![4, 4],
                ),
                // Wv [4, 4]
                pool_tensor(
                    (0..16).map(|i| (i as f32) * 0.04 - 0.2).collect(),
                    vec![4, 4],
                ),
                // shape [1, 3, 2, 2] — three copies for Q, K, V
                pool_tensor(vec![1i64, 3, 2, 2], vec![4]),
                pool_tensor(vec![1i64, 3, 2, 2], vec![4]),
                pool_tensor(vec![1i64, 3, 2, 2], vec![4]),
            ],
        );
    }

    // =========================================================================
    // GPT-2 scale tests — exercising dimensions seen in the real model
    // =========================================================================

    #[test]
    fn test_reduce_mean_gpt2_scale() {
        // ReduceMean on [4, 768] axis=-1 with large values — GPT-2 LayerNorm scale.
        // This should be bit-perfect with sequential accumulation.
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let axes = crate::milli_graph::ops::Constant::from_vec(g, vec![-1i64], rng);
                let b = crate::milli_graph::ops::ReduceMean::push_new(
                    g,
                    a,
                    Some(axes),
                    true,
                    false,
                    rng,
                );
                (vec![a], vec![b])
            },
            vec![pool_tensor(
                (0..3072)
                    .map(|i| ((i as f32) * 7.3 - 1500.0).sin() * 5000.0)
                    .collect(),
                vec![4, 768],
            )],
        );
    }

    #[test]
    fn test_reduce_mean_large_axis() {
        // ReduceMean on [2, 4, 64] axis -1 → [2, 4, 1] (GPT-2 layernorm scale)
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let axes = crate::milli_graph::ops::Constant::from_vec(g, vec![-1i64], rng);
                let b = crate::milli_graph::ops::ReduceMean::push_new(
                    g,
                    a,
                    Some(axes),
                    true,
                    false,
                    rng,
                );
                (vec![a], vec![b])
            },
            vec![pool_tensor(
                (0..512).map(|i| (i as f32) * 0.01 - 2.56).collect(),
                vec![2, 4, 64],
            )],
        );
    }

    #[test]
    fn test_matmul_large_k() {
        // MatMul [2, 4, 64] @ [64, 64] → [2, 4, 64] — GPT-2 projection scale
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let b = g.add_input(rng);
                let c = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    g,
                    a,
                    b,
                    NumericDType::F32,
                    rng,
                );
                (vec![a, b], vec![c])
            },
            vec![
                pool_tensor(
                    (0..512).map(|i| (i as f32) * 0.01 - 2.56).collect(),
                    vec![2, 4, 64],
                ),
                pool_tensor(
                    (0..4096).map(|i| (i as f32) * 0.001 - 2.048).collect(),
                    vec![64, 64],
                ),
            ],
        );
    }

    #[test]
    fn test_reshape_transpose_matmul_gpt2_scale() {
        // Full attention head split at GPT-2 scale:
        // Q [2, 4, 64] → reshape [2, 4, 4, 16] → transpose [0,2,1,3] → [2, 4, 4, 16]
        // matmul with K^T [2, 4, 16, 4] → scores [2, 4, 4, 4]
        check_integrity(
            |g, rng| {
                let q_flat = g.add_input(rng);
                let q_shape = g.add_input(rng);
                let k_flat = g.add_input(rng);
                let k_shape = g.add_input(rng);

                // Reshape Q: [2, 4, 64] → [2, 4, 4, 16]
                let q_4d =
                    crate::milli_graph::ops::Reshape::push_new(g, q_flat, q_shape, false, rng);
                // Transpose Q: [2, 4, 4, 16] → [2, 4, 4, 16] perm=[0,2,1,3]
                let q_t = crate::milli_graph::ops::Transpose::push_new(
                    g,
                    q_4d,
                    Some(vec![0, 2, 1, 3]),
                    rng,
                );

                // Reshape K: [2, 4, 64] → [2, 4, 4, 16]
                let k_4d =
                    crate::milli_graph::ops::Reshape::push_new(g, k_flat, k_shape, false, rng);
                // Transpose K: → [2, 4, 4, 16] perm=[0,2,1,3]
                let k_t1 = crate::milli_graph::ops::Transpose::push_new(
                    g,
                    k_4d,
                    Some(vec![0, 2, 1, 3]),
                    rng,
                );
                // Transpose K^T: [2, 4, 4, 16] → [2, 4, 16, 4] perm=[0,1,3,2]
                let k_t2 = crate::milli_graph::ops::Transpose::push_new(
                    g,
                    k_t1,
                    Some(vec![0, 1, 3, 2]),
                    rng,
                );

                // Attention scores: Q @ K^T = [2, 4, 4, 16] @ [2, 4, 16, 4] → [2, 4, 4, 4]
                let scores = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    g,
                    q_t,
                    k_t2,
                    NumericDType::F32,
                    rng,
                );

                (vec![q_flat, q_shape, k_flat, k_shape], vec![scores])
            },
            vec![
                pool_tensor(
                    (0..512).map(|i| (i as f32) * 0.01 - 2.56).collect(),
                    vec![2, 4, 64],
                ),
                pool_tensor(vec![2i64, 4, 4, 16], vec![4]),
                pool_tensor(
                    (0..512).map(|i| (i as f32) * 0.005 - 1.28).collect(),
                    vec![2, 4, 64],
                ),
                pool_tensor(vec![2i64, 4, 4, 16], vec![4]),
            ],
        );
    }

    #[test]
    fn test_split_last_axis_3way() {
        // GPT-2 QKV split: [2, 4, 12] → split axis=2 into 3 × [2, 4, 4]
        // Then add each piece with a bias to force evaluation.
        check_integrity(
            |g, rng| {
                let x = g.add_input(rng);
                let bias_q = g.add_input(rng);
                let bias_k = g.add_input(rng);
                let bias_v = g.add_input(rng);

                let split_sizes =
                    crate::milli_graph::ops::Constant::from_vec(g, vec![4i64, 4, 4], rng);

                let q = crate::milli_graph::ops::Split::push_new(
                    g,
                    x,
                    Some(split_sizes),
                    2,
                    Some(3),
                    0,
                    rng,
                );
                let k = crate::milli_graph::ops::Split::push_new(
                    g,
                    x,
                    Some(split_sizes),
                    2,
                    Some(3),
                    1,
                    rng,
                );
                let v = crate::milli_graph::ops::Split::push_new(
                    g,
                    x,
                    Some(split_sizes),
                    2,
                    Some(3),
                    2,
                    rng,
                );

                let q_biased = crate::milli_graph::ops::SimpleBinary::add(g, q, bias_q, rng);
                let k_biased = crate::milli_graph::ops::SimpleBinary::add(g, k, bias_k, rng);
                let v_biased = crate::milli_graph::ops::SimpleBinary::add(g, v, bias_v, rng);

                (
                    vec![x, bias_q, bias_k, bias_v],
                    vec![q_biased, k_biased, v_biased],
                )
            },
            vec![
                pool_tensor((0..96).map(|i| (i as f32) * 0.1).collect(), vec![2, 4, 12]),
                pool_tensor(vec![1.0f32; 4], vec![4]),
                pool_tensor(vec![2.0f32; 4], vec![4]),
                pool_tensor(vec![3.0f32; 4], vec![4]),
            ],
        );
    }

    #[test]
    fn test_matmul_then_split() {
        // MatMul → Split pattern (GPT-2 QKV projection):
        // x [2, 3, 4] @ W [4, 6] → [2, 3, 6] → split axis=-1 into 3 × [2, 3, 2]
        check_integrity(
            |g, rng| {
                let x = g.add_input(rng);
                let w = g.add_input(rng);
                let proj = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    g,
                    x,
                    w,
                    NumericDType::F32,
                    rng,
                );

                let split_sizes =
                    crate::milli_graph::ops::Constant::from_vec(g, vec![2i64, 2, 2], rng);

                let q = crate::milli_graph::ops::Split::push_new(
                    g,
                    proj,
                    Some(split_sizes),
                    2,
                    Some(3),
                    0,
                    rng,
                );
                let k = crate::milli_graph::ops::Split::push_new(
                    g,
                    proj,
                    Some(split_sizes),
                    2,
                    Some(3),
                    1,
                    rng,
                );
                let v = crate::milli_graph::ops::Split::push_new(
                    g,
                    proj,
                    Some(split_sizes),
                    2,
                    Some(3),
                    2,
                    rng,
                );

                (vec![x, w], vec![q, k, v])
            },
            vec![
                pool_tensor((0..24).map(|i| (i as f32) * 0.1).collect(), vec![2, 3, 4]),
                pool_tensor(
                    (0..24).map(|i| (i as f32) * 0.05 - 0.6).collect(),
                    vec![4, 6],
                ),
            ],
        );
    }

    #[test]
    fn test_qkv_attention_full() {
        // Full QKV attention pattern (no softmax):
        // x [1, 4, 8] @ Wqkv [8, 24] → [1, 4, 24] → Split into Q,K,V [1, 4, 8]
        // → Reshape [1, 4, 2, 4] → Transpose [0,2,1,3] → [1, 2, 4, 4]
        // K: → Transpose [0,1,3,2] → [1, 2, 4, 4]
        // scores = Q @ K^T → [1, 2, 4, 4]
        // output = scores @ V → [1, 2, 4, 4]
        check_integrity(
            |g, rng| {
                let x = g.add_input(rng);
                let wqkv = g.add_input(rng);

                // QKV projection: [1, 4, 8] @ [8, 24] → [1, 4, 24]
                let proj = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    g,
                    x,
                    wqkv,
                    NumericDType::F32,
                    rng,
                );

                // Split into Q, K, V each [1, 4, 8]
                let split_sizes =
                    crate::milli_graph::ops::Constant::from_vec(g, vec![8i64, 8, 8], rng);

                let q_raw = crate::milli_graph::ops::Split::push_new(
                    g,
                    proj,
                    Some(split_sizes),
                    2,
                    Some(3),
                    0,
                    rng,
                );
                let k_raw = crate::milli_graph::ops::Split::push_new(
                    g,
                    proj,
                    Some(split_sizes),
                    2,
                    Some(3),
                    1,
                    rng,
                );
                let v_raw = crate::milli_graph::ops::Split::push_new(
                    g,
                    proj,
                    Some(split_sizes),
                    2,
                    Some(3),
                    2,
                    rng,
                );

                // Reshape to [1, 4, 2, 4] for head split
                let shape4d_q = g.add_input(rng);
                let shape4d_k = g.add_input(rng);
                let shape4d_v = g.add_input(rng);
                let q4 =
                    crate::milli_graph::ops::Reshape::push_new(g, q_raw, shape4d_q, false, rng);
                let k4 =
                    crate::milli_graph::ops::Reshape::push_new(g, k_raw, shape4d_k, false, rng);
                let v4 =
                    crate::milli_graph::ops::Reshape::push_new(g, v_raw, shape4d_v, false, rng);

                // Transpose to [1, 2, 4, 4] (B, H, S, D)
                let qt = crate::milli_graph::ops::Transpose::push_new(
                    g,
                    q4,
                    Some(vec![0, 2, 1, 3]),
                    rng,
                );
                let kt = crate::milli_graph::ops::Transpose::push_new(
                    g,
                    k4,
                    Some(vec![0, 2, 1, 3]),
                    rng,
                );
                let vt = crate::milli_graph::ops::Transpose::push_new(
                    g,
                    v4,
                    Some(vec![0, 2, 1, 3]),
                    rng,
                );

                // K^T: [1, 2, 4, 4] → [1, 2, 4, 4] (transpose last 2 dims)
                let kt_t = crate::milli_graph::ops::Transpose::push_new(
                    g,
                    kt,
                    Some(vec![0, 1, 3, 2]),
                    rng,
                );

                // Attention scores: Q @ K^T
                let scores = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    g,
                    qt,
                    kt_t,
                    NumericDType::F32,
                    rng,
                );

                // Attention output: scores @ V
                let attn_out = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    g,
                    scores,
                    vt,
                    NumericDType::F32,
                    rng,
                );

                (
                    vec![x, wqkv, shape4d_q, shape4d_k, shape4d_v],
                    vec![q_raw, k_raw, v_raw, scores, attn_out],
                )
            },
            vec![
                // x [1, 4, 8]
                pool_tensor(
                    (0..32).map(|i| (i as f32) * 0.1 - 1.6).collect(),
                    vec![1, 4, 8],
                ),
                // Wqkv [8, 24]
                pool_tensor(
                    (0..192).map(|i| (i as f32) * 0.01 - 0.96).collect(),
                    vec![8, 24],
                ),
                // shape tensors for Q, K, V reshape
                pool_tensor(vec![1i64, 4, 2, 4], vec![4]),
                pool_tensor(vec![1i64, 4, 2, 4], vec![4]),
                pool_tensor(vec![1i64, 4, 2, 4], vec![4]),
            ],
        );
    }
}
