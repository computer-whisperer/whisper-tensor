//! Lowering from MilliOpGraph to NanoGraph.
//!
//! Walks the MilliOpGraph in topological order, classifying each tensor's
//! dimensions as Known (expanded to atoms) or Symbolic (iteration params).
//! View ops dissolve into addressing changes. Elementwise ops become groups
//! with the same atom count. Ops that don't fit become boundary groups.

use std::collections::HashMap;

use crate::dtype::DType;
use crate::graph::{GlobalId, Graph, Node};
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::AnyMilliOp;
use crate::nano_graph::ops::ScalarOp;
use crate::nano_graph::pattern::{AtomId, InputRef, NanoGraph, SymDim};
use crate::numeric_scalar::NumericScalar;
use crate::tensor_info::TensorInfo;

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
pub struct LowerResult {
    pub graph: NanoGraph,
    /// Ops that could not be lowered (treated as boundary).
    pub unsupported: Vec<(GlobalId, String)>,
    /// Human-readable detail for each unsupported op (input/output shapes).
    pub unsupported_details: Vec<String>,
    /// Mapping from milli tensor GlobalId to nano atom group.
    pub tensor_map: HashMap<GlobalId, TensorAtomMapInfo>,
}

/// Public view of how a milli tensor maps to nano atoms.
#[derive(Debug, Clone)]
pub struct TensorAtomMapInfo {
    pub base_id: AtomId,
    pub count: u64,
    pub dtype: DType,
    pub sym_dims: Vec<SymDim>,
    pub known_strides: Vec<u64>,
    pub known_dims: Vec<u64>,
    /// Concat segments (empty for simple views).
    pub segments: Vec<(usize, u64, u64, AtomId, Vec<u64>)>, // (concat_dim, start, size, base_id, strides)
}

impl TensorAtomMapInfo {
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
    pub dtype: DType,
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
        dtype: DType,
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
        dtype: DType,
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
pub fn lower(
    graph: &MilliOpGraph,
    inputs: &HashMap<GlobalId, TensorInfo>,
) -> Result<LowerResult, LowerError> {
    let t0 = std::time::Instant::now();
    let all_infos = graph.infer_all(inputs)?;
    eprintln!(
        "  [lower] infer_all: {:.1}ms",
        t0.elapsed().as_secs_f64() * 1e3
    );
    let mut ctx = NanoLoweringContext::new(&all_infos);

    // Register all tensors that exist before ops run (graph inputs + inferred
    // constants) and are NOT produced by any op.
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
            ctx.register_input(*id, info);
        }
    }

    // Walk ops in topological order.
    let t1 = std::time::Instant::now();
    for &op_id in graph.op_ordering() {
        let Some(op) = graph.get_node_by_id(&op_id) else {
            continue;
        };
        ctx.lower_op(op);
    }
    eprintln!(
        "  [lower] ops: {:.1}ms ({} groups, {} atoms)",
        t1.elapsed().as_secs_f64() * 1e3,
        ctx.nano.num_groups(),
        ctx.nano.num_atoms()
    );

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
                for i in 0..tam.count {
                    ctx.nano.outputs.push(tam.atom_id_for_element(i));
                }
            }
        }
    }
    // If still no outputs, register all mapped tensors from the last ops.
    // This is a fallback — proper output mapping will be fixed later.

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
    })
}

/// Backward-compatible alias for `lower()`. Will be removed once all call
/// sites are migrated.
pub fn lower_with_info(
    graph: &MilliOpGraph,
    inputs: &HashMap<GlobalId, TensorInfo>,
) -> Result<LowerResult, LowerError> {
    lower(graph, inputs)
}

/// Context for lowering milli ops to nano graph.
///
/// Holds all mutable state needed during lowering. The `all_infos` field
/// is a shared reference to tensor shape information computed by inference,
/// so methods can freely read tensor info without borrowing self.
pub struct NanoLoweringContext<'a> {
    pub nano: NanoGraph,
    pub tensor_map: HashMap<GlobalId, TensorAtomMap>,
    pub all_infos: &'a HashMap<GlobalId, TensorInfo>,
    next_anon_sym: usize,
    pub unsupported: Vec<(GlobalId, String)>,
    pub unsupported_details: Vec<String>,
}

impl<'a> NanoLoweringContext<'a> {
    pub fn new(all_infos: &'a HashMap<GlobalId, TensorInfo>) -> Self {
        Self {
            nano: NanoGraph::new(),
            tensor_map: HashMap::new(),
            all_infos,
            next_anon_sym: 0,
            unsupported: Vec::new(),
            unsupported_details: Vec::new(),
        }
    }

    /// Classify tensor dims and return layout info.
    /// Returns None if rank is unknown or atom count overflows u32.
    pub fn classify_dims(&mut self, info: &TensorInfo) -> Option<DimClassification> {
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
    /// Extracts scalar values from the TensorInfo and creates Literal groups.
    /// Runs of identical values are coalesced into single groups. If the
    /// TensorInfo has no numeric data, falls back to `register_input`.
    pub fn register_constant(&mut self, id: GlobalId, info: &TensorInfo) {
        let Some(numeric) = info.as_numeric() else {
            self.register_input(id, info);
            return;
        };

        let Some((layout, known_dims, sym_dims, count)) = self.classify_dims(info) else {
            self.register_input(id, info);
            return;
        };

        let strides = TensorAtomMap::compute_strides(&known_dims);
        let count = count.max(1) as usize;
        let dt = info.dtype();

        // Extract flat scalar values from the tensor.
        let nd = numeric.to_ndarray().unwrap();
        let flat = nd.flatten();
        let n_elems = flat.num_elements();
        let scalars: Vec<NumericScalar> = (0..n_elems)
            .map(|i| flat.get(&[i as u64]).unwrap())
            .collect();

        if scalars.is_empty() {
            self.register_input(id, info);
            return;
        }

        let n = scalars.len().min(count);

        // Create Literal groups, coalescing runs of identical values.
        let mut base_id = None;
        let mut i = 0;
        while i < n {
            // Find run of identical values.
            let run_val = &scalars[i];
            let mut run_len = 1;
            while i + run_len < n && scalars[i + run_len] == *run_val {
                run_len += 1;
            }
            let gid = self.nano.push_group(
                run_len as u64,
                dt,
                ScalarOp::Literal(run_val.clone()),
                sym_dims.clone(),
                vec![],
            );
            if base_id.is_none() {
                base_id = Some(gid);
            }
            i += run_len;
        }

        self.tensor_map.insert(
            id,
            TensorAtomMap::simple(base_id.unwrap(), n as u64, dt, layout, strides, sym_dims),
        );
    }

    /// Register a graph input as leaf atoms.
    ///
    /// Creates Literal groups with placeholder value 0.0. For external inputs
    /// (user-provided tensors), the executor fills in actual values at runtime.
    /// For constant tensors whose values are known at lowering time, the
    /// Literal(0.0) placeholder is similarly overridden by the executor using
    /// the tensor data from `all_infos`.
    pub fn register_input(&mut self, id: GlobalId, info: &TensorInfo) {
        let Some((layout, known_dims, sym_dims, count)) = self.classify_dims(info) else {
            // Unknown rank — register a single atom.
            let dt = info.dtype();
            let base_id = self.nano.add_input_tensor(id, 1, dt);
            self.tensor_map.insert(
                id,
                TensorAtomMap::simple(base_id, 1, dt, vec![], vec![], vec![]),
            );
            return;
        };

        let strides = TensorAtomMap::compute_strides(&known_dims);
        let count = count.max(1);
        let dt = info.dtype();

        let base_id = self.nano.add_input_tensor(id, count, dt);

        self.tensor_map.insert(
            id,
            TensorAtomMap::simple(base_id, count, dt, layout, strides, sym_dims),
        );
    }

    /// Register a tensor as a boundary (opaque) group.
    /// Boundary atoms are leaves — they use Literal(0) with no inputs.
    pub fn register_boundary(&mut self, output_id: GlobalId, info: &TensorInfo, _op_kind: &str) {
        let dt = info.dtype();
        let Some((layout, known_dims, sym_dims, count)) = self.classify_dims(info) else {
            let base_id = self.nano.push_atom(
                dt,
                ScalarOp::Literal(NumericScalar::F32(0.0)),
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
            ScalarOp::Literal(NumericScalar::F32(0.0)),
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
        consumer_info: &TensorInfo,
        producer_info: &TensorInfo,
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
                return InputRef::Affine {
                    base: producer.base_id,
                    stride: 1,
                };
            }
        }

        // Scalar producer → broadcast.
        if producer.count == 1 {
            return InputRef::Broadcast(producer.base_id);
        }

        // General broadcast: build Explicit.
        self.build_broadcast_explicit(consumer, producer, consumer_info, producer_info)
    }

    /// Build an Explicit InputRef for broadcast patterns.
    fn build_broadcast_explicit(
        &self,
        consumer: &TensorAtomMap,
        producer: &TensorAtomMap,
        consumer_info: &TensorInfo,
        producer_info: &TensorInfo,
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
            return InputRef::Affine {
                base: ids[0],
                stride,
            };
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
                    return InputRef::StridedBroadcast {
                        base: ids[0],
                        stride: block_stride,
                        repeat,
                    };
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
                return InputRef::Modular {
                    base: ids[0],
                    stride: inner_stride,
                    modulus: period as u64,
                };
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
        consumer_info: &TensorInfo,
        producer_info: &TensorInfo,
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
        op.lower_to_nano(self);
    }

    /// Default lowering for unsupported ops: if all outputs are numeric
    /// (constant-folded), register as constants; otherwise register as boundary.
    pub fn lower_default(&mut self, op: &AnyMilliOp) {
        let all_infos = self.all_infos;
        let op_kind = op.op_kind();
        let all_numeric = op.outputs().all(|out_id| {
            all_infos
                .get(&out_id)
                .is_some_and(|i| i.as_numeric().is_some())
        });
        for out_id in op.outputs() {
            if let Some(info) = all_infos.get(&out_id) {
                if all_numeric {
                    self.register_constant(out_id, info);
                } else {
                    self.register_boundary(out_id, info, &op_kind);
                }
            } else {
                self.register_opaque(out_id);
            }
        }
        if !all_numeric {
            self.push_unsupported(op, &op_kind);
        }
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
            InputRef::Affine {
                base: in_map.base_id,
                stride: 1,
            }
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
    ) {
        let all_infos = self.all_infos;
        let in_id = Node::inputs(op).next().unwrap();
        let out_id = Node::outputs(op).next().unwrap();

        let Some(in_map) = self.tensor_map.get(&in_id).cloned() else {
            self.lower_as_boundary_named(op, "ViewOp");
            return;
        };
        let Some(out_info) = all_infos.get(&out_id) else {
            self.lower_as_boundary_named(op, "ViewOp");
            return;
        };

        let in_dt = all_infos.get(&in_id).map(|i| i.dtype());
        let out_dt = out_info.dtype();

        // If dtypes match, this is a no-op — just re-register the tensor.
        if in_dt == Some(out_dt) {
            self.tensor_map.insert(out_id, in_map);
            return;
        }

        // Dtype differs — emit an Identity group for the cast.
        let known_dims = in_map.known_dims();
        let input_ref = Self::pointwise_input_ref(&in_map);

        let base_id = self.nano.push_group(
            in_map.count,
            out_dt,
            ScalarOp::Identity,
            in_map.sym_dims.clone(),
            vec![input_ref],
        );

        // The output has freshly allocated atoms in row-major order.
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
    }


    /// View op: no compute, just re-register with the new shape.
    pub fn lower_view_op<T: Node>(&mut self, op: &T) {
        let all_infos = self.all_infos;
        let in_id = Node::inputs(op).next().unwrap();
        let out_id = Node::outputs(op).next().unwrap();

        let Some(in_map) = self.tensor_map.get(&in_id).cloned() else {
            self.lower_as_boundary_named(op, "ViewOp");
            return;
        };
        let Some(out_info) = all_infos.get(&out_id) else {
            self.register_opaque(out_id);
            return;
        };

        let Some((layout, known_dims, sym_dims, count)) = self.classify_dims(out_info) else {
            self.lower_as_boundary_named(op, "ViewOp");
            return;
        };
        let count = count.max(1);

        if count == in_map.count {
            // Atom-count preserving: just re-register with new layout.
            self.tensor_map.insert(
                out_id,
                TensorAtomMap::simple(
                    in_map.base_id,
                    count,
                    out_info.dtype(),
                    layout,
                    TensorAtomMap::compute_strides(&known_dims),
                    sym_dims,
                ),
            );
        } else {
            self.register_boundary(out_id, out_info, "ViewOp");
            let name = format!("ViewOp(count {} → {})", in_map.count, count);
            self.push_unsupported(op, &name);
        }
    }








    /// Extract concrete i64 values from a tensor in all_infos.
    pub fn extract_i64(all_infos: &HashMap<GlobalId, TensorInfo>, id: &GlobalId) -> Option<Vec<i64>> {
        let info = all_infos.get(id)?;
        let tensor = info.as_numeric()?;
        let as_i64 = tensor
            .cast(
                DType::I64,
                &mut crate::backends::eval_backend::EvalBackend::NDArray,
            )
            .ok()?;
        let rank1 = as_i64.try_to_rank::<typenum::P1>().ok()?;
        Vec::<i64>::try_from(rank1.to_ndarray().ok()?).ok()
    }


    /// Lower ReduceSum or ReduceMax over known axes.
    pub fn lower_reduce<R, F>(
        &mut self,
        reduce: &R,
        make_reduce_op: F,
    ) where
        R: Node,
        R: ReduceAccessors,
        F: Fn(DType, u64, i64) -> ScalarOp,
    {
        let all_infos = self.all_infos;
        let in_id = Node::inputs(reduce).next().unwrap();
        let out_id = Node::outputs(reduce).next().unwrap();

        let Some(in_map) = self.tensor_map.get(&in_id).cloned() else {
            self.lower_as_boundary_named(reduce, "Reduce");
            return;
        };
        let Some(out_info) = all_infos.get(&out_id) else {
            self.lower_as_boundary_named(reduce, "Reduce");
            return;
        };

        // Get the concrete reduction axes.
        let axes: Vec<i64> = if let Some(axes_id) = reduce.axes_tensor() {
            let Some(vals) = Self::extract_i64(all_infos, &axes_id) else {
                self.lower_as_boundary_named(reduce, "Reduce");
                return;
            };
            vals
        } else if reduce.noop_with_empty_axes() {
            // No axes + noop = identity.
            let Some((layout, known_dims, sym_dims, count)) = self.classify_dims(out_info) else {
                self.lower_as_boundary_named(reduce, "Reduce");
                return;
            };
            let count = count.max(1);
            let dt = out_info.dtype();
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
            return;
        } else {
            // No axes = reduce all. Boundary for now.
            self.lower_as_boundary_named(reduce, "Reduce");
            return;
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
                self.lower_as_boundary_named(reduce, "Reduce");
                return;
            }
            let Some(ki) = axis_to_known_idx[ax] else {
                // Reducing a symbolic dim — use the symbolic reduce path.
                // For now, boundary.
                self.lower_as_boundary_named(reduce, "Reduce");
                return;
            };
            reduce_known_indices.push(ki);
        }

        // Compute the reduction extent (product of reduced known dims).
        let reduce_extent: u64 = reduce_known_indices
            .iter()
            .map(|&ki| in_known[ki])
            .product();
        if reduce_extent == 0 {
            self.lower_as_boundary_named(reduce, "Reduce");
            return;
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
            self.lower_as_boundary_named(reduce, "Reduce");
            return;
        }

        let rki = reduce_known_indices[0];
        // Use the input's actual strides (may be non-row-major after Transpose).
        let in_strides = &in_map.known_strides;
        let reduce_stride = in_strides[rki] as i64;

        // Build output known strides.
        let out_strides_local = TensorAtomMap::compute_strides(&out_known);

        // For each output atom, compute the base input atom.
        // The output atom at flat index `f` maps to input indices where the
        // reduced dim is 0. We need to compute the input flat index with
        // the reduced dim set to 0.
        // Build Explicit mapping: output flat → input flat (at k=0).
        // For each output atom, decompose into non-reduced dims, then compute
        // the input flat index (with reduced dim = 0).
        let mut base_ids = Vec::with_capacity(out_count as usize);
        for flat_out in 0..out_count {
            // Decompose flat_out into output known-dim indices.
            let mut out_indices = vec![0u64; out_known.len()];
            let mut rem = flat_out;
            for (i, &stride) in out_strides_local.iter().enumerate() {
                if stride > 0 {
                    out_indices[i] = rem / stride;
                    rem %= stride;
                }
            }

            // Map back to input known-dim indices (insert 0 for reduced dim).
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

        // Check if the base_ids form a simple affine pattern.
        let is_affine = if out_count <= 1 {
            true
        } else {
            let stride = base_ids[1] as i64 - base_ids[0] as i64;
            base_ids
                .windows(2)
                .all(|w| (w[1] as i64 - w[0] as i64) == stride)
        };

        // Build the input ref: Affine addressing for the base (at k=0),
        // with reduce_count and reduce_stride encoded in the op itself.
        let input_ref = if is_affine && out_count > 0 {
            let stride_i = if out_count > 1 {
                base_ids[1] as i64 - base_ids[0] as i64
            } else {
                1
            };
            InputRef::Affine {
                base: in_map.base_id.offset(base_ids[0]),
                stride: stride_i,
            }
        } else if out_count > 0 {
            InputRef::Explicit(
                base_ids
                    .iter()
                    .map(|&offset| in_map.base_id.offset(offset))
                    .collect(),
            )
        } else {
            self.lower_as_boundary_named(reduce, "Reduce");
            return;
        };

        // Determine compute dtype: ReduceSum upcasts BF16/F16 → F32 to match
        // milli eval precision semantics (see reduce_sum.rs lines 193-196).
        let out_dt = out_info.dtype();
        let in_dt = all_infos.get(&in_id).map(|i| i.dtype()).unwrap_or(out_dt);
        let compute_dt = match in_dt {
            DType::BF16 | DType::F16 => DType::F32,
            other => other,
        };

        // Classify output for proper layout.
        let Some((out_layout, out_known_dims_full, out_sym_dims, _)) = self.classify_dims(out_info)
        else {
            // This shouldn't happen since we computed out_count, but be safe.
            self.lower_as_boundary_named(reduce, "Reduce");
            return;
        };

        // ReduceSum/ReduceMax group: the op carries reduce_count and reduce_stride,
        // and the input uses Affine addressing (base at k=0).
        // The evaluator loops k=0..reduce_count, reading at base + stride*i + k*reduce_stride.
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
    }



    /// Generic boundary fallback. Always registers outputs in tensor_map
    /// so downstream ops can reference them.
    pub fn lower_as_boundary_named<T: Node>(
        &mut self,
        op: &T,
        name: &str,
    ) {
        let all_infos = self.all_infos;
        let _op_id = op.global_id();
        for out_id in op.outputs() {
            if let Some(info) = all_infos.get(&out_id) {
                self.register_boundary(out_id, info, name);
            } else {
                // No info available — register a minimal opaque atom so downstream
                // ops always find this tensor in tensor_map.
                self.register_opaque(out_id);
            }
        }
        self.push_unsupported(op, name);
    }

    pub fn push_unsupported<T: Node>(
        &mut self,
        op: &T,
        name: &str,
    ) {
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

    pub fn fmt_info(info: Option<&TensorInfo>) -> String {
        let Some(info) = info else {
            return "?".to_string();
        };
        let r = info.rank_if_known().unwrap_or(0);
        let prefix = if info.as_numeric().is_some() {
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
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
        );
        self.tensor_map.insert(
            id,
            TensorAtomMap::simple(base_id, 1, DType::F32, vec![], vec![], vec![]),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DynRank;
    use crate::backends::eval_backend::EvalBackend;
    use crate::graph::Graph;
    use crate::milli_graph::MilliOpGraph;
    use crate::milli_graph::ops::MilliOp;
    use crate::backends::ndarray_backend::NDArrayNumericTensor;
    use crate::nano_graph::eval;
    use crate::nano_graph::pattern::AtomRange;
    use crate::numeric_tensor::NumericTensor;

    /// Extract flat values from a NumericTensor, returned as f64.
    fn tensor_to_f64(t: &NumericTensor<DynRank>) -> Vec<f64> {
        let mut backend = EvalBackend::NDArray;
        // BOOL can't cast to F32 — handle separately.
        if t.dtype() == crate::dtype::DType::BOOL {
            let nd = t.to_ndarray().unwrap();
            match nd {
                NDArrayNumericTensor::BOOL(a) => {
                    return a.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect()
                }
                _ => unreachable!(),
            }
        }
        let f32_tensor = t.cast(crate::dtype::DType::F32, &mut backend).unwrap();
        let flat = f32_tensor.flatten().unwrap();
        let v: Vec<f32> = flat.to_ndarray().unwrap().try_into().unwrap();
        v.into_iter().map(|x| x as f64).collect()
    }

    /// Convert a NumericTensor to an NDArrayNumericTensor for eval input.
    fn to_ndarray(t: &NumericTensor<DynRank>) -> NDArrayNumericTensor<DynRank> {
        t.to_ndarray().unwrap()
    }

    /// Build a milli graph, eval through both milli and nano, compare results.
    fn check_integrity(
        build_graph: impl FnOnce(
            &mut MilliOpGraph,
            &mut rand::rngs::ThreadRng,
        ) -> (Vec<GlobalId>, Vec<GlobalId>),
        inputs: Vec<NumericTensor<DynRank>>,
    ) {
        let mut rng = rand::rng();
        let (mut milli, _ext_map) = MilliOpGraph::new(std::iter::empty(), &mut rng);

        let (input_ids, output_ids) = build_graph(&mut milli, &mut rng);
        assert_eq!(input_ids.len(), inputs.len());

        // Prepare inputs.
        let mut info_inputs: HashMap<GlobalId, TensorInfo> = HashMap::new();
        let mut intermediates: HashMap<GlobalId, NumericTensor<DynRank>> = HashMap::new();
        for (id, tensor) in input_ids.iter().zip(inputs.iter()) {
            info_inputs.insert(*id, TensorInfo::from(tensor.clone()));
            intermediates.insert(*id, tensor.clone());
        }

        // Eval through MilliOpGraph.
        let mut backend = EvalBackend::NDArray;
        for &op_id in milli.op_ordering() {
            let op = milli.get_node_by_id(&op_id).unwrap();
            for (tid, val) in op.eval(&intermediates, &mut backend).unwrap() {
                intermediates.insert(tid, val);
            }
        }

        // Lower to NanoGraph.
        let result = lower_with_info(&milli, &info_inputs).unwrap();
        assert!(
            result.unsupported.is_empty(),
            "Unsupported ops: {:?}",
            result.unsupported_details
        );

        // Build eval inputs from input_tensors (the graph knows where each tensor lives).
        let nd_inputs: Vec<NDArrayNumericTensor<DynRank>> = input_ids
            .iter()
            .zip(inputs.iter())
            .map(|(_, t)| to_ndarray(t))
            .collect();
        let eval_inputs: Vec<(AtomId, &NDArrayNumericTensor<DynRank>)> = result
            .graph
            .input_tensors()
            .iter()
            .filter_map(|it| {
                let idx = input_ids.iter().position(|&id| id == it.tensor_id)?;
                Some((it.base_id, &nd_inputs[idx]))
            })
            .collect();

        // Build output ranges per group (not per tensor_map entry, which may be segmented).
        // Collect all groups that contain output atoms.
        let mut all_output_ranges: Vec<AtomRange> = Vec::new();
        for out_id in &output_ids {
            let tam = result.tensor_map.get(out_id).unwrap();
            assert!(tam.sym_dims.is_empty(), "Sym dims not yet supported in test");
            // For segmented tensors, we need all groups that contain atoms.
            // Collect unique groups by walking atom_id_for_element.
            let mut seen_groups = std::collections::HashSet::new();
            for i in 0..tam.count {
                let atom = tam.atom_id_for_element(i);
                if let Some(gi) = result.graph.find_group_idx(atom) {
                    if seen_groups.insert(gi) {
                        let g = &result.graph.groups()[gi];
                        all_output_ranges.push(AtomRange {
                            base: g.base_id,
                            count: g.count,
                            dtype: g.output_dtype,
                        });
                    }
                }
                // Also check input tensor ranges.
                if let Some((ti, _)) = result.graph.find_input_idx(atom) {
                    let it = &result.graph.input_tensors()[ti];
                    let fake_gi = usize::MAX - ti;
                    if seen_groups.insert(fake_gi) {
                        all_output_ranges.push(AtomRange {
                            base: it.base_id,
                            count: it.count,
                            dtype: it.dtype,
                        });
                    }
                }
            }
        }

        // Eval NanoGraph.
        let nano_results = eval::eval(&result.graph, &eval_inputs, &all_output_ranges);

        // Build a lookup from AtomId -> f64.
        let mut atom_vals: std::collections::HashMap<u64, f64> = std::collections::HashMap::new();
        for (range, tensor) in all_output_ranges.iter().zip(nano_results.iter()) {
            let flat: Vec<f64> = match tensor {
                NDArrayNumericTensor::F32(a) => a.iter().map(|&v| v as f64).collect(),
                NDArrayNumericTensor::F64(a) => a.iter().copied().collect(),
                NDArrayNumericTensor::I64(a) => a.iter().map(|&v| v as f64).collect(),
                NDArrayNumericTensor::I32(a) => a.iter().map(|&v| v as f64).collect(),
                NDArrayNumericTensor::BOOL(a) => {
                    a.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect()
                }
                other => panic!("Unsupported output dtype: {:?}", other.dtype()),
            };
            for (i, &v) in flat.iter().enumerate() {
                atom_vals.insert(range.base.0 + i as u64, v);
            }
        }

        // Compare outputs.
        for out_id in &output_ids {
            let milli_tensor = &intermediates[out_id];
            let milli_flat = tensor_to_f64(milli_tensor);
            let tam = result.tensor_map.get(out_id).unwrap();

            let nano_flat: Vec<f64> = (0..tam.count)
                .map(|i| {
                    let atom = tam.atom_id_for_element(i);
                    *atom_vals.get(&atom.0).unwrap_or_else(|| {
                        panic!("atom {} not found in eval results", atom)
                    })
                })
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
            let is_bool = tam.dtype == crate::dtype::DType::BOOL;
            for (i, (m, n)) in milli_flat.iter().zip(nano_flat.iter()).enumerate() {
                let diff = if is_bool {
                    // Both should be 0.0 or 1.0
                    let mb: f64 = if *m != 0.0 { 1.0 } else { 0.0 };
                    let nb: f64 = if *n != 0.0 { 1.0 } else { 0.0 };
                    (mb - nb).abs()
                } else {
                    (m - n).abs()
                };
                let tol = if is_bool { 0.0 } else { 1e-4 * m.abs().max(1.0) };
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
                NumericTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap(),
                NumericTensor::from_vec_shape(vec![10.0f32, 20.0, 30.0, 40.0], vec![4]).unwrap(),
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
                NumericTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0], vec![3]).unwrap(),
                NumericTensor::from_vec_shape(vec![4.0f32, 5.0, 6.0], vec![3]).unwrap(),
                NumericTensor::from_vec_shape(vec![0.1f32, 0.2, 0.3], vec![3]).unwrap(),
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
                NumericTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])
                    .unwrap(),
                NumericTensor::from_vec_shape(vec![10.0f32, 20.0, 30.0], vec![3]).unwrap(),
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
            vec![NumericTensor::from_vec_shape(vec![0.0f32, 1.0, -1.0, 0.5], vec![4]).unwrap()],
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
                    crate::dtype::DType::F32,
                    rng,
                );
                (vec![a, b], vec![c])
            },
            vec![
                NumericTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])
                    .unwrap(),
                NumericTensor::from_vec_shape(vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2])
                    .unwrap(),
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
                    crate::dtype::DType::F32,
                    rng,
                );
                (vec![a, b], vec![c])
            },
            vec![
                NumericTensor::from_vec_shape(vec![1.0f32; 4 * 8], vec![4, 8]).unwrap(),
                NumericTensor::from_vec_shape(vec![1.0f32; 8 * 16], vec![8, 16]).unwrap(),
            ],
        );

        // Also verify group structure directly.
        let mut rng = rand::rng();
        let (mut milli, _ext_map) =
            crate::milli_graph::MilliOpGraph::new(std::iter::empty(), &mut rng);
        let a_id = milli.add_input(&mut rng);
        let b_id = milli.add_input(&mut rng);
        let _c_id = crate::milli_graph::ops::MatMul::push_new_default_precision(
            &mut milli,
            a_id,
            b_id,
            crate::dtype::DType::F32,
            &mut rng,
        );
        let a_tensor: crate::numeric_tensor::NumericTensor<crate::DynRank> =
            NumericTensor::from_vec_shape(vec![1.0f32; 4 * 8], vec![4, 8]).unwrap();
        let b_tensor: crate::numeric_tensor::NumericTensor<crate::DynRank> =
            NumericTensor::from_vec_shape(vec![1.0f32; 8 * 16], vec![8, 16]).unwrap();

        let mut info = std::collections::HashMap::new();
        info.insert(a_id, crate::tensor_info::TensorInfo::from(a_tensor));
        info.insert(b_id, crate::tensor_info::TensorInfo::from(b_tensor));

        let result = super::lower_with_info(&milli, &info).unwrap();
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
                InputRef::StridedBroadcast { repeat, .. } => {
                    assert_eq!(*repeat, 16, "StridedBroadcast repeat should be N=16");
                }
                other => panic!("Expected StridedBroadcast for input 0, got {:?}", other),
            }
            // Input 1 should be Affine with stride=1.
            match &g.inputs[1] {
                InputRef::Affine { stride, .. } => {
                    assert_eq!(*stride, 1, "Affine stride should be 1");
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
                NumericTensor::from_vec_shape(
                    vec![
                        10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0,
                        120.0,
                    ],
                    vec![4, 3],
                )
                .unwrap(),
                NumericTensor::from_vec_shape(vec![1.0f32, 3.0], vec![2]).unwrap(),
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
            vec![
                NumericTensor::from_vec_shape(
                    vec![1.0f32, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
                    vec![4, 3],
                )
                .unwrap(),
            ],
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
        let tensor: NumericTensor<DynRank> =
            NumericTensor::from_vec_shape(vec![1.0f32; 12], vec![4, 3]).unwrap();
        let mut info = std::collections::HashMap::new();
        info.insert(data, TensorInfo::from(tensor));
        let result = super::lower_with_info(&milli, &info).unwrap();
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
            vec![
                NumericTensor::from_vec_shape((1..=12).map(|v| v as f32).collect(), vec![2, 6])
                    .unwrap(),
            ],
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
            vec![NumericTensor::from_vec_shape(vec![1.0f32, 2., 3., 4.], vec![4]).unwrap()],
        );
    }

    #[test]
    fn test_slice_zero_cost() {
        // Slice [6] with start=1, end=4, step=1 → [3]
        use crate::backends::ndarray_backend::NDArrayNumericTensor;
        check_integrity(
            |graph, rng| {
                let data = graph.add_input(rng);
                let starts = crate::milli_graph::ops::Constant::push_new(
                    graph,
                    NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![1i64], &vec![1]).unwrap(),
                    rng,
                );
                let ends = crate::milli_graph::ops::Constant::push_new(
                    graph,
                    NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![4i64], &vec![1]).unwrap(),
                    rng,
                );
                let out = crate::milli_graph::ops::Slice::push_new(
                    graph, data, starts, ends, None, None, rng,
                );
                (vec![data], vec![out])
            },
            vec![
                NumericTensor::from_vec_shape(vec![10.0f32, 20., 30., 40., 50., 60.], vec![6])
                    .unwrap(),
            ],
        );
    }

    #[test]
    fn test_slice_zero_cost_2d() {
        // Slice [3, 4] along axis 0 with start=1, end=3 → [2, 4]
        use crate::backends::ndarray_backend::NDArrayNumericTensor;
        check_integrity(
            |graph, rng| {
                let data = graph.add_input(rng);
                let starts = crate::milli_graph::ops::Constant::push_new(
                    graph,
                    NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![1i64], &vec![1]).unwrap(),
                    rng,
                );
                let ends = crate::milli_graph::ops::Constant::push_new(
                    graph,
                    NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![3i64], &vec![1]).unwrap(),
                    rng,
                );
                let axes = crate::milli_graph::ops::Constant::push_new(
                    graph,
                    NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![0i64], &vec![1]).unwrap(),
                    rng,
                );
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
            vec![
                NumericTensor::from_vec_shape((1..=12).map(|v| v as f32).collect(), vec![3, 4])
                    .unwrap(),
            ],
        );
    }

    #[test]
    fn test_slice_then_add() {
        // Slice [4] two ways and add: data[0:2] + data[2:4]
        use crate::backends::ndarray_backend::NDArrayNumericTensor;
        check_integrity(
            |graph, rng| {
                let data = graph.add_input(rng);
                let s0 = crate::milli_graph::ops::Constant::push_new(
                    graph,
                    NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![0i64], &vec![1]).unwrap(),
                    rng,
                );
                let e0 = crate::milli_graph::ops::Constant::push_new(
                    graph,
                    NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![2i64], &vec![1]).unwrap(),
                    rng,
                );
                let s1 = crate::milli_graph::ops::Constant::push_new(
                    graph,
                    NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![2i64], &vec![1]).unwrap(),
                    rng,
                );
                let e1 = crate::milli_graph::ops::Constant::push_new(
                    graph,
                    NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![4i64], &vec![1]).unwrap(),
                    rng,
                );
                let out0 =
                    crate::milli_graph::ops::Slice::push_new(graph, data, s0, e0, None, None, rng);
                let out1 =
                    crate::milli_graph::ops::Slice::push_new(graph, data, s1, e1, None, None, rng);
                let sum = crate::milli_graph::ops::SimpleBinary::add(graph, out0, out1, rng);
                (vec![data], vec![sum])
            },
            vec![NumericTensor::from_vec_shape(vec![1.0f32, 2., 3., 4.], vec![4]).unwrap()],
        );
    }

    #[test]
    fn test_slice_zero_cost_inner_axis() {
        // Slice [3, 4] along axis 1 with start=1, end=3 → [3, 2]
        // This is an inner-axis slice that falls back to Explicit, but
        // should still produce correct results.
        use crate::backends::ndarray_backend::NDArrayNumericTensor;
        check_integrity(
            |graph, rng| {
                let data = graph.add_input(rng);
                let starts = crate::milli_graph::ops::Constant::push_new(
                    graph,
                    NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![1i64], &vec![1]).unwrap(),
                    rng,
                );
                let ends = crate::milli_graph::ops::Constant::push_new(
                    graph,
                    NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![3i64], &vec![1]).unwrap(),
                    rng,
                );
                let axes = crate::milli_graph::ops::Constant::push_new(
                    graph,
                    NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![1i64], &vec![1]).unwrap(),
                    rng,
                );
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
            vec![
                NumericTensor::from_vec_shape((1..=12).map(|v| v as f32).collect(), vec![3, 4])
                    .unwrap(),
            ],
        );
    }

    #[test]
    fn test_slice_no_identity_groups() {
        // Verify Slice [6] start=1 end=4 creates no Identity groups (zero-cost).
        use crate::backends::ndarray_backend::NDArrayNumericTensor;
        let mut rng = rand::rng();
        let (mut milli, _) = MilliOpGraph::new(std::iter::empty(), &mut rng);
        let data = milli.add_input(&mut rng);
        let starts = crate::milli_graph::ops::Constant::push_new(
            &mut milli,
            NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![1i64], &vec![1]).unwrap(),
            &mut rng,
        );
        let ends = crate::milli_graph::ops::Constant::push_new(
            &mut milli,
            NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![4i64], &vec![1]).unwrap(),
            &mut rng,
        );
        let _out = crate::milli_graph::ops::Slice::push_new(
            &mut milli, data, starts, ends, None, None, &mut rng,
        );
        let tensor: NumericTensor<DynRank> =
            NumericTensor::from_vec_shape(vec![1.0f32; 6], vec![6]).unwrap();
        let mut info = std::collections::HashMap::new();
        info.insert(data, TensorInfo::from(tensor));
        let result = super::lower_with_info(&milli, &info).unwrap();
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
            vec![
                NumericTensor::from_vec_shape(vec![10.0f32, 20., 30., 40., 50., 60.], vec![6])
                    .unwrap(),
            ],
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
                NumericTensor::from_vec_shape(vec![1.0f32, 2., 3.], vec![3]).unwrap(),
                NumericTensor::from_vec_shape(vec![4.0f32, 5., 6.], vec![3]).unwrap(),
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
                NumericTensor::from_vec_shape(vec![100.0f32, 200.0, 300.0, 400.0], vec![4])
                    .unwrap(),
                NumericTensor::from_vec_shape(vec![0.0f32, 2.0, 3.0], vec![3]).unwrap(),
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
                    graph, a, b, DType::F32, DType::F32, DType::F32, DType::F32, rng,
                );
                (vec![a, b], vec![c])
            },
            vec![
                NumericTensor::from_vec_shape(a_data, vec![m, k]).unwrap(),
                NumericTensor::from_vec_shape(b_data, vec![k, n]).unwrap(),
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

                let mm1 = MatMul::push_new(graph, x, w1, DType::F32, DType::F32, DType::F32, DType::F32, rng);
                let h = SimpleBinary::add(graph, mm1, b1, rng);
                let mm2 = MatMul::push_new(graph, h, w2, DType::F32, DType::F32, DType::F32, DType::F32, rng);
                let out = SimpleBinary::add(graph, mm2, b2, rng);
                (vec![x, w1, b1, w2, b2], vec![out])
            },
            vec![
                NumericTensor::from_vec_shape((0..256).map(|i| (i as f32) * 0.01 - 1.28).collect(), vec![4, 64]).unwrap(),
                NumericTensor::from_vec_shape((0..4096).map(|i| (i as f32) * 0.002 - 4.0).collect(), vec![64, 64]).unwrap(),
                NumericTensor::from_vec_shape((0..64).map(|i| (i as f32) * 0.1 - 3.2).collect(), vec![64]).unwrap(),
                NumericTensor::from_vec_shape((0..2048).map(|i| (i as f32) * 0.003 - 3.0).collect(), vec![64, 32]).unwrap(),
                NumericTensor::from_vec_shape((0..32).map(|i| (i as f32) * 0.05 - 0.8).collect(), vec![32]).unwrap(),
            ],
        );
    }

    /// MatMul -> Transpose -> LayerNorm-like chain.
    #[test]
    fn test_three_way_transpose_layernorm() {
        use crate::backends::ndarray_backend::NDArrayNumericTensor;
        use crate::milli_graph::ops::{
            Constant, MatMul, Pow, ReduceMean, SimpleBinary, SimpleUnaryOp, Transpose,
        };
        use ndarray::{ArcArray, IxDyn};

        check_integrity(
            |graph, rng| {
                let x = graph.add_input(rng);
                let w = graph.add_input(rng);
                let gamma = graph.add_input(rng);
                let beta = graph.add_input(rng);

                let mm = MatMul::push_new(graph, x, w, DType::F32, DType::F32, DType::F32, DType::F32, rng);
                let transposed = Transpose::push_new(graph, mm, Some(vec![0, 2, 1]), rng);

                let axes_tensor = NDArrayNumericTensor::I64(ArcArray::from_shape_vec(IxDyn(&[1]), vec![-1i64]).unwrap());
                let axes_id = Constant::push_new(graph, axes_tensor, rng);
                let mean = ReduceMean::push_new(graph, transposed, Some(axes_id), true, false, rng);
                let centered = SimpleBinary::sub(graph, transposed, mean, rng);

                let pow2_tensor = NDArrayNumericTensor::F32(ArcArray::from_shape_vec(IxDyn(&[1]), vec![2.0f32]).unwrap());
                let pow2_id = Constant::push_new(graph, pow2_tensor, rng);
                let squared = Pow::push_new(graph, centered, pow2_id, rng);

                let axes_id2 = Constant::push_new(graph, NDArrayNumericTensor::I64(ArcArray::from_shape_vec(IxDyn(&[1]), vec![-1i64]).unwrap()), rng);
                let var = ReduceMean::push_new(graph, squared, Some(axes_id2), true, false, rng);

                let eps_tensor = NDArrayNumericTensor::F32(ArcArray::from_shape_vec(IxDyn(&[1]), vec![1e-5f32]).unwrap());
                let eps_id = Constant::push_new(graph, eps_tensor, rng);
                let var_eps = SimpleBinary::add(graph, var, eps_id, rng);
                let std_dev = SimpleUnaryOp::sqrt(graph, var_eps, rng);
                let normed = SimpleBinary::div(graph, centered, std_dev, rng);
                let scaled = SimpleBinary::mul(graph, normed, gamma, rng);
                let out = SimpleBinary::add(graph, scaled, beta, rng);

                (vec![x, w, gamma, beta], vec![out])
            },
            vec![
                NumericTensor::from_vec_shape((0..64).map(|i| (i as f32) * 0.1 - 3.2).collect(), vec![2, 4, 8]).unwrap(),
                NumericTensor::from_vec_shape((0..64).map(|i| (i as f32) * 0.02 - 0.64).collect(), vec![8, 8]).unwrap(),
                NumericTensor::from_vec_shape(vec![1.0f32, 1.1, 0.9, 1.05], vec![4]).unwrap(),
                NumericTensor::from_vec_shape(vec![0.0f32, 0.1, -0.1, 0.05], vec![4]).unwrap(),
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
                NumericTensor::from_vec_shape(vec![10.0f32, 20.0, 30.0], vec![3]).unwrap(),
                NumericTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0], vec![3]).unwrap(),
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
                NumericTensor::from_vec_shape(vec![12.0f32, 20.0, 30.0, 7.0], vec![4]).unwrap(),
                NumericTensor::from_vec_shape(vec![3.0f32, 4.0, 5.0, 2.0], vec![4]).unwrap(),
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
                NumericTensor::from_vec_shape(vec![1.0f32, 5.0, 3.0, 8.0], vec![4]).unwrap(),
                NumericTensor::from_vec_shape(vec![4.0f32, 2.0, 6.0, 1.0], vec![4]).unwrap(),
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
                NumericTensor::from_vec_shape(vec![1.0f32, 5.0, 3.0, 8.0], vec![4]).unwrap(),
                NumericTensor::from_vec_shape(vec![4.0f32, 2.0, 6.0, 1.0], vec![4]).unwrap(),
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
                NumericTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap(),
                NumericTensor::from_vec_shape(vec![1.0f32, 9.0, 3.0, 0.0], vec![4]).unwrap(),
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
                NumericTensor::from_vec_shape(vec![1.0f32, 5.0, 3.0, 4.0], vec![4]).unwrap(),
                NumericTensor::from_vec_shape(vec![4.0f32, 2.0, 3.0, 8.0], vec![4]).unwrap(),
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
                NumericTensor::from_vec_shape(vec![1.0f32, 5.0, 3.0, 4.0], vec![4]).unwrap(),
                NumericTensor::from_vec_shape(vec![4.0f32, 2.0, 3.0, 8.0], vec![4]).unwrap(),
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
                NumericTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap(),
                NumericTensor::from_vec_shape(vec![10.0f32], vec![1]).unwrap(),
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
                NumericTensor::from_vec_shape(
                    (0..12).map(|i| i as f32).collect(),
                    vec![3, 4],
                )
                .unwrap(),
                NumericTensor::from_vec_shape(vec![100.0f32, 200.0, 300.0, 400.0], vec![4])
                    .unwrap(),
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
            vec![NumericTensor::from_vec_shape(vec![1.0f32, -2.0, 0.0, 3.5], vec![4]).unwrap()],
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
            vec![NumericTensor::from_vec_shape(vec![-3.0f32, 2.0, -1.0, 0.0], vec![4]).unwrap()],
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
            vec![NumericTensor::from_vec_shape(vec![4.0f32, 9.0, 16.0, 1.0], vec![4]).unwrap()],
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
            vec![NumericTensor::from_vec_shape(vec![2.0f32, 4.0, 0.5, 1.0], vec![4]).unwrap()],
        );
    }

    #[test]
    fn test_unary_tanh() {
        check_integrity(
            |g, rng| {
                let a = g.add_input(rng);
                let b = crate::milli_graph::ops::SimpleUnaryOp::trig(g, a, crate::TrigOp::Tanh, rng);
                (vec![a], vec![b])
            },
            vec![NumericTensor::from_vec_shape(vec![0.0f32, 1.0, -1.0, 2.0], vec![4]).unwrap()],
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
            vec![NumericTensor::from_vec_shape(vec![1.0f32, 2.0, 0.5, 10.0], vec![4]).unwrap()],
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
            vec![NumericTensor::from_vec_shape(vec![1.5f32, -1.5, 2.0, 0.1], vec![4]).unwrap()],
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
                NumericTensor::from_vec_shape(vec![2.0f32, 3.0, 4.0, 5.0], vec![4]).unwrap(),
                NumericTensor::from_vec_shape(vec![3.0f32, 2.0, 0.5, 1.0], vec![4]).unwrap(),
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
            vec![NumericTensor::from_vec_shape(vec![-2.0f32, -0.5, 0.0, 1.0, 3.0], vec![5])
                .unwrap()],
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
                NumericTensor::from_vec_shape(vec![1.0f32, -1.0, 2.0, -2.0], vec![4]).unwrap(),
                NumericTensor::from_vec_shape(vec![10.0f32, 20.0, 30.0, 40.0], vec![4]).unwrap(),
                NumericTensor::from_vec_shape(vec![100.0f32, 200.0, 300.0, 400.0], vec![4])
                    .unwrap(),
                NumericTensor::from_vec_shape(vec![0.0f32, 0.0, 0.0, 0.0], vec![4]).unwrap(),
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
                NumericTensor::from_vec_shape(
                    (0..12).map(|i| i as f32).collect(),
                    vec![3, 4],
                )
                .unwrap(),
                NumericTensor::from_vec_shape(vec![2i64, 6], vec![2]).unwrap(),
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
                NumericTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0], vec![1, 3]).unwrap(),
                NumericTensor::from_vec_shape(vec![0i64], vec![1]).unwrap(),
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
                NumericTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0], vec![3]).unwrap(),
                NumericTensor::from_vec_shape(vec![0i64], vec![1]).unwrap(),
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
                NumericTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0], vec![1, 3]).unwrap(),
                NumericTensor::from_vec_shape(vec![4i64, 3], vec![2]).unwrap(),
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
            vec![NumericTensor::from_vec_shape(
                (0..12).map(|i| i as f32).collect(),
                vec![3, 4],
            )
            .unwrap()],
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
            vec![NumericTensor::from_vec_shape(
                (0..24).map(|i| i as f32).collect(),
                vec![2, 3, 4],
            )
            .unwrap()],
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
                let b = crate::milli_graph::ops::Cast::push_new(
                    g,
                    a,
                    crate::dtype::DType::F64,
                    rng,
                );
                (vec![a], vec![b])
            },
            vec![NumericTensor::from_vec_shape(vec![1.5f32, -2.5, 0.0, 3.14], vec![4]).unwrap()],
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
                use crate::backends::ndarray_backend::NDArrayNumericTensor;
                use ndarray::{ArcArray, IxDyn};
                let a = g.add_input(rng);
                let axes_tensor =
                    NDArrayNumericTensor::I64(ArcArray::from_shape_vec(IxDyn(&[1]), vec![0i64]).unwrap());
                let axes = crate::milli_graph::ops::Constant::push_new(g, axes_tensor, rng);
                let b = crate::milli_graph::ops::ReduceSum::push_new(g, a, Some(axes), false, false, rng);
                (vec![a], vec![b])
            },
            vec![NumericTensor::from_vec_shape(
                (1..=12).map(|i| i as f32).collect(),
                vec![3, 4],
            )
            .unwrap()],
        );
    }

    #[test]
    fn test_reduce_sum_axis1() {
        // [3, 4] → reduce sum axis 1 → [3]
        check_integrity(
            |g, rng| {
                use crate::backends::ndarray_backend::NDArrayNumericTensor;
                use ndarray::{ArcArray, IxDyn};
                let a = g.add_input(rng);
                let axes_tensor =
                    NDArrayNumericTensor::I64(ArcArray::from_shape_vec(IxDyn(&[1]), vec![1i64]).unwrap());
                let axes = crate::milli_graph::ops::Constant::push_new(g, axes_tensor, rng);
                let b = crate::milli_graph::ops::ReduceSum::push_new(g, a, Some(axes), false, false, rng);
                (vec![a], vec![b])
            },
            vec![NumericTensor::from_vec_shape(
                (1..=12).map(|i| i as f32).collect(),
                vec![3, 4],
            )
            .unwrap()],
        );
    }

    #[test]
    fn test_reduce_max_axis1() {
        // [3, 4] → reduce max axis 1 → [3]
        check_integrity(
            |g, rng| {
                use crate::backends::ndarray_backend::NDArrayNumericTensor;
                use ndarray::{ArcArray, IxDyn};
                let a = g.add_input(rng);
                let axes_tensor =
                    NDArrayNumericTensor::I64(ArcArray::from_shape_vec(IxDyn(&[1]), vec![1i64]).unwrap());
                let axes = crate::milli_graph::ops::Constant::push_new(g, axes_tensor, rng);
                let b = crate::milli_graph::ops::ReduceMax::push_new(g, a, Some(axes), false, false, rng);
                (vec![a], vec![b])
            },
            vec![NumericTensor::from_vec_shape(
                vec![3.0f32, 1.0, 4.0, 1.5, 9.0, 2.6, 5.3, 5.8, 9.7, 9.3, 2.3, 8.4],
                vec![3, 4],
            )
            .unwrap()],
        );
    }

    #[test]
    fn test_reduce_mean_axis1() {
        // [3, 4] → reduce mean axis 1 → [3]
        check_integrity(
            |g, rng| {
                use crate::backends::ndarray_backend::NDArrayNumericTensor;
                use ndarray::{ArcArray, IxDyn};
                let a = g.add_input(rng);
                let axes_tensor =
                    NDArrayNumericTensor::I64(ArcArray::from_shape_vec(IxDyn(&[1]), vec![1i64]).unwrap());
                let axes = crate::milli_graph::ops::Constant::push_new(g, axes_tensor, rng);
                let b = crate::milli_graph::ops::ReduceMean::push_new(g, a, Some(axes), false, false, rng);
                (vec![a], vec![b])
            },
            vec![NumericTensor::from_vec_shape(
                (1..=12).map(|i| i as f32).collect(),
                vec![3, 4],
            )
            .unwrap()],
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
                    g, a, b, DType::F32, rng,
                );
                (vec![a, b], vec![c])
            },
            vec![
                NumericTensor::from_vec_shape(
                    (0..24).map(|i| (i as f32) * 0.1).collect(),
                    vec![2, 3, 4],
                )
                .unwrap(),
                NumericTensor::from_vec_shape(
                    (0..20).map(|i| (i as f32) * 0.1).collect(),
                    vec![4, 5],
                )
                .unwrap(),
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
                NumericTensor::from_vec_shape(
                    (0..12).map(|i| i as f32).collect(),
                    vec![3, 4],
                )
                .unwrap(),
                NumericTensor::from_vec_shape(vec![100.0f32, 200.0, 300.0], vec![3]).unwrap(),
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
                    g, at, b, DType::F32, rng,
                );
                (vec![a, b], vec![c])
            },
            vec![
                NumericTensor::from_vec_shape(
                    (0..12).map(|i| (i as f32) * 0.1).collect(),
                    vec![3, 4],
                )
                .unwrap(),
                NumericTensor::from_vec_shape(
                    (0..6).map(|i| (i as f32) * 0.1).collect(),
                    vec![3, 2],
                )
                .unwrap(),
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
                // Note: ConstantOfShape::push_new returns the OP id (not output tensor id).
                let cos_op_id = crate::milli_graph::ops::ConstantOfShape::push_new(
                    g,
                    NumericScalar::F32(7.0),
                    shape,
                    rng,
                );
                let constant = g.get_node_by_id(&cos_op_id).unwrap().outputs().next().unwrap();
                let out = crate::milli_graph::ops::SimpleBinary::add(g, x, constant, rng);
                (vec![shape, x], vec![out])
            },
            vec![
                NumericTensor::from_vec_shape(vec![2i64, 3], vec![2]).unwrap(),
                NumericTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])
                    .unwrap(),
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
                    g, a, starts, ends, Some(steps), Some(axes), rng,
                );
                (vec![a, starts, ends, axes, steps], vec![b])
            },
            vec![
                NumericTensor::from_vec_shape(
                    (0..8).map(|i| (i as f32) * 10.0).collect(),
                    vec![8],
                )
                .unwrap(),
                NumericTensor::from_vec_shape(vec![0i64], vec![1]).unwrap(),
                NumericTensor::from_vec_shape(vec![8i64], vec![1]).unwrap(),
                NumericTensor::from_vec_shape(vec![0i64], vec![1]).unwrap(),
                NumericTensor::from_vec_shape(vec![2i64], vec![1]).unwrap(),
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
                    g, cat, w, DType::F32, rng,
                );
                (vec![a, b, w], vec![out])
            },
            vec![
                NumericTensor::from_vec_shape(
                    (0..6).map(|i| (i as f32) * 0.1).collect(),
                    vec![2, 3],
                )
                .unwrap(),
                NumericTensor::from_vec_shape(
                    (6..12).map(|i| (i as f32) * 0.1).collect(),
                    vec![2, 3],
                )
                .unwrap(),
                NumericTensor::from_vec_shape(
                    (0..6).map(|i| (i as f32) * 0.1).collect(),
                    vec![3, 2],
                )
                .unwrap(),
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
                NumericTensor::from_vec_shape(
                    (0..40).map(|i| (i as f32) * 0.1).collect(),
                    vec![10, 4],
                )
                .unwrap(),
                NumericTensor::from_vec_shape(vec![0i64, 3, 7, 1, 5, 9], vec![2, 3]).unwrap(),
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
                let tok_emb = crate::milli_graph::ops::Gather::push_new(g, tok_data, indices, 0, rng);
                let out = crate::milli_graph::ops::SimpleBinary::add(g, tok_emb, pos_data, rng);
                (vec![tok_data, pos_data, indices], vec![out])
            },
            vec![
                NumericTensor::from_vec_shape(
                    (0..80).map(|i| (i as f32) * 0.01).collect(),
                    vec![10, 8],
                )
                .unwrap(),
                NumericTensor::from_vec_shape(
                    (0..24).map(|i| (i as f32) * 0.1).collect(),
                    vec![3, 8],
                )
                .unwrap(),
                NumericTensor::from_vec_shape(vec![0i64, 3, 7, 1, 5, 9], vec![2, 3]).unwrap(),
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
                NumericTensor::from_vec_shape(
                    (0..96).map(|i| (i as f32) * 0.01).collect(),
                    vec![2, 4, 12],
                )
                .unwrap(),
                NumericTensor::from_vec_shape(vec![2i64, 4, 3, 4], vec![4]).unwrap(),
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
                let b = crate::milli_graph::ops::Transpose::push_new(
                    g, a, Some(vec![0, 2, 1, 3]), rng,
                );
                (vec![a], vec![b])
            },
            vec![NumericTensor::from_vec_shape(
                (0..48).map(|i| (i as f32) * 0.1).collect(),
                vec![2, 3, 2, 4], // [B=2, S=3, H=2, D=4]
            )
            .unwrap()],
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
                    g, q, k, DType::F32, rng,
                );
                (vec![q, k], vec![c])
            },
            vec![
                NumericTensor::from_vec_shape(
                    (0..24).map(|i| (i as f32) * 0.1).collect(),
                    vec![1, 2, 3, 4], // Q: [B, H, S, D]
                )
                .unwrap(),
                NumericTensor::from_vec_shape(
                    (0..24).map(|i| (i as f32) * 0.1).collect(),
                    vec![1, 2, 4, 3], // K^T: [B, H, D, S]
                )
                .unwrap(),
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
                let q_4d = crate::milli_graph::ops::Reshape::push_new(
                    g, q_flat, q_shape, false, rng,
                );
                let q_perm = crate::milli_graph::ops::Transpose::push_new(
                    g, q_4d, Some(vec![0, 2, 1, 3]), rng,
                );
                let scores = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    g, q_perm, k_t, DType::F32, rng,
                );
                (vec![q_flat, q_shape, k_t], vec![scores])
            },
            vec![
                NumericTensor::from_vec_shape(
                    (0..48).map(|i| (i as f32) * 0.01).collect(),
                    vec![2, 3, 8], // Q: [B=2, S=3, D=8]
                )
                .unwrap(),
                NumericTensor::from_vec_shape(vec![2i64, 3, 2, 4], vec![4]).unwrap(),
                NumericTensor::from_vec_shape(
                    (0..48).map(|i| (i as f32) * 0.01).collect(),
                    vec![2, 2, 4, 3], // K^T: [B=2, H=2, D=4, S=3]
                )
                .unwrap(),
            ],
        );
    }

    #[test]
    fn test_softmax_pattern() {
        // Softmax: ReduceMax → Sub → Exp → ReduceSum → Div
        // input [2, 4]
        check_integrity(
            |g, rng| {
                use crate::backends::ndarray_backend::NDArrayNumericTensor;
                use ndarray::{ArcArray, IxDyn};

                let x = g.add_input(rng);
                let axes_tensor = NDArrayNumericTensor::I64(
                    ArcArray::from_shape_vec(IxDyn(&[1]), vec![1i64]).unwrap(),
                );
                let axes = crate::milli_graph::ops::Constant::push_new(g, axes_tensor, rng);
                let axes2 = crate::milli_graph::ops::Constant::push_new(
                    g,
                    NDArrayNumericTensor::I64(
                        ArcArray::from_shape_vec(IxDyn(&[1]), vec![1i64]).unwrap(),
                    ),
                    rng,
                );

                // max_val = ReduceMax(x, axis=1, keepdims=true)
                let max_val =
                    crate::milli_graph::ops::ReduceMax::push_new(g, x, Some(axes), true, false, rng);
                // shifted = x - max_val
                let shifted = crate::milli_graph::ops::SimpleBinary::sub(g, x, max_val, rng);
                // exp_val = Exp(shifted)
                let exp_val = crate::milli_graph::ops::SimpleUnaryOp::exp(g, shifted, rng);
                // sum_exp = ReduceSum(exp_val, axis=1, keepdims=true)
                let sum_exp = crate::milli_graph::ops::ReduceSum::push_new(
                    g, exp_val, Some(axes2), true, false, rng,
                );
                // result = exp_val / sum_exp
                let result = crate::milli_graph::ops::SimpleBinary::div(g, exp_val, sum_exp, rng);

                (vec![x], vec![result])
            },
            vec![NumericTensor::from_vec_shape(
                vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 1.0, 0.5, 2.0],
                vec![2, 4],
            )
            .unwrap()],
        );
    }

    #[test]
    fn test_layernorm_pattern() {
        // LayerNorm: ReduceMean → Sub → Pow(2) → ReduceMean → Add(eps) → Sqrt → Div → Mul(gamma) → Add(beta)
        // input [2, 4], gamma [4], beta [4]
        check_integrity(
            |g, rng| {
                use crate::backends::ndarray_backend::NDArrayNumericTensor;
                use ndarray::{ArcArray, IxDyn};

                let x = g.add_input(rng);
                let gamma = g.add_input(rng);
                let beta = g.add_input(rng);

                let axes1 = crate::milli_graph::ops::Constant::push_new(
                    g,
                    NDArrayNumericTensor::I64(
                        ArcArray::from_shape_vec(IxDyn(&[1]), vec![-1i64]).unwrap(),
                    ),
                    rng,
                );
                let axes2 = crate::milli_graph::ops::Constant::push_new(
                    g,
                    NDArrayNumericTensor::I64(
                        ArcArray::from_shape_vec(IxDyn(&[1]), vec![-1i64]).unwrap(),
                    ),
                    rng,
                );
                let pow2 = crate::milli_graph::ops::Constant::push_new(
                    g,
                    NDArrayNumericTensor::F32(
                        ArcArray::from_shape_vec(IxDyn(&[1]), vec![2.0f32]).unwrap(),
                    ),
                    rng,
                );
                let eps = crate::milli_graph::ops::Constant::push_new(
                    g,
                    NDArrayNumericTensor::F32(
                        ArcArray::from_shape_vec(IxDyn(&[1]), vec![1e-5f32]).unwrap(),
                    ),
                    rng,
                );

                let mean = crate::milli_graph::ops::ReduceMean::push_new(
                    g, x, Some(axes1), true, false, rng,
                );
                let centered = crate::milli_graph::ops::SimpleBinary::sub(g, x, mean, rng);
                let squared = crate::milli_graph::ops::Pow::push_new(g, centered, pow2, rng);
                let var = crate::milli_graph::ops::ReduceMean::push_new(
                    g, squared, Some(axes2), true, false, rng,
                );
                let var_eps = crate::milli_graph::ops::SimpleBinary::add(g, var, eps, rng);
                let std_dev = crate::milli_graph::ops::SimpleUnaryOp::sqrt(g, var_eps, rng);
                let normed = crate::milli_graph::ops::SimpleBinary::div(g, centered, std_dev, rng);
                let scaled = crate::milli_graph::ops::SimpleBinary::mul(g, normed, gamma, rng);
                let out = crate::milli_graph::ops::SimpleBinary::add(g, scaled, beta, rng);

                (vec![x, gamma, beta], vec![out])
            },
            vec![
                NumericTensor::from_vec_shape(
                    vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 1.0, 0.5, 2.0],
                    vec![2, 4],
                )
                .unwrap(),
                NumericTensor::from_vec_shape(vec![1.0f32, 1.0, 1.0, 1.0], vec![4]).unwrap(),
                NumericTensor::from_vec_shape(vec![0.0f32, 0.0, 0.0, 0.0], vec![4]).unwrap(),
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

                let k4 = crate::milli_graph::ops::Reshape::push_new(
                    g, k_flat, k_shape, false, rng,
                );
                let kt1 = crate::milli_graph::ops::Transpose::push_new(
                    g, k4, Some(vec![0, 2, 1, 3]), rng,
                );
                let kt2 = crate::milli_graph::ops::Transpose::push_new(
                    g, kt1, Some(vec![0, 1, 3, 2]), rng,
                );
                let scores = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    g, q, kt2, DType::F32, rng,
                );

                (vec![k_flat, k_shape, q], vec![scores])
            },
            vec![
                // K [1, 3, 4]
                NumericTensor::from_vec_shape(
                    (0..12).map(|i| (i as f32) * 0.1).collect(),
                    vec![1, 3, 4],
                )
                .unwrap(),
                // shape for [1, 3, 2, 2]
                NumericTensor::from_vec_shape(vec![1i64, 3, 2, 2], vec![4]).unwrap(),
                // Q [1, 2, 3, 2]
                NumericTensor::from_vec_shape(
                    (0..12).map(|i| (i as f32) * 0.1).collect(),
                    vec![1, 2, 3, 2],
                )
                .unwrap(),
            ],
        );
    }

    #[test]
    fn test_attention_block_small() {
        // Minimal attention: Q/K/V projections → head split → attention scores → output
        // B=1, S=3, D=4, H=2, D_h=2
        check_integrity(
            |g, rng| {
                let x = g.add_input(rng);     // [1, 3, 4]
                let wq = g.add_input(rng);    // [4, 4]
                let wk = g.add_input(rng);    // [4, 4]
                let wv = g.add_input(rng);    // [4, 4]

                // Q = x @ Wq, K = x @ Wk, V = x @ Wv  → all [1, 3, 4]
                let q = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    g, x, wq, DType::F32, rng,
                );
                let k = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    g, x, wk, DType::F32, rng,
                );
                let v = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    g, x, wv, DType::F32, rng,
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
                    g, q4, Some(vec![0, 2, 1, 3]), rng,
                );
                let kt = crate::milli_graph::ops::Transpose::push_new(
                    g, k4, Some(vec![0, 2, 1, 3]), rng,
                );
                let vt = crate::milli_graph::ops::Transpose::push_new(
                    g, v4, Some(vec![0, 2, 1, 3]), rng,
                );

                // K^T: [1, 2, 3, 2] → [1, 2, 2, 3]
                let kt_t = crate::milli_graph::ops::Transpose::push_new(
                    g, kt, Some(vec![0, 1, 3, 2]), rng,
                );

                // Attention scores: Q @ K^T = [1, 2, 3, 2] @ [1, 2, 2, 3] → [1, 2, 3, 3]
                let scores = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    g, qt, kt_t, DType::F32, rng,
                );

                // Attention output: scores @ V = [1, 2, 3, 3] @ [1, 2, 3, 2] → [1, 2, 3, 2]
                let attn_out = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    g, scores, vt, DType::F32, rng,
                );

                (
                    vec![x, wq, wk, wv, shape4d, shape4d_k, shape4d_v],
                    vec![scores, attn_out],
                )
            },
            vec![
                // x [1, 3, 4]
                NumericTensor::from_vec_shape(
                    (0..12).map(|i| (i as f32) * 0.1).collect(),
                    vec![1, 3, 4],
                )
                .unwrap(),
                // Wq [4, 4]
                NumericTensor::from_vec_shape(
                    (0..16).map(|i| (i as f32) * 0.05 - 0.4).collect(),
                    vec![4, 4],
                )
                .unwrap(),
                // Wk [4, 4]
                NumericTensor::from_vec_shape(
                    (0..16).map(|i| (i as f32) * 0.03 + 0.1).collect(),
                    vec![4, 4],
                )
                .unwrap(),
                // Wv [4, 4]
                NumericTensor::from_vec_shape(
                    (0..16).map(|i| (i as f32) * 0.04 - 0.2).collect(),
                    vec![4, 4],
                )
                .unwrap(),
                // shape [1, 3, 2, 2] — three copies for Q, K, V
                NumericTensor::from_vec_shape(vec![1i64, 3, 2, 2], vec![4]).unwrap(),
                NumericTensor::from_vec_shape(vec![1i64, 3, 2, 2], vec![4]).unwrap(),
                NumericTensor::from_vec_shape(vec![1i64, 3, 2, 2], vec![4]).unwrap(),
            ],
        );
    }
}
