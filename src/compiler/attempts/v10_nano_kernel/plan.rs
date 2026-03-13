#![allow(clippy::all, dead_code, unreachable_patterns)]
//! NanoGraph analysis, fusion, and kernel planning.
//!
//! Pipeline:
//! 1. Build a dependency DAG between AtomGroups.
//! 2. Identify materialization boundaries (multi-consumer, graph output, reduction output).
//! 3. Walk backwards from each boundary to fuse single-consumer pointwise chains.
//! 4. Emit KernelPlan for each fused cluster.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::dtype::DType;
use crate::nano_graph::lower::LowerResult;
use crate::nano_graph::{AtomGroup, AtomId, InputRef, NanoGraph};
use crate::nano_graph::{ScalarBinOp, ScalarOp, ScalarUnaryOp};
use crate::numeric_scalar::NumericScalar;

// ---------------------------------------------------------------------------
// Group dependency graph
// ---------------------------------------------------------------------------

/// Which group a group reads from, and via which input index.
#[derive(Debug, Clone)]
struct GroupDep {
    /// Index of the source group in NanoGraph::groups().
    source_group: usize,
    /// Which input slot of the consumer references this source.
    input_idx: usize,
}

/// Dependency info for one group.
#[derive(Debug)]
struct GroupInfo {
    /// Groups this group reads from (may contain duplicates if multiple inputs
    /// reference the same source group).
    deps: Vec<GroupDep>,
    /// Groups that read from this group.
    consumers: Vec<usize>,
    /// Whether this group's output must be materialized to a buffer.
    must_materialize: bool,
}

/// Build the group dependency graph.
fn build_group_graph(graph: &NanoGraph) -> Vec<GroupInfo> {
    let groups = graph.groups();
    let n = groups.len();
    let mut infos: Vec<GroupInfo> = (0..n)
        .map(|_| GroupInfo {
            deps: Vec::new(),
            consumers: Vec::new(),
            must_materialize: false,
        })
        .collect();

    for (gi, group) in groups.iter().enumerate() {
        for (inp_idx, input_ref) in group.inputs.iter().enumerate() {
            // For SymAffine, the input may span multiple source groups across k.
            // We need to discover ALL source groups, not just the one at k=0.
            let mut source_groups_seen: HashSet<usize> = HashSet::new();

            match input_ref {
                InputRef::SymAffine {
                    base,
                    stride_i,
                    stride_k,
                } => {
                    // Find all source groups by sampling across k values.
                    // The sym_dim bound tells us how many k values there are.
                    if !group.reduce_dims.is_empty() {
                        let rd = group.reduce_dims[0];
                        let bound = graph.sym_dim_bounds.get(&rd).copied().unwrap_or(1);
                        for k in 0..bound as u32 {
                            let source_atom = input_ref.resolve(0, k);
                            if let Some(src_gi) = find_group_idx(groups, source_atom) {
                                if source_groups_seen.insert(src_gi) {
                                    infos[gi].deps.push(GroupDep {
                                        source_group: src_gi,
                                        input_idx: inp_idx,
                                    });
                                    infos[src_gi].consumers.push(gi);
                                }
                            }
                        }
                    } else {
                        // SymAffine without reduce_dims (unusual).
                        let source_atom: AtomId = input_ref.resolve(0, 0);
                        if let Some(src_gi) = find_group_idx(groups, source_atom) {
                            infos[gi].deps.push(GroupDep {
                                source_group: src_gi,
                                input_idx: inp_idx,
                            });
                            infos[src_gi].consumers.push(gi);
                        }
                    }
                }
                InputRef::Explicit(ids) => {
                    // Explicit may reference multiple source groups.
                    for id in ids {
                        if let Some(src_gi) = find_group_idx(groups, *id) {
                            if source_groups_seen.insert(src_gi) {
                                infos[gi].deps.push(GroupDep {
                                    source_group: src_gi,
                                    input_idx: inp_idx,
                                });
                                infos[src_gi].consumers.push(gi);
                            }
                        }
                    }
                }
                _ => {
                    let source_atom: AtomId = input_ref.resolve(0, 0);
                    if let Some(src_gi) = find_group_idx(groups, source_atom) {
                        infos[gi].deps.push(GroupDep {
                            source_group: src_gi,
                            input_idx: inp_idx,
                        });
                        infos[src_gi].consumers.push(gi);
                    }
                }
            }
        }
    }

    infos
}

/// Binary search for the group containing an atom.
fn find_group_idx(groups: &[AtomGroup], id: AtomId) -> Option<usize> {
    let idx = groups.partition_point(|g| g.base_id.0 <= id.0);
    if idx == 0 {
        return None;
    }
    let gi = idx - 1;
    if groups[gi].contains(id) {
        Some(gi)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Fusion analysis
// ---------------------------------------------------------------------------

/// Determines whether `consumer` can fuse (inline) `producer`.
///
/// Fusion criteria:
/// - Producer has exactly one consumer (this consumer).
/// - Producer is a pointwise op (not a reduction, not a literal).
/// - Same `count` and same `sym_dims`.
/// - The consumer reads the producer with stride-1 Affine addressing
///   (i.e., consumer atom i reads producer atom i).
fn can_fuse_pointwise(
    groups: &[AtomGroup],
    infos: &[GroupInfo],
    consumer_gi: usize,
    producer_gi: usize,
    input_idx: usize,
) -> bool {
    let consumer = &groups[consumer_gi];
    let producer = &groups[producer_gi];

    // Producer must have exactly one consumer.
    if infos[producer_gi].consumers.len() != 1 {
        return false;
    }

    // Producer must not be a literal (literals are always external buffers).
    if matches!(producer.op, ScalarOp::Literal(_)) {
        return false;
    }

    // Producer must not be a reduction (reductions force materialization).
    if producer.op.is_reduce() {
        return false;
    }

    // Same count.
    if consumer.count != producer.count {
        return false;
    }

    // Same sym_dims.
    if consumer.sym_dims != producer.sym_dims {
        return false;
    }

    // The consumer must read the producer with stride-1 Affine
    // (i.e., consumer atom i reads producer atom i).
    let input_ref = &consumer.inputs[input_idx];
    match input_ref {
        InputRef::Affine { base, stride } => {
            // Must start at producer's base and stride 1.
            *base == producer.base_id && *stride == 1
        }
        InputRef::Explicit(ids) => {
            // Can fuse if the Explicit is effectively identity: ids[i] = producer.base_id + i
            ids.len() == producer.count as usize
                && ids
                    .iter()
                    .enumerate()
                    .all(|(i, id)| *id == producer.base_id.offset(i as u32))
        }
        _ => false,
    }
}

/// Determines whether a reduction group can absorb a producer into its k-loop.
///
/// The producer must:
/// - Have exactly one consumer (this reduction).
/// - Not be a literal or reduction itself.
/// - Be referenced via SymAffine addressing from the reduction.
fn can_fuse_into_reduction(
    groups: &[AtomGroup],
    infos: &[GroupInfo],
    reduce_gi: usize,
    producer_gi: usize,
    input_idx: usize,
) -> bool {
    let reduce_group = &groups[reduce_gi];
    let producer = &groups[producer_gi];

    // Reduction must actually be a reduce op.
    if !reduce_group.op.is_reduce() {
        return false;
    }

    // Producer must have exactly one consumer.
    if infos[producer_gi].consumers.len() != 1 {
        return false;
    }

    // Producer must not be a literal or reduction.
    if matches!(producer.op, ScalarOp::Literal(_)) || producer.op.is_reduce() {
        return false;
    }

    // The reduction must read the producer via SymAffine.
    let input_ref = &reduce_group.inputs[input_idx];
    matches!(input_ref, InputRef::SymAffine { .. })
}

// ---------------------------------------------------------------------------
// Kernel IR
// ---------------------------------------------------------------------------

/// Index into a kernel's value table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VReg(pub u16);

/// Index into the buffer pointer table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(pub u32);

/// An operation in the kernel's outer loop body.
#[derive(Debug, Clone)]
pub enum KOp {
    /// Load from buffer: buf[base + stride * i]
    Load {
        buffer: BufferId,
        base_offset: u32,
        stride: i32,
    },
    /// Broadcast load: buf[offset] (same value for all i)
    BroadcastLoad { buffer: BufferId, offset: u32 },
    /// Modular load: buf[i % modulus]. For broadcast/tiling patterns.
    ModLoad { buffer: BufferId, modulus: u32 },
    /// Index table load: buf[offsets[i]]. For irregular Explicit patterns.
    TableLoad { buffer: BufferId, offsets: Vec<u32> },
    /// Literal constant.
    Literal(f32),
    /// Binary scalar op.
    Binary {
        op: ScalarBinOp,
        a: VReg,
        b: VReg,
        compute_dt: DType,
    },
    /// Unary scalar op.
    Unary {
        op: ScalarUnaryOp,
        x: VReg,
        compute_dt: DType,
    },
    /// Ternary select: cond ? x : y
    Select {
        cond: VReg,
        x: VReg,
        y: VReg,
        compute_dt: DType,
    },
    /// Identity / cast.
    Cast { x: VReg, to: DType },
    /// Reduction result (references KernelReduction by index in the kernel).
    ReduceResult(usize),
}

/// An operation inside a reduction's k-loop.
#[derive(Debug, Clone)]
pub enum KReduceOp {
    /// Load from buffer: buf[base + stride_i * i + stride_k * k]
    SymLoad {
        buffer: BufferId,
        base_offset: u32,
        stride_i: i32,
        stride_k: i32,
    },
    /// Broadcast load from buffer (k-invariant, i-invariant).
    BroadcastLoad { buffer: BufferId, offset: u32 },
    /// Load a value from the outer scope (computed before the reduction loop).
    OuterRef(VReg),
    /// Literal.
    Literal(f32),
    /// Binary op on body-local values.
    Binary {
        op: ScalarBinOp,
        a: u16,
        b: u16,
        compute_dt: DType,
    },
    /// Unary op on a body-local value.
    Unary {
        op: ScalarUnaryOp,
        x: u16,
        compute_dt: DType,
    },
    /// Cast a body-local value.
    Cast { x: u16, to: DType },
}

/// A reduction within a kernel.
#[derive(Debug, Clone)]
pub struct KernelReduction {
    /// Sum or Max.
    pub is_sum: bool,
    /// Number of iterations.
    pub bound: u64,
    /// Precision for accumulation.
    pub compute_dt: DType,
    /// Output precision.
    pub output_dt: DType,
    /// Ops inside the k-loop body.
    pub body: Vec<KReduceOp>,
    /// Which body value (index) to accumulate.
    pub body_result: u16,
}

/// A compiled kernel plan — a fused computation producing one buffer.
#[derive(Debug)]
pub struct KernelPlan {
    /// Output buffer.
    pub output: BufferId,
    /// Offset within the output buffer where this kernel writes.
    /// Non-zero when the output group shares a buffer with other groups.
    pub output_offset: u32,
    /// Number of output elements per sym_dim iteration.
    pub extent: u32,
    /// Ops in the outer loop body (executed per element).
    pub ops: Vec<KOp>,
    /// Reductions (referenced by KOp::ReduceResult).
    pub reductions: Vec<KernelReduction>,
    /// Which VReg is the final value to store.
    pub result: VReg,
    /// Input buffers (for dependency tracking).
    pub input_buffers: Vec<BufferId>,
}

// ---------------------------------------------------------------------------
// Buffer assignment
// ---------------------------------------------------------------------------

/// A buffer in the execution layout.
#[derive(Debug)]
pub struct BufferInfo {
    pub id: BufferId,
    /// Which group's atoms this buffer stores (base_id, count).
    pub base_atom: AtomId,
    pub count: u32,
    /// Whether this buffer is pre-filled with constant data.
    pub is_constant: bool,
}

/// The full compilation plan.
#[derive(Debug)]
pub struct CompilationPlan {
    pub buffers: Vec<BufferInfo>,
    pub kernels: Vec<KernelPlan>,
    /// Maps atom base_id → buffer id, for groups that have buffers.
    pub atom_to_buffer: HashMap<u32, BufferId>,
    /// Constant data to pre-fill buffers. Maps buffer id → (offset, value) pairs.
    pub constants: HashMap<BufferId, Vec<(u32, f32)>>,
    /// Which buffers correspond to graph outputs.
    pub output_buffers: Vec<BufferId>,
}

// ---------------------------------------------------------------------------
// Plan builder
// ---------------------------------------------------------------------------

pub fn build_plan(lower: &LowerResult) -> CompilationPlan {
    let graph = &lower.graph;
    let groups = graph.groups();
    let n = groups.len();

    // Step 1: Build dependency graph.
    let mut infos = build_group_graph(graph);

    // Step 2: Mark materialization boundaries.
    // Graph output atoms → find their groups.
    let mut output_groups: HashSet<usize> = HashSet::new();
    for &out_atom in &graph.outputs {
        if let Some(gi) = find_group_idx(groups, out_atom) {
            output_groups.insert(gi);
        }
    }

    let mut mat_reason = [0usize; 7]; // literal, output, multi_consumer, reduction, unfusable_pw, unfusable_red, no_input_idx
    let mut fused_count = 0usize;

    for gi in 0..n {
        let group = &groups[gi];

        // Literals always materialize (they're input data).
        if matches!(group.op, ScalarOp::Literal(_)) {
            infos[gi].must_materialize = true;
            mat_reason[0] += 1;
            continue;
        }

        // Graph outputs must materialize.
        if output_groups.contains(&gi) {
            infos[gi].must_materialize = true;
            mat_reason[1] += 1;
            continue;
        }

        // Multi-consumer groups must materialize.
        if infos[gi].consumers.len() != 1 {
            infos[gi].must_materialize = true;
            mat_reason[2] += 1;
            continue;
        }

        // Reductions always materialize their output.
        if group.op.is_reduce() {
            infos[gi].must_materialize = true;
            mat_reason[3] += 1;
            continue;
        }

        // Check if the single consumer can actually fuse us.
        let consumer_gi = infos[gi].consumers[0];
        // Find which input of the consumer references us.
        let input_idx = infos[consumer_gi]
            .deps
            .iter()
            .find(|d| d.source_group == gi)
            .map(|d| d.input_idx);

        if let Some(inp_idx) = input_idx {
            let consumer_is_reduce = groups[consumer_gi].op.is_reduce();
            let fusable = if consumer_is_reduce {
                can_fuse_into_reduction(groups, &infos, consumer_gi, gi, inp_idx)
            } else {
                can_fuse_pointwise(groups, &infos, consumer_gi, gi, inp_idx)
            };
            if !fusable {
                infos[gi].must_materialize = true;
                if consumer_is_reduce {
                    mat_reason[5] += 1;
                } else {
                    mat_reason[4] += 1;
                }
            } else {
                fused_count += 1;
            }
        } else {
            // Shouldn't happen, but be safe.
            infos[gi].must_materialize = true;
            mat_reason[6] += 1;
        }
    }

    eprintln!(
        "[v10 plan] {} groups: {} fused, materialize reasons: literal={} output={} multi_consumer={} reduce={} unfusable_pw={} unfusable_red={} no_idx={}",
        n,
        fused_count,
        mat_reason[0],
        mat_reason[1],
        mat_reason[2],
        mat_reason[3],
        mat_reason[4],
        mat_reason[5],
        mat_reason[6]
    );

    // Step 3: Identify groups that must share a contiguous buffer because
    // a SymAffine input spans them. Use union-find to group them.
    let mut shared_buf_parent: Vec<usize> = (0..n).collect(); // union-find

    fn uf_find(parent: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while parent[r] != r {
            r = parent[r];
        }
        let mut c = x;
        while parent[c] != r {
            let next = parent[c];
            parent[c] = r;
            c = next;
        }
        r
    }
    fn uf_union(parent: &mut [usize], a: usize, b: usize) {
        let ra = uf_find(parent, a);
        let rb = uf_find(parent, b);
        if ra != rb {
            parent[rb] = ra;
        }
    }

    for (gi, group) in groups.iter().enumerate() {
        for input_ref in &group.inputs {
            if let InputRef::SymAffine {
                base,
                stride_i,
                stride_k,
            } = input_ref
            {
                // Find all groups this SymAffine spans.
                if !group.reduce_dims.is_empty() {
                    let rd = group.reduce_dims[0];
                    let bound = graph.sym_dim_bounds.get(&rd).copied().unwrap_or(1);
                    let mut prev_gi: Option<usize> = None;
                    for k in 0..bound as u32 {
                        let atom = input_ref.resolve(0, k);
                        if let Some(src_gi) = find_group_idx(groups, atom) {
                            if let Some(pg) = prev_gi {
                                uf_union(&mut shared_buf_parent, pg, src_gi);
                            }
                            prev_gi = Some(src_gi);
                        }
                    }
                }
            }
        }
    }

    // Step 3a: Also union groups that belong to the same tensor (via tensor_map).
    // This ensures output tensors spanning multiple groups get a single buffer.
    for tam in lower.tensor_map.values() {
        let start_atom = tam.base_id.0;
        let end_atom = start_atom + tam.count;
        let mut groups_in_tensor: Vec<usize> = Vec::new();
        for (gi, group) in groups.iter().enumerate() {
            let group_start = group.base_id.0;
            let group_end = group_start + group.count;
            // Group overlaps with tensor range.
            if group_start < end_atom && group_end > start_atom {
                groups_in_tensor.push(gi);
            }
        }
        for i in 1..groups_in_tensor.len() {
            uf_union(
                &mut shared_buf_parent,
                groups_in_tensor[0],
                groups_in_tensor[i],
            );
        }
    }

    // Step 3b: Assign buffers. Groups in the same union-find set share a buffer.
    let mut buffers: Vec<BufferInfo> = Vec::new();
    let mut atom_to_buffer: HashMap<u32, BufferId> = HashMap::new();
    let mut group_to_buffer: HashMap<usize, BufferId> = HashMap::new();
    let mut root_to_buffer: HashMap<usize, BufferId> = HashMap::new();
    // Track per-buffer: atom offset mapping (buf_id → (global_base_atom, total_count))
    let mut buf_atom_ranges: HashMap<BufferId, (u32, u32)> = HashMap::new(); // buf → (min_atom, max_atom_exclusive)

    for gi in 0..n {
        if !infos[gi].must_materialize {
            continue;
        }
        let group = &groups[gi];
        let root = uf_find(&mut shared_buf_parent, gi);

        if let Some(&buf_id) = root_to_buffer.get(&root) {
            // Extend existing shared buffer.
            let range = buf_atom_ranges.get_mut(&buf_id).unwrap();
            let atom_start = group.base_id.0;
            let atom_end = atom_start + group.count;
            range.0 = range.0.min(atom_start);
            range.1 = range.1.max(atom_end);
            group_to_buffer.insert(gi, buf_id);
        } else {
            // New buffer.
            let buf_id = BufferId(buffers.len() as u32);
            let atom_start = group.base_id.0;
            let atom_end = atom_start + group.count;
            buffers.push(BufferInfo {
                id: buf_id,
                base_atom: group.base_id,
                count: group.count,
                is_constant: matches!(group.op, ScalarOp::Literal(_)),
            });
            atom_to_buffer.insert(group.base_id.0, buf_id);
            group_to_buffer.insert(gi, buf_id);
            root_to_buffer.insert(root, buf_id);
            buf_atom_ranges.insert(buf_id, (atom_start, atom_end));
        }
    }

    // Fixup shared buffers: update base_atom and count to cover the full range,
    // and update atom_to_buffer for all group bases.
    for (&buf_id, &(min_atom, max_atom)) in &buf_atom_ranges {
        let buf = &mut buffers[buf_id.0 as usize];
        buf.base_atom = AtomId(min_atom);
        buf.count = max_atom - min_atom;
        buf.is_constant = false; // shared buffers are computed, not constant
    }
    // Re-populate atom_to_buffer for ALL materialized groups.
    atom_to_buffer.clear();
    for gi in 0..n {
        if let Some(&buf_id) = group_to_buffer.get(&gi) {
            atom_to_buffer.insert(groups[gi].base_id.0, buf_id);
        }
    }

    // Step 4: Pre-fill constant buffers from numeric_overrides.
    let mut constants: HashMap<BufferId, Vec<(u32, f32)>> = HashMap::new();
    for (gi, group) in groups.iter().enumerate() {
        if !matches!(group.op, ScalarOp::Literal(_)) {
            continue;
        }
        let Some(&buf_id) = group_to_buffer.get(&gi) else {
            continue;
        };
        let mut pairs = Vec::new();
        let buf = &buffers[buf_id.0 as usize];
        let buf_base = buf.base_atom.0;
        for i in 0..group.count {
            let atom_idx = group.base_id.0 + i;
            let buf_offset = atom_idx - buf_base;
            if let Some(scalar) = lower.numeric_overrides.get(&atom_idx) {
                pairs.push((buf_offset, scalar.to_f64() as f32));
            } else {
                // Use the literal's default value.
                if let ScalarOp::Literal(ref s) = group.op {
                    pairs.push((buf_offset, s.to_f64() as f32));
                }
            }
        }
        if !pairs.is_empty() {
            constants.insert(buf_id, pairs);
        }
    }

    // Step 5: Build kernels for each non-literal materialized group.
    let mut kernels: Vec<KernelPlan> = Vec::new();

    for gi in 0..n {
        if !infos[gi].must_materialize {
            continue;
        }
        if matches!(groups[gi].op, ScalarOp::Literal(_)) {
            continue; // Literals don't need kernels, they're pre-filled.
        }

        let kernel = build_kernel(
            graph,
            groups,
            &infos,
            &group_to_buffer,
            &atom_to_buffer,
            &buffers,
            gi,
        );
        kernels.push(kernel);
    }

    // Step 6: Topologically sort kernels by their dependencies.
    let kernels = toposort_kernels(kernels);

    // Step 7: Identify output buffers.
    let output_buffers: Vec<BufferId> = output_groups
        .iter()
        .filter_map(|&gi| group_to_buffer.get(&gi).copied())
        .collect();

    CompilationPlan {
        buffers,
        kernels,
        atom_to_buffer,
        constants,
        output_buffers,
    }
}

// ---------------------------------------------------------------------------
// Kernel construction (the interesting part)
// ---------------------------------------------------------------------------

fn build_kernel(
    graph: &NanoGraph,
    groups: &[AtomGroup],
    infos: &[GroupInfo],
    group_to_buffer: &HashMap<usize, BufferId>,
    atom_to_buffer: &HashMap<u32, BufferId>,
    buffers: &[BufferInfo],
    output_gi: usize,
) -> KernelPlan {
    let output_group = &groups[output_gi];
    let output_buffer = group_to_buffer[&output_gi];
    let extent = output_group.count;

    let mut builder = KernelBuilder::new(output_gi, buffers);

    if output_group.op.is_reduce() {
        // Build a reduction kernel.
        build_reduction_kernel(
            &mut builder,
            graph,
            groups,
            infos,
            group_to_buffer,
            atom_to_buffer,
            output_gi,
        );
    } else {
        // Build a pointwise kernel (possibly with fused producers).
        let result = emit_group_value(
            &mut builder,
            graph,
            groups,
            infos,
            group_to_buffer,
            atom_to_buffer,
            output_gi,
        );
        builder.result = Some(result);
    }

    let input_buffers: Vec<BufferId> = builder.input_buffers.iter().copied().collect();
    let output_offset = builder.buf_offset(output_buffer, output_group.base_id);

    KernelPlan {
        output: output_buffer,
        output_offset,
        extent,
        ops: builder.ops,
        reductions: builder.reductions,
        result: builder.result.unwrap(),
        input_buffers,
    }
}

struct KernelBuilder {
    ops: Vec<KOp>,
    reductions: Vec<KernelReduction>,
    next_vreg: u16,
    result: Option<VReg>,
    input_buffers: HashSet<BufferId>,
    /// Cache: group index → VReg, to avoid recomputing fused groups.
    group_cache: HashMap<usize, VReg>,
    /// The output group of this kernel — should NOT short-circuit to a load.
    kernel_root: usize,
    /// Buffer base atom for each BufferId (for offset computation).
    buf_base_atoms: HashMap<BufferId, u32>,
}

impl KernelBuilder {
    fn new(kernel_root: usize, buffers: &[BufferInfo]) -> Self {
        let buf_base_atoms: HashMap<BufferId, u32> =
            buffers.iter().map(|b| (b.id, b.base_atom.0)).collect();
        Self {
            ops: Vec::new(),
            reductions: Vec::new(),
            next_vreg: 0,
            result: None,
            input_buffers: HashSet::new(),
            group_cache: HashMap::new(),
            kernel_root,
            buf_base_atoms,
        }
    }

    fn buf_offset(&self, buf_id: BufferId, atom: AtomId) -> u32 {
        atom.0 - self.buf_base_atoms[&buf_id]
    }

    fn push_op(&mut self, op: KOp) -> VReg {
        let vreg = VReg(self.next_vreg);
        self.next_vreg += 1;
        self.ops.push(op);
        vreg
    }
}

/// Recursively emit the computation for a group, inlining fused producers.
/// Returns the VReg holding the result.
fn emit_group_value(
    builder: &mut KernelBuilder,
    graph: &NanoGraph,
    groups: &[AtomGroup],
    infos: &[GroupInfo],
    group_to_buffer: &HashMap<usize, BufferId>,
    atom_to_buffer: &HashMap<u32, BufferId>,
    gi: usize,
) -> VReg {
    // Check cache first.
    if let Some(&vreg) = builder.group_cache.get(&gi) {
        return vreg;
    }

    let group = &groups[gi];

    // If this group has a buffer (is materialized) and it's NOT the kernel root,
    // we DON'T emit a load here — the caller (which knows the InputRef) will
    // handle the load with correct addressing. This path should not be hit
    // for materialized groups; see emit_group_input below.
    if gi != builder.kernel_root && group_to_buffer.contains_key(&gi) {
        panic!(
            "emit_group_value called on materialized group {} — use emit_group_input instead",
            gi
        );
    }

    // This group is fused or is the kernel root — emit its inputs, then compute.
    let input_vregs: Vec<VReg> = group
        .inputs
        .iter()
        .enumerate()
        .map(|(inp_idx, input_ref): (usize, &InputRef)| {
            let source_atom: AtomId = input_ref.resolve(0, 0);
            let src_gi =
                find_group_idx(groups, source_atom).expect("input ref resolves to valid group");
            emit_group_input(
                builder,
                graph,
                groups,
                infos,
                group_to_buffer,
                atom_to_buffer,
                src_gi,
                input_ref,
            )
        })
        .collect();

    let vreg = emit_scalar_op(builder, &group.op, &input_vregs);
    builder.group_cache.insert(gi, vreg);
    vreg
}

/// Emit a value for a source group, using the given InputRef for addressing.
/// If the source group is materialized, emits a load with correct addressing.
/// If fused, recursively emits the computation.
fn emit_group_input(
    builder: &mut KernelBuilder,
    graph: &NanoGraph,
    groups: &[AtomGroup],
    infos: &[GroupInfo],
    group_to_buffer: &HashMap<usize, BufferId>,
    atom_to_buffer: &HashMap<u32, BufferId>,
    src_gi: usize,
    input_ref: &InputRef,
) -> VReg {
    // Check cache first.
    if let Some(&vreg) = builder.group_cache.get(&src_gi) {
        return vreg;
    }

    // If the source group is materialized, emit a load using the InputRef addressing.
    if let Some(&buf_id) = group_to_buffer.get(&src_gi) {
        let src_group = &groups[src_gi];
        let vreg = emit_addressed_load_from_ref(builder, buf_id, src_group, input_ref);
        // Note: we do NOT cache this, because different consumers of the same
        // materialized group may use different addressing (e.g., one stride-1,
        // another broadcast). Each InputRef produces a different load.
        return vreg;
    }

    // Source is fused — recursively compute it.
    emit_group_value(
        builder,
        graph,
        groups,
        infos,
        group_to_buffer,
        atom_to_buffer,
        src_gi,
    )
}

/// Emit a load from a materialized buffer using the given InputRef for addressing.
fn emit_addressed_load_from_ref(
    builder: &mut KernelBuilder,
    buf_id: BufferId,
    src_group: &AtomGroup,
    input_ref: &InputRef,
) -> VReg {
    builder.input_buffers.insert(buf_id);
    match input_ref {
        InputRef::Broadcast(atom_id) => {
            let offset = builder.buf_offset(buf_id, *atom_id);
            builder.push_op(KOp::BroadcastLoad {
                buffer: buf_id,
                offset,
            })
        }
        InputRef::Affine { base, stride } => {
            let base_offset = builder.buf_offset(buf_id, *base);
            builder.push_op(KOp::Load {
                buffer: buf_id,
                base_offset,
                stride: *stride,
            })
        }
        InputRef::Explicit(ids) => {
            if ids.is_empty() {
                return builder.push_op(KOp::Literal(0.0));
            }

            // Check if it's a repeating pattern (broadcast tiling).
            let src_count = src_group.count as usize;
            let is_repeat = ids.len() > src_count
                && ids.chunks(src_count).all(|chunk| {
                    chunk
                        .iter()
                        .enumerate()
                        .all(|(i, id)| *id == src_group.base_id.offset(i as u32))
                });
            if is_repeat {
                return builder.push_op(KOp::ModLoad {
                    buffer: buf_id,
                    modulus: src_count as u32,
                });
            }

            // Check for affine pattern: ids[i] maps to buf_offset = base + stride * i
            let buf_base = builder.buf_base_atoms[&buf_id];
            let first_off = ids[0].0 as i64 - buf_base as i64;
            if ids.len() == 1 {
                // Single element — broadcast.
                return builder.push_op(KOp::BroadcastLoad {
                    buffer: buf_id,
                    offset: first_off as u32,
                });
            }
            let stride = ids[1].0 as i64 - ids[0].0 as i64;
            let is_affine = ids
                .iter()
                .enumerate()
                .all(|(i, id)| id.0 as i64 == ids[0].0 as i64 + stride * i as i64);
            if is_affine {
                return builder.push_op(KOp::Load {
                    buffer: buf_id,
                    base_offset: first_off as u32,
                    stride: stride as i32,
                });
            }

            // Truly irregular: emit an index table load.
            // Convert each atom ID to a buffer-relative offset.
            let offsets: Vec<u32> = ids
                .iter()
                .map(|id| (id.0 as i64 - buf_base as i64) as u32)
                .collect();
            builder.push_op(KOp::TableLoad {
                buffer: buf_id,
                offsets,
            })
        }
        InputRef::SymAffine { .. } => {
            panic!("SymAffine in non-reduction context");
        }
    }
}

/// Emit a scalar op given its input VRegs.
fn emit_scalar_op(builder: &mut KernelBuilder, op: &ScalarOp, inputs: &[VReg]) -> VReg {
    match op {
        ScalarOp::Identity {
            compute_dtype,
            output_dtype,
        } => {
            let mut v = inputs[0];
            if compute_dtype != output_dtype {
                v = builder.push_op(KOp::Cast {
                    x: v,
                    to: *output_dtype,
                });
            }
            v
        }
        ScalarOp::Binary {
            op,
            compute_dtype,
            output_dtype,
        } => {
            let v = builder.push_op(KOp::Binary {
                op: *op,
                a: inputs[0],
                b: inputs[1],
                compute_dt: *compute_dtype,
            });
            if compute_dtype != output_dtype {
                builder.push_op(KOp::Cast {
                    x: v,
                    to: *output_dtype,
                })
            } else {
                v
            }
        }
        ScalarOp::Unary {
            op,
            compute_dtype,
            output_dtype,
        } => {
            let v = builder.push_op(KOp::Unary {
                op: *op,
                x: inputs[0],
                compute_dt: *compute_dtype,
            });
            if compute_dtype != output_dtype {
                builder.push_op(KOp::Cast {
                    x: v,
                    to: *output_dtype,
                })
            } else {
                v
            }
        }
        ScalarOp::Select {
            compute_dtype,
            output_dtype,
        } => {
            let v = builder.push_op(KOp::Select {
                cond: inputs[0],
                x: inputs[1],
                y: inputs[2],
                compute_dt: *compute_dtype,
            });
            if compute_dtype != output_dtype {
                builder.push_op(KOp::Cast {
                    x: v,
                    to: *output_dtype,
                })
            } else {
                v
            }
        }
        ScalarOp::Literal(scalar) => builder.push_op(KOp::Literal(scalar.to_f64() as f32)),
        ScalarOp::ReduceSum { .. } | ScalarOp::ReduceMax { .. } => {
            panic!("emit_scalar_op called with reduce op — use build_reduction_kernel");
        }
    }
}

// ---------------------------------------------------------------------------
// Reduction kernel construction
// ---------------------------------------------------------------------------

fn build_reduction_kernel(
    builder: &mut KernelBuilder,
    graph: &NanoGraph,
    groups: &[AtomGroup],
    infos: &[GroupInfo],
    group_to_buffer: &HashMap<usize, BufferId>,
    atom_to_buffer: &HashMap<u32, BufferId>,
    reduce_gi: usize,
) {
    let reduce_group = &groups[reduce_gi];
    let (compute_dt, output_dt, is_sum) = match &reduce_group.op {
        ScalarOp::ReduceSum {
            compute_dtype,
            output_dtype,
        } => (*compute_dtype, *output_dtype, true),
        ScalarOp::ReduceMax {
            compute_dtype,
            output_dtype,
        } => (*compute_dtype, *output_dtype, false),
        _ => panic!("not a reduce group"),
    };

    assert_eq!(
        reduce_group.reduce_dims.len(),
        1,
        "single reduce dim expected"
    );
    let rd = reduce_group.reduce_dims[0];
    let bound = graph
        .sym_dim_bounds
        .get(&rd)
        .copied()
        .expect("reduce dim must have bound");

    // The reduction has one input. Check if we can fuse the source chain
    // into the reduction's inner loop.
    let input_ref = &reduce_group.inputs[0];
    let source_atom = input_ref.resolve(0, 0);
    let src_gi = find_group_idx(groups, source_atom).unwrap();

    // Try to fuse the source chain into the reduction body.
    // Don't fuse if the source group is materialized (e.g., because its SymAffine
    // spans multiple groups and they were forced to materialize).
    let reduction = if can_fuse_into_reduction(groups, infos, reduce_gi, src_gi, 0)
        && !group_to_buffer.contains_key(&src_gi)
    {
        build_fused_reduction_body(
            builder,
            graph,
            groups,
            infos,
            group_to_buffer,
            atom_to_buffer,
            reduce_gi,
            src_gi,
            bound,
            compute_dt,
            output_dt,
            is_sum,
        )
    } else {
        // Can't fuse — load from the source buffer in the k-loop.
        build_simple_reduction_body(
            builder,
            groups,
            group_to_buffer,
            reduce_gi,
            bound,
            compute_dt,
            output_dt,
            is_sum,
        )
    };

    let red_idx = builder.reductions.len();
    builder.reductions.push(reduction);

    let result = builder.push_op(KOp::ReduceResult(red_idx));
    builder.result = Some(result);
}

/// Build a reduction body that loads directly from a materialized source buffer.
fn build_simple_reduction_body(
    builder: &mut KernelBuilder,
    groups: &[AtomGroup],
    group_to_buffer: &HashMap<usize, BufferId>,
    reduce_gi: usize,
    bound: u64,
    compute_dt: DType,
    output_dt: DType,
    is_sum: bool,
) -> KernelReduction {
    let reduce_group = &groups[reduce_gi];
    let input_ref = &reduce_group.inputs[0];

    let body_op = match input_ref {
        InputRef::SymAffine {
            base,
            stride_i,
            stride_k,
        } => {
            let src_gi = find_group_idx(groups, *base).unwrap();
            let buf_id = group_to_buffer[&src_gi];
            let base_offset = builder.buf_offset(buf_id, *base);
            builder.input_buffers.insert(buf_id);
            KReduceOp::SymLoad {
                buffer: buf_id,
                base_offset,
                stride_i: *stride_i,
                stride_k: *stride_k,
            }
        }
        InputRef::Affine { base, stride } => {
            let src_gi = find_group_idx(groups, *base).unwrap();
            let buf_id = group_to_buffer[&src_gi];
            let base_offset = builder.buf_offset(buf_id, *base);
            builder.input_buffers.insert(buf_id);
            KReduceOp::SymLoad {
                buffer: buf_id,
                base_offset,
                stride_i: *stride,
                stride_k: 0,
            }
        }
        InputRef::Broadcast(atom_id) => {
            let src_gi = find_group_idx(groups, *atom_id).unwrap();
            let buf_id = group_to_buffer[&src_gi];
            let offset = builder.buf_offset(buf_id, *atom_id);
            builder.input_buffers.insert(buf_id);
            KReduceOp::BroadcastLoad {
                buffer: buf_id,
                offset,
            }
        }
        _ => panic!("unsupported InputRef in reduction"),
    };

    KernelReduction {
        is_sum,
        bound,
        compute_dt,
        output_dt,
        body: vec![body_op],
        body_result: 0,
    }
}

/// Build a reduction body with fused pointwise producers inlined into the k-loop.
///
/// Walks backwards from the reduction's source group, pulling in single-consumer
/// pointwise producers. Their computations become inline ops in the reduction body.
///
/// **K-aware fusion:** When the reduction's SymAffine spans multiple sibling groups
/// across k, the fused chain is built from the k=0 representative. For each external
/// input of the chain, we sample the corresponding sibling groups across k to discover
/// how the input address shifts with k (the `stride_k`). This is general — it works
/// for any pattern where sibling groups' inputs shift affinely across k.
fn build_fused_reduction_body(
    builder: &mut KernelBuilder,
    graph: &NanoGraph,
    groups: &[AtomGroup],
    infos: &[GroupInfo],
    group_to_buffer: &HashMap<usize, BufferId>,
    atom_to_buffer: &HashMap<u32, BufferId>,
    reduce_gi: usize,
    source_gi: usize,
    bound: u64,
    compute_dt: DType,
    output_dt: DType,
    is_sum: bool,
) -> KernelReduction {
    let reduce_group = &groups[reduce_gi];
    let reduce_input_ref = &reduce_group.inputs[0];

    // Collect the chain of groups to fuse into the reduction body.
    // Walk backwards from source_gi (the k=0 representative), collecting
    // single-consumer pointwise producers.
    let fused_chain = collect_reduction_chain(groups, infos, group_to_buffer, source_gi);
    let chain_set: HashSet<usize> = fused_chain.iter().copied().collect();

    // Build a mapping: for each k value, map k=0 chain group → k=K sibling group.
    // We do this by tracing from the k=K source group along the same structural path.
    // For efficiency, we only sample k=0 and k=1 (to discover affine stride_k).
    let k0_to_k1_sibling = build_sibling_map(
        groups,
        infos,
        &fused_chain,
        &chain_set,
        reduce_input_ref,
        bound,
    );

    // Build the reduction body ops.
    // The chain is in dependency order (producers first, source_gi last).
    let mut body: Vec<KReduceOp> = Vec::new();
    let mut chain_vreg: HashMap<usize, u16> = HashMap::new();

    for &chain_gi in &fused_chain {
        let chain_group = &groups[chain_gi];

        // Emit loads for this group's inputs.
        let input_body_vals: Vec<u16> = chain_group
            .inputs
            .iter()
            .enumerate()
            .map(|(inp_idx, input_ref): (usize, &InputRef)| {
                let source_atom: AtomId = input_ref.resolve(0, 0);
                let inp_src_gi = find_group_idx(groups, source_atom).unwrap();

                // Is the input another fused group in our chain?
                if let Some(&bv) = chain_vreg.get(&inp_src_gi) {
                    let idx = body.len() as u16;
                    body.push(KReduceOp::Cast {
                        x: bv,
                        to: compute_dt,
                    });
                    return idx;
                }

                // External buffer load — use k-aware addressing.
                emit_k_aware_reduce_load(
                    builder,
                    &mut body,
                    groups,
                    group_to_buffer,
                    chain_gi,
                    inp_idx,
                    input_ref,
                    &k0_to_k1_sibling,
                    bound,
                )
            })
            .collect();

        // Emit the scalar op.
        let result_idx =
            emit_reduce_body_op(&mut body, &chain_group.op, &input_body_vals, compute_dt);
        chain_vreg.insert(chain_gi, result_idx);
    }

    let body_result = *chain_vreg.get(&source_gi).unwrap();

    KernelReduction {
        is_sum,
        bound,
        compute_dt,
        output_dt,
        body,
        body_result,
    }
}

/// Build a mapping from k=0 chain group → k=1 sibling group.
///
/// Starting from the k=1 source group (found via the reduce's SymAffine),
/// we trace the same structural dep path to find each chain group's sibling.
/// Returns None for a chain group if no sibling exists (single-group SymAffine).
fn build_sibling_map(
    groups: &[AtomGroup],
    infos: &[GroupInfo],
    fused_chain: &[usize],
    chain_set: &HashSet<usize>,
    reduce_input_ref: &InputRef,
    bound: u64,
) -> HashMap<usize, usize> {
    let mut k0_to_k1: HashMap<usize, usize> = HashMap::new();

    if bound < 2 {
        return k0_to_k1; // No k=1 to sample
    }

    // Find the source group at k=1.
    let k1_source_atom = reduce_input_ref.resolve(0, 1);
    let k1_source_gi = match find_group_idx(groups, k1_source_atom) {
        Some(gi) => gi,
        None => return k0_to_k1,
    };

    // The source_gi at k=0 is the last element of fused_chain.
    let k0_source_gi = *fused_chain.last().unwrap();

    if k1_source_gi == k0_source_gi {
        // Same group for k=0 and k=1 — SymAffine stays within one group.
        // No sibling mapping needed; the existing InputRef handles it directly.
        return k0_to_k1;
    }

    k0_to_k1.insert(k0_source_gi, k1_source_gi);

    // Trace deeper: for each chain group (walking from source backwards to leaves),
    // find its k=1 sibling by matching structural deps from the k=1 source.
    // The chain is in dependency order (leaves first, source last), so we walk
    // it in reverse (source → leaves) to propagate sibling info downward.
    for &chain_gi in fused_chain.iter().rev() {
        if chain_gi == k0_source_gi {
            continue; // Already mapped above
        }
        let chain_group = &groups[chain_gi];

        // Find which chain group consumes us (there's exactly one since we're
        // single-consumer and in the chain).
        let consumer_gi = infos[chain_gi].consumers[0];
        if !chain_set.contains(&consumer_gi) {
            continue; // Consumer is outside chain (shouldn't happen for non-source)
        }

        // Get the consumer's k=1 sibling.
        let consumer_k1 = match k0_to_k1.get(&consumer_gi) {
            Some(&gi) => gi,
            None => continue, // Can't trace further
        };

        // Find which input slot of the consumer references chain_gi.
        let consumer_k0_group = &groups[consumer_gi];
        let mut found = false;
        for (inp_idx, inp_ref) in consumer_k0_group.inputs.iter().enumerate() {
            let resolved = inp_ref.resolve(0, 0);
            let resolved_gi = find_group_idx(groups, resolved);
            if resolved_gi == Some(chain_gi) {
                // The consumer's k=1 sibling should have the same input structure.
                let consumer_k1_group = &groups[consumer_k1];
                if inp_idx < consumer_k1_group.inputs.len() {
                    let k1_inp_ref = &consumer_k1_group.inputs[inp_idx];
                    let k1_atom = k1_inp_ref.resolve(0, 0);
                    if let Some(k1_gi) = find_group_idx(groups, k1_atom) {
                        k0_to_k1.insert(chain_gi, k1_gi);
                        found = true;
                    }
                }
                break;
            }
        }
    }

    k0_to_k1
}

/// Emit a k-aware load for an external input in a fused reduction body.
///
/// Discovers how the input address varies across k by comparing the k=0 and k=1
/// sibling groups' InputRefs for the same input slot. The result is a SymLoad
/// with the discovered stride_k, or a BroadcastLoad if the input is k-invariant.
fn emit_k_aware_reduce_load(
    builder: &mut KernelBuilder,
    body: &mut Vec<KReduceOp>,
    groups: &[AtomGroup],
    group_to_buffer: &HashMap<usize, BufferId>,
    chain_gi: usize,
    inp_idx: usize,
    input_ref: &InputRef,
    k0_to_k1: &HashMap<usize, usize>,
    bound: u64,
) -> u16 {
    let idx = body.len() as u16;
    let chain_group = &groups[chain_gi];

    // Resolve the k=0 atom for this input.
    let k0_atom = input_ref.resolve(0, 0);
    let src_gi = find_group_idx(groups, k0_atom).unwrap();

    // Determine stride_i from the k=0 InputRef.
    let stride_i = match input_ref {
        InputRef::Broadcast(_) => 0,
        InputRef::Affine { stride, .. } => *stride,
        InputRef::Explicit(ids) => {
            // Try to detect affine stride from the explicit list.
            if ids.len() >= 2 {
                (ids[1].0 as i64 - ids[0].0 as i64) as i32
            } else {
                0
            }
        }
        InputRef::SymAffine {
            stride_i, stride_k, ..
        } => {
            // This input is itself a SymAffine — it already carries k-addressing.
            // Emit it directly using the SymAffine's own strides.
            let buf_id = group_to_buffer[&src_gi];
            let base_offset = builder.buf_offset(buf_id, input_ref.resolve(0, 0));
            builder.input_buffers.insert(buf_id);
            body.push(KReduceOp::SymLoad {
                buffer: buf_id,
                base_offset,
                stride_i: *stride_i,
                stride_k: *stride_k,
            });
            return idx;
        }
    };

    // Now discover stride_k by comparing k=0 and k=1 sibling groups.
    let stride_k = if let Some(&k1_chain_gi) = k0_to_k1.get(&chain_gi) {
        let k1_group = &groups[k1_chain_gi];
        if inp_idx < k1_group.inputs.len() {
            let k1_input_ref = &k1_group.inputs[inp_idx];
            let k1_atom = k1_input_ref.resolve(0, 0);
            // stride_k = difference in resolved atoms between k=1 and k=0
            k1_atom.0 as i64 - k0_atom.0 as i64
        } else {
            0 // Structural mismatch, treat as k-invariant
        }
    } else {
        0 // No sibling — single group, k-invariant
    };

    // Find the buffer for this source.
    let buf_id = group_to_buffer[&src_gi];
    let base_offset = builder.buf_offset(buf_id, k0_atom);
    builder.input_buffers.insert(buf_id);

    if stride_i == 0 && stride_k == 0 {
        // Fully broadcast (k-invariant and i-invariant).
        body.push(KReduceOp::BroadcastLoad {
            buffer: buf_id,
            offset: base_offset,
        });
    } else {
        body.push(KReduceOp::SymLoad {
            buffer: buf_id,
            base_offset,
            stride_i,
            stride_k: stride_k as i32,
        });
    }

    idx
}

/// Collect the chain of groups to fuse into a reduction body.
/// Returns groups in dependency order (leaves first).
fn collect_reduction_chain(
    groups: &[AtomGroup],
    infos: &[GroupInfo],
    group_to_buffer: &HashMap<usize, BufferId>,
    source_gi: usize,
) -> Vec<usize> {
    let mut chain: Vec<usize> = Vec::new();
    let mut visited: HashSet<usize> = HashSet::new();
    let mut stack: Vec<usize> = vec![source_gi];

    // DFS to find all fusable producers.
    while let Some(gi) = stack.pop() {
        if visited.contains(&gi) {
            continue;
        }
        visited.insert(gi);

        let group = &groups[gi];
        // Don't recurse into materialized groups (they're external loads).
        if group_to_buffer.contains_key(&gi) && gi != source_gi {
            continue;
        }
        // Don't recurse into literals.
        if matches!(group.op, ScalarOp::Literal(_)) {
            continue;
        }

        // Add producers to the stack.
        for dep in &infos[gi].deps {
            let prod_gi = dep.source_group;
            if !visited.contains(&prod_gi)
                && !group_to_buffer.contains_key(&prod_gi)
                && !matches!(groups[prod_gi].op, ScalarOp::Literal(_))
                && infos[prod_gi].consumers.len() == 1
            {
                stack.push(prod_gi);
            }
        }

        chain.push(gi);
    }

    // Reverse to get dependency order (leaves first).
    chain.reverse();
    // Actually we need topological sort within the chain.
    toposort_chain(groups, infos, &chain)
}

/// Topologically sort a set of group indices based on their dependencies.
fn toposort_chain(groups: &[AtomGroup], infos: &[GroupInfo], chain: &[usize]) -> Vec<usize> {
    let chain_set: HashSet<usize> = chain.iter().copied().collect();
    let mut in_degree: HashMap<usize, usize> = HashMap::new();
    for &gi in chain {
        in_degree.insert(gi, 0);
    }
    for &gi in chain {
        for dep in &infos[gi].deps {
            if chain_set.contains(&dep.source_group) {
                *in_degree.entry(gi).or_default() += 1;
            }
        }
    }

    let mut queue: VecDeque<usize> = in_degree
        .iter()
        .filter(|(_, deg)| **deg == 0)
        .map(|(&gi, _)| gi)
        .collect();
    let mut sorted = Vec::new();

    while let Some(gi) = queue.pop_front() {
        sorted.push(gi);
        for &consumer_gi in &infos[gi].consumers {
            if chain_set.contains(&consumer_gi) {
                if let Some(deg) = in_degree.get_mut(&consumer_gi) {
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(consumer_gi);
                    }
                }
            }
        }
    }

    sorted
}

/// Emit a scalar op in the reduction body.
fn emit_reduce_body_op(
    body: &mut Vec<KReduceOp>,
    op: &ScalarOp,
    inputs: &[u16],
    compute_dt: DType,
) -> u16 {
    let idx = body.len() as u16;
    match op {
        ScalarOp::Binary {
            op, compute_dtype, ..
        } => {
            body.push(KReduceOp::Binary {
                op: *op,
                a: inputs[0],
                b: inputs[1],
                compute_dt: *compute_dtype,
            });
        }
        ScalarOp::Unary {
            op, compute_dtype, ..
        } => {
            body.push(KReduceOp::Unary {
                op: *op,
                x: inputs[0],
                compute_dt: *compute_dtype,
            });
        }
        ScalarOp::Identity { output_dtype, .. } => {
            body.push(KReduceOp::Cast {
                x: inputs[0],
                to: *output_dtype,
            });
        }
        ScalarOp::Literal(s) => {
            body.push(KReduceOp::Literal(s.to_f64() as f32));
        }
        _ => panic!("unsupported op in reduction body: {:?}", op),
    }
    idx
}

// ---------------------------------------------------------------------------
// Topological sort of kernels
// ---------------------------------------------------------------------------

fn toposort_kernels(kernels: Vec<KernelPlan>) -> Vec<KernelPlan> {
    if kernels.len() <= 1 {
        return kernels;
    }

    // Build output_buffer → kernel indices map (multiple kernels may write to same buffer).
    let mut output_to_idx: HashMap<BufferId, Vec<usize>> = HashMap::new();
    for (ki, k) in kernels.iter().enumerate() {
        output_to_idx.entry(k.output).or_default().push(ki);
    }

    // Build adjacency.
    let n = kernels.len();
    let mut in_degree = vec![0usize; n];
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (ki, k) in kernels.iter().enumerate() {
        for buf in &k.input_buffers {
            if let Some(producer_kis) = output_to_idx.get(buf) {
                for &producer_ki in producer_kis {
                    if producer_ki != ki {
                        in_degree[ki] += 1;
                        dependents[producer_ki].push(ki);
                    }
                }
            }
        }
    }

    let mut queue: VecDeque<usize> = in_degree
        .iter()
        .enumerate()
        .filter(|(_, d)| **d == 0)
        .map(|(i, _)| i)
        .collect();
    let mut order: Vec<usize> = Vec::with_capacity(n);

    while let Some(ki) = queue.pop_front() {
        order.push(ki);
        for &dep_ki in &dependents[ki] {
            in_degree[dep_ki] -= 1;
            if in_degree[dep_ki] == 0 {
                queue.push_back(dep_ki);
            }
        }
    }

    // Reorder kernels.
    let mut result: Vec<KernelPlan> = Vec::with_capacity(n);
    // Need to move out of the original vec by index.
    let mut slots: Vec<Option<KernelPlan>> = kernels.into_iter().map(Some).collect();
    for ki in order {
        result.push(slots[ki].take().unwrap());
    }
    result
}
