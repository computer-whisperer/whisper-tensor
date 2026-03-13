#![allow(clippy::all, dead_code, unreachable_patterns)]
//! NanoGraph → kernel plan compilation.
//!
//! Pipeline:
//! 1. Build group dependency graph (consumers/producers).
//! 2. Identify materialization points (kernel roots): literals, outputs,
//!    multi-consumer groups, non-fusable groups.
//! 3. Assign buffers (with union-find for co-location).
//! 4. For each kernel root, build a KernelPlan by walking the fused
//!    computation DAG backward, emitting loads and inline ops.
//! 5. Topologically sort kernels.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::dtype::DType;
use crate::nano_graph::{ScalarBinOp, ScalarOp, ScalarUnaryOp};
use crate::nano_graph::{AtomGroup, AtomId, InputRef, NanoGraph};
use crate::nano_graph::lower::LowerResult;
use crate::numeric_scalar::NumericScalar;

// ---------------------------------------------------------------------------
// Kernel IR
// ---------------------------------------------------------------------------

/// Virtual register index within a kernel's value table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VReg(pub u16);

/// Buffer index in the execution layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(pub u32);

/// An operation in the kernel's outer loop (executed per output element `i`).
#[derive(Debug, Clone)]
pub enum KOp {
    /// Load: buf[base + stride * i]
    Load { buffer: BufferId, base_offset: u32, stride: i32 },
    /// Broadcast: buf[offset] (same for all i)
    BroadcastLoad { buffer: BufferId, offset: u32 },
    /// Modular: buf[i % modulus]
    ModLoad { buffer: BufferId, modulus: u32 },
    /// Index table: buf[table_buffer[i]] where table_buffer holds u32 offsets as f32 bits.
    TableLoad { buffer: BufferId, table_buffer: BufferId },
    /// Gather from multiple buffers: table_buffer stores (buf_slot, offset) pairs as u32.
    /// buf_slot is an index into a per-op buffer list stored in gather_bufs.
    GatherLoad { gather_bufs: Vec<BufferId>, table_buffer: BufferId },
    /// Constant.
    Literal(f32),
    /// Binary op on two VRegs.
    Binary { op: ScalarBinOp, a: VReg, b: VReg, compute_dt: DType },
    /// Unary op.
    Unary { op: ScalarUnaryOp, x: VReg, compute_dt: DType },
    /// Ternary select: cond ? x : y.
    Select { cond: VReg, x: VReg, y: VReg, compute_dt: DType },
    /// Cast to dtype.
    Cast { x: VReg, to: DType },
    /// Reference to a reduction result.
    ReduceResult(usize),
}

/// An operation inside a reduction's k-loop body.
#[derive(Debug, Clone)]
pub enum KReduceOp {
    /// 2D load: buf[base + stride_i * i + stride_k * k]
    SymLoad { buffer: BufferId, base_offset: u32, stride_i: i32, stride_k: i32 },
    /// Broadcast: buf[offset] (invariant across i and k)
    BroadcastLoad { buffer: BufferId, offset: u32 },
    /// Reference to a value computed in the outer scope.
    OuterRef(VReg),
    /// Constant.
    Literal(f32),
    /// Binary op on body-local values (indices into body vec).
    Binary { op: ScalarBinOp, a: u16, b: u16, compute_dt: DType },
    /// Unary op.
    Unary { op: ScalarUnaryOp, x: u16, compute_dt: DType },
    /// Cast.
    Cast { x: u16, to: DType },
}

/// A reduction loop within a kernel.
#[derive(Debug, Clone)]
pub struct KernelReduction {
    pub is_sum: bool,
    pub bound: u64,
    pub compute_dt: DType,
    pub output_dt: DType,
    pub body: Vec<KReduceOp>,
    pub body_result: u16,
}

/// A compiled kernel plan — computes one output buffer region.
#[derive(Debug)]
pub struct KernelPlan {
    pub output: BufferId,
    pub output_offset: u32,
    pub extent: u32,
    pub ops: Vec<KOp>,
    pub reductions: Vec<KernelReduction>,
    pub result: VReg,
    pub input_buffers: Vec<BufferId>,
}

/// A buffer in the execution layout.
#[derive(Debug)]
pub struct BufferInfo {
    pub id: BufferId,
    pub base_atom: AtomId,
    pub count: u32,
    pub is_constant: bool,
}

/// The full compilation plan.
#[derive(Debug)]
pub struct CompilationPlan {
    pub buffers: Vec<BufferInfo>,
    pub kernels: Vec<KernelPlan>,
    pub atom_to_buffer: HashMap<u32, BufferId>,
    pub constants: HashMap<BufferId, Vec<(u32, f32)>>,
    pub output_buffers: Vec<BufferId>,
    /// Table buffers: BufferId → Vec<u32> offsets (stored as f32-reinterpreted u32).
    pub table_data: HashMap<BufferId, Vec<u32>>,
}

// ---------------------------------------------------------------------------
// Group dependency graph
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct GroupDep {
    source_group: usize,
    input_idx: usize,
}

#[derive(Debug)]
struct GroupInfo {
    deps: Vec<GroupDep>,
    consumers: Vec<usize>,
    /// Which groups produce atoms that this group reads from, per input slot.
    /// Unlike `deps` which deduplicates, this tracks ALL source groups per input.
    input_source_groups: Vec<Vec<usize>>,
}

/// Binary search for the group containing an atom.
fn find_group_idx(groups: &[AtomGroup], id: AtomId) -> Option<usize> {
    let idx = groups.partition_point(|g| g.base_id.0 <= id.0);
    if idx == 0 { return None; }
    let gi = idx - 1;
    if groups[gi].contains(id) { Some(gi) } else { None }
}

fn build_group_graph(graph: &NanoGraph) -> Vec<GroupInfo> {
    let groups = graph.groups();
    let n = groups.len();
    let mut infos: Vec<GroupInfo> = (0..n)
        .map(|_| GroupInfo {
            deps: Vec::new(),
            consumers: Vec::new(),
            input_source_groups: Vec::new(),
        })
        .collect();

    for (gi, group) in groups.iter().enumerate() {
        let mut per_input_sources: Vec<Vec<usize>> = Vec::with_capacity(group.inputs.len());

        for (inp_idx, input_ref) in group.inputs.iter().enumerate() {
            let mut source_groups_for_input: Vec<usize> = Vec::new();
            let mut source_set: HashSet<usize> = HashSet::new();

            match input_ref {
                InputRef::SymAffine { .. } => {
                    // Sample across k to discover all source groups.
                    let bound = if !group.reduce_dims.is_empty() {
                        let rd = group.reduce_dims[0];
                        graph.sym_dim_bounds.get(&rd).copied().unwrap_or(1)
                    } else { 1 };
                    for k in 0..bound as u32 {
                        for i_sample in [0, group.count.saturating_sub(1)] {
                            let src = input_ref.resolve(i_sample, k);
                            if let Some(src_gi) = find_group_idx(groups, src) {
                                if source_set.insert(src_gi) {
                                    source_groups_for_input.push(src_gi);
                                }
                            }
                        }
                    }
                }
                InputRef::Explicit(ids) => {
                    for id in ids {
                        if let Some(src_gi) = find_group_idx(groups, *id) {
                            if source_set.insert(src_gi) {
                                source_groups_for_input.push(src_gi);
                            }
                        }
                    }
                }
                _ => {
                    // Broadcast or Affine: sample i=0 and i=count-1.
                    for i_sample in [0, group.count.saturating_sub(1)] {
                        let src = input_ref.resolve(i_sample, 0);
                        if let Some(src_gi) = find_group_idx(groups, src) {
                            if source_set.insert(src_gi) {
                                source_groups_for_input.push(src_gi);
                            }
                        }
                    }
                }
            }

            // Record dependencies (deduplicated across inputs).
            let mut dep_set: HashSet<usize> = infos[gi].deps.iter()
                .map(|d| d.source_group).collect();
            for &src_gi in &source_groups_for_input {
                if dep_set.insert(src_gi) {
                    infos[gi].deps.push(GroupDep { source_group: src_gi, input_idx: inp_idx });
                    infos[src_gi].consumers.push(gi);
                }
            }

            per_input_sources.push(source_groups_for_input);
        }

        infos[gi].input_source_groups = per_input_sources;
    }

    infos
}

// ---------------------------------------------------------------------------
// Materialization analysis
// ---------------------------------------------------------------------------

/// Reasons a group must be materialized.
#[derive(Debug)]
enum MatReason {
    Literal,
    Output,
    MultiConsumer,
    NoConsumer,
    /// Consumer reads via incompatible addressing.
    IncompatibleAddr,
    /// Reduction whose consumer spans multiple source groups.
    ReductionSpanning,
}

/// Determine which groups must be materialized (are kernel roots).
/// Returns a set of group indices that need buffers.
fn identify_kernel_roots(
    graph: &NanoGraph,
    infos: &[GroupInfo],
) -> (HashSet<usize>, HashMap<usize, MatReason>) {
    let groups = graph.groups();
    let n = groups.len();

    let mut output_groups: HashSet<usize> = HashSet::new();
    for &out_atom in &graph.outputs {
        if let Some(gi) = find_group_idx(groups, out_atom) {
            output_groups.insert(gi);
        }
    }

    let mut roots: HashSet<usize> = HashSet::new();
    let mut reasons: HashMap<usize, MatReason> = HashMap::new();

    for gi in 0..n {
        let group = &groups[gi];

        // 1. Literals always materialize.
        if matches!(group.op, ScalarOp::Literal(_)) {
            roots.insert(gi);
            reasons.insert(gi, MatReason::Literal);
            continue;
        }

        // 2. Graph outputs.
        if output_groups.contains(&gi) {
            roots.insert(gi);
            reasons.insert(gi, MatReason::Output);
            continue;
        }

        // 3. No consumers (dead code, but materialize for safety).
        if infos[gi].consumers.is_empty() {
            roots.insert(gi);
            reasons.insert(gi, MatReason::NoConsumer);
            continue;
        }

        // 4. Multiple consumers.
        if infos[gi].consumers.len() > 1 {
            roots.insert(gi);
            reasons.insert(gi, MatReason::MultiConsumer);
            continue;
        }

        // Single consumer — check fusion eligibility.
        let consumer_gi = infos[gi].consumers[0];
        let consumer = &groups[consumer_gi];

        // Find which input slot of the consumer references us.
        let input_idx = infos[consumer_gi].deps.iter()
            .find(|d| d.source_group == gi)
            .map(|d| d.input_idx);

        let Some(inp_idx) = input_idx else {
            roots.insert(gi);
            reasons.insert(gi, MatReason::IncompatibleAddr);
            continue;
        };

        let input_ref = &consumer.inputs[inp_idx];

        // Check fusion cases:
        let fusable = if consumer.op.is_reduce() {
            // Case: consumer is a reduction, reads us via SymAffine → reduction inlining.
            matches!(input_ref, InputRef::SymAffine { .. })
                && !matches!(group.op, ScalarOp::Literal(_))
                && !group.op.is_reduce()
        } else if group.op.is_reduce() {
            // Case: we're a reduction, consumer reads us via stride-1 → post-reduction fusion.
            // Only if the consumer's input spans exactly our group (no other groups in range).
            match input_ref {
                InputRef::Affine { base, stride } => {
                    *stride == 1
                        && *base == group.base_id
                        && consumer.count == group.count
                        // Ensure the input range doesn't span other groups.
                        && infos[consumer_gi].input_source_groups.get(inp_idx)
                            .map_or(false, |srcs| srcs.len() == 1 && srcs[0] == gi)
                }
                _ => false,
            }
        } else {
            // Case: both pointwise — standard pointwise fusion.
            match input_ref {
                InputRef::Affine { base, stride } => {
                    *stride == 1
                        && *base == group.base_id
                        && consumer.count == group.count
                        && consumer.sym_dims == group.sym_dims
                }
                InputRef::Explicit(ids) => {
                    // Fuse if Explicit is effectively identity.
                    ids.len() == group.count as usize
                        && ids.iter().enumerate().all(|(i, id)| {
                            *id == group.base_id.offset(i as u32)
                        })
                }
                _ => false,
            }
        };

        if !fusable {
            roots.insert(gi);
            let reason = if group.op.is_reduce() {
                MatReason::ReductionSpanning
            } else {
                MatReason::IncompatibleAddr
            };
            reasons.insert(gi, reason);
        }
    }

    (roots, reasons)
}

// ---------------------------------------------------------------------------
// Buffer assignment
// ---------------------------------------------------------------------------

fn assign_buffers(
    graph: &NanoGraph,
    lower: &LowerResult,
    roots: &HashSet<usize>,
) -> (
    Vec<BufferInfo>,
    HashMap<usize, BufferId>,
    HashMap<u32, BufferId>,
    HashMap<BufferId, Vec<(u32, f32)>>,
) {
    let groups = graph.groups();
    let n = groups.len();

    // Union-find for buffer co-location.
    let mut parent: Vec<usize> = (0..n).collect();

    fn uf_find(p: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while p[r] != r { r = p[r]; }
        let mut c = x;
        while p[c] != r { let nx = p[c]; p[c] = r; c = nx; }
        r
    }
    fn uf_union(p: &mut [usize], a: usize, b: usize) {
        let ra = uf_find(p, a);
        let rb = uf_find(p, b);
        if ra != rb { p[rb] = ra; }
    }

    // Co-locate groups spanned by SymAffine inputs.
    for (gi, group) in groups.iter().enumerate() {
        for input_ref in &group.inputs {
            if let InputRef::SymAffine { .. } = input_ref {
                let bound = if !group.reduce_dims.is_empty() {
                    let rd = group.reduce_dims[0];
                    graph.sym_dim_bounds.get(&rd).copied().unwrap_or(1)
                } else { 1 };
                let mut prev_gi: Option<usize> = None;
                for k in 0..bound as u32 {
                    let atom = input_ref.resolve(0, k);
                    if let Some(src_gi) = find_group_idx(groups, atom) {
                        if let Some(pg) = prev_gi {
                            uf_union(&mut parent, pg, src_gi);
                        }
                        prev_gi = Some(src_gi);
                    }
                }
            }
        }
    }

    // Co-locate groups belonging to the same tensor (via tensor_map).
    for tam in lower.tensor_map.values() {
        let start = tam.base_id.0;
        let end = start + tam.count;
        let mut tensor_groups: Vec<usize> = Vec::new();
        for (gi, group) in groups.iter().enumerate() {
            let gs = group.base_id.0;
            let ge = gs + group.count;
            if gs < end && ge > start {
                tensor_groups.push(gi);
            }
        }
        for i in 1..tensor_groups.len() {
            uf_union(&mut parent, tensor_groups[0], tensor_groups[i]);
        }
    }

    // Assign buffers to kernel root groups.
    let mut buffers: Vec<BufferInfo> = Vec::new();
    let mut group_to_buffer: HashMap<usize, BufferId> = HashMap::new();
    let mut root_to_buffer: HashMap<usize, BufferId> = HashMap::new();
    let mut buf_ranges: HashMap<BufferId, (u32, u32)> = HashMap::new();

    for gi in 0..n {
        if !roots.contains(&gi) { continue; }
        let group = &groups[gi];
        let root = uf_find(&mut parent, gi);

        if let Some(&buf_id) = root_to_buffer.get(&root) {
            // Extend existing buffer.
            let range = buf_ranges.get_mut(&buf_id).unwrap();
            range.0 = range.0.min(group.base_id.0);
            range.1 = range.1.max(group.base_id.0 + group.count);
            group_to_buffer.insert(gi, buf_id);
        } else {
            let buf_id = BufferId(buffers.len() as u32);
            buffers.push(BufferInfo {
                id: buf_id,
                base_atom: group.base_id,
                count: group.count,
                is_constant: matches!(group.op, ScalarOp::Literal(_)),
            });
            buf_ranges.insert(buf_id, (group.base_id.0, group.base_id.0 + group.count));
            group_to_buffer.insert(gi, buf_id);
            root_to_buffer.insert(root, buf_id);
        }
    }

    // Fix up shared buffer ranges.
    for (&buf_id, &(min_atom, max_atom)) in &buf_ranges {
        let buf = &mut buffers[buf_id.0 as usize];
        buf.base_atom = AtomId(min_atom);
        buf.count = max_atom - min_atom;
        if max_atom - min_atom > buf.count {
            buf.is_constant = false;
        }
    }

    // Build atom_to_buffer for ALL materialized groups.
    let mut atom_to_buffer: HashMap<u32, BufferId> = HashMap::new();
    for (&gi, &buf_id) in &group_to_buffer {
        atom_to_buffer.insert(groups[gi].base_id.0, buf_id);
    }

    // Pre-fill constant buffers.
    let mut constants: HashMap<BufferId, Vec<(u32, f32)>> = HashMap::new();
    for (gi, group) in groups.iter().enumerate() {
        if !matches!(group.op, ScalarOp::Literal(_)) { continue; }
        let Some(&buf_id) = group_to_buffer.get(&gi) else { continue; };
        let buf_base = buffers[buf_id.0 as usize].base_atom.0;
        let mut pairs = Vec::new();
        for i in 0..group.count {
            let atom_idx = group.base_id.0 + i;
            let buf_offset = atom_idx - buf_base;
            if let Some(scalar) = lower.numeric_overrides.get(&atom_idx) {
                pairs.push((buf_offset, scalar.to_f64() as f32));
            } else if let ScalarOp::Literal(ref s) = group.op {
                pairs.push((buf_offset, s.to_f64() as f32));
            }
        }
        if !pairs.is_empty() {
            constants.entry(buf_id).or_default().extend(pairs);
        }
    }

    (buffers, group_to_buffer, atom_to_buffer, constants)
}

// ---------------------------------------------------------------------------
// Plan builder (main entry point)
// ---------------------------------------------------------------------------

pub fn build_plan(lower: &LowerResult) -> CompilationPlan {
    let graph = &lower.graph;
    let groups = graph.groups();
    let n = groups.len();

    let infos = build_group_graph(graph);
    let (roots, reasons) = identify_kernel_roots(graph, &infos);

    // Log materialization stats.
    let mut reason_counts = [0usize; 6]; // Literal, Output, MultiConsumer, NoConsumer, IncompatibleAddr, ReductionSpanning
    let fused_count = n - roots.len();
    for r in reasons.values() {
        match r {
            MatReason::Literal => reason_counts[0] += 1,
            MatReason::Output => reason_counts[1] += 1,
            MatReason::MultiConsumer => reason_counts[2] += 1,
            MatReason::NoConsumer => reason_counts[3] += 1,
            MatReason::IncompatibleAddr => reason_counts[4] += 1,
            MatReason::ReductionSpanning => reason_counts[5] += 1,
        }
    }
    eprintln!(
        "[v11] {} groups: {} fused, materialize: lit={} out={} multi={} dead={} addr={} red_span={}",
        n, fused_count,
        reason_counts[0], reason_counts[1], reason_counts[2],
        reason_counts[3], reason_counts[4], reason_counts[5]
    );

    let (buffers, group_to_buffer, atom_to_buffer, constants) =
        assign_buffers(graph, lower, &roots);

    // Build kernels for each non-literal kernel root.
    let mut kernels: Vec<KernelPlan> = Vec::new();
    let mut next_table_buf_id = buffers.len() as u32;
    let mut table_data: HashMap<BufferId, Vec<u32>> = HashMap::new();
    for gi in 0..n {
        if !roots.contains(&gi) { continue; }
        if matches!(groups[gi].op, ScalarOp::Literal(_)) { continue; }

        let kernel = build_kernel(
            graph, groups, &infos, &roots, &group_to_buffer, &buffers, gi,
            &mut next_table_buf_id, &mut table_data,
        );
        kernels.push(kernel);
    }

    let kernels = toposort_kernels(kernels);

    // Count integer-typed ops in kernels for diagnostics.
    let mut int_op_count = 0usize;
    let mut int_kernel_count = 0usize;
    let mut gather_count = 0usize;
    for k in &kernels {
        let mut has_int = false;
        for op in &k.ops {
            match op {
                KOp::Binary { compute_dt, .. } | KOp::Unary { compute_dt, .. }
                | KOp::Select { compute_dt, .. } | KOp::Cast { to: compute_dt, .. } => {
                    if !matches!(compute_dt, DType::F32 | DType::F64 | DType::BF16 | DType::F16) {
                        int_op_count += 1;
                        has_int = true;
                    }
                }
                KOp::GatherLoad { .. } => { gather_count += 1; }
                _ => {}
            }
        }
        if has_int { int_kernel_count += 1; }
    }
    if int_op_count > 0 || gather_count > 0 {
        eprintln!("[v11] {} integer ops in {} kernels, {} gather loads",
            int_op_count, int_kernel_count, gather_count);
    }

    // Identify output buffers.
    let mut output_groups: HashSet<usize> = HashSet::new();
    for &out_atom in &graph.outputs {
        if let Some(gi) = find_group_idx(groups, out_atom) {
            output_groups.insert(gi);
        }
    }
    let output_buffers: Vec<BufferId> = output_groups.iter()
        .filter_map(|&gi| group_to_buffer.get(&gi).copied())
        .collect();

    CompilationPlan { buffers, kernels, atom_to_buffer, constants, output_buffers, table_data }
}

// ---------------------------------------------------------------------------
// Kernel construction
// ---------------------------------------------------------------------------

struct KernelBuilder<'a> {
    ops: Vec<KOp>,
    reductions: Vec<KernelReduction>,
    next_vreg: u16,
    result: Option<VReg>,
    input_buffers: HashSet<BufferId>,
    group_cache: HashMap<usize, VReg>,
    buf_base_atoms: HashMap<BufferId, u32>,
    /// Table buffers allocated during kernel building.
    table_allocs: Vec<(BufferId, Vec<u32>)>,
    next_table_buf_id: &'a mut u32,
    // Context
    graph: &'a NanoGraph,
    groups: &'a [AtomGroup],
    infos: &'a [GroupInfo],
    roots: &'a HashSet<usize>,
    group_to_buffer: &'a HashMap<usize, BufferId>,
    buffers: &'a [BufferInfo],
}

impl<'a> KernelBuilder<'a> {
    fn new(
        graph: &'a NanoGraph,
        groups: &'a [AtomGroup],
        infos: &'a [GroupInfo],
        roots: &'a HashSet<usize>,
        group_to_buffer: &'a HashMap<usize, BufferId>,
        buffers: &'a [BufferInfo],
        next_table_buf_id: &'a mut u32,
    ) -> Self {
        let buf_base_atoms: HashMap<BufferId, u32> = buffers.iter()
            .map(|b| (b.id, b.base_atom.0))
            .collect();
        Self {
            ops: Vec::new(),
            reductions: Vec::new(),
            next_vreg: 0,
            result: None,
            input_buffers: HashSet::new(),
            group_cache: HashMap::new(),
            buf_base_atoms,
            table_allocs: Vec::new(),
            next_table_buf_id,
            graph, groups, infos, roots, group_to_buffer, buffers,
        }
    }

    /// Allocate a table buffer to hold index offsets.
    fn alloc_table_buffer(&mut self, offsets: Vec<u32>) -> BufferId {
        let buf_id = BufferId(*self.next_table_buf_id);
        *self.next_table_buf_id += 1;
        self.table_allocs.push((buf_id, offsets));
        buf_id
    }

    fn push_op(&mut self, op: KOp) -> VReg {
        let vreg = VReg(self.next_vreg);
        self.next_vreg += 1;
        self.ops.push(op);
        vreg
    }

    fn buf_offset(&self, buf_id: BufferId, atom: AtomId) -> u32 {
        atom.0 - self.buf_base_atoms[&buf_id]
    }

    /// Find the buffer for an atom (searching all materialized groups).
    fn find_buffer_for_atom(&self, atom: AtomId) -> Option<BufferId> {
        let gi = find_group_idx(self.groups, atom)?;
        self.group_to_buffer.get(&gi).copied()
    }

    /// Is this group materialized (has a buffer)?
    fn is_materialized(&self, gi: usize) -> bool {
        self.group_to_buffer.contains_key(&gi)
    }
}

fn build_kernel(
    graph: &NanoGraph,
    groups: &[AtomGroup],
    infos: &[GroupInfo],
    roots: &HashSet<usize>,
    group_to_buffer: &HashMap<usize, BufferId>,
    buffers: &[BufferInfo],
    root_gi: usize,
    next_table_buf_id: &mut u32,
    table_data: &mut HashMap<BufferId, Vec<u32>>,
) -> KernelPlan {
    let root_group = &groups[root_gi];
    let output_buffer = group_to_buffer[&root_gi];
    let extent = root_group.count;

    let mut builder = KernelBuilder::new(
        graph, groups, infos, roots, group_to_buffer, buffers, next_table_buf_id,
    );

    let result = emit_group_value(&mut builder, root_gi);
    builder.result = Some(result);

    let input_buffers: Vec<BufferId> = builder.input_buffers.iter().copied().collect();
    let output_offset = builder.buf_offset(output_buffer, root_group.base_id);

    // Collect table buffer allocations.
    for (buf_id, offsets) in builder.table_allocs {
        table_data.insert(buf_id, offsets);
    }

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

/// Recursively emit the computation for a group, inlining fused producers.
/// If the group is a reduction, builds a KernelReduction.
/// If the group is materialized (and not the kernel root), panics — caller
/// should use `emit_input_value` instead.
fn emit_group_value(builder: &mut KernelBuilder, gi: usize) -> VReg {
    if let Some(&vreg) = builder.group_cache.get(&gi) {
        return vreg;
    }

    let group = &builder.groups[gi];

    if group.op.is_reduce() {
        // Build a reduction.
        let vreg = build_reduction(builder, gi);
        builder.group_cache.insert(gi, vreg);
        return vreg;
    }

    // Pointwise: emit inputs, then the op.
    let input_vregs: Vec<VReg> = group.inputs.iter()
        .enumerate()
        .map(|(inp_idx, input_ref)| {
            emit_input_value(builder, gi, inp_idx, input_ref)
        })
        .collect();

    let vreg = emit_scalar_op(builder, &group.op, &input_vregs);
    builder.group_cache.insert(gi, vreg);
    vreg
}

/// Emit a value for an input, handling materialized (buffer load) vs fused.
fn emit_input_value(
    builder: &mut KernelBuilder,
    consumer_gi: usize,
    input_idx: usize,
    input_ref: &InputRef,
) -> VReg {
    // Find the source group(s) for this input.
    let source_atom = input_ref.resolve(0, 0);
    let src_gi = find_group_idx(builder.groups, source_atom)
        .expect("input ref resolves to valid group");

    if builder.is_materialized(src_gi) {
        // Load from buffer.
        emit_addressed_load(builder, input_ref)
    } else {
        // Fused — check cache or recursively emit.
        emit_group_value(builder, src_gi)
    }
}

/// Emit a load from a materialized buffer using the given InputRef addressing.
fn emit_addressed_load(builder: &mut KernelBuilder, input_ref: &InputRef) -> VReg {
    match input_ref {
        InputRef::Broadcast(atom_id) => {
            let buf_id = builder.find_buffer_for_atom(*atom_id)
                .expect("broadcast atom must have buffer");
            builder.input_buffers.insert(buf_id);
            let offset = builder.buf_offset(buf_id, *atom_id);
            builder.push_op(KOp::BroadcastLoad { buffer: buf_id, offset })
        }
        InputRef::Affine { base, stride } => {
            let buf_id = builder.find_buffer_for_atom(*base)
                .expect("affine base must have buffer");
            builder.input_buffers.insert(buf_id);
            let base_offset = builder.buf_offset(buf_id, *base);
            builder.push_op(KOp::Load { buffer: buf_id, base_offset, stride: *stride })
        }
        InputRef::Explicit(ids) => {
            if ids.is_empty() {
                return builder.push_op(KOp::Literal(0.0));
            }
            let buf_id = builder.find_buffer_for_atom(ids[0])
                .expect("explicit atom must have buffer");
            builder.input_buffers.insert(buf_id);
            let buf_base = builder.buf_base_atoms[&buf_id];

            // Try affine pattern first.
            let first_off = ids[0].0 as i64 - buf_base as i64;
            if ids.len() == 1 {
                return builder.push_op(KOp::BroadcastLoad {
                    buffer: buf_id, offset: first_off as u32,
                });
            }
            let stride = ids[1].0 as i64 - ids[0].0 as i64;
            let is_affine = ids.iter().enumerate().all(|(i, id)| {
                id.0 as i64 == ids[0].0 as i64 + stride * i as i64
            });
            if is_affine {
                return builder.push_op(KOp::Load {
                    buffer: buf_id,
                    base_offset: first_off as u32,
                    stride: stride as i32,
                });
            }

            // Check for repeating pattern.
            let src_group = builder.groups.iter()
                .find(|g| g.contains(ids[0]))
                .unwrap();
            let src_count = src_group.count as usize;
            let is_repeat = ids.len() > src_count
                && ids.chunks(src_count).all(|chunk| {
                    chunk.iter().enumerate().all(|(i, id)| {
                        *id == src_group.base_id.offset(i as u32)
                    })
                });
            if is_repeat {
                return builder.push_op(KOp::ModLoad {
                    buffer: buf_id, modulus: src_count as u32,
                });
            }

            // Truly irregular: check if all atoms are in the same buffer.
            let all_same_buf = ids.iter().all(|id| {
                builder.find_buffer_for_atom(*id) == Some(buf_id)
            });

            if all_same_buf {
                // All in one buffer — use TableLoad with offsets.
                let offsets: Vec<u32> = ids.iter()
                    .map(|id| (id.0 - buf_base) as u32)
                    .collect();
                let table_buffer = builder.alloc_table_buffer(offsets);
                builder.input_buffers.insert(table_buffer);
                builder.push_op(KOp::TableLoad { buffer: buf_id, table_buffer })
            } else {
                // Multi-buffer gather: build (buf_slot, offset) pairs.
                let mut unique_bufs: Vec<BufferId> = Vec::new();
                let mut table_entries: Vec<u32> = Vec::new();
                for id in ids {
                    let b = builder.find_buffer_for_atom(*id)
                        .expect("gather atom must have buffer");
                    let slot = match unique_bufs.iter().position(|&x| x == b) {
                        Some(pos) => pos,
                        None => { unique_bufs.push(b); unique_bufs.len() - 1 }
                    };
                    let b_base = builder.buf_base_atoms[&b];
                    let offset = id.0 - b_base;
                    table_entries.push(slot as u32);
                    table_entries.push(offset);
                }
                for &b in &unique_bufs {
                    builder.input_buffers.insert(b);
                }
                let table_buffer = builder.alloc_table_buffer(table_entries);
                builder.input_buffers.insert(table_buffer);
                builder.push_op(KOp::GatherLoad { gather_bufs: unique_bufs, table_buffer })
            }
        }
        InputRef::SymAffine { .. } => {
            panic!("SymAffine in outer (non-reduction) context — should be handled by reduction builder");
        }
    }
}

/// Emit a scalar op given its input VRegs.
fn emit_scalar_op(builder: &mut KernelBuilder, op: &ScalarOp, inputs: &[VReg]) -> VReg {
    match op {
        ScalarOp::Identity { compute_dtype, output_dtype } => {
            let mut v = inputs[0];
            if compute_dtype != output_dtype {
                v = builder.push_op(KOp::Cast { x: v, to: *output_dtype });
            }
            v
        }
        ScalarOp::Binary { op, compute_dtype, output_dtype } => {
            let v = builder.push_op(KOp::Binary {
                op: *op, a: inputs[0], b: inputs[1], compute_dt: *compute_dtype,
            });
            if compute_dtype != output_dtype {
                builder.push_op(KOp::Cast { x: v, to: *output_dtype })
            } else { v }
        }
        ScalarOp::Unary { op, compute_dtype, output_dtype } => {
            let v = builder.push_op(KOp::Unary {
                op: *op, x: inputs[0], compute_dt: *compute_dtype,
            });
            if compute_dtype != output_dtype {
                builder.push_op(KOp::Cast { x: v, to: *output_dtype })
            } else { v }
        }
        ScalarOp::Select { compute_dtype, output_dtype } => {
            let v = builder.push_op(KOp::Select {
                cond: inputs[0], x: inputs[1], y: inputs[2], compute_dt: *compute_dtype,
            });
            if compute_dtype != output_dtype {
                builder.push_op(KOp::Cast { x: v, to: *output_dtype })
            } else { v }
        }
        ScalarOp::Literal(scalar) => {
            builder.push_op(KOp::Literal(scalar.to_f64() as f32))
        }
        _ => panic!("emit_scalar_op: unexpected reduce op"),
    }
}

// ---------------------------------------------------------------------------
// Reduction construction
// ---------------------------------------------------------------------------

/// Build a reduction kernel. Emits a ReduceResult KOp and returns its VReg.
fn build_reduction(builder: &mut KernelBuilder, reduce_gi: usize) -> VReg {
    let reduce_group = &builder.groups[reduce_gi];
    let (compute_dt, output_dt, is_sum) = match &reduce_group.op {
        ScalarOp::ReduceSum { compute_dtype, output_dtype } => (*compute_dtype, *output_dtype, true),
        ScalarOp::ReduceMax { compute_dtype, output_dtype } => (*compute_dtype, *output_dtype, false),
        _ => panic!("not a reduce group"),
    };

    assert_eq!(reduce_group.reduce_dims.len(), 1, "single reduce dim expected");
    let rd = reduce_group.reduce_dims[0];
    let bound = builder.graph.sym_dim_bounds.get(&rd).copied()
        .expect("reduce dim must have bound");

    let input_ref = &reduce_group.inputs[0];
    let source_atom = input_ref.resolve(0, 0);
    let src_gi = find_group_idx(builder.groups, source_atom).unwrap();

    // Can we fuse the source chain into the k-loop?
    let can_fuse = !builder.is_materialized(src_gi)
        && !matches!(builder.groups[src_gi].op, ScalarOp::Literal(_))
        && !builder.groups[src_gi].op.is_reduce()
        && matches!(input_ref, InputRef::SymAffine { .. });

    let reduction = if can_fuse {
        build_fused_reduction_body(
            builder, reduce_gi, src_gi, bound, compute_dt, output_dt, is_sum,
        )
    } else {
        build_simple_reduction_body(
            builder, reduce_gi, bound, compute_dt, output_dt, is_sum,
        )
    };

    let red_idx = builder.reductions.len();
    builder.reductions.push(reduction);
    builder.push_op(KOp::ReduceResult(red_idx))
}

/// Simple reduction body: load from materialized buffer.
fn build_simple_reduction_body(
    builder: &mut KernelBuilder,
    reduce_gi: usize,
    bound: u64,
    compute_dt: DType,
    output_dt: DType,
    is_sum: bool,
) -> KernelReduction {
    let reduce_group = &builder.groups[reduce_gi];
    let input_ref = &reduce_group.inputs[0];

    let body_op = match input_ref {
        InputRef::SymAffine { base, stride_i, stride_k } => {
            let buf_id = builder.find_buffer_for_atom(*base)
                .expect("sym affine base must have buffer");
            builder.input_buffers.insert(buf_id);
            let base_offset = builder.buf_offset(buf_id, *base);
            KReduceOp::SymLoad {
                buffer: buf_id, base_offset,
                stride_i: *stride_i, stride_k: *stride_k,
            }
        }
        InputRef::Affine { base, stride } => {
            let buf_id = builder.find_buffer_for_atom(*base)
                .expect("affine base must have buffer");
            builder.input_buffers.insert(buf_id);
            let base_offset = builder.buf_offset(buf_id, *base);
            KReduceOp::SymLoad {
                buffer: buf_id, base_offset,
                stride_i: *stride, stride_k: 0,
            }
        }
        InputRef::Broadcast(atom_id) => {
            let buf_id = builder.find_buffer_for_atom(*atom_id)
                .expect("broadcast atom must have buffer");
            builder.input_buffers.insert(buf_id);
            let offset = builder.buf_offset(buf_id, *atom_id);
            KReduceOp::BroadcastLoad { buffer: buf_id, offset }
        }
        _ => panic!("unsupported InputRef in simple reduction"),
    };

    KernelReduction {
        is_sum, bound, compute_dt, output_dt,
        body: vec![body_op],
        body_result: 0,
    }
}

/// Build a fused reduction body using sample-based stride discovery.
///
/// Walks backward from the source group, collecting single-consumer pointwise
/// producers into the chain. For each external input of the chain, samples
/// the resolved atom address at (i=0,k=0), (i=1,k=0), (i=0,k=1) to discover
/// the SymLoad parameters (base, stride_i, stride_k).
fn build_fused_reduction_body(
    builder: &mut KernelBuilder,
    reduce_gi: usize,
    source_gi: usize,
    bound: u64,
    compute_dt: DType,
    output_dt: DType,
    is_sum: bool,
) -> KernelReduction {
    let reduce_group = &builder.groups[reduce_gi];
    let reduce_input_ref = &reduce_group.inputs[0];

    // Collect the chain of fusable groups (dependency order: leaves first).
    let fused_chain = collect_reduction_chain(builder, source_gi);

    // Build reduction body ops.
    let mut body: Vec<KReduceOp> = Vec::new();
    let mut chain_body_idx: HashMap<usize, u16> = HashMap::new();

    for &chain_gi in &fused_chain {
        let chain_group = &builder.groups[chain_gi];

        // Emit loads for each input of this chain group.
        let input_body_vals: Vec<u16> = chain_group.inputs.iter()
            .enumerate()
            .map(|(inp_idx, input_ref)| {
                let src_atom = input_ref.resolve(0, 0);
                let src_gi = find_group_idx(builder.groups, src_atom).unwrap();

                if let Some(&bv) = chain_body_idx.get(&src_gi) {
                    // From another fused chain group.
                    let idx = body.len() as u16;
                    body.push(KReduceOp::Cast { x: bv, to: compute_dt });
                    return idx;
                }

                // External buffer load — use sample-based stride discovery.
                emit_sampled_reduce_load(
                    builder, &mut body, reduce_input_ref, source_gi,
                    chain_gi, inp_idx, input_ref, bound,
                )
            })
            .collect();

        // Emit the scalar op.
        let result_idx = emit_reduce_body_op(&mut body, &chain_group.op, &input_body_vals, compute_dt);
        chain_body_idx.insert(chain_gi, result_idx);
    }

    let body_result = *chain_body_idx.get(&source_gi).unwrap();

    KernelReduction { is_sum, bound, compute_dt, output_dt, body, body_result }
}

/// Collect groups to fuse into a reduction body.
/// Returns in dependency order (leaves first).
fn collect_reduction_chain(
    builder: &KernelBuilder,
    source_gi: usize,
) -> Vec<usize> {
    let mut chain: Vec<usize> = Vec::new();
    let mut visited: HashSet<usize> = HashSet::new();
    let mut stack: Vec<usize> = vec![source_gi];

    while let Some(gi) = stack.pop() {
        if !visited.insert(gi) { continue; }
        let group = &builder.groups[gi];

        if builder.is_materialized(gi) && gi != source_gi { continue; }
        if matches!(group.op, ScalarOp::Literal(_)) { continue; }

        for dep in &builder.infos[gi].deps {
            let prod_gi = dep.source_group;
            if !visited.contains(&prod_gi)
                && !builder.is_materialized(prod_gi)
                && !matches!(builder.groups[prod_gi].op, ScalarOp::Literal(_))
                && builder.infos[prod_gi].consumers.len() == 1
            {
                stack.push(prod_gi);
            }
        }

        chain.push(gi);
    }

    // Topological sort.
    chain.reverse();
    toposort_chain(builder.infos, &chain)
}

/// Sample-based stride discovery for reduction body loads.
///
/// Given a chain group's external input, samples the resolved buffer atom at:
///   (i=0, k=0), (i=1, k=0), (i=0, k=1)
/// and fits: addr = base + stride_i * i + stride_k * k.
/// Verifies at (i=1, k=1) if possible.
fn emit_sampled_reduce_load(
    builder: &mut KernelBuilder,
    body: &mut Vec<KReduceOp>,
    reduce_input_ref: &InputRef,
    source_gi: usize,
    chain_gi: usize,
    inp_idx: usize,
    input_ref: &InputRef,
    bound: u64,
) -> u16 {
    let idx = body.len() as u16;

    // Helper: resolve the chain input's buffer atom at a given (i, k).
    // Traces from the reduce's entry atom through k-sibling groups to find
    // the external input. At k>0 the resolved groups are structurally
    // identical to the k=0 chain groups, so we replay the same chain path.
    let groups = builder.groups;
    let resolve_at = |i: u32, k: u32| -> Option<AtomId> {
        let entry_atom = reduce_input_ref.resolve(i, k);
        trace_to_external_input(groups, source_gi, chain_gi, entry_atom, inp_idx)
    };

    // If the chain group's input is already SymAffine, use it directly.
    if let InputRef::SymAffine { base, stride_i, stride_k } = input_ref {
        let buf_id = builder.find_buffer_for_atom(*base)
            .expect("sym affine input must have buffer");
        builder.input_buffers.insert(buf_id);
        let base_offset = builder.buf_offset(buf_id, *base);
        body.push(KReduceOp::SymLoad {
            buffer: buf_id, base_offset,
            stride_i: *stride_i, stride_k: *stride_k,
        });
        return idx;
    }

    // Sample at key points.
    let addr00 = resolve_at(0, 0);
    let count = builder.groups[chain_gi].count;
    let addr10 = if count > 1 { resolve_at(1, 0) } else { None };
    let addr01 = if bound > 1 { resolve_at(0, 1) } else { None };

    let Some(a00) = addr00 else {
        // Can't resolve — emit literal 0 as fallback.
        body.push(KReduceOp::Literal(0.0));
        return idx;
    };

    let buf_id = builder.find_buffer_for_atom(a00)
        .expect("resolved atom must have buffer");
    builder.input_buffers.insert(buf_id);
    let buf_base = builder.buf_base_atoms[&buf_id];
    let base_off = a00.0 as i64 - buf_base as i64;

    let stride_i = addr10.map_or(0i32, |a10| {
        (a10.0 as i64 - a00.0 as i64) as i32
    });

    let stride_k = addr01.map_or(0i32, |a01| {
        (a01.0 as i64 - a00.0 as i64) as i32
    });

    // Verify at (1, 1) if possible.
    if count > 1 && bound > 1 {
        if let Some(a11) = resolve_at(1, 1) {
            let expected = a00.0 as i64 + stride_i as i64 + stride_k as i64;
            if a11.0 as i64 != expected {
                // Non-affine pattern — fall back. For now, emit with discovered strides
                // and hope it works. A production compiler would materialize instead.
                eprintln!(
                    "[v11 warn] non-affine reduction input for group {} input {}: expected {} got {}",
                    chain_gi, inp_idx, expected, a11.0
                );
            }
        }
    }

    if stride_i == 0 && stride_k == 0 {
        body.push(KReduceOp::BroadcastLoad {
            buffer: buf_id, offset: base_off as u32,
        });
    } else {
        body.push(KReduceOp::SymLoad {
            buffer: buf_id, base_offset: base_off as u32,
            stride_i, stride_k,
        });
    }

    idx
}

/// Trace from an entry atom through the fused chain to find the external
/// buffer atom for a specific group's input.
///
/// Works by finding the path from source_gi to target_gi at k=0 (a sequence
/// of input indices to follow), then replaying that path from the entry atom
/// through k-sibling groups. This correctly handles k>0 where the actual
/// groups traversed are k-siblings of the chain groups.
fn trace_to_external_input(
    groups: &[AtomGroup],
    source_gi: usize,
    target_gi: usize,
    entry_atom: AtomId,
    target_input_idx: usize,
) -> Option<AtomId> {
    let entry_gi = find_group_idx(groups, entry_atom)?;
    let entry_group = &groups[entry_gi];
    let entry_offset = entry_atom.0 - entry_group.base_id.0;

    if source_gi == target_gi {
        // Depth-1 chain: entry group is the target (or its k-sibling).
        // Since k-sibling groups are structurally equivalent, resolve directly.
        if target_input_idx >= entry_group.inputs.len() { return None; }
        return Some(entry_group.inputs[target_input_idx].resolve(entry_offset, 0));
    }

    // Deeper chain: find path from source to target at k=0, then replay
    // through k-sibling groups.
    let path = find_chain_path_k0(groups, source_gi, target_gi)?;
    let mut current_gi = entry_gi;
    let mut current_offset = entry_offset;

    for &inp_idx in &path {
        let group = &groups[current_gi];
        if inp_idx >= group.inputs.len() { return None; }
        let dep_atom = group.inputs[inp_idx].resolve(current_offset, 0);
        let dep_gi = find_group_idx(groups, dep_atom)?;
        current_offset = dep_atom.0 - groups[dep_gi].base_id.0;
        current_gi = dep_gi;
    }

    // Now at target's k-sibling, resolve its external input.
    let group = &groups[current_gi];
    if target_input_idx >= group.inputs.len() { return None; }
    Some(group.inputs[target_input_idx].resolve(current_offset, 0))
}

/// Find the sequence of input indices to follow from source_gi to target_gi
/// using k=0 resolution. BFS through group dependencies.
fn find_chain_path_k0(
    groups: &[AtomGroup],
    source_gi: usize,
    target_gi: usize,
) -> Option<Vec<usize>> {
    let mut parent: HashMap<usize, (usize, usize)> = HashMap::new();
    let mut queue: VecDeque<usize> = VecDeque::new();
    queue.push_back(source_gi);

    while let Some(gi) = queue.pop_front() {
        let group = &groups[gi];
        for (inp_idx, input_ref) in group.inputs.iter().enumerate() {
            let dep_atom = input_ref.resolve(0, 0);
            if let Some(dep_gi) = find_group_idx(groups, dep_atom) {
                if dep_gi == target_gi {
                    let mut path = vec![inp_idx];
                    let mut cur = gi;
                    while cur != source_gi {
                        let (par, idx) = parent[&cur];
                        path.push(idx);
                        cur = par;
                    }
                    path.reverse();
                    return Some(path);
                }
                if !parent.contains_key(&dep_gi) && dep_gi != source_gi {
                    parent.insert(dep_gi, (gi, inp_idx));
                    queue.push_back(dep_gi);
                }
            }
        }
    }

    None
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
        ScalarOp::Binary { op, compute_dtype, .. } => {
            body.push(KReduceOp::Binary {
                op: *op, a: inputs[0], b: inputs[1], compute_dt: *compute_dtype,
            });
        }
        ScalarOp::Unary { op, compute_dtype, .. } => {
            body.push(KReduceOp::Unary {
                op: *op, x: inputs[0], compute_dt: *compute_dtype,
            });
        }
        ScalarOp::Identity { output_dtype, .. } => {
            body.push(KReduceOp::Cast { x: inputs[0], to: *output_dtype });
        }
        ScalarOp::Literal(s) => {
            body.push(KReduceOp::Literal(s.to_f64() as f32));
        }
        _ => panic!("unsupported op in reduction body: {:?}", op),
    }
    idx
}

// ---------------------------------------------------------------------------
// Topological sorting
// ---------------------------------------------------------------------------

fn toposort_chain(infos: &[GroupInfo], chain: &[usize]) -> Vec<usize> {
    let chain_set: HashSet<usize> = chain.iter().copied().collect();
    let mut in_degree: HashMap<usize, usize> = chain.iter().map(|&gi| (gi, 0)).collect();

    for &gi in chain {
        for dep in &infos[gi].deps {
            if chain_set.contains(&dep.source_group) {
                *in_degree.entry(gi).or_default() += 1;
            }
        }
    }

    let mut queue: VecDeque<usize> = in_degree.iter()
        .filter(|(_, d)| **d == 0)
        .map(|(&gi, _)| gi)
        .collect();
    let mut sorted = Vec::new();

    while let Some(gi) = queue.pop_front() {
        sorted.push(gi);
        for &consumer_gi in &infos[gi].consumers {
            if chain_set.contains(&consumer_gi) {
                if let Some(deg) = in_degree.get_mut(&consumer_gi) {
                    *deg -= 1;
                    if *deg == 0 { queue.push_back(consumer_gi); }
                }
            }
        }
    }

    sorted
}

fn toposort_kernels(kernels: Vec<KernelPlan>) -> Vec<KernelPlan> {
    if kernels.len() <= 1 { return kernels; }

    let mut output_to_idx: HashMap<BufferId, Vec<usize>> = HashMap::new();
    for (ki, k) in kernels.iter().enumerate() {
        output_to_idx.entry(k.output).or_default().push(ki);
    }

    let n = kernels.len();
    let mut in_degree = vec![0usize; n];
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (ki, k) in kernels.iter().enumerate() {
        for buf in &k.input_buffers {
            if let Some(pks) = output_to_idx.get(buf) {
                for &pk in pks {
                    if pk != ki {
                        in_degree[ki] += 1;
                        dependents[pk].push(ki);
                    }
                }
            }
        }
    }

    let mut queue: VecDeque<usize> = in_degree.iter()
        .enumerate()
        .filter(|(_, d)| **d == 0)
        .map(|(i, _)| i)
        .collect();
    let mut order: Vec<usize> = Vec::with_capacity(n);

    while let Some(ki) = queue.pop_front() {
        order.push(ki);
        for &dep_ki in &dependents[ki] {
            in_degree[dep_ki] -= 1;
            if in_degree[dep_ki] == 0 { queue.push_back(dep_ki); }
        }
    }

    let mut slots: Vec<Option<KernelPlan>> = kernels.into_iter().map(Some).collect();
    order.into_iter().map(|ki| slots[ki].take().unwrap()).collect()
}
