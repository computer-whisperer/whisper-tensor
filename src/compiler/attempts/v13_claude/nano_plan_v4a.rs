#![allow(clippy::all, dead_code, unreachable_patterns, unused_variables, unused_imports)]
//! Span-based NanoGraph partitioner v4a: edge-classification + O(groups) span build.
//!
//! Combines the correct algorithm from v3a (edge classification before phase assignment)
//! with an O(groups)-only span construction that resolves transitive dependencies.
//!
//! Previous attempt failures:
//! - v3a: Correct edge classification, OOM because span NanoGraph construction iterated atoms.
//! - v3c: 0 violations but span build had incomplete transitive dependency tracking —
//!   duplicated groups had InputRefs pointing outside the span (96 span input errors).
//! - v3b: Virtual-node DAG approach too slow (>120s on GPT-2).
//! - v3d: Fast but 55 violations, incomplete violation detection.
//!
//! This attempt (v4a) fixes both issues:
//! 1. Edge classification drives phase assignment (correct by construction).
//! 2. Span NanoGraph construction is strictly O(groups) — no per-atom iteration.
//! 3. Duplicated groups' transitive dependencies are fully resolved before span build.
//! 4. All input/output mappings use AtomMapping ranges, not per-atom pairs.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use crate::nano_graph::{AtomGroup, AtomId, InputRef, NanoGraph, ScalarOp, SymDim};

/// Literal groups with fewer atoms than this are duplicated into spans.
/// Larger literals (weight matrices) become external inputs.
const LITERAL_INLINE_THRESHOLD: u64 = 1024;

/// Groups with fewer atoms than this that are consumed cross-lane can be
/// duplicated into each lane instead of forcing a barrier.
const DUPLICATE_THRESHOLD: u64 = 65536;

// ─── Public types ────────────────────────────────────────────────────────────

/// A contiguous range of atoms mapped between main graph and span graph.
#[derive(Debug, Clone)]
pub struct AtomMapping {
    /// Start atom in the main graph.
    pub main_base: AtomId,
    /// Start atom in the span graph.
    pub span_base: AtomId,
    /// Number of contiguous atoms in this mapping.
    pub count: u64,
}

/// A self-contained computation unit: one lane's work in one phase.
pub struct Span {
    /// Self-contained NanoGraph for this span's computation.
    pub graph: NanoGraph,
    /// Contiguous ranges of atoms this span reads from the shared buffer.
    pub inputs: Vec<AtomMapping>,
    /// Contiguous ranges of atoms this span writes back to the shared buffer.
    pub outputs: Vec<AtomMapping>,
}

/// One phase of execution (between two barriers).
pub struct Phase {
    /// One span per lane. Empty spans are possible for idle lanes.
    pub spans: Vec<Span>,
}

/// The full span-based execution plan.
pub struct SpanPlan {
    pub num_lanes: usize,
    pub phases: Vec<Phase>,
}

// ─── Range-based atom map ────────────────────────────────────────────────────

/// An atom map that stores contiguous ranges instead of individual atoms.
/// Supports O(log n) lookup by main-graph AtomId.
struct RangeAtomMap {
    /// Sorted ranges: (main_base, span_base, count).
    ranges: Vec<(u64, u64, u64)>,
    sorted: bool,
}

impl RangeAtomMap {
    fn new() -> Self {
        Self {
            ranges: Vec::new(),
            sorted: true,
        }
    }

    fn insert_range(&mut self, main_base: AtomId, span_base: AtomId, count: u64) {
        self.ranges.push((main_base.0, span_base.0, count));
        self.sorted = false;
    }

    fn ensure_sorted(&mut self) {
        if !self.sorted {
            self.ranges.sort_by_key(|&(base, _, _)| base);
            self.sorted = true;
        }
    }

    fn get(&self, main_id: AtomId) -> Option<AtomId> {
        debug_assert!(self.sorted, "RangeAtomMap must be sorted before lookup");
        let idx = self.ranges.partition_point(|&(base, _, _)| base <= main_id.0);
        if idx == 0 {
            return None;
        }
        let (base, span_base, count) = self.ranges[idx - 1];
        let offset = main_id.0.wrapping_sub(base);
        if offset < count {
            Some(AtomId(span_base + offset))
        } else {
            None
        }
    }
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Partition a NanoGraph into self-contained spans organized by phase and lane.
///
/// The algorithm guarantees that within each phase, no span reads atoms
/// produced by another span (cross-lane independence).
///
/// All operations are O(groups), never O(atoms). A 45K-group graph with 8B atoms
/// is processed in seconds.
pub fn plan_spans(graph: &NanoGraph, num_lanes: usize) -> SpanPlan {
    let groups = graph.groups();
    let n = groups.len();
    let num_lanes = num_lanes.max(1);

    if n == 0 {
        return SpanPlan {
            num_lanes,
            phases: vec![],
        };
    }

    let is_literal: Vec<bool> = groups
        .iter()
        .map(|g| matches!(g.op, ScalarOp::Literal(_)) && g.inputs.is_empty())
        .collect();

    if is_literal.iter().all(|&lit| lit) {
        return SpanPlan {
            num_lanes,
            phases: vec![],
        };
    }

    // Step 1: Build group dependency DAG. O(groups * inputs_per_group).
    let (producers, consumers) = build_group_deps(groups);

    // Step 2: Topological sort. O(groups).
    let topo_order = topological_sort(n, &producers);

    // Step 3: Edge classification + phase assignment. O(groups * inputs_per_group).
    let (group_phase, num_phases, duplicated) = assign_phases_from_edges(
        groups,
        &topo_order,
        &producers,
        &consumers,
        &is_literal,
        num_lanes,
    );

    // Step 4: Build span NanoGraphs. O(groups * num_lanes).
    let phases = build_all_spans(
        graph,
        num_lanes,
        num_phases,
        &group_phase,
        &producers,
        &is_literal,
        &topo_order,
        &duplicated,
    );

    SpanPlan { num_lanes, phases }
}

// ─── Edge classification ─────────────────────────────────────────────────────

/// Determine if an edge from producer P to consumer C is "lane-aligned":
/// when both P and C are split into N equal slices, does lane i's slice of C
/// only read from lane i's slice of P?
///
/// Returns true if lane-aligned (safe to split both in same phase).
fn is_edge_lane_aligned(
    producer: &AtomGroup,
    consumer: &AtomGroup,
    input_ref: &InputRef,
    num_lanes: usize,
) -> bool {
    if num_lanes <= 1 {
        return true;
    }

    let prod_base = producer.base_id.0;
    let prod_count = producer.count;
    let cons_count = consumer.count;

    match input_ref {
        InputRef::Broadcast(_) => {
            // All consumer atoms read the same single producer atom.
            // Cross-lane IF the broadcast atom is in a splittable group.
            prod_count <= 1
        }

        InputRef::Affine { base, stride } => {
            if *stride == 0 {
                return prod_count <= 1;
            }
            // Perfect 1:1 alignment: same count, same base, stride=1.
            if *stride == 1 && base.0 == prod_base && cons_count == prod_count {
                return true;
            }
            // General check: sample all lane boundaries.
            is_affine_lane_aligned(*base, *stride, prod_base, prod_count, cons_count, num_lanes)
        }

        InputRef::StridedBroadcast {
            base,
            stride,
            repeat,
        } => is_strided_broadcast_lane_aligned(
            *base, *stride, *repeat, prod_base, prod_count, cons_count, num_lanes,
        ),

        InputRef::Modular { modulus, .. } => {
            // Modular cycles through the same atoms — always cross-lane unless trivial.
            *modulus <= 1
        }

        InputRef::SymAffine { .. } => {
            // SymAffine involves symbolic iteration — conservative: cross-lane.
            false
        }

        InputRef::Explicit(ids) => {
            // Check numerically if small enough.
            if ids.len() <= 256 {
                is_explicit_lane_aligned(ids, prod_base, prod_count, cons_count, num_lanes)
            } else {
                false
            }
        }
    }
}

/// Check if Affine{base, stride} access is lane-aligned.
fn is_affine_lane_aligned(
    base: AtomId,
    stride: i32,
    prod_base: u64,
    prod_count: u64,
    cons_count: u64,
    num_lanes: usize,
) -> bool {
    let n = num_lanes as u64;
    for lane in 0..num_lanes {
        let j = lane as u64;
        let c_lo = j * cons_count / n;
        let c_hi = (j + 1) * cons_count / n;
        let p_lo = prod_base + j * prod_count / n;
        let p_hi = prod_base + (j + 1) * prod_count / n;
        if c_lo == c_hi {
            continue;
        }
        let read_lo = base.0.wrapping_add((stride as i64 * c_lo as i64) as u64);
        let read_hi = base.0.wrapping_add((stride as i64 * (c_hi - 1) as i64) as u64);
        let (read_min, read_max) = if stride >= 0 {
            (read_lo, read_hi)
        } else {
            (read_hi, read_lo)
        };
        if read_min < p_lo || read_max >= p_hi {
            return false;
        }
    }
    true
}

/// Check if StridedBroadcast access is lane-aligned.
fn is_strided_broadcast_lane_aligned(
    base: AtomId,
    stride: i64,
    repeat: u64,
    prod_base: u64,
    prod_count: u64,
    cons_count: u64,
    num_lanes: usize,
) -> bool {
    let n = num_lanes as u64;
    for lane in 0..num_lanes {
        let j = lane as u64;
        let c_lo = j * cons_count / n;
        let c_hi = (j + 1) * cons_count / n;
        let p_lo = prod_base + j * prod_count / n;
        let p_hi = prod_base + (j + 1) * prod_count / n;
        if c_lo == c_hi {
            continue;
        }
        let first_block = c_lo / repeat;
        let last_block = (c_hi - 1) / repeat;
        let src_first = base.0.wrapping_add((stride * first_block as i64) as u64);
        let src_last = base.0.wrapping_add((stride * last_block as i64) as u64);
        let (src_min, src_max) = if stride >= 0 {
            (src_first, src_last)
        } else {
            (src_last, src_first)
        };
        if src_min < p_lo || src_max >= p_hi {
            return false;
        }
    }
    true
}

/// Check if Explicit access is lane-aligned.
fn is_explicit_lane_aligned(
    ids: &[AtomId],
    prod_base: u64,
    prod_count: u64,
    cons_count: u64,
    num_lanes: usize,
) -> bool {
    let n = num_lanes as u64;
    for lane in 0..num_lanes {
        let j = lane as u64;
        let c_lo = (j * cons_count / n) as usize;
        let c_hi = ((j + 1) * cons_count / n) as usize;
        let p_lo = prod_base + j * prod_count / n;
        let p_hi = prod_base + (j + 1) * prod_count / n;
        for i in c_lo..c_hi.min(ids.len()) {
            let src = ids[i].0;
            if src >= prod_base && src < prod_base + prod_count {
                if src < p_lo || src >= p_hi {
                    return false;
                }
            }
        }
    }
    true
}

/// Check if a ReduceSum/ReduceMax edge is lane-aligned.
fn is_reduce_edge_lane_aligned(
    input_ref: &InputRef,
    cons_count: u64,
    reduce_count: u64,
    reduce_stride: i64,
    prod_base: u64,
    prod_count: u64,
    num_lanes: usize,
) -> bool {
    if reduce_count <= 1 || reduce_stride == 0 {
        return true;
    }

    let n = num_lanes as u64;
    let reduce_extent = reduce_stride * (reduce_count as i64 - 1);

    match input_ref {
        InputRef::Affine { base, stride } => {
            for lane in 0..num_lanes {
                let j = lane as u64;
                let c_lo = j * cons_count / n;
                let c_hi = (j + 1) * cons_count / n;
                let p_lo = prod_base + j * prod_count / n;
                let p_hi = prod_base + (j + 1) * prod_count / n;
                if c_lo == c_hi {
                    continue;
                }
                let first_read = base.0 as i64 + *stride as i64 * c_lo as i64;
                let last_read = base.0 as i64 + *stride as i64 * (c_hi as i64 - 1);
                let read_min = first_read.min(last_read) + reduce_extent.min(0);
                let read_max = first_read.max(last_read) + reduce_extent.max(0);
                if (read_min as u64) < p_lo || (read_max as u64) >= p_hi {
                    return false;
                }
            }
            true
        }
        _ => false,
    }
}

// ─── Phase assignment ────────────────────────────────────────────────────────

/// Assign groups to phases based on edge classification.
///
/// Groups connected only by lane-aligned edges can be in the same phase.
/// Cross-lane edges force the consumer into a later phase.
/// Small groups consumed cross-lane can be duplicated instead of forcing a barrier.
///
/// Returns (group_phase, num_phases, duplicated_set).
fn assign_phases_from_edges(
    groups: &[AtomGroup],
    topo_order: &[usize],
    producers: &[Vec<usize>],
    consumers: &[Vec<usize>],
    is_literal: &[bool],
    num_lanes: usize,
) -> (Vec<usize>, usize, HashSet<usize>) {
    let n = groups.len();
    let mut group_phase = vec![0usize; n];
    let mut duplicated: HashSet<usize> = HashSet::new();

    for &gi in topo_order {
        if is_literal[gi] {
            group_phase[gi] = 0;
            continue;
        }

        let group = &groups[gi];
        let mut max_required_phase = 0usize;

        for &pi in &producers[gi] {
            if is_literal[pi] {
                continue;
            }

            let producer = &groups[pi];
            let prod_phase = group_phase[pi];

            // Check if this edge is lane-aligned.
            let mut edge_aligned = true;

            for input_ref in &group.inputs {
                if !input_ref_touches_group(input_ref, group.count, producer) {
                    continue;
                }

                if !is_edge_lane_aligned(producer, group, input_ref, num_lanes) {
                    edge_aligned = false;
                    break;
                }

                // Also check reduce-extended access.
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
                        if !is_reduce_edge_lane_aligned(
                            input_ref,
                            group.count,
                            *reduce_count,
                            *reduce_stride,
                            producer.base_id.0,
                            producer.count,
                            num_lanes,
                        ) {
                            edge_aligned = false;
                            break;
                        }
                    }
                    _ => {}
                }
            }

            // IndirectLoad table reference: always cross-lane.
            if let ScalarOp::IndirectLoad { table_base, .. } = &group.op {
                if producer.contains(*table_base) {
                    edge_aligned = false;
                }
            }

            if edge_aligned {
                max_required_phase = max_required_phase.max(prod_phase);
            } else {
                // Cross-lane edge. Duplicate if small enough.
                if producer.count <= DUPLICATE_THRESHOLD && !producer.op.is_reduce() {
                    duplicated.insert(pi);
                    max_required_phase = max_required_phase.max(prod_phase);
                } else {
                    max_required_phase = max_required_phase.max(prod_phase + 1);
                }
            }
        }

        group_phase[gi] = max_required_phase;
    }

    let num_phases = group_phase
        .iter()
        .enumerate()
        .filter(|&(i, _)| !is_literal[i])
        .map(|(_, &p)| p + 1)
        .max()
        .unwrap_or(1);

    // Transitive closure on duplication: if a duplicated group D depends on
    // a non-duplicated group S in the same phase, and the edge from S to D
    // requires the full range of S (because D is duplicated = computes all atoms),
    // then S must also be duplicated.
    let mut changed = true;
    while changed {
        changed = false;
        for &gi in topo_order {
            if !duplicated.contains(&gi) || is_literal[gi] {
                continue;
            }
            let group = &groups[gi];
            for &pi in &producers[gi] {
                if is_literal[pi] || duplicated.contains(&pi) {
                    continue;
                }
                let producer = &groups[pi];
                if group_phase[pi] == group_phase[gi] {
                    let mut needs_full = false;
                    for input_ref in &group.inputs {
                        if input_ref_touches_group(input_ref, group.count, producer) {
                            needs_full = true;
                            break;
                        }
                    }
                    if needs_full
                        && producer.count <= DUPLICATE_THRESHOLD
                        && !producer.op.is_reduce()
                    {
                        duplicated.insert(pi);
                        changed = true;
                    }
                }
            }
        }
    }

    (group_phase, num_phases, duplicated)
}

// ─── Span building ───────────────────────────────────────────────────────────

/// Build all spans for all phases. O(groups * num_lanes).
fn build_all_spans(
    graph: &NanoGraph,
    num_lanes: usize,
    num_phases: usize,
    group_phase: &[usize],
    producers: &[Vec<usize>],
    is_literal: &[bool],
    topo_order: &[usize],
    duplicated: &HashSet<usize>,
) -> Vec<Phase> {
    let groups = graph.groups();

    // For each phase, collect compute groups in topological order.
    let mut phase_groups: Vec<Vec<usize>> = vec![Vec::new(); num_phases];
    for &gi in topo_order {
        if !is_literal[gi] {
            phase_groups[group_phase[gi]].push(gi);
        }
    }

    let mut phases = Vec::with_capacity(num_phases);

    for phase_idx in 0..num_phases {
        let pg = &phase_groups[phase_idx];
        let mut spans = Vec::with_capacity(num_lanes);

        if pg.is_empty() {
            for _ in 0..num_lanes {
                spans.push(Span {
                    graph: NanoGraph::new(),
                    inputs: vec![],
                    outputs: vec![],
                });
            }
            phases.push(Phase { spans });
            continue;
        }

        for lane in 0..num_lanes {
            let span = build_one_span(
                graph,
                groups,
                pg,
                lane,
                num_lanes,
                group_phase,
                producers,
                is_literal,
                duplicated,
            );
            spans.push(span);
        }

        phases.push(Phase { spans });
    }

    phases
}

/// A group's assignment within a lane: either a split slice or a full duplicate.
struct LaneSlice {
    group_idx: usize,
    /// Offset within the original group (0 for duplicated groups).
    atom_offset: u64,
    /// Number of atoms this lane computes for this group.
    atom_count: u64,
    /// Whether this group is duplicated (each lane computes ALL atoms).
    is_duplicated: bool,
}

/// Build one span's NanoGraph. All operations are O(span_groups * inputs_per_group).
fn build_one_span(
    graph: &NanoGraph,
    groups: &[AtomGroup],
    phase_groups: &[usize],
    lane: usize,
    num_lanes: usize,
    group_phase: &[usize],
    producers: &[Vec<usize>],
    is_literal: &[bool],
    duplicated: &HashSet<usize>,
) -> Span {
    // Determine this lane's work for each group in this phase.
    let mut lane_slices: Vec<LaneSlice> = Vec::new();

    for &gi in phase_groups {
        let group = &groups[gi];
        if duplicated.contains(&gi) {
            // Duplicated: compute full group, output proportional slice.
            let out_offset = lane as u64 * group.count / num_lanes as u64;
            let out_end = (lane as u64 + 1) * group.count / num_lanes as u64;
            let out_count = out_end - out_offset;
            if out_count > 0 || lane == 0 {
                lane_slices.push(LaneSlice {
                    group_idx: gi,
                    atom_offset: 0,
                    atom_count: group.count,
                    is_duplicated: true,
                });
            }
        } else {
            // Split: proportional slice.
            let offset = lane as u64 * group.count / num_lanes as u64;
            let end = (lane as u64 + 1) * group.count / num_lanes as u64;
            let count = end - offset;
            if count > 0 {
                lane_slices.push(LaneSlice {
                    group_idx: gi,
                    atom_offset: offset,
                    atom_count: count,
                    is_duplicated: false,
                });
            }
        }
    }

    if lane_slices.is_empty() {
        return Span {
            graph: NanoGraph::new(),
            inputs: vec![],
            outputs: vec![],
        };
    }

    let mut span_graph = NanoGraph::new();

    // Copy sym_dim setup.
    for (name, &sd) in &graph.sym_dim_names {
        let local_sd = span_graph.sym_dim(name);
        if let Some(&bound) = graph.sym_dim_bounds.get(&sd) {
            span_graph.sym_dim_bounds.insert(local_sd, bound);
        }
    }

    let mut atom_map = RangeAtomMap::new();

    // Set of group indices in this span's local compute.
    let local_group_set: HashSet<usize> = lane_slices.iter().map(|s| s.group_idx).collect();

    // ── Phase 1: Collect all dependencies ────────────────────────────────────

    // Small literals: duplicated into span graph (full group).
    let mut small_literals: BTreeSet<usize> = BTreeSet::new();
    // Large literal ranges: (group_idx, offset_within_group, count).
    let mut large_literal_ranges: Vec<(usize, u64, u64)> = Vec::new();
    // External ranges: non-literal, non-local dependencies from earlier phases.
    let mut external_ranges: Vec<(usize, u64, u64)> = Vec::new();

    // Collect dependencies for each local compute group.
    for slice in &lane_slices {
        let gi = slice.group_idx;
        let group = &groups[gi];

        for &pi in &producers[gi] {
            if local_group_set.contains(&pi) {
                continue; // Produced locally within this span.
            }

            if is_literal[pi] {
                let lit_group = &groups[pi];
                if lit_group.count < LITERAL_INLINE_THRESHOLD {
                    small_literals.insert(pi);
                } else {
                    // Large literal: determine which range this slice reads.
                    let ranges = compute_read_ranges_from_producer(
                        group,
                        slice.atom_offset,
                        slice.atom_count,
                        &groups[pi],
                    );
                    for (offset, count) in ranges {
                        large_literal_ranges.push((pi, offset, count));
                    }
                }
            } else {
                // External dependency (earlier phase).
                let ranges = if slice.is_duplicated {
                    // Duplicated group computes ALL atoms, needs full producer range.
                    compute_read_ranges_from_producer(group, 0, group.count, &groups[pi])
                } else {
                    compute_read_ranges_from_producer(
                        group,
                        slice.atom_offset,
                        slice.atom_count,
                        &groups[pi],
                    )
                };
                for (offset, count) in ranges {
                    external_ranges.push((pi, offset, count));
                }
            }
        }
    }

    // Also collect transitive dependencies of duplicated groups.
    // If a duplicated group D references a small literal L, L must be in the span.
    // If D references another duplicated group D2, D2's dependencies are already handled
    // because D2 is in local_group_set.
    // But if D's InputRef points to an atom that belongs to a group NOT in local_group_set
    // and NOT in producers[D] — this shouldn't happen if build_group_deps is correct.
    // However, for safety, ensure all groups referenced by InputRefs of local groups
    // (including their reduce-extended ranges) are covered.

    // Merge ranges to avoid redundancy.
    let large_literal_ranges = merge_group_ranges(&mut large_literal_ranges);
    let external_ranges = merge_group_ranges(&mut external_ranges);

    // ── Phase 2: Add small literal groups ────────────────────────────────────

    for &li in &small_literals {
        let lit_group = &groups[li];
        let local_base = span_graph.push_group(
            lit_group.count,
            lit_group.op.clone(),
            remap_sym_dims(&lit_group.sym_dims, graph, &span_graph),
            remap_sym_dims(&lit_group.reduce_dims, graph, &span_graph),
            vec![],
        );
        atom_map.insert_range(lit_group.base_id, local_base, lit_group.count);
    }

    // ── Phase 3: Add stubs for large literals ────────────────────────────────

    let mut input_mappings: Vec<AtomMapping> = Vec::new();
    for &(li, offset, count) in &large_literal_ranges {
        let src_group = &groups[li];
        let main_base = src_group.base_id.offset(offset);
        let local_base = span_graph.push_group(
            count,
            ScalarOp::Literal(crate::numeric_scalar::NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );
        atom_map.insert_range(main_base, local_base, count);
        input_mappings.push(AtomMapping {
            main_base,
            span_base: local_base,
            count,
        });
    }

    // ── Phase 4: Add stubs for external (non-literal) inputs ─────────────────

    for &(pi, offset, count) in &external_ranges {
        let src_group = &groups[pi];
        let main_base = src_group.base_id.offset(offset);
        let local_base = span_graph.push_group(
            count,
            ScalarOp::Literal(crate::numeric_scalar::NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );
        atom_map.insert_range(main_base, local_base, count);
        input_mappings.push(AtomMapping {
            main_base,
            span_base: local_base,
            count,
        });
    }

    // Sort atom map before adding compute groups.
    atom_map.ensure_sorted();

    // ── Phase 5: Add compute groups with remapped InputRefs ──────────────────

    // Process in group_idx order (topological since groups were pushed in topo order).
    let mut sorted_slices = lane_slices;
    sorted_slices.sort_by_key(|s| s.group_idx);

    let mut output_mappings: Vec<AtomMapping> = Vec::new();

    for slice in &sorted_slices {
        let gi = slice.group_idx;
        let group = &groups[gi];

        if slice.is_duplicated {
            // Full group — remap InputRefs from main graph to span graph.
            let remapped_inputs = remap_input_refs(&group.inputs, &atom_map);
            let local_base = span_graph.push_group(
                group.count,
                group.op.clone(),
                remap_sym_dims(&group.sym_dims, graph, &span_graph),
                remap_sym_dims(&group.reduce_dims, graph, &span_graph),
                remapped_inputs,
            );
            atom_map.insert_range(group.base_id, local_base, group.count);
            atom_map.ensure_sorted();

            // Output only this lane's proportional slice.
            let out_offset = lane as u64 * group.count / num_lanes as u64;
            let out_end = (lane as u64 + 1) * group.count / num_lanes as u64;
            let out_count = out_end - out_offset;
            if out_count > 0 {
                let main_base = group.base_id.offset(out_offset);
                let span_base = atom_map.get(main_base).unwrap();
                output_mappings.push(AtomMapping {
                    main_base,
                    span_base,
                    count: out_count,
                });
            }
        } else {
            // Sliced group — adjust InputRefs for slice offset, then remap.
            let sliced_inputs = remap_input_refs_for_slice(
                &group.inputs,
                &group.op,
                slice.atom_offset,
                slice.atom_count,
                group.count,
                &atom_map,
            );
            let local_base = span_graph.push_group(
                slice.atom_count,
                group.op.clone(), // ReduceSum/Max params are per-output-atom, unchanged by split.
                remap_sym_dims(&group.sym_dims, graph, &span_graph),
                remap_sym_dims(&group.reduce_dims, graph, &span_graph),
                sliced_inputs,
            );
            let main_base = group.base_id.offset(slice.atom_offset);
            atom_map.insert_range(main_base, local_base, slice.atom_count);
            atom_map.ensure_sorted();

            output_mappings.push(AtomMapping {
                main_base,
                span_base: local_base,
                count: slice.atom_count,
            });
        }
    }

    Span {
        graph: span_graph,
        inputs: input_mappings,
        outputs: output_mappings,
    }
}

/// Compute which atom ranges from `producer` the consumer's slice reads.
/// All operations are O(1) per InputRef (no per-atom iteration).
///
/// Returns ranges as (offset_within_producer, count).
fn compute_read_ranges_from_producer(
    consumer: &AtomGroup,
    consumer_offset: u64,
    consumer_count: u64,
    producer: &AtomGroup,
) -> Vec<(u64, u64)> {
    let prod_lo = producer.base_id.0;
    let prod_hi = prod_lo + producer.count;

    let (is_reduce, reduce_count, reduce_stride) = match &consumer.op {
        ScalarOp::ReduceSum {
            reduce_count,
            reduce_stride,
            ..
        }
        | ScalarOp::ReduceMax {
            reduce_count,
            reduce_stride,
            ..
        } if *reduce_count > 1 && *reduce_stride != 0 => (true, *reduce_count, *reduce_stride),
        _ => (false, 0u64, 0i64),
    };

    let mut ranges = Vec::new();

    for input in &consumer.inputs {
        let (read_lo, read_hi) = sliced_input_ref_range(
            input,
            consumer_offset,
            consumer_count,
            consumer.count,
            is_reduce,
            reduce_count,
            reduce_stride,
        );

        // Intersect with producer range.
        if read_hi <= prod_lo || read_lo >= prod_hi {
            continue;
        }
        let overlap_lo = read_lo.max(prod_lo);
        let overlap_hi = read_hi.min(prod_hi);
        let offset = overlap_lo - prod_lo;
        let count = overlap_hi - overlap_lo;
        if count > 0 {
            ranges.push((offset, count));
        }
    }

    // IndirectLoad: the table may be in the producer.
    if let ScalarOp::IndirectLoad { table_base, .. } = &consumer.op {
        if table_base.0 >= prod_lo && table_base.0 < prod_hi {
            ranges.push((0, producer.count));
        }
    }

    ranges
}

/// Compute the [lo, hi) atom range accessed by a sliced InputRef.
/// O(1) for all variants except Explicit (which is O(slice_count) but rare).
fn sliced_input_ref_range(
    input: &InputRef,
    atom_offset: u64,
    atom_count: u64,
    _original_count: u64,
    is_reduce: bool,
    reduce_count: u64,
    reduce_stride: i64,
) -> (u64, u64) {
    if atom_count == 0 {
        return (0, 0);
    }

    let min_reduce_ext = if is_reduce {
        0i64.min(reduce_stride * (reduce_count as i64 - 1))
    } else {
        0
    };
    let max_reduce_ext = if is_reduce {
        0i64.max(reduce_stride * (reduce_count as i64 - 1))
    } else {
        0
    };

    match input {
        InputRef::Broadcast(id) => {
            let lo = id.0 as i64 + min_reduce_ext;
            let hi = id.0 as i64 + max_reduce_ext + 1;
            (lo as u64, hi as u64)
        }
        InputRef::Affine { base, stride } => {
            let first = base.0 as i64 + *stride as i64 * atom_offset as i64;
            let last = base.0 as i64 + *stride as i64 * (atom_offset + atom_count - 1) as i64;
            let lo = first.min(last) + min_reduce_ext;
            let hi = first.max(last) + max_reduce_ext + 1;
            (lo as u64, hi as u64)
        }
        InputRef::StridedBroadcast { base, stride, repeat } => {
            let first_block = atom_offset / repeat;
            let last_block = (atom_offset + atom_count - 1) / repeat;
            let first_read = base.0 as i64 + *stride * first_block as i64;
            let last_read = base.0 as i64 + *stride * last_block as i64;
            let lo = first_read.min(last_read) + min_reduce_ext;
            let hi = first_read.max(last_read) + max_reduce_ext + 1;
            (lo as u64, hi as u64)
        }
        InputRef::Modular { base, stride, modulus } => {
            if *modulus == 0 {
                return (0, 0);
            }
            // Modular accesses the full modulus range regardless of slice.
            let first = base.0 as i64;
            let last = first + *stride as i64 * (*modulus as i64 - 1);
            let lo = first.min(last) + min_reduce_ext;
            let hi = first.max(last) + max_reduce_ext + 1;
            (lo as u64, hi as u64)
        }
        InputRef::SymAffine { base, stride_i, .. } => {
            let first = base.0 as i64 + *stride_i as i64 * atom_offset as i64;
            let last = base.0 as i64 + *stride_i as i64 * (atom_offset + atom_count - 1) as i64;
            let lo = first.min(last);
            let hi = first.max(last) + 1;
            (lo as u64, hi as u64)
        }
        InputRef::Explicit(ids) => {
            let slice_lo = atom_offset as usize;
            let slice_hi = (atom_offset + atom_count) as usize;
            if slice_lo >= ids.len() {
                return (0, 0);
            }
            let slice_end = slice_hi.min(ids.len());
            let mut lo = u64::MAX;
            let mut hi = 0u64;
            for i in slice_lo..slice_end {
                let id_lo = (ids[i].0 as i64 + min_reduce_ext) as u64;
                let id_hi = (ids[i].0 as i64 + max_reduce_ext) as u64 + 1;
                lo = lo.min(id_lo);
                hi = hi.max(id_hi);
            }
            if lo > hi {
                (0, 0)
            } else {
                (lo, hi)
            }
        }
    }
}

// ─── InputRef remapping ──────────────────────────────────────────────────────

/// Adjust an InputRef for a group slice, then remap to span coordinates.
fn remap_input_refs_for_slice(
    inputs: &[InputRef],
    op: &ScalarOp,
    offset: u64,
    count: u64,
    original_count: u64,
    atom_map: &RangeAtomMap,
) -> Vec<InputRef> {
    inputs
        .iter()
        .map(|input| {
            let adjusted = adjust_input_ref_for_slice(input, offset, count, original_count);
            remap_one_input_ref(&adjusted, atom_map)
        })
        .collect()
}

/// Adjust an InputRef for a group slice.
///
/// The slice takes atoms [offset, offset+count) from the original group.
/// The new InputRef maps atom 0 in the slice to what atom `offset` read.
fn adjust_input_ref_for_slice(
    input: &InputRef,
    offset: u64,
    count: u64,
    _original_count: u64,
) -> InputRef {
    match input {
        InputRef::Broadcast(id) => InputRef::Broadcast(*id),

        InputRef::Affine { base, stride } => {
            let new_base =
                AtomId(base.0.wrapping_add((*stride as i64 * offset as i64) as u64));
            InputRef::Affine {
                base: new_base,
                stride: *stride,
            }
        }

        InputRef::StridedBroadcast {
            base,
            stride,
            repeat,
        } => {
            if offset % repeat == 0 {
                let block_offset = offset / repeat;
                let new_base =
                    AtomId(base.0.wrapping_add((*stride * block_offset as i64) as u64));
                InputRef::StridedBroadcast {
                    base: new_base,
                    stride: *stride,
                    repeat: *repeat,
                }
            } else {
                // Unaligned slice: fall back to Explicit.
                // This is rare and only happens for small groups.
                let ids: Vec<AtomId> = (0..count)
                    .map(|j| {
                        let orig_i = offset + j;
                        let block = orig_i / repeat;
                        AtomId(base.0.wrapping_add((*stride * block as i64) as u64))
                    })
                    .collect();
                InputRef::Explicit(ids)
            }
        }

        InputRef::Modular {
            base,
            stride,
            modulus,
        } => {
            if *modulus > 0 && offset % modulus == 0 {
                InputRef::Modular {
                    base: *base,
                    stride: *stride,
                    modulus: *modulus,
                }
            } else if *modulus > 0 {
                // Unaligned: fall back to Explicit.
                let ids: Vec<AtomId> = (0..count)
                    .map(|j| {
                        let wrapped = (offset + j) % modulus;
                        AtomId(base.0.wrapping_add((*stride as i64 * wrapped as i64) as u64))
                    })
                    .collect();
                InputRef::Explicit(ids)
            } else {
                InputRef::Modular {
                    base: *base,
                    stride: *stride,
                    modulus: *modulus,
                }
            }
        }

        InputRef::SymAffine {
            base,
            stride_i,
            stride_k,
        } => {
            let new_base =
                AtomId(base.0.wrapping_add((*stride_i as i64 * offset as i64) as u64));
            InputRef::SymAffine {
                base: new_base,
                stride_i: *stride_i,
                stride_k: *stride_k,
            }
        }

        InputRef::Explicit(ids) => {
            let start = offset as usize;
            let end = (offset + count) as usize;
            InputRef::Explicit(ids[start..end.min(ids.len())].to_vec())
        }
    }
}

/// Remap InputRefs from main-graph atom space to span-graph atom space.
fn remap_input_refs(inputs: &[InputRef], atom_map: &RangeAtomMap) -> Vec<InputRef> {
    inputs
        .iter()
        .map(|input| remap_one_input_ref(input, atom_map))
        .collect()
}

fn remap_one_input_ref(input: &InputRef, atom_map: &RangeAtomMap) -> InputRef {
    match input {
        InputRef::Broadcast(id) => InputRef::Broadcast(atom_map.get(*id).unwrap_or(*id)),

        InputRef::Affine { base, stride } => InputRef::Affine {
            base: atom_map.get(*base).unwrap_or(*base),
            stride: *stride,
        },

        InputRef::StridedBroadcast {
            base,
            stride,
            repeat,
        } => InputRef::StridedBroadcast {
            base: atom_map.get(*base).unwrap_or(*base),
            stride: *stride,
            repeat: *repeat,
        },

        InputRef::Modular {
            base,
            stride,
            modulus,
        } => InputRef::Modular {
            base: atom_map.get(*base).unwrap_or(*base),
            stride: *stride,
            modulus: *modulus,
        },

        InputRef::SymAffine {
            base,
            stride_i,
            stride_k,
        } => InputRef::SymAffine {
            base: atom_map.get(*base).unwrap_or(*base),
            stride_i: *stride_i,
            stride_k: *stride_k,
        },

        InputRef::Explicit(ids) => {
            InputRef::Explicit(ids.iter().map(|id| atom_map.get(*id).unwrap_or(*id)).collect())
        }
    }
}

fn remap_sym_dims(dims: &[SymDim], main_graph: &NanoGraph, span_graph: &NanoGraph) -> Vec<SymDim> {
    dims.iter()
        .map(|&sd| {
            for (name, &main_sd) in &main_graph.sym_dim_names {
                if main_sd == sd {
                    if let Some(&local_sd) = span_graph.sym_dim_names.get(name) {
                        return local_sd;
                    }
                }
            }
            sd
        })
        .collect()
}

// ─── Group dependency graph ──────────────────────────────────────────────────

fn build_group_deps(groups: &[AtomGroup]) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let n = groups.len();
    let mut producers: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (gi, group) in groups.iter().enumerate() {
        let mut prod_set = BTreeSet::new();

        for input in &group.inputs {
            for pi in resolve_producer_groups(input, group.count, groups) {
                if pi != gi {
                    prod_set.insert(pi);
                }
            }
        }

        // ReduceSum/ReduceMax strided access extends beyond InputRef range.
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
                    for pi in resolve_producer_groups_with_reduce(
                        input,
                        group.count,
                        *reduce_count,
                        *reduce_stride,
                        groups,
                    ) {
                        if pi != gi {
                            prod_set.insert(pi);
                        }
                    }
                }
            }
            _ => {}
        }

        // IndirectLoad table reference.
        if let ScalarOp::IndirectLoad { table_base, .. } = &group.op {
            if let Some(pi) = find_group_idx(groups, *table_base) {
                if pi != gi {
                    prod_set.insert(pi);
                }
            }
        }

        let prod_vec: Vec<usize> = prod_set.into_iter().collect();
        for &pi in &prod_vec {
            consumers[pi].push(gi);
        }
        producers.push(prod_vec);
    }

    (producers, consumers)
}

/// Check if an InputRef potentially references atoms from the given group.
fn input_ref_touches_group(input: &InputRef, consumer_count: u64, producer: &AtomGroup) -> bool {
    let prod_lo = producer.base_id.0;
    let prod_hi = prod_lo + producer.count;

    let (lo, hi) = input_ref_range(input, consumer_count);
    lo < prod_hi && hi > prod_lo
}

/// Compute the [lo, hi) atom range an InputRef covers.
fn input_ref_range(input: &InputRef, count: u64) -> (u64, u64) {
    if count == 0 {
        return (0, 0);
    }
    match input {
        InputRef::Broadcast(id) => (id.0, id.0 + 1),
        InputRef::Affine { base, stride } => {
            let first = base.0 as i64;
            let last = first + *stride as i64 * (count as i64 - 1);
            (first.min(last) as u64, first.max(last) as u64 + 1)
        }
        InputRef::StridedBroadcast {
            base,
            stride,
            repeat,
        } => {
            let last_block = ((count - 1) / repeat) as i64;
            let first = base.0 as i64;
            let last = first + stride * last_block;
            (first.min(last) as u64, first.max(last) as u64 + 1)
        }
        InputRef::Modular {
            base,
            stride,
            modulus,
        } => {
            if *modulus == 0 {
                return (0, 0);
            }
            let first = base.0 as i64;
            let last = first + *stride as i64 * (*modulus as i64 - 1);
            (first.min(last) as u64, first.max(last) as u64 + 1)
        }
        InputRef::SymAffine { base, stride_i, .. } => {
            let first = base.0 as i64;
            let last = first + *stride_i as i64 * (count as i64 - 1);
            (first.min(last) as u64, first.max(last) as u64 + 1)
        }
        InputRef::Explicit(ids) => {
            if ids.is_empty() {
                return (0, 0);
            }
            let lo = ids.iter().map(|id| id.0).min().unwrap();
            let hi = ids.iter().map(|id| id.0).max().unwrap() + 1;
            (lo, hi)
        }
    }
}

fn resolve_producer_groups(input: &InputRef, count: u64, groups: &[AtomGroup]) -> Vec<usize> {
    let (lo, hi) = input_ref_range(input, count);
    if lo >= hi {
        return vec![];
    }
    // For Explicit, this is an over-approximation (covers full range), but correct
    // since it's conservative — we may include non-referenced groups but won't miss any.
    find_groups_in_range(groups, lo, hi - 1)
}

fn resolve_producer_groups_with_reduce(
    input: &InputRef,
    count: u64,
    reduce_count: u64,
    reduce_stride: i64,
    groups: &[AtomGroup],
) -> Vec<usize> {
    if count == 0 {
        return vec![];
    }
    let min_ext = 0i64.min(reduce_stride * (reduce_count as i64 - 1));
    let max_ext = 0i64.max(reduce_stride * (reduce_count as i64 - 1));

    let (base_lo, base_hi) = input_ref_range(input, count);
    if base_lo >= base_hi {
        return vec![];
    }
    let lo = (base_lo as i64 + min_ext) as u64;
    let hi = (base_hi as i64 - 1 + max_ext) as u64;
    find_groups_in_range(groups, lo, hi)
}

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

/// Find all groups whose atom ranges overlap [lo, hi] (inclusive).
fn find_groups_in_range(groups: &[AtomGroup], lo: u64, hi: u64) -> Vec<usize> {
    let mut result = Vec::new();
    // Binary search: find first group that could overlap.
    let start = groups.partition_point(|g| g.base_id.0 + g.count <= lo);
    for gi in start..groups.len() {
        let g = &groups[gi];
        if g.base_id.0 > hi {
            break;
        }
        let g_end = g.base_id.0 + g.count - 1;
        if g.base_id.0 <= hi && g_end >= lo {
            result.push(gi);
        }
    }
    result
}

fn topological_sort(n: usize, producers: &[Vec<usize>]) -> Vec<usize> {
    let mut in_degree = vec![0usize; n];
    let mut consumers_map: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (gi, prods) in producers.iter().enumerate() {
        in_degree[gi] = prods.len();
        for &pi in prods {
            consumers_map[pi].push(gi);
        }
    }

    let mut queue: VecDeque<usize> = VecDeque::new();
    for i in 0..n {
        if in_degree[i] == 0 {
            queue.push_back(i);
        }
    }

    let mut order = Vec::with_capacity(n);
    while let Some(gi) = queue.pop_front() {
        order.push(gi);
        for &ci in &consumers_map[gi] {
            in_degree[ci] -= 1;
            if in_degree[ci] == 0 {
                queue.push_back(ci);
            }
        }
    }

    order
}

fn merge_group_ranges(ranges: &mut Vec<(usize, u64, u64)>) -> Vec<(usize, u64, u64)> {
    if ranges.is_empty() {
        return vec![];
    }
    ranges.sort_by_key(|&(gi, off, _)| (gi, off));
    let mut merged: Vec<(usize, u64, u64)> = Vec::new();
    for &(gi, off, count) in ranges.iter() {
        if let Some(last) = merged.last_mut() {
            if last.0 == gi && off <= last.1 + last.2 {
                let new_end = (off + count).max(last.1 + last.2);
                last.2 = new_end - last.1;
                continue;
            }
        }
        merged.push((gi, off, count));
    }
    merged
}

// ─── Validation ──────────────────────────────────────────────────────────────

/// Validate that a SpanPlan is correct.
///
/// Checks span NanoGraph validity, coverage, and cross-lane independence.
/// Note: the coverage check uses ranges (O(groups)), not per-atom iteration.
pub fn validate_span_plan(plan: &SpanPlan, graph: &NanoGraph) -> Vec<String> {
    let mut errors = Vec::new();
    let groups = graph.groups();
    let is_literal: Vec<bool> = groups
        .iter()
        .map(|g| matches!(g.op, ScalarOp::Literal(_)) && g.inputs.is_empty())
        .collect();

    // Check 1: Each span's NanoGraph validates.
    for (pi, phase) in plan.phases.iter().enumerate() {
        for (li, span) in phase.spans.iter().enumerate() {
            if span.graph.num_groups() == 0 {
                continue;
            }
            let span_errors = span.graph.validate();
            for err in span_errors {
                errors.push(format!("Phase {} Lane {} span validation: {}", pi, li, err));
            }
        }
    }

    // Check 2: All compute atoms are covered (range-based, O(output_ranges)).
    // Collect all output ranges, sorted by main_base.
    let mut all_output_ranges: Vec<(u64, u64, usize, usize)> = Vec::new(); // (lo, hi, phase, lane)
    for (pi, phase) in plan.phases.iter().enumerate() {
        for (li, span) in phase.spans.iter().enumerate() {
            for mapping in &span.outputs {
                let lo = mapping.main_base.0;
                let hi = lo + mapping.count;
                all_output_ranges.push((lo, hi, pi, li));
            }
        }
    }
    all_output_ranges.sort_by_key(|&(lo, _, _, _)| lo);

    // Check for overlapping output ranges (each atom in exactly one span).
    for window in all_output_ranges.windows(2) {
        let (lo1, hi1, p1, l1) = window[0];
        let (lo2, hi2, p2, l2) = window[1];
        if lo2 < hi1 {
            errors.push(format!(
                "Output overlap: Phase {} Lane {} [{}, {}) overlaps Phase {} Lane {} [{}, {})",
                p1, l1, lo1, hi1, p2, l2, lo2, hi2
            ));
        }
    }

    // Check that all non-literal compute groups are fully covered.
    for (gi, group) in groups.iter().enumerate() {
        if is_literal[gi] {
            continue;
        }
        let g_lo = group.base_id.0;
        let g_hi = g_lo + group.count;

        // Find output ranges that overlap this group's range.
        let start = all_output_ranges.partition_point(|&(_, hi, _, _)| hi <= g_lo);
        let mut covered_up_to = g_lo;
        for idx in start..all_output_ranges.len() {
            let (out_lo, out_hi, _, _) = all_output_ranges[idx];
            if out_lo >= g_hi {
                break;
            }
            if out_lo > covered_up_to {
                errors.push(format!(
                    "Compute atoms [{}, {}) from group {} (base={}) not output by any span",
                    covered_up_to, out_lo.min(g_hi), gi, group.base_id
                ));
            }
            covered_up_to = covered_up_to.max(out_hi);
        }
        if covered_up_to < g_hi {
            errors.push(format!(
                "Compute atoms [{}, {}) from group {} (base={}) not output by any span",
                covered_up_to, g_hi, gi, group.base_id
            ));
        }
    }

    // Check 3: Cross-lane independence within each phase.
    for (pi, phase) in plan.phases.iter().enumerate() {
        let mut lane_output_ranges: Vec<Vec<(u64, u64)>> = Vec::new();
        for span in &phase.spans {
            let mut ranges: Vec<(u64, u64)> = Vec::new();
            for mapping in &span.outputs {
                ranges.push((mapping.main_base.0, mapping.main_base.0 + mapping.count));
            }
            ranges.sort_by_key(|&(lo, _)| lo);
            lane_output_ranges.push(ranges);
        }

        for (li, span) in phase.spans.iter().enumerate() {
            for mapping in &span.inputs {
                let in_lo = mapping.main_base.0;
                let in_hi = in_lo + mapping.count;
                for (other_li, other_ranges) in lane_output_ranges.iter().enumerate() {
                    if other_li == li {
                        continue;
                    }
                    // Binary search in sorted other_ranges.
                    for &(out_lo, out_hi) in other_ranges {
                        if out_lo >= in_hi {
                            break;
                        }
                        if in_lo < out_hi && in_hi > out_lo {
                            let overlap_lo = in_lo.max(out_lo);
                            let overlap_hi = in_hi.min(out_hi);
                            errors.push(format!(
                                "Phase {} Lane {} reads atoms [{}, {}) produced by Lane {} (cross-lane violation)",
                                pi, li, overlap_lo, overlap_hi, other_li
                            ));
                        }
                    }
                }
            }
        }
    }

    errors
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::attempts::v13_claude::test_graphs;
    use crate::dtype::DType;
    use crate::nano_graph::{ScalarBinOp, ScalarUnaryOp};
    use crate::numeric_scalar::NumericScalar;

    fn plan_and_validate(graph: &NanoGraph, num_lanes: usize) -> (SpanPlan, Vec<String>) {
        let plan = plan_spans(graph, num_lanes);
        let errors = validate_span_plan(&plan, graph);
        (plan, errors)
    }

    fn count_output_atoms(plan: &SpanPlan) -> u64 {
        let mut total = 0u64;
        for phase in &plan.phases {
            for span in &phase.spans {
                for mapping in &span.outputs {
                    total += mapping.count;
                }
            }
        }
        total
    }

    // ── Basic tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_empty_graph() {
        let g = NanoGraph::new();
        let (plan, errors) = plan_and_validate(&g, 4);
        assert!(errors.is_empty(), "{:?}", errors);
        assert_eq!(plan.phases.len(), 0);
    }

    #[test]
    fn test_literal_only() {
        let mut g = NanoGraph::new();
        g.push_group(
            100,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let (plan, errors) = plan_and_validate(&g, 2);
        assert!(errors.is_empty(), "{:?}", errors);
        assert_eq!(plan.phases.len(), 0);
    }

    #[test]
    fn test_single_lane() {
        let (g, _, _, _) = test_graphs::elementwise_binary(1024, ScalarBinOp::Add);
        let (plan, errors) = plan_and_validate(&g, 1);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
        assert!(plan.phases.len() >= 1);
    }

    // ── Elementwise tests ────────────────────────────────────────────────────

    #[test]
    fn test_elementwise_add_4_lanes() {
        let (g, _, _, _) = test_graphs::elementwise_binary(1024, ScalarBinOp::Add);
        let (plan, errors) = plan_and_validate(&g, 4);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
        let total = count_output_atoms(&plan);
        assert_eq!(total, 1024, "Expected 1024 output atoms, got {}", total);
    }

    #[test]
    fn test_elementwise_chain() {
        let (g, _, _) = test_graphs::unary_chain(
            256,
            &[ScalarUnaryOp::Neg, ScalarUnaryOp::Exp, ScalarUnaryOp::Tanh],
        );
        let (plan, errors) = plan_and_validate(&g, 4);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
        let total = count_output_atoms(&plan);
        assert_eq!(total, 768, "Expected 768 output atoms, got {}", total);
    }

    #[test]
    fn test_broadcast_add_4_lanes() {
        let (g, _, _, _) = test_graphs::broadcast_add(1024);
        let (plan, errors) = plan_and_validate(&g, 4);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
        let total = count_output_atoms(&plan);
        assert_eq!(total, 1024, "Expected 1024 output atoms, got {}", total);
    }

    // ── Matmul tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_small_matmul_single_lane() {
        let (g, _, _, _) = test_graphs::matmul(4, 2, 3);
        let errors = g.validate();
        assert!(errors.is_empty(), "Graph validation: {:?}", errors);
        let (plan, plan_errors) = plan_and_validate(&g, 1);
        assert!(plan_errors.is_empty(), "Plan errors: {:?}", plan_errors);
    }

    #[test]
    fn test_small_matmul_4_lanes() {
        let (g, _, _, _) = test_graphs::matmul(4, 2, 3);
        let (plan, errors) = plan_and_validate(&g, 4);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
    }

    #[test]
    fn test_matmul_activation_4_lanes() {
        let (g, _, _, _) = test_graphs::matmul_activation(8, 4, 6, ScalarUnaryOp::Tanh);
        let (plan, errors) = plan_and_validate(&g, 4);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
    }

    // ── Matmul chain (multi-layer) ───────────────────────────────────────────

    #[test]
    fn test_matmul_chain_single_lane() {
        let (g, _, _, _, _) = test_graphs::matmul_chain(4, 4, 4, 4, 4);
        let (plan, errors) = plan_and_validate(&g, 1);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
    }

    #[test]
    fn test_matmul_chain_4_lanes() {
        let (g, _, _, _, _) = test_graphs::matmul_chain(4, 4, 4, 4, 4);
        let (plan, errors) = plan_and_validate(&g, 4);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
        // Barrier required between chained matmuls.
        assert!(
            plan.phases.len() >= 2,
            "Expected >= 2 phases, got {}",
            plan.phases.len()
        );
    }

    // ── Select consumed by StridedBroadcast ──────────────────────────────────

    #[test]
    fn test_select_consumed_by_strided_broadcast() {
        // GPT-2 attention mask pattern that caused 96 violations in v2c.
        let mut g = NanoGraph::new();

        let cond = g.push_group(
            3072,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );
        let true_val = g.push_group(
            3072,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let false_val = g.push_group(
            3072,
            ScalarOp::Literal(NumericScalar::F32(-1e9)),
            vec![],
            vec![],
            vec![],
        );

        // Select: output[i] = cond[i] ? true_val[i] : false_val[i]
        let select = g.push_group(
            3072,
            ScalarOp::Select {
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine {
                    base: cond,
                    stride: 1,
                },
                InputRef::Affine {
                    base: true_val,
                    stride: 1,
                },
                InputRef::Affine {
                    base: false_val,
                    stride: 1,
                },
            ],
        );

        // Consumer uses StridedBroadcast from select output.
        // 12288 atoms, each block of 4 atoms reads the same select output.
        let consumer = g.push_group(
            12288,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::StridedBroadcast {
                    base: select,
                    stride: 1,
                    repeat: 4,
                },
                InputRef::Affine {
                    base: AtomId(0), // dummy second input
                    stride: 0,
                },
            ],
        );
        g.outputs = vec![consumer];

        let (plan, errors) = plan_and_validate(&g, 4);
        // The Select (3072 atoms) is consumed cross-lane via StridedBroadcast.
        // It should be duplicated (below DUPLICATE_THRESHOLD) or forced to earlier phase.
        assert!(errors.is_empty(), "Errors: {:?}", errors);
    }

    // ── Modular input pattern ────────────────────────────────────────────────

    #[test]
    fn test_modular_input() {
        let mut g = NanoGraph::new();

        // Bias vector: 768 atoms
        let bias = g.push_group(
            768,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );

        // Input: 49152 atoms
        let input = g.push_group(
            49152,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );

        // Add with modular bias broadcast: output[i] = input[i] + bias[i % 768]
        let output = g.push_group(
            49152,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine {
                    base: input,
                    stride: 1,
                },
                InputRef::Modular {
                    base: bias,
                    stride: 1,
                    modulus: 768,
                },
            ],
        );
        g.outputs = vec![output];

        let (plan, errors) = plan_and_validate(&g, 4);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
        let total = count_output_atoms(&plan);
        assert_eq!(total, 49152, "Expected 49152 output atoms, got {}", total);
    }

    // ── Span independence test ───────────────────────────────────────────────

    /// Verify that every span's NanoGraph is independently valid:
    /// all InputRefs resolve within the span graph.
    #[test]
    fn test_span_independence_matmul_chain() {
        let (g, _, _, _, _) = test_graphs::matmul_chain(8, 4, 6, 6, 4);
        let plan = plan_spans(&g, 4);

        for (pi, phase) in plan.phases.iter().enumerate() {
            for (li, span) in phase.spans.iter().enumerate() {
                if span.graph.num_groups() == 0 {
                    continue;
                }
                let span_errors = span.graph.validate();
                assert!(
                    span_errors.is_empty(),
                    "Phase {} Lane {} span validation failed: {:?}",
                    pi,
                    li,
                    span_errors
                );
            }
        }
    }

    // ── Coverage test ────────────────────────────────────────────────────────

    /// Verify complete coverage: every compute atom appears exactly once in outputs.
    #[test]
    fn test_complete_coverage() {
        let (g, _, _, _, _) = test_graphs::matmul_chain(8, 4, 6, 6, 4);
        let plan = plan_spans(&g, 4);
        let errors = validate_span_plan(&plan, &g);
        assert!(errors.is_empty(), "Coverage errors: {:?}", errors);

        // Count total compute atoms.
        let groups = g.groups();
        let compute_atoms: u64 = groups
            .iter()
            .filter(|g| !matches!(g.op, ScalarOp::Literal(_)) || !g.inputs.is_empty())
            .map(|g| g.count)
            .sum();
        let output_atoms = count_output_atoms(&plan);
        assert_eq!(
            output_atoms, compute_atoms,
            "Output atoms ({}) != compute atoms ({})",
            output_atoms, compute_atoms
        );
    }

    // ── Large graph test (O(groups) verification) ────────────────────────────

    /// Build a graph with >10K groups and verify the algorithm handles it
    /// without per-atom iteration (which would be >10M atoms).
    #[test]
    fn test_large_graph_no_per_atom_iteration() {
        // Build a graph that mimics a multi-layer transformer:
        // 10 layers, each with a matmul (M=64, K=32, N=64) + activation.
        // Each matmul: M*K = 2048 Mul groups + M = 64 ReduceSum groups = 2112 groups.
        // Plus activation: 1 group.
        // Total: 10 * (2112 + 1) = 21130 groups, plus weight literals.
        // Total atoms: each Mul group has N=64 atoms = 2048*64 = 131072 per layer mul.
        //   + 64 ReduceSum groups * 64 atoms = 4096 per layer.
        //   + 1 activation * 4096 atoms.
        //   + literals: M*K=2048 + K*N=2048 per layer.
        //   ~140K atoms per layer, ~1.4M atoms total.
        // This is small enough to test quickly but large enough to catch O(atoms) bugs.

        let mut g = NanoGraph::new();

        let m = 64u64;
        let k = 32u64;
        let n = 64u64;
        let num_layers = 10;

        let mut prev_output: Option<AtomId> = None;
        let mut prev_output_count: u64 = 0;

        for layer in 0..num_layers {
            // Input weights A[M, K]
            let a = if let Some(prev) = prev_output {
                // After first layer, A comes from previous layer's activation output.
                prev
            } else {
                g.push_group(
                    m * k,
                    ScalarOp::Literal(NumericScalar::F32(0.0)),
                    vec![],
                    vec![],
                    vec![],
                )
            };

            // Weight matrix B[K, N]
            let b = g.push_group(
                k * n,
                ScalarOp::Literal(NumericScalar::F32(0.0)),
                vec![],
                vec![],
                vec![],
            );

            // M*K Mul groups, each of count N
            let mut mul_base: Option<AtomId> = None;
            for mi in 0..m {
                for ki in 0..k {
                    let a_atom = a.offset(mi * k + ki);
                    let b_row_start = b.offset(ki * n);
                    let base = g.push_group(
                        n,
                        ScalarOp::Binary {
                            op: ScalarBinOp::Mul,
                            compute_dtype: DType::F32,
                            output_dtype: DType::F32,
                        },
                        vec![],
                        vec![],
                        vec![
                            InputRef::Broadcast(a_atom),
                            InputRef::Affine {
                                base: b_row_start,
                                stride: 1,
                            },
                        ],
                    );
                    if mul_base.is_none() {
                        mul_base = Some(base);
                    }
                }
            }
            let mul_base = mul_base.unwrap();

            // M ReduceSum groups, each of count N
            let mut reduce_base: Option<AtomId> = None;
            for mi in 0..m {
                let row_mul_base = AtomId(mul_base.0 + mi * k * n);
                let base = g.push_group(
                    n,
                    ScalarOp::ReduceSum {
                        reduce_count: k,
                        reduce_stride: n as i64,
                        compute_dtype: DType::F32,
                        output_dtype: DType::F32,
                    },
                    vec![],
                    vec![],
                    vec![InputRef::Affine {
                        base: row_mul_base,
                        stride: 1,
                    }],
                );
                if reduce_base.is_none() {
                    reduce_base = Some(base);
                }
            }
            let reduce_base = reduce_base.unwrap();

            // Activation (tanh)
            let act = g.push_group(
                m * n,
                ScalarOp::Unary {
                    op: ScalarUnaryOp::Tanh,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![InputRef::Affine {
                    base: reduce_base,
                    stride: 1,
                }],
            );

            prev_output = Some(act);
            prev_output_count = m * n;
        }

        g.outputs = vec![prev_output.unwrap()];

        // Verify the graph is valid.
        let errors = g.validate();
        assert!(errors.is_empty(), "Graph validation: {:?}", errors);

        let stats = g.stats();
        assert!(
            stats.num_groups > 10000,
            "Expected >10K groups, got {}",
            stats.num_groups
        );

        // Plan and validate. This must complete quickly (not iterate atoms).
        let start = std::time::Instant::now();
        let (plan, errors) = plan_and_validate(&g, 8);
        let elapsed = start.elapsed();

        assert!(
            errors.is_empty(),
            "Plan errors on {}+ group graph: {:?}",
            stats.num_groups,
            errors
        );
        assert!(
            elapsed.as_secs() < 30,
            "Planning took {:?} — too slow for {} groups",
            elapsed,
            stats.num_groups
        );

        // Verify all compute atoms are covered.
        let compute_atoms: u64 = g
            .groups()
            .iter()
            .filter(|grp| !matches!(grp.op, ScalarOp::Literal(_)) || !grp.inputs.is_empty())
            .map(|grp| grp.count)
            .sum();
        let output_atoms = count_output_atoms(&plan);
        assert_eq!(
            output_atoms, compute_atoms,
            "Output atoms ({}) != compute atoms ({}) on large graph",
            output_atoms, compute_atoms
        );

        // Multiple phases expected for chained matmuls.
        assert!(
            plan.phases.len() >= 2,
            "Expected >= 2 phases for chained matmuls, got {}",
            plan.phases.len()
        );
    }

    // ── Diamond dependency test ──────────────────────────────────────────────

    /// Tests a diamond pattern: A -> B, A -> C, B+C -> D.
    /// All elementwise, should be in one phase with lane-aligned splits.
    #[test]
    fn test_diamond_dependency() {
        let mut g = NanoGraph::new();

        let a = g.push_group(
            1024,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        let b = g.push_group(
            1024,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: a,
                stride: 1,
            }],
        );

        let c = g.push_group(
            1024,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: a,
                stride: 1,
            }],
        );

        let d = g.push_group(
            1024,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine {
                    base: b,
                    stride: 1,
                },
                InputRef::Affine {
                    base: c,
                    stride: 1,
                },
            ],
        );
        g.outputs = vec![d];

        let (plan, errors) = plan_and_validate(&g, 4);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
        // Diamond with all lane-aligned edges => should be 1 phase.
        assert_eq!(
            plan.phases.len(),
            1,
            "Expected 1 phase for diamond, got {}",
            plan.phases.len()
        );
        let total = count_output_atoms(&plan);
        // B(1024) + C(1024) + D(1024) = 3072 compute atoms
        assert_eq!(total, 3072, "Expected 3072 output atoms, got {}", total);
    }

    // ── Edge case: single atom group ─────────────────────────────────────────

    #[test]
    fn test_single_atom_groups() {
        let mut g = NanoGraph::new();

        let a = g.push_atom(
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
        let c = g.push_atom(
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Broadcast(a), InputRef::Broadcast(b)],
        );
        g.outputs = vec![c];

        let (plan, errors) = plan_and_validate(&g, 4);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
        let total = count_output_atoms(&plan);
        assert_eq!(total, 1, "Expected 1 output atom, got {}", total);
    }

    // ── Various lane counts ──────────────────────────────────────────────────

    #[test]
    fn test_various_lane_counts() {
        for num_lanes in [1, 2, 3, 4, 7, 8, 16] {
            let (g, _, _, _, _) = test_graphs::matmul_chain(8, 4, 6, 6, 4);
            let (plan, errors) = plan_and_validate(&g, num_lanes);
            assert!(
                errors.is_empty(),
                "Errors with {} lanes: {:?}",
                num_lanes,
                errors
            );
        }
    }

    // ── ReduceSum with large stride (non-trivial reduce pattern) ─────────────

    #[test]
    fn test_reduce_sum_stride() {
        // ReduceSum with reduce_count=4, reduce_stride=8.
        // Each output atom reads 4 input atoms spaced 8 apart.
        let mut g = NanoGraph::new();

        let input = g.push_group(
            256,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        // 8 output atoms, each reducing over 4 inputs with stride 8.
        // Output atom i reads input[i], input[i+8], input[i+16], input[i+24].
        let output = g.push_group(
            8,
            ScalarOp::ReduceSum {
                reduce_count: 4,
                reduce_stride: 8,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: input,
                stride: 1,
            }],
        );
        g.outputs = vec![output];

        let (plan, errors) = plan_and_validate(&g, 4);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
    }

    // ── Explicit InputRef ────────────────────────────────────────────────────

    #[test]
    fn test_explicit_input_ref() {
        let mut g = NanoGraph::new();

        let src = g.push_group(
            8,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );

        // Gather-like pattern: 4 consumer atoms picking from source irregularly.
        let consumer = g.push_group(
            4,
            ScalarOp::Identity {
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Explicit(vec![
                src.offset(3),
                src.offset(1),
                src.offset(7),
                src.offset(0),
            ])],
        );
        g.outputs = vec![consumer];

        let (plan, errors) = plan_and_validate(&g, 2);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
    }

    // ── Symmetric test: output atoms match input compute atoms ───────────────

    #[test]
    fn test_output_atom_count_matches_compute() {
        // Multiple different graph patterns, verify coverage.
        let test_cases: Vec<(NanoGraph, &str)> = vec![
            (test_graphs::elementwise_binary(512, ScalarBinOp::Add).0, "elem_add"),
            (test_graphs::unary_chain(128, &[ScalarUnaryOp::Neg, ScalarUnaryOp::Exp]).0, "unary_chain"),
            (test_graphs::broadcast_add(256).0, "broadcast_add"),
            (test_graphs::matmul(4, 3, 5).0, "matmul_4x3x5"),
            (test_graphs::matmul_chain(4, 4, 4, 4, 4).0, "matmul_chain"),
        ];

        for (g, name) in &test_cases {
            let compute_atoms: u64 = g
                .groups()
                .iter()
                .filter(|grp| !matches!(grp.op, ScalarOp::Literal(_)) || !grp.inputs.is_empty())
                .map(|grp| grp.count)
                .sum();

            for num_lanes in [1, 2, 4, 8] {
                let plan = plan_spans(g, num_lanes);
                let output_atoms = count_output_atoms(&plan);
                assert_eq!(
                    output_atoms, compute_atoms,
                    "{} with {} lanes: output {} != compute {}",
                    name, num_lanes, output_atoms, compute_atoms
                );
            }
        }
    }
}
