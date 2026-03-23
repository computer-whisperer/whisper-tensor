#![allow(
    clippy::all,
    dead_code,
    unreachable_code,
    unreachable_patterns,
    unused_imports,
    unused_variables
)]

//! Top-Down Tiling Partitioner (attempt I)
//!
//! Key insight: splitting is the default, not the exception. Every group is
//! assumed to be split N ways across lanes. The algorithm identifies the few
//! exceptions (literals, reductions reading across the split dimension, tiny
//! groups) and handles them specially.
//!
//! Algorithm:
//! 1. Walk the graph in topological order. For each group, decide: split,
//!    duplicate, or keep-whole.
//! 2. Split = divide atoms across lanes (elementwise, matmul mul/reduce, etc.)
//! 3. Duplicate = every lane gets a full copy (literals, very small groups)
//! 4. Keep-whole = only one lane computes it (rare — only reductions that
//!    can't be split and aren't worth duplicating)
//! 5. Phase boundaries: a new phase is needed only when a downstream group
//!    needs data from ALL lanes (fan-in). Between fan-in points, entire
//!    chains run within each lane with no barrier.
//! 6. For each lane, build a span NanoGraph with correct atom_offset values.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::dtype::DType;
use crate::graph::GlobalId;
use crate::nano_graph::pattern::InputTensor;
use crate::nano_graph::{AtomGroup, AtomId, AtomRange, InputRef, NanoGraph, ScalarOp, SymDim};

use super::types::{Phase, Span};

// ─── Classification ─────────────────────────────────────────────────────────

/// How a group should be distributed across lanes.
#[derive(Debug, Clone, PartialEq, Eq)]
enum GroupStrategy {
    /// Split across lanes: each lane gets a contiguous chunk of atoms.
    /// The group's count is divided by num_lanes.
    Split,
    /// Duplicate into every lane: each lane gets the full group.
    /// Used for literals and very small groups.
    Duplicate,
    /// Entire group on one lane. Used for reductions that read across
    /// split source data (fan-in). The output is then available to later phases.
    Whole,
}

/// Classify a group's distribution strategy.
fn classify_group(
    group: &AtomGroup,
    num_lanes: usize,
    producers: &[usize],
    strategies: &[GroupStrategy],
) -> GroupStrategy {
    let nl = num_lanes as u64;

    // Literals: always duplicate (every lane needs the full constant).
    if matches!(group.op, ScalarOp::Literal(_)) {
        return GroupStrategy::Duplicate;
    }

    // Very small groups: not worth splitting, just duplicate.
    if group.count < nl {
        return GroupStrategy::Duplicate;
    }

    // Reduce ops need special handling.
    if let ScalarOp::Reduce {
        reduce_count,
        reduce_stride,
        ..
    } = &group.op
    {
        // A reduce reads `reduce_count` atoms strided from each output atom's
        // input base. If the source was split across lanes, a single reduce atom
        // might need data from multiple lanes — that's a fan-in requiring a barrier.
        //
        // However, if the reduce's output count >= num_lanes and each output atom's
        // reduction window is independent (no cross-lane data needed), we CAN split
        // the reduce itself. This is the common matmul case: M independent ReduceSum
        // groups, each reducing over K elements that were all on the same lane.
        //
        // Heuristic: if all producers are split, and the reduce_stride is 1 (contiguous
        // reduction over a dimension), then the reduce is a "row reduce" and each
        // output element's window falls within the same lane's data IF the window
        // size divides evenly into the source split. We detect this by checking if
        // the reduction window (reduce_count * |reduce_stride|) is small relative
        // to the source group's count.
        //
        // Simple rule: if count >= num_lanes, split the reduce output across lanes.
        // Each lane computes its chunk of the M output elements. The reduce_count
        // elements for each output atom are resolved via InputRef + stride, and
        // as long as the source data is available (either produced in this lane
        // or declared as input), it works.
        //
        // The key insight: even if the source was split, the reduce's InputRef
        // resolution still works via the shared atom ID space. If a reduce atom
        // on lane 0 needs source atoms that were produced by lane 3, we need a
        // barrier. But if all source atoms are from input tensors (weights) or
        // from same-lane groups, no barrier is needed.
        //
        // For now: if count >= num_lanes, split. The phase assignment logic will
        // handle barriers if cross-lane deps exist.
        if group.count >= nl {
            return GroupStrategy::Split;
        } else {
            return GroupStrategy::Duplicate;
        }
    }

    // IndirectLoad: each lookup is independent. Split if large enough.
    if matches!(group.op, ScalarOp::IndirectLoad { .. }) {
        if group.count >= nl {
            return GroupStrategy::Split;
        } else {
            return GroupStrategy::Duplicate;
        }
    }

    // Elementwise ops (Binary, Unary, Select, Identity): always split.
    GroupStrategy::Split
}

// ─── Dependency analysis ────────────────────────────────────────────────────

/// Build group-level dependency DAG.
/// Returns (producers, successors) where:
/// - producers[gi] = list of group indices that gi depends on
/// - successors[gi] = list of group indices that depend on gi
fn build_dependency_dag(graph: &NanoGraph) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let groups = graph.groups();
    let n = groups.len();
    let mut producers: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut successors: Vec<Vec<usize>> = vec![vec![]; n];

    for (gi, group) in groups.iter().enumerate() {
        let mut seen = HashSet::new();
        graph.collect_all_producer_indices(group, gi, &mut seen);
        let deps: Vec<usize> = seen.into_iter().collect();
        for &pi in &deps {
            successors[pi].push(gi);
        }
        producers.push(deps);
    }

    (producers, successors)
}

// ─── Phase assignment ───────────────────────────────────────────────────────

/// Determine which phase each group belongs to.
///
/// A new phase (barrier) is needed when a group requires data that was
/// produced by a group in a DIFFERENT strategy partition and that data
/// crosses lane boundaries. Concretely:
///
/// - If a split group reads from another split group, they can be in the
///   same phase (each lane's slice reads from the same lane's slice of the
///   producer — no cross-lane data flow).
/// - If a split group reads from a duplicated group, same phase (each lane
///   has its own copy).
/// - If a group reads from a Whole group that's on a different conceptual
///   lane, we need a barrier. But since Whole groups are rare and their
///   output is small, we handle them by outputting and reading back.
/// - If a reduce reads source data that was split across lanes, it needs
///   all lanes' data → barrier before the reduce.
///
/// The simplest correct approach: walk topological order, track which groups
/// have been "materialized" (their output is in the value store after a
/// barrier). A group can be in the current phase if all its producers are
/// either (a) in the current phase on the same lane, or (b) materialized.
///
/// For the top-down tiling approach, we use a simpler model:
/// - All split/duplicated groups that form a chain with compatible tiling
///   go in the same phase.
/// - A barrier is needed when a group needs data from multiple lanes
///   (fan-in point).
fn assign_phases(
    groups: &[AtomGroup],
    strategies: &[GroupStrategy],
    producers: &[Vec<usize>],
    successors: &[Vec<usize>],
    num_lanes: usize,
    graph: &NanoGraph,
) -> Vec<usize> {
    let n = groups.len();
    let mut phase_of = vec![0usize; n];

    // A group needs to start a new phase if any of its producers are
    // "cross-lane" relative to it. This happens when:
    // 1. The group is Split/Duplicate but reads from a Whole group
    //    (the whole group was only on one lane, now all lanes need it).
    // 2. The group is Whole but reads from Split groups (fan-in).
    //
    // In both cases, the producer must have completed (barrier) before
    // this group can start.
    //
    // If a Split group reads from another Split group with the same
    // tiling, they are in the same phase — each lane's chunk only reads
    // from the corresponding chunk of the producer.
    //
    // If a Split group reads from a Duplicate group, same phase — each
    // lane has its copy.

    for gi in 0..n {
        let my_strat = &strategies[gi];
        let mut my_phase = 0usize;

        for &pi in &producers[gi] {
            let prod_strat = &strategies[pi];
            let prod_phase = phase_of[pi];

            let needs_barrier = match (my_strat, prod_strat) {
                // Split reads Split: same phase if tiling is compatible.
                // We assume compatible tiling (same lane gets same slice).
                (GroupStrategy::Split, GroupStrategy::Split) => false,

                // Split reads Duplicate: same phase (each lane has copy).
                (GroupStrategy::Split, GroupStrategy::Duplicate) => false,

                // Duplicate reads Split: needs barrier (duplicate needs ALL
                // of the split group's output, spanning all lanes).
                (GroupStrategy::Duplicate, GroupStrategy::Split) => true,

                // Duplicate reads Duplicate: same phase.
                (GroupStrategy::Duplicate, GroupStrategy::Duplicate) => false,

                // Whole reads anything: needs barrier if producer is Split
                // (fan-in from multiple lanes).
                (GroupStrategy::Whole, GroupStrategy::Split) => true,
                (GroupStrategy::Whole, _) => false,

                // Anything reads Whole: needs barrier (Whole output is on
                // one lane, other lanes need it).
                (_, GroupStrategy::Whole) => true,
            };

            if needs_barrier {
                my_phase = my_phase.max(prod_phase + 1);
            } else {
                my_phase = my_phase.max(prod_phase);
            }
        }

        phase_of[gi] = my_phase;
    }

    phase_of
}

// ─── Source range computation ───────────────────────────────────────────────

/// Compute the (lo, hi) inclusive source atom range for an InputRef.
fn input_ref_source_range(input: &InputRef, count: u64, atom_offset: u64) -> (u64, u64) {
    if count == 0 {
        return (0, 0);
    }
    match input {
        InputRef::Broadcast(base) => (base.0, base.0),
        InputRef::Strided { .. } => {
            let first = input.resolve(atom_offset);
            let last = input.resolve(atom_offset + count - 1);
            (first.0.min(last.0), first.0.max(last.0))
        }
        InputRef::Strided { base, stride_inner: stride, modulus, .. } => {
            let a = base.0;
            let b = (base.0 as i64 + *stride * (*modulus as i64 - 1)) as u64;
            (a.min(b), a.max(b))
        }
        InputRef::Explicit(ids) => {
            let slice = &ids[atom_offset as usize..(atom_offset + count) as usize];
            let lo = slice.iter().map(|id| id.0).min().unwrap_or(0);
            let hi = slice.iter().map(|id| id.0).max().unwrap_or(0);
            (lo, hi)
        }
    }
}

// ─── External dependency collection ─────────────────────────────────────────

/// Record an external atom range dependency.
fn record_external_range(
    ranges: &mut BTreeMap<u64, (u64, DType)>,
    base: AtomId,
    count: u64,
    dtype: DType,
) {
    ranges
        .entry(base.0)
        .and_modify(|(existing_count, _)| {
            *existing_count = (*existing_count).max(count);
        })
        .or_insert((count, dtype));
}

/// Collect external dependencies in [lo, hi] that aren't produced by
/// the span's internal atom ranges.
///
/// `internal_ranges` is a sorted (by start) list of `(start, end)` half-open
/// intervals representing atoms that will actually be present in this span
/// (from work item fragments + inlined literals).
fn collect_external_in_range(
    graph: &NanoGraph,
    lo: u64,
    hi: u64,
    fallback_dtype: DType,
    input_tensors: &[InputTensor],
    internal_ranges: &[(u64, u64)],
    needed_external: &mut BTreeMap<u64, (u64, DType)>,
) {
    // Compute uncovered sub-intervals of [lo, hi+1) (half-open) against internal_ranges.
    let mut uncovered: Vec<(u64, u64)> = Vec::new();
    let mut cursor = lo;
    let end = hi + 1;

    for &(ir_start, ir_end) in internal_ranges {
        if ir_start >= end {
            break;
        }
        if ir_end <= cursor {
            continue;
        }
        // This internal range overlaps or is ahead of cursor.
        if ir_start > cursor {
            // Gap: [cursor, ir_start) is uncovered.
            uncovered.push((cursor, ir_start.min(end)));
        }
        cursor = cursor.max(ir_end);
        if cursor >= end {
            break;
        }
    }
    if cursor < end {
        uncovered.push((cursor, end));
    }

    // For each uncovered sub-interval, find which groups/input_tensors provide those atoms.
    let groups = graph.groups();
    for (unc_lo, unc_hi) in &uncovered {
        // Check groups.
        for g in groups.iter() {
            let g_lo = g.base_id.0;
            let g_hi = g_lo + g.count;
            if g_lo >= *unc_hi {
                break;
            }
            if g_hi <= *unc_lo {
                continue;
            }
            let range_lo = g_lo.max(*unc_lo);
            let range_hi = g_hi.min(*unc_hi);
            let range_count = range_hi - range_lo;
            if range_count > 0 {
                record_external_range(
                    needed_external,
                    AtomId(range_lo),
                    range_count,
                    g.output_dtype,
                );
            }
        }

        // Check input_tensors.
        for it in input_tensors {
            let it_lo = it.base_id.0;
            let it_hi = it_lo + it.count;
            if it_lo < *unc_hi && it_hi > *unc_lo {
                let range_lo = it_lo.max(*unc_lo);
                let range_hi = it_hi.min(*unc_hi);
                let range_count = range_hi - range_lo;
                if range_count > 0 {
                    record_external_range(needed_external, AtomId(range_lo), range_count, it.dtype);
                }
            }
        }
    }
}

/// Merge overlapping/adjacent external ranges.
fn merge_external_ranges(ranges: &BTreeMap<u64, (u64, DType)>) -> Vec<(AtomId, u64, DType)> {
    let mut merged: Vec<(AtomId, u64, DType)> = Vec::new();
    for (&base, &(count, dtype)) in ranges {
        if let Some(last) = merged.last_mut() {
            let last_end = last.0.0 + last.1;
            if base <= last_end && dtype == last.2 {
                // Extend the last range.
                let new_end = (base + count).max(last_end);
                last.1 = new_end - last.0.0;
                continue;
            }
        }
        merged.push((AtomId(base), count, dtype));
    }
    merged
}

// ─── Span construction ──────────────────────────────────────────────────────

/// A work item for a lane within a phase.
struct LaneWorkItem {
    /// Index into main graph's groups().
    group_idx: usize,
    /// Offset within the original group (0 for full/duplicate groups).
    atom_offset: u64,
    /// Number of atoms this lane handles.
    atom_count: u64,
    /// Whether this is a duplicated copy (the full group, not a split fragment).
    is_duplicate: bool,
}

/// Build a single span for one lane within one phase.
fn build_span(
    graph: &NanoGraph,
    work_items: &[LaneWorkItem],
    strategies: &[GroupStrategy],
    phase_of: &[usize],
    successors: &[Vec<usize>],
    phase_idx: usize,
    num_phases: usize,
    input_tensors: &[InputTensor],
    output_group_set: &HashSet<usize>,
    all_phase_groups: &HashSet<usize>,
    lane_internal_groups: &HashSet<usize>,
) -> Span {
    let groups = graph.groups();
    let mut span_graph = NanoGraph::new();

    // Track which main-graph group indices are computed in this span.
    // For split groups, we track the original group index.
    let mut span_compute_set: HashSet<usize> = HashSet::new();
    for item in work_items {
        span_compute_set.insert(item.group_idx);
    }

    // Build the set of actual atom ranges that will be present in this span.
    // For split groups, only the lane's fragment is present. For duplicate
    // groups, the full range is present.
    // These are half-open intervals: [start, end).
    let mut internal_ranges: Vec<(u64, u64)> = Vec::new();
    for item in work_items {
        let g = &groups[item.group_idx];
        let start = g.base_id.0 + item.atom_offset;
        let end = start + item.atom_count;
        internal_ranges.push((start, end));
    }

    // Find literals that are needed by compute groups but not in the compute set.
    // These get inlined (duplicated) into the span.
    let mut needed_literals: BTreeSet<usize> = BTreeSet::new();

    // Helper: collect needed source ranges from a work item.
    // Returns a list of (lo, hi) inclusive ranges.
    let collect_source_ranges = |item: &LaneWorkItem| -> Vec<(u64, u64)> {
        let group = &groups[item.group_idx];
        let mut ranges = Vec::new();

        for input in &group.inputs {
            let (lo, hi) = input_ref_source_range(input, item.atom_count, item.atom_offset);
            ranges.push((lo, hi));
        }

        // Reduce stride extended range.
        if let ScalarOp::Reduce {
            reduce_count,
            reduce_stride,
            ..
        } = &group.op
        {
            if *reduce_count > 1 && *reduce_stride != 0 {
                for input in &group.inputs {
                    let first = input.resolve(item.atom_offset);
                    let last = input.resolve(item.atom_offset + item.atom_count - 1);
                    let end_off = (*reduce_count as i64 - 1) * reduce_stride;
                    let endpoints = [
                        first.0,
                        (first.0 as i64 + end_off) as u64,
                        last.0,
                        (last.0 as i64 + end_off) as u64,
                    ];
                    let lo = *endpoints.iter().min().unwrap();
                    let hi = *endpoints.iter().max().unwrap();
                    ranges.push((lo, hi));
                }
            }
        }

        ranges
    };

    for item in work_items {
        let source_ranges = collect_source_ranges(item);
        for (lo, hi) in &source_ranges {
            for (gi, g) in groups.iter().enumerate() {
                if g.base_id.0 > *hi {
                    break;
                }
                if g.base_id.0 + g.count <= *lo {
                    continue;
                }
                if !span_compute_set.contains(&gi)
                    && matches!(g.op, ScalarOp::Literal(_))
                    && g.inputs.is_empty()
                {
                    needed_literals.insert(gi);
                }
            }
        }

        // IndirectLoad table reference.
        let group = &groups[item.group_idx];
        if let ScalarOp::IndirectLoad { table_base, .. } = &group.op {
            if let Some(pi) = graph.find_group_idx(*table_base) {
                if !span_compute_set.contains(&pi)
                    && matches!(groups[pi].op, ScalarOp::Literal(_))
                    && groups[pi].inputs.is_empty()
                {
                    needed_literals.insert(pi);
                }
            }
        }
    }

    // Add literal ranges to internal_ranges (full groups, not split).
    for &li in &needed_literals {
        let g = &groups[li];
        internal_ranges.push((g.base_id.0, g.base_id.0 + g.count));
    }

    // Sort and merge internal ranges for efficient gap-finding.
    internal_ranges.sort_by_key(|&(start, _)| start);
    // Merge overlapping/adjacent ranges.
    let mut merged_internal: Vec<(u64, u64)> = Vec::new();
    for (start, end) in &internal_ranges {
        if let Some(last) = merged_internal.last_mut() {
            if *start <= last.1 {
                last.1 = last.1.max(*end);
                continue;
            }
        }
        merged_internal.push((*start, *end));
    }
    let internal_ranges = merged_internal;

    // Collect external dependencies: atoms needed by this span's groups
    // that are not covered by internal ranges.
    let mut needed_external: BTreeMap<u64, (u64, DType)> = BTreeMap::new();

    for item in work_items {
        let group = &groups[item.group_idx];
        let source_ranges = collect_source_ranges(item);
        for (lo, hi) in &source_ranges {
            collect_external_in_range(
                graph,
                *lo,
                *hi,
                group.output_dtype,
                input_tensors,
                &internal_ranges,
                &mut needed_external,
            );
        }

        // IndirectLoad table reference — check against internal ranges.
        if let ScalarOp::IndirectLoad { table_base, .. } = &group.op {
            let tb = table_base.0;
            // Check if the table base is covered by internal ranges.
            let is_internal = internal_ranges
                .iter()
                .any(|&(start, end)| tb >= start && tb < end);
            if !is_internal {
                if let Some(pi) = graph.find_group_idx(*table_base) {
                    let g = &groups[pi];
                    record_external_range(&mut needed_external, g.base_id, g.count, g.output_dtype);
                } else if let Some((ti, _)) = graph.find_input_idx(*table_base) {
                    let it = &input_tensors[ti];
                    record_external_range(&mut needed_external, it.base_id, it.count, it.dtype);
                }
            }
        }
    }

    let merged_external = merge_external_ranges(&needed_external);

    // Build items to insert into span graph, sorted by atom ID.
    #[derive(Debug)]
    enum InsertItem {
        ExternalInput {
            base: AtomId,
            count: u64,
            dtype: DType,
        },
        InlineLiteral {
            gi: usize,
        },
        ComputeGroup {
            gi: usize,
            atom_offset: u64,
            atom_count: u64,
        },
    }

    let mut items: Vec<(u64, InsertItem)> = Vec::new();

    for &(base, count, dtype) in &merged_external {
        items.push((base.0, InsertItem::ExternalInput { base, count, dtype }));
    }

    for &li in &needed_literals {
        items.push((groups[li].base_id.0, InsertItem::InlineLiteral { gi: li }));
    }

    for item in work_items {
        let base = groups[item.group_idx].base_id.offset(item.atom_offset);
        items.push((
            base.0,
            InsertItem::ComputeGroup {
                gi: item.group_idx,
                atom_offset: item.atom_offset,
                atom_count: item.atom_count,
            },
        ));
    }

    items.sort_by_key(|(base, _)| *base);

    // Track inserted ranges to avoid duplicates.
    let mut inserted_ranges: Vec<(u64, u64)> = Vec::new();
    let mut span_inputs: Vec<AtomRange> = Vec::new();
    let mut span_outputs: Vec<AtomRange> = Vec::new();

    for (_, item) in &items {
        match item {
            InsertItem::ExternalInput { base, count, dtype } => {
                if would_overlap(&inserted_ranges, base.0, *count) {
                    continue;
                }
                span_graph.insert_input_tensor_at(*base, GlobalId(0), *count, *dtype);
                span_inputs.push(AtomRange {
                    base: *base,
                    count: *count,
                    dtype: *dtype,
                });
                inserted_ranges.push((base.0, *count));
            }
            InsertItem::InlineLiteral { gi } => {
                let group = &groups[*gi];
                if would_overlap(&inserted_ranges, group.base_id.0, group.count) {
                    continue;
                }
                span_graph.insert_group_at(
                    group.base_id,
                    group.count,
                    group.atom_offset,
                    group.output_dtype,
                    group.op.clone(),
                    group.sym_dims.clone(),
                    group.inputs.clone(),
                );
                inserted_ranges.push((group.base_id.0, group.count));
            }
            InsertItem::ComputeGroup {
                gi,
                atom_offset,
                atom_count,
            } => {
                let group = &groups[*gi];
                let base = group.base_id.offset(*atom_offset);
                if would_overlap(&inserted_ranges, base.0, *atom_count) {
                    continue;
                }

                span_graph.insert_group_at(
                    base,
                    *atom_count,
                    *atom_offset,
                    group.output_dtype,
                    group.op.clone(),
                    group.sym_dims.clone(),
                    group.inputs.clone(),
                );
                inserted_ranges.push((base.0, *atom_count));

                // Determine if this group's output needs to be in span outputs.
                let needs_output = output_group_set.contains(gi) || {
                    let my_phase = phase_of[*gi];
                    successors[*gi].iter().any(|&si| {
                        let succ_phase = phase_of[si];
                        if succ_phase > my_phase {
                            true
                        } else if succ_phase == my_phase && !lane_internal_groups.contains(&si) {
                            // Same phase but different lane — need to output
                            // only if the successor can't get it from its own
                            // lane's duplicate. For split groups consumed by
                            // duplicate groups, the duplicate group needs ALL
                            // lanes' data, so we must output.
                            true
                        } else {
                            false
                        }
                    })
                };

                if needs_output {
                    span_outputs.push(AtomRange {
                        base,
                        count: *atom_count,
                        dtype: group.output_dtype,
                    });
                }
            }
        }
    }

    // Merge adjacent output ranges.
    span_outputs = merge_atom_ranges(span_outputs);

    Span {
        graph: span_graph,
        inputs: span_inputs,
        outputs: span_outputs,
    }
}

/// Check if a new range would overlap any already-inserted range.
fn would_overlap(inserted: &[(u64, u64)], base: u64, count: u64) -> bool {
    let end = base + count;
    for &(ib, ic) in inserted {
        let ie = ib + ic;
        if base < ie && end > ib {
            return true;
        }
    }
    false
}

/// Merge overlapping/adjacent AtomRanges.
fn merge_atom_ranges(mut ranges: Vec<AtomRange>) -> Vec<AtomRange> {
    if ranges.is_empty() {
        return ranges;
    }
    ranges.sort_by_key(|r| r.base.0);
    let mut merged: Vec<AtomRange> = vec![ranges[0].clone()];
    for r in &ranges[1..] {
        let last = merged.last_mut().unwrap();
        let last_end = last.base.0 + last.count;
        if r.base.0 <= last_end && r.dtype == last.dtype {
            let new_end = (r.base.0 + r.count).max(last_end);
            last.count = new_end - last.base.0;
        } else {
            merged.push(r.clone());
        }
    }
    merged
}

// ─── Public API ─────────────────────────────────────────────────────────────

/// Partition a NanoGraph into phases and spans for parallel execution.
///
/// Returns a sequence of phases, each containing one span per lane.
pub fn plan(
    graph: &NanoGraph,
    num_lanes: usize,
    input_tensors: &[InputTensor],
    output_atom_ids: &[AtomId],
) -> Vec<Phase> {
    let num_lanes = num_lanes.max(1);
    let groups = graph.groups();
    let n = groups.len();

    if n == 0 {
        return vec![Phase {
            spans: (0..num_lanes)
                .map(|_| Span {
                    graph: NanoGraph::new(),
                    inputs: vec![],
                    outputs: vec![],
                })
                .collect(),
        }];
    }

    // Step 1: Build dependency DAG.
    let (producers, successors) = build_dependency_dag(graph);

    // Step 2: Classify every group.
    let mut strategies: Vec<GroupStrategy> = Vec::with_capacity(n);
    for (gi, group) in groups.iter().enumerate() {
        let strat = classify_group(group, num_lanes, &producers[gi], &strategies);
        strategies.push(strat);
    }

    // Step 3: Assign phases.
    let phase_of = assign_phases(
        groups,
        &strategies,
        &producers,
        &successors,
        num_lanes,
        graph,
    );

    // Step 4: Identify output groups.
    let output_group_set = identify_output_groups(graph, output_atom_ids);

    // Step 5: Group by phase.
    let num_phases = phase_of.iter().copied().max().unwrap_or(0) + 1;
    let mut phase_groups: Vec<Vec<usize>> = vec![vec![]; num_phases];
    for (gi, &phase) in phase_of.iter().enumerate() {
        phase_groups[phase].push(gi);
    }

    // Step 6: For each phase, build work items per lane and construct spans.
    let mut phases = Vec::with_capacity(num_phases);

    for phase_idx in 0..num_phases {
        let group_indices = &phase_groups[phase_idx];

        // Build work items for each lane.
        let mut lane_work: Vec<Vec<LaneWorkItem>> = (0..num_lanes).map(|_| Vec::new()).collect();

        // Track all groups in this phase for internal-set computation.
        let all_phase_set: HashSet<usize> = group_indices.iter().copied().collect();

        for &gi in group_indices {
            let group = &groups[gi];
            match &strategies[gi] {
                GroupStrategy::Split => {
                    // Split across lanes.
                    let chunk = group.count / num_lanes as u64;
                    if chunk == 0 {
                        // Degenerate: fewer atoms than lanes. Give each atom to one lane.
                        for lane in 0..group.count.min(num_lanes as u64) as usize {
                            let offset = lane as u64;
                            lane_work[lane].push(LaneWorkItem {
                                group_idx: gi,
                                atom_offset: offset,
                                atom_count: 1,
                                is_duplicate: false,
                            });
                        }
                    } else {
                        for lane in 0..num_lanes {
                            let start = lane as u64 * chunk;
                            let count = if lane == num_lanes - 1 {
                                group.count - start
                            } else {
                                chunk
                            };
                            lane_work[lane].push(LaneWorkItem {
                                group_idx: gi,
                                atom_offset: start,
                                atom_count: count,
                                is_duplicate: false,
                            });
                        }
                    }
                }
                GroupStrategy::Duplicate => {
                    // Every lane gets the full group.
                    for lane in 0..num_lanes {
                        lane_work[lane].push(LaneWorkItem {
                            group_idx: gi,
                            atom_offset: 0,
                            atom_count: group.count,
                            is_duplicate: true,
                        });
                    }
                }
                GroupStrategy::Whole => {
                    // Assign to lane 0 (or least-loaded, but keep it simple).
                    lane_work[0].push(LaneWorkItem {
                        group_idx: gi,
                        atom_offset: 0,
                        atom_count: group.count,
                        is_duplicate: false,
                    });
                }
            }
        }

        // Sort each lane's work items by base_id for topological correctness.
        for lane in 0..num_lanes {
            lane_work[lane].sort_by_key(|item| groups[item.group_idx].base_id.0 + item.atom_offset);
        }

        // Build spans.
        let mut spans = Vec::with_capacity(num_lanes);
        for lane in 0..num_lanes {
            // Compute which group indices are internal to this lane.
            let lane_internal: HashSet<usize> =
                lane_work[lane].iter().map(|item| item.group_idx).collect();

            let span = build_span(
                graph,
                &lane_work[lane],
                &strategies,
                &phase_of,
                &successors,
                phase_idx,
                num_phases,
                input_tensors,
                &output_group_set,
                &all_phase_set,
                &lane_internal,
            );
            spans.push(span);
        }

        phases.push(Phase { spans });
    }

    phases
}

/// Identify which group indices produce output atoms.
fn identify_output_groups(graph: &NanoGraph, output_atom_ids: &[AtomId]) -> HashSet<usize> {
    let mut set = HashSet::new();
    for &atom_id in output_atom_ids {
        if let Some(gi) = graph.find_group_idx(atom_id) {
            set.insert(gi);
        }
    }
    set
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::nano_graph::ops::{ReduceKind, ScalarBinOp, ScalarUnaryOp};
    use crate::numeric_scalar::NumericScalar;

    /// Helper: assert that each phase has exactly `num_lanes` spans.
    fn assert_lane_count(phases: &[Phase], num_lanes: usize) {
        for (pi, phase) in phases.iter().enumerate() {
            assert_eq!(
                phase.spans.len(),
                num_lanes,
                "Phase {} has {} spans, expected {}",
                pi,
                phase.spans.len(),
                num_lanes,
            );
        }
    }

    /// Helper: count total atoms across all spans in a phase.
    fn phase_total_atoms(phase: &Phase) -> u64 {
        phase
            .spans
            .iter()
            .map(|s| s.graph.groups().iter().map(|g| g.count).sum::<u64>())
            .sum()
    }

    /// Helper: count atoms per lane in a phase.
    fn lane_atom_counts(phase: &Phase) -> Vec<u64> {
        phase
            .spans
            .iter()
            .map(|s| s.graph.groups().iter().map(|g| g.count).sum::<u64>())
            .collect()
    }

    /// Helper: count non-literal compute atoms per lane (excludes duplicated literals).
    fn lane_compute_atoms(phase: &Phase) -> Vec<u64> {
        phase
            .spans
            .iter()
            .map(|s| {
                s.graph
                    .groups()
                    .iter()
                    .filter(|g| !matches!(g.op, ScalarOp::Literal(_)))
                    .map(|g| g.count)
                    .sum::<u64>()
            })
            .collect()
    }

    // ─── Test: linear chain splitting ───────────────────────────────────────

    #[test]
    fn test_linear_chain_split_across_lanes() {
        // Build: Literal(8) -> Add(8) -> Mul(8) -> output
        // With 4 lanes, each group of 8 should be split into 2 atoms per lane.
        let num_lanes = 4;
        let count = 8u64;

        let mut graph = NanoGraph::new();

        let lit_id = graph.push_group(
            count,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );

        let add_id = graph.push_group(
            count,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::affine(lit_id, 1),
                InputRef::affine(lit_id, 1),
            ],
        );

        let mul_id = graph.push_group(
            count,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::affine(add_id, 1),
                InputRef::affine(add_id, 1),
            ],
        );

        graph.outputs = vec![mul_id];

        let input_tensors = graph.input_tensors().to_vec();
        let output_ids: Vec<AtomId> = graph.outputs.clone();

        let phases = plan(&graph, num_lanes, &input_tensors, &output_ids);

        assert_lane_count(&phases, num_lanes);

        // The chain Literal -> Add -> Mul should be in ONE phase (no barriers
        // needed — Split reads Split within each lane, Literal is duplicated).
        assert_eq!(
            phases.len(),
            1,
            "Linear chain should be in a single phase, got {}",
            phases.len()
        );

        // Each lane should have compute work (Add + Mul split, Literal duplicated).
        let compute = lane_compute_atoms(&phases[0]);
        for (lane, &atoms) in compute.iter().enumerate() {
            assert!(
                atoms > 0,
                "Lane {} has 0 compute atoms — group was not split!",
                lane
            );
        }

        // Each lane should have count/num_lanes = 2 atoms for Add and 2 for Mul.
        let expected_per_lane = count / num_lanes as u64;
        for (lane, &atoms) in compute.iter().enumerate() {
            assert_eq!(
                atoms,
                2 * expected_per_lane,
                "Lane {} has {} compute atoms, expected {} (2 groups * {} atoms each)",
                lane,
                atoms,
                2 * expected_per_lane,
                expected_per_lane,
            );
        }
    }

    // ─── Test: larger chain demonstrating real splitting ────────────────────

    #[test]
    fn test_large_chain_all_lanes_active() {
        // Sub(49152) -> Pow(49152) -> Div(49152) with 8 lanes.
        // Every group should be split, all lanes active, single phase.
        let num_lanes = 8;
        let count = 49152u64;

        let mut graph = NanoGraph::new();

        let lit = graph.push_group(
            count,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
        );

        let sub_id = graph.push_group(
            count,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Sub,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::affine(lit, 1),
                InputRef::affine(lit, 1),
            ],
        );

        let pow_id = graph.push_group(
            count,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Pow,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::affine(sub_id, 1),
                InputRef::Broadcast(lit),
            ],
        );

        let div_id = graph.push_group(
            count,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Div,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::affine(pow_id, 1),
                InputRef::Broadcast(lit),
            ],
        );

        graph.outputs = vec![div_id];

        let input_tensors = graph.input_tensors().to_vec();
        let output_ids = graph.outputs.clone();

        let phases = plan(&graph, num_lanes, &input_tensors, &output_ids);

        assert_lane_count(&phases, num_lanes);

        // Single phase: the entire chain is split identically.
        assert_eq!(phases.len(), 1, "Chain should be single phase");

        // Every lane should have work.
        let compute = lane_compute_atoms(&phases[0]);
        for (lane, &atoms) in compute.iter().enumerate() {
            assert!(atoms > 0, "Lane {} has no compute atoms — not split!", lane);
        }

        // Each lane should have roughly 3 * (49152/8) = 18432 compute atoms.
        let expected = 3 * (count / num_lanes as u64);
        for (lane, &atoms) in compute.iter().enumerate() {
            // Allow for rounding on last lane.
            assert!(
                atoms >= expected - 3 && atoms <= expected + 3,
                "Lane {} has {} compute atoms, expected ~{}",
                lane,
                atoms,
                expected,
            );
        }
    }

    // ─── Test: diamond graph ────────────────────────────────────────────────

    #[test]
    fn test_diamond_graph_split() {
        // Diamond: A -> B, A -> C, B+C -> D
        // All elementwise, all should be split, single phase.
        let num_lanes = 4;
        let count = 16u64;

        let mut graph = NanoGraph::new();

        let a = graph.push_group(
            count,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );

        let b = graph.push_group(
            count,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::affine(a, 1)],
        );

        let c = graph.push_group(
            count,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::affine(a, 1)],
        );

        let d = graph.push_group(
            count,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::affine(b, 1),
                InputRef::affine(c, 1),
            ],
        );

        graph.outputs = vec![d];

        let input_tensors = graph.input_tensors().to_vec();
        let output_ids = graph.outputs.clone();

        let phases = plan(&graph, num_lanes, &input_tensors, &output_ids);

        assert_lane_count(&phases, num_lanes);

        // Single phase: all groups split the same way.
        assert_eq!(phases.len(), 1, "Diamond should be single phase");

        // All lanes active.
        let compute = lane_compute_atoms(&phases[0]);
        for (lane, &atoms) in compute.iter().enumerate() {
            assert!(atoms > 0, "Lane {} has no compute atoms in diamond", lane);
        }

        // Each lane: 3 groups * 4 atoms = 12 compute atoms.
        let expected = 3 * (count / num_lanes as u64);
        for (lane, &atoms) in compute.iter().enumerate() {
            assert_eq!(
                atoms, expected,
                "Lane {} has {} atoms, expected {}",
                lane, atoms, expected
            );
        }
    }

    // ─── Test: matmul-like structure ────────────────────────────────────────

    #[test]
    fn test_matmul_split() {
        // Simplified matmul: M=8 output rows, K=4 reduction.
        // Structure:
        //   weights: Literal(M*K = 32)
        //   input: Literal(K = 4)
        //   mul: Binary::Mul(M*K=32, inputs: weight[Affine], input[StridedBroadcast repeat=K])
        //   reduce: Reduce(M=8, reduce_count=K, reduce_stride=1, input: Affine(mul))
        //
        // With 4 lanes:
        //   - weights/input: duplicated (literal)
        //   - mul: split 32/4 = 8 per lane
        //   - reduce: split 8/4 = 2 per lane
        let num_lanes = 4;
        let m = 8u64;
        let k = 4u64;

        let mut graph = NanoGraph::new();

        // Weights: M*K literal.
        let weights = graph.push_group(
            m * k,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.5)),
            vec![],
            vec![],
        );

        // Input vector: K literal.
        let input_vec = graph.push_group(
            k,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );

        // Mul: M*K atoms, weight[i] * input[i % K].
        let mul = graph.push_group(
            m * k,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::affine(weights, 1),
                InputRef::modular(input_vec, 1, k),
            ],
        );

        // ReduceSum: M atoms, each summing K consecutive mul atoms.
        let reduce = graph.push_group(
            m,
            DType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: k,
                reduce_stride: 1,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::affine(mul, k as i64)],
        );

        graph.outputs = vec![reduce];

        let input_tensors = graph.input_tensors().to_vec();
        let output_ids = graph.outputs.clone();

        let phases = plan(&graph, num_lanes, &input_tensors, &output_ids);

        assert_lane_count(&phases, num_lanes);

        // The mul and reduce should be split across lanes.
        // Check that compute atoms are distributed.
        let mut total_compute = 0u64;
        let mut lanes_with_work = 0;
        for phase in &phases {
            let compute = lane_compute_atoms(phase);
            for &atoms in &compute {
                if atoms > 0 {
                    lanes_with_work += 1;
                    total_compute += atoms;
                }
            }
        }

        // All lanes should have work (mul and reduce are both split).
        assert!(
            lanes_with_work >= num_lanes,
            "Only {} lanes have work, expected all {} to have work (matmul not split)",
            lanes_with_work,
            num_lanes,
        );

        // Total compute: mul(32) + reduce(8) = 40 atoms spread across lanes.
        assert_eq!(
            total_compute,
            m * k + m,
            "Total compute atoms: got {}, expected {} (mul) + {} (reduce)",
            total_compute,
            m * k,
            m,
        );
    }

    // ─── Test: literal duplication ──────────────────────────────────────────

    #[test]
    fn test_literal_duplicated_to_all_lanes() {
        // Literal(8) -> Mul(8, Literal * Literal)
        // With 4 lanes, the literal should be duplicated in all 4 spans
        // (not split).
        let num_lanes = 4;
        let count = 8u64;

        let mut graph = NanoGraph::new();

        let lit = graph.push_group(
            count,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(3.0)),
            vec![],
            vec![],
        );

        let mul = graph.push_group(
            count,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::affine(lit, 1),
                InputRef::affine(lit, 1),
            ],
        );

        graph.outputs = vec![mul];

        let input_tensors = graph.input_tensors().to_vec();
        let output_ids = graph.outputs.clone();

        let phases = plan(&graph, num_lanes, &input_tensors, &output_ids);

        assert_lane_count(&phases, num_lanes);
        assert_eq!(phases.len(), 1, "Should be single phase");

        // Every lane should have the literal group (duplicated, full count=8).
        for (lane, span) in phases[0].spans.iter().enumerate() {
            let lit_groups: Vec<&AtomGroup> = span
                .graph
                .groups()
                .iter()
                .filter(|g| matches!(g.op, ScalarOp::Literal(_)))
                .collect();

            assert!(
                !lit_groups.is_empty(),
                "Lane {} has no literal group — literal not duplicated!",
                lane
            );

            // The literal should be the FULL group (count=8, not split).
            let total_lit_atoms: u64 = lit_groups.iter().map(|g| g.count).sum();
            assert_eq!(
                total_lit_atoms, count,
                "Lane {} has {} literal atoms, expected {} (full duplicate)",
                lane, total_lit_atoms, count
            );
        }

        // Each lane should also have count/num_lanes = 2 compute atoms for Mul.
        let compute = lane_compute_atoms(&phases[0]);
        let expected = count / num_lanes as u64;
        for (lane, &atoms) in compute.iter().enumerate() {
            assert_eq!(
                atoms, expected,
                "Lane {} has {} compute atoms, expected {}",
                lane, atoms, expected
            );
        }
    }

    // ─── Test: reduce causes barrier when reading split data ────────────────

    #[test]
    fn test_reduce_barrier_when_reading_all_lanes() {
        // split_source(32) -> small_reduce(1, reduce_count=32) -> output
        // The reduce has count=1 which is < num_lanes, so it will be Duplicated.
        // Since a Duplicate group reads from a Split group, a barrier is needed.
        let num_lanes = 4;

        let mut graph = NanoGraph::new();

        let lit = graph.push_group(
            1,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );

        let source = graph.push_group(
            32,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::affine(lit, 0),
                InputRef::affine(lit, 0),
            ],
        );

        // Reduce all 32 atoms to 1 output.
        let reduce = graph.push_group(
            1,
            DType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: 32,
                reduce_stride: 1,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::affine(source, 1)],
        );

        graph.outputs = vec![reduce];

        let input_tensors = graph.input_tensors().to_vec();
        let output_ids = graph.outputs.clone();

        let phases = plan(&graph, num_lanes, &input_tensors, &output_ids);

        assert_lane_count(&phases, num_lanes);

        // The source is Split (count=32 >= 4 lanes), but the reduce is
        // Duplicate (count=1 < 4). Duplicate reading Split requires a barrier.
        assert!(
            phases.len() >= 2,
            "Expected at least 2 phases (barrier between split source and reduce fan-in), got {}",
            phases.len()
        );

        // Phase 0: source should be split across all lanes.
        let p0_compute = lane_compute_atoms(&phases[0]);
        let active_lanes = p0_compute.iter().filter(|&&a| a > 0).count();
        assert!(
            active_lanes >= 2,
            "Phase 0 should have multiple active lanes (source split), got {}",
            active_lanes
        );
    }

    // ─── Test: single lane edge case ────────────────────────────────────────

    #[test]
    fn test_single_lane_no_split() {
        // With num_lanes=1, nothing should be split. Single lane, single phase.
        let num_lanes = 1;
        let count = 100u64;

        let mut graph = NanoGraph::new();

        let lit = graph.push_group(
            count,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );

        let add = graph.push_group(
            count,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::affine(lit, 1),
                InputRef::affine(lit, 1),
            ],
        );

        graph.outputs = vec![add];

        let input_tensors = graph.input_tensors().to_vec();
        let output_ids = graph.outputs.clone();

        let phases = plan(&graph, num_lanes, &input_tensors, &output_ids);

        assert_eq!(phases.len(), 1);
        assert_eq!(phases[0].spans.len(), 1);

        // The single lane should have all 100 compute atoms.
        let compute = lane_compute_atoms(&phases[0]);
        assert_eq!(compute[0], count);
    }

    // ─── Test: empty graph ──────────────────────────────────────────────────

    #[test]
    fn test_empty_graph() {
        let graph = NanoGraph::new();
        let phases = plan(&graph, 4, &[], &[]);
        assert_eq!(phases.len(), 1);
        assert_eq!(phases[0].spans.len(), 4);
        for span in &phases[0].spans {
            assert_eq!(span.graph.num_groups(), 0);
        }
    }

    // ─── Test: span NanoGraph validates ─────────────────────────────────────

    #[test]
    fn test_span_graphs_validate() {
        // Build a non-trivial graph and check that all span NanoGraphs pass
        // validation (correct topological order, all InputRefs resolve).
        let num_lanes = 4;
        let count = 16u64;

        let mut graph = NanoGraph::new();

        let lit = graph.push_group(
            count,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
        );

        let add = graph.push_group(
            count,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::affine(lit, 1),
                InputRef::affine(lit, 1),
            ],
        );

        let mul = graph.push_group(
            count,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::affine(add, 1),
                InputRef::Broadcast(lit),
            ],
        );

        graph.outputs = vec![mul];

        let input_tensors = graph.input_tensors().to_vec();
        let output_ids = graph.outputs.clone();

        let phases = plan(&graph, num_lanes, &input_tensors, &output_ids);

        for (pi, phase) in phases.iter().enumerate() {
            for (li, span) in phase.spans.iter().enumerate() {
                let errors = span.graph.validate();
                assert!(
                    errors.is_empty(),
                    "Phase {} Lane {} validation errors: {:?}",
                    pi,
                    li,
                    errors
                );
            }
        }
    }

    // ─── Test: all output atoms are produced ────────────────────────────────

    #[test]
    fn test_all_output_atoms_produced() {
        let num_lanes = 4;
        let count = 16u64;

        let mut graph = NanoGraph::new();

        let lit = graph.push_group(
            count,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );

        let add = graph.push_group(
            count,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::affine(lit, 1),
                InputRef::affine(lit, 1),
            ],
        );

        graph.outputs = vec![add];

        let input_tensors = graph.input_tensors().to_vec();
        let output_ids = graph.outputs.clone();

        let phases = plan(&graph, num_lanes, &input_tensors, &output_ids);

        // Collect all output ranges from all spans in all phases.
        let mut produced: HashSet<u64> = HashSet::new();
        for phase in &phases {
            for span in &phase.spans {
                for out in &span.outputs {
                    for i in 0..out.count {
                        produced.insert(out.base.0 + i);
                    }
                }
            }
        }

        // All output atom IDs should be produced.
        for &out_id in &output_ids {
            assert!(
                produced.contains(&out_id.0),
                "Output atom {} not produced by any span",
                out_id,
            );
        }
    }

    // ─── Test: verify split atom_offset values ──────────────────────────────

    #[test]
    fn test_split_groups_have_correct_atom_offset() {
        // Build: Literal(32) -> Add(32), split across 4 lanes.
        // Lane 0 should have Add atoms [0..8) with atom_offset=0.
        // Lane 1 should have Add atoms [8..16) with atom_offset=8.
        // etc.
        let num_lanes = 4;
        let count = 32u64;

        let mut graph = NanoGraph::new();

        let lit = graph.push_group(
            count,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );

        let add = graph.push_group(
            count,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::affine(lit, 1),
                InputRef::affine(lit, 1),
            ],
        );

        graph.outputs = vec![add];

        let input_tensors = graph.input_tensors().to_vec();
        let output_ids = graph.outputs.clone();

        let phases = plan(&graph, num_lanes, &input_tensors, &output_ids);

        assert_eq!(phases.len(), 1);

        let chunk = count / num_lanes as u64; // 8
        for (lane, span) in phases[0].spans.iter().enumerate() {
            // Find the Add group fragment in this span.
            let add_groups: Vec<&AtomGroup> = span
                .graph
                .groups()
                .iter()
                .filter(|g| {
                    matches!(
                        g.op,
                        ScalarOp::Binary {
                            op: ScalarBinOp::Add,
                            ..
                        }
                    )
                })
                .collect();

            assert_eq!(
                add_groups.len(),
                1,
                "Lane {} should have exactly 1 Add fragment, got {}",
                lane,
                add_groups.len()
            );

            let frag = add_groups[0];
            let expected_offset = lane as u64 * chunk;
            let expected_base = add.0 + expected_offset;

            assert_eq!(
                frag.base_id.0, expected_base,
                "Lane {} Add fragment base_id: got {}, expected {}",
                lane, frag.base_id.0, expected_base
            );
            assert_eq!(
                frag.atom_offset, expected_offset,
                "Lane {} Add fragment atom_offset: got {}, expected {}",
                lane, frag.atom_offset, expected_offset
            );
            assert_eq!(
                frag.count, chunk,
                "Lane {} Add fragment count: got {}, expected {}",
                lane, frag.count, chunk
            );
        }
    }

    // ─── Test: IndirectLoad groups are split ────────────────────────────────

    #[test]
    fn test_indirect_load_split() {
        // Table: Literal(256) -> IndirectLoad(64) with index input
        // Each IndirectLoad atom reads one table entry.
        // With 4 lanes, the 64 IndirectLoad atoms should be split 16 per lane.
        let num_lanes = 4;
        let table_count = 256u64;
        let load_count = 64u64;

        let mut graph = NanoGraph::new();

        let table = graph.push_group(
            table_count,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
        );

        // Index input: literal indices (0, 1, 2, ... 63).
        let index = graph.push_group(
            load_count,
            DType::I64,
            ScalarOp::Literal(NumericScalar::I64(0)),
            vec![],
            vec![],
        );

        let load = graph.push_group(
            load_count,
            DType::F32,
            ScalarOp::IndirectLoad { table_base: table },
            vec![],
            vec![InputRef::affine(index, 1)],
        );

        graph.outputs = vec![load];

        let input_tensors = graph.input_tensors().to_vec();
        let output_ids = graph.outputs.clone();

        let phases = plan(&graph, num_lanes, &input_tensors, &output_ids);

        assert_lane_count(&phases, num_lanes);

        // IndirectLoad should be split.
        let compute = lane_compute_atoms(&phases[0]);
        for (lane, &atoms) in compute.iter().enumerate() {
            assert!(
                atoms > 0,
                "Lane {} has no IndirectLoad atoms — not split!",
                lane
            );
        }

        let expected = load_count / num_lanes as u64;
        for (lane, &atoms) in compute.iter().enumerate() {
            assert_eq!(
                atoms, expected,
                "Lane {} has {} IndirectLoad atoms, expected {}",
                lane, atoms, expected
            );
        }
    }

    // ─── Test: verify groups ARE split (not just assigned whole) ────────────

    #[test]
    fn test_groups_actually_split_not_whole() {
        // The most important test: verify that a 1000-atom group with 4 lanes
        // results in 4 fragments of ~250 atoms each, NOT 1 fragment of 1000
        // atoms on one lane.
        let num_lanes = 4;
        let count = 1000u64;

        let mut graph = NanoGraph::new();

        let lit = graph.push_group(
            count,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );

        let unary = graph.push_group(
            count,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::affine(lit, 1)],
        );

        graph.outputs = vec![unary];

        let input_tensors = graph.input_tensors().to_vec();
        let output_ids = graph.outputs.clone();

        let phases = plan(&graph, num_lanes, &input_tensors, &output_ids);

        // Check that NO lane has the full 1000 atoms of the Exp group.
        for (pi, phase) in phases.iter().enumerate() {
            for (li, span) in phase.spans.iter().enumerate() {
                for g in span.graph.groups() {
                    if matches!(
                        g.op,
                        ScalarOp::Unary {
                            op: ScalarUnaryOp::Exp,
                            ..
                        }
                    ) {
                        assert!(
                            g.count < count,
                            "Phase {} Lane {} has a full Exp group ({} atoms) — NOT SPLIT!",
                            pi,
                            li,
                            g.count
                        );
                        assert_eq!(
                            g.count,
                            count / num_lanes as u64,
                            "Phase {} Lane {} Exp fragment has {} atoms, expected {}",
                            pi,
                            li,
                            g.count,
                            count / num_lanes as u64
                        );
                    }
                }
            }
        }
    }
}
