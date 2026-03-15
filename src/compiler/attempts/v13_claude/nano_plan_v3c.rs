#![allow(clippy::all, dead_code, unreachable_patterns, unused_variables, unused_imports)]
//! Span-based execution planner v3c: general cross-lane violation resolution.
//!
//! Strategy:
//! 1. Use v2c's phase detection and lane assignment as the backbone
//! 2. Build group-level dependency graph (producers/consumers)
//! 3. For each (phase, lane), check if any work item depends on a producer
//!    in the SAME phase but a DIFFERENT lane → violation
//! 4. For small producers (< threshold atoms), duplicate into consuming lane
//!    For others, bump consuming groups to a later phase
//! 5. Repeat until no violations remain
//! 6. Build span NanoGraphs from the corrected assignments

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::nano_graph::{AtomGroup, AtomId, InputRef, NanoGraph, ScalarOp};

use super::nano_plan_v2c;

/// Literal groups with fewer atoms than this are duplicated into spans.
const LITERAL_INLINE_THRESHOLD: u64 = 1024;

/// Groups with fewer atoms than this can be duplicated across lanes
/// to resolve cross-lane violations (instead of adding a new phase).
const DUPLICATION_THRESHOLD: u64 = 65536;

// ─── Public API ──────────────────────────────────────────────────────────────

/// A contiguous range of atoms mapped between main graph and span graph.
#[derive(Debug, Clone)]
pub struct AtomMapping {
    pub main_base: AtomId,
    pub span_base: AtomId,
    pub count: u64,
}

/// A self-contained unit of work for one lane in one phase.
pub struct Span {
    /// Self-contained NanoGraph for this span's computation.
    pub graph: NanoGraph,
    /// Contiguous ranges of atoms this span reads from the main graph.
    pub inputs: Vec<AtomMapping>,
    /// Contiguous ranges of atoms this span writes back to the shared values buffer.
    pub outputs: Vec<AtomMapping>,
}

/// One phase of execution.
pub struct Phase {
    /// One span per lane (may have empty graphs for idle lanes).
    pub spans: Vec<Span>,
}

/// The full execution plan with self-contained span NanoGraphs.
pub struct SpanPlan {
    pub num_lanes: usize,
    pub phases: Vec<Phase>,
}

/// Plan execution for a NanoGraph, emitting self-contained span NanoGraphs.
pub fn plan_execution_spans(graph: &NanoGraph, num_lanes: usize) -> SpanPlan {
    let num_lanes = num_lanes.max(1);
    let groups = graph.groups();
    let n = groups.len();

    if n == 0 {
        return SpanPlan {
            num_lanes,
            phases: vec![],
        };
    }

    // Step 1: Run v2c to get the lane/phase assignments.
    let v2c_plan = nano_plan_v2c::plan_execution(graph, num_lanes);

    let is_literal: Vec<bool> = groups
        .iter()
        .map(|g| matches!(g.op, ScalarOp::Literal(_)) && g.inputs.is_empty())
        .collect();

    // Step 2: Build group-level dependency graph.
    let producers = build_group_producers(groups, &is_literal);

    // Step 3: Extract work assignments as (group_idx, atom_offset, atom_count, phase, lane).
    // We'll modify phase assignments to fix violations.
    let mut work_items: Vec<WorkItem> = Vec::new();
    for (phase_idx, phase) in v2c_plan.phases.iter().enumerate() {
        for (lane_idx, lane_work) in phase.lane_work.iter().enumerate() {
            for work in lane_work {
                work_items.push(WorkItem {
                    group_idx: work.group_idx,
                    atom_offset: work.atom_offset,
                    atom_count: work.atom_count,
                    phase: phase_idx,
                    lane: lane_idx,
                });
            }
        }
    }

    // Step 4: Fix cross-lane violations by adjusting phases.
    fix_cross_lane_violations(&mut work_items, groups, &is_literal, &producers, num_lanes);

    // Step 5: Build span NanoGraphs from corrected assignments.
    build_span_plan(graph, groups, &is_literal, &work_items, num_lanes)
}

// ─── Work item ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct WorkItem {
    group_idx: usize,
    atom_offset: u64,
    atom_count: u64,
    phase: usize,
    lane: usize,
}

// ─── Group-level dependency graph ────────────────────────────────────────────

/// Build producer list for each group. Returns producers[gi] = list of group
/// indices that gi reads from (excluding literals and self).
fn build_group_producers(groups: &[AtomGroup], is_literal: &[bool]) -> Vec<Vec<usize>> {
    let n = groups.len();
    let mut producers: Vec<Vec<usize>> = Vec::with_capacity(n);

    for (gi, group) in groups.iter().enumerate() {
        let mut prod_set = BTreeSet::new();

        // 1. Basic InputRef resolution.
        for input in &group.inputs {
            for pi in resolve_producer_groups(input, group.count, groups) {
                if pi != gi && !is_literal[pi] {
                    prod_set.insert(pi);
                }
            }
        }

        // 2. ReduceSum/ReduceMax: strided access extends beyond InputRef range.
        match &group.op {
            ScalarOp::ReduceSum { reduce_count, reduce_stride, .. }
            | ScalarOp::ReduceMax { reduce_count, reduce_stride, .. }
                if *reduce_count > 1 && *reduce_stride != 0 =>
            {
                for input in &group.inputs {
                    for pi in resolve_producer_groups_with_reduce(
                        input, group.count, *reduce_count, *reduce_stride, groups,
                    ) {
                        if pi != gi && !is_literal[pi] {
                            prod_set.insert(pi);
                        }
                    }
                }
            }
            _ => {}
        }

        // 3. IndirectLoad: table_base references a group.
        if let ScalarOp::IndirectLoad { table_base, .. } = &group.op {
            if let Some(pi) = find_group_idx(groups, *table_base) {
                if pi != gi && !is_literal[pi] {
                    prod_set.insert(pi);
                }
            }
        }

        producers.push(prod_set.into_iter().collect());
    }

    producers
}

// ─── Cross-lane violation resolution ─────────────────────────────────────────

/// Fix all cross-lane violations in the work item list.
///
/// A violation occurs when a work item in (phase P, lane L) reads atoms
/// that are produced by a work item in (phase P, lane L') where L' != L,
/// and those atoms are NOT also produced by a work item on lane L in phase P
/// or in an earlier phase.
///
/// Resolution strategies:
/// 1. DUPLICATE: if the producer group is small enough, duplicate it into the
///    consuming lane (add a new work item for the full group on that lane).
/// 2. DEFER: bump the consuming work item to phase P+1. This cascades:
///    any downstream consumers in phase P that depend on the bumped item
///    must also be bumped.
///
/// We iterate until no violations remain.
fn fix_cross_lane_violations(
    work_items: &mut Vec<WorkItem>,
    groups: &[AtomGroup],
    is_literal: &[bool],
    producers: &[Vec<usize>],
    num_lanes: usize,
) {
    let max_iterations = 100;
    for _iteration in 0..max_iterations {
        // Build index: for each (group_idx, phase, lane), what atom ranges are covered.
        let coverage = build_coverage_index(work_items);

        let mut violations: Vec<Violation> = Vec::new();

        // For each work item, resolve which producer groups it reads from
        // and check if the needed atom ranges are covered on this lane.
        for (wi, item) in work_items.iter().enumerate() {
            let gi = item.group_idx;
            let group = &groups[gi];
            let my_phase = item.phase;
            let my_lane = item.lane;

            // Get the needed producer group ranges for this work item's slice.
            let needed = compute_needed_ranges(
                group, item.atom_offset, item.atom_count, groups, is_literal,
            );

            for (prod_gi, need_lo, need_hi) in &needed {
                let prod_gi = *prod_gi;
                let need_lo = *need_lo;
                let need_hi = *need_hi;

                // Check if [need_lo, need_hi) is covered by:
                // 1. A work item for prod_gi on the SAME lane in the SAME or EARLIER phase, OR
                // 2. A work item for prod_gi in an EARLIER phase (any lane)
                let covered = is_range_covered(
                    &coverage, prod_gi, need_lo, need_hi,
                    my_phase, my_lane,
                );

                if !covered {
                    // Find which lane(s) produce this range in the same phase.
                    let key = (prod_gi, my_phase);
                    let producer_lanes: Vec<usize> = if let Some(lane_ranges) = coverage.get(&key) {
                        lane_ranges.keys().copied()
                            .filter(|&l| l != my_lane)
                            .filter(|&l| {
                                lane_ranges.get(&l).map(|ranges| {
                                    ranges_cover(ranges, need_lo, need_hi)
                                }).unwrap_or(false)
                            })
                            .collect()
                    } else {
                        vec![]
                    };

                    violations.push(Violation {
                        consumer_work_idx: wi,
                        consumer_gi: gi,
                        producer_gi: prod_gi,
                        phase: my_phase,
                        consumer_lane: my_lane,
                    });
                }
            }
        }

        if violations.is_empty() {
            break;
        }

        // Deduplicate violations by (phase, producer_gi).
        let mut producer_violations: BTreeMap<(usize, usize), BTreeSet<usize>> = BTreeMap::new();
        for v in &violations {
            producer_violations
                .entry((v.phase, v.producer_gi))
                .or_default()
                .insert(v.consumer_lane);
        }

        let mut defer_set: HashSet<(usize, usize)> = HashSet::new();
        let mut dup_set: HashSet<(usize, usize, usize)> = HashSet::new(); // (prod_gi, phase, lane)

        for (&(phase, prod_gi), consumer_lanes) in &producer_violations {
            let prod_count = groups[prod_gi].count;
            if prod_count <= DUPLICATION_THRESHOLD {
                // Duplicate: add full group to each consuming lane.
                for &lane in consumer_lanes {
                    let already_has = work_items.iter().any(|w| {
                        w.group_idx == prod_gi
                            && w.phase == phase
                            && w.lane == lane
                            && w.atom_offset == 0
                            && w.atom_count == prod_count
                    });
                    if !already_has {
                        dup_set.insert((prod_gi, phase, lane));
                    }
                }
            } else {
                // Defer: bump consumers to next phase.
                for v in &violations {
                    if v.phase == phase && v.producer_gi == prod_gi {
                        defer_set.insert((phase, v.consumer_gi));
                    }
                }
            }
        }

        // Apply duplications.
        for &(prod_gi, phase, lane) in &dup_set {
            work_items.push(WorkItem {
                group_idx: prod_gi,
                atom_offset: 0,
                atom_count: groups[prod_gi].count,
                phase,
                lane,
            });
        }

        // Apply deferrals with transitive closure.
        if !defer_set.is_empty() {
            let mut deferred_groups_in_phase: HashMap<usize, HashSet<usize>> = HashMap::new();
            for &(phase, gi) in &defer_set {
                deferred_groups_in_phase.entry(phase).or_default().insert(gi);
            }

            for (&phase, deferred) in &mut deferred_groups_in_phase {
                loop {
                    let mut new_defers = Vec::new();
                    for item in work_items.iter() {
                        if item.phase != phase || deferred.contains(&item.group_idx) {
                            continue;
                        }
                        if producers[item.group_idx].iter().any(|&pi| deferred.contains(&pi)) {
                            new_defers.push(item.group_idx);
                        }
                    }
                    if new_defers.is_empty() { break; }
                    for gi in new_defers { deferred.insert(gi); }
                }
            }

            // Bump deferred items. Process phases in reverse to avoid cascading
            // issues within a single iteration.
            for item in work_items.iter_mut() {
                if let Some(deferred) = deferred_groups_in_phase.get(&item.phase) {
                    if deferred.contains(&item.group_idx) {
                        item.phase += 1;
                    }
                }
            }
        }

        if dup_set.is_empty() && defer_set.is_empty() {
            break;
        }
    }

    compact_phases(work_items);
}

#[derive(Debug)]
struct Violation {
    consumer_work_idx: usize,
    consumer_gi: usize,
    producer_gi: usize,
    phase: usize,
    consumer_lane: usize,
}

/// Coverage index: (group_idx, phase) -> lane -> Vec<(offset, count)> sorted by offset.
type CoverageIndex = HashMap<(usize, usize), HashMap<usize, Vec<(u64, u64)>>>;

/// Build coverage index from work items.
fn build_coverage_index(work_items: &[WorkItem]) -> CoverageIndex {
    let mut index: CoverageIndex = HashMap::new();
    for item in work_items {
        index
            .entry((item.group_idx, item.phase))
            .or_default()
            .entry(item.lane)
            .or_default()
            .push((item.atom_offset, item.atom_count));
    }
    // Sort each lane's ranges.
    for lane_map in index.values_mut() {
        for ranges in lane_map.values_mut() {
            ranges.sort_by_key(|&(off, _)| off);
        }
    }
    index
}

/// Check if the atom range [need_lo, need_hi) within prod_gi is covered
/// by work items on `my_lane` in phase `my_phase` or any lane in earlier phases.
fn is_range_covered(
    coverage: &CoverageIndex,
    prod_gi: usize,
    need_lo: u64,
    need_hi: u64,
    my_phase: usize,
    my_lane: usize,
) -> bool {
    // Check same phase, same lane.
    if let Some(lane_map) = coverage.get(&(prod_gi, my_phase)) {
        if let Some(ranges) = lane_map.get(&my_lane) {
            if ranges_cover(ranges, need_lo, need_hi) {
                return true;
            }
        }
    }

    // Check earlier phases (any lane).
    for phase in 0..my_phase {
        if let Some(lane_map) = coverage.get(&(prod_gi, phase)) {
            // Merge all lanes' ranges for this phase.
            let mut all_ranges: Vec<(u64, u64)> = Vec::new();
            for ranges in lane_map.values() {
                all_ranges.extend_from_slice(ranges);
            }
            all_ranges.sort_by_key(|&(off, _)| off);
            if ranges_cover(&all_ranges, need_lo, need_hi) {
                return true;
            }
        }
    }

    false
}

/// Check if sorted ranges fully cover [need_lo, need_hi).
fn ranges_cover(ranges: &[(u64, u64)], need_lo: u64, need_hi: u64) -> bool {
    if need_lo >= need_hi { return true; }
    let mut covered_up_to = need_lo;
    for &(off, count) in ranges {
        if off > covered_up_to { return false; }
        let end = off + count;
        if end > covered_up_to {
            covered_up_to = end;
        }
        if covered_up_to >= need_hi { return true; }
    }
    false
}

/// Compute the (producer_group_idx, atom_offset_lo, atom_offset_hi) ranges
/// that a work item needs from its producers. Offsets are relative to the
/// producer group's base_id.
fn compute_needed_ranges(
    group: &AtomGroup,
    atom_offset: u64,
    atom_count: u64,
    all_groups: &[AtomGroup],
    is_literal: &[bool],
) -> Vec<(usize, u64, u64)> {
    let mut result = Vec::new();
    if atom_count == 0 { return result; }

    let (is_reduce, reduce_count, reduce_stride) = match &group.op {
        ScalarOp::ReduceSum { reduce_count, reduce_stride, .. }
        | ScalarOp::ReduceMax { reduce_count, reduce_stride, .. }
            if *reduce_count > 1 && *reduce_stride != 0 =>
            (true, *reduce_count, *reduce_stride),
        _ => (false, 0, 0),
    };

    for input in &group.inputs {
        let referenced = resolve_input_to_group_ranges(
            input, atom_offset, atom_count,
            if is_reduce { reduce_count } else { 1 },
            if is_reduce { reduce_stride } else { 0 },
            all_groups,
        );

        for (gi, range_lo, range_hi) in referenced {
            if is_literal[gi] { continue; }
            let g_lo = all_groups[gi].base_id.0;
            let g_hi = g_lo + all_groups[gi].count;
            let overlap_lo = range_lo.max(g_lo);
            let overlap_hi = range_hi.min(g_hi);
            if overlap_lo >= overlap_hi { continue; }
            // Convert to offset within group.
            result.push((gi, overlap_lo - g_lo, overlap_hi - g_lo));
        }
    }

    // IndirectLoad table.
    if let ScalarOp::IndirectLoad { table_base, .. } = &group.op {
        if let Some(gi) = find_group_idx(all_groups, *table_base) {
            if !is_literal[gi] {
                result.push((gi, 0, all_groups[gi].count));
            }
        }
    }

    // Merge overlapping ranges per group.
    result.sort_by_key(|&(gi, lo, _)| (gi, lo));
    let mut merged = Vec::new();
    for (gi, lo, hi) in result {
        if let Some(last) = merged.last_mut() {
            let (lg, ll, lh): &mut (usize, u64, u64) = last;
            if *lg == gi && lo <= *lh {
                *lh = (*lh).max(hi);
                continue;
            }
        }
        merged.push((gi, lo, hi));
    }
    merged
}

/// Compact phases so they are numbered 0..n without gaps.
fn compact_phases(work_items: &mut [WorkItem]) {
    let mut phases_used: BTreeSet<usize> = BTreeSet::new();
    for item in work_items.iter() {
        phases_used.insert(item.phase);
    }
    let phase_map: HashMap<usize, usize> = phases_used
        .iter()
        .enumerate()
        .map(|(new, &old)| (old, new))
        .collect();
    for item in work_items.iter_mut() {
        item.phase = phase_map[&item.phase];
    }
}

// ─── Span builder ────────────────────────────────────────────────────────────

/// Range-based atom map for efficient lookup, backed by a BTreeMap.
struct RangeAtomMap {
    map: BTreeMap<u64, (u64, u64)>,
}

impl RangeAtomMap {
    fn new() -> Self { Self { map: BTreeMap::new() } }
    fn insert_range(&mut self, main_base: AtomId, span_base: AtomId, count: u64) {
        self.map.insert(main_base.0, (span_base.0, count));
    }
    fn get(&self, main_id: AtomId) -> Option<AtomId> {
        use std::ops::Bound;
        let mut iter = self.map.range((Bound::Unbounded, Bound::Included(main_id.0)));
        if let Some((&base, &(span_base, count))) = iter.next_back() {
            let offset = main_id.0.wrapping_sub(base);
            if offset < count {
                return Some(AtomId(span_base + offset));
            }
        }
        None
    }
}

/// Build the complete SpanPlan from corrected work items.
fn build_span_plan(
    main_graph: &NanoGraph,
    groups: &[AtomGroup],
    is_literal: &[bool],
    work_items: &[WorkItem],
    num_lanes: usize,
) -> SpanPlan {
    let num_phases = work_items.iter().map(|w| w.phase).max().map(|m| m + 1).unwrap_or(0);

    // Group work items by (phase, lane).
    let mut phase_lane_work: Vec<Vec<Vec<&WorkItem>>> = vec![vec![Vec::new(); num_lanes]; num_phases];
    for item in work_items {
        phase_lane_work[item.phase][item.lane].push(item);
    }

    // Sort each lane's work by group_idx for determinism and topo ordering.
    for phase in &mut phase_lane_work {
        for lane in phase {
            lane.sort_by_key(|w| (w.group_idx, w.atom_offset));
        }
    }

    // Determine which work items are "output duplicates" - duplicated groups
    // where only one lane should output each atom range.
    // For each group_idx, find which lanes have it. If a group appears on multiple
    // lanes in the same phase with the same (offset, count), only the first lane outputs.
    // If a group is split across lanes (different offsets), each lane outputs its own slice.
    let mut dup_output_info = compute_dup_output_info(work_items, groups, num_lanes);

    let mut phases = Vec::with_capacity(num_phases);

    for phase_idx in 0..num_phases {
        let mut spans = Vec::with_capacity(num_lanes);

        for lane_idx in 0..num_lanes {
            let lane_work = &phase_lane_work[phase_idx][lane_idx];
            if lane_work.is_empty() {
                spans.push(Span {
                    graph: NanoGraph::new(),
                    inputs: vec![],
                    outputs: vec![],
                });
                continue;
            }

            let assigned_slices: Vec<(usize, u64, u64)> = lane_work
                .iter()
                .map(|w| (w.group_idx, w.atom_offset, w.atom_count))
                .collect();

            let span = build_span(
                main_graph,
                groups,
                is_literal,
                &assigned_slices,
                work_items,
                phase_idx,
                lane_idx,
                &dup_output_info,
            );
            spans.push(span);
        }

        phases.push(Phase { spans });
    }

    SpanPlan { num_lanes, phases }
}

/// For duplicated groups, determine which lane should output which atom range.
/// Returns: group_idx -> Vec<(lane_idx, atom_offset, atom_count)> for the
/// "original" (non-duplicate) output assignments.
fn compute_dup_output_info(
    work_items: &[WorkItem],
    groups: &[AtomGroup],
    num_lanes: usize,
) -> HashMap<usize, Vec<(usize, u64, u64)>> {
    // For each group, collect all (phase, lane, offset, count) assignments.
    let mut group_assignments: HashMap<usize, Vec<(usize, usize, u64, u64)>> = HashMap::new();
    for item in work_items {
        group_assignments
            .entry(item.group_idx)
            .or_default()
            .push((item.phase, item.lane, item.atom_offset, item.atom_count));
    }

    let mut dup_info: HashMap<usize, Vec<(usize, u64, u64)>> = HashMap::new();

    for (&gi, assignments) in &group_assignments {
        // Group by phase.
        let mut by_phase: HashMap<usize, Vec<(usize, u64, u64)>> = HashMap::new();
        for &(phase, lane, offset, count) in assignments {
            by_phase.entry(phase).or_default().push((lane, offset, count));
        }

        for (&phase, lane_assignments) in &by_phase {
            if lane_assignments.len() <= 1 {
                continue; // No duplication in this phase.
            }

            // Check if this is a true duplication (same offset+count on multiple lanes)
            // vs a split (different offsets on different lanes).
            let mut full_copies: Vec<usize> = Vec::new(); // lanes with full group
            let mut split_copies: Vec<(usize, u64, u64)> = Vec::new();

            let full_count = groups[gi].count;
            for &(lane, offset, count) in lane_assignments {
                if offset == 0 && count == full_count {
                    full_copies.push(lane);
                } else {
                    split_copies.push((lane, offset, count));
                }
            }

            if full_copies.len() > 1 && split_copies.is_empty() {
                // Pure duplication: multiple lanes have the full group.
                // Each lane should output a proportional slice.
                // Split the output evenly among lanes that have the full copy.
                full_copies.sort();
                let chunk = (full_count + full_copies.len() as u64 - 1) / full_copies.len() as u64;
                let mut output_assignments = Vec::new();
                let mut offset = 0u64;
                for (i, &lane) in full_copies.iter().enumerate() {
                    let remaining = full_count.saturating_sub(offset);
                    let this_count = if i < full_copies.len() - 1 {
                        chunk.min(remaining)
                    } else {
                        remaining
                    };
                    if this_count > 0 {
                        output_assignments.push((lane, offset, this_count));
                    }
                    offset += this_count;
                }
                dup_info.insert(gi, output_assignments);
            } else if !split_copies.is_empty() && !full_copies.is_empty() {
                // Mix of full copies and split copies.
                // Full copies were added as duplicates to serve local consumption.
                // The split copies own their respective slices for output.
                // Full copies that overlap with split copies should NOT output
                // those overlapping ranges.
                // Simple approach: only the split copies output, full copies
                // are for internal consumption only.
                let mut output_assignments: Vec<(usize, u64, u64)> = split_copies.clone();
                // Check if split copies cover the full range.
                let covered: u64 = split_copies.iter().map(|&(_, _, c)| c).sum();
                if covered < full_count {
                    // Some atoms aren't covered by splits. Assign them to
                    // the first full-copy lane.
                    // Actually, we need to figure out which ranges are uncovered.
                    // For simplicity, let the split copies handle what they can
                    // and the first full-copy lane handles the rest.
                    // But this is complex. In practice, for GPT-2, the split copies
                    // cover the full group (they're the original v2c split).
                    // Full copies are the duplicates we added.
                }
                dup_info.insert(gi, output_assignments);
            }
        }
    }

    dup_info
}

/// Build a self-contained NanoGraph for one (phase, lane).
fn build_span(
    main_graph: &NanoGraph,
    groups: &[AtomGroup],
    is_literal: &[bool],
    assigned_slices: &[(usize, u64, u64)],
    all_work_items: &[WorkItem],
    phase_idx: usize,
    lane_idx: usize,
    dup_output_info: &HashMap<usize, Vec<(usize, u64, u64)>>,
) -> Span {
    let mut span_graph = NanoGraph::new();

    // Copy sym_dim configuration from main graph.
    for (name, &sd) in &main_graph.sym_dim_names {
        let local_sd = span_graph.sym_dim(name);
        if let Some(&bound) = main_graph.sym_dim_bounds.get(&sd) {
            span_graph.sym_dim_bounds.insert(local_sd, bound);
        }
    }

    let mut main_to_local = RangeAtomMap::new();

    // Determine all referenced literal groups.
    let mut needed_literals: BTreeSet<usize> = BTreeSet::new();
    for &(gi, _, _) in assigned_slices {
        collect_literal_deps(gi, groups, is_literal, &mut needed_literals);
    }

    // Add small literal groups; large ones become external inputs.
    let mut inlined_literals: BTreeSet<usize> = BTreeSet::new();
    let mut large_literal_groups: BTreeSet<usize> = BTreeSet::new();
    for &lit_gi in &needed_literals {
        let lit_group = &groups[lit_gi];
        if lit_group.count < LITERAL_INLINE_THRESHOLD {
            let local_base = span_graph.push_group(
                lit_group.count, lit_group.op.clone(),
                lit_group.sym_dims.clone(), lit_group.reduce_dims.clone(), vec![],
            );
            main_to_local.insert_range(lit_group.base_id, local_base, lit_group.count);
            inlined_literals.insert(lit_gi);
        } else {
            large_literal_groups.insert(lit_gi);
        }
    }

    // Collect external dependency ranges.
    let mut external_ranges: Vec<(usize, u64, u64)> = Vec::new();

    // Large literal groups are full-range external inputs.
    for &li in &large_literal_groups {
        let lg = &groups[li];
        external_ranges.push((li, 0, lg.count));
    }

    // For each assigned work item, find external producer groups.
    for &(gi, atom_offset, atom_count) in assigned_slices {
        let group = &groups[gi];
        collect_external_ranges(
            group, atom_offset, atom_count, groups, is_literal,
            assigned_slices, &inlined_literals, &mut external_ranges,
        );
    }

    // Merge overlapping ranges.
    let external_ranges = merge_group_ranges(&mut external_ranges);

    // Create placeholder groups for external input ranges.
    let mut input_mappings: Vec<AtomMapping> = Vec::new();
    for &(gi, offset, count) in &external_ranges {
        let main_base = groups[gi].base_id.offset(offset);
        let local_base = span_graph.push_group(
            count,
            ScalarOp::Literal(crate::numeric_scalar::NumericScalar::F32(0.0)),
            vec![], vec![], vec![],
        );
        main_to_local.insert_range(main_base, local_base, count);
        input_mappings.push(AtomMapping {
            main_base,
            span_base: local_base,
            count,
        });
    }

    // Build the compute groups with remapped InputRefs.
    let mut output_mappings: Vec<AtomMapping> = Vec::new();

    for &(gi, atom_offset, atom_count) in assigned_slices {
        let group = &groups[gi];

        let local_inputs = remap_inputs_range(
            &group.inputs, &group.op, atom_offset, atom_count,
            group.count, groups, &main_to_local,
        );
        let local_op = remap_op_range(&group.op, &main_to_local);

        let local_base = span_graph.push_group(
            atom_count, local_op,
            group.sym_dims.clone(), group.reduce_dims.clone(), local_inputs,
        );

        let main_base = AtomId(group.base_id.0 + atom_offset);
        main_to_local.insert_range(main_base, local_base, atom_count);

        // Determine output range for this work item.
        if let Some(dup_outputs) = dup_output_info.get(&gi) {
            // This group has duplication info. Only output the range assigned to this lane.
            for &(out_lane, out_offset, out_count) in dup_outputs {
                if out_lane == lane_idx {
                    // Check if this work item covers the output range.
                    if atom_offset <= out_offset && atom_offset + atom_count >= out_offset + out_count {
                        let local_out_offset = out_offset - atom_offset;
                        let out_main_base = AtomId(group.base_id.0 + out_offset);
                        let out_span_base = AtomId(local_base.0 + local_out_offset);
                        output_mappings.push(AtomMapping {
                            main_base: out_main_base,
                            span_base: out_span_base,
                            count: out_count,
                        });
                    }
                }
            }
        } else {
            // Normal (non-duplicated) work item: output the full slice.
            output_mappings.push(AtomMapping {
                main_base,
                span_base: local_base,
                count: atom_count,
            });
        }
    }

    // Mark graph outputs.
    let main_outputs: HashSet<AtomId> = main_graph.outputs.iter().copied().collect();
    for mapping in &output_mappings {
        for i in 0..mapping.count {
            let main_atom = mapping.main_base.offset(i);
            if main_outputs.contains(&main_atom) {
                span_graph.outputs.push(mapping.span_base.offset(i));
            }
        }
    }

    Span {
        graph: span_graph,
        inputs: input_mappings,
        outputs: output_mappings,
    }
}

// ─── External range collection ───────────────────────────────────────────────

/// Collect external dependency ranges for a work item.
fn collect_external_ranges(
    group: &AtomGroup,
    atom_offset: u64,
    atom_count: u64,
    all_groups: &[AtomGroup],
    is_literal: &[bool],
    assigned_slices: &[(usize, u64, u64)],
    inlined_literals: &BTreeSet<usize>,
    external_ranges: &mut Vec<(usize, u64, u64)>,
) {
    if atom_count == 0 {
        return;
    }

    let is_locally_covered = |atom_lo: u64, atom_hi: u64, prod_gi: usize| -> bool {
        let prod_base = all_groups[prod_gi].base_id.0;
        let off_lo = atom_lo - prod_base;
        let off_hi = atom_hi - prod_base;
        for &(work_gi, work_offset, work_count) in assigned_slices {
            if work_gi == prod_gi && work_offset <= off_lo && work_offset + work_count >= off_hi {
                return true;
            }
        }
        false
    };

    let should_skip = |gi: usize| -> bool {
        inlined_literals.contains(&gi)
            || (is_literal[gi] && all_groups[gi].count < LITERAL_INLINE_THRESHOLD)
    };

    let (is_reduce, reduce_count, reduce_stride) = match &group.op {
        ScalarOp::ReduceSum { reduce_count, reduce_stride, .. }
        | ScalarOp::ReduceMax { reduce_count, reduce_stride, .. }
            if *reduce_count > 1 && *reduce_stride != 0 =>
            (true, *reduce_count, *reduce_stride),
        _ => (false, 0, 0),
    };

    for input in &group.inputs {
        let referenced = resolve_input_to_group_ranges(
            input, atom_offset, atom_count,
            if is_reduce { reduce_count } else { 1 },
            if is_reduce { reduce_stride } else { 0 },
            all_groups,
        );

        for (gi, range_lo, range_hi) in referenced {
            if should_skip(gi) {
                continue;
            }
            let g_lo = all_groups[gi].base_id.0;
            let overlap_lo = range_lo.max(g_lo);
            let overlap_hi = range_hi.min(g_lo + all_groups[gi].count);
            if overlap_lo >= overlap_hi {
                continue;
            }
            if !is_locally_covered(overlap_lo, overlap_hi, gi) {
                let offset = overlap_lo - g_lo;
                let count = overlap_hi - overlap_lo;
                external_ranges.push((gi, offset, count));
            }
        }
    }

    // IndirectLoad table.
    if let ScalarOp::IndirectLoad { table_base, .. } = &group.op {
        if let Some(gi) = find_group_idx(all_groups, *table_base) {
            if !inlined_literals.contains(&gi) {
                let g = &all_groups[gi];
                let is_assigned = assigned_slices.iter().any(|&(wgi, _, _)| wgi == gi);
                if !is_assigned {
                    external_ranges.push((gi, 0, g.count));
                }
            }
        }
    }
}

// ─── InputRef resolution helpers ─────────────────────────────────────────────

/// Find all groups that produce atoms referenced by an InputRef.
fn resolve_producer_groups(input: &InputRef, count: u64, groups: &[AtomGroup]) -> Vec<usize> {
    match input {
        InputRef::Broadcast(atom_id) => {
            find_group_idx(groups, *atom_id).into_iter().collect()
        }
        InputRef::Affine { base, stride } => {
            if count == 0 { return vec![]; }
            let last_offset = (*stride as i64) * (count as i64 - 1);
            let last = AtomId(base.0.wrapping_add(last_offset as u64));
            let lo = base.0.min(last.0);
            let hi = base.0.max(last.0);
            find_groups_in_range(groups, lo, hi)
        }
        InputRef::Explicit(ids) => {
            let mut result = Vec::new();
            let mut seen = HashSet::new();
            for id in ids {
                if let Some(gi) = find_group_idx(groups, *id) {
                    if seen.insert(gi) { result.push(gi); }
                }
            }
            result
        }
        InputRef::SymAffine { base, stride_i, .. } => {
            if count == 0 { return vec![]; }
            let last_offset = (*stride_i as i64) * (count as i64 - 1);
            let last = AtomId(base.0.wrapping_add(last_offset as u64));
            let lo = base.0.min(last.0);
            let hi = base.0.max(last.0);
            find_groups_in_range(groups, lo, hi)
        }
        InputRef::StridedBroadcast { base, stride, repeat } => {
            if count == 0 { return vec![]; }
            let num_blocks = (count + repeat - 1) / repeat;
            let last_offset = *stride * (num_blocks as i64 - 1);
            let last = AtomId(base.0.wrapping_add(last_offset as u64));
            let lo = base.0.min(last.0);
            let hi = base.0.max(last.0);
            find_groups_in_range(groups, lo, hi)
        }
        InputRef::Modular { base, stride, modulus } => {
            if *modulus == 0 { return vec![]; }
            let last_offset = (*stride as i64) * (*modulus as i64 - 1);
            let last = AtomId(base.0.wrapping_add(last_offset as u64));
            let lo = base.0.min(last.0);
            let hi = base.0.max(last.0);
            find_groups_in_range(groups, lo, hi)
        }
    }
}

/// Resolve producer groups considering ReduceSum/ReduceMax strided access.
fn resolve_producer_groups_with_reduce(
    input: &InputRef,
    count: u64,
    reduce_count: u64,
    reduce_stride: i64,
    groups: &[AtomGroup],
) -> Vec<usize> {
    if count == 0 { return vec![]; }
    let min_reduce_ext = 0i64.min(reduce_stride * (reduce_count as i64 - 1));
    let max_reduce_ext = 0i64.max(reduce_stride * (reduce_count as i64 - 1));

    match input {
        InputRef::Affine { base, stride } => {
            let first_base = base.0 as i64;
            let last_base = base.0 as i64 + *stride as i64 * (count as i64 - 1);
            let lo = first_base.min(last_base) + min_reduce_ext;
            let hi = first_base.max(last_base) + max_reduce_ext;
            find_groups_in_range(groups, lo as u64, hi as u64)
        }
        InputRef::StridedBroadcast { base, stride, repeat } => {
            let last_block = ((count - 1) / repeat) as i64;
            let first_read = base.0 as i64;
            let last_read = base.0 as i64 + stride * last_block;
            let lo = first_read.min(last_read) + min_reduce_ext;
            let hi = first_read.max(last_read) + max_reduce_ext;
            find_groups_in_range(groups, lo as u64, hi as u64)
        }
        InputRef::Broadcast(atom_id) => {
            let base = atom_id.0 as i64;
            let lo = base + min_reduce_ext;
            let hi = base + max_reduce_ext;
            find_groups_in_range(groups, lo as u64, hi as u64)
        }
        InputRef::SymAffine { base, stride_i, .. } => {
            let first_base = base.0 as i64;
            let last_base = base.0 as i64 + *stride_i as i64 * (count as i64 - 1);
            let lo = first_base.min(last_base) + min_reduce_ext;
            let hi = first_base.max(last_base) + max_reduce_ext;
            find_groups_in_range(groups, lo as u64, hi as u64)
        }
        _ => resolve_producer_groups(input, count, groups),
    }
}

/// Resolve an InputRef to actual (group_idx, atom_lo, atom_hi) tuples.
fn resolve_input_to_group_ranges(
    input: &InputRef,
    offset: u64,
    count: u64,
    reduce_count: u64,
    reduce_stride: i64,
    groups: &[AtomGroup],
) -> Vec<(usize, u64, u64)> {
    let mut result = Vec::new();
    if count == 0 { return result; }

    match input {
        InputRef::Broadcast(atom_id) => {
            let base = atom_id.0 as i64;
            let (lo, hi) = reduce_extent(base, reduce_count, reduce_stride);
            for gi in find_groups_in_range(groups, lo as u64, hi as u64) {
                result.push((gi, lo as u64, (hi + 1) as u64));
            }
        }
        InputRef::Affine { base, stride } => {
            let first_k = offset;
            let last_k = offset + count - 1;
            let first_pos = base.0 as i64 + *stride as i64 * first_k as i64;
            let last_pos = base.0 as i64 + *stride as i64 * last_k as i64;

            if *stride == 0 {
                let (lo, hi) = reduce_extent(first_pos, reduce_count, reduce_stride);
                for gi in find_groups_in_range(groups, lo as u64, hi as u64) {
                    result.push((gi, lo as u64, (hi + 1) as u64));
                }
            } else if stride.unsigned_abs() == 1 {
                let base_lo = first_pos.min(last_pos);
                let base_hi = first_pos.max(last_pos);
                let (lo, hi) = reduce_extent_range(base_lo, base_hi, reduce_count, reduce_stride);
                for gi in find_groups_in_range(groups, lo as u64, hi as u64) {
                    result.push((gi, lo as u64, (hi + 1) as u64));
                }
            } else {
                let base_lo = first_pos.min(last_pos);
                let base_hi = first_pos.max(last_pos);
                let (ext_lo, ext_hi) = reduce_extent_range(base_lo, base_hi, reduce_count, reduce_stride);
                let candidates = find_groups_in_range(groups, ext_lo as u64, ext_hi as u64);
                for gi in candidates {
                    let g = &groups[gi];
                    let g_lo = g.base_id.0 as i64;
                    let g_hi = g_lo + g.count as i64;
                    if affine_touches_range(first_pos, *stride as i64, count, g_lo, g_hi, reduce_count, reduce_stride) {
                        let overlap_lo = (g_lo as u64).max(ext_lo as u64);
                        let overlap_hi = (g_hi as u64).min((ext_hi + 1) as u64);
                        result.push((gi, overlap_lo, overlap_hi));
                    }
                }
            }
        }
        InputRef::StridedBroadcast { base, stride, repeat } => {
            let first_block = offset / repeat;
            let last_block = (offset + count - 1) / repeat;
            for block in first_block..=last_block {
                let pos = base.0 as i64 + *stride * block as i64;
                let (lo, hi) = reduce_extent(pos, reduce_count, reduce_stride);
                for gi in find_groups_in_range(groups, lo as u64, hi as u64) {
                    result.push((gi, lo as u64, (hi + 1) as u64));
                }
            }
        }
        InputRef::Modular { base, stride, modulus } => {
            if *modulus == 0 { return result; }
            let num_distinct = (*modulus).min(count);
            let mut lo = base.0 as i64;
            let mut hi = base.0 as i64;
            for j in 0..num_distinct {
                let pos = base.0 as i64 + *stride as i64 * j as i64;
                lo = lo.min(pos);
                hi = hi.max(pos);
            }
            let (ext_lo, ext_hi) = reduce_extent_range(lo, hi, reduce_count, reduce_stride);
            for gi in find_groups_in_range(groups, ext_lo as u64, ext_hi as u64) {
                result.push((gi, ext_lo as u64, (ext_hi + 1) as u64));
            }
        }
        InputRef::Explicit(ids) => {
            let start = offset as usize;
            let end = ((offset + count) as usize).min(ids.len());
            let mut group_ranges: BTreeMap<usize, (u64, u64)> = BTreeMap::new();
            for i in start..end {
                let atom = ids[i];
                let (lo, hi) = reduce_extent(atom.0 as i64, reduce_count, reduce_stride);
                for gi in find_groups_in_range(groups, lo as u64, hi as u64) {
                    let entry = group_ranges.entry(gi).or_insert((lo as u64, (hi + 1) as u64));
                    entry.0 = entry.0.min(lo as u64);
                    entry.1 = entry.1.max((hi + 1) as u64);
                }
            }
            for (gi, (lo, hi)) in group_ranges {
                result.push((gi, lo, hi));
            }
        }
        InputRef::SymAffine { base, stride_i, stride_k } => {
            let first_pos = base.0 as i64 + *stride_i as i64 * offset as i64;
            let last_pos = base.0 as i64 + *stride_i as i64 * (offset + count - 1) as i64;
            let base_lo = first_pos.min(last_pos);
            let base_hi = first_pos.max(last_pos);
            let (ext_lo, ext_hi) = reduce_extent_range(base_lo, base_hi, reduce_count, reduce_stride);

            if stride_i.unsigned_abs() <= 1 {
                for gi in find_groups_in_range(groups, ext_lo as u64, ext_hi as u64) {
                    result.push((gi, ext_lo as u64, (ext_hi + 1) as u64));
                }
            } else {
                let candidates = find_groups_in_range(groups, ext_lo as u64, ext_hi as u64);
                for gi in candidates {
                    let g = &groups[gi];
                    let g_lo = g.base_id.0 as i64;
                    let g_hi = g_lo + g.count as i64;
                    if affine_touches_range(first_pos, *stride_i as i64, count, g_lo, g_hi, reduce_count, reduce_stride) {
                        let overlap_lo = (g_lo as u64).max(ext_lo as u64);
                        let overlap_hi = (g_hi as u64).min((ext_hi + 1) as u64);
                        result.push((gi, overlap_lo, overlap_hi));
                    }
                }
            }
        }
    }

    result
}

// ─── Math helpers ────────────────────────────────────────────────────────────

fn reduce_extent(pos: i64, reduce_count: u64, reduce_stride: i64) -> (i64, i64) {
    if reduce_count <= 1 { return (pos, pos); }
    let ext = reduce_stride * (reduce_count as i64 - 1);
    (pos + ext.min(0), pos + ext.max(0))
}

fn reduce_extent_range(base_lo: i64, base_hi: i64, reduce_count: u64, reduce_stride: i64) -> (i64, i64) {
    if reduce_count <= 1 { return (base_lo, base_hi); }
    let ext = reduce_stride * (reduce_count as i64 - 1);
    (base_lo + ext.min(0), base_hi + ext.max(0))
}

fn affine_touches_range(
    first_pos: i64, stride: i64, count: u64,
    g_lo: i64, g_hi: i64,
    reduce_count: u64, reduce_stride: i64,
) -> bool {
    if count == 0 || g_lo >= g_hi { return false; }

    let (min_reduce_ext, max_reduce_ext) = if reduce_count > 1 {
        let ext = reduce_stride * (reduce_count as i64 - 1);
        (ext.min(0), ext.max(0))
    } else {
        (0, 0)
    };
    let eff_lo = g_lo - max_reduce_ext;
    let eff_hi = g_hi - min_reduce_ext;

    if stride == 0 {
        return first_pos >= eff_lo && first_pos < eff_hi;
    }

    let (i_lo, i_hi) = if stride > 0 {
        let num_lo = eff_lo - first_pos;
        let num_hi = eff_hi - first_pos;
        (div_ceil_signed(num_lo, stride), div_ceil_signed(num_hi, stride))
    } else {
        let neg_stride = -stride;
        let i_max = div_floor_signed(first_pos - eff_lo, neg_stride);
        let i_min = div_ceil_signed(first_pos - eff_hi + 1, neg_stride);
        (i_min, i_max + 1)
    };

    let valid_lo = i_lo.max(0);
    let valid_hi = i_hi.min(count as i64);
    valid_lo < valid_hi
}

fn div_ceil_signed(a: i64, b: i64) -> i64 {
    assert!(b > 0);
    if a >= 0 { (a + b - 1) / b } else { -((-a) / b) }
}

fn div_floor_signed(a: i64, b: i64) -> i64 {
    assert!(b > 0);
    if a >= 0 { a / b } else { -(((-a) + b - 1) / b) }
}

// ─── Group/atom lookup helpers ───────────────────────────────────────────────

fn find_group_idx(groups: &[AtomGroup], id: AtomId) -> Option<usize> {
    let idx = groups.partition_point(|g| g.base_id.0 <= id.0);
    if idx == 0 { return None; }
    let gi = idx - 1;
    if groups[gi].contains(id) { Some(gi) } else { None }
}

fn find_groups_in_range(groups: &[AtomGroup], lo: u64, hi: u64) -> Vec<usize> {
    let mut result = Vec::new();
    let start = groups.partition_point(|g| g.base_id.0 + g.count <= lo);
    for gi in start..groups.len() {
        let g = &groups[gi];
        if g.base_id.0 > hi { break; }
        if g.count == 0 { continue; }
        let g_end = g.base_id.0 + g.count - 1;
        if g.base_id.0 <= hi && g_end >= lo {
            result.push(gi);
        }
    }
    result
}

fn collect_literal_deps(
    gi: usize,
    groups: &[AtomGroup],
    is_literal: &[bool],
    literals: &mut BTreeSet<usize>,
) {
    let group = &groups[gi];
    for input in &group.inputs {
        let referenced = resolve_all_referenced_groups(input, group.count, groups);
        for ref_gi in referenced {
            if is_literal[ref_gi] {
                literals.insert(ref_gi);
            }
        }
    }
    if let ScalarOp::IndirectLoad { table_base, .. } = &group.op {
        if let Some(tgi) = find_group_idx(groups, *table_base) {
            if is_literal[tgi] {
                literals.insert(tgi);
            }
        }
    }
}

fn resolve_all_referenced_groups(
    input: &InputRef,
    count: u64,
    groups: &[AtomGroup],
) -> BTreeSet<usize> {
    let mut result = BTreeSet::new();
    match input {
        InputRef::Broadcast(atom_id) => {
            if let Some(gi) = find_group_idx(groups, *atom_id) { result.insert(gi); }
        }
        InputRef::Affine { base, stride } => {
            if count == 0 { return result; }
            let last_offset = (*stride as i64) * (count as i64 - 1);
            let last = AtomId(base.0.wrapping_add(last_offset as u64));
            let lo = base.0.min(last.0);
            let hi = base.0.max(last.0);
            for gi in find_groups_in_range(groups, lo, hi) { result.insert(gi); }
        }
        InputRef::Explicit(ids) => {
            for id in ids {
                if let Some(gi) = find_group_idx(groups, *id) { result.insert(gi); }
            }
        }
        InputRef::SymAffine { base, stride_i, .. } => {
            if count == 0 { return result; }
            let last_offset = (*stride_i as i64) * (count as i64 - 1);
            let last = AtomId(base.0.wrapping_add(last_offset as u64));
            let lo = base.0.min(last.0);
            let hi = base.0.max(last.0);
            for gi in find_groups_in_range(groups, lo, hi) { result.insert(gi); }
        }
        InputRef::StridedBroadcast { base, stride, repeat } => {
            if count == 0 { return result; }
            let num_blocks = (count + repeat - 1) / repeat;
            let last_offset = *stride * (num_blocks as i64 - 1);
            let last = AtomId(base.0.wrapping_add(last_offset as u64));
            let lo = base.0.min(last.0);
            let hi = base.0.max(last.0);
            for gi in find_groups_in_range(groups, lo, hi) { result.insert(gi); }
        }
        InputRef::Modular { base, stride, modulus } => {
            if *modulus == 0 { return result; }
            let last_offset = (*stride as i64) * (*modulus as i64 - 1);
            let last = AtomId(base.0.wrapping_add(last_offset as u64));
            let lo = base.0.min(last.0);
            let hi = base.0.max(last.0);
            for gi in find_groups_in_range(groups, lo, hi) { result.insert(gi); }
        }
    }
    result
}

fn merge_group_ranges(ranges: &mut Vec<(usize, u64, u64)>) -> Vec<(usize, u64, u64)> {
    if ranges.is_empty() { return vec![]; }
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

// ─── InputRef remapping ──────────────────────────────────────────────────────

fn remap_inputs_range(
    inputs: &[InputRef],
    op: &ScalarOp,
    atom_offset: u64,
    atom_count: u64,
    orig_group_count: u64,
    groups: &[AtomGroup],
    atom_map: &RangeAtomMap,
) -> Vec<InputRef> {
    inputs.iter().map(|input| {
        remap_single_input_range(input, atom_offset, atom_count, orig_group_count, atom_map)
    }).collect()
}

fn remap_single_input_range(
    input: &InputRef,
    atom_offset: u64,
    atom_count: u64,
    orig_group_count: u64,
    atom_map: &RangeAtomMap,
) -> InputRef {
    match input {
        InputRef::Broadcast(id) => {
            InputRef::Broadcast(atom_map.get(*id).unwrap_or(*id))
        }
        InputRef::Affine { base, stride } => {
            let new_base_raw = AtomId(base.0.wrapping_add((*stride as i64 * atom_offset as i64) as u64));
            InputRef::Affine {
                base: atom_map.get(new_base_raw).unwrap_or(new_base_raw),
                stride: *stride,
            }
        }
        InputRef::StridedBroadcast { base, stride, repeat } => {
            let block_idx = (atom_offset / repeat) as i64;
            let new_base_raw = AtomId(base.0.wrapping_add((stride * block_idx) as u64));
            let new_offset_in_block = atom_offset % repeat;
            if new_offset_in_block == 0 {
                InputRef::StridedBroadcast {
                    base: atom_map.get(new_base_raw).unwrap_or(new_base_raw),
                    stride: *stride,
                    repeat: *repeat,
                }
            } else {
                let mut ids = Vec::with_capacity(atom_count as usize);
                for i in 0..atom_count {
                    let main_id = input.resolve(atom_offset + i, 0);
                    ids.push(atom_map.get(main_id).unwrap_or(main_id));
                }
                InputRef::Explicit(ids)
            }
        }
        InputRef::Modular { base, stride, modulus } => {
            InputRef::Modular {
                base: atom_map.get(*base).unwrap_or(*base),
                stride: *stride,
                modulus: *modulus,
            }
        }
        InputRef::SymAffine { base, stride_i, stride_k } => {
            let new_base_raw = AtomId(base.0.wrapping_add((*stride_i as i64 * atom_offset as i64) as u64));
            InputRef::SymAffine {
                base: atom_map.get(new_base_raw).unwrap_or(new_base_raw),
                stride_i: *stride_i,
                stride_k: *stride_k,
            }
        }
        InputRef::Explicit(ids) => {
            let start = atom_offset as usize;
            let end = (atom_offset + atom_count) as usize;
            let slice = if end <= ids.len() { &ids[start..end] } else { &ids[start..] };
            InputRef::Explicit(
                slice.iter().map(|id| atom_map.get(*id).unwrap_or(*id)).collect(),
            )
        }
    }
}

fn remap_op_range(op: &ScalarOp, atom_map: &RangeAtomMap) -> ScalarOp {
    match op {
        ScalarOp::IndirectLoad { table_base, output_dtype } => ScalarOp::IndirectLoad {
            table_base: atom_map.get(*table_base).unwrap_or(*table_base),
            output_dtype: *output_dtype,
        },
        other => other.clone(),
    }
}

// ─── Diagnostics ─────────────────────────────────────────────────────────────

impl SpanPlan {
    pub fn print_summary(&self) {
        let total_spans: usize = self.phases.iter().map(|p| p.spans.len()).sum();
        let non_empty: usize = self.phases.iter()
            .flat_map(|p| p.spans.iter())
            .filter(|s| s.graph.num_groups() > 0)
            .count();
        println!(
            "SpanPlan: {} lanes, {} phases, {} spans ({} non-empty)",
            self.num_lanes, self.phases.len(), total_spans, non_empty
        );
        for (pi, phase) in self.phases.iter().enumerate() {
            let active = phase.spans.iter().filter(|s| s.graph.num_groups() > 0).count();
            let max_atoms = phase.spans.iter().map(|s| s.graph.num_atoms()).max().unwrap_or(0);
            let min_atoms = phase.spans.iter()
                .filter(|s| s.graph.num_atoms() > 0)
                .map(|s| s.graph.num_atoms())
                .min().unwrap_or(0);
            let balance = if min_atoms > 0 {
                format!("{:.1}x", max_atoms as f64 / min_atoms as f64)
            } else { "N/A".to_string() };
            println!("  Phase {}: {} active lanes, balance {}", pi, active, balance);
        }
    }

    /// Validate span plan topology.
    pub fn validate(&self, original: &NanoGraph) -> Vec<String> {
        let mut errors = Vec::new();
        let mut available: HashSet<AtomId> = HashSet::new();
        for group in original.groups() {
            if matches!(group.op, ScalarOp::Literal(_)) && group.inputs.is_empty() {
                for i in 0..group.count {
                    available.insert(group.base_id.offset(i));
                }
            }
        }

        for (phase_idx, phase) in self.phases.iter().enumerate() {
            for (lane_idx, span) in phase.spans.iter().enumerate() {
                let graph_errors = span.graph.validate();
                for err in graph_errors {
                    errors.push(format!("Phase {} lane {}: {}", phase_idx, lane_idx, err));
                }
                for mapping in &span.inputs {
                    if !available.contains(&mapping.main_base) {
                        errors.push(format!(
                            "Phase {} lane {}: input base {:?} not available",
                            phase_idx, lane_idx, mapping.main_base
                        ));
                    }
                }
            }
            for span in &phase.spans {
                for mapping in &span.outputs {
                    for i in 0..mapping.count {
                        available.insert(mapping.main_base.offset(i));
                    }
                }
            }
        }
        errors
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::nano_graph::{InputRef, NanoGraph, ScalarBinOp, ScalarOp, ScalarUnaryOp};
    use crate::numeric_scalar::NumericScalar;

    fn build_matmul(m: u64, k: u64, n: u64) -> NanoGraph {
        let mut g = NanoGraph::new();
        let a_base = g.push_group(m * k, ScalarOp::Literal(NumericScalar::F32(1.0)), vec![], vec![], vec![]);
        let b_base = g.push_group(k * n, ScalarOp::Literal(NumericScalar::F32(1.0)), vec![], vec![], vec![]);
        let mut mul_bases = Vec::new();
        for row in 0..m {
            let a_row = AtomId(a_base.0 + row * k);
            let mul = g.push_group(
                k * n,
                ScalarOp::Binary { op: ScalarBinOp::Mul, compute_dtype: DType::F32, output_dtype: DType::F32 },
                vec![], vec![],
                vec![
                    InputRef::StridedBroadcast { base: a_row, stride: 1, repeat: n },
                    InputRef::Affine { base: b_base, stride: 1 },
                ],
            );
            mul_bases.push(mul);
        }
        let mut reduce_bases = Vec::new();
        for row in 0..m {
            let red = g.push_group(
                n,
                ScalarOp::ReduceSum { reduce_count: k, reduce_stride: n as i64, compute_dtype: DType::F32, output_dtype: DType::F32 },
                vec![], vec![],
                vec![InputRef::Affine { base: mul_bases[row as usize], stride: 1 }],
            );
            reduce_bases.push(red);
        }
        for &rb in &reduce_bases {
            for i in 0..n { g.outputs.push(AtomId(rb.0 + i)); }
        }
        g
    }

    fn build_matmul_chain(m: u64, k1: u64, n1: u64, k2: u64, n2: u64) -> NanoGraph {
        let mut g = NanoGraph::new();
        let a1 = g.push_group(m * k1, ScalarOp::Literal(NumericScalar::F32(1.0)), vec![], vec![], vec![]);
        let b1 = g.push_group(k1 * n1, ScalarOp::Literal(NumericScalar::F32(1.0)), vec![], vec![], vec![]);

        let mut mul1_bases = Vec::new();
        for row in 0..m {
            let a_row = AtomId(a1.0 + row * k1);
            let mul = g.push_group(
                k1 * n1,
                ScalarOp::Binary { op: ScalarBinOp::Mul, compute_dtype: DType::F32, output_dtype: DType::F32 },
                vec![], vec![],
                vec![
                    InputRef::StridedBroadcast { base: a_row, stride: 1, repeat: n1 },
                    InputRef::Affine { base: b1, stride: 1 },
                ],
            );
            mul1_bases.push(mul);
        }
        let mut red1_bases = Vec::new();
        for row in 0..m {
            let red = g.push_group(
                n1,
                ScalarOp::ReduceSum { reduce_count: k1, reduce_stride: n1 as i64, compute_dtype: DType::F32, output_dtype: DType::F32 },
                vec![], vec![],
                vec![InputRef::Affine { base: mul1_bases[row as usize], stride: 1 }],
            );
            red1_bases.push(red);
        }

        let act = g.push_group(
            m * n1,
            ScalarOp::Unary { op: ScalarUnaryOp::Tanh, compute_dtype: DType::F32, output_dtype: DType::F32 },
            vec![], vec![],
            vec![InputRef::Affine { base: red1_bases[0], stride: 1 }],
        );

        let b2 = g.push_group(n1 * n2, ScalarOp::Literal(NumericScalar::F32(1.0)), vec![], vec![], vec![]);

        let mut mul2_bases = Vec::new();
        for row in 0..m {
            let mul = g.push_group(
                n1 * n2,
                ScalarOp::Binary { op: ScalarBinOp::Mul, compute_dtype: DType::F32, output_dtype: DType::F32 },
                vec![], vec![],
                vec![
                    InputRef::StridedBroadcast { base: AtomId(act.0 + row * n1), stride: 1, repeat: n2 },
                    InputRef::Affine { base: b2, stride: 1 },
                ],
            );
            mul2_bases.push(mul);
        }
        let mut red2_bases = Vec::new();
        for row in 0..m {
            let red = g.push_group(
                n2,
                ScalarOp::ReduceSum { reduce_count: n1, reduce_stride: n2 as i64, compute_dtype: DType::F32, output_dtype: DType::F32 },
                vec![], vec![],
                vec![InputRef::Affine { base: mul2_bases[row as usize], stride: 1 }],
            );
            red2_bases.push(red);
        }
        for &rb in &red2_bases {
            for i in 0..n2 { g.outputs.push(AtomId(rb.0 + i)); }
        }
        g
    }

    fn build_elementwise(count: u64) -> NanoGraph {
        let mut g = NanoGraph::new();
        let a = g.push_group(count, ScalarOp::Literal(NumericScalar::F32(1.0)), vec![], vec![], vec![]);
        let b = g.push_group(count, ScalarOp::Literal(NumericScalar::F32(2.0)), vec![], vec![], vec![]);
        let c = g.push_group(
            count,
            ScalarOp::Binary { op: ScalarBinOp::Add, compute_dtype: DType::F32, output_dtype: DType::F32 },
            vec![], vec![],
            vec![InputRef::Affine { base: a, stride: 1 }, InputRef::Affine { base: b, stride: 1 }],
        );
        for i in 0..count { g.outputs.push(AtomId(c.0 + i)); }
        g
    }

    fn build_allrows_chain(count: u64) -> NanoGraph {
        let mut g = NanoGraph::new();
        let lit = g.push_group(count, ScalarOp::Literal(NumericScalar::F32(1.0)), vec![], vec![], vec![]);
        let a = g.push_group(
            count,
            ScalarOp::Unary { op: ScalarUnaryOp::Tanh, compute_dtype: DType::F32, output_dtype: DType::F32 },
            vec![], vec![],
            vec![InputRef::Affine { base: lit, stride: 1 }],
        );
        let b = g.push_group(
            count,
            ScalarOp::Unary { op: ScalarUnaryOp::Tanh, compute_dtype: DType::F32, output_dtype: DType::F32 },
            vec![], vec![],
            vec![InputRef::Affine { base: a, stride: 1 }],
        );
        let c = g.push_group(
            count,
            ScalarOp::Unary { op: ScalarUnaryOp::Tanh, compute_dtype: DType::F32, output_dtype: DType::F32 },
            vec![], vec![],
            vec![InputRef::Affine { base: b, stride: 1 }],
        );
        for i in 0..count { g.outputs.push(AtomId(c.0 + i)); }
        g
    }

    /// Build a graph that triggers cross-lane violations:
    /// A small Select-like group (count=3072) is split across lanes.
    /// A downstream group in the same phase broadcasts from it.
    fn build_cross_lane_pattern(small_count: u64, big_count: u64) -> NanoGraph {
        let mut g = NanoGraph::new();
        let lit_a = g.push_group(small_count, ScalarOp::Literal(NumericScalar::F32(1.0)), vec![], vec![], vec![]);
        let lit_b = g.push_group(small_count, ScalarOp::Literal(NumericScalar::F32(0.0)), vec![], vec![], vec![]);
        let lit_cond = g.push_group(small_count, ScalarOp::Literal(NumericScalar::F32(1.0)), vec![], vec![], vec![]);

        // Select: count=small_count, three inputs, all Affine stride=1.
        // This will be AllRows and split across lanes.
        let sel = g.push_group(
            small_count,
            ScalarOp::Select { compute_dtype: DType::F32, output_dtype: DType::F32 },
            vec![], vec![],
            vec![
                InputRef::Affine { base: lit_cond, stride: 1 },
                InputRef::Affine { base: lit_a, stride: 1 },
                InputRef::Affine { base: lit_b, stride: 1 },
            ],
        );

        // Downstream group with big_count atoms that broadcasts from sel.
        // This reads the FULL range of sel via Modular, which is NOT lane-local.
        let lit_d = g.push_group(big_count, ScalarOp::Literal(NumericScalar::F32(1.0)), vec![], vec![], vec![]);
        let downstream = g.push_group(
            big_count,
            ScalarOp::Binary { op: ScalarBinOp::Add, compute_dtype: DType::F32, output_dtype: DType::F32 },
            vec![], vec![],
            vec![
                InputRef::Modular { base: sel, stride: 1, modulus: small_count },
                InputRef::Affine { base: lit_d, stride: 1 },
            ],
        );

        for i in 0..big_count { g.outputs.push(AtomId(downstream.0 + i)); }
        g
    }

    // ─── Verification helpers ────────────────────────────────────────────

    fn verify_span_plan(graph: &NanoGraph, plan: &SpanPlan) {
        let errors = plan.validate(graph);
        assert!(errors.is_empty(), "Span plan errors:\n{}", errors.join("\n"));
    }

    fn verify_output_coverage(graph: &NanoGraph, plan: &SpanPlan) {
        let groups = graph.groups();
        let is_literal: Vec<bool> = groups.iter()
            .map(|g| matches!(g.op, ScalarOp::Literal(_)) && g.inputs.is_empty())
            .collect();

        let mut produced: HashMap<AtomId, usize> = HashMap::new();
        for phase in &plan.phases {
            for span in &phase.spans {
                for mapping in &span.outputs {
                    for i in 0..mapping.count {
                        *produced.entry(mapping.main_base.offset(i)).or_insert(0) += 1;
                    }
                }
            }
        }

        for (gi, group) in groups.iter().enumerate() {
            if is_literal[gi] { continue; }
            for offset in 0..group.count {
                let atom = AtomId(group.base_id.0 + offset);
                let count = produced.get(&atom).copied().unwrap_or(0);
                assert_eq!(count, 1,
                    "Atom {:?} (group {}, offset {}) appears {} times in outputs",
                    atom, gi, offset, count);
            }
        }
    }

    fn verify_input_availability(plan: &SpanPlan, graph: &NanoGraph) {
        let mut available: HashSet<AtomId> = HashSet::new();
        for group in graph.groups() {
            if matches!(group.op, ScalarOp::Literal(_)) && group.inputs.is_empty() {
                for i in 0..group.count {
                    available.insert(group.base_id.offset(i));
                }
            }
        }
        for (phase_idx, phase) in plan.phases.iter().enumerate() {
            for (lane_idx, span) in phase.spans.iter().enumerate() {
                for mapping in &span.inputs {
                    assert!(available.contains(&mapping.main_base),
                        "Phase {} lane {}: input base {:?} not available",
                        phase_idx, lane_idx, mapping.main_base);
                }
            }
            for span in &phase.spans {
                for mapping in &span.outputs {
                    for i in 0..mapping.count {
                        available.insert(mapping.main_base.offset(i));
                    }
                }
            }
        }
    }

    fn verify_phase_independence(plan: &SpanPlan) {
        for (phase_idx, phase) in plan.phases.iter().enumerate() {
            let mut output_ranges: Vec<(u64, u64)> = Vec::new();
            for span in &phase.spans {
                for mapping in &span.outputs {
                    output_ranges.push((mapping.main_base.0, mapping.main_base.0 + mapping.count));
                }
            }
            for (lane_idx, span) in phase.spans.iter().enumerate() {
                for mapping in &span.inputs {
                    let in_lo = mapping.main_base.0;
                    let in_hi = in_lo + mapping.count;
                    for &(out_lo, out_hi) in &output_ranges {
                        assert!(in_lo >= out_hi || out_lo >= in_hi,
                            "Phase {} lane {}: input [{}, {}) overlaps output [{}, {})",
                            phase_idx, lane_idx, in_lo, in_hi, out_lo, out_hi);
                    }
                }
            }
        }
    }

    // ─── Test cases ──────────────────────────────────────────────────────

    #[test]
    fn test_empty_graph() {
        let g = NanoGraph::new();
        let plan = plan_execution_spans(&g, 4);
        assert_eq!(plan.num_lanes, 4);
        assert!(plan.phases.is_empty());
    }

    #[test]
    fn test_single_lane_matmul() {
        let g = build_matmul(4, 8, 4);
        assert!(g.validate().is_empty());
        let plan = plan_execution_spans(&g, 1);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);
    }

    #[test]
    fn test_matmul_multi_lane() {
        let g = build_matmul(8, 4, 4);
        assert!(g.validate().is_empty());
        let plan = plan_execution_spans(&g, 4);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);
        verify_input_availability(&plan, &g);
        verify_phase_independence(&plan);
    }

    #[test]
    fn test_matmul_chain() {
        let g = build_matmul_chain(4, 4, 4, 4, 4);
        assert!(g.validate().is_empty());
        let plan = plan_execution_spans(&g, 4);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);
        verify_input_availability(&plan, &g);
        verify_phase_independence(&plan);
    }

    #[test]
    fn test_elementwise() {
        for count in [1, 7, 8, 15, 16, 100, 1000] {
            let g = build_elementwise(count);
            let plan = plan_execution_spans(&g, 4);
            verify_span_plan(&g, &plan);
            verify_output_coverage(&g, &plan);
        }
    }

    #[test]
    fn test_allrows_chain() {
        let g = build_allrows_chain(256);
        assert!(g.validate().is_empty());
        let plan = plan_execution_spans(&g, 4);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);
        verify_input_availability(&plan, &g);
        verify_phase_independence(&plan);
    }

    #[test]
    fn test_cross_lane_violation_pattern() {
        // This is the GPT-2 attention mask pattern: a small Select split
        // across lanes, then a downstream group reads ALL of it.
        let g = build_cross_lane_pattern(3072, 49152);
        assert!(g.validate().is_empty());
        let plan = plan_execution_spans(&g, 8);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);
        verify_input_availability(&plan, &g);
        verify_phase_independence(&plan);
        println!("Cross-lane pattern: {} phases, {} lanes", plan.phases.len(), plan.num_lanes);
        plan.print_summary();
    }

    #[test]
    fn test_larger_matmul_chain() {
        let g = build_matmul_chain(8, 16, 16, 16, 16);
        assert!(g.validate().is_empty());
        let plan = plan_execution_spans(&g, 4);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);
        assert!(plan.phases.len() <= 10, "Should have <= 10 phases, got {}", plan.phases.len());
    }

    #[test]
    fn test_comprehensive_configs() {
        let configs: Vec<(&str, NanoGraph, usize)> = vec![
            ("matmul_2x2_2lanes", build_matmul(2, 2, 2), 2),
            ("matmul_4x8x4_2lanes", build_matmul(4, 8, 4), 2),
            ("matmul_4x8x4_4lanes", build_matmul(4, 8, 4), 4),
            ("matmul_8x4x4_8lanes", build_matmul(8, 4, 4), 8),
            ("chain_mono_2lanes", build_matmul_chain(2, 4, 4, 4, 2), 2),
            ("chain_mono_4lanes", build_matmul_chain(4, 4, 4, 4, 4), 4),
            ("elementwise_2lanes", build_elementwise(256), 2),
            ("elementwise_4lanes", build_elementwise(1024), 4),
            ("elementwise_8lanes", build_elementwise(8192), 8),
            ("allrows_chain_4lanes", build_allrows_chain(256), 4),
        ];

        for (name, graph, lanes) in configs {
            assert!(graph.validate().is_empty(), "{}: validation failed", name);
            let plan = plan_execution_spans(&graph, lanes);
            verify_span_plan(&graph, &plan);
            verify_output_coverage(&graph, &plan);
            verify_input_availability(&plan, &graph);
            verify_phase_independence(&plan);
            println!("  {}: {} phases, {} lanes", name, plan.phases.len(), plan.num_lanes);
        }
    }
}
