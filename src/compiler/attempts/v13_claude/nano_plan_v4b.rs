#![allow(
    clippy::all,
    dead_code,
    unreachable_patterns,
    unused_variables,
    unused_imports
)]
//! Span-based partitioner v4b: Group-level DAG with transitive dependency closure.
//!
//! Design principles (learned from 15+ failed attempts):
//!
//! 1. ALL operations O(groups). ZERO per-atom iteration.
//! 2. Self-contained — no dependency on v2c or other planners.
//! 3. Correct transitive dependency closure: when a group is included in a span
//!    (whether assigned or duplicated), ALL its non-literal producer groups must
//!    be available in that span (either also included, or declared as external input).
//! 4. Phase assignment respects cross-lane independence: no span reads another
//!    span's output within the same phase.
//!
//! Algorithm:
//!
//! 1. Build group-level producer DAG.
//! 2. Topological sort.
//! 3. For each InputRef on each group, classify the dependency as "lane-aligned"
//!    (splitting the consumer proportionally splits its read of the producer) or
//!    "all-rows" (consumer needs ALL of producer regardless of split position).
//! 4. Walk the DAG in topological order assigning groups to phases. A group's phase
//!    is determined by: if it has an all-rows dependency on a producer that's split
//!    across lanes, it must be in a LATER phase (after a barrier). Otherwise it can
//!    be in the same phase as its producers.
//! 5. Within each phase, split groups across lanes for balance.
//! 6. Build span NanoGraphs with full transitive dependency closure:
//!    - Start with assigned groups.
//!    - Resolve each group's InputRefs to producer groups.
//!    - If a producer is a literal: inline (if small) or declare external.
//!    - If a producer is from an earlier phase: declare external input.
//!    - If a producer is in the same phase, different lane, and small: duplicate
//!      it AND recursively include its transitive dependencies.
//!    - This recursion is the key piece v3c got wrong.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use crate::nano_graph::{AtomGroup, AtomId, InputRef, NanoGraph, ScalarOp};

/// Literal groups with fewer atoms than this are inlined into spans.
const LITERAL_INLINE_THRESHOLD: u64 = 1024;

/// Non-literal groups with fewer atoms than this may be duplicated across lanes
/// to avoid inserting a barrier phase.
const DUPLICATION_THRESHOLD: u64 = 65536;

// ─── Public types ────────────────────────────────────────────────────────────

/// A contiguous range of atoms mapped between main graph and span graph.
#[derive(Debug, Clone)]
pub struct AtomMapping {
    pub main_base: AtomId,
    pub span_base: AtomId,
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

// ─── Public API ──────────────────────────────────────────────────────────────

/// Partition a NanoGraph into self-contained spans organized by phase and lane.
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

    // If all groups are literals, nothing to compute.
    if is_literal.iter().all(|&lit| lit) {
        return SpanPlan {
            num_lanes,
            phases: vec![],
        };
    }

    // Step 1: Build group-level producer DAG (including reduce-extended ranges).
    let producers = build_group_producers(groups, &is_literal);
    let consumers = build_consumers(n, &producers);

    // Step 2: Topological sort.
    let topo_order = topological_sort(n, &producers);

    // Step 3: Classify dependencies as lane-aligned vs all-rows.
    let dep_class = classify_dependencies(groups, &producers, &is_literal);

    // Step 4: Assign phases (determines barrier placement).
    // Step 5: Assign lanes (determines splitting).
    let (group_phase, group_lane_assignments) = assign_phases_and_lanes(
        groups,
        num_lanes,
        &topo_order,
        &producers,
        &consumers,
        &is_literal,
        &dep_class,
    );

    // Step 6: Build span NanoGraphs with transitive dependency closure.
    let num_phases = group_phase.iter().copied().max().unwrap_or(0) + 1;
    build_span_plan(
        graph,
        num_lanes,
        num_phases,
        &group_phase,
        &group_lane_assignments,
        &producers,
        &is_literal,
    )
}

// ─── Group-level producer DAG ────────────────────────────────────────────────

/// Build producer list for each group. Returns producers[gi] = sorted list of
/// group indices that gi reads from (excluding self and literals).
///
/// Accounts for:
/// 1. InputRef resolution (basic case)
/// 2. ReduceSum/ReduceMax strided access (extends read range)
/// 3. IndirectLoad table_base
fn build_group_producers(groups: &[AtomGroup], is_literal: &[bool]) -> Vec<Vec<usize>> {
    let n = groups.len();
    let mut producers: Vec<Vec<usize>> = Vec::with_capacity(n);

    for (gi, group) in groups.iter().enumerate() {
        let mut prod_set = BTreeSet::new();

        // ReduceSum/ReduceMax parameters for extended access.
        let (reduce_count, reduce_stride) = match &group.op {
            ScalarOp::ReduceSum {
                reduce_count,
                reduce_stride,
                ..
            }
            | ScalarOp::ReduceMax {
                reduce_count,
                reduce_stride,
                ..
            } if *reduce_count > 1 && *reduce_stride != 0 => (*reduce_count, *reduce_stride),
            _ => (1, 0),
        };

        for input in &group.inputs {
            let referenced = resolve_input_producer_groups(
                input,
                group.count,
                reduce_count,
                reduce_stride,
                groups,
            );
            for pi in referenced {
                if pi != gi && !is_literal[pi] {
                    prod_set.insert(pi);
                }
            }
        }

        // IndirectLoad table_base.
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

/// Build consumer list from producers.
fn build_consumers(n: usize, producers: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (gi, prods) in producers.iter().enumerate() {
        for &pi in prods {
            consumers[pi].push(gi);
        }
    }
    consumers
}

/// Topological sort using Kahn's algorithm.
fn topological_sort(n: usize, producers: &[Vec<usize>]) -> Vec<usize> {
    let mut in_degree = vec![0usize; n];
    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (gi, prods) in producers.iter().enumerate() {
        in_degree[gi] = prods.len();
        for &pi in prods {
            consumers[pi].push(gi);
        }
    }

    let mut queue: VecDeque<usize> = VecDeque::new();
    for gi in 0..n {
        if in_degree[gi] == 0 {
            queue.push_back(gi);
        }
    }

    let mut order = Vec::with_capacity(n);
    while let Some(gi) = queue.pop_front() {
        order.push(gi);
        for &ci in &consumers[gi] {
            in_degree[ci] -= 1;
            if in_degree[ci] == 0 {
                queue.push_back(ci);
            }
        }
    }
    order
}

// ─── Dependency classification ───────────────────────────────────────────────

/// How a consumer's dependency on a producer relates to splitting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DepKind {
    /// Splitting the consumer proportionally splits its read of the producer.
    /// E.g., Affine stride=1: consumer[offset..offset+count] reads producer[offset..offset+count].
    LaneAligned,
    /// Consumer needs ALL of the producer regardless of how the consumer is split.
    /// E.g., Broadcast, Modular, or ReduceSum that reads across the full producer range.
    AllRows,
}

/// For each group, classify each of its producer dependencies.
/// Returns dep_class[gi] = vec of (producer_gi, DepKind).
fn classify_dependencies(
    groups: &[AtomGroup],
    producers: &[Vec<usize>],
    is_literal: &[bool],
) -> Vec<Vec<(usize, DepKind)>> {
    let n = groups.len();
    let mut dep_class: Vec<Vec<(usize, DepKind)>> = Vec::with_capacity(n);

    for (gi, group) in groups.iter().enumerate() {
        let mut deps = Vec::new();

        // Get reduce params.
        let (reduce_count, reduce_stride) = match &group.op {
            ScalarOp::ReduceSum {
                reduce_count,
                reduce_stride,
                ..
            }
            | ScalarOp::ReduceMax {
                reduce_count,
                reduce_stride,
                ..
            } if *reduce_count > 1 && *reduce_stride != 0 => (*reduce_count, *reduce_stride),
            _ => (1, 0),
        };

        // Track which producers we've already classified (keep strongest = AllRows).
        let mut prod_kind: BTreeMap<usize, DepKind> = BTreeMap::new();

        for input in &group.inputs {
            let kind = classify_input_ref(input, group.count, reduce_count, reduce_stride, groups);
            // Find which producers this input touches.
            let referenced = resolve_input_producer_groups(
                input,
                group.count,
                reduce_count,
                reduce_stride,
                groups,
            );
            for pi in referenced {
                if pi != gi && !is_literal[pi] {
                    let entry = prod_kind.entry(pi).or_insert(kind);
                    // AllRows is stronger than LaneAligned.
                    if kind == DepKind::AllRows {
                        *entry = DepKind::AllRows;
                    }
                }
            }
        }

        // IndirectLoad table is always AllRows.
        if let ScalarOp::IndirectLoad { table_base, .. } = &group.op {
            if let Some(pi) = find_group_idx(groups, *table_base) {
                if pi != gi && !is_literal[pi] {
                    prod_kind.insert(pi, DepKind::AllRows);
                }
            }
        }

        deps.extend(prod_kind.into_iter());
        dep_class.push(deps);
    }

    dep_class
}

/// Classify a single InputRef's dependency kind.
///
/// An InputRef is "lane-aligned" if splitting the consumer into [offset, offset+chunk)
/// causes the read range on the producer to also be a proportional sub-range.
///
/// An InputRef is "all-rows" if the consumer needs ALL of the producer's atoms
/// regardless of which slice of the consumer we're looking at.
///
/// Critical insight: ReduceSum/ReduceMax with reduce_stride causes interleaved
/// access on the producer. Even if the InputRef itself is Affine stride=1, the
/// reduce extends the access range so that each consumer atom reads K scattered
/// positions across the producer. Splitting the consumer does NOT proportionally
/// split the producer read — it's effectively AllRows for the producer group.
fn classify_input_ref(
    input: &InputRef,
    consumer_count: u64,
    reduce_count: u64,
    reduce_stride: i64,
    groups: &[AtomGroup],
) -> DepKind {
    // If reduce extends the access range significantly, the dependency is AllRows
    // because splitting the consumer causes interleaved (non-contiguous) reads
    // on the producer. Specifically: if reduce_stride != 0 and reduce_count > 1,
    // each consumer atom reads reduce_count positions separated by reduce_stride,
    // spanning a range much larger than the consumer's own proportional slice.
    let has_extending_reduce = reduce_count > 1 && reduce_stride != 0;

    match input {
        // Broadcast: every consumer atom reads the same producer atom.
        InputRef::Broadcast(_) => DepKind::AllRows,

        // Affine stride=1: consumer[i] reads producer[base + i*stride].
        // Without reduce: splitting proportionally splits the read. LaneAligned.
        // With reduce: consumer[i] reads base+i*stride, base+i*stride+reduce_stride,
        //   ..., base+i*stride+(K-1)*reduce_stride. The total range spans the
        //   ENTIRE producer group (not just a proportional slice). AllRows.
        InputRef::Affine { stride, .. } => {
            if *stride == 0 {
                DepKind::AllRows
            } else if has_extending_reduce {
                DepKind::AllRows
            } else {
                DepKind::LaneAligned
            }
        }

        // StridedBroadcast: consumer[i] reads producer[base + stride * (i / repeat)].
        // Without reduce: splitting proportionally splits the read. LaneAligned.
        // With reduce: same interleaving problem as Affine.
        InputRef::StridedBroadcast { .. } => {
            if has_extending_reduce {
                DepKind::AllRows
            } else {
                DepKind::LaneAligned
            }
        }

        // Modular: consumer[i] reads producer[base + stride * (i % modulus)].
        // ANY slice of the consumer may read any of the modulus distinct producer atoms.
        InputRef::Modular { .. } => DepKind::AllRows,

        // SymAffine: consumer[i] reads producer[base + stride_i*i + stride_k*k].
        // The i dimension is lane-aligned (splitting consumer splits the read).
        // The k dimension is reduction (handled by sym dims, not reduce_stride).
        // SymAffine is used for matmul Mul groups where the reduction is over sym dims,
        // NOT via ReduceSum's reduce_stride. So has_extending_reduce shouldn't apply.
        InputRef::SymAffine { stride_i, .. } => {
            if *stride_i == 0 {
                DepKind::AllRows
            } else if has_extending_reduce {
                DepKind::AllRows
            } else {
                DepKind::LaneAligned
            }
        }

        // Explicit: arbitrary per-atom mapping. Conservative: treat as all-rows.
        InputRef::Explicit(_) => DepKind::AllRows,
    }
}

// ─── Phase + lane assignment ─────────────────────────────────────────────────

/// A group's lane assignment: either split across lanes or whole on one lane.
#[derive(Debug, Clone)]
enum LaneAssignment {
    /// Group is split evenly across all lanes.
    /// split_ranges[lane] = (offset, count) within the group.
    Split(Vec<(u64, u64)>),
    /// Group is assigned whole to one lane.
    Whole(usize),
}

/// Assign phases and lanes to all groups.
///
/// Phase rule: a group's phase = 1 + max(phase of any producer P where the
/// dependency on P is AllRows AND P is split across lanes). If no such producer
/// exists, the group's phase = max(phase of any producer).
///
/// Lane rule: within a phase, groups are either split across all lanes (for
/// balance) or assigned whole to one lane (if too small to split).
fn assign_phases_and_lanes(
    groups: &[AtomGroup],
    num_lanes: usize,
    topo_order: &[usize],
    producers: &[Vec<usize>],
    consumers: &[Vec<usize>],
    is_literal: &[bool],
    dep_class: &[Vec<(usize, DepKind)>],
) -> (Vec<usize>, Vec<LaneAssignment>) {
    let n = groups.len();
    let mut group_phase = vec![0usize; n];
    let mut is_split = vec![false; n];
    let min_split_size = num_lanes as u64; // Don't split groups smaller than num_lanes.

    // First pass: determine which groups will be split.
    // A group is split if it has enough atoms and at least one consumer could
    // benefit (i.e., has a lane-aligned dependency on it).
    for gi in 0..n {
        if is_literal[gi] {
            continue;
        }
        // Split if large enough.
        is_split[gi] = groups[gi].count >= min_split_size;
    }

    // Second pass: assign phases in topological order.
    for &gi in topo_order {
        if is_literal[gi] {
            continue;
        }

        let mut phase = 0usize;
        for &(pi, kind) in &dep_class[gi] {
            let prod_phase = group_phase[pi];
            match kind {
                DepKind::AllRows if is_split[pi] => {
                    // Producer is split across lanes, and we need ALL of it.
                    // We must be in a later phase (after a barrier).
                    // UNLESS the producer is small enough to duplicate.
                    if groups[pi].count <= DUPLICATION_THRESHOLD {
                        // We'll duplicate it into our span. Same phase is OK.
                        phase = phase.max(prod_phase);
                    } else {
                        // Must wait for barrier.
                        phase = phase.max(prod_phase + 1);
                    }
                }
                DepKind::AllRows => {
                    // Producer is not split (whole on one lane).
                    // If we're on a different lane, we need a barrier.
                    // But we don't know lanes yet. For a whole producer,
                    // any consumer on any lane can read it after the phase completes.
                    // So: same phase + 1 (conservative, ensures barrier).
                    // Actually: if the producer is whole on one lane, other lanes
                    // can only read it after a barrier. But if we assign the
                    // consumer to the SAME lane, no barrier needed.
                    // We'll handle this by assigning the consumer to the same lane
                    // when possible. For now, be conservative.
                    // Actually the simplest correct thing: if the dep is AllRows
                    // and the producer is whole, the consumer must either be on
                    // the same lane OR in a later phase.
                    // We'll resolve this in lane assignment.
                    phase = phase.max(prod_phase);
                }
                DepKind::LaneAligned => {
                    // Lane-aligned: splitting both proportionally is fine.
                    // Same phase is OK.
                    phase = phase.max(prod_phase);
                }
            }
        }
        group_phase[gi] = phase;
    }

    // Third pass: assign lanes.
    let mut group_lane = Vec::with_capacity(n);
    // Track which lane each whole-assigned group goes to, per phase.
    let mut next_lane_in_phase: HashMap<usize, usize> = HashMap::new();

    for gi in 0..n {
        if is_literal[gi] {
            group_lane.push(LaneAssignment::Whole(0)); // placeholder, not used
            continue;
        }

        if is_split[gi] {
            // Split evenly across lanes.
            let count = groups[gi].count;
            let chunk = count / num_lanes as u64;
            let remainder = count % num_lanes as u64;
            let mut ranges = Vec::with_capacity(num_lanes);
            let mut offset = 0u64;
            for lane in 0..num_lanes {
                let this_count = chunk + if (lane as u64) < remainder { 1 } else { 0 };
                ranges.push((offset, this_count));
                offset += this_count;
            }
            group_lane.push(LaneAssignment::Split(ranges));
        } else {
            // Assign whole to a lane.
            // Try to co-locate with AllRows producers that are also whole.
            let mut preferred_lane: Option<usize> = None;
            for &(pi, kind) in &dep_class[gi] {
                if kind == DepKind::AllRows && !is_split[pi] && group_phase[pi] == group_phase[gi] {
                    if let LaneAssignment::Whole(lane) = &group_lane[pi] {
                        preferred_lane = Some(*lane);
                        break;
                    }
                }
            }

            let lane = if let Some(l) = preferred_lane {
                l
            } else {
                // Round-robin within the phase.
                let phase = group_phase[gi];
                let l = next_lane_in_phase.entry(phase).or_insert(0);
                let lane = *l % num_lanes;
                *l += 1;
                lane
            };
            group_lane.push(LaneAssignment::Whole(lane));
        }
    }

    // Fourth pass: fix remaining cross-lane AllRows violations.
    // If a consumer has an AllRows dep on a whole producer in the same phase
    // but different lane, bump the consumer to the next phase.
    let mut changed = true;
    let mut max_iters = 50;
    while changed && max_iters > 0 {
        changed = false;
        max_iters -= 1;
        for &gi in topo_order {
            if is_literal[gi] {
                continue;
            }
            for &(pi, kind) in &dep_class[gi] {
                if kind != DepKind::AllRows {
                    continue;
                }
                if group_phase[pi] != group_phase[gi] {
                    continue; // Already in a later phase, fine.
                }
                if is_split[pi] && groups[pi].count <= DUPLICATION_THRESHOLD {
                    continue; // Will be duplicated, fine.
                }
                if is_split[pi] && groups[pi].count > DUPLICATION_THRESHOLD {
                    // Should have been caught in phase assignment, but check.
                    if group_phase[gi] <= group_phase[pi] {
                        group_phase[gi] = group_phase[pi] + 1;
                        changed = true;
                    }
                    continue;
                }
                // Producer is whole. Check if on same lane.
                let same_lane = match (&group_lane[gi], &group_lane[pi]) {
                    (LaneAssignment::Whole(l1), LaneAssignment::Whole(l2)) => l1 == l2,
                    (LaneAssignment::Split(_), LaneAssignment::Whole(_)) => false,
                    _ => true, // shouldn't happen
                };
                if !same_lane {
                    // Different lane, same phase, AllRows dep.
                    // Option 1: Move consumer to next phase.
                    // Option 2: Move consumer to same lane.
                    // If the producer is small enough to duplicate, keep same phase.
                    if groups[pi].count <= DUPLICATION_THRESHOLD {
                        // Will be duplicated. Fine.
                        continue;
                    }
                    // Otherwise bump phase.
                    group_phase[gi] = group_phase[pi] + 1;
                    changed = true;
                }
            }
        }
    }

    (group_phase, group_lane)
}

// ─── Span building ───────────────────────────────────────────────────────────

/// Build the complete SpanPlan from phase and lane assignments.
fn build_span_plan(
    main_graph: &NanoGraph,
    num_lanes: usize,
    num_phases: usize,
    group_phase: &[usize],
    group_lane: &[LaneAssignment],
    producers: &[Vec<usize>],
    is_literal: &[bool],
) -> SpanPlan {
    let groups = main_graph.groups();
    let n = groups.len();

    // Build work items: (group_idx, offset, count, phase, lane).
    // Each work item represents a slice of a group assigned to a specific lane/phase.
    let mut work_items: Vec<WorkItem> = Vec::new();

    for gi in 0..n {
        if is_literal[gi] {
            continue;
        }
        let phase = group_phase[gi];
        match &group_lane[gi] {
            LaneAssignment::Split(ranges) => {
                for (lane, &(offset, count)) in ranges.iter().enumerate() {
                    if count > 0 {
                        work_items.push(WorkItem {
                            group_idx: gi,
                            atom_offset: offset,
                            atom_count: count,
                            phase,
                            lane,
                            is_duplicate: false,
                        });
                    }
                }
            }
            LaneAssignment::Whole(lane) => {
                work_items.push(WorkItem {
                    group_idx: gi,
                    atom_offset: 0,
                    atom_count: groups[gi].count,
                    phase,
                    lane: *lane,
                    is_duplicate: false,
                });
            }
        }
    }

    // For each span (phase, lane), collect work items and resolve transitive deps.
    // This is where we handle duplication with full transitive closure.
    let mut phase_lane_work: Vec<Vec<Vec<usize>>> = vec![vec![Vec::new(); num_lanes]; num_phases];

    for (wi, item) in work_items.iter().enumerate() {
        phase_lane_work[item.phase][item.lane].push(wi);
    }

    // For each span, find groups that need to be duplicated or upgraded to full copies.
    // A group needs duplication if:
    // 1. It's needed by this span but not assigned here at all, OR
    // 2. It's partially assigned here (split slice) but the span needs MORE of it
    //    than the assigned slice covers (e.g., ReduceSum with strided access).
    // Then transitively resolve their dependencies.
    let mut all_duplicates: Vec<WorkItem> = Vec::new();

    for phase in 0..num_phases {
        for lane in 0..num_lanes {
            let span_work_indices = &phase_lane_work[phase][lane];

            // Build coverage: for each group in this span, what atom ranges are covered.
            let mut span_coverage: HashMap<usize, Vec<(u64, u64)>> = HashMap::new();
            for &wi in span_work_indices {
                let item = &work_items[wi];
                span_coverage
                    .entry(item.group_idx)
                    .or_default()
                    .push((item.atom_offset, item.atom_count));
            }

            // Find all producer groups needed by this span's groups that aren't
            // fully covered by this span.
            let mut needed: BTreeSet<usize> = BTreeSet::new();
            let mut queue: VecDeque<usize> = VecDeque::new();

            for &wi in span_work_indices {
                let gi = work_items[wi].group_idx;
                let group = &groups[gi];
                let offset = work_items[wi].atom_offset;
                let count = work_items[wi].atom_count;

                // Find producer groups and the ranges needed from each.
                let (reduce_count, reduce_stride) = match &group.op {
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
                        (*reduce_count, *reduce_stride)
                    }
                    _ => (1, 0),
                };

                for input in &group.inputs {
                    let ranges = resolve_input_to_group_ranges(
                        input,
                        offset,
                        count,
                        reduce_count,
                        reduce_stride,
                        groups,
                    );
                    for (pi, range_lo, range_hi) in ranges {
                        if is_literal[pi] {
                            continue;
                        }
                        if group_phase[pi] < phase {
                            continue; // Earlier phase, external input.
                        }
                        // Check if the needed range is fully covered by this span.
                        let prod = &groups[pi];
                        let need_off = range_lo.saturating_sub(prod.base_id.0);
                        let need_end = (range_hi.saturating_sub(prod.base_id.0)).min(prod.count);
                        if !is_range_covered_in_span(&span_coverage, pi, need_off, need_end)
                            && !needed.contains(&pi)
                        {
                            needed.insert(pi);
                            queue.push_back(pi);
                        }
                    }
                }

                // IndirectLoad table.
                if let ScalarOp::IndirectLoad { table_base, .. } = &group.op {
                    if let Some(pi) = find_group_idx(groups, *table_base) {
                        if !is_literal[pi]
                            && group_phase[pi] >= phase
                            && !is_range_covered_in_span(&span_coverage, pi, 0, groups[pi].count)
                            && !needed.contains(&pi)
                        {
                            needed.insert(pi);
                            queue.push_back(pi);
                        }
                    }
                }
            }

            // Transitive closure: for each group we're duplicating, also duplicate
            // its non-literal producers (if they're not already fully covered or
            // from an earlier phase).
            while let Some(gi) = queue.pop_front() {
                // We need the FULL group gi. Check what IT needs.
                let group = &groups[gi];
                let (reduce_count, reduce_stride) = match &group.op {
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
                        (*reduce_count, *reduce_stride)
                    }
                    _ => (1, 0),
                };

                for input in &group.inputs {
                    let ranges = resolve_input_to_group_ranges(
                        input,
                        0,
                        group.count,
                        reduce_count,
                        reduce_stride,
                        groups,
                    );
                    for (pi, range_lo, range_hi) in ranges {
                        if is_literal[pi] || needed.contains(&pi) {
                            continue;
                        }
                        if group_phase[pi] < phase {
                            continue;
                        }
                        let prod = &groups[pi];
                        let need_off = range_lo.saturating_sub(prod.base_id.0);
                        let need_end = (range_hi.saturating_sub(prod.base_id.0)).min(prod.count);
                        if !is_range_covered_in_span(&span_coverage, pi, need_off, need_end) {
                            needed.insert(pi);
                            queue.push_back(pi);
                        }
                    }
                }

                // IndirectLoad table.
                if let ScalarOp::IndirectLoad { table_base, .. } = &group.op {
                    if let Some(pi) = find_group_idx(groups, *table_base) {
                        if !is_literal[pi]
                            && !needed.contains(&pi)
                            && group_phase[pi] >= phase
                            && !is_range_covered_in_span(&span_coverage, pi, 0, groups[pi].count)
                        {
                            needed.insert(pi);
                            queue.push_back(pi);
                        }
                    }
                }
            }

            // Create duplicate work items for needed groups.
            for &gi in &needed {
                all_duplicates.push(WorkItem {
                    group_idx: gi,
                    atom_offset: 0,
                    atom_count: groups[gi].count,
                    phase,
                    lane,
                    is_duplicate: true,
                });
            }
        }
    }

    // Add duplicates to the work items list.
    let dup_start = work_items.len();
    work_items.extend(all_duplicates);

    // Rebuild phase_lane_work with duplicates.
    let mut phase_lane_work: Vec<Vec<Vec<usize>>> = vec![vec![Vec::new(); num_lanes]; num_phases];
    for (wi, item) in work_items.iter().enumerate() {
        phase_lane_work[item.phase][item.lane].push(wi);
    }

    // (Output ownership is handled per-slice via output_range in SpanSlice.)

    // Build each span.
    let mut phases = Vec::with_capacity(num_phases);

    for phase_idx in 0..num_phases {
        let mut spans = Vec::with_capacity(num_lanes);

        for lane_idx in 0..num_lanes {
            let wi_list = &phase_lane_work[phase_idx][lane_idx];
            if wi_list.is_empty() {
                spans.push(Span {
                    graph: NanoGraph::new(),
                    inputs: vec![],
                    outputs: vec![],
                });
                continue;
            }

            // Collect all slices for this span, sorted by group_idx for topo order.
            let mut span_slices: Vec<SpanSlice> = wi_list
                .iter()
                .map(|&wi| {
                    let item = &work_items[wi];
                    let output_range = if item.is_duplicate {
                        None // Duplicates don't produce output by default.
                    } else {
                        Some((item.atom_offset, item.atom_count))
                    };
                    SpanSlice {
                        group_idx: item.group_idx,
                        atom_offset: item.atom_offset,
                        atom_count: item.atom_count,
                        output_range,
                        work_item_idx: wi,
                    }
                })
                .collect();
            // Sort by group_idx (topological order since groups are in insertion order).
            span_slices.sort_by_key(|s| (s.group_idx, s.atom_offset));
            // Deduplicate: if a group appears both as original split slice and as
            // full duplicate, the full duplicate subsumes the split slice.
            span_slices = deduplicate_slices(span_slices, groups);

            let span = build_single_span(
                main_graph,
                groups,
                is_literal,
                &span_slices,
                phase_idx,
                lane_idx,
                group_phase,
            );
            spans.push(span);
        }

        phases.push(Phase { spans });
    }

    SpanPlan { num_lanes, phases }
}

/// Work item: a slice of a group assigned to a specific phase/lane.
#[derive(Debug, Clone)]
struct WorkItem {
    group_idx: usize,
    atom_offset: u64,
    atom_count: u64,
    phase: usize,
    lane: usize,
    is_duplicate: bool,
}

/// A group slice in a span.
#[derive(Debug, Clone)]
struct SpanSlice {
    group_idx: usize,
    atom_offset: u64,
    atom_count: u64,
    /// The range of atoms this slice should output (relative to group base).
    /// None means this slice produces no output (pure duplicate for internal use).
    /// Some((off, count)) means output atoms [off, off+count) of the group.
    output_range: Option<(u64, u64)>,
    work_item_idx: usize,
}

/// Deduplicate slices: if a group has both a split slice and a full duplicate,
/// keep only the full duplicate (it covers the split slice), but transfer the
/// output ownership from the split slice to the full duplicate.
fn deduplicate_slices(slices: Vec<SpanSlice>, groups: &[AtomGroup]) -> Vec<SpanSlice> {
    // Find groups that have a full duplicate.
    let has_full_dup: HashSet<usize> = slices
        .iter()
        .filter(|s| {
            s.output_range.is_none()
                && s.atom_offset == 0
                && s.atom_count == groups[s.group_idx].count
        })
        .map(|s| s.group_idx)
        .collect();

    // Collect original output ranges for groups that have full duplicates.
    let mut original_output: HashMap<usize, (u64, u64)> = HashMap::new();
    for s in &slices {
        if has_full_dup.contains(&s.group_idx) && s.output_range.is_some() {
            original_output.insert(s.group_idx, s.output_range.unwrap());
        }
    }

    let mut result = Vec::with_capacity(slices.len());
    let mut seen_full: HashSet<usize> = HashSet::new();

    for mut s in slices {
        if has_full_dup.contains(&s.group_idx) {
            // Is this the full duplicate?
            if s.output_range.is_none()
                && s.atom_offset == 0
                && s.atom_count == groups[s.group_idx].count
            {
                if seen_full.insert(s.group_idx) {
                    // Transfer output ownership from the original split slice.
                    if let Some(&(out_off, out_count)) = original_output.get(&s.group_idx) {
                        s.output_range = Some((out_off, out_count));
                    }
                    result.push(s);
                }
            }
            // Skip original split slices for groups that have a full copy.
        } else {
            result.push(s);
        }
    }
    result
}

/// Sorted-range lookup map for O(log n) main→span atom ID translation.
struct RangeAtomMap {
    map: BTreeMap<u64, (u64, u64)>, // main_base -> (span_base, count)
}

impl RangeAtomMap {
    fn new() -> Self {
        Self {
            map: BTreeMap::new(),
        }
    }

    fn insert_range(&mut self, main_base: AtomId, span_base: AtomId, count: u64) {
        self.map.insert(main_base.0, (span_base.0, count));
    }

    fn get(&self, main_id: AtomId) -> Option<AtomId> {
        use std::ops::Bound;
        let mut iter = self
            .map
            .range((Bound::Unbounded, Bound::Included(main_id.0)));
        if let Some((&base, &(span_base, count))) = iter.next_back() {
            let offset = main_id.0.wrapping_sub(base);
            if offset < count {
                return Some(AtomId(span_base + offset));
            }
        }
        None
    }
}

/// Build a single span NanoGraph.
fn build_single_span(
    main_graph: &NanoGraph,
    groups: &[AtomGroup],
    is_literal: &[bool],
    span_slices: &[SpanSlice],
    phase_idx: usize,
    lane_idx: usize,
    group_phase: &[usize],
) -> Span {
    let mut span_graph = NanoGraph::new();
    let mut main_to_local = RangeAtomMap::new();

    // Copy sym_dim configuration.
    let mut sym_dim_remap: HashMap<crate::nano_graph::SymDim, crate::nano_graph::SymDim> =
        HashMap::new();
    for (name, &sd) in &main_graph.sym_dim_names {
        let local_sd = span_graph.sym_dim(name);
        sym_dim_remap.insert(sd, local_sd);
        if let Some(&bound) = main_graph.sym_dim_bounds.get(&sd) {
            span_graph.sym_dim_bounds.insert(local_sd, bound);
        }
    }

    // Step 1: Identify all literal groups needed by this span's compute groups.
    let mut needed_literals: BTreeSet<usize> = BTreeSet::new();
    for slice in span_slices {
        collect_all_literal_deps(
            slice.group_idx,
            slice.atom_offset,
            slice.atom_count,
            groups,
            is_literal,
            &mut needed_literals,
        );
    }

    // Step 2: Add inlined literals and track large literals as external.
    let mut inlined_literals: BTreeSet<usize> = BTreeSet::new();
    let mut external_lit_groups: BTreeSet<usize> = BTreeSet::new();
    for &lit_gi in &needed_literals {
        let lit_group = &groups[lit_gi];
        if lit_group.count < LITERAL_INLINE_THRESHOLD {
            let local_base = span_graph.push_group(
                lit_group.count,
                lit_group.op.clone(),
                remap_sym_dims(&lit_group.sym_dims, &sym_dim_remap),
                remap_sym_dims(&lit_group.reduce_dims, &sym_dim_remap),
                vec![],
            );
            main_to_local.insert_range(lit_group.base_id, local_base, lit_group.count);
            inlined_literals.insert(lit_gi);
        } else {
            external_lit_groups.insert(lit_gi);
        }
    }

    // Step 3: Determine external input ranges.
    // These are: large literals + non-literal producers from earlier phases.
    let mut external_ranges: Vec<(usize, u64, u64)> = Vec::new();

    // Large literals.
    for &li in &external_lit_groups {
        external_ranges.push((li, 0, groups[li].count));
    }

    // Non-literal producers from earlier phases.
    let span_group_set: HashSet<usize> = span_slices.iter().map(|s| s.group_idx).collect();
    for slice in span_slices {
        let group = &groups[slice.group_idx];
        let slice_prods = compute_slice_producers(
            group,
            slice.atom_offset,
            slice.atom_count,
            groups,
            is_literal,
        );
        for pi in slice_prods {
            if is_literal[pi] || span_group_set.contains(&pi) || inlined_literals.contains(&pi) {
                continue;
            }
            // This producer is from an earlier phase — external input.
            // Compute the specific range needed.
            let needed = compute_needed_range_on_producer(
                group,
                slice.atom_offset,
                slice.atom_count,
                pi,
                groups,
            );
            for (off, cnt) in needed {
                external_ranges.push((pi, off, cnt));
            }
        }
    }

    // Merge overlapping external ranges.
    let external_ranges = merge_group_ranges(&mut external_ranges);

    // Step 4: Add external input placeholder groups.
    let mut input_mappings: Vec<AtomMapping> = Vec::new();
    for &(gi, offset, count) in &external_ranges {
        let main_base = groups[gi].base_id.offset(offset);
        let local_base = span_graph.push_group(
            count,
            ScalarOp::Literal(crate::numeric_scalar::NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );
        main_to_local.insert_range(main_base, local_base, count);
        input_mappings.push(AtomMapping {
            main_base,
            span_base: local_base,
            count,
        });
    }

    // Step 5: Add compute groups with remapped InputRefs.
    let mut output_mappings: Vec<AtomMapping> = Vec::new();

    for slice in span_slices {
        let gi = slice.group_idx;
        let group = &groups[gi];
        let offset = slice.atom_offset;
        let count = slice.atom_count;

        let local_inputs =
            remap_inputs_for_slice(&group.inputs, offset, count, group.count, &main_to_local);
        let local_op = remap_op(&group.op, &main_to_local);

        let local_base = span_graph.push_group(
            count,
            local_op,
            remap_sym_dims(&group.sym_dims, &sym_dim_remap),
            remap_sym_dims(&group.reduce_dims, &sym_dim_remap),
            local_inputs,
        );

        let main_base = AtomId(group.base_id.0 + offset);
        main_to_local.insert_range(main_base, local_base, count);

        // Output: only for the owned output range.
        if let Some((out_off, out_count)) = slice.output_range {
            // The output range is relative to the group base.
            // The slice covers [offset, offset+count) of the group.
            // The output range is [out_off, out_off+out_count).
            // If the slice is a full copy (offset=0, count=full group),
            // out_off/out_count might be a sub-range.
            let out_main_base = AtomId(group.base_id.0 + out_off);
            let out_span_base = AtomId(local_base.0 + (out_off - offset));
            output_mappings.push(AtomMapping {
                main_base: out_main_base,
                span_base: out_span_base,
                count: out_count,
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

// ─── Producer resolution helpers ─────────────────────────────────────────────

/// Find all non-literal producer groups for a specific slice of a group.
fn compute_slice_producers(
    group: &AtomGroup,
    offset: u64,
    count: u64,
    all_groups: &[AtomGroup],
    is_literal: &[bool],
) -> Vec<usize> {
    let mut result = BTreeSet::new();
    if count == 0 {
        return vec![];
    }

    let (reduce_count, reduce_stride) = match &group.op {
        ScalarOp::ReduceSum {
            reduce_count,
            reduce_stride,
            ..
        }
        | ScalarOp::ReduceMax {
            reduce_count,
            reduce_stride,
            ..
        } if *reduce_count > 1 && *reduce_stride != 0 => (*reduce_count, *reduce_stride),
        _ => (1, 0),
    };

    for input in &group.inputs {
        let ranges = resolve_input_to_group_ranges(
            input,
            offset,
            count,
            reduce_count,
            reduce_stride,
            all_groups,
        );
        for (gi, _, _) in ranges {
            if !is_literal[gi] {
                result.insert(gi);
            }
        }
    }

    // IndirectLoad table.
    if let ScalarOp::IndirectLoad { table_base, .. } = &group.op {
        if let Some(gi) = find_group_idx(all_groups, *table_base) {
            if !is_literal[gi] {
                result.insert(gi);
            }
        }
    }

    result.into_iter().collect()
}

/// Compute the specific atom range needed from a producer group by a consumer slice.
/// Returns Vec<(offset_in_producer, count)>.
fn compute_needed_range_on_producer(
    consumer: &AtomGroup,
    consumer_offset: u64,
    consumer_count: u64,
    producer_gi: usize,
    all_groups: &[AtomGroup],
) -> Vec<(u64, u64)> {
    let mut ranges = Vec::new();
    if consumer_count == 0 {
        return ranges;
    }

    let (reduce_count, reduce_stride) = match &consumer.op {
        ScalarOp::ReduceSum {
            reduce_count,
            reduce_stride,
            ..
        }
        | ScalarOp::ReduceMax {
            reduce_count,
            reduce_stride,
            ..
        } if *reduce_count > 1 && *reduce_stride != 0 => (*reduce_count, *reduce_stride),
        _ => (1, 0),
    };

    let prod_group = &all_groups[producer_gi];
    let prod_base = prod_group.base_id.0;
    let prod_end = prod_base + prod_group.count;

    for input in &consumer.inputs {
        let group_ranges = resolve_input_to_group_ranges(
            input,
            consumer_offset,
            consumer_count,
            reduce_count,
            reduce_stride,
            all_groups,
        );
        for (gi, range_lo, range_hi) in group_ranges {
            if gi != producer_gi {
                continue;
            }
            // Clamp to producer group bounds and convert to offset within producer.
            let clamped_lo = range_lo.max(prod_base);
            let clamped_hi = range_hi.min(prod_end);
            if clamped_lo < clamped_hi {
                ranges.push((clamped_lo - prod_base, clamped_hi - clamped_lo));
            }
        }
    }

    // IndirectLoad table.
    if let ScalarOp::IndirectLoad { table_base, .. } = &consumer.op {
        if let Some(gi) = find_group_idx(all_groups, *table_base) {
            if gi == producer_gi {
                ranges.push((0, prod_group.count));
            }
        }
    }

    // Merge overlapping ranges.
    if ranges.len() > 1 {
        ranges.sort_by_key(|&(off, _)| off);
        let mut merged = Vec::new();
        for (off, cnt) in ranges {
            if let Some(last) = merged.last_mut() {
                let (lo, lc): &mut (u64, u64) = last;
                if off <= *lo + *lc {
                    *lc = (*lo + *lc).max(off + cnt) - *lo;
                    continue;
                }
            }
            merged.push((off, cnt));
        }
        merged
    } else {
        ranges
    }
}

/// Collect ALL literal dependencies for a group slice, including reduce-extended ranges.
fn collect_all_literal_deps(
    gi: usize,
    offset: u64,
    count: u64,
    groups: &[AtomGroup],
    is_literal: &[bool],
    literals: &mut BTreeSet<usize>,
) {
    let group = &groups[gi];

    let (reduce_count, reduce_stride) = match &group.op {
        ScalarOp::ReduceSum {
            reduce_count,
            reduce_stride,
            ..
        }
        | ScalarOp::ReduceMax {
            reduce_count,
            reduce_stride,
            ..
        } if *reduce_count > 1 && *reduce_stride != 0 => (*reduce_count, *reduce_stride),
        _ => (1, 0),
    };

    for input in &group.inputs {
        let ranges = resolve_input_to_group_ranges(
            input,
            offset,
            count,
            reduce_count,
            reduce_stride,
            groups,
        );
        for (ref_gi, _, _) in ranges {
            if is_literal[ref_gi] {
                literals.insert(ref_gi);
            }
        }
    }

    // IndirectLoad table.
    if let ScalarOp::IndirectLoad { table_base, .. } = &group.op {
        if let Some(tgi) = find_group_idx(groups, *table_base) {
            if is_literal[tgi] {
                literals.insert(tgi);
            }
        }
    }
}

// ─── InputRef to group range resolution ──────────────────────────────────────

/// Resolve which groups an InputRef references, returning (group_idx, atom_lo, atom_hi).
/// Handles reduce-extended access ranges. atom_lo and atom_hi are in absolute atom space.
fn resolve_input_to_group_ranges(
    input: &InputRef,
    offset: u64,
    count: u64,
    reduce_count: u64,
    reduce_stride: i64,
    groups: &[AtomGroup],
) -> Vec<(usize, u64, u64)> {
    let mut result = Vec::new();
    if count == 0 {
        return result;
    }

    match input {
        InputRef::Broadcast(atom_id) => {
            let base = atom_id.0 as i64;
            let (lo, hi) = reduce_extent(base, reduce_count, reduce_stride);
            for gi in find_groups_in_range(groups, lo as u64, hi as u64) {
                result.push((gi, lo as u64, (hi + 1) as u64));
            }
        }

        InputRef::Affine { base, stride } => {
            let first = base.0 as i64 + *stride as i64 * offset as i64;
            let last = base.0 as i64 + *stride as i64 * (offset + count - 1) as i64;
            let base_lo = first.min(last);
            let base_hi = first.max(last);
            let (ext_lo, ext_hi) =
                reduce_extent_range(base_lo, base_hi, reduce_count, reduce_stride);
            for gi in find_groups_in_range(groups, ext_lo as u64, ext_hi as u64) {
                let g = &groups[gi];
                let g_lo = g.base_id.0;
                let g_hi = g_lo + g.count;
                let overlap_lo = (ext_lo as u64).max(g_lo);
                let overlap_hi = ((ext_hi + 1) as u64).min(g_hi);
                if overlap_lo < overlap_hi {
                    result.push((gi, overlap_lo, overlap_hi));
                }
            }
        }

        InputRef::StridedBroadcast {
            base,
            stride,
            repeat,
        } => {
            let first_block = offset / repeat;
            let last_block = (offset + count - 1) / repeat;
            let first_pos = base.0 as i64 + *stride * first_block as i64;
            let last_pos = base.0 as i64 + *stride * last_block as i64;
            let base_lo = first_pos.min(last_pos);
            let base_hi = first_pos.max(last_pos);
            let (ext_lo, ext_hi) =
                reduce_extent_range(base_lo, base_hi, reduce_count, reduce_stride);
            for gi in find_groups_in_range(groups, ext_lo as u64, ext_hi as u64) {
                let g = &groups[gi];
                let g_lo = g.base_id.0;
                let g_hi = g_lo + g.count;
                let overlap_lo = (ext_lo as u64).max(g_lo);
                let overlap_hi = ((ext_hi + 1) as u64).min(g_hi);
                if overlap_lo < overlap_hi {
                    result.push((gi, overlap_lo, overlap_hi));
                }
            }
        }

        InputRef::Modular {
            base,
            stride,
            modulus,
        } => {
            if *modulus == 0 {
                return result;
            }
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
                let g = &groups[gi];
                let g_lo = g.base_id.0;
                let g_hi = g_lo + g.count;
                let overlap_lo = (ext_lo as u64).max(g_lo);
                let overlap_hi = ((ext_hi + 1) as u64).min(g_hi);
                if overlap_lo < overlap_hi {
                    result.push((gi, overlap_lo, overlap_hi));
                }
            }
        }

        InputRef::SymAffine {
            base,
            stride_i,
            stride_k,
        } => {
            let first = base.0 as i64 + *stride_i as i64 * offset as i64;
            let last = base.0 as i64 + *stride_i as i64 * (offset + count - 1) as i64;
            let base_lo = first.min(last);
            let base_hi = first.max(last);
            let (ext_lo, ext_hi) =
                reduce_extent_range(base_lo, base_hi, reduce_count, reduce_stride);
            // SymAffine also has stride_k — the k dimension extends the range.
            // But stride_k is for reduction iteration, not for splitting.
            // The reduce_stride on the ScalarOp handles this already.
            for gi in find_groups_in_range(groups, ext_lo as u64, ext_hi as u64) {
                let g = &groups[gi];
                let g_lo = g.base_id.0;
                let g_hi = g_lo + g.count;
                let overlap_lo = (ext_lo as u64).max(g_lo);
                let overlap_hi = ((ext_hi + 1) as u64).min(g_hi);
                if overlap_lo < overlap_hi {
                    result.push((gi, overlap_lo, overlap_hi));
                }
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
                    let entry = group_ranges
                        .entry(gi)
                        .or_insert((lo as u64, (hi + 1) as u64));
                    entry.0 = entry.0.min(lo as u64);
                    entry.1 = entry.1.max((hi + 1) as u64);
                }
            }
            for (gi, (lo, hi)) in group_ranges {
                result.push((gi, lo, hi));
            }
        }
    }

    result
}

/// Resolve InputRef to producer group indices (flattened, no ranges).
fn resolve_input_producer_groups(
    input: &InputRef,
    count: u64,
    reduce_count: u64,
    reduce_stride: i64,
    groups: &[AtomGroup],
) -> Vec<usize> {
    let ranges =
        resolve_input_to_group_ranges(input, 0, count, reduce_count, reduce_stride, groups);
    let mut result: BTreeSet<usize> = BTreeSet::new();
    for (gi, _, _) in ranges {
        result.insert(gi);
    }
    result.into_iter().collect()
}

// ─── InputRef remapping for span building ────────────────────────────────────

/// Remap InputRefs for a slice of a group being included in a span.
fn remap_inputs_for_slice(
    inputs: &[InputRef],
    atom_offset: u64,
    atom_count: u64,
    orig_group_count: u64,
    atom_map: &RangeAtomMap,
) -> Vec<InputRef> {
    inputs
        .iter()
        .map(|input| remap_single_input(input, atom_offset, atom_count, orig_group_count, atom_map))
        .collect()
}

/// Remap a single InputRef for a slice.
fn remap_single_input(
    input: &InputRef,
    atom_offset: u64,
    atom_count: u64,
    orig_group_count: u64,
    atom_map: &RangeAtomMap,
) -> InputRef {
    match input {
        InputRef::Broadcast(id) => InputRef::Broadcast(atom_map.get(*id).unwrap_or(*id)),

        InputRef::Affine { base, stride } => {
            let new_base_raw = AtomId(
                base.0
                    .wrapping_add((*stride as i64 * atom_offset as i64) as u64),
            );
            InputRef::Affine {
                base: atom_map.get(new_base_raw).unwrap_or(new_base_raw),
                stride: *stride,
            }
        }

        InputRef::StridedBroadcast {
            base,
            stride,
            repeat,
        } => {
            let block_idx = (atom_offset / repeat) as i64;
            let new_base_raw = AtomId(base.0.wrapping_add((stride * block_idx) as u64));
            let offset_in_block = atom_offset % repeat;
            if offset_in_block == 0 {
                InputRef::StridedBroadcast {
                    base: atom_map.get(new_base_raw).unwrap_or(new_base_raw),
                    stride: *stride,
                    repeat: *repeat,
                }
            } else {
                // Misaligned split — fall back to Explicit.
                // This is per-atom but only for the slice size, not the full group.
                let mut ids = Vec::with_capacity(atom_count as usize);
                for i in 0..atom_count {
                    let main_id = input.resolve(atom_offset + i, 0);
                    ids.push(atom_map.get(main_id).unwrap_or(main_id));
                }
                InputRef::Explicit(ids)
            }
        }

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
        } => {
            let new_base_raw = AtomId(
                base.0
                    .wrapping_add((*stride_i as i64 * atom_offset as i64) as u64),
            );
            InputRef::SymAffine {
                base: atom_map.get(new_base_raw).unwrap_or(new_base_raw),
                stride_i: *stride_i,
                stride_k: *stride_k,
            }
        }

        InputRef::Explicit(ids) => {
            let start = atom_offset as usize;
            let end = (atom_offset + atom_count) as usize;
            let slice = if end <= ids.len() {
                &ids[start..end]
            } else {
                &ids[start..]
            };
            InputRef::Explicit(
                slice
                    .iter()
                    .map(|id| atom_map.get(*id).unwrap_or(*id))
                    .collect(),
            )
        }
    }
}

/// Remap ScalarOp (for IndirectLoad table_base).
fn remap_op(op: &ScalarOp, atom_map: &RangeAtomMap) -> ScalarOp {
    match op {
        ScalarOp::IndirectLoad {
            table_base,
            output_dtype,
        } => ScalarOp::IndirectLoad {
            table_base: atom_map.get(*table_base).unwrap_or(*table_base),
            output_dtype: *output_dtype,
        },
        other => other.clone(),
    }
}

/// Remap sym dims through the mapping.
fn remap_sym_dims(
    dims: &[crate::nano_graph::SymDim],
    remap: &HashMap<crate::nano_graph::SymDim, crate::nano_graph::SymDim>,
) -> Vec<crate::nano_graph::SymDim> {
    dims.iter()
        .map(|d| remap.get(d).copied().unwrap_or(*d))
        .collect()
}

// ─── Group/atom lookup helpers ───────────────────────────────────────────────

/// Binary search for the group containing an AtomId.
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

/// Find all groups overlapping the atom range [lo, hi].
fn find_groups_in_range(groups: &[AtomGroup], lo: u64, hi: u64) -> Vec<usize> {
    let mut result = Vec::new();
    if lo > hi {
        return result;
    }
    let start = groups.partition_point(|g| g.base_id.0 + g.count <= lo);
    for gi in start..groups.len() {
        let g = &groups[gi];
        if g.base_id.0 > hi {
            break;
        }
        if g.count == 0 {
            continue;
        }
        let g_end = g.base_id.0 + g.count - 1;
        if g.base_id.0 <= hi && g_end >= lo {
            result.push(gi);
        }
    }
    result
}

// ─── Math helpers ────────────────────────────────────────────────────────────

fn reduce_extent(pos: i64, reduce_count: u64, reduce_stride: i64) -> (i64, i64) {
    if reduce_count <= 1 {
        return (pos, pos);
    }
    let ext = reduce_stride * (reduce_count as i64 - 1);
    (pos + ext.min(0), pos + ext.max(0))
}

fn reduce_extent_range(
    base_lo: i64,
    base_hi: i64,
    reduce_count: u64,
    reduce_stride: i64,
) -> (i64, i64) {
    if reduce_count <= 1 {
        return (base_lo, base_hi);
    }
    let ext = reduce_stride * (reduce_count as i64 - 1);
    (base_lo + ext.min(0), base_hi + ext.max(0))
}

/// Check if a range [need_off, need_end) within a group is fully covered
/// by the spans' coverage for that group.
fn is_range_covered_in_span(
    span_coverage: &HashMap<usize, Vec<(u64, u64)>>,
    group_idx: usize,
    need_off: u64,
    need_end: u64,
) -> bool {
    if need_off >= need_end {
        return true;
    }
    let Some(ranges) = span_coverage.get(&group_idx) else {
        return false;
    };
    // Check if sorted ranges cover [need_off, need_end).
    let mut sorted: Vec<(u64, u64)> = ranges.clone();
    sorted.sort_by_key(|&(off, _)| off);
    let mut covered_up_to = need_off;
    for &(off, count) in &sorted {
        if off > covered_up_to {
            return false;
        }
        let end = off + count;
        if end > covered_up_to {
            covered_up_to = end;
        }
        if covered_up_to >= need_end {
            return true;
        }
    }
    false
}

/// Merge overlapping or adjacent ranges per group.
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

// ─── Diagnostics ─────────────────────────────────────────────────────────────

impl SpanPlan {
    pub fn print_summary(&self) {
        let total_spans: usize = self.phases.iter().map(|p| p.spans.len()).sum();
        let non_empty: usize = self
            .phases
            .iter()
            .flat_map(|p| p.spans.iter())
            .filter(|s| s.graph.num_groups() > 0)
            .count();
        println!(
            "SpanPlan: {} lanes, {} phases, {} spans ({} non-empty)",
            self.num_lanes,
            self.phases.len(),
            total_spans,
            non_empty
        );
        for (pi, phase) in self.phases.iter().enumerate() {
            let active = phase
                .spans
                .iter()
                .filter(|s| s.graph.num_groups() > 0)
                .count();
            let max_atoms = phase
                .spans
                .iter()
                .map(|s| s.graph.num_atoms())
                .max()
                .unwrap_or(0);
            let min_atoms = phase
                .spans
                .iter()
                .filter(|s| s.graph.num_atoms() > 0)
                .map(|s| s.graph.num_atoms())
                .min()
                .unwrap_or(0);
            let balance = if min_atoms > 0 {
                format!("{:.1}x", max_atoms as f64 / min_atoms as f64)
            } else {
                "N/A".to_string()
            };
            println!(
                "  Phase {}: {} active lanes, balance {}",
                pi, active, balance
            );
        }
    }

    /// Validate span plan topology.
    pub fn validate(&self, original: &NanoGraph) -> Vec<String> {
        let mut errors = Vec::new();

        // Track which atoms are available (from literals and previous phases' outputs).
        let mut available: HashSet<AtomId> = HashSet::new();
        for group in original.groups() {
            if matches!(group.op, ScalarOp::Literal(_)) && group.inputs.is_empty() {
                for i in 0..group.count {
                    available.insert(group.base_id.offset(i));
                }
            }
        }

        for (phase_idx, phase) in self.phases.iter().enumerate() {
            // Validate each span's internal consistency.
            for (lane_idx, span) in phase.spans.iter().enumerate() {
                let graph_errors = span.graph.validate();
                for err in graph_errors {
                    errors.push(format!("Phase {} lane {}: {}", phase_idx, lane_idx, err));
                }

                // Check that span inputs are available.
                for mapping in &span.inputs {
                    for i in 0..mapping.count {
                        if !available.contains(&mapping.main_base.offset(i)) {
                            errors.push(format!(
                                "Phase {} lane {}: input atom {:?} not available",
                                phase_idx,
                                lane_idx,
                                mapping.main_base.offset(i)
                            ));
                            break; // Don't spam errors for every atom in the range.
                        }
                    }
                }
            }

            // Add this phase's outputs to available.
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

    // ─── Test graph builders ─────────────────────────────────────────────

    /// Simple matmul: C[M,N] = A[M,K] * B[K,N]
    fn build_matmul(m: u64, k: u64, n: u64) -> NanoGraph {
        let mut g = NanoGraph::new();
        let a_base = g.push_group(
            m * k,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let b_base = g.push_group(
            k * n,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let mut mul_bases = Vec::new();
        for row in 0..m {
            let a_row = AtomId(a_base.0 + row * k);
            let mul = g.push_group(
                k * n,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![
                    InputRef::StridedBroadcast {
                        base: a_row,
                        stride: 1,
                        repeat: n,
                    },
                    InputRef::Affine {
                        base: b_base,
                        stride: 1,
                    },
                ],
            );
            mul_bases.push(mul);
        }
        let mut reduce_bases = Vec::new();
        for row in 0..m {
            let red = g.push_group(
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
                    base: mul_bases[row as usize],
                    stride: 1,
                }],
            );
            reduce_bases.push(red);
        }
        for &rb in &reduce_bases {
            for i in 0..n {
                g.outputs.push(AtomId(rb.0 + i));
            }
        }
        g
    }

    /// Matmul chain: two matmuls with an activation in between.
    fn build_matmul_chain(m: u64, k1: u64, n1: u64, k2: u64, n2: u64) -> NanoGraph {
        let mut g = NanoGraph::new();
        let a1 = g.push_group(
            m * k1,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let b1 = g.push_group(
            k1 * n1,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        let mut mul1_bases = Vec::new();
        for row in 0..m {
            let a_row = AtomId(a1.0 + row * k1);
            let mul = g.push_group(
                k1 * n1,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![
                    InputRef::StridedBroadcast {
                        base: a_row,
                        stride: 1,
                        repeat: n1,
                    },
                    InputRef::Affine {
                        base: b1,
                        stride: 1,
                    },
                ],
            );
            mul1_bases.push(mul);
        }
        let mut red1_bases = Vec::new();
        for row in 0..m {
            let red = g.push_group(
                n1,
                ScalarOp::ReduceSum {
                    reduce_count: k1,
                    reduce_stride: n1 as i64,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![InputRef::Affine {
                    base: mul1_bases[row as usize],
                    stride: 1,
                }],
            );
            red1_bases.push(red);
        }

        // Activation: Tanh on the reduce output.
        let act = g.push_group(
            m * n1,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Tanh,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: red1_bases[0],
                stride: 1,
            }],
        );

        let b2 = g.push_group(
            n1 * n2,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        let mut mul2_bases = Vec::new();
        for row in 0..m {
            let mul = g.push_group(
                n1 * n2,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![
                    InputRef::StridedBroadcast {
                        base: AtomId(act.0 + row * n1),
                        stride: 1,
                        repeat: n2,
                    },
                    InputRef::Affine {
                        base: b2,
                        stride: 1,
                    },
                ],
            );
            mul2_bases.push(mul);
        }
        let mut red2_bases = Vec::new();
        for row in 0..m {
            let red = g.push_group(
                n2,
                ScalarOp::ReduceSum {
                    reduce_count: n1,
                    reduce_stride: n2 as i64,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![InputRef::Affine {
                    base: mul2_bases[row as usize],
                    stride: 1,
                }],
            );
            red2_bases.push(red);
        }
        for &rb in &red2_bases {
            for i in 0..n2 {
                g.outputs.push(AtomId(rb.0 + i));
            }
        }
        g
    }

    /// Simple elementwise: C[N] = A[N] + B[N]
    fn build_elementwise(count: u64) -> NanoGraph {
        let mut g = NanoGraph::new();
        let a = g.push_group(
            count,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let b = g.push_group(
            count,
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
            vec![],
        );
        let c = g.push_group(
            count,
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
        for i in 0..count {
            g.outputs.push(AtomId(c.0 + i));
        }
        g
    }

    /// Chain of elementwise ops: A → Tanh → Tanh → Tanh
    fn build_allrows_chain(count: u64) -> NanoGraph {
        let mut g = NanoGraph::new();
        let lit = g.push_group(
            count,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let a = g.push_group(
            count,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Tanh,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: lit,
                stride: 1,
            }],
        );
        let b = g.push_group(
            count,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Tanh,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: a, stride: 1 }],
        );
        let c = g.push_group(
            count,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Tanh,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: b, stride: 1 }],
        );
        for i in 0..count {
            g.outputs.push(AtomId(c.0 + i));
        }
        g
    }

    /// Cross-lane violation pattern: small Select split across lanes,
    /// downstream Modular reads ALL of it.
    fn build_cross_lane_pattern(small_count: u64, big_count: u64) -> NanoGraph {
        let mut g = NanoGraph::new();
        let lit_a = g.push_group(
            small_count,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let lit_b = g.push_group(
            small_count,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );
        let lit_cond = g.push_group(
            small_count,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        let sel = g.push_group(
            small_count,
            ScalarOp::Select {
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine {
                    base: lit_cond,
                    stride: 1,
                },
                InputRef::Affine {
                    base: lit_a,
                    stride: 1,
                },
                InputRef::Affine {
                    base: lit_b,
                    stride: 1,
                },
            ],
        );

        let lit_d = g.push_group(
            big_count,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let downstream = g.push_group(
            big_count,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::Modular {
                    base: sel,
                    stride: 1,
                    modulus: small_count,
                },
                InputRef::Affine {
                    base: lit_d,
                    stride: 1,
                },
            ],
        );

        for i in 0..big_count {
            g.outputs.push(AtomId(downstream.0 + i));
        }
        g
    }

    /// IndirectLoad pattern (Gather-like).
    fn build_indirect_load_pattern(num_indices: u64, d_total: u64, table_size: u64) -> NanoGraph {
        let mut g = NanoGraph::new();

        let indices = g.push_group(
            num_indices,
            ScalarOp::Literal(NumericScalar::I64(0)),
            vec![],
            vec![],
            vec![],
        );
        let stride_lit = g.push_atom(
            ScalarOp::Literal(NumericScalar::I64(d_total as i64)),
            vec![],
            vec![],
            vec![],
        );

        let out_count = num_indices * d_total;

        let mut mul_ids = Vec::with_capacity(out_count as usize);
        for flat in 0..out_count {
            let row = flat / d_total;
            mul_ids.push(indices.offset(row));
        }
        let mul_base = g.push_group(
            out_count,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::I64,
                output_dtype: DType::I64,
            },
            vec![],
            vec![],
            vec![InputRef::Explicit(mul_ids), InputRef::Broadcast(stride_lit)],
        );

        let col_offsets = g.push_group(
            out_count,
            ScalarOp::Literal(NumericScalar::I64(0)),
            vec![],
            vec![],
            vec![],
        );

        let add_base = g.push_group(
            out_count,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::I64,
                output_dtype: DType::I64,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine {
                    base: mul_base,
                    stride: 1,
                },
                InputRef::Affine {
                    base: col_offsets,
                    stride: 1,
                },
            ],
        );

        let data_table = g.push_group(
            table_size,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        let indirect = g.push_group(
            out_count,
            ScalarOp::IndirectLoad {
                table_base: data_table,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: add_base,
                stride: 1,
            }],
        );

        for i in 0..out_count {
            g.outputs.push(indirect.offset(i));
        }
        g
    }

    /// Multi-phase with literal: matmul → reduce → divide by literal.
    fn build_multi_phase_with_literals(m: u64, k: u64, n: u64) -> NanoGraph {
        let mut g = NanoGraph::new();
        let a_base = g.push_group(
            m * k,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let b_base = g.push_group(
            k * n,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        let mut mul_bases = Vec::new();
        for row in 0..m {
            let a_row = AtomId(a_base.0 + row * k);
            let mul = g.push_group(
                k * n,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![
                    InputRef::StridedBroadcast {
                        base: a_row,
                        stride: 1,
                        repeat: n,
                    },
                    InputRef::Affine {
                        base: b_base,
                        stride: 1,
                    },
                ],
            );
            mul_bases.push(mul);
        }
        let mut red_bases = Vec::new();
        for row in 0..m {
            let red = g.push_group(
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
                    base: mul_bases[row as usize],
                    stride: 1,
                }],
            );
            red_bases.push(red);
        }

        let divisor = g.push_atom(
            ScalarOp::Literal(NumericScalar::I64(42)),
            vec![],
            vec![],
            vec![],
        );

        let total_out = m * n;
        let div_result = g.push_group(
            total_out,
            ScalarOp::Binary {
                op: ScalarBinOp::Div,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine {
                    base: red_bases[0],
                    stride: 1,
                },
                InputRef::Broadcast(divisor),
            ],
        );

        for i in 0..total_out {
            g.outputs.push(div_result.offset(i));
        }
        g
    }

    /// ReduceSum that strides through adjacent literal groups.
    fn build_reduce_extends_into_literal(n: u64) -> NanoGraph {
        let mut g = NanoGraph::new();

        let data0 = g.push_group(
            n,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let _lit_adj = g.push_group(
            n,
            ScalarOp::Literal(NumericScalar::I64(42)),
            vec![],
            vec![],
            vec![],
        );
        let _data1 = g.push_group(
            n,
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
            vec![],
        );
        let _data2 = g.push_group(
            n,
            ScalarOp::Literal(NumericScalar::F32(3.0)),
            vec![],
            vec![],
            vec![],
        );

        let reduce_out = g.push_group(
            n,
            ScalarOp::ReduceSum {
                reduce_count: 4,
                reduce_stride: n as i64,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: data0,
                stride: 1,
            }],
        );

        for i in 0..n {
            g.outputs.push(reduce_out.offset(i));
        }
        g
    }

    /// Build a graph with Broadcast dependency (small scalar used by large group).
    fn build_broadcast_dependency() -> NanoGraph {
        let mut g = NanoGraph::new();

        // A small compute group (not literal) that produces a scalar.
        let lit_x = g.push_atom(
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
            vec![],
        );
        let lit_y = g.push_atom(
            ScalarOp::Literal(NumericScalar::F32(3.0)),
            vec![],
            vec![],
            vec![],
        );
        // scalar = x + y
        let scalar = g.push_atom(
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Broadcast(lit_x), InputRef::Broadcast(lit_y)],
        );

        // Large group that broadcasts scalar.
        let lit_data = g.push_group(
            1024,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let result = g.push_group(
            1024,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine {
                    base: lit_data,
                    stride: 1,
                },
                InputRef::Broadcast(scalar),
            ],
        );

        for i in 0..1024 {
            g.outputs.push(result.offset(i));
        }
        g
    }

    /// Diamond dependency: A → B, A → C, B+C → D
    fn build_diamond(count: u64) -> NanoGraph {
        let mut g = NanoGraph::new();
        let lit = g.push_group(
            count,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let a = g.push_group(
            count,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Tanh,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: lit,
                stride: 1,
            }],
        );
        let b = g.push_group(
            count,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: a, stride: 1 }],
        );
        let c = g.push_group(
            count,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: a, stride: 1 }],
        );
        let d = g.push_group(
            count,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine { base: b, stride: 1 },
                InputRef::Affine { base: c, stride: 1 },
            ],
        );
        for i in 0..count {
            g.outputs.push(d.offset(i));
        }
        g
    }

    // ─── Verification helpers ────────────────────────────────────────────

    fn verify_span_plan(graph: &NanoGraph, plan: &SpanPlan) {
        let errors = plan.validate(graph);
        assert!(
            errors.is_empty(),
            "Span plan errors:\n{}",
            errors.join("\n")
        );
    }

    fn verify_output_coverage(graph: &NanoGraph, plan: &SpanPlan) {
        let groups = graph.groups();
        let is_literal: Vec<bool> = groups
            .iter()
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
            if is_literal[gi] {
                continue;
            }
            for offset in 0..group.count {
                let atom = AtomId(group.base_id.0 + offset);
                let count = produced.get(&atom).copied().unwrap_or(0);
                assert_eq!(
                    count, 1,
                    "Atom {:?} (group {}, offset {}) appears {} times in outputs",
                    atom, gi, offset, count
                );
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
                    for i in 0..mapping.count {
                        assert!(
                            available.contains(&mapping.main_base.offset(i)),
                            "Phase {} lane {}: input atom {:?} not available",
                            phase_idx,
                            lane_idx,
                            mapping.main_base.offset(i)
                        );
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
                        assert!(
                            in_lo >= out_hi || out_lo >= in_hi,
                            "Phase {} lane {}: input [{}, {}) overlaps output [{}, {})",
                            phase_idx,
                            lane_idx,
                            in_lo,
                            in_hi,
                            out_lo,
                            out_hi
                        );
                    }
                }
            }
        }
    }

    /// Validate that every non-literal group in every span has all its
    /// InputRef targets present in the span graph (including reduce-extended atoms).
    fn validate_all_spans_self_contained(plan: &SpanPlan) {
        for (phase_idx, phase) in plan.phases.iter().enumerate() {
            for (lane_idx, span) in phase.spans.iter().enumerate() {
                let groups = span.graph.groups();
                for (gi, group) in groups.iter().enumerate() {
                    if matches!(group.op, ScalarOp::Literal(_)) && group.inputs.is_empty() {
                        continue;
                    }

                    let (reduce_count, reduce_stride) = match &group.op {
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
                            (*reduce_count, *reduce_stride)
                        }
                        _ => (1, 0),
                    };

                    for (inp_idx, input) in group.inputs.iter().enumerate() {
                        for i in 0..group.count {
                            let base = input.resolve(i, 0);
                            assert!(
                                span.graph.contains_atom(base),
                                "Phase {} lane {} group {} (base={}, op={:?}) input {} offset {}: \
                                 references atom {} which doesn't exist in span graph",
                                phase_idx,
                                lane_idx,
                                gi,
                                group.base_id,
                                group.op,
                                inp_idx,
                                i,
                                base,
                            );
                            for k in 1..reduce_count {
                                let ext_atom =
                                    AtomId((base.0 as i64 + k as i64 * reduce_stride) as u64);
                                assert!(
                                    span.graph.contains_atom(ext_atom),
                                    "Phase {} lane {} group {} (base={}, op={:?}) input {} offset {} \
                                     reduce step k={}: atom {} not in span graph",
                                    phase_idx,
                                    lane_idx,
                                    gi,
                                    group.base_id,
                                    group.op,
                                    inp_idx,
                                    i,
                                    k,
                                    ext_atom,
                                );
                            }
                        }
                    }
                    if let ScalarOp::IndirectLoad { table_base, .. } = &group.op {
                        assert!(
                            span.graph.contains_atom(*table_base),
                            "Phase {} lane {} group {} (IndirectLoad): table_base {} not in span",
                            phase_idx,
                            lane_idx,
                            gi,
                            table_base,
                        );
                    }
                }
            }
        }
    }

    // ─── Test cases ──────────────────────────────────────────────────────

    #[test]
    fn test_empty_graph() {
        let g = NanoGraph::new();
        let plan = plan_spans(&g, 4);
        assert_eq!(plan.num_lanes, 4);
        assert!(plan.phases.is_empty());
    }

    #[test]
    fn test_single_lane_matmul() {
        let g = build_matmul(4, 8, 4);
        assert!(g.validate().is_empty());
        let plan = plan_spans(&g, 1);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);
        validate_all_spans_self_contained(&plan);
    }

    #[test]
    fn test_matmul_multi_lane() {
        let g = build_matmul(8, 4, 4);
        assert!(g.validate().is_empty());
        let plan = plan_spans(&g, 4);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);
        verify_input_availability(&plan, &g);
        verify_phase_independence(&plan);
        validate_all_spans_self_contained(&plan);
    }

    #[test]
    fn test_matmul_chain() {
        let g = build_matmul_chain(4, 4, 4, 4, 4);
        assert!(g.validate().is_empty());
        let plan = plan_spans(&g, 4);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);
        verify_input_availability(&plan, &g);
        verify_phase_independence(&plan);
        validate_all_spans_self_contained(&plan);
    }

    #[test]
    fn test_elementwise() {
        for count in [1, 7, 8, 15, 16, 100, 1000] {
            let g = build_elementwise(count);
            let plan = plan_spans(&g, 4);
            verify_span_plan(&g, &plan);
            verify_output_coverage(&g, &plan);
            validate_all_spans_self_contained(&plan);
        }
    }

    #[test]
    fn test_allrows_chain() {
        let g = build_allrows_chain(256);
        assert!(g.validate().is_empty());
        let plan = plan_spans(&g, 4);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);
        verify_input_availability(&plan, &g);
        verify_phase_independence(&plan);
        validate_all_spans_self_contained(&plan);
    }

    #[test]
    fn test_cross_lane_violation_pattern() {
        // The GPT-2 attention mask pattern: small Select split across lanes,
        // downstream Modular reads ALL of it.
        let g = build_cross_lane_pattern(3072, 49152);
        assert!(g.validate().is_empty());
        let plan = plan_spans(&g, 8);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);
        verify_input_availability(&plan, &g);
        verify_phase_independence(&plan);
        validate_all_spans_self_contained(&plan);
        plan.print_summary();
    }

    #[test]
    fn test_indirect_load_single_lane() {
        let g = build_indirect_load_pattern(4, 8, 256);
        assert!(g.validate().is_empty());
        let plan = plan_spans(&g, 1);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);
        validate_all_spans_self_contained(&plan);
    }

    #[test]
    fn test_indirect_load_multi_lane() {
        let g = build_indirect_load_pattern(8, 4, 128);
        assert!(g.validate().is_empty());
        let plan = plan_spans(&g, 4);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);
        verify_input_availability(&plan, &g);
        verify_phase_independence(&plan);
        validate_all_spans_self_contained(&plan);
    }

    #[test]
    fn test_indirect_load_large_table() {
        let g = build_indirect_load_pattern(4, 8, 2048);
        assert!(g.validate().is_empty());
        let plan = plan_spans(&g, 4);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);
        verify_input_availability(&plan, &g);
        verify_phase_independence(&plan);
        validate_all_spans_self_contained(&plan);
    }

    #[test]
    fn test_indirect_load_gpt2_like() {
        let g = build_indirect_load_pattern(16, 128, 4096);
        assert!(g.validate().is_empty());
        let plan = plan_spans(&g, 8);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);
        verify_input_availability(&plan, &g);
        verify_phase_independence(&plan);
        validate_all_spans_self_contained(&plan);
    }

    #[test]
    fn test_multi_phase_literal_dependency() {
        let g = build_multi_phase_with_literals(8, 4, 4);
        assert!(g.validate().is_empty());
        let plan = plan_spans(&g, 4);
        plan.print_summary();
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);
        verify_input_availability(&plan, &g);
        verify_phase_independence(&plan);
        validate_all_spans_self_contained(&plan);
    }

    #[test]
    fn test_reduce_stride_into_adjacent_literal() {
        let g = build_reduce_extends_into_literal(8);
        assert!(g.validate().is_empty());
        let plan = plan_spans(&g, 2);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);
        validate_all_spans_self_contained(&plan);
    }

    #[test]
    fn test_broadcast_dependency() {
        let g = build_broadcast_dependency();
        assert!(g.validate().is_empty());
        let plan = plan_spans(&g, 4);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);
        verify_input_availability(&plan, &g);
        verify_phase_independence(&plan);
        validate_all_spans_self_contained(&plan);
    }

    #[test]
    fn test_diamond_dependency() {
        let g = build_diamond(256);
        assert!(g.validate().is_empty());
        let plan = plan_spans(&g, 4);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);
        verify_input_availability(&plan, &g);
        verify_phase_independence(&plan);
        validate_all_spans_self_contained(&plan);
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
            ("cross_lane", build_cross_lane_pattern(3072, 49152), 8),
            ("broadcast_dep", build_broadcast_dependency(), 4),
            ("diamond_256", build_diamond(256), 4),
            ("diamond_1024", build_diamond(1024), 8),
        ];

        for (name, graph, lanes) in configs {
            assert!(
                graph.validate().is_empty(),
                "{}: graph validation failed",
                name
            );
            let plan = plan_spans(&graph, lanes);
            verify_span_plan(&graph, &plan);
            verify_output_coverage(&graph, &plan);
            verify_input_availability(&plan, &graph);
            verify_phase_independence(&plan);
            validate_all_spans_self_contained(&plan);
            println!(
                "  {}: {} phases, {} lanes",
                name,
                plan.phases.len(),
                plan.num_lanes
            );
        }
    }

    #[test]
    fn test_larger_matmul_chain() {
        let g = build_matmul_chain(8, 16, 16, 16, 16);
        assert!(g.validate().is_empty());
        let plan = plan_spans(&g, 4);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);
        verify_input_availability(&plan, &g);
        verify_phase_independence(&plan);
        validate_all_spans_self_contained(&plan);
        assert!(
            plan.phases.len() <= 10,
            "Should have <= 10 phases, got {}",
            plan.phases.len()
        );
    }

    /// Test with a single non-literal group.
    #[test]
    fn test_single_compute_group() {
        let mut g = NanoGraph::new();
        let lit = g.push_group(
            16,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let comp = g.push_group(
            16,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Tanh,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: lit,
                stride: 1,
            }],
        );
        for i in 0..16 {
            g.outputs.push(comp.offset(i));
        }
        let plan = plan_spans(&g, 4);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);
        validate_all_spans_self_contained(&plan);
    }

    /// Test with only literals (no compute).
    #[test]
    fn test_all_literals() {
        let mut g = NanoGraph::new();
        g.push_group(
            100,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let plan = plan_spans(&g, 4);
        assert!(plan.phases.is_empty());
    }

    /// Matmul where M < num_lanes (some lanes will be idle).
    #[test]
    fn test_matmul_fewer_rows_than_lanes() {
        let g = build_matmul(2, 4, 4);
        assert!(g.validate().is_empty());
        let plan = plan_spans(&g, 8);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);
        verify_input_availability(&plan, &g);
        verify_phase_independence(&plan);
        validate_all_spans_self_contained(&plan);
    }

    /// Stress test: moderately large matmul with many lanes.
    #[test]
    fn test_matmul_stress() {
        let g = build_matmul(16, 32, 16);
        assert!(g.validate().is_empty());
        let plan = plan_spans(&g, 8);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);
        verify_input_availability(&plan, &g);
        verify_phase_independence(&plan);
        validate_all_spans_self_contained(&plan);
    }

    /// Test with Modular InputRef to ensure AllRows classification works.
    #[test]
    fn test_modular_inputref_classification() {
        let mut g = NanoGraph::new();
        let bias = g.push_group(
            64,
            ScalarOp::Literal(NumericScalar::F32(0.5)),
            vec![],
            vec![],
            vec![],
        );
        let data = g.push_group(
            256,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        // Add data + bias (tiled via Modular).
        let result = g.push_group(
            256,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine {
                    base: data,
                    stride: 1,
                },
                InputRef::Modular {
                    base: bias,
                    stride: 1,
                    modulus: 64,
                },
            ],
        );
        for i in 0..256 {
            g.outputs.push(result.offset(i));
        }
        let plan = plan_spans(&g, 4);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);
        validate_all_spans_self_contained(&plan);
    }

    /// Chain of ops where a middle op uses Broadcast from a computed scalar.
    #[test]
    fn test_computed_scalar_broadcast_chain() {
        let mut g = NanoGraph::new();

        // Step 1: Compute a scalar (ReduceSum of 16 elements → 1 element).
        let input_data = g.push_group(
            16,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let sum_scalar = g.push_group(
            1,
            ScalarOp::ReduceSum {
                reduce_count: 16,
                reduce_stride: 1,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: input_data,
                stride: 1,
            }],
        );

        // Step 2: Large group broadcasts from the computed scalar.
        let big_data = g.push_group(
            512,
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
            vec![],
        );
        let result = g.push_group(
            512,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine {
                    base: big_data,
                    stride: 1,
                },
                InputRef::Broadcast(sum_scalar),
            ],
        );

        for i in 0..512 {
            g.outputs.push(result.offset(i));
        }

        assert!(g.validate().is_empty());
        let plan = plan_spans(&g, 4);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);
        verify_input_availability(&plan, &g);
        verify_phase_independence(&plan);
        validate_all_spans_self_contained(&plan);
    }
}
