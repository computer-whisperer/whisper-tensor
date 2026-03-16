#![allow(
    clippy::all,
    dead_code,
    unreachable_patterns,
    unused_variables,
    unused_imports
)]
//! Span-based partitioner v4c: Closure-first span construction.
//!
//! Previous attempts (v3a-v3d) all tried variants of "assign groups to lanes/phases,
//! then fix violations." They failed because span NanoGraph construction didn't
//! transitively resolve all dependencies of duplicated groups.
//!
//! ## Approach: Build spans by closure, then schedule
//!
//! 1. **Group-level dependency DAG**: Build producer map for every group.
//!    O(groups × inputs_per_group). No per-atom iteration.
//!
//! 2. **Topological sort**: Determine valid execution ordering.
//!
//! 3. **Phase assignment via depth**: Groups at the same "depth" (longest
//!    path from a source) can potentially run in parallel. A phase boundary
//!    is inserted when a group requires ALL atoms from a split producer
//!    (cross-lane dependency).
//!
//! 4. **Lane assignment**: Within each phase, split large groups across lanes.
//!    Small groups that are consumed by multiple lanes get duplicated.
//!
//! 5. **Closure computation**: For each (phase, lane) span, compute the
//!    transitive closure of group dependencies. Every group in the closure
//!    must either be:
//!    - In this span (compute or duplicated literal)
//!    - Declared as an external input
//!
//! 6. **Span NanoGraph construction**: Build self-contained NanoGraphs from
//!    the closed group sets, remapping InputRefs.
//!
//! All operations are O(groups). No per-atom data structures.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use crate::nano_graph::{AtomGroup, AtomId, InputRef, NanoGraph, ScalarOp};

/// Literal groups with fewer atoms than this are duplicated into spans.
const LITERAL_INLINE_THRESHOLD: u64 = 1024;

/// Compute groups with fewer atoms than this threshold may be duplicated
/// across lanes to avoid introducing a phase boundary.
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

    // Step 1: Classify groups.
    let is_literal: Vec<bool> = groups
        .iter()
        .map(|g| matches!(g.op, ScalarOp::Literal(_)) && g.inputs.is_empty())
        .collect();

    let compute_indices: Vec<usize> = (0..n).filter(|&i| !is_literal[i]).collect();
    if compute_indices.is_empty() {
        return SpanPlan {
            num_lanes,
            phases: vec![],
        };
    }

    // Step 2: Build group-level dependency DAG.
    let producers = build_group_producers(groups, &is_literal);

    // Step 3: Topological sort.
    let topo_order = topological_sort(n, &producers, &is_literal);

    // Step 4: Compute "depth" for each group (longest path from any source).
    // This establishes a partial ordering that respects all dependencies.
    let depths = compute_depths(&topo_order, &producers, &is_literal, n);

    // Step 5: Determine which groups are "full-width consumers" — they need
    // ALL atoms from a producer that will be split across lanes. These create
    // mandatory phase boundaries: the producer must complete on all lanes
    // before the consumer can start.
    //
    // Strategy: find groups where at least one InputRef spans the full range
    // of a splittable producer. "Splittable" means the producer will be split
    // across lanes (it's a compute group with enough atoms to warrant splitting,
    // and it has lane-aligned access patterns).
    let phase_assignments = assign_phases(
        &topo_order,
        &producers,
        groups,
        &is_literal,
        &depths,
        num_lanes,
    );

    let num_phases = phase_assignments.iter().copied().max().unwrap_or(0) + 1;

    // Step 6: Assign groups to lanes within each phase.
    let lane_assignments = assign_lanes(
        groups,
        &is_literal,
        &producers,
        &phase_assignments,
        num_phases,
        num_lanes,
        &topo_order,
    );

    // Step 7: Build span NanoGraphs with transitive closure.
    build_span_plan(
        graph,
        groups,
        &is_literal,
        &producers,
        &lane_assignments,
        &phase_assignments,
        num_phases,
        num_lanes,
    )
}

// ─── Group-level dependency DAG ──────────────────────────────────────────────

/// Build producer list for each group. Returns producers[gi] = sorted list of
/// compute (non-literal) group indices that gi depends on.
fn build_group_producers(groups: &[AtomGroup], is_literal: &[bool]) -> Vec<Vec<usize>> {
    let n = groups.len();
    let mut producers: Vec<Vec<usize>> = Vec::with_capacity(n);

    for (gi, group) in groups.iter().enumerate() {
        let mut prod_set = BTreeSet::new();

        // Resolve InputRef targets to producer groups.
        for input in &group.inputs {
            for pi in resolve_producer_groups(input, group.count, groups) {
                if pi != gi && !is_literal[pi] {
                    prod_set.insert(pi);
                }
            }
        }

        // ReduceSum/ReduceMax: strided access extends beyond the InputRef base range.
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
                        if pi != gi && !is_literal[pi] {
                            prod_set.insert(pi);
                        }
                    }
                }
            }
            _ => {}
        }

        // IndirectLoad: table_base references a group.
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

// ─── Topological sort ────────────────────────────────────────────────────────

fn topological_sort(n: usize, producers: &[Vec<usize>], is_literal: &[bool]) -> Vec<usize> {
    let mut in_degree = vec![0u32; n];
    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); n];

    for gi in 0..n {
        if is_literal[gi] {
            continue;
        }
        for &pi in &producers[gi] {
            consumers[pi].push(gi);
            in_degree[gi] += 1;
        }
    }

    let mut queue: VecDeque<usize> = VecDeque::new();
    for gi in 0..n {
        if !is_literal[gi] && in_degree[gi] == 0 {
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

// ─── Depth computation ───────────────────────────────────────────────────────

fn compute_depths(
    topo_order: &[usize],
    producers: &[Vec<usize>],
    is_literal: &[bool],
    n: usize,
) -> Vec<usize> {
    let mut depths = vec![0usize; n];
    for &gi in topo_order {
        if is_literal[gi] {
            continue;
        }
        let max_prod_depth = producers[gi]
            .iter()
            .map(|&pi| depths[pi] + 1)
            .max()
            .unwrap_or(0);
        depths[gi] = max_prod_depth;
    }
    depths
}

// ─── Phase assignment ────────────────────────────────────────────────────────

/// Assign each compute group to a phase.
///
/// The key insight: a phase boundary is needed when a group consumes the FULL
/// output of a group that would be split across lanes. Because each lane only
/// produces its slice, the full output isn't available until all lanes finish
/// (barrier).
///
/// We detect this by checking: for group G reading from producer P, does G's
/// InputRef cover P's full atom range? If so, and P would be split across lanes,
/// G must be in a later phase than P.
///
/// Groups that only read lane-local slices (e.g., elementwise chains where
/// atom i reads atom i) can be in the same phase as their producer.
fn assign_phases(
    topo_order: &[usize],
    producers: &[Vec<usize>],
    groups: &[AtomGroup],
    is_literal: &[bool],
    depths: &[usize],
    num_lanes: usize,
) -> Vec<usize> {
    let n = groups.len();
    let mut phase = vec![0usize; n];

    // Determine which groups would be split across lanes.
    let is_splittable: Vec<bool> = groups
        .iter()
        .enumerate()
        .map(|(gi, g)| !is_literal[gi] && g.count > 1 && would_split(g, num_lanes))
        .collect();

    for &gi in topo_order {
        if is_literal[gi] {
            continue;
        }
        let group = &groups[gi];

        // My phase must be >= max(producer phases).
        // If I need the FULL range of a splittable producer, I need phase > producer's phase.
        let mut min_phase = 0usize;

        for &pi in &producers[gi] {
            let prod = &groups[pi];
            let needs_barrier = is_splittable[pi] && needs_full_producer(group, prod, groups);

            if needs_barrier {
                min_phase = min_phase.max(phase[pi] + 1);
            } else {
                min_phase = min_phase.max(phase[pi]);
            }
        }

        phase[gi] = min_phase;
    }

    phase
}

/// Would this group be split across lanes?
fn would_split(group: &AtomGroup, num_lanes: usize) -> bool {
    // Groups with very few atoms won't be split.
    if group.count <= num_lanes as u64 {
        return false;
    }
    // Reduce groups can split along their output dimension (count).
    // Elementwise groups can split along count.
    // Broadcast-only groups (count=1) don't split.
    true
}

/// Does group `consumer` need the FULL atom range of `producer`?
///
/// Returns true if any InputRef in consumer covers the entire producer range,
/// meaning lane-local slices of the producer aren't sufficient.
fn needs_full_producer(consumer: &AtomGroup, producer: &AtomGroup, groups: &[AtomGroup]) -> bool {
    let prod_base = producer.base_id.0;
    let prod_end = prod_base + producer.count;

    for input in &consumer.inputs {
        let (ref_lo, ref_hi) = input_ref_range(input, consumer.count, &consumer.op);
        // If the InputRef's range encompasses the full producer range, we need
        // all of the producer's atoms.
        if ref_lo <= prod_base && ref_hi >= prod_end {
            return true;
        }
    }

    // Also check if a Broadcast/Modular ref touches the producer — these
    // inherently need the full producer.
    for input in &consumer.inputs {
        match input {
            InputRef::Broadcast(id) if producer.contains(*id) => return true,
            InputRef::Modular { base, modulus, .. } => {
                // Modular wraps around, so it accesses the full modulus range.
                if producer.contains(*base) && *modulus >= producer.count {
                    return true;
                }
            }
            _ => {}
        }
    }

    false
}

/// Compute the atom ID range [lo, hi) that an InputRef accesses.
/// Includes reduce stride extensions if the op is a reduce.
fn input_ref_range(input: &InputRef, count: u64, op: &ScalarOp) -> (u64, u64) {
    if count == 0 {
        return (u64::MAX, 0);
    }

    let (reduce_count, reduce_stride) = match op {
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
        _ => (1, 0i64),
    };

    let base_range = match input {
        InputRef::Broadcast(id) => (id.0, id.0 + 1),
        InputRef::Affine { base, stride } => {
            let last = base.0 as i64 + *stride as i64 * (count as i64 - 1);
            let lo = (base.0 as i64).min(last) as u64;
            let hi = (base.0 as i64).max(last) as u64 + 1;
            (lo, hi)
        }
        InputRef::Explicit(ids) => {
            let lo = ids.iter().map(|id| id.0).min().unwrap_or(0);
            let hi = ids.iter().map(|id| id.0).max().unwrap_or(0) + 1;
            (lo, hi)
        }
        InputRef::SymAffine { base, stride_i, .. } => {
            let last = base.0 as i64 + *stride_i as i64 * (count as i64 - 1);
            let lo = (base.0 as i64).min(last) as u64;
            let hi = (base.0 as i64).max(last) as u64 + 1;
            (lo, hi)
        }
        InputRef::StridedBroadcast {
            base,
            stride,
            repeat,
        } => {
            let num_blocks = (count + repeat - 1) / repeat;
            let last = base.0 as i64 + *stride * (num_blocks as i64 - 1);
            let lo = (base.0 as i64).min(last) as u64;
            let hi = (base.0 as i64).max(last) as u64 + 1;
            (lo, hi)
        }
        InputRef::Modular {
            base,
            stride,
            modulus,
        } => {
            if *modulus == 0 {
                return (u64::MAX, 0);
            }
            let last = base.0 as i64 + *stride as i64 * (*modulus as i64 - 1);
            let lo = (base.0 as i64).min(last) as u64;
            let hi = (base.0 as i64).max(last) as u64 + 1;
            (lo, hi)
        }
    };

    // Extend for reduce stride.
    if reduce_count > 1 {
        let ext = reduce_stride * (reduce_count as i64 - 1);
        let ext_lo = ext.min(0);
        let ext_hi = ext.max(0);
        (
            (base_range.0 as i64 + ext_lo) as u64,
            (base_range.1 as i64 + ext_hi) as u64,
        )
    } else {
        base_range
    }
}

// ─── Lane assignment ─────────────────────────────────────────────────────────

/// A work assignment: which slice of a group goes to which lane.
#[derive(Debug, Clone)]
struct WorkAssignment {
    group_idx: usize,
    lane: usize,
    atom_offset: u64,
    atom_count: u64,
    /// True if this is a duplicated copy (not the "owner" for output purposes).
    is_duplicate: bool,
}

/// Assign groups to lanes within each phase.
fn assign_lanes(
    groups: &[AtomGroup],
    is_literal: &[bool],
    producers: &[Vec<usize>],
    phase_assignments: &[usize],
    num_phases: usize,
    num_lanes: usize,
    topo_order: &[usize],
) -> Vec<WorkAssignment> {
    let n = groups.len();
    let mut assignments: Vec<WorkAssignment> = Vec::new();

    // Organize groups by phase (in topo order within each phase).
    let mut phase_groups: Vec<Vec<usize>> = vec![Vec::new(); num_phases];
    for &gi in topo_order {
        if !is_literal[gi] {
            phase_groups[phase_assignments[gi]].push(gi);
        }
    }

    // For each group, check if it can be lane-aligned with a producer.
    // Lane-aligned means: if producer P is split into N slices, consumer C
    // can also be split such that C's slice i only reads P's slice i.
    //
    // This is true for Affine(stride=1) chains and ReduceSum where the
    // input Mul groups are row-indexed.

    // Track which lane each group (or slice) ends up on.
    // For split groups: group_lane_info[gi] = None (split across lanes)
    // For unsplit groups: group_lane_info[gi] = Some(lane)
    let mut group_lane_info: Vec<Option<usize>> = vec![None; n];

    for phase_idx in 0..num_phases {
        for &gi in &phase_groups[phase_idx] {
            let group = &groups[gi];

            // Check if this group should be split.
            if group.count > num_lanes as u64 && can_split_evenly(group, num_lanes) {
                // Split across lanes.
                let chunk = (group.count + num_lanes as u64 - 1) / num_lanes as u64;
                for lane in 0..num_lanes {
                    let offset = lane as u64 * chunk;
                    let count = chunk.min(group.count.saturating_sub(offset));
                    if count == 0 {
                        continue;
                    }
                    assignments.push(WorkAssignment {
                        group_idx: gi,
                        lane,
                        atom_offset: offset,
                        atom_count: count,
                        is_duplicate: false,
                    });
                }
                // Mark as split.
                group_lane_info[gi] = None;
            } else if group.count <= DUPLICATION_THRESHOLD {
                // Small enough to duplicate if needed.
                // For now, assign to lane 0 unless a producer gives us a hint.
                let preferred_lane =
                    find_preferred_lane(gi, producers, &group_lane_info, num_lanes);
                assignments.push(WorkAssignment {
                    group_idx: gi,
                    lane: preferred_lane,
                    atom_offset: 0,
                    atom_count: group.count,
                    is_duplicate: false,
                });
                group_lane_info[gi] = Some(preferred_lane);
            } else {
                // Large, unsplittable group. Assign to the least-loaded lane.
                let preferred_lane =
                    find_preferred_lane(gi, producers, &group_lane_info, num_lanes);
                assignments.push(WorkAssignment {
                    group_idx: gi,
                    lane: preferred_lane,
                    atom_offset: 0,
                    atom_count: group.count,
                    is_duplicate: false,
                });
                group_lane_info[gi] = Some(preferred_lane);
            }
        }
    }

    // Now: for each span (phase, lane), compute the closure and add duplications.
    // A span needs access to every producer of every group it contains.
    // If producer P is in the same phase on a different lane, we have a violation.
    // Resolution: duplicate P into this lane (if small) or bump to later phase
    // (already handled in phase assignment).
    //
    // The remaining violations are: a group assigned to one lane reads from a
    // group that's ALSO assigned to one lane (not split), but on a different lane
    // in the SAME phase. This happens when phase assignment missed the dependency
    // because the producer wasn't "splittable" but ended up on a different lane.
    fix_same_phase_violations(
        &mut assignments,
        groups,
        is_literal,
        producers,
        phase_assignments,
        &group_lane_info,
        num_lanes,
    );

    assignments
}

/// Check if a group can be split evenly across lanes.
fn can_split_evenly(group: &AtomGroup, num_lanes: usize) -> bool {
    // ReduceSum/ReduceMax with reduce_stride: can split along the output
    // dimension (count), but each output atom's reduction is self-contained.
    // Elementwise: trivially splittable.
    // Groups with Explicit InputRefs: harder to split (but still possible
    // by slicing the explicit vec).
    true
}

/// Find a preferred lane for a group based on its producers.
fn find_preferred_lane(
    gi: usize,
    producers: &[Vec<usize>],
    group_lane_info: &[Option<usize>],
    num_lanes: usize,
) -> usize {
    // If any non-split producer is on a specific lane, prefer that lane.
    for &pi in &producers[gi] {
        if let Some(lane) = group_lane_info[pi] {
            return lane;
        }
    }
    // Default to lane 0.
    0
}

/// Fix violations where a group reads from another group on a different lane
/// in the same phase. Resolution: duplicate the producer into the consumer's lane.
fn fix_same_phase_violations(
    assignments: &mut Vec<WorkAssignment>,
    groups: &[AtomGroup],
    is_literal: &[bool],
    producers: &[Vec<usize>],
    phase_assignments: &[usize],
    group_lane_info: &[Option<usize>],
    num_lanes: usize,
) {
    // Build index: (group_idx) -> list of (assignment_idx, lane)
    // For a split group, there are multiple entries (one per lane).
    let mut group_to_assignments: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (ai, a) in assignments.iter().enumerate() {
        group_to_assignments
            .entry(a.group_idx)
            .or_default()
            .push((ai, a.lane));
    }

    let max_iterations = 50;
    for _iter in 0..max_iterations {
        // Rebuild index after modifications.
        group_to_assignments.clear();
        for (ai, a) in assignments.iter().enumerate() {
            group_to_assignments
                .entry(a.group_idx)
                .or_default()
                .push((ai, a.lane));
        }

        // For each assignment, check if all its producers are available on
        // the same lane (same or earlier phase) or are literals.
        let mut duplications: Vec<WorkAssignment> = Vec::new();
        let mut found_violation = false;

        for a in assignments.iter() {
            let gi = a.group_idx;
            let my_lane = a.lane;
            let my_phase = phase_assignments[gi];

            for &pi in &producers[gi] {
                let prod_phase = phase_assignments[pi];

                if prod_phase < my_phase {
                    // Producer is in an earlier phase — always available after barrier.
                    continue;
                }

                // Same phase: producer must be on the same lane.
                if let Some(prod_assignments) = group_to_assignments.get(&pi) {
                    let on_my_lane = prod_assignments.iter().any(|&(_, lane)| lane == my_lane);
                    if !on_my_lane {
                        // Violation! Producer pi is not on my lane in the same phase.
                        // Duplicate it onto my lane.
                        let already_queued = duplications
                            .iter()
                            .any(|d| d.group_idx == pi && d.lane == my_lane);
                        if !already_queued {
                            duplications.push(WorkAssignment {
                                group_idx: pi,
                                lane: my_lane,
                                atom_offset: 0,
                                atom_count: groups[pi].count,
                                is_duplicate: true,
                            });
                            found_violation = true;
                        }
                    }
                }
            }
        }

        if !found_violation {
            break;
        }

        // Add duplications and iterate (duplicated groups may have their own
        // producers that also need to be on this lane).
        assignments.extend(duplications);
    }
}

// ─── Span NanoGraph construction ─────────────────────────────────────────────

/// Build the full SpanPlan from work assignments.
fn build_span_plan(
    main_graph: &NanoGraph,
    groups: &[AtomGroup],
    is_literal: &[bool],
    producers: &[Vec<usize>],
    assignments: &[WorkAssignment],
    phase_assignments: &[usize],
    num_phases: usize,
    num_lanes: usize,
) -> SpanPlan {
    // Organize assignments by (phase, lane).
    let mut phase_lane_work: Vec<Vec<Vec<&WorkAssignment>>> =
        vec![vec![Vec::new(); num_lanes]; num_phases];
    for a in assignments {
        let phase = phase_assignments[a.group_idx];
        phase_lane_work[phase][a.lane].push(a);
    }

    // Sort each lane's work by group_idx for topo ordering within the span.
    for phase in &mut phase_lane_work {
        for lane in phase {
            lane.sort_by_key(|a| (a.group_idx, a.atom_offset));
            // Dedup: if a group appears multiple times on the same lane
            // (e.g., original assignment + duplication), keep only one.
            lane.dedup_by_key(|a| (a.group_idx, a.atom_offset));
        }
    }

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

            let span = build_single_span(
                main_graph,
                groups,
                is_literal,
                producers,
                lane_work,
                assignments,
                phase_idx,
                lane_idx,
                phase_assignments,
                num_lanes,
            );
            spans.push(span);
        }

        phases.push(Phase { spans });
    }

    SpanPlan { num_lanes, phases }
}

/// Build a single span NanoGraph for one (phase, lane).
///
/// This is the critical function. It must ensure the span is SELF-CONTAINED:
/// every InputRef in the span either resolves to an atom within the span
/// or to a declared external input.
fn build_single_span(
    main_graph: &NanoGraph,
    groups: &[AtomGroup],
    is_literal: &[bool],
    producers: &[Vec<usize>],
    lane_work: &[&WorkAssignment],
    all_assignments: &[WorkAssignment],
    phase_idx: usize,
    lane_idx: usize,
    phase_assignments: &[usize],
    num_lanes: usize,
) -> Span {
    let mut span_graph = NanoGraph::new();

    // Copy sym_dim configuration from main graph.
    for (name, &sd) in &main_graph.sym_dim_names {
        let local_sd = span_graph.sym_dim(name);
        if let Some(&bound) = main_graph.sym_dim_bounds.get(&sd) {
            span_graph.sym_dim_bounds.insert(local_sd, bound);
        }
    }

    // Collect all compute group indices assigned to this span.
    // This includes both primary assignments and duplicated groups.
    let mut span_compute_groups: Vec<(usize, u64, u64)> = Vec::new(); // (gi, offset, count)
    for a in lane_work {
        span_compute_groups.push((a.group_idx, a.atom_offset, a.atom_count));
    }

    // Compute the CLOSURE: find all groups that span_compute_groups transitively
    // depend on. For each producer not already in the span, either:
    // - It's a literal: inline if small, external input if large
    // - It's a compute group in an earlier phase: external input
    // - It's a compute group in this phase on this lane: already in span
    // - It's a compute group in this phase on another lane: BUG (should have
    //   been caught by fix_same_phase_violations)

    // Determine which groups are "in this span" as compute groups.
    let span_compute_set: HashSet<usize> =
        span_compute_groups.iter().map(|&(gi, _, _)| gi).collect();

    // Collect ALL transitive dependencies (group-level BFS).
    let mut needed_groups: BTreeSet<usize> = BTreeSet::new();
    let mut queue: VecDeque<usize> = VecDeque::new();

    for &gi in &span_compute_set {
        queue.push_back(gi);
    }

    while let Some(gi) = queue.pop_front() {
        // Find all producer groups of gi.
        // Include both the dependency graph producers AND any groups referenced
        // by InputRefs that might not be in the producers list (e.g., literal
        // groups referenced via ReduceSum strided access).
        let group = &groups[gi];

        // Direct producers (compute groups).
        for &pi in &producers[gi] {
            if needed_groups.insert(pi) {
                if span_compute_set.contains(&pi) {
                    // Already in span, but may need to recurse its deps.
                    queue.push_back(pi);
                }
                // If pi is in an earlier phase or is a literal, it's an external dep.
                // Don't recurse — its deps are already satisfied externally.
            }
        }

        // Literal groups referenced by InputRefs (not in producers list).
        for input in &group.inputs {
            let lit_groups =
                resolve_literal_producer_groups(input, group.count, groups, is_literal);
            for pi in lit_groups {
                needed_groups.insert(pi);
            }
        }

        // ReduceSum/ReduceMax: extended access range may touch additional literal groups.
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
                    let lit_groups = resolve_literal_producer_groups_with_reduce(
                        input,
                        group.count,
                        *reduce_count,
                        *reduce_stride,
                        groups,
                        is_literal,
                    );
                    for pi in lit_groups {
                        needed_groups.insert(pi);
                    }
                }
            }
            _ => {}
        }

        // IndirectLoad table.
        if let ScalarOp::IndirectLoad { table_base, .. } = &group.op {
            if let Some(pi) = find_group_idx(groups, *table_base) {
                needed_groups.insert(pi);
            }
        }
    }

    // Now classify needed_groups into:
    // 1. Small literals → inline (copy into span)
    // 2. Large literals → external input
    // 3. Compute groups in this span → already handled
    // 4. Compute groups NOT in this span → external input

    let mut inlined_literals: BTreeSet<usize> = BTreeSet::new();
    let mut external_groups: BTreeMap<usize, (u64, u64)> = BTreeMap::new(); // gi -> (offset, count)

    for &gi in &needed_groups {
        if span_compute_set.contains(&gi) {
            continue; // Already in span as compute.
        }
        if is_literal[gi] {
            if groups[gi].count < LITERAL_INLINE_THRESHOLD {
                inlined_literals.insert(gi);
            } else {
                // Large literal → external input (full range).
                external_groups.insert(gi, (0, groups[gi].count));
            }
        } else {
            // Compute group not in this span → external input.
            // Determine which range we actually need.
            let range =
                compute_needed_range_from_span(gi, &span_compute_groups, groups, is_literal);
            if let Some((off, cnt)) = range {
                let entry = external_groups.entry(gi).or_insert((off, cnt));
                // Extend range if needed.
                let new_lo = entry.0.min(off);
                let new_hi = (entry.0 + entry.1).max(off + cnt);
                *entry = (new_lo, new_hi - new_lo);
            } else {
                // Need full group.
                external_groups.insert(gi, (0, groups[gi].count));
            }
        }
    }

    // Also need to find external ranges for compute groups in the span that read
    // from groups outside the span. The needed_groups BFS above found WHICH groups,
    // but we need the exact RANGES.
    //
    // Refine: for each compute group in the span, resolve its InputRefs and find
    // exactly which atom ranges from external groups are needed.
    let mut refined_external: Vec<(usize, u64, u64)> = Vec::new();

    for &(gi, atom_offset, atom_count) in &span_compute_groups {
        let group = &groups[gi];
        let ext_ranges = collect_external_ranges_for_group(
            group,
            atom_offset,
            atom_count,
            groups,
            is_literal,
            &span_compute_set,
            &inlined_literals,
        );
        refined_external.extend(ext_ranges);
    }

    // Merge overlapping ranges per group.
    refined_external.sort_by_key(|&(gi, off, _)| (gi, off));
    let merged_external = merge_group_ranges(&refined_external);

    // Build the span graph.
    let mut main_to_local = RangeAtomMap::new();

    // 1. Inline small literals.
    for &lit_gi in &inlined_literals {
        let lit_group = &groups[lit_gi];
        let local_base = span_graph.push_group(
            lit_group.count,
            lit_group.op.clone(),
            lit_group.sym_dims.clone(),
            lit_group.reduce_dims.clone(),
            vec![],
        );
        main_to_local.insert_range(lit_group.base_id, local_base, lit_group.count);
    }

    // 2. Create placeholder groups for external inputs.
    let mut input_mappings: Vec<AtomMapping> = Vec::new();
    for &(gi, offset, count) in &merged_external {
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

    // 3. Build compute groups in topo order (group_idx order is topo order).
    let mut output_mappings: Vec<AtomMapping> = Vec::new();

    // Determine which slices this lane "owns" for output.
    // Non-duplicate assignments output their slice. Duplicate assignments don't output.
    let owned_slices: HashSet<(usize, u64)> = lane_work
        .iter()
        .filter(|a| !a.is_duplicate)
        .map(|a| (a.group_idx, a.atom_offset))
        .collect();

    for &(gi, atom_offset, atom_count) in &span_compute_groups {
        let group = &groups[gi];

        let local_inputs = remap_inputs_range(
            &group.inputs,
            &group.op,
            atom_offset,
            atom_count,
            group.count,
            groups,
            &main_to_local,
        );
        let local_op = remap_op(&group.op, &main_to_local);

        let local_base = span_graph.push_group(
            atom_count,
            local_op,
            group.sym_dims.clone(),
            group.reduce_dims.clone(),
            local_inputs,
        );

        let main_base = AtomId(group.base_id.0 + atom_offset);
        main_to_local.insert_range(main_base, local_base, atom_count);

        // Output mapping: only for non-duplicate assignments.
        if owned_slices.contains(&(gi, atom_offset)) {
            output_mappings.push(AtomMapping {
                main_base,
                span_base: local_base,
                count: atom_count,
            });
        }
    }

    Span {
        graph: span_graph,
        inputs: input_mappings,
        outputs: output_mappings,
    }
}

/// Compute the range of a producer group that's needed by the span's compute groups.
fn compute_needed_range_from_span(
    prod_gi: usize,
    span_compute: &[(usize, u64, u64)],
    groups: &[AtomGroup],
    is_literal: &[bool],
) -> Option<(u64, u64)> {
    // For now, return None (full range). A more precise implementation would
    // trace each InputRef to determine the exact sub-range needed.
    None
}

/// Collect external input ranges for a single compute group in the span.
fn collect_external_ranges_for_group(
    group: &AtomGroup,
    atom_offset: u64,
    atom_count: u64,
    all_groups: &[AtomGroup],
    is_literal: &[bool],
    span_compute_set: &HashSet<usize>,
    inlined_literals: &BTreeSet<usize>,
) -> Vec<(usize, u64, u64)> {
    let mut result = Vec::new();
    if atom_count == 0 {
        return result;
    }

    let (is_reduce, reduce_count, reduce_stride) = match &group.op {
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
        _ => (false, 0, 0),
    };

    let is_in_span =
        |gi: usize| -> bool { span_compute_set.contains(&gi) || inlined_literals.contains(&gi) };

    for input in &group.inputs {
        let referenced = resolve_input_to_group_ranges(
            input,
            atom_offset,
            atom_count,
            if is_reduce { reduce_count } else { 1 },
            if is_reduce { reduce_stride } else { 0 },
            all_groups,
        );

        for (gi, range_lo, range_hi) in referenced {
            if is_in_span(gi) {
                continue;
            }
            let g_lo = all_groups[gi].base_id.0;
            let g_hi = g_lo + all_groups[gi].count;
            let overlap_lo = range_lo.max(g_lo);
            let overlap_hi = range_hi.min(g_hi);
            if overlap_lo >= overlap_hi {
                continue;
            }
            let offset = overlap_lo - g_lo;
            let count = overlap_hi - overlap_lo;
            result.push((gi, offset, count));
        }
    }

    // IndirectLoad table.
    if let ScalarOp::IndirectLoad { table_base, .. } = &group.op {
        if let Some(gi) = find_group_idx(all_groups, *table_base) {
            if !is_in_span(gi) {
                result.push((gi, 0, all_groups[gi].count));
            }
        }
    }

    result
}

// ─── Range-based atom map ────────────────────────────────────────────────────

/// Efficient main→span atom ID remapping using sorted ranges.
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

// ─── InputRef resolution helpers ─────────────────────────────────────────────

/// Find all groups that produce atoms referenced by an InputRef.
fn resolve_producer_groups(input: &InputRef, count: u64, groups: &[AtomGroup]) -> Vec<usize> {
    match input {
        InputRef::Broadcast(atom_id) => find_group_idx(groups, *atom_id).into_iter().collect(),
        InputRef::Affine { base, stride } => {
            if count == 0 {
                return vec![];
            }
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
                    if seen.insert(gi) {
                        result.push(gi);
                    }
                }
            }
            result
        }
        InputRef::SymAffine { base, stride_i, .. } => {
            if count == 0 {
                return vec![];
            }
            let last_offset = (*stride_i as i64) * (count as i64 - 1);
            let last = AtomId(base.0.wrapping_add(last_offset as u64));
            let lo = base.0.min(last.0);
            let hi = base.0.max(last.0);
            find_groups_in_range(groups, lo, hi)
        }
        InputRef::StridedBroadcast {
            base,
            stride,
            repeat,
        } => {
            if count == 0 {
                return vec![];
            }
            let num_blocks = (count + repeat - 1) / repeat;
            let last_offset = *stride * (num_blocks as i64 - 1);
            let last = AtomId(base.0.wrapping_add(last_offset as u64));
            let lo = base.0.min(last.0);
            let hi = base.0.max(last.0);
            find_groups_in_range(groups, lo, hi)
        }
        InputRef::Modular {
            base,
            stride,
            modulus,
        } => {
            if *modulus == 0 {
                return vec![];
            }
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
    if count == 0 {
        return vec![];
    }
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
        InputRef::StridedBroadcast {
            base,
            stride,
            repeat,
        } => {
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

/// Resolve producer groups that are literals.
fn resolve_literal_producer_groups(
    input: &InputRef,
    count: u64,
    groups: &[AtomGroup],
    is_literal: &[bool],
) -> Vec<usize> {
    resolve_producer_groups(input, count, groups)
        .into_iter()
        .filter(|&gi| is_literal[gi])
        .collect()
}

/// Resolve literal producer groups considering reduce stride.
fn resolve_literal_producer_groups_with_reduce(
    input: &InputRef,
    count: u64,
    reduce_count: u64,
    reduce_stride: i64,
    groups: &[AtomGroup],
    is_literal: &[bool],
) -> Vec<usize> {
    resolve_producer_groups_with_reduce(input, count, reduce_count, reduce_stride, groups)
        .into_iter()
        .filter(|&gi| is_literal[gi])
        .collect()
}

/// Resolve an InputRef to actual (group_idx, atom_lo, atom_hi) tuples.
/// atom_lo and atom_hi are absolute atom IDs (not offsets within the group).
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
            let first_k = offset;
            let last_k = offset + count - 1;
            let first_pos = base.0 as i64 + *stride as i64 * first_k as i64;
            let last_pos = base.0 as i64 + *stride as i64 * last_k as i64;

            let base_lo = first_pos.min(last_pos);
            let base_hi = first_pos.max(last_pos);
            let (ext_lo, ext_hi) =
                reduce_extent_range(base_lo, base_hi, reduce_count, reduce_stride);
            for gi in find_groups_in_range(groups, ext_lo as u64, ext_hi as u64) {
                result.push((gi, ext_lo as u64, (ext_hi + 1) as u64));
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
                result.push((gi, ext_lo as u64, (ext_hi + 1) as u64));
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
        InputRef::SymAffine {
            base,
            stride_i,
            stride_k,
        } => {
            let first_pos = base.0 as i64 + *stride_i as i64 * offset as i64;
            let last_pos = base.0 as i64 + *stride_i as i64 * (offset + count - 1) as i64;
            let base_lo = first_pos.min(last_pos);
            let base_hi = first_pos.max(last_pos);
            let (ext_lo, ext_hi) =
                reduce_extent_range(base_lo, base_hi, reduce_count, reduce_stride);
            for gi in find_groups_in_range(groups, ext_lo as u64, ext_hi as u64) {
                result.push((gi, ext_lo as u64, (ext_hi + 1) as u64));
            }
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

// ─── Group/atom lookup helpers ───────────────────────────────────────────────

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

fn find_groups_in_range(groups: &[AtomGroup], lo: u64, hi: u64) -> Vec<usize> {
    let mut result = Vec::new();
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

fn merge_group_ranges(ranges: &[(usize, u64, u64)]) -> Vec<(usize, u64, u64)> {
    if ranges.is_empty() {
        return vec![];
    }
    let mut sorted: Vec<(usize, u64, u64)> = ranges.to_vec();
    sorted.sort_by_key(|&(gi, off, _)| (gi, off));
    let mut merged: Vec<(usize, u64, u64)> = Vec::new();
    for (gi, off, count) in sorted {
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
    inputs
        .iter()
        .map(|input| remap_single_input(input, atom_offset, atom_count, orig_group_count, atom_map))
        .collect()
}

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
            let new_offset_in_block = atom_offset % repeat;
            if new_offset_in_block == 0 {
                InputRef::StridedBroadcast {
                    base: atom_map.get(new_base_raw).unwrap_or(new_base_raw),
                    stride: *stride,
                    repeat: *repeat,
                }
            } else {
                // Can't maintain StridedBroadcast pattern with non-aligned offset.
                // Fall back to Explicit.
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
    /// Returns a list of error strings; empty means valid.
    pub fn validate(&self, original: &NanoGraph) -> Vec<String> {
        let mut errors = Vec::new();

        // Check each span's NanoGraph is internally valid.
        for (phase_idx, phase) in self.phases.iter().enumerate() {
            for (lane_idx, span) in phase.spans.iter().enumerate() {
                let graph_errors = span.graph.validate();
                for err in graph_errors {
                    errors.push(format!("Phase {} lane {}: {}", phase_idx, lane_idx, err));
                }
            }
        }

        // Check that all span inputs are available.
        let mut available: HashSet<AtomId> = HashSet::new();
        for group in original.groups() {
            if matches!(group.op, ScalarOp::Literal(_)) && group.inputs.is_empty() {
                for i in 0..group.count {
                    available.insert(group.base_id.offset(i));
                }
            }
        }

        for (phase_idx, phase) in self.phases.iter().enumerate() {
            // Check inputs are available.
            for (lane_idx, span) in phase.spans.iter().enumerate() {
                for mapping in &span.inputs {
                    for i in 0..mapping.count {
                        let atom = mapping.main_base.offset(i);
                        if !available.contains(&atom) {
                            errors.push(format!(
                                "Phase {} lane {}: input atom {} not available",
                                phase_idx, lane_idx, atom
                            ));
                            break; // One error per mapping is enough.
                        }
                    }
                }
            }

            // After the phase, all outputs become available.
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
    use crate::compiler::attempts::v13_claude::test_graphs;
    use crate::dtype::DType;
    use crate::nano_graph::{InputRef, NanoGraph, ScalarBinOp, ScalarOp, ScalarUnaryOp};
    use crate::numeric_scalar::NumericScalar;

    // ─── Helper functions ────────────────────────────────────────────────

    /// Verify that every span's NanoGraph is internally valid.
    fn verify_span_graphs(plan: &SpanPlan) {
        for (pi, phase) in plan.phases.iter().enumerate() {
            for (li, span) in phase.spans.iter().enumerate() {
                let errors = span.graph.validate();
                assert!(
                    errors.is_empty(),
                    "Phase {} lane {} graph validation failed:\n{}",
                    pi,
                    li,
                    errors.join("\n")
                );
            }
        }
    }

    /// Verify that all compute atoms appear in exactly one span's outputs.
    fn verify_output_coverage(graph: &NanoGraph, plan: &SpanPlan) {
        let groups = graph.groups();
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
            if matches!(group.op, ScalarOp::Literal(_)) && group.inputs.is_empty() {
                continue;
            }
            for offset in 0..group.count {
                let atom = AtomId(group.base_id.0 + offset);
                let count = produced.get(&atom).copied().unwrap_or(0);
                assert_eq!(
                    count, 1,
                    "Atom {} (group {}, offset {}) appears {} times in outputs (expected 1)",
                    atom, gi, offset, count
                );
            }
        }
    }

    /// Verify that span inputs are available from earlier phases or literals.
    fn verify_input_availability(graph: &NanoGraph, plan: &SpanPlan) {
        let errors = plan.validate(graph);
        assert!(
            errors.is_empty(),
            "Input availability errors:\n{}",
            errors.join("\n")
        );
    }

    /// Verify no cross-span reads within the same phase.
    /// Each span should only read atoms that are either:
    /// - Produced by earlier phases (available after barrier)
    /// - Literal groups
    /// - Produced within the same span
    fn verify_no_cross_span_reads(graph: &NanoGraph, plan: &SpanPlan) {
        let mut available: HashSet<AtomId> = HashSet::new();

        // All literals are always available.
        for group in graph.groups() {
            if matches!(group.op, ScalarOp::Literal(_)) && group.inputs.is_empty() {
                for i in 0..group.count {
                    available.insert(group.base_id.offset(i));
                }
            }
        }

        for (phase_idx, phase) in plan.phases.iter().enumerate() {
            // Collect atoms produced by each span in this phase.
            let mut span_outputs: Vec<HashSet<AtomId>> = Vec::new();
            for span in &phase.spans {
                let mut outputs = HashSet::new();
                for mapping in &span.outputs {
                    for i in 0..mapping.count {
                        outputs.insert(mapping.main_base.offset(i));
                    }
                }
                span_outputs.push(outputs);
            }

            // Check each span's inputs don't reference another span's outputs
            // in the same phase.
            for (lane_idx, span) in phase.spans.iter().enumerate() {
                for mapping in &span.inputs {
                    for i in 0..mapping.count {
                        let atom = mapping.main_base.offset(i);
                        if !available.contains(&atom) {
                            // This atom must come from the same phase.
                            // Check it's not from another span.
                            for (other_lane, other_outputs) in span_outputs.iter().enumerate() {
                                if other_lane != lane_idx && other_outputs.contains(&atom) {
                                    panic!(
                                        "Cross-span read! Phase {} lane {} reads atom {} \
                                         which is produced by lane {} in the same phase",
                                        phase_idx, lane_idx, atom, other_lane
                                    );
                                }
                            }
                        }
                    }
                }
            }

            // After the phase, all outputs become available.
            for outputs in &span_outputs {
                available.extend(outputs);
            }
        }
    }

    /// Run all verification checks on a plan.
    fn verify_plan(graph: &NanoGraph, plan: &SpanPlan) {
        verify_span_graphs(plan);
        verify_output_coverage(graph, plan);
        verify_input_availability(graph, plan);
        verify_no_cross_span_reads(graph, plan);
    }

    // ─── Test cases ──────────────────────────────────────────────────────

    #[test]
    fn test_empty_graph() {
        let g = NanoGraph::new();
        let plan = plan_execution_spans(&g, 4);
        assert_eq!(plan.phases.len(), 0);
    }

    #[test]
    fn test_single_lane() {
        let (g, _, _, _) = test_graphs::elementwise_binary(1024, ScalarBinOp::Add);
        let plan = plan_execution_spans(&g, 1);
        verify_plan(&g, &plan);
        // With 1 lane, should be 1 phase, 1 span.
        assert_eq!(plan.num_lanes, 1);
        assert!(plan.phases.len() >= 1);
    }

    #[test]
    fn test_elementwise_split() {
        let (g, _, _, _) = test_graphs::elementwise_binary(1024, ScalarBinOp::Add);
        let plan = plan_execution_spans(&g, 4);
        verify_plan(&g, &plan);
    }

    #[test]
    fn test_broadcast_add() {
        let (g, _, _, _) = test_graphs::broadcast_add(1024);
        let plan = plan_execution_spans(&g, 4);
        verify_plan(&g, &plan);
    }

    #[test]
    fn test_unary_chain() {
        let (g, _, _) = test_graphs::unary_chain(
            256,
            &[ScalarUnaryOp::Exp, ScalarUnaryOp::Neg, ScalarUnaryOp::Tanh],
        );
        let plan = plan_execution_spans(&g, 4);
        verify_plan(&g, &plan);
    }

    #[test]
    fn test_matmul_small() {
        let (g, _, _, _) = test_graphs::matmul(4, 8, 16);
        let plan = plan_execution_spans(&g, 2);
        verify_plan(&g, &plan);
    }

    #[test]
    fn test_matmul_activation() {
        let (g, _, _, _) = test_graphs::matmul_activation(4, 8, 16, ScalarUnaryOp::Tanh);
        let plan = plan_execution_spans(&g, 2);
        verify_plan(&g, &plan);
    }

    #[test]
    fn test_matmul_chain() {
        let (g, _, _, _, _) = test_graphs::matmul_chain(4, 8, 16, 16, 32);
        let plan = plan_execution_spans(&g, 2);
        verify_plan(&g, &plan);
    }

    #[test]
    fn test_matmul_4_lanes() {
        let (g, _, _, _) = test_graphs::matmul(8, 4, 16);
        let plan = plan_execution_spans(&g, 4);
        verify_plan(&g, &plan);
    }

    #[test]
    fn test_cross_lane_pattern() {
        // This is the pattern that defeated previous attempts:
        // A small Select-like group (split across lanes) feeds a downstream
        // group that needs the FULL output via Modular.
        let g = build_cross_lane_graph(3072, 49152);
        let plan = plan_execution_spans(&g, 4);
        verify_plan(&g, &plan);
    }

    #[test]
    fn test_cross_lane_small() {
        let g = build_cross_lane_graph(16, 64);
        let plan = plan_execution_spans(&g, 4);
        verify_plan(&g, &plan);
    }

    #[test]
    fn test_many_lanes() {
        let (g, _, _, _) = test_graphs::elementwise_binary(8192, ScalarBinOp::Mul);
        let plan = plan_execution_spans(&g, 8);
        verify_plan(&g, &plan);
    }

    #[test]
    fn test_matmul_larger() {
        // Larger matmul to stress-test.
        let (g, _, _, _) = test_graphs::matmul(16, 8, 32);
        let plan = plan_execution_spans(&g, 4);
        verify_plan(&g, &plan);
    }

    #[test]
    fn test_self_contained_span_graphs() {
        // Specifically test that each span's NanoGraph has all InputRefs
        // resolving to atoms within the span.
        let (g, _, _, _) = test_graphs::matmul(4, 4, 8);
        let plan = plan_execution_spans(&g, 2);

        for (pi, phase) in plan.phases.iter().enumerate() {
            for (li, span) in phase.spans.iter().enumerate() {
                let errors = span.graph.validate();
                assert!(
                    errors.is_empty(),
                    "Phase {} lane {} is not self-contained:\n{}",
                    pi,
                    li,
                    errors.join("\n")
                );
            }
        }
    }

    #[test]
    fn test_reduce_sum_chain() {
        // Two ReduceSums in sequence (like layernorm).
        let mut g = NanoGraph::new();
        let input = g.push_group(
            64,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        // First matmul: 2 rows, K=4, N=8
        let mut mul_bases = Vec::new();
        for row in 0..2u64 {
            let mul = g.push_group(
                32, // K*N = 4*8
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![
                    InputRef::StridedBroadcast {
                        base: AtomId(input.0 + row * 4),
                        stride: 1,
                        repeat: 8,
                    },
                    InputRef::Affine {
                        base: AtomId(input.0 + 8), // second half as B
                        stride: 1,
                    },
                ],
            );
            mul_bases.push(mul);
        }

        let mut red_bases = Vec::new();
        for row in 0..2u64 {
            let red = g.push_group(
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
                    base: mul_bases[row as usize],
                    stride: 1,
                }],
            );
            red_bases.push(red);
        }

        // Elementwise activation on the 16 reduce outputs.
        let act = g.push_group(
            16,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Tanh,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: red_bases[0],
                stride: 1,
            }],
        );
        g.outputs = vec![act];

        let plan = plan_execution_spans(&g, 2);
        verify_plan(&g, &plan);
    }

    // ─── Test graph builders ─────────────────────────────────────────────

    /// Build a graph with the cross-lane violation pattern:
    /// Small group (Select-like) → downstream group with Modular access.
    fn build_cross_lane_graph(small_count: u64, big_count: u64) -> NanoGraph {
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

        // Select group — will be split across lanes.
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

        // Downstream group that reads the FULL select output via Modular.
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
}
