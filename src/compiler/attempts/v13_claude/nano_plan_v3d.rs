#![allow(
    clippy::all,
    dead_code,
    unreachable_patterns,
    unused_variables,
    unused_imports
)]
//! Span-based partitioner v3d: Optimistic plan → verify → repair loop.
//!
//! Previous attempts tried to get the plan right on the first pass. That's hard
//! because splitting and phase assignment interact in complex ways. Instead:
//!
//! 1. **Optimistic assignment**: Assign every group to a lane and phase. Split
//!    large groups across lanes for balance. Group same-lane dependencies into
//!    the same phase.
//!
//! 2. **Verify**: For each phase, check every span against every other span. If
//!    span A reads atoms produced by span B (same phase), that's a violation.
//!    Uses GROUP-level checks: resolve each InputRef to its producer group, check
//!    if that group is in a different lane's span in the same phase.
//!
//! 3. **Repair**: For each violation, apply the cheapest general fix:
//!    - If the producing group is small (< DUPLICATION_THRESHOLD atoms): duplicate
//!      it into the consuming span.
//!    - Otherwise: bump the consuming group (and its same-lane dependents) to a
//!      new phase (insert barrier).
//!    Repeat verify+repair until clean.
//!
//! 4. **Build span NanoGraphs**: Only after the plan is violation-free, build the
//!    span NanoGraphs with AtomMapping ranges.
//!
//! The verify+repair loop is GENERAL — it doesn't know anything about op types or
//! InputRef patterns. It just checks "does span A read from span B in the same
//! phase?" and fixes it.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use crate::nano_graph::{AtomGroup, AtomId, InputRef, NanoGraph, ScalarOp};

/// Literal groups with fewer atoms than this are duplicated into spans.
const LITERAL_INLINE_THRESHOLD: u64 = 1024;

/// Groups with fewer atoms than this threshold may be duplicated to resolve
/// cross-lane violations instead of inserting a barrier.
const DUPLICATION_THRESHOLD: u64 = 8192;

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

// ─── Range-based atom map ────────────────────────────────────────────────────

/// Sorted ranges for O(log n) main→span atom ID lookup.
struct RangeAtomMap {
    ranges: Vec<(u64, u64, u64)>, // (main_base, span_base, count)
}

impl RangeAtomMap {
    fn new() -> Self {
        Self { ranges: Vec::new() }
    }

    fn insert_range(&mut self, main_base: AtomId, span_base: AtomId, count: u64) {
        self.ranges.push((main_base.0, span_base.0, count));
    }

    fn sort(&mut self) {
        self.ranges.sort_by_key(|&(base, _, _)| base);
    }

    fn get(&self, main_id: AtomId) -> Option<AtomId> {
        let idx = self
            .ranges
            .partition_point(|&(base, _, _)| base <= main_id.0);
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
/// Uses an optimistic assignment + verify + repair loop.
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

    // Step 1: Build group dependency DAG.
    let (producers, consumers) = build_group_deps(groups);
    let topo_order = topological_sort(n, &producers);

    // Step 2: Optimistic phase + lane assignment.
    let (mut group_phase, mut group_lane, mut duplicates) = optimistic_assignment(
        groups,
        num_lanes,
        &topo_order,
        &producers,
        &consumers,
        &is_literal,
    );

    // Step 3: Verify + repair loop.
    let max_iterations = 50;
    for iteration in 0..max_iterations {
        let violations = verify(
            groups,
            num_lanes,
            &group_phase,
            &group_lane,
            &producers,
            &is_literal,
            &duplicates,
        );
        if violations.is_empty() {
            break;
        }
        repair(
            groups,
            num_lanes,
            &mut group_phase,
            &mut group_lane,
            &mut duplicates,
            &producers,
            &consumers,
            &is_literal,
            &violations,
            &topo_order,
        );
    }

    // Step 4: Build span NanoGraphs.
    let num_phases = group_phase.iter().copied().max().unwrap_or(0) + 1;
    build_span_plan(
        graph,
        num_lanes,
        num_phases,
        &group_phase,
        &group_lane,
        &producers,
        &is_literal,
        &duplicates,
    )
}

// ─── Group dependency DAG ────────────────────────────────────────────────────

/// Build producer and consumer DAG at group level.
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

        // ReduceSum/ReduceMax strided access.
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

// ─── Topological sort ────────────────────────────────────────────────────────

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

// ─── Phase 1: Optimistic assignment ──────────────────────────────────────────

/// Assign every compute group to a (phase, lane) pair.
///
/// Strategy:
/// - Compute depth for each group (longest path from root).
/// - Identify convergence points (groups reading from multiple independent families)
///   as mandatory phase boundaries.
/// - Within each phase, assign groups to lanes using greedy load balancing,
///   keeping same-phase dependent chains on the same lane.
///
/// Returns (group_phase, group_lane, duplicates).
/// `duplicates` is a map: for each phase, which groups should be duplicated
/// into all lanes (initially empty — populated during repair).
fn optimistic_assignment(
    groups: &[AtomGroup],
    num_lanes: usize,
    topo_order: &[usize],
    producers: &[Vec<usize>],
    consumers: &[Vec<usize>],
    is_literal: &[bool],
) -> (Vec<usize>, Vec<usize>, HashMap<usize, HashSet<usize>>) {
    let n = groups.len();

    // Compute depth (longest path from any root).
    let depth = compute_depth(n, topo_order, producers, is_literal);

    // Identify independent families for convergence detection.
    let families = identify_families(groups, topo_order, producers, is_literal);

    // Assign phases.
    let mut group_phase = vec![0usize; n];
    for &gi in topo_order {
        if is_literal[gi] {
            continue;
        }

        let max_producer_phase = producers[gi]
            .iter()
            .filter(|&&pi| !is_literal[pi])
            .map(|&pi| group_phase[pi])
            .max()
            .unwrap_or(0);

        // Check convergence: multiple distinct families from producers.
        let producer_families: HashSet<usize> = producers[gi]
            .iter()
            .filter(|&&pi| !is_literal[pi])
            .map(|&pi| families[pi])
            .collect();

        if producer_families.len() > 1 {
            group_phase[gi] = max_producer_phase + 1;
        } else {
            group_phase[gi] = max_producer_phase;
        }
    }

    // Assign lanes within each phase.
    let num_phases = group_phase.iter().copied().max().unwrap_or(0) + 1;
    let mut group_lane = vec![0usize; n];

    for phase in 0..num_phases {
        let phase_groups: Vec<usize> = (0..n)
            .filter(|&gi| !is_literal[gi] && group_phase[gi] == phase)
            .collect();

        if phase_groups.is_empty() {
            continue;
        }

        let phase_set: HashSet<usize> = phase_groups.iter().copied().collect();

        // Union-Find: chain groups with within-phase dependencies.
        let mut parent: Vec<usize> = (0..n).collect();
        for &gi in &phase_groups {
            for &pi in &producers[gi] {
                if phase_set.contains(&pi) {
                    uf_union(&mut parent, gi, pi);
                }
            }
        }

        // Collect chains.
        let mut chains: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for &gi in &phase_groups {
            let root = uf_find(&parent, gi);
            chains.entry(root).or_default().push(gi);
        }

        // Sort chains largest-first for better balance.
        let mut chain_list: Vec<Vec<usize>> = chains.into_values().collect();
        chain_list.sort_by(|a, b| {
            let count_a: u64 = a.iter().map(|&gi| groups[gi].count).sum();
            let count_b: u64 = b.iter().map(|&gi| groups[gi].count).sum();
            count_b.cmp(&count_a)
        });

        // Greedy lane assignment: assign each chain to the least-loaded lane.
        let mut lane_load = vec![0u64; num_lanes];
        for chain in &chain_list {
            let chain_atoms: u64 = chain.iter().map(|&gi| groups[gi].count).sum();
            let best_lane = lane_load
                .iter()
                .enumerate()
                .min_by_key(|&(_, load)| *load)
                .unwrap()
                .0;
            for &gi in chain {
                group_lane[gi] = best_lane;
            }
            lane_load[best_lane] += chain_atoms;
        }
    }

    (group_phase, group_lane, HashMap::new())
}

/// Compute depth (longest path from root).
fn compute_depth(
    n: usize,
    topo_order: &[usize],
    producers: &[Vec<usize>],
    is_literal: &[bool],
) -> Vec<usize> {
    let mut depth = vec![0usize; n];
    for &gi in topo_order {
        if is_literal[gi] {
            depth[gi] = 0;
            continue;
        }
        let max_prod_depth = producers[gi].iter().map(|&pi| depth[pi]).max().unwrap_or(0);
        depth[gi] = if producers[gi].iter().all(|&pi| is_literal[pi]) {
            1
        } else {
            max_prod_depth + 1
        };
    }
    depth
}

/// Identify independent root families for convergence detection.
fn identify_families(
    groups: &[AtomGroup],
    topo_order: &[usize],
    producers: &[Vec<usize>],
    is_literal: &[bool],
) -> Vec<usize> {
    let n = groups.len();
    let mut family = vec![0usize; n];
    let mut next_family = 0usize;

    // Root compute groups get unique families.
    for &gi in topo_order {
        if is_literal[gi] {
            continue;
        }
        let has_compute_producer = producers[gi].iter().any(|&pi| !is_literal[pi]);
        if !has_compute_producer {
            family[gi] = next_family;
            next_family += 1;
        }
    }

    // Forward-propagate.
    for &gi in topo_order {
        if is_literal[gi] {
            continue;
        }
        let has_compute_producer = producers[gi].iter().any(|&pi| !is_literal[pi]);
        if !has_compute_producer {
            continue;
        }

        let producer_families: BTreeSet<usize> = producers[gi]
            .iter()
            .filter(|&&pi| !is_literal[pi])
            .map(|&pi| family[pi])
            .collect();

        if producer_families.len() == 1 {
            family[gi] = *producer_families.iter().next().unwrap();
        } else {
            family[gi] = next_family;
            next_family += 1;
        }
    }

    family
}

// ─── Phase 2: Verify ─────────────────────────────────────────────────────────

/// A violation: group `consumer_gi` in (phase, lane) reads from group `producer_gi`
/// which is in (same phase, different lane).
#[derive(Debug, Clone)]
struct Violation {
    consumer_gi: usize,
    consumer_lane: usize,
    producer_gi: usize,
    producer_lane: usize,
    phase: usize,
}

/// Check all phases for cross-lane violations.
///
/// For each phase, for each compute group, resolve all its producer groups.
/// If any producer is in the same phase but a different lane (and is not a
/// literal, and is not in the duplicates set for this phase), it's a violation.
///
/// Crucially, duplicated groups must also be checked: a duplicated group exists
/// on ALL lanes, so its producers must be accessible from ALL lanes. If a dup
/// group's producer is in the same phase, not itself duplicated, then it's only
/// on one lane, causing a violation on all other lanes.
fn verify(
    groups: &[AtomGroup],
    num_lanes: usize,
    group_phase: &[usize],
    group_lane: &[usize],
    producers: &[Vec<usize>],
    is_literal: &[bool],
    duplicates: &HashMap<usize, HashSet<usize>>,
) -> Vec<Violation> {
    let n = groups.len();
    let mut violations = Vec::new();

    for gi in 0..n {
        if is_literal[gi] {
            continue;
        }
        let phase = group_phase[gi];
        let lane = group_lane[gi];
        let phase_dups = duplicates.get(&phase);
        let gi_is_dup = phase_dups.map_or(false, |d| d.contains(&gi));

        for &pi in &producers[gi] {
            if is_literal[pi] {
                continue;
            }
            // Skip if producer is duplicated into all lanes in this phase.
            if let Some(dups) = phase_dups {
                if dups.contains(&pi) {
                    continue;
                }
            }

            if group_phase[pi] != phase {
                continue; // Producer is in a different phase — fine.
            }

            if gi_is_dup {
                // gi is duplicated to ALL lanes. Its producer pi is only on
                // group_lane[pi]. This is a cross-lane violation for every
                // other lane. Report it once (the repair will handle it).
                violations.push(Violation {
                    consumer_gi: gi,
                    consumer_lane: group_lane[pi], // arbitrary, it's cross-lane from other lanes
                    producer_gi: pi,
                    producer_lane: group_lane[pi],
                    phase,
                });
            } else if group_lane[pi] != lane {
                violations.push(Violation {
                    consumer_gi: gi,
                    consumer_lane: lane,
                    producer_gi: pi,
                    producer_lane: group_lane[pi],
                    phase,
                });
            }
        }
    }

    violations
}

// ─── Phase 3: Repair ─────────────────────────────────────────────────────────

/// Fix violations by either duplicating small producers or bumping consumers
/// to a new phase.
fn repair(
    groups: &[AtomGroup],
    num_lanes: usize,
    group_phase: &mut Vec<usize>,
    group_lane: &mut Vec<usize>,
    duplicates: &mut HashMap<usize, HashSet<usize>>,
    producers: &[Vec<usize>],
    consumers: &[Vec<usize>],
    is_literal: &[bool],
    violations: &[Violation],
    topo_order: &[usize],
) {
    // Collect unique (phase, producer_gi) pairs from violations.
    let mut producers_to_fix: BTreeSet<(usize, usize)> = BTreeSet::new();
    let mut consumers_to_bump: BTreeSet<(usize, usize)> = BTreeSet::new();

    for v in violations {
        let prod_atoms = groups[v.producer_gi].count;
        if prod_atoms < DUPLICATION_THRESHOLD {
            // Small producer: duplicate it into all lanes.
            producers_to_fix.insert((v.phase, v.producer_gi));
        } else {
            // Large producer: bump the consumer to a later phase.
            consumers_to_bump.insert((v.phase, v.consumer_gi));
        }
    }

    // Apply duplications.
    for (phase, prod_gi) in &producers_to_fix {
        duplicates.entry(*phase).or_default().insert(*prod_gi);
    }

    // Apply bumps: move consumer and its same-lane, same-phase dependents to
    // a new phase = current_phase + 1. We need to cascade: if C is bumped,
    // any group in the same phase that depends on C (directly or transitively)
    // and is on the same lane must also be bumped.
    if !consumers_to_bump.is_empty() {
        // For each bumped consumer, find all same-phase, same-lane transitive
        // dependents and bump them all.
        let n = groups.len();
        let mut bumped: HashSet<usize> = HashSet::new();

        for &(phase, consumer_gi) in &consumers_to_bump {
            if bumped.contains(&consumer_gi) {
                continue;
            }
            let lane = group_lane[consumer_gi];

            // BFS from consumer_gi through same-phase, same-lane consumers.
            let mut queue: VecDeque<usize> = VecDeque::new();
            queue.push_back(consumer_gi);
            bumped.insert(consumer_gi);

            while let Some(gi) = queue.pop_front() {
                for &ci in &consumers[gi] {
                    if is_literal[ci] {
                        continue;
                    }
                    if group_phase[ci] == phase && group_lane[ci] == lane && !bumped.contains(&ci) {
                        bumped.insert(ci);
                        queue.push_back(ci);
                    }
                }
            }
        }

        // Bump all collected groups to phase + 1.
        // We need to shift all phases after the bump point as well.
        // Strategy: just increment the phase of bumped groups by 1.
        // Then re-propagate forward: any group whose producer is now in a later
        // phase than itself must be bumped too.
        for &gi in &bumped {
            group_phase[gi] += 1;
        }

        // Forward propagate: ensure no group has a phase < max(producer phases).
        // Process in topo order.
        for &gi in topo_order {
            if is_literal[gi] {
                continue;
            }
            let min_phase = producers[gi]
                .iter()
                .filter(|&&pi| !is_literal[pi])
                .map(|&pi| group_phase[pi])
                .max()
                .unwrap_or(0);
            if group_phase[gi] < min_phase {
                group_phase[gi] = min_phase;
            }
        }

        // Re-assign lanes for any groups whose phase changed, maintaining
        // chain integrity within each phase.
        let num_phases = group_phase.iter().copied().max().unwrap_or(0) + 1;
        reassign_lanes_for_phases(
            groups,
            num_lanes,
            num_phases,
            group_phase,
            group_lane,
            producers,
            is_literal,
        );
    }
}

/// Re-assign lanes for groups to maintain chain integrity.
fn reassign_lanes_for_phases(
    groups: &[AtomGroup],
    num_lanes: usize,
    num_phases: usize,
    group_phase: &[usize],
    group_lane: &mut [usize],
    producers: &[Vec<usize>],
    is_literal: &[bool],
) {
    let n = groups.len();

    for phase in 0..num_phases {
        let phase_groups: Vec<usize> = (0..n)
            .filter(|&gi| !is_literal[gi] && group_phase[gi] == phase)
            .collect();

        if phase_groups.is_empty() {
            continue;
        }

        let phase_set: HashSet<usize> = phase_groups.iter().copied().collect();

        // Union-Find for within-phase dependency chains.
        let mut parent: Vec<usize> = (0..n).collect();
        for &gi in &phase_groups {
            for &pi in &producers[gi] {
                if phase_set.contains(&pi) {
                    uf_union(&mut parent, gi, pi);
                }
            }
        }

        // Collect chains.
        let mut chains: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for &gi in &phase_groups {
            let root = uf_find(&parent, gi);
            chains.entry(root).or_default().push(gi);
        }

        // Sort largest-first.
        let mut chain_list: Vec<Vec<usize>> = chains.into_values().collect();
        chain_list.sort_by(|a, b| {
            let count_a: u64 = a.iter().map(|&gi| groups[gi].count).sum();
            let count_b: u64 = b.iter().map(|&gi| groups[gi].count).sum();
            count_b.cmp(&count_a)
        });

        // Greedy lane assignment.
        let mut lane_load = vec![0u64; num_lanes];
        for chain in &chain_list {
            let chain_atoms: u64 = chain.iter().map(|&gi| groups[gi].count).sum();
            let best_lane = lane_load
                .iter()
                .enumerate()
                .min_by_key(|&(_, load)| *load)
                .unwrap()
                .0;
            for &gi in chain {
                group_lane[gi] = best_lane;
            }
            lane_load[best_lane] += chain_atoms;
        }
    }
}

// ─── Union-Find helpers ──────────────────────────────────────────────────────

fn uf_find(parent: &[usize], mut x: usize) -> usize {
    while parent[x] != x {
        x = parent[x];
    }
    x
}

fn uf_union(parent: &mut [usize], a: usize, b: usize) {
    let ra = uf_find(parent, a);
    let rb = uf_find(parent, b);
    if ra != rb {
        parent[ra] = rb;
    }
}

// ─── Phase 4: Build span NanoGraphs ──────────────────────────────────────────

fn build_span_plan(
    graph: &NanoGraph,
    num_lanes: usize,
    num_phases: usize,
    group_phase: &[usize],
    group_lane: &[usize],
    producers: &[Vec<usize>],
    is_literal: &[bool],
    duplicates: &HashMap<usize, HashSet<usize>>,
) -> SpanPlan {
    let groups = graph.groups();
    let n = groups.len();

    // Pre-compute consumers for output tracking.
    let mut group_consumers: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (gi, prods) in producers.iter().enumerate() {
        for &pi in prods {
            group_consumers[pi].push(gi);
        }
    }

    let mut phases = Vec::with_capacity(num_phases);

    for phase in 0..num_phases {
        let mut spans = Vec::with_capacity(num_lanes);
        let phase_dups = duplicates.get(&phase);

        for lane in 0..num_lanes {
            // Collect groups assigned to this (phase, lane).
            let mut cell_groups: Vec<usize> = (0..n)
                .filter(|&gi| !is_literal[gi] && group_phase[gi] == phase && group_lane[gi] == lane)
                .collect();

            // Add duplicated groups for this phase.
            if let Some(dups) = phase_dups {
                for &dup_gi in dups {
                    if !cell_groups.contains(&dup_gi) {
                        cell_groups.push(dup_gi);
                    }
                }
            }

            if cell_groups.is_empty() {
                spans.push(Span {
                    graph: NanoGraph::new(),
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                });
                continue;
            }

            // Sort for topological order (group indices are topo-ordered).
            cell_groups.sort();
            cell_groups.dedup();

            let cell_set: HashSet<usize> = cell_groups.iter().copied().collect();

            // Identify literal dependencies (small: inline, large: external input).
            let mut small_literal_deps: BTreeSet<usize> = BTreeSet::new();
            let mut large_literal_deps: BTreeSet<usize> = BTreeSet::new();
            for &gi in &cell_groups {
                for &pi in &producers[gi] {
                    if is_literal[pi] {
                        if groups[pi].count < LITERAL_INLINE_THRESHOLD {
                            small_literal_deps.insert(pi);
                        } else {
                            large_literal_deps.insert(pi);
                        }
                    }
                }
            }

            // Collect external input ranges (from groups not in this cell and not
            // inlined as small literals).
            let mut external_ranges: Vec<(usize, u64, u64)> = Vec::new();
            for &gi in &cell_groups {
                let group = &groups[gi];
                collect_external_ranges(
                    group,
                    groups,
                    &cell_set,
                    &small_literal_deps,
                    is_literal,
                    &mut external_ranges,
                );
            }
            // Large literals become external inputs too.
            for &li in &large_literal_deps {
                external_ranges.push((li, 0, groups[li].count));
            }
            let external_ranges = merge_group_ranges(&mut external_ranges);

            // Determine which original (non-duplicated) groups this lane owns
            // for output purposes. Duplicated groups only output from their
            // original lane assignment.
            let owned_groups: Vec<usize> = cell_groups
                .iter()
                .copied()
                .filter(|&gi| {
                    // This lane owns the group if:
                    // 1. The group is originally assigned here, OR
                    // 2. The group is duplicated AND this is lane 0 (arbitrary tie-break)
                    //    BUT actually, for duplicated groups, each lane computes it
                    //    independently but we only need ONE lane to output it.
                    //    The original lane should output it.
                    if let Some(dups) = phase_dups {
                        if dups.contains(&gi) {
                            // This group is duplicated. Only the originally assigned lane outputs.
                            return group_lane[gi] == lane;
                        }
                    }
                    true
                })
                .collect();

            // Collect output ranges: atoms consumed by groups outside this cell.
            let mut output_ranges: Vec<(usize, u64, u64)> = Vec::new();
            for &gi in &owned_groups {
                let group = &groups[gi];
                for &ci in &group_consumers[gi] {
                    if is_literal[ci] {
                        continue;
                    }
                    if cell_set.contains(&ci) {
                        // Internal consumer — but need to check: is ci actually on
                        // this lane in this phase, or is it a dup?
                        // If ci is a dup, it computes locally, so no output needed.
                        continue;
                    }
                    if let Some((offset, count)) = compute_consumed_range(gi, ci, groups) {
                        output_ranges.push((gi, offset, count));
                    }
                }
                // Also check if any atoms are graph outputs.
                for &out_id in &graph.outputs {
                    if group.contains(out_id) {
                        let offset = out_id.0 - group.base_id.0;
                        output_ranges.push((gi, offset, 1));
                    }
                }
            }
            let output_ranges = merge_group_ranges(&mut output_ranges);

            // Build the span graph.
            let span = build_span_graph(
                graph,
                &cell_groups,
                &small_literal_deps,
                &external_ranges,
                &output_ranges,
            );
            spans.push(span);
        }

        phases.push(Phase { spans });
    }

    SpanPlan { num_lanes, phases }
}

/// Build a self-contained NanoGraph span from the given groups.
fn build_span_graph(
    graph: &NanoGraph,
    cell_groups: &[usize],
    literal_deps: &BTreeSet<usize>,
    external_ranges: &[(usize, u64, u64)],
    output_ranges: &[(usize, u64, u64)],
) -> Span {
    let groups = graph.groups();
    let mut span_graph = NanoGraph::new();

    // Copy sym_dim setup.
    for (name, &sd) in &graph.sym_dim_names {
        let local_sd = span_graph.sym_dim(name);
        if let Some(&bound) = graph.sym_dim_bounds.get(&sd) {
            span_graph.sym_dim_bounds.insert(local_sd, bound);
        }
    }

    let mut atom_map = RangeAtomMap::new();

    // 1. Add small literal groups.
    for &li in literal_deps {
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

    // 2. Add stub literals for external input ranges.
    let mut input_mappings: Vec<AtomMapping> = Vec::new();
    for &(source_group_idx, offset, count) in external_ranges {
        let src_group = &groups[source_group_idx];
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

    atom_map.sort();

    // 3. Add compute groups (already sorted by group index = topo order).
    for &gi in cell_groups {
        let group = &groups[gi];
        let remapped_inputs = remap_input_refs(&group.inputs, &atom_map);
        let local_base = span_graph.push_group(
            group.count,
            group.op.clone(),
            remap_sym_dims(&group.sym_dims, graph, &span_graph),
            remap_sym_dims(&group.reduce_dims, graph, &span_graph),
            remapped_inputs,
        );
        atom_map.insert_range(group.base_id, local_base, group.count);
    }

    // 4. Build output mappings.
    let mut output_mappings: Vec<AtomMapping> = Vec::new();
    for &(gi, offset, count) in output_ranges {
        let group = &groups[gi];
        let main_base = group.base_id.offset(offset);
        if let Some(local_base) = atom_map.get(main_base) {
            output_mappings.push(AtomMapping {
                main_base,
                span_base: local_base,
                count,
            });
        }
    }

    Span {
        graph: span_graph,
        inputs: input_mappings,
        outputs: output_mappings,
    }
}

// ─── InputRef resolution helpers ─────────────────────────────────────────────

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
        InputRef::Modular {
            base,
            stride,
            modulus,
        } => {
            if *modulus == 0 {
                return vec![];
            }
            let first_read = base.0 as i64;
            let last_read = base.0 as i64 + *stride as i64 * (*modulus as i64 - 1);
            let lo = first_read.min(last_read) + min_reduce_ext;
            let hi = first_read.max(last_read) + max_reduce_ext;
            find_groups_in_range(groups, lo as u64, hi as u64)
        }
        InputRef::Explicit(ids) => {
            let mut result = HashSet::new();
            for id in ids {
                let base_val = id.0 as i64;
                let lo = base_val + min_reduce_ext;
                let hi = base_val + max_reduce_ext;
                for gi in find_groups_in_range(groups, lo as u64, hi as u64) {
                    result.insert(gi);
                }
            }
            result.into_iter().collect()
        }
    }
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

fn find_groups_in_range(groups: &[AtomGroup], lo: u64, hi: u64) -> Vec<usize> {
    let mut result = Vec::new();
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

// ─── External range collection ───────────────────────────────────────────────

/// Collect external atom ranges that a consumer group reads from outside the cell.
fn collect_external_ranges(
    group: &AtomGroup,
    all_groups: &[AtomGroup],
    cell_set: &HashSet<usize>,
    literal_deps: &BTreeSet<usize>,
    is_literal: &[bool],
    external_ranges: &mut Vec<(usize, u64, u64)>,
) {
    let producer_gis = find_all_producer_group_indices(group, all_groups);

    for pi in producer_gis {
        if cell_set.contains(&pi) || literal_deps.contains(&pi) {
            continue;
        }
        if is_literal[pi] && all_groups[pi].count < LITERAL_INLINE_THRESHOLD {
            continue;
        }
        let (overlap_lo, overlap_hi) =
            compute_read_range_bounds(group, &all_groups[pi], all_groups);
        if overlap_hi > overlap_lo {
            let prod = &all_groups[pi];
            let offset = overlap_lo - prod.base_id.0;
            let count = overlap_hi - overlap_lo;
            external_ranges.push((pi, offset, count));
        }
    }
}

/// Find all producer group indices for a consumer group (including reduce access).
fn find_all_producer_group_indices(group: &AtomGroup, all_groups: &[AtomGroup]) -> BTreeSet<usize> {
    let mut result = BTreeSet::new();

    for input in &group.inputs {
        for pi in resolve_producer_groups(input, group.count, all_groups) {
            result.insert(pi);
        }
    }

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
                    all_groups,
                ) {
                    result.insert(pi);
                }
            }
        }
        _ => {}
    }

    if let ScalarOp::IndirectLoad { table_base, .. } = &group.op {
        if let Some(pi) = find_group_idx(all_groups, *table_base) {
            result.insert(pi);
        }
    }

    result
}

// ─── Range computation helpers ───────────────────────────────────────────────

/// Compute the [lo, hi) range of atoms from `producer` that `consumer` reads.
fn compute_read_range_bounds(
    consumer: &AtomGroup,
    producer: &AtomGroup,
    all_groups: &[AtomGroup],
) -> (u64, u64) {
    let prod_lo = producer.base_id.0;
    let prod_hi = producer.base_id.0 + producer.count;

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
        _ => (false, 0, 0),
    };

    let mut overall_lo = u64::MAX;
    let mut overall_hi = 0u64;

    for input in &consumer.inputs {
        let (range_lo, range_hi) = input_ref_atom_range(
            input,
            consumer.count,
            is_reduce,
            reduce_count,
            reduce_stride,
        );
        if range_hi <= prod_lo || range_lo >= prod_hi {
            continue;
        }
        let overlap_lo = range_lo.max(prod_lo);
        let overlap_hi = range_hi.min(prod_hi);
        overall_lo = overall_lo.min(overlap_lo);
        overall_hi = overall_hi.max(overlap_hi);
    }

    if overall_lo >= overall_hi {
        (0, 0)
    } else {
        (overall_lo, overall_hi)
    }
}

/// Compute the consumed range of producer by consumer as (offset, count).
fn compute_consumed_range(
    producer_gi: usize,
    consumer_gi: usize,
    groups: &[AtomGroup],
) -> Option<(u64, u64)> {
    let producer = &groups[producer_gi];
    let consumer = &groups[consumer_gi];
    let (lo, hi) = compute_read_range_bounds(consumer, producer, groups);
    if hi > lo {
        let offset = lo - producer.base_id.0;
        Some((offset, hi - lo))
    } else {
        None
    }
}

/// Compute [lo, hi) atom range covered by an InputRef (including reduce expansion).
fn input_ref_atom_range(
    input: &InputRef,
    count: u64,
    is_reduce: bool,
    reduce_count: u64,
    reduce_stride: i64,
) -> (u64, u64) {
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
            if count == 0 {
                return (0, 0);
            }
            let first = base.0 as i64;
            let last = base.0 as i64 + *stride as i64 * (count as i64 - 1);
            let lo = first.min(last) + min_reduce_ext;
            let hi = first.max(last) + max_reduce_ext + 1;
            (lo as u64, hi as u64)
        }
        InputRef::StridedBroadcast {
            base,
            stride,
            repeat,
        } => {
            if count == 0 {
                return (0, 0);
            }
            let last_block = ((count - 1) / repeat) as i64;
            let first_read = base.0 as i64;
            let last_read = base.0 as i64 + stride * last_block;
            let lo = first_read.min(last_read) + min_reduce_ext;
            let hi = first_read.max(last_read) + max_reduce_ext + 1;
            (lo as u64, hi as u64)
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
            let last = base.0 as i64 + *stride as i64 * (*modulus as i64 - 1);
            let lo = first.min(last) + min_reduce_ext;
            let hi = first.max(last) + max_reduce_ext + 1;
            (lo as u64, hi as u64)
        }
        InputRef::SymAffine { base, stride_i, .. } => {
            if count == 0 {
                return (0, 0);
            }
            let first = base.0 as i64;
            let last = base.0 as i64 + *stride_i as i64 * (count as i64 - 1);
            let lo = first.min(last);
            let hi = first.max(last) + 1;
            (lo as u64, hi as u64)
        }
        InputRef::Explicit(ids) => {
            if ids.is_empty() {
                return (0, 0);
            }
            let mut lo = u64::MAX;
            let mut hi = 0u64;
            for id in ids {
                let id_lo = (id.0 as i64 + min_reduce_ext) as u64;
                let id_hi = (id.0 as i64 + max_reduce_ext) as u64 + 1;
                lo = lo.min(id_lo);
                hi = hi.max(id_hi);
            }
            (lo, hi)
        }
    }
}

// ─── Remapping helpers ───────────────────────────────────────────────────────

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
        InputRef::Explicit(ids) => InputRef::Explicit(
            ids.iter()
                .map(|id| atom_map.get(*id).unwrap_or(*id))
                .collect(),
        ),
    }
}

fn remap_sym_dims(
    dims: &[crate::nano_graph::SymDim],
    main_graph: &NanoGraph,
    span_graph: &NanoGraph,
) -> Vec<crate::nano_graph::SymDim> {
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

// ─── Range merging ───────────────────────────────────────────────────────────

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
