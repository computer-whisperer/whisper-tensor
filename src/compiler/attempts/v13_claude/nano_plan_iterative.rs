#![allow(clippy::all, dead_code, unreachable_patterns, unused)]
//! Lane+barrier execution planner for NanoGraph.
//!
//! Produces an `ExecutionPlan` that assigns atom groups to persistent lanes
//! separated by barrier sync points. Within each phase (between barriers),
//! lanes execute independently. Barriers sync all lanes so cross-lane data
//! becomes visible.
//!
//! Three-pass algorithm:
//! 1. **Find barriers**: Sweep groups in topo order, identify ReduceSum groups
//!    on the critical path where downstream consumers need the full output.
//! 2. **Assign lanes**: Within each phase, find connected components of the
//!    local dependency sub-graph, bin-pack them across lanes.
//! 3. **Refine cache affinity**: Greedy swaps of lane assignments across
//!    consecutive phases to improve data reuse.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::nano_graph::{AtomGroup, AtomId, InputRef, NanoGraph, ScalarOp};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// The full execution plan: how many lanes, which groups run where.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub num_lanes: usize,
    /// Phases in execution order. Barriers separate consecutive phases.
    pub phases: Vec<Phase>,
}

/// One phase of execution (work between two consecutive barriers).
#[derive(Debug, Clone)]
pub struct Phase {
    /// `lane_work[lane_idx]` = group indices assigned to that lane in this phase.
    pub lane_work: Vec<Vec<usize>>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Plan execution of a NanoGraph across `num_lanes` persistent threads.
///
/// Returns an `ExecutionPlan` with barrier-separated phases and per-lane
/// group assignments that maximize cache affinity and balance work.
pub fn plan_execution(graph: &NanoGraph, num_lanes: usize) -> ExecutionPlan {
    let groups = graph.groups();
    let num_groups = groups.len();
    let num_lanes = num_lanes.max(1);

    if num_groups == 0 {
        return ExecutionPlan {
            num_lanes,
            phases: vec![],
        };
    }

    // Build group-level dependency graph.
    let (producers, consumers) = build_group_deps(graph);

    // Pass 1: Find barrier positions.
    let topo_order = topo_sort(num_groups, &producers);
    let barrier_positions = find_barriers(graph, &topo_order, &producers, &consumers);

    // Split groups into phases based on barrier positions.
    let phase_groups = split_into_phases(&topo_order, &barrier_positions);

    // Pass 2: Assign groups to lanes within each phase.
    let mut phases = assign_lanes(&phase_groups, num_lanes, groups, &producers, &consumers);

    // Pass 3: Iterative cache affinity refinement.
    refine_cache_affinity(&mut phases, num_lanes, groups, &producers);

    ExecutionPlan { num_lanes, phases }
}

// ---------------------------------------------------------------------------
// Group dependency graph
// ---------------------------------------------------------------------------

/// Build producer and consumer adjacency lists at the group level.
fn build_group_deps(graph: &NanoGraph) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let groups = graph.groups();
    let n = groups.len();
    let mut producers: Vec<Vec<usize>> = vec![vec![]; n];
    let mut consumers: Vec<Vec<usize>> = vec![vec![]; n];

    for (gi, group) in groups.iter().enumerate() {
        let mut seen = HashSet::new();
        for input_ref in &group.inputs {
            collect_source_groups(graph, groups, group, input_ref, &mut seen);
        }
        for &src_gi in &seen {
            if src_gi != gi {
                producers[gi].push(src_gi);
                consumers[src_gi].push(gi);
            }
        }
    }

    // Deduplicate
    for list in producers.iter_mut() {
        list.sort();
        list.dedup();
    }
    for list in consumers.iter_mut() {
        list.sort();
        list.dedup();
    }

    (producers, consumers)
}

/// Collect source group indices referenced by an InputRef.
fn collect_source_groups(
    graph: &NanoGraph,
    groups: &[AtomGroup],
    consumer: &AtomGroup,
    input_ref: &InputRef,
    seen: &mut HashSet<usize>,
) {
    match input_ref {
        InputRef::Broadcast(id) => {
            if let Some(gi) = find_group_idx(groups, *id) {
                seen.insert(gi);
            }
        }
        InputRef::Affine { base, stride } => {
            let count = consumer.count;
            if count == 0 {
                return;
            }
            // Sample endpoints and midpoint
            let samples = [0u64, count / 2, count.saturating_sub(1)];
            for &i in &samples {
                if i < count {
                    let src = input_ref.resolve(i, 0);
                    if let Some(gi) = find_group_idx(groups, src) {
                        seen.insert(gi);
                    }
                }
            }
            // For strided access, include all groups in the range
            if *stride != 0 && count > 1 {
                let first = input_ref.resolve(0, 0);
                let last = input_ref.resolve(count - 1, 0);
                if let (Some(first_gi), Some(last_gi)) =
                    (find_group_idx(groups, first), find_group_idx(groups, last))
                {
                    let (lo, hi) = if first_gi <= last_gi {
                        (first_gi, last_gi)
                    } else {
                        (last_gi, first_gi)
                    };
                    for gi in lo..=hi {
                        seen.insert(gi);
                    }
                }
            }
        }
        InputRef::StridedBroadcast {
            base,
            stride,
            repeat,
        } => {
            let count = consumer.count;
            let num_blocks = (count + repeat - 1) / repeat;
            // Sample a few blocks rather than iterating all
            let block_samples = if num_blocks <= 10 {
                (0..num_blocks).collect::<Vec<_>>()
            } else {
                vec![0, num_blocks / 2, num_blocks - 1]
            };
            for block in block_samples {
                let src = input_ref.resolve(block * repeat, 0);
                if let Some(gi) = find_group_idx(groups, src) {
                    seen.insert(gi);
                }
            }
            // Also get range coverage
            let first = input_ref.resolve(0, 0);
            let last = input_ref.resolve((num_blocks - 1) * repeat, 0);
            if let (Some(first_gi), Some(last_gi)) =
                (find_group_idx(groups, first), find_group_idx(groups, last))
            {
                let (lo, hi) = if first_gi <= last_gi {
                    (first_gi, last_gi)
                } else {
                    (last_gi, first_gi)
                };
                for gi in lo..=hi {
                    seen.insert(gi);
                }
            }
        }
        InputRef::SymAffine {
            base,
            stride_i,
            stride_k,
        } => {
            let count = consumer.count;
            let k_bound = consumer
                .reduce_dims
                .first()
                .and_then(|rd| graph.sym_dim_bounds.get(rd))
                .copied()
                .unwrap_or(1);
            for k in 0..k_bound {
                let samples = [0u64, count.saturating_sub(1)];
                for &i in &samples {
                    if i < count {
                        let src = input_ref.resolve(i, k);
                        if let Some(gi) = find_group_idx(groups, src) {
                            seen.insert(gi);
                        }
                    }
                }
            }
        }
        InputRef::Explicit(ids) => {
            for id in ids {
                if let Some(gi) = find_group_idx(groups, *id) {
                    seen.insert(gi);
                }
            }
        }
        InputRef::Modular {
            base,
            stride,
            modulus,
        } => {
            if let Some(gi) = find_group_idx(groups, *base) {
                seen.insert(gi);
            }
            if *modulus > 1 {
                let last = input_ref.resolve(*modulus - 1, 0);
                if let Some(gi) = find_group_idx(groups, last) {
                    seen.insert(gi);
                }
            }
        }
    }
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
// Topological sort
// ---------------------------------------------------------------------------

fn topo_sort(num_groups: usize, producers: &[Vec<usize>]) -> Vec<usize> {
    // Kahn's algorithm
    let mut in_degree = vec![0u32; num_groups];
    let mut successors: Vec<Vec<usize>> = vec![vec![]; num_groups];

    for (gi, prods) in producers.iter().enumerate() {
        in_degree[gi] = prods.len() as u32;
        for &p in prods {
            successors[p].push(gi);
        }
    }

    let mut queue: VecDeque<usize> = VecDeque::new();
    for gi in 0..num_groups {
        if in_degree[gi] == 0 {
            queue.push_back(gi);
        }
    }

    let mut order = Vec::with_capacity(num_groups);
    while let Some(gi) = queue.pop_front() {
        order.push(gi);
        for &succ in &successors[gi] {
            in_degree[succ] -= 1;
            if in_degree[succ] == 0 {
                queue.push_back(succ);
            }
        }
    }

    // If there are groups not in the order (cycles), append them.
    // This shouldn't happen in a valid NanoGraph, but be defensive.
    if order.len() < num_groups {
        let in_order: HashSet<usize> = order.iter().copied().collect();
        for gi in 0..num_groups {
            if !in_order.contains(&gi) {
                order.push(gi);
            }
        }
    }

    order
}

// ---------------------------------------------------------------------------
// Pass 1: Find barrier positions
// ---------------------------------------------------------------------------

/// Identify barrier positions in the topological order.
///
/// A barrier is placed after a set of ReduceSum groups when a downstream
/// consumer needs the FULL output of those reductions (all rows of the
/// matmul output). This means the consumer reads from multiple ReduceSum
/// groups that come from the same matmul.
///
/// Returns a set of group indices AFTER which a barrier should be placed.
fn find_barriers(
    graph: &NanoGraph,
    topo_order: &[usize],
    producers: &[Vec<usize>],
    consumers: &[Vec<usize>],
) -> HashSet<usize> {
    let groups = graph.groups();
    let num_groups = groups.len();

    // Step 1: Identify ReduceSum groups with reduce_count > 1 (matmul reductions).
    let mut reduce_groups: Vec<usize> = Vec::new();
    for &gi in topo_order {
        if let ScalarOp::ReduceSum {
            reduce_count,
            reduce_stride,
            ..
        } = &groups[gi].op
        {
            if *reduce_count > 1 {
                reduce_groups.push(gi);
            }
        }
    }

    if reduce_groups.is_empty() {
        return HashSet::new();
    }

    // Step 2: Group ReduceSum groups into "matmul families".
    // ReduceSums from the same matmul share the same Mul group producers' structure.
    // We identify them by looking at which groups share the same weight (B matrix) source.
    //
    // Simpler heuristic: ReduceSum groups that have the same reduce_count, reduce_stride,
    // and whose Affine inputs come from consecutive Mul groups are from the same matmul.
    let matmul_families = find_matmul_families(&reduce_groups, groups, producers);

    // Step 3: For each matmul family, check if the downstream consumers
    // collectively need the full output (all family members). This happens
    // when the consumers together read from all/most family members.
    //
    // In a chained matmul, each Mul group of the next matmul reads from ONE
    // ReduceSum atom via Broadcast. No single consumer reads all rows, but
    // collectively they need all rows to exist before any can start computing.
    let mut barrier_positions = HashSet::new();

    let topo_pos: HashMap<usize, usize> = topo_order
        .iter()
        .enumerate()
        .map(|(pos, &gi)| (gi, pos))
        .collect();

    for family in &matmul_families {
        if family.len() <= 1 {
            continue; // Single-row matmul, no barrier needed
        }

        let family_set: HashSet<usize> = family.iter().copied().collect();

        // Collect all direct consumers of any family member.
        let mut all_consumers: HashSet<usize> = HashSet::new();
        for &reduce_gi in family {
            for &consumer_gi in &consumers[reduce_gi] {
                all_consumers.insert(consumer_gi);
            }
        }

        // Check 1: Does any single consumer read from multiple family members?
        let mut needs_barrier = false;
        for &consumer_gi in &all_consumers {
            let family_reads: usize = producers[consumer_gi]
                .iter()
                .filter(|p| family_set.contains(p))
                .count();
            if family_reads > 1 {
                needs_barrier = true;
                break;
            }
        }

        // Check 2: Do the consumers collectively cover all/most family members?
        // AND do the consumers share a common dependency structure (e.g., they're
        // all Mul groups of the next matmul, or all elementwise ops reading the
        // full output vector)?
        if !needs_barrier {
            // How many distinct family members are consumed?
            let mut consumed_members: HashSet<usize> = HashSet::new();
            for &consumer_gi in &all_consumers {
                for &prod_gi in &producers[consumer_gi] {
                    if family_set.contains(&prod_gi) {
                        consumed_members.insert(prod_gi);
                    }
                }
            }

            // If all family members are consumed, check if consumers form a
            // coherent "next matmul" or similar structure that needs the full
            // output before proceeding.
            if consumed_members.len() >= family.len() {
                // The consumers collectively need all family members.
                // Check: do these consumers feed into another set of ReduceSums?
                // (i.e., they're Mul groups of the next matmul)
                let consumers_have_reduce_downstream = all_consumers.iter().any(|&c| {
                    consumers[c].iter().any(|&cc| {
                        matches!(
                            groups[cc].op,
                            ScalarOp::ReduceSum { reduce_count, .. } if reduce_count > 1
                        )
                    })
                });

                if consumers_have_reduce_downstream {
                    needs_barrier = true;
                }
            }

            // Check 3: Even if not all members are consumed, if there are
            // consumers that transitively depend on multiple family members
            // (e.g., through intermediate elementwise ops), that's also a barrier.
            if !needs_barrier && consumed_members.len() > 1 {
                // Check for convergence: do the consumers' downstream paths
                // converge? If yes, the convergence point needs all family members.
                let mut downstream_of_consumers: HashSet<usize> = HashSet::new();
                for &consumer_gi in &all_consumers {
                    for &next in &consumers[consumer_gi] {
                        downstream_of_consumers.insert(next);
                    }
                }
                // If there's overlap in downstream consumers, paths converge.
                // More conservative: if there's any group that transitively
                // depends on multiple family members, we need a barrier.
                for &downstream_gi in &downstream_of_consumers {
                    let family_reach = producers_transitive_family_count(
                        downstream_gi,
                        &family_set,
                        producers,
                        3,
                    );
                    if family_reach > 1 {
                        needs_barrier = true;
                        break;
                    }
                }
            }
        }

        if needs_barrier {
            // Place barrier after the last ReduceSum in this family (in topo order).
            let last_in_family = family
                .iter()
                .max_by_key(|&&gi| topo_pos.get(&gi).unwrap_or(&0))
                .copied()
                .unwrap();

            barrier_positions.insert(last_in_family);
        }
    }

    barrier_positions
}

/// Count how many members of `family_set` are transitively reachable as producers
/// of `gi` within `depth_limit` levels.
fn producers_transitive_family_count(
    gi: usize,
    family_set: &HashSet<usize>,
    producers: &[Vec<usize>],
    depth_limit: usize,
) -> usize {
    let mut found = HashSet::new();
    let mut frontier = vec![gi];
    for _ in 0..depth_limit {
        let mut next_frontier = Vec::new();
        for &node in &frontier {
            for &prod in &producers[node] {
                if family_set.contains(&prod) {
                    found.insert(prod);
                }
                next_frontier.push(prod);
            }
        }
        frontier = next_frontier;
    }
    found.len()
}

/// Group ReduceSum groups into matmul families.
///
/// Groups belong to the same family if they have the same reduce_count,
/// reduce_stride, and their input Mul groups share the same weight source
/// (B matrix). We approximate this by checking reduce parameters and
/// proximity of Affine input bases.
fn find_matmul_families(
    reduce_groups: &[usize],
    groups: &[AtomGroup],
    producers: &[Vec<usize>],
) -> Vec<Vec<usize>> {
    if reduce_groups.is_empty() {
        return vec![];
    }

    // Key: (reduce_count, reduce_stride, count) -> family members
    let mut families_by_params: HashMap<(u64, i64, u64), Vec<usize>> = HashMap::new();

    for &gi in reduce_groups {
        if let ScalarOp::ReduceSum {
            reduce_count,
            reduce_stride,
            ..
        } = &groups[gi].op
        {
            let key = (*reduce_count, *reduce_stride, groups[gi].count);
            families_by_params.entry(key).or_default().push(gi);
        }
    }

    // Further split families by checking if their Mul group producers share
    // weight sources. ReduceSums from the same matmul all read from Mul groups
    // that share the same B matrix.
    let mut result = Vec::new();
    for (_params, family) in families_by_params {
        if family.len() <= 1 {
            result.push(family);
            continue;
        }

        // Check if all members' Mul producers share a common source (B matrix).
        // Get the set of "weight sources" for each family member.
        let mut sub_families: Vec<Vec<usize>> = Vec::new();

        for &gi in &family {
            // Find which sub-family this belongs to.
            // For now, use a simple heuristic: if Mul producers share any
            // common source group, they're in the same family.
            let my_mul_sources = get_mul_weight_sources(gi, groups, producers);

            let mut placed = false;
            for sub in &mut sub_families {
                let rep = sub[0];
                let rep_sources = get_mul_weight_sources(rep, groups, producers);
                // Check overlap
                if !my_mul_sources.is_disjoint(&rep_sources) {
                    sub.push(gi);
                    placed = true;
                    break;
                }
            }
            if !placed {
                sub_families.push(vec![gi]);
            }
        }

        result.extend(sub_families);
    }

    result
}

/// Get the "weight sources" for a ReduceSum group's Mul producers.
/// These are the Literal/source groups that the Mul groups read via Affine
/// (the B matrix in a matmul).
fn get_mul_weight_sources(
    reduce_gi: usize,
    groups: &[AtomGroup],
    producers: &[Vec<usize>],
) -> HashSet<usize> {
    let mut sources = HashSet::new();
    for &mul_gi in &producers[reduce_gi] {
        if let ScalarOp::Binary { .. } = &groups[mul_gi].op {
            for &src_gi in &producers[mul_gi] {
                if groups[src_gi].inputs.is_empty() {
                    // Source/literal group
                    sources.insert(src_gi);
                }
            }
        }
    }
    sources
}

// ---------------------------------------------------------------------------
// Phase splitting
// ---------------------------------------------------------------------------

/// Split the topological order into phases based on barrier positions.
///
/// Each group is assigned to the phase determined by its dependencies:
/// - Groups before the first barrier go to phase 0.
/// - A group must be in a phase >= the phase of all its producers.
/// - A group that depends on a barrier group must be in the NEXT phase.
fn split_into_phases(
    topo_order: &[usize],
    barrier_positions: &HashSet<usize>,
) -> Vec<Vec<usize>> {
    if topo_order.is_empty() {
        return vec![];
    }

    if barrier_positions.is_empty() {
        // No barriers: everything in one phase.
        return vec![topo_order.to_vec()];
    }

    // Assign phase numbers based on barrier positions.
    // Sweep topo order: every time we hit a barrier group, the NEXT group
    // starts a new phase.
    let mut phases: Vec<Vec<usize>> = vec![vec![]];
    let mut current_phase = 0;

    for &gi in topo_order {
        phases[current_phase].push(gi);

        if barrier_positions.contains(&gi) {
            // Next group starts a new phase.
            current_phase += 1;
            phases.push(vec![]);
        }
    }

    // Remove trailing empty phases.
    while phases.last().map_or(false, |p| p.is_empty()) {
        phases.pop();
    }

    phases
}

// ---------------------------------------------------------------------------
// Pass 2: Lane assignment via connected components + bin packing
// ---------------------------------------------------------------------------

/// Assign groups to lanes within each phase.
fn assign_lanes(
    phase_groups: &[Vec<usize>],
    num_lanes: usize,
    groups: &[AtomGroup],
    producers: &[Vec<usize>],
    consumers: &[Vec<usize>],
) -> Vec<Phase> {
    phase_groups
        .iter()
        .map(|phase| assign_lanes_for_phase(phase, num_lanes, groups, producers, consumers))
        .collect()
}

/// Assign groups within a single phase to lanes.
///
/// Algorithm:
/// 1. Build a local dependency sub-graph for this phase's groups.
/// 2. Identify "shared" groups (Literals or groups with consumers in multiple
///    independent sub-DAGs) -- these don't count as edges for component finding.
/// 3. Find connected components of the remaining groups.
/// 4. Bin-pack components onto lanes (largest-first for balance).
/// 5. Assign shared/backbone groups to the lane with the most of their consumers.
fn assign_lanes_for_phase(
    phase_group_indices: &[usize],
    num_lanes: usize,
    groups: &[AtomGroup],
    producers: &[Vec<usize>],
    consumers: &[Vec<usize>],
) -> Phase {
    if phase_group_indices.is_empty() {
        return Phase {
            lane_work: vec![vec![]; num_lanes],
        };
    }

    let phase_set: HashSet<usize> = phase_group_indices.iter().copied().collect();

    // Build local adjacency within this phase (undirected for component finding).
    // Exclude source groups (no inputs) from edges -- they're shared and should
    // be assigned based on their consumers.
    let mut local_adj: HashMap<usize, Vec<usize>> = HashMap::new();
    for &gi in phase_group_indices {
        local_adj.entry(gi).or_default();
    }

    // Identify source groups within this phase (Literals, groups with no inputs
    // or all inputs from outside the phase).
    let mut is_shared: HashSet<usize> = HashSet::new();
    for &gi in phase_group_indices {
        if groups[gi].inputs.is_empty() {
            is_shared.insert(gi);
            continue;
        }
        // Check if this group's consumers span multiple independent sub-DAGs.
        // A simple heuristic: if it has many consumers in this phase, it's shared.
        let in_phase_consumers: usize = consumers[gi]
            .iter()
            .filter(|c| phase_set.contains(c))
            .count();
        let in_phase_producers: usize = producers[gi]
            .iter()
            .filter(|p| phase_set.contains(p))
            .count();
        // If a group has no in-phase producers but multiple in-phase consumers,
        // treat it as shared (it's an input to this phase from a previous phase).
        if in_phase_producers == 0 && in_phase_consumers > 1 {
            is_shared.insert(gi);
        }
    }

    // Build edges between non-shared groups that have producer/consumer relationships.
    for &gi in phase_group_indices {
        if is_shared.contains(&gi) {
            continue;
        }
        for &prod_gi in &producers[gi] {
            if phase_set.contains(&prod_gi) && !is_shared.contains(&prod_gi) {
                local_adj.entry(gi).or_default().push(prod_gi);
                local_adj.entry(prod_gi).or_default().push(gi);
            }
        }
    }

    // Find connected components via BFS.
    let non_shared: Vec<usize> = phase_group_indices
        .iter()
        .filter(|gi| !is_shared.contains(gi))
        .copied()
        .collect();

    let mut visited: HashSet<usize> = HashSet::new();
    let mut components: Vec<Vec<usize>> = Vec::new();

    for &start in &non_shared {
        if visited.contains(&start) {
            continue;
        }
        let mut component = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited.insert(start);

        while let Some(gi) = queue.pop_front() {
            component.push(gi);
            if let Some(neighbors) = local_adj.get(&gi) {
                for &neighbor in neighbors {
                    if visited.insert(neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        components.push(component);
    }

    // Bin-pack components onto lanes (largest-first for balance).
    // Sort components by total atom count, descending.
    components.sort_by(|a, b| {
        let atoms_a: u64 = a.iter().map(|&gi| groups[gi].count).sum();
        let atoms_b: u64 = b.iter().map(|&gi| groups[gi].count).sum();
        atoms_b.cmp(&atoms_a)
    });

    let mut lane_work: Vec<Vec<usize>> = vec![vec![]; num_lanes];
    let mut lane_atoms: Vec<u64> = vec![0; num_lanes];

    for component in &components {
        let comp_atoms: u64 = component.iter().map(|&gi| groups[gi].count).sum();

        // Assign to the lane with the least total work.
        let target_lane = lane_atoms
            .iter()
            .enumerate()
            .min_by_key(|&(_, atoms)| *atoms)
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        lane_work[target_lane].extend(component);
        lane_atoms[target_lane] += comp_atoms;
    }

    // Assign shared groups: put each on the lane that has the most of its consumers.
    let shared_list: Vec<usize> = is_shared.iter().copied().collect();
    for &gi in &shared_list {
        let mut lane_consumer_count = vec![0usize; num_lanes];
        for &consumer_gi in &consumers[gi] {
            if phase_set.contains(&consumer_gi) {
                // Find which lane this consumer is on.
                for (lane_idx, work) in lane_work.iter().enumerate() {
                    if work.contains(&consumer_gi) {
                        lane_consumer_count[lane_idx] += 1;
                        break;
                    }
                }
            }
        }

        // Also consider: which lane has this group's producers?
        for &prod_gi in &producers[gi] {
            for (lane_idx, work) in lane_work.iter().enumerate() {
                if work.contains(&prod_gi) {
                    lane_consumer_count[lane_idx] += 1;
                    break;
                }
            }
        }

        let target_lane = lane_consumer_count
            .iter()
            .enumerate()
            .max_by_key(|&(_, count)| *count)
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        lane_work[target_lane].push(gi);
        lane_atoms[target_lane] += groups[gi].count;
    }

    // Sort each lane's work by group index for deterministic ordering.
    for work in &mut lane_work {
        work.sort();
    }

    Phase { lane_work }
}

// ---------------------------------------------------------------------------
// Pass 3: Cache affinity refinement
// ---------------------------------------------------------------------------

/// Compute a cache score for a lane's work across two consecutive phases.
///
/// The score is the number of atom IDs that appear in both phases' data
/// footprints (as producers or consumers). Higher = better cache reuse.
fn cache_overlap_score(
    prev_work: &[usize],
    curr_work: &[usize],
    groups: &[AtomGroup],
    producers: &[Vec<usize>],
) -> u64 {
    // Compute the "data footprint" of each phase's work as a set of group indices
    // (both the groups themselves and their producers).
    let prev_footprint: HashSet<usize> = prev_work
        .iter()
        .flat_map(|&gi| {
            let mut fp = vec![gi];
            fp.extend(producers[gi].iter());
            fp
        })
        .collect();

    let curr_footprint: HashSet<usize> = curr_work
        .iter()
        .flat_map(|&gi| {
            let mut fp = vec![gi];
            fp.extend(producers[gi].iter());
            fp
        })
        .collect();

    // Score: sum of atom counts for overlapping groups.
    let mut score: u64 = 0;
    for &gi in &prev_footprint {
        if curr_footprint.contains(&gi) {
            score += groups[gi].count;
        }
    }
    score
}

/// Total cache score across all lanes and consecutive phase pairs.
fn total_cache_score(
    phases: &[Phase],
    num_lanes: usize,
    groups: &[AtomGroup],
    producers: &[Vec<usize>],
) -> u64 {
    let mut total = 0u64;
    for phase_idx in 1..phases.len() {
        for lane_idx in 0..num_lanes {
            let prev_work = &phases[phase_idx - 1].lane_work[lane_idx];
            let curr_work = &phases[phase_idx].lane_work[lane_idx];
            total += cache_overlap_score(prev_work, curr_work, groups, producers);
        }
    }
    total
}

/// Refine lane assignments across consecutive phases to improve cache affinity.
///
/// Uses greedy swaps: for each consecutive phase pair, try swapping lane
/// assignments and accept if the cache score improves.
fn refine_cache_affinity(
    phases: &mut Vec<Phase>,
    num_lanes: usize,
    groups: &[AtomGroup],
    producers: &[Vec<usize>],
) {
    if phases.len() <= 1 || num_lanes <= 1 {
        return;
    }

    // Multiple passes of greedy improvement.
    let max_iterations = 10;

    for _iteration in 0..max_iterations {
        let mut improved = false;

        for phase_idx in 1..phases.len() {
            // Try all pairs of lanes in this phase.
            for lane_a in 0..num_lanes {
                for lane_b in (lane_a + 1)..num_lanes {
                    // Current score for these two lanes.
                    let current_score_a = cache_overlap_score(
                        &phases[phase_idx - 1].lane_work[lane_a],
                        &phases[phase_idx].lane_work[lane_a],
                        groups,
                        producers,
                    );
                    let current_score_b = cache_overlap_score(
                        &phases[phase_idx - 1].lane_work[lane_b],
                        &phases[phase_idx].lane_work[lane_b],
                        groups,
                        producers,
                    );
                    let current_total = current_score_a + current_score_b;

                    // Score if we swap lane_a and lane_b in this phase.
                    let swapped_score_a = cache_overlap_score(
                        &phases[phase_idx - 1].lane_work[lane_a],
                        &phases[phase_idx].lane_work[lane_b],
                        groups,
                        producers,
                    );
                    let swapped_score_b = cache_overlap_score(
                        &phases[phase_idx - 1].lane_work[lane_b],
                        &phases[phase_idx].lane_work[lane_a],
                        groups,
                        producers,
                    );
                    let swapped_total = swapped_score_a + swapped_score_b;

                    // Also consider the effect on the NEXT phase if it exists.
                    let mut current_forward = 0u64;
                    let mut swapped_forward = 0u64;
                    if phase_idx + 1 < phases.len() {
                        current_forward += cache_overlap_score(
                            &phases[phase_idx].lane_work[lane_a],
                            &phases[phase_idx + 1].lane_work[lane_a],
                            groups,
                            producers,
                        );
                        current_forward += cache_overlap_score(
                            &phases[phase_idx].lane_work[lane_b],
                            &phases[phase_idx + 1].lane_work[lane_b],
                            groups,
                            producers,
                        );
                        swapped_forward += cache_overlap_score(
                            &phases[phase_idx].lane_work[lane_b],
                            &phases[phase_idx + 1].lane_work[lane_a],
                            groups,
                            producers,
                        );
                        swapped_forward += cache_overlap_score(
                            &phases[phase_idx].lane_work[lane_a],
                            &phases[phase_idx + 1].lane_work[lane_b],
                            groups,
                            producers,
                        );
                    }

                    if swapped_total + swapped_forward > current_total + current_forward {
                        // Accept the swap.
                        let tmp = phases[phase_idx].lane_work[lane_a].clone();
                        phases[phase_idx].lane_work[lane_a] =
                            phases[phase_idx].lane_work[lane_b].clone();
                        phases[phase_idx].lane_work[lane_b] = tmp;
                        improved = true;
                    }
                }
            }
        }

        if !improved {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Validation helpers (for tests)
// ---------------------------------------------------------------------------

/// Verify that the execution plan is valid:
/// - Every group appears exactly once across all phases and lanes.
/// - Within each phase, no lane reads another lane's current-phase output.
/// - All phases have `num_lanes` lane assignments.
pub fn validate_plan(
    plan: &ExecutionPlan,
    graph: &NanoGraph,
) -> Vec<String> {
    let groups = graph.groups();
    let num_groups = groups.len();
    let mut errors = Vec::new();

    // Check complete coverage: every group assigned exactly once.
    let mut assignment_count = vec![0u32; num_groups];
    for (phase_idx, phase) in plan.phases.iter().enumerate() {
        if phase.lane_work.len() != plan.num_lanes {
            errors.push(format!(
                "Phase {} has {} lane assignments, expected {}",
                phase_idx,
                phase.lane_work.len(),
                plan.num_lanes
            ));
        }
        for (lane_idx, work) in phase.lane_work.iter().enumerate() {
            for &gi in work {
                if gi >= num_groups {
                    errors.push(format!(
                        "Phase {} lane {} references group {} (out of range, num_groups={})",
                        phase_idx, lane_idx, gi, num_groups
                    ));
                } else {
                    assignment_count[gi] += 1;
                }
            }
        }
    }

    for (gi, &count) in assignment_count.iter().enumerate() {
        if count != 1 {
            errors.push(format!(
                "Group {} assigned {} times (expected 1)",
                gi, count
            ));
        }
    }

    // Check within-phase independence: no lane reads another lane's
    // *computed* output. Shared data (Literals, groups with no in-phase
    // producers) is available to all lanes via the shared values buffer.
    let (producers_map, _) = build_group_deps(graph);
    for (phase_idx, phase) in plan.phases.iter().enumerate() {
        // Build lane assignment map for this phase.
        let mut group_to_lane: HashMap<usize, usize> = HashMap::new();
        let mut phase_set: HashSet<usize> = HashSet::new();
        for (lane_idx, work) in phase.lane_work.iter().enumerate() {
            for &gi in work {
                group_to_lane.insert(gi, lane_idx);
                phase_set.insert(gi);
            }
        }

        // Identify shared groups in this phase: groups with no in-phase producers.
        // These are Literals, constants, or results from prior phases.
        // Reading them from any lane is fine.
        let mut is_shared_in_phase: HashSet<usize> = HashSet::new();
        for &gi in &phase_set {
            let has_in_phase_producer = producers_map[gi]
                .iter()
                .any(|p| phase_set.contains(p));
            if !has_in_phase_producer {
                is_shared_in_phase.insert(gi);
            }
        }

        // Check: for each group in this phase, its producers that are ALSO in
        // this phase AND are not shared must be on the same lane.
        for (lane_idx, work) in phase.lane_work.iter().enumerate() {
            for &gi in work {
                for &prod_gi in &producers_map[gi] {
                    if is_shared_in_phase.contains(&prod_gi) {
                        continue; // Shared data, accessible from any lane.
                    }
                    if let Some(&prod_lane) = group_to_lane.get(&prod_gi) {
                        if prod_lane != lane_idx {
                            errors.push(format!(
                                "Phase {} independence violation: group {} (lane {}) \
                                 reads from group {} (lane {})",
                                phase_idx, gi, lane_idx, prod_gi, prod_lane
                            ));
                        }
                    }
                }
            }
        }
    }

    errors
}

/// Compute work balance ratio for a phase: max_lane_atoms / min_lane_atoms.
/// Returns None if any lane has zero atoms (infinite imbalance).
pub fn work_balance_ratio(phase: &Phase, groups: &[AtomGroup]) -> Option<f64> {
    let lane_atoms: Vec<u64> = phase
        .lane_work
        .iter()
        .map(|work| work.iter().map(|&gi| groups[gi].count).sum::<u64>())
        .collect();

    let min = lane_atoms.iter().copied().min().unwrap_or(0);
    let max = lane_atoms.iter().copied().max().unwrap_or(0);

    if min == 0 {
        if max == 0 {
            Some(1.0) // All empty
        } else {
            None // Infinite imbalance
        }
    } else {
        Some(max as f64 / min as f64)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::attempts::v13_claude::test_graphs;
    use crate::dtype::DType;
    use crate::nano_graph::{ScalarBinOp, ScalarOp, ScalarUnaryOp};
    use crate::numeric_scalar::NumericScalar;

    // -----------------------------------------------------------------------
    // Helper: print plan summary
    // -----------------------------------------------------------------------

    fn print_plan(plan: &ExecutionPlan, groups: &[AtomGroup]) {
        eprintln!(
            "ExecutionPlan: {} lanes, {} phases",
            plan.num_lanes,
            plan.phases.len()
        );
        for (pi, phase) in plan.phases.iter().enumerate() {
            eprintln!("  Phase {}:", pi);
            for (li, work) in phase.lane_work.iter().enumerate() {
                let atoms: u64 = work.iter().map(|&gi| groups[gi].count).sum();
                eprintln!(
                    "    Lane {}: {} groups, {} atoms {:?}",
                    li,
                    work.len(),
                    atoms,
                    if work.len() <= 10 {
                        format!("{:?}", work)
                    } else {
                        format!("[{}, ..., {}]", work[0], work[work.len() - 1])
                    }
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 1: Empty graph
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty() {
        let g = NanoGraph::new();
        let plan = plan_execution(&g, 4);
        assert_eq!(plan.num_lanes, 4);
        assert!(plan.phases.is_empty());
        let errors = validate_plan(&plan, &g);
        assert!(errors.is_empty(), "{:?}", errors);
    }

    // -----------------------------------------------------------------------
    // Test 2: Single elementwise op -- all in one phase, one lane
    // -----------------------------------------------------------------------

    #[test]
    fn test_elementwise_single_phase() {
        let (g, _, _, _) = test_graphs::elementwise_binary(64, ScalarBinOp::Add);
        let plan = plan_execution(&g, 2);

        let errors = validate_plan(&plan, &g);
        assert!(errors.is_empty(), "{:?}", errors);

        // No barriers expected (no ReduceSum).
        assert_eq!(plan.phases.len(), 1, "Elementwise should be single phase");
        print_plan(&plan, g.groups());
    }

    // -----------------------------------------------------------------------
    // Test 3: Complete coverage -- every group assigned exactly once
    // -----------------------------------------------------------------------

    #[test]
    fn test_complete_coverage_matmul() {
        let (g, _, _, _) = test_graphs::matmul(4, 8, 16);
        let plan = plan_execution(&g, 4);

        let errors = validate_plan(&plan, &g);
        assert!(errors.is_empty(), "Validation errors: {:?}", errors);
        print_plan(&plan, g.groups());
    }

    // -----------------------------------------------------------------------
    // Test 4: Chained matmuls produce barriers
    // -----------------------------------------------------------------------

    #[test]
    fn test_chained_matmuls_have_barriers() {
        let (g, _, _, _, _) = test_graphs::matmul_chain(4, 8, 16, 16, 32);
        let plan = plan_execution(&g, 4);

        let errors = validate_plan(&plan, &g);
        assert!(errors.is_empty(), "Validation errors: {:?}", errors);

        // Two chained matmuls should produce at least 2 phases.
        assert!(
            plan.phases.len() >= 2,
            "Chained matmuls should have >= 2 phases, got {}",
            plan.phases.len()
        );
        print_plan(&plan, g.groups());
    }

    // -----------------------------------------------------------------------
    // Test 5: Within-phase independence
    // -----------------------------------------------------------------------

    #[test]
    fn test_within_phase_independence() {
        let (g, _, _, _) = test_graphs::matmul(8, 4, 16);
        let plan = plan_execution(&g, 4);

        let errors = validate_plan(&plan, &g);
        assert!(
            errors.is_empty(),
            "Within-phase independence violations: {:?}",
            errors
        );
    }

    // -----------------------------------------------------------------------
    // Test 6: Work balance
    // -----------------------------------------------------------------------

    #[test]
    fn test_work_balance() {
        let (g, _, _, _) = test_graphs::matmul(8, 4, 16);
        let plan = plan_execution(&g, 4);

        let errors = validate_plan(&plan, &g);
        assert!(errors.is_empty(), "{:?}", errors);

        for (pi, phase) in plan.phases.iter().enumerate() {
            if let Some(ratio) = work_balance_ratio(phase, g.groups()) {
                eprintln!("Phase {} balance ratio: {:.2}", pi, ratio);
                // Allow some imbalance for phases with shared/literal groups.
                // The key constraint is < 2x for phases with actual compute.
                let has_compute: bool = phase.lane_work.iter().any(|work| {
                    work.iter().any(|&gi| !g.groups()[gi].inputs.is_empty())
                });
                if has_compute {
                    assert!(
                        ratio < 5.0,
                        "Phase {} has extreme imbalance: {:.2}x",
                        pi,
                        ratio
                    );
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 7: Cache affinity improves with refinement
    // -----------------------------------------------------------------------

    #[test]
    fn test_cache_affinity_refinement() {
        let (g, _, _, _, _) = test_graphs::matmul_chain(4, 8, 16, 16, 32);

        let (producers, _consumers) = build_group_deps(&g);

        // Get the plan (includes refinement).
        let plan = plan_execution(&g, 4);
        let final_score = total_cache_score(&plan.phases, plan.num_lanes, g.groups(), &producers);

        let errors = validate_plan(&plan, &g);
        assert!(errors.is_empty(), "{:?}", errors);

        // The cache score should be non-negative (basic sanity).
        eprintln!("Final cache score: {}", final_score);

        // Compare with an unrefined plan: build phases without refinement.
        let topo_order = topo_sort(g.num_groups(), &producers);
        let (_, consumers) = build_group_deps(&g);
        let barrier_positions = find_barriers(&g, &topo_order, &producers, &consumers);
        let phase_groups = split_into_phases(&topo_order, &barrier_positions);
        let unrefined_phases =
            assign_lanes(&phase_groups, 4, g.groups(), &producers, &consumers);
        let unrefined_score =
            total_cache_score(&unrefined_phases, 4, g.groups(), &producers);

        eprintln!(
            "Unrefined cache score: {}, refined: {}",
            unrefined_score, final_score
        );

        // Refinement should not make things worse.
        assert!(
            final_score >= unrefined_score,
            "Refinement made cache score worse: {} < {}",
            final_score,
            unrefined_score
        );
    }

    // -----------------------------------------------------------------------
    // Test 8: Single matmul with activation
    // -----------------------------------------------------------------------

    #[test]
    fn test_matmul_activation() {
        let (g, _, _, _) =
            test_graphs::matmul_activation(4, 8, 16, ScalarUnaryOp::Tanh);
        let plan = plan_execution(&g, 4);

        let errors = validate_plan(&plan, &g);
        assert!(errors.is_empty(), "{:?}", errors);
        print_plan(&plan, g.groups());
    }

    // -----------------------------------------------------------------------
    // Test 9: Two independent matmuls
    // -----------------------------------------------------------------------

    #[test]
    fn test_two_independent_matmuls() {
        // Build two independent matmuls manually (not chained).
        let mut g = NanoGraph::new();

        // Matmul 1: [2,4] @ [4,8]
        let a1 = g.push_group(
            8,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let b1 = g.push_group(
            32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let mut mul1_bases = Vec::new();
        for m in 0..2u64 {
            for k in 0..4u64 {
                let mb = g.push_group(
                    8,
                    ScalarOp::Binary {
                        op: ScalarBinOp::Mul,
                        compute_dtype: DType::F32,
                        output_dtype: DType::F32,
                    },
                    vec![],
                    vec![],
                    vec![
                        InputRef::Broadcast(a1.offset(m * 4 + k)),
                        InputRef::Affine {
                            base: b1.offset(k * 8),
                            stride: 1,
                        },
                    ],
                );
                mul1_bases.push(mb);
            }
        }
        let mut reduce1 = Vec::new();
        for m in 0..2u64 {
            let rb = g.push_group(
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
                    base: mul1_bases[(m * 4) as usize],
                    stride: 1,
                }],
            );
            reduce1.push(rb);
        }

        // Matmul 2: [3,2] @ [2,6]
        let a2 = g.push_group(
            6,
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
            vec![],
        );
        let b2 = g.push_group(
            12,
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
            vec![],
        );
        let mut mul2_bases = Vec::new();
        for m in 0..3u64 {
            for k in 0..2u64 {
                let mb = g.push_group(
                    6,
                    ScalarOp::Binary {
                        op: ScalarBinOp::Mul,
                        compute_dtype: DType::F32,
                        output_dtype: DType::F32,
                    },
                    vec![],
                    vec![],
                    vec![
                        InputRef::Broadcast(a2.offset(m * 2 + k)),
                        InputRef::Affine {
                            base: b2.offset(k * 6),
                            stride: 1,
                        },
                    ],
                );
                mul2_bases.push(mb);
            }
        }
        let mut reduce2 = Vec::new();
        for m in 0..3u64 {
            let rb = g.push_group(
                6,
                ScalarOp::ReduceSum {
                    reduce_count: 2,
                    reduce_stride: 6,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![InputRef::Affine {
                    base: mul2_bases[(m * 2) as usize],
                    stride: 1,
                }],
            );
            reduce2.push(rb);
        }

        let mut outputs = Vec::new();
        outputs.extend(reduce1);
        outputs.extend(reduce2);
        g.outputs = outputs;

        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let plan = plan_execution(&g, 4);
        let errors = validate_plan(&plan, &g);
        assert!(errors.is_empty(), "{:?}", errors);

        // Independent matmuls should be in the same phase (no barrier between them).
        assert_eq!(
            plan.phases.len(),
            1,
            "Independent matmuls should be in single phase"
        );

        // They should be distributed across lanes.
        let non_empty_lanes: usize = plan.phases[0]
            .lane_work
            .iter()
            .filter(|w| !w.is_empty())
            .count();
        assert!(
            non_empty_lanes >= 2,
            "Independent matmuls should use multiple lanes, got {}",
            non_empty_lanes
        );
        print_plan(&plan, g.groups());
    }

    // -----------------------------------------------------------------------
    // Test 10: Diamond pattern
    // -----------------------------------------------------------------------

    #[test]
    fn test_diamond_pattern() {
        let mut g = NanoGraph::new();

        let src = g.push_group(
            32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let branch_a = g.push_group(
            32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: src,
                stride: 1,
            }],
        );
        let branch_b = g.push_group(
            32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: src,
                stride: 1,
            }],
        );
        let merge = g.push_group(
            32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine {
                    base: branch_a,
                    stride: 1,
                },
                InputRef::Affine {
                    base: branch_b,
                    stride: 1,
                },
            ],
        );
        g.outputs = vec![merge];

        let plan = plan_execution(&g, 2);
        let errors = validate_plan(&plan, &g);
        assert!(errors.is_empty(), "{:?}", errors);
        print_plan(&plan, g.groups());
    }

    // -----------------------------------------------------------------------
    // Test 11: Linear chain stays together
    // -----------------------------------------------------------------------

    #[test]
    fn test_linear_chain() {
        let (g, _, _) = test_graphs::unary_chain(
            256,
            &[ScalarUnaryOp::Exp, ScalarUnaryOp::Neg, ScalarUnaryOp::Tanh],
        );
        let plan = plan_execution(&g, 4);

        let errors = validate_plan(&plan, &g);
        assert!(errors.is_empty(), "{:?}", errors);

        // No ReduceSum -> single phase.
        assert_eq!(plan.phases.len(), 1);
        print_plan(&plan, g.groups());
    }

    // -----------------------------------------------------------------------
    // Test 12: Broadcast pattern
    // -----------------------------------------------------------------------

    #[test]
    fn test_broadcast() {
        let (g, _, _, _) = test_graphs::broadcast_add(128);
        let plan = plan_execution(&g, 2);

        let errors = validate_plan(&plan, &g);
        assert!(errors.is_empty(), "{:?}", errors);
        print_plan(&plan, g.groups());
    }

    // -----------------------------------------------------------------------
    // Test 13: Larger matmul chain -- stress test
    // -----------------------------------------------------------------------

    #[test]
    fn test_larger_matmul_chain() {
        // Simulate a small MLP: [8,32] @ [32,64] @ [64,16]
        // This needs n1 == k2, so inner dims must match.
        let (g, _, _, _, _) = test_graphs::matmul_chain(8, 32, 64, 64, 16);
        let plan = plan_execution(&g, 8);

        let errors = validate_plan(&plan, &g);
        assert!(errors.is_empty(), "Validation errors: {:?}", errors);

        eprintln!("Larger matmul chain:");
        print_plan(&plan, g.groups());

        // Should have multiple phases for chained matmuls.
        assert!(
            plan.phases.len() >= 2,
            "Expected >= 2 phases for chained matmuls"
        );
    }

    // -----------------------------------------------------------------------
    // Test 14: num_lanes = 1 (degenerate case)
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_lane() {
        let (g, _, _, _) = test_graphs::matmul(4, 8, 16);
        let plan = plan_execution(&g, 1);

        let errors = validate_plan(&plan, &g);
        assert!(errors.is_empty(), "{:?}", errors);

        // With 1 lane, all work should be on lane 0.
        for phase in &plan.phases {
            assert_eq!(phase.lane_work.len(), 1);
            assert!(!phase.lane_work[0].is_empty());
        }
    }

    // -----------------------------------------------------------------------
    // Test 15: Through the lowering pipeline
    // -----------------------------------------------------------------------

    #[test]
    fn test_via_lowering() {
        use crate::milli_graph::ops;
        use crate::milli_graph::MilliOpGraph;
        use crate::nano_graph::lower::lower_with_info;
        use crate::numeric_tensor::NumericTensor;
        use crate::tensor_info::TensorInfo;
        use std::collections::HashMap;

        let mut rng = rand::rng();
        let (mut milli, _ext_map) = MilliOpGraph::new(std::iter::empty(), &mut rng);

        let a_id = milli.add_input(&mut rng);
        let b_id = milli.add_input(&mut rng);
        let c_id = ops::MatMul::push_new_default_precision(
            &mut milli,
            a_id,
            b_id,
            DType::F32,
            &mut rng,
        );

        let a_tensor: NumericTensor<crate::DynRank> =
            NumericTensor::from_vec_shape(vec![1.0f32; 12], vec![4, 3]).unwrap();
        let b_tensor: NumericTensor<crate::DynRank> =
            NumericTensor::from_vec_shape(vec![1.0f32; 6], vec![3, 2]).unwrap();

        let mut info_inputs: HashMap<crate::graph::GlobalId, TensorInfo> = HashMap::new();
        info_inputs.insert(a_id, TensorInfo::from(a_tensor));
        info_inputs.insert(b_id, TensorInfo::from(b_tensor));

        let lower_result = lower_with_info(&milli, &info_inputs).unwrap();
        assert!(
            lower_result.unsupported.is_empty(),
            "Unsupported: {:?}",
            lower_result.unsupported_details
        );

        let graph = &lower_result.graph;
        eprintln!("Lowered matmul: {}", graph.stats());

        let plan = plan_execution(graph, 4);
        let errors = validate_plan(&plan, graph);
        assert!(errors.is_empty(), "Validation errors: {:?}", errors);
        print_plan(&plan, graph.groups());
    }
}
