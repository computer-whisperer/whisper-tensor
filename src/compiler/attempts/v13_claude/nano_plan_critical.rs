#![allow(clippy::all, dead_code, unreachable_patterns)]
//! Lane+barrier execution planner using critical path + slack analysis.
//!
//! The planner produces an `ExecutionPlan` that assigns atom groups to lanes
//! across barrier-separated phases. The critical path determines barrier
//! positions (at matmul reduction boundaries), and off-critical-path groups
//! are distributed across lanes for parallelism.
//!
//! See `src/compiler/problem_shape.md` "Lanes, Phases, and Barriers" for the
//! full execution model.

use std::collections::{HashMap, HashSet};

use crate::nano_graph::{AtomId, InputRef, NanoGraph, ScalarOp};

/// The execution plan: lanes and barrier-separated phases.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Number of lanes (persistent threads).
    pub num_lanes: usize,
    /// Phases separated by barriers. Each phase contains per-lane work.
    pub phases: Vec<Phase>,
}

/// One phase of execution (work between two consecutive barriers).
#[derive(Debug, Clone)]
pub struct Phase {
    /// For each lane index, the group indices assigned to that lane in this phase.
    /// `lane_work[lane_idx]` = list of group indices.
    pub lane_work: Vec<Vec<usize>>,
}

/// Plan execution for a NanoGraph.
///
/// Produces an `ExecutionPlan` with the given number of lanes. The critical
/// path is assigned to lane 0, and off-critical-path parallel work is
/// distributed across lanes 1..num_lanes.
pub fn plan_execution(graph: &NanoGraph, num_lanes: usize) -> ExecutionPlan {
    let groups = graph.groups();
    let num_groups = groups.len();

    if num_groups == 0 || num_lanes == 0 {
        return ExecutionPlan {
            num_lanes: num_lanes.max(1),
            phases: vec![],
        };
    }

    // Step 1: Build group-level dependency DAG.
    let (predecessors, successors) = build_group_dag(graph);

    // Step 2: Identify which groups are Literals (no predecessors, no compute).
    let is_literal: Vec<bool> = groups
        .iter()
        .map(|g| matches!(g.op, ScalarOp::Literal(_)))
        .collect();

    // Step 3: Compute depth (longest path from any root) for each group.
    // depth[g] = max(depth[pred] + pred.count) for all predecessors, treating
    // Literal groups as depth 0 with zero cost.
    let (depth, earliest_finish) = compute_depths(groups, &predecessors, &is_literal);

    // Step 4: Find the critical path (longest path through the DAG).
    let critical_path = find_critical_path(groups, &predecessors, &successors, &depth, &earliest_finish, &is_literal);
    let critical_set: HashSet<usize> = critical_path.iter().copied().collect();

    // Step 5: Identify barrier positions from critical-path ReduceSums that
    // read from matmul Mul groups (StridedBroadcast input).
    let barrier_positions = find_barrier_positions(graph, &critical_path);

    // Step 6: Split all groups into phases based on barrier positions.
    // A phase is the set of groups that can execute between two barriers.
    let phase_assignments = assign_phases(
        graph,
        &predecessors,
        &critical_path,
        &barrier_positions,
        &is_literal,
    );

    // Step 7: Compute slack for each group.
    let latest_start = compute_latest_starts(groups, &successors, &earliest_finish, &is_literal);
    let earliest_start: Vec<u64> = (0..num_groups)
        .map(|g| if earliest_finish[g] >= groups[g].count { earliest_finish[g] - groups[g].count } else { 0 })
        .collect();
    let slack: Vec<u64> = (0..num_groups)
        .map(|g| latest_start[g].saturating_sub(earliest_start[g]))
        .collect();

    // Step 8: Build phases with lane assignments.
    let num_phases = phase_assignments.iter().copied().max().map(|m| m + 1).unwrap_or(1);
    let mut phases: Vec<Phase> = (0..num_phases)
        .map(|_| Phase {
            lane_work: vec![vec![]; num_lanes],
        })
        .collect();

    // Collect groups per phase (excluding Literals).
    let mut groups_per_phase: Vec<Vec<usize>> = vec![vec![]; num_phases];
    for gi in 0..num_groups {
        if !is_literal[gi] {
            let phase = phase_assignments[gi];
            groups_per_phase[phase].push(gi);
        }
    }

    // For each phase, assign critical-path groups to lane 0, then expand
    // lane 0 to include all same-phase predecessors (transitively), then
    // distribute the rest.
    for phase_idx in 0..num_phases {
        let phase_groups = &groups_per_phase[phase_idx];
        let phase_set: HashSet<usize> = phase_groups.iter().copied().collect();

        // Start with critical-path groups on lane 0.
        let mut lane0_set: HashSet<usize> = phase_groups
            .iter()
            .copied()
            .filter(|gi| critical_set.contains(gi))
            .collect();

        // Expand lane 0: any same-phase predecessor of a lane 0 group must
        // also be on lane 0 (transitively).
        let mut changed = true;
        while changed {
            changed = false;
            let current: Vec<usize> = lane0_set.iter().copied().collect();
            for gi in current {
                for &pred in &predecessors[gi] {
                    if phase_set.contains(&pred) && !lane0_set.contains(&pred) {
                        lane0_set.insert(pred);
                        changed = true;
                    }
                }
            }
        }

        // Also expand lane 0: any same-phase successor of a lane 0 group must
        // be on lane 0 (if a lane 0 group produces data consumed by another
        // group in the same phase, that consumer must be on the same lane).
        changed = true;
        while changed {
            changed = false;
            let current: Vec<usize> = lane0_set.iter().copied().collect();
            for gi in current {
                for &succ in &successors[gi] {
                    if phase_set.contains(&succ) && !lane0_set.contains(&succ) {
                        lane0_set.insert(succ);
                        changed = true;
                    }
                }
            }
        }

        let mut lane0_groups: Vec<usize> = lane0_set.iter().copied().collect();
        lane0_groups.sort(); // Maintain topological-ish order.

        let parallel_groups: Vec<usize> = phase_groups
            .iter()
            .copied()
            .filter(|gi| !lane0_set.contains(gi))
            .collect();

        // Lane 0 gets the critical path and its same-phase dependencies.
        phases[phase_idx].lane_work[0].extend(&lane0_groups);

        if num_lanes == 1 {
            // Only one lane, everything goes there.
            phases[phase_idx].lane_work[0].extend(&parallel_groups);
        } else {
            // Distribute parallel groups across lanes 1..num_lanes.
            distribute_parallel_groups(
                graph,
                &parallel_groups,
                &mut phases,
                phase_idx,
                num_lanes,
                &predecessors,
            );
        }
    }

    // Literal groups are NOT assigned to any lane — they're pre-existing shared
    // data (weights, constants) available to all lanes from the start.

    ExecutionPlan { num_lanes, phases }
}

/// Build predecessor and successor lists for the group DAG.
/// predecessors[g] = set of group indices that g reads from.
/// successors[g] = set of group indices that read from g.
fn build_group_dag(graph: &NanoGraph) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let groups = graph.groups();
    let num_groups = groups.len();

    let mut predecessors: Vec<Vec<usize>> = vec![vec![]; num_groups];
    let mut successors: Vec<Vec<usize>> = vec![vec![]; num_groups];

    for (gi, group) in groups.iter().enumerate() {
        let mut pred_set = HashSet::new();
        for input in &group.inputs {
            // Sample atoms to find which groups are referenced.
            let source_ids = sample_source_atoms(input, group.count);
            for src_id in source_ids {
                if let Some(src_gi) = find_group_index(groups, src_id) {
                    if src_gi != gi {
                        pred_set.insert(src_gi);
                    }
                }
            }
        }
        let preds: Vec<usize> = pred_set.into_iter().collect();
        for &p in &preds {
            successors[p].push(gi);
        }
        predecessors[gi] = preds;
    }

    // Deduplicate successors.
    for s in &mut successors {
        s.sort();
        s.dedup();
    }

    (predecessors, successors)
}

/// Sample source atoms from an InputRef to determine which groups it references.
fn sample_source_atoms(input: &InputRef, count: u64) -> Vec<AtomId> {
    match input {
        InputRef::Broadcast(id) => vec![*id],
        InputRef::Affine { base, stride } => {
            // Sample first, last, and a few middle atoms.
            let mut ids = vec![*base];
            if count > 1 {
                ids.push(AtomId(base.0.wrapping_add((*stride as i64 * (count as i64 - 1)) as u64)));
            }
            // Middle sample for large groups.
            if count > 2 {
                let mid = count / 2;
                ids.push(AtomId(base.0.wrapping_add((*stride as i64 * mid as i64) as u64)));
            }
            ids
        }
        InputRef::Explicit(ids) => {
            // Sample a few entries.
            let mut sampled = vec![ids[0]];
            if ids.len() > 1 {
                sampled.push(*ids.last().unwrap());
            }
            if ids.len() > 2 {
                sampled.push(ids[ids.len() / 2]);
            }
            sampled
        }
        InputRef::SymAffine { base, stride_i, stride_k } => {
            vec![*base]
        }
        InputRef::StridedBroadcast { base, stride, repeat } => {
            // Sample first and last block sources.
            let mut ids = vec![*base];
            if count > *repeat {
                let last_block = (count - 1) / repeat;
                ids.push(AtomId(base.0.wrapping_add((*stride * last_block as i64) as u64)));
            }
            ids
        }
        InputRef::Modular { base, stride, modulus } => {
            let mut ids = vec![*base];
            if *modulus > 1 {
                ids.push(AtomId(base.0.wrapping_add((*stride as i64 * (*modulus as i64 - 1)) as u64)));
            }
            ids
        }
    }
}

/// Find the group index containing a given AtomId via binary search.
fn find_group_index(groups: &[crate::nano_graph::AtomGroup], id: AtomId) -> Option<usize> {
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

/// Compute depth (earliest finish time) for each group.
/// Returns (depth, earliest_finish) where earliest_finish[g] = earliest time
/// group g finishes, and depth[g] = earliest_start[g] = earliest_finish[g] - cost[g].
fn compute_depths(
    groups: &[crate::nano_graph::AtomGroup],
    predecessors: &[Vec<usize>],
    is_literal: &[bool],
) -> (Vec<u64>, Vec<u64>) {
    let n = groups.len();
    let mut earliest_finish = vec![0u64; n];

    // Topological order via Kahn's algorithm.
    let mut in_degree: Vec<usize> = vec![0; n];
    for gi in 0..n {
        in_degree[gi] = predecessors[gi].len();
    }

    let mut queue: Vec<usize> = Vec::new();
    for gi in 0..n {
        if in_degree[gi] == 0 {
            queue.push(gi);
        }
    }

    // We need successors for forward propagation.
    let mut successors: Vec<Vec<usize>> = vec![vec![]; n];
    for (gi, preds) in predecessors.iter().enumerate() {
        for &p in preds {
            successors[p].push(gi);
        }
    }

    let mut topo_order = Vec::with_capacity(n);
    let mut head = 0;
    while head < queue.len() {
        let gi = queue[head];
        head += 1;
        topo_order.push(gi);

        for &succ in &successors[gi] {
            in_degree[succ] -= 1;
            if in_degree[succ] == 0 {
                queue.push(succ);
            }
        }
    }

    // Forward sweep: compute earliest finish times.
    for &gi in &topo_order {
        let cost = if is_literal[gi] { 0 } else { groups[gi].count };
        let earliest_start = predecessors[gi]
            .iter()
            .map(|&p| earliest_finish[p])
            .max()
            .unwrap_or(0);
        earliest_finish[gi] = earliest_start + cost;
    }

    let depth: Vec<u64> = (0..n)
        .map(|gi| {
            let cost = if is_literal[gi] { 0 } else { groups[gi].count };
            if earliest_finish[gi] >= cost {
                earliest_finish[gi] - cost
            } else {
                0
            }
        })
        .collect();

    (depth, earliest_finish)
}

/// Find the critical path through the DAG (longest path).
/// Returns group indices in topological order along the path.
fn find_critical_path(
    groups: &[crate::nano_graph::AtomGroup],
    predecessors: &[Vec<usize>],
    successors: &[Vec<usize>],
    depth: &[u64],
    earliest_finish: &[u64],
    is_literal: &[bool],
) -> Vec<usize> {
    let n = groups.len();
    if n == 0 {
        return vec![];
    }

    // Find the group with the highest earliest_finish — that's the end of the critical path.
    let end = (0..n)
        .filter(|&gi| !is_literal[gi])
        .max_by_key(|&gi| earliest_finish[gi])
        .unwrap_or(0);

    // Trace backwards from the end: at each step, pick the predecessor with the
    // highest earliest_finish time (breaking ties by count).
    let mut path = vec![end];
    let mut current = end;
    loop {
        let preds: Vec<usize> = predecessors[current]
            .iter()
            .copied()
            .filter(|&p| !is_literal[p])
            .collect();
        if preds.is_empty() {
            break;
        }
        // Pick the predecessor with the highest earliest_finish.
        let best = preds
            .iter()
            .copied()
            .max_by_key(|&p| (earliest_finish[p], groups[p].count))
            .unwrap();
        path.push(best);
        current = best;
    }

    path.reverse();
    path
}

/// Find barrier positions on the critical path.
///
/// A barrier goes after each ReduceSum on the critical path that reads from
/// a matmul Mul group (indicated by the Mul group having a StridedBroadcast input,
/// or by the ReduceSum having reduce_count > 1 which means it's a matmul contraction).
///
/// Returns the indices (within critical_path) where barriers should be placed AFTER.
fn find_barrier_positions(graph: &NanoGraph, critical_path: &[usize]) -> Vec<usize> {
    let groups = graph.groups();
    let mut barriers = Vec::new();

    for (cp_idx, &gi) in critical_path.iter().enumerate() {
        let group = &groups[gi];

        // Check if this is a ReduceSum with reduce_count > 1 (matmul contraction).
        let is_matmul_reduce = match &group.op {
            ScalarOp::ReduceSum { reduce_count, .. } => *reduce_count > 1,
            _ => false,
        };

        if !is_matmul_reduce {
            continue;
        }

        // Check if its input comes from a Mul group with StridedBroadcast.
        // For the older test_graphs format (M*K small groups), check if the
        // input comes from a Binary Mul group with a Broadcast input.
        let reads_from_matmul_mul = group.inputs.iter().any(|input| {
            let source_ids = sample_source_atoms(input, group.count);
            source_ids.iter().any(|&src_id| {
                if let Some(src_gi) = find_group_index(groups, src_id) {
                    let src_group = &groups[src_gi];
                    matches!(
                        &src_group.op,
                        ScalarOp::Binary {
                            op: crate::nano_graph::ScalarBinOp::Mul,
                            ..
                        }
                    ) && src_group.inputs.iter().any(|inp| matches!(
                        inp,
                        InputRef::StridedBroadcast { .. } | InputRef::Broadcast(_)
                    ))
                } else {
                    false
                }
            })
        });

        if reads_from_matmul_mul {
            barriers.push(cp_idx);
        }
    }

    barriers
}

/// Assign each group to a phase based on barrier positions on the critical path.
///
/// Groups are assigned to the earliest phase they can execute in, given their
/// data dependencies. A group in phase P means all its predecessors are in
/// phases <= P, and if a predecessor is in phase P, that predecessor is in the
/// same lane (within-phase independence) OR is separated by a barrier.
fn assign_phases(
    graph: &NanoGraph,
    predecessors: &[Vec<usize>],
    critical_path: &[usize],
    barrier_positions: &[usize], // indices within critical_path
    is_literal: &[bool],
) -> Vec<usize> {
    let groups = graph.groups();
    let n = groups.len();

    // Map critical path groups to their phase.
    // Phase 0 = groups before first barrier, phase 1 = after first barrier, etc.
    let critical_set: HashSet<usize> = critical_path.iter().copied().collect();
    let mut cp_phase: HashMap<usize, usize> = HashMap::new();

    // Convert barrier positions (indices into critical_path) to sets of groups
    // that are barrier boundaries.
    let barrier_groups: HashSet<usize> = barrier_positions
        .iter()
        .map(|&cp_idx| critical_path[cp_idx])
        .collect();

    let mut current_phase = 0;
    for &gi in critical_path {
        cp_phase.insert(gi, current_phase);
        if barrier_groups.contains(&gi) {
            current_phase += 1;
        }
    }
    let num_phases = current_phase + 1;

    // Assign all groups to phases. Use topological order: each group goes into
    // the max phase of its predecessors (or 0 if no predecessors).
    // But if a predecessor is a barrier group, the consumer goes into the NEXT phase.
    let mut phase_assignment = vec![0usize; n];

    // Build topological order.
    let mut in_degree: Vec<usize> = vec![0; n];
    for gi in 0..n {
        in_degree[gi] = predecessors[gi].len();
    }
    let mut successors: Vec<Vec<usize>> = vec![vec![]; n];
    for (gi, preds) in predecessors.iter().enumerate() {
        for &p in preds {
            successors[p].push(gi);
        }
    }

    let mut queue: Vec<usize> = Vec::new();
    for gi in 0..n {
        if in_degree[gi] == 0 {
            queue.push(gi);
        }
    }

    let mut topo_order = Vec::with_capacity(n);
    let mut head = 0;
    while head < queue.len() {
        let gi = queue[head];
        head += 1;
        topo_order.push(gi);
        for &succ in &successors[gi] {
            in_degree[succ] -= 1;
            if in_degree[succ] == 0 {
                queue.push(succ);
            }
        }
    }

    for &gi in &topo_order {
        if is_literal[gi] {
            phase_assignment[gi] = 0;
            continue;
        }

        // If this group is on the critical path, use its assigned phase.
        if let Some(&ph) = cp_phase.get(&gi) {
            phase_assignment[gi] = ph;
            continue;
        }

        // Otherwise, this group goes into the max phase of its predecessors.
        // If a predecessor is a barrier group, we need the phase AFTER the barrier.
        let mut max_phase = 0;
        for &pred in &predecessors[gi] {
            let pred_phase = phase_assignment[pred];
            let effective_phase = if barrier_groups.contains(&pred) {
                pred_phase + 1
            } else {
                pred_phase
            };
            max_phase = max_phase.max(effective_phase);
        }

        // Clamp to valid range.
        phase_assignment[gi] = max_phase.min(num_phases - 1);
    }

    phase_assignment
}

/// Compute latest start times (for slack analysis).
/// latest_start[g] = the latest time g can start without delaying the overall finish.
fn compute_latest_starts(
    groups: &[crate::nano_graph::AtomGroup],
    successors: &[Vec<usize>],
    earliest_finish: &[u64],
    is_literal: &[bool],
) -> Vec<u64> {
    let n = groups.len();
    let max_finish = earliest_finish.iter().copied().max().unwrap_or(0);

    let mut latest_finish = vec![max_finish; n];

    // Build reverse topological order.
    let mut in_degree: Vec<usize> = vec![0; n];
    for gi in 0..n {
        in_degree[gi] = successors[gi].len();
    }

    let mut queue: Vec<usize> = Vec::new();
    for gi in 0..n {
        if in_degree[gi] == 0 {
            queue.push(gi);
        }
    }

    // Build predecessors from successors for reverse traversal.
    let mut predecessors: Vec<Vec<usize>> = vec![vec![]; n];
    for (gi, succs) in successors.iter().enumerate() {
        for &s in succs {
            predecessors[s].push(gi);
        }
    }

    let mut rev_topo = Vec::with_capacity(n);
    let mut head = 0;
    while head < queue.len() {
        let gi = queue[head];
        head += 1;
        rev_topo.push(gi);
        for &pred in &predecessors[gi] {
            in_degree[pred] -= 1;
            if in_degree[pred] == 0 {
                queue.push(pred);
            }
        }
    }

    // Backward sweep: latest_finish[g] = min(latest_start[succ]) for all successors.
    for &gi in &rev_topo {
        let cost = if is_literal[gi] { 0 } else { groups[gi].count };
        for &succ in &successors[gi] {
            let succ_cost = if is_literal[succ] { 0 } else { groups[succ].count };
            let succ_latest_start = if latest_finish[succ] >= succ_cost {
                latest_finish[succ] - succ_cost
            } else {
                0
            };
            latest_finish[gi] = latest_finish[gi].min(succ_latest_start);
        }
    }

    // latest_start = latest_finish - cost.
    (0..n)
        .map(|gi| {
            let cost = if is_literal[gi] { 0 } else { groups[gi].count };
            if latest_finish[gi] >= cost {
                latest_finish[gi] - cost
            } else {
                0
            }
        })
        .collect()
}

/// Distribute parallel (off-critical-path) groups across lanes.
///
/// First computes connected components of within-phase dependencies (groups
/// that must be on the same lane because one reads from the other). Then
/// assigns each component as a unit to a lane, using cache affinity and
/// load balancing.
fn distribute_parallel_groups(
    graph: &NanoGraph,
    parallel_groups: &[usize],
    phases: &mut Vec<Phase>,
    phase_idx: usize,
    num_lanes: usize,
    predecessors: &[Vec<usize>],
) {
    if parallel_groups.is_empty() || num_lanes <= 1 {
        return;
    }

    let groups = graph.groups();

    // Build the set of parallel groups in this phase for fast lookup.
    let parallel_set: HashSet<usize> = parallel_groups.iter().copied().collect();

    // Also include critical-path groups from this phase in the set of
    // "groups computed this phase" for dependency checking.
    let mut this_phase_set: HashSet<usize> = parallel_set.clone();
    for &gi in &phases[phase_idx].lane_work[0] {
        this_phase_set.insert(gi);
    }

    // Union-Find for connected components of within-phase dependencies.
    // Only parallel groups participate — critical-path groups are already on lane 0.
    let mut parent: HashMap<usize, usize> = HashMap::new();
    for &gi in parallel_groups {
        parent.insert(gi, gi);
    }

    fn find(parent: &mut HashMap<usize, usize>, x: usize) -> usize {
        let p = parent[&x];
        if p == x {
            return x;
        }
        let root = find(parent, p);
        parent.insert(x, root);
        root
    }

    fn union(parent: &mut HashMap<usize, usize>, a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent.insert(ra, rb);
        }
    }

    // For each parallel group, if it depends on another parallel group in this
    // phase, union them (they must be on the same lane).
    for &gi in parallel_groups {
        for &pred in &predecessors[gi] {
            if parallel_set.contains(&pred) {
                union(&mut parent, gi, pred);
            }
        }
    }

    // Collect components.
    let mut components: HashMap<usize, Vec<usize>> = HashMap::new();
    for &gi in parallel_groups {
        let root = find(&mut parent, gi);
        components.entry(root).or_default().push(gi);
    }

    // Sort components by total cost (largest first) for better load balancing.
    let mut component_list: Vec<Vec<usize>> = components.into_values().collect();
    component_list.sort_by(|a, b| {
        let cost_a: u64 = a.iter().map(|&gi| groups[gi].count).sum();
        let cost_b: u64 = b.iter().map(|&gi| groups[gi].count).sum();
        cost_b.cmp(&cost_a) // Largest first.
    });

    // Build a cache profile for each lane from previous phases.
    let mut lane_prev_groups: Vec<HashSet<usize>> = vec![HashSet::new(); num_lanes];
    for prev_phase in 0..phase_idx {
        for lane in 0..num_lanes {
            for &gi in &phases[prev_phase].lane_work[lane] {
                lane_prev_groups[lane].insert(gi);
            }
        }
    }

    // Track work assigned to each lane in this phase (for balancing).
    let mut lane_work_count: Vec<u64> = vec![0; num_lanes];
    // Include critical-path work already on lane 0.
    for &gi in &phases[phase_idx].lane_work[0] {
        lane_work_count[0] += groups[gi].count;
    }

    // Assign each component to the best lane.
    let start_lane = if num_lanes > 1 { 1 } else { 0 };
    for component in &component_list {
        let comp_cost: u64 = component.iter().map(|&gi| groups[gi].count).sum();

        // Collect all predecessors of the component (for cache affinity).
        let comp_preds: HashSet<usize> = component
            .iter()
            .flat_map(|&gi| predecessors[gi].iter().copied())
            .collect();

        let mut best_lane = start_lane;
        let mut best_score: i64 = i64::MIN;

        for lane in start_lane..num_lanes {
            // Cache affinity: how many of this component's predecessors are in this lane?
            let cache_hits = comp_preds
                .iter()
                .filter(|&&p| lane_prev_groups[lane].contains(&p))
                .count() as i64;

            // Load balance: prefer less-loaded lanes.
            let load_penalty = lane_work_count[lane] as i64;

            let score = cache_hits * 1000 - load_penalty;

            if score > best_score {
                best_score = score;
                best_lane = lane;
            }
        }

        for &gi in component {
            phases[phase_idx].lane_work[best_lane].push(gi);
        }
        lane_work_count[best_lane] += comp_cost;
    }
}

// ---- Verification helpers ----

/// Verify that within each phase, no lane reads another lane's current-phase output.
/// Returns a list of violations (empty = valid).
pub fn verify_within_phase_independence(graph: &NanoGraph, plan: &ExecutionPlan) -> Vec<String> {
    let groups = graph.groups();
    let mut errors = Vec::new();

    for (phase_idx, phase) in plan.phases.iter().enumerate() {
        // Build a map: group_index -> lane that computes it in this phase.
        let mut group_to_lane: HashMap<usize, usize> = HashMap::new();
        for (lane_idx, lane_groups) in phase.lane_work.iter().enumerate() {
            for &gi in lane_groups {
                group_to_lane.insert(gi, lane_idx);
            }
        }

        // For each group in this phase, check that its predecessors in this same
        // phase are on the same lane.
        for (lane_idx, lane_groups) in phase.lane_work.iter().enumerate() {
            for &gi in lane_groups {
                let group = &groups[gi];
                for input in &group.inputs {
                    let source_ids = sample_source_atoms(input, group.count);
                    for src_id in source_ids {
                        if let Some(src_gi) = find_group_index(groups, src_id) {
                            if let Some(&src_lane) = group_to_lane.get(&src_gi) {
                                if src_lane != lane_idx {
                                    errors.push(format!(
                                        "Phase {}: group {} (lane {}) reads from group {} (lane {})",
                                        phase_idx, gi, lane_idx, src_gi, src_lane
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    errors
}

/// Verify that every non-literal group appears in exactly one phase+lane.
/// Literal groups are pre-existing shared data and are not assigned to lanes.
pub fn verify_complete_coverage(graph: &NanoGraph, plan: &ExecutionPlan) -> Vec<String> {
    let groups = graph.groups();
    let num_groups = groups.len();
    let mut seen = vec![false; num_groups];
    let mut errors = Vec::new();

    let is_literal: Vec<bool> = groups
        .iter()
        .map(|g| matches!(g.op, ScalarOp::Literal(_)))
        .collect();

    for (phase_idx, phase) in plan.phases.iter().enumerate() {
        for (lane_idx, lane_groups) in phase.lane_work.iter().enumerate() {
            for &gi in lane_groups {
                if gi >= num_groups {
                    errors.push(format!(
                        "Phase {} lane {}: group index {} out of range ({})",
                        phase_idx, lane_idx, gi, num_groups
                    ));
                    continue;
                }
                if seen[gi] {
                    errors.push(format!(
                        "Phase {} lane {}: group {} appears multiple times",
                        phase_idx, lane_idx, gi
                    ));
                }
                seen[gi] = true;
            }
        }
    }

    for gi in 0..num_groups {
        if !seen[gi] && !is_literal[gi] {
            errors.push(format!("Group {} not assigned to any phase/lane", gi));
        }
    }

    errors
}

/// Compute work balance ratio per phase: max_lane_work / min_lane_work.
/// Returns (phase_index, ratio) pairs. ratio < 2.0 is the target.
pub fn compute_balance(graph: &NanoGraph, plan: &ExecutionPlan) -> Vec<(usize, f64)> {
    let groups = graph.groups();
    let mut ratios = Vec::new();

    for (phase_idx, phase) in plan.phases.iter().enumerate() {
        let lane_costs: Vec<u64> = phase
            .lane_work
            .iter()
            .map(|lane_groups| {
                lane_groups
                    .iter()
                    .map(|&gi| groups[gi].count)
                    .sum::<u64>()
            })
            .collect();

        let max_cost = lane_costs.iter().copied().max().unwrap_or(0);
        let min_cost = lane_costs
            .iter()
            .copied()
            .filter(|&c| c > 0)
            .min()
            .unwrap_or(1);

        let ratio = if min_cost == 0 {
            if max_cost == 0 {
                1.0
            } else {
                f64::INFINITY
            }
        } else {
            max_cost as f64 / min_cost as f64
        };

        ratios.push((phase_idx, ratio));
    }

    ratios
}

/// Verify that dependency ordering is respected: if group A is a predecessor of
/// group B, then A must be in an earlier phase, or in the same phase on the
/// same lane with A appearing before B in the lane's work list.
pub fn verify_dependency_order(graph: &NanoGraph, plan: &ExecutionPlan) -> Vec<String> {
    let groups = graph.groups();
    let mut errors = Vec::new();

    // Build map: group_index -> (phase, lane, position_in_lane).
    let mut group_location: HashMap<usize, (usize, usize, usize)> = HashMap::new();
    for (phase_idx, phase) in plan.phases.iter().enumerate() {
        for (lane_idx, lane_groups) in phase.lane_work.iter().enumerate() {
            for (pos, &gi) in lane_groups.iter().enumerate() {
                group_location.insert(gi, (phase_idx, lane_idx, pos));
            }
        }
    }

    // Check each group's predecessors.
    let (predecessors, _) = build_group_dag(graph);
    for (gi, preds) in predecessors.iter().enumerate() {
        if let Some(&(gi_phase, gi_lane, gi_pos)) = group_location.get(&gi) {
            for &pred in preds {
                if let Some(&(pred_phase, pred_lane, pred_pos)) = group_location.get(&pred) {
                    if pred_phase > gi_phase {
                        errors.push(format!(
                            "Group {} (phase {}) depends on group {} (phase {}): predecessor in later phase",
                            gi, gi_phase, pred, pred_phase
                        ));
                    } else if pred_phase == gi_phase && pred_lane == gi_lane && pred_pos >= gi_pos {
                        errors.push(format!(
                            "Group {} (phase {} lane {} pos {}) depends on group {} (same phase/lane pos {}): wrong order",
                            gi, gi_phase, gi_lane, gi_pos, pred, pred_pos
                        ));
                    }
                    // pred_phase == gi_phase && pred_lane != gi_lane is the
                    // cross-lane same-phase dependency we check in
                    // verify_within_phase_independence.
                }
            }
        }
    }

    errors
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::attempts::v13_claude::test_graphs;
    use crate::nano_graph::{ScalarBinOp, ScalarUnaryOp};

    #[test]
    fn test_empty_graph() {
        let g = NanoGraph::new();
        let plan = plan_execution(&g, 4);
        assert_eq!(plan.num_lanes, 4);
        assert!(plan.phases.is_empty());
    }

    #[test]
    fn test_elementwise_single_lane() {
        let (g, _, _, _) = test_graphs::elementwise_binary(1024, ScalarBinOp::Add);
        let plan = plan_execution(&g, 1);
        let coverage = verify_complete_coverage(&g, &plan);
        assert!(coverage.is_empty(), "Coverage errors: {:?}", coverage);
        let deps = verify_dependency_order(&g, &plan);
        assert!(deps.is_empty(), "Dependency errors: {:?}", deps);
    }

    #[test]
    fn test_elementwise_multi_lane() {
        let (g, _, _, _) = test_graphs::elementwise_binary(1024, ScalarBinOp::Add);
        let plan = plan_execution(&g, 4);
        let coverage = verify_complete_coverage(&g, &plan);
        assert!(coverage.is_empty(), "Coverage errors: {:?}", coverage);
        let independence = verify_within_phase_independence(&g, &plan);
        assert!(
            independence.is_empty(),
            "Independence errors: {:?}",
            independence
        );
    }

    #[test]
    fn test_matmul_single_lane() {
        let (g, _, _, _) = test_graphs::matmul(4, 8, 16);
        let plan = plan_execution(&g, 1);
        let coverage = verify_complete_coverage(&g, &plan);
        assert!(coverage.is_empty(), "Coverage errors: {:?}", coverage);
        let deps = verify_dependency_order(&g, &plan);
        assert!(deps.is_empty(), "Dependency errors: {:?}", deps);
    }

    #[test]
    fn test_matmul_multi_lane() {
        let (g, _, _, _) = test_graphs::matmul(4, 8, 16);
        let plan = plan_execution(&g, 4);

        let coverage = verify_complete_coverage(&g, &plan);
        assert!(coverage.is_empty(), "Coverage errors: {:?}", coverage);

        let independence = verify_within_phase_independence(&g, &plan);
        assert!(
            independence.is_empty(),
            "Independence errors: {:?}",
            independence
        );

        let deps = verify_dependency_order(&g, &plan);
        assert!(deps.is_empty(), "Dependency errors: {:?}", deps);

        // Should have at least 1 phase.
        assert!(!plan.phases.is_empty(), "Should have at least one phase");
    }

    #[test]
    fn test_matmul_chain_multi_lane() {
        let (g, _, _, _, _) = test_graphs::matmul_chain(4, 8, 16, 16, 32);
        let plan = plan_execution(&g, 4);

        let coverage = verify_complete_coverage(&g, &plan);
        assert!(coverage.is_empty(), "Coverage errors: {:?}", coverage);

        let independence = verify_within_phase_independence(&g, &plan);
        assert!(
            independence.is_empty(),
            "Independence errors: {:?}",
            independence
        );

        let deps = verify_dependency_order(&g, &plan);
        assert!(deps.is_empty(), "Dependency errors: {:?}", deps);

        // A chained matmul should have at least 2 phases (barrier between matmuls).
        assert!(
            plan.phases.len() >= 2,
            "Chained matmul should have >= 2 phases, got {}",
            plan.phases.len()
        );
    }

    #[test]
    fn test_matmul_activation_multi_lane() {
        let (g, _, _, _) =
            test_graphs::matmul_activation(4, 8, 16, ScalarUnaryOp::Tanh);
        let plan = plan_execution(&g, 4);

        let coverage = verify_complete_coverage(&g, &plan);
        assert!(coverage.is_empty(), "Coverage errors: {:?}", coverage);

        let independence = verify_within_phase_independence(&g, &plan);
        assert!(
            independence.is_empty(),
            "Independence errors: {:?}",
            independence
        );
    }

    #[test]
    fn test_unary_chain() {
        let (g, _, _) = test_graphs::unary_chain(
            256,
            &[ScalarUnaryOp::Exp, ScalarUnaryOp::Neg, ScalarUnaryOp::Tanh],
        );
        let plan = plan_execution(&g, 4);

        let coverage = verify_complete_coverage(&g, &plan);
        assert!(coverage.is_empty(), "Coverage errors: {:?}", coverage);

        let independence = verify_within_phase_independence(&g, &plan);
        assert!(
            independence.is_empty(),
            "Independence errors: {:?}",
            independence
        );
    }

    #[test]
    fn test_broadcast_add() {
        let (g, _, _, _) = test_graphs::broadcast_add(1024);
        let plan = plan_execution(&g, 4);

        let coverage = verify_complete_coverage(&g, &plan);
        assert!(coverage.is_empty(), "Coverage errors: {:?}", coverage);

        let independence = verify_within_phase_independence(&g, &plan);
        assert!(
            independence.is_empty(),
            "Independence errors: {:?}",
            independence
        );
    }

    #[test]
    fn test_balance_matmul() {
        let (g, _, _, _) = test_graphs::matmul(8, 16, 32);
        let plan = plan_execution(&g, 4);

        let balance = compute_balance(&g, &plan);
        for (phase_idx, ratio) in &balance {
            // Allow some imbalance but not extreme.
            assert!(
                *ratio < 100.0,
                "Phase {} has extreme imbalance: ratio = {}",
                phase_idx,
                ratio
            );
        }
    }

    /// Test with a merged-matmul structure (StridedBroadcast), matching real lowering.
    #[test]
    fn test_merged_matmul() {
        use crate::dtype::DType;
        use crate::nano_graph::NanoGraph;
        use crate::numeric_scalar::NumericScalar;

        let m: u64 = 4;
        let k: u64 = 8;
        let n: u64 = 16;

        let mut g = NanoGraph::new();

        // A[M, K]
        let a = g.push_group(
            m * k,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![], vec![], vec![],
        );
        // B[K, N]
        let b = g.push_group(
            k * n,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![], vec![], vec![],
        );

        // M merged Mul groups, each of count K*N with StridedBroadcast.
        let mut mul_bases = Vec::new();
        for mi in 0..m {
            let a_row_base = a.offset(mi * k); // A[m, 0]
            let base = g.push_group(
                k * n,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![], vec![],
                vec![
                    InputRef::StridedBroadcast {
                        base: a_row_base,
                        stride: 1,
                        repeat: n,
                    },
                    InputRef::Affine { base: b, stride: 1 },
                ],
            );
            mul_bases.push(base);
        }

        // M ReduceSum groups, each of count N.
        let mut reduce_base = None;
        for mi in 0..m {
            let base = g.push_group(
                n,
                ScalarOp::ReduceSum {
                    reduce_count: k,
                    reduce_stride: n as i64,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![], vec![],
                vec![InputRef::Affine {
                    base: mul_bases[mi as usize],
                    stride: 1,
                }],
            );
            if reduce_base.is_none() {
                reduce_base = Some(base);
            }
        }
        g.outputs = vec![reduce_base.unwrap()];

        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let plan = plan_execution(&g, 4);

        let coverage = verify_complete_coverage(&g, &plan);
        assert!(coverage.is_empty(), "Coverage errors: {:?}", coverage);

        let independence = verify_within_phase_independence(&g, &plan);
        assert!(
            independence.is_empty(),
            "Independence errors: {:?}",
            independence
        );

        let deps = verify_dependency_order(&g, &plan);
        assert!(deps.is_empty(), "Dependency errors: {:?}", deps);
    }

    /// Test with a two-matmul chain using merged (StridedBroadcast) structure.
    #[test]
    fn test_merged_matmul_chain() {
        use crate::dtype::DType;
        use crate::nano_graph::NanoGraph;
        use crate::numeric_scalar::NumericScalar;

        let m: u64 = 4;
        let k1: u64 = 8;
        let n1: u64 = 16;
        let k2: u64 = n1;
        let n2: u64 = 32;

        let mut g = NanoGraph::new();

        // A[M, K1], B[K1, N1], C[K2, N2]
        let a = g.push_group(
            m * k1,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![], vec![], vec![],
        );
        let b = g.push_group(
            k1 * n1,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![], vec![], vec![],
        );
        let c = g.push_group(
            k2 * n2,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![], vec![], vec![],
        );

        // First matmul: AB = A @ B [M, N1]
        let mut mul1_bases = Vec::new();
        for mi in 0..m {
            let a_row = a.offset(mi * k1);
            let base = g.push_group(
                k1 * n1,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![], vec![],
                vec![
                    InputRef::StridedBroadcast {
                        base: a_row,
                        stride: 1,
                        repeat: n1,
                    },
                    InputRef::Affine { base: b, stride: 1 },
                ],
            );
            mul1_bases.push(base);
        }

        let mut ab_bases = Vec::new();
        for mi in 0..m {
            let base = g.push_group(
                n1,
                ScalarOp::ReduceSum {
                    reduce_count: k1,
                    reduce_stride: n1 as i64,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![], vec![],
                vec![InputRef::Affine {
                    base: mul1_bases[mi as usize],
                    stride: 1,
                }],
            );
            ab_bases.push(base);
        }

        // Second matmul: out = AB @ C [M, N2]
        // Each row of AB is ab_bases[mi], with n1 atoms.
        let mut mul2_bases = Vec::new();
        for mi in 0..m {
            let ab_row = ab_bases[mi as usize];
            let base = g.push_group(
                k2 * n2,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![], vec![],
                vec![
                    InputRef::StridedBroadcast {
                        base: ab_row,
                        stride: 1,
                        repeat: n2,
                    },
                    InputRef::Affine { base: c, stride: 1 },
                ],
            );
            mul2_bases.push(base);
        }

        let mut out_base = None;
        for mi in 0..m {
            let base = g.push_group(
                n2,
                ScalarOp::ReduceSum {
                    reduce_count: k2,
                    reduce_stride: n2 as i64,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![], vec![],
                vec![InputRef::Affine {
                    base: mul2_bases[mi as usize],
                    stride: 1,
                }],
            );
            if out_base.is_none() {
                out_base = Some(base);
            }
        }
        g.outputs = vec![out_base.unwrap()];

        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let plan = plan_execution(&g, 4);

        let coverage = verify_complete_coverage(&g, &plan);
        assert!(coverage.is_empty(), "Coverage errors: {:?}", coverage);

        let independence = verify_within_phase_independence(&g, &plan);
        assert!(
            independence.is_empty(),
            "Independence errors: {:?}",
            independence
        );

        let deps = verify_dependency_order(&g, &plan);
        assert!(deps.is_empty(), "Dependency errors: {:?}", deps);

        // Should have at least 2 phases for a chained matmul.
        assert!(
            plan.phases.len() >= 2,
            "Merged matmul chain should have >= 2 phases, got {}",
            plan.phases.len()
        );
    }
}
