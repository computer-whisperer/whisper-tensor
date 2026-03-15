#![allow(clippy::all, dead_code, unreachable_patterns, unused_variables, unused_imports)]
//! Lane+barrier execution plan for NanoGraph.
//!
//! Implements the execution model from `problem_shape.md`: persistent lanes
//! (pinned threads) executing phases separated by barrier sync points.
//!
//! Algorithm:
//!
//! 1. **Build group-level DAG**: Resolve InputRefs to find producer groups.
//!    Classify groups as data (Literal) vs compute.
//!
//! 2. **Chain extraction**: Walk backward from output/sink groups through
//!    exclusive producer-consumer edges. Each chain is a maximal sequence
//!    of groups connected by fan-out=1 edges. Chains are the atomic units
//!    for lane assignment.
//!
//! 3. **Find barrier positions**: Identify ReduceSum groups on the critical
//!    path (longest chain through the DAG). Consecutive ReduceSum groups on
//!    the critical path define phase boundaries.
//!
//! 4. **Assign chains to phases**: Each chain goes into the phase determined
//!    by its latest ReduceSum barrier position.
//!
//! 5. **Lane assignment via agglomerative clustering**: Within each phase,
//!    cluster chains by shared input data (weight groups). Chains sharing
//!    more weight data go on the same lane.
//!
//! 6. **Cache-consistent lane assignment across phases**: When assigning
//!    phase N+1, prefer putting chains on lanes that already have their
//!    input data hot from phase N.

use std::collections::{BTreeSet, HashMap, HashSet, BinaryHeap, VecDeque};
use std::cmp::Ordering;

use crate::nano_graph::{AtomGroup, AtomId, InputRef, NanoGraph, ScalarOp};

// ─── Public API ──────────────────────────────────────────────────────────────

/// An execution plan: lanes running through phases separated by barriers.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub num_lanes: usize,
    pub phases: Vec<Phase>,
}

/// One phase of execution. All lanes work independently within a phase.
#[derive(Debug, Clone)]
pub struct Phase {
    /// lane_idx -> group indices that this lane executes in this phase.
    pub lane_work: Vec<Vec<usize>>,
}

/// Plan execution for a NanoGraph with `num_lanes` persistent threads.
pub fn plan_execution(graph: &NanoGraph, num_lanes: usize) -> ExecutionPlan {
    let groups = graph.groups();
    let n = groups.len();
    let num_lanes = num_lanes.max(1);

    if n == 0 {
        return ExecutionPlan {
            num_lanes,
            phases: vec![],
        };
    }

    // Classify groups.
    let is_data: Vec<bool> = groups
        .iter()
        .map(|g| matches!(g.op, ScalarOp::Literal(_)) && g.inputs.is_empty())
        .collect();

    let compute_indices: Vec<usize> = (0..n).filter(|&i| !is_data[i]).collect();
    if compute_indices.is_empty() {
        return ExecutionPlan {
            num_lanes,
            phases: vec![],
        };
    }

    // Step 1: Build group-level dependency DAG.
    let (producers, consumers) = build_group_deps(groups);

    // Compute fan-out: how many distinct compute groups consume each group.
    let fan_out: Vec<usize> = consumers
        .iter()
        .enumerate()
        .map(|(_gi, cons)| cons.iter().filter(|&&ci| !is_data[ci]).count())
        .collect();

    // Step 2: Find barrier depths (critical path ReduceSum depths).
    let topo_order = topological_sort(n, &producers);
    let (barrier_depths, group_depth) =
        find_barrier_depths(groups, &topo_order, &producers, &is_data);

    // Step 3: Extract chains. Chains stop at ReduceSum groups that are at
    // barrier depths -- because downstream groups (next matmul) require a
    // barrier before they can start.
    let reduce_barrier_set: HashSet<usize> = (0..n)
        .filter(|&gi| {
            !is_data[gi]
                && matches!(
                    groups[gi].op,
                    ScalarOp::ReduceSum { .. } | ScalarOp::ReduceMax { .. }
                )
                && barrier_depths.contains(&group_depth[gi])
        })
        .collect();

    let chains = extract_chains(
        n,
        &producers,
        &consumers,
        &fan_out,
        &is_data,
        &reduce_barrier_set,
    );
    let _group_to_chain = build_group_to_chain_map(n, &chains);

    // Step 4: Assign chains to phases based on barrier depths.
    let phase_assignments =
        assign_chains_to_phases_by_depth(&chains, &group_depth, &barrier_depths);
    let num_phases = phase_assignments
        .iter()
        .copied()
        .max()
        .map(|m| m + 1)
        .unwrap_or(1);

    // Step 5: Compute chain input signatures for clustering.
    let chain_inputs = compute_chain_inputs(&chains, &producers, &is_data);

    // Step 6: Assign chains to lanes per phase via agglomerative clustering
    // with cache-consistent assignment across phases.
    let phases = assign_lanes(
        num_lanes,
        num_phases,
        &chains,
        &chain_inputs,
        &phase_assignments,
        groups,
        &producers,
        &is_data,
    );

    ExecutionPlan {
        num_lanes,
        phases,
    }
}

// ─── Group dependency graph ──────────────────────────────────────────────────

/// Build producer and consumer graphs at group level.
fn build_group_deps(groups: &[AtomGroup]) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let n = groups.len();
    let mut producers: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (gi, group) in groups.iter().enumerate() {
        let mut prod_set = BTreeSet::new();
        for input in &group.inputs {
            let prods = resolve_producer_groups(input, group.count, groups);
            for pi in prods {
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

/// Find all groups that produce atoms referenced by an InputRef.
fn resolve_producer_groups(input: &InputRef, count: u64, groups: &[AtomGroup]) -> Vec<usize> {
    match input {
        InputRef::Broadcast(atom_id) => {
            find_group_idx(groups, *atom_id).into_iter().collect()
        }
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
        InputRef::StridedBroadcast { base, stride, repeat } => {
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
        InputRef::Modular { base, stride, modulus } => {
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

/// Find all groups whose atom ranges overlap [lo, hi].
fn find_groups_in_range(groups: &[AtomGroup], lo: u64, hi: u64) -> Vec<usize> {
    let mut result = Vec::new();
    // Find the first group that could contain `lo`.
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

// ─── Chain extraction ────────────────────────────────────────────────────────

/// Extract chains: maximal groups connected by exclusive (fan-out=1) edges.
/// Each compute group belongs to exactly one chain.
/// Chains do NOT cross barrier boundaries: if a group is a barrier ReduceSum,
/// the chain stops there and the next segment starts a new chain.
fn extract_chains(
    n: usize,
    producers: &[Vec<usize>],
    consumers: &[Vec<usize>],
    fan_out: &[usize],
    is_data: &[bool],
    barrier_set: &HashSet<usize>,
) -> Vec<Vec<usize>> {
    let mut chain_of: Vec<Option<usize>> = vec![None; n];
    let mut chains: Vec<Vec<usize>> = Vec::new();

    // Process in reverse insertion order (sinks first, since groups are in topo order).
    for gi in (0..n).rev() {
        if is_data[gi] || chain_of[gi].is_some() {
            continue;
        }

        let chain_id = chains.len();
        let mut chain = vec![gi];
        chain_of[gi] = Some(chain_id);

        // If this group is a barrier reduce, don't extend backward into
        // the previous phase's territory. The barrier reduce starts a new chain
        // from its side.
        // Actually: we're walking backward from sinks. If `gi` is a barrier,
        // we still walk backward from it, but we stop when we hit another barrier.
        // If `gi` is NOT a barrier, we walk backward but stop at barriers.

        // Walk backward through exclusive producers, stopping at barriers.
        let mut current = gi;
        loop {
            let exclusive_prods: Vec<usize> = producers[current]
                .iter()
                .copied()
                .filter(|&pi| !is_data[pi] && chain_of[pi].is_none() && fan_out[pi] == 1)
                .collect();

            if exclusive_prods.len() == 1 {
                let pred = exclusive_prods[0];
                // Don't cross INTO a barrier reduce from the other side.
                // A barrier reduce is a natural chain terminator.
                // If `pred` is a barrier and current is NOT a barrier, stop:
                // this means we'd be crossing a phase boundary backward.
                if barrier_set.contains(&pred) && !barrier_set.contains(&current) {
                    break;
                }
                chain.push(pred);
                chain_of[pred] = Some(chain_id);
                current = pred;
                // If we just added a barrier reduce, stop extending further backward.
                // The barrier is the start of this chain's phase.
                if barrier_set.contains(&pred) {
                    break;
                }
            } else {
                break;
            }
        }

        chain.reverse(); // Put in topo order (earliest first).
        chains.push(chain);
    }

    // Assign any remaining unassigned compute groups to singleton chains.
    for gi in 0..n {
        if !is_data[gi] && chain_of[gi].is_none() {
            let chain_id = chains.len();
            chain_of[gi] = Some(chain_id);
            chains.push(vec![gi]);
        }
    }

    chains
}

/// Build reverse map: group index -> chain index.
fn build_group_to_chain_map(n: usize, chains: &[Vec<usize>]) -> Vec<Option<usize>> {
    let mut map = vec![None; n];
    for (ci, chain) in chains.iter().enumerate() {
        for &gi in chain {
            map[gi] = Some(ci);
        }
    }
    map
}

// ─── Topological sort ────────────────────────────────────────────────────────

fn topological_sort(n: usize, producers: &[Vec<usize>]) -> Vec<usize> {
    let mut in_degree = vec![0usize; n];
    for (gi, prods) in producers.iter().enumerate() {
        in_degree[gi] = prods.len();
    }

    let mut consumers_map: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (gi, prods) in producers.iter().enumerate() {
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

// ─── Barrier position detection ──────────────────────────────────────────────

/// Find barrier depths: the topological depths of ReduceSum/ReduceMax groups
/// on the critical path. Returns (barrier_depths, per-group depth vector).
///
/// Each barrier depth corresponds to one matmul operation on the critical path.
/// All rows of that matmul (all ReduceSum groups at that depth) must complete
/// before the next phase can start.
fn find_barrier_depths(
    groups: &[AtomGroup],
    topo_order: &[usize],
    producers: &[Vec<usize>],
    is_data: &[bool],
) -> (BTreeSet<usize>, Vec<usize>) {
    let n = groups.len();
    if n == 0 {
        return (BTreeSet::new(), vec![]);
    }

    // Compute longest path distance from any source.
    let mut dist = vec![0usize; n];
    let mut pred = vec![usize::MAX; n];

    for &gi in topo_order {
        for &pi in &producers[gi] {
            let new_dist = dist[pi] + 1;
            if new_dist > dist[gi] {
                dist[gi] = new_dist;
                pred[gi] = pi;
            }
        }
    }

    // Trace the critical path backwards from the endpoint.
    let end = (0..n).max_by_key(|&i| dist[i]).unwrap_or(0);
    let mut critical_path = Vec::new();
    let mut cur = end;
    while cur != usize::MAX {
        critical_path.push(cur);
        cur = pred[cur];
    }
    critical_path.reverse();

    // Find depths of ReduceSum/ReduceMax groups on the critical path.
    let mut barrier_depths: BTreeSet<usize> = BTreeSet::new();
    for &gi in &critical_path {
        if matches!(groups[gi].op, ScalarOp::ReduceSum { .. } | ScalarOp::ReduceMax { .. }) {
            barrier_depths.insert(dist[gi]);
        }
    }

    (barrier_depths, dist)
}

// ─── Phase assignment ────────────────────────────────────────────────────────

/// Assign each chain to a phase based on barrier depths.
///
/// Phase boundaries are defined by barrier depths (sorted). A chain's phase
/// is determined by the maximum depth of any group in the chain relative to
/// the barrier depths.
///
/// - Phase 0: chains whose max depth <= first barrier depth
/// - Phase 1: chains whose max depth > first barrier depth and <= second barrier depth
/// - etc.
fn assign_chains_to_phases_by_depth(
    chains: &[Vec<usize>],
    group_depth: &[usize],
    barrier_depths: &BTreeSet<usize>,
) -> Vec<usize> {
    if barrier_depths.is_empty() {
        return vec![0; chains.len()];
    }

    let sorted_barriers: Vec<usize> = barrier_depths.iter().copied().collect();

    chains
        .iter()
        .map(|chain| {
            let max_depth = chain.iter().map(|&gi| group_depth[gi]).max().unwrap_or(0);
            // Phase = number of barrier depths strictly less than max_depth.
            // This puts the ReduceSum itself (at a barrier depth) in the phase
            // that ends with that barrier. Groups after the barrier are in the next phase.
            sorted_barriers
                .iter()
                .filter(|&&bd| bd < max_depth)
                .count()
        })
        .collect()
}

// ─── Input signature computation ─────────────────────────────────────────────

/// For each chain, compute the set of data (Literal) group indices it reads from
/// (directly or transitively through shared groups).
fn compute_chain_inputs(
    chains: &[Vec<usize>],
    producers: &[Vec<usize>],
    is_data: &[bool],
) -> Vec<BTreeSet<usize>> {
    chains
        .iter()
        .map(|chain| {
            let mut data_deps = BTreeSet::new();
            let chain_set: HashSet<usize> = chain.iter().copied().collect();
            let mut visited = HashSet::new();
            let mut stack: Vec<usize> = chain.clone();

            while let Some(gi) = stack.pop() {
                if !visited.insert(gi) {
                    continue;
                }
                for &pi in &producers[gi] {
                    if is_data[pi] {
                        data_deps.insert(pi);
                    } else if !chain_set.contains(&pi) {
                        // Non-data, non-chain producer: follow transitively
                        // to find which data groups feed into this chain.
                        // But we stop at shared (multi-consumer) producers
                        // to avoid pulling in the whole graph.
                        // Just record direct non-data producers as dependencies too
                        // since they represent intermediate results that this chain needs.
                        data_deps.insert(pi);
                    }
                }
            }

            data_deps
        })
        .collect()
}

// ─── Lane assignment ─────────────────────────────────────────────────────────

/// Assign chains to lanes across all phases using agglomerative clustering
/// with cache affinity across phases.
fn assign_lanes(
    num_lanes: usize,
    num_phases: usize,
    chains: &[Vec<usize>],
    chain_inputs: &[BTreeSet<usize>],
    phase_assignments: &[usize],
    groups: &[AtomGroup],
    producers: &[Vec<usize>],
    is_data: &[bool],
) -> Vec<Phase> {
    let nc = chains.len();

    // Track what data each lane has "hot" (accumulated across phases).
    let mut lane_hot_data: Vec<HashSet<usize>> = vec![HashSet::new(); num_lanes];

    let mut phases = Vec::with_capacity(num_phases);

    for phase_idx in 0..num_phases {
        // Gather chains belonging to this phase.
        let phase_chain_ids: Vec<usize> = (0..nc)
            .filter(|&ci| phase_assignments[ci] == phase_idx)
            .collect();

        if phase_chain_ids.is_empty() {
            phases.push(Phase {
                lane_work: vec![Vec::new(); num_lanes],
            });
            continue;
        }

        // Compute chain work estimates (total atom count).
        let chain_work: Vec<u64> = phase_chain_ids
            .iter()
            .map(|&ci| {
                chains[ci]
                    .iter()
                    .map(|&gi| groups[gi].count)
                    .sum::<u64>()
            })
            .collect();

        // Assign chains to lanes.
        let lane_assignments = if phase_idx == 0 {
            // First phase: cluster by shared inputs.
            cluster_chains_to_lanes(
                &phase_chain_ids,
                chain_inputs,
                &chain_work,
                num_lanes,
            )
        } else {
            // Subsequent phases: use cache affinity from previous phases.
            cache_affine_assign(
                &phase_chain_ids,
                chain_inputs,
                &chain_work,
                num_lanes,
                &lane_hot_data,
            )
        };

        // Build the Phase structure and update hot data.
        let mut lane_work: Vec<Vec<usize>> = vec![Vec::new(); num_lanes];
        for (local_idx, &ci) in phase_chain_ids.iter().enumerate() {
            let lane = lane_assignments[local_idx];
            for &gi in &chains[ci] {
                lane_work[lane].push(gi);
            }
            // Update lane's hot data with this chain's inputs.
            for &data_gi in &chain_inputs[ci] {
                lane_hot_data[lane].insert(data_gi);
            }
        }

        // Sort group indices within each lane for determinism.
        for lane in &mut lane_work {
            lane.sort();
        }

        phases.push(Phase { lane_work });
    }

    phases
}

/// Cluster chains into lanes by shared input data (agglomerative).
/// Used for the first phase where there's no cache history.
fn cluster_chains_to_lanes(
    phase_chain_ids: &[usize],
    chain_inputs: &[BTreeSet<usize>],
    chain_work: &[u64],
    num_lanes: usize,
) -> Vec<usize> {
    let npc = phase_chain_ids.len();
    if npc == 0 {
        return vec![];
    }
    if npc <= num_lanes {
        // Fewer chains than lanes: one chain per lane.
        return (0..npc).collect();
    }

    // Use union-find agglomerative clustering.
    let mut uf = UnionFind::new(npc);

    // Compute shared input counts between pairs.
    // Build inverted index: input -> list of local chain indices.
    let mut input_to_chains: HashMap<usize, Vec<usize>> = HashMap::new();
    for (local_idx, &ci) in phase_chain_ids.iter().enumerate() {
        for &inp in &chain_inputs[ci] {
            input_to_chains.entry(inp).or_default().push(local_idx);
        }
    }

    // Build merge candidates from shared inputs.
    let mut shared: HashMap<(usize, usize), u32> = HashMap::new();
    for (_, chains_using) in &input_to_chains {
        for (a, &ci) in chains_using.iter().enumerate() {
            for &cj in &chains_using[a + 1..] {
                let key = if ci < cj { (ci, cj) } else { (cj, ci) };
                *shared.entry(key).or_default() += 1;
            }
        }
    }

    let mut heap: BinaryHeap<MergeCandidate> = BinaryHeap::new();
    for (&(g1, g2), &count) in &shared {
        heap.push(MergeCandidate {
            g1,
            g2,
            benefit: count as i64,
        });
    }

    // Track group input sets for recomputation.
    let mut group_input_sets: Vec<HashSet<usize>> = phase_chain_ids
        .iter()
        .map(|&ci| chain_inputs[ci].iter().copied().collect())
        .collect();

    // Track group work.
    let mut group_work: Vec<u64> = chain_work.to_vec();

    let total_work: u64 = chain_work.iter().sum();
    let max_work_per_lane = (total_work / num_lanes as u64) * 2 + 1;

    let mut num_groups = npc;
    while num_groups > num_lanes {
        let cand = match heap.pop() {
            Some(c) => c,
            None => break,
        };

        let r1 = uf.find(cand.g1);
        let r2 = uf.find(cand.g2);
        if r1 == r2 {
            continue;
        }

        // Stale candidate: recompute.
        if r1 != cand.g1 || r2 != cand.g2 {
            let shared_count = group_input_sets[r1]
                .intersection(&group_input_sets[r2])
                .count();
            if shared_count > 0 {
                heap.push(MergeCandidate {
                    g1: r1,
                    g2: r2,
                    benefit: shared_count as i64,
                });
            }
            continue;
        }

        // Check work balance.
        if group_work[r1] + group_work[r2] > max_work_per_lane {
            continue;
        }

        // Merge.
        let new_root = uf.union(r1, r2);
        let other = if new_root == r1 { r2 } else { r1 };

        let other_inputs: HashSet<usize> = group_input_sets[other].clone();
        group_input_sets[new_root].extend(other_inputs);
        group_work[new_root] = group_work[r1] + group_work[r2];

        num_groups -= 1;

        // Re-evaluate neighbors.
        for &inp in &group_input_sets[new_root] {
            if let Some(chains_using) = input_to_chains.get(&inp) {
                for &local_ci in chains_using {
                    let ri = uf.find(local_ci);
                    if ri != new_root {
                        let shared_count = group_input_sets[new_root]
                            .intersection(&group_input_sets[ri])
                            .count();
                        if shared_count > 0 {
                            heap.push(MergeCandidate {
                                g1: new_root,
                                g2: ri,
                                benefit: shared_count as i64,
                            });
                        }
                    }
                }
            }
        }
    }

    // Map each chain to a lane.
    let mut root_to_lane: HashMap<usize, usize> = HashMap::new();
    let mut next_lane = 0;
    let mut assignments = vec![0usize; npc];
    for i in 0..npc {
        let root = uf.find(i);
        let lane = *root_to_lane.entry(root).or_insert_with(|| {
            let l = next_lane % num_lanes;
            next_lane += 1;
            l
        });
        assignments[i] = lane;
    }

    assignments
}

/// Assign chains to lanes using cache affinity from previous phases.
fn cache_affine_assign(
    phase_chain_ids: &[usize],
    chain_inputs: &[BTreeSet<usize>],
    chain_work: &[u64],
    num_lanes: usize,
    lane_hot_data: &[HashSet<usize>],
) -> Vec<usize> {
    let npc = phase_chain_ids.len();
    if npc == 0 {
        return vec![];
    }

    // For each chain, compute affinity score with each lane based on how many
    // of the chain's inputs are already hot in that lane.
    let mut assignments = vec![0usize; npc];
    let mut lane_work_totals = vec![0u64; num_lanes];

    // Sort chains by descending work (assign big chains first for better balance).
    let mut sorted_indices: Vec<usize> = (0..npc).collect();
    sorted_indices.sort_by(|&a, &b| chain_work[b].cmp(&chain_work[a]));

    let total_work: u64 = chain_work.iter().sum();
    let target_work = total_work / num_lanes as u64 + 1;

    for &local_idx in &sorted_indices {
        let ci = phase_chain_ids[local_idx];

        // Compute affinity with each lane.
        let mut best_lane = 0;
        let mut best_score: i64 = i64::MIN;

        for lane in 0..num_lanes {
            // Penalty for overloading a lane.
            let balance_penalty = if lane_work_totals[lane] + chain_work[local_idx] > target_work * 2
            {
                -1000
            } else {
                0
            };

            // Affinity: how many of this chain's inputs are hot in this lane.
            let affinity: i64 = chain_inputs[ci]
                .iter()
                .filter(|inp| lane_hot_data[lane].contains(inp))
                .count() as i64;

            // Load balance score: prefer less-loaded lanes.
            let balance_score = -(lane_work_totals[lane] as i64);

            let score = affinity * 10 + balance_score + balance_penalty;
            if score > best_score {
                best_score = score;
                best_lane = lane;
            }
        }

        assignments[local_idx] = best_lane;
        lane_work_totals[best_lane] += chain_work[local_idx];
    }

    assignments
}

// ─── Union-Find ──────────────────────────────────────────────────────────────

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            rank: vec![0; n],
            size: vec![1; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) -> usize {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return ra;
        }
        let (big, small) = if self.rank[ra] >= self.rank[rb] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.parent[small] = big;
        self.size[big] += self.size[small];
        if self.rank[big] == self.rank[small] {
            self.rank[big] += 1;
        }
        big
    }
}

// ─── Merge candidate for agglomerative clustering ───────────────────────────

#[derive(Clone)]
struct MergeCandidate {
    g1: usize,
    g2: usize,
    benefit: i64,
}

impl PartialEq for MergeCandidate {
    fn eq(&self, o: &Self) -> bool {
        self.benefit == o.benefit
    }
}
impl Eq for MergeCandidate {}
impl PartialOrd for MergeCandidate {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for MergeCandidate {
    fn cmp(&self, o: &Self) -> Ordering {
        self.benefit.cmp(&o.benefit)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::nano_graph::{InputRef, NanoGraph, ScalarBinOp, ScalarOp, ScalarUnaryOp};
    use crate::numeric_scalar::NumericScalar;

    /// Build a simple matmul C[M,N] = A[M,K] @ B[K,N].
    /// Returns (graph, mul_group_indices, reduce_group_indices).
    fn build_matmul(m: u64, k: u64, n: u64) -> NanoGraph {
        let mut g = NanoGraph::new();

        // Weight A: M*K values (one group per row).
        let a_base = g.push_group(
            m * k,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        // Weight B: K*N values.
        let b_base = g.push_group(
            k * n,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        // M Mul groups, each of size K*N.
        // Mul group m: input0 = StridedBroadcast(A[m,0], stride=1, repeat=N)
        //              input1 = Affine(B[0,0], stride=1)
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

        // M ReduceSum groups, each of size N.
        // ReduceSum group m: input = Affine(mul_group_m, stride=1)
        //                    reduce_count=K, reduce_stride=N
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

        // Mark outputs.
        for &rb in &reduce_bases {
            for i in 0..n {
                g.outputs.push(AtomId(rb.0 + i));
            }
        }

        g
    }

    /// Build a chain: matmul1 -> elementwise op -> matmul2.
    fn build_matmul_chain(m: u64, k1: u64, n1: u64, k2: u64, n2: u64) -> NanoGraph {
        let mut g = NanoGraph::new();

        // Weights for matmul 1.
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

        // Matmul 1: M*K1*N1 Mul groups, M*N1 ReduceSum groups.
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

        // Elementwise activation (Tanh) on matmul1 output.
        // The output of matmul1 is M groups of N1 values = M*N1 total.
        // We apply tanh to all of them as M separate groups.
        let mut act_bases = Vec::new();
        for row in 0..m {
            let act = g.push_group(
                n1,
                ScalarOp::Unary {
                    op: ScalarUnaryOp::Tanh,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![InputRef::Affine {
                    base: red1_bases[row as usize],
                    stride: 1,
                }],
            );
            act_bases.push(act);
        }

        // Weights for matmul 2. Input dimension is n1 (output of matmul1).
        let b2 = g.push_group(
            n1 * n2,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        // Matmul 2: needs to read from activation output.
        // A2[m, k] = act_bases[m][k], so A2 for row m starts at act_bases[m].
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
                        base: act_bases[row as usize],
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

    /// Build two parallel matmuls from the same input (like Q/K projections).
    fn build_parallel_matmuls(m: u64, k: u64, n: u64) -> NanoGraph {
        let mut g = NanoGraph::new();

        // Shared input A: M*K.
        let a = g.push_group(
            m * k,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        // Weight B1 for matmul 1.
        let b1 = g.push_group(
            k * n,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        // Weight B2 for matmul 2.
        let b2 = g.push_group(
            k * n,
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
            vec![],
        );

        // Matmul 1.
        let mut red1_bases = Vec::new();
        for row in 0..m {
            let a_row = AtomId(a.0 + row * k);
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
                        base: b1,
                        stride: 1,
                    },
                ],
            );
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
                    base: mul,
                    stride: 1,
                }],
            );
            red1_bases.push(red);
        }

        // Matmul 2.
        let mut red2_bases = Vec::new();
        for row in 0..m {
            let a_row = AtomId(a.0 + row * k);
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
                        base: b2,
                        stride: 1,
                    },
                ],
            );
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
                    base: mul,
                    stride: 1,
                }],
            );
            red2_bases.push(red);
        }

        // Outputs: both matmul results.
        for &rb in &red1_bases {
            for i in 0..n {
                g.outputs.push(AtomId(rb.0 + i));
            }
        }
        for &rb in &red2_bases {
            for i in 0..n {
                g.outputs.push(AtomId(rb.0 + i));
            }
        }

        g
    }

    /// Build a simple elementwise graph: two Literal inputs -> Add -> output.
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

    // ─── Validation helpers ──────────────────────────────────────────────────

    /// Verify every compute group appears exactly once across all lanes/phases.
    fn verify_coverage(graph: &NanoGraph, plan: &ExecutionPlan) {
        let groups = graph.groups();
        let is_data: Vec<bool> = groups
            .iter()
            .map(|g| matches!(g.op, ScalarOp::Literal(_)) && g.inputs.is_empty())
            .collect();

        let mut seen: HashMap<usize, (usize, usize)> = HashMap::new(); // gi -> (phase, lane)
        for (phase_idx, phase) in plan.phases.iter().enumerate() {
            for (lane_idx, lane_groups) in phase.lane_work.iter().enumerate() {
                for &gi in lane_groups {
                    if let Some(&(prev_phase, prev_lane)) = seen.get(&gi) {
                        panic!(
                            "Group {} appears in phase {}/lane {} AND phase {}/lane {}",
                            gi, prev_phase, prev_lane, phase_idx, lane_idx
                        );
                    }
                    seen.insert(gi, (phase_idx, lane_idx));
                }
            }
        }

        // Every compute group must appear.
        for gi in 0..groups.len() {
            if !is_data[gi] && !seen.contains_key(&gi) {
                panic!("Compute group {} not assigned to any lane/phase", gi);
            }
        }
    }

    /// Verify within-phase independence: no lane reads another lane's current-phase output.
    fn verify_phase_independence(graph: &NanoGraph, plan: &ExecutionPlan) {
        let groups = graph.groups();
        let (producers, _) = build_group_deps(groups);

        for (phase_idx, phase) in plan.phases.iter().enumerate() {
            // Build map: group -> lane for this phase.
            let mut group_lane: HashMap<usize, usize> = HashMap::new();
            for (lane_idx, lane_groups) in phase.lane_work.iter().enumerate() {
                for &gi in lane_groups {
                    group_lane.insert(gi, lane_idx);
                }
            }

            // Check: for each group in this phase, none of its producers
            // should be in a different lane in the same phase.
            for (lane_idx, lane_groups) in phase.lane_work.iter().enumerate() {
                for &gi in lane_groups {
                    for &pi in &producers[gi] {
                        if let Some(&prod_lane) = group_lane.get(&pi) {
                            if prod_lane != lane_idx {
                                panic!(
                                    "Phase {}: group {} (lane {}) reads from group {} (lane {}) - within-phase cross-lane dependency!",
                                    phase_idx, gi, lane_idx, pi, prod_lane
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Verify work balance: max lane work / min lane work < threshold.
    fn verify_balance(plan: &ExecutionPlan, groups: &[AtomGroup], threshold: f64) {
        for (phase_idx, phase) in plan.phases.iter().enumerate() {
            let lane_works: Vec<u64> = phase
                .lane_work
                .iter()
                .map(|lane| lane.iter().map(|&gi| groups[gi].count).sum::<u64>())
                .collect();

            let non_zero: Vec<u64> = lane_works.iter().copied().filter(|&w| w > 0).collect();
            if non_zero.len() <= 1 {
                continue; // Only one active lane, balance is trivially ok.
            }

            let max_work = *non_zero.iter().max().unwrap();
            let min_work = *non_zero.iter().min().unwrap();
            if min_work > 0 {
                let ratio = max_work as f64 / min_work as f64;
                // Relaxed check: we just want it not to be catastrophically unbalanced.
                assert!(
                    ratio < threshold,
                    "Phase {}: work imbalance {:.1}x (max={}, min={}) exceeds {}x threshold",
                    phase_idx,
                    ratio,
                    max_work,
                    min_work,
                    threshold
                );
            }
        }
    }

    // ─── Test cases ──────────────────────────────────────────────────────────

    #[test]
    fn test_single_matmul_2lanes() {
        let g = build_matmul(4, 8, 4);
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let plan = plan_execution(&g, 2);
        assert_eq!(plan.num_lanes, 2);
        assert!(!plan.phases.is_empty(), "Should have at least one phase");

        verify_coverage(&g, &plan);
        verify_phase_independence(&g, &plan);

        println!("Single matmul 2 lanes: {} phases", plan.phases.len());
        for (i, phase) in plan.phases.iter().enumerate() {
            let counts: Vec<usize> = phase.lane_work.iter().map(|l| l.len()).collect();
            println!("  Phase {}: lane group counts = {:?}", i, counts);
        }
    }

    #[test]
    fn test_single_matmul_4lanes() {
        let g = build_matmul(8, 4, 4);
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let plan = plan_execution(&g, 4);
        assert_eq!(plan.num_lanes, 4);

        verify_coverage(&g, &plan);
        verify_phase_independence(&g, &plan);

        println!("Single matmul 4 lanes: {} phases", plan.phases.len());
        for (i, phase) in plan.phases.iter().enumerate() {
            let counts: Vec<usize> = phase.lane_work.iter().map(|l| l.len()).collect();
            println!("  Phase {}: lane group counts = {:?}", i, counts);
        }
    }

    #[test]
    fn test_matmul_chain_barriers() {
        // Two matmuls in sequence: should produce at least 2 phases
        // (barrier between matmul1 reduces and matmul2 muls).
        let g = build_matmul_chain(4, 4, 4, 4, 4);
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let plan = plan_execution(&g, 2);
        assert!(
            plan.phases.len() >= 2,
            "Matmul chain should have >= 2 phases, got {}",
            plan.phases.len()
        );

        verify_coverage(&g, &plan);
        verify_phase_independence(&g, &plan);

        println!("Matmul chain 2 lanes: {} phases", plan.phases.len());
        for (i, phase) in plan.phases.iter().enumerate() {
            let counts: Vec<usize> = phase.lane_work.iter().map(|l| l.len()).collect();
            println!("  Phase {}: lane group counts = {:?}", i, counts);
        }
    }

    #[test]
    fn test_parallel_matmuls_same_phase() {
        // Two independent matmuls from same input: should be in same phase,
        // distributed across lanes.
        let g = build_parallel_matmuls(4, 4, 4);
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let plan = plan_execution(&g, 4);

        verify_coverage(&g, &plan);
        verify_phase_independence(&g, &plan);

        println!("Parallel matmuls 4 lanes: {} phases", plan.phases.len());
        for (i, phase) in plan.phases.iter().enumerate() {
            let counts: Vec<usize> = phase.lane_work.iter().map(|l| l.len()).collect();
            println!("  Phase {}: lane group counts = {:?}", i, counts);
        }
    }

    #[test]
    fn test_elementwise_single_phase() {
        // Simple elementwise: should be one phase, work distributed across lanes.
        let g = build_elementwise(1024);
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let plan = plan_execution(&g, 4);
        assert_eq!(plan.phases.len(), 1, "Elementwise should be 1 phase");

        verify_coverage(&g, &plan);
        verify_phase_independence(&g, &plan);
    }

    #[test]
    fn test_balance_matmul() {
        let g = build_matmul(8, 4, 4);
        assert!(g.validate().is_empty());

        let plan = plan_execution(&g, 4);
        verify_coverage(&g, &plan);
        verify_phase_independence(&g, &plan);
        // Relaxed: 10x threshold for now (balance is secondary to correctness).
        verify_balance(&plan, g.groups(), 10.0);
    }

    #[test]
    fn test_single_lane_trivial() {
        // With 1 lane, everything should go in one lane per phase.
        let g = build_matmul(4, 4, 4);
        let plan = plan_execution(&g, 1);
        assert_eq!(plan.num_lanes, 1);

        verify_coverage(&g, &plan);
        verify_phase_independence(&g, &plan);

        for phase in &plan.phases {
            // Only lane 0 should have work.
            assert!(phase.lane_work.len() >= 1);
            // All other lanes (if any) should be empty.
            for (i, lane) in phase.lane_work.iter().enumerate() {
                if i > 0 {
                    assert!(lane.is_empty(), "Lane {} should be empty with 1 lane", i);
                }
            }
        }
    }

    #[test]
    fn test_empty_graph() {
        let g = NanoGraph::new();
        let plan = plan_execution(&g, 4);
        assert_eq!(plan.num_lanes, 4);
        assert!(plan.phases.is_empty());
    }

    #[test]
    fn test_coverage_exhaustive() {
        // Test coverage for various configurations.
        let configs: Vec<(&str, NanoGraph, usize)> = vec![
            ("matmul_2x2_2lanes", build_matmul(2, 2, 2), 2),
            ("matmul_4x8x4_2lanes", build_matmul(4, 8, 4), 2),
            ("matmul_4x8x4_4lanes", build_matmul(4, 8, 4), 4),
            ("chain_2lanes", build_matmul_chain(2, 4, 4, 4, 2), 2),
            ("chain_4lanes", build_matmul_chain(4, 4, 4, 4, 4), 4),
            ("parallel_2lanes", build_parallel_matmuls(4, 4, 4), 2),
            ("parallel_4lanes", build_parallel_matmuls(4, 4, 4), 4),
            ("elementwise_2lanes", build_elementwise(256), 2),
        ];

        for (name, graph, lanes) in configs {
            assert!(graph.validate().is_empty(), "{}: validation failed", name);
            let plan = plan_execution(&graph, lanes);
            verify_coverage(&graph, &plan);
            verify_phase_independence(&graph, &plan);
            println!(
                "  {}: {} phases, {} lanes",
                name,
                plan.phases.len(),
                plan.num_lanes
            );
        }
    }
}
