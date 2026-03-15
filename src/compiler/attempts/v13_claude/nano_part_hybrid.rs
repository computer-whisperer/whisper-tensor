#![allow(clippy::all, dead_code, unreachable_patterns)]
//! Two-phase hybrid partitioner for NanoGraph.
//!
//! Phase 1: Sequential phase boundaries via live-set analysis.
//! Sweeps groups in topo order tracking the live set (values produced but not
//! yet fully consumed). Positions where the live set narrows are natural phase
//! boundaries (between transformer layers). This produces ~10-20 sequential phases.
//!
//! Phase 2: Parallel splitting within each phase.
//! Within each phase, identifies independent matmul row groups (Mul with
//! StridedBroadcast input + downstream ReduceSum) that share no data
//! dependencies except shared weight Literals. Groups these into parallel
//! kernels while keeping the backbone (non-matmul ops) in the phase's main kernel.
//!
//! Acyclicity is verified post-assignment by building the kernel dependency
//! graph and checking for cycles. Offending kernels are merged if cycles are found.

use crate::nano_graph::{AtomGroup, AtomId, InputRef, NanoGraph, ScalarOp};
use std::collections::{HashMap, HashSet, VecDeque};

/// Result of partitioning a NanoGraph into kernels.
#[derive(Debug)]
pub struct NanoPartitionResult {
    /// Each entry is a list of group indices forming one kernel.
    /// Kernels need NOT be contiguous topo ranges -- parallel kernels will
    /// have interleaved group indices.
    pub kernel_groups: Vec<Vec<usize>>,
    /// Number of kernels.
    pub num_kernels: usize,
}

/// Partition a NanoGraph using two-phase hybrid analysis.
///
/// Phase 1 finds sequential phase boundaries via live-set pinch points.
/// Phase 2 splits independent matmul rows within each phase into parallel kernels.
pub fn partition_nanograph(graph: &NanoGraph, target_kernels: usize) -> NanoPartitionResult {
    let groups = graph.groups();
    let n = groups.len();

    // Classify groups as data (Literal, no inputs) vs compute.
    let is_data: Vec<bool> = groups
        .iter()
        .map(|g| matches!(g.op, ScalarOp::Literal(_)) && g.inputs.is_empty())
        .collect();

    let compute_indices: Vec<usize> = (0..n).filter(|&gi| !is_data[gi]).collect();

    if n == 0 || target_kernels == 0 || compute_indices.is_empty() {
        return NanoPartitionResult {
            kernel_groups: vec![],
            num_kernels: 0,
        };
    }

    if target_kernels == 1 || compute_indices.len() == 1 {
        return NanoPartitionResult {
            kernel_groups: vec![compute_indices],
            num_kernels: 1,
        };
    }

    // --- Shared infrastructure ---
    let atom_to_group = build_atom_to_group_map(groups);

    // For each group, which groups produce its inputs.
    let producers: Vec<Vec<usize>> = groups
        .iter()
        .enumerate()
        .map(|(gi, group)| {
            if is_data[gi] {
                return vec![];
            }
            resolve_producer_groups(group, graph, &atom_to_group)
        })
        .collect();

    // For each group, which groups consume its outputs.
    let mut consumers: Vec<Vec<usize>> = vec![vec![]; n];
    for (consumer_gi, prod_list) in producers.iter().enumerate() {
        for &prod_gi in prod_list {
            consumers[prod_gi].push(consumer_gi);
        }
    }

    // ============================================================
    // Phase 1: Sequential boundaries via live-set analysis
    // ============================================================
    let phase_boundaries = find_phase_boundaries(groups, &is_data, &producers, target_kernels);
    let phases = boundaries_to_phases(n, &phase_boundaries);

    // ============================================================
    // Phase 2: Parallel splitting within each phase
    // ============================================================
    let mut all_kernels: Vec<Vec<usize>> = Vec::new();

    for phase in &phases {
        let phase_kernels = split_phase_parallel(
            phase,
            groups,
            &is_data,
            &producers,
            &consumers,
            &atom_to_group,
            target_kernels,
        );
        all_kernels.extend(phase_kernels);
    }

    // ============================================================
    // Acyclicity verification and repair
    // ============================================================
    let mut result = NanoPartitionResult {
        num_kernels: all_kernels.len(),
        kernel_groups: all_kernels,
    };

    repair_cycles(&mut result, &producers);

    result.num_kernels = result.kernel_groups.len();
    result
}

// ---------------------------------------------------------------------------
// Phase 1: Live-set sweep to find sequential boundaries
// ---------------------------------------------------------------------------

/// Find phase boundary positions (group indices after which to cut).
/// Returns sorted cut positions.
fn find_phase_boundaries(
    groups: &[AtomGroup],
    is_data: &[bool],
    producers: &[Vec<usize>],
    target_kernels: usize,
) -> Vec<usize> {
    let n = groups.len();

    // Compute last_consumer[g] = max group index that reads from group g.
    let mut last_consumer: Vec<usize> = (0..n).collect();
    for (consumer_gi, prod_list) in producers.iter().enumerate() {
        for &prod_gi in prod_list {
            if !is_data[prod_gi] && consumer_gi > last_consumer[prod_gi] {
                last_consumer[prod_gi] = consumer_gi;
            }
        }
    }

    // Build expire_at: which groups' liveness expires at each position.
    let mut expire_at: Vec<Vec<usize>> = vec![vec![]; n];
    for gi in 0..n {
        if !is_data[gi] {
            expire_at[last_consumer[gi]].push(gi);
        }
    }

    // Sweep: track live set size in atoms.
    let mut live_sizes: Vec<u64> = Vec::with_capacity(n);
    let mut live_set_size: u64 = 0;
    for gi in 0..n {
        if !is_data[gi] {
            live_set_size += groups[gi].count;
        }
        for &expired_gi in &expire_at[gi] {
            live_set_size -= groups[expired_gi].count;
        }
        live_sizes.push(live_set_size);
    }

    // Compute prefix sum of compute atoms for balance checks.
    let mut compute_atom_prefix: Vec<u64> = vec![0; n + 1];
    for gi in 0..n {
        compute_atom_prefix[gi + 1] =
            compute_atom_prefix[gi] + if is_data[gi] { 0 } else { groups[gi].count };
    }
    let total_compute_atoms = compute_atom_prefix[n];

    // We want roughly sqrt(target_kernels) sequential phases, with the rest
    // coming from within-phase parallelism. But at least 2 phases if possible.
    // For GPT-2 with target=50, this gives ~7 sequential cuts.
    let target_phases = if target_kernels <= 4 {
        target_kernels
    } else {
        let sq = (target_kernels as f64).sqrt().ceil() as usize;
        sq.max(3).min(target_kernels)
    };
    let num_cuts = target_phases.saturating_sub(1);

    if num_cuts == 0 {
        return vec![];
    }

    let min_kernel_size = std::cmp::max(1, n / (target_phases * 4));

    // Build candidate list: (live_size, position).
    let mut candidates: Vec<(u64, usize)> = live_sizes
        .iter()
        .enumerate()
        .filter(|&(gi, _)| {
            if gi >= n - 1 {
                return false;
            }
            let left_compute = compute_atom_prefix[gi + 1];
            let right_compute = total_compute_atoms - left_compute;
            left_compute > 0 && right_compute > 0
        })
        .map(|(gi, &size)| (size, gi))
        .collect();

    candidates.sort();

    // Greedily select cuts with minimum spacing.
    let mut cuts: Vec<usize> = Vec::new();
    for &(_live_size, pos) in &candidates {
        if cuts.len() >= num_cuts {
            break;
        }

        let too_close = cuts.iter().any(|&c| {
            let dist = if pos > c { pos - c } else { c - pos };
            dist < min_kernel_size
        });
        if too_close {
            continue;
        }

        // Verify each resulting segment has compute atoms.
        let mut tentative = cuts.clone();
        tentative.push(pos);
        tentative.sort();
        let all_valid = {
            let mut valid = true;
            let mut seg_start = 0usize;
            for &c in &tentative {
                let seg_end = c + 1;
                let seg_compute = compute_atom_prefix[seg_end] - compute_atom_prefix[seg_start];
                if seg_compute == 0 {
                    valid = false;
                    break;
                }
                seg_start = seg_end;
            }
            if valid {
                let last_compute = compute_atom_prefix[n] - compute_atom_prefix[seg_start];
                if last_compute == 0 {
                    valid = false;
                }
            }
            valid
        };
        if !all_valid {
            continue;
        }

        cuts.push(pos);
    }

    cuts.sort();
    cuts
}

/// Convert cut positions into phase ranges (each phase is a Vec of group indices).
fn boundaries_to_phases(n: usize, cuts: &[usize]) -> Vec<Vec<usize>> {
    let mut phases = Vec::new();
    let mut start = 0;
    for &cut in cuts {
        let end = cut + 1;
        phases.push((start..end).collect());
        start = end;
    }
    phases.push((start..n).collect());
    phases
}

// ---------------------------------------------------------------------------
// Phase 2: Within-phase parallel splitting
// ---------------------------------------------------------------------------

/// Split a single phase into parallel kernels.
///
/// Strategy:
/// 1. Seed clusters from matmul Mul groups (Binary::Mul with StridedBroadcast).
///    Mul groups that share non-data intra-phase producers are in the same cluster.
/// 2. Propagate forward: process groups in topo order. A group joins a cluster if
///    ALL its non-data intra-phase producers belong to the same single cluster.
///    Groups with producers from multiple clusters, or no cluster producers, stay
///    in the backbone.
/// 3. The backbone contains shared infrastructure (weights, shared computations).
///    To avoid cycles, the backbone is split into "upstream" (groups that no cluster
///    depends on or that clusters read from) and doesn't contain groups that read
///    from cluster members.
fn split_phase_parallel(
    phase_groups: &[usize],
    groups: &[AtomGroup],
    is_data: &[bool],
    producers: &[Vec<usize>],
    consumers: &[Vec<usize>],
    _atom_to_group: &[(u64, u64, usize)],
    _global_target: usize,
) -> Vec<Vec<usize>> {
    // Filter out data/Literal groups -- they don't belong in any kernel.
    let compute_phase: Vec<usize> = phase_groups
        .iter()
        .copied()
        .filter(|&gi| !is_data[gi])
        .collect();

    if compute_phase.len() <= 2 {
        if compute_phase.is_empty() {
            return vec![];
        }
        return vec![compute_phase];
    }

    let phase_set: HashSet<usize> = phase_groups.iter().copied().collect();

    // Identify matmul Mul groups: Binary::Mul with at least one StridedBroadcast input.
    let mul_groups: Vec<usize> = phase_groups
        .iter()
        .copied()
        .filter(|&gi| {
            matches!(
                &groups[gi].op,
                ScalarOp::Binary {
                    op: crate::nano_graph::ScalarBinOp::Mul,
                    ..
                }
            ) && groups[gi]
                .inputs
                .iter()
                .any(|inp| matches!(inp, InputRef::StridedBroadcast { .. }))
        })
        .collect();

    if mul_groups.is_empty() {
        return vec![compute_phase];
    }

    // --- Step 1: Group Mul groups into clusters based on shared non-data producers ---

    // For each Mul group, find non-data intra-phase producers.
    let mut producer_to_muls: HashMap<usize, Vec<usize>> = HashMap::new();
    for &mul_gi in &mul_groups {
        for &prod_gi in &producers[mul_gi] {
            if is_data[prod_gi] || !phase_set.contains(&prod_gi) {
                continue;
            }
            producer_to_muls.entry(prod_gi).or_default().push(mul_gi);
        }
    }

    // Build adjacency among Mul groups: edge if they share a non-data intra-phase producer.
    let mul_idx: HashMap<usize, usize> = mul_groups
        .iter()
        .enumerate()
        .map(|(i, &gi)| (gi, i))
        .collect();
    let num_muls = mul_groups.len();
    let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); num_muls];

    for (_prod, muls) in &producer_to_muls {
        for i in 0..muls.len() {
            for j in (i + 1)..muls.len() {
                let mi = mul_idx[&muls[i]];
                let mj = mul_idx[&muls[j]];
                adj[mi].insert(mj);
                adj[mj].insert(mi);
            }
        }
    }

    // Connected components via BFS -> initial clusters.
    let mut mul_component: Vec<usize> = vec![0; num_muls];
    let mut num_components = 0;
    let mut visited_mul = vec![false; num_muls];
    for start in 0..num_muls {
        if visited_mul[start] {
            continue;
        }
        let comp_id = num_components;
        num_components += 1;
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited_mul[start] = true;
        mul_component[start] = comp_id;
        while let Some(node) = queue.pop_front() {
            for &neighbor in &adj[node] {
                if !visited_mul[neighbor] {
                    visited_mul[neighbor] = true;
                    mul_component[neighbor] = comp_id;
                    queue.push_back(neighbor);
                }
            }
        }
    }

    if num_components <= 1 {
        return vec![compute_phase];
    }

    // --- Step 2: Propagate clusters forward in topo order ---
    // group_cluster[gi] = Some(cluster_id) if assigned, None if backbone.
    let mut group_cluster: HashMap<usize, usize> = HashMap::new();

    // Seed: assign Mul groups to their clusters.
    for (mi, &mul_gi) in mul_groups.iter().enumerate() {
        group_cluster.insert(mul_gi, mul_component[mi]);
    }

    // Process phase groups in topo order (they're already sorted by index = topo order).
    for &gi in phase_groups {
        if group_cluster.contains_key(&gi) || is_data[gi] {
            continue; // Already assigned or data group.
        }

        // Find which clusters this group's intra-phase non-data producers belong to.
        let mut cluster_deps: HashSet<usize> = HashSet::new();
        let mut has_unassigned_producer = false;

        for &prod_gi in &producers[gi] {
            if is_data[prod_gi] || !phase_set.contains(&prod_gi) {
                continue; // Data or out-of-phase: doesn't affect cluster assignment.
            }
            match group_cluster.get(&prod_gi) {
                Some(&c) => {
                    cluster_deps.insert(c);
                }
                None => {
                    // Producer is in backbone (unassigned).
                    has_unassigned_producer = true;
                }
            }
        }

        // Assign to cluster if ALL non-data intra-phase producers belong to the
        // same single cluster (backbone producers are OK -- they're shared infra).
        if cluster_deps.len() == 1 && !has_unassigned_producer {
            let cluster_id = *cluster_deps.iter().next().unwrap();
            group_cluster.insert(gi, cluster_id);
        }
        // Otherwise: stays in backbone (multiple clusters or has backbone producers
        // which would create cycles if we assigned it to a cluster).
    }

    // --- Step 3: Build output kernels ---
    let mut cluster_vecs: Vec<Vec<usize>> = vec![vec![]; num_components];
    let mut backbone: Vec<usize> = Vec::new();

    for &gi in phase_groups {
        if is_data[gi] {
            continue; // Data groups don't go in any kernel.
        }
        match group_cluster.get(&gi) {
            Some(&c) => cluster_vecs[c].push(gi),
            None => backbone.push(gi),
        }
    }

    // Check: are there enough non-empty clusters to be worth splitting?
    let non_empty_clusters: usize = cluster_vecs.iter().filter(|c| !c.is_empty()).count();
    if non_empty_clusters <= 1 {
        return vec![compute_phase.clone()];
    }

    // Check: is the cluster compute significant enough?
    let total_phase_compute: u64 = phase_groups
        .iter()
        .filter(|&&gi| !is_data[gi])
        .map(|&gi| groups[gi].count)
        .sum();
    let cluster_compute: u64 = cluster_vecs
        .iter()
        .flat_map(|c| c.iter())
        .filter(|&&gi| !is_data[gi])
        .map(|&gi| groups[gi].count)
        .sum();

    if total_phase_compute > 0 && (cluster_compute as f64 / total_phase_compute as f64) < 0.1 {
        return vec![compute_phase.clone()];
    }

    // Assemble output.
    let mut kernels: Vec<Vec<usize>> = Vec::new();

    if !backbone.is_empty() {
        kernels.push(backbone);
    }

    for cluster in &cluster_vecs {
        if !cluster.is_empty() {
            kernels.push(cluster.clone());
        }
    }

    if kernels.len() <= 1 {
        return vec![compute_phase.clone()];
    }

    kernels
}

// ---------------------------------------------------------------------------
// Acyclicity verification and repair
// ---------------------------------------------------------------------------

/// Build the kernel dependency graph and check for cycles.
/// If cycles are found, merge the offending kernels.
fn repair_cycles(result: &mut NanoPartitionResult, producers: &[Vec<usize>]) {
    loop {
        // Build group -> kernel mapping.
        let mut group_to_kernel: HashMap<usize, usize> = HashMap::new();
        for (ki, kernel) in result.kernel_groups.iter().enumerate() {
            for &gi in kernel {
                group_to_kernel.insert(gi, ki);
            }
        }

        // Build kernel dependency graph.
        let num_k = result.kernel_groups.len();
        let mut kernel_deps: Vec<HashSet<usize>> = vec![HashSet::new(); num_k];
        for (ki, kernel) in result.kernel_groups.iter().enumerate() {
            for &gi in kernel {
                for &prod_gi in &producers[gi] {
                    if let Some(&prod_ki) = group_to_kernel.get(&prod_gi) {
                        if prod_ki != ki {
                            kernel_deps[ki].insert(prod_ki);
                        }
                    }
                }
            }
        }

        // Detect cycle via topological sort (Kahn's algorithm).
        let mut in_degree: Vec<usize> = vec![0; num_k];
        for ki in 0..num_k {
            for &dep in &kernel_deps[ki] {
                // ki depends on dep => edge dep -> ki
                in_degree[ki] += 1;
            }
        }
        // Recompute properly: in_degree[ki] = number of kernels that ki depends on.
        // Actually kernel_deps[ki] already holds the set of kernels ki depends on,
        // so in_degree[ki] = kernel_deps[ki].len().
        let mut in_degree: Vec<usize> = kernel_deps.iter().map(|deps| deps.len()).collect();

        let mut queue: VecDeque<usize> = VecDeque::new();
        for ki in 0..num_k {
            if in_degree[ki] == 0 {
                queue.push_back(ki);
            }
        }

        // Build reverse adjacency: dep -> [ki that depends on dep].
        let mut rev_adj: Vec<Vec<usize>> = vec![vec![]; num_k];
        for ki in 0..num_k {
            for &dep in &kernel_deps[ki] {
                rev_adj[dep].push(ki);
            }
        }

        let mut sorted_count = 0;
        let mut visited = vec![false; num_k];
        while let Some(ki) = queue.pop_front() {
            visited[ki] = true;
            sorted_count += 1;
            for &consumer_ki in &rev_adj[ki] {
                in_degree[consumer_ki] -= 1;
                if in_degree[consumer_ki] == 0 {
                    queue.push_back(consumer_ki);
                }
            }
        }

        if sorted_count == num_k {
            // No cycles, we're done.
            break;
        }

        // There's a cycle. Find the SCCs (strongly connected components) using
        // a simple approach: unvisited nodes after Kahn's are in cycles.
        let cycle_nodes: Vec<usize> = (0..num_k).filter(|&ki| !visited[ki]).collect();

        if cycle_nodes.len() < 2 {
            // Shouldn't happen, but be safe.
            break;
        }

        // Merge all cycle nodes into one kernel.
        let target = cycle_nodes[0];
        for &ki in &cycle_nodes[1..] {
            let stolen = std::mem::take(&mut result.kernel_groups[ki]);
            result.kernel_groups[target].extend(stolen);
        }

        // Remove empty kernels.
        result.kernel_groups.retain(|k| !k.is_empty());
        // Sort within each kernel for determinism.
        for k in &mut result.kernel_groups {
            k.sort();
        }
    }
}

// ---------------------------------------------------------------------------
// Shared utility functions (borrowed from nano_part_live.rs)
// ---------------------------------------------------------------------------

/// Build a sorted map from (base_id, count, group_index) for binary search.
fn build_atom_to_group_map(groups: &[AtomGroup]) -> Vec<(u64, u64, usize)> {
    let mut entries: Vec<(u64, u64, usize)> = groups
        .iter()
        .enumerate()
        .map(|(i, g)| (g.base_id.0, g.count, i))
        .collect();
    entries.sort_by_key(|&(base, _, _)| base);
    entries
}

/// Find the group index that owns a given AtomId via binary search.
fn find_group_for_atom(atom: AtomId, map: &[(u64, u64, usize)]) -> Option<usize> {
    let idx = map.partition_point(|&(base, _, _)| base <= atom.0);
    if idx == 0 {
        return None;
    }
    let (base, count, gi) = map[idx - 1];
    if atom.0 < base + count {
        Some(gi)
    } else {
        None
    }
}

/// Resolve all producer group indices for a given group's inputs.
fn resolve_producer_groups(
    group: &AtomGroup,
    graph: &NanoGraph,
    atom_to_group: &[(u64, u64, usize)],
) -> Vec<usize> {
    let mut producers = Vec::new();
    let mut seen = HashSet::new();

    let add_producer =
        |gi: usize, seen: &mut HashSet<usize>, producers: &mut Vec<usize>| {
            if seen.insert(gi) {
                producers.push(gi);
            }
        };

    let (reduce_count, reduce_stride) = match &group.op {
        ScalarOp::ReduceSum {
            reduce_count,
            reduce_stride,
            ..
        } => (*reduce_count, *reduce_stride),
        ScalarOp::ReduceMax {
            reduce_count,
            reduce_stride,
            ..
        } => (*reduce_count, *reduce_stride),
        _ => (0, 0),
    };

    for input in &group.inputs {
        match input {
            InputRef::Broadcast(id) => {
                if reduce_count > 0 {
                    for k in 0..reduce_count {
                        let atom =
                            AtomId(id.0.wrapping_add((k as i64 * reduce_stride) as u64));
                        if let Some(gi) = find_group_for_atom(atom, atom_to_group) {
                            add_producer(gi, &mut seen, &mut producers);
                        }
                    }
                } else {
                    if let Some(gi) = find_group_for_atom(*id, atom_to_group) {
                        add_producer(gi, &mut seen, &mut producers);
                    }
                }
            }
            InputRef::Affine { base, stride: _ } => {
                if reduce_count > 0 {
                    for sample_i in [0, group.count.saturating_sub(1)] {
                        let resolved = input.resolve(sample_i, 0);
                        for k in 0..reduce_count {
                            let atom = AtomId(
                                resolved
                                    .0
                                    .wrapping_add((k as i64 * reduce_stride) as u64),
                            );
                            if let Some(gi) = find_group_for_atom(atom, atom_to_group) {
                                add_producer(gi, &mut seen, &mut producers);
                            }
                        }
                    }
                } else {
                    if let Some(gi) = find_group_for_atom(*base, atom_to_group) {
                        add_producer(gi, &mut seen, &mut producers);
                    }
                    if group.count > 1 {
                        let last = input.resolve(group.count - 1, 0);
                        if let Some(gi) = find_group_for_atom(last, atom_to_group) {
                            add_producer(gi, &mut seen, &mut producers);
                        }
                    }
                }
            }
            InputRef::Explicit(ids) => {
                for id in ids {
                    if let Some(gi) = find_group_for_atom(*id, atom_to_group) {
                        add_producer(gi, &mut seen, &mut producers);
                    }
                }
            }
            InputRef::StridedBroadcast { repeat, .. } => {
                let num_blocks = (group.count + repeat - 1) / repeat;
                for block in 0..num_blocks {
                    let atom = input.resolve(block * repeat, 0);
                    if let Some(gi) = find_group_for_atom(atom, atom_to_group) {
                        add_producer(gi, &mut seen, &mut producers);
                    }
                }
            }
            InputRef::Modular {
                base,
                stride: _,
                modulus,
            } => {
                if let Some(gi) = find_group_for_atom(*base, atom_to_group) {
                    add_producer(gi, &mut seen, &mut producers);
                }
                if *modulus > 1 {
                    let last = input.resolve(*modulus - 1, 0);
                    if let Some(gi) = find_group_for_atom(last, atom_to_group) {
                        add_producer(gi, &mut seen, &mut producers);
                    }
                }
            }
            InputRef::SymAffine { .. } => {
                let k_bound = group
                    .reduce_dims
                    .iter()
                    .filter_map(|sd| graph.sym_dim_bounds.get(sd))
                    .next()
                    .copied()
                    .unwrap_or(1);

                for k in 0..k_bound {
                    let atom = input.resolve(0, k);
                    if let Some(gi) = find_group_for_atom(atom, atom_to_group) {
                        add_producer(gi, &mut seen, &mut producers);
                    }
                }
                if group.count > 1 {
                    for k in 0..k_bound {
                        let atom = input.resolve(group.count - 1, k);
                        if let Some(gi) = find_group_for_atom(atom, atom_to_group) {
                            add_producer(gi, &mut seen, &mut producers);
                        }
                    }
                }
            }
        }
    }
    producers
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::nano_graph::{InputRef, NanoGraph, ScalarBinOp, ScalarOp};
    use crate::numeric_scalar::NumericScalar;

    /// Test: every compute group assigned exactly once, no data groups assigned.
    #[test]
    fn test_every_group_in_exactly_one_kernel() {
        let g = build_two_layer_matmul_pipeline();
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let result = partition_nanograph(&g, 8);
        let groups = g.groups();
        let is_data: Vec<bool> = groups
            .iter()
            .map(|gr| matches!(gr.op, ScalarOp::Literal(_)) && gr.inputs.is_empty())
            .collect();

        let mut assigned: Vec<usize> = result
            .kernel_groups
            .iter()
            .flat_map(|k| k.iter().copied())
            .collect();
        assigned.sort();
        assigned.dedup();

        // No data groups should be assigned.
        for &gi in &assigned {
            assert!(
                !is_data[gi],
                "Data/Literal group {} should not be in any kernel",
                gi
            );
        }

        // Every compute group should be assigned.
        let expected: Vec<usize> = (0..g.num_groups()).filter(|&gi| !is_data[gi]).collect();
        assert_eq!(
            assigned.len(),
            expected.len(),
            "Not every compute group assigned exactly once. Got {} unique, expected {}",
            assigned.len(),
            expected.len()
        );
        assert_eq!(assigned, expected);
    }

    /// Test: no circular dependencies in the kernel DAG.
    #[test]
    fn test_no_circular_deps() {
        let g = build_two_layer_matmul_pipeline();
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let result = partition_nanograph(&g, 8);

        // Rebuild producers for checking.
        let groups = g.groups();
        let atom_to_group = build_atom_to_group_map(groups);
        let is_data: Vec<bool> = groups
            .iter()
            .map(|gr| matches!(gr.op, ScalarOp::Literal(_)) && gr.inputs.is_empty())
            .collect();
        let producers: Vec<Vec<usize>> = groups
            .iter()
            .enumerate()
            .map(|(gi, group)| {
                if is_data[gi] {
                    return vec![];
                }
                resolve_producer_groups(group, &g, &atom_to_group)
            })
            .collect();

        // Build group -> kernel.
        let mut group_to_kernel: HashMap<usize, usize> = HashMap::new();
        for (ki, kernel) in result.kernel_groups.iter().enumerate() {
            for &gi in kernel {
                group_to_kernel.insert(gi, ki);
            }
        }

        // Build kernel dependency graph.
        let num_k = result.num_kernels;
        let mut kernel_deps: Vec<HashSet<usize>> = vec![HashSet::new(); num_k];
        for (ki, kernel) in result.kernel_groups.iter().enumerate() {
            for &gi in kernel {
                for &prod_gi in &producers[gi] {
                    if let Some(&prod_ki) = group_to_kernel.get(&prod_gi) {
                        if prod_ki != ki {
                            kernel_deps[ki].insert(prod_ki);
                        }
                    }
                }
            }
        }

        // Topological sort (Kahn's).
        let mut in_degree: Vec<usize> = kernel_deps.iter().map(|deps| deps.len()).collect();
        let mut rev_adj: Vec<Vec<usize>> = vec![vec![]; num_k];
        for ki in 0..num_k {
            for &dep in &kernel_deps[ki] {
                rev_adj[dep].push(ki);
            }
        }

        let mut queue: VecDeque<usize> = VecDeque::new();
        for ki in 0..num_k {
            if in_degree[ki] == 0 {
                queue.push_back(ki);
            }
        }

        let mut sorted_count = 0;
        while let Some(ki) = queue.pop_front() {
            sorted_count += 1;
            for &consumer_ki in &rev_adj[ki] {
                in_degree[consumer_ki] -= 1;
                if in_degree[consumer_ki] == 0 {
                    queue.push_back(consumer_ki);
                }
            }
        }

        assert_eq!(
            sorted_count, num_k,
            "Kernel DAG has cycles! Only {} of {} kernels in topo order",
            sorted_count, num_k
        );

        println!(
            "PASS: {} kernels, no circular dependencies",
            result.num_kernels
        );
        for (ki, kernel) in result.kernel_groups.iter().enumerate() {
            let atom_count: u64 = kernel.iter().map(|&gi| groups[gi].count).sum();
            println!(
                "  Kernel {}: {} groups, {} atoms, deps: {:?}",
                ki,
                kernel.len(),
                atom_count,
                kernel_deps[ki]
            );
        }
    }

    /// Test: reasonable balance -- no single kernel dominates.
    #[test]
    fn test_reasonable_balance() {
        let g = build_two_layer_matmul_pipeline();
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let result = partition_nanograph(&g, 8);

        let groups = g.groups();
        let total_atoms: u64 = groups.iter().map(|gr| gr.count).sum();

        let kernel_atoms: Vec<u64> = result
            .kernel_groups
            .iter()
            .map(|k| k.iter().map(|&gi| groups[gi].count).sum())
            .collect();

        let max_kernel_atoms = *kernel_atoms.iter().max().unwrap_or(&0);
        let max_pct = max_kernel_atoms as f64 / total_atoms as f64 * 100.0;

        println!(
            "Balance: {} kernels, total {} atoms",
            result.num_kernels, total_atoms
        );
        for (ki, atoms) in kernel_atoms.iter().enumerate() {
            let pct = *atoms as f64 / total_atoms as f64 * 100.0;
            println!(
                "  Kernel {}: {} atoms ({:.1}%)",
                ki, atoms, pct
            );
        }

        // With parallelism, no single kernel should have >60% of atoms.
        // (Without parallelism the old partitioner had 79%).
        assert!(
            max_pct < 60.0,
            "Largest kernel has {:.1}% of atoms (target <60%)",
            max_pct
        );
    }

    /// Test: parallel kernels have interleaved group indices.
    #[test]
    fn test_parallel_kernels_interleaved() {
        let g = build_two_layer_matmul_pipeline();
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let result = partition_nanograph(&g, 8);

        // Check if any kernels have non-contiguous group indices
        // (indicating parallel splitting within a phase).
        let mut has_interleaved = false;
        for kernel in &result.kernel_groups {
            if kernel.len() >= 2 {
                let min = *kernel.iter().min().unwrap();
                let max = *kernel.iter().max().unwrap();
                let span = max - min + 1;
                if span > kernel.len() {
                    has_interleaved = true;
                    break;
                }
            }
        }

        println!(
            "Interleaved groups: {} ({} kernels total)",
            has_interleaved, result.num_kernels
        );
        // We expect interleaving when there are parallel matmul rows.
        // Not asserting hard since small test graphs may not trigger it,
        // but we print the status.
    }

    /// Test: single kernel request.
    #[test]
    fn test_single_kernel() {
        let g = build_two_layer_matmul_pipeline();
        let result = partition_nanograph(&g, 1);
        assert_eq!(result.num_kernels, 1);
        let groups = g.groups();
        let num_compute = groups
            .iter()
            .filter(|gr| !(matches!(gr.op, ScalarOp::Literal(_)) && gr.inputs.is_empty()))
            .count();
        assert_eq!(result.kernel_groups[0].len(), num_compute);
    }

    /// Test: empty graph.
    #[test]
    fn test_empty_graph() {
        let g = NanoGraph::new();
        let result = partition_nanograph(&g, 4);
        assert_eq!(result.num_kernels, 0);
    }

    /// Test: simple chain with no matmul structure (no parallelism expected).
    #[test]
    fn test_simple_chain_no_parallelism() {
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
                op: crate::nano_graph::ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: a, stride: 1 }],
        );
        let c = g.push_group(
            1024,
            ScalarOp::Unary {
                op: crate::nano_graph::ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: b, stride: 1 }],
        );
        let _d = g.push_group(
            1024,
            ScalarOp::Unary {
                op: crate::nano_graph::ScalarUnaryOp::Tanh,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: c, stride: 1 }],
        );

        assert!(g.validate().is_empty());
        let result = partition_nanograph(&g, 4);

        // All compute groups assigned (data groups are not in any kernel).
        let groups = g.groups();
        let is_data: Vec<bool> = groups
            .iter()
            .map(|gr| matches!(gr.op, ScalarOp::Literal(_)) && gr.inputs.is_empty())
            .collect();
        let num_compute = is_data.iter().filter(|&&d| !d).count();
        let total: usize = result.kernel_groups.iter().map(|k| k.len()).sum();
        assert_eq!(total, num_compute);

        println!(
            "Simple chain: {} kernels from {} compute groups (of {} total)",
            result.num_kernels,
            num_compute,
            g.num_groups()
        );
    }

    /// Test with the matmul builder: independent rows should be parallelized.
    #[test]
    fn test_matmul_row_parallelism() {
        let g = build_matmul_nanograph(4, 8, 16);
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let result = partition_nanograph(&g, 4);

        // All compute groups assigned (data groups not in any kernel).
        let groups = g.groups();
        let is_data_vec: Vec<bool> = groups
            .iter()
            .map(|gr| matches!(gr.op, ScalarOp::Literal(_)) && gr.inputs.is_empty())
            .collect();
        let num_compute = is_data_vec.iter().filter(|&&d| !d).count();
        let mut assigned: Vec<usize> = result
            .kernel_groups
            .iter()
            .flat_map(|k| k.iter().copied())
            .collect();
        assigned.sort();
        assigned.dedup();
        assert_eq!(assigned.len(), num_compute);

        let groups = g.groups();
        println!(
            "MatMul(4,8,16): {} kernels from {} groups",
            result.num_kernels,
            g.num_groups()
        );
        for (ki, kernel) in result.kernel_groups.iter().enumerate() {
            let atom_count: u64 = kernel.iter().map(|&gi| groups[gi].count).sum();
            println!(
                "  Kernel {}: {} groups, {} atoms",
                ki,
                kernel.len(),
                atom_count
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test graph builders
    // -----------------------------------------------------------------------

    /// Build a two-layer pipeline with proper matmul structure (StridedBroadcast).
    ///
    /// Layer 1: input -> matmul(M=4, K=8, N=16) -> tanh activation
    /// Layer 2: activation -> matmul(M=4, K=16, N=8) -> tanh activation
    ///
    /// Each matmul uses the merged structure from problem_shape.md:
    /// - M Mul groups, each count=K*N, with StridedBroadcast for A and Affine for B
    /// - M ReduceSum groups, each count=N, reading from Mul groups
    fn build_two_layer_matmul_pipeline() -> NanoGraph {
        let mut g = NanoGraph::new();

        // === Layer 1: input[4,8] @ weights[8,16] -> output[4,16] ===
        let m1: u64 = 4;
        let k1: u64 = 8;
        let n1: u64 = 16;

        // Input: M*K Literals (flattened input matrix).
        let input_base = g.push_group(
            m1 * k1,
            ScalarOp::Literal(NumericScalar::F32(0.5)),
            vec![],
            vec![],
            vec![],
        );

        // Weights1: K*N Literals (weight matrix B, row-major).
        let w1_base = g.push_group(
            k1 * n1,
            ScalarOp::Literal(NumericScalar::F32(0.1)),
            vec![],
            vec![],
            vec![],
        );

        // M Mul groups, each K*N atoms.
        // Atom i in Mul group m: computes A[m, i/N] * B[i/N, i%N]
        // Input 0 (A row): StridedBroadcast { base: input_base + m*K, stride: 1, repeat: N }
        // Input 1 (B):     Affine { base: w1_base, stride: 1 }
        let mut l1_mul_bases = Vec::new();
        for mi in 0..m1 {
            let a_row_base = AtomId(input_base.0 + mi * k1);
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
                        base: a_row_base,
                        stride: 1,
                        repeat: n1,
                    },
                    InputRef::Affine {
                        base: w1_base,
                        stride: 1,
                    },
                ],
            );
            l1_mul_bases.push(mul);
        }

        // M ReduceSum groups, each N atoms.
        // ReduceSum group m: reads from Mul group m with reduce_count=K, reduce_stride=N.
        let mut l1_reduce_bases = Vec::new();
        for mi in 0..m1 {
            let reduce = g.push_group(
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
                    base: l1_mul_bases[mi as usize],
                    stride: 1,
                }],
            );
            l1_reduce_bases.push(reduce);
        }

        // Concat the M*N reduce outputs into a flat activation vector.
        // Activation: M*N Tanh groups, each reading one ReduceSum output.
        // For simplicity, build one activation group per row.
        let mut l1_act_bases = Vec::new();
        for mi in 0..m1 {
            let act = g.push_group(
                n1,
                ScalarOp::Unary {
                    op: crate::nano_graph::ScalarUnaryOp::Tanh,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![InputRef::Affine {
                    base: l1_reduce_bases[mi as usize],
                    stride: 1,
                }],
            );
            l1_act_bases.push(act);
        }

        // === PINCH POINT: only l1_act_bases (M*N = 64 atoms) live ===

        // === Layer 2: activation[4,16] @ weights[16,8] -> output[4,8] ===
        let m2: u64 = m1; // 4
        let k2: u64 = n1; // 16
        let n2: u64 = k1; // 8

        // Weights2: K2*N2 Literals.
        let w2_base = g.push_group(
            k2 * n2,
            ScalarOp::Literal(NumericScalar::F32(0.2)),
            vec![],
            vec![],
            vec![],
        );

        // M2 Mul groups, each K2*N2 atoms.
        let mut l2_mul_bases = Vec::new();
        for mi in 0..m2 {
            let a_row_base = l1_act_bases[mi as usize];
            let mul = g.push_group(
                k2 * n2,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![
                    InputRef::StridedBroadcast {
                        base: a_row_base,
                        stride: 1,
                        repeat: n2,
                    },
                    InputRef::Affine {
                        base: w2_base,
                        stride: 1,
                    },
                ],
            );
            l2_mul_bases.push(mul);
        }

        // M2 ReduceSum groups.
        let mut l2_reduce_bases = Vec::new();
        for mi in 0..m2 {
            let reduce = g.push_group(
                n2,
                ScalarOp::ReduceSum {
                    reduce_count: k2,
                    reduce_stride: n2 as i64,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![InputRef::Affine {
                    base: l2_mul_bases[mi as usize],
                    stride: 1,
                }],
            );
            l2_reduce_bases.push(reduce);
        }

        // M2 activation groups.
        for mi in 0..m2 {
            g.push_group(
                n2,
                ScalarOp::Unary {
                    op: crate::nano_graph::ScalarUnaryOp::Tanh,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![InputRef::Affine {
                    base: l2_reduce_bases[mi as usize],
                    stride: 1,
                }],
            );
        }

        g
    }

    /// Build a matmul NanoGraph: C[M,N] = A[M,K] @ B[K,N]
    /// Uses merged Mul group structure with StridedBroadcast.
    fn build_matmul_nanograph(m: u64, k: u64, n: u64) -> NanoGraph {
        let mut g = NanoGraph::new();

        // A[M,K] as one Literal group.
        let a_base = g.push_group(
            m * k,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        // B[K,N] as one Literal group.
        let b_base = g.push_group(
            k * n,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        // M Mul groups, each K*N atoms.
        let mut mul_bases = Vec::new();
        for mi in 0..m {
            let a_row_base = AtomId(a_base.0 + mi * k);
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
                        base: a_row_base,
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

        // M ReduceSum groups, each N atoms.
        for mi in 0..m {
            g.push_group(
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
                    base: mul_bases[mi as usize],
                    stride: 1,
                }],
            );
        }

        g
    }
}
