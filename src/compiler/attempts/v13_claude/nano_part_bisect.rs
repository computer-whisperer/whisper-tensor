#![allow(clippy::all, dead_code, unreachable_patterns)]
//! Top-down recursive bisection partitioner for NanoGraph.
//!
//! Starts with all groups in one kernel and recursively splits until reaching
//! the target kernel count. Each split:
//!
//! 1. Finds independent sub-DAGs within the kernel (connected components after
//!    ignoring Literal/weight groups). If multiple components exist, distributes
//!    them into two children for parallelism.
//!
//! 2. If only one component (a single dependency chain), splits at the topo
//!    position that minimizes cross-cut traffic (live-set analysis within the
//!    kernel's groups).
//!
//! Acyclicity is guaranteed: independent components have no edges between them
//! (so no cycles), and topo-order cuts within a single chain are inherently
//! acyclic.
//!
//! Literal groups (weights/constants) are shared freely — they don't create
//! inter-kernel dependencies because they're always available.

use crate::nano_graph::{AtomGroup, AtomId, InputRef, NanoGraph, ScalarOp};
use std::collections::{HashMap, HashSet, VecDeque};

/// Result of partitioning a NanoGraph into kernels.
#[derive(Debug)]
pub struct NanoPartitionResult {
    /// Each entry is a list of group indices forming one kernel.
    pub kernel_groups: Vec<Vec<usize>>,
    /// Number of kernels.
    pub num_kernels: usize,
}

/// Partition a NanoGraph into kernels using top-down recursive bisection.
///
/// `target_kernels` is the desired number of output kernels. The algorithm
/// recursively splits the largest kernel until we reach the target count.
pub fn partition_nanograph(graph: &NanoGraph, target_kernels: usize) -> NanoPartitionResult {
    let groups = graph.groups();
    let n = groups.len();

    if n == 0 || target_kernels == 0 {
        return NanoPartitionResult {
            kernel_groups: vec![],
            num_kernels: 0,
        };
    }

    if target_kernels == 1 || n == 1 {
        return NanoPartitionResult {
            kernel_groups: vec![(0..n).collect()],
            num_kernels: 1,
        };
    }

    // Precompute: classify groups as data (Literal, no inputs) vs compute.
    let is_data: Vec<bool> = groups
        .iter()
        .map(|g| matches!(g.op, ScalarOp::Literal(_)) && g.inputs.is_empty())
        .collect();

    // Precompute: producer map (sorted base_ids for binary search).
    let atom_to_group = build_atom_to_group_map(groups);

    // Precompute: for each group, its producer groups (excluding self, data-only).
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

    // Precompute: for each group, its consumer groups (reverse of producers).
    // Only non-data -> non-data edges.
    let mut consumers: Vec<Vec<usize>> = vec![vec![]; n];
    for (consumer_gi, prod_list) in producers.iter().enumerate() {
        if is_data[consumer_gi] {
            continue;
        }
        for &prod_gi in prod_list {
            if !is_data[prod_gi] {
                consumers[prod_gi].push(consumer_gi);
            }
        }
    }
    // Deduplicate consumer lists.
    for c in consumers.iter_mut() {
        c.sort_unstable();
        c.dedup();
    }

    // Atom count per group (work metric).
    let group_atoms: Vec<u64> = groups.iter().map(|g| g.count).collect();

    let ctx = BisectContext {
        n,
        is_data: &is_data,
        producers: &producers,
        consumers: &consumers,
        group_atoms: &group_atoms,
    };

    // Start with all groups in one kernel.
    let all_groups: Vec<usize> = (0..n).collect();
    let mut kernels: Vec<Vec<usize>> = vec![all_groups];

    // Repeatedly split the largest kernel until we reach target count.
    while kernels.len() < target_kernels {
        // Find the kernel with the most compute atoms to split.
        let best_idx = kernels
            .iter()
            .enumerate()
            .map(|(i, k)| {
                let compute_atoms: u64 = k
                    .iter()
                    .filter(|&&gi| !is_data[gi])
                    .map(|&gi| group_atoms[gi])
                    .sum();
                (compute_atoms, i)
            })
            .max()
            .map(|(_, i)| i)
            .unwrap();

        let kernel_to_split = kernels.swap_remove(best_idx);

        // Count compute groups.
        let compute_count = kernel_to_split
            .iter()
            .filter(|&&gi| !ctx.is_data[gi])
            .count();

        if compute_count <= 1 {
            // Can't split a kernel with 0-1 compute groups.
            kernels.push(kernel_to_split);
            break;
        }

        let (child_a, child_b) = ctx.split_kernel(&kernel_to_split);

        // Check that both children have at least one compute group.
        let a_compute = child_a.iter().any(|&gi| !ctx.is_data[gi]);
        let b_compute = child_b.iter().any(|&gi| !ctx.is_data[gi]);

        if !a_compute || !b_compute {
            // Split failed to produce two viable kernels.
            kernels.push(kernel_to_split);
            break;
        }

        kernels.push(child_a);
        kernels.push(child_b);
    }

    // Assign data (Literal) groups to the kernel that contains their first consumer.
    // During splitting, data groups may end up in multiple kernels' dependency sets.
    // We need each data group in exactly one kernel.
    // Strategy: each data group goes to the kernel where its first consumer lives.
    // If a data group has no consumers, it goes to kernel 0.
    finalize_data_group_assignment(&mut kernels, &ctx);

    let num_kernels = kernels.len();
    NanoPartitionResult {
        kernel_groups: kernels,
        num_kernels,
    }
}

struct BisectContext<'a> {
    n: usize,
    is_data: &'a [bool],
    producers: &'a [Vec<usize>],
    consumers: &'a [Vec<usize>],
    group_atoms: &'a [u64],
}

impl<'a> BisectContext<'a> {
    /// Split a kernel's groups into two children.
    ///
    /// First tries to find independent sub-DAGs (connected components among
    /// compute groups). If multiple exist, assigns them to two children in a
    /// balanced way. If only one component exists, falls back to topo-order
    /// live-set splitting.
    fn split_kernel(&self, groups: &[usize]) -> (Vec<usize>, Vec<usize>) {
        // Separate data and compute groups.
        let compute_groups: Vec<usize> = groups
            .iter()
            .filter(|&&gi| !self.is_data[gi])
            .copied()
            .collect();
        let data_groups: Vec<usize> = groups
            .iter()
            .filter(|&&gi| self.is_data[gi])
            .copied()
            .collect();

        if compute_groups.len() <= 1 {
            // Nothing to split.
            return (groups.to_vec(), vec![]);
        }

        // Build adjacency restricted to compute groups in this kernel.
        let group_set: HashSet<usize> = compute_groups.iter().copied().collect();

        // Find connected components (undirected: if A produces for B or B produces
        // for A, they're in the same component). Only consider edges where BOTH
        // endpoints are in this kernel's compute groups.
        let components = self.find_components(&compute_groups, &group_set);

        if components.len() >= 2 {
            // We have independent sub-DAGs. Distribute components into two
            // children to balance atom counts.
            let (child_a_compute, child_b_compute) =
                balance_components(&components, self.group_atoms);

            // Assign data groups to whichever child contains more of their consumers.
            let child_a_set: HashSet<usize> = child_a_compute.iter().copied().collect();
            let child_b_set: HashSet<usize> = child_b_compute.iter().copied().collect();

            let mut child_a = child_a_compute;
            let mut child_b = child_b_compute;

            for &dg in &data_groups {
                let a_consumers = self.consumers[dg]
                    .iter()
                    .filter(|&&c| child_a_set.contains(&c))
                    .count();
                let b_consumers = self.consumers[dg]
                    .iter()
                    .filter(|&&c| child_b_set.contains(&c))
                    .count();

                if a_consumers >= b_consumers {
                    child_a.push(dg);
                } else {
                    child_b.push(dg);
                }
            }

            (child_a, child_b)
        } else {
            // Single connected component — everything is on one dependency chain.
            // Fall back to topo-order splitting at the minimum live-set point.
            self.split_by_livesets(groups, &compute_groups, &data_groups, &group_set)
        }
    }

    /// Find connected components among compute groups, treating the DAG as
    /// undirected (A->B means A and B are in the same component). Only edges
    /// within `group_set` are considered.
    fn find_components(
        &self,
        compute_groups: &[usize],
        group_set: &HashSet<usize>,
    ) -> Vec<Vec<usize>> {
        let mut visited: HashSet<usize> = HashSet::new();
        let mut components: Vec<Vec<usize>> = Vec::new();

        for &gi in compute_groups {
            if visited.contains(&gi) {
                continue;
            }

            // BFS from gi.
            let mut component = Vec::new();
            let mut queue = VecDeque::new();
            queue.push_back(gi);
            visited.insert(gi);

            while let Some(cur) = queue.pop_front() {
                component.push(cur);

                // Forward edges: producers of cur that are in group_set.
                for &prod in &self.producers[cur] {
                    if group_set.contains(&prod) && !self.is_data[prod] && visited.insert(prod) {
                        queue.push_back(prod);
                    }
                }

                // Backward edges: consumers of cur that are in group_set.
                for &cons in &self.consumers[cur] {
                    if group_set.contains(&cons) && visited.insert(cons) {
                        queue.push_back(cons);
                    }
                }
            }

            components.push(component);
        }

        components
    }

    /// Split a single-component kernel at the topo position that minimizes
    /// the live set (cross-cut traffic).
    fn split_by_livesets(
        &self,
        all_groups: &[usize],
        compute_groups: &[usize],
        data_groups: &[usize],
        group_set: &HashSet<usize>,
    ) -> (Vec<usize>, Vec<usize>) {
        // Compute a local topological order of compute groups within this kernel.
        let topo_order = self.local_topo_sort(compute_groups, group_set);

        if topo_order.len() <= 1 {
            return (all_groups.to_vec(), vec![]);
        }

        // Map from group index -> position in topo_order.
        let mut topo_pos: HashMap<usize, usize> = HashMap::new();
        for (pos, &gi) in topo_order.iter().enumerate() {
            topo_pos.insert(gi, pos);
        }

        // Compute last_consumer within this kernel for each compute group.
        let topo_len = topo_order.len();
        let mut last_consumer_pos: Vec<usize> = (0..topo_len).collect(); // default: self

        for (pos, &gi) in topo_order.iter().enumerate() {
            for &prod in &self.producers[gi] {
                if self.is_data[prod] {
                    continue;
                }
                if let Some(&prod_pos) = topo_pos.get(&prod) {
                    if pos > last_consumer_pos[prod_pos] {
                        last_consumer_pos[prod_pos] = pos;
                    }
                }
            }
        }

        // Sweep: compute live set size at each topo position boundary.
        let mut expire_at: Vec<Vec<usize>> = vec![vec![]; topo_len];
        for pos in 0..topo_len {
            expire_at[last_consumer_pos[pos]].push(pos);
        }

        let mut live_sizes: Vec<u64> = Vec::with_capacity(topo_len);
        let mut live_set_size: u64 = 0;

        for pos in 0..topo_len {
            let gi = topo_order[pos];
            live_set_size += self.group_atoms[gi];

            for &expired_pos in &expire_at[pos] {
                let expired_gi = topo_order[expired_pos];
                live_set_size -= self.group_atoms[expired_gi];
            }

            live_sizes.push(live_set_size);
        }

        // Find the best cut position (minimizing live set).
        // Don't cut at the very end. Ensure both sides have compute groups.
        let total_compute_atoms: u64 = compute_groups
            .iter()
            .map(|&gi| self.group_atoms[gi])
            .sum();

        // Compute prefix sum of atoms in topo order.
        let mut prefix_atoms: Vec<u64> = vec![0; topo_len + 1];
        for pos in 0..topo_len {
            prefix_atoms[pos + 1] = prefix_atoms[pos] + self.group_atoms[topo_order[pos]];
        }

        let min_fraction = 0.1; // Each side must have at least 10% of compute atoms.
        let min_atoms = (total_compute_atoms as f64 * min_fraction) as u64;

        let mut best_cut: Option<(u64, usize)> = None; // (live_size, pos_after_which_to_cut)

        for pos in 0..(topo_len - 1) {
            let left_atoms = prefix_atoms[pos + 1];
            let right_atoms = total_compute_atoms - left_atoms;

            if left_atoms < min_atoms || right_atoms < min_atoms {
                continue;
            }

            let live = live_sizes[pos];
            if best_cut.is_none() || live < best_cut.unwrap().0 {
                best_cut = Some((live, pos));
            }
        }

        let cut_pos = match best_cut {
            Some((_, pos)) => pos,
            None => topo_len / 2, // fallback: split in the middle
        };

        // Groups in topo_order[0..=cut_pos] go to child A.
        // Groups in topo_order[cut_pos+1..] go to child B.
        let child_a_set: HashSet<usize> =
            topo_order[0..=cut_pos].iter().copied().collect();
        let child_b_set: HashSet<usize> =
            topo_order[cut_pos + 1..].iter().copied().collect();

        let mut child_a: Vec<usize> = topo_order[0..=cut_pos].to_vec();
        let mut child_b: Vec<usize> = topo_order[cut_pos + 1..].to_vec();

        // Assign data groups.
        for &dg in data_groups {
            let a_consumers = self.consumers[dg]
                .iter()
                .filter(|&&c| child_a_set.contains(&c))
                .count();
            let b_consumers = self.consumers[dg]
                .iter()
                .filter(|&&c| child_b_set.contains(&c))
                .count();

            if a_consumers >= b_consumers {
                child_a.push(dg);
            } else {
                child_b.push(dg);
            }
        }

        (child_a, child_b)
    }

    /// Topologically sort compute groups within a kernel.
    /// Only considers edges where both endpoints are in `group_set`.
    fn local_topo_sort(
        &self,
        compute_groups: &[usize],
        group_set: &HashSet<usize>,
    ) -> Vec<usize> {
        // Count in-degree for each compute group (within this kernel).
        let mut in_degree: HashMap<usize, usize> = HashMap::new();
        for &gi in compute_groups {
            in_degree.insert(gi, 0);
        }

        for &gi in compute_groups {
            for &prod in &self.producers[gi] {
                if group_set.contains(&prod) && !self.is_data[prod] {
                    *in_degree.entry(gi).or_default() += 1;
                }
            }
        }

        // Kahn's algorithm.
        let mut queue: VecDeque<usize> = compute_groups
            .iter()
            .filter(|&&gi| in_degree[&gi] == 0)
            .copied()
            .collect();

        // Sort initial queue for determinism.
        let mut sorted_queue: Vec<usize> = queue.drain(..).collect();
        sorted_queue.sort_unstable();
        queue.extend(sorted_queue);

        let mut order = Vec::with_capacity(compute_groups.len());

        while let Some(gi) = queue.pop_front() {
            order.push(gi);

            // For each consumer of gi that's in this kernel...
            for &cons in &self.consumers[gi] {
                if let Some(deg) = in_degree.get_mut(&cons) {
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(cons);
                    }
                }
            }
        }

        // If there are groups not in the topo order (shouldn't happen in a DAG,
        // but be defensive), append them.
        if order.len() < compute_groups.len() {
            let in_order: HashSet<usize> = order.iter().copied().collect();
            for &gi in compute_groups {
                if !in_order.contains(&gi) {
                    order.push(gi);
                }
            }
        }

        order
    }
}

/// Distribute connected components into two balanced sets.
///
/// Greedy algorithm: sort components by total atom count (descending),
/// assign each to the lighter side. Classic bin-packing heuristic.
fn balance_components(
    components: &[Vec<usize>],
    group_atoms: &[u64],
) -> (Vec<usize>, Vec<usize>) {
    // Compute total atoms for each component.
    let mut comp_sizes: Vec<(u64, usize)> = components
        .iter()
        .enumerate()
        .map(|(ci, comp)| {
            let total: u64 = comp.iter().map(|&gi| group_atoms[gi]).sum();
            (total, ci)
        })
        .collect();

    // Sort descending by size for greedy balance.
    comp_sizes.sort_by(|a, b| b.0.cmp(&a.0));

    let mut a_atoms: u64 = 0;
    let mut b_atoms: u64 = 0;
    let mut a_indices: Vec<usize> = Vec::new();
    let mut b_indices: Vec<usize> = Vec::new();

    for &(size, ci) in &comp_sizes {
        if a_atoms <= b_atoms {
            a_indices.push(ci);
            a_atoms += size;
        } else {
            b_indices.push(ci);
            b_atoms += size;
        }
    }

    // Flatten into group indices.
    let child_a: Vec<usize> = a_indices
        .iter()
        .flat_map(|&ci| components[ci].iter().copied())
        .collect();
    let child_b: Vec<usize> = b_indices
        .iter()
        .flat_map(|&ci| components[ci].iter().copied())
        .collect();

    (child_a, child_b)
}

/// Ensure each data group appears in exactly one kernel.
///
/// During recursive splitting, data groups may get duplicated across children
/// (both children claim the weight). This pass deduplicates: each data group
/// goes to the kernel that has the most of its consumers.
fn finalize_data_group_assignment(kernels: &mut Vec<Vec<usize>>, ctx: &BisectContext) {
    let n = ctx.n;

    // Build group -> kernel assignment for compute groups.
    let mut group_kernel: Vec<Option<usize>> = vec![None; n];
    for (ki, kernel) in kernels.iter().enumerate() {
        for &gi in kernel {
            if !ctx.is_data[gi] {
                group_kernel[gi] = Some(ki);
            }
        }
    }

    // Remove data groups from all kernels.
    for kernel in kernels.iter_mut() {
        kernel.retain(|&gi| !ctx.is_data[gi]);
    }

    // Assign each data group to the kernel with the most of its consumers.
    for gi in 0..n {
        if !ctx.is_data[gi] {
            continue;
        }

        // Count consumers in each kernel.
        let mut kernel_counts: HashMap<usize, usize> = HashMap::new();
        for &cons in &ctx.consumers[gi] {
            if let Some(ki) = group_kernel[cons] {
                *kernel_counts.entry(ki).or_default() += 1;
            }
        }

        // Also check producers (for the rare case a data group feeds another data group).
        // Actually, data groups have no producers by definition (Literal, no inputs).
        // But their consumers' kernels are what matter.

        let best_kernel = kernel_counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(ki, _)| ki)
            .unwrap_or(0); // No consumers? Put in kernel 0.

        kernels[best_kernel].push(gi);
        group_kernel[gi] = Some(best_kernel);
    }

    // Verify all groups are assigned.
    let total: usize = kernels.iter().map(|k| k.len()).sum();
    debug_assert_eq!(total, n, "Not all groups assigned after finalization");
}

// -----------------------------------------------------------------------
// Producer resolution (shared with nano_part_live.rs logic)
// -----------------------------------------------------------------------

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
///
/// For ReduceSum/ReduceMax, the op iterates k=0..reduce_count at stride reduce_stride,
/// so we must sample across the full reduction range to find all producer groups.
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

// -----------------------------------------------------------------------
// Verification utilities
// -----------------------------------------------------------------------

/// Check that the kernel dependency graph is a DAG (no circular dependencies).
///
/// Returns Ok(()) if acyclic, Err with a description if cycles exist.
pub fn verify_no_cycles(
    graph: &NanoGraph,
    result: &NanoPartitionResult,
) -> Result<(), String> {
    let groups = graph.groups();
    let n = groups.len();

    // Classify data groups.
    let is_data: Vec<bool> = groups
        .iter()
        .map(|g| matches!(g.op, ScalarOp::Literal(_)) && g.inputs.is_empty())
        .collect();

    // Build group -> kernel map.
    let mut group_kernel: Vec<usize> = vec![0; n];
    for (ki, kernel) in result.kernel_groups.iter().enumerate() {
        for &gi in kernel {
            group_kernel[gi] = ki;
        }
    }

    // Build producer info.
    let atom_to_group = build_atom_to_group_map(groups);
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

    // Build kernel dependency edges (non-data only).
    let nk = result.num_kernels;
    let mut kernel_deps: Vec<HashSet<usize>> = vec![HashSet::new(); nk];

    for gi in 0..n {
        if is_data[gi] {
            continue;
        }
        let my_kernel = group_kernel[gi];
        for &prod in &producers[gi] {
            if is_data[prod] {
                continue; // Weights don't create dependencies.
            }
            let prod_kernel = group_kernel[prod];
            if prod_kernel != my_kernel {
                kernel_deps[my_kernel].insert(prod_kernel);
            }
        }
    }

    // Check for cycles using DFS.
    // States: 0 = unvisited, 1 = in stack, 2 = done.
    let mut state: Vec<u8> = vec![0; nk];

    fn dfs(
        ki: usize,
        kernel_deps: &[HashSet<usize>],
        state: &mut [u8],
    ) -> Result<(), String> {
        state[ki] = 1;
        for &dep in &kernel_deps[ki] {
            match state[dep] {
                1 => {
                    return Err(format!(
                        "Cycle detected: kernel {} depends on kernel {} which is in the current DFS stack",
                        ki, dep
                    ));
                }
                0 => {
                    dfs(dep, kernel_deps, state)?;
                }
                _ => {} // Already fully visited.
            }
        }
        state[ki] = 2;
        Ok(())
    }

    for ki in 0..nk {
        if state[ki] == 0 {
            dfs(ki, &kernel_deps, &mut state)?;
        }
    }

    Ok(())
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::nano_graph::{InputRef, NanoGraph, ScalarBinOp, ScalarOp, ScalarUnaryOp};
    use crate::numeric_scalar::NumericScalar;

    /// Verify: every group assigned exactly once.
    fn assert_all_groups_assigned_once(graph: &NanoGraph, result: &NanoPartitionResult) {
        let n = graph.num_groups();
        let mut assigned = vec![false; n];
        for kernel in &result.kernel_groups {
            for &gi in kernel {
                assert!(
                    !assigned[gi],
                    "Group {} assigned to multiple kernels",
                    gi
                );
                assigned[gi] = true;
            }
        }
        for gi in 0..n {
            assert!(assigned[gi], "Group {} not assigned to any kernel", gi);
        }
    }

    /// Verify: no circular kernel dependencies.
    fn assert_no_cycles(graph: &NanoGraph, result: &NanoPartitionResult) {
        match verify_no_cycles(graph, result) {
            Ok(()) => {}
            Err(msg) => panic!("Cycle detected: {}", msg),
        }
    }

    /// Verify balance: no kernel has more than max_fraction of total compute atoms.
    fn assert_balanced(
        graph: &NanoGraph,
        result: &NanoPartitionResult,
        max_fraction: f64,
    ) {
        let groups = graph.groups();
        let is_data: Vec<bool> = groups
            .iter()
            .map(|g| matches!(g.op, ScalarOp::Literal(_)) && g.inputs.is_empty())
            .collect();

        let total_compute: u64 = groups
            .iter()
            .enumerate()
            .filter(|(gi, _)| !is_data[*gi])
            .map(|(_, g)| g.count)
            .sum();

        for (ki, kernel) in result.kernel_groups.iter().enumerate() {
            let kernel_compute: u64 = kernel
                .iter()
                .filter(|&&gi| !is_data[gi])
                .map(|&gi| groups[gi].count)
                .sum();

            let fraction = kernel_compute as f64 / total_compute as f64;
            assert!(
                fraction <= max_fraction,
                "Kernel {} has {:.1}% of compute atoms ({}/{}), exceeds {:.0}% limit",
                ki,
                fraction * 100.0,
                kernel_compute,
                total_compute,
                max_fraction * 100.0
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test graph builders
    // -----------------------------------------------------------------------

    /// Chain: a -> b -> c -> d -> e
    fn build_chain() -> NanoGraph {
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
            vec![InputRef::Affine { base: a, stride: 1 }],
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
            vec![InputRef::Affine { base: b, stride: 1 }],
        );
        let d = g.push_group(
            1024,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Sqrt,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: c, stride: 1 }],
        );
        let _e = g.push_group(
            1024,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Tanh,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: d, stride: 1 }],
        );
        g
    }

    /// Diamond: w -> a, w -> b, a -> c, b -> c (parallel branches).
    fn build_diamond() -> NanoGraph {
        let mut g = NanoGraph::new();
        let w = g.push_group(
            1024,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let a = g.push_group(
            1024,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: w, stride: 1 }],
        );
        let b = g.push_group(
            1024,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: w, stride: 1 }],
        );
        let _c = g.push_group(
            1024,
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
        g
    }

    /// Two independent branches from separate weights, merged at the end.
    ///
    /// w1 -> a1 -> a2 --\
    ///                    --> merge
    /// w2 -> b1 -> b2 --/
    fn build_two_branches() -> NanoGraph {
        let mut g = NanoGraph::new();
        let w1 = g.push_group(
            1024,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let w2 = g.push_group(
            1024,
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
            vec![],
        );
        let a1 = g.push_group(
            1024,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: w1,
                stride: 1,
            }],
        );
        let a2 = g.push_group(
            1024,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: a1,
                stride: 1,
            }],
        );
        let b1 = g.push_group(
            1024,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Sqrt,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: w2,
                stride: 1,
            }],
        );
        let b2 = g.push_group(
            1024,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Tanh,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: b1,
                stride: 1,
            }],
        );
        let _merge = g.push_group(
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
                    base: a2,
                    stride: 1,
                },
                InputRef::Affine {
                    base: b2,
                    stride: 1,
                },
            ],
        );
        g
    }

    /// Two-layer pipeline mimicking a simplified transformer:
    /// Layer 1: weights -> matmul -> activation
    /// Layer 2: weights -> matmul -> activation
    fn build_two_layer_pipeline() -> NanoGraph {
        let mut g = NanoGraph::new();
        let hidden = 64u64;
        let k_dim = 32u64;

        // Input data
        let input = g.push_group(
            hidden,
            ScalarOp::Literal(NumericScalar::F32(0.5)),
            vec![],
            vec![],
            vec![],
        );

        // Layer 1 weights
        let mut l1_weight_rows = Vec::new();
        for _ in 0..k_dim {
            let w = g.push_group(
                hidden,
                ScalarOp::Literal(NumericScalar::F32(0.1)),
                vec![],
                vec![],
                vec![],
            );
            l1_weight_rows.push(w);
        }

        // Layer 1 Mul groups
        let mut l1_mul_bases = Vec::new();
        for ki in 0..k_dim {
            let input_atom = AtomId(input.0 + ki);
            let weight_row = l1_weight_rows[ki as usize];
            let mul = g.push_group(
                hidden,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![
                    InputRef::Broadcast(input_atom),
                    InputRef::Affine {
                        base: weight_row,
                        stride: 1,
                    },
                ],
            );
            l1_mul_bases.push(mul);
        }

        // Layer 1 ReduceSum
        let l1_reduce = g.push_group(
            hidden,
            ScalarOp::ReduceSum {
                reduce_count: k_dim,
                reduce_stride: hidden as i64,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: l1_mul_bases[0],
                stride: 1,
            }],
        );

        // Layer 1 activation
        let l1_act = g.push_group(
            hidden,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Tanh,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: l1_reduce,
                stride: 1,
            }],
        );

        // === PINCH POINT ===

        // Layer 2 weights
        let mut l2_weight_rows = Vec::new();
        for _ in 0..k_dim {
            let w = g.push_group(
                hidden,
                ScalarOp::Literal(NumericScalar::F32(0.2)),
                vec![],
                vec![],
                vec![],
            );
            l2_weight_rows.push(w);
        }

        // Layer 2 Mul groups
        let mut l2_mul_bases = Vec::new();
        for ki in 0..k_dim {
            let act_atom = AtomId(l1_act.0 + ki);
            let weight_row = l2_weight_rows[ki as usize];
            let mul = g.push_group(
                hidden,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![
                    InputRef::Broadcast(act_atom),
                    InputRef::Affine {
                        base: weight_row,
                        stride: 1,
                    },
                ],
            );
            l2_mul_bases.push(mul);
        }

        // Layer 2 ReduceSum
        let l2_reduce = g.push_group(
            hidden,
            ScalarOp::ReduceSum {
                reduce_count: k_dim,
                reduce_stride: hidden as i64,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: l2_mul_bases[0],
                stride: 1,
            }],
        );

        // Layer 2 activation
        let _l2_act = g.push_group(
            hidden,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Tanh,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: l2_reduce,
                stride: 1,
            }],
        );

        g
    }

    /// Build a matmul NanoGraph: C[M,N] = A[M,K] @ B[K,N]
    /// Uses StridedBroadcast for the A-side (more realistic, matches GPT-2).
    fn build_matmul(m: u64, k: u64, n: u64) -> NanoGraph {
        let mut g = NanoGraph::new();

        // A[M,K] as Literal groups: M groups of K atoms each.
        let mut a_row_bases = Vec::new();
        for _ in 0..m {
            let id = g.push_group(
                k,
                ScalarOp::Literal(NumericScalar::F32(1.0)),
                vec![],
                vec![],
                vec![],
            );
            a_row_bases.push(id);
        }

        // B[K,N] as K groups of N atoms each.
        let mut b_row_bases = Vec::new();
        for _ in 0..k {
            let id = g.push_group(
                n,
                ScalarOp::Literal(NumericScalar::F32(1.0)),
                vec![],
                vec![],
                vec![],
            );
            b_row_bases.push(id);
        }

        // M Mul groups, each K*N atoms.
        // Uses StridedBroadcast for A (each block of N atoms shares one A element).
        let mut mul_bases = Vec::new();
        for mi in 0..m {
            let a_base = a_row_bases[mi as usize];
            let b_base = b_row_bases[0]; // All B rows are contiguous.
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
                        base: a_base,
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

    // -----------------------------------------------------------------------
    // Test: chain graph
    // -----------------------------------------------------------------------

    #[test]
    fn test_chain_all_assigned() {
        let g = build_chain();
        assert!(g.validate().is_empty());
        let result = partition_nanograph(&g, 3);
        assert_all_groups_assigned_once(&g, &result);
        assert_no_cycles(&g, &result);
        println!(
            "Chain: {} kernels from {} groups",
            result.num_kernels,
            g.num_groups()
        );
        for (ki, kernel) in result.kernel_groups.iter().enumerate() {
            println!("  Kernel {}: {:?}", ki, kernel);
        }
    }

    // -----------------------------------------------------------------------
    // Test: diamond graph
    // -----------------------------------------------------------------------

    #[test]
    fn test_diamond_all_assigned() {
        let g = build_diamond();
        assert!(g.validate().is_empty());
        let result = partition_nanograph(&g, 3);
        assert_all_groups_assigned_once(&g, &result);
        assert_no_cycles(&g, &result);
        println!(
            "Diamond: {} kernels from {} groups",
            result.num_kernels,
            g.num_groups()
        );
        for (ki, kernel) in result.kernel_groups.iter().enumerate() {
            println!("  Kernel {}: {:?}", ki, kernel);
        }
    }

    // -----------------------------------------------------------------------
    // Test: two independent branches
    // -----------------------------------------------------------------------

    #[test]
    fn test_two_branches_parallelism() {
        let g = build_two_branches();
        assert!(g.validate().is_empty());

        let result = partition_nanograph(&g, 3);
        assert_all_groups_assigned_once(&g, &result);
        assert_no_cycles(&g, &result);

        // With two independent branches, the partitioner should find at least
        // 2 independent sub-DAGs before the merge.
        println!(
            "Two branches: {} kernels from {} groups",
            result.num_kernels,
            g.num_groups()
        );
        for (ki, kernel) in result.kernel_groups.iter().enumerate() {
            let compute: Vec<usize> = kernel
                .iter()
                .filter(|&&gi| !matches!(g.groups()[gi].op, ScalarOp::Literal(_)))
                .copied()
                .collect();
            let data: Vec<usize> = kernel
                .iter()
                .filter(|&&gi| matches!(g.groups()[gi].op, ScalarOp::Literal(_)))
                .copied()
                .collect();
            println!(
                "  Kernel {}: compute={:?}, data={:?}",
                ki, compute, data
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test: two-layer pipeline (pinch point detection)
    // -----------------------------------------------------------------------

    #[test]
    fn test_two_layer_pipeline() {
        let g = build_two_layer_pipeline();
        assert!(g.validate().is_empty());

        let result = partition_nanograph(&g, 4);
        assert_all_groups_assigned_once(&g, &result);
        assert_no_cycles(&g, &result);

        let groups = g.groups();
        let is_data: Vec<bool> = groups
            .iter()
            .map(|gr| matches!(gr.op, ScalarOp::Literal(_)))
            .collect();

        println!(
            "Two-layer pipeline: {} kernels from {} groups",
            result.num_kernels,
            g.num_groups()
        );
        for (ki, kernel) in result.kernel_groups.iter().enumerate() {
            let compute_atoms: u64 = kernel
                .iter()
                .filter(|&&gi| !is_data[gi])
                .map(|&gi| groups[gi].count)
                .sum();
            let data_groups = kernel.iter().filter(|&&gi| is_data[gi]).count();
            println!(
                "  Kernel {}: {} groups ({} data), {} compute atoms",
                ki,
                kernel.len(),
                data_groups,
                compute_atoms
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test: matmul structure
    // -----------------------------------------------------------------------

    #[test]
    fn test_matmul_partitioning() {
        let g = build_matmul(4, 8, 16);
        assert!(g.validate().is_empty());

        let result = partition_nanograph(&g, 4);
        assert_all_groups_assigned_once(&g, &result);
        assert_no_cycles(&g, &result);

        let groups = g.groups();
        let is_data: Vec<bool> = groups
            .iter()
            .map(|gr| matches!(gr.op, ScalarOp::Literal(_)))
            .collect();

        println!(
            "MatMul(4,8,16): {} kernels from {} groups",
            result.num_kernels,
            g.num_groups()
        );
        for (ki, kernel) in result.kernel_groups.iter().enumerate() {
            let compute_atoms: u64 = kernel
                .iter()
                .filter(|&&gi| !is_data[gi])
                .map(|&gi| groups[gi].count)
                .sum();
            println!(
                "  Kernel {}: {} groups, {} compute atoms",
                ki,
                kernel.len(),
                compute_atoms
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test: matmul parallelism (rows should go to different kernels)
    // -----------------------------------------------------------------------

    #[test]
    fn test_matmul_rows_parallel() {
        // Build a matmul with M=4 rows. With target_kernels=4, we expect
        // the partitioner to put different rows in different kernels.
        let g = build_matmul(4, 8, 16);
        assert!(g.validate().is_empty());

        let result = partition_nanograph(&g, 4);
        assert_all_groups_assigned_once(&g, &result);
        assert_no_cycles(&g, &result);

        // The matmul has 4 Mul groups and 4 ReduceSum groups (one per row).
        // Each Mul+ReduceSum pair for a given row is independent.
        // Groups:
        //   0..3: A rows (data)
        //   4..11: B rows (data)
        //   12..15: Mul groups (compute, one per M row)
        //   16..19: ReduceSum groups (compute, one per M row)

        let groups = g.groups();
        let is_data: Vec<bool> = groups
            .iter()
            .map(|gr| matches!(gr.op, ScalarOp::Literal(_)))
            .collect();

        // Check that at least 2 different Mul groups are in different kernels.
        let group_kernel: Vec<usize> = {
            let mut gk = vec![0; g.num_groups()];
            for (ki, kernel) in result.kernel_groups.iter().enumerate() {
                for &gi in kernel {
                    gk[gi] = ki;
                }
            }
            gk
        };

        // Find which kernel each Mul group is in.
        let mul_group_kernels: Vec<usize> = (0..g.num_groups())
            .filter(|&gi| {
                !is_data[gi]
                    && matches!(
                        groups[gi].op,
                        ScalarOp::Binary {
                            op: ScalarBinOp::Mul,
                            ..
                        }
                    )
            })
            .map(|gi| group_kernel[gi])
            .collect();

        let unique_kernels: HashSet<usize> = mul_group_kernels.iter().copied().collect();
        println!(
            "Mul groups distributed across {} kernels: {:?}",
            unique_kernels.len(),
            mul_group_kernels
        );

        // We expect at least 2 different kernels for the 4 Mul groups.
        assert!(
            unique_kernels.len() >= 2,
            "Expected matmul rows in at least 2 kernels, got {} kernel(s): {:?}",
            unique_kernels.len(),
            mul_group_kernels
        );
    }

    // -----------------------------------------------------------------------
    // Test: balance
    // -----------------------------------------------------------------------

    #[test]
    fn test_balance_two_layer() {
        let g = build_two_layer_pipeline();
        assert!(g.validate().is_empty());

        let result = partition_nanograph(&g, 4);
        assert_all_groups_assigned_once(&g, &result);
        assert_no_cycles(&g, &result);

        // With 4 kernels, no single kernel should have more than 60% of compute.
        assert_balanced(&g, &result, 0.60);
    }

    // -----------------------------------------------------------------------
    // Test: edge case - single group
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_group() {
        let mut g = NanoGraph::new();
        g.push_group(
            100,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        let result = partition_nanograph(&g, 5);
        assert_eq!(result.num_kernels, 1);
        assert_all_groups_assigned_once(&g, &result);
    }

    // -----------------------------------------------------------------------
    // Test: edge case - empty graph
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_graph() {
        let g = NanoGraph::new();
        let result = partition_nanograph(&g, 5);
        assert_eq!(result.num_kernels, 0);
    }

    // -----------------------------------------------------------------------
    // Test: target_kernels=1 returns everything in one kernel
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_kernel() {
        let g = build_chain();
        let result = partition_nanograph(&g, 1);
        assert_eq!(result.num_kernels, 1);
        assert_all_groups_assigned_once(&g, &result);
    }

    // -----------------------------------------------------------------------
    // Test: larger matmul stress test
    // -----------------------------------------------------------------------

    #[test]
    fn test_matmul_stress() {
        // 16 output rows, k=32, n=64. 80 groups total.
        let g = build_matmul(16, 32, 64);
        assert!(g.validate().is_empty());

        let result = partition_nanograph(&g, 8);
        assert_all_groups_assigned_once(&g, &result);
        assert_no_cycles(&g, &result);

        let groups = g.groups();
        let is_data: Vec<bool> = groups
            .iter()
            .map(|gr| matches!(gr.op, ScalarOp::Literal(_)))
            .collect();

        let total_compute: u64 = groups
            .iter()
            .enumerate()
            .filter(|(gi, _)| !is_data[*gi])
            .map(|(_, gr)| gr.count)
            .sum();

        println!(
            "MatMul(16,32,64): {} kernels from {} groups, {} total compute atoms",
            result.num_kernels,
            g.num_groups(),
            total_compute
        );
        for (ki, kernel) in result.kernel_groups.iter().enumerate() {
            let compute_atoms: u64 = kernel
                .iter()
                .filter(|&&gi| !is_data[gi])
                .map(|&gi| groups[gi].count)
                .sum();
            let pct = compute_atoms as f64 / total_compute as f64 * 100.0;
            println!(
                "  Kernel {}: {} groups, {} compute atoms ({:.1}%)",
                ki,
                kernel.len(),
                compute_atoms,
                pct
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test: two matmuls in sequence (pipeline parallelism)
    // -----------------------------------------------------------------------

    #[test]
    fn test_two_matmul_pipeline() {
        let mut g = NanoGraph::new();
        let m = 4u64;
        let k = 8u64;
        let n = 16u64;

        // === Matmul 1: C1 = A @ B ===

        // A[M,K] as Literals
        let mut a_rows = Vec::new();
        for _ in 0..m {
            let id = g.push_group(
                k,
                ScalarOp::Literal(NumericScalar::F32(1.0)),
                vec![],
                vec![],
                vec![],
            );
            a_rows.push(id);
        }

        // B[K,N] as Literals
        let mut b_rows = Vec::new();
        for _ in 0..k {
            let id = g.push_group(
                n,
                ScalarOp::Literal(NumericScalar::F32(1.0)),
                vec![],
                vec![],
                vec![],
            );
            b_rows.push(id);
        }

        // M Mul groups
        let mut mul1_bases = Vec::new();
        for mi in 0..m {
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
                        base: a_rows[mi as usize],
                        stride: 1,
                        repeat: n,
                    },
                    InputRef::Affine {
                        base: b_rows[0],
                        stride: 1,
                    },
                ],
            );
            mul1_bases.push(mul);
        }

        // M ReduceSum groups -> output C1[M,N]
        let mut c1_rows = Vec::new();
        for mi in 0..m {
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
                    base: mul1_bases[mi as usize],
                    stride: 1,
                }],
            );
            c1_rows.push(red);
        }

        // === Matmul 2: C2 = C1 @ D ===

        // D[N,N] as Literals
        let mut d_rows = Vec::new();
        for _ in 0..n {
            let id = g.push_group(
                n,
                ScalarOp::Literal(NumericScalar::F32(1.0)),
                vec![],
                vec![],
                vec![],
            );
            d_rows.push(id);
        }

        // M Mul groups (C1 * D)
        let mut mul2_bases = Vec::new();
        for mi in 0..m {
            let mul = g.push_group(
                n * n,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![
                    InputRef::StridedBroadcast {
                        base: c1_rows[mi as usize],
                        stride: 1,
                        repeat: n,
                    },
                    InputRef::Affine {
                        base: d_rows[0],
                        stride: 1,
                    },
                ],
            );
            mul2_bases.push(mul);
        }

        // M ReduceSum groups
        for mi in 0..m {
            g.push_group(
                n,
                ScalarOp::ReduceSum {
                    reduce_count: n,
                    reduce_stride: n as i64,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![InputRef::Affine {
                    base: mul2_bases[mi as usize],
                    stride: 1,
                }],
            );
        }

        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let result = partition_nanograph(&g, 6);
        assert_all_groups_assigned_once(&g, &result);
        assert_no_cycles(&g, &result);

        let groups = g.groups();
        let is_data: Vec<bool> = groups
            .iter()
            .map(|gr| matches!(gr.op, ScalarOp::Literal(_)))
            .collect();

        println!(
            "Two-matmul pipeline: {} kernels from {} groups",
            result.num_kernels,
            g.num_groups()
        );
        for (ki, kernel) in result.kernel_groups.iter().enumerate() {
            let compute_atoms: u64 = kernel
                .iter()
                .filter(|&&gi| !is_data[gi])
                .map(|&gi| groups[gi].count)
                .sum();
            println!(
                "  Kernel {}: {} groups, {} compute atoms",
                ki,
                kernel.len(),
                compute_atoms
            );
        }
    }
}
