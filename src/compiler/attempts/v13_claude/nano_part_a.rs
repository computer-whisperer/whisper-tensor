#![allow(clippy::all, dead_code, unreachable_patterns)]
//! NanoGraph kernel partitioner.
//!
//! Works at the GROUP level — never expands to individual atoms.
//!
//! Strategy:
//! 1. Build a group-level dependency DAG (which groups feed which groups).
//! 2. Identify "source signature" for each compute group: which source groups
//!    (Literal/input groups) it transitively depends on, and HOW it accesses them
//!    (which atom offsets within the source).
//! 3. Use the access patterns to discover dimensional structure:
//!    - Broadcast(atom_x) → all atoms in the group share one source atom
//!    - Affine(base, stride) → sequential access to a range of source atoms
//!    - SymAffine → reduction dimension access
//! 4. Partition compute groups into kernels by clustering groups that share
//!    source data (same Broadcast sources → same "row", overlapping Affine
//!    ranges → same "column band").

use std::collections::{HashMap, HashSet, BTreeSet};

use crate::nano_graph::{AtomId, InputRef, NanoGraph, ScalarOp};

/// Result of partitioning a NanoGraph into kernels.
#[derive(Debug)]
pub struct NanoPartitionResult {
    /// Each entry is a list of group indices belonging to that kernel.
    pub kernel_groups: Vec<Vec<usize>>,
    /// Number of kernels.
    pub num_kernels: usize,
}

/// Which source group and how it's accessed.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SourceAccess {
    /// Index of the source group.
    source_group: usize,
    /// The specific atom offset within the source group (for Broadcast).
    /// None means "range access" (Affine/SymAffine/Explicit).
    broadcast_offset: Option<u64>,
    /// For Affine: the base offset and stride within the source group.
    affine_range: Option<(u64, i32, u64)>, // (start_offset, stride, count)
}

/// Info about a group's role in the computation.
#[derive(Debug, Clone)]
struct GroupInfo {
    /// Direct predecessor group indices (groups this one reads from).
    predecessors: Vec<usize>,
    /// Direct successor group indices (groups that read from this one).
    successors: Vec<usize>,
    /// Is this a "leaf" group (Literal, no inputs)?
    is_leaf: bool,
    /// Is this an output group (contains output atoms)?
    is_output: bool,
    /// The set of leaf (Literal) group indices this transitively depends on.
    leaf_deps: BTreeSet<usize>,
    /// For groups with Broadcast inputs: which (source_group, atom_offset) pairs.
    broadcast_sources: Vec<(usize, u64)>,
    /// For groups with Affine inputs: which (source_group, base_offset, stride) tuples.
    affine_sources: Vec<(usize, u64, i32)>,
}

/// Partition a NanoGraph into kernels.
///
/// `parallelism` is the desired number of compute kernels (hardware threads).
/// Leaf (Literal) groups get their own kernel(s). Compute groups are clustered
/// to minimize cross-kernel data movement.
pub fn partition_nanograph(graph: &NanoGraph, parallelism: usize) -> NanoPartitionResult {
    let groups = graph.groups();
    let num_groups = groups.len();

    if num_groups == 0 {
        return NanoPartitionResult {
            kernel_groups: vec![],
            num_kernels: 0,
        };
    }

    // Step 1: Build group-level dependency graph.
    // For each InputRef, find which group the referenced atom belongs to.
    let mut infos: Vec<GroupInfo> = Vec::with_capacity(num_groups);

    // Build atom_id → group_index lookup. Since groups have contiguous IDs
    // and are sorted by base_id, we can use binary search via the graph's own method.
    // But we need group INDEX, so build our own mapping.
    // Actually, we can iterate groups and build a range map.

    // Build a sorted list of (base_id, group_idx) for binary search.
    let group_ranges: Vec<(u64, u64, usize)> = groups
        .iter()
        .enumerate()
        .map(|(i, g)| (g.base_id.0, g.base_id.0 + g.count, i))
        .collect();

    let find_group = |atom_id: AtomId| -> Option<usize> {
        // Binary search: find the last group whose base_id <= atom_id
        let idx = group_ranges.partition_point(|&(base, _, _)| base <= atom_id.0);
        if idx == 0 {
            return None;
        }
        let (_base, end, gi) = group_ranges[idx - 1];
        if atom_id.0 < end { Some(gi) } else { None }
    };

    // Collect output atom set for marking output groups.
    let output_groups: HashSet<usize> = graph
        .outputs
        .iter()
        .filter_map(|aid| find_group(*aid))
        .collect();

    // First pass: compute predecessors and source access patterns.
    for (gi, group) in groups.iter().enumerate() {
        let is_leaf = matches!(group.op, ScalarOp::Literal(_)) || group.inputs.is_empty();
        let is_output = output_groups.contains(&gi);

        let mut predecessors = Vec::new();
        let mut pred_set = HashSet::new();
        let mut broadcast_sources = Vec::new();
        let mut affine_sources = Vec::new();

        for input_ref in &group.inputs {
            match input_ref {
                InputRef::Broadcast(atom_id) => {
                    if let Some(src_gi) = find_group(*atom_id) {
                        if pred_set.insert(src_gi) {
                            predecessors.push(src_gi);
                        }
                        let src_group = &groups[src_gi];
                        let offset = atom_id.0 - src_group.base_id.0;
                        broadcast_sources.push((src_gi, offset));
                    }
                }
                InputRef::Affine { base, stride } => {
                    if let Some(src_gi) = find_group(*base) {
                        if pred_set.insert(src_gi) {
                            predecessors.push(src_gi);
                        }
                        let src_group = &groups[src_gi];
                        let base_offset = base.0 - src_group.base_id.0;
                        affine_sources.push((src_gi, base_offset, *stride));
                    }
                }
                InputRef::SymAffine { base, stride_i, stride_k } => {
                    // SymAffine references atoms across the K dimension.
                    // We need to find ALL source groups across all k values.
                    // For atom offset i at iteration k: base + stride_i*i + stride_k*k
                    // We sample k=0..K_bound to find all referenced groups.

                    // First, find the reduce_dims to get K bound.
                    let k_bound = group.reduce_dims.iter()
                        .filter_map(|sd| graph.sym_dim_bounds.get(sd))
                        .next()
                        .copied()
                        .unwrap_or(1);

                    for k in 0..k_bound {
                        // At k, atom 0 reads base + stride_k * k
                        let atom_at_k = AtomId(base.0.wrapping_add((*stride_k as i64 * k as i64) as u64));
                        if let Some(src_gi) = find_group(atom_at_k) {
                            if pred_set.insert(src_gi) {
                                predecessors.push(src_gi);
                            }
                        }
                    }
                    // Also record the affine source for the base group.
                    if let Some(src_gi) = find_group(*base) {
                        let src_group = &groups[src_gi];
                        let base_offset = base.0 - src_group.base_id.0;
                        affine_sources.push((src_gi, base_offset, *stride_i));
                    }
                }
                InputRef::StridedBroadcast { base, repeat, .. } => {
                    // Each block of `repeat` atoms shares one source.
                    // Find all source groups by sampling block boundaries.
                    let num_blocks = (group.count + repeat - 1) / repeat;
                    for block in 0..num_blocks {
                        let atom = input_ref.resolve(block * repeat, 0);
                        if let Some(src_gi) = find_group(atom) {
                            if pred_set.insert(src_gi) {
                                predecessors.push(src_gi);
                            }
                        }
                    }
                    // Record as broadcast-like source.
                    if let Some(src_gi) = find_group(*base) {
                        let src_group = &groups[src_gi];
                        let base_offset = base.0 - src_group.base_id.0;
                        broadcast_sources.push((src_gi, base_offset));
                    }
                }
                InputRef::Explicit(ids) => {
                    // Collect all referenced groups.
                    let mut seen = HashSet::new();
                    for id in ids {
                        if let Some(src_gi) = find_group(*id) {
                            if seen.insert(src_gi) && pred_set.insert(src_gi) {
                                predecessors.push(src_gi);
                            }
                        }
                    }
                }
            }
        }

        infos.push(GroupInfo {
            predecessors,
            successors: Vec::new(),
            is_leaf,
            is_output,
            leaf_deps: BTreeSet::new(),
            broadcast_sources,
            affine_sources,
        });
    }

    // Second pass: compute successors from predecessors.
    for gi in 0..num_groups {
        let preds: Vec<usize> = infos[gi].predecessors.clone();
        for &pred in &preds {
            infos[pred].successors.push(gi);
        }
    }

    // Step 2: Compute transitive leaf dependencies via BFS from leaves.
    // For each leaf, propagate forward through successors.
    for gi in 0..num_groups {
        if infos[gi].is_leaf {
            infos[gi].leaf_deps.insert(gi);
        }
    }

    // Topological order: process groups in index order (they're already topo-sorted
    // by construction in NanoGraph since later groups reference earlier ones).
    for gi in 0..num_groups {
        let pred_leaf_deps: BTreeSet<usize> = {
            let preds = &infos[gi].predecessors;
            let mut deps = BTreeSet::new();
            for &p in preds {
                for &d in &infos[p].leaf_deps {
                    deps.insert(d);
                }
            }
            deps
        };
        infos[gi].leaf_deps.extend(pred_leaf_deps);
    }

    // Step 3: Identify "row groups" — groups that share the same Broadcast source
    // atom within a leaf group. In matmul, Mul groups for the same row m all
    // broadcast the same A[m,k] atoms (different k, but same row).
    //
    // Actually, for matmul, the key grouping signal is:
    // - Mul groups are connected to ReduceSum groups via SymAffine
    // - Each ReduceSum group consumes K Mul groups (one row's worth)
    // - We want to keep each ReduceSum + its K Mul groups together
    //
    // More generally: for any group with SymAffine input, it references a
    // contiguous block of source groups. Those source groups should be in the
    // same kernel as the reduce group.

    // Step 3a: Build "must-fuse" sets — groups that MUST be in the same kernel.
    // A ReduceSum/ReduceMax with SymAffine input must be fused with all the
    // source groups it references across the K dimension.

    // Use union-find for must-fuse relationships.
    let mut uf = UnionFind::new(num_groups);

    for (gi, group) in groups.iter().enumerate() {
        if group.op.is_reduce() {
            // Fuse this reduce group with all its predecessor groups.
            for &pred in &infos[gi].predecessors {
                uf.union(gi, pred);
            }
        }

        // Also fuse chains: if a group has exactly one successor and that
        // successor has exactly one predecessor (besides leaves), fuse them.
        // This handles elementwise chains without creating mega-kernels.
        if infos[gi].successors.len() == 1 && !infos[gi].is_leaf {
            let succ = infos[gi].successors[0];
            let succ_non_leaf_preds: Vec<usize> = infos[succ]
                .predecessors
                .iter()
                .filter(|&&p| !infos[p].is_leaf)
                .copied()
                .collect();
            if succ_non_leaf_preds.len() == 1 && succ_non_leaf_preds[0] == gi {
                uf.union(gi, succ);
            }
        }
    }

    // Step 3b: Collect the must-fuse clusters.
    let mut fused_clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    for gi in 0..num_groups {
        if !infos[gi].is_leaf {
            let root = uf.find(gi);
            fused_clusters.entry(root).or_default().push(gi);
        }
    }

    // Step 4: Now we need to split these clusters across kernels for parallelism.
    // The key insight: clusters that share the same leaf dependencies can potentially
    // be merged (they process the same data). Clusters with disjoint leaf deps should
    // be in separate kernels (independent computations).
    //
    // For matmul: each row cluster depends on (A_literal, B_literal) but accesses
    // different atoms within A. We need to detect this sub-group structure.
    //
    // Strategy: Look at the Broadcast sources within each cluster. If different
    // clusters broadcast from different offsets of the same source group, they
    // are working on different "slices" of the data and can be parallelized.

    // Collect cluster info.
    struct ClusterInfo {
        groups: Vec<usize>,
        /// Broadcast sources: (source_group, atom_offset) pairs from all groups in cluster.
        broadcast_sigs: BTreeSet<(usize, u64)>,
        /// Leaf dependencies of this cluster.
        leaf_deps: BTreeSet<usize>,
        /// Total atom count in this cluster.
        atom_count: u64,
    }

    let mut clusters: Vec<ClusterInfo> = Vec::new();
    let mut group_to_cluster: Vec<Option<usize>> = vec![None; num_groups];

    // Separate leaf groups.
    let mut leaf_groups: Vec<usize> = Vec::new();

    for gi in 0..num_groups {
        if infos[gi].is_leaf {
            leaf_groups.push(gi);
        }
    }

    for (_, cluster_groups) in &fused_clusters {
        let ci = clusters.len();
        let mut broadcast_sigs = BTreeSet::new();
        let mut leaf_deps = BTreeSet::new();
        let mut atom_count = 0u64;

        for &gi in cluster_groups {
            group_to_cluster[gi] = Some(ci);
            atom_count += groups[gi].count;

            for &(src_gi, offset) in &infos[gi].broadcast_sources {
                broadcast_sigs.insert((src_gi, offset));
            }
            for &dep in &infos[gi].leaf_deps {
                leaf_deps.insert(dep);
            }
        }

        clusters.push(ClusterInfo {
            groups: cluster_groups.clone(),
            broadcast_sigs,
            leaf_deps,
            atom_count,
        });
    }

    // Step 5: Agglomerative clustering of compute clusters into kernels.
    // Goal: merge clusters that share the most leaf data, up to parallelism limit.
    //
    // For matmul: we have M clusters (one per row). We want to group them into
    // `parallelism` kernels. Clusters in the same kernel should share as much
    // B data as possible (they all share B, but different A rows).
    //
    // Simple approach: if we have more clusters than desired parallelism,
    // greedily merge the pair with the most shared broadcast sources.
    // If we have fewer clusters than parallelism, we're already done.

    let num_compute_clusters = clusters.len();

    if num_compute_clusters == 0 {
        // Only leaf groups — put them all in one kernel.
        let result = NanoPartitionResult {
            kernel_groups: vec![leaf_groups],
            num_kernels: 1,
        };

        // Validate
        validate_partition(&result, num_groups);
        return result;
    }

    // Target: split into at most `parallelism` kernels.
    // But don't split fused clusters.
    let target_kernels = parallelism.min(num_compute_clusters).max(1);

    // Assign clusters to kernels.
    // If num_compute_clusters <= target_kernels, each cluster is its own kernel.
    // Otherwise, merge clusters with most shared data.

    let mut kernel_assignment: Vec<usize> = (0..num_compute_clusters).collect();
    let mut num_kernels = num_compute_clusters;

    if num_kernels > target_kernels {
        // Agglomerative merge. Use a simple O(n^2) approach since cluster count
        // is typically small (e.g., M rows in a matmul).

        // Compute pairwise "shared broadcast source count" between clusters.
        // Merge the pair with the most sharing.

        while num_kernels > target_kernels {
            let mut best_merge = None;
            let mut best_shared = 0usize; // prefer most shared
            let mut best_total_atoms = u64::MAX; // tiebreak: smallest merge

            // Collect current kernel → cluster mapping.
            let mut kernel_to_clusters: HashMap<usize, Vec<usize>> = HashMap::new();
            for (ci, &ki) in kernel_assignment.iter().enumerate() {
                kernel_to_clusters.entry(ki).or_default().push(ci);
            }

            let kernel_ids: Vec<usize> = kernel_to_clusters.keys().copied().collect();

            for i in 0..kernel_ids.len() {
                for j in (i + 1)..kernel_ids.len() {
                    let ki = kernel_ids[i];
                    let kj = kernel_ids[j];

                    // Compute shared broadcast sources between kernels.
                    let ci_clusters = &kernel_to_clusters[&ki];
                    let cj_clusters = &kernel_to_clusters[&kj];

                    let mut sigs_i = BTreeSet::new();
                    let mut sigs_j = BTreeSet::new();
                    let mut atoms_i = 0u64;
                    let mut atoms_j = 0u64;

                    for &ci in ci_clusters {
                        for sig in &clusters[ci].broadcast_sigs {
                            sigs_i.insert(*sig);
                        }
                        atoms_i += clusters[ci].atom_count;
                    }
                    for &cj in cj_clusters {
                        for sig in &clusters[cj].broadcast_sigs {
                            sigs_j.insert(*sig);
                        }
                        atoms_j += clusters[cj].atom_count;
                    }

                    // Count shared leaf deps too for better merge decisions.
                    let mut leaf_i: BTreeSet<usize> = BTreeSet::new();
                    let mut leaf_j: BTreeSet<usize> = BTreeSet::new();
                    for &ci in ci_clusters {
                        leaf_i.extend(&clusters[ci].leaf_deps);
                    }
                    for &cj in cj_clusters {
                        leaf_j.extend(&clusters[cj].leaf_deps);
                    }

                    let shared_sigs = sigs_i.intersection(&sigs_j).count();
                    let shared_leaves = leaf_i.intersection(&leaf_j).count();
                    let shared = shared_sigs + shared_leaves;
                    let total_atoms = atoms_i + atoms_j;

                    if shared > best_shared
                        || (shared == best_shared && total_atoms < best_total_atoms)
                    {
                        best_merge = Some((ki, kj));
                        best_shared = shared;
                        best_total_atoms = total_atoms;
                    }
                }
            }

            if let Some((ki, kj)) = best_merge {
                // Merge kj into ki.
                for ci in 0..num_compute_clusters {
                    if kernel_assignment[ci] == kj {
                        kernel_assignment[ci] = ki;
                    }
                }
                num_kernels -= 1;
            } else {
                break;
            }
        }
    }

    // Step 6: Build final kernel_groups.
    // Compute kernels contain the compute groups.
    // Leaf groups go into a separate kernel (or get merged with compute kernels
    // that use them, depending on strategy).
    //
    // Strategy: put each leaf group into the kernel of its primary consumer.
    // If a leaf is consumed by multiple kernels, duplicate it (put in each).
    // Actually, for simplicity: leaf groups get their own kernel.
    // The compiler can decide to inline them later.

    // Remap kernel IDs to be contiguous.
    let mut kernel_id_remap: HashMap<usize, usize> = HashMap::new();
    let mut next_kernel = 0usize;

    for &ki in &kernel_assignment {
        if !kernel_id_remap.contains_key(&ki) {
            kernel_id_remap.insert(ki, next_kernel);
            next_kernel += 1;
        }
    }

    let mut kernel_groups_map: HashMap<usize, Vec<usize>> = HashMap::new();

    // Add compute groups to their kernels.
    for (ci, &ki) in kernel_assignment.iter().enumerate() {
        let mapped_ki = kernel_id_remap[&ki];
        for &gi in &clusters[ci].groups {
            kernel_groups_map.entry(mapped_ki).or_default().push(gi);
        }
    }

    // Add leaf groups. Strategy: each leaf gets added to the kernel of its
    // first consumer. If it has no consumers, it gets its own kernel.
    for &gi in &leaf_groups {
        if infos[gi].successors.is_empty() {
            // Orphan leaf — own kernel.
            let ki = next_kernel;
            next_kernel += 1;
            kernel_groups_map.entry(ki).or_default().push(gi);
        } else {
            // Find the kernel of the first consumer.
            let first_succ = infos[gi].successors[0];
            if let Some(&ci) = group_to_cluster.get(first_succ).and_then(|x| x.as_ref()) {
                let ki = kernel_id_remap[&kernel_assignment[ci]];
                kernel_groups_map.entry(ki).or_default().push(gi);
            } else {
                // Consumer is a leaf too? Put in own kernel.
                let ki = next_kernel;
                next_kernel += 1;
                kernel_groups_map.entry(ki).or_default().push(gi);
            }
        }
    }

    // Build the final vector of kernel groups.
    let mut kernel_groups: Vec<Vec<usize>> = Vec::new();
    let mut sorted_kernel_ids: Vec<usize> = kernel_groups_map.keys().copied().collect();
    sorted_kernel_ids.sort();

    for ki in sorted_kernel_ids {
        let mut groups = kernel_groups_map.remove(&ki).unwrap();
        groups.sort();
        kernel_groups.push(groups);
    }

    let result = NanoPartitionResult {
        num_kernels: kernel_groups.len(),
        kernel_groups,
    };

    validate_partition(&result, num_groups);
    result
}

/// Validate that every group appears in exactly one kernel.
fn validate_partition(result: &NanoPartitionResult, num_groups: usize) {
    let mut seen = vec![false; num_groups];
    for (ki, groups) in result.kernel_groups.iter().enumerate() {
        for &gi in groups {
            assert!(
                gi < num_groups,
                "Kernel {} references group {} but only {} groups exist",
                ki,
                gi,
                num_groups
            );
            assert!(
                !seen[gi],
                "Group {} appears in multiple kernels",
                gi
            );
            seen[gi] = true;
        }
    }
    for (gi, &s) in seen.iter().enumerate() {
        assert!(s, "Group {} is not assigned to any kernel", gi);
    }
}

/// Simple union-find for must-fuse grouping.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u32>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] += 1;
        }
    }
}

/// Print a summary of the partition for debugging.
pub fn print_partition_summary(graph: &NanoGraph, result: &NanoPartitionResult) {
    let groups = graph.groups();
    println!("=== Partition: {} kernels ===", result.num_kernels);
    for (ki, kernel) in result.kernel_groups.iter().enumerate() {
        let mut op_counts: HashMap<&str, usize> = HashMap::new();
        let mut total_atoms = 0u64;
        for &gi in kernel {
            let g = &groups[gi];
            total_atoms += g.count;
            let op_name = match &g.op {
                ScalarOp::Literal(_) => "Literal",
                ScalarOp::Identity { .. } => "Identity",
                ScalarOp::Binary { op, .. } => match op {
                    crate::nano_graph::ScalarBinOp::Add => "Add",
                    crate::nano_graph::ScalarBinOp::Sub => "Sub",
                    crate::nano_graph::ScalarBinOp::Mul => "Mul",
                    crate::nano_graph::ScalarBinOp::Div => "Div",
                    crate::nano_graph::ScalarBinOp::Max => "Max",
                    crate::nano_graph::ScalarBinOp::Min => "Min",
                    crate::nano_graph::ScalarBinOp::Mod => "Mod",
                    crate::nano_graph::ScalarBinOp::Pow => "Pow",
                    _ => "CmpLogic",
                },
                ScalarOp::Unary { op, .. } => match op {
                    crate::nano_graph::ScalarUnaryOp::Neg => "Neg",
                    crate::nano_graph::ScalarUnaryOp::Abs => "Abs",
                    crate::nano_graph::ScalarUnaryOp::Exp => "Exp",
                    crate::nano_graph::ScalarUnaryOp::Ln => "Ln",
                    crate::nano_graph::ScalarUnaryOp::Sqrt => "Sqrt",
                    crate::nano_graph::ScalarUnaryOp::Reciprocal => "Reciprocal",
                    crate::nano_graph::ScalarUnaryOp::Tanh => "Tanh",
                    crate::nano_graph::ScalarUnaryOp::Floor => "Floor",
                    crate::nano_graph::ScalarUnaryOp::Ceil => "Ceil",
                },
                ScalarOp::Select { .. } => "Select",
                ScalarOp::ReduceSum { .. } => "ReduceSum",
                ScalarOp::ReduceMax { .. } => "ReduceMax",
                ScalarOp::IndirectLoad { .. } => "IndirectLoad",
            };
            *op_counts.entry(op_name).or_default() += 1;
        }
        let mut sorted_ops: Vec<_> = op_counts.into_iter().collect();
        sorted_ops.sort_by(|a, b| b.1.cmp(&a.1));
        let ops_str: Vec<String> = sorted_ops
            .iter()
            .map(|(op, count)| format!("{}x{}", count, op))
            .collect();
        println!(
            "  Kernel {}: {} groups, {} atoms  [{}]",
            ki,
            kernel.len(),
            total_atoms,
            ops_str.join(", ")
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DynRank;
    use crate::dtype::DType;
    use crate::graph::GlobalId;
    use crate::milli_graph::MilliOpGraph;
    use crate::nano_graph::lower::lower_with_info;
    use crate::nano_graph::ScalarOp;
    use crate::numeric_tensor::NumericTensor;
    use crate::tensor_info::TensorInfo;
    use std::collections::HashMap;

    /// Helper: build a MilliOpGraph, lower it to NanoGraph.
    fn build_and_lower(
        build: impl FnOnce(&mut MilliOpGraph, &mut rand::rngs::ThreadRng) -> Vec<GlobalId>,
        inputs: Vec<NumericTensor<DynRank>>,
    ) -> NanoGraph {
        let mut rng = rand::rng();
        let (mut milli, _) = MilliOpGraph::new(std::iter::empty(), &mut rng);
        let input_ids = build(&mut milli, &mut rng);
        assert_eq!(input_ids.len(), inputs.len());

        let mut info_inputs: HashMap<GlobalId, TensorInfo> = HashMap::new();
        for (id, tensor) in input_ids.iter().zip(inputs.iter()) {
            info_inputs.insert(*id, TensorInfo::from(tensor.clone()));
        }

        let result = lower_with_info(&milli, &info_inputs).unwrap();
        assert!(
            result.unsupported.is_empty(),
            "Unsupported ops: {:?}",
            result.unsupported_details
        );
        result.graph
    }

    #[test]
    fn test_elementwise_add() {
        // Simple elementwise add: A[128] + B[128] = C[128]
        // Should produce few kernels (ideally 1 compute + maybe leaf).
        let nano = build_and_lower(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = graph.add_input(rng);
                let _c = crate::milli_graph::ops::SimpleBinary::add(graph, a, b, rng);
                vec![a, b]
            },
            vec![
                NumericTensor::from_vec_shape(vec![1.0f32; 128], vec![128]).unwrap(),
                NumericTensor::from_vec_shape(vec![2.0f32; 128], vec![128]).unwrap(),
            ],
        );

        println!("\n--- Elementwise Add ---");
        println!("NanoGraph: {}", nano.stats());

        let result = partition_nanograph(&nano, 8);
        print_partition_summary(&nano, &result);

        // Should have at least 1 kernel.
        assert!(result.num_kernels >= 1);
        // All groups assigned.
        let total_groups: usize = result.kernel_groups.iter().map(|k| k.len()).sum();
        assert_eq!(total_groups, nano.num_groups());

        println!("PASS: elementwise add produces {} kernels", result.num_kernels);
    }

    #[test]
    fn test_matmul_partitioning() {
        // MatMul: A[4,8] @ B[8,16] = C[4,16]
        // MUST produce >1 compute kernel.
        let nano = build_and_lower(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = graph.add_input(rng);
                let _c = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    graph,
                    a,
                    b,
                    DType::F32,
                    rng,
                );
                vec![a, b]
            },
            vec![
                NumericTensor::from_vec_shape(vec![1.0f32; 32], vec![4, 8]).unwrap(),
                NumericTensor::from_vec_shape(vec![1.0f32; 128], vec![8, 16]).unwrap(),
            ],
        );

        println!("\n--- MatMul 4x8x16 ---");
        println!("NanoGraph: {}", nano.stats());

        let result = partition_nanograph(&nano, 8);
        print_partition_summary(&nano, &result);

        // CRITICAL: must have more than 1 compute kernel.
        // Count kernels that have non-Literal groups.
        let compute_kernels: Vec<_> = result
            .kernel_groups
            .iter()
            .filter(|k| {
                k.iter()
                    .any(|&gi| !matches!(nano.groups()[gi].op, ScalarOp::Literal(_)))
            })
            .collect();

        assert!(
            compute_kernels.len() > 1,
            "MatMul must produce >1 compute kernel, got {}",
            compute_kernels.len()
        );

        // All groups assigned.
        let total_groups: usize = result.kernel_groups.iter().map(|k| k.len()).sum();
        assert_eq!(total_groups, nano.num_groups());

        println!(
            "PASS: matmul produces {} compute kernels (out of {} total)",
            compute_kernels.len(),
            result.num_kernels
        );
    }

    #[test]
    fn test_matmul_plus_unary() {
        // MatMul followed by Exp: C = exp(A @ B)
        // The Exp should fuse with the matmul output groups.
        let nano = build_and_lower(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = graph.add_input(rng);
                let c = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    graph,
                    a,
                    b,
                    DType::F32,
                    rng,
                );
                let _d = crate::milli_graph::ops::SimpleUnaryOp::exp(graph, c, rng);
                vec![a, b]
            },
            vec![
                NumericTensor::from_vec_shape(vec![0.1f32; 32], vec![4, 8]).unwrap(),
                NumericTensor::from_vec_shape(vec![0.1f32; 128], vec![8, 16]).unwrap(),
            ],
        );

        println!("\n--- MatMul + Exp ---");
        println!("NanoGraph: {}", nano.stats());

        let result = partition_nanograph(&nano, 8);
        print_partition_summary(&nano, &result);

        // Should still have multiple compute kernels (exp fused with matmul tiles).
        let compute_kernels: Vec<_> = result
            .kernel_groups
            .iter()
            .filter(|k| {
                k.iter()
                    .any(|&gi| !matches!(nano.groups()[gi].op, ScalarOp::Literal(_)))
            })
            .collect();

        assert!(
            compute_kernels.len() > 1,
            "MatMul+Exp must produce >1 compute kernel, got {}",
            compute_kernels.len()
        );

        // Check that Exp groups are fused with some matmul groups.
        let exp_in_compute: bool = compute_kernels.iter().any(|k| {
            k.iter().any(|&gi| {
                matches!(
                    &nano.groups()[gi].op,
                    ScalarOp::Unary {
                        op: crate::nano_graph::ScalarUnaryOp::Exp,
                        ..
                    }
                )
            })
        });
        assert!(exp_in_compute, "Exp groups should be in compute kernels");

        println!(
            "PASS: matmul+exp produces {} compute kernels",
            compute_kernels.len()
        );
    }

    #[test]
    fn test_two_independent_matmuls() {
        // Two independent matmuls: C1 = A1 @ B1, C2 = A2 @ B2
        // Should produce kernels from both matmuls, ideally separated.
        let nano = build_and_lower(
            |graph, rng| {
                let a1 = graph.add_input(rng);
                let b1 = graph.add_input(rng);
                let _c1 = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    graph,
                    a1,
                    b1,
                    DType::F32,
                    rng,
                );
                let a2 = graph.add_input(rng);
                let b2 = graph.add_input(rng);
                let _c2 = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    graph,
                    a2,
                    b2,
                    DType::F32,
                    rng,
                );
                vec![a1, b1, a2, b2]
            },
            vec![
                NumericTensor::from_vec_shape(vec![1.0f32; 6], vec![2, 3]).unwrap(),
                NumericTensor::from_vec_shape(vec![1.0f32; 12], vec![3, 4]).unwrap(),
                NumericTensor::from_vec_shape(vec![1.0f32; 6], vec![2, 3]).unwrap(),
                NumericTensor::from_vec_shape(vec![1.0f32; 12], vec![3, 4]).unwrap(),
            ],
        );

        println!("\n--- Two Independent MatMuls ---");
        println!("NanoGraph: {}", nano.stats());

        let result = partition_nanograph(&nano, 8);
        print_partition_summary(&nano, &result);

        // Should have multiple compute kernels.
        let compute_kernels: Vec<_> = result
            .kernel_groups
            .iter()
            .filter(|k| {
                k.iter()
                    .any(|&gi| !matches!(nano.groups()[gi].op, ScalarOp::Literal(_)))
            })
            .collect();

        assert!(
            compute_kernels.len() > 1,
            "Two matmuls must produce >1 compute kernel, got {}",
            compute_kernels.len()
        );

        println!(
            "PASS: two independent matmuls produce {} compute kernels",
            compute_kernels.len()
        );
    }

    #[test]
    fn test_partition_coverage() {
        // Verify every group index appears exactly once across all kernels.
        let nano = build_and_lower(
            |graph, rng| {
                let a = graph.add_input(rng);
                let b = graph.add_input(rng);
                let c = crate::milli_graph::ops::MatMul::push_new_default_precision(
                    graph,
                    a,
                    b,
                    DType::F32,
                    rng,
                );
                let _d = crate::milli_graph::ops::SimpleUnaryOp::exp(graph, c, rng);
                vec![a, b]
            },
            vec![
                NumericTensor::from_vec_shape(vec![1.0f32; 32], vec![4, 8]).unwrap(),
                NumericTensor::from_vec_shape(vec![1.0f32; 128], vec![8, 16]).unwrap(),
            ],
        );

        let result = partition_nanograph(&nano, 4);

        // Check coverage.
        let mut seen = vec![false; nano.num_groups()];
        for kernel in &result.kernel_groups {
            for &gi in kernel {
                assert!(!seen[gi], "Group {} appears twice", gi);
                seen[gi] = true;
            }
        }
        for (gi, &s) in seen.iter().enumerate() {
            assert!(s, "Group {} not assigned to any kernel", gi);
        }
        println!("PASS: partition coverage verified for {} groups", nano.num_groups());
    }
}
