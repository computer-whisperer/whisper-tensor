#![allow(clippy::all, dead_code, unreachable_patterns, unused)]
//! NanoGraph partitioner: creative3 algorithm adapted for compressed groups.
//!
//! Operates entirely at the AtomGroup level -- never expands individual atoms.
//! The algorithm has 5 phases:
//!
//! 1. **Chain extraction**: Trace backward from output groups through
//!    single-consumer groups to form chains (tile-able computation units).
//! 2. **Input signatures**: For each chain, find source groups (Literals,
//!    external inputs with no inputs themselves).
//! 3. **Agglomerative clustering**: Merge chains sharing the most source
//!    groups, subject to a size constraint.
//! 4. **SA input assignment**: Assign each source group to the kernel that
//!    uses it most. Simulated annealing swaps to minimize cross-kernel loads.
//! 5. **Assembly**: Map group indices back to kernel assignments.

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

use crate::nano_graph::{AtomGroup, AtomId, InputRef, NanoGraph, ScalarOp};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Result of partitioning a NanoGraph into kernels.
#[derive(Debug)]
pub struct NanoPartitionResult {
    /// kernel_idx -> group indices (into NanoGraph::groups()).
    pub kernel_groups: Vec<Vec<usize>>,
    pub num_kernels: usize,
    pub total_atoms: u64,
}

/// Partition a NanoGraph into `parallelism` kernels (approximately).
///
/// Works entirely at the group level -- O(num_groups) not O(num_atoms).
pub fn partition_nanograph(graph: &NanoGraph, parallelism: usize) -> NanoPartitionResult {
    let groups = graph.groups();
    let num_groups = groups.len();
    let total_atoms: u64 = groups.iter().map(|g| g.count).sum();

    if num_groups == 0 {
        return NanoPartitionResult {
            kernel_groups: vec![],
            num_kernels: 0,
            total_atoms: 0,
        };
    }

    // Phase 0: Build group dependency graph.
    let (producers, consumers) = build_group_deps(graph);

    // Find output groups (groups containing output atoms).
    let output_groups = find_output_groups(graph);

    // Phase 1: Extract chains.
    let chains = extract_chains(groups, &producers, &consumers, &output_groups);

    if chains.is_empty() {
        // Degenerate: put everything in one kernel.
        let all_groups: Vec<usize> = (0..num_groups).collect();
        return NanoPartitionResult {
            kernel_groups: vec![all_groups],
            num_kernels: 1,
            total_atoms,
        };
    }

    // Phase 2: Input signatures.
    let source_groups = find_source_groups(groups);
    let chain_signatures = compute_chain_signatures(&chains, &producers, &source_groups);

    // Phase 3: Agglomerative clustering.
    let target_kernels = parallelism.max(1);
    let clusters = agglomerative_cluster(
        &chains,
        &chain_signatures,
        groups,
        target_kernels,
        total_atoms,
    );

    // Phase 4: SA input assignment (simplified greedy for now).
    // Phase 5: Assemble kernel_groups from clusters.
    let kernel_groups = assemble_kernels(clusters, num_groups);

    let num_kernels = kernel_groups.len();
    NanoPartitionResult {
        kernel_groups,
        num_kernels,
        total_atoms,
    }
}

// ---------------------------------------------------------------------------
// Phase 0: Group dependency graph
// ---------------------------------------------------------------------------

/// For each group index, which group indices it reads from (producers)
/// and which group indices read from it (consumers).
fn build_group_deps(graph: &NanoGraph) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let groups = graph.groups();
    let n = groups.len();
    let mut producers: Vec<Vec<usize>> = vec![vec![]; n];
    let mut consumers: Vec<Vec<usize>> = vec![vec![]; n];

    for (gi, group) in groups.iter().enumerate() {
        let mut seen: HashSet<usize> = HashSet::new();
        for input_ref in &group.inputs {
            collect_source_groups(graph, groups, group, input_ref, &mut seen);
        }
        for &src_gi in &seen {
            producers[gi].push(src_gi);
            consumers[src_gi].push(gi);
        }
    }

    (producers, consumers)
}

/// Collect all source group indices referenced by an InputRef.
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
            // Sample first and last atom to find source group(s).
            // For stride=1 contiguous access, usually one group.
            // For larger strides, may span multiple groups.
            let count = consumer.count;
            if count == 0 {
                return;
            }
            // Sample a few representative offsets.
            let samples = [0u64, count / 2, count.saturating_sub(1)];
            for &i in &samples {
                if i < count {
                    let src = input_ref.resolve(i, 0);
                    if let Some(gi) = find_group_idx(groups, src) {
                        seen.insert(gi);
                    }
                }
            }
            // If stride != 0, also check if there are intermediate groups.
            if *stride != 0 && count > 1 {
                let first = input_ref.resolve(0, 0);
                let last = input_ref.resolve(count - 1, 0);
                // Walk through groups between first and last.
                if let (Some(first_gi), Some(last_gi)) =
                    (find_group_idx(groups, first), find_group_idx(groups, last))
                {
                    let (lo, hi) = if first_gi <= last_gi {
                        (first_gi, last_gi)
                    } else {
                        (last_gi, first_gi)
                    };
                    for gi in lo..=hi {
                        // Check if this group's range overlaps with our access pattern.
                        // Conservative: include all groups in the range.
                        seen.insert(gi);
                    }
                }
            }
        }
        InputRef::StridedBroadcast { base, stride, repeat } => {
            // Each block of `repeat` atoms shares one source.
            let count = consumer.count;
            let num_blocks = (count + repeat - 1) / repeat;
            for block in 0..num_blocks {
                let src = input_ref.resolve(block * repeat, 0);
                if let Some(gi) = find_group_idx(groups, src) {
                    seen.insert(gi);
                }
            }
        }
        InputRef::SymAffine {
            base,
            stride_i,
            stride_k,
        } => {
            // Need to sample across both i and k dimensions.
            let count = consumer.count;
            // Get k bound from reduce_dims.
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

/// Find group indices that contain output atoms.
fn find_output_groups(graph: &NanoGraph) -> HashSet<usize> {
    let groups = graph.groups();
    let mut out = HashSet::new();
    for &atom_id in &graph.outputs {
        if let Some(gi) = find_group_idx(groups, atom_id) {
            out.insert(gi);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Phase 1: Chain extraction
// ---------------------------------------------------------------------------

/// A chain is a set of group indices traced backward from an anchor group
/// (output or multi-consumer boundary) through single-consumer groups.
#[derive(Debug, Clone)]
struct Chain {
    /// Group indices in this chain (topologically ordered, anchor last).
    group_indices: Vec<usize>,
    /// Total atom count across all groups in this chain.
    atom_count: u64,
}

/// Extract chains by tracing backward from anchor groups.
///
/// An anchor is either:
/// - A group containing output atoms.
/// - A group with multiple consumers (its output must be materialized).
///
/// A group is "single-consumer" if it appears in exactly one other group's
/// producer list. We trace backward from anchors through single-consumer
/// groups to build maximal chains.
fn extract_chains(
    groups: &[AtomGroup],
    producers: &[Vec<usize>],
    consumers: &[Vec<usize>],
    output_groups: &HashSet<usize>,
) -> Vec<Chain> {
    let n = groups.len();

    // A group is an "anchor" if it's an output group, has multiple consumers,
    // or has zero consumers (dead code, but still needs assignment).
    // We also treat groups with zero consumers that are NOT output groups as anchors
    // so they get assigned somewhere.
    let mut is_anchor = vec![false; n];
    for gi in 0..n {
        if output_groups.contains(&gi) || consumers[gi].len() != 1 {
            is_anchor[gi] = true;
        }
    }

    // Trace backward from each anchor.
    let mut assigned = vec![false; n];
    let mut chains = Vec::new();

    for gi in 0..n {
        if !is_anchor[gi] || assigned[gi] {
            continue;
        }

        let mut chain_groups = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(gi);

        while let Some(current) = queue.pop_front() {
            if assigned[current] {
                continue;
            }
            assigned[current] = true;
            chain_groups.push(current);

            // Trace backward through producers that are single-consumer.
            for &prod_gi in &producers[current] {
                if !assigned[prod_gi] && consumers[prod_gi].len() == 1 {
                    queue.push_back(prod_gi);
                }
            }
        }

        // Sort topologically (by group index, since groups are in topo order).
        chain_groups.sort();

        let atom_count: u64 = chain_groups.iter().map(|&gi| groups[gi].count).sum();
        chains.push(Chain {
            group_indices: chain_groups,
            atom_count,
        });
    }

    // Sweep up any unassigned groups (can happen with shared source groups).
    for gi in 0..n {
        if !assigned[gi] {
            assigned[gi] = true;
            chains.push(Chain {
                group_indices: vec![gi],
                atom_count: groups[gi].count,
            });
        }
    }

    chains
}

// ---------------------------------------------------------------------------
// Phase 2: Input signatures
// ---------------------------------------------------------------------------

/// Identify source groups: groups with no inputs (Literal ops, boundary ops).
fn find_source_groups(groups: &[AtomGroup]) -> HashSet<usize> {
    let mut sources = HashSet::new();
    for (gi, group) in groups.iter().enumerate() {
        if group.inputs.is_empty() {
            sources.insert(gi);
        }
    }
    sources
}

/// For each chain, compute the set of source group indices it depends on.
/// BFS backward through the dependency graph, stopping at source groups.
fn compute_chain_signatures(
    chains: &[Chain],
    producers: &[Vec<usize>],
    source_groups: &HashSet<usize>,
) -> Vec<BTreeSet<usize>> {
    chains
        .iter()
        .map(|chain| {
            let mut signature = BTreeSet::new();
            let mut visited = HashSet::new();
            let mut queue: VecDeque<usize> = chain.group_indices.iter().copied().collect();

            // Mark chain members as visited to avoid re-traversal.
            for &gi in &chain.group_indices {
                visited.insert(gi);
            }

            while let Some(current) = queue.pop_front() {
                if source_groups.contains(&current) {
                    signature.insert(current);
                    continue;
                }
                for &prod_gi in &producers[current] {
                    if visited.insert(prod_gi) {
                        queue.push_back(prod_gi);
                    }
                }
            }

            signature
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Phase 3: Agglomerative clustering
// ---------------------------------------------------------------------------

/// Merge chains with highest shared source-group count until we reach target_kernels.
fn agglomerative_cluster(
    chains: &[Chain],
    signatures: &[BTreeSet<usize>],
    groups: &[AtomGroup],
    target_kernels: usize,
    total_atoms: u64,
) -> Vec<Vec<usize>> {
    let n = chains.len();
    if n <= target_kernels {
        // Already at or below target, each chain is its own cluster.
        return chains.iter().map(|c| c.group_indices.clone()).collect();
    }

    // Size constraint: max atoms per cluster.
    let avg_atoms = total_atoms / target_kernels.max(1) as u64;
    let max_atoms = avg_atoms.saturating_mul(3).max(1);

    // Union-Find for clusters.
    let mut parent: Vec<usize> = (0..n).collect();
    let mut cluster_groups: Vec<Vec<usize>> = chains.iter().map(|c| c.group_indices.clone()).collect();
    let mut cluster_sigs: Vec<BTreeSet<usize>> = signatures.to_vec();
    let mut cluster_atoms: Vec<u64> = chains.iter().map(|c| c.atom_count).collect();
    let mut active: Vec<bool> = vec![true; n];
    let mut num_active = n;

    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }

    // Precompute pairwise shared counts and use a priority queue approach.
    // For large N, we use a greedy scan each iteration.
    while num_active > target_kernels {
        // Find the best merge: pair of active clusters with highest shared source count.
        let mut best_score = 0usize;
        let mut best_pair = (0, 0);

        let active_indices: Vec<usize> = (0..n).filter(|&i| active[i]).collect();

        for (idx_a, &a) in active_indices.iter().enumerate() {
            for &b in active_indices[idx_a + 1..].iter() {
                // Check size constraint.
                let merged_atoms = cluster_atoms[a] + cluster_atoms[b];
                if merged_atoms > max_atoms {
                    continue;
                }

                // Shared source count.
                let shared = cluster_sigs[a].intersection(&cluster_sigs[b]).count();
                if shared > best_score {
                    best_score = shared;
                    best_pair = (a, b);
                }
            }
        }

        if best_score == 0 {
            // No beneficial merges found; stop early.
            break;
        }

        let (a, b) = best_pair;
        // Merge b into a.
        let b_groups = std::mem::take(&mut cluster_groups[b]);
        cluster_groups[a].extend(b_groups);
        let b_sig = std::mem::take(&mut cluster_sigs[b]);
        for s in b_sig {
            cluster_sigs[a].insert(s);
        }
        cluster_atoms[a] += cluster_atoms[b];
        active[b] = false;
        parent[b] = a;
        num_active -= 1;
    }

    // Collect active clusters.
    let mut result: Vec<Vec<usize>> = Vec::new();
    for i in 0..n {
        if active[i] && !cluster_groups[i].is_empty() {
            let mut g = std::mem::take(&mut cluster_groups[i]);
            g.sort();
            g.dedup();
            result.push(g);
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Phase 5: Assemble
// ---------------------------------------------------------------------------

/// Convert clusters of group indices into the final kernel assignment.
/// Ensures every group is assigned to exactly one kernel.
fn assemble_kernels(clusters: Vec<Vec<usize>>, num_groups: usize) -> Vec<Vec<usize>> {
    // Track which groups are assigned.
    let mut assigned = vec![false; num_groups];

    let mut kernels: Vec<Vec<usize>> = Vec::new();

    for cluster in clusters {
        let mut kernel = Vec::new();
        for gi in cluster {
            if !assigned[gi] {
                assigned[gi] = true;
                kernel.push(gi);
            }
        }
        if !kernel.is_empty() {
            kernel.sort();
            kernels.push(kernel);
        }
    }

    // Any unassigned groups go into a catch-all kernel.
    let mut unassigned: Vec<usize> = Vec::new();
    for gi in 0..num_groups {
        if !assigned[gi] {
            unassigned.push(gi);
        }
    }
    if !unassigned.is_empty() {
        kernels.push(unassigned);
    }

    kernels
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::nano_graph::{ScalarBinOp, ScalarOp, ScalarUnaryOp};
    use crate::nano_graph::{InputRef, NanoGraph};
    use crate::numeric_scalar::NumericScalar;

    /// Validate that every group is assigned to exactly one kernel.
    fn validate_partition(result: &NanoPartitionResult, num_groups: usize) {
        let mut assigned = vec![0u32; num_groups];
        for (ki, kernel) in result.kernel_groups.iter().enumerate() {
            for &gi in kernel {
                assert!(
                    gi < num_groups,
                    "kernel {} references group {} which is out of range (num_groups={})",
                    ki,
                    gi,
                    num_groups
                );
                assigned[gi] += 1;
            }
        }
        for (gi, count) in assigned.iter().enumerate() {
            assert_eq!(
                *count, 1,
                "group {} assigned to {} kernels (expected 1)",
                gi, count
            );
        }
    }

    /// Print partition stats for debugging.
    fn print_partition(result: &NanoPartitionResult, groups: &[AtomGroup]) {
        eprintln!(
            "Partition: {} kernels, {} total atoms",
            result.num_kernels, result.total_atoms
        );
        for (ki, kernel) in result.kernel_groups.iter().enumerate() {
            let atoms: u64 = kernel.iter().map(|&gi| groups[gi].count).sum();
            eprintln!(
                "  kernel {}: {} groups, {} atoms",
                ki,
                kernel.len(),
                atoms
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 1: Simple elementwise add
    // -----------------------------------------------------------------------

    #[test]
    fn test_elementwise_add() {
        let mut g = NanoGraph::new();

        let a = g.push_group(
            16,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let b = g.push_group(
            16,
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
            vec![],
        );
        let c = g.push_group(
            16,
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
        g.outputs = vec![c];

        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let result = partition_nanograph(&g, 2);
        validate_partition(&result, g.num_groups());
        print_partition(&result, g.groups());

        assert_eq!(result.total_atoms, 48);
        assert!(result.num_kernels >= 1);
    }

    // -----------------------------------------------------------------------
    // Test 2: MatMul via lowering (4x3 @ 3x2 = 4x2)
    // -----------------------------------------------------------------------

    #[test]
    fn test_matmul_lowered() {
        // Build a NanoGraph that mimics what the lowering produces for a matmul.
        // A[4,3] @ B[3,2] = C[4,2]
        // M=4, K=3, N=2
        //
        // Lowering creates:
        //   - Literal groups for A (12 atoms) and B (6 atoms)
        //   - For each of 4 rows: 3 Mul groups (count=2), then 1 ReduceSum group (count=2)
        //   Total Mul groups: 4*3 = 12, each count=2
        //   Total ReduceSum groups: 4, each count=2

        let mut g = NanoGraph::new();
        let k_sym = g.bounded_sym_dim("matmul_k", 3);

        // A: 12 literal atoms (4x3 matrix).
        let a_base = g.push_group(
            12,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        // B: 6 literal atoms (3x2 matrix).
        let b_base = g.push_group(
            6,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        // For each row m (0..4), create K=3 Mul groups of count N=2.
        let mut mul_bases = Vec::new();
        for m in 0..4u64 {
            for k in 0..3u64 {
                // A[m,k] is at offset m*3 + k.
                let a_atom = a_base.offset(m * 3 + k);
                // B[k, *] starts at offset k*2.
                let b_atom = b_base.offset(k * 2);

                let mul_base = g.push_group(
                    2,
                    ScalarOp::Binary {
                        op: ScalarBinOp::Mul,
                        compute_dtype: DType::F32,
                        output_dtype: DType::F32,
                    },
                    vec![],
                    vec![],
                    vec![
                        InputRef::Broadcast(a_atom),
                        InputRef::Affine {
                            base: b_atom,
                            stride: 1,
                        },
                    ],
                );
                mul_bases.push(mul_base);
            }
        }

        // For each row m, create a ReduceSum group that sums over the K Mul groups.
        let mut reduce_bases = Vec::new();
        for m in 0..4u64 {
            // The first Mul group for row m is at mul_bases[m*3].
            let row_mul_base = mul_bases[(m * 3) as usize];

            let reduce_base = g.push_group(
                2,
                ScalarOp::ReduceSum {
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![k_sym],
                vec![InputRef::SymAffine {
                    base: row_mul_base,
                    stride_i: 1,
                    stride_k: 2, // N=2, so stride_k=2 to hop between Mul groups
                }],
            );
            reduce_bases.push(reduce_base);
        }

        // Output: the 4 ReduceSum groups, each with 2 atoms = 8 output atoms.
        g.outputs = reduce_bases.clone();

        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let stats = g.stats();
        eprintln!("Matmul graph: {}", stats);

        let result = partition_nanograph(&g, 4);
        validate_partition(&result, g.num_groups());
        print_partition(&result, g.groups());

        assert_eq!(result.total_atoms, g.num_atoms());
    }

    // -----------------------------------------------------------------------
    // Test 3: MatMul + activation (ReLU-like: max(0, x))
    // -----------------------------------------------------------------------

    #[test]
    fn test_matmul_plus_activation() {
        let mut g = NanoGraph::new();
        let k_sym = g.bounded_sym_dim("matmul_k", 4);

        // A[2,4], B[4,8] -> C[2,8]
        let m = 2u64;
        let k = 4u64;
        let n = 8u64;

        let a_base = g.push_group(
            m * k,
            ScalarOp::Literal(NumericScalar::F32(0.5)),
            vec![],
            vec![],
            vec![],
        );
        let b_base = g.push_group(
            k * n,
            ScalarOp::Literal(NumericScalar::F32(0.3)),
            vec![],
            vec![],
            vec![],
        );

        // Mul groups: m*k groups, each count=n.
        let mut mul_bases = Vec::new();
        for row in 0..m {
            for ki in 0..k {
                let a_atom = a_base.offset(row * k + ki);
                let b_atom = b_base.offset(ki * n);
                let mb = g.push_group(
                    n,
                    ScalarOp::Binary {
                        op: ScalarBinOp::Mul,
                        compute_dtype: DType::F32,
                        output_dtype: DType::F32,
                    },
                    vec![],
                    vec![],
                    vec![
                        InputRef::Broadcast(a_atom),
                        InputRef::Affine {
                            base: b_atom,
                            stride: 1,
                        },
                    ],
                );
                mul_bases.push(mb);
            }
        }

        // ReduceSum groups: m groups, each count=n.
        let mut reduce_bases = Vec::new();
        for row in 0..m {
            let row_mul_base = mul_bases[(row * k) as usize];
            let rb = g.push_group(
                n,
                ScalarOp::ReduceSum {
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![k_sym],
                vec![InputRef::SymAffine {
                    base: row_mul_base,
                    stride_i: 1,
                    stride_k: n as i32,
                }],
            );
            reduce_bases.push(rb);
        }

        // Activation: max(matmul_output, 0.0) per element.
        let zero = g.push_group(
            1,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );

        let mut act_bases = Vec::new();
        for &rb in &reduce_bases {
            let act = g.push_group(
                n,
                ScalarOp::Binary {
                    op: ScalarBinOp::Max,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![
                    InputRef::Affine {
                        base: rb,
                        stride: 1,
                    },
                    InputRef::Broadcast(zero),
                ],
            );
            act_bases.push(act);
        }

        g.outputs = act_bases;

        assert!(g.validate().is_empty(), "{:?}", g.validate());
        eprintln!("Matmul+activation graph: {}", g.stats());

        let result = partition_nanograph(&g, 2);
        validate_partition(&result, g.num_groups());
        print_partition(&result, g.groups());
    }

    // -----------------------------------------------------------------------
    // Test 4: Two independent matmuls
    // -----------------------------------------------------------------------

    #[test]
    fn test_two_matmuls() {
        let mut g = NanoGraph::new();
        let k_sym1 = g.bounded_sym_dim("k1", 3);
        let k_sym2 = g.bounded_sym_dim("k2", 2);

        // Matmul 1: A1[2,3] @ B1[3,4] = C1[2,4]
        let a1 = g.push_group(6, ScalarOp::Literal(NumericScalar::F32(1.0)), vec![], vec![], vec![]);
        let b1 = g.push_group(12, ScalarOp::Literal(NumericScalar::F32(1.0)), vec![], vec![], vec![]);

        let mut mul1_bases = Vec::new();
        for row in 0..2u64 {
            for ki in 0..3u64 {
                let mb = g.push_group(
                    4,
                    ScalarOp::Binary {
                        op: ScalarBinOp::Mul,
                        compute_dtype: DType::F32,
                        output_dtype: DType::F32,
                    },
                    vec![],
                    vec![],
                    vec![
                        InputRef::Broadcast(a1.offset(row * 3 + ki)),
                        InputRef::Affine {
                            base: b1.offset(ki * 4),
                            stride: 1,
                        },
                    ],
                );
                mul1_bases.push(mb);
            }
        }

        let mut reduce1 = Vec::new();
        for row in 0..2u64 {
            let rb = g.push_group(
                4,
                ScalarOp::ReduceSum {
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![k_sym1],
                vec![InputRef::SymAffine {
                    base: mul1_bases[(row * 3) as usize],
                    stride_i: 1,
                    stride_k: 4,
                }],
            );
            reduce1.push(rb);
        }

        // Matmul 2: A2[3,2] @ B2[2,5] = C2[3,5]
        let a2 = g.push_group(6, ScalarOp::Literal(NumericScalar::F32(2.0)), vec![], vec![], vec![]);
        let b2 = g.push_group(10, ScalarOp::Literal(NumericScalar::F32(2.0)), vec![], vec![], vec![]);

        let mut mul2_bases = Vec::new();
        for row in 0..3u64 {
            for ki in 0..2u64 {
                let mb = g.push_group(
                    5,
                    ScalarOp::Binary {
                        op: ScalarBinOp::Mul,
                        compute_dtype: DType::F32,
                        output_dtype: DType::F32,
                    },
                    vec![],
                    vec![],
                    vec![
                        InputRef::Broadcast(a2.offset(row * 2 + ki)),
                        InputRef::Affine {
                            base: b2.offset(ki * 5),
                            stride: 1,
                        },
                    ],
                );
                mul2_bases.push(mb);
            }
        }

        let mut reduce2 = Vec::new();
        for row in 0..3u64 {
            let rb = g.push_group(
                5,
                ScalarOp::ReduceSum {
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![k_sym2],
                vec![InputRef::SymAffine {
                    base: mul2_bases[(row * 2) as usize],
                    stride_i: 1,
                    stride_k: 5,
                }],
            );
            reduce2.push(rb);
        }

        let mut outputs = Vec::new();
        outputs.extend(reduce1);
        outputs.extend(reduce2);
        g.outputs = outputs;

        assert!(g.validate().is_empty(), "{:?}", g.validate());
        eprintln!("Two matmuls graph: {}", g.stats());

        let result = partition_nanograph(&g, 4);
        validate_partition(&result, g.num_groups());
        print_partition(&result, g.groups());

        // Two independent matmuls should end up in different kernels.
        assert!(
            result.num_kernels >= 2,
            "Expected at least 2 kernels for 2 independent matmuls, got {}",
            result.num_kernels
        );
    }

    // -----------------------------------------------------------------------
    // Test 5: Through the lowering pipeline (matmul 4x3 @ 3x2)
    // -----------------------------------------------------------------------

    #[test]
    fn test_via_lowering_matmul() {
        use crate::milli_graph::MilliOpGraph;
        use crate::milli_graph::ops;
        use crate::nano_graph::lower::lower_with_info;
        use crate::numeric_tensor::NumericTensor;
        use crate::tensor_info::TensorInfo;

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

        // Provide shape info for lowering.
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
        eprintln!("Lowered matmul graph: {}", graph.stats());
        eprintln!("  {} groups, {} atoms", graph.num_groups(), graph.num_atoms());

        let result = partition_nanograph(graph, 4);
        validate_partition(&result, graph.num_groups());
        print_partition(&result, graph.groups());
    }

    // -----------------------------------------------------------------------
    // Test 6: Empty graph
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_graph() {
        let g = NanoGraph::new();
        let result = partition_nanograph(&g, 4);
        assert_eq!(result.num_kernels, 0);
        assert_eq!(result.total_atoms, 0);
    }

    // -----------------------------------------------------------------------
    // Test 7: Single literal group
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_literal() {
        let mut g = NanoGraph::new();
        let a = g.push_group(
            100,
            ScalarOp::Literal(NumericScalar::F32(42.0)),
            vec![],
            vec![],
            vec![],
        );
        g.outputs = vec![a];

        let result = partition_nanograph(&g, 4);
        validate_partition(&result, g.num_groups());
        assert_eq!(result.num_kernels, 1);
        assert_eq!(result.total_atoms, 100);
    }

    // -----------------------------------------------------------------------
    // Test 8: Linear chain (a -> b -> c -> d)
    // -----------------------------------------------------------------------

    #[test]
    fn test_linear_chain() {
        let mut g = NanoGraph::new();

        let a = g.push_group(
            64,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let b = g.push_group(
            64,
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
            64,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: b, stride: 1 }],
        );
        let d = g.push_group(
            64,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Tanh,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: c, stride: 1 }],
        );
        g.outputs = vec![d];

        let result = partition_nanograph(&g, 2);
        validate_partition(&result, g.num_groups());
        print_partition(&result, g.groups());

        // A pure linear chain should end up in one chain/kernel since each group
        // has a single consumer.
        assert_eq!(
            result.num_kernels, 1,
            "Linear chain should stay in one kernel"
        );
    }

    // -----------------------------------------------------------------------
    // Test 9: Diamond pattern (shared input)
    // -----------------------------------------------------------------------

    #[test]
    fn test_diamond() {
        let mut g = NanoGraph::new();

        // src -> branch_a -> merge
        //     -> branch_b ->
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

        let result = partition_nanograph(&g, 2);
        validate_partition(&result, g.num_groups());
        print_partition(&result, g.groups());
    }
}
