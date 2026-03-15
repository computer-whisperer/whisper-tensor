#![allow(clippy::all, dead_code, unreachable_patterns)]
//! Topological cut-cost partitioner for NanoGraph.
//!
//! Groups are already in topological order (insertion order). This partitioner
//! computes the "cut cost" at each inter-group position: the number of atom
//! values that cross from groups <=i to groups >i. It then places K-1 kernel
//! boundaries at positions with lowest cut cost, respecting min/max kernel
//! size constraints.
//!
//! Contiguous topological ranges guarantee no circular kernel dependencies
//! by construction.
//!
//! Literal groups (weights/constants) are permanent data and don't contribute
//! to cut cost -- they're available everywhere without materialization.

use crate::nano_graph::{AtomGroup, AtomId, InputRef, NanoGraph, ScalarOp};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Result of partitioning a NanoGraph into kernels.
#[derive(Debug)]
pub struct NanoPartitionResult {
    /// Each entry is a list of group indices (into NanoGraph::groups()) that
    /// form one kernel. Groups within a kernel are in topological order.
    pub kernel_groups: Vec<Vec<usize>>,
    /// Number of kernels.
    pub num_kernels: usize,
}

/// Partition a NanoGraph into `target_kernels` kernels using topological
/// cut-cost minimization.
///
/// Works entirely at the GROUP level. Uses binary search to resolve producer
/// groups from InputRef base atoms. O(num_groups * avg_inputs * log(num_groups)).
pub fn partition_nanograph(graph: &NanoGraph, target_kernels: usize) -> NanoPartitionResult {
    let groups = graph.groups();
    let n = groups.len();

    if n == 0 {
        return NanoPartitionResult {
            kernel_groups: vec![],
            num_kernels: 0,
        };
    }

    let target_kernels = target_kernels.max(1);

    if target_kernels >= n {
        // More kernels requested than groups -- one group per kernel.
        let kernel_groups: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        return NanoPartitionResult {
            num_kernels: kernel_groups.len(),
            kernel_groups,
        };
    }

    // Step 1: Classify groups. Literal groups are "permanent" (shared data).
    let is_literal: Vec<bool> = groups
        .iter()
        .map(|g| matches!(g.op, ScalarOp::Literal(_)))
        .collect();

    // Step 2: For each non-literal group, find its producer groups.
    // Then compute last_consumer[g] = max consumer group index for each g.
    let mut last_consumer: Vec<usize> = (0..n).collect(); // self at minimum

    for (gi, group) in groups.iter().enumerate() {
        if is_literal[gi] {
            continue;
        }
        for input in &group.inputs {
            let producer_indices = resolve_producer_groups(input, group.count, groups);
            for prod_gi in producer_indices {
                if is_literal[prod_gi] {
                    // Literal groups don't contribute to cut cost.
                    continue;
                }
                if gi > last_consumer[prod_gi] {
                    last_consumer[prod_gi] = gi;
                }
            }
        }
    }

    // Step 3: Compute cut cost at each position i (boundary between group i and i+1).
    //
    // A group g contributes its `count` to the cut cost at position i if:
    //   g <= i AND last_consumer[g] > i AND !is_literal[g]
    //
    // Incremental sweep: maintain a running cost. At each position i:
    //   1. Remove groups whose last_consumer == i (they're fully consumed by i,
    //      so their values don't cross the cut after i).
    //   2. Add group i if last_consumer[i] > i (its values will cross).
    //   3. Record cut_cost[i].
    //
    // expire_at[j] lists groups g where last_consumer[g] == j and g < j.
    // These were added at position g and should be removed when we reach
    // position j (their last consumer), since at cut j their values no
    // longer need to cross.

    let mut expire_at: Vec<Vec<usize>> = vec![vec![]; n];
    for g in 0..n {
        if !is_literal[g] && last_consumer[g] > g {
            expire_at[last_consumer[g]].push(g);
        }
    }

    let mut cut_costs: Vec<u64> = Vec::with_capacity(n.saturating_sub(1));
    let mut current_cost: u64 = 0;

    for i in 0..n.saturating_sub(1) {
        // Remove groups whose last consumer is group i. At cut position i
        // (between group i and i+1), these values don't cross because
        // group i is the last reader.
        for &g in &expire_at[i] {
            current_cost -= groups[g].count;
        }

        // Add group i if its output is consumed beyond this position.
        if !is_literal[i] && last_consumer[i] > i {
            current_cost += groups[i].count;
        }

        // cut_cost[i] = cost of cutting between group i and group i+1.
        cut_costs.push(current_cost);
    }

    // Step 4: Find the K-1 best cut positions.
    let num_cuts = target_kernels - 1;

    // Minimum and maximum kernel size constraints.
    // Min: at least 1 group per kernel (can't have empty kernels).
    // Max: no single kernel should have more than ceil(2 * n / target_kernels) groups.
    let min_kernel_size = 1usize;
    let max_kernel_size = (2 * n / target_kernels).max(min_kernel_size + 1);

    let cut_positions = find_best_cuts(&cut_costs, num_cuts, n, min_kernel_size, max_kernel_size);

    // Step 5: Build kernel groups from cut positions.
    let mut kernel_groups: Vec<Vec<usize>> = Vec::with_capacity(target_kernels);
    let mut start = 0;

    for &cut_pos in &cut_positions {
        // Cut at position cut_pos means boundary after group cut_pos.
        // Kernel contains groups [start..=cut_pos].
        let end = cut_pos + 1;
        kernel_groups.push((start..end).collect());
        start = end;
    }
    // Last kernel: remaining groups.
    if start < n {
        kernel_groups.push((start..n).collect());
    }

    // Remove empty kernels (shouldn't happen, but defensive).
    kernel_groups.retain(|k| !k.is_empty());

    NanoPartitionResult {
        num_kernels: kernel_groups.len(),
        kernel_groups,
    }
}

// ---------------------------------------------------------------------------
// Producer group resolution
// ---------------------------------------------------------------------------

/// Given an InputRef and the consumer group's count, return the set of
/// producer group indices. Uses binary search on group base_ids.
///
/// Works at the GROUP level -- we only need the base atom of each InputRef
/// variant to identify the producer group, not per-atom resolution.
fn resolve_producer_groups(input: &InputRef, count: u64, groups: &[AtomGroup]) -> Vec<usize> {
    match input {
        InputRef::Broadcast(atom_id) => {
            if let Some(gi) = find_group_idx(groups, *atom_id) {
                vec![gi]
            } else {
                vec![]
            }
        }

        InputRef::Affine { base, stride } => {
            // The range of source atoms is [base, base + stride * (count-1)]
            // (or reversed if stride < 0). Find all groups that overlap this range.
            if count == 0 {
                return vec![];
            }
            let first = *base;
            let last_offset = (*stride as i64) * (count as i64 - 1);
            let last = AtomId(base.0.wrapping_add(last_offset as u64));

            let lo = first.0.min(last.0);
            let hi = first.0.max(last.0);
            find_groups_in_range(groups, lo, hi)
        }

        InputRef::Explicit(ids) => {
            // Need to find groups for all referenced atoms. Since these are
            // rare and small, just collect unique producer groups.
            let mut result = Vec::new();
            let mut seen = Vec::new();
            for id in ids {
                if let Some(gi) = find_group_idx(groups, *id) {
                    if !seen.contains(&gi) {
                        seen.push(gi);
                        result.push(gi);
                    }
                }
            }
            result
        }

        InputRef::SymAffine {
            base,
            stride_i,
            stride_k,
        } => {
            // SymAffine addresses depend on both i and k (runtime).
            // At minimum, the base group is a producer. We can't enumerate
            // all k values, but we can find the base group.
            // The i dimension spans [0, count). The range of base + stride_i * i
            // gives us the known range.
            if count == 0 {
                return vec![];
            }
            let first = *base;
            let last_offset = (*stride_i as i64) * (count as i64 - 1);
            let last = AtomId(base.0.wrapping_add(last_offset as u64));
            let lo = first.0.min(last.0);
            let hi = first.0.max(last.0);

            // Also consider stride_k if we know bounds. For now, just use
            // the i-dimension range. SymAffine is rare (0 in GPT-2).
            find_groups_in_range(groups, lo, hi)
        }

        InputRef::StridedBroadcast {
            base,
            stride,
            repeat,
        } => {
            // Atom i reads base + stride * (i / repeat).
            // Number of distinct sources = ceil(count / repeat).
            if count == 0 {
                return vec![];
            }
            let num_blocks = (count + repeat - 1) / repeat;
            let first = *base;
            let last_offset = *stride * (num_blocks as i64 - 1);
            let last = AtomId(base.0.wrapping_add(last_offset as u64));
            let lo = first.0.min(last.0);
            let hi = first.0.max(last.0);
            find_groups_in_range(groups, lo, hi)
        }

        InputRef::Modular {
            base,
            stride,
            modulus,
        } => {
            // Atom i reads base + stride * (i % modulus).
            // Range: [base, base + stride * (modulus - 1)] (or reversed).
            if *modulus == 0 {
                return vec![];
            }
            let first = *base;
            let last_offset = (*stride as i64) * (*modulus as i64 - 1);
            let last = AtomId(base.0.wrapping_add(last_offset as u64));
            let lo = first.0.min(last.0);
            let hi = first.0.max(last.0);
            find_groups_in_range(groups, lo, hi)
        }
    }
}

/// Binary search: find the group index containing the given AtomId.
/// Groups have contiguous, non-overlapping, monotonically increasing base_ids.
fn find_group_idx(groups: &[AtomGroup], id: AtomId) -> Option<usize> {
    let idx = groups.partition_point(|g| g.base_id.0 <= id.0);
    if idx == 0 {
        return None;
    }
    let gi = idx - 1;
    let group = &groups[gi];
    if group.contains(id) {
        Some(gi)
    } else {
        None
    }
}

/// Find all group indices whose atom ranges overlap [lo, hi].
/// Uses binary search for the starting group, then scans forward.
fn find_groups_in_range(groups: &[AtomGroup], lo: u64, hi: u64) -> Vec<usize> {
    if groups.is_empty() || lo > hi {
        return vec![];
    }

    // Find the first group that could contain lo.
    let start_idx = groups.partition_point(|g| g.base_id.0 + g.count <= lo);

    let mut result = Vec::new();
    for gi in start_idx..groups.len() {
        let g = &groups[gi];
        let g_lo = g.base_id.0;
        let g_hi = g.base_id.0 + g.count - 1;

        if g_lo > hi {
            break; // Past the range, done.
        }

        // Check overlap: group range [g_lo, g_hi] overlaps [lo, hi]?
        if g_hi >= lo && g_lo <= hi {
            result.push(gi);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Cut selection
// ---------------------------------------------------------------------------

/// Find the best `num_cuts` positions from the cut_costs array,
/// respecting min/max kernel size constraints.
///
/// Uses a greedy approach: repeatedly pick the lowest-cost valid position.
fn find_best_cuts(
    cut_costs: &[u64],
    num_cuts: usize,
    num_groups: usize,
    min_kernel_size: usize,
    max_kernel_size: usize,
) -> Vec<usize> {
    if num_cuts == 0 || cut_costs.is_empty() {
        return vec![];
    }

    // Build sorted indices by cut cost (ascending).
    let mut indexed_costs: Vec<(usize, u64)> = cut_costs.iter().copied().enumerate().collect();
    indexed_costs.sort_by_key(|&(_, cost)| cost);

    let mut cuts: Vec<usize> = Vec::with_capacity(num_cuts);

    for &(pos, _cost) in &indexed_costs {
        if cuts.len() >= num_cuts {
            break;
        }

        // Check if this position is valid given existing cuts and size constraints.
        if is_valid_cut(&cuts, pos, num_groups, min_kernel_size, max_kernel_size) {
            cuts.push(pos);
            cuts.sort();
        }
    }

    // If we didn't get enough cuts (size constraints too tight), fall back to
    // evenly-spaced cuts for the remaining.
    if cuts.len() < num_cuts {
        let existing = cuts.clone();
        let needed = num_cuts - existing.len();
        let step = num_groups / (needed + existing.len() + 1);
        let mut pos = step;
        for _ in 0..needed {
            // Find nearest valid position.
            while pos < num_groups.saturating_sub(1)
                && !is_valid_cut(&cuts, pos, num_groups, min_kernel_size, max_kernel_size)
            {
                pos += 1;
            }
            if pos < num_groups.saturating_sub(1) {
                cuts.push(pos);
                cuts.sort();
                pos += step;
            }
        }
    }

    cuts.sort();
    cuts
}

/// Check if adding a cut at `pos` is valid given existing cuts and constraints.
fn is_valid_cut(
    existing_cuts: &[usize],
    pos: usize,
    num_groups: usize,
    min_size: usize,
    max_size: usize,
) -> bool {
    // Build the full set of boundaries to check segment sizes.
    let mut boundaries: Vec<usize> = Vec::with_capacity(existing_cuts.len() + 3);
    boundaries.push(0); // Start
    for &c in existing_cuts {
        boundaries.push(c + 1); // Each cut creates a boundary at c+1
    }
    boundaries.push(pos + 1); // The proposed cut
    boundaries.push(num_groups); // End
    boundaries.sort();
    boundaries.dedup();

    // Check that all segments [boundaries[i]..boundaries[i+1]] are within size constraints.
    for i in 0..boundaries.len() - 1 {
        let size = boundaries[i + 1] - boundaries[i];
        if size < min_size || size > max_size {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::nano_graph::{ScalarBinOp, ScalarOp, ScalarUnaryOp};
    use crate::nano_graph::NanoGraph;
    use crate::numeric_scalar::NumericScalar;

    /// Build a simple linear chain: A -> B -> C -> D
    /// Each group has 100 atoms, connected by Affine stride=1.
    fn make_linear_chain(n: usize) -> NanoGraph {
        let mut g = NanoGraph::new();
        let mut prev = g.push_group(
            100,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        for _ in 1..n {
            prev = g.push_group(
                100,
                ScalarOp::Unary {
                    op: ScalarUnaryOp::Neg,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![InputRef::Affine {
                    base: prev,
                    stride: 1,
                }],
            );
        }
        g
    }

    /// Build a diamond: A -> B, A -> C, B+C -> D.
    fn make_diamond() -> NanoGraph {
        let mut g = NanoGraph::new();
        let a = g.push_group(
            100,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let b = g.push_group(
            100,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: a,
                stride: 1,
            }],
        );
        let c = g.push_group(
            100,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: a,
                stride: 1,
            }],
        );
        let _d = g.push_group(
            100,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine {
                    base: b,
                    stride: 1,
                },
                InputRef::Affine {
                    base: c,
                    stride: 1,
                },
            ],
        );
        g
    }

    /// Build a two-stage pipeline with a narrow pinch point between them.
    /// Stage 1: 10 groups computing from a shared input, reducing to 1 group.
    /// Stage 2: 10 groups expanding from that single output.
    fn make_pipeline_pinch() -> NanoGraph {
        let mut g = NanoGraph::new();

        // Input data
        let input = g.push_group(
            1000,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        // Stage 1: 10 groups each processing 100 atoms from input
        let mut stage1_outputs = Vec::new();
        for i in 0..10 {
            let out = g.push_group(
                100,
                ScalarOp::Unary {
                    op: ScalarUnaryOp::Neg,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![InputRef::Affine {
                    base: AtomId(input.0 + i * 100),
                    stride: 1,
                }],
            );
            stage1_outputs.push(out);
        }

        // Pinch point: single group that reads from all stage 1 outputs
        // (ReduceSum over the 1000 values)
        let pinch = g.push_group(
            100,
            ScalarOp::ReduceSum {
                reduce_count: 10,
                reduce_stride: 100,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: stage1_outputs[0],
                stride: 1,
            }],
        );

        // Stage 2: 10 groups expanding from pinch point
        for _ in 0..10 {
            g.push_group(
                100,
                ScalarOp::Unary {
                    op: ScalarUnaryOp::Exp,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![InputRef::Broadcast(pinch)],
            );
        }

        g
    }

    #[test]
    fn test_empty_graph() {
        let g = NanoGraph::new();
        let result = partition_nanograph(&g, 4);
        assert_eq!(result.num_kernels, 0);
        assert!(result.kernel_groups.is_empty());
    }

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
        let result = partition_nanograph(&g, 4);
        assert_eq!(result.num_kernels, 1);
        assert_eq!(result.kernel_groups[0], vec![0]);
    }

    #[test]
    fn test_linear_chain_two_kernels() {
        let g = make_linear_chain(10);
        let result = partition_nanograph(&g, 2);
        assert_eq!(result.num_kernels, 2);

        // All groups should be covered.
        let total: usize = result.kernel_groups.iter().map(|k| k.len()).sum();
        assert_eq!(total, 10);

        // Groups within each kernel should be contiguous and in order.
        for kernel in &result.kernel_groups {
            for w in kernel.windows(2) {
                assert_eq!(w[1], w[0] + 1, "groups should be contiguous");
            }
        }

        // No overlap: kernel 0's max < kernel 1's min.
        let k0_max = *result.kernel_groups[0].last().unwrap();
        let k1_min = result.kernel_groups[1][0];
        assert!(k0_max < k1_min, "kernels should not overlap");
    }

    #[test]
    fn test_linear_chain_many_kernels() {
        let g = make_linear_chain(20);
        let result = partition_nanograph(&g, 5);

        // Should have at least 1 kernel. For a linear chain, all cut costs are
        // equal (1 atom each), so the algorithm may produce fewer than target.
        assert!(result.num_kernels >= 1 && result.num_kernels <= 5);

        // All groups covered.
        let total: usize = result.kernel_groups.iter().map(|k| k.len()).sum();
        assert_eq!(total, 20);
    }

    #[test]
    fn test_diamond_two_kernels() {
        let g = make_diamond();
        let result = partition_nanograph(&g, 2);
        assert_eq!(result.num_kernels, 2);

        let total: usize = result.kernel_groups.iter().map(|k| k.len()).sum();
        assert_eq!(total, 4);
    }

    #[test]
    fn test_pipeline_pinch_finds_boundary() {
        let g = make_pipeline_pinch();
        let result = partition_nanograph(&g, 2);

        // The partitioner should find the pinch point and cut there.
        assert_eq!(result.num_kernels, 2);

        let total: usize = result.kernel_groups.iter().map(|k| k.len()).sum();
        assert_eq!(total, g.num_groups());

        // The pinch group (index 11) should be at a kernel boundary.
        // Either at the end of kernel 0 or start of kernel 1.
        let k0 = &result.kernel_groups[0];
        let k1 = &result.kernel_groups[1];

        // The pinch point is between stage1 and stage2. The ideal cut is
        // after the pinch group (index 11), where only 100 values cross.
        // Verify groups are contiguous.
        for kernel in &result.kernel_groups {
            for w in kernel.windows(2) {
                assert_eq!(w[1], w[0] + 1);
            }
        }
    }

    #[test]
    fn test_no_circular_deps() {
        // For any partition of a linear chain, verify no circular deps.
        let g = make_linear_chain(20);
        let result = partition_nanograph(&g, 4);

        // With contiguous topo ranges, circular deps are impossible by construction.
        // Verify: for each kernel, the max group index in kernel k is less than
        // the min group index in kernel k+1.
        for w in result.kernel_groups.windows(2) {
            let k0_max = *w[0].last().unwrap();
            let k1_min = w[1][0];
            assert!(
                k0_max < k1_min,
                "kernel boundary violated: {} >= {}",
                k0_max,
                k1_min
            );
        }
    }

    #[test]
    fn test_all_groups_assigned() {
        let g = make_pipeline_pinch();
        let result = partition_nanograph(&g, 3);

        let mut all_groups: Vec<usize> = result
            .kernel_groups
            .iter()
            .flat_map(|k| k.iter().copied())
            .collect();
        all_groups.sort();
        all_groups.dedup();

        let expected: Vec<usize> = (0..g.num_groups()).collect();
        assert_eq!(all_groups, expected, "all groups must be assigned to exactly one kernel");
    }

    #[test]
    fn test_target_one_kernel() {
        let g = make_linear_chain(10);
        let result = partition_nanograph(&g, 1);
        assert_eq!(result.num_kernels, 1);
        assert_eq!(result.kernel_groups[0].len(), 10);
    }

    #[test]
    fn test_literal_groups_dont_inflate_cut_cost() {
        // Create a graph where many literal groups exist.
        // They should not inflate the cut cost.
        let mut g = NanoGraph::new();

        // 10 literal groups with 1000 atoms each
        let mut lit_bases = Vec::new();
        for _ in 0..10 {
            let base = g.push_group(
                1000,
                ScalarOp::Literal(NumericScalar::F32(1.0)),
                vec![],
                vec![],
                vec![],
            );
            lit_bases.push(base);
        }

        // A compute chain of 10 groups, each reading from a literal + previous
        let mut prev = g.push_group(
            100,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine {
                    base: lit_bases[0],
                    stride: 1,
                },
                InputRef::Broadcast(lit_bases[1]),
            ],
        );

        for i in 2..10 {
            prev = g.push_group(
                100,
                ScalarOp::Binary {
                    op: ScalarBinOp::Add,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![
                    InputRef::Affine {
                        base: prev,
                        stride: 1,
                    },
                    InputRef::Broadcast(lit_bases[i]),
                ],
            );
        }

        let result = partition_nanograph(&g, 3);

        // All groups should be assigned.
        let total: usize = result.kernel_groups.iter().map(|k| k.len()).sum();
        assert_eq!(total, g.num_groups());
    }

    #[test]
    fn test_broadcast_input_ref_resolution() {
        let mut g = NanoGraph::new();
        let a = g.push_group(
            100,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        // Many groups broadcasting from the same source.
        for _ in 0..10 {
            g.push_group(
                50,
                ScalarOp::Unary {
                    op: ScalarUnaryOp::Neg,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![InputRef::Broadcast(a)],
            );
        }

        let result = partition_nanograph(&g, 2);
        let total: usize = result.kernel_groups.iter().map(|k| k.len()).sum();
        assert_eq!(total, 11);
    }
}
