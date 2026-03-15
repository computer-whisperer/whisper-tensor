#![allow(clippy::all, dead_code, unreachable_patterns)]
//! Live-set partitioner for NanoGraph.
//!
//! Partitions groups into kernels by sweeping through topological order,
//! tracking the "live set" of intermediate atoms. Groups whose outputs have
//! been fully consumed leave the live set. Positions where the live set is
//! smallest are natural kernel boundaries (pinch points), corresponding to
//! phase transitions in the computation (e.g., between transformer layers).
//!
//! Key properties:
//! - Kernels are contiguous topo ranges -> no circular dependencies by construction
//! - Literal groups (weights/constants) are excluded from liveness tracking
//! - Works at group granularity, not individual atoms

use crate::nano_graph::{AtomGroup, AtomId, InputRef, NanoGraph, ScalarOp};

/// Result of partitioning a NanoGraph into kernels.
#[derive(Debug)]
pub struct NanoPartitionResult {
    /// Each entry is a list of group indices forming one kernel.
    /// Kernels are contiguous ranges in topological order.
    pub kernel_groups: Vec<Vec<usize>>,
    /// Number of kernels.
    pub num_kernels: usize,
}

/// Partition a NanoGraph into kernels using live-set analysis with pinch-point detection.
///
/// `target_kernels` is the desired number of output kernels. The algorithm finds
/// `target_kernels - 1` cut points with the smallest live set sizes, subject to
/// minimum kernel size constraints.
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

    // Step 1: Build producer lookup (sorted by base_id for binary search).
    let atom_to_group = build_atom_to_group_map(groups);

    // Step 2: Classify groups as data (Literal, no inputs) vs compute.
    let is_data: Vec<bool> = groups
        .iter()
        .map(|g| matches!(g.op, ScalarOp::Literal(_)) && g.inputs.is_empty())
        .collect();

    // Step 3: For each group, find its producer groups (which groups produce its inputs).
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

    // Step 4: Compute last_consumer[g] = max group index that reads from group g.
    // Only track non-data producers. Data groups are "always live" (weights).
    let mut last_consumer: Vec<usize> = (0..n).collect(); // default: self (consumed immediately)
    for (consumer_gi, prod_list) in producers.iter().enumerate() {
        for &prod_gi in prod_list {
            if !is_data[prod_gi] && consumer_gi > last_consumer[prod_gi] {
                last_consumer[prod_gi] = consumer_gi;
            }
        }
    }

    // Step 5: Sweep through groups in topo order (= insertion order), tracking live set.
    // A non-data group enters the live set when produced. It leaves when we pass its
    // last_consumer. Track live_set_size measured in ATOMS.
    let mut live_sizes: Vec<u64> = Vec::with_capacity(n);
    let mut live_set_size: u64 = 0;

    // Track which groups are currently live and their atom counts, indexed by group.
    // We use a counter of groups whose last_consumer boundary we haven't passed yet.
    // When processing group gi:
    //   1. Add gi's atoms to live set (if non-data)
    //   2. Record live_set_size
    //   3. Expire any groups whose last_consumer == gi

    // To efficiently expire, build a list of groups expiring at each position.
    let mut expire_at: Vec<Vec<usize>> = vec![vec![]; n];
    for gi in 0..n {
        if !is_data[gi] {
            expire_at[last_consumer[gi]].push(gi);
        }
    }

    for gi in 0..n {
        // Add this group to the live set (if non-data).
        if !is_data[gi] {
            live_set_size += groups[gi].count;
        }

        // Record live set size at this boundary (after producing gi, before expiring).
        // Actually, we want the live set size *between* groups, i.e., after processing
        // gi and expiring anything whose last consumer is gi. This represents the data
        // that must cross the boundary if we cut here.

        // Expire groups whose last consumer is gi.
        for &expired_gi in &expire_at[gi] {
            live_set_size -= groups[expired_gi].count;
        }

        live_sizes.push(live_set_size);
    }

    // Step 6: Find pinch points. We want to cut *after* group gi, meaning kernel
    // boundary is between gi and gi+1. The live_sizes[gi] tells us how many atoms
    // are live after processing gi (i.e., must cross the boundary).
    //
    // We select the K-1 positions with smallest live_sizes, subject to:
    // - Can't cut at position n-1 (that's the end)
    // - Each resulting kernel must contain at least one compute group
    // - Minimum kernel size of max(1, n / (target_kernels * 4)) groups

    let min_kernel_size = std::cmp::max(1, n / (target_kernels * 4));
    let num_cuts = target_kernels - 1;

    // Precompute prefix sum of compute atom counts for balance scoring.
    let mut compute_atom_prefix: Vec<u64> = vec![0; n + 1];
    for gi in 0..n {
        compute_atom_prefix[gi + 1] = compute_atom_prefix[gi]
            + if is_data[gi] { 0 } else { groups[gi].count };
    }
    let total_compute_atoms = compute_atom_prefix[n];

    // A candidate cut at position `pos` means we cut after group `pos`.
    // Only consider positions where there are compute atoms on BOTH sides.
    // Score: live_set_size at the cut. Lower is better.
    let mut candidates: Vec<(u64, usize)> = live_sizes
        .iter()
        .enumerate()
        .filter(|&(gi, _)| {
            if gi >= n - 1 {
                return false; // can't cut after the last group
            }
            // Must have compute atoms on both sides.
            let left_compute = compute_atom_prefix[gi + 1];
            let right_compute = total_compute_atoms - left_compute;
            left_compute > 0 && right_compute > 0
        })
        .map(|(gi, &size)| (size, gi))
        .collect();

    // Sort by live set size (ascending), then by position for stability.
    candidates.sort();

    // Greedily select cuts, enforcing minimum distance between cuts.
    let mut cuts: Vec<usize> = Vec::new();
    for &(_live_size, pos) in &candidates {
        if cuts.len() >= num_cuts {
            break;
        }

        // Check minimum distance from existing cuts.
        let too_close = cuts.iter().any(|&c| {
            let dist = if pos > c { pos - c } else { c - pos };
            dist < min_kernel_size
        });

        if too_close {
            continue;
        }

        // Verify each resulting segment has compute atoms.
        // Build tentative sorted cuts including this one.
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
            // Last segment
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

    // Sort cuts by position.
    cuts.sort();

    // Step 7: Assign groups to kernels. Groups between cut[i-1]+1 and cut[i] form kernel i.
    let mut kernel_groups: Vec<Vec<usize>> = Vec::new();
    let mut start = 0;
    for &cut in &cuts {
        let end = cut + 1; // cut is inclusive, kernel is [start, end)
        kernel_groups.push((start..end).collect());
        start = end;
    }
    // Last kernel: from last cut+1 to end.
    kernel_groups.push((start..n).collect());

    let num_kernels = kernel_groups.len();
    NanoPartitionResult {
        kernel_groups,
        num_kernels,
    }
}

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
    let mut seen = std::collections::HashSet::new();

    let add_producer = |gi: usize, seen: &mut std::collections::HashSet<usize>, producers: &mut Vec<usize>| {
        if seen.insert(gi) {
            producers.push(gi);
        }
    };

    // For ReduceSum/ReduceMax, we need to account for the reduction range.
    // The input ref resolves to a base atom, and then the op reads at offsets
    // k * reduce_stride for k in 0..reduce_count from that resolved atom.
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
                    // Broadcast + reduce: all atoms read same base, but reduction
                    // reads at offsets k*reduce_stride from that base.
                    for k in 0..reduce_count {
                        let atom = AtomId(id.0.wrapping_add((k as i64 * reduce_stride) as u64));
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
                    // Affine + reduce: atom i reads base + stride*i, then reduction
                    // reads at offsets k*reduce_stride. Sample at i=0 and i=count-1
                    // across all k values.
                    for sample_i in [0, group.count.saturating_sub(1)] {
                        let resolved = input.resolve(sample_i, 0);
                        for k in 0..reduce_count {
                            let atom = AtomId(resolved.0.wrapping_add((k as i64 * reduce_stride) as u64));
                            if let Some(gi) = find_group_for_atom(atom, atom_to_group) {
                                add_producer(gi, &mut seen, &mut producers);
                            }
                        }
                    }
                } else {
                    // Non-reduce affine: sample start and end.
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
                // Each block of `repeat` atoms shares one source.
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
                // Sample base and base + stride*(modulus-1).
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
                // Determine K range from reduce_dims and sym_dim_bounds.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::nano_graph::{InputRef, NanoGraph, ScalarBinOp, ScalarOp};
    use crate::numeric_scalar::NumericScalar;

    /// Test 1: Every group in exactly one kernel.
    #[test]
    fn test_every_group_in_exactly_one_kernel() {
        let g = build_two_layer_pipeline();
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let result = partition_nanograph(&g, 4);

        // Collect all assigned group indices.
        let mut assigned: Vec<usize> = result
            .kernel_groups
            .iter()
            .flat_map(|k| k.iter().copied())
            .collect();
        assigned.sort();

        // Every group should appear exactly once.
        let expected: Vec<usize> = (0..g.num_groups()).collect();
        assert_eq!(
            assigned, expected,
            "Not every group assigned exactly once. Assigned: {:?}, Expected: {:?}",
            assigned, expected
        );

        // No duplicates.
        let unique: std::collections::HashSet<usize> = assigned.iter().copied().collect();
        assert_eq!(
            unique.len(),
            g.num_groups(),
            "Some groups assigned to multiple kernels"
        );

        println!("PASS: Every group in exactly one kernel ({} groups, {} kernels)",
            g.num_groups(), result.num_kernels);
    }

    /// Test 2: Kernels are contiguous topo ranges.
    #[test]
    fn test_kernels_are_contiguous_topo_ranges() {
        let g = build_two_layer_pipeline();
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let result = partition_nanograph(&g, 4);

        for (ki, kernel) in result.kernel_groups.iter().enumerate() {
            assert!(!kernel.is_empty(), "Kernel {} is empty", ki);
            let min_gi = *kernel.iter().min().unwrap();
            let max_gi = *kernel.iter().max().unwrap();
            let expected_len = max_gi - min_gi + 1;
            assert_eq!(
                kernel.len(),
                expected_len,
                "Kernel {} is not a contiguous range: min={}, max={}, len={}, expected_len={}",
                ki, min_gi, max_gi, kernel.len(), expected_len
            );

            // Also verify sorted order.
            let mut sorted = kernel.clone();
            sorted.sort();
            assert_eq!(
                kernel, &sorted,
                "Kernel {} groups not in sorted order",
                ki
            );
        }

        // Verify kernels are in order (kernel i's max < kernel i+1's min).
        for i in 0..result.kernel_groups.len() - 1 {
            let max_i = *result.kernel_groups[i].iter().max().unwrap();
            let min_next = *result.kernel_groups[i + 1].iter().min().unwrap();
            assert!(
                max_i < min_next,
                "Kernel {} (max={}) overlaps with kernel {} (min={})",
                i, max_i, i + 1, min_next
            );
        }

        println!("PASS: All {} kernels are contiguous topo ranges", result.num_kernels);
        for (ki, kernel) in result.kernel_groups.iter().enumerate() {
            println!("  Kernel {}: groups [{}..={}] ({} groups)",
                ki, kernel.first().unwrap(), kernel.last().unwrap(), kernel.len());
        }
    }

    /// Test 3: Print live set sizes at cut points.
    #[test]
    fn test_print_live_set_sizes() {
        let g = build_two_layer_pipeline();
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let result = partition_nanograph(&g, 4);

        // Recompute live set sizes for printing.
        let groups = g.groups();
        let n = groups.len();
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

        let mut last_consumer: Vec<usize> = (0..n).collect();
        for (consumer_gi, prod_list) in producers.iter().enumerate() {
            for &prod_gi in prod_list {
                if !is_data[prod_gi] && consumer_gi > last_consumer[prod_gi] {
                    last_consumer[prod_gi] = consumer_gi;
                }
            }
        }

        let mut expire_at: Vec<Vec<usize>> = vec![vec![]; n];
        for gi in 0..n {
            if !is_data[gi] {
                expire_at[last_consumer[gi]].push(gi);
            }
        }

        let mut live_set_size: u64 = 0;
        let mut live_sizes: Vec<u64> = Vec::with_capacity(n);
        for gi in 0..n {
            if !is_data[gi] {
                live_set_size += groups[gi].count;
            }
            for &expired_gi in &expire_at[gi] {
                live_set_size -= groups[expired_gi].count;
            }
            live_sizes.push(live_set_size);
        }

        println!("=== Live Set Analysis for Two-Layer Pipeline ===");
        println!("Total groups: {}, Total kernels: {}", n, result.num_kernels);
        println!("\nLive set sizes at kernel boundaries:");

        for (ki, kernel) in result.kernel_groups.iter().enumerate() {
            let last_in_kernel = *kernel.last().unwrap();
            let atom_count: u64 = kernel.iter().map(|&gi| groups[gi].count).sum();
            let compute_count: usize = kernel.iter().filter(|&&gi| !is_data[gi]).count();
            let data_count = kernel.len() - compute_count;

            if last_in_kernel < n - 1 {
                println!(
                    "  Kernel {} (groups [{}..={}]): {} groups ({} compute, {} data), {} atoms | live set at boundary: {} atoms",
                    ki,
                    kernel.first().unwrap(),
                    last_in_kernel,
                    kernel.len(),
                    compute_count,
                    data_count,
                    atom_count,
                    live_sizes[last_in_kernel]
                );
            } else {
                println!(
                    "  Kernel {} (groups [{}..={}]): {} groups ({} compute, {} data), {} atoms | FINAL",
                    ki,
                    kernel.first().unwrap(),
                    last_in_kernel,
                    kernel.len(),
                    compute_count,
                    data_count,
                    atom_count,
                );
            }
        }

        // Print overall live set profile (sample every N groups).
        let sample_interval = std::cmp::max(1, n / 40);
        println!("\nLive set profile (sampled every {} groups):", sample_interval);
        for gi in (0..n).step_by(sample_interval) {
            let bar_len = (live_sizes[gi] / std::cmp::max(1, *live_sizes.iter().max().unwrap_or(&1) / 60)) as usize;
            let bar: String = "#".repeat(bar_len);
            println!("  [{:5}] {:>8} atoms  {}", gi, live_sizes[gi], bar);
        }
    }

    /// Test 4: Simple elementwise chain - should produce valid partitioning.
    #[test]
    fn test_simple_elementwise_chain() {
        let mut g = NanoGraph::new();

        // Chain: a -> b -> c -> d -> e (elementwise ops)
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
        let d = g.push_group(
            1024,
            ScalarOp::Unary {
                op: crate::nano_graph::ScalarUnaryOp::Sqrt,
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
                op: crate::nano_graph::ScalarUnaryOp::Tanh,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: d, stride: 1 }],
        );

        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let result = partition_nanograph(&g, 3);

        // Every group assigned exactly once.
        let total: usize = result.kernel_groups.iter().map(|k| k.len()).sum();
        assert_eq!(total, g.num_groups());

        // Contiguous ranges.
        for (ki, kernel) in result.kernel_groups.iter().enumerate() {
            let min = *kernel.iter().min().unwrap();
            let max = *kernel.iter().max().unwrap();
            assert_eq!(kernel.len(), max - min + 1,
                "Kernel {} not contiguous", ki);
        }

        println!("Simple chain: {} kernels from {} groups", result.num_kernels, g.num_groups());
    }

    /// Test 5: Single matmul partitioning.
    #[test]
    fn test_matmul_partitioning() {
        let g = build_matmul_nanograph(4, 8, 16);
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let result = partition_nanograph(&g, 4);

        let total: usize = result.kernel_groups.iter().map(|k| k.len()).sum();
        assert_eq!(total, g.num_groups());

        println!("MatMul(4,8,16): {} kernels from {} groups", result.num_kernels, g.num_groups());
        for (ki, kernel) in result.kernel_groups.iter().enumerate() {
            let atom_count: u64 = kernel.iter().map(|&gi| g.groups()[gi].count).sum();
            println!("  Kernel {}: {} groups, {} atoms, range [{}..={}]",
                ki, kernel.len(), atom_count,
                kernel.first().unwrap(), kernel.last().unwrap());
        }
    }

    // -----------------------------------------------------------------------
    // Test graph builders
    // -----------------------------------------------------------------------

    /// Build a two-layer pipeline mimicking a simplified transformer:
    /// Layer 1: weights -> matmul -> activation
    /// Layer 2: weights -> matmul -> activation
    /// The "pinch point" between layers should be visible as a live set minimum.
    fn build_two_layer_pipeline() -> NanoGraph {
        let mut g = NanoGraph::new();
        let hidden = 64u64; // hidden dimension
        let k_dim = 32u64;  // contraction dimension

        // === Layer 1 ===

        // Input data (external, treated as Literal for simplicity)
        let input = g.push_group(
            hidden,
            ScalarOp::Literal(NumericScalar::F32(0.5)),
            vec![],
            vec![],
            vec![],
        );

        // Layer 1 weights: k_dim groups of hidden Literals (B matrix rows)
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

        // Layer 1 Mul groups: k_dim groups, each hidden atoms
        // Each Mul group: Broadcast(input[k]) * Affine(weight_row[k])
        // This is a simplified matmul where we have 1 output row.
        let mut l1_mul_bases = Vec::new();
        for ki in 0..k_dim {
            let input_atom = AtomId(input.0 + ki); // input[ki]
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
                    InputRef::Affine { base: weight_row, stride: 1 },
                ],
            );
            l1_mul_bases.push(mul);
        }

        // Layer 1 ReduceSum: 1 group of hidden atoms, reducing over k_dim
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

        // Layer 1 activation (Tanh)
        let l1_act = g.push_group(
            hidden,
            ScalarOp::Unary {
                op: crate::nano_graph::ScalarUnaryOp::Tanh,
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

        // === PINCH POINT: only l1_act (hidden atoms) is live here ===

        // === Layer 2 ===

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
            let act_atom = AtomId(l1_act.0 + ki); // l1_act[ki]
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
                    InputRef::Affine { base: weight_row, stride: 1 },
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
                op: crate::nano_graph::ScalarUnaryOp::Tanh,
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
    fn build_matmul_nanograph(m: u64, k: u64, n: u64) -> NanoGraph {
        let mut g = NanoGraph::new();

        // A[M,K] as M*K singleton Literals.
        let mut a_atoms = Vec::new();
        for _ in 0..m {
            for _ in 0..k {
                let id = g.push_group(
                    1,
                    ScalarOp::Literal(NumericScalar::F32(1.0)),
                    vec![],
                    vec![],
                    vec![],
                );
                a_atoms.push(id);
            }
        }

        // B[K,N] as K groups of N Literals each.
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

        // M*K Mul groups, each with N atoms.
        let mut mul_bases = Vec::new();
        for mi in 0..m {
            for ki in 0..k {
                let a_atom = a_atoms[(mi * k + ki) as usize];
                let b_base = b_row_bases[ki as usize];

                let mul_base = g.push_group(
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
                            base: b_base,
                            stride: 1,
                        },
                    ],
                );
                mul_bases.push(mul_base);
            }
        }

        // M ReduceSum groups, each with N atoms.
        for mi in 0..m {
            let row_mul_base = mul_bases[(mi * k) as usize];
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
                    base: row_mul_base,
                    stride: 1,
                }],
            );
        }

        g
    }
}
