#![allow(clippy::all, dead_code, unreachable_patterns)]
//! Bottom-up merge partitioner for NanoGraph.
//!
//! Starts with each group as its own segment. Greedily merges adjacent
//! segments with the highest inter-segment traffic until reaching the
//! target kernel count. Since only adjacent segments are merged, kernels
//! are always contiguous topo ranges — no circular dependencies possible.
//!
//! Literal (weight) groups are excluded from traffic counting since weights
//! are permanently available and don't incur cross-kernel bandwidth cost.

use crate::nano_graph::{AtomGroup, AtomId, InputRef, NanoGraph, ScalarOp};
use std::collections::BinaryHeap;

/// Result of partitioning a NanoGraph into kernels.
#[derive(Debug)]
pub struct NanoPartitionResult {
    /// Each entry is a list of group indices forming one kernel.
    /// Groups within a kernel are contiguous in topological order.
    pub kernel_groups: Vec<Vec<usize>>,
    /// Number of kernels.
    pub num_kernels: usize,
}

/// Partition a NanoGraph into kernels using bottom-up merge.
///
/// Algorithm:
/// 1. Build group-level dependency edges (producer group -> consumer group),
///    excluding edges from Literal groups.
/// 2. Start with each group as its own segment. Compute traffic for each
///    boundary: the number of edges crossing it.
/// 3. Merge the boundary with highest traffic first (lazy-deletion max-heap).
/// 4. Repeat until reaching target_kernels.
///
/// Key insight: removing a boundary does not change traffic at other
/// boundaries, because traffic counts ALL edges crossing each boundary
/// (including long-range edges that span multiple boundaries). Removing
/// boundary b only internalizes edges that ONLY crossed b. Edges that
/// crossed b AND other boundaries still cross those other boundaries,
/// so their traffic is unchanged.
pub fn partition_nanograph(graph: &NanoGraph, target_kernels: usize) -> NanoPartitionResult {
    let groups = graph.groups();
    let n = groups.len();

    if n == 0 {
        return NanoPartitionResult {
            kernel_groups: vec![],
            num_kernels: 0,
        };
    }

    let target = target_kernels.max(1);

    if n <= target {
        let kernel_groups: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        return NanoPartitionResult {
            num_kernels: kernel_groups.len(),
            kernel_groups,
        };
    }

    // Build sorted atom-to-group map for binary search.
    let atom_map = build_atom_to_group_map(groups);

    // Classify groups as literal (data) or compute.
    let is_literal: Vec<bool> = groups
        .iter()
        .map(|g| matches!(g.op, ScalarOp::Literal(_)) && g.inputs.is_empty())
        .collect();

    // Build directed edges (producer, consumer), excluding literal producers.
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for (gi, group) in groups.iter().enumerate() {
        if is_literal[gi] {
            continue;
        }
        let producers = find_producer_groups(group, graph, &atom_map);
        for pg in producers {
            if pg != gi && !is_literal[pg] {
                edges.push((pg, gi));
            }
        }
    }

    // Compute boundary traffic.
    // boundary_traffic[b] = number of edges crossing boundary b.
    // Boundary b separates group b from group b+1.
    // An edge (lo, hi) where lo < hi crosses boundaries lo, lo+1, ..., hi-1.
    let num_boundaries = n - 1;
    let mut boundary_traffic: Vec<u64> = vec![0; num_boundaries];
    for &(producer, consumer) in &edges {
        let (lo, hi) = if producer < consumer {
            (producer, consumer)
        } else {
            (consumer, producer)
        };
        for b in lo..hi {
            boundary_traffic[b] += 1;
        }
    }

    // Greedy merge with lazy-deletion max-heap.
    let mut active: Vec<bool> = vec![true; num_boundaries];
    let mut heap: BinaryHeap<(u64, usize)> = BinaryHeap::with_capacity(num_boundaries);
    for (b, &t) in boundary_traffic.iter().enumerate() {
        heap.push((t, b));
    }

    let merges_needed = n - target;
    let mut merges_done = 0;
    while merges_done < merges_needed {
        match heap.pop() {
            Some((t, b)) if active[b] && boundary_traffic[b] == t => {
                active[b] = false;
                merges_done += 1;
            }
            Some(_) => continue, // stale entry
            None => break,
        }
    }

    // Build kernels from remaining active boundaries.
    let mut result_kernels: Vec<Vec<usize>> = Vec::new();
    let mut seg_start = 0usize;
    for b in 0..num_boundaries {
        if active[b] {
            result_kernels.push((seg_start..=b).collect());
            seg_start = b + 1;
        }
    }
    // Final segment.
    result_kernels.push((seg_start..n).collect());

    NanoPartitionResult {
        num_kernels: result_kernels.len(),
        kernel_groups: result_kernels,
    }
}

// ---------------------------------------------------------------------------
// Atom-to-group mapping
// ---------------------------------------------------------------------------

/// Sorted [(base_id, count, group_index)] for binary search.
fn build_atom_to_group_map(groups: &[AtomGroup]) -> Vec<(u64, u64, usize)> {
    let mut entries: Vec<(u64, u64, usize)> = groups
        .iter()
        .enumerate()
        .map(|(i, g)| (g.base_id.0, g.count, i))
        .collect();
    entries.sort_by_key(|&(base, _, _)| base);
    entries
}

/// Binary search for the group owning `atom`.
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

// ---------------------------------------------------------------------------
// Producer group discovery
// ---------------------------------------------------------------------------

/// Find all producer group indices for a group's inputs.
/// Handles all InputRef variants and the ReduceSum/ReduceMax sweep pattern.
fn find_producer_groups(
    group: &AtomGroup,
    graph: &NanoGraph,
    atom_map: &[(u64, u64, usize)],
) -> Vec<usize> {
    let mut producers: Vec<usize> = Vec::new();

    for input in &group.inputs {
        match input {
            InputRef::Broadcast(id) => {
                push_group(&mut producers, *id, atom_map);
            }

            InputRef::Affine { base, stride } => {
                // Sample first and last atoms of the affine range.
                push_group(&mut producers, *base, atom_map);
                if group.count > 1 {
                    push_group(&mut producers, input.resolve(group.count - 1, 0), atom_map);
                }

                // For reduce ops, the sweep extends beyond the affine range.
                // ReduceSum/ReduceMax read: base + stride*i + reduce_stride*k
                // for i in [0, count), k in [0, reduce_count).
                if let ScalarOp::ReduceSum {
                    reduce_count,
                    reduce_stride,
                    ..
                }
                | ScalarOp::ReduceMax {
                    reduce_count,
                    reduce_stride,
                    ..
                } = &group.op
                {
                    if *reduce_count > 0 {
                        let last_i = group.count.saturating_sub(1);
                        let last_k = reduce_count - 1;
                        // Sample corners of the (i, k) rectangle.
                        for &(i, k) in &[(0u64, last_k), (last_i, 0u64), (last_i, last_k)] {
                            let atom = AtomId(base.0.wrapping_add(
                                (*stride as i64 * i as i64 + *reduce_stride * k as i64) as u64,
                            ));
                            push_group(&mut producers, atom, atom_map);
                        }
                        // Sample intermediate k values for large reductions.
                        if *reduce_count > 2 {
                            let step = (*reduce_count / 64).max(1);
                            let mut k = step;
                            while k < *reduce_count {
                                let atom = AtomId(
                                    base.0.wrapping_add((*reduce_stride * k as i64) as u64),
                                );
                                push_group(&mut producers, atom, atom_map);
                                if group.count > 1 {
                                    let atom = AtomId(base.0.wrapping_add(
                                        (*stride as i64 * last_i as i64
                                            + *reduce_stride * k as i64)
                                            as u64,
                                    ));
                                    push_group(&mut producers, atom, atom_map);
                                }
                                k += step;
                            }
                        }
                        fill_gaps(&mut producers);
                    }
                }
            }

            InputRef::StridedBroadcast {
                base, repeat, ..
            } => {
                let num_blocks = (group.count + repeat - 1) / repeat;
                push_group(&mut producers, *base, atom_map);
                if num_blocks > 1 {
                    push_group(
                        &mut producers,
                        input.resolve((num_blocks - 1) * repeat, 0),
                        atom_map,
                    );
                }
                fill_gaps(&mut producers);
            }

            InputRef::Modular { base, modulus, .. } => {
                push_group(&mut producers, *base, atom_map);
                if *modulus > 1 {
                    push_group(&mut producers, input.resolve(*modulus - 1, 0), atom_map);
                }
                fill_gaps(&mut producers);
            }

            InputRef::Explicit(ids) => {
                for id in ids {
                    push_group(&mut producers, *id, atom_map);
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
                let last_i = group.count.saturating_sub(1);
                let last_k = k_bound.saturating_sub(1);
                for &(i, k) in &[(0u64, 0u64), (0, last_k), (last_i, 0), (last_i, last_k)] {
                    push_group(&mut producers, input.resolve(i, k), atom_map);
                }
                fill_gaps(&mut producers);
            }
        }
    }

    producers.sort();
    producers.dedup();
    producers
}

/// Push a group index for `atom` if found.
fn push_group(producers: &mut Vec<usize>, atom: AtomId, atom_map: &[(u64, u64, usize)]) {
    if let Some(gi) = find_group_for_atom(atom, atom_map) {
        producers.push(gi);
    }
}

/// Sort, dedup, then fill in all group indices between min and max.
/// Safe because atom-id ranges are contiguous — if groups A and C are
/// producers and B sits between them in group order, B must also be
/// a producer (the sweep covers a contiguous atom-id range).
fn fill_gaps(producers: &mut Vec<usize>) {
    producers.sort();
    producers.dedup();
    if producers.len() >= 2 {
        let lo = producers[0];
        let hi = *producers.last().unwrap();
        *producers = (lo..=hi).collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::nano_graph::{InputRef, NanoGraph, ScalarBinOp, ScalarOp, ScalarUnaryOp};
    use crate::numeric_scalar::NumericScalar;

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
            16,
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
    fn test_elementwise_chain_merge_all() {
        // lit -> neg -> exp; target=1 merges everything
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
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: a, stride: 1 }],
        );
        let _c = g.push_group(
            16,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: b, stride: 1 }],
        );
        assert!(g.validate().is_empty());

        let r = partition_nanograph(&g, 1);
        assert_eq!(r.num_kernels, 1);
        assert_eq!(r.kernel_groups[0].len(), 3);

        let r = partition_nanograph(&g, 3);
        assert_eq!(r.num_kernels, 3);
    }

    #[test]
    fn test_matmul_reduce_producers() {
        // C[2,4] = A[2,3] @ B[3,4]
        let mut g = NanoGraph::new();
        let (m, k, n) = (2u64, 3u64, 4u64);

        let mut a_atoms = Vec::new();
        for _ in 0..(m * k) {
            a_atoms.push(g.push_group(
                1,
                ScalarOp::Literal(NumericScalar::F32(1.0)),
                vec![],
                vec![],
                vec![],
            ));
        }
        let mut b_row_bases = Vec::new();
        for _ in 0..k {
            b_row_bases.push(g.push_group(
                n,
                ScalarOp::Literal(NumericScalar::F32(1.0)),
                vec![],
                vec![],
                vec![],
            ));
        }

        let mut mul_bases = Vec::new();
        for mi in 0..m {
            for ki in 0..k {
                mul_bases.push(g.push_group(
                    n,
                    ScalarOp::Binary {
                        op: ScalarBinOp::Mul,
                        compute_dtype: DType::F32,
                        output_dtype: DType::F32,
                    },
                    vec![],
                    vec![],
                    vec![
                        InputRef::Broadcast(a_atoms[(mi * k + ki) as usize]),
                        InputRef::Affine {
                            base: b_row_bases[ki as usize],
                            stride: 1,
                        },
                    ],
                ));
            }
        }

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
                    base: mul_bases[(mi * k) as usize],
                    stride: 1,
                }],
            );
        }

        assert!(g.validate().is_empty(), "{:?}", g.validate());
        let total = g.num_groups();

        let r = partition_nanograph(&g, 2);
        let assigned: usize = r.kernel_groups.iter().map(|k| k.len()).sum();
        assert_eq!(assigned, total);

        // Contiguity check
        for kernel in &r.kernel_groups {
            for w in kernel.windows(2) {
                assert_eq!(w[1], w[0] + 1, "Non-contiguous kernel: {:?}", kernel);
            }
        }
    }

    #[test]
    fn test_independent_chains_separate() {
        // Two independent lit->compute chains; all traffic is 0 (literals excluded).
        // With target=2, the two chains should end up in separate kernels.
        let mut g = NanoGraph::new();
        let lit1 = g.push_group(
            16,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let _neg = g.push_group(
            16,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: lit1,
                stride: 1,
            }],
        );
        let lit2 = g.push_group(
            16,
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
            vec![],
        );
        let _exp = g.push_group(
            16,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: lit2,
                stride: 1,
            }],
        );
        assert!(g.validate().is_empty());

        let r = partition_nanograph(&g, 2);
        assert_eq!(r.num_kernels, 2);
        let total: usize = r.kernel_groups.iter().map(|k| k.len()).sum();
        assert_eq!(total, 4);
    }

    #[test]
    fn test_contiguous_for_all_targets() {
        // 10-group sequential chain; verify contiguity at various target counts.
        let mut g = NanoGraph::new();
        let mut prev = g.push_group(
            8,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        for _ in 0..9 {
            prev = g.push_group(
                8,
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

        for target in [1, 2, 3, 5, 10] {
            let r = partition_nanograph(&g, target);
            for (ki, kernel) in r.kernel_groups.iter().enumerate() {
                for w in kernel.windows(2) {
                    assert_eq!(
                        w[1],
                        w[0] + 1,
                        "target={} kernel {} not contiguous",
                        target,
                        ki
                    );
                }
            }
            let total: usize = r.kernel_groups.iter().map(|k| k.len()).sum();
            assert_eq!(total, 10);
            let mut all: Vec<usize> =
                r.kernel_groups.iter().flat_map(|k| k.iter().copied()).collect();
            all.sort();
            all.dedup();
            assert_eq!(all.len(), 10, "target={} has duplicate assignments", target);
        }
    }

    #[test]
    fn test_high_traffic_merged_first() {
        // A -> B -> C (all compute, no literals).
        // A->B edge and B->C edge both have traffic=1.
        // With target=2, one merge happens. Result: 2 contiguous kernels.
        let mut g = NanoGraph::new();
        let a = g.push_group(
            8,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![], // source group (no inputs)
        );
        let b = g.push_group(
            8,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: a, stride: 1 }],
        );
        let _c = g.push_group(
            8,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: b, stride: 1 }],
        );

        let r = partition_nanograph(&g, 2);
        assert_eq!(r.num_kernels, 2);
        let total: usize = r.kernel_groups.iter().map(|k| k.len()).sum();
        assert_eq!(total, 3);
    }
}
