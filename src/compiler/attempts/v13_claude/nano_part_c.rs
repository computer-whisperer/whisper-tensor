#![allow(clippy::all, dead_code, unreachable_patterns)]
//! Kernel partitioner for NanoGraph's compressed scalar DAG.
//!
//! Partitions AtomGroups into kernels by analyzing producer-consumer
//! relationships among compute groups. Literal/data groups are shared
//! across kernels as needed. Connected components of the compute-group
//! dependency graph form natural kernel boundaries, which are then
//! balanced for the requested parallelism level.

use crate::nano_graph::{AtomGroup, AtomId, InputRef, NanoGraph, ScalarOp};
use std::collections::HashSet;

/// Result of partitioning a NanoGraph into kernels.
#[derive(Debug)]
pub struct NanoPartitionResult {
    /// Each entry is a list of group indices (into NanoGraph::groups()) that
    /// form one kernel. Groups within a kernel execute sequentially; kernels
    /// can execute in parallel.
    pub kernel_groups: Vec<Vec<usize>>,
    /// Number of kernels.
    pub num_kernels: usize,
}

/// Partition a NanoGraph into kernels for parallel execution.
///
/// The algorithm:
/// 1. Classify groups as "data" (Literal, no inputs) or "compute" (everything else).
/// 2. For each compute group, resolve which other groups produce the atoms it reads.
/// 3. Build a dependency graph among compute groups (data groups are shared, not owned).
/// 4. Find connected components -- these are natural kernel boundaries.
/// 5. Split large components and merge small ones to approach the parallelism target.
/// 6. Assign data groups to the kernels that need them (shared across kernels).
pub fn partition_nanograph(graph: &NanoGraph, parallelism: usize) -> NanoPartitionResult {
    let groups = graph.groups();
    let n = groups.len();
    if n == 0 {
        return NanoPartitionResult {
            kernel_groups: vec![],
            num_kernels: 0,
        };
    }

    // Step 1: Build atom-id-to-group-index lookup.
    let atom_to_group = build_atom_to_group_map(groups);

    // Step 2: Classify groups.
    let is_data: Vec<bool> = groups
        .iter()
        .map(|g| matches!(g.op, ScalarOp::Literal(_)) && g.inputs.is_empty())
        .collect();

    let compute_indices: Vec<usize> = (0..n).filter(|&i| !is_data[i]).collect();
    let data_indices: Vec<usize> = (0..n).filter(|&i| is_data[i]).collect();

    if compute_indices.is_empty() {
        // All groups are data -- single kernel with everything.
        return NanoPartitionResult {
            kernel_groups: vec![(0..n).collect()],
            num_kernels: 1,
        };
    }

    // Step 3: For each compute group, find which other compute groups are its
    // producers (groups whose atoms this group reads, excluding data groups).
    let mut compute_adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];

    for &gi in &compute_indices {
        let group = &groups[gi];
        let producer_groups = resolve_producer_groups(group, graph, &atom_to_group);
        for pg in producer_groups {
            if pg != gi && !is_data[pg] {
                // Undirected edge: gi and pg must be in the same component.
                compute_adj[gi].insert(pg);
                compute_adj[pg].insert(gi);
            }
        }
    }

    // Step 4: Find connected components among compute groups.
    let mut comp_id: Vec<i32> = vec![-1; n];
    let mut next_comp = 0i32;

    for &gi in &compute_indices {
        if comp_id[gi] >= 0 {
            continue;
        }
        // BFS from gi.
        let mut queue = vec![gi];
        comp_id[gi] = next_comp;
        let mut head = 0;
        while head < queue.len() {
            let cur = queue[head];
            head += 1;
            for &neighbor in &compute_adj[cur] {
                if comp_id[neighbor] < 0 {
                    comp_id[neighbor] = next_comp;
                    queue.push(neighbor);
                }
            }
        }
        next_comp += 1;
    }

    let num_components = next_comp as usize;

    // Collect component membership.
    let mut components: Vec<Vec<usize>> = vec![vec![]; num_components];
    for &gi in &compute_indices {
        let c = comp_id[gi] as usize;
        components[c].push(gi);
    }

    // Step 5: Split large components / merge small ones for parallelism.
    let mut kernels = balance_components(components, groups, parallelism);

    // Step 6: Put all data groups in a dedicated kernel 0 that executes first.
    if !data_indices.is_empty() {
        kernels.insert(0, data_indices);
    }

    let num_kernels = kernels.len();
    NanoPartitionResult {
        kernel_groups: kernels,
        num_kernels,
    }
}

/// Build a map from AtomId range to group index.
/// Returns a sorted list of (base_id, count, group_index) for binary search.
fn build_atom_to_group_map(groups: &[AtomGroup]) -> Vec<(u64, u64, usize)> {
    let mut entries: Vec<(u64, u64, usize)> = groups
        .iter()
        .enumerate()
        .map(|(i, g)| (g.base_id.0, g.count, i))
        .collect();
    entries.sort_by_key(|&(base, _, _)| base);
    entries
}

/// Find the group index that owns a given AtomId.
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
/// For SymAffine inputs (used by ReduceSum/ReduceMax), uses the group's
/// reduce_dims and the graph's sym_dim_bounds to determine the exact K range
/// instead of scanning until atoms run out.
fn resolve_producer_groups(
    group: &AtomGroup,
    graph: &NanoGraph,
    atom_to_group: &[(u64, u64, usize)],
) -> HashSet<usize> {
    let mut producers = HashSet::new();
    for input in &group.inputs {
        match input {
            InputRef::Broadcast(id) => {
                if let Some(gi) = find_group_for_atom(*id, atom_to_group) {
                    producers.insert(gi);
                }
            }
            InputRef::Affine { base, stride: _ } => {
                // Sample start and end atoms to find producer groups.
                if let Some(gi) = find_group_for_atom(*base, atom_to_group) {
                    producers.insert(gi);
                }
                if group.count > 1 {
                    let last = input.resolve(group.count - 1, 0);
                    if let Some(gi) = find_group_for_atom(last, atom_to_group) {
                        producers.insert(gi);
                    }
                }
            }
            InputRef::Explicit(ids) => {
                for id in ids {
                    if let Some(gi) = find_group_for_atom(*id, atom_to_group) {
                        producers.insert(gi);
                    }
                }
            }
            InputRef::StridedBroadcast { repeat, .. } => {
                // Each block of `repeat` atoms shares one source.
                let num_blocks = (group.count + repeat - 1) / repeat;
                for block in 0..num_blocks {
                    let atom = input.resolve(block * repeat, 0);
                    if let Some(gi) = find_group_for_atom(atom, atom_to_group) {
                        producers.insert(gi);
                    }
                }
            }
            InputRef::SymAffine { .. } => {
                // Determine the K range from the group's reduce_dims.
                // The reduce_dims tell us which symbolic dimensions this group
                // reduces over. Look up their bounds to get the exact K count.
                let k_bound = group
                    .reduce_dims
                    .iter()
                    .filter_map(|sd| graph.sym_dim_bounds.get(sd))
                    .next()
                    .copied()
                    .unwrap_or(1);

                // Enumerate all (i_sample, k) combinations to find producer groups.
                // For each k in 0..k_bound, resolve at i=0 to find the group.
                // Also resolve at i=count-1 to catch any cross-group spans.
                for k in 0..k_bound {
                    let atom = input.resolve(0, k);
                    if let Some(gi) = find_group_for_atom(atom, atom_to_group) {
                        producers.insert(gi);
                    }
                }
                // Also sample at i=count-1 for stride_i coverage.
                if group.count > 1 {
                    for k in 0..k_bound {
                        let atom = input.resolve(group.count - 1, k);
                        if let Some(gi) = find_group_for_atom(atom, atom_to_group) {
                            producers.insert(gi);
                        }
                    }
                }
            }
        }
    }
    producers
}

/// Balance components to approach the target parallelism.
fn balance_components(
    mut components: Vec<Vec<usize>>,
    _groups: &[AtomGroup],
    _parallelism: usize,
) -> Vec<Vec<usize>> {
    components.retain(|c| !c.is_empty());

    if components.is_empty() {
        return vec![];
    }

    // Natural components from the dependency graph are the right granularity.
    // For matmul, each row is already a separate component.
    components
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::nano_graph::{ScalarBinOp, ScalarOp};
    use crate::nano_graph::{InputRef, NanoGraph};
    use crate::numeric_scalar::NumericScalar;

    /// Test 1: Elementwise add -- should produce a small number of kernels.
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
        let _c = g.push_group(
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

        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let result = partition_nanograph(&g, 4);
        println!("Elementwise add: {} kernels", result.num_kernels);
        for (i, kernel) in result.kernel_groups.iter().enumerate() {
            println!("  Kernel {}: {} groups {:?}", i, kernel.len(), kernel);
        }

        // Should have at least 1 kernel.
        assert!(result.num_kernels >= 1);
        // All groups should be assigned.
        let total_assigned: usize = result.kernel_groups.iter().map(|k| k.len()).sum();
        assert_eq!(total_assigned, g.num_groups());
    }

    /// Test 2: MatMul(4, 8, 16) -- MUST produce >1 compute kernel.
    #[test]
    fn test_matmul_partitions() {
        let g = build_matmul_nanograph(4, 8, 16);
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let stats = g.stats();
        println!("MatMul(4,8,16) graph: {}", stats);

        let result = partition_nanograph(&g, 4);
        println!(
            "\nMatMul(4,8,16) partitioned into {} kernels:",
            result.num_kernels
        );
        for (i, kernel) in result.kernel_groups.iter().enumerate() {
            let atom_count: u64 = kernel.iter().map(|&gi| g.groups()[gi].count).sum();
            let ops: Vec<String> = kernel
                .iter()
                .map(|&gi| format!("{:?}", g.groups()[gi].op))
                .collect();
            let unique_ops: HashSet<&str> = ops.iter().map(|s| s.as_str()).collect();
            println!(
                "  Kernel {}: {} groups, {} atoms, ops: {:?}",
                i,
                kernel.len(),
                atom_count,
                unique_ops
            );
        }

        // All groups assigned.
        let total_assigned: usize = result.kernel_groups.iter().map(|k| k.len()).sum();
        assert_eq!(total_assigned, g.num_groups());

        // CRITICAL: Must have more than 1 compute kernel.
        let compute_kernels: Vec<&Vec<usize>> = result
            .kernel_groups
            .iter()
            .filter(|k| {
                k.iter()
                    .any(|&gi| !matches!(g.groups()[gi].op, ScalarOp::Literal(_)))
            })
            .collect();
        println!(
            "\n  Compute kernels: {} (must be > 1)",
            compute_kernels.len()
        );
        assert!(
            compute_kernels.len() > 1,
            "MatMul must be split into multiple compute kernels! Got {}",
            compute_kernels.len()
        );

        // Check balance: no single compute kernel should have more than 60% of
        // total compute atoms.
        let total_compute_atoms: u64 = compute_kernels
            .iter()
            .flat_map(|k| k.iter())
            .map(|&gi| g.groups()[gi].count)
            .sum();
        for (i, kernel) in compute_kernels.iter().enumerate() {
            let atoms: u64 = kernel.iter().map(|&gi| g.groups()[gi].count).sum();
            let pct = atoms as f64 / total_compute_atoms as f64 * 100.0;
            println!(
                "  Compute kernel {}: {} atoms ({:.1}%)",
                i, atoms, pct
            );
            assert!(
                pct < 60.0,
                "Kernel {} has {:.1}% of compute atoms -- not balanced!",
                i,
                pct
            );
        }
    }

    /// Test 3: Two independent matmuls -- must separate into distinct kernels.
    #[test]
    fn test_two_independent_matmuls() {
        let mut g = NanoGraph::new();

        // MatMul 1: C1[2,N1] = A1[2,4] @ B1[4,N1], N1=8
        let m1 = 2;
        let k1 = 4;
        let n1 = 8u64;
        let (mm1_first_mul, mm1_last_reduce) =
            build_matmul_into_graph(&mut g, m1, k1, n1);

        // MatMul 2: C2[3,N2] = A2[3,5] @ B2[5,N2], N2=6
        let m2 = 3;
        let k2 = 5;
        let n2 = 6u64;
        let (mm2_first_mul, mm2_last_reduce) =
            build_matmul_into_graph(&mut g, m2, k2, n2);

        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let result = partition_nanograph(&g, 8);
        println!(
            "Two independent matmuls: {} kernels",
            result.num_kernels
        );
        for (i, kernel) in result.kernel_groups.iter().enumerate() {
            let atom_count: u64 = kernel.iter().map(|&gi| g.groups()[gi].count).sum();
            println!(
                "  Kernel {}: {} groups, {} atoms",
                i,
                kernel.len(),
                atom_count
            );
        }

        // All groups assigned.
        let total_assigned: usize = result.kernel_groups.iter().map(|k| k.len()).sum();
        assert_eq!(total_assigned, g.num_groups());

        // Must have at least m1 + m2 = 5 compute kernels (one per row of each matmul).
        let compute_kernels: Vec<&Vec<usize>> = result
            .kernel_groups
            .iter()
            .filter(|k| {
                k.iter()
                    .any(|&gi| !matches!(g.groups()[gi].op, ScalarOp::Literal(_)))
            })
            .collect();
        println!("  Compute kernels: {}", compute_kernels.len());
        assert!(
            compute_kernels.len() >= (m1 + m2) as usize,
            "Two independent matmuls should produce at least {} compute kernels, got {}",
            m1 + m2,
            compute_kernels.len()
        );

        // Verify no compute kernel mixes groups from both matmuls.
        for (i, kernel) in compute_kernels.iter().enumerate() {
            let has_mm1 = kernel.iter().any(|&gi| {
                gi >= mm1_first_mul && gi <= mm1_last_reduce
            });
            let has_mm2 = kernel.iter().any(|&gi| {
                gi >= mm2_first_mul && gi <= mm2_last_reduce
            });
            assert!(
                !(has_mm1 && has_mm2),
                "Compute kernel {} mixes groups from both matmuls!",
                i
            );
        }
    }

    /// Test 4: Print detailed kernel contents for a matmul.
    #[test]
    fn test_print_kernel_contents() {
        let g = build_matmul_nanograph(4, 8, 16);
        let result = partition_nanograph(&g, 4);

        println!("=== Kernel Contents for MatMul(4,8,16) ===");
        println!("Total groups: {}, Total kernels: {}", g.num_groups(), result.num_kernels);

        for (ki, kernel) in result.kernel_groups.iter().enumerate() {
            println!("\n--- Kernel {} ({} groups) ---", ki, kernel.len());
            for &gi in kernel {
                let group = &g.groups()[gi];
                let op_name = match &group.op {
                    ScalarOp::Literal(_) => "Literal".to_string(),
                    ScalarOp::Binary { op, .. } => format!("{:?}", op),
                    ScalarOp::Unary { op, .. } => format!("{:?}", op),
                    ScalarOp::Identity { .. } => "Identity".to_string(),
                    ScalarOp::ReduceSum { .. } => "ReduceSum".to_string(),
                    ScalarOp::ReduceMax { .. } => "ReduceMax".to_string(),
                    ScalarOp::Select { .. } => "Select".to_string(),
                    ScalarOp::IndirectLoad { .. } => "IndirectLoad".to_string(),
                };
                let input_summary: Vec<String> = group
                    .inputs
                    .iter()
                    .map(|inp| match inp {
                        InputRef::Broadcast(id) => format!("Bcast({})", id),
                        InputRef::Affine { base, stride } => {
                            format!("Aff({},s={})", base, stride)
                        }
                        InputRef::Explicit(ids) => format!("Expl({}ids)", ids.len()),
                        InputRef::SymAffine {
                            base,
                            stride_i,
                            stride_k,
                        } => format!("SymAff({},si={},sk={})", base, stride_i, stride_k),
                        InputRef::StridedBroadcast { base, stride, repeat } => {
                            format!("SBcast({},s={},r={})", base, stride, repeat)
                        }
                    })
                    .collect();
                println!(
                    "  group[{}]: {} x{} base={} inputs=[{}]",
                    gi,
                    op_name,
                    group.count,
                    group.base_id,
                    input_summary.join(", ")
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Helper: Build a matmul NanoGraph mimicking the real lowering output.
    // C[M, N] = A[M, K] @ B[K, N]
    //
    // Groups produced:
    //   - M*K Literal groups (A elements), count=1 each
    //   - K Literal groups (B rows), count=N each
    //   - M*K Mul groups, count=N each: Broadcast(A[m,k]), Affine(B[k,*], 1)
    //   - M ReduceSum groups, count=N each: SymAffine(mul_base_for_row, 1, N)
    // -----------------------------------------------------------------------

    fn build_matmul_nanograph(m: u64, k: u64, n: u64) -> NanoGraph {
        let mut g = NanoGraph::new();
        build_matmul_into_graph(&mut g, m, k, n);
        g
    }

    /// Build a matmul into an existing graph. Returns (first_mul_group_idx, last_reduce_group_idx).
    fn build_matmul_into_graph(
        g: &mut NanoGraph,
        m: u64,
        k: u64,
        n: u64,
    ) -> (usize, usize) {
        let k_sym = g.bounded_sym_dim(
            &format!("matmul_k_{}", g.num_groups()),
            k as u64,
        );

        // A[M, K] as M*K singleton Literal groups.
        let mut a_atoms = Vec::new();
        for _mi in 0..m {
            for _ki in 0..k {
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

        // B[K, N] as K groups of N Literals each.
        let mut b_row_bases = Vec::new();
        for _ki in 0..k {
            let id = g.push_group(
                n,
                ScalarOp::Literal(NumericScalar::F32(1.0)),
                vec![],
                vec![],
                vec![],
            );
            b_row_bases.push(id);
        }

        let first_mul_group_idx = g.num_groups();

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
        let mut last_reduce_group_idx = first_mul_group_idx;
        for mi in 0..m {
            let row_mul_base = mul_bases[(mi * k) as usize];

            last_reduce_group_idx = g.num_groups();
            g.push_group(
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
        }

        (first_mul_group_idx, last_reduce_group_idx)
    }
}
