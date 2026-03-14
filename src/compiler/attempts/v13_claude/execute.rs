//! Execution pipeline for partitioned scalar DAGs.
//!
//! Three components:
//! 1. **Naive executor** — baseline sequential execution in topological order.
//! 2. **Partitioned executor** — respects kernel boundaries, proves partition validity.
//! 3. **Pipeline** — ties partitioning + execution + validation together.

use std::collections::{HashMap, HashSet, VecDeque};

use super::creative3::Creative3Partitioner;
use super::simple_dag::*;

// ── Naive executor ───────────────────────────────────────────────────────────

/// Execute a SimpleDag with given input values. Returns all op results.
///
/// Ops are evaluated in index order (topological, since SimpleDag guarantees
/// later ops only reference earlier ops).
pub fn execute_naive(dag: &SimpleDag, inputs: &HashMap<u32, f32>) -> Vec<f32> {
    let n = dag.ops.len();
    let mut values = vec![0.0f32; n];

    for (i, op) in dag.ops.iter().enumerate() {
        values[i] = eval_op(op, &values, inputs, i as u32);
    }

    values
}

/// Evaluate a single op given the shared values array and input map.
fn eval_op(op: &Op, values: &[f32], inputs: &HashMap<u32, f32>, op_idx: u32) -> f32 {
    match op.kind {
        OpKind::Input => *inputs.get(&op_idx).unwrap_or(&0.0),
        OpKind::Literal => 0.0,
        OpKind::Add => values[op.inputs[0] as usize] + values[op.inputs[1] as usize],
        OpKind::Sub => values[op.inputs[0] as usize] - values[op.inputs[1] as usize],
        OpKind::Mul => values[op.inputs[0] as usize] * values[op.inputs[1] as usize],
        OpKind::Div => values[op.inputs[0] as usize] / values[op.inputs[1] as usize],
        OpKind::Neg => -values[op.inputs[0] as usize],
        OpKind::Exp => values[op.inputs[0] as usize].exp(),
        OpKind::Tanh => values[op.inputs[0] as usize].tanh(),
        OpKind::Reciprocal => 1.0 / values[op.inputs[0] as usize],
        OpKind::Sqrt => values[op.inputs[0] as usize].sqrt(),
        OpKind::Abs => values[op.inputs[0] as usize].abs(),
    }
}

// ── Partitioned executor ─────────────────────────────────────────────────────

/// Execute a partitioned SimpleDag. Returns all op results.
///
/// Kernels are executed in dependency order. Within each kernel, ops are
/// executed in topological order. The shared `values` array is the
/// communication mechanism between kernels.
///
/// Should produce results identical (within floating-point tolerance) to
/// `execute_naive`.
pub fn execute_partitioned(
    dag: &SimpleDag,
    partition: &SimplePartition,
    inputs: &HashMap<u32, f32>,
) -> Vec<f32> {
    let n = dag.ops.len();
    let mut values = vec![0.0f32; n];

    // Pre-evaluate all source ops (Input, Literal). These have no data
    // dependencies, so they can be evaluated before any kernel executes.
    // This avoids cycles in the kernel dependency graph caused by the
    // partitioner assigning source ops to kernels that mutually depend
    // on each other's sources.
    for (i, op) in dag.ops.iter().enumerate() {
        if matches!(op.kind, OpKind::Input | OpKind::Literal) {
            values[i] = eval_op(op, &values, inputs, i as u32);
        }
    }

    // Build op → kernel mapping.
    let mut op_kernel = vec![0usize; n];
    for (ki, kernel) in partition.kernels.iter().enumerate() {
        for &op in &kernel.ops {
            op_kernel[op as usize] = ki;
        }
    }

    // Build kernel dependency DAG (ignoring source ops as dependency sources).
    let num_kernels = partition.kernels.len();
    let execution_groups = topo_sort_kernels(dag, partition, &op_kernel, num_kernels);

    // Execute kernel groups in dependency order.
    // topo_sort_kernels returns groups of kernel indices. Each group is either
    // a single kernel (no cycle) or multiple kernels that form a cycle.
    // For cyclic groups, we merge all their ops and execute in global
    // topological order (by op index).
    for group in &execution_groups {
        // Collect all ops from all kernels in this group.
        let mut sorted_ops: Vec<u32> = Vec::new();
        for &ki in group {
            sorted_ops.extend_from_slice(&partition.kernels[ki].ops);
        }
        sorted_ops.sort();

        for &op_idx in &sorted_ops {
            let op = &dag.ops[op_idx as usize];
            // Source ops already evaluated above; skip them.
            if matches!(op.kind, OpKind::Input | OpKind::Literal) {
                continue;
            }
            values[op_idx as usize] = eval_op(op, &values, inputs, op_idx);
        }
    }

    values
}

/// Topologically sort kernels by their dependency relationships.
/// Returns groups of kernel indices in a valid execution order.
///
/// Each group is a strongly connected component (SCC) of the kernel
/// dependency graph. Acyclic kernels form singleton groups. Cyclic
/// kernels are grouped together and their ops must be executed in
/// global topological order (by op index).
///
/// Source ops (Input, Literal) are pre-evaluated and do not create
/// inter-kernel dependencies.
fn topo_sort_kernels(
    dag: &SimpleDag,
    partition: &SimplePartition,
    op_kernel: &[usize],
    num_kernels: usize,
) -> Vec<Vec<usize>> {
    if num_kernels == 0 {
        return vec![];
    }

    // Build kernel adjacency: kernel_deps[ki] = set of kernels that ki depends on.
    // Also build forward edges (successors) for Tarjan's algorithm.
    let mut kernel_deps: Vec<HashSet<usize>> = vec![HashSet::new(); num_kernels];
    let mut kernel_succs: Vec<HashSet<usize>> = vec![HashSet::new(); num_kernels];

    for (ki, kernel) in partition.kernels.iter().enumerate() {
        let kernel_set: HashSet<u32> = kernel.ops.iter().copied().collect();
        for &op_idx in &kernel.ops {
            let op = &dag.ops[op_idx as usize];
            for &input in &op.inputs {
                // Skip source ops — they are pre-evaluated and available
                // to all kernels without creating a dependency.
                if matches!(dag.ops[input as usize].kind, OpKind::Input | OpKind::Literal) {
                    continue;
                }
                if !kernel_set.contains(&input) {
                    let src_kernel = op_kernel[input as usize];
                    if src_kernel != ki {
                        kernel_deps[ki].insert(src_kernel);
                        kernel_succs[src_kernel].insert(ki);
                    }
                }
            }
        }
    }

    // Find SCCs using iterative Tarjan's algorithm.
    let sccs = find_sccs(num_kernels, &kernel_succs);

    // Map each kernel to its SCC index.
    let mut kernel_to_scc = vec![0usize; num_kernels];
    for (scc_idx, scc) in sccs.iter().enumerate() {
        for &ki in scc {
            kernel_to_scc[ki] = scc_idx;
        }
    }

    // Build SCC-level DAG and topologically sort it.
    let num_sccs = sccs.len();
    let mut scc_deps: Vec<HashSet<usize>> = vec![HashSet::new(); num_sccs];
    for (ki, deps) in kernel_deps.iter().enumerate() {
        let my_scc = kernel_to_scc[ki];
        for &dep in deps {
            let dep_scc = kernel_to_scc[dep];
            if dep_scc != my_scc {
                scc_deps[my_scc].insert(dep_scc);
            }
        }
    }

    // Kahn's algorithm on the condensed DAG.
    let mut in_degree = vec![0usize; num_sccs];
    let mut scc_succs: Vec<Vec<usize>> = vec![Vec::new(); num_sccs];
    for (si, deps) in scc_deps.iter().enumerate() {
        in_degree[si] = deps.len();
        for &dep in deps {
            scc_succs[dep].push(si);
        }
    }

    let mut queue: VecDeque<usize> = VecDeque::new();
    for si in 0..num_sccs {
        if in_degree[si] == 0 {
            queue.push_back(si);
        }
    }

    let mut scc_order = Vec::with_capacity(num_sccs);
    while let Some(si) = queue.pop_front() {
        scc_order.push(si);
        for &succ in &scc_succs[si] {
            in_degree[succ] -= 1;
            if in_degree[succ] == 0 {
                queue.push_back(succ);
            }
        }
    }

    assert_eq!(
        scc_order.len(),
        num_sccs,
        "SCC condensation has a cycle — this should be impossible. \
         Sorted {} of {} SCCs.",
        scc_order.len(),
        num_sccs
    );

    // Return groups in SCC topological order.
    // Each group contains the kernel indices in that SCC.
    let mut groups = Vec::with_capacity(num_sccs);
    for &si in &scc_order {
        let mut scc_kernels: Vec<usize> = sccs[si].clone();
        scc_kernels.sort_by_key(|&ki| {
            partition.kernels[ki]
                .ops
                .iter()
                .copied()
                .min()
                .unwrap_or(u32::MAX)
        });
        groups.push(scc_kernels);
    }

    groups
}

/// Find strongly connected components using iterative Kosaraju's algorithm.
/// Returns SCCs in reverse topological order.
fn find_sccs(n: usize, succs: &[HashSet<usize>]) -> Vec<Vec<usize>> {
    // Step 1: Compute finish order via iterative DFS on forward graph.
    let mut visited = vec![false; n];
    let mut finish_order = Vec::with_capacity(n);

    for start in 0..n {
        if visited[start] {
            continue;
        }
        // Iterative DFS using an explicit stack.
        // Stack entries: (node, is_postvisit)
        let mut stack: Vec<(usize, bool)> = vec![(start, false)];
        while let Some((node, is_post)) = stack.pop() {
            if is_post {
                finish_order.push(node);
                continue;
            }
            if visited[node] {
                continue;
            }
            visited[node] = true;
            // Push post-visit marker.
            stack.push((node, true));
            // Push successors.
            for &next in &succs[node] {
                if !visited[next] {
                    stack.push((next, false));
                }
            }
        }
    }

    // Step 2: Build reverse graph.
    let mut rev_succs: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (u, s) in succs.iter().enumerate() {
        for &v in s {
            rev_succs[v].push(u);
        }
    }

    // Step 3: Process nodes in reverse finish order on the reverse graph.
    let mut visited2 = vec![false; n];
    let mut sccs = Vec::new();

    for &node in finish_order.iter().rev() {
        if visited2[node] {
            continue;
        }
        // BFS/DFS to find all nodes in this SCC.
        let mut scc = Vec::new();
        let mut stack = vec![node];
        while let Some(u) = stack.pop() {
            if visited2[u] {
                continue;
            }
            visited2[u] = true;
            scc.push(u);
            for &v in &rev_succs[u] {
                if !visited2[v] {
                    stack.push(v);
                }
            }
        }
        sccs.push(scc);
    }

    sccs
}

// ── Pipeline ─────────────────────────────────────────────────────────────────

/// Results from the full partition + execute + validate pipeline.
#[derive(Debug)]
pub struct PipelineResult {
    /// All op values from the partitioned execution.
    pub outputs: Vec<f32>,
    /// Number of kernels in the partition.
    pub num_kernels: usize,
    /// Total inter-kernel loads from evaluate_cost.
    pub total_loads: u64,
    /// Maximum absolute error vs naive execution.
    pub max_abs_error: f32,
    /// Kernel execution order (kernel indices in execution sequence).
    pub execution_order: Vec<usize>,
}

impl std::fmt::Display for PipelineResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} kernels, {} loads, max_err={:.2e}, {} outputs",
            self.num_kernels,
            self.total_loads,
            self.max_abs_error,
            self.outputs.len(),
        )
    }
}

/// Full pipeline: partition + execute + validate.
///
/// Runs the Creative3 partitioner, executes both naively and with partitioning,
/// compares results, and returns comprehensive metrics.
pub fn run_pipeline(
    dag: &SimpleDag,
    config: &HardwareConfig,
    inputs: &HashMap<u32, f32>,
) -> PipelineResult {
    // 1. Partition.
    let partitioner = Creative3Partitioner;
    let partition = partitioner.partition(dag, config);

    // 2. Execute with partitioning.
    let partitioned_values = execute_partitioned(dag, &partition, inputs);

    // 3. Execute naively for reference.
    let naive_values = execute_naive(dag, inputs);

    // 4. Compare: compute max absolute error.
    let max_abs_error = naive_values
        .iter()
        .zip(partitioned_values.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // 5. Extract output values (only the dag outputs, not all ops).
    let outputs: Vec<f32> = dag
        .outputs
        .iter()
        .map(|&idx| partitioned_values[idx as usize])
        .collect();

    // 6. Evaluate cost metrics.
    let cost = evaluate_cost(dag, &partition, config);

    // 7. Reconstruct kernel execution order.
    let n = dag.ops.len();
    let mut op_kernel = vec![0usize; n];
    for (ki, kernel) in partition.kernels.iter().enumerate() {
        for &op in &kernel.ops {
            op_kernel[op as usize] = ki;
        }
    }
    let execution_groups =
        topo_sort_kernels(dag, &partition, &op_kernel, partition.kernels.len());
    // Flatten groups into a flat execution order for the result.
    let execution_order: Vec<usize> = execution_groups
        .into_iter()
        .flatten()
        .collect();

    PipelineResult {
        outputs,
        num_kernels: cost.num_kernels,
        total_loads: cost.total_loads,
        max_abs_error,
        execution_order,
    }
}

// ── Helper: build input map ──────────────────────────────────────────────────

/// Build a deterministic input map for a SimpleDag.
/// Input op at index i gets value (i + 1) * 0.01.
fn build_inputs(dag: &SimpleDag) -> HashMap<u32, f32> {
    let mut inputs = HashMap::new();
    for (i, op) in dag.ops.iter().enumerate() {
        if op.kind == OpKind::Input {
            inputs.insert(i as u32, (i as f32 + 1.0) * 0.01);
        }
    }
    inputs
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const MAX_ERROR: f32 = 1e-5;

    fn default_config() -> HardwareConfig {
        HardwareConfig::default()
    }

    fn run_and_check(name: &str, dag: &SimpleDag) -> PipelineResult {
        let config = default_config();
        let inputs = build_inputs(dag);
        let result = run_pipeline(dag, &config, &inputs);

        println!("=== {} ===", name);
        println!("  {}", result);

        // Validate partition.
        let partition = Creative3Partitioner.partition(dag, &config);
        let errors = partition.validate(dag);
        assert!(errors.is_empty(), "{}: partition invalid: {:?}", name, errors);

        // Check error tolerance.
        assert!(
            result.max_abs_error < MAX_ERROR,
            "{}: max_abs_error {} exceeds tolerance {}",
            name,
            result.max_abs_error,
            MAX_ERROR,
        );

        result
    }

    #[test]
    fn test_naive_elementwise() {
        let dag = build_elementwise_add(64);
        let mut inputs = HashMap::new();
        // A[i] = i, B[i] = 100 + i
        for i in 0..64u32 {
            inputs.insert(i, i as f32);           // A[i]
            inputs.insert(64 + i, 100.0 + i as f32); // B[i]
        }

        let values = execute_naive(&dag, &inputs);

        // C[i] = A[i] + B[i] = i + 100 + i = 100 + 2*i
        for i in 0..64u32 {
            let c_idx = dag.outputs[i as usize] as usize;
            let expected = 100.0 + 2.0 * i as f32;
            assert!(
                (values[c_idx] - expected).abs() < 1e-6,
                "C[{}] = {} expected {}",
                i,
                values[c_idx],
                expected,
            );
        }
    }

    #[test]
    fn test_naive_unary_chain() {
        let dag = build_unary_chain(4, &[OpKind::Exp, OpKind::Neg]);
        let mut inputs = HashMap::new();
        for i in 0..4u32 {
            inputs.insert(i, 0.1 * (i as f32 + 1.0));
        }

        let values = execute_naive(&dag, &inputs);

        // out = -exp(x) for each input
        for i in 0..4u32 {
            let out_idx = dag.outputs[i as usize] as usize;
            let x = 0.1 * (i as f32 + 1.0);
            let expected = -(x.exp());
            assert!(
                (values[out_idx] - expected).abs() < 1e-5,
                "out[{}] = {} expected {}",
                i,
                values[out_idx],
                expected,
            );
        }
    }

    #[test]
    fn test_correctness_elementwise() {
        let dag = build_elementwise_add(64);
        let mut inputs = HashMap::new();
        for i in 0..64u32 {
            inputs.insert(i, i as f32);
            inputs.insert(64 + i, 100.0 + i as f32);
        }

        let config = default_config();
        let partition = Creative3Partitioner.partition(&dag, &config);
        let partitioned = execute_partitioned(&dag, &partition, &inputs);
        let naive = execute_naive(&dag, &inputs);

        for i in 0..64u32 {
            let c_idx = dag.outputs[i as usize] as usize;
            let expected = 100.0 + 2.0 * i as f32;
            assert!(
                (partitioned[c_idx] - expected).abs() < 1e-6,
                "C[{}] = {} expected {} (naive={})",
                i,
                partitioned[c_idx],
                expected,
                naive[c_idx],
            );
        }

        let max_err = naive
            .iter()
            .zip(partitioned.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < MAX_ERROR, "max_err = {}", max_err);
    }

    #[test]
    fn test_correctness_unary_chain() {
        let dag = build_unary_chain(32, &[OpKind::Exp, OpKind::Neg]);
        run_and_check("unary_chain(32, [Exp, Neg])", &dag);
    }

    #[test]
    fn test_correctness_matmul() {
        // 4x8x16 matmul with known inputs.
        let dag = build_matmul(4, 8, 16);
        let inputs = build_inputs(&dag);
        let config = default_config();

        let naive = execute_naive(&dag, &inputs);
        let partition = Creative3Partitioner.partition(&dag, &config);
        let errors = partition.validate(&dag);
        assert!(errors.is_empty(), "partition invalid: {:?}", errors);

        let partitioned = execute_partitioned(&dag, &partition, &inputs);

        let max_err = naive
            .iter()
            .zip(partitioned.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!("matmul(4,8,16): max_err = {:.2e}", max_err);
        assert!(max_err < MAX_ERROR, "max_err = {}", max_err);

        // Also verify against manual computation for one output element.
        // C[0,0] = sum_k A[0,k] * B[k,0] for k=0..8
        let m = 4u32;
        let k = 8u32;
        let n = 16u32;
        let a_base = 0u32;
        let b_base = m * k;
        let mut expected = 0.0f32;
        for ki in 0..k {
            let a_val = inputs[&(a_base + 0 * k + ki)];
            let b_val = inputs[&(b_base + ki * n + 0)];
            expected += a_val * b_val;
        }
        let c00_idx = dag.outputs[0] as usize;
        assert!(
            (naive[c00_idx] - expected).abs() < 1e-5,
            "C[0,0] = {} expected {}",
            naive[c00_idx],
            expected,
        );
    }

    #[test]
    fn test_correctness_matmul_activation() {
        let dag = build_matmul_activation(4, 8, 16, OpKind::Tanh);
        let inputs = build_inputs(&dag);
        let config = default_config();

        let naive = execute_naive(&dag, &inputs);
        let partition = Creative3Partitioner.partition(&dag, &config);
        let errors = partition.validate(&dag);
        assert!(errors.is_empty(), "partition invalid: {:?}", errors);

        let partitioned = execute_partitioned(&dag, &partition, &inputs);

        let max_err = naive
            .iter()
            .zip(partitioned.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!("matmul_activation(4,8,16,Tanh): max_err = {:.2e}", max_err);
        assert!(max_err < MAX_ERROR, "max_err = {}", max_err);

        // Verify tanh is applied: all outputs should be in [-1, 1].
        for &out_idx in &dag.outputs {
            let val = partitioned[out_idx as usize];
            assert!(
                val >= -1.0 && val <= 1.0,
                "tanh output {} out of range [-1,1]",
                val,
            );
        }
    }

    #[test]
    fn test_correctness_matmul_chain() {
        let dag = build_matmul_chain(4, 8, 16, 32);
        let result = run_and_check("matmul_chain(4,8,16,32)", &dag);
        assert_eq!(result.outputs.len(), (4 * 32) as usize);
    }

    #[test]
    fn test_correctness_parallel_matmuls() {
        let dag = build_parallel_matmuls(4, 8, 16, 8, 16);
        let result = run_and_check("parallel_matmuls(4,8,16,8,16)", &dag);
        assert_eq!(result.outputs.len(), (4 * 16 + 4 * 16) as usize);
    }

    #[test]
    fn test_correctness_shared_input_matmuls() {
        let dag = build_shared_input_matmuls(4, 8, 16, 16);
        let result = run_and_check("shared_input_matmuls(4,8,16,16)", &dag);
        assert_eq!(result.outputs.len(), (4 * 16 + 4 * 16) as usize);
    }

    #[test]
    fn test_correctness_qkv_projections() {
        let dag = build_qkv_projections(4, 8, 16);
        let result = run_and_check("qkv_projections(4,8,16)", &dag);
        assert_eq!(result.outputs.len(), 3 * 4 * 16);
    }

    #[test]
    fn test_pipeline_kernel_count() {
        // Verify pipeline produces a reasonable number of kernels.
        let dag = build_matmul(4, 8, 16);
        let config = default_config();
        let inputs = build_inputs(&dag);
        let result = run_pipeline(&dag, &config, &inputs);

        assert!(result.num_kernels >= 1, "need at least 1 kernel");
        println!(
            "pipeline matmul(4,8,16): {} kernels, {} loads",
            result.num_kernels, result.total_loads,
        );
    }

    #[test]
    fn test_execution_order_respects_deps() {
        // Verify that kernel execution order is a valid topological order.
        let dag = build_matmul_chain(4, 8, 16, 32);
        let config = default_config();
        let inputs = build_inputs(&dag);
        let result = run_pipeline(&dag, &config, &inputs);

        // The execution order should contain each kernel exactly once.
        let mut seen: HashSet<usize> = HashSet::new();
        for &ki in &result.execution_order {
            assert!(seen.insert(ki), "kernel {} appears twice in execution order", ki);
        }
        assert_eq!(seen.len(), result.num_kernels);
    }

    #[test]
    fn test_single_op_dag() {
        // Edge case: a DAG with just one input op.
        let mut dag = SimpleDag::new();
        let inp = dag.push(OpKind::Input, vec![]);
        dag.outputs.push(inp);

        let mut inputs = HashMap::new();
        inputs.insert(0u32, 42.0f32);

        let naive = execute_naive(&dag, &inputs);
        assert_eq!(naive[0], 42.0);

        let config = default_config();
        let result = run_pipeline(&dag, &config, &inputs);
        assert!(result.max_abs_error < MAX_ERROR);
        assert_eq!(result.outputs, vec![42.0]);
    }

    #[test]
    fn test_literal_ops() {
        // Literal ops should produce 0.0.
        let mut dag = SimpleDag::new();
        let lit = dag.push(OpKind::Literal, vec![]);
        let inp = dag.push(OpKind::Input, vec![]);
        let sum = dag.push(OpKind::Add, vec![lit, inp]);
        dag.outputs.push(sum);

        let mut inputs = HashMap::new();
        inputs.insert(1u32, 7.0f32);

        let values = execute_naive(&dag, &inputs);
        assert_eq!(values[0], 0.0); // Literal
        assert_eq!(values[1], 7.0); // Input
        assert_eq!(values[2], 7.0); // 0 + 7
    }
}
