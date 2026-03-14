//! Simplified scalar DAG for partitioner development.
//!
//! Strips away NanoGraph's compression (AtomGroups, InputRef addressing modes,
//! SymDims) to force the partitioner to discover structure from raw topology.
//! Every atom is an individual op with explicit input edges. Reductions are
//! fully expanded into binary op trees — no ReduceSum/SymDim shortcut.
//!
//! This is the honest representation of what the compiler actually needs to
//! schedule: a flat DAG of scalar ops with explicit dependencies.

use std::collections::{HashMap, HashSet};

/// Scalar operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    /// External input (no inputs). Represents a value provided from outside.
    Input,
    /// Constant literal (no inputs).
    Literal,
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    // Unary
    Neg,
    Exp,
    Tanh,
    Reciprocal,
    Sqrt,
    Abs,
}

/// A single scalar op in the DAG.
#[derive(Debug, Clone)]
pub struct Op {
    pub kind: OpKind,
    /// Indices into SimpleDag::ops of this op's inputs.
    pub inputs: Vec<u32>,
}

/// A flat scalar DAG. No compression, no groups, no symbolic dims.
/// Each op is one scalar computation with explicit input edges.
#[derive(Debug, Clone)]
pub struct SimpleDag {
    pub ops: Vec<Op>,
    /// Indices of ops that are final outputs.
    pub outputs: Vec<u32>,
}

impl SimpleDag {
    pub fn new() -> Self {
        Self {
            ops: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Add an op, returns its index.
    pub fn push(&mut self, kind: OpKind, inputs: Vec<u32>) -> u32 {
        let id = self.ops.len() as u32;
        self.ops.push(Op { kind, inputs });
        id
    }

    pub fn num_ops(&self) -> u32 {
        self.ops.len() as u32
    }

    /// Compute fan-out for each op (number of ops that consume it).
    pub fn fan_out(&self) -> Vec<u32> {
        let mut counts = vec![0u32; self.ops.len()];
        for op in &self.ops {
            for &input in &op.inputs {
                counts[input as usize] += 1;
            }
        }
        counts
    }

    /// Compute fan-in for each op (number of inputs).
    pub fn fan_in(&self) -> Vec<u32> {
        self.ops.iter().map(|op| op.inputs.len() as u32).collect()
    }

    /// Validate: all input indices are valid and < the op's own index (DAG property).
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        for (i, op) in self.ops.iter().enumerate() {
            for &input in &op.inputs {
                if input as usize >= self.ops.len() {
                    errors.push(format!("Op {} references nonexistent input {}", i, input));
                }
                if input >= i as u32 {
                    errors.push(format!("Op {} references non-earlier input {} (not a DAG)", i, input));
                }
            }
        }
        for &out in &self.outputs {
            if out as usize >= self.ops.len() {
                errors.push(format!("Output {} doesn't exist", out));
            }
        }
        errors
    }
}

// ── Partition types ──────────────────────────────────────────────────────

/// Hardware configuration for partitioning.
#[derive(Debug, Clone)]
pub struct HardwareConfig {
    /// L1 cache capacity in scalar values.
    pub cache_values: usize,
    /// Number of parallel execution units (cores, warps, etc.).
    /// The partitioner should produce at least this many independent kernels
    /// to fully utilize the hardware.
    pub parallelism: usize,
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            cache_values: 8192, // 32KB / 4 bytes
            parallelism: 8,     // 8 cores
        }
    }
}

/// A kernel: a set of op indices that execute together.
#[derive(Debug, Clone)]
pub struct SimpleKernel {
    /// Op indices in this kernel (sorted).
    pub ops: Vec<u32>,
}

/// Result of partitioning a SimpleDag.
#[derive(Debug, Clone)]
pub struct SimplePartition {
    pub kernels: Vec<SimpleKernel>,
}

impl SimplePartition {
    /// Validate: every op assigned to exactly one kernel.
    pub fn validate(&self, dag: &SimpleDag) -> Vec<String> {
        let mut errors = Vec::new();
        let mut assigned: HashMap<u32, usize> = HashMap::new();

        for (ki, kernel) in self.kernels.iter().enumerate() {
            for &op in &kernel.ops {
                if let Some(prev) = assigned.insert(op, ki) {
                    errors.push(format!("Op {} in kernel {} and {}", op, prev, ki));
                }
            }
        }

        for i in 0..dag.ops.len() as u32 {
            if !assigned.contains_key(&i) {
                errors.push(format!("Op {} not assigned to any kernel", i));
            }
        }

        errors
    }
}

/// Trait for partitioners on SimpleDag.
pub trait SimplePartitioner {
    fn partition(&self, dag: &SimpleDag, config: &HardwareConfig) -> SimplePartition;
}

// ── Cost evaluation ──────────────────────────────────────────────────────

/// Cost metrics for a partition.
#[derive(Debug, Clone)]
pub struct SimpleCost {
    pub num_kernels: usize,
    /// Total value-loads across all kernels. If a value is loaded by 3 kernels,
    /// it counts as 3. This is the primary optimization target.
    pub total_loads: u64,
    /// Number of unique values that must be materialized to memory
    /// (produced in one kernel, consumed in another).
    pub materialized_values: u64,
    /// Total compute ops.
    pub total_ops: u64,
    /// Arithmetic intensity: ops / loads.
    pub intensity: f64,
    /// How well the partition utilizes available parallelism.
    /// 1.0 = num_kernels >= parallelism, drops toward 0 as num_kernels → 1.
    pub parallelism_utilization: f64,
}

impl std::fmt::Display for SimpleCost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} kernels, {} loads, {} materialized, {:.1} intensity, {:.0}% parallel util",
            self.num_kernels,
            self.total_loads,
            self.materialized_values,
            self.intensity,
            self.parallelism_utilization * 100.0,
        )
    }
}

/// Evaluate partition cost.
pub fn evaluate_cost(dag: &SimpleDag, partition: &SimplePartition, config: &HardwareConfig) -> SimpleCost {
    // Build op → kernel mapping.
    let mut op_kernel = vec![0usize; dag.ops.len()];
    for (ki, kernel) in partition.kernels.iter().enumerate() {
        for &op in &kernel.ops {
            op_kernel[op as usize] = ki;
        }
    }

    // For each kernel, find external inputs (ops from other kernels).
    let mut total_loads: u64 = 0;
    let mut materialized: HashSet<u32> = HashSet::new();

    for (ki, kernel) in partition.kernels.iter().enumerate() {
        let kernel_set: HashSet<u32> = kernel.ops.iter().copied().collect();

        for &op_idx in &kernel.ops {
            let op = &dag.ops[op_idx as usize];
            for &input in &op.inputs {
                if !kernel_set.contains(&input) {
                    total_loads += 1;
                    materialized.insert(input);
                }
            }
        }
    }

    let total_ops = dag.ops.iter()
        .filter(|op| !matches!(op.kind, OpKind::Input | OpKind::Literal))
        .count() as u64;

    let intensity = if total_loads > 0 {
        total_ops as f64 / total_loads as f64
    } else {
        f64::INFINITY
    };

    let parallelism_utilization = if config.parallelism > 0 {
        (partition.kernels.len() as f64 / config.parallelism as f64).min(1.0)
    } else {
        1.0
    };

    SimpleCost {
        num_kernels: partition.kernels.len(),
        total_loads,
        materialized_values: materialized.len() as u64,
        total_ops,
        intensity,
        parallelism_utilization,
    }
}

// ── Test graph builders ──────────────────────────────────────────────────

/// Elementwise: C[i] = A[i] + B[i] for i in 0..n.
pub fn build_elementwise_add(n: u32) -> SimpleDag {
    let mut dag = SimpleDag::new();

    // A inputs
    let a_start = dag.num_ops();
    for _ in 0..n {
        dag.push(OpKind::Input, vec![]);
    }

    // B inputs
    let b_start = dag.num_ops();
    for _ in 0..n {
        dag.push(OpKind::Input, vec![]);
    }

    // C = A + B
    for i in 0..n {
        let c = dag.push(OpKind::Add, vec![a_start + i, b_start + i]);
        dag.outputs.push(c);
    }

    dag
}

/// Unary chain: out[i] = f3(f2(f1(input[i]))).
pub fn build_unary_chain(n: u32, ops: &[OpKind]) -> SimpleDag {
    let mut dag = SimpleDag::new();

    // Inputs
    let mut prev: Vec<u32> = (0..n).map(|_| dag.push(OpKind::Input, vec![])).collect();

    for &op in ops {
        let mut next = Vec::with_capacity(n as usize);
        for i in 0..n {
            next.push(dag.push(op, vec![prev[i as usize]]));
        }
        prev = next;
    }

    dag.outputs = prev;
    dag
}

/// MatMul with fully expanded reductions.
/// C[m, n] = Σ_k A[m, k] * B[k, n]
///
/// Reductions are binary Add chains (sequential accumulation):
/// acc_0 = A[m,0]*B[0,n]
/// acc_1 = acc_0 + A[m,1]*B[1,n]
/// ...
/// C[m,n] = acc_{K-1}
///
/// Returns the dag. The structure:
/// - Ops 0..M*K: A inputs (row-major, A[m,k] = m*K + k)
/// - Ops M*K..M*K+K*N: B inputs (row-major, B[k,n] = k*N + n)
/// - Then M*K*N Mul ops
/// - Then M*N*(K-1) Add ops (reduction chains)
///
/// Total ops: M*K + K*N + M*K*N + M*N*(K-1)
pub fn build_matmul(m: u32, k: u32, n: u32) -> SimpleDag {
    let mut dag = SimpleDag::new();

    // A[M, K] inputs
    let a_base = dag.num_ops();
    for _ in 0..m * k {
        dag.push(OpKind::Input, vec![]);
    }

    // B[K, N] inputs
    let b_base = dag.num_ops();
    for _ in 0..k * n {
        dag.push(OpKind::Input, vec![]);
    }

    // For each output element C[m, n], build the Mul + Add chain.
    // We'll lay out Mul ops first, then Add chains, to keep it organized.
    // But actually, for a proper DAG ordering, we need to interleave.
    // Let's build per-output-element.

    for mi in 0..m {
        for ni in 0..n {
            // Mul ops for this output: A[mi, k] * B[k, ni] for k = 0..K
            let mut mul_ops = Vec::with_capacity(k as usize);
            for ki in 0..k {
                let a_idx = a_base + mi * k + ki;
                let b_idx = b_base + ki * n + ni;
                mul_ops.push(dag.push(OpKind::Mul, vec![a_idx, b_idx]));
            }

            // Reduction chain: sequential accumulation
            let mut acc = mul_ops[0];
            for ki in 1..k {
                acc = dag.push(OpKind::Add, vec![acc, mul_ops[ki as usize]]);
            }
            dag.outputs.push(acc);
        }
    }

    dag
}

/// Build a matmul into an existing dag, using pre-existing ops as A and B inputs.
/// Returns the output op indices (M*N outputs in row-major order).
///
/// `a_ops`: M*K op indices for A (row-major, A[m,k] = a_ops[m*k_dim + k])
/// `b_ops`: K*N op indices for B (row-major, B[k,n] = b_ops[k*n_dim + n])
pub fn build_matmul_into(
    dag: &mut SimpleDag,
    a_ops: &[u32],
    b_ops: &[u32],
    m: u32,
    k: u32,
    n: u32,
) -> Vec<u32> {
    assert_eq!(a_ops.len(), (m * k) as usize);
    assert_eq!(b_ops.len(), (k * n) as usize);

    let mut outputs = Vec::with_capacity((m * n) as usize);

    for mi in 0..m {
        for ni in 0..n {
            let mut mul_ops = Vec::with_capacity(k as usize);
            for ki in 0..k {
                let a_idx = a_ops[(mi * k + ki) as usize];
                let b_idx = b_ops[(ki * n + ni) as usize];
                mul_ops.push(dag.push(OpKind::Mul, vec![a_idx, b_idx]));
            }

            let mut acc = mul_ops[0];
            for ki in 1..k {
                acc = dag.push(OpKind::Add, vec![acc, mul_ops[ki as usize]]);
            }
            outputs.push(acc);
        }
    }

    outputs
}

/// Two independent matmuls with no shared inputs.
/// C1[M, N1] = A1[M, K1] @ B1[K1, N1]
/// C2[M, N2] = A2[M, K2] @ B2[K2, N2]
///
/// Tests whether the partitioner can identify two separate patterns
/// and keep them appropriately separated.
pub fn build_parallel_matmuls(
    m: u32,
    k1: u32, n1: u32,
    k2: u32, n2: u32,
) -> SimpleDag {
    let mut dag = SimpleDag::new();

    // A1, B1 inputs
    let a1: Vec<u32> = (0..m * k1).map(|_| dag.push(OpKind::Input, vec![])).collect();
    let b1: Vec<u32> = (0..k1 * n1).map(|_| dag.push(OpKind::Input, vec![])).collect();

    // A2, B2 inputs
    let a2: Vec<u32> = (0..m * k2).map(|_| dag.push(OpKind::Input, vec![])).collect();
    let b2: Vec<u32> = (0..k2 * n2).map(|_| dag.push(OpKind::Input, vec![])).collect();

    // Build both matmuls
    let out1 = build_matmul_into(&mut dag, &a1, &b1, m, k1, n1);
    let out2 = build_matmul_into(&mut dag, &a2, &b2, m, k2, n2);

    dag.outputs.extend(out1);
    dag.outputs.extend(out2);
    dag
}

/// Two matmuls sharing the same A input (like Q and K projections in attention).
/// Q[M, Dq] = X[M, D] @ Wq[D, Dq]
/// K[M, Dk] = X[M, D] @ Wk[D, Dk]
///
/// Tests whether the partitioner recognizes shared input and tiles
/// both matmuls to share loads of X.
pub fn build_shared_input_matmuls(
    m: u32,
    d: u32,
    dq: u32,
    dk: u32,
) -> SimpleDag {
    let mut dag = SimpleDag::new();

    // Shared input X[M, D]
    let x: Vec<u32> = (0..m * d).map(|_| dag.push(OpKind::Input, vec![])).collect();

    // Wq[D, Dq]
    let wq: Vec<u32> = (0..d * dq).map(|_| dag.push(OpKind::Input, vec![])).collect();

    // Wk[D, Dk]
    let wk: Vec<u32> = (0..d * dk).map(|_| dag.push(OpKind::Input, vec![])).collect();

    // Q = X @ Wq
    let q = build_matmul_into(&mut dag, &x, &wq, m, d, dq);

    // K = X @ Wk (same X!)
    let k = build_matmul_into(&mut dag, &x, &wk, m, d, dk);

    dag.outputs.extend(q);
    dag.outputs.extend(k);
    dag
}

/// Three matmuls sharing input (Q, K, V projections) followed by
/// elementwise ops — a simplified attention-like pattern.
/// Q = tanh(X @ Wq)
/// K = tanh(X @ Wk)
/// V = tanh(X @ Wv)
pub fn build_qkv_projections(
    m: u32,
    d: u32,
    d_head: u32,
) -> SimpleDag {
    let mut dag = SimpleDag::new();

    // Shared input X[M, D]
    let x: Vec<u32> = (0..m * d).map(|_| dag.push(OpKind::Input, vec![])).collect();

    // Three weight matrices
    let wq: Vec<u32> = (0..d * d_head).map(|_| dag.push(OpKind::Input, vec![])).collect();
    let wk: Vec<u32> = (0..d * d_head).map(|_| dag.push(OpKind::Input, vec![])).collect();
    let wv: Vec<u32> = (0..d * d_head).map(|_| dag.push(OpKind::Input, vec![])).collect();

    // Q, K, V projections
    let q = build_matmul_into(&mut dag, &x, &wq, m, d, d_head);
    let k = build_matmul_into(&mut dag, &x, &wk, m, d, d_head);
    let v = build_matmul_into(&mut dag, &x, &wv, m, d, d_head);

    // Activation on each
    for &out in q.iter().chain(k.iter()).chain(v.iter()) {
        let act = dag.push(OpKind::Tanh, vec![out]);
        dag.outputs.push(act);
    }

    dag
}

/// MatMul followed by elementwise activation.
/// out[m, n] = activation(Σ_k A[m,k] * B[k,n])
pub fn build_matmul_activation(m: u32, k: u32, n: u32, activation: OpKind) -> SimpleDag {
    let mut dag = build_matmul(m, k, n);

    // Apply activation to each output.
    let old_outputs = std::mem::take(&mut dag.outputs);
    for &out in &old_outputs {
        let act = dag.push(activation, vec![out]);
        dag.outputs.push(act);
    }

    dag
}

/// Two chained matmuls: out = (A @ B) @ C.
/// A[M, K1], B[K1, N1], C[N1, N2].
pub fn build_matmul_chain(m: u32, k1: u32, n1: u32, n2: u32) -> SimpleDag {
    let mut dag = SimpleDag::new();

    // A[M, K1]
    let a_base = dag.num_ops();
    for _ in 0..m * k1 {
        dag.push(OpKind::Input, vec![]);
    }

    // B[K1, N1]
    let b_base = dag.num_ops();
    for _ in 0..k1 * n1 {
        dag.push(OpKind::Input, vec![]);
    }

    // First matmul: AB[m, n1] = Σ_k1 A[m, k1] * B[k1, n1]
    let mut ab_outputs = Vec::with_capacity((m * n1) as usize);
    for mi in 0..m {
        for ni in 0..n1 {
            let mut mul_ops = Vec::with_capacity(k1 as usize);
            for ki in 0..k1 {
                let a_idx = a_base + mi * k1 + ki;
                let b_idx = b_base + ki * n1 + ni;
                mul_ops.push(dag.push(OpKind::Mul, vec![a_idx, b_idx]));
            }
            let mut acc = mul_ops[0];
            for ki in 1..k1 {
                acc = dag.push(OpKind::Add, vec![acc, mul_ops[ki as usize]]);
            }
            ab_outputs.push(acc);
        }
    }

    // C[N1, N2]
    let c_base = dag.num_ops();
    for _ in 0..n1 * n2 {
        dag.push(OpKind::Input, vec![]);
    }

    // Second matmul: out[m, n2] = Σ_n1 AB[m, n1] * C[n1, n2]
    for mi in 0..m {
        for ni in 0..n2 {
            let mut mul_ops = Vec::with_capacity(n1 as usize);
            for ki in 0..n1 {
                let ab_idx = ab_outputs[(mi * n1 + ki) as usize];
                let c_idx = c_base + ki * n2 + ni;
                mul_ops.push(dag.push(OpKind::Mul, vec![ab_idx, c_idx]));
            }
            let mut acc = mul_ops[0];
            for ki in 1..n1 {
                acc = dag.push(OpKind::Add, vec![acc, mul_ops[ki as usize]]);
            }
            dag.outputs.push(acc);
        }
    }

    dag
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elementwise_valid() {
        let dag = build_elementwise_add(64);
        assert!(dag.validate().is_empty());
        assert_eq!(dag.num_ops(), 64 * 3); // A + B + C
        assert_eq!(dag.outputs.len(), 64);
    }

    #[test]
    fn test_unary_chain_valid() {
        let dag = build_unary_chain(32, &[OpKind::Exp, OpKind::Neg, OpKind::Tanh]);
        assert!(dag.validate().is_empty());
        assert_eq!(dag.num_ops(), 32 * 4); // input + 3 unary stages
    }

    #[test]
    fn test_matmul_valid() {
        let dag = build_matmul(4, 8, 16);
        let errors = dag.validate();
        assert!(errors.is_empty(), "{:?}", errors);
        // A: 32, B: 128, Mul: 4*8*16=512, Add: 4*16*7=448 → total 1120
        assert_eq!(dag.num_ops(), 4 * 8 + 8 * 16 + 4 * 8 * 16 + 4 * 16 * 7);
        assert_eq!(dag.outputs.len(), 4 * 16);
    }

    #[test]
    fn test_matmul_activation_valid() {
        let dag = build_matmul_activation(4, 8, 16, OpKind::Tanh);
        assert!(dag.validate().is_empty());
        assert_eq!(dag.outputs.len(), 4 * 16);
    }

    #[test]
    fn test_matmul_chain_valid() {
        let dag = build_matmul_chain(4, 8, 16, 32);
        let errors = dag.validate();
        assert!(errors.is_empty(), "{:?}", errors);
        assert_eq!(dag.outputs.len(), (4 * 32) as usize);
    }

    #[test]
    fn test_matmul_fan_out() {
        let dag = build_matmul(4, 8, 16);
        let fan = dag.fan_out();
        // A[0,0] is used in Mul for all N=16 columns → fan_out = 16
        assert_eq!(fan[0], 16);
        // B[0,0] is used in Mul for all M=4 rows → fan_out = 4
        let b_base = (4 * 8) as usize;
        assert_eq!(fan[b_base], 4);
    }

    #[test]
    fn test_single_kernel_cost() {
        let dag = build_elementwise_add(64);
        let partition = SimplePartition {
            kernels: vec![SimpleKernel {
                ops: (0..dag.num_ops()).collect(),
            }],
        };
        let config = HardwareConfig::default();
        let cost = evaluate_cost(&dag, &partition, &config);

        // All ops in one kernel → zero loads.
        assert_eq!(cost.total_loads, 0);
        assert_eq!(cost.num_kernels, 1);
        // But parallelism utilization is low (1/8).
        assert!(cost.parallelism_utilization < 0.2);
        println!("single kernel: {}", cost);
    }

    #[test]
    fn test_split_kernel_cost() {
        let dag = build_elementwise_add(64);
        // Split: inputs in kernel 0, adds in kernel 1.
        let partition = SimplePartition {
            kernels: vec![
                SimpleKernel { ops: (0..128).collect() },      // A and B
                SimpleKernel { ops: (128..192).collect() },     // C = A + B
            ],
        };
        let config = HardwareConfig::default();
        let cost = evaluate_cost(&dag, &partition, &config);

        // Kernel 1 needs 128 inputs from kernel 0.
        assert_eq!(cost.total_loads, 128);
        println!("split kernel: {}", cost);
    }

    #[test]
    fn test_matmul_medium_size() {
        // Verify a medium matmul builds in reasonable time.
        let dag = build_matmul(8, 16, 32);
        assert!(dag.validate().is_empty());
        let expected = 8 * 16 + 16 * 32 + 8 * 16 * 32 + 8 * 32 * 15;
        assert_eq!(dag.num_ops(), expected);
        println!("matmul 8x16x32: {} ops", dag.num_ops());
    }

    #[test]
    fn test_parallel_matmuls_valid() {
        let dag = build_parallel_matmuls(4, 8, 16, 8, 16);
        let errors = dag.validate();
        assert!(errors.is_empty(), "{:?}", errors);
        // Two independent matmuls, each 4x16 outputs.
        assert_eq!(dag.outputs.len(), (4 * 16 + 4 * 16) as usize);
        println!("parallel matmuls: {} ops", dag.num_ops());
    }

    #[test]
    fn test_parallel_matmuls_no_shared_inputs() {
        let dag = build_parallel_matmuls(4, 8, 16, 8, 16);
        let fan = dag.fan_out();
        // A1[0,0] should have fan_out = N1 = 16 (not more — no sharing with matmul 2).
        assert_eq!(fan[0], 16);
    }

    #[test]
    fn test_shared_input_matmuls_valid() {
        let dag = build_shared_input_matmuls(4, 8, 16, 16);
        let errors = dag.validate();
        assert!(errors.is_empty(), "{:?}", errors);
        assert_eq!(dag.outputs.len(), (4 * 16 + 4 * 16) as usize);
        println!("shared input matmuls: {} ops", dag.num_ops());
    }

    #[test]
    fn test_shared_input_matmuls_fan_out() {
        let dag = build_shared_input_matmuls(4, 8, 16, 16);
        let fan = dag.fan_out();
        // X[0,0] is used in BOTH matmuls — fan_out = Dq + Dk = 32.
        assert_eq!(fan[0], 16 + 16);
    }

    #[test]
    fn test_qkv_projections_valid() {
        let dag = build_qkv_projections(4, 8, 16);
        let errors = dag.validate();
        assert!(errors.is_empty(), "{:?}", errors);
        // 3 projections × 4*16 outputs, each with tanh.
        assert_eq!(dag.outputs.len(), 3 * 4 * 16);
        println!("qkv projections: {} ops", dag.num_ops());
    }

    #[test]
    fn test_qkv_shared_input_fan_out() {
        let dag = build_qkv_projections(4, 8, 16);
        let fan = dag.fan_out();
        // X[0,0] fans out to all three matmuls: 3 * D_head = 48.
        assert_eq!(fan[0], 3 * 16);
    }
}
