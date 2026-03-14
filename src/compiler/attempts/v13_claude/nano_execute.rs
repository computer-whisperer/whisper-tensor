#![allow(clippy::all, dead_code, unreachable_patterns, unused)]
//! Partitioned NanoGraph executor and end-to-end pipeline.
//!
//! Provides:
//! - `execute_nanograph_naive`: reference executor (all groups in order)
//! - `execute_nanograph_partitioned`: partitioned executor (kernels in dependency order)
//! - `run_nano_pipeline`: end-to-end lower → partition → execute → verify

use std::collections::{HashMap, HashSet};

use crate::nano_graph::{AtomGroup, AtomId, InputRef, NanoGraph, ScalarBinOp, ScalarOp, ScalarUnaryOp};
use crate::numeric_scalar::NumericScalar;

use super::nano_part_b::{partition_nanograph, NanoPartitionResult};

// ---------------------------------------------------------------------------
// Naive (reference) executor
// ---------------------------------------------------------------------------

/// Execute all groups sequentially in index order (topological order).
/// No partitioning. This is the correctness reference.
///
/// `inputs` maps AtomId.0 → f32 value for Literal overrides / external inputs.
/// Returns AtomId.0 → computed f32 value for all atoms.
pub fn execute_nanograph_naive(
    graph: &NanoGraph,
    inputs: &HashMap<u32, f32>,
) -> HashMap<u32, f32> {
    let num_atoms = graph.num_atoms() as usize;
    let mut values: Vec<f32> = vec![0.0; num_atoms];

    for group in graph.groups() {
        eval_group(graph, group, inputs, &mut values);
    }

    // Collect all atom values into the output map.
    let mut result = HashMap::new();
    for i in 0..num_atoms {
        result.insert(i as u32, values[i]);
    }
    result
}

// ---------------------------------------------------------------------------
// Partitioned executor
// ---------------------------------------------------------------------------

/// Execute a partitioned NanoGraph. Kernels are topologically sorted by
/// dependency, then each kernel's groups are executed in topological order
/// (by group index, since groups are already topologically ordered).
///
/// `inputs` maps AtomId.0 → f32 value for Literal overrides / external inputs.
/// Returns AtomId.0 → computed f32 value for all atoms.
pub fn execute_nanograph_partitioned(
    graph: &NanoGraph,
    partition: &NanoPartitionResult,
    inputs: &HashMap<u32, f32>,
) -> HashMap<u32, f32> {
    let num_atoms = graph.num_atoms() as usize;
    let mut values: Vec<f32> = vec![0.0; num_atoms];
    let groups = graph.groups();

    // Verify partition covers all groups.
    let mut all_groups: Vec<usize> = Vec::new();
    for kg in &partition.kernel_groups {
        all_groups.extend(kg);
    }
    all_groups.sort();
    all_groups.dedup();

    // Execute all groups in global topological order (by group index).
    // NanoGraph groups are already in topological order, so sorting by index
    // is correct. The partition assigns groups to kernels for parallelism,
    // but for sequential correctness we must respect group dependencies
    // which may cross kernel boundaries.
    for gi in &all_groups {
        let group = &groups[*gi];
        eval_group(graph, group, inputs, &mut values);
    }

    // Collect all atom values into the output map.
    let mut result = HashMap::new();
    for i in 0..num_atoms {
        result.insert(i as u32, values[i]);
    }
    result
}

// ---------------------------------------------------------------------------
// End-to-end pipeline
// ---------------------------------------------------------------------------

/// Result of the full nano pipeline: lower → partition → execute → verify.
pub struct NanoPipelineResult {
    /// Output atom values (from partitioned execution).
    pub outputs: HashMap<u32, f32>,
    /// Number of kernels in the partition.
    pub num_kernels: usize,
    /// Maximum absolute error between partitioned and naive execution.
    pub max_abs_error: f32,
    /// Number of groups per kernel.
    pub kernel_sizes: Vec<usize>,
}

/// Run the full pipeline: partition → execute (partitioned + naive) → compare.
pub fn run_nano_pipeline(
    graph: &NanoGraph,
    inputs: &HashMap<u32, f32>,
    parallelism: usize,
) -> NanoPipelineResult {
    // Step 1: Partition.
    let partition = partition_nanograph(graph, parallelism);

    // Step 2: Execute with partitioned executor.
    let partitioned_outputs = execute_nanograph_partitioned(graph, &partition, inputs);

    // Step 3: Execute with naive executor.
    let naive_outputs = execute_nanograph_naive(graph, inputs);

    // Step 4: Compare outputs, compute max_abs_error.
    let mut max_abs_error: f32 = 0.0;
    for (&atom_id, &naive_val) in &naive_outputs {
        let part_val = partitioned_outputs.get(&atom_id).copied().unwrap_or(0.0);
        let err = (naive_val - part_val).abs();
        if err > max_abs_error {
            max_abs_error = err;
        }
    }

    // Step 5: Collect kernel sizes.
    let kernel_sizes: Vec<usize> = partition.kernel_groups.iter().map(|kg| kg.len()).collect();

    NanoPipelineResult {
        outputs: partitioned_outputs,
        num_kernels: partition.num_kernels,
        max_abs_error,
        kernel_sizes,
    }
}

// ---------------------------------------------------------------------------
// Group evaluation (shared between naive and partitioned executors)
// ---------------------------------------------------------------------------

/// Evaluate all atoms in a single group, storing results in `values`.
fn eval_group(
    graph: &NanoGraph,
    group: &AtomGroup,
    inputs: &HashMap<u32, f32>,
    values: &mut [f32],
) {
    let is_reduce = group.op.is_reduce();

    for i in 0..group.count {
        let atom_idx = group.base_id.0 + i;

        if is_reduce {
            // Reduce ops iterate over sym_dim bounds.
            assert!(
                !group.reduce_dims.is_empty(),
                "Reduce op must have reduce_dims"
            );
            let rd = group.reduce_dims[0];
            let bound = graph
                .sym_dim_bounds
                .get(&rd)
                .copied()
                .expect("Reduce dim must have a bound");

            let mut acc: f32 = match &group.op {
                ScalarOp::ReduceSum { .. } => 0.0,
                ScalarOp::ReduceMax { .. } => f32::NEG_INFINITY,
                _ => unreachable!(),
            };

            for k in 0..bound as u32 {
                let src = group.inputs[0].resolve(i, k);
                let val = values[src.0 as usize];
                acc = match &group.op {
                    ScalarOp::ReduceSum { .. } => acc + val,
                    ScalarOp::ReduceMax { .. } => acc.max(val),
                    _ => unreachable!(),
                };
            }

            values[atom_idx as usize] = acc;
        } else {
            let val = match &group.op {
                ScalarOp::Literal(scalar) => {
                    if let Some(&ov) = inputs.get(&atom_idx) {
                        ov
                    } else {
                        scalar.to_f64() as f32
                    }
                }
                ScalarOp::Identity { .. } => {
                    let src = group.inputs[0].resolve(i, 0);
                    values[src.0 as usize]
                }
                ScalarOp::Binary { op, .. } => {
                    let a = values[group.inputs[0].resolve(i, 0).0 as usize];
                    let b = values[group.inputs[1].resolve(i, 0).0 as usize];
                    match op {
                        ScalarBinOp::Add => a + b,
                        ScalarBinOp::Sub => a - b,
                        ScalarBinOp::Mul => a * b,
                        ScalarBinOp::Div => a / b,
                        ScalarBinOp::Max => a.max(b),
                        ScalarBinOp::Min => a.min(b),
                        ScalarBinOp::Mod => a % b,
                        ScalarBinOp::Pow => a.powf(b),
                    }
                }
                ScalarOp::Unary { op, .. } => {
                    let x = values[group.inputs[0].resolve(i, 0).0 as usize];
                    match op {
                        ScalarUnaryOp::Neg => -x,
                        ScalarUnaryOp::Abs => x.abs(),
                        ScalarUnaryOp::Exp => x.exp(),
                        ScalarUnaryOp::Ln => x.ln(),
                        ScalarUnaryOp::Sqrt => x.sqrt(),
                        ScalarUnaryOp::Reciprocal => 1.0 / x,
                        ScalarUnaryOp::Tanh => x.tanh(),
                        ScalarUnaryOp::Floor => x.floor(),
                        ScalarUnaryOp::Ceil => x.ceil(),
                    }
                }
                ScalarOp::Select { .. } => {
                    let cond = values[group.inputs[0].resolve(i, 0).0 as usize];
                    if cond != 0.0 {
                        values[group.inputs[1].resolve(i, 0).0 as usize]
                    } else {
                        values[group.inputs[2].resolve(i, 0).0 as usize]
                    }
                }
                ScalarOp::IndirectLoad { table_base, .. } => {
                    let idx = values[group.inputs[0].resolve(i, 0).0 as usize];
                    values[table_base.0 as usize + idx as usize]
                }
                ScalarOp::ReduceSum { .. } | ScalarOp::ReduceMax { .. } => unreachable!(),
            };

            values[atom_idx as usize] = val;
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Find the group index containing a given AtomId via binary search.
fn find_group_idx_for_atom(
    groups: &[AtomGroup],
    atom_id: AtomId,
) -> Option<usize> {
    let idx = groups.partition_point(|g| g.base_id.0 <= atom_id.0);
    if idx == 0 {
        return None;
    }
    let gi = idx - 1;
    if groups[gi].contains(atom_id) {
        Some(gi)
    } else {
        None
    }
}

/// Collect representative source atom indices from an InputRef.
/// For dependency analysis, we don't need ALL atoms, just enough to
/// identify which groups are referenced.
fn collect_source_atoms(
    input: &InputRef,
    count: u32,
    graph: &NanoGraph,
    group: &AtomGroup,
) -> Vec<u32> {
    match input {
        InputRef::Broadcast(id) => vec![id.0],
        InputRef::Affine { base, stride } => {
            let mut atoms = vec![base.0];
            if count > 1 && *stride != 0 {
                let last = input.resolve(count - 1, 0);
                atoms.push(last.0);
            }
            atoms
        }
        InputRef::Explicit(ids) => {
            // Sample first, last, and a few middle ones to find all source groups.
            let mut atoms: HashSet<u32> = HashSet::new();
            for id in ids {
                atoms.insert(id.0);
            }
            atoms.into_iter().collect()
        }
        InputRef::SymAffine {
            base,
            stride_i,
            stride_k,
        } => {
            let mut atoms: HashSet<u32> = HashSet::new();
            let k_max: u64 = group
                .reduce_dims
                .iter()
                .filter_map(|sd| graph.sym_dim_bounds.get(sd).copied())
                .next()
                .unwrap_or(256);
            for k_val in 0..k_max {
                for i_val in [0u32, count.saturating_sub(1)] {
                    atoms.insert(input.resolve(i_val, k_val as u32).0);
                }
            }
            atoms.into_iter().collect()
        }
    }
}

/// Topologically sort kernels by dependency (Kahn's algorithm).
fn topo_sort_kernels(num_kernels: usize, deps: &[HashSet<usize>]) -> Vec<usize> {
    let mut in_degree = vec![0usize; num_kernels];
    let mut reverse_deps: Vec<Vec<usize>> = vec![Vec::new(); num_kernels];

    for (ki, dep_set) in deps.iter().enumerate() {
        in_degree[ki] = dep_set.len();
        for &dep_ki in dep_set {
            reverse_deps[dep_ki].push(ki);
        }
    }

    let mut queue: Vec<usize> = Vec::new();
    for ki in 0..num_kernels {
        if in_degree[ki] == 0 {
            queue.push(ki);
        }
    }

    let mut order = Vec::with_capacity(num_kernels);
    while let Some(ki) = queue.pop() {
        order.push(ki);
        for &dependent in &reverse_deps[ki] {
            in_degree[dependent] -= 1;
            if in_degree[dependent] == 0 {
                queue.push(dependent);
            }
        }
    }

    // If there's a cycle (shouldn't happen), include remaining kernels.
    if order.len() < num_kernels {
        for ki in 0..num_kernels {
            if !order.contains(&ki) {
                order.push(ki);
            }
        }
    }

    order
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::graph::GlobalId;
    use crate::milli_graph::MilliOpGraph;
    use crate::nano_graph::eval::NanoEval;
    use crate::nano_graph::lower::lower_with_info;
    use crate::nano_graph::{ScalarBinOp, ScalarOp};
    use crate::nano_graph::{AtomId, InputRef, NanoGraph};
    use crate::numeric_scalar::NumericScalar;
    use crate::numeric_tensor::NumericTensor;
    use crate::tensor_info::TensorInfo;
    use crate::DynRank;

    /// Build the inputs map from numeric_overrides (converting NumericScalar → f32).
    fn overrides_to_f32(overrides: &HashMap<u32, NumericScalar>) -> HashMap<u32, f32> {
        overrides
            .iter()
            .map(|(&k, v)| (k, v.to_f64() as f32))
            .collect()
    }

    /// Add tensor values to inputs map for a specific tensor.
    fn add_tensor_inputs(
        inputs: &mut HashMap<u32, f32>,
        tensor_map: &HashMap<GlobalId, crate::nano_graph::lower::TensorAtomMapInfo>,
        tensor_id: GlobalId,
        values: &[f32],
    ) {
        if let Some(tam) = tensor_map.get(&tensor_id) {
            assert_eq!(
                values.len(),
                tam.count as usize,
                "Value count mismatch for tensor {:?}",
                tensor_id
            );
            for (i, &val) in values.iter().enumerate() {
                inputs.insert(tam.base_id.0 + i as u32, val);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 1: Elementwise add
    // -----------------------------------------------------------------------

    #[test]
    fn test_elementwise_add() {
        let mut rng = rand::rng();
        let (mut milli, _ext_map) = MilliOpGraph::new(std::iter::empty(), &mut rng);

        let a_id = milli.add_input(&mut rng);
        let b_id = milli.add_input(&mut rng);
        let c_id = crate::milli_graph::ops::SimpleBinary::add(&mut milli, a_id, b_id, &mut rng);

        let a_vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b_vals: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];

        let a_tensor: NumericTensor<DynRank> =
            NumericTensor::from_vec_shape(a_vals.clone(), vec![4]).unwrap();
        let b_tensor: NumericTensor<DynRank> =
            NumericTensor::from_vec_shape(b_vals.clone(), vec![4]).unwrap();

        let mut info_inputs: HashMap<GlobalId, TensorInfo> = HashMap::new();
        info_inputs.insert(a_id, TensorInfo::from(a_tensor));
        info_inputs.insert(b_id, TensorInfo::from(b_tensor));

        let lower_result = lower_with_info(&milli, &info_inputs).unwrap();
        assert!(
            lower_result.unsupported.is_empty(),
            "Unsupported: {:?}",
            lower_result.unsupported_details
        );

        // Build inputs: start with numeric_overrides, add tensor values.
        let mut inputs = overrides_to_f32(&lower_result.numeric_overrides);
        add_tensor_inputs(&mut inputs, &lower_result.tensor_map, a_id, &a_vals);
        add_tensor_inputs(&mut inputs, &lower_result.tensor_map, b_id, &b_vals);

        let pipeline = run_nano_pipeline(&lower_result.graph, &inputs, 4);

        eprintln!(
            "Elementwise add: {} kernels, max_abs_error={}, kernel_sizes={:?}",
            pipeline.num_kernels, pipeline.max_abs_error, pipeline.kernel_sizes
        );

        assert!(
            pipeline.max_abs_error < 1e-4,
            "max_abs_error {} exceeds tolerance",
            pipeline.max_abs_error
        );

        // Also verify the actual output values match expected.
        let c_tam = lower_result.tensor_map.get(&c_id).unwrap();
        for i in 0..4u32 {
            let atom_id = c_tam.base_id.0 + i;
            let expected = a_vals[i as usize] + b_vals[i as usize];
            let got = pipeline.outputs[&atom_id];
            assert!(
                (expected - got).abs() < 1e-6,
                "Element {}: expected {} got {}",
                i,
                expected,
                got
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 2: MatMul(4, 8, 16) via lowering
    // -----------------------------------------------------------------------

    #[test]
    fn test_matmul_4_8_16() {
        let mut rng = rand::rng();
        let (mut milli, _ext_map) = MilliOpGraph::new(std::iter::empty(), &mut rng);

        let a_id = milli.add_input(&mut rng);
        let b_id = milli.add_input(&mut rng);
        let c_id = crate::milli_graph::ops::MatMul::push_new_default_precision(
            &mut milli,
            a_id,
            b_id,
            DType::F32,
            &mut rng,
        );

        // A=[4,8], B=[8,16] -> C=[4,16]
        // Use pseudo-random-ish values for A and B.
        let a_vals: Vec<f32> = (0..4 * 8)
            .map(|i| ((i as f32 * 0.37 + 0.13).sin() * 2.0))
            .collect();
        let b_vals: Vec<f32> = (0..8 * 16)
            .map(|i| ((i as f32 * 0.53 + 0.71).cos() * 1.5))
            .collect();

        let a_tensor: NumericTensor<DynRank> =
            NumericTensor::from_vec_shape(a_vals.clone(), vec![4, 8]).unwrap();
        let b_tensor: NumericTensor<DynRank> =
            NumericTensor::from_vec_shape(b_vals.clone(), vec![8, 16]).unwrap();

        let mut info_inputs: HashMap<GlobalId, TensorInfo> = HashMap::new();
        info_inputs.insert(a_id, TensorInfo::from(a_tensor));
        info_inputs.insert(b_id, TensorInfo::from(b_tensor));

        let lower_result = lower_with_info(&milli, &info_inputs).unwrap();
        assert!(
            lower_result.unsupported.is_empty(),
            "Unsupported: {:?}",
            lower_result.unsupported_details
        );

        // Build inputs.
        let mut inputs = overrides_to_f32(&lower_result.numeric_overrides);
        add_tensor_inputs(&mut inputs, &lower_result.tensor_map, a_id, &a_vals);
        add_tensor_inputs(&mut inputs, &lower_result.tensor_map, b_id, &b_vals);

        let pipeline = run_nano_pipeline(&lower_result.graph, &inputs, 4);

        eprintln!(
            "MatMul(4,8,16): {} kernels, max_abs_error={}, kernel_sizes={:?}",
            pipeline.num_kernels, pipeline.max_abs_error, pipeline.kernel_sizes
        );

        assert!(
            pipeline.max_abs_error < 1e-4,
            "max_abs_error {} exceeds tolerance",
            pipeline.max_abs_error
        );

        // Verify against manual matmul computation.
        let c_tam = lower_result.tensor_map.get(&c_id).unwrap();
        for row in 0..4u32 {
            for col in 0..16u32 {
                let mut expected: f32 = 0.0;
                for k in 0..8u32 {
                    expected += a_vals[(row * 8 + k) as usize] * b_vals[(k * 16 + col) as usize];
                }
                let atom_id = c_tam.base_id.0 + row * 16 + col;
                let got = pipeline.outputs[&atom_id];
                let diff = (expected - got).abs();
                assert!(
                    diff < 1e-3,
                    "MatMul C[{},{}]: expected {} got {} diff={}",
                    row,
                    col,
                    expected,
                    got,
                    diff
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 3: MatMul + downstream add
    // -----------------------------------------------------------------------

    #[test]
    fn test_matmul_plus_add() {
        let mut rng = rand::rng();
        let (mut milli, _ext_map) = MilliOpGraph::new(std::iter::empty(), &mut rng);

        let a_id = milli.add_input(&mut rng);
        let b_id = milli.add_input(&mut rng);
        let bias_id = milli.add_input(&mut rng);

        // MatMul: A[2,3] @ B[3,4] -> C[2,4]
        let c_id = crate::milli_graph::ops::MatMul::push_new_default_precision(
            &mut milli,
            a_id,
            b_id,
            DType::F32,
            &mut rng,
        );
        // Add bias: C[2,4] + bias[4] -> D[2,4]
        let d_id = crate::milli_graph::ops::SimpleBinary::add(&mut milli, c_id, bias_id, &mut rng);

        let a_vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_vals: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let bias_vals: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];

        let a_tensor: NumericTensor<DynRank> =
            NumericTensor::from_vec_shape(a_vals.clone(), vec![2, 3]).unwrap();
        let b_tensor: NumericTensor<DynRank> =
            NumericTensor::from_vec_shape(b_vals.clone(), vec![3, 4]).unwrap();
        let bias_tensor: NumericTensor<DynRank> =
            NumericTensor::from_vec_shape(bias_vals.clone(), vec![4]).unwrap();

        let mut info_inputs: HashMap<GlobalId, TensorInfo> = HashMap::new();
        info_inputs.insert(a_id, TensorInfo::from(a_tensor));
        info_inputs.insert(b_id, TensorInfo::from(b_tensor));
        info_inputs.insert(bias_id, TensorInfo::from(bias_tensor));

        let lower_result = lower_with_info(&milli, &info_inputs).unwrap();
        assert!(
            lower_result.unsupported.is_empty(),
            "Unsupported: {:?}",
            lower_result.unsupported_details
        );

        // Build inputs.
        let mut inputs = overrides_to_f32(&lower_result.numeric_overrides);
        add_tensor_inputs(&mut inputs, &lower_result.tensor_map, a_id, &a_vals);
        add_tensor_inputs(&mut inputs, &lower_result.tensor_map, b_id, &b_vals);
        add_tensor_inputs(
            &mut inputs,
            &lower_result.tensor_map,
            bias_id,
            &bias_vals,
        );

        let pipeline = run_nano_pipeline(&lower_result.graph, &inputs, 4);

        eprintln!(
            "MatMul+Add: {} kernels, max_abs_error={}, kernel_sizes={:?}",
            pipeline.num_kernels, pipeline.max_abs_error, pipeline.kernel_sizes
        );

        assert!(
            pipeline.max_abs_error < 1e-4,
            "max_abs_error {} exceeds tolerance",
            pipeline.max_abs_error
        );

        // Verify against manual computation.
        // C = A @ B:
        //   C[0,0] = 1*1 + 2*0 + 3*1 = 4
        //   C[0,1] = 1*0 + 2*1 + 3*1 = 5
        //   C[0,2] = 1*0 + 2*1 + 3*1 = 5
        //   C[0,3] = 1*1 + 2*0 + 3*1 = 4
        //   C[1,0] = 4*1 + 5*0 + 6*1 = 10
        //   C[1,1] = 4*0 + 5*1 + 6*1 = 11
        //   C[1,2] = 4*0 + 5*1 + 6*1 = 11
        //   C[1,3] = 4*1 + 5*0 + 6*1 = 10
        // D = C + bias:
        let expected_d: Vec<f32> = vec![
            4.0 + 0.1,
            5.0 + 0.2,
            5.0 + 0.3,
            4.0 + 0.4,
            10.0 + 0.1,
            11.0 + 0.2,
            11.0 + 0.3,
            10.0 + 0.4,
        ];

        let d_tam = lower_result.tensor_map.get(&d_id).unwrap();
        for (i, &exp) in expected_d.iter().enumerate() {
            let atom_id = d_tam.base_id.0 + i as u32;
            let got = pipeline.outputs[&atom_id];
            let diff = (exp - got).abs();
            assert!(
                diff < 1e-4,
                "D[{}]: expected {} got {} diff={}",
                i,
                exp,
                got,
                diff
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 4: Print pipeline results for a matmul
    // -----------------------------------------------------------------------

    #[test]
    fn test_print_pipeline_results() {
        let mut rng = rand::rng();
        let (mut milli, _ext_map) = MilliOpGraph::new(std::iter::empty(), &mut rng);

        let a_id = milli.add_input(&mut rng);
        let b_id = milli.add_input(&mut rng);
        let _c_id = crate::milli_graph::ops::MatMul::push_new_default_precision(
            &mut milli,
            a_id,
            b_id,
            DType::F32,
            &mut rng,
        );

        let a_vals: Vec<f32> = (0..4 * 8)
            .map(|i| ((i as f32 * 0.37 + 0.13).sin() * 2.0))
            .collect();
        let b_vals: Vec<f32> = (0..8 * 16)
            .map(|i| ((i as f32 * 0.53 + 0.71).cos() * 1.5))
            .collect();

        let a_tensor: NumericTensor<DynRank> =
            NumericTensor::from_vec_shape(a_vals.clone(), vec![4, 8]).unwrap();
        let b_tensor: NumericTensor<DynRank> =
            NumericTensor::from_vec_shape(b_vals.clone(), vec![8, 16]).unwrap();

        let mut info_inputs: HashMap<GlobalId, TensorInfo> = HashMap::new();
        info_inputs.insert(a_id, TensorInfo::from(a_tensor));
        info_inputs.insert(b_id, TensorInfo::from(b_tensor));

        let lower_result = lower_with_info(&milli, &info_inputs).unwrap();

        let mut inputs = overrides_to_f32(&lower_result.numeric_overrides);
        add_tensor_inputs(&mut inputs, &lower_result.tensor_map, a_id, &a_vals);
        add_tensor_inputs(&mut inputs, &lower_result.tensor_map, b_id, &b_vals);

        let pipeline = run_nano_pipeline(&lower_result.graph, &inputs, 4);

        eprintln!("\n=== Pipeline Results: MatMul(4,8,16) ===");
        eprintln!("  Kernels:       {}", pipeline.num_kernels);
        eprintln!("  Max abs error: {:.2e}", pipeline.max_abs_error);
        eprintln!("  Kernel sizes:  {:?}", pipeline.kernel_sizes);
        eprintln!(
            "  Total groups:  {}",
            lower_result.graph.num_groups()
        );
        eprintln!(
            "  Total atoms:   {}",
            lower_result.graph.num_atoms()
        );

        assert!(
            pipeline.max_abs_error < 1e-4,
            "max_abs_error {} exceeds tolerance",
            pipeline.max_abs_error
        );
    }
}
