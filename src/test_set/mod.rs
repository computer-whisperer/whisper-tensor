//! Test dataset infrastructure for verifying MilliOpGraph evaluation.
//!
//! Test data is constructed using the new [`NumericTensor`](crate::numeric_tensor::NumericTensor)
//! types with [`SystemPool`](crate::pool::SystemPool). At the eval boundary,
//! tensors are converted to legacy types for the current MilliOpGraph interpreter.
//!
//! Behind `#[cfg(feature = "tch")]`, data sets can be validated against PyTorch
//! via the tch crate as an independent oracle.

pub mod cast;
pub mod composite;
pub mod conv;
pub mod elementwise;
pub mod matmul;
pub mod opaque_ops;
pub mod pad;
pub mod reduce;
pub mod structural;

use std::collections::HashMap;

use crate::graph::GlobalId;
use crate::milli_graph::MilliOpGraph;
use crate::numeric_dtype::NumericDType;
use crate::numeric_scalar::NumericScalar;
use crate::numeric_tensor::{NumericTensor, NumericTensorView};
use crate::pool::SystemPool;
use crate::tensor_rank::DynRank;

/// Shared static SystemPool for all test/bridge allocations.
/// SystemPool is stateless (unit struct) — a static ref is always valid.
pub static POOL: SystemPool = SystemPool;

/// Type alias for test tensors (SystemPool, 'static lifetime).
pub type TestTensor = NumericTensor<'static, DynRank, SystemPool>;

/// A complete test case: one graph, multiple input/output data sets.
pub struct TestCase {
    pub name: String,
    pub graph: MilliOpGraph,
    pub data_sets: Vec<TestDataSet>,
}

/// One input/output pair for a test case.
pub struct TestDataSet {
    pub label: String,
    pub inputs: HashMap<GlobalId, TestTensor>,
    pub expected_outputs: HashMap<GlobalId, TestTensor>,
    pub tolerance: Tolerance,
}

/// Tolerance for comparing tensor values.
#[derive(Clone, Debug)]
pub struct Tolerance {
    pub atol: f64,
    pub rtol: f64,
}

impl Tolerance {
    pub fn for_dtype(dtype: NumericDType) -> Self {
        match dtype {
            NumericDType::F64 => Tolerance { atol: 1e-10, rtol: 1e-10 },
            NumericDType::F32 => Tolerance { atol: 1e-5, rtol: 1.3e-6 },
            NumericDType::F16 => Tolerance { atol: 1e-3, rtol: 4e-3 },
            NumericDType::BF16 => Tolerance { atol: 1e-2, rtol: 1.6e-2 },
            NumericDType::I64 | NumericDType::I32 | NumericDType::I16 | NumericDType::I8
            | NumericDType::U64 | NumericDType::U32 | NumericDType::U16 | NumericDType::U8
            | NumericDType::BOOL => Tolerance { atol: 0.0, rtol: 0.0 },
            _ => Tolerance { atol: 1e-3, rtol: 1e-3 }, // conservative default for exotic types
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor construction helpers
// ---------------------------------------------------------------------------

/// Create a test tensor from f32 values with an explicit shape.
pub fn tensor_f32_shaped(shape: Vec<u64>, values: &[f32]) -> TestTensor {
    assert_eq!(
        shape.iter().product::<u64>() as usize,
        values.len(),
        "shape product must match values length"
    );
    let mut t = NumericTensor::zeros(shape, NumericDType::F32, &POOL).unwrap();
    for (i, &v) in values.iter().enumerate() {
        t.write_element(i, NumericScalar::from_f32(v));
    }
    t
}

/// Create a 1D test tensor from f32 values.
pub fn tensor_f32(values: &[f32]) -> TestTensor {
    tensor_f32_shaped(vec![values.len() as u64], values)
}

/// Create a test tensor from f64 values with an explicit shape.
pub fn tensor_f64_shaped(shape: Vec<u64>, values: &[f64]) -> TestTensor {
    assert_eq!(shape.iter().product::<u64>() as usize, values.len());
    let mut t = NumericTensor::zeros(shape, NumericDType::F64, &POOL).unwrap();
    for (i, &v) in values.iter().enumerate() {
        t.write_element(i, NumericScalar::from_f64(v));
    }
    t
}

/// Create a test tensor from bf16 values with an explicit shape.
pub fn tensor_bf16_shaped(shape: Vec<u64>, values: &[half::bf16]) -> TestTensor {
    assert_eq!(shape.iter().product::<u64>() as usize, values.len());
    let mut t = NumericTensor::zeros(shape, NumericDType::BF16, &POOL).unwrap();
    for (i, &v) in values.iter().enumerate() {
        t.write_element(i, NumericScalar::from_bf16(v));
    }
    t
}

/// Create a 1D test tensor from bf16 values.
pub fn tensor_bf16(values: &[half::bf16]) -> TestTensor {
    tensor_bf16_shaped(vec![values.len() as u64], values)
}

/// Create a test tensor from f16 values with an explicit shape.
pub fn tensor_f16_shaped(shape: Vec<u64>, values: &[half::f16]) -> TestTensor {
    assert_eq!(shape.iter().product::<u64>() as usize, values.len());
    let mut t = NumericTensor::zeros(shape, NumericDType::F16, &POOL).unwrap();
    for (i, &v) in values.iter().enumerate() {
        t.write_element(i, NumericScalar::from_f16(v));
    }
    t
}

/// Create a 1D test tensor from f16 values.
pub fn tensor_f16(values: &[half::f16]) -> TestTensor {
    tensor_f16_shaped(vec![values.len() as u64], values)
}

/// Create a test tensor from i32 values with an explicit shape.
pub fn tensor_i32_shaped(shape: Vec<u64>, values: &[i32]) -> TestTensor {
    assert_eq!(shape.iter().product::<u64>() as usize, values.len());
    let mut t = NumericTensor::zeros(shape, NumericDType::I32, &POOL).unwrap();
    for (i, &v) in values.iter().enumerate() {
        t.write_element(i, NumericScalar::from_i32(v));
    }
    t
}

/// Create a 1D test tensor from i32 values.
pub fn tensor_i32(values: &[i32]) -> TestTensor {
    tensor_i32_shaped(vec![values.len() as u64], values)
}

/// Create a test tensor from i64 values with an explicit shape.
pub fn tensor_i64_shaped(shape: Vec<u64>, values: &[i64]) -> TestTensor {
    assert_eq!(shape.iter().product::<u64>() as usize, values.len());
    let mut t = NumericTensor::zeros(shape, NumericDType::I64, &POOL).unwrap();
    for (i, &v) in values.iter().enumerate() {
        t.write_element(i, NumericScalar::from_i64(v));
    }
    t
}

/// Create a 1D test tensor from i64 values.
pub fn tensor_i64(values: &[i64]) -> TestTensor {
    tensor_i64_shaped(vec![values.len() as u64], values)
}

/// Create a 1D test tensor of BOOL values.
pub fn tensor_bool(values: &[bool]) -> TestTensor {
    tensor_bool_shaped(vec![values.len() as u64], values)
}

/// Create a test tensor of BOOL values with an explicit shape.
pub fn tensor_bool_shaped(shape: Vec<u64>, values: &[bool]) -> TestTensor {
    assert_eq!(shape.iter().product::<u64>() as usize, values.len());
    let mut t = NumericTensor::zeros(shape, NumericDType::BOOL, &POOL).unwrap();
    for (i, &v) in values.iter().enumerate() {
        t.write_element(i, NumericScalar::from_bool(v));
    }
    t
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

/// Compare two new-type tensors element-wise within tolerance.
pub fn assert_tensors_close(
    actual: &NumericTensorView<'_, DynRank>,
    expected: &NumericTensorView<'_, DynRank>,
    tolerance: &Tolerance,
    context: &str,
) -> Result<(), String> {
    if actual.shape() != expected.shape() {
        return Err(format!(
            "{context}: shape mismatch: actual {:?} vs expected {:?}",
            actual.shape(), expected.shape()
        ));
    }
    if actual.dtype() != expected.dtype() {
        return Err(format!(
            "{context}: dtype mismatch: actual {:?} vs expected {:?}",
            actual.dtype(), expected.dtype()
        ));
    }

    let numel = actual.numel();
    for i in 0..numel {
        let a = actual.read_element(i).to_f64();
        let e = expected.read_element(i).to_f64();
        if a.is_nan() && e.is_nan() { continue; }
        if a.is_infinite() && e.is_infinite() && a.signum() == e.signum() { continue; }
        let err = (a - e).abs();
        let limit = tolerance.atol + tolerance.rtol * a.abs().max(e.abs());
        if err > limit {
            return Err(format!(
                "{context}: element [{i}] mismatch: actual={a}, expected={e}, \
                 err={err}, limit={limit} (atol={}, rtol={})",
                tolerance.atol, tolerance.rtol
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Test runners
// ---------------------------------------------------------------------------

use crate::migration::bridge;
use crate::migration::numeric_tensor::NumericTensor as LegacyNumericTensor;
use crate::DynRank as LegacyDynRank;

/// Collect all test cases from all submodules.
pub fn build_test_set() -> Vec<TestCase> {
    let mut cases = Vec::new();
    cases.extend(elementwise::build_cases());
    cases.extend(matmul::build_cases());
    cases.extend(cast::build_cases());
    cases.extend(reduce::build_cases());
    cases.extend(composite::build_cases());
    cases.extend(conv::build_cases());
    cases.extend(pad::build_cases());
    cases.extend(structural::build_cases());
    cases.extend(opaque_ops::build_cases());
    cases
}

/// Run all data sets of a test case through the MilliOpGraph interpreter.
pub fn run_case_via_milli_eval(case: &TestCase) -> Result<(), String> {
    use crate::backends::eval_backend::EvalBackend;

    let mut backend = EvalBackend::NDArray;

    for ds in &case.data_sets {
        let legacy_inputs: HashMap<GlobalId, LegacyNumericTensor<LegacyDynRank>> = ds
            .inputs
            .iter()
            .map(|(&id, t)| (id, bridge::view_to_legacy(&t.view())))
            .collect();

        let mut observer = ();
        let results: HashMap<GlobalId, LegacyNumericTensor<LegacyDynRank>> = case
            .graph
            .eval(&legacy_inputs, &mut observer, &mut backend)
            .map_err(|e| format!("{}[{}]: eval failed: {e}", case.name, ds.label))?
            .collect();

        for (&expected_id, expected_tensor) in &ds.expected_outputs {
            let legacy_actual = results.get(&expected_id).ok_or_else(|| {
                format!("{}[{}]: missing output {expected_id}", case.name, ds.label)
            })?;
            let actual = bridge::legacy_to_new(legacy_actual);
            let ctx = format!("{}[{}]", case.name, ds.label);
            assert_tensors_close(
                &actual.view(),
                &expected_tensor.view(),
                &ds.tolerance,
                &ctx,
            )?;
        }
    }
    Ok(())
}

/// Run all data sets of a test case through the pool-based nano eval.
///
/// Flow: MilliOpGraph → lower_to_nano → pool_eval → compare
pub fn run_case_via_pool_eval(case: &TestCase) -> Result<(), String> {
    use crate::nano_graph::lower;
    use crate::nano_graph::pattern::AtomRange;
    use crate::nano_graph::pool_eval;
    use crate::pool::TrackedPool;
    use crate::tensor_info::TensorInfo;

    for ds in &case.data_sets {
        // Build TensorInfo for each input (needed by lower).
        let legacy_inputs: HashMap<GlobalId, LegacyNumericTensor<LegacyDynRank>> = ds
            .inputs
            .iter()
            .map(|(&id, t)| (id, bridge::view_to_legacy(&t.view())))
            .collect();

        let info_inputs: HashMap<GlobalId, TensorInfo<'_, crate::pool::SystemPool>> = legacy_inputs
            .iter()
            .map(|(&id, t)| (id, TensorInfo::from_legacy(t, &crate::pool::SystemPool)))
            .collect();

        // Lower MilliOpGraph → NanoGraph.
        let lower_result = lower::lower(&case.graph, &info_inputs)
            .map_err(|e| format!("{}[{}]: lower failed: {e}", case.name, ds.label))?;

        if !lower_result.unsupported.is_empty() {
            return Err(format!(
                "{}[{}]: unsupported ops: {:?}",
                case.name, ds.label, lower_result.unsupported_details
            ));
        }

        // Map input tensors to (AtomId, view) pairs for pool_eval.
        // Collect (AtomId, tensor) pairs, then create views with stable references.
        let input_pairs: Vec<_> = ds.inputs.iter().filter_map(|(&ext_id, tensor)| {
            let internal_id = case.graph.input_map.get(&ext_id)?;
            let tam = lower_result.tensor_map.get(internal_id)?;
            Some((tam.base_id, tensor))
        }).collect();

        let input_views: Vec<_> = input_pairs
            .iter()
            .map(|(_, tensor)| tensor.view())
            .collect();

        let eval_inputs: Vec<_> = input_pairs
            .iter()
            .zip(input_views.iter())
            .map(|((atom_id, _), view)| (*atom_id, view))
            .collect();

        // Build output AtomRanges from the graph's output ids.
        let output_ids: Vec<GlobalId> = case.graph.output_ordering
            .as_ref()
            .map(|v| v.clone())
            .unwrap_or_default();

        let mut output_ranges: Vec<AtomRange> = Vec::new();
        for &out_id in &output_ids {
            if let Some(tam) = lower_result.tensor_map.get(&out_id) {
                let mut seen = std::collections::HashSet::new();
                for i in 0..tam.count {
                    let atom = tam.atom_id_for_element(i);
                    if let Some(gi) = lower_result.graph.find_group_idx(atom) {
                        if seen.insert(gi) {
                            let g = &lower_result.graph.groups()[gi];
                            output_ranges.push(AtomRange {
                                base: g.base_id,
                                count: g.count,
                                dtype: g.output_dtype,
                            });
                        }
                    }
                }
            }
        }

        // Run pool_eval.
        let pool = TrackedPool::new(None); // no budget limit for tests
        let eval_results = pool_eval::pool_eval(
            &lower_result.graph,
            &eval_inputs,
            &output_ranges,
            &pool,
        ).map_err(|e| format!("{}[{}]: pool_eval failed: {e}", case.name, ds.label))?;

        // Compare outputs.
        for (&expected_id, expected_tensor) in &ds.expected_outputs {
            // Fail loudly if the output isn't in tensor_map (don't silently skip).
            let tam = lower_result.tensor_map.get(&expected_id).ok_or_else(|| {
                format!(
                    "{}[{}]: output {expected_id} not in tensor_map — lowering may have failed",
                    case.name, ds.label
                )
            })?;

            let numel = tam.count as usize;
            let expected_dtype = expected_tensor.dtype();
            let mut actual = NumericTensor::zeros(
                expected_tensor.shape().clone(),
                expected_dtype,
                &POOL,
            ).unwrap();

            // For each logical element, find its atom in the eval results.
            for i in 0..numel {
                let atom = tam.atom_id_for_element(i as u64);
                let (rt_idx, offset) = output_ranges
                    .iter()
                    .enumerate()
                    .find_map(|(idx, range)| {
                        let range_end = range.base.0 + range.count;
                        if atom.0 >= range.base.0 && atom.0 < range_end {
                            Some((idx, (atom.0 - range.base.0) as usize))
                        } else {
                            None
                        }
                    })
                    .ok_or_else(|| {
                        format!("{}[{}]: atom {} not in any output range", case.name, ds.label, atom)
                    })?;

                let scalar = eval_results[rt_idx].read_element(offset);
                actual.write_element(i, scalar.cast_to(expected_dtype));
            }

            let ctx = format!("{}[{}] pool_eval", case.name, ds.label);
            assert_tensors_close(&actual.view(), &expected_tensor.view(), &ds.tolerance, &ctx)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_cases_via_milli_eval() {
        let cases = build_test_set();
        assert!(!cases.is_empty(), "test set should not be empty");
        let mut total_data_sets = 0;
        for case in &cases {
            run_case_via_milli_eval(case).unwrap_or_else(|e| panic!("{e}"));
            total_data_sets += case.data_sets.len();
        }
        eprintln!(
            "{} test cases ({total_data_sets} data sets) passed via milli eval",
            cases.len()
        );
    }

    #[test]
    fn test_all_cases_via_pool_eval() {
        let cases = build_test_set();
        assert!(!cases.is_empty(), "test set should not be empty");
        let mut total_data_sets = 0;
        for case in &cases {
            run_case_via_pool_eval(case).unwrap_or_else(|e| panic!("{e}"));
            total_data_sets += case.data_sets.len();
        }
        eprintln!(
            "{} test cases ({total_data_sets} data sets) passed via pool eval",
            cases.len()
        );
    }
}
