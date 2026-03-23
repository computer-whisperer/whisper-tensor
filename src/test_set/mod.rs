//! Test dataset infrastructure for verifying MilliOpGraph evaluation.
//!
//! Test data is constructed using the new [`NumericTensor`](crate::numeric_tensor::NumericTensor)
//! types with [`SystemPool`](crate::pool::SystemPool). At the eval boundary,
//! tensors are converted to legacy types for the current MilliOpGraph interpreter.
//!
//! Behind `#[cfg(feature = "tch")]`, data sets can be validated against PyTorch
//! via the tch crate as an independent oracle.

pub mod elementwise;

use std::collections::HashMap;

use crate::dtype::DType;
use crate::graph::GlobalId;
use crate::milli_graph::MilliOpGraph;
use crate::numeric_dtype::NumericDType;
use crate::numeric_scalar::NumericScalar;
use crate::numeric_tensor::{NumericTensor, NumericTensorView, TensorLayout};
use crate::pool::SystemPool;
use crate::tensor_rank::DynRank;
use crate::DynRank as LegacyDynRank;

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
    /// Input tensors keyed by their external (pre-mapping) GlobalId.
    pub inputs: HashMap<GlobalId, TestTensor>,
    /// Expected output tensors keyed by their internal output GlobalId.
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
            _ => Tolerance { atol: 0.0, rtol: 0.0 },
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor construction helpers
// ---------------------------------------------------------------------------

/// Static SystemPool instance for test tensors.
/// SystemPool is stateless (unit struct), so a static ref is fine.
static TEST_POOL: SystemPool = SystemPool;

/// Create a 1D test tensor from f32 values.
pub fn tensor_f32(values: &[f32]) -> TestTensor {
    let shape = vec![values.len() as u64];
    let mut t = NumericTensor::zeros(shape, NumericDType::F32, &TEST_POOL).unwrap();
    for (i, &v) in values.iter().enumerate() {
        t.write_element(i, NumericScalar::from_f32(v));
    }
    t
}

/// Create a 1D test tensor from bf16 values.
pub fn tensor_bf16(values: &[half::bf16]) -> TestTensor {
    let shape = vec![values.len() as u64];
    let mut t = NumericTensor::zeros(shape, NumericDType::BF16, &TEST_POOL).unwrap();
    for (i, &v) in values.iter().enumerate() {
        t.write_element(i, NumericScalar::from_bf16(v));
    }
    t
}

/// Create a 1D test tensor from f16 values.
pub fn tensor_f16(values: &[half::f16]) -> TestTensor {
    let shape = vec![values.len() as u64];
    let mut t = NumericTensor::zeros(shape, NumericDType::F16, &TEST_POOL).unwrap();
    for (i, &v) in values.iter().enumerate() {
        t.write_element(i, NumericScalar::from_f16(v));
    }
    t
}

// ---------------------------------------------------------------------------
// Comparison (using new types directly)
// ---------------------------------------------------------------------------

/// Compare two new-type tensors element-wise within tolerance.
pub fn assert_tensors_close(
    actual: &NumericTensorView<'_, DynRank>,
    expected: &NumericTensorView<'_, DynRank>,
    tolerance: &Tolerance,
    context: &str,
) -> Result<(), String> {
    let actual_shape = actual.shape();
    let expected_shape = expected.shape();
    if actual_shape != expected_shape {
        return Err(format!(
            "{context}: shape mismatch: actual {actual_shape:?} vs expected {expected_shape:?}"
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
// Legacy conversion bridge
// ---------------------------------------------------------------------------

use crate::migration::numeric_tensor::NumericTensor as LegacyNumericTensor;
use crate::backends::ndarray_backend::NDArrayNumericTensor;

/// Convert a new-type tensor view to a legacy NumericTensor for MilliOpGraph eval.
fn view_to_legacy(view: &NumericTensorView<'_, DynRank>) -> LegacyNumericTensor<LegacyDynRank> {
    let numel = view.numel();
    let shape: Vec<usize> = view.shape().iter().map(|&d| d as usize).collect();
    let dtype = view.dtype();

    match dtype {
        NumericDType::F32 => {
            let data: Vec<f32> = (0..numel).map(|i| {
                let s = view.read_element(i);
                f32::from_le_bytes([s.raw_bits()[0], s.raw_bits()[1], s.raw_bits()[2], s.raw_bits()[3]])
            }).collect();
            LegacyNumericTensor::NDArray(
                NDArrayNumericTensor::F32(ndarray::ArcArray::from_shape_vec(ndarray::IxDyn(&shape), data).unwrap())
            )
        }
        NumericDType::F64 => {
            let data: Vec<f64> = (0..numel).map(|i| {
                let s = view.read_element(i);
                f64::from_le_bytes(*s.raw_bits())
            }).collect();
            LegacyNumericTensor::NDArray(
                NDArrayNumericTensor::F64(ndarray::ArcArray::from_shape_vec(ndarray::IxDyn(&shape), data).unwrap())
            )
        }
        NumericDType::BF16 => {
            let data: Vec<half::bf16> = (0..numel).map(|i| {
                let s = view.read_element(i);
                half::bf16::from_le_bytes([s.raw_bits()[0], s.raw_bits()[1]])
            }).collect();
            LegacyNumericTensor::NDArray(
                NDArrayNumericTensor::BF16(ndarray::ArcArray::from_shape_vec(ndarray::IxDyn(&shape), data).unwrap())
            )
        }
        NumericDType::F16 => {
            let data: Vec<half::f16> = (0..numel).map(|i| {
                let s = view.read_element(i);
                half::f16::from_le_bytes([s.raw_bits()[0], s.raw_bits()[1]])
            }).collect();
            LegacyNumericTensor::NDArray(
                NDArrayNumericTensor::F16(ndarray::ArcArray::from_shape_vec(ndarray::IxDyn(&shape), data).unwrap())
            )
        }
        NumericDType::I32 => {
            let data: Vec<i32> = (0..numel).map(|i| {
                let s = view.read_element(i);
                i32::from_le_bytes([s.raw_bits()[0], s.raw_bits()[1], s.raw_bits()[2], s.raw_bits()[3]])
            }).collect();
            LegacyNumericTensor::NDArray(
                NDArrayNumericTensor::I32(ndarray::ArcArray::from_shape_vec(ndarray::IxDyn(&shape), data).unwrap())
            )
        }
        NumericDType::I64 => {
            let data: Vec<i64> = (0..numel).map(|i| {
                let s = view.read_element(i);
                i64::from_le_bytes(*s.raw_bits())
            }).collect();
            LegacyNumericTensor::NDArray(
                NDArrayNumericTensor::I64(ndarray::ArcArray::from_shape_vec(ndarray::IxDyn(&shape), data).unwrap())
            )
        }
        other => panic!("view_to_legacy: unsupported dtype {other}"),
    }
}

/// Convert a legacy NumericTensor back to a new-type TestTensor.
fn legacy_to_new(legacy: &LegacyNumericTensor<LegacyDynRank>) -> TestTensor {
    let legacy_dtype = legacy.dtype();
    let dtype = NumericDType::from_legacy(legacy_dtype).unwrap();
    let shape: Vec<u64> = legacy.shape().iter().map(|&d| d as u64).collect();

    let mut t = NumericTensor::zeros(shape, dtype, &TEST_POOL).unwrap();

    // Read elements from legacy via f64 cast, write to new tensor
    let nd = legacy.to_ndarray().unwrap().cast(DType::F64).unwrap()
        .flatten().try_to_vec::<f64>().unwrap();

    for (i, &v) in nd.iter().enumerate() {
        let scalar = NumericScalar::from_f64(v).cast_to(dtype);
        t.write_element(i, scalar);
    }
    t
}

// ---------------------------------------------------------------------------
// Test runners
// ---------------------------------------------------------------------------

/// Collect all test cases from all submodules.
pub fn build_test_set() -> Vec<TestCase> {
    let mut cases = Vec::new();
    cases.extend(elementwise::build_cases());
    cases
}

/// Run all data sets of a test case through the MilliOpGraph interpreter.
pub fn run_case_via_milli_eval(case: &TestCase) -> Result<(), String> {
    use crate::backends::eval_backend::EvalBackend;

    let mut backend = EvalBackend::NDArray;

    for ds in &case.data_sets {
        // Convert new-type inputs to legacy for the eval boundary
        let legacy_inputs: HashMap<GlobalId, LegacyNumericTensor<LegacyDynRank>> = ds
            .inputs
            .iter()
            .map(|(&id, t)| (id, view_to_legacy(&t.view())))
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

            // Convert legacy output back to new type for comparison
            let actual = legacy_to_new(legacy_actual);
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
}
