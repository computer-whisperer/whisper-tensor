//! Test dataset infrastructure for verifying MilliOpGraph evaluation.
//!
//! Each [`TestCase`] pairs a [`MilliOpGraph`] with one or more [`TestDataSet`]s
//! (input/output tensor pairs). The same graph is tested with each data set,
//! exercising different value ranges, edge cases, and dtypes.
//!
//! Behind `#[cfg(feature = "tch")]`, data sets can be validated against PyTorch
//! via the tch crate as an independent oracle.

pub mod elementwise;

use std::collections::HashMap;

use crate::dtype::DType;
use crate::graph::GlobalId;
use crate::migration::numeric_tensor::NumericTensor;
use crate::milli_graph::MilliOpGraph;
use crate::DynRank;

/// A complete test case: one graph, multiple input/output data sets.
pub struct TestCase {
    /// Human-readable name for diagnostics.
    pub name: String,
    /// The MilliOpGraph describing the operation under test.
    pub graph: MilliOpGraph,
    /// One or more input/output sets to test the graph with.
    pub data_sets: Vec<TestDataSet>,
}

/// One input/output pair for a test case.
pub struct TestDataSet {
    /// Label for this data set (e.g. "normal", "edge_cases", "zeros").
    pub label: String,
    /// Input tensors keyed by their external (pre-mapping) GlobalId.
    pub inputs: HashMap<GlobalId, NumericTensor<DynRank>>,
    /// Expected output tensors keyed by their internal output GlobalId
    /// (as returned by `set_outputs`).
    pub expected_outputs: HashMap<GlobalId, NumericTensor<DynRank>>,
    /// Comparison tolerance.
    pub tolerance: Tolerance,
}

/// Tolerance for comparing tensor values.
#[derive(Clone, Debug)]
pub struct Tolerance {
    pub atol: f64,
    pub rtol: f64,
}

impl Tolerance {
    pub fn for_dtype(dtype: DType) -> Self {
        match dtype {
            DType::F64 => Tolerance { atol: 1e-10, rtol: 1e-10 },
            DType::F32 => Tolerance { atol: 1e-5, rtol: 1.3e-6 },
            DType::F16 => Tolerance { atol: 1e-3, rtol: 4e-3 },
            DType::BF16 => Tolerance { atol: 1e-2, rtol: 1.6e-2 },
            _ => Tolerance { atol: 0.0, rtol: 0.0 },
        }
    }

    pub fn exact() -> Self {
        Tolerance { atol: 0.0, rtol: 0.0 }
    }
}

/// Compare two tensors element-wise within tolerance.
pub fn assert_tensors_close(
    actual: &NumericTensor<DynRank>,
    expected: &NumericTensor<DynRank>,
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

    let actual_nd = actual.to_ndarray().unwrap().cast(DType::F64).unwrap()
        .flatten().try_to_vec::<f64>().unwrap();
    let expected_nd = expected.to_ndarray().unwrap().cast(DType::F64).unwrap()
        .flatten().try_to_vec::<f64>().unwrap();

    for (i, (&a, &e)) in actual_nd.iter().zip(expected_nd.iter()).enumerate() {
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
        let mut observer = ();
        let results: HashMap<GlobalId, NumericTensor<DynRank>> = case
            .graph
            .eval(&ds.inputs, &mut observer, &mut backend)
            .map_err(|e| format!("{}[{}]: eval failed: {e}", case.name, ds.label))?
            .collect();

        for (&expected_id, expected_tensor) in &ds.expected_outputs {
            let actual = results.get(&expected_id).ok_or_else(|| {
                format!("{}[{}]: missing output {expected_id}", case.name, ds.label)
            })?;
            let ctx = format!("{}[{}]", case.name, ds.label);
            assert_tensors_close(actual, expected_tensor, &ds.tolerance, &ctx)?;
        }
    }
    Ok(())
}

/// Run all data sets of a test case through PyTorch (tch) as an independent oracle.
/// Validates that our expected_outputs match what PyTorch produces.
#[cfg(feature = "tch")]
pub fn validate_case_against_tch(case: &TestCase) -> Result<(), String> {
    // Each submodule provides a tch_validate function if it can
    // For now, this is a placeholder that submodules opt into
    let _ = case;
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
