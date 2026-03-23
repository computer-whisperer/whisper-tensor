//! Test dataset infrastructure for verifying MilliOpGraph evaluation.
//!
//! Each test case is a pure-Rust construction of:
//! - A [`MilliOpGraph`] describing one operation
//! - Input tensors (deterministic, constructed in code)
//! - Expected output tensors
//! - Per-dtype tolerance for comparison
//!
//! The same test cases run through every evaluation mode (nano-op eval,
//! compiled eval, etc.). Adding a new eval mode means one new test function.
//! Adding a new op means one new entry in a submodule.
//!
//! Behind `#[cfg(feature = "tch")]`, test cases can also be validated against
//! PyTorch via the tch crate, confirming our expected values are correct.

pub mod elementwise;

use std::collections::HashMap;

use crate::dtype::DType;
use crate::graph::GlobalId;
use crate::migration::numeric_tensor::NumericTensor;
use crate::milli_graph::MilliOpGraph;
use crate::DynRank;

/// A complete test case: graph + inputs + expected outputs + tolerance.
pub struct TestCase {
    /// Human-readable name for diagnostics.
    pub name: String,
    /// The MilliOpGraph describing the operation under test.
    pub graph: MilliOpGraph,
    /// Input tensors keyed by their external (pre-mapping) GlobalId.
    pub inputs: HashMap<GlobalId, NumericTensor<DynRank>>,
    /// Expected output tensors keyed by their external (post-mapping) GlobalId.
    pub expected_outputs: HashMap<GlobalId, NumericTensor<DynRank>>,
    /// Comparison tolerance.
    pub tolerance: Tolerance,
}

/// Tolerance for comparing tensor values.
#[derive(Clone, Debug)]
pub struct Tolerance {
    /// Absolute tolerance.
    pub atol: f64,
    /// Relative tolerance.
    pub rtol: f64,
}

impl Tolerance {
    /// Default tolerance for a given dtype.
    pub fn for_dtype(dtype: DType) -> Self {
        match dtype {
            DType::F64 => Tolerance {
                atol: 1e-10,
                rtol: 1e-10,
            },
            DType::F32 => Tolerance {
                atol: 1e-5,
                rtol: 1.3e-6,
            },
            DType::F16 => Tolerance {
                atol: 1e-3,
                rtol: 4e-3,
            },
            DType::BF16 => Tolerance {
                atol: 1e-2,
                rtol: 1.6e-2,
            },
            _ => Tolerance {
                atol: 0.0,
                rtol: 0.0,
            }, // exact match for integers/bool
        }
    }
}

/// Compare two tensors element-wise within tolerance.
/// Returns Ok(()) or an error message describing the first mismatch.
pub fn assert_tensors_close(
    actual: &NumericTensor<DynRank>,
    expected: &NumericTensor<DynRank>,
    tolerance: &Tolerance,
    context: &str,
) -> Result<(), String> {
    if actual.shape() != expected.shape() {
        return Err(format!(
            "{context}: shape mismatch: actual {:?} vs expected {:?}",
            actual.shape(),
            expected.shape()
        ));
    }
    if actual.dtype() != expected.dtype() {
        return Err(format!(
            "{context}: dtype mismatch: actual {:?} vs expected {:?}",
            actual.dtype(),
            expected.dtype()
        ));
    }

    let actual_nd = actual
        .to_ndarray()
        .unwrap()
        .cast(DType::F64)
        .unwrap()
        .flatten()
        .try_to_vec::<f64>()
        .unwrap();
    let expected_nd = expected
        .to_ndarray()
        .unwrap()
        .cast(DType::F64)
        .unwrap()
        .flatten()
        .try_to_vec::<f64>()
        .unwrap();

    for (i, (&a, &e)) in actual_nd.iter().zip(expected_nd.iter()).enumerate() {
        if a.is_nan() && e.is_nan() {
            continue; // both NaN = match
        }
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

/// Run a test case through the MilliOpGraph interpreter and check outputs.
pub fn run_case_via_milli_eval(case: &TestCase) -> Result<(), String> {
    use crate::backends::eval_backend::EvalBackend;

    let mut backend = EvalBackend::NDArray;
    let mut observer = ();

    let results: HashMap<GlobalId, NumericTensor<DynRank>> = case
        .graph
        .eval(&case.inputs, &mut observer, &mut backend)
        .map_err(|e| format!("{}: eval failed: {e}", case.name))?
        .collect();

    for (&expected_id, expected_tensor) in &case.expected_outputs {
        let actual = results
            .get(&expected_id)
            .ok_or_else(|| format!("{}: missing output {expected_id}", case.name))?;
        assert_tensors_close(actual, expected_tensor, &case.tolerance, &case.name)?;
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
        let mut pass_count = 0;
        for case in &cases {
            run_case_via_milli_eval(case).unwrap_or_else(|e| panic!("{e}"));
            pass_count += 1;
        }
        eprintln!("{pass_count} test cases passed via milli eval");
    }
}
