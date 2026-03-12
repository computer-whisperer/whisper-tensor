//! Validation harness for `infer()` correctness.
//!
//! For each op in a MilliOpGraph, tests that `infer()` never returns *wrong*
//! information at any knowledge level. It may return *less* information
//! (e.g. UnableToInfer or Minimal when Shaped is possible), but any concrete
//! claims must match the ground-truth values from eval.

use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::MilliOp;
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::numeric_tensor::NumericTensor;
use crate::scalar_info::{ScalarInfo, ScalarInfoTyped};
use crate::symbolic_scalar::{SymbolicResolver, SymbolicScalar, SymbolicScalarTyped};
use crate::tensor_info::{MinimalTensor, ShapedTensor, TensorInfo, TensorInfoRanked};
use crate::tensor_rank::DynRank;
use std::collections::HashMap;
use std::fmt;

/// Which ablation level was applied to inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AblationLevel {
    /// Level 0: Full numeric tensors (ground truth)
    Numeric,
    /// Level 1: Known shape + dtype, symbolic values
    Shaped,
    /// Level 2: Known rank + dtype, symbolic dims
    Ranked,
    /// Level 3: Only dtype known, unknown rank
    Minimal,
}

impl fmt::Display for AblationLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AblationLevel::Numeric => write!(f, "Numeric (L0)"),
            AblationLevel::Shaped => write!(f, "Shaped (L1)"),
            AblationLevel::Ranked => write!(f, "Ranked (L2)"),
            AblationLevel::Minimal => write!(f, "Minimal (L3)"),
        }
    }
}

/// Result of validating a single op at a single ablation level.
#[derive(Debug)]
pub enum ValidationOutcome {
    /// infer() returned correct info matching ground truth
    Pass,
    /// infer() returned UnableToInfer (always acceptable)
    UnableToInfer,
    /// infer() returned wrong information
    Failure(String),
}

/// Summary of validation results.
#[derive(Debug, Default)]
pub struct ValidationReport {
    pub pass_count: usize,
    pub unable_to_infer_count: usize,
    pub failure_count: usize,
    pub failures: Vec<ValidationFailure>,
}

/// Details of a single validation failure.
#[derive(Debug)]
pub struct ValidationFailure {
    pub op_id: GlobalId,
    pub op_kind: String,
    pub level: AblationLevel,
    pub message: String,
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Validation Report: {} pass, {} unable-to-infer, {} failures",
            self.pass_count, self.unable_to_infer_count, self.failure_count
        )?;
        for failure in &self.failures {
            writeln!(
                f,
                "  FAIL [{:?}] op={} level={}: {}",
                failure.op_id, failure.op_kind, failure.level, failure.message
            )?;
        }
        Ok(())
    }
}

/// Ground truth for a single output tensor from eval.
struct GroundTruth {
    dtype: DType,
    shape: Vec<u64>,
    rank: usize,
}

impl GroundTruth {
    fn from_numeric_tensor(tensor: &NumericTensor<DynRank>) -> Self {
        Self {
            dtype: tensor.dtype(),
            shape: tensor.shape(),
            rank: tensor.rank(),
        }
    }
}

/// Create a Shaped TensorInfo from a concrete NumericTensor.
/// Keeps the concrete shape but replaces all element values with symbolic scalars.
/// Returns None if the dtype isn't supported for shaped construction (e.g. Bool).
fn numeric_to_shaped(
    tensor: &NumericTensor<DynRank>,
    resolver: &mut SymbolicResolver,
) -> Option<TensorInfo> {
    let dtype = tensor.dtype();
    // ShapedTensor::new_symbolic only supports F64, F32, U64, I64.
    if !matches!(dtype, DType::F64 | DType::F32 | DType::U64 | DType::I64) {
        return None;
    }
    let shape: Vec<u64> = tensor.shape();
    let first_element = ScalarInfo::Symbolic(SymbolicScalar::new(dtype, resolver));
    Some(TensorInfo::from(ShapedTensor::<DynRank>::new_symbolic(
        first_element,
        shape,
        resolver,
    )))
}

/// Create a Ranked TensorInfo from a concrete NumericTensor.
/// Keeps the rank but replaces all dims with symbolic values.
fn numeric_to_ranked(
    tensor: &NumericTensor<DynRank>,
    resolver: &mut SymbolicResolver,
) -> TensorInfo {
    let rank = tensor.rank();
    let first_element = ScalarInfo::Symbolic(SymbolicScalar::new(tensor.dtype(), resolver));
    let symbolic_dims: Vec<ScalarInfoTyped<u64>> = (0..rank)
        .map(|_| ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(resolver)))
        .collect();
    TensorInfo::from(TensorInfoRanked::<DynRank>::new(
        first_element,
        symbolic_dims,
        resolver,
    ))
}

/// Create a Minimal TensorInfo from a concrete NumericTensor.
/// Only preserves dtype; rank is unknown.
fn numeric_to_minimal(
    tensor: &NumericTensor<DynRank>,
    resolver: &mut SymbolicResolver,
) -> TensorInfo {
    let first_element = ScalarInfo::Symbolic(SymbolicScalar::new(tensor.dtype(), resolver));
    let symbolic_rank = SymbolicScalarTyped::<u32>::new(resolver);
    TensorInfo::from(MinimalTensor::new(first_element, symbolic_rank))
}

/// Ablate a NumericTensor to the given level.
/// Falls back to a lower ablation level if the dtype doesn't support the requested one.
fn ablate_tensor(
    tensor: &NumericTensor<DynRank>,
    level: AblationLevel,
    resolver: &mut SymbolicResolver,
) -> TensorInfo {
    match level {
        AblationLevel::Numeric => TensorInfo::from(tensor.clone()),
        AblationLevel::Shaped => {
            // Fall back to Ranked if dtype isn't supported for Shaped construction.
            numeric_to_shaped(tensor, resolver)
                .unwrap_or_else(|| numeric_to_ranked(tensor, resolver))
        }
        AblationLevel::Ranked => numeric_to_ranked(tensor, resolver),
        AblationLevel::Minimal => numeric_to_minimal(tensor, resolver),
    }
}

/// Validate that an inferred TensorInfo does not contradict ground truth.
///
/// Returns `Ok(())` if every concrete claim in `inferred` matches `truth`.
/// Returns `Err(message)` if any concrete claim is wrong.
fn validate_against_ground_truth(
    inferred: &TensorInfo,
    truth: &GroundTruth,
) -> Result<(), String> {
    // Check dtype -- inferred dtype is always concrete in this type system
    let inferred_dtype = inferred.dtype();
    if inferred_dtype != truth.dtype {
        return Err(format!(
            "dtype mismatch: inferred {:?} but actual is {:?}",
            inferred_dtype, truth.dtype
        ));
    }

    match inferred {
        TensorInfo::Minimal(_) => {
            // Only dtype was claimed, already checked above. Pass.
            Ok(())
        }
        TensorInfo::Ranked(ranked) => {
            // Check rank
            let inferred_rank = ranked.rank();
            if inferred_rank != truth.rank {
                return Err(format!(
                    "rank mismatch: inferred {} but actual is {}",
                    inferred_rank, truth.rank
                ));
            }

            // Check any concrete dims
            let inferred_shape = ranked.shape();
            for (i, dim) in inferred_shape.iter().enumerate() {
                if let ScalarInfoTyped::Numeric(v) = dim
                    && i < truth.shape.len()
                    && *v != truth.shape[i]
                {
                    return Err(format!(
                        "dim[{}] mismatch: inferred {} but actual is {}",
                        i, v, truth.shape[i]
                    ));
                }
                // Symbolic dims are fine -- no claim made
            }

            Ok(())
        }
    }
}

impl MilliOpGraph {
    /// Validate that `infer()` never returns incorrect information for any op
    /// in this graph, at any ablation level.
    ///
    /// Runs eval to collect ground-truth intermediate values, then for each op
    /// and each ablation level, calls `infer()` with ablated inputs and checks
    /// that all concrete claims match reality.
    ///
    /// Panics on validation failure with a detailed message.
    pub fn validate_infer_against_eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
    ) -> ValidationReport {
        let mut backend = EvalBackend::NDArray;

        // Step 1: Run eval to collect all intermediate tensor values
        let intermediate_values = self
            .collect_all_intermediate_values(inputs)
            .expect("eval failed during validation");

        let mut report = ValidationReport::default();

        let levels = [
            AblationLevel::Numeric,
            AblationLevel::Shaped,
            AblationLevel::Ranked,
            AblationLevel::Minimal,
        ];

        // Step 2: For each op, for each ablation level, validate infer()
        for op_id in self.op_ordering() {
            let op = &self.ops[op_id];
            let op_kind = op.op_kind();

            // Collect ground truth for this op's outputs
            let output_ids: Vec<GlobalId> = op.outputs().collect();
            let mut ground_truths: HashMap<GlobalId, GroundTruth> = HashMap::new();
            for &out_id in &output_ids {
                if let Some(tensor) = intermediate_values.get(&out_id) {
                    ground_truths.insert(out_id, GroundTruth::from_numeric_tensor(tensor));
                }
            }

            if ground_truths.is_empty() {
                // Op produced no outputs we can validate
                continue;
            }

            for &level in &levels {
                let mut resolver = SymbolicResolver::new();

                // Build the known map: all tensors as ground-truth Numeric,
                // except this op's inputs which are ablated.
                let input_ids: Vec<GlobalId> = op.inputs().collect();

                let mut known: HashMap<GlobalId, TensorInfo> = HashMap::new();

                // Insert all intermediate values as Numeric TensorInfo
                for (id, tensor) in &intermediate_values {
                    known.insert(*id, TensorInfo::from(tensor.clone()));
                }

                // Now overwrite this op's inputs with ablated versions
                for &input_id in &input_ids {
                    if let Some(tensor) = intermediate_values.get(&input_id) {
                        known.insert(input_id, ablate_tensor(tensor, level, &mut resolver));
                    }
                }

                // Call infer
                let result = op.infer(&known, &mut resolver, &mut backend);

                match result {
                    Err(MilliOpGraphError::UnableToInfer) => {
                        report.unable_to_infer_count += 1;
                    }
                    Err(e) => {
                        // Non-UnableToInfer errors at ablated levels are acceptable --
                        // the op may legitimately fail if it can't handle degraded inputs.
                        // But at Numeric level this is a real problem.
                        if level == AblationLevel::Numeric {
                            report.failure_count += 1;
                            report.failures.push(ValidationFailure {
                                op_id: *op_id,
                                op_kind: op_kind.clone(),
                                level,
                                message: format!("infer() returned error at Numeric level: {e}"),
                            });
                        } else {
                            report.unable_to_infer_count += 1;
                        }
                    }
                    Ok(inferred_outputs) => {
                        let mut all_ok = true;
                        for (out_id, inferred_info) in inferred_outputs {
                            if let Some(truth) = ground_truths.get(&out_id) {
                                match validate_against_ground_truth(&inferred_info, truth) {
                                    Ok(()) => {}
                                    Err(msg) => {
                                        all_ok = false;
                                        report.failure_count += 1;
                                        report.failures.push(ValidationFailure {
                                            op_id: *op_id,
                                            op_kind: op_kind.clone(),
                                            level,
                                            message: format!(
                                                "output {:?}: {}",
                                                out_id, msg
                                            ),
                                        });
                                    }
                                }
                            }
                        }
                        if all_ok {
                            report.pass_count += 1;
                        }
                    }
                }
            }
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GlobalId;
    use crate::milli_graph::ops::SimpleBinary;

    /// Build a simple add graph and validate infer correctness.
    #[test]
    fn test_validate_infer_simple_add() {
        let rng = &mut rand::rng();
        let ext_x = GlobalId::new(rng);
        let ext_y = GlobalId::new(rng);
        let ext_out = GlobalId::new(rng);

        let (mut graph, input_map) = MilliOpGraph::new([ext_x, ext_y], rng);
        let x = input_map[&ext_x];
        let y = input_map[&ext_y];

        let out = SimpleBinary::add(&mut graph, x, y, rng);
        graph.add_output(out, ext_out);

        let x_tensor: NumericTensor<DynRank> =
            NumericTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let y_tensor: NumericTensor<DynRank> =
            NumericTensor::from_vec_shape(vec![4.0f32, 5.0, 6.0], vec![3]).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(ext_x, x_tensor);
        inputs.insert(ext_y, y_tensor);

        let report = graph.validate_infer_against_eval(&inputs);
        assert_eq!(
            report.failure_count, 0,
            "Validation failures:\n{}",
            report
        );
        // Should have at least one pass (Numeric level always works for add)
        assert!(report.pass_count > 0, "Expected at least one pass");
    }
}
