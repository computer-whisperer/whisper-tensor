//! Validation harness for `infer()` correctness.
//!
//! For each op in a MilliOpGraph, tests that `infer()` never returns *wrong*
//! information at any knowledge level. It may return *less* information
//! (e.g. UnableToInfer or Minimal when Shaped is possible), but any concrete
//! claims must match the ground-truth values from eval.

use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::MilliOp;
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::numeric_tensor::{NumericTensor as NewNumericTensor, NumericTensorView, TensorLayout};
use crate::pool::{Pool, SystemPool};
use crate::scalar_info::{ScalarInfo, ScalarInfoTyped};
use crate::symbolic_scalar::{SymbolicResolver, SymbolicScalar, SymbolicScalarTyped};
use crate::tensor_info::{MinimalTensor, ShapedTensor, TensorInfo, TensorInfoRanked};
use crate::tensor_rank::DynRank;
use crate::numeric_dtype::NumericDType;
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

/// Ground truth for a single output tensor.
struct GroundTruth {
    dtype: NumericDType,
    shape: Vec<u64>,
    rank: usize,
}

impl GroundTruth {
    fn from_view(view: &NumericTensorView<'_, DynRank>) -> Self {
        let shape = view.shape().clone();
        let rank = shape.len();
        Self {
            dtype: view.dtype(),
            shape,
            rank,
        }
    }
}

/// Ablate a tensor view to the given TensorInfo level.
fn ablate_view<'p, P: Pool + 'p>(
    view: &NumericTensorView<'_, DynRank>,
    level: AblationLevel,
    resolver: &mut SymbolicResolver,
    pool: &'p P,
) -> TensorInfo<'p, P> {
    let dtype = view.dtype();
    let shape = view.shape();
    let rank = shape.len();
    match level {
        AblationLevel::Numeric => TensorInfo::from_view(view, pool),
        AblationLevel::Shaped => {
            let first_element = ScalarInfo::Symbolic(SymbolicScalar::new(dtype, resolver));
            TensorInfo::from(ShapedTensor::<DynRank>::new_symbolic(
                first_element, shape.clone(), resolver,
            ))
        }
        AblationLevel::Ranked => {
            let first_element = ScalarInfo::Symbolic(SymbolicScalar::new(dtype, resolver));
            let symbolic_dims: Vec<ScalarInfoTyped<u64>> = (0..rank)
                .map(|_| ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(resolver)))
                .collect();
            TensorInfo::Ranked(TensorInfoRanked::new(first_element, symbolic_dims, resolver))
        }
        AblationLevel::Minimal => {
            let first_element = ScalarInfo::Symbolic(SymbolicScalar::new(dtype, resolver));
            TensorInfo::from(MinimalTensor::new(first_element, SymbolicScalarTyped::new(resolver)))
        }
    }
}

/// Validate that an inferred TensorInfo does not contradict ground truth.
///
/// Returns `Ok(())` if every concrete claim in `inferred` matches `truth`.
/// Returns `Err(message)` if any concrete claim is wrong.
fn validate_against_ground_truth(inferred: &TensorInfo<'_, impl crate::pool::Pool>, truth: &GroundTruth) -> Result<(), String> {
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
    /// Legacy entry point — converts legacy tensors and delegates to the pool-based version.
    pub fn validate_infer_against_eval(
        &self,
        inputs: &HashMap<GlobalId, crate::migration::numeric_tensor::NumericTensor<DynRank>>,
    ) -> ValidationReport {
        // Convert legacy tensors to TensorInfo, extract concrete new-type tensors.
        let infos: Vec<_> = inputs
            .iter()
            .map(|(&id, legacy)| (id, TensorInfo::from_legacy(legacy, &SystemPool)))
            .collect();
        let new_tensors: Vec<_> = infos
            .iter()
            .map(|(id, info)| (*id, info.as_concrete().expect("must be concrete").clone()))
            .collect();
        let views: Vec<_> = new_tensors.iter().map(|(id, t)| (*id, t.view())).collect();
        let view_map: HashMap<GlobalId, &NumericTensorView<'_, DynRank>> = views
            .iter()
            .map(|(id, view)| (*id, view))
            .collect();
        self.validate_infer_against_pool_eval(&view_map)
    }

    /// Validate that `infer()` never returns incorrect information for any op
    /// in this graph, at any ablation level.
    ///
    /// Runs each op through `eval_new()` to collect ground-truth shapes, then
    /// for each op and each ablation level, calls `infer()` with ablated inputs
    /// and checks that all concrete claims match reality.
    pub fn validate_infer_against_pool_eval<'v>(
        &self,
        inputs: &HashMap<GlobalId, &'v NumericTensorView<'v, DynRank>>,
    ) -> ValidationReport {
        static POOL: SystemPool = SystemPool;

        // Step 1: Walk ops in order, evaluating each via eval_new to get
        // ground-truth intermediate tensors.
        let mut intermediates: HashMap<GlobalId, NewNumericTensor<'static, DynRank, SystemPool>> =
            HashMap::new();

        // Map external inputs → internal IDs.
        for (&ext_id, &view) in inputs {
            if let Some(&int_id) = self.input_map.get(&ext_id) {
                // Copy view into a pool-backed tensor.
                let layout = TensorLayout::<DynRank>::row_major(view.shape().clone(), view.dtype());
                if let Ok(buf) = POOL.allocate(layout.buffer_size_bytes()) {
                    let mut tensor = NewNumericTensor::from_parts(buf, layout);
                    for i in 0..view.numel() {
                        tensor.write_element(i, view.read_element(i));
                    }
                    intermediates.insert(int_id, tensor);
                }
            }
        }

        // Evaluate each op via eval_new.
        for op_id in self.op_ordering() {
            let op = &self.ops[op_id];
            let input_ids: Vec<GlobalId> = op.inputs().collect();
            let input_views: Vec<_> = input_ids
                .iter()
                .filter_map(|id| intermediates.get(id).map(|t| t.view()))
                .collect();

            if input_views.len() != input_ids.len() {
                continue; // Missing inputs — skip
            }

            let input_view_refs: Vec<_> = input_views.iter().collect();
            match op.eval_new(&input_views, &POOL) {
                Ok(output_tensors) => {
                    let output_ids: Vec<GlobalId> = op.outputs().collect();
                    for (i, tensor) in output_tensors.into_iter().enumerate() {
                        if let Some(&out_id) = output_ids.get(i) {
                            intermediates.insert(out_id, tensor);
                        }
                    }
                }
                Err(_) => {} // eval_new failed — skip this op
            }
        }

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
                if let Some(tensor) = intermediates.get(&out_id) {
                    ground_truths.insert(out_id, GroundTruth::from_view(&tensor.view()));
                }
            }

            if ground_truths.is_empty() {
                continue;
            }

            for &level in &levels {
                let mut resolver = SymbolicResolver::new();
                let input_ids: Vec<GlobalId> = op.inputs().collect();

                let mut known: HashMap<GlobalId, TensorInfo<'_, SystemPool>> = HashMap::new();

                // Insert all intermediate values as Numeric TensorInfo
                for (id, tensor) in &intermediates {
                    known.insert(*id, TensorInfo::from_view(&tensor.view(), &POOL));
                }

                // Overwrite this op's inputs with ablated versions
                for &input_id in &input_ids {
                    if let Some(tensor) = intermediates.get(&input_id) {
                        known.insert(
                            input_id,
                            ablate_view(&tensor.view(), level, &mut resolver, &POOL),
                        );
                    }
                }

                let result = op.infer(&known, &mut resolver, &POOL);

                match result {
                    Err(MilliOpGraphError::UnableToInfer) => {
                        report.unable_to_infer_count += 1;
                    }
                    Err(e) => {
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
                                            message: format!("output {:?}: {}", out_id, msg),
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
    use crate::numeric_scalar::NumericScalar;

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

        // Build inputs using new types only.
        let layout_x = TensorLayout::<DynRank>::row_major(vec![3], NumericDType::F32);
        let buf_x = SystemPool.allocate(layout_x.buffer_size_bytes()).unwrap();
        let mut x_tensor: NewNumericTensor<'_, DynRank, SystemPool> = NewNumericTensor::from_parts(buf_x, layout_x);
        for (i, &v) in [1.0f32, 2.0, 3.0].iter().enumerate() {
            x_tensor.write_element(i, NumericScalar::from_f32(v));
        }
        let layout_y = TensorLayout::<DynRank>::row_major(vec![3], NumericDType::F32);
        let buf_y = SystemPool.allocate(layout_y.buffer_size_bytes()).unwrap();
        let mut y_tensor: NewNumericTensor<'_, DynRank, SystemPool> = NewNumericTensor::from_parts(buf_y, layout_y);
        for (i, &v) in [4.0f32, 5.0, 6.0].iter().enumerate() {
            y_tensor.write_element(i, NumericScalar::from_f32(v));
        }

        let x_view = x_tensor.view();
        let y_view = y_tensor.view();
        let inputs: HashMap<GlobalId, &NumericTensorView<'_, DynRank>> =
            HashMap::from([(ext_x, &x_view), (ext_y, &y_view)]);

        let report = graph.validate_infer_against_pool_eval(&inputs);
        assert_eq!(report.failure_count, 0, "Validation failures:\n{}", report);
        assert!(report.pass_count > 0, "Expected at least one pass");
    }
}
