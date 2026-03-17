//! Span-by-span executor for an ExecutionPlan.
//!
//! Evaluates each span's NanoGraph using the memory-efficient evaluator,
//! threading NDArrayNumericTensor values through the shared value store
//! between phases.

use std::collections::HashMap;

use crate::backends::ndarray_backend::numeric_tensor::NDArrayNumericTensor;
use crate::graph::GlobalId;
use crate::nano_graph::eval::eval_efficient;
use crate::nano_graph::AtomId;
use crate::DynRank;

use super::types::*;

/// Execute a plan using the memory-efficient NanoGraph evaluator.
///
/// `inputs` provides all external data (weights + user inputs) as
/// tensors keyed by their base AtomId in the graph. Typically built
/// by mapping the plan's tensor_map entries to actual tensor data.
///
/// Returns model outputs keyed by their external GlobalId.
pub fn execute(
    plan: &ExecutionPlan,
    inputs: Vec<(AtomId, NDArrayNumericTensor<DynRank>)>,
) -> HashMap<GlobalId, NDArrayNumericTensor<DynRank>> {
    // Value store: base AtomId → tensor data.
    // Populated initially from external inputs, then from span outputs.
    let mut store: HashMap<AtomId, NDArrayNumericTensor<DynRank>> = HashMap::new();
    for (base, tensor) in inputs {
        store.insert(base, tensor);
    }

    // Execute phases sequentially.
    for (phase_idx, phase) in plan.phases.iter().enumerate() {
        // Collect outputs from all spans in this phase, then commit
        // to the store after the phase completes (barrier semantics).
        let mut phase_outputs: Vec<(AtomId, NDArrayNumericTensor<DynRank>)> = Vec::new();

        for (lane_idx, span) in phase.spans.iter().enumerate() {
            if span.graph.num_groups() == 0 {
                continue; // idle lane
            }

            // Gather this span's inputs from the store.
            // Pre-allocate zero fills for missing ranges (boundary ops).
            let zero_fills: Vec<(usize, NDArrayNumericTensor<DynRank>)> = span
                .inputs
                .iter()
                .enumerate()
                .filter(|(_, range)| !store.contains_key(&range.base))
                .map(|(i, range)| (i, make_zeros(range.count as usize, range.dtype)))
                .collect();
            let missing = zero_fills.len();

            // Insert zero fills into the store temporarily.
            for (i, (_, tensor)) in zero_fills.iter().enumerate() {
                store.insert(span.inputs[zero_fills[i].0].base, tensor.clone());
            }

            let span_inputs: Vec<(AtomId, &NDArrayNumericTensor<DynRank>)> = span
                .inputs
                .iter()
                .map(|range| {
                    let tensor = store.get(&range.base).unwrap();
                    (range.base, tensor)
                })
                .collect();

            if missing > 0 && phase_idx == 0 && lane_idx == 0 {
                eprintln!(
                    "  Warning: {} of {} input ranges zero-filled (unsupported boundary ops)",
                    missing,
                    span.inputs.len()
                );
            }

            // Evaluate the span's NanoGraph with memory-efficient evaluator.
            let eval_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                eval_efficient(&span.graph, &span_inputs, &span.outputs)
            }));

            match eval_result {
                Ok(output_tensors) => {
                    for (output_range, tensor) in span.outputs.iter().zip(output_tensors) {
                        phase_outputs.push((output_range.base, tensor));
                    }
                }
                Err(_) => {
                    eprintln!(
                        "  Phase {} lane {}: eval panicked (likely div-by-zero from zero-filled boundary ops), zero-filling outputs",
                        phase_idx, lane_idx
                    );
                    for output_range in &span.outputs {
                        let tensor = make_zeros(output_range.count as usize, output_range.dtype);
                        phase_outputs.push((output_range.base, tensor));
                    }
                }
            }
        }

        // Commit phase outputs to the store (barrier).
        for (base, tensor) in phase_outputs {
            store.insert(base, tensor);
        }
    }

    // Extract model outputs.
    let mut results = HashMap::new();
    for output in &plan.model_outputs {
        if let Some(tensor) = store.get(&output.range.base) {
            results.insert(output.tensor_id, tensor.clone());
        }
    }
    results
}

/// Create a zero-filled 1D tensor of the given count and dtype.
fn make_zeros(count: usize, dtype: crate::dtype::DType) -> NDArrayNumericTensor<DynRank> {
    use ndarray::{ArcArray, IxDyn};
    let shape = IxDyn(&[count]);
    match dtype {
        crate::dtype::DType::F32 => {
            NDArrayNumericTensor::F32(ArcArray::zeros(shape))
        }
        crate::dtype::DType::F64 => {
            NDArrayNumericTensor::F64(ArcArray::zeros(shape))
        }
        crate::dtype::DType::I64 => {
            NDArrayNumericTensor::I64(ArcArray::zeros(shape))
        }
        crate::dtype::DType::I32 => {
            NDArrayNumericTensor::I32(ArcArray::zeros(shape))
        }
        _other => {
            // Fallback to F32 zeros for uncommon dtypes.
            NDArrayNumericTensor::F32(ArcArray::zeros(shape))
        }
    }
}
