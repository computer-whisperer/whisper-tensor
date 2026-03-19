//! Span-by-span executor for an ExecutionPlan.
//!
//! Evaluates each span's NanoGraph using the memory-efficient evaluator,
//! threading NDArrayNumericTensor values through the shared value store
//! between phases.
//!
//! Memory management: before execution, a use-count analysis determines
//! how many times each stored atom range will be read. Values are dropped
//! from the store as soon as their last consumer finishes, keeping peak
//! memory proportional to the live set rather than total atoms.

use std::collections::HashMap;

use crate::DynRank;
use crate::backends::ndarray_backend::numeric_tensor::NDArrayNumericTensor;
use crate::graph::GlobalId;
use crate::nano_graph::AtomId;
use crate::nano_graph::eval;

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
    // ── Step 1: Use-count analysis ──────────────────────────────────────
    //
    // For each atom range that appears as a span output or initial input,
    // count how many times it will be read as a span input in later phases.
    // Model outputs get an extra count so they survive until extraction.

    let mut use_counts: HashMap<AtomId, u32> = HashMap::new();

    // Initial inputs are consumed by phase 0 spans.
    // Count how many spans in all phases read each input base.
    for phase in &plan.phases {
        for span in &phase.spans {
            for inp in &span.inputs {
                *use_counts.entry(inp.base).or_default() += 1;
            }
        }
    }

    // Model outputs need to survive until extraction.
    for output in &plan.model_outputs {
        *use_counts.entry(output.range.base).or_default() += 1;
    }

    // ── Step 2: Execute phases ──────────────────────────────────────────

    let mut store: HashMap<AtomId, NDArrayNumericTensor<DynRank>> = HashMap::new();
    for (base, tensor) in inputs {
        store.insert(base, tensor);
    }

    for (phase_idx, phase) in plan.phases.iter().enumerate() {
        let mut phase_outputs: Vec<(AtomId, NDArrayNumericTensor<DynRank>)> = Vec::new();

        for (lane_idx, span) in phase.spans.iter().enumerate() {
            if span.graph.num_groups() == 0 {
                continue;
            }

            // Gather this span's inputs from the store.
            let zero_fills: Vec<(usize, NDArrayNumericTensor<DynRank>)> = span
                .inputs
                .iter()
                .enumerate()
                .filter(|(_, range)| !store.contains_key(&range.base))
                .map(|(i, range)| (i, make_zeros(range.count as usize, range.dtype)))
                .collect();
            let missing = zero_fills.len();

            for &(i, ref tensor) in &zero_fills {
                store.insert(span.inputs[i].base, tensor.clone());
            }

            // Slice store tensors to match the span's declared input count.
            let sliced_tensors: Vec<(AtomId, NDArrayNumericTensor<DynRank>)> = span
                .inputs
                .iter()
                .map(|range| {
                    let tensor = store.get(&range.base).unwrap();
                    let needed = range.count as usize;
                    let available = tensor.num_elements();
                    if available > needed {
                        (range.base, slice_tensor_prefix(tensor, needed))
                    } else {
                        (range.base, tensor.clone())
                    }
                })
                .collect();
            let span_inputs: Vec<(AtomId, &NDArrayNumericTensor<DynRank>)> = sliced_tensors
                .iter()
                .map(|(base, tensor)| (*base, tensor))
                .collect();

            if missing > 0 && phase_idx == 0 && lane_idx == 0 {
                eprintln!(
                    "  Warning: {} of {} input ranges zero-filled (unsupported boundary ops)",
                    missing,
                    span.inputs.len()
                );
            }

            // Evaluate the span.
            let eval_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                eval::eval(&span.graph, &span_inputs, &span.outputs)
            }));

            match eval_result {
                Ok(output_tensors) => {
                    for (output_range, tensor) in span.outputs.iter().zip(output_tensors) {
                        phase_outputs.push((output_range.base, tensor));
                    }
                }
                Err(_) => {
                    eprintln!(
                        "  Phase {} lane {}: eval panicked, zero-filling outputs",
                        phase_idx, lane_idx
                    );
                    for output_range in &span.outputs {
                        let tensor = make_zeros(output_range.count as usize, output_range.dtype);
                        phase_outputs.push((output_range.base, tensor));
                    }
                }
            }

            // Decrement use counts for consumed inputs and drop dead values.
            for inp in &span.inputs {
                if let Some(count) = use_counts.get_mut(&inp.base) {
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        store.remove(&inp.base);
                    }
                }
            }
        }

        // Commit phase outputs to the store (barrier).
        for (base, tensor) in phase_outputs {
            store.insert(base, tensor);
        }
    }

    // ── Step 3: Extract model outputs ───────────────────────────────────

    let mut results = HashMap::new();
    for output in &plan.model_outputs {
        if let Some(tensor) = store.remove(&output.range.base) {
            results.insert(output.tensor_id, tensor);
        }
    }
    results
}

/// Slice a tensor to its first `count` elements (flattened).
fn slice_tensor_prefix(
    tensor: &NDArrayNumericTensor<DynRank>,
    count: usize,
) -> NDArrayNumericTensor<DynRank> {
    use ndarray::{ArcArray, IxDyn};
    macro_rules! slice_variant {
        ($arr:expr, $variant:ident) => {{
            let flat: Vec<_> = $arr.iter().take(count).copied().collect();
            NDArrayNumericTensor::$variant(ArcArray::from_shape_vec(IxDyn(&[count]), flat).unwrap())
        }};
    }
    match tensor {
        NDArrayNumericTensor::F32(a) => slice_variant!(a, F32),
        NDArrayNumericTensor::F64(a) => slice_variant!(a, F64),
        NDArrayNumericTensor::I64(a) => slice_variant!(a, I64),
        NDArrayNumericTensor::I32(a) => slice_variant!(a, I32),
        NDArrayNumericTensor::BF16(a) => slice_variant!(a, BF16),
        NDArrayNumericTensor::F16(a) => slice_variant!(a, F16),
        NDArrayNumericTensor::U8(a) => slice_variant!(a, U8),
        NDArrayNumericTensor::I8(a) => slice_variant!(a, I8),
        other => other.clone(),
    }
}

/// Create a zero-filled 1D tensor of the given count and dtype.
fn make_zeros(count: usize, dtype: crate::dtype::DType) -> NDArrayNumericTensor<DynRank> {
    use ndarray::{ArcArray, IxDyn};
    let shape = IxDyn(&[count]);
    match dtype {
        crate::dtype::DType::F32 => NDArrayNumericTensor::F32(ArcArray::zeros(shape)),
        crate::dtype::DType::F64 => NDArrayNumericTensor::F64(ArcArray::zeros(shape)),
        crate::dtype::DType::I64 => NDArrayNumericTensor::I64(ArcArray::zeros(shape)),
        crate::dtype::DType::I32 => NDArrayNumericTensor::I32(ArcArray::zeros(shape)),
        _other => NDArrayNumericTensor::F32(ArcArray::zeros(shape)),
    }
}
