//! Execution plan construction.

use std::collections::HashMap;

use crate::graph::GlobalId;
use crate::nano_graph::lower::LowerResult;

use super::types::*;

/// Build a trivial execution plan: 1 phase, 1 lane, 1 span containing
/// the entire NanoGraph. No splitting or parallelism.
///
/// This is the starting point — validation and execution infrastructure
/// are built against this before implementing a real partitioner.
pub fn plan_trivial(
    result: LowerResult<'static, 'static, crate::pool::SystemPool>,
    tensor_map: HashMap<GlobalId, TensorMapping>,
    model_outputs: Vec<OutputMapping>,
) -> ExecutionPlan {
    // The single span's inputs = all input_tensors from the graph
    // (weights + user inputs — everything not produced by a group).
    let span_inputs: Vec<AtomRange> = result
        .graph
        .input_tensors()
        .iter()
        .map(|it| AtomRange {
            base: it.base_id,
            count: it.count,
            dtype: it.dtype,
        })
        .collect();

    // The single span's outputs = model output atom ranges.
    let span_outputs: Vec<AtomRange> = model_outputs.iter().map(|om| om.range.clone()).collect();

    let span = Span {
        graph: result.graph.clone(),
        inputs: span_inputs,
        outputs: span_outputs,
    };

    ExecutionPlan {
        graph: result.graph,
        tensor_map,
        phases: vec![Phase { spans: vec![span] }],
        model_outputs,
    }
}
