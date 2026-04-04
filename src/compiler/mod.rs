// See README.md in this directory — this module is a volatile sandbox.

pub mod attempts;

use crate::graph::GlobalId;
use crate::milli_graph::MilliOpGraph;
use crate::numeric_tensor::NumericTensorView;
use crate::tensor_rank::DynRank;

/// Run a MilliOpGraph through pool_eval and return results.
/// Convenience wrapper for benchmarking the compiler against the interpreter.
pub fn interpret_milli_graph<'p, P: crate::pool::Pool + 'p>(
    graph: &MilliOpGraph,
    inputs: &std::collections::HashMap<GlobalId, &NumericTensorView<'_, DynRank>>,
    pool: &'p P,
) -> Result<
    std::collections::HashMap<GlobalId, crate::numeric_tensor::NumericTensor<'p, DynRank, P>>,
    crate::milli_graph::MilliOpGraphError,
> {
    graph.pool_eval(inputs, pool)
}

/// Return a sorted list of (op_kind, count) for all ops in the graph.
pub fn op_census(graph: &MilliOpGraph) -> Vec<(String, usize)> {
    use crate::graph::{Graph, Node};
    let mut counts = std::collections::HashMap::<String, usize>::new();
    for id in graph.node_ids() {
        if let Some(op) = graph.get_node_by_id(&id) {
            *counts.entry(op.op_kind()).or_default() += 1;
        }
    }
    let mut sorted: Vec<_> = counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    sorted
}
