// See README.md in this directory — this module is a volatile sandbox.

pub mod attempts;

use crate::graph::GlobalId;
use crate::milli_graph::MilliOpGraph;
use crate::numeric_tensor::NumericTensorView;
use crate::tensor_rank::DynRank;
use serde::{Deserialize, Serialize};

// ─── User-facing compile options ────────────────────────────────────────────
//
// These enums select between alternative partitioner / codegen implementations
// inside the v14 attempt. They live here (not under attempts/v14) so that the
// surface stays stable as new attempts land — adding a new partitioner means
// adding a variant here and a dispatch arm in `compiled_eval::compile_nano_graph`.

/// Which partitioner to use when building the execution plan.
///
/// Default: `LaneSplit { num_lanes: 8 }` (the current best, attempt M).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PartitionerKind {
    /// Single phase, single span containing the whole NanoGraph. No
    /// parallelism, no group splitting. Useful as a debug baseline that
    /// isolates I/O / codegen issues from partitioner issues.
    Trivial,
    /// Spatial-tiling partitioner (attempt M). Splits splittable groups
    /// across `num_lanes` lanes and avoids barriers within elementwise
    /// chains. Currently the only non-trivial partitioner.
    LaneSplit { num_lanes: usize },
}

impl Default for PartitionerKind {
    fn default() -> Self {
        Self::LaneSplit { num_lanes: 8 }
    }
}

/// Which codegen backend to use for compiling each span.
///
/// Default: `Jit` (x86_jit / dynasm), with automatic fallback to pool_eval
/// for spans containing opaque ops or for which compilation fails.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub enum CodegenKind {
    /// x86_jit (dynasm). Falls back to pool_eval per-span on opaque ops or
    /// compile errors. Requires the `x86_compile` feature.
    #[default]
    Jit,
    /// Force pool_eval for every span (no JIT). Slow but useful as a
    /// correctness baseline that exercises the executor's I/O contract
    /// without depending on JIT codegen.
    PoolEval,
}

/// Options controlling how a NanoGraph is compiled into an executable plan.
///
/// Threaded through `SuperGraphEvalOptions::CompiledEval` and consumed by
/// `super_graph::compiled_eval::compile_nano_graph`. Each field selects
/// between alternative implementations exposed via the `PartitionerKind` /
/// `CodegenKind` enums above.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CompileOptions {
    pub partitioner: PartitionerKind,
    pub codegen: CodegenKind,
}

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
