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

/// Legacy interpreter — uses old eval path. Kept for examples/callers not yet migrated.
pub fn interpret_milli_graph_legacy(
    graph: &MilliOpGraph,
    inputs: &std::collections::HashMap<GlobalId, NumericTensor<DynRank>>,
) -> Result<std::collections::HashMap<GlobalId, NumericTensor<DynRank>>, crate::milli_graph::MilliOpGraphError> {
    let mut backend = EvalBackend::NDArray;
    Ok(graph.eval(inputs, &mut (), &mut backend)?.collect())
}

/// Legacy: return ALL intermediate tensors via old eval path.
pub fn interpret_milli_graph_all_intermediates(
    graph: &MilliOpGraph,
    inputs: &std::collections::HashMap<GlobalId, NumericTensor<DynRank>>,
) -> Result<std::collections::HashMap<GlobalId, NumericTensor<DynRank>>, crate::milli_graph::MilliOpGraphError> {
    graph.collect_all_intermediate_values(inputs)
}

/// Legacy: map each output tensor to its producing op info.
pub fn tensor_producers(
    graph: &MilliOpGraph,
    intermediates: &std::collections::HashMap<GlobalId, NumericTensor<DynRank>>,
) -> std::collections::HashMap<GlobalId, (String, Vec<(GlobalId, Vec<u64>)>)> {
    #![allow(clippy::type_complexity)]
    use crate::graph::{Graph, Node};
    let mut result = std::collections::HashMap::new();
    for &op_id in graph.op_ordering() {
        let Some(op) = graph.get_node_by_id(&op_id) else { continue };
        let kind = op.op_kind();
        let input_shapes: Vec<(GlobalId, Vec<u64>)> = op.inputs().map(|id| {
            let shape = intermediates.get(&id).map(|t| t.shape().to_vec()).unwrap_or_default();
            (id, shape)
        }).collect();
        for out_id in op.outputs() {
            result.insert(out_id, (kind.clone(), input_shapes.clone()));
        }
    }
    result
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

// ---------------------------------------------------------------------------
// Legacy API — kept for backward compatibility with existing callers
// (interfaces.rs, super_graph/, server, examples)
// ---------------------------------------------------------------------------

use crate::backends::eval_backend;
use crate::backends::eval_backend::EvalBackend;
use crate::migration::numeric_tensor::NumericTensor;
use crate::symbolic_graph::SymbolicGraph;
use crate::symbolic_graph::observer::SymbolicGraphObserver;
use crate::symbolic_graph::tensor_store::TensorStore;
use std::sync::Arc;
use std::time::Instant;

pub enum CompilationSubject {
    SymbolicGraph { symbolic_graph: Arc<SymbolicGraph> },
}

pub trait CompiledProgramObserver {
    fn on_op_executed(
        &mut self,
        node_path: &[GlobalId],
        start_instant: Instant,
        end_instant: Instant,
        backend: &mut EvalBackend,
    );
    fn on_tensor_assigned(
        &mut self,
        tensor_path: &[GlobalId],
        tensor: &NumericTensor<DynRank>,
        backend: &mut EvalBackend,
    );
    fn on_loading_weight(&mut self, path: &[GlobalId], weight_name: Option<String>);
    fn should_cancel(&mut self) -> bool {
        false
    }
}

#[derive(thiserror::Error, Debug)]
pub enum CompilerError {
    #[error(transparent)]
    EvalRuntimeError(#[from] eval_backend::EvalRuntimeError),
}

pub struct CompiledProgram {
    pub interim_graph: Arc<SymbolicGraph>,
}

struct SymbolicGraphObserverWrapper<'a, T: CompiledProgramObserver> {
    observer: &'a mut T,
}

impl<T: CompiledProgramObserver> SymbolicGraphObserver for SymbolicGraphObserverWrapper<'_, T> {
    fn on_op_executed(
        &mut self,
        node_path: &[GlobalId],
        start_instant: Instant,
        end_instant: Instant,
        backend: &mut EvalBackend,
    ) {
        self.observer
            .on_op_executed(node_path, start_instant, end_instant, backend);
    }

    fn on_tensor_assigned(
        &mut self,
        tensor_path: &[GlobalId],
        tensor: &NumericTensor<DynRank>,
        backend: &mut EvalBackend,
    ) {
        self.observer
            .on_tensor_assigned(tensor_path, tensor, backend);
    }

    fn on_loading_weight(&mut self, path: &[GlobalId], weight_name: Option<String>) {
        self.observer.on_loading_weight(path, weight_name);
    }

    fn should_cancel(&mut self) -> bool {
        self.observer.should_cancel()
    }
}

impl CompiledProgram {
    pub fn run<T: CompiledProgramObserver>(
        &self,
        eval_backend: &mut EvalBackend,
        tensor_store: &TensorStore,
        tensor_cache: Option<&mut crate::backends::ModelLoadedTensorCache>,
        inputs: impl IntoIterator<Item = (String, NumericTensor<DynRank>)>,
        observer: &mut T,
    ) -> Result<impl Iterator<Item = (String, NumericTensor<DynRank>)>, CompilerError> {
        let mut observer = SymbolicGraphObserverWrapper { observer };
        let res = eval_backend::run(
            &self.interim_graph,
            tensor_store,
            tensor_cache,
            eval_backend,
            &mut observer,
            inputs,
        )?;

        Ok(res.into_iter())
    }
}

pub fn build_program(subject: CompilationSubject) -> CompiledProgram {
    match subject {
        CompilationSubject::SymbolicGraph { symbolic_graph } => CompiledProgram {
            interim_graph: symbolic_graph,
        },
    }
}
