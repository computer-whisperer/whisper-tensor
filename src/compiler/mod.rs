// See README.md in this directory — this module is a volatile sandbox.

pub mod nano_op;

#[cfg(feature = "cranelift")]
pub mod codegen;

use crate::milli_graph::MilliOpGraph;

/// Run a MilliOpGraph through the interpreter and return results.
/// Convenience wrapper for benchmarking the compiler against the interpreter.
pub fn interpret_milli_graph(
    graph: &MilliOpGraph,
    inputs: &std::collections::HashMap<crate::graph::GlobalId, crate::numeric_tensor::NumericTensor<crate::DynRank>>,
) -> Result<
    std::collections::HashMap<crate::graph::GlobalId, crate::numeric_tensor::NumericTensor<crate::DynRank>>,
    crate::milli_graph::MilliOpGraphError,
> {
    let mut backend = crate::backends::eval_backend::EvalBackend::NDArray;
    Ok(graph.eval(inputs, &mut (), &mut backend)?.collect())
}

// ---------------------------------------------------------------------------
// Legacy API — kept for backward compatibility with existing callers
// (interfaces.rs, super_graph/, server, examples)
// ---------------------------------------------------------------------------

use crate::DynRank;
use crate::backends::eval_backend;
use crate::backends::eval_backend::EvalBackend;
use crate::graph::GlobalId;
use crate::numeric_tensor::NumericTensor;
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
