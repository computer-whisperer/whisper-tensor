use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::numeric_tensor::NumericTensor;
use crate::symbolic_graph::{SymbolicGraphNodePath, SymbolicGraphTensorPath};

pub trait SymbolicGraphObserver {
    fn on_op_executed(&mut self, node_path: &SymbolicGraphNodePath, backend: &mut EvalBackend);
    fn on_tensor_assigned(
        &mut self,
        tensor_path: &SymbolicGraphTensorPath,
        tensor: &NumericTensor<DynRank>,
        backend: &mut EvalBackend,
    );
}

impl SymbolicGraphObserver for () {
    fn on_op_executed(&mut self, _node_path: &SymbolicGraphNodePath, _backend: &mut EvalBackend) {}
    fn on_tensor_assigned(
        &mut self,
        _tensor_path: &SymbolicGraphTensorPath,
        _tensor: &NumericTensor<DynRank>,
        _backend: &mut EvalBackend,
    ) {
    }
}
