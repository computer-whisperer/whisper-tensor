use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::numeric_tensor::NumericTensor;
use crate::super_graph::{SuperGraphNodePath, SuperGraphTensorPath};

pub trait SuperGraphObserver {
    fn on_node_executed(&mut self, path: &SuperGraphNodePath, backend: &mut EvalBackend);
    fn on_tensor_assigned(
        &mut self,
        path: &SuperGraphTensorPath,
        tensor: &NumericTensor<DynRank>,
        backend: &mut EvalBackend,
    );
}

impl SuperGraphObserver for () {
    fn on_node_executed(&mut self, _path: &SuperGraphNodePath, _backend: &mut EvalBackend) {}
    fn on_tensor_assigned(
        &mut self,
        _path: &SuperGraphTensorPath,
        _tensor: &NumericTensor<DynRank>,
        _backend: &mut EvalBackend,
    ) {
    }
}
