use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::milli_graph::{MilliOpGraphNodePath, MilliOpGraphTensorPath};
use crate::numeric_tensor::NumericTensor;

pub trait MilliOpGraphObserver {
    fn on_tensor_assigned(
        &mut self,
        tensor_path: &MilliOpGraphTensorPath,
        tensor: &NumericTensor<DynRank>,
        backend: &mut EvalBackend,
    );
    fn on_node_executed(&mut self, node_path: &MilliOpGraphNodePath, backend: &mut EvalBackend);
}

impl MilliOpGraphObserver for () {
    fn on_tensor_assigned(
        &mut self,
        _tensor_path: &MilliOpGraphTensorPath,
        _tensor: &NumericTensor<DynRank>,
        _backend: &mut EvalBackend,
    ) {
    }
    fn on_node_executed(&mut self, _node_path: &MilliOpGraphNodePath, _backend: &mut EvalBackend) {}
}
