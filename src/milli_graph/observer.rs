use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::milli_graph::{MilliOpGraphNodePath, MilliOpGraphTensorPath};
use crate::numeric_tensor::NumericTensor;
use std::time::Instant;

pub trait MilliOpGraphObserver {
    fn on_tensor_assigned(
        &mut self,
        tensor_path: &MilliOpGraphTensorPath,
        tensor: &NumericTensor<DynRank>,
        backend: &mut EvalBackend,
    );
    fn on_node_executed(
        &mut self,
        node_path: &MilliOpGraphNodePath,
        start_instant: Instant,
        end_instant: Instant,
        backend: &mut EvalBackend,
    );
}

impl MilliOpGraphObserver for () {
    fn on_tensor_assigned(
        &mut self,
        _tensor_path: &MilliOpGraphTensorPath,
        _tensor: &NumericTensor<DynRank>,
        _backend: &mut EvalBackend,
    ) {
    }

    fn on_node_executed(
        &mut self,
        _node_path: &MilliOpGraphNodePath,
        _start_instant: Instant,
        _end_instant: Instant,
        _backend: &mut EvalBackend,
    ) {
    }
}
