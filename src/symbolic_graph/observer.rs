use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::graph::GlobalId;
use crate::numeric_tensor::NumericTensor;
use std::time::Instant;

pub trait SymbolicGraphObserver {
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

impl SymbolicGraphObserver for () {
    fn on_op_executed(
        &mut self,
        _node_path: &[GlobalId],
        _start_instant: Instant,
        _end_instant: Instant,
        _backend: &mut EvalBackend,
    ) {
    }
    fn on_tensor_assigned(
        &mut self,
        _tensor_path: &[GlobalId],
        _tensor: &NumericTensor<DynRank>,
        _backend: &mut EvalBackend,
    ) {
    }
    fn on_loading_weight(&mut self, _path: &[GlobalId], _weight_name: Option<String>) {}
    fn should_cancel(&mut self) -> bool {
        false
    }
}
