use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::graph::GlobalId;
use crate::numeric_tensor::NumericTensor;
use std::time::Instant;

pub trait SuperGraphObserver {
    fn on_node_executed(
        &mut self,
        path: &[GlobalId],
        op_kind: &str,
        start_instant: Instant,
        end_instant: Instant,
        backend: &mut EvalBackend,
    );
    fn on_tensor_assigned(
        &mut self,
        path: &[GlobalId],
        tensor: &NumericTensor<DynRank>,
        backend: &mut EvalBackend,
    );
}

impl SuperGraphObserver for () {
    fn on_node_executed(
        &mut self,
        _path: &[GlobalId],
        _op_kind: &str,
        _start_instant: Instant,
        _end_instant: Instant,
        _backend: &mut EvalBackend,
    ) {
    }
    fn on_tensor_assigned(
        &mut self,
        _path: &[GlobalId],
        _tensor: &NumericTensor<DynRank>,
        _backend: &mut EvalBackend,
    ) {
    }
}
