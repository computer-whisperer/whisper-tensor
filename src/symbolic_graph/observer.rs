use crate::graph::GlobalId;
use crate::numeric_tensor::NumericTensorView;
use crate::tensor_rank::DynRank;
use std::time::Instant;

pub trait SymbolicGraphObserver {
    fn on_op_executed(
        &mut self,
        node_path: &[GlobalId],
        start_instant: Instant,
        end_instant: Instant,
    );
    fn on_tensor_assigned(
        &mut self,
        tensor_path: &[GlobalId],
        tensor: &NumericTensorView<'_, DynRank>,
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
    ) {
    }
    fn on_tensor_assigned(
        &mut self,
        _tensor_path: &[GlobalId],
        _tensor: &NumericTensorView<'_, DynRank>,
    ) {
    }
    fn on_loading_weight(&mut self, _path: &[GlobalId], _weight_name: Option<String>) {}
    fn should_cancel(&mut self) -> bool {
        false
    }
}
