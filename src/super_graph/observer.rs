use crate::graph::GlobalId;
use crate::numeric_tensor::NumericTensorView;
use crate::tensor_rank::DynRank;
use std::time::Instant;

pub trait SuperGraphObserver {
    fn on_node_executed(
        &mut self,
        path: &[GlobalId],
        op_kind: &str,
        start_instant: Instant,
        end_instant: Instant,
    );
    fn on_tensor_assigned(&mut self, path: &[GlobalId], tensor: &NumericTensorView<'_, DynRank>);
    fn on_loading_weight(&mut self, path: &[GlobalId], weight_name: Option<String>);
    fn on_progress(&mut self, path: &[GlobalId], tier: i64, numerator: f64, denominator: f64);
    /// Tagged milestone for compiled execution. Emitted by `execute_compiled`
    /// and friends to surface the per-stage breakdown (info_inputs build,
    /// weight load, input prep, JIT execute, output extract, etc.) that the
    /// per-node `on_node_executed` event collapses into a single number.
    ///
    /// `path` identifies the owning super graph node. `stage` is a
    /// dot-separated label (e.g. `compiled.exec.weight_load`). Default impl
    /// is a no-op so that consumers that don't care about compiled-eval
    /// internals don't need to override anything.
    fn on_compiled_milestone(
        &mut self,
        path: &[GlobalId],
        stage: &str,
        iter: Option<u64>,
        start_instant: Instant,
        end_instant: Instant,
    ) {
        let _ = (path, stage, iter, start_instant, end_instant);
    }
    fn should_cancel(&mut self) -> bool {
        false
    }
}

impl SuperGraphObserver for () {
    fn on_node_executed(
        &mut self,
        _path: &[GlobalId],
        _op_kind: &str,
        _start_instant: Instant,
        _end_instant: Instant,
    ) {
    }
    fn on_tensor_assigned(&mut self, _path: &[GlobalId], _tensor: &NumericTensorView<'_, DynRank>) {
    }
    fn on_loading_weight(&mut self, _path: &[GlobalId], _weight_name: Option<String>) {}
    fn on_progress(&mut self, _path: &[GlobalId], _tier: i64, _numerator: f64, _denominator: f64) {}
    fn should_cancel(&mut self) -> bool {
        false
    }
}
