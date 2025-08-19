use crate::DynRank;
use crate::numeric_tensor::NumericTensor;
use crate::symbolic_graph::{SymbolicGraphNodePath, SymbolicGraphTensorPath};

pub trait SymbolicGraphObserver {
    fn on_op_executed(&mut self, node_path: &SymbolicGraphNodePath);
    fn on_tensor_assigned(
        &mut self,
        tensor_path: &SymbolicGraphTensorPath,
        tensor: &NumericTensor<DynRank>,
    );
}

impl SymbolicGraphObserver for () {
    fn on_op_executed(&mut self, _node_path: &SymbolicGraphNodePath) {}
    fn on_tensor_assigned(
        &mut self,
        _tensor_path: &SymbolicGraphTensorPath,
        _tensor: &NumericTensor<DynRank>,
    ) {
    }
}
