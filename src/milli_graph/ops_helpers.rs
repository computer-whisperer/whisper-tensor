use super::ops::*;
use crate::backends::ndarray_backend::conversions::NDArrayNumericTensorType;
use crate::milli_graph::{MilliOpGraph, MilliOpGraphTensorId};
use crate::graph::{Graph, InnerGraph, Node};
use crate::symbolic_graph::SymbolicGraphTensorId;

pub(crate) fn rank(
    graph: &mut MilliOpGraph<SymbolicGraphTensorId>,
    tensor: MilliOpGraphTensorId,
) -> MilliOpGraphTensorId {
    let shape_tid = MilliOpShape::new(graph, tensor);
    MilliOpShape::new(graph, shape_tid)
}

pub(crate) fn scalar_const<T>(
    graph: &mut MilliOpGraph<SymbolicGraphTensorId>,
    value: T,
) -> MilliOpGraphTensorId
where
    T: NDArrayNumericTensorType,
{
    let node = MilliOpConstant::new_scalar(graph, value);
    match graph.inner(&()).get_node(&node) {
        Some(AnyMilliOp::Constant(op)) => op.outputs().next().unwrap(),
        _ => unreachable!(),
    }
}

pub(crate) fn resolve_axes(
    graph: &mut MilliOpGraph<SymbolicGraphTensorId>,
    axes: MilliOpGraphTensorId,
    tensor: MilliOpGraphTensorId,
) -> MilliOpGraphTensorId {
    let shape_tid = MilliOpShape::new(graph, tensor);
    let rank_tid = MilliOpShape::new(graph, shape_tid);
    let axes2_tid = MilliOpSimpleBinary::add(graph, axes, rank_tid);
    MilliOpSimpleBinary::modulo(graph, axes2_tid, rank_tid, None)
}
