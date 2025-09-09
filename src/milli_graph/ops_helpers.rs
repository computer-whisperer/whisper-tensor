use super::ops::*;
use crate::backends::ndarray_backend::conversions::NDArrayNumericTensorType;
use crate::milli_graph::{MilliOpGraph, MilliOpGraphTensorId};
use crate::symbolic_graph::SymbolicGraphTensorId;

pub(crate) fn rank(
    graph: &mut MilliOpGraph<SymbolicGraphTensorId>,
    tensor: MilliOpGraphTensorId,
) -> MilliOpGraphTensorId {
    let shape_tid = Shape::push_new(graph, tensor);
    Shape::push_new(graph, shape_tid)
}

pub(crate) fn scalar_const<T>(
    graph: &mut MilliOpGraph<SymbolicGraphTensorId>,
    value: T,
) -> MilliOpGraphTensorId
where
    T: NDArrayNumericTensorType,
{
    Constant::new_scalar(graph, value)
}

pub(crate) fn resolve_axes(
    graph: &mut MilliOpGraph<SymbolicGraphTensorId>,
    axes: MilliOpGraphTensorId,
    tensor: MilliOpGraphTensorId,
) -> MilliOpGraphTensorId {
    let shape_tid = Shape::push_new(graph, tensor);
    let rank_tid = Shape::push_new(graph, shape_tid);
    let axes2_tid = SimpleBinary::add(graph, axes, rank_tid);
    SimpleBinary::modulo(graph, axes2_tid, rank_tid, None)
}
