use rand::Rng;
use super::ops::*;
use crate::backends::ndarray_backend::conversions::NDArrayNumericTensorType;
use crate::milli_graph::{MilliOpGraph, MilliOpGraphTensorId};
use crate::symbolic_graph::SymbolicGraphTensorId;

pub(crate) fn rank(
    graph: &mut MilliOpGraph<SymbolicGraphTensorId>,
    tensor: MilliOpGraphTensorId,
    rng: &mut impl Rng
) -> MilliOpGraphTensorId {
    let shape_tid = Shape::push_new(graph, tensor, rng);
    Shape::push_new(graph, shape_tid, rng)
}

pub(crate) fn scalar_const<T>(
    graph: &mut MilliOpGraph<SymbolicGraphTensorId>,
    value: T,
    rng: &mut impl Rng
) -> MilliOpGraphTensorId
where
    T: NDArrayNumericTensorType,
{
    Constant::new_scalar(graph, value, rng)
}

pub(crate) fn resolve_axes(
    graph: &mut MilliOpGraph<SymbolicGraphTensorId>,
    axes: MilliOpGraphTensorId,
    tensor: MilliOpGraphTensorId,
    rng: &mut impl Rng
) -> MilliOpGraphTensorId {
    let shape_tid = Shape::push_new(graph, tensor, rng);
    let rank_tid = Shape::push_new(graph, shape_tid, rng);
    let axes2_tid = SimpleBinary::add(graph, axes, rank_tid, rng);
    SimpleBinary::modulo(graph, axes2_tid, rank_tid, None, rng)
}
