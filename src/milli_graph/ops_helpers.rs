use super::ops::*;
use crate::backends::ndarray_backend::conversions::NDArrayNumericTensorType;
use crate::milli_graph::{MilliOpGraph, MilliOpGraphTensorId};
use crate::symbolic_graph::SymbolicGraphTensorId;

pub(crate) fn rank(
    graph: &mut MilliOpGraph<SymbolicGraphTensorId>,
    tensor: MilliOpGraphTensorId,
) -> MilliOpGraphTensorId {
    let shape = graph.push_op(AnyMilliOp::Shape(MilliOpShape::new(tensor)));
    graph.push_op(AnyMilliOp::Shape(MilliOpShape::new(shape)))
}

pub(crate) fn scalar_const<T>(graph: &mut MilliOpGraph<SymbolicGraphTensorId>, value: T) -> MilliOpGraphTensorId
where
    T: NDArrayNumericTensorType,
{
    graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(value)))
}

pub(crate) fn resolve_axes(
    graph: &mut MilliOpGraph<SymbolicGraphTensorId>,
    axes: MilliOpGraphTensorId,
    tensor: MilliOpGraphTensorId,
) -> MilliOpGraphTensorId {
    let shape = graph.push_op(AnyMilliOp::Shape(MilliOpShape::new(tensor)));
    let rank = graph.push_op(AnyMilliOp::Shape(MilliOpShape::new(shape)));
    let axes_2 = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::add(
        axes, rank,
    )));
    graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::modulo(
        axes_2, rank, None,
    )))
}
