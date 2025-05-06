use crate::ndarray_backend::conversions::NDArrayNumericTensorType;
use crate::ndarray_backend::NDArrayNumericTensor;
use super::milli_ops::*;

pub(crate) fn rank(graph: &mut MilliOpGraph, tensor: MilliOpGraphTensorId) -> MilliOpGraphTensorId {
    let shape = graph.push_op(AnyMilliOp::Shape(MilliOpShape::new(tensor)));
    graph.push_op(AnyMilliOp::Shape(MilliOpShape::new(shape)))
}

pub(crate) fn scalar_const<T>(graph: &mut MilliOpGraph, value: T) -> MilliOpGraphTensorId
where
    T:NDArrayNumericTensorType
{
    graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(value)))
}
