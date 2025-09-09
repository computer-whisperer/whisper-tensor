use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::backends::ndarray_backend::conversions::NDArrayNumericTensorType;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError, MilliOpGraphNodeId, MilliOpGraphTensorId};
use crate::graph::Node;
use crate::numeric_scalar::NumericScalar;
use crate::numeric_tensor::NumericTensor;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constant {
    output: MilliOpGraphTensorId,
    data: NDArrayNumericTensor<DynRank>,
}

impl Constant {
    pub fn new<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, a: NDArrayNumericTensor<DynRank>) -> MilliOpGraphNodeId {
        let node = Self { output: graph.get_new_tensor_id(), data: a };
        graph.push_op(AnyMilliOp::Constant(node))
    }

    pub(crate) fn new_scalar<T, Io: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<Io>, v: T) -> MilliOpGraphNodeId
    where
        T: NDArrayNumericTensorType,
    {
        let data = NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![v], &vec![1]).unwrap();
        let node = Self { output: graph.get_new_tensor_id(), data };
        graph.push_op(AnyMilliOp::Constant(node))
    }
}

impl Node<MilliOpGraphTensorId> for Constant {
    fn inputs(&self) -> impl Iterator<Item=MilliOpGraphTensorId> { std::iter::empty() }
    fn outputs(&self) -> impl Iterator<Item=MilliOpGraphTensorId> { vec![self.output].into_iter() }
}

impl MilliOp for Constant {
    fn eval(
        &self,
        _inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        _backend: &mut EvalBackend,
    ) -> Result<impl Iterator<Item=(MilliOpGraphTensorId, NumericTensor<DynRank>)>, MilliOpGraphError> {
        Ok([(self.output, self.data.clone().into())].into_iter())
    }

    fn get_name(&self) -> String {
        "Constant".to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantOfShape {
    output: MilliOpGraphTensorId,
    value: NumericScalar,
    shape: MilliOpGraphTensorId,
}

impl ConstantOfShape {
    pub fn new<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, value: NumericScalar, shape: MilliOpGraphTensorId) -> MilliOpGraphNodeId {
        let node = Self { output: graph.get_new_tensor_id(), value, shape };
        graph.push_op(AnyMilliOp::ConstantOfShape(node))
    }
}

impl Node<MilliOpGraphTensorId> for ConstantOfShape {
    fn inputs(&self) -> impl Iterator<Item=MilliOpGraphTensorId> { vec![self.shape].into_iter() }
    fn outputs(&self) -> impl Iterator<Item=MilliOpGraphTensorId> { vec![self.output].into_iter() }
}

impl MilliOp for ConstantOfShape {
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        _backend: &mut EvalBackend,
    ) -> Result<impl Iterator<Item=(MilliOpGraphTensorId, NumericTensor<DynRank>)>, MilliOpGraphError> {
        let shape: Vec<i64> = inputs[&self.shape].try_to_rank::<P1>()?.try_into()?;
        let shape_usize = shape.iter().map(|x| *x as u64).collect::<Vec<_>>();
        let out: NumericTensor<DynRank> = NDArrayNumericTensor::<DynRank>::fill(self.value.clone(), &shape_usize)?.into();
        Ok([(self.output, out)].into_iter())
    }

    fn get_name(&self) -> String {
        "Constant of Shape".to_string()
    }
}
