use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::backends::ndarray_backend::conversions::NDArrayNumericTensorType;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{
    MilliOpGraph, MilliOpGraphError,
};
use crate::numeric_scalar::NumericScalar;
use crate::numeric_tensor::NumericTensor;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rand::Rng;
use typenum::P1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constant {
    global_id: GlobalId,
    output: GlobalId,
    data: NDArrayNumericTensor<DynRank>,
}

impl Constant {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        a: NDArrayNumericTensor<DynRank>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let node = Self {
            global_id: GlobalId::new(rng),
            output: graph.get_new_tensor_id(rng),
            data: a,
        };
        let out = node.output;
        graph.push_op(AnyMilliOp::Constant(node));
        out
    }

    pub(crate) fn new_scalar<T>(
        graph: &mut MilliOpGraph,
        v: T,
        rng: &mut impl Rng,
    ) -> GlobalId
    where
        T: NDArrayNumericTensorType,
    {
        let data = NDArrayNumericTensor::<DynRank>::from_vec_shape(vec![v], &vec![1]).unwrap();
        let node = Self {
            global_id: GlobalId::new(rng),
            output: graph.get_new_tensor_id(rng),
            data,
        };
        let out = node.output;
        graph.push_op(AnyMilliOp::Constant(node));
        out
    }
}

impl Node for Constant {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Constant".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::empty())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
}

impl MilliOp for Constant {
    fn eval(
        &self,
        _inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        _backend: &mut EvalBackend,
    ) -> Result<
        Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>,
        MilliOpGraphError,
    > {
        Ok(Box::new(
            [(self.output, self.data.clone().into())].into_iter(),
        ))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantOfShape {
    global_id: GlobalId,
    output: GlobalId,
    value: NumericScalar,
    shape: GlobalId,
}

impl ConstantOfShape {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        value: NumericScalar,
        shape: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let node = Self {
            global_id: GlobalId::new(rng),
            output: graph.get_new_tensor_id(rng),
            value,
            shape,
        };
        graph.push_op(AnyMilliOp::ConstantOfShape(node))
    }
}

impl Node for ConstantOfShape {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "ConstantOfShape".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.shape].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
}

impl MilliOp for ConstantOfShape {
    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        _backend: &mut EvalBackend,
    ) -> Result<
        Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>,
        MilliOpGraphError,
    > {
        let shape: Vec<i64> = inputs[&self.shape].try_to_rank::<P1>()?.try_into()?;
        let shape_usize = shape.iter().map(|x| *x as u64).collect::<Vec<_>>();
        let out: NumericTensor<DynRank> =
            NDArrayNumericTensor::<DynRank>::fill(self.value.clone(), &shape_usize)?.into();
        Ok(Box::new([(self.output, out)].into_iter()))
    }
}
