use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::graph::Node;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReduceSum {
    output: MilliOpGraphTensorId,
    data: MilliOpGraphTensorId,
    axes: Option<MilliOpGraphTensorId>,
    keepdims: bool,
    noop_with_empty_axes: bool,
}

impl ReduceSum {
    pub fn push_new<T: std::hash::Hash + Clone + Eq + 'static>(
        graph: &mut MilliOpGraph<T>,
        data: MilliOpGraphTensorId,
        axes: Option<MilliOpGraphTensorId>,
        keepdims: bool,
        noop_with_empty_axes: bool,
    ) -> MilliOpGraphTensorId {
        let output = graph.get_new_tensor_id();
        let node = Self {
            output,
            data,
            axes,
            keepdims,
            noop_with_empty_axes,
        };
        graph.push_op(AnyMilliOp::ReduceSum(node));
        output
    }
}

impl MilliOp for ReduceSum {
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<
        Box<dyn Iterator<Item = (MilliOpGraphTensorId, NumericTensor<DynRank>)>>,
        MilliOpGraphError,
    > {
        let data = &inputs[&self.data];
        let axes = if let Some(axes) = self.axes {
            Vec::<i64>::try_from(inputs[&axes].try_to_rank::<P1>()?)?
        } else {
            (0i64..(data.shape().len() as i64)).collect()
        };
        let axes = if axes.is_empty() {
            if self.noop_with_empty_axes {
                // return identity
                let out_tensor = data.clone();
                return Ok(Box::new([(self.output, out_tensor)].into_iter()));
            } else {
                (0i64..(data.shape().len() as i64)).collect::<Vec<_>>()
            }
        } else {
            axes
        };
        let axes = axes
            .into_iter()
            .map(|x| {
                (if x < 0 {
                    x + data.shape().len() as i64
                } else {
                    x
                }) as usize
            })
            .collect::<Vec<_>>();
        let data_cast = match data.dtype() {
            DType::BF16 | DType::F16 => data.cast(DType::F32, backend)?,
            _ => data.clone(),
        };
        let out = data_cast.reduce_sum(axes, self.keepdims, backend)?;
        let out = if out.dtype() != data.dtype() {
            out.cast(data.dtype(), backend)?
        } else {
            out
        };

        Ok(Box::new([(self.output, out)].into_iter()))
    }
}

impl Node<MilliOpGraphTensorId> for ReduceSum {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "ReduceSum".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = MilliOpGraphTensorId>> {
        let it: Box<dyn Iterator<Item = MilliOpGraphTensorId>> = match self.axes {
            Some(ax) => Box::new(vec![self.data, ax].into_iter()),
            None => Box::new(vec![self.data].into_iter()),
        };
        it
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = MilliOpGraphTensorId>> {
        Box::new([self.output].into_iter())
    }
}
