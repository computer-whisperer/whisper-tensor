use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError, MilliOpGraphNodeId, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;
use crate::graph::Node;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpReduceSum {
    output: MilliOpGraphTensorId,
    data: MilliOpGraphTensorId,
    axes: Option<MilliOpGraphTensorId>,
    keepdims: bool,
    noop_with_empty_axes: bool,
}

impl MilliOpReduceSum {
    pub fn new<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        data: MilliOpGraphTensorId,
        axes: Option<MilliOpGraphTensorId>,
        keepdims: bool,
        noop_with_empty_axes: bool,
    ) -> MilliOpGraphNodeId {
        let node = Self {
            output: graph.get_new_tensor_id(),
            data,
            axes,
            keepdims,
            noop_with_empty_axes,
        };
        graph.push_op(AnyMilliOp::ReduceSum(node))
    }
}

impl MilliOp for MilliOpReduceSum {
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<impl Iterator<Item=(MilliOpGraphTensorId, NumericTensor<DynRank>)>, MilliOpGraphError> {
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
                return Ok([(self.output, out_tensor)].into_iter());
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

        Ok([(self.output, out)].into_iter())
    }

    fn get_name(&self) -> String {
        "ReduceSum".to_string()
    }
}

impl Node<MilliOpGraphTensorId> for MilliOpReduceSum {
    fn inputs(&self) -> impl Iterator<Item=MilliOpGraphTensorId> {
        match self.axes {
            Some(ax) => vec![self.data, ax].into_iter(),
            None => vec![self.data].into_iter(),
        }
    }
    fn outputs(&self) -> impl Iterator<Item=MilliOpGraphTensorId> {
        [self.output].into_iter()
    }
}
