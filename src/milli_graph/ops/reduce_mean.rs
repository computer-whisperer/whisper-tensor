use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::graph::Node;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReduceMean {
    output: MilliOpGraphTensorId,
    data: MilliOpGraphTensorId,
    axes: Option<MilliOpGraphTensorId>,
    keepdims: bool,
    noop_with_empty_axes: bool,
}

impl ReduceMean {
    pub fn new<T: std::hash::Hash + Clone + Eq>(
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
        graph.push_op(AnyMilliOp::ReduceMean(node));
        output
    }
}

impl Node<MilliOpGraphTensorId> for ReduceMean {
    fn inputs(&self) -> impl Iterator<Item = MilliOpGraphTensorId> {
        match self.axes {
            Some(ax) => vec![self.data, ax].into_iter(),
            None => vec![self.data].into_iter(),
        }
    }
    fn outputs(&self) -> impl Iterator<Item = MilliOpGraphTensorId> {
        vec![self.output].into_iter()
    }
}

impl MilliOp for ReduceMean {
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<
        impl Iterator<Item = (MilliOpGraphTensorId, NumericTensor<DynRank>)>,
        MilliOpGraphError,
    > {
        let data = &inputs[&self.data];
        let axes = if let Some(axes) = self.axes {
            Vec::<i64>::try_from(inputs[&axes].try_to_rank::<P1>()?)?
        } else {
            (0i64..(data.rank() as i64)).collect()
        };
        let axes = if axes.is_empty() {
            if self.noop_with_empty_axes {
                let out_tensor = data.clone();
                return Ok([(self.output, out_tensor)].into_iter());
            } else {
                (0i64..(data.rank() as i64)).collect::<Vec<_>>()
            }
        } else {
            axes
        };
        let axes = axes
            .into_iter()
            .map(|x| (if x < 0 { x + data.rank() as i64 } else { x }) as usize)
            .collect::<Vec<_>>();
        let out = data.reduce_mean(axes, self.keepdims, backend)?;
        Ok([(self.output, out)].into_iter())
    }

    fn get_name(&self) -> String {
        "ReduceMean".to_string()
    }
}
