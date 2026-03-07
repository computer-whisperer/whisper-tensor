use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReduceSum {
    global_id: GlobalId,
    output: GlobalId,
    data: GlobalId,
    axes: Option<GlobalId>,
    keepdims: bool,
    noop_with_empty_axes: bool,
}

impl ReduceSum {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        axes: Option<GlobalId>,
        keepdims: bool,
        noop_with_empty_axes: bool,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
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

impl ReduceSum {
    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.data, map);
        super::remap_opt(&mut self.axes, map);
    }
}

impl MilliOp for ReduceSum {
    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
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

    fn backward(
        &self,
        output_grads: &HashMap<GlobalId, GlobalId>,
        graph: &mut MilliOpGraph,
        rng: &mut impl rand::Rng,
    ) -> Option<HashMap<GlobalId, GlobalId>> {
        let grad_output = *output_grads.get(&self.output)?;
        // Backward of ReduceSum: expand grad_output back to input shape.
        // If keepdims=false, first unsqueeze along the reduced axes.
        let expanded_grad = if self.keepdims {
            grad_output
        } else if let Some(axes) = self.axes {
            // Unsqueeze along the specific reduced axes
            super::Unsqueeze::push_new(graph, grad_output, axes, rng)
        } else {
            // Reduced all axes to scalar — reshape to all-ones shape via unsqueeze
            // We don't know the rank, so use Reshape to [1,1,...] matching input shape
            // Actually, Expand handles broadcasting from scalar, just get input shape
            grad_output
        };
        // Expand to input shape
        let input_shape = super::Shape::push_new(graph, self.data, rng);
        let grad_input = super::Expand::push_new(graph, expanded_grad, input_shape, rng);
        let mut result = HashMap::new();
        result.insert(self.data, grad_input);
        Some(result)
    }
}

impl Node for ReduceSum {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "ReduceSum".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        let it: Box<dyn Iterator<Item = GlobalId>> = match self.axes {
            Some(ax) => Box::new(vec![self.data, ax].into_iter()),
            None => Box::new(vec![self.data].into_iter()),
        };
        it
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new([self.output].into_iter())
    }
}
