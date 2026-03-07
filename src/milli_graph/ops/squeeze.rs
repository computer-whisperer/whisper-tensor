use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::dtype::DType;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Squeeze {
    global_id: GlobalId,
    output: GlobalId,
    data: GlobalId,
    axes: GlobalId,
}

impl Squeeze {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        axes: GlobalId,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            output,
            data,
            axes,
            global_id: GlobalId::new(rng),
        };
        graph.push_op(AnyMilliOp::Squeeze(node));
        output
    }
}

impl Squeeze {
    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.data, map);
        super::remap(&mut self.axes, map);
    }
}

impl Node for Squeeze {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> String {
        "Squeeze".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new([self.data, self.axes].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new([self.output].into_iter())
    }
}

impl MilliOp for Squeeze {
    fn backward(
        &self,
        output_grads: &HashMap<GlobalId, GlobalId>,
        graph: &mut MilliOpGraph,
        rng: &mut impl rand::Rng,
    ) -> Option<HashMap<GlobalId, GlobalId>> {
        let grad_output = *output_grads.get(&self.output)?;
        // Squeeze backward = Unsqueeze along the same axes
        let grad_input = super::Unsqueeze::push_new(graph, grad_output, self.axes, rng);
        let mut result = HashMap::new();
        result.insert(self.data, grad_input);
        Some(result)
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        let axes_ndarray = NDArrayNumericTensor::<DynRank>::try_from(
            inputs[&self.axes].cast(DType::I64, backend)?,
        )?;
        let axes = Vec::<i64>::try_from(axes_ndarray.try_to_rank::<P1>()?)?;
        if axes.len() == 1 {
            let axis = axes[0];
            let input_shape = inputs[&self.data].shape();
            let axis = if axis >= 0 {
                axis as usize
            } else {
                (input_shape.len() as i64 + axis) as usize
            };
            let output = inputs[&self.data].squeeze(axis)?;
            Ok(Box::new([(self.output, output)].into_iter()))
        } else {
            // Multiple axes (use reshape)
            let input_shape = inputs[&self.data].shape();
            let normalized_axes: Vec<usize> = axes
                .iter()
                .map(|&a| {
                    if a < 0 {
                        (input_shape.len() as i64 + a) as usize
                    } else {
                        a as usize
                    }
                })
                .collect();
            let mut output_shape = Vec::new();
            for (i, &dim) in input_shape.iter().enumerate() {
                if !normalized_axes.contains(&i) {
                    output_shape.push(dim);
                }
            }
            let output = inputs[&self.data].reshape(output_shape, backend)?;
            Ok(Box::new([(self.output, output)].into_iter()))
        }
    }
}
