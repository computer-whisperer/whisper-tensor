use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::dtype::DType;
use crate::graph::Node;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Squeeze {
    output: MilliOpGraphTensorId,
    data: MilliOpGraphTensorId,
    axes: MilliOpGraphTensorId,
}

impl Squeeze {
    pub fn push_new<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        data: MilliOpGraphTensorId,
        axes: MilliOpGraphTensorId,
    ) -> MilliOpGraphTensorId {
        let output = graph.get_new_tensor_id();
        let node = Self { output, data, axes };
        graph.push_op(AnyMilliOp::Squeeze(node));
        output
    }
}

impl Node<MilliOpGraphTensorId> for Squeeze {
    type OpKind = String;
    fn op_kind(&self) -> String {
        "Squeeze".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = MilliOpGraphTensorId>> {
        Box::new([self.data, self.axes].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = MilliOpGraphTensorId>> {
        Box::new([self.output].into_iter())
    }
}

impl MilliOp for Squeeze {
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<
        Box<dyn Iterator<Item = (MilliOpGraphTensorId, NumericTensor<DynRank>)>>,
        MilliOpGraphError,
    > {
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
            let mut output_shape = Vec::new();
            for i in 0..(input_shape.len() - axes.len()) {
                let mut is_selected = false;
                for axis in &axes {
                    let axis = if *axis < 0 {
                        input_shape.len() as i64 + *axis
                    } else {
                        *axis
                    };
                    if axis == i as i64 {
                        is_selected = true;
                        break;
                    }
                }
                if is_selected {
                    // Skip it
                } else {
                    output_shape.push(input_shape[i]);
                }
            }
            let output = inputs[&self.data].reshape(output_shape, backend)?;
            Ok(Box::new([(self.output, output)].into_iter()))
        }
    }
}
