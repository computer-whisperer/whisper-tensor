use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Slice {
    output: MilliOpGraphTensorId,
    data: MilliOpGraphTensorId,
    starts: MilliOpGraphTensorId,
    ends: MilliOpGraphTensorId,
    steps: Option<MilliOpGraphTensorId>,
    axes: Option<MilliOpGraphTensorId>,
}

impl Slice {
    pub fn new<T: std::hash::Hash + Clone + Eq>(
        graph: &mut MilliOpGraph<T>,
        data: MilliOpGraphTensorId,
        starts: MilliOpGraphTensorId,
        ends: MilliOpGraphTensorId,
        steps: Option<MilliOpGraphTensorId>,
        axes: Option<MilliOpGraphTensorId>,
    ) -> MilliOpGraphTensorId {
        let output = graph.get_new_tensor_id();
        let node = Self {
            output,
            data,
            starts,
            ends,
            steps,
            axes,
        };
        graph.push_op(AnyMilliOp::Slice(node));
        output
    }
}

impl crate::graph::Node<MilliOpGraphTensorId> for Slice {
    fn inputs(&self) -> impl Iterator<Item=MilliOpGraphTensorId> {
        let mut res = vec![self.data, self.starts, self.ends];
        if let Some(steps) = &self.steps { res.push(*steps); }
        if let Some(axes) = &self.axes { res.push(*axes); }
        res.into_iter()
    }
    fn outputs(&self) -> impl Iterator<Item=MilliOpGraphTensorId> { vec![self.output].into_iter() }
}

impl MilliOp for Slice {
    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<impl Iterator<Item=(MilliOpGraphTensorId, NumericTensor<DynRank>)>, MilliOpGraphError> {
        let data_input = &inputs[&self.data];
        let input_shape = data_input.shape();
        let input_rank = data_input.rank();
        let axes: Vec<i64> = if let Some(axes) = &self.axes {
            inputs[axes]
                .cast(DType::I64, backend)?
                .try_to_rank::<P1>()?
                .try_into()?
        } else {
            (0i64..(input_rank as i64)).collect()
        };
        let steps: Vec<i64> = if let Some(steps) = &self.steps {
            inputs[steps]
                .cast(DType::I64, backend)?
                .try_to_rank::<P1>()?
                .try_into()?
        } else {
            axes.iter().map(|_| 1).collect()
        };
        let starts: Vec<i64> = inputs[&self.starts]
            .cast(DType::I64, backend)?
            .try_to_rank::<P1>()?
            .try_into()?;
        let ends: Vec<i64> = inputs[&self.ends]
            .cast(DType::I64, backend)?
            .try_to_rank::<P1>()?
            .try_into()?;
        let mut output_slice = vec![];
        for &dim in &input_shape {
            output_slice.push(0..dim);
        }
        for (i, axis) in axes.into_iter().enumerate() {
            let axis = if axis < 0 {
                (input_rank as i64 + axis) as usize
            } else {
                axis as usize
            };
            let step = steps[i];
            if step != 1 {
                return Err(MilliOpGraphError::InvalidInput(format!(
                    "Step {step} is not supported"
                )));
            }

            let start = (if starts[i] < 0 {
                input_shape[axis] as i64 + starts[i]
            } else {
                starts[i]
            })
            .min(input_shape[axis] as i64)
            .max(0) as u64;

            let end = (if ends[i] < 0 {
                input_shape[axis] as i64 + ends[i]
            } else {
                ends[i]
            })
            .min(input_shape[axis] as i64)
            .max(0) as u64;
            output_slice[axis] = start..end;
        }
        let output = data_input.slice(&output_slice, backend)?;
        Ok([(self.output, output)].into_iter())
    }

    fn get_name(&self) -> String {
        "Slice".to_string()
    }
}
