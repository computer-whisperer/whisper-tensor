use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::graph::GlobalId;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Slice {
    global_id: crate::graph::GlobalId,
    output: GlobalId,
    data: GlobalId,
    starts: GlobalId,
    ends: GlobalId,
    steps: Option<GlobalId>,
    axes: Option<GlobalId>,
}

impl Slice {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        starts: GlobalId,
        ends: GlobalId,
        steps: Option<GlobalId>,
        axes: Option<GlobalId>,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            output,
            data,
            starts,
            ends,
            steps,
            axes,
            global_id: GlobalId::new(rng),
        };
        graph.push_op(AnyMilliOp::Slice(node));
        output
    }
}

impl Slice {
    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.data, map);
        super::remap(&mut self.starts, map);
        super::remap(&mut self.ends, map);
        super::remap_opt(&mut self.steps, map);
        super::remap_opt(&mut self.axes, map);
    }
}

impl crate::graph::Node for Slice {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Slice".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        let mut res = vec![self.data, self.starts, self.ends];
        if let Some(steps) = &self.steps {
            res.push(*steps);
        }
        if let Some(axes) = &self.axes {
            res.push(*axes);
        }
        Box::new(res.into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new([self.output].into_iter())
    }
}

impl MilliOp for Slice {
    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
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
        Ok(Box::new([(self.output, output)].into_iter()))
    }
}
