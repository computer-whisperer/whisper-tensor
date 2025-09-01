use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::milli_graph::ops::MilliOp;
use crate::milli_graph::{MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpConcat {
    inputs: Vec<MilliOpGraphTensorId>,
    axis: i64,
}

impl MilliOpConcat {
    pub fn new(inputs: Vec<MilliOpGraphTensorId>, axis: i64) -> Self {
        Self { inputs, axis }
    }
}

impl MilliOp for MilliOpConcat {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        self.inputs.clone()
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let mut resolved_inputs = vec![];
        for input in &self.inputs {
            resolved_inputs.push(&inputs[input]);
        }
        let axis = if self.axis < 0 {
            resolved_inputs[0].shape().len() as i64 + self.axis
        } else {
            self.axis
        } as usize;
        Ok(NumericTensor::<DynRank>::concat(
            resolved_inputs.as_slice(),
            axis,
            backend,
        )?)
    }

    fn get_name(&self) -> String {
        "Concat".to_string()
    }
}
