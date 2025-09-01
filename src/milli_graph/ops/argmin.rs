use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::milli_graph::ops::MilliOp;
use crate::milli_graph::{MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpArgMin {
    input: MilliOpGraphTensorId,
    axis: i64,
    keepdims: bool,
    select_last_index: bool,
}

impl MilliOpArgMin {
    pub fn new(
        input: MilliOpGraphTensorId,
        axis: i64,
        keepdims: bool,
        select_last_index: bool,
    ) -> Self {
        Self {
            input,
            axis,
            keepdims,
            select_last_index,
        }
    }
}

impl MilliOp for MilliOpArgMin {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.input]
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let input = inputs[&self.input].clone();
        let axis = if self.axis < 0 {
            input.shape().len() as i64 + self.axis
        } else {
            self.axis
        } as usize;
        let max = input.argmin(axis, self.keepdims, self.select_last_index, backend)?;
        Ok(max)
    }

    fn get_name(&self) -> String {
        "ArgMin".to_string()
    }
}
