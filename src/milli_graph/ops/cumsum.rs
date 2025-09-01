use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::milli_graph::ops::MilliOp;
use crate::milli_graph::{MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_scalar::NumericScalarType;
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpCumSum {
    input: MilliOpGraphTensorId,
    axis: MilliOpGraphTensorId,
    exclusive: bool,
    reverse: bool,
}

impl MilliOpCumSum {
    pub(crate) fn new(
        input: MilliOpGraphTensorId,
        axis: MilliOpGraphTensorId,
        exclusive: bool,
        reverse: bool,
    ) -> Self {
        Self {
            input,
            axis,
            exclusive,
            reverse,
        }
    }
}

impl MilliOp for MilliOpCumSum {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.input, self.axis]
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let data = &inputs[&self.input];
        let axis = i64::cast_from_numeric_scalar(&inputs[&self.axis].first_element());
        let out = data.cumsum(Some(axis as isize), self.exclusive, self.reverse, backend)?;
        Ok(out)
    }

    fn get_name(&self) -> String {
        "CumSum".to_string()
    }
}
