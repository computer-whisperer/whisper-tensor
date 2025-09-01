use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::milli_graph::ops::MilliOp;
use crate::milli_graph::{MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpCast {
    data: MilliOpGraphTensorId,
    dtype: DType,
}

impl MilliOpCast {
    pub fn new(data: MilliOpGraphTensorId, dtype: DType) -> Self {
        Self { data, dtype }
    }
}

impl MilliOp for MilliOpCast {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.data]
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        Ok(inputs[&self.data].cast(self.dtype, backend)?)
    }

    fn get_name(&self) -> String {
        "Cast".to_string()
    }
}
