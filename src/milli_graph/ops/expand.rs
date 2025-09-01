use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::milli_graph::ops::MilliOp;
use crate::milli_graph::{MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpExpand {
    input: MilliOpGraphTensorId,
    shape: MilliOpGraphTensorId,
}

impl MilliOpExpand {
    pub fn new(input: MilliOpGraphTensorId, shape: MilliOpGraphTensorId) -> Self {
        Self { input, shape }
    }
}

impl MilliOp for MilliOpExpand {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.input, self.shape]
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        _backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let shape: Vec<i64> = inputs[&self.shape].try_to_rank::<P1>()?.try_into()?;
        let shape_u = shape.iter().map(|x| *x as u64).collect::<Vec<u64>>();
        let mut x = inputs[&self.input].clone();
        while x.rank() < shape_u.len() {
            x = x.unsqueeze(0)?;
        }
        let shape_u = shape_u
            .iter()
            .zip(x.shape().iter())
            .map(|(a, b)| std::cmp::max(*a, *b))
            .collect::<Vec<u64>>();
        Ok(x.expand(&shape_u)?)
    }

    fn get_name(&self) -> String {
        "Expand".to_string()
    }
}
