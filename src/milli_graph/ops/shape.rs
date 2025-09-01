use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::milli_graph::ops::MilliOp;
use crate::milli_graph::{MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use crate::symbolic_scalar::SymbolicResolver;
use crate::tensor_info::TensorInfo;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpShape {
    input: MilliOpGraphTensorId,
}

impl MilliOpShape {
    pub fn new(input: MilliOpGraphTensorId) -> Self {
        Self { input }
    }
}

impl MilliOp for MilliOpShape {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.input]
    }

    fn infer(
        &self,
        known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>,
        symbolic_resolver: &mut SymbolicResolver,
        _backend: &mut EvalBackend,
    ) -> Result<TensorInfo, MilliOpGraphError> {
        let input = &known_inputs[&self.input];
        Ok(TensorInfo::from(input.shape(symbolic_resolver)))
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        _backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        let output_shape = inputs[&self.input]
            .shape()
            .into_iter()
            .map(|x| x as i64)
            .collect::<Vec<_>>();
        Ok(NDArrayNumericTensor::<P1>::from(output_shape)
            .to_dyn()
            .into())
    }

    fn get_name(&self) -> String {
        "Shape".to_string()
    }
}
