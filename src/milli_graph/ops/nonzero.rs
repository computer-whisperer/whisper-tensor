use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::milli_graph::ops::MilliOp;
use crate::milli_graph::{MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use crate::scalar_info::ScalarInfo;
use crate::symbolic_scalar::{SymbolicResolver, SymbolicScalar, SymbolicScalarTyped};
use crate::tensor_info::{MinimalTensor, TensorInfo};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpNonZero {
    input: MilliOpGraphTensorId,
}

impl MilliOpNonZero {
    pub fn new(input: MilliOpGraphTensorId) -> Self {
        Self { input }
    }
}

impl MilliOp for MilliOpNonZero {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![self.input]
    }

    fn infer(
        &self,
        known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>,
        symbolic_resolver: &mut SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<TensorInfo, MilliOpGraphError> {
        let input = &known_inputs[&self.input];
        if let Some(x) = input.as_numeric() {
            let inputs = HashMap::from([(self.input, x.clone())]);
            Ok(TensorInfo::from(self.eval(&inputs, backend)?))
        } else {
            // Don't even try shape inference for now
            Ok(TensorInfo::Minimal(MinimalTensor::new(
                ScalarInfo::Symbolic(SymbolicScalar::new(DType::I64, symbolic_resolver)),
                SymbolicScalarTyped::new(symbolic_resolver),
            )))
        }
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        Ok(inputs[&self.input].nonzero(backend)?)
    }

    fn get_name(&self) -> String {
        "NonZero".to_string()
    }
}
