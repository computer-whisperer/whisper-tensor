use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::milli_graph::ops::MilliOp;
use crate::milli_graph::{MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use crate::scalar_info::{ScalarInfo, ScalarInfoTyped};
use crate::symbolic_scalar::{SymbolicResolver, SymbolicScalar, SymbolicScalarTyped};
use crate::tensor_info::{TensorInfo, TensorInfoRanked};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpRange {
    start: MilliOpGraphTensorId,
    end: MilliOpGraphTensorId,
    delta: MilliOpGraphTensorId,
}

impl MilliOpRange {
    pub fn new(
        start: MilliOpGraphTensorId,
        end: MilliOpGraphTensorId,
        delta: MilliOpGraphTensorId,
    ) -> Self {
        Self { start, end, delta }
    }
}

impl MilliOp for MilliOpRange {
    fn get_inputs(&self) -> Vec<MilliOpGraphTensorId> {
        vec![]
    }

    fn infer(
        &self,
        known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>,
        symbolic_resolver: &mut SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<TensorInfo, MilliOpGraphError> {
        let start = &known_inputs[&self.start].first_element();
        let end = &known_inputs[&self.end].first_element();
        let delta = &known_inputs[&self.delta].first_element();
        assert_eq!(start.dtype(), end.dtype());
        assert_eq!(start.dtype(), end.dtype());
        Ok(
            if let (
                ScalarInfo::Numeric(start),
                ScalarInfo::Numeric(end),
                ScalarInfo::Numeric(delta),
            ) = (start, end, delta)
            {
                // We have enough info, so just resolve it
                TensorInfo::from(NumericTensor::<P1>::range(
                    start.clone(),
                    end.clone(),
                    delta.clone(),
                    backend,
                )?)
            } else {
                TensorInfo::from(TensorInfoRanked::<DynRank>::new(
                    ScalarInfo::Symbolic(SymbolicScalar::new(DType::I64, symbolic_resolver)),
                    vec![ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(
                        symbolic_resolver,
                    ))],
                    symbolic_resolver,
                ))
            },
        )
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<NumericTensor<DynRank>, MilliOpGraphError> {
        Ok(NumericTensor::<P1>::range(
            inputs[&self.start].first_element(),
            inputs[&self.end].first_element(),
            inputs[&self.delta].first_element(),
            backend,
        )?
        .to_dyn_rank())
    }

    fn get_name(&self) -> String {
        "Range".to_string()
    }
}
