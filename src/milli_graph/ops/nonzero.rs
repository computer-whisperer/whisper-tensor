use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError, MilliOpGraphTensorId};
use crate::numeric_tensor::NumericTensor;
use crate::scalar_info::ScalarInfo;
use crate::symbolic_scalar::{SymbolicResolver, SymbolicScalar, SymbolicScalarTyped};
use crate::tensor_info::{MinimalTensor, TensorInfo};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::graph::Node;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilliOpNonZero {
    output: MilliOpGraphTensorId,
    input: MilliOpGraphTensorId,
}

impl MilliOpNonZero {
    pub fn new<T: std::hash::Hash + Clone + Eq>(graph: &mut MilliOpGraph<T>, input: MilliOpGraphTensorId) -> MilliOpGraphTensorId {
        let output = graph.get_new_tensor_id();
        let node = Self { output, input };
        graph.push_op(AnyMilliOp::NonZero(node));
        output
    }
}

impl Node<MilliOpGraphTensorId> for MilliOpNonZero {
    fn inputs(&self) -> impl Iterator<Item=MilliOpGraphTensorId> { vec![self.input].into_iter() }
    fn outputs(&self) -> impl Iterator<Item=MilliOpGraphTensorId> { vec![self.output].into_iter() }
}

impl MilliOp for MilliOpNonZero {
    fn infer(
        &self,
        known_inputs: &HashMap<MilliOpGraphTensorId, TensorInfo>,
        symbolic_resolver: &mut SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<impl Iterator<Item=(MilliOpGraphTensorId, TensorInfo)>, MilliOpGraphError> {
        if let Some(input) = known_inputs.get(&self.input).and_then(|ti| ti.as_numeric()) {
            let inputs = HashMap::from([(self.input, input.clone())]);
            let out = self.eval(&inputs, backend)?
                .map(|(tid, t)| (tid, TensorInfo::from(t)))
                .collect::<Vec<_>>();
            return Ok(out.into_iter());
        }
        // Fallback minimal info if unknown: dtype I64 vector of unknown size
        let minimal = TensorInfo::Minimal(MinimalTensor::new(
            ScalarInfo::Symbolic(SymbolicScalar::new(DType::I64, symbolic_resolver)),
            SymbolicScalarTyped::new(symbolic_resolver),
        ));
        let v: Vec<(MilliOpGraphTensorId, TensorInfo)> = vec![(self.output, minimal)];
        Ok(v.into_iter())
    }

    fn eval(
        &self,
        inputs: &HashMap<MilliOpGraphTensorId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<impl Iterator<Item=(MilliOpGraphTensorId, NumericTensor<DynRank>)>, MilliOpGraphError> {
        let out = inputs[&self.input].nonzero(backend)?;
        Ok([(self.output, out)].into_iter())
    }

    fn get_name(&self) -> String {
        "NonZero".to_string()
    }
}
