use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::numeric_tensor::NumericTensor;
use crate::scalar_info::ScalarInfo;
use crate::symbolic_scalar::{SymbolicResolver, SymbolicScalar, SymbolicScalarTyped};
use crate::tensor_info::{MinimalTensor, TensorInfo};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NonZero {
    global_id: GlobalId,
    output: GlobalId,
    input: GlobalId,
}

impl NonZero {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        input: GlobalId,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self { output, input, global_id: GlobalId::new(rng) };
        graph.push_op(AnyMilliOp::NonZero(node));
        output
    }
}

impl Node for NonZero {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "NonZero".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.input].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
}

impl MilliOp for NonZero {
    fn infer(
        &self,
        known_inputs: &HashMap<GlobalId, TensorInfo>,
        symbolic_resolver: &mut SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, TensorInfo)>>, MilliOpGraphError>
    {
        if let Some(input) = known_inputs.get(&self.input).and_then(|ti| ti.as_numeric()) {
            let inputs = HashMap::from([(self.input, input.clone())]);
            let out = self
                .eval(&inputs, backend)?
                .map(|(tid, t)| (tid, TensorInfo::from(t)))
                .collect::<Vec<_>>();
            return Ok(Box::new(out.into_iter()));
        }
        // Fallback minimal info if unknown: dtype I64 vector of unknown size
        let minimal = TensorInfo::Minimal(MinimalTensor::new(
            ScalarInfo::Symbolic(SymbolicScalar::new(DType::I64, symbolic_resolver)),
            SymbolicScalarTyped::new(symbolic_resolver),
        ));
        let v: Vec<(GlobalId, TensorInfo)> = vec![(self.output, minimal)];
        Ok(Box::new(v.into_iter()))
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<
        Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>,
        MilliOpGraphError,
    > {
        let out = inputs[&self.input].nonzero(backend)?;
        Ok(Box::new([(self.output, out)].into_iter()))
    }
}
