use crate::pool::Pool;
use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::migration::numeric_tensor::NumericTensor;
use crate::scalar_info::ScalarInfo;
use crate::symbolic_scalar::{SymbolicResolver, SymbolicScalar, SymbolicScalarTyped};
use crate::tensor_info::{MinimalTensor, TensorInfo};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NonZero {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    input: GlobalId,
}

impl NonZero {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        input: GlobalId,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        Self::push_new_with_label(graph, input, None, rng)
    }

    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        input: GlobalId,
        label: Option<String>,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            output,
            input,
            global_id: GlobalId::new(rng),
            label,
        };
        graph.push_op(AnyMilliOp::NonZero(node));
        output
    }
}

impl NonZero {
    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.input, map);
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
    fn infer<'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, TensorInfo<'p, P>>,
        symbolic_resolver: &mut SymbolicResolver,
        pool: &'p P,
    ) -> Result<Vec<(GlobalId, TensorInfo<'p, P>)>, MilliOpGraphError> {
        if let Some(input) = known_inputs.get(&self.input).and_then(|ti| ti.as_numeric()) {
            let inputs = HashMap::from([(self.input, input.clone())]);
            let out = self
                .eval(&inputs, &super::MilliEvalConfig::default(), &mut crate::backends::eval_backend::EvalBackend::NDArray)?
                .map(|(tid, t)| (tid, TensorInfo::from_legacy(&t, pool)))
                .collect::<Vec<_>>();
            return Ok(out);
        }
        // Fallback minimal info if unknown: dtype I64 vector of unknown size
        let minimal = TensorInfo::Minimal(MinimalTensor::new(
            ScalarInfo::Symbolic(SymbolicScalar::new(
                crate::numeric_dtype::NumericDType::from_legacy(DType::I64).unwrap(),
                symbolic_resolver,
            )),
            SymbolicScalarTyped::new(symbolic_resolver),
        ));
        let v: Vec<(GlobalId, TensorInfo<'p, P>)> = vec![(self.output, minimal)];
        Ok(v)
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        _config: &super::MilliEvalConfig,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        let out = inputs[&self.input].nonzero(backend)?;
        Ok(Box::new([(self.output, out)].into_iter()))
    }
}
