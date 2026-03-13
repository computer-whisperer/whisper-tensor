use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shape {
    global_id: GlobalId,
    output: GlobalId,
    input: GlobalId,
}

impl Shape {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        input: GlobalId,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            output,
            input,
            global_id: GlobalId::new(rng),
        };
        graph.push_op(AnyMilliOp::Shape(node));
        output
    }
}

impl Shape {
    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.input, map);
    }
}

impl MilliOp for Shape {
    fn infer(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo>,
        symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<
        Box<dyn Iterator<Item = (GlobalId, crate::tensor_info::TensorInfo)>>,
        crate::milli_graph::MilliOpGraphError,
    > {
        use crate::tensor_info::TensorInfo;

        let input_info = known_inputs
            .get(&self.input)
            .ok_or(crate::milli_graph::MilliOpGraphError::UnableToInfer)?;

        // If input is concrete, fall back to eval for exact shape.
        if input_info.as_numeric().is_some() {
            let mut resolved = HashMap::new();
            resolved.insert(self.input, input_info.as_numeric().unwrap().clone());
            let collected: Vec<(GlobalId, TensorInfo)> = self
                .eval(&resolved, backend)?
                .map(|(a, b)| (a, TensorInfo::from(b)))
                .collect();
            return Ok(Box::new(collected.into_iter()));
        }

        // Shape op returns a 1-D i64 tensor with the input's dim values.
        // If input has known rank, try to extract concrete dim values.
        if let Some(rank) = input_info.rank_if_known() {
            let mut all_known = true;
            let mut dim_vals = Vec::with_capacity(rank);
            for i in 0..rank {
                if let Some(v) = input_info.dim_if_known(i) {
                    dim_vals.push(v as i64);
                } else {
                    all_known = false;
                    break;
                }
            }
            if all_known {
                // All dims concrete — produce a Numeric tensor.
                let out: NumericTensor<DynRank> =
                    NDArrayNumericTensor::<P1>::from(dim_vals).to_dyn().into();
                return Ok(Box::new([(self.output, TensorInfo::from(out))].into_iter()));
            }
        }

        // Fallback: symbolic output with known rank=1.
        let first_elem =
            crate::scalar_info::ScalarInfo::Symbolic(crate::symbolic_scalar::SymbolicScalar::new(
                crate::dtype::DType::I64,
                symbolic_resolver,
            ));
        let out_info = TensorInfo::new_from_first_element_and_rank(
            first_elem,
            crate::scalar_info::ScalarInfoTyped::Numeric(1),
            symbolic_resolver,
        );

        Ok(Box::new([(self.output, out_info)].into_iter()))
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        _backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        let output_shape = inputs[&self.input]
            .shape()
            .into_iter()
            .map(|x| x as i64)
            .collect::<Vec<_>>();
        let out: NumericTensor<DynRank> = NDArrayNumericTensor::<P1>::from(output_shape)
            .to_dyn()
            .into();
        Ok(Box::new([(self.output, out)].into_iter()))
    }
}

impl Node for Shape {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Shape".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.input].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
}
