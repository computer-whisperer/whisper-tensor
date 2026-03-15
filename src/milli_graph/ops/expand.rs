use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::graph::GlobalId;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::numeric_tensor::NumericTensor;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Expand {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    input: GlobalId,
    shape: GlobalId,
}

impl Expand {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        input: GlobalId,
        shape: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new_with_label(graph, input, shape, None, rng)
    }

    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        input: GlobalId,
        shape: GlobalId,
        label: Option<String>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            label,
            output,
            input,
            shape,
        };
        graph.push_op(AnyMilliOp::Expand(node));
        output
    }
}

impl Expand {
    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.input, map);
        super::remap(&mut self.shape, map);
    }
}

impl crate::graph::Node for Expand {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Expand".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.input, self.shape].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
}

impl MilliOp for Expand {
    fn infer(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo>,
        symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<
        Box<dyn Iterator<Item = (GlobalId, crate::tensor_info::TensorInfo)>>,
        MilliOpGraphError,
    > {
        use crate::scalar_info::ScalarInfoTyped;
        use crate::symbolic_scalar::SymbolicScalarTyped;
        use crate::tensor_info::TensorInfo;

        let input_info = known_inputs
            .get(&self.input)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let shape_info = known_inputs
            .get(&self.shape)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        // If both inputs are concrete, delegate to eval
        if let (Some(input_num), Some(shape_num)) =
            (input_info.as_numeric(), shape_info.as_numeric())
        {
            let inputs = HashMap::from([
                (self.input, input_num.clone()),
                (self.shape, shape_num.clone()),
            ]);
            let out: Vec<_> = self
                .eval(&inputs, backend)?
                .map(|(id, t)| (id, TensorInfo::from(t)))
                .collect();
            return Ok(Box::new(out.into_iter()));
        }

        let first_elem = input_info.first_element();

        // If shape tensor is concrete, use its values for output shape
        if let Some(shape_num) = shape_info.as_numeric() {
            let shape_values: Vec<i64> = shape_num.try_to_rank::<P1>()?.try_into()?;
            let output_shape: Vec<ScalarInfoTyped<u64>> = shape_values
                .iter()
                .map(|&v| ScalarInfoTyped::Numeric(v as u64))
                .collect();

            // Like eval: broadcast with input shape (take max of each dim)
            if let Some(input_ranked) = input_info.as_ranked() {
                let input_shape = input_ranked.shape();
                let output_rank = output_shape.len().max(input_shape.len());
                let mut final_shape: Vec<ScalarInfoTyped<u64>> = Vec::new();
                for i in 0..output_rank {
                    let target_i = (i as i64 - output_rank as i64) + output_shape.len() as i64;
                    let input_i = (i as i64 - output_rank as i64) + input_shape.len() as i64;

                    let target_dim = if target_i >= 0 {
                        Some(output_shape[target_i as usize].clone())
                    } else {
                        None
                    };
                    let input_dim = if input_i >= 0 {
                        Some(input_shape[input_i as usize].clone())
                    } else {
                        None
                    };

                    let dim = match (target_dim, input_dim) {
                        (
                            Some(ScalarInfoTyped::Numeric(t)),
                            Some(ScalarInfoTyped::Numeric(inp)),
                        ) => ScalarInfoTyped::Numeric(t.max(inp)),
                        (Some(t), None) => t,
                        (None, Some(inp)) => inp,
                        (Some(ScalarInfoTyped::Numeric(t)), Some(_)) => {
                            // Target is known, input is symbolic -- use target
                            ScalarInfoTyped::Numeric(t)
                        }
                        (Some(_), Some(ScalarInfoTyped::Numeric(inp))) => {
                            ScalarInfoTyped::Numeric(inp)
                        }
                        _ => ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)),
                    };
                    final_shape.push(dim);
                }

                let out = TensorInfo::Ranked(crate::tensor_info::TensorInfoRanked::new(
                    first_elem,
                    final_shape,
                    symbolic_resolver,
                ));
                return Ok(Box::new([(self.output, out)].into_iter()));
            }

            // Input shape not known, but target shape is
            let out = TensorInfo::Ranked(crate::tensor_info::TensorInfoRanked::new(
                first_elem,
                output_shape,
                symbolic_resolver,
            ));
            return Ok(Box::new([(self.output, out)].into_iter()));
        }

        // Shape tensor not concrete. Try to get output rank from shape tensor length.
        if let Some(shape_ranked) = shape_info.as_ranked() {
            let shape_shape = shape_ranked.shape();
            if shape_shape.len() == 1
                && let Some(&output_rank) = shape_shape[0].as_numeric()
            {
                let out = TensorInfo::new_from_first_element_and_rank(
                    first_elem,
                    ScalarInfoTyped::Numeric(output_rank as u32),
                    symbolic_resolver,
                );
                return Ok(Box::new([(self.output, out)].into_iter()));
            }
        }

        // Fallback: propagate dtype only
        let out = TensorInfo::new_from_first_element_and_rank(
            first_elem,
            ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)),
            symbolic_resolver,
        );
        Ok(Box::new([(self.output, out)].into_iter()))
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        _backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
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
        let out = x.expand(&shape_u)?;
        Ok(Box::new([(self.output, out)].into_iter()))
    }
}
