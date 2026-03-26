use crate::pool::Pool;
use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::migration::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shape {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    input: GlobalId,
}

impl Shape {
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
        graph.push_op(AnyMilliOp::Shape(node));
        output
    }
}

impl Shape {
    pub fn lower_to_nano(&self, ctx: &mut crate::nano_graph::NanoLoweringContext) -> crate::milli_graph::ops::LowerResult {
        let out_id = self.output;
        if let Some(info) = ctx.all_infos.get(&out_id) {
            ctx.register_constant(out_id, info);
        } else {
            ctx.register_opaque(out_id);
        }
        crate::milli_graph::ops::LowerResult::Lowered
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.input, map);
    }
}

impl MilliOp for Shape {
    fn infer<'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'p, P>>,
        symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        pool: &'p P,
    ) -> Result<
        Vec<(GlobalId, crate::tensor_info::TensorInfo<'p, P>)>,
        crate::milli_graph::MilliOpGraphError,
    > {
        use crate::tensor_info::TensorInfo;

        let input_info = known_inputs
            .get(&self.input)
            .ok_or(crate::milli_graph::MilliOpGraphError::UnableToInfer)?;

        // Shape op returns a 1-D i64 tensor with the input's dim values.
        // If all dims are known, produce the result directly (no eval needed).
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
                return Ok(vec![((self.output, TensorInfo::from_legacy(&out, pool)))]);
            }
        }

        // Fallback: symbolic output with known rank=1.
        let first_elem =
            crate::scalar_info::ScalarInfo::Symbolic(crate::symbolic_scalar::SymbolicScalar::new(
                crate::numeric_dtype::NumericDType::from_legacy(crate::dtype::DType::I64).unwrap(),
                symbolic_resolver,
            ));
        let out_info = TensorInfo::new_from_first_element_and_rank(
            first_elem,
            crate::scalar_info::ScalarInfoTyped::Numeric(1),
            symbolic_resolver,
        );

        Ok(vec![((self.output, out_info))])
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        _config: &super::MilliEvalConfig,
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

    fn lower_to_nano(&self, ctx: &mut crate::nano_graph::NanoLoweringContext) -> crate::milli_graph::ops::LowerResult {
        Shape::lower_to_nano(self, ctx)
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
