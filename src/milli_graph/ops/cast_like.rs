use crate::pool::Pool;
use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::graph::GlobalId;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::migration::numeric_tensor::NumericTensor;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CastLike {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    data: GlobalId,
    target_type: GlobalId,
}

impl CastLike {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        target_type: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new_with_label(graph, data, target_type, None, rng)
    }

    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        target_type: GlobalId,
        label: Option<String>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            label,
            output,
            data,
            target_type,
        };
        graph.push_op(AnyMilliOp::CastLike(node));
        output
    }
}

impl CastLike {
    pub fn lower_to_nano(&self, ctx: &mut crate::nano_graph::NanoLoweringContext) {
        ctx.lower_identity_passthrough(self);
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.data, map);
        super::remap(&mut self.target_type, map);
    }
}

impl crate::graph::Node for CastLike {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "CastLike".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.data, self.target_type].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
}

impl MilliOp for CastLike {
    fn infer<'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'p, P>>,
        symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        pool: &'p P,
    ) -> Result<
        Vec<(GlobalId, crate::tensor_info::TensorInfo<'p, P>)>,
        MilliOpGraphError,
    > {
        use crate::tensor_info::TensorInfo;

        let data_info = known_inputs
            .get(&self.data)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let target_info = known_inputs
            .get(&self.target_type)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        // Same shape as data, dtype from target. Preserve per-dim shape info.
        let out_dtype = target_info.dtype();

        let out_info = if let Some(ranked) = data_info.as_ranked() {
            let dims = ranked.shape();
            TensorInfo::from_dtype_and_shape_scalars(out_dtype, &dims)
        } else {
            let first_elem = crate::scalar_info::ScalarInfo::Symbolic(
                crate::symbolic_scalar::SymbolicScalar::new(out_dtype, symbolic_resolver),
            );
            TensorInfo::new_from_first_element_and_rank(
                first_elem,
                data_info.rank(),
                symbolic_resolver,
            )
        };

        // If data is concrete and we know target dtype, try constant fold (eval-based fallback
        // for now; nano-based Cast lowering has a dtype size mismatch bug).
        if let Some(results) = super::constant_fold(self, known_inputs, &[], pool) {
            return Ok(results);
        }

        Ok(vec![((self.output, out_info))])
    }

    fn backward(
        &self,
        output_grads: &HashMap<GlobalId, GlobalId>,
        graph: &mut MilliOpGraph,
        rng: &mut impl Rng,
    ) -> Option<HashMap<GlobalId, GlobalId>> {
        let grad_output = *output_grads.get(&self.output)?;
        // Cast gradient back to match the original data input dtype
        let grad_input = CastLike::push_new(graph, grad_output, self.data, rng);
        let mut result = HashMap::new();
        result.insert(self.data, grad_input);
        Some(result)
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        _config: &super::MilliEvalConfig,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        let out = inputs[&self.data].cast(inputs[&self.target_type].dtype(), backend)?;
        Ok(Box::new([(self.output, out)].into_iter()))
    }

    fn lower_to_nano(&self, ctx: &mut crate::nano_graph::NanoLoweringContext) {
        CastLike::lower_to_nano(self, ctx);
    }
}
