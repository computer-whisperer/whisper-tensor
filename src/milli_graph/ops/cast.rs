use crate::pool::Pool;
use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::graph::GlobalId;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::migration::numeric_tensor::NumericTensor;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cast {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    data: GlobalId,
    dtype: DType,
}

impl Cast {
    #[allow(dead_code)] // used by compiler (cranelift feature)
    pub(crate) fn target_dtype(&self) -> DType {
        self.dtype
    }

    pub fn push_new(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        dtype: DType,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new_with_label(graph, data, dtype, None, rng)
    }

    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        dtype: DType,
        label: Option<String>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            label,
            output,
            data,
            dtype,
        };
        graph.push_op(AnyMilliOp::Cast(node));
        output
    }
}

impl Cast {
    pub fn lower_to_nano(&self, ctx: &mut crate::nano_graph::NanoLoweringContext) {
        ctx.lower_identity_passthrough(self);
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.data, map);
    }
}

impl crate::graph::Node for Cast {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Cast".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.data].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
}

impl MilliOp for Cast {
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

        let input_info = known_inputs
            .get(&self.data)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        // If input is concrete, fall back to eval.
        if let Some(results) = super::constant_fold(self, known_inputs, pool) {
            return Ok(results);
        }

        // Same shape, new dtype. Preserve per-dim shape info.
        let out_ndt = crate::numeric_dtype::NumericDType::from_legacy(self.dtype)
            .expect("unsupported Cast target dtype");
        if let Some(ranked) = input_info.as_ranked() {
            let dims = ranked.shape();
            let out_info = TensorInfo::from_dtype_and_shape_scalars(out_ndt, &dims);
            return Ok(vec![((self.output, out_info))]);
        }

        let first_elem = crate::scalar_info::ScalarInfo::Symbolic(
            crate::symbolic_scalar::SymbolicScalar::new(out_ndt, symbolic_resolver),
        );
        let out_info = TensorInfo::new_from_first_element_and_rank(
            first_elem,
            input_info.rank(),
            symbolic_resolver,
        );
        Ok(vec![((self.output, out_info))])
    }

    fn backward(
        &self,
        output_grads: &HashMap<GlobalId, GlobalId>,
        graph: &mut MilliOpGraph,
        rng: &mut impl Rng,
    ) -> Option<HashMap<GlobalId, GlobalId>> {
        let grad_output = *output_grads.get(&self.output)?;
        // Cast gradient back to match the original input dtype
        let grad_input = super::CastLike::push_new(graph, grad_output, self.data, rng);
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
        let out = inputs[&self.data].cast(self.dtype, backend)?;
        Ok(Box::new([(self.output, out)].into_iter()))
    }
}
