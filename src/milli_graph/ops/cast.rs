use crate::graph::GlobalId;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::numeric_dtype::NumericDType;
use crate::pool::Pool;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cast {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    data: GlobalId,
    dtype: NumericDType,
    /// When true, overflow clamps to ±max_finite instead of ±inf.
    /// ONNX Cast defaults to true for float8 targets.
    #[serde(default = "default_saturate")]
    pub(crate) saturate: bool,
}

fn default_saturate() -> bool {
    true
}

impl Cast {
    #[allow(dead_code)] // used by compiler (cranelift feature)
    pub(crate) fn target_dtype(&self) -> NumericDType {
        self.dtype
    }

    pub fn push_new(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        dtype: NumericDType,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new_with_options(graph, data, dtype, true, None, rng)
    }

    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        dtype: NumericDType,
        label: Option<String>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new_with_options(graph, data, dtype, true, label, rng)
    }

    pub fn push_new_with_options(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        dtype: NumericDType,
        saturate: bool,
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
            saturate,
        };
        graph.push_op(AnyMilliOp::Cast(node));
        output
    }
}

impl Cast {
    pub fn lower_to_nano<'p, P: crate::pool::Pool + 'p>(
        &self,
        ctx: &mut crate::nano_graph::NanoLoweringContext<'_, 'p, P>,
    ) -> crate::milli_graph::ops::LowerResult {
        ctx.lower_cast_passthrough(self, self.saturate)
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
    fn infer<'a, 'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'a, 'p, P>>,
        symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        pool: &'p P,
    ) -> Result<Vec<(GlobalId, crate::tensor_info::TensorInfo<'a, 'p, P>)>, MilliOpGraphError>
    where
        'p: 'a,
    {
        use crate::tensor_info::TensorInfo;

        let input_info = known_inputs
            .get(&self.data)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        // Same shape, new dtype. Preserve per-dim shape info.
        let out_ndt = self.dtype;
        let out_info = if let Some(ranked) = input_info.as_ranked() {
            let dims = ranked.shape();
            TensorInfo::from_dtype_and_shape_scalars(out_ndt, &dims)
        } else {
            let first_elem = crate::scalar_info::ScalarInfo::Symbolic(
                crate::symbolic_scalar::SymbolicScalar::new(out_ndt, symbolic_resolver),
            );
            TensorInfo::new_from_first_element_and_rank(
                first_elem,
                input_info.rank(),
                symbolic_resolver,
            )
        };

        // If input is concrete, try constant fold via nano+pool_eval path.
        if let Some(results) = super::constant_fold(
            self,
            known_inputs,
            &[(self.output, out_info.clone_with_pool(pool))],
            pool,
        ) {
            return Ok(results);
        }

        Ok(vec![(self.output, out_info)])
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

    fn eval_new<'p, P2: crate::pool::Pool + 'p>(
        &self,
        inputs: &[crate::numeric_tensor::NumericTensorView<'_, crate::tensor_rank::DynRank>],
        pool: &'p P2,
    ) -> Result<
        Vec<crate::numeric_tensor::NumericTensor<'p, crate::tensor_rank::DynRank, P2>>,
        crate::nano_graph::pool_eval::PoolEvalError,
    > {
        use crate::numeric_tensor::{NumericTensor, TensorLayout};
        use crate::tensor_rank::DynRank;

        let data = &inputs[0];
        let target_dtype = self.dtype;
        let shape = data.shape().clone();
        let numel = data.numel();
        let layout = TensorLayout::<DynRank>::row_major(shape, target_dtype);
        let buf = pool
            .allocate(layout.buffer_size_bytes())
            .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
        let mut out = NumericTensor::from_parts(buf, layout);
        for i in 0..numel {
            out.write_element(i, data.read_element(i).cast_to(target_dtype));
        }
        Ok(vec![out])
    }

    fn lower_to_nano<'p, P: crate::pool::Pool + 'p>(
        &self,
        ctx: &mut crate::nano_graph::NanoLoweringContext<'_, 'p, P>,
    ) -> crate::milli_graph::ops::LowerResult {
        Cast::lower_to_nano(self, ctx)
    }
}
