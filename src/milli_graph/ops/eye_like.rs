use crate::graph::GlobalId;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::pool::Pool;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// EyeLike: produces an identity-like matrix (zeros with ones on a diagonal).
///
/// Takes a single input (used only for its shape — must be 2D).
/// Produces output with 1s on diagonal offset by `k`, 0s elsewhere.
///
/// Constant-folds at infer time when input shape is known.
/// Lowers to nano as literal constants (no runtime computation needed).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EyeLike {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    input: GlobalId,
    /// Diagonal offset: 0 = main diagonal, positive = above, negative = below.
    k: i64,
    /// Output dtype.
    output_dtype: crate::numeric_dtype::NumericDType,
}

impl EyeLike {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        input: GlobalId,
        k: i64,
        output_dtype: crate::numeric_dtype::NumericDType,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            label: None,
            output,
            input,
            k,
            output_dtype,
        };
        graph.push_op(AnyMilliOp::EyeLike(node));
        output
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.input, map);
    }

    /// Build the concrete eye tensor given rows, cols, k, dtype.
    fn build_tensor<'p, P: Pool + 'p>(
        rows: usize,
        cols: usize,
        k: i64,
        dtype: crate::numeric_dtype::NumericDType,
        pool: &'p P,
    ) -> crate::numeric_tensor::NumericTensor<'p, crate::tensor_rank::DynRank, P> {
        use crate::numeric_scalar::NumericScalar;
        use crate::numeric_tensor::{NumericTensor, TensorLayout};
        use crate::tensor_rank::DynRank;

        let layout = TensorLayout::<DynRank>::row_major(vec![rows as u64, cols as u64], dtype);
        let buf = pool
            .allocate(layout.buffer_size_bytes())
            .expect("EyeLike: pool allocation failed");
        let mut out = NumericTensor::from_parts(buf, layout);

        let zero = NumericScalar::zero(dtype);
        let one = NumericScalar::from_f32(1.0).cast_to(dtype);
        for i in 0..(rows * cols) {
            out.write_element(i, zero);
        }
        for i in 0..rows {
            let j = i as i64 + k;
            if j >= 0 && (j as usize) < cols {
                out.write_element(i * cols + j as usize, one);
            }
        }
        out
    }
}

impl crate::graph::Node for EyeLike {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "EyeLike".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl MilliOp for EyeLike {
    fn infer<'a, 'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'a, 'p, P>>,
        _symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        pool: &'p P,
    ) -> Result<Vec<(GlobalId, crate::tensor_info::TensorInfo<'a, 'p, P>)>, MilliOpGraphError>
    where
        'p: 'a,
    {
        use crate::scalar_info::ScalarInfoTyped;

        let input_info = known_inputs
            .get(&self.input)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let ranked = input_info
            .as_ranked()
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let shape = ranked.shape();
        if shape.len() != 2 {
            return Err(MilliOpGraphError::UnableToInfer);
        }

        // Try to constant-fold when shape is concrete.
        if let (ScalarInfoTyped::Numeric(rows), ScalarInfoTyped::Numeric(cols)) =
            (&shape[0], &shape[1])
        {
            let tensor = Self::build_tensor(
                *rows as usize,
                *cols as usize,
                self.k,
                self.output_dtype,
                pool,
            );
            return Ok(vec![(
                self.output,
                crate::tensor_info::TensorInfo::from_view(&tensor.view(), pool),
            )]);
        }

        // Shape not fully known — return shape-only info.
        let out =
            crate::tensor_info::TensorInfo::from_dtype_and_shape_scalars(self.output_dtype, &shape);
        Ok(vec![(self.output, out)])
    }

    fn eval_new<'p, P2: Pool + 'p>(
        &self,
        inputs: &[crate::numeric_tensor::NumericTensorView<'_, crate::tensor_rank::DynRank>],
        pool: &'p P2,
    ) -> Result<
        Vec<crate::numeric_tensor::NumericTensor<'p, crate::tensor_rank::DynRank, P2>>,
        crate::nano_graph::pool_eval::PoolEvalError,
    > {
        let shape = inputs[0].shape();
        if shape.len() != 2 {
            return Err(crate::nano_graph::pool_eval::PoolEvalError::Unsupported(
                "EyeLike requires 2D input".into(),
            ));
        }
        Ok(vec![Self::build_tensor(
            shape[0] as usize,
            shape[1] as usize,
            self.k,
            self.output_dtype,
            pool,
        )])
    }

    fn lower_to_nano<'p, P: crate::pool::Pool + 'p>(
        &self,
        ctx: &mut crate::nano_graph::NanoLoweringContext<'_, 'p, P>,
    ) -> super::LowerResult {
        // If infer constant-folded, register_constant handles it automatically.
        // Otherwise fall through to opaque (eval_new).
        let info = ctx.all_infos.get(&self.output);
        if let Some(info) = info
            && info.as_concrete().is_some()
        {
            ctx.register_constant(self.output, info);
            return super::LowerResult::Lowered;
        }
        super::LowerResult::Unsupported
    }
}
