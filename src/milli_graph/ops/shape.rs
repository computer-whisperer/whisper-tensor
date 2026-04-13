use crate::graph::{GlobalId, Node};
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::pool::Pool;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    pub fn lower_to_nano<'p, P: crate::pool::Pool + 'p>(
        &self,
        ctx: &mut crate::nano_graph::NanoLoweringContext<'_, 'p, P>,
    ) -> crate::milli_graph::ops::LowerResult {
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
    fn infer<'a, 'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'a, 'p, P>>,
        rng: &mut impl Rng,
        pool: &'p P,
    ) -> Result<
        Vec<(GlobalId, crate::tensor_info::TensorInfo<'a, 'p, P>)>,
        crate::milli_graph::MilliOpGraphError,
    >
    where
        'p: 'a,
    {
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
                let tensor = crate::numeric_tensor::NumericTensor::from_fn(
                    vec![dim_vals.len() as u64],
                    crate::numeric_dtype::NumericDType::I64,
                    pool,
                    |i| crate::numeric_scalar::NumericScalar::from_i64(dim_vals[i]),
                )
                .map_err(|_| crate::milli_graph::MilliOpGraphError::UnableToInfer)?;
                return Ok(vec![(
                    self.output,
                    TensorInfo::from_view(&tensor.view(), pool),
                )]);
            }
        }

        // Fallback: symbolic output with known rank=1.
        let first_elem =
            crate::scalar_info::ScalarInfo::Symbolic(crate::symbolic_scalar::SymbolicScalar::new(
                crate::numeric_dtype::NumericDType::I64,
                rng,
            ));
        let out_info = TensorInfo::new_from_first_element_and_rank(
            first_elem,
            crate::scalar_info::ScalarInfoTyped::Numeric(1),
            rng,
        );

        Ok(vec![(self.output, out_info)])
    }

    fn eval_new<'p, P2: crate::pool::Pool + 'p>(
        &self,
        inputs: &[crate::numeric_tensor::NumericTensorView<'_, crate::tensor_rank::DynRank>],
        pool: &'p P2,
    ) -> Result<
        Vec<crate::numeric_tensor::NumericTensor<'p, crate::tensor_rank::DynRank, P2>>,
        crate::nano_graph::pool_eval::PoolEvalError,
    > {
        use crate::numeric_scalar::NumericScalar;
        use crate::numeric_tensor::{NumericTensor, TensorLayout};
        use crate::tensor_rank::DynRank;

        let input_shape = inputs[0].shape();
        let rank = input_shape.len();
        let layout = TensorLayout::<DynRank>::row_major(
            vec![rank as u64],
            crate::numeric_dtype::NumericDType::I64,
        );
        let buf = pool
            .allocate(layout.buffer_size_bytes())
            .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
        let mut out = NumericTensor::from_parts(buf, layout);
        for (i, &dim) in input_shape.iter().enumerate() {
            out.write_element(i, NumericScalar::from_i64(dim as i64));
        }
        Ok(vec![out])
    }

    fn lower_to_nano<'p, P: crate::pool::Pool + 'p>(
        &self,
        ctx: &mut crate::nano_graph::NanoLoweringContext<'_, 'p, P>,
    ) -> crate::milli_graph::ops::LowerResult {
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
