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
        use crate::nano_graph::lower::{ConcatSegment, DimKind, TensorAtomMap};
        use crate::nano_graph::ops::ScalarOp;
        use crate::numeric_dtype::NumericDType;
        use crate::numeric_scalar::NumericScalar;

        let in_id = Node::inputs(self).next().unwrap();
        let out_id = self.output;

        // Walk the input's dim list and emit one 1-atom
        // group per dim.  Known dims use `ScalarOp::Literal`; symbolic
        // dims use `ScalarOp::GcLiteral(gc)` so the value is resolved
        // from `gc_values` at evaluation time.  The output is a rank-1
        // tensor of length input_rank, assembled from the per-dim
        // groups via ConcatSegment.
        let Some(in_map) = ctx.tensor_map.get(&in_id).cloned() else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };
        let input_rank = in_map.dims.len();
        if input_rank == 0 {
            // Scalar input: Shape returns an empty tensor [] of rank 1
            // with size 0.  Unusual; let it fall through to opaque.
            return crate::milli_graph::ops::LowerResult::Unsupported;
        }

        let out_dt = NumericDType::I64;
        let mut segments: Vec<ConcatSegment> = Vec::with_capacity(input_rank);
        for (i, dim) in in_map.dims.iter().enumerate() {
            let op = match dim {
                DimKind::Known { size, .. } => {
                    ScalarOp::Literal(NumericScalar::from_i64(*size as i64))
                }
                DimKind::Sym { gc, .. } => ScalarOp::GcLiteral(*gc),
            };
            let base = ctx.nano.push_group(1, out_dt, op, vec![], vec![]);
            segments.push(ConcatSegment {
                concat_dim: 0,
                start: i as u64,
                size: 1,
                base_id: base,
                known_strides: vec![1],
            });
        }

        let out_dims = vec![DimKind::Known {
            size: input_rank as u64,
            stride: 1,
        }];
        ctx.tensor_map.insert(
            out_id,
            TensorAtomMap::segmented(input_rank as u64, out_dt, out_dims, segments),
        );
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
        use crate::scalar_info::ScalarInfoTyped;
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

            // Rank is known but some dims are symbolic.  The OUTPUT's shape
            // is still concrete: rank-1 of length `rank`.  Only the element
            // values are symbolic.
            let out_info = TensorInfo::from_dtype_and_shape_scalars(
                crate::numeric_dtype::NumericDType::I64,
                &[ScalarInfoTyped::Numeric(rank as u64)],
            );
            return Ok(vec![(self.output, out_info)]);
        }

        // Input rank unknown — fall back to rank-1 with symbolic length.
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
