use crate::graph::{GlobalId, Node};
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::MilliOpGraphError;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::pool::Pool;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CumSum {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    input: GlobalId,
    axis: GlobalId,
    exclusive: bool,
    reverse: bool,
}

impl CumSum {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        input: GlobalId,
        axis: GlobalId,
        exclusive: bool,
        reverse: bool,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new_with_label(graph, input, axis, exclusive, reverse, None, rng)
    }

    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        input: GlobalId,
        axis: GlobalId,
        exclusive: bool,
        reverse: bool,
        label: Option<String>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            label,
            output,
            input,
            axis,
            exclusive,
            reverse,
        };
        graph.push_op(AnyMilliOp::CumSum(node));
        output
    }
}

impl CumSum {
    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.input, map);
        super::remap(&mut self.axis, map);
    }

    pub fn lower_to_nano(
        &self,
        ctx: &mut crate::nano_graph::NanoLoweringContext,
    ) -> crate::milli_graph::ops::LowerResult {
        let exclusive = self.exclusive;
        let reverse = self.reverse;

        let input_ids: Vec<GlobalId> = self.inputs().collect();
        let output_ids = vec![self.output];

        let eval_fn = std::sync::Arc::new(CumSumEval { exclusive, reverse });
        ctx.register_opaque_op(eval_fn, "CumSum", &input_ids, &output_ids);
        crate::milli_graph::ops::LowerResult::Lowered
    }
}

/// Opaque eval implementation for CumSum on new types.
struct CumSumEval {
    exclusive: bool,
    reverse: bool,
}

impl crate::nano_graph::ops::OpaqueEval for CumSumEval {
    fn eval(
        &self,
        inputs: &[crate::numeric_tensor::NumericTensorView<'_, crate::tensor_rank::DynRank>],
    ) -> Result<
        Vec<
            crate::numeric_tensor::NumericTensor<
                'static,
                crate::tensor_rank::DynRank,
                crate::pool::SystemPool,
            >,
        >,
        crate::nano_graph::pool_eval::PoolEvalError,
    > {
        use crate::numeric_tensor::{NumericTensor, TensorLayout};
        use crate::pool::{Pool, SystemPool};
        use crate::tensor_rank::DynRank;

        static POOL: SystemPool = SystemPool;
        let data = &inputs[0];
        let shape = data.shape();
        let rank = shape.len();
        let dtype = data.dtype();

        // Extract axis from inputs[1] (scalar i64).
        let raw_axis = inputs[1].read_element(0).to_i64();
        let axis = if raw_axis < 0 {
            (raw_axis + rank as i64) as usize
        } else {
            raw_axis as usize
        };

        let layout = TensorLayout::<DynRank>::row_major(shape.clone(), dtype);
        let buf = POOL
            .allocate(layout.buffer_size_bytes())
            .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
        let mut out = NumericTensor::from_parts(buf, layout);

        // Compute strides for the input shape.
        let strides = {
            let mut s = vec![1usize; rank];
            for i in (0..rank.saturating_sub(1)).rev() {
                s[i] = s[i + 1] * shape[i + 1] as usize;
            }
            s
        };

        let axis_dim = shape[axis] as usize;
        let axis_stride = strides[axis];
        // Number of independent "lines" along the axis.
        let outer_count = data.numel() / axis_dim.max(1);

        // For each line along the axis, compute cumulative sum.
        for line in 0..outer_count {
            // Compute the flat index of the first element in this line (axis coord = 0).
            // Decompose `line` into the multi-index skipping the axis dimension.
            let base = {
                let mut rem = line;
                let mut idx = 0usize;
                for d in 0..rank {
                    if d == axis {
                        continue;
                    }
                    let dim_stride = strides[d];
                    // Effective stride in the "outer" iteration: product of non-axis dims after d.
                    let outer_stride = {
                        let mut os = 1usize;
                        for d2 in (d + 1)..rank {
                            if d2 != axis {
                                os *= shape[d2] as usize;
                            }
                        }
                        os
                    };
                    let coord = rem / outer_stride;
                    rem %= outer_stride;
                    idx += coord * dim_stride;
                }
                idx
            };

            let mut acc = 0.0f64;
            if self.reverse {
                for k in (0..axis_dim).rev() {
                    let flat = base + k * axis_stride;
                    let val = data.read_element(flat).to_f64();
                    if self.exclusive {
                        out.write_element(
                            flat,
                            crate::numeric_scalar::NumericScalar::from_f64(acc).cast_to(dtype),
                        );
                        acc += val;
                    } else {
                        acc += val;
                        out.write_element(
                            flat,
                            crate::numeric_scalar::NumericScalar::from_f64(acc).cast_to(dtype),
                        );
                    }
                }
            } else {
                for k in 0..axis_dim {
                    let flat = base + k * axis_stride;
                    let val = data.read_element(flat).to_f64();
                    if self.exclusive {
                        out.write_element(
                            flat,
                            crate::numeric_scalar::NumericScalar::from_f64(acc).cast_to(dtype),
                        );
                        acc += val;
                    } else {
                        acc += val;
                        out.write_element(
                            flat,
                            crate::numeric_scalar::NumericScalar::from_f64(acc).cast_to(dtype),
                        );
                    }
                }
            }
        }

        Ok(vec![out])
    }
}

impl Node for CumSum {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "CumSum".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.input, self.axis].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
}

impl MilliOp for CumSum {
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
        let shape = data.shape();
        let rank = shape.len();
        let dtype = data.dtype();

        // Extract axis from inputs[1] (scalar i64).
        let raw_axis = inputs[1].read_element(0).to_i64();
        let axis = if raw_axis < 0 {
            (raw_axis + rank as i64) as usize
        } else {
            raw_axis as usize
        };

        let layout = TensorLayout::<DynRank>::row_major(shape.clone(), dtype);
        let buf = pool
            .allocate(layout.buffer_size_bytes())
            .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
        let mut out = NumericTensor::from_parts(buf, layout);

        // Compute strides for the input shape.
        let strides = {
            let mut s = vec![1usize; rank];
            for i in (0..rank.saturating_sub(1)).rev() {
                s[i] = s[i + 1] * shape[i + 1] as usize;
            }
            s
        };

        let axis_dim = shape[axis] as usize;
        let axis_stride = strides[axis];
        // Number of independent "lines" along the axis.
        let outer_count = data.numel() / axis_dim.max(1);

        // For each line along the axis, compute cumulative sum.
        for line in 0..outer_count {
            // Compute the flat index of the first element in this line (axis coord = 0).
            let base = {
                let mut rem = line;
                let mut idx = 0usize;
                for d in 0..rank {
                    if d == axis {
                        continue;
                    }
                    let dim_stride = strides[d];
                    let outer_stride = {
                        let mut os = 1usize;
                        for d2 in (d + 1)..rank {
                            if d2 != axis {
                                os *= shape[d2] as usize;
                            }
                        }
                        os
                    };
                    let coord = rem / outer_stride;
                    rem %= outer_stride;
                    idx += coord * dim_stride;
                }
                idx
            };

            let mut acc = 0.0f64;
            if self.reverse {
                for k in (0..axis_dim).rev() {
                    let flat = base + k * axis_stride;
                    let val = data.read_element(flat).to_f64();
                    if self.exclusive {
                        out.write_element(
                            flat,
                            crate::numeric_scalar::NumericScalar::from_f64(acc).cast_to(dtype),
                        );
                        acc += val;
                    } else {
                        acc += val;
                        out.write_element(
                            flat,
                            crate::numeric_scalar::NumericScalar::from_f64(acc).cast_to(dtype),
                        );
                    }
                }
            } else {
                for k in 0..axis_dim {
                    let flat = base + k * axis_stride;
                    let val = data.read_element(flat).to_f64();
                    if self.exclusive {
                        out.write_element(
                            flat,
                            crate::numeric_scalar::NumericScalar::from_f64(acc).cast_to(dtype),
                        );
                        acc += val;
                    } else {
                        acc += val;
                        out.write_element(
                            flat,
                            crate::numeric_scalar::NumericScalar::from_f64(acc).cast_to(dtype),
                        );
                    }
                }
            }
        }

        Ok(vec![out])
    }

    fn infer<'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'p, P>>,
        symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        _pool: &'p P,
    ) -> Result<Vec<(GlobalId, crate::tensor_info::TensorInfo<'p, P>)>, MilliOpGraphError> {
        use crate::scalar_info::{ScalarInfo, ScalarInfoTyped};
        use crate::symbolic_scalar::SymbolicScalar;
        use crate::tensor_info::TensorInfo;

        let data_info = known_inputs
            .get(&self.input)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let out_dtype = data_info.dtype();

        // CumSum preserves shape — output shape == input shape.
        if let Some(ranked) = data_info.as_ranked() {
            let dims = ranked.shape();
            return Ok(vec![(
                self.output,
                TensorInfo::from_dtype_and_shape_scalars(out_dtype, &dims),
            )]);
        }

        // Fallback: unknown shape
        let first = ScalarInfo::Symbolic(SymbolicScalar::new(out_dtype, symbolic_resolver));
        Ok(vec![(
            self.output,
            TensorInfo::new_from_first_element_and_rank(first, data_info.rank(), symbolic_resolver),
        )])
    }

}
