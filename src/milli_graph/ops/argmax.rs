use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::pool::Pool;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArgMax {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    input: GlobalId,
    axis: i64,
    keepdims: bool,
    select_last_index: bool,
}

impl ArgMax {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        input: GlobalId,
        axis: i64,
        keepdims: bool,
        select_last_index: bool,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new_with_label(graph, input, axis, keepdims, select_last_index, None, rng)
    }

    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        input: GlobalId,
        axis: i64,
        keepdims: bool,
        select_last_index: bool,
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
            keepdims,
            select_last_index,
        };
        graph.push_op(AnyMilliOp::ArgMax(node));
        output
    }
}

impl ArgMax {
    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.input, map);
    }

    pub fn lower_to_nano(
        &self,
        ctx: &mut crate::nano_graph::NanoLoweringContext,
    ) -> crate::milli_graph::ops::LowerResult {
        let all_infos = ctx.all_infos;
        let data_rank = all_infos
            .get(&self.input)
            .and_then(|i| i.rank_if_known())
            .unwrap_or(0);
        let axis = if self.axis < 0 {
            (self.axis + data_rank as i64) as usize
        } else {
            self.axis as usize
        };

        let input_ids: Vec<GlobalId> = self.inputs().collect();
        let output_ids = vec![self.output];

        let eval_fn = std::sync::Arc::new(ArgMaxEval {
            axis,
            keepdims: self.keepdims,
            select_last_index: self.select_last_index,
        });
        ctx.register_opaque_op(eval_fn, "ArgMax", &input_ids, &output_ids);
        crate::milli_graph::ops::LowerResult::Lowered
    }
}

/// Opaque eval implementation for ArgMax on new types.
struct ArgMaxEval {
    axis: usize,
    keepdims: bool,
    select_last_index: bool,
}

impl crate::nano_graph::ops::OpaqueEval for ArgMaxEval {
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
        use crate::numeric_dtype::NumericDType;
        use crate::numeric_tensor::{NumericTensor, TensorLayout};
        use crate::pool::{Pool, SystemPool};
        use crate::tensor_rank::DynRank;

        static POOL: SystemPool = SystemPool;
        let data = &inputs[0];
        let shape = data.shape();
        let rank = shape.len();
        let axis = self.axis;
        let out_dtype = NumericDType::I64;

        // Compute output shape.
        let mut out_shape = Vec::new();
        for (i, &dim) in shape.iter().enumerate() {
            if i == axis {
                if self.keepdims {
                    out_shape.push(1u64);
                }
            } else {
                out_shape.push(dim);
            }
        }
        if out_shape.is_empty() {
            out_shape.push(1);
        }

        let out_numel: usize = out_shape.iter().product::<u64>() as usize;
        let layout = TensorLayout::<DynRank>::row_major(out_shape.clone(), out_dtype);
        let buf = POOL
            .allocate(layout.buffer_size_bytes())
            .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
        let mut out = NumericTensor::from_parts(buf, layout);

        // Compute strides for the input shape.
        let in_strides = {
            let mut s = vec![1usize; rank];
            for i in (0..rank.saturating_sub(1)).rev() {
                s[i] = s[i + 1] * shape[i + 1] as usize;
            }
            s
        };

        let axis_dim = shape[axis] as usize;
        let axis_stride = in_strides[axis];

        // Number of independent "lines" along the axis.
        let outer_count = out_numel;

        // Compute strides for the output shape.
        let out_strides = {
            let mut s = vec![1usize; out_shape.len()];
            for i in (0..out_shape.len().saturating_sub(1)).rev() {
                s[i] = s[i + 1] * out_shape[i + 1] as usize;
            }
            s
        };

        for out_flat in 0..outer_count {
            // Decompose out_flat into output coords, then map to input base index.
            let mut rem = out_flat;
            let mut base = 0usize;
            let mut out_dim_idx = 0;
            #[allow(clippy::needless_range_loop)]
            for d in 0..rank {
                if d == axis {
                    if self.keepdims {
                        out_dim_idx += 1; // skip the axis dim (always 0)
                    }
                    continue;
                }
                let coord = rem / out_strides[out_dim_idx];
                rem %= out_strides[out_dim_idx];
                base += coord * in_strides[d];
                out_dim_idx += 1;
            }

            // Scan along axis to find argmax.
            let mut best_idx: usize = 0;
            let mut best_val: f64 = f64::NEG_INFINITY;
            for k in 0..axis_dim {
                let flat = base + k * axis_stride;
                let val = data.read_element(flat).to_f64();
                if val > best_val || (self.select_last_index && val >= best_val) {
                    best_val = val;
                    best_idx = k;
                }
            }

            out.write_element(
                out_flat,
                crate::numeric_scalar::NumericScalar::from_i64(best_idx as i64),
            );
        }

        Ok(vec![out])
    }
}

impl MilliOp for ArgMax {
    fn eval_new<'p, P2: crate::pool::Pool + 'p>(
        &self,
        inputs: &[crate::numeric_tensor::NumericTensorView<'_, crate::tensor_rank::DynRank>],
        pool: &'p P2,
    ) -> Result<
        Vec<crate::numeric_tensor::NumericTensor<'p, crate::tensor_rank::DynRank, P2>>,
        crate::nano_graph::pool_eval::PoolEvalError,
    > {
        use crate::numeric_dtype::NumericDType;
        use crate::numeric_tensor::{NumericTensor, TensorLayout};
        use crate::tensor_rank::DynRank;

        let data = &inputs[0];
        let shape = data.shape();
        let rank = shape.len();
        let axis = if self.axis < 0 {
            (self.axis + rank as i64) as usize
        } else {
            self.axis as usize
        };
        let out_dtype = NumericDType::I64;

        // Compute output shape.
        let mut out_shape = Vec::new();
        for (i, &dim) in shape.iter().enumerate() {
            if i == axis {
                if self.keepdims {
                    out_shape.push(1u64);
                }
            } else {
                out_shape.push(dim);
            }
        }
        if out_shape.is_empty() {
            out_shape.push(1);
        }

        let out_numel: usize = out_shape.iter().product::<u64>() as usize;
        let layout = TensorLayout::<DynRank>::row_major(out_shape.clone(), out_dtype);
        let buf = pool
            .allocate(layout.buffer_size_bytes())
            .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
        let mut out = NumericTensor::from_parts(buf, layout);

        // Compute strides for the input shape.
        let in_strides = {
            let mut s = vec![1usize; rank];
            for i in (0..rank.saturating_sub(1)).rev() {
                s[i] = s[i + 1] * shape[i + 1] as usize;
            }
            s
        };

        let axis_dim = shape[axis] as usize;
        let axis_stride = in_strides[axis];
        let outer_count = out_numel;

        // Compute strides for the output shape.
        let out_strides = {
            let mut s = vec![1usize; out_shape.len()];
            for i in (0..out_shape.len().saturating_sub(1)).rev() {
                s[i] = s[i + 1] * out_shape[i + 1] as usize;
            }
            s
        };

        for out_flat in 0..outer_count {
            // Decompose out_flat into output coords, then map to input base index.
            let mut rem = out_flat;
            let mut base = 0usize;
            let mut out_dim_idx = 0;
            #[allow(clippy::needless_range_loop)]
            for d in 0..rank {
                if d == axis {
                    if self.keepdims {
                        out_dim_idx += 1;
                    }
                    continue;
                }
                let coord = rem / out_strides[out_dim_idx];
                rem %= out_strides[out_dim_idx];
                base += coord * in_strides[d];
                out_dim_idx += 1;
            }

            // Scan along axis to find argmax.
            let mut best_idx: usize = 0;
            let mut best_val: f64 = f64::NEG_INFINITY;
            for k in 0..axis_dim {
                let flat = base + k * axis_stride;
                let val = data.read_element(flat).to_f64();
                if val > best_val || (self.select_last_index && val >= best_val) {
                    best_val = val;
                    best_idx = k;
                }
            }

            out.write_element(
                out_flat,
                crate::numeric_scalar::NumericScalar::from_i64(best_idx as i64),
            );
        }

        Ok(vec![out])
    }

    fn infer<'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'p, P>>,
        symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        _pool: &'p P,
    ) -> Result<Vec<(GlobalId, crate::tensor_info::TensorInfo<'p, P>)>, MilliOpGraphError> {
        use crate::numeric_dtype::NumericDType;
        use crate::scalar_info::{ScalarInfo, ScalarInfoTyped};
        use crate::symbolic_scalar::SymbolicScalar;
        use crate::tensor_info::TensorInfo;

        let input_info = known_inputs
            .get(&self.input)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let out_dtype = NumericDType::I64;

        if let Some(ranked) = input_info.as_ranked() {
            let shape = ranked.shape();
            let rank = shape.len();
            let axis = if self.axis < 0 {
                (self.axis + rank as i64) as usize
            } else {
                self.axis as usize
            };
            let mut out_dims = Vec::new();
            for (i, dim) in shape.iter().enumerate() {
                if i == axis {
                    if self.keepdims {
                        out_dims.push(ScalarInfoTyped::Numeric(1));
                    }
                    // else: skip this dim
                } else {
                    out_dims.push(dim.clone());
                }
            }
            return Ok(vec![(
                self.output,
                TensorInfo::from_dtype_and_shape_scalars(out_dtype, &out_dims),
            )]);
        }

        // Fallback: unknown shape
        let first = ScalarInfo::Symbolic(SymbolicScalar::new(out_dtype, symbolic_resolver));
        Ok(vec![(
            self.output,
            TensorInfo::new_from_first_element_and_rank(
                first,
                input_info.rank(),
                symbolic_resolver,
            ),
        )])
    }

}

impl Node for ArgMax {
    type OpKind = String;
    fn op_kind(&self) -> Self::OpKind {
        "ArgMax".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.input].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
}
