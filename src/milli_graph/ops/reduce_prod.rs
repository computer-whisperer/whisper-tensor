use crate::pool::Pool;
use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::migration::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

use super::AccumulationMode;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReduceProd {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    data: GlobalId,
    axes: Option<GlobalId>,
    keepdims: bool,
    noop_with_empty_axes: bool,
    /// Prescribes the order in which elements are accumulated.
    /// Both milli-eval and nano-eval must follow this to produce identical results.
    accumulation_mode: AccumulationMode,
}

impl ReduceProd {
    pub fn accumulation_mode(&self) -> AccumulationMode {
        self.accumulation_mode
    }

    pub fn push_new(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        axes: Option<GlobalId>,
        keepdims: bool,
        noop_with_empty_axes: bool,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        Self::push_new_with_label(graph, data, axes, keepdims, noop_with_empty_axes, None, rng)
    }

    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        axes: Option<GlobalId>,
        keepdims: bool,
        noop_with_empty_axes: bool,
        label: Option<String>,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            label,
            output,
            data,
            axes,
            keepdims,
            noop_with_empty_axes,
            accumulation_mode: AccumulationMode::default(),
        };
        graph.push_op(AnyMilliOp::ReduceProd(node));
        output
    }
}

impl ReduceProd {
    pub(crate) fn axes_tensor(&self) -> Option<GlobalId> {
        self.axes
    }
    pub(crate) fn noop_with_empty_axes(&self) -> bool {
        self.noop_with_empty_axes
    }
    pub(crate) fn keepdims(&self) -> bool {
        self.keepdims
    }

    pub fn lower_to_nano(&self, ctx: &mut crate::nano_graph::NanoLoweringContext) {
        use crate::nano_graph::{ReduceKind, ScalarOp};
        ctx.lower_reduce(self, |compute_dt, count, stride| ScalarOp::Reduce {
            kind: ReduceKind::Prod,
            reduce_count: count,
            reduce_stride: stride,
            compute_dtype: compute_dt,
        });
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.data, map);
        super::remap_opt(&mut self.axes, map);
    }

}

impl Node for ReduceProd {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "ReduceProd".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        match self.axes {
            Some(ax) => Box::new(vec![self.data, ax].into_iter()),
            None => Box::new(vec![self.data].into_iter()),
        }
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
}

impl MilliOp for ReduceProd {
    fn lower_to_nano(&self, ctx: &mut crate::nano_graph::NanoLoweringContext) {
        ReduceProd::lower_to_nano(self, ctx);
    }

    fn eval_new<'p, P2: crate::pool::Pool + 'p>(
        &self,
        inputs: &[crate::numeric_tensor::NumericTensorView<'_, crate::tensor_rank::DynRank>],
        pool: &'p P2,
    ) -> Result<Vec<crate::numeric_tensor::NumericTensor<'p, crate::tensor_rank::DynRank, P2>>, crate::nano_graph::pool_eval::PoolEvalError> {
        use crate::numeric_tensor::{NumericTensor, TensorLayout};
        use crate::tensor_rank::DynRank;

        let data = &inputs[0];
        let shape = data.shape();
        let rank = shape.len();
        let dtype = data.dtype();

        // Extract axes: if axes input exists, read from inputs[1]; else reduce all.
        let axes: Vec<usize> = if self.axes.is_some() && inputs.len() > 1 {
            let ax_view = &inputs[1];
            let raw: Vec<i64> = (0..ax_view.numel()).map(|i| ax_view.read_element(i).to_i64()).collect();
            if raw.is_empty() && self.noop_with_empty_axes {
                // Noop: copy input.
                let layout = TensorLayout::<DynRank>::row_major(shape.clone(), dtype);
                let buf = pool.allocate(layout.buffer_size_bytes())
                    .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
                let mut out = NumericTensor::from_parts(buf, layout);
                for i in 0..data.numel() { out.write_element(i, data.read_element(i)); }
                return Ok(vec![out]);
            }
            if raw.is_empty() {
                (0..rank).collect()
            } else {
                raw.iter().map(|&a| if a < 0 { (a + rank as i64) as usize } else { a as usize }).collect()
            }
        } else {
            if self.noop_with_empty_axes && self.axes.is_none() {
                // No axes specified + noop_with_empty_axes → identity
            }
            (0..rank).collect()
        };

        // Compute output shape.
        let mut out_shape = Vec::new();
        for (i, &dim) in shape.iter().enumerate() {
            if axes.contains(&i) {
                if self.keepdims { out_shape.push(1u64); }
            } else {
                out_shape.push(dim);
            }
        }
        if out_shape.is_empty() { out_shape.push(1); }

        let out_numel: usize = out_shape.iter().product::<u64>() as usize;
        let layout = TensorLayout::<DynRank>::row_major(out_shape.clone(), dtype);
        let buf = pool.allocate(layout.buffer_size_bytes())
            .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
        let mut out = NumericTensor::from_parts(buf, layout);

        // Initialize to 1.0 (multiplicative identity).
        for i in 0..out_numel {
            out.write_element(i, crate::numeric_scalar::NumericScalar::from_f64(1.0).cast_to(dtype));
        }

        // Strides for index decomposition.
        let in_strides = {
            let mut s = vec![1usize; rank];
            for i in (0..rank.saturating_sub(1)).rev() { s[i] = s[i + 1] * shape[i + 1] as usize; }
            s
        };
        let out_strides = {
            let mut s = vec![1usize; out_shape.len()];
            for i in (0..out_shape.len().saturating_sub(1)).rev() { s[i] = s[i + 1] * out_shape[i + 1] as usize; }
            s
        };

        for flat_in in 0..data.numel() {
            let mut rem = flat_in;
            let mut out_flat = 0usize;
            let mut out_dim_idx = 0;
            for i in 0..rank {
                let idx = rem / in_strides[i];
                rem %= in_strides[i];
                if !axes.contains(&i) {
                    out_flat += idx * out_strides[out_dim_idx];
                    out_dim_idx += 1;
                } else if self.keepdims {
                    out_dim_idx += 1;
                }
            }
            let val = data.read_element(flat_in).to_f64();
            let cur = out.read_element(out_flat).to_f64();
            out.write_element(out_flat, crate::numeric_scalar::NumericScalar::from_f64(cur * val).cast_to(dtype));
        }
        Ok(vec![out])
    }

    fn infer<'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'p, P>>,
        symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        pool: &'p P,
    ) -> Result<
        Vec<(GlobalId, crate::tensor_info::TensorInfo<'p, P>)>,
        MilliOpGraphError,
    > {
        use crate::scalar_info::ScalarInfoTyped;
        use crate::tensor_info::TensorInfo;

        let data_info = known_inputs
            .get(&self.data)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        let out_dtype = data_info.dtype();

        // Compute symbolic output info.
        let out_info = if let Some(out_dims) = super::infer_reduce_output_shape(
            data_info,
            self.axes,
            self.keepdims,
            self.noop_with_empty_axes,
            known_inputs,
            symbolic_resolver,
        ) {
            TensorInfo::from_dtype_and_shape_scalars(out_dtype, &out_dims)
        } else {
            // Fallback: rank-only inference.
            let num_axes: Option<usize> = if let Some(ax_id) = self.axes {
                known_inputs.get(&ax_id).and_then(|ax_info| {
                    ax_info
                        .rank_if_known()
                        .and_then(|_| ax_info.dim_if_known(0).map(|n| n as usize))
                })
            } else {
                None
            };

            let out_rank: ScalarInfoTyped<u32> = match data_info.rank() {
                ScalarInfoTyped::Numeric(input_rank) => {
                    if self.axes.is_none() {
                        ScalarInfoTyped::Numeric(if self.keepdims { input_rank } else { 0 })
                    } else if let Some(n) = num_axes {
                        if n == 0 && self.noop_with_empty_axes {
                            ScalarInfoTyped::Numeric(input_rank)
                        } else if n == 0 {
                            ScalarInfoTyped::Numeric(if self.keepdims { input_rank } else { 0 })
                        } else if self.keepdims {
                            ScalarInfoTyped::Numeric(input_rank)
                        } else {
                            ScalarInfoTyped::Numeric(input_rank.saturating_sub(n as u32))
                        }
                    } else if self.keepdims {
                        ScalarInfoTyped::Numeric(input_rank)
                    } else {
                        ScalarInfoTyped::Symbolic(crate::symbolic_scalar::SymbolicScalarTyped::new(
                            symbolic_resolver,
                        ))
                    }
                }
                _ => ScalarInfoTyped::Symbolic(crate::symbolic_scalar::SymbolicScalarTyped::new(
                    symbolic_resolver,
                )),
            };

            let first_elem = crate::scalar_info::ScalarInfo::Symbolic(
                crate::symbolic_scalar::SymbolicScalar::new(out_dtype, symbolic_resolver),
            );
            TensorInfo::new_from_first_element_and_rank(first_elem, out_rank, symbolic_resolver)
        };

        // Check if all inputs are concrete; if so, try constant fold with output hints.
        if let Some(results) = super::constant_fold(self, known_inputs, &[(self.output, out_info.clone_with_pool(pool))], pool) {
            return Ok(results);
        }

        Ok(vec![((self.output, out_info))])
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        _config: &super::MilliEvalConfig,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        let data = &inputs[&self.data];
        let axes = if let Some(axes) = self.axes {
            Vec::<i64>::try_from(inputs[&axes].try_to_rank::<P1>()?)?
        } else {
            (0i64..(data.shape().len() as i64)).collect()
        };
        let axes = if axes.is_empty() {
            if self.noop_with_empty_axes {
                let out_tensor = data.clone();
                return Ok(Box::new([(self.output, out_tensor)].into_iter()));
            } else {
                (0i64..(data.shape().len() as i64)).collect::<Vec<_>>()
            }
        } else {
            axes
        };
        let axes = axes
            .into_iter()
            .map(|x| {
                (if x < 0 {
                    x + data.shape().len() as i64
                } else {
                    x
                }) as usize
            })
            .collect::<Vec<_>>();
        let out = data.reduce_prod(axes, self.keepdims, self.accumulation_mode, backend)?;
        Ok(Box::new([(self.output, out)].into_iter()))
    }
}
