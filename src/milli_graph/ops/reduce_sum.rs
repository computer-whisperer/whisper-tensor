use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::pool::Pool;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::AccumulationMode;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReduceSum {
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

impl ReduceSum {
    pub(crate) fn keepdims(&self) -> bool {
        self.keepdims
    }
    pub(crate) fn axes_tensor(&self) -> Option<GlobalId> {
        self.axes
    }
    #[allow(dead_code)] // used by compiler (cranelift feature)
    pub(crate) fn noop_with_empty_axes(&self) -> bool {
        self.noop_with_empty_axes
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
        graph.push_op(AnyMilliOp::ReduceSum(node));
        output
    }

    pub fn accumulation_mode(&self) -> AccumulationMode {
        self.accumulation_mode
    }
}

impl ReduceSum {
    pub fn lower_to_nano<'p, P: crate::pool::Pool + 'p>(
        &self,
        ctx: &mut crate::nano_graph::NanoLoweringContext<'_, 'p, P>,
    ) -> crate::milli_graph::ops::LowerResult {
        use crate::nano_graph::{ReduceKind, ScalarOp};
        ctx.lower_reduce(self, |compute_dt, count, stride| ScalarOp::Reduce {
            kind: ReduceKind::Sum,
            reduce_count: count,
            reduce_stride: stride,
            compute_dtype: compute_dt,
        })
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.data, map);
        super::remap_opt(&mut self.axes, map);
    }
}

impl MilliOp for ReduceSum {
    fn infer<'a, 'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'a, 'p, P>>,
        symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        pool: &'p P,
    ) -> Result<Vec<(GlobalId, crate::tensor_info::TensorInfo<'a, 'p, P>)>, MilliOpGraphError>
    where
        'p: 'a,
    {
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
        rng: &mut impl rand::Rng,
    ) -> Option<HashMap<GlobalId, GlobalId>> {
        let grad_output = *output_grads.get(&self.output)?;
        // Backward of ReduceSum: expand grad_output back to input shape.
        // If keepdims=false, first unsqueeze along the reduced axes.
        let expanded_grad = if self.keepdims {
            grad_output
        } else if let Some(axes) = self.axes {
            // Unsqueeze along the specific reduced axes
            super::Unsqueeze::push_new(graph, grad_output, axes, rng)
        } else {
            // Reduced all axes to scalar — reshape to all-ones shape via unsqueeze
            // We don't know the rank, so use Reshape to [1,1,...] matching input shape
            // Actually, Expand handles broadcasting from scalar, just get input shape
            grad_output
        };
        // Expand to input shape
        let input_shape = super::Shape::push_new(graph, self.data, rng);
        let grad_input = super::Expand::push_new(graph, expanded_grad, input_shape, rng);
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
        use crate::numeric_scalar::NumericScalar;
        let dtype = inputs[0].dtype();
        super::reduce_eval_new(
            inputs,
            if self.axes.is_some() { Some(1) } else { None },
            self.keepdims,
            self.noop_with_empty_axes,
            NumericScalar::zero(dtype),
            |cur, val| cur.add(val),
            |v, _count| v,
            pool,
        )
    }

    fn lower_to_nano<'p, P: crate::pool::Pool + 'p>(
        &self,
        ctx: &mut crate::nano_graph::NanoLoweringContext<'_, 'p, P>,
    ) -> crate::milli_graph::ops::LowerResult {
        ReduceSum::lower_to_nano(self, ctx)
    }
}

impl Node for ReduceSum {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "ReduceSum".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        let it: Box<dyn Iterator<Item = GlobalId>> = match self.axes {
            Some(ax) => Box::new(vec![self.data, ax].into_iter()),
            None => Box::new(vec![self.data].into_iter()),
        };
        it
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new([self.output].into_iter())
    }
}
