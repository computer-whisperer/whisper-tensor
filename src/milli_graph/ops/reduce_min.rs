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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReduceMin {
    output: GlobalId,
    data: GlobalId,
    axes: Option<GlobalId>,
    keepdims: bool,
    noop_with_empty_axes: bool,
    global_id: GlobalId,
    pub(crate) label: Option<String>,
}

impl ReduceMin {
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
        };
        graph.push_op(AnyMilliOp::ReduceMin(node));
        output
    }
}

impl ReduceMin {
    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.data, map);
        super::remap_opt(&mut self.axes, map);
    }
}

impl MilliOp for ReduceMin {
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
                let out = data.clone();
                return Ok(Box::new([(self.output, out)].into_iter()));
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
        let out = data.reduce_min(axes, self.keepdims, backend)?;
        Ok(Box::new([(self.output, out)].into_iter()))
    }
}

impl Node for ReduceMin {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "ReduceMin".to_string()
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
