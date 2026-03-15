use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::numeric_tensor::NumericTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

use super::AccumulationMode;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReduceMean {
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

impl ReduceMean {
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
        graph.push_op(AnyMilliOp::ReduceMean(node));
        output
    }
}

impl ReduceMean {
    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.data, map);
        super::remap_opt(&mut self.axes, map);
    }
}

impl Node for ReduceMean {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "ReduceMean".to_string()
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

impl MilliOp for ReduceMean {
    fn infer(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo>,
        symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        backend: &mut EvalBackend,
    ) -> Result<
        Box<dyn Iterator<Item = (GlobalId, crate::tensor_info::TensorInfo)>>,
        MilliOpGraphError,
    > {
        use crate::scalar_info::ScalarInfoTyped;
        use crate::tensor_info::TensorInfo;

        let data_info = known_inputs
            .get(&self.data)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        // Check if all inputs are concrete; if so, fall back to eval.
        let axes_concrete = self.axes.map(|ax_id| {
            known_inputs
                .get(&ax_id)
                .and_then(|info| info.as_numeric().cloned())
        });
        if data_info.as_numeric().is_some() {
            let axes_ok = match axes_concrete {
                Some(Some(_)) | None => true,
                Some(None) => false,
            };
            if axes_ok {
                let mut resolved = HashMap::new();
                resolved.insert(self.data, data_info.as_numeric().unwrap().clone());
                if let Some(ax_id) = self.axes {
                    resolved.insert(ax_id, axes_concrete.unwrap().unwrap());
                }
                let collected: Vec<(GlobalId, TensorInfo)> = self
                    .eval(&resolved, backend)?
                    .map(|(a, b)| (a, TensorInfo::from(b)))
                    .collect();
                return Ok(Box::new(collected.into_iter()));
            }
        }

        let out_dtype = data_info.dtype();

        // Try per-dim shape inference first.
        if let Some(out_dims) = super::infer_reduce_output_shape(
            data_info,
            self.axes,
            self.keepdims,
            self.noop_with_empty_axes,
            known_inputs,
            symbolic_resolver,
        ) {
            let out_info = TensorInfo::from_dtype_and_shape_scalars(out_dtype, &out_dims);
            return Ok(Box::new([(self.output, out_info)].into_iter()));
        }

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
        let out_info =
            TensorInfo::new_from_first_element_and_rank(first_elem, out_rank, symbolic_resolver);
        Ok(Box::new([(self.output, out_info)].into_iter()))
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        let data = &inputs[&self.data];
        let axes = if let Some(axes) = self.axes {
            Vec::<i64>::try_from(inputs[&axes].try_to_rank::<P1>()?)?
        } else {
            (0i64..(data.rank() as i64)).collect()
        };
        let axes = if axes.is_empty() {
            if self.noop_with_empty_axes {
                let out_tensor = data.clone();
                return Ok(Box::new([(self.output, out_tensor)].into_iter()));
            } else {
                (0i64..(data.rank() as i64)).collect::<Vec<_>>()
            }
        } else {
            axes
        };
        let axes = axes
            .into_iter()
            .map(|x| (if x < 0 { x + data.rank() as i64 } else { x }) as usize)
            .collect::<Vec<_>>();
        let out = data.reduce_mean(axes, self.keepdims, self.accumulation_mode, backend)?;
        Ok(Box::new([(self.output, out)].into_iter()))
    }

    fn backward(
        &self,
        output_grads: &HashMap<GlobalId, GlobalId>,
        graph: &mut MilliOpGraph,
        rng: &mut impl rand::Rng,
    ) -> Option<HashMap<GlobalId, GlobalId>> {
        let grad_output = *output_grads.get(&self.output)?;
        // ReduceMean backward: like ReduceSum but divide by number of reduced elements.
        // count = product of reduced dimensions = total_input_elems / total_output_elems
        // We compute: grad_input = expand(grad_output, input_shape) / count
        //
        // To get count dynamically: use Shape to get input shape, then ReduceProd
        // the relevant axes. Simpler: total_input / total_output.
        // Even simpler for the common cases: use Shape + Gather + ReduceProd.
        //
        // Approach: get input_shape, gather the reduced dims, product them = count,
        // then grad_input = expand(unsqueeze(grad_output), input_shape) / count

        let expanded_grad = if self.keepdims {
            grad_output
        } else if let Some(axes) = self.axes {
            super::Unsqueeze::push_new(graph, grad_output, axes, rng)
        } else {
            grad_output
        };
        let input_shape = super::Shape::push_new(graph, self.data, rng);
        let expanded = super::Expand::push_new(graph, expanded_grad, input_shape, rng);

        // Compute count of reduced elements: gather reduced dims from input_shape, then product
        if let Some(axes) = self.axes {
            // Gather the specific axes from input_shape, then ReduceProd
            let count = super::Gather::push_new(graph, input_shape, axes, 0, rng);
            let count_scalar = super::ReduceProd::push_new(graph, count, None, false, false, rng);
            let count_float =
                super::Cast::push_new(graph, count_scalar, crate::dtype::DType::F32, rng);
            let grad_input = super::SimpleBinary::div(graph, expanded, count_float, rng);
            let mut result = HashMap::new();
            result.insert(self.data, grad_input);
            Some(result)
        } else {
            // All axes reduced — count = total number of input elements
            // ReduceProd(input_shape) gives the total element count
            let total = super::ReduceProd::push_new(graph, input_shape, None, false, false, rng);
            let total_float = super::Cast::push_new(graph, total, crate::dtype::DType::F32, rng);
            let grad_input = super::SimpleBinary::div(graph, expanded, total_float, rng);
            let mut result = HashMap::new();
            result.insert(self.data, grad_input);
            Some(result)
        }
    }
}
