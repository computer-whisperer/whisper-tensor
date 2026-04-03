use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::pool::Pool;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Compress: selects elements/slices from input where condition is true.
///
/// Output shape is dynamic (depends on count of true values in condition).
/// axis=None: flatten input, select elements where condition[i] is true.
/// axis=Some(a): select slices along axis a where condition[i] is true.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Compress {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    input: GlobalId,
    condition: GlobalId,
    axis: Option<i64>,
}

impl Compress {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        input: GlobalId,
        condition: GlobalId,
        axis: Option<i64>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            label: None,
            output,
            input,
            condition,
            axis,
        };
        graph.push_op(AnyMilliOp::Compress(node));
        output
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.input, map);
        super::remap(&mut self.condition, map);
    }
}

impl Node for Compress {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Compress".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.input, self.condition].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl MilliOp for Compress {
    fn infer<'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'p, P>>,
        symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        pool: &'p P,
    ) -> Result<Vec<(GlobalId, crate::tensor_info::TensorInfo<'p, P>)>, MilliOpGraphError> {
        let input_info = known_inputs
            .get(&self.input)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let cond_info = known_inputs
            .get(&self.condition)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        let input_ranked = input_info
            .as_ranked()
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let input_shape = input_ranked.shape();
        let dtype = input_info.dtype();

        // Try to constant-fold when condition is concrete.
        if let Some(cond_concrete) = cond_info.as_concrete() {
            let true_count = (0..cond_concrete.numel())
                .filter(|&i| cond_concrete.read_element(i).to_i64() != 0)
                .count() as u64;

            let out_shape = if let Some(axis_val) = self.axis {
                let rank = input_shape.len();
                let axis = if axis_val < 0 {
                    (axis_val + rank as i64) as usize
                } else {
                    axis_val as usize
                };
                let mut s: Vec<crate::scalar_info::ScalarInfoTyped<u64>> = input_shape.clone();
                s[axis] = crate::scalar_info::ScalarInfoTyped::Numeric(true_count);
                s
            } else {
                vec![crate::scalar_info::ScalarInfoTyped::Numeric(true_count)]
            };

            let out_info =
                crate::tensor_info::TensorInfo::from_dtype_and_shape_scalars(dtype, &out_shape);

            // If all inputs are concrete, try to fully constant-fold.
            if let Some(results) = super::constant_fold(
                self,
                known_inputs,
                &[(self.output, out_info.clone_with_pool(pool))],
                pool,
            ) {
                return Ok(results);
            }

            return Ok(vec![(self.output, out_info)]);
        }

        // Condition not concrete — output size is unknown.
        // Return shape with symbolic dim.
        let out_shape = if let Some(axis_val) = self.axis {
            let rank = input_shape.len();
            let axis = if axis_val < 0 {
                (axis_val + rank as i64) as usize
            } else {
                axis_val as usize
            };
            let mut s: Vec<crate::scalar_info::ScalarInfoTyped<u64>> = input_shape.clone();
            s[axis] = crate::scalar_info::ScalarInfoTyped::Symbolic(
                crate::symbolic_scalar::SymbolicScalarTyped::new(symbolic_resolver),
            );
            s
        } else {
            vec![crate::scalar_info::ScalarInfoTyped::Symbolic(
                crate::symbolic_scalar::SymbolicScalarTyped::new(symbolic_resolver),
            )]
        };

        let out_info =
            crate::tensor_info::TensorInfo::from_dtype_and_shape_scalars(dtype, &out_shape);
        Ok(vec![(self.output, out_info)])
    }

    fn eval(
        &self,
        inputs: &HashMap<
            GlobalId,
            crate::migration::numeric_tensor::NumericTensor<crate::tensor_rank::DynRank>,
        >,
        _config: &super::MilliEvalConfig,
        _backend: &mut crate::backends::eval_backend::EvalBackend,
    ) -> super::EvalResult {
        let pool_tensors: Vec<_> = [self.input, self.condition]
            .iter()
            .map(|id| {
                crate::nano_graph::lower::legacy_numeric_to_new(
                    &inputs[id],
                    &crate::pool::SystemPool,
                )
            })
            .collect();
        let view_refs: Vec<_> = pool_tensors.iter().map(|t| t.view()).collect();
        let results = self
            .eval_new(&view_refs, &crate::pool::SystemPool)
            .map_err(|e| MilliOpGraphError::InvalidInput(format!("{e}")))?;
        let output = self.output;
        Ok(Box::new(results.into_iter().map(move |t| {
            (output, crate::nano_graph::lower::new_numeric_to_legacy(&t))
        })))
    }

    fn eval_new<'p, P2: Pool + 'p>(
        &self,
        inputs: &[crate::numeric_tensor::NumericTensorView<'_, crate::tensor_rank::DynRank>],
        pool: &'p P2,
    ) -> Result<
        Vec<crate::numeric_tensor::NumericTensor<'p, crate::tensor_rank::DynRank, P2>>,
        crate::nano_graph::pool_eval::PoolEvalError,
    > {
        use crate::numeric_tensor::{NumericTensor, TensorLayout};
        use crate::tensor_rank::DynRank;

        let input = &inputs[0];
        let condition = &inputs[1];
        let input_shape = input.shape();
        let dtype = input.dtype();

        // Collect condition booleans.
        let cond_len = condition.numel();
        let cond_bool: Vec<bool> = (0..cond_len)
            .map(|i| condition.read_element(i).to_i64() != 0)
            .collect();

        if let Some(axis_val) = self.axis {
            let rank = input_shape.len();
            let axis = if axis_val < 0 {
                (axis_val + rank as i64) as usize
            } else {
                axis_val as usize
            };

            let axis_size = input_shape[axis] as usize;
            let selected: Vec<usize> = cond_bool
                .iter()
                .take(axis_size)
                .enumerate()
                .filter(|&(_, &v)| v)
                .map(|(i, _)| i)
                .collect();

            let outer_size: usize = input_shape[..axis].iter().product::<u64>() as usize;
            let inner_size: usize = input_shape[axis + 1..].iter().product::<u64>().max(1) as usize;
            let outer_stride = axis_size * inner_size;

            let mut out_shape: Vec<u64> = input_shape.to_vec();
            out_shape[axis] = selected.len() as u64;
            let out_numel: usize = out_shape.iter().product::<u64>() as usize;

            let layout = TensorLayout::<DynRank>::row_major(out_shape, dtype);
            let buf = pool
                .allocate(layout.buffer_size_bytes())
                .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
            let mut out = NumericTensor::from_parts(buf, layout);

            let mut out_idx = 0;
            for outer in 0..outer_size {
                for &sel_idx in &selected {
                    let base = outer * outer_stride + sel_idx * inner_size;
                    for inner in 0..inner_size {
                        out.write_element(out_idx, input.read_element(base + inner));
                        out_idx += 1;
                    }
                }
            }
            debug_assert_eq!(out_idx, out_numel);

            Ok(vec![out])
        } else {
            // No axis: flatten input, select elements where condition is true.
            let flat_len = input.numel();
            let mut out_data = Vec::new();
            for i in 0..flat_len {
                let c = if i < cond_bool.len() {
                    cond_bool[i]
                } else {
                    false
                };
                if c {
                    out_data.push(input.read_element(i));
                }
            }

            let out_shape = vec![out_data.len() as u64];
            let layout = TensorLayout::<DynRank>::row_major(out_shape, dtype);
            let buf = pool
                .allocate(layout.buffer_size_bytes())
                .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
            let mut out = NumericTensor::from_parts(buf, layout);
            for (i, val) in out_data.iter().enumerate() {
                out.write_element(i, *val);
            }

            Ok(vec![out])
        }
    }
}
