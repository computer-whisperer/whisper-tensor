use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::graph::{GlobalId, Node};
use crate::migration::numeric_tensor::NumericTensor;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::nano_graph::lower::{NanoLoweringContext, TensorAtomMap};
use crate::nano_graph::ops::ScalarOp;
use crate::nano_graph::pattern::AtomId;
use crate::pool::Pool;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Expand {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    input: GlobalId,
    shape: GlobalId,
}

impl Expand {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        input: GlobalId,
        shape: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new_with_label(graph, input, shape, None, rng)
    }

    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        input: GlobalId,
        shape: GlobalId,
        label: Option<String>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            label,
            output,
            input,
            shape,
        };
        graph.push_op(AnyMilliOp::Expand(node));
        output
    }
}

impl Expand {
    pub fn lower_to_nano(
        &self,
        ctx: &mut crate::nano_graph::NanoLoweringContext,
    ) -> crate::milli_graph::ops::LowerResult {
        let all_infos = ctx.all_infos;
        let in_id = Node::inputs(self).next().unwrap();
        let out_id = Node::outputs(self).next().unwrap();

        let Some(in_map) = ctx.tensor_map.get(&in_id).cloned() else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };
        let Some(out_info) = all_infos.get(&out_id) else {
            ctx.register_opaque(out_id);
            return crate::milli_graph::ops::LowerResult::Lowered;
        };
        let in_info = all_infos.get(&in_id);

        let Some((layout, known_dims, sym_dims, count)) = ctx.classify_dims(out_info) else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };
        let count = count.max(1);

        let dt = NanoLoweringContext::ndt(out_info);
        if count == in_map.count {
            ctx.tensor_map.insert(
                out_id,
                TensorAtomMap::simple(
                    in_map.base_id,
                    count,
                    dt,
                    layout,
                    TensorAtomMap::compute_strides(&known_dims),
                    sym_dims,
                ),
            );
        } else {
            let out_tmp = TensorAtomMap::simple(
                AtomId(0),
                count,
                dt,
                layout.clone(),
                TensorAtomMap::compute_strides(&known_dims),
                sym_dims.clone(),
            );
            let input_ref =
                ctx.compute_input_ref(&out_tmp, &in_map, out_info, in_info.unwrap_or(out_info));

            let base_id = ctx.nano.push_group(
                count,
                dt,
                ScalarOp::Identity,
                sym_dims.clone(),
                vec![input_ref],
            );

            ctx.tensor_map.insert(
                out_id,
                TensorAtomMap::simple(
                    base_id,
                    count,
                    dt,
                    layout,
                    TensorAtomMap::compute_strides(&known_dims),
                    sym_dims,
                ),
            );
        }
        crate::milli_graph::ops::LowerResult::Lowered
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.input, map);
        super::remap(&mut self.shape, map);
    }
}

impl crate::graph::Node for Expand {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Expand".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.input, self.shape].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
}

impl MilliOp for Expand {
    fn infer<'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'p, P>>,
        symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        pool: &'p P,
    ) -> Result<Vec<(GlobalId, crate::tensor_info::TensorInfo<'p, P>)>, MilliOpGraphError> {
        use crate::scalar_info::ScalarInfoTyped;
        use crate::symbolic_scalar::SymbolicScalarTyped;
        use crate::tensor_info::TensorInfo;

        let input_info = known_inputs
            .get(&self.input)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let shape_info = known_inputs
            .get(&self.shape)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        // Build output hint: same dtype as input, shape from target shape tensor.
        let out_dtype = input_info.dtype();
        let output_hint = if let Some(shape_values) = shape_info.to_i64_vec() {
            let target_shape: Vec<ScalarInfoTyped<u64>> = shape_values
                .iter()
                .map(|&v| ScalarInfoTyped::Numeric(v as u64))
                .collect();
            // Broadcast with input shape if known (take max of each dim).
            if let Some(input_ranked) = input_info.as_ranked() {
                let input_shape = input_ranked.shape();
                let output_rank = target_shape.len().max(input_shape.len());
                let mut final_shape: Vec<ScalarInfoTyped<u64>> = Vec::new();
                for i in 0..output_rank {
                    let target_i = (i as i64 - output_rank as i64) + target_shape.len() as i64;
                    let input_i = (i as i64 - output_rank as i64) + input_shape.len() as i64;
                    let target_dim = if target_i >= 0 {
                        Some(target_shape[target_i as usize].clone())
                    } else {
                        None
                    };
                    let input_dim = if input_i >= 0 {
                        Some(input_shape[input_i as usize].clone())
                    } else {
                        None
                    };
                    let dim = match (target_dim, input_dim) {
                        (
                            Some(ScalarInfoTyped::Numeric(t)),
                            Some(ScalarInfoTyped::Numeric(inp)),
                        ) => ScalarInfoTyped::Numeric(t.max(inp)),
                        (Some(t), None) => t,
                        (None, Some(inp)) => inp,
                        (Some(ScalarInfoTyped::Numeric(t)), Some(_)) => ScalarInfoTyped::Numeric(t),
                        (Some(_), Some(ScalarInfoTyped::Numeric(inp))) => {
                            ScalarInfoTyped::Numeric(inp)
                        }
                        _ => ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)),
                    };
                    final_shape.push(dim);
                }
                TensorInfo::from_dtype_and_shape_scalars(out_dtype, &final_shape)
            } else {
                TensorInfo::from_dtype_and_shape_scalars(out_dtype, &target_shape)
            }
        } else {
            TensorInfo::from_dtype_and_shape_scalars(out_dtype, &[])
        };

        // If both inputs are concrete, try constant fold via nano+pool_eval path.
        if let Some(results) =
            super::constant_fold(self, known_inputs, &[(self.output, output_hint)], pool)
        {
            return Ok(results);
        }

        let first_elem = input_info.first_element();

        // If shape tensor is concrete, use its values for output shape
        if let Some(shape_values) = shape_info.to_i64_vec() {
            let output_shape: Vec<ScalarInfoTyped<u64>> = shape_values
                .iter()
                .map(|&v| ScalarInfoTyped::Numeric(v as u64))
                .collect();

            // Like eval: broadcast with input shape (take max of each dim)
            if let Some(input_ranked) = input_info.as_ranked() {
                let input_shape = input_ranked.shape();
                let output_rank = output_shape.len().max(input_shape.len());
                let mut final_shape: Vec<ScalarInfoTyped<u64>> = Vec::new();
                for i in 0..output_rank {
                    let target_i = (i as i64 - output_rank as i64) + output_shape.len() as i64;
                    let input_i = (i as i64 - output_rank as i64) + input_shape.len() as i64;

                    let target_dim = if target_i >= 0 {
                        Some(output_shape[target_i as usize].clone())
                    } else {
                        None
                    };
                    let input_dim = if input_i >= 0 {
                        Some(input_shape[input_i as usize].clone())
                    } else {
                        None
                    };

                    let dim = match (target_dim, input_dim) {
                        (
                            Some(ScalarInfoTyped::Numeric(t)),
                            Some(ScalarInfoTyped::Numeric(inp)),
                        ) => ScalarInfoTyped::Numeric(t.max(inp)),
                        (Some(t), None) => t,
                        (None, Some(inp)) => inp,
                        (Some(ScalarInfoTyped::Numeric(t)), Some(_)) => {
                            // Target is known, input is symbolic -- use target
                            ScalarInfoTyped::Numeric(t)
                        }
                        (Some(_), Some(ScalarInfoTyped::Numeric(inp))) => {
                            ScalarInfoTyped::Numeric(inp)
                        }
                        _ => ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)),
                    };
                    final_shape.push(dim);
                }

                let out = TensorInfo::Ranked(crate::tensor_info::TensorInfoRanked::new(
                    first_elem,
                    final_shape,
                    symbolic_resolver,
                ));
                return Ok(vec![((self.output, out))]);
            }

            // Input shape not known, but target shape is
            let out = TensorInfo::Ranked(crate::tensor_info::TensorInfoRanked::new(
                first_elem,
                output_shape,
                symbolic_resolver,
            ));
            return Ok(vec![((self.output, out))]);
        }

        // Shape tensor not concrete. Try to get output rank from shape tensor length.
        if let Some(shape_ranked) = shape_info.as_ranked() {
            let shape_shape = shape_ranked.shape();
            if shape_shape.len() == 1
                && let Some(&output_rank) = shape_shape[0].as_numeric()
            {
                let out = TensorInfo::new_from_first_element_and_rank(
                    first_elem,
                    ScalarInfoTyped::Numeric(output_rank as u32),
                    symbolic_resolver,
                );
                return Ok(vec![((self.output, out))]);
            }
        }

        // Fallback: propagate dtype only
        let out = TensorInfo::new_from_first_element_and_rank(
            first_elem,
            ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)),
            symbolic_resolver,
        );
        Ok(vec![((self.output, out))])
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        _config: &super::MilliEvalConfig,
        _backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        let shape: Vec<i64> = inputs[&self.shape].try_to_rank::<P1>()?.try_into()?;
        let shape_u = shape.iter().map(|x| *x as u64).collect::<Vec<u64>>();
        let mut x = inputs[&self.input].clone();
        while x.rank() < shape_u.len() {
            x = x.unsqueeze(0)?;
        }
        let shape_u = shape_u
            .iter()
            .zip(x.shape().iter())
            .map(|(a, b)| std::cmp::max(*a, *b))
            .collect::<Vec<u64>>();
        let out = x.expand(&shape_u)?;
        Ok(Box::new([(self.output, out)].into_iter()))
    }

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
        let shape_tensor = &inputs[1];
        let dtype = data.dtype();
        let input_shape = data.shape();

        // Read target shape
        let target_shape: Vec<u64> = (0..shape_tensor.numel())
            .map(|i| shape_tensor.read_element(i).to_i64() as u64)
            .collect();

        // Compute output shape with broadcasting (max of input and target per dim)
        let output_rank = target_shape.len().max(input_shape.len());
        let mut output_shape = vec![0u64; output_rank];
        let mut padded_input = vec![1u64; output_rank];
        for (i, &d) in input_shape.iter().enumerate() {
            padded_input[output_rank - input_shape.len() + i] = d;
        }
        for i in 0..output_rank {
            let t = if i < output_rank - target_shape.len() {
                1
            } else {
                target_shape[i - (output_rank - target_shape.len())]
            };
            output_shape[i] = t.max(padded_input[i]);
        }

        let out_numel: usize = output_shape.iter().product::<u64>() as usize;

        // Compute strides for output and padded input
        let mut out_strides = vec![1usize; output_rank];
        for i in (0..output_rank.saturating_sub(1)).rev() {
            out_strides[i] = out_strides[i + 1] * output_shape[i + 1] as usize;
        }
        let mut in_strides = vec![1usize; output_rank];
        for i in (0..output_rank.saturating_sub(1)).rev() {
            in_strides[i] = in_strides[i + 1] * padded_input[i + 1] as usize;
        }

        let layout = TensorLayout::<DynRank>::row_major(output_shape.clone(), dtype);
        let buf = pool
            .allocate(layout.buffer_size_bytes())
            .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
        let mut out = NumericTensor::from_parts(buf, layout);

        for out_flat in 0..out_numel {
            let mut rem = out_flat;
            let mut in_flat = 0usize;
            for d in 0..output_rank {
                let coord = rem / out_strides[d];
                rem %= out_strides[d];
                // Broadcast: if input dim is 1, always use coord 0
                let in_coord = if padded_input[d] == 1 { 0 } else { coord };
                in_flat += in_coord * in_strides[d];
            }
            out.write_element(out_flat, data.read_element(in_flat));
        }

        Ok(vec![out])
    }

    fn lower_to_nano(
        &self,
        ctx: &mut crate::nano_graph::NanoLoweringContext,
    ) -> crate::milli_graph::ops::LowerResult {
        Expand::lower_to_nano(self, ctx)
    }
}
