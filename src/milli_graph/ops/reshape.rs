use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::pool::Pool;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reshape {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    data: GlobalId,
    shape: GlobalId,
    allowzero: bool,
}

impl Reshape {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        shape: GlobalId,
        allowzero: bool,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        Self::push_new_with_label(graph, data, shape, allowzero, None, rng)
    }

    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        shape: GlobalId,
        allowzero: bool,
        label: Option<String>,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            label,
            output,
            data,
            shape,
            allowzero,
        };
        graph.push_op(AnyMilliOp::Reshape(node));
        output
    }

    fn calculate_new_shape(
        &self,
        data_input_shape: &[u64],
        shape_input_value: &[i64],
    ) -> Result<Vec<u64>, MilliOpGraphError> {
        let mut new_shape_dims = vec![];
        let mut backfill_dim: Option<usize> = None;
        for i in 0..shape_input_value.len() {
            new_shape_dims.push(if shape_input_value[i] == 0 {
                data_input_shape[i]
            } else if shape_input_value[i] == -1 {
                if backfill_dim.is_some() {
                    // Only one dimension can be inferred
                    Err(MilliOpGraphError::InvalidInput("Reshape".to_string()))?
                }
                backfill_dim = Some(i);
                1
            } else if shape_input_value[i] < -1 {
                Err(MilliOpGraphError::InvalidInput("Reshape".to_string()))?
            } else {
                shape_input_value[i] as u64
            });
        }

        // Backfill the inferred dimension
        if let Some(i) = backfill_dim {
            let total_input_size = data_input_shape.iter().product::<u64>();

            // Calculate the current product of the dimensions
            let mut current_product = 1;
            for (j, dim) in new_shape_dims.iter().enumerate() {
                if j != i {
                    current_product *= dim;
                }
            }
            // Calculate the inferred dimension size
            let inferred_size = total_input_size / current_product;
            new_shape_dims[i] = inferred_size;
        }
        let output_shape = new_shape_dims;

        // Verify that the dimensions are compatible
        if output_shape.iter().product::<u64>() != data_input_shape.iter().product::<u64>() {
            Err(MilliOpGraphError::InvalidInput("Reshape".to_string()))?
        }

        Ok(output_shape)
    }
}

impl Reshape {
    pub fn lower_to_nano<'p, P: crate::pool::Pool + 'p>(
        &self,
        ctx: &mut crate::nano_graph::NanoLoweringContext<'_, 'p, P>,
    ) -> crate::milli_graph::ops::LowerResult {
        ctx.lower_view_op(self)
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.data, map);
        super::remap(&mut self.shape, map);
    }
}

impl Node for Reshape {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Reshape".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.data, self.shape].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output].into_iter())
    }
}

impl MilliOp for Reshape {
    fn infer<'a, 'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'a, 'p, P>>,
        rng: &mut impl Rng,
        pool: &'p P,
    ) -> Result<Vec<(GlobalId, crate::tensor_info::TensorInfo<'a, 'p, P>)>, MilliOpGraphError>
    where
        'p: 'a,
    {
        use crate::scalar_info::ScalarInfoTyped;
        use crate::symbolic_scalar::SymbolicScalarTyped;
        use crate::tensor_info::TensorInfo;

        let data_info = known_inputs
            .get(&self.data)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let shape_info = known_inputs
            .get(&self.shape)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        // Build output hint: same dtype as data, rank from shape tensor length.
        let out_dtype = data_info.dtype();

        let data_shape_full: Option<Vec<ScalarInfoTyped<u64>>> =
            data_info.as_ranked().map(|r| r.shape().to_vec());

        // Helper: given concrete shape-tensor values, produce the output dims.
        // For ONNX 0-values, copy the input dim through — crucially keeping
        // the full `ScalarInfoTyped` so a Symbolic input dim survives as the
        // same Symbolic in the output (this is what ties batch sym identity
        // across the graph).
        let mut compose_reshape_dims = |shape_values: &[i64]| -> Vec<ScalarInfoTyped<u64>> {
            let mut dims: Vec<ScalarInfoTyped<u64>> = Vec::with_capacity(shape_values.len());
            let mut has_minus_one = false;
            for (i, &sv) in shape_values.iter().enumerate() {
                if sv == 0 {
                    // Copy-through semantic: preserve identity (Numeric or Symbolic).
                    let inherited = data_shape_full.as_ref().and_then(|ds| ds.get(i).cloned());
                    dims.push(inherited.unwrap_or_else(|| {
                        ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(rng))
                    }));
                } else if sv == -1 {
                    has_minus_one = true;
                    dims.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(rng)));
                } else if sv > 0 {
                    dims.push(ScalarInfoTyped::Numeric(sv as u64));
                } else {
                    dims.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(rng)));
                }
            }
            // Try to resolve the -1 dim. The -1 position is constrained by
            // total-elements balance:
            //   prod(input_dims) = prod(output_dims_with_-1_resolved)
            // If the symbolic factors on both sides cancel (same multiset of
            // symbol_ids appears in input and in the non-(-1) output dims),
            // the -1 resolves to a purely numeric value — prod(num_in) /
            // prod(num_out). Otherwise leave it as the fresh Symbolic minted
            // above (still correct, just less informative).
            if has_minus_one
                && let Some(ds) = data_shape_full.as_ref()
            {
                use std::collections::HashMap;
                let collect_syms = |vals: &[ScalarInfoTyped<u64>]| -> HashMap<u64, u32> {
                    let mut map: HashMap<u64, u32> = HashMap::new();
                    for v in vals {
                        if let ScalarInfoTyped::Symbolic(s) = v {
                            *map.entry(s.symbol_id()).or_insert(0) += 1;
                        }
                    }
                    map
                };
                let output_non_minus: Vec<ScalarInfoTyped<u64>> = dims
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| shape_values[*idx] != -1)
                    .map(|(_, d)| d.clone())
                    .collect();
                let input_syms = collect_syms(ds);
                let output_syms = collect_syms(&output_non_minus);
                if input_syms == output_syms {
                    // Syms cancel; -1 is determined purely by the numeric parts.
                    let in_num: u64 =
                        ds.iter().filter_map(|d| d.as_numeric().copied()).product();
                    let out_num: u64 = output_non_minus
                        .iter()
                        .filter_map(|d| d.as_numeric().copied())
                        .product();
                    if out_num > 0 && in_num % out_num == 0 {
                        let inferred = in_num / out_num;
                        let idx = shape_values.iter().position(|&v| v == -1).unwrap();
                        dims[idx] = ScalarInfoTyped::Numeric(inferred);
                    }
                }
            }
            dims
        };

        let output_hint = if let Some(shape_values) = shape_info.to_i64_vec() {
            let hint_dims = compose_reshape_dims(&shape_values);
            TensorInfo::from_dtype_and_shape_scalars(out_dtype, &hint_dims)
        } else {
            // Shape tensor not concrete — just provide dtype hint.
            TensorInfo::from_dtype_and_shape_scalars(out_dtype, &[])
        };

        // If both inputs are concrete, try constant fold via nano+pool_eval path.
        if let Some(results) =
            super::constant_fold(self, known_inputs, &[(self.output, output_hint)], pool)
        {
            return Ok(results);
        }

        let first_elem = data_info.first_element();

        // If shape tensor is concrete, we can determine the output shape.
        if let Some(shape_values) = shape_info.to_i64_vec() {
            if shape_values.iter().any(|&v| v < -1) {
                return Err(MilliOpGraphError::InvalidInput("Reshape".to_string()));
            }
            let output_dims = compose_reshape_dims(&shape_values);

            let out = TensorInfo::Ranked(crate::tensor_info::TensorInfoRanked::new(
                first_elem,
                output_dims,
                rng,
            ));
            return Ok(vec![(self.output, out)]);
        }

        // Shape tensor is not concrete. Try to get output rank from shape tensor's
        // rank (shape tensor is 1-D, so its length = output rank)
        if let Some(shape_ranked) = shape_info.as_ranked() {
            let shape_shape = shape_ranked.shape();
            if shape_shape.len() == 1
                && let Some(output_rank) = shape_shape[0].as_numeric()
            {
                let out = TensorInfo::new_from_first_element_and_rank(
                    first_elem,
                    ScalarInfoTyped::Numeric(*output_rank as u32),
                    rng,
                );
                return Ok(vec![(self.output, out)]);
            }
        }

        // Fallback: propagate dtype only
        let out = TensorInfo::new_from_first_element_and_rank(
            first_elem,
            ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(rng)),
            rng,
        );
        Ok(vec![(self.output, out)])
    }

    fn backward(
        &self,
        output_grads: &HashMap<GlobalId, GlobalId>,
        graph: &mut MilliOpGraph,
        rng: &mut impl rand::Rng,
    ) -> Option<HashMap<GlobalId, GlobalId>> {
        let grad_output = *output_grads.get(&self.output)?;
        // Reshape grad back to input shape
        let input_shape = super::Shape::push_new(graph, self.data, rng);
        let grad_input = Reshape::push_new(graph, grad_output, input_shape, false, rng);
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
        use crate::numeric_tensor::{NumericTensor, TensorLayout};
        use crate::tensor_rank::DynRank;

        let data = &inputs[0];
        let shape_tensor = &inputs[1];
        let dtype = data.dtype();
        let numel = data.numel();

        // Read shape values from inputs[1]
        let shape_values: Vec<i64> = (0..shape_tensor.numel())
            .map(|i| shape_tensor.read_element(i).to_i64())
            .collect();

        let data_shape: Vec<u64> = data.shape().clone();
        let output_shape = self
            .calculate_new_shape(&data_shape, &shape_values)
            .map_err(|e| {
                crate::nano_graph::pool_eval::PoolEvalError::Unsupported(format!("{e:?}"))
            })?;

        let layout = TensorLayout::<DynRank>::row_major(output_shape, dtype);
        let buf = pool
            .allocate(layout.buffer_size_bytes())
            .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
        let mut out = NumericTensor::from_parts(buf, layout);
        for i in 0..numel {
            out.write_element(i, data.read_element(i));
        }
        Ok(vec![out])
    }

    fn lower_to_nano<'p, P: crate::pool::Pool + 'p>(
        &self,
        ctx: &mut crate::nano_graph::NanoLoweringContext<'_, 'p, P>,
    ) -> crate::milli_graph::ops::LowerResult {
        Reshape::lower_to_nano(self, ctx)
    }
}
