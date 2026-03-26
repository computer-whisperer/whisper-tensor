use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::migration::numeric_tensor::NumericTensor;
use crate::pool::Pool;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopK {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    input: GlobalId,
    k: GlobalId,
    output_values: GlobalId,
    output_indices: GlobalId,
    axis: i64,
    largest: bool,
    sorted: bool,
}

impl TopK {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        input: GlobalId,
        k: GlobalId,
        axis: i64,
        largest: bool,
        sorted: bool,
        rng: &mut impl Rng,
    ) -> (GlobalId, GlobalId) {
        Self::push_new_with_label(graph, input, k, axis, largest, sorted, None, rng)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        input: GlobalId,
        k: GlobalId,
        axis: i64,
        largest: bool,
        sorted: bool,
        label: Option<String>,
        rng: &mut impl Rng,
    ) -> (GlobalId, GlobalId) {
        let output_values = graph.get_new_tensor_id(rng);
        let output_indices = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            label,
            input,
            k,
            output_values,
            output_indices,
            axis,
            largest,
            sorted,
        };
        graph.push_op(AnyMilliOp::TopK(node));
        (output_values, output_indices)
    }
}

impl TopK {
    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.input, map);
        super::remap(&mut self.k, map);
        super::remap(&mut self.output_values, map);
        super::remap(&mut self.output_indices, map);
    }
}

impl Node for TopK {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "TopK".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.input, self.k].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.output_values, self.output_indices].into_iter())
    }
}

impl MilliOp for TopK {
    fn infer<'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'p, P>>,
        symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        _pool: &'p P,
    ) -> Result<Vec<(GlobalId, crate::tensor_info::TensorInfo<'p, P>)>, MilliOpGraphError> {
        use crate::numeric_dtype::NumericDType;
        use crate::scalar_info::ScalarInfoTyped;
        use crate::symbolic_scalar::SymbolicScalarTyped;
        use crate::tensor_info::TensorInfo;

        let input_info = known_inputs
            .get(&self.input)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let k_info = known_inputs
            .get(&self.k)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let val_dtype = input_info.dtype();
        let idx_dtype = NumericDType::I64;

        if let Some(ranked) = input_info.as_ranked() {
            let shape = ranked.shape();
            let rank = shape.len();
            let axis = if self.axis < 0 {
                (self.axis + rank as i64) as usize
            } else {
                self.axis as usize
            };

            // Try to get concrete k value
            let k_val = k_info.to_i64_vec().and_then(|v| v.first().copied());

            let mut out_dims = Vec::new();
            for (i, dim) in shape.iter().enumerate() {
                if i == axis {
                    if let Some(k) = k_val {
                        out_dims.push(ScalarInfoTyped::Numeric(k as u64));
                    } else {
                        out_dims.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(
                            symbolic_resolver,
                        )));
                    }
                } else {
                    out_dims.push(dim.clone());
                }
            }

            let val_info = TensorInfo::from_dtype_and_shape_scalars(val_dtype, &out_dims);
            let idx_info = TensorInfo::from_dtype_and_shape_scalars(idx_dtype, &out_dims);
            return Ok(vec![
                (self.output_values, val_info),
                (self.output_indices, idx_info),
            ]);
        }

        Err(MilliOpGraphError::UnableToInfer)
    }

    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        _config: &super::MilliEvalConfig,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        let (values, indices) = NumericTensor::<DynRank>::topk(
            &inputs[&self.input],
            &inputs[&self.k],
            self.axis,
            self.largest,
            self.sorted,
            backend,
        )?;
        Ok(Box::new(
            [(self.output_values, values), (self.output_indices, indices)].into_iter(),
        ))
    }

    fn eval_new<'p, P2: crate::pool::Pool + 'p>(
        &self,
        inputs: &[crate::numeric_tensor::NumericTensorView<'_, crate::tensor_rank::DynRank>],
        pool: &'p P2,
    ) -> Result<Vec<crate::numeric_tensor::NumericTensor<'p, crate::tensor_rank::DynRank, P2>>, crate::nano_graph::pool_eval::PoolEvalError> {
        use crate::numeric_scalar::NumericScalar;
        use crate::numeric_tensor::{NumericTensor, TensorLayout};
        use crate::tensor_rank::DynRank;

        let data = &inputs[0];
        let k_tensor = &inputs[1];
        let input_shape = data.shape();
        let rank = input_shape.len();
        let dtype = data.dtype();
        let k = k_tensor.read_element(0).to_i64() as usize;

        let axis = if self.axis < 0 { (self.axis + rank as i64) as usize } else { self.axis as usize };

        // Output shape: same as input but axis dim = k
        let mut output_shape = input_shape.clone();
        output_shape[axis] = k as u64;

        let val_layout = TensorLayout::<DynRank>::row_major(output_shape.clone(), dtype);
        let idx_layout = TensorLayout::<DynRank>::row_major(output_shape.clone(), crate::numeric_dtype::NumericDType::I64);
        let val_buf = pool.allocate(val_layout.buffer_size_bytes())
            .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
        let idx_buf = pool.allocate(idx_layout.buffer_size_bytes())
            .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
        let mut val_out = NumericTensor::from_parts(val_buf, val_layout);
        let mut idx_out = NumericTensor::from_parts(idx_buf, idx_layout);

        // Compute strides
        let mut in_strides = vec![1usize; rank];
        for i in (0..rank.saturating_sub(1)).rev() { in_strides[i] = in_strides[i + 1] * input_shape[i + 1] as usize; }
        let mut out_strides = vec![1usize; rank];
        for i in (0..rank.saturating_sub(1)).rev() { out_strides[i] = out_strides[i + 1] * output_shape[i + 1] as usize; }

        let axis_len = input_shape[axis] as usize;

        // Compute outer_size (product of dims before axis) and inner_size (product of dims after axis)
        let outer_size: usize = input_shape[..axis].iter().product::<u64>() as usize;
        let inner_size: usize = input_shape[axis + 1..].iter().product::<u64>().max(1) as usize;

        for outer in 0..outer_size.max(1) {
            for inner in 0..inner_size {
                // Collect elements along axis for this slice
                let mut elements: Vec<(f64, usize)> = Vec::with_capacity(axis_len);
                for ai in 0..axis_len {
                    let _flat = outer * in_strides.get(0).copied().unwrap_or(1).max(if axis == 0 { axis_len * inner_size } else { 1 })
                        // Compute flat index properly
                        ;
                    // Actually compute it correctly using strides
                    let mut in_flat = 0;
                    // outer dims contribute
                    if axis > 0 {
                        let mut outer_rem = outer;
                        for d in (0..axis).rev() {
                            let dim_below: usize = input_shape[d + 1..axis].iter().product::<u64>().max(1) as usize;
                            let coord = outer_rem / dim_below;
                            outer_rem %= dim_below;
                            in_flat += coord * in_strides[d];
                        }
                    }
                    in_flat += ai * in_strides[axis];
                    if axis + 1 < rank {
                        let mut inner_rem = inner;
                        for d in (axis + 1..rank).rev() {
                            let dim_below: usize = input_shape[d + 1..].iter().product::<u64>().max(1) as usize;
                            let coord = inner_rem / dim_below;
                            inner_rem %= dim_below;
                            in_flat += coord * in_strides[d];
                        }
                    }
                    elements.push((data.read_element(in_flat).to_f64(), ai));
                }

                // Sort
                if self.largest {
                    elements.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                } else {
                    elements.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                }

                // Write top k
                for ki in 0..k {
                    let (val, orig_idx) = elements[ki];
                    // Compute output flat index
                    let mut out_flat = 0;
                    if axis > 0 {
                        let mut outer_rem = outer;
                        for d in (0..axis).rev() {
                            let dim_below: usize = output_shape[d + 1..axis].iter().product::<u64>().max(1) as usize;
                            let coord = outer_rem / dim_below;
                            outer_rem %= dim_below;
                            out_flat += coord * out_strides[d];
                        }
                    }
                    out_flat += ki * out_strides[axis];
                    if axis + 1 < rank {
                        let mut inner_rem = inner;
                        for d in (axis + 1..rank).rev() {
                            let dim_below: usize = output_shape[d + 1..].iter().product::<u64>().max(1) as usize;
                            let coord = inner_rem / dim_below;
                            inner_rem %= dim_below;
                            out_flat += coord * out_strides[d];
                        }
                    }
                    val_out.write_element(out_flat, NumericScalar::from_f64(val).cast_to(dtype));
                    idx_out.write_element(out_flat, NumericScalar::from_i64(orig_idx as i64));
                }
            }
        }

        Ok(vec![val_out, idx_out])
    }
}
