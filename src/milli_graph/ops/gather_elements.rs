use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::pool::Pool;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// GatherElements: gather values along an axis using element-wise indices.
///
/// output shape = indices shape.
/// output[i][j][k] = data[indices[i][j][k]][j][k]  (for axis=0), etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatherElements {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    data: GlobalId,
    indices: GlobalId,
    axis: i64,
}

impl GatherElements {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        indices: GlobalId,
        axis: i64,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            label: None,
            output,
            data,
            indices,
            axis,
        };
        graph.push_op(AnyMilliOp::GatherElements(node));
        output
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.data, map);
        super::remap(&mut self.indices, map);
    }
}

impl Node for GatherElements {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "GatherElements".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.data, self.indices].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl MilliOp for GatherElements {
    fn infer<'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'p, P>>,
        _symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        pool: &'p P,
    ) -> Result<Vec<(GlobalId, crate::tensor_info::TensorInfo<'p, P>)>, MilliOpGraphError> {
        let data_info = known_inputs
            .get(&self.data)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let indices_info = known_inputs
            .get(&self.indices)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        // Output shape = indices shape, dtype = data dtype.
        let indices_ranked = indices_info
            .as_ranked()
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let out_dims = indices_ranked.shape();
        let out_dtype = data_info.dtype();
        let out_info =
            crate::tensor_info::TensorInfo::from_dtype_and_shape_scalars(out_dtype, &out_dims);

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

    fn eval(
        &self,
        inputs: &HashMap<
            GlobalId,
            crate::migration::numeric_tensor::NumericTensor<crate::tensor_rank::DynRank>,
        >,
        _config: &super::MilliEvalConfig,
        _backend: &mut crate::backends::eval_backend::EvalBackend,
    ) -> super::EvalResult {
        let views: Vec<_> = [self.data, self.indices]
            .iter()
            .map(|id| crate::symbolic_graph::SharedPoolTensor::from_legacy(&inputs[id]))
            .collect();
        let view_refs: Vec<_> = views.iter().map(|s| s.0.view()).collect();
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

        let data = &inputs[0];
        let indices = &inputs[1];
        let data_shape = data.shape();
        let indices_shape = indices.shape();
        let rank = data_shape.len();
        let dtype = data.dtype();
        let axis = if self.axis < 0 {
            (self.axis + rank as i64) as usize
        } else {
            self.axis as usize
        };

        let total: usize = indices_shape.iter().product::<u64>() as usize;

        // Compute strides
        let mut data_strides = vec![1usize; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            data_strides[i] = data_strides[i + 1] * data_shape[i + 1] as usize;
        }
        let mut indices_strides = vec![1usize; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            indices_strides[i] = indices_strides[i + 1] * indices_shape[i + 1] as usize;
        }

        let layout = TensorLayout::<DynRank>::row_major(indices_shape.to_vec(), dtype);
        let buf = pool
            .allocate(layout.buffer_size_bytes())
            .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
        let mut out = NumericTensor::from_parts(buf, layout);

        for flat_idx in 0..total {
            // Convert flat index to multi-dimensional index in indices tensor
            let mut multi_idx = vec![0usize; rank];
            let mut remaining = flat_idx;
            for d in 0..rank {
                multi_idx[d] = remaining / indices_strides[d];
                remaining %= indices_strides[d];
            }

            let mut gather_idx = indices.read_element(flat_idx).to_i64();
            if gather_idx < 0 {
                gather_idx += data_shape[axis] as i64;
            }

            let mut data_flat_idx = 0usize;
            for d in 0..rank {
                let dim_idx = if d == axis {
                    gather_idx as usize
                } else {
                    multi_idx[d]
                };
                data_flat_idx += dim_idx * data_strides[d];
            }

            out.write_element(flat_idx, data.read_element(data_flat_idx));
        }

        Ok(vec![out])
    }

    fn lower_to_nano(
        &self,
        ctx: &mut crate::nano_graph::NanoLoweringContext,
    ) -> super::LowerResult {
        use crate::nano_graph::lower::{DimKind, TensorAtomMap};
        use crate::nano_graph::ops::{ScalarBinOp, ScalarOp};
        use crate::nano_graph::pattern::InputRef;
        use crate::numeric_dtype::NumericDType;
        use crate::numeric_scalar::NumericScalar;

        let data_id = self.data;
        let indices_id = self.indices;
        let out_id = self.output;

        let Some(data_map) = ctx.tensor_map.get(&data_id).cloned() else {
            return super::LowerResult::Unsupported;
        };
        let Some(indices_map) = ctx.tensor_map.get(&indices_id).cloned() else {
            return super::LowerResult::Unsupported;
        };
        let Some(out_info) = ctx.all_infos.get(&out_id) else {
            return super::LowerResult::Unsupported;
        };
        let Some(data_info) = ctx.all_infos.get(&data_id) else {
            return super::LowerResult::Unsupported;
        };

        let data_rank = match data_info.rank_if_known() {
            Some(r) => r,
            None => return super::LowerResult::Unsupported,
        };

        let axis = if self.axis < 0 {
            (self.axis + data_rank as i64) as usize
        } else {
            self.axis as usize
        };

        // Only handle axis=0 and 2D data for now.
        if axis != 0 || data_rank != 2 {
            return super::LowerResult::Unsupported;
        }

        let data_known: Vec<u64> = data_map
            .layout
            .iter()
            .filter_map(|d| {
                if let DimKind::Known(s) = d {
                    Some(*s)
                } else {
                    None
                }
            })
            .collect();
        if data_known.len() != data_rank {
            return super::LowerResult::Unsupported;
        }

        let cols = data_known[1];

        let Some((out_layout, out_known_dims, out_sym_dims, out_count)) =
            ctx.classify_dims(out_info)
        else {
            return super::LowerResult::Unsupported;
        };
        let out_count = out_count.max(1);
        let out_dt = crate::nano_graph::NanoLoweringContext::ndt(out_info);

        // For axis=0, 2D data: flat_index_into_data = indices[i] * cols + (i % cols)
        // But for GatherElements the output shape = indices shape and we iterate all elements.
        // Each element i: row = indices[i], col = i % cols
        // flat_data_idx = row * cols + col
        //
        // Step 1: stride literal
        let stride_lit = ctx.nano.push_atom(
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(cols as f32)),
            vec![],
            vec![],
        );

        // Column offset literals: 0..cols
        let col_lit_base = ctx.nano.push_atom(
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(0.0)),
            vec![],
            vec![],
        );
        for j in 1..cols {
            ctx.nano.push_atom(
                NumericDType::F32,
                ScalarOp::Literal(NumericScalar::from_f32(j as f32)),
                vec![],
                vec![],
            );
        }

        // Mul group: indices[i] * cols
        let mul_base = ctx.nano.push_group(
            out_count,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: NumericDType::F32,
            },
            out_sym_dims.clone(),
            vec![
                InputRef::affine(indices_map.base_id, 1),
                InputRef::Broadcast(stride_lit),
            ],
        );

        // Add group: (indices[i] * cols) + (i % cols)
        let add_base = ctx.nano.push_group(
            out_count,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            out_sym_dims.clone(),
            vec![
                InputRef::affine(mul_base, 1),
                InputRef::modular(col_lit_base, 1, cols),
            ],
        );

        // IndirectLoad group
        let base_id = ctx.nano.push_group(
            out_count,
            out_dt,
            ScalarOp::IndirectLoad {
                table_base: data_map.base_id,
            },
            out_sym_dims.clone(),
            vec![InputRef::affine(add_base, 1)],
        );

        let out_strides = TensorAtomMap::compute_strides(&out_known_dims);
        ctx.tensor_map.insert(
            out_id,
            TensorAtomMap::simple(
                base_id,
                out_count,
                out_dt,
                out_layout,
                out_strides,
                out_sym_dims,
            ),
        );
        super::LowerResult::Lowered
    }
}
