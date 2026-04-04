use crate::graph::GlobalId;
use crate::milli_graph::MilliOpGraphError;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::pool::Pool;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PadMode {
    Constant,
    Reflect,
    Edge,
    Wrap,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pad {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    data: GlobalId,
    pads: GlobalId,
    constant_value: Option<GlobalId>,
    axes: Option<GlobalId>,
    mode: PadMode,
}

impl Pad {
    pub fn push_new(
        graph: &mut crate::milli_graph::MilliOpGraph,
        data: GlobalId,
        pads: GlobalId,
        constant_value: Option<GlobalId>,
        axes: Option<GlobalId>,
        mode: PadMode,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        Self::push_new_with_label(graph, data, pads, constant_value, axes, mode, None, rng)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn push_new_with_label(
        graph: &mut crate::milli_graph::MilliOpGraph,
        data: GlobalId,
        pads: GlobalId,
        constant_value: Option<GlobalId>,
        axes: Option<GlobalId>,
        mode: PadMode,
        label: Option<String>,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            output,
            data,
            pads,
            constant_value,
            axes,
            mode,
            global_id: GlobalId::new(rng),
            label,
        };
        graph.push_op(AnyMilliOp::Pad(node));
        output
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.data, map);
        super::remap(&mut self.pads, map);
        super::remap_opt(&mut self.constant_value, map);
        super::remap_opt(&mut self.axes, map);
    }
}

impl crate::graph::Node for Pad {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Pad".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        let mut res = vec![self.data, self.pads];
        if let Some(cv) = self.constant_value {
            res.push(cv);
        }
        if let Some(ax) = self.axes {
            res.push(ax);
        }
        Box::new(res.into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        Box::new([self.output].into_iter())
    }
}

impl Pad {
    /// Lower Pad (constant mode) to nano ops: Literal + Identity atoms in row-major order.
    ///
    /// All dimensions must be known. Pads tensor must be constant-folded.
    pub fn lower_to_nano(
        &self,
        ctx: &mut crate::nano_graph::NanoLoweringContext,
    ) -> crate::milli_graph::ops::LowerResult {
        use crate::nano_graph::lower::{DimKind, NanoLoweringContext, TensorAtomMap};
        use crate::nano_graph::ops::ScalarOp;
        use crate::nano_graph::pattern::InputRef;
        use crate::numeric_scalar::NumericScalar;

        // Constant mode only.
        if !matches!(self.mode, PadMode::Constant) {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        }

        let all_infos = ctx.all_infos;

        let Some(in_map) = ctx.tensor_map.get(&self.data).cloned() else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };
        let Some(out_info) = all_infos.get(&self.output) else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };

        // No segmented inputs.
        if !in_map.segments.is_empty() {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        }

        // Extract concrete pad values.
        let Some(pads_raw) = NanoLoweringContext::extract_i64(all_infos, &self.pads) else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };

        // All input dims must be known.
        let in_known: Vec<u64> = in_map
            .layout
            .iter()
            .filter_map(|d| match d {
                DimKind::Known(s) => Some(*s),
                _ => None,
            })
            .collect();
        if in_known.len() != in_map.layout.len() {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        }
        let rank = in_known.len();

        // Resolve axes: if specified, only those axes are padded.
        let axes: Vec<usize> = if let Some(axes_id) = self.axes {
            let Some(axes_raw) = NanoLoweringContext::extract_i64(all_infos, &axes_id) else {
                return crate::milli_graph::ops::LowerResult::Unsupported;
            };
            axes_raw
                .iter()
                .map(|&a| {
                    if a < 0 {
                        (a + rank as i64) as usize
                    } else {
                        a as usize
                    }
                })
                .collect()
        } else {
            (0..rank).collect()
        };

        // Build per-dimension pad_begin / pad_end arrays.
        let n_pad_axes = axes.len();
        if pads_raw.len() != 2 * n_pad_axes {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        }
        let mut pad_begin = vec![0usize; rank];
        let mut pad_end = vec![0usize; rank];
        for (i, &ax) in axes.iter().enumerate() {
            pad_begin[ax] = pads_raw[i] as usize;
            pad_end[ax] = pads_raw[n_pad_axes + i] as usize;
        }

        // Compute output shape.
        let out_shape: Vec<u64> = (0..rank)
            .map(|d| in_known[d] + pad_begin[d] as u64 + pad_end[d] as u64)
            .collect();
        let out_total: u64 = out_shape.iter().product();

        // No-op: no actual padding — pass through as view/identity.
        if pad_begin.iter().all(|&p| p == 0) && pad_end.iter().all(|&p| p == 0) {
            return ctx.lower_view_op(self);
        }

        // Atom count cap.
        if out_total > 16_000_000 {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        }

        let in_strides = &in_map.known_strides;
        let sym_dims = in_map.sym_dims.clone();
        let dt = in_map.dtype;

        // Extract constant fill value (default 0 in the input's dtype).
        let fill_scalar = self
            .constant_value
            .and_then(|cv_id| all_infos.get(&cv_id))
            .and_then(|cv_info| cv_info.as_concrete())
            .map(|t| t.read_element(0).cast_to(dt))
            .unwrap_or(NumericScalar::zero(dt));

        // Compute output strides (row-major) for multi-index decomposition.
        let out_strides = TensorAtomMap::compute_strides(&out_shape);

        // Number of "rows" (all dims except last).
        let n_rows = if rank > 0 {
            out_shape[..rank - 1].iter().product::<u64>()
        } else {
            1
        };
        let last = rank - 1;
        let last_out = out_shape[last] as usize;
        let last_in = in_known[last] as usize;
        let last_pb = pad_begin[last];
        let last_pe = pad_end[last];

        // Emit groups row by row, tracking pending zeros for merging.
        let mut first_base = None;
        let mut pending_zeros = 0u64;

        // Helper closure: flush pending zeros as a Literal group.
        // Can't use a closure that borrows ctx mutably, so we'll inline it.

        for row_idx in 0..n_rows {
            // Decompose row_idx into multi-index for dims 0..rank-1.
            let mut coords = vec![0u64; rank];
            let mut rem = row_idx;
            for d in 0..rank - 1 {
                let dim_stride = out_strides[d] / out_shape[last];
                coords[d] = rem / dim_stride;
                rem %= dim_stride;
            }

            // Check if this row is entirely in padding (any outer dim in pad region).
            let mut is_pad_row = false;
            for d in 0..rank - 1 {
                if coords[d] < pad_begin[d] as u64 || coords[d] >= pad_begin[d] as u64 + in_known[d]
                {
                    is_pad_row = true;
                    break;
                }
            }

            if is_pad_row {
                pending_zeros += last_out as u64;
            } else {
                // Interior row: [left_pad, input_data, right_pad].

                // Left pad.
                pending_zeros += last_pb as u64;

                // Flush zeros before Identity group.
                if pending_zeros > 0 {
                    let b = ctx.nano.push_group(
                        pending_zeros,
                        dt,
                        ScalarOp::Literal(fill_scalar.clone()),
                        sym_dims.clone(),
                        vec![],
                    );
                    if first_base.is_none() {
                        first_base = Some(b);
                    }
                    pending_zeros = 0;
                }

                // Identity group for the inner data.
                let mut in_offset = 0u64;
                for d in 0..rank - 1 {
                    in_offset += (coords[d] - pad_begin[d] as u64) * in_strides[d];
                }
                let in_row_base = in_map.base_id.offset(in_offset);
                let in_last_stride = in_strides[last] as i64;

                let b = ctx.nano.push_group(
                    last_in as u64,
                    dt,
                    ScalarOp::Identity,
                    sym_dims.clone(),
                    vec![InputRef::affine(in_row_base, in_last_stride)],
                );
                if first_base.is_none() {
                    first_base = Some(b);
                }

                // Right pad (accumulated, will merge with next row's left or trailing zeros).
                pending_zeros += last_pe as u64;
            }
        }

        // Flush remaining zeros.
        if pending_zeros > 0 {
            ctx.nano.push_group(
                pending_zeros,
                dt,
                ScalarOp::Literal(fill_scalar),
                sym_dims.clone(),
                vec![],
            );
        }

        // Edge case: if entire tensor is padding (no interior rows).
        if first_base.is_none() {
            // Should have been caught by the no-op check, but handle gracefully.
            return crate::milli_graph::ops::LowerResult::Unsupported;
        }

        // Register output.
        let Some((out_layout, out_known_dims, out_sym_dims, _)) = ctx.classify_dims(out_info)
        else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };
        ctx.tensor_map.insert(
            self.output,
            TensorAtomMap::simple(
                first_base.unwrap(),
                out_total,
                dt,
                out_layout,
                TensorAtomMap::compute_strides(&out_known_dims),
                out_sym_dims,
            ),
        );
        crate::milli_graph::ops::LowerResult::Lowered
    }
}

impl MilliOp for Pad {
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
        let shape = data.shape();
        let rank = shape.len();
        let dtype = data.dtype();

        // Parse pads from inputs[1] (1D i64 tensor).
        let pads_view = &inputs[1];
        let pads_raw: Vec<i64> = (0..pads_view.numel())
            .map(|i| pads_view.read_element(i).to_i64())
            .collect();

        // Get constant value (default 0.0) from inputs[2] if present.
        let const_val: f64 = if self.constant_value.is_some() && inputs.len() > 2 {
            inputs[2].read_element(0).to_f64()
        } else {
            0.0
        };

        // Parse axes (optional) — if axes input exists it follows constant_value.
        let axes: Vec<usize> = if self.axes.is_some() {
            let axes_input_idx = if self.constant_value.is_some() { 3 } else { 2 };
            if inputs.len() > axes_input_idx {
                let ax_view = &inputs[axes_input_idx];
                (0..ax_view.numel())
                    .map(|i| {
                        let a = ax_view.read_element(i).to_i64();
                        if a < 0 {
                            (a + rank as i64) as usize
                        } else {
                            a as usize
                        }
                    })
                    .collect()
            } else {
                (0..rank).collect()
            }
        } else {
            (0..rank).collect()
        };

        let num_axes = axes.len();
        if pads_raw.len() != 2 * num_axes {
            return Err(crate::nano_graph::pool_eval::PoolEvalError::Unsupported(
                format!(
                    "Pad: expected pads length {}, got {}",
                    2 * num_axes,
                    pads_raw.len()
                ),
            ));
        }

        // Build per-axis (begin_pad, end_pad).
        let mut begin_pads = vec![0i64; rank];
        let mut end_pads = vec![0i64; rank];
        for (i, &axis) in axes.iter().enumerate() {
            begin_pads[axis] = pads_raw[i];
            end_pads[axis] = pads_raw[num_axes + i];
        }

        // Compute output shape.
        let out_shape: Vec<u64> = (0..rank)
            .map(|i| (shape[i] as i64 + begin_pads[i] + end_pads[i]) as u64)
            .collect();

        let out_numel: usize = out_shape.iter().product::<u64>() as usize;
        let layout = TensorLayout::<DynRank>::row_major(out_shape.clone(), dtype);
        let buf = pool
            .allocate(layout.buffer_size_bytes())
            .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
        let mut out = NumericTensor::from_parts(buf, layout);

        let const_scalar = crate::numeric_scalar::NumericScalar::from_f64(const_val).cast_to(dtype);

        // Compute strides for input and output.
        let in_strides = {
            let mut s = vec![1usize; rank];
            for i in (0..rank.saturating_sub(1)).rev() {
                s[i] = s[i + 1] * shape[i + 1] as usize;
            }
            s
        };
        let out_strides = {
            let mut s = vec![1usize; rank];
            for i in (0..rank.saturating_sub(1)).rev() {
                s[i] = s[i + 1] * out_shape[i + 1] as usize;
            }
            s
        };

        // For each output element, compute the source input coordinate per mode.
        for out_flat in 0..out_numel {
            let mut rem = out_flat;
            let mut in_flat = 0usize;
            let mut is_pad = false;

            for d in 0..rank {
                let out_coord = rem / out_strides[d];
                rem %= out_strides[d];

                // Map output coordinate to input coordinate.
                let in_coord_raw = out_coord as i64 - begin_pads[d];
                let dim = shape[d] as i64;

                let dim_usize = shape[d] as usize;
                let in_coord: usize = match self.mode {
                    PadMode::Constant => {
                        if in_coord_raw < 0 || in_coord_raw >= dim {
                            is_pad = true;
                            0
                        } else {
                            in_coord_raw as usize
                        }
                    }
                    PadMode::Edge => in_coord_raw.clamp(0, dim - 1) as usize,
                    PadMode::Reflect => reflect_index(in_coord_raw, dim_usize),
                    PadMode::Wrap => {
                        let m = in_coord_raw % dim;
                        (if m < 0 { m + dim } else { m }) as usize
                    }
                };

                if is_pad {
                    break;
                }
                in_flat += in_coord * in_strides[d];
            }

            if is_pad {
                out.write_element(out_flat, const_scalar);
            } else {
                out.write_element(out_flat, data.read_element(in_flat));
            }
        }

        Ok(vec![out])
    }

    fn infer<'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'p, P>>,
        symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        _pool: &'p P,
    ) -> Result<Vec<(GlobalId, crate::tensor_info::TensorInfo<'p, P>)>, MilliOpGraphError> {
        use crate::scalar_info::ScalarInfoTyped;
        use crate::symbolic_scalar::SymbolicScalarTyped;
        use crate::tensor_info::TensorInfo;

        let data_info = known_inputs
            .get(&self.data)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let out_dtype = data_info.dtype();

        let ranked = data_info
            .as_ranked()
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let shape = ranked.shape();
        let rank = shape.len();

        // Try to get concrete pads values
        let pads_info = known_inputs
            .get(&self.pads)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let pads_vec = match pads_info.to_i64_vec() {
            Some(v) => v,
            None => {
                // Pads are symbolic — return same rank with symbolic dims
                let out_dims: Vec<ScalarInfoTyped<u64>> = (0..rank)
                    .map(|_| ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)))
                    .collect();
                return Ok(vec![(
                    self.output,
                    TensorInfo::from_dtype_and_shape_scalars(out_dtype, &out_dims),
                )]);
            }
        };

        // Parse axes (optional)
        let axes: Vec<usize> = if let Some(axes_id) = self.axes {
            match known_inputs.get(&axes_id).and_then(|a| a.to_i64_vec()) {
                Some(raw) => raw
                    .iter()
                    .map(|&a| {
                        if a < 0 {
                            (rank as i64 + a) as usize
                        } else {
                            a as usize
                        }
                    })
                    .collect(),
                None => {
                    // Axes are symbolic — return same rank with symbolic dims
                    let out_dims: Vec<ScalarInfoTyped<u64>> = (0..rank)
                        .map(|_| {
                            ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver))
                        })
                        .collect();
                    return Ok(vec![(
                        self.output,
                        TensorInfo::from_dtype_and_shape_scalars(out_dtype, &out_dims),
                    )]);
                }
            }
        } else {
            (0..rank).collect()
        };

        let num_axes = axes.len();
        if pads_vec.len() != 2 * num_axes {
            return Err(MilliOpGraphError::UnableToInfer);
        }

        // Build per-axis (begin_pad, end_pad)
        let mut begin_pads = vec![0i64; rank];
        let mut end_pads = vec![0i64; rank];
        for (i, &axis) in axes.iter().enumerate() {
            begin_pads[axis] = pads_vec[i];
            end_pads[axis] = pads_vec[num_axes + i];
        }

        let mut out_dims = Vec::new();
        for (i, dim) in shape.iter().enumerate() {
            let pad_total = begin_pads[i] + end_pads[i];
            if pad_total == 0 {
                out_dims.push(dim.clone());
            } else {
                match dim {
                    ScalarInfoTyped::Numeric(v) => {
                        out_dims.push(ScalarInfoTyped::Numeric((*v as i64 + pad_total) as u64));
                    }
                    ScalarInfoTyped::Symbolic(_) => {
                        out_dims.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(
                            symbolic_resolver,
                        )));
                    }
                }
            }
        }

        Ok(vec![(
            self.output,
            TensorInfo::from_dtype_and_shape_scalars(out_dtype, &out_dims),
        )])
    }

}

/// Reflect index: maps negative or out-of-bounds indices via reflection.
fn reflect_index(idx: i64, dim: usize) -> usize {
    if dim <= 1 {
        return 0;
    }
    let period = 2 * (dim as i64 - 1);
    let mut i = ((idx % period) + period) % period;
    if i >= dim as i64 {
        i = period - i;
    }
    i as usize
}
