use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::nano_graph::lower::{DimKind, NanoLoweringContext, TensorAtomMap};
use crate::pool::Pool;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Slice {
    global_id: crate::graph::GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    data: GlobalId,
    starts: GlobalId,
    ends: GlobalId,
    steps: Option<GlobalId>,
    axes: Option<GlobalId>,
}

impl Slice {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        starts: GlobalId,
        ends: GlobalId,
        steps: Option<GlobalId>,
        axes: Option<GlobalId>,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        Self::push_new_with_label(graph, data, starts, ends, steps, axes, None, rng)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        data: GlobalId,
        starts: GlobalId,
        ends: GlobalId,
        steps: Option<GlobalId>,
        axes: Option<GlobalId>,
        label: Option<String>,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            output,
            data,
            starts,
            ends,
            steps,
            axes,
            global_id: GlobalId::new(rng),
            label,
        };
        graph.push_op(AnyMilliOp::Slice(node));
        output
    }
}

impl Slice {
    pub(crate) fn data_id(&self) -> GlobalId {
        self.data
    }
    pub(crate) fn starts_id(&self) -> GlobalId {
        self.starts
    }
    pub(crate) fn ends_id(&self) -> GlobalId {
        self.ends
    }
    pub(crate) fn steps_id(&self) -> Option<GlobalId> {
        self.steps
    }
    pub(crate) fn axes_id(&self) -> Option<GlobalId> {
        self.axes
    }

    pub fn lower_to_nano(
        &self,
        ctx: &mut crate::nano_graph::NanoLoweringContext,
    ) -> crate::milli_graph::ops::LowerResult {
        let all_infos = ctx.all_infos;
        let data_id = self.data_id();
        let out_id = Node::outputs(self).next().unwrap();

        let Some(in_map) = ctx.tensor_map.get(&data_id).cloned() else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };
        // Segmented inputs (from Concat) are not supported — base_id addressing
        // would skip segments. Fall back to opaque eval.
        if !in_map.segments.is_empty() {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };
        let Some(out_info) = all_infos.get(&out_id) else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };

        let Some((out_layout, out_known_dims, out_sym_dims, out_count)) =
            ctx.classify_dims(out_info)
        else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };
        let out_count = out_count.max(1);

        // Extract concrete slice parameters.
        let extract_i64 = |id: &GlobalId| -> Option<Vec<i64>> {
            let info = all_infos.get(id)?;
            info.to_i64_vec()
        };

        let starts = extract_i64(&self.starts_id());
        let ends = extract_i64(&self.ends_id());
        let steps: Option<Vec<i64>> = if let Some(steps_id) = self.steps_id() {
            extract_i64(&steps_id)
        } else {
            starts.as_ref().map(|s| s.iter().map(|_| 1i64).collect())
        };

        let in_rank = in_map.layout.len();
        let axes: Option<Vec<usize>> = if let Some(axes_id) = self.axes_id() {
            extract_i64(&axes_id).map(|a| {
                a.iter()
                    .map(|&v| {
                        if v < 0 {
                            (v + in_rank as i64) as usize
                        } else {
                            v as usize
                        }
                    })
                    .collect()
            })
        } else {
            starts.as_ref().map(|s| (0..s.len()).collect())
        };

        let (Some(starts), Some(_ends), Some(steps), Some(axes)) = (starts, ends, steps, axes)
        else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };

        // Build per-axis (start, step) for known dims.
        // The input's known dims define the coordinate space.
        let in_known: Vec<u64> = in_map
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

        // Map tensor-axis -> known-dim index (None if symbolic).
        let axis_to_known_idx: Vec<Option<usize>> = {
            let mut ki = 0;
            in_map
                .layout
                .iter()
                .map(|d| {
                    if matches!(d, DimKind::Known(_)) {
                        let idx = ki;
                        ki += 1;
                        Some(idx)
                    } else {
                        None
                    }
                })
                .collect()
        };

        // Build the start offset and step for each known dim.
        // Default: start=0, step=1 (full range).
        let mut known_starts = vec![0i64; in_known.len()];
        let mut known_steps = vec![1i64; in_known.len()];

        for (i, &axis) in axes.iter().enumerate() {
            if axis >= in_rank {
                return crate::milli_graph::ops::LowerResult::Unsupported;
            }
            let Some(ki) = axis_to_known_idx[axis] else {
                // Slicing a symbolic dim -- boundary.
                return crate::milli_graph::ops::LowerResult::Unsupported;
            };

            let dim = in_known[ki] as i64;
            let step = steps[i];
            if step == 0 {
                return crate::milli_graph::ops::LowerResult::Unsupported;
            }

            let start = if step > 0 {
                let s = starts[i].clamp(-dim, dim);
                if s < 0 { s + dim } else { s }
            } else {
                let s = starts[i].clamp(-dim, dim - 1);
                if s < 0 { s + dim } else { s }
            };

            known_starts[ki] = start;
            known_steps[ki] = step;
        }

        if out_known_dims.len() != in_known.len() {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        }

        // Zero-cost slice: if the output atoms are contiguous in the input's
        // atom space, we can reuse the input's atoms with an offset base_id.
        //
        // Contiguity requires that the output row-major strides match the
        // input strides scaled by steps. This fails when inner dims are
        // sliced (shrunk) because outer strides then differ.
        let out_rowmajor = TensorAtomMap::compute_strides(&out_known_dims);
        let contiguous = (0..in_known.len()).all(|k| {
            let expected = (in_map.known_strides[k] as i64 * known_steps[k]) as u64;
            out_rowmajor[k] == expected
        });

        let slice_dt = NanoLoweringContext::ndt(out_info);
        if contiguous {
            // All steps must be positive for simple offset-based addressing.
            let all_positive_steps = known_steps.iter().all(|&s| s > 0);
            if all_positive_steps {
                let mut base_offset: u64 = 0;
                for (&start, &stride) in known_starts.iter().zip(in_map.known_strides.iter()) {
                    base_offset += start as u64 * stride;
                }
                ctx.tensor_map.insert(
                    out_id,
                    TensorAtomMap::simple(
                        in_map.base_id.offset(base_offset),
                        out_count,
                        slice_dt,
                        out_layout,
                        TensorAtomMap::compute_strides(&out_known_dims),
                        out_sym_dims,
                    ),
                );
                return crate::milli_graph::ops::LowerResult::Lowered;
            }
        }

        // Non-contiguous slice: zero-cost view with strides that account for steps.
        // Output stride[k] = input_stride[k] * step[k] (for positive steps).
        let all_positive_steps = known_steps.iter().all(|&s| s > 0);
        if all_positive_steps {
            let mut base_offset: u64 = 0;
            let mut out_phys_strides = Vec::with_capacity(in_known.len());
            for ki in 0..in_known.len() {
                base_offset += known_starts[ki] as u64 * in_map.known_strides[ki];
                out_phys_strides.push((in_map.known_strides[ki] as i64 * known_steps[ki]) as u64);
            }
            ctx.tensor_map.insert(
                out_id,
                TensorAtomMap::simple(
                    in_map.base_id.offset(base_offset),
                    out_count,
                    slice_dt,
                    out_layout,
                    out_phys_strides,
                    out_sym_dims,
                ),
            );
            return crate::milli_graph::ops::LowerResult::Lowered;
        }

        // Negative steps: fall back to boundary (rare).
        crate::milli_graph::ops::LowerResult::Unsupported
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.data, map);
        super::remap(&mut self.starts, map);
        super::remap(&mut self.ends, map);
        super::remap_opt(&mut self.steps, map);
        super::remap_opt(&mut self.axes, map);
    }
}

impl crate::graph::Node for Slice {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Slice".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        let mut res = vec![self.data, self.starts, self.ends];
        if let Some(steps) = &self.steps {
            res.push(*steps);
        }
        if let Some(axes) = &self.axes {
            res.push(*axes);
        }
        Box::new(res.into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new([self.output].into_iter())
    }
}

impl MilliOp for Slice {
    fn infer<'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'p, P>>,
        _symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        pool: &'p P,
    ) -> Result<Vec<(GlobalId, crate::tensor_info::TensorInfo<'p, P>)>, MilliOpGraphError> {
        use crate::scalar_info::ScalarInfoTyped;
        use crate::tensor_info::TensorInfo;

        let data_info = known_inputs
            .get(&self.data)
            .ok_or(MilliOpGraphError::UnableToInfer)?;

        // Shape-only inference: compute output shape from data shape + slice params.
        let data_ranked = data_info
            .as_ranked()
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let data_shape = data_ranked.shape();
        let data_rank = data_shape.len();

        // Try to extract concrete i64 values from a tensor.
        let extract_i64 = |id: &GlobalId| -> Option<Vec<i64>> {
            let info = known_inputs.get(id)?;
            info.to_i64_vec()
        };

        let starts = extract_i64(&self.starts);
        let ends = extract_i64(&self.ends);
        let steps: Option<Vec<i64>> = if let Some(steps_id) = &self.steps {
            extract_i64(steps_id)
        } else {
            starts.as_ref().map(|s| s.iter().map(|_| 1i64).collect())
        };
        let axes: Option<Vec<usize>> = if let Some(axes_id) = &self.axes {
            extract_i64(axes_id).map(|a| {
                a.iter()
                    .map(|&v| {
                        if v < 0 {
                            (v + data_rank as i64) as usize
                        } else {
                            v as usize
                        }
                    })
                    .collect()
            })
        } else {
            starts.as_ref().map(|s| (0..s.len()).collect())
        };

        let mut out_dims = data_shape.clone();

        // If we have concrete slice params, compute exact output dims.
        // Otherwise, make sliced axes symbolic (we know the rank but not the dim sizes).
        if let (Some(starts), Some(ends), Some(steps), Some(axes)) = (&starts, &ends, &steps, &axes)
        {
            for (i, &axis) in axes.iter().enumerate() {
                if let ScalarInfoTyped::Numeric(dim_val) = &data_shape[axis] {
                    let dim = *dim_val as i64;
                    let step = steps[i];
                    let (start, end) = if step > 0 {
                        let s = starts[i].clamp(-dim, dim);
                        let s = if s < 0 { s + dim } else { s };
                        let e = ends[i].clamp(-dim, dim);
                        let e = if e < 0 { e + dim } else { e };
                        (s, e)
                    } else {
                        let s = starts[i].clamp(-dim, dim - 1);
                        let s = if s < 0 { s + dim } else { s };
                        let e = ends[i].clamp(-dim - 1, dim);
                        let e = if e < 0 { e + dim } else { e };
                        (s, e)
                    };
                    let sliced = ((end - start + (step - step.signum())) / step).max(0) as u64;
                    out_dims[axis] = ScalarInfoTyped::Numeric(sliced);
                }
                // If dim is symbolic, leave it symbolic.
            }
        } else if let Some(axes) = &axes {
            // We know which axes are sliced but not the exact values —
            // make those dims symbolic.
            for &axis in axes {
                out_dims[axis] = ScalarInfoTyped::Symbolic(
                    crate::symbolic_scalar::SymbolicScalarTyped::new(_symbolic_resolver),
                );
            }
        } else {
            // We don't know the axes — any dim could be sliced.
            // Make all dims symbolic to avoid claiming incorrect concrete sizes.
            for dim in out_dims.iter_mut() {
                *dim = ScalarInfoTyped::Symbolic(crate::symbolic_scalar::SymbolicScalarTyped::new(
                    _symbolic_resolver,
                ));
            }
        }

        let out_dtype = data_info.dtype();
        let out_info = TensorInfo::from_dtype_and_shape_scalars(out_dtype, &out_dims);

        // If all inputs are concrete, try constant fold via nano+pool_eval path.
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
        let input_shape = data.shape();
        let input_rank = input_shape.len();
        let dtype = data.dtype();

        // Parse starts (inputs[1]) and ends (inputs[2])
        let starts: Vec<i64> = (0..inputs[1].numel())
            .map(|i| inputs[1].read_element(i).to_i64())
            .collect();
        let ends: Vec<i64> = (0..inputs[2].numel())
            .map(|i| inputs[2].read_element(i).to_i64())
            .collect();

        // Parse steps and axes from optional inputs
        let mut input_idx = 3;
        let steps: Vec<i64> = if self.steps.is_some() && inputs.len() > input_idx {
            let s: Vec<i64> = (0..inputs[input_idx].numel())
                .map(|i| inputs[input_idx].read_element(i).to_i64())
                .collect();
            input_idx += 1;
            s
        } else {
            starts.iter().map(|_| 1i64).collect()
        };
        let axes: Vec<usize> = if self.axes.is_some() && inputs.len() > input_idx {
            (0..inputs[input_idx].numel())
                .map(|i| {
                    let a = inputs[input_idx].read_element(i).to_i64();
                    if a < 0 {
                        (a + input_rank as i64) as usize
                    } else {
                        a as usize
                    }
                })
                .collect()
        } else {
            (0..starts.len()).collect()
        };

        // Build per-axis (start, end, step)
        let mut slices: Vec<(i64, i64, i64)> =
            input_shape.iter().map(|&d| (0, d as i64, 1)).collect();
        for (i, &axis) in axes.iter().enumerate() {
            let dim = input_shape[axis] as i64;
            let step = steps[i];
            if step == 0 {
                return Err(crate::nano_graph::pool_eval::PoolEvalError::Unsupported(
                    "Slice: step must not be 0".into(),
                ));
            }
            let (start, end) = if step > 0 {
                let s = starts[i].clamp(-dim, dim);
                let s = if s < 0 { s + dim } else { s };
                let e = ends[i].clamp(-dim, dim);
                let e = if e < 0 { e + dim } else { e };
                (s, e)
            } else {
                let s = starts[i].clamp(-dim, dim - 1);
                let s = if s < 0 { s + dim } else { s };
                let e = ends[i].clamp(-dim - 1, dim);
                let e = if e < 0 { e + dim } else { e };
                (s, e)
            };
            slices[axis] = (start, end, step);
        }

        // Compute output shape
        let output_shape: Vec<u64> = slices
            .iter()
            .map(|&(s, e, step)| ((e - s + (step - step.signum())) / step).max(0) as u64)
            .collect();

        let out_layout = TensorLayout::<DynRank>::row_major(output_shape.clone(), dtype);
        let in_layout = data.layout().clone();

        let out = NumericTensor::<DynRank, P2>::from_fn(output_shape, dtype, pool, |out_flat| {
            let out_coords = out_layout.flat_to_coords(out_flat);
            let in_coords: Vec<usize> = out_coords
                .iter()
                .enumerate()
                .map(|(d, &coord)| {
                    let (start, _, step) = slices[d];
                    (start + coord as i64 * step) as usize
                })
                .collect();
            data.read_element(in_layout.coords_to_flat(&in_coords))
        })
        .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;

        Ok(vec![out])
    }

    fn lower_to_nano(
        &self,
        ctx: &mut crate::nano_graph::NanoLoweringContext,
    ) -> crate::milli_graph::ops::LowerResult {
        Slice::lower_to_nano(self, ctx)
    }
}
