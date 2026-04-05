use crate::graph::GlobalId;
use crate::milli_graph::MilliOpGraphError;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::pool::Pool;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Spatial dilation: insert `strides[i]-1` zeros between adjacent elements
/// along each spatial axis `2+i`. Batch and channel dims pass through unchanged.
///
/// Input shape:  `[N, C, D0, D1, ...]`
/// Output shape: `[N, C, (D0-1)*s0+1, (D1-1)*s1+1, ...]`
///
/// Used by ConvTranspose decomposition: ConvTranspose with stride S is equivalent
/// to Dilate(input, S) → Pad → Conv(stride=1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dilate {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    input: GlobalId,
    /// Per-spatial-axis dilation factors. Length = number of spatial dims.
    strides: Vec<i64>,
}

impl Dilate {
    pub fn push_new(
        graph: &mut crate::milli_graph::MilliOpGraph,
        input: GlobalId,
        strides: Vec<i64>,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        // All strides == 1 → no-op, return input directly.
        if strides.iter().all(|&s| s == 1) {
            return input;
        }
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            label: None,
            output,
            input,
            strides,
        };
        graph.push_op(AnyMilliOp::Dilate(node));
        output
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.input, map);
    }

    pub fn lower_to_nano<'p, P: crate::pool::Pool + 'p>(
        &self,
        ctx: &mut crate::nano_graph::NanoLoweringContext<'_, 'p, P>,
    ) -> crate::milli_graph::ops::LowerResult {
        use crate::nano_graph::lower::{DimKind, TensorAtomMap};
        use crate::nano_graph::ops::ScalarOp;
        use crate::nano_graph::pattern::InputRef;
        use crate::numeric_scalar::NumericScalar;

        let all_infos = ctx.all_infos;

        let Some(in_map) = ctx.tensor_map.get(&self.input).cloned() else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };
        let Some(out_info) = all_infos.get(&self.output) else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };

        if !in_map.segments.is_empty() {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        }

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
        if rank < 2 {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        }
        let n_spatial = rank - 2;
        if self.strides.len() != n_spatial {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        }

        let dt = in_map.dtype;
        let in_strides = &in_map.known_strides;
        let sym_dims = in_map.sym_dims.clone();
        let zero = NumericScalar::zero(dt);

        // Compute output shape.
        let mut out_shape = in_known.clone();
        for i in 0..n_spatial {
            let s = self.strides[i] as u64;
            let d = in_known[2 + i];
            out_shape[2 + i] = if d == 0 { 0 } else { (d - 1) * s + 1 };
        }
        let out_total: u64 = out_shape.iter().product();
        if out_total > 16_000_000 {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        }

        let out_strides_rm = TensorAtomMap::compute_strides(&out_shape);

        let last = rank - 1;
        let s_last = self.strides[n_spatial - 1] as u64;
        let in_last = in_known[last] as usize;
        let out_last = out_shape[last] as usize;

        // Number of rows = product of all dims except last.
        let n_rows: u64 = out_shape[..last].iter().product();

        let mut first_base = None;
        let mut pending_zeros = 0u64;

        for row_idx in 0..n_rows {
            // Decompose row_idx into multi-index for dims 0..last.
            let mut coords = vec![0u64; last];
            let mut rem = row_idx;
            for d in 0..last {
                let dim_stride = out_strides_rm[d] / out_shape[last];
                if dim_stride > 0 {
                    coords[d] = rem / dim_stride;
                    rem %= dim_stride;
                }
            }

            // Check if all spatial coords are stride-aligned.
            let mut aligned = true;
            for i in 0..n_spatial.saturating_sub(1) {
                let s = self.strides[i] as u64;
                if !coords[2 + i].is_multiple_of(s) {
                    aligned = false;
                    break;
                }
            }

            if !aligned {
                // Entire row is zeros.
                pending_zeros += out_last as u64;
            } else {
                // This row has data. Pattern along last axis:
                // [elem0, 0*(s-1), elem1, 0*(s-1), ..., elem_{D-1}]

                // Compute input offset for this row.
                let mut in_offset = 0u64;
                // Batch/channel: same coord.
                for d in 0..2 {
                    in_offset += coords[d] * in_strides[d];
                }
                // Spatial (except last): divide by stride to get input coord.
                for i in 0..n_spatial.saturating_sub(1) {
                    let s = self.strides[i] as u64;
                    in_offset += (coords[2 + i] / s) * in_strides[2 + i];
                }

                if s_last == 1 {
                    // No dilation on last axis — entire row is a contiguous copy.
                    if pending_zeros > 0 {
                        let b = ctx.nano.push_group(
                            pending_zeros,
                            dt,
                            ScalarOp::Literal(zero),
                            sym_dims.clone(),
                            vec![],
                        );
                        if first_base.is_none() {
                            first_base = Some(b);
                        }
                        pending_zeros = 0;
                    }
                    let in_row_base = in_map.base_id.offset(in_offset);
                    let b = ctx.nano.push_group(
                        in_last as u64,
                        dt,
                        ScalarOp::Identity,
                        sym_dims.clone(),
                        vec![InputRef::affine(in_row_base, in_strides[last] as i64)],
                    );
                    if first_base.is_none() {
                        first_base = Some(b);
                    }
                } else {
                    // Interleaved: for each input element, emit 1 copy then (s-1) zeros.
                    for e in 0..in_last {
                        // Flush pending zeros.
                        if pending_zeros > 0 {
                            let b = ctx.nano.push_group(
                                pending_zeros,
                                dt,
                                ScalarOp::Literal(zero),
                                sym_dims.clone(),
                                vec![],
                            );
                            if first_base.is_none() {
                                first_base = Some(b);
                            }
                            pending_zeros = 0;
                        }

                        // Copy one element.
                        let atom_offset = in_offset + (e as u64) * in_strides[last];
                        let b = ctx.nano.push_group(
                            1,
                            dt,
                            ScalarOp::Identity,
                            sym_dims.clone(),
                            vec![InputRef::affine(in_map.base_id.offset(atom_offset), 1)],
                        );
                        if first_base.is_none() {
                            first_base = Some(b);
                        }

                        // Zeros between elements (not after last).
                        if e + 1 < in_last {
                            pending_zeros += s_last - 1;
                        }
                    }
                }
            }
        }

        // Flush trailing zeros.
        if pending_zeros > 0 {
            let b = ctx.nano.push_group(
                pending_zeros,
                dt,
                ScalarOp::Literal(zero),
                sym_dims.clone(),
                vec![],
            );
            if first_base.is_none() {
                first_base = Some(b);
            }
        }

        let Some(base) = first_base else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };

        let Some((out_layout, out_known_dims, out_sym_dims, _)) = ctx.classify_dims(out_info)
        else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };
        ctx.tensor_map.insert(
            self.output,
            TensorAtomMap::simple(
                base,
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

impl crate::graph::Node for Dilate {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Dilate".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl MilliOp for Dilate {
    fn infer<'a, 'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'a, 'p, P>>,
        _symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        pool: &'p P,
    ) -> Result<Vec<(GlobalId, crate::tensor_info::TensorInfo<'a, 'p, P>)>, MilliOpGraphError>
    where
        'p: 'a,
    {
        use crate::scalar_info::ScalarInfoTyped;
        use crate::tensor_info::TensorInfo;

        let input_info = known_inputs
            .get(&self.input)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let out_dtype = input_info.dtype();

        let out_info = if let Some(ranked) = input_info.as_ranked() {
            let in_dims = ranked.shape();
            let rank = in_dims.len();
            let mut out_dims = in_dims.clone();
            // Dilate spatial dims (indices 2..rank).
            for (i, &s) in self.strides.iter().enumerate() {
                let axis = 2 + i;
                if axis < rank
                    && let ScalarInfoTyped::Numeric(d) = &in_dims[axis]
                {
                    out_dims[axis] =
                        ScalarInfoTyped::Numeric(if *d == 0 { 0 } else { (*d - 1) * s as u64 + 1 });
                }
            }
            TensorInfo::from_dtype_and_shape_scalars(out_dtype, &out_dims)
        } else {
            return Err(MilliOpGraphError::UnableToInfer);
        };

        // Try constant fold.
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::milli_graph::MilliOpGraph;
    use crate::numeric_dtype::NumericDType;
    use crate::numeric_scalar::NumericScalar;
    use crate::numeric_tensor::NumericTensor as PoolTensor;
    use crate::pool::SystemPool;

    static POOL: SystemPool = SystemPool;

    fn make_f32(
        shape: Vec<u64>,
        values: &[f32],
    ) -> PoolTensor<'static, crate::tensor_rank::DynRank, SystemPool> {
        let mut t = PoolTensor::zeros(shape, NumericDType::F32, &POOL).unwrap();
        for (i, &v) in values.iter().enumerate() {
            t.write_element(i, NumericScalar::from_f32(v));
        }
        t
    }

    fn read_f32(
        t: &PoolTensor<'_, crate::tensor_rank::DynRank, impl crate::pool::Pool>,
    ) -> Vec<f32> {
        (0..t.numel()).map(|i| t.read_element(i).to_f32()).collect()
    }

    fn pool_eval_one(
        graph: &MilliOpGraph,
        input_id: GlobalId,
        input: PoolTensor<'static, crate::tensor_rank::DynRank, SystemPool>,
        output_ext: GlobalId,
    ) -> PoolTensor<'static, crate::tensor_rank::DynRank, SystemPool> {
        let views: HashMap<GlobalId, _> = [(input_id, input)].into_iter().collect();
        let view_refs: HashMap<GlobalId, _> = views.iter().map(|(&id, t)| (id, t.view())).collect();
        let vr: HashMap<GlobalId, &_> = view_refs.iter().map(|(&id, v)| (id, v)).collect();
        let results = graph.pool_eval(&vr, &POOL).unwrap();
        results
            .into_iter()
            .find(|(id, _)| *id == output_ext)
            .unwrap()
            .1
    }

    #[test]
    fn test_dilate_1d_stride2() {
        // Input: [1, 1, 3] with values [1, 2, 3]
        // Stride: [2]
        // Expected output: [1, 1, 5] with values [1, 0, 2, 0, 3]
        let rng = &mut rand::rng();
        let ext_in = GlobalId::new(rng);
        let (mut graph, input_map) = MilliOpGraph::new(std::iter::once(ext_in), rng);
        let data = input_map[&ext_in];
        let out = Dilate::push_new(&mut graph, data, vec![2], rng);
        let ext_out = GlobalId::new(rng);
        let mut om = HashMap::new();
        om.insert(out, ext_out);
        graph.set_output_map(om);

        let input = make_f32(vec![1, 1, 3], &[1.0, 2.0, 3.0]);
        let result = pool_eval_one(&graph, ext_in, input, ext_out);
        assert_eq!(result.view().shape(), &[1u64, 1, 5]);
        assert_eq!(read_f32(&result), vec![1.0, 0.0, 2.0, 0.0, 3.0]);
    }

    #[test]
    fn test_dilate_2d_stride2x3() {
        // Input: [1, 1, 2, 2] with values [1, 2, 3, 4]
        // Strides: [2, 3]
        // Output: [1, 1, 3, 4] = [1, 1, (2-1)*2+1, (2-1)*3+1]
        // Expected:
        // [[1, 0, 0, 2],
        //  [0, 0, 0, 0],
        //  [3, 0, 0, 4]]
        let rng = &mut rand::rng();
        let ext_in = GlobalId::new(rng);
        let (mut graph, input_map) = MilliOpGraph::new(std::iter::once(ext_in), rng);
        let data = input_map[&ext_in];
        let out = Dilate::push_new(&mut graph, data, vec![2, 3], rng);
        let ext_out = GlobalId::new(rng);
        let mut om = HashMap::new();
        om.insert(out, ext_out);
        graph.set_output_map(om);

        let input = make_f32(vec![1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let result = pool_eval_one(&graph, ext_in, input, ext_out);
        assert_eq!(result.view().shape(), &[1u64, 1, 3, 4]);
        assert_eq!(
            read_f32(&result),
            vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0,]
        );
    }

    #[test]
    fn test_dilate_noop() {
        // Strides all 1 → push_new returns input directly.
        let rng = &mut rand::rng();
        let ext_in = GlobalId::new(rng);
        let (mut graph, input_map) = MilliOpGraph::new(std::iter::once(ext_in), rng);
        let data = input_map[&ext_in];
        let out = Dilate::push_new(&mut graph, data, vec![1, 1], rng);
        assert_eq!(out, data); // No op created.
    }
}
