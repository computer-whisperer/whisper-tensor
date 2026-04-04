use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::milli_graph::{MilliOpGraph, MilliOpGraphError};
use crate::pool::Pool;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ConvAutoPad {
    NotSet,
    SameUpper,
    SameLower,
    Valid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conv {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    input: GlobalId,
    weight: GlobalId,
    bias: Option<GlobalId>,
    auto_pad: ConvAutoPad,
    dilations: Vec<i64>,
    group: i64,
    kernel_shape: Vec<i64>,
    pads: Vec<i64>,
    strides: Vec<i64>,
}

impl Conv {
    #[allow(clippy::too_many_arguments)]
    pub fn push_new(
        graph: &mut MilliOpGraph,
        input: GlobalId,
        weight: GlobalId,
        bias: Option<GlobalId>,
        auto_pad: ConvAutoPad,
        dilations: Vec<i64>,
        group: i64,
        kernel_shape: Vec<i64>,
        pads: Vec<i64>,
        strides: Vec<i64>,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        Self::push_new_with_label(
            graph,
            input,
            weight,
            bias,
            auto_pad,
            dilations,
            group,
            kernel_shape,
            pads,
            strides,
            None,
            rng,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        input: GlobalId,
        weight: GlobalId,
        bias: Option<GlobalId>,
        auto_pad: ConvAutoPad,
        dilations: Vec<i64>,
        group: i64,
        kernel_shape: Vec<i64>,
        pads: Vec<i64>,
        strides: Vec<i64>,
        label: Option<String>,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            label,
            output,
            input,
            weight,
            bias,
            auto_pad,
            dilations,
            group,
            kernel_shape,
            pads,
            strides,
        };
        graph.push_op(AnyMilliOp::Conv(node));
        output
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.input, map);
        super::remap(&mut self.weight, map);
        super::remap_opt(&mut self.bias, map);
    }
}

impl crate::graph::Node for Conv {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Conv".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        let mut res = vec![self.input, self.weight];
        if let Some(bias) = self.bias {
            res.push(bias);
        }
        Box::new(res.into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

/// Row-major strides for a shape (usize version).
fn compute_row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0usize; shape.len()];
    if shape.is_empty() {
        return strides;
    }
    let mut stride = 1usize;
    for i in (0..shape.len()).rev() {
        strides[i] = stride;
        stride *= shape[i];
    }
    strides
}

/// Row-major strides for a shape (u64 version).
fn compute_row_major_strides_u64(shape: &[usize]) -> Vec<u64> {
    let mut strides = vec![0u64; shape.len()];
    if shape.is_empty() {
        return strides;
    }
    let mut stride = 1u64;
    for i in (0..shape.len()).rev() {
        strides[i] = stride;
        stride *= shape[i] as u64;
    }
    strides
}

/// Resolve padding from auto_pad mode and conv parameters.
fn resolve_padding(
    auto_pad: ConvAutoPad,
    pads: &[i64],
    n_spatial: usize,
    input_spatial: &[usize],
    strides: &[usize],
    dilated_kernel: &[usize],
) -> (Vec<usize>, Vec<usize>) {
    match auto_pad {
        ConvAutoPad::NotSet => {
            if pads.is_empty() {
                (vec![0usize; n_spatial], vec![0usize; n_spatial])
            } else {
                let pb: Vec<usize> = (0..n_spatial).map(|i| pads[i] as usize).collect();
                let pe: Vec<usize> = (0..n_spatial)
                    .map(|i| pads[n_spatial + i] as usize)
                    .collect();
                (pb, pe)
            }
        }
        ConvAutoPad::Valid => (vec![0; n_spatial], vec![0; n_spatial]),
        ConvAutoPad::SameUpper | ConvAutoPad::SameLower => {
            let mut pb = vec![0usize; n_spatial];
            let mut pe = vec![0usize; n_spatial];
            for i in 0..n_spatial {
                let out_size = input_spatial[i].div_ceil(strides[i]);
                let total_pad = ((out_size - 1) * strides[i] + dilated_kernel[i])
                    .saturating_sub(input_spatial[i]);
                if matches!(auto_pad, ConvAutoPad::SameUpper) {
                    pb[i] = total_pad / 2;
                    pe[i] = total_pad - pb[i];
                } else {
                    pe[i] = total_pad / 2;
                    pb[i] = total_pad - pe[i];
                }
            }
            (pb, pe)
        }
    }
}

/// Parameters for 2D im2col operation.
struct Im2Col2dParams {
    in_base: usize,
    channels_per_group_in: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    pad_top: usize,
    pad_left: usize,
}

/// im2col for 2D convolution: rearrange input patches into a column matrix.
/// For a single (batch, group), produces shape [cpg_in * kH * kW, OH * OW].
fn im2col_2d(input_data: &[f32], col: &mut [f32], p: &Im2Col2dParams) {
    let out_spatial = p.out_h * p.out_w;
    let in_channel_stride = p.in_h * p.in_w;

    for ci in 0..p.channels_per_group_in {
        let in_c_base = p.in_base + ci * in_channel_stride;
        for kh in 0..p.kernel_h {
            for kw in 0..p.kernel_w {
                let col_row = (ci * p.kernel_h + kh) * p.kernel_w + kw;
                let col_offset = col_row * out_spatial;
                for oh in 0..p.out_h {
                    let ih = (oh * p.stride_h + kh * p.dilation_h) as isize - p.pad_top as isize;
                    if ih < 0 || ih >= p.in_h as isize {
                        for ow in 0..p.out_w {
                            col[col_offset + oh * p.out_w + ow] = 0.0;
                        }
                        continue;
                    }
                    let in_row_base = in_c_base + ih as usize * p.in_w;
                    for ow in 0..p.out_w {
                        let iw =
                            (ow * p.stride_w + kw * p.dilation_w) as isize - p.pad_left as isize;
                        col[col_offset + oh * p.out_w + ow] = if iw >= 0 && iw < p.in_w as isize {
                            input_data[in_row_base + iw as usize]
                        } else {
                            0.0
                        };
                    }
                }
            }
        }
    }
}

/// Parameters for generic n-dimensional convolution.
struct ConvNdParams<'a> {
    input_data: &'a [f32],
    weight_data: &'a [f32],
    bias_data: Option<&'a [f32]>,
    batch_size: usize,
    in_channels: usize,
    out_channels: usize,
    group: usize,
    channels_per_group_in: usize,
    channels_per_group_out: usize,
    n_spatial: usize,
    input_spatial: &'a [usize],
    out_spatial: &'a [usize],
    kernel_shape: &'a [usize],
    strides: &'a [usize],
    dilations: &'a [usize],
    pad_begin: &'a [usize],
}

/// Generic n-dimensional convolution fallback (1D, 3D+).
fn conv_nd_generic(p: &ConvNdParams) -> Vec<f32> {
    let out_spatial_size: usize = p.out_spatial.iter().product();
    let total_out = p.batch_size * p.out_channels * out_spatial_size;
    let in_channel_stride: usize = p.input_spatial.iter().product();
    let in_batch_stride = p.in_channels * in_channel_stride;
    let w_kernel_size: usize = p.kernel_shape.iter().product();
    let w_cin_stride = w_kernel_size;
    let w_cout_stride = p.channels_per_group_in * w_cin_stride;

    let mut in_spatial_strides = vec![1usize; p.n_spatial];
    for i in (0..p.n_spatial.saturating_sub(1)).rev() {
        in_spatial_strides[i] = in_spatial_strides[i + 1] * p.input_spatial[i + 1];
    }
    let mut out_spatial_strides = vec![1usize; p.n_spatial];
    for i in (0..p.n_spatial.saturating_sub(1)).rev() {
        out_spatial_strides[i] = out_spatial_strides[i + 1] * p.out_spatial[i + 1];
    }
    let mut kernel_strides = vec![1usize; p.n_spatial];
    for i in (0..p.n_spatial.saturating_sub(1)).rev() {
        kernel_strides[i] = kernel_strides[i + 1] * p.kernel_shape[i + 1];
    }

    let mut output_data = vec![0.0f32; total_out];

    // Collect work items for parallelism
    let work_items: Vec<(usize, usize, usize)> = (0..p.batch_size)
        .flat_map(|n| {
            (0..p.group).flat_map(move |g| (0..p.channels_per_group_out).map(move |co| (n, g, co)))
        })
        .collect();

    let chunk_results: Vec<(usize, Vec<f32>)> = work_items
        .par_iter()
        .map(|&(n, g, co)| {
            let m = g * p.channels_per_group_out + co;
            let bias_val = p.bias_data.map_or(0.0, |b| b[m]);
            let mut out_buf = vec![0.0f32; out_spatial_size];

            let mut out_coords = vec![0usize; p.n_spatial];
            let mut k_coords = vec![0usize; p.n_spatial];

            for ci in 0..p.channels_per_group_in {
                let in_c = g * p.channels_per_group_in + ci;

                for (out_idx, out_val) in out_buf.iter_mut().enumerate() {
                    // Decompose out_idx into spatial coordinates
                    let mut remaining = out_idx;
                    for d in 0..p.n_spatial {
                        out_coords[d] = remaining / out_spatial_strides[d];
                        remaining %= out_spatial_strides[d];
                    }

                    let mut sum = 0.0f32;

                    for k_idx in 0..w_kernel_size {
                        let mut k_remaining = k_idx;
                        for d in 0..p.n_spatial {
                            k_coords[d] = k_remaining / kernel_strides[d];
                            k_remaining %= kernel_strides[d];
                        }

                        let mut in_bounds = true;
                        let mut in_spatial_offset = 0usize;
                        for d in 0..p.n_spatial {
                            let pos = (out_coords[d] * p.strides[d] + k_coords[d] * p.dilations[d])
                                as isize
                                - p.pad_begin[d] as isize;
                            if pos < 0 || pos >= p.input_spatial[d] as isize {
                                in_bounds = false;
                                break;
                            }
                            in_spatial_offset += pos as usize * in_spatial_strides[d];
                        }

                        if in_bounds {
                            let in_offset =
                                n * in_batch_stride + in_c * in_channel_stride + in_spatial_offset;
                            let w_offset = m * w_cout_stride + ci * w_cin_stride + k_idx;
                            sum += p.input_data[in_offset] * p.weight_data[w_offset];
                        }
                    }
                    *out_val += sum;
                }
            }

            // Add bias
            if bias_val != 0.0 {
                for v in &mut out_buf {
                    *v += bias_val;
                }
            }

            let out_offset = n * (p.out_channels * out_spatial_size) + m * out_spatial_size;
            (out_offset, out_buf)
        })
        .collect();

    for (offset, buf) in chunk_results {
        output_data[offset..offset + buf.len()].copy_from_slice(&buf);
    }

    output_data
}

/// Helper: build Conv output with known batch + out_channels but symbolic spatial dims.
fn make_symbolic_output<'p, P: Pool + 'p>(
    output_id: GlobalId,
    out_dtype: crate::numeric_dtype::NumericDType,
    batch: &crate::scalar_info::ScalarInfoTyped<u64>,
    out_channels: &crate::scalar_info::ScalarInfoTyped<u64>,
    n_spatial: usize,
    symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
) -> Result<
    Vec<(GlobalId, crate::tensor_info::TensorInfo<'p, P>)>,
    crate::milli_graph::MilliOpGraphError,
> {
    use crate::scalar_info::ScalarInfoTyped;
    use crate::symbolic_scalar::SymbolicScalarTyped;
    use crate::tensor_info::TensorInfo;

    let mut out_dims = Vec::with_capacity(2 + n_spatial);
    out_dims.push(batch.clone());
    out_dims.push(out_channels.clone());
    for _ in 0..n_spatial {
        out_dims.push(ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(
            symbolic_resolver,
        )));
    }
    Ok(vec![(
        output_id,
        TensorInfo::from_dtype_and_shape_scalars(out_dtype, &out_dims),
    )])
}

impl Conv {
    /// Lower Conv to nano ops: padded input atoms + Mul/ReduceSum per output channel.
    ///
    /// Scope: 2D, group=1, dilation=[1,1], all spatial/channel dims known, batch symbolic.
    pub fn lower_to_nano(
        &self,
        ctx: &mut crate::nano_graph::NanoLoweringContext,
    ) -> crate::milli_graph::ops::LowerResult {
        use crate::nano_graph::lower::DimKind;
        use crate::nano_graph::lower::TensorAtomMap;
        use crate::nano_graph::ops::{ReduceKind, ScalarBinOp, ScalarOp};
        use crate::nano_graph::pattern::InputRef;
        use crate::numeric_scalar::NumericScalar;

        let all_infos = ctx.all_infos;
        let in_id = self.input;
        let w_id = self.weight;
        let out_id = self.output;

        let (Some(in_map), Some(w_map)) = (
            ctx.tensor_map.get(&in_id).cloned(),
            ctx.tensor_map.get(&w_id).cloned(),
        ) else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };

        let bias_map = self.bias.and_then(|id| ctx.tensor_map.get(&id).cloned());
        if self.bias.is_some() && bias_map.is_none() {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        }

        let Some(out_info) = all_infos.get(&out_id) else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };

        let in_layout = &in_map.layout;
        let w_layout = &w_map.layout;

        // Need at least [N, C, spatial...] with at least 1 spatial dim.
        let n_spatial = in_layout.len().saturating_sub(2);
        if in_layout.len() < 3 || n_spatial == 0 {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        }

        // No segmented inputs.
        if !in_map.segments.is_empty() || !w_map.segments.is_empty() {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        }
        if let Some(ref bm) = bias_map {
            if !bm.segments.is_empty() {
                return crate::milli_graph::ops::LowerResult::Unsupported;
            }
        }

        let groups = self.group as u64;

        // Extract known channel + spatial dims from input (tail of known dims).
        let in_known: Vec<u64> = in_layout
            .iter()
            .filter_map(|d| match d {
                DimKind::Known(s) => Some(*s),
                _ => None,
            })
            .collect();
        // Need C_in + all spatial dims known.
        if in_known.len() < 1 + n_spatial {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        }
        let c_in = in_known[in_known.len() - 1 - n_spatial];
        let in_spatial: Vec<usize> = in_known[in_known.len() - n_spatial..]
            .iter()
            .map(|&d| d as usize)
            .collect();

        // Weight must be fully known [C_out, C_in/group, K0, K1, ...].
        let w_known: Vec<u64> = w_layout
            .iter()
            .filter_map(|d| match d {
                DimKind::Known(s) => Some(*s),
                _ => None,
            })
            .collect();
        if w_known.len() != w_layout.len() || w_known.len() != 2 + n_spatial {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        }
        let c_out = w_known[0];
        let c_in_per_group = w_known[1];

        let kernel: Vec<usize> = (0..n_spatial)
            .map(|i| {
                if i < self.kernel_shape.len() {
                    self.kernel_shape[i] as usize
                } else {
                    w_known[2 + i] as usize
                }
            })
            .collect();

        let dilations: Vec<usize> = if self.dilations.is_empty() {
            vec![1; n_spatial]
        } else {
            self.dilations.iter().map(|&x| x as usize).collect()
        };

        let strides: Vec<usize> = if self.strides.is_empty() {
            vec![1; n_spatial]
        } else {
            self.strides.iter().map(|&x| x as usize).collect()
        };

        // Effective (dilated) kernel sizes.
        let dilated_kernel: Vec<usize> = (0..n_spatial)
            .map(|i| (kernel[i] - 1) * dilations[i] + 1)
            .collect();

        // Resolve padding.
        let (pad_begin, pad_end) = resolve_padding(
            self.auto_pad,
            &self.pads,
            n_spatial,
            &in_spatial,
            &strides,
            &dilated_kernel,
        );

        // Padded and output spatial dims.
        let padded_spatial: Vec<usize> = (0..n_spatial)
            .map(|i| in_spatial[i] + pad_begin[i] + pad_end[i])
            .collect();
        let out_spatial: Vec<usize> = (0..n_spatial)
            .map(|i| (padded_spatial[i] - dilated_kernel[i]) / strides[i] + 1)
            .collect();
        let spatial_size: u64 = out_spatial.iter().map(|&d| d as u64).product();
        // s = total output elements per output channel (batch × spatial).
        // When batch is symbolic, batch_known=1 and sym_dims handle replication.

        // Kernel element count and contraction dim.
        let kernel_total: u64 = kernel.iter().map(|&k| k as u64).product();
        let k = c_in_per_group * kernel_total; // contraction dim per output channel

        // Atom count cap.
        let batch_known: u64 = in_layout
            .iter()
            .take(in_layout.len() - 1 - n_spatial)
            .filter_map(|d| match d {
                DimKind::Known(s) => Some(*s),
                _ => None,
            })
            .product::<u64>()
            .max(1);
        let s = batch_known * spatial_size;
        let total_mul_atoms = c_out * k * s;
        if total_mul_atoms > 16_000_000 {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        }

        // Classify output dims.
        let Some((out_layout, out_known_dims, out_sym_dims, _)) = ctx.classify_dims(out_info)
        else {
            return crate::milli_graph::ops::LowerResult::Unsupported;
        };

        let sym_dims = in_map.sym_dims.clone();
        let in_strides = &in_map.known_strides;
        let w_strides = &w_map.known_strides;
        let original_dtype = in_map.dtype;

        // --- Phase 1: Emit padded input atoms ---
        // Row-by-row emission: iterate over [C_in, spatial_0, ..., spatial_{nd-2}]
        // and for each row, emit [pad_left, identity_data, pad_right] along the
        // innermost spatial axis.
        let has_padding = pad_begin.iter().any(|&p| p > 0) || pad_end.iter().any(|&p| p > 0);
        let last = n_spatial - 1;
        let in_last = in_spatial[last];
        let padded_last = padded_spatial[last];

        let padded_base;

        if has_padding {
            let mut first_base = None;
            let mut pending_zeros = 0u64;

            // Input strides: [...batch, C_in, spatial_dims...].
            let in_c_stride = in_strides[in_strides.len() - 1 - n_spatial];
            let in_spatial_strides: Vec<u64> = (0..n_spatial)
                .map(|i| in_strides[in_strides.len() - n_spatial + i])
                .collect();
            // Batch stride: product of all dims after batch.
            let in_batch_stride: u64 = if in_strides.len() > 1 + n_spatial {
                in_strides[in_strides.len() - 2 - n_spatial]
            } else {
                0 // no batch dim (or batch=1 symbolic)
            };

            let n_outer: usize = padded_spatial[..last].iter().product::<usize>().max(1);
            let outer_strides = compute_row_major_strides(&padded_spatial[..last]);

            for bi in 0..batch_known {
                let batch_offset = bi * in_batch_stride;
                for ci in 0..c_in {
                    let c_offset = batch_offset + ci * in_c_stride;

                    for row_idx in 0..n_outer {
                        // Decompose row_idx into padded spatial coords for dims 0..last.
                        let mut coords = vec![0usize; last];
                        let mut rem = row_idx;
                        for d in 0..last {
                            if outer_strides[d] > 0 {
                                coords[d] = rem / outer_strides[d];
                                rem %= outer_strides[d];
                            }
                        }

                        // Check if any outer coord is in the pad region.
                        let in_pad = (0..last).any(|d| {
                            coords[d] < pad_begin[d] || coords[d] >= pad_begin[d] + in_spatial[d]
                        });

                        if in_pad {
                            pending_zeros += padded_last as u64;
                        } else {
                            // Left pad.
                            pending_zeros += pad_begin[last] as u64;

                            // Flush zeros.
                            if pending_zeros > 0 {
                                let b = ctx.nano.push_group(
                                    pending_zeros,
                                    original_dtype,
                                    ScalarOp::Literal(NumericScalar::zero(original_dtype)),
                                    sym_dims.clone(),
                                    vec![],
                                );
                                if first_base.is_none() {
                                    first_base = Some(b);
                                }
                                pending_zeros = 0;
                            }

                            // Identity: copy input row along the innermost spatial dim.
                            let mut in_offset = c_offset;
                            for d in 0..last {
                                in_offset +=
                                    (coords[d] - pad_begin[d]) as u64 * in_spatial_strides[d];
                            }
                            let b = ctx.nano.push_group(
                                in_last as u64,
                                original_dtype,
                                ScalarOp::Identity,
                                sym_dims.clone(),
                                vec![InputRef::affine(
                                    in_map.base_id.offset(in_offset),
                                    in_spatial_strides[last] as i64,
                                )],
                            );
                            if first_base.is_none() {
                                first_base = Some(b);
                            }

                            // Right pad.
                            pending_zeros += pad_end[last] as u64;
                        }
                    }
                } // for ci
            } // for bi

            if pending_zeros > 0 {
                let b = ctx.nano.push_group(
                    pending_zeros,
                    original_dtype,
                    ScalarOp::Literal(NumericScalar::zero(original_dtype)),
                    sym_dims.clone(),
                    vec![],
                );
                if first_base.is_none() {
                    first_base = Some(b);
                }
            }

            padded_base = first_base.unwrap();
        } else {
            padded_base = in_map.base_id;
        }

        // Reference strides for addressing into the padded/input atom space.
        // Padded atoms are row-major; raw input may have non-row-major strides.
        let ref_batch_stride: u64;
        let ref_c_stride: u64;
        let ref_spatial_strides: Vec<u64>;
        if has_padding {
            let padded_rm = compute_row_major_strides_u64(&padded_spatial);
            let padded_per_channel: u64 = padded_spatial.iter().map(|&d| d as u64).product();
            ref_c_stride = padded_per_channel;
            ref_batch_stride = c_in * padded_per_channel;
            ref_spatial_strides = padded_rm;
        } else {
            ref_c_stride = in_strides[in_strides.len() - 1 - n_spatial];
            ref_batch_stride = if in_strides.len() > 1 + n_spatial {
                in_strides[in_strides.len() - 2 - n_spatial]
            } else {
                0
            };
            ref_spatial_strides = (0..n_spatial)
                .map(|i| in_strides[in_strides.len() - n_spatial + i])
                .collect();
        }

        // --- Phase 2a: Emit mul groups ---
        // For each (output_channel, input_channel_in_group, kernel_position):
        //   mul[flat] = weight[co, ci_local, k...] * input[batch, ci, output_coords*stride + k*dilation]
        //
        // flat indexes over [batch, spatial_dims...] = s = batch_known * spatial_size.

        let kernel_strides = compute_row_major_strides(&kernel);

        // Emit atoms in [B, C_out, spatial] order (row-major output layout).
        // The spatial-only Strided InputRef addresses within one batch element.
        let spatial_dim_shape: Vec<u64> = {
            let mut shape = vec![u64::MAX];
            for &os in &out_spatial[1..] {
                shape.push(os as u64);
            }
            shape
        };
        let spatial_dim_strides: Vec<i64> = (0..n_spatial)
            .map(|d| (strides[d] as u64 * ref_spatial_strides[d]) as i64)
            .collect();

        let mut first_output_base = None;

        // Phase 2a: Emit ALL mul groups for all batches × channels × kernel positions.
        // Must come before reduce groups so the reduce atoms are contiguous in
        // [B, C_out, spatial] order (matching the output row-major layout).
        //
        // all_mul_bases[bi][co] = first mul group atom ID for that (batch, output_channel).
        let mut all_mul_bases: Vec<Vec<crate::nano_graph::pattern::AtomId>> =
            Vec::with_capacity(batch_known as usize);

        for bi in 0..batch_known {
            let batch_padded_offset = bi * ref_batch_stride;
            let mut mul_bases: Vec<_> = Vec::with_capacity(c_out as usize);

            for co in 0..c_out {
                let mut first_mul_base = None;
                let g = co / (c_out / groups);

                for ci_local in 0..c_in_per_group {
                    let ci = g * c_in_per_group + ci_local;

                    for k_flat in 0..kernel_total {
                        let mut k_coords = vec![0usize; n_spatial];
                        let mut k_rem = k_flat as usize;
                        for d in 0..n_spatial {
                            if kernel_strides[d] > 0 {
                                k_coords[d] = k_rem / kernel_strides[d];
                                k_rem %= kernel_strides[d];
                            }
                        }

                        let mut w_offset = co * w_strides[0] + ci_local * w_strides[1];
                        for d in 0..n_spatial {
                            w_offset += k_coords[d] as u64 * w_strides[2 + d];
                        }
                        let w_atom = w_map.base_id.offset(w_offset);

                        let mut in_base_offset = batch_padded_offset + ci * ref_c_stride;
                        for d in 0..n_spatial {
                            in_base_offset +=
                                k_coords[d] as u64 * dilations[d] as u64 * ref_spatial_strides[d];
                        }
                        let in_base = padded_base.offset(in_base_offset);

                        let input_ref = InputRef::Strided {
                            base: in_base,
                            dim_strides: spatial_dim_strides.clone(),
                            dim_shape: spatial_dim_shape.clone(),
                        };

                        let b = ctx.nano.push_group(
                            spatial_size,
                            original_dtype,
                            ScalarOp::Binary {
                                op: ScalarBinOp::Mul,
                                compute_dtype: original_dtype,
                            },
                            out_sym_dims.clone(),
                            vec![InputRef::Broadcast(w_atom), input_ref],
                        );
                        if first_mul_base.is_none() {
                            first_mul_base = Some(b);
                        }
                    }
                }
                mul_bases.push(first_mul_base.unwrap());
            }
            all_mul_bases.push(mul_bases);
        }

        // Phase 2b: Emit ALL reduce groups in [B, C_out, spatial] order.
        // These must be contiguous so the output TensorAtomMap can address them
        // with simple row-major strides.
        for batch_mul_bases in &all_mul_bases {
            for &mul_base in batch_mul_bases {
                let b = ctx.nano.push_group(
                    spatial_size,
                    original_dtype,
                    ScalarOp::Reduce {
                        kind: ReduceKind::Sum,
                        reduce_count: k,
                        reduce_stride: spatial_size as i64,
                        compute_dtype: original_dtype,
                    },
                    out_sym_dims.clone(),
                    vec![InputRef::affine(mul_base, 1)],
                );
                if first_output_base.is_none() {
                    first_output_base = Some(b);
                }
            }
        }

        // Phase 3: Bias add (applied to all batch × channel × spatial atoms).
        let output_base;
        if let Some(ref bm) = bias_map {
            let reduce_base = first_output_base.unwrap();
            let mut first_bias_base = None;

            for bi in 0..batch_known {
                for co in 0..c_out {
                    let reduce_offset = (bi * c_out + co) * spatial_size;
                    let reduce_co_base = reduce_base.offset(reduce_offset);
                    let bias_atom = bm
                        .base_id
                        .offset(co * bm.known_strides.first().copied().unwrap_or(1));

                    let b = ctx.nano.push_group(
                        spatial_size,
                        original_dtype,
                        ScalarOp::Binary {
                            op: ScalarBinOp::Add,
                            compute_dtype: original_dtype,
                        },
                        out_sym_dims.clone(),
                        vec![
                            InputRef::affine(reduce_co_base, 1),
                            InputRef::Broadcast(bias_atom),
                        ],
                    );
                    if first_bias_base.is_none() {
                        first_bias_base = Some(b);
                    }
                }
            }
            output_base = first_bias_base.unwrap();
        } else {
            output_base = first_output_base.unwrap();
        }

        // --- Phase 4: Register output ---
        ctx.tensor_map.insert(
            out_id,
            TensorAtomMap::simple(
                output_base,
                out_known_dims.iter().product::<u64>().max(1),
                original_dtype,
                out_layout,
                TensorAtomMap::compute_strides(&out_known_dims),
                out_sym_dims,
            ),
        );
        crate::milli_graph::ops::LowerResult::Lowered
    }
}

impl MilliOp for Conv {
    fn infer<'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'p, P>>,
        symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        _pool: &'p P,
    ) -> Result<Vec<(GlobalId, crate::tensor_info::TensorInfo<'p, P>)>, MilliOpGraphError> {
        use crate::milli_graph::MilliOpGraphError::UnableToInfer;
        use crate::scalar_info::ScalarInfoTyped;
        use crate::symbolic_scalar::SymbolicScalarTyped;
        use crate::tensor_info::TensorInfo;

        let input_info = known_inputs.get(&self.input).ok_or(UnableToInfer)?;
        let weight_info = known_inputs.get(&self.weight).ok_or(UnableToInfer)?;
        let out_dtype = input_info.dtype();

        let input_ranked = input_info.as_ranked().ok_or(UnableToInfer)?;
        let weight_ranked = weight_info.as_ranked().ok_or(UnableToInfer)?;
        let input_shape = input_ranked.shape();
        let weight_shape = weight_ranked.shape();
        let rank = input_shape.len();

        if rank < 3 {
            return Err(UnableToInfer);
        }
        let n_spatial = rank - 2;

        // batch = input_shape[0]
        let batch = input_shape[0].clone();
        // out_channels = weight_shape[0]
        let out_channels = weight_shape[0].clone();

        // Try to compute concrete spatial output dims
        let kernel_shape: Vec<usize> = if self.kernel_shape.is_empty() {
            // Infer from weight shape[2..]
            let mut ks = Vec::new();
            for i in 0..n_spatial {
                match &weight_shape[i + 2] {
                    ScalarInfoTyped::Numeric(v) => ks.push(*v as usize),
                    ScalarInfoTyped::Symbolic(_) => {
                        return make_symbolic_output(
                            self.output,
                            out_dtype,
                            &batch,
                            &out_channels,
                            n_spatial,
                            symbolic_resolver,
                        );
                    }
                }
            }
            ks
        } else {
            self.kernel_shape.iter().map(|&x| x as usize).collect()
        };

        let strides: Vec<usize> = if self.strides.is_empty() {
            vec![1; n_spatial]
        } else {
            self.strides.iter().map(|&x| x as usize).collect()
        };
        let dilations: Vec<usize> = if self.dilations.is_empty() {
            vec![1; n_spatial]
        } else {
            self.dilations.iter().map(|&x| x as usize).collect()
        };

        let dilated_kernel: Vec<usize> = (0..n_spatial)
            .map(|i| dilations[i] * (kernel_shape[i] - 1) + 1)
            .collect();

        // Try to get concrete input spatial dims
        let input_spatial: Option<Vec<usize>> = (0..n_spatial)
            .map(|i| match &input_shape[i + 2] {
                ScalarInfoTyped::Numeric(v) => Some(*v as usize),
                ScalarInfoTyped::Symbolic(_) => None,
            })
            .collect();

        let spatial_dims: Vec<ScalarInfoTyped<u64>> = if let Some(input_spatial) = input_spatial {
            let (pad_begin, pad_end) = resolve_padding(
                self.auto_pad,
                &self.pads,
                n_spatial,
                &input_spatial,
                &strides,
                &dilated_kernel,
            );
            (0..n_spatial)
                .map(|i| {
                    let out_size = (input_spatial[i] + pad_begin[i] + pad_end[i]
                        - dilated_kernel[i])
                        / strides[i]
                        + 1;
                    ScalarInfoTyped::Numeric(out_size as u64)
                })
                .collect()
        } else {
            // For SameUpper/SameLower with symbolic input, we can still compute output = ceil(input/stride)
            // but for NotSet/Valid we need concrete dims
            match self.auto_pad {
                ConvAutoPad::SameUpper | ConvAutoPad::SameLower => (0..n_spatial)
                    .map(|i| match &input_shape[i + 2] {
                        ScalarInfoTyped::Numeric(v) => {
                            let out = (*v as usize).div_ceil(strides[i]);
                            ScalarInfoTyped::Numeric(out as u64)
                        }
                        ScalarInfoTyped::Symbolic(_) => {
                            ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver))
                        }
                    })
                    .collect(),
                _ => (0..n_spatial)
                    .map(|_| ScalarInfoTyped::Symbolic(SymbolicScalarTyped::new(symbolic_resolver)))
                    .collect(),
            }
        };

        let mut out_dims = Vec::with_capacity(rank);
        out_dims.push(batch);
        out_dims.push(out_channels);
        out_dims.extend(spatial_dims);

        Ok(vec![(
            self.output,
            TensorInfo::from_dtype_and_shape_scalars(out_dtype, &out_dims),
        )])
    }

    fn backward(
        &self,
        output_grads: &HashMap<GlobalId, GlobalId>,
        graph: &mut MilliOpGraph,
        rng: &mut impl Rng,
    ) -> Option<HashMap<GlobalId, GlobalId>> {
        let grad_output = *output_grads.get(&self.output)?;
        let mut result = HashMap::new();

        // dInput = col2im(W^T @ dY) — transposed convolution
        let grad_input = ConvInputGrad::push_new(
            graph,
            grad_output,
            self.weight,
            self.input,
            self.auto_pad,
            self.dilations.clone(),
            self.group,
            self.kernel_shape.clone(),
            self.pads.clone(),
            self.strides.clone(),
            rng,
        );
        result.insert(self.input, grad_input);

        // dWeight = dY @ im2col(X)^T, summed over batch
        let grad_weight = ConvWeightGrad::push_new(
            graph,
            grad_output,
            self.input,
            self.auto_pad,
            self.dilations.clone(),
            self.group,
            self.kernel_shape.clone(),
            self.pads.clone(),
            self.strides.clone(),
            rng,
        );
        result.insert(self.weight, grad_weight);

        // dBias = sum(dY, axes=[0, 2, 3, ...]) → [C_out]
        if let Some(bias_id) = self.bias {
            let grad_bias = ConvBiasGrad::push_new(graph, grad_output, rng);
            result.insert(bias_id, grad_bias);
        }

        Some(result)
    }
}

// ---------------------------------------------------------------------------
// Backward ops
// ---------------------------------------------------------------------------

/// col2im for 2D: scatter-add columns back into input space.
/// `col` has shape [cpg_in * kH * kW, OH * OW], output has shape [cpg_in, iH, iW].
fn col2im_2d(col: &[f32], output: &mut [f32], p: &Im2Col2dParams) {
    let out_spatial = p.out_h * p.out_w;
    let in_channel_stride = p.in_h * p.in_w;

    for ci in 0..p.channels_per_group_in {
        for kh in 0..p.kernel_h {
            for kw in 0..p.kernel_w {
                let col_row = (ci * p.kernel_h + kh) * p.kernel_w + kw;
                let col_offset = col_row * out_spatial;
                for oh in 0..p.out_h {
                    let ih = (oh * p.stride_h + kh * p.dilation_h) as isize - p.pad_top as isize;
                    if ih < 0 || ih >= p.in_h as isize {
                        continue;
                    }
                    let out_base = ci * in_channel_stride + ih as usize * p.in_w;
                    for ow in 0..p.out_w {
                        let iw =
                            (ow * p.stride_w + kw * p.dilation_w) as isize - p.pad_left as isize;
                        if iw >= 0 && iw < p.in_w as isize {
                            output[out_base + iw as usize] += col[col_offset + oh * p.out_w + ow];
                        }
                    }
                }
            }
        }
    }
}

// -- ConvInputGrad: dX = col2im(W^T @ dY) --

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvInputGrad {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    grad_output: GlobalId,
    weight: GlobalId,
    /// Original forward input — needed for its shape (stride>1 makes output→input ambiguous)
    input: GlobalId,
    auto_pad: ConvAutoPad,
    dilations: Vec<i64>,
    group: i64,
    kernel_shape: Vec<i64>,
    pads: Vec<i64>,
    strides: Vec<i64>,
}

impl ConvInputGrad {
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn push_new(
        graph: &mut MilliOpGraph,
        grad_output: GlobalId,
        weight: GlobalId,
        input: GlobalId,
        auto_pad: ConvAutoPad,
        dilations: Vec<i64>,
        group: i64,
        kernel_shape: Vec<i64>,
        pads: Vec<i64>,
        strides: Vec<i64>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new_with_label(
            graph,
            grad_output,
            weight,
            input,
            auto_pad,
            dilations,
            group,
            kernel_shape,
            pads,
            strides,
            None,
            rng,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        grad_output: GlobalId,
        weight: GlobalId,
        input: GlobalId,
        auto_pad: ConvAutoPad,
        dilations: Vec<i64>,
        group: i64,
        kernel_shape: Vec<i64>,
        pads: Vec<i64>,
        strides: Vec<i64>,
        label: Option<String>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            label,
            output,
            grad_output,
            weight,
            input,
            auto_pad,
            dilations,
            group,
            kernel_shape,
            pads,
            strides,
        };
        graph.push_op(AnyMilliOp::ConvInputGrad(node));
        output
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.grad_output, map);
        super::remap(&mut self.weight, map);
        super::remap(&mut self.input, map);
    }
}

impl Node for ConvInputGrad {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> String {
        "ConvInputGrad".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.grad_output, self.weight, self.input].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl MilliOp for ConvInputGrad {
    fn infer<'p, P: crate::pool::Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'p, P>>,
        _symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        _pool: &'p P,
    ) -> Result<Vec<(GlobalId, crate::tensor_info::TensorInfo<'p, P>)>, MilliOpGraphError> {
        // Output shape = original input shape.
        let input_info = known_inputs
            .get(&self.input)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let ranked = input_info
            .as_ranked()
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let out = crate::tensor_info::TensorInfo::from_dtype_and_shape_scalars(
            input_info.dtype(),
            &ranked.shape(),
        );
        Ok(vec![(self.output, out)])
    }

    fn eval_new<'p, P2: crate::pool::Pool + 'p>(
        &self,
        inputs: &[crate::numeric_tensor::NumericTensorView<'_, crate::tensor_rank::DynRank>],
        pool: &'p P2,
    ) -> Result<
        Vec<crate::numeric_tensor::NumericTensor<'p, crate::tensor_rank::DynRank, P2>>,
        crate::nano_graph::pool_eval::PoolEvalError,
    > {
        use crate::numeric_scalar::NumericScalar;
        use crate::numeric_tensor::{NumericTensor as PoolTensor, TensorLayout};
        use crate::tensor_rank::DynRank;

        // inputs: [grad_output, weight, input]
        let grad_view = &inputs[0];
        let weight_view = &inputs[1];
        let orig_input_view = &inputs[2];

        let grad_shape = grad_view.shape();
        let weight_shape = weight_view.shape();
        let input_shape = orig_input_view.shape();
        let n_spatial = grad_shape.len() - 2;
        let batch_size = grad_shape[0] as usize;
        let out_channels = grad_shape[1] as usize;
        let group = self.group as usize;
        let channels_per_group_in = weight_shape[1] as usize;
        let channels_per_group_out = out_channels / group;
        let in_channels = channels_per_group_in * group;

        let kernel_shape: Vec<usize> = if self.kernel_shape.is_empty() {
            (0..n_spatial)
                .map(|i| weight_shape[i + 2] as usize)
                .collect()
        } else {
            self.kernel_shape.iter().map(|&x| x as usize).collect()
        };
        let strides: Vec<usize> = if self.strides.is_empty() {
            vec![1; n_spatial]
        } else {
            self.strides.iter().map(|&x| x as usize).collect()
        };
        let dilations: Vec<usize> = if self.dilations.is_empty() {
            vec![1; n_spatial]
        } else {
            self.dilations.iter().map(|&x| x as usize).collect()
        };

        let out_spatial: Vec<usize> = (0..n_spatial).map(|i| grad_shape[i + 2] as usize).collect();
        let out_spatial_size: usize = out_spatial.iter().product();
        let k_per_group: usize = channels_per_group_in * kernel_shape.iter().product::<usize>();

        let input_spatial: Vec<usize> = (0..n_spatial)
            .map(|i| input_shape[i + 2] as usize)
            .collect();

        let dilated_kernel: Vec<usize> = (0..n_spatial)
            .map(|i| dilations[i] * (kernel_shape[i] - 1) + 1)
            .collect();
        let (pad_begin, _) = resolve_padding(
            self.auto_pad,
            &self.pads,
            n_spatial,
            &input_spatial,
            &strides,
            &dilated_kernel,
        );

        // Extract f32 data.
        let weight_data: Vec<f32> = (0..weight_view.numel())
            .map(|i| weight_view.read_element(i).to_f32())
            .collect();
        let grad_data: Vec<f32> = (0..grad_view.numel())
            .map(|i| grad_view.read_element(i).to_f32())
            .collect();

        let in_spatial_size: usize = input_spatial.iter().product();
        let mut dx = vec![0.0f32; batch_size * in_channels * in_spatial_size];

        assert_eq!(n_spatial, 2, "ConvInputGrad currently supports 2D only");
        let in_h = input_spatial[0];
        let in_w = input_spatial[1];
        let out_h = out_spatial[0];
        let out_w = out_spatial[1];

        for n in 0..batch_size {
            for g in 0..group {
                let grad_offset = n * out_channels * out_spatial_size
                    + g * channels_per_group_out * out_spatial_size;
                let grad_slice = &grad_data
                    [grad_offset..grad_offset + channels_per_group_out * out_spatial_size];

                // Weight for this group: [cpg_out, K] from flat weight_data.
                let w_offset = g * channels_per_group_out * k_per_group;
                let w_slice =
                    &weight_data[w_offset..w_offset + channels_per_group_out * k_per_group];

                // d_col = W^T @ dY  →  [K, out_spatial_size]
                let mut d_col = vec![0.0f32; k_per_group * out_spatial_size];
                for k in 0..k_per_group {
                    for s in 0..out_spatial_size {
                        let mut sum = 0.0f32;
                        for co in 0..channels_per_group_out {
                            sum += w_slice[co * k_per_group + k]
                                * grad_slice[co * out_spatial_size + s];
                        }
                        d_col[k * out_spatial_size + s] = sum;
                    }
                }

                // col2im: scatter d_col back to input space
                let dx_offset =
                    n * in_channels * in_spatial_size + g * channels_per_group_in * in_spatial_size;
                let dx_slice =
                    &mut dx[dx_offset..dx_offset + channels_per_group_in * in_spatial_size];
                col2im_2d(
                    &d_col,
                    dx_slice,
                    &Im2Col2dParams {
                        in_base: 0,
                        channels_per_group_in,
                        in_h,
                        in_w,
                        out_h,
                        out_w,
                        kernel_h: kernel_shape[0],
                        kernel_w: kernel_shape[1],
                        stride_h: strides[0],
                        stride_w: strides[1],
                        dilation_h: dilations[0],
                        dilation_w: dilations[1],
                        pad_top: pad_begin[0],
                        pad_left: pad_begin[1],
                    },
                );
            }
        }

        let mut result_shape: Vec<u64> = vec![batch_size as u64, in_channels as u64];
        result_shape.extend(input_spatial.iter().map(|&s| s as u64));
        let dtype = grad_view.dtype();
        let layout = TensorLayout::<DynRank>::row_major(result_shape, dtype);
        let buf = pool
            .allocate(layout.buffer_size_bytes())
            .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
        let mut out = PoolTensor::from_parts(buf, layout);
        for (i, &v) in dx.iter().enumerate() {
            out.write_element(i, NumericScalar::from_f32(v).cast_to(dtype));
        }
        Ok(vec![out])
    }
}

// -- ConvWeightGrad: dW = sum_n(dY @ im2col(X)^T) --

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvWeightGrad {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    grad_output: GlobalId,
    input: GlobalId,
    auto_pad: ConvAutoPad,
    dilations: Vec<i64>,
    group: i64,
    kernel_shape: Vec<i64>,
    pads: Vec<i64>,
    strides: Vec<i64>,
}

impl ConvWeightGrad {
    #[allow(clippy::too_many_arguments)]
    pub fn push_new(
        graph: &mut MilliOpGraph,
        grad_output: GlobalId,
        input: GlobalId,
        auto_pad: ConvAutoPad,
        dilations: Vec<i64>,
        group: i64,
        kernel_shape: Vec<i64>,
        pads: Vec<i64>,
        strides: Vec<i64>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new_with_label(
            graph,
            grad_output,
            input,
            auto_pad,
            dilations,
            group,
            kernel_shape,
            pads,
            strides,
            None,
            rng,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        grad_output: GlobalId,
        input: GlobalId,
        auto_pad: ConvAutoPad,
        dilations: Vec<i64>,
        group: i64,
        kernel_shape: Vec<i64>,
        pads: Vec<i64>,
        strides: Vec<i64>,
        label: Option<String>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            label,
            output,
            grad_output,
            input,
            auto_pad,
            dilations,
            group,
            kernel_shape,
            pads,
            strides,
        };
        graph.push_op(AnyMilliOp::ConvWeightGrad(node));
        output
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.grad_output, map);
        super::remap(&mut self.input, map);
    }
}

impl Node for ConvWeightGrad {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> String {
        "ConvWeightGrad".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.grad_output, self.input].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl MilliOp for ConvWeightGrad {
    fn infer<'p, P: crate::pool::Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'p, P>>,
        _symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        pool: &'p P,
    ) -> Result<Vec<(GlobalId, crate::tensor_info::TensorInfo<'p, P>)>, MilliOpGraphError> {
        // Weight shape: [out_channels, in_channels/groups, *kernel_shape]
        let grad_info = known_inputs
            .get(&self.grad_output)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let input_info = known_inputs
            .get(&self.input)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let grad_ranked = grad_info
            .as_ranked()
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let input_ranked = input_info
            .as_ranked()
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let grad_shape = grad_ranked.shape();
        let input_shape = input_ranked.shape();
        if grad_shape.len() < 2 || input_shape.len() < 2 {
            return Err(MilliOpGraphError::UnableToInfer);
        }
        use crate::scalar_info::ScalarInfoTyped;
        let out_channels = match &grad_shape[1] {
            ScalarInfoTyped::Numeric(v) => *v,
            _ => return Err(MilliOpGraphError::UnableToInfer),
        };
        let in_channels = match &input_shape[1] {
            ScalarInfoTyped::Numeric(v) => *v,
            _ => return Err(MilliOpGraphError::UnableToInfer),
        };
        let group = self.group.max(1) as u64;
        let channels_per_group = in_channels / group;
        let mut weight_shape = vec![
            ScalarInfoTyped::Numeric(out_channels),
            ScalarInfoTyped::Numeric(channels_per_group),
        ];
        for &k in &self.kernel_shape {
            weight_shape.push(ScalarInfoTyped::Numeric(k as u64));
        }
        let out = crate::tensor_info::TensorInfo::from_dtype_and_shape_scalars(
            grad_info.dtype(),
            &weight_shape,
        );
        Ok(vec![(self.output, out)])
    }

    fn eval_new<'p, P2: crate::pool::Pool + 'p>(
        &self,
        inputs: &[crate::numeric_tensor::NumericTensorView<'_, crate::tensor_rank::DynRank>],
        pool: &'p P2,
    ) -> Result<
        Vec<crate::numeric_tensor::NumericTensor<'p, crate::tensor_rank::DynRank, P2>>,
        crate::nano_graph::pool_eval::PoolEvalError,
    > {
        use crate::numeric_scalar::NumericScalar;
        use crate::numeric_tensor::{NumericTensor as PoolTensor, TensorLayout};
        use crate::tensor_rank::DynRank;

        // inputs: [grad_output, input]
        let grad_view = &inputs[0];
        let input_view = &inputs[1];

        let input_shape = input_view.shape();
        let grad_shape = grad_view.shape();
        let n_spatial = input_shape.len() - 2;
        let batch_size = input_shape[0] as usize;
        let out_channels = grad_shape[1] as usize;
        let group = self.group as usize;
        let channels_per_group_in = (input_shape[1] as usize) / group;
        let channels_per_group_out = out_channels / group;

        let kernel_shape: Vec<usize> = if self.kernel_shape.is_empty() {
            panic!("ConvWeightGrad requires explicit kernel_shape");
        } else {
            self.kernel_shape.iter().map(|&x| x as usize).collect()
        };
        let strides: Vec<usize> = if self.strides.is_empty() {
            vec![1; n_spatial]
        } else {
            self.strides.iter().map(|&x| x as usize).collect()
        };
        let dilations: Vec<usize> = if self.dilations.is_empty() {
            vec![1; n_spatial]
        } else {
            self.dilations.iter().map(|&x| x as usize).collect()
        };

        let input_spatial: Vec<usize> = (0..n_spatial)
            .map(|i| input_shape[i + 2] as usize)
            .collect();
        let out_spatial: Vec<usize> = (0..n_spatial).map(|i| grad_shape[i + 2] as usize).collect();
        let out_spatial_size: usize = out_spatial.iter().product();
        let k_per_group: usize = channels_per_group_in * kernel_shape.iter().product::<usize>();

        let dilated_kernel: Vec<usize> = (0..n_spatial)
            .map(|i| dilations[i] * (kernel_shape[i] - 1) + 1)
            .collect();
        let (pad_begin, _) = resolve_padding(
            self.auto_pad,
            &self.pads,
            n_spatial,
            &input_spatial,
            &strides,
            &dilated_kernel,
        );

        // Extract f32 data.
        let input_data: Vec<f32> = (0..input_view.numel())
            .map(|i| input_view.read_element(i).to_f32())
            .collect();
        let grad_data: Vec<f32> = (0..grad_view.numel())
            .map(|i| grad_view.read_element(i).to_f32())
            .collect();

        assert_eq!(n_spatial, 2, "ConvWeightGrad currently supports 2D only");
        let in_h = input_spatial[0];
        let in_w = input_spatial[1];
        let out_h = out_spatial[0];
        let out_w = out_spatial[1];
        let in_batch_stride = input_shape[1] as usize * in_h * in_w;

        let mut dw = vec![0.0f32; group * channels_per_group_out * k_per_group];

        for n in 0..batch_size {
            for g in 0..group {
                // im2col of input for this (n, g)
                let in_base = n * in_batch_stride + g * channels_per_group_in * in_h * in_w;
                let mut col = vec![0.0f32; k_per_group * out_spatial_size];
                im2col_2d(
                    &input_data,
                    &mut col,
                    &Im2Col2dParams {
                        in_base,
                        channels_per_group_in,
                        in_h,
                        in_w,
                        out_h,
                        out_w,
                        kernel_h: kernel_shape[0],
                        kernel_w: kernel_shape[1],
                        stride_h: strides[0],
                        stride_w: strides[1],
                        dilation_h: dilations[0],
                        dilation_w: dilations[1],
                        pad_top: pad_begin[0],
                        pad_left: pad_begin[1],
                    },
                );

                // dY for this (n, g): [cpg_out, out_spatial_size]
                let grad_offset = n * out_channels * out_spatial_size
                    + g * channels_per_group_out * out_spatial_size;
                let grad_slice = &grad_data
                    [grad_offset..grad_offset + channels_per_group_out * out_spatial_size];

                // dW_g += dY @ col^T  →  [cpg_out, K]
                let dw_offset = g * channels_per_group_out * k_per_group;
                for co in 0..channels_per_group_out {
                    for k in 0..k_per_group {
                        let mut sum = 0.0f32;
                        for s in 0..out_spatial_size {
                            sum += grad_slice[co * out_spatial_size + s]
                                * col[k * out_spatial_size + s];
                        }
                        dw[dw_offset + co * k_per_group + k] += sum;
                    }
                }
            }
        }

        // Reshape to [C_out, C_in/g, *kernel]
        let mut result_shape: Vec<u64> = vec![out_channels as u64, channels_per_group_in as u64];
        result_shape.extend(kernel_shape.iter().map(|&s| s as u64));
        let dtype = grad_view.dtype();
        let layout = TensorLayout::<DynRank>::row_major(result_shape, dtype);
        let buf = pool
            .allocate(layout.buffer_size_bytes())
            .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
        let mut out = PoolTensor::from_parts(buf, layout);
        for (i, &v) in dw.iter().enumerate() {
            out.write_element(i, NumericScalar::from_f32(v).cast_to(dtype));
        }
        Ok(vec![out])
    }
}

// -- ConvBiasGrad: dBias = sum(dY, axes=[0, 2, 3, ...]) --

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvBiasGrad {
    global_id: GlobalId,
    pub(crate) label: Option<String>,
    output: GlobalId,
    grad_output: GlobalId,
}

impl ConvBiasGrad {
    pub fn push_new(
        graph: &mut MilliOpGraph,
        grad_output: GlobalId,
        rng: &mut impl Rng,
    ) -> GlobalId {
        Self::push_new_with_label(graph, grad_output, None, rng)
    }

    pub fn push_new_with_label(
        graph: &mut MilliOpGraph,
        grad_output: GlobalId,
        label: Option<String>,
        rng: &mut impl Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            label,
            output,
            grad_output,
        };
        graph.push_op(AnyMilliOp::ConvBiasGrad(node));
        output
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.grad_output, map);
    }
}

impl Node for ConvBiasGrad {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> String {
        "ConvBiasGrad".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.grad_output))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl MilliOp for ConvBiasGrad {
    fn infer<'p, P: Pool + 'p>(
        &self,
        known_inputs: &HashMap<GlobalId, crate::tensor_info::TensorInfo<'p, P>>,
        _symbolic_resolver: &mut crate::symbolic_scalar::SymbolicResolver,
        _pool: &'p P,
    ) -> Result<Vec<(GlobalId, crate::tensor_info::TensorInfo<'p, P>)>, MilliOpGraphError> {
        use crate::tensor_info::TensorInfo;

        let grad_info = known_inputs
            .get(&self.grad_output)
            .ok_or(MilliOpGraphError::UnableToInfer)?;
        let out_dtype = grad_info.dtype();

        // Output is [C_out] where C_out = grad_output.shape[1].
        if let Some(ranked) = grad_info.as_ranked() {
            let shape = ranked.shape();
            if shape.len() >= 2 {
                return Ok(vec![(
                    self.output,
                    TensorInfo::from_dtype_and_shape_scalars(out_dtype, &[shape[1].clone()]),
                )]);
            }
        }

        Err(MilliOpGraphError::UnableToInfer)
    }

    fn eval_new<'p, P2: crate::pool::Pool + 'p>(
        &self,
        inputs: &[crate::numeric_tensor::NumericTensorView<'_, crate::tensor_rank::DynRank>],
        pool: &'p P2,
    ) -> Result<
        Vec<crate::numeric_tensor::NumericTensor<'p, crate::tensor_rank::DynRank, P2>>,
        crate::nano_graph::pool_eval::PoolEvalError,
    > {
        use crate::numeric_scalar::NumericScalar;
        use crate::numeric_tensor::{NumericTensor, TensorLayout};
        use crate::tensor_rank::DynRank;

        let grad = &inputs[0];
        let shape = grad.shape();
        let rank = shape.len();
        let dtype = grad.dtype();

        // Output: [C_out] = sum over axes [0, 2, 3, ...].
        let c_out = if rank >= 2 { shape[1] } else { shape[0] };
        let out_shape = vec![c_out];
        let out_numel = c_out as usize;

        let layout = TensorLayout::<DynRank>::row_major(out_shape, dtype);
        let buf = pool
            .allocate(layout.buffer_size_bytes())
            .map_err(crate::nano_graph::pool_eval::PoolEvalError::Allocation)?;
        let mut out = NumericTensor::from_parts(buf, layout);

        for i in 0..out_numel {
            out.write_element(i, NumericScalar::zero(dtype));
        }

        // Sum all elements with the same channel index.
        let channel_stride = if rank >= 2 {
            let mut s = 1usize;
            for d in 2..rank {
                s *= shape[d] as usize;
            }
            s
        } else {
            1
        };
        let spatial_size = channel_stride;
        let batch_stride = c_out as usize * spatial_size;

        for flat in 0..grad.numel() {
            let co = (flat % batch_stride) / spatial_size;
            let val = grad.read_element(flat);
            let cur = out.read_element(co);
            out.write_element(co, cur.add(val));
        }

        Ok(vec![out])
    }
}
