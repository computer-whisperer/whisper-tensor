use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::graph::GlobalId;
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::numeric_tensor::NumericTensor;
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
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
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
                let total_pad =
                    ((out_size - 1) * strides[i] + dilated_kernel[i]).saturating_sub(input_spatial[i]);
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

/// Specialized 2D convolution with rayon parallelism.
/// Parallelizes across output channels (batch * group * channels_per_group_out).
fn conv2d_parallel(
    input_data: &[f32],
    weight_data: &[f32],
    bias_data: Option<&[f32]>,
    batch_size: usize,
    in_channels: usize,
    out_channels: usize,
    group: usize,
    channels_per_group_in: usize,
    channels_per_group_out: usize,
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
) -> Vec<f32> {
    let out_spatial = out_h * out_w;
    let in_channel_stride = in_h * in_w;
    let in_batch_stride = in_channels * in_channel_stride;
    let w_kernel_size = kernel_h * kernel_w;
    let w_cin_stride = w_kernel_size;
    let w_cout_stride = channels_per_group_in * w_cin_stride;
    let total_out = batch_size * out_channels * out_spatial;

    let mut output_data = vec![0.0f32; total_out];

    // Parallelize over (batch, output_channel) pairs
    let work_items: Vec<(usize, usize, usize)> = (0..batch_size)
        .flat_map(|n| {
            (0..group).flat_map(move |g| {
                (0..channels_per_group_out).map(move |co| (n, g, co))
            })
        })
        .collect();

    let chunk_results: Vec<(usize, Vec<f32>)> = work_items
        .par_iter()
        .map(|&(n, g, co)| {
            let m = g * channels_per_group_out + co;
            let bias_val = bias_data.map_or(0.0, |b| b[m]);
            let mut out_buf = vec![0.0f32; out_spatial];

            for ci in 0..channels_per_group_in {
                let in_c = g * channels_per_group_in + ci;
                let in_base = n * in_batch_stride + in_c * in_channel_stride;
                let w_base = m * w_cout_stride + ci * w_cin_stride;

                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0f32;
                        for kh in 0..kernel_h {
                            let ih = oh * stride_h + kh * dilation_h;
                            let ih = ih as isize - pad_top as isize;
                            if ih < 0 || ih >= in_h as isize {
                                continue;
                            }
                            let ih = ih as usize;
                            let in_row = in_base + ih * in_w;
                            let w_row = w_base + kh * kernel_w;
                            for kw in 0..kernel_w {
                                let iw = ow * stride_w + kw * dilation_w;
                                let iw = iw as isize - pad_left as isize;
                                if iw < 0 || iw >= in_w as isize {
                                    continue;
                                }
                                sum += input_data[in_row + iw as usize] * weight_data[w_row + kw];
                            }
                        }
                        out_buf[oh * out_w + ow] += sum;
                    }
                }
            }

            // Add bias
            if bias_val != 0.0 {
                for v in &mut out_buf {
                    *v += bias_val;
                }
            }

            let out_offset = n * (out_channels * out_spatial) + m * out_spatial;
            (out_offset, out_buf)
        })
        .collect();

    for (offset, buf) in chunk_results {
        output_data[offset..offset + out_spatial].copy_from_slice(&buf);
    }

    output_data
}

/// Generic n-dimensional convolution fallback (1D, 3D+).
fn conv_nd_generic(
    input_data: &[f32],
    weight_data: &[f32],
    bias_data: Option<&[f32]>,
    batch_size: usize,
    in_channels: usize,
    out_channels: usize,
    group: usize,
    channels_per_group_in: usize,
    channels_per_group_out: usize,
    n_spatial: usize,
    input_spatial: &[usize],
    out_spatial: &[usize],
    kernel_shape: &[usize],
    strides: &[usize],
    dilations: &[usize],
    pad_begin: &[usize],
) -> Vec<f32> {
    let out_spatial_size: usize = out_spatial.iter().product();
    let total_out = batch_size * out_channels * out_spatial_size;
    let in_channel_stride: usize = input_spatial.iter().product();
    let in_batch_stride = in_channels * in_channel_stride;
    let w_kernel_size: usize = kernel_shape.iter().product();
    let w_cin_stride = w_kernel_size;
    let w_cout_stride = channels_per_group_in * w_cin_stride;

    let mut in_spatial_strides = vec![1usize; n_spatial];
    for i in (0..n_spatial.saturating_sub(1)).rev() {
        in_spatial_strides[i] = in_spatial_strides[i + 1] * input_spatial[i + 1];
    }
    let mut out_spatial_strides = vec![1usize; n_spatial];
    for i in (0..n_spatial.saturating_sub(1)).rev() {
        out_spatial_strides[i] = out_spatial_strides[i + 1] * out_spatial[i + 1];
    }
    let mut kernel_strides = vec![1usize; n_spatial];
    for i in (0..n_spatial.saturating_sub(1)).rev() {
        kernel_strides[i] = kernel_strides[i + 1] * kernel_shape[i + 1];
    }

    let mut output_data = vec![0.0f32; total_out];

    // Collect work items for parallelism
    let work_items: Vec<(usize, usize, usize)> = (0..batch_size)
        .flat_map(|n| {
            (0..group).flat_map(move |g| {
                (0..channels_per_group_out).map(move |co| (n, g, co))
            })
        })
        .collect();

    let chunk_results: Vec<(usize, Vec<f32>)> = work_items
        .par_iter()
        .map(|&(n, g, co)| {
            let m = g * channels_per_group_out + co;
            let bias_val = bias_data.map_or(0.0, |b| b[m]);
            let mut out_buf = vec![0.0f32; out_spatial_size];

            let mut out_coords = vec![0usize; n_spatial];
            let mut k_coords = vec![0usize; n_spatial];

            for ci in 0..channels_per_group_in {
                let in_c = g * channels_per_group_in + ci;

                for out_idx in 0..out_spatial_size {
                    // Decompose out_idx into spatial coordinates
                    let mut remaining = out_idx;
                    for d in 0..n_spatial {
                        out_coords[d] = remaining / out_spatial_strides[d];
                        remaining %= out_spatial_strides[d];
                    }

                    let mut sum = 0.0f32;

                    for k_idx in 0..w_kernel_size {
                        let mut k_remaining = k_idx;
                        for d in 0..n_spatial {
                            k_coords[d] = k_remaining / kernel_strides[d];
                            k_remaining %= kernel_strides[d];
                        }

                        let mut in_bounds = true;
                        let mut in_spatial_offset = 0usize;
                        for d in 0..n_spatial {
                            let pos = (out_coords[d] * strides[d] + k_coords[d] * dilations[d])
                                as isize
                                - pad_begin[d] as isize;
                            if pos < 0 || pos >= input_spatial[d] as isize {
                                in_bounds = false;
                                break;
                            }
                            in_spatial_offset += pos as usize * in_spatial_strides[d];
                        }

                        if in_bounds {
                            let in_offset =
                                n * in_batch_stride + in_c * in_channel_stride + in_spatial_offset;
                            let w_offset = m * w_cout_stride + ci * w_cin_stride + k_idx;
                            sum += input_data[in_offset] * weight_data[w_offset];
                        }
                    }
                    out_buf[out_idx] += sum;
                }
            }

            // Add bias
            if bias_val != 0.0 {
                for v in &mut out_buf {
                    *v += bias_val;
                }
            }

            let out_offset = n * (out_channels * out_spatial_size) + m * out_spatial_size;
            (out_offset, out_buf)
        })
        .collect();

    for (offset, buf) in chunk_results {
        output_data[offset..offset + buf.len()].copy_from_slice(&buf);
    }

    output_data
}

impl MilliOp for Conv {
    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> super::EvalResult {
        let input = &inputs[&self.input];
        let weight = &inputs[&self.weight];
        let original_dtype = input.dtype();

        let input_shape: Vec<usize> = input.shape().iter().map(|&x| x as usize).collect();
        let weight_shape: Vec<usize> = weight.shape().iter().map(|&x| x as usize).collect();

        // Cast to f32 and flatten for computation
        let input_f32 = input.cast(DType::F32, backend)?;
        let weight_f32 = weight.cast(DType::F32, backend)?;
        let input_data: Vec<f32> = input_f32.to_ndarray()?.flatten().try_into()?;
        let weight_data: Vec<f32> = weight_f32.to_ndarray()?.flatten().try_into()?;
        let bias_data: Option<Vec<f32>> = if let Some(bias_id) = self.bias {
            let bias = &inputs[&bias_id];
            let bias_f32 = bias.cast(DType::F32, backend)?;
            Some(bias_f32.to_ndarray()?.flatten().try_into()?)
        } else {
            None
        };

        let n_spatial = input_shape.len() - 2;
        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let out_channels = weight_shape[0];
        let group = self.group as usize;
        let channels_per_group_in = weight_shape[1];
        let channels_per_group_out = out_channels / group;

        debug_assert_eq!(in_channels, channels_per_group_in * group);

        let kernel_shape: Vec<usize> = if self.kernel_shape.is_empty() {
            (0..n_spatial).map(|i| weight_shape[i + 2]).collect()
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

        let input_spatial: Vec<usize> = (0..n_spatial).map(|i| input_shape[i + 2]).collect();
        let (pad_begin, pad_end) = resolve_padding(
            self.auto_pad,
            &self.pads,
            n_spatial,
            &input_spatial,
            &strides,
            &dilated_kernel,
        );

        let out_spatial: Vec<usize> = (0..n_spatial)
            .map(|i| {
                (input_spatial[i] + pad_begin[i] + pad_end[i] - dilated_kernel[i]) / strides[i] + 1
            })
            .collect();

        let output_data = if n_spatial == 2 {
            conv2d_parallel(
                &input_data,
                &weight_data,
                bias_data.as_deref(),
                batch_size,
                in_channels,
                out_channels,
                group,
                channels_per_group_in,
                channels_per_group_out,
                input_spatial[0],
                input_spatial[1],
                out_spatial[0],
                out_spatial[1],
                kernel_shape[0],
                kernel_shape[1],
                strides[0],
                strides[1],
                dilations[0],
                dilations[1],
                pad_begin[0],
                pad_begin[1],
            )
        } else {
            conv_nd_generic(
                &input_data,
                &weight_data,
                bias_data.as_deref(),
                batch_size,
                in_channels,
                out_channels,
                group,
                channels_per_group_in,
                channels_per_group_out,
                n_spatial,
                &input_spatial,
                &out_spatial,
                &kernel_shape,
                &strides,
                &dilations,
                &pad_begin,
            )
        };

        let mut out_shape = Vec::with_capacity(2 + n_spatial);
        out_shape.push(batch_size);
        out_shape.push(out_channels);
        out_shape.extend_from_slice(&out_spatial);

        let result = NumericTensor::<DynRank>::from_vec_shape(output_data, out_shape)?;
        let result = result.cast(original_dtype, backend)?;

        Ok(Box::new(std::iter::once((self.output, result))))
    }
}
