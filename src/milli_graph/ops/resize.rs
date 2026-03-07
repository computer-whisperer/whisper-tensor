use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::graph::GlobalId;
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::MilliOpGraphError;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::numeric_tensor::NumericTensor;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ResizeMode {
    Nearest,
    Linear,
    Cubic,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ResizeCoordTransform {
    HalfPixel,
    HalfPixelSymmetric,
    PytorchHalfPixel,
    AlignCorners,
    Asymmetric,
    TFCropAndResize,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ResizeNearestMode {
    RoundPreferFloor,
    RoundPreferCeil,
    Ceil,
    Floor,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ResizeKeepAspectRatioPolicy {
    Stretch,
    NotLarger,
    NotSmaller,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resize {
    global_id: GlobalId,
    output: GlobalId,
    input: GlobalId,
    roi: Option<GlobalId>,
    scales: Option<GlobalId>,
    sizes: Option<GlobalId>,
    mode: ResizeMode,
    coord_transform: ResizeCoordTransform,
    nearest_mode: ResizeNearestMode,
    cubic_coeff_a: f32,
    antialias: bool,
    exclude_outside: bool,
    extrapolation_value: f32,
    axes: Vec<i64>,
    keep_aspect_ratio_policy: ResizeKeepAspectRatioPolicy,
}

impl Resize {
    #[allow(clippy::too_many_arguments)]
    pub fn push_new(
        graph: &mut MilliOpGraph,
        input: GlobalId,
        roi: Option<GlobalId>,
        scales: Option<GlobalId>,
        sizes: Option<GlobalId>,
        mode: ResizeMode,
        coord_transform: ResizeCoordTransform,
        nearest_mode: ResizeNearestMode,
        cubic_coeff_a: f32,
        antialias: bool,
        exclude_outside: bool,
        extrapolation_value: f32,
        axes: Vec<i64>,
        keep_aspect_ratio_policy: ResizeKeepAspectRatioPolicy,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let output = graph.get_new_tensor_id(rng);
        let node = Self {
            global_id: GlobalId::new(rng),
            output,
            input,
            roi,
            scales,
            sizes,
            mode,
            coord_transform,
            nearest_mode,
            cubic_coeff_a,
            antialias,
            exclude_outside,
            extrapolation_value,
            axes,
            keep_aspect_ratio_policy,
        };
        graph.push_op(AnyMilliOp::Resize(node));
        output
    }

    pub fn remap_tensors(&mut self, map: &HashMap<GlobalId, GlobalId>, rng: &mut impl rand::Rng) {
        self.global_id = GlobalId::new(rng);
        super::remap(&mut self.output, map);
        super::remap(&mut self.input, map);
        super::remap_opt(&mut self.roi, map);
        super::remap_opt(&mut self.scales, map);
        super::remap_opt(&mut self.sizes, map);
    }
}

impl crate::graph::Node for Resize {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Resize".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        let mut res = vec![self.input];
        if let Some(roi) = self.roi {
            res.push(roi);
        }
        if let Some(scales) = self.scales {
            res.push(scales);
        }
        if let Some(sizes) = self.sizes {
            res.push(sizes);
        }
        Box::new(res.into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

// =============================================================================
// Fast paths for common cases
// =============================================================================

/// Compute the original (input-space) coordinate for a given output coordinate.
/// Shared by both nearest and linear fast paths.
fn fast_path_original_coord(
    out_coord: usize,
    scale: f32,
    in_size: usize,
    out_size: usize,
    coord_transform: ResizeCoordTransform,
) -> f32 {
    let x = out_coord as f32;
    match coord_transform {
        ResizeCoordTransform::HalfPixel => (x + 0.5) / scale - 0.5,
        ResizeCoordTransform::Asymmetric => x / scale,
        ResizeCoordTransform::PytorchHalfPixel => {
            if out_size > 1 { (x + 0.5) / scale - 0.5 } else { -0.5 }
        }
        ResizeCoordTransform::AlignCorners => {
            let output_width = scale * in_size as f32;
            if output_width <= 1.0 { 0.0 } else { x * (in_size as f32 - 1.0) / (output_width - 1.0) }
        }
        ResizeCoordTransform::HalfPixelSymmetric => {
            let output_width = scale * in_size as f32;
            let adjustment = out_size as f32 / output_width;
            let center = in_size as f32 / 2.0;
            let offset = center * (1.0 - adjustment);
            offset + (x + 0.5) / scale - 0.5
        }
        ResizeCoordTransform::TFCropAndResize => x / scale, // simplified, no ROI
    }
}

/// Map an output coordinate back to an input coordinate for nearest-mode.
/// Returns the clamped input index.
fn nearest_input_coord(
    out_coord: usize,
    scale: f32,
    in_size: usize,
    out_size: usize,
    coord_transform: ResizeCoordTransform,
    nearest_mode: ResizeNearestMode,
) -> usize {
    let x_ori = fast_path_original_coord(out_coord, scale, in_size, out_size, coord_transform);

    let nearest_idx = match nearest_mode {
        ResizeNearestMode::RoundPreferFloor => {
            if x_ori == x_ori.floor() + 0.5 { x_ori.floor() as i64 } else { (x_ori + 0.5).floor() as i64 }
        }
        ResizeNearestMode::RoundPreferCeil => x_ori.round() as i64,
        ResizeNearestMode::Floor => x_ori.floor() as i64,
        ResizeNearestMode::Ceil => x_ori.ceil() as i64,
    };
    nearest_idx.max(0).min(in_size as i64 - 1) as usize
}

/// Fast path: NCHW nearest resize where only H and W change.
/// Parallelizes across N*C channels.
fn resize_nchw_nearest(
    input: &[f32],
    n: usize,
    c: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    scale_h: f32,
    scale_w: f32,
    coord_transform: ResizeCoordTransform,
    nearest_mode: ResizeNearestMode,
) -> Vec<f32> {
    // Precompute index maps for H and W
    let h_map: Vec<usize> = (0..out_h)
        .map(|oh| nearest_input_coord(oh, scale_h, in_h, out_h, coord_transform, nearest_mode))
        .collect();
    let w_map: Vec<usize> = (0..out_w)
        .map(|ow| nearest_input_coord(ow, scale_w, in_w, out_w, coord_transform, nearest_mode))
        .collect();

    let in_spatial = in_h * in_w;
    let out_spatial = out_h * out_w;
    let total_channels = n * c;

    let chunks: Vec<Vec<f32>> = (0..total_channels)
        .into_par_iter()
        .map(|ch| {
            let in_base = ch * in_spatial;
            let mut out_buf = vec![0.0f32; out_spatial];
            for oh in 0..out_h {
                let ih = h_map[oh];
                let in_row = in_base + ih * in_w;
                let out_row = oh * out_w;
                for ow in 0..out_w {
                    out_buf[out_row + ow] = input[in_row + w_map[ow]];
                }
            }
            out_buf
        })
        .collect();

    let mut output = Vec::with_capacity(total_channels * out_spatial);
    for chunk in chunks {
        output.extend_from_slice(&chunk);
    }
    output
}

/// Fast path: NCHW bilinear resize where only H and W change.
/// Parallelizes across N*C channels.
fn resize_nchw_linear(
    input: &[f32],
    n: usize,
    c: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    scale_h: f32,
    scale_w: f32,
    coord_transform: ResizeCoordTransform,
) -> Vec<f32> {
    // Precompute interpolation parameters for H and W
    let h_params: Vec<(usize, usize, f32)> = (0..out_h)
        .map(|oh| bilinear_params(oh, scale_h, in_h, out_h, coord_transform))
        .collect();
    let w_params: Vec<(usize, usize, f32)> = (0..out_w)
        .map(|ow| bilinear_params(ow, scale_w, in_w, out_w, coord_transform))
        .collect();

    let in_spatial = in_h * in_w;
    let out_spatial = out_h * out_w;
    let total_channels = n * c;

    let chunks: Vec<Vec<f32>> = (0..total_channels)
        .into_par_iter()
        .map(|ch| {
            let in_base = ch * in_spatial;
            let mut out_buf = vec![0.0f32; out_spatial];
            for oh in 0..out_h {
                let (ih0, ih1, fh) = h_params[oh];
                let out_row = oh * out_w;
                for ow in 0..out_w {
                    let (iw0, iw1, fw) = w_params[ow];
                    // Bilinear: (1-fh)*(1-fw)*TL + (1-fh)*fw*TR + fh*(1-fw)*BL + fh*fw*BR
                    let tl = input[in_base + ih0 * in_w + iw0];
                    let tr = input[in_base + ih0 * in_w + iw1];
                    let bl = input[in_base + ih1 * in_w + iw0];
                    let br = input[in_base + ih1 * in_w + iw1];
                    out_buf[out_row + ow] =
                        (1.0 - fh) * ((1.0 - fw) * tl + fw * tr)
                        + fh * ((1.0 - fw) * bl + fw * br);
                }
            }
            out_buf
        })
        .collect();

    let mut output = Vec::with_capacity(total_channels * out_spatial);
    for chunk in chunks {
        output.extend_from_slice(&chunk);
    }
    output
}

/// Compute bilinear interpolation parameters for one output coordinate.
/// Returns (idx0, idx1, frac) where frac is the weight for idx1.
fn bilinear_params(
    out_coord: usize,
    scale: f32,
    in_size: usize,
    out_size: usize,
    coord_transform: ResizeCoordTransform,
) -> (usize, usize, f32) {
    let x_ori = fast_path_original_coord(out_coord, scale, in_size, out_size, coord_transform);

    let x0_raw = x_ori.floor() as i64;
    let x1_raw = x0_raw + 1;
    let frac = x_ori - x0_raw as f32;
    // Clamp independently for edge-padding
    let x0 = x0_raw.max(0).min(in_size as i64 - 1) as usize;
    let x1 = x1_raw.max(0).min(in_size as i64 - 1) as usize;
    (x0, x1, frac)
}

// =============================================================================
// Generic fallback (ONNX-spec-compliant separable interpolation)
// =============================================================================

fn cubic_kernel(x: f32, a: f32) -> f32 {
    let abs_x = x.abs();
    if abs_x <= 1.0 {
        ((a + 2.0) * abs_x - (a + 3.0)) * abs_x * abs_x + 1.0
    } else if abs_x < 2.0 {
        ((a * abs_x - 5.0 * a) * abs_x + 8.0 * a) * abs_x - 4.0 * a
    } else {
        0.0
    }
}

fn cubic_coeffs(ratio: f32, a: f32) -> [f32; 4] {
    [
        cubic_kernel(ratio + 1.0, a),
        cubic_kernel(ratio, a),
        cubic_kernel(1.0 - ratio, a),
        cubic_kernel(2.0 - ratio, a),
    ]
}

fn cubic_coeffs_antialias(ratio: f32, scale: f32, a: f32) -> Vec<f32> {
    let s = scale.min(1.0);
    let i_start = (-2.0 / s).floor() as i32 + 1;
    let i_end = 2 - i_start;
    let mut coeffs = Vec::with_capacity((i_end - i_start) as usize);
    for i in i_start..i_end {
        let x = s * (i as f32 - ratio);
        coeffs.push(cubic_kernel(x, a));
    }
    let sum: f32 = coeffs.iter().sum();
    if sum > 0.0 {
        for c in &mut coeffs {
            *c /= sum;
        }
    }
    coeffs
}

fn linear_coeffs_antialias(ratio: f32, scale: f32) -> Vec<f32> {
    let s = scale.min(1.0);
    let start = (-1.0 / s).floor() as i32 + 1;
    let footprint = (2 - 2 * start) as usize;
    let mut coeffs = Vec::with_capacity(footprint);
    for i in 0..footprint {
        let idx = start + i as i32;
        let x = (idx as f32 - ratio) * s;
        coeffs.push((1.0 - x.abs()).max(0.0));
    }
    let sum: f32 = coeffs.iter().sum();
    if sum > 0.0 {
        for c in &mut coeffs {
            *c /= sum;
        }
    }
    coeffs
}

fn get_neighbor_idxes(x: f32, n: usize, limit: usize) -> Vec<i64> {
    let mut idxes: Vec<i64> = (0..limit as i64).collect();
    idxes.sort_by(|&a, &b| {
        let da = (x - a as f32).abs();
        let db = (x - b as f32).abs();
        da.partial_cmp(&db).unwrap().then(a.cmp(&b))
    });
    idxes.truncate(n);
    idxes.sort();
    idxes
}

fn get_neighbor(x: f32, n: usize, data: &[f32]) -> (Vec<i64>, Vec<f32>) {
    let pad_width = ((n as f32) / 2.0).ceil() as usize;
    let padded_len = data.len() + 2 * pad_width;
    let mut padded = vec![0.0f32; padded_len];
    for p in &mut padded[..pad_width] {
        *p = data[0];
    }
    for (i, &v) in data.iter().enumerate() {
        padded[pad_width + i] = v;
    }
    for p in &mut padded[pad_width + data.len()..] {
        *p = data[data.len() - 1];
    }

    let x_padded = x + pad_width as f32;
    let idxes = get_neighbor_idxes(x_padded, n, padded_len);
    let values: Vec<f32> = idxes.iter().map(|&i| padded[i as usize]).collect();
    let original_idxes: Vec<i64> = idxes.iter().map(|&i| i - pad_width as i64).collect();
    (original_idxes, values)
}

fn get_original_coordinate(
    x: f32,
    scale_factor: f32,
    input_width: usize,
    output_width_int: usize,
    roi: &[f32],
    coord_transform: ResizeCoordTransform,
) -> f32 {
    let output_width = scale_factor * input_width as f32;
    match coord_transform {
        ResizeCoordTransform::HalfPixel => (x + 0.5) / scale_factor - 0.5,
        ResizeCoordTransform::HalfPixelSymmetric => {
            let adjustment = output_width_int as f32 / output_width;
            let center = input_width as f32 / 2.0;
            let offset = center * (1.0 - adjustment);
            offset + (x + 0.5) / scale_factor - 0.5
        }
        ResizeCoordTransform::PytorchHalfPixel => {
            if output_width_int > 1 {
                (x + 0.5) / scale_factor - 0.5
            } else {
                -0.5
            }
        }
        ResizeCoordTransform::AlignCorners => {
            if output_width <= 1.0 {
                0.0
            } else {
                x * (input_width as f32 - 1.0) / (output_width - 1.0)
            }
        }
        ResizeCoordTransform::Asymmetric => x / scale_factor,
        ResizeCoordTransform::TFCropAndResize => {
            if roi.len() < 2 {
                0.0
            } else if output_width <= 1.0 {
                (roi[1] - roi[0]) * (input_width as f32 - 1.0) / 2.0
            } else {
                let x_ori =
                    x * (roi[1] - roi[0]) * (input_width as f32 - 1.0) / (output_width - 1.0);
                x_ori + roi[0] * (input_width as f32 - 1.0)
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn interpolate_1d_with_x(
    data: &[f32],
    scale_factor: f32,
    output_width_int: usize,
    x: f32,
    mode: ResizeMode,
    coord_transform: ResizeCoordTransform,
    nearest_mode: ResizeNearestMode,
    cubic_coeff_a: f32,
    antialias: bool,
    exclude_outside: bool,
    extrapolation_value: f32,
    roi: &[f32],
) -> f32 {
    let input_width = data.len();
    let x_ori = get_original_coordinate(
        x,
        scale_factor,
        input_width,
        output_width_int,
        roi,
        coord_transform,
    );

    if matches!(coord_transform, ResizeCoordTransform::TFCropAndResize)
        && (x_ori < 0.0 || x_ori > (input_width as f32 - 1.0))
    {
        return extrapolation_value;
    }

    match mode {
        ResizeMode::Nearest => {
            let nearest_idx = match nearest_mode {
                ResizeNearestMode::RoundPreferFloor => {
                    if x_ori == x_ori.floor() + 0.5 {
                        x_ori.floor() as i64
                    } else {
                        (x_ori + 0.5).floor() as i64
                    }
                }
                ResizeNearestMode::RoundPreferCeil => x_ori.round() as i64,
                ResizeNearestMode::Floor => x_ori.floor() as i64,
                ResizeNearestMode::Ceil => x_ori.ceil() as i64,
            };
            let clamped = nearest_idx.max(0).min(input_width as i64 - 1) as usize;
            data[clamped]
        }
        ResizeMode::Linear => {
            let ratio = if (x_ori - x_ori.floor()).abs() < f32::EPSILON {
                1.0f32
            } else {
                x_ori - x_ori.floor()
            };

            if antialias {
                let mut coeffs = linear_coeffs_antialias(ratio, scale_factor);
                let n = coeffs.len();
                let (idxes, points) = get_neighbor(x_ori, n, data);

                if exclude_outside {
                    for (i, &idx) in idxes.iter().enumerate() {
                        if idx < 0 || idx >= input_width as i64 {
                            coeffs[i] = 0.0;
                        }
                    }
                    let sum: f32 = coeffs.iter().sum();
                    if sum > 0.0 {
                        for c in &mut coeffs {
                            *c /= sum;
                        }
                    }
                }

                coeffs.iter().zip(points.iter()).map(|(c, p)| c * p).sum()
            } else {
                let mut coeffs = vec![1.0 - ratio, ratio];
                let (idxes, points) = get_neighbor(x_ori, 2, data);

                if exclude_outside {
                    for (i, &idx) in idxes.iter().enumerate() {
                        if idx < 0 || idx >= input_width as i64 {
                            coeffs[i] = 0.0;
                        }
                    }
                    let sum: f32 = coeffs.iter().sum();
                    if sum > 0.0 {
                        for c in &mut coeffs {
                            *c /= sum;
                        }
                    }
                }

                coeffs.iter().zip(points.iter()).map(|(c, p)| c * p).sum()
            }
        }
        ResizeMode::Cubic => {
            let ratio = if (x_ori - x_ori.floor()).abs() < f32::EPSILON {
                1.0f32
            } else {
                x_ori - x_ori.floor()
            };

            if antialias {
                let mut coeffs = cubic_coeffs_antialias(ratio, scale_factor, cubic_coeff_a);
                let n = coeffs.len();
                let (idxes, points) = get_neighbor(x_ori, n, data);

                if exclude_outside {
                    for (i, &idx) in idxes.iter().enumerate() {
                        if idx < 0 || idx >= input_width as i64 {
                            coeffs[i] = 0.0;
                        }
                    }
                    let sum: f32 = coeffs.iter().sum();
                    if sum > 0.0 {
                        for c in &mut coeffs {
                            *c /= sum;
                        }
                    }
                }

                coeffs.iter().zip(points.iter()).map(|(c, p)| c * p).sum()
            } else {
                let coeffs = cubic_coeffs(ratio, cubic_coeff_a);
                let (idxes, points) = get_neighbor(x_ori, 4, data);

                let mut coeffs_adj = coeffs.to_vec();
                if exclude_outside {
                    for (i, &idx) in idxes.iter().enumerate() {
                        if idx < 0 || idx >= input_width as i64 {
                            coeffs_adj[i] = 0.0;
                        }
                    }
                    let sum: f32 = coeffs_adj.iter().sum();
                    if sum > 0.0 {
                        for c in &mut coeffs_adj {
                            *c /= sum;
                        }
                    }
                }

                coeffs_adj
                    .iter()
                    .zip(points.iter())
                    .map(|(c, p)| c * p)
                    .sum()
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn interpolate_nd(
    data: &[f32],
    input_shape: &[usize],
    output_shape: &[usize],
    scale_factors: &[f32],
    output_coords: &[usize],
    mode: ResizeMode,
    coord_transform: ResizeCoordTransform,
    nearest_mode: ResizeNearestMode,
    cubic_coeff_a: f32,
    antialias: bool,
    exclude_outside: bool,
    extrapolation_value: f32,
    roi: &[f32],
) -> f32 {
    let n = input_shape.len();
    if n == 1 {
        let dim_roi_pair: Vec<f32> = if roi.len() >= 2 {
            vec![roi[0], roi[1]]
        } else {
            vec![0.0, 1.0]
        };
        return interpolate_1d_with_x(
            data,
            scale_factors[0],
            output_shape[0],
            output_coords[0] as f32,
            mode,
            coord_transform,
            nearest_mode,
            cubic_coeff_a,
            antialias,
            exclude_outside,
            extrapolation_value,
            &dim_roi_pair,
        );
    }

    let inner_size: usize = input_shape[1..].iter().product();
    let mut res1d = Vec::with_capacity(input_shape[0]);

    for i in 0..input_shape[0] {
        let slice_start = i * inner_size;
        let slice_end = slice_start + inner_size;
        let inner_data = &data[slice_start..slice_end];

        let inner_roi = if roi.len() >= 2 * n {
            let mut ir = Vec::with_capacity(2 * (n - 1));
            ir.extend_from_slice(&roi[1..n]);
            ir.extend_from_slice(&roi[n + 1..2 * n]);
            ir
        } else {
            vec![]
        };

        let val = interpolate_nd(
            inner_data,
            &input_shape[1..],
            &output_shape[1..],
            &scale_factors[1..],
            &output_coords[1..],
            mode,
            coord_transform,
            nearest_mode,
            cubic_coeff_a,
            antialias,
            exclude_outside,
            extrapolation_value,
            &inner_roi,
        );
        res1d.push(val);
    }

    let dim_roi = if roi.len() >= 2 * n {
        vec![roi[0], roi[n]]
    } else {
        vec![0.0, 1.0]
    };

    interpolate_1d_with_x(
        &res1d,
        scale_factors[0],
        output_shape[0],
        output_coords[0] as f32,
        mode,
        coord_transform,
        nearest_mode,
        cubic_coeff_a,
        antialias,
        exclude_outside,
        extrapolation_value,
        &dim_roi,
    )
}

// =============================================================================
// Shape / scale computation
// =============================================================================

fn compute_output_shape(
    input_shape: &[u64],
    scales: Option<&[f32]>,
    sizes: Option<&[i64]>,
    axes: &[i64],
    keep_aspect_ratio_policy: ResizeKeepAspectRatioPolicy,
) -> Vec<usize> {
    let rank = input_shape.len();
    let mut output_shape: Vec<usize> = input_shape.iter().map(|&x| x as usize).collect();

    let resolved_axes: Vec<usize> = if axes.is_empty() {
        (0..rank).collect()
    } else {
        axes.iter()
            .map(|&a| if a < 0 { (rank as i64 + a) as usize } else { a as usize })
            .collect()
    };

    if let Some(sizes) = sizes {
        match keep_aspect_ratio_policy {
            ResizeKeepAspectRatioPolicy::Stretch => {
                for (i, &axis) in resolved_axes.iter().enumerate() {
                    output_shape[axis] = sizes[i] as usize;
                }
            }
            ResizeKeepAspectRatioPolicy::NotLarger => {
                let min_scale = resolved_axes
                    .iter()
                    .enumerate()
                    .map(|(i, &axis)| sizes[i] as f32 / input_shape[axis] as f32)
                    .fold(f32::INFINITY, f32::min);
                for &axis in &resolved_axes {
                    output_shape[axis] = round_half_up(min_scale * input_shape[axis] as f32);
                }
            }
            ResizeKeepAspectRatioPolicy::NotSmaller => {
                let max_scale = resolved_axes
                    .iter()
                    .enumerate()
                    .map(|(i, &axis)| sizes[i] as f32 / input_shape[axis] as f32)
                    .fold(0.0f32, f32::max);
                for &axis in &resolved_axes {
                    output_shape[axis] = round_half_up(max_scale * input_shape[axis] as f32);
                }
            }
        }
    } else if let Some(scales) = scales {
        for (i, &axis) in resolved_axes.iter().enumerate() {
            output_shape[axis] = (input_shape[axis] as f32 * scales[i]).floor() as usize;
        }
    }

    output_shape
}

fn round_half_up(x: f32) -> usize {
    (x + 0.5).floor() as usize
}

fn compute_scales(
    input_shape: &[u64],
    output_shape: &[usize],
    provided_scales: Option<&[f32]>,
    axes: &[i64],
    keep_aspect_ratio_policy: ResizeKeepAspectRatioPolicy,
) -> Vec<f32> {
    let rank = input_shape.len();
    let resolved_axes: Vec<usize> = if axes.is_empty() {
        (0..rank).collect()
    } else {
        axes.iter()
            .map(|&a| if a < 0 { (rank as i64 + a) as usize } else { a as usize })
            .collect()
    };

    let mut scales = vec![1.0f32; rank];

    if let Some(provided) = provided_scales {
        for (i, &axis) in resolved_axes.iter().enumerate() {
            scales[axis] = provided[i];
        }
        if !matches!(keep_aspect_ratio_policy, ResizeKeepAspectRatioPolicy::Stretch) {
            for &axis in &resolved_axes {
                scales[axis] = output_shape[axis] as f32 / input_shape[axis] as f32;
            }
        }
    } else {
        for &axis in &resolved_axes {
            scales[axis] = output_shape[axis] as f32 / input_shape[axis] as f32;
        }
    }

    scales
}

// =============================================================================
// eval
// =============================================================================

impl MilliOp for Resize {
    fn eval(
        &self,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
        backend: &mut EvalBackend,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, MilliOpGraphError>
    {
        let input = &inputs[&self.input];
        let input_shape = input.shape();
        let original_dtype = input.dtype();
        let rank = input_shape.len();

        let input_f32 = input.cast(DType::F32, backend)?;
        let input_flat: Vec<f32> = input_f32.flatten()?.try_into()?;

        // Get ROI
        let roi_raw: Vec<f32> = if let Some(roi_id) = self.roi {
            if let Some(roi_tensor) = inputs.get(&roi_id) {
                let roi_f32 = roi_tensor.cast(DType::F32, backend)?;
                roi_f32.try_to_rank::<P1>()?.try_into()?
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        let full_roi = if !self.axes.is_empty() && !roi_raw.is_empty() {
            let resolved_axes: Vec<usize> = self.axes.iter()
                .map(|&a| if a < 0 { (rank as i64 + a) as usize } else { a as usize })
                .collect();
            let num_axes = resolved_axes.len();
            let mut full = vec![0.0f32; 2 * rank];
            for d in 0..rank {
                full[d + rank] = 1.0;
            }
            for (i, &axis) in resolved_axes.iter().enumerate() {
                full[axis] = roi_raw[i];
                full[axis + rank] = roi_raw[i + num_axes];
            }
            full
        } else {
            roi_raw
        };

        // Get scales
        let scales_vec: Option<Vec<f32>> = if let Some(scales_id) = self.scales {
            if let Some(scales_tensor) = inputs.get(&scales_id) {
                let s: Vec<f32> = scales_tensor
                    .cast(DType::F32, backend)?
                    .try_to_rank::<P1>()?
                    .try_into()?;
                if s.iter().all(|&x| x == 0.0) { None } else { Some(s) }
            } else {
                None
            }
        } else {
            None
        };

        // Get sizes
        let sizes_vec: Option<Vec<i64>> = if let Some(sizes_id) = self.sizes {
            if let Some(sizes_tensor) = inputs.get(&sizes_id) {
                Some(
                    sizes_tensor
                        .cast(DType::I64, backend)?
                        .try_to_rank::<P1>()?
                        .try_into()?,
                )
            } else {
                None
            }
        } else {
            None
        };

        let output_shape = compute_output_shape(
            &input_shape,
            scales_vec.as_deref(),
            sizes_vec.as_deref(),
            &self.axes,
            self.keep_aspect_ratio_policy,
        );

        let scales = compute_scales(
            &input_shape,
            &output_shape,
            scales_vec.as_deref(),
            &self.axes,
            self.keep_aspect_ratio_policy,
        );

        // Try NCHW fast paths: rank==4, only H/W dimensions change, no antialias,
        // no exclude_outside, no TFCropAndResize ROI
        let use_fast_path = rank == 4
            && output_shape[0] == input_shape[0] as usize
            && output_shape[1] == input_shape[1] as usize
            && !self.antialias
            && !self.exclude_outside
            && !matches!(self.coord_transform, ResizeCoordTransform::TFCropAndResize);

        let output_data = if use_fast_path {
            let n = input_shape[0] as usize;
            let c = input_shape[1] as usize;
            let in_h = input_shape[2] as usize;
            let in_w = input_shape[3] as usize;
            let out_h = output_shape[2];
            let out_w = output_shape[3];

            match self.mode {
                ResizeMode::Nearest => resize_nchw_nearest(
                    &input_flat, n, c, in_h, in_w, out_h, out_w,
                    scales[2], scales[3],
                    self.coord_transform, self.nearest_mode,
                ),
                ResizeMode::Linear => resize_nchw_linear(
                    &input_flat, n, c, in_h, in_w, out_h, out_w,
                    scales[2], scales[3],
                    self.coord_transform,
                ),
                ResizeMode::Cubic => {
                    // Fall through to generic for cubic
                    resize_generic(
                        &input_flat, &input_shape, &output_shape, &scales,
                        &full_roi, self.mode, self.coord_transform, self.nearest_mode,
                        self.cubic_coeff_a, self.antialias, self.exclude_outside,
                        self.extrapolation_value,
                    )
                }
            }
        } else {
            resize_generic(
                &input_flat, &input_shape, &output_shape, &scales,
                &full_roi, self.mode, self.coord_transform, self.nearest_mode,
                self.cubic_coeff_a, self.antialias, self.exclude_outside,
                self.extrapolation_value,
            )
        };

        let output_tensor: NumericTensor<DynRank> =
            NumericTensor::from_vec_shape(output_data, output_shape)?;
        let output_tensor = output_tensor.cast(original_dtype, backend)?;

        Ok(Box::new(std::iter::once((self.output, output_tensor))))
    }
}

/// Generic fallback using recursive separable interpolation with rayon.
fn resize_generic(
    input_flat: &[f32],
    input_shape: &[u64],
    output_shape: &[usize],
    scales: &[f32],
    full_roi: &[f32],
    mode: ResizeMode,
    coord_transform: ResizeCoordTransform,
    nearest_mode: ResizeNearestMode,
    cubic_coeff_a: f32,
    antialias: bool,
    exclude_outside: bool,
    extrapolation_value: f32,
) -> Vec<f32> {
    let rank = input_shape.len();
    let input_shape_usize: Vec<usize> = input_shape.iter().map(|&x| x as usize).collect();

    let output_numel: usize = output_shape.iter().product();
    let mut output_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
    }

    let output_data: Vec<f32> = (0..output_numel)
        .into_par_iter()
        .map(|out_idx| {
            let mut out_coords = vec![0usize; rank];
            let mut remaining = out_idx;
            for d in 0..rank {
                out_coords[d] = remaining / output_strides[d];
                remaining %= output_strides[d];
            }

            interpolate_nd(
                input_flat,
                &input_shape_usize,
                output_shape,
                scales,
                &out_coords,
                mode,
                coord_transform,
                nearest_mode,
                cubic_coeff_a,
                antialias,
                exclude_outside,
                extrapolation_value,
                full_roi,
            )
        })
        .collect();

    output_data
}
