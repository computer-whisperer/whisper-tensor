use crate::DynRank;
use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::graph::GlobalId;
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::MilliOpGraphError;
use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
use crate::numeric_tensor::NumericTensor;
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
// ONNX-spec-compliant resize implementation (separable, 1D-at-a-time)
// =============================================================================

/// Standard cubic interpolation kernel (Keys' convolution)
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

/// Compute cubic coefficients for a given ratio (ONNX convention).
/// `ratio` is in (0, 1] where integer coords get ratio=1.
/// Returns 4 coefficients for taps at [-1, 0, 1, 2] relative to floor(x_ori).
fn cubic_coeffs(ratio: f32, a: f32) -> [f32; 4] {
    [
        cubic_kernel(ratio + 1.0, a),
        cubic_kernel(ratio, a),
        cubic_kernel(1.0 - ratio, a),
        cubic_kernel(2.0 - ratio, a),
    ]
}

/// Compute cubic coefficients with antialias scaling (ONNX reference-compatible).
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

/// Compute linear coefficients with antialias scaling (ONNX reference-compatible).
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

/// Get n nearest neighbor indices to x, preferring smaller indices.
/// Returns indices that may be < 0 or >= limit (for edge padding).
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

/// Get n neighbors of x with edge-padding, returning (original_indices, values).
fn get_neighbor(x: f32, n: usize, data: &[f32]) -> (Vec<i64>, Vec<f32>) {
    let pad_width = ((n as f32) / 2.0).ceil() as usize;
    // Create edge-padded data
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

/// Transform output coordinate to input coordinate.
/// Uses `output_width` (float = scale * input_width) for align_corners, etc.
fn get_original_coordinate(
    x: f32,
    scale_factor: f32,
    input_width: usize,
    output_width_int: usize,
    roi: &[f32], // [roi_start, roi_end] for this dimension
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

/// 1D interpolation at a single output coordinate.
/// `data` is a 1D slice of f32 values.
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
    roi: &[f32], // [roi_start, roi_end]
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

    // TF crop and resize: out-of-bounds returns extrapolation value
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

/// Recursively perform N-dimensional separable interpolation.
/// `data` is a flat array with shape `input_shape`.
/// This processes dimension 0 first, then recurses on the remaining dimensions.
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
    roi: &[f32], // full ROI: [start_d0, start_d1, ..., end_d0, end_d1, ...]
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

    // Build 1D slices along dimension 0 by recursing on remaining dimensions
    let inner_size: usize = input_shape[1..].iter().product();
    let mut res1d = Vec::with_capacity(input_shape[0]);

    for i in 0..input_shape[0] {
        let slice_start = i * inner_size;
        let slice_end = slice_start + inner_size;
        let inner_data = &data[slice_start..slice_end];

        // ROI for inner dimensions: remove dimension 0's entries
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

    // Now interpolate along dimension 0
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

/// Compute output shape from scales or sizes.
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
            .map(|&a| {
                if a < 0 {
                    (rank as i64 + a) as usize
                } else {
                    a as usize
                }
            })
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

/// Compute per-axis scale factors.
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
            .map(|&a| {
                if a < 0 {
                    (rank as i64 + a) as usize
                } else {
                    a as usize
                }
            })
            .collect()
    };

    let mut scales = vec![1.0f32; rank];

    if let Some(provided) = provided_scales {
        for (i, &axis) in resolved_axes.iter().enumerate() {
            scales[axis] = provided[i];
        }
        // For keep_aspect_ratio_policy with sizes, recompute scales from output/input
        if !matches!(
            keep_aspect_ratio_policy,
            ResizeKeepAspectRatioPolicy::Stretch
        ) {
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

        // Cast input to f32 and flatten for computation
        let input_f32 = input.cast(DType::F32, backend)?;
        let input_flat: Vec<f32> = input_f32.flatten()?.try_into()?;

        // Get ROI if provided
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

        // Expand ROI from axes-relative to full-rank
        let full_roi = if !self.axes.is_empty() && !roi_raw.is_empty() {
            let resolved_axes: Vec<usize> = self
                .axes
                .iter()
                .map(|&a| {
                    if a < 0 {
                        (rank as i64 + a) as usize
                    } else {
                        a as usize
                    }
                })
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

        // Get scales if provided
        let scales_vec: Option<Vec<f32>> = if let Some(scales_id) = self.scales {
            if let Some(scales_tensor) = inputs.get(&scales_id) {
                let s: Vec<f32> = scales_tensor
                    .cast(DType::F32, backend)?
                    .try_to_rank::<P1>()?
                    .try_into()?;
                if s.iter().all(|&x| x == 0.0) {
                    None
                } else {
                    Some(s)
                }
            } else {
                None
            }
        } else {
            None
        };

        // Get sizes if provided
        let sizes_vec: Option<Vec<i64>> = if let Some(sizes_id) = self.sizes {
            if let Some(sizes_tensor) = inputs.get(&sizes_id) {
                let s: Vec<i64> = sizes_tensor
                    .cast(DType::I64, backend)?
                    .try_to_rank::<P1>()?
                    .try_into()?;
                Some(s)
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

        let input_shape_usize: Vec<usize> = input_shape.iter().map(|&x| x as usize).collect();

        // Compute output
        let output_numel: usize = output_shape.iter().product();
        let mut output_data = vec![0.0f32; output_numel];

        // Compute output strides
        let mut output_strides = vec![1usize; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
        }

        for (out_idx, output_val) in output_data.iter_mut().enumerate() {
            let mut out_coords = vec![0usize; rank];
            let mut remaining = out_idx;
            for d in 0..rank {
                out_coords[d] = remaining / output_strides[d];
                remaining %= output_strides[d];
            }

            let val = interpolate_nd(
                &input_flat,
                &input_shape_usize,
                &output_shape,
                &scales,
                &out_coords,
                self.mode,
                self.coord_transform,
                self.nearest_mode,
                self.cubic_coeff_a,
                self.antialias,
                self.exclude_outside,
                self.extrapolation_value,
                &full_roi,
            );

            *output_val = val;
        }

        let output_tensor: NumericTensor<DynRank> =
            NumericTensor::from_vec_shape(output_data, output_shape)?;
        let output_tensor = output_tensor.cast(original_dtype, backend)?;

        Ok(Box::new(std::iter::once((self.output, output_tensor))))
    }
}
