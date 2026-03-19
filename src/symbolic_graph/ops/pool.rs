use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::dtype::DType;
use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::ops as milli_ops;
use crate::milli_graph::{MilliLoweringContext, MilliOpGraph, ops_helpers};
use crate::onnx;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{
    ONNXDecodingError, query_attribute_int, query_attribute_ints, query_attribute_string,
};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) enum PoolAutoPad {
    NotSet,
    SameUpper,
    SameLower,
    Valid,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AveragePoolOperation {
    global_id: GlobalId,
    input: GlobalId,
    output: GlobalId,
    kernel_shape: Vec<i64>,
    strides: Vec<i64>,
    pads: Vec<i64>,
    dilations: Vec<i64>,
    auto_pad: PoolAutoPad,
    count_include_pad: bool,
    ceil_mode: bool,
}

impl AveragePoolOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("AveragePool"));
        }
        if outputs.is_empty() {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("AveragePool"));
        }

        let auto_pad_str = query_attribute_string(attributes, "auto_pad");
        let auto_pad = match auto_pad_str {
            Some(x) => match x.to_lowercase().as_str() {
                "notset" => PoolAutoPad::NotSet,
                "same_upper" => PoolAutoPad::SameUpper,
                "same_lower" => PoolAutoPad::SameLower,
                "valid" => PoolAutoPad::Valid,
                _ => PoolAutoPad::NotSet,
            },
            _ => PoolAutoPad::NotSet,
        };

        let kernel_shape = query_attribute_ints(attributes, "kernel_shape").unwrap_or_default();
        let strides = query_attribute_ints(attributes, "strides").unwrap_or_default();
        let pads = query_attribute_ints(attributes, "pads").unwrap_or_default();
        let dilations = query_attribute_ints(attributes, "dilations").unwrap_or_default();
        let count_include_pad =
            query_attribute_int(attributes, "count_include_pad").unwrap_or(0) != 0;
        let ceil_mode = query_attribute_int(attributes, "ceil_mode").unwrap_or(0) != 0;

        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("AveragePool"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("AveragePool"))?,
            kernel_shape,
            strides,
            pads,
            dilations,
            auto_pad,
            count_include_pad,
            ceil_mode,
        })
    }
}

impl Node for AveragePoolOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "AveragePool".to_string()
    }

    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }

    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for AveragePoolOperation {
    fn parameters(&self) -> Vec<Property> {
        let mut params = Vec::new();
        let auto_pad_str = match &self.auto_pad {
            PoolAutoPad::NotSet => "NOTSET",
            PoolAutoPad::SameUpper => "SAME_UPPER",
            PoolAutoPad::SameLower => "SAME_LOWER",
            PoolAutoPad::Valid => "VALID",
        };
        params.push(Property::new(
            "auto_pad",
            PropertyValue::String(auto_pad_str.to_string()),
        ));
        if !self.kernel_shape.is_empty() {
            params.push(Property::new(
                "kernel_shape",
                PropertyValue::IntList(self.kernel_shape.clone()),
            ));
        }
        if !self.strides.is_empty() {
            params.push(Property::new(
                "strides",
                PropertyValue::IntList(self.strides.clone()),
            ));
        }
        if !self.pads.is_empty() {
            params.push(Property::new(
                "pads",
                PropertyValue::IntList(self.pads.clone()),
            ));
        }
        if !self.dilations.is_empty() {
            params.push(Property::new(
                "dilations",
                PropertyValue::IntList(self.dilations.clone()),
            ));
        }
        if self.count_include_pad {
            params.push(Property::new("count_include_pad", PropertyValue::Int(1)));
        }
        if self.ceil_mode {
            params.push(Property::new("ceil_mode", PropertyValue::Int(1)));
        }
        params
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let original_input = input_map[&self.input];

        let input_f32 = milli_ops::Cast::push_new(&mut graph, original_input, DType::F32, rng);

        let n_spatial = self.kernel_shape.len();
        let kernel_shape: Vec<i64> = self.kernel_shape.clone();
        let strides: Vec<i64> = if self.strides.is_empty() {
            vec![1; n_spatial]
        } else {
            self.strides.clone()
        };
        let dilations: Vec<i64> = if self.dilations.is_empty() {
            vec![1; n_spatial]
        } else {
            self.dilations.clone()
        };

        let dilated_kernel: Vec<i64> = (0..n_spatial)
            .map(|i| dilations[i] * (kernel_shape[i] - 1) + 1)
            .collect();

        // Compute pad tensor for the Pad op. Format: [N_b=0, C_b=0, spatial_begins..., N_e=0, C_e=0, spatial_ends...]
        let pads_const: Option<GlobalId> = match &self.auto_pad {
            PoolAutoPad::Valid => None,
            PoolAutoPad::NotSet => {
                if self.pads.is_empty() || self.pads.iter().all(|&p| p == 0) {
                    None
                } else {
                    let mut pv = vec![0i64; 2 + n_spatial];
                    pv[2..2 + n_spatial].copy_from_slice(&self.pads[..n_spatial]);
                    pv.extend(std::iter::repeat_n(0i64, 2));
                    pv.extend_from_slice(&self.pads[n_spatial..2 * n_spatial]);
                    Some(milli_ops::Constant::push_new(
                        &mut graph,
                        NDArrayNumericTensor::from(pv).to_dyn(),
                        rng,
                    ))
                }
            }
            auto_pad @ (PoolAutoPad::SameUpper | PoolAutoPad::SameLower) => {
                // Compute padding dynamically from input shape.
                // out_d = ceil(input_d / stride_d)
                // total_pad_d = max(0, (out_d - 1) * stride_d + dilated_kernel_d - input_d)
                let input_shape = milli_ops::Shape::push_new(&mut graph, input_f32, rng);
                let two = ops_helpers::scalar_const(&mut graph, 2i64, rng);
                let n_end = ops_helpers::scalar_const(&mut graph, (2 + n_spatial) as i64, rng);
                let spatial = milli_ops::Slice::push_new(
                    &mut graph, input_shape, two, n_end, None, None, rng,
                );
                let strides_t = milli_ops::Constant::push_new(
                    &mut graph,
                    NDArrayNumericTensor::from(strides.clone()).to_dyn(),
                    rng,
                );
                let dk_t = milli_ops::Constant::push_new(
                    &mut graph,
                    NDArrayNumericTensor::from(dilated_kernel.clone()).to_dyn(),
                    rng,
                );
                let one = ops_helpers::scalar_const(&mut graph, 1i64, rng);
                let zero = ops_helpers::scalar_const(&mut graph, 0i64, rng);

                // out_d = ceil(spatial / strides) = (spatial + strides - 1) / strides
                let s_m1 = milli_ops::SimpleBinary::sub(&mut graph, strides_t, one, rng);
                let numer = milli_ops::SimpleBinary::add(&mut graph, spatial, s_m1, rng);
                let out_d = milli_ops::SimpleBinary::div(&mut graph, numer, strides_t, rng);
                // total_pad = (out_d - 1) * strides + dk - spatial
                let out_m1 = milli_ops::SimpleBinary::sub(&mut graph, out_d, one, rng);
                let t1 = milli_ops::SimpleBinary::mul(&mut graph, out_m1, strides_t, rng);
                let t2 = milli_ops::SimpleBinary::add(&mut graph, t1, dk_t, rng);
                let total_pad = milli_ops::SimpleBinary::sub(&mut graph, t2, spatial, rng);
                // clamp to 0
                let total_pad = milli_ops::SimpleBinary::max(&mut graph, total_pad, zero, rng);

                let half = ops_helpers::scalar_const(&mut graph, 2i64, rng);
                let half_pad = milli_ops::SimpleBinary::div(&mut graph, total_pad, half, rng);
                let other_half =
                    milli_ops::SimpleBinary::sub(&mut graph, total_pad, half_pad, rng);

                let (pad_begin, pad_end) = if matches!(auto_pad, PoolAutoPad::SameUpper) {
                    (half_pad, other_half)
                } else {
                    (other_half, half_pad)
                };

                // Build [0, 0, pad_begin..., 0, 0, pad_end...]
                let zeros_2 = milli_ops::Constant::push_new(
                    &mut graph,
                    NDArrayNumericTensor::from(vec![0i64, 0]).to_dyn(),
                    rng,
                );
                let pads = milli_ops::Concat::push_new(
                    &mut graph,
                    vec![zeros_2, pad_begin, zeros_2, pad_end],
                    0,
                    rng,
                );
                Some(pads)
            }
        };

        let has_padding = pads_const.is_some();

        let padded = if let Some(pc) = pads_const {
            let zero_val = ops_helpers::scalar_const(&mut graph, 0.0f32, rng);
            milli_ops::Pad::push_new(
                &mut graph, input_f32, pc, Some(zero_val), None,
                milli_ops::PadMode::Constant, rng,
            )
        } else {
            input_f32
        };

        // For count_include_pad=false with padding, we need a parallel ones tensor
        // padded with zeros to track how many valid (non-pad) elements contribute
        // to each output position.
        // We need a per-position count of valid taps when:
        // - There is padding and count_include_pad=false (padded zeros shouldn't count)
        // - ceil_mode is true (boundary windows may have out-of-bounds taps)
        let need_count_tensor =
            (has_padding && !self.count_include_pad) || self.ceil_mode;
        let ones_padded = if need_count_tensor {
            let zero_c = ops_helpers::scalar_const(&mut graph, 0.0f32, rng);
            let one_c = ops_helpers::scalar_const(&mut graph, 1.0f32, rng);
            if self.count_include_pad || !has_padding {
                // count_include_pad=true: padded positions count as 1.
                // Or no padding: all positions valid. Either way, ones_of(padded shape).
                let ones = milli_ops::SimpleBinary::mul(&mut graph, padded, zero_c, rng);
                Some(milli_ops::SimpleBinary::add(&mut graph, ones, one_c, rng))
            } else {
                // count_include_pad=false: padded positions count as 0.
                // Build ones of input shape, then pad with 0.
                let ones = milli_ops::SimpleBinary::mul(&mut graph, input_f32, zero_c, rng);
                let ones = milli_ops::SimpleBinary::add(&mut graph, ones, one_c, rng);
                let pc = pads_const.unwrap();
                let zero_val = ops_helpers::scalar_const(&mut graph, 0.0f32, rng);
                Some(milli_ops::Pad::push_new(
                    &mut graph, ones, pc, Some(zero_val), None,
                    milli_ops::PadMode::Constant, rng,
                ))
            }
        } else {
            None
        };

        // Compute output spatial size dynamically from padded input shape.
        // padded_shape = Shape(padded) => [N, C, S0, S1, ...]
        // For each spatial dim d: out_d = floor((S_d - dilated_kernel_d) / stride_d) + 1
        //   (or ceil for ceil_mode)
        // We compute end_d for each tap: end_d = start_d + out_d * stride_d
        // This ensures all taps produce the same output shape.
        let padded_shape = milli_ops::Shape::push_new(&mut graph, padded, rng);

        // Extract spatial dims from shape: Slice(shape, [2], [2+n_spatial])
        let spatial_shape = {
            let s = ops_helpers::scalar_const(&mut graph, 2i64, rng);
            let e = ops_helpers::scalar_const(&mut graph, (2 + n_spatial) as i64, rng);
            milli_ops::Slice::push_new(&mut graph, padded_shape, s, e, None, None, rng)

        };

        // out_spatial = (spatial_shape - dilated_kernel) / strides + 1
        let dk_const = milli_ops::Constant::push_new(
            &mut graph,
            NDArrayNumericTensor::from(dilated_kernel).to_dyn(),
            rng,
        );
        let strides_const = milli_ops::Constant::push_new(
            &mut graph,
            NDArrayNumericTensor::from(strides.clone()).to_dyn(),
            rng,
        );
        let one_const = ops_helpers::scalar_const(&mut graph, 1i64, rng);

        // numerator = spatial_shape - dilated_kernel
        let numer = milli_ops::SimpleBinary::sub(&mut graph, spatial_shape, dk_const, rng);
        // out_spatial = numer / strides + 1
        // For ceil_mode: out_spatial = ceil(numer / strides) + 1
        //   = (numer + strides - 1) / strides + 1
        let out_spatial = if self.ceil_mode {
            let strides_m1 =
                milli_ops::SimpleBinary::sub(&mut graph, strides_const, one_const, rng);
            let numer_ceil =
                milli_ops::SimpleBinary::add(&mut graph, numer, strides_m1, rng);
            let ceil_div = milli_ops::SimpleBinary::div(&mut graph, numer_ceil, strides_const, rng);
            let ceil_out = milli_ops::SimpleBinary::add(&mut graph, ceil_div, one_const, rng);
            // ONNX spec: drop last window if its start >= (input_spatial + pad_begin).
            // i.e., it starts entirely in the right-side padding.
            // max_out = floor((input_spatial + pad_begin - 1) / stride) + 1
            // input_spatial + pad_begin = padded_spatial - pad_end.
            // For simplicity, use input shape directly:
            let input_shape_full = milli_ops::Shape::push_new(&mut graph, input_f32, rng);
            let two2 = ops_helpers::scalar_const(&mut graph, 2i64, rng);
            let n_end2 = ops_helpers::scalar_const(&mut graph, (2 + n_spatial) as i64, rng);
            let input_spatial = milli_ops::Slice::push_new(
                &mut graph, input_shape_full, two2, n_end2, None, None, rng,
            );
            // threshold = input_spatial + pad_begin
            // pad_begin: for NOTSET, it's pads[0..n_spatial]; for SAME_*, we compute from
            // the padded shape. Simplest: threshold = padded_spatial - pad_end.
            // But for NOTSET: pad_end = pads[n_spatial..]. For SAME_*: pad_end = total_pad - pad_begin.
            // Easier: threshold = padded_spatial - pad_end = input_spatial + pad_begin.
            // We can get this as: padded_spatial - (padded_spatial - input_spatial - pad_begin)
            // ... or just: spatial_shape - (spatial_shape - input_spatial) + ... this is circular.
            // Simplest: pad_begin is the first n_spatial values from the pads tensor.
            // For NOTSET:
            let pad_begin_spatial = match &self.auto_pad {
                PoolAutoPad::NotSet => {
                    let pb: Vec<i64> = (0..n_spatial).map(|i| {
                        if i < self.pads.len() { self.pads[i] } else { 0 }
                    }).collect();
                    milli_ops::Constant::push_new(
                        &mut graph,
                        NDArrayNumericTensor::from(pb).to_dyn(),
                        rng,
                    )
                }
                _ => {
                    // For SAME_*: pad_begin = (padded - input) spatial dims, but only the begin half.
                    // We stored pads_const earlier. Extract first n_spatial spatial values from it
                    // (indices 2..2+n_spatial of the full pad tensor).
                    // But we don't have it here. Compute from shapes:
                    // pad_begin + pad_end = padded - input
                    // For SAME_UPPER: pad_begin = total/2
                    // For SAME_LOWER: pad_begin = total - total/2
                    let total_pad = milli_ops::SimpleBinary::sub(
                        &mut graph, spatial_shape, input_spatial, rng,
                    );
                    let two_c = ops_helpers::scalar_const(&mut graph, 2i64, rng);
                    let half = milli_ops::SimpleBinary::div(&mut graph, total_pad, two_c, rng);
                    if matches!(self.auto_pad, PoolAutoPad::SameUpper) {
                        half
                    } else {
                        milli_ops::SimpleBinary::sub(&mut graph, total_pad, half, rng)
                    }
                }
            };
            let threshold = milli_ops::SimpleBinary::add(
                &mut graph, input_spatial, pad_begin_spatial, rng,
            );
            let threshold_m1 = milli_ops::SimpleBinary::sub(
                &mut graph, threshold, one_const, rng,
            );
            let max_div = milli_ops::SimpleBinary::div(&mut graph, threshold_m1, strides_const, rng);
            let max_out = milli_ops::SimpleBinary::add(&mut graph, max_div, one_const, rng);
            milli_ops::SimpleBinary::min(&mut graph, ceil_out, max_out, rng)
        } else {
            let div = milli_ops::SimpleBinary::div(&mut graph, numer, strides_const, rng);
            milli_ops::SimpleBinary::add(&mut graph, div, one_const, rng)
        };

        // Ensure the padded tensor is large enough for all taps.
        // The last kernel tap starts at max_start_d = (kernel_d-1)*dilation_d.
        // It needs (out_d-1)*stride_d elements beyond that.
        // Required padded size = max_start + (out_d-1)*stride + 1
        // Extra trailing pad = required - actual_padded_size, clamped to >= 0
        let max_start: Vec<i64> = (0..n_spatial)
            .map(|i| (kernel_shape[i] - 1) * dilations[i])
            .collect();
        let max_start_const = milli_ops::Constant::push_new(
            &mut graph,
            NDArrayNumericTensor::from(max_start).to_dyn(),
            rng,
        );
        let out_m1 = milli_ops::SimpleBinary::sub(&mut graph, out_spatial, one_const, rng);
        let out_m1_times_s = milli_ops::SimpleBinary::mul(&mut graph, out_m1, strides_const, rng);
        let required = milli_ops::SimpleBinary::add(&mut graph, max_start_const, out_m1_times_s, rng);
        let required_plus1 = milli_ops::SimpleBinary::add(&mut graph, required, one_const, rng);
        let extra_pad = milli_ops::SimpleBinary::sub(&mut graph, required_plus1, spatial_shape, rng);
        let zero_const = ops_helpers::scalar_const(&mut graph, 0i64, rng);
        let extra_pad = milli_ops::SimpleBinary::max(&mut graph, extra_pad, zero_const, rng);

        // Build pad tensor [0,0,0,...0, 0,0,extra_d0,extra_d1,...] for trailing pad only
        let zeros_batch_ch = milli_ops::Constant::push_new(
            &mut graph,
            NDArrayNumericTensor::from(vec![0i64; 2 + n_spatial]).to_dyn(),
            rng,
        );
        let zeros_bc2 = milli_ops::Constant::push_new(
            &mut graph,
            NDArrayNumericTensor::from(vec![0i64; 2]).to_dyn(),
            rng,
        );
        let extra_pads = milli_ops::Concat::push_new(
            &mut graph,
            vec![zeros_batch_ch, zeros_bc2, extra_pad],
            0,
            rng,
        );

        // Only apply if extra_pad > 0 anywhere. Since we can't branch in a milli graph,
        // just always Pad — zero-padding with all-zero pads is a no-op for the values.
        let padded = {
            let zero_val = ops_helpers::scalar_const(&mut graph, 0.0f32, rng);
            milli_ops::Pad::push_new(
                &mut graph, padded, extra_pads, Some(zero_val), None,
                milli_ops::PadMode::Constant, rng,
            )
        };

        // Also pad ones_padded if it exists
        let ones_padded = ones_padded.map(|ones_p| {
            let extra_pads2 = milli_ops::Concat::push_new(
                &mut graph,
                vec![zeros_batch_ch, zeros_bc2, extra_pad],
                0,
                rng,
            );
            let zero_val = ops_helpers::scalar_const(&mut graph, 0.0f32, rng);
            milli_ops::Pad::push_new(
                &mut graph, ones_p, extra_pads2, Some(zero_val), None,
                milli_ops::PadMode::Constant, rng,
            )
        });

        // For each tap: end = start + out_spatial * strides
        let out_times_strides =
            milli_ops::SimpleBinary::mul(&mut graph, out_spatial, strides_const, rng);

        let kernel_size: usize = kernel_shape.iter().map(|&x| x as usize).product();
        let axes: Vec<i64> = (0..n_spatial).map(|i| (i + 2) as i64).collect();
        let axes_const = milli_ops::Constant::push_new(
            &mut graph,
            NDArrayNumericTensor::from(axes).to_dyn(),
            rng,
        );
        let steps_const = milli_ops::Constant::push_new(
            &mut graph,
            NDArrayNumericTensor::from(strides.clone()).to_dyn(),
            rng,
        );

        let mut sum_acc: Option<GlobalId> = None;
        let mut count_acc: Option<GlobalId> = None;

        let mut kernel_strides_table = vec![1usize; n_spatial];
        for i in (0..n_spatial.saturating_sub(1)).rev() {
            kernel_strides_table[i] = kernel_strides_table[i + 1] * kernel_shape[i + 1] as usize;
        }

        for k_idx in 0..kernel_size {
            let mut starts = Vec::with_capacity(n_spatial);
            let mut remaining = k_idx;
            for d in 0..n_spatial {
                let k_coord = remaining / kernel_strides_table[d];
                remaining %= kernel_strides_table[d];
                starts.push(k_coord as i64 * dilations[d]);
            }

            let starts_const = milli_ops::Constant::push_new(
                &mut graph,
                NDArrayNumericTensor::from(starts).to_dyn(),
                rng,
            );

            // end = start + out_spatial * strides
            let ends_const =
                milli_ops::SimpleBinary::add(&mut graph, starts_const, out_times_strides, rng);

            let tap = milli_ops::Slice::push_new(
                &mut graph, padded, starts_const, ends_const,
                Some(steps_const), Some(axes_const), rng,
            );

            sum_acc = Some(match sum_acc {
                None => tap,
                Some(acc) => milli_ops::SimpleBinary::add(&mut graph, acc, tap, rng),
            });

            if let Some(ones_p) = ones_padded {
                let count_tap = milli_ops::Slice::push_new(
                    &mut graph, ones_p, starts_const, ends_const,
                    Some(steps_const), Some(axes_const), rng,
                );
                count_acc = Some(match count_acc {
                    None => count_tap,
                    Some(acc) => milli_ops::SimpleBinary::add(&mut graph, acc, count_tap, rng),
                });
            }
        }

        let sum = sum_acc.unwrap();

        let result = if let Some(count) = count_acc {
            milli_ops::SimpleBinary::div(&mut graph, sum, count, rng)
        } else {
            let divisor = ops_helpers::scalar_const(&mut graph, kernel_size as f32, rng);
            milli_ops::SimpleBinary::div(&mut graph, sum, divisor, rng)
        };

        let out = milli_ops::CastLike::push_new(&mut graph, result, original_input, rng);

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
