use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::dtype::DType;
use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::{MilliLoweringContext, MilliOpGraph};
use crate::numeric_tensor::NumericTensor;
use crate::onnx;
use crate::symbolic_graph::ops::{EvalError, Operation};
use crate::symbolic_graph::{
    ONNXDecodingError, query_attribute_int, query_attribute_ints, query_attribute_string,
};
use crate::tensor_rank::DynRank;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
enum AutoPad {
    NotSet,
    SameUpper,
    SameLower,
    Valid,
}

/// ONNX ConvTranspose (transposed / deconvolution).
///
/// Currently supports 1D only (sufficient for Kokoro and most audio models).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConvTransposeOperation {
    global_id: GlobalId,
    input: GlobalId,
    weight: GlobalId,
    bias: Option<GlobalId>,
    output: GlobalId,
    auto_pad: AutoPad,
    dilations: Vec<i64>,
    group: i64,
    kernel_shape: Vec<i64>,
    output_padding: Vec<i64>,
    output_shape: Vec<i64>,
    pads: Vec<i64>,
    strides: Vec<i64>,
}

impl ConvTransposeOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() < 2 || inputs.len() > 3 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("ConvTranspose"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("ConvTranspose"));
        }

        let auto_pad = match query_attribute_string(attributes, "auto_pad").as_deref() {
            Some("SAME_UPPER") => AutoPad::SameUpper,
            Some("SAME_LOWER") => AutoPad::SameLower,
            Some("VALID") => AutoPad::Valid,
            _ => AutoPad::NotSet,
        };

        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("ConvTranspose"))?,
            weight: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("ConvTranspose"))?,
            bias: inputs.get(2).and_then(|x| *x),
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("ConvTranspose"))?,
            auto_pad,
            dilations: query_attribute_ints(attributes, "dilations").unwrap_or_default(),
            group: query_attribute_int(attributes, "group").unwrap_or(1),
            kernel_shape: query_attribute_ints(attributes, "kernel_shape").unwrap_or_default(),
            output_padding: query_attribute_ints(attributes, "output_padding").unwrap_or_default(),
            output_shape: query_attribute_ints(attributes, "output_shape").unwrap_or_default(),
            pads: query_attribute_ints(attributes, "pads").unwrap_or_default(),
            strides: query_attribute_ints(attributes, "strides").unwrap_or_default(),
        })
    }
}

impl Node for ConvTransposeOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "ConvTranspose".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        let mut v = vec![self.input, self.weight];
        if let Some(b) = self.bias {
            v.push(b);
        }
        Box::new(v.into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for ConvTransposeOperation {
    fn parameters(&self) -> Vec<Property> {
        let mut params = Vec::new();
        params.push(Property::new("group", PropertyValue::Int(self.group)));
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
        params
    }

    fn is_differentiable(&self) -> bool {
        false
    }

    fn eval(
        &self,
        backend: &mut EvalBackend,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, EvalError> {
        let x = &inputs[&self.input];
        let w = &inputs[&self.weight];
        let original_dtype = x.dtype();

        let x_shape: Vec<usize> = x.shape().iter().map(|&v| v as usize).collect();
        let w_shape: Vec<usize> = w.shape().iter().map(|&v| v as usize).collect();
        let nd = x_shape.len() - 2; // number of spatial dimensions

        let batch = x_shape[0];
        let c_in = x_shape[1];
        let groups = self.group as usize;
        // W shape: [C_in, C_out/group, K0, K1, ...]
        let c_out_per_group = w_shape[1];
        let c_out = c_out_per_group * groups;
        let c_in_per_group = c_in / groups;

        // Per-axis parameters with defaults
        let kernel: Vec<usize> = (0..nd)
            .map(|i| {
                if i < self.kernel_shape.len() { self.kernel_shape[i] as usize }
                else { w_shape[2 + i] }
            })
            .collect();
        let stride: Vec<usize> = (0..nd)
            .map(|i| if i < self.strides.len() { self.strides[i] as usize } else { 1 })
            .collect();
        let dilation: Vec<usize> = (0..nd)
            .map(|i| if i < self.dilations.len() { self.dilations[i] as usize } else { 1 })
            .collect();
        let output_pad: Vec<usize> = (0..nd)
            .map(|i| if i < self.output_padding.len() { self.output_padding[i] as usize } else { 0 })
            .collect();
        let in_spatial: Vec<usize> = (0..nd).map(|i| x_shape[2 + i]).collect();

        // Compute pads: either explicit, derived from output_shape, or from auto_pad.
        let (pad_begin, pad_end, output_pad) = if !self.output_shape.is_empty() {
            // output_shape given: compute pads and output_padding to achieve it.
            // ONNX formula: output_shape[i] = stride[i]*(in-1) + output_padding[i] + ek - pbegin - pend
            // First compute no-pad output, then derive what adjustments are needed.
            let mut pb = vec![0usize; nd];
            let mut pe = vec![0usize; nd];
            let mut op = output_pad.clone();
            for i in 0..nd {
                let ek = (kernel[i] - 1) * dilation[i] + 1;
                let no_pad_out = (in_spatial[i] - 1) * stride[i] + ek;
                let target = self.output_shape[i] as usize;
                if no_pad_out >= target {
                    let total_pad = no_pad_out - target;
                    pb[i] = total_pad / 2;
                    pe[i] = total_pad - pb[i];
                } else {
                    // Need output_padding to reach target
                    op[i] = target - no_pad_out;
                }
            }
            (pb, pe, op)
        } else {
            match self.auto_pad {
                AutoPad::SameUpper | AutoPad::SameLower => {
                    let mut pb = vec![0usize; nd];
                    let mut pe = vec![0usize; nd];
                    for i in 0..nd {
                        let out_i = in_spatial[i] * stride[i];
                        let ek = (kernel[i] - 1) * dilation[i] + 1;
                        let total_pad =
                            stride[i] * (in_spatial[i] - 1) + output_pad[i] + ek - out_i;
                        if matches!(self.auto_pad, AutoPad::SameLower) {
                            pe[i] = total_pad / 2;
                            pb[i] = total_pad - pe[i];
                        } else {
                            pb[i] = total_pad / 2;
                            pe[i] = total_pad - pb[i];
                        }
                    }
                    (pb, pe, output_pad)
                }
                _ => {
                    let pb = (0..nd)
                        .map(|i| if i < self.pads.len() { self.pads[i] as usize } else { 0 })
                        .collect();
                    let pe = (0..nd)
                        .map(|i| {
                            if nd + i < self.pads.len() { self.pads[nd + i] as usize } else { 0 }
                        })
                        .collect();
                    (pb, pe, output_pad)
                }
            }
        };

        // Compute spatial output sizes
        let out_spatial: Vec<usize> = (0..nd)
            .map(|i| {
                let ek = (kernel[i] - 1) * dilation[i] + 1;
                (in_spatial[i] - 1) * stride[i] - pad_begin[i] - pad_end[i] + ek + output_pad[i]
            })
            .collect();

        // Compute strides for flat indexing
        let in_spatial_stride = {
            let mut s = vec![1usize; nd];
            for i in (0..nd - 1).rev() { s[i] = s[i + 1] * in_spatial[i + 1]; }
            s
        };
        let out_spatial_stride = {
            let mut s = vec![1usize; nd];
            for i in (0..nd - 1).rev() { s[i] = s[i + 1] * out_spatial[i + 1]; }
            s
        };
        let kernel_stride = {
            let mut s = vec![1usize; nd];
            for i in (0..nd - 1).rev() { s[i] = s[i + 1] * kernel[i + 1]; }
            s
        };

        let in_spatial_total: usize = in_spatial.iter().product();
        let out_spatial_total: usize = out_spatial.iter().product();
        let kernel_total: usize = kernel.iter().product();

        // Cast to f32 for computation
        let x_f32 = x.cast(DType::F32, backend)?;
        let w_f32 = w.cast(DType::F32, backend)?;
        let x_data: Vec<f32> = x_f32.to_ndarray()?.flatten().try_into()?;
        let w_data: Vec<f32> = w_f32.to_ndarray()?.flatten().try_into()?;

        let mut out_data = vec![0.0f32; batch * c_out * out_spatial_total];

        for b in 0..batch {
            for g in 0..groups {
                for c_i in 0..c_in_per_group {
                    let in_ch = g * c_in_per_group + c_i;
                    for c_o in 0..c_out_per_group {
                        let out_ch = g * c_out_per_group + c_o;
                        for in_flat in 0..in_spatial_total {
                            let x_val = x_data
                                [b * c_in * in_spatial_total + in_ch * in_spatial_total + in_flat];
                            if x_val == 0.0 {
                                continue;
                            }
                            // Decompose in_flat into per-axis indices
                            let mut in_idx = vec![0usize; nd];
                            {
                                let mut rem = in_flat;
                                for d in 0..nd {
                                    in_idx[d] = rem / in_spatial_stride[d];
                                    rem %= in_spatial_stride[d];
                                }
                            }
                            for k_flat in 0..kernel_total {
                                // Decompose k_flat into per-axis kernel indices
                                let mut valid = true;
                                let mut out_flat = 0usize;
                                {
                                    let mut rem = k_flat;
                                    for d in 0..nd {
                                        let kd = rem / kernel_stride[d];
                                        rem %= kernel_stride[d];
                                        let o = in_idx[d] as isize * stride[d] as isize
                                            + kd as isize * dilation[d] as isize
                                            - pad_begin[d] as isize;
                                        if o < 0 || o as usize >= out_spatial[d] {
                                            valid = false;
                                            break;
                                        }
                                        out_flat += o as usize * out_spatial_stride[d];
                                    }
                                }
                                if valid {
                                    out_data[b * c_out * out_spatial_total
                                        + out_ch * out_spatial_total
                                        + out_flat] += x_val
                                        * w_data[in_ch * c_out_per_group * kernel_total
                                            + c_o * kernel_total
                                            + k_flat];
                                }
                            }
                        }
                    }
                }
            }
        }

        // Add bias
        if let Some(bias_id) = self.bias {
            let bias = &inputs[&bias_id];
            let bias_f32 = bias.cast(DType::F32, backend)?;
            let bias_data: Vec<f32> = bias_f32.to_ndarray()?.flatten().try_into()?;
            for b in 0..batch {
                for c in 0..c_out {
                    let bias_val = bias_data[c];
                    for s in 0..out_spatial_total {
                        out_data[b * c_out * out_spatial_total + c * out_spatial_total + s] +=
                            bias_val;
                    }
                }
            }
        }

        let mut out_shape_full = vec![batch as u64, c_out as u64];
        out_shape_full.extend(out_spatial.iter().map(|&s| s as u64));

        let mut out = NumericTensor::NDArray(NDArrayNumericTensor::from_vec_shape(
            out_data,
            &out_shape_full,
        )?);

        // Cast back to original dtype if needed
        if original_dtype != DType::F32 {
            out = out.cast(original_dtype, backend)?;
        }

        let mut result = HashMap::new();
        result.insert(self.output, out);
        Ok(Box::new(result.into_iter()))
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, _rng: &mut impl Rng) -> MilliOpGraph {
        panic!("ConvTranspose uses custom eval")
    }
}
