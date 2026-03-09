use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::dtype::DType;
use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::MilliOpGraph;
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
        let spatial_dims = x_shape.len() - 2;

        assert_eq!(
            spatial_dims, 1,
            "ConvTranspose currently only supports 1D (got {spatial_dims}D)"
        );

        let batch = x_shape[0];
        let c_in = x_shape[1];
        let l_in = x_shape[2];
        // W shape: [C_in, C_out/group, kernel]
        let c_out_per_group = w_shape[1];
        let kernel = if self.kernel_shape.is_empty() {
            w_shape[2]
        } else {
            self.kernel_shape[0] as usize
        };
        let groups = self.group as usize;
        let c_out = c_out_per_group * groups;
        let stride = if self.strides.is_empty() {
            1
        } else {
            self.strides[0] as usize
        };
        let dilation = if self.dilations.is_empty() {
            1
        } else {
            self.dilations[0] as usize
        };
        let (pad_begin, pad_end) = if self.pads.len() >= 2 {
            (self.pads[0] as usize, self.pads[1] as usize)
        } else {
            (0, 0)
        };
        let output_pad = if self.output_padding.is_empty() {
            0
        } else {
            self.output_padding[0] as usize
        };

        let effective_kernel = (kernel - 1) * dilation + 1;
        let l_out = (l_in - 1) * stride - pad_begin - pad_end + effective_kernel + output_pad;
        let c_in_per_group = c_in / groups;

        // Cast to f32 for computation
        let x_f32 = x.cast(DType::F32, backend)?;
        let w_f32 = w.cast(DType::F32, backend)?;
        let x_data: Vec<f32> = x_f32.to_ndarray()?.flatten().try_into()?;
        let w_data: Vec<f32> = w_f32.to_ndarray()?.flatten().try_into()?;

        let mut out_data = vec![0.0f32; batch * c_out * l_out];

        for b in 0..batch {
            for g in 0..groups {
                for c_i in 0..c_in_per_group {
                    let in_ch = g * c_in_per_group + c_i;
                    for c_o in 0..c_out_per_group {
                        let out_ch = g * c_out_per_group + c_o;
                        for i in 0..l_in {
                            let x_val = x_data[b * c_in * l_in + in_ch * l_in + i];
                            if x_val == 0.0 {
                                continue;
                            }
                            for k in 0..kernel {
                                let o_pos = i as isize * stride as isize
                                    + k as isize * dilation as isize
                                    - pad_begin as isize;
                                if o_pos >= 0 && (o_pos as usize) < l_out {
                                    out_data
                                        [b * c_out * l_out + out_ch * l_out + o_pos as usize] +=
                                        x_val
                                            * w_data[in_ch * c_out_per_group * kernel
                                                + c_o * kernel
                                                + k];
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
                    for l in 0..l_out {
                        out_data[b * c_out * l_out + c * l_out + l] += bias_val;
                    }
                }
            }
        }

        let mut out = NumericTensor::NDArray(NDArrayNumericTensor::from_vec_shape(
            out_data,
            &vec![batch as u64, c_out as u64, l_out as u64],
        )?);

        // Cast back to original dtype if needed
        if original_dtype != DType::F32 {
            out = out.cast(original_dtype, backend)?;
        }

        let mut result = HashMap::new();
        result.insert(self.output, out);
        Ok(Box::new(result.into_iter()))
    }

    fn get_milli_op_graph(&self, _rng: &mut impl Rng) -> MilliOpGraph {
        panic!("ConvTranspose uses custom eval")
    }
}
