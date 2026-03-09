use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::dtype::DType;
use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::MilliOpGraph;
use crate::numeric_tensor::NumericTensor;
use crate::onnx;
use crate::symbolic_graph::ops::{EvalError, Operation};
use crate::symbolic_graph::{ONNXDecodingError, query_attribute_int};
use crate::tensor_rank::DynRank;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// ONNX STFT (Short-Time Fourier Transform).
///
/// Inputs:
///   0: signal       [batch, signal_length, 1]
///   1: frame_step   [] (scalar i64)
///   2: window       [window_length] (optional, empty string if absent)
///   3: frame_length [] (scalar i64, optional)
///
/// Output: [batch, num_frames, fft_length, 2]  (real, imag)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StftOperation {
    global_id: GlobalId,
    inputs: Vec<Option<GlobalId>>,
    output: GlobalId,
    onesided: bool,
}

impl StftOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() < 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("STFT"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("STFT"));
        }

        let onesided = query_attribute_int(attributes, "onesided").unwrap_or(1) != 0;

        Ok(Self {
            global_id: GlobalId::new(rng),
            inputs: inputs.to_vec(),
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("STFT"))?,
            onesided,
        })
    }
}

impl Node for StftOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "STFT".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(
            self.inputs
                .iter()
                .filter_map(|x| *x)
                .collect::<Vec<_>>()
                .into_iter(),
        )
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for StftOperation {
    fn parameters(&self) -> Vec<Property> {
        vec![Property::new(
            "onesided",
            PropertyValue::Bool(self.onesided),
        )]
    }

    fn is_differentiable(&self) -> bool {
        false
    }

    fn eval(
        &self,
        backend: &mut EvalBackend,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, EvalError> {
        let signal = &inputs[&self.inputs[0].unwrap()];
        let frame_step_tensor = &inputs[&self.inputs[1].unwrap()];

        // frame_step: scalar i64
        let frame_step_data: Vec<i64> = frame_step_tensor.to_ndarray()?.flatten().try_into()?;
        let frame_step = frame_step_data[0] as usize;

        // window (optional, input 2)
        let window: Option<Vec<f32>> = self
            .inputs
            .get(2)
            .and_then(|id| id.as_ref())
            .and_then(|id| inputs.get(id))
            .map(|w| -> Result<Vec<f32>, EvalError> {
                let w_f32 = w.cast(DType::F32, backend)?;
                Ok(w_f32.to_ndarray()?.flatten().try_into()?)
            })
            .transpose()?;

        // frame_length (optional, input 3)
        let frame_length: usize = if let Some(fl_id) = self.inputs.get(3).and_then(|x| *x) {
            let fl_tensor = &inputs[&fl_id];
            let fl_data: Vec<i64> = fl_tensor.to_ndarray()?.flatten().try_into()?;
            fl_data[0] as usize
        } else if let Some(ref w) = window {
            w.len()
        } else {
            return Err(EvalError::InvalidInput(
                "STFT requires either frame_length or window".to_string(),
            ));
        };

        let signal_f32 = signal.cast(DType::F32, backend)?;
        let signal_data: Vec<f32> = signal_f32.to_ndarray()?.flatten().try_into()?;
        let signal_shape: Vec<usize> = signal.shape().iter().map(|&v| v as usize).collect();

        let batch = signal_shape[0];
        let signal_length = signal_shape[1];
        // signal_shape[2] should be 1

        let num_frames = (signal_length - frame_length) / frame_step + 1;
        let fft_length = if self.onesided {
            frame_length / 2 + 1
        } else {
            frame_length
        };

        // Precompute DFT twiddle factors in f64 for precision
        let twiddles: Vec<(f64, f64)> = (0..fft_length)
            .flat_map(|k| {
                (0..frame_length).map(move |n| {
                    let angle =
                        -2.0 * std::f64::consts::PI * k as f64 * n as f64 / frame_length as f64;
                    (angle.cos(), angle.sin())
                })
            })
            .collect();

        let mut out_data = vec![0.0f32; batch * num_frames * fft_length * 2];

        for b in 0..batch {
            for f in 0..num_frames {
                let frame_start = f * frame_step;
                for k in 0..fft_length {
                    let mut real = 0.0f64;
                    let mut imag = 0.0f64;
                    for n in 0..frame_length {
                        let mut sample = signal_data[b * signal_length + frame_start + n] as f64;
                        if let Some(ref w) = window {
                            sample *= w[n] as f64;
                        }
                        let (cos_val, sin_val) = twiddles[k * frame_length + n];
                        real += sample * cos_val;
                        imag += sample * sin_val;
                    }
                    let idx = ((b * num_frames + f) * fft_length + k) * 2;
                    out_data[idx] = real as f32;
                    out_data[idx + 1] = imag as f32;
                }
            }
        }

        let out = NumericTensor::NDArray(NDArrayNumericTensor::from_vec_shape(
            out_data,
            &vec![batch as u64, num_frames as u64, fft_length as u64, 2u64],
        )?);

        let mut result = HashMap::new();
        result.insert(self.output, out);
        Ok(Box::new(result.into_iter()))
    }

    fn get_milli_op_graph(&self, _rng: &mut impl Rng) -> MilliOpGraph {
        panic!("STFT uses custom eval")
    }
}
