use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::dtype::DType;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::{MilliLoweringContext, MilliOpGraph};
use crate::numeric_tensor::NumericTensor;
use crate::symbolic_graph::ops::{EvalError, Operation};
use crate::symbolic_graph::{ONNXDecodingError, query_attribute_int};
use crate::{DynRank, onnx};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// ONNX MelWeightMatrix: generates a mel-scale filterbank matrix.
///
/// Output shape: [floor(dft_length/2) + 1, num_mel_bins]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MelWeightMatrixOperation {
    global_id: GlobalId,
    num_mel_bins: GlobalId,
    dft_length: GlobalId,
    sample_rate: GlobalId,
    lower_edge_hertz: GlobalId,
    upper_edge_hertz: GlobalId,
    output: GlobalId,
    output_datatype: i64,
}

impl MelWeightMatrixOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 5 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("MelWeightMatrix"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs(
                "MelWeightMatrix",
            ));
        }
        Ok(Self {
            global_id: GlobalId::new(rng),
            num_mel_bins: inputs[0]
                .ok_or(ONNXDecodingError::InvalidOperatorInputs("MelWeightMatrix"))?,
            dft_length: inputs[1]
                .ok_or(ONNXDecodingError::InvalidOperatorInputs("MelWeightMatrix"))?,
            sample_rate: inputs[2]
                .ok_or(ONNXDecodingError::InvalidOperatorInputs("MelWeightMatrix"))?,
            lower_edge_hertz: inputs[3]
                .ok_or(ONNXDecodingError::InvalidOperatorInputs("MelWeightMatrix"))?,
            upper_edge_hertz: inputs[4]
                .ok_or(ONNXDecodingError::InvalidOperatorInputs("MelWeightMatrix"))?,
            output: outputs[0]
                .ok_or(ONNXDecodingError::InvalidOperatorOutputs("MelWeightMatrix"))?,
            output_datatype: query_attribute_int(attributes, "output_datatype").unwrap_or(1),
        })
    }
}

impl Node for MelWeightMatrixOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "MelWeightMatrix".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        Box::new(
            [
                self.num_mel_bins,
                self.dft_length,
                self.sample_rate,
                self.lower_edge_hertz,
                self.upper_edge_hertz,
            ]
            .into_iter(),
        )
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        Box::new(std::iter::once(self.output))
    }
}

/// Convert Hz to mel scale: mel(f) = 2595 * log10(1 + f/700)
fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert mel to Hz: hz(m) = 700 * (10^(m/2595) - 1)
fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0f64.powf(mel / 2595.0) - 1.0)
}

fn tensor_to_i64(tensor: &NumericTensor<DynRank>) -> i64 {
    tensor.first_element().to_f64() as i64
}

fn tensor_to_f64(tensor: &NumericTensor<DynRank>) -> f64 {
    tensor.first_element().to_f64()
}

impl Operation for MelWeightMatrixOperation {
    fn is_differentiable(&self) -> bool {
        false
    }

    fn eval(
        &self,
        backend: &mut EvalBackend,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, EvalError> {
        let num_mel_bins = tensor_to_i64(&inputs[&self.num_mel_bins]) as usize;
        let dft_length = tensor_to_i64(&inputs[&self.dft_length]) as usize;
        let sample_rate = tensor_to_i64(&inputs[&self.sample_rate]) as f64;
        let lower_edge_hertz = tensor_to_f64(&inputs[&self.lower_edge_hertz]);
        let upper_edge_hertz = tensor_to_f64(&inputs[&self.upper_edge_hertz]);

        let num_spectrogram_bins = dft_length / 2 + 1;

        // Compute mel-spaced frequency bin edges (num_mel_bins + 2 points)
        let low_mel = hz_to_mel(lower_edge_hertz);
        let high_mel = hz_to_mel(upper_edge_hertz);
        let num_points = num_mel_bins + 2;
        let mel_step = (high_mel - low_mel) / num_points as f64;

        // Convert mel points back to Hz, then to DFT bin indices
        let frequency_bins: Vec<usize> = (0..num_points)
            .map(|i| {
                let mel = i as f64 * mel_step + low_mel;
                let hz = mel_to_hz(mel);
                ((dft_length as f64 + 1.0) * hz / sample_rate) as usize
            })
            .collect();

        // Build triangular filterbank matrix
        let mut output = vec![0.0f32; num_spectrogram_bins * num_mel_bins];
        for i in 0..num_mel_bins {
            let lower = frequency_bins[i];
            let center = frequency_bins[i + 1];
            let higher = frequency_bins[i + 2];

            let low_to_center = center - lower;
            if low_to_center == 0 {
                if center < num_spectrogram_bins {
                    output[center * num_mel_bins + i] = 1.0;
                }
            } else {
                for j in lower..=center {
                    if j < num_spectrogram_bins {
                        output[j * num_mel_bins + i] =
                            (j - lower) as f32 / low_to_center as f32;
                    }
                }
            }

            let center_to_high = higher - center;
            if center_to_high > 0 {
                for j in center..higher {
                    if j < num_spectrogram_bins {
                        output[j * num_mel_bins + i] =
                            (higher - j) as f32 / center_to_high as f32;
                    }
                }
            }
        }

        let shape = vec![num_spectrogram_bins as u64, num_mel_bins as u64];
        let mut out_tensor =
            NumericTensor::NDArray(NDArrayNumericTensor::from_vec_shape(output, &shape)?);

        // Cast to requested output dtype
        let out_dtype = {
            let onnx_dt = onnx::tensor_proto::DataType::try_from(self.output_datatype as i32)
                .unwrap_or(onnx::tensor_proto::DataType::Float);
            DType::try_from(onnx_dt).unwrap_or(DType::F32)
        };
        if out_dtype != DType::F32 {
            out_tensor = out_tensor.cast(out_dtype, backend)?;
        }

        Ok(Box::new(std::iter::once((self.output, out_tensor))))
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, _rng: &mut impl Rng) -> MilliOpGraph {
        panic!("MelWeightMatrix uses custom eval, not milli-op decomposition")
    }
}
