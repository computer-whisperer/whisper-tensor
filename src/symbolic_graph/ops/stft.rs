use crate::TrigOp;
use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::{self, MilliLoweringContext, MilliOpGraph};
use crate::numeric_dtype::NumericDType;
use crate::onnx;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{ONNXDecodingError, query_attribute_int};
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

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);

        let signal = input_map[&self.inputs[0].unwrap()];
        let frame_step_input = input_map[&self.inputs[1].unwrap()];
        let has_window = self.inputs.get(2).and_then(|x| *x).is_some();
        let has_frame_length_input = self.inputs.get(3).and_then(|x| *x).is_some();

        // Reusable scalar constants
        let zero_i64 = milli_graph::ops::Constant::new_scalar(&mut graph, 0i64, rng);
        let one_i64 = milli_graph::ops::Constant::new_scalar(&mut graph, 1i64, rng);
        let two_i64 = milli_graph::ops::Constant::new_scalar(&mut graph, 2i64, rng);

        // Axes constants (1D tensors for Squeeze/Unsqueeze)
        let axes_2 = milli_graph::ops::Constant::from_vec(&mut graph, vec![2i64], rng);
        let axes_0 = milli_graph::ops::Constant::from_vec(&mut graph, vec![0i64], rng);
        let axes_1 = milli_graph::ops::Constant::from_vec(&mut graph, vec![1i64], rng);
        let axes_neg1 = milli_graph::ops::Constant::from_vec(&mut graph, vec![-1i64], rng);

        // Step 1: Squeeze signal trailing dim: [batch, signal_length, 1] -> [batch, signal_length]
        let signal_2d = milli_graph::ops::Squeeze::push_new(&mut graph, signal, axes_2, rng);

        // Step 2: Cast to F32
        let signal_f32 =
            milli_graph::ops::Cast::push_new(&mut graph, signal_2d, NumericDType::F32, rng);

        // Step 3: Determine frame_length
        // If input[3] exists, use it; else use Shape(window)[0]
        let frame_length = if has_frame_length_input {
            input_map[&self.inputs[3].unwrap()]
        } else {
            // Must have window; frame_length = Shape(window)[0]
            let window_id = input_map[&self.inputs[2].unwrap()];
            let window_shape = milli_graph::ops::Shape::push_new(&mut graph, window_id, rng);
            milli_graph::ops::Gather::push_new(&mut graph, window_shape, zero_i64, 0, rng)
        };

        // Step 4: Build frame indices
        // signal_length = Shape(signal_2d)[1]
        let signal_2d_shape = milli_graph::ops::Shape::push_new(&mut graph, signal_2d, rng);
        let signal_length =
            milli_graph::ops::Gather::push_new(&mut graph, signal_2d_shape, one_i64, 0, rng);

        // end = signal_length - frame_length + 1
        let sl_minus_fl =
            milli_graph::ops::SimpleBinary::sub(&mut graph, signal_length, frame_length, rng);
        let range_end = milli_graph::ops::SimpleBinary::add(&mut graph, sl_minus_fl, one_i64, rng);

        // frame_starts = Range(0, range_end, frame_step) -> [num_frames]
        let frame_starts = milli_graph::ops::Range::push_new(
            &mut graph,
            zero_i64,
            range_end,
            frame_step_input,
            rng,
        );

        // within_frame = Range(0, frame_length, 1) -> [frame_length]
        let within_frame =
            milli_graph::ops::Range::push_new(&mut graph, zero_i64, frame_length, one_i64, rng);

        // Unsqueeze frame_starts at dim 1: [num_frames] -> [num_frames, 1]
        let frame_starts_2d =
            milli_graph::ops::Unsqueeze::push_new(&mut graph, frame_starts, axes_1, rng);

        // Unsqueeze within_frame at dim 0: [frame_length] -> [1, frame_length]
        let within_frame_2d =
            milli_graph::ops::Unsqueeze::push_new(&mut graph, within_frame, axes_0, rng);

        // indices = frame_starts_2d + within_frame_2d -> [num_frames, frame_length]
        let indices =
            milli_graph::ops::SimpleBinary::add(&mut graph, frame_starts_2d, within_frame_2d, rng);

        // Step 5: Gather(signal_f32, indices, axis=1) -> [batch, num_frames, frame_length]
        let frames = milli_graph::ops::Gather::push_new(&mut graph, signal_f32, indices, 1, rng);

        // Step 6: If window present, cast to F32 and multiply (broadcasts)
        let frames = if has_window {
            let window_id = input_map[&self.inputs[2].unwrap()];
            let window_f32 =
                milli_graph::ops::Cast::push_new(&mut graph, window_id, NumericDType::F32, rng);
            milli_graph::ops::SimpleBinary::mul(&mut graph, frames, window_f32, rng)
        } else {
            frames
        };

        // Step 7: Build DFT twiddle matrices
        // n_indices = Cast(within_frame, F32) -> [frame_length]
        let n_indices =
            milli_graph::ops::Cast::push_new(&mut graph, within_frame, NumericDType::F32, rng);

        // fft_length: if onesided: frame_length/2 + 1, else frame_length
        let fft_length = if self.onesided {
            let half = milli_graph::ops::SimpleBinary::div(&mut graph, frame_length, two_i64, rng);
            milli_graph::ops::SimpleBinary::add(&mut graph, half, one_i64, rng)
        } else {
            frame_length
        };

        // k_range = Range(0, fft_length, 1), cast to F32 -> [fft_length]
        let k_range =
            milli_graph::ops::Range::push_new(&mut graph, zero_i64, fft_length, one_i64, rng);
        let k_indices =
            milli_graph::ops::Cast::push_new(&mut graph, k_range, NumericDType::F32, rng);

        // Outer product: unsqueeze k at dim 1, unsqueeze n at dim 0, multiply
        // k_col: [fft_length] -> [fft_length, 1]
        let k_col = milli_graph::ops::Unsqueeze::push_new(&mut graph, k_indices, axes_1, rng);
        // n_row: [frame_length] -> [1, frame_length]
        let n_row = milli_graph::ops::Unsqueeze::push_new(&mut graph, n_indices, axes_0, rng);
        // kn = k_col * n_row -> [fft_length, frame_length]
        let kn = milli_graph::ops::SimpleBinary::mul(&mut graph, k_col, n_row, rng);

        // angle = -2 * pi * kn / frame_length
        let neg_two_pi =
            milli_graph::ops::Constant::new_scalar(&mut graph, -2.0f32 * std::f32::consts::PI, rng);
        let angle_num = milli_graph::ops::SimpleBinary::mul(&mut graph, neg_two_pi, kn, rng);
        let frame_length_f32 =
            milli_graph::ops::Cast::push_new(&mut graph, frame_length, NumericDType::F32, rng);
        let angle =
            milli_graph::ops::SimpleBinary::div(&mut graph, angle_num, frame_length_f32, rng);

        // cos_matrix = Cos(angle), sin_matrix = Sin(angle)  [fft_length, frame_length]
        let cos_matrix = milli_graph::ops::SimpleUnaryOp::trig(&mut graph, angle, TrigOp::Cos, rng);
        let sin_matrix = milli_graph::ops::SimpleUnaryOp::trig(&mut graph, angle, TrigOp::Sin, rng);

        // Transpose both to [frame_length, fft_length]
        let cos_t =
            milli_graph::ops::Transpose::push_new(&mut graph, cos_matrix, Some(vec![1, 0]), rng);
        let sin_t =
            milli_graph::ops::Transpose::push_new(&mut graph, sin_matrix, Some(vec![1, 0]), rng);

        // Step 8: MatMul
        // frames @ cos_t -> real [batch, num_frames, fft_length]
        let real = milli_graph::ops::MatMul::push_new_default_precision(
            &mut graph,
            frames,
            cos_t,
            NumericDType::F32,
            rng,
        );
        // frames @ sin_t -> imag [batch, num_frames, fft_length]
        let imag = milli_graph::ops::MatMul::push_new_default_precision(
            &mut graph,
            frames,
            sin_t,
            NumericDType::F32,
            rng,
        );

        // Step 9: Unsqueeze both at last dim, concat on axis -1
        // real: [batch, num_frames, fft_length] -> [batch, num_frames, fft_length, 1]
        let real_u = milli_graph::ops::Unsqueeze::push_new(&mut graph, real, axes_neg1, rng);
        // imag: [batch, num_frames, fft_length] -> [batch, num_frames, fft_length, 1]
        let imag_u = milli_graph::ops::Unsqueeze::push_new(&mut graph, imag, axes_neg1, rng);
        // Concat on axis -1 -> [batch, num_frames, fft_length, 2]
        let output = milli_graph::ops::Concat::push_new(&mut graph, vec![real_u, imag_u], -1, rng);

        let mut output_map = HashMap::new();
        output_map.insert(output, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
