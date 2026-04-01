use crate::graph::GlobalId;
use crate::graph::Node;
use crate::milli_graph::ops as milli_ops;
use crate::milli_graph::{MilliLoweringContext, MilliOpGraph, ops_helpers};
use crate::numeric_dtype::NumericDType;
use crate::onnx;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{ONNXDecodingError, query_attribute_int};
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
            return Err(ONNXDecodingError::InvalidOperatorOutputs("MelWeightMatrix"));
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

impl Operation for MelWeightMatrixOperation {
    fn is_differentiable(&self) -> bool {
        false
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        // Follows the ONNX reference implementation exactly:
        // 1. Compute mel-spaced bin indices (integers)
        // 2. Build triangular filters using integer bin iteration

        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let num_mel_bins_in = input_map[&self.num_mel_bins];
        let dft_length_in = input_map[&self.dft_length];
        let sample_rate_in = input_map[&self.sample_rate];
        let lower_hz_in = input_map[&self.lower_edge_hertz];
        let upper_hz_in = input_map[&self.upper_edge_hertz];

        let zero_i64 = ops_helpers::scalar_const(&mut graph, 0i64, rng);
        let one_i64 = ops_helpers::scalar_const(&mut graph, 1i64, rng);
        let two_i64 = ops_helpers::scalar_const(&mut graph, 2i64, rng);

        // num_spectrogram_bins = dft_length // 2 + 1
        let dft_i64 = milli_ops::Cast::push_new(&mut graph, dft_length_in, NumericDType::I64, rng);
        let half_dft = milli_ops::SimpleBinary::div(&mut graph, dft_i64, two_i64, rng);
        let num_spec_bins = milli_ops::SimpleBinary::add(&mut graph, half_dft, one_i64, rng);

        let nmel_i64 =
            milli_ops::Cast::push_new(&mut graph, num_mel_bins_in, NumericDType::I64, rng);

        // frequency_bins = arange(0, num_mel_bins + 2)
        let nmel_plus_2 = milli_ops::SimpleBinary::add(&mut graph, nmel_i64, two_i64, rng);
        let fb_indices =
            milli_ops::Range::push_new(&mut graph, zero_i64, nmel_plus_2, one_i64, rng);
        let fb_f32 = milli_ops::Cast::push_new(&mut graph, fb_indices, NumericDType::F32, rng);

        // Hz → mel
        let lower_f32 = milli_ops::Cast::push_new(&mut graph, lower_hz_in, NumericDType::F32, rng);
        let upper_f32 = milli_ops::Cast::push_new(&mut graph, upper_hz_in, NumericDType::F32, rng);
        let low_mel = hz_to_mel(&mut graph, lower_f32, rng);
        let high_mel = hz_to_mel(&mut graph, upper_f32, rng);

        // mel_step = (high_mel - low_mel) / len(frequency_bins)
        // Note: ONNX divides by num_mel_bins+2, not num_mel_bins+1
        let nmel_plus_2_f32 =
            milli_ops::Cast::push_new(&mut graph, nmel_plus_2, NumericDType::F32, rng);
        let mel_range = milli_ops::SimpleBinary::sub(&mut graph, high_mel, low_mel, rng);
        let mel_step = milli_ops::SimpleBinary::div(&mut graph, mel_range, nmel_plus_2_f32, rng);

        // frequency_bins = frequency_bins * mel_step + low_mel (mel values)
        let fb_mel = milli_ops::SimpleBinary::mul(&mut graph, fb_f32, mel_step, rng);
        let fb_mel = milli_ops::SimpleBinary::add(&mut graph, fb_mel, low_mel, rng);

        // mel → Hz
        let fb_hz = mel_to_hz(&mut graph, fb_mel, rng);

        // Convert to integer FFT bin indices: ((dft_length + 1) * hz) // sample_rate
        let dft_plus_1 = milli_ops::SimpleBinary::add(&mut graph, dft_i64, one_i64, rng);
        let dft_plus_1_f32 =
            milli_ops::Cast::push_new(&mut graph, dft_plus_1, NumericDType::F32, rng);
        let sr_f32 = milli_ops::Cast::push_new(&mut graph, sample_rate_in, NumericDType::F32, rng);
        let fb_scaled = milli_ops::SimpleBinary::mul(&mut graph, dft_plus_1_f32, fb_hz, rng);
        let fb_div = milli_ops::SimpleBinary::div(&mut graph, fb_scaled, sr_f32, rng);
        // Floor to get integer bin indices
        let fb_floor = milli_ops::SimpleUnaryOp::floor(&mut graph, fb_div, rng);
        let fb_bins = milli_ops::Cast::push_new(&mut graph, fb_floor, NumericDType::I64, rng);

        // Build output [num_spectrogram_bins, num_mel_bins] using broadcasting.
        // For each mel bin i: left=fb_bins[i], center=fb_bins[i+1], right=fb_bins[i+2]
        // For each spectrogram bin j:
        //   up_slope = (j - left) / (center - left)   if center != left
        //   down_slope = (right - j) / (right - center)  if right != center
        //   weight = max(0, min(up_slope, down_slope))
        //
        // Using integer bin indices means the triangular filter hits exact integer points.
        let axes_0 = milli_ops::Constant::from_vec(&mut graph, vec![0i64], rng);
        let nmel_plus_1 = milli_ops::SimpleBinary::add(&mut graph, nmel_i64, one_i64, rng);
        let left = milli_ops::Slice::push_new(
            &mut graph,
            fb_bins,
            zero_i64,
            nmel_i64,
            None,
            Some(axes_0),
            rng,
        );
        let center = milli_ops::Slice::push_new(
            &mut graph,
            fb_bins,
            one_i64,
            nmel_plus_1,
            None,
            Some(axes_0),
            rng,
        );
        let right = milli_ops::Slice::push_new(
            &mut graph,
            fb_bins,
            two_i64,
            nmel_plus_2,
            None,
            Some(axes_0),
            rng,
        );

        // Cast to F32 for division.
        let left_f = milli_ops::Cast::push_new(&mut graph, left, NumericDType::F32, rng);
        let center_f = milli_ops::Cast::push_new(&mut graph, center, NumericDType::F32, rng);
        let right_f = milli_ops::Cast::push_new(&mut graph, right, NumericDType::F32, rng);

        // j = Range(0, num_spectrogram_bins) as F32
        let j_range = milli_ops::Range::push_new(&mut graph, zero_i64, num_spec_bins, one_i64, rng);
        let j_f = milli_ops::Cast::push_new(&mut graph, j_range, NumericDType::F32, rng);

        // Unsqueeze for broadcasting: j → [num_spec, 1], filters → [1, num_mel]
        let axes_1_vec = milli_ops::Constant::from_vec(&mut graph, vec![1i64], rng);
        let axes_0_vec = milli_ops::Constant::from_vec(&mut graph, vec![0i64], rng);
        let j_col = milli_ops::Unsqueeze::push_new(&mut graph, j_f, axes_1_vec, rng);
        let left_row = milli_ops::Unsqueeze::push_new(&mut graph, left_f, axes_0_vec, rng);
        let center_row = milli_ops::Unsqueeze::push_new(&mut graph, center_f, axes_0_vec, rng);
        let right_row = milli_ops::Unsqueeze::push_new(&mut graph, right_f, axes_0_vec, rng);

        let zero_f32 = milli_ops::Constant::new_scalar(&mut graph, 0.0f32, rng);
        let one_f32 = milli_ops::Constant::new_scalar(&mut graph, 1.0f32, rng);

        // Following ONNX reference: ascending part [left..=center], descending [center..right).
        // Ascending: weight = (j - left) / (center - left), or 1 at center when center==left.
        // Descending: weight = (right - j) / (right - center), only when right > center.

        // Region masks.
        let j_ge_left = milli_ops::SimpleBinary::greater_or_equal(&mut graph, j_col, left_row, rng);
        let j_le_center =
            milli_ops::SimpleBinary::less_or_equal(&mut graph, j_col, center_row, rng);
        let in_ascending = milli_ops::SimpleBinary::and(&mut graph, j_ge_left, j_le_center, rng);

        let j_gt_center = milli_ops::SimpleBinary::greater(&mut graph, j_col, center_row, rng);
        let j_lt_right = milli_ops::SimpleBinary::less(&mut graph, j_col, right_row, rng);
        let in_descending = milli_ops::SimpleBinary::and(&mut graph, j_gt_center, j_lt_right, rng);

        // Ascending weight: (j - left) / max(1, center - left).
        // Special case: when center == left, weight = 1 at j == center.
        let j_minus_left = milli_ops::SimpleBinary::sub(&mut graph, j_col, left_row, rng);
        let c_minus_l = milli_ops::SimpleBinary::sub(&mut graph, center_row, left_row, rng);
        let c_minus_l_safe = milli_ops::SimpleBinary::max(&mut graph, c_minus_l, one_f32, rng);
        let asc_normal =
            milli_ops::SimpleBinary::div(&mut graph, j_minus_left, c_minus_l_safe, rng);
        let c_eq_l = milli_ops::SimpleBinary::equal(&mut graph, center_row, left_row, rng);
        let j_eq_center = milli_ops::SimpleBinary::equal(&mut graph, j_col, center_row, rng);
        let j_eq_c_f = milli_ops::Cast::push_new(&mut graph, j_eq_center, NumericDType::F32, rng);
        let asc_weight = milli_ops::Where::push_new(&mut graph, c_eq_l, j_eq_c_f, asc_normal, rng);

        // Descending weight: (right - j) / max(1, right - center).
        let r_minus_j = milli_ops::SimpleBinary::sub(&mut graph, right_row, j_col, rng);
        let r_minus_c = milli_ops::SimpleBinary::sub(&mut graph, right_row, center_row, rng);
        let r_minus_c_safe = milli_ops::SimpleBinary::max(&mut graph, r_minus_c, one_f32, rng);
        let desc_weight = milli_ops::SimpleBinary::div(&mut graph, r_minus_j, r_minus_c_safe, rng);

        // Combine: ascending region uses asc_weight, descending uses desc_weight, else 0.
        let weight =
            milli_ops::Where::push_new(&mut graph, in_ascending, asc_weight, zero_f32, rng);
        let weight =
            milli_ops::Where::push_new(&mut graph, in_descending, desc_weight, weight, rng);

        // Cast to output dtype.
        let out_ndt = onnx_datatype_to_numeric(self.output_datatype);
        let result = milli_ops::Cast::push_new(&mut graph, weight, out_ndt, rng);

        let mut output_map = HashMap::new();
        output_map.insert(result, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

/// hz_to_mel: mel = 2595 * log10(1 + hz / 700)
fn hz_to_mel(graph: &mut MilliOpGraph, hz: GlobalId, rng: &mut impl Rng) -> GlobalId {
    let c700 = milli_ops::Constant::new_scalar(graph, 700.0f32, rng);
    let c2595 = milli_ops::Constant::new_scalar(graph, 2595.0f32, rng);
    let ratio = milli_ops::SimpleBinary::div(graph, hz, c700, rng);
    let one = milli_ops::Constant::new_scalar(graph, 1.0f32, rng);
    let one_plus = milli_ops::SimpleBinary::add(graph, one, ratio, rng);
    let ln_val = milli_ops::SimpleUnaryOp::ln(graph, one_plus, rng);
    let ln10 = milli_ops::Constant::new_scalar(graph, std::f32::consts::LN_10, rng);
    let log10_val = milli_ops::SimpleBinary::div(graph, ln_val, ln10, rng);
    milli_ops::SimpleBinary::mul(graph, c2595, log10_val, rng)
}

/// mel_to_hz: hz = 700 * (10^(mel/2595) - 1)
fn mel_to_hz(graph: &mut MilliOpGraph, mel: GlobalId, rng: &mut impl Rng) -> GlobalId {
    let c700 = milli_ops::Constant::new_scalar(graph, 700.0f32, rng);
    let c2595 = milli_ops::Constant::new_scalar(graph, 2595.0f32, rng);
    let ten = milli_ops::Constant::new_scalar(graph, 10.0f32, rng);
    let one = milli_ops::Constant::new_scalar(graph, 1.0f32, rng);
    let mel_div = milli_ops::SimpleBinary::div(graph, mel, c2595, rng);
    let pow10 = milli_ops::Pow::push_new(graph, ten, mel_div, rng);
    let minus_one = milli_ops::SimpleBinary::sub(graph, pow10, one, rng);
    milli_ops::SimpleBinary::mul(graph, c700, minus_one, rng)
}

fn onnx_datatype_to_numeric(dt: i64) -> NumericDType {
    // ONNX DataType enum values.
    match dt {
        1 => NumericDType::F32,
        2 => NumericDType::U8,
        3 => NumericDType::I8,
        5 => NumericDType::I16,
        6 => NumericDType::I32,
        7 => NumericDType::I64,
        10 => NumericDType::F16,
        11 => NumericDType::F64,
        12 => NumericDType::U32,
        13 => NumericDType::U64,
        16 => NumericDType::BF16,
        _ => NumericDType::F32, // default
    }
}
