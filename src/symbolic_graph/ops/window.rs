use crate::dtype::DType;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::*;
use crate::milli_graph::{MilliLoweringContext, MilliOpGraph};
use crate::numeric_dtype::NumericDType;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{ONNXDecodingError, query_attribute_int};
use crate::{TrigOp, onnx};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Which window function to generate.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum WindowKind {
    Blackman,
    Hamming,
    Hann,
}

/// ONNX BlackmanWindow / HammingWindow / HannWindow.
///
/// All three are generalized cosine windows:
///   w[n] = a0 - a1 * cos(2π * n / N) + a2 * cos(4π * n / N)
///
/// where N = size - 1 (symmetric) or N = size (periodic).
///
/// Coefficients:
///   Blackman: a0=0.42, a1=0.50, a2=0.08
///   Hamming:  a0=0.54, a1=0.46, a2=0.00
///   Hann:     a0=0.50, a1=0.50, a2=0.00
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WindowOperation {
    global_id: GlobalId,
    kind: WindowKind,
    size: GlobalId,
    output: GlobalId,
    /// 1 = periodic (default), 0 = symmetric
    periodic: i64,
    /// ONNX DataType enum value for output (default 1 = FLOAT)
    output_datatype: i64,
}

impl WindowOperation {
    pub(crate) fn from_onnx(
        kind: WindowKind,
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Window"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Window"));
        }
        Ok(Self {
            global_id: GlobalId::new(rng),
            kind,
            size: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Window"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Window"))?,
            periodic: query_attribute_int(attributes, "periodic").unwrap_or(1),
            output_datatype: query_attribute_int(attributes, "output_datatype").unwrap_or(1),
        })
    }

    fn coefficients(&self) -> (f64, f64, f64) {
        match self.kind {
            // ONNX uses exact fractions for Blackman: 42/100, 50/100, 8/100
            WindowKind::Blackman => (0.42, 0.50, 0.08),
            // ONNX uses exact fractions: 25/46, 21/46
            WindowKind::Hamming => (25.0 / 46.0, 21.0 / 46.0, 0.0),
            WindowKind::Hann => (0.50, 0.50, 0.0),
        }
    }

    fn op_name(&self) -> &'static str {
        match self.kind {
            WindowKind::Blackman => "BlackmanWindow",
            WindowKind::Hamming => "HammingWindow",
            WindowKind::Hann => "HannWindow",
        }
    }
}

impl Node for WindowOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        self.op_name().to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        Box::new(std::iter::once(self.size))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for WindowOperation {
    fn is_differentiable(&self) -> bool {
        false
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut g, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let size_input = input_map[&self.size];

        let (a0, a1, a2) = self.coefficients();

        // Determine output dtype from ONNX datatype attribute
        let out_dtype = {
            let onnx_dt = onnx::tensor_proto::DataType::try_from(self.output_datatype as i32)
                .unwrap_or(onnx::tensor_proto::DataType::Float);
            DType::try_from(onnx_dt).unwrap_or(DType::F32)
        };

        // Ensure size is INT64 for Range
        let size_i64 = Cast::push_new(&mut g, size_input, NumericDType::I64, rng);

        // Cast size to float for arithmetic
        let size_f32 = Cast::push_new(&mut g, size_i64, NumericDType::F32, rng);

        // N = size (periodic) or size - 1 (symmetric)
        let denom = if self.periodic == 1 {
            size_f32
        } else {
            let one = Constant::new_scalar(&mut g, 1.0f32, rng);
            SimpleBinary::sub(&mut g, size_f32, one, rng)
        };

        // n = range(0, size, 1) as f32
        let zero_i64 = Constant::new_scalar(&mut g, 0i64, rng);
        let one_i64 = Constant::new_scalar(&mut g, 1i64, rng);
        let n_i64 = Range::push_new(&mut g, zero_i64, size_i64, one_i64, rng);
        let n = Cast::push_new(&mut g, n_i64, NumericDType::F32, rng);

        // ratio = n / N
        let ratio = SimpleBinary::div(&mut g, n, denom, rng);

        // two_pi_ratio = 2π * ratio
        let two_pi = Constant::new_scalar(&mut g, (2.0 * std::f64::consts::PI) as f32, rng);
        let two_pi_ratio = SimpleBinary::mul(&mut g, two_pi, ratio, rng);

        // w = a0 - a1 * cos(2π * ratio)
        let cos1 = SimpleUnaryOp::trig(&mut g, two_pi_ratio, TrigOp::Cos, rng);
        let a0_c = Constant::new_scalar(&mut g, a0 as f32, rng);
        let a1_c = Constant::new_scalar(&mut g, a1 as f32, rng);
        let a1_cos1 = SimpleBinary::mul(&mut g, a1_c, cos1, rng);
        let mut w = SimpleBinary::sub(&mut g, a0_c, a1_cos1, rng);

        // + a2 * cos(4π * ratio)  (only nonzero for Blackman)
        if a2 != 0.0 {
            let four_pi = Constant::new_scalar(&mut g, (4.0 * std::f64::consts::PI) as f32, rng);
            let four_pi_ratio = SimpleBinary::mul(&mut g, four_pi, ratio, rng);
            let cos2 = SimpleUnaryOp::trig(&mut g, four_pi_ratio, TrigOp::Cos, rng);
            let a2_c = Constant::new_scalar(&mut g, a2 as f32, rng);
            let a2_cos2 = SimpleBinary::mul(&mut g, a2_c, cos2, rng);
            w = SimpleBinary::add(&mut g, w, a2_cos2, rng);
        }

        // Cast to requested output dtype
        if out_dtype != DType::F32 {
            w = Cast::push_new(
                &mut g,
                w,
                NumericDType::from_legacy(out_dtype).unwrap(),
                rng,
            );
        }

        let mut output_map = HashMap::new();
        output_map.insert(w, self.output);
        g.set_output_map(output_map);
        g
    }
}
