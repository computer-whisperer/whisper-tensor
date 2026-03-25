use crate::graph::{GlobalId, Node};
use crate::milli_graph::ops::*;
use crate::milli_graph::{MilliLoweringContext, MilliOpGraph};
use crate::numeric_dtype::NumericDType;
use crate::symbolic_graph::ONNXDecodingError;
use crate::symbolic_graph::ops::Operation;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// ONNX MatMulInteger: integer matrix multiplication with zero-point offsets.
///
/// Y = (A - a_zero_point) @ (B - b_zero_point)
///
/// All arithmetic in INT32.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MatMulIntegerOperation {
    global_id: GlobalId,
    a: GlobalId,
    b: GlobalId,
    a_zero_point: Option<GlobalId>,
    b_zero_point: Option<GlobalId>,
    output: GlobalId,
}

impl MatMulIntegerOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        _attributes: &[crate::onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() < 2 || inputs.len() > 4 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("MatMulInteger"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("MatMulInteger"));
        }
        Ok(Self {
            global_id: GlobalId::new(rng),
            a: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("MatMulInteger"))?,
            b: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("MatMulInteger"))?,
            a_zero_point: inputs.get(2).and_then(|x| *x),
            b_zero_point: inputs.get(3).and_then(|x| *x),
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("MatMulInteger"))?,
        })
    }
}

impl Node for MatMulIntegerOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "MatMulInteger".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        let mut v = vec![self.a, self.b];
        if let Some(zp) = self.a_zero_point {
            v.push(zp);
        }
        if let Some(zp) = self.b_zero_point {
            v.push(zp);
        }
        Box::new(v.into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for MatMulIntegerOperation {
    fn is_differentiable(&self) -> bool {
        false
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut g, input_map) = MilliOpGraph::new(self.inputs(), rng);

        // Cast A and B to INT32
        let mut a_i32 = Cast::push_new(&mut g, input_map[&self.a], NumericDType::I32, rng);
        let mut b_i32 = Cast::push_new(&mut g, input_map[&self.b], NumericDType::I32, rng);

        // Subtract zero points (cast to I32 first, broadcasting handled by Sub)
        if let Some(azp) = self.a_zero_point {
            let azp_i32 = Cast::push_new(&mut g, input_map[&azp], NumericDType::I32, rng);
            a_i32 = SimpleBinary::sub(&mut g, a_i32, azp_i32, rng);
        }
        if let Some(bzp) = self.b_zero_point {
            let bzp_i32 = Cast::push_new(&mut g, input_map[&bzp], NumericDType::I32, rng);
            b_i32 = SimpleBinary::sub(&mut g, b_i32, bzp_i32, rng);
        }

        // MatMul in INT32
        let out = MatMul::push_new(
            &mut g,
            a_i32,
            b_i32,
            NumericDType::I32,
            NumericDType::I32,
            NumericDType::I32,
            NumericDType::I32,
            rng,
        );

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        g.set_output_map(output_map);
        g
    }
}
