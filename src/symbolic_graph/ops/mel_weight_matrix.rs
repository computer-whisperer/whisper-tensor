use crate::graph::{GlobalId, Node};
use crate::milli_graph::{MilliLoweringContext, MilliOpGraph};
use crate::onnx;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{ONNXDecodingError, query_attribute_int};
use rand::Rng;
use serde::{Deserialize, Serialize};

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

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, _rng: &mut impl Rng) -> MilliOpGraph {
        panic!("MelWeightMatrix uses custom eval, not milli-op decomposition")
    }
}
