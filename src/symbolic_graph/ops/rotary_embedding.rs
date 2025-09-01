use crate::milli_graph::MilliOpGraph;
use crate::onnx;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{ONNXDecodingError, SymbolicGraphTensorId, query_attribute_int};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RotaryEmbeddingOperation {
    data_input: SymbolicGraphTensorId,
    cos_cache: SymbolicGraphTensorId,
    sin_cache: SymbolicGraphTensorId,
    position_ids: Option<SymbolicGraphTensorId>,
    data_output: SymbolicGraphTensorId,
    interleaved: bool,
    num_heads: Option<i64>,
    rotary_embedding_dim: i64,
}

impl RotaryEmbeddingOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() < 3 || inputs.len() > 4 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("RotaryEmbedding"));
        }
        if outputs.is_empty() || outputs.len() > 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("RotaryEmbedding"));
        }
        let interleaved = query_attribute_int(attributes, "interleaved").unwrap_or_default() != 0;
        let num_heads = query_attribute_int(attributes, "num_heads");
        let rotary_embedding_dim =
            query_attribute_int(attributes, "rotary_embedding_dim").unwrap_or_default();
        Ok(Self {
            data_input: inputs[0]
                .ok_or(ONNXDecodingError::InvalidOperatorInputs("RotaryEmbedding"))?,
            cos_cache: inputs[1]
                .ok_or(ONNXDecodingError::InvalidOperatorInputs("RotaryEmbedding"))?,
            sin_cache: inputs[2]
                .ok_or(ONNXDecodingError::InvalidOperatorInputs("RotaryEmbedding"))?,
            position_ids: inputs.get(3).cloned().flatten(),
            data_output: outputs[0]
                .ok_or(ONNXDecodingError::InvalidOperatorOutputs("RotaryEmbedding"))?,
            interleaved,
            num_heads,
            rotary_embedding_dim,
        })
    }
}

impl Operation for RotaryEmbeddingOperation {
    fn get_op_type_name(&self) -> String {
        "Rotary Embedding".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        let mut v = vec![self.data_input, self.cos_cache, self.sin_cache];
        if let Some(x) = self.position_ids {
            v.push(x);
        }
        v
    }

    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.data_output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        todo!()
    }
}
