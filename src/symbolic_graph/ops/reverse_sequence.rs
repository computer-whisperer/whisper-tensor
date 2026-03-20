use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::{MilliLoweringContext, MilliOpGraph};
use crate::numeric_tensor::NumericTensor;
use crate::onnx::AttributeProto;
use crate::symbolic_graph::ops::{EvalError, Operation};
use crate::symbolic_graph::{ONNXDecodingError, query_attribute_int};
use crate::tensor_rank::DynRank;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use typenum::P1;

/// ONNX ReverseSequence operator.
///
/// Reverses variable-length slices along a time axis, independently for each
/// element along a batch axis.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReverseSequenceOperation {
    global_id: GlobalId,
    input: GlobalId,
    sequence_lens: GlobalId,
    output: GlobalId,
    batch_axis: i64,
    time_axis: i64,
}

impl ReverseSequenceOperation {
    pub fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() < 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("ReverseSequence"));
        }
        if outputs.is_empty() {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("ReverseSequence"));
        }

        let batch_axis = query_attribute_int(attributes, "batch_axis").unwrap_or(1);
        let time_axis = query_attribute_int(attributes, "time_axis").unwrap_or(0);

        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("ReverseSequence"))?,
            sequence_lens: inputs[1]
                .ok_or(ONNXDecodingError::InvalidOperatorInputs("ReverseSequence"))?,
            output: outputs[0]
                .ok_or(ONNXDecodingError::InvalidOperatorOutputs("ReverseSequence"))?,
            batch_axis,
            time_axis,
        })
    }
}

impl Node for ReverseSequenceOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "ReverseSequence".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        Box::new([self.input, self.sequence_lens].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        Box::new([self.output].into_iter())
    }
}

impl Operation for ReverseSequenceOperation {
    fn parameters(&self) -> Vec<Property> {
        vec![
            Property::new("batch_axis", PropertyValue::Int(self.batch_axis)),
            Property::new("time_axis", PropertyValue::Int(self.time_axis)),
        ]
    }

    fn eval(
        &self,
        backend: &mut EvalBackend,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, EvalError> {
        let input = &inputs[&self.input];
        let rank = input.rank();
        let shape = input.shape();
        let orig_dtype = input.dtype();

        let batch_axis = if self.batch_axis < 0 {
            (rank as i64 + self.batch_axis) as usize
        } else {
            self.batch_axis as usize
        };
        let time_axis = if self.time_axis < 0 {
            (rank as i64 + self.time_axis) as usize
        } else {
            self.time_axis as usize
        };

        let seq_lens: Vec<i64> = inputs[&self.sequence_lens]
            .cast(DType::I64, backend)?
            .try_to_rank::<P1>()?
            .try_into()?;

        let data_f32 = input.cast(DType::F32, backend)?;
        let data_flat: Vec<f32> = data_f32.to_ndarray()?.flatten().try_into()?;

        // Compute strides
        let mut strides = vec![1usize; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1] as usize;
        }

        let total: usize = shape.iter().map(|&s| s as usize).product();
        let mut out_flat = vec![0.0f32; total];

        #[allow(clippy::needless_range_loop)]
        for flat_idx in 0..total {
            // Decompose flat index into coordinates
            let mut remaining = flat_idx;
            let mut coords = vec![0usize; rank];
            for d in 0..rank {
                coords[d] = remaining / strides[d];
                remaining %= strides[d];
            }

            let batch_idx = coords[batch_axis];
            let time_idx = coords[time_axis];
            let seq_len = seq_lens[batch_idx] as usize;

            // Within the sequence length, reverse the time axis
            let src_time = if time_idx < seq_len {
                seq_len - 1 - time_idx
            } else {
                time_idx
            };

            // Compute source flat index
            let mut src_flat = 0;
            for d in 0..rank {
                let c = if d == time_axis { src_time } else { coords[d] };
                src_flat += c * strides[d];
            }

            out_flat[flat_idx] = data_flat[src_flat];
        }

        let output = NumericTensor::<DynRank>::from_vec_shape(
            out_flat,
            shape.iter().map(|&s| s as usize).collect(),
        )
        .map_err(|e| EvalError::InvalidInput(format!("ReverseSequence: {e}")))?;
        let output = output.cast(orig_dtype, backend)?;

        Ok(Box::new([(self.output, output)].into_iter()))
    }

    fn is_differentiable(&self) -> bool {
        false
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, _rng: &mut impl Rng) -> MilliOpGraph {
        panic!("ReverseSequence uses custom eval")
    }
}
