use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::MilliOpGraph;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{ONNXDecodingError, query_attribute_int};
use crate::{milli_graph, onnx};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rand::Rng;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RotaryEmbeddingOperation {
    global_id: GlobalId,
    data_input: GlobalId,
    cos_cache: GlobalId,
    sin_cache: GlobalId,
    position_ids: Option<GlobalId>,
    data_output: GlobalId,
    interleaved: bool,
    num_heads: Option<i64>,
    rotary_embedding_dim: i64,
}

impl RotaryEmbeddingOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
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
            global_id: GlobalId::new(rng),
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

impl Node for RotaryEmbeddingOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "RotaryEmbedding".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        let mut v = vec![self.data_input, self.cos_cache, self.sin_cache];
        if let Some(x) = self.position_ids {
            v.push(x);
        }
        Box::new(v.into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.data_output))
    }
}
impl Operation for RotaryEmbeddingOperation {
    fn parameters(&self) -> Vec<Property> {
        let mut params = Vec::new();
        params.push(Property::new("interleaved", PropertyValue::Bool(self.interleaved)));
        if let Some(num_heads) = self.num_heads {
            params.push(Property::new("num_heads", PropertyValue::Int(num_heads)));
        }
        if self.rotary_embedding_dim != 0 {
            params.push(Property::new("rotary_embedding_dim", PropertyValue::Int(self.rotary_embedding_dim)));
        }
        params
    }

    fn get_milli_op_graph(&self, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);

        let data = input_map[&self.data_input];
        let cos_cache = input_map[&self.cos_cache];
        let sin_cache = input_map[&self.sin_cache];
        let pos_ids = self.position_ids.map(|x| input_map[&x]);

        // Prepare input to shape [B, S, H, D]
        let prepared = if self.num_heads.is_some() {
            // 3D path: [B, S, hidden] -> [B, S, num_heads, head_size]
            let new_shape = milli_graph::ops::Constant::push_new(
                &mut graph,
                NDArrayNumericTensor::from(vec![0i64, 0i64, self.num_heads.unwrap(), -1i64])
                    .to_dyn(),
                rng
            );
            milli_graph::ops::Reshape::push_new(&mut graph, data, new_shape, false, rng)
        } else {
            // 4D path: [B, H, S, D] -> [B, S, H, D]
            milli_graph::ops::Transpose::push_new(&mut graph, data, Some(vec![0, 2, 1, 3]), rng)
        };

        // Determine rotary dimension handling
        // If rotary_embedding_dim == 0 => use full head size.
        let use_partial = self.rotary_embedding_dim > 0;

        // Build 1D vector constants for slice parameters
        let starts0v = milli_graph::ops::Constant::push_new(
            &mut graph,
            NDArrayNumericTensor::from(vec![0i64]).to_dyn(),
            rng
        );
        let axeslastv = milli_graph::ops::Constant::push_new(
            &mut graph,
            NDArrayNumericTensor::from(vec![-1i64]).to_dyn(),
            rng
        );
        let endsrotv = if use_partial {
            milli_graph::ops::Constant::push_new(
                &mut graph,
                NDArrayNumericTensor::from(vec![self.rotary_embedding_dim]).to_dyn(),
                rng
            )
        } else {
            milli_graph::ops::Constant::push_new(
                &mut graph,
                NDArrayNumericTensor::from(vec![i64::MAX]).to_dyn(),
                rng
            )
        };
        let x_rotate = milli_graph::ops::Slice::push_new(
            &mut graph,
            prepared,
            starts0v,
            endsrotv,
            None,
            Some(axeslastv),
            rng
        );

        // x_not_rotate: from rotary_dim to end (empty if using full)
        let startsrotv = if use_partial {
            milli_graph::ops::Constant::push_new(
                &mut graph,
                NDArrayNumericTensor::from(vec![self.rotary_embedding_dim]).to_dyn(),
                rng
            )
        } else {
            milli_graph::ops::Constant::push_new(
                &mut graph,
                NDArrayNumericTensor::from(vec![i64::MAX]).to_dyn(),
                rng
            )
        };
        let endsbigv = milli_graph::ops::Constant::push_new(
            &mut graph,
            NDArrayNumericTensor::from(vec![i64::MAX]).to_dyn(),
            rng
        );
        let x_not_rotate = milli_graph::ops::Slice::push_new(
            &mut graph,
            prepared,
            startsrotv,
            endsbigv,
            None,
            Some(axeslastv),
            rng
        );

        // Prepare cos/sin caches
        let mut cos = if let Some(pid) = pos_ids {
            milli_graph::ops::Gather::push_new(&mut graph, cos_cache, pid, 0, rng)
        } else {
            cos_cache
        };
        let mut sin = if let Some(pid) = pos_ids {
            milli_graph::ops::Gather::push_new(&mut graph, sin_cache, pid, 0, rng)
        } else {
            sin_cache
        };
        // If position_ids is not provided, caches may be [B, S, D/2] or [S, D/2].
        // Do not add an extra batch dimension; rely on broadcasting after unsqueezing axis=2 below to match ONNX reference behavior.

        if use_partial {
            // Slice cos/sin to rotary_dim/2 on last axis
            let half = self.rotary_embedding_dim / 2;
            let endshalfv = milli_graph::ops::Constant::push_new(
                &mut graph,
                NDArrayNumericTensor::from(vec![half]).to_dyn(),
                rng
            );
            let cos_s = milli_graph::ops::Slice::push_new(
                &mut graph,
                cos,
                starts0v,
                endshalfv,
                None,
                Some(axeslastv),
                rng
            );
            let sin_s = milli_graph::ops::Slice::push_new(
                &mut graph,
                sin,
                starts0v,
                endshalfv,
                None,
                Some(axeslastv),
                rng
            );
            cos = cos_s;
            sin = sin_s;
        }

        // Unsqueeze cos/sin at axis=2 to broadcast: [B, S, 1, D/2]
        let axis2 = milli_graph::ops::Constant::push_new(
            &mut graph,
            NDArrayNumericTensor::from(vec![2i64]).to_dyn(),
            rng
        );
        let cos = milli_graph::ops::Unsqueeze::push_new(&mut graph, cos, axis2, rng);
        let sin = milli_graph::ops::Unsqueeze::push_new(&mut graph, sin, axis2, rng);

        // Split x_rotate into x1 and x2 depending on interleaving
        let (x1, x2) = if self.interleaved {
            // Reshape [..., D] -> [..., -1, 2]
            let new_shape = milli_graph::ops::Constant::push_new(
                &mut graph,
                NDArrayNumericTensor::from(vec![0i64, 0i64, 0i64, -1i64, 2i64]).to_dyn(),
                rng
            );
            let xr5 = milli_graph::ops::Reshape::push_new(&mut graph, x_rotate, new_shape, false, rng);
            // Split last axis [1,1]
            let split_sizes = milli_graph::ops::Constant::push_new(
                &mut graph,
                NDArrayNumericTensor::from(vec![1i64, 1i64]).to_dyn(),
                rng
            );
            let x1u = milli_graph::ops::Split::push_new(
                &mut graph,
                xr5,
                Some(milli_graph::ops::MilliOpTensorIDOrLiteral::TensorID(
                    split_sizes,
                )),
                -1,
                None,
                0,
                rng
            );
            let x2u = milli_graph::ops::Split::push_new(
                &mut graph,
                xr5,
                Some(milli_graph::ops::MilliOpTensorIDOrLiteral::TensorID(
                    split_sizes,
                )),
                -1,
                None,
                1,
                rng
            );
            let ax_last = milli_graph::ops::Constant::push_new(
                &mut graph,
                NDArrayNumericTensor::from(vec![-1i64]).to_dyn(),
                rng
            );
            let x1 = milli_graph::ops::Squeeze::push_new(&mut graph, x1u, ax_last, rng);
            let x2 = milli_graph::ops::Squeeze::push_new(&mut graph, x2u, ax_last, rng);
            (x1, x2)
        } else {
            // Split last axis [D/2, D/2]
            if use_partial {
                let half = self.rotary_embedding_dim / 2;
                let split_sizes = milli_graph::ops::Constant::push_new(
                    &mut graph,
                    NDArrayNumericTensor::from(vec![half, half]).to_dyn(),
                    rng
                );
                let x1 = milli_graph::ops::Split::push_new(
                    &mut graph,
                    x_rotate,
                    Some(milli_graph::ops::MilliOpTensorIDOrLiteral::TensorID(
                        split_sizes,
                    )),
                    -1,
                    None,
                    0,
                    rng
                );
                let x2 = milli_graph::ops::Split::push_new(
                    &mut graph,
                    x_rotate,
                    Some(milli_graph::ops::MilliOpTensorIDOrLiteral::TensorID(
                        split_sizes,
                    )),
                    -1,
                    None,
                    1,
                    rng
                );
                (x1, x2)
            } else {
                // For full rotation and non-interleaved: split into contiguous halves.
                // Reshape [..., D] -> [..., 2, D/2], then split axis -2 and squeeze it.
                let new_shape = milli_graph::ops::Constant::push_new(
                    &mut graph,
                    NDArrayNumericTensor::from(vec![0i64, 0i64, 0i64, 2i64, -1i64]).to_dyn(),
                    rng
                );
                let xr5 =
                    milli_graph::ops::Reshape::push_new(&mut graph, x_rotate, new_shape, false, rng);
                let split_sizes = milli_graph::ops::Constant::push_new(
                    &mut graph,
                    NDArrayNumericTensor::from(vec![1i64, 1i64]).to_dyn(),
                    rng
                );
                let x1u = milli_graph::ops::Split::push_new(
                    &mut graph,
                    xr5,
                    Some(milli_graph::ops::MilliOpTensorIDOrLiteral::TensorID(
                        split_sizes,
                    )),
                    -2,
                    None,
                    0,
                    rng
                );
                let x2u = milli_graph::ops::Split::push_new(
                    &mut graph,
                    xr5,
                    Some(milli_graph::ops::MilliOpTensorIDOrLiteral::TensorID(
                        split_sizes,
                    )),
                    -2,
                    None,
                    1,
                    rng
                );
                let ax_minus2 = milli_graph::ops::Constant::push_new(
                    &mut graph,
                    NDArrayNumericTensor::from(vec![-2i64]).to_dyn(),
                    rng
                );
                let x1 = milli_graph::ops::Squeeze::push_new(&mut graph, x1u, ax_minus2, rng);
                let x2 = milli_graph::ops::Squeeze::push_new(&mut graph, x2u, ax_minus2, rng);
                (x1, x2)
            }
        };

        // real = cos*x1 - sin*x2; imag = sin*x1 + cos*x2
        let cos_x1 = milli_graph::ops::SimpleBinary::mul(&mut graph, cos, x1, rng);
        let sin_x2 = milli_graph::ops::SimpleBinary::mul(&mut graph, sin, x2, rng);
        let real = milli_graph::ops::SimpleBinary::sub(&mut graph, cos_x1, sin_x2, rng);
        let sin_x1 = milli_graph::ops::SimpleBinary::mul(&mut graph, sin, x1, rng);
        let cos_x2 = milli_graph::ops::SimpleBinary::mul(&mut graph, cos, x2, rng);
        let imag = milli_graph::ops::SimpleBinary::add(&mut graph, sin_x1, cos_x2, rng);

        // Reassemble rotated part
        let rotated = if self.interleaved {
            let ax_last = milli_graph::ops::Constant::push_new(
                &mut graph,
                NDArrayNumericTensor::from(vec![-1i64]).to_dyn(),
                rng
            );
            let real_u = milli_graph::ops::Unsqueeze::push_new(&mut graph, real, ax_last, rng);
            let imag_u = milli_graph::ops::Unsqueeze::push_new(&mut graph, imag, ax_last, rng);
            let stacked = milli_graph::ops::Concat::push_new(&mut graph, vec![real_u, imag_u], -1, rng);
            let new_shape = milli_graph::ops::Constant::push_new(
                &mut graph,
                NDArrayNumericTensor::from(vec![0i64, 0i64, 0i64, -1i64]).to_dyn(),
                rng
            );
            milli_graph::ops::Reshape::push_new(&mut graph, stacked, new_shape, false, rng)
        } else {
            milli_graph::ops::Concat::push_new(&mut graph, vec![real, imag], -1, rng)
        };

        // Combine with non-rotated tail
        let combined =
            milli_graph::ops::Concat::push_new(&mut graph, vec![rotated, x_not_rotate], -1, rng);

        // Return to original layout
        let out = if self.num_heads.is_some() {
            let data_shape = milli_graph::ops::Shape::push_new(&mut graph, data, rng);
            milli_graph::ops::Reshape::push_new(&mut graph, combined, data_shape, false, rng)
        } else {
            milli_graph::ops::Transpose::push_new(&mut graph, combined, Some(vec![0, 2, 1, 3]), rng)
        };

        let mut output_map = HashMap::new();
        output_map.insert(out, self.data_output);
        graph.set_output_map(output_map);
        graph
    }
}
