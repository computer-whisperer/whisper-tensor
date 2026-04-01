use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::ops as milli_ops;
use crate::milli_graph::{MilliLoweringContext, MilliOpGraph, ops_helpers};
use crate::numeric_dtype::NumericDType;
use crate::onnx::AttributeProto;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{ONNXDecodingError, query_attribute_float, query_attribute_int};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// ONNX LRN (Local Response Normalization) operator.
///
/// Y[n,c,d1,...,dk] = X[n,c,d1,...,dk] / (bias + alpha/size * sum(X[n,j,d1,...,dk]^2))^beta
/// where j ranges over max(0, c - floor((size-1)/2)) to min(C-1, c + ceil((size-1)/2))
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LrnOperation {
    global_id: GlobalId,
    input: GlobalId,
    output: GlobalId,
    alpha: f32,
    beta: f32,
    bias: f32,
    size: i64,
}

impl LrnOperation {
    pub fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.is_empty() {
            return Err(ONNXDecodingError::InvalidOperatorInputs("LRN"));
        }
        if outputs.is_empty() {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("LRN"));
        }

        let size = query_attribute_int(attributes, "size")
            .ok_or(ONNXDecodingError::MissingField("size"))?;

        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("LRN"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("LRN"))?,
            alpha: query_attribute_float(attributes, "alpha").unwrap_or(0.0001),
            beta: query_attribute_float(attributes, "beta").unwrap_or(0.75),
            bias: query_attribute_float(attributes, "bias").unwrap_or(1.0),
            size,
        })
    }
}

impl Node for LrnOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "LRN".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for LrnOperation {
    fn parameters(&self) -> Vec<Property> {
        vec![
            Property::new("alpha", PropertyValue::Float(self.alpha.into())),
            Property::new("beta", PropertyValue::Float(self.beta.into())),
            Property::new("bias", PropertyValue::Float(self.bias.into())),
            Property::new("size", PropertyValue::Int(self.size)),
        ]
    }

    fn is_differentiable(&self) -> bool {
        false
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let x = input_map[&self.input];

        // Cast to F32 for computation.
        let x_f32 = milli_ops::Cast::push_new(&mut graph, x, NumericDType::F32, rng);

        // x_squared = x * x
        let x_sq = milli_ops::SimpleBinary::mul(&mut graph, x_f32, x_f32, rng);

        // Pad x_squared along channel axis (axis 1) with zeros.
        // pad_before = floor((size-1)/2), pad_after = ceil((size-1)/2)
        let pad_before = (self.size - 1) / 2;
        let pad_after = self.size - 1 - pad_before;

        // Build pad vector dynamically: [0, pad_before, 0, ..., 0, pad_after, 0, ...]
        // Pad format: [begin_0, begin_1, ..., begin_rank, end_0, end_1, ..., end_rank]
        // We only pad axis 1. Use Shape to get rank, build accordingly.
        let rank = ops_helpers::rank(&mut graph, x_sq, rng);
        let zero_scalar = ops_helpers::scalar_const(&mut graph, 0i64, rng);
        let one_scalar = ops_helpers::scalar_const(&mut graph, 1i64, rng);

        // zeros_before_ch = ConstantOfShape([1], value=0) — 1 zero for axis 0
        let pad_b_val = ops_helpers::scalar_const(&mut graph, pad_before, rng);
        let pad_e_val = ops_helpers::scalar_const(&mut graph, pad_after, rng);
        // rank_minus_2 = rank - 2 (number of spatial dims)
        let two = ops_helpers::scalar_const(&mut graph, 2i64, rng);
        let n_trailing = milli_ops::SimpleBinary::sub(&mut graph, rank, two, rng);
        let trailing_zeros = milli_ops::ConstantOfShape::push_new(
            &mut graph,
            crate::numeric_scalar::NumericScalar::from_i64(0),
            n_trailing,
            rng,
        );
        // pad_tensor = [0, pad_before, 0...0, 0, pad_after, 0...0]
        let pad_tensor = milli_ops::Concat::push_new(
            &mut graph,
            vec![
                zero_scalar,
                pad_b_val,
                trailing_zeros,
                zero_scalar,
                pad_e_val,
                trailing_zeros,
            ],
            0,
            rng,
        );

        let zero_f32 = milli_ops::Constant::new_scalar(&mut graph, 0.0f32, rng);
        let padded = milli_ops::Pad::push_new(
            &mut graph,
            x_sq,
            pad_tensor,
            Some(zero_f32),
            None,
            milli_ops::PadMode::Constant,
            rng,
        );

        // Sum `size` shifted slices along the channel axis.
        // For each k in 0..size: slice channels [k, k+C) from padded.
        // C = number of channels in the original input.
        let x_shape = milli_ops::Shape::push_new(&mut graph, x_f32, rng);
        let c_dim = milli_ops::Gather::push_new(&mut graph, x_shape, one_scalar, 0, rng);
        let axes_1 = milli_ops::Constant::from_vec(&mut graph, vec![1i64], rng);

        let mut channel_sum: Option<GlobalId> = None;
        for k in 0..self.size {
            let start = milli_ops::Constant::from_vec(&mut graph, vec![k], rng);
            let end = milli_ops::SimpleBinary::add(&mut graph, start, c_dim, rng);
            let tap =
                milli_ops::Slice::push_new(&mut graph, padded, start, end, None, Some(axes_1), rng);
            channel_sum = Some(match channel_sum {
                None => tap,
                Some(acc) => milli_ops::SimpleBinary::add(&mut graph, acc, tap, rng),
            });
        }

        // scale = bias + (alpha / size) * channel_sum
        let alpha_over_size =
            milli_ops::Constant::new_scalar(&mut graph, self.alpha / self.size as f32, rng);
        let bias_const = milli_ops::Constant::new_scalar(&mut graph, self.bias, rng);
        let scaled_sum =
            milli_ops::SimpleBinary::mul(&mut graph, alpha_over_size, channel_sum.unwrap(), rng);
        let scale = milli_ops::SimpleBinary::add(&mut graph, bias_const, scaled_sum, rng);

        // y = x / scale^beta
        let beta_const = milli_ops::Constant::new_scalar(&mut graph, self.beta, rng);
        let scale_pow = milli_ops::Pow::push_new(&mut graph, scale, beta_const, rng);
        let result_f32 = milli_ops::SimpleBinary::div(&mut graph, x_f32, scale_pow, rng);

        let result = milli_ops::CastLike::push_new(&mut graph, result_f32, x, rng);

        let mut output_map = HashMap::new();
        output_map.insert(result, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
