use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::{MilliLoweringContext, MilliOpGraph, ops_helpers};
use crate::numeric_dtype::NumericDType;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{ONNXDecodingError, query_attribute_float, query_attribute_int};
use crate::{milli_graph, onnx};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LpNormalizationOperation {
    global_id: GlobalId,
    input: GlobalId,
    output: GlobalId,
    axis: i64,
    p: i64,
}

impl LpNormalizationOperation {
    pub fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("LpNormalization"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("LpNormalization"));
        }
        let mut axis = -1;
        let mut p = 2;
        for attr in attributes {
            match attr.name.as_str() {
                "axis" => axis = attr.i,
                "p" => p = attr.i,
                _ => {}
            }
        }
        match p {
            1 | 2 => {}
            _ => return Err(ONNXDecodingError::InvalidOperatorInputs("LpNormalization")),
        }

        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("LpNormalization"))?,
            output: outputs[0]
                .ok_or(ONNXDecodingError::InvalidOperatorOutputs("LpNormalization"))?,
            axis,
            p,
        })
    }
}

impl Node for LpNormalizationOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "LpNormalization".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}
impl Operation for LpNormalizationOperation {
    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let input = input_map[&self.input];

        // abs(input)
        let abs_tid = milli_graph::ops::SimpleUnaryOp::abs(&mut graph, input, rng);

        let mut x_tid = match self.p {
            1 => abs_tid,
            2 => milli_graph::ops::SimpleBinary::mul(&mut graph, input, input, rng),
            _ => panic!(),
        };
        let axis_tid = ops_helpers::scalar_const(&mut graph, self.axis, rng);
        x_tid = milli_graph::ops::Cast::push_new(&mut graph, x_tid, NumericDType::F32, rng);
        x_tid = milli_graph::ops::ReduceSum::push_new(
            &mut graph,
            x_tid,
            Some(axis_tid),
            true,
            false,
            rng,
        );
        if self.p == 2 {
            x_tid = milli_graph::ops::SimpleUnaryOp::sqrt(&mut graph, x_tid, rng);
        }
        let input_cast_tid =
            milli_graph::ops::Cast::push_new(&mut graph, input, NumericDType::F32, rng);
        let out_tid = milli_graph::ops::SimpleBinary::div(&mut graph, input_cast_tid, x_tid, rng);
        let out = milli_graph::ops::CastLike::push_new(&mut graph, out_tid, input, rng);

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }

    fn parameters(&self) -> Vec<Property> {
        vec![
            Property::new("axis", PropertyValue::Int(self.axis)),
            Property::new("p", PropertyValue::Int(self.p)),
        ]
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GroupNormalizationOperation {
    global_id: GlobalId,
    input: GlobalId,
    scale: GlobalId,
    bias: GlobalId,
    output: GlobalId,
    epsilon: f32,
    num_groups: usize,
    stash_type: NumericDType,
}

impl GroupNormalizationOperation {
    pub fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 3 {
            return Err(ONNXDecodingError::InvalidOperatorInputs(
                "GroupNormalization",
            ));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs(
                "GroupNormalization",
            ));
        }
        let mut epsilon = 1e-5;
        let mut num_groups = None;
        for attr in attributes {
            match attr.name.as_str() {
                "epsilon" => epsilon = attr.f,
                "num_groups" => num_groups = Some(attr.i),
                _ => {}
            }
        }
        let stash_type = if query_attribute_int(attributes, "stash_type").unwrap_or(1) == 1 {
            NumericDType::F32
        } else {
            NumericDType::BF16
        };
        let num_groups = num_groups.ok_or(ONNXDecodingError::MissingAttribute(
            "GroupNormalization".to_string(),
            "num_groups".to_string(),
        ))? as usize;
        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                "GroupNormalization",
            ))?,
            scale: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                "GroupNormalization",
            ))?,
            bias: inputs[2].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                "GroupNormalization",
            ))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs(
                "GroupNormalization",
            ))?,
            epsilon,
            num_groups,
            stash_type,
        })
    }
}

impl Node for GroupNormalizationOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "GroupNormalization".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new([self.input, self.scale, self.bias].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for GroupNormalizationOperation {
    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let original_input = input_map[&self.input];
        let input_cast =
            milli_graph::ops::Cast::push_new(&mut graph, original_input, self.stash_type, rng);

        let input_shape = milli_graph::ops::Shape::push_new(&mut graph, input_cast, rng);
        let num_channels = {
            let starts = ops_helpers::scalar_const(&mut graph, 1i64, rng);
            let ends = ops_helpers::scalar_const(&mut graph, 2i64, rng);
            milli_graph::ops::Slice::push_new(
                &mut graph,
                input_shape,
                starts,
                ends,
                None,
                None,
                rng,
            )
        };
        let reshaped_input = {
            let new_shape = milli_graph::ops::Constant::from_vec(
                &mut graph,
                vec![0i64, self.num_groups as i64, -1],
                rng,
            );
            milli_graph::ops::Reshape::push_new(&mut graph, input_cast, new_shape, false, rng)
        };

        let mean_axis = ops_helpers::scalar_const(&mut graph, 2i64, rng);
        let mean = milli_graph::ops::ReduceMean::push_new(
            &mut graph,
            reshaped_input,
            Some(mean_axis),
            true,
            false,
            rng,
        );

        let input = milli_graph::ops::SimpleBinary::sub(&mut graph, reshaped_input, mean, rng);

        let variance = {
            let x = milli_graph::ops::SimpleBinary::mul(&mut graph, input, input, rng);
            milli_graph::ops::ReduceMean::push_new(&mut graph, x, Some(mean_axis), true, false, rng)
        };

        let input_normalized = {
            let epsilon = milli_graph::ops::Constant::new_scalar(&mut graph, self.epsilon, rng);
            let epsilon = milli_graph::ops::CastLike::push_new(&mut graph, epsilon, variance, rng);
            let var_plus_eps =
                milli_graph::ops::SimpleBinary::add(&mut graph, variance, epsilon, rng);
            let val = milli_graph::ops::SimpleUnaryOp::sqrt(&mut graph, var_plus_eps, rng);
            milli_graph::ops::SimpleBinary::div(&mut graph, input, val, rng)
        };

        let zero = milli_graph::ops::Constant::new_scalar(&mut graph, 0i64, rng);
        let neg_one = milli_graph::ops::Constant::new_scalar(&mut graph, -1i64, rng);
        let one = milli_graph::ops::Constant::new_scalar(&mut graph, 1i64, rng);

        let y = {
            let new_shape = milli_graph::ops::Concat::push_new(
                &mut graph,
                vec![zero, num_channels, neg_one],
                0,
                rng,
            );
            milli_graph::ops::Reshape::push_new(&mut graph, input_normalized, new_shape, false, rng)
        };

        let y = {
            let scale_cast = milli_graph::ops::Cast::push_new(
                &mut graph,
                input_map[&self.scale],
                self.stash_type,
                rng,
            );
            let scale = milli_graph::ops::Unsqueeze::push_new(&mut graph, scale_cast, one, rng);
            milli_graph::ops::SimpleBinary::mul(&mut graph, y, scale, rng)
        };

        let y = {
            let bias_cast = milli_graph::ops::Cast::push_new(
                &mut graph,
                input_map[&self.bias],
                self.stash_type,
                rng,
            );
            let bias = milli_graph::ops::Unsqueeze::push_new(&mut graph, bias_cast, one, rng);
            milli_graph::ops::SimpleBinary::add(&mut graph, y, bias, rng)
        };

        let out = milli_graph::ops::Reshape::push_new(&mut graph, y, input_shape, false, rng);

        let out = milli_graph::ops::CastLike::push_new(&mut graph, out, original_input, rng);

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }

    fn parameters(&self) -> Vec<Property> {
        vec![
            Property::new("epsilon", PropertyValue::Float(self.epsilon as f64)),
            Property::new("num_groups", PropertyValue::Int(self.num_groups as i64)),
            Property::new(
                "stash_type",
                PropertyValue::DType(crate::numeric_dtype::ONNXDType::Numeric(self.stash_type)),
            ),
        ]
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RMSNormalizationOperation {
    global_id: GlobalId,
    input: GlobalId,
    scale: GlobalId,
    bias: Option<GlobalId>,
    output: GlobalId,
    mean_output: Option<GlobalId>,
    inv_std_dev_output: Option<GlobalId>,
    axis: i64,
    epsilon: f32,
    stash_type: NumericDType,
}

impl RMSNormalizationOperation {
    pub fn new(
        input: GlobalId,
        scale: GlobalId,
        bias: Option<GlobalId>,
        output: GlobalId,
        epsilon: f32,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            input,
            scale,
            bias,
            output,
            mean_output: None,
            inv_std_dev_output: None,
            axis: -1,
            epsilon,
            stash_type: NumericDType::F32,
        }
    }

    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() < 2 || inputs.len() > 3 {
            return Err(ONNXDecodingError::InvalidOperatorInputs(
                "LayerNormalization",
            ));
        }
        if outputs.is_empty() || outputs.len() > 3 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs(
                "LayerNormalization",
            ));
        }
        let mut axis = -1;
        let mut epsilon = 1e-5;
        for attribute in attributes {
            match attribute.name.as_str() {
                "axis" => {
                    axis = attribute.i;
                }
                "epsilon" => {
                    epsilon = attribute.f;
                }
                _ => {}
            }
        }
        let stash_type = if query_attribute_int(attributes, "stash_type").unwrap_or(1) == 1 {
            NumericDType::F32
        } else {
            NumericDType::BF16
        };
        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                "LayerNormalization",
            ))?,
            scale: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                "LayerNormalization",
            ))?,
            bias: if inputs.len() == 3 {
                Some(inputs[2].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                    "LayerNormalization",
                ))?)
            } else {
                None
            },
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                "LayerNormalization",
            ))?,
            mean_output: if outputs.len() > 1 {
                Some(outputs[1].ok_or(ONNXDecodingError::InvalidOperatorOutputs(
                    "LayerNormalization",
                ))?)
            } else {
                None
            },
            inv_std_dev_output: if outputs.len() > 2 {
                Some(outputs[2].ok_or(ONNXDecodingError::InvalidOperatorOutputs(
                    "LayerNormalization",
                ))?)
            } else {
                None
            },
            axis,
            epsilon,
            stash_type,
        })
    }
}

impl Node for RMSNormalizationOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "RMSNormalization".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        let mut v = vec![self.input, self.scale];
        if let Some(bias) = self.bias {
            v.push(bias);
        }
        Box::new(v.into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        let mut res = vec![self.output];
        if let Some(mean_output) = self.mean_output {
            res.push(mean_output);
        }
        if let Some(inv_std_dev_output) = self.inv_std_dev_output {
            res.push(inv_std_dev_output);
        }
        Box::new(res.into_iter())
    }
}

impl Operation for RMSNormalizationOperation {
    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let input_data = input_map[&self.input];
        let input_scale = input_map[&self.scale];

        let input_f32 =
            milli_graph::ops::Cast::push_new(&mut graph, input_data, self.stash_type, rng);

        let axis = ops_helpers::scalar_const(&mut graph, self.axis, rng);
        let axis = ops_helpers::resolve_axes(&mut graph, axis, input_data, rng);

        let rank_tid = ops_helpers::rank(&mut graph, input_data, rng);
        let step_tid = ops_helpers::scalar_const(&mut graph, 1i64, rng);
        let normalized_axes =
            milli_graph::ops::Range::push_new(&mut graph, axis, rank_tid, step_tid, rng);

        let input_squared =
            milli_graph::ops::SimpleBinary::mul(&mut graph, input_f32, input_f32, rng);

        let squared_mean = milli_graph::ops::ReduceMean::push_new(
            &mut graph,
            input_squared,
            Some(normalized_axes),
            true,
            false,
            rng,
        );

        let epsilon = milli_graph::ops::Constant::new_scalar(&mut graph, self.epsilon, rng);
        let epsilon = milli_graph::ops::CastLike::push_new(&mut graph, epsilon, squared_mean, rng);
        let mean_plus_eps =
            milli_graph::ops::SimpleBinary::add(&mut graph, squared_mean, epsilon, rng);
        let rms = milli_graph::ops::SimpleUnaryOp::sqrt(&mut graph, mean_plus_eps, rng);
        let rms_inv = milli_graph::ops::SimpleUnaryOp::reciprocal(&mut graph, rms, rng);

        let normalized = milli_graph::ops::SimpleBinary::mul(&mut graph, input_f32, rms_inv, rng);

        // Cast normalized result back to input dtype before scaling, matching
        // PyTorch's RMSNorm which does `weight * normalized.to(input_dtype)`.
        // This ensures the scale multiply happens in the original precision,
        // preventing accumulated rounding divergence through many layers.
        let normalized =
            milli_graph::ops::CastLike::push_new(&mut graph, normalized, input_data, rng);

        let out = milli_graph::ops::SimpleBinary::mul(&mut graph, normalized, input_scale, rng);

        let out = if let Some(bias) = self.bias {
            let bias = input_map[&bias];
            milli_graph::ops::SimpleBinary::add(&mut graph, out, bias, rng)
        } else {
            out
        };

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }

    fn parameters(&self) -> Vec<Property> {
        vec![
            Property::new("axis", PropertyValue::Int(self.axis)),
            Property::new("epsilon", PropertyValue::Float(self.epsilon as f64)),
            Property::new(
                "stash_type",
                PropertyValue::DType(crate::numeric_dtype::ONNXDType::Numeric(self.stash_type)),
            ),
        ]
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerNormalizationOperation {
    global_id: GlobalId,
    input: GlobalId,
    scale: GlobalId,
    bias: Option<GlobalId>,
    output: GlobalId,
    mean_output: Option<GlobalId>,
    inv_std_dev_output: Option<GlobalId>,
    axis: i64,
    epsilon: f32,
    stash_type: NumericDType,
}

impl LayerNormalizationOperation {
    pub fn new(
        input: GlobalId,
        scale: GlobalId,
        bias: Option<GlobalId>,
        output: GlobalId,
        epsilon: f32,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            input,
            scale,
            bias,
            output,
            mean_output: None,
            inv_std_dev_output: None,
            axis: -1,
            epsilon,
            stash_type: NumericDType::F32,
        }
    }

    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() < 2 || inputs.len() > 3 {
            return Err(ONNXDecodingError::InvalidOperatorInputs(
                "LayerNormalization",
            ));
        }
        if outputs.is_empty() || outputs.len() > 3 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs(
                "LayerNormalization",
            ));
        }
        let mut axis = -1;
        let mut epsilon = 1e-5;
        for attribute in attributes {
            match attribute.name.as_str() {
                "axis" => {
                    axis = attribute.i;
                }
                "epsilon" => {
                    epsilon = attribute.f;
                }
                _ => {}
            }
        }
        let stash_type = if query_attribute_int(attributes, "stash_type").unwrap_or(1) == 1 {
            NumericDType::F32
        } else {
            NumericDType::BF16
        };
        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                "LayerNormalization",
            ))?,
            scale: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                "LayerNormalization",
            ))?,
            bias: if inputs.len() == 3 {
                Some(inputs[2].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                    "LayerNormalization",
                ))?)
            } else {
                None
            },
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                "LayerNormalization",
            ))?,
            mean_output: if outputs.len() > 1 {
                Some(outputs[1].ok_or(ONNXDecodingError::InvalidOperatorOutputs(
                    "LayerNormalization",
                ))?)
            } else {
                None
            },
            inv_std_dev_output: if outputs.len() > 2 {
                Some(outputs[2].ok_or(ONNXDecodingError::InvalidOperatorOutputs(
                    "LayerNormalization",
                ))?)
            } else {
                None
            },
            axis,
            epsilon,
            stash_type,
        })
    }
}

impl Node for LayerNormalizationOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "LayerNormalization".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        let mut v = vec![self.input, self.scale];
        if let Some(bias) = self.bias {
            v.push(bias);
        }
        Box::new(v.into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        let mut res = vec![self.output];
        if let Some(mean_output) = self.mean_output {
            res.push(mean_output);
        }
        if let Some(inv_std_dev_output) = self.inv_std_dev_output {
            res.push(inv_std_dev_output);
        }
        Box::new(res.into_iter())
    }
}
impl Operation for LayerNormalizationOperation {
    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let input_data = input_map[&self.input];
        let input_scale = input_map[&self.scale];

        let input_f32 =
            milli_graph::ops::Cast::push_new(&mut graph, input_data, self.stash_type, rng);

        let axis = ops_helpers::scalar_const(&mut graph, self.axis, rng);
        let axis = ops_helpers::resolve_axes(&mut graph, axis, input_data, rng);

        let rank_tid = ops_helpers::rank(&mut graph, input_data, rng);
        let step_tid = ops_helpers::scalar_const(&mut graph, 1i64, rng);
        let normalized_axes =
            milli_graph::ops::Range::push_new(&mut graph, axis, rank_tid, step_tid, rng);

        let mean = milli_graph::ops::ReduceMean::push_new(
            &mut graph,
            input_f32,
            Some(normalized_axes),
            true,
            false,
            rng,
        );

        let d = milli_graph::ops::SimpleBinary::sub(&mut graph, input_f32, mean, rng);
        let dd = milli_graph::ops::SimpleBinary::mul(&mut graph, d, d, rng);
        let variance = milli_graph::ops::ReduceMean::push_new(
            &mut graph,
            dd,
            Some(normalized_axes),
            true,
            false,
            rng,
        );
        let epsilon = milli_graph::ops::Constant::new_scalar(&mut graph, self.epsilon, rng);
        let epsilon = milli_graph::ops::CastLike::push_new(&mut graph, epsilon, variance, rng);
        let var_plus_eps = milli_graph::ops::SimpleBinary::add(&mut graph, variance, epsilon, rng);
        let stddev = milli_graph::ops::SimpleUnaryOp::sqrt(&mut graph, var_plus_eps, rng);
        let inv_stddev = milli_graph::ops::SimpleUnaryOp::reciprocal(&mut graph, stddev, rng);

        let normalized = milli_graph::ops::SimpleBinary::mul(&mut graph, d, inv_stddev, rng);

        // Cast normalized result back to input dtype before scaling, matching
        // PyTorch's LayerNorm which does `weight * normalized.to(input_dtype) + bias`.
        let normalized =
            milli_graph::ops::CastLike::push_new(&mut graph, normalized, input_data, rng);

        let out = milli_graph::ops::SimpleBinary::mul(&mut graph, normalized, input_scale, rng);

        let out = if let Some(bias) = self.bias {
            let bias = input_map[&bias];
            milli_graph::ops::SimpleBinary::add(&mut graph, out, bias, rng)
        } else {
            out
        };

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        if let Some(x) = self.mean_output {
            output_map.insert(mean, x);
        }
        if let Some(x) = self.inv_std_dev_output {
            output_map.insert(inv_stddev, x);
        }
        graph.set_output_map(output_map);
        graph
    }

    fn parameters(&self) -> Vec<Property> {
        vec![
            Property::new("axis", PropertyValue::Int(self.axis)),
            Property::new("epsilon", PropertyValue::Float(self.epsilon as f64)),
            Property::new(
                "stash_type",
                PropertyValue::DType(crate::numeric_dtype::ONNXDType::Numeric(self.stash_type)),
            ),
        ]
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InstanceNormalizationOperation {
    global_id: GlobalId,
    input: GlobalId,
    scale: GlobalId,
    bias: GlobalId,
    output: GlobalId,
    epsilon: Option<f32>,
}

impl InstanceNormalizationOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 3 {
            return Err(ONNXDecodingError::InvalidOperatorInputs(
                "InstanceNormalization",
            ));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs(
                "InstanceNormalization",
            ));
        }
        let epsilon = query_attribute_float(attributes, "epsilon");

        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                "InstanceNormalization",
            ))?,
            scale: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                "InstanceNormalization",
            ))?,
            bias: inputs[2].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                "InstanceNormalization",
            ))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs(
                "InstanceNormalization",
            ))?,
            epsilon,
        })
    }
}

impl Node for InstanceNormalizationOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "InstanceNormalization".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new([self.input, self.scale, self.bias].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for InstanceNormalizationOperation {
    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        // InstanceNorm: for each (N, C), normalize over spatial dims.
        // Same structure as GroupNorm with num_groups = C.
        // Input: [N, C, D1, D2, ...] → reshape to [N, C, -1] → normalize axis 2
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let original_input = input_map[&self.input];

        let input_cast =
            milli_graph::ops::Cast::push_new(&mut graph, original_input, NumericDType::F32, rng);

        // Save original shape for reshaping back later
        let input_shape = milli_graph::ops::Shape::push_new(&mut graph, input_cast, rng);

        // Reshape [N, C, D1, D2, ...] → [N, C, -1]
        let reshaped_input = {
            let new_shape =
                milli_graph::ops::Constant::from_vec(&mut graph, vec![0i64, 0i64, -1i64], rng);
            milli_graph::ops::Reshape::push_new(&mut graph, input_cast, new_shape, false, rng)
        };

        // Mean over spatial dim (axis 2)
        let mean_axis = ops_helpers::scalar_const(&mut graph, 2i64, rng);
        let mean = milli_graph::ops::ReduceMean::push_new(
            &mut graph,
            reshaped_input,
            Some(mean_axis),
            true,
            false,
            rng,
        );

        let d = milli_graph::ops::SimpleBinary::sub(&mut graph, reshaped_input, mean, rng);

        // Variance = mean(d^2) over axis 2
        let variance = {
            let dd = milli_graph::ops::SimpleBinary::mul(&mut graph, d, d, rng);
            milli_graph::ops::ReduceMean::push_new(
                &mut graph,
                dd,
                Some(mean_axis),
                true,
                false,
                rng,
            )
        };

        // Normalize: d / sqrt(variance + epsilon)
        let normalized = {
            let eps_val = self.epsilon.unwrap_or(1e-5);
            let epsilon = milli_graph::ops::Constant::new_scalar(&mut graph, eps_val, rng);
            let epsilon = milli_graph::ops::CastLike::push_new(&mut graph, epsilon, variance, rng);
            let var_plus_eps =
                milli_graph::ops::SimpleBinary::add(&mut graph, variance, epsilon, rng);
            let stddev = milli_graph::ops::SimpleUnaryOp::sqrt(&mut graph, var_plus_eps, rng);
            milli_graph::ops::SimpleBinary::div(&mut graph, d, stddev, rng)
        };

        // Apply scale and bias while still in [N, C, -1] shape (3D),
        // so unsqueeze(scale, axis=1) gives [C, 1] which broadcasts correctly.
        let one = milli_graph::ops::Constant::new_scalar(&mut graph, 1i64, rng);
        let y = {
            let scale_cast = milli_graph::ops::Cast::push_new(
                &mut graph,
                input_map[&self.scale],
                NumericDType::F32,
                rng,
            );
            let scale = milli_graph::ops::Unsqueeze::push_new(&mut graph, scale_cast, one, rng);
            milli_graph::ops::SimpleBinary::mul(&mut graph, normalized, scale, rng)
        };

        let y = {
            let bias_cast = milli_graph::ops::Cast::push_new(
                &mut graph,
                input_map[&self.bias],
                NumericDType::F32,
                rng,
            );
            let bias = milli_graph::ops::Unsqueeze::push_new(&mut graph, bias_cast, one, rng);
            milli_graph::ops::SimpleBinary::add(&mut graph, y, bias, rng)
        };

        // Reshape back to original shape
        let y = milli_graph::ops::Reshape::push_new(&mut graph, y, input_shape, false, rng);

        let out = milli_graph::ops::CastLike::push_new(&mut graph, y, original_input, rng);

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }

    fn parameters(&self) -> Vec<Property> {
        let mut params = Vec::new();
        if let Some(eps) = self.epsilon {
            params.push(Property::new("epsilon", PropertyValue::Float(eps as f64)));
        }
        params
    }
}

/// ONNX BatchNormalization (inference mode only).
/// output = scale * (x - mean) / sqrt(var + epsilon) + bias
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BatchNormalizationOperation {
    global_id: GlobalId,
    x: GlobalId,
    scale: GlobalId,
    bias: GlobalId,
    input_mean: GlobalId,
    input_var: GlobalId,
    output: GlobalId,
    epsilon: f32,
}

impl BatchNormalizationOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 5 {
            return Err(ONNXDecodingError::InvalidOperatorInputs(
                "BatchNormalization",
            ));
        }
        if outputs.is_empty() {
            return Err(ONNXDecodingError::InvalidOperatorOutputs(
                "BatchNormalization",
            ));
        }
        Ok(Self {
            global_id: GlobalId::new(rng),
            x: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                "BatchNormalization",
            ))?,
            scale: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                "BatchNormalization",
            ))?,
            bias: inputs[2].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                "BatchNormalization",
            ))?,
            input_mean: inputs[3].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                "BatchNormalization",
            ))?,
            input_var: inputs[4].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                "BatchNormalization",
            ))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs(
                "BatchNormalization",
            ))?,
            epsilon: query_attribute_float(attributes, "epsilon").unwrap_or(1e-5),
        })
    }
}

impl Node for BatchNormalizationOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "BatchNormalization".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(
            [
                self.x,
                self.scale,
                self.bias,
                self.input_mean,
                self.input_var,
            ]
            .into_iter(),
        )
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for BatchNormalizationOperation {
    fn parameters(&self) -> Vec<Property> {
        vec![Property::new(
            "epsilon",
            PropertyValue::Float(self.epsilon as f64),
        )]
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        // BN(x) = scale * (x - mean) / sqrt(var + epsilon) + bias
        // scale, bias, mean, var are 1-D [C], x is [N, C, D1, D2, ...].
        //
        // PyTorch always computes BN in F32 regardless of input dtype,
        // matching the InstanceNorm pattern in this codebase.
        //
        // Strategy: cast to F32, reshape x to [N, C, -1], unsqueeze params
        // to [C, 1], do the math in 3D, reshape back, cast to original dtype.
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let original_x = input_map[&self.x];

        // Cast input and params to F32 for numerical stability
        let x = milli_graph::ops::Cast::push_new(&mut graph, original_x, NumericDType::F32, rng);

        // Save original shape for reshaping back
        let orig_shape = milli_graph::ops::Shape::push_new(&mut graph, x, rng);

        // Reshape [N, C, D1, D2, ...] → [N, C, -1]
        let new_shape =
            milli_graph::ops::Constant::from_vec(&mut graph, vec![0i64, 0i64, -1i64], rng);
        let x3d = milli_graph::ops::Reshape::push_new(&mut graph, x, new_shape, false, rng);

        // Unsqueeze 1-D params [C] → [C, 1] so they broadcast over the spatial dim,
        // casting each to F32.
        let axis1 = milli_graph::ops::Constant::new_scalar(&mut graph, 1i64, rng);
        let scale = milli_graph::ops::Cast::push_new(
            &mut graph,
            input_map[&self.scale],
            NumericDType::F32,
            rng,
        );
        let scale = milli_graph::ops::Unsqueeze::push_new(&mut graph, scale, axis1, rng);
        let bias = milli_graph::ops::Cast::push_new(
            &mut graph,
            input_map[&self.bias],
            NumericDType::F32,
            rng,
        );
        let bias = milli_graph::ops::Unsqueeze::push_new(&mut graph, bias, axis1, rng);
        let mean = milli_graph::ops::Cast::push_new(
            &mut graph,
            input_map[&self.input_mean],
            NumericDType::F32,
            rng,
        );
        let mean = milli_graph::ops::Unsqueeze::push_new(&mut graph, mean, axis1, rng);
        let var = milli_graph::ops::Cast::push_new(
            &mut graph,
            input_map[&self.input_var],
            NumericDType::F32,
            rng,
        );
        let var = milli_graph::ops::Unsqueeze::push_new(&mut graph, var, axis1, rng);

        let eps = milli_graph::ops::Constant::new_scalar(&mut graph, self.epsilon, rng);
        let var_eps = milli_graph::ops::SimpleBinary::add(&mut graph, var, eps, rng);
        let std_dev = milli_graph::ops::SimpleUnaryOp::sqrt(&mut graph, var_eps, rng);
        let x_norm = milli_graph::ops::SimpleBinary::sub(&mut graph, x3d, mean, rng);
        let x_norm = milli_graph::ops::SimpleBinary::div(&mut graph, x_norm, std_dev, rng);
        let scaled = milli_graph::ops::SimpleBinary::mul(&mut graph, scale, x_norm, rng);
        let y3d = milli_graph::ops::SimpleBinary::add(&mut graph, scaled, bias, rng);

        // Reshape back to original shape, cast back to original dtype
        let y = milli_graph::ops::Reshape::push_new(&mut graph, y3d, orig_shape, false, rng);
        let out_tid = milli_graph::ops::CastLike::push_new(&mut graph, y, original_x, rng);
        let mut output_map = HashMap::new();
        output_map.insert(out_tid, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GlobalId;
    use crate::numeric_dtype::NumericDType;
    use crate::numeric_scalar::NumericScalar;
    use crate::numeric_tensor::NumericTensor as PoolTensor;
    use crate::pool::SystemPool;
    use crate::tensor_rank::DynRank;
    use half::bf16;

    static POOL: SystemPool = SystemPool;

    fn make_bf16(shape: Vec<u64>, values: &[bf16]) -> PoolTensor<'static, DynRank, SystemPool> {
        let mut t = PoolTensor::zeros(shape, NumericDType::BF16, &POOL).unwrap();
        for (i, &v) in values.iter().enumerate() {
            t.write_element(i, NumericScalar::from_bf16(v));
        }
        t
    }

    fn read_f32_vec(t: &PoolTensor<'_, DynRank, impl crate::pool::Pool>) -> Vec<f32> {
        (0..t.numel()).map(|i| t.read_element(i).to_f32()).collect()
    }

    fn pool_eval_milli(
        graph: &crate::milli_graph::MilliOpGraph,
        inputs: &HashMap<GlobalId, PoolTensor<'static, DynRank, SystemPool>>,
    ) -> HashMap<GlobalId, PoolTensor<'static, DynRank, SystemPool>> {
        let views: HashMap<GlobalId, _> = inputs.iter().map(|(&id, t)| (id, t.view())).collect();
        let view_refs: HashMap<GlobalId, _> = views.iter().map(|(&id, v)| (id, v)).collect();
        graph.pool_eval(&view_refs, &POOL).unwrap()
    }

    #[test]
    fn test_rmsnorm_bf16_matches_pytorch() {
        // Test: RMSNorm([1,2,3,4], weight=[0.5,0.5,0.5,0.5], eps=1e-6)
        // PyTorch result: [0.1826171875, 0.365234375, 0.546875, 0.73046875]
        let mut rng = wyrand::WyRand::new(42);

        let x_vals: Vec<bf16> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .map(|v| bf16::from_f32(*v))
            .collect();
        let w_vals: Vec<bf16> = [0.5f32, 0.5, 0.5, 0.5]
            .iter()
            .map(|v| bf16::from_f32(*v))
            .collect();
        let expected: Vec<f32> = vec![0.1826171875, 0.365234375, 0.546875, 0.73046875];

        let x_id = GlobalId::new(&mut rng);
        let w_id = GlobalId::new(&mut rng);
        let out_id = GlobalId::new(&mut rng);

        let op = RMSNormalizationOperation {
            global_id: GlobalId::new(&mut rng),
            input: x_id,
            scale: w_id,
            bias: None,
            output: out_id,
            mean_output: None,
            inv_std_dev_output: None,
            axis: -1,
            epsilon: 1e-6,
            stash_type: NumericDType::F32,
        };

        let tensor_dtypes = HashMap::from([
            (x_id, crate::numeric_dtype::NumericDType::BF16),
            (w_id, crate::numeric_dtype::NumericDType::BF16),
        ]);
        let ctx = crate::milli_graph::MilliLoweringContext::new(tensor_dtypes);
        let milli_graph = op.get_milli_op_graph(&ctx, &mut rng);

        let mut inputs = HashMap::new();
        inputs.insert(x_id, make_bf16(vec![1, 4], &x_vals));
        inputs.insert(w_id, make_bf16(vec![4], &w_vals));

        let results = pool_eval_milli(&milli_graph, &inputs);
        let result_f32 = read_f32_vec(&results[&out_id]);

        for (i, (got, want)) in result_f32.iter().zip(expected.iter()).enumerate() {
            assert!(
                (*got - *want).abs() < 1e-6,
                "element {i}: got {got} want {want}"
            );
        }
    }

    /// Same test but rank-3 input [1, 1, 4] to match real model shapes.
    #[test]
    fn test_rmsnorm_bf16_rank3() {
        let mut rng = wyrand::WyRand::new(42);
        let x_vals: Vec<bf16> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .map(|v| bf16::from_f32(*v))
            .collect();
        let w_vals: Vec<bf16> = [0.5f32, 0.5, 0.5, 0.5]
            .iter()
            .map(|v| bf16::from_f32(*v))
            .collect();
        let expected: Vec<f32> = vec![0.1826171875, 0.365234375, 0.546875, 0.73046875];

        let x_id = GlobalId::new(&mut rng);
        let w_id = GlobalId::new(&mut rng);
        let out_id = GlobalId::new(&mut rng);

        let op = RMSNormalizationOperation {
            global_id: GlobalId::new(&mut rng),
            input: x_id,
            scale: w_id,
            bias: None,
            output: out_id,
            mean_output: None,
            inv_std_dev_output: None,
            axis: -1,
            epsilon: 1e-6,
            stash_type: NumericDType::F32,
        };

        let tensor_dtypes = HashMap::from([
            (x_id, crate::numeric_dtype::NumericDType::BF16),
            (w_id, crate::numeric_dtype::NumericDType::BF16),
        ]);
        let ctx = crate::milli_graph::MilliLoweringContext::new(tensor_dtypes);
        let milli_graph = op.get_milli_op_graph(&ctx, &mut rng);

        let mut inputs = HashMap::new();
        inputs.insert(x_id, make_bf16(vec![1, 1, 4], &x_vals));
        inputs.insert(w_id, make_bf16(vec![4], &w_vals));

        let results = pool_eval_milli(&milli_graph, &inputs);
        let result = &results[&out_id];
        assert_eq!(result.view().shape(), &[1u64, 1, 4]);
        let result_f32 = read_f32_vec(result);

        for (i, (got, want)) in result_f32.iter().zip(expected.iter()).enumerate() {
            assert!(
                (*got - *want).abs() < 1e-6,
                "element {i}: got {got} want {want}"
            );
        }
    }
}
