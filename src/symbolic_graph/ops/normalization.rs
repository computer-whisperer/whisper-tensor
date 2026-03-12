use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::dtype::DType;
use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::{MilliLoweringContext, MilliOpGraph, ops_helpers};
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
        x_tid = milli_graph::ops::Cast::push_new(&mut graph, x_tid, DType::F32, rng);
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
        let input_cast_tid = milli_graph::ops::Cast::push_new(&mut graph, input, DType::F32, rng);
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
    stash_type: DType,
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
            DType::F32
        } else {
            DType::BF16
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
            let new_shape_tensor =
                NDArrayNumericTensor::from(vec![0i64, self.num_groups as i64, -1]);
            let new_shape =
                milli_graph::ops::Constant::push_new(&mut graph, new_shape_tensor.to_dyn(), rng);
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
            Property::new("stash_type", PropertyValue::DType(self.stash_type)),
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
    stash_type: DType,
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
            stash_type: DType::F32,
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
            DType::F32
        } else {
            DType::BF16
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

        let input_scale_cast =
            milli_graph::ops::Cast::push_new(&mut graph, input_scale, self.stash_type, rng);
        let out =
            milli_graph::ops::SimpleBinary::mul(&mut graph, normalized, input_scale_cast, rng);

        let out = if let Some(bias) = self.bias {
            let bias_cast = milli_graph::ops::Cast::push_new(
                &mut graph,
                input_map[&bias],
                self.stash_type,
                rng,
            );
            milli_graph::ops::SimpleBinary::add(&mut graph, out, bias_cast, rng)
        } else {
            out
        };

        let out = milli_graph::ops::CastLike::push_new(&mut graph, out, input_data, rng);

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }

    fn parameters(&self) -> Vec<Property> {
        vec![
            Property::new("axis", PropertyValue::Int(self.axis)),
            Property::new("epsilon", PropertyValue::Float(self.epsilon as f64)),
            Property::new("stash_type", PropertyValue::DType(self.stash_type)),
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
    stash_type: DType,
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
            stash_type: DType::F32,
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
            DType::F32
        } else {
            DType::BF16
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

        let input_scale_cast =
            milli_graph::ops::Cast::push_new(&mut graph, input_scale, self.stash_type, rng);
        let out =
            milli_graph::ops::SimpleBinary::mul(&mut graph, normalized, input_scale_cast, rng);

        let out = if let Some(bias) = self.bias {
            let bias_cast = milli_graph::ops::Cast::push_new(
                &mut graph,
                input_map[&bias],
                self.stash_type,
                rng,
            );
            milli_graph::ops::SimpleBinary::add(&mut graph, out, bias_cast, rng)
        } else {
            out
        };

        let out = milli_graph::ops::CastLike::push_new(&mut graph, out, input_data, rng);

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
            Property::new("stash_type", PropertyValue::DType(self.stash_type)),
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
            milli_graph::ops::Cast::push_new(&mut graph, original_input, DType::F32, rng);

        // Save original shape for reshaping back later
        let input_shape = milli_graph::ops::Shape::push_new(&mut graph, input_cast, rng);

        // Reshape [N, C, D1, D2, ...] → [N, C, -1]
        let reshaped_input = {
            let new_shape_tensor = NDArrayNumericTensor::from(vec![0i64, 0i64, -1]);
            let new_shape =
                milli_graph::ops::Constant::push_new(&mut graph, new_shape_tensor.to_dyn(), rng);
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
                DType::F32,
                rng,
            );
            let scale = milli_graph::ops::Unsqueeze::push_new(&mut graph, scale_cast, one, rng);
            milli_graph::ops::SimpleBinary::mul(&mut graph, normalized, scale, rng)
        };

        let y = {
            let bias_cast = milli_graph::ops::Cast::push_new(
                &mut graph,
                input_map[&self.bias],
                DType::F32,
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
