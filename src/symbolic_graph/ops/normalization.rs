use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::dtype::DType;
use crate::milli_graph::ops::*;
use crate::milli_graph::{MilliOpGraph, ops_helpers};
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{
    ONNXDecodingError, SymbolicGraphTensorId, query_attribute_float, query_attribute_int,
};
use crate::{DynRank, onnx};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LpNormalizationOperation {
    input: SymbolicGraphTensorId,
    output: SymbolicGraphTensorId,
    axis: i64,
    p: i64,
}

impl LpNormalizationOperation {
    pub fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        attributes: &[onnx::AttributeProto],
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
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("LpNormalization"))?,
            output: outputs[0]
                .ok_or(ONNXDecodingError::InvalidOperatorOutputs("LpNormalization"))?,
            axis,
            p,
        })
    }
}

impl Operation for LpNormalizationOperation {
    fn get_op_type_name(&self) -> String {
        "LpNormalization".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.input]
    }
    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let input = input_map[&self.input];

        let x = graph.push_op(AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::abs(input)));

        let x = match self.p {
            1 => x,
            2 => graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::mul(
                input, input,
            ))),
            _ => panic!(),
        };
        let axis_tensor = NDArrayNumericTensor::from(vec![self.axis])
            .try_to_rank::<DynRank>()
            .unwrap();
        let axis = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new(axis_tensor)));
        let x = graph.push_op(AnyMilliOp::Cast(MilliOpCast::new(x, DType::F32)));
        let x = graph.push_op(AnyMilliOp::ReduceSum(MilliOpReduceSum::new(
            x,
            Some(axis),
            true,
            false,
        )));
        let x = match self.p {
            1 => x,
            2 => graph.push_op(AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::sqrt(x))),
            _ => panic!(),
        };
        let input_cast = graph.push_op(AnyMilliOp::Cast(MilliOpCast::new(input, DType::F32)));
        let out = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::div(
            input_cast, x,
        )));
        let out = graph.push_op(AnyMilliOp::CastLike(MilliOpCastLike::new(out, input)));

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GroupNormalizationOperation {
    input: SymbolicGraphTensorId,
    scale: SymbolicGraphTensorId,
    bias: SymbolicGraphTensorId,
    output: SymbolicGraphTensorId,
    epsilon: f32,
    num_groups: usize,
    stash_type: DType,
}

impl GroupNormalizationOperation {
    pub fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        attributes: &[onnx::AttributeProto],
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

impl Operation for GroupNormalizationOperation {
    fn get_op_type_name(&self) -> String {
        "GroupNormalization".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.input, self.scale, self.bias]
    }
    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let original_input = input_map[&self.input];
        let input_cast = graph.push_op(AnyMilliOp::Cast(MilliOpCast::new(
            original_input,
            self.stash_type,
        )));

        let input_shape = graph.push_op(AnyMilliOp::Shape(MilliOpShape::new(input_cast)));
        let num_channels = {
            let starts = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(1i64)));
            let ends = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(2i64)));
            graph.push_op(AnyMilliOp::Slice(MilliOpSlice::new(
                input_shape,
                starts,
                ends,
                None,
                None,
            )))
        };
        let reshaped_input = {
            let new_shape_tensor =
                NDArrayNumericTensor::from(vec![0i64, self.num_groups as i64, -1]);
            let new_shape = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new(
                new_shape_tensor.to_dyn(),
            )));
            graph.push_op(AnyMilliOp::Reshape(MilliOpReshape::new(
                input_cast, new_shape, false,
            )))
        };

        let mean_axis = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(2i64)));
        let mean = graph.push_op(AnyMilliOp::ReduceMean(MilliOpReduceMean::new(
            reshaped_input,
            Some(mean_axis),
            true,
            false,
        )));

        let input = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::sub(
            reshaped_input,
            mean,
        )));

        let variance = {
            let x = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::mul(
                input, input,
            )));
            graph.push_op(AnyMilliOp::ReduceMean(MilliOpReduceMean::new(
                x,
                Some(mean_axis),
                true,
                false,
            )))
        };

        let input_normalized = {
            let epsilon = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(
                self.epsilon,
            )));
            let epsilon = graph.push_op(AnyMilliOp::CastLike(MilliOpCastLike::new(
                epsilon, variance,
            )));
            let var_plus_eps = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::add(
                variance, epsilon,
            )));
            let val = graph.push_op(AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::sqrt(
                var_plus_eps,
            )));
            graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::div(
                input, val,
            )))
        };

        let zero = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(0i64)));
        let neg_one = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(-1i64)));
        let one = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(1i64)));

        let y = {
            let new_shape = graph.push_op(AnyMilliOp::Concat(MilliOpConcat::new(
                vec![zero, num_channels, neg_one],
                0,
            )));
            graph.push_op(AnyMilliOp::Reshape(MilliOpReshape::new(
                input_normalized,
                new_shape,
                false,
            )))
        };

        let y = {
            let scale_cast = graph.push_op(AnyMilliOp::Cast(MilliOpCast::new(
                input_map[&self.scale],
                self.stash_type,
            )));
            let scale = graph.push_op(AnyMilliOp::Unsqueeze(MilliOpUnsqueeze::new(
                scale_cast, one,
            )));
            graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::mul(y, scale)))
        };

        let y = {
            let bias_cast = graph.push_op(AnyMilliOp::Cast(MilliOpCast::new(
                input_map[&self.bias],
                self.stash_type,
            )));
            let bias = graph.push_op(AnyMilliOp::Unsqueeze(MilliOpUnsqueeze::new(bias_cast, one)));
            graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::add(y, bias)))
        };

        let out = graph.push_op(AnyMilliOp::Reshape(MilliOpReshape::new(
            y,
            input_shape,
            false,
        )));

        let out = graph.push_op(AnyMilliOp::CastLike(MilliOpCastLike::new(
            out,
            original_input,
        )));

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RMSNormalizationOperation {
    input: SymbolicGraphTensorId,
    scale: SymbolicGraphTensorId,
    bias: Option<SymbolicGraphTensorId>,
    output: SymbolicGraphTensorId,
    mean_output: Option<SymbolicGraphTensorId>,
    inv_std_dev_output: Option<SymbolicGraphTensorId>,
    axis: i64,
    epsilon: f32,
    stash_type: DType,
}

impl RMSNormalizationOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        attributes: &[onnx::AttributeProto],
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

impl Operation for RMSNormalizationOperation {
    fn get_op_type_name(&self) -> String {
        "RMS Normalization".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        let mut v = vec![self.input, self.scale];
        if let Some(bias) = self.bias {
            v.push(bias);
        }
        v
    }
    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        let mut res = vec![self.output];
        if let Some(mean_output) = self.mean_output {
            res.push(mean_output);
        }
        if let Some(inv_std_dev_output) = self.inv_std_dev_output {
            res.push(inv_std_dev_output);
        }
        res
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let input_data = input_map[&self.input];
        let input_scale = input_map[&self.scale];

        let input_f32 = graph.push_op(AnyMilliOp::Cast(MilliOpCast::new(
            input_data,
            self.stash_type,
        )));

        let axis = ops_helpers::scalar_const(&mut graph, self.axis);
        let axis = ops_helpers::resolve_axes(&mut graph, axis, input_data);

        let normalized_axes = {
            let r = MilliOpRange::new(
                axis,
                ops_helpers::rank(&mut graph, input_data),
                ops_helpers::scalar_const(&mut graph, 1i64),
            );
            graph.push_op(AnyMilliOp::Range(r))
        };

        let input_squared = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::mul(
            input_f32, input_f32,
        )));

        let squared_mean = graph.push_op(AnyMilliOp::ReduceMean(MilliOpReduceMean::new(
            input_squared,
            Some(normalized_axes),
            true,
            false,
        )));

        let epsilon = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(
            self.epsilon,
        )));
        let epsilon = graph.push_op(AnyMilliOp::CastLike(MilliOpCastLike::new(
            epsilon,
            squared_mean,
        )));
        let mean_plus_eps = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::add(
            squared_mean,
            epsilon,
        )));
        let rms = graph.push_op(AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::sqrt(
            mean_plus_eps,
        )));
        let rms_inv = graph.push_op(AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::reciprocal(rms)));

        let normalized = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::mul(
            input_f32, rms_inv,
        )));

        let input_scale_cast = graph.push_op(AnyMilliOp::Cast(MilliOpCast::new(
            input_scale,
            self.stash_type,
        )));
        let out = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::mul(
            normalized,
            input_scale_cast,
        )));

        let out = if let Some(bias) = self.bias {
            let bias_cast = graph.push_op(AnyMilliOp::Cast(MilliOpCast::new(
                input_map[&bias],
                self.stash_type,
            )));
            graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::add(
                out, bias_cast,
            )))
        } else {
            out
        };

        let out = graph.push_op(AnyMilliOp::CastLike(MilliOpCastLike::new(out, input_data)));

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerNormalizationOperation {
    input: SymbolicGraphTensorId,
    scale: SymbolicGraphTensorId,
    bias: Option<SymbolicGraphTensorId>,
    output: SymbolicGraphTensorId,
    mean_output: Option<SymbolicGraphTensorId>,
    inv_std_dev_output: Option<SymbolicGraphTensorId>,
    axis: i64,
    epsilon: f32,
    stash_type: DType,
}

impl LayerNormalizationOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        attributes: &[onnx::AttributeProto],
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

impl Operation for LayerNormalizationOperation {
    fn get_op_type_name(&self) -> String {
        "Layer Normalization".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        let mut v = vec![self.input, self.scale];
        if let Some(bias) = self.bias {
            v.push(bias);
        }
        v
    }
    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        let mut res = vec![self.output];
        if let Some(mean_output) = self.mean_output {
            res.push(mean_output);
        }
        if let Some(inv_std_dev_output) = self.inv_std_dev_output {
            res.push(inv_std_dev_output);
        }
        res
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let input_data = input_map[&self.input];
        let input_scale = input_map[&self.scale];

        let input_f32 = graph.push_op(AnyMilliOp::Cast(MilliOpCast::new(
            input_data,
            self.stash_type,
        )));

        let axis = ops_helpers::scalar_const(&mut graph, self.axis);
        let axis = ops_helpers::resolve_axes(&mut graph, axis, input_data);

        let normalized_axes = {
            let r = MilliOpRange::new(
                axis,
                ops_helpers::rank(&mut graph, input_data),
                ops_helpers::scalar_const(&mut graph, 1i64),
            );
            graph.push_op(AnyMilliOp::Range(r))
        };

        let mean = graph.push_op(AnyMilliOp::ReduceMean(MilliOpReduceMean::new(
            input_f32,
            Some(normalized_axes),
            true,
            false,
        )));

        let d = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::sub(
            input_f32, mean,
        )));
        let dd = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::mul(d, d)));
        let variance = graph.push_op(AnyMilliOp::ReduceMean(MilliOpReduceMean::new(
            dd,
            Some(normalized_axes),
            true,
            false,
        )));
        let epsilon = graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new_scalar(
            self.epsilon,
        )));
        let epsilon = graph.push_op(AnyMilliOp::CastLike(MilliOpCastLike::new(
            epsilon, variance,
        )));
        let var_plus_eps = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::add(
            variance, epsilon,
        )));
        let stddev = graph.push_op(AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::sqrt(
            var_plus_eps,
        )));
        let inv_stddev = graph.push_op(AnyMilliOp::SimpleUnary(MilliOpSimpleUnary::reciprocal(
            stddev,
        )));

        let normalized = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::mul(
            d, inv_stddev,
        )));

        /*
                let normalized_cast = graph.push_op(AnyMilliOp::CastLike(MilliOpCastLike::new(
                    normalized, input_data,
                )));

                let out = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::mul(
                    normalized_cast,
                    input_scale,
                )));

                let out = if let Some(bias) = self.bias {
                    graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::add(
                        out,
                        input_map[&bias],
                    )))
                } else {
                    out
                };
        */
        let input_scale_cast = graph.push_op(AnyMilliOp::Cast(MilliOpCast::new(
            input_scale,
            self.stash_type,
        )));
        let out = graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::mul(
            normalized,
            input_scale_cast,
        )));

        let out = if let Some(bias) = self.bias {
            let bias_cast = graph.push_op(AnyMilliOp::Cast(MilliOpCast::new(
                input_map[&bias],
                self.stash_type,
            )));
            graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::add(
                out, bias_cast,
            )))
        } else {
            out
        };

        let out = graph.push_op(AnyMilliOp::CastLike(MilliOpCastLike::new(out, input_data)));

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
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InstanceNormalizationOperation {
    input: SymbolicGraphTensorId,
    scale: SymbolicGraphTensorId,
    bias: SymbolicGraphTensorId,
    output: SymbolicGraphTensorId,
    epsilon: Option<f32>,
}

impl InstanceNormalizationOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        attributes: &[onnx::AttributeProto],
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

impl Operation for InstanceNormalizationOperation {
    fn get_op_type_name(&self) -> String {
        "Instance Normalization".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.input, self.scale, self.bias]
    }

    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        unimplemented!();
    }
}
