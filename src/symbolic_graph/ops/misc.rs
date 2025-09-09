use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::*;
use crate::numeric_tensor::NumericTensor;
use crate::symbolic_graph::ops::{EvalError, Operation};
use crate::symbolic_graph::{
    ONNXDecodingError, SymbolicGraphInner, SymbolicGraphMutator, SymbolicGraphTensorId,
    query_attribute_float, query_attribute_graph, query_attribute_string,
};
use crate::{DynRank, onnx};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WhereOperation {
    condition: SymbolicGraphTensorId,
    x: SymbolicGraphTensorId,
    y: SymbolicGraphTensorId,
    output: SymbolicGraphTensorId,
}

impl WhereOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        _attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 3 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Where"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Where"));
        }

        Ok(Self {
            condition: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Where"))?,
            x: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Where"))?,
            y: inputs[2].ok_or(ONNXDecodingError::InvalidOperatorInputs("Where"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Where"))?,
        })
    }
}

impl Operation for WhereOperation {
    fn get_op_type_name(&self) -> String {
        "Where".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.condition, self.x, self.y]
    }
    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let out = Where::push_new(
            &mut graph,
            input_map[&self.condition],
            input_map[&self.x],
            input_map[&self.y],
        );
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IfOperation {
    outputs: Vec<SymbolicGraphTensorId>,
    condition: SymbolicGraphTensorId,
    then_branch: SymbolicGraphInner,
    else_branch: SymbolicGraphInner,
}

impl IfOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        attributes: &[onnx::AttributeProto],
        symbolic_graph_mutator: &mut SymbolicGraphMutator,
        core_opset_version: usize,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("If"));
        }
        if outputs.is_empty() {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("If"));
        }

        let then_branch_graph = query_attribute_graph(attributes, "then_branch")
            .ok_or(ONNXDecodingError::MissingField("then_branch"))?;
        let then_branch_graph = {
            let mut inner_graph = SymbolicGraphInner::new();
            inner_graph.populate(
                symbolic_graph_mutator,
                then_branch_graph,
                core_opset_version,
            )?;
            inner_graph
        };
        let else_branch_graph = query_attribute_graph(attributes, "else_branch")
            .ok_or(ONNXDecodingError::MissingField("else_branch"))?;
        let else_branch_graph = {
            let mut inner_graph = SymbolicGraphInner::new();
            inner_graph.populate(
                symbolic_graph_mutator,
                else_branch_graph,
                core_opset_version,
            )?;
            inner_graph
        };

        Ok(Self {
            condition: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("If"))?,
            outputs: outputs
                .iter()
                .map(|x| x.ok_or(ONNXDecodingError::InvalidOperatorOutputs("Min")))
                .collect::<Result<_, _>>()?,
            then_branch: then_branch_graph,
            else_branch: else_branch_graph,
        })
    }
}

impl Operation for IfOperation {
    fn get_op_type_name(&self) -> String {
        "If".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        let mut inputs_set = HashSet::new();
        inputs_set.insert(self.condition);
        inputs_set.extend(self.then_branch.get_foreign_tensor_ids());
        inputs_set.extend(self.else_branch.get_foreign_tensor_ids());
        let mut inputs_vec: Vec<_> = inputs_set.into_iter().collect();
        inputs_vec.sort(); // Deterministic ordering
        inputs_vec
    }

    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        self.outputs.clone()
    }

    fn eval(
        &self,
        backend: &mut EvalBackend,
        inputs: &HashMap<SymbolicGraphTensorId, NumericTensor<DynRank>>,
    ) -> Result<HashMap<SymbolicGraphTensorId, NumericTensor<DynRank>>, EvalError> {
        let condition = inputs.get(&self.condition).unwrap();
        let condition: bool = condition.first_element().into();
        let (active_tensors, output_ids) = if condition {
            let tensors = self.then_branch.eval(inputs, backend)?;
            (tensors, &self.then_branch.ordered_outputs)
        } else {
            let tensors = self.else_branch.eval(inputs, backend)?;
            (tensors, &self.else_branch.ordered_outputs)
        };

        // Get all outputs
        let mut outputs = HashMap::new();
        for (to_id, from_id) in self.outputs.iter().zip(output_ids.iter()) {
            outputs.insert(*to_id, active_tensors.get(from_id).unwrap().clone());
        }
        Ok(outputs)
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) enum PadMode {
    Constant,
    Reflect,
    Edge,
    Wrap,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PadOperation {
    input: SymbolicGraphTensorId,
    pads: SymbolicGraphTensorId,
    constant_value: Option<SymbolicGraphTensorId>,
    axes: Option<SymbolicGraphTensorId>,
    mode: PadMode,
    output: SymbolicGraphTensorId,
}

impl PadOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() < 2 || inputs.len() > 4 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Pad"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Pad"));
        }

        let pad_mode = query_attribute_string(attributes, "mode").unwrap_or("constant".to_string());
        let pad_mode = match pad_mode.as_str() {
            "constant" => PadMode::Constant,
            "reflect" => PadMode::Reflect,
            "edge" => PadMode::Edge,
            "wrap" => PadMode::Wrap,
            _ => PadMode::Constant,
        };

        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Pad"))?,
            pads: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Pad"))?,
            constant_value: if inputs.len() > 2 {
                Some(inputs[2].ok_or(ONNXDecodingError::InvalidOperatorInputs("Pad"))?)
            } else {
                None
            },
            axes: if inputs.len() > 3 {
                Some(inputs[3].ok_or(ONNXDecodingError::InvalidOperatorInputs("Pad"))?)
            } else {
                None
            },
            mode: pad_mode,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Pad"))?,
        })
    }
}

impl Operation for PadOperation {
    fn get_op_type_name(&self) -> String {
        "Pad".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        let mut ret = vec![self.input, self.pads];
        if let Some(constant_value) = self.constant_value {
            ret.push(constant_value);
        }
        if let Some(axes) = self.axes {
            ret.push(axes);
        }
        ret
    }

    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RandomNormalLikeOperation {
    input: SymbolicGraphTensorId,
    output: SymbolicGraphTensorId,
    dtype: Option<DType>,
    mean: f32,
    scale: f32,
    seed: Option<f32>,
}

impl RandomNormalLikeOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("RandomNormalLike"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs(
                "RandomNormalLike",
            ));
        }

        let dtype = attributes.iter().find(|a| a.name == "dtype");
        let dtype = if let Some(dtype) = dtype {
            let to_datatype = onnx::tensor_proto::DataType::try_from(dtype.i as i32)
                .map_err(|x| ONNXDecodingError::ProtobufDecodeError(x.into()))?;
            Some(DType::try_from(to_datatype)?)
        } else {
            None
        };

        let mean = query_attribute_float(attributes, "mean").unwrap_or(0.0);
        let scale = query_attribute_float(attributes, "scale").unwrap_or(1.0);
        let seed = query_attribute_float(attributes, "seed");

        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("RandomNormalLike"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs(
                "RandomNormalLike",
            ))?,
            dtype,
            mean,
            scale,
            seed,
        })
    }
}

impl Operation for RandomNormalLikeOperation {
    fn get_op_type_name(&self) -> String {
        "Random Normal Like".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.input]
    }

    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExpandOperation {
    input: SymbolicGraphTensorId,
    shape: SymbolicGraphTensorId,
    output: SymbolicGraphTensorId,
}

impl ExpandOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        _attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Expand"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Expand"));
        }

        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Expand"))?,
            shape: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Expand"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Expand"))?,
        })
    }
}

impl Operation for ExpandOperation {
    fn get_op_type_name(&self) -> String {
        "Expand".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.input, self.shape]
    }

    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());

        let x = Expand::push_new(&mut graph, input_map[&self.input], input_map[&self.shape]);

        let mut output_map = HashMap::new();
        output_map.insert(x, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClipOperation {
    input: SymbolicGraphTensorId,
    min: Option<SymbolicGraphTensorId>,
    max: Option<SymbolicGraphTensorId>,
    output: SymbolicGraphTensorId,
}

impl ClipOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
        _attributes: &[onnx::AttributeProto],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.is_empty() || inputs.len() > 3 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Clip"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Clip"));
        }

        Ok(Self {
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Clip"))?,
            min: if inputs.len() > 1 {
                Some(inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Clip"))?)
            } else {
                None
            },
            max: if inputs.len() > 2 {
                Some(inputs[2].ok_or(ONNXDecodingError::InvalidOperatorInputs("Clip"))?)
            } else {
                None
            },
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Clip"))?,
        })
    }
}

impl Operation for ClipOperation {
    fn get_op_type_name(&self) -> String {
        "Clip".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        let mut o = vec![self.input];
        if let Some(min) = self.min {
            o.push(min);
        }
        if let Some(max) = self.max {
            o.push(max);
        }
        o
    }

    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());
        let mut x = input_map[&self.input];
        if let Some(min) = self.min {
            let min = input_map[&min];
            x = SimpleBinary::max(&mut graph, x, min);
        }
        if let Some(max) = self.max {
            let max = input_map[&max];
            x = SimpleBinary::min(&mut graph, x, max);
        }
        let mut output_map = HashMap::new();
        output_map.insert(x, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RangeOperation {
    start: SymbolicGraphTensorId,
    end: SymbolicGraphTensorId,
    delta: SymbolicGraphTensorId,
    output: SymbolicGraphTensorId,
}

impl RangeOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<SymbolicGraphTensorId>],
        outputs: &[Option<SymbolicGraphTensorId>],
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 3 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Range"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Range"));
        }
        Ok(Self {
            start: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Range"))?,
            end: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Range"))?,
            delta: inputs[2].ok_or(ONNXDecodingError::InvalidOperatorInputs("Range"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Range"))?,
        })
    }
}

impl Operation for RangeOperation {
    fn get_op_type_name(&self) -> String {
        "Range".to_string()
    }

    fn get_inputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.start, self.end, self.delta]
    }
    fn get_outputs(&self) -> Vec<SymbolicGraphTensorId> {
        vec![self.output]
    }

    fn get_milli_op_graph(&self) -> MilliOpGraph<SymbolicGraphTensorId> {
        let (mut graph, input_map) = MilliOpGraph::new(&self.get_inputs());

        let out = Range::push_new(
            &mut graph,
            input_map[&self.start],
            input_map[&self.end],
            input_map[&self.delta],
        );

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
