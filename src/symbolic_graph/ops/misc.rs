use crate::backends::eval_backend::EvalBackend;
use crate::dtype::DType;
use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::*;
use crate::numeric_tensor::NumericTensor;
use crate::symbolic_graph::ops::{EvalError, Operation};
use crate::symbolic_graph::{ONNXDecodingError, SymbolicGraphMutator, query_attribute_float, query_attribute_graph, query_attribute_string, SymbolicGraph};
use crate::{DynRank, onnx};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use rand::Rng;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WhereOperation {
    global_id: GlobalId,
    condition: GlobalId,
    x: GlobalId,
    y: GlobalId,
    output: GlobalId,
}

impl WhereOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        _attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 3 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Where"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Where"));
        }

        Ok(Self {
            global_id: GlobalId::new(rng),
            condition: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Where"))?,
            x: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Where"))?,
            y: inputs[2].ok_or(ONNXDecodingError::InvalidOperatorInputs("Where"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Where"))?,
        })
    }
}

impl Node for WhereOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Where".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new([self.condition, self.x, self.y].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for WhereOperation {
    fn get_milli_op_graph(&self, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let out = Where::push_new(
            &mut graph,
            input_map[&self.condition],
            input_map[&self.x],
            input_map[&self.y],
            rng
        );
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IfOperation {
    global_id: GlobalId,
    outputs: Vec<GlobalId>,
    condition: GlobalId,
    then_branch: SymbolicGraph,
    else_branch: SymbolicGraph,
}

impl IfOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        symbolic_graph_mutator: &mut SymbolicGraphMutator,
        core_opset_version: usize,
        rng: &mut impl Rng,
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
            let mut inner_graph = SymbolicGraph::new(rng);
            inner_graph.populate(
                symbolic_graph_mutator,
                then_branch_graph,
                core_opset_version,
                rng
            )?;
            inner_graph
        };
        let else_branch_graph = query_attribute_graph(attributes, "else_branch")
            .ok_or(ONNXDecodingError::MissingField("else_branch"))?;
        let else_branch_graph = {
            let mut inner_graph = SymbolicGraph::new(rng);
            inner_graph.populate(
                symbolic_graph_mutator,
                else_branch_graph,
                core_opset_version,
                rng
            )?;
            inner_graph
        };

        Ok(Self {
            global_id: GlobalId::new(rng),
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

impl Node for IfOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "If".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        let mut inputs_set = HashSet::new();
        inputs_set.insert(self.condition);
        inputs_set.extend(self.then_branch.get_foreign_tensor_ids());
        inputs_set.extend(self.else_branch.get_foreign_tensor_ids());
        let mut inputs_vec: Vec<_> = inputs_set.into_iter().collect();
        inputs_vec.sort(); // Deterministic ordering
        Box::new(inputs_vec.into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(self.outputs.clone().into_iter())
    }
}

impl Operation for IfOperation {
    fn parameters(&self) -> Vec<Property> {
        vec![Property::new("num_outputs", PropertyValue::Int(self.outputs.len() as i64))]
    }

    fn get_sub_graphs(&self) -> Vec<&SymbolicGraph> {
        vec![&self.then_branch, &self.else_branch]
    }

    fn eval(
        &self,
        backend: &mut EvalBackend,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, EvalError>
    {
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
        Ok(Box::new(outputs.into_iter()))
    }

    fn get_milli_op_graph(&self, _rng: &mut impl Rng) -> MilliOpGraph {
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
    global_id: GlobalId,
    input: GlobalId,
    pads: GlobalId,
    constant_value: Option<GlobalId>,
    axes: Option<GlobalId>,
    mode: PadMode,
    output: GlobalId,
}

impl PadOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
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
            global_id: GlobalId::new(rng),
        })
    }
}

impl Node for PadOperation {
    type OpKind = String;

    fn global_id(&self) -> GlobalId {
        self.global_id
    }

    fn op_kind(&self) -> Self::OpKind {
        "Pad".to_string()
    }

    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        let mut ret = vec![self.input, self.pads];
        if let Some(constant_value) = self.constant_value {
            ret.push(constant_value);
        }
        if let Some(axes) = self.axes {
            ret.push(axes);
        }
        Box::new(ret.into_iter())
    }

    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for PadOperation {
    fn parameters(&self) -> Vec<Property> {
        let mode_str = match &self.mode {
            PadMode::Constant => "constant",
            PadMode::Reflect => "reflect",
            PadMode::Edge => "edge",
            PadMode::Wrap => "wrap",
        };
        vec![Property::new("mode", PropertyValue::String(mode_str.to_string()))]
    }

    fn get_milli_op_graph(&self, _rng: &mut impl Rng) -> MilliOpGraph {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RandomNormalLikeOperation {
    global_id: GlobalId,
    input: GlobalId,
    output: GlobalId,
    dtype: Option<DType>,
    mean: f32,
    scale: f32,
    seed: Option<f32>,
}

impl RandomNormalLikeOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
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
            global_id: GlobalId::new(rng),
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

impl Node for RandomNormalLikeOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "RandomNormalLike".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for RandomNormalLikeOperation {
    fn parameters(&self) -> Vec<Property> {
        let mut params = Vec::new();
        if let Some(dtype) = self.dtype {
            params.push(Property::new("dtype", PropertyValue::DType(dtype)));
        }
        params.push(Property::new("mean", PropertyValue::Float(self.mean as f64)));
        params.push(Property::new("scale", PropertyValue::Float(self.scale as f64)));
        if let Some(seed) = self.seed {
            params.push(Property::new("seed", PropertyValue::Float(seed as f64)));
        }
        params
    }

    fn get_milli_op_graph(&self, _rng: &mut impl Rng) -> MilliOpGraph {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExpandOperation {
    global_id: GlobalId,
    input: GlobalId,
    shape: GlobalId,
    output: GlobalId,
}

impl ExpandOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        _attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Expand"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Expand"));
        }

        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Expand"))?,
            shape: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Expand"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Expand"))?,
        })
    }
}

impl Node for ExpandOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Expand".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input).chain(std::iter::once(self.shape)))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for ExpandOperation {
    fn get_milli_op_graph(&self, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);

        let x = Expand::push_new(&mut graph, input_map[&self.input], input_map[&self.shape], rng);

        let mut output_map = HashMap::new();
        output_map.insert(x, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClipOperation {
    global_id: GlobalId,
    input: GlobalId,
    min: Option<GlobalId>,
    max: Option<GlobalId>,
    output: GlobalId,
}

impl ClipOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        _attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.is_empty() || inputs.len() > 3 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Clip"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Clip"));
        }

        Ok(Self {
            global_id: GlobalId::new(rng),
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

impl Node for ClipOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Clip".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        let mut o = vec![self.input];
        if let Some(min) = self.min {
            o.push(min);
        }
        if let Some(max) = self.max {
            o.push(max);
        }
        Box::new(o.into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}
impl Operation for ClipOperation {
    fn get_milli_op_graph(&self, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let mut x = input_map[&self.input];
        if let Some(min) = self.min {
            let min = input_map[&min];
            x = SimpleBinary::max(&mut graph, x, min, rng);
        }
        if let Some(max) = self.max {
            let max = input_map[&max];
            x = SimpleBinary::min(&mut graph, x, max, rng);
        }
        let mut output_map = HashMap::new();
        output_map.insert(x, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RangeOperation {
    global_id: GlobalId,
    start: GlobalId,
    end: GlobalId,
    delta: GlobalId,
    output: GlobalId,
}

impl RangeOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 3 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Range"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Range"));
        }
        Ok(Self {
            global_id: GlobalId::new(rng),
            start: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Range"))?,
            end: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Range"))?,
            delta: inputs[2].ok_or(ONNXDecodingError::InvalidOperatorInputs("Range"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Range"))?,
        })
    }
}

impl Node for RangeOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Range".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(
            std::iter::once(self.start)
                .chain(std::iter::once(self.end))
                .chain(std::iter::once(self.delta)),
        )
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for RangeOperation {
    fn get_milli_op_graph(&self, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);

        let out = Range::push_new(
            &mut graph,
            input_map[&self.start],
            input_map[&self.end],
            input_map[&self.delta],
            rng
        );

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
