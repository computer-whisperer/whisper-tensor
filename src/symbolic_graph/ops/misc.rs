use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::ops::*;
use crate::milli_graph::{MilliLoweringContext, MilliOpGraph};
use crate::numeric_dtype::{NumericDType, ONNXDType};
use crate::symbolic_graph::ops::{EvalError, Operation};
use crate::symbolic_graph::{
    ONNXDecodingError, SymbolicGraph, SymbolicGraphMutator, query_attribute_float,
    query_attribute_graph, query_attribute_int, query_attribute_ints, query_attribute_string,
};
use crate::onnx;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

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
    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let out = Where::push_new(
            &mut graph,
            input_map[&self.condition],
            input_map[&self.x],
            input_map[&self.y],
            rng,
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
                rng,
                None,
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
                rng,
                None,
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
        vec![Property::new(
            "num_outputs",
            PropertyValue::Int(self.outputs.len() as i64),
        )]
    }

    fn get_sub_graphs(&self) -> Vec<&SymbolicGraph> {
        vec![&self.then_branch, &self.else_branch]
    }

    fn eval_pool<'p, P: crate::pool::Pool + 'p>(
        &self,
        inputs: &HashMap<GlobalId, crate::numeric_dtype::ONNXTensorView<'_>>,
        pool: &'p P,
    ) -> Result<HashMap<GlobalId, crate::numeric_dtype::ONNXTensor<'p, P>>, EvalError> {
        // Extract numeric views for branch evaluation.
        let mut numeric_views: HashMap<
            GlobalId,
            &crate::numeric_tensor::NumericTensorView<'_, crate::tensor_rank::DynRank>,
        > = HashMap::new();
        for (&id, v) in inputs {
            if let Ok(nv) = v.as_numeric() {
                numeric_views.insert(id, nv);
            }
        }

        let cond_view = numeric_views.get(&self.condition).unwrap();
        let condition: bool = cond_view.read_element(0).to_f64() != 0.0;
        let (branch_results, output_ids) = if condition {
            (
                self.then_branch.eval_pool(&numeric_views, pool)?,
                &self.then_branch.ordered_outputs,
            )
        } else {
            (
                self.else_branch.eval_pool(&numeric_views, pool)?,
                &self.else_branch.ordered_outputs,
            )
        };

        let mut outputs = HashMap::new();
        for (to_id, from_id) in self.outputs.iter().zip(output_ids.iter()) {
            if let Some(tensor) = branch_results.get(from_id) {
                let view = tensor.view();
                let layout =
                    crate::numeric_tensor::TensorLayout::<crate::tensor_rank::DynRank>::row_major(
                        view.shape().to_vec(),
                        view.dtype(),
                    );
                let buf = pool
                    .allocate(layout.buffer_size_bytes())
                    .map_err(|e| EvalError::InvalidInput(format!("pool allocation: {e}")))?;
                let mut out = crate::numeric_tensor::NumericTensor::from_parts(buf, layout);
                for i in 0..view.numel() {
                    out.write_element(i, view.read_element(i));
                }
                outputs.insert(*to_id, crate::numeric_dtype::ONNXTensor::Numeric(out));
            }
        }
        Ok(outputs)
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, _rng: &mut impl Rng) -> MilliOpGraph {
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
            constant_value: inputs.get(2).and_then(|x| *x),
            axes: inputs.get(3).and_then(|x| *x),
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
        vec![Property::new(
            "mode",
            PropertyValue::String(mode_str.to_string()),
        )]
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let mode = match &self.mode {
            PadMode::Constant => crate::milli_graph::ops::PadMode::Constant,
            PadMode::Reflect => crate::milli_graph::ops::PadMode::Reflect,
            PadMode::Edge => crate::milli_graph::ops::PadMode::Edge,
            PadMode::Wrap => crate::milli_graph::ops::PadMode::Wrap,
        };
        let out = crate::milli_graph::ops::Pad::push_new(
            &mut graph,
            input_map[&self.input],
            input_map[&self.pads],
            self.constant_value.map(|x| input_map[&x]),
            self.axes.map(|x| input_map[&x]),
            mode,
            rng,
        );
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RandomNormalLikeOperation {
    global_id: GlobalId,
    input: GlobalId,
    output: GlobalId,
    dtype: Option<NumericDType>,
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
            Some(ONNXDType::from_onnx_proto(to_datatype)?.expect_numeric("RandomNormalLike"))
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
            params.push(Property::new(
                "dtype",
                PropertyValue::DType(dtype.to_legacy()),
            ));
        }
        params.push(Property::new(
            "mean",
            PropertyValue::Float(self.mean as f64),
        ));
        params.push(Property::new(
            "scale",
            PropertyValue::Float(self.scale as f64),
        ));
        if let Some(seed) = self.seed {
            params.push(Property::new("seed", PropertyValue::Float(seed as f64)));
        }
        params
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let out = crate::milli_graph::ops::RandomNormalLike::push_new(
            &mut graph,
            input_map[&self.input],
            self.dtype.map(|dt| dt.to_legacy()),
            self.mean,
            self.scale,
            self.seed,
            rng,
        );
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
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
    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);

        let x = Expand::push_new(
            &mut graph,
            input_map[&self.input],
            input_map[&self.shape],
            rng,
        );

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
            min: inputs.get(1).and_then(|x| *x),
            max: inputs.get(2).and_then(|x| *x),
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
    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
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
    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);

        let out = Range::push_new(
            &mut graph,
            input_map[&self.start],
            input_map[&self.end],
            input_map[&self.delta],
            rng,
        );

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

/// ONNX Tile: repeat input along each axis according to `repeats` tensor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TileOperation {
    global_id: GlobalId,
    input: GlobalId,
    repeats: GlobalId,
    output: GlobalId,
}

impl TileOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        _attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Tile"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Tile"));
        }
        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Tile"))?,
            repeats: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Tile"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Tile"))?,
        })
    }
}

impl Node for TileOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Tile".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new([self.input, self.repeats].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for TileOperation {
    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        // Tile via interleaved reshape + expand:
        // 1. input_shape = Shape(input)  -- [d0, d1, ..., dn]
        // 2. interleaved_shape = interleave([1,1,...], input_shape) = [1,d0,1,d1,...]
        // 3. reshaped = Reshape(input, interleaved_shape)
        // 4. expand_shape = interleave(repeats, input_shape) = [r0,d0,r1,d1,...]
        // 5. expanded = Expand(reshaped, expand_shape)
        // 6. final_shape = input_shape * repeats
        // 7. output = Reshape(expanded, final_shape)
        //
        // Interleaving two [N] tensors: unsqueeze each to [N,1], concat on axis 1
        // to get [N,2], reshape to [-1].
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let input = input_map[&self.input];
        let repeats = input_map[&self.repeats];

        let input_shape = Shape::push_new(&mut graph, input, rng);

        // ones = ConstantOfShape(Shape(input_shape), value=1)
        let shape_of_shape = Shape::push_new(&mut graph, input_shape, rng);
        let ones = {
            crate::milli_graph::ops::ConstantOfShape::push_new(
                &mut graph,
                crate::numeric_scalar::NumericScalar::from_i64(1),
                shape_of_shape,
                rng,
            )
        };

        let axis1 = Constant::new_scalar(&mut graph, 1i64, rng);

        // interleaved_shape = interleave(ones, input_shape)
        let ones_us = Unsqueeze::push_new(&mut graph, ones, axis1, rng);
        let shape_us1 = Unsqueeze::push_new(&mut graph, input_shape, axis1, rng);
        let stacked1 = Concat::push_new(&mut graph, vec![ones_us, shape_us1], 1, rng);
        let neg1_shape1 = Constant::from_vec(&mut graph, vec![-1i64], rng);
        let interleaved_shape = Reshape::push_new(&mut graph, stacked1, neg1_shape1, false, rng);

        let reshaped = Reshape::push_new(&mut graph, input, interleaved_shape, false, rng);

        // expand_shape = interleave(repeats, input_shape)
        let repeats_us = Unsqueeze::push_new(&mut graph, repeats, axis1, rng);
        let shape_us2 = Unsqueeze::push_new(&mut graph, input_shape, axis1, rng);
        let stacked2 = Concat::push_new(&mut graph, vec![repeats_us, shape_us2], 1, rng);
        let neg1_shape2 = Constant::from_vec(&mut graph, vec![-1i64], rng);
        let expand_shape = Reshape::push_new(&mut graph, stacked2, neg1_shape2, false, rng);

        let expanded = Expand::push_new(&mut graph, reshaped, expand_shape, rng);

        // final_shape = input_shape * repeats
        let final_shape = SimpleBinary::mul(&mut graph, input_shape, repeats, rng);
        let out = Reshape::push_new(&mut graph, expanded, final_shape, false, rng);

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

/// ONNX Dropout (inference mode): pass-through identity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DropoutOperation {
    global_id: GlobalId,
    input: GlobalId,
    output: GlobalId,
}

impl DropoutOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        _attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.is_empty() {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Dropout"));
        }
        if outputs.is_empty() {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Dropout"));
        }
        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Dropout"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Dropout"))?,
        })
    }
}

impl Node for DropoutOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Dropout".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for DropoutOperation {
    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let mut output_map = HashMap::new();
        output_map.insert(input_map[&self.input], self.output);
        graph.set_output_map(output_map);
        graph
    }
}

/// ONNX GlobalAveragePool: average over all spatial dimensions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GlobalAveragePoolOperation {
    global_id: GlobalId,
    input: GlobalId,
    output: GlobalId,
}

impl GlobalAveragePoolOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        _attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs(
                "GlobalAveragePool",
            ));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs(
                "GlobalAveragePool",
            ));
        }
        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                "GlobalAveragePool",
            ))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs(
                "GlobalAveragePool",
            ))?,
        })
    }
}

impl Node for GlobalAveragePoolOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "GlobalAveragePool".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for GlobalAveragePoolOperation {
    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        // Reshape [N, C, D1, ...] → [N, C, -1], mean axis 2 keepdims, reshape back
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let input = input_map[&self.input];
        let input_shape = Shape::push_new(&mut graph, input, rng);

        let shape_3d = Constant::from_vec(&mut graph, vec![0i64, 0i64, -1], rng);
        let x3d = Reshape::push_new(&mut graph, input, shape_3d, false, rng);

        let axis2 = Constant::new_scalar(&mut graph, 2i64, rng);
        let pooled = crate::milli_graph::ops::ReduceMean::push_new(
            &mut graph,
            x3d,
            Some(axis2),
            true,
            false,
            rng,
        );

        // Output shape: replace spatial dims with 1s
        let c_zero = Constant::from_vec(&mut graph, vec![0i64], rng);
        let c_two = Constant::from_vec(&mut graph, vec![2i64], rng);
        let bc = Slice::push_new(&mut graph, input_shape, c_zero, c_two, None, None, rng);
        let shape_len = Shape::push_new(&mut graph, input_shape, rng);
        let two_1d = Constant::from_vec(&mut graph, vec![2i64], rng);
        let n_spatial = SimpleBinary::sub(&mut graph, shape_len, two_1d, rng);
        let spatial_ones = crate::milli_graph::ops::ConstantOfShape::push_new(
            &mut graph,
            crate::numeric_scalar::NumericScalar::from_i64(1),
            n_spatial,
            rng,
        );
        let out_shape = Concat::push_new(&mut graph, vec![bc, spatial_ones], 0, rng);
        let out = Reshape::push_new(&mut graph, pooled, out_shape, false, rng);

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

/// ONNX GlobalMaxPool: max over all spatial dimensions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GlobalMaxPoolOperation {
    global_id: GlobalId,
    input: GlobalId,
    output: GlobalId,
}

impl GlobalMaxPoolOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        _attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("GlobalMaxPool"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("GlobalMaxPool"));
        }
        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("GlobalMaxPool"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("GlobalMaxPool"))?,
        })
    }
}

impl Node for GlobalMaxPoolOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "GlobalMaxPool".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for GlobalMaxPoolOperation {
    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let input = input_map[&self.input];
        let input_shape = Shape::push_new(&mut graph, input, rng);

        let shape_3d = Constant::from_vec(&mut graph, vec![0i64, 0i64, -1], rng);
        let x3d = Reshape::push_new(&mut graph, input, shape_3d, false, rng);

        let axis2 = Constant::new_scalar(&mut graph, 2i64, rng);
        let pooled = crate::milli_graph::ops::ReduceMax::push_new(
            &mut graph,
            x3d,
            Some(axis2),
            true,
            false,
            rng,
        );

        let c_zero = Constant::from_vec(&mut graph, vec![0i64], rng);
        let c_two = Constant::from_vec(&mut graph, vec![2i64], rng);
        let bc = Slice::push_new(&mut graph, input_shape, c_zero, c_two, None, None, rng);
        let shape_len = Shape::push_new(&mut graph, input_shape, rng);
        let two_1d = Constant::from_vec(&mut graph, vec![2i64], rng);
        let n_spatial = SimpleBinary::sub(&mut graph, shape_len, two_1d, rng);
        let spatial_ones = crate::milli_graph::ops::ConstantOfShape::push_new(
            &mut graph,
            crate::numeric_scalar::NumericScalar::from_i64(1),
            n_spatial,
            rng,
        );
        let out_shape = Concat::push_new(&mut graph, vec![bc, spatial_ones], 0, rng);
        let out = Reshape::push_new(&mut graph, pooled, out_shape, false, rng);

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

/// ONNX Mean: element-wise mean of N inputs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeanOperation {
    global_id: GlobalId,
    inputs: Vec<GlobalId>,
    output: GlobalId,
}

impl MeanOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        _attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.is_empty() {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Mean"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Mean"));
        }
        let inputs: Vec<GlobalId> = inputs
            .iter()
            .map(|x| x.ok_or(ONNXDecodingError::InvalidOperatorInputs("Mean")))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            global_id: GlobalId::new(rng),
            inputs,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Mean"))?,
        })
    }
}

impl Node for MeanOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Mean".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(self.inputs.clone().into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for MeanOperation {
    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let mut sum = input_map[&self.inputs[0]];
        for inp in &self.inputs[1..] {
            sum = SimpleBinary::add(&mut graph, sum, input_map[inp], rng);
        }
        let n = Constant::new_scalar(&mut graph, self.inputs.len() as f32, rng);
        let n = CastLike::push_new(&mut graph, n, sum, rng);
        let out = SimpleBinary::div(&mut graph, sum, n, rng);
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

/// ONNX Sum: element-wise sum of N inputs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SumOperation {
    global_id: GlobalId,
    inputs: Vec<GlobalId>,
    output: GlobalId,
}

impl SumOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        _attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.is_empty() {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Sum"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Sum"));
        }
        let inputs: Vec<GlobalId> = inputs
            .iter()
            .map(|x| x.ok_or(ONNXDecodingError::InvalidOperatorInputs("Sum")))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            global_id: GlobalId::new(rng),
            inputs,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Sum"))?,
        })
    }
}

impl Node for SumOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Sum".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(self.inputs.clone().into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for SumOperation {
    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let mut sum = input_map[&self.inputs[0]];
        for inp in &self.inputs[1..] {
            sum = SimpleBinary::add(&mut graph, sum, input_map[inp], rng);
        }
        let mut output_map = HashMap::new();
        output_map.insert(sum, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

/// ONNX SpaceToDepth: [N, C, H, W] → [N, C*bs*bs, H/bs, W/bs]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpaceToDepthOperation {
    global_id: GlobalId,
    input: GlobalId,
    output: GlobalId,
    blocksize: i64,
}

impl SpaceToDepthOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("SpaceToDepth"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("SpaceToDepth"));
        }
        let blocksize = query_attribute_int(attributes, "blocksize")
            .ok_or(ONNXDecodingError::InvalidOperatorInputs("SpaceToDepth"))?;
        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("SpaceToDepth"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("SpaceToDepth"))?,
            blocksize,
        })
    }
}

impl Node for SpaceToDepthOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "SpaceToDepth".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for SpaceToDepthOperation {
    fn parameters(&self) -> Vec<Property> {
        vec![Property::new(
            "blocksize",
            PropertyValue::Int(self.blocksize),
        )]
    }
    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        // [N,C,H,W] → reshape [N,C,H/bs,bs,W/bs,bs] → transpose [0,1,3,5,2,4] → reshape [N,C*bs²,H/bs,W/bs]
        let bs = self.blocksize;
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let input = input_map[&self.input];
        let input_shape = Shape::push_new(&mut graph, input, rng);
        let c0 = Constant::from_vec(&mut graph, vec![0i64], rng);
        let c1 = Constant::from_vec(&mut graph, vec![1i64], rng);
        let c2 = Constant::from_vec(&mut graph, vec![2i64], rng);
        let c3 = Constant::from_vec(&mut graph, vec![3i64], rng);
        let c4 = Constant::from_vec(&mut graph, vec![4i64], rng);
        let s0 = Slice::push_new(&mut graph, input_shape, c0, c1, None, None, rng);
        let s1 = Slice::push_new(&mut graph, input_shape, c1, c2, None, None, rng);
        let s2 = Slice::push_new(&mut graph, input_shape, c2, c3, None, None, rng);
        let s3 = Slice::push_new(&mut graph, input_shape, c3, c4, None, None, rng);
        let bs_t = Constant::from_vec(&mut graph, vec![bs], rng);
        let h_div = SimpleBinary::div(&mut graph, s2, bs_t, rng);
        let w_div = SimpleBinary::div(&mut graph, s3, bs_t, rng);
        let inter = Concat::push_new(&mut graph, vec![s0, s1, h_div, bs_t, w_div, bs_t], 0, rng);
        let reshaped = Reshape::push_new(&mut graph, input, inter, false, rng);
        let transposed =
            Transpose::push_new(&mut graph, reshaped, Some(vec![0, 3, 5, 1, 2, 4]), rng);
        let c_bs_sq = SimpleBinary::mul(&mut graph, s1, bs_t, rng);
        let c_bs_sq = SimpleBinary::mul(&mut graph, c_bs_sq, bs_t, rng);
        let final_shape = Concat::push_new(&mut graph, vec![s0, c_bs_sq, h_div, w_div], 0, rng);
        let out = Reshape::push_new(&mut graph, transposed, final_shape, false, rng);
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

/// ONNX DepthToSpace: [N, C, H, W] → [N, C/(bs²), H*bs, W*bs]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DepthToSpaceOperation {
    global_id: GlobalId,
    input: GlobalId,
    output: GlobalId,
    blocksize: i64,
    mode: String,
}

impl DepthToSpaceOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("DepthToSpace"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("DepthToSpace"));
        }
        let blocksize = query_attribute_int(attributes, "blocksize")
            .ok_or(ONNXDecodingError::InvalidOperatorInputs("DepthToSpace"))?;
        let mode = query_attribute_string(attributes, "mode").unwrap_or_else(|| "DCR".to_string());
        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("DepthToSpace"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("DepthToSpace"))?,
            blocksize,
            mode,
        })
    }
}

impl Node for DepthToSpaceOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "DepthToSpace".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for DepthToSpaceOperation {
    fn parameters(&self) -> Vec<Property> {
        vec![
            Property::new("blocksize", PropertyValue::Int(self.blocksize)),
            Property::new("mode", PropertyValue::String(self.mode.clone())),
        ]
    }
    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let bs = self.blocksize;
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let input = input_map[&self.input];
        let input_shape = Shape::push_new(&mut graph, input, rng);
        let c0 = Constant::from_vec(&mut graph, vec![0i64], rng);
        let c1 = Constant::from_vec(&mut graph, vec![1i64], rng);
        let c2 = Constant::from_vec(&mut graph, vec![2i64], rng);
        let c3 = Constant::from_vec(&mut graph, vec![3i64], rng);
        let c4 = Constant::from_vec(&mut graph, vec![4i64], rng);
        let s0 = Slice::push_new(&mut graph, input_shape, c0, c1, None, None, rng);
        let s1 = Slice::push_new(&mut graph, input_shape, c1, c2, None, None, rng);
        let s2 = Slice::push_new(&mut graph, input_shape, c2, c3, None, None, rng);
        let s3 = Slice::push_new(&mut graph, input_shape, c3, c4, None, None, rng);
        let bs_t = Constant::from_vec(&mut graph, vec![bs], rng);
        let bs_sq = Constant::from_vec(&mut graph, vec![bs * bs], rng);
        let c_red = SimpleBinary::div(&mut graph, s1, bs_sq, rng);
        let h_mul = SimpleBinary::mul(&mut graph, s2, bs_t, rng);
        let w_mul = SimpleBinary::mul(&mut graph, s3, bs_t, rng);
        let (inter, perm) = if self.mode == "CRD" {
            (
                Concat::push_new(&mut graph, vec![s0, c_red, bs_t, bs_t, s2, s3], 0, rng),
                vec![0, 1, 4, 2, 5, 3],
            )
        } else {
            (
                Concat::push_new(&mut graph, vec![s0, bs_t, bs_t, c_red, s2, s3], 0, rng),
                vec![0, 3, 4, 1, 5, 2],
            )
        };
        let reshaped = Reshape::push_new(&mut graph, input, inter, false, rng);
        let transposed = Transpose::push_new(&mut graph, reshaped, Some(perm), rng);
        let final_shape = Concat::push_new(&mut graph, vec![s0, c_red, h_mul, w_mul], 0, rng);
        let out = Reshape::push_new(&mut graph, transposed, final_shape, false, rng);
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

/// ONNX Trilu: upper or lower triangular part of the last two dims.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TriluOperation {
    global_id: GlobalId,
    input: GlobalId,
    k: Option<GlobalId>,
    output: GlobalId,
    upper: bool,
}

impl TriluOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.is_empty() || inputs.len() > 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Trilu"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Trilu"));
        }
        let upper = query_attribute_int(attributes, "upper").unwrap_or(1) != 0;
        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Trilu"))?,
            k: if inputs.len() > 1 { inputs[1] } else { None },
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Trilu"))?,
            upper,
        })
    }
}

impl Node for TriluOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Trilu".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        if let Some(k) = self.k {
            Box::new([self.input, k].into_iter())
        } else {
            Box::new(std::iter::once(self.input))
        }
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for TriluOperation {
    fn parameters(&self) -> Vec<Property> {
        vec![Property::new("upper", PropertyValue::Bool(self.upper))]
    }
    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        // Build row/col index matrices, compare col-row vs k, mask with Where.
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let input = input_map[&self.input];
        let k = if let Some(k_id) = self.k {
            input_map[&k_id]
        } else {
            Constant::new_scalar(&mut graph, 0i64, rng)
        };

        let input_shape = Shape::push_new(&mut graph, input, rng);
        // rows = shape[-2], cols = shape[-1]
        let neg1 = Constant::from_vec(&mut graph, vec![-1i64], rng);
        let neg2 = Constant::from_vec(&mut graph, vec![-2i64], rng);
        let big = Constant::from_vec(&mut graph, vec![i64::MAX], rng);
        let rows = Slice::push_new(&mut graph, input_shape, neg2, neg1, None, None, rng);
        let cols = Slice::push_new(&mut graph, input_shape, neg1, big, None, None, rng);

        let zero = Constant::new_scalar(&mut graph, 0i64, rng);
        let one = Constant::new_scalar(&mut graph, 1i64, rng);
        let empty_shape = Constant::from_vec(&mut graph, Vec::<i64>::new(), rng);
        let rows_s = Reshape::push_new(&mut graph, rows, empty_shape, false, rng);
        let cols_s = Reshape::push_new(&mut graph, cols, empty_shape, false, rng);
        let row_idx = crate::milli_graph::ops::Range::push_new(&mut graph, zero, rows_s, one, rng);
        let col_idx = crate::milli_graph::ops::Range::push_new(&mut graph, zero, cols_s, one, rng);

        let one_1d = Constant::from_vec(&mut graph, vec![1i64], rng);
        let row_shape = Concat::push_new(&mut graph, vec![rows, one_1d], 0, rng);
        let col_shape = Concat::push_new(&mut graph, vec![one_1d, cols], 0, rng);
        let row_mat = Reshape::push_new(&mut graph, row_idx, row_shape, false, rng);
        let col_mat = Reshape::push_new(&mut graph, col_idx, col_shape, false, rng);

        let diff = SimpleBinary::sub(&mut graph, col_mat, row_mat, rng); // col - row
        let mask = if self.upper {
            SimpleBinary::greater_or_equal(&mut graph, diff, k, rng)
        } else {
            SimpleBinary::less_or_equal(&mut graph, diff, k, rng)
        };

        let zero_like = Constant::new_scalar(&mut graph, 0.0f32, rng);
        let zero_like = CastLike::push_new(&mut graph, zero_like, input, rng);
        let out = Where::push_new(&mut graph, mask, input, zero_like, rng);

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

// ─── EyeLike ──────────────────────────────────────────────────────────

/// ONNX EyeLike: generates an identity-like matrix.
/// Input: a 2D tensor (used only for shape). Output: identity-like matrix.
/// Attributes: dtype (optional), k (diagonal offset, default 0).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EyeLikeOperation {
    global_id: GlobalId,
    input: GlobalId,
    output: GlobalId,
    dtype: Option<i64>,
    k: i64,
}

impl EyeLikeOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("EyeLike"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("EyeLike"));
        }
        let dtype = query_attribute_int(attributes, "dtype");
        let k = query_attribute_int(attributes, "k").unwrap_or(0);
        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("EyeLike"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("EyeLike"))?,
            dtype,
            k,
        })
    }
}

impl Node for EyeLikeOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "EyeLike".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for EyeLikeOperation {
    fn parameters(&self) -> Vec<Property> {
        let mut params = Vec::new();
        if let Some(dtype) = self.dtype {
            params.push(Property::new("dtype", PropertyValue::Int(dtype)));
        }
        params.push(Property::new("k", PropertyValue::Int(self.k)));
        params
    }

    fn is_differentiable(&self) -> bool {
        false
    }

    fn get_milli_op_graph(&self, ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);

        // Determine output dtype.
        let out_dtype = if let Some(dtype_int) = self.dtype {
            crate::numeric_dtype::ONNXDType::from_onnx_i32(dtype_int as i32)
                .expect("EyeLike: unsupported dtype")
                .expect_numeric("EyeLike")
        } else if let Some(dt) = ctx.tensor_dtypes.get(&self.input) {
            *dt
        } else {
            crate::numeric_dtype::NumericDType::F32
        };

        let out = crate::milli_graph::ops::EyeLike::push_new(
            &mut graph,
            input_map[&self.input],
            self.k,
            out_dtype,
            rng,
        );

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

// ─── Shrink ───────────────────────────────────────────────────────────

/// ONNX Shrink: if x < -lambd: y = x + bias; elif x > lambd: y = x - bias; else: y = 0
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShrinkOperation {
    global_id: GlobalId,
    input: GlobalId,
    output: GlobalId,
    lambd: f32,
    bias: f32,
}

impl ShrinkOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Shrink"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Shrink"));
        }
        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Shrink"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Shrink"))?,
            lambd: query_attribute_float(attributes, "lambd").unwrap_or(0.5),
            bias: query_attribute_float(attributes, "bias").unwrap_or(0.0),
        })
    }
}

impl Node for ShrinkOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Shrink".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for ShrinkOperation {
    fn parameters(&self) -> Vec<Property> {
        vec![
            Property::new("lambd", PropertyValue::Float(self.lambd.into())),
            Property::new("bias", PropertyValue::Float(self.bias.into())),
        ]
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        // Shrink(x) = x + bias if x < -lambd
        //           = x - bias if x > lambd
        //           = 0 otherwise
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let x = input_map[&self.input];

        let neg_lambd = Constant::new_scalar(&mut graph, -self.lambd, rng);
        let neg_lambd = CastLike::push_new(&mut graph, neg_lambd, x, rng);
        let lambd = Constant::new_scalar(&mut graph, self.lambd, rng);
        let lambd = CastLike::push_new(&mut graph, lambd, x, rng);
        let bias = Constant::new_scalar(&mut graph, self.bias, rng);
        let bias = CastLike::push_new(&mut graph, bias, x, rng);
        let zero = Constant::new_scalar(&mut graph, 0.0f32, rng);
        let zero = CastLike::push_new(&mut graph, zero, x, rng);

        let cond_neg = SimpleBinary::less(&mut graph, x, neg_lambd, rng);
        let cond_pos = SimpleBinary::greater(&mut graph, x, lambd, rng);
        let x_plus_bias = SimpleBinary::add(&mut graph, x, bias, rng);
        let x_minus_bias = SimpleBinary::sub(&mut graph, x, bias, rng);

        // result = Where(cond_neg, x+bias, Where(cond_pos, x-bias, 0))
        let inner = Where::push_new(&mut graph, cond_pos, x_minus_bias, zero, rng);
        let out_tid = Where::push_new(&mut graph, cond_neg, x_plus_bias, inner, rng);

        let mut output_map = HashMap::new();
        output_map.insert(out_tid, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

// ─── Hardmax ──────────────────────────────────────────────────────────

/// ONNX Hardmax: one-hot output where the position of the max value gets 1.0, all others 0.0.
/// Operates along the specified axis (default -1).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HardmaxOperation {
    global_id: GlobalId,
    input: GlobalId,
    output: GlobalId,
    axis: i64,
}

impl HardmaxOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Hardmax"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Hardmax"));
        }
        let axis = query_attribute_int(attributes, "axis").unwrap_or(-1);
        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Hardmax"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Hardmax"))?,
            axis,
        })
    }
}

impl Node for HardmaxOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Hardmax".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for HardmaxOperation {
    fn parameters(&self) -> Vec<Property> {
        vec![Property::new("axis", PropertyValue::Int(self.axis))]
    }

    fn is_differentiable(&self) -> bool {
        false
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        use crate::milli_graph::ops::*;
        use crate::numeric_dtype::NumericDType;
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let x = input_map[&self.input];

        // ArgMax along axis, keepdims=true → [.., 1, ..] with I64 indices.
        let argmax = ArgMax::push_new(&mut graph, x, self.axis, true, false, rng);

        // Shape(x) → [rank] I64 tensor.
        let shape_tensor = Shape::push_new(&mut graph, x, rng);

        // axis index constant [1] I64
        let axis_const = Constant::new_scalar(&mut graph, self.axis, rng);

        // axis_dim = Gather(shape_tensor, axis_const, axis=0) → [1] I64
        // (Gather handles negative indices via wrapping)
        let axis_dim = Gather::push_new(&mut graph, shape_tensor, axis_const, 0, rng);

        // Range(0, axis_dim, 1) → [axis_size] I64
        let zero_const = Constant::new_scalar(&mut graph, 0i64, rng);
        let one_const = Constant::new_scalar(&mut graph, 1i64, rng);
        let arange = Range::push_new(&mut graph, zero_const, axis_dim, one_const, rng);

        // shape_of_shape = Shape(shape_tensor) → [1] containing rank
        let shape_of_shape = Shape::push_new(&mut graph, shape_tensor, rng);

        // ones_shape = ConstantOfShape(shape_of_shape, value=1) → [rank] filled with 1s
        let ones_shape = ConstantOfShape::push_new(
            &mut graph,
            crate::numeric_scalar::NumericScalar::from_i64(1),
            shape_of_shape,
            rng,
        );

        // Normalize axis for negative values: norm_axis = axis + rank
        let norm_axis = if self.axis < 0 {
            SimpleBinary::add(&mut graph, axis_const, shape_of_shape, rng)
        } else {
            axis_const
        };
        let norm_axis_plus_one = SimpleBinary::add(&mut graph, norm_axis, one_const, rng);

        // before = Slice(ones_shape, 0:norm_axis)
        let before = Slice::push_new(
            &mut graph, ones_shape, zero_const, norm_axis, None, None, rng,
        );

        // after = Slice(ones_shape, norm_axis+1:rank)
        let after = Slice::push_new(
            &mut graph,
            ones_shape,
            norm_axis_plus_one,
            shape_of_shape, // end = rank
            None,
            None,
            rng,
        );

        // reshape_shape = Concat(before, axis_dim, after) along axis 0
        let reshape_shape = Concat::push_new(&mut graph, vec![before, axis_dim, after], 0, rng);

        // Reshape arange to [1,..,axis_size,..,1]
        let arange_reshaped = Reshape::push_new(&mut graph, arange, reshape_shape, false, rng);

        // Cast argmax to I64 (it should already be, but be explicit)
        let argmax_i64 = Cast::push_new(&mut graph, argmax, NumericDType::I64, rng);

        // Equal(argmax_i64, arange_reshaped) → broadcasts to input shape, BOOL
        let mask = SimpleBinary::equal(&mut graph, argmax_i64, arange_reshaped, rng);

        // Cast back to input dtype.
        let out_dtype = _ctx
            .tensor_dtypes
            .get(&self.input)
            .copied()
            .unwrap_or(NumericDType::F32);
        let result = Cast::push_new(&mut graph, mask, out_dtype, rng);

        let mut output_map = HashMap::new();
        output_map.insert(result, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

// ─── Compress ─────────────────────────────────────────────────────────

/// ONNX Compress: select elements from input using a boolean condition tensor along an axis.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompressOperation {
    global_id: GlobalId,
    input: GlobalId,
    condition: GlobalId,
    output: GlobalId,
    axis: Option<i64>,
}

impl CompressOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Compress"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Compress"));
        }
        let axis = query_attribute_int(attributes, "axis");
        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Compress"))?,
            condition: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Compress"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Compress"))?,
            axis,
        })
    }
}

impl Node for CompressOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Compress".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(vec![self.input, self.condition].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for CompressOperation {
    fn parameters(&self) -> Vec<Property> {
        let mut params = Vec::new();
        if let Some(axis) = self.axis {
            params.push(Property::new("axis", PropertyValue::Int(axis)));
        }
        params
    }

    fn is_differentiable(&self) -> bool {
        false
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let out = crate::milli_graph::ops::Compress::push_new(
            &mut graph,
            input_map[&self.input],
            input_map[&self.condition],
            self.axis,
            rng,
        );
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

// ─── MeanVarianceNormalization ────────────────────────────────────────

/// ONNX MeanVarianceNormalization: (x - mean(x, axes)) / sqrt(variance(x, axes) + epsilon)
/// Default axes: [0, 2, 3]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeanVarianceNormalizationOperation {
    global_id: GlobalId,
    input: GlobalId,
    output: GlobalId,
    axes: Vec<i64>,
}

impl MeanVarianceNormalizationOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs(
                "MeanVarianceNormalization",
            ));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs(
                "MeanVarianceNormalization",
            ));
        }
        let axes = query_attribute_ints(attributes, "axes").unwrap_or_else(|| vec![0, 2, 3]);
        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                "MeanVarianceNormalization",
            ))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs(
                "MeanVarianceNormalization",
            ))?,
            axes,
        })
    }
}

impl Node for MeanVarianceNormalizationOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "MeanVarianceNormalization".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for MeanVarianceNormalizationOperation {
    fn parameters(&self) -> Vec<Property> {
        vec![Property::new(
            "axes",
            PropertyValue::IntList(self.axes.clone()),
        )]
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        // MVN(x) = (x - mean) / sqrt(variance + epsilon)
        // where variance = mean((x - mean)^2)
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let x = input_map[&self.input];

        // Create axes constant
        let axes_tid = Constant::from_vec(&mut graph, self.axes.clone(), rng);

        // mean = ReduceMean(x, axes, keepdims=true)
        let mean = ReduceMean::push_new(&mut graph, x, Some(axes_tid), true, false, rng);

        // x_centered = x - mean
        let x_centered = SimpleBinary::sub(&mut graph, x, mean, rng);

        // variance = ReduceMean(x_centered * x_centered, axes, keepdims=true)
        let x_sq = SimpleBinary::mul(&mut graph, x_centered, x_centered, rng);
        // Need a separate axes constant for the second reduce
        let axes_tid2 = Constant::from_vec(&mut graph, self.axes.clone(), rng);
        let variance = ReduceMean::push_new(&mut graph, x_sq, Some(axes_tid2), true, false, rng);

        // epsilon = 1e-9
        let epsilon = Constant::new_scalar(&mut graph, 1e-9f32, rng);
        let epsilon = CastLike::push_new(&mut graph, epsilon, x, rng);

        // sqrt(variance + epsilon)
        let var_eps = SimpleBinary::add(&mut graph, variance, epsilon, rng);
        let std_dev = SimpleUnaryOp::sqrt(&mut graph, var_eps, rng);

        // result = x_centered / std_dev
        let out_tid = SimpleBinary::div(&mut graph, x_centered, std_dev, rng);

        let mut output_map = HashMap::new();
        output_map.insert(out_tid, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
