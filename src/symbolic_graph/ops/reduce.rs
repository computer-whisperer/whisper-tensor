use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::{self, MilliLoweringContext, MilliOpGraph};
use crate::onnx;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{ONNXDecodingError, query_attribute_int, query_attribute_ints};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CumSumOperation {
    global_id: GlobalId,
    input: GlobalId,
    output: GlobalId,
    axis: GlobalId,
    exclusive: bool,
    reverse: bool,
}

impl CumSumOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("CumSum"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("CumSum"));
        }
        let exclusive = query_attribute_int(attributes, "exclusive").unwrap_or_default() != 0;
        let reverse = query_attribute_int(attributes, "reverse").unwrap_or_default() != 0;
        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Unary"))?,
            axis: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Unary"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Unary"))?,
            exclusive,
            reverse,
        })
    }
}

impl Node for CumSumOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "CumSum".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new([self.input, self.axis].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for CumSumOperation {
    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let a = input_map[&self.input];
        let b = input_map[&self.axis];

        let out =
            milli_graph::ops::CumSum::push_new(&mut graph, a, b, self.exclusive, self.reverse, rng);

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }

    fn parameters(&self) -> Vec<Property> {
        vec![
            Property::new("exclusive", PropertyValue::Bool(self.exclusive)),
            Property::new("reverse", PropertyValue::Bool(self.reverse)),
        ]
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReduceMeanOperation {
    global_id: GlobalId,
    keepdims: Option<bool>,
    noop_with_empty_axes: Option<bool>,
    input_data: GlobalId,
    input_axes: Option<GlobalId>,
    axes_attr: Option<Vec<i64>>,
    output: GlobalId,
}

impl ReduceMeanOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.is_empty() || inputs.len() > 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("ReduceMean"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("ReduceMean"));
        }

        let axes_attr = query_attribute_ints(attributes, "axes");
        let keepdims = query_attribute_int(attributes, "keepdims").map(|x| x != 0);
        let noop_with_empty_axes =
            query_attribute_int(attributes, "noop_with_empty_axes").map(|x| x != 0);

        Ok(Self {
            global_id: GlobalId::new(rng),
            keepdims,
            noop_with_empty_axes,
            input_data: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("ReduceMean"))?,
            input_axes: if inputs.len() > 1 {
                Some(inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("ReduceMean"))?)
            } else {
                None
            },
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("ReduceMean"))?,
            axes_attr,
        })
    }
}

impl Node for ReduceMeanOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "ReduceMean".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        if let Some(input_axes) = self.input_axes {
            Box::new([self.input_data, input_axes].into_iter())
        } else {
            Box::new(std::iter::once(self.input_data))
        }
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}
impl Operation for ReduceMeanOperation {
    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let axes = if let Some(input_axes) = &self.input_axes {
            Some(input_map[input_axes])
        } else if let Some(axes) = &self.axes_attr {
            let tid = milli_graph::ops::Constant::from_vec(&mut graph, axes.clone(), rng);
            Some(tid)
        } else {
            None
        };
        let out = milli_graph::ops::ReduceMean::push_new(
            &mut graph,
            input_map[&self.input_data],
            axes,
            self.keepdims.unwrap_or(true),
            self.noop_with_empty_axes.unwrap_or(false),
            rng,
        );
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }

    fn parameters(&self) -> Vec<Property> {
        let mut params = Vec::new();
        if let Some(keepdims) = self.keepdims {
            params.push(Property::new("keepdims", PropertyValue::Bool(keepdims)));
        }
        if let Some(axes) = &self.axes_attr {
            params.push(Property::new("axes", PropertyValue::IntList(axes.clone())));
        }
        if let Some(noop) = self.noop_with_empty_axes {
            params.push(Property::new(
                "noop_with_empty_axes",
                PropertyValue::Bool(noop),
            ));
        }
        params
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReduceSumOperation {
    global_id: GlobalId,
    keepdims: Option<bool>,
    noop_with_empty_axes: Option<bool>,
    input_data: GlobalId,
    input_axes: Option<GlobalId>,
    axes_attr: Option<Vec<i64>>,
    output: GlobalId,
}

impl ReduceSumOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.is_empty() || inputs.len() > 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("ReduceSum"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("ReduceSum"));
        }

        let axes_attr = query_attribute_ints(attributes, "attr");
        let keepdims = query_attribute_int(attributes, "keepdims").map(|x| x != 0);
        let noop_with_empty_axes =
            query_attribute_int(attributes, "noop_with_empty_axes").map(|x| x != 0);

        Ok(Self {
            global_id: GlobalId::new(rng),
            keepdims,
            noop_with_empty_axes,
            input_data: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("ReduceSum"))?,
            input_axes: if inputs.len() > 1 {
                Some(inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("ReduceSum"))?)
            } else {
                None
            },
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("ReduceSum"))?,
            axes_attr,
        })
    }
}

impl Node for ReduceSumOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "ReduceSum".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        if let Some(input_axes) = self.input_axes {
            Box::new([self.input_data, input_axes].into_iter())
        } else {
            Box::new(std::iter::once(self.input_data))
        }
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}
impl Operation for ReduceSumOperation {
    fn parameters(&self) -> Vec<Property> {
        let mut params = Vec::new();
        if let Some(keepdims) = self.keepdims {
            params.push(Property::new("keepdims", PropertyValue::Bool(keepdims)));
        }
        if let Some(axes) = &self.axes_attr {
            params.push(Property::new("axes", PropertyValue::IntList(axes.clone())));
        }
        if let Some(noop) = self.noop_with_empty_axes {
            params.push(Property::new(
                "noop_with_empty_axes",
                PropertyValue::Bool(noop),
            ));
        }
        params
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let axes = if let Some(input_axes) = &self.input_axes {
            Some(input_map[input_axes])
        } else if let Some(axes) = &self.axes_attr {
            let tid = milli_graph::ops::Constant::from_vec(&mut graph, axes.clone(), rng);
            Some(tid)
        } else {
            None
        };
        let out = milli_graph::ops::ReduceSum::push_new(
            &mut graph,
            input_map[&self.input_data],
            axes,
            self.keepdims.unwrap_or(true),
            self.noop_with_empty_axes.unwrap_or(false),
            rng,
        );
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReduceMaxOperation {
    global_id: GlobalId,
    keepdims: Option<bool>,
    noop_with_empty_axes: Option<bool>,
    input_data: GlobalId,
    input_axes: Option<GlobalId>,
    axes_attr: Option<Vec<i64>>,
    output: GlobalId,
}

impl ReduceMaxOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.is_empty() || inputs.len() > 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("ReduceMax"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("ReduceMax"));
        }

        let axes_attr = query_attribute_ints(attributes, "attr");
        let keepdims = query_attribute_int(attributes, "keepdims").map(|x| x != 0);
        let noop_with_empty_axes =
            query_attribute_int(attributes, "noop_with_empty_axes").map(|x| x != 0);

        Ok(Self {
            global_id: GlobalId::new(rng),
            keepdims,
            noop_with_empty_axes,
            input_data: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("ReduceMax"))?,
            input_axes: if inputs.len() > 1 {
                Some(inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("ReduceMax"))?)
            } else {
                None
            },
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("ReduceMax"))?,
            axes_attr,
        })
    }
}

impl Node for ReduceMaxOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "ReduceMax".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        if let Some(input_axes) = self.input_axes {
            Box::new([self.input_data, input_axes].into_iter())
        } else {
            Box::new(std::iter::once(self.input_data))
        }
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}
impl Operation for ReduceMaxOperation {
    fn parameters(&self) -> Vec<Property> {
        let mut params = Vec::new();
        if let Some(keepdims) = self.keepdims {
            params.push(Property::new("keepdims", PropertyValue::Bool(keepdims)));
        }
        if let Some(axes) = &self.axes_attr {
            params.push(Property::new("axes", PropertyValue::IntList(axes.clone())));
        }
        if let Some(noop) = self.noop_with_empty_axes {
            params.push(Property::new(
                "noop_with_empty_axes",
                PropertyValue::Bool(noop),
            ));
        }
        params
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let axes = if let Some(input_axes) = &self.input_axes {
            Some(input_map[input_axes])
        } else if let Some(axes) = &self.axes_attr {
            let tid = milli_graph::ops::Constant::from_vec(&mut graph, axes.clone(), rng);
            Some(tid)
        } else {
            None
        };
        let out = milli_graph::ops::ReduceMax::push_new(
            &mut graph,
            input_map[&self.input_data],
            axes,
            self.keepdims.unwrap_or(true),
            self.noop_with_empty_axes.unwrap_or(false),
            rng,
        );
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReduceMinOperation {
    global_id: GlobalId,
    keepdims: Option<bool>,
    noop_with_empty_axes: Option<bool>,
    input_data: GlobalId,
    input_axes: Option<GlobalId>,
    axes_attr: Option<Vec<i64>>,
    output: GlobalId,
}

impl ReduceMinOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.is_empty() || inputs.len() > 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("ReduceMin"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("ReduceMin"));
        }

        let axes_attr = query_attribute_ints(attributes, "attr");
        let keepdims = query_attribute_int(attributes, "keepdims").map(|x| x != 0);
        let noop_with_empty_axes =
            query_attribute_int(attributes, "noop_with_empty_axes").map(|x| x != 0);

        Ok(Self {
            global_id: GlobalId::new(rng),
            keepdims,
            noop_with_empty_axes,
            input_data: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("ReduceMin"))?,
            input_axes: if inputs.len() > 1 {
                Some(inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("ReduceMin"))?)
            } else {
                None
            },
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("ReduceMin"))?,
            axes_attr,
        })
    }
}

impl Node for ReduceMinOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "ReduceMin".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        if let Some(input_axes) = self.input_axes {
            Box::new([self.input_data, input_axes].into_iter())
        } else {
            Box::new(std::iter::once(self.input_data))
        }
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for ReduceMinOperation {
    fn parameters(&self) -> Vec<Property> {
        let mut params = Vec::new();
        if let Some(keepdims) = self.keepdims {
            params.push(Property::new("keepdims", PropertyValue::Bool(keepdims)));
        }
        if let Some(axes) = &self.axes_attr {
            params.push(Property::new("axes", PropertyValue::IntList(axes.clone())));
        }
        if let Some(noop) = self.noop_with_empty_axes {
            params.push(Property::new(
                "noop_with_empty_axes",
                PropertyValue::Bool(noop),
            ));
        }
        params
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let axes = if let Some(input_axes) = &self.input_axes {
            Some(input_map[input_axes])
        } else if let Some(axes) = &self.axes_attr {
            let tid = milli_graph::ops::Constant::from_vec(&mut graph, axes.clone(), rng);
            Some(tid)
        } else {
            None
        };
        let out = milli_graph::ops::ReduceMin::push_new(
            &mut graph,
            input_map[&self.input_data],
            axes,
            self.keepdims.unwrap_or(true),
            self.noop_with_empty_axes.unwrap_or(false),
            rng,
        );
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReduceProdOperation {
    global_id: GlobalId,
    keepdims: Option<bool>,
    noop_with_empty_axes: Option<bool>,
    input_data: GlobalId,
    input_axes: Option<GlobalId>,
    axes_attr: Option<Vec<i64>>,
    output: GlobalId,
}

impl ReduceProdOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.is_empty() || inputs.len() > 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("ReduceProd"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("ReduceProd"));
        }

        let axes_attr = query_attribute_ints(attributes, "attr");
        let keepdims = query_attribute_int(attributes, "keepdims").map(|x| x != 0);
        let noop_with_empty_axes =
            query_attribute_int(attributes, "noop_with_empty_axes").map(|x| x != 0);

        Ok(Self {
            global_id: GlobalId::new(rng),
            keepdims,
            noop_with_empty_axes,
            input_data: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("ReduceProd"))?,
            input_axes: if inputs.len() > 1 {
                Some(inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("ReduceProd"))?)
            } else {
                None
            },
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("ReduceProd"))?,
            axes_attr,
        })
    }
}

impl Node for ReduceProdOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "ReduceProd".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        if let Some(input_axes) = self.input_axes {
            Box::new([self.input_data, input_axes].into_iter())
        } else {
            Box::new(std::iter::once(self.input_data))
        }
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for ReduceProdOperation {
    fn parameters(&self) -> Vec<Property> {
        let mut params = Vec::new();
        if let Some(keepdims) = self.keepdims {
            params.push(Property::new("keepdims", PropertyValue::Bool(keepdims)));
        }
        if let Some(axes) = &self.axes_attr {
            params.push(Property::new("axes", PropertyValue::IntList(axes.clone())));
        }
        if let Some(noop) = self.noop_with_empty_axes {
            params.push(Property::new(
                "noop_with_empty_axes",
                PropertyValue::Bool(noop),
            ));
        }
        params
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let axes = if let Some(input_axes) = &self.input_axes {
            Some(input_map[input_axes])
        } else if let Some(axes) = &self.axes_attr {
            let tid = milli_graph::ops::Constant::from_vec(&mut graph, axes.clone(), rng);
            Some(tid)
        } else {
            None
        };
        let out = milli_graph::ops::ReduceProd::push_new(
            &mut graph,
            input_map[&self.input_data],
            axes,
            self.keepdims.unwrap_or(true),
            self.noop_with_empty_axes.unwrap_or(false),
            rng,
        );
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

/// Macro to generate reduce-variant operations that share the same parsing/Node impl
/// but differ in their milli-graph decomposition.
macro_rules! define_reduce_variant {
    ($name:ident, $op_name:expr) => {
        #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
        pub struct $name {
            global_id: GlobalId,
            keepdims: Option<bool>,
            noop_with_empty_axes: Option<bool>,
            input_data: GlobalId,
            input_axes: Option<GlobalId>,
            axes_attr: Option<Vec<i64>>,
            output: GlobalId,
        }

        impl $name {
            pub(crate) fn from_onnx(
                inputs: &[Option<GlobalId>],
                outputs: &[Option<GlobalId>],
                attributes: &[onnx::AttributeProto],
                rng: &mut impl Rng,
            ) -> Result<Self, ONNXDecodingError> {
                if inputs.is_empty() || inputs.len() > 2 {
                    return Err(ONNXDecodingError::InvalidOperatorInputs($op_name));
                }
                if outputs.len() != 1 {
                    return Err(ONNXDecodingError::InvalidOperatorOutputs($op_name));
                }
                let axes_attr = query_attribute_ints(attributes, "axes");
                let keepdims = query_attribute_int(attributes, "keepdims").map(|x| x != 0);
                let noop_with_empty_axes =
                    query_attribute_int(attributes, "noop_with_empty_axes").map(|x| x != 0);
                Ok(Self {
                    global_id: GlobalId::new(rng),
                    keepdims,
                    noop_with_empty_axes,
                    input_data: inputs[0]
                        .ok_or(ONNXDecodingError::InvalidOperatorInputs($op_name))?,
                    input_axes: if inputs.len() > 1 {
                        Some(inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs($op_name))?)
                    } else {
                        None
                    },
                    output: outputs[0]
                        .ok_or(ONNXDecodingError::InvalidOperatorOutputs($op_name))?,
                    axes_attr,
                })
            }

            fn resolve_axes(
                &self,
                graph: &mut MilliOpGraph,
                input_map: &HashMap<GlobalId, GlobalId>,
                rng: &mut impl Rng,
            ) -> Option<GlobalId> {
                if let Some(input_axes) = &self.input_axes {
                    Some(input_map[input_axes])
                } else if let Some(axes) = &self.axes_attr {
                    let tid = milli_graph::ops::Constant::from_vec(graph, axes.clone(), rng);
                    Some(tid)
                } else {
                    None
                }
            }
        }

        impl Node for $name {
            type OpKind = String;
            fn global_id(&self) -> GlobalId {
                self.global_id
            }
            fn op_kind(&self) -> Self::OpKind {
                $op_name.to_string()
            }
            fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
                if let Some(input_axes) = self.input_axes {
                    Box::new([self.input_data, input_axes].into_iter())
                } else {
                    Box::new(std::iter::once(self.input_data))
                }
            }
            fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
                Box::new(std::iter::once(self.output))
            }
        }
    };
}

define_reduce_variant!(ReduceL1Operation, "ReduceL1");
impl Operation for ReduceL1Operation {
    fn parameters(&self) -> Vec<Property> {
        reduce_params(self.keepdims, &self.axes_attr, self.noop_with_empty_axes)
    }
    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        // ReduceL1(x) = ReduceSum(Abs(x))
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let x = input_map[&self.input_data];
        let abs_x = milli_graph::ops::SimpleUnaryOp::abs(&mut graph, x, rng);
        let axes = self.resolve_axes(&mut graph, &input_map, rng);
        let out = milli_graph::ops::ReduceSum::push_new(
            &mut graph,
            abs_x,
            axes,
            self.keepdims.unwrap_or(true),
            self.noop_with_empty_axes.unwrap_or(false),
            rng,
        );
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

define_reduce_variant!(ReduceSumSquareOperation, "ReduceSumSquare");
impl Operation for ReduceSumSquareOperation {
    fn parameters(&self) -> Vec<Property> {
        reduce_params(self.keepdims, &self.axes_attr, self.noop_with_empty_axes)
    }
    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        // ReduceSumSquare(x) = ReduceSum(x * x)
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let x = input_map[&self.input_data];
        let x_sq = milli_graph::ops::SimpleBinary::mul(&mut graph, x, x, rng);
        let axes = self.resolve_axes(&mut graph, &input_map, rng);
        let out = milli_graph::ops::ReduceSum::push_new(
            &mut graph,
            x_sq,
            axes,
            self.keepdims.unwrap_or(true),
            self.noop_with_empty_axes.unwrap_or(false),
            rng,
        );
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

define_reduce_variant!(ReduceLogSumOperation, "ReduceLogSum");
impl Operation for ReduceLogSumOperation {
    fn parameters(&self) -> Vec<Property> {
        reduce_params(self.keepdims, &self.axes_attr, self.noop_with_empty_axes)
    }
    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        // ReduceLogSum(x) = Log(ReduceSum(x))
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let x = input_map[&self.input_data];
        let axes = self.resolve_axes(&mut graph, &input_map, rng);
        let sum = milli_graph::ops::ReduceSum::push_new(
            &mut graph,
            x,
            axes,
            self.keepdims.unwrap_or(true),
            self.noop_with_empty_axes.unwrap_or(false),
            rng,
        );
        let out = milli_graph::ops::SimpleUnaryOp::ln(&mut graph, sum, rng);
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

define_reduce_variant!(ReduceLogSumExpOperation, "ReduceLogSumExp");
impl Operation for ReduceLogSumExpOperation {
    fn parameters(&self) -> Vec<Property> {
        reduce_params(self.keepdims, &self.axes_attr, self.noop_with_empty_axes)
    }
    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        // ReduceLogSumExp(x) = Log(ReduceSum(Exp(x)))
        // Numerically stable: subtract max first
        // max is always computed with keepdims=true so subtraction broadcasts.
        // The final result uses the requested keepdims setting.
        let keepdims = self.keepdims.unwrap_or(true);
        let noop = self.noop_with_empty_axes.unwrap_or(false);
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let x = input_map[&self.input_data];
        let axes = self.resolve_axes(&mut graph, &input_map, rng);
        // ReduceMax with keepdims=true for broadcasting
        let row_max_kd =
            milli_graph::ops::ReduceMax::push_new(&mut graph, x, axes, true, noop, rng);
        let shifted = milli_graph::ops::SimpleBinary::sub(&mut graph, x, row_max_kd, rng);
        let exp_shifted = milli_graph::ops::SimpleUnaryOp::exp(&mut graph, shifted, rng);
        let axes2 = self.resolve_axes(&mut graph, &input_map, rng);
        let sum = milli_graph::ops::ReduceSum::push_new(
            &mut graph,
            exp_shifted,
            axes2,
            keepdims,
            noop,
            rng,
        );
        let log_sum = milli_graph::ops::SimpleUnaryOp::ln(&mut graph, sum, rng);
        // Get row_max in the final shape (may need to drop keepdims)
        let row_max_final = if keepdims {
            row_max_kd
        } else {
            let axes3 = self.resolve_axes(&mut graph, &input_map, rng);
            milli_graph::ops::ReduceMax::push_new(&mut graph, x, axes3, false, noop, rng)
        };
        let out = milli_graph::ops::SimpleBinary::add(&mut graph, log_sum, row_max_final, rng);
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

fn reduce_params(
    keepdims: Option<bool>,
    axes_attr: &Option<Vec<i64>>,
    noop_with_empty_axes: Option<bool>,
) -> Vec<Property> {
    let mut params = Vec::new();
    if let Some(keepdims) = keepdims {
        params.push(Property::new("keepdims", PropertyValue::Bool(keepdims)));
    }
    if let Some(axes) = axes_attr {
        params.push(Property::new("axes", PropertyValue::IntList(axes.clone())));
    }
    if let Some(noop) = noop_with_empty_axes {
        params.push(Property::new(
            "noop_with_empty_axes",
            PropertyValue::Bool(noop),
        ));
    }
    params
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReduceL2Operation {
    global_id: GlobalId,
    keepdims: Option<bool>,
    noop_with_empty_axes: Option<bool>,
    input_data: GlobalId,
    input_axes: Option<GlobalId>,
    axes_attr: Option<Vec<i64>>,
    output: GlobalId,
}

impl ReduceL2Operation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.is_empty() || inputs.len() > 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("ReduceL2"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("ReduceL2"));
        }

        let axes_attr = query_attribute_ints(attributes, "axes");
        let keepdims = query_attribute_int(attributes, "keepdims").map(|x| x != 0);
        let noop_with_empty_axes =
            query_attribute_int(attributes, "noop_with_empty_axes").map(|x| x != 0);

        Ok(Self {
            global_id: GlobalId::new(rng),
            keepdims,
            noop_with_empty_axes,
            input_data: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("ReduceL2"))?,
            input_axes: if inputs.len() > 1 {
                Some(inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("ReduceL2"))?)
            } else {
                None
            },
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("ReduceL2"))?,
            axes_attr,
        })
    }
}

impl Node for ReduceL2Operation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "ReduceL2".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        if let Some(input_axes) = self.input_axes {
            Box::new([self.input_data, input_axes].into_iter())
        } else {
            Box::new(std::iter::once(self.input_data))
        }
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for ReduceL2Operation {
    fn parameters(&self) -> Vec<Property> {
        let mut params = Vec::new();
        if let Some(keepdims) = self.keepdims {
            params.push(Property::new("keepdims", PropertyValue::Bool(keepdims)));
        }
        if let Some(axes) = &self.axes_attr {
            params.push(Property::new("axes", PropertyValue::IntList(axes.clone())));
        }
        if let Some(noop) = self.noop_with_empty_axes {
            params.push(Property::new(
                "noop_with_empty_axes",
                PropertyValue::Bool(noop),
            ));
        }
        params
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        // ReduceL2(x) = sqrt(ReduceSum(x^2))
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let x = input_map[&self.input_data];
        let x_sq = milli_graph::ops::SimpleBinary::mul(&mut graph, x, x, rng);
        let axes = if let Some(input_axes) = &self.input_axes {
            Some(input_map[input_axes])
        } else if let Some(axes) = &self.axes_attr {
            let tid = milli_graph::ops::Constant::from_vec(&mut graph, axes.clone(), rng);
            Some(tid)
        } else {
            None
        };
        let sum = milli_graph::ops::ReduceSum::push_new(
            &mut graph,
            x_sq,
            axes,
            self.keepdims.unwrap_or(true),
            self.noop_with_empty_axes.unwrap_or(false),
            rng,
        );
        let out = milli_graph::ops::SimpleUnaryOp::sqrt(&mut graph, sum, rng);
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
