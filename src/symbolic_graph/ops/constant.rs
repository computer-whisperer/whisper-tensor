use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::graph::{GlobalId, Graph, Node, Property, PropertyValue};
use crate::milli_graph::{MilliLoweringContext, MilliOpGraph};
use crate::milli_graph::ops::*;
use crate::numeric_scalar::NumericScalar;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{
    ONNXDecodingError, query_attribute_float, query_attribute_floats, query_attribute_int,
    query_attribute_ints, query_attribute_tensor,
};
use crate::{DynRank, onnx};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConstantOfShapeOperation {
    global_id: GlobalId,
    value: NumericScalar,
    input: GlobalId,
    output: GlobalId,
}

impl ConstantOfShapeOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("ConstantOfShape"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("ConstantOfShape"));
        }

        let value = query_attribute_tensor(attributes, "value")
            .map(|x| x.first_element())
            .unwrap_or(NumericScalar::F32(0.0));

        Ok(Self {
            global_id: GlobalId::new(rng),
            value,
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("ConstantOfShape"))?,
            output: outputs[0]
                .ok_or(ONNXDecodingError::InvalidOperatorOutputs("ConstantOfShape"))?,
        })
    }
}

impl Node for ConstantOfShapeOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "ConstantOfShape".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.input))
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for ConstantOfShapeOperation {
    fn parameters(&self) -> Vec<Property> {
        let value_str = match &self.value {
            NumericScalar::F32(v) => format!("{}", v),
            NumericScalar::F64(v) => format!("{}", v),
            NumericScalar::I32(v) => format!("{}", v),
            NumericScalar::I64(v) => format!("{}", v),
            NumericScalar::U8(v) => format!("{}", v),
            NumericScalar::U32(v) => format!("{}", v),
            NumericScalar::BOOL(v) => format!("{}", v),
            _ => format!("{:?}", self.value),
        };
        vec![Property::new("value", PropertyValue::String(value_str))]
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let node =
            ConstantOfShape::push_new(&mut graph, self.value.clone(), input_map[&self.input], rng);
        let out = match graph.get_node_by_id(&node) {
            Some(AnyMilliOp::ConstantOfShape(op)) => op.outputs().next().unwrap(),
            _ => unreachable!(),
        };
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConstantOperation {
    global_id: GlobalId,
    pub value: NDArrayNumericTensor<DynRank>,
    output: GlobalId,
}

impl ConstantOperation {
    pub fn new(value: NDArrayNumericTensor<DynRank>, output: GlobalId, rng: &mut impl Rng) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            value,
            output,
        }
    }

    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if !inputs.is_empty() {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Constant"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Constant"));
        }

        let value = if let Some(tensor) = query_attribute_tensor(attributes, "value") {
            tensor
        } else if let Some(value_float) = query_attribute_float(attributes, "value_float") {
            NDArrayNumericTensor::from(vec![value_float]).try_to_rank()?
        } else if let Some(value_floats) = query_attribute_floats(attributes, "value_floats") {
            NDArrayNumericTensor::from(value_floats).try_to_rank()?
        } else if let Some(value_int) = query_attribute_int(attributes, "value_int") {
            NDArrayNumericTensor::from(vec![value_int]).try_to_rank()?
        } else if let Some(value_ints) = query_attribute_ints(attributes, "value_ints") {
            NDArrayNumericTensor::from(value_ints).try_to_rank()?
        } else {
            Err(ONNXDecodingError::MissingAttribute(
                "Constant".to_string(),
                "value".to_string(),
            ))?
        };

        Ok(Self {
            global_id: GlobalId::new(rng),
            value,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Constant"))?,
        })
    }
}

impl Node for ConstantOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Constant".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::empty())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(std::iter::once(self.output))
    }
}

impl Operation for ConstantOperation {
    fn parameters(&self) -> Vec<Property> {
        let shape: Vec<i64> = self.value.shape().iter().map(|&x| x as i64).collect();
        let total_elements: usize = shape.iter().map(|&x| x as usize).product();

        let mut params = vec![
            Property::new("dtype", PropertyValue::DType(self.value.dtype())),
            Property::new("shape", PropertyValue::IntList(shape)),
        ];

        // For small constants, show the actual value
        if total_elements == 1 {
            let value_str = format!("{}", self.value.first_element());
            params.push(Property::new("value", PropertyValue::String(value_str)));
        } else if total_elements <= 8 {
            // Show first few elements for small tensors
            let flat = self.value.flatten();
            if let Ok(values) = TryInto::<Vec<f32>>::try_into(flat) {
                let preview: Vec<String> = values.iter().map(|v| format!("{:.4}", v)).collect();
                params.push(Property::new(
                    "values",
                    PropertyValue::String(format!("[{}]", preview.join(", "))),
                ));
            }
        }

        params
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, _input_map) = MilliOpGraph::new(self.inputs(), rng);

        let out = Constant::push_new(&mut graph, self.value.clone(), rng);

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
