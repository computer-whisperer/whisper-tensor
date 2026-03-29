use crate::graph::{GlobalId, Graph, Node, Property, PropertyValue};
use crate::milli_graph::ops::*;
use crate::milli_graph::{MilliLoweringContext, MilliOpGraph};
use crate::numeric_scalar::NumericScalar as NewNumericScalar;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{
    ONNXDecodingError, SharedPoolTensor, query_attribute_float, query_attribute_floats,
    query_attribute_int, query_attribute_ints, query_attribute_tensor,
};
use crate::{DynRank, onnx};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConstantOfShapeOperation {
    global_id: GlobalId,
    value: NewNumericScalar,
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
            .map(|x| {
                // Extract first element as new-type scalar.
                crate::nano_graph::lower::legacy_scalar_to_new(&x.first_element())
            })
            .unwrap_or(NewNumericScalar::from_f32(0.0));

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
        vec![Property::new(
            "value",
            PropertyValue::String(format!("{}", self.value)),
        )]
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let out = ConstantOfShape::push_new(&mut graph, self.value, input_map[&self.input], rng);
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConstantOperation {
    global_id: GlobalId,
    pub value: SharedPoolTensor,
    output: GlobalId,
}

impl ConstantOperation {
    pub fn new(value: SharedPoolTensor, output: GlobalId, rng: &mut impl Rng) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            value,
            output,
        }
    }

    /// Helper: create a pool tensor from typed scalar values.
    fn pool_tensor_from_f32(vals: Vec<f32>) -> SharedPoolTensor {
        use crate::numeric_dtype::NumericDType;
        use crate::numeric_tensor::{NumericTensor, TensorLayout};
        use crate::pool::{Pool, SystemPool};
        let shape = vec![vals.len() as u64];
        let layout = TensorLayout::<DynRank>::row_major(shape, NumericDType::F32);
        let buf = SystemPool.allocate(layout.buffer_size_bytes()).unwrap();
        let mut t = NumericTensor::from_parts(buf, layout);
        for (i, &v) in vals.iter().enumerate() {
            t.write_element(i, NewNumericScalar::from_f32(v));
        }
        SharedPoolTensor(std::sync::Arc::new(t))
    }

    fn pool_tensor_from_i64(vals: Vec<i64>) -> SharedPoolTensor {
        use crate::numeric_dtype::NumericDType;
        use crate::numeric_tensor::{NumericTensor, TensorLayout};
        use crate::pool::{Pool, SystemPool};
        let shape = vec![vals.len() as u64];
        let layout = TensorLayout::<DynRank>::row_major(shape, NumericDType::I64);
        let buf = SystemPool.allocate(layout.buffer_size_bytes()).unwrap();
        let mut t = NumericTensor::from_parts(buf, layout);
        for (i, &v) in vals.iter().enumerate() {
            t.write_element(i, NewNumericScalar::from_i64(v));
        }
        SharedPoolTensor(std::sync::Arc::new(t))
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
            // Bridge legacy NDArray → pool tensor.
            let pool_t = crate::symbolic_graph::tensor_proto_to_pool_tensor_from_ndarray(&tensor)?;
            SharedPoolTensor(std::sync::Arc::new(pool_t))
        } else if let Some(value_float) = query_attribute_float(attributes, "value_float") {
            Self::pool_tensor_from_f32(vec![value_float])
        } else if let Some(value_floats) = query_attribute_floats(attributes, "value_floats") {
            Self::pool_tensor_from_f32(value_floats)
        } else if let Some(value_int) = query_attribute_int(attributes, "value_int") {
            Self::pool_tensor_from_i64(vec![value_int])
        } else if let Some(value_ints) = query_attribute_ints(attributes, "value_ints") {
            Self::pool_tensor_from_i64(value_ints)
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
        let t = &*self.value.0;
        let shape: Vec<i64> = t.shape().iter().map(|&x| x as i64).collect();
        let total_elements = t.numel();

        let mut params = vec![
            Property::new("dtype", PropertyValue::DType(t.dtype().to_legacy())),
            Property::new("shape", PropertyValue::IntList(shape)),
        ];

        if total_elements == 1 {
            params.push(Property::new(
                "value",
                PropertyValue::String(format!("{}", t.read_element(0))),
            ));
        } else if total_elements <= 8 {
            let preview: Vec<String> = (0..total_elements)
                .map(|i| format!("{:.4}", t.read_element(i).to_f32()))
                .collect();
            params.push(Property::new(
                "values",
                PropertyValue::String(format!("[{}]", preview.join(", "))),
            ));
        }

        params
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, _input_map) = MilliOpGraph::new(self.inputs(), rng);

        let out = Constant::push_new_pool(&mut graph, self.value.clone(), None, rng);

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
