use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::{MilliLoweringContext, MilliOpGraph};
use crate::onnx::AttributeProto;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{ONNXDecodingError, query_attribute_float, query_attribute_int};
use rand::Rng;
use serde::{Deserialize, Serialize};

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

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, _rng: &mut impl Rng) -> MilliOpGraph {
        panic!("LRN uses custom eval, not milli-op decomposition")
    }
}
