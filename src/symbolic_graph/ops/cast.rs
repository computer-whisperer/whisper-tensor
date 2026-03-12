use crate::dtype::DType;
use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::{self, MilliLoweringContext, MilliOpGraph};
use crate::onnx;
use crate::symbolic_graph::ONNXDecodingError;
use crate::symbolic_graph::ops::Operation;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CastLikeOperation {
    global_id: GlobalId,
    input: GlobalId,
    target_type: GlobalId,
    output: GlobalId,
}

impl CastLikeOperation {
    pub fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        _attributes: &[onnx::AttributeProto],
        rng: &mut impl rand::Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("CastLike"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("CastLike"));
        }
        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("CastLike"))?,
            target_type: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("CastLike"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("CastLike"))?,
        })
    }
}

impl Node for CastLikeOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "CastLike".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new([self.input, self.target_type].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new([self.output].into_iter())
    }
}

impl Operation for CastLikeOperation {
    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl rand::Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let out = milli_graph::ops::CastLike::push_new(
            &mut graph,
            input_map[&self.input],
            input_map[&self.target_type],
            rng,
        );
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CastOperation {
    global_id: GlobalId,
    input: GlobalId,
    output: GlobalId,
    to: DType,
}

impl CastOperation {
    pub fn new(input: GlobalId, output: GlobalId, to: DType, rng: &mut impl rand::Rng) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            input,
            output,
            to,
        }
    }

    pub fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl rand::Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Cast"));
        }
        if outputs.len() != 1 {
            return Err(ONNXDecodingError::InvalidOperatorOutputs("Cast"));
        }
        let to_i = attributes
            .iter()
            .find(|a| a.name == "to")
            .ok_or(ONNXDecodingError::MissingAttribute(
                "Cast".to_string(),
                "to".to_string(),
            ))?
            .i as i32;
        let to_datatype = onnx::tensor_proto::DataType::try_from(to_i)
            .map_err(|x| ONNXDecodingError::ProtobufDecodeError(x.into()))?;
        let to = DType::try_from(to_datatype)?;
        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Cast"))?,
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs("Cast"))?,
            to,
        })
    }
}

impl Node for CastOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Cast".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new([self.input].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new([self.output].into_iter())
    }
}

impl Operation for CastOperation {
    fn parameters(&self) -> Vec<Property> {
        vec![Property::new("to", PropertyValue::DType(self.to))]
    }

    fn eval(
        &self,
        backend: &mut crate::backends::eval_backend::EvalBackend,
        inputs: &HashMap<
            GlobalId,
            crate::numeric_tensor::NumericTensor<crate::tensor_rank::DynRank>,
        >,
    ) -> super::OperationEvalRet {
        let input = inputs
            .get(&self.input)
            .ok_or_else(|| super::EvalError::InvalidInput("Cast: missing input".into()))?;

        // Handle packed (quantized) inputs: dequantize to F32, then cast if needed
        if let crate::numeric_tensor::NumericTensor::Packed(p) = input {
            let f32_tensor = crate::numeric_tensor::NumericTensor::NDArray(p.dequantize());
            let result = if self.to == crate::dtype::DType::F32 {
                f32_tensor
            } else {
                f32_tensor
                    .cast(self.to, backend)
                    .map_err(super::EvalError::from)?
            };
            return Ok(Box::new(std::iter::once((self.output, result))));
        }

        // Default: delegate to milli graph
        let ctx = MilliLoweringContext::empty();
        let mut rng = wyrand::WyRand::new(Default::default());
        let milli_graph = self.get_milli_op_graph(&ctx, &mut rng);
        Ok(milli_graph.eval(inputs, &mut (), backend)?)
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl rand::Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);
        let out =
            milli_graph::ops::Cast::push_new(&mut graph, input_map[&self.input], self.to, rng);
        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
