use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::graph::{GlobalId, Node};
use crate::milli_graph::{self, MilliOpGraph};
use crate::onnx;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{
    ONNXDecodingError, query_attribute_int, query_attribute_ints,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rand::Rng;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SplitOperation {
    global_id: GlobalId,
    axis: Option<i64>,
    num_outputs: Option<i64>,
    input: GlobalId,
    split: Option<GlobalId>,
    split_attribute: Option<Vec<i64>>,
    outputs: Vec<GlobalId>,
}

impl SplitOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.is_empty() || inputs.len() > 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs("Split"));
        }

        let axis = query_attribute_int(attributes, "axis");
        let num_outputs = query_attribute_int(attributes, "num_outputs");
        let split_attribute = query_attribute_ints(attributes, "split");

        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs("Split"))?,
            split: if inputs.len() > 1 {
                Some(inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs("Split"))?)
            } else {
                None
            },
            outputs: outputs
                .iter()
                .map(|x| x.ok_or(ONNXDecodingError::InvalidOperatorOutputs("Split")))
                .collect::<Result<_, _>>()?,
            split_attribute,
            axis,
            num_outputs,
        })
    }
}

impl Node for SplitOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "Split".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        let mut inputs = vec![self.input];
        if let Some(split) = self.split {
            inputs.push(split);
        }
        Box::new(inputs.into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(self.outputs.clone().into_iter())
    }
}
impl Operation for SplitOperation {
    fn get_milli_op_graph(&self, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);

        let mut output_map = HashMap::new();

        let split = if let Some(split) = self.split {
            Some(milli_graph::ops::MilliOpTensorIDOrLiteral::TensorID(
                input_map[&split],
            ))
        } else {
            self.split_attribute.as_ref().map(|split| {
                milli_graph::ops::MilliOpTensorIDOrLiteral::Literal(
                    NDArrayNumericTensor::from_vec(split.clone()).to_dyn(),
                )
            })
        };

        for (output_id, output_tensor_id) in self.outputs.iter().enumerate() {
            let out = milli_graph::ops::Split::push_new(
                &mut graph,
                input_map[&self.input],
                split.clone(),
                self.axis.unwrap_or_default(),
                self.num_outputs.map(|x| x as usize),
                output_id,
                rng
            );

            output_map.insert(out, *output_tensor_id);
        }
        graph.set_output_map(output_map);
        graph
    }
}
