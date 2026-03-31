use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::{self, MilliLoweringContext, MilliOpGraph};
use crate::numeric_dtype::NumericDType;
use crate::symbolic_graph::ops::Operation;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Quantized matrix multiplication (fused dequantize-transpose-matmul).
///
/// Computes `output = input @ dequant(weight)^T` where:
/// - `input` has shape `[..., K]` and is a standard numeric tensor (F32/BF16)
/// - `weight` has shape `[N, K]` and may be a packed (quantized) tensor
/// - `output` has shape `[..., N]`
///
/// The weight is stored in row-major `[out_features, in_features]` layout,
/// matching the convention used by GGUF and PyTorch linear layers.
///
/// At eval time, if the weight is packed, it is dequantized to F32 before
/// the transpose and matmul are performed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuantMatMulOperation {
    global_id: GlobalId,
    input: GlobalId,
    weight: GlobalId,
    output: GlobalId,
}

impl QuantMatMulOperation {
    pub fn new(input: GlobalId, weight: GlobalId, output: GlobalId, rng: &mut impl Rng) -> Self {
        Self {
            global_id: GlobalId::new(rng),
            input,
            weight,
            output,
        }
    }
}

impl Node for QuantMatMulOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "QuantMatMul".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new([self.input, self.weight].into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new([self.output].into_iter())
    }
}

impl Operation for QuantMatMulOperation {
    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let input_ids = vec![self.input, self.weight];
        let (mut graph, input_map) = MilliOpGraph::new(input_ids, rng);
        let a = input_map[&self.input];
        let b = input_map[&self.weight];
        // Transpose weight [N, K] -> [K, N]
        let bt = milli_graph::ops::Transpose::push_new(&mut graph, b, Some(vec![1, 0]), rng);
        // MatMul: input @ weight^T — weight is dequantized to F32
        let out = milli_graph::ops::MatMul::push_new_default_precision(
            &mut graph,
            a,
            bt,
            NumericDType::F32,
            rng,
        );
        graph.set_output_map(std::iter::once((out, self.output)));
        graph
    }

    fn parameters(&self) -> Vec<Property> {
        vec![Property::new(
            "operation",
            PropertyValue::String("QuantMatMul".to_string()),
        )]
    }

    fn is_differentiable(&self) -> bool {
        false
    }
}
