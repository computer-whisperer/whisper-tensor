use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::{self, MilliOpGraph};
use crate::numeric_tensor::NumericTensor;
use crate::symbolic_graph::ops::{EvalError, Operation, OperationEvalRet};
use crate::tensor_rank::DynRank;
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
    fn eval(
        &self,
        backend: &mut crate::backends::eval_backend::EvalBackend,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
    ) -> OperationEvalRet {
        let weight = inputs
            .get(&self.weight)
            .ok_or_else(|| EvalError::InvalidInput("QuantMatMul: missing weight".into()))?;

        // Dequantize packed weight to F32 if needed
        let weight_f32: NumericTensor<DynRank> = match weight {
            NumericTensor::Packed(p) => NumericTensor::NDArray(p.dequantize()),
            other => other.clone(),
        };

        // Replace the packed weight with the dequantized version and delegate
        // to the milli graph (Transpose + MatMul)
        let mut patched = inputs.clone();
        patched.insert(self.weight, weight_f32);

        let mut rng = wyrand::WyRand::new(Default::default());
        let milli_graph = self.get_milli_op_graph(&mut rng);
        Ok(milli_graph.eval(&patched, &mut (), backend)?)
    }

    fn get_milli_op_graph(&self, rng: &mut impl Rng) -> MilliOpGraph {
        let input_ids = vec![self.input, self.weight];
        let (mut graph, input_map) = MilliOpGraph::new(input_ids, rng);
        let a = input_map[&self.input];
        let b = input_map[&self.weight];
        // Transpose weight [N, K] -> [K, N]
        let bt = milli_graph::ops::Transpose::push_new(&mut graph, b, Some(vec![1, 0]), rng);
        // MatMul: input @ weight^T
        let out = milli_graph::ops::MatMul::push_new(&mut graph, a, bt, rng);
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
