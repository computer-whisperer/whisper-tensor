use crate::dtype::DType;
use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::{self, MilliLoweringContext, MilliOpGraph};
use crate::onnx::AttributeProto;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::{ONNXDecodingError, query_attribute_int, query_attribute_string};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// ONNX NegativeLogLikelihoodLoss operator.
///
/// Computes NLL loss: for each sample, picks -log_prob[target_class] and
/// optionally weights/reduces the result.
///
/// Inputs: log_prob [N, C, d1, ..., dk], target [N, d1, ..., dk], optional weight [C]
/// Output: loss (shape depends on reduction)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NegativeLogLikelihoodLossOperation {
    global_id: GlobalId,
    input: GlobalId,          // log probabilities
    target: GlobalId,         // class indices
    weight: Option<GlobalId>, // per-class weights
    output: GlobalId,
    reduction: String, // "none", "mean", "sum"
    ignore_index: Option<i64>,
}

impl NegativeLogLikelihoodLossOperation {
    pub fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() < 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs(
                "NegativeLogLikelihoodLoss",
            ));
        }
        if outputs.is_empty() {
            return Err(ONNXDecodingError::InvalidOperatorOutputs(
                "NegativeLogLikelihoodLoss",
            ));
        }

        let reduction =
            query_attribute_string(attributes, "reduction").unwrap_or_else(|| "mean".to_string());
        let ignore_index = query_attribute_int(attributes, "ignore_index");

        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                "NegativeLogLikelihoodLoss",
            ))?,
            target: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                "NegativeLogLikelihoodLoss",
            ))?,
            weight: inputs.get(2).and_then(|x| *x),
            output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs(
                "NegativeLogLikelihoodLoss",
            ))?,
            reduction,
            ignore_index,
        })
    }
}

impl Node for NegativeLogLikelihoodLossOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "NegativeLogLikelihoodLoss".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        let mut v = vec![self.input, self.target];
        if let Some(w) = self.weight {
            v.push(w);
        }
        Box::new(v.into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        Box::new([self.output].into_iter())
    }
}

/// Build the GatherElements-equivalent subgraph for axis=1:
/// Given log_prob [N, C, d...] and target [N, d...], produce gathered [N, d...]
/// by computing flat linear indices and using Gather on flattened data.
pub(super) fn gather_class_axis(
    graph: &mut MilliOpGraph,
    log_prob: GlobalId,
    target: GlobalId,
    rng: &mut impl Rng,
) -> GlobalId {
    let idx0 = milli_graph::ops::Constant::new_scalar(graph, 0i64, rng);
    let idx1 = milli_graph::ops::Constant::new_scalar(graph, 1i64, rng);
    let neg1 = milli_graph::ops::Constant::new_scalar(graph, -1i64, rng);

    // Get shape info
    let shape_lp = milli_graph::ops::Shape::push_new(graph, log_prob, rng);
    let target_shape = milli_graph::ops::Shape::push_new(graph, target, rng);

    // N = shape_lp[0], C = shape_lp[1]
    let n_val = milli_graph::ops::Gather::push_new(graph, shape_lp, idx0, 0, rng);
    let c_val = milli_graph::ops::Gather::push_new(graph, shape_lp, idx1, 0, rng);

    // Flatten target -> [N_total]
    let flat_shape = milli_graph::ops::Reshape::push_new(graph, neg1, neg1, false, rng);
    let flat_target = milli_graph::ops::Reshape::push_new(graph, target, flat_shape, false, rng);

    // Cast target to i64 for index arithmetic
    let flat_target_i64 = milli_graph::ops::Cast::push_new(graph, flat_target, DType::I64, rng);

    // N_total = product of target shape elements
    let target_flat_shape = milli_graph::ops::Shape::push_new(graph, flat_target_i64, rng);
    let n_total = milli_graph::ops::Gather::push_new(graph, target_flat_shape, idx0, 0, rng);

    // S = N_total / N (spatial product)
    let s_val = milli_graph::ops::SimpleBinary::div(graph, n_total, n_val, rng);

    // C * S
    let c_times_s = milli_graph::ops::SimpleBinary::mul(graph, c_val, s_val, rng);

    // arange [0..N_total)
    let arange = milli_graph::ops::Range::push_new(graph, idx0, n_total, idx1, rng);

    // batch = arange / S
    let batch = milli_graph::ops::SimpleBinary::div(graph, arange, s_val, rng);
    // spatial = arange - batch * S  (= arange % S)
    let batch_times_s = milli_graph::ops::SimpleBinary::mul(graph, batch, s_val, rng);
    let spatial = milli_graph::ops::SimpleBinary::sub(graph, arange, batch_times_s, rng);

    // flat_index = batch * C * S + target * S + spatial
    let offset_batch = milli_graph::ops::SimpleBinary::mul(graph, batch, c_times_s, rng);
    let offset_target = milli_graph::ops::SimpleBinary::mul(graph, flat_target_i64, s_val, rng);
    let idx_sum = milli_graph::ops::SimpleBinary::add(graph, offset_batch, offset_target, rng);
    let flat_index = milli_graph::ops::SimpleBinary::add(graph, idx_sum, spatial, rng);

    // Flatten log_prob and gather
    let flat_lp = milli_graph::ops::Reshape::push_new(graph, log_prob, flat_shape, false, rng);
    let gathered_flat = milli_graph::ops::Gather::push_new(graph, flat_lp, flat_index, 0, rng);

    // Reshape back to target shape
    milli_graph::ops::Reshape::push_new(graph, gathered_flat, target_shape, false, rng)
}

impl Operation for NegativeLogLikelihoodLossOperation {
    fn parameters(&self) -> Vec<Property> {
        let mut p = vec![Property::new(
            "reduction",
            PropertyValue::String(self.reduction.clone()),
        )];
        if let Some(ii) = self.ignore_index {
            p.push(Property::new("ignore_index", PropertyValue::Int(ii)));
        }
        p
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, rng: &mut impl Rng) -> MilliOpGraph {
        let (mut graph, input_map) = MilliOpGraph::new(self.inputs(), rng);

        let log_prob = input_map[&self.input];
        let target = input_map[&self.target];

        // When ignore_index is set, target values may be out-of-bounds (e.g. 10
        // when C=5). Clamp them to 0 before indexing, then zero out loss later.
        let (safe_target, is_ignored) = if let Some(ii) = self.ignore_index {
            let target_i64 = milli_graph::ops::Cast::push_new(&mut graph, target, DType::I64, rng);
            let ii_const = milli_graph::ops::Constant::new_scalar(&mut graph, ii, rng);
            let mask = milli_graph::ops::SimpleBinary::equal(&mut graph, target_i64, ii_const, rng);
            let zero_i64 = milli_graph::ops::Constant::new_scalar(&mut graph, 0i64, rng);
            // Replace ignored indices with 0 so gathers don't go OOB
            let clamped =
                milli_graph::ops::Where::push_new(&mut graph, mask, zero_i64, target_i64, rng);
            (clamped, Some(mask))
        } else {
            (target, None)
        };

        // Gather log_prob at target class indices along axis 1
        let gathered = gather_class_axis(&mut graph, log_prob, safe_target, rng);

        // Negate: loss = -log_prob[class]
        let mut loss = milli_graph::ops::SimpleUnaryOp::neg(&mut graph, gathered, rng);

        // Per-sample weight from weight[C] gathered by target
        let flat_safe_target = {
            let neg1 = milli_graph::ops::Constant::new_scalar(&mut graph, -1i64, rng);
            let flat_shape =
                milli_graph::ops::Reshape::push_new(&mut graph, neg1, neg1, false, rng);
            let flat = milli_graph::ops::Reshape::push_new(
                &mut graph,
                safe_target,
                flat_shape,
                false,
                rng,
            );
            milli_graph::ops::Cast::push_new(&mut graph, flat, DType::I64, rng)
        };

        let mut sample_weight = if let Some(w_id) = self.weight {
            let w = input_map[&w_id];
            let w_flat =
                milli_graph::ops::Gather::push_new(&mut graph, w, flat_safe_target, 0, rng);
            let target_shape = milli_graph::ops::Shape::push_new(&mut graph, target, rng);
            let w_per_sample =
                milli_graph::ops::Reshape::push_new(&mut graph, w_flat, target_shape, false, rng);
            loss = milli_graph::ops::SimpleBinary::mul(&mut graph, loss, w_per_sample, rng);
            Some(w_per_sample)
        } else {
            None
        };

        // Zero out loss and weights for ignored samples
        if let Some(mask) = is_ignored {
            let zero = milli_graph::ops::Constant::new_scalar(&mut graph, 0.0f32, rng);
            loss = milli_graph::ops::Where::push_new(&mut graph, mask, zero, loss, rng);

            if let Some(sw) = sample_weight {
                let w_masked = milli_graph::ops::Where::push_new(&mut graph, mask, zero, sw, rng);
                sample_weight = Some(w_masked);
            } else {
                let one = milli_graph::ops::Constant::new_scalar(&mut graph, 1.0f32, rng);
                let w_masked = milli_graph::ops::Where::push_new(&mut graph, mask, zero, one, rng);
                sample_weight = Some(w_masked);
            }
        }

        // Apply reduction
        let out = match self.reduction.as_str() {
            "none" => loss,
            "sum" => {
                milli_graph::ops::ReduceSum::push_new(&mut graph, loss, None, false, false, rng)
            }
            "mean" => {
                if let Some(sw) = sample_weight {
                    // Weighted or ignore_index mean: sum(loss) / sum(weights)
                    let sum = milli_graph::ops::ReduceSum::push_new(
                        &mut graph, loss, None, false, false, rng,
                    );
                    let w_sum = milli_graph::ops::ReduceSum::push_new(
                        &mut graph, sw, None, false, false, rng,
                    );
                    milli_graph::ops::SimpleBinary::div(&mut graph, sum, w_sum, rng)
                } else {
                    milli_graph::ops::ReduceMean::push_new(
                        &mut graph, loss, None, false, false, rng,
                    )
                }
            }
            _ => panic!("NLLL: unknown reduction '{}'", self.reduction),
        };

        let mut output_map = HashMap::new();
        output_map.insert(out, self.output);
        graph.set_output_map(output_map);
        graph
    }
}
