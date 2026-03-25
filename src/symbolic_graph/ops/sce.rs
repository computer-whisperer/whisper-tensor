use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::{self, MilliLoweringContext, MilliOpGraph};
use crate::numeric_dtype::NumericDType;
use crate::onnx::AttributeProto;
use crate::symbolic_graph::ops::Operation;
use crate::symbolic_graph::ops::nlll::gather_class_axis;
use crate::symbolic_graph::{ONNXDecodingError, query_attribute_int, query_attribute_string};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// ONNX SoftmaxCrossEntropyLoss operator.
///
/// Computes LogSoftmax on the input along the class axis, then NLL loss.
/// Optionally outputs the log probabilities as a second output.
///
/// Inputs: scores [N, C, d1, ..., dk], labels [N, d1, ..., dk], optional weight [C]
/// Outputs: loss (reduced), optional log_prob [N, C, d1, ..., dk]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SoftmaxCrossEntropyLossOperation {
    global_id: GlobalId,
    input: GlobalId,
    labels: GlobalId,
    weight: Option<GlobalId>,
    loss_output: GlobalId,
    log_prob_output: Option<GlobalId>,
    reduction: String,
    ignore_index: Option<i64>,
}

impl SoftmaxCrossEntropyLossOperation {
    pub fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        if inputs.len() < 2 {
            return Err(ONNXDecodingError::InvalidOperatorInputs(
                "SoftmaxCrossEntropyLoss",
            ));
        }
        if outputs.is_empty() {
            return Err(ONNXDecodingError::InvalidOperatorOutputs(
                "SoftmaxCrossEntropyLoss",
            ));
        }

        let reduction =
            query_attribute_string(attributes, "reduction").unwrap_or_else(|| "mean".to_string());
        let ignore_index = query_attribute_int(attributes, "ignore_index");

        Ok(Self {
            global_id: GlobalId::new(rng),
            input: inputs[0].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                "SoftmaxCrossEntropyLoss",
            ))?,
            labels: inputs[1].ok_or(ONNXDecodingError::InvalidOperatorInputs(
                "SoftmaxCrossEntropyLoss",
            ))?,
            weight: inputs.get(2).and_then(|x| *x),
            loss_output: outputs[0].ok_or(ONNXDecodingError::InvalidOperatorOutputs(
                "SoftmaxCrossEntropyLoss",
            ))?,
            log_prob_output: outputs.get(1).and_then(|x| *x),
            reduction,
            ignore_index,
        })
    }
}

impl Node for SoftmaxCrossEntropyLossOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "SoftmaxCrossEntropyLoss".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        let mut v = vec![self.input, self.labels];
        if let Some(w) = self.weight {
            v.push(w);
        }
        Box::new(v.into_iter())
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId> + '_> {
        let mut v = vec![self.loss_output];
        if let Some(lp) = self.log_prob_output {
            v.push(lp);
        }
        Box::new(v.into_iter())
    }
}

impl Operation for SoftmaxCrossEntropyLossOperation {
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

        let scores = input_map[&self.input];
        let target = input_map[&self.labels];

        // --- LogSoftmax along axis=1 (class axis) ---
        let axis1 = milli_graph::ops::Constant::new_scalar(&mut graph, 1i64, rng);
        let row_max = milli_graph::ops::ReduceMax::push_new(
            &mut graph,
            scores,
            Some(axis1),
            true,
            false,
            rng,
        );
        let shifted = milli_graph::ops::SimpleBinary::sub(&mut graph, scores, row_max, rng);
        let exp = milli_graph::ops::SimpleUnaryOp::exp(&mut graph, shifted, rng);
        let sum_exp =
            milli_graph::ops::ReduceSum::push_new(&mut graph, exp, Some(axis1), true, false, rng);
        let log_sum = milli_graph::ops::SimpleUnaryOp::ln(&mut graph, sum_exp, rng);
        let log_prob = milli_graph::ops::SimpleBinary::sub(&mut graph, shifted, log_sum, rng);

        // --- NLL: clamp OOB targets before indexing, then zero out later ---
        let (safe_target, is_ignored) = if let Some(ii) = self.ignore_index {
            let target_i64 = milli_graph::ops::Cast::push_new(&mut graph, target, NumericDType::I64, rng);
            let ii_const = milli_graph::ops::Constant::new_scalar(&mut graph, ii, rng);
            let mask = milli_graph::ops::SimpleBinary::equal(&mut graph, target_i64, ii_const, rng);
            let zero_i64 = milli_graph::ops::Constant::new_scalar(&mut graph, 0i64, rng);
            let clamped =
                milli_graph::ops::Where::push_new(&mut graph, mask, zero_i64, target_i64, rng);
            (clamped, Some(mask))
        } else {
            (target, None)
        };

        let gathered = gather_class_axis(&mut graph, log_prob, safe_target, rng);
        let mut loss = milli_graph::ops::SimpleUnaryOp::neg(&mut graph, gathered, rng);

        // Per-sample weight from weight[C] gathered by safe target
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
            milli_graph::ops::Cast::push_new(&mut graph, flat, NumericDType::I64, rng)
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
            _ => panic!("SCE: unknown reduction '{}'", self.reduction),
        };

        let mut output_map = HashMap::new();
        output_map.insert(out, self.loss_output);
        if let Some(lp_out) = self.log_prob_output {
            output_map.insert(log_prob, lp_out);
        }
        graph.set_output_map(output_map);
        graph
    }
}
