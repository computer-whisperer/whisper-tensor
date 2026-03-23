use crate::TrigOp;
use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::{MilliLoweringContext, MilliOpGraph};
use crate::migration::numeric_tensor::NumericTensor;
use crate::onnx;
use crate::symbolic_graph::ops::{EvalError, Operation};
use crate::symbolic_graph::{ONNXDecodingError, query_attribute_int, query_attribute_string};
use crate::tensor_rank::DynRank;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum RnnDirection {
    Forward,
    Reverse,
    Bidirectional,
}

fn parse_direction(attributes: &[onnx::AttributeProto]) -> RnnDirection {
    match query_attribute_string(attributes, "direction").as_deref() {
        Some("reverse") => RnnDirection::Reverse,
        Some("bidirectional") => RnnDirection::Bidirectional,
        _ => RnnDirection::Forward,
    }
}

fn r(start: usize, end: usize) -> std::ops::Range<u64> {
    start as u64..end as u64
}

fn zeros_f32(shape: &[u64]) -> Result<NumericTensor<DynRank>, EvalError> {
    let n: u64 = shape.iter().product();
    Ok(NumericTensor::NDArray(
        NDArrayNumericTensor::from_vec_shape(vec![0.0f32; n as usize], &shape.to_vec())?,
    ))
}

fn scalar_f32(v: f32) -> Result<NumericTensor<DynRank>, EvalError> {
    Ok(NumericTensor::NDArray(
        NDArrayNumericTensor::from_vec_shape(vec![v], &vec![1u64])?,
    ))
}

fn sigmoid(
    x: &NumericTensor<DynRank>,
    backend: &mut EvalBackend,
) -> Result<NumericTensor<DynRank>, EvalError> {
    let neg_x = x.neg(backend)?;
    let exp_neg = neg_x.exp(backend)?;
    let one = scalar_f32(1.0)?;
    let denom = NumericTensor::add(&one, &exp_neg, backend)?;
    Ok(denom.reciprocal(backend)?)
}

fn apply_activation(
    x: &NumericTensor<DynRank>,
    activation: &str,
    backend: &mut EvalBackend,
) -> Result<NumericTensor<DynRank>, EvalError> {
    match activation {
        "Tanh" => Ok(x.trig(TrigOp::Tanh, backend)?),
        "Relu" => {
            let zero = scalar_f32(0.0)?;
            Ok(NumericTensor::max(x, &zero, backend)?)
        }
        "Sigmoid" => sigmoid(x, backend),
        _ => Err(EvalError::UnimplementedOperatorError(format!(
            "RNN activation: {activation}"
        ))),
    }
}

// ─── SimpleRNN ────────────────────────────────────────────────────────────────

/// ONNX RNN (SimpleRNN) operator.
///
/// H_t = activation(X_t * W^T + H_{t-1} * R^T + Wb + Rb)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimpleRnnOperation {
    global_id: GlobalId,
    inputs: Vec<Option<GlobalId>>,
    outputs: Vec<Option<GlobalId>>,
    hidden_size: usize,
    direction: RnnDirection,
    layout: usize,
    activation: String,
}

impl SimpleRnnOperation {
    pub fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        let hidden_size = query_attribute_int(attributes, "hidden_size")
            .ok_or(ONNXDecodingError::MissingField("hidden_size"))?
            as usize;

        let direction = parse_direction(attributes);
        let layout = query_attribute_int(attributes, "layout").unwrap_or(0) as usize;

        // activations attribute: list of activation functions. Default is ["Tanh"]
        let activation = {
            let attr = attributes.iter().find(|a| a.name == "activations");
            if let Some(a) = attr {
                if !a.strings.is_empty() {
                    String::from_utf8(a.strings[0].clone()).unwrap_or_else(|_| "Tanh".into())
                } else {
                    "Tanh".into()
                }
            } else {
                "Tanh".into()
            }
        };

        Ok(Self {
            global_id: GlobalId::new(rng),
            inputs: inputs.to_vec(),
            outputs: outputs.to_vec(),
            hidden_size,
            direction,
            layout,
            activation,
        })
    }
}

impl Node for SimpleRnnOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "RNN".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(
            self.inputs
                .iter()
                .filter_map(|x| *x)
                .collect::<Vec<_>>()
                .into_iter(),
        )
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(
            self.outputs
                .iter()
                .filter_map(|x| *x)
                .collect::<Vec<_>>()
                .into_iter(),
        )
    }
}

impl Operation for SimpleRnnOperation {
    fn parameters(&self) -> Vec<Property> {
        vec![
            Property::new("hidden_size", PropertyValue::Int(self.hidden_size as i64)),
            Property::new(
                "direction",
                PropertyValue::String(format!("{:?}", self.direction)),
            ),
        ]
    }

    fn is_differentiable(&self) -> bool {
        false
    }

    fn eval(
        &self,
        backend: &mut EvalBackend,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, EvalError> {
        // Inputs: X, W, R, B (optional), sequence_lens (optional), initial_h (optional)
        let x = &inputs[&self.inputs[0].unwrap()];
        let w = &inputs[&self.inputs[1].unwrap()];
        let rv = &inputs[&self.inputs[2].unwrap()];

        let get_opt = |idx: usize| -> Option<&NumericTensor<DynRank>> {
            self.inputs
                .get(idx)
                .and_then(|id| id.as_ref())
                .and_then(|id| inputs.get(id))
        };

        let b = get_opt(3);
        let init_h = get_opt(5);

        let x_seq_first = if self.layout == 1 {
            x.transpose(Some(vec![1, 0, 2]), backend)?
        } else {
            x.clone()
        };

        let seq_len = x_seq_first.shape()[0] as usize;
        let batch = x_seq_first.shape()[1] as usize;
        let input_size = x_seq_first.shape()[2] as usize;
        let hs = self.hidden_size;
        let num_dirs: usize = match self.direction {
            RnnDirection::Bidirectional => 2,
            _ => 1,
        };

        let mut all_y: Vec<NumericTensor<DynRank>> = Vec::new();
        let mut all_yh: Vec<NumericTensor<DynRank>> = Vec::new();

        for dir in 0..num_dirs {
            let reverse = match self.direction {
                RnnDirection::Reverse => true,
                RnnDirection::Bidirectional => dir == 1,
                _ => false,
            };

            // W_dir [hs, input_size], R_dir [hs, hs]
            let w_dir = w.slice(&[r(dir, dir + 1), r(0, hs), r(0, input_size)], backend)?;
            let w_dir = w_dir.squeeze(0)?;
            let r_dir = rv.slice(&[r(dir, dir + 1), r(0, hs), r(0, hs)], backend)?;
            let r_dir = r_dir.squeeze(0)?;

            let w_t = w_dir.transpose(Some(vec![1, 0]), backend)?;
            let r_t = r_dir.transpose(Some(vec![1, 0]), backend)?;

            let bias = if let Some(b) = b {
                let b_dir = b.slice(&[r(dir, dir + 1), r(0, 2 * hs)], backend)?;
                let b_dir = b_dir.squeeze(0)?;
                let wb = b_dir.slice(&[r(0, hs)], backend)?;
                let rb = b_dir.slice(&[r(hs, 2 * hs)], backend)?;
                Some(NumericTensor::add(&wb, &rb, backend)?)
            } else {
                None
            };

            let mut h_t = if let Some(init_h) = init_h {
                if self.layout == 1 {
                    let h = init_h.slice(&[r(0, batch), r(dir, dir + 1), r(0, hs)], backend)?;
                    h.squeeze(1)?
                } else {
                    let h = init_h.slice(&[r(dir, dir + 1), r(0, batch), r(0, hs)], backend)?;
                    h.squeeze(0)?
                }
            } else {
                zeros_f32(&[batch as u64, hs as u64])?
            };

            let mut h_seq: Vec<NumericTensor<DynRank>> = Vec::with_capacity(seq_len);

            for t in 0..seq_len {
                let actual_t = if reverse { seq_len - 1 - t } else { t };
                let x_t = x_seq_first.slice(
                    &[r(actual_t, actual_t + 1), r(0, batch), r(0, input_size)],
                    backend,
                )?;
                let x_t = x_t.squeeze(0)?;

                // H_t = activation(X_t @ W^T + H_{t-1} @ R^T + bias)
                let xw = NumericTensor::matmul(
                    &x_t,
                    &w_t,
                    None,
                    x_t.dtype(),
                    crate::milli_graph::ops::AccumulationMode::default(),
                    backend,
                )?;
                let hr = NumericTensor::matmul(
                    &h_t,
                    &r_t,
                    None,
                    x_t.dtype(),
                    crate::milli_graph::ops::AccumulationMode::default(),
                    backend,
                )?;
                let mut gates = NumericTensor::add(&xw, &hr, backend)?;
                if let Some(ref bias) = bias {
                    gates = NumericTensor::add(&gates, bias, backend)?;
                }

                h_t = apply_activation(&gates, &self.activation, backend)?;
                h_seq.push(h_t.clone());
            }

            if reverse {
                h_seq.reverse();
            }

            // Stack -> [seq_len, 1, batch, hidden]
            let h_seq_unsqueezed: Vec<NumericTensor<DynRank>> = h_seq
                .iter()
                .map(|h| h.unsqueeze(0).and_then(|h| h.unsqueeze(1)))
                .collect::<Result<_, _>>()?;
            let h_seq_refs: Vec<&NumericTensor<DynRank>> = h_seq_unsqueezed.iter().collect();
            let y_dir = NumericTensor::concat(&h_seq_refs, 0, backend)?;
            all_y.push(y_dir);

            if self.layout == 1 {
                all_yh.push(h_t.unsqueeze(1)?);
            } else {
                all_yh.push(h_t.unsqueeze(0)?);
            }
        }

        let mut result = HashMap::new();

        if let Some(y_id) = self.outputs.first().and_then(|x| *x) {
            let y_refs: Vec<&NumericTensor<DynRank>> = all_y.iter().collect();
            let y = NumericTensor::concat(&y_refs, 1, backend)?;
            let y = if self.layout == 1 {
                y.transpose(Some(vec![2, 0, 1, 3]), backend)?
            } else {
                y
            };
            result.insert(y_id, y);
        }
        if let Some(yh_id) = self.outputs.get(1).and_then(|x| *x) {
            let yh_refs: Vec<&NumericTensor<DynRank>> = all_yh.iter().collect();
            let cat_dim = if self.layout == 1 { 1 } else { 0 };
            let yh = NumericTensor::concat(&yh_refs, cat_dim, backend)?;
            result.insert(yh_id, yh);
        }

        Ok(Box::new(result.into_iter()))
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, _rng: &mut impl Rng) -> MilliOpGraph {
        panic!("SimpleRNN uses custom eval, not milli-op decomposition")
    }
}

// ─── GRU ──────────────────────────────────────────────────────────────────────

/// ONNX GRU operator.
///
/// z_t = sigmoid(X_t * Wz^T + H_{t-1} * Rz^T + Wbz + Rbz)
/// r_t = sigmoid(X_t * Wr^T + H_{t-1} * Rr^T + Wbr + Rbr)
/// h_t = tanh(X_t * Wh^T + (r_t ⊙ H_{t-1}) * Rh^T + Wbh + Rbh)  [linear_before_reset=0]
/// h_t = tanh(X_t * Wh^T + r_t ⊙ (H_{t-1} * Rh^T + Rbh) + Wbh)  [linear_before_reset=1]
/// H_t = (1 - z_t) ⊙ h_t + z_t ⊙ H_{t-1}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GruOperation {
    global_id: GlobalId,
    inputs: Vec<Option<GlobalId>>,
    outputs: Vec<Option<GlobalId>>,
    hidden_size: usize,
    direction: RnnDirection,
    layout: usize,
    linear_before_reset: bool,
}

impl GruOperation {
    pub fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        let hidden_size = query_attribute_int(attributes, "hidden_size")
            .ok_or(ONNXDecodingError::MissingField("hidden_size"))?
            as usize;

        let direction = parse_direction(attributes);
        let layout = query_attribute_int(attributes, "layout").unwrap_or(0) as usize;
        let linear_before_reset =
            query_attribute_int(attributes, "linear_before_reset").unwrap_or(0) != 0;

        Ok(Self {
            global_id: GlobalId::new(rng),
            inputs: inputs.to_vec(),
            outputs: outputs.to_vec(),
            hidden_size,
            direction,
            layout,
            linear_before_reset,
        })
    }
}

impl Node for GruOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "GRU".to_string()
    }
    fn inputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(
            self.inputs
                .iter()
                .filter_map(|x| *x)
                .collect::<Vec<_>>()
                .into_iter(),
        )
    }
    fn outputs(&self) -> Box<dyn Iterator<Item = GlobalId>> {
        Box::new(
            self.outputs
                .iter()
                .filter_map(|x| *x)
                .collect::<Vec<_>>()
                .into_iter(),
        )
    }
}

impl Operation for GruOperation {
    fn parameters(&self) -> Vec<Property> {
        vec![
            Property::new("hidden_size", PropertyValue::Int(self.hidden_size as i64)),
            Property::new(
                "direction",
                PropertyValue::String(format!("{:?}", self.direction)),
            ),
            Property::new(
                "linear_before_reset",
                PropertyValue::Int(self.linear_before_reset as i64),
            ),
        ]
    }

    fn is_differentiable(&self) -> bool {
        false
    }

    fn eval(
        &self,
        backend: &mut EvalBackend,
        inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
    ) -> Result<Box<dyn Iterator<Item = (GlobalId, NumericTensor<DynRank>)>>, EvalError> {
        // Inputs: X, W, R, B (optional), sequence_lens (optional), initial_h (optional)
        let x = &inputs[&self.inputs[0].unwrap()];
        let w = &inputs[&self.inputs[1].unwrap()];
        let rv = &inputs[&self.inputs[2].unwrap()];

        let get_opt = |idx: usize| -> Option<&NumericTensor<DynRank>> {
            self.inputs
                .get(idx)
                .and_then(|id| id.as_ref())
                .and_then(|id| inputs.get(id))
        };

        let b = get_opt(3);
        let init_h = get_opt(5);

        let x_seq_first = if self.layout == 1 {
            x.transpose(Some(vec![1, 0, 2]), backend)?
        } else {
            x.clone()
        };

        let seq_len = x_seq_first.shape()[0] as usize;
        let batch = x_seq_first.shape()[1] as usize;
        let input_size = x_seq_first.shape()[2] as usize;
        let hs = self.hidden_size;
        let num_dirs: usize = match self.direction {
            RnnDirection::Bidirectional => 2,
            _ => 1,
        };

        let mut all_y: Vec<NumericTensor<DynRank>> = Vec::new();
        let mut all_yh: Vec<NumericTensor<DynRank>> = Vec::new();

        for dir in 0..num_dirs {
            let reverse = match self.direction {
                RnnDirection::Reverse => true,
                RnnDirection::Bidirectional => dir == 1,
                _ => false,
            };

            // W_dir [3*hs, input_size], R_dir [3*hs, hs]
            let w_dir = w.slice(&[r(dir, dir + 1), r(0, 3 * hs), r(0, input_size)], backend)?;
            let w_dir = w_dir.squeeze(0)?;
            let r_dir = rv.slice(&[r(dir, dir + 1), r(0, 3 * hs), r(0, hs)], backend)?;
            let r_dir = r_dir.squeeze(0)?;

            let w_t = w_dir.transpose(Some(vec![1, 0]), backend)?;
            let r_t = r_dir.transpose(Some(vec![1, 0]), backend)?;

            // Split R^T into [Rz, Rr, Rh] each [hs, hs]
            // r_t is [hs, 3*hs] (after transpose from [3*hs, hs])
            let rz_t = r_t.slice(&[r(0, hs), r(0, hs)], backend)?;
            let rr_t = r_t.slice(&[r(0, hs), r(hs, 2 * hs)], backend)?;
            let rh_t = r_t.slice(&[r(0, hs), r(2 * hs, 3 * hs)], backend)?;

            // Biases: ONNX GRU has [Wb_z, Wb_r, Wb_h, Rb_z, Rb_r, Rb_h] = 6*hs
            let (wb_z, wb_r, wb_h, rb_z, rb_r, rb_h) = if let Some(b) = b {
                let b_dir = b.slice(&[r(dir, dir + 1), r(0, 6 * hs)], backend)?;
                let b_dir = b_dir.squeeze(0)?;
                (
                    Some(b_dir.slice(&[r(0, hs)], backend)?),
                    Some(b_dir.slice(&[r(hs, 2 * hs)], backend)?),
                    Some(b_dir.slice(&[r(2 * hs, 3 * hs)], backend)?),
                    Some(b_dir.slice(&[r(3 * hs, 4 * hs)], backend)?),
                    Some(b_dir.slice(&[r(4 * hs, 5 * hs)], backend)?),
                    Some(b_dir.slice(&[r(5 * hs, 6 * hs)], backend)?),
                )
            } else {
                (None, None, None, None, None, None)
            };

            let mut h_t = if let Some(init_h) = init_h {
                if self.layout == 1 {
                    let h = init_h.slice(&[r(0, batch), r(dir, dir + 1), r(0, hs)], backend)?;
                    h.squeeze(1)?
                } else {
                    let h = init_h.slice(&[r(dir, dir + 1), r(0, batch), r(0, hs)], backend)?;
                    h.squeeze(0)?
                }
            } else {
                zeros_f32(&[batch as u64, hs as u64])?
            };

            let mut h_seq: Vec<NumericTensor<DynRank>> = Vec::with_capacity(seq_len);

            for t in 0..seq_len {
                let actual_t = if reverse { seq_len - 1 - t } else { t };
                let x_t = x_seq_first.slice(
                    &[r(actual_t, actual_t + 1), r(0, batch), r(0, input_size)],
                    backend,
                )?;
                let x_t = x_t.squeeze(0)?;

                // Compute all X @ W^T at once: [batch, 3*hs]
                let xw = NumericTensor::matmul(
                    &x_t,
                    &w_t,
                    None,
                    x_t.dtype(),
                    crate::milli_graph::ops::AccumulationMode::default(),
                    backend,
                )?;

                // Split into z, r, h components
                let xw_z = xw.slice(&[r(0, batch), r(0, hs)], backend)?;
                let xw_r = xw.slice(&[r(0, batch), r(hs, 2 * hs)], backend)?;
                let xw_h = xw.slice(&[r(0, batch), r(2 * hs, 3 * hs)], backend)?;

                // z_t = sigmoid(xw_z + h_{t-1} @ Rz^T + Wbz + Rbz)
                let hr_z = NumericTensor::matmul(
                    &h_t,
                    &rz_t,
                    None,
                    x_t.dtype(),
                    crate::milli_graph::ops::AccumulationMode::default(),
                    backend,
                )?;
                let mut z_gate = NumericTensor::add(&xw_z, &hr_z, backend)?;
                if let Some(ref wb) = wb_z {
                    z_gate = NumericTensor::add(&z_gate, wb, backend)?;
                }
                if let Some(ref rb) = rb_z {
                    z_gate = NumericTensor::add(&z_gate, rb, backend)?;
                }
                let z_t = sigmoid(&z_gate, backend)?;

                // r_t = sigmoid(xw_r + h_{t-1} @ Rr^T + Wbr + Rbr)
                let hr_r = NumericTensor::matmul(
                    &h_t,
                    &rr_t,
                    None,
                    x_t.dtype(),
                    crate::milli_graph::ops::AccumulationMode::default(),
                    backend,
                )?;
                let mut r_gate = NumericTensor::add(&xw_r, &hr_r, backend)?;
                if let Some(ref wb) = wb_r {
                    r_gate = NumericTensor::add(&r_gate, wb, backend)?;
                }
                if let Some(ref rb) = rb_r {
                    r_gate = NumericTensor::add(&r_gate, rb, backend)?;
                }
                let r_t = sigmoid(&r_gate, backend)?;

                // h_tilde computation depends on linear_before_reset
                let h_tilde = if self.linear_before_reset {
                    // h_tilde = tanh(xw_h + Wbh + r_t ⊙ (h_{t-1} @ Rh^T + Rbh))
                    let hr_h = NumericTensor::matmul(
                        &h_t,
                        &rh_t,
                        None,
                        x_t.dtype(),
                        crate::milli_graph::ops::AccumulationMode::default(),
                        backend,
                    )?;
                    let mut lin = hr_h;
                    if let Some(ref rb) = rb_h {
                        lin = NumericTensor::add(&lin, rb, backend)?;
                    }
                    let gated = NumericTensor::mul(&r_t, &lin, backend)?;
                    let mut h_cand = NumericTensor::add(&xw_h, &gated, backend)?;
                    if let Some(ref wb) = wb_h {
                        h_cand = NumericTensor::add(&h_cand, wb, backend)?;
                    }
                    h_cand.trig(TrigOp::Tanh, backend)?
                } else {
                    // h_tilde = tanh(xw_h + (r_t ⊙ h_{t-1}) @ Rh^T + Wbh + Rbh)
                    let r_h = NumericTensor::mul(&r_t, &h_t, backend)?;
                    let rh_matmul = NumericTensor::matmul(
                        &r_h,
                        &rh_t,
                        None,
                        x_t.dtype(),
                        crate::milli_graph::ops::AccumulationMode::default(),
                        backend,
                    )?;
                    let mut h_cand = NumericTensor::add(&xw_h, &rh_matmul, backend)?;
                    if let Some(ref wb) = wb_h {
                        h_cand = NumericTensor::add(&h_cand, wb, backend)?;
                    }
                    if let Some(ref rb) = rb_h {
                        h_cand = NumericTensor::add(&h_cand, rb, backend)?;
                    }
                    h_cand.trig(TrigOp::Tanh, backend)?
                };

                // H_t = (1 - z_t) ⊙ h_tilde + z_t ⊙ H_{t-1}
                let one = scalar_f32(1.0)?;
                let one_minus_z = NumericTensor::sub(&one, &z_t, backend)?;
                let lhs = NumericTensor::mul(&one_minus_z, &h_tilde, backend)?;
                let rhs = NumericTensor::mul(&z_t, &h_t, backend)?;
                h_t = NumericTensor::add(&lhs, &rhs, backend)?;
                h_seq.push(h_t.clone());
            }

            if reverse {
                h_seq.reverse();
            }

            // Stack -> [seq_len, 1, batch, hidden]
            let h_seq_unsqueezed: Vec<NumericTensor<DynRank>> = h_seq
                .iter()
                .map(|h| h.unsqueeze(0).and_then(|h| h.unsqueeze(1)))
                .collect::<Result<_, _>>()?;
            let h_seq_refs: Vec<&NumericTensor<DynRank>> = h_seq_unsqueezed.iter().collect();
            let y_dir = NumericTensor::concat(&h_seq_refs, 0, backend)?;
            all_y.push(y_dir);

            if self.layout == 1 {
                all_yh.push(h_t.unsqueeze(1)?);
            } else {
                all_yh.push(h_t.unsqueeze(0)?);
            }
        }

        let mut result = HashMap::new();

        if let Some(y_id) = self.outputs.first().and_then(|x| *x) {
            let y_refs: Vec<&NumericTensor<DynRank>> = all_y.iter().collect();
            let y = NumericTensor::concat(&y_refs, 1, backend)?;
            let y = if self.layout == 1 {
                y.transpose(Some(vec![2, 0, 1, 3]), backend)?
            } else {
                y
            };
            result.insert(y_id, y);
        }
        if let Some(yh_id) = self.outputs.get(1).and_then(|x| *x) {
            let yh_refs: Vec<&NumericTensor<DynRank>> = all_yh.iter().collect();
            let cat_dim = if self.layout == 1 { 1 } else { 0 };
            let yh = NumericTensor::concat(&yh_refs, cat_dim, backend)?;
            result.insert(yh_id, yh);
        }

        Ok(Box::new(result.into_iter()))
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, _rng: &mut impl Rng) -> MilliOpGraph {
        panic!("GRU uses custom eval, not milli-op decomposition")
    }
}
