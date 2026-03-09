use crate::TrigOp;
use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::milli_graph::MilliOpGraph;
use crate::numeric_tensor::NumericTensor;
use crate::symbolic_graph::ops::{EvalError, Operation};
use crate::symbolic_graph::{ONNXDecodingError, query_attribute_int, query_attribute_string};
use crate::tensor_rank::DynRank;
use crate::onnx;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum LstmDirection {
    Forward,
    Reverse,
    Bidirectional,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LstmOperation {
    global_id: GlobalId,
    inputs: Vec<Option<GlobalId>>,
    outputs: Vec<Option<GlobalId>>,
    hidden_size: usize,
    direction: LstmDirection,
    layout: usize,
}

impl LstmOperation {
    pub(crate) fn from_onnx(
        inputs: &[Option<GlobalId>],
        outputs: &[Option<GlobalId>],
        attributes: &[onnx::AttributeProto],
        rng: &mut impl Rng,
    ) -> Result<Self, ONNXDecodingError> {
        let hidden_size = query_attribute_int(attributes, "hidden_size")
            .ok_or(ONNXDecodingError::MissingField("hidden_size"))? as usize;

        let direction = match query_attribute_string(attributes, "direction").as_deref() {
            Some("reverse") => LstmDirection::Reverse,
            Some("bidirectional") => LstmDirection::Bidirectional,
            _ => LstmDirection::Forward,
        };

        let layout = query_attribute_int(attributes, "layout").unwrap_or(0) as usize;

        Ok(Self {
            global_id: GlobalId::new(rng),
            inputs: inputs.to_vec(),
            outputs: outputs.to_vec(),
            hidden_size,
            direction,
            layout,
        })
    }
}

impl Node for LstmOperation {
    type OpKind = String;
    fn global_id(&self) -> GlobalId {
        self.global_id
    }
    fn op_kind(&self) -> Self::OpKind {
        "LSTM".to_string()
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

/// Helper to make a u64 range for slice.
fn r(start: usize, end: usize) -> std::ops::Range<u64> {
    start as u64..end as u64
}

impl Operation for LstmOperation {
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
        let init_c = get_opt(6);
        let p = get_opt(7);

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
            LstmDirection::Bidirectional => 2,
            _ => 1,
        };

        let mut all_y: Vec<NumericTensor<DynRank>> = Vec::new();
        let mut all_yh: Vec<NumericTensor<DynRank>> = Vec::new();
        let mut all_yc: Vec<NumericTensor<DynRank>> = Vec::new();

        for dir in 0..num_dirs {
            let reverse = match self.direction {
                LstmDirection::Reverse => true,
                LstmDirection::Bidirectional => dir == 1,
                _ => false,
            };

            // W_dir [4*hs, input_size], R_dir [4*hs, hs]
            let w_dir = w.slice(&[r(dir, dir + 1), r(0, 4 * hs), r(0, input_size)], backend)?;
            let w_dir = w_dir.squeeze(0)?;
            let r_dir = rv.slice(&[r(dir, dir + 1), r(0, 4 * hs), r(0, hs)], backend)?;
            let r_dir = r_dir.squeeze(0)?;

            // Transpose for X @ W^T shape
            let w_t = w_dir.transpose(Some(vec![1, 0]), backend)?;
            let r_t = r_dir.transpose(Some(vec![1, 0]), backend)?;

            let bias = if let Some(b) = b {
                let b_dir = b.slice(&[r(dir, dir + 1), r(0, 8 * hs)], backend)?;
                let b_dir = b_dir.squeeze(0)?;
                let wb = b_dir.slice(&[r(0, 4 * hs)], backend)?;
                let rb = b_dir.slice(&[r(4 * hs, 8 * hs)], backend)?;
                Some(NumericTensor::add(&wb, &rb, backend)?)
            } else {
                None
            };

            let peepholes = if let Some(p) = p {
                let p_dir = p.slice(&[r(dir, dir + 1), r(0, 3 * hs)], backend)?;
                let p_dir = p_dir.squeeze(0)?;
                let pi = p_dir.slice(&[r(0, hs)], backend)?;
                let po = p_dir.slice(&[r(hs, 2 * hs)], backend)?;
                let pf = p_dir.slice(&[r(2 * hs, 3 * hs)], backend)?;
                Some((pi, po, pf))
            } else {
                None
            };

            // Initial states: layout=0 -> [num_dirs, batch, hs], layout=1 -> [batch, num_dirs, hs]
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
            let mut c_t = if let Some(init_c) = init_c {
                if self.layout == 1 {
                    let c = init_c.slice(&[r(0, batch), r(dir, dir + 1), r(0, hs)], backend)?;
                    c.squeeze(1)?
                } else {
                    let c = init_c.slice(&[r(dir, dir + 1), r(0, batch), r(0, hs)], backend)?;
                    c.squeeze(0)?
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

                // gates = x_t @ W^T + h_{t-1} @ R^T [+ bias]
                let xw = NumericTensor::matmul(&x_t, &w_t, None, backend)?;
                let hr = NumericTensor::matmul(&h_t, &r_t, None, backend)?;
                let mut gates = NumericTensor::add(&xw, &hr, backend)?;
                if let Some(ref bias) = bias {
                    gates = NumericTensor::add(&gates, bias, backend)?;
                }

                // ONNX gate order: [i, o, f, c]
                let gi = gates.slice(&[r(0, batch), r(0, hs)], backend)?;
                let go = gates.slice(&[r(0, batch), r(hs, 2 * hs)], backend)?;
                let gf = gates.slice(&[r(0, batch), r(2 * hs, 3 * hs)], backend)?;
                let gc = gates.slice(&[r(0, batch), r(3 * hs, 4 * hs)], backend)?;

                // Peepholes on i, f (before activation)
                let gi = if let Some((ref pi, _, _)) = peepholes {
                    NumericTensor::add(&gi, &NumericTensor::mul(pi, &c_t, backend)?, backend)?
                } else {
                    gi
                };
                let gf = if let Some((_, _, ref pf)) = peepholes {
                    NumericTensor::add(&gf, &NumericTensor::mul(pf, &c_t, backend)?, backend)?
                } else {
                    gf
                };

                let i_t = sigmoid(&gi, backend)?;
                let f_t = sigmoid(&gf, backend)?;
                let c_cand = gc.trig(TrigOp::Tanh, backend)?;

                // C_t = f_t * C_{t-1} + i_t * c_cand
                let fc = NumericTensor::mul(&f_t, &c_t, backend)?;
                let ic = NumericTensor::mul(&i_t, &c_cand, backend)?;
                c_t = NumericTensor::add(&fc, &ic, backend)?;

                // Peephole on o (after cell update)
                let go = if let Some((_, ref po, _)) = peepholes {
                    NumericTensor::add(&go, &NumericTensor::mul(po, &c_t, backend)?, backend)?
                } else {
                    go
                };
                let o_t = sigmoid(&go, backend)?;

                // H_t = o_t * tanh(C_t)
                h_t = NumericTensor::mul(&o_t, &c_t.trig(TrigOp::Tanh, backend)?, backend)?;
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
                // [batch, hs] -> [batch, 1, hs] for later concat on dim=1
                all_yh.push(h_t.unsqueeze(1)?);
                all_yc.push(c_t.unsqueeze(1)?);
            } else {
                // [batch, hs] -> [1, batch, hs] for later concat on dim=0
                all_yh.push(h_t.unsqueeze(0)?);
                all_yc.push(c_t.unsqueeze(0)?);
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
        if let Some(yc_id) = self.outputs.get(2).and_then(|x| *x) {
            let yc_refs: Vec<&NumericTensor<DynRank>> = all_yc.iter().collect();
            let cat_dim = if self.layout == 1 { 1 } else { 0 };
            let yc = NumericTensor::concat(&yc_refs, cat_dim, backend)?;
            result.insert(yc_id, yc);
        }

        Ok(Box::new(result.into_iter()))
    }

    fn get_milli_op_graph(&self, _rng: &mut impl Rng) -> MilliOpGraph {
        panic!("LSTM uses custom eval, not milli-op decomposition")
    }
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

/// sigmoid(x) = 1 / (1 + exp(-x))
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
