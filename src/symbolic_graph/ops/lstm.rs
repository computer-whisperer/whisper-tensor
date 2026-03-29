use crate::TrigOp;
use crate::backends::eval_backend::EvalBackend;
use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::graph::{GlobalId, Node, Property, PropertyValue};
use crate::migration::numeric_tensor::NumericTensor;
use crate::milli_graph::{MilliLoweringContext, MilliOpGraph};
use crate::onnx;
use crate::symbolic_graph::ops::{EvalError, Operation};
use crate::symbolic_graph::{ONNXDecodingError, query_attribute_int, query_attribute_string};
use crate::tensor_rank::DynRank;
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
            .ok_or(ONNXDecodingError::MissingField("hidden_size"))?
            as usize;

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

    fn eval_pool<'p, P: crate::pool::Pool + 'p>(
        &self,
        inputs: &HashMap<
            GlobalId,
            &crate::numeric_tensor::NumericTensorView<'_, crate::tensor_rank::DynRank>,
        >,
        pool: &'p P,
    ) -> Result<
        HashMap<GlobalId, crate::numeric_tensor::NumericTensor<'p, crate::tensor_rank::DynRank, P>>,
        super::EvalError,
    > {
        lstm_eval_pool(self, inputs, pool)
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, _rng: &mut impl Rng) -> MilliOpGraph {
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

// ---------------------------------------------------------------------------
// LSTM pool-based eval
// ---------------------------------------------------------------------------

/// Pool-based LSTM evaluation using an inner MilliOpGraph per timestep.
///
/// Uses 4 separate weight matrices (Wi, Wo, Wf, Wc) instead of a single
/// combined W to avoid needing Slice ops in the body graph.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn lstm_eval_pool<'p, P: crate::pool::Pool + 'p>(
    op: &LstmOperation,
    inputs: &HashMap<GlobalId, &crate::numeric_tensor::NumericTensorView<'_, DynRank>>,
    pool: &'p P,
) -> Result<HashMap<GlobalId, crate::numeric_tensor::NumericTensor<'p, DynRank, P>>, super::EvalError>
{
    use super::rnn::{
        assemble_rnn_outputs, build_sigmoid, extract_init_state, extract_timestep,
        prepare_x_seq_first, rnn_alloc_err, rnn_map_err, stack_hidden_seq, unsqueeze_hidden,
    };
    use crate::milli_graph::MilliOpGraph;
    use crate::milli_graph::ops::binary::{MatMul, SimpleBinary};
    use crate::milli_graph::ops::unary::SimpleUnaryOp;
    use crate::numeric_tensor::NumericTensor as PoolTensor;

    let x_view = inputs[&op.inputs[0].unwrap()];
    let w_view = inputs[&op.inputs[1].unwrap()];
    let r_view = inputs[&op.inputs[2].unwrap()];

    let get_opt_view =
        |idx: usize| -> Option<&crate::numeric_tensor::NumericTensorView<'_, DynRank>> {
            op.inputs
                .get(idx)
                .and_then(|id| id.as_ref())
                .and_then(|id| inputs.get(id))
                .copied()
        };

    let b_view = get_opt_view(3);
    let init_h_view = get_opt_view(5);
    let init_c_view = get_opt_view(6);
    let p_view = get_opt_view(7);

    let x_seq_first = prepare_x_seq_first(x_view, op.layout, pool)?;

    let seq_len = x_seq_first.shape()[0] as usize;
    let batch = x_seq_first.shape()[1] as usize;
    let input_size = x_seq_first.shape()[2] as usize;
    let hs = op.hidden_size;
    let num_dirs: usize = match op.direction {
        LstmDirection::Bidirectional => 2,
        _ => 1,
    };
    let dtype = x_view.dtype();

    let mut all_y: Vec<PoolTensor<'p, DynRank, P>> = Vec::new();
    let mut all_yh: Vec<PoolTensor<'p, DynRank, P>> = Vec::new();
    let mut all_yc: Vec<PoolTensor<'p, DynRank, P>> = Vec::new();

    for dir in 0..num_dirs {
        let reverse = match op.direction {
            LstmDirection::Reverse => true,
            LstmDirection::Bidirectional => dir == 1,
            _ => false,
        };

        // Helper to slice a weight sub-block from [num_dirs, 4*hs, cols] at rows [start_row..start_row+hs]
        // and transpose to [cols, hs].
        let slice_w_sub = |src: &crate::numeric_tensor::NumericTensorView<'_, DynRank>,
                           start_row: usize,
                           cols: usize|
         -> Result<PoolTensor<'p, DynRank, P>, EvalError> {
            let sliced = src
                .slice(&[
                    (dir as u64, dir as u64 + 1),
                    (start_row as u64, (start_row + hs) as u64),
                    (0, cols as u64),
                ])
                .map_err(rnn_map_err)?;
            PoolTensor::<DynRank, P>::from_fn(vec![cols as u64, hs as u64], dtype, pool, |flat| {
                let c = flat / hs;
                let r = flat % hs;
                sliced.read_element(r * cols + c)
            })
            .map_err(rnn_alloc_err)
        };

        // ONNX gate order: i, o, f, c — each [hs, input_size] or [hs, hs]
        // W is [num_dirs, 4*hs, input_size], R is [num_dirs, 4*hs, hs]
        let wi_t = slice_w_sub(w_view, 0, input_size)?;
        let wo_t = slice_w_sub(w_view, hs, input_size)?;
        let wf_t = slice_w_sub(w_view, 2 * hs, input_size)?;
        let wc_t = slice_w_sub(w_view, 3 * hs, input_size)?;

        let ri_t = slice_w_sub(r_view, 0, hs)?;
        let ro_t = slice_w_sub(r_view, hs, hs)?;
        let rf_t = slice_w_sub(r_view, 2 * hs, hs)?;
        let rc_t = slice_w_sub(r_view, 3 * hs, hs)?;

        // Biases: ONNX LSTM has [Wb_i, Wb_o, Wb_f, Wb_c, Rb_i, Rb_o, Rb_f, Rb_c] = 8*hs
        // We pre-combine Wb + Rb for each gate → 4 biases of [hs]
        let gate_biases: Option<(
            PoolTensor<'p, DynRank, P>,
            PoolTensor<'p, DynRank, P>,
            PoolTensor<'p, DynRank, P>,
            PoolTensor<'p, DynRank, P>,
        )> = if let Some(b) = b_view {
            let b_dir = b
                .slice(&[(dir as u64, dir as u64 + 1), (0, 8 * hs as u64)])
                .map_err(rnn_map_err)?;
            let make_combined_bias = |wb_start: usize,
                                      rb_start: usize|
             -> Result<PoolTensor<'p, DynRank, P>, EvalError> {
                PoolTensor::<DynRank, P>::from_fn(vec![hs as u64], dtype, pool, |i| {
                    let wb = b_dir.read_element(wb_start + i);
                    let rb = b_dir.read_element(rb_start + i);
                    crate::numeric_scalar::NumericScalar::from_f32(
                        wb.to_f64() as f32 + rb.to_f64() as f32,
                    )
                })
                .map_err(rnn_alloc_err)
            };
            Some((
                make_combined_bias(0, 4 * hs)?,      // bi = Wb_i + Rb_i
                make_combined_bias(hs, 5 * hs)?,     // bo = Wb_o + Rb_o
                make_combined_bias(2 * hs, 6 * hs)?, // bf = Wb_f + Rb_f
                make_combined_bias(3 * hs, 7 * hs)?, // bc = Wb_c + Rb_c
            ))
        } else {
            None
        };

        // Peepholes: P is [num_dirs, 3*hs] with order [pi, po, pf]
        let peepholes: Option<(
            PoolTensor<'p, DynRank, P>,
            PoolTensor<'p, DynRank, P>,
            PoolTensor<'p, DynRank, P>,
        )> = if let Some(p) = p_view {
            let p_dir = p
                .slice(&[(dir as u64, dir as u64 + 1), (0, 3 * hs as u64)])
                .map_err(rnn_map_err)?;
            let make_peep = |start: usize| -> Result<PoolTensor<'p, DynRank, P>, EvalError> {
                PoolTensor::<DynRank, P>::from_fn(vec![hs as u64], dtype, pool, |i| {
                    p_dir.read_element(start + i)
                })
                .map_err(rnn_alloc_err)
            };
            Some((make_peep(0)?, make_peep(hs)?, make_peep(2 * hs)?))
        } else {
            None
        };

        // Initial states
        let mut h_t = extract_init_state(init_h_view, dir, batch, hs, op.layout, dtype, pool)?;
        let mut c_t = extract_init_state(init_c_view, dir, batch, hs, op.layout, dtype, pool)?;

        // Build timestep MilliOpGraph
        let mut rng = wyrand::WyRand::new(42);

        // External IDs for inputs
        let ext_xt = GlobalId::new(&mut rng);
        let ext_hprev = GlobalId::new(&mut rng);
        let ext_cprev = GlobalId::new(&mut rng);
        // 4 W matrices (transposed)
        let ext_wi_t = GlobalId::new(&mut rng);
        let ext_wo_t = GlobalId::new(&mut rng);
        let ext_wf_t = GlobalId::new(&mut rng);
        let ext_wc_t = GlobalId::new(&mut rng);
        // 4 R matrices (transposed)
        let ext_ri_t = GlobalId::new(&mut rng);
        let ext_ro_t = GlobalId::new(&mut rng);
        let ext_rf_t = GlobalId::new(&mut rng);
        let ext_rc_t = GlobalId::new(&mut rng);
        // 4 combined biases
        let ext_bi = GlobalId::new(&mut rng);
        let ext_bo = GlobalId::new(&mut rng);
        let ext_bf = GlobalId::new(&mut rng);
        let ext_bc = GlobalId::new(&mut rng);
        // 3 peepholes
        let ext_pi = GlobalId::new(&mut rng);
        let ext_po = GlobalId::new(&mut rng);
        let ext_pf = GlobalId::new(&mut rng);
        // Outputs
        let ext_hout = GlobalId::new(&mut rng);
        let ext_cout = GlobalId::new(&mut rng);

        let mut body = MilliOpGraph::new_empty(&mut rng);
        let int_xt = body.add_input_with_id(ext_xt, &mut rng);
        let int_hprev = body.add_input_with_id(ext_hprev, &mut rng);
        let int_cprev = body.add_input_with_id(ext_cprev, &mut rng);
        let int_wi_t = body.add_input_with_id(ext_wi_t, &mut rng);
        let int_wo_t = body.add_input_with_id(ext_wo_t, &mut rng);
        let int_wf_t = body.add_input_with_id(ext_wf_t, &mut rng);
        let int_wc_t = body.add_input_with_id(ext_wc_t, &mut rng);
        let int_ri_t = body.add_input_with_id(ext_ri_t, &mut rng);
        let int_ro_t = body.add_input_with_id(ext_ro_t, &mut rng);
        let int_rf_t = body.add_input_with_id(ext_rf_t, &mut rng);
        let int_rc_t = body.add_input_with_id(ext_rc_t, &mut rng);

        let (int_bi, int_bo, int_bf, int_bc) = if gate_biases.is_some() {
            (
                Some(body.add_input_with_id(ext_bi, &mut rng)),
                Some(body.add_input_with_id(ext_bo, &mut rng)),
                Some(body.add_input_with_id(ext_bf, &mut rng)),
                Some(body.add_input_with_id(ext_bc, &mut rng)),
            )
        } else {
            (None, None, None, None)
        };

        let (int_pi, int_po, int_pf) = if peepholes.is_some() {
            (
                Some(body.add_input_with_id(ext_pi, &mut rng)),
                Some(body.add_input_with_id(ext_po, &mut rng)),
                Some(body.add_input_with_id(ext_pf, &mut rng)),
            )
        } else {
            (None, None, None)
        };

        // Gate computations: gi = x_t @ wi_t + h_prev @ ri_t + bi (each [batch, hs])
        let xw_i = MatMul::push_new(
            &mut body, int_xt, int_wi_t, dtype, dtype, dtype, dtype, &mut rng,
        );
        let hr_i = MatMul::push_new(
            &mut body, int_hprev, int_ri_t, dtype, dtype, dtype, dtype, &mut rng,
        );
        let mut gi = SimpleBinary::add(&mut body, xw_i, hr_i, &mut rng);
        if let Some(b) = int_bi {
            gi = SimpleBinary::add(&mut body, gi, b, &mut rng);
        }

        let xw_o = MatMul::push_new(
            &mut body, int_xt, int_wo_t, dtype, dtype, dtype, dtype, &mut rng,
        );
        let hr_o = MatMul::push_new(
            &mut body, int_hprev, int_ro_t, dtype, dtype, dtype, dtype, &mut rng,
        );
        let mut go = SimpleBinary::add(&mut body, xw_o, hr_o, &mut rng);
        if let Some(b) = int_bo {
            go = SimpleBinary::add(&mut body, go, b, &mut rng);
        }

        let xw_f = MatMul::push_new(
            &mut body, int_xt, int_wf_t, dtype, dtype, dtype, dtype, &mut rng,
        );
        let hr_f = MatMul::push_new(
            &mut body, int_hprev, int_rf_t, dtype, dtype, dtype, dtype, &mut rng,
        );
        let mut gf = SimpleBinary::add(&mut body, xw_f, hr_f, &mut rng);
        if let Some(b) = int_bf {
            gf = SimpleBinary::add(&mut body, gf, b, &mut rng);
        }

        let xw_c = MatMul::push_new(
            &mut body, int_xt, int_wc_t, dtype, dtype, dtype, dtype, &mut rng,
        );
        let hr_c = MatMul::push_new(
            &mut body, int_hprev, int_rc_t, dtype, dtype, dtype, dtype, &mut rng,
        );
        let mut gc = SimpleBinary::add(&mut body, xw_c, hr_c, &mut rng);
        if let Some(b) = int_bc {
            gc = SimpleBinary::add(&mut body, gc, b, &mut rng);
        }

        // Peephole on i and f (uses c_prev)
        if let Some(pi) = int_pi {
            let pi_c = SimpleBinary::mul(&mut body, pi, int_cprev, &mut rng);
            gi = SimpleBinary::add(&mut body, gi, pi_c, &mut rng);
        }
        if let Some(pf) = int_pf {
            let pf_c = SimpleBinary::mul(&mut body, pf, int_cprev, &mut rng);
            gf = SimpleBinary::add(&mut body, gf, pf_c, &mut rng);
        }

        // Activations
        let i_t = build_sigmoid(&mut body, gi, &mut rng);
        let f_t = build_sigmoid(&mut body, gf, &mut rng);
        let c_cand = SimpleUnaryOp::trig(&mut body, gc, TrigOp::Tanh, &mut rng);

        // C_t = f_t * c_prev + i_t * c_cand
        let fc = SimpleBinary::mul(&mut body, f_t, int_cprev, &mut rng);
        let ic = SimpleBinary::mul(&mut body, i_t, c_cand, &mut rng);
        let c_new = SimpleBinary::add(&mut body, fc, ic, &mut rng);

        // Peephole on o (uses NEW c_t)
        if let Some(po) = int_po {
            let po_c = SimpleBinary::mul(&mut body, po, c_new, &mut rng);
            go = SimpleBinary::add(&mut body, go, po_c, &mut rng);
        }
        let o_t = build_sigmoid(&mut body, go, &mut rng);

        // H_t = o_t * tanh(C_t)
        let c_tanh = SimpleUnaryOp::trig(&mut body, c_new, TrigOp::Tanh, &mut rng);
        let h_out = SimpleBinary::mul(&mut body, o_t, c_tanh, &mut rng);

        body.add_output(h_out, ext_hout);
        body.add_output(c_new, ext_cout);

        // Timestep loop
        let mut h_seq: Vec<PoolTensor<'p, DynRank, P>> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let actual_t = if reverse { seq_len - 1 - t } else { t };

            let x_t = extract_timestep(&x_seq_first, actual_t, batch, input_size, dtype, pool)?;

            let x_t_view_ref = x_t.view();
            let h_prev_view = h_t.view();
            let c_prev_view = c_t.view();
            let wi_t_view = wi_t.view();
            let wo_t_view = wo_t.view();
            let wf_t_view = wf_t.view();
            let wc_t_view = wc_t.view();
            let ri_t_view = ri_t.view();
            let ro_t_view = ro_t.view();
            let rf_t_view = rf_t.view();
            let rc_t_view = rc_t.view();

            let mut body_inputs: HashMap<
                GlobalId,
                &crate::numeric_tensor::NumericTensorView<'_, DynRank>,
            > = HashMap::new();
            body_inputs.insert(ext_xt, &x_t_view_ref);
            body_inputs.insert(ext_hprev, &h_prev_view);
            body_inputs.insert(ext_cprev, &c_prev_view);
            body_inputs.insert(ext_wi_t, &wi_t_view);
            body_inputs.insert(ext_wo_t, &wo_t_view);
            body_inputs.insert(ext_wf_t, &wf_t_view);
            body_inputs.insert(ext_wc_t, &wc_t_view);
            body_inputs.insert(ext_ri_t, &ri_t_view);
            body_inputs.insert(ext_ro_t, &ro_t_view);
            body_inputs.insert(ext_rf_t, &rf_t_view);
            body_inputs.insert(ext_rc_t, &rc_t_view);

            let (bi_view, bo_view, bf_view, bc_view);
            if let Some(ref b) = gate_biases {
                bi_view = b.0.view();
                bo_view = b.1.view();
                bf_view = b.2.view();
                bc_view = b.3.view();
                body_inputs.insert(ext_bi, &bi_view);
                body_inputs.insert(ext_bo, &bo_view);
                body_inputs.insert(ext_bf, &bf_view);
                body_inputs.insert(ext_bc, &bc_view);
            }

            let (pi_view, po_view, pf_view);
            if let Some(ref p) = peepholes {
                pi_view = p.0.view();
                po_view = p.1.view();
                pf_view = p.2.view();
                body_inputs.insert(ext_pi, &pi_view);
                body_inputs.insert(ext_po, &po_view);
                body_inputs.insert(ext_pf, &pf_view);
            }

            let body_outputs = body
                .pool_eval(&body_inputs, pool)
                .map_err(|e| EvalError::InvalidInput(format!("LSTM timestep eval failed: {e}")))?;

            // Extract h_t and c_t from outputs
            let mut found_h = None;
            let mut found_c = None;
            for (id, tensor) in body_outputs {
                if id == ext_hout {
                    found_h = Some(tensor);
                } else if id == ext_cout {
                    found_c = Some(tensor);
                }
            }
            h_t = found_h.ok_or_else(|| EvalError::InvalidInput("LSTM: missing h_out".into()))?;
            c_t = found_c.ok_or_else(|| EvalError::InvalidInput("LSTM: missing c_out".into()))?;

            h_seq.push(h_t.to_tensor(pool).map_err(rnn_alloc_err)?);
        }

        if reverse {
            h_seq.reverse();
        }

        let y_dir = stack_hidden_seq(&h_seq, batch, hs, dtype, pool)?;
        all_y.push(y_dir);

        let yh = unsqueeze_hidden(&h_t, batch, hs, op.layout, dtype, pool)?;
        all_yh.push(yh);

        let yc = unsqueeze_hidden(&c_t, batch, hs, op.layout, dtype, pool)?;
        all_yc.push(yc);
    }

    assemble_rnn_outputs(
        &op.outputs,
        all_y,
        all_yh,
        Some(all_yc),
        seq_len,
        num_dirs,
        batch,
        hs,
        op.layout,
        dtype,
        pool,
    )
}
