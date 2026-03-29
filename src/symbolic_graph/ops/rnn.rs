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
        simple_rnn_eval_pool(self, inputs, pool)
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, _rng: &mut impl Rng) -> MilliOpGraph {
        panic!("SimpleRNN uses custom eval, not milli-op decomposition")
    }
}

/// Pool-based SimpleRNN evaluation using an inner MilliOpGraph per timestep.
///
/// Weight prep (slice, transpose) happens once via pool tensor utilities.
/// The timestep body (matmul + add + activation) is a MilliOpGraph evaluated
/// per timestep via pool_eval.
fn simple_rnn_eval_pool<'p, P: crate::pool::Pool + 'p>(
    op: &SimpleRnnOperation,
    inputs: &HashMap<GlobalId, &crate::numeric_tensor::NumericTensorView<'_, DynRank>>,
    pool: &'p P,
) -> Result<HashMap<GlobalId, crate::numeric_tensor::NumericTensor<'p, DynRank, P>>, super::EvalError>
{
    use crate::TrigOp;
    use crate::graph::GlobalId;
    use crate::milli_graph::MilliOpGraph;
    use crate::milli_graph::ops::binary::{MatMul, SimpleBinary};
    use crate::milli_graph::ops::constant::Constant;
    use crate::milli_graph::ops::unary::{ClampMin, SimpleUnaryOp};
    use crate::numeric_tensor::NumericTensor as PoolTensor;

    let map_err = |e: crate::numeric_tensor::TensorLayoutError| {
        EvalError::InvalidInput(format!("SimpleRNN tensor prep: {e}"))
    };
    let alloc_err = |e: crate::pool::AllocationError| {
        EvalError::InvalidInput(format!("SimpleRNN allocation: {e}"))
    };

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

    // Transpose X to seq-first if layout==1
    let x_seq_first: PoolTensor<'p, DynRank, P>;
    let x_sf_ref = if op.layout == 1 {
        x_seq_first = x_view
            .transpose(&[1, 0, 2])
            .map_err(map_err)?
            .to_tensor(pool)
            .map_err(alloc_err)?;
        &x_seq_first
    } else {
        // No transpose needed — but we need an owned tensor for lifetime reasons
        x_seq_first = x_view.to_tensor(pool).map_err(alloc_err)?;
        &x_seq_first
    };

    let seq_len = x_sf_ref.shape()[0] as usize;
    let batch = x_sf_ref.shape()[1] as usize;
    let input_size = x_sf_ref.shape()[2] as usize;
    let hs = op.hidden_size;
    let num_dirs: usize = match op.direction {
        RnnDirection::Bidirectional => 2,
        _ => 1,
    };
    let dtype = x_view.dtype();

    let mut all_y: Vec<PoolTensor<'p, DynRank, P>> = Vec::new();
    let mut all_yh: Vec<PoolTensor<'p, DynRank, P>> = Vec::new();

    for dir in 0..num_dirs {
        let reverse = match op.direction {
            RnnDirection::Reverse => true,
            RnnDirection::Bidirectional => dir == 1,
            _ => false,
        };

        // Slice and transpose weights for this direction
        let w_dir = w_view
            .slice(&[
                (dir as u64, dir as u64 + 1),
                (0, hs as u64),
                (0, input_size as u64),
            ])
            .map_err(map_err)?;
        // w_dir is [1, hs, input_size] — materialize as transposed [input_size, hs]
        let w_t_tensor = PoolTensor::<DynRank, P>::from_fn(
            vec![input_size as u64, hs as u64],
            dtype,
            pool,
            |flat| {
                // output[j, i] = w_dir[0, i, j]
                let j = flat / hs;
                let i = flat % hs;
                w_dir.read_element(i * input_size + j)
            },
        )
        .map_err(alloc_err)?;

        let r_dir = r_view
            .slice(&[(dir as u64, dir as u64 + 1), (0, hs as u64), (0, hs as u64)])
            .map_err(map_err)?;
        let r_t_tensor =
            PoolTensor::<DynRank, P>::from_fn(vec![hs as u64, hs as u64], dtype, pool, |flat| {
                // output[j, i] = r_dir[0, i, j]
                let j = flat / hs;
                let i = flat % hs;
                r_dir.read_element(i * hs + j)
            })
            .map_err(alloc_err)?;

        // Combine biases: Wb + Rb → [hs]
        let bias_tensor: Option<PoolTensor<'p, DynRank, P>> = if let Some(b) = b_view {
            let b_dir = b
                .slice(&[(dir as u64, dir as u64 + 1), (0, 2 * hs as u64)])
                .map_err(map_err)?;
            Some(
                PoolTensor::<DynRank, P>::from_fn(vec![hs as u64], dtype, pool, |i| {
                    let wb = b_dir.read_element(i);
                    let rb = b_dir.read_element(hs + i);
                    crate::numeric_scalar::NumericScalar::from_f32(
                        wb.to_f64() as f32 + rb.to_f64() as f32,
                    )
                })
                .map_err(alloc_err)?,
            )
        } else {
            None
        };

        // Initial hidden state
        let mut h_t: PoolTensor<'p, DynRank, P> = if let Some(init_h) = init_h_view {
            if op.layout == 1 {
                let h = init_h
                    .slice(&[
                        (0, batch as u64),
                        (dir as u64, dir as u64 + 1),
                        (0, hs as u64),
                    ])
                    .map_err(map_err)?;
                // [batch, 1, hs] → materialize as [batch, hs]
                PoolTensor::from_fn(vec![batch as u64, hs as u64], dtype, pool, |flat| {
                    let b_idx = flat / hs;
                    let h_idx = flat % hs;
                    h.read_element(b_idx * hs + h_idx)
                })
                .map_err(alloc_err)?
            } else {
                let h = init_h
                    .slice(&[
                        (dir as u64, dir as u64 + 1),
                        (0, batch as u64),
                        (0, hs as u64),
                    ])
                    .map_err(map_err)?;
                // [1, batch, hs] → materialize as [batch, hs]
                PoolTensor::from_fn(vec![batch as u64, hs as u64], dtype, pool, |flat| {
                    h.read_element(flat)
                })
                .map_err(alloc_err)?
            }
        } else {
            PoolTensor::zeros(vec![batch as u64, hs as u64], dtype, pool).map_err(alloc_err)?
        };

        // Build timestep MilliOpGraph:
        //   inputs: x_t [batch, input_size], h_prev [batch, hs], w_t [input_size, hs], r_t [hs, hs]
        //   optionally: bias [hs]
        //   body: h_t = activation(x_t @ w_t + h_prev @ r_t + bias)
        //   output: h_t [batch, hs]
        let mut rng = wyrand::WyRand::new(42);

        // External IDs for the body graph inputs/outputs
        let ext_xt = GlobalId::new(&mut rng);
        let ext_hprev = GlobalId::new(&mut rng);
        let ext_wt = GlobalId::new(&mut rng);
        let ext_rt = GlobalId::new(&mut rng);
        let ext_bias = GlobalId::new(&mut rng);
        let ext_hout = GlobalId::new(&mut rng);

        let mut body = MilliOpGraph::new_empty(&mut rng);
        let int_xt = body.add_input_with_id(ext_xt, &mut rng);
        let int_hprev = body.add_input_with_id(ext_hprev, &mut rng);
        let int_wt = body.add_input_with_id(ext_wt, &mut rng);
        let int_rt = body.add_input_with_id(ext_rt, &mut rng);
        let int_bias = if bias_tensor.is_some() {
            Some(body.add_input_with_id(ext_bias, &mut rng))
        } else {
            None
        };

        // x_t @ w_t  →  [batch, hs]
        let xw = MatMul::push_new(
            &mut body, int_xt, int_wt, dtype, dtype, dtype, dtype, &mut rng,
        );
        // h_prev @ r_t  →  [batch, hs]
        let hr = MatMul::push_new(
            &mut body, int_hprev, int_rt, dtype, dtype, dtype, dtype, &mut rng,
        );
        // xw + hr
        let mut gates = SimpleBinary::add(&mut body, xw, hr, &mut rng);
        // + bias
        if let Some(int_b) = int_bias {
            gates = SimpleBinary::add(&mut body, gates, int_b, &mut rng);
        }
        // activation
        let h_out = match op.activation.as_str() {
            "Tanh" => SimpleUnaryOp::trig(&mut body, gates, TrigOp::Tanh, &mut rng),
            "Relu" => ClampMin::push_new(&mut body, gates, 0.0, &mut rng),
            "Sigmoid" => {
                // sigmoid(x) = 1 / (1 + exp(-x))
                let neg = SimpleUnaryOp::neg(&mut body, gates, &mut rng);
                let exp_neg = SimpleUnaryOp::exp(&mut body, neg, &mut rng);
                let one = Constant::new_scalar(&mut body, 1.0f32, &mut rng);
                let denom = SimpleBinary::add(&mut body, one, exp_neg, &mut rng);
                SimpleUnaryOp::reciprocal(&mut body, denom, &mut rng)
            }
            other => {
                return Err(EvalError::UnimplementedOperatorError(format!(
                    "SimpleRNN activation: {other}"
                )));
            }
        };
        body.add_output(h_out, ext_hout);
        // Op ordering is already correct — push_op appends in dependency order.

        // Timestep loop
        let mut h_seq: Vec<PoolTensor<'p, DynRank, P>> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let actual_t = if reverse { seq_len - 1 - t } else { t };

            // Slice x_t from sequence: [actual_t..actual_t+1, 0..batch, 0..input_size]
            // then materialize as [batch, input_size]
            let x_t_view = x_sf_ref
                .slice(&[
                    (actual_t as u64, actual_t as u64 + 1),
                    (0, batch as u64),
                    (0, input_size as u64),
                ])
                .map_err(map_err)?;
            // Squeeze dim 0: materialize as [batch, input_size]
            let x_t = PoolTensor::<DynRank, P>::from_fn(
                vec![batch as u64, input_size as u64],
                dtype,
                pool,
                |flat| x_t_view.read_element(flat),
            )
            .map_err(alloc_err)?;

            // Build input map for body
            let x_t_view_ref = x_t.view();
            let h_prev_view = h_t.view();
            let w_t_view = w_t_tensor.view();
            let r_t_view = r_t_tensor.view();

            let mut body_inputs: HashMap<
                GlobalId,
                &crate::numeric_tensor::NumericTensorView<'_, DynRank>,
            > = HashMap::new();
            body_inputs.insert(ext_xt, &x_t_view_ref);
            body_inputs.insert(ext_hprev, &h_prev_view);
            body_inputs.insert(ext_wt, &w_t_view);
            body_inputs.insert(ext_rt, &r_t_view);

            let bias_view;
            if let Some(ref bias_t) = bias_tensor {
                bias_view = bias_t.view();
                body_inputs.insert(ext_bias, &bias_view);
            }

            let body_outputs = body.pool_eval(&body_inputs, pool).map_err(|e| {
                EvalError::InvalidInput(format!("SimpleRNN timestep eval failed: {e}"))
            })?;

            h_t = body_outputs
                .into_iter()
                .find(|(id, _)| *id == ext_hout)
                .map(|(_, t)| t)
                .ok_or_else(|| EvalError::InvalidInput("SimpleRNN: missing h_out".into()))?;
            h_seq.push(h_t.to_tensor(pool).map_err(alloc_err)?);
        }

        if reverse {
            h_seq.reverse();
        }

        // Stack h_seq into [seq_len, 1, batch, hs]
        let y_dir = PoolTensor::<DynRank, P>::from_fn(
            vec![seq_len as u64, 1, batch as u64, hs as u64],
            dtype,
            pool,
            |flat| {
                let t_idx = flat / (batch * hs);
                let rem = flat % (batch * hs);
                h_seq[t_idx].read_element(rem)
            },
        )
        .map_err(alloc_err)?;
        all_y.push(y_dir);

        // Final hidden state: unsqueeze to [1, batch, hs] or [batch, 1, hs]
        if op.layout == 1 {
            let yh = PoolTensor::<DynRank, P>::from_fn(
                vec![batch as u64, 1, hs as u64],
                dtype,
                pool,
                |flat| {
                    let b_idx = flat / hs;
                    let h_idx = flat % hs;
                    h_t.read_element(b_idx * hs + h_idx)
                },
            )
            .map_err(alloc_err)?;
            all_yh.push(yh);
        } else {
            let yh = PoolTensor::<DynRank, P>::from_fn(
                vec![1, batch as u64, hs as u64],
                dtype,
                pool,
                |flat| h_t.read_element(flat),
            )
            .map_err(alloc_err)?;
            all_yh.push(yh);
        }
    }

    // Assemble outputs
    let mut result = HashMap::new();

    if let Some(y_id) = op.outputs.first().and_then(|x| *x) {
        // Concat all_y along dim 1 (dir dimension) → [seq_len, num_dirs, batch, hs]
        let y = if num_dirs == 1 {
            // Already [seq_len, 1, batch, hs]
            all_y.into_iter().next().unwrap()
        } else {
            // Interleave [seq_len, 1, batch, hs] × 2 → [seq_len, 2, batch, hs]
            PoolTensor::<DynRank, P>::from_fn(
                vec![seq_len as u64, num_dirs as u64, batch as u64, hs as u64],
                dtype,
                pool,
                |flat| {
                    let s = flat / (num_dirs * batch * hs);
                    let rem = flat % (num_dirs * batch * hs);
                    let d = rem / (batch * hs);
                    let inner = rem % (batch * hs);
                    all_y[d].read_element(s * batch * hs + inner)
                },
            )
            .map_err(alloc_err)?
        };

        let y = if op.layout == 1 {
            // Transpose [seq, dir, batch, hs] → [batch, seq, dir, hs]
            y.view()
                .transpose(&[2, 0, 1, 3])
                .map_err(map_err)?
                .to_tensor(pool)
                .map_err(alloc_err)?
        } else {
            y
        };
        result.insert(y_id, y);
    }

    if let Some(yh_id) = op.outputs.get(1).and_then(|x| *x) {
        let yh = if num_dirs == 1 {
            all_yh.into_iter().next().unwrap()
        } else if op.layout == 1 {
            // [batch, 1, hs] × 2 → [batch, 2, hs]
            PoolTensor::from_fn(
                vec![batch as u64, num_dirs as u64, hs as u64],
                dtype,
                pool,
                |flat| {
                    let b_idx = flat / (num_dirs * hs);
                    let rem = flat % (num_dirs * hs);
                    let d = rem / hs;
                    let h_idx = rem % hs;
                    all_yh[d].read_element(b_idx * hs + h_idx)
                },
            )
            .map_err(alloc_err)?
        } else {
            // [1, batch, hs] × 2 → [2, batch, hs]
            PoolTensor::from_fn(
                vec![num_dirs as u64, batch as u64, hs as u64],
                dtype,
                pool,
                |flat| {
                    let d = flat / (batch * hs);
                    let inner = flat % (batch * hs);
                    all_yh[d].read_element(inner)
                },
            )
            .map_err(alloc_err)?
        };
        result.insert(yh_id, yh);
    }

    Ok(result)
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
        super::eval_pool_via_legacy(self, inputs, pool)
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, _rng: &mut impl Rng) -> MilliOpGraph {
        panic!("GRU uses custom eval, not milli-op decomposition")
    }
}
