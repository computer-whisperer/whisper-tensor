use crate::TrigOp;
use crate::graph::{GlobalId, Node, Property, PropertyValue};
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

// ---------------------------------------------------------------------------
// Shared helpers for pool-based RNN evaluation
// ---------------------------------------------------------------------------

pub(super) type RnnPoolTensor<'p, P> = crate::numeric_tensor::NumericTensor<'p, DynRank, P>;
pub(super) type RnnPoolView<'a> = crate::numeric_tensor::NumericTensorView<'a, DynRank>;

pub(super) fn rnn_map_err(e: crate::numeric_tensor::TensorLayoutError) -> EvalError {
    EvalError::InvalidInput(format!("RNN tensor prep: {e}"))
}

pub(super) fn rnn_alloc_err(e: crate::pool::AllocationError) -> EvalError {
    EvalError::InvalidInput(format!("RNN allocation: {e}"))
}

/// Slice a weight matrix from [num_dirs, rows, cols] for one direction,
/// then materialize as transposed [cols, rows].
pub(super) fn slice_and_transpose_weight<'p, P: crate::pool::Pool + 'p>(
    view: &RnnPoolView<'_>,
    dir: usize,
    rows: usize,
    cols: usize,
    dtype: crate::numeric_dtype::NumericDType,
    pool: &'p P,
) -> Result<RnnPoolTensor<'p, P>, EvalError> {
    let sliced = view
        .slice(&[
            (dir as u64, dir as u64 + 1),
            (0, rows as u64),
            (0, cols as u64),
        ])
        .map_err(rnn_map_err)?;
    // sliced is [1, rows, cols] — materialize as [cols, rows] (transposed, dim 0 squeezed)
    crate::numeric_tensor::NumericTensor::<DynRank, P>::from_fn(
        vec![cols as u64, rows as u64],
        dtype,
        pool,
        |flat| {
            let c = flat / rows;
            let r = flat % rows;
            sliced.read_element(r * cols + c)
        },
    )
    .map_err(rnn_alloc_err)
}

/// Extract initial hidden state for one direction, squeezing the dir dimension.
pub(super) fn extract_init_state<'p, P: crate::pool::Pool + 'p>(
    init: Option<&RnnPoolView<'_>>,
    dir: usize,
    batch: usize,
    hs: usize,
    layout: usize,
    dtype: crate::numeric_dtype::NumericDType,
    pool: &'p P,
) -> Result<RnnPoolTensor<'p, P>, EvalError> {
    if let Some(init_view) = init {
        let sliced = if layout == 1 {
            // [batch, num_dirs, hs] → slice dir → [batch, 1, hs]
            init_view
                .slice(&[
                    (0, batch as u64),
                    (dir as u64, dir as u64 + 1),
                    (0, hs as u64),
                ])
                .map_err(rnn_map_err)?
        } else {
            // [num_dirs, batch, hs] → slice dir → [1, batch, hs]
            init_view
                .slice(&[
                    (dir as u64, dir as u64 + 1),
                    (0, batch as u64),
                    (0, hs as u64),
                ])
                .map_err(rnn_map_err)?
        };
        // Materialize as [batch, hs] (squeeze the size-1 dir dimension)
        crate::numeric_tensor::NumericTensor::from_fn(
            vec![batch as u64, hs as u64],
            dtype,
            pool,
            |flat| sliced.read_element(flat),
        )
        .map_err(rnn_alloc_err)
    } else {
        crate::numeric_tensor::NumericTensor::zeros(vec![batch as u64, hs as u64], dtype, pool)
            .map_err(rnn_alloc_err)
    }
}

/// Slice x_t from a seq-first tensor [seq_len, batch, input_size] at timestep t,
/// returning [batch, input_size].
pub(super) fn extract_timestep<'p, P: crate::pool::Pool + 'p>(
    x_seq_first: &RnnPoolTensor<'p, P>,
    t: usize,
    batch: usize,
    feat_size: usize,
    dtype: crate::numeric_dtype::NumericDType,
    pool: &'p P,
) -> Result<RnnPoolTensor<'p, P>, EvalError> {
    let sliced = x_seq_first
        .slice(&[
            (t as u64, t as u64 + 1),
            (0, batch as u64),
            (0, feat_size as u64),
        ])
        .map_err(rnn_map_err)?;
    crate::numeric_tensor::NumericTensor::from_fn(
        vec![batch as u64, feat_size as u64],
        dtype,
        pool,
        |flat| sliced.read_element(flat),
    )
    .map_err(rnn_alloc_err)
}

/// Build a sigmoid sub-graph: sigmoid(x) = 1 / (1 + exp(-x)).
pub(super) fn build_sigmoid(
    body: &mut crate::milli_graph::MilliOpGraph,
    input: GlobalId,
    rng: &mut impl rand::Rng,
) -> GlobalId {
    use crate::milli_graph::ops::binary::SimpleBinary;
    use crate::milli_graph::ops::constant::Constant;
    use crate::milli_graph::ops::unary::SimpleUnaryOp;
    let neg = SimpleUnaryOp::neg(body, input, rng);
    let exp_neg = SimpleUnaryOp::exp(body, neg, rng);
    let one = Constant::new_scalar(body, 1.0f32, rng);
    let denom = SimpleBinary::add(body, one, exp_neg, rng);
    SimpleUnaryOp::reciprocal(body, denom, rng)
}

/// Stack h_seq [batch, hs] × seq_len into [seq_len, 1, batch, hs].
pub(super) fn stack_hidden_seq<'p, P: crate::pool::Pool + 'p>(
    h_seq: &[RnnPoolTensor<'p, P>],
    batch: usize,
    hs: usize,
    dtype: crate::numeric_dtype::NumericDType,
    pool: &'p P,
) -> Result<RnnPoolTensor<'p, P>, EvalError> {
    let seq_len = h_seq.len();
    crate::numeric_tensor::NumericTensor::from_fn(
        vec![seq_len as u64, 1, batch as u64, hs as u64],
        dtype,
        pool,
        |flat| {
            let t_idx = flat / (batch * hs);
            let rem = flat % (batch * hs);
            h_seq[t_idx].read_element(rem)
        },
    )
    .map_err(rnn_alloc_err)
}

/// Unsqueeze [batch, hs] → [1, batch, hs] or [batch, 1, hs] depending on layout.
pub(super) fn unsqueeze_hidden<'p, P: crate::pool::Pool + 'p>(
    h: &RnnPoolTensor<'p, P>,
    batch: usize,
    hs: usize,
    layout: usize,
    dtype: crate::numeric_dtype::NumericDType,
    pool: &'p P,
) -> Result<RnnPoolTensor<'p, P>, EvalError> {
    if layout == 1 {
        crate::numeric_tensor::NumericTensor::from_fn(
            vec![batch as u64, 1, hs as u64],
            dtype,
            pool,
            |flat| {
                let b_idx = flat / hs;
                let h_idx = flat % hs;
                h.read_element(b_idx * hs + h_idx)
            },
        )
        .map_err(rnn_alloc_err)
    } else {
        crate::numeric_tensor::NumericTensor::from_fn(
            vec![1, batch as u64, hs as u64],
            dtype,
            pool,
            |flat| h.read_element(flat),
        )
        .map_err(rnn_alloc_err)
    }
}

/// Assemble directional outputs into final Y, Y_h (and optionally Y_c).
/// all_y: [seq_len, 1, batch, hs] per direction.
/// all_yh: unsqueezed [1, batch, hs] or [batch, 1, hs] per direction.
#[allow(clippy::too_many_arguments)]
pub(super) fn assemble_rnn_outputs<'p, P: crate::pool::Pool + 'p>(
    outputs: &[Option<GlobalId>],
    all_y: Vec<RnnPoolTensor<'p, P>>,
    all_yh: Vec<RnnPoolTensor<'p, P>>,
    all_yc: Option<Vec<RnnPoolTensor<'p, P>>>,
    seq_len: usize,
    num_dirs: usize,
    batch: usize,
    hs: usize,
    layout: usize,
    dtype: crate::numeric_dtype::NumericDType,
    pool: &'p P,
) -> Result<HashMap<GlobalId, RnnPoolTensor<'p, P>>, EvalError> {
    let mut result = HashMap::new();

    // Y output
    if let Some(y_id) = outputs.first().and_then(|x| *x) {
        let y = if num_dirs == 1 {
            all_y.into_iter().next().unwrap()
        } else {
            crate::numeric_tensor::NumericTensor::from_fn(
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
            .map_err(rnn_alloc_err)?
        };
        let y = if layout == 1 {
            y.view()
                .transpose(&[2, 0, 1, 3])
                .map_err(rnn_map_err)?
                .to_tensor(pool)
                .map_err(rnn_alloc_err)?
        } else {
            y
        };
        result.insert(y_id, y);
    }

    // Y_h output
    if let Some(yh_id) = outputs.get(1).and_then(|x| *x) {
        let yh = concat_dir_outputs(all_yh, num_dirs, batch, hs, layout, dtype, pool)?;
        result.insert(yh_id, yh);
    }

    // Y_c output (LSTM only)
    if let (Some(yc_id), Some(all_yc)) = (outputs.get(2).and_then(|x| *x), all_yc) {
        let yc = concat_dir_outputs(all_yc, num_dirs, batch, hs, layout, dtype, pool)?;
        result.insert(yc_id, yc);
    }

    Ok(result)
}

pub(super) fn concat_dir_outputs<'p, P: crate::pool::Pool + 'p>(
    all: Vec<RnnPoolTensor<'p, P>>,
    num_dirs: usize,
    batch: usize,
    hs: usize,
    layout: usize,
    dtype: crate::numeric_dtype::NumericDType,
    pool: &'p P,
) -> Result<RnnPoolTensor<'p, P>, EvalError> {
    if num_dirs == 1 {
        Ok(all.into_iter().next().unwrap())
    } else if layout == 1 {
        crate::numeric_tensor::NumericTensor::from_fn(
            vec![batch as u64, num_dirs as u64, hs as u64],
            dtype,
            pool,
            |flat| {
                let b_idx = flat / (num_dirs * hs);
                let rem = flat % (num_dirs * hs);
                let d = rem / hs;
                let h_idx = rem % hs;
                all[d].read_element(b_idx * hs + h_idx)
            },
        )
        .map_err(rnn_alloc_err)
    } else {
        crate::numeric_tensor::NumericTensor::from_fn(
            vec![num_dirs as u64, batch as u64, hs as u64],
            dtype,
            pool,
            |flat| {
                let d = flat / (batch * hs);
                let inner = flat % (batch * hs);
                all[d].read_element(inner)
            },
        )
        .map_err(rnn_alloc_err)
    }
}

/// Transpose X to seq-first if layout==1, otherwise copy.
pub(super) fn prepare_x_seq_first<'p, P: crate::pool::Pool + 'p>(
    x_view: &RnnPoolView<'_>,
    layout: usize,
    pool: &'p P,
) -> Result<RnnPoolTensor<'p, P>, EvalError> {
    if layout == 1 {
        x_view
            .transpose(&[1, 0, 2])
            .map_err(rnn_map_err)?
            .to_tensor(pool)
            .map_err(rnn_alloc_err)
    } else {
        x_view.to_tensor(pool).map_err(rnn_alloc_err)
    }
}

// ---------------------------------------------------------------------------
// SimpleRNN
// ---------------------------------------------------------------------------

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
        let r_t_tensor = crate::numeric_tensor::NumericTensor::<DynRank, P>::from_fn(
            vec![hs as u64, hs as u64],
            dtype,
            pool,
            |flat| {
                // output[j, i] = r_dir[0, i, j]
                let j = flat / hs;
                let i = flat % hs;
                r_dir.read_element(i * hs + j)
            },
        )
        .map_err(alloc_err)?;

        // Combine biases: Wb + Rb → [hs]
        let bias_tensor: Option<PoolTensor<'p, DynRank, P>> = if let Some(b) = b_view {
            let b_dir = b
                .slice(&[(dir as u64, dir as u64 + 1), (0, 2 * hs as u64)])
                .map_err(map_err)?;
            Some(
                crate::numeric_tensor::NumericTensor::<DynRank, P>::from_fn(
                    vec![hs as u64],
                    dtype,
                    pool,
                    |i| {
                        let wb = b_dir.read_element(i);
                        let rb = b_dir.read_element(hs + i);
                        crate::numeric_scalar::NumericScalar::from_f32(
                            wb.to_f64() as f32 + rb.to_f64() as f32,
                        )
                    },
                )
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
                crate::numeric_tensor::NumericTensor::from_fn(
                    vec![batch as u64, hs as u64],
                    dtype,
                    pool,
                    |flat| {
                        let b_idx = flat / hs;
                        let h_idx = flat % hs;
                        h.read_element(b_idx * hs + h_idx)
                    },
                )
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
                crate::numeric_tensor::NumericTensor::from_fn(
                    vec![batch as u64, hs as u64],
                    dtype,
                    pool,
                    |flat| h.read_element(flat),
                )
                .map_err(alloc_err)?
            }
        } else {
            crate::numeric_tensor::NumericTensor::zeros(vec![batch as u64, hs as u64], dtype, pool)
                .map_err(alloc_err)?
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
            crate::numeric_tensor::NumericTensor::<DynRank, P>::from_fn(
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
            crate::numeric_tensor::NumericTensor::from_fn(
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
            crate::numeric_tensor::NumericTensor::from_fn(
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
        gru_eval_pool(self, inputs, pool)
    }

    fn get_milli_op_graph(&self, _ctx: &MilliLoweringContext, _rng: &mut impl Rng) -> MilliOpGraph {
        panic!("GRU uses custom eval, not milli-op decomposition")
    }
}

// ---------------------------------------------------------------------------
// GRU pool-based eval
// ---------------------------------------------------------------------------

/// Pool-based GRU evaluation using an inner MilliOpGraph per timestep.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn gru_eval_pool<'p, P: crate::pool::Pool + 'p>(
    op: &GruOperation,
    inputs: &HashMap<GlobalId, &crate::numeric_tensor::NumericTensorView<'_, DynRank>>,
    pool: &'p P,
) -> Result<HashMap<GlobalId, crate::numeric_tensor::NumericTensor<'p, DynRank, P>>, super::EvalError>
{
    use crate::milli_graph::MilliOpGraph;
    use crate::milli_graph::ops::binary::{MatMul, SimpleBinary};
    use crate::milli_graph::ops::constant::Constant;
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

    let x_seq_first = prepare_x_seq_first(x_view, op.layout, pool)?;

    let seq_len = x_seq_first.shape()[0] as usize;
    let batch = x_seq_first.shape()[1] as usize;
    let input_size = x_seq_first.shape()[2] as usize;
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

        // Slice and transpose W into 3 separate weight matrices: Wz, Wr, Wh
        // W is [num_dirs, 3*hs, input_size] — we want slices [hs, input_size] transposed to [input_size, hs]
        let wz_t = slice_and_transpose_weight(w_view, dir, hs, input_size, dtype, pool)?;
        // For rows hs..2*hs, we need a custom slice since slice_and_transpose_weight always starts at row 0
        let wr_t = {
            let sliced = w_view
                .slice(&[
                    (dir as u64, dir as u64 + 1),
                    (hs as u64, 2 * hs as u64),
                    (0, input_size as u64),
                ])
                .map_err(rnn_map_err)?;
            PoolTensor::<DynRank, P>::from_fn(
                vec![input_size as u64, hs as u64],
                dtype,
                pool,
                |flat| {
                    let c = flat / hs;
                    let r = flat % hs;
                    sliced.read_element(r * input_size + c)
                },
            )
            .map_err(rnn_alloc_err)?
        };
        let wh_t = {
            let sliced = w_view
                .slice(&[
                    (dir as u64, dir as u64 + 1),
                    (2 * hs as u64, 3 * hs as u64),
                    (0, input_size as u64),
                ])
                .map_err(rnn_map_err)?;
            PoolTensor::<DynRank, P>::from_fn(
                vec![input_size as u64, hs as u64],
                dtype,
                pool,
                |flat| {
                    let c = flat / hs;
                    let r = flat % hs;
                    sliced.read_element(r * input_size + c)
                },
            )
            .map_err(rnn_alloc_err)?
        };

        // Slice and transpose R into 3 separate weight matrices: Rz, Rr, Rh
        // R is [num_dirs, 3*hs, hs] — we want slices [hs, hs] transposed to [hs, hs]
        let rz_t = slice_and_transpose_weight(r_view, dir, hs, hs, dtype, pool)?;
        let rr_t = {
            let sliced = r_view
                .slice(&[
                    (dir as u64, dir as u64 + 1),
                    (hs as u64, 2 * hs as u64),
                    (0, hs as u64),
                ])
                .map_err(rnn_map_err)?;
            PoolTensor::<DynRank, P>::from_fn(vec![hs as u64, hs as u64], dtype, pool, |flat| {
                let c = flat / hs;
                let r = flat % hs;
                sliced.read_element(r * hs + c)
            })
            .map_err(rnn_alloc_err)?
        };
        let rh_t = {
            let sliced = r_view
                .slice(&[
                    (dir as u64, dir as u64 + 1),
                    (2 * hs as u64, 3 * hs as u64),
                    (0, hs as u64),
                ])
                .map_err(rnn_map_err)?;
            PoolTensor::<DynRank, P>::from_fn(vec![hs as u64, hs as u64], dtype, pool, |flat| {
                let c = flat / hs;
                let r = flat % hs;
                sliced.read_element(r * hs + c)
            })
            .map_err(rnn_alloc_err)?
        };

        // Biases: ONNX GRU has [Wb_z, Wb_r, Wb_h, Rb_z, Rb_r, Rb_h] = 6*hs
        // Each bias is [hs]
        let biases: Option<(
            PoolTensor<'p, DynRank, P>,
            PoolTensor<'p, DynRank, P>,
            PoolTensor<'p, DynRank, P>,
            PoolTensor<'p, DynRank, P>,
            PoolTensor<'p, DynRank, P>,
            PoolTensor<'p, DynRank, P>,
        )> = if let Some(b) = b_view {
            let b_dir = b
                .slice(&[(dir as u64, dir as u64 + 1), (0, 6 * hs as u64)])
                .map_err(rnn_map_err)?;
            let make_bias = |start: usize| -> Result<PoolTensor<'p, DynRank, P>, EvalError> {
                PoolTensor::<DynRank, P>::from_fn(vec![hs as u64], dtype, pool, |i| {
                    b_dir.read_element(start + i)
                })
                .map_err(rnn_alloc_err)
            };
            Some((
                make_bias(0)?,
                make_bias(hs)?,
                make_bias(2 * hs)?,
                make_bias(3 * hs)?,
                make_bias(4 * hs)?,
                make_bias(5 * hs)?,
            ))
        } else {
            None
        };

        // Initial hidden state
        let mut h_t = extract_init_state(init_h_view, dir, batch, hs, op.layout, dtype, pool)?;

        // Build timestep MilliOpGraph
        let mut rng = wyrand::WyRand::new(42);

        let ext_xt = GlobalId::new(&mut rng);
        let ext_hprev = GlobalId::new(&mut rng);
        let ext_wz_t = GlobalId::new(&mut rng);
        let ext_wr_t = GlobalId::new(&mut rng);
        let ext_wh_t = GlobalId::new(&mut rng);
        let ext_rz_t = GlobalId::new(&mut rng);
        let ext_rr_t = GlobalId::new(&mut rng);
        let ext_rh_t = GlobalId::new(&mut rng);
        let ext_wb_z = GlobalId::new(&mut rng);
        let ext_wb_r = GlobalId::new(&mut rng);
        let ext_wb_h = GlobalId::new(&mut rng);
        let ext_rb_z = GlobalId::new(&mut rng);
        let ext_rb_r = GlobalId::new(&mut rng);
        let ext_rb_h = GlobalId::new(&mut rng);
        let ext_hout = GlobalId::new(&mut rng);

        let mut body = MilliOpGraph::new_empty(&mut rng);
        let int_xt = body.add_input_with_id(ext_xt, &mut rng);
        let int_hprev = body.add_input_with_id(ext_hprev, &mut rng);
        let int_wz_t = body.add_input_with_id(ext_wz_t, &mut rng);
        let int_wr_t = body.add_input_with_id(ext_wr_t, &mut rng);
        let int_wh_t = body.add_input_with_id(ext_wh_t, &mut rng);
        let int_rz_t = body.add_input_with_id(ext_rz_t, &mut rng);
        let int_rr_t = body.add_input_with_id(ext_rr_t, &mut rng);
        let int_rh_t = body.add_input_with_id(ext_rh_t, &mut rng);

        let (int_wb_z, int_wb_r, int_wb_h, int_rb_z, int_rb_r, int_rb_h) = if biases.is_some() {
            (
                Some(body.add_input_with_id(ext_wb_z, &mut rng)),
                Some(body.add_input_with_id(ext_wb_r, &mut rng)),
                Some(body.add_input_with_id(ext_wb_h, &mut rng)),
                Some(body.add_input_with_id(ext_rb_z, &mut rng)),
                Some(body.add_input_with_id(ext_rb_r, &mut rng)),
                Some(body.add_input_with_id(ext_rb_h, &mut rng)),
            )
        } else {
            (None, None, None, None, None, None)
        };

        // z_t = sigmoid(x_t @ wz_t + h_prev @ rz_t + wb_z + rb_z)
        let xw_z = MatMul::push_new(
            &mut body, int_xt, int_wz_t, dtype, dtype, dtype, dtype, &mut rng,
        );
        let hr_z = MatMul::push_new(
            &mut body, int_hprev, int_rz_t, dtype, dtype, dtype, dtype, &mut rng,
        );
        let mut z_gate = SimpleBinary::add(&mut body, xw_z, hr_z, &mut rng);
        if let Some(int_b) = int_wb_z {
            z_gate = SimpleBinary::add(&mut body, z_gate, int_b, &mut rng);
        }
        if let Some(int_b) = int_rb_z {
            z_gate = SimpleBinary::add(&mut body, z_gate, int_b, &mut rng);
        }
        let z_t = build_sigmoid(&mut body, z_gate, &mut rng);

        // r_t = sigmoid(x_t @ wr_t + h_prev @ rr_t + wb_r + rb_r)
        let xw_r = MatMul::push_new(
            &mut body, int_xt, int_wr_t, dtype, dtype, dtype, dtype, &mut rng,
        );
        let hr_r = MatMul::push_new(
            &mut body, int_hprev, int_rr_t, dtype, dtype, dtype, dtype, &mut rng,
        );
        let mut r_gate = SimpleBinary::add(&mut body, xw_r, hr_r, &mut rng);
        if let Some(int_b) = int_wb_r {
            r_gate = SimpleBinary::add(&mut body, r_gate, int_b, &mut rng);
        }
        if let Some(int_b) = int_rb_r {
            r_gate = SimpleBinary::add(&mut body, r_gate, int_b, &mut rng);
        }
        let r_t = build_sigmoid(&mut body, r_gate, &mut rng);

        // h_tilde computation depends on linear_before_reset
        let h_tilde = if op.linear_before_reset {
            // h_tilde = tanh(x_t @ wh_t + wb_h + r_t * (h_prev @ rh_t + rb_h))
            let xw_h = MatMul::push_new(
                &mut body, int_xt, int_wh_t, dtype, dtype, dtype, dtype, &mut rng,
            );
            let hr_h = MatMul::push_new(
                &mut body, int_hprev, int_rh_t, dtype, dtype, dtype, dtype, &mut rng,
            );
            let mut lin = hr_h;
            if let Some(int_b) = int_rb_h {
                lin = SimpleBinary::add(&mut body, lin, int_b, &mut rng);
            }
            let gated = SimpleBinary::mul(&mut body, r_t, lin, &mut rng);
            let mut h_cand = SimpleBinary::add(&mut body, xw_h, gated, &mut rng);
            if let Some(int_b) = int_wb_h {
                h_cand = SimpleBinary::add(&mut body, h_cand, int_b, &mut rng);
            }
            SimpleUnaryOp::trig(&mut body, h_cand, TrigOp::Tanh, &mut rng)
        } else {
            // h_tilde = tanh(x_t @ wh_t + (r_t * h_prev) @ rh_t + wb_h + rb_h)
            let xw_h = MatMul::push_new(
                &mut body, int_xt, int_wh_t, dtype, dtype, dtype, dtype, &mut rng,
            );
            let r_h = SimpleBinary::mul(&mut body, r_t, int_hprev, &mut rng);
            let rh_matmul = MatMul::push_new(
                &mut body, r_h, int_rh_t, dtype, dtype, dtype, dtype, &mut rng,
            );
            let mut h_cand = SimpleBinary::add(&mut body, xw_h, rh_matmul, &mut rng);
            if let Some(int_b) = int_wb_h {
                h_cand = SimpleBinary::add(&mut body, h_cand, int_b, &mut rng);
            }
            if let Some(int_b) = int_rb_h {
                h_cand = SimpleBinary::add(&mut body, h_cand, int_b, &mut rng);
            }
            SimpleUnaryOp::trig(&mut body, h_cand, TrigOp::Tanh, &mut rng)
        };

        // H_t = (1 - z_t) * h_tilde + z_t * h_prev
        let one = Constant::new_scalar(&mut body, 1.0f32, &mut rng);
        let one_minus_z = SimpleBinary::sub(&mut body, one, z_t, &mut rng);
        let lhs = SimpleBinary::mul(&mut body, one_minus_z, h_tilde, &mut rng);
        let rhs = SimpleBinary::mul(&mut body, z_t, int_hprev, &mut rng);
        let h_out = SimpleBinary::add(&mut body, lhs, rhs, &mut rng);
        body.add_output(h_out, ext_hout);

        // Timestep loop
        let mut h_seq: Vec<PoolTensor<'p, DynRank, P>> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let actual_t = if reverse { seq_len - 1 - t } else { t };

            let x_t = extract_timestep(&x_seq_first, actual_t, batch, input_size, dtype, pool)?;

            let x_t_view_ref = x_t.view();
            let h_prev_view = h_t.view();
            let wz_t_view = wz_t.view();
            let wr_t_view = wr_t.view();
            let wh_t_view = wh_t.view();
            let rz_t_view = rz_t.view();
            let rr_t_view = rr_t.view();
            let rh_t_view = rh_t.view();

            let mut body_inputs: HashMap<
                GlobalId,
                &crate::numeric_tensor::NumericTensorView<'_, DynRank>,
            > = HashMap::new();
            body_inputs.insert(ext_xt, &x_t_view_ref);
            body_inputs.insert(ext_hprev, &h_prev_view);
            body_inputs.insert(ext_wz_t, &wz_t_view);
            body_inputs.insert(ext_wr_t, &wr_t_view);
            body_inputs.insert(ext_wh_t, &wh_t_view);
            body_inputs.insert(ext_rz_t, &rz_t_view);
            body_inputs.insert(ext_rr_t, &rr_t_view);
            body_inputs.insert(ext_rh_t, &rh_t_view);

            let (wb_z_view, wb_r_view, wb_h_view, rb_z_view, rb_r_view, rb_h_view);
            if let Some(ref b) = biases {
                wb_z_view = b.0.view();
                wb_r_view = b.1.view();
                wb_h_view = b.2.view();
                rb_z_view = b.3.view();
                rb_r_view = b.4.view();
                rb_h_view = b.5.view();
                body_inputs.insert(ext_wb_z, &wb_z_view);
                body_inputs.insert(ext_wb_r, &wb_r_view);
                body_inputs.insert(ext_wb_h, &wb_h_view);
                body_inputs.insert(ext_rb_z, &rb_z_view);
                body_inputs.insert(ext_rb_r, &rb_r_view);
                body_inputs.insert(ext_rb_h, &rb_h_view);
            }

            let body_outputs = body
                .pool_eval(&body_inputs, pool)
                .map_err(|e| EvalError::InvalidInput(format!("GRU timestep eval failed: {e}")))?;

            h_t = body_outputs
                .into_iter()
                .find(|(id, _)| *id == ext_hout)
                .map(|(_, t)| t)
                .ok_or_else(|| EvalError::InvalidInput("GRU: missing h_out".into()))?;
            h_seq.push(h_t.to_tensor(pool).map_err(rnn_alloc_err)?);
        }

        if reverse {
            h_seq.reverse();
        }

        let y_dir = stack_hidden_seq(&h_seq, batch, hs, dtype, pool)?;
        all_y.push(y_dir);

        let yh = unsqueeze_hidden(&h_t, batch, hs, op.layout, dtype, pool)?;
        all_yh.push(yh);
    }

    assemble_rnn_outputs(
        &op.outputs,
        all_y,
        all_yh,
        None,
        seq_len,
        num_dirs,
        batch,
        hs,
        op.layout,
        dtype,
        pool,
    )
}
