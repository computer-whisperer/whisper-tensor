//! Composite graph test cases: multi-op patterns from real neural networks.
//!
//! Tests atom map propagation across chains of nano-lowered ops.

use std::collections::HashMap;

use ndarray::{ArcArray, IxDyn};
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::graph::GlobalId;
use crate::milli_graph::ops::{self, Constant};
use crate::milli_graph::{self, MilliOpGraph, ops_helpers};
use crate::numeric_dtype::NumericDType;

use super::tensor_f32_shaped;
use super::{TestCase, TestDataSet, Tolerance};

pub fn build_cases() -> Vec<TestCase> {
    vec![
        linear_relu(),
        softmax_1d(),
        layernorm(),
        residual_add(),
        attention_scores(),
        gelu_approx(),
        conv_relu(),
        gather_layernorm(),
        rms_norm_multirow(),
        reduce_mean_broadcast_mul(),
        cast_then_self_mul(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_set::run_case_via_graph_pool_eval;

    #[test]
    fn test_rms_norm_via_symbolic_eval_pool() {
        test_rms_norm_symbolic_eval_pool();
    }

    #[test]
    fn test_rotary_embedding_via_symbolic_eval_pool() {
        use crate::numeric_tensor::NumericTensorView;
        use crate::pool::TrackedPool;
        use crate::symbolic_graph::ops::{Operation, RotaryEmbeddingOperation};

        let mut rng = rng();
        let data_id = GlobalId::new(&mut rng);
        let cos_id = GlobalId::new(&mut rng);
        let sin_id = GlobalId::new(&mut rng);
        let pos_id = GlobalId::new(&mut rng);
        let out_id = GlobalId::new(&mut rng);

        let op = RotaryEmbeddingOperation::new(
            data_id,
            cos_id,
            sin_id,
            Some(pos_id),
            out_id,
            false, // not interleaved
            None,  // no num_heads (4D input)
            0,     // full rotation
            &mut rng,
        );

        let pool = TrackedPool::new(None);

        // Minimal test: [B=1, H=1, S=1, D=4], cos/sin cache [2, 2], pos_ids [1, 1]
        // x1 = data[:2] = [1, 2], x2 = data[2:] = [3, 4]
        // cos = cos_cache[pos[0,0]] = cos_cache[1], sin = sin_cache[1]
        // real = cos*x1 - sin*x2, imag = sin*x1 + cos*x2
        // output = [real[0], real[1], imag[0], imag[1]]
        let data = super::tensor_f32_shaped(vec![1, 1, 1, 4], &[1.0, 2.0, 3.0, 4.0]);
        let cos_cache = super::tensor_f32_shaped(vec![2, 2], &[0.5, 0.6, 0.7, 0.8]);
        let sin_cache = super::tensor_f32_shaped(vec![2, 2], &[0.1, 0.2, 0.3, 0.4]);
        let pos_ids = crate::test_set::tensor_i64_shaped(vec![1, 1], &[1]);

        let input_views: HashMap<GlobalId, _> = HashMap::from([
            (data_id, data.view()),
            (cos_id, cos_cache.view()),
            (sin_id, sin_cache.view()),
            (pos_id, pos_ids.view()),
        ]);
        let input_refs: HashMap<GlobalId, &NumericTensorView<'_, crate::tensor_rank::DynRank>> =
            input_views.iter().map(|(&id, v)| (id, v)).collect();

        // Run through legacy eval with observer to get correct intermediate values.
        {
            use crate::milli_graph::observer::MilliOpGraphObserver;
            struct DumpObserver;
            impl MilliOpGraphObserver for DumpObserver {
                fn on_tensor_assigned(
                    &mut self,
                    path: &[GlobalId],
                    tensor: &crate::numeric_tensor::NumericTensorView<
                        '_,
                        crate::tensor_rank::DynRank,
                    >,
                ) {
                    let id = path.last().copied().unwrap_or(GlobalId(0));
                    let vals: Vec<f64> = (0..tensor.numel().min(8))
                        .map(|i| tensor.read_element(i).to_f64())
                        .collect();
                    eprintln!(
                        "  [legacy] {id} shape={:?} dtype={:?} vals={vals:?}",
                        tensor.shape(),
                        tensor.dtype()
                    );
                }
                fn on_node_executed(
                    &mut self,
                    _: &[GlobalId],
                    _: std::time::Instant,
                    _: std::time::Instant,
                ) {
                }
            }

            let tensor_dtypes: HashMap<GlobalId, crate::numeric_dtype::NumericDType> = input_refs
                .iter()
                .map(|(id, view)| (*id, view.dtype()))
                .collect();
            let mctx = crate::milli_graph::MilliLoweringContext::new(tensor_dtypes);
            let mut rng2 = SmallRng::seed_from_u64(0);
            let mg = op.get_milli_op_graph(&mctx, &mut rng2);

            // Legacy eval — should produce correct results.
            let legacy_inputs: HashMap<
                GlobalId,
                crate::migration::numeric_tensor::NumericTensor<crate::tensor_rank::DynRank>,
            > = input_views
                .iter()
                .map(|(&id, v)| (id, crate::migration::bridge::view_to_legacy(v)))
                .collect();
            let mut backend = crate::backends::eval_backend::EvalBackend::NDArray;
            let mut obs = DumpObserver;
            eprintln!("=== Legacy eval ===");
            let legacy_out: HashMap<_, _> = mg
                .eval(&legacy_inputs, &mut obs, &mut backend)
                .expect("legacy eval")
                .collect();
            for (id, t) in &legacy_out {
                let pool_t = crate::migration::bridge::legacy_to_new(t);
                let vals: Vec<f64> = (0..pool_t.numel().min(8))
                    .map(|i| pool_t.read_element(i).to_f64())
                    .collect();
                eprintln!(
                    "  [legacy output] {id} shape={:?} vals={vals:?}",
                    pool_t.shape()
                );
            }

            // Pool eval for comparison.
            eprintln!("=== Pool eval ===");
            let pool2 = TrackedPool::new(None);
            let pool_out = mg.pool_eval(&input_refs, &pool2).expect("pool_eval");
            for (id, t) in &pool_out {
                let vals: Vec<f64> = (0..t.numel().min(8))
                    .map(|i| t.read_element(i).to_f64())
                    .collect();
                eprintln!("  [pool output] {id} shape={:?} vals={vals:?}", t.shape());
            }
        }

        let onnx_inputs: HashMap<GlobalId, crate::numeric_dtype::ONNXTensorView<'_>> = [
            (
                data_id,
                crate::numeric_dtype::ONNXTensorView::Numeric(data.view()),
            ),
            (
                cos_id,
                crate::numeric_dtype::ONNXTensorView::Numeric(cos_cache.view()),
            ),
            (
                sin_id,
                crate::numeric_dtype::ONNXTensorView::Numeric(sin_cache.view()),
            ),
            (
                pos_id,
                crate::numeric_dtype::ONNXTensorView::Numeric(pos_ids.view()),
            ),
        ]
        .into_iter()
        .collect();
        let results = op.eval_pool(&onnx_inputs, &pool).expect("eval_pool failed");
        let out = results
            .get(&out_id)
            .expect("output not found")
            .as_numeric()
            .unwrap();

        // After transpose [B,H,S,D] -> [B,S,H,D]: still [1,1,1,4]
        // x1 = [1, 2], x2 = [3, 4]
        // cos_cache[1] = [0.7, 0.8], sin_cache[1] = [0.3, 0.4]
        // real = [0.7*1 - 0.3*3, 0.8*2 - 0.4*4] = [0.7-0.9, 1.6-1.6] = [-0.2, 0.0]
        // imag = [0.3*1 + 0.7*3, 0.4*2 + 0.8*4] = [0.3+2.1, 0.8+3.2] = [2.4, 4.0]
        // output (pre-transpose) = [-0.2, 0.0, 2.4, 4.0]
        // After transpose back [B,S,H,D] -> [B,H,S,D]: still [1,1,1,4]
        let expected: Vec<f32> = vec![-0.2, 0.0, 2.4, 4.0];

        eprintln!("rotary_embedding output ({} elements):", out.numel());
        for i in 0..out.numel() {
            let actual = out.read_element(i).to_f64();
            eprintln!("  [{i}] actual={actual:.6} expected={:.6}", expected[i]);
        }

        for (i, &exp) in expected.iter().enumerate() {
            let actual = out.read_element(i).to_f64() as f32;
            let diff = (actual - exp).abs();
            assert!(
                diff < 1e-4,
                "element {i}: actual {actual} vs expected {exp} (diff {diff})"
            );
        }
    }
}

fn rng() -> SmallRng {
    SmallRng::seed_from_u64(700)
}

fn f32_const(
    graph: &mut MilliOpGraph,
    vals: Vec<f32>,
    shape: Vec<usize>,
    rng: &mut SmallRng,
) -> GlobalId {
    let t = NDArrayNumericTensor::F32(ArcArray::from_shape_vec(IxDyn(&shape), vals).unwrap());
    Constant::push_new(graph, t, rng)
}

fn i64_const(graph: &mut MilliOpGraph, vals: Vec<i64>, rng: &mut SmallRng) -> GlobalId {
    Constant::from_vec(graph, vals, rng)
}

// ---------------------------------------------------------------------------
// Linear + ReLU: y = max(0, x @ W + b)
// ---------------------------------------------------------------------------

fn linear_relu() -> TestCase {
    let mut rng = rng();
    let ext_x = GlobalId::new(&mut rng);
    let (mut graph, imap) = MilliOpGraph::new([ext_x], &mut rng);
    let x = imap[&ext_x];

    // W [3, 2], b [2]
    let w = f32_const(
        &mut graph,
        vec![1.0, -1.0, 0.5, 0.5, -0.5, 1.0],
        vec![3, 2],
        &mut rng,
    );
    let b = f32_const(&mut graph, vec![0.1, -0.1], vec![1, 2], &mut rng);

    // y = ClampMin(x @ W + b, 0)
    let xw = ops::MatMul::push_new_default_precision(&mut graph, x, w, NumericDType::F32, &mut rng);
    let xwb = ops::SimpleBinary::add(&mut graph, xw, b, &mut rng);
    let out = ops::ClampMin::push_new(&mut graph, xwb, 0.0, &mut rng);
    graph.set_outputs(vec![out]);

    // x = [[1, 0, -1], [2, 1, 0]]
    // x@W = [[1*1+0*0.5+(-1)*(-0.5), 1*(-1)+0*0.5+(-1)*1],
    //        [2*1+1*0.5+0*(-0.5),    2*(-1)+1*0.5+0*1    ]]
    //     = [[1.5, -2.0], [2.5, -1.5]]
    // +b  = [[1.6, -2.1], [2.6, -1.6]]
    // relu= [[1.6,  0.0], [2.6,  0.0]]
    TestCase {
        name: "linear_relu".into(),
        graph,
        data_sets: vec![TestDataSet {
            label: "2x3".into(),
            inputs: HashMap::from([(
                ext_x,
                tensor_f32_shaped(vec![2, 3], &[1.0, 0.0, -1.0, 2.0, 1.0, 0.0]),
            )]),
            expected_outputs: HashMap::from([(
                out,
                tensor_f32_shaped(vec![2, 2], &[1.6, 0.0, 2.6, 0.0]),
            )]),
            tolerance: Tolerance {
                atol: 1e-5,
                rtol: 1e-5,
            },
        }],
    }
}

// ---------------------------------------------------------------------------
// Softmax: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
// ---------------------------------------------------------------------------

fn softmax_1d() -> TestCase {
    let mut rng = rng();
    let ext_x = GlobalId::new(&mut rng);
    let (mut graph, imap) = MilliOpGraph::new([ext_x], &mut rng);
    let x = imap[&ext_x];

    // ReduceMax along axis 1
    let axes = i64_const(&mut graph, vec![1], &mut rng);
    let x_max = ops::ReduceMax::push_new(&mut graph, x, Some(axes), true, false, &mut rng);

    // x - max (broadcast)
    let x_shifted = ops::SimpleBinary::sub(&mut graph, x, x_max, &mut rng);

    // exp(x - max)
    let x_exp = ops::SimpleUnaryOp::exp(&mut graph, x_shifted, &mut rng);

    // sum(exp(x - max)) along axis 1
    let axes2 = i64_const(&mut graph, vec![1], &mut rng);
    let x_sum = ops::ReduceSum::push_new(&mut graph, x_exp, Some(axes2), true, false, &mut rng);

    // exp / sum
    let out = ops::SimpleBinary::div(&mut graph, x_exp, x_sum, &mut rng);
    graph.set_outputs(vec![out]);

    // x = [[1, 2, 3]]
    // max = 3, shifted = [-2, -1, 0]
    // exp = [e^-2, e^-1, 1] = [0.1353, 0.3679, 1.0]
    // sum = 1.5032
    // softmax = [0.0900, 0.2447, 0.6652]
    let e0 = (-2.0f32).exp();
    let e1 = (-1.0f32).exp();
    let e2 = 1.0f32;
    let s = e0 + e1 + e2;

    TestCase {
        name: "softmax_1d".into(),
        graph,
        data_sets: vec![TestDataSet {
            label: "1x3".into(),
            inputs: HashMap::from([(ext_x, tensor_f32_shaped(vec![1, 3], &[1.0, 2.0, 3.0]))]),
            expected_outputs: HashMap::from([(
                out,
                tensor_f32_shaped(vec![1, 3], &[e0 / s, e1 / s, e2 / s]),
            )]),
            tolerance: Tolerance {
                atol: 1e-5,
                rtol: 1e-5,
            },
        }],
    }
}

// ---------------------------------------------------------------------------
// LayerNorm: y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
// ---------------------------------------------------------------------------

fn layernorm() -> TestCase {
    let mut rng = rng();
    let ext_x = GlobalId::new(&mut rng);
    let (mut graph, imap) = MilliOpGraph::new([ext_x], &mut rng);
    let x = imap[&ext_x];

    let gamma = f32_const(&mut graph, vec![1.0, 1.0, 1.0, 1.0], vec![1, 4], &mut rng);
    let beta = f32_const(&mut graph, vec![0.0, 0.0, 0.0, 0.0], vec![1, 4], &mut rng);
    let eps = f32_const(&mut graph, vec![1e-5], vec![1, 1], &mut rng);
    let two = f32_const(&mut graph, vec![2.0], vec![1, 1], &mut rng);

    // mean = ReduceMean(x, axis=-1, keepdims=true)
    let axes = i64_const(&mut graph, vec![-1], &mut rng);
    let mean = ops::ReduceMean::push_new(&mut graph, x, Some(axes), true, false, &mut rng);

    // x_centered = x - mean
    let x_c = ops::SimpleBinary::sub(&mut graph, x, mean, &mut rng);

    // var = mean(x_centered^2)
    let x_c2 = ops::Pow::push_new(&mut graph, x_c, two, &mut rng);
    let axes2 = i64_const(&mut graph, vec![-1], &mut rng);
    let var = ops::ReduceMean::push_new(&mut graph, x_c2, Some(axes2), true, false, &mut rng);

    // std = sqrt(var + eps)
    let var_eps = ops::SimpleBinary::add(&mut graph, var, eps, &mut rng);
    let std = ops::SimpleUnaryOp::sqrt(&mut graph, var_eps, &mut rng);

    // norm = x_centered / std
    let norm = ops::SimpleBinary::div(&mut graph, x_c, std, &mut rng);

    // y = norm * gamma + beta
    let scaled = ops::SimpleBinary::mul(&mut graph, norm, gamma, &mut rng);
    let out = ops::SimpleBinary::add(&mut graph, scaled, beta, &mut rng);
    graph.set_outputs(vec![out]);

    // x = [[1, 2, 3, 4]]
    // mean = 2.5
    // centered = [-1.5, -0.5, 0.5, 1.5]
    // var = (2.25 + 0.25 + 0.25 + 2.25)/4 = 1.25
    // std = sqrt(1.25 + 1e-5) ≈ 1.11803
    // With gamma=1, beta=0: just the normalized values
    let c = [-1.5f32, -0.5, 0.5, 1.5];
    let v = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2] + c[3] * c[3]) / 4.0;
    let s = (v + 1e-5).sqrt();
    let expected: Vec<f32> = c.iter().map(|&ci| ci / s).collect();

    TestCase {
        name: "layernorm".into(),
        graph,
        data_sets: vec![TestDataSet {
            label: "1x4".into(),
            inputs: HashMap::from([(ext_x, tensor_f32_shaped(vec![1, 4], &[1.0, 2.0, 3.0, 4.0]))]),
            expected_outputs: HashMap::from([(out, tensor_f32_shaped(vec![1, 4], &expected))]),
            tolerance: Tolerance {
                atol: 1e-4,
                rtol: 1e-4,
            },
        }],
    }
}

// ---------------------------------------------------------------------------
// Residual: y = relu(x + conv(x))
// ---------------------------------------------------------------------------

fn residual_add() -> TestCase {
    let mut rng = rng();
    let ext_x = GlobalId::new(&mut rng);
    let (mut graph, imap) = MilliOpGraph::new([ext_x], &mut rng);
    let x = imap[&ext_x];

    // Simple 1x1 conv (identity-ish): weight [1, 1, 1, 1], no bias
    let w = f32_const(&mut graph, vec![2.0], vec![1, 1, 1, 1], &mut rng);
    let conv_out = ops::Conv::push_new(
        &mut graph,
        x,
        w,
        None,
        ops::ConvAutoPad::Valid,
        vec![],
        1,
        vec![1, 1],
        vec![],
        vec![1, 1],
        &mut rng,
    );

    // residual = x + conv(x)
    let res = ops::SimpleBinary::add(&mut graph, x, conv_out, &mut rng);
    let out = ops::ClampMin::push_new(&mut graph, res, 0.0, &mut rng);
    graph.set_outputs(vec![out]);

    // x = [[[[1, -1], [2, -2]]]]  shape [1,1,2,2]
    // conv(x) = 2*x = [[[[2, -2], [4, -4]]]]
    // x + conv(x) = [[[[3, -3], [6, -6]]]]
    // relu = [[[[3, 0], [6, 0]]]]
    TestCase {
        name: "residual_add".into(),
        graph,
        data_sets: vec![TestDataSet {
            label: "1x1x2x2".into(),
            inputs: HashMap::from([(
                ext_x,
                tensor_f32_shaped(vec![1, 1, 2, 2], &[1.0, -1.0, 2.0, -2.0]),
            )]),
            expected_outputs: HashMap::from([(
                out,
                tensor_f32_shaped(vec![1, 1, 2, 2], &[3.0, 0.0, 6.0, 0.0]),
            )]),
            tolerance: Tolerance::for_dtype(NumericDType::F32),
        }],
    }
}

// ---------------------------------------------------------------------------
// Attention scores: scores = (Q @ K^T) / sqrt(d_k)
// ---------------------------------------------------------------------------

fn attention_scores() -> TestCase {
    let mut rng = rng();
    let ext_q = GlobalId::new(&mut rng);
    let ext_k = GlobalId::new(&mut rng);
    let (mut graph, imap) = MilliOpGraph::new([ext_q, ext_k], &mut rng);

    // Q [1, 2, 3], K [1, 2, 3] → K^T [1, 3, 2] → Q @ K^T [1, 2, 2]
    let k_t = ops::Transpose::push_new(&mut graph, imap[&ext_k], Some(vec![0, 2, 1]), &mut rng);
    let qk = ops::MatMul::push_new_default_precision(
        &mut graph,
        imap[&ext_q],
        k_t,
        NumericDType::F32,
        &mut rng,
    );

    // / sqrt(d_k) where d_k = 3
    let sqrt_dk = f32_const(&mut graph, vec![3.0f32.sqrt()], vec![1, 1, 1], &mut rng);
    let out = ops::SimpleBinary::div(&mut graph, qk, sqrt_dk, &mut rng);
    graph.set_outputs(vec![out]);

    // Q = [[1,0,0],[0,1,0]], K = [[1,0,0],[0,0,1]]
    // K^T = [[1,0],[0,0],[0,1]]
    // Q@K^T = [[1,0],[0,1]]
    // / sqrt(3) = [[0.5774, 0], [0, 0.5774]]
    let s = 1.0 / 3.0f32.sqrt();

    TestCase {
        name: "attention_scores".into(),
        graph,
        data_sets: vec![TestDataSet {
            label: "1x2x3".into(),
            inputs: HashMap::from([
                (
                    ext_q,
                    tensor_f32_shaped(vec![1, 2, 3], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
                ),
                (
                    ext_k,
                    tensor_f32_shaped(vec![1, 2, 3], &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
                ),
            ]),
            expected_outputs: HashMap::from([(
                out,
                tensor_f32_shaped(vec![1, 2, 2], &[s, 0.0, 0.0, 0.0]),
            )]),
            tolerance: Tolerance {
                atol: 1e-5,
                rtol: 1e-5,
            },
        }],
    }
}

// ---------------------------------------------------------------------------
// GELU approximation: gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
// ---------------------------------------------------------------------------

fn gelu_approx() -> TestCase {
    let mut rng = rng();
    let ext_x = GlobalId::new(&mut rng);
    let (mut graph, imap) = MilliOpGraph::new([ext_x], &mut rng);
    let x = imap[&ext_x];

    let half = f32_const(&mut graph, vec![0.5], vec![1], &mut rng);
    let sqrt2 = f32_const(
        &mut graph,
        vec![std::f32::consts::SQRT_2],
        vec![1],
        &mut rng,
    );
    let one = f32_const(&mut graph, vec![1.0], vec![1], &mut rng);

    // x / sqrt(2)
    let x_s = ops::SimpleBinary::div(&mut graph, x, sqrt2, &mut rng);
    // erf(x / sqrt(2))
    let erf_x = ops::SimpleUnaryOp::erf(&mut graph, x_s, &mut rng);
    // 1 + erf(...)
    let one_plus = ops::SimpleBinary::add(&mut graph, one, erf_x, &mut rng);
    // 0.5 * x
    let half_x = ops::SimpleBinary::mul(&mut graph, half, x, &mut rng);
    // result
    let out = ops::SimpleBinary::mul(&mut graph, half_x, one_plus, &mut rng);
    graph.set_outputs(vec![out]);

    // Abramowitz & Stegun erf approximation (matches scalar_ops::erf)
    fn erf_approx(x: f64) -> f64 {
        let sign = x.signum();
        let x = x.abs();
        let t = 1.0 / (1.0 + 0.3275911 * x);
        let poly = t
            * (0.254829592
                + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
        sign * (1.0 - poly * (-x * x).exp())
    }

    fn gelu(x: f32) -> f32 {
        let xd = x as f64;
        (0.5 * xd * (1.0 + erf_approx(xd / std::f64::consts::SQRT_2))) as f32
    }

    TestCase {
        name: "gelu_approx".into(),
        graph,
        data_sets: vec![TestDataSet {
            label: "4elem".into(),
            inputs: HashMap::from([(ext_x, tensor_f32_shaped(vec![4], &[-1.0, 0.0, 1.0, 2.0]))]),
            expected_outputs: HashMap::from([(
                out,
                tensor_f32_shaped(vec![4], &[gelu(-1.0), gelu(0.0), gelu(1.0), gelu(2.0)]),
            )]),
            tolerance: Tolerance {
                atol: 1e-4,
                rtol: 1e-4,
            },
        }],
    }
}

// ---------------------------------------------------------------------------
// Conv + ReLU
// ---------------------------------------------------------------------------

fn conv_relu() -> TestCase {
    let mut rng = rng();
    let ext_x = GlobalId::new(&mut rng);
    let (mut graph, imap) = MilliOpGraph::new([ext_x], &mut rng);
    let x = imap[&ext_x];

    // 3x3 conv with bias, then ReLU
    // Input [1,1,4,4], Weight [1,1,3,3], Bias [1]
    let w = f32_const(
        &mut graph,
        vec![0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0],
        vec![1, 1, 3, 3],
        &mut rng,
    );
    let b = f32_const(&mut graph, vec![1.0], vec![1], &mut rng);

    let conv = ops::Conv::push_new(
        &mut graph,
        x,
        w,
        Some(b),
        ops::ConvAutoPad::Valid,
        vec![],
        1,
        vec![3, 3],
        vec![],
        vec![1, 1],
        &mut rng,
    );
    let out = ops::ClampMin::push_new(&mut graph, conv, 0.0, &mut rng);
    graph.set_outputs(vec![out]);

    // Input: 4x4 of all 1s → Laplacian filter + bias
    // For interior: 1*0 + 1*1 + 1*0 + 1*1 + 1*(-4) + 1*1 + 1*0 + 1*1 + 1*0 = 0
    // + bias 1 = 1. All outputs = 1. relu(1) = 1.
    TestCase {
        name: "conv_relu".into(),
        graph,
        data_sets: vec![TestDataSet {
            label: "1x1x4x4_ones".into(),
            inputs: HashMap::from([(ext_x, tensor_f32_shaped(vec![1, 1, 4, 4], &[1.0; 16]))]),
            expected_outputs: HashMap::from([(
                out,
                tensor_f32_shaped(vec![1, 1, 2, 2], &[1.0, 1.0, 1.0, 1.0]),
            )]),
            tolerance: Tolerance::for_dtype(NumericDType::F32),
        }],
    }
}

// ---------------------------------------------------------------------------
// Gather + LayerNorm (embedding lookup + normalization)
// ---------------------------------------------------------------------------

fn gather_layernorm() -> TestCase {
    let mut rng = rng();
    let ext_ids = GlobalId::new(&mut rng);
    let (mut graph, imap) = MilliOpGraph::new([ext_ids], &mut rng);
    let ids = imap[&ext_ids];

    // Embedding table [4, 3] (4 tokens, dim 3)
    let table = f32_const(
        &mut graph,
        vec![
            1.0, 0.0, 0.0, // token 0
            0.0, 2.0, 0.0, // token 1
            0.0, 0.0, 3.0, // token 2
            1.0, 1.0, 1.0, // token 3
        ],
        vec![4, 3],
        &mut rng,
    );

    // Gather: lookup tokens
    let emb = ops::Gather::push_new(&mut graph, table, ids, 0, &mut rng);

    // LayerNorm on last axis (simplified: gamma=1, beta=0)
    let eps = f32_const(&mut graph, vec![1e-5], vec![1, 1], &mut rng);
    let two = f32_const(&mut graph, vec![2.0], vec![1, 1], &mut rng);

    let axes = i64_const(&mut graph, vec![-1], &mut rng);
    let mean = ops::ReduceMean::push_new(&mut graph, emb, Some(axes), true, false, &mut rng);
    let centered = ops::SimpleBinary::sub(&mut graph, emb, mean, &mut rng);
    let sq = ops::Pow::push_new(&mut graph, centered, two, &mut rng);
    let axes2 = i64_const(&mut graph, vec![-1], &mut rng);
    let var = ops::ReduceMean::push_new(&mut graph, sq, Some(axes2), true, false, &mut rng);
    let var_eps = ops::SimpleBinary::add(&mut graph, var, eps, &mut rng);
    let std = ops::SimpleUnaryOp::sqrt(&mut graph, var_eps, &mut rng);
    let out = ops::SimpleBinary::div(&mut graph, centered, std, &mut rng);
    graph.set_outputs(vec![out]);

    // indices = [3, 0] → embed = [[1,1,1], [1,0,0]]
    // token 3: mean=1, centered=[0,0,0], var=0, std=sqrt(1e-5)≈0.00316
    //   → [0, 0, 0]
    // token 0: mean=1/3, centered=[2/3, -1/3, -1/3]
    //   var = (4/9 + 1/9 + 1/9)/3 = 6/27 = 2/9 ≈ 0.2222
    //   std = sqrt(2/9 + 1e-5) ≈ 0.4714
    //   norm = [2/3 / 0.4714, -1/3 / 0.4714, -1/3 / 0.4714]
    //        ≈ [1.4142, -0.7071, -0.7071]
    let m0 = 1.0f32 / 3.0;
    let c0 = [1.0 - m0, 0.0 - m0, 0.0 - m0];
    let v0 = (c0[0] * c0[0] + c0[1] * c0[1] + c0[2] * c0[2]) / 3.0;
    let s0 = (v0 + 1e-5).sqrt();

    TestCase {
        name: "gather_layernorm".into(),
        graph,
        data_sets: vec![TestDataSet {
            label: "2_tokens".into(),
            inputs: HashMap::from([(ext_ids, super::tensor_i64_shaped(vec![2], &[3, 0]))]),
            expected_outputs: HashMap::from([(
                out,
                tensor_f32_shaped(
                    vec![2, 3],
                    &[0.0, 0.0, 0.0, c0[0] / s0, c0[1] / s0, c0[2] / s0],
                ),
            )]),
            tolerance: Tolerance {
                atol: 1e-3,
                rtol: 1e-3,
            },
        }],
    }
}

// ---------------------------------------------------------------------------
// RMS Normalization (multi-row) — matches ONNX RMSNormalization structure.
// y = (x / rms(x)) * scale, where rms = sqrt(mean(x², axis=-1) + eps)
// Uses Cast to F32, matching the real symbolic op's milli graph.
// ---------------------------------------------------------------------------

fn rms_norm_multirow() -> TestCase {
    let mut rng = rng();
    let ext_x = GlobalId::new(&mut rng);
    let ext_scale = GlobalId::new(&mut rng);
    let (mut graph, imap) = MilliOpGraph::new([ext_x, ext_scale], &mut rng);
    let x = imap[&ext_x];
    let scale = imap[&ext_scale];

    // Cast to F32 (identity when input is already F32, but exercises the Cast op path)
    let x_f32 = ops::Cast::push_new(&mut graph, x, NumericDType::F32, &mut rng);

    // x² = x_f32 * x_f32
    let x_sq = ops::SimpleBinary::mul(&mut graph, x_f32, x_f32, &mut rng);

    // mean(x², axis=-1, keepdims=true) — use resolve_axes + Range to match the
    // real RMSNormalizationOperation::get_milli_op_graph exactly.
    let axis_raw = ops_helpers::scalar_const(&mut graph, -1i64, &mut rng);
    let axis_resolved = ops_helpers::resolve_axes(&mut graph, axis_raw, x, &mut rng);
    let rank_tid = ops_helpers::rank(&mut graph, x, &mut rng);
    let step_tid = ops_helpers::scalar_const(&mut graph, 1i64, &mut rng);
    let normalized_axes =
        milli_graph::ops::Range::push_new(&mut graph, axis_resolved, rank_tid, step_tid, &mut rng);
    let sq_mean = ops::ReduceMean::push_new(
        &mut graph,
        x_sq,
        Some(normalized_axes),
        true,
        false,
        &mut rng,
    );

    // sqrt(mean + eps)
    let eps = f32_const(&mut graph, vec![1e-5], vec![1, 1], &mut rng);
    let mean_eps = ops::SimpleBinary::add(&mut graph, sq_mean, eps, &mut rng);
    let rms = ops::SimpleUnaryOp::sqrt(&mut graph, mean_eps, &mut rng);

    // 1 / rms
    let rms_inv = ops::SimpleUnaryOp::reciprocal(&mut graph, rms, &mut rng);

    // normalized = x_f32 * rms_inv
    let normalized = ops::SimpleBinary::mul(&mut graph, x_f32, rms_inv, &mut rng);

    // CastLike back to input dtype (identity for F32)
    let normalized = ops::CastLike::push_new(&mut graph, normalized, x, &mut rng);

    // out = normalized * scale
    let out = ops::SimpleBinary::mul(&mut graph, normalized, scale, &mut rng);
    graph.set_outputs(vec![out]);

    // x = [[1.7640524, 0.4001572, 0.978738, 2.2408931],
    //      [1.867558, -0.9772779, 0.95008844, -0.1513572],
    //      [-0.10321885, 0.41059852, 0.14404356, 1.4542735]]
    // scale = [1.2302907, 1.2023798, -0.3873268, -0.30230275]
    let x_data: Vec<f32> = vec![
        1.7640524,
        0.4001572,
        0.978738,
        2.2408931,
        1.867558,
        -0.9772779,
        0.95008844,
        -0.1513572,
        -0.10321885,
        0.41059852,
        0.14404356,
        1.4542735,
    ];
    let scale_data: Vec<f32> = vec![1.2302907, 1.2023798, -0.3873268, -0.30230275];

    // Compute expected output manually.
    let mut expected = vec![0.0f32; 12];
    for row in 0..3 {
        let sq_sum: f32 = (0..4)
            .map(|c| x_data[row * 4 + c] * x_data[row * 4 + c])
            .sum();
        let rms_val = (sq_sum / 4.0 + 1e-5).sqrt();
        for col in 0..4 {
            expected[row * 4 + col] = x_data[row * 4 + col] / rms_val * scale_data[col];
        }
    }

    TestCase {
        name: "rms_norm_multirow".into(),
        graph,
        data_sets: vec![TestDataSet {
            label: "3x4".into(),
            inputs: HashMap::from([
                (ext_x, tensor_f32_shaped(vec![3, 4], &x_data)),
                (ext_scale, tensor_f32_shaped(vec![4], &scale_data)),
            ]),
            expected_outputs: HashMap::from([(out, tensor_f32_shaped(vec![3, 4], &expected))]),
            tolerance: Tolerance {
                atol: 1e-5,
                rtol: 1e-5,
            },
        }],
    }
}

// ---------------------------------------------------------------------------
// ReduceMean → broadcast multiply: isolates the broadcast pattern.
// out = x * mean(x, axis=-1, keepdims=true)
// ---------------------------------------------------------------------------

fn reduce_mean_broadcast_mul() -> TestCase {
    let mut rng = rng();
    let ext_x = GlobalId::new(&mut rng);
    let (mut graph, imap) = MilliOpGraph::new([ext_x], &mut rng);
    let x = imap[&ext_x];

    let axes = i64_const(&mut graph, vec![-1], &mut rng);
    let mean = ops::ReduceMean::push_new(&mut graph, x, Some(axes), true, false, &mut rng);
    let out = ops::SimpleBinary::mul(&mut graph, x, mean, &mut rng);
    graph.set_outputs(vec![out]);

    // x = [[1, 2, 3], [4, 5, 6]]
    // mean(axis=-1) = [[2], [5]]
    // out = [[1*2, 2*2, 3*2], [4*5, 5*5, 6*5]] = [[2, 4, 6], [20, 25, 30]]
    TestCase {
        name: "reduce_mean_broadcast_mul".into(),
        graph,
        data_sets: vec![TestDataSet {
            label: "2x3".into(),
            inputs: HashMap::from([(
                ext_x,
                tensor_f32_shaped(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            )]),
            expected_outputs: HashMap::from([(
                out,
                tensor_f32_shaped(vec![2, 3], &[2.0, 4.0, 6.0, 20.0, 25.0, 30.0]),
            )]),
            tolerance: Tolerance {
                atol: 1e-5,
                rtol: 1e-5,
            },
        }],
    }
}

// ---------------------------------------------------------------------------
// Cast (identity F32→F32) then self-multiply: x_f32 = Cast(x), out = x_f32 * x_f32
// Tests that Cast's opaque op output is correctly registered in the tensor_map
// so both inputs of Mul see the same atoms.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// RMS Normalization via symbolic Operation::eval_pool — tests the exact code
// path the ONNX tests use (symbolic op → milli graph → pool_eval).
// ---------------------------------------------------------------------------

fn rms_norm_symbolic_eval_pool() -> TestCase {
    use crate::symbolic_graph::ops::{Operation, RMSNormalizationOperation};

    let mut rng = rng();
    let input_id = GlobalId::new(&mut rng);
    let scale_id = GlobalId::new(&mut rng);
    let output_id = GlobalId::new(&mut rng);

    let op = RMSNormalizationOperation::new(input_id, scale_id, None, output_id, 1e-5, &mut rng);

    let x_data: Vec<f32> = vec![
        1.7640524,
        0.4001572,
        0.978738,
        2.2408931,
        1.867558,
        -0.9772779,
        0.95008844,
        -0.1513572,
        -0.10321885,
        0.41059852,
        0.14404356,
        1.4542735,
    ];
    let scale_data: Vec<f32> = vec![1.2302907, 1.2023798, -0.3873268, -0.30230275];

    // Expected: same as rms_norm_multirow
    let mut expected = vec![0.0f32; 12];
    for row in 0..3 {
        let sq_sum: f32 = (0..4)
            .map(|c| x_data[row * 4 + c] * x_data[row * 4 + c])
            .sum();
        let rms_val = (sq_sum / 4.0 + 1e-5).sqrt();
        for col in 0..4 {
            expected[row * 4 + col] = x_data[row * 4 + col] / rms_val * scale_data[col];
        }
    }

    // We can't use the TestCase/TestDataSet runner here since it expects a MilliOpGraph.
    // Instead, run the symbolic op's eval_pool directly in a test function.
    // Use a dummy TestCase with empty data_sets, and add a custom runner below.
    let (graph, _) = MilliOpGraph::new(vec![input_id, scale_id].into_iter(), &mut rng);
    TestCase {
        name: "rms_norm_symbolic_eval_pool".into(),
        graph,
        data_sets: vec![], // empty — tested via custom test below
    }
}

#[cfg(test)]
fn test_rms_norm_symbolic_eval_pool() {
    use crate::pool::TrackedPool;
    use crate::symbolic_graph::ops::{Operation, RMSNormalizationOperation};

    let mut rng = rng();
    let input_id = GlobalId::new(&mut rng);
    let scale_id = GlobalId::new(&mut rng);
    let output_id = GlobalId::new(&mut rng);

    let op = RMSNormalizationOperation::new(input_id, scale_id, None, output_id, 1e-5, &mut rng);

    let pool = TrackedPool::new(None);

    let x_data: Vec<f32> = vec![
        1.7640524,
        0.4001572,
        0.978738,
        2.2408931,
        1.867558,
        -0.9772779,
        0.95008844,
        -0.1513572,
        -0.10321885,
        0.41059852,
        0.14404356,
        1.4542735,
    ];
    let scale_data: Vec<f32> = vec![1.2302907, 1.2023798, -0.3873268, -0.30230275];

    let x_tensor = super::tensor_f32_shaped(vec![3, 4], &x_data);
    let s_tensor = super::tensor_f32_shaped(vec![4], &scale_data);

    // Test 1: via Operation::eval_pool (the exact ONNX path)
    let input_views: HashMap<GlobalId, _> =
        HashMap::from([(input_id, x_tensor.view()), (scale_id, s_tensor.view())]);
    let input_refs: HashMap<
        GlobalId,
        &crate::numeric_tensor::NumericTensorView<'_, crate::tensor_rank::DynRank>,
    > = input_views.iter().map(|(&id, v)| (id, v)).collect();

    let onnx_inputs: HashMap<GlobalId, crate::numeric_dtype::ONNXTensorView<'_>> = [
        (
            input_id,
            crate::numeric_dtype::ONNXTensorView::Numeric(x_tensor.view()),
        ),
        (
            scale_id,
            crate::numeric_dtype::ONNXTensorView::Numeric(s_tensor.view()),
        ),
    ]
    .into_iter()
    .collect();
    let results = op.eval_pool(&onnx_inputs, &pool).expect("eval_pool failed");
    let out = results
        .get(&output_id)
        .expect("output not found")
        .as_numeric()
        .unwrap();

    // Test 2: via get_milli_op_graph + pool_eval directly
    let tensor_dtypes: HashMap<GlobalId, crate::numeric_dtype::NumericDType> = input_refs
        .iter()
        .map(|(id, view)| (*id, view.dtype()))
        .collect();
    let ctx = crate::milli_graph::MilliLoweringContext::new(tensor_dtypes);
    let mut rng_milli = SmallRng::seed_from_u64(700);
    let milli_graph = op.get_milli_op_graph(&ctx, &mut rng_milli);
    let milli_results = milli_graph
        .pool_eval(&input_refs, &pool)
        .expect("milli pool_eval failed");

    // Compare: milli pool_eval should match eval_pool
    for (id, milli_tensor) in &milli_results {
        eprintln!(
            "milli output {id}: shape={:?} numel={}",
            milli_tensor.shape(),
            milli_tensor.numel()
        );
        for i in 0..milli_tensor.numel() {
            eprintln!("  [{i}] = {}", milli_tensor.read_element(i).to_f64());
        }
    }
    eprintln!("--- eval_pool output ---");
    for i in 0..out.numel() {
        eprintln!("  [{i}] = {}", out.read_element(i).to_f64());
    }

    let mut expected = vec![0.0f32; 12];
    for row in 0..3 {
        let sq_sum: f32 = (0..4)
            .map(|c| x_data[row * 4 + c] * x_data[row * 4 + c])
            .sum();
        let rms_val = (sq_sum / 4.0 + 1e-5).sqrt();
        for col in 0..4 {
            expected[row * 4 + col] = x_data[row * 4 + col] / rms_val * scale_data[col];
        }
    }

    for i in 0..12 {
        let actual = out.read_element(i).to_f64() as f32;
        let exp = expected[i];
        let diff = (actual - exp).abs();
        assert!(
            diff < 1e-4,
            "element {i}: actual {actual} vs expected {exp} (diff {diff})"
        );
    }
}

fn cast_then_self_mul() -> TestCase {
    let mut rng = rng();
    let ext_x = GlobalId::new(&mut rng);
    let (mut graph, imap) = MilliOpGraph::new([ext_x], &mut rng);
    let x = imap[&ext_x];

    let x_f32 = ops::Cast::push_new(&mut graph, x, NumericDType::F32, &mut rng);
    let out = ops::SimpleBinary::mul(&mut graph, x_f32, x_f32, &mut rng);
    graph.set_outputs(vec![out]);

    // x = [[2, 3], [4, 5]]
    // cast is identity, out = x² = [[4, 9], [16, 25]]
    TestCase {
        name: "cast_then_self_mul".into(),
        graph,
        data_sets: vec![TestDataSet {
            label: "2x2".into(),
            inputs: HashMap::from([(ext_x, tensor_f32_shaped(vec![2, 2], &[2.0, 3.0, 4.0, 5.0]))]),
            expected_outputs: HashMap::from([(
                out,
                tensor_f32_shaped(vec![2, 2], &[4.0, 9.0, 16.0, 25.0]),
            )]),
            tolerance: Tolerance {
                atol: 1e-6,
                rtol: 1e-6,
            },
        }],
    }
}
