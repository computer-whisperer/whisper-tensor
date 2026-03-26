//! Composite graph test cases: multi-op patterns from real neural networks.
//!
//! Tests atom map propagation across chains of nano-lowered ops.

use std::collections::HashMap;

use ndarray::{ArcArray, IxDyn};
use rand::rngs::SmallRng;
use rand::SeedableRng;

use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::graph::GlobalId;
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::{self, Constant};
use crate::numeric_dtype::NumericDType;

use super::{TestCase, TestDataSet, Tolerance};
use super::tensor_f32_shaped;

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
    ]
}

fn rng() -> SmallRng {
    SmallRng::seed_from_u64(700)
}

fn f32_const(graph: &mut MilliOpGraph, vals: Vec<f32>, shape: Vec<usize>, rng: &mut SmallRng) -> GlobalId {
    let t = NDArrayNumericTensor::F32(ArcArray::from_shape_vec(IxDyn(&shape), vals).unwrap());
    Constant::push_new(graph, t, rng)
}

fn i64_const(graph: &mut MilliOpGraph, vals: Vec<i64>, rng: &mut SmallRng) -> GlobalId {
    let t = NDArrayNumericTensor::I64(ArcArray::from_shape_vec(IxDyn(&[vals.len()]), vals).unwrap());
    Constant::push_new(graph, t, rng)
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
    let w = f32_const(&mut graph, vec![1.0, -1.0, 0.5, 0.5, -0.5, 1.0], vec![3, 2], &mut rng);
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
            inputs: HashMap::from([(ext_x, tensor_f32_shaped(vec![2, 3], &[1.0, 0.0, -1.0, 2.0, 1.0, 0.0]))]),
            expected_outputs: HashMap::from([(out, tensor_f32_shaped(vec![2, 2], &[1.6, 0.0, 2.6, 0.0]))]),
            tolerance: Tolerance { atol: 1e-5, rtol: 1e-5 },
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
            expected_outputs: HashMap::from([(out, tensor_f32_shaped(vec![1, 3], &[e0/s, e1/s, e2/s]))]),
            tolerance: Tolerance { atol: 1e-5, rtol: 1e-5 },
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
    let v = (c[0]*c[0] + c[1]*c[1] + c[2]*c[2] + c[3]*c[3]) / 4.0;
    let s = (v + 1e-5).sqrt();
    let expected: Vec<f32> = c.iter().map(|&ci| ci / s).collect();

    TestCase {
        name: "layernorm".into(),
        graph,
        data_sets: vec![TestDataSet {
            label: "1x4".into(),
            inputs: HashMap::from([(ext_x, tensor_f32_shaped(vec![1, 4], &[1.0, 2.0, 3.0, 4.0]))]),
            expected_outputs: HashMap::from([(out, tensor_f32_shaped(vec![1, 4], &expected))]),
            tolerance: Tolerance { atol: 1e-4, rtol: 1e-4 },
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
        &mut graph, x, w, None,
        ops::ConvAutoPad::Valid, vec![], 1, vec![1, 1], vec![], vec![1, 1], &mut rng,
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
            inputs: HashMap::from([(ext_x, tensor_f32_shaped(vec![1, 1, 2, 2], &[1.0, -1.0, 2.0, -2.0]))]),
            expected_outputs: HashMap::from([(out, tensor_f32_shaped(vec![1, 1, 2, 2], &[3.0, 0.0, 6.0, 0.0]))]),
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
    let qk = ops::MatMul::push_new_default_precision(&mut graph, imap[&ext_q], k_t, NumericDType::F32, &mut rng);

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
                (ext_q, tensor_f32_shaped(vec![1, 2, 3], &[1.0,0.0,0.0, 0.0,1.0,0.0])),
                (ext_k, tensor_f32_shaped(vec![1, 2, 3], &[1.0,0.0,0.0, 0.0,0.0,1.0])),
            ]),
            expected_outputs: HashMap::from([(out, tensor_f32_shaped(vec![1, 2, 2], &[s, 0.0, 0.0, 0.0]))]),
            tolerance: Tolerance { atol: 1e-5, rtol: 1e-5 },
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
    let sqrt2 = f32_const(&mut graph, vec![std::f32::consts::SQRT_2], vec![1], &mut rng);
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
        let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
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
            expected_outputs: HashMap::from([(out, tensor_f32_shaped(vec![4], &[
                gelu(-1.0), gelu(0.0), gelu(1.0), gelu(2.0),
            ]))]),
            tolerance: Tolerance { atol: 1e-4, rtol: 1e-4 },
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
    let w = f32_const(&mut graph, vec![
        0.0, 1.0, 0.0,
        1.0,-4.0, 1.0,
        0.0, 1.0, 0.0,
    ], vec![1, 1, 3, 3], &mut rng);
    let b = f32_const(&mut graph, vec![1.0], vec![1], &mut rng);

    let conv = ops::Conv::push_new(
        &mut graph, x, w, Some(b),
        ops::ConvAutoPad::Valid, vec![], 1, vec![3, 3], vec![], vec![1, 1], &mut rng,
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
            expected_outputs: HashMap::from([(out, tensor_f32_shaped(vec![1, 1, 2, 2], &[1.0, 1.0, 1.0, 1.0]))]),
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
    let table = f32_const(&mut graph, vec![
        1.0, 0.0, 0.0,   // token 0
        0.0, 2.0, 0.0,   // token 1
        0.0, 0.0, 3.0,   // token 2
        1.0, 1.0, 1.0,   // token 3
    ], vec![4, 3], &mut rng);

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
    let v0 = (c0[0]*c0[0] + c0[1]*c0[1] + c0[2]*c0[2]) / 3.0;
    let s0 = (v0 + 1e-5).sqrt();

    TestCase {
        name: "gather_layernorm".into(),
        graph,
        data_sets: vec![TestDataSet {
            label: "2_tokens".into(),
            inputs: HashMap::from([(ext_ids, super::tensor_i64_shaped(vec![2], &[3, 0]))]),
            expected_outputs: HashMap::from([(out, tensor_f32_shaped(vec![2, 3], &[
                0.0, 0.0, 0.0,
                c0[0]/s0, c0[1]/s0, c0[2]/s0,
            ]))]),
            tolerance: Tolerance { atol: 1e-3, rtol: 1e-3 },
        }],
    }
}
