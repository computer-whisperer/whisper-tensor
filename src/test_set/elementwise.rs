//! Elementwise operation test cases (add, sub, mul, neg, exp, etc.)

use std::collections::HashMap;

use half::{bf16, f16};
use rand::rngs::SmallRng;
use rand::SeedableRng;

use crate::dtype::DType;
use crate::graph::GlobalId;
use crate::migration::numeric_tensor::NumericTensor;
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::binary::SimpleBinary;
use crate::milli_graph::ops::unary::SimpleUnaryOp;
use crate::DynRank;

use super::{TestCase, Tolerance};

/// Build all elementwise test cases.
pub fn build_cases() -> Vec<TestCase> {
    let mut cases = Vec::new();
    cases.extend(add_cases());
    cases.extend(mul_cases());
    cases.extend(neg_cases());
    cases.extend(exp_cases());
    cases
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn rng() -> SmallRng {
    SmallRng::seed_from_u64(42)
}

/// Build a binary op test case: graph with one binary op, two 1D inputs, one output.
fn binary_test(
    name: &str,
    dtype: DType,
    a_data: NumericTensor<DynRank>,
    b_data: NumericTensor<DynRank>,
    expected: NumericTensor<DynRank>,
    build_op: fn(&mut MilliOpGraph, GlobalId, GlobalId, &mut SmallRng) -> GlobalId,
) -> TestCase {
    let mut rng = rng();
    let ext_a = GlobalId::new(&mut rng);
    let ext_b = GlobalId::new(&mut rng);

    let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
    let int_a = input_map[&ext_a];
    let int_b = input_map[&ext_b];
    let out = build_op(&mut graph, int_a, int_b, &mut rng);
    graph.set_outputs(vec![out]);

    TestCase {
        name: name.to_string(),
        graph,
        inputs: HashMap::from([(ext_a, a_data), (ext_b, b_data)]),
        expected_outputs: HashMap::from([(out, expected)]),
        tolerance: Tolerance::for_dtype(dtype),
    }
}

/// Build a unary op test case: graph with one unary op, one 1D input, one output.
fn unary_test(
    name: &str,
    dtype: DType,
    input_data: NumericTensor<DynRank>,
    expected: NumericTensor<DynRank>,
    build_op: fn(&mut MilliOpGraph, GlobalId, &mut SmallRng) -> GlobalId,
) -> TestCase {
    let mut rng = rng();
    let ext_in = GlobalId::new(&mut rng);

    let (mut graph, input_map) = MilliOpGraph::new([ext_in], &mut rng);
    let int_in = input_map[&ext_in];
    let out = build_op(&mut graph, int_in, &mut rng);
    graph.set_outputs(vec![out]);

    TestCase {
        name: name.to_string(),
        graph,
        inputs: HashMap::from([(ext_in, input_data)]),
        expected_outputs: HashMap::from([(out, expected)]),
        tolerance: Tolerance::for_dtype(dtype),
    }
}

// ---------------------------------------------------------------------------
// Add
// ---------------------------------------------------------------------------

fn add_cases() -> Vec<TestCase> {
    vec![
        binary_test(
            "add_f32",
            DType::F32,
            NumericTensor::from_vec(vec![0.15163845f32, 0.31361532, 5.393808]).to_dyn_rank(),
            NumericTensor::from_vec(vec![1.3424649f32, 0.004955234, 6.920299]).to_dyn_rank(),
            NumericTensor::from_vec(vec![1.4941034f32, 0.31857055, 12.314107]).to_dyn_rank(),
            SimpleBinary::add,
        ),
        binary_test(
            "add_bf16",
            DType::BF16,
            NumericTensor::from_vec(vec![
                bf16::from_f32(0.75390625),
                bf16::from_f32(0.93359375),
                bf16::from_f32(0.13671875),
            ])
            .to_dyn_rank(),
            NumericTensor::from_vec(vec![
                bf16::from_f32(4.40625),
                bf16::from_f32(5.65625),
                bf16::from_f32(38.25),
            ])
            .to_dyn_rank(),
            NumericTensor::from_vec(vec![
                bf16::from_f32(5.15625),
                bf16::from_f32(6.59375),
                bf16::from_f32(38.25),
            ])
            .to_dyn_rank(),
            SimpleBinary::add,
        ),
        binary_test(
            "add_f16",
            DType::F16,
            NumericTensor::from_vec(vec![
                f16::from_f32(0.75390625),
                f16::from_f32(0.93359375),
                f16::from_f32(0.13671875),
            ])
            .to_dyn_rank(),
            NumericTensor::from_vec(vec![
                f16::from_f32(4.40625),
                f16::from_f32(5.65625),
                f16::from_f32(38.25),
            ])
            .to_dyn_rank(),
            NumericTensor::from_vec(vec![
                f16::from_f32(5.15625),
                f16::from_f32(6.59375),
                f16::from_f32(38.375),
            ])
            .to_dyn_rank(),
            SimpleBinary::add,
        ),
    ]
}

// ---------------------------------------------------------------------------
// Mul
// ---------------------------------------------------------------------------

fn mul_cases() -> Vec<TestCase> {
    vec![
        binary_test(
            "mul_f32",
            DType::F32,
            NumericTensor::from_vec(vec![2.0f32, 3.0, 0.5]).to_dyn_rank(),
            NumericTensor::from_vec(vec![4.0f32, 5.0, 6.0]).to_dyn_rank(),
            NumericTensor::from_vec(vec![8.0f32, 15.0, 3.0]).to_dyn_rank(),
            SimpleBinary::mul,
        ),
        binary_test(
            "mul_bf16",
            DType::BF16,
            NumericTensor::from_vec(vec![
                bf16::from_f32(2.0),
                bf16::from_f32(3.0),
                bf16::from_f32(0.5),
            ])
            .to_dyn_rank(),
            NumericTensor::from_vec(vec![
                bf16::from_f32(4.0),
                bf16::from_f32(5.0),
                bf16::from_f32(6.0),
            ])
            .to_dyn_rank(),
            NumericTensor::from_vec(vec![
                bf16::from_f32(8.0),
                bf16::from_f32(15.0),
                bf16::from_f32(3.0),
            ])
            .to_dyn_rank(),
            SimpleBinary::mul,
        ),
    ]
}

// ---------------------------------------------------------------------------
// Neg
// ---------------------------------------------------------------------------

fn neg_cases() -> Vec<TestCase> {
    vec![
        unary_test(
            "neg_f32",
            DType::F32,
            NumericTensor::from_vec(vec![1.0f32, -2.5, 0.0]).to_dyn_rank(),
            NumericTensor::from_vec(vec![-1.0f32, 2.5, -0.0]).to_dyn_rank(),
            SimpleUnaryOp::neg,
        ),
        unary_test(
            "neg_bf16",
            DType::BF16,
            NumericTensor::from_vec(vec![
                bf16::from_f32(1.0),
                bf16::from_f32(-2.5),
            ])
            .to_dyn_rank(),
            NumericTensor::from_vec(vec![
                bf16::from_f32(-1.0),
                bf16::from_f32(2.5),
            ])
            .to_dyn_rank(),
            SimpleUnaryOp::neg,
        ),
    ]
}

// ---------------------------------------------------------------------------
// Exp
// ---------------------------------------------------------------------------

fn exp_cases() -> Vec<TestCase> {
    vec![unary_test(
        "exp_f32",
        DType::F32,
        NumericTensor::from_vec(vec![0.0f32, 1.0, -1.0, 2.0]).to_dyn_rank(),
        NumericTensor::from_vec(vec![
            1.0f32,
            std::f32::consts::E,
            1.0 / std::f32::consts::E,
            std::f32::consts::E * std::f32::consts::E,
        ])
        .to_dyn_rank(),
        SimpleUnaryOp::exp,
    )]
}
