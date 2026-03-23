//! Elementwise operation test cases (add, sub, mul, neg, exp, etc.)
//!
//! Test data is constructed using the new NumericTensor types.
//! Conversion to legacy types happens at the eval boundary in mod.rs.

use std::collections::HashMap;

use half::{bf16, f16};
use rand::rngs::SmallRng;
use rand::SeedableRng;

use crate::graph::GlobalId;
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::binary::SimpleBinary;
use crate::milli_graph::ops::unary::SimpleUnaryOp;
use crate::numeric_dtype::NumericDType;

use super::{TestCase, TestDataSet, TestTensor, Tolerance};
use super::{tensor_f32, tensor_bf16, tensor_f16};

pub fn build_cases() -> Vec<TestCase> {
    vec![add_case(), mul_case(), neg_case(), exp_case()]
}

// ---------------------------------------------------------------------------
// Graph construction helpers
// ---------------------------------------------------------------------------

fn rng() -> SmallRng {
    SmallRng::seed_from_u64(42)
}

pub(crate) struct BinaryGraphIds {
    pub ext_a: GlobalId,
    pub ext_b: GlobalId,
    pub out: GlobalId,
}

pub(crate) struct UnaryGraphIds {
    pub ext_in: GlobalId,
    pub out: GlobalId,
}

fn build_binary_graph(
    build_op: fn(&mut MilliOpGraph, GlobalId, GlobalId, &mut SmallRng) -> GlobalId,
) -> (MilliOpGraph, BinaryGraphIds) {
    let mut rng = rng();
    let ext_a = GlobalId::new(&mut rng);
    let ext_b = GlobalId::new(&mut rng);
    let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
    let int_a = input_map[&ext_a];
    let int_b = input_map[&ext_b];
    let out = build_op(&mut graph, int_a, int_b, &mut rng);
    graph.set_outputs(vec![out]);
    (graph, BinaryGraphIds { ext_a, ext_b, out })
}

fn build_unary_graph(
    build_op: fn(&mut MilliOpGraph, GlobalId, &mut SmallRng) -> GlobalId,
) -> (MilliOpGraph, UnaryGraphIds) {
    let mut rng = rng();
    let ext_in = GlobalId::new(&mut rng);
    let (mut graph, input_map) = MilliOpGraph::new([ext_in], &mut rng);
    let int_in = input_map[&ext_in];
    let out = build_op(&mut graph, int_in, &mut rng);
    graph.set_outputs(vec![out]);
    (graph, UnaryGraphIds { ext_in, out })
}

// ---------------------------------------------------------------------------
// Data set helpers
// ---------------------------------------------------------------------------

fn binary_data_set(
    label: &str,
    ids: &BinaryGraphIds,
    a: TestTensor,
    b: TestTensor,
    expected: TestTensor,
    tolerance: Tolerance,
) -> TestDataSet {
    TestDataSet {
        label: label.to_string(),
        inputs: HashMap::from([(ids.ext_a, a), (ids.ext_b, b)]),
        expected_outputs: HashMap::from([(ids.out, expected)]),
        tolerance,
    }
}

fn unary_data_set(
    label: &str,
    ids: &UnaryGraphIds,
    input: TestTensor,
    expected: TestTensor,
    tolerance: Tolerance,
) -> TestDataSet {
    TestDataSet {
        label: label.to_string(),
        inputs: HashMap::from([(ids.ext_in, input)]),
        expected_outputs: HashMap::from([(ids.out, expected)]),
        tolerance,
    }
}

// ---------------------------------------------------------------------------
// Add
// ---------------------------------------------------------------------------

fn add_case() -> TestCase {
    let (graph, ids) = build_binary_graph(SimpleBinary::add);

    TestCase {
        name: "add".to_string(),
        graph,
        data_sets: vec![
            binary_data_set(
                "f32_normal", &ids,
                tensor_f32(&[0.15163845, 0.31361532, 5.393808]),
                tensor_f32(&[1.3424649, 0.004955234, 6.920299]),
                tensor_f32(&[1.4941034, 0.31857055, 12.314107]),
                Tolerance::for_dtype(NumericDType::F32),
            ),
            binary_data_set(
                "f32_zeros_negatives", &ids,
                tensor_f32(&[0.0, -1.0, 1e38]),
                tensor_f32(&[0.0, 1.0, -1e38]),
                tensor_f32(&[0.0, 0.0, 0.0]),
                Tolerance::for_dtype(NumericDType::F32),
            ),
            binary_data_set(
                "bf16", &ids,
                tensor_bf16(&[bf16::from_f32(0.75390625), bf16::from_f32(0.93359375), bf16::from_f32(0.13671875)]),
                tensor_bf16(&[bf16::from_f32(4.40625), bf16::from_f32(5.65625), bf16::from_f32(38.25)]),
                tensor_bf16(&[bf16::from_f32(5.15625), bf16::from_f32(6.59375), bf16::from_f32(38.25)]),
                Tolerance::for_dtype(NumericDType::BF16),
            ),
            binary_data_set(
                "f16", &ids,
                tensor_f16(&[f16::from_f32(0.75390625), f16::from_f32(0.93359375), f16::from_f32(0.13671875)]),
                tensor_f16(&[f16::from_f32(4.40625), f16::from_f32(5.65625), f16::from_f32(38.25)]),
                tensor_f16(&[f16::from_f32(5.15625), f16::from_f32(6.59375), f16::from_f32(38.375)]),
                Tolerance::for_dtype(NumericDType::F16),
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// Mul
// ---------------------------------------------------------------------------

fn mul_case() -> TestCase {
    let (graph, ids) = build_binary_graph(SimpleBinary::mul);

    TestCase {
        name: "mul".to_string(),
        graph,
        data_sets: vec![
            binary_data_set(
                "f32_normal", &ids,
                tensor_f32(&[2.0, 3.0, 0.5]),
                tensor_f32(&[4.0, 5.0, 6.0]),
                tensor_f32(&[8.0, 15.0, 3.0]),
                Tolerance::for_dtype(NumericDType::F32),
            ),
            binary_data_set(
                "f32_special", &ids,
                tensor_f32(&[0.0, f32::INFINITY, -1.0]),
                tensor_f32(&[100.0, 2.0, -1.0]),
                tensor_f32(&[0.0, f32::INFINITY, 1.0]),
                Tolerance::for_dtype(NumericDType::F32),
            ),
            binary_data_set(
                "bf16", &ids,
                tensor_bf16(&[bf16::from_f32(2.0), bf16::from_f32(3.0), bf16::from_f32(0.5)]),
                tensor_bf16(&[bf16::from_f32(4.0), bf16::from_f32(5.0), bf16::from_f32(6.0)]),
                tensor_bf16(&[bf16::from_f32(8.0), bf16::from_f32(15.0), bf16::from_f32(3.0)]),
                Tolerance::for_dtype(NumericDType::BF16),
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// Neg
// ---------------------------------------------------------------------------

fn neg_case() -> TestCase {
    let (graph, ids) = build_unary_graph(SimpleUnaryOp::neg);

    TestCase {
        name: "neg".to_string(),
        graph,
        data_sets: vec![
            unary_data_set(
                "f32", &ids,
                tensor_f32(&[1.0, -2.5, 0.0, 3.14]),
                tensor_f32(&[-1.0, 2.5, -0.0, -3.14]),
                Tolerance::for_dtype(NumericDType::F32),
            ),
            unary_data_set(
                "bf16", &ids,
                tensor_bf16(&[bf16::from_f32(1.0), bf16::from_f32(-2.5)]),
                tensor_bf16(&[bf16::from_f32(-1.0), bf16::from_f32(2.5)]),
                Tolerance::for_dtype(NumericDType::BF16),
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// Exp
// ---------------------------------------------------------------------------

fn exp_case() -> TestCase {
    let (graph, ids) = build_unary_graph(SimpleUnaryOp::exp);

    TestCase {
        name: "exp".to_string(),
        graph,
        data_sets: vec![
            unary_data_set(
                "f32", &ids,
                tensor_f32(&[0.0, 1.0, -1.0, 2.0]),
                tensor_f32(&[1.0, std::f32::consts::E, 1.0 / std::f32::consts::E, std::f32::consts::E * std::f32::consts::E]),
                Tolerance::for_dtype(NumericDType::F32),
            ),
        ],
    }
}
