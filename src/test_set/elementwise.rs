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
use super::{tensor_f32, tensor_bf16, tensor_f16, tensor_bool};

pub fn build_cases() -> Vec<TestCase> {
    vec![
        add_case(),
        mul_case(),
        neg_case(),
        exp_case(),
        sub_case(),
        div_case(),
        abs_case(),
        sqrt_case(),
        ln_case(),
        floor_case(),
        ceil_case(),
        reciprocal_case(),
        max_case(),
        min_case(),
        equal_case(),
        greater_case(),
        less_case(),
        round_case(),
        sign_case(),
        is_nan_case(),
        erf_case(),
        sin_case(),
        cos_case(),
        is_inf_case(),
        log1p_case(),
        tan_case(),
        asin_case(),
        acos_case(),
        atan_case(),
        sinh_case(),
        cosh_case(),
    ]
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

// ---------------------------------------------------------------------------
// Sub
// ---------------------------------------------------------------------------

fn sub_case() -> TestCase {
    let (graph, ids) = build_binary_graph(SimpleBinary::sub);

    TestCase {
        name: "sub".to_string(),
        graph,
        data_sets: vec![
            binary_data_set(
                "f32", &ids,
                tensor_f32(&[5.0, 3.0, -1.0, 0.0]),
                tensor_f32(&[2.0, 3.0, -4.0, 7.5]),
                tensor_f32(&[3.0, 0.0, 3.0, -7.5]),
                Tolerance::for_dtype(NumericDType::F32),
            ),
            binary_data_set(
                "bf16", &ids,
                tensor_bf16(&[bf16::from_f32(10.0), bf16::from_f32(1.0), bf16::from_f32(0.5)]),
                tensor_bf16(&[bf16::from_f32(3.0), bf16::from_f32(1.0), bf16::from_f32(0.25)]),
                tensor_bf16(&[bf16::from_f32(7.0), bf16::from_f32(0.0), bf16::from_f32(0.25)]),
                Tolerance::for_dtype(NumericDType::BF16),
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// Div
// ---------------------------------------------------------------------------

fn div_case() -> TestCase {
    let (graph, ids) = build_binary_graph(SimpleBinary::div);

    TestCase {
        name: "div".to_string(),
        graph,
        data_sets: vec![
            binary_data_set(
                "f32", &ids,
                tensor_f32(&[6.0, 1.0, -10.0, 1.0]),
                tensor_f32(&[3.0, 4.0, 5.0, 1e-30]),
                tensor_f32(&[2.0, 0.25, -2.0, 1e30]),
                Tolerance::for_dtype(NumericDType::F32),
            ),
            binary_data_set(
                "bf16", &ids,
                tensor_bf16(&[bf16::from_f32(8.0), bf16::from_f32(1.0), bf16::from_f32(3.0)]),
                tensor_bf16(&[bf16::from_f32(2.0), bf16::from_f32(4.0), bf16::from_f32(3.0)]),
                tensor_bf16(&[bf16::from_f32(4.0), bf16::from_f32(0.25), bf16::from_f32(1.0)]),
                Tolerance::for_dtype(NumericDType::BF16),
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// Abs
// ---------------------------------------------------------------------------

fn abs_case() -> TestCase {
    let (graph, ids) = build_unary_graph(SimpleUnaryOp::abs);

    TestCase {
        name: "abs".to_string(),
        graph,
        data_sets: vec![
            unary_data_set(
                "f32", &ids,
                tensor_f32(&[-3.5, 0.0, 2.0, -0.0, -100.0]),
                tensor_f32(&[3.5, 0.0, 2.0, 0.0, 100.0]),
                Tolerance::for_dtype(NumericDType::F32),
            ),
            unary_data_set(
                "bf16", &ids,
                tensor_bf16(&[bf16::from_f32(-2.0), bf16::from_f32(0.0), bf16::from_f32(5.0)]),
                tensor_bf16(&[bf16::from_f32(2.0), bf16::from_f32(0.0), bf16::from_f32(5.0)]),
                Tolerance::for_dtype(NumericDType::BF16),
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// Sqrt
// ---------------------------------------------------------------------------

fn sqrt_case() -> TestCase {
    let (graph, ids) = build_unary_graph(SimpleUnaryOp::sqrt);

    TestCase {
        name: "sqrt".to_string(),
        graph,
        data_sets: vec![
            unary_data_set(
                "f32", &ids,
                tensor_f32(&[0.0, 1.0, 4.0, 9.0, 16.0]),
                tensor_f32(&[0.0, 1.0, 2.0, 3.0, 4.0]),
                Tolerance::for_dtype(NumericDType::F32),
            ),
            unary_data_set(
                "f32_non_perfect", &ids,
                tensor_f32(&[2.0, 0.25]),
                tensor_f32(&[std::f32::consts::SQRT_2, 0.5]),
                Tolerance::for_dtype(NumericDType::F32),
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// Ln
// ---------------------------------------------------------------------------

fn ln_case() -> TestCase {
    let (graph, ids) = build_unary_graph(SimpleUnaryOp::ln);

    TestCase {
        name: "ln".to_string(),
        graph,
        data_sets: vec![
            unary_data_set(
                "f32", &ids,
                tensor_f32(&[1.0, std::f32::consts::E, 0.5, 10.0]),
                tensor_f32(&[0.0, 1.0, -std::f32::consts::LN_2, std::f32::consts::LN_10]),
                Tolerance::for_dtype(NumericDType::F32),
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// Floor
// ---------------------------------------------------------------------------

fn floor_case() -> TestCase {
    let (graph, ids) = build_unary_graph(SimpleUnaryOp::floor);

    TestCase {
        name: "floor".to_string(),
        graph,
        data_sets: vec![
            unary_data_set(
                "f32", &ids,
                tensor_f32(&[3.7, -3.7, 0.0, 2.0, -0.1]),
                tensor_f32(&[3.0, -4.0, 0.0, 2.0, -1.0]),
                Tolerance::for_dtype(NumericDType::F32),
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// Ceil
// ---------------------------------------------------------------------------

fn ceil_case() -> TestCase {
    let (graph, ids) = build_unary_graph(SimpleUnaryOp::ceil);

    TestCase {
        name: "ceil".to_string(),
        graph,
        data_sets: vec![
            unary_data_set(
                "f32", &ids,
                tensor_f32(&[3.2, -3.7, 0.0, 2.0, -0.1]),
                tensor_f32(&[4.0, -3.0, 0.0, 2.0, 0.0]),
                Tolerance::for_dtype(NumericDType::F32),
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// Reciprocal
// ---------------------------------------------------------------------------

fn reciprocal_case() -> TestCase {
    let (graph, ids) = build_unary_graph(SimpleUnaryOp::reciprocal);

    TestCase {
        name: "reciprocal".to_string(),
        graph,
        data_sets: vec![
            unary_data_set(
                "f32", &ids,
                tensor_f32(&[2.0, 0.5, -1.0, 4.0, 0.25]),
                tensor_f32(&[0.5, 2.0, -1.0, 0.25, 4.0]),
                Tolerance::for_dtype(NumericDType::F32),
            ),
            unary_data_set(
                "bf16", &ids,
                tensor_bf16(&[bf16::from_f32(2.0), bf16::from_f32(0.5), bf16::from_f32(-1.0)]),
                tensor_bf16(&[bf16::from_f32(0.5), bf16::from_f32(2.0), bf16::from_f32(-1.0)]),
                Tolerance::for_dtype(NumericDType::BF16),
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// Max (binary)
// ---------------------------------------------------------------------------

fn max_case() -> TestCase {
    let (graph, ids) = build_binary_graph(SimpleBinary::max);

    TestCase {
        name: "max".to_string(),
        graph,
        data_sets: vec![
            binary_data_set(
                "f32_mixed", &ids,
                tensor_f32(&[1.0, 5.0, -3.0, 0.0, 7.0]),
                tensor_f32(&[2.0, 5.0, -1.0, 0.0, -7.0]),
                tensor_f32(&[2.0, 5.0, -1.0, 0.0, 7.0]),
                Tolerance::for_dtype(NumericDType::F32),
            ),
            binary_data_set(
                "f32_negative", &ids,
                tensor_f32(&[-10.0, -1.0, -100.0]),
                tensor_f32(&[-5.0, -2.0, -50.0]),
                tensor_f32(&[-5.0, -1.0, -50.0]),
                Tolerance::for_dtype(NumericDType::F32),
            ),
            binary_data_set(
                "bf16", &ids,
                tensor_bf16(&[bf16::from_f32(1.0), bf16::from_f32(-3.0), bf16::from_f32(0.0)]),
                tensor_bf16(&[bf16::from_f32(2.0), bf16::from_f32(-1.0), bf16::from_f32(0.0)]),
                tensor_bf16(&[bf16::from_f32(2.0), bf16::from_f32(-1.0), bf16::from_f32(0.0)]),
                Tolerance::for_dtype(NumericDType::BF16),
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// Min (binary)
// ---------------------------------------------------------------------------

fn min_case() -> TestCase {
    let (graph, ids) = build_binary_graph(SimpleBinary::min);

    TestCase {
        name: "min".to_string(),
        graph,
        data_sets: vec![
            binary_data_set(
                "f32_mixed", &ids,
                tensor_f32(&[1.0, 5.0, -3.0, 0.0, 7.0]),
                tensor_f32(&[2.0, 5.0, -1.0, 0.0, -7.0]),
                tensor_f32(&[1.0, 5.0, -3.0, 0.0, -7.0]),
                Tolerance::for_dtype(NumericDType::F32),
            ),
            binary_data_set(
                "f32_positive", &ids,
                tensor_f32(&[10.0, 1.0, 100.0]),
                tensor_f32(&[5.0, 2.0, 50.0]),
                tensor_f32(&[5.0, 1.0, 50.0]),
                Tolerance::for_dtype(NumericDType::F32),
            ),
            binary_data_set(
                "bf16", &ids,
                tensor_bf16(&[bf16::from_f32(1.0), bf16::from_f32(-3.0), bf16::from_f32(5.0)]),
                tensor_bf16(&[bf16::from_f32(2.0), bf16::from_f32(-1.0), bf16::from_f32(3.0)]),
                tensor_bf16(&[bf16::from_f32(1.0), bf16::from_f32(-3.0), bf16::from_f32(3.0)]),
                Tolerance::for_dtype(NumericDType::BF16),
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// Equal (comparison → BOOL output)
// ---------------------------------------------------------------------------

fn equal_case() -> TestCase {
    let (graph, ids) = build_binary_graph(SimpleBinary::equal);

    TestCase {
        name: "equal".to_string(),
        graph,
        data_sets: vec![
            binary_data_set(
                "f32_mixed", &ids,
                tensor_f32(&[1.0, 2.0, 3.0, 0.0, -1.0]),
                tensor_f32(&[1.0, 3.0, 3.0, -0.0, -1.0]),
                // 1==1→T, 2==3→F, 3==3→T, 0==-0→T, -1==-1→T
                tensor_bool(&[true, false, true, true, true]),
                Tolerance::for_dtype(NumericDType::BOOL),
            ),
            binary_data_set(
                "f32_all_different", &ids,
                tensor_f32(&[1.0, 2.0, 3.0]),
                tensor_f32(&[4.0, 5.0, 6.0]),
                tensor_bool(&[false, false, false]),
                Tolerance::for_dtype(NumericDType::BOOL),
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// Greater (comparison → BOOL output)
// ---------------------------------------------------------------------------

fn greater_case() -> TestCase {
    let (graph, ids) = build_binary_graph(SimpleBinary::greater);

    TestCase {
        name: "greater".to_string(),
        graph,
        data_sets: vec![
            binary_data_set(
                "f32_mixed", &ids,
                tensor_f32(&[5.0, 2.0, 3.0, 0.0, -1.0]),
                tensor_f32(&[1.0, 3.0, 3.0, 0.0, -2.0]),
                // 5>1→T, 2>3→F, 3>3→F, 0>0→F, -1>-2→T
                tensor_bool(&[true, false, false, false, true]),
                Tolerance::for_dtype(NumericDType::BOOL),
            ),
            binary_data_set(
                "f32_negative", &ids,
                tensor_f32(&[-1.0, -5.0, 0.0]),
                tensor_f32(&[-2.0, -3.0, 0.0]),
                // -1>-2→T, -5>-3→F, 0>0→F
                tensor_bool(&[true, false, false]),
                Tolerance::for_dtype(NumericDType::BOOL),
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// Less (comparison → BOOL output)
// ---------------------------------------------------------------------------

fn less_case() -> TestCase {
    let (graph, ids) = build_binary_graph(SimpleBinary::less);

    TestCase {
        name: "less".to_string(),
        graph,
        data_sets: vec![
            binary_data_set(
                "f32_mixed", &ids,
                tensor_f32(&[1.0, 3.0, 3.0, 0.0, -2.0]),
                tensor_f32(&[5.0, 2.0, 3.0, 0.0, -1.0]),
                // 1<5→T, 3<2→F, 3<3→F, 0<0→F, -2<-1→T
                tensor_bool(&[true, false, false, false, true]),
                Tolerance::for_dtype(NumericDType::BOOL),
            ),
            binary_data_set(
                "f32_all_less", &ids,
                tensor_f32(&[1.0, 2.0, 3.0]),
                tensor_f32(&[4.0, 5.0, 6.0]),
                tensor_bool(&[true, true, true]),
                Tolerance::for_dtype(NumericDType::BOOL),
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// Round
// ---------------------------------------------------------------------------

fn round_case() -> TestCase {
    let (graph, ids) = build_unary_graph(SimpleUnaryOp::round);

    TestCase {
        name: "round".to_string(),
        graph,
        data_sets: vec![
            unary_data_set(
                "f32", &ids,
                tensor_f32(&[3.5, 2.5, -1.7, 0.0, 4.5, -0.5]),
                // ONNX uses banker's rounding (round half to even):
                // 3.5→4.0, 2.5→2.0, -1.7→-2.0, 0.0→0.0, 4.5→4.0, -0.5→0.0
                tensor_f32(&[4.0, 2.0, -2.0, 0.0, 4.0, 0.0]),
                Tolerance::for_dtype(NumericDType::F32),
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// Sign
// ---------------------------------------------------------------------------

fn sign_case() -> TestCase {
    let (graph, ids) = build_unary_graph(SimpleUnaryOp::sign);

    TestCase {
        name: "sign".to_string(),
        graph,
        data_sets: vec![
            unary_data_set(
                "f32", &ids,
                tensor_f32(&[3.14, -2.0, 0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY]),
                tensor_f32(&[1.0, -1.0, 0.0, 0.0, 1.0, -1.0]),
                Tolerance::for_dtype(NumericDType::F32),
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// IsNan
// ---------------------------------------------------------------------------

fn is_nan_case() -> TestCase {
    let (graph, ids) = build_unary_graph(SimpleUnaryOp::is_nan);

    TestCase {
        name: "is_nan".to_string(),
        graph,
        data_sets: vec![
            unary_data_set(
                "f32", &ids,
                tensor_f32(&[1.0, f32::NAN, 0.0, f32::INFINITY, f32::NAN]),
                tensor_bool(&[false, true, false, false, true]),
                Tolerance::for_dtype(NumericDType::BOOL),
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// Erf
// ---------------------------------------------------------------------------

fn erf_case() -> TestCase {
    let (graph, ids) = build_unary_graph(SimpleUnaryOp::erf);

    TestCase {
        name: "erf".to_string(),
        graph,
        data_sets: vec![
            unary_data_set(
                "f32", &ids,
                tensor_f32(&[0.0, 1.0, -1.0, 3.0]),
                // erf(0)=0, erf(1)≈0.8427, erf(-1)≈-0.8427, erf(3)≈0.9999
                tensor_f32(&[0.0, 0.8427008, -0.8427008, 0.9999779]),
                Tolerance { atol: 2e-4, rtol: 1e-3 },
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// Sin
// ---------------------------------------------------------------------------

fn sin_case() -> TestCase {
    let (graph, ids) = build_unary_graph(|g, input, rng| {
        SimpleUnaryOp::trig(g, input, crate::TrigOp::Sin, rng)
    });

    TestCase {
        name: "sin".to_string(),
        graph,
        data_sets: vec![
            unary_data_set(
                "f32", &ids,
                tensor_f32(&[0.0, std::f32::consts::FRAC_PI_2, std::f32::consts::PI, -std::f32::consts::FRAC_PI_2]),
                tensor_f32(&[0.0, 1.0, 0.0, -1.0]),
                Tolerance { atol: 1e-5, rtol: 1e-5 },
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// Cos
// ---------------------------------------------------------------------------

fn cos_case() -> TestCase {
    let (graph, ids) = build_unary_graph(|g, input, rng| {
        SimpleUnaryOp::trig(g, input, crate::TrigOp::Cos, rng)
    });

    TestCase {
        name: "cos".to_string(),
        graph,
        data_sets: vec![
            unary_data_set(
                "f32", &ids,
                tensor_f32(&[0.0, std::f32::consts::FRAC_PI_2, std::f32::consts::PI]),
                tensor_f32(&[1.0, 0.0, -1.0]),
                Tolerance { atol: 1e-5, rtol: 1e-5 },
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// IsInf
// ---------------------------------------------------------------------------

fn is_inf_case() -> TestCase {
    let (graph, ids) = build_unary_graph(|g, input, rng| {
        SimpleUnaryOp::is_inf(g, input, true, true, rng)
    });

    TestCase {
        name: "is_inf".to_string(),
        graph,
        data_sets: vec![
            unary_data_set(
                "f32", &ids,
                tensor_f32(&[1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 0.0]),
                tensor_bool(&[false, true, true, false, false]),
                Tolerance::for_dtype(NumericDType::BOOL),
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// Log1p
// ---------------------------------------------------------------------------

fn log1p_case() -> TestCase {
    let (graph, ids) = build_unary_graph(SimpleUnaryOp::log1p);

    TestCase {
        name: "log1p".to_string(),
        graph,
        data_sets: vec![
            unary_data_set(
                "f32", &ids,
                tensor_f32(&[0.0, 1.0, -0.5]),
                // ln(1+0)=0, ln(1+1)=ln(2), ln(1-0.5)=ln(0.5)
                tensor_f32(&[0.0, std::f32::consts::LN_2, -(std::f32::consts::LN_2)]),
                Tolerance { atol: 1e-5, rtol: 1e-5 },
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// Tan
// ---------------------------------------------------------------------------

fn tan_case() -> TestCase {
    let (graph, ids) = build_unary_graph(|g, input, rng| {
        SimpleUnaryOp::trig(g, input, crate::TrigOp::Tan, rng)
    });

    TestCase {
        name: "tan".to_string(),
        graph,
        data_sets: vec![
            unary_data_set(
                "f32", &ids,
                tensor_f32(&[0.0, std::f32::consts::FRAC_PI_4]),
                tensor_f32(&[0.0, 1.0]),
                Tolerance { atol: 1e-5, rtol: 1e-5 },
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// Asin
// ---------------------------------------------------------------------------

fn asin_case() -> TestCase {
    let (graph, ids) = build_unary_graph(|g, input, rng| {
        SimpleUnaryOp::trig(g, input, crate::TrigOp::Asin, rng)
    });

    TestCase {
        name: "asin".to_string(),
        graph,
        data_sets: vec![
            unary_data_set(
                "f32", &ids,
                tensor_f32(&[0.0, 1.0]),
                tensor_f32(&[0.0, std::f32::consts::FRAC_PI_2]),
                Tolerance { atol: 1e-5, rtol: 1e-5 },
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// Acos
// ---------------------------------------------------------------------------

fn acos_case() -> TestCase {
    let (graph, ids) = build_unary_graph(|g, input, rng| {
        SimpleUnaryOp::trig(g, input, crate::TrigOp::Acos, rng)
    });

    TestCase {
        name: "acos".to_string(),
        graph,
        data_sets: vec![
            unary_data_set(
                "f32", &ids,
                tensor_f32(&[1.0, 0.0]),
                tensor_f32(&[0.0, std::f32::consts::FRAC_PI_2]),
                Tolerance { atol: 1e-5, rtol: 1e-5 },
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// Atan
// ---------------------------------------------------------------------------

fn atan_case() -> TestCase {
    let (graph, ids) = build_unary_graph(|g, input, rng| {
        SimpleUnaryOp::trig(g, input, crate::TrigOp::Atan, rng)
    });

    TestCase {
        name: "atan".to_string(),
        graph,
        data_sets: vec![
            unary_data_set(
                "f32", &ids,
                tensor_f32(&[0.0, 1.0]),
                tensor_f32(&[0.0, std::f32::consts::FRAC_PI_4]),
                Tolerance { atol: 1e-5, rtol: 1e-5 },
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// Sinh
// ---------------------------------------------------------------------------

fn sinh_case() -> TestCase {
    let (graph, ids) = build_unary_graph(|g, input, rng| {
        SimpleUnaryOp::trig(g, input, crate::TrigOp::Sinh, rng)
    });

    TestCase {
        name: "sinh".to_string(),
        graph,
        data_sets: vec![
            unary_data_set(
                "f32", &ids,
                tensor_f32(&[0.0, 1.0]),
                // sinh(0)=0, sinh(1)≈1.1752012
                tensor_f32(&[0.0, 1.1752012]),
                Tolerance { atol: 1e-5, rtol: 1e-5 },
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// Cosh
// ---------------------------------------------------------------------------

fn cosh_case() -> TestCase {
    let (graph, ids) = build_unary_graph(|g, input, rng| {
        SimpleUnaryOp::trig(g, input, crate::TrigOp::Cosh, rng)
    });

    TestCase {
        name: "cosh".to_string(),
        graph,
        data_sets: vec![
            unary_data_set(
                "f32", &ids,
                tensor_f32(&[0.0, 1.0]),
                // cosh(0)=1, cosh(1)≈1.5430806
                tensor_f32(&[1.0, 1.5430806]),
                Tolerance { atol: 1e-5, rtol: 1e-5 },
            ),
        ],
    }
}
