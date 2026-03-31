//! Cast operation test cases.
//!
//! Tests dtype conversion through the MilliOpGraph interpreter.

use std::collections::HashMap;

use half::bf16;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::graph::GlobalId;
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::Cast;
use crate::numeric_dtype::NumericDType;

use super::{TestCase, TestDataSet, TestTensor, Tolerance};
use super::{tensor_bf16, tensor_f32, tensor_from_f64, tensor_i32};

pub fn build_cases() -> Vec<TestCase> {
    vec![
        cast_f32_to_bf16_case(),
        cast_f32_to_i32_case(),
        cast_i32_to_f32_case(),
        cast_f32_to_f8e5m2_saturating(),
        cast_f32_to_f8e5m2_non_saturating(),
        cast_f32_to_f8e4m3fn_saturating(),
    ]
}

// ---------------------------------------------------------------------------
// Graph construction helpers
// ---------------------------------------------------------------------------

fn rng() -> SmallRng {
    SmallRng::seed_from_u64(200)
}

struct CastGraphIds {
    ext_in: GlobalId,
    out: GlobalId,
}

fn build_cast_graph(target_dtype: NumericDType) -> (MilliOpGraph, CastGraphIds) {
    build_cast_graph_with_saturate(target_dtype, true)
}

fn build_cast_graph_with_saturate(
    target_dtype: NumericDType,
    saturate: bool,
) -> (MilliOpGraph, CastGraphIds) {
    let mut rng = rng();
    let ext_in = GlobalId::new(&mut rng);
    let (mut graph, input_map) = MilliOpGraph::new([ext_in], &mut rng);
    let int_in = input_map[&ext_in];
    let out =
        Cast::push_new_with_options(&mut graph, int_in, target_dtype, saturate, None, &mut rng);
    graph.set_outputs(vec![out]);
    (graph, CastGraphIds { ext_in, out })
}

fn cast_data_set(
    label: &str,
    ids: &CastGraphIds,
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
// F32 → BF16
// ---------------------------------------------------------------------------

fn cast_f32_to_bf16_case() -> TestCase {
    let (graph, ids) = build_cast_graph(NumericDType::BF16);

    // Values that are exactly representable in bf16
    // and values that lose precision.
    // bf16 has ~7 bits of mantissa → values like 1.0, 0.5, 2.0 are exact
    // but 0.1 is not.
    TestCase {
        name: "cast_f32_to_bf16".to_string(),
        graph,
        data_sets: vec![
            cast_data_set(
                "exact_values",
                &ids,
                tensor_f32(&[1.0, -2.0, 0.5, 0.0, 128.0]),
                tensor_bf16(&[
                    bf16::from_f32(1.0),
                    bf16::from_f32(-2.0),
                    bf16::from_f32(0.5),
                    bf16::from_f32(0.0),
                    bf16::from_f32(128.0),
                ]),
                Tolerance::for_dtype(NumericDType::BF16),
            ),
            cast_data_set(
                "lossy_values",
                &ids,
                tensor_f32(&[0.1, 3.14, -7.77]),
                tensor_bf16(&[
                    bf16::from_f32(0.1),
                    bf16::from_f32(3.14),
                    bf16::from_f32(-7.77),
                ]),
                Tolerance::for_dtype(NumericDType::BF16),
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// F32 → I32 (truncation)
// ---------------------------------------------------------------------------

fn cast_f32_to_i32_case() -> TestCase {
    let (graph, ids) = build_cast_graph(NumericDType::I32);

    TestCase {
        name: "cast_f32_to_i32".to_string(),
        graph,
        data_sets: vec![
            cast_data_set(
                "truncation",
                &ids,
                tensor_f32(&[3.7, -2.9, 0.0, 1.0, -0.5]),
                // f32→i32 truncates toward zero
                tensor_i32(&[3, -2, 0, 1, 0]),
                Tolerance::for_dtype(NumericDType::I32),
            ),
            cast_data_set(
                "whole_numbers",
                &ids,
                tensor_f32(&[100.0, -50.0, 0.0]),
                tensor_i32(&[100, -50, 0]),
                Tolerance::for_dtype(NumericDType::I32),
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// I32 → F32
// ---------------------------------------------------------------------------

fn cast_i32_to_f32_case() -> TestCase {
    let (graph, ids) = build_cast_graph(NumericDType::F32);

    TestCase {
        name: "cast_i32_to_f32".to_string(),
        graph,
        data_sets: vec![cast_data_set(
            "normal",
            &ids,
            tensor_i32(&[1, -2, 0, 100, -999]),
            tensor_f32(&[1.0, -2.0, 0.0, 100.0, -999.0]),
            Tolerance::for_dtype(NumericDType::F32),
        )],
    }
}

// ---------------------------------------------------------------------------
// F32 → F8E5M2 (saturating — default ONNX behavior)
// ---------------------------------------------------------------------------
//
// F8E5M2: 5 exponent bits, 2 mantissa bits, bias=15, has_infinity=true.
// Max finite = 1.75 × 2^15 = 57344.
// With saturate=true: overflow and ±inf both clamp to ±57344.

fn cast_f32_to_f8e5m2_saturating() -> TestCase {
    let dt = NumericDType::F8E5M2;
    let (graph, ids) = build_cast_graph_with_saturate(dt, true);

    TestCase {
        name: "cast_f32_to_f8e5m2_saturating".to_string(),
        graph,
        data_sets: vec![
            cast_data_set(
                "normal_values",
                &ids,
                tensor_f32(&[0.5, 1.0, -1.0, 0.0]),
                tensor_from_f64(dt, &[0.5, 1.0, -1.0, 0.0]),
                Tolerance::EXACT,
            ),
            cast_data_set(
                "overflow_saturates",
                &ids,
                // 1e6 and inf should both saturate to max finite (57344)
                tensor_f32(&[1e6, f32::INFINITY, -1e6, f32::NEG_INFINITY]),
                tensor_from_f64(dt, &[57344.0, 57344.0, -57344.0, -57344.0]),
                Tolerance::EXACT,
            ),
            cast_data_set(
                "nan_passthrough",
                &ids,
                tensor_f32(&[f32::NAN]),
                tensor_from_f64(dt, &[f64::NAN]),
                Tolerance::EXACT, // NaN == NaN in assert_tensors_close
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// F32 → F8E5M2 (non-saturating)
// ---------------------------------------------------------------------------

fn cast_f32_to_f8e5m2_non_saturating() -> TestCase {
    let dt = NumericDType::F8E5M2;
    let (graph, ids) = build_cast_graph_with_saturate(dt, false);

    TestCase {
        name: "cast_f32_to_f8e5m2_non_saturating".to_string(),
        graph,
        data_sets: vec![
            cast_data_set(
                "overflow_to_inf",
                &ids,
                tensor_f32(&[1e6, f32::INFINITY, -1e6, f32::NEG_INFINITY]),
                tensor_from_f64(dt, &[f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY]),
                Tolerance::EXACT,
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// F32 → F8E4M3FN (saturating)
// ---------------------------------------------------------------------------
//
// F8E4M3FN: 4 exponent bits, 3 mantissa bits, bias=7, has_infinity=false.
// Max finite = 1.875 × 2^8 = 480.
// has_infinity=false means overflow always saturates to 480 regardless of
// the saturate flag, but we test the saturating path for consistency.

fn cast_f32_to_f8e4m3fn_saturating() -> TestCase {
    let dt = NumericDType::F8E4M3FN;
    let (graph, ids) = build_cast_graph_with_saturate(dt, true);

    TestCase {
        name: "cast_f32_to_f8e4m3fn_saturating".to_string(),
        graph,
        data_sets: vec![
            cast_data_set(
                "normal_values",
                &ids,
                tensor_f32(&[0.5, 1.0, -1.0, 0.0, 2.0]),
                tensor_from_f64(dt, &[0.5, 1.0, -1.0, 0.0, 2.0]),
                Tolerance::EXACT,
            ),
            cast_data_set(
                "overflow_saturates",
                &ids,
                // 1000 > 480 → saturates to 480. inf → 480 (no inf in F8E4M3FN).
                tensor_f32(&[1000.0, f32::INFINITY, -1000.0, f32::NEG_INFINITY]),
                tensor_from_f64(dt, &[480.0, 480.0, -480.0, -480.0]),
                Tolerance::EXACT,
            ),
            cast_data_set(
                "nan_passthrough",
                &ids,
                tensor_f32(&[f32::NAN]),
                tensor_from_f64(dt, &[f64::NAN]),
                Tolerance::EXACT,
            ),
        ],
    }
}
