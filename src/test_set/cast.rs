//! Cast operation test cases.
//!
//! Tests dtype conversion through the MilliOpGraph interpreter.

use std::collections::HashMap;

use half::bf16;
use rand::rngs::SmallRng;
use rand::SeedableRng;

use crate::dtype::DType;
use crate::graph::GlobalId;
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::Cast;
use crate::numeric_dtype::NumericDType;

use super::{TestCase, TestDataSet, TestTensor, Tolerance};
use super::{tensor_f32, tensor_i32, tensor_bf16};

pub fn build_cases() -> Vec<TestCase> {
    vec![
        cast_f32_to_bf16_case(),
        cast_f32_to_i32_case(),
        cast_i32_to_f32_case(),
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

fn build_cast_graph(target_dtype: DType) -> (MilliOpGraph, CastGraphIds) {
    let mut rng = rng();
    let ext_in = GlobalId::new(&mut rng);
    let (mut graph, input_map) = MilliOpGraph::new([ext_in], &mut rng);
    let int_in = input_map[&ext_in];
    let out = Cast::push_new(&mut graph, int_in, target_dtype, &mut rng);
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
    let (graph, ids) = build_cast_graph(DType::BF16);

    // Values that are exactly representable in bf16
    // and values that lose precision.
    // bf16 has ~7 bits of mantissa → values like 1.0, 0.5, 2.0 are exact
    // but 0.1 is not.
    TestCase {
        name: "cast_f32_to_bf16".to_string(),
        graph,
        data_sets: vec![
            cast_data_set(
                "exact_values", &ids,
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
                "lossy_values", &ids,
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
    let (graph, ids) = build_cast_graph(DType::I32);

    TestCase {
        name: "cast_f32_to_i32".to_string(),
        graph,
        data_sets: vec![
            cast_data_set(
                "truncation", &ids,
                tensor_f32(&[3.7, -2.9, 0.0, 1.0, -0.5]),
                // f32→i32 truncates toward zero
                tensor_i32(&[3, -2, 0, 1, 0]),
                Tolerance::for_dtype(NumericDType::I32),
            ),
            cast_data_set(
                "whole_numbers", &ids,
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
    let (graph, ids) = build_cast_graph(DType::F32);

    TestCase {
        name: "cast_i32_to_f32".to_string(),
        graph,
        data_sets: vec![
            cast_data_set(
                "normal", &ids,
                tensor_i32(&[1, -2, 0, 100, -999]),
                tensor_f32(&[1.0, -2.0, 0.0, 100.0, -999.0]),
                Tolerance::for_dtype(NumericDType::F32),
            ),
        ],
    }
}
