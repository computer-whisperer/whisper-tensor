//! MatMul test cases.
//!
//! Tests matrix multiplication through the MilliOpGraph interpreter.

use std::collections::HashMap;

use rand::rngs::SmallRng;
use rand::SeedableRng;

use crate::graph::GlobalId;
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::MatMul;
use crate::numeric_dtype::NumericDType;

use super::{TestCase, TestDataSet, Tolerance};
use super::tensor_f32_shaped;

pub fn build_cases() -> Vec<TestCase> {
    vec![matmul_case()]
}

// ---------------------------------------------------------------------------
// Graph construction
// ---------------------------------------------------------------------------

fn rng() -> SmallRng {
    SmallRng::seed_from_u64(99)
}

struct MatMulGraphIds {
    ext_a: GlobalId,
    ext_b: GlobalId,
    out: GlobalId,
}

fn build_matmul_graph() -> (MilliOpGraph, MatMulGraphIds) {
    let mut rng = rng();
    let ext_a = GlobalId::new(&mut rng);
    let ext_b = GlobalId::new(&mut rng);
    let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
    let int_a = input_map[&ext_a];
    let int_b = input_map[&ext_b];
    let out = MatMul::push_new(
        &mut graph,
        int_a,
        int_b,
        NumericDType::F32,
        NumericDType::F32,
        NumericDType::F32,
        NumericDType::F32,
        &mut rng,
    );
    graph.set_outputs(vec![out]);
    (graph, MatMulGraphIds { ext_a, ext_b, out })
}

fn matmul_data_set(
    label: &str,
    ids: &MatMulGraphIds,
    a: super::TestTensor,
    b: super::TestTensor,
    expected: super::TestTensor,
    tolerance: Tolerance,
) -> TestDataSet {
    TestDataSet {
        label: label.to_string(),
        inputs: HashMap::from([(ids.ext_a, a), (ids.ext_b, b)]),
        expected_outputs: HashMap::from([(ids.out, expected)]),
        tolerance,
    }
}

// ---------------------------------------------------------------------------
// MatMul
// ---------------------------------------------------------------------------

fn matmul_case() -> TestCase {
    let (graph, ids) = build_matmul_graph();

    // 2x3 @ 3x4 → 2x4
    //
    // A = [[1, 2, 3],    B = [[1, 0, 2, 1],
    //      [4, 5, 6]]         [0, 1, 0, 2],
    //                         [1, 0, 1, 0]]
    //
    // C[0,0] = 1*1 + 2*0 + 3*1 = 4
    // C[0,1] = 1*0 + 2*1 + 3*0 = 2
    // C[0,2] = 1*2 + 2*0 + 3*1 = 5
    // C[0,3] = 1*1 + 2*2 + 3*0 = 5
    // C[1,0] = 4*1 + 5*0 + 6*1 = 10
    // C[1,1] = 4*0 + 5*1 + 6*0 = 5
    // C[1,2] = 4*2 + 5*0 + 6*1 = 14
    // C[1,3] = 4*1 + 5*2 + 6*0 = 14
    let a_2x3 = tensor_f32_shaped(
        vec![2, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
    let b_3x4 = tensor_f32_shaped(
        vec![3, 4],
        &[1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0],
    );
    let expected_2x4 = tensor_f32_shaped(
        vec![2, 4],
        &[4.0, 2.0, 5.0, 5.0, 10.0, 5.0, 14.0, 14.0],
    );

    // 1x1 @ 1x1 → 1x1 (scalar multiply)
    let a_1x1 = tensor_f32_shaped(vec![1, 1], &[3.0]);
    let b_1x1 = tensor_f32_shaped(vec![1, 1], &[7.0]);
    let expected_1x1 = tensor_f32_shaped(vec![1, 1], &[21.0]);

    TestCase {
        name: "matmul".to_string(),
        graph,
        data_sets: vec![
            matmul_data_set(
                "f32_2x3_3x4", &ids,
                a_2x3, b_3x4, expected_2x4,
                Tolerance::for_dtype(NumericDType::F32),
            ),
            matmul_data_set(
                "f32_1x1_scalar", &ids,
                a_1x1, b_1x1, expected_1x1,
                Tolerance::for_dtype(NumericDType::F32),
            ),
        ],
    }
}
