//! Reduce operation test cases (ReduceSum, ReduceMax).
//!
//! Tests reduction operations through the MilliOpGraph interpreter.

use std::collections::HashMap;

use crate::graph::GlobalId;
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::Constant;
use crate::milli_graph::ops::ReduceMax;
use crate::milli_graph::ops::ReduceSum;
use crate::numeric_dtype::NumericDType;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use super::tensor_f32_shaped;
use super::{TestCase, TestDataSet, TestTensor, Tolerance};

pub fn build_cases() -> Vec<TestCase> {
    vec![
        reduce_sum_axis1_case(),
        reduce_sum_axis0_case(),
        reduce_max_axis1_case(),
        reduce_max_axis0_case(),
    ]
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn rng() -> SmallRng {
    SmallRng::seed_from_u64(300)
}

struct ReduceGraphIds {
    ext_data: GlobalId,
    out: GlobalId,
}

/// Build a graph: input → ReduceSum(axis=`axis`, keepdims=false) → output
fn build_reduce_sum_graph(axis: i64) -> (MilliOpGraph, ReduceGraphIds) {
    let mut rng = rng();
    let ext_data = GlobalId::new(&mut rng);
    let (mut graph, input_map) = MilliOpGraph::new([ext_data], &mut rng);
    let int_data = input_map[&ext_data];

    // Create the axes constant tensor (1D i64 with one element)
    let axes_id = Constant::from_vec(&mut graph, vec![axis], &mut rng);

    let out = ReduceSum::push_new(
        &mut graph,
        int_data,
        Some(axes_id),
        false, // keepdims
        false, // noop_with_empty_axes
        &mut rng,
    );
    graph.set_outputs(vec![out]);
    (graph, ReduceGraphIds { ext_data, out })
}

/// Build a graph: input → ReduceMax(axis=`axis`, keepdims=false) → output
fn build_reduce_max_graph(axis: i64) -> (MilliOpGraph, ReduceGraphIds) {
    let mut rng = rng();
    let ext_data = GlobalId::new(&mut rng);
    let (mut graph, input_map) = MilliOpGraph::new([ext_data], &mut rng);
    let int_data = input_map[&ext_data];

    let axes_id = Constant::from_vec(&mut graph, vec![axis], &mut rng);

    let out = ReduceMax::push_new(
        &mut graph,
        int_data,
        Some(axes_id),
        false, // keepdims
        false, // noop_with_empty_axes
        &mut rng,
    );
    graph.set_outputs(vec![out]);
    (graph, ReduceGraphIds { ext_data, out })
}

fn reduce_data_set(
    label: &str,
    ids: &ReduceGraphIds,
    input: TestTensor,
    expected: TestTensor,
    tolerance: Tolerance,
) -> TestDataSet {
    TestDataSet {
        label: label.to_string(),
        inputs: HashMap::from([(ids.ext_data, input)]),
        expected_outputs: HashMap::from([(ids.out, expected)]),
        tolerance,
    }
}

// ---------------------------------------------------------------------------
// ReduceSum along axis 1: [2,3] → [2]
// ---------------------------------------------------------------------------

fn reduce_sum_axis1_case() -> TestCase {
    let (graph, ids) = build_reduce_sum_graph(1);

    // [[1, 2, 3], [4, 5, 6]] → reduce axis 1 → [6, 15]
    // row 0: 1+2+3 = 6
    // row 1: 4+5+6 = 15
    TestCase {
        name: "reduce_sum_axis1".to_string(),
        graph,
        data_sets: vec![
            reduce_data_set(
                "f32_2x3",
                &ids,
                tensor_f32_shaped(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                tensor_f32_shaped(vec![2], &[6.0, 15.0]),
                Tolerance::for_dtype(NumericDType::F32),
            ),
            reduce_data_set(
                "f32_2x3_negatives",
                &ids,
                tensor_f32_shaped(vec![2, 3], &[-1.0, 2.0, -3.0, 0.0, 0.5, -0.5]),
                // row 0: -1+2-3 = -2, row 1: 0+0.5-0.5 = 0
                tensor_f32_shaped(vec![2], &[-2.0, 0.0]),
                Tolerance::for_dtype(NumericDType::F32),
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// ReduceSum along axis 0: [2,3] → [3]
// ---------------------------------------------------------------------------

fn reduce_sum_axis0_case() -> TestCase {
    let (graph, ids) = build_reduce_sum_graph(0);

    // [[1, 2, 3], [4, 5, 6]] → reduce axis 0 → [5, 7, 9]
    // col 0: 1+4=5, col 1: 2+5=7, col 2: 3+6=9
    TestCase {
        name: "reduce_sum_axis0".to_string(),
        graph,
        data_sets: vec![reduce_data_set(
            "f32_2x3",
            &ids,
            tensor_f32_shaped(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            tensor_f32_shaped(vec![3], &[5.0, 7.0, 9.0]),
            Tolerance::for_dtype(NumericDType::F32),
        )],
    }
}

// ---------------------------------------------------------------------------
// ReduceMax along axis 1: [2,3] → [2]
// ---------------------------------------------------------------------------

fn reduce_max_axis1_case() -> TestCase {
    let (graph, ids) = build_reduce_max_graph(1);

    // [[1, 5, 3], [4, 2, 6]] → reduce max axis 1 → [5, 6]
    // row 0: max(1,5,3) = 5
    // row 1: max(4,2,6) = 6
    TestCase {
        name: "reduce_max_axis1".to_string(),
        graph,
        data_sets: vec![
            reduce_data_set(
                "f32_2x3",
                &ids,
                tensor_f32_shaped(vec![2, 3], &[1.0, 5.0, 3.0, 4.0, 2.0, 6.0]),
                tensor_f32_shaped(vec![2], &[5.0, 6.0]),
                Tolerance::for_dtype(NumericDType::F32),
            ),
            reduce_data_set(
                "f32_negatives",
                &ids,
                tensor_f32_shaped(vec![2, 3], &[-10.0, -5.0, -1.0, -3.0, -7.0, -2.0]),
                // row 0: max(-10,-5,-1) = -1, row 1: max(-3,-7,-2) = -2
                tensor_f32_shaped(vec![2], &[-1.0, -2.0]),
                Tolerance::for_dtype(NumericDType::F32),
            ),
        ],
    }
}

// ---------------------------------------------------------------------------
// ReduceMax along axis 0: [2,3] → [3]
// ---------------------------------------------------------------------------

fn reduce_max_axis0_case() -> TestCase {
    let (graph, ids) = build_reduce_max_graph(0);

    // [[1, 5, 3], [4, 2, 6]] → reduce max axis 0 → [4, 5, 6]
    // col 0: max(1,4)=4, col 1: max(5,2)=5, col 2: max(3,6)=6
    TestCase {
        name: "reduce_max_axis0".to_string(),
        graph,
        data_sets: vec![reduce_data_set(
            "f32_2x3",
            &ids,
            tensor_f32_shaped(vec![2, 3], &[1.0, 5.0, 3.0, 4.0, 2.0, 6.0]),
            tensor_f32_shaped(vec![3], &[4.0, 5.0, 6.0]),
            Tolerance::for_dtype(NumericDType::F32),
        )],
    }
}
