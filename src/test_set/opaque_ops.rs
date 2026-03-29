//! Test cases for ops that use the OpaqueOp nano-graph mechanism.
//!
//! These ops can't be decomposed into scalar nano-ops and instead use
//! the opaque eval side-channel in pool_eval.

use std::collections::HashMap;

use crate::graph::GlobalId;
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::{self, Constant};
use crate::numeric_dtype::NumericDType;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use super::tensor_f32_shaped;
use super::{TestCase, TestDataSet, Tolerance};

pub fn build_cases() -> Vec<TestCase> {
    vec![
        reduce_min_axis1(),
        reduce_min_axis0(),
        reduce_min_keepdims(),
        reduce_prod_axis0(),
        reduce_prod_axis1(),
        reduce_mean_axis1(),
        reduce_sum_multi_axis(),
        cumsum_axis0(),
        cumsum_axis1(),
        argmax_axis1(),
        argmax_axis0(),
        argmin_axis1(),
        argmin_axis0(),
        nonzero_2d(),
        range_int(),
        range_float(),
        sum_to_unbroadcast(),
        conv_bias_grad(),
        pad_constant_1d(),
    ]
}

fn rng() -> SmallRng {
    SmallRng::seed_from_u64(500)
}

// ---------------------------------------------------------------------------
// ReduceMin
// ---------------------------------------------------------------------------

struct ReduceIds {
    ext_data: GlobalId,
    out: GlobalId,
}

fn build_reduce_min_graph(axis: i64, keepdims: bool) -> (MilliOpGraph, ReduceIds) {
    let mut rng = rng();
    let ext_data = GlobalId::new(&mut rng);
    let (mut graph, input_map) = MilliOpGraph::new([ext_data], &mut rng);
    let int_data = input_map[&ext_data];

    let axes_id = Constant::from_vec(&mut graph, vec![axis], &mut rng);

    let out = ops::ReduceMin::push_new(
        &mut graph,
        int_data,
        Some(axes_id),
        keepdims,
        false,
        &mut rng,
    );
    graph.set_outputs(vec![out]);
    (graph, ReduceIds { ext_data, out })
}

fn reduce_min_axis1() -> TestCase {
    let (graph, ids) = build_reduce_min_graph(1, false);
    TestCase {
        name: "reduce_min_axis1".to_string(),
        graph,
        data_sets: vec![TestDataSet {
            label: "2x3_f32".to_string(),
            inputs: HashMap::from([(
                ids.ext_data,
                tensor_f32_shaped(vec![2, 3], &[3.0, 1.0, 4.0, 1.0, 5.0, 9.0]),
            )]),
            expected_outputs: HashMap::from([(ids.out, tensor_f32_shaped(vec![2], &[1.0, 1.0]))]),
            tolerance: Tolerance::for_dtype(NumericDType::F32),
        }],
    }
}

fn reduce_min_axis0() -> TestCase {
    let (graph, ids) = build_reduce_min_graph(0, false);
    TestCase {
        name: "reduce_min_axis0".to_string(),
        graph,
        data_sets: vec![TestDataSet {
            label: "2x3_f32".to_string(),
            inputs: HashMap::from([(
                ids.ext_data,
                tensor_f32_shaped(vec![2, 3], &[3.0, 1.0, 4.0, 1.0, 5.0, 9.0]),
            )]),
            expected_outputs: HashMap::from([(
                ids.out,
                tensor_f32_shaped(vec![3], &[1.0, 1.0, 4.0]),
            )]),
            tolerance: Tolerance::for_dtype(NumericDType::F32),
        }],
    }
}

fn reduce_min_keepdims() -> TestCase {
    let (graph, ids) = build_reduce_min_graph(1, true);
    TestCase {
        name: "reduce_min_keepdims".to_string(),
        graph,
        data_sets: vec![TestDataSet {
            label: "2x3_f32".to_string(),
            inputs: HashMap::from([(
                ids.ext_data,
                tensor_f32_shaped(vec![2, 3], &[3.0, 1.0, 4.0, 1.0, 5.0, 9.0]),
            )]),
            expected_outputs: HashMap::from([(
                ids.out,
                tensor_f32_shaped(vec![2, 1], &[1.0, 1.0]),
            )]),
            tolerance: Tolerance::for_dtype(NumericDType::F32),
        }],
    }
}

// ---------------------------------------------------------------------------
// ReduceProd
// ---------------------------------------------------------------------------

fn build_reduce_prod_graph(axis: i64, keepdims: bool) -> (MilliOpGraph, ReduceIds) {
    let mut rng = rng();
    let ext_data = GlobalId::new(&mut rng);
    let (mut graph, input_map) = MilliOpGraph::new([ext_data], &mut rng);
    let int_data = input_map[&ext_data];

    let axes_id = Constant::from_vec(&mut graph, vec![axis], &mut rng);

    let out = ops::ReduceProd::push_new(
        &mut graph,
        int_data,
        Some(axes_id),
        keepdims,
        false,
        &mut rng,
    );
    graph.set_outputs(vec![out]);
    (graph, ReduceIds { ext_data, out })
}

fn reduce_prod_axis0() -> TestCase {
    let (graph, ids) = build_reduce_prod_graph(0, false);
    TestCase {
        name: "reduce_prod_axis0".to_string(),
        graph,
        data_sets: vec![TestDataSet {
            label: "2x2_f32".to_string(),
            inputs: HashMap::from([(
                ids.ext_data,
                tensor_f32_shaped(vec![2, 2], &[1.0, 2.0, 3.0, 4.0]),
            )]),
            expected_outputs: HashMap::from([(ids.out, tensor_f32_shaped(vec![2], &[3.0, 8.0]))]),
            tolerance: Tolerance::for_dtype(NumericDType::F32),
        }],
    }
}

fn reduce_prod_axis1() -> TestCase {
    let (graph, ids) = build_reduce_prod_graph(1, false);
    TestCase {
        name: "reduce_prod_axis1".to_string(),
        graph,
        data_sets: vec![TestDataSet {
            label: "2x2_f32".to_string(),
            inputs: HashMap::from([(
                ids.ext_data,
                tensor_f32_shaped(vec![2, 2], &[1.0, 2.0, 3.0, 4.0]),
            )]),
            expected_outputs: HashMap::from([(ids.out, tensor_f32_shaped(vec![2], &[2.0, 12.0]))]),
            tolerance: Tolerance::for_dtype(NumericDType::F32),
        }],
    }
}

// ---------------------------------------------------------------------------
// ReduceMean
// ---------------------------------------------------------------------------

fn build_reduce_mean_graph(axis: i64) -> (MilliOpGraph, ReduceIds) {
    let mut rng = rng();
    let ext_data = GlobalId::new(&mut rng);
    let (mut graph, input_map) = MilliOpGraph::new([ext_data], &mut rng);
    let int_data = input_map[&ext_data];

    let axes_id = Constant::from_vec(&mut graph, vec![axis], &mut rng);

    let out =
        ops::ReduceMean::push_new(&mut graph, int_data, Some(axes_id), false, false, &mut rng);
    graph.set_outputs(vec![out]);
    (graph, ReduceIds { ext_data, out })
}

fn reduce_mean_axis1() -> TestCase {
    // [[1, 2, 3], [4, 5, 6]] → mean(axis=1) → [2, 5]
    let (graph, ids) = build_reduce_mean_graph(1);
    TestCase {
        name: "reduce_mean_axis1".to_string(),
        graph,
        data_sets: vec![TestDataSet {
            label: "2x3_f32".to_string(),
            inputs: HashMap::from([(
                ids.ext_data,
                tensor_f32_shaped(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            )]),
            expected_outputs: HashMap::from([(ids.out, tensor_f32_shaped(vec![2], &[2.0, 5.0]))]),
            tolerance: Tolerance::for_dtype(NumericDType::F32),
        }],
    }
}

// ---------------------------------------------------------------------------
// ReduceSum multi-axis (tests eval_new fallback since lower only handles 1 axis)
// ---------------------------------------------------------------------------

fn build_reduce_sum_multi_axis_graph(axes: Vec<i64>) -> (MilliOpGraph, ReduceIds) {
    let mut rng = rng();
    let ext_data = GlobalId::new(&mut rng);
    let (mut graph, input_map) = MilliOpGraph::new([ext_data], &mut rng);
    let int_data = input_map[&ext_data];

    let axes_id = Constant::from_vec(&mut graph, axes, &mut rng);

    let out = ops::ReduceSum::push_new(&mut graph, int_data, Some(axes_id), false, false, &mut rng);
    graph.set_outputs(vec![out]);
    (graph, ReduceIds { ext_data, out })
}

fn reduce_sum_multi_axis() -> TestCase {
    // [2,3,2] → sum(axes=[0,2]) → [3]
    // input: [[[1,2],[3,4],[5,6]], [[7,8],[9,10],[11,12]]]
    // sum axes 0,2: col 0: 1+2+7+8=18, col 1: 3+4+9+10=26, col 2: 5+6+11+12=34
    let (graph, ids) = build_reduce_sum_multi_axis_graph(vec![0, 2]);
    TestCase {
        name: "reduce_sum_multi_axis".to_string(),
        graph,
        data_sets: vec![TestDataSet {
            label: "2x3x2_f32".to_string(),
            inputs: HashMap::from([(
                ids.ext_data,
                tensor_f32_shaped(
                    vec![2, 3, 2],
                    &[
                        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                    ],
                ),
            )]),
            expected_outputs: HashMap::from([(
                ids.out,
                tensor_f32_shaped(vec![3], &[18.0, 26.0, 34.0]),
            )]),
            tolerance: Tolerance::for_dtype(NumericDType::F32),
        }],
    }
}

// ---------------------------------------------------------------------------
// CumSum
// ---------------------------------------------------------------------------

fn build_cumsum_graph(axis: i64) -> (MilliOpGraph, ReduceIds) {
    let mut rng = rng();
    let ext_data = GlobalId::new(&mut rng);
    let (mut graph, input_map) = MilliOpGraph::new([ext_data], &mut rng);
    let int_data = input_map[&ext_data];

    let axis_id = Constant::from_vec(&mut graph, vec![axis], &mut rng);

    let out = ops::CumSum::push_new(&mut graph, int_data, axis_id, false, false, &mut rng);
    graph.set_outputs(vec![out]);
    (graph, ReduceIds { ext_data, out })
}

fn cumsum_axis0() -> TestCase {
    // [1, 2, 3] → cumsum(axis=0) → [1, 3, 6]
    let (graph, ids) = build_cumsum_graph(0);
    TestCase {
        name: "cumsum_axis0".to_string(),
        graph,
        data_sets: vec![TestDataSet {
            label: "3_f32".to_string(),
            inputs: HashMap::from([(ids.ext_data, tensor_f32_shaped(vec![3], &[1.0, 2.0, 3.0]))]),
            expected_outputs: HashMap::from([(
                ids.out,
                tensor_f32_shaped(vec![3], &[1.0, 3.0, 6.0]),
            )]),
            tolerance: Tolerance::for_dtype(NumericDType::F32),
        }],
    }
}

fn cumsum_axis1() -> TestCase {
    // [[1, 2, 3], [4, 5, 6]] → cumsum(axis=1) → [[1, 3, 6], [4, 9, 15]]
    let (graph, ids) = build_cumsum_graph(1);
    TestCase {
        name: "cumsum_axis1".to_string(),
        graph,
        data_sets: vec![TestDataSet {
            label: "2x3_f32".to_string(),
            inputs: HashMap::from([(
                ids.ext_data,
                tensor_f32_shaped(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            )]),
            expected_outputs: HashMap::from([(
                ids.out,
                tensor_f32_shaped(vec![2, 3], &[1.0, 3.0, 6.0, 4.0, 9.0, 15.0]),
            )]),
            tolerance: Tolerance::for_dtype(NumericDType::F32),
        }],
    }
}

// ---------------------------------------------------------------------------
// ArgMax
// ---------------------------------------------------------------------------

fn build_argmax_graph(axis: i64, keepdims: bool) -> (MilliOpGraph, ReduceIds) {
    let mut rng = rng();
    let ext_data = GlobalId::new(&mut rng);
    let (mut graph, input_map) = MilliOpGraph::new([ext_data], &mut rng);
    let int_data = input_map[&ext_data];

    let out = ops::ArgMax::push_new(&mut graph, int_data, axis, keepdims, false, &mut rng);
    graph.set_outputs(vec![out]);
    (graph, ReduceIds { ext_data, out })
}

fn argmax_axis1() -> TestCase {
    // [[3, 1, 4], [1, 5, 9]] → argmax(axis=1) → [2, 2] (indices of max along axis 1)
    let (graph, ids) = build_argmax_graph(1, false);
    TestCase {
        name: "argmax_axis1".to_string(),
        graph,
        data_sets: vec![TestDataSet {
            label: "2x3_f32".to_string(),
            inputs: HashMap::from([(
                ids.ext_data,
                tensor_f32_shaped(vec![2, 3], &[3.0, 1.0, 4.0, 1.0, 5.0, 9.0]),
            )]),
            expected_outputs: HashMap::from([(
                ids.out,
                super::tensor_i64_shaped(vec![2], &[2, 2]),
            )]),
            tolerance: Tolerance::for_dtype(NumericDType::I64),
        }],
    }
}

fn argmax_axis0() -> TestCase {
    // [[3, 1, 4], [1, 5, 9]] → argmax(axis=0) → [0, 1, 1]
    let (graph, ids) = build_argmax_graph(0, false);
    TestCase {
        name: "argmax_axis0".to_string(),
        graph,
        data_sets: vec![TestDataSet {
            label: "2x3_f32".to_string(),
            inputs: HashMap::from([(
                ids.ext_data,
                tensor_f32_shaped(vec![2, 3], &[3.0, 1.0, 4.0, 1.0, 5.0, 9.0]),
            )]),
            expected_outputs: HashMap::from([(
                ids.out,
                super::tensor_i64_shaped(vec![3], &[0, 1, 1]),
            )]),
            tolerance: Tolerance::for_dtype(NumericDType::I64),
        }],
    }
}

// ---------------------------------------------------------------------------
// Pad (constant mode)
// ---------------------------------------------------------------------------

fn build_pad_graph(pads: Vec<i64>, constant_value: f32) -> (MilliOpGraph, ReduceIds) {
    let mut rng = rng();
    let ext_data = GlobalId::new(&mut rng);
    let (mut graph, input_map) = MilliOpGraph::new([ext_data], &mut rng);
    let int_data = input_map[&ext_data];

    let pads_id = Constant::from_vec(&mut graph, pads, &mut rng);

    let const_id = Constant::from_vec(&mut graph, vec![constant_value], &mut rng);

    let out = ops::Pad::push_new(
        &mut graph,
        int_data,
        pads_id,
        Some(const_id),
        None, // no axes
        ops::PadMode::Constant,
        &mut rng,
    );
    graph.set_outputs(vec![out]);
    (graph, ReduceIds { ext_data, out })
}

// ---------------------------------------------------------------------------
// ArgMin
// ---------------------------------------------------------------------------

fn build_argmin_graph(axis: i64, keepdims: bool) -> (MilliOpGraph, ReduceIds) {
    let mut rng = rng();
    let ext_data = GlobalId::new(&mut rng);
    let (mut graph, input_map) = MilliOpGraph::new([ext_data], &mut rng);
    let int_data = input_map[&ext_data];

    let out = ops::ArgMin::push_new(&mut graph, int_data, axis, keepdims, false, &mut rng);
    graph.set_outputs(vec![out]);
    (graph, ReduceIds { ext_data, out })
}

fn argmin_axis1() -> TestCase {
    // [[3, 1, 4], [9, 5, 1]] → argmin(axis=1) → [1, 2]
    let (graph, ids) = build_argmin_graph(1, false);
    TestCase {
        name: "argmin_axis1".to_string(),
        graph,
        data_sets: vec![TestDataSet {
            label: "2x3_f32".to_string(),
            inputs: HashMap::from([(
                ids.ext_data,
                tensor_f32_shaped(vec![2, 3], &[3.0, 1.0, 4.0, 9.0, 5.0, 1.0]),
            )]),
            expected_outputs: HashMap::from([(
                ids.out,
                super::tensor_i64_shaped(vec![2], &[1, 2]),
            )]),
            tolerance: Tolerance::for_dtype(NumericDType::I64),
        }],
    }
}

fn argmin_axis0() -> TestCase {
    // [[3, 1, 4], [1, 5, 9]] → argmin(axis=0) → [1, 0, 0]
    let (graph, ids) = build_argmin_graph(0, false);
    TestCase {
        name: "argmin_axis0".to_string(),
        graph,
        data_sets: vec![TestDataSet {
            label: "2x3_f32".to_string(),
            inputs: HashMap::from([(
                ids.ext_data,
                tensor_f32_shaped(vec![2, 3], &[3.0, 1.0, 4.0, 1.0, 5.0, 9.0]),
            )]),
            expected_outputs: HashMap::from([(
                ids.out,
                super::tensor_i64_shaped(vec![3], &[1, 0, 0]),
            )]),
            tolerance: Tolerance::for_dtype(NumericDType::I64),
        }],
    }
}

// ---------------------------------------------------------------------------
// NonZero
// ---------------------------------------------------------------------------

fn nonzero_2d() -> TestCase {
    // [[1, 0], [0, 2]] → nonzero → [[0, 1], [0, 1]] (row indices, col indices of nonzeros)
    let mut rng = rng();
    let ext_data = GlobalId::new(&mut rng);
    let (mut graph, input_map) = MilliOpGraph::new([ext_data], &mut rng);
    let int_data = input_map[&ext_data];

    let out = ops::NonZero::push_new(&mut graph, int_data, &mut rng);
    graph.set_outputs(vec![out]);

    TestCase {
        name: "nonzero_2d".to_string(),
        graph,
        data_sets: vec![TestDataSet {
            label: "2x2_f32".to_string(),
            inputs: HashMap::from([(
                ext_data,
                tensor_f32_shaped(vec![2, 2], &[1.0, 0.0, 0.0, 2.0]),
            )]),
            expected_outputs: HashMap::from([(
                out,
                super::tensor_i64_shaped(vec![2, 2], &[0, 1, 0, 1]),
            )]),
            tolerance: Tolerance::for_dtype(NumericDType::I64),
        }],
    }
}

// ---------------------------------------------------------------------------
// Range
// ---------------------------------------------------------------------------

struct RangeIds {
    ext_start: GlobalId,
    ext_end: GlobalId,
    ext_delta: GlobalId,
    out: GlobalId,
}

fn build_range_graph() -> (MilliOpGraph, RangeIds) {
    let mut rng = rng();
    let ext_start = GlobalId::new(&mut rng);
    let ext_end = GlobalId::new(&mut rng);
    let ext_delta = GlobalId::new(&mut rng);
    let (mut graph, input_map) = MilliOpGraph::new([ext_start, ext_end, ext_delta], &mut rng);
    let int_start = input_map[&ext_start];
    let int_end = input_map[&ext_end];
    let int_delta = input_map[&ext_delta];

    let out = ops::Range::push_new(&mut graph, int_start, int_end, int_delta, &mut rng);
    graph.set_outputs(vec![out]);
    (
        graph,
        RangeIds {
            ext_start,
            ext_end,
            ext_delta,
            out,
        },
    )
}

fn range_int() -> TestCase {
    // range(0, 5, 1) → [0, 1, 2, 3, 4]
    let (graph, ids) = build_range_graph();
    TestCase {
        name: "range_int".to_string(),
        graph,
        data_sets: vec![TestDataSet {
            label: "0_to_5".to_string(),
            inputs: HashMap::from([
                (ids.ext_start, super::tensor_i64_shaped(vec![1], &[0])),
                (ids.ext_end, super::tensor_i64_shaped(vec![1], &[5])),
                (ids.ext_delta, super::tensor_i64_shaped(vec![1], &[1])),
            ]),
            expected_outputs: HashMap::from([(
                ids.out,
                super::tensor_i64_shaped(vec![5], &[0, 1, 2, 3, 4]),
            )]),
            tolerance: Tolerance::for_dtype(NumericDType::I64),
        }],
    }
}

fn range_float() -> TestCase {
    // range(0.0, 1.0, 0.25) → [0.0, 0.25, 0.5, 0.75]
    let (graph, ids) = build_range_graph();
    TestCase {
        name: "range_float".to_string(),
        graph,
        data_sets: vec![TestDataSet {
            label: "0_to_1_step_0.25".to_string(),
            inputs: HashMap::from([
                (ids.ext_start, tensor_f32_shaped(vec![1], &[0.0])),
                (ids.ext_end, tensor_f32_shaped(vec![1], &[1.0])),
                (ids.ext_delta, tensor_f32_shaped(vec![1], &[0.25])),
            ]),
            expected_outputs: HashMap::from([(
                ids.out,
                tensor_f32_shaped(vec![4], &[0.0, 0.25, 0.5, 0.75]),
            )]),
            tolerance: Tolerance::for_dtype(NumericDType::F32),
        }],
    }
}

// ---------------------------------------------------------------------------
// SumTo (un-broadcast reduction)
// ---------------------------------------------------------------------------

fn sum_to_unbroadcast() -> TestCase {
    // data [2,3] → target_shape [1,3] → sum axis 0, result [1,3]
    let mut rng = rng();
    let ext_data = GlobalId::new(&mut rng);
    let ext_shape = GlobalId::new(&mut rng);
    let (mut graph, input_map) = MilliOpGraph::new([ext_data, ext_shape], &mut rng);
    let int_data = input_map[&ext_data];
    let int_shape = input_map[&ext_shape];

    let out = ops::SumTo::push_new(&mut graph, int_data, int_shape, &mut rng);
    graph.set_outputs(vec![out]);

    // [[1,2,3],[4,5,6]] → sum_to [1,3] → [[5,7,9]]
    TestCase {
        name: "sum_to_unbroadcast".to_string(),
        graph,
        data_sets: vec![TestDataSet {
            label: "2x3_to_1x3".to_string(),
            inputs: HashMap::from([
                (
                    ext_data,
                    tensor_f32_shaped(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                ),
                (ext_shape, super::tensor_i64_shaped(vec![2], &[1, 3])),
            ]),
            expected_outputs: HashMap::from([(
                out,
                tensor_f32_shaped(vec![1, 3], &[5.0, 7.0, 9.0]),
            )]),
            tolerance: Tolerance::for_dtype(NumericDType::F32),
        }],
    }
}

// ---------------------------------------------------------------------------
// ConvBiasGrad
// ---------------------------------------------------------------------------

fn conv_bias_grad() -> TestCase {
    // grad_output [1, 2, 3, 3] → sum over axes [0,2,3] → [2]
    // Channel 0: sum of 9 elements, Channel 1: sum of 9 elements
    let mut rng = rng();
    let ext_grad = GlobalId::new(&mut rng);
    let (mut graph, input_map) = MilliOpGraph::new([ext_grad], &mut rng);
    let int_grad = input_map[&ext_grad];

    let out = ops::ConvBiasGrad::push_new(&mut graph, int_grad, &mut rng);
    graph.set_outputs(vec![out]);

    // grad: all 1s for channel 0, all 2s for channel 1
    let mut grad_vals = vec![1.0f32; 9];
    grad_vals.extend(vec![2.0f32; 9]);

    TestCase {
        name: "conv_bias_grad".to_string(),
        graph,
        data_sets: vec![TestDataSet {
            label: "1x2x3x3".to_string(),
            inputs: HashMap::from([(ext_grad, tensor_f32_shaped(vec![1, 2, 3, 3], &grad_vals))]),
            expected_outputs: HashMap::from([(out, tensor_f32_shaped(vec![2], &[9.0, 18.0]))]),
            tolerance: Tolerance::for_dtype(NumericDType::F32),
        }],
    }
}

// ---------------------------------------------------------------------------
// Pad (constant mode)
// ---------------------------------------------------------------------------

fn pad_constant_1d() -> TestCase {
    // [1, 2, 3] with pads [2, 1] (2 before, 1 after) → [0, 0, 1, 2, 3, 0]
    let (graph, ids) = build_pad_graph(vec![2, 1], 0.0);
    TestCase {
        name: "pad_constant_1d".to_string(),
        graph,
        data_sets: vec![TestDataSet {
            label: "3_f32".to_string(),
            inputs: HashMap::from([(ids.ext_data, tensor_f32_shaped(vec![3], &[1.0, 2.0, 3.0]))]),
            expected_outputs: HashMap::from([(
                ids.out,
                tensor_f32_shaped(vec![6], &[0.0, 0.0, 1.0, 2.0, 3.0, 0.0]),
            )]),
            tolerance: Tolerance::for_dtype(NumericDType::F32),
        }],
    }
}
