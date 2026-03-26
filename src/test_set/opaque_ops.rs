//! Test cases for ops that use the OpaqueOp nano-graph mechanism.
//!
//! These ops can't be decomposed into scalar nano-ops and instead use
//! the opaque eval side-channel in pool_eval.

use std::collections::HashMap;

use ndarray::{ArcArray, IxDyn};
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::graph::GlobalId;
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::{self, Constant};
use crate::numeric_dtype::NumericDType;

use super::{TestCase, TestDataSet, TestTensor, Tolerance};
use super::tensor_f32_shaped;

pub fn build_cases() -> Vec<TestCase> {
    vec![
        reduce_min_axis1(),
        reduce_min_axis0(),
        reduce_min_keepdims(),
        reduce_prod_axis0(),
        reduce_prod_axis1(),
        cumsum_axis0(),
        cumsum_axis1(),
        argmax_axis1(),
        argmax_axis0(),
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

    let axes_tensor = NDArrayNumericTensor::I64(
        ArcArray::from_shape_vec(IxDyn(&[1]), vec![axis]).unwrap(),
    );
    let axes_id = Constant::push_new(&mut graph, axes_tensor, &mut rng);

    let out = ops::ReduceMin::push_new(
        &mut graph, int_data, Some(axes_id), keepdims, false, &mut rng,
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
            inputs: HashMap::from([(ids.ext_data, tensor_f32_shaped(vec![2, 3], &[3.0, 1.0, 4.0, 1.0, 5.0, 9.0]))]),
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
            inputs: HashMap::from([(ids.ext_data, tensor_f32_shaped(vec![2, 3], &[3.0, 1.0, 4.0, 1.0, 5.0, 9.0]))]),
            expected_outputs: HashMap::from([(ids.out, tensor_f32_shaped(vec![3], &[1.0, 1.0, 4.0]))]),
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
            inputs: HashMap::from([(ids.ext_data, tensor_f32_shaped(vec![2, 3], &[3.0, 1.0, 4.0, 1.0, 5.0, 9.0]))]),
            expected_outputs: HashMap::from([(ids.out, tensor_f32_shaped(vec![2, 1], &[1.0, 1.0]))]),
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

    let axes_tensor = NDArrayNumericTensor::I64(
        ArcArray::from_shape_vec(IxDyn(&[1]), vec![axis]).unwrap(),
    );
    let axes_id = Constant::push_new(&mut graph, axes_tensor, &mut rng);

    let out = ops::ReduceProd::push_new(
        &mut graph, int_data, Some(axes_id), keepdims, false, &mut rng,
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
            inputs: HashMap::from([(ids.ext_data, tensor_f32_shaped(vec![2, 2], &[1.0, 2.0, 3.0, 4.0]))]),
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
            inputs: HashMap::from([(ids.ext_data, tensor_f32_shaped(vec![2, 2], &[1.0, 2.0, 3.0, 4.0]))]),
            expected_outputs: HashMap::from([(ids.out, tensor_f32_shaped(vec![2], &[2.0, 12.0]))]),
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

    let axis_tensor = NDArrayNumericTensor::I64(
        ArcArray::from_shape_vec(IxDyn(&[1]), vec![axis]).unwrap(),
    );
    let axis_id = Constant::push_new(&mut graph, axis_tensor, &mut rng);

    let out = ops::CumSum::push_new(
        &mut graph, int_data, axis_id, false, false, &mut rng,
    );
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
            expected_outputs: HashMap::from([(ids.out, tensor_f32_shaped(vec![3], &[1.0, 3.0, 6.0]))]),
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
            inputs: HashMap::from([(ids.ext_data, tensor_f32_shaped(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))]),
            expected_outputs: HashMap::from([(ids.out, tensor_f32_shaped(vec![2, 3], &[1.0, 3.0, 6.0, 4.0, 9.0, 15.0]))]),
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

    let out = ops::ArgMax::push_new(
        &mut graph, int_data, axis, keepdims, false, &mut rng,
    );
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
            inputs: HashMap::from([(ids.ext_data, tensor_f32_shaped(vec![2, 3], &[3.0, 1.0, 4.0, 1.0, 5.0, 9.0]))]),
            expected_outputs: HashMap::from([(ids.out, super::tensor_i64_shaped(vec![2], &[2, 2]))]),
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
            inputs: HashMap::from([(ids.ext_data, tensor_f32_shaped(vec![2, 3], &[3.0, 1.0, 4.0, 1.0, 5.0, 9.0]))]),
            expected_outputs: HashMap::from([(ids.out, super::tensor_i64_shaped(vec![3], &[0, 1, 1]))]),
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

    let pads_tensor = NDArrayNumericTensor::I64(
        ArcArray::from_shape_vec(IxDyn(&[pads.len()]), pads).unwrap(),
    );
    let pads_id = Constant::push_new(&mut graph, pads_tensor, &mut rng);

    let const_tensor = NDArrayNumericTensor::F32(
        ArcArray::from_shape_vec(IxDyn(&[1]), vec![constant_value]).unwrap(),
    );
    let const_id = Constant::push_new(&mut graph, const_tensor, &mut rng);

    let out = ops::Pad::push_new(
        &mut graph,
        int_data,
        pads_id,
        Some(const_id),
        None,     // no axes
        ops::PadMode::Constant,
        &mut rng,
    );
    graph.set_outputs(vec![out]);
    (graph, ReduceIds { ext_data, out })
}

fn pad_constant_1d() -> TestCase {
    // [1, 2, 3] with pads [2, 1] (2 before, 1 after) → [0, 0, 1, 2, 3, 0]
    let (graph, ids) = build_pad_graph(vec![2, 1], 0.0);
    TestCase {
        name: "pad_constant_1d".to_string(),
        graph,
        data_sets: vec![TestDataSet {
            label: "3_f32".to_string(),
            inputs: HashMap::from([(ids.ext_data, tensor_f32_shaped(vec![3], &[1.0, 2.0, 3.0]))]),
            expected_outputs: HashMap::from([(ids.out, tensor_f32_shaped(vec![6], &[0.0, 0.0, 1.0, 2.0, 3.0, 0.0]))]),
            tolerance: Tolerance::for_dtype(NumericDType::F32),
        }],
    }
}
