//! Test cases for structural/layout ops: Transpose, Reshape, Squeeze, Unsqueeze,
//! Expand, Shape, Slice, Concat, Split, Where, ClampMin, Gather.

use std::collections::HashMap;

use ndarray::{ArcArray, IxDyn};
use rand::rngs::SmallRng;
use rand::SeedableRng;

use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::graph::GlobalId;
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::{self, Constant};
use crate::numeric_dtype::NumericDType;

use super::{TestCase, TestDataSet, TestTensor, Tolerance};
use super::{tensor_f32_shaped, tensor_i64_shaped, tensor_bool_shaped};

pub fn build_cases() -> Vec<TestCase> {
    vec![
        transpose_2d(),
        transpose_3d(),
        reshape_basic(),
        squeeze_basic(),
        unsqueeze_basic(),
        expand_broadcast(),
        shape_op(),
        slice_basic(),
        slice_negative_step(),
        concat_axis0(),
        concat_axis1(),
        split_axis0(),
        where_basic(),
        where_broadcast(),
        clampmin_basic(),
        gather_axis0(),
        gather_axis1(),
        topk_basic(),
        resize_nearest(),
    ]
}

fn rng() -> SmallRng {
    SmallRng::seed_from_u64(600)
}

struct SingleInputIds {
    ext_data: GlobalId,
    out: GlobalId,
}

// ---------------------------------------------------------------------------
// Helpers for constant tensor creation
// ---------------------------------------------------------------------------

fn i64_const(graph: &mut MilliOpGraph, vals: Vec<i64>, rng: &mut SmallRng) -> GlobalId {
    let t = NDArrayNumericTensor::I64(ArcArray::from_shape_vec(IxDyn(&[vals.len()]), vals).unwrap());
    Constant::push_new(graph, t, rng)
}

fn f32_const(graph: &mut MilliOpGraph, vals: Vec<f32>, shape: Vec<usize>, rng: &mut SmallRng) -> GlobalId {
    let t = NDArrayNumericTensor::F32(ArcArray::from_shape_vec(IxDyn(&shape), vals).unwrap());
    Constant::push_new(graph, t, rng)
}

fn scalar_i64_const(graph: &mut MilliOpGraph, val: i64, rng: &mut SmallRng) -> GlobalId {
    let t = NDArrayNumericTensor::I64(ArcArray::from_shape_vec(IxDyn(&[1]), vec![val]).unwrap());
    Constant::push_new(graph, t, rng)
}

// ---------------------------------------------------------------------------
// Transpose
// ---------------------------------------------------------------------------

fn transpose_2d() -> TestCase {
    // Identity (forces computed atoms) → transpose [1,0]
    // [[1,2,3],[4,5,6]] → transpose → [[1,4],[2,5],[3,6]]
    let mut rng = rng();
    let ext = GlobalId::new(&mut rng);
    let (mut graph, imap) = MilliOpGraph::new([ext], &mut rng);
    // ClampMin with -inf materializes computed atoms so pool_eval test can find them.
    let ident = ops::ClampMin::push_new(&mut graph, imap[&ext], f32::NEG_INFINITY, &mut rng);
    let out = ops::Transpose::push_new(&mut graph, ident, Some(vec![1, 0]), &mut rng);
    graph.set_outputs(vec![out]);

    TestCase {
        name: "transpose_2d".into(),
        graph,
        data_sets: vec![TestDataSet {
            label: "2x3".into(),
            inputs: HashMap::from([(ext, tensor_f32_shaped(vec![2, 3], &[1.0,2.0,3.0,4.0,5.0,6.0]))]),
            expected_outputs: HashMap::from([(out, tensor_f32_shaped(vec![3, 2], &[1.0,4.0,2.0,5.0,3.0,6.0]))]),
            tolerance: Tolerance::for_dtype(NumericDType::F32),
        }],
    }
}

fn transpose_3d() -> TestCase {
    let mut rng = rng();
    let ext = GlobalId::new(&mut rng);
    let (mut graph, imap) = MilliOpGraph::new([ext], &mut rng);
    let ident = ops::ClampMin::push_new(&mut graph, imap[&ext], f32::NEG_INFINITY, &mut rng);
    let out = ops::Transpose::push_new(&mut graph, ident, Some(vec![2, 0, 1]), &mut rng);
    graph.set_outputs(vec![out]);

    // Input [1,2,3]: [[[1,2,3],[4,5,6]]]
    // Perm [2,0,1]: out[k,i,j] = in[i,j,k], output shape [3,1,2]
    // out[0,0,:] = [in[0,0,0], in[0,1,0]] = [1,4]
    // out[1,0,:] = [in[0,0,1], in[0,1,1]] = [2,5]
    // out[2,0,:] = [in[0,0,2], in[0,1,2]] = [3,6]
    TestCase {
        name: "transpose_3d".into(),
        graph,
        data_sets: vec![TestDataSet {
            label: "1x2x3_perm201".into(),
            inputs: HashMap::from([(ext, tensor_f32_shaped(vec![1, 2, 3], &[1.0,2.0,3.0,4.0,5.0,6.0]))]),
            expected_outputs: HashMap::from([(out, tensor_f32_shaped(vec![3, 1, 2], &[1.0,4.0,2.0,5.0,3.0,6.0]))]),
            tolerance: Tolerance::for_dtype(NumericDType::F32),
        }],
    }
}

// ---------------------------------------------------------------------------
// Reshape
// ---------------------------------------------------------------------------

fn reshape_basic() -> TestCase {
    let mut rng = rng();
    let ext = GlobalId::new(&mut rng);
    let (mut graph, imap) = MilliOpGraph::new([ext], &mut rng);
    let ident = ops::ClampMin::push_new(&mut graph, imap[&ext], f32::NEG_INFINITY, &mut rng);
    let shape_id = i64_const(&mut graph, vec![3, 2], &mut rng);
    let out = ops::Reshape::push_new(&mut graph, ident, shape_id, false, &mut rng);
    graph.set_outputs(vec![out]);

    TestCase {
        name: "reshape_basic".into(),
        graph,
        data_sets: vec![TestDataSet {
            label: "2x3_to_3x2".into(),
            inputs: HashMap::from([(ext, tensor_f32_shaped(vec![2, 3], &[1.0,2.0,3.0,4.0,5.0,6.0]))]),
            expected_outputs: HashMap::from([(out, tensor_f32_shaped(vec![3, 2], &[1.0,2.0,3.0,4.0,5.0,6.0]))]),
            tolerance: Tolerance::for_dtype(NumericDType::F32),
        }],
    }
}

// ---------------------------------------------------------------------------
// Squeeze / Unsqueeze
// ---------------------------------------------------------------------------

fn squeeze_basic() -> TestCase {
    let mut rng = rng();
    let ext = GlobalId::new(&mut rng);
    let (mut graph, imap) = MilliOpGraph::new([ext], &mut rng);
    let ident = ops::ClampMin::push_new(&mut graph, imap[&ext], f32::NEG_INFINITY, &mut rng);
    let axes_id = i64_const(&mut graph, vec![0, 2], &mut rng);
    let out = ops::Squeeze::push_new(&mut graph, ident, axes_id, &mut rng);
    graph.set_outputs(vec![out]);

    TestCase {
        name: "squeeze_basic".into(),
        graph,
        data_sets: vec![TestDataSet {
            label: "1x3x1_to_3".into(),
            inputs: HashMap::from([(ext, tensor_f32_shaped(vec![1, 3, 1], &[1.0, 2.0, 3.0]))]),
            expected_outputs: HashMap::from([(out, tensor_f32_shaped(vec![3], &[1.0, 2.0, 3.0]))]),
            tolerance: Tolerance::for_dtype(NumericDType::F32),
        }],
    }
}

fn unsqueeze_basic() -> TestCase {
    let mut rng = rng();
    let ext = GlobalId::new(&mut rng);
    let (mut graph, imap) = MilliOpGraph::new([ext], &mut rng);
    let ident = ops::ClampMin::push_new(&mut graph, imap[&ext], f32::NEG_INFINITY, &mut rng);
    let axes_id = i64_const(&mut graph, vec![0, 2], &mut rng);
    let out = ops::Unsqueeze::push_new(&mut graph, ident, axes_id, &mut rng);
    graph.set_outputs(vec![out]);

    TestCase {
        name: "unsqueeze_basic".into(),
        graph,
        data_sets: vec![TestDataSet {
            label: "3_to_1x3x1".into(),
            inputs: HashMap::from([(ext, tensor_f32_shaped(vec![3], &[1.0, 2.0, 3.0]))]),
            expected_outputs: HashMap::from([(out, tensor_f32_shaped(vec![1, 3, 1], &[1.0, 2.0, 3.0]))]),
            tolerance: Tolerance::for_dtype(NumericDType::F32),
        }],
    }
}

// ---------------------------------------------------------------------------
// Expand
// ---------------------------------------------------------------------------

fn expand_broadcast() -> TestCase {
    let mut rng = rng();
    let ext = GlobalId::new(&mut rng);
    let (mut graph, imap) = MilliOpGraph::new([ext], &mut rng);
    let ident = ops::ClampMin::push_new(&mut graph, imap[&ext], f32::NEG_INFINITY, &mut rng);
    let shape_id = i64_const(&mut graph, vec![2, 3], &mut rng);
    let out = ops::Expand::push_new(&mut graph, ident, shape_id, &mut rng);
    graph.set_outputs(vec![out]);

    TestCase {
        name: "expand_broadcast".into(),
        graph,
        data_sets: vec![TestDataSet {
            label: "1x3_to_2x3".into(),
            inputs: HashMap::from([(ext, tensor_f32_shaped(vec![1, 3], &[1.0, 2.0, 3.0]))]),
            expected_outputs: HashMap::from([(out, tensor_f32_shaped(vec![2, 3], &[1.0,2.0,3.0,1.0,2.0,3.0]))]),
            tolerance: Tolerance::for_dtype(NumericDType::F32),
        }],
    }
}

// ---------------------------------------------------------------------------
// Shape
// ---------------------------------------------------------------------------

fn shape_op() -> TestCase {
    let mut rng = rng();
    let ext = GlobalId::new(&mut rng);
    let (mut graph, imap) = MilliOpGraph::new([ext], &mut rng);
    let ident = ops::ClampMin::push_new(&mut graph, imap[&ext], f32::NEG_INFINITY, &mut rng);
    let out = ops::Shape::push_new(&mut graph, ident, &mut rng);
    graph.set_outputs(vec![out]);

    TestCase {
        name: "shape_op".into(),
        graph,
        data_sets: vec![TestDataSet {
            label: "2x3x4".into(),
            inputs: HashMap::from([(ext, tensor_f32_shaped(vec![2, 3, 4], &vec![0.0; 24]))]),
            expected_outputs: HashMap::from([(out, tensor_i64_shaped(vec![3], &[2, 3, 4]))]),
            tolerance: Tolerance::for_dtype(NumericDType::I64),
        }],
    }
}

// ---------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------

fn slice_basic() -> TestCase {
    let mut rng = rng();
    let ext = GlobalId::new(&mut rng);
    let (mut graph, imap) = MilliOpGraph::new([ext], &mut rng);
    let ident = ops::ClampMin::push_new(&mut graph, imap[&ext], f32::NEG_INFINITY, &mut rng);
    let starts = i64_const(&mut graph, vec![0, 1], &mut rng);
    let ends = i64_const(&mut graph, vec![2, 3], &mut rng);
    let axes = i64_const(&mut graph, vec![0, 1], &mut rng);
    let steps = i64_const(&mut graph, vec![1, 1], &mut rng);
    let out = ops::Slice::push_new(&mut graph, ident, starts, ends, Some(steps), Some(axes), &mut rng);
    graph.set_outputs(vec![out]);

    // Input: [[1,2,3,4],[5,6,7,8]]
    // Slice [0:2, 1:3] → [[2,3],[6,7]]
    TestCase {
        name: "slice_basic".into(),
        graph,
        data_sets: vec![TestDataSet {
            label: "2x4_slice".into(),
            inputs: HashMap::from([(ext, tensor_f32_shaped(vec![2, 4], &[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]))]),
            expected_outputs: HashMap::from([(out, tensor_f32_shaped(vec![2, 2], &[2.0,3.0,6.0,7.0]))]),
            tolerance: Tolerance::for_dtype(NumericDType::F32),
        }],
    }
}

fn slice_negative_step() -> TestCase {
    let mut rng = rng();
    let ext = GlobalId::new(&mut rng);
    let (mut graph, imap) = MilliOpGraph::new([ext], &mut rng);
    let ident = ops::ClampMin::push_new(&mut graph, imap[&ext], f32::NEG_INFINITY, &mut rng);
    let starts = i64_const(&mut graph, vec![5], &mut rng);
    let ends = i64_const(&mut graph, vec![-7], &mut rng); // effectively -1 past start
    let axes = i64_const(&mut graph, vec![0], &mut rng);
    let steps = i64_const(&mut graph, vec![-1], &mut rng);
    let out = ops::Slice::push_new(&mut graph, ident, starts, ends, Some(steps), Some(axes), &mut rng);
    graph.set_outputs(vec![out]);

    // [1,2,3,4,5,6] reversed → [6,5,4,3,2,1]
    TestCase {
        name: "slice_negative_step".into(),
        graph,
        data_sets: vec![TestDataSet {
            label: "reverse_6".into(),
            inputs: HashMap::from([(ext, tensor_f32_shaped(vec![6], &[1.0,2.0,3.0,4.0,5.0,6.0]))]),
            expected_outputs: HashMap::from([(out, tensor_f32_shaped(vec![6], &[6.0,5.0,4.0,3.0,2.0,1.0]))]),
            tolerance: Tolerance::for_dtype(NumericDType::F32),
        }],
    }
}

// ---------------------------------------------------------------------------
// Concat
// ---------------------------------------------------------------------------

fn concat_axis0() -> TestCase {
    let mut rng = rng();
    let ext_a = GlobalId::new(&mut rng);
    let ext_b = GlobalId::new(&mut rng);
    let (mut graph, imap) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
    let a = ops::ClampMin::push_new(&mut graph, imap[&ext_a], f32::NEG_INFINITY, &mut rng);
    let b = ops::ClampMin::push_new(&mut graph, imap[&ext_b], f32::NEG_INFINITY, &mut rng);
    let out = ops::Concat::push_new(&mut graph, vec![a, b], 0, &mut rng);
    graph.set_outputs(vec![out]);

    TestCase {
        name: "concat_axis0".into(),
        graph,
        data_sets: vec![TestDataSet {
            label: "2x3_1x3".into(),
            inputs: HashMap::from([
                (ext_a, tensor_f32_shaped(vec![2, 3], &[1.0,2.0,3.0,4.0,5.0,6.0])),
                (ext_b, tensor_f32_shaped(vec![1, 3], &[7.0,8.0,9.0])),
            ]),
            expected_outputs: HashMap::from([(out, tensor_f32_shaped(vec![3, 3], &[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]))]),
            tolerance: Tolerance::for_dtype(NumericDType::F32),
        }],
    }
}

fn concat_axis1() -> TestCase {
    let mut rng = rng();
    let ext_a = GlobalId::new(&mut rng);
    let ext_b = GlobalId::new(&mut rng);
    let (mut graph, imap) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
    let a = ops::ClampMin::push_new(&mut graph, imap[&ext_a], f32::NEG_INFINITY, &mut rng);
    let b = ops::ClampMin::push_new(&mut graph, imap[&ext_b], f32::NEG_INFINITY, &mut rng);
    let out = ops::Concat::push_new(&mut graph, vec![a, b], 1, &mut rng);
    graph.set_outputs(vec![out]);

    TestCase {
        name: "concat_axis1".into(),
        graph,
        data_sets: vec![TestDataSet {
            label: "2x2_2x1".into(),
            inputs: HashMap::from([
                (ext_a, tensor_f32_shaped(vec![2, 2], &[1.0,2.0,3.0,4.0])),
                (ext_b, tensor_f32_shaped(vec![2, 1], &[5.0,6.0])),
            ]),
            expected_outputs: HashMap::from([(out, tensor_f32_shaped(vec![2, 3], &[1.0,2.0,5.0,3.0,4.0,6.0]))]),
            tolerance: Tolerance::for_dtype(NumericDType::F32),
        }],
    }
}

// ---------------------------------------------------------------------------
// Split
// ---------------------------------------------------------------------------

fn split_axis0() -> TestCase {
    // [4,2] split axis=0 into [2,2] and [2,2]
    // Split creates one op per output chunk (output_id=0 for first, 1 for second).
    let mut rng = rng();
    let ext = GlobalId::new(&mut rng);
    let (mut graph, imap) = MilliOpGraph::new([ext], &mut rng);

    let split_lit = ops::MilliOpTensorIDOrLiteral::Literal(
        NDArrayNumericTensor::I64(ArcArray::from_shape_vec(IxDyn(&[2]), vec![2, 2]).unwrap()),
    );
    let ident = ops::ClampMin::push_new(&mut graph, imap[&ext], f32::NEG_INFINITY, &mut rng);
    let out0 = ops::Split::push_new(&mut graph, ident, Some(split_lit.clone()), 0, None, 0, &mut rng);
    let out1 = ops::Split::push_new(&mut graph, ident, Some(split_lit), 0, None, 1, &mut rng);
    graph.set_outputs(vec![out0, out1]);

    TestCase {
        name: "split_axis0".into(),
        graph,
        data_sets: vec![TestDataSet {
            label: "4x2_to_2x2x2".into(),
            inputs: HashMap::from([(ext, tensor_f32_shaped(vec![4, 2], &[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]))]),
            expected_outputs: HashMap::from([
                (out0, tensor_f32_shaped(vec![2, 2], &[1.0,2.0,3.0,4.0])),
                (out1, tensor_f32_shaped(vec![2, 2], &[5.0,6.0,7.0,8.0])),
            ]),
            tolerance: Tolerance::for_dtype(NumericDType::F32),
        }],
    }
}

// ---------------------------------------------------------------------------
// Where
// ---------------------------------------------------------------------------

fn where_basic() -> TestCase {
    // cond=[T,F,T], x=[1,2,3], y=[4,5,6] → [1,5,3]
    let mut rng = rng();
    let ext_c = GlobalId::new(&mut rng);
    let ext_x = GlobalId::new(&mut rng);
    let ext_y = GlobalId::new(&mut rng);
    let (mut graph, imap) = MilliOpGraph::new([ext_c, ext_x, ext_y], &mut rng);
    let out = ops::Where::push_new(&mut graph, imap[&ext_c], imap[&ext_x], imap[&ext_y], &mut rng);
    graph.set_outputs(vec![out]);

    TestCase {
        name: "where_basic".into(),
        graph,
        data_sets: vec![TestDataSet {
            label: "3elem".into(),
            inputs: HashMap::from([
                (ext_c, tensor_bool_shaped(vec![3], &[true, false, true])),
                (ext_x, tensor_f32_shaped(vec![3], &[1.0, 2.0, 3.0])),
                (ext_y, tensor_f32_shaped(vec![3], &[4.0, 5.0, 6.0])),
            ]),
            expected_outputs: HashMap::from([(out, tensor_f32_shaped(vec![3], &[1.0, 5.0, 3.0]))]),
            tolerance: Tolerance::for_dtype(NumericDType::F32),
        }],
    }
}

fn where_broadcast() -> TestCase {
    // cond=[T,F] broadcasts to [2,2], x=[2,2], y=scalar
    let mut rng = rng();
    let ext_c = GlobalId::new(&mut rng);
    let ext_x = GlobalId::new(&mut rng);
    let ext_y = GlobalId::new(&mut rng);
    let (mut graph, imap) = MilliOpGraph::new([ext_c, ext_x, ext_y], &mut rng);
    let out = ops::Where::push_new(&mut graph, imap[&ext_c], imap[&ext_x], imap[&ext_y], &mut rng);
    graph.set_outputs(vec![out]);

    TestCase {
        name: "where_broadcast".into(),
        graph,
        data_sets: vec![TestDataSet {
            label: "broadcast".into(),
            inputs: HashMap::from([
                (ext_c, tensor_bool_shaped(vec![2], &[true, false])),
                (ext_x, tensor_f32_shaped(vec![2, 2], &[1.0, 2.0, 3.0, 4.0])),
                (ext_y, tensor_f32_shaped(vec![1], &[10.0])),
            ]),
            expected_outputs: HashMap::from([(out, tensor_f32_shaped(vec![2, 2], &[1.0, 10.0, 3.0, 10.0]))]),
            tolerance: Tolerance::for_dtype(NumericDType::F32),
        }],
    }
}

// ---------------------------------------------------------------------------
// ClampMin
// ---------------------------------------------------------------------------

fn clampmin_basic() -> TestCase {
    // clamp_min([-2, -1, 0, 1, 2], min=0) → [0, 0, 0, 1, 2]
    let mut rng = rng();
    let ext = GlobalId::new(&mut rng);
    let (mut graph, imap) = MilliOpGraph::new([ext], &mut rng);
    let out = ops::ClampMin::push_new(&mut graph, imap[&ext], 0.0, &mut rng);
    graph.set_outputs(vec![out]);

    TestCase {
        name: "clampmin_basic".into(),
        graph,
        data_sets: vec![TestDataSet {
            label: "5elem".into(),
            inputs: HashMap::from([(ext, tensor_f32_shaped(vec![5], &[-2.0, -1.0, 0.0, 1.0, 2.0]))]),
            expected_outputs: HashMap::from([(out, tensor_f32_shaped(vec![5], &[0.0, 0.0, 0.0, 1.0, 2.0]))]),
            tolerance: Tolerance::for_dtype(NumericDType::F32),
        }],
    }
}

// ---------------------------------------------------------------------------
// Gather
// ---------------------------------------------------------------------------

fn gather_axis0() -> TestCase {
    let mut rng = rng();
    let ext_data = GlobalId::new(&mut rng);
    let ext_idx = GlobalId::new(&mut rng);
    let (mut graph, imap) = MilliOpGraph::new([ext_data, ext_idx], &mut rng);
    let data = ops::ClampMin::push_new(&mut graph, imap[&ext_data], f32::NEG_INFINITY, &mut rng);
    let out = ops::Gather::push_new(&mut graph, data, imap[&ext_idx], 0, &mut rng);
    graph.set_outputs(vec![out]);

    TestCase {
        name: "gather_axis0".into(),
        graph,
        data_sets: vec![TestDataSet {
            label: "3x2_idx2".into(),
            inputs: HashMap::from([
                (ext_data, tensor_f32_shaped(vec![3, 2], &[1.0,2.0,3.0,4.0,5.0,6.0])),
                (ext_idx, tensor_i64_shaped(vec![2], &[0, 2])),
            ]),
            expected_outputs: HashMap::from([(out, tensor_f32_shaped(vec![2, 2], &[1.0,2.0,5.0,6.0]))]),
            tolerance: Tolerance::for_dtype(NumericDType::F32),
        }],
    }
}

fn gather_axis1() -> TestCase {
    let mut rng = rng();
    let ext_data = GlobalId::new(&mut rng);
    let ext_idx = GlobalId::new(&mut rng);
    let (mut graph, imap) = MilliOpGraph::new([ext_data, ext_idx], &mut rng);
    let data = ops::ClampMin::push_new(&mut graph, imap[&ext_data], f32::NEG_INFINITY, &mut rng);
    let out = ops::Gather::push_new(&mut graph, data, imap[&ext_idx], 1, &mut rng);
    graph.set_outputs(vec![out]);

    TestCase {
        name: "gather_axis1".into(),
        graph,
        data_sets: vec![TestDataSet {
            label: "2x3_axis1".into(),
            inputs: HashMap::from([
                (ext_data, tensor_f32_shaped(vec![2, 3], &[1.0,2.0,3.0,4.0,5.0,6.0])),
                (ext_idx, tensor_i64_shaped(vec![2], &[2, 0])),
            ]),
            expected_outputs: HashMap::from([(out, tensor_f32_shaped(vec![2, 2], &[3.0,1.0,6.0,4.0]))]),
            tolerance: Tolerance::for_dtype(NumericDType::F32),
        }],
    }
}

// ---------------------------------------------------------------------------
// TopK
// ---------------------------------------------------------------------------

fn topk_basic() -> TestCase {
    // [3,4] topk(k=2, axis=1, largest=true) → values [3,2], indices [3,2]
    let mut rng = rng();
    let ext = GlobalId::new(&mut rng);
    let (mut graph, imap) = MilliOpGraph::new([ext], &mut rng);
    let k_id = scalar_i64_const(&mut graph, 2, &mut rng);
    let (val_out, idx_out) = ops::TopK::push_new(&mut graph, imap[&ext], k_id, 1, true, true, &mut rng);
    graph.set_outputs(vec![val_out, idx_out]);

    // Input: [[1,4,2,3],[8,5,7,6],[9,10,12,11]]
    // Top-2 along axis=1 (largest, sorted descending):
    // row 0: [4,3] at [1,3]
    // row 1: [8,7] at [0,2]
    // row 2: [12,11] at [2,3]
    TestCase {
        name: "topk_basic".into(),
        graph,
        data_sets: vec![TestDataSet {
            label: "3x4_k2".into(),
            inputs: HashMap::from([(ext, tensor_f32_shaped(vec![3, 4], &[
                1.0, 4.0, 2.0, 3.0,
                8.0, 5.0, 7.0, 6.0,
                9.0, 10.0, 12.0, 11.0,
            ]))]),
            expected_outputs: HashMap::from([
                (val_out, tensor_f32_shaped(vec![3, 2], &[4.0,3.0, 8.0,7.0, 12.0,11.0])),
                (idx_out, tensor_i64_shaped(vec![3, 2], &[1,3, 0,2, 2,3])),
            ]),
            tolerance: Tolerance::for_dtype(NumericDType::F32),
        }],
    }
}

// ---------------------------------------------------------------------------
// Resize (nearest neighbor)
// ---------------------------------------------------------------------------

fn resize_nearest() -> TestCase {
    // [1,1,2,2] resize to [1,1,4,4] via sizes, nearest mode
    let mut rng = rng();
    let ext = GlobalId::new(&mut rng);
    let (mut graph, imap) = MilliOpGraph::new([ext], &mut rng);

    let roi_id = f32_const(&mut graph, vec![], vec![0], &mut rng);
    let scales_id = f32_const(&mut graph, vec![], vec![0], &mut rng);
    let sizes_id = i64_const(&mut graph, vec![1, 1, 4, 4], &mut rng);

    let out = ops::Resize::push_new(
        &mut graph,
        imap[&ext],
        Some(roi_id),
        Some(scales_id),
        Some(sizes_id),
        ops::ResizeMode::Nearest,
        ops::ResizeCoordTransform::Asymmetric,
        ops::ResizeNearestMode::Floor,
        -0.75,
        false,
        false,
        0.0,
        vec![],
        ops::ResizeKeepAspectRatioPolicy::Stretch,
        &mut rng,
    );
    graph.set_outputs(vec![out]);

    // Input: [[1,2],[3,4]] → nearest 2x upscale:
    // [[1,1,2,2],[1,1,2,2],[3,3,4,4],[3,3,4,4]]
    TestCase {
        name: "resize_nearest".into(),
        graph,
        data_sets: vec![TestDataSet {
            label: "2x2_to_4x4".into(),
            inputs: HashMap::from([(ext, tensor_f32_shaped(vec![1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]))]),
            expected_outputs: HashMap::from([(out, tensor_f32_shaped(vec![1, 1, 4, 4], &[
                1.0, 1.0, 2.0, 2.0,
                1.0, 1.0, 2.0, 2.0,
                3.0, 3.0, 4.0, 4.0,
                3.0, 3.0, 4.0, 4.0,
            ]))]),
            tolerance: Tolerance::for_dtype(NumericDType::F32),
        }],
    }
}
