//! Pad test cases for nano lowering verification.

use std::collections::HashMap;

use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::graph::GlobalId;
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::{Pad, PadMode};
use crate::numeric_dtype::NumericDType;

use super::{TestCase, TestDataSet, Tolerance, tensor_f32_shaped, tensor_i64_shaped};

pub fn build_cases() -> Vec<TestCase> {
    vec![
        pad_1d_constant(),
        pad_2d_constant(),
        pad_4d_spatial(),
        pad_with_value(),
    ]
}

fn rng() -> SmallRng {
    SmallRng::seed_from_u64(77)
}

struct PadGraphIds {
    ext_data: GlobalId,
    ext_pads: GlobalId,
    ext_cv: Option<GlobalId>,
    out: GlobalId,
}

fn build_pad_graph(has_constant_value: bool) -> (MilliOpGraph, PadGraphIds) {
    let mut rng = rng();
    let ext_data = GlobalId::new(&mut rng);
    let ext_pads = GlobalId::new(&mut rng);
    let ext_cv = if has_constant_value {
        Some(GlobalId::new(&mut rng))
    } else {
        None
    };

    let mut ext_ids: Vec<GlobalId> = vec![ext_data, ext_pads];
    if let Some(cv) = ext_cv {
        ext_ids.push(cv);
    }

    let (mut graph, input_map) = MilliOpGraph::new(ext_ids.iter().copied(), &mut rng);
    let int_data = input_map[&ext_data];
    let int_pads = input_map[&ext_pads];
    let int_cv = ext_cv.map(|id| input_map[&id]);

    let out = Pad::push_new(
        &mut graph,
        int_data,
        int_pads,
        int_cv,
        None, // no axes
        PadMode::Constant,
        &mut rng,
    );
    graph.set_outputs(vec![out]);
    (
        graph,
        PadGraphIds {
            ext_data,
            ext_pads,
            ext_cv,
            out,
        },
    )
}

fn pad_data_set(
    label: &str,
    ids: &PadGraphIds,
    data: super::TestTensor,
    pads: super::TestTensor,
    cv: Option<super::TestTensor>,
    expected: super::TestTensor,
    tolerance: Tolerance,
) -> TestDataSet {
    let mut inputs = HashMap::from([(ids.ext_data, data), (ids.ext_pads, pads)]);
    if let (Some(ext_cv), Some(cv_tensor)) = (ids.ext_cv, cv) {
        inputs.insert(ext_cv, cv_tensor);
    }
    TestDataSet {
        label: label.to_string(),
        inputs,
        expected_outputs: HashMap::from([(ids.out, expected)]),
        tolerance,
    }
}

// ---------------------------------------------------------------------------
// Reference implementation
// ---------------------------------------------------------------------------

fn pad_ref(data: &[f32], shape: &[usize], pads: &[i64], fill: f32) -> (Vec<f32>, Vec<usize>) {
    let rank = shape.len();
    let mut pb = vec![0usize; rank];
    let mut pe = vec![0usize; rank];
    for d in 0..rank {
        pb[d] = pads[d] as usize;
        pe[d] = pads[rank + d] as usize;
    }
    let out_shape: Vec<usize> = (0..rank).map(|d| shape[d] + pb[d] + pe[d]).collect();
    let out_total: usize = out_shape.iter().product();
    let mut out = vec![fill; out_total];

    // Compute strides for both shapes.
    let mut in_strides = vec![1usize; rank];
    for d in (0..rank.saturating_sub(1)).rev() {
        in_strides[d] = in_strides[d + 1] * shape[d + 1];
    }
    let mut out_strides = vec![1usize; rank];
    for d in (0..rank.saturating_sub(1)).rev() {
        out_strides[d] = out_strides[d + 1] * out_shape[d + 1];
    }

    // Copy interior elements.
    let in_total: usize = shape.iter().product();
    for i in 0..in_total {
        let mut rem = i;
        let mut out_idx = 0;
        for d in 0..rank {
            let coord = rem / in_strides[d];
            rem %= in_strides[d];
            out_idx += (coord + pb[d]) * out_strides[d];
        }
        out[out_idx] = data[i];
    }
    (out, out_shape)
}

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------

/// 1D pad: [5] with pads [2, 3] → [10].
fn pad_1d_constant() -> TestCase {
    let (graph, ids) = build_pad_graph(false);

    let data_vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let pads_vals = vec![2i64, 3];
    let (expected_vals, out_shape) = pad_ref(&data_vals, &[5], &pads_vals, 0.0);

    let data = tensor_f32_shaped(vec![5], &data_vals);
    let pads = tensor_i64_shaped(vec![2], &pads_vals);
    let expected = tensor_f32_shaped(
        out_shape.iter().map(|&x| x as u64).collect(),
        &expected_vals,
    );

    TestCase {
        name: "pad_1d_constant".to_string(),
        graph,
        data_sets: vec![pad_data_set(
            "f32_1d",
            &ids,
            data,
            pads,
            None,
            expected,
            Tolerance::for_dtype(NumericDType::F32),
        )],
    }
}

/// 2D pad: [3,4] with pads [1,2,1,2] → [5,8].
fn pad_2d_constant() -> TestCase {
    let (graph, ids) = build_pad_graph(false);

    let data_vals: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let pads_vals = vec![1i64, 2, 1, 2];
    let (expected_vals, out_shape) = pad_ref(&data_vals, &[3, 4], &pads_vals, 0.0);

    let data = tensor_f32_shaped(vec![3, 4], &data_vals);
    let pads = tensor_i64_shaped(vec![4], &pads_vals);
    let expected = tensor_f32_shaped(
        out_shape.iter().map(|&x| x as u64).collect(),
        &expected_vals,
    );

    TestCase {
        name: "pad_2d_constant".to_string(),
        graph,
        data_sets: vec![pad_data_set(
            "f32_3x4",
            &ids,
            data,
            pads,
            None,
            expected,
            Tolerance::for_dtype(NumericDType::F32),
        )],
    }
}

/// 4D pad on spatial dims only: [1,1,3,3] with pads [0,0,1,1,0,0,1,1] → [1,1,5,5].
fn pad_4d_spatial() -> TestCase {
    let (graph, ids) = build_pad_graph(false);

    let data_vals: Vec<f32> = (0..9).map(|i| (i + 1) as f32).collect();
    let pads_vals = vec![0i64, 0, 1, 1, 0, 0, 1, 1];
    let (expected_vals, out_shape) = pad_ref(&data_vals, &[1, 1, 3, 3], &pads_vals, 0.0);

    let data = tensor_f32_shaped(vec![1, 1, 3, 3], &data_vals);
    let pads = tensor_i64_shaped(vec![8], &pads_vals);
    let expected = tensor_f32_shaped(
        out_shape.iter().map(|&x| x as u64).collect(),
        &expected_vals,
    );

    TestCase {
        name: "pad_4d_spatial".to_string(),
        graph,
        data_sets: vec![pad_data_set(
            "f32_1x1x3x3",
            &ids,
            data,
            pads,
            None,
            expected,
            Tolerance::for_dtype(NumericDType::F32),
        )],
    }
}

/// Pad with non-zero constant value.
fn pad_with_value() -> TestCase {
    let (graph, ids) = build_pad_graph(true);

    let data_vals = vec![1.0, 2.0, 3.0];
    let pads_vals = vec![1i64, 1];
    let (expected_vals, out_shape) = pad_ref(&data_vals, &[3], &pads_vals, -1.0);

    let data = tensor_f32_shaped(vec![3], &data_vals);
    let pads = tensor_i64_shaped(vec![2], &pads_vals);
    let cv = tensor_f32_shaped(vec![1], &[-1.0]);
    let expected = tensor_f32_shaped(
        out_shape.iter().map(|&x| x as u64).collect(),
        &expected_vals,
    );

    TestCase {
        name: "pad_with_value".to_string(),
        graph,
        data_sets: vec![pad_data_set(
            "f32_fill_neg1",
            &ids,
            data,
            pads,
            Some(cv),
            expected,
            Tolerance::for_dtype(NumericDType::F32),
        )],
    }
}
