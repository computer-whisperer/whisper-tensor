//! Conv test cases for nano lowering verification.

use std::collections::HashMap;

use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::graph::GlobalId;
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::{Conv, ConvAutoPad};
use crate::numeric_dtype::NumericDType;

use super::{TestCase, TestDataSet, Tolerance, tensor_f32_shaped};

pub fn build_cases() -> Vec<TestCase> {
    vec![
        conv_3x3_no_pad(),
        conv_3x3_same_pad(),
        conv_1x1(),
        conv_3x3_stride2_no_pad(),
        conv_with_bias(),
    ]
}

fn rng() -> SmallRng {
    SmallRng::seed_from_u64(42)
}

struct ConvGraphIds {
    ext_input: GlobalId,
    ext_weight: GlobalId,
    ext_bias: Option<GlobalId>,
    out: GlobalId,
}

#[allow(clippy::too_many_arguments)]
fn build_conv_graph(
    auto_pad: ConvAutoPad,
    dilations: Vec<i64>,
    group: i64,
    kernel_shape: Vec<i64>,
    pads: Vec<i64>,
    strides: Vec<i64>,
    has_bias: bool,
) -> (MilliOpGraph, ConvGraphIds) {
    let mut rng = rng();
    let ext_input = GlobalId::new(&mut rng);
    let ext_weight = GlobalId::new(&mut rng);
    let ext_bias = if has_bias {
        Some(GlobalId::new(&mut rng))
    } else {
        None
    };

    let mut ext_ids: Vec<GlobalId> = vec![ext_input, ext_weight];
    if let Some(b) = ext_bias {
        ext_ids.push(b);
    }

    let (mut graph, input_map) = MilliOpGraph::new(ext_ids.iter().copied(), &mut rng);
    let int_input = input_map[&ext_input];
    let int_weight = input_map[&ext_weight];
    let int_bias = ext_bias.map(|b| input_map[&b]);

    let out = Conv::push_new(
        &mut graph,
        int_input,
        int_weight,
        int_bias,
        auto_pad,
        dilations,
        group,
        kernel_shape,
        pads,
        strides,
        &mut rng,
    );
    graph.set_outputs(vec![out]);
    (
        graph,
        ConvGraphIds {
            ext_input,
            ext_weight,
            ext_bias,
            out,
        },
    )
}

fn conv_data_set(
    label: &str,
    ids: &ConvGraphIds,
    input: super::TestTensor,
    weight: super::TestTensor,
    bias: Option<super::TestTensor>,
    expected: super::TestTensor,
    tolerance: Tolerance,
) -> TestDataSet {
    let mut inputs = HashMap::from([(ids.ext_input, input), (ids.ext_weight, weight)]);
    if let (Some(ext_b), Some(b)) = (ids.ext_bias, bias) {
        inputs.insert(ext_b, b);
    }
    TestDataSet {
        label: label.to_string(),
        inputs,
        expected_outputs: HashMap::from([(ids.out, expected)]),
        tolerance,
    }
}

// ---------------------------------------------------------------------------
// Helper: manual 2D convolution for computing expected outputs
// ---------------------------------------------------------------------------

fn conv2d_ref(
    input: &[f32],
    in_shape: [usize; 4], // [N, C_in, IH, IW]
    weight: &[f32],
    w_shape: [usize; 4], // [C_out, C_in, KH, KW]
    bias: Option<&[f32]>,
    pad: [usize; 4],    // [top, left, bottom, right]
    stride: [usize; 2], // [sh, sw]
) -> (Vec<f32>, [usize; 4]) {
    let [n, c_in, ih, iw] = in_shape;
    let [c_out, _w_cin, kh, kw] = w_shape;
    let [pt, pl, pb, pr] = pad;
    let [sh, sw] = stride;

    let oh = (ih + pt + pb - kh) / sh + 1;
    let ow = (iw + pl + pr - kw) / sw + 1;
    let out_shape = [n, c_out, oh, ow];
    let mut out = vec![0.0f32; n * c_out * oh * ow];

    for ni in 0..n {
        for co in 0..c_out {
            for ohi in 0..oh {
                for owi in 0..ow {
                    let mut sum = 0.0f32;
                    for ci in 0..c_in {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let ih_pos = (ohi * sh + ki) as isize - pt as isize;
                                let iw_pos = (owi * sw + kj) as isize - pl as isize;
                                if ih_pos >= 0
                                    && ih_pos < ih as isize
                                    && iw_pos >= 0
                                    && iw_pos < iw as isize
                                {
                                    let in_idx = ni * c_in * ih * iw
                                        + ci * ih * iw
                                        + ih_pos as usize * iw
                                        + iw_pos as usize;
                                    let w_idx = co * c_in * kh * kw + ci * kh * kw + ki * kw + kj;
                                    sum += input[in_idx] * weight[w_idx];
                                }
                            }
                        }
                    }
                    if let Some(b) = bias {
                        sum += b[co];
                    }
                    let out_idx = ni * c_out * oh * ow + co * oh * ow + ohi * ow + owi;
                    out[out_idx] = sum;
                }
            }
        }
    }
    (out, out_shape)
}

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------

/// 3×3 conv, no padding, stride=1. Input [1,1,5,5], weight [1,1,3,3] → [1,1,3,3].
fn conv_3x3_no_pad() -> TestCase {
    let (graph, ids) = build_conv_graph(
        ConvAutoPad::Valid,
        vec![],
        1,
        vec![3, 3],
        vec![],
        vec![1, 1],
        false,
    );

    #[rustfmt::skip]
    let input_vals: Vec<f32> = (0..25).map(|i| i as f32).collect();
    #[rustfmt::skip]
    let weight_vals = vec![
        1.0, 0.0, -1.0,
        2.0, 0.0, -2.0,
        1.0, 0.0, -1.0,
    ];

    let (expected_vals, out_shape) = conv2d_ref(
        &input_vals,
        [1, 1, 5, 5],
        &weight_vals,
        [1, 1, 3, 3],
        None,
        [0, 0, 0, 0],
        [1, 1],
    );

    let input = tensor_f32_shaped(vec![1, 1, 5, 5], &input_vals);
    let weight = tensor_f32_shaped(vec![1, 1, 3, 3], &weight_vals);
    let expected = tensor_f32_shaped(
        out_shape.iter().map(|&x| x as u64).collect(),
        &expected_vals,
    );

    TestCase {
        name: "conv_3x3_no_pad".to_string(),
        graph,
        data_sets: vec![conv_data_set(
            "f32_1x1x5x5",
            &ids,
            input,
            weight,
            None,
            expected,
            Tolerance::for_dtype(NumericDType::F32),
        )],
    }
}

/// 3×3 conv, same padding (SameUpper), stride=1. Input [1,1,5,5] → [1,1,5,5].
fn conv_3x3_same_pad() -> TestCase {
    let (graph, ids) = build_conv_graph(
        ConvAutoPad::SameUpper,
        vec![],
        1,
        vec![3, 3],
        vec![],
        vec![1, 1],
        false,
    );

    let input_vals: Vec<f32> = (0..25).map(|i| (i as f32) * 0.1).collect();
    let weight_vals: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

    // SameUpper with stride=1, kernel=3: pad_top=1, pad_left=1, pad_bottom=1, pad_right=1
    let (expected_vals, out_shape) = conv2d_ref(
        &input_vals,
        [1, 1, 5, 5],
        &weight_vals,
        [1, 1, 3, 3],
        None,
        [1, 1, 1, 1],
        [1, 1],
    );

    let input = tensor_f32_shaped(vec![1, 1, 5, 5], &input_vals);
    let weight = tensor_f32_shaped(vec![1, 1, 3, 3], &weight_vals);
    let expected = tensor_f32_shaped(
        out_shape.iter().map(|&x| x as u64).collect(),
        &expected_vals,
    );

    TestCase {
        name: "conv_3x3_same_pad".to_string(),
        graph,
        data_sets: vec![conv_data_set(
            "f32_1x1x5x5_same",
            &ids,
            input,
            weight,
            None,
            expected,
            Tolerance::for_dtype(NumericDType::F32),
        )],
    }
}

/// 1×1 conv, no padding. Input [1,3,4,4], weight [2,3,1,1] → [1,2,4,4].
fn conv_1x1() -> TestCase {
    let (graph, ids) = build_conv_graph(
        ConvAutoPad::Valid,
        vec![],
        1,
        vec![1, 1],
        vec![],
        vec![1, 1],
        false,
    );

    let input_vals: Vec<f32> = (0..48).map(|i| (i as f32) * 0.01).collect();
    // 2 output channels, 3 input channels, 1×1 kernel
    let weight_vals: Vec<f32> = vec![
        1.0, 2.0, 3.0, // co=0
        -1.0, 0.5, 0.0, // co=1
    ];

    let (expected_vals, out_shape) = conv2d_ref(
        &input_vals,
        [1, 3, 4, 4],
        &weight_vals,
        [2, 3, 1, 1],
        None,
        [0, 0, 0, 0],
        [1, 1],
    );

    let input = tensor_f32_shaped(vec![1, 3, 4, 4], &input_vals);
    let weight = tensor_f32_shaped(vec![2, 3, 1, 1], &weight_vals);
    let expected = tensor_f32_shaped(
        out_shape.iter().map(|&x| x as u64).collect(),
        &expected_vals,
    );

    TestCase {
        name: "conv_1x1".to_string(),
        graph,
        data_sets: vec![conv_data_set(
            "f32_1x3x4x4",
            &ids,
            input,
            weight,
            None,
            expected,
            Tolerance::for_dtype(NumericDType::F32),
        )],
    }
}

/// 3×3 conv, stride=2, no padding. Input [1,1,6,6], weight [1,1,3,3] → [1,1,2,2].
fn conv_3x3_stride2_no_pad() -> TestCase {
    let (graph, ids) = build_conv_graph(
        ConvAutoPad::Valid,
        vec![],
        1,
        vec![3, 3],
        vec![],
        vec![2, 2],
        false,
    );

    let input_vals: Vec<f32> = (0..36).map(|i| i as f32).collect();
    let weight_vals = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    let (expected_vals, out_shape) = conv2d_ref(
        &input_vals,
        [1, 1, 6, 6],
        &weight_vals,
        [1, 1, 3, 3],
        None,
        [0, 0, 0, 0],
        [2, 2],
    );

    let input = tensor_f32_shaped(vec![1, 1, 6, 6], &input_vals);
    let weight = tensor_f32_shaped(vec![1, 1, 3, 3], &weight_vals);
    let expected = tensor_f32_shaped(
        out_shape.iter().map(|&x| x as u64).collect(),
        &expected_vals,
    );

    TestCase {
        name: "conv_3x3_stride2_no_pad".to_string(),
        graph,
        data_sets: vec![conv_data_set(
            "f32_1x1x6x6_s2",
            &ids,
            input,
            weight,
            None,
            expected,
            Tolerance::for_dtype(NumericDType::F32),
        )],
    }
}

/// 3×3 conv with bias, same padding. Input [1,2,4,4], weight [2,2,3,3], bias [2] → [1,2,4,4].
fn conv_with_bias() -> TestCase {
    let (graph, ids) = build_conv_graph(
        ConvAutoPad::SameUpper,
        vec![],
        1,
        vec![3, 3],
        vec![],
        vec![1, 1],
        true,
    );

    let input_vals: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
    let weight_vals: Vec<f32> = (0..36).map(|i| (i as f32) * 0.01 - 0.18).collect();
    let bias_vals = vec![0.5, -0.5];

    let (expected_vals, out_shape) = conv2d_ref(
        &input_vals,
        [1, 2, 4, 4],
        &weight_vals,
        [2, 2, 3, 3],
        Some(&bias_vals),
        [1, 1, 1, 1],
        [1, 1],
    );

    let input = tensor_f32_shaped(vec![1, 2, 4, 4], &input_vals);
    let weight = tensor_f32_shaped(vec![2, 2, 3, 3], &weight_vals);
    let bias = tensor_f32_shaped(vec![2], &bias_vals);
    let expected = tensor_f32_shaped(
        out_shape.iter().map(|&x| x as u64).collect(),
        &expected_vals,
    );

    TestCase {
        name: "conv_with_bias".to_string(),
        graph,
        data_sets: vec![conv_data_set(
            "f32_1x2x4x4_bias",
            &ids,
            input,
            weight,
            Some(bias),
            expected,
            Tolerance::for_dtype(NumericDType::F32),
        )],
    }
}
