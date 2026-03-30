//! Dtype discipline tests.
//!
//! These tests verify that the nano lowering and pool_eval paths respect
//! tensor dtypes precisely — no silent truncation to F32, no hardcoded types.
//! Each test uses values specifically chosen to distinguish correct dtype
//! behavior from incorrect truncation.

use std::collections::HashMap;

use half::bf16;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::graph::GlobalId;
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::binary::SimpleBinary;
use crate::milli_graph::ops::{Constant, Conv, ConvAutoPad, MatMul, ReduceMean, ReduceSum};
use crate::numeric_dtype::NumericDType;

use super::{TestCase, TestDataSet, Tolerance, tensor_bf16_shaped, tensor_f64_shaped};

pub fn build_cases() -> Vec<TestCase> {
    vec![
        f64_reduce_sum_precision(),
        f64_add_precision(),
        f64_matmul_precision(),
        f64_reduce_mean(),
        bf16_reduce_sum_quantization(),
        bf16_add_quantization(),
        f64_conv_precision(),
    ]
}

fn rng() -> SmallRng {
    SmallRng::seed_from_u64(7777)
}

// ---------------------------------------------------------------------------
// F64 ReduceSum — values that distinguish F64 from F32 accumulation
// ---------------------------------------------------------------------------

fn f64_reduce_sum_precision() -> TestCase {
    let mut rng = rng();
    let ext_data = GlobalId::new(&mut rng);
    let (mut graph, input_map) = MilliOpGraph::new([ext_data], &mut rng);
    let int_data = input_map[&ext_data];

    let axes_id = Constant::from_vec(&mut graph, vec![0i64], &mut rng);
    let out = ReduceSum::push_new(&mut graph, int_data, Some(axes_id), true, false, &mut rng);
    graph.set_outputs(vec![out]);

    // 2^24 = 16777216.0. F32 ULP at this value is 2.0, so 16777216 + 1 = 16777216 in F32.
    // F64 ULP at this value is ~4e-9, so 16777216 + 1 = 16777217 exactly in F64.
    // Sequential accumulation: (16777216 + 1) - 16777216
    //   F64: 16777217 - 16777216 = 1.0
    //   F32: 16777216 - 16777216 = 0.0
    TestCase {
        name: "f64_reduce_sum_precision".to_string(),
        graph,
        data_sets: vec![TestDataSet {
            label: "cancellation".to_string(),
            inputs: HashMap::from([(
                ext_data,
                tensor_f64_shaped(vec![3], &[16777216.0, 1.0, -16777216.0]),
            )]),
            expected_outputs: HashMap::from([(out, tensor_f64_shaped(vec![1], &[1.0]))]),
            tolerance: Tolerance::for_dtype(NumericDType::F64),
        }],
    }
}

// ---------------------------------------------------------------------------
// F64 Add — values where F32 truncation loses information
// ---------------------------------------------------------------------------

fn f64_add_precision() -> TestCase {
    let mut rng = rng();
    let ext_a = GlobalId::new(&mut rng);
    let ext_b = GlobalId::new(&mut rng);
    let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
    let int_a = input_map[&ext_a];
    let int_b = input_map[&ext_b];
    let out = SimpleBinary::add(&mut graph, int_a, int_b, &mut rng);
    graph.set_outputs(vec![out]);

    // 16777216.0 + 1.0: In F64 = 16777217.0 (exact). In F32 = 16777216.0 (ULP swallows 1).
    // 0.1 + 0.2: Both F32 and F64 give ~0.3, but the exact bits differ.
    // Use the 2^24 case to clearly distinguish F32 from F64.
    TestCase {
        name: "f64_add_precision".to_string(),
        graph,
        data_sets: vec![TestDataSet {
            label: "f32_ulp_boundary".to_string(),
            inputs: HashMap::from([
                (ext_a, tensor_f64_shaped(vec![1], &[16777216.0])),
                (ext_b, tensor_f64_shaped(vec![1], &[1.0])),
            ]),
            expected_outputs: HashMap::from([(out, tensor_f64_shaped(vec![1], &[16777217.0]))]),
            tolerance: Tolerance::for_dtype(NumericDType::F64),
        }],
    }
}

// ---------------------------------------------------------------------------
// F64 MatMul — dot product where F32 intermediate loses precision
// ---------------------------------------------------------------------------

fn f64_matmul_precision() -> TestCase {
    let mut rng = rng();
    let ext_a = GlobalId::new(&mut rng);
    let ext_b = GlobalId::new(&mut rng);
    let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
    let int_a = input_map[&ext_a];
    let int_b = input_map[&ext_b];
    let out =
        MatMul::push_new_default_precision(&mut graph, int_a, int_b, NumericDType::F64, &mut rng);
    graph.set_outputs(vec![out]);

    // [16777216.0, 1.0] @ [1.0, 1.0]^T = 16777216 + 1 = 16777217.0 in F64
    // In F32 accumulation: 16777216 + 1 = 16777216
    TestCase {
        name: "f64_matmul_precision".to_string(),
        graph,
        data_sets: vec![TestDataSet {
            label: "precision_sensitive_dot".to_string(),
            inputs: HashMap::from([
                (ext_a, tensor_f64_shaped(vec![1, 2], &[16777216.0, 1.0])),
                (ext_b, tensor_f64_shaped(vec![2, 1], &[1.0, 1.0])),
            ]),
            expected_outputs: HashMap::from([(out, tensor_f64_shaped(vec![1, 1], &[16777217.0]))]),
            tolerance: Tolerance::for_dtype(NumericDType::F64),
        }],
    }
}

// ---------------------------------------------------------------------------
// F64 ReduceMean — tests the full pipeline including the extent divisor
// ---------------------------------------------------------------------------

fn f64_reduce_mean() -> TestCase {
    let mut rng = rng();
    let ext_data = GlobalId::new(&mut rng);
    let (mut graph, input_map) = MilliOpGraph::new([ext_data], &mut rng);
    let int_data = input_map[&ext_data];

    let axes_id = Constant::from_vec(&mut graph, vec![1i64], &mut rng);
    let out = ReduceMean::push_new(&mut graph, int_data, Some(axes_id), true, false, &mut rng);
    graph.set_outputs(vec![out]);

    // Mean of [16777216.0, 16777217.0, 16777218.0] along axis 1
    // = (16777216 + 16777217 + 16777218) / 3 = 50331651 / 3 = 16777217.0
    // In F32: 16777217 rounds to 16777216, 16777218 rounds to 16777218
    //   sum = 16777216 + 16777216 + 16777218 = 50331650, mean ≈ 16777216.666...
    // In F64: all values exact, mean = 16777217.0 exactly
    let row = [16777216.0, 16777217.0, 16777218.0];
    TestCase {
        name: "f64_reduce_mean".to_string(),
        graph,
        data_sets: vec![TestDataSet {
            label: "precision".to_string(),
            inputs: HashMap::from([(ext_data, tensor_f64_shaped(vec![1, 3], &row))]),
            expected_outputs: HashMap::from([(out, tensor_f64_shaped(vec![1, 1], &[16777217.0]))]),
            tolerance: Tolerance::for_dtype(NumericDType::F64),
        }],
    }
}

// ---------------------------------------------------------------------------
// BF16 ReduceSum — verifies accumulation quantization behavior
// ---------------------------------------------------------------------------

fn bf16_reduce_sum_quantization() -> TestCase {
    let mut rng = rng();
    let ext_data = GlobalId::new(&mut rng);
    let (mut graph, input_map) = MilliOpGraph::new([ext_data], &mut rng);
    let int_data = input_map[&ext_data];

    let axes_id = Constant::from_vec(&mut graph, vec![0i64], &mut rng);
    let out = ReduceSum::push_new(&mut graph, int_data, Some(axes_id), true, false, &mut rng);
    graph.set_outputs(vec![out]);

    // BF16 currently accumulates in F32 (design choice in lower_reduce).
    // 256.0 + 0.5 + 0.5 = 257.0
    // BF16 values: 256.0 is exact, 0.5 is exact.
    // F32 sum: 257.0 (exact in F32), cast to BF16 = 256.0 (ULP at 256 is 2.0 in BF16)
    // Pure BF16 sum: 256 + 0.5 = 256 (rounds), + 0.5 = 256
    //
    // The F32 intermediate accumulation gives 257.0, which rounds to 256.0 in BF16.
    // This test documents the current behavior (F32 accumulation for BF16 inputs).
    let vals: Vec<bf16> = vec![
        bf16::from_f32(256.0),
        bf16::from_f32(0.5),
        bf16::from_f32(0.5),
    ];
    // F32 accumulation: 256 + 0.5 + 0.5 = 257.0 → BF16 output = 256.0
    let expected = vec![bf16::from_f32(256.0)];

    TestCase {
        name: "bf16_reduce_sum_quantization".to_string(),
        graph,
        data_sets: vec![TestDataSet {
            label: "f32_accumulation".to_string(),
            inputs: HashMap::from([(ext_data, tensor_bf16_shaped(vec![3], &vals))]),
            expected_outputs: HashMap::from([(out, tensor_bf16_shaped(vec![1], &expected))]),
            tolerance: Tolerance {
                atol: 0.0,
                rtol: 0.0,
            },
        }],
    }
}

// ---------------------------------------------------------------------------
// BF16 Add — verifies BF16 compute precision in elementwise ops
// ---------------------------------------------------------------------------

fn bf16_add_quantization() -> TestCase {
    let mut rng = rng();
    let ext_a = GlobalId::new(&mut rng);
    let ext_b = GlobalId::new(&mut rng);
    let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
    let int_a = input_map[&ext_a];
    let int_b = input_map[&ext_b];
    let out = SimpleBinary::add(&mut graph, int_a, int_b, &mut rng);
    graph.set_outputs(vec![out]);

    // 256.0 + 1.0 in BF16:
    // BF16 ULP at 256.0 = 2.0, so 256.0 + 1.0 = 256.0 (rounds to even).
    // This verifies the add respects BF16 compute precision.
    TestCase {
        name: "bf16_add_quantization".to_string(),
        graph,
        data_sets: vec![TestDataSet {
            label: "below_ulp".to_string(),
            inputs: HashMap::from([
                (ext_a, tensor_bf16_shaped(vec![1], &[bf16::from_f32(256.0)])),
                (ext_b, tensor_bf16_shaped(vec![1], &[bf16::from_f32(1.0)])),
            ]),
            expected_outputs: HashMap::from([(
                out,
                tensor_bf16_shaped(vec![1], &[bf16::from_f32(256.0)]),
            )]),
            tolerance: Tolerance {
                atol: 0.0,
                rtol: 0.0,
            },
        }],
    }
}

// ---------------------------------------------------------------------------
// F64 Conv — exposes hardcoded F32 in conv nano lowering
// ---------------------------------------------------------------------------

fn f64_conv_precision() -> TestCase {
    let mut rng = rng();
    let ext_input = GlobalId::new(&mut rng);
    let ext_weight = GlobalId::new(&mut rng);
    let (mut graph, input_map) = MilliOpGraph::new([ext_input, ext_weight], &mut rng);
    let int_input = input_map[&ext_input];
    let int_weight = input_map[&ext_weight];

    let out = Conv::push_new(
        &mut graph,
        int_input,
        int_weight,
        None, // no bias
        ConvAutoPad::NotSet,
        vec![1, 1],       // dilations
        1,                // group
        vec![1, 1],       // kernel_shape (1x1 conv = pointwise multiply)
        vec![0, 0, 0, 0], // no padding
        vec![1, 1],       // strides
        &mut rng,
    );
    graph.set_outputs(vec![out]);

    // 1x1 conv with F64 input [1, 1, 1, 1] and weight [1, 1, 1, 1]:
    // output = input * weight = input value.
    // Use 16777217.0 which is exact in F64 but rounds to 16777216.0 in F32.
    // If the conv lowering truncates to F32, the output will be 16777216.0.
    TestCase {
        name: "f64_conv_precision".to_string(),
        graph,
        data_sets: vec![TestDataSet {
            label: "f32_truncation".to_string(),
            inputs: HashMap::from([
                (
                    ext_input,
                    tensor_f64_shaped(vec![1, 1, 1, 1], &[16777217.0]),
                ),
                (ext_weight, tensor_f64_shaped(vec![1, 1, 1, 1], &[1.0])),
            ]),
            expected_outputs: HashMap::from([(
                out,
                tensor_f64_shaped(vec![1, 1, 1, 1], &[16777217.0]),
            )]),
            tolerance: Tolerance::for_dtype(NumericDType::F64),
        }],
    }
}
