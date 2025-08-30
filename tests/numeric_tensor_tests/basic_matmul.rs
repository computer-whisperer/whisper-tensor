use crate::numeric_tensor_tests::{test_eq_bf16, test_eq_f16, test_eq_f32};
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::dtype::DType;
use whisper_tensor::numeric_tensor::NumericTensor;

pub fn test_matmul_2_3_bf16(backend: &mut EvalBackend) {
    let tensor_a = NumericTensor::from_vec(vec![1.0, -2.5, 3.0, 4.0, 0.5, -1.0])
        .to_dyn_rank()
        .reshape(vec![2, 3], backend)
        .unwrap()
        .cast(DType::BF16, backend)
        .unwrap();
    let tensor_b = NumericTensor::from_vec(vec![1.0, 2.0, 0.25, -0.75, 3.5, 0.0])
        .to_dyn_rank()
        .reshape(vec![3, 2], backend)
        .unwrap()
        .cast(DType::BF16, backend)
        .unwrap();
    let result = NumericTensor::matmul(&tensor_a, &tensor_b, Some(DType::F32), backend).unwrap();
    let correct = NumericTensor::from_vec(vec![10.875, 3.875, 0.625, 7.625])
        .to_dyn_rank()
        .reshape(vec![2, 2], backend)
        .unwrap()
        .cast(DType::BF16, backend)
        .unwrap();
    test_eq_bf16(result, correct);
}

pub fn test_matmul_2_3_f16(backend: &mut EvalBackend) {
    let tensor_a = NumericTensor::from_vec(vec![1.0, -2.5, 3.0, 4.0, 0.5, -1.0])
        .to_dyn_rank()
        .reshape(vec![2, 3], backend)
        .unwrap()
        .cast(DType::F16, backend)
        .unwrap();
    let tensor_b = NumericTensor::from_vec(vec![1.0, 2.0, 0.25, -0.75, 3.5, 0.0])
        .to_dyn_rank()
        .reshape(vec![3, 2], backend)
        .unwrap()
        .cast(DType::F16, backend)
        .unwrap();
    let result = NumericTensor::matmul(&tensor_a, &tensor_b, Some(DType::F32), backend).unwrap();
    let correct = NumericTensor::from_vec(vec![10.875, 3.875, 0.625, 7.625])
        .to_dyn_rank()
        .reshape(vec![2, 2], backend)
        .unwrap()
        .cast(DType::F16, backend)
        .unwrap();
    test_eq_f16(result, correct);
}

pub fn test_matmul_2_3_fp32(backend: &mut EvalBackend) {
    let tensor_a = NumericTensor::from_vec(vec![1.0, -2.5, 3.0, 4.0, 0.5, -1.0])
        .to_dyn_rank()
        .reshape(vec![2, 3], backend)
        .unwrap()
        .cast(DType::F32, backend)
        .unwrap();
    let tensor_b = NumericTensor::from_vec(vec![1.0, 2.0, 0.25, -0.75, 3.5, 0.0])
        .to_dyn_rank()
        .reshape(vec![3, 2], backend)
        .unwrap()
        .cast(DType::F32, backend)
        .unwrap();
    let result = NumericTensor::matmul(&tensor_a, &tensor_b, Some(DType::F32), backend).unwrap();
    let correct = NumericTensor::from_vec(vec![10.875, 3.875, 0.625, 7.625])
        .to_dyn_rank()
        .reshape(vec![2, 2], backend)
        .unwrap()
        .cast(DType::F32, backend)
        .unwrap();
    test_eq_f32(result, correct);
}

#[allow(clippy::approx_constant)]
pub fn test_matmul_3_3_bf16(backend: &mut EvalBackend) {
    let tensor_a = NumericTensor::from_vec(vec![
        1.000976562,
        0.9990234375,
        -2.0,
        3.14159,
        2.71828,
        -1.41421,
        0.0001,
        100.0,
        -0.01,
    ])
    .to_dyn_rank()
    .reshape(vec![3, 3], backend)
    .unwrap()
    .cast(DType::BF16, backend)
    .unwrap();
    let tensor_b =
        NumericTensor::from_vec(vec![-1.0, 0.5, 2.0, 0.125, -0.125, 3.0, 10.0, -20.0, 0.25])
            .to_dyn_rank()
            .reshape(vec![3, 3], backend)
            .unwrap()
            .cast(DType::BF16, backend)
            .unwrap();
    let result = NumericTensor::matmul(&tensor_a, &tensor_b, Some(DType::F32), backend).unwrap();
    let correct = NumericTensor::from_vec(vec![
        -20.87609863,
        40.37561035,
        4.499023438,
        -16.94390488,
        29.5152092,
        14.08446789,
        12.39989948,
        -12.2999506,
        299.9977112,
    ])
    .to_dyn_rank()
    .reshape(vec![3, 3], backend)
    .unwrap()
    .cast(DType::BF16, backend)
    .unwrap();
    test_eq_bf16(result, correct);
}

#[allow(clippy::approx_constant)]
pub fn test_matmul_3_3_f16(backend: &mut EvalBackend) {
    let tensor_a = NumericTensor::from_vec(vec![
        1.000976562,
        0.9990234375,
        -2.0,
        3.14159,
        2.71828,
        -1.41421,
        0.0001,
        100.0,
        -0.01,
    ])
    .to_dyn_rank()
    .reshape(vec![3, 3], backend)
    .unwrap()
    .cast(DType::F16, backend)
    .unwrap();
    let tensor_b =
        NumericTensor::from_vec(vec![-1.0, 0.5, 2.0, 0.125, -0.125, 3.0, 10.0, -20.0, 0.25])
            .to_dyn_rank()
            .reshape(vec![3, 3], backend)
            .unwrap()
            .cast(DType::F16, backend)
            .unwrap();
    let result = NumericTensor::matmul(&tensor_a, &tensor_b, Some(DType::F32), backend).unwrap();
    let correct = NumericTensor::from_vec(vec![
        -20.87609863,
        40.37561035,
        4.499023438,
        -16.94390488,
        29.5152092,
        14.08446789,
        12.39989948,
        -12.2999506,
        299.9977112,
    ])
    .to_dyn_rank()
    .reshape(vec![3, 3], backend)
    .unwrap()
    .cast(DType::F16, backend)
    .unwrap();
    test_eq_f16(result, correct);
}

#[allow(clippy::approx_constant)]
pub fn test_matmul_3_3_fp32(backend: &mut EvalBackend) {
    let tensor_a = NumericTensor::from_vec(vec![
        1.000976562,
        0.9990234375,
        -2.0,
        3.14159,
        2.71828,
        -1.41421,
        0.0001,
        100.0,
        -0.01,
    ])
    .to_dyn_rank()
    .reshape(vec![3, 3], backend)
    .unwrap()
    .cast(DType::F32, backend)
    .unwrap();
    let tensor_b =
        NumericTensor::from_vec(vec![-1.0, 0.5, 2.0, 0.125, -0.125, 3.0, 10.0, -20.0, 0.25])
            .to_dyn_rank()
            .reshape(vec![3, 3], backend)
            .unwrap()
            .cast(DType::F32, backend)
            .unwrap();
    let result = NumericTensor::matmul(&tensor_a, &tensor_b, Some(DType::F32), backend).unwrap();
    let correct = NumericTensor::from_vec(vec![
        -20.87609863,
        40.37561035,
        4.499023438,
        -16.94390488,
        29.5152092,
        14.08446789,
        12.39989948,
        -12.2999506,
        299.9977112,
    ])
    .to_dyn_rank()
    .reshape(vec![3, 3], backend)
    .unwrap()
    .cast(DType::F32, backend)
    .unwrap();
    test_eq_f32(result, correct);
}

pub fn test_matmul_1_4_4_1_fp32(backend: &mut EvalBackend) {
    let tensor_a = NumericTensor::from_vec(vec![0.1, -0.2, 0.3, -0.4])
        .to_dyn_rank()
        .reshape(vec![1, 4], backend)
        .unwrap()
        .cast(DType::F32, backend)
        .unwrap();
    let tensor_b = NumericTensor::from_vec(vec![10.0, 20.0, -30.0, 40.0])
        .to_dyn_rank()
        .reshape(vec![4, 1], backend)
        .unwrap()
        .cast(DType::F32, backend)
        .unwrap();
    let result = NumericTensor::matmul(&tensor_a, &tensor_b, Some(DType::F32), backend).unwrap();
    let correct = NumericTensor::from_vec(vec![-28.0])
        .to_dyn_rank()
        .reshape(vec![1, 1], backend)
        .unwrap()
        .cast(DType::F32, backend)
        .unwrap();
    test_eq_f32(result, correct);
}

pub fn test_matmul_1_4_4_1_f16(backend: &mut EvalBackend) {
    let tensor_a = NumericTensor::from_vec(vec![0.1, -0.2, 0.3, -0.4])
        .to_dyn_rank()
        .reshape(vec![1, 4], backend)
        .unwrap()
        .cast(DType::F16, backend)
        .unwrap();
    let tensor_b = NumericTensor::from_vec(vec![10.0, 20.0, -30.0, 40.0])
        .to_dyn_rank()
        .reshape(vec![4, 1], backend)
        .unwrap()
        .cast(DType::F16, backend)
        .unwrap();
    let result = NumericTensor::matmul(&tensor_a, &tensor_b, Some(DType::F32), backend).unwrap();
    let correct = NumericTensor::from_vec(vec![-28.0])
        .to_dyn_rank()
        .reshape(vec![1, 1], backend)
        .unwrap()
        .cast(DType::F16, backend)
        .unwrap();
    test_eq_f16(result, correct);
}

pub fn test_matmul_1_4_4_1_bf16(backend: &mut EvalBackend) {
    let tensor_a = NumericTensor::from_vec(vec![0.1, -0.2, 0.3, -0.4])
        .to_dyn_rank()
        .reshape(vec![1, 4], backend)
        .unwrap()
        .cast(DType::BF16, backend)
        .unwrap();
    let tensor_b = NumericTensor::from_vec(vec![10.0, 20.0, -30.0, 40.0])
        .to_dyn_rank()
        .reshape(vec![4, 1], backend)
        .unwrap()
        .cast(DType::BF16, backend)
        .unwrap();
    let result = NumericTensor::matmul(&tensor_a, &tensor_b, Some(DType::F32), backend).unwrap();
    let correct = NumericTensor::from_vec(vec![-28.0])
        .to_dyn_rank()
        .reshape(vec![1, 1], backend)
        .unwrap()
        .cast(DType::BF16, backend)
        .unwrap();
    test_eq_bf16(result, correct);
}

pub fn test_matmul_4_1_1_4_fp32(backend: &mut EvalBackend) {
    let tensor_a = NumericTensor::from_vec(vec![1.5, -2.5, 0.03125, 1.0078125])
        .to_dyn_rank()
        .reshape(vec![4, 1], backend)
        .unwrap()
        .cast(DType::F32, backend)
        .unwrap();
    let tensor_b = NumericTensor::from_vec(vec![-0.5, 2.0, -3.0, 5.5])
        .to_dyn_rank()
        .reshape(vec![1, 4], backend)
        .unwrap()
        .cast(DType::F32, backend)
        .unwrap();
    let result = NumericTensor::matmul(&tensor_a, &tensor_b, Some(DType::F32), backend).unwrap();
    let correct = NumericTensor::from_vec(vec![
        -0.75,
        3.0,
        -4.5,
        8.25,
        1.25,
        -5.0,
        7.5,
        -13.75,
        -0.015625,
        0.0625,
        -0.09375,
        0.171875,
        -0.50390625,
        2.015625,
        -3.0234375,
        5.54296875,
    ])
    .to_dyn_rank()
    .reshape(vec![4, 4], backend)
    .unwrap()
    .cast(DType::F32, backend)
    .unwrap();
    test_eq_f32(result, correct);
}

pub fn test_matmul_4_1_1_4_f16(backend: &mut EvalBackend) {
    let tensor_a = NumericTensor::from_vec(vec![1.5, -2.5, 0.03125, 1.0078125])
        .to_dyn_rank()
        .reshape(vec![4, 1], backend)
        .unwrap()
        .cast(DType::F16, backend)
        .unwrap();
    let tensor_b = NumericTensor::from_vec(vec![-0.5, 2.0, -3.0, 5.5])
        .to_dyn_rank()
        .reshape(vec![1, 4], backend)
        .unwrap()
        .cast(DType::F16, backend)
        .unwrap();
    let result = NumericTensor::matmul(&tensor_a, &tensor_b, Some(DType::F32), backend).unwrap();
    let correct = NumericTensor::from_vec(vec![
        -0.75,
        3.0,
        -4.5,
        8.25,
        1.25,
        -5.0,
        7.5,
        -13.75,
        -0.015625,
        0.0625,
        -0.09375,
        0.171875,
        -0.50390625,
        2.015625,
        -3.0234375,
        5.54296875,
    ])
    .to_dyn_rank()
    .reshape(vec![4, 4], backend)
    .unwrap()
    .cast(DType::F16, backend)
    .unwrap();
    test_eq_f16(result, correct);
}

pub fn test_matmul_4_1_1_4_bf16(backend: &mut EvalBackend) {
    let tensor_a = NumericTensor::from_vec(vec![1.5, -2.5, 0.03125, 1.0078125])
        .to_dyn_rank()
        .reshape(vec![4, 1], backend)
        .unwrap()
        .cast(DType::BF16, backend)
        .unwrap();
    let tensor_b = NumericTensor::from_vec(vec![-0.5, 2.0, -3.0, 5.5])
        .to_dyn_rank()
        .reshape(vec![1, 4], backend)
        .unwrap()
        .cast(DType::BF16, backend)
        .unwrap();
    let result = NumericTensor::matmul(&tensor_a, &tensor_b, Some(DType::F32), backend).unwrap();
    let correct = NumericTensor::from_vec(vec![
        -0.75,
        3.0,
        -4.5,
        8.25,
        1.25,
        -5.0,
        7.5,
        -13.75,
        -0.015625,
        0.0625,
        -0.09375,
        0.171875,
        -0.50390625,
        2.015625,
        -3.0234375,
        5.54296875,
    ])
    .to_dyn_rank()
    .reshape(vec![4, 4], backend)
    .unwrap()
    .cast(DType::BF16, backend)
    .unwrap();
    test_eq_bf16(result, correct);
}

pub fn test_matmul_2_2_2_2_fp32(backend: &mut EvalBackend) {
    let tensor_a = NumericTensor::from_vec(vec![256.0, 0.001, -0.0001, -128.0])
        .to_dyn_rank()
        .reshape(vec![2, 2], backend)
        .unwrap()
        .cast(DType::F32, backend)
        .unwrap();
    let tensor_b = NumericTensor::from_vec(vec![0.001, -64.0, 32.0, 0.0002])
        .to_dyn_rank()
        .reshape(vec![2, 2], backend)
        .unwrap()
        .cast(DType::F32, backend)
        .unwrap();
    let result = NumericTensor::matmul(&tensor_a, &tensor_b, Some(DType::F32), backend).unwrap();
    let correct = NumericTensor::from_vec(vec![0.2880000174, -16384.0, -4096.0, -0.01919999905])
        .to_dyn_rank()
        .reshape(vec![2, 2], backend)
        .unwrap()
        .cast(DType::F32, backend)
        .unwrap();
    test_eq_f32(result, correct);
}

pub fn test_matmul_2_2_2_2_bf16(backend: &mut EvalBackend) {
    let tensor_a = NumericTensor::from_vec(vec![256.0, 0.001, -0.0001, -128.0])
        .to_dyn_rank()
        .reshape(vec![2, 2], backend)
        .unwrap()
        .cast(DType::BF16, backend)
        .unwrap();
    let tensor_b = NumericTensor::from_vec(vec![0.001, -64.0, 32.0, 0.0002])
        .to_dyn_rank()
        .reshape(vec![2, 2], backend)
        .unwrap()
        .cast(DType::BF16, backend)
        .unwrap();
    let result = NumericTensor::matmul(&tensor_a, &tensor_b, Some(DType::F32), backend).unwrap();
    let correct = NumericTensor::from_vec(vec![0.2880000174, -16384.0, -4096.0, -0.01919999905])
        .to_dyn_rank()
        .reshape(vec![2, 2], backend)
        .unwrap()
        .cast(DType::BF16, backend)
        .unwrap();
    test_eq_bf16(result, correct);
}

pub fn test_matmul_2_2_2_2_f16(backend: &mut EvalBackend) {
    let tensor_a = NumericTensor::from_vec(vec![256.0, 0.001, -0.0001, -128.0])
        .to_dyn_rank()
        .reshape(vec![2, 2], backend)
        .unwrap()
        .cast(DType::F16, backend)
        .unwrap();
    let tensor_b = NumericTensor::from_vec(vec![0.001, -64.0, 32.0, 0.0002])
        .to_dyn_rank()
        .reshape(vec![2, 2], backend)
        .unwrap()
        .cast(DType::F16, backend)
        .unwrap();
    let result = NumericTensor::matmul(&tensor_a, &tensor_b, Some(DType::F32), backend).unwrap();
    let correct = NumericTensor::from_vec(vec![0.2880000174, -16384.0, -4096.0, -0.01919999905])
        .to_dyn_rank()
        .reshape(vec![2, 2], backend)
        .unwrap()
        .cast(DType::F16, backend)
        .unwrap();
    test_eq_f16(result, correct);
}

pub fn get_test_vec() -> Vec<f32> {
    vec![
        0.16369684,
        0.20416842,
        -0.043642867,
        0.07657923,
        0.03815732,
        -0.119812675,
        -0.14914201,
        0.031554546,
        0.20317698,
        -0.0084144,
        0.12148691,
        0.04097984,
        0.074499905,
        -0.08559261,
        0.12156696,
        -0.045107406,
        0.04029052,
        0.005795969,
        -0.06511796,
        0.00010094733,
        0.12966533,
        -0.26140738,
        0.0180814,
        -0.03998516,
        -0.025973825,
        -0.015186188,
        0.2051485,
        0.10276304,
        -0.19122495,
        0.12554674,
        -0.13080767,
        -0.05775433,
        0.033585824,
        0.19300039,
        -0.062995076,
        -0.20645921,
        -0.061975203,
        -0.05426843,
        -0.29516232,
        0.07645906,
        -0.121257514,
        0.0036634307,
        -0.12617704,
        0.17496423,
        0.32335556,
        0.100867726,
        0.19640127,
        -0.05346455,
        0.10433401,
        0.058668386,
        0.14131315,
        -0.11332793,
        0.08253743,
        0.07175747,
        0.047049366,
        -0.07439455,
        -0.03245342,
        0.06304925,
        -0.18647335,
        -0.17298856,
        0.024200175,
        0.00022550671,
        -0.09976465,
        0.014589956,
    ]
}

pub fn test_matmul_rank4_fp32(backend: &mut EvalBackend) {
    let mut test_vec_a = vec![];
    for i in 0..1024 {
        test_vec_a.extend(get_test_vec().iter().map(|&x| x * (0.1 * i as f32)));
    }
    // 65536 elements
    let mut test_vec_b = vec![];
    for i in 0..16 {
        test_vec_b.extend(get_test_vec().iter().map(|&x| x * (0.1 * i as f32)));
    }
    // 16 elements

    let tensor_a = NumericTensor::from_vec(test_vec_a.clone())
        .to_dyn_rank()
        .reshape(vec![1, 16, 64, 64], backend)
        .unwrap()
        .cast(DType::F32, backend)
        .unwrap();
    let tensor_b = NumericTensor::from_vec(test_vec_b.clone())
        .to_dyn_rank()
        .reshape(vec![1, 16, 64, 1], backend)
        .unwrap()
        .cast(DType::F32, backend)
        .unwrap();
    let tensor_a_tgt = backend.to_native_type(&tensor_a);
    let tensor_b_tgt = backend.to_native_type(&tensor_b);

    let test_result = NumericTensor::matmul(
        &tensor_a_tgt.neg(backend).unwrap(),
        &tensor_b_tgt,
        Some(DType::F32),
        backend,
    )
    .unwrap();
    let control_result = NumericTensor::matmul(
        &tensor_a.neg(backend).unwrap(),
        &tensor_b,
        Some(DType::F32),
        &mut EvalBackend::NDArray,
    )
    .unwrap();

    test_eq_f32(test_result, control_result);
}
