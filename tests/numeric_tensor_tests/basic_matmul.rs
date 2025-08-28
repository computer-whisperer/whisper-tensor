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
