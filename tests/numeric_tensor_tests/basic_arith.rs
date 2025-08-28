use crate::numeric_tensor_tests::{test_eq_bf16, test_eq_f16, test_eq_f32};
use half::{bf16, f16};
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::dtype::DType;
use whisper_tensor::numeric_tensor::NumericTensor;

pub fn test_add_bf16(backend: &mut EvalBackend) {
    let tensor_a = NumericTensor::from_vec(vec![
        bf16::from_f32(0.75390625),
        bf16::from_f32(0.93359375),
        bf16::from_f32(0.13671875),
    ])
    .to_dyn_rank();
    let tensor_b = NumericTensor::from_vec(vec![
        bf16::from_f32(4.40625),
        bf16::from_f32(5.65625),
        bf16::from_f32(38.25),
    ])
    .to_dyn_rank();
    let result = NumericTensor::add(&tensor_a, &tensor_b, backend).unwrap();
    let correct = NumericTensor::from_vec(vec![
        bf16::from_f32(5.15625),
        bf16::from_f32(6.59375),
        bf16::from_f32(38.25),
    ])
    .to_dyn_rank();
    test_eq_bf16(result, correct);
}

pub fn test_add_f16(backend: &mut EvalBackend) {
    let tensor_a = NumericTensor::from_vec(vec![
        f16::from_f32(0.75390625),
        f16::from_f32(0.93359375),
        f16::from_f32(0.13671875),
    ])
    .to_dyn_rank();
    let tensor_b = NumericTensor::from_vec(vec![
        f16::from_f32(4.40625),
        f16::from_f32(5.65625),
        f16::from_f32(38.25),
    ])
    .to_dyn_rank();
    let result = NumericTensor::add(&tensor_a, &tensor_b, backend).unwrap();
    let correct = NumericTensor::from_vec(vec![
        f16::from_f32(5.15625),
        f16::from_f32(6.59375),
        f16::from_f32(38.375),
    ])
    .to_dyn_rank();
    test_eq_f16(result, correct);
}

pub fn test_add_fp32(backend: &mut EvalBackend) {
    let tensor_a = NumericTensor::from_vec(vec![0.15163845f32, 0.31361532, 5.393808]).to_dyn_rank();
    let tensor_b = NumericTensor::from_vec(vec![1.3424649f32, 0.004955234, 6.920299]).to_dyn_rank();
    let result = NumericTensor::add(&tensor_a, &tensor_b, backend).unwrap();
    let correct = NumericTensor::from_vec(vec![1.4941034f32, 0.31857055, 12.314107]).to_dyn_rank();
    test_eq_f32(result, correct);
}

pub fn test_sub_bf16(backend: &mut EvalBackend) {
    let tensor_a = NumericTensor::from_vec(vec![
        bf16::from_f32(0.75390625),
        bf16::from_f32(0.93359375),
        bf16::from_f32(0.13671875),
    ])
    .to_dyn_rank();
    let tensor_b = NumericTensor::from_vec(vec![
        bf16::from_f32(4.40625),
        bf16::from_f32(5.65625),
        bf16::from_f32(38.25),
    ])
    .to_dyn_rank();
    let result = NumericTensor::sub(&tensor_a, &tensor_b, backend).unwrap();
    let correct = NumericTensor::from_vec(vec![
        bf16::from_f32(-3.65625),
        bf16::from_f32(-4.71875),
        bf16::from_f32(-38.00),
    ])
    .to_dyn_rank();
    test_eq_bf16(result, correct);
}

pub fn test_sub_f16(backend: &mut EvalBackend) {
    let tensor_a = NumericTensor::from_vec(vec![
        f16::from_f32(0.75390625),
        f16::from_f32(0.93359375),
        f16::from_f32(0.13671875),
    ])
    .to_dyn_rank();
    let tensor_b = NumericTensor::from_vec(vec![
        f16::from_f32(4.40625),
        f16::from_f32(5.65625),
        f16::from_f32(38.25),
    ])
    .to_dyn_rank();
    let result = NumericTensor::sub(&tensor_a, &tensor_b, backend).unwrap();
    let correct = NumericTensor::from_vec(vec![
        f16::from_f32(-3.6523438),
        f16::from_f32(-4.7226563),
        f16::from_f32(-38.125),
    ])
    .to_dyn_rank();
    test_eq_f16(result, correct);
}

pub fn test_sub_fp32(backend: &mut EvalBackend) {
    let tensor_a = NumericTensor::from_vec(vec![0.15163845f32, 0.31361532, 5.393808]).to_dyn_rank();
    let tensor_b = NumericTensor::from_vec(vec![1.3424649f32, 0.004955234, 6.920299]).to_dyn_rank();
    let result = NumericTensor::sub(&tensor_a, &tensor_b, backend).unwrap();
    let correct = NumericTensor::from_vec(vec![-1.1908265f32, 0.3086601, -1.5264912]).to_dyn_rank();
    test_eq_f32(result, correct);
}

pub fn test_mul_bf16(backend: &mut EvalBackend) {
    let tensor_a = NumericTensor::from_vec(vec![
        bf16::from_f32(0.75390625),
        bf16::from_f32(0.93359375),
        bf16::from_f32(0.13671875),
    ])
    .to_dyn_rank();
    let tensor_b = NumericTensor::from_vec(vec![
        bf16::from_f32(4.40625),
        bf16::from_f32(5.65625),
        bf16::from_f32(38.25),
    ])
    .to_dyn_rank();
    let result = NumericTensor::mul(&tensor_a, &tensor_b, backend).unwrap();
    let correct = NumericTensor::from_vec(vec![
        bf16::from_f32(3.328125),
        bf16::from_f32(5.28125),
        bf16::from_f32(5.21875),
    ])
    .to_dyn_rank();
    test_eq_bf16(result, correct);
}

pub fn test_mul_f16(backend: &mut EvalBackend) {
    let tensor_a = NumericTensor::from_vec(vec![
        f16::from_f32(0.75390625),
        f16::from_f32(0.93359375),
        f16::from_f32(0.13671875),
    ])
    .to_dyn_rank();
    let tensor_b = NumericTensor::from_vec(vec![
        f16::from_f32(4.40625),
        f16::from_f32(5.65625),
        f16::from_f32(38.25),
    ])
    .to_dyn_rank();
    let result = NumericTensor::mul(&tensor_a, &tensor_b, backend).unwrap();
    let correct = NumericTensor::from_vec(vec![
        f16::from_f32(3.3222656),
        f16::from_f32(5.28125),
        f16::from_f32(5.2304688),
    ])
    .to_dyn_rank();
    test_eq_f16(result, correct);
}

pub fn test_mul_fp32(backend: &mut EvalBackend) {
    let tensor_a = NumericTensor::from_vec(vec![0.15163845f32, 0.31361532, 5.393808]).to_dyn_rank();
    let tensor_b = NumericTensor::from_vec(vec![1.3424649f32, 0.004955234, 6.920299]).to_dyn_rank();
    let result = NumericTensor::mul(&tensor_a, &tensor_b, backend).unwrap();
    let correct =
        NumericTensor::from_vec(vec![0.2035693f32, 0.0015540373, 37.326763]).to_dyn_rank();
    test_eq_f32(result, correct);
}

pub fn test_div_bf16(backend: &mut EvalBackend) {
    let tensor_a = NumericTensor::from_vec(vec![
        bf16::from_f32(0.75390625),
        bf16::from_f32(0.93359375),
        bf16::from_f32(0.13671875),
    ])
    .to_dyn_rank();
    let tensor_b = NumericTensor::from_vec(vec![
        bf16::from_f32(4.40625),
        bf16::from_f32(5.65625),
        bf16::from_f32(38.25),
    ])
    .to_dyn_rank();
    let result = NumericTensor::div(&tensor_a, &tensor_b, backend).unwrap();
    let correct = NumericTensor::from_vec(vec![
        bf16::from_f32(0.17089844),
        bf16::from_f32(0.16503906),
        bf16::from_f32(0.0035705566),
    ])
    .to_dyn_rank();
    test_eq_bf16(result, correct);
}

pub fn test_div_f16(backend: &mut EvalBackend) {
    let tensor_a = NumericTensor::from_vec(vec![
        f16::from_f32(0.75390625),
        f16::from_f32(0.93359375),
        f16::from_f32(0.13671875),
    ])
    .to_dyn_rank();
    let tensor_b = NumericTensor::from_vec(vec![
        f16::from_f32(4.40625),
        f16::from_f32(5.65625),
        f16::from_f32(38.25),
    ])
    .to_dyn_rank();
    let result = NumericTensor::div(&tensor_a, &tensor_b, backend).unwrap();
    let correct = NumericTensor::from_vec(vec![
        f16::from_f32(0.17114258),
        f16::from_f32(0.16503906),
        f16::from_f32(0.0035705566),
    ])
    .to_dyn_rank();
    test_eq_f16(result, correct);
}

pub fn test_div_fp32(backend: &mut EvalBackend) {
    let tensor_a = NumericTensor::from_vec(vec![0.15163845f32, 0.31361532, 5.393808]).to_dyn_rank();
    let tensor_b = NumericTensor::from_vec(vec![1.3424649f32, 0.004955234, 6.920299]).to_dyn_rank();
    let result = NumericTensor::div(&tensor_a, &tensor_b, backend).unwrap();
    let correct = NumericTensor::from_vec(vec![0.11295524f32, 63.28971, 0.77941835]).to_dyn_rank();
    test_eq_f32(result, correct);
}

pub fn test_pow_fp32(backend: &mut EvalBackend) {
    let tensor_a =
        NumericTensor::from_vec(vec![0.15163845f32, -0.31361532, 5.393808, 38.25]).to_dyn_rank();
    let tensor_b = NumericTensor::from_vec(vec![2.0f32]).to_dyn_rank();
    let result = NumericTensor::pow(&tensor_a, &tensor_b, backend).unwrap();
    let correct = NumericTensor::from_vec(vec![0.022994218f32, 0.09835457, 29.093164, 1463.0625])
        .to_dyn_rank();
    test_eq_f32(result, correct);
}

pub fn test_pow_bf16(backend: &mut EvalBackend) {
    let tensor_a = NumericTensor::from_vec(vec![0.15163845f32, -0.31361532, 5.393808, 38.25])
        .to_dyn_rank()
        .cast(DType::BF16, backend)
        .unwrap();
    let tensor_b = NumericTensor::from_vec(vec![2.0f32])
        .to_dyn_rank()
        .cast(DType::BF16, backend)
        .unwrap();
    let result = NumericTensor::pow(&tensor_a, &tensor_b, backend).unwrap();
    let correct = NumericTensor::from_vec(vec![0.022994218f32, 0.09835457, 29.093164, 1463.0625])
        .to_dyn_rank()
        .cast(DType::BF16, backend)
        .unwrap();
    test_eq_bf16(result, correct);
}

pub fn test_pow_f16(backend: &mut EvalBackend) {
    let tensor_a = NumericTensor::from_vec(vec![0.15163845f32, -0.31361532, 5.393808, 38.25])
        .to_dyn_rank()
        .cast(DType::F16, backend)
        .unwrap();
    let tensor_b = NumericTensor::from_vec(vec![2.0f32])
        .to_dyn_rank()
        .cast(DType::F16, backend)
        .unwrap();
    let result = NumericTensor::pow(&tensor_a, &tensor_b, backend).unwrap();
    let correct = NumericTensor::from_vec(vec![0.022994218f32, 0.09844971, 29.093164, 1463.0625])
        .to_dyn_rank()
        .cast(DType::F16, backend)
        .unwrap();
    test_eq_f16(result, correct);
}
