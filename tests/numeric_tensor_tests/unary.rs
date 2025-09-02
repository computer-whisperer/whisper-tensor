use crate::numeric_tensor_tests::{test_eq_bf16, test_eq_f16, test_eq_f32};
use half::{bf16, f16};
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::numeric_tensor::NumericTensor;

fn exp_correct(
    input: &NumericTensor<whisper_tensor::DynRank>,
) -> NumericTensor<whisper_tensor::DynRank> {
    let mut be = EvalBackend::NDArray;
    input.exp(&mut be).unwrap()
}
fn ln_correct(
    input: &NumericTensor<whisper_tensor::DynRank>,
) -> NumericTensor<whisper_tensor::DynRank> {
    let mut be = EvalBackend::NDArray;
    input.ln(&mut be).unwrap()
}
fn abs_correct(
    input: &NumericTensor<whisper_tensor::DynRank>,
) -> NumericTensor<whisper_tensor::DynRank> {
    let mut be = EvalBackend::NDArray;
    input.abs(&mut be).unwrap()
}
fn floor_correct(
    input: &NumericTensor<whisper_tensor::DynRank>,
) -> NumericTensor<whisper_tensor::DynRank> {
    let mut be = EvalBackend::NDArray;
    input.floor(&mut be).unwrap()
}
fn ceil_correct(
    input: &NumericTensor<whisper_tensor::DynRank>,
) -> NumericTensor<whisper_tensor::DynRank> {
    let mut be = EvalBackend::NDArray;
    input.ceil(&mut be).unwrap()
}
fn round_correct(
    input: &NumericTensor<whisper_tensor::DynRank>,
) -> NumericTensor<whisper_tensor::DynRank> {
    let mut be = EvalBackend::NDArray;
    input.round(&mut be).unwrap()
}

// exp
pub fn test_exp_fp32(backend: &mut EvalBackend) {
    let x = NumericTensor::from_vec(vec![0.0f32, 1.0, -1.0, 3.5]).to_dyn_rank();
    let y = x.exp(backend).unwrap();
    let correct = exp_correct(&x);
    test_eq_f32(y, correct);
}

pub fn test_exp_bf16(backend: &mut EvalBackend) {
    let x = NumericTensor::from_vec(vec![
        bf16::from_f32(0.0),
        bf16::from_f32(1.0),
        bf16::from_f32(-1.0),
        bf16::from_f32(3.5),
    ])
    .to_dyn_rank();
    let y = x.exp(backend).unwrap();
    let correct = exp_correct(&x);
    test_eq_bf16(y, correct);
}

pub fn test_exp_f16(backend: &mut EvalBackend) {
    let x = NumericTensor::from_vec(vec![
        f16::from_f32(0.0),
        f16::from_f32(1.0),
        f16::from_f32(-1.0),
        f16::from_f32(3.5),
    ])
    .to_dyn_rank();
    let y = x.exp(backend).unwrap();
    let correct = exp_correct(&x);
    test_eq_f16(y, correct);
}

// ln
pub fn test_ln_fp32(backend: &mut EvalBackend) {
    let x = NumericTensor::from_vec(vec![1.0f32, 2.7182817, 0.5, 10.0]).to_dyn_rank();
    let y = x.ln(backend).unwrap();
    let correct = ln_correct(&x);
    test_eq_f32(y, correct);
}

pub fn test_ln_bf16(backend: &mut EvalBackend) {
    let x = NumericTensor::from_vec(vec![
        bf16::from_f32(1.0),
        bf16::from_f32(2.7182817),
        bf16::from_f32(0.5),
        bf16::from_f32(10.0),
    ])
    .to_dyn_rank();
    let y = x.ln(backend).unwrap();
    let correct = ln_correct(&x);
    test_eq_bf16(y, correct);
}

pub fn test_ln_f16(backend: &mut EvalBackend) {
    let x = NumericTensor::from_vec(vec![
        f16::from_f32(1.0),
        f16::from_f32(2.7182817),
        f16::from_f32(0.5),
        f16::from_f32(10.0),
    ])
    .to_dyn_rank();
    let y = x.ln(backend).unwrap();
    let correct = ln_correct(&x);
    test_eq_f16(y, correct);
}

// abs
pub fn test_abs_fp32(backend: &mut EvalBackend) {
    let x = NumericTensor::from_vec(vec![-1.25f32, 0.0, 3.14159, -2.0]).to_dyn_rank();
    let y = x.abs(backend).unwrap();
    let correct = abs_correct(&x);
    test_eq_f32(y, correct);
}

pub fn test_abs_bf16(backend: &mut EvalBackend) {
    let x = NumericTensor::from_vec(vec![
        bf16::from_f32(-1.25),
        bf16::from_f32(0.0),
        bf16::from_f32(3.14159),
        bf16::from_f32(-2.0),
    ])
    .to_dyn_rank();
    let y = x.abs(backend).unwrap();
    let correct = abs_correct(&x);
    test_eq_bf16(y, correct);
}

pub fn test_abs_f16(backend: &mut EvalBackend) {
    let x = NumericTensor::from_vec(vec![
        f16::from_f32(-1.25),
        f16::from_f32(0.0),
        f16::from_f32(3.14159),
        f16::from_f32(-2.0),
    ])
    .to_dyn_rank();
    let y = x.abs(backend).unwrap();
    let correct = abs_correct(&x);
    test_eq_f16(y, correct);
}

// floor
pub fn test_floor_fp32(backend: &mut EvalBackend) {
    let x = NumericTensor::from_vec(vec![1.9f32, -1.1, 0.0, 2.0, -2.0]).to_dyn_rank();
    let y = x.floor(backend).unwrap();
    let correct = floor_correct(&x);
    test_eq_f32(y, correct);
}

pub fn test_floor_bf16(backend: &mut EvalBackend) {
    let x = NumericTensor::from_vec(vec![
        bf16::from_f32(1.9),
        bf16::from_f32(-1.1),
        bf16::from_f32(0.0),
        bf16::from_f32(2.0),
        bf16::from_f32(-2.0),
    ])
    .to_dyn_rank();
    let y = x.floor(backend).unwrap();
    let correct = floor_correct(&x);
    test_eq_bf16(y, correct);
}

pub fn test_floor_f16(backend: &mut EvalBackend) {
    let x = NumericTensor::from_vec(vec![
        f16::from_f32(1.9),
        f16::from_f32(-1.1),
        f16::from_f32(0.0),
        f16::from_f32(2.0),
        f16::from_f32(-2.0),
    ])
    .to_dyn_rank();
    let y = x.floor(backend).unwrap();
    let correct = floor_correct(&x);
    test_eq_f16(y, correct);
}

// ceil
pub fn test_ceil_fp32(backend: &mut EvalBackend) {
    let x = NumericTensor::from_vec(vec![1.1f32, -1.9, 0.0, 2.0, -2.0]).to_dyn_rank();
    let y = x.ceil(backend).unwrap();
    let correct = ceil_correct(&x);
    test_eq_f32(y, correct);
}

pub fn test_ceil_bf16(backend: &mut EvalBackend) {
    let x = NumericTensor::from_vec(vec![
        bf16::from_f32(1.1),
        bf16::from_f32(-1.9),
        bf16::from_f32(0.0),
        bf16::from_f32(2.0),
        bf16::from_f32(-2.0),
    ])
    .to_dyn_rank();
    let y = x.ceil(backend).unwrap();
    let correct = ceil_correct(&x);
    test_eq_bf16(y, correct);
}

pub fn test_ceil_f16(backend: &mut EvalBackend) {
    let x = NumericTensor::from_vec(vec![
        f16::from_f32(1.1),
        f16::from_f32(-1.9),
        f16::from_f32(0.0),
        f16::from_f32(2.0),
        f16::from_f32(-2.0),
    ])
    .to_dyn_rank();
    let y = x.ceil(backend).unwrap();
    let correct = ceil_correct(&x);
    test_eq_f16(y, correct);
}

// round (ties to even per backend semantics)
pub fn test_round_fp32(backend: &mut EvalBackend) {
    let x = NumericTensor::from_vec(vec![1.5f32, 2.5, -1.5, -2.5, 0.49, -0.51]).to_dyn_rank();
    let y = x.round(backend).unwrap();
    let correct = round_correct(&x);
    test_eq_f32(y, correct);
}

pub fn test_round_bf16(backend: &mut EvalBackend) {
    let x = NumericTensor::from_vec(vec![
        bf16::from_f32(1.5),
        bf16::from_f32(2.5),
        bf16::from_f32(-1.5),
        bf16::from_f32(-2.5),
        bf16::from_f32(0.49),
        bf16::from_f32(-0.51),
    ])
    .to_dyn_rank();
    let y = x.round(backend).unwrap();
    let correct = round_correct(&x);
    test_eq_bf16(y, correct);
}

pub fn test_round_f16(backend: &mut EvalBackend) {
    let x = NumericTensor::from_vec(vec![
        f16::from_f32(1.5),
        f16::from_f32(2.5),
        f16::from_f32(-1.5),
        f16::from_f32(-2.5),
        f16::from_f32(0.49),
        f16::from_f32(-0.51),
    ])
    .to_dyn_rank();
    let y = x.round(backend).unwrap();
    let correct = round_correct(&x);
    test_eq_f16(y, correct);
}
