use crate::numeric_tensor_tests::{test_eq_bf16, test_eq_f16, test_eq_f32};
use half::{bf16, f16};
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::migration::numeric_tensor::NumericTensor;

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

// ln near 1.0 — exercises precision for softplus(x) = ln(1 + exp(x)) when x is very negative
pub fn test_ln_near_one_fp32(backend: &mut EvalBackend) {
    let x = NumericTensor::from_vec(vec![
        1.00004553794860839844f32, // 1 + exp(-9.998) — the Mish failure case
        1.001f32,
        1.0001f32,
        1.00001f32,
        1.000001f32,
    ])
    .to_dyn_rank();
    let y = x.ln(backend).unwrap();
    let correct = ln_correct(&x);
    test_eq_f32(y, correct);
}

// exp of large negative — exercises exp(-10) which is tiny
pub fn test_exp_large_negative_fp32(backend: &mut EvalBackend) {
    let x = NumericTensor::from_vec(vec![-9.998f32, -20.0, -50.0, -5.0, -1.0]).to_dyn_rank();
    let y = x.exp(backend).unwrap();
    let correct = exp_correct(&x);
    test_eq_f32(y, correct);
}

// tanh of small values — exercises tanh(~0) precision.
// Includes the exact softplus(-9.998) intermediate (4.5536912e-5) that
// causes the Mish ONNX test failure when tanh precision diverges.
pub fn test_tanh_small_fp32(backend: &mut EvalBackend) {
    let x = NumericTensor::from_vec(vec![
        4.5536912e-5f32, // exact softplus(-9.998) in F32 — the Mish failure value
        4.55e-5f32,
        1e-4f32,
        1e-5f32,
        1e-6f32,
        1e-7f32,
        0.5f32,
        -0.5f32,
    ])
    .to_dyn_rank();
    let y = x.trig(whisper_tensor::TrigOp::Tanh, backend).unwrap();
    let correct = {
        let mut be = EvalBackend::NDArray;
        x.trig(whisper_tensor::TrigOp::Tanh, &mut be).unwrap()
    };
    test_eq_f32(y, correct);
}

// Mish chain: x * tanh(ln(1 + exp(x))) — each step on the given backend.
// This reproduces the exact computation that fails in the ONNX Mish test.
// Uses 10K elements (matching the ONNX test size) because Vulkan kernels
// may behave differently at scale due to work group scheduling.
// Each intermediate result is compared against NDArray to isolate where
// divergence first appears.
pub fn test_mish_chain_fp32(backend: &mut EvalBackend) {
    // 10K elements, linearly spaced from -10 to 10 (matches ONNX test_mish input range)
    let n = 10000;
    let vals: Vec<f32> = (0..n)
        .map(|i| -10.0 + 20.0 * (i as f32) / (n as f32 - 1.0))
        .collect();
    let one_f32 = NumericTensor::from_vec(vec![1.0f32; n]).to_dyn_rank();
    let x = NumericTensor::from_vec(vals).to_dyn_rank();

    let mut nda = EvalBackend::NDArray;

    // Step 1: exp(x)
    let exp_x = x.exp(backend).unwrap();
    let exp_x_ref = x.exp(&mut nda).unwrap();
    test_eq_f32(exp_x.clone(), exp_x_ref.clone());

    // Step 2: 1 + exp(x)
    let one_plus_exp = NumericTensor::add(&one_f32, &exp_x, backend).unwrap();
    let one_plus_ref = NumericTensor::add(&one_f32, &exp_x_ref, &mut nda).unwrap();
    test_eq_f32(one_plus_exp.clone(), one_plus_ref.clone());

    // Step 3: ln(1 + exp(x))  — softplus
    let softplus = one_plus_exp.ln(backend).unwrap();
    let sp_ref = one_plus_ref.ln(&mut nda).unwrap();
    test_eq_f32(softplus.clone(), sp_ref.clone());

    // Step 4: tanh(softplus)
    let tanh_sp = softplus
        .trig(whisper_tensor::TrigOp::Tanh, backend)
        .unwrap();
    let tanh_ref = sp_ref.trig(whisper_tensor::TrigOp::Tanh, &mut nda).unwrap();
    test_eq_f32(tanh_sp.clone(), tanh_ref.clone());

    // Step 5: x * tanh(softplus)  — final mish
    let mish = NumericTensor::mul(&x, &tanh_sp, backend).unwrap();
    let mish_ref = NumericTensor::mul(&x, &tanh_ref, &mut nda).unwrap();
    test_eq_f32(mish, mish_ref);
}

// test_softplus_via_operation_fp32 — removed. Used legacy Operation::eval()
// which has been deleted. Softplus is covered by test_set composite cases
// and the mish chain test above.
//
// test_mish_via_model_eval_fp32 — removed. Used legacy Model::eval() and
// ONNX .pb reference loading. Model-level accuracy is covered by
// tests/accuracy.rs golden snapshot tests.

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
    let x = NumericTensor::from_vec(vec![-1.25f32, 0.0, std::f32::consts::PI, -2.0]).to_dyn_rank();
    let y = x.abs(backend).unwrap();
    let correct = abs_correct(&x);
    test_eq_f32(y, correct);
}

pub fn test_abs_bf16(backend: &mut EvalBackend) {
    let x = NumericTensor::from_vec(vec![
        bf16::from_f32(-1.25),
        bf16::from_f32(0.0),
        bf16::from_f32(std::f32::consts::PI),
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
        f16::from_f32(std::f32::consts::PI),
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
