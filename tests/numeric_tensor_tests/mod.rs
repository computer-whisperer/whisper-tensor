use whisper_tensor::DynRank;
use whisper_tensor::dtype::DType;
use whisper_tensor::numeric_tensor::NumericTensor;

pub mod basic_arith;
pub mod basic_matmul;
pub mod cumsum;
pub mod reshape;
pub mod unary;

fn test_eq(value: NumericTensor<DynRank>, correct: NumericTensor<DynRank>, atol: f64, rtol: f64) {
    assert_eq!(value.shape(), correct.shape());
    let value_vec: Vec<f64> = value
        .to_ndarray()
        .unwrap()
        .cast(DType::F64)
        .unwrap()
        .flatten()
        .try_to_vec()
        .unwrap();
    let correct_vec: Vec<f64> = correct
        .to_ndarray()
        .unwrap()
        .cast(DType::F64)
        .unwrap()
        .flatten()
        .try_to_vec()
        .unwrap();
    for i in 0..value_vec.len() {
        let a = value_vec[i];
        let b = correct_vec[i];
        let err = (a - b).abs();
        let limit = atol + rtol * (a.abs().max(b.abs()));
        assert!(err < limit, "{a} != {b}: {err} < {limit}");
    }
}

fn test_eq_f16(value: NumericTensor<DynRank>, correct: NumericTensor<DynRank>) {
    assert_eq!(value.dtype(), DType::F16);
    assert_eq!(correct.dtype(), DType::F16);
    test_eq(value, correct, 1e-5, 4e-3);
}

fn test_eq_bf16(value: NumericTensor<DynRank>, correct: NumericTensor<DynRank>) {
    assert_eq!(value.dtype(), DType::BF16);
    assert_eq!(correct.dtype(), DType::BF16);
    test_eq(value, correct, 1e-5, 1.6e-2);
}

fn test_eq_f32(value: NumericTensor<DynRank>, correct: NumericTensor<DynRank>) {
    assert_eq!(value.dtype(), DType::F32);
    assert_eq!(correct.dtype(), DType::F32);
    test_eq(value, correct, 1e-5, 1.3e-6);
}
