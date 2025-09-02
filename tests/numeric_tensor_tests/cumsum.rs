use crate::numeric_tensor_tests::test_eq_f32;
use whisper_tensor::DynRank;
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::numeric_tensor::NumericTensor;

fn cumsum_correct(
    input: &NumericTensor<whisper_tensor::DynRank>,
    axis: Option<isize>,
    exclusive: bool,
    reverse: bool,
) -> NumericTensor<whisper_tensor::DynRank> {
    // Always compute reference with NDArray backend
    let mut be = EvalBackend::NDArray;
    input.cumsum(axis, exclusive, reverse, &mut be).unwrap()
}

pub fn test_cumsum_1d_f32_inclusive_forward(backend: &mut EvalBackend) {
    let x = NumericTensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]).to_dyn_rank();
    let y = x.cumsum(Some(0), false, false, backend).unwrap();
    let correct = cumsum_correct(&x, Some(0), false, false);
    test_eq_f32(y, correct);
}

pub fn test_cumsum_1d_f32_exclusive_forward(backend: &mut EvalBackend) {
    let x = NumericTensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]).to_dyn_rank();
    let y = x.cumsum(Some(0), true, false, backend).unwrap();
    let correct = cumsum_correct(&x, Some(0), true, false);
    test_eq_f32(y, correct);
}

pub fn test_cumsum_1d_f32_inclusive_reverse(backend: &mut EvalBackend) {
    let x = NumericTensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]).to_dyn_rank();
    let y = x.cumsum(Some(0), false, true, backend).unwrap();
    let correct = cumsum_correct(&x, Some(0), false, true);
    test_eq_f32(y, correct);
}

pub fn test_cumsum_2d_axis0(backend: &mut EvalBackend) {
    // Shape (3,4)
    let x = NumericTensor::<DynRank>::from_vec_shape(
        vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        vec![3, 4],
    )
    .unwrap()
    .to_dyn_rank();
    let y = x.cumsum(Some(0), false, false, backend).unwrap();
    let correct = cumsum_correct(&x, Some(0), false, false);
    test_eq_f32(y, correct);
}

pub fn test_cumsum_2d_axis1(backend: &mut EvalBackend) {
    // Shape (3,4)
    let x = NumericTensor::<DynRank>::from_vec_shape(
        vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        vec![3, 4],
    )
    .unwrap()
    .to_dyn_rank();
    let y = x.cumsum(Some(1), false, false, backend).unwrap();
    let correct = cumsum_correct(&x, Some(1), false, false);
    test_eq_f32(y, correct);
}

pub fn test_cumsum_2d_negative_axis(backend: &mut EvalBackend) {
    // Shape (2,3)
    let x =
        NumericTensor::<DynRank>::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])
            .unwrap()
            .to_dyn_rank();
    // axis = -1 means last dimension
    let y = x.cumsum(Some(-1), false, false, backend).unwrap();
    let correct = cumsum_correct(&x, Some(-1), false, false);
    test_eq_f32(y, correct);
}
