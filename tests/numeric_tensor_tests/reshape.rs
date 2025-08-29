use crate::numeric_tensor_tests::test_eq_f32;
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::numeric_tensor::NumericTensor;

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

pub fn test_reshape_fp32(backend: &mut EvalBackend) {
    let test_values: Vec<f32> = get_test_vec();
    let tensor_a = NumericTensor::from_vec(test_values.clone()).to_dyn_rank();
    let tensor_a = tensor_a
        .reshape(vec![1, 4, 2, 8], &mut EvalBackend::NDArray)
        .unwrap();
    let result = backend
        .to_native_type(&tensor_a)
        .reshape(vec![1, 2, 4, 8], backend)
        .unwrap();
    let correct = NumericTensor::from_vec(test_values.clone())
        .to_dyn_rank()
        .reshape(vec![1, 2, 4, 8], &mut EvalBackend::NDArray)
        .unwrap();
    test_eq_f32(result, correct);
}

pub fn test_transpose_reshape_fp32(backend: &mut EvalBackend) {
    let test_values: Vec<f32> = get_test_vec();
    let tensor_a = NumericTensor::from_vec(test_values.clone())
        .to_dyn_rank()
        .reshape(vec![1, 4, 2, 8], &mut EvalBackend::NDArray)
        .unwrap();
    let result = backend
        .to_native_type(&tensor_a)
        .transpose(Some(vec![0, 2, 1, 3]), backend)
        .unwrap()
        .reshape(vec![1, 2, 4, 8], backend)
        .unwrap();
    let correct = tensor_a
        .transpose(Some(vec![0, 2, 1, 3]), &mut EvalBackend::NDArray)
        .unwrap()
        .reshape(vec![1, 2, 4, 8], &mut EvalBackend::NDArray)
        .unwrap();
    test_eq_f32(result, correct);
}
