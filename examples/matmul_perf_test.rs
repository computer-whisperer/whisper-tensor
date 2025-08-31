use std::time::Instant;
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::backends::vulkan_backend::{VulkanContext, VulkanImmediateExecutor};
use whisper_tensor::dtype::DType;
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

fn main() {
    tracing_subscriber::fmt::init();

    let vulkan_context = VulkanContext::new().unwrap();
    let mut vulkan_runtime = VulkanImmediateExecutor::new(vulkan_context).unwrap();

    let mut backend = EvalBackend::Vulkan(&mut vulkan_runtime);

    let mut test_vec_a = vec![];
    for i in 0..65536 {
        test_vec_a.extend(get_test_vec().iter().map(|&x| x * (0.1 * i as f32)));
    }
    // 2048*2048 elements
    let mut test_vec_b = vec![];
    for i in 0..32 {
        test_vec_b.extend(get_test_vec().iter().map(|&x| x * (0.1 * i as f32)));
    }
    // 2048 elements

    let tensor_a = NumericTensor::from_vec(test_vec_a.clone())
        .to_dyn_rank()
        .reshape(vec![1, 2048, 2048], &mut backend)
        .unwrap()
        .cast(DType::F32, &mut backend)
        .unwrap();
    let tensor_b = NumericTensor::from_vec(test_vec_b.clone())
        .to_dyn_rank()
        .reshape(vec![1, 2048, 1], &mut backend)
        .unwrap()
        .cast(DType::F32, &mut backend)
        .unwrap();
    let tensor_a_tgt = backend.to_native_type(&tensor_a);
    let tensor_b_tgt = backend.to_native_type(&tensor_b);

    let mut live_vec = tensor_b_tgt;
    //warmup
    let n = 20;
    for _ in 0..n {
        live_vec = NumericTensor::matmul(&tensor_a_tgt, &live_vec, Some(DType::F32), &mut backend)
            .unwrap();
    }

    let start_instant = Instant::now();
    let n = 100;
    for _ in 0..n {
        live_vec = NumericTensor::matmul(&tensor_a_tgt, &live_vec, Some(DType::F32), &mut backend)
            .unwrap();
    }
    println!("Time per matmul: {:?}", start_instant.elapsed() / n);
}
