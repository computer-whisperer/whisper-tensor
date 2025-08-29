use paste::paste;
use whisper_tensor::backends::eval_backend::EvalBackend;
mod numeric_tensor_tests;
use numeric_tensor_tests::basic_arith::*;
use numeric_tensor_tests::basic_matmul::*;
use numeric_tensor_tests::reshape::*;

fn run_ndarray_test(test: impl FnOnce(&mut EvalBackend)) {
    test(&mut EvalBackend::NDArray)
}

#[cfg(feature = "tch")]
fn run_tch_test(test: impl FnOnce(&mut EvalBackend)) {
    test(&mut EvalBackend::TCH)
}

#[cfg(feature = "vulkan")]
fn run_vulkan_test(test: impl FnOnce(&mut EvalBackend)) {
    use whisper_tensor::backends::vulkan_backend::{VulkanContext, VulkanImmediateExecutor};
    let vulkan_context = VulkanContext::new().unwrap();
    let mut vulkan_runtime = VulkanImmediateExecutor::new(vulkan_context).unwrap();

    let mut eval_backend = EvalBackend::Vulkan(&mut vulkan_runtime);
    test(&mut eval_backend)
}

macro_rules! do_test {
    ($runner_fn:expr, $runner_name:ident, $test_name:ident) => {
        paste! {
            #[allow(non_snake_case)]
            #[test]
            fn [<$runner_name _ $test_name>]() {
                $runner_fn($test_name);
            }
        }
    };
}

macro_rules! do_tests {
    ($runner_fn:expr, $runner_name:ident) => {
        do_test!($runner_fn, $runner_name, test_add_bf16);
        do_test!($runner_fn, $runner_name, test_add_f16);
        do_test!($runner_fn, $runner_name, test_add_fp32);
        do_test!($runner_fn, $runner_name, test_sub_bf16);
        do_test!($runner_fn, $runner_name, test_sub_f16);
        do_test!($runner_fn, $runner_name, test_sub_fp32);
        do_test!($runner_fn, $runner_name, test_mul_bf16);
        do_test!($runner_fn, $runner_name, test_mul_f16);
        do_test!($runner_fn, $runner_name, test_mul_fp32);
        do_test!($runner_fn, $runner_name, test_div_bf16);
        do_test!($runner_fn, $runner_name, test_div_f16);
        do_test!($runner_fn, $runner_name, test_div_fp32);
        do_test!($runner_fn, $runner_name, test_matmul_2_3_bf16);
        do_test!($runner_fn, $runner_name, test_matmul_2_3_f16);
        do_test!($runner_fn, $runner_name, test_matmul_2_3_fp32);
        do_test!($runner_fn, $runner_name, test_matmul_3_3_bf16);
        do_test!($runner_fn, $runner_name, test_matmul_3_3_f16);
        do_test!($runner_fn, $runner_name, test_matmul_3_3_fp32);
        do_test!($runner_fn, $runner_name, test_matmul_1_4_4_1_fp32);
        do_test!($runner_fn, $runner_name, test_matmul_1_4_4_1_bf16);
        do_test!($runner_fn, $runner_name, test_matmul_1_4_4_1_f16);
        do_test!($runner_fn, $runner_name, test_matmul_4_1_1_4_f16);
        do_test!($runner_fn, $runner_name, test_matmul_4_1_1_4_fp32);
        do_test!($runner_fn, $runner_name, test_matmul_4_1_1_4_bf16);
        do_test!($runner_fn, $runner_name, test_matmul_2_2_2_2_fp32);
        do_test!($runner_fn, $runner_name, test_matmul_2_2_2_2_bf16);
        do_test!($runner_fn, $runner_name, test_matmul_2_2_2_2_f16);
        do_test!($runner_fn, $runner_name, test_pow_fp32);
        do_test!($runner_fn, $runner_name, test_pow_bf16);
        do_test!($runner_fn, $runner_name, test_pow_f16);
        do_test!($runner_fn, $runner_name, test_reshape_fp32);
        do_test!($runner_fn, $runner_name, test_transpose_reshape_fp32);
    };
}

do_tests!(run_ndarray_test, ndarray);
#[cfg(feature = "vulkan")]
do_tests!(run_vulkan_test, vulkan);
#[cfg(feature = "tch")]
do_tests!(run_tch_test, tch);
