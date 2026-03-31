use paste::paste;
use whisper_tensor::backends::eval_backend::EvalBackend;
mod numeric_tensor_tests;
use numeric_tensor_tests::basic_arith::*;
use numeric_tensor_tests::basic_matmul::*;
use numeric_tensor_tests::cumsum::*;
use numeric_tensor_tests::reshape::*;
use numeric_tensor_tests::unary::*;

fn run_ndarray_test(test: impl FnOnce(&mut EvalBackend)) {
    test(&mut EvalBackend::NDArray)
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
        do_test!($runner_fn, $runner_name, test_matmul_rank4_fp32);
        // Unary ops
        do_test!($runner_fn, $runner_name, test_exp_fp32);
        do_test!($runner_fn, $runner_name, test_exp_bf16);
        do_test!($runner_fn, $runner_name, test_exp_f16);
        do_test!($runner_fn, $runner_name, test_ln_fp32);
        do_test!($runner_fn, $runner_name, test_ln_near_one_fp32);
        do_test!($runner_fn, $runner_name, test_ln_bf16);
        do_test!($runner_fn, $runner_name, test_ln_f16);
        do_test!($runner_fn, $runner_name, test_exp_large_negative_fp32);
        do_test!($runner_fn, $runner_name, test_tanh_small_fp32);
        do_test!($runner_fn, $runner_name, test_mish_chain_fp32);
        // test_softplus_via_operation_fp32 and test_mish_via_model_eval_fp32
        // removed — used deleted legacy Operation::eval() / Model::eval().
        do_test!($runner_fn, $runner_name, test_abs_fp32);
        do_test!($runner_fn, $runner_name, test_abs_bf16);
        do_test!($runner_fn, $runner_name, test_abs_f16);
        do_test!($runner_fn, $runner_name, test_floor_fp32);
        do_test!($runner_fn, $runner_name, test_floor_bf16);
        do_test!($runner_fn, $runner_name, test_floor_f16);
        do_test!($runner_fn, $runner_name, test_ceil_fp32);
        do_test!($runner_fn, $runner_name, test_ceil_bf16);
        do_test!($runner_fn, $runner_name, test_ceil_f16);
        do_test!($runner_fn, $runner_name, test_round_fp32);
        do_test!($runner_fn, $runner_name, test_round_bf16);
        do_test!($runner_fn, $runner_name, test_round_f16);
        // CumSum
        do_test!(
            $runner_fn,
            $runner_name,
            test_cumsum_1d_f32_inclusive_forward
        );
        do_test!(
            $runner_fn,
            $runner_name,
            test_cumsum_1d_f32_exclusive_forward
        );
        do_test!(
            $runner_fn,
            $runner_name,
            test_cumsum_1d_f32_inclusive_reverse
        );
        do_test!($runner_fn, $runner_name, test_cumsum_2d_axis0);
        do_test!($runner_fn, $runner_name, test_cumsum_2d_axis1);
        do_test!($runner_fn, $runner_name, test_cumsum_2d_negative_axis);
    };
}

do_tests!(run_ndarray_test, ndarray);
