// tests/onnx_node_tests.rs

use paste::paste;
use prost::Message;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Once;
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
#[cfg(feature = "vulkan")]
use whisper_tensor::backends::vulkan_backend::{VulkanContext, VulkanImmediateExecutor};
use whisper_tensor::dtype::{DType, DTypeError};
use whisper_tensor::model::Model;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::onnx::TensorProto;
use whisper_tensor::symbolic_graph::ONNXDecodingError;
use whisper_tensor::tensor_rank::DynRank;

// Structure to hold a test case
struct OnnxNodeTest {
    name: String,
    model_path: PathBuf,
    test_data_sets: Vec<TestDataSet>,
    rtol: f64,
    atol: f64,
}

#[derive(Debug, thiserror::Error)]
enum TestError {
    #[error("Error: {0}")]
    ErrorS(String),
    #[error(transparent)]
    DTypeError(#[from] DTypeError),
    #[error(transparent)]
    OtherError(#[from] anyhow::Error),
}

impl From<String> for TestError {
    fn from(value: String) -> Self {
        TestError::ErrorS(value)
    }
}

static INIT: Once = Once::new();

// Structure to hold a test data set (inputs and expected outputs)
struct TestDataSet {
    inputs: HashMap<String, Vec<u8>>,  // Raw protobuf data
    outputs: HashMap<String, Vec<u8>>, // Raw protobuf data
}

impl OnnxNodeTest {
    // Parse a test directory and create a test case
    fn from_directory(dir_path: &Path) -> Option<Self> {
        // Initialize Logger
        INIT.call_once(|| {
            env_logger::init();
        });

        let model_path = dir_path.join("model.onnx");
        if !model_path.exists() {
            log::warn!("No model.onnx found in {}", dir_path.display());
            return None;
        }

        // Find test data directories
        let mut test_data_sets = Vec::new();
        for entry in fs::read_dir(dir_path).ok()? {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.is_dir()
                && path
                    .file_name()?
                    .to_string_lossy()
                    .starts_with("test_data_set_")
                && let Some(data_set) = TestDataSet::from_directory(&path)
            {
                test_data_sets.push(data_set);
            }
        }

        if test_data_sets.is_empty() {
            log::warn!("No test data sets found in {}", dir_path.display());
            return None;
        }

        Some(OnnxNodeTest {
            name: dir_path.file_name()?.to_string_lossy().to_string(),
            model_path,
            test_data_sets,
            rtol: 1e-3,
            atol: 1e-7,
        })
    }

    // Run the test using a specified backend
    fn run(&self, backend: &mut EvalBackend) -> Result<(), TestError> {
        log::info!("Running test: {}", self.name);

        // Load the model
        let model_bytes =
            fs::read(&self.model_path).map_err(|e| format!("Failed to read model file: {e}"))?;

        let model = Model::new_from_onnx(&model_bytes)
            .map_err(|e| format!("Failed to load model: {e:?}"))?;

        // Run each test data set
        for (i, test_data_set) in self.test_data_sets.iter().enumerate() {
            log::info!("  Running test data set {i}");

            // Convert protobuf inputs to NumericTensor
            let inputs = test_data_set
                .parse_inputs()
                .map_err(|e| format!("Failed to parse inputs: {e}"))?;

            // Parse expected outputs
            let expected = test_data_set
                .parse_outputs()
                .map_err(|e| format!("Failed to parse expected outputs: {e}"))?;

            // Run the model
            let outputs = model
                .eval(inputs, &mut (), None, backend)
                .map_err(|e| format!("Model execution failed: {e:?}"))?;

            // Compare outputs with expected values
            for (name, expected_tensor) in &expected {
                let actual_tensor = outputs
                    .get(name)
                    .ok_or_else(|| format!("Output '{name}' not found in model results"))?;

                // Compare tensors with tolerance
                self.compare_tensors(actual_tensor, expected_tensor)?;
            }
        }

        Ok(())
    }

    // Compare tensors with tolerance
    fn compare_tensors(
        &self,
        actual: &NumericTensor<DynRank>,
        expected: &NumericTensor<DynRank>,
    ) -> Result<(), TestError> {
        // Check shapes match
        if actual.dtype() != expected.dtype() {
            Err(format!(
                "Data type mismatch: actual {:?} vs expected {:?}",
                actual.dtype(),
                expected.dtype()
            ))?
        }

        // Compare shapes
        if actual.shape() != expected.shape() {
            Err(format!(
                "Shape mismatch: actual {:?} vs expected {:?}",
                actual.shape(),
                expected.shape()
            ))?;
        }

        // Compare values when cast to 64 bit
        let actual_values: Vec<f64> = actual
            .cast(DType::F64, &mut EvalBackend::NDArray)
            .unwrap()
            .to_ndarray()
            .unwrap()
            .flatten()
            .try_to_vec()
            .unwrap();
        let expected_values: Vec<f64> = expected
            .cast(DType::F64, &mut EvalBackend::NDArray)
            .unwrap()
            .to_ndarray()
            .unwrap()
            .flatten()
            .try_to_vec()
            .unwrap();

        for (actual_value, expected_value) in actual_values.iter().zip(expected_values.iter()) {
            let abs_diff = (actual_value - expected_value).abs();
            let tolerance = self.atol + self.rtol * expected_value.abs();
            if abs_diff > tolerance {
                Err(format!(
                    "Value mismatch: actual {actual_value} vs expected {expected_value}"
                ))?;
            }
        }

        Ok(())
    }
}

impl TestDataSet {
    // Parse a test data directory
    fn from_directory(dir_path: &Path) -> Option<Self> {
        let mut inputs = HashMap::new();
        let mut outputs = HashMap::new();

        // Read input files
        for entry in fs::read_dir(dir_path).ok()? {
            let entry = entry.ok()?;
            let path = entry.path();
            let file_name = path.file_name()?.to_string_lossy().to_string();

            if file_name.starts_with("input_") && file_name.ends_with(".pb") {
                // Read the file
                let mut file = File::open(&path).ok()?;
                let mut buffer = Vec::new();
                file.read_to_end(&mut buffer).ok()?;

                // Extract the input name/index
                let name = file_name.strip_prefix("input_")?.strip_suffix(".pb")?;
                inputs.insert(name.to_string(), buffer);
            } else if file_name.starts_with("output_") && file_name.ends_with(".pb") {
                // Read the file
                let mut file = File::open(&path).ok()?;
                let mut buffer = Vec::new();
                file.read_to_end(&mut buffer).ok()?;

                // Extract the output name/index
                let name = file_name.strip_prefix("output_")?.strip_suffix(".pb")?;
                outputs.insert(name.to_string(), buffer);
            }
        }

        if inputs.is_empty() || outputs.is_empty() {
            return None;
        }

        Some(TestDataSet { inputs, outputs })
    }

    // Parse input protobuf data into NumericTensors
    fn parse_inputs(&self) -> Result<HashMap<String, NumericTensor<DynRank>>, ONNXDecodingError> {
        let mut result = HashMap::new();

        // This will need to use the model to get input names
        // For now, we'll use placeholder indices as names
        for proto_data in self.inputs.values() {
            let tensor_proto = TensorProto::decode(proto_data.as_slice()).unwrap();
            let tensor: NumericTensor<DynRank> =
                NDArrayNumericTensor::try_from(&tensor_proto)?.into();

            // Use a name based on index (you'll need a better naming strategy)
            result.insert(tensor_proto.name, tensor);
        }

        Ok(result)
    }

    // Parse output protobuf data into NumericTensors
    fn parse_outputs(&self) -> Result<HashMap<String, NumericTensor<DynRank>>, ONNXDecodingError> {
        let mut result = HashMap::new();

        for proto_data in self.outputs.values() {
            let tensor_proto = TensorProto::decode(proto_data.as_slice()).unwrap();
            let tensor: NumericTensor<DynRank> =
                NDArrayNumericTensor::try_from(&tensor_proto)?.into();

            result.insert(tensor_proto.name, tensor);
        }

        Ok(result)
    }
}

fn run_ndarray_test(path: &Path) {
    let test = OnnxNodeTest::from_directory(path).unwrap();
    test.run(&mut EvalBackend::NDArray).unwrap()
}

/*
#[cfg(feature = "candle")]
fn run_candle_test(path: &Path) {
    let test = OnnxNodeTest::from_directory(&path).unwrap();
    test.run(&mut EvalBackend::Candle(candle_core::Device::Cpu)).unwrap()
}*/

#[cfg(feature = "vulkan")]
fn run_vulkan_test(path: &Path) {
    let vulkan_context = VulkanContext::new().unwrap();
    let mut vulkan_runtime = VulkanImmediateExecutor::new(vulkan_context).unwrap();
    let mut eval_backend = EvalBackend::Vulkan(&mut vulkan_runtime);

    let test = OnnxNodeTest::from_directory(path).unwrap();
    test.run(&mut eval_backend).unwrap()
}

macro_rules! do_test {
    ($runner_fn:expr, $runner_name:ident, $test_name:ident) => {
        paste! {
            #[allow(non_snake_case)]
            #[test]
            fn [<$runner_name _ $test_name>]() {
                $runner_fn(PathBuf::from(concat!("libs/onnx/onnx/backend/test/data/node/", stringify!($test_name))).as_path());
            }
        }
    };
}

macro_rules! do_tests {
    ($runner_fn:expr, $runner_name:ident) => {
        do_test!($runner_fn, $runner_name, test_acos);
        do_test!($runner_fn, $runner_name, test_acos_example);
        do_test!($runner_fn, $runner_name, test_acosh);
        do_test!($runner_fn, $runner_name, test_acosh_example);

        //do_test!($runner_fn, $runner_name, test_adagrad);
        //do_test!($runner_fn, $runner_name, test_adagrad_multiple);
        //do_test!($runner_fn, $runner_name, test_adam);
        //do_test!($runner_fn, $runner_name, test_adam_multiple);

        do_test!($runner_fn, $runner_name, test_add);
        do_test!($runner_fn, $runner_name, test_add_bcast);
        do_test!($runner_fn, $runner_name, test_add_uint8);

     /*   do_test!($runner_fn, $runner_name, test_affine_grid_2d);
        do_test!($runner_fn, $runner_name, test_affine_grid_2d_align_corners);
        do_test!($runner_fn, $runner_name, test_affine_grid_2d_align_corners_expanded);
        do_test!($runner_fn, $runner_name, test_affine_grid_2d_expanded);
        do_test!($runner_fn, $runner_name, test_affine_grid_3d);
        do_test!($runner_fn, $runner_name, test_affine_grid_3d_align_corners);
        do_test!($runner_fn, $runner_name, test_affine_grid_3d_align_corners_expanded);
        do_test!($runner_fn, $runner_name, test_affine_grid_3d_expanded);
        do_test!($runner_fn, $runner_name, test_ai_onnx_ml_array_feature_extractor);
        do_test!($runner_fn, $runner_name, test_ai_onnx_ml_binarizer);
        do_test!($runner_fn, $runner_name, test_ai_onnx_ml_label_encoder_string_int);
        do_test!($runner_fn, $runner_name, test_ai_onnx_ml_label_encoder_string_int_no_default);
        do_test!($runner_fn, $runner_name, test_ai_onnx_ml_label_encoder_tensor_mapping);
        do_test!($runner_fn, $runner_name, test_ai_onnx_ml_label_encoder_tensor_value_only_mapping);
        do_test!($runner_fn, $runner_name, test_ai_onnx_ml_tree_ensemble_set_membership);
        do_test!($runner_fn, $runner_name, test_ai_onnx_ml_tree_ensemble_single_tree);*/

        do_test!($runner_fn, $runner_name, test_and2d);
        do_test!($runner_fn, $runner_name, test_and3d);
        do_test!($runner_fn, $runner_name, test_and4d);
        do_test!($runner_fn, $runner_name, test_and_bcast3v1d);
        do_test!($runner_fn, $runner_name, test_and_bcast3v2d);
        do_test!($runner_fn, $runner_name, test_and_bcast4v2d);
        do_test!($runner_fn, $runner_name, test_and_bcast4v3d);
        do_test!($runner_fn, $runner_name, test_and_bcast4v4d);

        do_test!($runner_fn, $runner_name, test_argmax_default_axis_example);
        do_test!($runner_fn, $runner_name, test_argmax_default_axis_example_select_last_index);
        do_test!($runner_fn, $runner_name, test_argmax_default_axis_random);
        do_test!($runner_fn, $runner_name, test_argmax_default_axis_random_select_last_index);
        do_test!($runner_fn, $runner_name, test_argmax_keepdims_example);
        do_test!($runner_fn, $runner_name, test_argmax_keepdims_example_select_last_index);
        do_test!($runner_fn, $runner_name, test_argmax_keepdims_random);
        do_test!($runner_fn, $runner_name, test_argmax_keepdims_random_select_last_index);
        do_test!($runner_fn, $runner_name, test_argmax_negative_axis_keepdims_example);
        do_test!($runner_fn, $runner_name, test_argmax_negative_axis_keepdims_example_select_last_index);
        do_test!($runner_fn, $runner_name, test_argmax_negative_axis_keepdims_random);
        do_test!($runner_fn, $runner_name, test_argmax_negative_axis_keepdims_random_select_last_index);
        do_test!($runner_fn, $runner_name, test_argmax_no_keepdims_example);
        do_test!($runner_fn, $runner_name, test_argmax_no_keepdims_example_select_last_index);
        do_test!($runner_fn, $runner_name, test_argmax_no_keepdims_random);
        do_test!($runner_fn, $runner_name, test_argmax_no_keepdims_random_select_last_index);
        do_test!($runner_fn, $runner_name, test_argmin_default_axis_example);
        do_test!($runner_fn, $runner_name, test_argmin_default_axis_example_select_last_index);
        do_test!($runner_fn, $runner_name, test_argmin_default_axis_random);
        do_test!($runner_fn, $runner_name, test_argmin_default_axis_random_select_last_index);
        do_test!($runner_fn, $runner_name, test_argmin_keepdims_example);
        do_test!($runner_fn, $runner_name, test_argmin_keepdims_example_select_last_index);
        do_test!($runner_fn, $runner_name, test_argmin_keepdims_random);
        do_test!($runner_fn, $runner_name, test_argmin_keepdims_random_select_last_index);
        do_test!($runner_fn, $runner_name, test_argmin_negative_axis_keepdims_example);
        do_test!($runner_fn, $runner_name, test_argmin_negative_axis_keepdims_example_select_last_index);
        do_test!($runner_fn, $runner_name, test_argmin_negative_axis_keepdims_random);
        do_test!($runner_fn, $runner_name, test_argmin_negative_axis_keepdims_random_select_last_index);
        do_test!($runner_fn, $runner_name, test_argmin_no_keepdims_example);
        do_test!($runner_fn, $runner_name, test_argmin_no_keepdims_example_select_last_index);
        do_test!($runner_fn, $runner_name, test_argmin_no_keepdims_random);
        do_test!($runner_fn, $runner_name, test_argmin_no_keepdims_random_select_last_index);

        do_test!($runner_fn, $runner_name, test_asin);
        do_test!($runner_fn, $runner_name, test_asin_example);
        do_test!($runner_fn, $runner_name, test_asinh);
        do_test!($runner_fn, $runner_name, test_asinh_example);
        do_test!($runner_fn, $runner_name, test_atan);
        do_test!($runner_fn, $runner_name, test_atan_example);
        do_test!($runner_fn, $runner_name, test_atanh);
        do_test!($runner_fn, $runner_name, test_atanh_example);

    /*    do_test!($runner_fn, $runner_name, test_attention_3d);
        do_test!($runner_fn, $runner_name, test_attention_3d_attn_mask);
        do_test!($runner_fn, $runner_name, test_attention_3d_attn_mask_expanded);
        do_test!($runner_fn, $runner_name, test_attention_3d_causal);
        do_test!($runner_fn, $runner_name, test_attention_3d_causal_expanded);
        do_test!($runner_fn, $runner_name, test_attention_3d_diff_heads_sizes);
        do_test!($runner_fn, $runner_name, test_attention_3d_diff_heads_sizes_attn_mask);
        do_test!($runner_fn, $runner_name, test_attention_3d_diff_heads_sizes_attn_mask_expanded);
        do_test!($runner_fn, $runner_name, test_attention_3d_diff_heads_sizes_causal);
        do_test!($runner_fn, $runner_name, test_attention_3d_diff_heads_sizes_causal_expanded);
        do_test!($runner_fn, $runner_name, test_attention_3d_diff_heads_sizes_expanded);
        do_test!($runner_fn, $runner_name, test_attention_3d_diff_heads_sizes_scaled);
        do_test!($runner_fn, $runner_name, test_attention_3d_diff_heads_sizes_scaled_expanded);
        do_test!($runner_fn, $runner_name, test_attention_3d_diff_heads_sizes_softcap);
        do_test!($runner_fn, $runner_name, test_attention_3d_diff_heads_sizes_softcap_expanded);
        do_test!($runner_fn, $runner_name, test_attention_3d_diff_heads_with_past_and_present);
        do_test!($runner_fn, $runner_name, test_attention_3d_diff_heads_with_past_and_present_expanded);
        do_test!($runner_fn, $runner_name, test_attention_3d_expanded);
        do_test!($runner_fn, $runner_name, test_attention_3d_gqa);
        do_test!($runner_fn, $runner_name, test_attention_3d_gqa_attn_mask);
        do_test!($runner_fn, $runner_name, test_attention_3d_gqa_attn_mask_expanded);
        do_test!($runner_fn, $runner_name, test_attention_3d_gqa_causal);
        do_test!($runner_fn, $runner_name, test_attention_3d_gqa_causal_expanded);
        do_test!($runner_fn, $runner_name, test_attention_3d_gqa_expanded);
        do_test!($runner_fn, $runner_name, test_attention_3d_gqa_scaled);
        do_test!($runner_fn, $runner_name, test_attention_3d_gqa_scaled_expanded);
        do_test!($runner_fn, $runner_name, test_attention_3d_gqa_softcap);
        do_test!($runner_fn, $runner_name, test_attention_3d_gqa_softcap_expanded);
        do_test!($runner_fn, $runner_name, test_attention_3d_gqa_with_past_and_present);
        do_test!($runner_fn, $runner_name, test_attention_3d_gqa_with_past_and_present_expanded);
        do_test!($runner_fn, $runner_name, test_attention_3d_scaled);
        do_test!($runner_fn, $runner_name, test_attention_3d_scaled_expanded);
        do_test!($runner_fn, $runner_name, test_attention_3d_softcap);
        do_test!($runner_fn, $runner_name, test_attention_3d_softcap_expanded);
        do_test!($runner_fn, $runner_name, test_attention_3d_with_past_and_present);
        do_test!($runner_fn, $runner_name, test_attention_3d_with_past_and_present_expanded);
        do_test!($runner_fn, $runner_name, test_attention_3d_with_past_and_present_qk_matmul);
        do_test!($runner_fn, $runner_name, test_attention_3d_with_past_and_present_qk_matmul_bias);
        do_test!($runner_fn, $runner_name, test_attention_3d_with_past_and_present_qk_matmul_bias_expanded);
        do_test!($runner_fn, $runner_name, test_attention_3d_with_past_and_present_qk_matmul_expanded);
        do_test!($runner_fn, $runner_name, test_attention_3d_with_past_and_present_qk_matmul_softcap);
        do_test!($runner_fn, $runner_name, test_attention_3d_with_past_and_present_qk_matmul_softcap_expanded);
        do_test!($runner_fn, $runner_name, test_attention_3d_with_past_and_present_qk_matmul_softmax);
        do_test!($runner_fn, $runner_name, test_attention_3d_with_past_and_present_qk_matmul_softmax_expanded);
        do_test!($runner_fn, $runner_name, test_attention_4d);
        do_test!($runner_fn, $runner_name, test_attention_4d_attn_mask);
        do_test!($runner_fn, $runner_name, test_attention_4d_attn_mask_bool);
        do_test!($runner_fn, $runner_name, test_attention_4d_attn_mask_bool_expanded);
        do_test!($runner_fn, $runner_name, test_attention_4d_attn_mask_expanded);
        do_test!($runner_fn, $runner_name, test_attention_4d_causal);
        do_test!($runner_fn, $runner_name, test_attention_4d_causal_expanded);
        do_test!($runner_fn, $runner_name, test_attention_4d_diff_heads_sizes);
        do_test!($runner_fn, $runner_name, test_attention_4d_diff_heads_sizes_attn_mask);
        do_test!($runner_fn, $runner_name, test_attention_4d_diff_heads_sizes_attn_mask_expanded);
        do_test!($runner_fn, $runner_name, test_attention_4d_diff_heads_sizes_causal);
        do_test!($runner_fn, $runner_name, test_attention_4d_diff_heads_sizes_causal_expanded);
        do_test!($runner_fn, $runner_name, test_attention_4d_diff_heads_sizes_expanded);
        do_test!($runner_fn, $runner_name, test_attention_4d_diff_heads_sizes_scaled);
        do_test!($runner_fn, $runner_name, test_attention_4d_diff_heads_sizes_scaled_expanded);
        do_test!($runner_fn, $runner_name, test_attention_4d_diff_heads_sizes_softcap);
        do_test!($runner_fn, $runner_name, test_attention_4d_diff_heads_sizes_softcap_expanded);
        do_test!($runner_fn, $runner_name, test_attention_4d_diff_heads_with_past_and_present);
        do_test!($runner_fn, $runner_name, test_attention_4d_diff_heads_with_past_and_present_expanded);
        do_test!($runner_fn, $runner_name, test_attention_4d_expanded);
        do_test!($runner_fn, $runner_name, test_attention_4d_gqa);
        do_test!($runner_fn, $runner_name, test_attention_4d_gqa_attn_mask);
        do_test!($runner_fn, $runner_name, test_attention_4d_gqa_attn_mask_expanded);
        do_test!($runner_fn, $runner_name, test_attention_4d_gqa_causal);
        do_test!($runner_fn, $runner_name, test_attention_4d_gqa_causal_expanded);
        do_test!($runner_fn, $runner_name, test_attention_4d_gqa_expanded);
        do_test!($runner_fn, $runner_name, test_attention_4d_gqa_scaled);
        do_test!($runner_fn, $runner_name, test_attention_4d_gqa_scaled_expanded);
        do_test!($runner_fn, $runner_name, test_attention_4d_gqa_softcap);
        do_test!($runner_fn, $runner_name, test_attention_4d_gqa_softcap_expanded);
        do_test!($runner_fn, $runner_name, test_attention_4d_gqa_with_past_and_present);
        do_test!($runner_fn, $runner_name, test_attention_4d_gqa_with_past_and_present_expanded);
        do_test!($runner_fn, $runner_name, test_attention_4d_scaled);
        do_test!($runner_fn, $runner_name, test_attention_4d_scaled_expanded);
        do_test!($runner_fn, $runner_name, test_attention_4d_softcap);
        do_test!($runner_fn, $runner_name, test_attention_4d_softcap_expanded);
        do_test!($runner_fn, $runner_name, test_attention_4d_with_past_and_present);
        do_test!($runner_fn, $runner_name, test_attention_4d_with_past_and_present_expanded);
        do_test!($runner_fn, $runner_name, test_attention_4d_with_past_and_present_qk_matmul);
        do_test!($runner_fn, $runner_name, test_attention_4d_with_past_and_present_qk_matmul_bias);
        do_test!($runner_fn, $runner_name, test_attention_4d_with_past_and_present_qk_matmul_bias_expanded);
        do_test!($runner_fn, $runner_name, test_attention_4d_with_past_and_present_qk_matmul_expanded);
        do_test!($runner_fn, $runner_name, test_attention_4d_with_qk_matmul);
        do_test!($runner_fn, $runner_name, test_attention_4d_with_qk_matmul_bias);
        do_test!($runner_fn, $runner_name, test_attention_4d_with_qk_matmul_bias_expanded);
        do_test!($runner_fn, $runner_name, test_attention_4d_with_qk_matmul_expanded);
        do_test!($runner_fn, $runner_name, test_attention_4d_with_qk_matmul_softcap);
        do_test!($runner_fn, $runner_name, test_attention_4d_with_qk_matmul_softcap_expanded);
        do_test!($runner_fn, $runner_name, test_attention_4d_with_qk_matmul_softmax);
        do_test!($runner_fn, $runner_name, test_attention_4d_with_qk_matmul_softmax_expanded);*/

      /*  do_test!($runner_fn, $runner_name, test_averagepool_1d_default);
        do_test!($runner_fn, $runner_name, test_averagepool_2d_ceil);
        do_test!($runner_fn, $runner_name, test_averagepool_2d_ceil_last_window_starts_on_pad);
        do_test!($runner_fn, $runner_name, test_averagepool_2d_default);
        do_test!($runner_fn, $runner_name, test_averagepool_2d_dilations);
        do_test!($runner_fn, $runner_name, test_averagepool_2d_pads);
        do_test!($runner_fn, $runner_name, test_averagepool_2d_pads_count_include_pad);
        do_test!($runner_fn, $runner_name, test_averagepool_2d_precomputed_pads);
        do_test!($runner_fn, $runner_name, test_averagepool_2d_precomputed_pads_count_include_pad);
        do_test!($runner_fn, $runner_name, test_averagepool_2d_precomputed_same_upper);
        do_test!($runner_fn, $runner_name, test_averagepool_2d_precomputed_strides);
        do_test!($runner_fn, $runner_name, test_averagepool_2d_same_lower);
        do_test!($runner_fn, $runner_name, test_averagepool_2d_same_upper);
        do_test!($runner_fn, $runner_name, test_averagepool_2d_strides);
        do_test!($runner_fn, $runner_name, test_averagepool_3d_default);
        do_test!($runner_fn, $runner_name, test_averagepool_3d_dilations_large_count_include_pad_is_0_ceil_mode_is_False);
        do_test!($runner_fn, $runner_name, test_averagepool_3d_dilations_large_count_include_pad_is_0_ceil_mode_is_True);
        do_test!($runner_fn, $runner_name, test_averagepool_3d_dilations_large_count_include_pad_is_1_ceil_mode_is_False);
        do_test!($runner_fn, $runner_name, test_averagepool_3d_dilations_large_count_include_pad_is_1_ceil_mode_is_True);
        do_test!($runner_fn, $runner_name, test_averagepool_3d_dilations_small);*/

        /*do_test!($runner_fn, $runner_name, test_basic_conv_without_padding);
        do_test!($runner_fn, $runner_name, test_basic_conv_with_padding);
        do_test!($runner_fn, $runner_name, test_basic_deform_conv_without_padding);
        do_test!($runner_fn, $runner_name, test_basic_deform_conv_with_padding);*/

        /*do_test!($runner_fn, $runner_name, test_batchnorm_epsilon);
        do_test!($runner_fn, $runner_name, test_batchnorm_epsilon_training_mode);
        do_test!($runner_fn, $runner_name, test_batchnorm_example);
        do_test!($runner_fn, $runner_name, test_batchnorm_example_training_mode);*/

        /*do_test!($runner_fn, $runner_name, test_bernoulli);
        do_test!($runner_fn, $runner_name, test_bernoulli_double);
        do_test!($runner_fn, $runner_name, test_bernoulli_double_expanded);
        do_test!($runner_fn, $runner_name, test_bernoulli_expanded);
        do_test!($runner_fn, $runner_name, test_bernoulli_seed);
        do_test!($runner_fn, $runner_name, test_bernoulli_seed_expanded);*/

       /* do_test!($runner_fn, $runner_name, test_bitshift_left_uint16);
        do_test!($runner_fn, $runner_name, test_bitshift_left_uint32);
        do_test!($runner_fn, $runner_name, test_bitshift_left_uint64);
        do_test!($runner_fn, $runner_name, test_bitshift_left_uint8);
        do_test!($runner_fn, $runner_name, test_bitshift_right_uint16);
        do_test!($runner_fn, $runner_name, test_bitshift_right_uint32);
        do_test!($runner_fn, $runner_name, test_bitshift_right_uint64);
        do_test!($runner_fn, $runner_name, test_bitshift_right_uint8);*/

        do_test!($runner_fn, $runner_name, test_bitwise_and_i16_3d);
        do_test!($runner_fn, $runner_name, test_bitwise_and_i32_2d);
        do_test!($runner_fn, $runner_name, test_bitwise_and_ui64_bcast_3v1d);
        do_test!($runner_fn, $runner_name, test_bitwise_and_ui8_bcast_4v3d);
        do_test!($runner_fn, $runner_name, test_bitwise_not_2d);
        do_test!($runner_fn, $runner_name, test_bitwise_not_3d);
        do_test!($runner_fn, $runner_name, test_bitwise_not_4d);
        do_test!($runner_fn, $runner_name, test_bitwise_or_i16_4d);
        do_test!($runner_fn, $runner_name, test_bitwise_or_i32_2d);
        do_test!($runner_fn, $runner_name, test_bitwise_or_ui64_bcast_3v1d);
        do_test!($runner_fn, $runner_name, test_bitwise_or_ui8_bcast_4v3d);
        do_test!($runner_fn, $runner_name, test_bitwise_xor_i16_3d);
        do_test!($runner_fn, $runner_name, test_bitwise_xor_i32_2d);
        do_test!($runner_fn, $runner_name, test_bitwise_xor_ui64_bcast_3v1d);
        do_test!($runner_fn, $runner_name, test_bitwise_xor_ui8_bcast_4v3d);

        //do_test!($runner_fn, $runner_name, test_blackmanwindow);
        do_test!($runner_fn, $runner_name, test_blackmanwindow_expanded);
        //do_test!($runner_fn, $runner_name, test_blackmanwindow_symmetric);
        do_test!($runner_fn, $runner_name, test_blackmanwindow_symmetric_expanded);
        //do_test!($runner_fn, $runner_name, test_cast_BFLOAT16_to_FLOAT);
        do_test!($runner_fn, $runner_name, test_cast_DOUBLE_to_FLOAT);
        do_test!($runner_fn, $runner_name, test_cast_DOUBLE_to_FLOAT16);
        do_test!($runner_fn, $runner_name, test_cast_FLOAT16_to_DOUBLE);
        do_test!($runner_fn, $runner_name, test_cast_FLOAT16_to_FLOAT);
        //do_test!($runner_fn, $runner_name, test_cast_FLOAT16_to_FLOAT4E2M1);
        //do_test!($runner_fn, $runner_name, test_cast_FLOAT16_to_FLOAT8E4M3FN);
        //do_test!($runner_fn, $runner_name, test_cast_FLOAT16_to_FLOAT8E4M3FNUZ);
        //do_test!($runner_fn, $runner_name, test_cast_FLOAT16_to_FLOAT8E5M2);
        //do_test!($runner_fn, $runner_name, test_cast_FLOAT16_to_FLOAT8E5M2FNUZ);
        //do_test!($runner_fn, $runner_name, test_cast_FLOAT16_to_INT4);
        //do_test!($runner_fn, $runner_name, test_cast_FLOAT16_to_UINT4);
        //do_test!($runner_fn, $runner_name, test_cast_FLOAT4E2M1_to_FLOAT);
        //do_test!($runner_fn, $runner_name, test_cast_FLOAT4E2M1_to_FLOAT16);
        //do_test!($runner_fn, $runner_name, test_cast_FLOAT8E4M3FN_to_FLOAT);
        //do_test!($runner_fn, $runner_name, test_cast_FLOAT8E4M3FN_to_FLOAT16);
        //do_test!($runner_fn, $runner_name, test_cast_FLOAT8E4M3FNUZ_to_FLOAT);
        //do_test!($runner_fn, $runner_name, test_cast_FLOAT8E4M3FNUZ_to_FLOAT16);
        //do_test!($runner_fn, $runner_name, test_cast_FLOAT8E5M2FNUZ_to_FLOAT);
        //do_test!($runner_fn, $runner_name, test_cast_FLOAT8E5M2FNUZ_to_FLOAT16);
        //do_test!($runner_fn, $runner_name, test_cast_FLOAT8E5M2_to_FLOAT);
        //do_test!($runner_fn, $runner_name, test_cast_FLOAT8E5M2_to_FLOAT16);
        //do_test!($runner_fn, $runner_name, test_cast_FLOAT_to_BFLOAT16);
        do_test!($runner_fn, $runner_name, test_cast_FLOAT_to_DOUBLE);
        do_test!($runner_fn, $runner_name, test_cast_FLOAT_to_FLOAT16);
        //do_test!($runner_fn, $runner_name, test_cast_FLOAT_to_FLOAT4E2M1);
        //do_test!($runner_fn, $runner_name, test_cast_FLOAT_to_FLOAT8E4M3FN);
        //do_test!($runner_fn, $runner_name, test_cast_FLOAT_to_FLOAT8E4M3FNUZ);
        //do_test!($runner_fn, $runner_name, test_cast_FLOAT_to_FLOAT8E5M2);
        //do_test!($runner_fn, $runner_name, test_cast_FLOAT_to_FLOAT8E5M2FNUZ);
        //do_test!($runner_fn, $runner_name, test_cast_FLOAT_to_INT4);
        //do_test!($runner_fn, $runner_name, test_cast_FLOAT_to_STRING);
        //do_test!($runner_fn, $runner_name, test_cast_FLOAT_to_UINT4);
        //do_test!($runner_fn, $runner_name, test_cast_INT4_to_FLOAT);
        //do_test!($runner_fn, $runner_name, test_cast_INT4_to_FLOAT16);
        //do_test!($runner_fn, $runner_name, test_cast_INT4_to_INT8);
        //do_test!($runner_fn, $runner_name, test_castlike_BFLOAT16_to_FLOAT);
        //do_test!($runner_fn, $runner_name, test_castlike_BFLOAT16_to_FLOAT_expanded);

        do_test!($runner_fn, $runner_name, test_castlike_DOUBLE_to_FLOAT);
        do_test!($runner_fn, $runner_name, test_castlike_DOUBLE_to_FLOAT16);
        do_test!($runner_fn, $runner_name, test_castlike_DOUBLE_to_FLOAT16_expanded);
        do_test!($runner_fn, $runner_name, test_castlike_DOUBLE_to_FLOAT_expanded);
        do_test!($runner_fn, $runner_name, test_castlike_FLOAT16_to_DOUBLE);
        do_test!($runner_fn, $runner_name, test_castlike_FLOAT16_to_DOUBLE_expanded);
        do_test!($runner_fn, $runner_name, test_castlike_FLOAT16_to_FLOAT);
        do_test!($runner_fn, $runner_name, test_castlike_FLOAT16_to_FLOAT_expanded);

        //do_test!($runner_fn, $runner_name, test_castlike_FLOAT8E4M3FN_to_FLOAT);
        //do_test!($runner_fn, $runner_name, test_castlike_FLOAT8E4M3FN_to_FLOAT_expanded);
        //do_test!($runner_fn, $runner_name, test_castlike_FLOAT8E4M3FNUZ_to_FLOAT);
        //do_test!($runner_fn, $runner_name, test_castlike_FLOAT8E4M3FNUZ_to_FLOAT_expanded);
        //do_test!($runner_fn, $runner_name, test_castlike_FLOAT8E5M2FNUZ_to_FLOAT);
        //do_test!($runner_fn, $runner_name, test_castlike_FLOAT8E5M2FNUZ_to_FLOAT_expanded);
        //do_test!($runner_fn, $runner_name, test_castlike_FLOAT8E5M2_to_FLOAT);
        //do_test!($runner_fn, $runner_name, test_castlike_FLOAT8E5M2_to_FLOAT_expanded);
        //do_test!($runner_fn, $runner_name, test_castlike_FLOAT_to_BFLOAT16);
        //do_test!($runner_fn, $runner_name, test_castlike_FLOAT_to_BFLOAT16_expanded);

        do_test!($runner_fn, $runner_name, test_castlike_FLOAT_to_DOUBLE);
        do_test!($runner_fn, $runner_name, test_castlike_FLOAT_to_DOUBLE_expanded);
        do_test!($runner_fn, $runner_name, test_castlike_FLOAT_to_FLOAT16);
        do_test!($runner_fn, $runner_name, test_castlike_FLOAT_to_FLOAT16_expanded);

        //do_test!($runner_fn, $runner_name, test_castlike_FLOAT_to_FLOAT8E4M3FN);
        //do_test!($runner_fn, $runner_name, test_castlike_FLOAT_to_FLOAT8E4M3FN_expanded);
        //do_test!($runner_fn, $runner_name, test_castlike_FLOAT_to_FLOAT8E4M3FNUZ);
        //do_test!($runner_fn, $runner_name, test_castlike_FLOAT_to_FLOAT8E4M3FNUZ_expanded);
        //do_test!($runner_fn, $runner_name, test_castlike_FLOAT_to_FLOAT8E5M2);
        //do_test!($runner_fn, $runner_name, test_castlike_FLOAT_to_FLOAT8E5M2_expanded);
        //do_test!($runner_fn, $runner_name, test_castlike_FLOAT_to_FLOAT8E5M2FNUZ);
        //do_test!($runner_fn, $runner_name, test_castlike_FLOAT_to_FLOAT8E5M2FNUZ_expanded);
        //do_test!($runner_fn, $runner_name, test_castlike_FLOAT_to_STRING);
        //do_test!($runner_fn, $runner_name, test_castlike_FLOAT_to_STRING_expanded);
        //do_test!($runner_fn, $runner_name, test_castlike_STRING_to_FLOAT);
        //do_test!($runner_fn, $runner_name, test_castlike_STRING_to_FLOAT_expanded);
        //do_test!($runner_fn, $runner_name, test_cast_no_saturate_FLOAT16_to_FLOAT8E4M3FN);
        //do_test!($runner_fn, $runner_name, test_cast_no_saturate_FLOAT16_to_FLOAT8E4M3FNUZ);
        //do_test!($runner_fn, $runner_name, test_cast_no_saturate_FLOAT16_to_FLOAT8E5M2);
        //do_test!($runner_fn, $runner_name, test_cast_no_saturate_FLOAT16_to_FLOAT8E5M2FNUZ);
        //do_test!($runner_fn, $runner_name, test_cast_no_saturate_FLOAT_to_FLOAT8E4M3FN);
        //do_test!($runner_fn, $runner_name, test_cast_no_saturate_FLOAT_to_FLOAT8E4M3FNUZ);
        //do_test!($runner_fn, $runner_name, test_cast_no_saturate_FLOAT_to_FLOAT8E5M2);
        //do_test!($runner_fn, $runner_name, test_cast_no_saturate_FLOAT_to_FLOAT8E5M2FNUZ);
        //do_test!($runner_fn, $runner_name, test_cast_STRING_to_FLOAT);
        //do_test!($runner_fn, $runner_name, test_cast_UINT4_to_FLOAT);
        //do_test!($runner_fn, $runner_name, test_cast_UINT4_to_FLOAT16);
        //do_test!($runner_fn, $runner_name, test_cast_UINT4_to_UINT8);

        do_test!($runner_fn, $runner_name, test_ceil);
        do_test!($runner_fn, $runner_name, test_ceil_example);

        //do_test!($runner_fn, $runner_name, test_celu);
        //do_test!($runner_fn, $runner_name, test_celu_expanded);

        /*do_test!($runner_fn, $runner_name, test_center_crop_pad_crop);
        do_test!($runner_fn, $runner_name, test_center_crop_pad_crop_and_pad);
        do_test!($runner_fn, $runner_name, test_center_crop_pad_crop_and_pad_expanded);
        do_test!($runner_fn, $runner_name, test_center_crop_pad_crop_axes_chw);
        do_test!($runner_fn, $runner_name, test_center_crop_pad_crop_axes_chw_expanded);
        do_test!($runner_fn, $runner_name, test_center_crop_pad_crop_axes_hwc);
        do_test!($runner_fn, $runner_name, test_center_crop_pad_crop_axes_hwc_expanded);
        do_test!($runner_fn, $runner_name, test_center_crop_pad_crop_expanded);
        do_test!($runner_fn, $runner_name, test_center_crop_pad_crop_negative_axes_hwc);
        do_test!($runner_fn, $runner_name, test_center_crop_pad_crop_negative_axes_hwc_expanded);
        do_test!($runner_fn, $runner_name, test_center_crop_pad_pad);
        do_test!($runner_fn, $runner_name, test_center_crop_pad_pad_expanded);*/

        do_test!($runner_fn, $runner_name, test_clip);
        //do_test!($runner_fn, $runner_name, test_clip_default_inbounds);
        do_test!($runner_fn, $runner_name, test_clip_default_inbounds_expanded);
        //do_test!($runner_fn, $runner_name, test_clip_default_int8_inbounds);
        do_test!($runner_fn, $runner_name, test_clip_default_int8_inbounds_expanded);
        //do_test!($runner_fn, $runner_name, test_clip_default_int8_max);
        do_test!($runner_fn, $runner_name, test_clip_default_int8_max_expanded);
        do_test!($runner_fn, $runner_name, test_clip_default_int8_min);
        do_test!($runner_fn, $runner_name, test_clip_default_int8_min_expanded);
        //do_test!($runner_fn, $runner_name, test_clip_default_max);
        do_test!($runner_fn, $runner_name, test_clip_default_max_expanded);
        do_test!($runner_fn, $runner_name, test_clip_default_min);
        do_test!($runner_fn, $runner_name, test_clip_default_min_expanded);
        do_test!($runner_fn, $runner_name, test_clip_example);
        do_test!($runner_fn, $runner_name, test_clip_example_expanded);
        do_test!($runner_fn, $runner_name, test_clip_expanded);
        do_test!($runner_fn, $runner_name, test_clip_inbounds);
        do_test!($runner_fn, $runner_name, test_clip_inbounds_expanded);
        do_test!($runner_fn, $runner_name, test_clip_min_greater_than_max);
        do_test!($runner_fn, $runner_name, test_clip_min_greater_than_max_expanded);
        do_test!($runner_fn, $runner_name, test_clip_outbounds);
        do_test!($runner_fn, $runner_name, test_clip_outbounds_expanded);
        do_test!($runner_fn, $runner_name, test_clip_splitbounds);
        do_test!($runner_fn, $runner_name, test_clip_splitbounds_expanded);

        /*do_test!($runner_fn, $runner_name, test_col2im);
        do_test!($runner_fn, $runner_name, test_col2im_5d);
        do_test!($runner_fn, $runner_name, test_col2im_dilations);
        do_test!($runner_fn, $runner_name, test_col2im_pads);
        do_test!($runner_fn, $runner_name, test_col2im_strides);*/

        /*do_test!($runner_fn, $runner_name, test_compress_0);
        do_test!($runner_fn, $runner_name, test_compress_1);
        do_test!($runner_fn, $runner_name, test_compress_default_axis);
        do_test!($runner_fn, $runner_name, test_compress_negative_axis);*/

        do_test!($runner_fn, $runner_name, test_concat_1d_axis_0);
        do_test!($runner_fn, $runner_name, test_concat_1d_axis_negative_1);
        do_test!($runner_fn, $runner_name, test_concat_2d_axis_0);
        do_test!($runner_fn, $runner_name, test_concat_2d_axis_1);
        do_test!($runner_fn, $runner_name, test_concat_2d_axis_negative_1);
        do_test!($runner_fn, $runner_name, test_concat_2d_axis_negative_2);
        do_test!($runner_fn, $runner_name, test_concat_3d_axis_0);
        do_test!($runner_fn, $runner_name, test_concat_3d_axis_1);
        do_test!($runner_fn, $runner_name, test_concat_3d_axis_2);
        do_test!($runner_fn, $runner_name, test_concat_3d_axis_negative_1);
        do_test!($runner_fn, $runner_name, test_concat_3d_axis_negative_2);
        do_test!($runner_fn, $runner_name, test_concat_3d_axis_negative_3);

        //do_test!($runner_fn, $runner_name, test_constant);
        do_test!($runner_fn, $runner_name, test_constantofshape_float_ones);
        do_test!($runner_fn, $runner_name, test_constantofshape_int_shape_zero);
        do_test!($runner_fn, $runner_name, test_constantofshape_int_zeros);

        /*
        do_test!($runner_fn, $runner_name, test_constant_pad);
        do_test!($runner_fn, $runner_name, test_constant_pad_axes);
        do_test!($runner_fn, $runner_name, test_constant_pad_negative_axes);*/

        /*do_test!($runner_fn, $runner_name, test_convinteger_without_padding);
        do_test!($runner_fn, $runner_name, test_convinteger_with_padding);
        do_test!($runner_fn, $runner_name, test_convtranspose);
        do_test!($runner_fn, $runner_name, test_convtranspose_1d);
        do_test!($runner_fn, $runner_name, test_convtranspose_3d);
        do_test!($runner_fn, $runner_name, test_convtranspose_autopad_same);
        do_test!($runner_fn, $runner_name, test_convtranspose_dilations);
        do_test!($runner_fn, $runner_name, test_convtranspose_group_2);
        do_test!($runner_fn, $runner_name, test_convtranspose_group_2_image_3);
        do_test!($runner_fn, $runner_name, test_convtranspose_kernel_shape);
        do_test!($runner_fn, $runner_name, test_convtranspose_output_shape);
        do_test!($runner_fn, $runner_name, test_convtranspose_pad);
        do_test!($runner_fn, $runner_name, test_convtranspose_pads);
        do_test!($runner_fn, $runner_name, test_conv_with_autopad_same);
        do_test!($runner_fn, $runner_name, test_conv_with_strides_and_asymmetric_padding);
        do_test!($runner_fn, $runner_name, test_conv_with_strides_no_padding);
        do_test!($runner_fn, $runner_name, test_conv_with_strides_padding);*/

        do_test!($runner_fn, $runner_name, test_cos);
        do_test!($runner_fn, $runner_name, test_cos_example);
        do_test!($runner_fn, $runner_name, test_cosh);
        do_test!($runner_fn, $runner_name, test_cosh_example);

        do_test!($runner_fn, $runner_name, test_cumsum_1d);
        do_test!($runner_fn, $runner_name, test_cumsum_1d_exclusive);
        do_test!($runner_fn, $runner_name, test_cumsum_1d_reverse);
        do_test!($runner_fn, $runner_name, test_cumsum_1d_reverse_exclusive);
        do_test!($runner_fn, $runner_name, test_cumsum_2d_axis_0);
        do_test!($runner_fn, $runner_name, test_cumsum_2d_axis_1);
        do_test!($runner_fn, $runner_name, test_cumsum_2d_negative_axis);

        /*
        do_test!($runner_fn, $runner_name, test_deform_conv_with_mask_bias);
        do_test!($runner_fn, $runner_name, test_deform_conv_with_multiple_offset_groups);
        do_test!($runner_fn, $runner_name, test_depthtospace_crd_mode_example);
        do_test!($runner_fn, $runner_name, test_depthtospace_example);*/

        /*
        do_test!($runner_fn, $runner_name, test_dequantizelinear);
        do_test!($runner_fn, $runner_name, test_dequantizelinear_axis);
        do_test!($runner_fn, $runner_name, test_dequantizelinear_blocked);
        do_test!($runner_fn, $runner_name, test_dequantizelinear_e4m3fn);
        do_test!($runner_fn, $runner_name, test_dequantizelinear_e4m3fn_float16);
        do_test!($runner_fn, $runner_name, test_dequantizelinear_e4m3fn_zero_point);
        do_test!($runner_fn, $runner_name, test_dequantizelinear_e5m2);
        do_test!($runner_fn, $runner_name, test_dequantizelinear_float4e2m1);
        do_test!($runner_fn, $runner_name, test_dequantizelinear_int16);
        do_test!($runner_fn, $runner_name, test_dequantizelinear_int4);
        do_test!($runner_fn, $runner_name, test_dequantizelinear_uint16);
        do_test!($runner_fn, $runner_name, test_dequantizelinear_uint4);*/

        /*
        do_test!($runner_fn, $runner_name, test_det_2d);
        do_test!($runner_fn, $runner_name, test_det_nd);*/

        /*
        do_test!($runner_fn, $runner_name, test_dft);
        do_test!($runner_fn, $runner_name, test_dft_axis);
        do_test!($runner_fn, $runner_name, test_dft_axis_opset19);
        do_test!($runner_fn, $runner_name, test_dft_inverse);
        do_test!($runner_fn, $runner_name, test_dft_inverse_opset19);
        do_test!($runner_fn, $runner_name, test_dft_opset19);*/

        do_test!($runner_fn, $runner_name, test_div);
        do_test!($runner_fn, $runner_name, test_div_bcast);
        do_test!($runner_fn, $runner_name, test_div_example);
        do_test!($runner_fn, $runner_name, test_div_uint8);

        /*
        do_test!($runner_fn, $runner_name, test_dropout_default);
        do_test!($runner_fn, $runner_name, test_dropout_default_mask);
        do_test!($runner_fn, $runner_name, test_dropout_default_mask_ratio);
        do_test!($runner_fn, $runner_name, test_dropout_default_old);
        do_test!($runner_fn, $runner_name, test_dropout_default_ratio);
        do_test!($runner_fn, $runner_name, test_dropout_random_old);*/

        /*
        do_test!($runner_fn, $runner_name, test_dynamicquantizelinear);
        do_test!($runner_fn, $runner_name, test_dynamicquantizelinear_expanded);
        do_test!($runner_fn, $runner_name, test_dynamicquantizelinear_max_adjusted);
        do_test!($runner_fn, $runner_name, test_dynamicquantizelinear_max_adjusted_expanded);
        do_test!($runner_fn, $runner_name, test_dynamicquantizelinear_min_adjusted);
        do_test!($runner_fn, $runner_name, test_dynamicquantizelinear_min_adjusted_expanded);
        */

        //do_test!($runner_fn, $runner_name, test_edge_pad);

        /*
        do_test!($runner_fn, $runner_name, test_einsum_batch_diagonal);
        do_test!($runner_fn, $runner_name, test_einsum_batch_matmul);
        do_test!($runner_fn, $runner_name, test_einsum_inner_prod);
        do_test!($runner_fn, $runner_name, test_einsum_sum);
        do_test!($runner_fn, $runner_name, test_einsum_transpose);*/

        /*
        do_test!($runner_fn, $runner_name, test_elu);
        do_test!($runner_fn, $runner_name, test_elu_default);
        do_test!($runner_fn, $runner_name, test_elu_default_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_elu_example);
        do_test!($runner_fn, $runner_name, test_elu_example_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_elu_expanded_ver18);*/

        do_test!($runner_fn, $runner_name, test_equal);
        do_test!($runner_fn, $runner_name, test_equal_bcast);
        do_test!($runner_fn, $runner_name, test_equal_string);
        do_test!($runner_fn, $runner_name, test_equal_string_broadcast);

        //do_test!($runner_fn, $runner_name, test_erf);
        do_test!($runner_fn, $runner_name, test_exp);
        do_test!($runner_fn, $runner_name, test_expand_dim_changed);
        do_test!($runner_fn, $runner_name, test_expand_dim_unchanged);
        do_test!($runner_fn, $runner_name, test_exp_example);

        /*
        do_test!($runner_fn, $runner_name, test_eyelike_populate_off_main_diagonal);
        do_test!($runner_fn, $runner_name, test_eyelike_with_dtype);
        do_test!($runner_fn, $runner_name, test_eyelike_without_dtype);*/

        do_test!($runner_fn, $runner_name, test_flatten_axis0);
        do_test!($runner_fn, $runner_name, test_flatten_axis1);
        do_test!($runner_fn, $runner_name, test_flatten_axis2);
        do_test!($runner_fn, $runner_name, test_flatten_axis3);
        do_test!($runner_fn, $runner_name, test_flatten_default_axis);
        do_test!($runner_fn, $runner_name, test_flatten_negative_axis1);
        do_test!($runner_fn, $runner_name, test_flatten_negative_axis2);
        do_test!($runner_fn, $runner_name, test_flatten_negative_axis3);
        do_test!($runner_fn, $runner_name, test_flatten_negative_axis4);
        do_test!($runner_fn, $runner_name, test_floor);
        do_test!($runner_fn, $runner_name, test_floor_example);
        do_test!($runner_fn, $runner_name, test_gather_0);
        do_test!($runner_fn, $runner_name, test_gather_1);
        do_test!($runner_fn, $runner_name, test_gather_2d_indices);
/*
        do_test!($runner_fn, $runner_name, test_gather_elements_0);
        do_test!($runner_fn, $runner_name, test_gather_elements_1);
        do_test!($runner_fn, $runner_name, test_gather_elements_negative_indices);*/

        /*
        do_test!($runner_fn, $runner_name, test_gathernd_example_float32);
        do_test!($runner_fn, $runner_name, test_gathernd_example_int32);
        do_test!($runner_fn, $runner_name, test_gathernd_example_int32_batch_dim1);

         */

        do_test!($runner_fn, $runner_name, test_gather_negative_indices);

        /*
        do_test!($runner_fn, $runner_name, test_gelu_default_1);
        do_test!($runner_fn, $runner_name, test_gelu_default_1_expanded);
        do_test!($runner_fn, $runner_name, test_gelu_default_2);
        do_test!($runner_fn, $runner_name, test_gelu_default_2_expanded);
        do_test!($runner_fn, $runner_name, test_gelu_tanh_1);
        do_test!($runner_fn, $runner_name, test_gelu_tanh_1_expanded);
        do_test!($runner_fn, $runner_name, test_gelu_tanh_2);
        do_test!($runner_fn, $runner_name, test_gelu_tanh_2_expanded);*/

        do_test!($runner_fn, $runner_name, test_gemm_all_attributes);
        do_test!($runner_fn, $runner_name, test_gemm_alpha);
        do_test!($runner_fn, $runner_name, test_gemm_beta);
        do_test!($runner_fn, $runner_name, test_gemm_default_matrix_bias);
        do_test!($runner_fn, $runner_name, test_gemm_default_no_bias);
        do_test!($runner_fn, $runner_name, test_gemm_default_scalar_bias);
        do_test!($runner_fn, $runner_name, test_gemm_default_single_elem_vector_bias);
        do_test!($runner_fn, $runner_name, test_gemm_default_vector_bias);
        do_test!($runner_fn, $runner_name, test_gemm_default_zero_bias);
        do_test!($runner_fn, $runner_name, test_gemm_transposeA);
        do_test!($runner_fn, $runner_name, test_gemm_transposeB);

        /*
        do_test!($runner_fn, $runner_name, test_globalaveragepool);
        do_test!($runner_fn, $runner_name, test_globalaveragepool_precomputed);
        do_test!($runner_fn, $runner_name, test_globalmaxpool);
        do_test!($runner_fn, $runner_name, test_globalmaxpool_precomputed);*/

        do_test!($runner_fn, $runner_name, test_greater);
        do_test!($runner_fn, $runner_name, test_greater_bcast);
        do_test!($runner_fn, $runner_name, test_greater_equal);
        do_test!($runner_fn, $runner_name, test_greater_equal_bcast);
        do_test!($runner_fn, $runner_name, test_greater_equal_bcast_expanded);
        do_test!($runner_fn, $runner_name, test_greater_equal_expanded);

        /*
        do_test!($runner_fn, $runner_name, test_gridsample);
        do_test!($runner_fn, $runner_name, test_gridsample_aligncorners_true);
        do_test!($runner_fn, $runner_name, test_gridsample_bicubic);
        do_test!($runner_fn, $runner_name, test_gridsample_bicubic_align_corners_0_additional_1);
        do_test!($runner_fn, $runner_name, test_gridsample_bicubic_align_corners_1_additional_1);
        do_test!($runner_fn, $runner_name, test_gridsample_bilinear);
        do_test!($runner_fn, $runner_name, test_gridsample_bilinear_align_corners_0_additional_1);
        do_test!($runner_fn, $runner_name, test_gridsample_bilinear_align_corners_1_additional_1);
        do_test!($runner_fn, $runner_name, test_gridsample_border_padding);
        do_test!($runner_fn, $runner_name, test_gridsample_nearest);
        do_test!($runner_fn, $runner_name, test_gridsample_nearest_align_corners_0_additional_1);
        do_test!($runner_fn, $runner_name, test_gridsample_nearest_align_corners_1_additional_1);
        do_test!($runner_fn, $runner_name, test_gridsample_reflection_padding);
        do_test!($runner_fn, $runner_name, test_gridsample_volumetric_bilinear_align_corners_0);
        do_test!($runner_fn, $runner_name, test_gridsample_volumetric_bilinear_align_corners_1);
        do_test!($runner_fn, $runner_name, test_gridsample_volumetric_nearest_align_corners_0);
        do_test!($runner_fn, $runner_name, test_gridsample_volumetric_nearest_align_corners_1);
        do_test!($runner_fn, $runner_name, test_gridsample_zeros_padding);*/

        do_test!($runner_fn, $runner_name, test_group_normalization_epsilon);
        do_test!($runner_fn, $runner_name, test_group_normalization_epsilon_expanded);
        do_test!($runner_fn, $runner_name, test_group_normalization_example);
        do_test!($runner_fn, $runner_name, test_group_normalization_example_expanded);

        /*
        do_test!($runner_fn, $runner_name, test_gru_batchwise);
        do_test!($runner_fn, $runner_name, test_gru_defaults);
        do_test!($runner_fn, $runner_name, test_gru_seq_length);
        do_test!($runner_fn, $runner_name, test_gru_with_initial_bias);*/

        //do_test!($runner_fn, $runner_name, test_hammingwindow);
        do_test!($runner_fn, $runner_name, test_hammingwindow_expanded);
        //do_test!($runner_fn, $runner_name, test_hammingwindow_symmetric);
        do_test!($runner_fn, $runner_name, test_hammingwindow_symmetric_expanded);
        //do_test!($runner_fn, $runner_name, test_hannwindow);
        do_test!($runner_fn, $runner_name, test_hannwindow_expanded);
        //do_test!($runner_fn, $runner_name, test_hannwindow_symmetric);
        do_test!($runner_fn, $runner_name, test_hannwindow_symmetric_expanded);

        /*
        do_test!($runner_fn, $runner_name, test_hardmax_axis_0);
        do_test!($runner_fn, $runner_name, test_hardmax_axis_1);
        do_test!($runner_fn, $runner_name, test_hardmax_axis_2);
        do_test!($runner_fn, $runner_name, test_hardmax_default_axis);
        do_test!($runner_fn, $runner_name, test_hardmax_example);
        do_test!($runner_fn, $runner_name, test_hardmax_negative_axis);
        do_test!($runner_fn, $runner_name, test_hardmax_one_hot);
         */

        /*
        do_test!($runner_fn, $runner_name, test_hardsigmoid);
        do_test!($runner_fn, $runner_name, test_hardsigmoid_default);
        do_test!($runner_fn, $runner_name, test_hardsigmoid_default_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_hardsigmoid_example);
        do_test!($runner_fn, $runner_name, test_hardsigmoid_example_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_hardsigmoid_expanded_ver18);*/

        /*
        do_test!($runner_fn, $runner_name, test_hardswish);
        do_test!($runner_fn, $runner_name, test_hardswish_expanded);*/

        do_test!($runner_fn, $runner_name, test_identity);
        //do_test!($runner_fn, $runner_name, test_identity_opt);
        //do_test!($runner_fn, $runner_name, test_identity_sequence);


        do_test!($runner_fn, $runner_name, test_if);
        /*do_test!($runner_fn, $runner_name, test_if_opt);
        do_test!($runner_fn, $runner_name, test_if_seq);*/

        /*
        do_test!($runner_fn, $runner_name, test_image_decoder_decode_bmp_rgb);
        do_test!($runner_fn, $runner_name, test_image_decoder_decode_jpeg2k_rgb);
        do_test!($runner_fn, $runner_name, test_image_decoder_decode_jpeg_bgr);
        do_test!($runner_fn, $runner_name, test_image_decoder_decode_jpeg_grayscale);
        do_test!($runner_fn, $runner_name, test_image_decoder_decode_jpeg_rgb);
        do_test!($runner_fn, $runner_name, test_image_decoder_decode_png_rgb);
        do_test!($runner_fn, $runner_name, test_image_decoder_decode_pnm_rgb);
        do_test!($runner_fn, $runner_name, test_image_decoder_decode_tiff_rgb);
        do_test!($runner_fn, $runner_name, test_image_decoder_decode_webp_rgb);*/

        //do_test!($runner_fn, $runner_name, test_instancenorm_epsilon);
        //do_test!($runner_fn, $runner_name, test_instancenorm_example);

        do_test!($runner_fn, $runner_name, test_isinf);
        do_test!($runner_fn, $runner_name, test_isinf_float16);
        do_test!($runner_fn, $runner_name, test_isinf_negative);
        do_test!($runner_fn, $runner_name, test_isinf_positive);
        do_test!($runner_fn, $runner_name, test_isnan);
        do_test!($runner_fn, $runner_name, test_isnan_float16);
        do_test!($runner_fn, $runner_name, test_layer_normalization_2d_axis0);
        do_test!($runner_fn, $runner_name, test_layer_normalization_2d_axis0_expanded);
        do_test!($runner_fn, $runner_name, test_layer_normalization_2d_axis0_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_layer_normalization_2d_axis1);
        do_test!($runner_fn, $runner_name, test_layer_normalization_2d_axis1_expanded);
        do_test!($runner_fn, $runner_name, test_layer_normalization_2d_axis1_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_layer_normalization_2d_axis_negative_1);
        do_test!($runner_fn, $runner_name, test_layer_normalization_2d_axis_negative_1_expanded);
        do_test!($runner_fn, $runner_name, test_layer_normalization_2d_axis_negative_1_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_layer_normalization_2d_axis_negative_2);
        do_test!($runner_fn, $runner_name, test_layer_normalization_2d_axis_negative_2_expanded);
        do_test!($runner_fn, $runner_name, test_layer_normalization_2d_axis_negative_2_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_layer_normalization_3d_axis0_epsilon);
        do_test!($runner_fn, $runner_name, test_layer_normalization_3d_axis0_epsilon_expanded);
        do_test!($runner_fn, $runner_name, test_layer_normalization_3d_axis0_epsilon_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_layer_normalization_3d_axis1_epsilon);
        do_test!($runner_fn, $runner_name, test_layer_normalization_3d_axis1_epsilon_expanded);
        do_test!($runner_fn, $runner_name, test_layer_normalization_3d_axis1_epsilon_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_layer_normalization_3d_axis2_epsilon);
        do_test!($runner_fn, $runner_name, test_layer_normalization_3d_axis2_epsilon_expanded);
        do_test!($runner_fn, $runner_name, test_layer_normalization_3d_axis2_epsilon_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_layer_normalization_3d_axis_negative_1_epsilon);
        do_test!($runner_fn, $runner_name, test_layer_normalization_3d_axis_negative_1_epsilon_expanded);
        do_test!($runner_fn, $runner_name, test_layer_normalization_3d_axis_negative_1_epsilon_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_layer_normalization_3d_axis_negative_2_epsilon);
        do_test!($runner_fn, $runner_name, test_layer_normalization_3d_axis_negative_2_epsilon_expanded);
        do_test!($runner_fn, $runner_name, test_layer_normalization_3d_axis_negative_2_epsilon_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_layer_normalization_3d_axis_negative_3_epsilon);
        do_test!($runner_fn, $runner_name, test_layer_normalization_3d_axis_negative_3_epsilon_expanded);
        do_test!($runner_fn, $runner_name, test_layer_normalization_3d_axis_negative_3_epsilon_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_layer_normalization_4d_axis0);
        do_test!($runner_fn, $runner_name, test_layer_normalization_4d_axis0_expanded);
        do_test!($runner_fn, $runner_name, test_layer_normalization_4d_axis0_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_layer_normalization_4d_axis1);
        do_test!($runner_fn, $runner_name, test_layer_normalization_4d_axis1_expanded);
        do_test!($runner_fn, $runner_name, test_layer_normalization_4d_axis1_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_layer_normalization_4d_axis2);
        do_test!($runner_fn, $runner_name, test_layer_normalization_4d_axis2_expanded);
        do_test!($runner_fn, $runner_name, test_layer_normalization_4d_axis2_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_layer_normalization_4d_axis3);
        do_test!($runner_fn, $runner_name, test_layer_normalization_4d_axis3_expanded);
        do_test!($runner_fn, $runner_name, test_layer_normalization_4d_axis3_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_layer_normalization_4d_axis_negative_1);
        do_test!($runner_fn, $runner_name, test_layer_normalization_4d_axis_negative_1_expanded);
        do_test!($runner_fn, $runner_name, test_layer_normalization_4d_axis_negative_1_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_layer_normalization_4d_axis_negative_2);
        do_test!($runner_fn, $runner_name, test_layer_normalization_4d_axis_negative_2_expanded);
        do_test!($runner_fn, $runner_name, test_layer_normalization_4d_axis_negative_2_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_layer_normalization_4d_axis_negative_3);
        do_test!($runner_fn, $runner_name, test_layer_normalization_4d_axis_negative_3_expanded);
        do_test!($runner_fn, $runner_name, test_layer_normalization_4d_axis_negative_3_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_layer_normalization_4d_axis_negative_4);
        do_test!($runner_fn, $runner_name, test_layer_normalization_4d_axis_negative_4_expanded);
        do_test!($runner_fn, $runner_name, test_layer_normalization_4d_axis_negative_4_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_layer_normalization_default_axis);
        do_test!($runner_fn, $runner_name, test_layer_normalization_default_axis_expanded);
        do_test!($runner_fn, $runner_name, test_layer_normalization_default_axis_expanded_ver18);


        //do_test!($runner_fn, $runner_name, test_leakyrelu);
        //do_test!($runner_fn, $runner_name, test_leakyrelu_default);
        do_test!($runner_fn, $runner_name, test_leakyrelu_default_expanded);
        //do_test!($runner_fn, $runner_name, test_leakyrelu_example);
        do_test!($runner_fn, $runner_name, test_leakyrelu_example_expanded);

        do_test!($runner_fn, $runner_name, test_leakyrelu_expanded);

        do_test!($runner_fn, $runner_name, test_less);
        do_test!($runner_fn, $runner_name, test_less_bcast);
        do_test!($runner_fn, $runner_name, test_less_equal);
        do_test!($runner_fn, $runner_name, test_less_equal_bcast);
        do_test!($runner_fn, $runner_name, test_less_equal_bcast_expanded);
        do_test!($runner_fn, $runner_name, test_less_equal_expanded);
        do_test!($runner_fn, $runner_name, test_log);
        do_test!($runner_fn, $runner_name, test_log_example);
        do_test!($runner_fn, $runner_name, test_logsoftmax_axis_0);
        //do_test!($runner_fn, $runner_name, test_logsoftmax_axis_0_expanded);
        //do_test!($runner_fn, $runner_name, test_logsoftmax_axis_0_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_logsoftmax_axis_1);
        //do_test!($runner_fn, $runner_name, test_logsoftmax_axis_1_expanded);
        //do_test!($runner_fn, $runner_name, test_logsoftmax_axis_1_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_logsoftmax_axis_2);
        //do_test!($runner_fn, $runner_name, test_logsoftmax_axis_2_expanded);
        //do_test!($runner_fn, $runner_name, test_logsoftmax_axis_2_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_logsoftmax_default_axis);
        //do_test!($runner_fn, $runner_name, test_logsoftmax_default_axis_expanded);
        //do_test!($runner_fn, $runner_name, test_logsoftmax_default_axis_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_logsoftmax_example_1);
        //do_test!($runner_fn, $runner_name, test_logsoftmax_example_1_expanded);
        //do_test!($runner_fn, $runner_name, test_logsoftmax_example_1_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_logsoftmax_large_number);
        //do_test!($runner_fn, $runner_name, test_logsoftmax_large_number_expanded);
        //do_test!($runner_fn, $runner_name, test_logsoftmax_large_number_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_logsoftmax_negative_axis);
        //do_test!($runner_fn, $runner_name, test_logsoftmax_negative_axis_expanded);
        //do_test!($runner_fn, $runner_name, test_logsoftmax_negative_axis_expanded_ver18);

        /*
        do_test!($runner_fn, $runner_name, test_loop11);
        do_test!($runner_fn, $runner_name, test_loop13_seq);
        do_test!($runner_fn, $runner_name, test_loop16_seq_none);*/

        /*
        do_test!($runner_fn, $runner_name, test_lppool_1d_default);
        do_test!($runner_fn, $runner_name, test_lppool_2d_default);
        do_test!($runner_fn, $runner_name, test_lppool_2d_dilations);
        do_test!($runner_fn, $runner_name, test_lppool_2d_pads);
        do_test!($runner_fn, $runner_name, test_lppool_2d_same_lower);
        do_test!($runner_fn, $runner_name, test_lppool_2d_same_upper);
        do_test!($runner_fn, $runner_name, test_lppool_2d_strides);
        do_test!($runner_fn, $runner_name, test_lppool_3d_default);*/

        /*
        do_test!($runner_fn, $runner_name, test_lrn);
        do_test!($runner_fn, $runner_name, test_lrn_default);*/

        /*
        do_test!($runner_fn, $runner_name, test_lstm_batchwise);
        do_test!($runner_fn, $runner_name, test_lstm_defaults);
        do_test!($runner_fn, $runner_name, test_lstm_with_initial_bias);
        do_test!($runner_fn, $runner_name, test_lstm_with_peepholes);*/

        do_test!($runner_fn, $runner_name, test_matmul_2d);
        do_test!($runner_fn, $runner_name, test_matmul_3d);
        do_test!($runner_fn, $runner_name, test_matmul_4d);
        //do_test!($runner_fn, $runner_name, test_matmulinteger);
        do_test!($runner_fn, $runner_name, test_max_example);
        do_test!($runner_fn, $runner_name, test_max_float16);
        do_test!($runner_fn, $runner_name, test_max_float32);
        do_test!($runner_fn, $runner_name, test_max_float64);
        do_test!($runner_fn, $runner_name, test_max_int16);
        do_test!($runner_fn, $runner_name, test_max_int32);
        do_test!($runner_fn, $runner_name, test_max_int64);
        do_test!($runner_fn, $runner_name, test_max_int8);
        do_test!($runner_fn, $runner_name, test_max_one_input);

        /*
        do_test!($runner_fn, $runner_name, test_maxpool_1d_default);
        do_test!($runner_fn, $runner_name, test_maxpool_2d_ceil);
        do_test!($runner_fn, $runner_name, test_maxpool_2d_ceil_output_size_reduce_by_one);
        do_test!($runner_fn, $runner_name, test_maxpool_2d_default);
        do_test!($runner_fn, $runner_name, test_maxpool_2d_dilations);
        do_test!($runner_fn, $runner_name, test_maxpool_2d_pads);
        do_test!($runner_fn, $runner_name, test_maxpool_2d_precomputed_pads);
        do_test!($runner_fn, $runner_name, test_maxpool_2d_precomputed_same_upper);
        do_test!($runner_fn, $runner_name, test_maxpool_2d_precomputed_strides);
        do_test!($runner_fn, $runner_name, test_maxpool_2d_same_lower);
        do_test!($runner_fn, $runner_name, test_maxpool_2d_same_upper);
        do_test!($runner_fn, $runner_name, test_maxpool_2d_strides);
        do_test!($runner_fn, $runner_name, test_maxpool_2d_uint8);
        do_test!($runner_fn, $runner_name, test_maxpool_3d_default);
        do_test!($runner_fn, $runner_name, test_maxpool_3d_dilations);
        do_test!($runner_fn, $runner_name, test_maxpool_3d_dilations_use_ref_impl);
        do_test!($runner_fn, $runner_name, test_maxpool_3d_dilations_use_ref_impl_large);
        do_test!($runner_fn, $runner_name, test_maxpool_with_argmax_2d_precomputed_pads);
        do_test!($runner_fn, $runner_name, test_maxpool_with_argmax_2d_precomputed_strides);*/

        do_test!($runner_fn, $runner_name, test_max_two_inputs);
        do_test!($runner_fn, $runner_name, test_max_uint16);
        do_test!($runner_fn, $runner_name, test_max_uint32);
        do_test!($runner_fn, $runner_name, test_max_uint64);
        do_test!($runner_fn, $runner_name, test_max_uint8);

        //do_test!($runner_fn, $runner_name, test_maxunpool_export_without_output_shape);
        //do_test!($runner_fn, $runner_name, test_maxunpool_export_with_output_shape);

        /*
        do_test!($runner_fn, $runner_name, test_mean_example);
        do_test!($runner_fn, $runner_name, test_mean_one_input);
        do_test!($runner_fn, $runner_name, test_mean_two_inputs);*/

        // do_test!($runner_fn, $runner_name, test_melweightmatrix);

        do_test!($runner_fn, $runner_name, test_min_example);
        do_test!($runner_fn, $runner_name, test_min_float16);
        do_test!($runner_fn, $runner_name, test_min_float32);
        do_test!($runner_fn, $runner_name, test_min_float64);
        do_test!($runner_fn, $runner_name, test_min_int16);
        do_test!($runner_fn, $runner_name, test_min_int32);
        do_test!($runner_fn, $runner_name, test_min_int64);
        do_test!($runner_fn, $runner_name, test_min_int8);
        do_test!($runner_fn, $runner_name, test_min_one_input);
        do_test!($runner_fn, $runner_name, test_min_two_inputs);
        do_test!($runner_fn, $runner_name, test_min_uint16);
        do_test!($runner_fn, $runner_name, test_min_uint32);
        do_test!($runner_fn, $runner_name, test_min_uint64);
        do_test!($runner_fn, $runner_name, test_min_uint8);

        //do_test!($runner_fn, $runner_name, test_mish);
        //do_test!($runner_fn, $runner_name, test_mish_expanded);

        do_test!($runner_fn, $runner_name, test_mod_broadcast);
        do_test!($runner_fn, $runner_name, test_mod_int64_fmod);
        do_test!($runner_fn, $runner_name, test_mod_mixed_sign_float16);
        do_test!($runner_fn, $runner_name, test_mod_mixed_sign_float32);
        do_test!($runner_fn, $runner_name, test_mod_mixed_sign_float64);
        //do_test!($runner_fn, $runner_name, test_mod_mixed_sign_int16);
        //do_test!($runner_fn, $runner_name, test_mod_mixed_sign_int32);
        //do_test!($runner_fn, $runner_name, test_mod_mixed_sign_int64);
        //do_test!($runner_fn, $runner_name, test_mod_mixed_sign_int8);
        do_test!($runner_fn, $runner_name, test_mod_uint16);
        do_test!($runner_fn, $runner_name, test_mod_uint32);
        do_test!($runner_fn, $runner_name, test_mod_uint64);
        do_test!($runner_fn, $runner_name, test_mod_uint8);

        /*
        do_test!($runner_fn, $runner_name, test_momentum);
        do_test!($runner_fn, $runner_name, test_momentum_multiple);*/

        do_test!($runner_fn, $runner_name, test_mul);
        do_test!($runner_fn, $runner_name, test_mul_bcast);
        do_test!($runner_fn, $runner_name, test_mul_example);
        do_test!($runner_fn, $runner_name, test_mul_uint8);

        /*
        do_test!($runner_fn, $runner_name, test_mvn);
        do_test!($runner_fn, $runner_name, test_mvn_expanded);
        do_test!($runner_fn, $runner_name, test_mvn_expanded_ver18);
        */

        do_test!($runner_fn, $runner_name, test_neg);
        do_test!($runner_fn, $runner_name, test_neg_example);

        //do_test!($runner_fn, $runner_name, test_nesterov_momentum);

        /*
        do_test!($runner_fn, $runner_name, test_nllloss_NC);
        do_test!($runner_fn, $runner_name, test_nllloss_NCd1);
        do_test!($runner_fn, $runner_name, test_nllloss_NCd1d2);
        do_test!($runner_fn, $runner_name, test_nllloss_NCd1d2d3d4d5_mean_weight);
        do_test!($runner_fn, $runner_name, test_nllloss_NCd1d2d3d4d5_mean_weight_expanded);
        do_test!($runner_fn, $runner_name, test_nllloss_NCd1d2d3d4d5_none_no_weight);
        do_test!($runner_fn, $runner_name, test_nllloss_NCd1d2d3d4d5_none_no_weight_expanded);
        do_test!($runner_fn, $runner_name, test_nllloss_NCd1d2d3_none_no_weight_negative_ii);
        do_test!($runner_fn, $runner_name, test_nllloss_NCd1d2d3_none_no_weight_negative_ii_expanded);
        do_test!($runner_fn, $runner_name, test_nllloss_NCd1d2d3_sum_weight_high_ii);
        do_test!($runner_fn, $runner_name, test_nllloss_NCd1d2d3_sum_weight_high_ii_expanded);
        do_test!($runner_fn, $runner_name, test_nllloss_NCd1d2_expanded);
        do_test!($runner_fn, $runner_name, test_nllloss_NCd1d2_no_weight_reduction_mean_ii);
        do_test!($runner_fn, $runner_name, test_nllloss_NCd1d2_no_weight_reduction_mean_ii_expanded);
        do_test!($runner_fn, $runner_name, test_nllloss_NCd1d2_reduction_mean);
        do_test!($runner_fn, $runner_name, test_nllloss_NCd1d2_reduction_mean_expanded);
        do_test!($runner_fn, $runner_name, test_nllloss_NCd1d2_reduction_sum);
        do_test!($runner_fn, $runner_name, test_nllloss_NCd1d2_reduction_sum_expanded);
        do_test!($runner_fn, $runner_name, test_nllloss_NCd1d2_with_weight);
        do_test!($runner_fn, $runner_name, test_nllloss_NCd1d2_with_weight_expanded);
        do_test!($runner_fn, $runner_name, test_nllloss_NCd1d2_with_weight_reduction_mean);
        do_test!($runner_fn, $runner_name, test_nllloss_NCd1d2_with_weight_reduction_mean_expanded);
        do_test!($runner_fn, $runner_name, test_nllloss_NCd1d2_with_weight_reduction_sum);
        do_test!($runner_fn, $runner_name, test_nllloss_NCd1d2_with_weight_reduction_sum_expanded);
        do_test!($runner_fn, $runner_name, test_nllloss_NCd1d2_with_weight_reduction_sum_ii);
        do_test!($runner_fn, $runner_name, test_nllloss_NCd1d2_with_weight_reduction_sum_ii_expanded);
        do_test!($runner_fn, $runner_name, test_nllloss_NCd1_expanded);
        do_test!($runner_fn, $runner_name, test_nllloss_NCd1_ii);
        do_test!($runner_fn, $runner_name, test_nllloss_NCd1_ii_expanded);
        do_test!($runner_fn, $runner_name, test_nllloss_NCd1_mean_weight_negative_ii);
        do_test!($runner_fn, $runner_name, test_nllloss_NCd1_mean_weight_negative_ii_expanded);
        do_test!($runner_fn, $runner_name, test_nllloss_NCd1_weight);
        do_test!($runner_fn, $runner_name, test_nllloss_NCd1_weight_expanded);
        do_test!($runner_fn, $runner_name, test_nllloss_NCd1_weight_ii);
        do_test!($runner_fn, $runner_name, test_nllloss_NCd1_weight_ii_expanded);
        do_test!($runner_fn, $runner_name, test_nllloss_NC_expanded);*/

        /*
        do_test!($runner_fn, $runner_name, test_nonmaxsuppression_center_point_box_format);
        do_test!($runner_fn, $runner_name, test_nonmaxsuppression_flipped_coordinates);
        do_test!($runner_fn, $runner_name, test_nonmaxsuppression_identical_boxes);
        do_test!($runner_fn, $runner_name, test_nonmaxsuppression_limit_output_size);
        do_test!($runner_fn, $runner_name, test_nonmaxsuppression_single_box);
        do_test!($runner_fn, $runner_name, test_nonmaxsuppression_suppress_by_IOU);
        do_test!($runner_fn, $runner_name, test_nonmaxsuppression_suppress_by_IOU_and_scores);
        do_test!($runner_fn, $runner_name, test_nonmaxsuppression_two_batches);
        do_test!($runner_fn, $runner_name, test_nonmaxsuppression_two_classes);*/

        do_test!($runner_fn, $runner_name, test_nonzero_example);
        do_test!($runner_fn, $runner_name, test_not_2d);
        do_test!($runner_fn, $runner_name, test_not_3d);
        do_test!($runner_fn, $runner_name, test_not_4d);

        /*
        do_test!($runner_fn, $runner_name, test_onehot_negative_indices);
        do_test!($runner_fn, $runner_name, test_onehot_with_axis);
        do_test!($runner_fn, $runner_name, test_onehot_with_negative_axis);
        do_test!($runner_fn, $runner_name, test_onehot_without_axis);*/

        /*
        do_test!($runner_fn, $runner_name, test_optional_get_element_optional_sequence);
        do_test!($runner_fn, $runner_name, test_optional_get_element_optional_tensor);
        do_test!($runner_fn, $runner_name, test_optional_get_element_sequence);
        do_test!($runner_fn, $runner_name, test_optional_get_element_tensor);
        do_test!($runner_fn, $runner_name, test_optional_has_element_empty_no_input_name_optional_input);
        do_test!($runner_fn, $runner_name, test_optional_has_element_empty_no_input_name_tensor_input);
        do_test!($runner_fn, $runner_name, test_optional_has_element_empty_no_input_optional_input);
        do_test!($runner_fn, $runner_name, test_optional_has_element_empty_no_input_tensor_input);
        do_test!($runner_fn, $runner_name, test_optional_has_element_empty_optional_input);
        do_test!($runner_fn, $runner_name, test_optional_has_element_optional_input);
        do_test!($runner_fn, $runner_name, test_optional_has_element_tensor_input);*/

        do_test!($runner_fn, $runner_name, test_or2d);
        do_test!($runner_fn, $runner_name, test_or3d);
        do_test!($runner_fn, $runner_name, test_or4d);
        do_test!($runner_fn, $runner_name, test_or_bcast3v1d);
        do_test!($runner_fn, $runner_name, test_or_bcast3v2d);
        do_test!($runner_fn, $runner_name, test_or_bcast4v2d);
        do_test!($runner_fn, $runner_name, test_or_bcast4v3d);
        do_test!($runner_fn, $runner_name, test_or_bcast4v4d);
        do_test!($runner_fn, $runner_name, test_pow);
        do_test!($runner_fn, $runner_name, test_pow_bcast_array);
        do_test!($runner_fn, $runner_name, test_pow_bcast_scalar);
        do_test!($runner_fn, $runner_name, test_pow_example);
        do_test!($runner_fn, $runner_name, test_pow_types_float32_int32);
        do_test!($runner_fn, $runner_name, test_pow_types_float32_int64);
        do_test!($runner_fn, $runner_name, test_pow_types_float32_uint32);
        do_test!($runner_fn, $runner_name, test_pow_types_float32_uint64);
        do_test!($runner_fn, $runner_name, test_pow_types_int32_float32);
        do_test!($runner_fn, $runner_name, test_pow_types_int32_int32);
        do_test!($runner_fn, $runner_name, test_pow_types_int64_float32);
        do_test!($runner_fn, $runner_name, test_pow_types_int64_int64);

        //do_test!($runner_fn, $runner_name, test_prelu_broadcast);
        do_test!($runner_fn, $runner_name, test_prelu_broadcast_expanded);
        //do_test!($runner_fn, $runner_name, test_prelu_example);
        do_test!($runner_fn, $runner_name, test_prelu_example_expanded);

        /*
        do_test!($runner_fn, $runner_name, test_qlinearconv);
        do_test!($runner_fn, $runner_name, test_qlinearmatmul_2D_int8_float16);
        do_test!($runner_fn, $runner_name, test_qlinearmatmul_2D_int8_float32);
        do_test!($runner_fn, $runner_name, test_qlinearmatmul_2D_uint8_float16);
        do_test!($runner_fn, $runner_name, test_qlinearmatmul_2D_uint8_float32);
        do_test!($runner_fn, $runner_name, test_qlinearmatmul_3D_int8_float16);
        do_test!($runner_fn, $runner_name, test_qlinearmatmul_3D_int8_float32);
        do_test!($runner_fn, $runner_name, test_qlinearmatmul_3D_uint8_float16);
        do_test!($runner_fn, $runner_name, test_qlinearmatmul_3D_uint8_float32);*/

        /*
        do_test!($runner_fn, $runner_name, test_quantizelinear);
        do_test!($runner_fn, $runner_name, test_quantizelinear_axis);
        do_test!($runner_fn, $runner_name, test_quantizelinear_blocked_asymmetric);
        do_test!($runner_fn, $runner_name, test_quantizelinear_blocked_symmetric);
        do_test!($runner_fn, $runner_name, test_quantizelinear_e4m3fn);
        do_test!($runner_fn, $runner_name, test_quantizelinear_e5m2);
        do_test!($runner_fn, $runner_name, test_quantizelinear_float4e2m1);
        do_test!($runner_fn, $runner_name, test_quantizelinear_int16);
        do_test!($runner_fn, $runner_name, test_quantizelinear_int4);
        do_test!($runner_fn, $runner_name, test_quantizelinear_uint16);
        do_test!($runner_fn, $runner_name, test_quantizelinear_uint4);*/

        do_test!($runner_fn, $runner_name, test_range_float_type_positive_delta);
        //do_test!($runner_fn, $runner_name, test_range_float_type_positive_delta_expanded);
        do_test!($runner_fn, $runner_name, test_range_int32_type_negative_delta);
        //do_test!($runner_fn, $runner_name, test_range_int32_type_negative_delta_expanded);

        do_test!($runner_fn, $runner_name, test_reciprocal);
        do_test!($runner_fn, $runner_name, test_reciprocal_example);

        /*
        do_test!($runner_fn, $runner_name, test_reduce_l1_default_axes_keepdims_example);
        do_test!($runner_fn, $runner_name, test_reduce_l1_default_axes_keepdims_example_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_l1_default_axes_keepdims_random);
        do_test!($runner_fn, $runner_name, test_reduce_l1_default_axes_keepdims_random_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_l1_do_not_keepdims_example);
        do_test!($runner_fn, $runner_name, test_reduce_l1_do_not_keepdims_example_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_l1_do_not_keepdims_random);
        do_test!($runner_fn, $runner_name, test_reduce_l1_do_not_keepdims_random_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_l1_empty_set);
        do_test!($runner_fn, $runner_name, test_reduce_l1_empty_set_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_l1_keep_dims_example);
        do_test!($runner_fn, $runner_name, test_reduce_l1_keep_dims_example_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_l1_keep_dims_random);
        do_test!($runner_fn, $runner_name, test_reduce_l1_keep_dims_random_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_l1_negative_axes_keep_dims_example);
        do_test!($runner_fn, $runner_name, test_reduce_l1_negative_axes_keep_dims_example_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_l1_negative_axes_keep_dims_random);
        do_test!($runner_fn, $runner_name, test_reduce_l1_negative_axes_keep_dims_random_expanded);*/

        /*
        do_test!($runner_fn, $runner_name, test_reduce_l2_default_axes_keepdims_example);
        do_test!($runner_fn, $runner_name, test_reduce_l2_default_axes_keepdims_example_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_l2_default_axes_keepdims_random);
        do_test!($runner_fn, $runner_name, test_reduce_l2_default_axes_keepdims_random_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_l2_do_not_keepdims_example);
        do_test!($runner_fn, $runner_name, test_reduce_l2_do_not_keepdims_example_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_l2_do_not_keepdims_random);
        do_test!($runner_fn, $runner_name, test_reduce_l2_do_not_keepdims_random_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_l2_empty_set);
        do_test!($runner_fn, $runner_name, test_reduce_l2_empty_set_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_l2_keep_dims_example);
        do_test!($runner_fn, $runner_name, test_reduce_l2_keep_dims_example_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_l2_keep_dims_random);
        do_test!($runner_fn, $runner_name, test_reduce_l2_keep_dims_random_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_l2_negative_axes_keep_dims_example);
        do_test!($runner_fn, $runner_name, test_reduce_l2_negative_axes_keep_dims_example_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_l2_negative_axes_keep_dims_random);
        do_test!($runner_fn, $runner_name, test_reduce_l2_negative_axes_keep_dims_random_expanded);*/

        /*
        do_test!($runner_fn, $runner_name, test_reduce_log_sum_asc_axes);
        do_test!($runner_fn, $runner_name, test_reduce_log_sum_asc_axes_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_log_sum_default);
        do_test!($runner_fn, $runner_name, test_reduce_log_sum_default_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_log_sum_desc_axes);
        do_test!($runner_fn, $runner_name, test_reduce_log_sum_desc_axes_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_log_sum_empty_set);
        do_test!($runner_fn, $runner_name, test_reduce_log_sum_empty_set_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_log_sum_exp_default_axes_keepdims_example);
        do_test!($runner_fn, $runner_name, test_reduce_log_sum_exp_default_axes_keepdims_example_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_log_sum_exp_default_axes_keepdims_random);
        do_test!($runner_fn, $runner_name, test_reduce_log_sum_exp_default_axes_keepdims_random_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_log_sum_exp_do_not_keepdims_example);
        do_test!($runner_fn, $runner_name, test_reduce_log_sum_exp_do_not_keepdims_example_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_log_sum_exp_do_not_keepdims_random);
        do_test!($runner_fn, $runner_name, test_reduce_log_sum_exp_do_not_keepdims_random_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_log_sum_exp_empty_set);
        do_test!($runner_fn, $runner_name, test_reduce_log_sum_exp_empty_set_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_log_sum_exp_keepdims_example);
        do_test!($runner_fn, $runner_name, test_reduce_log_sum_exp_keepdims_example_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_log_sum_exp_keepdims_random);
        do_test!($runner_fn, $runner_name, test_reduce_log_sum_exp_keepdims_random_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_log_sum_exp_negative_axes_keepdims_example);
        do_test!($runner_fn, $runner_name, test_reduce_log_sum_exp_negative_axes_keepdims_example_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_log_sum_exp_negative_axes_keepdims_random);
        do_test!($runner_fn, $runner_name, test_reduce_log_sum_exp_negative_axes_keepdims_random_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_log_sum_negative_axes);
        do_test!($runner_fn, $runner_name, test_reduce_log_sum_negative_axes_expanded);*/

        /*
        do_test!($runner_fn, $runner_name, test_reduce_max_bool_inputs);
        do_test!($runner_fn, $runner_name, test_reduce_max_default_axes_keepdim_example);
        do_test!($runner_fn, $runner_name, test_reduce_max_default_axes_keepdims_random);
        do_test!($runner_fn, $runner_name, test_reduce_max_do_not_keepdims_example);
        do_test!($runner_fn, $runner_name, test_reduce_max_do_not_keepdims_random);
        do_test!($runner_fn, $runner_name, test_reduce_max_empty_set);
        do_test!($runner_fn, $runner_name, test_reduce_max_keepdims_example);
        do_test!($runner_fn, $runner_name, test_reduce_max_keepdims_random);
        do_test!($runner_fn, $runner_name, test_reduce_max_negative_axes_keepdims_example);
        do_test!($runner_fn, $runner_name, test_reduce_max_negative_axes_keepdims_random);*/


        do_test!($runner_fn, $runner_name, test_reduce_mean_default_axes_keepdims_example);
        do_test!($runner_fn, $runner_name, test_reduce_mean_default_axes_keepdims_random);
        do_test!($runner_fn, $runner_name, test_reduce_mean_do_not_keepdims_example);
        do_test!($runner_fn, $runner_name, test_reduce_mean_do_not_keepdims_random);
        do_test!($runner_fn, $runner_name, test_reduce_mean_keepdims_example);
        do_test!($runner_fn, $runner_name, test_reduce_mean_keepdims_random);
        do_test!($runner_fn, $runner_name, test_reduce_mean_negative_axes_keepdims_example);
        do_test!($runner_fn, $runner_name, test_reduce_mean_negative_axes_keepdims_random);

        /*
        do_test!($runner_fn, $runner_name, test_reduce_min_bool_inputs);
        do_test!($runner_fn, $runner_name, test_reduce_min_default_axes_keepdims_example);
        do_test!($runner_fn, $runner_name, test_reduce_min_default_axes_keepdims_random);
        do_test!($runner_fn, $runner_name, test_reduce_min_do_not_keepdims_example);
        do_test!($runner_fn, $runner_name, test_reduce_min_do_not_keepdims_random);
        do_test!($runner_fn, $runner_name, test_reduce_min_empty_set);
        do_test!($runner_fn, $runner_name, test_reduce_min_keepdims_example);
        do_test!($runner_fn, $runner_name, test_reduce_min_keepdims_random);
        do_test!($runner_fn, $runner_name, test_reduce_min_negative_axes_keepdims_example);
        do_test!($runner_fn, $runner_name, test_reduce_min_negative_axes_keepdims_random);*/

        do_test!($runner_fn, $runner_name, test_reduce_prod_default_axes_keepdims_example);
        do_test!($runner_fn, $runner_name, test_reduce_prod_default_axes_keepdims_random);
        do_test!($runner_fn, $runner_name, test_reduce_prod_do_not_keepdims_example);
        do_test!($runner_fn, $runner_name, test_reduce_prod_do_not_keepdims_random);
        do_test!($runner_fn, $runner_name, test_reduce_prod_empty_set);
        do_test!($runner_fn, $runner_name, test_reduce_prod_keepdims_example);
        do_test!($runner_fn, $runner_name, test_reduce_prod_keepdims_random);
        do_test!($runner_fn, $runner_name, test_reduce_prod_negative_axes_keepdims_example);
        do_test!($runner_fn, $runner_name, test_reduce_prod_negative_axes_keepdims_random);

        do_test!($runner_fn, $runner_name, test_reduce_sum_default_axes_keepdims_example);
        do_test!($runner_fn, $runner_name, test_reduce_sum_default_axes_keepdims_random);
        do_test!($runner_fn, $runner_name, test_reduce_sum_do_not_keepdims_example);
        do_test!($runner_fn, $runner_name, test_reduce_sum_do_not_keepdims_random);
        do_test!($runner_fn, $runner_name, test_reduce_sum_empty_axes_input_noop);
        do_test!($runner_fn, $runner_name, test_reduce_sum_empty_axes_input_noop_example);
        do_test!($runner_fn, $runner_name, test_reduce_sum_empty_set);
        do_test!($runner_fn, $runner_name, test_reduce_sum_empty_set_non_reduced_axis_zero);
        do_test!($runner_fn, $runner_name, test_reduce_sum_keepdims_example);
        do_test!($runner_fn, $runner_name, test_reduce_sum_keepdims_random);
        do_test!($runner_fn, $runner_name, test_reduce_sum_negative_axes_keepdims_example);
        do_test!($runner_fn, $runner_name, test_reduce_sum_negative_axes_keepdims_random);

        /*
        do_test!($runner_fn, $runner_name, test_reduce_sum_square_default_axes_keepdims_example);
        do_test!($runner_fn, $runner_name, test_reduce_sum_square_default_axes_keepdims_example_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_sum_square_default_axes_keepdims_random);
        do_test!($runner_fn, $runner_name, test_reduce_sum_square_default_axes_keepdims_random_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_sum_square_do_not_keepdims_example);
        do_test!($runner_fn, $runner_name, test_reduce_sum_square_do_not_keepdims_example_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_sum_square_do_not_keepdims_random);
        do_test!($runner_fn, $runner_name, test_reduce_sum_square_do_not_keepdims_random_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_sum_square_empty_set);
        do_test!($runner_fn, $runner_name, test_reduce_sum_square_empty_set_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_sum_square_keepdims_example);
        do_test!($runner_fn, $runner_name, test_reduce_sum_square_keepdims_example_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_sum_square_keepdims_random);
        do_test!($runner_fn, $runner_name, test_reduce_sum_square_keepdims_random_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_sum_square_negative_axes_keepdims_example);
        do_test!($runner_fn, $runner_name, test_reduce_sum_square_negative_axes_keepdims_example_expanded);
        do_test!($runner_fn, $runner_name, test_reduce_sum_square_negative_axes_keepdims_random);
        do_test!($runner_fn, $runner_name, test_reduce_sum_square_negative_axes_keepdims_random_expanded);*/

        //do_test!($runner_fn, $runner_name, test_reflect_pad);

        /*
        do_test!($runner_fn, $runner_name, test_regex_full_match_basic);
        do_test!($runner_fn, $runner_name, test_regex_full_match_email_domain);
        do_test!($runner_fn, $runner_name, test_regex_full_match_empty);*/

        do_test!($runner_fn, $runner_name, test_relu);
        do_test!($runner_fn, $runner_name, test_relu_expanded_ver18);
        //do_test!($runner_fn, $runner_name, test_reshape_allowzero_reordered);
        do_test!($runner_fn, $runner_name, test_reshape_extended_dims);
        do_test!($runner_fn, $runner_name, test_reshape_negative_dim);
        do_test!($runner_fn, $runner_name, test_reshape_negative_extended_dims);
        do_test!($runner_fn, $runner_name, test_reshape_one_dim);
        do_test!($runner_fn, $runner_name, test_reshape_reduced_dims);
        do_test!($runner_fn, $runner_name, test_reshape_reordered_all_dims);
        do_test!($runner_fn, $runner_name, test_reshape_reordered_last_dims);
        do_test!($runner_fn, $runner_name, test_reshape_zero_and_negative_dim);
        do_test!($runner_fn, $runner_name, test_reshape_zero_dim);
        /*
        do_test!($runner_fn, $runner_name, test_resize_downsample_scales_cubic);
        do_test!($runner_fn, $runner_name, test_resize_downsample_scales_cubic_align_corners);
        do_test!($runner_fn, $runner_name, test_resize_downsample_scales_cubic_A_n0p5_exclude_outside);
        do_test!($runner_fn, $runner_name, test_resize_downsample_scales_cubic_antialias);
        do_test!($runner_fn, $runner_name, test_resize_downsample_scales_linear);
        do_test!($runner_fn, $runner_name, test_resize_downsample_scales_linear_align_corners);
        do_test!($runner_fn, $runner_name, test_resize_downsample_scales_linear_antialias);
        do_test!($runner_fn, $runner_name, test_resize_downsample_scales_linear_half_pixel_symmetric);
        do_test!($runner_fn, $runner_name, test_resize_downsample_scales_nearest);
        do_test!($runner_fn, $runner_name, test_resize_downsample_sizes_cubic);
        do_test!($runner_fn, $runner_name, test_resize_downsample_sizes_cubic_antialias);
        do_test!($runner_fn, $runner_name, test_resize_downsample_sizes_linear_antialias);
        do_test!($runner_fn, $runner_name, test_resize_downsample_sizes_linear_pytorch_half_pixel);
        do_test!($runner_fn, $runner_name, test_resize_downsample_sizes_nearest);
        do_test!($runner_fn, $runner_name, test_resize_downsample_sizes_nearest_not_larger);
        do_test!($runner_fn, $runner_name, test_resize_downsample_sizes_nearest_not_smaller);
        do_test!($runner_fn, $runner_name, test_resize_tf_crop_and_resize);
        do_test!($runner_fn, $runner_name, test_resize_tf_crop_and_resize_axes_2_3);
        do_test!($runner_fn, $runner_name, test_resize_tf_crop_and_resize_axes_3_2);
        do_test!($runner_fn, $runner_name, test_resize_tf_crop_and_resize_extrapolation_value);
        do_test!($runner_fn, $runner_name, test_resize_upsample_scales_cubic);
        do_test!($runner_fn, $runner_name, test_resize_upsample_scales_cubic_align_corners);
        do_test!($runner_fn, $runner_name, test_resize_upsample_scales_cubic_A_n0p5_exclude_outside);
        do_test!($runner_fn, $runner_name, test_resize_upsample_scales_cubic_asymmetric);
        do_test!($runner_fn, $runner_name, test_resize_upsample_scales_linear);
        do_test!($runner_fn, $runner_name, test_resize_upsample_scales_linear_align_corners);
        do_test!($runner_fn, $runner_name, test_resize_upsample_scales_linear_half_pixel_symmetric);
        do_test!($runner_fn, $runner_name, test_resize_upsample_scales_nearest);
        do_test!($runner_fn, $runner_name, test_resize_upsample_scales_nearest_axes_2_3);
        do_test!($runner_fn, $runner_name, test_resize_upsample_scales_nearest_axes_3_2);
        do_test!($runner_fn, $runner_name, test_resize_upsample_sizes_cubic);
        do_test!($runner_fn, $runner_name, test_resize_upsample_sizes_nearest);
        do_test!($runner_fn, $runner_name, test_resize_upsample_sizes_nearest_axes_2_3);
        do_test!($runner_fn, $runner_name, test_resize_upsample_sizes_nearest_axes_3_2);
        do_test!($runner_fn, $runner_name, test_resize_upsample_sizes_nearest_ceil_half_pixel);
        do_test!($runner_fn, $runner_name, test_resize_upsample_sizes_nearest_floor_align_corners);
        do_test!($runner_fn, $runner_name, test_resize_upsample_sizes_nearest_not_larger);
        do_test!($runner_fn, $runner_name, test_resize_upsample_sizes_nearest_not_smaller);
        do_test!($runner_fn, $runner_name, test_resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric);
        */

        //do_test!($runner_fn, $runner_name, test_reversesequence_batch);
        //do_test!($runner_fn, $runner_name, test_reversesequence_time);


        do_test!($runner_fn, $runner_name, test_rms_normalization_2d_axis0);
        do_test!($runner_fn, $runner_name, test_rms_normalization_2d_axis0_expanded);
        do_test!($runner_fn, $runner_name, test_rms_normalization_2d_axis1);
        do_test!($runner_fn, $runner_name, test_rms_normalization_2d_axis1_expanded);
        do_test!($runner_fn, $runner_name, test_rms_normalization_2d_axis_negative_1);
        do_test!($runner_fn, $runner_name, test_rms_normalization_2d_axis_negative_1_expanded);
        do_test!($runner_fn, $runner_name, test_rms_normalization_2d_axis_negative_2);
        do_test!($runner_fn, $runner_name, test_rms_normalization_2d_axis_negative_2_expanded);
        do_test!($runner_fn, $runner_name, test_rms_normalization_3d_axis0_epsilon);
        do_test!($runner_fn, $runner_name, test_rms_normalization_3d_axis0_epsilon_expanded);
        do_test!($runner_fn, $runner_name, test_rms_normalization_3d_axis1_epsilon);
        do_test!($runner_fn, $runner_name, test_rms_normalization_3d_axis1_epsilon_expanded);
        do_test!($runner_fn, $runner_name, test_rms_normalization_3d_axis2_epsilon);
        do_test!($runner_fn, $runner_name, test_rms_normalization_3d_axis2_epsilon_expanded);
        do_test!($runner_fn, $runner_name, test_rms_normalization_3d_axis_negative_1_epsilon);
        do_test!($runner_fn, $runner_name, test_rms_normalization_3d_axis_negative_1_epsilon_expanded);
        do_test!($runner_fn, $runner_name, test_rms_normalization_3d_axis_negative_2_epsilon);
        do_test!($runner_fn, $runner_name, test_rms_normalization_3d_axis_negative_2_epsilon_expanded);
        do_test!($runner_fn, $runner_name, test_rms_normalization_3d_axis_negative_3_epsilon);
        do_test!($runner_fn, $runner_name, test_rms_normalization_3d_axis_negative_3_epsilon_expanded);
        do_test!($runner_fn, $runner_name, test_rms_normalization_4d_axis0);
        do_test!($runner_fn, $runner_name, test_rms_normalization_4d_axis0_expanded);
        do_test!($runner_fn, $runner_name, test_rms_normalization_4d_axis1);
        do_test!($runner_fn, $runner_name, test_rms_normalization_4d_axis1_expanded);
        do_test!($runner_fn, $runner_name, test_rms_normalization_4d_axis2);
        do_test!($runner_fn, $runner_name, test_rms_normalization_4d_axis2_expanded);
        do_test!($runner_fn, $runner_name, test_rms_normalization_4d_axis3);
        do_test!($runner_fn, $runner_name, test_rms_normalization_4d_axis3_expanded);
        do_test!($runner_fn, $runner_name, test_rms_normalization_4d_axis_negative_1);
        do_test!($runner_fn, $runner_name, test_rms_normalization_4d_axis_negative_1_expanded);
        do_test!($runner_fn, $runner_name, test_rms_normalization_4d_axis_negative_2);
        do_test!($runner_fn, $runner_name, test_rms_normalization_4d_axis_negative_2_expanded);
        do_test!($runner_fn, $runner_name, test_rms_normalization_4d_axis_negative_3);
        do_test!($runner_fn, $runner_name, test_rms_normalization_4d_axis_negative_3_expanded);
        do_test!($runner_fn, $runner_name, test_rms_normalization_4d_axis_negative_4);
        do_test!($runner_fn, $runner_name, test_rms_normalization_4d_axis_negative_4_expanded);
        do_test!($runner_fn, $runner_name, test_rms_normalization_default_axis);
        do_test!($runner_fn, $runner_name, test_rms_normalization_default_axis_expanded);

        //do_test!($runner_fn, $runner_name, test_rnn_seq_length);

        /*
        do_test!($runner_fn, $runner_name, test_roialign_aligned_false);
        do_test!($runner_fn, $runner_name, test_roialign_aligned_true);
        do_test!($runner_fn, $runner_name, test_roialign_mode_max);*/


        do_test!($runner_fn, $runner_name, test_rotary_embedding);
        do_test!($runner_fn, $runner_name, test_rotary_embedding_3d_input);
        do_test!($runner_fn, $runner_name, test_rotary_embedding_3d_input_expanded);
        do_test!($runner_fn, $runner_name, test_rotary_embedding_expanded);
        //do_test!($runner_fn, $runner_name, test_rotary_embedding_interleaved);
        //do_test!($runner_fn, $runner_name, test_rotary_embedding_interleaved_expanded);
        do_test!($runner_fn, $runner_name, test_rotary_embedding_no_position_ids);
        do_test!($runner_fn, $runner_name, test_rotary_embedding_no_position_ids_expanded);
        //do_test!($runner_fn, $runner_name, test_rotary_embedding_no_position_ids_interleaved);
        //do_test!($runner_fn, $runner_name, test_rotary_embedding_no_position_ids_interleaved_expanded);
        do_test!($runner_fn, $runner_name, test_rotary_embedding_no_position_ids_rotary_dim);
        do_test!($runner_fn, $runner_name, test_rotary_embedding_no_position_ids_rotary_dim_expanded);
        //do_test!($runner_fn, $runner_name, test_rotary_embedding_with_interleaved_rotary_dim);
        //do_test!($runner_fn, $runner_name, test_rotary_embedding_with_interleaved_rotary_dim_expanded);
        do_test!($runner_fn, $runner_name, test_rotary_embedding_with_rotary_dim);
        do_test!($runner_fn, $runner_name, test_rotary_embedding_with_rotary_dim_expanded);

        do_test!($runner_fn, $runner_name, test_round);

        do_test!($runner_fn, $runner_name, test_scan9_sum);
        //do_test!($runner_fn, $runner_name, test_scan_sum);

        /*
        do_test!($runner_fn, $runner_name, test_scatter_elements_with_axis);
        do_test!($runner_fn, $runner_name, test_scatter_elements_with_duplicate_indices);
        do_test!($runner_fn, $runner_name, test_scatter_elements_with_negative_indices);
        do_test!($runner_fn, $runner_name, test_scatter_elements_without_axis);
        do_test!($runner_fn, $runner_name, test_scatter_elements_with_reduction_max);
        do_test!($runner_fn, $runner_name, test_scatter_elements_with_reduction_min);*/

        /*
        do_test!($runner_fn, $runner_name, test_scatternd);
        do_test!($runner_fn, $runner_name, test_scatternd_add);
        do_test!($runner_fn, $runner_name, test_scatternd_max);
        do_test!($runner_fn, $runner_name, test_scatternd_min);
        do_test!($runner_fn, $runner_name, test_scatternd_multiply);
        do_test!($runner_fn, $runner_name, test_scatter_with_axis);
        do_test!($runner_fn, $runner_name, test_scatter_without_axis);*/

        /*
        do_test!($runner_fn, $runner_name, test_sce_mean);
        do_test!($runner_fn, $runner_name, test_sce_mean_3d);
        do_test!($runner_fn, $runner_name, test_sce_mean_3d_expanded);
        do_test!($runner_fn, $runner_name, test_sce_mean_3d_log_prob);
        do_test!($runner_fn, $runner_name, test_sce_mean_3d_log_prob_expanded);
        do_test!($runner_fn, $runner_name, test_sce_mean_expanded);
        do_test!($runner_fn, $runner_name, test_sce_mean_log_prob);
        do_test!($runner_fn, $runner_name, test_sce_mean_log_prob_expanded);
        do_test!($runner_fn, $runner_name, test_sce_mean_no_weight_ii);
        do_test!($runner_fn, $runner_name, test_sce_mean_no_weight_ii_3d);
        do_test!($runner_fn, $runner_name, test_sce_mean_no_weight_ii_3d_expanded);
        do_test!($runner_fn, $runner_name, test_sce_mean_no_weight_ii_3d_log_prob);
        do_test!($runner_fn, $runner_name, test_sce_mean_no_weight_ii_3d_log_prob_expanded);
        do_test!($runner_fn, $runner_name, test_sce_mean_no_weight_ii_4d);
        do_test!($runner_fn, $runner_name, test_sce_mean_no_weight_ii_4d_expanded);
        do_test!($runner_fn, $runner_name, test_sce_mean_no_weight_ii_4d_log_prob);
        do_test!($runner_fn, $runner_name, test_sce_mean_no_weight_ii_4d_log_prob_expanded);
        do_test!($runner_fn, $runner_name, test_sce_mean_no_weight_ii_expanded);
        do_test!($runner_fn, $runner_name, test_sce_mean_no_weight_ii_log_prob);
        do_test!($runner_fn, $runner_name, test_sce_mean_no_weight_ii_log_prob_expanded);
        do_test!($runner_fn, $runner_name, test_sce_mean_weight);
        do_test!($runner_fn, $runner_name, test_sce_mean_weight_expanded);
        do_test!($runner_fn, $runner_name, test_sce_mean_weight_ii);
        do_test!($runner_fn, $runner_name, test_sce_mean_weight_ii_3d);
        do_test!($runner_fn, $runner_name, test_sce_mean_weight_ii_3d_expanded);
        do_test!($runner_fn, $runner_name, test_sce_mean_weight_ii_3d_log_prob);
        do_test!($runner_fn, $runner_name, test_sce_mean_weight_ii_3d_log_prob_expanded);
        do_test!($runner_fn, $runner_name, test_sce_mean_weight_ii_4d);
        do_test!($runner_fn, $runner_name, test_sce_mean_weight_ii_4d_expanded);
        do_test!($runner_fn, $runner_name, test_sce_mean_weight_ii_4d_log_prob);
        do_test!($runner_fn, $runner_name, test_sce_mean_weight_ii_4d_log_prob_expanded);
        do_test!($runner_fn, $runner_name, test_sce_mean_weight_ii_expanded);
        do_test!($runner_fn, $runner_name, test_sce_mean_weight_ii_log_prob);
        do_test!($runner_fn, $runner_name, test_sce_mean_weight_ii_log_prob_expanded);
        do_test!($runner_fn, $runner_name, test_sce_mean_weight_log_prob);
        do_test!($runner_fn, $runner_name, test_sce_mean_weight_log_prob_expanded);
        do_test!($runner_fn, $runner_name, test_sce_NCd1d2d3d4d5_mean_weight);
        do_test!($runner_fn, $runner_name, test_sce_NCd1d2d3d4d5_mean_weight_expanded);
        do_test!($runner_fn, $runner_name, test_sce_NCd1d2d3d4d5_mean_weight_log_prob);
        do_test!($runner_fn, $runner_name, test_sce_NCd1d2d3d4d5_mean_weight_log_prob_expanded);
        do_test!($runner_fn, $runner_name, test_sce_NCd1d2d3d4d5_none_no_weight);
        do_test!($runner_fn, $runner_name, test_sce_NCd1d2d3d4d5_none_no_weight_expanded);
        do_test!($runner_fn, $runner_name, test_sce_NCd1d2d3d4d5_none_no_weight_log_prob);
        do_test!($runner_fn, $runner_name, test_sce_NCd1d2d3d4d5_none_no_weight_log_prob_expanded);
        do_test!($runner_fn, $runner_name, test_sce_NCd1d2d3_none_no_weight_negative_ii);
        do_test!($runner_fn, $runner_name, test_sce_NCd1d2d3_none_no_weight_negative_ii_expanded);
        do_test!($runner_fn, $runner_name, test_sce_NCd1d2d3_none_no_weight_negative_ii_log_prob);
        do_test!($runner_fn, $runner_name, test_sce_NCd1d2d3_none_no_weight_negative_ii_log_prob_expanded);
        do_test!($runner_fn, $runner_name, test_sce_NCd1d2d3_sum_weight_high_ii);
        do_test!($runner_fn, $runner_name, test_sce_NCd1d2d3_sum_weight_high_ii_expanded);
        do_test!($runner_fn, $runner_name, test_sce_NCd1d2d3_sum_weight_high_ii_log_prob);
        do_test!($runner_fn, $runner_name, test_sce_NCd1d2d3_sum_weight_high_ii_log_prob_expanded);
        do_test!($runner_fn, $runner_name, test_sce_NCd1_mean_weight_negative_ii);
        do_test!($runner_fn, $runner_name, test_sce_NCd1_mean_weight_negative_ii_expanded);
        do_test!($runner_fn, $runner_name, test_sce_NCd1_mean_weight_negative_ii_log_prob);
        do_test!($runner_fn, $runner_name, test_sce_NCd1_mean_weight_negative_ii_log_prob_expanded);
        do_test!($runner_fn, $runner_name, test_sce_none);
        do_test!($runner_fn, $runner_name, test_sce_none_expanded);
        do_test!($runner_fn, $runner_name, test_sce_none_log_prob);
        do_test!($runner_fn, $runner_name, test_sce_none_log_prob_expanded);
        do_test!($runner_fn, $runner_name, test_sce_none_weights);
        do_test!($runner_fn, $runner_name, test_sce_none_weights_expanded);
        do_test!($runner_fn, $runner_name, test_sce_none_weights_log_prob);
        do_test!($runner_fn, $runner_name, test_sce_none_weights_log_prob_expanded);
        do_test!($runner_fn, $runner_name, test_sce_sum);
        do_test!($runner_fn, $runner_name, test_sce_sum_expanded);
        do_test!($runner_fn, $runner_name, test_sce_sum_log_prob);
        do_test!($runner_fn, $runner_name, test_sce_sum_log_prob_expanded);*/

        /*
        do_test!($runner_fn, $runner_name, test_selu);
        do_test!($runner_fn, $runner_name, test_selu_default);
        do_test!($runner_fn, $runner_name, test_selu_default_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_selu_example);
        do_test!($runner_fn, $runner_name, test_selu_example_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_selu_expanded_ver18);*/


        /*
        do_test!($runner_fn, $runner_name, test_sequence_insert_at_back);
        do_test!($runner_fn, $runner_name, test_sequence_insert_at_front);
        do_test!($runner_fn, $runner_name, test_sequence_map_add_1_sequence_1_tensor);
        do_test!($runner_fn, $runner_name, test_sequence_map_add_1_sequence_1_tensor_expanded);
        do_test!($runner_fn, $runner_name, test_sequence_map_add_2_sequences);
        do_test!($runner_fn, $runner_name, test_sequence_map_add_2_sequences_expanded);
        do_test!($runner_fn, $runner_name, test_sequence_map_extract_shapes);
        do_test!($runner_fn, $runner_name, test_sequence_map_extract_shapes_expanded);
        do_test!($runner_fn, $runner_name, test_sequence_map_identity_1_sequence);
        do_test!($runner_fn, $runner_name, test_sequence_map_identity_1_sequence_1_tensor);
        do_test!($runner_fn, $runner_name, test_sequence_map_identity_1_sequence_1_tensor_expanded);
        do_test!($runner_fn, $runner_name, test_sequence_map_identity_1_sequence_expanded);
        do_test!($runner_fn, $runner_name, test_sequence_map_identity_2_sequences);
        do_test!($runner_fn, $runner_name, test_sequence_map_identity_2_sequences_expanded);*/

        do_test!($runner_fn, $runner_name, test_shape);
        do_test!($runner_fn, $runner_name, test_shape_clip_end);
        do_test!($runner_fn, $runner_name, test_shape_clip_start);
        do_test!($runner_fn, $runner_name, test_shape_end_1);
        do_test!($runner_fn, $runner_name, test_shape_end_negative_1);
        do_test!($runner_fn, $runner_name, test_shape_example);
        do_test!($runner_fn, $runner_name, test_shape_start_1);
        do_test!($runner_fn, $runner_name, test_shape_start_1_end_2);
        do_test!($runner_fn, $runner_name, test_shape_start_1_end_negative_1);
        do_test!($runner_fn, $runner_name, test_shape_start_negative_1);

        /*
        do_test!($runner_fn, $runner_name, test_shrink_hard);
        do_test!($runner_fn, $runner_name, test_shrink_hard_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_shrink_soft);
        do_test!($runner_fn, $runner_name, test_shrink_soft_expanded_ver18);*/

        do_test!($runner_fn, $runner_name, test_sigmoid);
        do_test!($runner_fn, $runner_name, test_sigmoid_example);
        do_test!($runner_fn, $runner_name, test_sign);

        /*
        do_test!($runner_fn, $runner_name, test_simple_rnn_batchwise);
        do_test!($runner_fn, $runner_name, test_simple_rnn_defaults);
        do_test!($runner_fn, $runner_name, test_simple_rnn_with_initial_bias);*/

        do_test!($runner_fn, $runner_name, test_sin);
        do_test!($runner_fn, $runner_name, test_sin_example);
        do_test!($runner_fn, $runner_name, test_sinh);
        do_test!($runner_fn, $runner_name, test_sinh_example);
        do_test!($runner_fn, $runner_name, test_size);
        do_test!($runner_fn, $runner_name, test_size_example);
        do_test!($runner_fn, $runner_name, test_slice);
        do_test!($runner_fn, $runner_name, test_slice_default_axes);
        do_test!($runner_fn, $runner_name, test_slice_default_steps);
        do_test!($runner_fn, $runner_name, test_slice_end_out_of_bounds);
        do_test!($runner_fn, $runner_name, test_slice_neg);
        do_test!($runner_fn, $runner_name, test_slice_negative_axes);
        //do_test!($runner_fn, $runner_name, test_slice_neg_steps);
        do_test!($runner_fn, $runner_name, test_slice_start_out_of_bounds);
        do_test!($runner_fn, $runner_name, test_softmax_axis_0);
        //do_test!($runner_fn, $runner_name, test_softmax_axis_0_expanded);
        //do_test!($runner_fn, $runner_name, test_softmax_axis_0_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_softmax_axis_1);
        //do_test!($runner_fn, $runner_name, test_softmax_axis_1_expanded);
        //do_test!($runner_fn, $runner_name, test_softmax_axis_1_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_softmax_axis_2);
        //do_test!($runner_fn, $runner_name, test_softmax_axis_2_expanded);
        //do_test!($runner_fn, $runner_name, test_softmax_axis_2_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_softmax_default_axis);
        //do_test!($runner_fn, $runner_name, test_softmax_default_axis_expanded);
        //do_test!($runner_fn, $runner_name, test_softmax_default_axis_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_softmax_example);
        //do_test!($runner_fn, $runner_name, test_softmax_example_expanded);
        //do_test!($runner_fn, $runner_name, test_softmax_example_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_softmax_large_number);
        //do_test!($runner_fn, $runner_name, test_softmax_large_number_expanded);
        //do_test!($runner_fn, $runner_name, test_softmax_large_number_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_softmax_negative_axis);
        //do_test!($runner_fn, $runner_name, test_softmax_negative_axis_expanded);
        //do_test!($runner_fn, $runner_name, test_softmax_negative_axis_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_softplus);
        do_test!($runner_fn, $runner_name, test_softplus_example);
        //do_test!($runner_fn, $runner_name, test_softplus_example_expanded_ver18);
        //do_test!($runner_fn, $runner_name, test_softplus_expanded_ver18);

        /*
        do_test!($runner_fn, $runner_name, test_softsign);
        do_test!($runner_fn, $runner_name, test_softsign_example);
        do_test!($runner_fn, $runner_name, test_softsign_example_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_softsign_expanded_ver18);*/


        //do_test!($runner_fn, $runner_name, test_spacetodepth);
        //test_do!($runner_fn, $runner_name, test_spacetodepth_example);


        do_test!($runner_fn, $runner_name, test_split_1d_uneven_split_opset18);
        do_test!($runner_fn, $runner_name, test_split_2d_uneven_split_opset18);
        //do_test!($runner_fn, $runner_name, test_split_equal_parts_1d_opset13);
        do_test!($runner_fn, $runner_name, test_split_equal_parts_1d_opset18);
        do_test!($runner_fn, $runner_name, test_split_equal_parts_2d);
        //do_test!($runner_fn, $runner_name, test_split_equal_parts_2d_opset13);
        //do_test!($runner_fn, $runner_name, test_split_equal_parts_default_axis_opset13);
        do_test!($runner_fn, $runner_name, test_split_equal_parts_default_axis_opset18);
        //do_test!($runner_fn, $runner_name, test_split_to_sequence_1);
        //do_test!($runner_fn, $runner_name, test_split_to_sequence_2);
        //do_test!($runner_fn, $runner_name, test_split_to_sequence_nokeepdims);


        do_test!($runner_fn, $runner_name, test_split_variable_parts_1d_opset13);
        do_test!($runner_fn, $runner_name, test_split_variable_parts_1d_opset18);
        do_test!($runner_fn, $runner_name, test_split_variable_parts_2d_opset13);
        do_test!($runner_fn, $runner_name, test_split_variable_parts_2d_opset18);
        do_test!($runner_fn, $runner_name, test_split_variable_parts_default_axis_opset13);
        do_test!($runner_fn, $runner_name, test_split_variable_parts_default_axis_opset18);
        do_test!($runner_fn, $runner_name, test_split_zero_size_splits_opset13);
        do_test!($runner_fn, $runner_name, test_split_zero_size_splits_opset18);
        do_test!($runner_fn, $runner_name, test_sqrt);
        do_test!($runner_fn, $runner_name, test_sqrt_example);
        do_test!($runner_fn, $runner_name, test_squeeze);
        do_test!($runner_fn, $runner_name, test_squeeze_negative_axes);

        /*
        do_test!($runner_fn, $runner_name, test_stft);
        do_test!($runner_fn, $runner_name, test_stft_with_window);*/

        /*
        do_test!($runner_fn, $runner_name, test_string_concat);
        do_test!($runner_fn, $runner_name, test_string_concat_broadcasting);
        do_test!($runner_fn, $runner_name, test_string_concat_empty_string);
        do_test!($runner_fn, $runner_name, test_string_concat_utf8);
        do_test!($runner_fn, $runner_name, test_string_concat_zero_dimensional);
        do_test!($runner_fn, $runner_name, test_string_split_basic);
        do_test!($runner_fn, $runner_name, test_string_split_consecutive_delimiters);
        do_test!($runner_fn, $runner_name, test_string_split_empty_string_delimiter);
        do_test!($runner_fn, $runner_name, test_string_split_empty_tensor);
        do_test!($runner_fn, $runner_name, test_string_split_maxsplit);
        do_test!($runner_fn, $runner_name, test_string_split_no_delimiter);*/

        /*
        do_test!($runner_fn, $runner_name, test_strnormalizer_export_monday_casesensintive_lower);
        do_test!($runner_fn, $runner_name, test_strnormalizer_export_monday_casesensintive_nochangecase);
        do_test!($runner_fn, $runner_name, test_strnormalizer_export_monday_casesensintive_upper);
        do_test!($runner_fn, $runner_name, test_strnormalizer_export_monday_empty_output);
        do_test!($runner_fn, $runner_name, test_strnormalizer_export_monday_insensintive_upper_twodim);
        do_test!($runner_fn, $runner_name, test_strnormalizer_nostopwords_nochangecase);*/

        do_test!($runner_fn, $runner_name, test_sub);
        do_test!($runner_fn, $runner_name, test_sub_bcast);
        do_test!($runner_fn, $runner_name, test_sub_example);
        do_test!($runner_fn, $runner_name, test_sub_uint8);
        /*
        do_test!($runner_fn, $runner_name, test_sum_example);
        do_test!($runner_fn, $runner_name, test_sum_one_input);
        do_test!($runner_fn, $runner_name, test_sum_two_inputs);*/

        do_test!($runner_fn, $runner_name, test_tan);
        do_test!($runner_fn, $runner_name, test_tan_example);
        do_test!($runner_fn, $runner_name, test_tanh);
        do_test!($runner_fn, $runner_name, test_tanh_example);

        /*
        do_test!($runner_fn, $runner_name, test_tfidfvectorizer_tf_batch_onlybigrams_skip0);
        do_test!($runner_fn, $runner_name, test_tfidfvectorizer_tf_batch_onlybigrams_skip5);
        do_test!($runner_fn, $runner_name, test_tfidfvectorizer_tf_batch_uniandbigrams_skip5);
        do_test!($runner_fn, $runner_name, test_tfidfvectorizer_tf_onlybigrams_levelempty);
        do_test!($runner_fn, $runner_name, test_tfidfvectorizer_tf_only_bigrams_skip0);
        do_test!($runner_fn, $runner_name, test_tfidfvectorizer_tf_onlybigrams_skip5);
        do_test!($runner_fn, $runner_name, test_tfidfvectorizer_tf_uniandbigrams_skip5);*/

        /*
        do_test!($runner_fn, $runner_name, test_thresholdedrelu);
        do_test!($runner_fn, $runner_name, test_thresholdedrelu_default);
        do_test!($runner_fn, $runner_name, test_thresholdedrelu_default_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_thresholdedrelu_example);
        do_test!($runner_fn, $runner_name, test_thresholdedrelu_example_expanded_ver18);
        do_test!($runner_fn, $runner_name, test_thresholdedrelu_expanded_ver18);*/

        //do_test!($runner_fn, $runner_name, test_tile);
        //do_test!($runner_fn, $runner_name, test_tile_precomputed);

        /*
        do_test!($runner_fn, $runner_name, test_top_k);
        do_test!($runner_fn, $runner_name, test_top_k_negative_axis);
        do_test!($runner_fn, $runner_name, test_top_k_same_values);
        do_test!($runner_fn, $runner_name, test_top_k_same_values_2d);
        do_test!($runner_fn, $runner_name, test_top_k_same_values_largest);
        do_test!($runner_fn, $runner_name, test_top_k_smallest);
        do_test!($runner_fn, $runner_name, test_top_k_uint64);*/

        /*
        do_test!($runner_fn, $runner_name, test_training_dropout);
        do_test!($runner_fn, $runner_name, test_training_dropout_default);
        do_test!($runner_fn, $runner_name, test_training_dropout_default_mask);
        do_test!($runner_fn, $runner_name, test_training_dropout_mask);
        do_test!($runner_fn, $runner_name, test_training_dropout_zero_ratio);
        do_test!($runner_fn, $runner_name, test_training_dropout_zero_ratio_mask);*/

        do_test!($runner_fn, $runner_name, test_transpose_all_permutations_0);
        do_test!($runner_fn, $runner_name, test_transpose_all_permutations_1);
        do_test!($runner_fn, $runner_name, test_transpose_all_permutations_2);
        do_test!($runner_fn, $runner_name, test_transpose_all_permutations_3);
        do_test!($runner_fn, $runner_name, test_transpose_all_permutations_4);
        do_test!($runner_fn, $runner_name, test_transpose_all_permutations_5);
        do_test!($runner_fn, $runner_name, test_transpose_default);

        /*
        do_test!($runner_fn, $runner_name, test_tril);
        do_test!($runner_fn, $runner_name, test_tril_neg);
        do_test!($runner_fn, $runner_name, test_tril_one_row_neg);
        do_test!($runner_fn, $runner_name, test_tril_out_neg);
        do_test!($runner_fn, $runner_name, test_tril_out_pos);
        do_test!($runner_fn, $runner_name, test_tril_pos);
        do_test!($runner_fn, $runner_name, test_tril_square);
        do_test!($runner_fn, $runner_name, test_tril_square_neg);
        do_test!($runner_fn, $runner_name, test_tril_zero);
        do_test!($runner_fn, $runner_name, test_triu);
        do_test!($runner_fn, $runner_name, test_triu_neg);
        do_test!($runner_fn, $runner_name, test_triu_one_row);
        do_test!($runner_fn, $runner_name, test_triu_out_neg_out);
        do_test!($runner_fn, $runner_name, test_triu_out_pos);
        do_test!($runner_fn, $runner_name, test_triu_pos);
        do_test!($runner_fn, $runner_name, test_triu_square);
        do_test!($runner_fn, $runner_name, test_triu_square_neg);
        do_test!($runner_fn, $runner_name, test_triu_zero);*/

        /*
        do_test!($runner_fn, $runner_name, test_unique_length_1);
        do_test!($runner_fn, $runner_name, test_unique_not_sorted_without_axis);
        do_test!($runner_fn, $runner_name, test_unique_sorted_with_axis);
        do_test!($runner_fn, $runner_name, test_unique_sorted_with_axis_3d);
        do_test!($runner_fn, $runner_name, test_unique_sorted_with_negative_axis);
        do_test!($runner_fn, $runner_name, test_unique_sorted_without_axis);*/

        do_test!($runner_fn, $runner_name, test_unsqueeze_axis_0);
        do_test!($runner_fn, $runner_name, test_unsqueeze_axis_1);
        do_test!($runner_fn, $runner_name, test_unsqueeze_axis_2);
        do_test!($runner_fn, $runner_name, test_unsqueeze_negative_axes);
        do_test!($runner_fn, $runner_name, test_unsqueeze_three_axes);
        do_test!($runner_fn, $runner_name, test_unsqueeze_two_axes);
        do_test!($runner_fn, $runner_name, test_unsqueeze_unsorted_axes);

        //do_test!($runner_fn, $runner_name, test_upsample_nearest);

        do_test!($runner_fn, $runner_name, test_where_example);
        do_test!($runner_fn, $runner_name, test_where_long_example);

        //do_test!($runner_fn, $runner_name, test_wrap_pad);

        do_test!($runner_fn, $runner_name, test_xor2d);
        do_test!($runner_fn, $runner_name, test_xor3d);
        do_test!($runner_fn, $runner_name, test_xor4d);
        do_test!($runner_fn, $runner_name, test_xor_bcast3v1d);
        do_test!($runner_fn, $runner_name, test_xor_bcast3v2d);
        do_test!($runner_fn, $runner_name, test_xor_bcast4v2d);
        do_test!($runner_fn, $runner_name, test_xor_bcast4v3d);
        do_test!($runner_fn, $runner_name, test_xor_bcast4v4d);


    }
}

do_tests!(run_ndarray_test, ndarray);
#[cfg(feature = "vulkan")]
do_tests!(run_vulkan_test, vulkan);

//#[cfg(feature = "candle")]
//do_tests!(run_candle_test, candle);
