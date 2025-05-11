// tests/onnx_node_tests.rs

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Read;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::Once;
use prost::Message;
use crate::{RuntimeBackend, RuntimeEnvironment, NumericTensor, RuntimeError, RuntimeModel};
use crate::dtype::{DType, DTypeError};
use crate::eval_backend::EvalBackend;
use crate::ndarray_backend::NDArrayNumericTensor;
use crate::onnx::TensorProto;
use crate::symbolic_graph::ONNXDecodingError;
use crate::tensor_rank::DynRank;

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
    OtherError(#[from] anyhow::Error)
}

impl From<String> for TestError {
    fn from(value: String) -> Self {
        TestError::ErrorS(value)
    }
}

static INIT: Once = Once::new();

// Structure to hold a test data set (inputs and expected outputs)
struct TestDataSet {
    inputs: HashMap<String, Vec<u8>>,    // Raw protobuf data
    outputs: HashMap<String, Vec<u8>>,   // Raw protobuf data
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
            if path.is_dir() && path.file_name()?.to_string_lossy().starts_with("test_data_set_") {
                if let Some(data_set) = TestDataSet::from_directory(&path) {
                    test_data_sets.push(data_set);
                }
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
    fn run(&self, backend: RuntimeBackend) -> Result<(), TestError> {
        log::info!("Running test: {}", self.name);

        // Load the model
        let model_bytes = fs::read(&self.model_path)
            .map_err(|e| format!("Failed to read model file: {}", e))?;

        let mut model = RuntimeModel::load_onnx(
            &model_bytes,
            backend.clone(),
            RuntimeEnvironment::default()
        ).map_err(|e| format!("Failed to load model: {:?}", e))?;

        // Run each test data set
        for (i, test_data_set) in self.test_data_sets.iter().enumerate() {
            log::info!("  Running test data set {}", i);

            // Convert protobuf inputs to NumericTensor
            let inputs = test_data_set.parse_inputs()
                .map_err(|e| format!("Failed to parse inputs: {}", e))?;

            // Run the model
            let outputs = model.run(inputs)
                .map_err(|e| format!("Model execution failed: {:?}", e))?;

            // Parse expected outputs
            let expected = test_data_set.parse_outputs()
                .map_err(|e| format!("Failed to parse expected outputs: {}", e))?;

            // Compare outputs with expected values
            for (name, expected_tensor) in &expected {
                let actual_tensor = outputs.get(name)
                    .ok_or_else(|| format!("Output '{}' not found in model results", name))?;

                // Compare tensors with tolerance
                self.compare_tensors(actual_tensor, expected_tensor)?;
            }
        }

        Ok(())
    }

    // Compare tensors with tolerance
    fn compare_tensors(&self, actual: &NumericTensor<DynRank>, expected: &NumericTensor<DynRank>) -> Result<(), TestError> {
        // Check shapes match
        if actual.dtype() != expected.dtype() {
            Err(format!(
                "Data type mismatch: actual {:?} vs expected {:?}", actual.dtype(), expected.dtype()))?
        }

        // Compare shapes
        if actual.shape() != expected.shape() {
            Err(format!(
                "Shape mismatch: actual {:?} vs expected {:?}", actual.shape(), expected.shape()
            ))?;
        }

        // Compare values when cast to 64 bit
        let actual_values: Vec<f64> = actual.cast(DType::F64, &EvalBackend::NDArray).unwrap().to_ndarray().unwrap().flatten().try_to_vec().unwrap();
        let expected_values: Vec<f64> = expected.cast(DType::F64, &EvalBackend::NDArray).unwrap().to_ndarray().unwrap().flatten().try_to_vec().unwrap();

        for (actual_value, expected_value) in actual_values.iter().zip(expected_values.iter()) {
            let abs_diff = (actual_value - expected_value).abs();
            let tolerance = self.atol + self.rtol * expected_value.abs();
            if abs_diff > tolerance {
                Err(format!("Value mismatch: actual {} vs expected {}", actual_value, expected_value))?;
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
        for (_index, (_, proto_data)) in self.inputs.iter().enumerate() {
            let tensor_proto = TensorProto::decode(proto_data.as_slice()).unwrap();
            let tensor: NumericTensor<DynRank> = NDArrayNumericTensor::try_from(&tensor_proto)?.into();

            // Use a name based on index (you'll need a better naming strategy)
            result.insert(tensor_proto.name, tensor);
        }

        Ok(result)
    }

    // Parse output protobuf data into NumericTensors
    fn parse_outputs(&self) -> Result<HashMap<String, NumericTensor<DynRank>>, ONNXDecodingError> {
        let mut result = HashMap::new();

        for (_index, (_, proto_data)) in self.outputs.iter().enumerate() {
            let tensor_proto = TensorProto::decode(proto_data.as_slice()).unwrap();
            let tensor: NumericTensor<DynRank> = NDArrayNumericTensor::try_from(&tensor_proto)?.into();
            
            result.insert(tensor_proto.name, tensor);
        }

        Ok(result)
    }
}

// Function to parse TensorProto to NumericTensor
fn parse_tensor_proto(proto_data: &[u8]) -> Result<NumericTensor<DynRank>, String> {
    let tensor_proto = TensorProto::decode(proto_data).unwrap();
    Ok(NDArrayNumericTensor::try_from(&tensor_proto).unwrap().into())
}

// Function to discover and run all node tests
fn run_node_tests(backend: RuntimeBackend) -> (usize, usize) {
    // Path to the ONNX node tests
    let node_test_path = PathBuf::from(
        std::env::var("ONNX_NODE_TEST_PATH")
            .unwrap_or_else(|_| "libs/onnx/onnx/backend/test/data/node".to_string())
    );

    if !node_test_path.exists() {
        panic!("ONNX node test path not found: {}", node_test_path.display());
    }

    let mut total = 0;
    let mut passed = 0;

    // Iterate through test directories
    for entry in fs::read_dir(node_test_path).expect("Failed to read node test directory") {
        let entry = entry.expect("Failed to read directory entry");
        let path = entry.path();

        if path.is_dir() {
            total += 1;

            // Parse and run the test
            if let Some(test) = OnnxNodeTest::from_directory(&path) {
                match test.run(backend.clone()) {
                    Ok(_) => {
                        log::info!("✅ Test passed: {}", test.name);
                        passed += 1;
                    },
                    Err(error) => {
                        log::error!("❌ Test failed: {}\nError: {}", test.name, error);
                    }
                }
            } else {
                log::warn!("⚠️ Could not parse test: {}", path.display());
            }
        }
    }

    (passed, total)
}

// Create test cases for different backends
// This macro helps avoid code duplication for different backends
macro_rules! backend_tests {
    ($backend:expr, $name:ident) => {
        #[test]
        fn $name() {
            let (passed, total) = run_node_tests($backend);
            println!("\nPassed {}/{} tests", passed, total);

            // Fail the test if not all tests passed
            assert_eq!(passed, total, "Not all ONNX node tests passed");
        }
    };
}

// Define tests for each backend
backend_tests!(RuntimeBackend::Eval(EvalBackend::NDArray), test_ndarray_backend);

/*#[cfg(feature = "candle")]
backend_tests!(
    RuntimeBackend::Eval(EvalBackend::Candle(candle_core::Device::Cpu)),
    test_eval_candle_backend
);

#[cfg(feature = "candle")]
backend_tests!(
    RuntimeBackend::Candle,
    test_candle_backend
);

#[cfg(feature = "ort")]
backend_tests!(RuntimeBackend::ORT, test_ort_backend);*/

// Optional: Run individual tests for specific operators
#[test]
fn test_add_operator() {
    let node_test_paths = vec![PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_add")];
    
    for path in node_test_paths {
        if !path.exists() {
            println!("Test directory not found, skipping");
            return;
        }

        if let Some(test) = OnnxNodeTest::from_directory(&path) {
            test.run(RuntimeBackend::Eval(EvalBackend::NDArray)).expect("Add operator test failed");
        } else {
            panic!("Could not parse add operator test");
        }
    }
}

#[test]
fn test_pow_operator() {
    let node_test_paths = vec![PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_pow"), PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_pow_example")];

    for path in node_test_paths {
        if !path.exists() {
            println!("Test directory not found, skipping");
            return;
        }

        if let Some(test) = OnnxNodeTest::from_directory(&path) {
            test.run(RuntimeBackend::Eval(EvalBackend::NDArray)).expect("operator test failed");
        } else {
            panic!("Could not parse operator test");
        }
    }
}

#[test]
fn test_gather_operator() {
    let node_test_paths = vec![
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_gather_0"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_gather_1"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_gather_2d_indices")
    ];

    for path in node_test_paths {
        if !path.exists() {
            println!("Test directory not found, skipping");
            return;
        }

        if let Some(test) = OnnxNodeTest::from_directory(&path) {
            test.run(RuntimeBackend::Eval(EvalBackend::NDArray)).expect("operator test failed");
        } else {
            panic!("Could not parse operator test");
        }
    }
}

#[test]
fn test_groupnorm_operator() {
    let node_test_paths = vec![
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_group_normalization_example"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_group_normalization_example_expanded"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_group_normalization_epsilon"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_group_normalization_epsilon_expanded"),
    ];

    for path in node_test_paths {
        if !path.exists() {
            println!("Test directory not found, skipping");
            return;
        }

        if let Some(test) = OnnxNodeTest::from_directory(&path) {
            test.run(RuntimeBackend::Eval(EvalBackend::NDArray)).expect("operator test failed");
        } else {
            panic!("Could not parse operator test");
        }
    }
}

#[test]
fn test_layernorm_operator() {
    let node_test_paths = vec![
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_layer_normalization_2d_axis0"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_layer_normalization_2d_axis1"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_layer_normalization_2d_axis_negative_1"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_layer_normalization_2d_axis_negative_2"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_layer_normalization_3d_axis0_epsilon"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_layer_normalization_3d_axis1_epsilon"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_layer_normalization_3d_axis2_epsilon"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_layer_normalization_3d_axis_negative_1_epsilon"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_layer_normalization_3d_axis_negative_2_epsilon"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_layer_normalization_3d_axis_negative_3_epsilon"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_layer_normalization_4d_axis0"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_layer_normalization_4d_axis1"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_layer_normalization_4d_axis2"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_layer_normalization_4d_axis3"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_layer_normalization_4d_axis_negative_1"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_layer_normalization_4d_axis_negative_2"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_layer_normalization_4d_axis_negative_3"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_layer_normalization_4d_axis_negative_4"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_layer_normalization_default_axis"),
    ];

    for path in node_test_paths {
        if !path.exists() {
            println!("Test directory not found, skipping");
            return;
        }

        if let Some(test) = OnnxNodeTest::from_directory(&path) {
            test.run(RuntimeBackend::Eval(EvalBackend::NDArray)).expect("operator test failed");
        } else {
            panic!("Could not parse operator test");
        }
    }
}


#[test]
fn test_layernorm_expanded() {
    let node_test_paths = vec![
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_layer_normalization_2d_axis0_expanded"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_layer_normalization_2d_axis1_expanded"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_layer_normalization_2d_axis_negative_1_expanded"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_layer_normalization_3d_axis1_epsilon_expanded"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_layer_normalization_3d_axis2_epsilon_expanded"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_layer_normalization_4d_axis1_expanded"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_layer_normalization_4d_axis2_expanded"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_layer_normalization_4d_axis3_expanded"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_layer_normalization_4d_axis_negative_1_expanded"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_layer_normalization_4d_axis_negative_2_expanded"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_layer_normalization_4d_axis_negative_3_expanded"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_layer_normalization_4d_default_axis_expanded"),
    ];

    for path in node_test_paths {
        if !path.exists() {
            println!("Test directory not found, skipping");
            return;
        }

        if let Some(test) = OnnxNodeTest::from_directory(&path) {
            test.run(RuntimeBackend::Eval(EvalBackend::NDArray)).expect("operator test failed");
        } else {
            panic!("Could not parse operator test");
        }
    }
}

#[test]
fn test_lpnorm_operator() {
    let node_test_paths = vec![
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_lpnorm_2d_axis0"),
    ];

    for path in node_test_paths {
        if !path.exists() {
            println!("Test directory not found, skipping");
            return;
        }

        if let Some(test) = OnnxNodeTest::from_directory(&path) {
            test.run(RuntimeBackend::Eval(EvalBackend::NDArray)).expect("operator test failed");
        } else {
            panic!("Could not parse operator test");
        }
    }
}

#[test]
fn test_shape_operator() {
    let node_test_paths = vec![
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_shape"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_shape_clip_end"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_shape_clip_start"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_shape_end_1"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_shape_start_1")
    ];

    for path in node_test_paths {
        if !path.exists() {
            println!("Test directory not found, skipping");
            return;
        }

        if let Some(test) = OnnxNodeTest::from_directory(&path) {
            test.run(RuntimeBackend::Eval(EvalBackend::NDArray)).expect("operator test failed");
        } else {
            panic!("Could not parse operator test");
        }
    }
}

#[test]
fn test_size_operator() {
    let node_test_paths = vec![
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_size"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_size_example"),
    ];

    for path in node_test_paths {
        if !path.exists() {
            println!("Test directory not found, skipping");
            return;
        }

        if let Some(test) = OnnxNodeTest::from_directory(&path) {
            test.run(RuntimeBackend::Eval(EvalBackend::NDArray)).expect("operator test failed");
        } else {
            panic!("Could not parse operator test");
        }
    }
}

#[test]
fn test_reduce_mean_operator() {
    let node_test_paths = vec![
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_reduce_mean_default_axes_keepdims_example"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_reduce_mean_default_axes_keepdims_random"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_reduce_mean_do_not_keepdims_example"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_reduce_mean_do_not_keepdims_random"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_reduce_mean_keepdims_example"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_reduce_mean_keepdims_random"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_reduce_mean_negative_axes_keepdims_example"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_reduce_mean_negative_axes_keepdims_random"),
    ];

    for path in node_test_paths {
        if !path.exists() {
            println!("Test directory not found, skipping");
            return;
        }

        if let Some(test) = OnnxNodeTest::from_directory(&path) {
            test.run(RuntimeBackend::Eval(EvalBackend::NDArray)).expect("operator test failed");
        } else {
            panic!("Could not parse operator test");
        }
    }
}

#[test]
fn test_reduce_sum_operator() {
    let node_test_paths = vec![
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_reduce_sum_keepdims_example"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_reduce_sum_default_axes_keepdims_example"),
    ];

    for path in node_test_paths {
        if !path.exists() {
            println!("Test directory not found, skipping");
            return;
        }

        if let Some(test) = OnnxNodeTest::from_directory(&path) {
            test.run(RuntimeBackend::Eval(EvalBackend::NDArray)).expect("operator test failed");
        } else {
            panic!("Could not parse operator test");
        }
    }
}

#[test]
fn test_squeeze_operator() {
    let node_test_paths = vec![
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_squeeze"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_squeeze_negative_axes"),
    ];

    for path in node_test_paths {
        if !path.exists() {
            println!("Test directory not found, skipping");
            return;
        }

        if let Some(test) = OnnxNodeTest::from_directory(&path) {
            test.run(RuntimeBackend::Eval(EvalBackend::NDArray)).expect("operator test failed");
        } else {
            panic!("Could not parse operator test");
        }
    }
}

#[test]
fn test_unsqueeze_operator() {
    let node_test_paths = vec![
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_unsqueeze"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_unsqueeze_negative_axes"),
    ];

    for path in node_test_paths {
        if !path.exists() {
            println!("Test directory not found, skipping");
            return;
        }

        if let Some(test) = OnnxNodeTest::from_directory(&path) {
            test.run(RuntimeBackend::Eval(EvalBackend::NDArray)).expect("operator test failed");
        } else {
            panic!("Could not parse operator test");
        }
    }
}

#[test]
fn test_range_operator() {
    let node_test_paths = vec![
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_range_float_type_positive_delta"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_range_int32_type_negative_delta"),
    ];

    for path in node_test_paths {
        if !path.exists() {
            println!("Test directory not found, skipping");
            return;
        }

        if let Some(test) = OnnxNodeTest::from_directory(&path) {
            test.run(RuntimeBackend::Eval(EvalBackend::NDArray)).expect("operator test failed");
        } else {
            panic!("Could not parse operator test");
        }
    }
}

#[test]
fn test_flatten_operator() {
    let node_test_paths = vec![
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_flatten_axis0"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_flatten_axis1"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_flatten_axis2"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_flatten_axis3"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_flatten_default_axis"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_flatten_negative_axis1"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_flatten_negative_axis2"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_flatten_negative_axis3"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_flatten_negative_axis4"),
    ];

    for path in node_test_paths {
        if !path.exists() {
            println!("Test directory not found, skipping");
            return;
        }

        if let Some(test) = OnnxNodeTest::from_directory(&path) {
            test.run(RuntimeBackend::Eval(EvalBackend::NDArray)).expect("operator test failed");
        } else {
            panic!("Could not parse operator test");
        }
    }
}

#[test]
fn test_round_operator() {
    let node_test_paths = vec![
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_round"),
    ];

    for path in node_test_paths {
        if !path.exists() {
            println!("Test directory not found, skipping");
            return;
        }

        if let Some(test) = OnnxNodeTest::from_directory(&path) {
            test.run(RuntimeBackend::Eval(EvalBackend::NDArray)).expect("operator test failed");
        } else {
            panic!("Could not parse operator test");
        }
    }
}

#[test]
fn test_isinf_operator() {
    let node_test_paths = vec![
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_isinf"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_isinf_float16"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_isinf_negative"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_isinf_positive"),

    ];

    for path in node_test_paths {
        if !path.exists() {
            println!("Test directory not found, skipping");
            return;
        }

        if let Some(test) = OnnxNodeTest::from_directory(&path) {
            test.run(RuntimeBackend::Eval(EvalBackend::NDArray)).expect("operator test failed");
        } else {
            panic!("Could not parse operator test");
        }
    }
}

#[test]
fn test_mod_operator() {
    let node_test_paths = vec![
        //PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_mod_broadcast"),
        //PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_mod_int64_fmod"),
        //PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_mod_mixed_sign_float16"),
        //PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_mod_mixed_sign_float32"),
        //PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_mod_mixed_sign_float64"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_mod_mixed_sign_int64"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_mod_mixed_sign_int32"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_mod_mixed_sign_int16"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_mod_mixed_sign_int8"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_mod_uint64"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_mod_uint32"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_mod_uint16"),
        PathBuf::from("libs/onnx/onnx/backend/test/data/node/test_mod_uint8"),
    ];

    for path in node_test_paths {
        if !path.exists() {
            println!("Test directory not found, skipping");
            return;
        }

        if let Some(test) = OnnxNodeTest::from_directory(&path) {
            test.run(RuntimeBackend::Eval(EvalBackend::NDArray)).expect("operator test failed");
        } else {
            panic!("Could not parse operator test");
        }
    }
}
