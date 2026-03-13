//! Tests that validate `infer()` correctness by running the validation harness
//! against ONNX node test models.

use prost::Message;
use std::collections::HashMap;
use std::path::Path;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::graph::GlobalId;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::onnx::TensorProto;
use whisper_tensor::symbolic_graph::SymbolicGraphMutator;
use whisper_tensor::tensor_rank::DynRank;

/// Load an ONNX test directory and run the infer validation harness.
///
/// The directory should contain:
/// - `model.onnx`
/// - `test_data_set_0/input_*.pb` and `test_data_set_0/output_*.pb`
fn validate_infer_for_onnx_dir(dir: &Path) {
    let model_path = dir.join("model.onnx");
    if !model_path.exists() {
        panic!("Model not found: {}", model_path.display());
    }

    let model_bytes = std::fs::read(&model_path).unwrap();
    let rng = &mut rand::rng();
    let (graph, tensor_store) = SymbolicGraphMutator::from_onnx_bytes(&model_bytes, rng, None)
        .unwrap()
        .get_inner();

    // Load test data set 0 -- use tensor names from protobuf
    let test_data_dir = dir.join("test_data_set_0");
    let mut user_inputs: HashMap<String, NumericTensor<DynRank>> = HashMap::new();
    for entry in std::fs::read_dir(&test_data_dir).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("input_") && name.ends_with(".pb") {
            let data = std::fs::read(entry.path()).unwrap();
            let tensor_proto = TensorProto::decode(data.as_slice()).unwrap();
            let tensor: NumericTensor<DynRank> =
                NDArrayNumericTensor::<DynRank>::try_from(&tensor_proto)
                    .unwrap()
                    .into();
            user_inputs.insert(tensor_proto.name, tensor);
        }
    }

    // Build full inputs (initialized tensors + user inputs)
    let initialized_tensors = graph.get_initialized_tensors(&tensor_store);
    let mut all_inputs: HashMap<GlobalId, NumericTensor<DynRank>> = initialized_tensors;
    let tensors_by_name = graph.get_tensors_by_name();
    for (name, tensor) in &user_inputs {
        let tensor_id = tensors_by_name[name];
        all_inputs.insert(tensor_id, tensor.clone());
    }

    // Generate milli graph
    let milli_graph = graph.generate_milli_graph(rng);

    // Run validation
    let report = milli_graph.validate_infer_against_eval(&all_inputs);

    // Report results
    eprintln!(
        "  {} pass, {} unable-to-infer, {} failures",
        report.pass_count, report.unable_to_infer_count, report.failure_count
    );

    assert_eq!(
        report.failure_count,
        0,
        "Infer validation failures for {}:\n{}",
        dir.display(),
        report
    );
}

// -- Tests using ONNX node test suite --
// These cover a range of op types: unary, binary, shape manipulation, reductions.

macro_rules! validate_infer_test {
    ($test_name:ident) => {
        #[allow(non_snake_case)]
        #[test]
        fn $test_name() {
            let dir = std::path::PathBuf::from(concat!(
                "libs/onnx/onnx/backend/test/data/node/",
                stringify!($test_name)
            ));
            validate_infer_for_onnx_dir(&dir);
        }
    };
}

// Unary ops
validate_infer_test!(test_abs);
validate_infer_test!(test_relu);
validate_infer_test!(test_sigmoid);
validate_infer_test!(test_sqrt);
validate_infer_test!(test_tanh);
validate_infer_test!(test_exp);
validate_infer_test!(test_log);
validate_infer_test!(test_neg);
validate_infer_test!(test_floor);
validate_infer_test!(test_ceil);

// Binary ops
validate_infer_test!(test_add);
validate_infer_test!(test_add_bcast);
validate_infer_test!(test_sub);
validate_infer_test!(test_mul);
validate_infer_test!(test_div);

// Shape ops
validate_infer_test!(test_reshape_extended_dims);
validate_infer_test!(test_reshape_negative_dim);
validate_infer_test!(test_reshape_one_dim);
validate_infer_test!(test_reshape_reduced_dims);
validate_infer_test!(test_transpose_default);
validate_infer_test!(test_squeeze);
validate_infer_test!(test_unsqueeze_axis_0);
validate_infer_test!(test_unsqueeze_axis_1);
validate_infer_test!(test_unsqueeze_axis_2);

// Reduce ops
validate_infer_test!(test_reduce_sum_default_axes_keepdims_example);
validate_infer_test!(test_reduce_mean_default_axes_keepdims_example);

// Gather / Slice / Concat
validate_infer_test!(test_gather_0);
validate_infer_test!(test_gather_1);
validate_infer_test!(test_slice);
validate_infer_test!(test_slice_neg);
validate_infer_test!(test_concat_1d_axis_0);
validate_infer_test!(test_concat_2d_axis_0);
validate_infer_test!(test_concat_2d_axis_1);

// MatMul
validate_infer_test!(test_matmul_2d);
validate_infer_test!(test_matmul_3d);

// Cast
validate_infer_test!(test_cast_FLOAT_to_DOUBLE);

// Shape op (returns shape as tensor)
validate_infer_test!(test_shape);
validate_infer_test!(test_shape_example);

// Where
validate_infer_test!(test_where_example);

// Expand
validate_infer_test!(test_expand_dim_changed);
validate_infer_test!(test_expand_dim_unchanged);

// Pow
validate_infer_test!(test_pow);
validate_infer_test!(test_pow_bcast_scalar);
