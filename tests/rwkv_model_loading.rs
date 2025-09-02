use std::path::PathBuf;

use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::model::{Model, ModelExecutionRuntime};

fn find_rwkv_pth() -> Option<PathBuf> {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.push("test_models");
    let rd = std::fs::read_dir(&dir).ok()?;
    for entry in rd.flatten() {
        let p = entry.path();
        if p.is_file() {
            if let Some(ext) = p.extension() {
                if ext == "pth"
                    && p.file_name()
                        .and_then(|s| s.to_str())
                        .map(|s| s.to_lowercase())
                        .map(|s| s.contains("rwkv") || s.contains("rwkv7") || s.contains("world"))
                        .unwrap_or(false)
                {
                    return Some(p);
                }
            }
        }
    }
    None
}

#[test]
fn rwkv01b_model_loads() {
    // Use the vendored RWKV 0.1B pth in test_models via importer identify_and_load
    let Some(pth_path) = find_rwkv_pth() else {
        eprintln!("Skipping: no RWKV .pth found under test_models/");
        return;
    };
    // Use EmbeddedData output method to avoid external file dependencies
    for strategy in [whisper_tensor_import::onnx_graph::WeightStorageStrategy::EmbeddedData] {
        let onnx_bytes = whisper_tensor_import::identify_and_load(
            &pth_path,
            strategy,
            Some(whisper_tensor_import::ModelTypeHint::RWKV7),
        )
        .expect("import rwkv7 to onnx");
        let model = Model::new_from_onnx(&onnx_bytes).expect("model loads");

        // Ensure we can construct Eval backend and load constant tensors (which should trigger
        // lazy external loading for any external_data entries)
        let mut eval = EvalBackend::NDArray;
        let mut cache = whisper_tensor::backends::ModelLoadedTensorCache::default();
        model
            .load_tensors(&mut cache, &mut eval)
            .expect("tensors load");

        // Check graph has at least one input and one output
        assert!(
            !model.get_symbolic_graph().get_inputs().is_empty(),
            "model has inputs"
        );
        assert!(
            !model.get_symbolic_graph().get_outputs().is_empty(),
            "model has outputs"
        );
    }
    return;
}

#[test]
fn rwkv01b_single_step_runs_shape_sanity() {
    // This is a smoke test: we try to run a single forward step with zeroed inputs
    // respecting the model's declared input shapes (with small batch/time if symbolic).
    let Some(pth_path) = find_rwkv_pth() else {
        eprintln!("Skipping: no RWKV .pth found under test_models/");
        return;
    };
    let onnx_bytes = whisper_tensor_import::identify_and_load(
        &pth_path,
        whisper_tensor_import::onnx_graph::WeightStorageStrategy::EmbeddedData,
        Some(whisper_tensor_import::ModelTypeHint::RWKV7),
    )
    .expect("import rwkv7 to onnx");
    let model = Model::new_from_onnx(&onnx_bytes).expect("model loads");

    let mut eval = EvalBackend::NDArray;
    let mut observer = ();

    // Prepare zeroed inputs by introspecting model input tensor info
    let input_infos = model.get_input_tensor_info().expect("introspect inputs");
    use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
    use whisper_tensor::{dtype::DType, numeric_tensor::NumericTensor};

    let mut inputs = std::collections::HashMap::new();
    for (name, (dtype, shape_desc)) in input_infos.into_iter() {
        // Replace unknown dims with small test sizes
        let mut shape: Vec<u64> = shape_desc.into_iter().map(|d| d.unwrap_or(1)).collect();
        // Clamp sequence or time-like dims if obviously large
        for s in &mut shape {
            if *s == 0 {
                *s = 1;
            }
        }
        let numel: usize = shape.iter().product::<u64>() as usize;
        let zeros = match dtype {
            DType::F32 => vec![0u8; numel * 4],
            DType::BF16 | DType::F16 => vec![0u8; numel * 2],
            DType::I64 | DType::U64 => vec![0u8; numel * 8],
            DType::I32 | DType::U32 | DType::U16 | DType::I16 => vec![0u8; numel * 4],
            DType::U8 | DType::I8 | DType::BOOL => vec![0u8; numel],
            DType::F64 => vec![0u8; numel * 8],
            DType::STRING => panic!("string input not supported in test"),
        };
        let nd = NDArrayNumericTensor::from_raw_data(&zeros, dtype, shape.clone())
            .expect("build zero tensor");
        inputs.insert(name, NumericTensor::NDArray(nd));
    }

    // Run a single step on Eval runtime
    let mut runtime = ModelExecutionRuntime::Eval(EvalBackend::NDArray);
    let outputs = model
        .run(inputs, &mut observer, &mut runtime)
        .expect("run ok");
    assert!(!outputs.is_empty(), "has outputs");

    // Basic sanity: each output tensor matches declared shape/dtype automatically in run()
}

#[test]
fn rwkv01b_model_loads_with_binfile() {
    // Validate that the BinFile strategy works by writing weights to a temp file
    let Some(pth_path) = find_rwkv_pth() else {
        eprintln!("Skipping: no RWKV .pth found under test_models/");
        return;
    };

    let tempdir = tempfile::tempdir().expect("create tempdir");
    let bin_path = tempdir.path().join("weights.bin");

    // Build ONNX with external_data pointing at weights.bin (relative name is embedded)
    let onnx_bytes = whisper_tensor_import::identify_and_load(
        &pth_path,
        whisper_tensor_import::onnx_graph::WeightStorageStrategy::BinFile(bin_path.clone()),
        Some(whisper_tensor_import::ModelTypeHint::RWKV7),
    )
    .expect("import rwkv7 to onnx (BinFile)");

    // Ensure the side-channel bin file exists and is non-empty
    let meta = std::fs::metadata(&bin_path).expect("bin file exists");
    assert!(meta.len() > 0, "bin file should be non-empty");

    // The ONNX external_data 'location' is just the file name. Switch CWD to the tempdir
    // so the model loader can find the side-channel file when lazily loading tensors.
    let old_cwd = std::env::current_dir().expect("get cwd");
    std::env::set_current_dir(tempdir.path()).expect("set cwd to tempdir");

    let model = Model::new_from_onnx(&onnx_bytes).expect("model loads from onnx bytes");

    // Trigger lazy tensor loading via the eval path
    let mut eval = EvalBackend::NDArray;
    let mut cache = whisper_tensor::backends::ModelLoadedTensorCache::default();
    model
        .load_tensors(&mut cache, &mut eval)
        .expect("tensors load (external data)");

    // Restore CWD
    std::env::set_current_dir(old_cwd).expect("restore cwd");
}

#[cfg(feature = "candle")]
#[test]
fn rwkv01b_model_loads_with_origin_reference() {
    // Validate that the OriginReference strategy works by keeping weights in the original .pth
    let Some(pth_path) = find_rwkv_pth() else {
        eprintln!("Skipping: no RWKV .pth found under test_models/");
        return;
    };

    // Build ONNX with external_data entries pointing to the original .pth tensors
    let onnx_bytes = whisper_tensor_import::identify_and_load(
        &pth_path,
        whisper_tensor_import::onnx_graph::WeightStorageStrategy::OriginReference,
        Some(whisper_tensor_import::ModelTypeHint::RWKV7),
    )
    .expect("import rwkv7 to onnx (OriginReference)");

    // No cwd or symlink manipulation needed: location is absolute inside ONNX now
    let model = Model::new_from_onnx(&onnx_bytes).expect("model loads from onnx bytes");

    // Trigger lazy tensor loading via the eval path (fetch tensors from the .pth via candle)
    let mut eval = EvalBackend::NDArray;
    let mut cache = whisper_tensor::backends::ModelLoadedTensorCache::default();
    model
        .load_tensors(&mut cache, &mut eval)
        .expect("tensors load (origin reference pth)");

    // Basic sanity checks
    assert!(
        !model.get_symbolic_graph().get_inputs().is_empty(),
        "model has inputs"
    );
    assert!(
        !model.get_symbolic_graph().get_outputs().is_empty(),
        "model has outputs"
    );
}
