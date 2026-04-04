use std::collections::HashMap;
use std::path::PathBuf;

use whisper_tensor::model::Model;
use whisper_tensor::numeric_dtype::NumericDType;
use whisper_tensor::numeric_tensor::{NumericTensor, NumericTensorView, TensorLayout};
use whisper_tensor::pool::{Pool, SystemPool};
use whisper_tensor::tensor_rank::DynRank;

/// Returns `true` if the path is a real model file, `false` if it is an LFS pointer.
fn is_real_model_file(path: &std::path::Path) -> bool {
    use std::fs;
    let Ok(md) = fs::metadata(path) else {
        return false;
    };
    if md.len() < 1024
        && let Ok(bytes) = fs::read(path)
        && bytes.starts_with(b"version https://git-lfs.github.com/spec/v1")
    {
        return false;
    }
    true
}

fn find_rwkv_pth() -> Option<PathBuf> {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.push("test_models");
    let rd = std::fs::read_dir(&dir).ok()?;
    for entry in rd.flatten() {
        let p = entry.path();
        if p.is_file()
            && let Some(ext) = p.extension()
            && ext == "pth"
            && p.file_name()
                .and_then(|s| s.to_str())
                .map(|s| s.to_lowercase())
                .map(|s| s.contains("rwkv") || s.contains("rwkv7") || s.contains("world"))
                .unwrap_or(false)
        {
            return Some(p);
        }
    }
    None
}

/// Load the RWKV model, returning None (with eprintln) if not available.
fn load_rwkv_model() -> Option<(Model, Vec<u8>)> {
    let pth_path = find_rwkv_pth()?;
    if !is_real_model_file(&pth_path) {
        eprintln!(
            "Skipping: {} is a Git LFS pointer, not the real model",
            pth_path.display()
        );
        return None;
    }
    let onnx_bytes = whisper_tensor_import::identify_and_load(
        &pth_path,
        whisper_tensor_import::onnx_graph::WeightStorageStrategy::EmbeddedData,
    )
    .expect("import rwkv7 to onnx");
    let mut rng = rand::rng();
    let model = Model::new_from_onnx(&onnx_bytes, &mut rng, None).expect("model loads");
    Some((model, onnx_bytes))
}

/// Build zero-filled input tensors for a model based on its declared input info.
fn build_zero_inputs(
    model: &Model,
) -> HashMap<String, NumericTensor<'static, DynRank, SystemPool>> {
    let input_infos = model.get_input_tensor_info();
    let mut inputs = HashMap::new();
    for (name, (legacy_dtype, shape_desc)) in input_infos {
        let Some(dtype) = NumericDType::from_legacy(legacy_dtype) else {
            continue;
        };
        let shape: Vec<u64> = shape_desc
            .into_iter()
            .map(|d| d.unwrap_or(1).max(1))
            .collect();
        let layout = TensorLayout::<DynRank>::row_major(shape, dtype);
        let buf = SystemPool
            .allocate(layout.buffer_size_bytes())
            .expect("alloc");
        inputs.insert(name, NumericTensor::from_parts(buf, layout));
    }
    inputs
}

#[test]
fn rwkv01b_model_loads() {
    let Some(pth_path) = find_rwkv_pth() else {
        eprintln!("Skipping: no RWKV .pth found under test_models/");
        return;
    };
    if !is_real_model_file(&pth_path) {
        eprintln!(
            "Skipping: {} is a Git LFS pointer, not the real model",
            pth_path.display()
        );
        return;
    }
    let onnx_bytes = whisper_tensor_import::identify_and_load(
        &pth_path,
        whisper_tensor_import::onnx_graph::WeightStorageStrategy::EmbeddedData,
    )
    .expect("import rwkv7 to onnx");
    let mut rng = rand::rng();
    let model = Model::new_from_onnx(&onnx_bytes, &mut rng, None).expect("model loads");

    assert!(
        !model.get_symbolic_graph().get_inputs().is_empty(),
        "model has inputs"
    );
    assert!(
        !model.get_symbolic_graph().get_outputs().is_empty(),
        "model has outputs"
    );
}

#[test]
fn rwkv01b_single_step_runs_shape_sanity() {
    let Some((model, _)) = load_rwkv_model() else {
        eprintln!("Skipping: no RWKV .pth found or LFS pointer");
        return;
    };

    let owned_inputs = build_zero_inputs(&model);
    let views: HashMap<String, NumericTensorView<'_, DynRank>> = owned_inputs
        .iter()
        .map(|(n, t)| (n.clone(), t.view()))
        .collect();
    let view_refs: HashMap<String, &NumericTensorView<'_, DynRank>> =
        views.iter().map(|(n, v)| (n.clone(), v)).collect();

    let outputs = model
        .eval_pool(view_refs, &SystemPool)
        .expect("eval_pool ok");
    assert!(!outputs.is_empty(), "has outputs");
}

#[test]
fn rwkv01b_model_loads_with_binfile() {
    let Some(pth_path) = find_rwkv_pth() else {
        eprintln!("Skipping: no RWKV .pth found under test_models/");
        return;
    };
    if !is_real_model_file(&pth_path) {
        eprintln!(
            "Skipping: {} is a Git LFS pointer, not the real model",
            pth_path.display()
        );
        return;
    }

    let tempdir = tempfile::tempdir().expect("create tempdir");
    let bin_path = tempdir.path().join("weights.bin");

    let onnx_bytes = whisper_tensor_import::identify_and_load(
        &pth_path,
        whisper_tensor_import::onnx_graph::WeightStorageStrategy::BinFile(bin_path.clone()),
    )
    .expect("import rwkv7 to onnx (BinFile)");

    let meta = std::fs::metadata(&bin_path).expect("bin file exists");
    assert!(meta.len() > 0, "bin file should be non-empty");

    // external_data 'location' is just the filename — set CWD so loader finds it.
    let old_cwd = std::env::current_dir().expect("get cwd");
    std::env::set_current_dir(tempdir.path()).expect("set cwd to tempdir");

    let mut rng = rand::rng();
    let model =
        Model::new_from_onnx(&onnx_bytes, &mut rng, None).expect("model loads from onnx bytes");

    // Verify we can run eval (which loads tensors from the store internally).
    let owned_inputs = build_zero_inputs(&model);
    let views: HashMap<String, NumericTensorView<'_, DynRank>> = owned_inputs
        .iter()
        .map(|(n, t)| (n.clone(), t.view()))
        .collect();
    let view_refs: HashMap<String, &NumericTensorView<'_, DynRank>> =
        views.iter().map(|(n, v)| (n.clone(), v)).collect();

    let outputs = model
        .eval_pool(view_refs, &SystemPool)
        .expect("eval_pool ok");
    assert!(!outputs.is_empty(), "has outputs");

    std::env::set_current_dir(old_cwd).expect("restore cwd");
}

/// Full end-to-end integrity check: evaluate the RWKV 0.1B model through
/// both the MilliOpGraph interpreter and the NanoGraph scalar eval, then
/// compare every output element.
#[test]
#[ignore]
fn rwkv01b_nano_graph_integrity() {
    // TODO: migrate to pool_eval (nano_graph::eval was deleted).
    // This test needs MilliOpGraph::pool_eval for ground truth, then
    // NanoGraph pool_eval for comparison. Blocked on the same infrastructure
    // the test_set runners use, but at model scale.
    todo!("migrate rwkv01b_nano_graph_integrity to pool_eval");
}
