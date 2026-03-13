use std::path::{Path, PathBuf};

fn find_gemma_dir() -> Option<PathBuf> {
    let candidates = [
        "/ceph/public/neural_models/llms/gemma-2b-it",
        "/ceph/public/neural_models/llms/codegemma-2b",
    ];
    for c in candidates {
        let p = Path::new(c);
        if p.join("config.json").exists() {
            return Some(p.to_path_buf());
        }
    }
    None
}

/// Test that a Gemma ONNX graph builds successfully from safetensors weights.
#[test]
fn gemma_graph_builds() {
    let Some(model_dir) = find_gemma_dir() else {
        eprintln!("Skipping: no Gemma model directory found");
        return;
    };

    let config_str = std::fs::read_to_string(model_dir.join("config.json")).unwrap();
    let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let num_layers = config["num_hidden_layers"].as_u64().unwrap() as usize;

    let onnx_bytes = whisper_tensor_import::identify_and_load(
        &model_dir,
        whisper_tensor_import::onnx_graph::WeightStorageStrategy::None,
    )
    .expect("graph should build successfully");

    eprintln!("ONNX proto size: {} bytes", onnx_bytes.len());

    let mut rng = rand::rng();
    let model = whisper_tensor::model::Model::new_from_onnx(&onnx_bytes, &mut rng, None)
        .expect("model should parse from ONNX");

    let graph = model.get_symbolic_graph();
    let names = graph.get_tensors_by_name();

    assert!(names.contains_key("input_ids"), "should have input_ids");
    for i in 0..num_layers {
        assert!(
            names.contains_key(&format!("kv_cache_input_k_{i}")),
            "missing kv_cache_input_k_{i}"
        );
        assert!(
            names.contains_key(&format!("kv_cache_input_v_{i}")),
            "missing kv_cache_input_v_{i}"
        );
        assert!(
            names.contains_key(&format!("kv_cache_output_k_{i}")),
            "missing kv_cache_output_k_{i}"
        );
        assert!(
            names.contains_key(&format!("kv_cache_output_v_{i}")),
            "missing kv_cache_output_v_{i}"
        );
    }
    assert!(names.contains_key("logits"), "should have logits output");
    assert!(
        names.contains_key("cos_cache"),
        "should have cos_cache constant"
    );
    assert!(
        names.contains_key("sin_cache"),
        "should have sin_cache constant"
    );
}

/// Test that the TransformersLoader produces a working interface for Gemma.
#[test]
fn gemma_loads_via_transformers_loader() {
    let Some(model_dir) = find_gemma_dir() else {
        eprintln!("Skipping: no Gemma model directory found");
        return;
    };

    use whisper_tensor::loader::{ConfigValue, ConfigValues, Loader};
    use whisper_tensor_import::loaders::TransformersLoader;

    let config = ConfigValues::from([("path".to_string(), ConfigValue::FilePath(model_dir))]);

    let output = TransformersLoader
        .load(config)
        .expect("TransformersLoader should succeed");

    assert_eq!(output.models.len(), 1, "should produce one model");
    assert!(
        !output.interfaces.is_empty(),
        "should produce at least one interface"
    );

    let interface = &output.interfaces[0];
    assert_eq!(
        interface.interface.name(),
        "TextInferenceTokensInLogitsOut",
        "should be a text inference interface"
    );
}
