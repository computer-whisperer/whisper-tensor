use std::path::{Path, PathBuf};

fn find_llama3_dir() -> Option<PathBuf> {
    let candidates = [
        "/mnt/secondary/neural_networks/llms/Llama-3.1-8B-Instruct",
        "/ceph/public/neural_models/llms/Llama-3.1-8B-Instruct",
    ];
    for c in candidates {
        let p = Path::new(c);
        if p.join("config.json").exists() {
            return Some(p.to_path_buf());
        }
    }
    None
}

/// Test that the Llama 3 ONNX graph builds successfully from safetensors weights.
/// Uses WeightStorageStrategy::None to avoid serializing 16GB of weights.
#[test]
fn llama3_graph_builds() {
    let Some(model_dir) = find_llama3_dir() else {
        eprintln!("Skipping: no Llama 3 model directory found");
        return;
    };

    // Build the ONNX graph without embedding weights (just validates the graph structure)
    let onnx_bytes = whisper_tensor_import::identify_and_load(
        &model_dir,
        whisper_tensor_import::onnx_graph::WeightStorageStrategy::None,
    )
    .expect("graph should build successfully");

    eprintln!("ONNX proto size: {} bytes", onnx_bytes.len());

    // Verify the model can be parsed back
    let mut rng = rand::rng();
    let model = whisper_tensor::model::Model::new_from_onnx(&onnx_bytes, &mut rng, None)
        .expect("model should parse from ONNX");

    let graph = model.get_symbolic_graph();
    let names = graph.get_tensors_by_name();

    // Verify expected inputs exist
    assert!(names.contains_key("input_ids"), "should have input_ids");

    // Verify KV cache pairs exist for all 32 layers
    for i in 0..32 {
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

    // Verify logits output exists
    assert!(names.contains_key("logits"), "should have logits output");

    // Verify RoPE caches are present as initialized tensors (not dynamic inputs)
    assert!(
        names.contains_key("cos_cache"),
        "should have cos_cache constant"
    );
    assert!(
        names.contains_key("sin_cache"),
        "should have sin_cache constant"
    );

    eprintln!(
        "Graph has {} named tensors, {} inputs, {} outputs",
        names.len(),
        graph.get_inputs().len(),
        graph.get_outputs().len(),
    );
}

/// Test that the TransformersLoader produces a working interface.
#[test]
fn llama3_loads_via_transformers_loader() {
    let Some(model_dir) = find_llama3_dir() else {
        eprintln!("Skipping: no Llama 3 model directory found");
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
    eprintln!(
        "Interface: {} ({})",
        interface.name,
        interface.interface.name()
    );
}
