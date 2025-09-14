use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use whisper_tensor::backends::ModelLoadedTensorCache;
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::compiler;
use whisper_tensor::model::Model;
use whisper_tensor::super_graph::cache::SuperGraphCache;
use whisper_tensor_import::onnx_graph::WeightStorageStrategy;
use whisper_tensor_import::{ModelTypeHint, identify_and_load};

fn main() {
    tracing_subscriber::fmt::init();
    let input_path =
        Path::new("/mnt/secondary/rwkv-7-world/RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth");
    //let input_path = Path::new("/mnt/secondary/neural_networks/llms/Llama-3.1-8B-Instruct");
    //let onnx_out = Path::new("out.onnx");
    //let _bin_out = Path::new("out.bin");
    let onnx_data = identify_and_load(
        input_path,
        WeightStorageStrategy::OriginReference,
        Some(ModelTypeHint::RWKV7),
    )
    .unwrap();

    let mut eval_backend = EvalBackend::NDArray;

    let model = Arc::new(Model::new_from_onnx(&onnx_data).unwrap());
    let compiled_model = compiler::build_program(compiler::CompilationSubject::Model {
        model: model.clone(),
    });

    let mut super_graph_caches = SuperGraphCache::new();
    let prompt = "The first 10 numbers in the fibonacci sequence are:".to_string();
    let mut tokenizer_cache = HashMap::new();
    print!("{prompt:}");
    std::io::stdout().flush().unwrap();
    let mut context = prompt.clone();
    let mut model_cache = ModelLoadedTensorCache::new();
    for _ in 0..10 {
        let res = model
            .text_inference_tokens_in_logits_out_interface
            .as_ref()
            .unwrap()
            .run_string_in_string_out(
                &model,
                Some(&compiled_model),
                context.clone(),
                &mut tokenizer_cache,
                Some(&mut model_cache),
                Some(&mut super_graph_caches),
                &mut eval_backend,
            )
            .unwrap();
        print!("{res:}");
        std::io::stdout().flush().unwrap();
        context.push_str(&res);
    }
}
