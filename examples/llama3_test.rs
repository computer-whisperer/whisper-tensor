use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use whisper_tensor::backends::ModelLoadedTensorCache;
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::loader::{ConfigValue, ConfigValues, Loader};
use whisper_tensor::super_graph::cache::SuperGraphCache;
use whisper_tensor_import::loaders::TransformersLoader;

fn main() {
    tracing_subscriber::fmt::init();

    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/mnt/secondary/neural_networks/llms/Llama-3.1-8B-Instruct".into());

    let config = ConfigValues::from([(
        "path".to_string(),
        ConfigValue::FilePath(PathBuf::from(&model_dir)),
    )]);

    eprintln!("Loading model from {model_dir}...");
    let output = TransformersLoader.load(config).unwrap();
    let model = &output.models[0].model;
    let interface = output
        .interfaces
        .iter()
        .find_map(|i| {
            if let whisper_tensor::interfaces::AnyInterface::TextInferenceTokensInLogitOutInterface(
                x,
            ) = &i.interface
            {
                Some(x)
            } else {
                None
            }
        })
        .expect("No text inference interface found");

    let mut backend = EvalBackend::NDArray;
    let mut tokenizer_cache = HashMap::new();
    let mut super_graph_caches = SuperGraphCache::new();
    let mut model_cache = ModelLoadedTensorCache::new();

    let prompt = "The capital of France is".to_string();
    print!("{prompt}");
    std::io::stdout().flush().unwrap();

    let mut context = prompt;
    for _ in 0..10 {
        let token = interface
            .run_string_in_string_out(
                model,
                None,
                context.clone(),
                &mut tokenizer_cache,
                Some(&mut model_cache),
                Some(&mut super_graph_caches),
                &mut backend,
            )
            .unwrap();
        print!("{token}");
        std::io::stdout().flush().unwrap();
        context.push_str(&token);
    }
    println!();
}
