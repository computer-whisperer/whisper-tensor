use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::loader::{ConfigValue, ConfigValues, Loader};
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

    eprintln!("Model loaded. Running inference...");

    let prompt = "The capital of France is".to_string();
    let mut tokenizer_cache = HashMap::new();
    let mut backend = EvalBackend::NDArray;
    let mut context = prompt.clone();

    print!("{prompt}");
    std::io::stdout().flush().unwrap();

    for _ in 0..20 {
        let token = interface
            .run_string_in_string_out(
                model,
                None,
                context.clone(),
                &mut tokenizer_cache,
                None,
                None,
                &mut backend,
            )
            .unwrap();
        print!("{token}");
        std::io::stdout().flush().unwrap();
        context.push_str(&token);
    }
    println!();
}
