use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::loader::{ConfigValue, ConfigValues, Loader};
use whisper_tensor_import::loaders::OnnxLoader;

fn main() {
    tracing_subscriber::fmt::init();

    let config = ConfigValues::from([
        (
            "path".to_string(),
            ConfigValue::FilePath(PathBuf::from("gpt2-lm-head-10.onnx")),
        ),
        (
            "model_type".to_string(),
            ConfigValue::String("GPT2".to_string()),
        ),
    ]);

    let output = OnnxLoader.load(config).unwrap();
    let model = &output.models[0].model;

    let prompt = "Mary had a little lamb".to_string();
    let mut tokenizer_cache = HashMap::new();
    print!("{prompt:}");
    std::io::stdout().flush().unwrap();
    let mut context = prompt.clone();

    for _ in 0..10 {
        let res = model
            .text_inference_tokens_in_logits_out_interface
            .as_ref()
            .unwrap()
            .run_string_in_string_out(
                model,
                None,
                context.clone(),
                &mut tokenizer_cache,
                None,
                None,
                &mut EvalBackend::NDArray,
            )
            .unwrap();
        print!("{res:}");
        std::io::stdout().flush().unwrap();
        context.push_str(&res);
    }
}
