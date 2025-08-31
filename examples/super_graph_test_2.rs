use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::model::Model;
use whisper_tensor_import::onnx_graph::WeightStorageStrategy;
use whisper_tensor_import::{ModelTypeHint, identify_and_load};

fn main() {
    tracing_subscriber::fmt::init();

    let input_path = Path::new("gpt2-lm-head-10.onnx");
    let onnx_data = identify_and_load(
        input_path,
        WeightStorageStrategy::EmbeddedData,
        Some(ModelTypeHint::GPT2),
    )
    .unwrap();

    let model = Model::new_from_onnx(&onnx_data).unwrap();

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
                &model,
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
