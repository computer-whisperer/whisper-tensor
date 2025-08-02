use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::model::{Model};
use whisper_tensor::super_graph::cache::SuperGraphCache;
use whisper_tensor_import::{identify_and_load, ModelTypeHint};
use whisper_tensor_import::onnx_graph::{WeightStorageStrategy};

fn main() {
    tracing_subscriber::fmt::init();
    let input_path = Path::new("/mnt/secondary/rwkv-7-world/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth");
    //let onnx_out = Path::new("out.onnx");
    let _bin_out = Path::new("out.bin");
    let onnx_data = identify_and_load(input_path, WeightStorageStrategy::EmbeddedData, Some(ModelTypeHint::RWKV7)).unwrap();

    let model = Model::new_from_onnx(&onnx_data).unwrap();


    let mut super_graph_caches = SuperGraphCache::new();
    let prompt = "Mary had a little lamb".to_string();
    let mut tokenizer_cache = HashMap::new();
    print!("{:}", prompt);
    std::io::stdout().flush().unwrap();
    let mut context = prompt.clone();
    for _ in 0..50 {
        let res = model.text_inference_tokens_in_logits_out_interface.as_ref().unwrap().run_string_in_string_out(&model, context.clone(), &mut tokenizer_cache, Some(&mut super_graph_caches), &mut EvalBackend::NDArray).unwrap();
        print!("{:}", res);
        std::io::stdout().flush().unwrap();
        context.push_str(&res);
    }
}