use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use whisper_tensor::backends::ModelLoadedTensorCache;
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::backends::vulkan_backend::{VulkanContext, VulkanImmediateExecutor};
use whisper_tensor::loader::{ConfigValue, ConfigValues, Loader};
use whisper_tensor::super_graph::cache::SuperGraphCache;
use whisper_tensor_import::loaders::Rwkv7Loader;

fn main() {
    tracing_subscriber::fmt::init();

    let config = ConfigValues::from([(
        "path".to_string(),
        ConfigValue::FilePath(PathBuf::from(
            "/mnt/secondary/rwkv-7-world/RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth",
        )),
    )]);

    let output = Rwkv7Loader.load(config).unwrap();
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

    let vulkan_context = VulkanContext::new().unwrap();
    let mut vulkan_runtime = VulkanImmediateExecutor::new(vulkan_context).unwrap();
    let mut eval_backend = EvalBackend::Vulkan(&mut vulkan_runtime);

    let mut super_graph_caches = SuperGraphCache::new();
    let prompt = "The first 10 numbers in the fibonacci sequence are:".to_string();
    let mut tokenizer_cache = HashMap::new();
    print!("{prompt:}");
    std::io::stdout().flush().unwrap();
    let mut context = prompt.clone();
    let mut model_cache = ModelLoadedTensorCache::new();
    for _ in 0..10 {
        let res = interface
            .run_string_in_string_out(
                model,
                None,
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
