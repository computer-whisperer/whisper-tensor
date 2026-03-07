use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;
use whisper_tensor::DynRank;
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::model::Model;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor_import::identify_and_load;
use whisper_tensor_import::onnx_graph::WeightStorageStrategy;

const SD_BASE: &str = "/mnt/secondary/neural_networks/stable-diffusion-1.5-onnx-fp16";

fn main() {
    tracing_subscriber::fmt::init();

    let input_path = Path::new(SD_BASE).join("text_encoder").join("model.onnx");
    println!("Loading text_encoder from {}", input_path.display());

    let onnx_data = identify_and_load(&input_path, WeightStorageStrategy::EmbeddedData, None)
        .expect("Failed to import model");

    let mut rng = rand::rng();
    let model = Model::new_from_onnx(&onnx_data, &mut rng, input_path.parent())
        .expect("Failed to load model");

    // Print model input/output info
    let input_info = model
        .get_input_tensor_info()
        .expect("Failed to get input info");
    println!("Model inputs:");
    for (name, (dtype, shape)) in &input_info {
        println!("  {name}: dtype={dtype:?}, shape={shape:?}");
    }

    // text_encoder expects: input_ids [batch=1, sequence=77] as i32 (ONNX dtype 6 = INT32)
    // CLIP tokenizer uses 49406 as BOS, 49407 as EOS, 0 as padding
    let seq_len = 77;
    let mut input_ids = vec![0i32; seq_len];
    input_ids[0] = 49406; // BOS
    // "a photo of a cat" - dummy token IDs (not real CLIP tokens, but structurally valid)
    input_ids[1] = 320; // "a"
    input_ids[2] = 1125; // "photo"
    input_ids[3] = 539; // "of"
    input_ids[4] = 320; // "a"
    input_ids[5] = 2368; // "cat"
    input_ids[6] = 49407; // EOS

    let input_tensor =
        NumericTensor::<DynRank>::from_vec_shape(input_ids, vec![1, seq_len]).unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("input_ids".to_string(), input_tensor);

    let mut backend = EvalBackend::NDArray;
    println!("\nRunning inference...");
    let start = Instant::now();
    let outputs = model
        .eval(inputs, &mut (), None, &mut backend)
        .expect("Inference failed");
    let elapsed = start.elapsed();

    println!("Inference completed in {elapsed:.2?}");
    println!("Outputs:");
    for (name, tensor) in &outputs {
        let shape = tensor.shape();
        let dtype = tensor.dtype();
        println!("  {name}: dtype={dtype:?}, shape={shape:?}");
    }
}
