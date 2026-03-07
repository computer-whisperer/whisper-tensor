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

    let input_path = Path::new(SD_BASE).join("vae_encoder").join("model.onnx");
    println!("Loading vae_encoder from {}", input_path.display());

    let onnx_data = identify_and_load(&input_path, WeightStorageStrategy::EmbeddedData, None)
        .expect("Failed to import model");

    let mut rng = rand::rng();
    let model = Model::new_from_onnx(&onnx_data, &mut rng, input_path.parent())
        .expect("Failed to load model");

    let input_info = model
        .get_input_tensor_info()
        .expect("Failed to get input info");
    println!("Model inputs:");
    for (name, (dtype, shape)) in &input_info {
        println!("  {name}: dtype={dtype:?}, shape={shape:?}");
    }

    // vae_encoder expects: sample [batch=1, channels=3, height=64, width=64] as f16
    // Using small 64x64 image for fast testing — real SD uses 512x512
    let h = 64;
    let w = 64;
    let n = 1 * 3 * h * w;
    // Simulate a simple image with values in [-1, 1] range (SD normalization)
    let image_data: Vec<half::f16> = (0..n)
        .map(|i| half::f16::from_f32((i as f32 / n as f32) * 2.0 - 1.0))
        .collect();

    let input_tensor =
        NumericTensor::<DynRank>::from_vec_shape(image_data, vec![1, 3, h, w]).unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("sample".to_string(), input_tensor);

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
