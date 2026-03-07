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

    let input_path = Path::new(SD_BASE).join("unet").join("model.onnx");
    println!("Loading unet from {}", input_path.display());

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

    // unet expects three inputs, all f16:
    //   sample: [batch=1, channels=4, height=8, width=8] — latent noise
    //   timestep: [batch=1] — diffusion timestep
    //   encoder_hidden_states: [batch=1, sequence=77, 768] — text encoder output
    // Using small 8x8 latent for fast testing (real SD uses 64x64)
    let h = 8;
    let w = 8;

    let sample_n = 1 * 4 * h * w;
    let sample_data: Vec<half::f16> = (0..sample_n)
        .map(|i| half::f16::from_f32((i as f32 / sample_n as f32) * 0.1))
        .collect();
    let sample = NumericTensor::<DynRank>::from_vec_shape(sample_data, vec![1, 4, h, w]).unwrap();

    // timestep = 500 (midpoint of typical 1000-step schedule)
    let timestep_data = vec![half::f16::from_f32(500.0)];
    let timestep = NumericTensor::<DynRank>::from_vec_shape(timestep_data, vec![1]).unwrap();

    // encoder_hidden_states: [1, 77, 768] — dummy text embeddings
    let seq_len = 77;
    let hidden_dim = 768;
    let enc_n = 1 * seq_len * hidden_dim;
    let enc_data: Vec<half::f16> = (0..enc_n)
        .map(|i| half::f16::from_f32((i as f32 / enc_n as f32) * 0.01))
        .collect();
    let encoder_hidden_states =
        NumericTensor::<DynRank>::from_vec_shape(enc_data, vec![1, seq_len, hidden_dim]).unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("sample".to_string(), sample);
    inputs.insert("timestep".to_string(), timestep);
    inputs.insert("encoder_hidden_states".to_string(), encoder_hidden_states);

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
