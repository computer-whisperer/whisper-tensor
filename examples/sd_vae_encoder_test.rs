use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;
use whisper_tensor::DynRank;
use whisper_tensor::numeric_dtype::NumericDType;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::pool::SystemPool;
use whisper_tensor_import::identify_and_load;
use whisper_tensor_import::onnx_graph::WeightStorageStrategy;

const SD_BASE: &str = "/mnt/secondary/neural_networks/stable-diffusion-1.5-onnx-fp16";
const REF_DIR: &str = "/tmp/sd_reference";

fn load_npy_as_f16_tensor<'a>(
    name: &str,
    pool: &'a SystemPool,
) -> NumericTensor<'a, DynRank, SystemPool> {
    let path = format!("{REF_DIR}/{name}");
    let f32_tensor =
        whisper_tensor::npy::read_npy_file(Path::new(&path), pool).expect("read npy");
    NumericTensor::from_fn(f32_tensor.shape().clone(), NumericDType::F16, pool, |i| {
        f32_tensor.read_element(i).cast_to(NumericDType::F16)
    })
    .unwrap()
}

fn main() {
    tracing_subscriber::fmt::init();
    let pool = SystemPool;

    let input_path = Path::new(SD_BASE).join("vae_encoder").join("model.onnx");
    println!("Loading vae_encoder from {}", input_path.display());

    let onnx_data = identify_and_load(&input_path, WeightStorageStrategy::EmbeddedData)
        .expect("Failed to import model");
    let mut rng = rand::rng();
    let model = whisper_tensor::model::Model::new_from_onnx(&onnx_data, &mut rng, input_path.parent())
        .expect("Failed to load model");

    println!("\n=== VAE Encoder ===");
    let image = load_npy_as_f16_tensor("vae_encoder_sample_float16.npy", &pool);
    println!("  sample: {:?} {:?}", image.dtype(), image.shape());

    let image_view = image.view();
    let mut inputs = HashMap::new();
    inputs.insert("sample".to_string(), &image_view);

    let start = Instant::now();
    let outputs = model
        .eval_pool(inputs, &pool)
        .expect("Inference failed");
    println!("  Inference took {:.2?}", start.elapsed());

    // VAE encoder uses RandomNormalLike, so outputs won't match reference exactly.
    // We verify: correct shape/dtype, and output statistics are in reasonable range.
    if let Some(out) = outputs.get("latent_sample") {
        let ref_path = format!("{REF_DIR}/vae_encoder_latent_sample_float16.npy");
        let ref_tensor =
            whisper_tensor::npy::read_npy_file(std::path::Path::new(&ref_path), &pool)
                .expect("read ref npy");

        println!("  Output: dtype={:?}, shape={:?}", out.dtype(), out.shape());
        assert_eq!(
            out.shape(), ref_tensor.shape(),
            "Shape mismatch: actual={:?} vs ref={:?}", out.shape(), ref_tensor.shape()
        );
        println!("  Shape: PASS");

        let numel = out.numel();

        // Compare statistics rather than exact values
        let actual_mean = (0..numel).map(|i| out.read_element(i).to_f32()).sum::<f32>() / numel as f32;
        let actual_std = ((0..numel)
            .map(|i| { let v = out.read_element(i).to_f32(); (v - actual_mean) * (v - actual_mean) })
            .sum::<f32>()
            / numel as f32)
            .sqrt();

        let ref_mean = (0..numel).map(|i| ref_tensor.read_element(i).to_f32()).sum::<f32>() / numel as f32;
        let ref_std = ((0..numel)
            .map(|i| { let v = ref_tensor.read_element(i).to_f32(); (v - ref_mean) * (v - ref_mean) })
            .sum::<f32>()
            / numel as f32)
            .sqrt();

        println!("  Actual  stats: mean={actual_mean:.4}, std={actual_std:.4}");
        println!("  Reference stats: mean={ref_mean:.4}, std={ref_std:.4}");

        // Check that statistics are in the same ballpark (both should be latent-space values)
        let mean_diff = (actual_mean - ref_mean).abs();
        let std_ratio = if ref_std > 0.0 {
            actual_std / ref_std
        } else {
            1.0
        };
        println!("  Mean diff: {mean_diff:.4}, std ratio: {std_ratio:.4}");

        // The encoder is deterministic up to RandomNormalLike, so statistics
        // should be similar order of magnitude
        if std_ratio > 0.1 && std_ratio < 10.0 {
            println!("  Statistics: PASS (within order of magnitude)");
        } else {
            println!("  Statistics: WARNING (std ratio outside expected range)");
        }
    }
}
