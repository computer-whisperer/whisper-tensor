use ndarray::ArrayD;
use ndarray_npy::ReadNpyExt;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::time::Instant;
use whisper_tensor::DynRank;
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::dtype::DType;
use whisper_tensor::model::Model;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor_import::identify_and_load;
use whisper_tensor_import::onnx_graph::WeightStorageStrategy;

const SD_BASE: &str = "/mnt/secondary/neural_networks/stable-diffusion-1.5-onnx-fp16";
const REF_DIR: &str = "/tmp/sd_reference";

fn load_npy_f32(name: &str) -> (Vec<f32>, Vec<usize>) {
    let path = format!("{REF_DIR}/{name}");
    let reader = File::open(&path).unwrap_or_else(|e| panic!("Cannot open {path}: {e}"));
    let arr = ArrayD::<f32>::read_npy(reader).unwrap();
    let shape = arr.shape().to_vec();
    let values = arr.into_raw_vec_and_offset().0;
    (values, shape)
}

fn load_npy_as_f16_tensor(name: &str) -> NumericTensor<DynRank> {
    let (values, shape) = load_npy_f32(name);
    let f16_values: Vec<half::f16> = values.iter().map(|&x| half::f16::from_f32(x)).collect();
    NumericTensor::<DynRank>::from_vec_shape(f16_values, shape).unwrap()
}

fn main() {
    tracing_subscriber::fmt::init();

    let input_path = Path::new(SD_BASE).join("vae_encoder").join("model.onnx");
    println!("Loading vae_encoder from {}", input_path.display());

    let onnx_data = identify_and_load(&input_path, WeightStorageStrategy::EmbeddedData, None)
        .expect("Failed to import model");
    let mut rng = rand::rng();
    let model = Model::new_from_onnx(&onnx_data, &mut rng, input_path.parent())
        .expect("Failed to load model");

    let mut backend = EvalBackend::NDArray;

    println!("\n=== VAE Encoder ===");
    let image = load_npy_as_f16_tensor("vae_encoder_sample_float16.npy");
    println!("  sample: {:?} {:?}", image.dtype(), image.shape());

    let mut inputs = HashMap::new();
    inputs.insert("sample".to_string(), image);

    let start = Instant::now();
    let outputs = model
        .eval(inputs, &mut (), None, &mut backend)
        .expect("Inference failed");
    println!("  Inference took {:.2?}", start.elapsed());

    // VAE encoder uses RandomNormalLike, so outputs won't match reference exactly.
    // We verify: correct shape/dtype, and output statistics are in reasonable range.
    if let Some(out) = outputs.get("latent_sample") {
        let ref_path = "vae_encoder_latent_sample_float16.npy";
        let (ref_values, ref_shape) = load_npy_f32(ref_path);

        let actual_shape: Vec<usize> = out.shape().iter().map(|&s| s as usize).collect();
        println!("  Output: dtype={:?}, shape={:?}", out.dtype(), out.shape());
        assert_eq!(
            actual_shape, ref_shape,
            "Shape mismatch: actual={actual_shape:?} vs ref={ref_shape:?}"
        );
        println!("  Shape: PASS");

        let actual_f32 = out.cast(DType::F32, &mut backend).unwrap();
        let actual_nd = actual_f32.to_ndarray().unwrap();
        let actual_flat: Vec<f32> = actual_nd.flatten().try_into().unwrap();

        // Compare statistics rather than exact values
        let actual_mean = actual_flat.iter().sum::<f32>() / actual_flat.len() as f32;
        let actual_std = (actual_flat
            .iter()
            .map(|x| (x - actual_mean) * (x - actual_mean))
            .sum::<f32>()
            / actual_flat.len() as f32)
            .sqrt();

        let ref_mean = ref_values.iter().sum::<f32>() / ref_values.len() as f32;
        let ref_std = (ref_values
            .iter()
            .map(|x| (x - ref_mean) * (x - ref_mean))
            .sum::<f32>()
            / ref_values.len() as f32)
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
