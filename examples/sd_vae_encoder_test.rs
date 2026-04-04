use ndarray::ArrayD;
use ndarray_npy::ReadNpyExt;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::time::Instant;
use whisper_tensor::DynRank;
use whisper_tensor::numeric_dtype::NumericDType;
use whisper_tensor::numeric_scalar::NumericScalar;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::pool::SystemPool;
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

fn load_npy_as_f16_tensor<'a>(
    name: &str,
    pool: &'a SystemPool,
) -> NumericTensor<'a, DynRank, SystemPool> {
    let (values, shape) = load_npy_f32(name);
    let shape_u64: Vec<u64> = shape.iter().map(|&s| s as u64).collect();
    NumericTensor::from_fn(shape_u64, NumericDType::F16, pool, |i| {
        NumericScalar::from_f32(values[i]).cast_to(NumericDType::F16)
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
        let ref_path = "vae_encoder_latent_sample_float16.npy";
        let (ref_values, ref_shape) = load_npy_f32(ref_path);

        let actual_shape: Vec<usize> = out.shape().iter().map(|&s| s as usize).collect();
        println!("  Output: dtype={:?}, shape={:?}", out.dtype(), out.shape());
        assert_eq!(
            actual_shape, ref_shape,
            "Shape mismatch: actual={actual_shape:?} vs ref={ref_shape:?}"
        );
        println!("  Shape: PASS");

        let actual_f32 = cast_tensor(out, NumericDType::F32, &pool);
        let actual_flat: Vec<f32> = (0..actual_f32.numel())
            .map(|i| actual_f32.read_element(i).to_f32())
            .collect();

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

fn cast_tensor<'a>(
    tensor: &NumericTensor<'_, DynRank, SystemPool>,
    target_dtype: NumericDType,
    pool: &'a SystemPool,
) -> NumericTensor<'a, DynRank, SystemPool> {
    let shape: Vec<u64> = tensor.shape().clone();
    NumericTensor::from_fn(shape, target_dtype, pool, |i| {
        tensor.read_element(i).cast_to(target_dtype)
    })
    .unwrap()
}
