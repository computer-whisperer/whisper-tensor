use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use whisper_tensor::DynRank;
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::dtype::DType;
use whisper_tensor::model::Model;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor_import::onnx_graph::WeightStorageStrategy;
use whisper_tensor_import::onnx_graph::weights::SafetensorsWeightManager;

const DIT_PATH: &str = "/ceph/public/neural_models/comfyui/unet/flux1-schnell.safetensors";
const VAE_PATH: &str = "/ceph/public/neural_models/comfyui/vae/ae.safetensors";

fn test_dit() {
    let total_start = Instant::now();
    let dit_path = Path::new(DIT_PATH);

    println!("=== Building Flux DiT ===");
    let start = Instant::now();

    let file = std::fs::File::open(dit_path).expect("open DiT weights");
    let mmap = unsafe { memmap2::Mmap::map(&file) }.expect("mmap");
    let wm = SafetensorsWeightManager::new(vec![Arc::new(mmap)]).expect("weight manager");

    let config = whisper_tensor_import::flux::FluxConfig::schnell_1024(256);
    let onnx_data = whisper_tensor_import::flux::load_flux_dit_with_origin(
        wm,
        config,
        WeightStorageStrategy::OriginReference,
        Some(dit_path),
    )
    .expect("Flux DiT build failed");

    println!(
        "  Built in {:.2?} ({:.1}MB ONNX)",
        start.elapsed(),
        onnx_data.len() as f64 / 1e6
    );

    // Load model
    let start = Instant::now();
    let mut rng = rand::rng();
    let model = Model::new_from_onnx(&onnx_data, &mut rng, Some(dit_path.parent().unwrap()))
        .expect("model load failed");
    println!("  Model loaded in {:.2?}", start.elapsed());

    let mut backend = EvalBackend::NDArray;

    // Create dummy inputs
    let latent = NumericTensor::<DynRank>::from_vec_shape(
        vec![0.0f32; 1 * 16 * 128 * 128],
        vec![1, 16, 128, 128],
    )
    .unwrap()
    .cast(DType::BF16, &mut backend)
    .unwrap();

    let timestep = NumericTensor::<DynRank>::from_vec_shape(vec![0.5f32], vec![1, 1]).unwrap();

    let clip_pooled = NumericTensor::<DynRank>::from_vec_shape(
        vec![0.0f32; 768],
        vec![1, 768],
    )
    .unwrap()
    .cast(DType::BF16, &mut backend)
    .unwrap();

    let t5_hidden = NumericTensor::<DynRank>::from_vec_shape(
        vec![0.0f32; 256 * 4096],
        vec![1, 256, 4096],
    )
    .unwrap()
    .cast(DType::BF16, &mut backend)
    .unwrap();

    println!("\n=== Running Flux DiT ===");
    let start = Instant::now();
    let output = model
        .eval(
            std::collections::HashMap::from([
                ("latent_sample".to_string(), latent),
                ("timestep".to_string(), timestep),
                ("clip_pooled".to_string(), clip_pooled),
                ("t5_hidden_states".to_string(), t5_hidden),
            ]),
            &mut (),
            None,
            &mut backend,
        )
        .expect("Flux DiT eval failed");

    let out = output.get("out_sample").unwrap();
    println!(
        "  Output: dtype={:?}, shape={:?}, took {:.2?}",
        out.dtype(),
        out.shape(),
        start.elapsed(),
    );

    // Check values
    let f32_tensor = out.cast(DType::F32, &mut backend).expect("cast failed");
    let ndarray = f32_tensor.to_ndarray().expect("to_ndarray failed");
    let flat: Vec<f32> = ndarray.flatten().try_into().expect("flatten failed");
    let nan_count = flat.iter().filter(|v| v.is_nan()).count();
    let inf_count = flat.iter().filter(|v| v.is_infinite()).count();
    let min_val = flat.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "  out_sample: min={min_val:.4}, max={max_val:.4}, nan={nan_count}, inf={inf_count}"
    );

    println!("\n=== DiT complete in {:.2?} ===", total_start.elapsed());
}

fn test_vae() {
    let vae_path = Path::new(VAE_PATH);

    println!("\n=== Building Flux VAE decoder ===");
    let start = Instant::now();

    let file = std::fs::File::open(vae_path).expect("open VAE weights");
    let mmap = unsafe { memmap2::Mmap::map(&file) }.expect("mmap");
    let wm = SafetensorsWeightManager::new(vec![Arc::new(mmap)]).expect("weight manager");

    let onnx_data = whisper_tensor_import::sd_common::build_flux_vae_decoder(
        wm,
        whisper_tensor_import::onnx_graph::tensor::DType::F32,
        WeightStorageStrategy::OriginReference,
        vae_path,
    )
    .expect("VAE build failed");

    println!(
        "  Built in {:.2?} ({:.1}MB ONNX)",
        start.elapsed(),
        onnx_data.len() as f64 / 1e6
    );

    // Load model
    let start = Instant::now();
    let mut rng = rand::rng();
    let model = Model::new_from_onnx(&onnx_data, &mut rng, Some(vae_path.parent().unwrap()))
        .expect("model load failed");
    println!("  Model loaded in {:.2?}", start.elapsed());

    let mut backend = EvalBackend::NDArray;

    // Small latent input: [1, 16, 8, 8] → [1, 3, 64, 64]
    let latent = NumericTensor::<DynRank>::from_vec_shape(
        vec![0.0f32; 1 * 16 * 8 * 8],
        vec![1, 16, 8, 8],
    )
    .unwrap();

    println!("\n=== Running Flux VAE decoder ===");
    let start = Instant::now();
    let output = model
        .eval(
            std::collections::HashMap::from([("latent_sample".to_string(), latent)]),
            &mut (),
            None,
            &mut backend,
        )
        .expect("VAE eval failed");

    let out = output.get("sample").unwrap();
    println!(
        "  Output: dtype={:?}, shape={:?}, took {:.2?}",
        out.dtype(),
        out.shape(),
        start.elapsed(),
    );

    let f32_tensor = out.cast(DType::F32, &mut backend).expect("cast failed");
    let ndarray = f32_tensor.to_ndarray().expect("to_ndarray failed");
    let flat: Vec<f32> = ndarray.flatten().try_into().expect("flatten failed");
    let nan_count = flat.iter().filter(|v| v.is_nan()).count();
    let inf_count = flat.iter().filter(|v| v.is_infinite()).count();
    println!(
        "  sample: nan={nan_count}, inf={inf_count}, len={}",
        flat.len()
    );
}

fn main() {
    tracing_subscriber::fmt::init();

    test_vae();
    test_dit();
}
