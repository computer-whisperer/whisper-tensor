use std::path::PathBuf;
use std::time::Instant;
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::interfaces::AnyInterface;
use whisper_tensor::loader::{ConfigValue, ConfigValues, Loader};
use whisper_tensor_import::loaders::SDXLLoader;

const CHECKPOINT: &str = "/mnt/secondary/neural_networks/sd_xl_base_1.0.safetensors";

fn main() {
    tracing_subscriber::fmt::init();
    let total_start = Instant::now();

    // Load via SDXL loader
    println!("=== Loading SDXL via SDXLLoader ===");
    let start = Instant::now();
    let config = ConfigValues::from([(
        "path".to_string(),
        ConfigValue::FilePath(PathBuf::from(CHECKPOINT)),
    )]);
    let output = SDXLLoader.load(config).expect("SDXLLoader failed");
    println!("  Loaded in {:.2?}", start.elapsed());
    println!(
        "  Models: {} ({} total), Interfaces: {}",
        output
            .models
            .iter()
            .map(|m| m.name.as_str())
            .collect::<Vec<_>>()
            .join(", "),
        output.models.len(),
        output.interfaces.len(),
    );

    let models: Vec<&_> = output.models.iter().map(|m| m.model.as_ref()).collect();

    let interface = output
        .interfaces
        .iter()
        .find_map(|i| match &i.interface {
            AnyInterface::ImageGenerationInterface(x) => Some(x),
            _ => None,
        })
        .expect("No ImageGeneration interface");

    let prompt = "a photo of a cat";
    let negative_prompt = "";

    // Tiny latent for fast test
    let latent_h = 8;
    let latent_w = 8;
    let steps = 2;
    let guidance_scale = 7.5f32;
    let seed = 42u64;

    let channels = interface.latent_channels;
    let latent_n = channels * latent_h * latent_w;
    let initial_noise = generate_gaussian_noise(latent_n, seed);

    println!(
        "\n=== Running interface ({steps} steps, {}x{}) ===",
        latent_w * 8,
        latent_h * 8,
    );
    let mut backend = EvalBackend::NDArray;
    let start = Instant::now();

    let image_tensor = interface
        .run(
            &models,
            prompt.to_string(),
            Some(negative_prompt.to_string()),
            initial_noise,
            vec![1, channels, latent_h, latent_w],
            steps,
            guidance_scale,
            &mut backend,
        )
        .expect("Interface run failed")
        .tensor;

    println!(
        "  Output: dtype={:?}, shape={:?}, took {:.2?}",
        image_tensor.dtype(),
        image_tensor.shape(),
        start.elapsed(),
    );

    let f32_tensor = image_tensor
        .cast(whisper_tensor::dtype::DType::F32, &mut backend)
        .expect("cast failed");
    let ndarray = f32_tensor.to_ndarray().expect("to_ndarray failed");
    let flat: Vec<f32> = ndarray.flatten().try_into().expect("flatten failed");
    let nan_count = flat.iter().filter(|v| v.is_nan()).count();
    let min_val = flat.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("  Image: min={min_val:.4}, max={max_val:.4}, nan={nan_count}");
    println!("\n=== Complete in {:.2?} ===", total_start.elapsed());
}

fn generate_gaussian_noise(n: usize, seed: u64) -> Vec<f32> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut vals = Vec::with_capacity(n);
    while vals.len() + 1 < n {
        let u1: f32 = rand::RngExt::random_range(&mut rng, f32::EPSILON..1.0);
        let u2: f32 = rand::RngExt::random_range(&mut rng, 0.0f32..std::f32::consts::TAU);
        let r = (-2.0f32 * u1.ln()).sqrt();
        vals.push(r * u2.cos());
        vals.push(r * u2.sin());
    }
    if vals.len() < n {
        let u1: f32 = rand::RngExt::random_range(&mut rng, f32::EPSILON..1.0);
        let u2: f32 = rand::RngExt::random_range(&mut rng, 0.0f32..std::f32::consts::TAU);
        vals.push((-2.0f32 * u1.ln()).sqrt() * u2.cos());
    }
    vals
}
