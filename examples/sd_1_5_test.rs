use std::path::Path;
use std::time::Instant;
use whisper_tensor::DynRank;
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::interfaces::StableDiffusionInterface;
use whisper_tensor::model::Model;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor_import::identify_and_load;
use whisper_tensor_import::onnx_graph::WeightStorageStrategy;

const SD_BASE: &str = "/mnt/secondary/neural_networks/stable-diffusion-1.5-onnx-fp16";

fn load_model(name: &str, subpath: &str) -> Model {
    let input_path = Path::new(SD_BASE).join(subpath).join("model.onnx");
    println!("Loading {name} from {}", input_path.display());
    let start = Instant::now();
    let onnx_data = identify_and_load(&input_path, WeightStorageStrategy::EmbeddedData, None)
        .expect("Failed to import model");
    let mut rng = rand::rng();
    let model = Model::new_from_onnx(&onnx_data, &mut rng, input_path.parent())
        .expect("Failed to load model");
    println!("  Loaded in {:.2?}", start.elapsed());
    model
}

fn main() {
    tracing_subscriber::fmt::init();

    let num_inference_steps = 20;
    let latent_h = 8;
    let latent_w = 8;
    let guidance_scale: f32 = 7.5;
    let total_start = Instant::now();

    // --- Load all models ---
    let text_encoder = load_model("text_encoder", "text_encoder");
    let unet = load_model("unet", "unet");
    let vae_decoder = load_model("vae_decoder", "vae_decoder");
    println!();

    let mut backend = EvalBackend::NDArray;

    // --- Build interface ---
    println!("=== Building StableDiffusionInterface ===");
    let start = Instant::now();
    let mut rng = rand::rng();
    let sd_interface = StableDiffusionInterface::new(&mut rng);
    println!("  Built in {:.2?}", start.elapsed());
    println!();

    // --- Prepare inputs ---
    let seq_len = 77;

    // Conditional prompt: "a photo of a cat"
    let mut cond_ids = vec![0i32; seq_len];
    cond_ids[0] = 49406; // BOS
    cond_ids[1] = 320; // "a"
    cond_ids[2] = 1125; // "photo"
    cond_ids[3] = 539; // "of"
    cond_ids[4] = 320; // "a"
    cond_ids[5] = 2368; // "cat"
    cond_ids[6] = 49407; // EOS

    // Unconditional prompt: empty (just BOS + EOS + padding)
    let mut uncond_ids = vec![0i32; seq_len];
    uncond_ids[0] = 49406; // BOS
    uncond_ids[1] = 49407; // EOS

    let cond_input =
        NumericTensor::<DynRank>::from_vec_shape(cond_ids, vec![1, seq_len]).unwrap();
    let uncond_input =
        NumericTensor::<DynRank>::from_vec_shape(uncond_ids, vec![1, seq_len]).unwrap();

    // Generate initial random latent noise (seeded for reproducibility)
    let latent_n = 1 * 4 * latent_h * latent_w;
    use rand::SeedableRng;
    let mut latent_rng = rand::rngs::StdRng::seed_from_u64(42);
    let initial_noise: Vec<f32> = {
        let mut vals = Vec::with_capacity(latent_n);
        while vals.len() + 1 < latent_n {
            let u1: f32 = rand::Rng::random_range(&mut latent_rng, f32::EPSILON..1.0);
            let u2: f32 = rand::Rng::random_range(&mut latent_rng, 0.0f32..std::f32::consts::TAU);
            let r = (-2.0 * u1.ln()).sqrt();
            vals.push(r * u2.cos());
            vals.push(r * u2.sin());
        }
        if vals.len() < latent_n {
            let u1: f32 = rand::Rng::random_range(&mut latent_rng, f32::EPSILON..1.0);
            let u2: f32 = rand::Rng::random_range(&mut latent_rng, 0.0f32..std::f32::consts::TAU);
            vals.push((-2.0 * u1.ln()).sqrt() * u2.cos());
        }
        vals
    };

    // --- Run pipeline ---
    println!("=== Running SD pipeline ({num_inference_steps} steps) ===");
    let start = Instant::now();
    let image_tensor = sd_interface
        .run(
            &text_encoder,
            &unet,
            &vae_decoder,
            cond_input,
            uncond_input,
            initial_noise,
            vec![1, 4, latent_h, latent_w],
            num_inference_steps,
            guidance_scale,
            &mut backend,
        )
        .expect("SD pipeline failed");
    println!("  Pipeline took {:.2?}", start.elapsed());
    println!(
        "  Output image: dtype={:?}, shape={:?}",
        image_tensor.dtype(),
        image_tensor.shape()
    );
    println!();

    // --- Save PNG ---
    println!("=== Save PNG ===");
    let image_f32 = tensor_to_f32(&image_tensor, &mut backend);
    let shape = image_tensor.shape();
    let ch = shape[1] as usize;
    let img_h = shape[2] as u32;
    let img_w = shape[3] as u32;

    // image_f32 is NCHW; convert to RGB bytes, clamping [-1,1] -> [0,255]
    let mut pixels = vec![0u8; (img_h * img_w * 3) as usize];
    for y in 0..img_h as usize {
        for x in 0..img_w as usize {
            for c in 0..ch.min(3) {
                let idx = c * (img_h as usize * img_w as usize) + y * img_w as usize + x;
                let v = (image_f32[idx] + 1.0) * 0.5;
                let byte = (v.clamp(0.0, 1.0) * 255.0) as u8;
                pixels[y * img_w as usize * 3 + x * 3 + c] = byte;
            }
        }
    }

    let out_path = "sd_1_5_output.png";
    image::save_buffer(out_path, &pixels, img_w, img_h, image::ColorType::Rgb8)
        .expect("Failed to save PNG");
    println!("  Saved to {out_path}");
    println!();

    // --- Summary ---
    println!("=== Complete ===");
    println!("  Total time: {:.2?}", total_start.elapsed());
    println!("  Output image: {img_w}x{img_h} pixels");
}

/// Extract f32 values from a tensor (casting if needed).
fn tensor_to_f32(tensor: &NumericTensor<DynRank>, backend: &mut EvalBackend) -> Vec<f32> {
    use whisper_tensor::dtype::DType;
    let f32_tensor = tensor
        .cast(DType::F32, backend)
        .expect("Cast to f32 failed");
    let ndarray = f32_tensor.to_ndarray().expect("to_ndarray failed");
    let flat = ndarray.flatten();
    flat.try_into().expect("flatten to vec failed")
}
