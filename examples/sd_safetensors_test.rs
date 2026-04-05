use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;
use whisper_tensor::DynRank;
use whisper_tensor::loader::{ConfigValue, ConfigValues, Loader};
use whisper_tensor::numeric_dtype::NumericDType;
use whisper_tensor::numeric_scalar::NumericScalar;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::pool::SystemPool;
use whisper_tensor_import::loaders::SD15Loader;

const CHECKPOINT: &str =
    "/ceph/public/neural_models/comfyui/checkpoints/v1-5-pruned-emaonly.safetensors";

fn main() {
    tracing_subscriber::fmt::init();
    let pool = SystemPool;

    let total_start = Instant::now();

    // --- Load via SD15 loader ---
    println!("=== Loading SD 1.5 checkpoint via SD15Loader ===");
    let start = Instant::now();
    let config = ConfigValues::from([(
        "path".to_string(),
        ConfigValue::FilePath(PathBuf::from(CHECKPOINT)),
    )]);
    let output = SD15Loader.load(config).unwrap();
    println!("  Loaded in {:.2?}", start.elapsed());

    // Extract models by name
    let text_encoder = &output.models[0].model;
    let unet = &output.models[1].model;
    let vae_decoder = &output.models[2].model;

    println!(
        "  Models: {}, {}, {} ({} interfaces)",
        output.models[0].name,
        output.models[1].name,
        output.models[2].name,
        output.interfaces.len(),
    );

    // --- Run text encoder ---
    println!("\n=== Text encoder ===");
    let seq_len = 77u64;

    // "a photo of a cat"
    let mut cond_ids = vec![0i32; seq_len as usize];
    cond_ids[0] = 49406; // BOS
    cond_ids[1] = 320; // "a"
    cond_ids[2] = 1125; // "photo"
    cond_ids[3] = 539; // "of"
    cond_ids[4] = 320; // "a"
    cond_ids[5] = 2368; // "cat"
    cond_ids[6] = 49407; // EOS

    let mut uncond_ids = vec![0i32; seq_len as usize];
    uncond_ids[0] = 49406;
    uncond_ids[1] = 49407;

    let cond_input = make_i32_tensor(&cond_ids, vec![1, seq_len], &pool);
    let uncond_input = make_i32_tensor(&uncond_ids, vec![1, seq_len], &pool);

    let start = Instant::now();
    let cond_input_view = cond_input.view();
    let cond_hidden_out = text_encoder
        .eval_pool(
            HashMap::from([("input_ids".to_string(), &cond_input_view)]),
            &pool,
        )
        .expect("text_encoder cond failed");
    let cond_hidden = cond_hidden_out.get("last_hidden_state").unwrap();
    println!(
        "  cond_hidden: dtype={:?}, shape={:?} ({:.2?})",
        cond_hidden.dtype(),
        cond_hidden.shape(),
        start.elapsed(),
    );
    debug_tensor("  cond_hidden", cond_hidden, &pool);

    let start = Instant::now();
    let uncond_input_view = uncond_input.view();
    let uncond_hidden_out = text_encoder
        .eval_pool(
            HashMap::from([("input_ids".to_string(), &uncond_input_view)]),
            &pool,
        )
        .expect("text_encoder uncond failed");
    let uncond_hidden = uncond_hidden_out.get("last_hidden_state").unwrap();
    println!(
        "  uncond_hidden: dtype={:?}, shape={:?} ({:.2?})",
        uncond_hidden.dtype(),
        uncond_hidden.shape(),
        start.elapsed(),
    );
    debug_tensor("  uncond_hidden", uncond_hidden, &pool);

    // --- Scheduler ---
    let num_inference_steps = 20;
    let latent_h = 64u64;
    let latent_w = 64u64;
    let guidance_scale: f32 = 7.5;

    let (timestep_values, dt_values, sigmas, init_sigma) =
        whisper_tensor::interfaces::ImageGenerationInterface::compute_euler_schedule(
            num_inference_steps,
        );
    println!("\n=== Scheduler ===");
    println!("  init_sigma={init_sigma}");
    println!(
        "  timesteps={:?}",
        &timestep_values[..timestep_values.len().min(3)]
    );

    // --- Initial latent noise ---
    use rand::SeedableRng;
    let mut latent_rng = rand::rngs::StdRng::seed_from_u64(42);
    let latent_n = (4 * latent_h * latent_w) as usize;
    let initial_noise: Vec<f32> = {
        let mut vals = Vec::with_capacity(latent_n);
        while vals.len() + 1 < latent_n {
            let u1: f32 = rand::RngExt::random_range(&mut latent_rng, f32::EPSILON..1.0);
            let u2: f32 =
                rand::RngExt::random_range(&mut latent_rng, 0.0f32..std::f32::consts::TAU);
            let r = (-2.0f32 * u1.ln()).sqrt();
            vals.push(r * u2.cos());
            vals.push(r * u2.sin());
        }
        if vals.len() < latent_n {
            let u1: f32 = rand::RngExt::random_range(&mut latent_rng, f32::EPSILON..1.0);
            let u2: f32 =
                rand::RngExt::random_range(&mut latent_rng, 0.0f32..std::f32::consts::TAU);
            vals.push((-2.0f32 * u1.ln()).sqrt() * u2.cos());
        }
        vals
    };
    let scaled_noise: Vec<f32> = initial_noise.iter().map(|&x| x * init_sigma).collect();
    let mut latent = make_f32_tensor(&scaled_noise, vec![1, 4, latent_h, latent_w], &pool);

    // --- Denoising loop ---
    println!("\n=== Denoising ({num_inference_steps} steps) ===");
    let denoise_start = Instant::now();
    for step in 0..num_inference_steps {
        let sigma = sigmas[step];
        let ts = timestep_values[step];
        let dt = dt_values[step];

        // Scale: latent / sqrt(sigma^2 + 1)
        let scale = 1.0 / (sigma * sigma + 1.0).sqrt();
        let latent_vals = tensor_to_f32(&latent);
        let scaled_vals: Vec<f32> = latent_vals.iter().map(|&v| v * scale).collect();
        let scaled_latent = make_f32_tensor(&scaled_vals, vec![1, 4, latent_h, latent_w], &pool);

        let f16_latent = cast_tensor(&scaled_latent, NumericDType::F16, &pool);
        let ts_f32 = make_f32_tensor(&[ts], vec![1], &pool);
        let f16_ts = cast_tensor(&ts_f32, NumericDType::F16, &pool);

        // UNet unconditional
        let step_start = Instant::now();
        let f16_latent_view = f16_latent.view();
        let f16_ts_view = f16_ts.view();
        let uncond_hidden_view = uncond_hidden.view();
        let uncond_out = unet
            .eval_pool(
                HashMap::from([
                    ("sample".to_string(), &f16_latent_view),
                    ("timestep".to_string(), &f16_ts_view),
                    ("encoder_hidden_states".to_string(), &uncond_hidden_view),
                ]),
                &pool,
            )
            .expect("unet uncond failed");
        let uncond_noise = uncond_out.get("out_sample").unwrap();

        // UNet conditional
        let cond_hidden_view = cond_hidden.view();
        let cond_out = unet
            .eval_pool(
                HashMap::from([
                    ("sample".to_string(), &f16_latent_view),
                    ("timestep".to_string(), &f16_ts_view),
                    ("encoder_hidden_states".to_string(), &cond_hidden_view),
                ]),
                &pool,
            )
            .expect("unet cond failed");
        let cond_noise = cond_out.get("out_sample").unwrap();

        // Cast to f32 for CFG
        let uncond_f32 = tensor_to_f32(&cast_tensor(uncond_noise, NumericDType::F32, &pool));
        let cond_f32 = tensor_to_f32(&cast_tensor(cond_noise, NumericDType::F32, &pool));
        let latent_vals = tensor_to_f32(&latent);

        // CFG + Euler step
        let new_vals: Vec<f32> = latent_vals
            .iter()
            .zip(uncond_f32.iter().zip(cond_f32.iter()))
            .map(|(&l, (&u, &c))| {
                let guided = u + guidance_scale * (c - u);
                l + guided * dt
            })
            .collect();

        let nan_count = new_vals.iter().filter(|v| v.is_nan()).count();
        if step == 0 || step % 5 == 0 || step == num_inference_steps - 1 || nan_count > 0 {
            let min = new_vals.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = new_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!(
                "  Step {step}/{num_inference_steps}: min={min:.4}, max={max:.4}, nan={nan_count} ({:.1?})",
                step_start.elapsed()
            );
        }

        latent = make_f32_tensor(&new_vals, vec![1, 4, latent_h, latent_w], &pool);
    }
    println!("  Denoising took {:.2?}", denoise_start.elapsed());

    // --- VAE decode ---
    println!("\n=== VAE decode ===");
    let lat_f32 = tensor_to_f32(&latent);
    let scaled: Vec<f32> = lat_f32.iter().map(|&v| v / 0.18215).collect();
    let scaled_tensor = make_f32_tensor(&scaled, vec![1, 4, latent_h, latent_w], &pool);
    let scaled_f16 = cast_tensor(&scaled_tensor, NumericDType::F16, &pool);
    debug_tensor("  vae_input", &scaled_f16, &pool);

    let start = Instant::now();
    let scaled_f16_view = scaled_f16.view();
    let vae_out = vae_decoder
        .eval_pool(
            HashMap::from([("latent_sample".to_string(), &scaled_f16_view)]),
            &pool,
        )
        .expect("vae_decoder failed");
    let image_tensor = vae_out.get("sample").unwrap();
    println!(
        "  VAE decode: dtype={:?}, shape={:?} ({:.2?})",
        image_tensor.dtype(),
        image_tensor.shape(),
        start.elapsed()
    );

    // --- Output ---
    let image_f32_tensor = cast_tensor(image_tensor, NumericDType::F32, &pool);
    let image_f32 = tensor_to_f32(&image_f32_tensor);
    let nan_count = image_f32.iter().filter(|v| v.is_nan()).count();
    let min_val = image_f32.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = image_f32.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("  Image values: min={min_val:.4}, max={max_val:.4}, nan={nan_count}");

    // Save PNG
    let shape = image_tensor.shape();
    let ch = shape[1] as usize;
    let img_h = shape[2] as u32;
    let img_w = shape[3] as u32;

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

    let out_path = "sd_safetensors_output.png";
    image::save_buffer(out_path, &pixels, img_w, img_h, image::ColorType::Rgb8)
        .expect("Failed to save PNG");
    println!("\n  Saved to {out_path}");
    println!("  Total time: {:.2?}", total_start.elapsed());
}

fn make_f32_tensor<'a>(
    data: &[f32],
    shape: Vec<u64>,
    pool: &'a SystemPool,
) -> NumericTensor<'a, DynRank, SystemPool> {
    NumericTensor::from_fn(shape, NumericDType::F32, pool, |i| {
        NumericScalar::from_f32(data[i])
    })
    .unwrap()
}

fn make_i32_tensor<'a>(
    data: &[i32],
    shape: Vec<u64>,
    pool: &'a SystemPool,
) -> NumericTensor<'a, DynRank, SystemPool> {
    NumericTensor::from_fn(shape, NumericDType::I32, pool, |i| {
        NumericScalar::from_i32(data[i])
    })
    .unwrap()
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

fn tensor_to_f32(tensor: &NumericTensor<'_, DynRank, SystemPool>) -> Vec<f32> {
    (0..tensor.numel())
        .map(|i| tensor.read_element(i).cast_to(NumericDType::F32).to_f32())
        .collect()
}

fn debug_tensor(name: &str, tensor: &NumericTensor<'_, DynRank, SystemPool>, pool: &SystemPool) {
    let f32_tensor = cast_tensor(tensor, NumericDType::F32, pool);
    let flat: Vec<f32> = (0..f32_tensor.numel())
        .map(|i| f32_tensor.read_element(i).to_f32())
        .collect();
    let nan_count = flat.iter().filter(|v| v.is_nan()).count();
    let inf_count = flat.iter().filter(|v| v.is_infinite()).count();
    let min = flat.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("{name}: min={min:.4}, max={max:.4}, nan={nan_count}, inf={inf_count}");
}
