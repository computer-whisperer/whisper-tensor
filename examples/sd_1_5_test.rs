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

/// Simple Euler discrete scheduler for SD 1.5.
/// Produces a linear beta schedule and computes alphas_cumprod.
struct EulerDiscreteScheduler {
    timesteps: Vec<usize>,
    alphas_cumprod: Vec<f32>,
}

impl EulerDiscreteScheduler {
    fn new(num_inference_steps: usize) -> Self {
        let num_train_timesteps = 1000;
        let beta_start: f32 = 0.00085;
        let beta_end: f32 = 0.012;

        // Linear beta schedule (scaled_linear in diffusers — sqrt of linear interp)
        let betas: Vec<f32> = (0..num_train_timesteps)
            .map(|i| {
                let t = i as f32 / (num_train_timesteps - 1) as f32;
                let beta_sqrt = beta_start.sqrt() + t * (beta_end.sqrt() - beta_start.sqrt());
                beta_sqrt * beta_sqrt
            })
            .collect();

        let mut alphas_cumprod = Vec::with_capacity(num_train_timesteps);
        let mut cumprod = 1.0f32;
        for &beta in &betas {
            cumprod *= 1.0 - beta;
            alphas_cumprod.push(cumprod);
        }

        // Evenly spaced timesteps (descending)
        let step_ratio = num_train_timesteps / num_inference_steps;
        let timesteps: Vec<usize> = (0..num_inference_steps)
            .rev()
            .map(|i| i * step_ratio + step_ratio - 1)
            .collect();

        Self {
            timesteps,
            alphas_cumprod,
        }
    }

    /// Euler step: given model output (predicted noise), current sample, and timestep index,
    /// produce the denoised sample for the next step.
    fn step(&self, model_output: &[f32], sample: &[f32], step_idx: usize) -> Vec<f32> {
        let t = self.timesteps[step_idx];
        let alpha_prod_t = self.alphas_cumprod[t];
        let sigma_t = ((1.0 - alpha_prod_t) / alpha_prod_t).sqrt();

        // Predict x0 from noise prediction: x0 = (sample - sigma * noise) / sqrt(alpha)
        // Then step towards x0 based on next timestep's noise level
        if step_idx + 1 < self.timesteps.len() {
            let t_next = self.timesteps[step_idx + 1];
            let alpha_prod_next = self.alphas_cumprod[t_next];
            let sigma_next = ((1.0 - alpha_prod_next) / alpha_prod_next).sqrt();

            // Euler method: move from sigma_t towards sigma_next
            let dt = sigma_next - sigma_t;
            sample
                .iter()
                .zip(model_output.iter())
                .map(|(&s, &noise)| s + noise * dt)
                .collect()
        } else {
            // Final step: predict x0 directly
            let sqrt_alpha = alpha_prod_t.sqrt();
            let sqrt_one_minus_alpha = (1.0 - alpha_prod_t).sqrt();
            sample
                .iter()
                .zip(model_output.iter())
                .map(|(&s, &noise)| (s - sqrt_one_minus_alpha * noise) / sqrt_alpha)
                .collect()
        }
    }
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

    // --- Step 1: Text encoding ---
    println!("=== Step 1: Text Encoding ===");
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

    let start = Instant::now();

    // Encode conditional
    let cond_input = NumericTensor::<DynRank>::from_vec_shape(cond_ids, vec![1, seq_len]).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("input_ids".to_string(), cond_input);
    let cond_outputs = text_encoder
        .eval(inputs, &mut (), None, &mut backend)
        .expect("Conditional text encoding failed");
    let cond_hidden = cond_outputs
        .get("last_hidden_state")
        .expect("Missing last_hidden_state");

    // Encode unconditional
    let uncond_input =
        NumericTensor::<DynRank>::from_vec_shape(uncond_ids, vec![1, seq_len]).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("input_ids".to_string(), uncond_input);
    let uncond_outputs = text_encoder
        .eval(inputs, &mut (), None, &mut backend)
        .expect("Unconditional text encoding failed");
    let uncond_hidden = uncond_outputs
        .get("last_hidden_state")
        .expect("Missing last_hidden_state");

    println!(
        "  Conditional hidden: dtype={:?}, shape={:?}",
        cond_hidden.dtype(),
        cond_hidden.shape()
    );
    println!(
        "  Unconditional hidden: dtype={:?}, shape={:?}",
        uncond_hidden.dtype(),
        uncond_hidden.shape()
    );
    println!("  Text encoding took {:.2?}", start.elapsed());
    println!();

    // --- Step 2: Initialize latent noise ---
    println!("=== Step 2: Initialize Latent Noise ===");
    let latent_n = 1 * 4 * latent_h * latent_w;
    // Generate initial random latent (seeded for reproducibility)
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

    let scheduler = EulerDiscreteScheduler::new(num_inference_steps);

    // Scale initial noise by initial sigma
    let init_sigma = ((1.0 - scheduler.alphas_cumprod[scheduler.timesteps[0]])
        / scheduler.alphas_cumprod[scheduler.timesteps[0]])
        .sqrt();
    let mut latent: Vec<f32> = initial_noise.iter().map(|&x| x * init_sigma).collect();

    println!("  Latent shape: [1, 4, {latent_h}, {latent_w}], initial sigma: {init_sigma:.4}");
    println!();

    // --- Step 3: Denoising loop ---
    println!("=== Step 3: Denoising ({num_inference_steps} steps) ===");
    let denoise_start = Instant::now();

    for (step_idx, &timestep) in scheduler.timesteps.iter().enumerate() {
        let step_start = Instant::now();

        // Convert latent to f16 tensor
        let latent_f16: Vec<half::f16> = latent.iter().map(|&x| half::f16::from_f32(x)).collect();
        let latent_tensor =
            NumericTensor::<DynRank>::from_vec_shape(latent_f16, vec![1, 4, latent_h, latent_w])
                .unwrap();

        let timestep_data = vec![half::f16::from_f32(timestep as f32)];
        let timestep_tensor =
            NumericTensor::<DynRank>::from_vec_shape(timestep_data, vec![1]).unwrap();

        // Run unconditional prediction
        let mut uncond_inputs = HashMap::new();
        uncond_inputs.insert("sample".to_string(), latent_tensor.clone());
        uncond_inputs.insert("timestep".to_string(), timestep_tensor.clone());
        uncond_inputs.insert("encoder_hidden_states".to_string(), uncond_hidden.clone());
        let uncond_out = unet
            .eval(uncond_inputs, &mut (), None, &mut backend)
            .expect("Unconditional unet failed");
        let uncond_noise = uncond_out.get("out_sample").expect("Missing out_sample");

        // Run conditional prediction
        let mut cond_inputs = HashMap::new();
        cond_inputs.insert("sample".to_string(), latent_tensor);
        cond_inputs.insert("timestep".to_string(), timestep_tensor);
        cond_inputs.insert("encoder_hidden_states".to_string(), cond_hidden.clone());
        let cond_out = unet
            .eval(cond_inputs, &mut (), None, &mut backend)
            .expect("Conditional unet failed");
        let cond_noise = cond_out.get("out_sample").expect("Missing out_sample");

        // Classifier-free guidance: noise = uncond + guidance_scale * (cond - uncond)
        let uncond_f32: Vec<f32> = tensor_to_f32(uncond_noise, &mut backend);
        let cond_f32: Vec<f32> = tensor_to_f32(cond_noise, &mut backend);
        let guided_noise: Vec<f32> = uncond_f32
            .iter()
            .zip(cond_f32.iter())
            .map(|(&u, &c)| u + guidance_scale * (c - u))
            .collect();

        // Scheduler step
        latent = scheduler.step(&guided_noise, &latent, step_idx);

        println!(
            "  Step {}/{num_inference_steps} (t={timestep}) took {:.2?}",
            step_idx + 1,
            step_start.elapsed()
        );
    }
    println!("  Denoising took {:.2?}", denoise_start.elapsed());
    println!();

    // --- Step 4: VAE decode ---
    println!("=== Step 4: VAE Decoding ===");
    let start = Instant::now();

    // Scale latent by 1/0.18215 (SD VAE scaling factor)
    let vae_scale = 1.0 / 0.18215;
    let scaled_latent: Vec<half::f16> = latent
        .iter()
        .map(|&x| half::f16::from_f32(x * vae_scale))
        .collect();
    let latent_tensor =
        NumericTensor::<DynRank>::from_vec_shape(scaled_latent, vec![1, 4, latent_h, latent_w])
            .unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("latent_sample".to_string(), latent_tensor);
    let vae_outputs = vae_decoder
        .eval(inputs, &mut (), None, &mut backend)
        .expect("VAE decoding failed");

    let image_tensor = vae_outputs.get("sample").expect("Missing sample output");
    println!(
        "  Output image: dtype={:?}, shape={:?}",
        image_tensor.dtype(),
        image_tensor.shape()
    );
    println!("  VAE decoding took {:.2?}", start.elapsed());
    println!();

    // --- Step 5: Save PNG ---
    println!("=== Step 5: Save PNG ===");
    let image_f32 = tensor_to_f32(image_tensor, &mut backend);
    let shape = image_tensor.shape();
    let ch = shape[1] as usize;
    let img_h = shape[2] as u32;
    let img_w = shape[3] as u32;

    // image_f32 is NCHW; convert to RGB bytes, clamping [0,1] -> [0,255]
    let mut pixels = vec![0u8; (img_h * img_w * 3) as usize];
    for y in 0..img_h as usize {
        for x in 0..img_w as usize {
            for c in 0..ch.min(3) {
                let idx = c * (img_h as usize * img_w as usize) + y * img_w as usize + x;
                // SD output is roughly [-1, 1], remap to [0, 1]
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
