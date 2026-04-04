use std::path::Path;
use std::time::Instant;
use whisper_tensor::DynRank;
use whisper_tensor::interfaces::ImageGenerationInterface;
use whisper_tensor::numeric_dtype::NumericDType;
use whisper_tensor::numeric_scalar::NumericScalar;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::model::Model;
use whisper_tensor::pool::SystemPool;
#[allow(unused_imports)]
use whisper_tensor::symbolic_graph::observer::SymbolicGraphObserver;
use whisper_tensor_import::identify_and_load;
use whisper_tensor_import::onnx_graph::WeightStorageStrategy;

const SD_BASE: &str = "/mnt/secondary/neural_networks/stable-diffusion-1.5-onnx-fp16";

fn load_model(name: &str, subpath: &str) -> Model {
    let input_path = Path::new(SD_BASE).join(subpath).join("model.onnx");
    println!("Loading {name} from {}", input_path.display());
    let start = Instant::now();
    let onnx_data = identify_and_load(&input_path, WeightStorageStrategy::EmbeddedData)
        .expect("Failed to import model");
    let mut rng = rand::rng();
    let model = Model::new_from_onnx(&onnx_data, &mut rng, input_path.parent())
        .expect("Failed to load model");
    println!("  Loaded in {:.2?}", start.elapsed());
    model
}

fn main() {
    tracing_subscriber::fmt::init();
    let pool = SystemPool;

    let num_inference_steps = 20;
    let latent_h = 8u64;
    let latent_w = 8u64;
    let guidance_scale: f32 = 7.5;
    let total_start = Instant::now();

    // --- Load all models ---
    let text_encoder = load_model("text_encoder", "text_encoder");

    // Sanity check: run text encoder immediately after loading
    {
        let mut test_ids = vec![0i32; 77];
        test_ids[0] = 49406;
        test_ids[1] = 320;
        test_ids[2] = 1125;
        test_ids[3] = 539;
        test_ids[4] = 320;
        test_ids[5] = 2368;
        test_ids[6] = 49407;
        let test_input = make_i32_tensor(&test_ids, vec![1, 77], &pool);
        let test_input_view = test_input.view();
        let out = text_encoder
            .eval_pool(
                std::collections::HashMap::from([("input_ids".to_string(), &test_input_view)]),
                &pool,
            )
            .expect("sanity check failed");
        let hidden = out.get("last_hidden_state").unwrap();
        println!(
            "  SANITY CHECK: dtype={:?}, shape={:?}",
            hidden.dtype(),
            hidden.shape()
        );
        debug_tensor("sanity_check", hidden, &pool);
    }

    let unet = load_model("unet", "unet");
    let vae_decoder = load_model("vae_decoder", "vae_decoder");
    println!();

    // Interface construction now lives in whisper-tensor-import crate.
    // The manual pipeline below serves as a standalone validation test.

    // --- Prepare inputs ---
    let seq_len = 77u64;

    // Conditional prompt: "a photo of a cat"
    let mut cond_ids = vec![0i32; seq_len as usize];
    cond_ids[0] = 49406; // BOS
    cond_ids[1] = 320; // "a"
    cond_ids[2] = 1125; // "photo"
    cond_ids[3] = 539; // "of"
    cond_ids[4] = 320; // "a"
    cond_ids[5] = 2368; // "cat"
    cond_ids[6] = 49407; // EOS

    // Unconditional prompt: empty (just BOS + EOS + padding)
    let mut uncond_ids = vec![0i32; seq_len as usize];
    uncond_ids[0] = 49406; // BOS
    uncond_ids[1] = 49407; // EOS

    let cond_input = make_i32_tensor(&cond_ids, vec![1, seq_len], &pool);
    let uncond_input = make_i32_tensor(&uncond_ids, vec![1, seq_len], &pool);

    // Generate initial random latent noise (seeded for reproducibility)
    let latent_n = (4 * latent_h * latent_w) as usize;
    use rand::SeedableRng;
    let mut latent_rng = rand::rngs::StdRng::seed_from_u64(42);
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

    // --- Manual pipeline (no SuperGraph) for comparison ---
    println!("=== Running MANUAL pipeline ({num_inference_steps} steps) ===");
    let start = Instant::now();
    let image_tensor = {
        use std::collections::HashMap as HM;

        // Text encoder
        let cond_input_view = cond_input.view();
        let cond_hidden_out = text_encoder
            .eval_pool(
                HM::from([("input_ids".to_string(), &cond_input_view)]),
                &pool,
            )
            .expect("text_encoder failed");
        let cond_hidden = cond_hidden_out.get("last_hidden_state").unwrap();
        println!(
            "  cond_hidden: dtype={:?}, shape={:?}",
            cond_hidden.dtype(),
            cond_hidden.shape()
        );
        debug_tensor("cond_hidden", cond_hidden, &pool);
        save_npy("cond_hidden", cond_hidden);

        let uncond_input_view = uncond_input.view();
        let uncond_hidden_out = text_encoder
            .eval_pool(
                HM::from([("input_ids".to_string(), &uncond_input_view)]),
                &pool,
            )
            .expect("text_encoder failed");
        let uncond_hidden = uncond_hidden_out.get("last_hidden_state").unwrap();
        save_npy("uncond_hidden", uncond_hidden);

        // Scheduler
        let (timestep_values, dt_values, sigmas, init_sigma) =
            ImageGenerationInterface::compute_euler_schedule(num_inference_steps);
        println!("  init_sigma={init_sigma}");
        println!("  timesteps[0..3]={:?}", &timestep_values[..3]);
        println!("  dt[0..3]={:?}", &dt_values[..3]);

        // Scale initial noise
        let scaled_noise: Vec<f32> = initial_noise.iter().map(|&x| x * init_sigma).collect();
        let mut latent = make_f32_tensor(&scaled_noise, vec![1, 4, latent_h, latent_w], &pool);
        save_npy("initial_latent", &latent);

        // Denoising loop
        for step in 0..num_inference_steps {
            let ts = timestep_values[step];
            let dt = dt_values[step];

            // Scale model input: latent / sqrt(sigma^2 + 1)
            let sigma = sigmas[step];
            let scale = 1.0 / (sigma * sigma + 1.0).sqrt();
            let latent_vals = tensor_to_f32(&latent);
            let scaled_latent_vals: Vec<f32> = latent_vals.iter().map(|&v| v * scale).collect();
            let scaled_latent = make_f32_tensor(
                &scaled_latent_vals,
                latent.shape().clone(),
                &pool,
            );

            // Cast latent f32->f16
            let f16_latent = cast_tensor(&scaled_latent, NumericDType::F16, &pool);
            // Timestep as f16 [1]
            let ts_f32 = make_f32_tensor(&[ts], vec![1], &pool);
            let f16_ts = cast_tensor(&ts_f32, NumericDType::F16, &pool);

            // UNet unconditional
            let f16_latent_view = f16_latent.view();
            let f16_ts_view = f16_ts.view();
            let uncond_hidden_view = uncond_hidden.view();
            let uncond_out = unet
                .eval_pool(
                    HM::from([
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
                    HM::from([
                        ("sample".to_string(), &f16_latent_view),
                        ("timestep".to_string(), &f16_ts_view),
                        ("encoder_hidden_states".to_string(), &cond_hidden_view),
                    ]),
                    &pool,
                )
                .expect("unet cond failed");
            let cond_noise = cond_out.get("out_sample").unwrap();

            if step == 0 {
                save_npy("step0_unet_uncond", uncond_noise);
                save_npy("step0_unet_cond", cond_noise);
            }

            // Cast to f32
            let uncond_f32 = cast_tensor(uncond_noise, NumericDType::F32, &pool);
            let cond_f32 = cast_tensor(cond_noise, NumericDType::F32, &pool);

            // CFG + Euler (element-wise)
            let uncond_vals = tensor_to_f32(&uncond_f32);
            let cond_vals = tensor_to_f32(&cond_f32);
            let latent_vals = tensor_to_f32(&latent);

            let new_latent_vals: Vec<f32> = latent_vals
                .iter()
                .zip(uncond_vals.iter().zip(cond_vals.iter()))
                .map(|(&l, (&u, &c))| {
                    let guided = u + guidance_scale * (c - u);
                    l + guided * dt
                })
                .collect();

            if step == 0 {
                let nan_count = new_latent_vals.iter().filter(|v| v.is_nan()).count();
                println!(
                    "  Step {step}: latent min={:.4}, max={:.4}, nan={nan_count}",
                    new_latent_vals
                        .iter()
                        .cloned()
                        .fold(f32::INFINITY, f32::min),
                    new_latent_vals
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max)
                );
            }

            latent = make_f32_tensor(
                &new_latent_vals,
                vec![1, 4, latent_h, latent_w],
                &pool,
            );

            if step == 0 {
                save_npy("step0_latent_after", &latent);
            }

            if step % 5 == 0 || step == num_inference_steps - 1 {
                println!(
                    "  Step {step}/{num_inference_steps} done ({:.1?})",
                    start.elapsed()
                );
            }
        }

        save_npy("final_latent", &latent);

        // Scale by 1/0.18215 and cast to f16
        let lat_f32 = tensor_to_f32(&latent);
        let scaled: Vec<f32> = lat_f32.iter().map(|&v| v / 0.18215).collect();
        let scaled_tensor = make_f32_tensor(
            &scaled,
            latent.shape().clone(),
            &pool,
        );
        let scaled_f16 = cast_tensor(&scaled_tensor, NumericDType::F16, &pool);

        debug_tensor("vae_input", &scaled_f16, &pool);

        // VAE decoder
        let scaled_f16_view = scaled_f16.view();
        let vae_out = vae_decoder
            .eval_pool(
                HM::from([("latent_sample".to_string(), &scaled_f16_view)]),
                &pool,
            )
            .expect("vae_decoder failed");
        // Clone/copy the output so it outlives the block
        let sample = vae_out.get("sample").unwrap();
        sample.to_tensor(&pool).unwrap()
    };
    println!("  Pipeline took {:.2?}", start.elapsed());
    println!(
        "  Output image: dtype={:?}, shape={:?}",
        image_tensor.dtype(),
        image_tensor.shape()
    );
    println!();

    // --- Debug output values ---
    let image_f32 = tensor_to_f32(&image_tensor);
    let nan_count = image_f32.iter().filter(|v| v.is_nan()).count();
    let inf_count = image_f32.iter().filter(|v| v.is_infinite()).count();
    let min_val = image_f32.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = image_f32.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean_val: f32 = image_f32.iter().sum::<f32>() / image_f32.len() as f32;
    println!("  Image values: min={min_val}, max={max_val}, mean={mean_val}");
    println!("  NaN count: {nan_count}, Inf count: {inf_count}");
    println!(
        "  First 10 values: {:?}",
        &image_f32[..10.min(image_f32.len())]
    );
    println!();

    // --- Save PNG ---
    println!("=== Save PNG ===");
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

fn debug_tensor(name: &str, tensor: &NumericTensor<'_, DynRank, SystemPool>, _pool: &SystemPool) {
    let vals = tensor_to_f32(tensor);
    let nan_count = vals.iter().filter(|v| v.is_nan()).count();
    let inf_count = vals.iter().filter(|v| v.is_infinite()).count();
    let min_val = vals.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("  {name}: min={min_val:.4}, max={max_val:.4}, nan={nan_count}, inf={inf_count}");
}

fn save_npy(name: &str, tensor: &NumericTensor<'_, DynRank, SystemPool>) {
    std::fs::create_dir_all("sd_dumps").ok();
    let path = format!("sd_dumps/{name}.npy");
    whisper_tensor::npy::write_npy_file(Path::new(&path), &tensor.view()).expect("write npy");
    println!("  Saved {path}");
}
