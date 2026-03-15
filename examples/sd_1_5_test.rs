use ndarray::ArrayD;
use ndarray_npy::WriteNpyExt;
use std::path::Path;
use std::time::Instant;
use whisper_tensor::DynRank;
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::interfaces::ImageGenerationInterface;
use whisper_tensor::model::Model;
use whisper_tensor::numeric_tensor::NumericTensor;
#[allow(unused_imports)]
use whisper_tensor::symbolic_graph::observer::SymbolicGraphObserver;
use whisper_tensor_import::identify_and_load;
use whisper_tensor_import::onnx_graph::{TokenizerInfo, WeightStorageStrategy};

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

    let num_inference_steps = 20;
    let latent_h = 8;
    let latent_w = 8;
    let guidance_scale: f32 = 7.5;
    let total_start = Instant::now();

    // --- Load all models ---
    let text_encoder = load_model("text_encoder", "text_encoder");

    // Sanity check: run text encoder immediately after loading
    {
        let mut backend = EvalBackend::NDArray;
        let mut test_ids = vec![0i32; 77];
        test_ids[0] = 49406;
        test_ids[1] = 320;
        test_ids[2] = 1125;
        test_ids[3] = 539;
        test_ids[4] = 320;
        test_ids[5] = 2368;
        test_ids[6] = 49407;
        let test_input = NumericTensor::<DynRank>::from_vec_shape(test_ids, vec![1, 77]).unwrap();
        let out = text_encoder
            .eval(
                std::collections::HashMap::from([("input_ids".to_string(), test_input)]),
                &mut (),
                None,
                &mut backend,
            )
            .expect("sanity check failed");
        let hidden = out.get("last_hidden_state").unwrap();
        println!(
            "  SANITY CHECK: dtype={:?}, shape={:?}",
            hidden.dtype(),
            hidden.shape()
        );
        debug_tensor("sanity_check", hidden, &mut backend);
    }

    let unet = load_model("unet", "unet");
    let vae_decoder = load_model("vae_decoder", "vae_decoder");
    println!();

    let mut backend = EvalBackend::NDArray;

    // --- Build interface ---
    println!("=== Building ImageGenerationInterface ===");
    let start = Instant::now();
    let mut rng = rand::rng();
    let _interface = ImageGenerationInterface::new_single_te_cfg(
        &mut rng,
        TokenizerInfo::HFTokenizer("openai/clip-vit-large-patch14".to_string()),
        whisper_tensor::dtype::DType::F16,
        0.18215,
    );
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

    let cond_input = NumericTensor::<DynRank>::from_vec_shape(cond_ids, vec![1, seq_len]).unwrap();
    let uncond_input =
        NumericTensor::<DynRank>::from_vec_shape(uncond_ids, vec![1, seq_len]).unwrap();

    // Generate initial random latent noise (seeded for reproducibility)
    let latent_n = 4 * latent_h * latent_w;
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
        use whisper_tensor::dtype::DType;

        use std::collections::HashMap as HM;
        let mut obs = ();

        // Text encoder
        let cond_hidden = text_encoder
            .eval(
                HM::from([("input_ids".to_string(), cond_input)]),
                &mut obs,
                None,
                &mut backend,
            )
            .expect("text_encoder failed");
        let cond_hidden = cond_hidden.get("last_hidden_state").unwrap().clone();
        println!(
            "  cond_hidden: dtype={:?}, shape={:?}",
            cond_hidden.dtype(),
            cond_hidden.shape()
        );
        debug_tensor("cond_hidden", &cond_hidden, &mut backend);
        save_npy("cond_hidden", &cond_hidden, &mut backend);

        let uncond_hidden = text_encoder
            .eval(
                HM::from([("input_ids".to_string(), uncond_input)]),
                &mut obs,
                None,
                &mut backend,
            )
            .expect("text_encoder failed");
        let uncond_hidden = uncond_hidden.get("last_hidden_state").unwrap().clone();
        save_npy("uncond_hidden", &uncond_hidden, &mut backend);

        // Scheduler
        let (timestep_values, dt_values, sigmas, init_sigma) =
            ImageGenerationInterface::compute_euler_schedule(num_inference_steps);
        println!("  init_sigma={init_sigma}");
        println!("  timesteps[0..3]={:?}", &timestep_values[..3]);
        println!("  dt[0..3]={:?}", &dt_values[..3]);

        // Scale initial noise
        let scaled_noise: Vec<f32> = initial_noise.iter().map(|&x| x * init_sigma).collect();
        let mut latent =
            NumericTensor::<DynRank>::from_vec_shape(scaled_noise, vec![1, 4, latent_h, latent_w])
                .unwrap();
        save_npy("initial_latent", &latent, &mut backend);

        // Denoising loop
        for step in 0..num_inference_steps {
            let ts = timestep_values[step];
            let dt = dt_values[step];

            // Scale model input: latent / sqrt(sigma^2 + 1)
            let sigma = sigmas[step];
            let scale = 1.0 / (sigma * sigma + 1.0).sqrt();
            let latent_vals = tensor_to_f32(&latent, &mut backend);
            let scaled_latent_vals: Vec<f32> = latent_vals.iter().map(|&v| v * scale).collect();
            let scaled_latent = NumericTensor::<DynRank>::from_vec_shape(
                scaled_latent_vals,
                latent.shape().iter().map(|&x| x as usize).collect(),
            )
            .unwrap();

            // Cast latent f32→f16
            let f16_latent = scaled_latent.cast(DType::F16, &mut backend).unwrap();
            // Timestep as f16 [1]
            let ts_tensor = NumericTensor::<DynRank>::from_vec_shape(vec![ts], vec![1]).unwrap();
            let f16_ts = ts_tensor.cast(DType::F16, &mut backend).unwrap();

            // UNet unconditional
            let uncond_out = unet
                .eval(
                    HM::from([
                        ("sample".to_string(), f16_latent.clone()),
                        ("timestep".to_string(), f16_ts.clone()),
                        ("encoder_hidden_states".to_string(), uncond_hidden.clone()),
                    ]),
                    &mut obs,
                    None,
                    &mut backend,
                )
                .expect("unet uncond failed");
            let uncond_noise = uncond_out.get("out_sample").unwrap().clone();

            // UNet conditional
            let cond_out = unet
                .eval(
                    HM::from([
                        ("sample".to_string(), f16_latent),
                        ("timestep".to_string(), f16_ts),
                        ("encoder_hidden_states".to_string(), cond_hidden.clone()),
                    ]),
                    &mut obs,
                    None,
                    &mut backend,
                )
                .expect("unet cond failed");
            let cond_noise = cond_out.get("out_sample").unwrap().clone();

            if step == 0 {
                save_npy("step0_unet_uncond", &uncond_noise, &mut backend);
                save_npy("step0_unet_cond", &cond_noise, &mut backend);
            }

            // Cast to f32
            let uncond_f32 = uncond_noise.cast(DType::F32, &mut backend).unwrap();
            let cond_f32 = cond_noise.cast(DType::F32, &mut backend).unwrap();

            // CFG + Euler (element-wise)
            let uncond_vals = tensor_to_f32(&uncond_f32, &mut backend);
            let cond_vals = tensor_to_f32(&cond_f32, &mut backend);
            let latent_vals = tensor_to_f32(&latent, &mut backend);

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

            latent = NumericTensor::<DynRank>::from_vec_shape(
                new_latent_vals,
                vec![1, 4, latent_h, latent_w],
            )
            .unwrap();

            if step == 0 {
                save_npy("step0_latent_after", &latent, &mut backend);
            }

            if step % 5 == 0 || step == num_inference_steps - 1 {
                println!(
                    "  Step {step}/{num_inference_steps} done ({:.1?})",
                    start.elapsed()
                );
            }
        }

        save_npy("final_latent", &latent, &mut backend);

        // Scale by 1/0.18215 and cast to f16
        let lat_f32 = tensor_to_f32(&latent, &mut backend);
        let scaled: Vec<f32> = lat_f32.iter().map(|&v| v / 0.18215).collect();
        let scaled_tensor = NumericTensor::<DynRank>::from_vec_shape(
            scaled,
            latent.shape().iter().map(|&x| x as usize).collect(),
        )
        .unwrap();
        let scaled_f16 = scaled_tensor.cast(DType::F16, &mut backend).unwrap();

        debug_tensor("vae_input", &scaled_f16, &mut backend);

        // VAE decoder
        let vae_out = vae_decoder
            .eval(
                HM::from([("latent_sample".to_string(), scaled_f16)]),
                &mut obs,
                None,
                &mut backend,
            )
            .expect("vae_decoder failed");
        vae_out.get("sample").unwrap().clone()
    };
    println!("  Pipeline took {:.2?}", start.elapsed());
    println!(
        "  Output image: dtype={:?}, shape={:?}",
        image_tensor.dtype(),
        image_tensor.shape()
    );
    println!();

    // --- Debug output values ---
    let image_f32 = tensor_to_f32(&image_tensor, &mut backend);
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

fn debug_tensor(name: &str, tensor: &NumericTensor<DynRank>, backend: &mut EvalBackend) {
    let vals = tensor_to_f32(tensor, backend);
    let nan_count = vals.iter().filter(|v| v.is_nan()).count();
    let inf_count = vals.iter().filter(|v| v.is_infinite()).count();
    let min_val = vals.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("  {name}: min={min_val:.4}, max={max_val:.4}, nan={nan_count}, inf={inf_count}");
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

fn save_npy(name: &str, tensor: &NumericTensor<DynRank>, backend: &mut EvalBackend) {
    let vals = tensor_to_f32(tensor, backend);
    let shape: Vec<usize> = tensor.shape().iter().map(|&x| x as usize).collect();
    let arr = ArrayD::from_shape_vec(shape, vals).expect("shape mismatch");
    std::fs::create_dir_all("sd_dumps").ok();
    let path = format!("sd_dumps/{name}.npy");
    let mut f = std::fs::File::create(&path).expect("create file");
    arr.write_npy(&mut f).expect("write npy");
    println!("  Saved {path}");
}
