use std::path::Path;
use std::time::Instant;
use whisper_tensor::DynRank;
use whisper_tensor::model::Model;
use whisper_tensor::numeric_dtype::NumericDType;
use whisper_tensor::numeric_scalar::NumericScalar;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::pool::SystemPool;
use whisper_tensor_import::models::diffusion::sd2;
use whisper_tensor_import::onnx_graph::WeightStorageStrategy;

const CHECKPOINT: &str = "/mnt/secondary/neural_networks/sd2.1/v2-1_768-ema-pruned.safetensors";

fn main() {
    tracing_subscriber::fmt::init();
    let pool = SystemPool;

    let total_start = Instant::now();
    let checkpoint_path = Path::new(CHECKPOINT);

    // --- Build ONNX models from checkpoint ---
    println!("=== Building SD 2.1 models from checkpoint ===");
    let start = Instant::now();
    let (te_onnx, unet_onnx, vae_onnx) =
        sd2::load_sd2_checkpoint(checkpoint_path, WeightStorageStrategy::OriginReference)
            .expect("Failed to build SD 2 models");
    println!("  Built in {:.2?}", start.elapsed());
    println!(
        "  ONNX sizes: te={:.1}MB, unet={:.1}MB, vae={:.1}MB",
        te_onnx.len() as f64 / 1e6,
        unet_onnx.len() as f64 / 1e6,
        vae_onnx.len() as f64 / 1e6,
    );

    // --- Load into Model instances ---
    println!("\n=== Loading models ===");
    let base_dir = checkpoint_path.parent();

    let start = Instant::now();
    let mut rng = rand::rng();
    let text_encoder =
        Model::new_from_onnx(&te_onnx, &mut rng, base_dir).expect("text_encoder load failed");
    println!("  text_encoder loaded in {:.2?}", start.elapsed());

    let start = Instant::now();
    let unet = Model::new_from_onnx(&unet_onnx, &mut rng, base_dir).expect("unet load failed");
    println!("  unet loaded in {:.2?}", start.elapsed());

    let start = Instant::now();
    let vae_decoder =
        Model::new_from_onnx(&vae_onnx, &mut rng, base_dir).expect("vae_decoder load failed");
    println!("  vae_decoder loaded in {:.2?}", start.elapsed());

    // --- Test text encoder ---
    println!("\n=== Testing text encoder ===");
    {
        let seq_len = 77;
        // "a photo of a cat" -- using CLIP BOS/EOS token IDs (same vocab as SD 1.5)
        let mut cond_ids = vec![0i32; seq_len];
        cond_ids[0] = 49406; // BOS
        cond_ids[1] = 320; // "a"
        cond_ids[2] = 1125; // "photo"
        cond_ids[3] = 539; // "of"
        cond_ids[4] = 320; // "a"
        cond_ids[5] = 2368; // "cat"
        cond_ids[6] = 49407; // EOS
        let input = make_i32_tensor(&cond_ids, vec![1, seq_len as u64], &pool);

        let input_view = input.view();
        let start = Instant::now();
        let out = text_encoder
            .eval_pool(
                std::collections::HashMap::from([("input_ids".to_string(), &input_view)]),
                &pool,
            )
            .expect("text_encoder eval failed");
        let hidden = out.get("last_hidden_state").unwrap();
        println!(
            "  Output: dtype={:?}, shape={:?}, took {:.2?}",
            hidden.dtype(),
            hidden.shape(),
            start.elapsed()
        );
        debug_tensor("  hidden_state", hidden, &pool);
    }

    // --- Test UNet (single step) ---
    println!("\n=== Testing UNet (single step) ===");
    {
        let latent_h = 8u64;
        let latent_w = 8u64;

        // Dummy latent
        let latent = make_f32_tensor(
            &vec![0.0f32; (4 * latent_h * latent_w) as usize],
            vec![1, 4, latent_h, latent_w],
            &pool,
        );

        // Dummy timestep
        let timestep = make_f32_tensor(&[999.0f32], vec![1], &pool);

        // Dummy context (1024-dim for SD 2)
        let context = make_f32_tensor(&vec![0.0f32; 77 * 1024], vec![1, 77, 1024], &pool);

        let latent_view = latent.view();
        let timestep_view = timestep.view();
        let context_view = context.view();

        let start = Instant::now();
        let out = unet
            .eval_pool(
                std::collections::HashMap::from([
                    ("sample".to_string(), &latent_view),
                    ("timestep".to_string(), &timestep_view),
                    ("encoder_hidden_states".to_string(), &context_view),
                ]),
                &pool,
            )
            .expect("unet eval failed");
        let noise = out.get("out_sample").unwrap();
        println!(
            "  Output: dtype={:?}, shape={:?}, took {:.2?}",
            noise.dtype(),
            noise.shape(),
            start.elapsed()
        );
        debug_tensor("  noise_pred", noise, &pool);
    }

    // --- Test VAE decoder ---
    println!("\n=== Testing VAE decoder ===");
    {
        let latent_h = 8u64;
        let latent_w = 8u64;
        let latent = make_f32_tensor(
            &vec![0.0f32; (4 * latent_h * latent_w) as usize],
            vec![1, 4, latent_h, latent_w],
            &pool,
        );

        let latent_view = latent.view();
        let start = Instant::now();
        let out = vae_decoder
            .eval_pool(
                std::collections::HashMap::from([("latent_sample".to_string(), &latent_view)]),
                &pool,
            )
            .expect("vae_decoder eval failed");
        let image = out.get("sample").unwrap();
        println!(
            "  Output: dtype={:?}, shape={:?}, took {:.2?}",
            image.dtype(),
            image.shape(),
            start.elapsed()
        );
        debug_tensor("  image", image, &pool);
    }

    println!("\n=== Complete in {:.2?} ===", total_start.elapsed());
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

fn debug_tensor(name: &str, tensor: &NumericTensor<'_, DynRank, SystemPool>, pool: &SystemPool) {
    let f32_tensor = cast_tensor(tensor, NumericDType::F32, pool);
    let flat: Vec<f32> = (0..f32_tensor.numel())
        .map(|i| f32_tensor.read_element(i).to_f32())
        .collect();
    let nan_count = flat.iter().filter(|v| v.is_nan()).count();
    let inf_count = flat.iter().filter(|v| v.is_infinite()).count();
    let min_val = flat.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("{name}: min={min_val:.4}, max={max_val:.4}, nan={nan_count}, inf={inf_count}");
}
