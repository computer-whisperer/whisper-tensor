use std::path::Path;
use std::time::Instant;
use whisper_tensor::DynRank;
use whisper_tensor::model::Model;
use whisper_tensor::numeric_dtype::NumericDType;
use whisper_tensor::numeric_scalar::NumericScalar;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::pool::SystemPool;
use whisper_tensor_import::models::diffusion::sd_xl;
use whisper_tensor_import::onnx_graph::WeightStorageStrategy;

const CHECKPOINT: &str = "/mnt/secondary/neural_networks/sd_xl_base_1.0.safetensors";

fn main() {
    tracing_subscriber::fmt::init();
    let pool = SystemPool;

    let total_start = Instant::now();
    let checkpoint_path = Path::new(CHECKPOINT);

    // Detect model dtype
    let model_dtype = {
        use memmap2::Mmap;
        use std::sync::Arc;
        use whisper_tensor_import::onnx_graph::weights::SafetensorsWeightManager;
        let file = std::fs::File::open(checkpoint_path).expect("open checkpoint");
        let mmap = unsafe { Mmap::map(&file) }.expect("mmap");
        let wm = SafetensorsWeightManager::new(vec![Arc::new(mmap)]).expect("wm");
        let import_dtype =
            whisper_tensor_import::models::diffusion::sd_common::detect_model_dtype(&wm);
        match import_dtype {
            whisper_tensor_import::onnx_graph::tensor::DType::F16 => NumericDType::F16,
            whisper_tensor_import::onnx_graph::tensor::DType::F32 => NumericDType::F32,
            other => panic!("Unsupported dtype: {:?}", other),
        }
    };
    println!("Model dtype: {:?}", model_dtype);

    // --- Build ONNX models from checkpoint ---
    println!("=== Building SDXL models from checkpoint ===");
    let start = Instant::now();
    let (te1_onnx, te2_onnx, unet_onnx, vae_onnx) =
        sd_xl::load_sdxl_checkpoint(checkpoint_path, WeightStorageStrategy::OriginReference)
            .expect("Failed to build SDXL models");
    println!("  Built in {:.2?}", start.elapsed());
    println!(
        "  ONNX sizes: te1={:.1}MB, te2={:.1}MB, unet={:.1}MB, vae={:.1}MB",
        te1_onnx.len() as f64 / 1e6,
        te2_onnx.len() as f64 / 1e6,
        unet_onnx.len() as f64 / 1e6,
        vae_onnx.len() as f64 / 1e6,
    );

    // --- Load into Model instances ---
    println!("\n=== Loading models ===");
    let base_dir = checkpoint_path.parent();

    let start = Instant::now();
    let mut rng = rand::rng();
    let te1 =
        Model::new_from_onnx(&te1_onnx, &mut rng, base_dir).expect("text_encoder_1 load failed");
    println!("  text_encoder_1 loaded in {:.2?}", start.elapsed());

    let start = Instant::now();
    let te2 =
        Model::new_from_onnx(&te2_onnx, &mut rng, base_dir).expect("text_encoder_2 load failed");
    println!("  text_encoder_2 loaded in {:.2?}", start.elapsed());

    let start = Instant::now();
    let unet = Model::new_from_onnx(&unet_onnx, &mut rng, base_dir).expect("unet load failed");
    println!("  unet loaded in {:.2?}", start.elapsed());

    let start = Instant::now();
    let vae_decoder =
        Model::new_from_onnx(&vae_onnx, &mut rng, base_dir).expect("vae_decoder load failed");
    println!("  vae_decoder loaded in {:.2?}", start.elapsed());

    // --- Test text encoder 1 (CLIP ViT-L/14) ---
    println!("\n=== Testing text encoder 1 (CLIP ViT-L/14) ===");
    {
        let seq_len = 77u64;
        let mut cond_ids = vec![0i32; seq_len as usize];
        cond_ids[0] = 49406; // BOS
        cond_ids[1] = 320; // "a"
        cond_ids[2] = 1125; // "photo"
        cond_ids[3] = 539; // "of"
        cond_ids[4] = 320; // "a"
        cond_ids[5] = 2368; // "cat"
        cond_ids[6] = 49407; // EOS
        let input = make_i32_tensor(&cond_ids, vec![1, seq_len], &pool);

        let input_view = input.view();
        let start = Instant::now();
        let out = te1
            .eval_pool(
                std::collections::HashMap::from([("input_ids".to_string(), &input_view)]),
                &pool,
            )
            .expect("text_encoder_1 eval failed");
        let hidden = out.get("last_hidden_state").unwrap();
        println!(
            "  Output: dtype={:?}, shape={:?}, took {:.2?}",
            hidden.dtype(),
            hidden.shape(),
            start.elapsed()
        );
        debug_tensor("  hidden_state", hidden, &pool);
    }

    // --- Test text encoder 2 (OpenCLIP ViT-bigG/14) ---
    println!("\n=== Testing text encoder 2 (OpenCLIP ViT-bigG/14) ===");
    {
        let seq_len = 77u64;
        let mut cond_ids = vec![0i32; seq_len as usize];
        cond_ids[0] = 49406; // BOS
        cond_ids[1] = 320; // "a"
        cond_ids[2] = 1125; // "photo"
        cond_ids[3] = 539; // "of"
        cond_ids[4] = 320; // "a"
        cond_ids[5] = 2368; // "cat"
        cond_ids[6] = 49407; // EOS
        let input = make_i32_tensor(&cond_ids, vec![1, seq_len], &pool);

        // EOS index = 6 (position of the EOS token)
        let eos_indices = make_i64_tensor(&[6i64], vec![1], &pool);

        let input_view = input.view();
        let eos_view = eos_indices.view();
        let start = Instant::now();
        let out = te2
            .eval_pool(
                std::collections::HashMap::from([
                    ("input_ids".to_string(), &input_view),
                    ("eos_indices".to_string(), &eos_view),
                ]),
                &pool,
            )
            .expect("text_encoder_2 eval failed");

        let hidden = out.get("penultimate_hidden_state").unwrap();
        println!(
            "  Penultimate: dtype={:?}, shape={:?}, took {:.2?}",
            hidden.dtype(),
            hidden.shape(),
            start.elapsed()
        );
        debug_tensor("  penultimate", hidden, &pool);

        let pooled = out.get("pooled_output").unwrap();
        println!(
            "  Pooled: dtype={:?}, shape={:?}",
            pooled.dtype(),
            pooled.shape()
        );
        debug_tensor("  pooled", pooled, &pool);
    }

    // --- Test UNet (single step) ---
    println!("\n=== Testing UNet (single step) ===");
    {
        let latent_h = 8u64;
        let latent_w = 8u64;

        let latent_f32 = make_f32_tensor(
            &vec![0.0f32; (4 * latent_h * latent_w) as usize],
            vec![1, 4, latent_h, latent_w],
            &pool,
        );
        let latent = cast_tensor(&latent_f32, model_dtype, &pool);

        let timestep_f32 = make_f32_tensor(&[999.0f32], vec![1], &pool);
        let timestep = cast_tensor(&timestep_f32, model_dtype, &pool);

        // Context in model_dtype (interface casts text encoder F32 output to model_dtype)
        let context_f32 = make_f32_tensor(&vec![0.0f32; 77 * 2048], vec![1, 77, 2048], &pool);
        let context = cast_tensor(&context_f32, model_dtype, &pool);

        // ADM conditioning in model_dtype
        let y_f32 = make_f32_tensor(&vec![0.0f32; 2816], vec![1, 2816], &pool);
        let y = cast_tensor(&y_f32, model_dtype, &pool);

        let latent_view = latent.view();
        let timestep_view = timestep.view();
        let context_view = context.view();
        let y_view = y.view();

        let start = Instant::now();
        let out = unet
            .eval_pool(
                std::collections::HashMap::from([
                    ("sample".to_string(), &latent_view),
                    ("timestep".to_string(), &timestep_view),
                    ("encoder_hidden_states".to_string(), &context_view),
                    ("y".to_string(), &y_view),
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
        let latent_f32 = make_f32_tensor(
            &vec![0.0f32; (4 * latent_h * latent_w) as usize],
            vec![1, 4, latent_h, latent_w],
            &pool,
        );
        let latent = cast_tensor(&latent_f32, model_dtype, &pool);

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

fn make_i64_tensor<'a>(
    data: &[i64],
    shape: Vec<u64>,
    pool: &'a SystemPool,
) -> NumericTensor<'a, DynRank, SystemPool> {
    NumericTensor::from_fn(shape, NumericDType::I64, pool, |i| {
        NumericScalar::from_i64(data[i])
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
