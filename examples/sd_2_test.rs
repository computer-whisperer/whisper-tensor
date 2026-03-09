use std::path::Path;
use std::time::Instant;
use whisper_tensor::DynRank;
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::model::Model;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor_import::onnx_graph::WeightStorageStrategy;
use whisper_tensor_import::sd2;

const CHECKPOINT: &str = "/mnt/secondary/neural_networks/sd2.1/v2-1_768-ema-pruned.safetensors";

fn main() {
    tracing_subscriber::fmt::init();

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

    let mut backend = EvalBackend::NDArray;

    // --- Test text encoder ---
    println!("\n=== Testing text encoder ===");
    {
        let seq_len = 77;
        // "a photo of a cat" — using CLIP BOS/EOS token IDs (same vocab as SD 1.5)
        let mut cond_ids = vec![0i32; seq_len];
        cond_ids[0] = 49406; // BOS
        cond_ids[1] = 320; // "a"
        cond_ids[2] = 1125; // "photo"
        cond_ids[3] = 539; // "of"
        cond_ids[4] = 320; // "a"
        cond_ids[5] = 2368; // "cat"
        cond_ids[6] = 49407; // EOS
        let input = NumericTensor::<DynRank>::from_vec_shape(cond_ids, vec![1, seq_len]).unwrap();

        let start = Instant::now();
        let out = text_encoder
            .eval(
                std::collections::HashMap::from([("input_ids".to_string(), input)]),
                &mut (),
                None,
                &mut backend,
            )
            .expect("text_encoder eval failed");
        let hidden = out.get("last_hidden_state").unwrap();
        println!(
            "  Output: dtype={:?}, shape={:?}, took {:.2?}",
            hidden.dtype(),
            hidden.shape(),
            start.elapsed()
        );
        debug_tensor("  hidden_state", hidden, &mut backend);
    }

    // --- Test UNet (single step) ---
    println!("\n=== Testing UNet (single step) ===");
    {
        let latent_h = 8;
        let latent_w = 8;

        // Dummy latent
        let latent = NumericTensor::<DynRank>::from_vec_shape(
            vec![0.0f32; 4 * latent_h * latent_w],
            vec![1, 4, latent_h, latent_w],
        )
        .unwrap();

        // Dummy timestep
        let timestep = NumericTensor::<DynRank>::from_vec_shape(vec![999.0f32], vec![1]).unwrap();

        // Dummy context (1024-dim for SD 2)
        let context =
            NumericTensor::<DynRank>::from_vec_shape(vec![0.0f32; 77 * 1024], vec![1, 77, 1024])
                .unwrap();

        let start = Instant::now();
        let out = unet
            .eval(
                std::collections::HashMap::from([
                    ("sample".to_string(), latent),
                    ("timestep".to_string(), timestep),
                    ("encoder_hidden_states".to_string(), context),
                ]),
                &mut (),
                None,
                &mut backend,
            )
            .expect("unet eval failed");
        let noise = out.get("out_sample").unwrap();
        println!(
            "  Output: dtype={:?}, shape={:?}, took {:.2?}",
            noise.dtype(),
            noise.shape(),
            start.elapsed()
        );
        debug_tensor("  noise_pred", noise, &mut backend);
    }

    // --- Test VAE decoder ---
    println!("\n=== Testing VAE decoder ===");
    {
        let latent_h = 8;
        let latent_w = 8;
        let latent = NumericTensor::<DynRank>::from_vec_shape(
            vec![0.0f32; 4 * latent_h * latent_w],
            vec![1, 4, latent_h, latent_w],
        )
        .unwrap();

        let start = Instant::now();
        let out = vae_decoder
            .eval(
                std::collections::HashMap::from([("latent_sample".to_string(), latent)]),
                &mut (),
                None,
                &mut backend,
            )
            .expect("vae_decoder eval failed");
        let image = out.get("sample").unwrap();
        println!(
            "  Output: dtype={:?}, shape={:?}, took {:.2?}",
            image.dtype(),
            image.shape(),
            start.elapsed()
        );
        debug_tensor("  image", image, &mut backend);
    }

    println!("\n=== Complete in {:.2?} ===", total_start.elapsed());
}

fn debug_tensor(name: &str, tensor: &NumericTensor<DynRank>, backend: &mut EvalBackend) {
    use whisper_tensor::dtype::DType;
    let f32_tensor = tensor.cast(DType::F32, backend).expect("cast failed");
    let ndarray = f32_tensor.to_ndarray().expect("to_ndarray failed");
    let flat: Vec<f32> = ndarray.flatten().try_into().expect("flatten failed");
    let nan_count = flat.iter().filter(|v| v.is_nan()).count();
    let inf_count = flat.iter().filter(|v| v.is_infinite()).count();
    let min_val = flat.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("{name}: min={min_val:.4}, max={max_val:.4}, nan={nan_count}, inf={inf_count}");
}
