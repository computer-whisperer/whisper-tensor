use std::path::Path;
use std::time::Instant;
use whisper_tensor::DynRank;
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::dtype::DType;
use whisper_tensor::model::Model;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor_import::onnx_graph::WeightStorageStrategy;
use whisper_tensor_import::sd_xl;

const CHECKPOINT: &str = "/mnt/secondary/neural_networks/sd_xl_base_1.0.safetensors";

fn main() {
    tracing_subscriber::fmt::init();

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
        let import_dtype = whisper_tensor_import::sd_common::detect_model_dtype(&wm);
        match import_dtype {
            whisper_tensor_import::onnx_graph::tensor::DType::F16 => DType::F16,
            whisper_tensor_import::onnx_graph::tensor::DType::F32 => DType::F32,
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

    let mut backend = EvalBackend::NDArray;

    // --- Test text encoder 1 (CLIP ViT-L/14) ---
    println!("\n=== Testing text encoder 1 (CLIP ViT-L/14) ===");
    {
        let seq_len = 77;
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
        let out = te1
            .eval(
                std::collections::HashMap::from([("input_ids".to_string(), input)]),
                &mut (),
                None,
                &mut backend,
            )
            .expect("text_encoder_1 eval failed");
        let hidden = out.get("last_hidden_state").unwrap();
        println!(
            "  Output: dtype={:?}, shape={:?}, took {:.2?}",
            hidden.dtype(),
            hidden.shape(),
            start.elapsed()
        );
        debug_tensor("  hidden_state", hidden, &mut backend);
    }

    // --- Test text encoder 2 (OpenCLIP ViT-bigG/14) ---
    println!("\n=== Testing text encoder 2 (OpenCLIP ViT-bigG/14) ===");
    {
        let seq_len = 77;
        let mut cond_ids = vec![0i32; seq_len];
        cond_ids[0] = 49406; // BOS
        cond_ids[1] = 320; // "a"
        cond_ids[2] = 1125; // "photo"
        cond_ids[3] = 539; // "of"
        cond_ids[4] = 320; // "a"
        cond_ids[5] = 2368; // "cat"
        cond_ids[6] = 49407; // EOS
        let input = NumericTensor::<DynRank>::from_vec_shape(cond_ids, vec![1, seq_len]).unwrap();

        // EOS index = 6 (position of the EOS token)
        let eos_indices =
            NumericTensor::<DynRank>::from_vec_shape(vec![6i64], vec![1]).unwrap();

        let start = Instant::now();
        let out = te2
            .eval(
                std::collections::HashMap::from([
                    ("input_ids".to_string(), input),
                    ("eos_indices".to_string(), eos_indices),
                ]),
                &mut (),
                None,
                &mut backend,
            )
            .expect("text_encoder_2 eval failed");

        let hidden = out.get("penultimate_hidden_state").unwrap();
        println!(
            "  Penultimate: dtype={:?}, shape={:?}, took {:.2?}",
            hidden.dtype(),
            hidden.shape(),
            start.elapsed()
        );
        debug_tensor("  penultimate", hidden, &mut backend);

        let pooled = out.get("pooled_output").unwrap();
        println!(
            "  Pooled: dtype={:?}, shape={:?}",
            pooled.dtype(),
            pooled.shape()
        );
        debug_tensor("  pooled", pooled, &mut backend);
    }

    // --- Test UNet (single step) ---
    println!("\n=== Testing UNet (single step) ===");
    {
        let latent_h = 8;
        let latent_w = 8;

        let latent = NumericTensor::<DynRank>::from_vec_shape(
            vec![0.0f32; 4 * latent_h * latent_w],
            vec![1, 4, latent_h, latent_w],
        )
        .unwrap()
        .cast(model_dtype, &mut backend)
        .expect("cast latent");

        let timestep = NumericTensor::<DynRank>::from_vec_shape(vec![999.0f32], vec![1])
            .unwrap()
            .cast(model_dtype, &mut backend)
            .expect("cast timestep");

        // Context stays F32 (text encoder outputs are F32)
        let context = NumericTensor::<DynRank>::from_vec_shape(
            vec![0.0f32; 77 * 2048],
            vec![1, 77, 2048],
        )
        .unwrap();

        // ADM conditioning stays F32
        let y = NumericTensor::<DynRank>::from_vec_shape(
            vec![0.0f32; 2816],
            vec![1, 2816],
        )
        .unwrap();

        let start = Instant::now();
        let out = unet
            .eval(
                std::collections::HashMap::from([
                    ("sample".to_string(), latent),
                    ("timestep".to_string(), timestep),
                    ("encoder_hidden_states".to_string(), context),
                    ("y".to_string(), y),
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
        .unwrap()
        .cast(model_dtype, &mut backend)
        .expect("cast latent");

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
    let f32_tensor = tensor.cast(DType::F32, backend).expect("cast failed");
    let ndarray = f32_tensor.to_ndarray().expect("to_ndarray failed");
    let flat: Vec<f32> = ndarray.flatten().try_into().expect("flatten failed");
    let nan_count = flat.iter().filter(|v| v.is_nan()).count();
    let inf_count = flat.iter().filter(|v| v.is_infinite()).count();
    let min_val = flat.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("{name}: min={min_val:.4}, max={max_val:.4}, nan={nan_count}, inf={inf_count}");
}
