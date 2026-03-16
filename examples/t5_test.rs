use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use whisper_tensor::DynRank;
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::dtype::DType;
use whisper_tensor::model::Model;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor_import::models::diffusion::t5::{T5Config, load_t5_encoder_with_origin};
use whisper_tensor_import::onnx_graph::WeightStorageStrategy;
use whisper_tensor_import::onnx_graph::weights::SafetensorsWeightManager;

const T5_PATH: &str = "/ceph/public/neural_models/comfyui/clip/t5xxl_fp16.safetensors";

fn main() {
    tracing_subscriber::fmt::init();

    let total_start = Instant::now();
    let t5_path = Path::new(T5_PATH);

    // Load weights
    println!("=== Building T5-XXL encoder ===");
    let start = Instant::now();

    let file = std::fs::File::open(t5_path).expect("open T5 weights");
    let mmap = unsafe { memmap2::Mmap::map(&file) }.expect("mmap");
    let wm = SafetensorsWeightManager::new(vec![Arc::new(mmap)]).expect("weight manager");

    let config = T5Config::t5_xxl(256);
    let onnx_data = load_t5_encoder_with_origin(
        wm,
        config,
        WeightStorageStrategy::OriginReference,
        Some(t5_path),
    )
    .expect("T5 build failed");

    println!(
        "  Built in {:.2?} ({:.1}MB ONNX)",
        start.elapsed(),
        onnx_data.len() as f64 / 1e6
    );

    // Load model
    let start = Instant::now();
    let mut rng = rand::rng();
    let model = Model::new_from_onnx(&onnx_data, &mut rng, Some(t5_path.parent().unwrap()))
        .expect("model load failed");
    println!("  Model loaded in {:.2?}", start.elapsed());

    let mut backend = EvalBackend::NDArray;

    // Test: encode a short sequence (pad to 256 with 0 = <pad>)
    // T5 tokenizer: "a photo of a cat" → some token IDs
    // Using SentencePiece IDs for T5: these are approximate/placeholder
    let seq_len = 256;
    let mut input_ids = vec![0i32; seq_len];
    // "a photo of a cat" in T5 SentencePiece vocab (approximate IDs)
    input_ids[0] = 3; // "a"
    input_ids[1] = 1246; // "photo"
    input_ids[2] = 13; // "of"
    input_ids[3] = 3; // "a"
    input_ids[4] = 1712; // "cat"
    input_ids[5] = 1; // </s> (EOS)

    let input = NumericTensor::<DynRank>::from_vec_shape(input_ids, vec![1, seq_len]).unwrap();

    println!("\n=== Running T5 encoder ===");
    let start = Instant::now();
    let output = model
        .eval(
            std::collections::HashMap::from([("input_ids".to_string(), input)]),
            &mut (),
            None,
            &mut backend,
        )
        .expect("T5 eval failed");

    let hidden = output.get("hidden_states").unwrap();
    println!(
        "  Output: dtype={:?}, shape={:?}, took {:.2?}",
        hidden.dtype(),
        hidden.shape(),
        start.elapsed(),
    );

    // Check values
    let f32_tensor = hidden.cast(DType::F32, &mut backend).expect("cast failed");
    let ndarray = f32_tensor.to_ndarray().expect("to_ndarray failed");
    let flat: Vec<f32> = ndarray.flatten().try_into().expect("flatten failed");
    let nan_count = flat.iter().filter(|v| v.is_nan()).count();
    let inf_count = flat.iter().filter(|v| v.is_infinite()).count();
    let min_val = flat.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "  hidden_states: min={min_val:.4}, max={max_val:.4}, nan={nan_count}, inf={inf_count}"
    );

    // Check first few values at position 0
    let first_8: Vec<f32> = flat[..8].to_vec();
    println!("  First 8 values: {:?}", first_8);

    println!("\n=== Complete in {:.2?} ===", total_start.elapsed());
}
