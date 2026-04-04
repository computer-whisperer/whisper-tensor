use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use whisper_tensor::DynRank;
use whisper_tensor::model::Model;
use whisper_tensor::numeric_dtype::NumericDType;
use whisper_tensor::numeric_scalar::NumericScalar;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::pool::SystemPool;
use whisper_tensor_import::models::diffusion::t5::{T5Config, load_t5_encoder_with_origin};
use whisper_tensor_import::onnx_graph::WeightStorageStrategy;
use whisper_tensor_import::onnx_graph::weights::SafetensorsWeightManager;

const T5_PATH: &str = "/ceph/public/neural_models/comfyui/clip/t5xxl_fp16.safetensors";

fn main() {
    tracing_subscriber::fmt::init();

    let total_start = Instant::now();
    let t5_path = Path::new(T5_PATH);
    let pool = SystemPool;

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

    // Test: encode a short sequence (pad to 256 with 0 = <pad>)
    // T5 tokenizer: "a photo of a cat" -> some token IDs
    // Using SentencePiece IDs for T5: these are approximate/placeholder
    let seq_len = 256;
    let mut input_ids_data = vec![0i32; seq_len];
    // "a photo of a cat" in T5 SentencePiece vocab (approximate IDs)
    input_ids_data[0] = 3; // "a"
    input_ids_data[1] = 1246; // "photo"
    input_ids_data[2] = 13; // "of"
    input_ids_data[3] = 3; // "a"
    input_ids_data[4] = 1712; // "cat"
    input_ids_data[5] = 1; // </s> (EOS)

    let input = NumericTensor::<DynRank, _>::from_fn(
        vec![1, seq_len as u64],
        NumericDType::I32,
        &pool,
        |i| NumericScalar::from_i32(input_ids_data[i]),
    )
    .unwrap();

    println!("\n=== Running T5 encoder ===");
    let start = Instant::now();
    let output = model
        .eval_pool(
            std::collections::HashMap::from([("input_ids".to_string(), &input.view())]),
            &pool,
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
    let view = hidden.view();
    let numel = view.numel();
    let mut nan_count = 0usize;
    let mut inf_count = 0usize;
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    for i in 0..numel {
        let v = view.read_element(i).to_f64();
        if v.is_nan() {
            nan_count += 1;
        }
        if v.is_infinite() {
            inf_count += 1;
        }
        min_val = min_val.min(v);
        max_val = max_val.max(v);
    }
    println!(
        "  hidden_states: min={min_val:.4}, max={max_val:.4}, nan={nan_count}, inf={inf_count}"
    );

    // Check first few values at position 0
    let first_8: Vec<f64> = (0..8).map(|i| view.read_element(i).to_f64()).collect();
    println!("  First 8 values: {:?}", first_8);

    println!("\n=== Complete in {:.2?} ===", total_start.elapsed());
}
