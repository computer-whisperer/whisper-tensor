use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;
use whisper_tensor::DynRank;
use whisper_tensor::numeric_dtype::NumericDType;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::pool::SystemPool;
use whisper_tensor_import::identify_and_load;
use whisper_tensor_import::onnx_graph::WeightStorageStrategy;

const SD_BASE: &str = "/mnt/secondary/neural_networks/stable-diffusion-1.5-onnx-fp16";
const REF_DIR: &str = "/tmp/sd_reference";

fn compare<'a>(
    name: &str,
    actual: &NumericTensor<'a, DynRank, SystemPool>,
    ref_name: &str,
    pool: &SystemPool,
) {
    let ref_path = format!("{REF_DIR}/{ref_name}");
    let ref_tensor =
        whisper_tensor::npy::read_npy_file(Path::new(&ref_path), pool).expect("read ref npy");
    let ref_shape: Vec<usize> = ref_tensor.shape().iter().map(|&s| s as usize).collect();

    let actual_shape: Vec<usize> = actual.shape().iter().map(|&s| s as usize).collect();
    assert_eq!(
        actual_shape, ref_shape,
        "{name}: shape mismatch: actual={actual_shape:?} vs ref={ref_shape:?}"
    );

    let numel = actual.numel();
    assert_eq!(numel, ref_tensor.numel(), "{name}: element count mismatch");

    let mut max_abs_diff: f32 = 0.0;
    let mut max_rel_diff: f32 = 0.0;
    let mut num_mismatches = 0;
    let atol: f32 = 1e-2;
    let rtol: f32 = 5e-2;

    for i in 0..numel {
        let a = actual.read_element(i).to_f32();
        let r = ref_tensor.read_element(i).to_f32();
        if a.is_nan() && r.is_nan() {
            continue;
        }
        let abs_diff = (a - r).abs();
        let rel_diff = if r.abs() > 1e-6 {
            abs_diff / r.abs()
        } else {
            abs_diff
        };
        max_abs_diff = max_abs_diff.max(abs_diff);
        max_rel_diff = max_rel_diff.max(rel_diff);
        if abs_diff > atol + rtol * r.abs() {
            if num_mismatches < 5 {
                println!("  MISMATCH at [{i}]: actual={a:.6}, ref={r:.6}, abs_diff={abs_diff:.6}");
            }
            num_mismatches += 1;
        }
    }

    let total = numel;
    let pass_pct = 100.0 * (total - num_mismatches) as f64 / total as f64;
    println!(
        "  {name}: max_abs_diff={max_abs_diff:.6}, max_rel_diff={max_rel_diff:.6}, \
         mismatches={num_mismatches}/{total} ({pass_pct:.2}% pass)"
    );
    if num_mismatches > 0 {
        println!(
            "  WARNING: {num_mismatches} elements exceed tolerance (atol={atol}, rtol={rtol})"
        );
    } else {
        println!("  PASS");
    }
}

fn main() {
    tracing_subscriber::fmt::init();
    let pool = SystemPool;

    let input_path = Path::new(SD_BASE).join("text_encoder").join("model.onnx");
    println!("Loading text_encoder from {}", input_path.display());

    let onnx_data = identify_and_load(&input_path, WeightStorageStrategy::EmbeddedData)
        .expect("Failed to import model");
    let mut rng = rand::rng();
    let model =
        whisper_tensor::model::Model::new_from_onnx(&onnx_data, &mut rng, input_path.parent())
            .expect("Failed to load model");

    // --- Conditional ---
    println!("\n=== Conditional encoding ===");
    let input_tensor = whisper_tensor::npy::read_npy_file(
        Path::new(&format!("{REF_DIR}/text_encoder_input_ids_int32.npy")),
        &pool,
    )
    .expect("read input npy");

    let input_view = input_tensor.view();
    let mut inputs = HashMap::new();
    inputs.insert("input_ids".to_string(), &input_view);

    let start = Instant::now();
    let outputs = model.eval_pool(inputs, &pool).expect("Inference failed");
    println!("  Inference took {:.2?}", start.elapsed());

    for (name, tensor) in &outputs {
        println!(
            "  Output {name}: dtype={:?}, shape={:?}",
            tensor.dtype(),
            tensor.shape()
        );
    }

    if let Some(hidden) = outputs.get("last_hidden_state") {
        compare(
            "last_hidden_state",
            hidden,
            "text_encoder_last_hidden_state_float16.npy",
            &pool,
        );
    }
    if let Some(pooler) = outputs.get("pooler_output") {
        compare(
            "pooler_output",
            pooler,
            "text_encoder_pooler_output_float16.npy",
            &pool,
        );
    }

    // --- Unconditional ---
    println!("\n=== Unconditional encoding ===");
    let input_tensor = whisper_tensor::npy::read_npy_file(
        Path::new(&format!(
            "{REF_DIR}/text_encoder_uncond_input_ids_int32.npy"
        )),
        &pool,
    )
    .expect("read uncond input npy");

    let input_view = input_tensor.view();
    let mut inputs = HashMap::new();
    inputs.insert("input_ids".to_string(), &input_view);

    let start = Instant::now();
    let outputs = model.eval_pool(inputs, &pool).expect("Inference failed");
    println!("  Inference took {:.2?}", start.elapsed());

    if let Some(hidden) = outputs.get("last_hidden_state") {
        compare(
            "last_hidden_state",
            hidden,
            "text_encoder_uncond_last_hidden_state_float16.npy",
            &pool,
        );
    }
    if let Some(pooler) = outputs.get("pooler_output") {
        compare(
            "pooler_output",
            pooler,
            "text_encoder_uncond_pooler_output_float16.npy",
            &pool,
        );
    }
}
