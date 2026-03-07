use ndarray::ArrayD;
use ndarray_npy::ReadNpyExt;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::time::Instant;
use whisper_tensor::DynRank;
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::dtype::DType;
use whisper_tensor::model::Model;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor_import::identify_and_load;
use whisper_tensor_import::onnx_graph::WeightStorageStrategy;

const SD_BASE: &str = "/mnt/secondary/neural_networks/stable-diffusion-1.5-onnx-fp16";
const REF_DIR: &str = "/tmp/sd_reference";

fn load_npy_f32(name: &str) -> (Vec<f32>, Vec<usize>) {
    let path = format!("{REF_DIR}/{name}");
    let reader = File::open(&path).unwrap_or_else(|e| panic!("Cannot open {path}: {e}"));
    let arr = ArrayD::<f32>::read_npy(reader).unwrap();
    let shape = arr.shape().to_vec();
    let values = arr.into_raw_vec_and_offset().0;
    (values, shape)
}

fn compare(name: &str, actual: &NumericTensor<DynRank>, ref_name: &str, backend: &mut EvalBackend) {
    let (ref_values, ref_shape) = load_npy_f32(ref_name);

    let actual_shape: Vec<usize> = actual.shape().iter().map(|&s| s as usize).collect();
    assert_eq!(
        actual_shape, ref_shape,
        "{name}: shape mismatch: actual={actual_shape:?} vs ref={ref_shape:?}"
    );

    let actual_f32 = actual.cast(DType::F32, backend).unwrap();
    let actual_nd = actual_f32.to_ndarray().unwrap();
    let actual_flat: Vec<f32> = actual_nd.flatten().try_into().unwrap();

    assert_eq!(
        actual_flat.len(),
        ref_values.len(),
        "{name}: element count mismatch"
    );

    let mut max_abs_diff: f32 = 0.0;
    let mut max_rel_diff: f32 = 0.0;
    let mut num_mismatches = 0;
    let atol: f32 = 1e-2;
    let rtol: f32 = 5e-2;

    for (i, (&a, &r)) in actual_flat.iter().zip(ref_values.iter()).enumerate() {
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

    let total = actual_flat.len();
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

    let input_path = Path::new(SD_BASE).join("text_encoder").join("model.onnx");
    println!("Loading text_encoder from {}", input_path.display());

    let onnx_data = identify_and_load(&input_path, WeightStorageStrategy::EmbeddedData, None)
        .expect("Failed to import model");
    let mut rng = rand::rng();
    let model = Model::new_from_onnx(&onnx_data, &mut rng, input_path.parent())
        .expect("Failed to load model");

    let mut backend = EvalBackend::NDArray;

    // --- Conditional ---
    println!("\n=== Conditional encoding ===");
    let (input_f32, input_shape) = load_npy_f32("text_encoder_input_ids_int32.npy");
    let input_i32: Vec<i32> = input_f32.iter().map(|&x| x as i32).collect();
    let input_tensor = NumericTensor::<DynRank>::from_vec_shape(input_i32, input_shape).unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("input_ids".to_string(), input_tensor);

    let start = Instant::now();
    let outputs = model
        .eval(inputs, &mut (), None, &mut backend)
        .expect("Inference failed");
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
            &mut backend,
        );
    }
    if let Some(pooler) = outputs.get("pooler_output") {
        compare(
            "pooler_output",
            pooler,
            "text_encoder_pooler_output_float16.npy",
            &mut backend,
        );
    }

    // --- Unconditional ---
    println!("\n=== Unconditional encoding ===");
    let (input_f32, input_shape) = load_npy_f32("text_encoder_uncond_input_ids_int32.npy");
    let input_i32: Vec<i32> = input_f32.iter().map(|&x| x as i32).collect();
    let input_tensor = NumericTensor::<DynRank>::from_vec_shape(input_i32, input_shape).unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("input_ids".to_string(), input_tensor);

    let start = Instant::now();
    let outputs = model
        .eval(inputs, &mut (), None, &mut backend)
        .expect("Inference failed");
    println!("  Inference took {:.2?}", start.elapsed());

    if let Some(hidden) = outputs.get("last_hidden_state") {
        compare(
            "last_hidden_state",
            hidden,
            "text_encoder_uncond_last_hidden_state_float16.npy",
            &mut backend,
        );
    }
    if let Some(pooler) = outputs.get("pooler_output") {
        compare(
            "pooler_output",
            pooler,
            "text_encoder_uncond_pooler_output_float16.npy",
            &mut backend,
        );
    }
}
