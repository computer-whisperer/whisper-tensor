use ndarray::ArrayD;
use ndarray_npy::ReadNpyExt;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::time::Instant;
use whisper_tensor::DynRank;
use whisper_tensor::numeric_dtype::NumericDType;
use whisper_tensor::numeric_scalar::NumericScalar;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::pool::SystemPool;
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

fn compare<'a>(
    name: &str,
    actual: &NumericTensor<'a, DynRank, SystemPool>,
    ref_name: &str,
    pool: &SystemPool,
) {
    let (ref_values, ref_shape) = load_npy_f32(ref_name);

    let actual_shape: Vec<usize> = actual.shape().iter().map(|&s| s as usize).collect();
    assert_eq!(
        actual_shape, ref_shape,
        "{name}: shape mismatch: actual={actual_shape:?} vs ref={ref_shape:?}"
    );

    let actual_f32 = cast_tensor(actual, NumericDType::F32, pool);
    let actual_flat: Vec<f32> = (0..actual_f32.numel())
        .map(|i| actual_f32.read_element(i).to_f32())
        .collect();

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
    let pool = SystemPool;

    let input_path = Path::new(SD_BASE).join("text_encoder").join("model.onnx");
    println!("Loading text_encoder from {}", input_path.display());

    let onnx_data = identify_and_load(&input_path, WeightStorageStrategy::EmbeddedData)
        .expect("Failed to import model");
    let mut rng = rand::rng();
    let model = whisper_tensor::model::Model::new_from_onnx(&onnx_data, &mut rng, input_path.parent())
        .expect("Failed to load model");

    // --- Conditional ---
    println!("\n=== Conditional encoding ===");
    let (input_f32, input_shape) = load_npy_f32("text_encoder_input_ids_int32.npy");
    let input_i32: Vec<i32> = input_f32.iter().map(|&x| x as i32).collect();
    let shape_u64: Vec<u64> = input_shape.iter().map(|&s| s as u64).collect();
    let input_tensor = NumericTensor::from_fn(shape_u64, NumericDType::I32, &pool, |i| {
        NumericScalar::from_i32(input_i32[i])
    })
    .unwrap();

    let input_view = input_tensor.view();
    let mut inputs = HashMap::new();
    inputs.insert("input_ids".to_string(), &input_view);

    let start = Instant::now();
    let outputs = model
        .eval_pool(inputs, &pool)
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
    let (input_f32, input_shape) = load_npy_f32("text_encoder_uncond_input_ids_int32.npy");
    let input_i32: Vec<i32> = input_f32.iter().map(|&x| x as i32).collect();
    let shape_u64: Vec<u64> = input_shape.iter().map(|&s| s as u64).collect();
    let input_tensor = NumericTensor::from_fn(shape_u64, NumericDType::I32, &pool, |i| {
        NumericScalar::from_i32(input_i32[i])
    })
    .unwrap();

    let input_view = input_tensor.view();
    let mut inputs = HashMap::new();
    inputs.insert("input_ids".to_string(), &input_view);

    let start = Instant::now();
    let outputs = model
        .eval_pool(inputs, &pool)
        .expect("Inference failed");
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
