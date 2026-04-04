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

fn load_npy_as_f16_tensor<'a>(
    name: &str,
    pool: &'a SystemPool,
) -> NumericTensor<'a, DynRank, SystemPool> {
    let (values, shape) = load_npy_f32(name);
    let shape_u64: Vec<u64> = shape.iter().map(|&s| s as u64).collect();
    NumericTensor::from_fn(shape_u64, NumericDType::F16, pool, |i| {
        NumericScalar::from_f32(values[i]).cast_to(NumericDType::F16)
    })
    .unwrap()
}

fn compare(
    name: &str,
    actual: &NumericTensor<'_, DynRank, SystemPool>,
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

    let input_path = Path::new(SD_BASE).join("unet").join("model.onnx");
    println!("Loading unet from {}", input_path.display());

    let onnx_data = identify_and_load(&input_path, WeightStorageStrategy::EmbeddedData)
        .expect("Failed to import model");
    let mut rng = rand::rng();
    let model = whisper_tensor::model::Model::new_from_onnx(&onnx_data, &mut rng, input_path.parent())
        .expect("Failed to load model");

    println!("\n=== UNet single step ===");
    let sample = load_npy_as_f16_tensor("unet_sample_float16.npy", &pool);
    let timestep = load_npy_as_f16_tensor("unet_timestep_float16.npy", &pool);
    let encoder_hidden_states = load_npy_as_f16_tensor("unet_encoder_hidden_states_float16.npy", &pool);

    println!("  sample: {:?} {:?}", sample.dtype(), sample.shape());
    println!("  timestep: {:?} {:?}", timestep.dtype(), timestep.shape());
    println!(
        "  encoder_hidden_states: {:?} {:?}",
        encoder_hidden_states.dtype(),
        encoder_hidden_states.shape()
    );

    let sample_view = sample.view();
    let timestep_view = timestep.view();
    let ehs_view = encoder_hidden_states.view();

    let mut inputs = HashMap::new();
    inputs.insert("sample".to_string(), &sample_view);
    inputs.insert("timestep".to_string(), &timestep_view);
    inputs.insert("encoder_hidden_states".to_string(), &ehs_view);

    let start = Instant::now();
    let outputs = model
        .eval_pool(inputs, &pool)
        .expect("Inference failed");
    println!("  Inference took {:.2?}", start.elapsed());

    if let Some(out) = outputs.get("out_sample") {
        compare(
            "out_sample",
            out,
            "unet_out_sample_float16.npy",
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
