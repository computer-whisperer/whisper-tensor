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

fn load_npy_as_f16_tensor<'a>(
    name: &str,
    pool: &'a SystemPool,
) -> NumericTensor<'a, DynRank, SystemPool> {
    let path = format!("{REF_DIR}/{name}");
    let f32_tensor = whisper_tensor::npy::read_npy_file(Path::new(&path), pool).expect("read npy");
    NumericTensor::from_fn(f32_tensor.shape().clone(), NumericDType::F16, pool, |i| {
        f32_tensor.read_element(i).cast_to(NumericDType::F16)
    })
    .unwrap()
}

fn compare(
    name: &str,
    actual: &NumericTensor<'_, DynRank, SystemPool>,
    ref_name: &str,
    pool: &SystemPool,
) {
    let ref_path = format!("{REF_DIR}/{ref_name}");
    let ref_tensor = whisper_tensor::npy::read_npy_file(std::path::Path::new(&ref_path), pool)
        .expect("read ref npy");

    assert_eq!(
        actual.shape(),
        ref_tensor.shape(),
        "{name}: shape mismatch: actual={:?} vs ref={:?}",
        actual.shape(),
        ref_tensor.shape()
    );

    let numel = actual.numel();
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

    let input_path = Path::new(SD_BASE).join("unet").join("model.onnx");
    println!("Loading unet from {}", input_path.display());

    let onnx_data = identify_and_load(&input_path, WeightStorageStrategy::EmbeddedData)
        .expect("Failed to import model");
    let mut rng = rand::rng();
    let model =
        whisper_tensor::model::Model::new_from_onnx(&onnx_data, &mut rng, input_path.parent())
            .expect("Failed to load model");

    println!("\n=== UNet single step ===");
    let sample = load_npy_as_f16_tensor("unet_sample_float16.npy", &pool);
    let timestep = load_npy_as_f16_tensor("unet_timestep_float16.npy", &pool);
    let encoder_hidden_states =
        load_npy_as_f16_tensor("unet_encoder_hidden_states_float16.npy", &pool);

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
    let outputs = model.eval_pool(inputs, &pool).expect("Inference failed");
    println!("  Inference took {:.2?}", start.elapsed());

    if let Some(out) = outputs.get("out_sample") {
        compare("out_sample", out, "unet_out_sample_float16.npy", &pool);
    }
}
