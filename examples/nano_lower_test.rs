//! Fast lowering + span-based partitioning diagnostic for ONNX models.
//!
//! Usage:
//!   cargo run --release --example nano_lower_test -- test_models/gpt2-lm-head-10.onnx

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use whisper_tensor::compiler::op_census;
use whisper_tensor::graph::GlobalId;
use whisper_tensor::model::Model;
use whisper_tensor::pool::SystemPool;
use whisper_tensor::tensor_info::TensorInfo;
use whisper_tensor_import::identify_and_load;
use whisper_tensor_import::onnx_graph::WeightStorageStrategy;

fn main() {
    let onnx_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "test_models/gpt2-lm-head-10.onnx".to_string());
    let path = Path::new(&onnx_path);

    if !path.exists() {
        eprintln!("Model file not found: {}", onnx_path);
        std::process::exit(1);
    }

    // ---- Load model ----
    let t0 = Instant::now();
    let onnx_data = identify_and_load(path, WeightStorageStrategy::EmbeddedData).unwrap();
    let mut rng = rand::rng();
    let model = Model::new_from_onnx(&onnx_data, &mut rng, None).unwrap();
    eprintln!("Loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let milli_graph = model.get_symbolic_graph().generate_milli_graph(&mut rng);
    eprintln!("MilliOpGraph: {:.1}ms", t0.elapsed().as_secs_f64() * 1e3);

    let census = op_census(&milli_graph);
    let total_ops: usize = census.iter().map(|(_, c)| c).sum();
    println!("=== Milli Ops ({} total) ===", total_ops);
    for (kind, count) in &census {
        println!("  {:>4}x  {}", count, kind);
    }

    // ---- Build tensor info ----
    let input_info = model.get_input_tensor_info().unwrap();
    let sym_graph = model.get_symbolic_graph();
    let tensor_store = model.get_tensor_store();
    let tensors_by_name = sym_graph.get_tensors_by_name();

    let mut all_infos: HashMap<GlobalId, TensorInfo<'_, whisper_tensor::pool::SystemPool>> =
        HashMap::new();
    for (name, (dtype, shape_dims)) in &input_info {
        let shape: Vec<u64> = shape_dims.iter().map(|d| d.unwrap_or(4)).collect();
        let ndt = whisper_tensor::numeric_dtype::NumericDType::from_legacy(*dtype)
            .expect("unsupported input dtype");
        let info = TensorInfo::from_dtype_and_shape(ndt, &shape);
        println!("Input '{}': {:?} {:?}", name, dtype, shape);
        if let Some(id) = tensors_by_name.get(name) {
            all_infos.insert(*id, info);
        }
    }
    let initialized = sym_graph.get_initialized_tensors(tensor_store);
    let mut n_full = 0usize;
    let mut n_shape_only = 0usize;
    for (id, tensor) in &initialized {
        if tensor.num_elements() <= 1024 {
            // Small constants (axes, indices, shape values): keep full data
            // so infer_all can resolve Shape/Gather/Reshape ops.
            all_infos.insert(*id, TensorInfo::from_legacy(tensor, &SystemPool));
            n_full += 1;
        } else {
            // Large weight matrices: shape+dtype only.
            let shape: Vec<u64> = tensor.shape().to_vec();
            let dtype = whisper_tensor::numeric_dtype::NumericDType::from_legacy(tensor.dtype())
                .expect("unsupported weight dtype");
            all_infos.insert(*id, TensorInfo::from_dtype_and_shape(dtype, &shape));
            n_shape_only += 1;
        }
    }
    println!(
        "Tensor info: {} full (small constants), {} shape-only (weights)",
        n_full, n_shape_only
    );

    // ---- Lower ----
    let t0 = Instant::now();
    let result = whisper_tensor::nano_graph::lower::lower(&milli_graph, &all_infos).unwrap();
    eprintln!("lower: {:.1}s", t0.elapsed().as_secs_f64());

    let stats = result.graph.stats();
    println!("\n=== NanoGraph ===\n{}", stats);

    if !result.unsupported.is_empty() {
        println!("\nUnsupported ({}):", result.unsupported.len());
        let mut counts: HashMap<String, usize> = HashMap::new();
        for (_, kind) in &result.unsupported {
            *counts.entry(kind.clone()).or_default() += 1;
        }
        let mut sorted: Vec<_> = counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        for (kind, count) in sorted {
            println!("  {:>4}x  {}", count, kind);
        }
    }
}
