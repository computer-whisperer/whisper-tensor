//! Lower a real ONNX model (GPT-2) to a NanoGraph and report stats.
//!
//! Usage:
//!   cargo run --example nano_graph_model_test -- test_models/gpt2-lm-head-10.onnx

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::compiler::op_census;
use whisper_tensor::dtype::DType;
use whisper_tensor::graph::GlobalId;
use whisper_tensor::model::Model;
use whisper_tensor::nano_graph::lower;
use whisper_tensor::nano_graph::optimize;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::tensor_rank::DynRank;
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
    println!("Loading model from {}...", onnx_path);
    let t0 = Instant::now();
    let onnx_data = identify_and_load(path, WeightStorageStrategy::EmbeddedData).unwrap();
    let mut rng = rand::rng();
    let model = Model::new_from_onnx(&onnx_data, &mut rng, None).unwrap();
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f64());

    // ---- Generate MilliOpGraph ----
    println!("\nGenerating MilliOpGraph...");
    let t0 = Instant::now();
    let milli_graph = model.get_symbolic_graph().generate_milli_graph(&mut rng);
    println!("  Generated in {:.1}ms", t0.elapsed().as_secs_f64() * 1e3);

    // ---- Op census ----
    let census = op_census(&milli_graph);
    let total_ops: usize = census.iter().map(|(_, c)| c).sum();
    println!("\n=== Op Census ({} ops total) ===", total_ops);
    for (kind, count) in &census {
        println!("  {:>4}x  {}", count, kind);
    }

    // ---- Create dummy inputs ----
    let input_info = model.get_input_tensor_info().unwrap();
    let sym_graph = model.get_symbolic_graph();
    let tensor_store = model.get_tensor_store();
    let tensors_by_name = sym_graph.get_tensors_by_name();

    let mut inputs_by_name: HashMap<String, NumericTensor<DynRank>> = HashMap::new();
    for (name, (dtype, shape_dims)) in &input_info {
        let concrete_shape: Vec<usize> = shape_dims
            .iter()
            .map(|d| d.map(|v| v as usize).unwrap_or(4))
            .collect();
        let numel: usize = concrete_shape.iter().product();
        let shape_u64: Vec<u64> = concrete_shape.iter().map(|&d| d as u64).collect();
        let tensor: NumericTensor<DynRank> = match dtype {
            DType::I64 => {
                let data: Vec<i64> = (0..numel as i64).collect();
                NDArrayNumericTensor::<DynRank>::from_vec_shape(data, &shape_u64)
                    .unwrap()
                    .into()
            }
            DType::I32 => {
                let data: Vec<i32> = (0..numel as i32).collect();
                NDArrayNumericTensor::<DynRank>::from_vec_shape(data, &shape_u64)
                    .unwrap()
                    .into()
            }
            DType::F32 => {
                let data: Vec<f32> = vec![0.0; numel];
                NDArrayNumericTensor::<DynRank>::from_vec_shape(data, &shape_u64)
                    .unwrap()
                    .into()
            }
            DType::F64 => {
                let data: Vec<f64> = vec![0.0; numel];
                NDArrayNumericTensor::<DynRank>::from_vec_shape(data, &shape_u64)
                    .unwrap()
                    .into()
            }
            DType::BF16 => {
                let data: Vec<half::bf16> = vec![half::bf16::ZERO; numel];
                NDArrayNumericTensor::<DynRank>::from_vec_shape(data, &shape_u64)
                    .unwrap()
                    .into()
            }
            DType::F16 => {
                let data: Vec<half::f16> = vec![half::f16::ZERO; numel];
                NDArrayNumericTensor::<DynRank>::from_vec_shape(data, &shape_u64)
                    .unwrap()
                    .into()
            }
            _ => {
                println!("  Skipping input '{}' with dtype {:?}", name, dtype);
                continue;
            }
        };
        println!("  Input '{}': {:?} {:?}", name, dtype, tensor.shape());
        inputs_by_name.insert(name.clone(), tensor);
    }

    // Map inputs to GlobalIds and add weights.
    let milli_inputs: HashMap<GlobalId, NumericTensor<DynRank>> = inputs_by_name
        .into_iter()
        .filter_map(|(name, tensor)| tensors_by_name.get(&name).map(|id| (*id, tensor)))
        .collect();

    let initialized = sym_graph.get_initialized_tensors(tensor_store);
    let mut all_inputs = milli_inputs;
    for (id, tensor) in initialized {
        all_inputs.insert(id, tensor);
    }
    println!("  Total input tensors: {}", all_inputs.len());

    // ---- Lower to NanoGraph ----
    println!("\nLowering to NanoGraph...");
    let t0 = Instant::now();
    let result = lower::lower(&milli_graph, &all_inputs).unwrap();
    let elapsed = t0.elapsed();
    println!("  Lowered in {:.1}ms", elapsed.as_secs_f64() * 1e3);

    // ---- Report ----
    let stats = result.graph.stats();
    println!("\n=== NanoGraph Stats ===");
    println!("  {}", stats);

    if !result.unsupported.is_empty() {
        println!("\n=== Unsupported Ops ({}) ===", result.unsupported.len());
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

    // Report compression ratio.
    let block_count = result.graph.len();
    let concrete = stats.total_concrete_ops;
    if concrete > 0 {
        let block_bytes = block_count * std::mem::size_of::<whisper_tensor::nano_graph::PatternBlock>();
        println!(
            "\n=== Compression ===");
        println!(
            "  {} blocks representing {} scalar ops",
            block_count, concrete
        );
        println!(
            "  Approx block storage: {} bytes ({:.1} KB)",
            block_bytes,
            block_bytes as f64 / 1024.0
        );
        println!(
            "  Compression ratio: {:.0}x (vs naive {} bytes/op)",
            concrete as f64 / block_count as f64,
            std::mem::size_of::<u64>() * 4 // rough: op + 2 inputs + output
        );
    }

    println!("\n  Output blocks: {}", result.graph.output_blocks.len());

    let errors = result.graph.validate();
    if errors.is_empty() {
        println!("  Validation: PASSED");
    } else {
        println!("  Validation: {} ERRORS", errors.len());
        for e in errors.iter().take(10) {
            println!("    {}", e);
        }
    }

    // ---- Optimization passes ----
    println!("\n=== Optimization Passes ===");
    let mut graph = result.graph;

    let t0 = Instant::now();
    let sr = optimize::strength_reduce(graph);
    println!(
        "  Strength reduction: {} blocks simplified in {:.1}ms",
        sr.eliminated,
        t0.elapsed().as_secs_f64() * 1e3
    );
    graph = sr.graph;

    let t0 = Instant::now();
    let cse_result = optimize::cse(graph);
    println!(
        "  CSE: {} blocks deduplicated in {:.1}ms",
        cse_result.eliminated,
        t0.elapsed().as_secs_f64() * 1e3
    );
    graph = cse_result.graph;

    let t0 = Instant::now();
    let dce_result = optimize::dce(graph);
    println!(
        "  DCE: {} dead blocks removed in {:.1}ms",
        dce_result.eliminated,
        t0.elapsed().as_secs_f64() * 1e3
    );
    graph = dce_result.graph;

    let opt_stats = graph.stats();
    println!("\n=== After Optimization ===");
    println!("  {}", opt_stats);

    let opt_errors = graph.validate();
    if opt_errors.is_empty() {
        println!("  Validation: PASSED");
    } else {
        println!("  Validation: {} ERRORS", opt_errors.len());
        for e in opt_errors.iter().take(10) {
            println!("    {}", e);
        }
    }
}
