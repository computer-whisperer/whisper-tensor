//! Attempt to compile a real ONNX model (GPT-2) through the v8 generic kernel
//! compiler.
//!
//! Usage:
//!   cargo run --example v8_model_test --features cranelift -- gpt2-lm-head-10.onnx
//!
//! This example:
//! 1. Loads the ONNX model
//! 2. Generates the combined MilliOpGraph
//! 3. Reports an op census (which ops, how many, v8-compatible or not)
//! 4. Runs the graph through the interpreter to collect all tensor shapes
//! 5. Attempts v8 compilation and reports the result

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::compiler::attempts::v8_generic_kernel::codegen as v8_codegen;
use whisper_tensor::compiler::op_census;
use whisper_tensor::dtype::DType;
use whisper_tensor::graph::GlobalId;
use whisper_tensor::model::Model;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::tensor_rank::DynRank;
use whisper_tensor_import::identify_and_load;
use whisper_tensor_import::onnx_graph::WeightStorageStrategy;

/// Ops that the v8 nano-op expander can handle.
const V8_SUPPORTED_OPS: &[&str] = &[
    "Constant",
    "ConstantOfShape",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Max",
    "Min",
    "Neg",
    "Abs",
    "Exp",
    "Ln",
    "Sqrt",
    "Reciprocal",
    "Tanh",
    "Floor",
    "Ceil",
    "MatMul",
];

fn main() {
    tracing_subscriber::fmt::init();

    let onnx_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "gpt2-lm-head-10.onnx".to_string());
    let path = Path::new(&onnx_path);

    if !path.exists() {
        eprintln!("Model file not found: {}", onnx_path);
        eprintln!("Download GPT-2 ONNX model: https://github.com/onnx/models");
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
    let mut supported_ops = 0;
    let mut unsupported_ops = 0;

    println!("\n=== Op Census ({} ops total) ===", total_ops);
    for (kind, count) in &census {
        let supported = V8_SUPPORTED_OPS.contains(&kind.as_str());
        let marker = if supported { "  OK" } else { "  XX" };
        println!("  {:>4}x  {:<20} {}", count, kind, marker);
        if supported {
            supported_ops += count;
        } else {
            unsupported_ops += count;
        }
    }
    println!(
        "\n  v8-compatible: {}/{} ({:.0}%)",
        supported_ops,
        total_ops,
        100.0 * supported_ops as f64 / total_ops as f64
    );
    println!("  unsupported:   {}/{}", unsupported_ops, total_ops);

    // ---- Collect shapes via interpreter run ----
    println!("\nRunning interpreter to collect tensor shapes...");

    // Get model input info to create dummy inputs
    let input_info = model.get_input_tensor_info().unwrap();
    println!("  Model inputs:");
    for (name, (dtype, shape)) in &input_info {
        println!("    {}: {:?} {:?}", name, dtype, shape);
    }

    // Create dummy inputs (GPT-2 expects input1: [batch, seq_len] of i64)
    let sym_graph = model.get_symbolic_graph();
    let tensor_store = model.get_tensor_store();
    let mut inputs_by_name: HashMap<String, NumericTensor<DynRank>> = HashMap::new();

    for (name, (dtype, shape_dims)) in &input_info {
        // Replace dynamic dims (None) with small concrete values
        let concrete_shape: Vec<usize> = shape_dims
            .iter()
            .map(|d| d.map(|v| v as usize).unwrap_or(4)) // dynamic dims → 4
            .collect();
        let numel: usize = concrete_shape.iter().product();

        let shape_u64: Vec<u64> = concrete_shape.iter().map(|&d| d as u64).collect();
        let tensor: NumericTensor<DynRank> = match dtype {
            DType::I64 => {
                // Token IDs: use small integers
                let data: Vec<i64> = (0..numel as i64).collect();
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
            _ => {
                println!(
                    "  Skipping input '{}' with unsupported dtype {:?}",
                    name, dtype
                );
                continue;
            }
        };
        println!("  Created dummy input '{}': {:?}", name, tensor.shape());
        inputs_by_name.insert(name.clone(), tensor);
    }

    // Run at symbolic graph level first to validate
    println!("\n  Running symbolic graph through interpreter...");
    let t0 = Instant::now();
    let sym_result = whisper_tensor::backends::eval_backend::run(
        sym_graph,
        tensor_store,
        None,
        &mut EvalBackend::NDArray,
        &mut (),
        inputs_by_name.clone(),
    );
    match &sym_result {
        Ok(outputs) => {
            println!(
                "  Interpreter finished in {:.1}ms",
                t0.elapsed().as_secs_f64() * 1e3
            );
            for (name, tensor) in outputs {
                println!(
                    "    output '{}': {:?} {:?}",
                    name,
                    tensor.dtype(),
                    tensor.shape()
                );
            }
        }
        Err(e) => {
            println!("  Interpreter failed: {}", e);
            println!("  (Will still attempt shape collection on MilliOpGraph)");
        }
    }

    // Now collect shapes from the MilliOpGraph
    // Map named inputs to GlobalId inputs
    let tensors_by_name = sym_graph.get_tensors_by_name();
    let milli_inputs: HashMap<GlobalId, NumericTensor<DynRank>> = inputs_by_name
        .into_iter()
        .filter_map(|(name, tensor)| tensors_by_name.get(&name).map(|id| (*id, tensor)))
        .collect();

    // Also add initialized tensors (weights)
    let initialized = sym_graph.get_initialized_tensors(tensor_store);
    let mut all_inputs = milli_inputs;
    for (id, tensor) in initialized {
        all_inputs.insert(id, tensor);
    }

    println!(
        "\n  Collecting shapes from MilliOpGraph ({} input tensors)...",
        all_inputs.len()
    );
    let t0 = Instant::now();
    let shapes_result = milli_graph.collect_all_shapes(&all_inputs);
    match shapes_result {
        Ok(shapes) => {
            println!(
                "  Collected {} tensor shapes in {:.1}ms",
                shapes.len(),
                t0.elapsed().as_secs_f64() * 1e3
            );

            // ---- Attempt v8 compilation ----
            println!("\nAttempting v8 compilation...");
            let t0 = Instant::now();
            match v8_codegen::compile_graph(&milli_graph, &shapes) {
                Ok((compiled, _artifacts)) => {
                    println!(
                        "  SUCCESS! Compiled in {:.1}ms",
                        t0.elapsed().as_secs_f64() * 1e3
                    );
                    println!("  {} kernels generated", compiled.kernels.len());
                    for (i, k) in compiled.kernels.iter().enumerate() {
                        println!(
                            "    kernel {}: output={:?}, parallel_extent={}, inputs={}",
                            i,
                            k.output_tensor,
                            k.parallel_extent,
                            k.input_tensors.len()
                        );
                    }
                }
                Err(e) => {
                    println!(
                        "  FAILED after {:.1}ms: {}",
                        t0.elapsed().as_secs_f64() * 1e3,
                        e
                    );
                }
            }
        }
        Err(e) => {
            println!("  Shape collection failed: {}", e);
            println!("  (Cannot attempt v8 compilation without shapes)");
        }
    }
}
