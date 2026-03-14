//! Lower a real ONNX model (GPT-2) to a NanoGraph, execute both the milli
//! interpreter and the nano interpreter, and compare results.
//!
//! Usage:
//!   cargo run --release --example nano_graph_model_test -- test_models/gpt2-lm-head-10.onnx

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use whisper_tensor::compiler::interpret_milli_graph;
use whisper_tensor::compiler::op_census;
use whisper_tensor::dtype::DType;
use whisper_tensor::graph::GlobalId;
use whisper_tensor::model::Model;
use whisper_tensor::nano_graph::lower;
use whisper_tensor::numeric_scalar::NumericScalar;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::tensor_info::TensorInfo;
use whisper_tensor::DynRank;
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

    // ---- Create inputs ----
    let input_info = model.get_input_tensor_info().unwrap();
    let sym_graph = model.get_symbolic_graph();
    let tensor_store = model.get_tensor_store();
    let tensors_by_name = sym_graph.get_tensors_by_name();

    // Build user input tensors (concrete data for milli interpreter).
    let mut milli_inputs: HashMap<GlobalId, NumericTensor<DynRank>> = HashMap::new();

    // Model weights — Numeric (full data).
    let initialized = sym_graph.get_initialized_tensors(tensor_store);
    for (id, tensor) in initialized {
        milli_inputs.insert(id, tensor);
    }

    // User inputs — fill with small integer values.
    for (name, (dtype, shape_dims)) in &input_info {
        let shape: Vec<u64> = shape_dims.iter().map(|d| d.unwrap_or(4)).collect();
        let num_elements: u64 = shape.iter().product();
        println!("  Input '{}': {:?} {:?} ({} elements)", name, dtype, shape, num_elements);

        let tensor: NumericTensor<DynRank> = match dtype {
            DType::I64 => {
                let data: Vec<i64> = (0..num_elements).map(|i| (i % 64) as i64).collect();
                let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
                NumericTensor::from_vec_shape(data, shape_usize).unwrap()
            }
            DType::I32 => {
                let data: Vec<i32> = (0..num_elements).map(|i| (i % 64) as i32).collect();
                let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
                NumericTensor::from_vec_shape(data, shape_usize).unwrap()
            }
            DType::F32 => {
                let data: Vec<f32> = (0..num_elements).map(|i| (i % 64) as f32 * 0.01).collect();
                let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
                NumericTensor::from_vec_shape(data, shape_usize).unwrap()
            }
            _ => {
                // Fallback: fill with f32 zeros, then cast
                let data: Vec<f32> = vec![0.0; num_elements as usize];
                let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
                NumericTensor::from_vec_shape(data, shape_usize).unwrap()
            }
        };

        if let Some(id) = tensors_by_name.get(name) {
            milli_inputs.insert(*id, tensor);
        }
    }
    println!("  Total input tensors: {}", milli_inputs.len());

    // ---- Build TensorInfo for lowering (shapes only for user inputs, full data for weights) ----
    let mut all_infos: HashMap<GlobalId, TensorInfo> = HashMap::new();
    for (id, tensor) in &milli_inputs {
        all_infos.insert(*id, TensorInfo::from(tensor.clone()));
    }

    // ---- Step 1: Run milli interpreter ----
    println!("\n=== Step 1: Milli Interpreter ===");
    let t_milli = Instant::now();
    let milli_outputs = interpret_milli_graph(&milli_graph, &milli_inputs).unwrap();
    let milli_elapsed = t_milli.elapsed();
    println!("  Milli interpreter completed in {:.1}s", milli_elapsed.as_secs_f64());
    println!("  Output tensors: {}", milli_outputs.len());

    // Print output shapes.
    for (id, tensor) in &milli_outputs {
        let shape: Vec<u64> = tensor.shape().to_vec();
        println!("    Output {:?}: {:?} {:?} ({} elements)",
            id, tensor.dtype(), shape, tensor.num_elements());
    }

    // ---- Lower to NanoGraph ----
    println!("\n=== Lowering to NanoGraph ===");
    let t_lower = Instant::now();
    let result = lower::lower_with_info(&milli_graph, &all_infos).unwrap();
    let lower_elapsed = t_lower.elapsed();
    println!("  Lowered in {:.1}ms", lower_elapsed.as_secs_f64() * 1e3);

    // ---- Report NanoGraph stats ----
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
        for detail in &result.unsupported_details {
            println!("    {}", detail);
        }
    }

    println!("\n  Output atoms: {}", result.graph.outputs.len());

    let errors = result.graph.validate();
    if errors.is_empty() {
        println!("  Validation: PASSED");
    } else {
        println!("  Validation: {} ERRORS", errors.len());
        for e in errors.iter().take(10) {
            println!("    {}", e);
        }
    }

    // ---- Step 2: Check feasibility of nano execution ----
    let scalar_size = std::mem::size_of::<NumericScalar>();
    let num_atoms = result.graph.num_atoms();
    let buffer_bytes = num_atoms as u128 * scalar_size as u128;
    let buffer_gb = buffer_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    println!("\n=== Step 2: Nano Execution Feasibility ===");
    println!("  sizeof(NumericScalar) = {} bytes", scalar_size);
    println!("  num_atoms = {}", num_atoms);
    println!("  Required buffer = {:.1} GB", buffer_gb);

    let max_buffer_gb = 240.0; // 128GB RAM + 128GB swap available
    let nano_feasible = buffer_gb <= max_buffer_gb;

    if !nano_feasible {
        println!("  SKIPPING nano execution: buffer ({:.1} GB) exceeds {:.0} GB limit", buffer_gb, max_buffer_gb);
        println!("  To run, need a machine with at least {:.0} GB RAM", buffer_gb * 1.5);
    } else {
        println!("  Buffer fits in memory, proceeding with nano execution");

        // Build nano inputs:
        // 1. Start with numeric_overrides (weights, constants)
        let mut nano_inputs: HashMap<u64, NumericScalar> = result.numeric_overrides.clone();
        println!("  Numeric overrides (weights/constants): {} atoms", nano_inputs.len());

        // 2. Add user input tensor values using tensor_map
        let mut user_input_atoms = 0u64;
        let mut backend = whisper_tensor::backends::eval_backend::EvalBackend::NDArray;
        for (name, (_dtype, _shape_dims)) in &input_info {
            let Some(id) = tensors_by_name.get(name) else { continue };
            let Some(tam) = result.tensor_map.get(id) else {
                println!("    WARNING: input '{}' ({:?}) not found in tensor_map", name, id);
                continue;
            };

            let tensor = &milli_inputs[id];
            // Cast to F32, flatten, extract values
            let f32_tensor = tensor.cast(DType::F32, &mut backend).unwrap();
            let flat = f32_tensor.flatten().unwrap();
            let nd = flat.to_ndarray().unwrap();
            let v: Vec<f32> = nd.try_into().unwrap();

            if v.len() != tam.count as usize {
                println!("    WARNING: input '{}' has {} elements but tensor_map says {} atoms",
                    name, v.len(), tam.count);
                continue;
            }

            for (i, &val) in v.iter().enumerate() {
                let atom_id = tam.base_id.0 + i as u64;
                // Use original dtype for the scalar value
                let scalar = match tensor.dtype() {
                    DType::I64 => {
                        // Recover the original i64 value
                        NumericScalar::I64(val as i64)
                    }
                    DType::I32 => NumericScalar::I32(val as i32),
                    _ => NumericScalar::F32(val),
                };
                nano_inputs.insert(atom_id, scalar);
            }
            user_input_atoms += tam.count;
            println!("    Input '{}': {} atoms at base {:?}", name, tam.count, tam.base_id);
        }
        println!("  User input atoms: {}", user_input_atoms);
        println!("  Total input atoms: {}", nano_inputs.len());

        // ---- Run nano interpreter ----
        println!("\n=== Step 3: Nano Interpreter ===");
        use whisper_tensor::compiler::attempts::v13_claude::nano_execute::execute_nanograph_naive;

        let t_nano = Instant::now();
        let nano_outputs = execute_nanograph_naive(&result.graph, &nano_inputs);
        let nano_elapsed = t_nano.elapsed();
        println!("  Nano interpreter completed in {:.1}s", nano_elapsed.as_secs_f64());

        // ---- Step 3: Compare outputs ----
        println!("\n=== Step 4: Output Comparison ===");

        // Build reverse output_map: external_id -> internal_id
        let reverse_output_map: HashMap<GlobalId, GlobalId> = milli_graph
            .output_map
            .as_ref()
            .map(|m| m.iter().map(|(&int, &ext)| (ext, int)).collect())
            .unwrap_or_default();

        let mut total_compared = 0u64;
        let mut max_abs_error: f64 = 0.0;
        let mut max_rel_error: f64 = 0.0;
        let mut comparison_errors = Vec::new();

        for (ext_id, milli_tensor) in &milli_outputs {
            // Find the internal ID
            let internal_id = reverse_output_map.get(ext_id).unwrap_or(ext_id);

            // Look up in tensor_map
            let Some(tam) = result.tensor_map.get(internal_id) else {
                // Try the external ID as a fallback
                let Some(tam) = result.tensor_map.get(ext_id) else {
                    comparison_errors.push(format!(
                        "Output {:?} (internal {:?}) not found in tensor_map", ext_id, internal_id
                    ));
                    continue;
                };
                compare_tensor_with_nano(
                    ext_id, milli_tensor, tam, &nano_outputs,
                    &mut total_compared, &mut max_abs_error, &mut max_rel_error,
                    &mut backend, &mut comparison_errors,
                );
                continue;
            };

            compare_tensor_with_nano(
                ext_id, milli_tensor, tam, &nano_outputs,
                &mut total_compared, &mut max_abs_error, &mut max_rel_error,
                &mut backend, &mut comparison_errors,
            );
        }

        println!("  Elements compared: {}", total_compared);
        println!("  Max absolute error: {:.6e}", max_abs_error);
        println!("  Max relative error: {:.6e}", max_rel_error);

        if !comparison_errors.is_empty() {
            println!("  Comparison issues ({}):", comparison_errors.len());
            for e in &comparison_errors {
                println!("    {}", e);
            }
        }

        if total_compared > 0 && max_abs_error < 1e-3 {
            println!("  RESULT: PASS (max abs error < 1e-3)");
        } else if total_compared > 0 {
            println!("  RESULT: MISMATCH (max abs error = {:.6e})", max_abs_error);
        } else {
            println!("  RESULT: NO ELEMENTS COMPARED");
        }

        // ---- Timing summary ----
        println!("\n=== Timing Summary ===");
        println!("  Milli interpreter: {:.3}s", milli_elapsed.as_secs_f64());
        println!("  NanoGraph lowering: {:.3}s", lower_elapsed.as_secs_f64());
        println!("  Nano interpreter:  {:.3}s", nano_elapsed.as_secs_f64());
    }

    // ---- Step 5: Partitioner (runs regardless of nano execution feasibility) ----
    println!("\n=== Step 5: Partitioner ===");
    use whisper_tensor::compiler::attempts::v13_claude::nano_part_b::partition_nanograph;

    let t_part = Instant::now();
    let partition = partition_nanograph(&result.graph, 8);
    let part_elapsed = t_part.elapsed();

    println!("  Partition: {} kernels", partition.num_kernels);
    println!("  Partitioning time: {:.3}s", part_elapsed.as_secs_f64());

    // Report kernel sizes
    let groups = result.graph.groups();
    let mut kernel_sizes: Vec<u64> = partition
        .kernel_groups
        .iter()
        .map(|kg| kg.iter().map(|&gi| groups[gi].count).sum())
        .collect();
    kernel_sizes.sort_unstable_by(|a, b| b.cmp(a));

    if !kernel_sizes.is_empty() {
        println!("  Largest kernel:  {} atoms", kernel_sizes[0]);
        println!("  Smallest kernel: {} atoms", kernel_sizes.last().unwrap());
        let total: u64 = kernel_sizes.iter().sum();
        println!("  Average kernel:  {} atoms", total / kernel_sizes.len() as u64);

        // Show distribution
        println!("  Kernel sizes (top 10):");
        for (i, size) in kernel_sizes.iter().take(10).enumerate() {
            println!("    kernel {:>3}: {:>12} atoms", i, size);
        }
    }

    // ---- Step 6: JIT Codegen Execution (f32 buffer — 4 bytes/atom) ----
    #[cfg(feature = "cranelift")]
    {
        let f32_buffer_gb = num_atoms as f64 * 4.0 / (1024.0 * 1024.0 * 1024.0);
        println!("\n=== Step 6: JIT Codegen Execution ===");
        println!("  f32 buffer: {:.1} GB ({} atoms × 4 bytes)", f32_buffer_gb, num_atoms);

        if f32_buffer_gb > 240.0 {
            println!("  SKIPPING: f32 buffer too large");
        } else {
            use whisper_tensor::compiler::attempts::v13_claude::nano_codegen::CompiledPipeline;

            // Build f32 overrides from numeric_overrides
            let mut overrides_f32: HashMap<u64, f32> = HashMap::new();
            for (&atom_idx, scalar) in &result.numeric_overrides {
                overrides_f32.insert(atom_idx, scalar.to_f64() as f32);
            }

            // Add user input values
            let mut backend = whisper_tensor::backends::eval_backend::EvalBackend::NDArray;
            for (name, (_dtype, _shape_dims)) in &input_info {
                let Some(id) = tensors_by_name.get(name) else { continue };
                let Some(tam) = result.tensor_map.get(id) else { continue };
                let tensor = &milli_inputs[id];
                let f32_tensor = tensor.cast(DType::F32, &mut backend).unwrap();
                let flat = f32_tensor.flatten().unwrap();
                let nd = flat.to_ndarray().unwrap();
                let v: Vec<f32> = nd.try_into().unwrap();
                for (i, &val) in v.iter().enumerate() {
                    overrides_f32.insert(tam.base_id.0 + i as u64, val);
                }
            }
            println!("  Total f32 overrides: {}", overrides_f32.len());

            // Compile per-kernel
            println!("  Compiling {} kernels...", partition.num_kernels);
            let t_compile = Instant::now();
            match CompiledPipeline::compile_partitioned(&result.graph, &partition) {
                Ok(pipeline) => {
                    let compile_elapsed = t_compile.elapsed();
                    println!("  Compiled in {:.3}s", compile_elapsed.as_secs_f64());

                    // Execute
                    println!("  Executing...");
                    let t_exec = Instant::now();
                    let jit_values = pipeline.execute_full(&overrides_f32);
                    let exec_elapsed = t_exec.elapsed();
                    println!("  Executed in {:.3}s", exec_elapsed.as_secs_f64());

                    // Compare JIT output vs milli interpreter
                    let reverse_output_map: HashMap<GlobalId, GlobalId> = milli_graph
                        .output_map
                        .as_ref()
                        .map(|m| m.iter().map(|(&int, &ext)| (ext, int)).collect())
                        .unwrap_or_default();

                    let mut total_compared = 0u64;
                    let mut max_abs_error: f64 = 0.0;
                    let mut max_rel_error: f64 = 0.0;

                    for (ext_id, milli_tensor) in &milli_outputs {
                        let internal_id = reverse_output_map.get(ext_id).unwrap_or(ext_id);
                        let tam = result.tensor_map.get(internal_id)
                            .or_else(|| result.tensor_map.get(ext_id));
                        let Some(tam) = tam else {
                            println!("    Output {:?}: not in tensor_map", ext_id);
                            continue;
                        };

                        let f32_tensor = milli_tensor.cast(DType::F32, &mut backend).unwrap();
                        let flat = f32_tensor.flatten().unwrap();
                        let nd = flat.to_ndarray().unwrap();
                        let milli_vals: Vec<f32> = nd.try_into().unwrap();

                        let mut local_max_abs = 0.0f64;
                        for (i, &milli_val) in milli_vals.iter().enumerate() {
                            let atom_idx = (tam.base_id.0 + i as u64) as usize;
                            if atom_idx >= jit_values.len() { break; }
                            let jit_val = jit_values[atom_idx];
                            let abs_err = (milli_val - jit_val).abs() as f64;
                            local_max_abs = local_max_abs.max(abs_err);
                            let rel_err = if milli_val.abs() > 1e-8 {
                                abs_err / milli_val.abs() as f64
                            } else { 0.0 };
                            max_rel_error = max_rel_error.max(rel_err);
                            total_compared += 1;
                        }
                        max_abs_error = max_abs_error.max(local_max_abs);
                        println!("    Output {:?}: {} elems, max_abs_err={:.6e}",
                            ext_id, milli_vals.len(), local_max_abs);
                    }

                    println!("  Elements compared: {}", total_compared);
                    println!("  Max absolute error: {:.6e}", max_abs_error);
                    println!("  Max relative error: {:.6e}", max_rel_error);

                    if total_compared > 0 && max_abs_error < 1e-2 {
                        println!("  RESULT: PASS");
                    } else if total_compared > 0 {
                        println!("  RESULT: MISMATCH (max abs error = {:.6e})", max_abs_error);
                    } else {
                        println!("  RESULT: NO ELEMENTS COMPARED");
                    }

                    println!("\n=== JIT Timing Summary ===");
                    println!("  Milli interpreter: {:.3}s", milli_elapsed.as_secs_f64());
                    println!("  NanoGraph lowering: {:.3}s", lower_elapsed.as_secs_f64());
                    println!("  Partitioning:      {:.3}s", part_elapsed.as_secs_f64());
                    println!("  JIT compilation:   {:.3}s", compile_elapsed.as_secs_f64());
                    println!("  JIT execution:     {:.3}s", exec_elapsed.as_secs_f64());
                }
                Err(e) => {
                    println!("  Compilation FAILED: {}", e);
                }
            }
        }
    }
}

fn compare_tensor_with_nano(
    ext_id: &GlobalId,
    milli_tensor: &NumericTensor<DynRank>,
    tam: &lower::TensorAtomMapInfo,
    nano_outputs: &HashMap<u64, NumericScalar>,
    total_compared: &mut u64,
    max_abs_error: &mut f64,
    max_rel_error: &mut f64,
    backend: &mut whisper_tensor::backends::eval_backend::EvalBackend,
    comparison_errors: &mut Vec<String>,
) {
    // Cast milli output to f32 and flatten
    let Ok(f32_tensor) = milli_tensor.cast(DType::F32, backend) else {
        comparison_errors.push(format!("Output {:?}: failed to cast to F32", ext_id));
        return;
    };
    let Ok(flat) = f32_tensor.flatten() else {
        comparison_errors.push(format!("Output {:?}: failed to flatten", ext_id));
        return;
    };
    let Ok(nd) = flat.to_ndarray() else {
        comparison_errors.push(format!("Output {:?}: failed to convert to ndarray", ext_id));
        return;
    };
    let Ok(milli_values): Result<Vec<f32>, _> = nd.try_into() else {
        comparison_errors.push(format!("Output {:?}: failed to extract f32 values", ext_id));
        return;
    };

    if milli_values.len() != tam.count as usize {
        comparison_errors.push(format!(
            "Output {:?}: milli has {} elements but tensor_map has {} atoms",
            ext_id, milli_values.len(), tam.count
        ));
        return;
    }

    let mut local_max_abs = 0.0f64;
    let mut local_max_rel = 0.0f64;
    let mut compared = 0u64;

    for (i, &milli_val) in milli_values.iter().enumerate() {
        let atom_id = tam.base_id.0 + i as u64;
        let Some(nano_scalar) = nano_outputs.get(&atom_id) else {
            comparison_errors.push(format!(
                "Output {:?}: atom {} not found in nano outputs", ext_id, atom_id
            ));
            return;
        };

        let nano_val = nano_scalar.to_f64() as f32;
        let abs_err = (milli_val - nano_val).abs() as f64;
        let rel_err = if milli_val.abs() > 1e-8 {
            abs_err / milli_val.abs() as f64
        } else {
            0.0
        };

        local_max_abs = local_max_abs.max(abs_err);
        local_max_rel = local_max_rel.max(rel_err);
        compared += 1;
    }

    println!("    Output {:?}: {} elements, max_abs_err={:.6e}, max_rel_err={:.6e}",
        ext_id, compared, local_max_abs, local_max_rel);

    *total_compared += compared;
    *max_abs_error = max_abs_error.max(local_max_abs);
    *max_rel_error = max_rel_error.max(local_max_rel);
}
