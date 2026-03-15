//! Fast lowering + partitioning diagnostic for ONNX models.
//! Skips interpreter execution and numeric_overrides (the slow parts).
//!
//! Usage:
//!   cargo run --release --example nano_lower_test -- test_models/gpt2-lm-head-10.onnx

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use whisper_tensor::compiler::op_census;
use whisper_tensor::compiler::attempts::v13_claude::nano_plan_v2c::ExecutionPlan as V2cPlan;
use whisper_tensor::graph::GlobalId;
use whisper_tensor::model::Model;
use whisper_tensor::nano_graph::{InputRef, NanoGraph, ScalarOp};
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

    // ---- Generate MilliOpGraph ----
    let t0 = Instant::now();
    let milli_graph = model.get_symbolic_graph().generate_milli_graph(&mut rng);
    eprintln!("MilliOpGraph: {:.1}ms", t0.elapsed().as_secs_f64() * 1e3);

    let census = op_census(&milli_graph);
    let total_ops: usize = census.iter().map(|(_, c)| c).sum();
    println!("=== Milli Ops ({} total) ===", total_ops);
    for (kind, count) in &census {
        println!("  {:>4}x  {}", count, kind);
    }

    // ---- Build tensor info (shapes only, skip weight data extraction) ----
    let t0 = Instant::now();
    let input_info = model.get_input_tensor_info().unwrap();
    let sym_graph = model.get_symbolic_graph();
    let tensor_store = model.get_tensor_store();
    let tensors_by_name = sym_graph.get_tensors_by_name();

    let mut all_infos: HashMap<GlobalId, TensorInfo> = HashMap::new();

    // User inputs
    for (name, (dtype, shape_dims)) in &input_info {
        let shape: Vec<u64> = shape_dims.iter().map(|d| d.unwrap_or(4)).collect();
        let info = TensorInfo::from_dtype_and_shape(*dtype, &shape);
        println!("Input '{}': {:?} {:?}", name, dtype, shape);
        if let Some(id) = tensors_by_name.get(name) {
            all_infos.insert(*id, info);
        }
    }

    // Model weights
    let initialized = sym_graph.get_initialized_tensors(tensor_store);
    for (id, tensor) in initialized {
        all_infos.insert(id, TensorInfo::from(tensor));
    }
    eprintln!("Tensor info: {:.1}ms ({} tensors)", t0.elapsed().as_secs_f64() * 1e3, all_infos.len());

    // ---- Infer shapes ----
    let t0 = Instant::now();
    let inferred = milli_graph.infer_all(&all_infos).unwrap();
    eprintln!("infer_all: {:.1}ms ({} tensors)", t0.elapsed().as_secs_f64() * 1e3, inferred.len());

    // ---- Lower (graph only, skip numeric_overrides for speed) ----
    let t0 = Instant::now();
    let result = whisper_tensor::nano_graph::lower::lower_graph_only(&milli_graph, &all_infos).unwrap();
    let lower_elapsed = t0.elapsed();
    eprintln!("lower_with_info: {:.1}s", lower_elapsed.as_secs_f64());

    // ---- NanoGraph stats ----
    let ts = Instant::now();
    let stats = result.graph.stats();
    eprintln!("stats: {:.1}ms", ts.elapsed().as_secs_f64() * 1e3);
    println!("\n=== NanoGraph ===");
    println!("{}", stats);

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

    if result.graph.num_atoms() < 10_000_000 {
        let tv = Instant::now();
        let errors = result.graph.validate();
        eprintln!("validate: {:.1}ms", tv.elapsed().as_secs_f64() * 1e3);
        if errors.is_empty() {
            println!("Validation: PASSED");
        } else {
            println!("Validation: {} ERRORS", errors.len());
            for e in errors.iter().take(10) { println!("  {}", e); }
        }
    } else {
        println!("Validation: SKIPPED ({}B atoms too large for full validation)", result.graph.num_atoms());
    }

    // ---- Explicit InputRef diagnostic ----
    let te = Instant::now();
    print_explicit_diagnostic(&result.graph);
    eprintln!("explicit diagnostic: {:.1}ms", te.elapsed().as_secs_f64() * 1e3);

    // ---- InputRef type distribution ----
    let ti = Instant::now();
    print_inputref_distribution(&result.graph);
    eprintln!("inputref distribution: {:.1}ms", ti.elapsed().as_secs_f64() * 1e3);

    // ---- ReduceSum diagnostic ----
    print_reduce_diagnostic(&result.graph);

    // ---- Lane+Barrier Execution Plans ----
    let num_lanes = std::env::var("NUM_LANES").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    println!("\n=== Lane+Barrier Plans (num_lanes={}) ===", num_lanes);

    {
        let t0 = Instant::now();
        let plan = whisper_tensor::compiler::attempts::v13_claude::nano_plan_creative::plan_execution(&result.graph, num_lanes);
        let elapsed = t0.elapsed();
        let raw: Vec<Vec<Vec<usize>>> = plan.phases.iter().map(|p| p.lane_work.clone()).collect();
        print_plan_summary(&result.graph, &raw, plan.num_lanes, "creative", elapsed);
    }
    {
        let t0 = Instant::now();
        let plan = whisper_tensor::compiler::attempts::v13_claude::nano_plan_critical::plan_execution(&result.graph, num_lanes);
        let elapsed = t0.elapsed();
        let raw: Vec<Vec<Vec<usize>>> = plan.phases.iter().map(|p| p.lane_work.clone()).collect();
        print_plan_summary(&result.graph, &raw, plan.num_lanes, "critical", elapsed);
    }
    {
        let t0 = Instant::now();
        let plan = whisper_tensor::compiler::attempts::v13_claude::nano_plan_iterative::plan_execution(&result.graph, num_lanes);
        let elapsed = t0.elapsed();
        let raw: Vec<Vec<Vec<usize>>> = plan.phases.iter().map(|p| p.lane_work.clone()).collect();
        print_plan_summary(&result.graph, &raw, plan.num_lanes, "iterative", elapsed);
    }

    // ---- V2 Lane Planners (group splitting) ----
    println!("\n=== V2 Lane+Barrier Plans (num_lanes={}) ===", num_lanes);
    {
        let t0 = Instant::now();
        let plan = whisper_tensor::compiler::attempts::v13_claude::nano_plan_v2a::plan_execution(&result.graph, num_lanes);
        let elapsed = t0.elapsed();
        let raw: Vec<Vec<Vec<usize>>> = plan.phases.iter().map(|p| {
            p.lane_work.iter().map(|lw| lw.iter().map(|w| w.group_idx).collect()).collect()
        }).collect();
        // Compute actual atom-level balance
        let mut max_imb: f64 = 0.0;
        for phase in &plan.phases {
            let lane_atoms: Vec<u64> = phase.lane_work.iter()
                .map(|lw| lw.iter().map(|w| w.atom_count).sum::<u64>()).collect();
            let mx = lane_atoms.iter().copied().max().unwrap_or(0);
            let mn = lane_atoms.iter().copied().filter(|&a| a > 0).min().unwrap_or(1);
            if mn > 0 { max_imb = max_imb.max(mx as f64 / mn as f64); }
        }
        let total_work: u64 = plan.phases.iter().flat_map(|p| p.lane_work.iter())
            .flat_map(|lw| lw.iter()).map(|w| w.atom_count).sum();
        println!("  [v2a] {:.1}ms, {} lanes, {} phases, {:.1}B atoms, max_imbalance={:.1}x",
            elapsed.as_secs_f64() * 1e3, plan.num_lanes, plan.phases.len(),
            total_work as f64 / 1e9, max_imb);
    }
    {
        let t0 = Instant::now();
        let plan = whisper_tensor::compiler::attempts::v13_claude::nano_plan_v2b::plan_execution(&result.graph, num_lanes);
        let elapsed = t0.elapsed();
        let mut max_imb: f64 = 0.0;
        for phase in &plan.phases {
            let lane_atoms: Vec<u64> = phase.lane_work.iter()
                .map(|lw| lw.iter().map(|w| w.atom_count).sum::<u64>()).collect();
            let mx = lane_atoms.iter().copied().max().unwrap_or(0);
            let mn = lane_atoms.iter().copied().filter(|&a| a > 0).min().unwrap_or(1);
            if mn > 0 { max_imb = max_imb.max(mx as f64 / mn as f64); }
        }
        let total_work: u64 = plan.phases.iter().flat_map(|p| p.lane_work.iter())
            .flat_map(|lw| lw.iter()).map(|w| w.atom_count).sum();
        println!("  [v2b] {:.1}ms, {} lanes, {} phases, {:.1}B atoms, max_imbalance={:.1}x",
            elapsed.as_secs_f64() * 1e3, plan.num_lanes, plan.phases.len(),
            total_work as f64 / 1e9, max_imb);
    }
    {
        let t0 = Instant::now();
        let plan = whisper_tensor::compiler::attempts::v13_claude::nano_plan_v2c::plan_execution(&result.graph, num_lanes);
        let elapsed = t0.elapsed();
        let mut max_imb: f64 = 0.0;
        for phase in &plan.phases {
            let lane_atoms: Vec<u64> = phase.lane_work.iter()
                .map(|lw| lw.iter().map(|w| w.atom_count).sum::<u64>()).collect();
            let mx = lane_atoms.iter().copied().max().unwrap_or(0);
            let mn = lane_atoms.iter().copied().filter(|&a| a > 0).min().unwrap_or(1);
            if mn > 0 { max_imb = max_imb.max(mx as f64 / mn as f64); }
        }
        let total_work: u64 = plan.phases.iter().flat_map(|p| p.lane_work.iter())
            .flat_map(|lw| lw.iter()).map(|w| w.atom_count).sum();
        println!("  [v2c] {:.1}ms, {} lanes, {} phases, {:.1}B atoms, max_imbalance={:.1}x",
            elapsed.as_secs_f64() * 1e3, plan.num_lanes, plan.phases.len(),
            total_work as f64 / 1e9, max_imb);

        // ---- V2C Deep Diagnostic ----
        print_v2c_diagnostic(&result.graph, &plan);
    }

    // ---- V2C Codegen Execution + Validity Check ----
    #[cfg(feature = "cranelift")]
    {
        use whisper_tensor::compiler::attempts::v13_claude::nano_codegen_v2::CompiledPlan;
        use whisper_tensor::compiler::attempts::v13_claude::nano_plan_v2c;

        // Run milli interpreter for reference
        println!("\n=== Milli Interpreter (reference) ===");
        let t0 = Instant::now();
        // Need full lower with numeric_overrides for the codegen
        let full_result = whisper_tensor::nano_graph::lower::lower_with_info(&milli_graph, &all_infos).unwrap();
        eprintln!("  Full lower: {:.1}s", t0.elapsed().as_secs_f64());

        // Build input tensors for milli interpreter
        let mut milli_inputs: HashMap<GlobalId, whisper_tensor::numeric_tensor::NumericTensor<whisper_tensor::DynRank>> = HashMap::new();
        let initialized2 = sym_graph.get_initialized_tensors(tensor_store);
        for (id, tensor) in initialized2 {
            milli_inputs.insert(id, tensor);
        }
        for (name, (dtype, shape_dims)) in &input_info {
            let shape: Vec<u64> = shape_dims.iter().map(|d| d.unwrap_or(4)).collect();
            let numel: usize = shape.iter().product::<u64>() as usize;
            if let Some(id) = tensors_by_name.get(name) {
                let data: Vec<i64> = (0..numel as i64).collect();
                let tensor = whisper_tensor::numeric_tensor::NumericTensor::from_vec_shape(
                    data, shape.iter().map(|&s| s as usize).collect()
                ).unwrap();
                milli_inputs.insert(*id, tensor);
            }
        }
        let t0 = Instant::now();
        let milli_outputs = whisper_tensor::compiler::interpret_milli_graph(&milli_graph, &milli_inputs).unwrap();
        println!("  Milli interpreter: {:.1}s, {} outputs", t0.elapsed().as_secs_f64(), milli_outputs.len());

        println!("\n=== V2C Codegen Execution ===");
        let t0 = Instant::now();
        let plan = nano_plan_v2c::plan_execution(&result.graph, num_lanes);
        eprintln!("  Plan: {:.1}ms, {} phases", t0.elapsed().as_secs_f64() * 1e3, plan.phases.len());

        let f32_buffer_gb = result.graph.num_atoms() as f64 * 4.0 / (1024.0 * 1024.0 * 1024.0);
        println!("  f32 buffer: {:.1} GB", f32_buffer_gb);

        if f32_buffer_gb > 120.0 {
            println!("  SKIPPING: buffer too large");
        } else {
            let t0 = Instant::now();
            match CompiledPlan::compile(&result.graph, &plan) {
                Ok(compiled) => {
                    let compile_time = t0.elapsed();
                    println!("  Compiled in {:.1}s ({} phase modules)",
                        compile_time.as_secs_f64(), plan.phases.len());

                    // Build f32 overrides from numeric_overrides (weights + constants)
                    let mut overrides: HashMap<u64, f32> = HashMap::new();
                    for (&atom_idx, scalar) in &full_result.numeric_overrides {
                        overrides.insert(atom_idx, scalar.to_f64() as f32);
                    }
                    // Add user input values
                    let mut backend_v2c = whisper_tensor::backends::eval_backend::EvalBackend::NDArray;
                    for (name, (_dtype, _shape_dims)) in &input_info {
                        let Some(id) = tensors_by_name.get(name) else { continue };
                        let Some(tam) = full_result.tensor_map.get(id) else { continue };
                        let Some(tensor) = milli_inputs.get(id) else { continue };
                        let f32_tensor = tensor.cast(whisper_tensor::dtype::DType::F32, &mut backend_v2c).unwrap();
                        let flat = f32_tensor.flatten().unwrap();
                        let nd = flat.to_ndarray().unwrap();
                        let v: Vec<f32> = nd.try_into().unwrap();
                        for (i, &val) in v.iter().enumerate() {
                            let atom_id = tam.atom_id_for_element(i as u64);
                            overrides.insert(atom_id.0, val);
                        }
                    }
                    println!("  Overrides: {} entries (weights + user inputs)", overrides.len());

                    // Execute via JIT
                    let t0 = Instant::now();
                    let values = compiled.execute(&overrides);
                    let exec_time = t0.elapsed();
                    println!("  JIT executed in {:.1}s", exec_time.as_secs_f64());

                    // Also execute via interpreter (same plan, no codegen)
                    let t0 = Instant::now();
                    let groups = full_result.graph.groups();
                    let num_atoms = full_result.graph.num_atoms() as usize;
                    let mut interp_values = vec![0.0f32; num_atoms];
                    // Pre-fill Literal atoms from the ScalarOp values
                    for group in groups {
                        if let ScalarOp::Literal(scalar) = &group.op {
                            let val = scalar.to_f64() as f32;
                            for i in 0..group.count {
                                interp_values[(group.base_id.0 + i) as usize] = val;
                            }
                        }
                    }
                    // Then override with numeric_overrides (which may have different values)
                    for (&idx, &val) in &overrides {
                        interp_values[idx as usize] = val;
                    }
                    // Execute plan phases in order, all lanes
                    for phase in &plan.phases {
                        for lane_work in &phase.lane_work {
                            for w in lane_work {
                                let group = &groups[w.group_idx];
                                if matches!(&group.op, ScalarOp::Literal(_)) { continue; }
                                for i in w.atom_offset..w.atom_offset + w.atom_count {
                                    let atom_idx = (group.base_id.0 + i) as usize;
                                    let val = match &group.op {
                                        ScalarOp::Literal(_) => continue,
                                        ScalarOp::Identity { .. } => {
                                            interp_values[group.inputs[0].resolve(i, 0).0 as usize]
                                        }
                                        ScalarOp::Binary { op, .. } => {
                                            let a = interp_values[group.inputs[0].resolve(i, 0).0 as usize];
                                            let b = interp_values[group.inputs[1].resolve(i, 0).0 as usize];
                                            match op {
                                                whisper_tensor::nano_graph::ScalarBinOp::Add => a + b,
                                                whisper_tensor::nano_graph::ScalarBinOp::Sub => a - b,
                                                whisper_tensor::nano_graph::ScalarBinOp::Mul => a * b,
                                                whisper_tensor::nano_graph::ScalarBinOp::Div => a / b,
                                                whisper_tensor::nano_graph::ScalarBinOp::Max => a.max(b),
                                                whisper_tensor::nano_graph::ScalarBinOp::Min => a.min(b),
                                                whisper_tensor::nano_graph::ScalarBinOp::Pow => a.powf(b),
                                                whisper_tensor::nano_graph::ScalarBinOp::Mod => a % b,
                                                whisper_tensor::nano_graph::ScalarBinOp::Equal => if a == b { 1.0 } else { 0.0 },
                                                whisper_tensor::nano_graph::ScalarBinOp::Greater => if a > b { 1.0 } else { 0.0 },
                                                whisper_tensor::nano_graph::ScalarBinOp::GreaterOrEqual => if a >= b { 1.0 } else { 0.0 },
                                                whisper_tensor::nano_graph::ScalarBinOp::Less => if a < b { 1.0 } else { 0.0 },
                                                whisper_tensor::nano_graph::ScalarBinOp::LessOrEqual => if a <= b { 1.0 } else { 0.0 },
                                                whisper_tensor::nano_graph::ScalarBinOp::And => if a != 0.0 && b != 0.0 { 1.0 } else { 0.0 },
                                                whisper_tensor::nano_graph::ScalarBinOp::Or => if a != 0.0 || b != 0.0 { 1.0 } else { 0.0 },
                                                whisper_tensor::nano_graph::ScalarBinOp::Xor => if (a != 0.0) ^ (b != 0.0) { 1.0 } else { 0.0 },
                                            }
                                        }
                                        ScalarOp::Unary { op, .. } => {
                                            let x = interp_values[group.inputs[0].resolve(i, 0).0 as usize];
                                            match op {
                                                whisper_tensor::nano_graph::ScalarUnaryOp::Neg => -x,
                                                whisper_tensor::nano_graph::ScalarUnaryOp::Abs => x.abs(),
                                                whisper_tensor::nano_graph::ScalarUnaryOp::Exp => x.exp(),
                                                whisper_tensor::nano_graph::ScalarUnaryOp::Ln => x.ln(),
                                                whisper_tensor::nano_graph::ScalarUnaryOp::Sqrt => x.sqrt(),
                                                whisper_tensor::nano_graph::ScalarUnaryOp::Reciprocal => 1.0 / x,
                                                whisper_tensor::nano_graph::ScalarUnaryOp::Tanh => x.tanh(),
                                                whisper_tensor::nano_graph::ScalarUnaryOp::Floor => x.floor(),
                                                whisper_tensor::nano_graph::ScalarUnaryOp::Ceil => x.ceil(),
                                            }
                                        }
                                        ScalarOp::Select { .. } => {
                                            let cond = interp_values[group.inputs[0].resolve(i, 0).0 as usize];
                                            if cond != 0.0 {
                                                interp_values[group.inputs[1].resolve(i, 0).0 as usize]
                                            } else {
                                                interp_values[group.inputs[2].resolve(i, 0).0 as usize]
                                            }
                                        }
                                        ScalarOp::ReduceSum { reduce_count, reduce_stride, .. } => {
                                            let base = group.inputs[0].resolve(i, 0);
                                            let mut acc = 0.0f32;
                                            for k in 0..*reduce_count {
                                                let src = (base.0 as i64 + k as i64 * reduce_stride) as usize;
                                                acc += interp_values[src];
                                            }
                                            acc
                                        }
                                        ScalarOp::ReduceMax { reduce_count, reduce_stride, .. } => {
                                            let base = group.inputs[0].resolve(i, 0);
                                            let mut acc = f32::NEG_INFINITY;
                                            for k in 0..*reduce_count {
                                                let src = (base.0 as i64 + k as i64 * reduce_stride) as usize;
                                                acc = acc.max(interp_values[src]);
                                            }
                                            acc
                                        }
                                        ScalarOp::IndirectLoad { table_base, .. } => {
                                            let idx = interp_values[group.inputs[0].resolve(i, 0).0 as usize];
                                            interp_values[table_base.0 as usize + idx as usize]
                                        }
                                    };
                                    interp_values[atom_idx] = val;
                                }
                            }
                        }
                    }
                    let interp_exec_time = t0.elapsed();
                    println!("  Plan interpreter executed in {:.1}s", interp_exec_time.as_secs_f64());

                    // Compare against milli interpreter
                    let reverse_output_map: HashMap<GlobalId, GlobalId> = milli_graph
                        .output_map
                        .as_ref()
                        .map(|m| m.iter().map(|(&int, &ext)| (ext, int)).collect())
                        .unwrap_or_default();

                    let mut backend = whisper_tensor::backends::eval_backend::EvalBackend::NDArray;

                    // Compare BOTH JIT and plan-interpreter against milli
                    for (label, test_values) in [("JIT", &values), ("PlanInterp", &interp_values)] {
                        println!("\n  --- {} vs Milli ---", label);
                        let mut total_compared = 0u64;
                        let mut max_abs_error: f64 = 0.0;
                        let mut max_rel_error: f64 = 0.0;

                        for (ext_id, milli_tensor) in &milli_outputs {
                            let internal_id = reverse_output_map.get(ext_id).unwrap_or(ext_id);
                            let tam = full_result.tensor_map.get(internal_id)
                                .or_else(|| full_result.tensor_map.get(ext_id));
                            let Some(tam) = tam else { continue };

                            let f32_tensor = milli_tensor.cast(
                                whisper_tensor::dtype::DType::F32, &mut backend).unwrap();
                            let flat = f32_tensor.flatten().unwrap();
                            let nd = flat.to_ndarray().unwrap();
                            let milli_vals: Vec<f32> = nd.try_into().unwrap();

                            let mut local_max_abs = 0.0f64;
                            for (i, &milli_val) in milli_vals.iter().enumerate() {
                                let atom_id = tam.atom_id_for_element(i as u64);
                                let idx = atom_id.0 as usize;
                                if idx >= test_values.len() { break; }
                                let test_val = test_values[idx];
                                let abs_err = (milli_val - test_val).abs() as f64;
                                local_max_abs = local_max_abs.max(abs_err);
                                let rel_err = if milli_val.abs() > 1e-8 {
                                    abs_err / milli_val.abs() as f64
                                } else { 0.0 };
                                max_rel_error = max_rel_error.max(rel_err);
                                total_compared += 1;
                            }
                            max_abs_error = max_abs_error.max(local_max_abs);
                        }
                        println!("  {} Elements compared: {}", label, total_compared);
                        println!("  {} Max absolute error: {:.6e}", label, max_abs_error);
                        if total_compared > 0 && max_abs_error < 1e-1 {
                            println!("  {} RESULT: PASS", label);
                        } else if total_compared > 0 {
                            println!("  {} RESULT: MISMATCH", label);
                        } else {
                            println!("  {} RESULT: NO ELEMENTS COMPARED", label);
                        }
                    }
                }
                Err(e) => {
                    println!("  Compilation FAILED: {}", e);
                }
            }
        }
    }

    // ---- Old-style Partition (compare approaches) ----
    let target_kernels = 200;

    // New partitioners (allow interleaved group indices, expose parallelism)
    {
        let name = "bisect";
        println!("\n=== Partitioner: {} ===", name);
        let t0 = Instant::now();
        let p = whisper_tensor::compiler::attempts::v13_claude::nano_part_bisect::partition_nanograph(&result.graph, target_kernels);
        eprintln!("{}: {:.1}ms, {} kernels", name, t0.elapsed().as_secs_f64() * 1e3, p.num_kernels);
        print_partition_summary(&result.graph, &p.kernel_groups, name);
    }
    {
        let name = "hybrid";
        println!("\n=== Partitioner: {} ===", name);
        let t0 = Instant::now();
        let p = whisper_tensor::compiler::attempts::v13_claude::nano_part_hybrid::partition_nanograph(&result.graph, target_kernels);
        eprintln!("{}: {:.1}ms, {} kernels", name, t0.elapsed().as_secs_f64() * 1e3, p.num_kernels);
        print_partition_summary(&result.graph, &p.kernel_groups, name);
    }
    {
        let name = "creative";
        println!("\n=== Partitioner: {} ===", name);
        let t0 = Instant::now();
        let p = whisper_tensor::compiler::attempts::v13_claude::nano_part_creative::partition_nanograph(&result.graph, target_kernels);
        eprintln!("{}: {:.1}ms, {} kernels", name, t0.elapsed().as_secs_f64() * 1e3, p.num_kernels);
        print_partition_summary(&result.graph, &p.kernel_groups, name);
    }

    // Use creative for the detailed breakdown
    let t0 = Instant::now();
    let partition = whisper_tensor::compiler::attempts::v13_claude::nano_part_creative::partition_nanograph(&result.graph, target_kernels);
    eprintln!("Partitioned in {:.1}ms", t0.elapsed().as_secs_f64() * 1e3);
    println!("{} kernels", partition.num_kernels);

    let groups = result.graph.groups();
    // Build group→kernel map
    let mut group_to_kernel = vec![usize::MAX; groups.len()];
    for (ki, kg) in partition.kernel_groups.iter().enumerate() {
        for &gi in kg {
            group_to_kernel[gi] = ki;
        }
    }
    let group_base_ids: Vec<u64> = groups.iter().map(|g| g.base_id.0).collect();
    let find_group_idx = |atom_id: whisper_tensor::nano_graph::AtomId| -> Option<usize> {
        match group_base_ids.binary_search(&atom_id.0) {
            Ok(i) => Some(i),
            Err(0) => None,
            Err(i) => {
                let gi = i - 1;
                if atom_id.0 < groups[gi].base_id.0 + groups[gi].count { Some(gi) } else { None }
            }
        }
    };

    for (ki, kg) in partition.kernel_groups.iter().enumerate() {
        let total_atoms: u64 = kg.iter().map(|&gi| groups[gi].count).sum();
        let mut op_counts: HashMap<String, usize> = HashMap::new();
        let mut explicit_entries = 0u64;
        for &gi in kg {
            let op_name = format!("{:?}", groups[gi].op)
                .chars().take_while(|c| *c != ' ' && *c != '{' && *c != '(').collect::<String>();
            *op_counts.entry(op_name).or_default() += 1;
            for input in &groups[gi].inputs {
                if let InputRef::Explicit(ids) = input {
                    explicit_entries += ids.len() as u64;
                }
            }
        }
        let mut sorted_ops: Vec<_> = op_counts.into_iter().collect();
        sorted_ops.sort_by(|a, b| b.1.cmp(&a.1));
        let op_summary: String = sorted_ops.iter().take(5)
            .map(|(op, count)| format!("{}x{}", count, op)).collect::<Vec<_>>().join(", ");

        let mut reads_from: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        for &gi in kg {
            for input in &groups[gi].inputs {
                let sample_atoms: Vec<whisper_tensor::nano_graph::AtomId> = match input {
                    InputRef::Broadcast(id) => vec![*id],
                    InputRef::Affine { base, .. } | InputRef::StridedBroadcast { base, .. }
                    | InputRef::SymAffine { base, .. } | InputRef::Modular { base, .. } => vec![*base],
                    InputRef::Explicit(ids) => {
                        let mut s = vec![];
                        if !ids.is_empty() { s.push(ids[0]); }
                        if ids.len() > 1 { s.push(ids[ids.len()-1]); }
                        s
                    }
                };
                for id in sample_atoms {
                    if let Some(src_gi) = find_group_idx(id) {
                        let src_ki = group_to_kernel[src_gi];
                        if src_ki != ki && src_ki != usize::MAX { reads_from.insert(src_ki); }
                    }
                }
            }
        }
        let min_gi = kg.iter().copied().min().unwrap_or(0);
        let max_gi = kg.iter().copied().max().unwrap_or(0);
        println!("  k{:>2}: {:>6} grp [{:>5}..{:>5}] {:>12} atoms {:>8} explicit  deps={:?}  {}",
            ki, kg.len(), min_gi, max_gi, total_atoms, explicit_entries, reads_from, op_summary);
    }
}

fn print_partition_summary(graph: &NanoGraph, kernel_groups: &[Vec<usize>], name: &str) {
    let groups = graph.groups();
    let num_kernels = kernel_groups.len();

    // Kernel sizes
    let mut sizes: Vec<(usize, u64, usize)> = kernel_groups.iter().enumerate()
        .map(|(ki, kg)| (ki, kg.iter().map(|&gi| groups[gi].count).sum::<u64>(), kg.len()))
        .collect();
    sizes.sort_by(|a, b| b.1.cmp(&a.1));

    let total_atoms: u64 = sizes.iter().map(|s| s.1).sum();
    let max_pct = if total_atoms > 0 { sizes[0].1 as f64 / total_atoms as f64 * 100.0 } else { 0.0 };

    println!("  {} kernels, max kernel {:.1}% of total", num_kernels, max_pct);

    // Check acyclicity of the kernel dependency graph
    let groups = graph.groups();
    let group_base_ids: Vec<u64> = groups.iter().map(|g| g.base_id.0).collect();
    let find_gi = |atom_id: whisper_tensor::nano_graph::AtomId| -> Option<usize> {
        match group_base_ids.binary_search(&atom_id.0) {
            Ok(i) => Some(i),
            Err(0) => None,
            Err(i) => {
                let gi = i - 1;
                if atom_id.0 < groups[gi].base_id.0 + groups[gi].count { Some(gi) } else { None }
            }
        }
    };
    let mut g2k = vec![usize::MAX; groups.len()];
    for (ki, kg) in kernel_groups.iter().enumerate() {
        for &gi in kg { g2k[gi] = ki; }
    }
    // Build kernel dep edges
    let mut kernel_deps: Vec<std::collections::BTreeSet<usize>> = vec![std::collections::BTreeSet::new(); num_kernels];
    for (ki, kg) in kernel_groups.iter().enumerate() {
        for &gi in kg {
            for input in &groups[gi].inputs {
                use whisper_tensor::nano_graph::InputRef;
                let bases: Vec<whisper_tensor::nano_graph::AtomId> = match input {
                    InputRef::Broadcast(id) => vec![*id],
                    InputRef::Affine { base, .. } | InputRef::StridedBroadcast { base, .. }
                    | InputRef::SymAffine { base, .. } | InputRef::Modular { base, .. } => vec![*base],
                    InputRef::Explicit(ids) => {
                        let mut s = vec![];
                        if !ids.is_empty() { s.push(ids[0]); }
                        if ids.len() > 1 { s.push(ids[ids.len()-1]); }
                        s
                    }
                };
                for id in bases {
                    if let Some(src_gi) = find_gi(id) {
                        let src_ki = g2k[src_gi];
                        if src_ki != ki && src_ki != usize::MAX {
                            kernel_deps[ki].insert(src_ki);
                        }
                    }
                }
            }
        }
    }
    // Check for cycles via Kahn's algorithm
    let mut in_degree = vec![0usize; num_kernels];
    for deps in &kernel_deps {
        for &dep in deps { in_degree[dep] += 1; } // note: this counts reverse edges
    }
    // Actually: kernel_deps[ki] = set of kernels ki reads FROM. So edges are dep→ki.
    let mut in_deg = vec![0usize; num_kernels];
    for (ki, deps) in kernel_deps.iter().enumerate() {
        in_deg[ki] = deps.len(); // ki has in_deg = number of kernels it depends on
    }
    let mut queue: std::collections::VecDeque<usize> = in_deg.iter().enumerate()
        .filter(|&(_, d)| *d == 0).map(|(i, _)| i).collect();
    let mut visited = 0;
    while let Some(ki) = queue.pop_front() {
        visited += 1;
        // Find kernels that depend on ki
        for (other, deps) in kernel_deps.iter().enumerate() {
            if deps.contains(&ki) {
                in_deg[other] -= 1;
                if in_deg[other] == 0 { queue.push_back(other); }
            }
        }
    }
    let acyclic = visited == num_kernels;
    let num_dep_edges: usize = kernel_deps.iter().map(|d| d.len()).sum();
    println!("  acyclic={}, dep_edges={}", acyclic, num_dep_edges);

    // If not acyclic, find and print one cycle
    if !acyclic {
        // Find a kernel still with nonzero in-degree (part of a cycle)
        if let Some(start) = in_deg.iter().position(|&d| d > 0) {
            // DFS to find cycle
            let mut path = vec![start];
            let mut visited_set = std::collections::HashSet::new();
            visited_set.insert(start);
            let mut found_cycle = false;
            'outer: loop {
                let cur = *path.last().unwrap();
                let mut next = None;
                for &dep in &kernel_deps[cur] {
                    if in_deg[dep] > 0 { // still in a cycle component
                        if visited_set.contains(&dep) {
                            // Found cycle: dep appears earlier in path
                            let cycle_start = path.iter().position(|&k| k == dep).unwrap();
                            let cycle: Vec<usize> = path[cycle_start..].to_vec();
                            println!("  CYCLE (len {}): {:?}", cycle.len(), cycle);
                            // Print ALL edges in the cycle
                            let mut cycle_ext = cycle.clone();
                            cycle_ext.push(cycle[0]); // close the loop
                            for w in cycle_ext.windows(2) {
                                let (ka, kb) = (w[0], w[1]);
                                // ka depends on kb (ka reads from kb)
                                let mut edge_count = 0;
                                for &gi in &kernel_groups[ka] {
                                    for input in &groups[gi].inputs {
                                        use whisper_tensor::nano_graph::InputRef;
                                        let bases: Vec<whisper_tensor::nano_graph::AtomId> = match input {
                                            InputRef::Broadcast(id) => vec![*id],
                                            InputRef::Affine { base, .. } | InputRef::StridedBroadcast { base, .. }
                                            | InputRef::SymAffine { base, .. } | InputRef::Modular { base, .. } => vec![*base],
                                            InputRef::Explicit(ids) if !ids.is_empty() => vec![ids[0]],
                                            _ => vec![],
                                        };
                                        for id in bases {
                                            if let Some(src_gi) = find_gi(id) {
                                                if g2k[src_gi] == kb && edge_count < 3 {
                                                    let c_op = format!("{:?}", groups[gi].op).chars().take_while(|c| *c != ' ' && *c != '{' && *c != '(').collect::<String>();
                                                    let p_op = format!("{:?}", groups[src_gi].op).chars().take_while(|c| *c != ' ' && *c != '{' && *c != '(').collect::<String>();
                                                    println!("    k{} reads k{}: g{} ({} cnt={}) ← g{} ({} cnt={})",
                                                        ka, kb, gi, c_op, groups[gi].count, src_gi, p_op, groups[src_gi].count);
                                                    edge_count += 1;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            // Print details of each edge in the cycle
                            for w in cycle.windows(2) {
                                let (ka, kb) = (w[0], w[1]);
                                // Find the actual group edge
                                for &gi in &kernel_groups[ka] {
                                    for input in &groups[gi].inputs {
                                        use whisper_tensor::nano_graph::InputRef;
                                        let bases: Vec<whisper_tensor::nano_graph::AtomId> = match input {
                                            InputRef::Broadcast(id) => vec![*id],
                                            InputRef::Affine { base, .. } | InputRef::StridedBroadcast { base, .. }
                                            | InputRef::SymAffine { base, .. } | InputRef::Modular { base, .. } => vec![*base],
                                            InputRef::Explicit(ids) if !ids.is_empty() => vec![ids[0]],
                                            _ => vec![],
                                        };
                                        for id in bases {
                                            if let Some(src_gi) = find_gi(id) {
                                                if g2k[src_gi] == kb {
                                                    let consumer_op = format!("{:?}", groups[gi].op)
                                                        .chars().take_while(|c| *c != ' ' && *c != '{' && *c != '(').collect::<String>();
                                                    let producer_op = format!("{:?}", groups[src_gi].op)
                                                        .chars().take_while(|c| *c != ' ' && *c != '{' && *c != '(').collect::<String>();
                                                    println!("    k{}→k{}: g{} ({}, count={}) reads from g{} ({}, count={})",
                                                        kb, ka, gi, consumer_op, groups[gi].count,
                                                        src_gi, producer_op, groups[src_gi].count);
                                                    break 'outer;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            found_cycle = true;
                            break 'outer;
                        }
                        next = Some(dep);
                        break;
                    }
                }
                if let Some(n) = next {
                    path.push(n);
                    visited_set.insert(n);
                } else {
                    break; // dead end
                }
            }
        }
    }

    // Top 5 kernels
    for &(ki, atoms, ngroups) in sizes.iter().take(5) {
        println!("    k{}: {} groups, {} atoms ({:.1}%)", ki, ngroups, atoms, atoms as f64 / total_atoms as f64 * 100.0);
    }
    if sizes.len() > 5 {
        println!("    ... and {} more kernels", sizes.len() - 5);
    }
}

fn print_plan_summary(
    graph: &NanoGraph,
    phases: &[Vec<Vec<usize>>], // phase -> lane -> group indices
    num_lanes: usize,
    name: &str,
    elapsed: std::time::Duration,
) {
    let groups = graph.groups();
    let num_phases = phases.len();

    // Count total assigned groups
    let mut total_assigned = 0usize;
    let mut total_atoms = 0u64;
    for phase in phases {
        for lane_groups in phase {
            total_assigned += lane_groups.len();
            total_atoms += lane_groups.iter().map(|&gi| groups[gi].count).sum::<u64>();
        }
    }

    // Per-phase balance
    let mut max_imbalance: f64 = 0.0;
    let mut phase_sizes: Vec<(usize, u64, u64, usize)> = Vec::new(); // (phase, max_lane, min_lane, num_groups)
    for (pi, phase) in phases.iter().enumerate() {
        let lane_atoms: Vec<u64> = phase.iter()
            .map(|lg| lg.iter().map(|&gi| groups[gi].count).sum::<u64>())
            .collect();
        let max_lane = lane_atoms.iter().copied().max().unwrap_or(0);
        let min_lane = lane_atoms.iter().copied().filter(|&a| a > 0).min().unwrap_or(0);
        let num_groups: usize = phase.iter().map(|lg| lg.len()).sum();
        if min_lane > 0 {
            max_imbalance = max_imbalance.max(max_lane as f64 / min_lane as f64);
        }
        phase_sizes.push((pi, max_lane, min_lane, num_groups));
    }

    println!("\n  [{}] {:.1}ms, {} lanes, {} phases, {} groups assigned, {:.1}B atoms",
        name, elapsed.as_secs_f64() * 1e3, num_lanes, num_phases,
        total_assigned, total_atoms as f64 / 1e9);
    println!("    max phase imbalance: {:.1}x", max_imbalance);

    // Show first few and last few phases
    let show = 3;
    for (pi, max_l, min_l, ng) in phase_sizes.iter().take(show).copied() {
        let balance = if min_l > 0 { format!("{:.1}x", max_l as f64 / min_l as f64) } else { "inf".to_string() };
        println!("    phase {:>3}: {:>6} groups, max_lane={:>12}, balance={}", pi, ng, max_l, balance);
    }
    if num_phases > show * 2 {
        println!("    ... ({} more phases) ...", num_phases - show * 2);
    }
    for (pi, max_l, min_l, ng) in phase_sizes.iter().rev().take(show).copied().collect::<Vec<_>>().into_iter().rev() {
        let balance = if min_l > 0 { format!("{:.1}x", max_l as f64 / min_l as f64) } else { "inf".to_string() };
        println!("    phase {:>3}: {:>6} groups, max_lane={:>12}, balance={}", pi, ng, max_l, balance);
    }
}

fn print_explicit_diagnostic(graph: &NanoGraph) {
    let groups = graph.groups();
    let group_base_ids: Vec<u64> = groups.iter().map(|g| g.base_id.0).collect();
    let find_group_idx = |atom_id: whisper_tensor::nano_graph::AtomId| -> Option<usize> {
        match group_base_ids.binary_search(&atom_id.0) {
            Ok(i) => Some(i),
            Err(0) => None,
            Err(i) => {
                let gi = i - 1;
                if atom_id.0 < groups[gi].base_id.0 + groups[gi].count { Some(gi) } else { None }
            }
        }
    };

    let mut explicit_groups = 0u64;
    let mut total_entries = 0u64;
    let mut by_op: HashMap<String, (u64, u64)> = HashMap::new();
    // Collect size distribution of Explicit tables
    let mut size_buckets: HashMap<String, Vec<u64>> = HashMap::new(); // op -> vec of counts

    for group in groups {
        let mut has = false;
        let mut entries = 0u64;
        for input in &group.inputs {
            if let InputRef::Explicit(ids) = input {
                has = true;
                entries += ids.len() as u64;
            }
        }
        if has {
            explicit_groups += 1;
            total_entries += entries;
            let op_name = format!("{:?}", group.op)
                .chars().take_while(|c| *c != ' ' && *c != '{' && *c != '(').collect::<String>();
            let e = by_op.entry(op_name.clone()).or_default();
            e.0 += 1; e.1 += entries;
            size_buckets.entry(op_name).or_default().push(group.count);
        }
    }
    println!("\n=== Explicit InputRef ===");
    println!("{} groups, {} entries ({:.1} MB)", explicit_groups, total_entries,
        total_entries as f64 * 8.0 / (1024.0 * 1024.0));
    let mut sorted: Vec<_> = by_op.into_iter().collect();
    sorted.sort_by(|a, b| b.1.1.cmp(&a.1.1));
    for (op, (g, e)) in &sorted {
        let sizes = size_buckets.get(op).unwrap();
        let min = sizes.iter().copied().min().unwrap_or(0);
        let max = sizes.iter().copied().max().unwrap_or(0);
        println!("  {}: {} groups, {} entries ({:.1} MB), count range [{}, {}]",
            op, g, e, *e as f64 * 8.0 / (1024.0*1024.0), min, max);
    }

    // Show Explicit groups, prioritizing large ones
    println!("\n  Identity Explicit groups (largest first):");
    let mut identity_explicits: Vec<(usize, usize, &Vec<whisper_tensor::nano_graph::AtomId>)> = Vec::new();
    for (gi, group) in groups.iter().enumerate() {
        if !matches!(&group.op, whisper_tensor::nano_graph::ScalarOp::Identity { .. }) { continue; }
        for (input_idx, input) in group.inputs.iter().enumerate() {
            if let InputRef::Explicit(ids) = input {
                identity_explicits.push((gi, input_idx, ids));
            }
        }
    }
    identity_explicits.sort_by(|a, b| b.2.len().cmp(&a.2.len()));
    for &(gi, input_idx, ids) in identity_explicits.iter().take(15) {
        let group = &groups[gi];
        let producer = if !ids.is_empty() { find_group_idx(ids[0]) } else { None };
        let prod_info = producer.map(|pi| {
            let pg = &groups[pi];
            let op_name = format!("{:?}", pg.op)
                .chars().take_while(|c| *c != ' ' && *c != '{' && *c != '(').collect::<String>();
            format!("g{}:{} count={}", pi, op_name, pg.count)
        }).unwrap_or("?".to_string());
        // Check stride pattern within the explicit
        let inner_pattern = if ids.len() >= 4 {
            // Check global stride pattern
            let s0 = ids[1].0 as i64 - ids[0].0 as i64;
            let all_affine = ids.windows(2).take(20).all(|w| (w[1].0 as i64 - w[0].0 as i64) == s0);
            if all_affine {
                // Verify on a few more samples
                let truly_affine = ids.len() < 100 || ids.windows(2).all(|w| (w[1].0 as i64 - w[0].0 as i64) == s0);
                if truly_affine { format!("affine(stride={})", s0) }
                else { format!("~affine(stride={},breaks)", s0) }
            } else {
                // Check if it's a segmented pattern (chunks of stride-1 with gaps)
                let mut chunk_len = 1u64;
                while (chunk_len as usize) < ids.len() && ids[chunk_len as usize].0 == ids[0].0 + chunk_len { chunk_len += 1; }
                if chunk_len > 1 && chunk_len < ids.len() as u64 {
                    let gap = ids[chunk_len as usize].0 as i64 - ids[chunk_len as usize - 1].0 as i64;
                    // Check if all chunks have the same length and gap
                    let num_chunks = (ids.len() as u64 + chunk_len - 1) / chunk_len;
                    format!("chunks(len={},gap={},n={})", chunk_len, gap, num_chunks)
                } else { "mixed".to_string() }
            }
        } else { "tiny".to_string() };
        // Check if this looks like a slice of the producer (regular stride through a larger array)
        let first = ids[0].0;
        let last = if ids.len() > 1 { ids[ids.len()-1].0 } else { first };
        let span = last - first + 1;
        let ratio = if !ids.is_empty() { span as f64 / ids.len() as f64 } else { 0.0 };
        // For top 3, show strides at chunk boundaries
        let stride_detail = if identity_explicits.iter().position(|x| x.0 == gi).unwrap_or(99) < 3 && ids.len() > 10 {
            let strides: Vec<i64> = ids.windows(2).take(10).map(|w| w[1].0 as i64 - w[0].0 as i64).collect();
            // Also find first break in stride=1
            let first_break = ids.windows(2).position(|w| w[1].0 != w[0].0 + 1).unwrap_or(ids.len());
            format!(" strides={:?} first_break@{}", strides, first_break)
        } else { String::new() };
        println!("    g{} count={} entries={} inner={} span={} ratio={:.1} producer={}{}",
            gi, group.count, ids.len(), inner_pattern, span, ratio, prod_info, stride_detail);
    }

    println!("\n  Other Explicit groups (first 10):");
    let mut shown = 0;
    for (gi, group) in groups.iter().enumerate() {
        if matches!(&group.op, whisper_tensor::nano_graph::ScalarOp::Identity { .. }) { continue; }
        for (input_idx, input) in group.inputs.iter().enumerate() {
            if let InputRef::Explicit(ids) = input {
                if shown >= 10 { break; }
                // Check if the Explicit pattern is regular
                let pattern = if ids.len() >= 2 {
                    let stride = ids[1].0 as i64 - ids[0].0 as i64;
                    let is_affine = ids.windows(2).all(|w| (w[1].0 as i64 - w[0].0 as i64) == stride);
                    if is_affine { format!("affine(stride={})", stride) }
                    else {
                        // Check for repeated blocks
                        let mut rep = 1u64;
                        while (rep as usize) < ids.len() && ids[rep as usize].0 == ids[0].0 { rep += 1; }
                        if rep > 1 { format!("blocks(repeat={})", rep) }
                        else { "irregular".to_string() }
                    }
                } else { "tiny".to_string() };

                // Find producer group
                let producer = if !ids.is_empty() { find_group_idx(ids[0]) } else { None };
                let prod_info = producer.map(|pi| {
                    let pg = &groups[pi];
                    let op_name = format!("{:?}", pg.op)
                        .chars().take_while(|c| *c != ' ' && *c != '{' && *c != '(').collect::<String>();
                    format!("g{}:{} count={}", pi, op_name, pg.count)
                }).unwrap_or("?".to_string());

                let op_name = format!("{:?}", group.op)
                    .chars().take_while(|c| *c != ' ' && *c != '{' && *c != '(').collect::<String>();
                // Show first few entries for irregular patterns
                let sample = if ids.len() > 8 {
                    let first4: Vec<u64> = ids[..4].iter().map(|a| a.0).collect();
                    let last2: Vec<u64> = ids[ids.len()-2..].iter().map(|a| a.0).collect();
                    format!("[{:?}...{:?}]", first4, last2)
                } else {
                    format!("{:?}", ids.iter().map(|a| a.0).collect::<Vec<_>>())
                };
                println!("    g{} {} count={} input[{}]: {} entries, pattern={}, producer={}, ids={}",
                    gi, op_name, group.count, input_idx, ids.len(), pattern, prod_info, sample);
                shown += 1;
            }
        }
        if shown >= 10 { break; }
    }
}

fn print_inputref_distribution(graph: &NanoGraph) {
    let groups = graph.groups();
    let mut broadcast = 0u64;
    let mut affine = 0u64;
    let mut strided_broadcast = 0u64;
    let mut sym_affine = 0u64;
    let mut explicit = 0u64;
    let mut modular = 0u64;
    for group in groups {
        for input in &group.inputs {
            match input {
                InputRef::Broadcast(_) => broadcast += 1,
                InputRef::Affine { .. } => affine += 1,
                InputRef::StridedBroadcast { .. } => strided_broadcast += 1,
                InputRef::SymAffine { .. } => sym_affine += 1,
                InputRef::Explicit(_) => explicit += 1,
                InputRef::Modular { .. } => modular += 1,
            }
        }
    }
    println!("\n=== InputRef Distribution ===");
    println!("  Broadcast: {}", broadcast);
    println!("  Affine: {}", affine);
    println!("  StridedBroadcast: {}", strided_broadcast);
    println!("  SymAffine: {}", sym_affine);
    println!("  Explicit: {}", explicit);
    println!("  Modular: {}", modular);
}

fn print_reduce_diagnostic(graph: &NanoGraph) {
    let groups = graph.groups();
    let mut reduce_count = 0;
    let mut reduce_atoms = 0u64;
    let mut has_sym_affine = 0;
    for group in groups {
        let is_reduce = matches!(&group.op,
            whisper_tensor::nano_graph::ScalarOp::ReduceSum { .. } |
            whisper_tensor::nano_graph::ScalarOp::ReduceMax { .. });
        if is_reduce {
            reduce_count += 1;
            reduce_atoms += group.count;
            if group.inputs.iter().any(|i| matches!(i, InputRef::SymAffine { .. })) {
                has_sym_affine += 1;
            }
        }
    }
    println!("\n=== Reduce Groups ===");
    println!("  {} reduce groups, {} atoms", reduce_count, reduce_atoms);
    println!("  {} with SymAffine input (symbolic reduction)", has_sym_affine);
    println!("  {} with other input patterns", reduce_count - has_sym_affine);
}

fn op_short_name(op: &ScalarOp) -> String {
    format!("{:?}", op)
        .chars()
        .take_while(|c| *c != ' ' && *c != '{' && *c != '(')
        .collect()
}

fn print_v2c_diagnostic(graph: &NanoGraph, plan: &V2cPlan) {
    let groups = graph.groups();
    let num_phases = plan.phases.len();
    let num_lanes = plan.num_lanes;

    println!("\n========================================");
    println!("=== V2C Deep Diagnostic ({} phases, {} lanes) ===", num_phases, num_lanes);
    println!("========================================");

    // ─── 1. Phase size distribution ─────────────────────────────────────
    println!("\n--- 1. Phase Size Distribution ---");
    let mut phase_group_counts: Vec<usize> = Vec::new();
    let mut phase_atom_counts: Vec<u64> = Vec::new();
    for phase in &plan.phases {
        let ng: usize = phase.lane_work.iter().map(|l| l.len()).sum();
        let na: u64 = phase.lane_work.iter().flat_map(|l| l.iter()).map(|w| w.atom_count).sum();
        phase_group_counts.push(ng);
        phase_atom_counts.push(na);
    }

    let total_atoms: u64 = phase_atom_counts.iter().sum();

    let single_group = phase_group_counts.iter().filter(|&&c| c == 1).count();
    let small_2_10 = phase_group_counts.iter().filter(|&&c| c >= 2 && c <= 10).count();
    let medium_11_100 = phase_group_counts.iter().filter(|&&c| c >= 11 && c <= 100).count();
    let large_100_plus = phase_group_counts.iter().filter(|&&c| c > 100).count();

    println!("  Phases with 1 group:    {:>4}", single_group);
    println!("  Phases with 2-10:       {:>4}", small_2_10);
    println!("  Phases with 11-100:     {:>4}", medium_11_100);
    println!("  Phases with 100+:       {:>4}", large_100_plus);

    let mut sorted_gc = phase_group_counts.clone();
    sorted_gc.sort();
    let median_groups = if sorted_gc.is_empty() { 0 } else { sorted_gc[sorted_gc.len() / 2] };
    let mean_groups = if num_phases > 0 { sorted_gc.iter().sum::<usize>() as f64 / num_phases as f64 } else { 0.0 };
    println!("  Median groups/phase:    {:>4}", median_groups);
    println!("  Mean groups/phase:      {:>6.1}", mean_groups);

    // Atoms in tiny phases
    let tiny_atoms: u64 = phase_group_counts.iter().zip(phase_atom_counts.iter())
        .filter(|(gc, _)| **gc < 10)
        .map(|(_, ac)| *ac)
        .sum();
    println!("  Atoms in tiny phases (<10 groups): {:.3}B ({:.1}% of total {:.3}B)",
        tiny_atoms as f64 / 1e9,
        if total_atoms > 0 { tiny_atoms as f64 / total_atoms as f64 * 100.0 } else { 0.0 },
        total_atoms as f64 / 1e9);

    // ─── 2. Barrier analysis ────────────────────────────────────────────
    println!("\n--- 2. Barrier Analysis ---");

    // Classify what ops are at phase boundaries
    // For each phase, what ops are in it?
    let mut phase_op_breakdown: Vec<HashMap<String, (usize, u64)>> = Vec::new(); // (count, atoms)
    for phase in &plan.phases {
        let mut ops: HashMap<String, (usize, u64)> = HashMap::new();
        for lane in &phase.lane_work {
            for w in lane {
                let name = op_short_name(&groups[w.group_idx].op);
                let e = ops.entry(name).or_default();
                e.0 += 1;
                e.1 += w.atom_count;
            }
        }
        phase_op_breakdown.push(ops);
    }

    // Count barriers between matmul phases vs elementwise phases
    // A "matmul phase" contains ReduceSum or Mul groups; "elementwise" does not
    let is_matmul_phase: Vec<bool> = phase_op_breakdown.iter().map(|ops| {
        ops.contains_key("ReduceSum") || ops.contains_key("ReduceMax")
            || (ops.contains_key("Binary") && ops.values().map(|v| v.1).sum::<u64>() > 1_000_000)
    }).collect();

    let has_reduce: Vec<bool> = phase_op_breakdown.iter().map(|ops| {
        ops.contains_key("ReduceSum") || ops.contains_key("ReduceMax")
    }).collect();

    let mut barriers_matmul_to_matmul = 0;
    let mut barriers_matmul_to_elem = 0;
    let mut barriers_elem_to_matmul = 0;
    let mut barriers_elem_to_elem = 0;
    let mut barriers_reduce_boundary = 0; // barrier where prev phase has ReduceSum
    for i in 1..num_phases {
        let prev_mm = is_matmul_phase[i - 1];
        let cur_mm = is_matmul_phase[i];
        match (prev_mm, cur_mm) {
            (true, true) => barriers_matmul_to_matmul += 1,
            (true, false) => barriers_matmul_to_elem += 1,
            (false, true) => barriers_elem_to_matmul += 1,
            (false, false) => barriers_elem_to_elem += 1,
        }
        if has_reduce[i - 1] {
            barriers_reduce_boundary += 1;
        }
    }
    println!("  Total barriers: {}", num_phases - 1);
    println!("  Barriers at ReduceSum boundary: {}", barriers_reduce_boundary);
    println!("  matmul->matmul: {}", barriers_matmul_to_matmul);
    println!("  matmul->elem:   {}", barriers_matmul_to_elem);
    println!("  elem->matmul:   {}", barriers_elem_to_matmul);
    println!("  elem->elem:     {}", barriers_elem_to_elem);

    // Potentially unnecessary barriers: phases where all work uses only 1 lane
    // or where there's no actual cross-lane dependency
    let mut single_lane_phases = 0;
    let mut all_same_lane_phases = 0;
    for phase in &plan.phases {
        let active_lanes: Vec<usize> = phase.lane_work.iter().enumerate()
            .filter(|(_, l)| !l.is_empty())
            .map(|(i, _)| i)
            .collect();
        if active_lanes.len() <= 1 {
            single_lane_phases += 1;
        }
        // Check if all work is on the same set of lanes (suggesting maybe barrier wasn't needed)
        let distinct_groups: std::collections::HashSet<usize> = phase.lane_work.iter()
            .flat_map(|l| l.iter().map(|w| w.group_idx))
            .collect();
        if distinct_groups.len() <= 1 {
            all_same_lane_phases += 1;
        }
    }
    println!("  Phases with only 1 active lane: {}", single_lane_phases);
    println!("  Phases with only 1 distinct group: {}", all_same_lane_phases);

    // ─── 3. Lane work consistency ───────────────────────────────────────
    println!("\n--- 3. Lane Work Consistency ---");

    // Find a sequence of 4 consecutive "interesting" phases (ones with >10 groups)
    let interesting_phases: Vec<usize> = (0..num_phases)
        .filter(|&pi| phase_group_counts[pi] > 10)
        .collect();
    let show_phases: Vec<usize> = if interesting_phases.len() >= 4 {
        // Pick 4 consecutive interesting phases from the middle
        let mid = interesting_phases.len() / 2;
        let start = if mid >= 2 { mid - 2 } else { 0 };
        interesting_phases[start..start.min(interesting_phases.len()) + 4.min(interesting_phases.len() - start)].to_vec()
    } else {
        // Just pick 4 phases from the middle
        let start = if num_phases > 4 { num_phases / 2 - 2 } else { 0 };
        (start..num_phases.min(start + 4)).collect()
    };

    for &pi in &show_phases {
        let phase = &plan.phases[pi];
        println!("  Phase {}:", pi);
        for (lane_idx, lane) in phase.lane_work.iter().enumerate() {
            if lane.is_empty() { continue; }
            let total_atoms: u64 = lane.iter().map(|w| w.atom_count).sum();
            let gis: Vec<usize> = lane.iter().map(|w| w.group_idx).collect();
            let min_gi = gis.iter().copied().min().unwrap_or(0);
            let max_gi = gis.iter().copied().max().unwrap_or(0);
            // Summarize ops
            let mut lane_ops: HashMap<String, usize> = HashMap::new();
            for w in lane {
                *lane_ops.entry(op_short_name(&groups[w.group_idx].op)).or_default() += 1;
            }
            let mut ops_sorted: Vec<_> = lane_ops.into_iter().collect();
            ops_sorted.sort_by(|a, b| b.1.cmp(&a.1));
            let op_str: String = ops_sorted.iter().take(3)
                .map(|(op, c)| format!("{}x{}", c, op)).collect::<Vec<_>>().join(", ");
            println!("    lane {}: {} items, {:.1}M atoms, groups [{}-{}], ops: {}",
                lane_idx, lane.len(), total_atoms as f64 / 1e6, min_gi, max_gi, op_str);
        }
    }

    // Check lane consistency: do the same lanes handle the same group index ranges across phases?
    println!("\n  Lane consistency across phases (group index range per lane):");
    // Sample 6 phases spread across the plan
    let sample_phases: Vec<usize> = if num_phases <= 6 {
        (0..num_phases).collect()
    } else {
        (0..6).map(|i| i * (num_phases - 1) / 5).collect()
    };
    print!("  {:>8}", "Phase");
    for li in 0..num_lanes { print!("  lane{:<8}", li); }
    println!();
    for &pi in &sample_phases {
        let phase = &plan.phases[pi];
        print!("  {:>8}", pi);
        for lane in &phase.lane_work {
            if lane.is_empty() {
                print!("  {:>12}", "-");
            } else {
                let min_gi = lane.iter().map(|w| w.group_idx).min().unwrap();
                let max_gi = lane.iter().map(|w| w.group_idx).max().unwrap();
                print!("  {:>5}-{:<5}", min_gi, max_gi);
            }
        }
        println!();
    }

    // ─── 4. Phase content analysis ──────────────────────────────────────
    println!("\n--- 4. Phase Content Analysis ---");

    // 10 largest phases
    let mut indexed_atom_counts: Vec<(usize, u64, usize)> = phase_atom_counts.iter()
        .enumerate()
        .map(|(i, &a)| (i, a, phase_group_counts[i]))
        .collect();
    indexed_atom_counts.sort_by(|a, b| b.1.cmp(&a.1));

    println!("  10 largest phases (by atoms):");
    for &(pi, atoms, ngroups) in indexed_atom_counts.iter().take(10) {
        let ops = &phase_op_breakdown[pi];
        let mut ops_sorted: Vec<_> = ops.iter().map(|(k, v)| (k.clone(), v.0, v.1)).collect();
        ops_sorted.sort_by(|a, b| b.2.cmp(&a.2));
        let op_str: String = ops_sorted.iter().take(5)
            .map(|(op, c, a)| format!("{}x{} ({:.1}M)", c, op, *a as f64 / 1e6))
            .collect::<Vec<_>>().join(", ");
        // Balance for this phase
        let phase = &plan.phases[pi];
        let lane_atoms: Vec<u64> = phase.lane_work.iter()
            .map(|l| l.iter().map(|w| w.atom_count).sum::<u64>()).collect();
        let mx = lane_atoms.iter().copied().max().unwrap_or(0);
        let mn = lane_atoms.iter().copied().filter(|&a| a > 0).min().unwrap_or(1);
        let balance = if mn > 0 { mx as f64 / mn as f64 } else { f64::INFINITY };
        println!("    phase {:>3}: {:>6} groups, {:>12.1}M atoms, balance={:.2}x | {}",
            pi, ngroups, atoms as f64 / 1e6, balance, op_str);
    }

    // 10 smallest phases (non-empty)
    let mut smallest: Vec<(usize, u64, usize)> = indexed_atom_counts.iter()
        .filter(|&&(_, a, _)| a > 0)
        .copied()
        .collect();
    smallest.sort_by(|a, b| a.1.cmp(&b.1));

    println!("\n  10 smallest phases (by atoms):");
    for &(pi, atoms, ngroups) in smallest.iter().take(10) {
        let ops = &phase_op_breakdown[pi];
        let mut ops_sorted: Vec<_> = ops.iter().map(|(k, v)| (k.clone(), v.0, v.1)).collect();
        ops_sorted.sort_by(|a, b| b.2.cmp(&a.2));
        let op_str: String = ops_sorted.iter().take(5)
            .map(|(op, c, a)| format!("{}x{} ({})", c, op, a))
            .collect::<Vec<_>>().join(", ");
        // Check if adjacent phases could absorb this
        let prev_atoms = if pi > 0 { phase_atom_counts[pi - 1] } else { 0 };
        let next_atoms = if pi + 1 < num_phases { phase_atom_counts[pi + 1] } else { 0 };
        println!("    phase {:>3}: {:>6} groups, {:>10} atoms | {} | neighbors: prev={}, next={}",
            pi, ngroups, atoms, op_str, prev_atoms, next_atoms);
    }

    // ─── 5. Overall Assessment ──────────────────────────────────────────
    println!("\n--- 5. Overall Assessment ---");

    // Count matmuls and reduces
    let mut total_reduce_groups = 0usize;
    let mut total_reduce_atoms = 0u64;
    let _total_mul_groups = 0usize;
    let mut total_identity_groups = 0usize;
    let mut total_binary_groups = 0usize;
    let mut total_unary_groups = 0usize;
    let mut total_literal_groups = 0usize;
    for g in groups {
        match &g.op {
            ScalarOp::ReduceSum { .. } | ScalarOp::ReduceMax { .. } => {
                total_reduce_groups += 1;
                total_reduce_atoms += g.count;
            }
            ScalarOp::Binary { .. } => total_binary_groups += 1,
            ScalarOp::Unary { .. } => total_unary_groups += 1,
            ScalarOp::Identity { .. } => total_identity_groups += 1,
            ScalarOp::Literal(_) => total_literal_groups += 1,
            _ => {}
        }
    }

    // Count phases that contain reduces
    let phases_with_reduce = has_reduce.iter().filter(|&&r| r).count();

    // Phases per reduce: how many phases does each reduce group span?
    // Theoretical minimum barriers = number of "levels" in the reduce DAG
    // For GPT-2 with 10 layers, each with ~7 matmuls: 70 matmuls, each creating
    // a reduce boundary = ~140+ barriers minimum if all matmuls are sequential.

    println!("  Total groups: {} ({} compute, {} literal)", groups.len(),
        groups.len() - total_literal_groups, total_literal_groups);
    println!("  Reduce groups: {} ({} atoms)", total_reduce_groups, total_reduce_atoms);
    println!("  Binary groups: {}", total_binary_groups);
    println!("  Unary groups: {}", total_unary_groups);
    println!("  Identity groups: {}", total_identity_groups);
    println!("  Phases with ReduceSum/Max: {} / {}", phases_with_reduce, num_phases);

    // Phase efficiency: how much time is "wasted" in barrier overhead?
    // Metric: what fraction of phases are "small" (< 1% of total work)?
    let one_pct = total_atoms / 100;
    let small_phases = phase_atom_counts.iter().filter(|&&a| a < one_pct).count();
    println!("  Phases with <1% of total work: {} / {}", small_phases, num_phases);

    // Compute effective parallelism: sum of (atoms in phase) / (max lane atoms in phase)
    let mut effective_lanes_sum = 0.0f64;
    let mut weighted_sum = 0.0f64;
    for phase in &plan.phases {
        let lane_atoms: Vec<u64> = phase.lane_work.iter()
            .map(|l| l.iter().map(|w| w.atom_count).sum::<u64>()).collect();
        let total_phase: u64 = lane_atoms.iter().sum();
        let max_lane = lane_atoms.iter().copied().max().unwrap_or(0);
        if max_lane > 0 {
            let eff = total_phase as f64 / max_lane as f64;
            effective_lanes_sum += eff;
            weighted_sum += eff * total_phase as f64;
        }
    }
    let avg_effective_lanes = if num_phases > 0 { effective_lanes_sum / num_phases as f64 } else { 0.0 };
    let weighted_effective_lanes = if total_atoms > 0 { weighted_sum / total_atoms as f64 } else { 0.0 };
    println!("  Average effective lanes (unweighted): {:.2}", avg_effective_lanes);
    println!("  Average effective lanes (atom-weighted): {:.2}", weighted_effective_lanes);

    // Total serialized work vs parallel work
    let max_lane_sum: u64 = plan.phases.iter().map(|phase| {
        phase.lane_work.iter()
            .map(|l| l.iter().map(|w| w.atom_count).sum::<u64>())
            .max().unwrap_or(0)
    }).sum();
    println!("  Total atoms: {:.3}B", total_atoms as f64 / 1e9);
    println!("  Critical path atoms (sum of max lane per phase): {:.3}B", max_lane_sum as f64 / 1e9);
    println!("  Theoretical speedup from {} lanes: {:.2}x (ideal: {}x)",
        num_lanes, total_atoms as f64 / max_lane_sum as f64, num_lanes);

    // Phase length histogram (atoms)
    println!("\n  Phase atom histogram:");
    let mut buckets: Vec<(&str, u64, u64, usize, u64)> = vec![
        ("<1K",     0,        1_000,       0, 0),
        ("1K-10K",  1_000,    10_000,      0, 0),
        ("10K-100K",10_000,   100_000,     0, 0),
        ("100K-1M", 100_000,  1_000_000,   0, 0),
        ("1M-10M",  1_000_000,10_000_000,  0, 0),
        ("10M-100M",10_000_000,100_000_000,0, 0),
        ("100M-1B", 100_000_000,1_000_000_000, 0, 0),
        (">1B",     1_000_000_000, u64::MAX,   0, 0),
    ];
    for &ac in &phase_atom_counts {
        for b in buckets.iter_mut() {
            if ac >= b.1 && ac < b.2 {
                b.3 += 1;
                b.4 += ac;
                break;
            }
        }
    }
    for (label, _, _, count, atoms) in &buckets {
        if *count > 0 {
            println!("    {:>10}: {:>4} phases, {:>12.3}B atoms ({:.1}%)",
                label, count, *atoms as f64 / 1e9,
                if total_atoms > 0 { *atoms as f64 / total_atoms as f64 * 100.0 } else { 0.0 });
        }
    }

    // Consecutive small phase runs (merging opportunities)
    println!("\n  Consecutive small-phase runs (phases with <10 groups):");
    let mut runs: Vec<(usize, usize)> = Vec::new(); // (start, length)
    let mut run_start = None;
    for i in 0..num_phases {
        if phase_group_counts[i] < 10 {
            if run_start.is_none() { run_start = Some(i); }
        } else {
            if let Some(s) = run_start {
                runs.push((s, i - s));
                run_start = None;
            }
        }
    }
    if let Some(s) = run_start { runs.push((s, num_phases - s)); }
    runs.sort_by(|a, b| b.1.cmp(&a.1));
    if runs.is_empty() {
        println!("    No consecutive small-phase runs found.");
    } else {
        for &(start, len) in runs.iter().take(10) {
            let total_run_atoms: u64 = (start..start+len).map(|i| phase_atom_counts[i]).sum();
            let total_run_groups: usize = (start..start+len).map(|i| phase_group_counts[i]).sum();
            println!("    phases {}-{}: {} phases, {} groups, {:.3}M atoms",
                start, start + len - 1, len, total_run_groups, total_run_atoms as f64 / 1e6);
        }
        let total_small_runs: usize = runs.len();
        let total_mergeable_phases: usize = runs.iter().map(|r| r.1).sum();
        println!("    {} runs totaling {} phases could potentially be merged", total_small_runs, total_mergeable_phases);
    }

    println!("\n========================================");
}
