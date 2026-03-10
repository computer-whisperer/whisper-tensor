//! Compile a real model through the v9 fused expression compiler and validate accuracy.
//!
//! Usage (RWKV 0.1B):
//!   cargo run --release --features cranelift,candle --example v9_model_test -- \
//!     /mnt/secondary/rwkv-7-world/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth
//!
//! Usage (ONNX):
//!   cargo run --release --features cranelift --example v9_model_test -- model.onnx

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::compiler::attempts::v9_fused_expr::pipeline as v9_pipeline;
use whisper_tensor::compiler::op_census;
use whisper_tensor::dtype::DType;
use whisper_tensor::graph::GlobalId;
use whisper_tensor::model::Model;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::tensor_rank::DynRank;
use whisper_tensor_import::identify_and_load;
use whisper_tensor_import::onnx_graph::WeightStorageStrategy;

/// Copy a NumericTensor's raw bytes into a buffer slot.
fn load_tensor_into_buffer(
    tensor: &NumericTensor<DynRank>,
    buf: &mut [u8],
) {
    let nd = tensor.to_ndarray().expect("can convert to ndarray");
    let bytes = nd.to_contiguous_bytes();
    let copy_len = bytes.len().min(buf.len());
    buf[..copy_len].copy_from_slice(&bytes[..copy_len]);
}

/// Read a buffer slot back as an NDArrayNumericTensor.
fn read_buffer_as_tensor(
    buf: &[u8],
    shape: &[usize],
    dtype: DType,
) -> NDArrayNumericTensor<DynRank> {
    let shape_u64: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
    let numel: usize = shape.iter().product::<usize>().max(1);
    let byte_size = dtype.size().unwrap_or(4);
    let needed = numel * byte_size;
    NDArrayNumericTensor::from_raw_data(&buf[..needed], dtype, shape_u64).unwrap()
}

/// Compare two tensors element-wise, return (max_abs_error, mean_abs_error, num_elements).
fn compare_tensors(
    a: &NDArrayNumericTensor<DynRank>,
    b: &NDArrayNumericTensor<DynRank>,
) -> (f64, f64, usize) {
    let a_f32 = a.cast(DType::F32).unwrap();
    let b_f32 = b.cast(DType::F32).unwrap();

    if let (NDArrayNumericTensor::F32(a_arr), NDArrayNumericTensor::F32(b_arr)) = (&a_f32, &b_f32) {
        // Force standard (C-order) layout before comparison.  ndarray's map() preserves
        // non-standard strides (e.g. from Transpose views), so as_slice_memory_order()
        // would return elements in different orders for transposed vs contiguous arrays.
        let a_std = a_arr.as_standard_layout();
        let b_std = b_arr.as_standard_layout();
        let a_slice = a_std.as_slice_memory_order().unwrap();
        let b_slice = b_std.as_slice_memory_order().unwrap();
        assert_eq!(a_slice.len(), b_slice.len(), "element count mismatch");

        let mut max_err: f64 = 0.0;
        let mut sum_err: f64 = 0.0;
        let n = a_slice.len();
        for i in 0..n {
            let diff = (a_slice[i] as f64 - b_slice[i] as f64).abs();
            max_err = max_err.max(diff);
            sum_err += diff;
        }
        (max_err, sum_err / n as f64, n)
    } else {
        panic!("cast to F32 failed");
    }
}

fn load_model(path: &Path) -> Arc<Model> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    // Deterministic RNG for reproducible GlobalIds across runs.
    let mut rng = wyrand::WyRand::new(1);

    let onnx_data = match ext {
        "pth" => {
            println!("Loading via RWKV7 pth → ONNX...");
            whisper_tensor_import::rwkv7::load_rwkv7_pth(path, WeightStorageStrategy::EmbeddedData)
                .unwrap()
        }
        _ => {
            println!("Loading via ONNX...");
            identify_and_load(path, WeightStorageStrategy::EmbeddedData).unwrap()
        }
    };
    Arc::new(Model::new_from_onnx(&onnx_data, &mut rng, None).unwrap())
}

fn main() {
    tracing_subscriber::fmt::init();

    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| {
            eprintln!("Usage: v9_model_test <model_path>");
            std::process::exit(1);
        });
    let path = Path::new(&model_path);

    if !path.exists() {
        eprintln!("Model file not found: {}", model_path);
        std::process::exit(1);
    }

    // ---- Load model ----
    println!("Loading model from {}...", model_path);
    let t0 = Instant::now();
    let model = load_model(path);
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f64());

    // ---- Generate MilliOpGraph ----
    println!("\nGenerating MilliOpGraph...");
    let t0 = Instant::now();
    let mut rng = wyrand::WyRand::new(42);
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
    println!("\nCollecting tensor shapes...");

    let input_info = model.get_input_tensor_info().unwrap();
    println!("  Model inputs: {} tensors", input_info.len());

    let sym_graph = model.get_symbolic_graph();
    let tensor_store = model.get_tensor_store();
    let mut inputs_by_name: HashMap<String, NumericTensor<DynRank>> = HashMap::new();

    for (name, (dtype, shape_dims)) in &input_info {
        let concrete_shape: Vec<usize> = shape_dims
            .iter()
            .map(|d| d.map(|v| v as usize).unwrap_or(4))
            .collect();
        let numel: usize = concrete_shape.iter().product();
        let shape_u64: Vec<u64> = concrete_shape.iter().map(|&d| d as u64).collect();

        let raw_bytes = vec![0u8; numel * dtype.size().unwrap_or(4)];
        let tensor: NumericTensor<DynRank> =
            NDArrayNumericTensor::from_raw_data(&raw_bytes, *dtype, shape_u64.clone())
                .unwrap()
                .into();
        inputs_by_name.insert(name.clone(), tensor);
    }

    // ---- Run interpreter (ground truth) ----
    let inputs_by_name_clone = inputs_by_name.clone();
    println!("\n  Running interpreter...");
    let t0 = Instant::now();
    let sym_result = whisper_tensor::backends::eval_backend::run(
        sym_graph,
        tensor_store,
        None,
        &mut EvalBackend::NDArray,
        &mut (),
        inputs_by_name.clone(),
    );
    let interp_outputs = match sym_result {
        Ok(outputs) => {
            println!("  Interpreter finished in {:.1}ms", t0.elapsed().as_secs_f64() * 1e3);
            println!("  {} outputs", outputs.len());
            outputs
        }
        Err(e) => {
            println!("  Interpreter failed: {}", e);
            return;
        }
    };

    // ---- Collect shapes from MilliOpGraph ----
    let tensors_by_name = sym_graph.get_tensors_by_name();
    let milli_inputs: HashMap<GlobalId, NumericTensor<DynRank>> = inputs_by_name
        .into_iter()
        .filter_map(|(name, tensor)| {
            tensors_by_name.get(&name).map(|id| (*id, tensor))
        })
        .collect();

    let initialized = sym_graph.get_initialized_tensors(tensor_store);
    let mut all_inputs: HashMap<GlobalId, NumericTensor<DynRank>> = milli_inputs;
    for (id, tensor) in initialized {
        all_inputs.insert(id, tensor);
    }

    // Translate external→internal IDs for buffer loading
    let all_inputs_internal: HashMap<GlobalId, &NumericTensor<DynRank>> = all_inputs
        .iter()
        .filter_map(|(ext_id, tensor)| {
            milli_graph.input_map.get(ext_id).map(|&int_id| (int_id, tensor))
        })
        .collect();

    // ---- Cross-check: milli graph vs symbolic graph (no buffer round-trip) ----
    println!("\n  Running milli graph interpreter (no buffers)...");
    let t0 = Instant::now();
    let milli_intermediates = milli_graph.collect_all_intermediate_values(&all_inputs)
        .expect("milli graph eval failed");
    println!("  Milli eval finished in {:.1}ms", t0.elapsed().as_secs_f64() * 1e3);

    // Compare milli outputs vs symbolic outputs
    let output_map_ref = milli_graph.output_map.as_ref().unwrap();
    let mut milli_vs_sym_fails = 0usize;
    for (name, interp_tensor) in &interp_outputs {
        let external_id = match tensors_by_name.get(name) {
            Some(id) => *id,
            None => continue,
        };
        let internal_id = match output_map_ref.iter().find(|(_, ext)| **ext == external_id) {
            Some((int, _)) => *int,
            None => continue,
        };
        if let Some(milli_tensor) = milli_intermediates.get(&internal_id) {
            let interp_nd = interp_tensor.to_ndarray().unwrap();
            let milli_nd = milli_tensor.to_ndarray().unwrap();
            let (max_err, mean_err, _n) = compare_tensors(&interp_nd, &milli_nd);
            if max_err > 1e-4 {
                println!("  MILLI≠SYM: {:40} max_err={:.2e} mean={:.2e} {:?}",
                    name, max_err, mean_err, milli_nd.dtype());
                milli_vs_sym_fails += 1;
            }
        }
    }
    if milli_vs_sym_fails == 0 {
        println!("  Milli graph matches symbolic graph for all outputs.");
    } else {
        println!("  {} outputs differ between milli and symbolic graph!", milli_vs_sym_fails);
    }

    println!("\n  Collecting shapes and dtypes ({} input tensors)...", all_inputs.len());
    let t0 = Instant::now();
    let (shapes, dtypes) = match milli_graph.collect_all_shapes_and_dtypes(&all_inputs) {
        Ok(x) => x,
        Err(e) => {
            println!("  Shape collection failed: {}", e);
            return;
        }
    };
    println!(
        "  Collected {} tensor shapes in {:.1}ms",
        shapes.len(),
        t0.elapsed().as_secs_f64() * 1e3
    );

    // Check dtype coverage
    {
        let n_bf16 = dtypes.values().filter(|&&dt| dt == DType::BF16).count();
        let n_f32 = dtypes.values().filter(|&&dt| dt == DType::F32).count();
        let n_other = dtypes.len() - n_bf16 - n_f32;
        let n_missing = shapes.len() - dtypes.len();
        eprintln!("  dtypes: {} BF16, {} F32, {} other, {} missing (default to F32)", n_bf16, n_f32, n_other, n_missing);
    }

    // Get internal output IDs for v9
    let final_output_ids: HashSet<GlobalId> = milli_graph
        .output_map
        .as_ref()
        .map(|m| m.keys().cloned().collect())
        .unwrap_or_default();
    println!("  Final outputs: {} tensors", final_output_ids.len());

    // Build reverse map: external_id -> internal_id (for reading v9 outputs)
    let output_map = milli_graph.output_map.as_ref().unwrap();
    let external_to_internal: HashMap<GlobalId, GlobalId> = output_map
        .iter()
        .map(|(&internal, &external)| (external, internal))
        .collect();

    // Build name -> external_id for output comparison
    let name_to_external: HashMap<&str, GlobalId> = tensors_by_name
        .iter()
        .map(|(name, &id)| (name.as_str(), id))
        .collect();

    // ---- Compile hybrid execution plan ----
    println!("\nAttempting v9 hybrid compilation...");
    let t0 = Instant::now();
    let plan = match v9_pipeline::compile_hybrid(&milli_graph, &shapes, &final_output_ids, 0, &dtypes) {
        Ok(p) => {
            println!(
                "  SUCCESS! Compiled in {:.1}ms, {} kernels, {} interpreted ops, {} steps",
                t0.elapsed().as_secs_f64() * 1e3,
                p.compiled.kernels.len(),
                p.interpreted_ops.len(),
                p.steps.len(),
            );
            p
        }
        Err(e) => {
            println!(
                "  FAILED after {:.1}ms: {}",
                t0.elapsed().as_secs_f64() * 1e3,
                e
            );
            return;
        }
    };

    // Determinism check: fingerprint kernel ordering and step sequence
    {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        for k in &plan.compiled.kernels {
            k.output_tensor.hash(&mut h);
            for &inp in &k.input_tensors {
                inp.hash(&mut h);
            }
        }
        println!("  kernel fingerprint: {:016x} ({} kernels)", h.finish(), plan.compiled.kernels.len());
    }

    let layout = &plan.compiled.layout;

    // ---- Allocate and fill v9 buffers (external inputs only) ----
    println!("\nLoading tensor data into v9 buffers...");
    let t0 = Instant::now();
    let mut bufs: Vec<Vec<u8>> = (0..layout.num_buffers)
        .map(|_| Vec::new())
        .collect();

    // Allocate all buffers
    for (id, &size) in &layout.tensor_sizes {
        let idx = layout.tensor_index[id];
        let dt = layout.dtype_of(id);
        let byte_size = dt.size().unwrap_or(4);
        if bufs[idx].is_empty() {
            bufs[idx] = vec![0u8; size * byte_size];
        }
    }

    // Load only external inputs (weights + runtime inputs) — interpreted ops
    // will fill in intermediate values during execution.
    let mut loaded_count = 0usize;
    let mut loaded_bytes = 0usize;
    for (int_id, tensor) in &all_inputs_internal {
        if let Some(&idx) = layout.tensor_index.get(int_id) {
            load_tensor_into_buffer(tensor, &mut bufs[idx]);
            loaded_count += 1;
            loaded_bytes += bufs[idx].len();
        }
    }
    println!(
        "  Loaded {} external tensors ({:.1} MB) in {:.1}ms",
        loaded_count,
        loaded_bytes as f64 / 1e6,
        t0.elapsed().as_secs_f64() * 1e3
    );

    // ---- Execute v9 hybrid plan ----
    let interp_only = std::env::var("V9_INTERP_ONLY").is_ok();
    eprint!("\nExecuting v9 hybrid plan ({})...", if interp_only { "interp-only" } else { "serial" });
    let mut ptrs: Vec<*mut f32> = bufs
        .iter_mut()
        .map(|v| v.as_mut_ptr() as *mut f32)
        .collect();

    let t0 = Instant::now();
    // Execute step-by-step, checking each step's output against milli ground truth
    let debug_steps = std::env::var("V9_DEBUG_STEPS").is_ok();
    let mut first_error_step = None;
    for (si, step) in plan.steps.iter().enumerate() {
        match step {
            v9_pipeline::ExecStep::Kernel(ki) => {
                unsafe { plan.compiled.kernels[*ki].execute(ptrs.as_ptr()) };
                if debug_steps {
                    let kernel = &plan.compiled.kernels[*ki];
                    let out_id = kernel.output_tensor;
                    if let (Some(&buf_idx), Some(shape), Some(milli_val)) = (
                        layout.tensor_index.get(&out_id),
                        shapes.get(&out_id),
                        milli_intermediates.get(&out_id),
                    ) {
                        let dtype = layout.dtype_of(&out_id);
                        let buf_tensor = read_buffer_as_tensor(&bufs[buf_idx], shape, dtype);
                        let milli_nd = milli_val.to_ndarray().unwrap();
                        let milli_nd = milli_nd.cast(dtype).unwrap_or(milli_nd);
                        let (max_err, mean_err, n) = compare_tensors(&milli_nd, &buf_tensor);
                        if max_err > 1e-4 {
                            // Check if all inputs are correct (fresh error)
                            let mut max_inp_err: f64 = 0.0;
                            let mut num_reds = 0;
                            for inp_id in &kernel.input_tensors {
                                if let (Some(&inp_idx), Some(inp_shape)) = (
                                    layout.tensor_index.get(inp_id),
                                    shapes.get(inp_id),
                                ) {
                                    let inp_dtype = layout.dtype_of(inp_id);
                                    if let Some(milli_inp) = milli_intermediates.get(inp_id) {
                                        let inp_buf = read_buffer_as_tensor(&bufs[inp_idx], inp_shape, inp_dtype);
                                        let milli_inp_nd = milli_inp.to_ndarray().unwrap();
                                        let milli_inp_nd = milli_inp_nd.cast(inp_dtype).unwrap_or(milli_inp_nd);
                                        let (inp_err, _, _) = compare_tensors(&milli_inp_nd, &inp_buf);
                                        max_inp_err = max_inp_err.max(inp_err);
                                    }
                                }
                            }
                            // Count reductions from the pattern log (approximate via input count)
                            let is_fresh = max_inp_err < 1e-6;
                            if is_fresh {
                                let inp_dtypes: Vec<_> = kernel.input_tensors.iter()
                                    .map(|id| format!("{:?}={:?}", id, layout.dtype_of(id)))
                                    .collect();
                                eprintln!("  FRESH ERROR step {} K{}: {:?} {} elems {:?} shape={:?} max_err={:.2e} mean={:.2e} inputs=[{}]",
                                    si, ki, out_id, n, dtype, shape, max_err, mean_err, inp_dtypes.join(", "));
                                // Print first few differing elements
                                let milli_f32 = milli_nd.cast(dtype).unwrap_or(milli_nd.clone());
                                let buf_f32 = buf_tensor.cast(dtype).unwrap_or(buf_tensor.clone());
                                if let (NDArrayNumericTensor::F32(m), NDArrayNumericTensor::F32(b)) =
                                    (&milli_f32, &buf_f32)
                                {
                                    let ms = m.as_standard_layout();
                                    let bs = b.as_standard_layout();
                                    let ms = ms.as_slice_memory_order().unwrap();
                                    let bs = bs.as_slice_memory_order().unwrap();
                                    let mut printed = 0;
                                    for i in 0..ms.len().min(bs.len()) {
                                        let diff = (ms[i] as f64 - bs[i] as f64).abs();
                                        if diff > 1e-6 && printed < 5 {
                                            eprintln!("    elem[{}]: milli={:.8e} v9={:.8e} diff={:.2e}",
                                                i, ms[i], bs[i], diff);
                                            printed += 1;
                                        }
                                    }
                                }
                            }
                            if first_error_step.is_none() {
                                first_error_step = Some(si);
                            }
                        }
                    }
                }
            }
            v9_pipeline::ExecStep::Eval(ei) => {
                plan.run_interpreted_op(*ei, &ptrs);
                if debug_steps && first_error_step.is_none() {
                    let iop = &plan.interpreted_ops[*ei];
                    for out_id in &iop.output_ids {
                        if let (Some(&buf_idx), Some(shape), Some(milli_val)) = (
                            layout.tensor_index.get(out_id),
                            shapes.get(out_id),
                            milli_intermediates.get(out_id),
                        ) {
                            let dtype = layout.dtype_of(out_id);
                            let buf_tensor = read_buffer_as_tensor(&bufs[buf_idx], shape, dtype);
                            let milli_nd = milli_val.to_ndarray().unwrap();
                            let milli_nd = milli_nd.cast(dtype).unwrap_or(milli_nd);
                            let (max_err, _, n) = compare_tensors(&milli_nd, &buf_tensor);
                            if max_err > 1e-4 {
                                eprintln!("  FIRST ERROR at step {} (Eval {}): {:?} {} elems {:?} shape={:?} max_err={:.2e}",
                                    si, ei, out_id, n, dtype, shape, max_err);
                                first_error_step = Some(si);
                            }
                        }
                    }
                }
            }
        }
    }
    eprintln!(" {:.1}ms", t0.elapsed().as_secs_f64() * 1e3);

    // ---- Compare ALL intermediate values: buffer round-trip vs no-buffer milli ----
    {
        println!("\n=== Buffer Round-Trip Check (vs milli intermediates) ===");
        let mut first_mismatch = true;
        let mut total_mismatches = 0usize;
        for (&int_id, milli_tensor) in &milli_intermediates {
            if let Some(&buf_idx) = layout.tensor_index.get(&int_id) {
                let shape = match shapes.get(&int_id) {
                    Some(s) => s,
                    None => continue,
                };
                let dtype = layout.dtype_of(&int_id);
                let buf_tensor = read_buffer_as_tensor(&bufs[buf_idx], shape, dtype);
                let milli_nd = milli_tensor.to_ndarray().unwrap();
                // Cast milli to match layout dtype for fair comparison
                let milli_nd = milli_nd.cast(dtype).unwrap_or(milli_nd);
                let (max_err, mean_err, n) = compare_tensors(&milli_nd, &buf_tensor);
                if max_err > 1e-4 {
                    total_mismatches += 1;
                    if first_mismatch || total_mismatches <= 10 {
                        let actual_dtype = milli_tensor.to_ndarray().unwrap().dtype();
                        println!("  MISMATCH {:?}: {} elems, max_err={:.2e}, mean={:.2e}, layout={:?} actual={:?} shape={:?}",
                            int_id, n, max_err, mean_err, dtype, actual_dtype, shape);
                        first_mismatch = false;
                    }
                }
            }
        }
        println!("  {} / {} tensors differ", total_mismatches, milli_intermediates.len());
    }

    // ---- Compare outputs ----
    println!("\n=== Accuracy Comparison ===");
    let mut total_checked = 0usize;
    let mut total_exact = 0usize;
    let mut worst_name = String::new();
    let mut worst_max_err: f64 = 0.0;
    let mut all_pass = true;

    for (name, interp_tensor) in &interp_outputs {
        // Map: output name -> external id -> internal id -> buffer index
        let external_id = match name_to_external.get(name.as_str()) {
            Some(id) => *id,
            None => continue,
        };
        let internal_id = match external_to_internal.get(&external_id) {
            Some(id) => *id,
            None => continue,
        };
        let buf_idx = match layout.tensor_index.get(&internal_id) {
            Some(&idx) => idx,
            None => continue,
        };

        let shape = match shapes.get(&internal_id) {
            Some(s) => s,
            None => continue,
        };
        let dtype = layout.dtype_of(&internal_id);

        // Read v9 output
        let v9_tensor = read_buffer_as_tensor(&bufs[buf_idx], shape, dtype);

        // Get interpreter output as ndarray
        let interp_nd = interp_tensor.to_ndarray().unwrap();

        let (max_err, mean_err, n) = compare_tensors(&interp_nd, &v9_tensor);
        total_checked += 1;

        let status = if max_err < 1e-4 {
            total_exact += 1;
            "OK"
        } else if max_err < 1e-2 {
            "WARN"
        } else {
            all_pass = false;
            "FAIL"
        };

        if max_err > worst_max_err {
            worst_max_err = max_err;
            worst_name = name.clone();
        }

        if status != "OK" || total_checked <= 10 {
            println!(
                "  [{:4}] {:40} {:>5} elems  max_err={:.2e}  mean_err={:.2e}  {}",
                status, name, n, max_err, mean_err, dtype
            );
        }
    }

    println!("\n  {}/{} outputs checked", total_checked, interp_outputs.len());
    println!("  {}/{} within 1e-4 tolerance", total_exact, total_checked);
    println!("  Worst: '{}' max_err={:.2e}", worst_name, worst_max_err);

    if all_pass {
        println!("  PASS: All outputs within acceptable tolerance.");
    } else {
        println!("  FAIL: Some outputs exceed error threshold!");
    }

    // ---- Benchmark: serial vs parallel execution timing ----
    if std::env::var("V9_BENCH").is_ok() {
        let iters: u32 = std::env::var("V9_BENCH_ITERS")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(3);

        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        // Re-load buffers fresh for benchmark
        let reload = || -> Vec<Vec<u8>> {
            let mut fresh: Vec<Vec<u8>> = (0..layout.num_buffers)
                .map(|_| Vec::new())
                .collect();
            for (id, &size) in &layout.tensor_sizes {
                let idx = layout.tensor_index[id];
                let dt = layout.dtype_of(id);
                let byte_size = dt.size().unwrap_or(4);
                if fresh[idx].is_empty() {
                    fresh[idx] = vec![0u8; size * byte_size];
                }
            }
            for (int_id, tensor) in &all_inputs_internal {
                if let Some(&idx) = layout.tensor_index.get(int_id) {
                    load_tensor_into_buffer(tensor, &mut fresh[idx]);
                }
            }
            fresh
        };

        // Serial benchmark
        println!("\n=== Benchmark: {} iters ===", iters);
        {
            // Warmup
            let mut warm_bufs = reload();
            let mut warm_ptrs: Vec<*mut f32> = warm_bufs.iter_mut().map(|v| v.as_mut_ptr() as *mut f32).collect();
            unsafe { plan.execute(&mut warm_ptrs) };

            let t = Instant::now();
            for _ in 0..iters {
                let mut b = reload();
                let mut p: Vec<*mut f32> = b.iter_mut().map(|v| v.as_mut_ptr() as *mut f32).collect();
                unsafe { plan.execute(&mut p) };
            }
            let serial_avg = t.elapsed() / iters;
            println!("  serial:       {:>10.1}ms avg ({} iters)", serial_avg.as_secs_f64() * 1e3, iters);
        }

        // Parallel benchmark
        {
            // Warmup
            let mut warm_bufs = reload();
            let warm_ptrs: Vec<*mut f32> = warm_bufs.iter_mut().map(|v| v.as_mut_ptr() as *mut f32).collect();
            unsafe { plan.execute_parallel(&warm_ptrs, num_threads) };

            let t = Instant::now();
            for _ in 0..iters {
                let mut b = reload();
                let p: Vec<*mut f32> = b.iter_mut().map(|v| v.as_mut_ptr() as *mut f32).collect();
                unsafe { plan.execute_parallel(&p, num_threads) };
            }
            let par_avg = t.elapsed() / iters;
            println!("  parallel({}t): {:>10.1}ms avg ({} iters)", num_threads, par_avg.as_secs_f64() * 1e3, iters);
        }
    }
}
