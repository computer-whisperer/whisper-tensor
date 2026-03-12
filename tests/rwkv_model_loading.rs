use std::collections::HashMap;
use std::path::PathBuf;

use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::model::{Model, ModelExecutionRuntime};

/// Returns `true` if the path is a real model file, `false` if it is an LFS pointer.
fn is_real_model_file(path: &std::path::Path) -> bool {
    use std::fs;
    let Ok(md) = fs::metadata(path) else {
        return false;
    };
    // LFS pointer files are tiny (typically < 200 bytes) and start with a known header
    if md.len() < 1024
        && let Ok(bytes) = fs::read(path)
        && bytes.starts_with(b"version https://git-lfs.github.com/spec/v1")
    {
        return false;
    }
    true
}

fn find_rwkv_pth() -> Option<PathBuf> {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.push("test_models");
    let rd = std::fs::read_dir(&dir).ok()?;
    for entry in rd.flatten() {
        let p = entry.path();
        if p.is_file()
            && let Some(ext) = p.extension()
            && ext == "pth"
            && p.file_name()
                .and_then(|s| s.to_str())
                .map(|s| s.to_lowercase())
                .map(|s| s.contains("rwkv") || s.contains("rwkv7") || s.contains("world"))
                .unwrap_or(false)
        {
            return Some(p);
        }
    }
    None
}

#[test]
fn rwkv01b_model_loads() {
    // Use the vendored RWKV 0.1B pth in test_models via importer identify_and_load
    let Some(pth_path) = find_rwkv_pth() else {
        eprintln!("Skipping: no RWKV .pth found under test_models/");
        return;
    };
    if !is_real_model_file(&pth_path) {
        eprintln!(
            "Skipping: {} is a Git LFS pointer, not the real model",
            pth_path.display()
        );
        return;
    }
    // Use EmbeddedData output method to avoid external file dependencies
    let strategy = whisper_tensor_import::onnx_graph::WeightStorageStrategy::EmbeddedData;
    let onnx_bytes = whisper_tensor_import::identify_and_load(&pth_path, strategy)
        .expect("import rwkv7 to onnx");
    let mut rng = rand::rng();
    let model = Model::new_from_onnx(&onnx_bytes, &mut rng, None).expect("model loads");

    // Ensure we can construct Eval backend and load constant tensors (which should trigger
    // lazy external loading for any external_data entries)
    let mut eval = EvalBackend::NDArray;
    let mut cache = whisper_tensor::backends::ModelLoadedTensorCache::default();
    model
        .load_tensors(&mut cache, &mut eval)
        .expect("tensors load");

    // Check graph has at least one input and one output
    assert!(
        !model.get_symbolic_graph().get_inputs().is_empty(),
        "model has inputs"
    );
    assert!(
        !model.get_symbolic_graph().get_outputs().is_empty(),
        "model has outputs"
    );
}

#[test]
fn rwkv01b_single_step_runs_shape_sanity() {
    // This is a smoke test: we try to run a single forward step with zeroed inputs
    // respecting the model's declared input shapes (with small batch/time if symbolic).
    let Some(pth_path) = find_rwkv_pth() else {
        eprintln!("Skipping: no RWKV .pth found under test_models/");
        return;
    };
    if !is_real_model_file(&pth_path) {
        eprintln!(
            "Skipping: {} is a Git LFS pointer, not the real model",
            pth_path.display()
        );
        return;
    }
    let onnx_bytes = whisper_tensor_import::identify_and_load(
        &pth_path,
        whisper_tensor_import::onnx_graph::WeightStorageStrategy::EmbeddedData,
    )
    .expect("import rwkv7 to onnx");
    let mut rng = rand::rng();
    let model = Model::new_from_onnx(&onnx_bytes, &mut rng, None).expect("model loads");

    //let mut eval = EvalBackend::NDArray;
    let mut observer = ();

    // Prepare zeroed inputs by introspecting model input tensor info
    let input_infos = model.get_input_tensor_info().expect("introspect inputs");
    use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
    use whisper_tensor::{dtype::DType, numeric_tensor::NumericTensor};

    let mut inputs = std::collections::HashMap::new();
    for (name, (dtype, shape_desc)) in input_infos.into_iter() {
        // Replace unknown dims with small test sizes
        let mut shape: Vec<u64> = shape_desc.into_iter().map(|d| d.unwrap_or(1)).collect();
        // Clamp sequence or time-like dims if obviously large
        for s in &mut shape {
            if *s == 0 {
                *s = 1;
            }
        }
        let numel: usize = shape.iter().product::<u64>() as usize;
        let zeros = match dtype {
            DType::F32 => vec![0u8; numel * 4],
            DType::BF16 | DType::F16 => vec![0u8; numel * 2],
            DType::I64 | DType::U64 => vec![0u8; numel * 8],
            DType::I32 | DType::U32 | DType::U16 | DType::I16 => vec![0u8; numel * 4],
            DType::U8 | DType::I8 | DType::BOOL => vec![0u8; numel],
            DType::F64 => vec![0u8; numel * 8],
            DType::STRING => panic!("string input not supported in test"),
            DType::F8E4M3 | DType::F8E5M2 => vec![0u8; numel],
            DType::Packed(_) => panic!("packed dtype not supported in test"),
        };
        let nd = NDArrayNumericTensor::from_raw_data(&zeros, dtype, shape.clone())
            .expect("build zero tensor");
        inputs.insert(name, NumericTensor::NDArray(nd));
    }

    // Run a single step on Eval runtime
    let mut runtime = ModelExecutionRuntime::Eval(EvalBackend::NDArray);
    let outputs = model
        .run(inputs, &mut observer, &mut runtime)
        .expect("run ok");
    assert!(!outputs.is_empty(), "has outputs");

    // Basic sanity: each output tensor matches declared shape/dtype automatically in run()
}

#[test]
fn rwkv01b_model_loads_with_binfile() {
    // Validate that the BinFile strategy works by writing weights to a temp file
    let Some(pth_path) = find_rwkv_pth() else {
        eprintln!("Skipping: no RWKV .pth found under test_models/");
        return;
    };
    if !is_real_model_file(&pth_path) {
        eprintln!(
            "Skipping: {} is a Git LFS pointer, not the real model",
            pth_path.display()
        );
        return;
    }

    let tempdir = tempfile::tempdir().expect("create tempdir");
    let bin_path = tempdir.path().join("weights.bin");

    // Build ONNX with external_data pointing at weights.bin (relative name is embedded)
    let onnx_bytes = whisper_tensor_import::identify_and_load(
        &pth_path,
        whisper_tensor_import::onnx_graph::WeightStorageStrategy::BinFile(bin_path.clone()),
    )
    .expect("import rwkv7 to onnx (BinFile)");

    // Ensure the side-channel bin file exists and is non-empty
    let meta = std::fs::metadata(&bin_path).expect("bin file exists");
    assert!(meta.len() > 0, "bin file should be non-empty");

    // The ONNX external_data 'location' is just the file name. Switch CWD to the tempdir
    // so the model loader can find the side-channel file when lazily loading tensors.
    let old_cwd = std::env::current_dir().expect("get cwd");
    std::env::set_current_dir(tempdir.path()).expect("set cwd to tempdir");

    let mut rng = rand::rng();
    let model =
        Model::new_from_onnx(&onnx_bytes, &mut rng, None).expect("model loads from onnx bytes");

    // Trigger lazy tensor loading via the eval path
    let mut eval = EvalBackend::NDArray;
    let mut cache = whisper_tensor::backends::ModelLoadedTensorCache::default();
    model
        .load_tensors(&mut cache, &mut eval)
        .expect("tensors load (external data)");

    // Restore CWD
    std::env::set_current_dir(old_cwd).expect("restore cwd");
}

#[cfg(feature = "candle")]
#[test]
fn rwkv01b_model_loads_with_origin_reference() {
    // Validate that the OriginReference strategy works by keeping weights in the original .pth
    let Some(pth_path) = find_rwkv_pth() else {
        eprintln!("Skipping: no RWKV .pth found under test_models/");
        return;
    };
    if !is_real_model_file(&pth_path) {
        eprintln!(
            "Skipping: {} is a Git LFS pointer, not the real model",
            pth_path.display()
        );
        return;
    }

    // Build ONNX with external_data entries pointing to the original .pth tensors
    let onnx_bytes = whisper_tensor_import::identify_and_load(
        &pth_path,
        whisper_tensor_import::onnx_graph::WeightStorageStrategy::OriginReference,
    )
    .expect("import rwkv7 to onnx (OriginReference)");

    // No cwd or symlink manipulation needed: location is absolute inside ONNX now
    let mut rng = rand::rng();
    let model =
        Model::new_from_onnx(&onnx_bytes, &mut rng, None).expect("model loads from onnx bytes");

    // Trigger lazy tensor loading via the eval path (fetch tensors from the .pth via candle)
    let mut eval = EvalBackend::NDArray;
    let mut cache = whisper_tensor::backends::ModelLoadedTensorCache::default();
    model
        .load_tensors(&mut cache, &mut eval)
        .expect("tensors load (origin reference pth)");

    // Basic sanity checks
    assert!(
        !model.get_symbolic_graph().get_inputs().is_empty(),
        "model has inputs"
    );
    assert!(
        !model.get_symbolic_graph().get_outputs().is_empty(),
        "model has outputs"
    );
}

/// Full end-to-end integrity check: evaluate the RWKV 0.1B model through
/// both the MilliOpGraph interpreter and the NanoGraph scalar eval, then
/// compare every output element.
#[test]
#[ignore]
fn rwkv01b_nano_graph_integrity() {
    use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
    use whisper_tensor::graph::GlobalId;
    use whisper_tensor::nano_graph::{eval::NanoEval, lower};
    use whisper_tensor::numeric_tensor::NumericTensor;
    use whisper_tensor::tensor_info::TensorInfo;
    use whisper_tensor::{DynRank, dtype::DType};

    let Some(pth_path) = find_rwkv_pth() else {
        eprintln!("Skipping: no RWKV .pth found under test_models/");
        return;
    };
    if !is_real_model_file(&pth_path) {
        eprintln!("Skipping: {} is a Git LFS pointer", pth_path.display());
        return;
    }

    // ---- Load model ----
    let t0 = std::time::Instant::now();
    let onnx_bytes = whisper_tensor_import::identify_and_load(
        &pth_path,
        whisper_tensor_import::onnx_graph::WeightStorageStrategy::EmbeddedData,
    )
    .expect("import rwkv7 to onnx");
    let mut rng = rand::rng();
    let model = Model::new_from_onnx(&onnx_bytes, &mut rng, None).expect("model loads");
    eprintln!("  Model loaded in {:.1}s", t0.elapsed().as_secs_f64());

    // ---- Generate MilliOpGraph ----
    let t0 = std::time::Instant::now();
    let milli = model.get_symbolic_graph().generate_milli_graph(&mut rng);
    eprintln!("  Milli graph generated in {:.1}ms", t0.elapsed().as_secs_f64() * 1e3);

    // ---- Create input tensors ----
    let input_info = model.get_input_tensor_info().expect("introspect inputs");
    let sym_graph = model.get_symbolic_graph();
    let tensor_store = model.get_tensor_store();
    let tensors_by_name = sym_graph.get_tensors_by_name();

    // User inputs: small random tensors (avoid degenerate zeros).
    let mut input_tensors: HashMap<GlobalId, NumericTensor<DynRank>> = HashMap::new();
    for (name, (dtype, shape_desc)) in &input_info {
        let shape: Vec<u64> = shape_desc.iter().map(|d| d.unwrap_or(1)).collect();
        let numel: usize = shape.iter().product::<u64>().max(1) as usize;
        eprintln!("  Input '{}': {:?} {:?} ({} elements)", name, dtype, shape, numel);
        // Use small non-zero values: 0.01 for floats, 1 for ints.
        let data: Vec<u8> = match dtype {
            DType::F32 => {
                let v: Vec<f32> = (0..numel).map(|i| 0.01 * (i as f32 + 1.0)).collect();
                v.iter().flat_map(|x| x.to_le_bytes()).collect()
            }
            DType::I64 => {
                let v: Vec<i64> = (0..numel).map(|i| (i as i64) + 1).collect();
                v.iter().flat_map(|x| x.to_le_bytes()).collect()
            }
            DType::I32 => {
                let v: Vec<i32> = (0..numel).map(|i| (i as i32) + 1).collect();
                v.iter().flat_map(|x| x.to_le_bytes()).collect()
            }
            _ => {
                // Fallback: use small non-zero bytes.
                let elem_size = match dtype {
                    DType::BF16 | DType::F16 => 2,
                    DType::U64 => 8,
                    DType::U32 | DType::U16 | DType::I16 => 4,
                    DType::U8 | DType::I8 | DType::BOOL => 1,
                    DType::F64 => 8,
                    _ => panic!("unsupported dtype {:?}", dtype),
                };
                vec![1u8; numel * elem_size]
            }
        };
        let nd = NDArrayNumericTensor::from_raw_data(&data, *dtype, shape)
            .expect("build input tensor");
        if let Some(&id) = tensors_by_name.get(name) {
            input_tensors.insert(id, NumericTensor::NDArray(nd));
        }
    }

    // Model weights (Numeric).
    let weight_tensors = sym_graph.get_initialized_tensors(tensor_store);

    // ---- Eval through MilliOpGraph (capturing all intermediates) ----
    use whisper_tensor::milli_graph::observer::MilliOpGraphObserver;
    use std::time::Instant;

    struct CapturingObserver {
        intermediates: HashMap<GlobalId, NumericTensor<DynRank>>,
    }
    impl MilliOpGraphObserver for CapturingObserver {
        fn on_tensor_assigned(
            &mut self,
            tensor_path: &[GlobalId],
            tensor: &NumericTensor<DynRank>,
            _backend: &mut EvalBackend,
        ) {
            self.intermediates.insert(tensor_path[0], tensor.clone());
        }
        fn on_node_executed(
            &mut self, _: &[GlobalId], _: Instant, _: Instant, _: &mut EvalBackend,
        ) {}
    }

    let t0 = Instant::now();
    let mut backend = EvalBackend::NDArray;
    let mut observer = CapturingObserver { intermediates: HashMap::new() };

    // milli.eval() wants external IDs.
    let mut milli_inputs: HashMap<GlobalId, NumericTensor<DynRank>> = HashMap::new();
    for (id, t) in &input_tensors {
        milli_inputs.insert(*id, t.clone());
    }
    for (id, t) in &weight_tensors {
        milli_inputs.insert(*id, t.clone());
    }

    let milli_outputs: HashMap<GlobalId, NumericTensor<DynRank>> = milli
        .eval(&milli_inputs, &mut observer, &mut backend)
        .expect("milli eval")
        .collect();
    let milli_intermediates = observer.intermediates;
    eprintln!(
        "  Milli eval done in {:.1}s ({} outputs, {} intermediates)",
        t0.elapsed().as_secs_f64(),
        milli_outputs.len(),
        milli_intermediates.len()
    );

    // ---- Lower to NanoGraph ----
    // lower_with_info wants external IDs (it maps ext→int internally).
    let mut lower_infos: HashMap<GlobalId, TensorInfo> = HashMap::new();
    for (id, t) in &input_tensors {
        lower_infos.insert(*id, TensorInfo::from(t.clone()));
    }
    for (id, t) in &weight_tensors {
        lower_infos.insert(*id, TensorInfo::from(t.clone()));
    }

    let t0 = std::time::Instant::now();
    let result = lower::lower_with_info(&milli, &lower_infos).expect("lower");
    eprintln!(
        "  Lowered in {:.1}ms  (atoms: {}, unsupported: {})",
        t0.elapsed().as_secs_f64() * 1e3,
        result.graph.num_atoms(),
        result.unsupported.len()
    );
    if !result.unsupported.is_empty() {
        eprintln!("  Unsupported ops: {:?}", result.unsupported_details);
    }

    // ---- Build NanoGraph overrides ----
    // Start with the numeric overrides from lowering (weights + constant-folded values).
    let t0 = std::time::Instant::now();
    let mut overrides = result.numeric_overrides;
    eprintln!(
        "  {} numeric overrides from lowering",
        overrides.len()
    );

    // Add user input values on top (these are Shaped, not Numeric, so not in numeric_overrides).
    use whisper_tensor::numeric_scalar::NumericScalar;
    let tensor_to_scalars = |t: &NumericTensor<DynRank>| -> Vec<NumericScalar> {
        let mut be = EvalBackend::NDArray;
        let dtype = t.dtype();
        let f32t = t.cast(DType::F32, &mut be).unwrap();
        let flat = f32t.flatten().unwrap();
        let v: Vec<f32> = flat.to_ndarray().unwrap().try_into().unwrap();
        v.into_iter().map(|x| NumericScalar::F32(x).cast_to(dtype)).collect()
    };

    for (ext_id, tensor) in &input_tensors {
        let Some(&int_id) = milli.input_map.get(ext_id) else { continue };
        let Some(tam) = result.tensor_map.get(&int_id) else { continue };
        let scalars = tensor_to_scalars(tensor);
        assert_eq!(scalars.len(), tam.count as usize,
            "Input {:?} count mismatch: {} vs {}", ext_id, scalars.len(), tam.count);
        for (i, val) in scalars.into_iter().enumerate() {
            overrides.insert(tam.base_id.0 + i as u32, val);
        }
    }
    eprintln!(
        "  Total {} overrides in {:.1}ms",
        overrides.len(),
        t0.elapsed().as_secs_f64() * 1e3
    );

    // ---- Eval NanoGraph ----
    let t0 = std::time::Instant::now();
    let nano_eval = NanoEval::eval_debug(&result.graph, &overrides);
    eprintln!("  Nano eval done in {:.1}s", t0.elapsed().as_secs_f64());

    // ---- Compare ALL intermediate tensors ----
    // The observer captured intermediates keyed by internal GlobalId.
    // tensor_map also uses internal GlobalIds.
    // Walk milli op ordering to compare tensors in topological order.
    let tensor_to_f64 = |t: &NumericTensor<DynRank>| -> Vec<f64> {
        let mut be = EvalBackend::NDArray;
        let f32t = t.cast(DType::F32, &mut be).unwrap();
        let flat = f32t.flatten().unwrap();
        let v: Vec<f32> = flat.to_ndarray().unwrap().try_into().unwrap();
        v.into_iter().map(|x| x as f64).collect()
    };

    let mut total_tensors = 0usize;
    let mut total_elements = 0usize;
    let mut max_rel_err = 0.0f64;
    let mut first_divergent: Option<(GlobalId, usize, f64, f64)> = None;

    // Sort tensor_map entries by base_id (topological order in nano graph).
    let mut sorted_tensors: Vec<_> = result.tensor_map.iter().collect();
    sorted_tensors.sort_by_key(|(_, tam)| tam.base_id.0);
    for (int_id, tam) in &sorted_tensors {
        if !tam.sym_dims.is_empty() {
            continue; // Skip symbolic-dim tensors
        }
        let Some(milli_tensor) = milli_intermediates.get(int_id) else {
            continue; // Input tensor or not captured
        };

        let milli_flat = tensor_to_f64(milli_tensor);
        if milli_flat.len() != tam.count as usize {
            eprintln!(
                "  SIZE MISMATCH: {:?} milli={} nano={}",
                int_id, milli_flat.len(), tam.count
            );
            continue;
        }

        total_tensors += 1;
        for (i, &m) in milli_flat.iter().enumerate() {
            let n = nano_eval.get(tam.base_id.offset(i as u32));
            let diff = (m - n).abs();
            let rel = diff / m.abs().max(1e-10);
            if rel > max_rel_err {
                max_rel_err = rel;
            }
            total_elements += 1;

            // Report first significant divergence.
            let tol = 1e-2 * m.abs().max(1.0);
            if first_divergent.is_none() && diff > tol {
                first_divergent = Some((**int_id, i, m, n));
                // Find which milli op produced this tensor.
                use whisper_tensor::graph::{Graph, Node};
                let producing_op = milli.node_ids().find_map(|op_id| {
                    let node = milli.get_node_by_id(&op_id).unwrap();
                    if node.outputs().any(|o| o == **int_id) {
                        Some((op_id, format!("{}", node.op_kind())))
                    } else {
                        None
                    }
                });
                eprintln!(
                    "  FIRST DIVERGENCE (after {} clean tensors): tensor {:?} (base_id={}) element {}",
                    total_tensors - 1, int_id, tam.base_id.0, i
                );
                if let Some((op_id, op_kind)) = &producing_op {
                    eprintln!("    produced by: {} (op {:?})", op_kind, op_id);
                    // Check inputs to the producing op, and trace back one more level.
                    let node = milli.get_node_by_id(op_id).unwrap();
                    for inp_id in node.inputs() {
                        if let Some(inp_tam) = result.tensor_map.get(&inp_id) {
                            if !inp_tam.sym_dims.is_empty() {
                                eprintln!("    input {:?}: {} atoms, sym_dims={:?} (skipped)",
                                    inp_id, inp_tam.count, inp_tam.sym_dims);
                                continue;
                            }
                            let Some(inp_milli) = milli_intermediates.get(&inp_id) else {
                                eprintln!("    input {:?}: {} atoms (no milli intermediate)",
                                    inp_id, inp_tam.count);
                                continue;
                            };
                            {
                                let inp_flat = tensor_to_f64(inp_milli);
                                let mut inp_max_diff = 0.0f64;
                                let mut inp_diverge_count = 0usize;
                                let mut inp_max_diff_idx = 0usize;
                                for (j, &mv) in inp_flat.iter().enumerate() {
                                    if j >= inp_tam.count as usize { break; }
                                    let nv = nano_eval.get(inp_tam.base_id.offset(j as u32));
                                    let d = (mv - nv).abs();
                                    let t = 1e-2 * mv.abs().max(1.0);
                                    if d > t { inp_diverge_count += 1; }
                                    if d > inp_max_diff { inp_max_diff = d; inp_max_diff_idx = j; }
                                }
                                eprintln!("    input {:?}: {} elements, max_diff={:.6} at [{}], divergent={}",
                                    inp_id, inp_flat.len(), inp_max_diff, inp_max_diff_idx, inp_diverge_count);
                                // Print values around max diff
                                let start = inp_max_diff_idx.saturating_sub(1);
                                let end = (inp_max_diff_idx + 4).min(inp_flat.len()).min(inp_tam.count as usize);
                                for j in start..end {
                                    let nv = nano_eval.get(inp_tam.base_id.offset(j as u32));
                                    let marker = if j == inp_max_diff_idx { " <--" } else { "" };
                                    eprintln!("      [{:4}] milli={:16.10} nano={:16.10}{}",
                                        j, inp_flat[j], nv, marker);
                                }
                                // Trace back: what op produced this input?
                                let inp_producing_op = milli.node_ids().find_map(|oid| {
                                    let n = milli.get_node_by_id(&oid).unwrap();
                                    if n.outputs().any(|o| o == inp_id) {
                                        Some((oid, format!("{}", n.op_kind())))
                                    } else {
                                        None
                                    }
                                });
                                if let Some((inp_op_id, inp_op_kind)) = &inp_producing_op {
                                    eprintln!("      produced by: {} (op {:?})", inp_op_kind, inp_op_id);
                                    // Iteratively trace back up to 5 levels
                                    let mut trace_ops: Vec<GlobalId> = vec![*inp_op_id];
                                    for depth in 0..5 {
                                        let Some(&curr_op_id) = trace_ops.last() else { break };
                                        let indent = "  ".repeat(depth + 4);
                                        let curr_node = milli.get_node_by_id(&curr_op_id).unwrap();
                                        let mut next_op: Option<GlobalId> = None;
                                        for inp2_id in curr_node.inputs() {
                                            if let (Some(inp2_tam), Some(inp2_milli)) = (result.tensor_map.get(&inp2_id), milli_intermediates.get(&inp2_id)) {
                                                if inp2_tam.sym_dims.is_empty() {
                                                    let inp2_flat = tensor_to_f64(inp2_milli);
                                                    let mut d2_max = 0.0f64;
                                                    let mut d2_cnt = 0usize;
                                                    let mut d2_idx = 0usize;
                                                    let count = inp2_flat.len().min(inp2_tam.count as usize);
                                                    for (j, &mv) in inp2_flat.iter().enumerate() {
                                                        if j >= count { break; }
                                                        let nv = nano_eval.get(inp2_tam.base_id.offset(j as u32));
                                                        let d = (mv - nv).abs();
                                                        let t = 1e-2 * mv.abs().max(1.0);
                                                        if d > t { d2_cnt += 1; }
                                                        if d > d2_max { d2_max = d; d2_idx = j; }
                                                    }
                                                    let prod = milli.node_ids().find_map(|oid| {
                                                        let n = milli.get_node_by_id(&oid).unwrap();
                                                        if n.outputs().any(|o| o == inp2_id) {
                                                            Some((oid, format!("{}", n.op_kind())))
                                                        } else {
                                                            None
                                                        }
                                                    });
                                                    let prod_str = prod.as_ref().map(|(_, k)| k.as_str()).unwrap_or("input");
                                                    if count <= 4 {
                                                        for j in 0..count {
                                                            let nv = nano_eval.get(inp2_tam.base_id.offset(j as u32));
                                                            eprintln!("{}input {:?} (from {}): [{}/{}] milli={:.10} nano={:.10}",
                                                                indent, inp2_id, prod_str, j, count, inp2_flat[j], nv);
                                                        }
                                                    } else {
                                                        eprintln!("{}input {:?} (from {}): {} elems, max_diff={:.10} at [{}], divergent={}",
                                                            indent, inp2_id, prod_str, count, d2_max, d2_idx, d2_cnt);
                                                    }
                                                    // Follow the input with largest diff for next trace level
                                                    if let Some((ref pid, _)) = prod {
                                                        if next_op.is_none() || d2_max > 0.0 {
                                                            next_op = Some(*pid);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        if let Some(nop) = next_op {
                                            let nop_kind = milli.get_node_by_id(&nop).map(|n| format!("{}", n.op_kind())).unwrap_or_default();
                                            eprintln!("{}  ^-- produced by: {}", indent, nop_kind);
                                            trace_ops.push(nop);
                                        } else {
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    eprintln!("    produced by: input/unknown");
                }
                eprintln!(
                    "    milli={} nano={} diff={:.6} rel={:.6}",
                    m, n, diff, rel
                );
                // Also print surrounding elements for context.
                let start = i.saturating_sub(1);
                let end = (i + 8).min(milli_flat.len());
                for j in start..end {
                    let nv = nano_eval.get(tam.base_id.offset(j as u32));
                    let marker = if j == i { " <--" } else { "" };
                    eprintln!(
                        "    [{:4}] milli={:12.6} nano={:12.6}{}",
                        j, milli_flat[j], nv, marker
                    );
                }
            }
        }
    }

    eprintln!(
        "  Compared {} tensors ({} elements), max rel error: {:.2e}",
        total_tensors, total_elements, max_rel_err
    );

    if let Some((id, elem, m, n)) = first_divergent {
        panic!(
            "Intermediate divergence: tensor {:?} element {}: milli={} nano={}",
            id, elem, m, n
        );
    }

    // Also check final outputs.
    let output_map_rev: HashMap<GlobalId, GlobalId> = milli
        .output_map
        .as_ref()
        .expect("milli has output_map")
        .iter()
        .map(|(&int, &ext)| (ext, int))
        .collect();
    for (ext_id, milli_tensor) in &milli_outputs {
        let Some(&int_id) = output_map_rev.get(ext_id) else { continue };
        let Some(tam) = result.tensor_map.get(&int_id) else { continue };
        if !tam.sym_dims.is_empty() { continue; }

        let milli_flat = tensor_to_f64(milli_tensor);
        for (i, &m) in milli_flat.iter().enumerate() {
            let n = nano_eval.get(tam.base_id.offset(i as u32));
            let diff = (m - n).abs();
            let tol = 1e-3 * m.abs().max(1.0);
            assert!(
                diff < tol,
                "Output {:?} element {}: milli={} nano={} diff={} rel={}",
                ext_id, i, m, n, diff, diff / m.abs().max(1e-10)
            );
        }
    }

    eprintln!("  INTEGRITY CHECK PASSED");
}
