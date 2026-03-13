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
    // Insert weights first, then user inputs on top — user inputs take precedence
    // when a tensor ID appears in both (e.g. vk_state is initialized to zeros in
    // the model weights but we provide nonzero test values).
    let mut milli_inputs: HashMap<GlobalId, NumericTensor<DynRank>> = HashMap::new();
    for (id, t) in &weight_tensors {
        milli_inputs.insert(*id, t.clone());
    }
    for (id, t) in &input_tensors {
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
    // Insert weights first, then user inputs on top — user inputs take precedence.
    // User inputs should be Shaped (not Numeric) so the lowering doesn't bake in
    // test values as constants; the test provides runtime overrides separately.
    let mut lower_infos: HashMap<GlobalId, TensorInfo> = HashMap::new();
    for (id, t) in &weight_tensors {
        lower_infos.insert(*id, TensorInfo::from(t.clone()));
    }
    for (id, t) in &input_tensors {
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

    // ---- Check if any op overwrites a vk_state input ----
    {
        use whisper_tensor::graph::{Graph, Node};
        let mut vk_state_int_ids: HashMap<GlobalId, GlobalId> = HashMap::new();
        for (ext_id, _tensor) in &input_tensors {
            let Some(&int_id) = milli.input_map.get(ext_id) else { continue };
            let Some(tam) = result.tensor_map.get(&int_id) else { continue };
            if tam.count == 49152 {
                vk_state_int_ids.insert(int_id, *ext_id);
            }
        }
        for op_id in milli.node_ids() {
            let node = milli.get_node_by_id(&op_id).unwrap();
            for out_id in node.outputs() {
                if let Some(ext_id) = vk_state_int_ids.get(&out_id) {
                    eprintln!("  WARNING: op {} ({:?}) overwrites vk_state input {:?} (ext {:?})",
                        node.op_kind(), op_id, out_id, ext_id);
                }
            }
        }
    }

    // ---- Check for atom overlap ----
    {
        let mut ranges: Vec<(u32, u32, GlobalId)> = result.tensor_map.iter()
            .map(|(id, tam)| (tam.base_id.0, tam.base_id.0 + tam.count, *id))
            .collect();
        ranges.sort();
        for w in ranges.windows(2) {
            let (start_a, end_a, id_a) = w[0];
            let (start_b, _end_b, id_b) = w[1];
            if end_a > start_b {
                eprintln!("  OVERLAP: {:?} [{}, {}) overlaps {:?} [{}, {})",
                    id_a, start_a, end_a, id_b, start_b, _end_b);
            }
        }
    }

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
    let mut first_any_diff_printed = false;

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

            // Track first tensor with ANY diff to find precision root.
            if !first_any_diff_printed && diff > 1e-10 {
                first_any_diff_printed = true;
                use whisper_tensor::graph::{Graph as _, Node as _};
                let prod_op = milli.node_ids().find_map(|op_id| {
                    let node = milli.get_node_by_id(&op_id).unwrap();
                    if node.outputs().any(|o| o == **int_id) {
                        Some(format!("{}", node.op_kind()))
                    } else { None }
                });
                eprintln!("  FIRST ANY DIFF: tensor {:?} (base_id={}) elem {} diff={:.15} milli={:.15} nano={:.15} op={:?}",
                    int_id, tam.base_id.0, i, diff, m, n, prod_op.as_deref());
            }

            // Report first significant divergence.
            let tol = 1e-2 * m.abs().max(1.0);
            if first_divergent.is_none() && diff > tol {
                first_divergent = Some((**int_id, i, m, n));
                use whisper_tensor::graph::{Graph, Node};
                let producing_op = milli.node_ids().find_map(|op_id| {
                    let node = milli.get_node_by_id(&op_id).unwrap();
                    if node.outputs().any(|o| o == **int_id) {
                        Some(format!("{}", node.op_kind()))
                    } else {
                        None
                    }
                });
                eprintln!(
                    "  FIRST DIVERGENCE (after {} clean tensors): tensor {:?} element {} op={:?} milli={} nano={} diff={:.6}",
                    total_tensors - 1, int_id, i, producing_op.as_deref(), m, n, diff
                );
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
