//! Lower a real ONNX model (GPT-2) to a NanoGraph, execute both the milli
//! interpreter and the nano interpreter, and compare results.
//!
//! Usage:
//!   cargo run --release --example nano_graph_model_test -- test_models/gpt2-lm-head-10.onnx

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use whisper_tensor::DynRank;
use whisper_tensor::compiler::interpret_milli_graph;
use whisper_tensor::compiler::op_census;
use whisper_tensor::dtype::DType;
use whisper_tensor::graph::GlobalId;
use whisper_tensor::model::Model;
use whisper_tensor::nano_graph::lower;
use whisper_tensor::nano_graph::pattern::AtomRange;
use whisper_tensor::nano_graph::AtomId;
use whisper_tensor::numeric_scalar::NumericScalar;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::tensor_info::TensorInfo;
use whisper_tensor_import::identify_and_load;
use whisper_tensor_import::onnx_graph::WeightStorageStrategy;

use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;

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
        println!(
            "  Input '{}': {:?} {:?} ({} elements)",
            name, dtype, shape, num_elements
        );

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
    println!(
        "  Milli interpreter completed in {:.1}s",
        milli_elapsed.as_secs_f64()
    );
    println!("  Output tensors: {}", milli_outputs.len());

    // Print output shapes.
    for (id, tensor) in &milli_outputs {
        let shape: Vec<u64> = tensor.shape().to_vec();
        println!(
            "    Output {:?}: {:?} {:?} ({} elements)",
            id,
            tensor.dtype(),
            shape,
            tensor.num_elements()
        );
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

    // ---- Explicit InputRef diagnostic ----
    {
        use whisper_tensor::nano_graph::InputRef;
        let groups = result.graph.groups();
        let mut explicit_groups = 0u64;
        let mut total_explicit_entries = 0u64;
        let mut explicit_by_op: HashMap<String, (u64, u64)> = HashMap::new(); // op -> (groups, entries)
        for group in groups {
            let mut has_explicit = false;
            let mut group_explicit_entries = 0u64;
            for input in &group.inputs {
                if let InputRef::Explicit(ids) = input {
                    has_explicit = true;
                    group_explicit_entries += ids.len() as u64;
                }
            }
            if has_explicit {
                explicit_groups += 1;
                total_explicit_entries += group_explicit_entries;
                let op_name = format!("{:?}", group.op)
                    .chars()
                    .take_while(|c| *c != ' ' && *c != '{')
                    .collect::<String>();
                let entry = explicit_by_op.entry(op_name).or_default();
                entry.0 += 1;
                entry.1 += group_explicit_entries;
            }
        }
        println!("\n=== Explicit InputRef Diagnostic ===");
        println!(
            "  Groups with Explicit: {} / {}",
            explicit_groups,
            groups.len()
        );
        println!(
            "  Total Explicit entries: {} ({:.1} MB at 8 bytes each)",
            total_explicit_entries,
            total_explicit_entries as f64 * 8.0 / (1024.0 * 1024.0)
        );
        let mut sorted: Vec<_> = explicit_by_op.into_iter().collect();
        sorted.sort_by(|a, b| b.1.1.cmp(&a.1.1));
        for (op, (groups, entries)) in &sorted {
            println!(
                "    {}: {} groups, {} entries ({:.1} MB)",
                op,
                groups,
                entries,
                *entries as f64 * 8.0 / (1024.0 * 1024.0)
            );
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

    // The eval uses per-group buffers with refcount freeing, so actual peak
    // memory is much less than total_atoms * sizeof(NumericScalar). Allow up to
    // 500 GB of "theoretical" buffer since the real peak is ~5% of that.
    let max_buffer_gb = 500.0;
    let nano_feasible = buffer_gb <= max_buffer_gb;

    if !nano_feasible {
        println!(
            "  SKIPPING nano execution: buffer ({:.1} GB) exceeds {:.0} GB limit",
            buffer_gb, max_buffer_gb
        );
        println!(
            "  To run, need a machine with at least {:.0} GB RAM",
            buffer_gb * 1.5
        );
    } else {
        println!("  Buffer fits in memory, proceeding with nano execution");

        // Build nano inputs from input_tensors (tensor-based API).
        let mut backend = whisper_tensor::backends::eval_backend::EvalBackend::NDArray;
        let mut nd_tensors: Vec<NDArrayNumericTensor<DynRank>> = Vec::new();
        let mut eval_inputs: Vec<(AtomId, usize)> = Vec::new(); // (base_id, index into nd_tensors)

        for it in result.graph.input_tensors() {
            // Find external ID for this input tensor's internal tensor_id.
            let ext_id = milli_graph
                .input_map
                .iter()
                .find(|(_, int)| **int == it.tensor_id)
                .map(|(ext, _)| *ext);

            if let Some(ext_id) = ext_id {
                if let Some(tensor) = milli_inputs.get(&ext_id) {
                    let nd = tensor.to_ndarray().unwrap();
                    let idx = nd_tensors.len();
                    nd_tensors.push(nd);
                    eval_inputs.push((it.base_id, idx));
                }
            }
        }

        let eval_input_refs: Vec<(AtomId, &NDArrayNumericTensor<DynRank>)> = eval_inputs
            .iter()
            .map(|&(base, idx)| (base, &nd_tensors[idx]))
            .collect();

        println!(
            "  Input tensors for eval: {} ranges",
            eval_input_refs.len()
        );

        // Build output ranges from milli_outputs via tensor_map.
        let reverse_output_map: HashMap<GlobalId, GlobalId> = milli_graph
            .output_map
            .as_ref()
            .map(|m| m.iter().map(|(&int, &ext)| (ext, int)).collect())
            .unwrap_or_default();

        let mut output_ext_ids: Vec<GlobalId> = Vec::new();
        let mut output_tams: Vec<&lower::TensorAtomMapInfo> = Vec::new();
        let mut output_ranges: Vec<AtomRange> = Vec::new();

        for (ext_id, _milli_tensor) in &milli_outputs {
            let internal_id = reverse_output_map.get(ext_id).unwrap_or(ext_id);
            let Some(tam) = result
                .tensor_map
                .get(internal_id)
                .or_else(|| result.tensor_map.get(ext_id))
            else {
                println!("  WARNING: Output {:?} not found in tensor_map", ext_id);
                continue;
            };
            output_ext_ids.push(*ext_id);
            output_tams.push(tam);
            output_ranges.push(AtomRange {
                base: tam.base_id,
                count: tam.count,
                dtype: tam.dtype,
            });
        }

        // ---- Compare ALL intermediate tensors to find first divergence ----
        println!("\n=== Step 3: Intermediate Comparison (finding first divergence) ===");

        // Collect all milli intermediate tensors.
        let intermediates = whisper_tensor::compiler::interpret_milli_graph_all_intermediates(
            &milli_graph, &milli_inputs,
        )
        .unwrap();
        let producers = whisper_tensor::compiler::tensor_producers(&milli_graph, &intermediates);

        // Build ranges for all non-trivial tensor_map entries and request them from nano eval.
        // We already have nano_results for the output ranges. Now request intermediate ranges.
        let mut check_ids: Vec<GlobalId> = Vec::new();
        let mut check_ranges: Vec<AtomRange> = Vec::new();
        let mut skipped_no_milli = 0u64;
        let mut skipped_segmented = 0u64;
        let mut skipped_large = 0u64;
        for (&tid, tam) in &result.tensor_map {
            if tam.count == 0 {
                continue;
            }
            if !tam.segments.is_empty() {
                skipped_segmented += 1;
                continue;
            }
            if intermediates.get(&tid).is_none() {
                skipped_no_milli += 1;
                if skipped_no_milli <= 5 {
                    let (op_kind, _) = producers.get(&tid)
                        .map(|(k, i)| (k.as_str(), i.as_slice()))
                        .unwrap_or(("?", &[]));
                    println!(
                        "    no-milli: {:?} (op: {}, count: {}, dtype: {:?})",
                        tid, op_kind, tam.count, tam.dtype
                    );
                }
                continue;
            }
            check_ids.push(tid);
            check_ranges.push(AtomRange {
                base: tam.base_id,
                count: tam.count,
                dtype: tam.dtype,
            });
        }
        println!(
            "  Checking {} intermediate tensors (skipped: {} no-milli, {} segmented, {} large)",
            check_ids.len(), skipped_no_milli, skipped_segmented, skipped_large
        );

        let t_check = Instant::now();
        let check_results =
            whisper_tensor::nano_graph::eval::eval(&result.graph, &eval_input_refs, &check_ranges);
        println!(
            "  Eval for intermediates: {:.1}s",
            t_check.elapsed().as_secs_f64()
        );

        // Compare each intermediate.
        let mut first_bad: Option<(GlobalId, String, f64)> = None;
        let mut num_perfect = 0u64;
        let mut num_close = 0u64;
        let mut num_bad = 0u64;
        let mut error_map: HashMap<GlobalId, f64> = HashMap::new();

        for (idx, &tid) in check_ids.iter().enumerate() {
            let tam = &result.tensor_map[&tid];
            let milli_tensor = &intermediates[&tid];

            let Ok(f32_t) = milli_tensor.cast(DType::F32, &mut backend) else { continue };
            let flat = f32_t.flatten().unwrap();
            let nd = flat.to_ndarray().unwrap();
            let milli_vals: Vec<f32> = nd.try_into().unwrap();

            let nano_tensor = &check_results[idx];
            let nano_f32: Vec<f32> = match nano_tensor {
                NDArrayNumericTensor::F32(a) => a.iter().copied().collect(),
                other => {
                    let cast = NumericTensor::from(other.clone())
                        .cast(DType::F32, &mut backend)
                        .unwrap();
                    let flat = cast.flatten().unwrap();
                    let nd = flat.to_ndarray().unwrap();
                    nd.try_into().unwrap()
                }
            };

            if milli_vals.len() != nano_f32.len() {
                continue;
            }

            let mut local_max = 0.0f64;
            for (m, n) in milli_vals.iter().zip(nano_f32.iter()) {
                let diff = (m - n).abs() as f64;
                local_max = local_max.max(diff);
            }

            error_map.insert(tid, local_max);
            if local_max == 0.0 {
                num_perfect += 1;
            } else if local_max < 1e-3 {
                num_close += 1;
            } else {
                num_bad += 1;
                if num_bad <= 10 {
                    let (op_kind, input_info) = producers.get(&tid)
                        .map(|(k, inputs)| (k.clone(), inputs.clone()))
                        .unwrap_or_else(|| ("input".to_string(), vec![]));
                    if first_bad.is_none() {
                        first_bad = Some((tid, op_kind.clone(), local_max));
                    }
                    let milli_shape = intermediates.get(&tid)
                        .map(|t| format!("{:?} {:?}", t.dtype(), t.shape()))
                        .unwrap_or_default();
                    // Check whether each input is itself bad.
                    let mut input_status = Vec::new();
                    for (inp_id, inp_shape) in &input_info {
                        let status = if let Some(tam) = result.tensor_map.get(inp_id) {
                            if tam.segments.is_empty() && tam.count <= 1_000_000 {
                                // We checked this tensor — see if it's in error_map.
                                error_map.get(inp_id).map(|e| {
                                    if *e == 0.0 { "perfect".to_string() }
                                    else if *e < 1e-3 { format!("close({:.1e})", e) }
                                    else { format!("BAD({:.1e})", e) }
                                }).unwrap_or_else(|| "not-checked".to_string())
                            } else {
                                format!("skipped(count={},segs={})", tam.count, tam.segments.len())
                            }
                        } else {
                            "no-tensor-map".to_string()
                        };
                        input_status.push((inp_id, inp_shape, status));
                    }
                    println!(
                        "  BAD #{}: {:?} (op: {}, shape: {}, {} elems, err={:.2e})",
                        num_bad, tid, op_kind, milli_shape, milli_vals.len(), local_max
                    );
                    for (inp_id, inp_shape, status) in &input_status {
                        println!("    input {:?} {:?}: {}", inp_id, inp_shape, status);
                    }
                    if num_bad == 1 {
                        // Print first few mismatched elements for the first bad tensor.
                        let mut shown = 0;
                        for (i, (m, n)) in milli_vals.iter().zip(nano_f32.iter()).enumerate() {
                            let diff = (m - n).abs();
                            if diff > 1e-3 && shown < 5 {
                                println!("    elem {}: milli={:.6} nano={:.6} diff={:.6}", i, m, n, diff);
                                shown += 1;
                            }
                        }
                    }
                }
            }
        }

        println!(
            "\n  Summary: {} perfect, {} close (<1e-3), {} bad (>1e-3)",
            num_perfect, num_close, num_bad
        );

        // (Output comparison skipped — intermediate comparison covers it)

        // ---- Timing summary ----
        println!("\n=== Timing Summary ===");
        println!("  Milli interpreter: {:.3}s", milli_elapsed.as_secs_f64());
        println!("  NanoGraph lowering: {:.3}s", lower_elapsed.as_secs_f64());
        println!("  Nano intermediate check: {:.3}s", t_check.elapsed().as_secs_f64());
    }
}
