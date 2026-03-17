//! v14 execution scaffold: load GPT-2, lower to NanoGraph, prepare for partitioning.
//!
//! Usage:
//!   cargo run --release --example v14_scaffold -- test_models/gpt2-lm-head-10.onnx

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use whisper_tensor::compiler::attempts::v14::types::*;
use whisper_tensor::compiler::op_census;
use whisper_tensor::dtype::DType;
use whisper_tensor::graph::GlobalId;
use whisper_tensor::model::Model;
use whisper_tensor::nano_graph::lower;
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

    // ── Step 1: Load ONNX model ──────────────────────────────────────────────

    let t0 = Instant::now();
    let onnx_data = identify_and_load(path, WeightStorageStrategy::EmbeddedData).unwrap();
    let mut rng = rand::rng();
    let model = Model::new_from_onnx(&onnx_data, &mut rng, None).unwrap();
    eprintln!("Model loaded in {:.1}s", t0.elapsed().as_secs_f64());

    // ── Step 2: Generate MilliOpGraph ────────────────────────────────────────

    let t0 = Instant::now();
    let sym_graph = model.get_symbolic_graph();
    let tensor_store = model.get_tensor_store();
    let milli_graph = sym_graph.generate_milli_graph(&mut rng);
    eprintln!(
        "MilliOpGraph generated in {:.1}ms",
        t0.elapsed().as_secs_f64() * 1e3
    );

    let census = op_census(&milli_graph);
    let total_ops: usize = census.iter().map(|(_, c)| c).sum();
    println!("=== Milli Op Census ({} ops) ===", total_ops);
    for (kind, count) in &census {
        println!("  {:>4}x  {}", count, kind);
    }

    // ── Step 3: Build TensorInfo for lowering (shapes + dtypes only) ─────────

    let t0 = Instant::now();
    let input_info = model.get_input_tensor_info().unwrap();
    let tensors_by_name = sym_graph.get_tensors_by_name();

    let mut all_infos: HashMap<GlobalId, TensorInfo> = HashMap::new();

    // User inputs: fill unknown dims with concrete values.
    for (name, (dtype, shape_dims)) in &input_info {
        let shape: Vec<u64> = shape_dims.iter().map(|d| d.unwrap_or(4)).collect();
        println!("Input '{}': {:?} {:?}", name, dtype, shape);
        if let Some(id) = tensors_by_name.get(name) {
            all_infos.insert(*id, TensorInfo::from_dtype_and_shape(*dtype, &shape));
        }
    }

    // Small constants (axes, indices, shape values): keep full data so
    // infer_all can resolve Shape/Gather/Reshape ops.
    // Large weight matrices: shape+dtype only.
    let initialized = sym_graph.get_initialized_tensors(tensor_store);
    for (id, tensor) in &initialized {
        if tensor.num_elements() <= 1024 {
            all_infos.insert(*id, TensorInfo::from(tensor.clone()));
        } else {
            let shape: Vec<u64> = tensor.shape().to_vec();
            let dtype = tensor.dtype();
            all_infos.insert(*id, TensorInfo::from_dtype_and_shape(dtype, &shape));
        }
    }
    eprintln!(
        "TensorInfo built in {:.1}s ({} tensors)",
        t0.elapsed().as_secs_f64(),
        all_infos.len()
    );

    // ── Step 4: Lower to NanoGraph ───────────────────────────────────────────

    let t0 = Instant::now();
    let result = lower::lower(&milli_graph, &all_infos).unwrap();
    eprintln!("Lowered in {:.1}ms", t0.elapsed().as_secs_f64() * 1e3);

    let stats = result.graph.stats();
    println!("\n=== NanoGraph ===\n{}", stats);

    if !result.unsupported.is_empty() {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for (_, kind) in &result.unsupported {
            *counts.entry(kind.clone()).or_default() += 1;
        }
        println!("\nUnsupported ops ({}):", result.unsupported.len());
        let mut sorted: Vec<_> = counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        for (kind, count) in sorted {
            println!("  {:>4}x  {}", count, kind);
        }
    }

    let errors = result.graph.validate();
    if errors.is_empty() {
        println!("Validation: PASSED");
    } else {
        println!("Validation: {} ERRORS", errors.len());
        for e in errors.iter().take(10) {
            println!("  {}", e);
        }
    }

    // ── Step 5: Build tensor_map for the ExecutionPlan ───────────────────────

    let tensor_map = build_tensor_map(
        &result.tensor_map,
        &milli_graph.input_map,
        &input_info,
        &tensors_by_name,
    );

    let mut n_weight = 0usize;
    let mut n_input = 0usize;
    let mut n_computed = 0usize;
    let mut weight_atoms = 0u64;
    let mut input_atoms = 0u64;
    let mut computed_atoms = 0u64;
    for tm in tensor_map.values() {
        match tm.kind {
            TensorKind::Weight => {
                n_weight += 1;
                weight_atoms += tm.range.count;
            }
            TensorKind::Input => {
                n_input += 1;
                input_atoms += tm.range.count;
            }
            TensorKind::Computed => {
                n_computed += 1;
                computed_atoms += tm.range.count;
            }
        }
    }
    println!("\n=== Tensor Map ===");
    println!("  Weights:  {} tensors, {} atoms", n_weight, weight_atoms);
    println!("  Inputs:   {} tensors, {} atoms", n_input, input_atoms);
    println!(
        "  Computed: {} tensors, {} atoms",
        n_computed, computed_atoms
    );
    println!("  Total atom ID space: {}", result.graph.num_atoms());

    // ── Step 6: Build model outputs ──────────────────────────────────────────

    let model_outputs = build_model_outputs(&milli_graph, &result.tensor_map);
    println!("\n=== Model Outputs ({}) ===", model_outputs.len());
    for out in &model_outputs {
        println!(
            "  {:?}: base={} count={} {:?}",
            out.tensor_id, out.range.base, out.range.count, out.range.dtype
        );
    }

    // ── Step 7: Liveness analysis ──────────────────────────────────────────

    let t0 = Instant::now();
    let liveness = result.graph.liveness();
    eprintln!("Liveness scan: {:.1}ms", t0.elapsed().as_secs_f64() * 1e3);

    let total_groups = liveness.len();
    let dead_groups = liveness.iter().filter(|g| g.use_count == 0).count();
    let dead_atoms: u64 = liveness
        .iter()
        .filter(|g| g.use_count == 0)
        .map(|g| g.count)
        .sum();
    let max_use = liveness.iter().map(|g| g.use_count).max().unwrap_or(0);
    let single_use = liveness.iter().filter(|g| g.use_count == 1).count();
    let single_use_atoms: u64 = liveness
        .iter()
        .filter(|g| g.use_count == 1)
        .map(|g| g.count)
        .sum();
    let multi_use = liveness.iter().filter(|g| g.use_count > 1).count();
    let total_atoms: u64 = liveness.iter().map(|g| g.count).sum();

    println!("\n=== Liveness ===");
    println!("  Groups: {}", total_groups);
    println!(
        "  Dead (use_count=0): {} groups, {} atoms",
        dead_groups, dead_atoms
    );
    println!(
        "  Single-use (use_count=1): {} groups, {} atoms",
        single_use, single_use_atoms
    );
    println!("  Multi-use (use_count>1): {} groups", multi_use);
    println!("  Max use count: {}", max_use);
    println!("  Total compute atoms: {}", total_atoms);

    // ── Step 8: Build trivial execution plan (1 phase, 1 lane) ─────────────

    use whisper_tensor::compiler::attempts::v14::plan;

    let exec_plan = plan::plan_trivial(result, tensor_map, model_outputs);

    let span = &exec_plan.phases[0].spans[0];
    println!("\n=== Execution Plan (trivial: 1 phase, 1 lane) ===");
    println!("  Phases: {}", exec_plan.phases.len());
    println!("  Span groups: {}", span.graph.num_groups());
    println!("  Span inputs: {} ranges", span.inputs.len());
    println!("  Span outputs: {} ranges", span.outputs.len());

    // ── Step 8: Build shared inputs ────────────────────────────────────────

    use whisper_tensor::backends::eval_backend::EvalBackend;
    use whisper_tensor::backends::ndarray_backend::numeric_tensor::NDArrayNumericTensor;
    use whisper_tensor::compiler::attempts::v14::execute;
    use whisper_tensor::nano_graph::AtomId;
    use whisper_tensor::numeric_tensor::NumericTensor;

    // Build user input tensors with valid token IDs.
    // GPT-2 vocabulary: common tokens like "Hello" = 15496, "," = 11, " world" = 995.
    let mut user_inputs: HashMap<String, NumericTensor<whisper_tensor::DynRank>> = HashMap::new();
    for (name, (dtype, shape_dims)) in &input_info {
        let shape: Vec<u64> = shape_dims.iter().map(|d| d.unwrap_or(4)).collect();
        let num_elements: u64 = shape.iter().product();
        let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        // Valid GPT-2 token IDs, cycled to fill the input shape.
        let token_ids: Vec<i64> = vec![15496, 11, 995, 0, 464, 1917, 318, 1049];
        let tensor = match dtype {
            DType::I64 => {
                let data: Vec<i64> = (0..num_elements as usize)
                    .map(|i| token_ids[i % token_ids.len()])
                    .collect();
                NumericTensor::from_vec_shape(data, shape_usize).unwrap()
            }
            DType::F32 => {
                let data: Vec<f32> = (0..num_elements as usize)
                    .map(|i| token_ids[i % token_ids.len()] as f32)
                    .collect();
                NumericTensor::from_vec_shape(data, shape_usize).unwrap()
            }
            _ => {
                let data: Vec<f32> = vec![0.0; num_elements as usize];
                NumericTensor::from_vec_shape(data, shape_usize).unwrap()
            }
        };
        println!("User input '{}': {:?} {:?}", name, dtype, shape);
        user_inputs.insert(name.clone(), tensor);
    }

    // Build milli-eval inputs: HashMap<external GlobalId, NumericTensor>.
    // Includes all initialized tensors (weights + small constants) and user inputs.
    let mut milli_inputs: HashMap<GlobalId, NumericTensor<whisper_tensor::DynRank>> =
        HashMap::new();
    for (ext_id, tensor) in &initialized {
        milli_inputs.insert(*ext_id, tensor.clone());
    }
    for (name, tensor) in &user_inputs {
        if let Some(&ext_id) = tensors_by_name.get(name.as_str()) {
            milli_inputs.insert(ext_id, tensor.clone());
        }
    }

    // ── Step 9: Run MilliOpGraph reference eval ─────────────────────────────

    println!("\n=== MilliOpGraph Reference Eval ===");
    let t0 = Instant::now();
    let mut backend = EvalBackend::NDArray;
    let milli_outputs: HashMap<GlobalId, NumericTensor<whisper_tensor::DynRank>> = milli_graph
        .eval(&milli_inputs, &mut (), &mut backend)
        .unwrap()
        .collect();
    println!(
        "  Executed in {:.1}s, {} outputs",
        t0.elapsed().as_secs_f64(),
        milli_outputs.len()
    );
    for (id, tensor) in &milli_outputs {
        let nd = tensor.to_ndarray().unwrap();
        let flat = nd.flatten();
        let first_few: Vec<f64> = (0..flat.num_elements().min(5))
            .map(|i| flat.get(&[i as u64]).unwrap().to_f64())
            .collect();
        println!(
            "  {:?}: {} elements, first={:?}",
            id,
            nd.num_elements(),
            first_few
        );
    }

    // ── Step 10: Run NanoGraph eval via v14 executor ────────────────────────

    // Build nano executor inputs from the same data.
    let mut exec_inputs: Vec<(AtomId, NDArrayNumericTensor<whisper_tensor::DynRank>)> = Vec::new();

    for it in exec_plan.graph.input_tensors() {
        let milli_id = it.tensor_id;
        let ext_id = milli_graph
            .input_map
            .iter()
            .find(|(_, int)| **int == milli_id)
            .map(|(ext, _)| *ext);

        let tensor = if let Some(ext) = ext_id {
            if let Some(t) = initialized.get(&ext) {
                Some(t.clone())
            } else {
                let name = sym_graph.get_tensor_name(ext);
                name.and_then(|n| user_inputs.get(n).cloned())
            }
        } else {
            None
        };

        if let Some(t) = tensor {
            let nd = match t {
                NumericTensor::NDArray(nd) => nd,
                _ => t.to_ndarray().unwrap(),
            };
            exec_inputs.push((it.base_id, nd));
        }
    }

    println!(
        "\n=== NanoGraph Eval ({} input tensors) ===",
        exec_inputs.len()
    );
    let t0 = Instant::now();
    let nano_outputs = execute::execute(&exec_plan, exec_inputs);
    println!(
        "  Executed in {:.1}s, {} outputs",
        t0.elapsed().as_secs_f64(),
        nano_outputs.len()
    );

    // ── Step 11: Compare outputs ────────────────────────────────────────────

    println!("\n=== Comparison ===");
    let mut all_match = true;
    for (ext_id, milli_tensor) in &milli_outputs {
        let milli_nd = milli_tensor.to_ndarray().unwrap();
        if let Some(nano_tensor) = nano_outputs.get(ext_id) {
            let n = milli_nd.num_elements().min(nano_tensor.num_elements());
            let milli_flat = milli_nd.flatten();
            let nano_flat = nano_tensor.flatten();

            let mut max_abs_diff = 0.0f64;
            let mut max_rel_diff = 0.0f64;
            let mut nan_count = 0usize;
            let mut mismatches = 0usize;

            for i in 0..n {
                let m = milli_flat.get(&[i as u64]).unwrap().to_f64();
                let n_val = nano_flat.get(&[i as u64]).unwrap().to_f64();

                if m.is_nan() || n_val.is_nan() {
                    if m.is_nan() != n_val.is_nan() {
                        mismatches += 1;
                    }
                    nan_count += 1;
                    continue;
                }

                let abs_diff = (m - n_val).abs();
                max_abs_diff = max_abs_diff.max(abs_diff);

                let denom = m.abs().max(1e-10);
                max_rel_diff = max_rel_diff.max(abs_diff / denom);

                if abs_diff > 1e-3 && abs_diff / denom > 1e-3 {
                    mismatches += 1;
                    if mismatches <= 3 {
                        eprintln!(
                            "  Mismatch at element {}: milli={} nano={} diff={}",
                            i, m, n_val, abs_diff
                        );
                    }
                }
            }

            let status = if mismatches == 0 { "MATCH" } else { "MISMATCH" };
            println!(
                "  {:?}: {} ({} elements, max_abs={:.6}, max_rel={:.6}, nans={}, mismatches={})",
                ext_id, status, n, max_abs_diff, max_rel_diff, nan_count, mismatches
            );

            // Show first few values from each.
            let first_n = n.min(5);
            let milli_vals: Vec<f64> = (0..first_n)
                .map(|i| milli_flat.get(&[i as u64]).unwrap().to_f64())
                .collect();
            let nano_vals: Vec<f64> = (0..first_n)
                .map(|i| nano_flat.get(&[i as u64]).unwrap().to_f64())
                .collect();
            println!("    milli: {:?}", milli_vals);
            println!("    nano:  {:?}", nano_vals);

            if mismatches > 0 {
                all_match = false;
            }
        } else {
            println!("  {:?}: MISSING from nano outputs", ext_id);
            all_match = false;
        }
    }

    if all_match {
        println!("\nAll outputs MATCH!");
    } else {
        println!("\nSome outputs MISMATCHED.");
    }
}

/// Build the ExecutionPlan's tensor_map from lowering results.
///
/// Classifies each tensor as Weight, Input, or Computed by cross-referencing
/// the milli_graph's input_map with the user-provided input_info.
fn build_tensor_map(
    lower_tensor_map: &HashMap<GlobalId, lower::TensorAtomMapInfo>,
    input_map: &HashMap<GlobalId, GlobalId>,
    input_info: &HashMap<String, (DType, Vec<Option<u64>>)>,
    tensors_by_name: &HashMap<String, GlobalId>,
) -> HashMap<GlobalId, TensorMapping> {
    // Build set of milli-internal IDs that are user inputs.
    let user_input_ids: HashMap<GlobalId, ()> = input_info
        .keys()
        .filter_map(|name| {
            let ext_id = tensors_by_name.get(name)?;
            let int_id = input_map.get(ext_id)?;
            Some((*int_id, ()))
        })
        .collect();

    // Build set of milli-internal IDs that are weight/constant inputs
    // (everything in input_map that isn't a user input).
    let weight_ids: HashMap<GlobalId, ()> = input_map
        .values()
        .filter(|int_id| !user_input_ids.contains_key(int_id))
        .map(|int_id| (*int_id, ()))
        .collect();

    let mut tensor_map = HashMap::new();
    for (&tensor_id, tam) in lower_tensor_map {
        let kind = if user_input_ids.contains_key(&tensor_id) {
            TensorKind::Input
        } else if weight_ids.contains_key(&tensor_id) {
            TensorKind::Weight
        } else {
            TensorKind::Computed
        };

        let dtype = tam.dtype;

        tensor_map.insert(
            tensor_id,
            TensorMapping {
                range: AtomRange {
                    base: tam.base_id,
                    count: tam.count,
                    dtype,
                },
                sym_dims: tam.sym_dims.clone(),
                kind,
            },
        );
    }
    tensor_map
}

/// Build model output mappings from the milli_graph's output_map.
fn build_model_outputs(
    milli_graph: &whisper_tensor::milli_graph::MilliOpGraph,
    lower_tensor_map: &HashMap<GlobalId, lower::TensorAtomMapInfo>,
) -> Vec<OutputMapping> {
    let mut outputs = Vec::new();

    // The milli_graph's output_map maps internal_id → external_id.
    // We need to find the atom ranges for each output.
    if let Some(output_map) = &milli_graph.output_map {
        for (&int_id, &ext_id) in output_map {
            if let Some(tam) = lower_tensor_map.get(&int_id) {
                outputs.push(OutputMapping {
                    tensor_id: ext_id,
                    range: AtomRange {
                        base: tam.base_id,
                        count: tam.count,
                        dtype: tam.dtype,
                    },
                    sym_dims: tam.sym_dims.clone(),
                });
            }
        }
    }
    outputs
}
