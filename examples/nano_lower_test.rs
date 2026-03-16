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
use whisper_tensor::nano_graph::{InputRef, NanoGraph, ScalarOp};
use whisper_tensor::tensor_info::TensorInfo;
use whisper_tensor_import::identify_and_load;
use whisper_tensor_import::onnx_graph::WeightStorageStrategy;

macro_rules! validate_v4_plan {
    ($name:expr, $plan:expr, $graph:expr) => {{
        let raw: Vec<Vec<(&NanoGraph, Vec<(u64, u64)>, Vec<(u64, u64)>)>> = $plan.phases.iter().map(|phase| {
            phase.spans.iter().map(|span| {
                let inputs: Vec<(u64, u64)> = span.inputs.iter().map(|m| (m.main_base.0, m.count)).collect();
                let outputs: Vec<(u64, u64)> = span.outputs.iter().map(|m| (m.main_base.0, m.count)).collect();
                (&span.graph as &NanoGraph, inputs, outputs)
            }).collect()
        }).collect();
        validate_span_topology_raw($name, &raw, $graph);
    }};
}

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

    let mut all_infos: HashMap<GlobalId, TensorInfo> = HashMap::new();
    for (name, (dtype, shape_dims)) in &input_info {
        let shape: Vec<u64> = shape_dims.iter().map(|d| d.unwrap_or(4)).collect();
        let info = TensorInfo::from_dtype_and_shape(*dtype, &shape);
        println!("Input '{}': {:?} {:?}", name, dtype, shape);
        if let Some(id) = tensors_by_name.get(name) {
            all_infos.insert(*id, info);
        }
    }
    let initialized = sym_graph.get_initialized_tensors(tensor_store);
    for (id, tensor) in initialized {
        all_infos.insert(id, TensorInfo::from(tensor));
    }

    // ---- Lower (graph only, skip numeric_overrides) ----
    let t0 = Instant::now();
    let result = whisper_tensor::nano_graph::lower::lower_graph_only(&milli_graph, &all_infos).unwrap();
    eprintln!("lower_graph_only: {:.1}s", t0.elapsed().as_secs_f64());

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
        for (kind, count) in sorted { println!("  {:>4}x  {}", count, kind); }
    }

    // ---- Span-Based Execution Plans ----
    let num_lanes = std::env::var("NUM_LANES").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    println!("\n=== Span-Based Plans (num_lanes={}) ===", num_lanes);

    let which = std::env::var("SPAN_PLANNER").unwrap_or("all".to_string());
    if which == "a" || which == "all" {
        use whisper_tensor::compiler::attempts::v13_claude::nano_plan_spans_a;
        let t0 = Instant::now();
        let plan = nano_plan_spans_a::plan_execution(&result.graph, num_lanes);
        let elapsed = t0.elapsed();
        let ss: Vec<Vec<SS>> = plan.phases.iter().map(|p| p.spans.iter().map(|s| SS {
            ng: s.graph.num_groups(), na: s.graph.num_atoms(), ni: s.inputs.len(), no: s.outputs.len(),
        }).collect()).collect();
        print_span_summary("spans_a", &ss, plan.num_lanes, elapsed);
    }
    if which == "b" || which == "all" {
        use whisper_tensor::compiler::attempts::v13_claude::nano_plan_spans_b;
        let t0 = Instant::now();
        let plan = nano_plan_spans_b::plan_spans(&result.graph, num_lanes);
        let elapsed = t0.elapsed();
        let ss: Vec<Vec<SS>> = plan.phases.iter().map(|p| p.spans.iter().map(|s| SS {
            ng: s.graph.num_groups(), na: s.graph.num_atoms(), ni: s.inputs.len(), no: s.outputs.len(),
        }).collect()).collect();
        print_span_summary("spans_b", &ss, plan.num_lanes, elapsed);
    }
    if which == "c" || which == "all" {
        use whisper_tensor::compiler::attempts::v13_claude::nano_plan_spans_c;
        let t0 = Instant::now();
        let plan = nano_plan_spans_c::plan_execution_spans(&result.graph, num_lanes);
        let elapsed = t0.elapsed();
        let ss: Vec<Vec<SS>> = plan.phases.iter().map(|p| p.spans.iter().map(|s| SS {
            ng: s.graph.num_groups(), na: s.graph.num_atoms(), ni: s.inputs.len(), no: s.outputs.len(),
        }).collect()).collect();
        print_span_summary("spans_c", &ss, plan.num_lanes, elapsed);

        // Topology validation (O(groups), not O(atoms))
        let t_val = Instant::now();
        let mut errors = 0usize;
        // Track which main-graph atom RANGES have been produced by earlier phases.
        // Store as (base, count) ranges sorted by base.
        let mut produced_ranges: Vec<(u64, u64)> = Vec::new();
        // Also include all Literal group ranges as always-available.
        for g in result.graph.groups() {
            if matches!(&g.op, ScalarOp::Literal(_)) && g.inputs.is_empty() {
                produced_ranges.push((g.base_id.0, g.count));
            }
        }
        produced_ranges.sort();

        let range_contains = |ranges: &[(u64, u64)], atom: u64| -> bool {
            match ranges.binary_search_by(|&(base, _)| base.cmp(&atom)) {
                Ok(_) => true,
                Err(0) => false,
                Err(i) => {
                    let (base, count) = ranges[i - 1];
                    atom < base + count
                }
            }
        };

        for (phase_idx, phase) in plan.phases.iter().enumerate() {
            // Check each span's declared inputs are available
            for (lane_idx, span) in phase.spans.iter().enumerate() {
                for mapping in &span.inputs {
                    // Check that main_base..main_base+count is covered by produced_ranges
                    if !range_contains(&produced_ranges, mapping.main_base.0) {
                        errors += 1;
                        if errors <= 10 {
                            println!("  SPAN VIOLATION: phase {}/lane {}: input base {:?} (count={}) not available",
                                phase_idx, lane_idx, mapping.main_base, mapping.count);
                            // Trace producer group for diagnostics
                            let main_groups = result.graph.groups();
                            let atom_val = mapping.main_base.0;
                            let idx = main_groups.partition_point(|g| g.base_id.0 <= atom_val);
                            if idx > 0 {
                                let g = &main_groups[idx - 1];
                                if atom_val < g.base_id.0 + g.count {
                                    let op_str = format!("{:?}", g.op).chars().take(60).collect::<String>();
                                    let mut in_phase = "not in any output".to_string();
                                    for (pi, ph) in plan.phases.iter().enumerate() {
                                        for sp in &ph.spans {
                                            for om in &sp.outputs {
                                                if om.main_base.0 <= atom_val && atom_val < om.main_base.0 + om.count {
                                                    in_phase = format!("phase {}", pi);
                                                }
                                            }
                                        }
                                    }
                                    println!("    group_idx={} base={:?} count={} op={} [{}]",
                                        idx - 1, g.base_id, g.count, op_str, in_phase);
                                }
                            }
                        }
                    }
                }

                // Check within each span: groups are in valid topo order
                let span_groups = span.graph.groups();
                let span_bases: Vec<u64> = span_groups.iter().map(|g| g.base_id.0).collect();
                for (gi, group) in span_groups.iter().enumerate() {
                    for input in &group.inputs {
                        // Resolve the first atom of this input
                        let src = input.resolve(0, 0);
                        // Find which span group it belongs to
                        let src_gi = match span_bases.binary_search(&src.0) {
                            Ok(i) => Some(i),
                            Err(0) => None,
                            Err(i) => {
                                let candidate = i - 1;
                                if src.0 < span_groups[candidate].base_id.0 + span_groups[candidate].count {
                                    Some(candidate)
                                } else { None }
                            }
                        };
                        if let Some(src_gi) = src_gi {
                            if src_gi > gi && !matches!(&span_groups[src_gi].op, ScalarOp::Literal(_)) {
                                errors += 1;
                                if errors <= 10 {
                                    let op = format!("{:?}", group.op).chars().take_while(|c| *c != ' ' && *c != '{').collect::<String>();
                                    let src_op = format!("{:?}", span_groups[src_gi].op).chars().take_while(|c| *c != ' ' && *c != '{').collect::<String>();
                                    println!("  SPAN TOPO VIOLATION: phase {}/lane {}: group {} ({}) reads from later group {} ({})",
                                        phase_idx, lane_idx, gi, op, src_gi, src_op);
                                }
                            }
                        }
                    }
                }
            }

            // After this phase, add all span outputs to produced_ranges
            for span in &phase.spans {
                for mapping in &span.outputs {
                    produced_ranges.push((mapping.main_base.0, mapping.count));
                }
            }
            produced_ranges.sort();
        }

        let val_time = t_val.elapsed();
        if errors == 0 {
            println!("    TOPOLOGY: VALID ({:.1}ms)", val_time.as_secs_f64() * 1e3);
        } else {
            println!("    TOPOLOGY: {} VIOLATIONS ({:.1}ms)", errors, val_time.as_secs_f64() * 1e3);
        }
    }
    if which == "v3c" || which == "all" {
        use whisper_tensor::compiler::attempts::v13_claude::nano_plan_v3c;
        let t0 = Instant::now();
        let plan = nano_plan_v3c::plan_execution_spans(&result.graph, num_lanes);
        let elapsed = t0.elapsed();
        let ss: Vec<Vec<SS>> = plan.phases.iter().map(|p| p.spans.iter().map(|s| SS {
            ng: s.graph.num_groups(), na: s.graph.num_atoms(), ni: s.inputs.len(), no: s.outputs.len(),
        }).collect()).collect();
        print_span_summary("v3c", &ss, plan.num_lanes, elapsed);

        // Topology validation (same as spans_c)
        let t_val = Instant::now();
        let mut errors = 0usize;
        let mut produced_ranges: Vec<(u64, u64)> = Vec::new();
        for g in result.graph.groups() {
            if matches!(&g.op, ScalarOp::Literal(_)) && g.inputs.is_empty() {
                produced_ranges.push((g.base_id.0, g.count));
            }
        }
        produced_ranges.sort();

        let range_contains = |ranges: &[(u64, u64)], atom: u64| -> bool {
            match ranges.binary_search_by(|&(base, _)| base.cmp(&atom)) {
                Ok(_) => true,
                Err(0) => false,
                Err(i) => {
                    let (base, count) = ranges[i - 1];
                    atom < base + count
                }
            }
        };

        for (phase_idx, phase) in plan.phases.iter().enumerate() {
            for (lane_idx, span) in phase.spans.iter().enumerate() {
                for mapping in &span.inputs {
                    if !range_contains(&produced_ranges, mapping.main_base.0) {
                        errors += 1;
                        if errors <= 10 {
                            println!("  V3C VIOLATION: phase {}/lane {}: input base {:?} (count={}) not available",
                                phase_idx, lane_idx, mapping.main_base, mapping.count);
                            let main_groups = result.graph.groups();
                            let atom_val = mapping.main_base.0;
                            let idx = main_groups.partition_point(|g| g.base_id.0 <= atom_val);
                            if idx > 0 {
                                let g = &main_groups[idx - 1];
                                if atom_val < g.base_id.0 + g.count {
                                    let op_str = format!("{:?}", g.op).chars().take(60).collect::<String>();
                                    let mut in_phase = "not in any output".to_string();
                                    for (pi, ph) in plan.phases.iter().enumerate() {
                                        for sp in &ph.spans {
                                            for om in &sp.outputs {
                                                if om.main_base.0 <= atom_val && atom_val < om.main_base.0 + om.count {
                                                    in_phase = format!("phase {}", pi);
                                                }
                                            }
                                        }
                                    }
                                    println!("    group_idx={} base={:?} count={} op={} [{}]",
                                        idx - 1, g.base_id, g.count, op_str, in_phase);
                                }
                            }
                        }
                    }
                }

                let span_groups = span.graph.groups();
                let span_bases: Vec<u64> = span_groups.iter().map(|g| g.base_id.0).collect();
                for (gi, group) in span_groups.iter().enumerate() {
                    for input in &group.inputs {
                        let src = input.resolve(0, 0);
                        let src_gi = match span_bases.binary_search(&src.0) {
                            Ok(i) => Some(i),
                            Err(0) => None,
                            Err(i) => {
                                let candidate = i - 1;
                                if src.0 < span_groups[candidate].base_id.0 + span_groups[candidate].count {
                                    Some(candidate)
                                } else { None }
                            }
                        };
                        if let Some(src_gi) = src_gi {
                            if src_gi > gi && !matches!(&span_groups[src_gi].op, ScalarOp::Literal(_)) {
                                errors += 1;
                                if errors <= 10 {
                                    let op = format!("{:?}", group.op).chars().take_while(|c| *c != ' ' && *c != '{').collect::<String>();
                                    let src_op = format!("{:?}", span_groups[src_gi].op).chars().take_while(|c| *c != ' ' && *c != '{').collect::<String>();
                                    println!("  V3C TOPO VIOLATION: phase {}/lane {}: group {} ({}) reads from later group {} ({})",
                                        phase_idx, lane_idx, gi, op, src_gi, src_op);
                                }
                            }
                        }
                    }
                }
            }

            for span in &phase.spans {
                for mapping in &span.outputs {
                    produced_ranges.push((mapping.main_base.0, mapping.count));
                }
            }
            produced_ranges.sort();
        }

        let val_time = t_val.elapsed();
        if errors == 0 {
            println!("    V3C TOPOLOGY: VALID ({:.1}ms)", val_time.as_secs_f64() * 1e3);
        } else {
            println!("    V3C TOPOLOGY: {} VIOLATIONS ({:.1}ms)", errors, val_time.as_secs_f64() * 1e3);
        }
        // ---- Span-by-span eval using trusted NanoEval ----
        if std::env::var("EVAL_SPANS").ok().as_deref() == Some("1") {
            use whisper_tensor::nano_graph::eval::NanoEval;
            use whisper_tensor::numeric_scalar::NumericScalar;

            println!("\n    === Span-by-span NanoEval ===");

            // Need full lower for numeric_overrides
            let t0 = Instant::now();
            let full_result = whisper_tensor::nano_graph::lower::lower_with_info(&milli_graph, &all_infos).unwrap();
            eprintln!("    Full lower: {:.1}s", t0.elapsed().as_secs_f64());

            // Shared f32 buffer (30GB for GPT-2)
            let num_main_atoms = result.graph.num_atoms() as usize;
            let f32_buffer_gb = num_main_atoms as f64 * 4.0 / (1024.0 * 1024.0 * 1024.0);
            println!("    Shared buffer: {:.1} GB ({} atoms)", f32_buffer_gb, num_main_atoms);

            if f32_buffer_gb > 120.0 {
                println!("    SKIPPING: buffer too large");
            } else {
                let t0 = Instant::now();
                let mut shared = vec![0.0f32; num_main_atoms];

                // Pre-fill from numeric_overrides
                for (&idx, scalar) in &full_result.numeric_overrides {
                    shared[idx as usize] = scalar.to_f64() as f32;
                }
                // Fill user inputs
                let mut backend_eval = whisper_tensor::backends::eval_backend::EvalBackend::NDArray;
                let sym_graph2 = model.get_symbolic_graph();
                let tensor_store2 = model.get_tensor_store();
                let initialized2 = sym_graph2.get_initialized_tensors(tensor_store2);
                let mut milli_inputs2: HashMap<GlobalId, whisper_tensor::numeric_tensor::NumericTensor<whisper_tensor::DynRank>> = HashMap::new();
                for (id, tensor) in initialized2 {
                    milli_inputs2.insert(id, tensor);
                }
                for (name, (dtype, shape_dims)) in &input_info {
                    let shape: Vec<u64> = shape_dims.iter().map(|d| d.unwrap_or(4)).collect();
                    let numel: usize = shape.iter().product::<u64>() as usize;
                    if let Some(id) = tensors_by_name.get(name) {
                        let data: Vec<i64> = (0..numel as i64).collect();
                        let tensor = whisper_tensor::numeric_tensor::NumericTensor::from_vec_shape(
                            data, shape.iter().map(|&s| s as usize).collect()
                        ).unwrap();
                        milli_inputs2.insert(*id, tensor.clone());
                        // Also fill shared buffer
                        if let Some(tam) = full_result.tensor_map.get(id) {
                            let f32_t = tensor.cast(whisper_tensor::dtype::DType::F32, &mut backend_eval).unwrap();
                            let flat = f32_t.flatten().unwrap();
                            let nd = flat.to_ndarray().unwrap();
                            let v: Vec<f32> = nd.try_into().unwrap();
                            for (i, &val) in v.iter().enumerate() {
                                let atom_id = tam.atom_id_for_element(i as u64);
                                shared[atom_id.0 as usize] = val;
                            }
                        }
                    }
                }
                eprintln!("    Buffer filled: {:.1}s", t0.elapsed().as_secs_f64());

                // Pre-eval: check all span graphs have resolvable InputRefs
                let t_check = Instant::now();
                let mut input_errors = 0usize;
                for (phase_idx, phase) in plan.phases.iter().enumerate() {
                    for (lane_idx, span) in phase.spans.iter().enumerate() {
                        let sg = span.graph.groups();
                        let span_max = span.graph.num_atoms();
                        // Build set of atom ranges in this span
                        for group in sg {
                            for input in &group.inputs {
                                // Check first and last atom resolve within span
                                for test_i in [0u64, group.count.saturating_sub(1)] {
                                    let resolved = input.resolve(test_i, 0);
                                    if resolved.0 >= span_max {
                                        input_errors += 1;
                                        if input_errors <= 5 {
                                            let op = format!("{:?}", group.op).chars().take_while(|c| *c != ' ' && *c != '{').collect::<String>();
                                            println!("    SPAN INPUT ERROR: phase {}/lane {}: group base={:?} ({}) atom {} resolves to {:?} (max={})",
                                                phase_idx, lane_idx, group.base_id, op, test_i, resolved, span_max);
                                        }
                                    }
                                }
                                // Also check reduce stride if applicable
                                match &group.op {
                                    ScalarOp::ReduceSum { reduce_count, reduce_stride, .. }
                                    | ScalarOp::ReduceMax { reduce_count, reduce_stride, .. } => {
                                        if *reduce_count > 0 && *reduce_stride != 0 {
                                            let base = input.resolve(0, 0);
                                            let last = (base.0 as i64 + (*reduce_count as i64 - 1) * reduce_stride) as u64;
                                            if last >= span_max {
                                                input_errors += 1;
                                                if input_errors <= 5 {
                                                    let op = format!("{:?}", group.op).chars().take_while(|c| *c != ' ' && *c != '{').collect::<String>();
                                                    println!("    SPAN REDUCE ERROR: phase {}/lane {}: group base={:?} ({}) reduce reaches atom {} (max={})",
                                                        phase_idx, lane_idx, group.base_id, op, last, span_max);
                                                }
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
                println!("    Span input check: {} errors ({:.1}ms)", input_errors, t_check.elapsed().as_secs_f64() * 1e3);
                if input_errors > 0 {
                    println!("    SKIPPING eval due to span input errors");
                } else {

                // Execute each phase's spans using NanoEval
                let t0 = Instant::now();
                for (phase_idx, phase) in plan.phases.iter().enumerate() {
                    for (lane_idx, span) in phase.spans.iter().enumerate() {
                        if span.graph.num_groups() == 0 { continue; }

                        // Build overrides for this span from shared buffer
                        let mut span_overrides: HashMap<u64, NumericScalar> = HashMap::new();
                        for mapping in &span.inputs {
                            for i in 0..mapping.count {
                                let main_atom = mapping.main_base.0 + i;
                                let span_atom = mapping.span_base.0 + i;
                                span_overrides.insert(span_atom, NumericScalar::F32(shared[main_atom as usize]));
                            }
                        }

                        // Eval the span
                        let span_result = NanoEval::eval(&span.graph, &span_overrides);

                        // Write outputs back to shared buffer
                        for mapping in &span.outputs {
                            for i in 0..mapping.count {
                                let span_atom = mapping.span_base.0 + i;
                                let main_atom = mapping.main_base.0 + i;
                                shared[main_atom as usize] = span_result.get(whisper_tensor::nano_graph::AtomId(span_atom)) as f32;
                            }
                        }
                    }
                }
                let eval_time = t0.elapsed();
                println!("    Span eval: {:.1}s ({} phases)", eval_time.as_secs_f64(), plan.phases.len());

                // Compare against milli interpreter
                let t0 = Instant::now();
                let milli_outputs = whisper_tensor::compiler::interpret_milli_graph(&milli_graph, &milli_inputs2).unwrap();
                eprintln!("    Milli interpreter: {:.1}s", t0.elapsed().as_secs_f64());

                let reverse_output_map: HashMap<GlobalId, GlobalId> = milli_graph
                    .output_map.as_ref()
                    .map(|m| m.iter().map(|(&int, &ext)| (ext, int)).collect())
                    .unwrap_or_default();

                let mut max_abs_error: f64 = 0.0;
                let mut total_compared = 0u64;
                for (ext_id, milli_tensor) in &milli_outputs {
                    let internal_id = reverse_output_map.get(ext_id).unwrap_or(ext_id);
                    let tam = full_result.tensor_map.get(internal_id)
                        .or_else(|| full_result.tensor_map.get(ext_id));
                    let Some(tam) = tam else { continue };

                    let f32_tensor = milli_tensor.cast(whisper_tensor::dtype::DType::F32, &mut backend_eval).unwrap();
                    let flat = f32_tensor.flatten().unwrap();
                    let nd = flat.to_ndarray().unwrap();
                    let milli_vals: Vec<f32> = nd.try_into().unwrap();

                    let mut local_max = 0.0f64;
                    for (i, &milli_val) in milli_vals.iter().enumerate() {
                        let atom_id = tam.atom_id_for_element(i as u64);
                        let span_val = shared[atom_id.0 as usize];
                        let err = (milli_val - span_val).abs() as f64;
                        local_max = local_max.max(err);
                        total_compared += 1;
                    }
                    max_abs_error = max_abs_error.max(local_max);
                }

                println!("    Compared {} elements, max_abs_error={:.6e}", total_compared, max_abs_error);
                if max_abs_error < 1e-2 {
                    println!("    SPAN EVAL: PASS");
                } else {
                    println!("    SPAN EVAL: MISMATCH");
                }
                } // end else (no input errors)
            }
        }
    }
    if which == "v4a" || which == "all_v4" {
        use whisper_tensor::compiler::attempts::v13_claude::nano_plan_v4a;
        let t0 = Instant::now();
        let plan = nano_plan_v4a::plan_spans(&result.graph, num_lanes);
        let elapsed = t0.elapsed();
        let ss: Vec<Vec<SS>> = plan.phases.iter().map(|p| p.spans.iter().map(|s| SS {
            ng: s.graph.num_groups(), na: s.graph.num_atoms(), ni: s.inputs.len(), no: s.outputs.len(),
        }).collect()).collect();
        print_span_summary("v4a", &ss, plan.num_lanes, elapsed);
        validate_v4_plan!("v4a", plan, &result.graph);
    }
    if which == "v4b" || which == "all_v4" {
        use whisper_tensor::compiler::attempts::v13_claude::nano_plan_v4b;
        let t0 = Instant::now();
        let plan = nano_plan_v4b::plan_spans(&result.graph, num_lanes);
        let elapsed = t0.elapsed();
        let ss: Vec<Vec<SS>> = plan.phases.iter().map(|p| p.spans.iter().map(|s| SS {
            ng: s.graph.num_groups(), na: s.graph.num_atoms(), ni: s.inputs.len(), no: s.outputs.len(),
        }).collect()).collect();
        print_span_summary("v4b", &ss, plan.num_lanes, elapsed);
        validate_v4_plan!("v4b", plan, &result.graph);

        // Span-by-span eval if requested
        if std::env::var("EVAL_SPANS").ok().as_deref() == Some("1") {
            use whisper_tensor::nano_graph::eval::NanoEval;
            use whisper_tensor::numeric_scalar::NumericScalar;

            println!("\n    === v4b Span-by-span NanoEval ===");
            let t0 = Instant::now();
            let full_result = whisper_tensor::nano_graph::lower::lower_with_info(&milli_graph, &all_infos).unwrap();
            eprintln!("    Full lower: {:.1}s", t0.elapsed().as_secs_f64());

            let num_main_atoms = result.graph.num_atoms() as usize;
            let f32_gb = num_main_atoms as f64 * 4.0 / (1024.0 * 1024.0 * 1024.0);
            println!("    Buffer: {:.1} GB", f32_gb);

            if f32_gb > 120.0 {
                println!("    SKIPPING: buffer too large");
            } else {
                let mut shared = vec![0.0f32; num_main_atoms];

                // Fill Literal atoms from ScalarOp values
                for g in result.graph.groups() {
                    if let ScalarOp::Literal(scalar) = &g.op {
                        let val = scalar.to_f64() as f32;
                        for i in 0..g.count { shared[(g.base_id.0 + i) as usize] = val; }
                    }
                }
                // Override with numeric_overrides
                for (&idx, scalar) in &full_result.numeric_overrides {
                    shared[idx as usize] = scalar.to_f64() as f32;
                }
                // Fill user inputs
                let sym_graph2 = model.get_symbolic_graph();
                let tensor_store2 = model.get_tensor_store();
                let initialized2 = sym_graph2.get_initialized_tensors(tensor_store2);
                let mut milli_inputs2: HashMap<GlobalId, whisper_tensor::numeric_tensor::NumericTensor<whisper_tensor::DynRank>> = HashMap::new();
                for (id, tensor) in initialized2 { milli_inputs2.insert(id, tensor); }
                let mut backend_v4 = whisper_tensor::backends::eval_backend::EvalBackend::NDArray;
                for (name, (_dtype, _shape_dims)) in &input_info {
                    let shape: Vec<u64> = _shape_dims.iter().map(|d| d.unwrap_or(4)).collect();
                    let numel: usize = shape.iter().product::<u64>() as usize;
                    if let Some(id) = tensors_by_name.get(name) {
                        let data: Vec<i64> = (0..numel as i64).collect();
                        let tensor = whisper_tensor::numeric_tensor::NumericTensor::from_vec_shape(
                            data, shape.iter().map(|&s| s as usize).collect()
                        ).unwrap();
                        milli_inputs2.insert(*id, tensor.clone());
                        if let Some(tam) = full_result.tensor_map.get(id) {
                            let f32_t = tensor.cast(whisper_tensor::dtype::DType::F32, &mut backend_v4).unwrap();
                            let flat = f32_t.flatten().unwrap();
                            let nd = flat.to_ndarray().unwrap();
                            let v: Vec<f32> = nd.try_into().unwrap();
                            for (i, &val) in v.iter().enumerate() {
                                let atom_id = tam.atom_id_for_element(i as u64);
                                shared[atom_id.0 as usize] = val;
                            }
                        }
                    }
                }
                eprintln!("    Buffer filled");

                // Execute spans phase by phase using flat f32 buffers (no HashMap)
                let t0 = Instant::now();
                let mut span_errors = 0usize;
                for (pi, phase) in plan.phases.iter().enumerate() {
                    for (li, span) in phase.spans.iter().enumerate() {
                        if span.graph.num_groups() == 0 { continue; }

                        let sn = span.graph.num_atoms() as usize;
                        let mut sbuf = vec![0.0f32; sn];

                        // Fill Literal atoms from span's ScalarOp values
                        for g in span.graph.groups() {
                            if let ScalarOp::Literal(scalar) = &g.op {
                                let val = scalar.to_f64() as f32;
                                for i in 0..g.count {
                                    sbuf[(g.base_id.0 + i) as usize] = val;
                                }
                            }
                        }
                        // Bulk copy inputs from shared buffer
                        for mapping in &span.inputs {
                            let src_start = mapping.main_base.0 as usize;
                            let dst_start = mapping.span_base.0 as usize;
                            let count = mapping.count as usize;
                            sbuf[dst_start..dst_start + count]
                                .copy_from_slice(&shared[src_start..src_start + count]);
                        }

                        // Eval each group using f32 arithmetic
                        let sg = span.graph.groups();
                        let mut panic_group = None;
                        for (gi, group) in sg.iter().enumerate() {
                            if matches!(&group.op, ScalarOp::Literal(_)) { continue; }
                            for i in 0..group.count {
                                let aidx = (group.base_id.0 + i) as usize;
                                let val = match &group.op {
                                    ScalarOp::Literal(_) => continue,
                                    ScalarOp::Identity { .. } => {
                                        sbuf[group.inputs[0].resolve(i, 0).0 as usize]
                                    }
                                    ScalarOp::Binary { op, .. } => {
                                        let a = sbuf[group.inputs[0].resolve(i, 0).0 as usize];
                                        let b = sbuf[group.inputs[1].resolve(i, 0).0 as usize];
                                        use whisper_tensor::nano_graph::ScalarBinOp::*;
                                        match op {
                                            Add => a + b, Sub => a - b, Mul => a * b,
                                            Div => { if b == 0.0 && panic_group.is_none() { panic_group = Some((pi, li, gi, "div by zero")); } a / b },
                                            Max => a.max(b), Min => a.min(b), Pow => a.powf(b), Mod => a % b,
                                            Equal => if a == b { 1.0 } else { 0.0 },
                                            Greater => if a > b { 1.0 } else { 0.0 },
                                            GreaterOrEqual => if a >= b { 1.0 } else { 0.0 },
                                            Less => if a < b { 1.0 } else { 0.0 },
                                            LessOrEqual => if a <= b { 1.0 } else { 0.0 },
                                            And => if a != 0.0 && b != 0.0 { 1.0 } else { 0.0 },
                                            Or => if a != 0.0 || b != 0.0 { 1.0 } else { 0.0 },
                                            Xor => if (a != 0.0) ^ (b != 0.0) { 1.0 } else { 0.0 },
                                        }
                                    }
                                    ScalarOp::Unary { op, .. } => {
                                        let x = sbuf[group.inputs[0].resolve(i, 0).0 as usize];
                                        use whisper_tensor::nano_graph::ScalarUnaryOp::*;
                                        match op {
                                            Neg => -x, Abs => x.abs(), Exp => x.exp(), Ln => x.ln(),
                                            Sqrt => x.sqrt(), Reciprocal => 1.0 / x, Tanh => x.tanh(),
                                            Floor => x.floor(), Ceil => x.ceil(),
                                        }
                                    }
                                    ScalarOp::Select { .. } => {
                                        let c = sbuf[group.inputs[0].resolve(i, 0).0 as usize];
                                        if c != 0.0 { sbuf[group.inputs[1].resolve(i, 0).0 as usize] }
                                        else { sbuf[group.inputs[2].resolve(i, 0).0 as usize] }
                                    }
                                    ScalarOp::ReduceSum { reduce_count, reduce_stride, .. } => {
                                        let base = group.inputs[0].resolve(i, 0);
                                        let mut acc = 0.0f32;
                                        for k in 0..*reduce_count {
                                            let src = (base.0 as i64 + k as i64 * reduce_stride) as usize;
                                            acc += sbuf[src];
                                        }
                                        acc
                                    }
                                    ScalarOp::ReduceMax { reduce_count, reduce_stride, .. } => {
                                        let base = group.inputs[0].resolve(i, 0);
                                        let mut acc = f32::NEG_INFINITY;
                                        for k in 0..*reduce_count {
                                            let src = (base.0 as i64 + k as i64 * reduce_stride) as usize;
                                            acc = acc.max(sbuf[src]);
                                        }
                                        acc
                                    }
                                    ScalarOp::IndirectLoad { table_base, .. } => {
                                        let idx = sbuf[group.inputs[0].resolve(i, 0).0 as usize];
                                        sbuf[table_base.0 as usize + idx as usize]
                                    }
                                };
                                sbuf[aidx] = val;
                            }
                        }
                        if let Some((p, l, g, msg)) = panic_group {
                            println!("    WARN: phase {}/lane {} group {}: {}", p, l, g, msg);
                            span_errors += 1;
                        }

                        // Bulk copy outputs back to shared buffer
                        for mapping in &span.outputs {
                            let src_start = mapping.span_base.0 as usize;
                            let dst_start = mapping.main_base.0 as usize;
                            let count = mapping.count as usize;
                            shared[dst_start..dst_start + count]
                                .copy_from_slice(&sbuf[src_start..src_start + count]);
                        }
                    }
                }
                let eval_time = t0.elapsed();
                println!("    Span eval: {:.1}s ({} phases)", eval_time.as_secs_f64(), plan.phases.len());

                // Compare against milli
                let t0 = Instant::now();
                let milli_outputs = whisper_tensor::compiler::interpret_milli_graph(&milli_graph, &milli_inputs2).unwrap();
                eprintln!("    Milli: {:.1}s", t0.elapsed().as_secs_f64());

                let reverse_map: HashMap<GlobalId, GlobalId> = milli_graph.output_map.as_ref()
                    .map(|m| m.iter().map(|(&i, &e)| (e, i)).collect()).unwrap_or_default();
                let mut max_err: f64 = 0.0;
                let mut total = 0u64;
                for (ext_id, mt) in &milli_outputs {
                    let int_id = reverse_map.get(ext_id).unwrap_or(ext_id);
                    let tam = full_result.tensor_map.get(int_id).or_else(|| full_result.tensor_map.get(ext_id));
                    let Some(tam) = tam else { continue };
                    let f32_t = mt.cast(whisper_tensor::dtype::DType::F32, &mut backend_v4).unwrap();
                    let flat = f32_t.flatten().unwrap();
                    let nd = flat.to_ndarray().unwrap();
                    let vals: Vec<f32> = nd.try_into().unwrap();
                    for (i, &mv) in vals.iter().enumerate() {
                        let aid = tam.atom_id_for_element(i as u64);
                        let sv = shared[aid.0 as usize];
                        max_err = max_err.max((mv - sv).abs() as f64);
                        total += 1;
                    }
                }
                println!("    Compared {} elements, max_abs_error={:.6e}", total, max_err);
                if max_err < 1e-2 { println!("    V4B EVAL: PASS"); }
                else { println!("    V4B EVAL: MISMATCH"); }
            }
        }
    }
    if which == "v4c" || which == "all_v4" {
        use whisper_tensor::compiler::attempts::v13_claude::nano_plan_v4c;
        let t0 = Instant::now();
        let plan = nano_plan_v4c::plan_execution_spans(&result.graph, num_lanes);
        let elapsed = t0.elapsed();
        let ss: Vec<Vec<SS>> = plan.phases.iter().map(|p| p.spans.iter().map(|s| SS {
            ng: s.graph.num_groups(), na: s.graph.num_atoms(), ni: s.inputs.len(), no: s.outputs.len(),
        }).collect()).collect();
        print_span_summary("v4c", &ss, plan.num_lanes, elapsed);
        validate_v4_plan!("v4c", plan, &result.graph);
    }
    if which == "d" || which == "all" {
        use whisper_tensor::compiler::attempts::v13_claude::nano_plan_spans_d;
        let t0 = Instant::now();
        let plan = nano_plan_spans_d::plan_spans(&result.graph, num_lanes);
        let elapsed = t0.elapsed();
        let ss: Vec<Vec<SS>> = plan.phases.iter().map(|p| p.spans.iter().map(|s| SS {
            ng: s.graph.num_groups(), na: s.graph.num_atoms(), ni: s.inputs.len(), no: s.outputs.len(),
        }).collect()).collect();
        print_span_summary("spans_d", &ss, plan.num_lanes, elapsed);
    }
    if which == "v3d" || which == "all" {
        use whisper_tensor::compiler::attempts::v13_claude::nano_plan_v3d;
        let t0 = Instant::now();
        let plan = nano_plan_v3d::plan_spans(&result.graph, num_lanes);
        let elapsed = t0.elapsed();
        let ss: Vec<Vec<SS>> = plan.phases.iter().map(|p| p.spans.iter().map(|s| SS {
            ng: s.graph.num_groups(), na: s.graph.num_atoms(), ni: s.inputs.len(), no: s.outputs.len(),
        }).collect()).collect();
        print_span_summary("v3d", &ss, plan.num_lanes, elapsed);

        // Topology validation
        let t_val = Instant::now();
        let mut errors = 0usize;
        let mut produced_ranges: Vec<(u64, u64)> = Vec::new();
        for g in result.graph.groups() {
            if matches!(&g.op, ScalarOp::Literal(_)) && g.inputs.is_empty() {
                produced_ranges.push((g.base_id.0, g.count));
            }
        }
        produced_ranges.sort();

        let range_contains = |ranges: &[(u64, u64)], atom: u64| -> bool {
            match ranges.binary_search_by(|&(base, _)| base.cmp(&atom)) {
                Ok(_) => true,
                Err(0) => false,
                Err(i) => {
                    let (base, count) = ranges[i - 1];
                    atom < base + count
                }
            }
        };

        for (phase_idx, phase) in plan.phases.iter().enumerate() {
            for (lane_idx, span) in phase.spans.iter().enumerate() {
                for mapping in &span.inputs {
                    if !range_contains(&produced_ranges, mapping.main_base.0) {
                        errors += 1;
                        if errors <= 10 {
                            println!("  V3D VIOLATION: phase {}/lane {}: input base {:?} (count={}) not available",
                                phase_idx, lane_idx, mapping.main_base, mapping.count);
                        }
                    }
                }

                let span_groups = span.graph.groups();
                let span_bases: Vec<u64> = span_groups.iter().map(|g| g.base_id.0).collect();
                for (gi, group) in span_groups.iter().enumerate() {
                    for input in &group.inputs {
                        let src = input.resolve(0, 0);
                        let src_gi = match span_bases.binary_search(&src.0) {
                            Ok(i) => Some(i),
                            Err(0) => None,
                            Err(i) => {
                                let candidate = i - 1;
                                if src.0 < span_groups[candidate].base_id.0 + span_groups[candidate].count {
                                    Some(candidate)
                                } else { None }
                            }
                        };
                        if let Some(src_gi) = src_gi {
                            if src_gi > gi && !matches!(&span_groups[src_gi].op, ScalarOp::Literal(_)) {
                                errors += 1;
                                if errors <= 10 {
                                    let op = format!("{:?}", group.op).chars().take_while(|c| *c != ' ' && *c != '{').collect::<String>();
                                    let src_op = format!("{:?}", span_groups[src_gi].op).chars().take_while(|c| *c != ' ' && *c != '{').collect::<String>();
                                    println!("  V3D TOPO VIOLATION: phase {}/lane {}: group {} ({}) reads from later group {} ({})",
                                        phase_idx, lane_idx, gi, op, src_gi, src_op);
                                }
                            }
                        }
                    }
                }
            }

            for span in &phase.spans {
                for mapping in &span.outputs {
                    produced_ranges.push((mapping.main_base.0, mapping.count));
                }
            }
            produced_ranges.sort();
        }

        let val_time = t_val.elapsed();
        if errors == 0 {
            println!("    TOPOLOGY: VALID ({:.1}ms)", val_time.as_secs_f64() * 1e3);
        } else {
            println!("    TOPOLOGY: {} VIOLATIONS ({:.1}ms)", errors, val_time.as_secs_f64() * 1e3);
        }
    }
}

struct SS { ng: usize, na: u64, ni: usize, no: usize }

/// Span topology validation using extracted data.
fn validate_span_topology_raw(
    name: &str,
    // phases[phase_idx][span_idx] = (graph, input_ranges, output_ranges)
    phases: &[Vec<(&NanoGraph, Vec<(u64, u64)>, Vec<(u64, u64)>)>],
    original: &NanoGraph,
) {
    let t0 = Instant::now();
    let mut produced_ranges: Vec<(u64, u64)> = Vec::new();
    // Seed with all Literal group ranges
    for g in original.groups() {
        if matches!(&g.op, ScalarOp::Literal(_)) && g.inputs.is_empty() {
            produced_ranges.push((g.base_id.0, g.count));
        }
    }
    produced_ranges.sort();

    let range_contains = |ranges: &[(u64, u64)], atom: u64| -> bool {
        match ranges.binary_search_by(|&(base, _)| base.cmp(&atom)) {
            Ok(_) => true,
            Err(0) => false,
            Err(i) => {
                let (base, count) = ranges[i - 1];
                atom < base + count
            }
        }
    };

    let mut input_errors = 0usize;
    let mut topo_errors = 0usize;
    let mut resolve_errors = 0usize;

    for (pi, phase) in phases.iter().enumerate() {
        for (li, (sg, inputs, _outputs)) in phase.iter().enumerate() {
            if sg.num_groups() == 0 { continue; }

            // Check inputs are available
            for &(main_base, count) in inputs {
                if !range_contains(&produced_ranges, main_base) {
                    input_errors += 1;
                    if input_errors <= 5 {
                        println!("    {} INPUT ERR: phase {}/lane {}: main atom {} (count={}) not available",
                            name, pi, li, main_base, count);
                    }
                }
            }

            // Check within-span topo order and InputRef resolution
            let span_groups = sg.groups();
            let span_bases: Vec<u64> = span_groups.iter().map(|g| g.base_id.0).collect();
            for (gi, group) in span_groups.iter().enumerate() {
                if matches!(&group.op, ScalarOp::Literal(_)) { continue; }
                for input in &group.inputs {
                    let resolved = input.resolve(0, 0);
                    if resolved.0 >= sg.num_atoms() {
                        resolve_errors += 1;
                        if resolve_errors <= 3 {
                            let op = format!("{:?}", group.op).chars().take_while(|c| *c != ' ' && *c != '{').collect::<String>();
                            println!("    {} RESOLVE ERR: phase {}/lane {}: group {} ({}) resolves to {} (max={})",
                                name, pi, li, gi, op, resolved.0, sg.num_atoms());
                        }
                    }
                }
                // Check ReduceSum stride
                match &group.op {
                    ScalarOp::ReduceSum { reduce_count, reduce_stride, .. }
                    | ScalarOp::ReduceMax { reduce_count, reduce_stride, .. } => {
                        if *reduce_count > 0 && *reduce_stride != 0 {
                            let base = group.inputs[0].resolve(0, 0);
                            let last = (base.0 as i64 + (*reduce_count as i64 - 1) * reduce_stride) as u64;
                            if last >= sg.num_atoms() {
                                resolve_errors += 1;
                                if resolve_errors <= 3 {
                                    let op = format!("{:?}", group.op).chars().take_while(|c| *c != ' ' && *c != '{').collect::<String>();
                                    println!("    {} REDUCE ERR: phase {}/lane {}: {} reaches atom {} (max={})",
                                        name, pi, li, op, last, sg.num_atoms());
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        // Add this phase's outputs to produced
        for (_sg, _inputs, outputs) in phase {
            for &(main_base, count) in outputs {
                produced_ranges.push((main_base, count));
            }
        }
        produced_ranges.sort();
    }

    let total = input_errors + topo_errors + resolve_errors;
    let elapsed = t0.elapsed();
    if total == 0 {
        println!("    {} TOPOLOGY: VALID ({:.1}ms)", name, elapsed.as_secs_f64() * 1e3);
    } else {
        println!("    {} TOPOLOGY: {} errors ({} input, {} resolve) ({:.1}ms)",
            name, total, input_errors, resolve_errors, elapsed.as_secs_f64() * 1e3);
    }
}

// Old macro/function definitions removed (moved to top of file)

fn print_span_summary(name: &str, phases: &[Vec<SS>], num_lanes: usize, elapsed: std::time::Duration) {
    let num_phases = phases.len();
    let total_spans: usize = phases.iter().map(|p| p.len()).sum();
    let total_groups: usize = phases.iter().flat_map(|p| p.iter()).map(|s| s.ng).sum();
    let total_atoms: u64 = phases.iter().flat_map(|p| p.iter()).map(|s| s.na).sum();
    let empty_spans: usize = phases.iter().flat_map(|p| p.iter()).filter(|s| s.ng == 0).count();

    let mut max_imbalance: f64 = 0.0;
    for phase in phases {
        let lane_atoms: Vec<u64> = phase.iter().map(|s| s.na).collect();
        let mx = lane_atoms.iter().copied().max().unwrap_or(0);
        let mn = lane_atoms.iter().copied().filter(|&a| a > 0).min().unwrap_or(1);
        if mn > 0 { max_imbalance = max_imbalance.max(mx as f64 / mn as f64); }
    }

    let total_inputs: usize = phases.iter().flat_map(|p| p.iter()).map(|s| s.ni).sum();
    let total_outputs: usize = phases.iter().flat_map(|p| p.iter()).map(|s| s.no).sum();

    println!("  [{}] {:.1}ms, {} lanes, {} phases, {} spans ({} empty)",
        name, elapsed.as_secs_f64() * 1e3, num_lanes, num_phases, total_spans, empty_spans);
    println!("    {} groups, {:.1}B atoms, max_imbalance={:.1}x",
        total_groups, total_atoms as f64 / 1e9, max_imbalance);
    println!("    total_inputs={}, total_outputs={}", total_inputs, total_outputs);
}
