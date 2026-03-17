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
        let raw: Vec<Vec<(&NanoGraph, Vec<(u64, u64)>, Vec<(u64, u64)>)>> = $plan
            .phases
            .iter()
            .map(|phase| {
                phase
                    .spans
                    .iter()
                    .map(|span| {
                        let inputs: Vec<(u64, u64)> = span
                            .inputs
                            .iter()
                            .map(|m| (m.main_base.0, m.count))
                            .collect();
                        let outputs: Vec<(u64, u64)> = span
                            .outputs
                            .iter()
                            .map(|m| (m.main_base.0, m.count))
                            .collect();
                        (&span.graph as &NanoGraph, inputs, outputs)
                    })
                    .collect()
            })
            .collect();
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
    let mut n_full = 0usize;
    let mut n_shape_only = 0usize;
    for (id, tensor) in &initialized {
        if tensor.num_elements() <= 1024 {
            // Small constants (axes, indices, shape values): keep full data
            // so infer_all can resolve Shape/Gather/Reshape ops.
            all_infos.insert(*id, TensorInfo::from(tensor.clone()));
            n_full += 1;
        } else {
            // Large weight matrices: shape+dtype only.
            let shape: Vec<u64> = tensor.shape().to_vec();
            let dtype = tensor.dtype();
            all_infos.insert(*id, TensorInfo::from_dtype_and_shape(dtype, &shape));
            n_shape_only += 1;
        }
    }
    println!(
        "Tensor info: {} full (small constants), {} shape-only (weights)",
        n_full, n_shape_only
    );

    // ---- Lower ----
    let t0 = Instant::now();
    let result = whisper_tensor::nano_graph::lower::lower(&milli_graph, &all_infos).unwrap();
    eprintln!("lower: {:.1}s", t0.elapsed().as_secs_f64());

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
        for (kind, count) in sorted {
            println!("  {:>4}x  {}", count, kind);
        }
    }

    // ---- Span-Based Execution Plans ----
    let num_lanes = std::env::var("NUM_LANES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    println!("\n=== Span-Based Plans (num_lanes={}) ===", num_lanes);

    let which = std::env::var("SPAN_PLANNER").unwrap_or("all".to_string());
    if which == "c" {
        use whisper_tensor::compiler::attempts::v13_claude::plan::spans_c;
        let t0 = Instant::now();
        let plan = spans_c::plan_execution_spans(&result.graph, num_lanes);
        let elapsed = t0.elapsed();
        let ss: Vec<Vec<SS>> = plan
            .phases
            .iter()
            .map(|p| {
                p.spans
                    .iter()
                    .map(|s| SS {
                        ng: s.graph.num_groups(),
                        na: s.graph.num_atoms(),
                        ni: s.inputs.len(),
                        no: s.outputs.len(),
                    })
                    .collect()
            })
            .collect();
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
                            println!(
                                "  SPAN VIOLATION: phase {}/lane {}: input base {:?} (count={}) not available",
                                phase_idx, lane_idx, mapping.main_base, mapping.count
                            );
                            // Trace producer group for diagnostics
                            let main_groups = result.graph.groups();
                            let atom_val = mapping.main_base.0;
                            let idx = main_groups.partition_point(|g| g.base_id.0 <= atom_val);
                            if idx > 0 {
                                let g = &main_groups[idx - 1];
                                if atom_val < g.base_id.0 + g.count {
                                    let op_str =
                                        format!("{:?}", g.op).chars().take(60).collect::<String>();
                                    let mut in_phase = "not in any output".to_string();
                                    for (pi, ph) in plan.phases.iter().enumerate() {
                                        for sp in &ph.spans {
                                            for om in &sp.outputs {
                                                if om.main_base.0 <= atom_val
                                                    && atom_val < om.main_base.0 + om.count
                                                {
                                                    in_phase = format!("phase {}", pi);
                                                }
                                            }
                                        }
                                    }
                                    println!(
                                        "    group_idx={} base={:?} count={} op={} [{}]",
                                        idx - 1,
                                        g.base_id,
                                        g.count,
                                        op_str,
                                        in_phase
                                    );
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
                                if src.0
                                    < span_groups[candidate].base_id.0
                                        + span_groups[candidate].count
                                {
                                    Some(candidate)
                                } else {
                                    None
                                }
                            }
                        };
                        if let Some(src_gi) = src_gi {
                            if src_gi > gi
                                && !matches!(&span_groups[src_gi].op, ScalarOp::Literal(_))
                            {
                                errors += 1;
                                if errors <= 10 {
                                    let op = format!("{:?}", group.op)
                                        .chars()
                                        .take_while(|c| *c != ' ' && *c != '{')
                                        .collect::<String>();
                                    let src_op = format!("{:?}", span_groups[src_gi].op)
                                        .chars()
                                        .take_while(|c| *c != ' ' && *c != '{')
                                        .collect::<String>();
                                    println!(
                                        "  SPAN TOPO VIOLATION: phase {}/lane {}: group {} ({}) reads from later group {} ({})",
                                        phase_idx, lane_idx, gi, op, src_gi, src_op
                                    );
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
            println!(
                "    TOPOLOGY: VALID ({:.1}ms)",
                val_time.as_secs_f64() * 1e3
            );
        } else {
            println!(
                "    TOPOLOGY: {} VIOLATIONS ({:.1}ms)",
                errors,
                val_time.as_secs_f64() * 1e3
            );
        }
    }
    if which == "v4c" {
        use whisper_tensor::compiler::attempts::v13_claude::plan::v4c;
        let t0 = Instant::now();
        let plan = v4c::plan_execution_spans(&result.graph, num_lanes);
        let elapsed = t0.elapsed();
        let ss: Vec<Vec<SS>> = plan
            .phases
            .iter()
            .map(|p| {
                p.spans
                    .iter()
                    .map(|s| SS {
                        ng: s.graph.num_groups(),
                        na: s.graph.num_atoms(),
                        ni: s.inputs.len(),
                        no: s.outputs.len(),
                    })
                    .collect()
            })
            .collect();
        print_span_summary("v4c", &ss, plan.num_lanes, elapsed);
        validate_v4_plan!("v4c", plan, &result.graph);
    }
}

struct SS {
    ng: usize,
    na: u64,
    ni: usize,
    no: usize,
}

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
            if sg.num_groups() == 0 {
                continue;
            }

            // Check inputs are available
            for &(main_base, count) in inputs {
                if !range_contains(&produced_ranges, main_base) {
                    input_errors += 1;
                    if input_errors <= 5 {
                        println!(
                            "    {} INPUT ERR: phase {}/lane {}: main atom {} (count={}) not available",
                            name, pi, li, main_base, count
                        );
                    }
                }
            }

            // Check within-span topo order and InputRef resolution
            let span_groups = sg.groups();
            let span_bases: Vec<u64> = span_groups.iter().map(|g| g.base_id.0).collect();
            for (gi, group) in span_groups.iter().enumerate() {
                if matches!(&group.op, ScalarOp::Literal(_)) {
                    continue;
                }
                for input in &group.inputs {
                    let resolved = input.resolve(0, 0);
                    if resolved.0 >= sg.num_atoms() {
                        resolve_errors += 1;
                        if resolve_errors <= 3 {
                            let op = format!("{:?}", group.op)
                                .chars()
                                .take_while(|c| *c != ' ' && *c != '{')
                                .collect::<String>();
                            println!(
                                "    {} RESOLVE ERR: phase {}/lane {}: group {} ({}) resolves to {} (max={})",
                                name,
                                pi,
                                li,
                                gi,
                                op,
                                resolved.0,
                                sg.num_atoms()
                            );
                        }
                    }
                }
                // Check ReduceSum stride
                match &group.op {
                    ScalarOp::ReduceSum {
                        reduce_count,
                        reduce_stride,
                        ..
                    }
                    | ScalarOp::ReduceMax {
                        reduce_count,
                        reduce_stride,
                        ..
                    } => {
                        if *reduce_count > 0 && *reduce_stride != 0 {
                            let base = group.inputs[0].resolve(0, 0);
                            let last =
                                (base.0 as i64 + (*reduce_count as i64 - 1) * reduce_stride) as u64;
                            if last >= sg.num_atoms() {
                                resolve_errors += 1;
                                if resolve_errors <= 3 {
                                    let op = format!("{:?}", group.op)
                                        .chars()
                                        .take_while(|c| *c != ' ' && *c != '{')
                                        .collect::<String>();
                                    println!(
                                        "    {} REDUCE ERR: phase {}/lane {}: {} reaches atom {} (max={})",
                                        name,
                                        pi,
                                        li,
                                        op,
                                        last,
                                        sg.num_atoms()
                                    );
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
        println!(
            "    {} TOPOLOGY: VALID ({:.1}ms)",
            name,
            elapsed.as_secs_f64() * 1e3
        );
    } else {
        println!(
            "    {} TOPOLOGY: {} errors ({} input, {} resolve) ({:.1}ms)",
            name,
            total,
            input_errors,
            resolve_errors,
            elapsed.as_secs_f64() * 1e3
        );
    }
}

// Old macro/function definitions removed (moved to top of file)

fn print_span_summary(
    name: &str,
    phases: &[Vec<SS>],
    num_lanes: usize,
    elapsed: std::time::Duration,
) {
    let num_phases = phases.len();
    let total_spans: usize = phases.iter().map(|p| p.len()).sum();
    let total_groups: usize = phases.iter().flat_map(|p| p.iter()).map(|s| s.ng).sum();
    let total_atoms: u64 = phases.iter().flat_map(|p| p.iter()).map(|s| s.na).sum();
    let empty_spans: usize = phases
        .iter()
        .flat_map(|p| p.iter())
        .filter(|s| s.ng == 0)
        .count();

    let mut max_imbalance: f64 = 0.0;
    for phase in phases {
        let lane_atoms: Vec<u64> = phase.iter().map(|s| s.na).collect();
        let mx = lane_atoms.iter().copied().max().unwrap_or(0);
        let mn = lane_atoms
            .iter()
            .copied()
            .filter(|&a| a > 0)
            .min()
            .unwrap_or(1);
        if mn > 0 {
            max_imbalance = max_imbalance.max(mx as f64 / mn as f64);
        }
    }

    let total_inputs: usize = phases.iter().flat_map(|p| p.iter()).map(|s| s.ni).sum();
    let total_outputs: usize = phases.iter().flat_map(|p| p.iter()).map(|s| s.no).sum();

    println!(
        "  [{}] {:.1}ms, {} lanes, {} phases, {} spans ({} empty)",
        name,
        elapsed.as_secs_f64() * 1e3,
        num_lanes,
        num_phases,
        total_spans,
        empty_spans
    );
    println!(
        "    {} groups, {:.1}B atoms, max_imbalance={:.1}x",
        total_groups,
        total_atoms as f64 / 1e9,
        max_imbalance
    );
    println!(
        "    total_inputs={}, total_outputs={}",
        total_inputs, total_outputs
    );
}
