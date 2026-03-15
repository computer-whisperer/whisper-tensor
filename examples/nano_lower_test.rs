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

    {
        use whisper_tensor::compiler::attempts::v13_claude::nano_plan_spans_a;
        let t0 = Instant::now();
        let plan = nano_plan_spans_a::plan_execution(&result.graph, num_lanes);
        let elapsed = t0.elapsed();
        let ss: Vec<Vec<SS>> = plan.phases.iter().map(|p| p.spans.iter().map(|s| SS {
            ng: s.graph.num_groups(), na: s.graph.num_atoms(), ni: s.inputs.len(), no: s.outputs.len(),
        }).collect()).collect();
        print_span_summary("spans_a", &ss, plan.num_lanes, elapsed);
    }
    {
        use whisper_tensor::compiler::attempts::v13_claude::nano_plan_spans_b;
        let t0 = Instant::now();
        let plan = nano_plan_spans_b::plan_spans(&result.graph, num_lanes);
        let elapsed = t0.elapsed();
        let ss: Vec<Vec<SS>> = plan.phases.iter().map(|p| p.spans.iter().map(|s| SS {
            ng: s.graph.num_groups(), na: s.graph.num_atoms(), ni: s.inputs.len(), no: s.outputs.len(),
        }).collect()).collect();
        print_span_summary("spans_b", &ss, plan.num_lanes, elapsed);
    }
    {
        use whisper_tensor::compiler::attempts::v13_claude::nano_plan_spans_c;
        let t0 = Instant::now();
        let plan = nano_plan_spans_c::plan_execution_spans(&result.graph, num_lanes);
        let elapsed = t0.elapsed();
        let ss: Vec<Vec<SS>> = plan.phases.iter().map(|p| p.spans.iter().map(|s| SS {
            ng: s.graph.num_groups(), na: s.graph.num_atoms(), ni: s.inputs.len(), no: s.outputs.len(),
        }).collect()).collect();
        print_span_summary("spans_c", &ss, plan.num_lanes, elapsed);
    }
    {
        use whisper_tensor::compiler::attempts::v13_claude::nano_plan_spans_d;
        let t0 = Instant::now();
        let plan = nano_plan_spans_d::plan_spans(&result.graph, num_lanes);
        let elapsed = t0.elapsed();
        let ss: Vec<Vec<SS>> = plan.phases.iter().map(|p| p.spans.iter().map(|s| SS {
            ng: s.graph.num_groups(), na: s.graph.num_atoms(), ni: s.inputs.len(), no: s.outputs.len(),
        }).collect()).collect();
        print_span_summary("spans_d", &ss, plan.num_lanes, elapsed);
    }
}

struct SS { ng: usize, na: u64, ni: usize, no: usize }

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
