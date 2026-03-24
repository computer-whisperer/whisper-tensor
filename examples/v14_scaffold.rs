//! v14 execution scaffold: load GPT-2, lower to NanoGraph, prepare for partitioning.
//!
//! Usage:
//!   cargo run --release --example v14_scaffold -- test_models/gpt2-lm-head-10.onnx

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

use whisper_tensor::compiler::attempts::v14::types::*;
use whisper_tensor::compiler::op_census;
use whisper_tensor::dtype::DType;
use whisper_tensor::graph::GlobalId;
use whisper_tensor::model::Model;
use whisper_tensor::nano_graph::lower;
use whisper_tensor::numeric_dtype::NumericDType;
use whisper_tensor::tensor_info::TensorInfo;
use whisper_tensor_import::identify_and_load;
use whisper_tensor_import::onnx_graph::WeightStorageStrategy;

// ─── Process resource profiler ──────────────────────────────────────────────

mod profiler {
    use std::sync::{Arc, Mutex};
    use std::thread::{self, JoinHandle};
    use std::time::{Duration, Instant};

    struct Shared {
        current_phase: String,
        stop: bool,
    }

    pub struct Profiler {
        shared: Arc<Mutex<Shared>>,
        thread: Option<JoinHandle<()>>,
    }

    impl Profiler {
        /// Start the profiler. `rss_limit_mb` sets a hard RSS ceiling —
        /// the process exits immediately if RSS exceeds it.
        /// Set via RSS_LIMIT_MB env var (default: no limit).
        pub fn start(rss_limit_mb: Option<u64>) -> Self {
            let shared = Arc::new(Mutex::new(Shared {
                current_phase: "init".to_string(),
                stop: false,
            }));
            let shared2 = shared.clone();
            let start = Instant::now();
            let rss_limit_bytes = rss_limit_mb.map(|mb| mb * 1024 * 1024);

            eprintln!(
                "  {:<20} {:>7} {:>10} {:>10} {:>10} {:>10} {:>7}",
                "Phase", "Wall", "RSS Start", "RSS Peak", "RSS End", "RSS Δ", "CPU"
            );

            let thread = thread::spawn(move || {
                let mut prev_phase = String::new();
                let mut phase_start_elapsed = Duration::ZERO;
                let mut phase_rss_start: u64 = 0;
                let mut phase_rss_peak: u64 = 0;
                let mut phase_cpu_start: u64 = 0;

                loop {
                    let rss = read_rss_bytes();
                    let cpu = read_cpu_ms();
                    let elapsed = start.elapsed();

                    let (phase, should_stop) = {
                        let s = shared2.lock().unwrap();
                        (s.current_phase.clone(), s.stop)
                    };

                    // Check RSS limit.
                    if let Some(limit) = rss_limit_bytes {
                        if rss > limit {
                            // Print whatever phase we're in before dying.
                            if !prev_phase.is_empty() {
                                print_phase_line(
                                    &prev_phase,
                                    elapsed.saturating_sub(phase_start_elapsed),
                                    phase_rss_start,
                                    phase_rss_peak.max(rss),
                                    rss,
                                    phase_cpu_start,
                                    cpu,
                                );
                            }
                            eprintln!(
                                "\n  KILLED: RSS {:.0}MB exceeds limit {:.0}MB (phase: {})",
                                rss as f64 / (1024.0 * 1024.0),
                                limit as f64 / (1024.0 * 1024.0),
                                phase,
                            );
                            std::process::exit(1);
                        }
                    }

                    // Phase transition — print summary of completed phase.
                    if phase != prev_phase {
                        if !prev_phase.is_empty() {
                            print_phase_line(
                                &prev_phase,
                                elapsed.saturating_sub(phase_start_elapsed),
                                phase_rss_start,
                                phase_rss_peak,
                                rss,
                                phase_cpu_start,
                                cpu,
                            );
                        }
                        prev_phase = phase;
                        phase_start_elapsed = elapsed;
                        phase_rss_start = rss;
                        phase_rss_peak = rss;
                        phase_cpu_start = cpu;
                    } else {
                        phase_rss_peak = phase_rss_peak.max(rss);
                    }

                    if should_stop {
                        // Print final phase.
                        if !prev_phase.is_empty() {
                            print_phase_line(
                                &prev_phase,
                                elapsed.saturating_sub(phase_start_elapsed),
                                phase_rss_start,
                                phase_rss_peak,
                                rss,
                                phase_cpu_start,
                                cpu,
                            );
                        }
                        let total_cpu = cpu;
                        eprintln!(
                            "  {:<20} {:>6.1}s {:>9} {:>8.0}MB",
                            "TOTAL",
                            elapsed.as_secs_f64(),
                            "",
                            phase_rss_peak as f64 / (1024.0 * 1024.0),
                        );
                        break;
                    }
                    thread::sleep(Duration::from_millis(250));
                }
            });
            Profiler {
                shared,
                thread: Some(thread),
            }
        }

        pub fn phase(&self, name: &str) {
            self.shared.lock().unwrap().current_phase = name.to_string();
        }

        pub fn finish(mut self) {
            self.shared.lock().unwrap().stop = true;
            if let Some(thread) = self.thread.take() {
                let _ = thread.join();
            }
        }
    }

    impl Drop for Profiler {
        fn drop(&mut self) {
            if let Some(thread) = self.thread.take() {
                self.shared.lock().unwrap().stop = true;
                let _ = thread.join();
            }
        }
    }

    fn read_rss_bytes() -> u64 {
        std::fs::read_to_string("/proc/self/statm")
            .ok()
            .and_then(|s| s.split_whitespace().nth(1)?.parse::<u64>().ok())
            .unwrap_or(0)
            * 4096
    }

    fn read_cpu_ms() -> u64 {
        let buf = match std::fs::read_to_string("/proc/self/stat") {
            Ok(s) => s,
            Err(_) => return 0,
        };
        let after_comm = match buf.find(')') {
            Some(i) => &buf[i + 2..],
            None => return 0,
        };
        let fields: Vec<&str> = after_comm.split_whitespace().collect();
        let utime: u64 = fields.get(11).and_then(|s| s.parse().ok()).unwrap_or(0);
        let stime: u64 = fields.get(12).and_then(|s| s.parse().ok()).unwrap_or(0);
        (utime + stime) * 10
    }

    fn print_phase_line(
        name: &str,
        wall: Duration,
        rss_start: u64,
        rss_peak: u64,
        rss_end: u64,
        cpu_start: u64,
        cpu_end: u64,
    ) {
        let mb = |b: u64| b as f64 / (1024.0 * 1024.0);
        let delta = rss_end as i64 - rss_start as i64;
        let delta_str = format!(
            "{}{:.0}",
            if delta >= 0 { "+" } else { "-" },
            (delta.unsigned_abs()) as f64 / (1024.0 * 1024.0)
        );
        let cpu = cpu_end.saturating_sub(cpu_start);
        eprintln!(
            "  {:<20} {:>6.1}s {:>8.0}MB {:>8.0}MB {:>8.0}MB {:>8}MB {:>5.1}s",
            name,
            wall.as_secs_f64(),
            mb(rss_start),
            mb(rss_peak),
            mb(rss_end),
            delta_str,
            cpu as f64 / 1e3,
        );
    }
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

    let rss_limit = std::env::var("RSS_LIMIT_MB")
        .ok()
        .and_then(|s| s.parse::<u64>().ok());
    let profiler = profiler::Profiler::start(rss_limit);

    // ── Step 1: Load ONNX model ──────────────────────────────────────────────

    profiler.phase("load_model");
    let t0 = Instant::now();
    let onnx_data = identify_and_load(path, WeightStorageStrategy::EmbeddedData).unwrap();
    let mut rng = rand::rng();
    let model = Model::new_from_onnx(&onnx_data, &mut rng, None).unwrap();
    eprintln!("Model loaded in {:.1}s", t0.elapsed().as_secs_f64());

    // ── Step 2: Generate MilliOpGraph ────────────────────────────────────────

    profiler.phase("gen_milli");
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

    profiler.phase("build_tensors");
    let t0 = Instant::now();
    let input_info = model.get_input_tensor_info().unwrap();
    let tensors_by_name = sym_graph.get_tensors_by_name();

    let mut all_infos: HashMap<GlobalId, TensorInfo<'_, whisper_tensor::pool::SystemPool>> = HashMap::new();

    // User inputs: build concrete tensors and insert as full-data TensorInfo.
    // The lowering's infer_all needs data for user inputs so that downstream
    // ops (Gather/embedding) can propagate shapes correctly.
    for (name, (dtype, shape_dims)) in &input_info {
        let shape: Vec<u64> = shape_dims.iter().map(|d| d.unwrap_or(4)).collect();
        let num_elements: u64 = shape.iter().product();
        let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        println!("Input '{}': {:?} {:?}", name, dtype, shape);
        let tensor: whisper_tensor::migration::numeric_tensor::NumericTensor<whisper_tensor::DynRank> =
            match dtype {
                DType::I64 => {
                    let data: Vec<i64> = (0..num_elements).map(|i| (i % 64) as i64).collect();
                    whisper_tensor::migration::numeric_tensor::NumericTensor::from_vec_shape(data, shape_usize)
                        .unwrap()
                }
                DType::F32 => {
                    let data: Vec<f32> =
                        (0..num_elements).map(|i| (i % 64) as f32 * 0.01).collect();
                    whisper_tensor::migration::numeric_tensor::NumericTensor::from_vec_shape(data, shape_usize)
                        .unwrap()
                }
                _ => {
                    let data: Vec<f32> = vec![0.0; num_elements as usize];
                    whisper_tensor::migration::numeric_tensor::NumericTensor::from_vec_shape(data, shape_usize)
                        .unwrap()
                }
            };
        if let Some(id) = tensors_by_name.get(name) {
            all_infos.insert(*id, TensorInfo::from(tensor));
        }
    }

    // All initialized tensors: pass full data so lowering can inline
    // constants and resolve Shape/Gather/Reshape ops correctly.
    // When LOWER_SHAPE_ONLY=1 is set, use shape+dtype only for large weights
    // to test the shape-only lowering path.
    let shape_only = std::env::var("LOWER_SHAPE_ONLY").is_ok();
    let initialized = sym_graph.get_initialized_tensors(tensor_store);
    for (id, tensor) in &initialized {
        if shape_only && tensor.num_elements() > 1024 {
            let shape: Vec<u64> = tensor.shape().to_vec();
            let dtype = whisper_tensor::numeric_dtype::NumericDType::from_legacy(tensor.dtype())
                .expect("unsupported weight dtype");
            all_infos.insert(*id, TensorInfo::from_dtype_and_shape(dtype, &shape));
        } else {
            all_infos.insert(*id, TensorInfo::from(tensor.clone()));
        }
    }
    eprintln!(
        "TensorInfo built in {:.1}s ({} tensors)",
        t0.elapsed().as_secs_f64(),
        all_infos.len()
    );

    // ── Step 4: Lower to NanoGraph ───────────────────────────────────────────

    profiler.phase("lower");
    let t0 = Instant::now();
    let result = lower::lower(&milli_graph, &all_infos).unwrap();
    drop(all_infos); // Only needed for lowering — free early.
    eprintln!("Lowered in {:.1}ms", t0.elapsed().as_secs_f64() * 1e3);

    let stats = result.graph.stats();
    println!("\n=== NanoGraph ===\n{}", stats);

    // Dtype census: what output_dtype and compute_dtype values actually appear?
    {
        let mut output_dtypes: HashMap<String, usize> = HashMap::new();
        let mut compute_dtypes: HashMap<String, usize> = HashMap::new();
        let mut input_dtypes: HashMap<String, usize> = HashMap::new();
        for g in result.graph.groups() {
            *output_dtypes
                .entry(format!("{:?}", g.output_dtype))
                .or_default() += 1;
            if let Some(cd) = g.op.compute_dtype() {
                *compute_dtypes.entry(format!("{:?}", cd)).or_default() += 1;
            }
        }
        for it in result.graph.input_tensors() {
            *input_dtypes.entry(format!("{:?}", it.dtype)).or_default() += 1;
        }
        let mut od: Vec<_> = output_dtypes.into_iter().collect();
        od.sort_by(|a, b| b.1.cmp(&a.1));
        let mut cd: Vec<_> = compute_dtypes.into_iter().collect();
        cd.sort_by(|a, b| b.1.cmp(&a.1));
        let mut id: Vec<_> = input_dtypes.into_iter().collect();
        id.sort_by(|a, b| b.1.cmp(&a.1));
        println!("\n=== DType Census ===");
        println!("  Output dtypes: {:?}", od);
        println!("  Compute dtypes: {:?}", cd);
        println!("  Input tensor dtypes: {:?}", id);
        // Detail: what ops have I64 compute?
        let mut i64_ops: HashMap<String, usize> = HashMap::new();
        for g in result.graph.groups() {
            if g.op.compute_dtype() == Some(NumericDType::I64) {
                let name = format!("{:?}", g.op).chars().take(40).collect::<String>();
                *i64_ops.entry(name).or_default() += 1;
            }
        }
        if !i64_ops.is_empty() {
            let mut v: Vec<_> = i64_ops.into_iter().collect();
            v.sort_by(|a, b| b.1.cmp(&a.1));
            println!("  I64 compute ops: {:?}", v);
        }
    }

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
    profiler.phase("plan_setup");

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

    // ── Partitioner tests ──────────────────────────────────────────────────

    {
        use whisper_tensor::compiler::attempts::v14::{
            partitioner_i, partitioner_j, partitioner_l, partitioner_m,
        };
        use whisper_tensor::nano_graph::AtomId as AId;

        let input_ts = result.graph.input_tensors();
        let output_ids: Vec<AId> = model_outputs.iter().map(|om| om.range.base).collect();
        let num_lanes = 8;

        // Structural partitioner tests. Set RUN_STRUCTURAL=1 to enable.
        let partitioners: Vec<(
            &str,
            fn(
                &NanoGraph,
                usize,
                &[whisper_tensor::nano_graph::pattern::InputTensor],
                &[AId],
            ) -> Vec<Phase>,
        )> = if std::env::var("RUN_STRUCTURAL").is_ok() {
            vec![
                ("I (top-down tiling)", partitioner_i::plan),
                ("J (critical path)", partitioner_j::plan),
                ("L (open-ended 1)", partitioner_l::plan),
                ("M (open-ended 2)", partitioner_m::plan),
            ]
        } else {
            vec![]
        };

        // Filter by PARTITIONER env var if set (e.g. PARTITIONER=B)
        let filter = std::env::var("PARTITIONER").ok();

        println!("\n=== Partitioner Tests ({} lanes) ===", num_lanes);
        for (name, plan_fn) in &partitioners {
            if let Some(ref f) = filter {
                if !name.starts_with(&format!("{} ", f)) && !name.contains(f.as_str()) {
                    continue;
                }
            }

            let t0 = Instant::now();
            let result_phases = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                plan_fn(&result.graph, num_lanes, input_ts, &output_ids)
            }));
            let elapsed = t0.elapsed();

            match result_phases {
                Ok(phases) => {
                    let num_phases = phases.len();
                    let total_spans: usize = phases.iter().map(|p| p.spans.len()).sum();
                    let total_groups: usize = phases
                        .iter()
                        .flat_map(|p| p.spans.iter())
                        .map(|s| s.graph.num_groups())
                        .sum();
                    let total_atoms: u64 = phases
                        .iter()
                        .flat_map(|p| p.spans.iter())
                        .map(|s| s.graph.num_atoms())
                        .sum();
                    let total_inputs: usize = phases
                        .iter()
                        .flat_map(|p| p.spans.iter())
                        .map(|s| s.inputs.len())
                        .sum();
                    let total_outputs: usize = phases
                        .iter()
                        .flat_map(|p| p.spans.iter())
                        .map(|s| s.outputs.len())
                        .sum();

                    // Check lane counts
                    let lane_ok = phases.iter().all(|p| p.spans.len() == num_lanes);

                    // Check span NanoGraph validation
                    let mut validation_errors = 0usize;
                    for (pi, phase) in phases.iter().enumerate() {
                        for (li, span) in phase.spans.iter().enumerate() {
                            let errs = span.graph.validate();
                            if !errs.is_empty() {
                                if validation_errors < 5 {
                                    eprintln!(
                                        "  {} phase {} lane {}: {} validation errors",
                                        name,
                                        pi,
                                        li,
                                        errs.len()
                                    );
                                    for e in errs.iter().take(3) {
                                        eprintln!("    {}", e);
                                    }
                                }
                                validation_errors += errs.len();
                            }
                        }
                    }

                    // Check cross-span independence within each phase (group-level)
                    let mut cross_lane_violations = 0usize;
                    for phase in &phases {
                        // Collect what each span produces as (base, end) ranges
                        let produced_by_span: Vec<Vec<(u64, u64)>> = phase
                            .spans
                            .iter()
                            .map(|span| {
                                span.graph
                                    .groups()
                                    .iter()
                                    .map(|g| (g.base_id.0, g.base_id.0 + g.count))
                                    .collect()
                            })
                            .collect();
                        // Check that no span's inputs overlap another span's produced ranges
                        for (si, span) in phase.spans.iter().enumerate() {
                            for inp in &span.inputs {
                                let inp_lo = inp.base.0;
                                let inp_hi = inp.base.0 + inp.count;
                                for (oi, other_ranges) in produced_by_span.iter().enumerate() {
                                    if oi != si {
                                        for &(lo, hi) in other_ranges {
                                            if inp_lo < hi && inp_hi > lo {
                                                cross_lane_violations += 1;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Work balance per phase
                    let mut max_imbalance = 0.0f64;
                    for phase in &phases {
                        let loads: Vec<u64> = phase
                            .spans
                            .iter()
                            .map(|s| s.graph.groups().iter().map(|g| g.count).sum::<u64>())
                            .collect();
                        let max_load = *loads.iter().max().unwrap_or(&0) as f64;
                        let min_load = *loads.iter().filter(|&&l| l > 0).min().unwrap_or(&1) as f64;
                        if min_load > 0.0 {
                            max_imbalance = max_imbalance.max(max_load / min_load);
                        }
                    }

                    println!("\n  {} — {:.1?}", name, elapsed,);
                    println!(
                        "    {} phases, {} total span groups, {} total span atoms",
                        num_phases, total_groups, total_atoms,
                    );
                    println!(
                        "    {} input ranges, {} output ranges",
                        total_inputs, total_outputs,
                    );
                    println!(
                        "    lanes_ok={}, validation_errors={}, cross_lane_violations={}, max_imbalance={:.1}x",
                        lane_ok, validation_errors, cross_lane_violations, max_imbalance,
                    );
                }
                Err(e) => {
                    let msg = if let Some(s) = e.downcast_ref::<String>() {
                        s.clone()
                    } else if let Some(s) = e.downcast_ref::<&str>() {
                        s.to_string()
                    } else {
                        "unknown panic".to_string()
                    };
                    println!("\n  {} — PANICKED in {:.1?}: {}", name, elapsed, msg);
                }
            }
        }

        // Early exit when running partitioner tests only
        if filter.is_some() {
            std::process::exit(0);
        }
    }

    // ── Build execution plan from selected partitioner ─────────────────
    profiler.phase("partition");

    use whisper_tensor::compiler::attempts::v14::partitioner_m;
    use whisper_tensor::nano_graph::AtomId;

    // Collect output atom IDs — one per group that contributes to any model
    // output. Walk all elements of each output tensor to find every unique
    // group, since a tensor may span many groups (e.g. the LM-head logits)
    // and segmented tensors scatter atoms across non-contiguous ranges.
    let b_output_ids: Vec<AtomId> = {
        let reverse_out: HashMap<GlobalId, GlobalId> = milli_graph
            .output_map
            .as_ref()
            .map(|m| m.iter().map(|(&int, &ext)| (ext, int)).collect())
            .unwrap_or_default();
        let mut ids = Vec::new();
        let mut seen_groups = std::collections::HashSet::new();
        for om in &model_outputs {
            let int_id = reverse_out
                .get(&om.tensor_id)
                .copied()
                .unwrap_or(om.tensor_id);
            if let Some(tam) = result.tensor_map.get(&int_id) {
                for i in 0..tam.count {
                    let atom = tam.atom_id_for_element(i);
                    if let Some(gi) = result.graph.find_group_idx(atom) {
                        if seen_groups.insert(gi) {
                            ids.push(atom);
                        }
                    }
                }
            }
        }
        ids
    };
    let t0 = Instant::now();
    let b_phases = partitioner_m::plan(
        &result.graph,
        8,
        result.graph.input_tensors(),
        &b_output_ids,
    );
    let part_name = "M";
    eprintln!(
        "Partitioner {}: {:.1?}, {} phases",
        part_name,
        t0.elapsed(),
        b_phases.len()
    );

    let b_exec_plan = ExecutionPlan {
        graph: result.graph.clone(),
        tensor_map: tensor_map.clone(),
        phases: b_phases,
        model_outputs: model_outputs.clone(),
    };

    // Quick stats on B's plan.
    {
        let total_output_atoms: u64 = b_exec_plan
            .phases
            .iter()
            .flat_map(|p| p.spans.iter())
            .flat_map(|s| s.outputs.iter())
            .map(|o| o.count)
            .sum();
        let total_compute_atoms: u64 = b_exec_plan
            .phases
            .iter()
            .flat_map(|p| p.spans.iter())
            .flat_map(|s| s.graph.groups().iter())
            .map(|g| g.count)
            .sum();
        let total_output_ranges: usize = b_exec_plan
            .phases
            .iter()
            .flat_map(|p| p.spans.iter())
            .map(|s| s.outputs.len())
            .sum();
        println!("\n=== B Plan Stats ===");
        println!("  Total compute atoms in spans: {}", total_compute_atoms);
        println!(
            "  Total output atoms declared: {} ({} ranges)",
            total_output_atoms, total_output_ranges
        );
        println!(
            "  Output ratio: {:.1}%",
            total_output_atoms as f64 / total_compute_atoms as f64 * 100.0
        );
    }

    // Validate B's spans are faithful subgraphs of the main graph.
    println!("\n=== Span Subgraph Validation ===");
    let t0 = Instant::now();
    let span_errors = execute::validate_spans(&b_exec_plan);
    eprintln!(
        "Span validation: {:.1?}, {} errors",
        t0.elapsed(),
        span_errors.len()
    );
    if !span_errors.is_empty() {
        for e in span_errors.iter().take(20) {
            eprintln!("  {}", e);
        }
        if span_errors.len() > 20 {
            eprintln!("  ... and {} more", span_errors.len() - 20);
        }
    }

    // ── Generate execution plan reports ─────────────────────────────────
    profiler.phase("reports");
    {
        use whisper_tensor::compiler::attempts::v14::report;

        let text = report::text_report(&b_exec_plan, &result.group_provenance, &result.graph);
        let report_path = "v14_report.txt";
        std::fs::write(report_path, &text).unwrap();
        eprintln!("Text report written to {}", report_path);

        let svg = report::svg_report(&b_exec_plan, &result.group_provenance, &result.graph);
        let svg_path = "v14_report.svg";
        std::fs::write(svg_path, &svg).unwrap();
        eprintln!("SVG report written to {}", svg_path);
    }

    // ── Single-span correctness test (skip unless SINGLE_SPAN=1) ─────────
    if std::env::var("SINGLE_SPAN").is_ok() {
        // TODO: migrate to pool_eval (nano_graph::eval was deleted).
        eprintln!("SINGLE_SPAN check not yet migrated to pool_eval");
    } // end SINGLE_SPAN

    // ── Phase-by-phase correctness check ─────────────────────────────────
    // Run B's executor phase by phase, after each phase compare a sample
    // of store values against the full-graph direct eval.
    if std::env::var("CHECK_PHASES").is_ok() {
        // TODO: migrate to pool_eval (nano_graph::eval was deleted).
        eprintln!("CHECK_PHASES check not yet migrated to pool_eval");
    }

    // Save tensor_map and graph for output comparison before trivial plan consumes result.
    let lower_tensor_map_for_compare = result.tensor_map.clone();
    let graph_for_compare = result.graph.clone();

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
    profiler.phase("build_inputs");

    use whisper_tensor::backends::eval_backend::EvalBackend;
    use whisper_tensor::backends::ndarray_backend::numeric_tensor::NDArrayNumericTensor;
    use whisper_tensor::compiler::attempts::v14::execute;
    use whisper_tensor::migration::numeric_tensor::NumericTensor;

    // Build user input tensors with valid token IDs.
    // GPT-2 vocabulary: common tokens like "Hello" = 15496, "," = 11, " world" = 995.
    let mut user_inputs: HashMap<String, NumericTensor<whisper_tensor::DynRank>> = HashMap::new();
    for (name, (dtype, shape_dims)) in &input_info {
        let shape: Vec<u64> = shape_dims.iter().map(|d| d.unwrap_or(4)).collect();
        let num_elements: u64 = shape.iter().product();
        let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        // Use same values as lowering (i % 64) so nano and milli see the same data.
        let tensor = match dtype {
            DType::I64 => {
                let data: Vec<i64> = (0..num_elements as usize)
                    .map(|i| (i % 64) as i64)
                    .collect();
                NumericTensor::from_vec_shape(data, shape_usize).unwrap()
            }
            DType::F32 => {
                let data: Vec<f32> = (0..num_elements as usize)
                    .map(|i| (i % 64) as f32 * 0.01)
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
    // Consume initialized directly — no clone needed, we're done with it.
    let mut milli_inputs: HashMap<GlobalId, NumericTensor<whisper_tensor::DynRank>> = initialized;
    for (name, tensor) in &user_inputs {
        if let Some(&ext_id) = tensors_by_name.get(name.as_str()) {
            milli_inputs.insert(ext_id, tensor.clone());
        }
    }

    // ── Step 9: Run MilliOpGraph reference eval ─────────────────────────────
    profiler.phase("milli_eval");

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

    // ── Step 10: Build shared nano inputs (AtomId-keyed) ────────────────────
    //
    // Built once from milli_inputs, then reused for direct eval, interpreter B,
    // and JIT. Avoids rebuilding from scratch per eval path.
    profiler.phase("nano_inputs");

    let nano_inputs: Vec<(AtomId, NDArrayNumericTensor<whisper_tensor::DynRank>)> = {
        let mut inputs = Vec::new();
        let mut matched = 0usize;
        let mut unmatched = 0usize;
        let mut unmatched_atoms = 0u64;

        for it in b_exec_plan.graph.input_tensors() {
            let milli_id = it.tensor_id;
            let ext_id = milli_graph
                .input_map
                .iter()
                .find(|(_, int)| **int == milli_id)
                .map(|(ext, _)| *ext);

            let tensor = ext_id.and_then(|ext| milli_inputs.remove(&ext));

            if tensor.is_none() {
                unmatched += 1;
                unmatched_atoms += it.count;
                if unmatched <= 5 {
                    eprintln!(
                        "  UNMATCHED input_tensor: milli_id={:?} base={} count={} {:?} ext_id={:?}",
                        milli_id, it.base_id.0, it.count, it.dtype, ext_id
                    );
                }
            } else {
                matched += 1;
            }

            if let Some(t) = tensor {
                let nd = match t {
                    NumericTensor::NDArray(nd) => nd,
                    _ => t.to_ndarray().unwrap(),
                };
                inputs.push((it.base_id, nd));
            }
        }

        println!(
            "\n=== NanoGraph Inputs ({} input tensors, {} matched, {} unmatched ({} atoms)) ===",
            b_exec_plan.graph.input_tensors().len(),
            matched,
            unmatched,
            unmatched_atoms
        );
        inputs
    };
    drop(milli_inputs); // Free remaining weight data not needed for nano eval.

    // Build output atom ranges — handle segmented (Concat) tensors
    // by collecting all underlying atom ranges.
    let reverse_output_map: HashMap<GlobalId, GlobalId> = milli_graph
        .output_map
        .as_ref()
        .map(|m| m.iter().map(|(&int, &ext)| (ext, int)).collect())
        .unwrap_or_default();

    let mut output_ranges: Vec<whisper_tensor::nano_graph::AtomRange> = Vec::new();
    let mut output_range_mapping: Vec<(GlobalId, usize, usize)> = Vec::new(); // (ext_id, start_idx, end_idx)

    for om in &exec_plan.model_outputs {
        let int_id = reverse_output_map
            .get(&om.tensor_id)
            .copied()
            .unwrap_or(om.tensor_id);
        let tam = &lower_tensor_map_for_compare[&int_id];
        let start = output_ranges.len();

        if tam.segments.is_empty() {
            // Simple tensor: single contiguous range.
            output_ranges.push(whisper_tensor::nano_graph::AtomRange {
                base: whisper_tensor::nano_graph::AtomId(tam.base_id.0),
                count: tam.count,
                dtype: tam.dtype,
            });
        } else {
            // Segmented tensor (Concat): collect each segment's underlying ranges.
            let mut seen = std::collections::HashSet::new();
            for i in 0..tam.count {
                let atom = tam.atom_id_for_element(i);
                if let Some(gi) = graph_for_compare.find_group_idx(atom) {
                    if seen.insert(("g", gi)) {
                        let g = &graph_for_compare.groups()[gi];
                        output_ranges.push(whisper_tensor::nano_graph::AtomRange {
                            base: g.base_id,
                            count: g.count,
                            dtype: g.output_dtype,
                        });
                    }
                } else if let Some((ti, _)) = graph_for_compare.find_input_idx(atom) {
                    if seen.insert(("i", ti)) {
                        let it = &graph_for_compare.input_tensors()[ti];
                        output_ranges.push(whisper_tensor::nano_graph::AtomRange {
                            base: it.base_id,
                            count: it.count,
                            dtype: it.dtype,
                        });
                    }
                }
            }
        }
        let end = output_ranges.len();
        output_range_mapping.push((om.tensor_id, start, end));
    }

    // Direct nano eval — slow, opt-in via RUN_DIRECT=1.
    if std::env::var("RUN_DIRECT").is_ok() {
        // TODO: migrate to pool_eval (nano_graph::eval was deleted).
        let _ = (&output_ranges, &nano_inputs, &output_range_mapping);
        let _ = (&reverse_output_map, &lower_tensor_map_for_compare, &milli_outputs);
        eprintln!("RUN_DIRECT check not yet migrated to pool_eval");
    } // end RUN_DIRECT

    /// Look up a single atom's value from a sorted store index.
    fn lookup_atom(
        store_index: &[(u64, &NDArrayNumericTensor<whisper_tensor::DynRank>)],
        atom: u64,
    ) -> Option<f64> {
        let idx = store_index.partition_point(|&(base, _)| base <= atom);
        if idx == 0 {
            return None;
        }
        let (base, tensor) = &store_index[idx - 1];
        let offset = atom - base;
        if offset < tensor.num_elements() as u64 {
            let flat = tensor.flatten();
            Some(flat.get(&[offset]).unwrap().to_f64())
        } else {
            None
        }
    }

    // ── Step 12: Run partitioner B through executor ─────────────────────────

    // Interpreter eval — slow, no eviction, opt-in via RUN_INTERP=1.
    if std::env::var("RUN_INTERP").is_ok() {
        profiler.phase("interp_eval");
        println!(
            "\n=== Partitioner B Eval ({} phases, {} lanes) ===",
            b_exec_plan.phases.len(),
            b_exec_plan.phases.first().map_or(0, |p| p.spans.len())
        );

        // Reuse shared nano_inputs — ArcArray clone is O(1) per tensor.
        let b_exec_inputs: Vec<(AtomId, NDArrayNumericTensor<whisper_tensor::DynRank>)> =
            nano_inputs
                .iter()
                .map(|(base, t)| (*base, t.clone()))
                .collect();

        // Run B's executor — returns raw store (base AtomId → tensor).
        let t0 = Instant::now();
        let b_store = execute::execute(&b_exec_plan, b_exec_inputs);
        println!(
            "  Executed in {:.1}s, {} store entries",
            t0.elapsed().as_secs_f64(),
            b_store.len()
        );

        // Compare B outputs against milli reference using atom_id_for_element
        // (handles segmented Concat tensors correctly).
        println!("\n=== Partitioner B vs Milli Reference ===");

        let total_store_atoms: u64 = b_store.values().map(|t| t.num_elements() as u64).sum();
        println!(
            "  Store: {} entries, {} total atoms",
            b_store.len(),
            total_store_atoms
        );

        // Build a sorted index for O(log n) atom lookup into the store.
        // Each entry is (base_atom_id, tensor_ref). Sorted by base for binary search.
        let mut store_index: Vec<(u64, &NDArrayNumericTensor<whisper_tensor::DynRank>)> =
            b_store.iter().map(|(id, t)| (id.0, t)).collect();
        store_index.sort_by_key(|&(base, _)| base);

        let mut b_all_match = true;
        for &(ext_id, _, _) in &output_range_mapping {
            let int_id = reverse_output_map.get(&ext_id).copied().unwrap_or(ext_id);
            let tam = &lower_tensor_map_for_compare[&int_id];
            if let Some(milli_tensor) = milli_outputs.get(&ext_id) {
                let milli_nd = milli_tensor.to_ndarray().unwrap();
                let n = milli_nd.num_elements().min(tam.count as usize);
                let milli_flat = milli_nd.flatten();
                let mut max_abs_diff = 0.0f64;
                let mut mismatches = 0usize;
                let mut missing_atoms = 0usize;
                for j in 0..n {
                    let m = milli_flat.get(&[j as u64]).unwrap().to_f64();
                    let atom = tam.atom_id_for_element(j as u64);
                    let Some(bv) = lookup_atom(&store_index, atom.0) else {
                        missing_atoms += 1;
                        continue;
                    };
                    if m.is_nan() || bv.is_nan() {
                        continue;
                    }
                    let abs_diff = (m - bv).abs();
                    max_abs_diff = max_abs_diff.max(abs_diff);
                    let denom = m.abs().max(1e-10);
                    if abs_diff > 1e-3 && abs_diff / denom > 1e-3 {
                        mismatches += 1;
                    }
                }
                let status = if mismatches == 0 && missing_atoms == 0 {
                    "MATCH"
                } else {
                    "MISMATCH"
                };
                let seg_info = if tam.segments.is_empty() {
                    ""
                } else {
                    " [segmented]"
                };
                println!(
                    "  {:?}: {} ({} elements, max_abs={:.6}, mismatches={}, missing_atoms={}){}",
                    ext_id, status, n, max_abs_diff, mismatches, missing_atoms, seg_info
                );
                if mismatches > 0 || missing_atoms > 0 {
                    b_all_match = false;
                }
            } else {
                println!("  {:?}: MISSING from milli outputs", ext_id);
                b_all_match = false;
            }
        }

        if b_all_match {
            println!("\nPartitioner B: All outputs MATCH!");
        } else {
            println!("\nPartitioner B: Some outputs MISMATCHED.");
        }
    } // end RUN_INTERP

    // ── Step 13: Compiled execution via new executor ────────────────────────

    #[cfg(feature = "cranelift")]
    {
        use whisper_tensor::compiler::attempts::v14::codegen::JitCompiledSpan;
        use whisper_tensor::compiler::attempts::v14::executor::{
            ExecutablePlanBuilder, TypedBuffer,
        };

        profiler.phase("jit_compile");
        println!("\n=== JIT Compilation (new executor) ===");
        let t0 = Instant::now();

        // Compile all spans sequentially. Cranelift's JITModule has
        // thread-safety issues that prevent parallel compilation.
        let mut plan_builder = ExecutablePlanBuilder::new();
        let mut compile_errors = Vec::new();

        for (pi, phase) in b_exec_plan.phases.iter().enumerate() {
            let mut lanes = Vec::new();
            for (si, span) in phase.spans.iter().enumerate() {
                match JitCompiledSpan::compile(&span.graph, &span.outputs) {
                    Ok(jit_span) => {
                        lanes.push((
                            Box::new(jit_span)
                                as Box<dyn whisper_tensor::compiler::attempts::v14::executor::CompiledSpanFn>,
                            span.inputs.clone(),
                            span.outputs.clone(),
                        ));
                    }
                    Err(e) => {
                        compile_errors.push(format!("phase {} span {}: {}", pi, si, e));
                        let noop = JitCompiledSpan::compile(
                            &whisper_tensor::nano_graph::NanoGraph::new(),
                            &[],
                        )
                        .unwrap();
                        lanes.push((
                            Box::new(noop)
                                as Box<dyn whisper_tensor::compiler::attempts::v14::executor::CompiledSpanFn>,
                            vec![],
                            vec![],
                        ));
                    }
                }
            }
            plan_builder.add_phase(lanes);
        }

        let exec_plan = plan_builder.build();
        println!(
            "  Compiled {} phases in {:.3}s ({} errors)",
            exec_plan.num_phases(),
            t0.elapsed().as_secs_f64(),
            compile_errors.len(),
        );
        if !compile_errors.is_empty() {
            for e in compile_errors.iter().take(5) {
                eprintln!("  COMPILE ERROR: {}", e);
            }
            if compile_errors.len() > 5 {
                eprintln!("  ... and {} more", compile_errors.len() - 5);
            }
        }

        // Convert shared nano_inputs to TypedBuffers for JIT, then drop
        // the NDArray data so we don't hold both formats during execution.
        let input_dtypes: HashMap<AtomId, NumericDType> = b_exec_plan
            .graph
            .input_tensors()
            .iter()
            .map(|it| (it.base_id, it.dtype))
            .collect();
        let jit_inputs: Vec<(AtomId, TypedBuffer)> = nano_inputs
            .iter()
            .map(|(base, nd)| {
                let dtype = input_dtypes.get(base).copied().unwrap_or(NumericDType::F32);
                (*base, ndarray_to_typed_buffer(nd, dtype))
            })
            .collect();
        drop(nano_inputs); // Free NDArray refs before JIT execution.

        profiler.phase("jit_execute");
        let t0 = Instant::now();
        let jit_store = exec_plan.execute_timed(jit_inputs);
        println!(
            "  JIT executed in {:.3}s, {} store entries",
            t0.elapsed().as_secs_f64(),
            jit_store.len()
        );

        // Compare JIT outputs against milli reference.
        profiler.phase("jit_compare");
        println!("\n=== JIT vs Milli Reference ===");

        let mut jit_all_match = true;
        for &(ext_id, _, _) in &output_range_mapping {
            let int_id = reverse_output_map.get(&ext_id).copied().unwrap_or(ext_id);
            let tam = &lower_tensor_map_for_compare[&int_id];
            if let Some(milli_tensor) = milli_outputs.get(&ext_id) {
                let milli_nd = milli_tensor.to_ndarray().unwrap();
                let n = milli_nd.num_elements().min(tam.count as usize);
                let milli_flat = milli_nd.flatten();
                let mut max_abs_diff = 0.0f64;
                let mut mismatches = 0usize;
                let mut missing_atoms = 0usize;
                for j in 0..n {
                    let m = milli_flat.get(&[j as u64]).unwrap().to_f64();
                    let atom = tam.atom_id_for_element(j as u64);
                    // Look up in PhaseStore.
                    let jv = lookup_atom_in_store(&jit_store, atom.0);
                    let Some(jv) = jv else {
                        missing_atoms += 1;
                        continue;
                    };
                    if m.is_nan() || jv.is_nan() {
                        continue;
                    }
                    let abs_diff = (m - jv).abs();
                    max_abs_diff = max_abs_diff.max(abs_diff);
                    let denom = m.abs().max(1e-10);
                    if abs_diff > 1e-3 && abs_diff / denom > 1e-3 {
                        mismatches += 1;
                    }
                }
                let status = if mismatches == 0 && missing_atoms == 0 {
                    "MATCH"
                } else {
                    "MISMATCH"
                };
                let seg_info = if tam.segments.is_empty() {
                    ""
                } else {
                    " [segmented]"
                };
                println!(
                    "  {:?}: {} ({} elements, max_abs={:.6}, mismatches={}, missing_atoms={}){}",
                    ext_id, status, n, max_abs_diff, mismatches, missing_atoms, seg_info
                );
                if mismatches > 0 || missing_atoms > 0 {
                    jit_all_match = false;
                }
            } else {
                println!("  {:?}: MISSING from milli outputs", ext_id);
                jit_all_match = false;
            }
        }

        if jit_all_match {
            println!("\nJIT: All outputs MATCH!");
        } else {
            println!("\nJIT: Some outputs MISMATCHED.");
        }
    }

    profiler.finish();
}

/// Build the ExecutionPlan's tensor_map from lowering results.
///
/// Classifies each tensor as Weight, Input, or Computed by cross-referencing
/// the milli_graph's input_map with the user-provided input_info.
type NdTensor = whisper_tensor::backends::ndarray_backend::numeric_tensor::NDArrayNumericTensor<
    whisper_tensor::DynRank,
>;

/// Convert an NDArray tensor to a TypedBuffer (raw bytes).
fn ndarray_to_typed_buffer(
    nd: &NdTensor,
    dtype: NumericDType,
) -> whisper_tensor::compiler::attempts::v14::executor::TypedBuffer {
    use whisper_tensor::compiler::attempts::v14::executor::TypedBuffer;
    macro_rules! to_bytes {
        ($arr:expr) => {{
            let slice = $arr.as_slice().expect("non-contiguous ndarray");
            let byte_len = slice.len() * std::mem::size_of_val(&slice[0]);
            let bytes: Vec<u8> =
                unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) }
                    .to_vec();
            TypedBuffer {
                data: bytes,
                dtype,
                count: slice.len() as u64,
            }
        }};
    }
    match nd {
        NdTensor::F32(a) => to_bytes!(a),
        NdTensor::F64(a) => to_bytes!(a),
        NdTensor::I64(a) => to_bytes!(a),
        NdTensor::I32(a) => to_bytes!(a),
        NdTensor::BF16(a) => to_bytes!(a),
        NdTensor::F16(a) => to_bytes!(a),
        NdTensor::U8(a) => to_bytes!(a),
        NdTensor::I8(a) => to_bytes!(a),
        NdTensor::BOOL(a) => {
            let data: Vec<u8> = a.iter().map(|&b| if b { 1 } else { 0 }).collect();
            TypedBuffer {
                data,
                dtype: NumericDType::BOOL,
                count: a.len() as u64,
            }
        }
        _ => TypedBuffer {
            data: vec![],
            dtype,
            count: 0,
        },
    }
}

/// Look up a single atom's f64 value in a PhaseStore.
fn lookup_atom_in_store(
    store: &whisper_tensor::compiler::attempts::v14::executor::PhaseStore,
    atom: u64,
) -> Option<f64> {
    use whisper_tensor::nano_graph::AtomId;
    let slices = store.gather(AtomId(atom), 1);
    for slice in &slices {
        if slice.base.0 <= atom && atom < slice.base.0 + slice.count {
            let offset = (atom - slice.base.0) as usize;
            let elem_bytes =
                whisper_tensor::compiler::attempts::v14::executor::dtype_elem_bytes(slice.dtype);
            let byte_off = offset * elem_bytes;
            if byte_off + elem_bytes <= slice.data.len() {
                return Some(match slice.dtype {
                    NumericDType::F32 => {
                        f32::from_le_bytes(slice.data[byte_off..byte_off + 4].try_into().unwrap())
                            as f64
                    }
                    NumericDType::F64 => {
                        f64::from_le_bytes(slice.data[byte_off..byte_off + 8].try_into().unwrap())
                    }
                    NumericDType::I64 => {
                        i64::from_le_bytes(slice.data[byte_off..byte_off + 8].try_into().unwrap())
                            as f64
                    }
                    NumericDType::BF16 => {
                        let bits = u16::from_le_bytes(
                            slice.data[byte_off..byte_off + 2].try_into().unwrap(),
                        );
                        half::bf16::from_bits(bits).to_f64()
                    }
                    _ => 0.0,
                });
            }
        }
    }
    None
}

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
