//! Fast lowering + partitioning diagnostic for ONNX models.
//! Skips interpreter execution and numeric_overrides (the slow parts).
//!
//! Usage:
//!   cargo run --release --example nano_lower_test -- test_models/gpt2-lm-head-10.onnx

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use whisper_tensor::compiler::op_census;
use whisper_tensor::graph::GlobalId;
use whisper_tensor::model::Model;
use whisper_tensor::nano_graph::{InputRef, NanoGraph};
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
    let num_lanes = 8;
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
