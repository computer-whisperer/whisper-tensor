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

    // ---- Lower (ops only, skip numeric_overrides) ----
    let t0 = Instant::now();
    // Use lower_with_info but we only care about the graph, not the overrides.
    // TODO: add a lower_graph_only that skips numeric_overrides
    let result = whisper_tensor::nano_graph::lower::lower_with_info(&milli_graph, &all_infos).unwrap();
    let lower_elapsed = t0.elapsed();
    eprintln!("lower_with_info: {:.1}s", lower_elapsed.as_secs_f64());

    // ---- NanoGraph stats ----
    let stats = result.graph.stats();
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

    let errors = result.graph.validate();
    if errors.is_empty() {
        println!("Validation: PASSED");
    } else {
        println!("Validation: {} ERRORS", errors.len());
        for e in errors.iter().take(10) { println!("  {}", e); }
    }

    // ---- Explicit InputRef diagnostic ----
    print_explicit_diagnostic(&result.graph);

    // ---- InputRef type distribution ----
    print_inputref_distribution(&result.graph);

    // ---- ReduceSum diagnostic ----
    print_reduce_diagnostic(&result.graph);

    // ---- Partition ----
    println!("\n=== Partitioner ===");
    use whisper_tensor::compiler::attempts::v13_claude::nano_part_b::partition_nanograph;
    let t0 = Instant::now();
    let partition = partition_nanograph(&result.graph, 8);
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
                    | InputRef::SymAffine { base, .. } => vec![*base],
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

fn print_explicit_diagnostic(graph: &NanoGraph) {
    let groups = graph.groups();
    let mut explicit_groups = 0u64;
    let mut total_entries = 0u64;
    let mut by_op: HashMap<String, (u64, u64)> = HashMap::new();
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
            let e = by_op.entry(op_name).or_default();
            e.0 += 1; e.1 += entries;
        }
    }
    println!("\n=== Explicit InputRef ===");
    println!("{} groups, {} entries ({:.1} MB)", explicit_groups, total_entries,
        total_entries as f64 * 8.0 / (1024.0 * 1024.0));
    let mut sorted: Vec<_> = by_op.into_iter().collect();
    sorted.sort_by(|a, b| b.1.1.cmp(&a.1.1));
    for (op, (g, e)) in &sorted {
        println!("  {}: {} groups, {} entries ({:.1} MB)", op, g, e, *e as f64 * 8.0 / (1024.0*1024.0));
    }
}

fn print_inputref_distribution(graph: &NanoGraph) {
    let groups = graph.groups();
    let mut broadcast = 0u64;
    let mut affine = 0u64;
    let mut strided_broadcast = 0u64;
    let mut sym_affine = 0u64;
    let mut explicit = 0u64;
    for group in groups {
        for input in &group.inputs {
            match input {
                InputRef::Broadcast(_) => broadcast += 1,
                InputRef::Affine { .. } => affine += 1,
                InputRef::StridedBroadcast { .. } => strided_broadcast += 1,
                InputRef::SymAffine { .. } => sym_affine += 1,
                InputRef::Explicit(_) => explicit += 1,
            }
        }
    }
    println!("\n=== InputRef Distribution ===");
    println!("  Broadcast: {}", broadcast);
    println!("  Affine: {}", affine);
    println!("  StridedBroadcast: {}", strided_broadcast);
    println!("  SymAffine: {}", sym_affine);
    println!("  Explicit: {}", explicit);
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
