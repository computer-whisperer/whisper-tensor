//! `wt dump-tensors` — run a model and dump selected intermediate tensors.
//!
//! Loads an ONNX-like model via the standard loader pipeline, feeds it the
//! given inputs (from .npy files), evaluates the full graph, and writes any
//! tensor whose ONNX name matches the requested list to the output directory
//! as .npy files.  Names that don't match any tensor in the graph are reported
//! but do not cause failure.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use whisper_tensor::graph::GlobalId;
use whisper_tensor::model::Model;
use whisper_tensor::numeric_dtype::NumericDType;
use whisper_tensor::numeric_tensor::{NumericTensorView, TensorLayout};
use whisper_tensor::pool::{Pool, SystemPool};
use whisper_tensor::symbolic_graph::SharedPoolTensor;
use whisper_tensor::symbolic_graph::observer::SymbolicGraphObserver;
use whisper_tensor::tensor_rank::DynRank;
use whisper_tensor::{npy, npy::write_npy_file};

use whisper_tensor_import::onnx_graph::WeightStorageStrategy;

// ---------------------------------------------------------------------------
// Observer
// ---------------------------------------------------------------------------

pub struct TensorDumpObserver {
    /// GlobalId → ONNX name, for the tensors we want to capture.
    watched_ids: HashMap<GlobalId, String>,
    /// Captured tensor values, keyed by ONNX name.
    pub captured: HashMap<String, SharedPoolTensor>,
    call_count: usize,
}

impl TensorDumpObserver {
    pub fn new(graph: &whisper_tensor::symbolic_graph::SymbolicGraph, names: &[String]) -> Self {
        let names_by_name = graph.get_tensors_by_name();
        let name_set: HashSet<&str> = names.iter().map(|s| s.as_str()).collect();
        let mut watched_ids = HashMap::new();
        for (name, id) in &names_by_name {
            if name_set.contains(name.as_str()) {
                watched_ids.insert(*id, name.clone());
            }
        }
        for name in names {
            if !names_by_name.contains_key(name) {
                eprintln!("Warning: tensor '{}' not found in graph", name);
            }
        }
        for (id, name) in &watched_ids {
            eprintln!("  watching {:?} → '{}'", id, name);
        }
        eprintln!("Watching {} tensors", watched_ids.len());
        Self {
            watched_ids,
            captured: HashMap::new(),
            call_count: 0,
        }
    }
}

impl SymbolicGraphObserver for TensorDumpObserver {
    fn on_op_executed(&mut self, _: &[GlobalId], _: Instant, _: Instant) {}
    fn on_tensor_assigned(
        &mut self,
        tensor_path: &[GlobalId],
        tensor: &NumericTensorView<'_, DynRank>,
    ) {
        if let Some(id) = tensor_path.last() {
            self.call_count += 1;
            if let Some(name) = self.watched_ids.get(id) {
                let n = tensor.numel().min(3);
                let preview: Vec<f64> = (0..n).map(|i| tensor.read_element(i).to_f64()).collect();
                eprintln!(
                    "  captured '{}': shape={:?} dtype={:?} first{}={:.6?}",
                    name,
                    tensor.shape(),
                    tensor.dtype(),
                    n,
                    &preview
                );
                self.captured
                    .insert(name.clone(), SharedPoolTensor::from_view(tensor));
            }
        }
    }
    fn on_loading_weight(&mut self, _: &[GlobalId], _: Option<String>) {}
}

// ---------------------------------------------------------------------------
// Command
// ---------------------------------------------------------------------------

pub fn cmd_dump_tensors(
    model_path: PathBuf,
    input_npys: Vec<(String, PathBuf)>,
    tensor_names: Vec<String>,
    output_dir: PathBuf,
) {
    let pool = SystemPool;

    // Import model
    let t0 = Instant::now();
    eprintln!("Loading model from {}...", model_path.display());
    let onnx_data = whisper_tensor_import::identify_and_load(
        &model_path,
        WeightStorageStrategy::OriginReference,
    )
    .unwrap_or_else(|e| {
        eprintln!("Failed to import model: {e}");
        std::process::exit(1);
    });

    eprintln!(
        "ONNX export: {:.1}s, {} bytes",
        t0.elapsed().as_secs_f32(),
        onnx_data.len()
    );
    use std::io::Write;
    std::io::stderr().flush().ok();
    let t1 = Instant::now();
    let mut rng = rand::rng();
    let model = Model::new_from_onnx(&onnx_data, &mut rng, Some(&model_path)).unwrap_or_else(|e| {
        eprintln!("Failed to load ONNX: {e}");
        std::process::exit(1);
    });
    eprintln!("ONNX parse: {:.1}s", t1.elapsed().as_secs_f32());
    std::io::stderr().flush().ok();

    let graph = model.get_symbolic_graph();

    // If no tensor names given, list all available named tensors and exit
    if tensor_names.is_empty() {
        let names = graph.get_tensors_by_name();
        let mut names_vec: Vec<_> = names.keys().collect();
        names_vec.sort();
        println!("Available tensors ({}):", names_vec.len());
        for name in names_vec {
            if !name.contains("kv_cache") {
                let id = names[name];
                let info = graph.get_tensor_info(id);
                let dtype = info.and_then(|i| i.dtype);
                println!("  {} (dtype={:?})", name, dtype);
            }
        }
        return;
    }

    let mut observer = TensorDumpObserver::new(graph, &tensor_names);

    // Prepare inputs
    let model_input_info = model.get_input_tensor_info().unwrap();

    // Load explicitly provided .npy inputs
    let npy_inputs: HashMap<String, _> = input_npys
        .iter()
        .map(|(name, path)| {
            let tensor = npy::read_npy_file(path, &pool).unwrap_or_else(|e| {
                eprintln!(
                    "Failed to read input '{}' from {}: {e}",
                    name,
                    path.display()
                );
                std::process::exit(1);
            });
            (name.clone(), tensor)
        })
        .collect();

    // Auto-fill missing inputs with zeros
    let mut zero_inputs = HashMap::new();
    for (name, (legacy_dtype, shape)) in &model_input_info {
        if npy_inputs.contains_key(name) {
            continue;
        }
        let Some(dtype) = NumericDType::from_legacy(*legacy_dtype) else {
            continue;
        };
        let concrete_shape: Vec<u64> = shape.iter().map(|d| d.unwrap_or(0)).collect();
        let layout = TensorLayout::<DynRank>::row_major(concrete_shape.clone(), dtype);
        let buf = pool
            .allocate(layout.buffer_size_bytes())
            .expect("pool alloc");
        // Buffer is zeroed by default.
        zero_inputs.insert(
            name.clone(),
            whisper_tensor::numeric_tensor::NumericTensor::<DynRank, SystemPool>::from_parts(
                buf, layout,
            ),
        );
        eprintln!(
            "  Auto-filled '{}': dtype={:?} shape={:?}",
            name, dtype, concrete_shape
        );
    }

    // Build view map: npy inputs + zero inputs → name→view
    let tensors_by_name = graph.get_tensors_by_name();

    let npy_views: Vec<_> = npy_inputs
        .iter()
        .map(|(name, t)| (name.clone(), t.view()))
        .collect();
    let zero_views: Vec<_> = zero_inputs
        .iter()
        .map(|(name, t)| (name.clone(), t.view()))
        .collect();

    let mut id_input_views = HashMap::new();
    for (name, view) in npy_views.iter().chain(zero_views.iter()) {
        if let Some(&id) = tensors_by_name.get(name) {
            id_input_views.insert(id, view);
        }
    }

    // Run eval with observer
    eprintln!("Running eval...");
    let tensor_store = model.get_tensor_store();
    let outputs = graph
        .pool_eval_with_store_observed(&id_input_views, tensor_store, &pool, &mut observer)
        .unwrap_or_else(|e| {
            eprintln!("Eval failed: {e}");
            std::process::exit(1);
        });

    // Map output IDs back to names for reporting
    let id_to_name: HashMap<GlobalId, &str> = tensors_by_name
        .iter()
        .map(|(name, &id)| (id, name.as_str()))
        .collect();

    eprintln!(
        "Eval complete. {} outputs, observer called {} times",
        outputs.len(),
        observer.call_count
    );
    for (id, t) in &outputs {
        let name = id_to_name.get(id).unwrap_or(&"?");
        eprintln!(
            "  output '{}': shape={:?} dtype={:?}",
            name,
            t.shape(),
            t.dtype()
        );
    }

    // Write captured tensors
    fs::create_dir_all(&output_dir).unwrap_or_else(|e| {
        eprintln!("Failed to create output dir: {e}");
        std::process::exit(1);
    });

    for (name, tensor) in &observer.captured {
        let safe_name = name.replace('/', "__").replace('.', "_");
        let path = output_dir.join(format!("{safe_name}.npy"));
        write_npy_file(&path, &tensor.view()).unwrap_or_else(|e| {
            eprintln!("Failed to write '{}': {e}", path.display());
        });
        eprintln!("Wrote {}", path.display());
    }

    eprintln!("Done. {} tensors captured.", observer.captured.len());
}
