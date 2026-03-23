//! `wt dump-tensors` — run a model and dump selected intermediate tensors.
//!
//! Loads an ONNX-like model via the standard loader pipeline, feeds it the
//! given inputs (from .npy files), evaluates the full graph, and writes any
//! tensor whose ONNX name matches the requested list to the output directory
//! as .npy files.  Names that don't match any tensor in the graph are reported
//! but do not cause failure.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::dtype::DType;
use whisper_tensor::graph::GlobalId;
use whisper_tensor::model::{Model, ModelExecutionRuntime};
use whisper_tensor::migration::numeric_tensor::NumericTensor;
use whisper_tensor::symbolic_graph::observer::SymbolicGraphObserver;
use whisper_tensor::tensor_rank::DynRank;

use whisper_tensor_import::onnx_graph::WeightStorageStrategy;

// ---------------------------------------------------------------------------
// .npy reader (minimal, matches the one in tests/accuracy.rs)
// ---------------------------------------------------------------------------

fn read_npy(path: &Path) -> Result<NumericTensor<DynRank>, String> {
    let data = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if data.len() < 10 || &data[..6] != b"\x93NUMPY" {
        return Err(format!("not a .npy file: {}", path.display()));
    }
    let major = data[6];
    let header_len = if major >= 2 {
        u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize
    } else {
        u16::from_le_bytes([data[8], data[9]]) as usize
    };
    let header_start = if major >= 2 { 12 } else { 10 };
    let header_end = header_start + header_len;
    let header =
        std::str::from_utf8(&data[header_start..header_end]).map_err(|e| format!("{e}"))?;

    let dtype = parse_npy_dtype(header)?;
    let shape = parse_npy_shape(header)?;
    let raw = &data[header_end..];

    let nd = NDArrayNumericTensor::from_raw_data(raw, dtype, shape)
        .map_err(|e| format!("build tensor: {e}"))?;
    Ok(NumericTensor::NDArray(nd))
}

fn parse_npy_dtype(header: &str) -> Result<DType, String> {
    let descr_start = header
        .find("'descr'")
        .or_else(|| header.find("\"descr\""))
        .ok_or("no 'descr' in npy header")?;
    let rest = &header[descr_start..];
    let colon = rest.find(':').ok_or("no colon after descr")?;
    let after = &rest[colon + 1..];
    let q = if after.contains('\'') { '\'' } else { '"' };
    let s = after.find(q).ok_or("no quote")? + 1;
    let inner = &after[s..];
    let e = inner.find(q).ok_or("no closing quote")?;
    let t = inner[..e].trim_start_matches(['<', '>', '=', '|']);
    match t {
        "f8" | "float64" => Ok(DType::F64),
        "f4" | "float32" => Ok(DType::F32),
        "f2" | "float16" => Ok(DType::F16),
        "i8" | "int64" => Ok(DType::I64),
        "i4" | "int32" => Ok(DType::I32),
        "i2" | "int16" => Ok(DType::I16),
        "i1" | "int8" => Ok(DType::I8),
        "u8" | "uint64" => Ok(DType::U64),
        "u4" | "uint32" => Ok(DType::U32),
        "u1" | "uint8" => Ok(DType::U8),
        "b1" => Ok(DType::BOOL),
        other => Err(format!("unsupported npy dtype: {other}")),
    }
}

fn parse_npy_shape(header: &str) -> Result<Vec<u64>, String> {
    let start = header
        .find("'shape'")
        .or_else(|| header.find("\"shape\""))
        .ok_or("no 'shape'")?;
    let rest = &header[start..];
    let open = rest.find('(').ok_or("no '('")?;
    let close = rest.find(')').ok_or("no ')'")?;
    let inner = rest[open + 1..close].trim();
    if inner.is_empty() {
        return Ok(vec![]);
    }
    inner
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<u64>().map_err(|e| format!("bad dim '{s}': {e}")))
        .collect()
}

// ---------------------------------------------------------------------------
// .npy writer
// ---------------------------------------------------------------------------

fn write_npy(path: &Path, tensor: &NumericTensor<DynRank>) -> Result<(), String> {
    let nd = tensor
        .to_ndarray()
        .map_err(|e| format!("to_ndarray: {e}"))?;
    let shape = nd.shape().to_vec();

    // Cast to F32 for .npy compatibility, then extract raw f32 values
    let nd = if nd.dtype() != DType::F32 {
        nd.cast(DType::F32)
            .map_err(|e| format!("cast to f32: {e}"))?
    } else {
        nd
    };
    let flat: Vec<f32> = nd
        .flatten()
        .try_to_vec()
        .map_err(|e| format!("flatten to vec: {e}"))?;

    // Write .npy v1 format
    let shape_str = shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let shape_str = if shape.len() == 1 {
        format!("{},", shape_str)
    } else {
        shape_str
    };
    let header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}), }}",
        shape_str
    );
    let prefix_len = 10usize;
    let unpadded = prefix_len + header.len() + 1;
    let padded = (unpadded + 63) & !63;
    let pad = padded - unpadded;
    let padded_header = format!("{}{}\n", header, " ".repeat(pad));

    let raw_bytes: &[u8] = bytemuck_cast_slice(&flat);
    let mut out = Vec::with_capacity(prefix_len + padded_header.len() + raw_bytes.len());
    out.extend_from_slice(b"\x93NUMPY");
    out.push(1);
    out.push(0);
    out.extend_from_slice(&(padded_header.len() as u16).to_le_bytes());
    out.extend_from_slice(padded_header.as_bytes());
    out.extend_from_slice(raw_bytes);
    fs::write(path, &out).map_err(|e| format!("write {}: {e}", path.display()))
}

fn bytemuck_cast_slice(data: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) }
}

// ---------------------------------------------------------------------------
// Observer
// ---------------------------------------------------------------------------

pub struct TensorDumpObserver {
    /// GlobalId → ONNX name, for the tensors we want to capture.
    watched_ids: HashMap<GlobalId, String>,
    /// Captured tensor values, keyed by ONNX name.
    pub captured: HashMap<String, NumericTensor<DynRank>>,
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
        // Report unmatched names
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
    fn on_op_executed(&mut self, _: &[GlobalId], _: Instant, _: Instant, _: &mut EvalBackend) {}
    fn on_tensor_assigned(
        &mut self,
        tensor_path: &[GlobalId],
        tensor: &NumericTensor<DynRank>,
        _backend: &mut EvalBackend,
    ) {
        if let Some(id) = tensor_path.last() {
            self.call_count += 1;
            if let Some(name) = self.watched_ids.get(id) {
                let preview: Vec<f32> = tensor
                    .to_ndarray()
                    .and_then(|nd| Ok(nd.cast(DType::F32)?))
                    .map(|nd: NDArrayNumericTensor<DynRank>| {
                        let v: Result<Vec<f32>, _> = nd.flatten().try_into();
                        v.unwrap_or_default()
                    })
                    .unwrap_or_default();
                let n = preview.len().min(3);
                eprintln!(
                    "  captured '{}': shape={:?} dtype={:?} first{}={:.6?}",
                    name,
                    tensor.shape(),
                    tensor.dtype(),
                    n,
                    &preview[..n]
                );
                self.captured.insert(name.clone(), tensor.clone());
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
    // Import model
    let t0 = std::time::Instant::now();
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
    let t1 = std::time::Instant::now();
    let mut rng = rand::rng();
    let model = Model::new_from_onnx(&onnx_data, &mut rng, Some(&model_path)).unwrap_or_else(|e| {
        eprintln!("Failed to load ONNX: {e}");
        std::process::exit(1);
    });
    eprintln!("ONNX parse: {:.1}s", t1.elapsed().as_secs_f32());
    std::io::stderr().flush().ok();

    // Build observer
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
    let mut inputs: HashMap<String, NumericTensor<DynRank>> = HashMap::new();

    // Load explicitly provided .npy inputs
    for (name, path) in &input_npys {
        let tensor = read_npy(path).unwrap_or_else(|e| {
            eprintln!(
                "Failed to read input '{}' from {}: {e}",
                name,
                path.display()
            );
            std::process::exit(1);
        });
        inputs.insert(name.clone(), tensor);
    }

    // Auto-fill missing inputs with zeros (same as accuracy test)
    for (name, (dtype, shape)) in &model_input_info {
        if inputs.contains_key(name) {
            continue;
        }
        let concrete_shape: Vec<u64> = shape.iter().map(|d| d.unwrap_or(0)).collect();
        let numel: usize = concrete_shape.iter().product::<u64>() as usize;
        if let Some(elem_size) = dtype.bytes_per_element() {
            let zeros = vec![0u8; numel * elem_size];
            if let Ok(nd) =
                NDArrayNumericTensor::from_raw_data(&zeros, *dtype, concrete_shape.clone())
            {
                inputs.insert(name.clone(), NumericTensor::NDArray(nd));
                eprintln!(
                    "  Auto-filled '{}': dtype={:?} shape={:?}",
                    name, dtype, concrete_shape
                );
            }
        }
    }

    // Run
    eprintln!("Running eval...");
    let mut runtime = ModelExecutionRuntime::Eval(EvalBackend::NDArray);
    let outputs = model
        .run(inputs, &mut observer, &mut runtime)
        .unwrap_or_else(|e| {
            eprintln!("Eval failed: {e}");
            std::process::exit(1);
        });

    eprintln!(
        "Eval complete. {} outputs, observer called {} times",
        outputs.len(),
        observer.call_count
    );
    for (name, t) in &outputs {
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
        write_npy(&path, tensor).unwrap_or_else(|e| {
            eprintln!("Failed to write '{}': {e}", path.display());
        });
        eprintln!("Wrote {}", path.display());
    }

    eprintln!("Done. {} tensors captured.", observer.captured.len());
}
