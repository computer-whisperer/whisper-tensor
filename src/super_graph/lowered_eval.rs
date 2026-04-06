//! Lowered evaluation path for ModelExecution nodes.
//!
//! Instead of evaluating a SymbolicGraph op-by-op (each independently lowering
//! to milli → nano → pool_eval), this module lowers the entire SymbolicGraph
//! to a single NanoGraph once, caches it, and re-executes via pool_eval on
//! subsequent calls.

use crate::graph::GlobalId;
use crate::nano_graph::lower::{self, TensorAtomMapInfo};
use crate::nano_graph::pattern::NanoGraph;
use crate::nano_graph::pool_eval;
use crate::numeric_tensor::{NumericTensor, NumericTensorView, TensorLayout};
use crate::pool::{Pool, SystemPool};
use crate::symbolic_graph::tensor_store::TensorStore;
use crate::symbolic_graph::{StoredOrNotTensor, SymbolicGraph, TensorType};
use crate::tensor_info::TensorInfo;
use crate::tensor_rank::DynRank;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

static POOL_S: SystemPool = SystemPool;

/// Cached result of lowering a symbolic graph to a NanoGraph.
pub struct CachedLoweredModel {
    /// Hash of the info_inputs used to produce this lowering.
    /// Used to detect when the cache is stale (different input shapes,
    /// different weight data for small constants, etc.).
    pub info_inputs_hash: u64,
    /// The lowered nano graph, ready for pool_eval.
    pub graph: NanoGraph<'static, SystemPool>,
    /// Mapping from milli-internal tensor GlobalId → nano atom layout.
    pub tensor_map: HashMap<GlobalId, TensorAtomMapInfo>,
    /// Milli graph input_map: external ID → internal ID.
    pub input_map: HashMap<GlobalId, GlobalId>,
    /// Milli graph output_map: internal ID → external ID.
    pub output_map: HashMap<GlobalId, GlobalId>,
    /// Milli-graph external IDs for user inputs (tokens, audio, etc.).
    pub user_input_ext_ids: Vec<GlobalId>,
    /// Milli-graph external IDs for weight/constant inputs that were NOT
    /// inlined during lowering (above the threshold). These must be loaded
    /// from the TensorStore and passed to pool_eval at runtime.
    pub weight_input_ext_ids: Vec<GlobalId>,
}

/// Compute a hash over the info_inputs map that captures everything affecting
/// the lowered NanoGraph structure: tensor IDs, dtypes, shapes, and the raw
/// data bytes of any inlined constants.
pub fn hash_info_inputs(info_inputs: &HashMap<GlobalId, TensorInfo<'_, '_, SystemPool>>) -> u64 {
    use std::collections::BTreeMap;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();

    // Sort by GlobalId for deterministic ordering.
    let sorted: BTreeMap<&GlobalId, &TensorInfo<'_, '_, SystemPool>> = info_inputs.iter().collect();

    for (&id, info) in &sorted {
        id.hash(&mut hasher);

        // Hash dtype (including payload — Float(F32) vs Float(BF16), etc.).
        info.dtype().hash(&mut hasher);

        // Hash shape (as concrete dims where available).
        if let Some(ranked) = info.as_ranked() {
            let shape = ranked.shape();
            shape.len().hash(&mut hasher);
            for dim in shape {
                match dim {
                    crate::scalar_info::ScalarInfoTyped::Numeric(v) => {
                        0u8.hash(&mut hasher);
                        v.hash(&mut hasher);
                    }
                    _ => {
                        1u8.hash(&mut hasher);
                    }
                }
            }
        }

        // Hash raw buffer bytes if concrete data is present (inlined constants).
        // Only entries under the threshold have data, so this is bounded.
        // Uses raw bytes rather than per-element f64 conversion to avoid
        // lossiness for non-f64 types (bf16, i64 > 2^53, etc.).
        if let Some(cow) = info.as_concrete() {
            cow.buffer().hash(&mut hasher);
        }
    }

    hasher.finish()
}

/// Check whether a symbolic graph can use the lowered eval path.
/// Returns false if any operation contains sub-graphs (Scan, If, LSTM).
pub fn can_lower_symbolic_graph(sym_graph: &SymbolicGraph) -> bool {
    use crate::symbolic_graph::ops::Operation;

    for op in sym_graph.get_operations().values() {
        if !op.op.get_sub_graphs().is_empty() {
            return false;
        }
    }
    true
}

/// Build the TensorInfo map for lowering, applying the threshold policy.
///
/// - User inputs: shape+dtype only (no data — would constant-fold everything).
/// - Constants with numel <= threshold: full data (enables constant folding of
///   shape ops, axis indices, small lookup tables).
/// - Constants with numel > threshold: shape+dtype only (flow as inputs at
///   pool_eval time).
///
/// Returns the info_inputs map plus the sets of external IDs classified as
/// user inputs vs weight inputs (above threshold).
pub fn build_info_inputs(
    sym_graph: &SymbolicGraph,
    tensor_store: &TensorStore,
    user_input_views: &HashMap<GlobalId, NumericTensorView<'_, DynRank>>,
    inline_constant_threshold: u64,
) -> (
    HashMap<GlobalId, TensorInfo<'static, 'static, SystemPool>>,
    Vec<GlobalId>, // user_input_ext_ids
    Vec<GlobalId>, // weight_input_ext_ids (above threshold)
) {
    let mut info_inputs: HashMap<GlobalId, TensorInfo<'static, 'static, SystemPool>> =
        HashMap::new();
    let mut user_input_ext_ids = Vec::new();
    let mut weight_input_ext_ids = Vec::new();

    // User inputs: shape+dtype only.
    for &input_id in sym_graph.get_ordered_inputs() {
        if let Some(view) = user_input_views.get(&input_id) {
            let shape: Vec<u64> = view.shape().to_vec();
            let dtype = view.dtype();
            eprintln!("[lowered_eval] user input {input_id:?}: {dtype:?} {shape:?}",);
            info_inputs.insert(input_id, TensorInfo::from_dtype_and_shape(dtype, &shape));
            user_input_ext_ids.push(input_id);
        } else {
            let name = sym_graph
                .get_tensor_info(input_id)
                .and_then(|t| t.onnx_name.as_deref())
                .unwrap_or("?");
            eprintln!(
                "[lowered_eval] WARNING: user input {input_id:?} ({name}) has no view — not in info_inputs",
            );
        }
    }

    // Constants and initialized inputs: apply threshold policy.
    for (&tensor_id, tensor_meta) in sym_graph.get_tensors() {
        let stored_ref = match &tensor_meta.tensor_type {
            TensorType::Constant(s) | TensorType::Input(Some(s)) => s,
            _ => continue,
        };
        // Skip if already handled as a user input.
        if info_inputs.contains_key(&tensor_id) {
            continue;
        }

        // Try to resolve the stored tensor.
        let resolved: Option<NumericTensor<'static, DynRank, SystemPool>> = match stored_ref {
            StoredOrNotTensor::Stored(store_id) => tensor_store
                .get_tensor(*store_id)
                .and_then(|s| s.to_pool_tensor(&POOL_S)),
            StoredOrNotTensor::Inline(shared) => {
                let src = shared.inner();
                let layout = TensorLayout::<DynRank>::row_major(src.shape().clone(), src.dtype());
                POOL_S.allocate(layout.buffer_size_bytes()).ok().map(|buf| {
                    let mut tensor = NumericTensor::from_parts(buf, layout);
                    for i in 0..src.numel() {
                        tensor.write_element(i, src.read_element(i));
                    }
                    tensor
                })
            }
        };

        if let Some(tensor) = resolved {
            let numel = NumericTensor::numel(&tensor) as u64;
            if numel <= inline_constant_threshold {
                // Small constant: full data for constant folding.
                info_inputs.insert(tensor_id, TensorInfo::from_view(&tensor.view(), &POOL_S));
            } else {
                // Large constant (weight): shape+dtype only, will be a runtime input.
                let shape: Vec<u64> = tensor.shape().clone();
                let dtype = tensor.dtype();
                info_inputs.insert(tensor_id, TensorInfo::from_dtype_and_shape(dtype, &shape));
                weight_input_ext_ids.push(tensor_id);
            }
        }
    }

    let n_user = user_input_ext_ids.len();
    let n_weight = weight_input_ext_ids.len();
    let n_inlined = info_inputs.len() - n_user - n_weight;
    eprintln!(
        "[lowered_eval] info_inputs: {} total ({n_user} user, {n_weight} weight, {n_inlined} inlined constants)",
        info_inputs.len(),
    );

    (info_inputs, user_input_ext_ids, weight_input_ext_ids)
}

/// Lower a symbolic graph to a NanoGraph and build the cache entry.
///
/// This performs: generate_milli_graph → lower (infer_all + nano lowering).
/// Returns None if lowering fails (caller should fall back to symbolic eval).
pub fn lower_symbolic_graph(
    sym_graph: &SymbolicGraph,
    info_inputs: &HashMap<GlobalId, TensorInfo<'static, 'static, SystemPool>>,
    info_inputs_hash: u64,
    user_input_ext_ids: Vec<GlobalId>,
    weight_input_ext_ids: Vec<GlobalId>,
) -> Option<CachedLoweredModel> {
    let mut rng = rand::rng();
    let milli_graph = sym_graph.generate_milli_graph(&mut rng);

    // Remap info_inputs: lower() expects keys that match the milli graph's
    // external input IDs, which generate_milli_graph preserves from the
    // symbolic graph tensor IDs.
    let lower_result = lower::lower(&milli_graph, info_inputs, &POOL_S).ok()?;

    // Report unsupported ops with diagnostics.
    if !lower_result.unsupported.is_empty() {
        eprintln!(
            "[lowered_eval] {} unsupported op(s) (will run as opaque):",
            lower_result.unsupported.len(),
        );
        for detail in &lower_result.unsupported_details {
            eprintln!("  {detail}");
        }

        // Trace root cause: find ops whose outputs have unknown dims but whose
        // inputs are all fully known. These are the origin of shape propagation
        // failures.
        use crate::graph::{Graph, Node};
        eprintln!("[lowered_eval] tracing shape propagation failures:");
        for op_id in milli_graph.op_ordering() {
            let Some(op) = milli_graph.get_node_by_id(op_id) else {
                continue;
            };
            let inputs_all_known = op.inputs().all(|id| {
                lower_result.all_infos.get(&id).is_some_and(|info| {
                    if let Some(r) = info.rank_if_known() {
                        (0..r).all(|i| info.dim_if_known(i).is_some())
                    } else {
                        false
                    }
                })
            });
            let any_output_unknown = op.outputs().any(|id| {
                lower_result.all_infos.get(&id).is_none_or(|info| {
                    info.rank_if_known().is_none()
                        || (0..info.rank_if_known().unwrap_or(0))
                            .any(|i| info.dim_if_known(i).is_none())
                })
            });
            if inputs_all_known && any_output_unknown {
                let in_shapes: Vec<String> = op
                    .inputs()
                    .map(|id| {
                        lower::NanoLoweringContext::<SystemPool>::fmt_info(
                            lower_result.all_infos.get(&id),
                        )
                    })
                    .collect();
                let out_shapes: Vec<String> = op
                    .outputs()
                    .map(|id| {
                        lower::NanoLoweringContext::<SystemPool>::fmt_info(
                            lower_result.all_infos.get(&id),
                        )
                    })
                    .collect();
                eprintln!(
                    "  ROOT: {} ({:?}) : ({}) → ({})",
                    op.op_kind(),
                    op_id,
                    in_shapes.join(", "),
                    out_shapes.join(", "),
                );
            }
        }
    }

    let output_map = milli_graph.output_map.as_ref().cloned().unwrap_or_default();

    Some(CachedLoweredModel {
        info_inputs_hash,
        graph: lower_result.graph,
        tensor_map: lower_result.tensor_map,
        input_map: milli_graph.input_map.clone(),
        output_map,
        user_input_ext_ids,
        weight_input_ext_ids,
    })
}

/// Execute a cached lowered model with the given inputs.
///
/// Loads weight data from the tensor store and combines with user inputs,
/// then runs pool_eval on the cached NanoGraph.
pub fn execute_lowered<'p, P: Pool + 'p>(
    cached: &CachedLoweredModel,
    sym_graph: &SymbolicGraph,
    tensor_store: &TensorStore,
    user_input_views: &HashMap<GlobalId, NumericTensorView<'_, DynRank>>,
    pool: &'p P,
) -> Result<HashMap<GlobalId, NumericTensor<'p, DynRank, P>>, super::SuperGraphError> {
    // Build the input pairs for pool_eval: (TAMI, &view) for each external input.
    //
    // We need to provide views for:
    // 1. User inputs (from super graph data)
    // 2. Weight inputs above threshold (loaded from tensor store)

    // Load weight tensors into SystemPool-backed storage.
    let mut weight_tensors: Vec<(GlobalId, NumericTensor<'_, DynRank, SystemPool>)> =
        Vec::with_capacity(cached.weight_input_ext_ids.len());
    for &ext_id in &cached.weight_input_ext_ids {
        let tensor_meta = sym_graph.get_tensor_info(ext_id).ok_or_else(|| {
            super::SuperGraphError::InvalidGraph(format!(
                "lowered_eval: missing tensor info for weight {ext_id:?}"
            ))
        })?;
        let stored_ref = match &tensor_meta.tensor_type {
            TensorType::Constant(s) | TensorType::Input(Some(s)) => s,
            _ => {
                return Err(super::SuperGraphError::InvalidGraph(format!(
                    "lowered_eval: weight {ext_id:?} is not a constant/initialized input"
                )));
            }
        };
        let tensor: NumericTensor<'_, DynRank, SystemPool> = match stored_ref {
            StoredOrNotTensor::Stored(store_id) => tensor_store
                .get_tensor(*store_id)
                .and_then(|s| s.to_pool_tensor(&POOL_S))
                .ok_or_else(|| {
                    super::SuperGraphError::InvalidGraph(format!(
                        "lowered_eval: failed to load weight {ext_id:?} from tensor store"
                    ))
                })?,
            StoredOrNotTensor::Inline(shared) => {
                let src = shared.inner();
                let layout = TensorLayout::<DynRank>::row_major(src.shape().clone(), src.dtype());
                let buf = POOL_S.allocate(layout.buffer_size_bytes()).map_err(|e| {
                    super::SuperGraphError::InvalidGraph(format!(
                        "lowered_eval: allocation for weight {ext_id:?}: {e}"
                    ))
                })?;
                let mut tensor = NumericTensor::from_parts(buf, layout);
                for i in 0..src.numel() {
                    tensor.write_element(i, src.read_element(i));
                }
                tensor
            }
        };
        weight_tensors.push((ext_id, tensor));
    }

    let weight_views: Vec<(GlobalId, NumericTensorView<'_, DynRank>)> = weight_tensors
        .iter()
        .map(|(id, t)| (*id, t.view()))
        .collect();

    // Collect all input pairs: (TAMI, view).
    // tensor_map uses milli-internal IDs, so translate ext→internal via input_map.
    let mut eval_inputs: Vec<(&TensorAtomMapInfo, &NumericTensorView<'_, DynRank>)> = Vec::new();

    let all_views: Vec<(GlobalId, &NumericTensorView<'_, DynRank>)> = user_input_views
        .iter()
        .map(|(&id, v)| (id, v))
        .chain(weight_views.iter().map(|(id, v)| (*id, v)))
        .collect();

    for &(ext_id, view) in &all_views {
        let internal_id = cached.input_map.get(&ext_id).copied().unwrap_or(ext_id);
        if let Some(tami) = cached.tensor_map.get(&internal_id) {
            eval_inputs.push((tami, view));
        }
    }

    // Build output TAMIs from the symbolic graph's ordered outputs.
    // output_map is internal→external, so build reverse (external→internal).
    let reverse_output: HashMap<GlobalId, GlobalId> = cached
        .output_map
        .iter()
        .map(|(&internal, &external)| (external, internal))
        .collect();

    let output_tamis: Vec<(&GlobalId, &TensorAtomMapInfo)> = sym_graph
        .get_ordered_outputs()
        .iter()
        .filter_map(|ext_id| {
            let internal_id = reverse_output.get(ext_id).copied().unwrap_or(*ext_id);
            cached
                .tensor_map
                .get(&internal_id)
                .map(|tami| (ext_id, tami))
        })
        .collect();
    let output_tami_refs: Vec<&TensorAtomMapInfo> =
        output_tamis.iter().map(|(_, tami)| *tami).collect();

    // Run pool_eval.
    let t0 = std::time::Instant::now();
    let eval_results =
        pool_eval::pool_eval(&cached.graph, &eval_inputs, &output_tami_refs, pool)
            .map_err(|e| super::SuperGraphError::InvalidGraph(format!("lowered pool_eval: {e}")))?;
    let dt = t0.elapsed();
    if dt.as_millis() > 10 {
        eprintln!("[lowered_eval] pool_eval: {:.0}ms", dt.as_secs_f64() * 1e3,);
    }

    // Map results to external output IDs.
    let mut outputs: HashMap<GlobalId, NumericTensor<'p, DynRank, P>> = HashMap::new();
    for ((ext_id, _), tensor) in output_tamis.iter().zip(eval_results) {
        outputs.insert(**ext_id, tensor);
    }

    Ok(outputs)
}
