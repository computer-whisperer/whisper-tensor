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
use crate::numeric_dtype::NumericDType;
use crate::numeric_tensor::{NumericTensor, NumericTensorView, TensorLayout};
use crate::pool::{Pool, SystemPool};
use crate::super_graph::cache::{CachedTensor, LoadedTensorCache};
use crate::symbolic_graph::tensor_store::TensorStore;
use crate::symbolic_graph::{StoredOrNotTensor, SymbolicGraph, TensorType};
use crate::tensor_info::TensorInfo;
use crate::tensor_rank::DynRank;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

static POOL_S: SystemPool = SystemPool;

/// Resolve a `StoredOrNotTensor` into a pool-backed tensor.
///
/// Allocates through the caller-provided pool so every weight-load
/// allocation is counted against whichever pool the eval path is using
/// (typically the scheduler's short-lived `TrackedPool`). Previously
/// this was hardcoded to `SystemPool`, which meant weight materialization
/// on every `execute_compiled` / `execute_lowered` call fell back to the
/// global allocator and was invisible to the stats sampler.
pub(super) fn resolve_stored_tensor<'p, P: Pool>(
    stored_ref: &StoredOrNotTensor,
    tensor_store: &TensorStore,
    pool: &'p P,
) -> Option<NumericTensor<'p, DynRank, P>> {
    match stored_ref {
        StoredOrNotTensor::Stored(store_id) => tensor_store
            .get_tensor(*store_id)
            .and_then(|s| s.to_pool_tensor(pool)),
        StoredOrNotTensor::Inline(shared) => {
            let src = shared.inner();
            let layout = TensorLayout::<DynRank>::row_major(src.shape().clone(), src.dtype());
            pool.allocate(layout.buffer_size_bytes()).ok().map(|buf| {
                let mut tensor = NumericTensor::from_parts(buf, layout);
                for i in 0..src.numel() {
                    tensor.write_element(i, src.read_element(i));
                }
                tensor
            })
        }
    }
}

/// Read shape and (post-dequant) element dtype from a `StoredOrNotTensor`
/// without materializing the tensor data.
///
/// For quantized formats this returns the dequantized element dtype (F32),
/// matching what `tensor.dtype()` reports on a tensor produced by
/// `resolve_stored_tensor` / `to_pool_tensor`.
///
/// Returns `None` only if the store id is missing — never reads the file.
pub(super) fn cheap_shape_dtype(
    stored_ref: &StoredOrNotTensor,
    tensor_store: &TensorStore,
) -> Option<(Vec<u64>, NumericDType)> {
    match stored_ref {
        StoredOrNotTensor::Stored(id) => {
            let st = tensor_store.get_tensor(*id)?;
            Some((st.shape(), st.format().element_dtype()))
        }
        StoredOrNotTensor::Inline(shared) => {
            let inner = shared.inner();
            Some((inner.shape().clone(), inner.dtype()))
        }
    }
}

/// Cached result of lowering a symbolic graph to a NanoGraph.
pub struct CachedLoweredModel {
    /// Hash of the info_inputs used to produce this lowering.
    /// Used to detect when the cache is stale (different input shapes,
    /// different weight data for small constants, etc.).
    pub info_inputs_hash: u64,
    /// The lowered nano graph, ready for pool_eval.
    /// Contains `tensor_map` for GlobalId → atom layout mapping.
    pub graph: NanoGraph<'static, SystemPool>,
    /// Milli graph input_map: external ID → internal ID.
    pub input_map: HashMap<GlobalId, GlobalId>,
    /// Milli graph output_map: internal ID → external ID.
    pub output_map: HashMap<GlobalId, GlobalId>,
    /// Symbolic graph tensor ID → milli-internal ID. Used to request
    /// intermediate tensors as nano graph outputs for observer reporting.
    pub sym_to_internal: HashMap<GlobalId, GlobalId>,
    /// Milli-graph external IDs for user inputs (tokens, audio, etc.).
    pub user_input_ext_ids: Vec<GlobalId>,
    /// Milli-graph external IDs for weight/constant inputs that were NOT
    /// inlined during lowering (above the threshold). These must be loaded
    /// from the TensorStore and passed to pool_eval at runtime.
    pub weight_input_ext_ids: Vec<GlobalId>,
    /// Provenance: for each nano group index, the (milli_op_id, op_kind)
    /// that produced it. Used by Build Inspector reporting.
    pub group_provenance: Vec<(GlobalId, String)>,
    /// Ops that could not be lowered (treated as opaque boundary).
    pub unsupported: Vec<(GlobalId, String)>,
    /// Human-readable detail for each unsupported op (input/output shapes).
    pub unsupported_details: Vec<String>,
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

/// Hash user-provided runtime inputs (IDs + dtype + concrete shape) plus the
/// inline-constant policy to detect whether compiled-eval info_inputs can be
/// reused without re-walking the whole symbolic graph.
pub fn hash_user_input_views(
    user_input_views: &HashMap<GlobalId, NumericTensorView<'_, DynRank>>,
    inline_constant_threshold: u64,
) -> u64 {
    use std::collections::BTreeMap;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    inline_constant_threshold.hash(&mut hasher);
    let sorted: BTreeMap<&GlobalId, &NumericTensorView<'_, DynRank>> =
        user_input_views.iter().collect();
    for (&id, view) in &sorted {
        id.hash(&mut hasher);
        view.dtype().hash(&mut hasher);
        let shape = view.shape();
        shape.len().hash(&mut hasher);
        for &dim in shape {
            dim.hash(&mut hasher);
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

    // Declared model inputs: always shape+dtype only, never inlined.
    // These are runtime-overridable slots — even if the model provides default
    // values (Input(Some(stored))), the user may supply different data at eval time.
    let ordered_input_set: std::collections::HashSet<GlobalId> =
        sym_graph.get_ordered_inputs().iter().copied().collect();

    for &input_id in sym_graph.get_ordered_inputs() {
        if let Some(view) = user_input_views.get(&input_id) {
            // User provided data — use its shape/dtype.
            let shape: Vec<u64> = view.shape().to_vec();
            let dtype = view.dtype();
            info_inputs.insert(input_id, TensorInfo::from_dtype_and_shape(dtype, &shape));
            user_input_ext_ids.push(input_id);
        } else if let Some(tensor_meta) = sym_graph.get_tensor_info(input_id) {
            // No user view — read shape/dtype from the model's stored default
            // without materializing it. We never need its data here: the default
            // or runtime data will be loaded at eval time.
            if let TensorType::Input(Some(stored_ref)) = &tensor_meta.tensor_type
                && let Some((shape, dtype)) = cheap_shape_dtype(stored_ref, tensor_store)
            {
                info_inputs.insert(input_id, TensorInfo::from_dtype_and_shape(dtype, &shape));
                user_input_ext_ids.push(input_id);
            }
        }
    }

    // Constants (not declared inputs): apply threshold policy.
    for (&tensor_id, tensor_meta) in sym_graph.get_tensors() {
        let stored_ref = match &tensor_meta.tensor_type {
            TensorType::Constant(s) | TensorType::Input(Some(s)) => s,
            _ => continue,
        };
        // Skip declared model inputs — handled above as runtime slots.
        if ordered_input_set.contains(&tensor_id) {
            continue;
        }
        // Skip if already in info_inputs.
        if info_inputs.contains_key(&tensor_id) {
            continue;
        }

        // Cheap shape/dtype lookup first — avoids materializing every weight
        // (~1.3 s/iter on RWKV-0.1B) just to check the threshold.
        let Some((shape, dtype)) = cheap_shape_dtype(stored_ref, tensor_store) else {
            continue;
        };
        let numel: u64 = shape.iter().product();

        if numel <= inline_constant_threshold {
            // Small constant: actually load it so the data is available for
            // constant folding during lowering. Lowering is one-time per
            // model and the resulting TensorInfo is cached in the lowered
            // model — we keep it on the static `POOL_S` so the cached info
            // retains its `'static` lifetime.
            if let Some(tensor) = resolve_stored_tensor(stored_ref, tensor_store, &POOL_S) {
                info_inputs.insert(tensor_id, TensorInfo::from_view(&tensor.view(), &POOL_S));
            }
        } else {
            // Large constant (weight): shape+dtype only — will be supplied as a
            // runtime input by `execute_compiled` (which loads the actual bytes
            // via `resolve_stored_tensor` at that point).
            info_inputs.insert(tensor_id, TensorInfo::from_dtype_and_shape(dtype, &shape));
            weight_input_ext_ids.push(tensor_id);
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
    let (milli_graph, sym_to_internal) = sym_graph.generate_milli_graph_with_id_map(&mut rng);

    // lower() expects keys matching the milli graph's external input IDs,
    // which generate_milli_graph preserves from the symbolic graph tensor IDs.
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
        input_map: milli_graph.input_map.clone(),
        output_map,
        sym_to_internal,
        user_input_ext_ids,
        weight_input_ext_ids,
        group_provenance: lower_result.group_provenance,
        unsupported: lower_result.unsupported,
        unsupported_details: lower_result.unsupported_details,
    })
}

/// Holding pen for resolved weight tensors. Each entry's view borrows from
/// either the long-lived `LoadedTensorCache` (cache hit) or from a transient
/// pool tensor stored in the `transients` vec (cache miss with no cache, or
/// `Inline` variant). The lifetime parameter `'a` is the shorter of the
/// cache borrow and the local stack frame.
pub(crate) struct PreparedWeights<'a, 'p, P: Pool + 'p> {
    /// Transient pool-allocated weights — held to keep their views alive
    /// for the duration of the eval call. Index-aligned with `slots` entries
    /// that have `transient_idx = Some(_)`.
    transients: Vec<NumericTensor<'p, DynRank, P>>,
    /// One entry per stored id, in input order, telling us where to find
    /// the view: either index into `transients` or borrow from the cache.
    slots: Vec<PreparedSlot<'a>>,
}

enum PreparedSlot<'a> {
    Transient {
        ext_id: GlobalId,
        idx: usize,
    },
    Cached {
        ext_id: GlobalId,
        tensor: &'a CachedTensor,
    },
}

impl<'a, 'p, P: Pool + 'p> PreparedWeights<'a, 'p, P> {
    pub(crate) fn views(&self) -> Vec<(GlobalId, NumericTensorView<'_, DynRank>)> {
        self.slots
            .iter()
            .map(|slot| match slot {
                PreparedSlot::Transient { ext_id, idx } => (*ext_id, self.transients[*idx].view()),
                PreparedSlot::Cached { ext_id, tensor } => (*ext_id, tensor.view()),
            })
            .collect()
    }
}

/// Resolve every external id in `stored_ids` to a weight tensor, using the
/// `loaded_tensor_cache` when available.
///
/// Behavior per id:
/// - If the symbolic graph's `StoredOrNotTensor` is `Stored(store_id)` and a
///   cache is present, ensure the cache has the entry (loading from disk on
///   miss into the cache pool) and borrow a view of it.
/// - Otherwise (no cache, or `Inline` variant), allocate a transient in the
///   execution `pool` via `resolve_stored_tensor`. The transient is held in
///   the returned `PreparedWeights` so its view stays valid for the duration
///   of the eval call.
pub(crate) fn prepare_weights<'a, 'p, P: Pool + 'p>(
    stored_ids: &[GlobalId],
    sym_graph: &SymbolicGraph,
    tensor_store: &TensorStore,
    pool: &'p P,
    mut loaded_tensor_cache: Option<&'a mut LoadedTensorCache>,
) -> Result<PreparedWeights<'a, 'p, P>, super::SuperGraphError> {
    // Walk the symbolic graph once to collect refs to each StoredOrNotTensor.
    let mut refs: Vec<(GlobalId, &StoredOrNotTensor)> = Vec::with_capacity(stored_ids.len());
    for &ext_id in stored_ids {
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
        refs.push((ext_id, stored_ref));
    }

    // Phase 1: mutate the cache to ensure all `Stored(id)` entries are
    // populated. We do this through a temporary `&mut` reborrow so that
    // we can later downgrade the original `&'a mut` to a `&'a` while
    // preserving the original lifetime.
    if let Some(cache) = loaded_tensor_cache.as_deref_mut() {
        for (_ext_id, stored_ref) in &refs {
            if let StoredOrNotTensor::Stored(store_id) = stored_ref {
                // ensure_loaded is idempotent and cheap on hit.
                let _ = cache.ensure_loaded(*store_id, tensor_store);
            }
        }
    }

    // Phase 2: downgrade the `Option<&'a mut LoadedTensorCache>` to
    // `Option<&'a LoadedTensorCache>`. The match form lets the compiler
    // coerce `&'a mut T` → `&'a T` while preserving the lifetime — `as_deref`
    // would shorten it to the local borrow and break PreparedSlot::Cached.
    let cache_ref: Option<&'a LoadedTensorCache> = match loaded_tensor_cache {
        Some(c) => Some(c),
        None => None,
    };

    let mut transients: Vec<NumericTensor<'p, DynRank, P>> = Vec::new();
    let mut slots: Vec<PreparedSlot<'a>> = Vec::with_capacity(refs.len());

    for (ext_id, stored_ref) in refs {
        let cached_view: Option<&CachedTensor> = match (cache_ref, stored_ref) {
            (Some(c), StoredOrNotTensor::Stored(id)) => c.get(id),
            _ => None,
        };
        if let Some(tensor) = cached_view {
            slots.push(PreparedSlot::Cached { ext_id, tensor });
        } else {
            let tensor =
                resolve_stored_tensor(stored_ref, tensor_store, pool).ok_or_else(|| {
                    super::SuperGraphError::InvalidGraph(format!(
                        "lowered_eval: failed to load stored tensor {ext_id:?}"
                    ))
                })?;
            let idx = transients.len();
            transients.push(tensor);
            slots.push(PreparedSlot::Transient { ext_id, idx });
        }
    }

    Ok(PreparedWeights { transients, slots })
}

/// Execute a cached lowered model with the given inputs.
///
/// Loads weight data from the tensor store and combines with user inputs,
/// then runs pool_eval on the cached NanoGraph.
///
/// If `intermediate_sym_ids` is non-empty, those symbolic graph tensor IDs
/// are also requested as nano graph outputs and returned in the result map
/// (for observer/debugging purposes).
pub fn execute_lowered<'p, P: Pool + 'p>(
    cached: &CachedLoweredModel,
    sym_graph: &SymbolicGraph,
    tensor_store: &TensorStore,
    user_input_views: &HashMap<GlobalId, NumericTensorView<'_, DynRank>>,
    intermediate_sym_ids: &[GlobalId],
    pool: &'p P,
    loaded_tensor_cache: Option<&mut LoadedTensorCache>,
) -> Result<HashMap<GlobalId, NumericTensor<'p, DynRank, P>>, super::SuperGraphError> {
    // Resolve weights via `prepare_weights`. Cache hits give zero-copy views;
    // misses are loaded into the cache (or kept as transients in the
    // execution pool when no cache is available).
    let stored_ids: Vec<GlobalId> = cached
        .weight_input_ext_ids
        .iter()
        .chain(
            cached
                .user_input_ext_ids
                .iter()
                .filter(|id| !user_input_views.contains_key(id)),
        )
        .copied()
        .collect();

    let prepared = prepare_weights(
        &stored_ids,
        sym_graph,
        tensor_store,
        pool,
        loaded_tensor_cache,
    )?;
    let weight_views = prepared.views();

    // Collect all input pairs: (TAMI, view).
    // tensor_map uses milli-internal IDs, so translate ext→internal via input_map.
    let mut eval_inputs: Vec<(&TensorAtomMapInfo, &NumericTensorView<'_, DynRank>)> = Vec::new();

    let all_views: Vec<(GlobalId, &NumericTensorView<'_, DynRank>)> = user_input_views
        .iter()
        .map(|(&id, v)| (id, v))
        .chain(weight_views.iter().map(|(id, v)| (*id, v)))
        .collect();

    let mut n_matched = 0usize;
    let mut n_skipped = 0usize;
    for &(ext_id, view) in &all_views {
        let internal_id = cached.input_map.get(&ext_id).copied().unwrap_or(ext_id);
        if let Some(tami) = cached.graph.tensor_map.get(&internal_id) {
            eval_inputs.push((tami, view));
            n_matched += 1;
        } else {
            n_skipped += 1;
        }
    }
    eprintln!(
        "[lowered_eval] inputs: {} matched, {} skipped (no TAMI), {} total in input_map",
        n_matched,
        n_skipped,
        cached.input_map.len(),
    );

    // Build output TAMIs from the symbolic graph's ordered outputs
    // plus any requested intermediate tensors.
    // output_map is internal→external, so build reverse (external→internal).
    let reverse_output: HashMap<GlobalId, GlobalId> = cached
        .output_map
        .iter()
        .map(|(&internal, &external)| (external, internal))
        .collect();

    // Resolve a symbolic tensor ID → milli-internal ID, checking both
    // the output_map (for graph outputs) and sym_to_internal (for intermediates).
    let resolve = |sym_id: &GlobalId| -> Option<GlobalId> {
        reverse_output
            .get(sym_id)
            .copied()
            .or_else(|| cached.sym_to_internal.get(sym_id).copied())
    };

    let mut output_tamis: Vec<(GlobalId, &TensorAtomMapInfo)> = sym_graph
        .get_ordered_outputs()
        .iter()
        .filter_map(|ext_id| {
            let internal_id = resolve(ext_id).unwrap_or(*ext_id);
            cached
                .graph
                .tensor_map
                .get(&internal_id)
                .map(|tami| (*ext_id, tami))
        })
        .collect();

    // Add requested intermediates (skip any already in outputs or missing from tensor_map).
    let output_sym_ids: std::collections::HashSet<GlobalId> =
        output_tamis.iter().map(|(id, _)| *id).collect();
    for sym_id in intermediate_sym_ids {
        if !output_sym_ids.contains(sym_id) {
            let internal_id = resolve(sym_id).unwrap_or(*sym_id);
            if let Some(tami) = cached.graph.tensor_map.get(&internal_id) {
                output_tamis.push((*sym_id, tami));
            }
        }
    }

    let output_tami_refs: Vec<&TensorAtomMapInfo> =
        output_tamis.iter().map(|(_, tami)| *tami).collect();

    // Run pool_eval.
    let t0 = std::time::Instant::now();
    let eval_results =
        pool_eval::pool_eval(&cached.graph, &eval_inputs, &output_tami_refs, &[], pool)
            .map_err(|e| super::SuperGraphError::InvalidGraph(format!("lowered pool_eval: {e}")))?;
    let dt = t0.elapsed();
    if dt.as_millis() > 10 {
        eprintln!("[lowered_eval] pool_eval: {:.0}ms", dt.as_secs_f64() * 1e3,);
    }

    // Map results to external output IDs.
    let mut outputs: HashMap<GlobalId, NumericTensor<'p, DynRank, P>> = HashMap::new();
    for ((ext_id, _), tensor) in output_tamis.iter().zip(eval_results) {
        outputs.insert(*ext_id, tensor);
    }

    Ok(outputs)
}
