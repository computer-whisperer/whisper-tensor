//! Compiled evaluation path for ModelExecution nodes.
//!
//! Extends the lowered eval path: after lowering SymbolicGraph → NanoGraph,
//! this module partitions and JIT-compiles the NanoGraph into an ExecutablePlan
//! that runs via the v14 executor. The compiled plan is cached for reuse.
//!
//! Requires the `cranelift` feature.
//!
//! The core compile+execute logic lives in standalone `pub(crate)` functions
//! (compile_nano_graph, prepare_compiled_inputs, relayout_to_flat, extract_outputs)
//! so that test_set can reuse them without supergraph dependencies.

use std::collections::HashMap;
use std::time::Instant;

use crate::compiler::attempts::v14::codegen::JitCompiledSpan;
use crate::compiler::attempts::v14::executor::{
    CompiledSpanFn, ExecutablePlan, ExecutablePlanBuilder, PhaseStore, PoolEvalSpan,
};
use crate::compiler::attempts::v14::partitioner_m;
use crate::compiler::attempts::v14::report::{self, PlanSummary};
use crate::compiler::{CodegenKind, CompileOptions, PartitionerKind};
use crate::graph::GlobalId;
use crate::nano_graph::AtomId;
use crate::nano_graph::lower::TensorAtomMapInfo;
use crate::nano_graph::pattern::{AtomRange, NanoGraph};
use crate::numeric_tensor::{NumericTensor, NumericTensorCOW, NumericTensorView, TensorLayout};
use crate::pool::{Pool, SystemPool};
use crate::symbolic_graph::TensorType;
use crate::symbolic_graph::tensor_store::TensorStore;
use crate::tensor_rank::DynRank;

use super::lowered_eval::{self, CachedLoweredModel};
use super::observer::SuperGraphObserver;

// ===========================================================================
// Compiled-eval observer
// ===========================================================================

/// Observer for tagged milestones inside the compiled execution path.
///
/// Mirrors the per-graph-layer observer pattern (SymbolicGraphObserver,
/// MilliOpGraphObserver). Compiled execution emits stage events here so that
/// consumers can build a per-stage timing breakdown without forcing the
/// compiled path to know about super graph types.
///
/// Stage labels are dot-separated, e.g. `compiled.exec.weight_load`,
/// `compiled.exec.jit`, `compiled.lower`. The `iter` field carries a scan
/// iteration index when available; it's `None` outside scan loops.
///
/// At the super graph boundary, `CompiledEvalObserverWrapper` bridges these
/// events into `SuperGraphObserver::on_compiled_milestone`.
pub trait CompiledEvalObserver {
    fn on_milestone(&mut self, stage: &str, iter: Option<u64>, start: Instant, end: Instant);
}

impl CompiledEvalObserver for () {
    fn on_milestone(&mut self, _stage: &str, _iter: Option<u64>, _start: Instant, _end: Instant) {}
}

/// Bridge a `SuperGraphObserver` into the `CompiledEvalObserver` interface
/// expected by the compiled execution path. The wrapper carries the owning
/// super graph node path and an optional scan iteration index, both of which
/// are forwarded with each milestone event.
pub struct CompiledEvalObserverWrapper<'a, T: SuperGraphObserver + ?Sized> {
    inner: &'a mut T,
    path: Vec<GlobalId>,
    iter: Option<u64>,
}

impl<'a, T: SuperGraphObserver + ?Sized> CompiledEvalObserverWrapper<'a, T> {
    pub fn new(inner: &'a mut T, path: Vec<GlobalId>, iter: Option<u64>) -> Self {
        Self { inner, path, iter }
    }
}

impl<T: SuperGraphObserver + ?Sized> CompiledEvalObserver for CompiledEvalObserverWrapper<'_, T> {
    fn on_milestone(&mut self, stage: &str, iter: Option<u64>, start: Instant, end: Instant) {
        // Per-call iter (e.g. inside an inner loop) wins over the wrapper's
        // default iter (e.g. the scan iteration the wrapper was constructed
        // with).
        let iter = iter.or(self.iter);
        self.inner
            .on_compiled_milestone(&self.path, stage, iter, start, end);
    }
}

/// Helper: time a closure and emit a milestone with the given stage label.
#[inline]
pub fn record_stage<R>(
    obs: &mut dyn CompiledEvalObserver,
    stage: &str,
    iter: Option<u64>,
    f: impl FnOnce() -> R,
) -> R {
    let t0 = Instant::now();
    let result = f();
    obs.on_milestone(stage, iter, t0, Instant::now());
    result
}

/// Cached compiled execution plan for a model.
pub struct CachedCompiledPlan {
    /// Hash of the info_inputs used to produce this plan.
    pub info_inputs_hash: u64,
    /// The compiled execution plan ready to run.
    pub executable_plan: ExecutablePlan,
    /// Output atom ranges and their external IDs, for extracting results
    /// from the PhaseStore after execution.
    pub output_ranges: Vec<(GlobalId, Vec<AtomRange>)>,
    /// Output tensor shapes for reassembly.
    pub output_shapes: Vec<(GlobalId, Vec<u64>)>,
    /// Structured summary of the execution plan for Build Inspector reporting.
    pub plan_summary: PlanSummary,
}

// ===========================================================================
// Core reusable functions (no supergraph dependency)
// ===========================================================================

/// Build output atom ranges and shapes from a tensor_map for the given output IDs.
///
/// For each output ID, looks up its TAMI via `resolve_id` (which maps external
/// IDs to internal tensor_map keys) and collects its atom ranges and dims.
/// Returns (output_ranges, output_shapes, all_output_atom_ranges).
#[allow(clippy::type_complexity)]
pub(crate) fn build_output_ranges(
    graph: &NanoGraph<'static, SystemPool>,
    tensor_map: &HashMap<GlobalId, TensorAtomMapInfo>,
    output_ids: &[GlobalId],
    resolve_id: impl Fn(&GlobalId) -> GlobalId,
) -> (
    Vec<(GlobalId, Vec<AtomRange>)>,
    Vec<(GlobalId, Vec<u64>)>,
    Vec<AtomRange>,
) {
    let mut output_ranges: Vec<(GlobalId, Vec<AtomRange>)> = Vec::new();
    let mut output_shapes: Vec<(GlobalId, Vec<u64>)> = Vec::new();
    let mut all_output_atom_ranges: Vec<AtomRange> = Vec::new();

    for ext_id in output_ids {
        let internal_id = resolve_id(ext_id);
        if let Some(tami) = tensor_map.get(&internal_id) {
            let ranges = tami.atom_ranges(graph);
            all_output_atom_ranges.extend(ranges.iter().cloned());
            output_shapes.push((*ext_id, tami.known_dims.clone()));
            output_ranges.push((*ext_id, ranges));
        }
    }

    (output_ranges, output_shapes, all_output_atom_ranges)
}

/// Partition and compile a NanoGraph into an ExecutablePlan.
///
/// `options`: selects between alternative partitioner / codegen
/// implementations exposed via `CompileOptions`.
///
/// `provenance`: optional group provenance for building a plan summary.
///
/// Returns (ExecutablePlan, PlanSummary, compile_error_count).
pub(crate) fn compile_nano_graph(
    graph: &NanoGraph<'static, SystemPool>,
    all_output_atom_ranges: &[AtomRange],
    options: &CompileOptions,
    provenance: Option<&report::GroupProvenance>,
    obs: &mut dyn CompiledEvalObserver,
) -> Result<(ExecutablePlan, PlanSummary, usize), String> {
    // Partition the NanoGraph — dispatch on the selected partitioner.
    let t0 = Instant::now();
    let phases = match &options.partitioner {
        PartitionerKind::Trivial => {
            // 1 phase, 1 span containing the whole graph. Debug baseline
            // that isolates I/O / codegen issues from partitioner issues.
            use crate::compiler::attempts::v14::types::{Phase, Span};
            let span_inputs: Vec<AtomRange> = graph
                .input_tensors()
                .iter()
                .map(|it| AtomRange {
                    base: it.base_id,
                    count: it.count,
                    dtype: it.dtype,
                })
                .collect();
            vec![Phase {
                spans: vec![Span {
                    graph: graph.clone(),
                    inputs: span_inputs,
                    outputs: all_output_atom_ranges.to_vec(),
                }],
            }]
        }
        PartitionerKind::LaneSplit { num_lanes } => partitioner_m::plan(
            graph,
            (*num_lanes).max(1),
            graph.input_tensors(),
            all_output_atom_ranges,
        ),
    };
    obs.on_milestone("compiled.partition", None, t0, Instant::now());

    // Extract plan summary before compilation consumes the phases.
    let empty_prov = Vec::new();
    let prov = provenance.unwrap_or(&empty_prov);
    let plan_summary = report::summarize_plan(&phases, prov, graph);

    // Validate partitioned spans: check nanograph integrity and cross-lane deps.
    {
        let mut span_errors = Vec::new();
        let mut cross_lane_violations = Vec::new();
        for (pi, phase) in phases.iter().enumerate() {
            // Validate each span's nanograph.
            for (li, span) in phase.spans.iter().enumerate() {
                for err in span.graph.validate() {
                    span_errors.push(format!("Phase {} Lane {}: {}", pi, li, err));
                }
            }
            // Check no span reads atoms produced by another span in the same phase.
            let span_produces: Vec<Vec<(u64, u64)>> = phase
                .spans
                .iter()
                .map(|s| {
                    s.graph
                        .groups()
                        .iter()
                        .map(|g| (g.base_id.0, g.base_id.0 + g.count))
                        .collect()
                })
                .collect();
            for (li, span) in phase.spans.iter().enumerate() {
                for input in &span.inputs {
                    let inp_lo = input.base.0;
                    let inp_hi = inp_lo + input.count;
                    for (other_li, other_ranges) in span_produces.iter().enumerate() {
                        if other_li == li {
                            continue;
                        }
                        for &(prod_lo, prod_hi) in other_ranges {
                            if inp_lo < prod_hi && prod_lo < inp_hi {
                                cross_lane_violations.push(format!(
                                    "Phase {} lane {} input [{}, {}) overlaps lane {} produced [{}, {})",
                                    pi, li, inp_lo, inp_hi, other_li, prod_lo, prod_hi,
                                ));
                            }
                        }
                    }
                }
            }
        }
        if !span_errors.is_empty() {
            eprintln!(
                "[compiled_eval] WARNING: {} span validation error(s):",
                span_errors.len()
            );
            for e in span_errors.iter().take(20) {
                eprintln!("  {e}");
            }
        }
        if !cross_lane_violations.is_empty() {
            eprintln!(
                "[compiled_eval] WARNING: {} cross-lane violation(s):",
                cross_lane_violations.len()
            );
            for v in cross_lane_violations.iter().take(20) {
                eprintln!("  {v}");
            }
        }
    }

    // Compile all spans — dispatch on the selected codegen.
    //
    // For Jit, opaque-op spans and any compile failures fall back to pool_eval
    // per-span. For PoolEval, every span uses pool_eval unconditionally.
    let t0 = Instant::now();
    let mut plan_builder = ExecutablePlanBuilder::new();
    plan_builder.pin_outputs(all_output_atom_ranges);
    let mut compile_errors = 0usize;
    let force_pool_eval = matches!(options.codegen, CodegenKind::PoolEval);

    for (pi, phase) in phases.iter().enumerate() {
        let mut lanes = Vec::new();
        for (si, span) in phase.spans.iter().enumerate() {
            let has_opaque = span
                .graph
                .groups()
                .iter()
                .any(|g| matches!(g.op, crate::nano_graph::ops::ScalarOp::OpaqueOutput { .. }));

            if force_pool_eval
                || has_opaque
                || (span.graph.groups().is_empty() && !span.graph.opaque_ops().is_empty())
            {
                // Pool_eval path: forced by options, or required by opaque ops.
                lanes.push((
                    Box::new(PoolEvalSpan::new(
                        span.graph.clone(),
                        span.inputs.clone(),
                        span.outputs.clone(),
                    )) as Box<dyn CompiledSpanFn>,
                    span.inputs.clone(),
                    span.outputs.clone(),
                ));
            } else {
                match JitCompiledSpan::compile(&span.graph, &span.outputs) {
                    Ok(jit_span) => {
                        lanes.push((
                            Box::new(jit_span) as Box<dyn CompiledSpanFn>,
                            span.inputs.clone(),
                            span.outputs.clone(),
                        ));
                    }
                    Err(e) => {
                        compile_errors += 1;
                        if compile_errors <= 5 {
                            eprintln!(
                                "[compiled_eval] compile error phase {} span {}: {}",
                                pi, si, e
                            );
                        }
                        // Fall back to pool_eval for this span.
                        lanes.push((
                            Box::new(PoolEvalSpan::new(
                                span.graph.clone(),
                                span.inputs.clone(),
                                span.outputs.clone(),
                            )) as Box<dyn CompiledSpanFn>,
                            span.inputs.clone(),
                            span.outputs.clone(),
                        ));
                    }
                }
            }
        }
        plan_builder.add_phase(lanes);
    }

    let executable_plan = plan_builder.build();
    obs.on_milestone("compiled.compile_phases", None, t0, Instant::now());

    Ok((executable_plan, plan_summary, compile_errors))
}

/// Relayout a tensor view to match a TAMI's atom ordering and return as a
/// flat 1D Cow.
///
/// On the matched-layout fast path (the common case for unrearranged
/// weights), the returned Cow is `Borrowed` — no allocation, no copy. The
/// borrow lifetime `'a` matches the input view's data lifetime, so callers
/// must keep the source view alive for at least as long as the Cow.
///
/// On the rearrange path, the returned Cow is `Owned` and carries a fresh
/// pool-allocated buffer with the rearranged bytes.
pub(crate) fn relayout_to_flat<'a, 'p, P: Pool + 'p>(
    tami: &TensorAtomMapInfo,
    view: &NumericTensorView<'a, DynRank>,
    pool: &'p P,
) -> Result<NumericTensorCOW<'a, 'p, DynRank, P>, String> {
    let element_bits = tami.dtype.total_bits() as u64;
    let strides_bits: Vec<u64> = tami
        .known_strides
        .iter()
        .map(|&s| s * element_bits)
        .collect();
    let target = TensorLayout::<DynRank>::ElementStrided {
        shape: tami.known_dims.clone(),
        dtype: tami.dtype,
        strides: strides_bits,
        offset_bits: 0,
    };

    // `relayout` returns Borrowed when the source view's bytes are already
    // in target order — that's the no-rearrange fast path. Either way, the
    // result is wrapped as flat 1D for the executor's atom-keyed store.
    let cow = view
        .relayout(target, pool)
        .map_err(|e| format!("relayout failed: {e}"))?;

    let flat_layout = TensorLayout::<DynRank>::row_major(vec![tami.count], tami.dtype);
    match cow {
        NumericTensorCOW::Borrowed(borrow_view) => {
            // Zero-copy: reinterpret the borrow's data as flat 1D and pass
            // it through. The byte slice is unchanged; only the layout
            // descriptor is replaced.
            Ok(NumericTensorCOW::Borrowed(NumericTensorView::new(
                borrow_view.data(),
                flat_layout,
            )))
        }
        NumericTensorCOW::Owned(owned) => {
            // Already owned — reinterpret as flat 1D (zero-cost layout change).
            Ok(NumericTensorCOW::Owned(owned.into_layout(flat_layout)))
        }
    }
}

/// Prepare initial inputs for the executor's PhaseStore.
///
/// For each (ext_id, view) pair, resolves the external ID to a TAMI via
/// `input_map` + `tensor_map`, relayouts the view to match TAMI strides,
/// and returns flat 1D Cow tensors keyed by base AtomId. The Cow propagates
/// the borrow when the source view's layout already matches the target,
/// avoiding the wasted alloc+copy that happened when this returned owned.
#[allow(clippy::type_complexity)]
pub(crate) fn prepare_compiled_inputs<'a, 'p, P: Pool + 'p>(
    all_views: &[(GlobalId, &NumericTensorView<'a, DynRank>)],
    input_map: &HashMap<GlobalId, GlobalId>,
    tensor_map: &HashMap<GlobalId, TensorAtomMapInfo>,
    pool: &'p P,
) -> Result<Vec<(AtomId, NumericTensorCOW<'a, 'p, DynRank, P>)>, String> {
    let mut initial_inputs: Vec<(AtomId, NumericTensorCOW<'a, 'p, DynRank, P>)> = Vec::new();

    for &(ext_id, view) in all_views {
        let internal_id = input_map.get(&ext_id).copied().unwrap_or(ext_id);
        let Some(tami) = tensor_map.get(&internal_id) else {
            continue;
        };

        if tami.segments.is_empty() {
            // Simple tensor: relayout to TAMI strides, store as flat 1D.
            let flat = relayout_to_flat(tami, view, pool)?;
            initial_inputs.push((tami.base_id, flat));
        } else {
            // Segmented tensor: process each segment separately.
            for &(concat_dim, start, size, seg_base, ref seg_strides) in &tami.segments {
                let mut seg_dims = tami.known_dims.clone();
                seg_dims[concat_dim] = size;
                let seg_count: u64 = seg_dims.iter().product();

                // Slice the view along the concat dimension.
                let ranges: Vec<(u64, u64)> = tami
                    .known_dims
                    .iter()
                    .enumerate()
                    .map(|(d, &dim)| {
                        if d == concat_dim {
                            (start, start + size)
                        } else {
                            (0, dim)
                        }
                    })
                    .collect();
                let sliced = view
                    .slice(&ranges)
                    .map_err(|e| format!("slice failed for segment: {e:?}"))?;

                let seg_tami = TensorAtomMapInfo {
                    base_id: seg_base,
                    count: seg_count,
                    dtype: tami.dtype,
                    sym_dims: vec![],
                    known_strides: seg_strides.clone(),
                    known_dims: seg_dims,
                    segments: vec![],
                };
                // `sliced` is a stack-local NumericTensorView whose data
                // borrow lifetime matches `view`'s `'a`. relayout_to_flat
                // either returns Borrowed (with that same `'a` data ptr) or
                // Owned (allocated in pool). For the segmented Borrowed
                // case the resulting Cow's `'a` is `view`'s `'a` — sliced
                // itself is dropped but its underlying bytes belong to `view`.
                let flat = relayout_to_flat(&seg_tami, &sliced, pool)?;
                initial_inputs.push((seg_base, flat));
            }
        }
    }

    Ok(initial_inputs)
}

/// Extract output tensors from the PhaseStore after execution.
pub(crate) fn extract_outputs<'a, 'p, P: Pool + 'p>(
    output_ranges: &[(GlobalId, Vec<AtomRange>)],
    output_shapes: &[(GlobalId, Vec<u64>)],
    store: &PhaseStore<'a, 'p, P>,
    pool: &'p P,
) -> Result<HashMap<GlobalId, NumericTensor<'p, DynRank, P>>, String> {
    let mut results: HashMap<GlobalId, NumericTensor<'p, DynRank, P>> = HashMap::new();

    for ((ext_id, ranges), (_, shape)) in output_ranges.iter().zip(output_shapes.iter()) {
        let dtype = ranges
            .first()
            .map(|r| r.dtype)
            .unwrap_or(crate::numeric_dtype::NumericDType::F32);
        let elem_bytes = dtype.bytes_per_element();

        // Allocate output tensor with the proper shape.
        // Use byte-aligned strides (matching JIT output layout) for sub-byte
        // dtypes like BOOL, then reshape to the correct dimensions.
        let bpe_bits = dtype.bytes_per_element() as u64 * 8;
        let dims = shape.as_slice();
        let mut strides = vec![0u64; dims.len()];
        if !dims.is_empty() {
            strides[dims.len() - 1] = bpe_bits;
            for i in (0..dims.len() - 1).rev() {
                strides[i] = strides[i + 1] * dims[i + 1];
            }
        }
        let layout = TensorLayout::<DynRank>::ElementStrided {
            shape: shape.clone(),
            dtype,
            strides,
            offset_bits: 0,
        };
        let buf = pool
            .allocate(layout.buffer_size_bytes())
            .map_err(|e| format!("output alloc failed: {e}"))?;
        let mut tensor = NumericTensor::from_parts(buf, layout);

        // Gather data from the store for each atom range.
        let mut write_offset = 0usize;
        for range in ranges {
            let slices = store.gather(range.base, range.count);
            for slice in &slices {
                let copy_bytes = slice.count as usize * elem_bytes;
                let dst = tensor.buffer_mut();
                if write_offset + copy_bytes <= dst.len() && copy_bytes <= slice.data.len() {
                    dst[write_offset..write_offset + copy_bytes]
                        .copy_from_slice(&slice.data[..copy_bytes]);
                }
                write_offset += copy_bytes;
            }
        }

        results.insert(*ext_id, tensor);
    }

    Ok(results)
}

// ===========================================================================
// Supergraph-specific wrappers
// ===========================================================================

/// Compile a lowered model into an ExecutablePlan.
///
/// Partitions the NanoGraph, JIT-compiles each span, and builds the
/// executable plan. Returns None if compilation fails.
pub fn compile_lowered_model(
    cached: &CachedLoweredModel,
    sym_graph: &crate::symbolic_graph::SymbolicGraph,
    options: &CompileOptions,
    obs: &mut dyn CompiledEvalObserver,
) -> Option<CachedCompiledPlan> {
    let graph = &cached.graph;

    // Build output atom ranges from the symbolic graph's ordered outputs.
    let reverse_output: HashMap<GlobalId, GlobalId> = cached
        .output_map
        .iter()
        .map(|(&internal, &external)| (external, internal))
        .collect();

    let ordered_outputs = sym_graph.get_ordered_outputs();
    let (output_ranges, output_shapes, all_output_atom_ranges) =
        build_output_ranges(graph, &cached.tensor_map, ordered_outputs, |ext_id| {
            reverse_output
                .get(ext_id)
                .copied()
                .or_else(|| cached.sym_to_internal.get(ext_id).copied())
                .unwrap_or(*ext_id)
        });
    if output_ranges.len() < ordered_outputs.len() {
        eprintln!(
            "[compiled_eval] WARNING: {} of {} outputs failed to resolve atom ranges",
            ordered_outputs.len() - output_ranges.len(),
            ordered_outputs.len(),
        );
    }

    let provenance = if cached.group_provenance.is_empty() {
        None
    } else {
        Some(&cached.group_provenance)
    };
    let (executable_plan, plan_summary, _compile_errors) =
        compile_nano_graph(graph, &all_output_atom_ranges, options, provenance, obs).ok()?;

    Some(CachedCompiledPlan {
        info_inputs_hash: cached.info_inputs_hash,
        executable_plan,
        output_ranges,
        output_shapes,
        plan_summary,
    })
}

/// Execute a compiled plan with the given inputs.
///
/// Converts input tensors to flat (AtomId, NumericTensor) pairs for the
/// executor's PhaseStore, runs the plan, then extracts output tensors.
pub fn execute_compiled<'p, P: Pool + 'p>(
    compiled: &CachedCompiledPlan,
    cached_lower: &CachedLoweredModel,
    sym_graph: &crate::symbolic_graph::SymbolicGraph,
    tensor_store: &TensorStore,
    user_input_views: &HashMap<GlobalId, NumericTensorView<'_, DynRank>>,
    pool: &'p P,
    obs: &mut dyn CompiledEvalObserver,
) -> Result<HashMap<GlobalId, NumericTensor<'p, DynRank, P>>, super::SuperGraphError> {
    // --- Weight load: resolve every stored tensor for inputs not provided by
    //     the user this iter (constants + weights). Steady-state hot path. ---
    let t_load = Instant::now();
    let stored_ids: Vec<GlobalId> = cached_lower
        .weight_input_ext_ids
        .iter()
        .chain(
            cached_lower
                .user_input_ext_ids
                .iter()
                .filter(|id| !user_input_views.contains_key(id)),
        )
        .copied()
        .collect();

    let mut weight_tensors: Vec<(GlobalId, NumericTensor<'_, DynRank, SystemPool>)> =
        Vec::with_capacity(stored_ids.len());
    for &ext_id in &stored_ids {
        let tensor_meta = sym_graph.get_tensor_info(ext_id).ok_or_else(|| {
            super::SuperGraphError::InvalidGraph(format!(
                "compiled_eval: missing tensor info for weight {ext_id:?}"
            ))
        })?;
        let stored_ref = match &tensor_meta.tensor_type {
            TensorType::Constant(s) | TensorType::Input(Some(s)) => s,
            _ => {
                return Err(super::SuperGraphError::InvalidGraph(format!(
                    "compiled_eval: weight {ext_id:?} is not a constant/initialized input"
                )));
            }
        };
        let tensor =
            lowered_eval::resolve_stored_tensor(stored_ref, tensor_store).ok_or_else(|| {
                super::SuperGraphError::InvalidGraph(format!(
                    "compiled_eval: failed to load stored tensor {ext_id:?}"
                ))
            })?;
        weight_tensors.push((ext_id, tensor));
    }

    let weight_views: Vec<(GlobalId, NumericTensorView<'_, DynRank>)> = weight_tensors
        .iter()
        .map(|(id, t)| (*id, t.view()))
        .collect();
    obs.on_milestone("compiled.exec.weight_load", None, t_load, Instant::now());

    // --- Input prep: relayout each input/weight view to TAMI strides. ---
    let t_prep = Instant::now();
    let all_views: Vec<(GlobalId, &NumericTensorView<'_, DynRank>)> = user_input_views
        .iter()
        .map(|(&id, v)| (id, v))
        .chain(weight_views.iter().map(|(id, v)| (*id, v)))
        .collect();

    let initial_inputs = prepare_compiled_inputs(
        &all_views,
        &cached_lower.input_map,
        &cached_lower.tensor_map,
        pool,
    )
    .map_err(|e| super::SuperGraphError::InvalidGraph(format!("compiled_eval: {e}")))?;
    obs.on_milestone("compiled.exec.input_prep", None, t_prep, Instant::now());

    // --- JIT execute ---
    let t_jit = Instant::now();
    let store = compiled.executable_plan.execute_timed(initial_inputs, pool);
    obs.on_milestone("compiled.exec.jit", None, t_jit, Instant::now());

    // --- Extract outputs ---
    let t_extract = Instant::now();
    let result = extract_outputs(
        &compiled.output_ranges,
        &compiled.output_shapes,
        &store,
        pool,
    )
    .map_err(|e| super::SuperGraphError::InvalidGraph(format!("compiled_eval: {e}")));
    obs.on_milestone(
        "compiled.exec.output_extract",
        None,
        t_extract,
        Instant::now(),
    );
    result
}
