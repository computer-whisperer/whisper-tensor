//! Compiled evaluation path for ModelExecution nodes.
//!
//! Extends the lowered eval path: after lowering SymbolicGraph → NanoGraph,
//! this module partitions and JIT-compiles the NanoGraph into an ExecutablePlan
//! that runs via the v14 executor. The compiled plan is cached for reuse.
//!
//! Requires the `x86_compile` feature.
//!
//! The core compile+execute logic lives in standalone `pub(crate)` functions
//! (compile_nano_graph, prepare_compiled_inputs, relayout_to_flat, extract_outputs)
//! so that test_set can reuse them without supergraph dependencies.

use std::collections::HashMap;
use std::env;
use std::time::Instant;

use crate::compiler::attempts::v14::executor::{
    CompiledSpanFn, ExecutablePlan, ExecutablePlanBuilder, PoolEvalSpan,
};
use crate::compiler::attempts::v14::partitioner_m;
use crate::compiler::attempts::v14::partitioner_n;
use crate::compiler::attempts::v14::placer::{AtomPlacementMap, run_placer};
use crate::compiler::attempts::v14::report::{self, PlanSummary};
use crate::compiler::{CodegenKind, CompileOptions, PartitionerKind};
use crate::graph::GlobalId;
use crate::nano_graph::AtomId;
use crate::nano_graph::lower::{DimKind, TensorAtomMapInfo};
use crate::nano_graph::pattern::{AtomRange, NanoGraph};
use crate::numeric_tensor::{NumericTensor, NumericTensorCOW, NumericTensorView, TensorLayout};
use crate::pool::{Pool, SystemPool};
use crate::super_graph::cache::LoadedTensorCache;
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
            output_shapes.push((*ext_id, tami.known_dims()));
            output_ranges.push((*ext_id, ranges));
        }
    }

    (output_ranges, output_shapes, all_output_atom_ranges)
}

fn resolve_partitioner_override(requested: &PartitionerKind) -> PartitionerKind {
    let override_kind = match env::var("WT_PARTITIONER_KIND") {
        Ok(v) => v.to_ascii_lowercase(),
        Err(_) => return requested.clone(),
    };

    let requested_lanes = match requested {
        PartitionerKind::LaneSplit { num_lanes } | PartitionerKind::LaneSplitV2 { num_lanes } => {
            (*num_lanes).max(1)
        }
        PartitionerKind::Trivial => 1,
    };
    let override_lanes = env::var("WT_PARTITIONER_LANES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|n| n.max(1))
        .unwrap_or(requested_lanes);

    match override_kind.as_str() {
        "trivial" => PartitionerKind::Trivial,
        "m" | "lanesplit" | "lane_split" => PartitionerKind::LaneSplit {
            num_lanes: override_lanes,
        },
        "n" | "lanesplitv2" | "lane_split_v2" => PartitionerKind::LaneSplitV2 {
            num_lanes: override_lanes,
        },
        _ => {
            eprintln!(
                "[compiled_eval] ignoring unknown WT_PARTITIONER_KIND='{}' (expected trivial|m|n)",
                override_kind
            );
            requested.clone()
        }
    }
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
    let partitioner = resolve_partitioner_override(&options.partitioner);
    let phases = match &partitioner {
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
        PartitionerKind::LaneSplitV2 { num_lanes } => partitioner_n::plan(
            graph,
            (*num_lanes).max(1),
            graph.input_tensors(),
            all_output_atom_ranges,
        ),
    };
    obs.on_milestone("compiled.partition", None, t0, Instant::now());

    // Cross-span slab coalescing audit (gated by env var, see
    // MEMORY_PLACEMENT.md §"Implementation order" step 2).
    if std::env::var("WT_AUDIT_SLABS")
        .ok()
        .is_some_and(|v| v != "0")
    {
        let report = crate::compiler::attempts::v14::audit::audit_slab_coalescing(
            graph,
            &phases,
            all_output_atom_ranges,
        );
        report.print();
    }

    // Run the global memory placer. Its output drives both per-span
    // codegen (`SlotInfo::buffer_id` / `BufferBases`) and the executor
    // (`buffer_ptrs` layout).
    let placement =
        run_placer(graph, &phases, all_output_atom_ranges).map_err(|e| format!("placer: {e}"))?;

    if std::env::var("WT_PRINT_PLACEMENT")
        .ok()
        .is_some_and(|v| v != "0")
    {
        placement.print_summary();
    }

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

    x86_jit_stats::enable();

    // Use parallel span compilation unless force_pool_eval is set.
    let use_parallel = !force_pool_eval;

    for (pi, phase) in phases.iter().enumerate() {
        let lanes = if use_parallel {
            compile_phase_parallel(pi, &phase.spans, &placement, &mut compile_errors)
        } else {
            compile_phase_pool_eval(&phase.spans, &placement)
        };
        plan_builder.add_phase(lanes);
    }

    let executable_plan = plan_builder.build(placement, graph);
    obs.on_milestone("compiled.compile_phases", None, t0, Instant::now());

    x86_jit_stats::print_summary();

    Ok((executable_plan, plan_summary, compile_errors))
}

type LaneTuple = (Box<dyn CompiledSpanFn>, Vec<AtomRange>, Vec<AtomRange>);

/// Compile one phase's spans in parallel using rayon.
fn compile_phase_parallel(
    pi: usize,
    spans: &[crate::compiler::attempts::v14::types::Span],
    placement: &AtomPlacementMap,
    compile_errors: &mut usize,
) -> Vec<LaneTuple> {
    use rayon::prelude::*;

    let results: Vec<_> = spans
        .par_iter()
        .map(|span| {
            let has_opaque = span
                .graph
                .groups()
                .iter()
                .any(|g| matches!(g.op, crate::nano_graph::ops::ScalarOp::OpaqueOutput { .. }));

            if has_opaque || (span.graph.groups().is_empty() && !span.graph.opaque_ops().is_empty())
            {
                Ok(Box::new(PoolEvalSpan::new(
                    span.graph.clone(),
                    span.inputs.clone(),
                    span.outputs.clone(),
                    placement,
                )) as Box<dyn CompiledSpanFn>)
            } else {
                compile_one_span_native(&span.graph, &span.outputs, placement)
            }
        })
        .collect();

    results
        .into_iter()
        .enumerate()
        .map(|(si, result)| match result {
            Ok(boxed) => (boxed, spans[si].inputs.clone(), spans[si].outputs.clone()),
            Err(e) => {
                *compile_errors += 1;
                if *compile_errors <= 5 {
                    eprintln!(
                        "[compiled_eval] compile error phase {} span {}: {}",
                        pi, si, e
                    );
                }
                (
                    Box::new(PoolEvalSpan::new(
                        spans[si].graph.clone(),
                        spans[si].inputs.clone(),
                        spans[si].outputs.clone(),
                        placement,
                    )) as Box<dyn CompiledSpanFn>,
                    spans[si].inputs.clone(),
                    spans[si].outputs.clone(),
                )
            }
        })
        .collect()
}

/// Wrap every span in a PoolEvalSpan (used when force_pool_eval is set).
fn compile_phase_pool_eval(
    spans: &[crate::compiler::attempts::v14::types::Span],
    placement: &AtomPlacementMap,
) -> Vec<LaneTuple> {
    spans
        .iter()
        .map(|span| {
            (
                Box::new(PoolEvalSpan::new(
                    span.graph.clone(),
                    span.inputs.clone(),
                    span.outputs.clone(),
                    placement,
                )) as Box<dyn CompiledSpanFn>,
                span.inputs.clone(),
                span.outputs.clone(),
            )
        })
        .collect()
}

/// Compile a single span into a boxed `CompiledSpanFn` via x86_jit (dynasm).
///
/// On compile error, the error propagates and the caller routes the span
/// through `PoolEvalSpan` instead.
fn compile_one_span_native(
    graph: &NanoGraph<'static, SystemPool>,
    outputs: &[AtomRange],
    placement: &AtomPlacementMap,
) -> Result<Box<dyn CompiledSpanFn>, String> {
    match crate::compiler::attempts::v14::x86_jit::X86JitSpan::compile(graph, outputs, placement) {
        Ok(s) => {
            x86_jit_stats::record_accept();
            Ok(Box::new(s) as Box<dyn CompiledSpanFn>)
        }
        Err(e) => {
            x86_jit_stats::record_fallback(&e);
            Err(e)
        }
    }
}

/// Per-process counters of which backend handled each span. Reset per
/// `compile_nano_graph` call (the printer prints + resets at the end of the
/// model compile).
mod x86_jit_stats {
    use std::collections::HashMap;
    use std::sync::Mutex;

    static STATE: Mutex<Option<State>> = Mutex::new(None);

    struct State {
        accepted: usize,
        fallback: usize,
        /// First-line of the err message → count, for grouping common
        /// reject reasons.
        reasons: HashMap<String, usize>,
    }

    pub(super) fn enable() {
        let mut g = STATE.lock().unwrap();
        *g = Some(State {
            accepted: 0,
            fallback: 0,
            reasons: HashMap::new(),
        });
    }

    pub(super) fn record_accept() {
        if let Some(st) = STATE.lock().unwrap().as_mut() {
            st.accepted += 1;
        }
    }

    pub(super) fn record_fallback(err: &str) {
        if let Some(st) = STATE.lock().unwrap().as_mut() {
            st.fallback += 1;
            // Strip the leading "x86_jit: group N " prefix so spans with
            // different group indices but the same reason group together.
            let key = err
                .strip_prefix("x86_jit: ")
                .unwrap_or(err)
                .splitn(3, ' ')
                .nth(2)
                .map(|s| s.to_string())
                .unwrap_or_else(|| err.to_string());
            *st.reasons.entry(key).or_insert(0) += 1;
        }
    }

    pub(super) fn print_summary() {
        let mut g = STATE.lock().unwrap();
        if let Some(st) = g.take() {
            let total = st.accepted + st.fallback;
            if total == 0 {
                return;
            }
            eprintln!();
            eprintln!(
                "=== X86_JIT span coverage: {}/{} ({:.1}%) accepted, {} fallback ===",
                st.accepted,
                total,
                100.0 * st.accepted as f64 / total as f64,
                st.fallback,
            );
            let mut reasons: Vec<_> = st.reasons.into_iter().collect();
            reasons.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
            for (reason, count) in reasons.into_iter().take(10) {
                eprintln!("  {count:>6}  {reason}");
            }
        }
    }
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
    let known_strides_v = tami.known_strides();
    let strides_bits: Vec<u64> = known_strides_v
        .iter()
        .map(|&s| s * element_bits)
        .collect();
    let target = TensorLayout::<DynRank>::ElementStrided {
        shape: tami.known_dims(),
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
            let known_dims_v = tami.known_dims();
            for seg in &tami.segments {
                let concat_dim = seg.concat_dim;
                let start = seg.start;
                let size = seg.size;
                let seg_base = seg.base_id;
                let seg_strides = &seg.known_strides;

                let mut seg_dim_sizes = known_dims_v.clone();
                seg_dim_sizes[concat_dim] = size;
                let seg_count: u64 = seg_dim_sizes.iter().product();

                // Slice the view along the concat dimension.
                let ranges: Vec<(u64, u64)> = known_dims_v
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

                // Build DimKind entries with segment strides.
                let seg_dim_kinds: Vec<DimKind> = seg_dim_sizes.iter().zip(seg_strides.iter())
                    .map(|(&sz, &st)| DimKind::Known { size: sz, stride: st })
                    .collect();
                let seg_tami = TensorAtomMapInfo {
                    base_id: seg_base,
                    count: seg_count,
                    dtype: tami.dtype,
                    dims: seg_dim_kinds,
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

/// Re-assemble the executor's flat output Vec into per-GlobalId tensors.
///
/// The executor returns `Vec<NumericTensor>` in the same order as the
/// flat `all_output_atom_ranges` list handed to `compile_nano_graph`
/// (one entry per `AtomRange`). A single GlobalId may correspond to
/// multiple atom ranges (Pad-style lowering splits an output into
/// `zeros | data | zeros`); we concatenate those into one tensor with
/// the declared shape.
///
/// The single-range fast path re-wraps the executor's allocation
/// with the target layout directly — no copy.
pub(crate) fn extract_outputs<'p, P: Pool + 'p>(
    output_ranges: &[(GlobalId, Vec<AtomRange>)],
    output_shapes: &[(GlobalId, Vec<u64>)],
    executor_outputs: Vec<NumericTensor<'p, DynRank, P>>,
    pool: &'p P,
) -> Result<HashMap<GlobalId, NumericTensor<'p, DynRank, P>>, String> {
    let mut results: HashMap<GlobalId, NumericTensor<'p, DynRank, P>> = HashMap::new();
    let mut iter = executor_outputs.into_iter();

    for ((ext_id, ranges), (_, shape)) in output_ranges.iter().zip(output_shapes.iter()) {
        let dtype = ranges
            .first()
            .map(|r| r.dtype)
            .unwrap_or(crate::numeric_dtype::NumericDType::F32);
        let elem_bytes = dtype.bytes_per_element();

        // Single-range fast path: take the executor's tensor and
        // reshape in place (no copy).
        if ranges.len() == 1 {
            let flat = iter.next().ok_or_else(|| {
                format!("extract_outputs: executor ran out of tensors for {ext_id:?}")
            })?;
            let bpe_bits = elem_bytes as u64 * 8;
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
            results.insert(*ext_id, flat.into_layout(layout));
            continue;
        }

        // Multi-range: concatenate in declaration order into a fresh
        // allocation.
        let bpe_bits = elem_bytes as u64 * 8;
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

        let mut write_offset = 0usize;
        for range in ranges {
            let src = iter.next().ok_or_else(|| {
                format!(
                    "extract_outputs: executor ran out of tensors for {ext_id:?} \
                     range base={}",
                    range.base.0
                )
            })?;
            let src_bytes = src.buffer();
            let copy_bytes = (range.count as usize) * elem_bytes;
            let dst = tensor.buffer_mut();
            if write_offset + copy_bytes <= dst.len() && copy_bytes <= src_bytes.len() {
                dst[write_offset..write_offset + copy_bytes]
                    .copy_from_slice(&src_bytes[..copy_bytes]);
            }
            write_offset += copy_bytes;
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
        build_output_ranges(graph, &cached.graph.tensor_map, ordered_outputs, |ext_id| {
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
#[allow(clippy::too_many_arguments)]
pub fn execute_compiled<'p, P: Pool + 'p>(
    compiled: &CachedCompiledPlan,
    cached_lower: &CachedLoweredModel,
    sym_graph: &crate::symbolic_graph::SymbolicGraph,
    tensor_store: &TensorStore,
    user_input_views: &HashMap<GlobalId, NumericTensorView<'_, DynRank>>,
    pool: &'p P,
    loaded_tensor_cache: Option<&mut LoadedTensorCache>,
    obs: &mut dyn CompiledEvalObserver,
) -> Result<HashMap<GlobalId, NumericTensor<'p, DynRank, P>>, super::SuperGraphError> {
    // --- Weight load: resolve every stored tensor for inputs not provided by
    //     the user this iter (constants + weights). Steady-state hot path.
    //     Goes through `prepare_weights` so cache hits skip the disk and the
    //     cross-pool copy. ---
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

    let prepared = lowered_eval::prepare_weights(
        &stored_ids,
        sym_graph,
        tensor_store,
        pool,
        loaded_tensor_cache,
    )?;
    let weight_views = prepared.views();
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
        &cached_lower.graph.tensor_map,
        pool,
    )
    .map_err(|e| super::SuperGraphError::InvalidGraph(format!("compiled_eval: {e}")))?;
    obs.on_milestone("compiled.exec.input_prep", None, t_prep, Instant::now());

    // --- Build input_ptrs array ---
    //
    // The executor addresses inputs through `buffer_ptrs[buffer_id]`.
    // For each (AtomId, NumericTensorCOW) initial input, look up the
    // placer's buffer_id and drop the tensor's byte pointer into that
    // slot. The Cow lives in `initial_inputs` across the execute call
    // so the pointers stay valid.
    //
    // Some lowering paths turn concrete input tensors into literal
    // groups rather than input tensors; the placer has no entry for
    // them (their data is in the literal buffer, not an input slot),
    // so we just skip them here.
    let placement = compiled.executable_plan.placement();
    let ptr_array_len = (placement.scratch_buffer_id as usize) + 1;
    let mut input_ptrs: Vec<*mut u8> = vec![std::ptr::null_mut(); ptr_array_len];
    for (atom_id, cow) in &initial_inputs {
        if let Some((buf_id, _)) = placement.byte_offset_of(*atom_id) {
            let ptr = cow.buffer().as_ptr() as *mut u8;
            if (buf_id.0 as usize) < input_ptrs.len() {
                input_ptrs[buf_id.0 as usize] = ptr;
            }
        }
    }

    // --- JIT execute ---
    let t_jit = Instant::now();
    let executor_outputs = compiled.executable_plan.execute_timed(&input_ptrs, pool);
    // Keep initial_inputs alive until after execute returns so the
    // input byte pointers remain valid. Explicit drop for clarity.
    drop(initial_inputs);
    obs.on_milestone("compiled.exec.jit", None, t_jit, Instant::now());

    // --- Extract outputs ---
    let t_extract = Instant::now();
    let result = extract_outputs(
        &compiled.output_ranges,
        &compiled.output_shapes,
        executor_outputs,
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
