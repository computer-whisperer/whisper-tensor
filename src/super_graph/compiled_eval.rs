//! Compiled evaluation path for ModelExecution nodes.
//!
//! Extends the lowered eval path: after lowering SymbolicGraph → NanoGraph,
//! this module partitions and JIT-compiles the NanoGraph into an ExecutablePlan
//! that runs via the v14 executor. The compiled plan is cached for reuse.
//!
//! Requires the `cranelift` feature.

use std::collections::HashMap;

use crate::compiler::attempts::v14::codegen::JitCompiledSpan;
use crate::compiler::attempts::v14::executor::{
    CompiledSpanFn, ExecutablePlan, ExecutablePlanBuilder, PhaseStore,
};
use crate::compiler::attempts::v14::partitioner_m;
use crate::graph::GlobalId;
use crate::nano_graph::lower::TensorAtomMapInfo;
use crate::nano_graph::pattern::{AtomRange, NanoGraph};
use crate::nano_graph::AtomId;
use crate::numeric_tensor::{NumericTensor, NumericTensorView, TensorLayout};
use crate::pool::{Pool, SystemPool};
use crate::symbolic_graph::tensor_store::TensorStore;
use crate::symbolic_graph::TensorType;
use crate::tensor_rank::DynRank;

use super::lowered_eval::{self, CachedLoweredModel};

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
}

/// Compile a lowered model into an ExecutablePlan.
///
/// Partitions the NanoGraph, JIT-compiles each span, and builds the
/// executable plan. Returns None if compilation fails.
pub fn compile_lowered_model(
    cached: &CachedLoweredModel,
    sym_graph: &crate::symbolic_graph::SymbolicGraph,
) -> Option<CachedCompiledPlan> {
    let graph = &cached.graph;

    // Build output atom ranges from the symbolic graph's ordered outputs.
    let reverse_output: HashMap<GlobalId, GlobalId> = cached
        .output_map
        .iter()
        .map(|(&internal, &external)| (external, internal))
        .collect();

    let mut output_ranges: Vec<(GlobalId, Vec<AtomRange>)> = Vec::new();
    let mut output_shapes: Vec<(GlobalId, Vec<u64>)> = Vec::new();
    let mut all_output_atom_ranges: Vec<AtomRange> = Vec::new();

    for &ext_id in sym_graph.get_ordered_outputs() {
        let internal_id = reverse_output
            .get(&ext_id)
            .copied()
            .or_else(|| cached.sym_to_internal.get(&ext_id).copied())
            .unwrap_or(ext_id);
        if let Some(tami) = cached.tensor_map.get(&internal_id) {
            let ranges = tami.atom_ranges(graph);
            all_output_atom_ranges.extend(ranges.iter().cloned());
            output_shapes.push((ext_id, tami.known_dims.clone()));
            output_ranges.push((ext_id, ranges));
        }
    }

    // Partition the NanoGraph.
    let t0 = std::time::Instant::now();
    let phases = partitioner_m::plan(
        graph,
        8, // lanes
        graph.input_tensors(),
        &all_output_atom_ranges,
    );
    eprintln!(
        "[compiled_eval] partitioned in {:.0}ms ({} phases)",
        t0.elapsed().as_secs_f64() * 1e3,
        phases.len(),
    );

    // Compile all spans via JIT.
    let t0 = std::time::Instant::now();
    let mut plan_builder = ExecutablePlanBuilder::new();
    let mut compile_errors = 0usize;

    for (pi, phase) in phases.iter().enumerate() {
        let mut lanes = Vec::new();
        for (si, span) in phase.spans.iter().enumerate() {
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
                        eprintln!("[compiled_eval] compile error phase {} span {}: {}", pi, si, e);
                    }
                    // Insert a no-op span so phase lane counts stay consistent.
                    let noop = JitCompiledSpan::compile(
                        &NanoGraph::new(),
                        &[],
                    )
                    .expect("empty span should compile");
                    lanes.push((
                        Box::new(noop) as Box<dyn CompiledSpanFn>,
                        vec![],
                        vec![],
                    ));
                }
            }
        }
        plan_builder.add_phase(lanes);
    }

    let executable_plan = plan_builder.build();
    let dt = t0.elapsed();
    eprintln!(
        "[compiled_eval] compiled {} phases in {:.1}s ({} errors)",
        executable_plan.num_phases(),
        dt.as_secs_f64(),
        compile_errors,
    );

    Some(CachedCompiledPlan {
        info_inputs_hash: cached.info_inputs_hash,
        executable_plan,
        output_ranges,
        output_shapes,
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
) -> Result<HashMap<GlobalId, NumericTensor<'p, DynRank, P>>, super::SuperGraphError> {
    // --- Build initial inputs for the PhaseStore ---
    //
    // For each input tensor (user inputs + weights), relayout the view to
    // match the TAMI's expected atom ordering, then store as a flat 1D
    // tensor keyed by its base AtomId.

    let mut initial_inputs: Vec<(AtomId, NumericTensor<'p, DynRank, P>)> = Vec::new();

    // Collect all input views: user inputs + weights from tensor store.
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

    // Load weight tensors from store.
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
        let tensor = lowered_eval::resolve_stored_tensor(stored_ref, tensor_store).ok_or_else(|| {
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

    // Process all inputs: relayout to TAMI atom order and store as flat tensors.
    let all_views: Vec<(GlobalId, &NumericTensorView<'_, DynRank>)> = user_input_views
        .iter()
        .map(|(&id, v)| (id, v))
        .chain(weight_views.iter().map(|(id, v)| (*id, v)))
        .collect();

    for &(ext_id, view) in &all_views {
        let internal_id = cached_lower.input_map.get(&ext_id).copied().unwrap_or(ext_id);
        let Some(tami) = cached_lower.tensor_map.get(&internal_id) else {
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
                let sliced = view.slice(&ranges).map_err(|e| {
                    super::SuperGraphError::InvalidGraph(format!(
                        "compiled_eval: slice failed for segment: {e:?}"
                    ))
                })?;

                let seg_tami = TensorAtomMapInfo {
                    base_id: seg_base,
                    count: seg_count,
                    dtype: tami.dtype,
                    sym_dims: vec![],
                    known_strides: seg_strides.clone(),
                    known_dims: seg_dims,
                    segments: vec![],
                };
                let flat = relayout_to_flat(&seg_tami, &sliced, pool)?;
                initial_inputs.push((seg_base, flat));
            }
        }
    }

    // Also load literal/constant InputTensors that were inlined during lowering.
    // These are already baked into the NanoGraph as LiteralSpans and don't need
    // explicit store entries — the JIT reads them from its literal template buffer.

    eprintln!(
        "[compiled_eval] {} initial inputs loaded",
        initial_inputs.len(),
    );

    // --- Execute ---
    let t0 = std::time::Instant::now();
    let store = compiled.executable_plan.execute_timed(initial_inputs, pool);
    let dt = t0.elapsed();
    eprintln!(
        "[compiled_eval] executed in {:.0}ms",
        dt.as_secs_f64() * 1e3,
    );

    // --- Extract outputs ---
    extract_outputs(&compiled.output_ranges, &compiled.output_shapes, &store, pool)
}

/// Relayout a tensor view to match a TAMI's atom ordering and return as flat 1D.
fn relayout_to_flat<'p, P: Pool + 'p>(
    tami: &TensorAtomMapInfo,
    view: &NumericTensorView<'_, DynRank>,
    pool: &'p P,
) -> Result<NumericTensor<'p, DynRank, P>, super::SuperGraphError> {
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

    let cow = view.relayout(target, pool).map_err(|e| {
        super::SuperGraphError::InvalidGraph(format!("compiled_eval: relayout failed: {e}"))
    })?;

    // The relayouted buffer's bytes are in atom order. Wrap as flat 1D.
    let flat_layout = TensorLayout::<DynRank>::row_major(vec![tami.count], tami.dtype);
    match cow {
        crate::numeric_tensor::NumericTensorCOW::Borrowed(borrow_view) => {
            // Need to copy since we need an owned tensor.
            let buf = pool
                .allocate(flat_layout.buffer_size_bytes())
                .map_err(|e| {
                    super::SuperGraphError::InvalidGraph(format!(
                        "compiled_eval: alloc failed: {e}"
                    ))
                })?;
            let mut tensor = NumericTensor::from_parts(buf, flat_layout);
            let src = borrow_view.data();
            let dst = tensor.buffer_mut();
            let len = src.len().min(dst.len());
            dst[..len].copy_from_slice(&src[..len]);
            Ok(tensor)
        }
        crate::numeric_tensor::NumericTensorCOW::Owned(owned) => {
            // Already owned — reinterpret as flat 1D (zero-cost layout change).
            Ok(owned.into_layout(flat_layout))
        }
    }
}

/// Extract output tensors from the PhaseStore after execution.
fn extract_outputs<'p, P: Pool + 'p>(
    output_ranges: &[(GlobalId, Vec<AtomRange>)],
    output_shapes: &[(GlobalId, Vec<u64>)],
    store: &PhaseStore<'p, P>,
    pool: &'p P,
) -> Result<HashMap<GlobalId, NumericTensor<'p, DynRank, P>>, super::SuperGraphError> {
    let mut results: HashMap<GlobalId, NumericTensor<'p, DynRank, P>> = HashMap::new();

    for ((ext_id, ranges), (_, shape)) in output_ranges.iter().zip(output_shapes.iter()) {
        let dtype = ranges.first().map(|r| r.dtype).unwrap_or(crate::numeric_dtype::NumericDType::F32);
        let elem_bytes = dtype.bytes_per_element();

        // Allocate output tensor with the proper shape.
        let layout = TensorLayout::<DynRank>::row_major(shape.clone(), dtype);
        let buf = pool.allocate(layout.buffer_size_bytes()).map_err(|e| {
            super::SuperGraphError::InvalidGraph(format!(
                "compiled_eval: output alloc failed: {e}"
            ))
        })?;
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
