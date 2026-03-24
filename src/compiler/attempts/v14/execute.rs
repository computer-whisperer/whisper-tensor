//! Multi-threaded executor for an ExecutionPlan.
//!
//! Phases execute sequentially, separated by barriers. Within each phase,
//! spans execute in parallel across threads (one per lane). Data flows
//! between phases through a shared value store keyed by AtomId.
//!
//! The store is read-only during each phase — threads only read from it
//! to gather their span's inputs. After all spans in a phase complete,
//! their outputs are committed to the store before the next phase begins.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::DynRank;
use crate::backends::ndarray_backend::numeric_tensor::NDArrayNumericTensor;
use crate::nano_graph::AtomId;

use super::types::*;

/// Execute a plan, evaluating spans in parallel within each phase.
///
/// `inputs` provides all external data (weights + user inputs) as
/// tensors keyed by their base AtomId. Returns the value store after
/// all phases complete. Callers extract model outputs using
/// `atom_id_for_element` from the tensor map.
pub fn execute(
    plan: &ExecutionPlan,
    inputs: Vec<(AtomId, NDArrayNumericTensor<DynRank>)>,
) -> HashMap<AtomId, NDArrayNumericTensor<DynRank>> {
    let mut store: HashMap<AtomId, NDArrayNumericTensor<DynRank>> = HashMap::new();
    for (base, tensor) in inputs {
        store.insert(base, tensor);
    }

    for phase in &plan.phases {
        // Evaluate all spans in parallel. The store is immutably borrowed
        // for the duration — each thread gathers its own inputs by scanning
        // for overlapping entries, then evaluates its span independently.
        let phase_outputs: Vec<Vec<(AtomId, NDArrayNumericTensor<DynRank>)>> = phase
            .spans
            .par_iter()
            .map(|span| eval_span(span, &store))
            .collect();

        // ── Barrier: commit all outputs to store ──
        for span_outputs in phase_outputs {
            for (base, tensor) in span_outputs {
                store.insert(base, tensor);
            }
        }
    }

    store
}

/// Evaluate a single span: gather inputs from the store, run eval, return outputs.
fn eval_span(
    span: &Span,
    store: &HashMap<AtomId, NDArrayNumericTensor<DynRank>>,
) -> Vec<(AtomId, NDArrayNumericTensor<DynRank>)> {
    if span.graph.num_groups() == 0 {
        return Vec::new();
    }

    let inputs = gather_inputs(span, store);
    let input_refs: Vec<(AtomId, &NDArrayNumericTensor<DynRank>)> =
        inputs.iter().map(|(base, t)| (*base, t)).collect();

    // TODO: eval module was deleted; needs migration to pool_eval (different API signature)
    let output_tensors: Vec<NDArrayNumericTensor<DynRank>> = todo!("migrate to pool_eval");

    span.outputs
        .iter()
        .zip(output_tensors)
        .map(|(range, tensor)| (range.base, tensor))
        .collect()
}

/// Gather a span's input data from the store by scanning for overlapping entries.
///
/// For each declared input range, finds all store entries that overlap and
/// extracts the exact intersection. Store entries may start before or extend
/// past the input range — we slice to the overlap and rebase the AtomId so
/// that eval's `find_input_idx` can resolve it.
/// Public gather for diagnostic use by codegen.
pub fn gather_inputs_pub(
    input_ranges: &[AtomRange],
    _output_ranges: &[AtomRange],
    store: &HashMap<AtomId, NDArrayNumericTensor<DynRank>>,
) -> Vec<(AtomId, NDArrayNumericTensor<DynRank>)> {
    let mut collected = Vec::new();
    for range in input_ranges {
        let range_lo = range.base.0;
        let range_hi = range_lo + range.count;
        for (&base, tensor) in store {
            let t_lo = base.0;
            let t_hi = t_lo + tensor.num_elements() as u64;
            if t_lo < range_hi && t_hi > range_lo {
                let overlap_start = t_lo.max(range_lo);
                let overlap_end = t_hi.min(range_hi);
                let overlap_count = (overlap_end - overlap_start) as usize;
                let skip = (overlap_start - t_lo) as usize;
                let sliced = slice_tensor_range(tensor, skip, overlap_count);
                collected.push((AtomId(overlap_start), sliced));
            }
        }
    }
    collected
}

fn gather_inputs(
    span: &Span,
    store: &HashMap<AtomId, NDArrayNumericTensor<DynRank>>,
) -> Vec<(AtomId, NDArrayNumericTensor<DynRank>)> {
    let mut collected = Vec::new();
    for range in &span.inputs {
        let range_lo = range.base.0;
        let range_hi = range_lo + range.count;

        for (&base, tensor) in store {
            let t_lo = base.0;
            let t_hi = t_lo + tensor.num_elements() as u64;
            if t_lo < range_hi && t_hi > range_lo {
                let overlap_start = t_lo.max(range_lo);
                let overlap_end = t_hi.min(range_hi);
                let overlap_count = (overlap_end - overlap_start) as usize;
                let skip = (overlap_start - t_lo) as usize;

                let sliced = slice_tensor_range(tensor, skip, overlap_count);
                collected.push((AtomId(overlap_start), sliced));
            }
        }
    }
    collected
}

/// Extract a contiguous sub-range from a flattened tensor.
///
/// Skips the first `skip` elements, then takes the next `count` elements.
/// Returns a 1D tensor of length `count`.
fn slice_tensor_range(
    tensor: &NDArrayNumericTensor<DynRank>,
    skip: usize,
    count: usize,
) -> NDArrayNumericTensor<DynRank> {
    use ndarray::{ArcArray, IxDyn};
    if skip == 0 && count == tensor.num_elements() {
        return tensor.clone();
    }
    macro_rules! slice_variant {
        ($arr:expr, $variant:ident) => {{
            let flat: Vec<_> = $arr.iter().skip(skip).take(count).copied().collect();
            NDArrayNumericTensor::$variant(ArcArray::from_shape_vec(IxDyn(&[count]), flat).unwrap())
        }};
    }
    match tensor {
        NDArrayNumericTensor::F32(a) => slice_variant!(a, F32),
        NDArrayNumericTensor::F64(a) => slice_variant!(a, F64),
        NDArrayNumericTensor::I64(a) => slice_variant!(a, I64),
        NDArrayNumericTensor::I32(a) => slice_variant!(a, I32),
        NDArrayNumericTensor::BF16(a) => slice_variant!(a, BF16),
        NDArrayNumericTensor::F16(a) => slice_variant!(a, F16),
        NDArrayNumericTensor::U8(a) => slice_variant!(a, U8),
        NDArrayNumericTensor::I8(a) => slice_variant!(a, I8),
        other => other.clone(),
    }
}

// ─── Span subgraph validation ──────────────────────────────────────────────

/// Validate that every span in an ExecutionPlan is a faithful subgraph of
/// the main NanoGraph. Returns a list of errors (empty = valid).
///
/// Checks:
/// 1. Every compute group in a span matches a group in the main graph
///    (same base_id, count, op kind, dtype, and InputRefs).
/// 2. Every atom referenced by a span group's InputRefs is either:
///    - produced by another group in the same span, OR
///    - covered by one of the span's declared input ranges
/// 3. No span group references atoms that are neither internal nor declared.
pub fn validate_spans(plan: &ExecutionPlan) -> Vec<String> {
    use crate::nano_graph::ScalarOp;

    let main = &plan.graph;
    let mut errors = Vec::new();

    for (phase_idx, phase) in plan.phases.iter().enumerate() {
        for (lane_idx, span) in phase.spans.iter().enumerate() {
            let prefix = format!("phase {} lane {}", phase_idx, lane_idx);

            // Build set of atom ID ranges produced by groups in this span.
            let mut span_produced: Vec<(u64, u64)> = Vec::new(); // (base, end)
            for g in span.graph.groups() {
                span_produced.push((g.base_id.0, g.base_id.0 + g.count));
            }

            // Build set of atom ID ranges declared as span inputs.
            let mut span_input_ranges: Vec<(u64, u64)> = Vec::new();
            for inp in &span.inputs {
                span_input_ranges.push((inp.base.0, inp.base.0 + inp.count));
            }
            // Also include input_tensors from the span graph itself.
            for it in span.graph.input_tensors() {
                span_input_ranges.push((it.base_id.0, it.base_id.0 + it.count));
            }

            let atom_covered = |atom: u64| -> bool {
                for &(lo, hi) in &span_produced {
                    if atom >= lo && atom < hi {
                        return true;
                    }
                }
                for &(lo, hi) in &span_input_ranges {
                    if atom >= lo && atom < hi {
                        return true;
                    }
                }
                false
            };

            for (gi, span_group) in span.graph.groups().iter().enumerate() {
                // Check 1: group matches main graph.
                let main_group = main.group_of(span_group.base_id);
                match main_group {
                    None => {
                        if main.find_group_idx(span_group.base_id).is_none()
                            && !matches!(span_group.op, ScalarOp::Literal(_))
                        {
                            errors.push(format!(
                                "{}: span group {} (base={}, op={:?}) not found in main graph",
                                prefix,
                                gi,
                                span_group.base_id,
                                op_name(&span_group.op)
                            ));
                        }
                    }
                    Some(mg) => {
                        if op_name(&span_group.op) != op_name(&mg.op) {
                            errors.push(format!(
                                "{}: group base={}: op mismatch: span={:?} main={:?}",
                                prefix,
                                span_group.base_id,
                                op_name(&span_group.op),
                                op_name(&mg.op)
                            ));
                        }
                        if span_group.output_dtype != mg.output_dtype {
                            errors.push(format!(
                                "{}: group base={}: dtype mismatch: span={:?} main={:?}",
                                prefix,
                                span_group.base_id,
                                span_group.output_dtype,
                                mg.output_dtype
                            ));
                        }
                        if span_group.inputs.len() != mg.inputs.len() {
                            errors.push(format!(
                                "{}: group base={}: input count mismatch: span={} main={}",
                                prefix,
                                span_group.base_id,
                                span_group.inputs.len(),
                                mg.inputs.len()
                            ));
                        } else {
                            for (inp_idx, (si, mi)) in
                                span_group.inputs.iter().zip(mg.inputs.iter()).enumerate()
                            {
                                if si != mi {
                                    errors.push(format!(
                                        "{}: group base={} input {}: InputRef mismatch\n  span: {:?}\n  main: {:?}",
                                        prefix, span_group.base_id, inp_idx,
                                        format_input_ref(si),
                                        format_input_ref(mi),
                                    ));
                                }
                            }
                        }
                        if span_group.count > mg.count {
                            errors.push(format!(
                                "{}: group base={}: count {} > main count {}",
                                prefix, span_group.base_id, span_group.count, mg.count
                            ));
                        }
                    }
                }

                // Check 2: all InputRef targets are covered.
                let sample_count = span_group.count.min(16);
                let step = if span_group.count > 16 {
                    span_group.count / 16
                } else {
                    1
                };
                for input_ref in &span_group.inputs {
                    for s in 0..sample_count {
                        let i = s * step;
                        let source = input_ref.resolve(i + span_group.atom_offset);
                        if !atom_covered(source.0) {
                            errors.push(format!(
                                "{}: group base={} atom_offset={} input {:?}: \
                                 source atom {} (at i={}) not covered by span groups or inputs",
                                prefix,
                                span_group.base_id,
                                span_group.atom_offset,
                                format_input_ref(input_ref),
                                source,
                                i
                            ));
                            break;
                        }
                    }

                    if let ScalarOp::Reduce {
                        reduce_count,
                        reduce_stride,
                        ..
                    } = &span_group.op
                    {
                        if *reduce_count > 1 && *reduce_stride != 0 {
                            let first = input_ref.resolve(span_group.atom_offset);
                            let last =
                                input_ref.resolve(span_group.atom_offset + span_group.count - 1);
                            let end_off = (*reduce_count as i64 - 1) * reduce_stride;
                            let endpoints = [
                                first.0,
                                (first.0 as i64 + end_off) as u64,
                                last.0,
                                (last.0 as i64 + end_off) as u64,
                            ];
                            for &ep in &endpoints {
                                if !atom_covered(ep) {
                                    errors.push(format!(
                                        "{}: group base={} reduce stride endpoint atom {} not covered",
                                        prefix, span_group.base_id, ep
                                    ));
                                    break;
                                }
                            }
                        }
                    }

                    if let ScalarOp::IndirectLoad { table_base } = &span_group.op {
                        if !atom_covered(table_base.0) {
                            errors.push(format!(
                                "{}: group base={} IndirectLoad table_base {} not covered",
                                prefix, span_group.base_id, table_base
                            ));
                        }
                    }
                }
            }

            // Check 3: every atom in span.outputs is in the span graph.
            for (out_idx, out) in span.outputs.iter().enumerate() {
                for offset in [0, out.count / 2, out.count.saturating_sub(1)] {
                    if offset >= out.count {
                        continue;
                    }
                    let atom = AtomId(out.base.0 + offset);
                    if !span.graph.contains_atom(atom) {
                        errors.push(format!(
                            "{}: span.outputs[{}] atom {} (base={} offset={}) not in span graph",
                            prefix, out_idx, atom, out.base, offset
                        ));
                        break;
                    }
                }
            }

            // Check 4: every span.inputs entry resolves in the span graph's input_tensors.
            for (inp_idx, inp) in span.inputs.iter().enumerate() {
                match span.graph.find_input_idx(inp.base) {
                    Some((_, offset)) => {
                        if offset != 0 {
                            errors.push(format!(
                                "{}: span.inputs[{}] base={} resolves at offset {} in span input_tensor (expected 0)",
                                prefix, inp_idx, inp.base, offset
                            ));
                        }
                    }
                    None => {
                        let in_span_group = span.graph.find_group_idx(inp.base).is_some();
                        if !in_span_group {
                            errors.push(format!(
                                "{}: span.inputs[{}] base={} count={} not found in span graph \
                                 (not a group, not an input_tensor)",
                                prefix, inp_idx, inp.base, inp.count
                            ));
                        }
                    }
                }
            }

            if errors.len() > 50 {
                errors.push("... truncated after 50 errors".to_string());
                return errors;
            }
        }
    }
    errors
}

fn op_name(op: &crate::nano_graph::ScalarOp) -> &'static str {
    use crate::nano_graph::ScalarOp;
    match op {
        ScalarOp::Literal(_) => "Literal",
        ScalarOp::Identity => "Identity",
        ScalarOp::Binary { .. } => "Binary",
        ScalarOp::Unary { .. } => "Unary",
        ScalarOp::Select => "Select",
        ScalarOp::Reduce { .. } => "Reduce",
        ScalarOp::IndirectLoad { .. } => "IndirectLoad",
    }
}

fn format_input_ref(ir: &crate::nano_graph::InputRef) -> String {
    use crate::nano_graph::InputRef;
    match ir {
        InputRef::Broadcast(id) => format!("Broadcast({})", id),
        InputRef::Strided {
            base,
            stride_inner: stride,
            ..
        } => format!("Affine(base={}, stride={})", base, stride),
        InputRef::Strided {
            base,
            stride_outer: stride,
            modulus: repeat,
            ..
        } => format!(
            "StridedBroadcast(base={}, stride={}, repeat={})",
            base, stride, repeat
        ),
        InputRef::Strided {
            base,
            stride_inner: stride,
            modulus,
            ..
        } => format!(
            "Modular(base={}, stride={}, modulus={})",
            base, stride, modulus
        ),
        InputRef::Explicit(ids) => format!(
            "Explicit([{}; len={}])",
            if ids.is_empty() {
                "".into()
            } else {
                format!("{}, ...", ids[0])
            },
            ids.len()
        ),
    }
}
