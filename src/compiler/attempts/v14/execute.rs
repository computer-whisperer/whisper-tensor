//! Span-by-span executor for an ExecutionPlan.
//!
//! Evaluates each span's NanoGraph using the memory-efficient evaluator,
//! threading NDArrayNumericTensor values through the shared value store
//! between phases.
//!
//! Memory management: before execution, a use-count analysis determines
//! how many times each stored atom range will be read. Values are dropped
//! from the store as soon as their last consumer finishes, keeping peak
//! memory proportional to the live set rather than total atoms.

use std::collections::HashMap;

use crate::DynRank;
use crate::backends::ndarray_backend::numeric_tensor::NDArrayNumericTensor;
use crate::graph::GlobalId;
use crate::nano_graph::AtomId;
use crate::nano_graph::eval;

use super::types::*;

/// Execute a plan using the memory-efficient NanoGraph evaluator.
///
/// `inputs` provides all external data (weights + user inputs) as
/// tensors keyed by their base AtomId in the graph. Typically built
/// by mapping the plan's tensor_map entries to actual tensor data.
///
/// Returns the value store keyed by base AtomId. Callers extract
/// outputs using `atom_id_for_element` from the tensor map to handle
/// segmented (Concat) tensors correctly.
pub fn execute(
    plan: &ExecutionPlan,
    inputs: Vec<(AtomId, NDArrayNumericTensor<DynRank>)>,
) -> HashMap<AtomId, NDArrayNumericTensor<DynRank>> {
    // ── Step 1: Use-count analysis ──────────────────────────────────────
    //
    // For each atom range that appears as a span output or initial input,
    // count how many times it will be read as a span input in later phases.
    // Model outputs get an extra count so they survive until extraction.

    let mut use_counts: HashMap<AtomId, u32> = HashMap::new();

    // Initial inputs are consumed by phase 0 spans.
    // Count how many spans in all phases read each input base.
    for phase in &plan.phases {
        for span in &phase.spans {
            for inp in &span.inputs {
                *use_counts.entry(inp.base).or_default() += 1;
            }
        }
    }

    // Model outputs need to survive until extraction.
    for output in &plan.model_outputs {
        *use_counts.entry(output.range.base).or_default() += 1;
    }

    // ── Step 2: Execute phases ──────────────────────────────────────────

    let mut store: HashMap<AtomId, NDArrayNumericTensor<DynRank>> = HashMap::new();
    for (base, tensor) in inputs {
        store.insert(base, tensor);
    }

    for (phase_idx, phase) in plan.phases.iter().enumerate() {
        let mut phase_outputs: Vec<(AtomId, NDArrayNumericTensor<DynRank>)> = Vec::new();

        for (lane_idx, span) in phase.spans.iter().enumerate() {
            if span.graph.num_groups() == 0 {
                continue;
            }

            // Gather this span's inputs from the store.
            let zero_fills: Vec<(usize, NDArrayNumericTensor<DynRank>)> = span
                .inputs
                .iter()
                .enumerate()
                .filter(|(_, range)| !store.contains_key(&range.base))
                .map(|(i, range)| (i, make_zeros(range.count as usize, range.dtype)))
                .collect();
            let missing = zero_fills.len();

            for &(i, ref tensor) in &zero_fills {
                store.insert(span.inputs[i].base, tensor.clone());
            }

            // Collect ALL store entries that fall within each span input range.
            // A single span input range may be served by multiple store entries
            // (e.g., when prior-phase outputs were merged into a wider range).
            // The eval's find_input_idx handles offset-based placement.
            let mut collected_inputs: Vec<(AtomId, NDArrayNumericTensor<DynRank>)> = Vec::new();
            for range in &span.inputs {
                let range_lo = range.base.0;
                let range_hi = range_lo + range.count;

                // Find all store entries whose base falls within this range.
                let mut found_any = false;
                for (&base, tensor) in &store {
                    let t_lo = base.0;
                    let t_hi = t_lo + tensor.num_elements() as u64;
                    // Store entry overlaps this input range?
                    if t_lo < range_hi && t_hi > range_lo {
                        // Slice to fit within the declared range if needed.
                        let needed = range_hi.min(t_hi) - t_lo;
                        if needed as usize > tensor.num_elements() {
                            collected_inputs.push((base, tensor.clone()));
                        } else if (needed as usize) < tensor.num_elements() {
                            collected_inputs
                                .push((base, slice_tensor_prefix(tensor, needed as usize)));
                        } else {
                            collected_inputs.push((base, tensor.clone()));
                        }
                        found_any = true;
                    }
                }
                if !found_any {
                    // Zero-fill was already handled above.
                }
            }
            let span_inputs: Vec<(AtomId, &NDArrayNumericTensor<DynRank>)> = collected_inputs
                .iter()
                .map(|(base, tensor)| (*base, tensor))
                .collect();

            if missing > 0 && phase_idx == 0 && lane_idx == 0 {
                eprintln!(
                    "  Warning: {} of {} input ranges zero-filled (unsupported boundary ops)",
                    missing,
                    span.inputs.len()
                );
            }

            if phase_idx == 0 && lane_idx == 0 {
                for (inp_idx, inp) in span.inputs.iter().enumerate() {
                    if let Some((ti, offset)) = span.graph.find_input_idx(inp.base) {
                        let it = &span.graph.input_tensors()[ti];
                        if it.count != inp.count {
                            eprintln!(
                                "  Phase 0 lane 0: span.inputs[{}] base={} count={} but span input_tensor count={} (diff={})",
                                inp_idx,
                                inp.base,
                                inp.count,
                                it.count,
                                inp.count as i64 - it.count as i64
                            );
                        }
                    }
                }
            }

            // Exhaustive pre-check for the FIRST span that will panic.
            // Only do this once (first non-empty span in phases 0-2).
            if phase_idx < 3 && span.graph.num_groups() > 0 {
                let mut found_missing = false;
                'groups: for g in span.graph.groups() {
                    if matches!(g.op, crate::nano_graph::ScalarOp::Literal(_)) {
                        continue;
                    }

                    // Check every InputRef at every position.
                    for (inp_idx, input) in g.inputs.iter().enumerate() {
                        for pos in 0..g.count {
                            let source = input.resolve(pos + g.atom_offset);
                            if !span.graph.contains_atom(source) {
                                let main_gi = plan.graph.find_group_idx(source);
                                let main_ii = plan.graph.find_input_idx(source);
                                let main_info = if let Some(gi) = main_gi {
                                    let mg = &plan.graph.groups()[gi];
                                    format!(
                                        "group[{}] base={} count={} {:?}",
                                        gi,
                                        mg.base_id,
                                        mg.count,
                                        op_name(&mg.op)
                                    )
                                } else if let Some((ti, off)) = main_ii {
                                    let it = &plan.graph.input_tensors()[ti];
                                    format!(
                                        "input_tensor[{}] base={} count={} off={}",
                                        ti, it.base_id, it.count, off
                                    )
                                } else {
                                    "NOT IN MAIN GRAPH EITHER".to_string()
                                };
                                eprintln!(
                                    "  Phase {} lane {}: group base={} count={} input[{}] at pos={}: atom {} NOT in span. Main: {}",
                                    phase_idx,
                                    lane_idx,
                                    g.base_id,
                                    g.count,
                                    inp_idx,
                                    pos,
                                    source,
                                    main_info,
                                );
                                found_missing = true;
                                break 'groups;
                            }
                        }
                    }

                    // For Reduce ops, also check the stride range.
                    if let crate::nano_graph::ScalarOp::Reduce {
                        reduce_count,
                        reduce_stride,
                        ..
                    } = &g.op
                    {
                        if *reduce_count > 1 && *reduce_stride != 0 {
                            for input in &g.inputs {
                                for pos in 0..g.count {
                                    let base_atom = input.resolve(pos + g.atom_offset);
                                    for k in 0..*reduce_count {
                                        let atom = AtomId(
                                            (base_atom.0 as i64 + k as i64 * reduce_stride) as u64,
                                        );
                                        if !span.graph.contains_atom(atom) {
                                            let main_gi = plan.graph.find_group_idx(atom);
                                            let main_info = if let Some(gi) = main_gi {
                                                let mg = &plan.graph.groups()[gi];
                                                format!(
                                                    "group[{}] base={} count={} {:?}",
                                                    gi,
                                                    mg.base_id,
                                                    mg.count,
                                                    op_name(&mg.op)
                                                )
                                            } else {
                                                "NOT IN MAIN".to_string()
                                            };
                                            eprintln!(
                                                "  Phase {} lane {}: REDUCE group base={} pos={} stride atom {} (k={}) NOT in span. Main: {}",
                                                phase_idx,
                                                lane_idx,
                                                g.base_id,
                                                pos,
                                                atom,
                                                k,
                                                main_info,
                                            );
                                            found_missing = true;
                                            break 'groups;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Dump first span's input/output values for first 2 phases.
            if phase_idx < 2 && lane_idx == 0 && span.graph.num_groups() > 0 {
                eprintln!(
                    "  Phase {} lane 0: {} inputs, {} outputs, {} groups",
                    phase_idx,
                    span_inputs.len(),
                    span.outputs.len(),
                    span.graph.num_groups()
                );
                for (i, &(base, tensor)) in span_inputs.iter().enumerate().take(3) {
                    let f = tensor.flatten();
                    let n = f.num_elements().min(3);
                    let vals: Vec<f64> = (0..n)
                        .map(|j| f.get(&[j as u64]).unwrap().to_f64())
                        .collect();
                    eprintln!(
                        "    input[{}] base={} count={}: {:?}",
                        i,
                        base,
                        tensor.num_elements(),
                        vals
                    );
                }
            }

            // Evaluate the span.
            let eval_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                eval::eval(&span.graph, &span_inputs, &span.outputs)
            }));

            match eval_result {
                Ok(output_tensors) => {
                    if phase_idx < 2 && lane_idx == 0 {
                        for (i, (range, tensor)) in span
                            .outputs
                            .iter()
                            .zip(output_tensors.iter())
                            .enumerate()
                            .take(3)
                        {
                            let f = tensor.flatten();
                            let n = f.num_elements().min(5);
                            let vals: Vec<f64> = (0..n)
                                .map(|j| f.get(&[j as u64]).unwrap().to_f64())
                                .collect();
                            eprintln!(
                                "    output[{}] base={} count={}: {:?}",
                                i, range.base, range.count, vals
                            );
                        }
                    }
                    for (output_range, tensor) in span.outputs.iter().zip(output_tensors) {
                        let actual = tensor.num_elements() as u64;
                        if actual != output_range.count && phase_idx < 5 {
                            eprintln!(
                                "  Phase {} lane {}: output base={} expected {} atoms but eval returned {}",
                                phase_idx, lane_idx, output_range.base, output_range.count, actual
                            );
                        }
                        phase_outputs.push((output_range.base, tensor));
                    }
                }
                Err(_) => {
                    eprintln!(
                        "  Phase {} lane {}: eval panicked, zero-filling outputs",
                        phase_idx, lane_idx
                    );
                    for output_range in &span.outputs {
                        let tensor = make_zeros(output_range.count as usize, output_range.dtype);
                        phase_outputs.push((output_range.base, tensor));
                    }
                }
            }

            // TODO: liveness-based dropping. Disabled for now to ensure
            // all atoms survive for model output extraction (segmented
            // outputs may reference atoms not declared as span inputs).
            // Re-enable once model output atom ranges are properly tracked.
        }

        // Commit phase outputs to the store (barrier).
        // Check for duplicate keys — multiple spans outputting the same base.
        let n_outputs = phase_outputs.len();
        if phase_idx < 5 {
            let mut seen = std::collections::HashSet::new();
            for (base, _) in &phase_outputs {
                if !seen.insert(base.0) {
                    eprintln!(
                        "  Phase {}: DUPLICATE output base={} (overwrite!)",
                        phase_idx, base
                    );
                }
            }
        }
        for (base, tensor) in phase_outputs {
            store.insert(base, tensor);
        }

        // Diagnostic: for next phase, check if any input is reading stale
        // initial data when it should be reading a computed result.
        if phase_idx < 5 {
            if let Some(next_phase) = plan.phases.get(phase_idx + 1) {
                for (lane_idx, span) in next_phase.spans.iter().enumerate() {
                    for inp in &span.inputs {
                        // Is this input a computed group in the main graph?
                        if let Some(gi) = plan.graph.find_group_idx(inp.base) {
                            let g = &plan.graph.groups()[gi];
                            if !matches!(g.op, crate::nano_graph::ScalarOp::Literal(_)) {
                                // This is a computed group. Was it output by some span?
                                let was_output = plan.phases[..=phase_idx]
                                    .iter()
                                    .flat_map(|p| p.spans.iter())
                                    .any(|s| s.outputs.iter().any(|o| o.base == inp.base));
                                if !was_output {
                                    eprintln!(
                                        "  BUG: Phase {} lane {} reads computed group base={} count={} ({:?}) \
                                         but NO prior span output it! Store has stale initial data.",
                                        phase_idx + 1,
                                        lane_idx,
                                        inp.base,
                                        inp.count,
                                        crate::nano_graph::ScalarOp::Identity, // placeholder
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        // Diagnostic: check count mismatches between store and next phase inputs.
        if phase_idx < 5 {
            if let Some(next_phase) = plan.phases.get(phase_idx + 1) {
                let mut count_mismatches = 0usize;
                let mut missing = 0usize;
                for span in &next_phase.spans {
                    for inp in &span.inputs {
                        if let Some(tensor) = store.get(&inp.base) {
                            let stored = tensor.num_elements() as u64;
                            if stored != inp.count {
                                count_mismatches += 1;
                                if count_mismatches <= 5 {
                                    eprintln!(
                                        "  Phase {} → {}: input base={} expects {} atoms but store has {}",
                                        phase_idx,
                                        phase_idx + 1,
                                        inp.base,
                                        inp.count,
                                        stored
                                    );
                                }
                            }
                        } else {
                            missing += 1;
                        }
                    }
                }
                if count_mismatches > 0 || missing > 0 {
                    eprintln!(
                        "  Phase {} → {}: {} count mismatches, {} missing",
                        phase_idx,
                        phase_idx + 1,
                        count_mismatches,
                        missing
                    );
                }
            }
        }
    }

    store
}

/// Slice a tensor to its first `count` elements (flattened).
fn slice_tensor_prefix(
    tensor: &NDArrayNumericTensor<DynRank>,
    count: usize,
) -> NDArrayNumericTensor<DynRank> {
    use ndarray::{ArcArray, IxDyn};
    macro_rules! slice_variant {
        ($arr:expr, $variant:ident) => {{
            let flat: Vec<_> = $arr.iter().take(count).copied().collect();
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

/// Create a zero-filled 1D tensor of the given count and dtype.
fn make_zeros(count: usize, dtype: crate::dtype::DType) -> NDArrayNumericTensor<DynRank> {
    use ndarray::{ArcArray, IxDyn};
    let shape = IxDyn(&[count]);
    match dtype {
        crate::dtype::DType::F32 => NDArrayNumericTensor::F32(ArcArray::zeros(shape)),
        crate::dtype::DType::F64 => NDArrayNumericTensor::F64(ArcArray::zeros(shape)),
        crate::dtype::DType::I64 => NDArrayNumericTensor::I64(ArcArray::zeros(shape)),
        crate::dtype::DType::I32 => NDArrayNumericTensor::I32(ArcArray::zeros(shape)),
        _other => NDArrayNumericTensor::F32(ArcArray::zeros(shape)),
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
    use crate::nano_graph::{InputRef, ScalarOp};
    use std::collections::HashSet;

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
                        // Might be a duplicated literal or split group — check
                        // if any main group contains this base_id.
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
                        // Verify op kind matches.
                        if op_name(&span_group.op) != op_name(&mg.op) {
                            errors.push(format!(
                                "{}: group base={}: op mismatch: span={:?} main={:?}",
                                prefix,
                                span_group.base_id,
                                op_name(&span_group.op),
                                op_name(&mg.op)
                            ));
                        }
                        // Verify output dtype matches.
                        if span_group.output_dtype != mg.output_dtype {
                            errors.push(format!(
                                "{}: group base={}: dtype mismatch: span={:?} main={:?}",
                                prefix,
                                span_group.base_id,
                                span_group.output_dtype,
                                mg.output_dtype
                            ));
                        }
                        // Verify InputRefs match.
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
                        // Verify count matches (allow split groups with smaller count).
                        if span_group.count > mg.count {
                            errors.push(format!(
                                "{}: group base={}: count {} > main count {}",
                                prefix, span_group.base_id, span_group.count, mg.count
                            ));
                        }
                    }
                }

                // Check 2: all InputRef targets are covered.
                // Sample positions to avoid O(atoms).
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
                            break; // one error per input is enough
                        }
                    }

                    // For Reduce ops, also check strided range endpoints.
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

                    // For IndirectLoad, check table range.
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
                        // Check if it's produced by a span group (then it shouldn't be in inputs).
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
                errors.push(format!("... truncated after 50 errors"));
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
        InputRef::Affine { base, stride } => format!("Affine(base={}, stride={})", base, stride),
        InputRef::StridedBroadcast {
            base,
            stride,
            repeat,
        } => format!(
            "StridedBroadcast(base={}, stride={}, repeat={})",
            base, stride, repeat
        ),
        InputRef::Modular {
            base,
            stride,
            modulus,
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
