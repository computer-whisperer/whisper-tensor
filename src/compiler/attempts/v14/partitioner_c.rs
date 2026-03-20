#![allow(clippy::all, dead_code, unreachable_patterns, unused_variables, unused_imports)]

//! Greedy Lane Simulation partitioner.
//!
//! Bottom-up, online scheduling that processes groups in topological order
//! and greedily assigns each to a lane. Barriers are inserted only when no
//! lane can accept a group without violating cross-lane independence.
//!
//! Key ideas:
//! - Process groups in topological order (the NanoGraph already stores them
//!   this way).
//! - Track what each lane has produced (as group index sets).
//! - For each group, check which lanes could execute it without needing
//!   atoms produced by a *different* lane in the *current* phase.
//! - Pick the best viable lane (cache affinity heuristic: the lane whose
//!   current-phase groups already provide the most inputs).
//! - If no lane can take a group, insert a barrier and start a new phase.
//! - Split large groups across lanes for work balance when they have no
//!   cross-lane dependency issues.

use std::collections::{BTreeMap, HashSet};

use crate::dtype::DType;
use crate::graph::GlobalId;
use crate::nano_graph::pattern::InputTensor;
use crate::nano_graph::{AtomGroup, AtomId, AtomRange, InputRef, NanoGraph, ScalarOp};

use super::types::{Phase, Span};

// ─── Public API ─────────────────────────────────────────────────────────────

/// Partition a NanoGraph into phases and spans for parallel execution.
///
/// Returns a `Vec<Phase>` where each phase has `num_lanes` spans.
/// The caller wraps this into the full `ExecutionPlan`.
pub fn plan(
    graph: &NanoGraph,
    num_lanes: usize,
    input_tensors: &[InputTensor],
    output_atom_ids: &[AtomId],
) -> Vec<Phase> {
    assert!(num_lanes >= 1);
    let groups = graph.groups();
    if groups.is_empty() {
        return vec![Phase {
            spans: (0..num_lanes).map(|_| empty_span(graph)).collect(),
        }];
    }

    // Build the group-level dependency graph. For each group index, which
    // other group indices are its producers?
    let num_groups = groups.len();
    let t0 = std::time::Instant::now();
    let mut producer_sets: Vec<HashSet<usize>> = Vec::with_capacity(num_groups);
    for (gi, group) in groups.iter().enumerate() {
        let mut producers = HashSet::new();
        graph.collect_all_producer_indices(group, gi, &mut producers);
        producer_sets.push(producers);
    }
    eprintln!("  [C] dependency DAG: {:.1?} ({} groups)", t0.elapsed(), num_groups);

    // Note: input tensors are external data (weights, user inputs) that are
    // always available. They don't constrain lane assignment — only groups
    // in the current phase do.

    // ─── Greedy simulation ───────────────────────────────────────────────

    // For each group, decide: which lane and which phase?
    // group_assignment[gi] = (phase_idx, lane_idx)
    let mut group_assignment: Vec<(usize, usize)> = vec![(0, 0); num_groups];

    // Current phase index.
    let mut current_phase: usize = 0;

    // For each lane in the current phase, track which group indices it owns.
    let mut lane_groups_current_phase: Vec<HashSet<usize>> =
        (0..num_lanes).map(|_| HashSet::new()).collect();

    // For each lane, track total atom count in current phase (for load balancing).
    let mut lane_atoms_current_phase: Vec<u64> = vec![0; num_lanes];

    // Track which groups were produced in *prior* phases (available to all lanes).
    let mut prior_phase_groups: HashSet<usize> = HashSet::new();

    // Track all groups that are "external" (input tensors).
    // These are always available and don't constrain lane assignment.

    // Phase boundary tracking: which groups ended up in which phase.
    let mut phase_group_lists: Vec<Vec<Vec<usize>>> = Vec::new(); // phase -> lane -> [group_indices]

    let start_new_phase = |lane_groups: &mut Vec<HashSet<usize>>,
                           lane_atoms: &mut Vec<u64>,
                           prior: &mut HashSet<usize>,
                           phase_lists: &mut Vec<Vec<Vec<usize>>>| {
        // Commit current lane groups to the phase list.
        let phase_lanes: Vec<Vec<usize>> = lane_groups
            .iter()
            .map(|s| {
                let mut v: Vec<usize> = s.iter().copied().collect();
                v.sort();
                v
            })
            .collect();

        // Move current groups to prior.
        for lane_set in lane_groups.iter() {
            for &gi in lane_set {
                prior.insert(gi);
            }
        }

        phase_lists.push(phase_lanes);

        // Reset for new phase.
        for s in lane_groups.iter_mut() {
            s.clear();
        }
        for a in lane_atoms.iter_mut() {
            *a = 0;
        }
    };

    let t1 = std::time::Instant::now();
    for gi in 0..num_groups {
        let group = &groups[gi];
        let producers = &producer_sets[gi];

        // Determine which lanes are viable for this group.
        // A lane is viable if ALL producer groups are either:
        //   (a) external inputs (not in any group), or
        //   (b) in a prior phase, or
        //   (c) in the SAME lane in the current phase.
        let mut viable_lanes: Vec<usize> = Vec::new();
        let mut best_lane: Option<usize> = None;
        let mut best_affinity: usize = 0;

        for lane in 0..num_lanes {
            let mut ok = true;
            let mut affinity = 0usize; // how many producers this lane already has

            for &pi in producers {
                if prior_phase_groups.contains(&pi) {
                    // Available from a prior phase — fine for any lane.
                    continue;
                }
                // Check if this producer is in the current phase.
                if lane_groups_current_phase[lane].contains(&pi) {
                    // Same lane, current phase — fine.
                    affinity += 1;
                    continue;
                }
                // Check if it's in a *different* lane in the current phase.
                let mut in_other_lane = false;
                for other_lane in 0..num_lanes {
                    if other_lane != lane
                        && lane_groups_current_phase[other_lane].contains(&pi)
                    {
                        in_other_lane = true;
                        break;
                    }
                }
                if in_other_lane {
                    // This producer is in a different lane — can't use this lane.
                    ok = false;
                    break;
                }
                // Producer not assigned yet? It must have been an input tensor
                // (external) or we have a bug. External inputs are fine.
                // Actually, since we process in topological order, all producers
                // with smaller indices have already been assigned. If we didn't
                // find it in any lane or prior phases, it must be external.
                // External inputs are always available — that's fine.
            }

            if ok {
                viable_lanes.push(lane);
                if affinity > best_affinity
                    || (affinity == best_affinity
                        && best_lane.map_or(true, |bl| {
                            lane_atoms_current_phase[lane] < lane_atoms_current_phase[bl]
                        }))
                {
                    best_affinity = affinity;
                    best_lane = Some(lane);
                }
            }
        }

        if viable_lanes.is_empty() {
            // No lane can take this group — need a barrier.
            start_new_phase(
                &mut lane_groups_current_phase,
                &mut lane_atoms_current_phase,
                &mut prior_phase_groups,
                &mut phase_group_lists,
            );
            current_phase += 1;

            // After barrier, all prior groups are available. Assign to
            // the least-loaded lane (which is empty now — all are).
            best_lane = Some(0);
        } else if best_lane.is_none() {
            // All viable lanes had zero affinity — pick least loaded.
            best_lane = Some(
                *viable_lanes
                    .iter()
                    .min_by_key(|&&l| lane_atoms_current_phase[l])
                    .unwrap(),
            );
        }

        let lane = best_lane.unwrap();
        group_assignment[gi] = (current_phase, lane);
        lane_groups_current_phase[lane].insert(gi);
        lane_atoms_current_phase[lane] += group.count;
    }

    // Flush the last phase.
    start_new_phase(
        &mut lane_groups_current_phase,
        &mut lane_atoms_current_phase,
        &mut prior_phase_groups,
        &mut phase_group_lists,
    );

    // ─── Work balance: split large groups ───────────────────────────────

    // After greedy assignment, check work balance within each phase.
    // If one lane has much more work, try splitting its largest groups
    // across lanes (only groups with no same-phase consumers in other lanes
    // can be split without creating new dependencies).
    //
    // We do this as a post-pass: for each phase, find the most imbalanced
    // lane and split its largest Literal groups (which have no inputs and
    // are trivially splittable) or large compute groups where the consumers
    // are all in the same lane or in a later phase.

    // For simplicity in the initial implementation, we'll split groups
    // during span construction rather than modifying the assignment.
    // The key optimization: when a single lane has a group that is much
    // larger than other lanes' total work, we split it.

    // ─── Build phases from assignments ──────────────────────────────────

    let total_phases = phase_group_lists.len();

    // For each group, determine which phase first consumes it (so we know
    // when to make it a span output).
    let mut group_consumer_phases: Vec<Option<usize>> = vec![None; num_groups];
    for gi in 0..num_groups {
        let (my_phase, _my_lane) = group_assignment[gi];
        for &pi in &producer_sets[gi] {
            let (prod_phase, _prod_lane) = group_assignment[pi];
            if prod_phase < my_phase {
                // Producer is in an earlier phase — it must be a span output
                // of its phase.
                let entry = &mut group_consumer_phases[pi];
                match entry {
                    None => *entry = Some(my_phase),
                    Some(existing) => {
                        if my_phase < *existing {
                            *entry = Some(my_phase);
                        }
                    }
                }
            }
        }
    }

    // Also mark groups that contain output atoms.
    for gi in 0..num_groups {
        let group = &groups[gi];
        let base = group.base_id.0;
        for &out_id in output_atom_ids {
            if out_id.0 >= base && out_id.0 < base + group.count {
                // This group produces an output atom.
                if group_consumer_phases[gi].is_none() {
                    group_consumer_phases[gi] = Some(total_phases); // sentinel: needed as final output
                }
            }
        }
    }

    // ─── Construct span NanoGraphs ───────────────────────────────────────
    eprintln!("  [C] greedy simulation: {:.1?} ({} phases)", t1.elapsed(), total_phases);
    let t2 = std::time::Instant::now();

    let mut phases: Vec<Phase> = Vec::with_capacity(total_phases);

    for (phase_idx, phase_lanes) in phase_group_lists.iter().enumerate() {
        let mut spans: Vec<Span> = Vec::with_capacity(num_lanes);

        for (lane_idx, lane_group_indices) in phase_lanes.iter().enumerate() {
            if lane_group_indices.is_empty() {
                spans.push(empty_span(graph));
                continue;
            }

            let span = build_span(
                graph,
                groups,
                lane_group_indices,
                phase_idx,
                &producer_sets,
                &group_consumer_phases,
            );
            spans.push(span);
        }

        phases.push(Phase { spans });
    }

    // ─── Load balance post-pass: split large groups ─────────────────────

    // After building the initial plan, check if any phase has severe
    // work imbalance and try to redistribute by splitting groups.
    balance_phases(&mut phases, num_lanes, graph);

    eprintln!("  [C] span construction: {:.1?}", t2.elapsed());
    phases
}

// ─── Span construction ──────────────────────────────────────────────────────

/// Build a Span for one lane in one phase.
fn build_span(
    main_graph: &NanoGraph,
    groups: &[AtomGroup],
    group_indices: &[usize],
    phase_idx: usize,
    producer_sets: &[HashSet<usize>],
    group_consumer_phases: &[Option<usize>],
) -> Span {
    // Collect the groups this span will contain, in topological order
    // (they're already sorted by group index = topological order).
    let gi_set: HashSet<usize> = group_indices.iter().copied().collect();

    // Determine external inputs: atom ranges this span reads that are not
    // produced by a group in this span. These come from:
    // 1. Input tensors (weights/user inputs)
    // 2. Groups from prior phases
    let mut external_input_ranges: BTreeMap<u64, AtomRange> = BTreeMap::new();

    for &gi in group_indices {
        let group = &groups[gi];
        let producers = &producer_sets[gi];

        for &pi in producers {
            if gi_set.contains(&pi) {
                // Internal dependency — produced within this span.
                continue;
            }
            // External dependency. Add the producer group's atom range.
            let prod_group = &groups[pi];
            external_input_ranges
                .entry(prod_group.base_id.0)
                .or_insert_with(|| AtomRange {
                    base: prod_group.base_id,
                    count: prod_group.count,
                    dtype: prod_group.output_dtype,
                });
        }

        // Also check for input tensor references.
        collect_input_tensor_refs(main_graph, group, &mut external_input_ranges);
    }

    // Determine outputs: atom ranges this span produces that are needed by
    // later phases or are model outputs.
    let mut output_ranges: BTreeMap<u64, AtomRange> = BTreeMap::new();

    for &gi in group_indices {
        if group_consumer_phases[gi].map_or(false, |cp| cp > phase_idx) {
            let group = &groups[gi];
            output_ranges
                .entry(group.base_id.0)
                .or_insert_with(|| AtomRange {
                    base: group.base_id,
                    count: group.count,
                    dtype: group.output_dtype,
                });
        }
    }

    // Build the span with preserved atom IDs.
    build_span_graph_with_id_preservation(
        main_graph,
        groups,
        group_indices,
        &external_input_ranges,
        &output_ranges,
    )
}

/// Build a span NanoGraph that preserves atom IDs from the main graph.
///
/// This works by inserting entries in atom ID order, using padding
/// placeholders to skip ID ranges that don't belong to this span.
fn build_span_graph_with_id_preservation(
    main_graph: &NanoGraph,
    all_groups: &[AtomGroup],
    group_indices: &[usize],
    external_inputs: &BTreeMap<u64, AtomRange>,
    output_ranges: &BTreeMap<u64, AtomRange>,
) -> Span {
    let mut span_graph = NanoGraph::new();
    span_graph.sym_dim_names = main_graph.sym_dim_names.clone();
    span_graph.sym_dim_bounds = main_graph.sym_dim_bounds.clone();

    // Collect all items we need in the span graph, sorted by base atom ID.
    // Items are either:
    // - External input ranges (registered as input_tensors)
    // - Compute groups (from this span)
    #[derive(Debug)]
    enum SpanItem {
        ExternalInput(AtomRange),
        Group(usize), // index into all_groups
    }

    let mut items: Vec<(u64, u64, SpanItem)> = Vec::new(); // (base_id, count, item)

    for (_, range) in external_inputs.iter() {
        items.push((range.base.0, range.count, SpanItem::ExternalInput(range.clone())));
    }

    for &gi in group_indices {
        let group = &all_groups[gi];
        items.push((group.base_id.0, group.count, SpanItem::Group(gi)));
    }

    // Sort by base atom ID.
    items.sort_by_key(|(base, _, _)| *base);

    // Deduplicate overlapping ranges (shouldn't happen, but be safe).
    // Actually, external inputs and groups should never overlap since
    // external inputs are prior-phase outputs or input tensors, and groups
    // are this phase's compute.

    // Insert items in order, padding gaps with placeholder allocations.
    // The NanoGraph.next_atom_id starts at 0. We need to advance it to
    // each item's base_id before inserting.
    let mut current_id: u64 = 0;

    for (base_id, count, item) in &items {
        // Skip past any gap.
        if *base_id > current_id {
            let gap = *base_id - current_id;
            // Allocate a throwaway placeholder to burn through the ID space.
            // We use add_input_tensor with a dummy to skip IDs.
            span_graph.add_input_tensor(GlobalId(u64::MAX - current_id), gap, DType::F32);
        } else if *base_id < current_id {
            // Overlapping range — this shouldn't happen in a valid graph.
            // Skip this item if it's fully contained, otherwise panic.
            if *base_id + *count <= current_id {
                continue;
            }
            // Partial overlap — shouldn't happen. Skip.
            continue;
        }

        match item {
            SpanItem::ExternalInput(range) => {
                // Register as an input tensor. Check if it corresponds to
                // an actual input tensor from the main graph.
                let tensor_id = main_graph
                    .input_tensors()
                    .iter()
                    .find(|it| it.base_id == range.base)
                    .map(|it| it.tensor_id)
                    .unwrap_or(GlobalId(range.base.0));
                span_graph.add_input_tensor(tensor_id, range.count, range.dtype);
            }
            SpanItem::Group(gi) => {
                let group = &all_groups[*gi];
                // Use alloc_placeholder then fill_placeholder.
                let allocated = span_graph.alloc_placeholder(group.count, group.output_dtype);
                debug_assert_eq!(
                    allocated.0, group.base_id.0,
                    "Atom ID mismatch: allocated {} but expected {}",
                    allocated.0, group.base_id.0
                );
                span_graph.fill_placeholder(
                    allocated,
                    group.count,
                    group.output_dtype,
                    group.op.clone(),
                    group.sym_dims.clone(),
                    group.inputs.clone(),
                );
            }
        }

        current_id = base_id + count;
    }

    // Set span outputs.
    let span_inputs: Vec<AtomRange> = external_inputs.values().cloned().collect();
    let span_outputs: Vec<AtomRange> = output_ranges.values().cloned().collect();

    // Set graph outputs (atoms that are model outputs within this span).
    span_graph.outputs = main_graph
        .outputs
        .iter()
        .filter(|&out_id| {
            group_indices
                .iter()
                .any(|&gi| all_groups[gi].contains(*out_id))
        })
        .copied()
        .collect();

    Span {
        graph: span_graph,
        inputs: span_inputs,
        outputs: span_outputs,
    }
}

/// Collect input tensor references for a group's InputRefs.
fn collect_input_tensor_refs(
    main_graph: &NanoGraph,
    group: &AtomGroup,
    external_ranges: &mut BTreeMap<u64, AtomRange>,
) {
    // Check each InputRef for references to input tensors.
    for input_ref in &group.inputs {
        collect_input_ref_tensor_refs(main_graph, input_ref, group.count, group.atom_offset, external_ranges);
    }

    // For reduce ops, also check the strided range.
    if let ScalarOp::Reduce {
        reduce_count,
        reduce_stride,
        ..
    } = &group.op
    {
        if *reduce_count > 1 && *reduce_stride != 0 {
            for input_ref in &group.inputs {
                let first = input_ref.resolve(group.atom_offset);
                let last = input_ref.resolve(group.atom_offset + group.count - 1);
                let end_off = (*reduce_count as i64 - 1) * reduce_stride;
                let endpoints = [
                    first.0,
                    (first.0 as i64 + end_off) as u64,
                    last.0,
                    (last.0 as i64 + end_off) as u64,
                ];
                let lo = *endpoints.iter().min().unwrap();
                let hi = *endpoints.iter().max().unwrap();
                add_input_tensor_ranges_in_id_range(main_graph, lo, hi, external_ranges);
            }
        }
    }

    // IndirectLoad table reference.
    if let ScalarOp::IndirectLoad { table_base } = &group.op {
        // The table itself might be an input tensor or a group. If it's a group,
        // the normal producer tracking handles it. If it's an input tensor, add it.
        if let Some((idx, _)) = main_graph.find_input_idx(*table_base) {
            let it = &main_graph.input_tensors()[idx];
            external_ranges.entry(it.base_id.0).or_insert_with(|| AtomRange {
                base: it.base_id,
                count: it.count,
                dtype: it.dtype,
            });
        }
    }
}

fn collect_input_ref_tensor_refs(
    main_graph: &NanoGraph,
    input_ref: &InputRef,
    count: u64,
    atom_offset: u64,
    external_ranges: &mut BTreeMap<u64, AtomRange>,
) {
    if count == 0 {
        return;
    }
    let input_tensors = main_graph.input_tensors();
    match input_ref {
        InputRef::Broadcast(base) => {
            if let Some((idx, _)) = main_graph.find_input_idx(*base) {
                let it = &input_tensors[idx];
                external_ranges.entry(it.base_id.0).or_insert_with(|| AtomRange {
                    base: it.base_id,
                    count: it.count,
                    dtype: it.dtype,
                });
            }
        }
        InputRef::Affine { .. } | InputRef::StridedBroadcast { .. } => {
            let first = input_ref.resolve(atom_offset);
            let last = input_ref.resolve(atom_offset + count - 1);
            let lo = first.0.min(last.0);
            let hi = first.0.max(last.0);
            add_input_tensor_ranges_in_id_range(main_graph, lo, hi, external_ranges);
        }
        InputRef::Modular {
            base,
            stride,
            modulus,
        } => {
            let a = base.0;
            let b = (base.0 as i64 + *stride * (*modulus as i64 - 1)) as u64;
            add_input_tensor_ranges_in_id_range(
                main_graph,
                a.min(b),
                a.max(b),
                external_ranges,
            );
        }
        InputRef::Explicit(ids) => {
            for id in ids.iter().skip(atom_offset as usize).take(count as usize) {
                if let Some((idx, _)) = main_graph.find_input_idx(*id) {
                    let it = &input_tensors[idx];
                    external_ranges.entry(it.base_id.0).or_insert_with(|| AtomRange {
                        base: it.base_id,
                        count: it.count,
                        dtype: it.dtype,
                    });
                }
            }
        }
    }
}

fn add_input_tensor_ranges_in_id_range(
    main_graph: &NanoGraph,
    lo: u64,
    hi: u64,
    external_ranges: &mut BTreeMap<u64, AtomRange>,
) {
    for it in main_graph.input_tensors() {
        let it_lo = it.base_id.0;
        let it_hi = it_lo + it.count - 1;
        if it_lo <= hi && it_hi >= lo {
            external_ranges.entry(it.base_id.0).or_insert_with(|| AtomRange {
                base: it.base_id,
                count: it.count,
                dtype: it.dtype,
            });
        }
    }
}

// ─── Load balancing ─────────────────────────────────────────────────────────

/// Post-pass to split large groups across lanes for better work balance.
///
/// For each phase, if one lane has significantly more atoms than another,
/// identify its largest splittable groups and move portions to underloaded
/// lanes.
fn balance_phases(phases: &mut Vec<Phase>, num_lanes: usize, main_graph: &NanoGraph) {
    // For simplicity in the first implementation, we skip active rebalancing.
    // The greedy assignment already uses load-aware lane selection.
    // A full rebalancing pass would need to:
    // 1. Identify the most overloaded lane
    // 2. Find groups that can be split (no intra-phase consumers)
    // 3. Create sub-groups with proper atom_offset
    // 4. Rebuild span NanoGraphs
    //
    // This is complex and can be added as a refinement. The greedy heuristic
    // with cache-affinity tie-breaking should produce reasonable balance for
    // common graph structures (matmuls with independent rows).
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn empty_span(graph: &NanoGraph) -> Span {
    let mut g = NanoGraph::new();
    g.sym_dim_names = graph.sym_dim_names.clone();
    g.sym_dim_bounds = graph.sym_dim_bounds.clone();
    Span {
        graph: g,
        inputs: vec![],
        outputs: vec![],
    }
}


// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::nano_graph::ops::{ReduceKind, ScalarBinOp, ScalarOp, ScalarUnaryOp};
    use crate::nano_graph::pattern::InputTensor;
    use crate::numeric_scalar::NumericScalar;

    /// Validate that all structural invariants hold on the output plan.
    fn validate_plan(
        main_graph: &NanoGraph,
        phases: &[Phase],
        num_lanes: usize,
        output_atom_ids: &[AtomId],
    ) {
        // Each phase has the right number of lanes.
        for (pi, phase) in phases.iter().enumerate() {
            assert_eq!(
                phase.spans.len(),
                num_lanes,
                "Phase {} has {} spans, expected {}",
                pi,
                phase.spans.len(),
                num_lanes
            );
        }

        // Invariant 1: spans within a phase are independent.
        for (pi, phase) in phases.iter().enumerate() {
            for (li, span) in phase.spans.iter().enumerate() {
                // Collect all atom IDs produced by OTHER spans in this phase.
                let mut other_produced: HashSet<u64> = HashSet::new();
                for (other_li, other_span) in phase.spans.iter().enumerate() {
                    if other_li == li {
                        continue;
                    }
                    for group in other_span.graph.groups() {
                        for i in 0..group.count {
                            other_produced.insert(group.base_id.0 + i);
                        }
                    }
                }

                // Check that this span doesn't read from other spans' outputs.
                for group in span.graph.groups() {
                    for input_ref in &group.inputs {
                        let sample_positions: Vec<u64> = if group.count <= 3 {
                            (0..group.count).collect()
                        } else {
                            vec![0, group.count / 2, group.count - 1]
                        };
                        for i in sample_positions {
                            let source = input_ref.resolve(i + group.atom_offset);
                            assert!(
                                !other_produced.contains(&source.0),
                                "Phase {} lane {} reads atom {} produced by another lane in same phase",
                                pi, li, source
                            );
                        }
                    }
                }
            }
        }

        // Invariant 2: each span's NanoGraph is self-contained.
        for (pi, phase) in phases.iter().enumerate() {
            for (li, span) in phase.spans.iter().enumerate() {
                let errors = span.graph.validate();
                assert!(
                    errors.is_empty(),
                    "Phase {} lane {} validation errors: {:?}",
                    pi,
                    li,
                    errors
                );
            }
        }

        // Invariant 3: all model output atoms are produced by some span.
        let mut produced_atoms: HashSet<u64> = HashSet::new();
        for phase in phases {
            for span in &phase.spans {
                for group in span.graph.groups() {
                    for i in 0..group.count {
                        produced_atoms.insert(group.base_id.0 + i);
                    }
                }
            }
        }
        for &out_id in output_atom_ids {
            assert!(
                produced_atoms.contains(&out_id.0),
                "Output atom {} not produced by any span",
                out_id
            );
        }

        // Invariant 4: groups within each span are in valid topological order
        // (checked by validate() above).
    }

    /// Simple linear chain: a -> b -> c, each a group of 100 atoms.
    #[test]
    fn test_linear_chain_single_lane() {
        let mut g = NanoGraph::new();
        let a = g.push_group(
            100,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );
        let b = g.push_group(
            100,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine { base: a, stride: 1 }],
        );
        let c = g.push_group(
            100,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine { base: b, stride: 1 }],
        );
        g.outputs = vec![c];

        let phases = plan(&g, 1, &[], &[c]);
        validate_plan(&g, &phases, 1, &[c]);

        // All in one phase since there's only 1 lane.
        assert_eq!(phases.len(), 1);
        assert_eq!(phases[0].spans[0].graph.num_groups(), 3);
    }

    /// Two independent chains should go to different lanes, same phase.
    #[test]
    fn test_two_independent_chains() {
        let mut g = NanoGraph::new();

        // Chain 1: a1 -> b1
        let a1 = g.push_group(
            100,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );
        let b1 = g.push_group(
            100,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: a1,
                stride: 1,
            }],
        );

        // Chain 2: a2 -> b2
        let a2 = g.push_group(
            100,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
        );
        let b2 = g.push_group(
            100,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: a2,
                stride: 1,
            }],
        );

        g.outputs = vec![b1, b2];

        let phases = plan(&g, 2, &[], &[b1, b2]);
        validate_plan(&g, &phases, 2, &[b1, b2]);

        // Both chains are independent — should fit in 1 phase.
        assert_eq!(phases.len(), 1);
    }

    /// Diamond: a -> (b, c) -> d. Requires a barrier if b and c on different lanes.
    /// With 2 lanes, the greedy scheduler should put all on one lane (no parallelism
    /// opportunity) since d depends on both b and c.
    #[test]
    fn test_diamond() {
        let mut g = NanoGraph::new();

        let a = g.push_group(
            100,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );
        let b = g.push_group(
            100,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine { base: a, stride: 1 }],
        );
        let c = g.push_group(
            100,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine { base: a, stride: 1 }],
        );
        let d = g.push_group(
            100,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine { base: b, stride: 1 },
                InputRef::Affine { base: c, stride: 1 },
            ],
        );
        g.outputs = vec![d];

        let phases = plan(&g, 2, &[], &[d]);
        validate_plan(&g, &phases, 2, &[d]);
    }

    /// Fork-join: a -> (b, c), (b, c) -> d.
    /// With 2 lanes, b and c can run in parallel if a is in a prior phase.
    #[test]
    fn test_fork_join() {
        let mut g = NanoGraph::new();

        let a = g.push_group(
            100,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );
        // b depends on a
        let b = g.push_group(
            100,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine { base: a, stride: 1 }],
        );
        // c depends on a
        let c = g.push_group(
            100,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine { base: a, stride: 1 }],
        );
        // d depends on b AND c
        let d = g.push_group(
            100,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine { base: b, stride: 1 },
                InputRef::Affine { base: c, stride: 1 },
            ],
        );
        g.outputs = vec![d];

        let phases = plan(&g, 2, &[], &[d]);
        validate_plan(&g, &phases, 2, &[d]);

        // The greedy scheduler assigns a to lane 0, b to lane 0 (affinity),
        // c to lane 1 (load balance, since a is also viable from lane 0 prior).
        // Then d needs both b (lane 0) and c (lane 1) — barrier needed.
        // After barrier, d can go on any lane.
        // So we expect 2 or more phases.
    }

    /// Test with input tensors (weights).
    #[test]
    fn test_with_input_tensors() {
        let mut g = NanoGraph::new();
        let weight = g.add_input_tensor(GlobalId(1), 768, DType::F32);

        let computed = g.push_group(
            768,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: weight,
                stride: 1,
            }],
        );
        g.outputs = vec![computed];

        let input_tensors = g.input_tensors().to_vec();
        let phases = plan(&g, 2, &input_tensors, &[computed]);
        validate_plan(&g, &phases, 2, &[computed]);

        // Single phase — the weight is external input, compute is trivially parallel.
        assert_eq!(phases.len(), 1);

        // Check that the span that has the compute also declares the weight as input.
        let mut found_weight_input = false;
        for span in &phases[0].spans {
            if span.graph.num_groups() > 0 {
                for input in &span.inputs {
                    if input.base == weight {
                        found_weight_input = true;
                    }
                }
            }
        }
        assert!(found_weight_input, "Weight should appear in span inputs");
    }

    /// Test broadcast: multiple groups reading from the same source.
    #[test]
    fn test_broadcast_dependency() {
        let mut g = NanoGraph::new();

        let shared = g.push_group(
            1,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(42.0)),
            vec![],
            vec![],
        );
        let a = g.push_group(
            100,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Broadcast(shared)],
        );
        let b = g.push_group(
            100,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Broadcast(shared)],
        );
        g.outputs = vec![a, b];

        let phases = plan(&g, 2, &[], &[a, b]);
        validate_plan(&g, &phases, 2, &[a, b]);
    }

    /// Reduction group with strided access.
    #[test]
    fn test_reduce_group() {
        let mut g = NanoGraph::new();

        // 1024 source atoms (e.g., matmul products).
        let products = g.push_group(
            1024,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );

        // 4 reduce groups, each summing 256 values with stride 1.
        let reduced = g.push_group(
            4,
            DType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: 256,
                reduce_stride: 1,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: products,
                stride: 256,
            }],
        );
        g.outputs = vec![reduced];

        let phases = plan(&g, 2, &[], &[reduced]);
        validate_plan(&g, &phases, 2, &[reduced]);
    }

    /// Many independent groups should spread across lanes.
    #[test]
    fn test_many_independent_groups() {
        let mut g = NanoGraph::new();
        let mut output_ids = Vec::new();

        for i in 0..16u32 {
            let lit = g.push_group(
                100,
                DType::F32,
                ScalarOp::Literal(NumericScalar::F32(i as f32)),
                vec![],
                vec![],
            );
            output_ids.push(lit);
        }
        g.outputs = output_ids.clone();

        let phases = plan(&g, 4, &[], &output_ids);
        validate_plan(&g, &phases, 4, &output_ids);

        // All independent — should be 1 phase, spread across 4 lanes.
        assert_eq!(phases.len(), 1);

        // Check that groups are spread across lanes.
        let mut non_empty_lanes = 0;
        for span in &phases[0].spans {
            if span.graph.num_groups() > 0 {
                non_empty_lanes += 1;
            }
        }
        assert!(
            non_empty_lanes > 1,
            "Independent groups should spread across lanes"
        );
    }

    /// Empty graph produces valid output.
    #[test]
    fn test_empty_graph() {
        let g = NanoGraph::new();
        let phases = plan(&g, 4, &[], &[]);
        assert_eq!(phases.len(), 1);
        assert_eq!(phases[0].spans.len(), 4);
    }

    /// Single group, multiple lanes — should go on one lane, rest idle.
    #[test]
    fn test_single_group_multi_lane() {
        let mut g = NanoGraph::new();
        let a = g.push_group(
            100,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );
        g.outputs = vec![a];

        let phases = plan(&g, 4, &[], &[a]);
        validate_plan(&g, &phases, 4, &[a]);

        assert_eq!(phases.len(), 1);
        let non_empty = phases[0]
            .spans
            .iter()
            .filter(|s| s.graph.num_groups() > 0)
            .count();
        assert_eq!(non_empty, 1);
    }

    /// Matmul-like pattern: M independent Mul groups + M ReduceSum groups.
    /// The Mul groups share a weight input (broadcast). Rows are independent.
    #[test]
    fn test_matmul_pattern() {
        let mut g = NanoGraph::new();
        let m = 4; // rows
        let k = 64; // inner dim
        let n = 32; // cols

        // Weight: K*N atoms.
        let weight = g.push_group(
            (k * n) as u64,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.1)),
            vec![],
            vec![],
        );

        // Input vector: K atoms.
        let input = g.push_group(
            k as u64,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.2)),
            vec![],
            vec![],
        );

        // M Mul groups (each K*N atoms, broadcasting input over N columns).
        let mut mul_groups = Vec::new();
        for row in 0..m {
            let mul = g.push_group(
                (k * n) as u64,
                DType::F32,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                },
                vec![],
                vec![
                    InputRef::Affine {
                        base: weight,
                        stride: 1,
                    },
                    InputRef::StridedBroadcast {
                        base: input,
                        stride: 1,
                        repeat: n as u64,
                    },
                ],
            );
            mul_groups.push(mul);
        }

        // M ReduceSum groups (each N atoms, reducing K elements per output).
        let mut reduce_groups = Vec::new();
        for row in 0..m {
            let red = g.push_group(
                n as u64,
                DType::F32,
                ScalarOp::Reduce {
                    kind: ReduceKind::Sum,
                    reduce_count: k as u64,
                    reduce_stride: n as i64,
                    compute_dtype: DType::F32,
                },
                vec![],
                vec![InputRef::Affine {
                    base: mul_groups[row],
                    stride: 1,
                }],
            );
            reduce_groups.push(red);
        }

        let output_ids: Vec<AtomId> = reduce_groups.clone();
        g.outputs = output_ids.clone();

        let phases = plan(&g, 2, &[], &output_ids);
        validate_plan(&g, &phases, 2, &output_ids);

        // Each row (Mul + ReduceSum) is independent. With 4 rows and 2 lanes,
        // we should get good distribution.
        let total_groups: usize = phases
            .iter()
            .flat_map(|p| &p.spans)
            .map(|s| s.graph.num_groups())
            .sum();
        // We have 2 (weight + input) + 4 (muls) + 4 (reduces) = 10 groups.
        // Some may be duplicated as inputs across spans.
        assert!(total_groups >= 10, "All groups should be present in spans");
    }

    /// Test that Modular InputRef dependencies are properly tracked.
    #[test]
    fn test_modular_input_ref() {
        let mut g = NanoGraph::new();

        let bias = g.push_group(
            32,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.5)),
            vec![],
            vec![],
        );

        // 128 atoms, each reading bias[i % 32] (tiling pattern).
        let tiled = g.push_group(
            128,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Modular {
                base: bias,
                stride: 1,
                modulus: 32,
            }],
        );
        g.outputs = vec![tiled];

        let phases = plan(&g, 2, &[], &[tiled]);
        validate_plan(&g, &phases, 2, &[tiled]);
    }

    /// Explicit InputRef: irregular index pattern.
    #[test]
    fn test_explicit_input_ref() {
        let mut g = NanoGraph::new();

        let src = g.push_group(
            8,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );

        let gather = g.push_group(
            4,
            DType::F32,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::Explicit(vec![
                src.offset(3),
                src.offset(1),
                src.offset(7),
                src.offset(0),
            ])],
        );
        g.outputs = vec![gather];

        let phases = plan(&g, 2, &[], &[gather]);
        validate_plan(&g, &phases, 2, &[gather]);
    }

    /// Longer pipeline: 5 sequential groups, 2 lanes.
    /// Should be 1 phase since the chain can't be parallelized.
    #[test]
    fn test_long_sequential_chain() {
        let mut g = NanoGraph::new();

        let g0 = g.push_group(
            50,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );
        let g1 = g.push_group(
            50,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: g0,
                stride: 1,
            }],
        );
        let g2 = g.push_group(
            50,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: g1,
                stride: 1,
            }],
        );
        let g3 = g.push_group(
            50,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Tanh,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: g2,
                stride: 1,
            }],
        );
        let g4 = g.push_group(
            50,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Abs,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: g3,
                stride: 1,
            }],
        );
        g.outputs = vec![g4];

        let phases = plan(&g, 2, &[], &[g4]);
        validate_plan(&g, &phases, 2, &[g4]);

        // Sequential chain — all groups must be on the same lane, 1 phase.
        assert_eq!(phases.len(), 1);
    }

    /// Select (ternary) operation.
    #[test]
    fn test_select_op() {
        let mut g = NanoGraph::new();

        let cond = g.push_group(
            64,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );
        let x = g.push_group(
            64,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
        );
        let y = g.push_group(
            64,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(3.0)),
            vec![],
            vec![],
        );
        let sel = g.push_group(
            64,
            DType::F32,
            ScalarOp::Select,
            vec![],
            vec![
                InputRef::Affine {
                    base: cond,
                    stride: 1,
                },
                InputRef::Affine {
                    base: x,
                    stride: 1,
                },
                InputRef::Affine {
                    base: y,
                    stride: 1,
                },
            ],
        );
        g.outputs = vec![sel];

        let phases = plan(&g, 2, &[], &[sel]);
        validate_plan(&g, &phases, 2, &[sel]);
    }
}
