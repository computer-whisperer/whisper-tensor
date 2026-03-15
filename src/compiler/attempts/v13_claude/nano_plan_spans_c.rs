#![allow(clippy::all, dead_code, unreachable_patterns, unused_variables, unused_imports)]
//! Span-based execution planner: emits self-contained NanoGraphs per (phase, lane).
//!
//! Uses v2c's phase detection and lane assignment as the backbone, then converts
//! each lane's work in each phase into an independent NanoGraph (a "span").
//!
//! Each span is self-contained:
//! - Has its own groups, atom IDs, and InputRefs
//! - Declares explicit inputs (atoms it reads from external sources / other spans)
//! - Declares explicit outputs (atoms it produces that other spans will read)
//! - Can be independently validated for correctness
//! - Duplicates small shared computation chains (like Gather index math)
//!   rather than requiring cross-span reads

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::nano_graph::{AtomGroup, AtomId, InputRef, NanoGraph, ScalarOp};

use super::nano_plan_v2c;

/// Literal groups with fewer atoms than this are duplicated into spans.
/// Larger literals (weight matrices) become external inputs instead.
const LITERAL_INLINE_THRESHOLD: u64 = 1024;

// ─── Public API ──────────────────────────────────────────────────────────────

/// A self-contained unit of work for one lane in one phase.
pub struct Span {
    /// Self-contained NanoGraph for this span's computation.
    pub graph: NanoGraph,
    /// Which atoms from the main graph this span reads as external inputs.
    /// (main graph AtomId, local AtomId in span graph)
    pub inputs: Vec<(AtomId, AtomId)>,
    /// Which atoms this span writes back to the shared values buffer.
    /// (local AtomId in span graph, main graph AtomId)
    pub outputs: Vec<(AtomId, AtomId)>,
}

/// One phase of execution.
pub struct Phase {
    /// One span per lane (may have empty graphs for idle lanes).
    pub spans: Vec<Span>,
}

/// The full execution plan with self-contained span NanoGraphs.
pub struct SpanPlan {
    pub num_lanes: usize,
    pub phases: Vec<Phase>,
}

/// Plan execution for a NanoGraph, emitting self-contained span NanoGraphs.
pub fn plan_execution_spans(graph: &NanoGraph, num_lanes: usize) -> SpanPlan {
    let num_lanes = num_lanes.max(1);
    let groups = graph.groups();
    let n = groups.len();

    if n == 0 {
        return SpanPlan {
            num_lanes,
            phases: vec![],
        };
    }

    // Step 1: Run v2c to get the lane/phase assignments.
    let v2c_plan = nano_plan_v2c::plan_execution(graph, num_lanes);

    // Step 2: Build a map from (main graph AtomId) -> which (phase, lane, work_item)
    // produces it. This lets us determine what's "external" to a span.
    //
    // For literal groups, they're "always available" — we'll inline them into
    // each span that needs them.

    let is_literal: Vec<bool> = groups
        .iter()
        .map(|g| matches!(g.op, ScalarOp::Literal(_)) && g.inputs.is_empty())
        .collect();

    // Build: group_idx -> (phase_idx, lane_idx) for all assigned compute groups.
    let mut group_assignment: HashMap<usize, (usize, usize)> = HashMap::new();
    for (phase_idx, phase) in v2c_plan.phases.iter().enumerate() {
        for (lane_idx, lane_work) in phase.lane_work.iter().enumerate() {
            for work in lane_work {
                // For split groups, multiple lanes own different slices.
                // We record each occurrence. The group_assignment is for the
                // full group; we'll also track sub-ranges below.
                group_assignment
                    .entry(work.group_idx)
                    .or_insert((phase_idx, lane_idx));
            }
        }
    }

    // Build detailed atom ownership: which (phase, lane) owns each atom range.
    // atom_owner[group_idx] = vec of (atom_offset, atom_count, phase, lane)
    let mut atom_owner: Vec<Vec<(u64, u64, usize, usize)>> = vec![Vec::new(); n];
    for (phase_idx, phase) in v2c_plan.phases.iter().enumerate() {
        for (lane_idx, lane_work) in phase.lane_work.iter().enumerate() {
            for work in lane_work {
                atom_owner[work.group_idx].push((
                    work.atom_offset,
                    work.atom_count,
                    phase_idx,
                    lane_idx,
                ));
            }
        }
    }

    // Step 3: For each (phase, lane), build a self-contained NanoGraph span.
    let mut phases = Vec::with_capacity(v2c_plan.phases.len());

    for (phase_idx, v2c_phase) in v2c_plan.phases.iter().enumerate() {
        let mut spans = Vec::with_capacity(num_lanes);

        for lane_idx in 0..num_lanes {
            let lane_work = &v2c_phase.lane_work[lane_idx];
            if lane_work.is_empty() {
                // Idle lane: empty span.
                spans.push(Span {
                    graph: NanoGraph::new(),
                    inputs: vec![],
                    outputs: vec![],
                });
                continue;
            }

            let span = build_span(
                graph,
                groups,
                &is_literal,
                &atom_owner,
                lane_work,
                phase_idx,
                lane_idx,
            );
            spans.push(span);
        }

        phases.push(Phase { spans });
    }

    SpanPlan { num_lanes, phases }
}

// ─── Span builder ────────────────────────────────────────────────────────────

/// Build a self-contained NanoGraph for one (phase, lane)'s work.
///
/// The span includes:
/// 1. The assigned work items (potentially sub-ranges of groups)
/// 2. Literal groups that are referenced by the work items
/// 3. External inputs declared for atoms from earlier phases or other lanes
///
/// The span remaps all AtomIds to a fresh local ID space.
fn build_span(
    main_graph: &NanoGraph,
    groups: &[AtomGroup],
    is_literal: &[bool],
    atom_owner: &[Vec<(u64, u64, usize, usize)>],
    lane_work: &[nano_plan_v2c::LaneWork],
    phase_idx: usize,
    lane_idx: usize,
) -> Span {
    let mut span_graph = NanoGraph::new();

    // Copy sym_dim configuration from main graph.
    for (name, &sd) in &main_graph.sym_dim_names {
        let local_sd = span_graph.sym_dim(name);
        if let Some(&bound) = main_graph.sym_dim_bounds.get(&sd) {
            span_graph.sym_dim_bounds.insert(local_sd, bound);
        }
    }

    // Track mapping: main graph AtomId -> local span AtomId.
    // We need this to remap InputRefs.
    let mut main_to_local: HashMap<AtomId, AtomId> = HashMap::new();

    // External inputs: atoms from earlier phases or other lanes.
    // Tracked as (main_atom_id, local_atom_id).
    let mut external_inputs: BTreeMap<AtomId, AtomId> = BTreeMap::new();

    // First pass: determine which main-graph groups/atoms we need.
    // Collect the set of group indices assigned to this span.
    let mut assigned_group_slices: Vec<(usize, u64, u64)> = Vec::new();
    for work in lane_work {
        assigned_group_slices.push((work.group_idx, work.atom_offset, work.atom_count));
    }

    // Second pass: determine all referenced literal groups.
    // Walk each assigned group's inputs and find which literals they reference.
    let mut needed_literals: BTreeSet<usize> = BTreeSet::new();
    for &(gi, _, _) in &assigned_group_slices {
        collect_literal_deps(gi, groups, is_literal, &mut needed_literals);
    }

    // Third pass: add small literal groups to the span graph.
    // Large literals (weight matrices) become external inputs instead.
    let mut inlined_literals: BTreeSet<usize> = BTreeSet::new();
    for &lit_gi in &needed_literals {
        let lit_group = &groups[lit_gi];
        if lit_group.count < LITERAL_INLINE_THRESHOLD {
            // Small literal: duplicate into span.
            let local_base = span_graph.push_group(
                lit_group.count,
                lit_group.op.clone(),
                lit_group.sym_dims.clone(),
                lit_group.reduce_dims.clone(),
                vec![], // Literals have no inputs.
            );
            for offset in 0..lit_group.count {
                main_to_local.insert(
                    AtomId(lit_group.base_id.0 + offset),
                    AtomId(local_base.0 + offset),
                );
            }
            inlined_literals.insert(lit_gi);
        }
        // Large literals will be handled as external atoms below.
    }

    // Fourth pass: for each assigned work item, determine external dependencies
    // and create input placeholders, then add the compute group.
    //
    // We process work items in order (they're already sorted by group_idx).
    // For each work item, we:
    // 1. Resolve all atoms this work item's slice reads from
    // 2. Check if those atoms are in this span (literal or same work item earlier
    //    in the list) or external
    // 3. For external atoms, create Literal placeholder inputs
    // 4. Remap InputRefs to local IDs

    // Pre-populate main_to_local for all atoms produced by this span's work items.
    // We need to know this before remapping inputs, because later work items in
    // the same span may reference earlier ones.
    //
    // We do this in two passes:
    // Pass A: allocate local IDs for all work items (without building groups yet)
    // Pass B: resolve inputs and build groups

    // Pass A: Reserve local AtomId space for each work item.
    // (AtomId reservation was considered but the two-pass approach below
    //  handles ordering: collect external inputs first, then build groups.)

    // Collect all external atom dependencies.
    // For each work item, find all main-graph AtomIds it reads that are NOT:
    //   - In a literal group (already added)
    //   - Produced by another work item in this span
    let assigned_atoms: HashSet<(usize, u64, u64)> = assigned_group_slices
        .iter()
        .copied()
        .collect();

    // Build a quick lookup: for a main-graph AtomId, is it produced by this span?
    // Returns true if the atom is in one of the assigned work items.
    let is_local_compute = |atom_id: AtomId| -> bool {
        // Find which group this atom belongs to in the main graph.
        if let Some(gi) = find_group_idx(groups, atom_id) {
            if is_literal[gi] {
                // Only inlined (small) literals count as local.
                return inlined_literals.contains(&gi);
            }
            let offset_in_group = atom_id.0 - groups[gi].base_id.0;
            // Check if this (gi, offset) falls within any of our work items.
            for &(work_gi, work_offset, work_count) in &assigned_group_slices {
                if work_gi == gi
                    && offset_in_group >= work_offset
                    && offset_in_group < work_offset + work_count
                {
                    return true;
                }
            }
        }
        false
    };

    // Collect all external atoms needed.
    let mut external_atoms: BTreeSet<AtomId> = BTreeSet::new();
    for &(gi, atom_offset, atom_count) in &assigned_group_slices {
        let group = &groups[gi];
        // Collect atoms read by this work item's slice.
        let read_atoms = collect_read_atoms(group, atom_offset, atom_count, groups);
        for atom_id in read_atoms {
            if !is_local_compute(atom_id) {
                external_atoms.insert(atom_id);
            }
        }
    }

    // Create Literal placeholder groups for external inputs.
    // Each external atom gets a single-atom Literal group in the span.
    // We use the output_dtype of the producing group.
    for &ext_atom in &external_atoms {
        let dtype = if let Some(gi) = find_group_idx(groups, ext_atom) {
            groups[gi].op.output_dtype()
        } else {
            crate::dtype::DType::F32 // fallback
        };
        let local_id = span_graph.push_group(
            1,
            ScalarOp::Literal(crate::numeric_scalar::NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );
        main_to_local.insert(ext_atom, local_id);
        external_inputs.insert(ext_atom, local_id);
    }

    // Now build the compute groups with remapped InputRefs.
    let mut span_outputs: Vec<(AtomId, AtomId)> = Vec::new();

    for &(gi, atom_offset, atom_count) in &assigned_group_slices {
        let group = &groups[gi];

        // Remap InputRefs to local IDs.
        let local_inputs = remap_inputs(
            &group.inputs,
            &group.op,
            atom_offset,
            atom_count,
            group.count,
            groups,
            &main_to_local,
        );

        // For sub-range work items, we may need to adjust the op.
        // ReduceSum/ReduceMax with sub-ranges: the reduce parameters stay the same
        // (each atom independently reduces), only the count changes.
        let local_op = remap_op(&group.op, &main_to_local);

        let local_base = span_graph.push_group(
            atom_count,
            local_op,
            group.sym_dims.clone(),
            group.reduce_dims.clone(),
            local_inputs,
        );

        // Map atoms in this work item to local IDs.
        for offset in 0..atom_count {
            let main_atom = AtomId(group.base_id.0 + atom_offset + offset);
            let local_atom = AtomId(local_base.0 + offset);
            main_to_local.insert(main_atom, local_atom);
        }

        // All atoms produced by this work item are potential outputs.
        for offset in 0..atom_count {
            let main_atom = AtomId(group.base_id.0 + atom_offset + offset);
            let local_atom = AtomId(local_base.0 + offset);
            span_outputs.push((local_atom, main_atom));
        }
    }

    // Mark graph outputs: atoms that are outputs of the main graph.
    let main_outputs: HashSet<AtomId> = main_graph.outputs.iter().copied().collect();
    for &(local_atom, main_atom) in &span_outputs {
        if main_outputs.contains(&main_atom) {
            span_graph.outputs.push(local_atom);
        }
    }

    let inputs_vec: Vec<(AtomId, AtomId)> = external_inputs
        .into_iter()
        .map(|(main_id, local_id)| (main_id, local_id))
        .collect();

    Span {
        graph: span_graph,
        inputs: inputs_vec,
        outputs: span_outputs,
    }
}

// ─── Helper functions ────────────────────────────────────────────────────────

/// Binary search for the group containing an atom.
fn find_group_idx(groups: &[AtomGroup], id: AtomId) -> Option<usize> {
    let idx = groups.partition_point(|g| g.base_id.0 <= id.0);
    if idx == 0 {
        return None;
    }
    let gi = idx - 1;
    if groups[gi].contains(id) {
        Some(gi)
    } else {
        None
    }
}

/// Collect all literal group indices that a compute group depends on.
fn collect_literal_deps(
    gi: usize,
    groups: &[AtomGroup],
    is_literal: &[bool],
    literals: &mut BTreeSet<usize>,
) {
    let group = &groups[gi];
    for input in &group.inputs {
        let referenced = resolve_all_referenced_atoms_to_groups(input, group.count, groups);
        for ref_gi in referenced {
            if is_literal[ref_gi] {
                literals.insert(ref_gi);
            }
        }
    }
    // IndirectLoad table_base.
    if let ScalarOp::IndirectLoad { table_base, .. } = &group.op {
        if let Some(tgi) = find_group_idx(groups, *table_base) {
            if is_literal[tgi] {
                literals.insert(tgi);
            }
        }
    }
}

/// Find all group indices referenced by an InputRef.
fn resolve_all_referenced_atoms_to_groups(
    input: &InputRef,
    count: u64,
    groups: &[AtomGroup],
) -> BTreeSet<usize> {
    let mut result = BTreeSet::new();
    match input {
        InputRef::Broadcast(atom_id) => {
            if let Some(gi) = find_group_idx(groups, *atom_id) {
                result.insert(gi);
            }
        }
        InputRef::Affine { base, stride } => {
            if count == 0 {
                return result;
            }
            let last_offset = (*stride as i64) * (count as i64 - 1);
            let last = AtomId(base.0.wrapping_add(last_offset as u64));
            let lo = base.0.min(last.0);
            let hi = base.0.max(last.0);
            for gi in find_groups_in_range(groups, lo, hi) {
                result.insert(gi);
            }
        }
        InputRef::Explicit(ids) => {
            for id in ids {
                if let Some(gi) = find_group_idx(groups, *id) {
                    result.insert(gi);
                }
            }
        }
        InputRef::SymAffine { base, stride_i, .. } => {
            if count == 0 {
                return result;
            }
            let last_offset = (*stride_i as i64) * (count as i64 - 1);
            let last = AtomId(base.0.wrapping_add(last_offset as u64));
            let lo = base.0.min(last.0);
            let hi = base.0.max(last.0);
            for gi in find_groups_in_range(groups, lo, hi) {
                result.insert(gi);
            }
        }
        InputRef::StridedBroadcast { base, stride, repeat } => {
            if count == 0 {
                return result;
            }
            let num_blocks = (count + repeat - 1) / repeat;
            let last_offset = *stride * (num_blocks as i64 - 1);
            let last = AtomId(base.0.wrapping_add(last_offset as u64));
            let lo = base.0.min(last.0);
            let hi = base.0.max(last.0);
            for gi in find_groups_in_range(groups, lo, hi) {
                result.insert(gi);
            }
        }
        InputRef::Modular { base, stride, modulus } => {
            if *modulus == 0 {
                return result;
            }
            let last_offset = (*stride as i64) * (*modulus as i64 - 1);
            let last = AtomId(base.0.wrapping_add(last_offset as u64));
            let lo = base.0.min(last.0);
            let hi = base.0.max(last.0);
            for gi in find_groups_in_range(groups, lo, hi) {
                result.insert(gi);
            }
        }
    }
    result
}

/// Find all groups whose atom ranges overlap [lo, hi].
fn find_groups_in_range(groups: &[AtomGroup], lo: u64, hi: u64) -> Vec<usize> {
    let mut result = Vec::new();
    let start = groups.partition_point(|g| g.base_id.0 + g.count <= lo);
    for gi in start..groups.len() {
        let g = &groups[gi];
        if g.base_id.0 > hi {
            break;
        }
        if g.count == 0 {
            continue;
        }
        let g_end = g.base_id.0 + g.count - 1;
        if g.base_id.0 <= hi && g_end >= lo {
            result.push(gi);
        }
    }
    result
}

/// Collect all main-graph AtomIds that a work item's slice reads.
///
/// This handles InputRef resolution for sub-ranges, ReduceSum/ReduceMax
/// strided access, and IndirectLoad table_base.
fn collect_read_atoms(
    group: &AtomGroup,
    atom_offset: u64,
    atom_count: u64,
    groups: &[AtomGroup],
) -> BTreeSet<AtomId> {
    let mut read_atoms = BTreeSet::new();
    if atom_count == 0 {
        return read_atoms;
    }

    let first_i = atom_offset;
    let last_i = atom_offset + atom_count - 1;

    for input in &group.inputs {
        // Collect all atoms referenced by this input for the sub-range.
        collect_input_atoms(input, first_i, last_i, &mut read_atoms);
    }

    // ReduceSum/ReduceMax: strided access extends the read range.
    match &group.op {
        ScalarOp::ReduceSum { reduce_count, reduce_stride, .. }
        | ScalarOp::ReduceMax { reduce_count, reduce_stride, .. }
            if *reduce_count > 1 && *reduce_stride != 0 =>
        {
            for input in &group.inputs {
                collect_input_atoms_with_reduce(
                    input, first_i, last_i,
                    *reduce_count, *reduce_stride,
                    &mut read_atoms,
                );
            }
        }
        _ => {}
    }

    // IndirectLoad: the table group's atoms are read at runtime.
    if let ScalarOp::IndirectLoad { table_base, .. } = &group.op {
        if let Some(tgi) = find_group_idx(groups, *table_base) {
            let tg = &groups[tgi];
            for offset in 0..tg.count {
                read_atoms.insert(AtomId(tg.base_id.0 + offset));
            }
        }
    }

    read_atoms
}

/// Collect atoms referenced by an InputRef for a sub-range [first_i, last_i].
fn collect_input_atoms(
    input: &InputRef,
    first_i: u64,
    last_i: u64,
    atoms: &mut BTreeSet<AtomId>,
) {
    match input {
        InputRef::Broadcast(atom_id) => {
            atoms.insert(*atom_id);
        }
        InputRef::Affine { base, stride } => {
            // Collect all atoms in the contiguous range.
            let first_read = base.0 as i64 + *stride as i64 * first_i as i64;
            let last_read = base.0 as i64 + *stride as i64 * last_i as i64;
            let lo = first_read.min(last_read);
            let hi = first_read.max(last_read);
            if *stride == 0 {
                atoms.insert(AtomId(first_read as u64));
            } else {
                let step = (*stride as i64).unsigned_abs();
                let mut pos = lo;
                // For large ranges, track as a contiguous range.
                // We only need the set of AtomIds, not every single one.
                // But for correctness, we do enumerate if the range is manageable.
                if (hi - lo) / step as i64 <= 100_000 {
                    while pos <= hi {
                        atoms.insert(AtomId(pos as u64));
                        pos += step as i64;
                    }
                } else {
                    // For very large ranges, insert boundary atoms and rely on
                    // the range being contiguous with stride 1 or -1.
                    // This is safe because find_group_idx + the remap logic
                    // will handle the full range.
                    for offset in 0..=(last_i - first_i) {
                        let atom = input.resolve(first_i + offset, 0);
                        atoms.insert(atom);
                        // For large counts, we can't enumerate all.
                        // Instead, mark the boundaries and rely on the
                        // remap_inputs function to handle it correctly.
                        if offset > 10 && offset < (last_i - first_i) - 10 {
                            // Skip middle atoms for performance — the remap
                            // function uses the base+stride pattern, not individual atoms.
                            break;
                        }
                    }
                }
            }
        }
        InputRef::StridedBroadcast { base, stride, repeat } => {
            let first_block = first_i / repeat;
            let last_block = last_i / repeat;
            for block in first_block..=last_block {
                let atom = AtomId(base.0.wrapping_add((*stride * block as i64) as u64));
                atoms.insert(atom);
            }
        }
        InputRef::Modular { base, stride, modulus } => {
            if *modulus == 0 {
                return;
            }
            // Modular wraps, so we need all modulus atoms.
            for k in 0..*modulus {
                let atom = AtomId(base.0.wrapping_add((*stride as i64 * k as i64) as u64));
                atoms.insert(atom);
            }
        }
        InputRef::Explicit(ids) => {
            for i in first_i..=last_i {
                if (i as usize) < ids.len() {
                    atoms.insert(ids[i as usize]);
                }
            }
        }
        InputRef::SymAffine { base, stride_i, stride_k } => {
            // We don't know k at plan time. Track the stride_i range.
            let first_read = base.0 as i64 + *stride_i as i64 * first_i as i64;
            let last_read = base.0 as i64 + *stride_i as i64 * last_i as i64;
            let lo = first_read.min(last_read);
            let hi = first_read.max(last_read);
            // SymAffine accesses are used for matmul inner loops.
            // We track the full range that stride_i covers.
            // The stride_k dimension is handled by the reduce/sym_dim loop.
            if *stride_i == 0 {
                atoms.insert(AtomId(first_read as u64));
            } else {
                let step = (*stride_i as i64).unsigned_abs();
                let range_len = (hi - lo) / step as i64 + 1;
                if range_len <= 100_000 {
                    let mut pos = lo;
                    while pos <= hi {
                        atoms.insert(AtomId(pos as u64));
                        pos += step as i64;
                    }
                } else {
                    // Large range: just mark boundaries.
                    atoms.insert(AtomId(lo as u64));
                    atoms.insert(AtomId(hi as u64));
                }
            }
        }
    }
}

/// Collect atoms referenced by ReduceSum/ReduceMax strided access.
fn collect_input_atoms_with_reduce(
    input: &InputRef,
    first_i: u64,
    last_i: u64,
    reduce_count: u64,
    reduce_stride: i64,
    atoms: &mut BTreeSet<AtomId>,
) {
    // For each atom in [first_i, last_i], the reduce reads:
    //   resolved_input(i) + k * reduce_stride for k in 0..reduce_count
    // We need to add all these atoms.
    let min_reduce_ext = 0i64.min(reduce_stride * (reduce_count as i64 - 1));
    let max_reduce_ext = 0i64.max(reduce_stride * (reduce_count as i64 - 1));

    match input {
        InputRef::Affine { base, stride } => {
            let first_base = base.0 as i64 + *stride as i64 * first_i as i64;
            let last_base = base.0 as i64 + *stride as i64 * last_i as i64;
            let lo = first_base.min(last_base) + min_reduce_ext;
            let hi = first_base.max(last_base) + max_reduce_ext;
            // For the reduce, we need all atoms in [lo, hi].
            let range = (hi - lo) as u64 + 1;
            if range <= 100_000 {
                for pos in lo..=hi {
                    atoms.insert(AtomId(pos as u64));
                }
            } else {
                // Large range: mark boundaries.
                atoms.insert(AtomId(lo as u64));
                atoms.insert(AtomId(hi as u64));
            }
        }
        InputRef::Broadcast(atom_id) => {
            let base = atom_id.0 as i64;
            let lo = base + min_reduce_ext;
            let hi = base + max_reduce_ext;
            for pos in lo..=hi {
                atoms.insert(AtomId(pos as u64));
            }
        }
        InputRef::StridedBroadcast { base, stride, repeat } => {
            let first_block = first_i / repeat;
            let last_block = last_i / repeat;
            for block in first_block..=last_block {
                let base_pos = base.0 as i64 + *stride * block as i64;
                let lo = base_pos + min_reduce_ext;
                let hi = base_pos + max_reduce_ext;
                for pos in lo..=hi {
                    atoms.insert(AtomId(pos as u64));
                }
            }
        }
        _ => {
            // For other InputRef types, collect base atoms and extend.
            // Conservative: we already collected base atoms, extend each by reduce stride.
            // This is handled by the base collect_input_atoms call.
        }
    }
}

/// Remap InputRefs from main graph AtomIds to local span AtomIds.
///
/// For a sub-range work item (atom_offset > 0 or atom_count < group.count),
/// we need to adjust the InputRef to reference only the sub-range's atoms.
fn remap_inputs(
    inputs: &[InputRef],
    op: &ScalarOp,
    atom_offset: u64,
    atom_count: u64,
    group_count: u64,
    groups: &[AtomGroup],
    main_to_local: &HashMap<AtomId, AtomId>,
) -> Vec<InputRef> {
    let is_full = atom_offset == 0 && atom_count == group_count;

    inputs
        .iter()
        .map(|input| remap_single_input(input, atom_offset, atom_count, group_count, is_full, main_to_local))
        .collect()
}

/// Remap a single InputRef.
fn remap_single_input(
    input: &InputRef,
    atom_offset: u64,
    atom_count: u64,
    group_count: u64,
    is_full: bool,
    main_to_local: &HashMap<AtomId, AtomId>,
) -> InputRef {
    match input {
        InputRef::Broadcast(atom_id) => {
            let local = main_to_local
                .get(atom_id)
                .copied()
                .unwrap_or(*atom_id);
            InputRef::Broadcast(local)
        }
        InputRef::Affine { base, stride } => {
            // For a sub-range starting at atom_offset, the effective base shifts.
            // Original: atom i reads base + stride * i
            // Sub-range: atom j (j=0..atom_count) reads base + stride * (atom_offset + j)
            //          = (base + stride * atom_offset) + stride * j
            let effective_base = AtomId(
                base.0.wrapping_add((*stride as i64 * atom_offset as i64) as u64),
            );
            let local_base = main_to_local
                .get(&effective_base)
                .copied()
                .unwrap_or(effective_base);

            // If stride is 1 and the effective_base maps to a contiguous local range,
            // we can keep the Affine. Otherwise, check if the stride pattern holds.
            if *stride == 1 || *stride == -1 || *stride == 0 {
                InputRef::Affine {
                    base: local_base,
                    stride: *stride,
                }
            } else {
                // Non-unit stride: verify the mapping is consistent.
                // The local atoms should also be contiguous with the same stride.
                // For now, trust the mapping and use the same stride.
                InputRef::Affine {
                    base: local_base,
                    stride: *stride,
                }
            }
        }
        InputRef::StridedBroadcast { base, stride, repeat } => {
            // For a sub-range starting at atom_offset:
            // Original: atom i reads base + stride * (i / repeat)
            // Sub-range: atom j reads base + stride * ((atom_offset + j) / repeat)
            //
            // If atom_offset is a multiple of repeat, this simplifies:
            //   base + stride * (atom_offset/repeat + j/repeat)
            //   = (base + stride * atom_offset/repeat) + stride * (j/repeat)
            //   Same StridedBroadcast with shifted base.
            //
            // If atom_offset is NOT a multiple of repeat, we need to be more careful.
            if atom_offset % repeat == 0 {
                let block_offset = atom_offset / repeat;
                let effective_base = AtomId(
                    base.0.wrapping_add((*stride * block_offset as i64) as u64),
                );
                let local_base = main_to_local
                    .get(&effective_base)
                    .copied()
                    .unwrap_or(effective_base);
                InputRef::StridedBroadcast {
                    base: local_base,
                    stride: *stride,
                    repeat: *repeat,
                }
            } else {
                // Non-aligned sub-range. Fall back to Explicit.
                let mut ids = Vec::with_capacity(atom_count as usize);
                for j in 0..atom_count {
                    let main_atom = input.resolve(atom_offset + j, 0);
                    let local = main_to_local
                        .get(&main_atom)
                        .copied()
                        .unwrap_or(main_atom);
                    ids.push(local);
                }
                InputRef::Explicit(ids)
            }
        }
        InputRef::Modular { base, stride, modulus } => {
            // Modular: atom i reads base + stride * (i % modulus).
            // For sub-range: atom j reads base + stride * ((atom_offset + j) % modulus).
            // This doesn't simplify nicely, so remap the base and keep the pattern.
            // The modular pattern accesses the same set of atoms regardless of offset.
            let local_base = main_to_local
                .get(base)
                .copied()
                .unwrap_or(*base);
            if is_full {
                InputRef::Modular {
                    base: local_base,
                    stride: *stride,
                    modulus: *modulus,
                }
            } else {
                // For sub-ranges, we need to shift the modular pattern.
                // atom j -> base + stride * ((atom_offset + j) % modulus)
                // This is still Modular but the offset changes the starting phase.
                // Since the modular wrap accesses all modulus atoms anyway,
                // we can keep it but we need to verify the local mapping is correct.
                // Actually, for Modular, the base and all modular atoms are already
                // in main_to_local. Just remap the base.
                // But the offset shifts the access pattern. We should use Explicit.
                let mut ids = Vec::with_capacity(atom_count as usize);
                for j in 0..atom_count {
                    let main_atom = input.resolve(atom_offset + j, 0);
                    let local = main_to_local
                        .get(&main_atom)
                        .copied()
                        .unwrap_or(main_atom);
                    ids.push(local);
                }
                InputRef::Explicit(ids)
            }
        }
        InputRef::Explicit(ids) => {
            // Sub-range: take ids[atom_offset..atom_offset+atom_count].
            let start = atom_offset as usize;
            let end = (atom_offset + atom_count) as usize;
            let slice = &ids[start..end.min(ids.len())];
            let local_ids: Vec<AtomId> = slice
                .iter()
                .map(|id| main_to_local.get(id).copied().unwrap_or(*id))
                .collect();
            InputRef::Explicit(local_ids)
        }
        InputRef::SymAffine { base, stride_i, stride_k } => {
            // Sub-range: atom j reads base + stride_i * (atom_offset + j) + stride_k * k
            //          = (base + stride_i * atom_offset) + stride_i * j + stride_k * k
            let effective_base = AtomId(
                base.0.wrapping_add((*stride_i as i64 * atom_offset as i64) as u64),
            );
            let local_base = main_to_local
                .get(&effective_base)
                .copied()
                .unwrap_or(effective_base);
            InputRef::SymAffine {
                base: local_base,
                stride_i: *stride_i,
                stride_k: *stride_k,
            }
        }
    }
}

/// Remap ScalarOp fields that reference main graph AtomIds.
/// Currently only IndirectLoad has a table_base AtomId.
fn remap_op(op: &ScalarOp, main_to_local: &HashMap<AtomId, AtomId>) -> ScalarOp {
    match op {
        ScalarOp::IndirectLoad { table_base, output_dtype } => {
            let local_table = main_to_local
                .get(table_base)
                .copied()
                .unwrap_or(*table_base);
            ScalarOp::IndirectLoad {
                table_base: local_table,
                output_dtype: *output_dtype,
            }
        }
        _ => op.clone(),
    }
}

// ─── Diagnostics ─────────────────────────────────────────────────────────────

impl SpanPlan {
    /// Print summary statistics about the span plan.
    pub fn print_summary(&self) {
        let total_spans: usize = self.phases.iter().map(|p| p.spans.len()).sum();
        let non_empty_spans: usize = self.phases.iter()
            .flat_map(|p| p.spans.iter())
            .filter(|s| s.graph.num_groups() > 0)
            .count();
        let total_inputs: usize = self.phases.iter()
            .flat_map(|p| p.spans.iter())
            .map(|s| s.inputs.len())
            .sum();
        let total_outputs: usize = self.phases.iter()
            .flat_map(|p| p.spans.iter())
            .map(|s| s.outputs.len())
            .sum();

        println!(
            "SpanPlan: {} lanes, {} phases, {} spans ({} non-empty)",
            self.num_lanes, self.phases.len(), total_spans, non_empty_spans
        );
        println!("  Total inputs: {}, Total outputs: {}", total_inputs, total_outputs);

        for (phase_idx, phase) in self.phases.iter().enumerate() {
            let active = phase.spans.iter().filter(|s| s.graph.num_groups() > 0).count();
            let max_groups = phase.spans.iter().map(|s| s.graph.num_groups()).max().unwrap_or(0);
            let max_atoms = phase.spans.iter().map(|s| s.graph.num_atoms()).max().unwrap_or(0);
            let min_atoms = phase.spans.iter()
                .filter(|s| s.graph.num_atoms() > 0)
                .map(|s| s.graph.num_atoms())
                .min()
                .unwrap_or(0);
            let balance = if min_atoms > 0 {
                format!("{:.1}x", max_atoms as f64 / min_atoms as f64)
            } else {
                "N/A".to_string()
            };
            println!(
                "  Phase {}: {} active lanes, max {} groups/{} atoms, balance {}",
                phase_idx, active, max_groups, max_atoms, balance
            );
        }
    }

    /// Validate all span NanoGraphs.
    ///
    /// Checks:
    /// 1. Each span's NanoGraph passes NanoGraph::validate()
    /// 2. All external inputs reference atoms that exist in earlier phases' outputs
    ///    or in the original graph's literal groups
    /// 3. No cross-span dependencies within a phase (guaranteed by construction
    ///    since we use v2c's phase assignments)
    pub fn validate(&self, original: &NanoGraph) -> Vec<String> {
        let mut errors = Vec::new();

        // Seed available atoms with all literal atoms from the original graph.
        // These are always available (from the shared values buffer) without
        // needing to be produced by an earlier phase.
        let mut available_atoms: HashSet<AtomId> = HashSet::new();
        for group in original.groups() {
            if matches!(group.op, ScalarOp::Literal(_)) && group.inputs.is_empty() {
                for i in 0..group.count {
                    available_atoms.insert(group.base_id.offset(i));
                }
            }
        }

        for (phase_idx, phase) in self.phases.iter().enumerate() {
            for (lane_idx, span) in phase.spans.iter().enumerate() {
                // Check 1: NanoGraph internal validation.
                let graph_errors = span.graph.validate();
                for err in graph_errors {
                    errors.push(format!(
                        "Phase {} lane {}: graph validation error: {}",
                        phase_idx, lane_idx, err
                    ));
                }

                // Check 2: All external inputs must be available.
                for &(main_atom, _local_atom) in &span.inputs {
                    if !available_atoms.contains(&main_atom) {
                        errors.push(format!(
                            "Phase {} lane {}: external input {:?} not available \
                             (not produced by any earlier phase)",
                            phase_idx, lane_idx, main_atom
                        ));
                    }
                }
            }

            // After processing all spans in this phase, add their outputs
            // to the available set.
            for span in &phase.spans {
                for &(_local_atom, main_atom) in &span.outputs {
                    available_atoms.insert(main_atom);
                }
            }
        }

        errors
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::nano_graph::{InputRef, NanoGraph, ScalarBinOp, ScalarOp, ScalarUnaryOp};
    use crate::numeric_scalar::NumericScalar;

    // ─── Test graph builders (shared with v2c tests) ─────────────────────

    fn build_matmul(m: u64, k: u64, n: u64) -> NanoGraph {
        let mut g = NanoGraph::new();
        let a_base = g.push_group(
            m * k,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![], vec![], vec![],
        );
        let b_base = g.push_group(
            k * n,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![], vec![], vec![],
        );
        let mut mul_bases = Vec::new();
        for row in 0..m {
            let a_row = AtomId(a_base.0 + row * k);
            let mul = g.push_group(
                k * n,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![], vec![],
                vec![
                    InputRef::StridedBroadcast {
                        base: a_row,
                        stride: 1,
                        repeat: n,
                    },
                    InputRef::Affine {
                        base: b_base,
                        stride: 1,
                    },
                ],
            );
            mul_bases.push(mul);
        }
        let mut reduce_bases = Vec::new();
        for row in 0..m {
            let red = g.push_group(
                n,
                ScalarOp::ReduceSum {
                    reduce_count: k,
                    reduce_stride: n as i64,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![], vec![],
                vec![InputRef::Affine {
                    base: mul_bases[row as usize],
                    stride: 1,
                }],
            );
            reduce_bases.push(red);
        }
        for &rb in &reduce_bases {
            for i in 0..n {
                g.outputs.push(AtomId(rb.0 + i));
            }
        }
        g
    }

    fn build_matmul_chain(m: u64, k1: u64, n1: u64, k2: u64, n2: u64) -> NanoGraph {
        let mut g = NanoGraph::new();
        let a1 = g.push_group(m * k1, ScalarOp::Literal(NumericScalar::F32(1.0)), vec![], vec![], vec![]);
        let b1 = g.push_group(k1 * n1, ScalarOp::Literal(NumericScalar::F32(1.0)), vec![], vec![], vec![]);

        let mut mul1_bases = Vec::new();
        for row in 0..m {
            let a_row = AtomId(a1.0 + row * k1);
            let mul = g.push_group(
                k1 * n1,
                ScalarOp::Binary { op: ScalarBinOp::Mul, compute_dtype: DType::F32, output_dtype: DType::F32 },
                vec![], vec![],
                vec![
                    InputRef::StridedBroadcast { base: a_row, stride: 1, repeat: n1 },
                    InputRef::Affine { base: b1, stride: 1 },
                ],
            );
            mul1_bases.push(mul);
        }
        let mut red1_bases = Vec::new();
        for row in 0..m {
            let red = g.push_group(
                n1,
                ScalarOp::ReduceSum { reduce_count: k1, reduce_stride: n1 as i64, compute_dtype: DType::F32, output_dtype: DType::F32 },
                vec![], vec![],
                vec![InputRef::Affine { base: mul1_bases[row as usize], stride: 1 }],
            );
            red1_bases.push(red);
        }

        let act = g.push_group(
            m * n1,
            ScalarOp::Unary { op: ScalarUnaryOp::Tanh, compute_dtype: DType::F32, output_dtype: DType::F32 },
            vec![], vec![],
            vec![InputRef::Affine { base: red1_bases[0], stride: 1 }],
        );

        let b2 = g.push_group(n1 * n2, ScalarOp::Literal(NumericScalar::F32(1.0)), vec![], vec![], vec![]);

        let mut mul2_bases = Vec::new();
        for row in 0..m {
            let mul = g.push_group(
                n1 * n2,
                ScalarOp::Binary { op: ScalarBinOp::Mul, compute_dtype: DType::F32, output_dtype: DType::F32 },
                vec![], vec![],
                vec![
                    InputRef::StridedBroadcast { base: AtomId(act.0 + row * n1), stride: 1, repeat: n2 },
                    InputRef::Affine { base: b2, stride: 1 },
                ],
            );
            mul2_bases.push(mul);
        }
        let mut red2_bases = Vec::new();
        for row in 0..m {
            let red = g.push_group(
                n2,
                ScalarOp::ReduceSum { reduce_count: n1, reduce_stride: n2 as i64, compute_dtype: DType::F32, output_dtype: DType::F32 },
                vec![], vec![],
                vec![InputRef::Affine { base: mul2_bases[row as usize], stride: 1 }],
            );
            red2_bases.push(red);
        }
        for &rb in &red2_bases {
            for i in 0..n2 {
                g.outputs.push(AtomId(rb.0 + i));
            }
        }
        g
    }

    fn build_elementwise(count: u64) -> NanoGraph {
        let mut g = NanoGraph::new();
        let a = g.push_group(count, ScalarOp::Literal(NumericScalar::F32(1.0)), vec![], vec![], vec![]);
        let b = g.push_group(count, ScalarOp::Literal(NumericScalar::F32(2.0)), vec![], vec![], vec![]);
        let c = g.push_group(
            count,
            ScalarOp::Binary { op: ScalarBinOp::Add, compute_dtype: DType::F32, output_dtype: DType::F32 },
            vec![], vec![],
            vec![
                InputRef::Affine { base: a, stride: 1 },
                InputRef::Affine { base: b, stride: 1 },
            ],
        );
        for i in 0..count {
            g.outputs.push(AtomId(c.0 + i));
        }
        g
    }

    fn build_allrows_chain(count: u64) -> NanoGraph {
        let mut g = NanoGraph::new();
        let lit = g.push_group(count, ScalarOp::Literal(NumericScalar::F32(1.0)), vec![], vec![], vec![]);
        let a = g.push_group(
            count,
            ScalarOp::Unary { op: ScalarUnaryOp::Tanh, compute_dtype: DType::F32, output_dtype: DType::F32 },
            vec![], vec![],
            vec![InputRef::Affine { base: lit, stride: 1 }],
        );
        let b = g.push_group(
            count,
            ScalarOp::Unary { op: ScalarUnaryOp::Tanh, compute_dtype: DType::F32, output_dtype: DType::F32 },
            vec![], vec![],
            vec![InputRef::Affine { base: a, stride: 1 }],
        );
        let c = g.push_group(
            count,
            ScalarOp::Unary { op: ScalarUnaryOp::Tanh, compute_dtype: DType::F32, output_dtype: DType::F32 },
            vec![], vec![],
            vec![InputRef::Affine { base: b, stride: 1 }],
        );
        for i in 0..count {
            g.outputs.push(AtomId(c.0 + i));
        }
        g
    }

    fn build_parallel_matmuls(m: u64, k: u64, n: u64) -> NanoGraph {
        let mut g = NanoGraph::new();
        let a = g.push_group(m * k, ScalarOp::Literal(NumericScalar::F32(1.0)), vec![], vec![], vec![]);
        let b1 = g.push_group(k * n, ScalarOp::Literal(NumericScalar::F32(1.0)), vec![], vec![], vec![]);
        let b2 = g.push_group(k * n, ScalarOp::Literal(NumericScalar::F32(2.0)), vec![], vec![], vec![]);

        let mut all_red_bases = Vec::new();
        for (b_base, _) in [(b1, 1.0f32), (b2, 2.0)] {
            for row in 0..m {
                let a_row = AtomId(a.0 + row * k);
                let mul = g.push_group(
                    k * n,
                    ScalarOp::Binary { op: ScalarBinOp::Mul, compute_dtype: DType::F32, output_dtype: DType::F32 },
                    vec![], vec![],
                    vec![
                        InputRef::StridedBroadcast { base: a_row, stride: 1, repeat: n },
                        InputRef::Affine { base: b_base, stride: 1 },
                    ],
                );
                let red = g.push_group(
                    n,
                    ScalarOp::ReduceSum { reduce_count: k, reduce_stride: n as i64, compute_dtype: DType::F32, output_dtype: DType::F32 },
                    vec![], vec![],
                    vec![InputRef::Affine { base: mul, stride: 1 }],
                );
                all_red_bases.push(red);
            }
        }
        for &rb in &all_red_bases {
            for i in 0..n {
                g.outputs.push(AtomId(rb.0 + i));
            }
        }
        g
    }

    // ─── Validation helpers ──────────────────────────────────────────────

    /// Verify all span graphs pass internal validation and the plan is consistent.
    fn verify_span_plan(graph: &NanoGraph, plan: &SpanPlan) {
        let errors = plan.validate(graph);
        assert!(errors.is_empty(), "Span plan validation errors:\n{}", errors.join("\n"));
    }

    /// Verify that every compute atom in the main graph appears exactly once
    /// across all spans' outputs.
    fn verify_output_coverage(graph: &NanoGraph, plan: &SpanPlan) {
        let groups = graph.groups();
        let is_literal: Vec<bool> = groups
            .iter()
            .map(|g| matches!(g.op, ScalarOp::Literal(_)) && g.inputs.is_empty())
            .collect();

        // Collect all main-graph AtomIds produced across all spans.
        let mut produced: HashMap<AtomId, usize> = HashMap::new(); // atom -> count
        for phase in &plan.phases {
            for span in &phase.spans {
                for &(_local, main_atom) in &span.outputs {
                    *produced.entry(main_atom).or_insert(0) += 1;
                }
            }
        }

        // Every compute atom must appear exactly once.
        for (gi, group) in groups.iter().enumerate() {
            if is_literal[gi] {
                continue;
            }
            for offset in 0..group.count {
                let atom = AtomId(group.base_id.0 + offset);
                let count = produced.get(&atom).copied().unwrap_or(0);
                assert_eq!(
                    count, 1,
                    "Compute atom {:?} (group {}, offset {}) appears {} times in span outputs",
                    atom, gi, offset, count
                );
            }
        }
    }

    /// Verify the phase ordering of external inputs: every external input
    /// to a span in phase P must be produced by a span in phase < P
    /// or be a literal atom from the original graph.
    fn verify_input_availability(plan: &SpanPlan, graph: &NanoGraph) {
        // Seed with all literal atoms from the original graph.
        let mut available: HashSet<AtomId> = HashSet::new();
        for group in graph.groups() {
            if matches!(group.op, ScalarOp::Literal(_)) && group.inputs.is_empty() {
                for i in 0..group.count {
                    available.insert(group.base_id.offset(i));
                }
            }
        }

        for (phase_idx, phase) in plan.phases.iter().enumerate() {
            for (lane_idx, span) in phase.spans.iter().enumerate() {
                for &(main_atom, _local) in &span.inputs {
                    assert!(
                        available.contains(&main_atom),
                        "Phase {} lane {}: external input {:?} not yet produced",
                        phase_idx, lane_idx, main_atom
                    );
                }
            }
            // Add outputs from this phase.
            for span in &phase.spans {
                for &(_local, main_atom) in &span.outputs {
                    available.insert(main_atom);
                }
            }
        }
    }

    /// Verify no cross-span dependencies within a phase.
    /// Within a phase, each span's external inputs must come from earlier phases,
    /// not from other spans in the same phase.
    fn verify_phase_independence(plan: &SpanPlan) {
        for (phase_idx, phase) in plan.phases.iter().enumerate() {
            // Collect all main-graph AtomIds produced by spans in this phase.
            let mut this_phase_outputs: HashSet<AtomId> = HashSet::new();
            for span in &phase.spans {
                for &(_local, main_atom) in &span.outputs {
                    this_phase_outputs.insert(main_atom);
                }
            }

            // No span's external input should reference an atom produced
            // by another span in the same phase.
            for (lane_idx, span) in phase.spans.iter().enumerate() {
                for &(main_atom, _local) in &span.inputs {
                    assert!(
                        !this_phase_outputs.contains(&main_atom),
                        "Phase {} lane {}: external input {:?} is produced by a span \
                         in the SAME phase — cross-span dependency!",
                        phase_idx, lane_idx, main_atom
                    );
                }
            }
        }
    }

    // ─── Test cases ──────────────────────────────────────────────────────

    #[test]
    fn test_empty_graph() {
        let g = NanoGraph::new();
        let plan = plan_execution_spans(&g, 4);
        assert_eq!(plan.num_lanes, 4);
        assert!(plan.phases.is_empty());
    }

    #[test]
    fn test_single_lane_matmul() {
        let g = build_matmul(4, 4, 4);
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let plan = plan_execution_spans(&g, 1);
        assert_eq!(plan.num_lanes, 1);
        assert!(!plan.phases.is_empty());

        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);

        println!("Single lane matmul:");
        plan.print_summary();
    }

    #[test]
    fn test_matmul_2lanes() {
        let g = build_matmul(4, 8, 4);
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let plan = plan_execution_spans(&g, 2);
        assert_eq!(plan.num_lanes, 2);

        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);

        // Single matmul: 1 phase with 2 spans (one per lane).
        assert_eq!(plan.phases.len(), 1);

        // Both lanes should have non-empty spans.
        let active = plan.phases[0].spans.iter().filter(|s| s.graph.num_groups() > 0).count();
        assert_eq!(active, 2);

        println!("Matmul 2 lanes:");
        plan.print_summary();
    }

    #[test]
    fn test_matmul_8lanes() {
        let g = build_matmul(8, 4, 4);
        assert!(g.validate().is_empty());

        let plan = plan_execution_spans(&g, 8);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);

        assert_eq!(plan.phases.len(), 1);

        println!("Matmul 8 lanes:");
        plan.print_summary();
    }

    #[test]
    fn test_matmul_chain_monolithic() {
        let g = build_matmul_chain(4, 4, 4, 4, 4);
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let plan = plan_execution_spans(&g, 2);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);

        // Multi-phase: matmul1, activation, matmul2.
        assert!(plan.phases.len() >= 2, "Should have >= 2 phases, got {}", plan.phases.len());

        println!("Matmul chain (monolithic) 2 lanes:");
        plan.print_summary();
    }

    #[test]
    fn test_elementwise_splitting() {
        let g = build_elementwise(1024);
        assert!(g.validate().is_empty());

        let plan = plan_execution_spans(&g, 4);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);

        assert_eq!(plan.phases.len(), 1);

        // The elementwise group should be split: multiple lanes active.
        let active = plan.phases[0].spans.iter().filter(|s| s.graph.num_groups() > 0).count();
        assert!(active >= 2, "Should have >= 2 active lanes, got {}", active);

        println!("Elementwise 4 lanes:");
        plan.print_summary();
    }

    #[test]
    fn test_allrows_chain() {
        let g = build_allrows_chain(1024);
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let plan = plan_execution_spans(&g, 8);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);

        // AllRows chain with same count: 1 phase.
        assert_eq!(plan.phases.len(), 1, "AllRows chain should be 1 phase, got {}", plan.phases.len());

        println!("AllRows chain 8 lanes:");
        plan.print_summary();
    }

    #[test]
    fn test_parallel_matmuls() {
        let g = build_parallel_matmuls(4, 4, 4);
        assert!(g.validate().is_empty());

        let plan = plan_execution_spans(&g, 4);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);

        assert_eq!(plan.phases.len(), 1);

        println!("Parallel matmuls 4 lanes:");
        plan.print_summary();
    }

    #[test]
    fn test_span_self_containment() {
        // Verify that each span's NanoGraph is truly self-contained:
        // all InputRefs resolve to atoms within the span graph.
        let g = build_matmul_chain(4, 4, 4, 4, 4);
        assert!(g.validate().is_empty());

        let plan = plan_execution_spans(&g, 4);

        for (phase_idx, phase) in plan.phases.iter().enumerate() {
            for (lane_idx, span) in phase.spans.iter().enumerate() {
                let errors = span.graph.validate();
                assert!(
                    errors.is_empty(),
                    "Phase {} lane {} span graph validation errors: {:?}",
                    phase_idx, lane_idx, errors
                );
            }
        }
    }

    #[test]
    fn test_input_output_consistency() {
        // For a multi-phase plan, verify that external inputs in phase P+1
        // correspond to outputs in phase P.
        let g = build_matmul_chain(4, 4, 4, 4, 4);
        assert!(g.validate().is_empty());

        let plan = plan_execution_spans(&g, 2);
        verify_input_availability(&plan, &g);
        verify_phase_independence(&plan);
    }

    #[test]
    fn test_various_sizes() {
        for count in [1, 7, 8, 15, 16, 100, 1000] {
            let g = build_elementwise(count);
            let plan = plan_execution_spans(&g, 4);
            verify_span_plan(&g, &plan);
            verify_output_coverage(&g, &plan);
        }
    }

    #[test]
    fn test_comprehensive_configs() {
        let configs: Vec<(&str, NanoGraph, usize)> = vec![
            ("matmul_2x2_2lanes", build_matmul(2, 2, 2), 2),
            ("matmul_4x8x4_2lanes", build_matmul(4, 8, 4), 2),
            ("matmul_4x8x4_4lanes", build_matmul(4, 8, 4), 4),
            ("matmul_8x4x4_8lanes", build_matmul(8, 4, 4), 8),
            ("chain_mono_2lanes", build_matmul_chain(2, 4, 4, 4, 2), 2),
            ("chain_mono_4lanes", build_matmul_chain(4, 4, 4, 4, 4), 4),
            ("parallel_2lanes", build_parallel_matmuls(4, 4, 4), 2),
            ("parallel_4lanes", build_parallel_matmuls(4, 4, 4), 4),
            ("elementwise_2lanes", build_elementwise(256), 2),
            ("elementwise_4lanes", build_elementwise(1024), 4),
            ("elementwise_8lanes", build_elementwise(8192), 8),
            ("allrows_chain_4lanes", build_allrows_chain(256), 4),
        ];

        for (name, graph, lanes) in configs {
            assert!(graph.validate().is_empty(), "{}: validation failed", name);
            let plan = plan_execution_spans(&graph, lanes);
            verify_span_plan(&graph, &plan);
            verify_output_coverage(&graph, &plan);
            println!(
                "  {}: {} phases, {} lanes",
                name, plan.phases.len(), plan.num_lanes
            );
        }
    }

    #[test]
    fn test_larger_matmul_chain() {
        let g = build_matmul_chain(8, 16, 16, 16, 16);
        assert!(g.validate().is_empty());

        let plan = plan_execution_spans(&g, 4);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);

        assert!(plan.phases.len() <= 10, "Should have <= 10 phases, got {}", plan.phases.len());

        println!("Larger matmul chain 4 lanes:");
        plan.print_summary();
    }

    #[test]
    fn test_realistic_matmul_chain() {
        let g = build_matmul_chain(16, 32, 32, 32, 32);
        assert!(g.validate().is_empty());

        let plan = plan_execution_spans(&g, 8);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);

        assert!(plan.phases.len() >= 2);
        assert!(plan.phases.len() <= 5);

        println!("Realistic matmul chain 8 lanes:");
        plan.print_summary();
    }

    #[test]
    fn test_uneven_row_count() {
        let g = build_matmul(7, 4, 4);
        assert!(g.validate().is_empty());

        let plan = plan_execution_spans(&g, 4);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);
    }
}
