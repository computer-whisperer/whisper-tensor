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

use super::{AtomMapping, Phase, Span, SpanPlan, v2c};

/// Literal groups with fewer atoms than this are duplicated into spans.
/// Larger literals (weight matrices) become external inputs instead.
const LITERAL_INLINE_THRESHOLD: u64 = 1024;

// ─── Public API ──────────────────────────────────────────────────────────────

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
    let mut v2c_plan = v2c::plan_execution(graph, num_lanes);

    let is_literal: Vec<bool> = groups
        .iter()
        .map(|g| matches!(g.op, ScalarOp::Literal(_)) && g.inputs.is_empty())
        .collect();

    // Step 1b: Fix cross-lane dependencies within phases.
    //
    // v2c splits AllRows groups across lanes. But if a downstream group in the
    // SAME phase reads the FULL output range of a split group (not just its
    // lane-local slice), that's a cross-lane dependency within a phase — which
    // violates the independence invariant.
    //
    // Fix: duplicate the split group's computation into every lane. Each lane
    // independently computes the full group (cheap for small groups like the
    // 3072-atom attention mask Selects in GPT-2). However, for output purposes,
    // each lane only outputs its original split slice to avoid duplicate outputs.
    let dup_info = fix_cross_lane_deps(&mut v2c_plan, groups, &is_literal, num_lanes);

    // Step 2: Build a map from (main graph AtomId) -> which (phase, lane, work_item)
    // produces it. This lets us determine what's "external" to a span.
    //
    // For literal groups, they're "always available" — we'll inline them into
    // each span that needs them.

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
                    literal_mappings: vec![],
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
                &dup_info,
            );
            spans.push(span);
        }

        phases.push(Phase { spans });
    }

    SpanPlan { num_lanes, phases }
}

// ─── Cross-lane dependency fix ───────────────────────────────────────────────

/// Duplication info: for each duplicated group, the original per-lane output slice.
/// Key = group_idx. Value = vec indexed by lane_idx, each entry = (atom_offset, atom_count).
/// Lanes that didn't originally have this group get (0, 0).
type DupInfo = HashMap<usize, Vec<(u64, u64)>>;

/// Detect and fix cross-lane dependencies within phases.
///
/// When an AllRows group G is split across lanes in phase P, and a consumer
/// group C in phase P reads a range of G that spans multiple lanes' slices,
/// C can't run independently — it needs data from other lanes in the same phase.
///
/// Fix: replace G's per-lane slices with full-group duplication to every lane.
/// Each lane independently computes all of G's atoms. This is cheap for small
/// groups (e.g., 3072-atom attention mask Selects in GPT-2).
///
/// Returns duplication info so `build_span` can output only the original
/// per-lane slices (avoiding duplicate output atoms across lanes).
fn fix_cross_lane_deps(
    plan: &mut v2c::ExecutionPlan,
    groups: &[AtomGroup],
    is_literal: &[bool],
    num_lanes: usize,
) -> DupInfo {
    let n = groups.len();

    // Build group_idx -> phase_idx mapping.
    let mut group_phase: Vec<Option<usize>> = vec![None; n];
    for (phase_idx, phase) in plan.phases.iter().enumerate() {
        for lane_work in &phase.lane_work {
            for work in lane_work {
                // For split groups, all lanes are in the same phase.
                group_phase[work.group_idx] = Some(phase_idx);
            }
        }
    }

    // Identify split groups: groups assigned to multiple lanes in the same phase.
    // split_groups[group_idx] = Some(phase_idx) if the group is split.
    let mut group_lane_count: Vec<usize> = vec![0; n];
    for phase in &plan.phases {
        for lane_work in &phase.lane_work {
            for work in lane_work {
                group_lane_count[work.group_idx] += 1;
            }
        }
    }
    let mut split_groups: HashSet<usize> = HashSet::new();
    for gi in 0..n {
        if group_lane_count[gi] > 1 {
            split_groups.insert(gi);
        }
    }

    if split_groups.is_empty() {
        return HashMap::new();
    }

    // For each phase, check if any consumer's inputs reference a split group
    // in a way that spans beyond the consumer's own lane-local slice.
    //
    // Approach: for each non-split (or split) group C in phase P, resolve its
    // producer groups. If any producer G is split in phase P, check whether
    // C's access to G is lane-local.
    let mut groups_to_duplicate: HashSet<usize> = HashSet::new();

    for (phase_idx, phase) in plan.phases.iter().enumerate() {
        // Collect all groups in this phase.
        let mut phase_groups_set: HashSet<usize> = HashSet::new();
        for lane_work in &phase.lane_work {
            for work in lane_work {
                phase_groups_set.insert(work.group_idx);
            }
        }

        // For each group in this phase, check its inputs.
        for &gi in &phase_groups_set {
            let group = &groups[gi];
            for input in &group.inputs {
                let referenced = resolve_all_referenced_atoms_to_groups(input, group.count, groups);
                for prod_gi in referenced {
                    if is_literal[prod_gi] || prod_gi == gi {
                        continue;
                    }
                    if !split_groups.contains(&prod_gi) {
                        continue;
                    }
                    if group_phase[prod_gi] != Some(phase_idx) {
                        continue;
                    }
                    // prod_gi is a split group in the same phase as gi.
                    // Check if the access is lane-local.
                    if !is_input_lane_local(input, group.count, &groups[prod_gi], num_lanes) {
                        groups_to_duplicate.insert(prod_gi);
                    }
                }
            }

            // Also check ReduceSum/ReduceMax extended access.
            match &group.op {
                ScalarOp::ReduceSum {
                    reduce_count,
                    reduce_stride,
                    ..
                }
                | ScalarOp::ReduceMax {
                    reduce_count,
                    reduce_stride,
                    ..
                } if *reduce_count > 1 && *reduce_stride != 0 => {
                    for input in &group.inputs {
                        let referenced = resolve_input_to_group_ranges(
                            input,
                            0,
                            group.count,
                            *reduce_count,
                            *reduce_stride,
                            groups,
                        );
                        for (prod_gi, _, _) in referenced {
                            if is_literal[prod_gi] || prod_gi == gi {
                                continue;
                            }
                            if !split_groups.contains(&prod_gi) {
                                continue;
                            }
                            if group_phase[prod_gi] != Some(phase_idx) {
                                continue;
                            }
                            // The reduce extends the read range, likely crossing lanes.
                            groups_to_duplicate.insert(prod_gi);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    if groups_to_duplicate.is_empty() {
        return HashMap::new();
    }

    // Transitive closure: duplicated groups might themselves depend on split
    // groups in the same phase. When we duplicate G, each lane computes ALL
    // of G — so G's inputs must be fully available on each lane too.
    // If G reads from a split group H in the same phase, H must also be
    // duplicated (unless the access is lane-local, but since G is now full,
    // it's reading the full range of H, which is never lane-local for a
    // split group).
    loop {
        let mut new_dups: Vec<usize> = Vec::new();
        for &gi in &groups_to_duplicate {
            let group = &groups[gi];
            let my_phase = group_phase[gi];
            for input in &group.inputs {
                let referenced = resolve_all_referenced_atoms_to_groups(input, group.count, groups);
                for prod_gi in referenced {
                    if is_literal[prod_gi] || prod_gi == gi {
                        continue;
                    }
                    if !split_groups.contains(&prod_gi) {
                        continue;
                    }
                    if group_phase[prod_gi] != my_phase {
                        continue;
                    }
                    if !groups_to_duplicate.contains(&prod_gi) {
                        new_dups.push(prod_gi);
                    }
                }
            }
            // Also check reduce extended access.
            match &group.op {
                ScalarOp::ReduceSum {
                    reduce_count,
                    reduce_stride,
                    ..
                }
                | ScalarOp::ReduceMax {
                    reduce_count,
                    reduce_stride,
                    ..
                } if *reduce_count > 1 && *reduce_stride != 0 => {
                    for input in &group.inputs {
                        let referenced = resolve_input_to_group_ranges(
                            input,
                            0,
                            group.count,
                            *reduce_count,
                            *reduce_stride,
                            groups,
                        );
                        for (prod_gi, _, _) in referenced {
                            if is_literal[prod_gi] || prod_gi == gi {
                                continue;
                            }
                            if !split_groups.contains(&prod_gi) {
                                continue;
                            }
                            if group_phase[prod_gi] != my_phase {
                                continue;
                            }
                            if !groups_to_duplicate.contains(&prod_gi) {
                                new_dups.push(prod_gi);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        if new_dups.is_empty() {
            break;
        }
        for gi in new_dups {
            groups_to_duplicate.insert(gi);
        }
    }

    // Record original per-lane splits for duplicated groups before modifying.
    let mut dup_info: DupInfo = HashMap::new();
    for &gi in &groups_to_duplicate {
        let mut per_lane: Vec<(u64, u64)> = vec![(0, 0); num_lanes];
        for (phase_idx, phase) in plan.phases.iter().enumerate() {
            if group_phase[gi] != Some(phase_idx) {
                continue;
            }
            for (lane_idx, lane_work) in phase.lane_work.iter().enumerate() {
                for work in lane_work {
                    if work.group_idx == gi {
                        per_lane[lane_idx] = (work.atom_offset, work.atom_count);
                    }
                }
            }
        }
        dup_info.insert(gi, per_lane);
    }

    // Modify the plan: for each group to duplicate, replace per-lane slices
    // with full-group assignment to every lane.
    for (phase_idx, phase) in plan.phases.iter_mut().enumerate() {
        let mut dups_in_phase: Vec<usize> = groups_to_duplicate
            .iter()
            .copied()
            .filter(|&gi| group_phase[gi] == Some(phase_idx))
            .collect();
        dups_in_phase.sort();

        if dups_in_phase.is_empty() {
            continue;
        }

        // Remove old split work items for these groups.
        for lane_work in &mut phase.lane_work {
            lane_work.retain(|work| !groups_to_duplicate.contains(&work.group_idx));
        }

        // Add full-group work to every lane.
        for &gi in &dups_in_phase {
            for lane_idx in 0..num_lanes {
                phase.lane_work[lane_idx].push(v2c::LaneWork {
                    group_idx: gi,
                    atom_offset: 0,
                    atom_count: groups[gi].count,
                });
            }
        }

        // Re-sort work items within each lane.
        for lane_work in &mut phase.lane_work {
            lane_work.sort_by_key(|w| (w.group_idx, w.atom_offset));
        }
    }

    dup_info
}

/// Check if a consumer's input access to a split producer is lane-local.
///
/// Lane-local means: when both consumer and producer are split evenly across
/// N lanes, lane j's slice of the consumer only reads lane j's slice of the producer.
fn is_input_lane_local(
    input: &InputRef,
    consumer_count: u64,
    producer: &AtomGroup,
    num_lanes: usize,
) -> bool {
    let producer_count = producer.count;
    match input {
        InputRef::Affine { stride, .. } => {
            // stride=1, same count: lane j's chunk maps 1:1.
            *stride == 1 && producer_count == consumer_count
        }
        InputRef::Broadcast(_) => {
            // One atom read by all lanes — cross-lane.
            false
        }
        InputRef::StridedBroadcast { repeat, .. } => {
            // atom i reads base + stride * (i / repeat).
            // Lane-local if producer count matches proportionally.
            let expected = (consumer_count + repeat - 1) / repeat;
            producer_count == expected
        }
        InputRef::Modular { .. } => {
            // Modular wraps around — not lane-local.
            false
        }
        InputRef::Explicit(_) => {
            // Arbitrary mapping — can't prove lane-locality.
            false
        }
        InputRef::SymAffine { .. } => {
            // Contraction — not lane-local.
            false
        }
    }
}

// ─── Span builder ────────────────────────────────────────────────────────────

/// Build a self-contained NanoGraph for one (phase, lane)'s work.
///
/// Range-based atom map for efficient lookup, backed by a BTreeMap.
/// Supports interleaved insertions and lookups in O(log n).
struct RangeAtomMap {
    /// Maps main_base -> (span_base, count). BTreeMap keeps entries sorted.
    map: BTreeMap<u64, (u64, u64)>,
}

impl RangeAtomMap {
    fn new() -> Self {
        Self {
            map: BTreeMap::new(),
        }
    }
    fn insert_range(&mut self, main_base: AtomId, span_base: AtomId, count: u64) {
        self.map.insert(main_base.0, (span_base.0, count));
    }
    fn get(&self, main_id: AtomId) -> Option<AtomId> {
        // Find the greatest key <= main_id.0.
        use std::ops::Bound;
        let mut iter = self
            .map
            .range((Bound::Unbounded, Bound::Included(main_id.0)));
        if let Some((&base, &(span_base, count))) = iter.next_back() {
            let offset = main_id.0.wrapping_sub(base);
            if offset < count {
                return Some(AtomId(span_base + offset));
            }
        }
        None
    }
}

/// The span includes:
/// 1. The assigned work items (potentially sub-ranges of groups)
/// 2. Literal groups that are referenced by the work items
/// 3. External inputs declared for atoms from earlier phases or other lanes
///
/// All operations are O(num_groups), not O(num_atoms).
fn build_span(
    main_graph: &NanoGraph,
    groups: &[AtomGroup],
    is_literal: &[bool],
    atom_owner: &[Vec<(u64, u64, usize, usize)>],
    lane_work: &[v2c::LaneWork],
    phase_idx: usize,
    lane_idx: usize,
    dup_info: &DupInfo,
) -> Span {
    let mut span_graph = NanoGraph::new();

    // Copy sym_dim configuration from main graph.
    for (name, &sd) in &main_graph.sym_dim_names {
        let local_sd = span_graph.sym_dim(name);
        if let Some(&bound) = main_graph.sym_dim_bounds.get(&sd) {
            span_graph.sym_dim_bounds.insert(local_sd, bound);
        }
    }

    // Range-based atom map.
    let mut main_to_local = RangeAtomMap::new();

    let mut assigned_group_slices: Vec<(usize, u64, u64)> = Vec::new();
    for work in lane_work {
        assigned_group_slices.push((work.group_idx, work.atom_offset, work.atom_count));
    }

    // Determine all referenced literal groups.
    let mut needed_literals: BTreeSet<usize> = BTreeSet::new();
    for &(gi, _, _) in &assigned_group_slices {
        collect_literal_deps(gi, groups, is_literal, &mut needed_literals);
    }

    // Add small literal groups; large ones become external inputs.
    let mut inlined_literals: BTreeSet<usize> = BTreeSet::new();
    let mut large_literal_groups: BTreeSet<usize> = BTreeSet::new();
    for &lit_gi in &needed_literals {
        let lit_group = &groups[lit_gi];
        if lit_group.count < LITERAL_INLINE_THRESHOLD {
            let local_base = span_graph.push_group(
                lit_group.count,
                lit_group.op.clone(),
                lit_group.sym_dims.clone(),
                lit_group.reduce_dims.clone(),
                vec![],
            );
            main_to_local.insert_range(lit_group.base_id, local_base, lit_group.count);
            inlined_literals.insert(lit_gi);
        } else {
            large_literal_groups.insert(lit_gi);
        }
    }

    // Collect external dependency ranges using group-level analysis.
    // For each work item, identify external producer groups and compute their ranges.
    let mut external_ranges: Vec<(usize, u64, u64)> = Vec::new();

    // Add large literal groups as full-range external inputs.
    for &li in &large_literal_groups {
        let lg = &groups[li];
        external_ranges.push((li, 0, lg.count));
    }

    // For each assigned work item, find external producer groups.
    for &(gi, atom_offset, atom_count) in &assigned_group_slices {
        let group = &groups[gi];
        collect_external_ranges_c(
            group,
            atom_offset,
            atom_count,
            groups,
            is_literal,
            &assigned_group_slices,
            &inlined_literals,
            &mut external_ranges,
        );
    }

    // Merge overlapping ranges.
    let external_ranges = merge_group_ranges_c(&mut external_ranges);

    // Create placeholder groups for external input ranges.
    let mut input_mappings: Vec<AtomMapping> = Vec::new();
    for &(gi, offset, count) in &external_ranges {
        let main_base = groups[gi].base_id.offset(offset);
        let local_base = span_graph.push_group(
            count,
            ScalarOp::Literal(crate::numeric_scalar::NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );
        main_to_local.insert_range(main_base, local_base, count);
        input_mappings.push(AtomMapping {
            main_base,
            span_base: local_base,
            count,
        });
    }

    // Build the compute groups with remapped InputRefs.
    let mut output_mappings: Vec<AtomMapping> = Vec::new();

    for &(gi, atom_offset, atom_count) in &assigned_group_slices {
        let group = &groups[gi];

        let local_inputs = remap_inputs_range(
            &group.inputs,
            &group.op,
            atom_offset,
            atom_count,
            group.count,
            groups,
            &main_to_local,
        );
        let local_op = remap_op_range(&group.op, &main_to_local);

        let local_base = span_graph.push_group(
            atom_count,
            local_op,
            group.sym_dims.clone(),
            group.reduce_dims.clone(),
            local_inputs,
        );

        let main_base = AtomId(group.base_id.0 + atom_offset);
        main_to_local.insert_range(main_base, local_base, atom_count);

        // For duplicated groups, only output the lane's original slice
        // (to avoid duplicate outputs across lanes). For normal groups,
        // output the full work item.
        if let Some(per_lane) = dup_info.get(&gi) {
            let (orig_offset, orig_count) = per_lane[lane_idx];
            if orig_count > 0 {
                // This lane's original slice within the group.
                // atom_offset is always 0 for duplicated groups (full group).
                let out_main_base = AtomId(group.base_id.0 + orig_offset);
                let out_span_base = AtomId(local_base.0 + orig_offset);
                output_mappings.push(AtomMapping {
                    main_base: out_main_base,
                    span_base: out_span_base,
                    count: orig_count,
                });
            }
        } else {
            output_mappings.push(AtomMapping {
                main_base,
                span_base: local_base,
                count: atom_count,
            });
        }
    }

    // Mark graph outputs.
    let main_outputs: HashSet<AtomId> = main_graph.outputs.iter().copied().collect();
    for mapping in &output_mappings {
        for i in 0..mapping.count {
            let main_atom = mapping.main_base.offset(i);
            if main_outputs.contains(&main_atom) {
                span_graph.outputs.push(mapping.span_base.offset(i));
            }
        }
    }

    Span {
        graph: span_graph,
        inputs: input_mappings,
        outputs: output_mappings,
        literal_mappings: vec![],
    }
}

/// Collect external dependency ranges for a work item (sub-range of a group).
///
/// For each InputRef, identifies which groups are ACTUALLY referenced (not just
/// bounding-box overlap) and checks whether they're locally produced. Groups
/// that aren't locally produced become external inputs.
fn collect_external_ranges_c(
    group: &AtomGroup,
    atom_offset: u64,
    atom_count: u64,
    all_groups: &[AtomGroup],
    is_literal: &[bool],
    assigned_slices: &[(usize, u64, u64)],
    inlined_literals: &BTreeSet<usize>,
    external_ranges: &mut Vec<(usize, u64, u64)>,
) {
    if atom_count == 0 {
        return;
    }

    // Check if a range [atom_lo, atom_hi) within group `prod_gi` is fully
    // covered by the current lane's assigned work slices.
    let is_locally_covered = |atom_lo: u64, atom_hi: u64, prod_gi: usize| -> bool {
        let prod_base = all_groups[prod_gi].base_id.0;
        let off_lo = atom_lo - prod_base;
        let off_hi = atom_hi - prod_base;
        for &(work_gi, work_offset, work_count) in assigned_slices {
            if work_gi == prod_gi && work_offset <= off_lo && work_offset + work_count >= off_hi {
                return true;
            }
        }
        false
    };

    let should_skip = |gi: usize| -> bool {
        inlined_literals.contains(&gi)
            || (is_literal[gi] && all_groups[gi].count < LITERAL_INLINE_THRESHOLD)
    };

    let (is_reduce, reduce_count, reduce_stride) = match &group.op {
        ScalarOp::ReduceSum {
            reduce_count,
            reduce_stride,
            ..
        }
        | ScalarOp::ReduceMax {
            reduce_count,
            reduce_stride,
            ..
        } if *reduce_count > 1 && *reduce_stride != 0 => (true, *reduce_count, *reduce_stride),
        _ => (false, 0, 0),
    };

    // For each InputRef, find the ACTUAL groups referenced (not bounding-box).
    for input in &group.inputs {
        // Collect (group_idx, read_lo, read_hi) for actual references.
        let referenced = resolve_input_to_group_ranges(
            input,
            atom_offset,
            atom_count,
            if is_reduce { reduce_count } else { 1 },
            if is_reduce { reduce_stride } else { 0 },
            all_groups,
        );

        for (gi, range_lo, range_hi) in referenced {
            if should_skip(gi) {
                continue;
            }
            let g_lo = all_groups[gi].base_id.0;
            let overlap_lo = range_lo.max(g_lo);
            let overlap_hi = range_hi.min(g_lo + all_groups[gi].count);
            if overlap_lo >= overlap_hi {
                continue;
            }
            if !is_locally_covered(overlap_lo, overlap_hi, gi) {
                let offset = overlap_lo - g_lo;
                let count = overlap_hi - overlap_lo;
                external_ranges.push((gi, offset, count));
            }
        }
    }

    // IndirectLoad table.
    if let ScalarOp::IndirectLoad { table_base, .. } = &group.op {
        if let Some(gi) = find_group_idx(all_groups, *table_base) {
            if !inlined_literals.contains(&gi) {
                let g = &all_groups[gi];
                let is_assigned = assigned_slices.iter().any(|&(wgi, _, _)| wgi == gi);
                if !is_assigned {
                    external_ranges.push((gi, 0, g.count));
                }
            }
        }
    }
}

/// Resolve an InputRef (for a sub-range of a consumer group, including reduce
/// extension) to the actual set of (group_idx, atom_lo, atom_hi) tuples.
///
/// Unlike `input_ref_range` which returns a bounding box, this function
/// identifies which groups are ACTUALLY touched, avoiding false positives
/// from high-stride Affine patterns that span many unrelated groups.
fn resolve_input_to_group_ranges(
    input: &InputRef,
    offset: u64,
    count: u64,
    reduce_count: u64,
    reduce_stride: i64,
    groups: &[AtomGroup],
) -> Vec<(usize, u64, u64)> {
    let mut result = Vec::new();
    if count == 0 {
        return result;
    }

    match input {
        InputRef::Broadcast(atom_id) => {
            // Single atom, possibly extended by reduce stride.
            let base = atom_id.0 as i64;
            let (lo, hi) = reduce_extent(base, reduce_count, reduce_stride);
            for gi in find_groups_in_range(groups, lo as u64, hi as u64) {
                result.push((gi, lo as u64, (hi + 1) as u64));
            }
        }
        InputRef::Affine { base, stride } => {
            // Atoms: base + stride * (offset + i) for i in 0..count
            // With reduce: each atom extended by reduce_stride * k for k in 0..reduce_count
            let first_k = offset;
            let last_k = offset + count - 1;
            let first_pos = base.0 as i64 + *stride as i64 * first_k as i64;
            let last_pos = base.0 as i64 + *stride as i64 * last_k as i64;

            if *stride == 0 {
                // All atoms read the same position.
                let (lo, hi) = reduce_extent(first_pos, reduce_count, reduce_stride);
                for gi in find_groups_in_range(groups, lo as u64, hi as u64) {
                    result.push((gi, lo as u64, (hi + 1) as u64));
                }
            } else if stride.unsigned_abs() == 1 {
                // Stride 1 or -1: contiguous range, bounding box is exact.
                let base_lo = first_pos.min(last_pos);
                let base_hi = first_pos.max(last_pos);
                let (lo, hi) = reduce_extent_range(base_lo, base_hi, reduce_count, reduce_stride);
                for gi in find_groups_in_range(groups, lo as u64, hi as u64) {
                    result.push((gi, lo as u64, (hi + 1) as u64));
                }
            } else {
                // High stride: check each group in the bounding box individually.
                let base_lo = first_pos.min(last_pos);
                let base_hi = first_pos.max(last_pos);
                let (ext_lo, ext_hi) =
                    reduce_extent_range(base_lo, base_hi, reduce_count, reduce_stride);
                let candidates = find_groups_in_range(groups, ext_lo as u64, ext_hi as u64);
                for gi in candidates {
                    let g = &groups[gi];
                    let g_lo = g.base_id.0 as i64;
                    let g_hi = g_lo + g.count as i64;
                    // Check if any strided atom (possibly extended by reduce) lands in this group.
                    if affine_touches_range(
                        first_pos,
                        *stride as i64,
                        count,
                        g_lo,
                        g_hi,
                        reduce_count,
                        reduce_stride,
                    ) {
                        let overlap_lo = (g_lo as u64).max(ext_lo as u64);
                        let overlap_hi = (g_hi as u64).min((ext_hi + 1) as u64);
                        result.push((gi, overlap_lo, overlap_hi));
                    }
                }
            }
        }
        InputRef::StridedBroadcast {
            base,
            stride,
            repeat,
        } => {
            // atom i reads base + stride * (i / repeat)
            // For sub-range [offset, offset+count): blocks offset/repeat .. (offset+count-1)/repeat
            let first_block = offset / repeat;
            let last_block = (offset + count - 1) / repeat;
            for block in first_block..=last_block {
                let pos = base.0 as i64 + *stride * block as i64;
                let (lo, hi) = reduce_extent(pos, reduce_count, reduce_stride);
                for gi in find_groups_in_range(groups, lo as u64, hi as u64) {
                    result.push((gi, lo as u64, (hi + 1) as u64));
                }
            }
        }
        InputRef::Modular {
            base,
            stride,
            modulus,
        } => {
            // Reads base + stride * (k % modulus) for k in [offset, offset+count).
            // The set of distinct values is base + stride * j for j in 0..modulus.
            if *modulus == 0 {
                return result;
            }
            let num_distinct = (*modulus).min(count);
            // Compute bounding box of the modular pattern.
            let mut lo = base.0 as i64;
            let mut hi = base.0 as i64;
            for j in 0..num_distinct {
                let pos = base.0 as i64 + *stride as i64 * j as i64;
                lo = lo.min(pos);
                hi = hi.max(pos);
            }
            let (ext_lo, ext_hi) = reduce_extent_range(lo, hi, reduce_count, reduce_stride);
            for gi in find_groups_in_range(groups, ext_lo as u64, ext_hi as u64) {
                result.push((gi, ext_lo as u64, (ext_hi + 1) as u64));
            }
        }
        InputRef::Explicit(ids) => {
            // Resolve each atom in the sub-range and track the min/max
            // range per group.
            let start = offset as usize;
            let end = ((offset + count) as usize).min(ids.len());
            let mut group_ranges: BTreeMap<usize, (u64, u64)> = BTreeMap::new();
            for i in start..end {
                let atom = ids[i];
                let (lo, hi) = reduce_extent(atom.0 as i64, reduce_count, reduce_stride);
                for gi in find_groups_in_range(groups, lo as u64, hi as u64) {
                    let entry = group_ranges
                        .entry(gi)
                        .or_insert((lo as u64, (hi + 1) as u64));
                    entry.0 = entry.0.min(lo as u64);
                    entry.1 = entry.1.max((hi + 1) as u64);
                }
            }
            for (gi, (lo, hi)) in group_ranges {
                result.push((gi, lo, hi));
            }
        }
        InputRef::SymAffine {
            base,
            stride_i,
            stride_k,
        } => {
            // atom i reads base + stride_i * i (stride_k is for the sym_dim loop).
            // For sub-range [offset, offset+count):
            let first_pos = base.0 as i64 + *stride_i as i64 * offset as i64;
            let last_pos = base.0 as i64 + *stride_i as i64 * (offset + count - 1) as i64;
            let base_lo = first_pos.min(last_pos);
            let base_hi = first_pos.max(last_pos);
            // stride_k extends in the k dimension — we don't know k at plan time.
            // Use bounding box for the i dimension; stride_k is handled by the
            // reduce/sym_dim loop at runtime.
            let (ext_lo, ext_hi) =
                reduce_extent_range(base_lo, base_hi, reduce_count, reduce_stride);

            if stride_i.unsigned_abs() <= 1 {
                // Contiguous or broadcast in i: bounding box is exact.
                for gi in find_groups_in_range(groups, ext_lo as u64, ext_hi as u64) {
                    result.push((gi, ext_lo as u64, (ext_hi + 1) as u64));
                }
            } else {
                // High stride in i: check each candidate group.
                let candidates = find_groups_in_range(groups, ext_lo as u64, ext_hi as u64);
                for gi in candidates {
                    let g = &groups[gi];
                    let g_lo = g.base_id.0 as i64;
                    let g_hi = g_lo + g.count as i64;
                    if affine_touches_range(
                        first_pos,
                        *stride_i as i64,
                        count,
                        g_lo,
                        g_hi,
                        reduce_count,
                        reduce_stride,
                    ) {
                        let overlap_lo = (g_lo as u64).max(ext_lo as u64);
                        let overlap_hi = (g_hi as u64).min((ext_hi + 1) as u64);
                        result.push((gi, overlap_lo, overlap_hi));
                    }
                }
            }
        }
    }

    result
}

/// Compute the extent [lo, hi] (inclusive) of a single position extended by reduce stride.
fn reduce_extent(pos: i64, reduce_count: u64, reduce_stride: i64) -> (i64, i64) {
    if reduce_count <= 1 {
        return (pos, pos);
    }
    let ext = reduce_stride * (reduce_count as i64 - 1);
    (pos + ext.min(0), pos + ext.max(0))
}

/// Compute the extent [lo, hi] (inclusive) of a range [base_lo, base_hi] extended by reduce stride.
fn reduce_extent_range(
    base_lo: i64,
    base_hi: i64,
    reduce_count: u64,
    reduce_stride: i64,
) -> (i64, i64) {
    if reduce_count <= 1 {
        return (base_lo, base_hi);
    }
    let ext = reduce_stride * (reduce_count as i64 - 1);
    (base_lo + ext.min(0), base_hi + ext.max(0))
}

/// Check if any atom in the strided sequence first_pos, first_pos+stride, ..., first_pos+stride*(count-1)
/// (optionally extended by reduce_stride * k for k in 0..reduce_count)
/// falls within the range [g_lo, g_hi).
fn affine_touches_range(
    first_pos: i64,
    stride: i64,
    count: u64,
    g_lo: i64,
    g_hi: i64,
    reduce_count: u64,
    reduce_stride: i64,
) -> bool {
    if count == 0 || g_lo >= g_hi {
        return false;
    }

    // Extend the group range inward by the reduce extent to simplify:
    // an atom at position p touches the group if any p + reduce_stride*k is in [g_lo, g_hi).
    // Equivalently, p is in [g_lo - max_reduce_ext, g_hi - min_reduce_ext).
    let (min_reduce_ext, max_reduce_ext) = if reduce_count > 1 {
        let ext = reduce_stride * (reduce_count as i64 - 1);
        (ext.min(0), ext.max(0))
    } else {
        (0, 0)
    };
    let eff_lo = g_lo - max_reduce_ext;
    let eff_hi = g_hi - min_reduce_ext;

    if stride == 0 {
        return first_pos >= eff_lo && first_pos < eff_hi;
    }

    // Find if any integer i in [0, count) satisfies: eff_lo <= first_pos + stride * i < eff_hi
    // Rearranging: (eff_lo - first_pos) / stride <= i < (eff_hi - first_pos) / stride
    // (careful with sign of stride for division direction)
    let (i_lo, i_hi) = if stride > 0 {
        // i >= ceil((eff_lo - first_pos) / stride)
        // i < ceil((eff_hi - first_pos) / stride)
        let num_lo = eff_lo - first_pos;
        let num_hi = eff_hi - first_pos;
        (
            div_ceil_signed(num_lo, stride),
            div_ceil_signed(num_hi, stride),
        )
    } else {
        // stride < 0: dividing by negative flips inequality
        // i >= ceil((eff_hi - 1 - first_pos) / stride)  -- tricky with negative stride
        // Simpler: pos = first_pos + stride*i >= eff_lo and pos < eff_hi
        // first_pos + stride*i >= eff_lo  =>  stride*i >= eff_lo - first_pos  =>  i <= (eff_lo - first_pos) / stride (since stride < 0)
        // first_pos + stride*i < eff_hi   =>  stride*i < eff_hi - first_pos   =>  i > (eff_hi - first_pos) / stride (since stride < 0)
        // So i in (floor((eff_hi - first_pos - 1) / stride), floor((eff_lo - first_pos) / stride)]
        // i.e., i_lo = floor((eff_hi - first_pos - 1) / stride) + 1, i_hi = floor((eff_lo - first_pos) / stride) + 1
        let neg_stride = -stride; // positive
        // first_pos + stride*i >= eff_lo => i <= (first_pos - eff_lo) / neg_stride
        // first_pos + stride*i < eff_hi  => i > (first_pos - eff_hi) / neg_stride
        //                                => i >= floor((first_pos - eff_hi) / neg_stride) + 1
        //                                   BUT need ceiling of (first_pos - eff_hi + 1) / neg_stride
        let i_max = div_floor_signed(first_pos - eff_lo, neg_stride);
        let i_min = div_ceil_signed(first_pos - eff_hi + 1, neg_stride);
        (i_min, i_max + 1) // [i_min, i_max] => [i_min, i_max+1) for consistency
    };

    // Check if [i_lo, i_hi) intersects [0, count).
    let valid_lo = i_lo.max(0);
    let valid_hi = i_hi.min(count as i64);
    valid_lo < valid_hi
}

fn div_ceil_signed(a: i64, b: i64) -> i64 {
    assert!(b > 0);
    if a >= 0 { (a + b - 1) / b } else { -((-a) / b) }
}

fn div_floor_signed(a: i64, b: i64) -> i64 {
    assert!(b > 0);
    if a >= 0 { a / b } else { -(((-a) + b - 1) / b) }
}

/// Compute the [lo, hi) atom range of an InputRef considering reduce stride.
fn input_ref_range(
    input: &InputRef,
    offset: u64,
    count: u64,
    reduce_count: u64,
    reduce_stride: i64,
) -> Option<(u64, u64)> {
    if count == 0 {
        return None;
    }
    let first = input.resolve(offset, 0).0 as i64;
    let last = input.resolve(offset + count - 1, 0).0 as i64;
    let min_reduce = 0i64.min(reduce_stride * (reduce_count as i64 - 1));
    let max_reduce = 0i64.max(reduce_stride * (reduce_count as i64 - 1));
    let lo = first.min(last) + min_reduce;
    let hi = first.max(last) + max_reduce + 1;
    Some((lo as u64, hi as u64))
}

/// Merge overlapping/adjacent ranges.
fn merge_group_ranges_c(ranges: &mut Vec<(usize, u64, u64)>) -> Vec<(usize, u64, u64)> {
    if ranges.is_empty() {
        return vec![];
    }
    ranges.sort_by_key(|&(gi, off, _)| (gi, off));
    let mut merged: Vec<(usize, u64, u64)> = Vec::new();
    for &(gi, off, count) in ranges.iter() {
        if let Some(last) = merged.last_mut() {
            if last.0 == gi && off <= last.1 + last.2 {
                let new_end = (off + count).max(last.1 + last.2);
                last.2 = new_end - last.1;
                continue;
            }
        }
        merged.push((gi, off, count));
    }
    merged
}

/// Remap InputRefs using RangeAtomMap.
fn remap_inputs_range(
    inputs: &[InputRef],
    op: &ScalarOp,
    atom_offset: u64,
    atom_count: u64,
    orig_group_count: u64,
    groups: &[AtomGroup],
    atom_map: &RangeAtomMap,
) -> Vec<InputRef> {
    inputs
        .iter()
        .map(|input| {
            remap_single_input_range(input, atom_offset, atom_count, orig_group_count, atom_map)
        })
        .collect()
}

fn remap_single_input_range(
    input: &InputRef,
    atom_offset: u64,
    atom_count: u64,
    orig_group_count: u64,
    atom_map: &RangeAtomMap,
) -> InputRef {
    match input {
        InputRef::Broadcast(id) => InputRef::Broadcast(atom_map.get(*id).unwrap_or(*id)),
        InputRef::Affine { base, stride } => {
            // For sub-range: new base = base + stride * atom_offset.
            let new_base_raw = AtomId(
                base.0
                    .wrapping_add((*stride as i64 * atom_offset as i64) as u64),
            );
            InputRef::Affine {
                base: atom_map.get(new_base_raw).unwrap_or(new_base_raw),
                stride: *stride,
            }
        }
        InputRef::StridedBroadcast {
            base,
            stride,
            repeat,
        } => {
            // Adjust base for sub-range offset.
            let block_idx = (atom_offset / repeat) as i64;
            let new_base_raw = AtomId(base.0.wrapping_add((stride * block_idx) as u64));
            let new_offset_in_block = atom_offset % repeat;
            // If starting mid-block, the repeat pattern shifts.
            if new_offset_in_block == 0 {
                InputRef::StridedBroadcast {
                    base: atom_map.get(new_base_raw).unwrap_or(new_base_raw),
                    stride: *stride,
                    repeat: *repeat,
                }
            } else {
                // Complex sub-range: fall back to explicit.
                let mut ids = Vec::with_capacity(atom_count as usize);
                for i in 0..atom_count {
                    let main_id = input.resolve(atom_offset + i, 0);
                    ids.push(atom_map.get(main_id).unwrap_or(main_id));
                }
                InputRef::Explicit(ids)
            }
        }
        InputRef::Modular {
            base,
            stride,
            modulus,
        } => {
            if atom_offset % modulus == 0 {
                InputRef::Modular {
                    base: atom_map.get(*base).unwrap_or(*base),
                    stride: *stride,
                    modulus: *modulus,
                }
            } else {
                // Misaligned split: fall back to explicit
                let mut ids = Vec::with_capacity(atom_count as usize);
                for i in 0..atom_count {
                    let main_id = input.resolve(atom_offset + i, 0);
                    ids.push(atom_map.get(main_id).unwrap_or(main_id));
                }
                InputRef::Explicit(ids)
            }
        }
        InputRef::SymAffine {
            base,
            stride_i,
            stride_k,
        } => {
            let new_base_raw = AtomId(
                base.0
                    .wrapping_add((*stride_i as i64 * atom_offset as i64) as u64),
            );
            InputRef::SymAffine {
                base: atom_map.get(new_base_raw).unwrap_or(new_base_raw),
                stride_i: *stride_i,
                stride_k: *stride_k,
            }
        }
        InputRef::Explicit(ids) => {
            let start = atom_offset as usize;
            let end = (atom_offset + atom_count) as usize;
            let slice = if end <= ids.len() {
                &ids[start..end]
            } else {
                &ids[start..]
            };
            InputRef::Explicit(
                slice
                    .iter()
                    .map(|id| atom_map.get(*id).unwrap_or(*id))
                    .collect(),
            )
        }
    }
}

/// Remap ScalarOp using RangeAtomMap.
fn remap_op_range(op: &ScalarOp, atom_map: &RangeAtomMap) -> ScalarOp {
    match op {
        ScalarOp::IndirectLoad {
            table_base,
            output_dtype,
        } => ScalarOp::IndirectLoad {
            table_base: atom_map.get(*table_base).unwrap_or(*table_base),
            output_dtype: *output_dtype,
        },
        other => other.clone(),
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
        InputRef::StridedBroadcast {
            base,
            stride,
            repeat,
        } => {
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
        InputRef::Modular {
            base,
            stride,
            modulus,
        } => {
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
        ScalarOp::ReduceSum {
            reduce_count,
            reduce_stride,
            ..
        }
        | ScalarOp::ReduceMax {
            reduce_count,
            reduce_stride,
            ..
        } if *reduce_count > 1 && *reduce_stride != 0 => {
            for input in &group.inputs {
                collect_input_atoms_with_reduce(
                    input,
                    first_i,
                    last_i,
                    *reduce_count,
                    *reduce_stride,
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
fn collect_input_atoms(input: &InputRef, first_i: u64, last_i: u64, atoms: &mut BTreeSet<AtomId>) {
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
        InputRef::StridedBroadcast {
            base,
            stride,
            repeat,
        } => {
            let first_block = first_i / repeat;
            let last_block = last_i / repeat;
            for block in first_block..=last_block {
                let atom = AtomId(base.0.wrapping_add((*stride * block as i64) as u64));
                atoms.insert(atom);
            }
        }
        InputRef::Modular {
            base,
            stride,
            modulus,
        } => {
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
        InputRef::SymAffine {
            base,
            stride_i,
            stride_k,
        } => {
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
        InputRef::StridedBroadcast {
            base,
            stride,
            repeat,
        } => {
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
        .map(|input| {
            remap_single_input(
                input,
                atom_offset,
                atom_count,
                group_count,
                is_full,
                main_to_local,
            )
        })
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
            let local = main_to_local.get(atom_id).copied().unwrap_or(*atom_id);
            InputRef::Broadcast(local)
        }
        InputRef::Affine { base, stride } => {
            // For a sub-range starting at atom_offset, the effective base shifts.
            // Original: atom i reads base + stride * i
            // Sub-range: atom j (j=0..atom_count) reads base + stride * (atom_offset + j)
            //          = (base + stride * atom_offset) + stride * j
            let effective_base = AtomId(
                base.0
                    .wrapping_add((*stride as i64 * atom_offset as i64) as u64),
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
        InputRef::StridedBroadcast {
            base,
            stride,
            repeat,
        } => {
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
                let effective_base =
                    AtomId(base.0.wrapping_add((*stride * block_offset as i64) as u64));
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
                    let local = main_to_local.get(&main_atom).copied().unwrap_or(main_atom);
                    ids.push(local);
                }
                InputRef::Explicit(ids)
            }
        }
        InputRef::Modular {
            base,
            stride,
            modulus,
        } => {
            // Modular: atom i reads base + stride * (i % modulus).
            // For sub-range: atom j reads base + stride * ((atom_offset + j) % modulus).
            // When atom_offset % modulus == 0, the pattern is unchanged (just remap base).
            // When misaligned, the phase shift means we must fall back to explicit.
            let local_base = main_to_local.get(base).copied().unwrap_or(*base);
            if atom_offset % modulus == 0 {
                InputRef::Modular {
                    base: local_base,
                    stride: *stride,
                    modulus: *modulus,
                }
            } else {
                // Misaligned split: fall back to explicit
                let mut ids = Vec::with_capacity(atom_count as usize);
                for j in 0..atom_count {
                    let main_atom = input.resolve(atom_offset + j, 0);
                    let local = main_to_local.get(&main_atom).copied().unwrap_or(main_atom);
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
        InputRef::SymAffine {
            base,
            stride_i,
            stride_k,
        } => {
            // Sub-range: atom j reads base + stride_i * (atom_offset + j) + stride_k * k
            //          = (base + stride_i * atom_offset) + stride_i * j + stride_k * k
            let effective_base = AtomId(
                base.0
                    .wrapping_add((*stride_i as i64 * atom_offset as i64) as u64),
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
        ScalarOp::IndirectLoad {
            table_base,
            output_dtype,
        } => {
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

/// Print summary statistics about the span plan.
pub fn print_summary(plan: &SpanPlan) {
    let total_spans: usize = plan.phases.iter().map(|p| p.spans.len()).sum();
    let non_empty_spans: usize = plan
        .phases
        .iter()
        .flat_map(|p| p.spans.iter())
        .filter(|s| s.graph.num_groups() > 0)
        .count();
    let total_inputs: usize = plan
        .phases
        .iter()
        .flat_map(|p| p.spans.iter())
        .map(|s| s.inputs.len())
        .sum();
    let total_outputs: usize = plan
        .phases
        .iter()
        .flat_map(|p| p.spans.iter())
        .map(|s| s.outputs.len())
        .sum();

    println!(
        "SpanPlan: {} lanes, {} phases, {} spans ({} non-empty)",
        plan.num_lanes,
        plan.phases.len(),
        total_spans,
        non_empty_spans
    );
    println!(
        "  Total inputs: {}, Total outputs: {}",
        total_inputs, total_outputs
    );

    for (phase_idx, phase) in plan.phases.iter().enumerate() {
        let active = phase
            .spans
            .iter()
            .filter(|s| s.graph.num_groups() > 0)
            .count();
        let max_groups = phase
            .spans
            .iter()
            .map(|s| s.graph.num_groups())
            .max()
            .unwrap_or(0);
        let max_atoms = phase
            .spans
            .iter()
            .map(|s| s.graph.num_atoms())
            .max()
            .unwrap_or(0);
        let min_atoms = phase
            .spans
            .iter()
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
pub fn validate(plan: &SpanPlan, original: &NanoGraph) -> Vec<String> {
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

    for (phase_idx, phase) in plan.phases.iter().enumerate() {
        for (lane_idx, span) in phase.spans.iter().enumerate() {
            // Check 1: NanoGraph internal validation.
            let graph_errors = span.graph.validate();
            for err in graph_errors {
                errors.push(format!(
                    "Phase {} lane {}: graph validation error: {}",
                    phase_idx, lane_idx, err
                ));
            }

            // Check 2: All external input base atoms must be available.
            for mapping in &span.inputs {
                if !available_atoms.contains(&mapping.main_base) {
                    errors.push(format!(
                        "Phase {} lane {}: external input base {:?} not available \
                         (not produced by any earlier phase)",
                        phase_idx, lane_idx, mapping.main_base
                    ));
                }
            }
        }

        // After processing all spans in this phase, add their outputs
        // to the available set.
        for span in &phase.spans {
            for mapping in &span.outputs {
                for i in 0..mapping.count {
                    available_atoms.insert(mapping.main_base.offset(i));
                }
            }
        }
    }

    errors
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
            vec![],
            vec![],
            vec![],
        );
        let b_base = g.push_group(
            k * n,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
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
                vec![],
                vec![],
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
                vec![],
                vec![],
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
        let a1 = g.push_group(
            m * k1,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let b1 = g.push_group(
            k1 * n1,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        let mut mul1_bases = Vec::new();
        for row in 0..m {
            let a_row = AtomId(a1.0 + row * k1);
            let mul = g.push_group(
                k1 * n1,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![
                    InputRef::StridedBroadcast {
                        base: a_row,
                        stride: 1,
                        repeat: n1,
                    },
                    InputRef::Affine {
                        base: b1,
                        stride: 1,
                    },
                ],
            );
            mul1_bases.push(mul);
        }
        let mut red1_bases = Vec::new();
        for row in 0..m {
            let red = g.push_group(
                n1,
                ScalarOp::ReduceSum {
                    reduce_count: k1,
                    reduce_stride: n1 as i64,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![InputRef::Affine {
                    base: mul1_bases[row as usize],
                    stride: 1,
                }],
            );
            red1_bases.push(red);
        }

        let act = g.push_group(
            m * n1,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Tanh,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: red1_bases[0],
                stride: 1,
            }],
        );

        let b2 = g.push_group(
            n1 * n2,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        let mut mul2_bases = Vec::new();
        for row in 0..m {
            let mul = g.push_group(
                n1 * n2,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![
                    InputRef::StridedBroadcast {
                        base: AtomId(act.0 + row * n1),
                        stride: 1,
                        repeat: n2,
                    },
                    InputRef::Affine {
                        base: b2,
                        stride: 1,
                    },
                ],
            );
            mul2_bases.push(mul);
        }
        let mut red2_bases = Vec::new();
        for row in 0..m {
            let red = g.push_group(
                n2,
                ScalarOp::ReduceSum {
                    reduce_count: n1,
                    reduce_stride: n2 as i64,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![InputRef::Affine {
                    base: mul2_bases[row as usize],
                    stride: 1,
                }],
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
        let a = g.push_group(
            count,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let b = g.push_group(
            count,
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
            vec![],
        );
        let c = g.push_group(
            count,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
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
        let lit = g.push_group(
            count,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let a = g.push_group(
            count,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Tanh,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: lit,
                stride: 1,
            }],
        );
        let b = g.push_group(
            count,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Tanh,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: a, stride: 1 }],
        );
        let c = g.push_group(
            count,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Tanh,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: b, stride: 1 }],
        );
        for i in 0..count {
            g.outputs.push(AtomId(c.0 + i));
        }
        g
    }

    fn build_parallel_matmuls(m: u64, k: u64, n: u64) -> NanoGraph {
        let mut g = NanoGraph::new();
        let a = g.push_group(
            m * k,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let b1 = g.push_group(
            k * n,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let b2 = g.push_group(
            k * n,
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
            vec![],
        );

        let mut all_red_bases = Vec::new();
        for (b_base, _) in [(b1, 1.0f32), (b2, 2.0)] {
            for row in 0..m {
                let a_row = AtomId(a.0 + row * k);
                let mul = g.push_group(
                    k * n,
                    ScalarOp::Binary {
                        op: ScalarBinOp::Mul,
                        compute_dtype: DType::F32,
                        output_dtype: DType::F32,
                    },
                    vec![],
                    vec![],
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
                let red = g.push_group(
                    n,
                    ScalarOp::ReduceSum {
                        reduce_count: k,
                        reduce_stride: n as i64,
                        compute_dtype: DType::F32,
                        output_dtype: DType::F32,
                    },
                    vec![],
                    vec![],
                    vec![InputRef::Affine {
                        base: mul,
                        stride: 1,
                    }],
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
        let errors = validate(plan, graph);
        assert!(
            errors.is_empty(),
            "Span plan validation errors:\n{}",
            errors.join("\n")
        );
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
                for mapping in &span.outputs {
                    for i in 0..mapping.count {
                        *produced.entry(mapping.main_base.offset(i)).or_insert(0) += 1;
                    }
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
                for mapping in &span.inputs {
                    // Check base atom is available.
                    assert!(
                        available.contains(&mapping.main_base),
                        "Phase {} lane {}: external input base {:?} not yet produced",
                        phase_idx,
                        lane_idx,
                        mapping.main_base
                    );
                }
            }
            // Add outputs from this phase.
            for span in &phase.spans {
                for mapping in &span.outputs {
                    for i in 0..mapping.count {
                        available.insert(mapping.main_base.offset(i));
                    }
                }
            }
        }
    }

    /// Verify no cross-span dependencies within a phase.
    /// Within a phase, each span's external inputs must come from earlier phases,
    /// not from other spans in the same phase.
    fn verify_phase_independence(plan: &SpanPlan) {
        for (phase_idx, phase) in plan.phases.iter().enumerate() {
            // Collect output ranges for this phase.
            let mut this_phase_output_ranges: Vec<(u64, u64)> = Vec::new();
            for span in &phase.spans {
                for mapping in &span.outputs {
                    this_phase_output_ranges
                        .push((mapping.main_base.0, mapping.main_base.0 + mapping.count));
                }
            }

            // No span's external input range should overlap with outputs in the same phase.
            for (lane_idx, span) in phase.spans.iter().enumerate() {
                for mapping in &span.inputs {
                    let in_lo = mapping.main_base.0;
                    let in_hi = in_lo + mapping.count;
                    for &(out_lo, out_hi) in &this_phase_output_ranges {
                        assert!(
                            in_lo >= out_hi || out_lo >= in_hi,
                            "Phase {} lane {}: external input range [{}, {}) overlaps with \
                             same-phase output range [{}, {}) — cross-span dependency!",
                            phase_idx,
                            lane_idx,
                            in_lo,
                            in_hi,
                            out_lo,
                            out_hi
                        );
                    }
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
        print_summary(&plan);
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
        let active = plan.phases[0]
            .spans
            .iter()
            .filter(|s| s.graph.num_groups() > 0)
            .count();
        assert_eq!(active, 2);

        println!("Matmul 2 lanes:");
        print_summary(&plan);
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
        print_summary(&plan);
    }

    #[test]
    fn test_matmul_chain_monolithic() {
        let g = build_matmul_chain(4, 4, 4, 4, 4);
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let plan = plan_execution_spans(&g, 2);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);

        // Multi-phase: matmul1, activation, matmul2.
        assert!(
            plan.phases.len() >= 2,
            "Should have >= 2 phases, got {}",
            plan.phases.len()
        );

        println!("Matmul chain (monolithic) 2 lanes:");
        print_summary(&plan);
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
        let active = plan.phases[0]
            .spans
            .iter()
            .filter(|s| s.graph.num_groups() > 0)
            .count();
        assert!(active >= 2, "Should have >= 2 active lanes, got {}", active);

        println!("Elementwise 4 lanes:");
        print_summary(&plan);
    }

    #[test]
    fn test_allrows_chain() {
        let g = build_allrows_chain(1024);
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let plan = plan_execution_spans(&g, 8);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);

        // AllRows chain with same count: 1 phase.
        assert_eq!(
            plan.phases.len(),
            1,
            "AllRows chain should be 1 phase, got {}",
            plan.phases.len()
        );

        println!("AllRows chain 8 lanes:");
        print_summary(&plan);
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
        print_summary(&plan);
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
                    phase_idx,
                    lane_idx,
                    errors
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
                name,
                plan.phases.len(),
                plan.num_lanes
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

        assert!(
            plan.phases.len() <= 10,
            "Should have <= 10 phases, got {}",
            plan.phases.len()
        );

        println!("Larger matmul chain 4 lanes:");
        print_summary(&plan);
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
        print_summary(&plan);
    }

    #[test]
    fn test_uneven_row_count() {
        let g = build_matmul(7, 4, 4);
        assert!(g.validate().is_empty());

        let plan = plan_execution_spans(&g, 4);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);
    }

    /// Build a graph where a large AllRows group (produced in phase 0) feeds
    /// into a consumer in phase 1 via a ReduceSum with large reduce_stride.
    ///
    /// The ReduceSum's `reduce_stride * reduce_count` bounding box can be very
    /// wide, overlapping many unrelated groups. The planner must only declare
    /// ACTUAL dependencies as external inputs, not bounding-box false positives.
    fn build_large_allrows_reduce(
        total_atoms: u64,
        reduce_count: u64,
        reduce_stride: i64,
    ) -> NanoGraph {
        let mut g = NanoGraph::new();

        // Source: a large AllRows group from a literal
        let src_lit = g.push_group(
            total_atoms,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        // Elementwise tanh on it (so it's a compute group, AllRows)
        let src = g.push_group(
            total_atoms,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Tanh,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: src_lit,
                stride: 1,
            }],
        );

        // Another AllRows group that's independent (to place in same phase as src)
        let other_lit = g.push_group(
            total_atoms,
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
            vec![],
        );
        let other = g.push_group(
            total_atoms,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Tanh,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: other_lit,
                stride: 1,
            }],
        );

        // ReduceSum that reads from `src` with large stride.
        // This is an AllRows consumer of an AllRows producer,
        // but with stride it may need a barrier.
        let out_count = total_atoms / (reduce_count * reduce_stride.unsigned_abs());
        let red = g.push_group(
            out_count.max(1),
            ScalarOp::ReduceSum {
                reduce_count,
                reduce_stride,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: src,
                stride: 1,
            }],
        );

        for i in 0..out_count.max(1) {
            g.outputs.push(AtomId(red.0 + i));
        }
        // Also output the other group so it's not dead
        for i in 0..total_atoms {
            g.outputs.push(AtomId(other.0 + i));
        }
        g
    }

    #[test]
    fn test_reduce_stride_no_false_deps() {
        // ReduceSum with large stride: bounding box spans the full src group.
        // The planner should only declare src as external, not the unrelated
        // `other` group that happens to be in the bounding box.
        let g = build_large_allrows_reduce(1024, 8, 128);
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let plan = plan_execution_spans(&g, 4);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);
        verify_input_availability(&plan, &g);

        println!("Reduce stride consumer 4 lanes:");
        print_summary(&plan);
    }

    #[test]
    fn test_reduce_stride_8lanes() {
        let g = build_large_allrows_reduce(2048, 16, 128);
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let plan = plan_execution_spans(&g, 8);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);
        verify_input_availability(&plan, &g);

        println!("Reduce stride consumer 8 lanes:");
        print_summary(&plan);
    }

    /// Build a graph with a cross-lane dependency: an AllRows Select-like group
    /// consumed in the same phase by another group via a non-lane-local pattern.
    ///
    /// This mimics GPT-2's attention mask: a Select (count=mask_size) is read
    /// by a larger consumer (count=consumer_size) via StridedBroadcast (each
    /// consumer atom reads from a proportionally different spot in the Select).
    /// When both are AllRows and in the same phase, splitting the Select across
    /// lanes creates a cross-lane dependency that must be fixed by duplication.
    fn build_cross_lane_select(mask_size: u64, consumer_size: u64) -> NanoGraph {
        let mut g = NanoGraph::new();

        // Condition literal for Select
        let cond_lit = g.push_group(
            mask_size,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        // Value literals for Select
        let val_a = g.push_group(
            mask_size,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );
        let val_b = g.push_group(
            mask_size,
            ScalarOp::Literal(NumericScalar::F32(-1e9)),
            vec![],
            vec![],
            vec![],
        );

        // Select group (AllRows, mask_size atoms)
        let select = g.push_group(
            mask_size,
            ScalarOp::Select {
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine {
                    base: cond_lit,
                    stride: 1,
                },
                InputRef::Affine {
                    base: val_a,
                    stride: 1,
                },
                InputRef::Affine {
                    base: val_b,
                    stride: 1,
                },
            ],
        );

        // Data literal for the consumer
        let data_lit = g.push_group(
            consumer_size,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        // Consumer: reads data_lit 1:1 and Select via StridedBroadcast
        // (consumer_size atoms, each reading from select with repeat pattern).
        // This means consumer atom i reads select atom i / repeat.
        let repeat = consumer_size / mask_size; // e.g., 8
        let consumer = g.push_group(
            consumer_size,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine {
                    base: data_lit,
                    stride: 1,
                },
                InputRef::StridedBroadcast {
                    base: select,
                    stride: 1,
                    repeat,
                },
            ],
        );

        for i in 0..consumer_size {
            g.outputs.push(AtomId(consumer.0 + i));
        }
        g
    }

    #[test]
    fn test_cross_lane_select_duplication() {
        // Mimics GPT-2 attention mask pattern: Select(3072) consumed by
        // a larger group(24576) via StridedBroadcast{repeat=8}.
        // Both are AllRows in the same phase. Without the fix, splitting
        // the Select across 8 lanes creates 96-like violations.
        let g = build_cross_lane_select(3072, 3072 * 8);
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let plan = plan_execution_spans(&g, 8);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);
        verify_input_availability(&plan, &g);

        println!("Cross-lane Select 8 lanes:");
        print_summary(&plan);
    }

    #[test]
    fn test_cross_lane_select_small() {
        // Smaller variant to test basic correctness.
        let g = build_cross_lane_select(64, 512);
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let plan = plan_execution_spans(&g, 4);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);
        verify_input_availability(&plan, &g);
    }

    #[test]
    fn test_cross_lane_broadcast_from_split() {
        // An AllRows group consumed via Broadcast (single atom) by another
        // same-phase AllRows group. This is a degenerate cross-lane case.
        let mut g = NanoGraph::new();
        let lit = g.push_group(
            1024,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let src = g.push_group(
            1024,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Tanh,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: lit,
                stride: 1,
            }],
        );
        // Consumer broadcasts a single atom from src. When src is split
        // across lanes, only one lane owns that atom.
        let consumer = g.push_group(
            1024,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine {
                    base: lit,
                    stride: 1,
                },
                InputRef::Broadcast(src), // reads atom 0 of src
            ],
        );
        for i in 0..1024 {
            g.outputs.push(AtomId(consumer.0 + i));
        }
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let plan = plan_execution_spans(&g, 4);
        verify_span_plan(&g, &plan);
        verify_output_coverage(&g, &plan);
        verify_input_availability(&plan, &g);
    }
}
