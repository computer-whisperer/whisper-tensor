#![allow(
    clippy::all,
    dead_code,
    unreachable_code,
    unreachable_patterns,
    unused_imports,
    unused_variables
)]

//! Stripe-first parallel partitioner (attempt L).
//!
//! Key insight: the natural unit of parallelism is the atom-range slice, not
//! the group. Lane assignment IS tiling — lane 0 always gets the first 1/K of
//! every splittable group. This preserves cache affinity across consecutive ops
//! and eliminates barriers between chains of identically-split operations.
//!
//! Algorithm:
//! 1. Build group-level dependency DAG.
//! 2. Walk groups topologically, assigning each to a "split class":
//!    - SPLIT: elementwise, matmul-mul, indirect load, select — split across lanes
//!    - DUPLICATE: literals — duplicate into every lane
//!    - WHOLE: reduces or tiny groups — run on one lane (or all lanes if small)
//! 3. Determine phases: a barrier is needed only when a downstream op reads
//!    atoms produced by multiple lanes (e.g., a reduce reading split data from
//!    a different phase). Chains of identically-split ops need zero barriers.
//! 4. Build spans: for each lane in each phase, construct a NanoGraph fragment
//!    with split fragments, duplicated literals, and external input declarations.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::graph::GlobalId;
use crate::nano_graph::pattern::InputTensor;
use crate::nano_graph::{AtomGroup, AtomId, AtomRange, InputRef, NanoGraph, ScalarOp, SymDim};
use crate::numeric_dtype::NumericDType;

use super::types::{Phase, Span};

// ─── Public API ────────────────────────────────────────────────────────────

pub fn plan(
    graph: &NanoGraph<'static, crate::pool::SystemPool>,
    num_lanes: usize,
    input_tensors: &[InputTensor],
    output_atom_ids: &[AtomId],
) -> Vec<Phase> {
    let num_lanes = num_lanes.max(1);
    let groups = graph.groups();
    let n = groups.len();

    if n == 0 {
        return vec![Phase {
            spans: (0..num_lanes).map(|_| empty_span()).collect(),
        }];
    }

    // Step 1: Build group-level dependency DAG.
    let (producers, successors) = build_dependency_dag(graph);

    // Step 2: Classify each group.
    let classes = classify_groups(groups, num_lanes);

    // Step 3: Assign phases. A group needs a new phase only if it depends on
    // a group that was SPLIT in a prior phase and is on a different lane-slice.
    // Specifically: if a non-split group (WHOLE) reads from a split group, it
    // needs a barrier. If a split group reads from a split group with the same
    // stripe pattern, no barrier needed.
    let phase_of = assign_phases(n, &producers, &classes, groups);

    // Step 4: Collect groups by phase.
    let num_phases = phase_of.iter().copied().max().unwrap_or(0) + 1;
    let mut phase_groups: Vec<Vec<usize>> = vec![vec![]; num_phases];
    for gi in 0..n {
        phase_groups[phase_of[gi]].push(gi);
    }

    // Step 5: Identify output groups.
    let output_group_set: HashSet<usize> = output_atom_ids
        .iter()
        .filter_map(|&aid| graph.find_group_idx(aid))
        .collect();

    // Step 6: Build phases.
    let mut phases = Vec::with_capacity(num_phases);
    for phase_idx in 0..num_phases {
        let phase = build_phase(
            graph,
            &phase_groups[phase_idx],
            &classes,
            &producers,
            &successors,
            &phase_of,
            num_lanes,
            input_tensors,
            &output_group_set,
            phase_idx,
            num_phases,
        );
        phases.push(phase);
    }

    phases
}

// ─── Group classification ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GroupClass {
    /// Split evenly across lanes. Each lane gets count/num_lanes atoms.
    Split,
    /// Duplicate into every lane (literals, tiny constants).
    Duplicate,
    /// Run whole on a single designated lane (reduce, very small groups).
    Whole,
}

fn classify_groups(
    groups: &[AtomGroup<'static, crate::pool::SystemPool>],
    num_lanes: usize,
) -> Vec<GroupClass> {
    groups.iter().map(|g| classify_one(g, num_lanes)).collect()
}

fn classify_one(g: &AtomGroup<'static, crate::pool::SystemPool>, num_lanes: usize) -> GroupClass {
    // Literals are always duplicated — no computation, every lane needs the value.
    if matches!(g.op, ScalarOp::Literal(_) | ScalarOp::LiteralSpan(_)) {
        return GroupClass::Duplicate;
    }

    // Reduce groups: each output atom independently contracts across a range.
    // The GROUP can be split (each lane gets some output atoms) if there are
    // enough output atoms. But if count is small, it must run whole on one lane.
    if g.op.is_reduce() {
        if g.count >= num_lanes as u64 {
            return GroupClass::Split;
        }
        return GroupClass::Whole;
    }

    // Non-reduce, non-literal groups with inputs: these are computations.
    // Very small groups (fewer atoms than lanes): not worth splitting.
    if g.count < num_lanes as u64 {
        if g.count <= 1 && g.inputs.is_empty() {
            // Truly trivial with no deps: duplicate.
            return GroupClass::Duplicate;
        }
        return GroupClass::Whole;
    }

    // Everything else with sufficient count: split.
    // This covers Binary, Unary, Select, Identity, IndirectLoad.
    GroupClass::Split
}

// ─── Dependency DAG ───────────────────────────────────────────────────────

fn build_dependency_dag(
    graph: &NanoGraph<'static, crate::pool::SystemPool>,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let groups = graph.groups();
    let n = groups.len();
    let mut producers: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut successors: Vec<Vec<usize>> = vec![vec![]; n];

    for (gi, group) in groups.iter().enumerate() {
        let mut seen = HashSet::new();
        graph.collect_all_producer_indices(group, gi, &mut seen);
        let deps: Vec<usize> = seen.into_iter().collect();
        for &pi in &deps {
            successors[pi].push(gi);
        }
        producers.push(deps);
    }

    (producers, successors)
}

// ─── Phase assignment ─────────────────────────────────────────────────────

/// Assign phases to groups.
///
/// Rules:
/// - A Split group reading only from Split producers (or Duplicate/input tensors)
///   can be in the same phase — the lane stripes align, no barrier needed.
/// - A Split group reading from a Whole producer needs a barrier (phase bump).
/// - A Whole group reading from Split producers in the current phase needs a
///   barrier because it must see all lanes' output from the split.
/// - A Whole or Duplicate group reading only from same-phase Whole/Duplicate
///   groups on the same effective lane can stay in the same phase, BUT since
///   spans are per-lane and Whole groups go to one lane, cross-lane deps need
///   a barrier. For simplicity: Whole groups that depend on Split groups always
///   get a new phase.
fn assign_phases(
    n: usize,
    producers: &[Vec<usize>],
    classes: &[GroupClass],
    groups: &[AtomGroup<'static, crate::pool::SystemPool>],
) -> Vec<usize> {
    let mut phase_of = vec![0usize; n];

    // Process in topological order (groups are in topo order by construction).
    for gi in 0..n {
        let my_class = classes[gi];
        let mut earliest = 0usize;

        for &pi in &producers[gi] {
            let prod_class = classes[pi];
            let prod_phase = phase_of[pi];

            match (my_class, prod_class) {
                // Split reading Split: same stripe pattern, no barrier needed.
                (GroupClass::Split, GroupClass::Split) => {
                    earliest = earliest.max(prod_phase);
                }
                // Split reading Duplicate: duplicate is in every lane, no barrier.
                (GroupClass::Split, GroupClass::Duplicate) => {
                    earliest = earliest.max(prod_phase);
                }
                // Split reading Whole: the whole group's output is on one lane,
                // but all lanes need it. Need barrier so it's in the value store.
                (GroupClass::Split, GroupClass::Whole) => {
                    earliest = earliest.max(prod_phase + 1);
                }
                // Duplicate reading anything: duplicates are re-emitted, no dep.
                (GroupClass::Duplicate, _) => {
                    // Literals/tiny duplicates have no inputs typically.
                    // If they somehow do, treat conservatively.
                    earliest = earliest.max(prod_phase);
                }
                // Whole reading Split: needs all lanes' data, barrier required.
                (GroupClass::Whole, GroupClass::Split) => {
                    earliest = earliest.max(prod_phase + 1);
                }
                // Whole reading Duplicate: no barrier, duplicate is everywhere.
                (GroupClass::Whole, GroupClass::Duplicate) => {
                    earliest = earliest.max(prod_phase);
                }
                // Whole reading Whole: if same lane, no barrier. But we don't
                // know lanes yet. Conservative: need barrier if different groups.
                // Actually, we CAN co-locate them on the same lane within a phase,
                // so no barrier needed.
                (GroupClass::Whole, GroupClass::Whole) => {
                    earliest = earliest.max(prod_phase);
                }
            }
        }

        phase_of[gi] = earliest;
    }

    phase_of
}

// ─── Phase construction ───────────────────────────────────────────────────

fn build_phase(
    graph: &NanoGraph<'static, crate::pool::SystemPool>,
    phase_gis: &[usize],
    classes: &[GroupClass],
    producers: &[Vec<usize>],
    successors: &[Vec<usize>],
    phase_of: &[usize],
    num_lanes: usize,
    input_tensors: &[InputTensor],
    output_group_set: &HashSet<usize>,
    phase_idx: usize,
    num_phases: usize,
) -> Phase {
    let groups = graph.groups();

    // Sort by topo order (= group index).
    let mut sorted = phase_gis.to_vec();
    sorted.sort_unstable();

    // Build per-lane work items. For split groups, each lane gets a slice.
    // For duplicates, every lane gets a copy. For whole, assign to one lane.
    let mut lane_items: Vec<Vec<WorkItem>> = vec![vec![]; num_lanes];

    // Track which group indices are computed in this phase (for dep resolution).
    let phase_set: HashSet<usize> = sorted.iter().copied().collect();

    // Bin-pack Whole groups: track per-lane load for load balancing.
    let mut whole_lane_load: Vec<u64> = vec![0; num_lanes];

    for &gi in &sorted {
        let g = &groups[gi];
        match classes[gi] {
            GroupClass::Split => {
                let chunk = g.count / num_lanes as u64;
                if chunk == 0 {
                    // Fewer atoms than lanes — just give it all to lane 0.
                    lane_items[0].push(WorkItem {
                        group_idx: gi,
                        atom_offset: 0,
                        atom_count: g.count,
                        is_duplicate: false,
                    });
                } else {
                    for lane in 0..num_lanes {
                        let start = lane as u64 * chunk;
                        let count = if lane == num_lanes - 1 {
                            g.count - start
                        } else {
                            chunk
                        };
                        lane_items[lane].push(WorkItem {
                            group_idx: gi,
                            atom_offset: start,
                            atom_count: count,
                            is_duplicate: false,
                        });
                    }
                }
            }
            GroupClass::Duplicate => {
                // Every lane gets a full copy.
                for lane in 0..num_lanes {
                    lane_items[lane].push(WorkItem {
                        group_idx: gi,
                        atom_offset: 0,
                        atom_count: g.count,
                        is_duplicate: true,
                    });
                }
            }
            GroupClass::Whole => {
                // Assign to the least-loaded lane. Prefer a lane that already
                // has a producer of this group (for locality).
                let mut best_lane = 0;
                let mut best_load = u64::MAX;

                // Check which lanes have producers of this group.
                let mut producer_lanes: HashSet<usize> = HashSet::new();
                for &pi in &producers[gi] {
                    if phase_set.contains(&pi) {
                        match classes[pi] {
                            GroupClass::Whole => {
                                // Find which lane has this whole group.
                                for (l, items) in lane_items.iter().enumerate() {
                                    if items.iter().any(|w| w.group_idx == pi && !w.is_duplicate) {
                                        producer_lanes.insert(l);
                                    }
                                }
                            }
                            _ => {} // Split/Duplicate: on all lanes.
                        }
                    }
                }

                for lane in 0..num_lanes {
                    let load = whole_lane_load[lane];
                    // Prefer lanes with producers (no cross-lane dep within phase).
                    let bonus = if producer_lanes.contains(&lane) {
                        0u64
                    } else {
                        // Penalize lanes without producers to prefer co-location.
                        g.count
                    };
                    let effective = load + bonus;
                    if effective < best_load {
                        best_load = effective;
                        best_lane = lane;
                    }
                }

                whole_lane_load[best_lane] += g.count;
                lane_items[best_lane].push(WorkItem {
                    group_idx: gi,
                    atom_offset: 0,
                    atom_count: g.count,
                    is_duplicate: false,
                });
            }
        }
    }

    // Sort each lane's items by group index for topological order.
    for lane in &mut lane_items {
        lane.sort_by_key(|w| (w.group_idx, w.atom_offset));
    }

    // Build spans.
    let spans: Vec<Span> = (0..num_lanes)
        .map(|lane_idx| {
            build_span(
                graph,
                &lane_items[lane_idx],
                classes,
                &phase_set,
                &phase_of,
                producers,
                successors,
                num_lanes,
                input_tensors,
                output_group_set,
                phase_idx,
                num_phases,
            )
        })
        .collect();

    Phase { spans }
}

// ─── Work item ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct WorkItem {
    group_idx: usize,
    atom_offset: u64,
    atom_count: u64,
    /// True if this is a duplicated group (literal). The span graph still uses
    /// the original atom IDs for the full group (not offset).
    is_duplicate: bool,
}

// ─── Span construction ────────────────────────────────────────────────────

fn build_span(
    graph: &NanoGraph<'static, crate::pool::SystemPool>,
    lane_work: &[WorkItem],
    classes: &[GroupClass],
    phase_set: &HashSet<usize>,
    phase_of: &[usize],
    producers: &[Vec<usize>],
    successors: &[Vec<usize>],
    num_lanes: usize,
    input_tensors: &[InputTensor],
    output_group_set: &HashSet<usize>,
    phase_idx: usize,
    num_phases: usize,
) -> Span {
    if lane_work.is_empty() {
        return empty_span();
    }

    let groups = graph.groups();
    let mut span_graph = NanoGraph::new();
    span_graph.sym_dim_names = graph.sym_dim_names.clone();
    span_graph.sym_dim_bounds = graph.sym_dim_bounds.clone();

    // Track which atom ranges are computed in this span.
    // Map: group_idx -> Vec<(atom_offset, atom_count)>
    let mut span_compute: HashMap<usize, Vec<(u64, u64)>> = HashMap::new();
    for w in lane_work {
        span_compute
            .entry(w.group_idx)
            .or_default()
            .push((w.atom_offset, w.atom_count));
    }
    let span_compute_set: HashSet<usize> = span_compute.keys().copied().collect();

    // Find all external dependencies needed by this span's groups.
    // External = atoms from earlier phases, or input tensors.
    // Internal = atoms from groups computed in this span.
    let mut needed_external: BTreeMap<u64, (u64, NumericDType)> = BTreeMap::new();
    let mut needed_literals: BTreeSet<usize> = BTreeSet::new();

    const LITERAL_INLINE_THRESHOLD: u64 = 131072;

    for w in lane_work {
        let gi = w.group_idx;
        let g = &groups[gi];
        let eff_offset = if w.is_duplicate {
            g.atom_offset
        } else {
            w.atom_offset
        };
        let eff_count = if w.is_duplicate {
            g.count
        } else {
            w.atom_count
        };

        // For each input of this group, find what atom ranges we need.
        for input in &g.inputs {
            collect_deps_for_input(
                graph,
                input,
                eff_count,
                eff_offset,
                &span_compute,
                input_tensors,
                &mut needed_external,
                &mut needed_literals,
            );
        }

        // Reduce stride extended range.
        if let ScalarOp::Reduce {
            reduce_count,
            reduce_stride,
            ..
        } = &g.op
        {
            if *reduce_count > 1 && *reduce_stride != 0 {
                for input in &g.inputs {
                    let first = input.resolve(eff_offset);
                    let last = input.resolve(eff_offset + eff_count - 1);
                    let end_off = (*reduce_count as i64 - 1) * reduce_stride;
                    let endpoints = [
                        first.0,
                        (first.0 as i64 + end_off) as u64,
                        last.0,
                        (last.0 as i64 + end_off) as u64,
                    ];
                    let lo = *endpoints.iter().min().unwrap();
                    let hi = *endpoints.iter().max().unwrap();

                    collect_external_in_range(
                        graph,
                        lo,
                        hi,
                        g.output_dtype,
                        input_tensors,
                        &span_compute,
                        &needed_literals,
                        &mut needed_external,
                    );
                }
            }
        }

        // IndirectLoad table reference.
        if let ScalarOp::IndirectLoad { table_base, .. } = &g.op {
            if let Some(pi) = graph.find_group_idx(*table_base) {
                let pg = &groups[pi];
                if let Some(fragments) = span_compute.get(&pi) {
                    // Check if the entire table group is covered by local fragments.
                    let gaps = uncovered_ranges(
                        pg.base_id.0,
                        pg.base_id.0 + pg.count - 1,
                        pg.base_id.0,
                        fragments,
                    );
                    for (gap_base, gap_count) in gaps {
                        record_external(&mut needed_external, gap_base, gap_count, pg.output_dtype);
                    }
                } else if is_literal_group(pg) && pg.count < LITERAL_INLINE_THRESHOLD {
                    needed_literals.insert(pi);
                } else {
                    record_external(
                        &mut needed_external,
                        pg.base_id.0,
                        pg.count,
                        pg.output_dtype,
                    );
                }
            } else if let Some((ti, _)) = graph.find_input_idx(*table_base) {
                let it = &input_tensors[ti];
                record_external(&mut needed_external, it.base_id.0, it.count, it.dtype);
            }
        }
    }

    // Also check deps of needed_literals (they're usually zero, but be safe).
    for &li in &needed_literals.clone() {
        let g = &groups[li];
        for input in &g.inputs {
            collect_input_tensor_deps(
                input,
                g.count,
                g.atom_offset,
                graph,
                input_tensors,
                &mut needed_external,
            );
        }
    }

    // Merge overlapping external ranges.
    let merged_ext = merge_external_ranges(&needed_external);

    // Build items to insert into span graph, sorted by base_id.
    #[derive(Debug)]
    enum InsertItem {
        External {
            base: u64,
            count: u64,
            dtype: NumericDType,
        },
        Literal {
            gi: usize,
        },
        Compute(WorkItem),
    }

    let mut items: Vec<(u64, InsertItem)> = Vec::new();

    for &(base, count, dtype) in &merged_ext {
        items.push((base, InsertItem::External { base, count, dtype }));
    }

    for &li in &needed_literals {
        // Don't add if it's already a compute group in this span.
        if !span_compute_set.contains(&li) {
            items.push((groups[li].base_id.0, InsertItem::Literal { gi: li }));
        }
    }

    for w in lane_work {
        let g = &groups[w.group_idx];
        let base = if w.is_duplicate {
            g.base_id.0
        } else {
            g.base_id.0 + w.atom_offset
        };
        items.push((base, InsertItem::Compute(w.clone())));
    }

    items.sort_by_key(|(base, _)| *base);

    // Insert into span graph, tracking inserted ranges to avoid overlaps.
    let mut inserted: Vec<(u64, u64)> = Vec::new();
    let mut span_inputs: Vec<AtomRange> = Vec::new();
    let mut span_outputs: Vec<AtomRange> = Vec::new();

    for (_, item) in &items {
        match item {
            InsertItem::External { base, count, dtype } => {
                if would_overlap(&inserted, *base, *count) {
                    continue;
                }
                span_graph.insert_input_tensor_at(AtomId(*base), GlobalId(0), *count, *dtype);
                span_inputs.push(AtomRange {
                    base: AtomId(*base),
                    count: *count,
                    dtype: *dtype,
                });
                inserted.push((*base, *count));
            }
            InsertItem::Literal { gi } => {
                let g = &groups[*gi];
                if would_overlap(&inserted, g.base_id.0, g.count) {
                    continue;
                }
                span_graph.insert_group_at(
                    g.base_id,
                    g.count,
                    g.atom_offset,
                    g.output_dtype,
                    g.op.clone(),
                    g.sym_dims.clone(),
                    g.inputs.clone(),
                );
                inserted.push((g.base_id.0, g.count));
            }
            InsertItem::Compute(w) => {
                let g = &groups[w.group_idx];

                if w.is_duplicate {
                    // Duplicate: insert full group at original position.
                    if would_overlap(&inserted, g.base_id.0, g.count) {
                        continue;
                    }
                    span_graph.insert_group_at(
                        g.base_id,
                        g.count,
                        g.atom_offset,
                        g.output_dtype,
                        g.op.clone(),
                        g.sym_dims.clone(),
                        g.inputs.clone(),
                    );
                    inserted.push((g.base_id.0, g.count));

                    // Output if consumed by later phases.
                    if group_needs_output(
                        w.group_idx,
                        output_group_set,
                        successors,
                        phase_of,
                        phase_idx,
                        &span_compute_set,
                    ) {
                        span_outputs.push(AtomRange {
                            base: g.base_id,
                            count: g.count,
                            dtype: g.output_dtype,
                        });
                    }
                } else {
                    // Split or whole: insert fragment.
                    let frag_base = g.base_id.0 + w.atom_offset;
                    if would_overlap(&inserted, frag_base, w.atom_count) {
                        continue;
                    }
                    span_graph.insert_group_at(
                        AtomId(frag_base),
                        w.atom_count,
                        w.atom_offset,
                        g.output_dtype,
                        g.op.clone(),
                        g.sym_dims.clone(),
                        g.inputs.clone(),
                    );
                    inserted.push((frag_base, w.atom_count));

                    // Output if consumed by later phases or is a model output.
                    if group_needs_output(
                        w.group_idx,
                        output_group_set,
                        successors,
                        phase_of,
                        phase_idx,
                        &span_compute_set,
                    ) {
                        span_outputs.push(AtomRange {
                            base: AtomId(frag_base),
                            count: w.atom_count,
                            dtype: g.output_dtype,
                        });
                    }
                }
            }
        }
    }

    span_outputs = merge_atom_ranges(span_outputs);

    Span {
        graph: span_graph,
        inputs: span_inputs,
        outputs: span_outputs,
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────

fn empty_span() -> Span {
    Span {
        graph: NanoGraph::new(),
        inputs: vec![],
        outputs: vec![],
    }
}

fn is_literal_group(g: &AtomGroup<'static, crate::pool::SystemPool>) -> bool {
    matches!(g.op, ScalarOp::Literal(_) | ScalarOp::LiteralSpan(_)) && g.inputs.is_empty()
}

/// Check if a group needs its output exported from this span.
fn group_needs_output(
    gi: usize,
    output_group_set: &HashSet<usize>,
    successors: &[Vec<usize>],
    phase_of: &[usize],
    phase_idx: usize,
    span_compute_set: &HashSet<usize>,
) -> bool {
    // Model output.
    if output_group_set.contains(&gi) {
        return true;
    }
    // Has a successor in a later phase, or a successor in this phase that's
    // NOT in our span (cross-lane).
    for &si in &successors[gi] {
        if phase_of[si] > phase_idx {
            return true;
        }
        if phase_of[si] == phase_idx && !span_compute_set.contains(&si) {
            return true;
        }
    }
    false
}

/// Compute the source range [lo, hi] for an InputRef at the given offset/count.
fn input_ref_range(input: &InputRef, count: u64, atom_offset: u64) -> (u64, u64) {
    if count == 0 {
        return (0, 0);
    }
    match input {
        InputRef::Broadcast(base) => (base.0, base.0),
        InputRef::Strided { .. } => {
            let first = input.resolve(atom_offset);
            let last = input.resolve(atom_offset + count - 1);
            (first.0.min(last.0), first.0.max(last.0))
        }
        InputRef::Explicit(ids) => {
            let slice = &ids[atom_offset as usize..(atom_offset + count) as usize];
            let lo = slice.iter().map(|id| id.0).min().unwrap_or(0);
            let hi = slice.iter().map(|id| id.0).max().unwrap_or(0);
            (lo, hi)
        }
    }
}

/// Collect dependencies for a single InputRef into external/literal sets.
fn collect_deps_for_input(
    graph: &NanoGraph<'static, crate::pool::SystemPool>,
    input: &InputRef,
    count: u64,
    atom_offset: u64,
    span_compute: &HashMap<usize, Vec<(u64, u64)>>,
    input_tensors: &[InputTensor],
    needed_external: &mut BTreeMap<u64, (u64, NumericDType)>,
    needed_literals: &mut BTreeSet<usize>,
) {
    let (lo, hi) = input_ref_range(input, count, atom_offset);
    let groups = graph.groups();

    // Walk groups overlapping [lo, hi].
    for (gi, g) in groups.iter().enumerate() {
        let g_lo = g.base_id.0;
        let g_hi = g_lo + g.count;
        if g_lo > hi {
            break;
        }
        if g_hi <= lo {
            continue;
        }
        // This group overlaps [lo, hi].

        // If this group is fully inlined as a literal, skip.
        if needed_literals.contains(&gi) {
            continue;
        }

        // Compute the needed range within this group's atom space.
        let need_lo = g_lo.max(lo);
        let need_hi = (g_hi - 1).min(hi); // inclusive

        if let Some(fragments) = span_compute.get(&gi) {
            // Group is (partially) computed in this span. Check coverage.
            let gaps = uncovered_ranges(need_lo, need_hi, g.base_id.0, fragments);
            for (gap_base, gap_count) in gaps {
                record_external(needed_external, gap_base, gap_count, g.output_dtype);
            }
            continue;
        }

        // Literal? Inline it.
        if is_literal_group(g) && g.count < 131072 {
            needed_literals.insert(gi);
            continue;
        }
        // External computed group.
        let range_hi = g_hi.min(hi + 1);
        if range_hi > need_lo {
            record_external(needed_external, need_lo, range_hi - need_lo, g.output_dtype);
        }
    }

    // Input tensors overlapping [lo, hi].
    for it in input_tensors {
        let it_lo = it.base_id.0;
        let it_hi = it_lo + it.count;
        if it_lo <= hi && it_hi > lo {
            let range_lo = it_lo.max(lo);
            let range_hi = it_hi.min(hi + 1);
            if range_hi > range_lo {
                record_external(needed_external, range_lo, range_hi - range_lo, it.dtype);
            }
        }
    }
}

/// Record external ranges for reduce ops that read extended stride ranges.
fn collect_external_in_range(
    graph: &NanoGraph<'static, crate::pool::SystemPool>,
    lo: u64,
    hi: u64,
    fallback_dtype: NumericDType,
    input_tensors: &[InputTensor],
    span_compute: &HashMap<usize, Vec<(u64, u64)>>,
    inlined_literals: &BTreeSet<usize>,
    needed_external: &mut BTreeMap<u64, (u64, NumericDType)>,
) {
    let groups = graph.groups();
    for (gi, g) in groups.iter().enumerate() {
        let g_lo = g.base_id.0;
        let g_hi = g_lo + g.count;
        if g_lo > hi {
            break;
        }
        if g_hi <= lo {
            continue;
        }
        if inlined_literals.contains(&gi) {
            continue;
        }

        let need_lo = g_lo.max(lo);
        let need_hi = (g_hi - 1).min(hi); // inclusive

        if let Some(fragments) = span_compute.get(&gi) {
            // Group is (partially) computed in this span. Check coverage.
            let gaps = uncovered_ranges(need_lo, need_hi, g.base_id.0, fragments);
            for (gap_base, gap_count) in gaps {
                record_external(needed_external, gap_base, gap_count, g.output_dtype);
            }
            continue;
        }

        let range_hi = g_hi.min(hi + 1);
        if range_hi > need_lo {
            record_external(needed_external, need_lo, range_hi - need_lo, g.output_dtype);
        }
    }

    for it in input_tensors {
        let it_lo = it.base_id.0;
        let it_hi = it_lo + it.count;
        if it_lo <= hi && it_hi > lo {
            let range_lo = it_lo.max(lo);
            let range_hi = it_hi.min(hi + 1);
            if range_hi > range_lo {
                record_external(needed_external, range_lo, range_hi - range_lo, it.dtype);
            }
        }
    }
}

fn collect_input_tensor_deps(
    input: &InputRef,
    count: u64,
    atom_offset: u64,
    graph: &NanoGraph<'static, crate::pool::SystemPool>,
    input_tensors: &[InputTensor],
    needed_external: &mut BTreeMap<u64, (u64, NumericDType)>,
) {
    let (lo, hi) = input_ref_range(input, count, atom_offset);
    for it in input_tensors {
        let it_lo = it.base_id.0;
        let it_hi = it_lo + it.count;
        if it_lo <= hi && it_hi > lo {
            record_external(needed_external, it.base_id.0, it.count, it.dtype);
        }
    }
}

fn record_external(
    ranges: &mut BTreeMap<u64, (u64, NumericDType)>,
    base: u64,
    count: u64,
    dtype: NumericDType,
) {
    ranges
        .entry(base)
        .and_modify(|(existing_count, _)| {
            *existing_count = (*existing_count).max(count);
        })
        .or_insert((count, dtype));
}

fn merge_external_ranges(
    ranges: &BTreeMap<u64, (u64, NumericDType)>,
) -> Vec<(u64, u64, NumericDType)> {
    if ranges.is_empty() {
        return vec![];
    }

    let sorted: Vec<(u64, u64, NumericDType)> = ranges
        .iter()
        .map(|(&base, &(count, dtype))| (base, count, dtype))
        .collect();

    let mut merged: Vec<(u64, u64, NumericDType)> = Vec::new();

    for (base, count, dtype) in sorted {
        if let Some(last) = merged.last_mut() {
            let last_end = last.0 + last.1;
            if base <= last_end && dtype == last.2 {
                let new_end = (base + count).max(last_end);
                last.1 = new_end - last.0;
                continue;
            }
        }
        merged.push((base, count, dtype));
    }

    merged
}

fn would_overlap(inserted: &[(u64, u64)], start: u64, count: u64) -> bool {
    let end = start + count;
    for &(existing_start, existing_count) in inserted {
        let existing_end = existing_start + existing_count;
        if start < existing_end && end > existing_start {
            return true;
        }
    }
    false
}

/// Given the atom range [need_lo, need_hi] (inclusive) within a group whose
/// base_id is `group_base`, and a set of local fragments `(atom_offset, atom_count)`,
/// return the sub-ranges of [need_lo, need_hi] NOT covered by any fragment.
fn uncovered_ranges(
    need_lo: u64,
    need_hi: u64, // inclusive
    group_base: u64,
    fragments: &[(u64, u64)],
) -> Vec<(u64, u64)> {
    // Build sorted list of covered absolute ranges.
    let mut covered: Vec<(u64, u64)> = fragments
        .iter()
        .map(|&(off, cnt)| {
            let abs_lo = group_base + off;
            let abs_hi = abs_lo + cnt; // exclusive
            (abs_lo, abs_hi)
        })
        .collect();
    covered.sort_by_key(|&(lo, _)| lo);

    let mut gaps = Vec::new();
    let mut cursor = need_lo;
    let end = need_hi + 1; // exclusive

    for &(cov_lo, cov_hi) in &covered {
        if cursor >= end {
            break;
        }
        if cov_hi <= cursor {
            continue;
        }
        if cov_lo > cursor {
            // Gap: [cursor, min(cov_lo, end))
            let gap_end = cov_lo.min(end);
            gaps.push((cursor, gap_end - cursor));
        }
        cursor = cursor.max(cov_hi);
    }
    if cursor < end {
        gaps.push((cursor, end - cursor));
    }
    gaps
}

fn merge_atom_ranges(mut ranges: Vec<AtomRange>) -> Vec<AtomRange> {
    if ranges.len() <= 1 {
        return ranges;
    }
    ranges.sort_by_key(|r| r.base.0);
    let mut merged = vec![ranges[0].clone()];
    for r in &ranges[1..] {
        let last = merged.last_mut().unwrap();
        let last_end = last.base.0 + last.count;
        if r.base.0 <= last_end && r.dtype == last.dtype {
            let new_end = (r.base.0 + r.count).max(last_end);
            last.count = new_end - last.base.0;
        } else {
            merged.push(r.clone());
        }
    }
    merged
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nano_graph::ops::{ReduceKind, ScalarBinOp, ScalarUnaryOp};
    use crate::numeric_dtype::NumericDType;
    use crate::numeric_scalar::NumericScalar;

    const NUM_LANES: usize = 4;

    /// Helper: count total atoms across all spans in a phase.
    fn phase_atoms(phase: &Phase) -> Vec<u64> {
        phase
            .spans
            .iter()
            .map(|s| {
                s.graph
                    .groups()
                    .iter()
                    .filter(|g| !matches!(g.op, ScalarOp::Literal(_) | ScalarOp::LiteralSpan(_)))
                    .map(|g| g.count)
                    .sum()
            })
            .collect()
    }

    /// Helper: check that a group was actually split across multiple lanes
    /// in some phase. Returns true if the group's base_id appears as a fragment
    /// in more than one span.
    fn group_is_split_across_lanes(phases: &[Phase], base_id: AtomId, total_count: u64) -> bool {
        for phase in phases {
            let mut lanes_with_fragment = 0;
            for span in &phase.spans {
                for g in span.graph.groups() {
                    // Check if this group's atom range overlaps with the original group.
                    let g_lo = g.base_id.0;
                    let g_hi = g_lo + g.count;
                    let orig_lo = base_id.0;
                    let orig_hi = orig_lo + total_count;
                    if g_lo >= orig_lo
                        && g_hi <= orig_hi
                        && !matches!(g.op, ScalarOp::Literal(_) | ScalarOp::LiteralSpan(_))
                    {
                        lanes_with_fragment += 1;
                    }
                }
            }
            if lanes_with_fragment > 1 {
                return true;
            }
        }
        false
    }

    /// Helper: validate that all span NanoGraphs pass structural validation.
    fn validate_all_spans(phases: &[Phase]) {
        for (pi, phase) in phases.iter().enumerate() {
            for (si, span) in phase.spans.iter().enumerate() {
                let errors = span.graph.validate();
                assert!(
                    errors.is_empty(),
                    "Phase {} span {} validation failed: {:?}",
                    pi,
                    si,
                    errors
                );
            }
        }
    }

    /// Helper: verify all output atoms are produced by some span.
    fn outputs_are_covered(phases: &[Phase], output_ids: &[AtomId]) {
        let mut produced: HashSet<u64> = HashSet::new();
        for phase in phases {
            for span in &phase.spans {
                for out in &span.outputs {
                    for i in 0..out.count {
                        produced.insert(out.base.0 + i);
                    }
                }
            }
        }
        for &aid in output_ids {
            assert!(
                produced.contains(&aid.0),
                "Output atom {} not produced by any span",
                aid
            );
        }
    }

    // ── Test: Linear chain splitting ──

    #[test]
    fn test_linear_chain_split() {
        // Chain: Literal(1000) -> Neg(1000) -> Exp(1000)
        // Expected: 1 phase, Neg and Exp each split across NUM_LANES lanes.
        let mut g = NanoGraph::new();

        let lit = g.push_group(
            1000,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(1.0)),
            vec![],
            vec![],
        );
        let neg = g.push_group(
            1000,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(lit, 1)],
        );
        let exp = g.push_group(
            1000,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(neg, 1)],
        );
        g.outputs.push(exp);

        let phases = plan(&g, NUM_LANES, &[], &[exp]);

        // Should be exactly 1 phase — the whole chain is split-compatible.
        assert_eq!(phases.len(), 1, "Linear chain should be 1 phase");

        // Neg should be split across lanes.
        assert!(
            group_is_split_across_lanes(&phases, neg, 1000),
            "Neg group should be split across lanes"
        );

        // Exp should be split across lanes.
        assert!(
            group_is_split_across_lanes(&phases, exp, 1000),
            "Exp group should be split across lanes"
        );

        validate_all_spans(&phases);
        outputs_are_covered(&phases, &[exp]);

        // Check balance: each lane should have roughly 250 atoms of Neg + 250 of Exp.
        let atoms = phase_atoms(&phases[0]);
        for (i, &a) in atoms.iter().enumerate() {
            assert!(
                a >= 400 && a <= 600,
                "Lane {} has {} atoms, expected ~500",
                i,
                a
            );
        }
    }

    // ── Test: Diamond graph ──

    #[test]
    fn test_diamond_split() {
        //    lit(1000)
        //    /       \
        // neg(1000) exp(1000)
        //    \       /
        //   add(1000)
        let mut g = NanoGraph::new();

        let lit = g.push_group(
            1000,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(2.0)),
            vec![],
            vec![],
        );
        let neg = g.push_group(
            1000,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(lit, 1)],
        );
        let exp = g.push_group(
            1000,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(lit, 1)],
        );
        let add = g.push_group(
            1000,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(neg, 1), InputRef::affine(exp, 1)],
        );
        g.outputs.push(add);

        let phases = plan(&g, NUM_LANES, &[], &[add]);

        // All split, same stripe → 1 phase.
        assert_eq!(phases.len(), 1, "Diamond should be 1 phase");

        // All compute groups split across lanes.
        assert!(group_is_split_across_lanes(&phases, neg, 1000));
        assert!(group_is_split_across_lanes(&phases, exp, 1000));
        assert!(group_is_split_across_lanes(&phases, add, 1000));

        validate_all_spans(&phases);
        outputs_are_covered(&phases, &[add]);
    }

    // ── Test: MatMul-like structure with split ──

    #[test]
    fn test_matmul_structure() {
        // Simulate matmul: M=16, K=8
        // - A literal (weights): 128 atoms (M*K, Literal)
        // - B input: 8 atoms (K vector)
        // - Mul group: 128 atoms, StridedBroadcast from B (each row of 8 broadcasts one B element)
        //   Actually for matmul: Mul has count=M*K=128, with:
        //     input 0: Affine from A (stride 1) — the weight matrix
        //     input 1: StridedBroadcast from B (base=B, stride=1, repeat=K... wait, no)
        //   Let's think more carefully. MatMul A(M,K) x B(K):
        //   For each output[m], we compute sum_{k=0}^{K-1} A[m,k] * B[k].
        //   Lowered as:
        //     Mul group: count=M*K, atom i computes A[i] * B[i % K]
        //       input 0: Affine{base=A, stride=1}
        //       input 1: Modular{base=B, stride=1, modulus=K}
        //     ReduceSum group: count=M, atom m reduces Mul[m*K..m*K+K-1]
        //       input 0: Affine{base=Mul, stride=K}
        //       reduce_count=K, reduce_stride=1
        let mut g = NanoGraph::new();

        let m = 16u64;
        let k = 8u64;

        // Weight matrix A: M*K literals.
        let a_base = g.push_group(
            m * k,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(0.5)),
            vec![],
            vec![],
        );

        // Input vector B: K elements (as input tensor).
        let b_base = g.add_input_tensor(GlobalId(42), k, NumericDType::F32);

        // Mul group: M*K atoms.
        let mul_base = g.push_group(
            m * k,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(a_base, 1), InputRef::modular(b_base, 1, k)],
        );

        // ReduceSum group: M atoms, each reduces K products.
        let red_base = g.push_group(
            m,
            NumericDType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: k,
                reduce_stride: 1,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(mul_base, k as i64)],
        );
        g.outputs.push(red_base);

        let input_tensors = vec![InputTensor {
            tensor_id: GlobalId(42),
            base_id: b_base,
            count: k,
            dtype: NumericDType::F32,
        }];

        let phases = plan(&g, NUM_LANES, &input_tensors, &[red_base]);

        // Mul should be split across lanes (128 atoms / 4 lanes = 32 each).
        assert!(
            group_is_split_across_lanes(&phases, mul_base, m * k),
            "Mul group should be split"
        );

        // ReduceSum: M=16 atoms, 4 lanes → 4 atoms each. Should be split.
        assert!(
            group_is_split_across_lanes(&phases, red_base, m),
            "ReduceSum group should be split across lanes"
        );

        validate_all_spans(&phases);
        outputs_are_covered(&phases, &[red_base]);
    }

    // ── Test: Literal duplication ──

    #[test]
    fn test_literal_duplication() {
        // A literal feeding into a split compute group.
        // The literal should appear in every lane's span.
        let mut g = NanoGraph::new();

        let lit = g.push_group(
            100,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(3.14)),
            vec![],
            vec![],
        );
        let neg = g.push_group(
            1000,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::Broadcast(lit)],
        );
        g.outputs.push(neg);

        let phases = plan(&g, NUM_LANES, &[], &[neg]);
        assert_eq!(phases.len(), 1);

        // Check that the literal appears in every lane's span.
        for (lane_idx, span) in phases[0].spans.iter().enumerate() {
            let has_literal = span.graph.groups().iter().any(|g| {
                g.base_id == lit && matches!(g.op, ScalarOp::Literal(_) | ScalarOp::LiteralSpan(_))
            });
            assert!(
                has_literal,
                "Lane {} should have a copy of the literal",
                lane_idx
            );
        }

        // Neg should be split.
        assert!(group_is_split_across_lanes(&phases, neg, 1000));

        validate_all_spans(&phases);
        outputs_are_covered(&phases, &[neg]);
    }

    // ── Test: Reduce requiring barrier ──

    #[test]
    fn test_reduce_barrier() {
        // Pattern: Literal(1000) -> Neg(1000) -> ReduceSum(1)
        // Neg is split across lanes. ReduceSum reads ALL of Neg's output,
        // so it needs a barrier (separate phase).
        let mut g = NanoGraph::new();

        let lit = g.push_group(
            1000,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(1.0)),
            vec![],
            vec![],
        );
        let neg = g.push_group(
            1000,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(lit, 1)],
        );

        // Single-atom ReduceSum over all 1000 Neg atoms.
        let red = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: 1000,
                reduce_stride: 1,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(neg, 1)],
        );
        g.outputs.push(red);

        let phases = plan(&g, NUM_LANES, &[], &[red]);

        // ReduceSum(count=1) is Whole. It reads from Split Neg.
        // => Needs a barrier => at least 2 phases.
        assert!(
            phases.len() >= 2,
            "Reduce reading split data should need a barrier, got {} phases",
            phases.len()
        );

        // Neg should be split in phase 0.
        assert!(
            group_is_split_across_lanes(&phases, neg, 1000),
            "Neg should be split across lanes"
        );

        validate_all_spans(&phases);
        outputs_are_covered(&phases, &[red]);
    }

    // ── Test: Large split reduces (matmul-style) ──

    #[test]
    fn test_large_reduce_split() {
        // M ReduceSum groups where M >= num_lanes: each reduces independently.
        // These should be split across lanes.
        let mut g = NanoGraph::new();

        let m = 32u64;
        let k = 16u64;

        let source = g.push_group(
            m * k,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(1.0)),
            vec![],
            vec![],
        );

        // M reduce groups, each reducing K elements.
        let red = g.push_group(
            m,
            NumericDType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: k,
                reduce_stride: 1,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(source, k as i64)],
        );
        g.outputs.push(red);

        let phases = plan(&g, NUM_LANES, &[], &[red]);

        // The reduce group has count=32, 4 lanes → 8 per lane. Should be split.
        assert!(
            group_is_split_across_lanes(&phases, red, m),
            "Large reduce group (count={}) should be split across {} lanes",
            m,
            NUM_LANES
        );

        validate_all_spans(&phases);
        outputs_are_covered(&phases, &[red]);
    }

    // ── Test: Elementwise chain, no barriers ──

    #[test]
    fn test_elementwise_chain_single_phase() {
        // Sub -> Pow -> Div: 3 elementwise ops in a chain. All split.
        // Should produce exactly 1 phase.
        let mut g = NanoGraph::new();

        let lit_a = g.push_group(
            4096,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(1.0)),
            vec![],
            vec![],
        );
        let lit_b = g.push_group(
            4096,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(2.0)),
            vec![],
            vec![],
        );

        let sub = g.push_group(
            4096,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Sub,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(lit_a, 1), InputRef::affine(lit_b, 1)],
        );

        let lit_c = g.push_group(
            4096,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(2.0)),
            vec![],
            vec![],
        );

        let pow = g.push_group(
            4096,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Pow,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(sub, 1), InputRef::affine(lit_c, 1)],
        );

        let lit_d = g.push_group(
            4096,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(0.5)),
            vec![],
            vec![],
        );

        let div = g.push_group(
            4096,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Div,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(pow, 1), InputRef::affine(lit_d, 1)],
        );
        g.outputs.push(div);

        let phases = plan(&g, NUM_LANES, &[], &[div]);

        assert_eq!(
            phases.len(),
            1,
            "Elementwise chain should produce exactly 1 phase, got {}",
            phases.len()
        );

        // All compute groups should be split.
        assert!(group_is_split_across_lanes(&phases, sub, 4096));
        assert!(group_is_split_across_lanes(&phases, pow, 4096));
        assert!(group_is_split_across_lanes(&phases, div, 4096));

        validate_all_spans(&phases);
        outputs_are_covered(&phases, &[div]);
    }

    // ── Test: Mixed chain with small reduce in middle ──

    #[test]
    fn test_chain_with_small_reduce() {
        // Pattern: Neg(1000) -> ReduceMean(1) -> Div(1000)
        // Neg is split. ReduceMean(1 atom) is Whole.
        // Div reads from both Neg (split) and ReduceMean (whole).
        // ReduceMean reading split Neg needs barrier.
        // Div broadcasting ReduceMean reads a Whole producer -> needs barrier.
        // So: Phase 0: Neg (split), Phase 1: ReduceMean (whole), Phase 2: Div (split).
        // Or: Phase 0: Neg (split), Phase 1: ReduceMean (whole) + Div (split) if Div
        //     can read ReduceMean from same phase. But ReduceMean is Whole on one lane,
        //     and Div is Split across all lanes — Div reading Whole in same phase
        //     would require cross-lane read. So Div needs another barrier? No —
        //     actually, the Whole group OUTPUT goes into the value store after its phase,
        //     and Div in the next phase reads it as external input.
        //
        // Actually per phase assignment rules:
        // - ReduceMean(Whole) reading Neg(Split) → phase >= neg.phase + 1
        // - Div(Split) reading ReduceMean(Whole) → phase >= reduce.phase + 1
        // - Div(Split) reading Neg(Split) → phase >= neg.phase (same phase OK)
        // So minimum: Neg=0, Reduce=1, Div=2. Three phases.
        //
        // But wait — Div doesn't necessarily read Neg directly. Let me make it so.
        // Pattern: Lit(1000) -> Neg(1000) -> ReduceSum(1) -> broadcast to Sqrt(1) -> Div(1000)
        // Where Div's inputs are Neg output and Sqrt output.

        let mut g = NanoGraph::new();

        let lit = g.push_group(
            1000,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(1.0)),
            vec![],
            vec![],
        );
        let neg = g.push_group(
            1000,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(lit, 1)],
        );

        // ReduceSum over all 1000 neg outputs → 1 atom.
        let red = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: 1000,
                reduce_stride: 1,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(neg, 1)],
        );

        // Div: each of 1000 atoms divides Neg by the ReduceSum scalar.
        let div = g.push_group(
            1000,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Div,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(neg, 1), InputRef::Broadcast(red)],
        );
        g.outputs.push(div);

        let phases = plan(&g, NUM_LANES, &[], &[div]);

        // Neg should be split.
        assert!(group_is_split_across_lanes(&phases, neg, 1000));

        // Div should be split.
        assert!(group_is_split_across_lanes(&phases, div, 1000));

        // Should have multiple phases due to the reduce barrier.
        assert!(
            phases.len() >= 2,
            "Should have at least 2 phases due to reduce barrier"
        );

        validate_all_spans(&phases);
        outputs_are_covered(&phases, &[div]);
    }

    // ── Test: Empty graph ──

    #[test]
    fn test_empty_graph() {
        let g = NanoGraph::new();
        let phases = plan(&g, NUM_LANES, &[], &[]);
        assert_eq!(phases.len(), 1);
        assert_eq!(phases[0].spans.len(), NUM_LANES);
    }

    // ── Test: Single-atom groups ──

    #[test]
    fn test_single_atom_groups() {
        // Very small groups: should still work.
        let mut g = NanoGraph::new();

        let lit = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(42.0)),
            vec![],
            vec![],
        );
        let neg = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(lit, 1)],
        );
        g.outputs.push(neg);

        let phases = plan(&g, NUM_LANES, &[], &[neg]);
        // Should work without panic.
        assert!(!phases.is_empty());
        validate_all_spans(&phases);
        outputs_are_covered(&phases, &[neg]);
    }

    // ── Test: Input tensor handling ──

    #[test]
    fn test_input_tensor() {
        let mut g = NanoGraph::new();

        let inp = g.add_input_tensor(GlobalId(1), 1000, NumericDType::F32);
        let neg = g.push_group(
            1000,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(inp, 1)],
        );
        g.outputs.push(neg);

        let input_tensors = vec![InputTensor {
            tensor_id: GlobalId(1),
            base_id: inp,
            count: 1000,
            dtype: NumericDType::F32,
        }];

        let phases = plan(&g, NUM_LANES, &input_tensors, &[neg]);

        assert_eq!(phases.len(), 1);
        assert!(group_is_split_across_lanes(&phases, neg, 1000));

        // Each span should declare inputs for the portion of the input tensor it needs.
        for span in &phases[0].spans {
            assert!(
                !span.inputs.is_empty(),
                "Span reading from input tensor should declare inputs"
            );
        }

        validate_all_spans(&phases);
        outputs_are_covered(&phases, &[neg]);
    }

    // ── Test: Verify lane count ──

    #[test]
    fn test_correct_lane_count() {
        let mut g = NanoGraph::new();
        let lit = g.push_group(
            100,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(1.0)),
            vec![],
            vec![],
        );
        let neg = g.push_group(
            100,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(lit, 1)],
        );
        g.outputs.push(neg);

        for num_lanes in [1, 2, 4, 8] {
            let phases = plan(&g, num_lanes, &[], &[neg]);
            for phase in &phases {
                assert_eq!(
                    phase.spans.len(),
                    num_lanes,
                    "Phase should have {} spans",
                    num_lanes
                );
            }
        }
    }

    // ── Test: IndirectLoad ──

    #[test]
    fn test_indirect_load() {
        let mut g = NanoGraph::new();

        // Table: 256 literal entries (e.g. embedding table).
        let table = g.push_group(
            256,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(0.0)),
            vec![],
            vec![],
        );

        // Index source: 100 atoms (e.g., token IDs).
        let indices = g.push_group(
            100,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(0.0)),
            vec![],
            vec![],
        );

        // IndirectLoad: 100 atoms, each looks up table[index].
        let load = g.push_group(
            100,
            NumericDType::F32,
            ScalarOp::IndirectLoad { table_base: table },
            vec![],
            vec![InputRef::affine(indices, 1)],
        );
        g.outputs.push(load);

        let phases = plan(&g, NUM_LANES, &[], &[load]);

        // IndirectLoad should be split across lanes.
        assert!(group_is_split_across_lanes(&phases, load, 100));

        validate_all_spans(&phases);
        outputs_are_covered(&phases, &[load]);
    }
}
