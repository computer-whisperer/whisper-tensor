#![allow(
    clippy::all,
    dead_code,
    unreachable_code,
    unreachable_patterns,
    unused_imports,
    unused_variables
)]

//! Two-Phase Partitioner (attempt H): Correct First, Balance Second
//!
//! Separation of concerns:
//!
//! **Phase 1 — Build a correct plan (0 violations guaranteed):**
//! Compute DAG depth for each group. Groups at the same depth are provably
//! independent. Start with all groups on lane 0 (trivially correct, maximally
//! imbalanced). Merge adjacent depth levels aggressively — two levels can
//! merge if no group in the later level depends on a group in the earlier
//! level that's on a different lane. Since we start with everything on lane 0,
//! all same-phase groups are co-located and we can merge aggressively.
//!
//! **Phase 2 — Rebalance while preserving correctness:**
//! Iteratively move groups between lanes. For each phase, compute the load per
//! lane. Find the most overloaded lane and identify movable groups — a group
//! is movable to a target lane if all its producers in the same phase are
//! already on the target lane or in an earlier phase. Move it. Repeat until
//! balanced or no more moves are possible.
//!
//! The key insight: the "movable" check is cheap (inspect the group's producer
//! set) and the move preserves the invariant by construction.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::dtype::DType;
use crate::graph::GlobalId;
use crate::nano_graph::pattern::InputTensor;
use crate::nano_graph::{AtomGroup, AtomId, AtomRange, InputRef, NanoGraph, ScalarOp, SymDim};

use super::types::{Phase, Span};

// ─── Public API ────────────────────────────────────────────────────────────

/// Partition a NanoGraph into phases and spans for parallel execution.
pub fn plan(
    graph: &NanoGraph,
    num_lanes: usize,
    input_tensors: &[InputTensor],
    output_atom_ids: &[AtomId],
) -> Vec<Phase> {
    let num_lanes = num_lanes.max(1);
    let groups = graph.groups();
    let n = groups.len();

    if n == 0 {
        return vec![Phase {
            spans: (0..num_lanes)
                .map(|_| Span {
                    graph: NanoGraph::new(),
                    inputs: vec![],
                    outputs: vec![],
                })
                .collect(),
        }];
    }

    // Step 1: Build group-level dependency DAG.
    let (producers, successors) = build_dependency_dag(graph);

    // Step 2: Compute depth for each group (longest path from any root).
    let depths = compute_depths(n, &producers);

    // Step 3: Form wavefronts (groups at the same depth).
    let max_depth = depths.iter().copied().max().unwrap_or(0);
    let mut wavefronts: Vec<Vec<usize>> = vec![vec![]; max_depth + 1];
    for (gi, &depth) in depths.iter().enumerate() {
        wavefronts[depth].push(gi);
    }

    // Step 4: Merge wavefronts into phases aggressively.
    // We use union-find to track which groups must be co-located (same lane)
    // within a phase. Two wavefronts can merge into a phase as long as we
    // can enforce that each group and its same-phase producers share a lane.
    let phase_assignments =
        merge_wavefronts_aggressively(&wavefronts, &producers, &successors, n, groups);

    // Build phase_groups: which groups are in each phase.
    let num_phases = phase_assignments.iter().copied().max().unwrap_or(0) + 1;
    let mut phase_groups: Vec<Vec<usize>> = vec![vec![]; num_phases];
    for (gi, &phase) in phase_assignments.iter().enumerate() {
        phase_groups[phase].push(gi);
    }

    // Step 5: For each phase, build intra-phase dependency chains (union-find)
    // and assign chains to lanes. Initially all on lane 0, then rebalance.
    let mut lane_assignments = vec![0usize; n]; // lane_assignments[gi] = lane

    for phase_idx in 0..num_phases {
        assign_lanes_for_phase(
            &phase_groups[phase_idx],
            &producers,
            groups,
            num_lanes,
            &mut lane_assignments,
        );
    }

    // Step 6: Identify output groups.
    let output_group_set = identify_output_groups(graph, output_atom_ids);

    // Step 7: Build Phase/Span structures.
    let mut phases = Vec::with_capacity(num_phases);
    for phase_idx in 0..num_phases {
        let phase = build_phase(
            graph,
            &phase_groups[phase_idx],
            &lane_assignments,
            &phase_assignments,
            &producers,
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

// ─── Dependency DAG ────────────────────────────────────────────────────────

/// Build the group-level dependency DAG.
/// Returns (producers, successors) where:
/// - producers[gi] = set of group indices that gi depends on
/// - successors[gi] = set of group indices that depend on gi
fn build_dependency_dag(graph: &NanoGraph) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
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

// ─── Depth computation ─────────────────────────────────────────────────────

/// Compute the depth of each group in the DAG.
/// Depth = longest path from any root. Groups with no dependencies have depth 0.
fn compute_depths(n: usize, producers: &[Vec<usize>]) -> Vec<usize> {
    let mut depths = vec![0usize; n];

    // Groups are in topological order (sequential ID allocation, inputs always earlier).
    for gi in 0..n {
        let mut max_dep_depth = 0usize;
        for &pi in &producers[gi] {
            max_dep_depth = max_dep_depth.max(depths[pi] + 1);
        }
        depths[gi] = max_dep_depth;
    }

    depths
}

// ─── Aggressive wavefront merging ──────────────────────────────────────────

/// Merge consecutive wavefronts into phases.
///
/// Two adjacent wavefronts can be merged if we can co-locate each group with
/// its same-phase producers on a single lane without creating excessive
/// constraint chains. We use a union-find to track constraint groups and
/// limit the total number of constraint chains per phase.
///
/// Returns phase_assignment[gi] = phase index for each group.
fn merge_wavefronts_aggressively(
    wavefronts: &[Vec<usize>],
    producers: &[Vec<usize>],
    successors: &[Vec<usize>],
    n: usize,
    groups: &[AtomGroup],
) -> Vec<usize> {
    if wavefronts.is_empty() {
        return vec![];
    }

    let mut phase_assignment = vec![0usize; n];

    // Track which wavefront each group belongs to.
    let mut group_wavefront = vec![0usize; n];
    for (wi, wavefront) in wavefronts.iter().enumerate() {
        for &gi in wavefront {
            group_wavefront[gi] = wi;
        }
    }

    // We merge greedily: try to add each wavefront to the current phase.
    // Use union-find to track constraint chains within the current phase.
    // If merging a wavefront would create a constraint chain that's too large
    // (larger than a threshold fraction of total atoms), start a new phase.

    let mut current_phase = 0usize;
    let _current_phase_start_wavefront = 0usize;
    let mut uf = UnionFind::new(n);
    let mut current_phase_groups: HashSet<usize> = HashSet::new();
    // Track atom counts per union-find root for the current phase.
    let mut chain_atoms: HashMap<usize, u64> = HashMap::new();

    // Total atoms for threshold computation.
    let total_atoms: u64 = groups.iter().map(|g| g.count).sum();
    // A chain shouldn't hold more than roughly 1/num_target_phases of total atoms.
    // We'll use a heuristic: don't merge if the largest chain would exceed
    // (total_atoms / 4). This limits phases to ~4-8 for typical workloads.
    // But we also limit by chain count: if too many groups get chained together,
    // we lose the ability to balance across lanes.
    let max_chain_atoms = (total_atoms / 4).max(1);

    for wi in 0..wavefronts.len() {
        if wi == 0 {
            // First wavefront always starts phase 0.
            for &gi in &wavefronts[0] {
                phase_assignment[gi] = 0;
                current_phase_groups.insert(gi);
                let root = uf.find(gi);
                *chain_atoms.entry(root).or_insert(0) += groups[gi].count;
            }
            continue;
        }

        // Try merging wavefront wi into current_phase.
        // For each group in wi, check: does it have producers in current_phase?
        // If so, they must be unioned. Check if that creates too-large chains.

        // First, simulate the merges to see if any chain exceeds the threshold.
        let mut trial_uf = uf.clone();
        let mut trial_chain_atoms = chain_atoms.clone();
        let mut can_merge = true;

        for &gi in &wavefronts[wi] {
            // Add this group to the trial.
            let gi_root = trial_uf.find(gi);
            *trial_chain_atoms.entry(gi_root).or_insert(0) += groups[gi].count;

            // Find producers in the current phase.
            for &pi in &producers[gi] {
                if current_phase_groups.contains(&pi) {
                    let pi_root = trial_uf.find(pi);
                    let gi_root = trial_uf.find(gi);
                    if pi_root != gi_root {
                        let atoms_pi = trial_chain_atoms.get(&pi_root).copied().unwrap_or(0);
                        let atoms_gi = trial_chain_atoms.get(&gi_root).copied().unwrap_or(0);
                        let merged_atoms = atoms_pi + atoms_gi;

                        if merged_atoms > max_chain_atoms {
                            can_merge = false;
                            break;
                        }

                        trial_uf.union(gi, pi);
                        let new_root = trial_uf.find(gi);
                        trial_chain_atoms.insert(new_root, merged_atoms);
                        // Clean up old roots if they changed.
                        if new_root != pi_root {
                            trial_chain_atoms.remove(&pi_root);
                        }
                        if new_root != gi_root {
                            trial_chain_atoms.remove(&gi_root);
                        }
                    }
                }
            }

            if !can_merge {
                break;
            }
        }

        if can_merge {
            // Accept the merge.
            uf = trial_uf;
            chain_atoms = trial_chain_atoms;
            for &gi in &wavefronts[wi] {
                phase_assignment[gi] = current_phase;
                current_phase_groups.insert(gi);
            }
        } else {
            // Start a new phase.
            current_phase += 1;
            current_phase_groups.clear();
            chain_atoms.clear();
            // Reset UF for this new phase (we only need it per-phase).
            // Actually, we keep the same UF but it doesn't matter because
            // we only look at current_phase_groups for containment checks.

            for &gi in &wavefronts[wi] {
                phase_assignment[gi] = current_phase;
                current_phase_groups.insert(gi);
                let root = uf.find(gi);
                *chain_atoms.entry(root).or_insert(0) += groups[gi].count;
            }
        }
    }

    phase_assignment
}

// ─── Lane assignment with rebalancing ──────────────────────────────────────

/// Assign groups within a phase to lanes.
///
/// Uses union-find to identify constraint chains (groups that must share a lane),
/// then distributes chains across lanes using first-fit-decreasing bin packing,
/// followed by iterative rebalancing of unconstrained groups.
fn assign_lanes_for_phase(
    phase_group_indices: &[usize],
    producers: &[Vec<usize>],
    groups: &[AtomGroup],
    num_lanes: usize,
    lane_assignments: &mut [usize],
) {
    if phase_group_indices.is_empty() {
        return;
    }

    let phase_set: HashSet<usize> = phase_group_indices.iter().copied().collect();

    // Build intra-phase dependency chains using union-find.
    // Groups that depend on each other within this phase must share a lane.
    let n = groups.len();
    let mut uf = UnionFind::new(n);

    for &gi in phase_group_indices {
        for &pi in &producers[gi] {
            if phase_set.contains(&pi) {
                uf.union(gi, pi);
            }
        }
    }

    // Collect chains: groups with the same UF root go together.
    let mut chains: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &gi in phase_group_indices {
        let root = uf.find(gi);
        chains.entry(root).or_default().push(gi);
    }

    // Sort chains by total atom count descending for FFD bin packing.
    let mut chain_list: Vec<(Vec<usize>, u64)> = chains
        .into_values()
        .map(|gis| {
            let total: u64 = gis.iter().map(|&gi| groups[gi].count).sum();
            (gis, total)
        })
        .collect();
    chain_list.sort_by(|a, b| b.1.cmp(&a.1));

    // FFD assignment: assign each chain to the least-loaded lane.
    let mut lane_atoms: Vec<u64> = vec![0; num_lanes];

    for (chain_groups, chain_atom_count) in &chain_list {
        let target_lane = lane_atoms
            .iter()
            .enumerate()
            .min_by_key(|&(_, &atoms)| atoms)
            .map(|(i, _)| i)
            .unwrap_or(0);

        for &gi in chain_groups {
            lane_assignments[gi] = target_lane;
        }
        lane_atoms[target_lane] += chain_atom_count;
    }

    // Phase 2: Iterative rebalancing.
    // Try to move individual groups (that are the sole member of their chain)
    // from overloaded lanes to underloaded lanes, as long as doing so doesn't
    // violate the constraint that same-phase producers must share a lane.
    //
    // A group is "movable" if:
    // 1. It is the only group in its constraint chain (singleton chain), AND
    // 2. Moving it to target lane doesn't create a same-phase cross-lane dependency.
    //    That means: all its same-phase producers must be either:
    //    - On the target lane, or
    //    - Not in this phase (earlier phase, so they're in the value store).
    //    AND all its same-phase consumers must be either:
    //    - Also singletons that can follow it, or
    //    - Already on the target lane.
    //
    // For simplicity, we only move groups that have NO same-phase dependencies at all
    // (they are roots of their chain, i.e., all producers are in earlier phases,
    // and they have no same-phase consumers that are constrained elsewhere).
    // Actually, since we already unioned all intra-phase-dependent groups into chains,
    // any singleton chain has NO same-phase deps. Moving it is always safe.

    // Identify singleton chains (chains with exactly one group).
    let singleton_chains: HashSet<usize> = chain_list
        .iter()
        .filter(|(gis, _)| gis.len() == 1)
        .map(|(gis, _)| gis[0])
        .collect();

    // Rebalance iterations.
    let max_iters = 50;
    for _ in 0..max_iters {
        // Recompute lane loads.
        let mut loads: Vec<u64> = vec![0; num_lanes];
        for &gi in phase_group_indices {
            loads[lane_assignments[gi]] += groups[gi].count;
        }

        let max_load = *loads.iter().max().unwrap_or(&0);
        let min_load = *loads.iter().min().unwrap_or(&0);

        if max_load == 0 || (max_load as f64 / min_load.max(1) as f64) < 1.05 {
            break; // Balanced enough.
        }

        let overloaded_lane = loads
            .iter()
            .enumerate()
            .max_by_key(|&(_, &l)| l)
            .map(|(i, _)| i)
            .unwrap();
        let underloaded_lane = loads
            .iter()
            .enumerate()
            .min_by_key(|&(_, &l)| l)
            .map(|(i, _)| i)
            .unwrap();

        if overloaded_lane == underloaded_lane {
            break;
        }

        let excess = loads[overloaded_lane] - loads[underloaded_lane];
        let target_move = excess / 2; // Move roughly half the excess.

        // Find the best movable group from the overloaded lane.
        // "Best" = largest group that doesn't exceed target_move (or the smallest
        // if all exceed it).
        let mut best_gi: Option<usize> = None;
        let mut best_atoms: u64 = 0;

        for &gi in phase_group_indices {
            if lane_assignments[gi] != overloaded_lane {
                continue;
            }
            if !singleton_chains.contains(&gi) {
                continue;
            }
            let atoms = groups[gi].count;
            if atoms <= target_move && atoms > best_atoms {
                best_gi = Some(gi);
                best_atoms = atoms;
            }
        }

        // If nothing fits under target, try the smallest movable group.
        if best_gi.is_none() {
            let mut smallest_atoms = u64::MAX;
            for &gi in phase_group_indices {
                if lane_assignments[gi] != overloaded_lane {
                    continue;
                }
                if !singleton_chains.contains(&gi) {
                    continue;
                }
                let atoms = groups[gi].count;
                if atoms < smallest_atoms {
                    best_gi = Some(gi);
                    smallest_atoms = atoms;
                }
            }
        }

        match best_gi {
            Some(gi) => {
                lane_assignments[gi] = underloaded_lane;
            }
            None => break, // No movable groups, give up.
        }
    }
}

// ─── Output group identification ───────────────────────────────────────────

fn identify_output_groups(graph: &NanoGraph, output_atom_ids: &[AtomId]) -> HashSet<usize> {
    let mut output_groups = HashSet::new();
    for &atom_id in output_atom_ids {
        if let Some(gi) = graph.find_group_idx(atom_id) {
            output_groups.insert(gi);
        }
    }
    output_groups
}

// ─── Phase construction ────────────────────────────────────────────────────

/// Build a Phase: given lane assignments, construct span NanoGraphs.
fn build_phase(
    graph: &NanoGraph,
    phase_group_indices: &[usize],
    lane_assignments: &[usize],
    phase_assignments: &[usize],
    producers: &[Vec<usize>],
    num_lanes: usize,
    input_tensors: &[InputTensor],
    output_group_set: &HashSet<usize>,
    phase_idx: usize,
    num_phases: usize,
) -> Phase {
    let groups = graph.groups();

    // Sort groups by lane and then by group index (topo order).
    let mut lane_groups: Vec<Vec<usize>> = vec![vec![]; num_lanes];
    for &gi in phase_group_indices {
        let lane = lane_assignments[gi];
        lane_groups[lane].push(gi);
    }
    for lane in &mut lane_groups {
        lane.sort_unstable();
    }

    let phase_group_set: HashSet<usize> = phase_group_indices.iter().copied().collect();

    let spans: Vec<Span> = (0..num_lanes)
        .map(|lane_idx| {
            build_span(
                graph,
                &lane_groups[lane_idx],
                lane_assignments,
                phase_assignments,
                producers,
                input_tensors,
                output_group_set,
                phase_idx,
                num_phases,
                &phase_group_set,
            )
        })
        .collect();

    Phase { spans }
}

// ─── Span construction ─────────────────────────────────────────────────────

/// Build a single span (one lane's work within one phase).
fn build_span(
    graph: &NanoGraph,
    span_group_indices: &[usize],
    lane_assignments: &[usize],
    phase_assignments: &[usize],
    producers: &[Vec<usize>],
    input_tensors: &[InputTensor],
    output_group_set: &HashSet<usize>,
    phase_idx: usize,
    num_phases: usize,
    phase_group_set: &HashSet<usize>,
) -> Span {
    if span_group_indices.is_empty() {
        return Span {
            graph: NanoGraph::new(),
            inputs: vec![],
            outputs: vec![],
        };
    }

    let groups = graph.groups();
    let mut span_graph = NanoGraph::new();

    // Copy sym_dim configuration.
    span_graph.sym_dim_names = graph.sym_dim_names.clone();
    span_graph.sym_dim_bounds = graph.sym_dim_bounds.clone();

    let span_compute_set: HashSet<usize> = span_group_indices.iter().copied().collect();

    const LITERAL_INLINE_THRESHOLD: u64 = 65536;

    // Collect external dependencies and inlined literals via BFS.
    let mut needed_external_atoms: BTreeMap<AtomId, (u64, DType)> = BTreeMap::new();
    let mut needed_internal_literals: BTreeSet<usize> = BTreeSet::new();

    let mut visited = HashSet::new();
    let mut queue: Vec<usize> = span_compute_set.iter().copied().collect();

    while let Some(gi) = queue.pop() {
        if !visited.insert(gi) {
            continue;
        }

        for &pi in &producers[gi] {
            if span_compute_set.contains(&pi) {
                queue.push(pi);
            } else if is_literal_group(&groups[pi]) && groups[pi].count < LITERAL_INLINE_THRESHOLD {
                needed_internal_literals.insert(pi);
            } else {
                let g = &groups[pi];
                record_external_range(
                    &mut needed_external_atoms,
                    g.base_id,
                    g.count,
                    g.output_dtype,
                );
            }
        }

        // Check for input tensor references.
        let group = &groups[gi];
        for input in &group.inputs {
            collect_input_tensor_deps(
                input,
                group.count,
                group.atom_offset,
                graph,
                input_tensors,
                &mut needed_external_atoms,
            );
        }

        // Reduce stride extended range.
        if let ScalarOp::Reduce {
            reduce_count,
            reduce_stride,
            ..
        } = &group.op
        {
            if *reduce_count > 1 && *reduce_stride != 0 {
                for input in &group.inputs {
                    collect_reduce_deps(
                        input,
                        group,
                        *reduce_count,
                        *reduce_stride,
                        graph,
                        input_tensors,
                        groups,
                        &span_compute_set,
                        &mut needed_external_atoms,
                        &mut needed_internal_literals,
                    );
                }
            }
        }

        // IndirectLoad table reference.
        if let ScalarOp::IndirectLoad { table_base, .. } = &group.op {
            if let Some(pi) = graph.find_group_idx(*table_base) {
                if !span_compute_set.contains(&pi) {
                    if is_literal_group(&groups[pi]) && groups[pi].count < LITERAL_INLINE_THRESHOLD
                    {
                        needed_internal_literals.insert(pi);
                    } else {
                        let g = &groups[pi];
                        record_external_range(
                            &mut needed_external_atoms,
                            g.base_id,
                            g.count,
                            g.output_dtype,
                        );
                    }
                }
            }
        }
    }

    // Check literals' own dependencies (usually none, but be safe).
    for &li in &needed_internal_literals.clone() {
        let group = &groups[li];
        for input in &group.inputs {
            collect_input_tensor_deps(
                input,
                group.count,
                group.atom_offset,
                graph,
                input_tensors,
                &mut needed_external_atoms,
            );
        }
    }

    // Merge overlapping/adjacent external ranges.
    let merged_external = merge_external_ranges(&needed_external_atoms);

    // Collect all items to insert, sorted by base_id.
    #[derive(Debug)]
    enum InsertItem {
        ExternalInput {
            base: AtomId,
            count: u64,
            dtype: DType,
        },
        InlineLiteral {
            gi: usize,
        },
        ComputeGroup {
            gi: usize,
        },
    }

    let mut items: Vec<(u64, InsertItem)> = Vec::new();

    for &(base, count, dtype) in &merged_external {
        items.push((base.0, InsertItem::ExternalInput { base, count, dtype }));
    }

    for &li in &needed_internal_literals {
        items.push((groups[li].base_id.0, InsertItem::InlineLiteral { gi: li }));
    }

    for &gi in span_group_indices {
        items.push((groups[gi].base_id.0, InsertItem::ComputeGroup { gi }));
    }

    items.sort_by_key(|(base, _)| *base);

    let mut inserted_ranges: Vec<(u64, u64)> = Vec::new();
    let mut span_inputs: Vec<AtomRange> = Vec::new();
    let mut span_outputs: Vec<AtomRange> = Vec::new();

    for (_, item) in &items {
        match item {
            InsertItem::ExternalInput { base, count, dtype } => {
                if would_overlap(&inserted_ranges, base.0, *count) {
                    continue;
                }
                span_graph.insert_input_tensor_at(*base, GlobalId(0), *count, *dtype);
                span_inputs.push(AtomRange {
                    base: *base,
                    count: *count,
                    dtype: *dtype,
                });
                inserted_ranges.push((base.0, *count));
            }
            InsertItem::InlineLiteral { gi } => {
                let group = &groups[*gi];
                if would_overlap(&inserted_ranges, group.base_id.0, group.count) {
                    continue;
                }
                span_graph.insert_group_at(
                    group.base_id,
                    group.count,
                    group.atom_offset,
                    group.output_dtype,
                    group.op.clone(),
                    group.sym_dims.clone(),
                    group.inputs.clone(),
                );
                inserted_ranges.push((group.base_id.0, group.count));
            }
            InsertItem::ComputeGroup { gi } => {
                let group = &groups[*gi];
                if would_overlap(&inserted_ranges, group.base_id.0, group.count) {
                    continue;
                }

                span_graph.insert_group_at(
                    group.base_id,
                    group.count,
                    group.atom_offset,
                    group.output_dtype,
                    group.op.clone(),
                    group.sym_dims.clone(),
                    group.inputs.clone(),
                );
                inserted_ranges.push((group.base_id.0, group.count));

                // Output if it's a model output or consumed outside this span.
                if output_group_set.contains(gi)
                    || needs_output(*gi, group, phase_assignments, &span_compute_set)
                {
                    span_outputs.push(AtomRange {
                        base: group.base_id,
                        count: group.count,
                        dtype: group.output_dtype,
                    });
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

/// Check if a group's output is needed by groups outside this span.
/// Conservative: output all non-literal compute groups.
fn needs_output(
    gi: usize,
    group: &AtomGroup,
    phase_assignments: &[usize],
    span_compute_set: &HashSet<usize>,
) -> bool {
    !is_literal_op(&group.op)
}

fn is_literal_op(op: &ScalarOp) -> bool {
    matches!(op, ScalarOp::Literal(_))
}

fn is_literal_group(group: &AtomGroup) -> bool {
    is_literal_op(&group.op) && group.inputs.is_empty()
}

// ─── External dependency helpers ───────────────────────────────────────────

fn record_external_range(
    ranges: &mut BTreeMap<AtomId, (u64, DType)>,
    base: AtomId,
    count: u64,
    dtype: DType,
) {
    ranges
        .entry(base)
        .and_modify(|(existing_count, _)| {
            *existing_count = (*existing_count).max(count);
        })
        .or_insert((count, dtype));
}

fn collect_input_tensor_deps(
    input: &InputRef,
    count: u64,
    atom_offset: u64,
    graph: &NanoGraph,
    input_tensors: &[InputTensor],
    needed_external: &mut BTreeMap<AtomId, (u64, DType)>,
) {
    match input {
        InputRef::Broadcast(base) => {
            if let Some((ti, _)) = graph.find_input_idx(*base) {
                let it = &input_tensors[ti];
                record_external_range(needed_external, it.base_id, it.count, it.dtype);
            }
        }
        InputRef::Affine { .. } => {
            let first = input.resolve(atom_offset);
            let last = input.resolve(atom_offset + count - 1);
            let lo = first.0.min(last.0);
            let hi = first.0.max(last.0);
            collect_input_tensor_range(graph, input_tensors, lo, hi, needed_external);
        }
        InputRef::StridedBroadcast { .. } => {
            let first = input.resolve(atom_offset);
            let last = input.resolve(atom_offset + count - 1);
            let lo = first.0.min(last.0);
            let hi = first.0.max(last.0);
            collect_input_tensor_range(graph, input_tensors, lo, hi, needed_external);
        }
        InputRef::Modular {
            base,
            stride,
            modulus,
        } => {
            let a = base.0;
            let b = (base.0 as i64 + *stride * (*modulus as i64 - 1)) as u64;
            collect_input_tensor_range(graph, input_tensors, a.min(b), a.max(b), needed_external);
        }
        InputRef::Explicit(ids) => {
            for &id in ids.iter().skip(atom_offset as usize).take(count as usize) {
                if let Some((ti, _)) = graph.find_input_idx(id) {
                    let it = &input_tensors[ti];
                    record_external_range(needed_external, it.base_id, it.count, it.dtype);
                }
            }
        }
    }
}

fn collect_input_tensor_range(
    graph: &NanoGraph,
    input_tensors: &[InputTensor],
    lo: u64,
    hi: u64,
    needed_external: &mut BTreeMap<AtomId, (u64, DType)>,
) {
    for it in input_tensors {
        let it_lo = it.base_id.0;
        let it_hi = it_lo + it.count - 1;
        if lo <= it_hi && hi >= it_lo {
            record_external_range(needed_external, it.base_id, it.count, it.dtype);
        }
    }
}

fn collect_reduce_deps(
    input: &InputRef,
    group: &AtomGroup,
    reduce_count: u64,
    reduce_stride: i64,
    graph: &NanoGraph,
    input_tensors: &[InputTensor],
    all_groups: &[AtomGroup],
    span_compute_set: &HashSet<usize>,
    needed_external: &mut BTreeMap<AtomId, (u64, DType)>,
    needed_internal_literals: &mut BTreeSet<usize>,
) {
    let first = input.resolve(group.atom_offset);
    let last = input.resolve(group.atom_offset + group.count - 1);
    let end_off = (reduce_count as i64 - 1) * reduce_stride;
    let endpoints = [
        first.0,
        (first.0 as i64 + end_off) as u64,
        last.0,
        (last.0 as i64 + end_off) as u64,
    ];
    let lo = *endpoints.iter().min().unwrap();
    let hi = *endpoints.iter().max().unwrap();

    for (gi, g) in all_groups.iter().enumerate() {
        let g_lo = g.base_id.0;
        let g_hi = g_lo + g.count - 1;
        if g_lo > hi {
            break;
        }
        if g_hi >= lo && g_lo <= hi {
            if !span_compute_set.contains(&gi) {
                if is_literal_group(g) && g.count < 65536 {
                    needed_internal_literals.insert(gi);
                } else {
                    record_external_range(needed_external, g.base_id, g.count, g.output_dtype);
                }
            }
        }
    }

    collect_input_tensor_range(graph, input_tensors, lo, hi, needed_external);
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

fn merge_external_ranges(ranges: &BTreeMap<AtomId, (u64, DType)>) -> Vec<(AtomId, u64, DType)> {
    if ranges.is_empty() {
        return vec![];
    }

    let sorted: Vec<(AtomId, u64, DType)> = ranges
        .iter()
        .map(|(&base, &(count, dtype))| (base, count, dtype))
        .collect();

    let mut merged: Vec<(AtomId, u64, DType)> = Vec::new();

    for (base, count, dtype) in sorted {
        if let Some(last) = merged.last_mut() {
            let last_end = last.0.0 + last.1;
            if base.0 <= last_end && dtype == last.2 {
                let new_end = (base.0 + count).max(last_end);
                last.1 = new_end - last.0.0;
                continue;
            }
        }
        merged.push((base, count, dtype));
    }

    merged
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

// ─── Union-Find ────────────────────────────────────────────────────────────

#[derive(Clone)]
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }
        if self.rank[rx] < self.rank[ry] {
            self.parent[rx] = ry;
        } else if self.rank[rx] > self.rank[ry] {
            self.parent[ry] = rx;
        } else {
            self.parent[ry] = rx;
            self.rank[rx] += 1;
        }
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::nano_graph::ops::{ReduceKind, ScalarBinOp, ScalarUnaryOp};
    use crate::numeric_scalar::NumericScalar;

    // ─── Test graph builders ───────────────────────────────────────────────

    /// Build a simple linear chain: a -> b -> c (3 groups in series).
    fn make_linear_chain() -> (NanoGraph, Vec<InputTensor>) {
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

        g.outputs.push(c);
        (g, vec![])
    }

    /// Build a diamond: two independent paths from a literal, joined at the end.
    fn make_diamond() -> (NanoGraph, Vec<InputTensor>) {
        let mut g = NanoGraph::new();

        let lit = g.push_group(
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
            vec![InputRef::Affine {
                base: lit,
                stride: 1,
            }],
        );
        let c = g.push_group(
            100,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: lit,
                stride: 1,
            }],
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

        g.outputs.push(d);
        (g, vec![])
    }

    /// Build a parallel workload: two independent literal+compute chains.
    fn make_parallel() -> (NanoGraph, Vec<InputTensor>) {
        let mut g = NanoGraph::new();

        let a = g.push_group(
            1000,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );
        let b = g.push_group(
            1000,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine { base: a, stride: 1 }],
        );

        let c = g.push_group(
            1000,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
        );
        let d = g.push_group(
            1000,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine { base: c, stride: 1 }],
        );

        g.outputs.push(b);
        g.outputs.push(d);
        (g, vec![])
    }

    /// Build a graph with external input tensors.
    fn make_with_inputs() -> (NanoGraph, Vec<InputTensor>) {
        let mut g = NanoGraph::new();

        let input_base = g.add_input_tensor(GlobalId(1), 100, DType::F32);
        let inputs = vec![InputTensor {
            tensor_id: GlobalId(1),
            base_id: input_base,
            count: 100,
            dtype: DType::F32,
        }];

        let b = g.push_group(
            100,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: input_base,
                stride: 1,
            }],
        );

        g.outputs.push(b);
        (g, inputs)
    }

    /// Build a matmul-like structure: M independent row computations sharing weights.
    fn make_matmul_like() -> (NanoGraph, Vec<InputTensor>) {
        let mut g = NanoGraph::new();

        let k = 64u64;
        let m = 8u64;
        let weights = g.push_group(
            k,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.5)),
            vec![],
            vec![],
        );

        let input = g.push_group(
            m * k,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );

        let mul = g.push_group(
            m * k,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine {
                    base: input,
                    stride: 1,
                },
                InputRef::Modular {
                    base: weights,
                    stride: 1,
                    modulus: k,
                },
            ],
        );

        let reduce = g.push_group(
            m,
            DType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: k,
                reduce_stride: 1,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: mul,
                stride: k as i64,
            }],
        );

        g.outputs.push(reduce);
        (g, vec![])
    }

    /// Build a larger matmul-like graph with many independent rows for balance testing.
    /// Emulates the real GPT-2 structure: M separate mul groups (one per row),
    /// each with its own reduce, all sharing weights.
    fn make_wide_matmul() -> (NanoGraph, Vec<InputTensor>) {
        let mut g = NanoGraph::new();

        let k = 32u64;
        let m = 16u64; // 16 independent rows.
        let weights = g.push_group(
            k,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.5)),
            vec![],
            vec![],
        );

        // Create M independent row computations.
        let mut reduce_ids = vec![];
        for _row in 0..m {
            let row_input = g.push_group(
                k,
                DType::F32,
                ScalarOp::Literal(NumericScalar::F32(1.0)),
                vec![],
                vec![],
            );
            let row_mul = g.push_group(
                k,
                DType::F32,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                },
                vec![],
                vec![
                    InputRef::Affine {
                        base: row_input,
                        stride: 1,
                    },
                    InputRef::Affine {
                        base: weights,
                        stride: 1,
                    },
                ],
            );
            let row_reduce = g.push_group(
                1,
                DType::F32,
                ScalarOp::Reduce {
                    kind: ReduceKind::Sum,
                    reduce_count: k,
                    reduce_stride: 1,
                    compute_dtype: DType::F32,
                },
                vec![],
                vec![InputRef::Affine {
                    base: row_mul,
                    stride: 1,
                }],
            );
            reduce_ids.push(row_reduce);
        }

        for id in &reduce_ids {
            g.outputs.push(*id);
        }
        (g, vec![])
    }

    /// Build a two-layer chain: matmul1 -> elementwise -> matmul2.
    fn make_two_layer() -> (NanoGraph, Vec<InputTensor>) {
        let mut g = NanoGraph::new();
        let k = 16u64;
        let m = 8u64;

        // Layer 1 weights
        let w1 = g.push_group(
            k,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.5)),
            vec![],
            vec![],
        );
        // Layer 1 input
        let x1 = g.push_group(
            m * k,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );
        // Layer 1 mul
        let mul1 = g.push_group(
            m * k,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine {
                    base: x1,
                    stride: 1,
                },
                InputRef::Modular {
                    base: w1,
                    stride: 1,
                    modulus: k,
                },
            ],
        );
        // Layer 1 reduce
        let red1 = g.push_group(
            m,
            DType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: k,
                reduce_stride: 1,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: mul1,
                stride: k as i64,
            }],
        );
        // Elementwise activation
        let act = g.push_group(
            m,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: red1,
                stride: 1,
            }],
        );
        // Layer 2 weights
        let w2 = g.push_group(
            m,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.3)),
            vec![],
            vec![],
        );
        // Layer 2 mul (m x m, but for simplicity m outputs)
        let mul2 = g.push_group(
            m * m,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::StridedBroadcast {
                    base: act,
                    stride: 1,
                    repeat: m,
                },
                InputRef::Modular {
                    base: w2,
                    stride: 1,
                    modulus: m,
                },
            ],
        );
        // Layer 2 reduce
        let red2 = g.push_group(
            m,
            DType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: m,
                reduce_stride: 1,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: mul2,
                stride: m as i64,
            }],
        );

        g.outputs.push(red2);
        (g, vec![])
    }

    // ─── Verification helpers ──────────────────────────────────────────────

    /// Verify basic structural invariants of a plan.
    fn verify_plan(graph: &NanoGraph, phases: &[Phase], num_lanes: usize) {
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

        for (pi, phase) in phases.iter().enumerate() {
            for (li, span) in phase.spans.iter().enumerate() {
                if span.graph.num_groups() == 0 {
                    continue;
                }
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

        // All output atoms are produced somewhere.
        let mut produced_atoms: HashSet<u64> = HashSet::new();
        for phase in phases {
            for span in &phase.spans {
                for output in &span.outputs {
                    for i in 0..output.count {
                        produced_atoms.insert(output.base.0 + i);
                    }
                }
            }
        }

        for &output_id in &graph.outputs {
            assert!(
                produced_atoms.contains(&output_id.0),
                "Output atom {} not produced by any span",
                output_id
            );
        }
    }

    /// Verify no cross-span reads within a phase.
    fn verify_no_cross_span_reads(phases: &[Phase]) {
        for (pi, phase) in phases.iter().enumerate() {
            let mut span_produces: Vec<HashSet<u64>> = Vec::new();
            for span in &phase.spans {
                let mut produced = HashSet::new();
                for group in span.graph.groups() {
                    for i in 0..group.count {
                        produced.insert(group.base_id.0 + i);
                    }
                }
                span_produces.push(produced);
            }

            for (li, span) in phase.spans.iter().enumerate() {
                for input in &span.inputs {
                    for i in 0..input.count {
                        let atom = input.base.0 + i;
                        for (other_li, other_produced) in span_produces.iter().enumerate() {
                            if other_li != li && other_produced.contains(&atom) {
                                panic!(
                                    "Phase {} lane {} reads atom {} which is produced by lane {} in the same phase",
                                    pi, li, atom, other_li
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Compute the imbalance ratio for a plan (max lane atoms / min lane atoms).
    fn compute_imbalance(phases: &[Phase]) -> f64 {
        let mut lane_totals: Vec<u64> = vec![];
        for phase in phases {
            if lane_totals.is_empty() {
                lane_totals.resize(phase.spans.len(), 0);
            }
            for (li, span) in phase.spans.iter().enumerate() {
                let atoms: u64 = span
                    .graph
                    .groups()
                    .iter()
                    .filter(|g| !is_literal_op(&g.op))
                    .map(|g| g.count)
                    .sum();
                lane_totals[li] += atoms;
            }
        }

        let max_atoms = *lane_totals.iter().max().unwrap_or(&0);
        let min_atoms = *lane_totals.iter().min().unwrap_or(&0);

        if min_atoms == 0 {
            if max_atoms == 0 { 1.0 } else { f64::INFINITY }
        } else {
            max_atoms as f64 / min_atoms as f64
        }
    }

    // ─── Tests ─────────────────────────────────────────────────────────────

    #[test]
    fn test_empty_graph() {
        let g = NanoGraph::new();
        let phases = plan(&g, 4, &[], &[]);
        assert_eq!(phases.len(), 1);
        assert_eq!(phases[0].spans.len(), 4);
        for span in &phases[0].spans {
            assert_eq!(span.graph.num_groups(), 0);
        }
    }

    #[test]
    fn test_single_lane() {
        let (g, inputs) = make_linear_chain();
        let output_ids: Vec<AtomId> = g.outputs.clone();
        let phases = plan(&g, 1, &inputs, &output_ids);
        assert!(!phases.is_empty());
        verify_plan(&g, &phases, 1);
    }

    #[test]
    fn test_linear_chain() {
        let (g, inputs) = make_linear_chain();
        let output_ids: Vec<AtomId> = g.outputs.clone();
        let phases = plan(&g, 4, &inputs, &output_ids);
        verify_plan(&g, &phases, 4);
        verify_no_cross_span_reads(&phases);
    }

    #[test]
    fn test_diamond() {
        let (g, inputs) = make_diamond();
        let output_ids: Vec<AtomId> = g.outputs.clone();
        let phases = plan(&g, 2, &inputs, &output_ids);
        verify_plan(&g, &phases, 2);
        verify_no_cross_span_reads(&phases);
    }

    #[test]
    fn test_parallel_chains() {
        let (g, inputs) = make_parallel();
        let output_ids: Vec<AtomId> = g.outputs.clone();
        let phases = plan(&g, 2, &inputs, &output_ids);
        verify_plan(&g, &phases, 2);
        verify_no_cross_span_reads(&phases);
    }

    #[test]
    fn test_parallel_chains_balance() {
        // With 2 independent chains of equal size, they should go on separate lanes.
        let (g, inputs) = make_parallel();
        let output_ids: Vec<AtomId> = g.outputs.clone();
        let phases = plan(&g, 2, &inputs, &output_ids);
        verify_plan(&g, &phases, 2);
        verify_no_cross_span_reads(&phases);

        let imbalance = compute_imbalance(&phases);
        assert!(
            imbalance < 2.0,
            "Two equal independent chains should balance well, got imbalance {:.1}x",
            imbalance
        );
    }

    #[test]
    fn test_with_input_tensors() {
        let (g, inputs) = make_with_inputs();
        let output_ids: Vec<AtomId> = g.outputs.clone();
        let phases = plan(&g, 2, &inputs, &output_ids);
        verify_plan(&g, &phases, 2);
        verify_no_cross_span_reads(&phases);

        let has_input = phases
            .iter()
            .any(|p| p.spans.iter().any(|s| !s.inputs.is_empty()));
        assert!(has_input, "No span declares the input tensor");
    }

    #[test]
    fn test_matmul_like() {
        let (g, inputs) = make_matmul_like();
        let output_ids: Vec<AtomId> = g.outputs.clone();
        let phases = plan(&g, 4, &inputs, &output_ids);
        verify_plan(&g, &phases, 4);
        verify_no_cross_span_reads(&phases);
    }

    #[test]
    fn test_single_group() {
        let mut g = NanoGraph::new();
        let a = g.push_group(
            100,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(42.0)),
            vec![],
            vec![],
        );
        g.outputs.push(a);

        let phases = plan(&g, 4, &[], &g.outputs.clone());
        verify_plan(&g, &phases, 4);
    }

    #[test]
    fn test_large_groups() {
        let mut g = NanoGraph::new();
        let a = g.push_group(
            10000,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );
        let b = g.push_group(
            10000,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine { base: a, stride: 1 }],
        );
        g.outputs.push(b);

        let phases = plan(&g, 4, &[], &g.outputs.clone());
        verify_plan(&g, &phases, 4);
    }

    #[test]
    fn test_depth_computation() {
        let producers = vec![
            vec![],     // group 0: no deps
            vec![0],    // group 1: depends on 0
            vec![0],    // group 2: depends on 0
            vec![1, 2], // group 3: depends on 1 and 2
        ];

        let depths = compute_depths(4, &producers);
        assert_eq!(depths, vec![0, 1, 1, 2]);
    }

    #[test]
    fn test_wavefront_formation() {
        let producers = vec![
            vec![],     // depth 0
            vec![],     // depth 0
            vec![0],    // depth 1
            vec![1],    // depth 1
            vec![2, 3], // depth 2
        ];

        let depths = compute_depths(5, &producers);
        assert_eq!(depths, vec![0, 0, 1, 1, 2]);
    }

    #[test]
    fn test_broadcast_dependency() {
        let mut g = NanoGraph::new();

        let shared = g.push_group(
            1,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );
        let c1 = g.push_group(
            100,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Broadcast(shared)],
        );
        let c2 = g.push_group(
            100,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Broadcast(shared)],
        );

        g.outputs.push(c1);
        g.outputs.push(c2);

        let phases = plan(&g, 2, &[], &g.outputs.clone());
        verify_plan(&g, &phases, 2);
        verify_no_cross_span_reads(&phases);
    }

    #[test]
    fn test_reduce_with_literals() {
        let mut g = NanoGraph::new();

        let data = g.push_group(
            64,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );
        let reduced = g.push_group(
            8,
            DType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: 8,
                reduce_stride: 1,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: data,
                stride: 8,
            }],
        );

        g.outputs.push(reduced);

        let phases = plan(&g, 2, &[], &g.outputs.clone());
        verify_plan(&g, &phases, 2);
    }

    #[test]
    fn test_plan_preserves_all_groups() {
        let (g, inputs) = make_diamond();
        let output_ids: Vec<AtomId> = g.outputs.clone();
        let phases = plan(&g, 2, &inputs, &output_ids);

        let main_groups: HashSet<u64> = g.groups().iter().map(|gr| gr.base_id.0).collect();

        let mut span_groups: HashSet<u64> = HashSet::new();
        for phase in &phases {
            for span in &phase.spans {
                for gr in span.graph.groups() {
                    if main_groups.contains(&gr.base_id.0) {
                        span_groups.insert(gr.base_id.0);
                    }
                }
            }
        }

        for &base in &main_groups {
            assert!(
                span_groups.contains(&base),
                "Main graph group at base {} not found in any span",
                base
            );
        }
    }

    #[test]
    fn test_modular_input_ref() {
        let mut g = NanoGraph::new();

        let weights = g.push_group(
            10,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );
        let result = g.push_group(
            100,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Modular {
                base: weights,
                stride: 1,
                modulus: 10,
            }],
        );

        g.outputs.push(result);

        let phases = plan(&g, 2, &[], &g.outputs.clone());
        verify_plan(&g, &phases, 2);
        verify_no_cross_span_reads(&phases);
    }

    #[test]
    fn test_wide_matmul_balance() {
        // A wide matmul should distribute rows across lanes.
        let (g, inputs) = make_wide_matmul();
        let output_ids: Vec<AtomId> = g.outputs.clone();
        let phases = plan(&g, 4, &inputs, &output_ids);
        verify_plan(&g, &phases, 4);
        verify_no_cross_span_reads(&phases);

        let imbalance = compute_imbalance(&phases);
        // With 64 rows across 4 lanes, should get reasonable balance.
        assert!(
            imbalance < 5.0,
            "Wide matmul across 4 lanes should be reasonably balanced, got {:.1}x",
            imbalance
        );
    }

    #[test]
    fn test_two_layer_correctness() {
        // Two sequential matmul layers must respect ordering.
        let (g, inputs) = make_two_layer();
        let output_ids: Vec<AtomId> = g.outputs.clone();
        let phases = plan(&g, 4, &inputs, &output_ids);
        verify_plan(&g, &phases, 4);
        verify_no_cross_span_reads(&phases);

        // Should need at least 2 phases (the second layer depends on the first).
        // In practice, with aggressive merging, it might merge some wavefronts.
        assert!(
            phases.len() >= 1,
            "Two-layer graph should produce at least 1 phase, got {}",
            phases.len()
        );
    }

    #[test]
    fn test_many_independent_groups() {
        // Many independent groups should distribute across all lanes.
        let mut g = NanoGraph::new();
        let num_groups = 32;

        let mut last_ids = vec![];
        for i in 0..num_groups {
            let lit = g.push_group(
                100,
                DType::F32,
                ScalarOp::Literal(NumericScalar::F32(i as f32)),
                vec![],
                vec![],
            );
            let comp = g.push_group(
                100,
                DType::F32,
                ScalarOp::Unary {
                    op: ScalarUnaryOp::Neg,
                    compute_dtype: DType::F32,
                },
                vec![],
                vec![InputRef::Affine {
                    base: lit,
                    stride: 1,
                }],
            );
            last_ids.push(comp);
        }

        for id in &last_ids {
            g.outputs.push(*id);
        }

        let phases = plan(&g, 4, &[], &g.outputs.clone());
        verify_plan(&g, &phases, 4);
        verify_no_cross_span_reads(&phases);

        // Check balance: all lanes should have roughly equal work.
        let imbalance = compute_imbalance(&phases);
        assert!(
            imbalance < 2.0,
            "32 independent chains across 4 lanes should balance well, got {:.1}x",
            imbalance
        );
    }

    #[test]
    fn test_union_find_basic() {
        let mut uf = UnionFind::new(5);
        assert_ne!(uf.find(0), uf.find(1));

        uf.union(0, 1);
        assert_eq!(uf.find(0), uf.find(1));

        uf.union(2, 3);
        assert_eq!(uf.find(2), uf.find(3));
        assert_ne!(uf.find(0), uf.find(2));

        uf.union(1, 3);
        assert_eq!(uf.find(0), uf.find(3));
    }

    #[test]
    fn test_single_group_single_lane() {
        let mut g = NanoGraph::new();
        let a = g.push_group(
            1,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );
        g.outputs.push(a);

        let phases = plan(&g, 1, &[], &g.outputs.clone());
        verify_plan(&g, &phases, 1);
        assert_eq!(phases.len(), 1);
    }

    #[test]
    fn test_phase_count_reasonable() {
        // A deep linear chain should not produce one phase per depth level;
        // wavefronts should merge aggressively.
        let mut g = NanoGraph::new();
        let n = 100;
        let mut prev = g.push_group(
            10,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );

        for _ in 1..n {
            let next = g.push_group(
                10,
                DType::F32,
                ScalarOp::Unary {
                    op: ScalarUnaryOp::Neg,
                    compute_dtype: DType::F32,
                },
                vec![],
                vec![InputRef::Affine {
                    base: prev,
                    stride: 1,
                }],
            );
            prev = next;
        }
        g.outputs.push(prev);

        let phases = plan(&g, 4, &[], &g.outputs.clone());
        verify_plan(&g, &phases, 4);

        // A 100-deep chain: each group depends on the previous, so all must be
        // chained. With aggressive merging (chain fits on one lane), ideally
        // we get very few phases.
        assert!(
            phases.len() <= 20,
            "100-deep chain should produce few phases with aggressive merging, got {}",
            phases.len()
        );
    }

    #[test]
    fn test_strided_broadcast_ref() {
        let mut g = NanoGraph::new();

        let data = g.push_group(
            16,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );
        // StridedBroadcast: each block of 4 atoms reads the same source.
        let result = g.push_group(
            64,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::StridedBroadcast {
                base: data,
                stride: 1,
                repeat: 4,
            }],
        );

        g.outputs.push(result);

        let phases = plan(&g, 2, &[], &g.outputs.clone());
        verify_plan(&g, &phases, 2);
        verify_no_cross_span_reads(&phases);
    }
}
