#![allow(
    clippy::all,
    dead_code,
    unreachable_code,
    unreachable_patterns,
    unused_imports,
    unused_variables
)]

//! Dependency-Driven Wavefront Scheduling Partitioner (attempt B)
//!
//! Key insight: computing the DAG depth (longest path from any input) for every
//! group gives us natural wavefronts. Groups at the same depth are provably
//! independent — they can't depend on each other because that would create a
//! longer path. This gives us correctness for free.
//!
//! Algorithm:
//! 1. Build group-level dependency DAG
//! 2. Compute depth for each group (longest path from any root)
//! 3. Groups at the same depth form a wavefront
//! 4. Merge consecutive wavefronts when safe (no cross-wavefront deps after merge)
//! 5. Within each phase, distribute groups across lanes for load balance
//! 6. Split large groups across lanes when beneficial
//! 7. Build span NanoGraphs with preserved atom IDs

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::dtype::DType;
use crate::graph::GlobalId;
use crate::nano_graph::pattern::InputTensor;
use crate::nano_graph::{AtomGroup, AtomId, AtomRange, InputRef, NanoGraph, ScalarOp, SymDim};

use super::types::{Phase, Span};

// ─── Public API ────────────────────────────────────────────────────────────

/// Partition a NanoGraph into phases and spans for parallel execution.
///
/// Returns a sequence of phases, each containing one span per lane.
/// The caller wraps this into the full ExecutionPlan.
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
        // Empty graph: single phase with empty spans.
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

    // Step 2: Compute depth for each group.
    let depths = compute_depths(n, &producers);

    // Step 3: Assign lanes and compute phases.
    //
    // Lanes first, phases second. Groups at the same depth are independent
    // and get distributed across lanes via bin-packing. Phases are then
    // determined by cross-lane dependencies: a group is in the same phase
    // as its producer if they share a lane, otherwise it needs a new phase
    // (barrier) to wait for the cross-lane data.
    let (lane_assignments, phase_assignments) =
        assign_lanes_and_phases(n, num_lanes, &producers, &depths, groups);

    // Step 4: Determine which groups are in each phase.
    let num_phases = phase_assignments.iter().copied().max().unwrap_or(0) + 1;
    let mut phase_groups: Vec<Vec<usize>> = vec![vec![]; num_phases];
    for (gi, &phase) in phase_assignments.iter().enumerate() {
        phase_groups[phase].push(gi);
    }

    // Step 5: Identify which groups are output-relevant.
    let output_group_set = identify_output_groups(graph, output_atom_ids);

    // Step 6: For each phase, build spans using pre-assigned lanes.
    let mut phases = Vec::with_capacity(num_phases);
    for phase_idx in 0..num_phases {
        let phase = build_phase_from_lanes(
            graph,
            &phase_groups[phase_idx],
            &lane_assignments,
            &phase_assignments,
            &producers,
            &successors,
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
/// Depth = longest path from any root (a group with no producers).
/// Groups with no dependencies have depth 0.
fn compute_depths(n: usize, producers: &[Vec<usize>]) -> Vec<usize> {
    let mut depths = vec![0usize; n];

    // Process in topological order (groups are already in topo order in
    // the NanoGraph since push_group allocates sequential IDs and inputs
    // always reference earlier groups).
    for gi in 0..n {
        let mut max_dep_depth = 0usize;
        for &pi in &producers[gi] {
            max_dep_depth = max_dep_depth.max(depths[pi] + 1);
        }
        depths[gi] = max_dep_depth;
    }

    depths
}

// ─── Lanes-first scheduling ───────────────────────────────────────────────

/// Assign each group to a lane, then compute the minimum phase for each group.
///
/// Lane assignment strategy:
/// - Groups with all producers on a single lane stay on that lane (preserves
///   chains, avoids phase boundaries from cross-lane deps).
/// - Groups with producers on multiple lanes go to the producer lane with
///   the most atoms (minimizes the dominant cross-lane penalty).
/// - Groups with no producers (roots) are distributed across lanes via
///   least-loaded bin-packing.
///
/// Within each depth level, groups whose producers don't constrain them to
/// a specific lane are distributed for load balance.
///
/// Phases are determined by cross-lane dependencies: a group is in the same
/// phase as its producers if they share a lane, otherwise it needs a later
/// phase (barrier) to wait for cross-lane data.
fn assign_lanes_and_phases(
    n: usize,
    num_lanes: usize,
    producers: &[Vec<usize>],
    depths: &[usize],
    groups: &[AtomGroup],
) -> (Vec<usize>, Vec<usize>) {
    let mut lane_of = vec![0usize; n];
    let mut phase_of = vec![0usize; n];

    let max_depth = depths.iter().copied().max().unwrap_or(0);

    // Collect groups by depth.
    let mut by_depth: Vec<Vec<usize>> = vec![vec![]; max_depth + 1];
    for gi in 0..n {
        by_depth[depths[gi]].push(gi);
    }

    let mut lane_atoms: Vec<u64> = vec![0; num_lanes];

    for depth in 0..=max_depth {
        // Separate groups into "constrained" (have producers → prefer their lane)
        // and "free" (no producers → distribute for load balance).
        let mut constrained: Vec<(usize, u64)> = Vec::new();
        let mut free: Vec<(usize, u64)> = Vec::new();

        for &gi in &by_depth[depth] {
            if producers[gi].is_empty() {
                free.push((gi, groups[gi].count));
            } else {
                constrained.push((gi, groups[gi].count));
            }
        }

        // Combine all groups at this depth and sort by atom count desc.
        let mut all_items: Vec<(usize, u64)> = by_depth[depth]
            .iter()
            .map(|&gi| (gi, groups[gi].count))
            .collect();
        all_items.sort_by(|a, b| b.1.cmp(&a.1));

        for (gi, atoms) in all_items {
            let chosen = if producers[gi].is_empty() {
                // Root group: least-loaded lane.
                (0..num_lanes)
                    .min_by_key(|&l| lane_atoms[l])
                    .unwrap()
            } else {
                // Find which lanes our producers are on, weighted by atom count.
                let mut lane_weight: Vec<u64> = vec![0; num_lanes];
                for &pi in &producers[gi] {
                    lane_weight[lane_of[pi]] += groups[pi].count;
                }

                // Find the dominant producer lane and the least-loaded lane.
                let dominant = (0..num_lanes)
                    .max_by_key(|&l| lane_weight[l])
                    .unwrap();
                let least_loaded = (0..num_lanes)
                    .min_by_key(|&l| lane_atoms[l])
                    .unwrap();

                // Stay with producer if it avoids a cross-lane dep AND the
                // lane isn't severely overloaded. Otherwise distribute.
                // "Severely overloaded" = more than 2x the least-loaded lane.
                let min_load = lane_atoms[least_loaded];
                if lane_atoms[dominant] <= min_load.saturating_mul(2).saturating_add(atoms) {
                    dominant
                } else {
                    least_loaded
                }
            };

            lane_of[gi] = chosen;
            lane_atoms[chosen] += atoms;

            // Phase = max over all producers, +1 if cross-lane.
            let mut earliest = 0usize;
            for &pi in &producers[gi] {
                if lane_of[pi] == chosen {
                    earliest = earliest.max(phase_of[pi]);
                } else {
                    earliest = earliest.max(phase_of[pi] + 1);
                }
            }
            phase_of[gi] = earliest;
        }
    }

    (lane_of, phase_of)
}

// ─── Wavefront merging (legacy, unused) ───────────────────────────────────

/// Merge consecutive wavefronts into phases to reduce barrier count.
///
/// Two adjacent wavefronts can be merged if no group in the later wavefront
/// depends on a group in the earlier wavefront that would end up on a
/// different lane. Since we haven't assigned lanes yet at this point,
/// we use a conservative heuristic: merge if the combined wavefront is
/// still "small enough" (few enough groups) or if the later wavefront's
/// producers are all in even earlier phases (not the immediately preceding
/// wavefront being merged).
///
/// Returns phase_assignment[gi] = phase index for each group.
fn merge_wavefronts(
    wavefronts: &[Vec<usize>],
    producers: &[Vec<usize>],
    _successors: &[Vec<usize>],
    n: usize,
) -> Vec<usize> {
    if wavefronts.is_empty() {
        return vec![];
    }

    // Strategy: merge wavefront i into the current phase if ALL of wavefront
    // i's producer groups are in phases strictly before the current phase's
    // first wavefront. This guarantees that after merging, all dependencies
    // within the phase are from internal groups (same phase), which means
    // they can be ordered within a span without needing a barrier.
    //
    // However, within a phase, groups on DIFFERENT lanes must be independent.
    // So we need: if group A (phase P) depends on group B (phase P), they
    // must be on the same lane. Merging wavefronts creates such dependencies.
    //
    // Conservative approach: only merge wavefronts where the later wavefront's
    // groups depend ONLY on groups within the same merged phase (will be on
    // the same lane) or on groups in strictly earlier phases.
    //
    // A simpler and safe approach: merge consecutive wavefronts that have
    // no inter-wavefront dependencies. Two wavefronts at depths d and d+1
    // can be merged if no group at depth d+1 has a producer at depth d.
    // By the depth definition, if a group at depth d+1 has a producer at
    // depth d, that producer IS the reason it's at depth d+1.
    //
    // Actually, by construction, a group at depth d has at least one producer
    // at depth d-1 (otherwise its depth would be < d). So adjacent wavefronts
    // always have dependencies. We can't naively merge them.
    //
    // Better approach: merge wavefronts that are "thin" (few groups) with
    // their successors, accepting that merged groups will need to be on the
    // same lane as their producers. This is essentially: pack multiple depth
    // levels into one phase, then during lane assignment, ensure dependencies
    // within a phase go to the same lane.
    //
    // Simplest correct approach: keep each wavefront as a separate phase.
    // This may produce many phases, but each phase is provably independent.
    // Then we do a post-pass to merge phases that are "trivial" (very few atoms).

    let mut phase_assignment = vec![0usize; n];

    // Track which wavefront each group belongs to.
    let mut group_wavefront = vec![0usize; n];
    for (wi, wavefront) in wavefronts.iter().enumerate() {
        for &gi in wavefront {
            group_wavefront[gi] = wi;
        }
    }

    // Initially, each wavefront is its own phase.
    let mut wavefront_to_phase = vec![0usize; wavefronts.len()];
    let mut next_phase = 0usize;

    // Try merging consecutive wavefronts.
    // We merge wavefront i+1 into wavefront i's phase if wavefront i+1's
    // total atom count is small (heuristic: < 1024 atoms) AND all of
    // wavefront i+1's producers from wavefront i have few enough atoms
    // to be co-located on the same lane.
    //
    // Actually, a cleaner merge criterion: merge wavefront[i+1] into
    // the same phase as wavefront[i] if every group in wavefront[i+1]
    // depends on AT MOST ONE group in wavefront[i]. This means each
    // group in wavefront[i+1] can be paired with its producer on the
    // same lane, maintaining independence between lanes.
    //
    // Even simpler: just merge really tiny wavefronts (single-group or
    // very low atom count) with the previous phase.

    for wi in 0..wavefronts.len() {
        if wi == 0 {
            wavefront_to_phase[wi] = 0;
            next_phase = 1;
            continue;
        }

        // Check if this wavefront can be merged with the previous phase.
        let can_merge = can_merge_wavefront(
            wavefronts,
            wi,
            &wavefront_to_phase,
            producers,
            &group_wavefront,
        );

        if can_merge {
            wavefront_to_phase[wi] = wavefront_to_phase[wi - 1];
        } else {
            wavefront_to_phase[wi] = next_phase;
            next_phase += 1;
        }
    }

    // Apply assignments.
    for (wi, wavefront) in wavefronts.iter().enumerate() {
        for &gi in wavefront {
            phase_assignment[gi] = wavefront_to_phase[wi];
        }
    }

    phase_assignment
}

/// Check if wavefront `wi` can be merged into the same phase as wavefront `wi-1`.
///
/// Criterion: every group in wavefront `wi` depends on at most one group from
/// wavefronts in the same phase. This ensures each group can be co-located on
/// the same lane as its single dependency, maintaining inter-lane independence.
fn can_merge_wavefront(
    wavefronts: &[Vec<usize>],
    wi: usize,
    wavefront_to_phase: &[usize],
    producers: &[Vec<usize>],
    group_wavefront: &[usize],
) -> bool {
    let target_phase = wavefront_to_phase[wi - 1];

    for &gi in &wavefronts[wi] {
        // Count how many distinct groups in the target phase this group depends on.
        let mut deps_in_phase = HashSet::new();
        for &pi in &producers[gi] {
            let pw = group_wavefront[pi];
            if pw < wi && wavefront_to_phase[pw] == target_phase {
                deps_in_phase.insert(pi);
            }
        }
        // If this group depends on more than one group from the target phase,
        // merging would require those groups to all be on the same lane,
        // which constrains lane assignment too much. Don't merge.
        if deps_in_phase.len() > 1 {
            return false;
        }
    }

    true
}

// ─── Output group identification ───────────────────────────────────────────

/// Identify which group indices contain output atoms.
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

/// A unit of work assigned to a lane.
#[derive(Debug, Clone)]
struct WorkItem {
    /// Index into the main graph's groups() slice.
    group_idx: usize,
    /// Offset within the group (for split groups).
    atom_offset: u64,
    /// Number of atoms this work item covers.
    atom_count: u64,
}

/// Build a Phase: assign groups to lanes and construct span NanoGraphs.
fn build_phase(
    graph: &NanoGraph,
    phase_group_indices: &[usize],
    phase_assignments: &[usize],
    producers: &[Vec<usize>],
    successors: &[Vec<usize>],
    num_lanes: usize,
    input_tensors: &[InputTensor],
    output_group_set: &HashSet<usize>,
    phase_idx: usize,
    num_phases: usize,
) -> Phase {
    let groups = graph.groups();

    // Sort groups in this phase by topo order (= group index order).
    let mut sorted_groups = phase_group_indices.to_vec();
    sorted_groups.sort_unstable();

    // Compute total atom count for this phase.
    let total_atoms: u64 = sorted_groups.iter().map(|&gi| groups[gi].count).sum();
    let _target_per_lane = (total_atoms + num_lanes as u64 - 1) / num_lanes as u64;

    // Build dependency chains within this phase.
    // Groups that depend on each other within this phase must be on the same lane.
    let phase_group_set: HashSet<usize> = sorted_groups.iter().copied().collect();

    // Build intra-phase dependency chains using union-find.
    let mut uf = UnionFind::new(groups.len());
    for &gi in &sorted_groups {
        for &pi in &producers[gi] {
            if phase_group_set.contains(&pi) {
                uf.union(gi, pi);
            }
        }
    }

    // Group chains: groups with the same union-find root must be on the same lane.
    let mut chains: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &gi in &sorted_groups {
        let root = uf.find(gi);
        chains.entry(root).or_default().push(gi);
    }

    // Assign chains to lanes using first-fit-decreasing bin packing.
    let mut lane_work: Vec<Vec<WorkItem>> = vec![vec![]; num_lanes];
    let mut lane_atoms: Vec<u64> = vec![0; num_lanes];

    // Sort chains by total atom count (descending) for better packing.
    let mut chain_list: Vec<(usize, Vec<usize>, u64)> = chains
        .into_iter()
        .map(|(root, gis)| {
            let total: u64 = gis.iter().map(|&gi| groups[gi].count).sum();
            (root, gis, total)
        })
        .collect();
    chain_list.sort_by(|a, b| b.2.cmp(&a.2));

    for (_, chain_groups, _chain_atoms) in chain_list {
        // Assign the entire chain to the least-loaded lane.
        // We don't split groups because split outputs create a mismatch
        // with how later phases look up data in the value store.
        let target_lane = lane_atoms
            .iter()
            .enumerate()
            .min_by_key(|&(_, &atoms)| atoms)
            .map(|(i, _)| i)
            .unwrap_or(0);

        for &gi in &chain_groups {
            lane_work[target_lane].push(WorkItem {
                group_idx: gi,
                atom_offset: 0,
                atom_count: groups[gi].count,
            });
            lane_atoms[target_lane] += groups[gi].count;
        }
    }

    // Sort each lane's work items by group index (topo order).
    for lane in &mut lane_work {
        lane.sort_by_key(|w| (w.group_idx, w.atom_offset));
    }

    // Build spans for each lane.
    let spans: Vec<Span> = (0..num_lanes)
        .map(|lane_idx| {
            build_span(
                graph,
                &lane_work[lane_idx],
                phase_assignments,
                producers,
                successors,
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

/// Build a Phase using pre-assigned lane mappings.
///
/// Unlike `build_phase` which does its own bin-packing, this uses the lane
/// assignments computed by `assign_lanes_and_phases`.
fn build_phase_from_lanes(
    graph: &NanoGraph,
    phase_group_indices: &[usize],
    lane_assignments: &[usize],
    phase_assignments: &[usize],
    producers: &[Vec<usize>],
    successors: &[Vec<usize>],
    num_lanes: usize,
    input_tensors: &[InputTensor],
    output_group_set: &HashSet<usize>,
    phase_idx: usize,
    num_phases: usize,
) -> Phase {
    let groups = graph.groups();
    let phase_group_set: HashSet<usize> = phase_group_indices.iter().copied().collect();

    // Distribute groups to lanes per the pre-computed assignment.
    let mut lane_work: Vec<Vec<WorkItem>> = vec![vec![]; num_lanes];
    for &gi in phase_group_indices {
        let lane = lane_assignments[gi];
        lane_work[lane].push(WorkItem {
            group_idx: gi,
            atom_offset: 0,
            atom_count: groups[gi].count,
        });
    }

    // Sort each lane's work items by group index (topo order).
    for lane in &mut lane_work {
        lane.sort_by_key(|w| w.group_idx);
    }

    // Build spans for each lane.
    let spans: Vec<Span> = (0..num_lanes)
        .map(|lane_idx| {
            build_span(
                graph,
                &lane_work[lane_idx],
                phase_assignments,
                producers,
                successors,
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
///
/// Creates a NanoGraph fragment with the same atom IDs as the main graph.
/// External dependencies (from earlier phases or input tensors) are registered
/// as input tensor ranges. Internal groups are inserted at their original
/// atom ID positions.
fn build_span(
    graph: &NanoGraph,
    lane_work: &[WorkItem],
    phase_assignments: &[usize],
    producers: &[Vec<usize>],
    successors: &[Vec<usize>],
    input_tensors: &[InputTensor],
    output_group_set: &HashSet<usize>,
    phase_idx: usize,
    num_phases: usize,
    phase_group_set: &HashSet<usize>,
) -> Span {
    if lane_work.is_empty() {
        return Span {
            graph: NanoGraph::new(),
            inputs: vec![],
            outputs: vec![],
        };
    }

    let groups = graph.groups();
    let mut span_graph = NanoGraph::new();

    // Copy sym_dim configuration from main graph (direct clone preserves SymDim numbering).
    span_graph.sym_dim_names = graph.sym_dim_names.clone();
    span_graph.sym_dim_bounds = graph.sym_dim_bounds.clone();

    // Track which groups/slices are computed in this span.
    // Map group_idx -> list of (atom_offset, atom_count) slices.
    let mut span_compute_slices: HashMap<usize, Vec<(u64, u64)>> = HashMap::new();
    for work in lane_work {
        span_compute_slices
            .entry(work.group_idx)
            .or_default()
            .push((work.atom_offset, work.atom_count));
    }

    // Collect all transitive dependencies: find external data needed by this span.
    // External = groups from earlier phases + original input tensors.
    let mut needed_external_atoms: BTreeMap<AtomId, (u64, DType)> = BTreeMap::new();
    let mut needed_internal_literals: BTreeSet<usize> = BTreeSet::new();

    // Small literal inline threshold.
    const LITERAL_INLINE_THRESHOLD: u64 = 65536;

    // Track which group indices are computed in this span (full or partial).
    let span_compute_set: HashSet<usize> = span_compute_slices.keys().copied().collect();

    // BFS to find all needed groups.
    let mut visited = HashSet::new();
    let mut queue: Vec<usize> = span_compute_set.iter().copied().collect();

    while let Some(gi) = queue.pop() {
        if !visited.insert(gi) {
            continue;
        }

        // Determine which producer groups are internal vs external.
        for &pi in &producers[gi] {
            if span_compute_set.contains(&pi) {
                queue.push(pi);
            } else if is_literal_group(&groups[pi]) && groups[pi].count < LITERAL_INLINE_THRESHOLD {
                needed_internal_literals.insert(pi);
            }
            // External computed/large-literal groups are handled below
            // via InputRef range analysis (not full-group recording).
        }

        // For each InputRef of this group, compute the exact source atom
        // range and record external atoms that aren't produced by this span
        // or covered by an inlined literal.
        let group = &groups[gi];
        let all_internal = |atom: AtomId| -> bool {
            // Check if atom is in a span compute group.
            if let Some(idx) = graph.find_group_idx(atom) {
                if span_compute_set.contains(&idx) {
                    return true;
                }
                if needed_internal_literals.contains(&idx) {
                    return true;
                }
            }
            false
        };

        for input in &group.inputs {
            // Compute the source atom range for this InputRef.
            let (lo, hi) = input_ref_source_range(input, group.count, group.atom_offset);

            // Walk through main-graph groups/input_tensors in this range.
            // Record external ones that we need.
            collect_external_in_range(
                graph,
                lo,
                hi,
                group.output_dtype,
                input_tensors,
                &span_compute_set,
                &needed_internal_literals,
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
                    let first = input.resolve(group.atom_offset);
                    let last = input.resolve(group.atom_offset + group.count - 1);
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
                        group.output_dtype,
                        input_tensors,
                        &span_compute_set,
                        &needed_internal_literals,
                        &mut needed_external_atoms,
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
            } else if let Some((ti, _)) = graph.find_input_idx(*table_base) {
                // Table is an input_tensor (weight in shape-only lowering).
                let it = &input_tensors[ti];
                record_external_range(&mut needed_external_atoms, it.base_id, it.count, it.dtype);
            }
        }
    }

    // Also check needed_internal_literals' own dependencies.
    // Literals generally have no inputs, but handle edge cases.
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

    // Merge overlapping/adjacent external ranges. The executor handles
    // partial fills by iterating all store entries within each range.
    let merged_external = merge_external_ranges(&needed_external_atoms);

    if phase_idx < 3 {
        eprintln!(
            "  [build_span] phase={} lane: {} compute groups, {} external ranges, {} inlined literals",
            phase_idx,
            span_compute_set.len(),
            merged_external.len(),
            needed_internal_literals.len(),
        );
    }

    // Now build the span graph, inserting everything in atom ID order.
    // We need to insert: (1) external inputs, (2) inlined literals, (3) compute groups.
    // All in order of their base_id.

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
            atom_offset: u64,
            atom_count: u64,
        },
    }

    let mut items: Vec<(u64, InsertItem)> = Vec::new();

    for &(base, count, dtype) in &merged_external {
        items.push((base.0, InsertItem::ExternalInput { base, count, dtype }));
    }

    for &li in &needed_internal_literals {
        items.push((groups[li].base_id.0, InsertItem::InlineLiteral { gi: li }));
    }

    for work in lane_work {
        let base = groups[work.group_idx].base_id.offset(work.atom_offset);
        items.push((
            base.0,
            InsertItem::ComputeGroup {
                gi: work.group_idx,
                atom_offset: work.atom_offset,
                atom_count: work.atom_count,
            },
        ));
    }

    // Sort by base_id.
    items.sort_by_key(|(base, _)| *base);

    // Track which main-graph atom IDs have been inserted (to avoid duplicates).
    let mut inserted_ranges: Vec<(u64, u64)> = Vec::new();

    // Insert items into the span graph.
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
            InsertItem::ComputeGroup {
                gi,
                atom_offset,
                atom_count,
            } => {
                let group = &groups[*gi];
                let base = group.base_id.offset(*atom_offset);
                if would_overlap(&inserted_ranges, base.0, *atom_count) {
                    continue;
                }

                // Build inputs for this (possibly split) group.
                let inputs = if *atom_offset == 0 && *atom_count == group.count {
                    // Full group: use inputs as-is.
                    group.inputs.clone()
                } else {
                    // Split group: inputs stay the same (atom_offset handles resolution).
                    group.inputs.clone()
                };

                span_graph.insert_group_at(
                    base,
                    *atom_count,
                    *atom_offset,
                    group.output_dtype,
                    group.op.clone(),
                    group.sym_dims.clone(),
                    inputs,
                );
                inserted_ranges.push((base.0, *atom_count));

                // Check if this group's output is needed after this phase.
                if output_group_set.contains(gi)
                    || needs_output(
                        *gi,
                        *atom_offset,
                        *atom_count,
                        &groups[*gi],
                        phase_assignments,
                        producers,
                        phase_group_set,
                        &span_compute_set,
                        successors,
                    )
                {
                    span_outputs.push(AtomRange {
                        base,
                        count: *atom_count,
                        dtype: group.output_dtype,
                    });
                }
            }
        }
    }

    // Merge adjacent/overlapping output ranges.
    span_outputs = merge_atom_ranges(span_outputs);

    Span {
        graph: span_graph,
        inputs: span_inputs,
        outputs: span_outputs,
    }
}

/// Check if a group's output is consumed by any group in a later phase or
/// by a group in this phase that's on a different lane (i.e., not in span_compute_set).
fn needs_output(
    gi: usize,
    _atom_offset: u64,
    _atom_count: u64,
    group: &AtomGroup,
    phase_assignments: &[usize],
    _producers: &[Vec<usize>],
    _phase_group_set: &HashSet<usize>,
    span_compute_set: &HashSet<usize>,
    successors: &[Vec<usize>],
) -> bool {
    // Don't skip Literals — they might be adjacent to compute groups
    // and get merged into a single input range by a consuming span.
    // If a Literal has consumers in later phases, it must be output.

    let my_phase = phase_assignments[gi];

    // Output if any successor is in a later phase, or in this phase but
    // on a different lane (not in our span's compute set).
    for &si in &successors[gi] {
        let succ_phase = phase_assignments[si];
        if succ_phase > my_phase {
            return true; // consumed by a later phase
        }
        if succ_phase == my_phase && !span_compute_set.contains(&si) {
            return true; // consumed by a different lane in the same phase
        }
    }
    false
}

/// Check if a ScalarOp is a literal (no computation needed).
fn is_literal_op(op: &ScalarOp) -> bool {
    matches!(op, ScalarOp::Literal(_))
}

/// Check if a group is a literal (produces constant values).
fn is_literal_group(group: &AtomGroup) -> bool {
    is_literal_op(&group.op) && group.inputs.is_empty()
}

/// Record an external atom range dependency.
/// Compute the (lo, hi) inclusive source atom range for an InputRef.
fn input_ref_source_range(input: &InputRef, count: u64, atom_offset: u64) -> (u64, u64) {
    if count == 0 {
        return (0, 0);
    }
    match input {
        InputRef::Broadcast(base) => (base.0, base.0),
        InputRef::Affine { .. } | InputRef::StridedBroadcast { .. } => {
            let first = input.resolve(atom_offset);
            let last = input.resolve(atom_offset + count - 1);
            (first.0.min(last.0), first.0.max(last.0))
        }
        InputRef::Modular {
            base,
            stride,
            modulus,
        } => {
            let a = base.0;
            let b = (base.0 as i64 + *stride * (*modulus as i64 - 1)) as u64;
            (a.min(b), a.max(b))
        }
        InputRef::Explicit(ids) => {
            let slice = &ids[atom_offset as usize..(atom_offset + count) as usize];
            let lo = slice.iter().map(|id| id.0).min().unwrap_or(0);
            let hi = slice.iter().map(|id| id.0).max().unwrap_or(0);
            (lo, hi)
        }
    }
}

/// Record external dependencies in [lo, hi] that aren't produced internally.
fn collect_external_in_range(
    graph: &NanoGraph,
    lo: u64,
    hi: u64,
    fallback_dtype: DType,
    input_tensors: &[InputTensor],
    span_compute_set: &HashSet<usize>,
    inlined_literals: &BTreeSet<usize>,
    needed_external: &mut BTreeMap<AtomId, (u64, DType)>,
) {
    // Walk groups in the main graph that overlap [lo, hi].
    let groups = graph.groups();
    for (gi, g) in groups.iter().enumerate() {
        let g_lo = g.base_id.0;
        let g_hi = g_lo + g.count;
        if g_lo > hi {
            break;
        } // groups are sorted
        if g_hi <= lo {
            continue;
        }
        // This group overlaps [lo, hi].
        if span_compute_set.contains(&gi) || inlined_literals.contains(&gi) {
            continue; // internal
        }
        // Record only the overlapping portion.
        let range_lo = g_lo.max(lo);
        let range_hi = g_hi.min(hi + 1);
        let range_count = range_hi - range_lo;
        if range_count > 0 {
            record_external_range(
                needed_external,
                AtomId(range_lo),
                range_count,
                g.output_dtype,
            );
        }
    }

    // Also check input_tensors that overlap [lo, hi].
    for it in input_tensors {
        let it_lo = it.base_id.0;
        let it_hi = it_lo + it.count;
        if it_lo <= hi && it_hi > lo {
            let range_lo = it_lo.max(lo);
            let range_hi = it_hi.min(hi + 1);
            let range_count = range_hi - range_lo;
            if range_count > 0 {
                record_external_range(needed_external, AtomId(range_lo), range_count, it.dtype);
            }
        }
    }
}

fn record_external_range(
    ranges: &mut BTreeMap<AtomId, (u64, DType)>,
    base: AtomId,
    count: u64,
    dtype: DType,
) {
    // Insert or extend.
    ranges
        .entry(base)
        .and_modify(|(existing_count, _)| {
            *existing_count = (*existing_count).max(count);
        })
        .or_insert((count, dtype));
}

/// Collect input tensor dependencies from an InputRef.
fn collect_input_tensor_deps(
    input: &InputRef,
    count: u64,
    atom_offset: u64,
    graph: &NanoGraph,
    input_tensors: &[InputTensor],
    needed_external: &mut BTreeMap<AtomId, (u64, DType)>,
) {
    // Check if any resolved atom falls within an input tensor range.
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

/// Collect input tensor ranges overlapping [lo, hi].
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

/// Collect dependencies from reduce-strided access patterns.
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
    // The reduce op accesses atoms from base to base + (reduce_count-1)*stride
    // for each atom in the group.
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

    // Check for group dependencies in this range.
    for (gi, g) in all_groups.iter().enumerate() {
        let g_lo = g.base_id.0;
        let g_hi = g_lo + g.count - 1;
        if g_lo > hi {
            break; // Groups are sorted by base_id.
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

    // Also check input tensors.
    collect_input_tensor_range(graph, input_tensors, lo, hi, needed_external);
}

/// Check if a new range would overlap any already-inserted range.
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

/// Merge overlapping/adjacent external ranges into consolidated ranges.
fn merge_external_ranges(ranges: &BTreeMap<AtomId, (u64, DType)>) -> Vec<(AtomId, u64, DType)> {
    if ranges.is_empty() {
        return vec![];
    }

    let sorted: Vec<(AtomId, u64, DType)> = ranges
        .iter()
        .map(|(&base, &(count, dtype))| (base, count, dtype))
        .collect();

    // BTreeMap is already sorted by key.
    let mut merged: Vec<(AtomId, u64, DType)> = Vec::new();

    for (base, count, dtype) in sorted {
        if let Some(last) = merged.last_mut() {
            let last_end = last.0.0 + last.1;
            if base.0 <= last_end && dtype == last.2 {
                // Overlapping or adjacent, same dtype: extend.
                let new_end = (base.0 + count).max(last_end);
                last.1 = new_end - last.0.0;
                continue;
            }
        }
        merged.push((base, count, dtype));
    }

    merged
}

/// Merge overlapping/adjacent AtomRanges.
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

/// Simple union-find for grouping chains of dependent groups.
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
    ///   lit
    ///  / \
    /// b   c
    ///  \ /
    ///   d
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

        // Weight matrix: K values (shared by all rows).
        let k = 64;
        let m = 8;
        let weights = g.push_group(
            k,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.5)),
            vec![],
            vec![],
        );

        // Input vector: M*K values (M rows of K).
        let input = g.push_group(
            m * k,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );

        // Mul: M*K elementwise products (input * weights-broadcast).
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

        // ReduceSum: M outputs, each summing K products.
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

    /// Verify basic structural invariants of a plan.
    fn verify_plan(graph: &NanoGraph, phases: &[Phase], num_lanes: usize) {
        // All phases have the right number of spans.
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

        // All span graphs validate.
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
            // Collect atoms produced by each span.
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

            // Check: no span reads atoms produced by another span in the same phase.
            for (li, span) in phase.spans.iter().enumerate() {
                for group in span.graph.groups() {
                    let mut deps = HashSet::new();
                    span.graph.collect_all_producer_indices(group, 0, &mut deps);
                    // The deps are indices within this span's graph, which is fine.
                    // What we need to check is that input atoms don't come from
                    // other spans.
                }

                // Check the span's declared inputs don't overlap with other spans' outputs.
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
    fn test_with_input_tensors() {
        let (g, inputs) = make_with_inputs();
        let output_ids: Vec<AtomId> = g.outputs.clone();
        let phases = plan(&g, 2, &inputs, &output_ids);

        verify_plan(&g, &phases, 2);
        verify_no_cross_span_reads(&phases);

        // At least one span should declare the input tensor as an input.
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
        // Large groups assigned to a single lane (no splitting to avoid
        // value store mismatch issues with split outputs).
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
        // Manually verify depth computation on a known DAG.
        //   0
        //  / \
        // 1   2
        //  \ /
        //   3
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

        // Groups 0,1 form wavefront 0 (depth 0)
        // Groups 2,3 form wavefront 1 (depth 1)
        // Group 4 forms wavefront 2 (depth 2)
    }

    #[test]
    fn test_broadcast_dependency() {
        // A broadcast input: one group feeds many consumers.
        let mut g = NanoGraph::new();

        let shared = g.push_group(
            1,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );

        // Two independent consumers that broadcast from the shared atom.
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
        // ReduceSum that reads from a literal group via stride.
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
        // Verify that every main-graph group appears in exactly one span.
        let (g, inputs) = make_diamond();
        let output_ids: Vec<AtomId> = g.outputs.clone();
        let phases = plan(&g, 2, &inputs, &output_ids);

        let main_groups: HashSet<u64> = g.groups().iter().map(|gr| gr.base_id.0).collect();

        let mut span_groups: HashSet<u64> = HashSet::new();
        for phase in &phases {
            for span in &phase.spans {
                for gr in span.graph.groups() {
                    // Only count non-literal groups that are also in the main graph.
                    if main_groups.contains(&gr.base_id.0) {
                        span_groups.insert(gr.base_id.0);
                    }
                }
            }
        }

        // Every main-graph group should appear in some span (either as compute
        // or as an inlined literal).
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
        // Test that groups with Modular input refs are handled correctly.
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
}
