#![allow(
    clippy::all,
    dead_code,
    unreachable_code,
    unreachable_patterns,
    unused_imports,
    unused_variables
)]

//! Dependency Chain + Barrier Placement Partitioner (attempt J)
//!
//! Algorithm:
//! 1. Build group-level dependency DAG. Identify which groups are splittable
//!    (elementwise, matmul mul/reduce, indirect loads) vs unsplittable
//!    (literals → duplicate, reductions reading split data → keep whole).
//! 2. Compute "compatible split" chains: consecutive ops in the DAG that
//!    can all be split identically across lanes with no barrier needed.
//!    A chain breaks at "fan-in" points where a group reads from atoms
//!    produced by multiple incompatible split regions.
//! 3. Groups between consecutive barriers form one Phase. Within each phase,
//!    split every splittable group across lanes (lane k gets atoms
//!    [k*chunk..(k+1)*chunk)). Literals are duplicated into every lane.
//!    Small unsplittable groups are assigned to one lane (or duplicated).
//! 4. Build span NanoGraphs with correct atom_offset, inputs, and outputs.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use crate::graph::GlobalId;
use crate::nano_graph::pattern::InputTensor;
use crate::nano_graph::{AtomGroup, AtomId, AtomRange, InputRef, NanoGraph, ScalarOp, SymDim};
use crate::numeric_dtype::NumericDType;

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
            spans: (0..num_lanes).map(|_| empty_span()).collect(),
        }];
    }

    // Step 1: Build dependency DAG (group-level).
    let (producers, successors) = build_dependency_dag(graph);

    // Step 2: Classify each group.
    let classifications = classify_groups(graph, &producers);

    // Step 3: Assign groups to phases using barrier placement.
    // A barrier is needed when a group reads data that was split across
    // lanes in a previous phase — i.e., it's a cross-lane fan-in.
    let phase_assignments = assign_phases(graph, &producers, &classifications, num_lanes);

    // Step 4: Collect groups per phase.
    let num_phases = phase_assignments.iter().copied().max().unwrap_or(0) + 1;
    let mut phase_groups: Vec<Vec<usize>> = vec![vec![]; num_phases];
    for (gi, &phase) in phase_assignments.iter().enumerate() {
        phase_groups[phase].push(gi);
    }

    // Step 5: Identify output groups.
    let output_group_set = identify_output_groups(graph, output_atom_ids);

    // Step 6: Build phases with split spans.
    let mut phases = Vec::with_capacity(num_phases);
    for phase_idx in 0..num_phases {
        let phase = build_phase(
            graph,
            &phase_groups[phase_idx],
            &classifications,
            &producers,
            &successors,
            &phase_assignments,
            &output_group_set,
            num_lanes,
            input_tensors,
            phase_idx,
            num_phases,
        );
        phases.push(phase);
    }

    phases
}

// ─── Group classification ──────────────────────────────────────────────────

/// How a group should be handled during partitioning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GroupKind {
    /// Splittable across lanes (elementwise, matmul components, indirect loads).
    Split,
    /// Literal: duplicate into every lane that needs it.
    Literal,
    /// Reduce that reads from split data: must stay whole, needs barrier before it.
    /// After it completes, its (small) output can be duplicated to downstream lanes.
    WholeReduce,
    /// Too small to split (count < num_lanes). Assign to one lane or duplicate.
    TooSmall,
}

fn classify_groups(graph: &NanoGraph, producers: &[Vec<usize>]) -> Vec<GroupKind> {
    let groups = graph.groups();
    groups
        .iter()
        .enumerate()
        .map(|(gi, g)| classify_one(g, groups, &producers[gi]))
        .collect()
}

fn classify_one(group: &AtomGroup, all_groups: &[AtomGroup], my_producers: &[usize]) -> GroupKind {
    // Literals: duplicate.
    if matches!(group.op, ScalarOp::Literal(_)) {
        return GroupKind::Literal;
    }

    // Reduces: check if this is a "real" reduction (reduce_count > 1).
    // ReduceSum with reduce_count=1 is effectively identity/passthrough.
    if let ScalarOp::Reduce { reduce_count, .. } = &group.op {
        if *reduce_count > 1 {
            // This is a true reduction. However, the output atoms of the
            // reduce are independent of each other — each one accumulates
            // its own range. So we CAN split the reduce across lanes if
            // the output count is large enough (each lane gets a subset
            // of output atoms, each performing its own independent reduction).
            if group.count >= 2 {
                return GroupKind::Split;
            }
            return GroupKind::WholeReduce;
        }
    }

    // Too small to bother splitting.
    // We use a threshold of 1 (single atom) — even count=2 can be split
    // across 2 of N lanes. The "too small" category is really just for
    // singleton groups.
    if group.count <= 1 {
        return GroupKind::TooSmall;
    }

    // Everything else is splittable: Binary, Unary, Identity, Select,
    // IndirectLoad, trivial Reduce (reduce_count=1).
    GroupKind::Split
}

// ─── Dependency DAG ────────────────────────────────────────────────────────

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

// ─── Phase assignment (barrier placement) ──────────────────────────────────

/// Assign each group to a phase. The key insight: barriers are only needed
/// when a group reads atoms produced by multiple lanes in the previous phase
/// AND those atoms cannot all be in a single lane's span.
///
/// For a chain of splittable groups that all split identically (same count,
/// or same split axis), NO barrier is needed — each lane handles its slice
/// of the entire chain.
///
/// A barrier IS needed when:
/// - A reduce reads from data that was split across lanes (fan-in).
/// - A group has explicit inputs that span multiple groups with different
///   split assignments.
///
/// For this implementation, we use a simpler but effective heuristic:
/// groups in the same "split-compatible chain" share a phase. A chain
/// breaks when a group depends on a WholeReduce or when dependencies
/// come from groups in multiple different chains.
fn assign_phases(
    graph: &NanoGraph,
    producers: &[Vec<usize>],
    classifications: &[GroupKind],
    num_lanes: usize,
) -> Vec<usize> {
    let groups = graph.groups();
    let n = groups.len();
    let mut phase_of = vec![0usize; n];

    // Process in topological order (groups are already in topo order).
    for gi in 0..n {
        let group = &groups[gi];
        let kind = classifications[gi];

        if producers[gi].is_empty() {
            // Root group (no dependencies): phase 0.
            phase_of[gi] = 0;
            continue;
        }

        // Determine the maximum phase of our producers.
        let max_producer_phase = producers[gi]
            .iter()
            .map(|&pi| phase_of[pi])
            .max()
            .unwrap_or(0);

        // Check if we need a new phase (barrier) after our producers.
        let needs_barrier = needs_barrier_before(
            gi,
            group,
            kind,
            &producers[gi],
            classifications,
            groups,
            num_lanes,
        );

        if needs_barrier {
            phase_of[gi] = max_producer_phase + 1;
        } else {
            // Same phase as our latest producer — no barrier needed.
            phase_of[gi] = max_producer_phase;
        }
    }

    phase_of
}

/// Determine if group `gi` needs a barrier (new phase) before it can execute.
///
/// A barrier is needed when:
/// 1. The group is a WholeReduce — it needs all lanes' data from the
///    previous computation.
/// 2. The group reads from a WholeReduce or TooSmall that was placed in
///    a single lane — other lanes need a barrier to access that data.
/// 3. The group has Explicit inputs that prevent clean splitting.
fn needs_barrier_before(
    gi: usize,
    group: &AtomGroup,
    kind: GroupKind,
    my_producers: &[usize],
    classifications: &[GroupKind],
    groups: &[AtomGroup],
    num_lanes: usize,
) -> bool {
    // Literals and roots never need barriers.
    if kind == GroupKind::Literal || my_producers.is_empty() {
        return false;
    }

    // Check if any producer is a WholeReduce or TooSmall that was assigned
    // to one lane — downstream groups on other lanes need the barrier to
    // get that data.
    for &pi in my_producers {
        let pk = classifications[pi];
        match pk {
            GroupKind::WholeReduce | GroupKind::TooSmall => {
                // Producer was on one lane; we need a barrier so the
                // data is available to all lanes.
                return true;
            }
            _ => {}
        }
    }

    // If this group itself is a WholeReduce, it needs a barrier to
    // collect data from all lanes.
    if kind == GroupKind::WholeReduce {
        // Only if producers were split across lanes.
        let has_split_producer = my_producers
            .iter()
            .any(|&pi| classifications[pi] == GroupKind::Split && groups[pi].count > 1);
        if has_split_producer {
            return true;
        }
    }

    // If this is a split group reading from split producers, check compatibility.
    // Two split groups are compatible if their data flows within each lane's slice.
    // This is the case for elementwise chains where the output of one feeds
    // directly into the next with stride=1 and same count.
    //
    // For simplicity: if all split producers have the same count (or are literals
    // that get duplicated), no barrier is needed. If counts differ, we need to
    // check if the split is compatible.
    if kind == GroupKind::Split {
        // Check each input ref to see if it reads from a range that would
        // span multiple lanes after splitting.
        for input in &group.inputs {
            if input_crosses_lane_boundary(input, group, groups, classifications, num_lanes) {
                return true;
            }
        }
    }

    false
}

/// Check if an InputRef would cause a group to read data from multiple lanes
/// after the source group is split.
///
/// For Affine stride=1 reading from a same-count group: no crossing.
/// For Broadcast: no crossing (all lanes read the same atom).
/// For StridedBroadcast with repeat > 1: the repeated reads stay within
///   a contiguous block, so splitting the consumer by rows is fine.
/// For Modular: reads wrap around, so they hit all lanes → crossing.
/// For Explicit: conservative — assume crossing.
fn input_crosses_lane_boundary(
    input: &InputRef,
    consumer: &AtomGroup,
    groups: &[AtomGroup],
    classifications: &[GroupKind],
    num_lanes: usize,
) -> bool {
    match input {
        InputRef::Broadcast(_) => {
            // Broadcast: every atom reads the same source. No crossing.
            false
        }
        InputRef::Strided {
            base,
            dim_strides,
            dim_shape,
        } => {
            let nd = dim_strides.len();
            // Determine the effective "affine stride" (innermost stride for 1D,
            // or the first stride for 2D patterns like StridedBroadcast/Modular).
            //
            // 1D affine: dim_strides=[s] → stride=s
            // 2D StridedBroadcast: dim_strides=[s, 0] → handle separately below
            // 2D Modular: dim_strides=[0, s] → handle separately below
            // 2D General: dim_strides=[so, si] → use like affine with si

            // StridedBroadcast: dim_strides[1]==0
            if nd == 2 && dim_strides[1] == 0 {
                // StridedBroadcast: atom i reads base + stride * (i / repeat).
                // When we split the consumer, lane k gets atoms [k*C..(k+1)*C).
                // Those atoms read source atoms at base + stride * (k*C/repeat)
                // through base + stride * ((k+1)*C-1)/repeat.
                // As long as the source range is contiguous per lane, this is fine.
                return false;
            }

            // Modular: dim_strides[0]==0
            if nd == 2 && dim_strides[0] == 0 {
                // Modular: every lane reads from the full modular range.
                // The source is typically a small bias/weight that gets duplicated
                // (classified as Literal) or provided as input. No barrier needed
                // because the source is either duplicated or available as external input.
                return false;
            }

            // Affine or general 2D with both strides non-zero: use innermost stride
            let stride = if nd == 1 {
                dim_strides[0]
            } else {
                dim_strides[1]
            };
            if stride == 0 {
                // Degenerate broadcast-like: all atoms read the same source.
                return false;
            }
            // For Affine with any stride: consumer atom i reads
            // source at base + stride * (i + atom_offset).
            // When we split the consumer into chunks of C = count/num_lanes,
            // lane k reads sources from base + stride*(k*C + atom_offset)
            // to base + stride*((k+1)*C - 1 + atom_offset).
            // The WIDTH of source atoms covered by one lane is |stride| * C.
            //
            // If the source group is also being split across the same lanes,
            // each lane gets source_count/num_lanes source atoms.
            // We need |stride| * C <= source_count / num_lanes, AND the
            // lane's source range must fall entirely within one lane's slice.
            //
            // For stride=1: C source atoms per lane, and source split into
            // source_count/num_lanes per lane. C <= source_count/num_lanes
            // always holds when source_count >= consumer count. Compatible.
            //
            // For stride > 1 (e.g., reduce input stride=128):
            // Lane k reads sources at indices k*C*stride through
            // ((k+1)*C-1)*stride from base. The total source range
            // per lane spans stride*C atoms. The source is split into
            // source_count/num_lanes atoms per lane.
            //
            // Key insight: if the source was split such that lane k's
            // source slice covers exactly stride*C atoms (which happens
            // when source_count = stride * consumer_count), then each
            // lane's reduce reads only from its own lane's source. This
            // is the matmul pattern: Mul has count=M*K, Reduce has
            // count=M, stride=K, so source_count = K * M = stride * count.
            //
            // Check: find the source group and verify the split is compatible.
            let abs_stride = stride.unsigned_abs();
            let consumer_chunk = consumer.count / num_lanes.max(1) as u64;
            if consumer_chunk == 0 {
                return false;
            }
            // Source atoms per lane = abs_stride * consumer_chunk.
            let source_atoms_per_lane = abs_stride.saturating_mul(consumer_chunk);

            // Find the source group to check its count.
            if let Some((source_group, _)) = groups
                .iter()
                .enumerate()
                .find(|(_, g)| g.contains(AtomId(base.0)))
                .map(|(_, g)| (g, 0))
            {
                let source_kind = if matches!(source_group.op, ScalarOp::Literal(_)) {
                    GroupKind::Literal
                } else if source_group.count <= 1 {
                    GroupKind::TooSmall
                } else {
                    GroupKind::Split
                };

                match source_kind {
                    GroupKind::Literal => {
                        // Source is a literal → duplicated, no crossing.
                        return false;
                    }
                    GroupKind::TooSmall | GroupKind::WholeReduce => {
                        // Source is on one lane → handled by barrier from
                        // that classification, not here.
                        return false;
                    }
                    GroupKind::Split => {
                        // Source is split across lanes. Check if each lane's
                        // consumer range fits within its source lane's slice.
                        let source_chunk = source_group.count / num_lanes.max(1) as u64;
                        if source_chunk == 0 {
                            return true; // Can't split source enough.
                        }
                        // Compatible if source_atoms_per_lane <= source_chunk.
                        return source_atoms_per_lane > source_chunk;
                    }
                }
            }
            // Source not found as a group (might be an input tensor) → no crossing.
            false
        }
        InputRef::Explicit(_ids) => {
            // Explicit: arbitrary pattern. Conservative — assume crossing
            // if the consumer is large enough to split.
            consumer.count > num_lanes as u64
        }
    }
}

// ─── Output group identification ───────────────────────────────────────────

fn identify_output_groups(graph: &NanoGraph, output_atom_ids: &[AtomId]) -> HashSet<usize> {
    let mut out = HashSet::new();
    for &aid in output_atom_ids {
        if let Some(gi) = graph.find_group_idx(aid) {
            out.insert(gi);
        }
    }
    // Also include all groups that are in the graph's outputs list.
    for &aid in &graph.outputs {
        if let Some(gi) = graph.find_group_idx(aid) {
            out.insert(gi);
        }
    }
    out
}

// ─── Phase building ────────────────────────────────────────────────────────

fn build_phase(
    graph: &NanoGraph,
    phase_group_indices: &[usize],
    classifications: &[GroupKind],
    producers: &[Vec<usize>],
    successors: &[Vec<usize>],
    phase_assignments: &[usize],
    output_group_set: &HashSet<usize>,
    num_lanes: usize,
    input_tensors: &[InputTensor],
    phase_idx: usize,
    num_phases: usize,
) -> Phase {
    let groups = graph.groups();

    // Initialize per-lane state.
    let mut lane_graphs: Vec<NanoGraph> = (0..num_lanes).map(|_| NanoGraph::new()).collect();
    let mut lane_inputs: Vec<Vec<AtomRange>> = vec![vec![]; num_lanes];
    let mut lane_outputs: Vec<Vec<AtomRange>> = vec![vec![]; num_lanes];

    // Track which atom ranges have been declared as inputs in each lane.
    let mut lane_input_atoms: Vec<HashSet<u64>> = vec![HashSet::new(); num_lanes];
    // Track which groups have been placed in each lane.
    let mut lane_group_set: Vec<HashSet<usize>> = vec![HashSet::new(); num_lanes];

    // Collect the set of group indices in this phase for quick lookup.
    let phase_gi_set: HashSet<usize> = phase_group_indices.iter().copied().collect();

    // Process groups in topological order (they're already sorted by base_id).
    let mut sorted_gis: Vec<usize> = phase_group_indices.to_vec();
    sorted_gis.sort();

    for &gi in &sorted_gis {
        let group = &groups[gi];
        let kind = classifications[gi];

        match kind {
            GroupKind::Split => {
                // Split this group across lanes.
                split_group_across_lanes(
                    graph,
                    gi,
                    group,
                    num_lanes,
                    &mut lane_graphs,
                    &mut lane_inputs,
                    &mut lane_outputs,
                    &mut lane_input_atoms,
                    &mut lane_group_set,
                    &phase_gi_set,
                    classifications,
                    producers,
                    successors,
                    phase_assignments,
                    output_group_set,
                    input_tensors,
                    phase_idx,
                    num_phases,
                );
            }
            GroupKind::Literal => {
                // Duplicate literal into every lane that will need it.
                // For simplicity, put it in all lanes. The cost is minimal
                // since literals are zero-input constant producers.
                duplicate_group_to_all_lanes(
                    graph,
                    gi,
                    group,
                    num_lanes,
                    &mut lane_graphs,
                    &mut lane_group_set,
                );
            }
            GroupKind::WholeReduce | GroupKind::TooSmall => {
                // Assign to lane 0 (or least-loaded lane).
                // These are small groups that don't benefit from splitting.
                let target_lane = pick_least_loaded_lane(&lane_graphs, num_lanes);
                place_whole_group_on_lane(
                    graph,
                    gi,
                    group,
                    target_lane,
                    &mut lane_graphs,
                    &mut lane_inputs,
                    &mut lane_outputs,
                    &mut lane_input_atoms,
                    &mut lane_group_set,
                    &phase_gi_set,
                    classifications,
                    producers,
                    successors,
                    phase_assignments,
                    output_group_set,
                    input_tensors,
                    phase_idx,
                    num_phases,
                );
            }
        }
    }

    // Copy sym_dim metadata to each lane's graph.
    for lg in &mut lane_graphs {
        lg.sym_dim_names = graph.sym_dim_names.clone();
        lg.sym_dim_bounds = graph.sym_dim_bounds.clone();
    }

    Phase {
        spans: lane_graphs
            .into_iter()
            .zip(lane_inputs.into_iter())
            .zip(lane_outputs.into_iter())
            .map(|((g, inp), out)| Span {
                graph: g,
                inputs: inp,
                outputs: out,
            })
            .collect(),
    }
}

fn pick_least_loaded_lane(lane_graphs: &[NanoGraph], num_lanes: usize) -> usize {
    let mut best_lane = 0;
    let mut best_atoms = u64::MAX;
    for lane in 0..num_lanes {
        let atoms: u64 = lane_graphs[lane].groups().iter().map(|g| g.count).sum();
        if atoms < best_atoms {
            best_atoms = atoms;
            best_lane = lane;
        }
    }
    best_lane
}

/// Split a group across all lanes. Lane k gets atoms [k*chunk..(k+1)*chunk).
fn split_group_across_lanes(
    graph: &NanoGraph,
    gi: usize,
    group: &AtomGroup,
    num_lanes: usize,
    lane_graphs: &mut [NanoGraph],
    lane_inputs: &mut [Vec<AtomRange>],
    lane_outputs: &mut [Vec<AtomRange>],
    lane_input_atoms: &mut [HashSet<u64>],
    lane_group_set: &mut [HashSet<usize>],
    phase_gi_set: &HashSet<usize>,
    classifications: &[GroupKind],
    producers: &[Vec<usize>],
    successors: &[Vec<usize>],
    phase_assignments: &[usize],
    output_group_set: &HashSet<usize>,
    input_tensors: &[InputTensor],
    phase_idx: usize,
    num_phases: usize,
) {
    let groups = graph.groups();
    let base = group.base_id.0;
    let total = group.count;

    // How many lanes actually get work.
    let effective_lanes = (num_lanes as u64).min(total) as usize;
    let chunk = total / effective_lanes as u64;

    for lane in 0..num_lanes {
        if lane >= effective_lanes {
            // This lane gets nothing from this group.
            continue;
        }

        let start = lane as u64 * chunk;
        let count = if lane == effective_lanes - 1 {
            total - start
        } else {
            chunk
        };

        if count == 0 {
            continue;
        }

        let fragment_base = AtomId(base + start);

        // Ensure dependencies are available as inputs in this lane.
        ensure_inputs_for_group(
            graph,
            gi,
            group,
            lane,
            start, // atom_offset for this fragment
            count,
            lane_graphs,
            lane_inputs,
            lane_input_atoms,
            lane_group_set,
            phase_gi_set,
            classifications,
            producers,
            input_tensors,
        );

        // Insert the fragment.
        lane_graphs[lane].insert_group_at(
            fragment_base,
            count,
            start + group.atom_offset, // atom_offset = fragment start + original offset
            group.output_dtype,
            group.op.clone(),
            group.sym_dims.clone(),
            group.inputs.clone(),
        );

        lane_group_set[lane].insert(gi);

        // Determine if this fragment needs to be in outputs.
        if needs_output(
            gi,
            group,
            successors,
            phase_assignments,
            output_group_set,
            phase_idx,
            num_phases,
        ) {
            lane_outputs[lane].push(AtomRange {
                base: fragment_base,
                count,
                dtype: group.output_dtype,
            });
        }
    }
}

/// Duplicate a group into all lanes (used for literals).
fn duplicate_group_to_all_lanes(
    graph: &NanoGraph,
    gi: usize,
    group: &AtomGroup,
    num_lanes: usize,
    lane_graphs: &mut [NanoGraph],
    lane_group_set: &mut [HashSet<usize>],
) {
    for lane in 0..num_lanes {
        lane_graphs[lane].insert_group_at(
            group.base_id,
            group.count,
            group.atom_offset,
            group.output_dtype,
            group.op.clone(),
            group.sym_dims.clone(),
            group.inputs.clone(),
        );
        lane_group_set[lane].insert(gi);
    }
}

/// Place a whole group on a single lane.
fn place_whole_group_on_lane(
    graph: &NanoGraph,
    gi: usize,
    group: &AtomGroup,
    lane: usize,
    lane_graphs: &mut [NanoGraph],
    lane_inputs: &mut [Vec<AtomRange>],
    lane_outputs: &mut [Vec<AtomRange>],
    lane_input_atoms: &mut [HashSet<u64>],
    lane_group_set: &mut [HashSet<usize>],
    phase_gi_set: &HashSet<usize>,
    classifications: &[GroupKind],
    producers: &[Vec<usize>],
    successors: &[Vec<usize>],
    phase_assignments: &[usize],
    output_group_set: &HashSet<usize>,
    input_tensors: &[InputTensor],
    phase_idx: usize,
    num_phases: usize,
) {
    let groups = graph.groups();

    // Ensure dependencies are available.
    ensure_inputs_for_group(
        graph,
        gi,
        group,
        lane,
        0,
        group.count,
        lane_graphs,
        lane_inputs,
        lane_input_atoms,
        lane_group_set,
        phase_gi_set,
        classifications,
        producers,
        input_tensors,
    );

    // Insert the group.
    lane_graphs[lane].insert_group_at(
        group.base_id,
        group.count,
        group.atom_offset,
        group.output_dtype,
        group.op.clone(),
        group.sym_dims.clone(),
        group.inputs.clone(),
    );

    lane_group_set[lane].insert(gi);

    // Output if needed.
    if needs_output(
        gi,
        group,
        successors,
        phase_assignments,
        output_group_set,
        phase_idx,
        num_phases,
    ) {
        lane_outputs[lane].push(AtomRange {
            base: group.base_id,
            count: group.count,
            dtype: group.output_dtype,
        });
    }
}

/// Ensure that all atoms referenced by a group's inputs are available
/// in the given lane — either as a group already placed there, or as
/// a declared external input.
fn ensure_inputs_for_group(
    graph: &NanoGraph,
    gi: usize,
    group: &AtomGroup,
    lane: usize,
    fragment_offset: u64, // where this fragment starts within the group
    fragment_count: u64,
    lane_graphs: &mut [NanoGraph],
    lane_inputs: &mut [Vec<AtomRange>],
    lane_input_atoms: &mut [HashSet<u64>],
    lane_group_set: &mut [HashSet<usize>],
    phase_gi_set: &HashSet<usize>,
    classifications: &[GroupKind],
    producers: &[Vec<usize>],
    input_tensors: &[InputTensor],
) {
    let groups = graph.groups();
    let effective_offset = fragment_offset + group.atom_offset;

    // For each input ref, find the source atom range and ensure it's available.
    for input in &group.inputs {
        ensure_input_ref(
            graph,
            input,
            effective_offset,
            fragment_count,
            lane,
            lane_graphs,
            lane_inputs,
            lane_input_atoms,
            lane_group_set,
            phase_gi_set,
            input_tensors,
        );
    }

    // For Reduce ops, also ensure the strided range is available.
    if let ScalarOp::Reduce {
        reduce_count,
        reduce_stride,
        ..
    } = &group.op
    {
        if *reduce_count > 1 && *reduce_stride != 0 {
            for input in &group.inputs {
                ensure_reduce_stride_range(
                    graph,
                    input,
                    effective_offset,
                    fragment_count,
                    *reduce_count,
                    *reduce_stride,
                    lane,
                    lane_graphs,
                    lane_inputs,
                    lane_input_atoms,
                    lane_group_set,
                    phase_gi_set,
                    input_tensors,
                );
            }
        }
    }

    // For IndirectLoad, ensure the table is available.
    if let ScalarOp::IndirectLoad { table_base } = &group.op {
        ensure_atom_available(
            graph,
            table_base.0,
            lane,
            lane_graphs,
            lane_inputs,
            lane_input_atoms,
            lane_group_set,
            phase_gi_set,
            input_tensors,
        );
    }
}

/// Ensure atoms referenced by an InputRef are available in a lane.
fn ensure_input_ref(
    graph: &NanoGraph,
    input: &InputRef,
    atom_offset: u64,
    count: u64,
    lane: usize,
    lane_graphs: &mut [NanoGraph],
    lane_inputs: &mut [Vec<AtomRange>],
    lane_input_atoms: &mut [HashSet<u64>],
    lane_group_set: &mut [HashSet<usize>],
    phase_gi_set: &HashSet<usize>,
    input_tensors: &[InputTensor],
) {
    if count == 0 {
        return;
    }

    // Compute the range of source atoms this input ref accesses.
    let (lo, hi) = input_ref_source_range(input, count, atom_offset);

    // Ensure all groups/input tensors in [lo, hi] are available.
    ensure_range_available(
        graph,
        lo,
        hi,
        lane,
        lane_graphs,
        lane_inputs,
        lane_input_atoms,
        lane_group_set,
        phase_gi_set,
        input_tensors,
    );
}

/// Ensure atoms in the reduce stride range are available.
fn ensure_reduce_stride_range(
    graph: &NanoGraph,
    input: &InputRef,
    atom_offset: u64,
    count: u64,
    reduce_count: u64,
    reduce_stride: i64,
    lane: usize,
    lane_graphs: &mut [NanoGraph],
    lane_inputs: &mut [Vec<AtomRange>],
    lane_input_atoms: &mut [HashSet<u64>],
    lane_group_set: &mut [HashSet<usize>],
    phase_gi_set: &HashSet<usize>,
    input_tensors: &[InputTensor],
) {
    // The reduce reads input.resolve(i + atom_offset) + k * reduce_stride
    // for k in 0..reduce_count. We need to cover the full range.
    let first = input.resolve(atom_offset);
    let last = input.resolve(atom_offset + count - 1);
    let end_off = (reduce_count as i64 - 1) * reduce_stride;

    let endpoints = [
        first.0,
        (first.0 as i64 + end_off) as u64,
        last.0,
        (last.0 as i64 + end_off) as u64,
    ];
    let lo = *endpoints.iter().min().unwrap();
    let hi = *endpoints.iter().max().unwrap();

    ensure_range_available(
        graph,
        lo,
        hi,
        lane,
        lane_graphs,
        lane_inputs,
        lane_input_atoms,
        lane_group_set,
        phase_gi_set,
        input_tensors,
    );
}

/// Ensure all atoms in [lo, hi] are available in a lane.
fn ensure_range_available(
    graph: &NanoGraph,
    lo: u64,
    hi: u64,
    lane: usize,
    lane_graphs: &mut [NanoGraph],
    lane_inputs: &mut [Vec<AtomRange>],
    lane_input_atoms: &mut [HashSet<u64>],
    lane_group_set: &mut [HashSet<usize>],
    phase_gi_set: &HashSet<usize>,
    input_tensors: &[InputTensor],
) {
    let groups = graph.groups();

    // Check input tensors first.
    for it in input_tensors {
        if it.count == 0 {
            continue;
        }
        let it_lo = it.base_id.0;
        let it_hi = it_lo + it.count - 1;
        if it_hi >= lo && it_lo <= hi {
            // This input tensor overlaps our range.
            declare_input_tensor_in_lane(it, lane, lane_graphs, lane_inputs, lane_input_atoms);
        }
    }

    // Check graph input tensors.
    for it in graph.input_tensors() {
        if it.count == 0 {
            continue;
        }
        let it_lo = it.base_id.0;
        let it_hi = it_lo + it.count - 1;
        if it_hi >= lo && it_lo <= hi {
            declare_input_tensor_in_lane(it, lane, lane_graphs, lane_inputs, lane_input_atoms);
        }
    }

    // Check groups in the range that are not in this phase (external deps).
    // These need to be declared as span inputs.
    for (gi, g) in groups.iter().enumerate() {
        if g.count == 0 {
            continue;
        }
        let g_lo = g.base_id.0;
        let g_hi = g_lo + g.count - 1;
        if g_lo > hi {
            break; // Groups are sorted by base_id.
        }
        if g_hi < lo {
            continue;
        }
        // This group overlaps our range.
        if !phase_gi_set.contains(&gi) {
            // External group: declare as input.
            declare_group_as_input(g, lane, lane_graphs, lane_inputs, lane_input_atoms);
        }
        // If it's in this phase but not yet in this lane, and it's a literal,
        // we may need to wait (it will be placed later in topo order, which is fine
        // since we process groups in sorted order).
    }
}

fn ensure_atom_available(
    graph: &NanoGraph,
    atom_id: u64,
    lane: usize,
    lane_graphs: &mut [NanoGraph],
    lane_inputs: &mut [Vec<AtomRange>],
    lane_input_atoms: &mut [HashSet<u64>],
    lane_group_set: &mut [HashSet<usize>],
    phase_gi_set: &HashSet<usize>,
    input_tensors: &[InputTensor],
) {
    ensure_range_available(
        graph,
        atom_id,
        atom_id,
        lane,
        lane_graphs,
        lane_inputs,
        lane_input_atoms,
        lane_group_set,
        phase_gi_set,
        input_tensors,
    );
}

fn declare_input_tensor_in_lane(
    it: &InputTensor,
    lane: usize,
    lane_graphs: &mut [NanoGraph],
    lane_inputs: &mut [Vec<AtomRange>],
    lane_input_atoms: &mut [HashSet<u64>],
) {
    if lane_input_atoms[lane].contains(&it.base_id.0) {
        return; // Already declared.
    }
    lane_input_atoms[lane].insert(it.base_id.0);
    lane_graphs[lane].insert_input_tensor_at(it.base_id, it.tensor_id, it.count, it.dtype);
    lane_inputs[lane].push(AtomRange {
        base: it.base_id,
        count: it.count,
        dtype: it.dtype,
    });
}

fn declare_group_as_input(
    group: &AtomGroup,
    lane: usize,
    lane_graphs: &mut [NanoGraph],
    lane_inputs: &mut [Vec<AtomRange>],
    lane_input_atoms: &mut [HashSet<u64>],
) {
    if lane_input_atoms[lane].contains(&group.base_id.0) {
        return; // Already declared.
    }
    lane_input_atoms[lane].insert(group.base_id.0);
    // Register the atom range in the span NanoGraph so that contains_atom()
    // returns true for these atoms during validation and execution.
    lane_graphs[lane].insert_input_tensor_at(
        group.base_id,
        GlobalId(u64::MAX),
        group.count,
        group.output_dtype,
    );
    lane_inputs[lane].push(AtomRange {
        base: group.base_id,
        count: group.count,
        dtype: group.output_dtype,
    });
}

/// Determine if a group's output needs to be declared in span outputs.
fn needs_output(
    gi: usize,
    group: &AtomGroup,
    successors: &[Vec<usize>],
    phase_assignments: &[usize],
    output_group_set: &HashSet<usize>,
    phase_idx: usize,
    num_phases: usize,
) -> bool {
    // If this is a final output, yes.
    if output_group_set.contains(&gi) {
        return true;
    }

    // If any successor is in a later phase, yes.
    for &si in &successors[gi] {
        if phase_assignments[si] > phase_idx {
            return true;
        }
    }

    false
}

// ─── Utility ───────────────────────────────────────────────────────────────

fn empty_span() -> Span {
    Span {
        graph: NanoGraph::new(),
        inputs: vec![],
        outputs: vec![],
    }
}

/// Compute the (lo, hi) inclusive source atom range for an InputRef.
fn input_ref_source_range(input: &InputRef, count: u64, atom_offset: u64) -> (u64, u64) {
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

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GlobalId;
    use crate::nano_graph::ops::{ReduceKind, ScalarBinOp, ScalarUnaryOp};
    use crate::nano_graph::pattern::InputTensor;
    use crate::nano_graph::{AtomId, InputRef, NanoGraph, ScalarOp};
    use crate::numeric_dtype::NumericDType;
    use crate::numeric_scalar::NumericScalar;

    /// Helper: create an input tensor in the graph.
    fn add_input(graph: &mut NanoGraph, id: u64, count: u64) -> AtomId {
        graph.add_input_tensor(GlobalId(id), count, NumericDType::F32)
    }

    /// Helper: create a literal group.
    fn add_literal(graph: &mut NanoGraph, count: u64) -> AtomId {
        graph.push_group(
            count,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(1.0)),
            vec![],
            vec![],
        )
    }

    /// Helper: create a binary op group.
    fn add_binary(
        graph: &mut NanoGraph,
        count: u64,
        a: AtomId,
        b: AtomId,
        op: ScalarBinOp,
    ) -> AtomId {
        graph.push_group(
            count,
            NumericDType::F32,
            ScalarOp::Binary {
                op,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(a, 1), InputRef::affine(b, 1)],
        )
    }

    /// Helper: create a unary op group.
    fn add_unary(graph: &mut NanoGraph, count: u64, input: AtomId, op: ScalarUnaryOp) -> AtomId {
        graph.push_group(
            count,
            NumericDType::F32,
            ScalarOp::Unary {
                op,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(input, 1)],
        )
    }

    /// Helper: create a reduce group.
    fn add_reduce(
        graph: &mut NanoGraph,
        out_count: u64,
        input: AtomId,
        reduce_count: u64,
    ) -> AtomId {
        graph.push_group(
            out_count,
            NumericDType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count,
                reduce_stride: 1,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(input, reduce_count as i64)],
        )
    }

    /// Count the total atoms across all lanes in a phase.
    fn phase_atom_counts(phase: &Phase) -> Vec<u64> {
        phase
            .spans
            .iter()
            .map(|s| s.graph.groups().iter().map(|g| g.count).sum())
            .collect()
    }

    /// Count how many lanes are active (non-empty) in a phase.
    fn active_lanes(phase: &Phase) -> usize {
        phase
            .spans
            .iter()
            .filter(|s| s.graph.num_groups() > 0)
            .count()
    }

    // ── Test: linear chain splitting ──────────────────────────────────

    #[test]
    fn test_linear_chain_split() {
        // Sub(1024) → Pow(1024): should be ONE phase, split across 4 lanes.
        let mut g = NanoGraph::new();
        let input = add_input(&mut g, 1, 1024);
        let sub = add_binary(&mut g, 1024, input, input, ScalarBinOp::Sub);
        let pow = add_unary(&mut g, 1024, sub, ScalarUnaryOp::Exp); // use Exp as "power-like"
        g.outputs = vec![pow];

        let input_tensors = g.input_tensors().to_vec();
        let output_ids: Vec<AtomId> = g.outputs.clone();
        let phases = plan(&g, 4, &input_tensors, &output_ids);

        // Should be a single phase (no barriers needed for elementwise chain).
        assert_eq!(
            phases.len(),
            1,
            "Linear chain should be 1 phase, got {}",
            phases.len()
        );

        let counts = phase_atom_counts(&phases[0]);
        // All 4 lanes should be active.
        assert_eq!(active_lanes(&phases[0]), 4, "All 4 lanes should be active");

        // Each lane should have roughly 256 + 256 = 512 atoms (1024/4 each for Sub and Exp).
        for (lane, &count) in counts.iter().enumerate() {
            assert!(
                count >= 400 && count <= 600,
                "Lane {} has {} atoms, expected ~512",
                lane,
                count,
            );
        }
    }

    // ── Test: diamond graph ───────────────────────────────────────────

    #[test]
    fn test_diamond_graph() {
        // Input → A(1024) → B(1024), A → C(1024), B+C → D(1024)
        // All elementwise, should be 1 phase, all lanes active.
        let mut g = NanoGraph::new();
        let input = add_input(&mut g, 1, 1024);
        let a = add_unary(&mut g, 1024, input, ScalarUnaryOp::Neg);
        let b = add_unary(&mut g, 1024, a, ScalarUnaryOp::Exp);
        let c = add_unary(&mut g, 1024, a, ScalarUnaryOp::Tanh);
        let d = add_binary(&mut g, 1024, b, c, ScalarBinOp::Add);
        g.outputs = vec![d];

        let input_tensors = g.input_tensors().to_vec();
        let output_ids = g.outputs.clone();
        let phases = plan(&g, 4, &input_tensors, &output_ids);

        // Single phase.
        assert_eq!(
            phases.len(),
            1,
            "Diamond should be 1 phase, got {}",
            phases.len()
        );
        assert_eq!(active_lanes(&phases[0]), 4, "All 4 lanes should be active");

        // Each lane gets 1/4 of each of the 4 groups = 4*256 = 1024 atoms.
        let counts = phase_atom_counts(&phases[0]);
        for (lane, &count) in counts.iter().enumerate() {
            assert!(
                count >= 800 && count <= 1200,
                "Lane {} has {} atoms, expected ~1024",
                lane,
                count,
            );
        }
    }

    // ── Test: matmul-like structure with split ────────────────────────

    #[test]
    fn test_matmul_like_split() {
        // Simulate a matmul: M=64 output elements, each a ReduceSum over K=128.
        // Structure:
        //   weights: Literal(64*128 = 8192) — duplicated
        //   input: InputTensor(128)
        //   mul: Binary Mul(8192) with StridedBroadcast input — split across lanes
        //   reduce: Reduce(64, reduce_count=128, stride=1) — split across lanes
        let mut g = NanoGraph::new();
        let input = add_input(&mut g, 1, 128);
        let weights = add_literal(&mut g, 8192);
        let mul = g.push_group(
            8192,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![
                InputRef::strided_broadcast(input, 1, 128),
                InputRef::affine(weights, 1),
            ],
        );
        let reduce = g.push_group(
            64,
            NumericDType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: 128,
                reduce_stride: 1,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(mul, 128)],
        );
        g.outputs = vec![reduce];

        let input_tensors = g.input_tensors().to_vec();
        let output_ids = g.outputs.clone();
        let phases = plan(&g, 4, &input_tensors, &output_ids);

        // The mul is splittable (8192 atoms). The reduce has count=64,
        // which is also splittable (each of the 64 outputs is independent).
        // However, the reduce reads from the mul via stride=128.
        // With 4 lanes, each lane gets 8192/4 = 2048 mul atoms and 16 reduce atoms.

        // Should be a small number of phases.
        assert!(
            phases.len() <= 3,
            "MatMul should have <= 3 phases, got {}",
            phases.len()
        );

        // At least 2 lanes should be active in the first phase.
        let total_active: usize = phases.iter().map(|p| active_lanes(p)).sum();
        assert!(total_active >= 4, "Should have good lane utilization");

        // Verify weights are duplicated (literal).
        let mut lanes_with_weights = 0;
        for phase in &phases {
            for span in &phase.spans {
                for grp in span.graph.groups() {
                    if matches!(grp.op, ScalarOp::Literal(_)) && grp.count == 8192 {
                        lanes_with_weights += 1;
                    }
                }
            }
        }
        // Weights should be in all lanes that do the Mul.
        assert!(
            lanes_with_weights >= 2,
            "Weights should be duplicated to multiple lanes, got {}",
            lanes_with_weights
        );
    }

    // ── Test: literal duplication ─────────────────────────────────────

    #[test]
    fn test_literal_duplication() {
        // A literal consumed by a split group should be duplicated to all lanes.
        let mut g = NanoGraph::new();
        let lit = add_literal(&mut g, 100);
        let input = add_input(&mut g, 1, 800);
        let add = g.push_group(
            800,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(input, 1), InputRef::modular(lit, 1, 100)],
        );
        g.outputs = vec![add];

        let input_tensors = g.input_tensors().to_vec();
        let output_ids = g.outputs.clone();
        let phases = plan(&g, 4, &input_tensors, &output_ids);

        // Literal should be in all 4 lanes.
        let mut lanes_with_lit = 0;
        for phase in &phases {
            for span in &phase.spans {
                for grp in span.graph.groups() {
                    if matches!(grp.op, ScalarOp::Literal(_)) && grp.count == 100 {
                        lanes_with_lit += 1;
                    }
                }
            }
        }
        assert_eq!(lanes_with_lit, 4, "Literal should be in all 4 lanes");

        // The add group (800 atoms) should be split across 4 lanes.
        assert_eq!(active_lanes(&phases[0]), 4);
        let counts = phase_atom_counts(&phases[0]);
        for (lane, &count) in counts.iter().enumerate() {
            // Each lane: 100 (literal) + 200 (split add) = 300
            assert!(
                count >= 250 && count <= 350,
                "Lane {} has {} atoms, expected ~300",
                lane,
                count,
            );
        }
    }

    // ── Test: reduce handling with barrier ────────────────────────────

    #[test]
    fn test_reduce_with_barrier() {
        // Large input → Split elementwise → Reduce (count=1, whole) → downstream
        // The single-output reduce should cause a barrier.
        let mut g = NanoGraph::new();
        let input = add_input(&mut g, 1, 1024);
        let neg = add_unary(&mut g, 1024, input, ScalarUnaryOp::Neg);
        // Reduce to 1 value: count=1, reduce_count=1024
        let reduce = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: 1024,
                reduce_stride: 1,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(neg, 1)],
        );
        // Downstream: broadcast the reduce result to 1024 atoms.
        let broadcast = g.push_group(
            1024,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Div,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(input, 1), InputRef::Broadcast(reduce)],
        );
        g.outputs = vec![broadcast];

        let input_tensors = g.input_tensors().to_vec();
        let output_ids = g.outputs.clone();
        let phases = plan(&g, 4, &input_tensors, &output_ids);

        // Should have multiple phases due to the whole-reduce barrier.
        // Phase 1: neg split across 4 lanes
        // Phase 2: reduce on one lane (needs barrier to read all lanes' neg output)
        // Phase 3: broadcast div split across 4 lanes (needs barrier to read reduce output)
        assert!(
            phases.len() >= 2,
            "Should have >= 2 phases due to reduce barrier, got {}",
            phases.len()
        );

        // First phase should have good utilization (neg split).
        assert!(
            active_lanes(&phases[0]) >= 2,
            "First phase should have multiple active lanes"
        );
    }

    // ── Test: groups are actually split (not just assigned whole) ─────

    #[test]
    fn test_groups_actually_split() {
        // Verify that a 1000-atom group is actually split into 4 fragments.
        let mut g = NanoGraph::new();
        let input = add_input(&mut g, 1, 1000);
        let neg = add_unary(&mut g, 1000, input, ScalarUnaryOp::Neg);
        g.outputs = vec![neg];

        let input_tensors = g.input_tensors().to_vec();
        let output_ids = g.outputs.clone();
        let phases = plan(&g, 4, &input_tensors, &output_ids);

        assert_eq!(phases.len(), 1);

        // Verify each lane has a fragment of the neg group.
        let mut total_neg_atoms = 0u64;
        let mut lanes_with_neg = 0;
        for span in &phases[0].spans {
            for grp in span.graph.groups() {
                if matches!(
                    grp.op,
                    ScalarOp::Unary {
                        op: ScalarUnaryOp::Neg,
                        ..
                    }
                ) {
                    total_neg_atoms += grp.count;
                    lanes_with_neg += 1;
                    // Each fragment should have count=250 (1000/4).
                    assert_eq!(
                        grp.count, 250,
                        "Fragment should have 250 atoms, got {}",
                        grp.count
                    );
                    // atom_offset should be set correctly.
                    assert_eq!(
                        grp.atom_offset,
                        grp.base_id.0 - neg.0,
                        "atom_offset should equal fragment position within original group"
                    );
                }
            }
        }
        assert_eq!(lanes_with_neg, 4, "Neg should be split into 4 fragments");
        assert_eq!(
            total_neg_atoms, 1000,
            "Total atoms should equal original count"
        );
    }

    // ── Test: split preserves atom IDs ────────────────────────────────

    #[test]
    fn test_split_preserves_atom_ids() {
        // Verify that split fragments maintain the original atom ID space.
        let mut g = NanoGraph::new();
        let input = add_input(&mut g, 1, 800);
        let add_id = add_unary(&mut g, 800, input, ScalarUnaryOp::Exp);
        g.outputs = vec![add_id];

        let input_tensors = g.input_tensors().to_vec();
        let output_ids = g.outputs.clone();
        let phases = plan(&g, 4, &input_tensors, &output_ids);

        // Collect all base_ids across all lanes.
        let mut fragment_ranges: Vec<(u64, u64)> = vec![];
        for span in &phases[0].spans {
            for grp in span.graph.groups() {
                if matches!(
                    grp.op,
                    ScalarOp::Unary {
                        op: ScalarUnaryOp::Exp,
                        ..
                    }
                ) {
                    fragment_ranges.push((grp.base_id.0, grp.base_id.0 + grp.count));
                }
            }
        }
        fragment_ranges.sort();

        // Fragments should cover the entire original range contiguously.
        assert_eq!(fragment_ranges.len(), 4);
        assert_eq!(
            fragment_ranges[0].0, add_id.0,
            "First fragment should start at original base"
        );
        for i in 1..fragment_ranges.len() {
            assert_eq!(
                fragment_ranges[i].0,
                fragment_ranges[i - 1].1,
                "Fragments should be contiguous"
            );
        }
        assert_eq!(
            fragment_ranges.last().unwrap().1,
            add_id.0 + 800,
            "Last fragment should end at original base + count"
        );
    }

    // ── Test: empty graph ────────────────────────────────────────────

    #[test]
    fn test_empty_graph() {
        let g = NanoGraph::new();
        let phases = plan(&g, 4, &[], &[]);
        assert_eq!(phases.len(), 1);
        assert_eq!(phases[0].spans.len(), 4);
    }

    // ── Test: single small group ────────────────────────────────────

    #[test]
    fn test_single_small_group() {
        let mut g = NanoGraph::new();
        let input = add_input(&mut g, 1, 1);
        let neg = add_unary(&mut g, 1, input, ScalarUnaryOp::Neg);
        g.outputs = vec![neg];

        let input_tensors = g.input_tensors().to_vec();
        let output_ids = g.outputs.clone();
        let phases = plan(&g, 4, &input_tensors, &output_ids);

        // Single atom can't be split meaningfully. Should still produce valid output.
        assert!(!phases.is_empty());
        // The atom should exist somewhere.
        let mut found = false;
        for phase in &phases {
            for span in &phase.spans {
                for grp in span.graph.groups() {
                    if matches!(grp.op, ScalarOp::Unary { .. }) {
                        found = true;
                    }
                }
            }
        }
        assert!(found, "Single atom group should be placed");
    }

    // ── Test: large reduce is split (independent outputs) ────────────

    #[test]
    fn test_large_reduce_is_split() {
        // 64 independent reductions: each output atom reduces 128 inputs.
        // The 64 outputs should be split across lanes.
        let mut g = NanoGraph::new();
        let input = add_input(&mut g, 1, 64 * 128);
        let reduce = g.push_group(
            64,
            NumericDType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: 128,
                reduce_stride: 1,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(input, 128)],
        );
        g.outputs = vec![reduce];

        let input_tensors = g.input_tensors().to_vec();
        let output_ids = g.outputs.clone();
        let phases = plan(&g, 4, &input_tensors, &output_ids);

        // The reduce has 64 independent outputs — should be split across 4 lanes.
        let mut lanes_with_reduce = 0;
        let mut total_reduce_atoms = 0u64;
        for phase in &phases {
            for span in &phase.spans {
                for grp in span.graph.groups() {
                    if matches!(grp.op, ScalarOp::Reduce { .. }) {
                        lanes_with_reduce += 1;
                        total_reduce_atoms += grp.count;
                    }
                }
            }
        }
        assert_eq!(
            lanes_with_reduce, 4,
            "Reduce should be split across 4 lanes"
        );
        assert_eq!(total_reduce_atoms, 64, "Total reduce atoms should be 64");
    }

    // ── Test: validate span NanoGraphs ────────────────────────────────

    #[test]
    fn test_span_nanographs_validate() {
        // Build a non-trivial graph and verify all span NanoGraphs pass validation.
        let mut g = NanoGraph::new();
        let input = add_input(&mut g, 1, 512);
        let lit = add_literal(&mut g, 512);
        let add = add_binary(&mut g, 512, input, lit, ScalarBinOp::Add);
        let neg = add_unary(&mut g, 512, add, ScalarUnaryOp::Neg);
        g.outputs = vec![neg];

        let input_tensors = g.input_tensors().to_vec();
        let output_ids = g.outputs.clone();
        let phases = plan(&g, 4, &input_tensors, &output_ids);

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
    }

    // ── Test: outputs are produced ────────────────────────────────────

    #[test]
    fn test_all_outputs_produced() {
        // Verify that every output atom is produced by some span.
        let mut g = NanoGraph::new();
        let input = add_input(&mut g, 1, 256);
        let a = add_unary(&mut g, 256, input, ScalarUnaryOp::Neg);
        let b = add_unary(&mut g, 256, a, ScalarUnaryOp::Exp);
        g.outputs = vec![b];

        let input_tensors = g.input_tensors().to_vec();
        let output_ids = g.outputs.clone();
        let phases = plan(&g, 4, &input_tensors, &output_ids);

        // Collect all output atom ranges across all phases.
        let mut output_atoms: HashSet<u64> = HashSet::new();
        for phase in &phases {
            for span in &phase.spans {
                for out in &span.outputs {
                    for i in 0..out.count {
                        output_atoms.insert(out.base.0 + i);
                    }
                }
            }
        }

        // The output group b should be fully covered.
        for i in 0..256 {
            assert!(
                output_atoms.contains(&(b.0 + i)),
                "Output atom {} not produced by any span",
                b.0 + i,
            );
        }
    }

    // ── Test: long chain stays in one phase ──────────────────────────

    #[test]
    fn test_long_elementwise_chain_one_phase() {
        // A → B → C → D → E, all elementwise with count=2048.
        // Should be ONE phase, all split across lanes.
        let mut g = NanoGraph::new();
        let input = add_input(&mut g, 1, 2048);
        let a = add_unary(&mut g, 2048, input, ScalarUnaryOp::Neg);
        let b = add_unary(&mut g, 2048, a, ScalarUnaryOp::Exp);
        let c = add_unary(&mut g, 2048, b, ScalarUnaryOp::Tanh);
        let d = add_unary(&mut g, 2048, c, ScalarUnaryOp::Sqrt);
        let e = add_unary(&mut g, 2048, d, ScalarUnaryOp::Reciprocal);
        g.outputs = vec![e];

        let input_tensors = g.input_tensors().to_vec();
        let output_ids = g.outputs.clone();
        let phases = plan(&g, 8, &input_tensors, &output_ids);

        assert_eq!(
            phases.len(),
            1,
            "Long elementwise chain should be 1 phase, got {}",
            phases.len()
        );
        assert_eq!(active_lanes(&phases[0]), 8, "All 8 lanes should be active");

        // Each lane should have 5 groups of 256 atoms each = 1280 atoms.
        let counts = phase_atom_counts(&phases[0]);
        for (lane, &count) in counts.iter().enumerate() {
            assert!(
                count >= 1100 && count <= 1500,
                "Lane {} has {} atoms, expected ~1280",
                lane,
                count,
            );
        }
    }
}
