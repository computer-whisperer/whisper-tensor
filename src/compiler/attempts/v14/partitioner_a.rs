#![allow(clippy::all, dead_code, unreachable_patterns, unused_imports)]

//! Pinch-Point Barriers partitioner.
//!
//! Finds natural phase boundaries by tracking group-level liveness through
//! a topological traversal of the NanoGraph. Pinch points — positions where
//! the live data set is minimal — become barrier positions. Within each
//! phase, independent groups are distributed across lanes for parallel
//! execution.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::dtype::DType;
use crate::graph::GlobalId;
use crate::nano_graph::pattern::InputTensor;
use crate::nano_graph::{AtomGroup, AtomId, AtomRange, InputRef, NanoGraph, ScalarOp, SymDim};

use super::types::{Phase, Span};

// ─── Public API ─────────────────────────────────────────────────────────────

/// Partition a NanoGraph into phases and lanes.
///
/// Returns a `Vec<Phase>` where each phase contains one span per lane.
/// The caller wraps this into an `ExecutionPlan` with metadata.
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

    // Step 1: Build group-level dependency info.
    let dep_info = build_dependency_info(graph);

    // Step 2: Compute liveness profile and find pinch points.
    let phase_boundaries = find_pinch_points(graph, &dep_info, output_atom_ids);

    // Step 3: Assign groups to phases based on boundaries.
    let phase_assignments = assign_phases(n, &phase_boundaries);

    // Step 4: Within each phase, assign groups to lanes.
    let num_phases = if phase_boundaries.is_empty() {
        1
    } else {
        phase_boundaries.len() + 1
    };
    let lane_result = assign_lanes(graph, &dep_info, &phase_assignments, num_phases, num_lanes);

    // Step 5: Build span NanoGraphs.
    build_phases(
        graph,
        input_tensors,
        output_atom_ids,
        &phase_assignments,
        &lane_result,
        num_phases,
        num_lanes,
        &dep_info,
    )
}

// ─── Dependency analysis ────────────────────────────────────────────────────

/// Per-group dependency information.
struct DepInfo {
    /// For each group index, the set of producer group indices.
    producers: Vec<Vec<usize>>,
    /// For each group index, the set of consumer group indices.
    consumers: Vec<Vec<usize>>,
    /// For each group, how many downstream groups consume its output.
    use_counts: Vec<u32>,
}

fn build_dependency_info(graph: &NanoGraph) -> DepInfo {
    let groups = graph.groups();
    let n = groups.len();
    let mut producers: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut use_counts = vec![0u32; n];

    for (gi, group) in groups.iter().enumerate() {
        let mut seen = HashSet::new();
        graph.collect_all_producer_indices(group, gi, &mut seen);
        for &pi in &seen {
            use_counts[pi] += 1;
            consumers[pi].push(gi);
        }
        producers[gi] = seen.into_iter().collect();
        producers[gi].sort_unstable();
    }

    // Deduplicate consumers (a producer can appear in multiple InputRefs).
    for c in consumers.iter_mut() {
        c.sort_unstable();
        c.dedup();
    }

    DepInfo {
        producers,
        consumers,
        use_counts,
    }
}

// ─── Pinch point detection ──────────────────────────────────────────────────

/// Find phase boundary positions (group indices after which a barrier is placed).
///
/// We track the "live set" as we traverse groups in topological order.
/// A group's output becomes live when we visit it, and becomes dead when
/// its last consumer has been visited. The live set size (measured in atoms)
/// expands and contracts. Local minima in this curve are pinch points.
fn find_pinch_points(
    graph: &NanoGraph,
    dep_info: &DepInfo,
    output_atom_ids: &[AtomId],
) -> Vec<usize> {
    let groups = graph.groups();
    let n = groups.len();

    if n <= 1 {
        return vec![];
    }

    // Build set of output group indices (groups whose atoms are model outputs).
    let mut output_group_set = HashSet::new();
    for &out_id in output_atom_ids {
        if let Some(gi) = graph.find_group_idx(out_id) {
            output_group_set.insert(gi);
        }
    }

    // Track remaining consumer count for each group.
    let mut remaining_uses: Vec<u32> = dep_info.use_counts.clone();

    // Output groups are never "dead" (they must survive to the end).
    // We model this by giving them an extra use that never gets decremented.
    for &gi in &output_group_set {
        remaining_uses[gi] += 1;
    }

    // Traverse in topological order (which is insertion order), tracking live set.
    let mut live_atoms: i64 = 0;
    let mut profile: Vec<i64> = Vec::with_capacity(n);

    for gi in 0..n {
        let group = &groups[gi];

        // This group's output becomes live.
        live_atoms += group.count as i64;

        // Decrement remaining uses for all producers of this group.
        for &pi in &dep_info.producers[gi] {
            remaining_uses[pi] -= 1;
            if remaining_uses[pi] == 0 {
                // Producer is dead — remove from live set.
                live_atoms -= groups[pi].count as i64;
            }
        }

        profile.push(live_atoms);
    }

    // Find pinch points: local minima in the liveness profile.
    // We look for positions where the live set is relatively small compared
    // to surrounding peaks. A pinch point at index `i` means we place a
    // barrier after group `i` (groups 0..=i in one phase, i+1.. in the next).
    //
    // Strategy: find valleys where the live set drops below a threshold.
    // The threshold is a fraction of the maximum live set size.

    if n < 4 {
        return vec![];
    }

    let max_live = *profile.iter().max().unwrap_or(&0);
    if max_live <= 0 {
        return vec![];
    }

    // For the pinch-point threshold, we use a fraction of the max live set.
    // A true pinch point (e.g., between transformer layers) will have very
    // few live atoms relative to within-layer computation.
    let threshold = (max_live as f64 * 0.15) as i64;

    // Find candidate pinch points: local minima below threshold.
    // We skip the first and last few groups to avoid degenerate boundaries.
    let margin = 2.min(n / 4);
    let mut candidates: Vec<(usize, i64)> = Vec::new();

    for i in margin..(n - margin) {
        let val = profile[i];
        if val > threshold {
            continue;
        }

        // Check if this is a local minimum within a window.
        let window = 5.min(i).min(n - 1 - i);
        let is_local_min =
            (1..=window).all(|d| profile[i] <= profile[i - d] && profile[i] <= profile[i + d]);

        if is_local_min {
            candidates.push((i, val));
        }
    }

    // Merge candidates that are close together (keep the deepest).
    let min_gap = 10.max(n / 50);
    let mut boundaries: Vec<usize> = Vec::new();

    let mut i = 0;
    while i < candidates.len() {
        let mut best_idx = candidates[i].0;
        let mut best_val = candidates[i].1;
        let mut j = i + 1;

        while j < candidates.len() && candidates[j].0 - candidates[i].0 < min_gap {
            if candidates[j].1 < best_val {
                best_val = candidates[j].1;
                best_idx = candidates[j].0;
            }
            j += 1;
        }

        boundaries.push(best_idx);
        i = j;
    }

    // If no pinch points found but the graph is large enough, try a simpler
    // heuristic: split at the deepest valley.
    if boundaries.is_empty() && n > 100 {
        let skip = n / 10;
        if let Some((best_i, _)) = profile[skip..(n - skip)]
            .iter()
            .enumerate()
            .min_by_key(|(_, v)| *v)
        {
            let actual_i = best_i + skip;
            // Only use it if it's at least 50% below the max.
            if profile[actual_i] < max_live / 2 {
                boundaries.push(actual_i);
            }
        }
    }

    boundaries
}

// ─── Phase assignment ───────────────────────────────────────────────────────

/// Assign each group to a phase based on the boundary positions.
fn assign_phases(num_groups: usize, boundaries: &[usize]) -> Vec<usize> {
    let mut assignments = vec![0usize; num_groups];
    let mut phase = 0;
    let mut boundary_idx = 0;

    for gi in 0..num_groups {
        if boundary_idx < boundaries.len() && gi > boundaries[boundary_idx] {
            phase += 1;
            boundary_idx += 1;
        }
        assignments[gi] = phase;
    }

    assignments
}

// ─── Lane assignment ────────────────────────────────────────────────────────

/// Result of lane assignment: per-group lane + set of groups to duplicate.
struct LaneResult {
    /// Lane assignment for each group.
    lane: Vec<usize>,
    /// Groups that should be duplicated into every lane that needs them.
    /// These are Literal groups consumed by multiple independent chains.
    /// They are NOT assigned to a specific lane — each lane that references
    /// them gets its own copy.
    duplicated: HashSet<usize>,
}

/// Assign groups within each phase to lanes for parallel execution.
///
/// Key insight: groups that share a common producer (e.g., matmul rows
/// sharing weight data) are independent — they read the same data but
/// produce independent outputs. We only union groups through "private"
/// producer edges (where the producer has exactly one consumer in this
/// phase). Shared producers with multiple consumers don't create
/// connectivity between their consumers.
///
/// Shared Literal groups (weights, constants) are duplicated into every
/// lane that needs them. Shared computed groups go to lane 0 and create
/// a sub-phase boundary if needed.
fn assign_lanes(
    graph: &NanoGraph,
    dep_info: &DepInfo,
    phase_assignments: &[usize],
    num_phases: usize,
    num_lanes: usize,
) -> LaneResult {
    let groups = graph.groups();
    let n = groups.len();
    let mut lane_assignments = vec![0usize; n];
    let mut duplicated = HashSet::new();

    if num_lanes <= 1 {
        return LaneResult {
            lane: lane_assignments,
            duplicated,
        };
    }

    for phase in 0..num_phases {
        // Collect groups in this phase.
        let phase_groups: Vec<usize> = (0..n)
            .filter(|&gi| phase_assignments[gi] == phase)
            .collect();

        if phase_groups.is_empty() {
            continue;
        }

        let phase_set: HashSet<usize> = phase_groups.iter().copied().collect();

        // For each group in this phase, find its intra-phase producers.
        let mut intra_producers: HashMap<usize, Vec<usize>> = HashMap::new();

        for &gi in &phase_groups {
            let mut intra = Vec::new();
            for &pi in &dep_info.producers[gi] {
                if phase_set.contains(&pi) {
                    intra.push(pi);
                }
            }
            intra_producers.insert(gi, intra);
        }

        // Count intra-phase consumers for each group in this phase.
        let mut intra_consumer_count: HashMap<usize, usize> = HashMap::new();
        for &gi in &phase_groups {
            for &pi in intra_producers.get(&gi).unwrap_or(&Vec::new()) {
                *intra_consumer_count.entry(pi).or_default() += 1;
            }
        }

        // Identify shared groups: consumed by multiple groups in this phase.
        // Shared Literal groups get duplicated into each lane.
        // Shared computed groups force all their consumers into the same lane.
        let mut shared_computed: HashSet<usize> = HashSet::new();

        for (&gi, &count) in &intra_consumer_count {
            if count > 1 {
                if matches!(groups[gi].op, ScalarOp::Literal(_)) {
                    duplicated.insert(gi);
                } else {
                    // Non-Literal shared group: must keep it and all its
                    // consumers on the same lane to preserve independence.
                    shared_computed.insert(gi);
                }
            }
        }

        // Build connected components. Union through:
        // - Private edges (producer has exactly 1 intra-phase consumer)
        // - Shared computed edges (force consumers onto producer's lane)
        // Skip duplicated (Literal) groups — they're replicated per-lane.
        let mut uf = UnionFind::new(n);

        for &gi in &phase_groups {
            if duplicated.contains(&gi) {
                continue;
            }
            for &pi in intra_producers.get(&gi).unwrap_or(&Vec::new()) {
                if duplicated.contains(&pi) {
                    continue; // Skip edges through duplicated Literal groups.
                }
                let consumer_count = intra_consumer_count.get(&pi).copied().unwrap_or(0);
                if consumer_count <= 1 || shared_computed.contains(&pi) {
                    // Private edge OR shared computed edge: union them.
                    // For shared computed, this forces all consumers
                    // onto the same lane as the shared group.
                    uf.union(gi, pi);
                }
                // Shared Literal edge: consumers are independent, don't union.
            }
        }

        // Collect connected components (non-duplicated groups only).
        let mut components: HashMap<usize, (Vec<usize>, u64)> = HashMap::new();
        for &gi in &phase_groups {
            if duplicated.contains(&gi) {
                continue;
            }
            let rep = uf.find(gi);
            let entry = components.entry(rep).or_insert_with(|| (Vec::new(), 0));
            entry.0.push(gi);
            entry.1 += groups[gi].count;
        }

        // Sort components by total atoms (descending) for greedy bin-packing.
        let mut sorted_components: Vec<(Vec<usize>, u64)> = components.into_values().collect();
        sorted_components.sort_by(|a, b| b.1.cmp(&a.1));

        // Greedy bin-packing: assign each component to the least-loaded lane.
        let mut lane_loads = vec![0u64; num_lanes];

        for (component_groups, _total_atoms) in &sorted_components {
            let target_lane = lane_loads
                .iter()
                .enumerate()
                .min_by_key(|(_, load)| *load)
                .unwrap()
                .0;

            for &gi in component_groups {
                lane_assignments[gi] = target_lane;
                lane_loads[target_lane] += groups[gi].count;
            }
        }

        // Duplicated groups: lane assignment doesn't matter (they go to
        // every lane that needs them). Set to 0 as a default.
        for &gi in &duplicated {
            if phase_set.contains(&gi) {
                lane_assignments[gi] = 0;
            }
        }
    }

    LaneResult {
        lane: lane_assignments,
        duplicated,
    }
}

// ─── Union-Find ─────────────────────────────────────────────────────────────

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]]; // path halving
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] += 1;
        }
    }
}

// ─── Span construction ─────────────────────────────────────────────────────

/// Build all phases and spans from the assignment data.
fn build_phases(
    graph: &NanoGraph,
    input_tensors: &[InputTensor],
    output_atom_ids: &[AtomId],
    phase_assignments: &[usize],
    lane_result: &LaneResult,
    num_phases: usize,
    num_lanes: usize,
    dep_info: &DepInfo,
) -> Vec<Phase> {
    let groups = graph.groups();
    let n = groups.len();
    let lane_assignments = &lane_result.lane;

    // For each phase+lane, collect the group indices.
    // Duplicated groups are added to EVERY lane that consumes them.
    let mut phase_lane_groups: Vec<Vec<Vec<usize>>> = vec![vec![Vec::new(); num_lanes]; num_phases];

    // First, place non-duplicated groups.
    for gi in 0..n {
        if lane_result.duplicated.contains(&gi) {
            continue; // Handle separately below.
        }
        let phase = phase_assignments[gi];
        let lane = lane_assignments[gi];
        phase_lane_groups[phase][lane].push(gi);
    }

    // For duplicated groups, find which lanes need them by checking which
    // lanes have consumers of the duplicated group.
    for &gi in &lane_result.duplicated {
        let phase = phase_assignments[gi];
        let mut needed_by_lanes: HashSet<usize> = HashSet::new();

        for &ci in &dep_info.consumers[gi] {
            if phase_assignments[ci] == phase && !lane_result.duplicated.contains(&ci) {
                needed_by_lanes.insert(lane_assignments[ci]);
            }
        }

        // If no lane explicitly needs it, put it on lane 0.
        if needed_by_lanes.is_empty() {
            needed_by_lanes.insert(0);
        }

        for lane in needed_by_lanes {
            phase_lane_groups[phase][lane].push(gi);
        }
    }

    // Sort each lane's groups by group index to maintain topological order.
    for phase_lanes in phase_lane_groups.iter_mut() {
        for lane_groups in phase_lanes.iter_mut() {
            lane_groups.sort_unstable();
        }
    }

    // Build set of groups in each phase+lane (after duplication).
    let mut phase_lane_sets: Vec<Vec<HashSet<usize>>> =
        vec![vec![HashSet::new(); num_lanes]; num_phases];
    for (phase, phase_lanes) in phase_lane_groups.iter().enumerate() {
        for (lane, lane_groups) in phase_lanes.iter().enumerate() {
            for &gi in lane_groups {
                phase_lane_sets[phase][lane].insert(gi);
            }
        }
    }

    // Determine which groups produce model outputs.
    let mut output_group_set = HashSet::new();
    for &out_id in output_atom_ids {
        if let Some(gi) = graph.find_group_idx(out_id) {
            output_group_set.insert(gi);
        }
    }

    // For each phase, determine the set of groups whose outputs cross
    // the phase boundary (consumed by a later phase or are model outputs).
    let mut phase_output_groups: Vec<HashSet<usize>> = vec![HashSet::new(); num_phases];
    for gi in 0..n {
        let my_phase = phase_assignments[gi];

        let crosses_boundary = dep_info.consumers[gi]
            .iter()
            .any(|&ci| phase_assignments[ci] > my_phase);

        if crosses_boundary || output_group_set.contains(&gi) {
            phase_output_groups[my_phase].insert(gi);
        }
    }

    // Now build each span.
    let mut phases: Vec<Phase> = Vec::with_capacity(num_phases);

    for phase in 0..num_phases {
        let mut spans: Vec<Span> = Vec::with_capacity(num_lanes);

        for lane in 0..num_lanes {
            let my_groups = &phase_lane_groups[phase][lane];

            if my_groups.is_empty() {
                spans.push(Span {
                    graph: NanoGraph::new(),
                    inputs: vec![],
                    outputs: vec![],
                });
                continue;
            }

            let span = build_span(
                graph,
                input_tensors,
                my_groups,
                &phase_lane_sets[phase][lane],
                &phase_output_groups[phase],
                dep_info,
                phase_assignments,
            );
            spans.push(span);
        }

        phases.push(Phase { spans });
    }

    phases
}

/// Build a single span NanoGraph for a given set of groups.
fn build_span(
    graph: &NanoGraph,
    input_tensors: &[InputTensor],
    my_group_indices: &[usize],
    my_group_set: &HashSet<usize>,
    phase_output_groups: &HashSet<usize>,
    dep_info: &DepInfo,
    phase_assignments: &[usize],
) -> Span {
    let groups = graph.groups();

    // Determine external inputs: atom ranges needed by this span's groups
    // that are not produced by other groups in this span.
    //
    // These come from:
    // 1. Input tensors (weights/user inputs)
    // 2. Groups in earlier phases
    // 3. Groups in the same phase but different lanes (this shouldn't happen
    //    if independence is maintained, but we handle it for safety)

    let mut external_ranges: BTreeMap<u64, AtomRange> = BTreeMap::new();

    for &gi in my_group_indices {
        let group = &groups[gi];

        // Find all producer indices for this group.
        for &pi in &dep_info.producers[gi] {
            if my_group_set.contains(&pi) {
                // Internal to this span — no external input needed.
                continue;
            }

            // External producer. Add its atom range as an input.
            let producer = &groups[pi];
            external_ranges
                .entry(producer.base_id.0)
                .or_insert_with(|| AtomRange {
                    base: producer.base_id,
                    count: producer.count,
                    dtype: producer.output_dtype,
                });
        }

        // Also check if any InputRef references an input tensor.
        for input_ref in &group.inputs {
            collect_input_tensor_refs(
                graph,
                input_ref,
                group.count,
                group.atom_offset,
                &mut external_ranges,
            );
        }

        // For reduce ops, check the reduce stride range for input tensor refs.
        if let ScalarOp::Reduce {
            reduce_count,
            reduce_stride,
            ..
        } = &group.op
        {
            if *reduce_count > 1 && *reduce_stride != 0 {
                for input_ref in &group.inputs {
                    collect_reduce_input_tensor_refs(
                        graph,
                        input_ref,
                        group.count,
                        group.atom_offset,
                        *reduce_count,
                        *reduce_stride,
                        &mut external_ranges,
                    );
                }
            }
        }

        // For IndirectLoad, check table_base.
        if let ScalarOp::IndirectLoad { table_base } = &group.op {
            // Check if table_base is in an input tensor.
            let in_input_tensor = graph
                .input_tensors()
                .iter()
                .any(|it| table_base.0 >= it.base_id.0 && table_base.0 < it.base_id.0 + it.count);
            if in_input_tensor {
                // Will be picked up by collect_input_tensor_refs via the group's InputRef.
                // Also add the input tensor range explicitly.
                for it in graph.input_tensors() {
                    if table_base.0 >= it.base_id.0 && table_base.0 < it.base_id.0 + it.count {
                        external_ranges
                            .entry(it.base_id.0)
                            .or_insert_with(|| AtomRange {
                                base: it.base_id,
                                count: it.count,
                                dtype: it.dtype,
                            });
                        break;
                    }
                }
            } else if let Some(pi) = graph.find_group_idx(*table_base) {
                if !my_group_set.contains(&pi) {
                    let producer = &groups[pi];
                    external_ranges
                        .entry(producer.base_id.0)
                        .or_insert_with(|| AtomRange {
                            base: producer.base_id,
                            count: producer.count,
                            dtype: producer.output_dtype,
                        });
                }
            }
        }
    }

    let span_inputs: Vec<AtomRange> = external_ranges.values().cloned().collect();

    // Determine outputs: groups in this span that are phase outputs.
    let mut span_outputs: Vec<AtomRange> = Vec::new();
    for &gi in my_group_indices {
        if phase_output_groups.contains(&gi) {
            let group = &groups[gi];
            span_outputs.push(AtomRange {
                base: group.base_id,
                count: group.count,
                dtype: group.output_dtype,
            });
        }
    }

    // Build the span's NanoGraph.
    let span_graph = build_span_nanograph(graph, my_group_indices, &span_inputs);

    Span {
        graph: span_graph,
        inputs: span_inputs,
        outputs: span_outputs,
    }
}

/// Collect references to input tensors from an InputRef.
fn collect_input_tensor_refs(
    graph: &NanoGraph,
    input_ref: &InputRef,
    count: u64,
    atom_offset: u64,
    external_ranges: &mut BTreeMap<u64, AtomRange>,
) {
    // Sample a few positions to check if the InputRef references input tensors.
    let sample_ids: Vec<AtomId> = match input_ref {
        InputRef::Broadcast(id) => vec![*id],
        InputRef::Affine { .. } | InputRef::StridedBroadcast { .. } => {
            if count == 0 {
                return;
            }
            let first = input_ref.resolve(atom_offset);
            let last = input_ref.resolve(atom_offset + count - 1);
            vec![first, last]
        }
        InputRef::Modular {
            base,
            stride,
            modulus,
        } => {
            vec![
                *base,
                AtomId((base.0 as i64 + *stride * (*modulus as i64 - 1)) as u64),
            ]
        }
        InputRef::Explicit(ids) => {
            // Sample first, last, and middle.
            let mut samples = Vec::new();
            if !ids.is_empty() {
                samples.push(ids[atom_offset as usize]);
                if count > 1 {
                    samples.push(ids[(atom_offset + count - 1) as usize]);
                }
            }
            samples
        }
    };

    for id in sample_ids {
        // Check if this atom is in an input tensor.
        for it in graph.input_tensors() {
            if id.0 >= it.base_id.0 && id.0 < it.base_id.0 + it.count {
                external_ranges
                    .entry(it.base_id.0)
                    .or_insert_with(|| AtomRange {
                        base: it.base_id,
                        count: it.count,
                        dtype: it.dtype,
                    });
                break;
            }
        }
    }
}

/// Collect input tensor refs for reduce stride patterns.
fn collect_reduce_input_tensor_refs(
    graph: &NanoGraph,
    input_ref: &InputRef,
    count: u64,
    atom_offset: u64,
    reduce_count: u64,
    reduce_stride: i64,
    external_ranges: &mut BTreeMap<u64, AtomRange>,
) {
    if count == 0 {
        return;
    }

    // Check endpoints of the reduce stride range.
    let first = input_ref.resolve(atom_offset);
    let last = input_ref.resolve(atom_offset + count - 1);
    let end_off = (reduce_count as i64 - 1) * reduce_stride;

    let endpoints = [
        first.0,
        (first.0 as i64 + end_off) as u64,
        last.0,
        (last.0 as i64 + end_off) as u64,
    ];

    for &ep in &endpoints {
        let id = AtomId(ep);
        for it in graph.input_tensors() {
            if id.0 >= it.base_id.0 && id.0 < it.base_id.0 + it.count {
                external_ranges
                    .entry(it.base_id.0)
                    .or_insert_with(|| AtomRange {
                        base: it.base_id,
                        count: it.count,
                        dtype: it.dtype,
                    });
                break;
            }
        }
    }
}

/// Build a NanoGraph for a span, preserving original atom IDs.
///
/// Uses `alloc_placeholder`/`fill_placeholder` to place groups at their
/// original atom ID positions. Input ranges are registered for external data.
fn build_span_nanograph(
    graph: &NanoGraph,
    group_indices: &[usize],
    inputs: &[AtomRange],
) -> NanoGraph {
    let groups = graph.groups();
    let mut span_graph = NanoGraph::new();

    // Copy symbolic dimension info from the main graph.
    span_graph.sym_dim_names = graph.sym_dim_names.clone();
    span_graph.sym_dim_bounds = graph.sym_dim_bounds.clone();

    // We need to place groups at specific atom IDs. The NanoGraph uses
    // sequential allocation, so we use input tensor registration and
    // placeholder allocation to "skip" to the right positions.
    //
    // Strategy: process groups and inputs in ascending atom ID order.
    // Between groups, use input tensors or placeholders to fill gaps.

    // Collect all ranges we need to place, sorted by base_id.
    #[derive(Debug, Clone)]
    enum PlaceItem {
        InputRange(AtomRange),
        Group(usize), // index into main graph's groups
    }

    let mut items: Vec<(u64, PlaceItem)> = Vec::new();

    // Add input ranges.
    for input in inputs {
        items.push((input.base.0, PlaceItem::InputRange(input.clone())));
    }

    // Add groups.
    for &gi in group_indices {
        items.push((groups[gi].base_id.0, PlaceItem::Group(gi)));
    }

    // Sort by atom ID position.
    items.sort_by_key(|(pos, _)| *pos);

    // Place items, filling gaps with dummy allocations to advance the ID counter.
    // We track the current next_atom_id position.
    let mut current_id: u64 = 0;

    for (target_pos, item) in &items {
        let target = *target_pos;

        // Skip ahead if needed by allocating a dummy input tensor to fill the gap.
        if target > current_id {
            let gap = target - current_id;
            span_graph.add_input_tensor(GlobalId(u64::MAX - current_id), gap, DType::F32);
            current_id = target;
        }

        match item {
            PlaceItem::InputRange(range) => {
                let base =
                    span_graph.add_input_tensor(GlobalId(range.base.0), range.count, range.dtype);
                debug_assert_eq!(
                    base.0, target,
                    "Input range placed at wrong position: expected {}, got {}",
                    target, base.0
                );
                current_id = target + range.count;
            }
            PlaceItem::Group(gi) => {
                let group = &groups[*gi];
                let base = span_graph.alloc_placeholder(group.count, group.output_dtype);
                debug_assert_eq!(
                    base.0, target,
                    "Group placed at wrong position: expected {}, got {}",
                    target, base.0
                );
                span_graph.fill_placeholder(
                    base,
                    group.count,
                    group.output_dtype,
                    group.op.clone(),
                    group.sym_dims.clone(),
                    group.inputs.clone(),
                );
                current_id = target + group.count;
            }
        }
    }

    span_graph
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::nano_graph::ops::{ReduceKind, ScalarBinOp, ScalarOp, ScalarUnaryOp};
    use crate::nano_graph::pattern::InputTensor;
    use crate::numeric_scalar::NumericScalar;

    /// Helper: create a literal group (constant data, like weights).
    fn push_literal(g: &mut NanoGraph, count: u64) -> AtomId {
        g.push_group(
            count,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        )
    }

    /// Helper: create an elementwise binary op group.
    fn push_add(g: &mut NanoGraph, count: u64, a: AtomId, b: AtomId) -> AtomId {
        g.push_group(
            count,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine { base: a, stride: 1 },
                InputRef::Affine { base: b, stride: 1 },
            ],
        )
    }

    /// Helper: create a multiply group.
    fn push_mul(
        g: &mut NanoGraph,
        count: u64,
        a: AtomId,
        a_stride: i64,
        b: AtomId,
        b_stride: i64,
    ) -> AtomId {
        g.push_group(
            count,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine {
                    base: a,
                    stride: a_stride,
                },
                InputRef::Affine {
                    base: b,
                    stride: b_stride,
                },
            ],
        )
    }

    /// Helper: create a reduce-sum group.
    fn push_reduce_sum(
        g: &mut NanoGraph,
        count: u64,
        input: AtomId,
        reduce_count: u64,
        reduce_stride: i64,
    ) -> AtomId {
        g.push_group(
            count,
            DType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count,
                reduce_stride,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: input,
                stride: 1,
            }],
        )
    }

    /// Trivial graph: single group.
    #[test]
    fn test_single_group() {
        let mut g = NanoGraph::new();
        let a = push_literal(&mut g, 100);
        g.outputs = vec![a];

        let input_tensors: Vec<InputTensor> = vec![];
        let phases = plan(&g, 2, &input_tensors, &g.outputs.clone());

        assert!(!phases.is_empty());
        // Single group should produce one phase.
        assert_eq!(phases.len(), 1);
        // Should have 2 spans (one per lane).
        assert_eq!(phases[0].spans.len(), 2);

        // One span should have the group, the other should be empty.
        let total_groups: usize = phases[0].spans.iter().map(|s| s.graph.num_groups()).sum();
        assert_eq!(total_groups, 1);
    }

    /// Linear chain: a -> b -> c.
    #[test]
    fn test_linear_chain() {
        let mut g = NanoGraph::new();
        let a = push_literal(&mut g, 100);
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

        let phases = plan(&g, 4, &[], &g.outputs.clone());

        // Linear chain — no parallelism, should be one phase.
        assert!(!phases.is_empty());

        // All groups should be covered.
        let total_groups: usize = phases
            .iter()
            .flat_map(|p| p.spans.iter())
            .map(|s| s.graph.num_groups())
            .sum();
        assert_eq!(total_groups, 3);

        // Validate each span's NanoGraph.
        for phase in &phases {
            for span in &phase.spans {
                if span.graph.num_groups() > 0 {
                    let errors = span.graph.validate();
                    assert!(errors.is_empty(), "Span validation errors: {:?}", errors);
                }
            }
        }
    }

    /// Diamond dependency: a -> b, a -> c, b+c -> d.
    #[test]
    fn test_diamond() {
        let mut g = NanoGraph::new();
        let a = push_literal(&mut g, 100);
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
        let d = push_add(&mut g, 100, b, c);
        g.outputs = vec![d];

        let phases = plan(&g, 2, &[], &g.outputs.clone());

        assert!(!phases.is_empty());

        // All 4 groups should be present.
        let total_groups: usize = phases
            .iter()
            .flat_map(|p| p.spans.iter())
            .map(|s| s.graph.num_groups())
            .sum();
        assert_eq!(total_groups, 4);

        for phase in &phases {
            for span in &phase.spans {
                if span.graph.num_groups() > 0 {
                    let errors = span.graph.validate();
                    assert!(errors.is_empty(), "Span validation errors: {:?}", errors);
                }
            }
        }
    }

    /// Two independent branches: a->b and c->d, should parallelize.
    #[test]
    fn test_independent_branches() {
        let mut g = NanoGraph::new();
        let a = push_literal(&mut g, 1000);
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
        let c = push_literal(&mut g, 1000);
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
        g.outputs = vec![b, d];

        let phases = plan(&g, 2, &[], &g.outputs.clone());

        assert!(!phases.is_empty());

        // With 2 lanes, the two branches should be in different lanes.
        // Check that at least one phase has work on both lanes.
        let has_parallel_phase = phases
            .iter()
            .any(|p| p.spans.iter().filter(|s| s.graph.num_groups() > 0).count() > 1);
        assert!(
            has_parallel_phase,
            "Expected parallel execution of independent branches"
        );

        let total_groups: usize = phases
            .iter()
            .flat_map(|p| p.spans.iter())
            .map(|s| s.graph.num_groups())
            .sum();
        assert_eq!(total_groups, 4);
    }

    /// Simulate a simple matmul pattern: M independent multiply+reduce rows
    /// sharing the same weight data.
    #[test]
    fn test_matmul_pattern() {
        let mut g = NanoGraph::new();
        let m = 8; // number of rows
        let k = 64; // reduction dimension

        // Weight matrix: k*m atoms (shared by all rows).
        let weights = push_literal(&mut g, k * m);

        // Input vector: k atoms.
        let input = push_literal(&mut g, k);

        // M independent multiply groups, each of size k.
        let mut mul_ids = Vec::new();
        for row in 0..m {
            let mul = g.push_group(
                k,
                DType::F32,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                },
                vec![],
                vec![
                    InputRef::Affine {
                        base: weights.offset(row * k),
                        stride: 1,
                    },
                    InputRef::Affine {
                        base: input,
                        stride: 1,
                    },
                ],
            );
            mul_ids.push(mul);
        }

        // M independent reduce groups, each reducing k atoms to 1.
        let mut reduce_ids = Vec::new();
        for row in 0..m {
            let reduce = push_reduce_sum(&mut g, 1, mul_ids[row as usize], k, 1);
            reduce_ids.push(reduce);
        }

        g.outputs = reduce_ids.clone();

        let phases = plan(&g, 4, &[], &g.outputs.clone());

        assert!(!phases.is_empty());

        // Verify all groups are covered.
        let total_groups: usize = phases
            .iter()
            .flat_map(|p| p.spans.iter())
            .map(|s| s.graph.num_groups())
            .sum();
        // 1 weight + 1 input + 8 mul + 8 reduce = 18 groups.
        // Weights and input may be duplicated across lanes since they're
        // Literal groups (no external input). The key check is that all
        // compute groups are present.
        assert!(
            total_groups >= 18,
            "Expected at least 18 groups, got {}",
            total_groups
        );

        // Validate all spans.
        for phase in &phases {
            for span in &phase.spans {
                if span.graph.num_groups() > 0 {
                    let errors = span.graph.validate();
                    assert!(errors.is_empty(), "Span validation errors: {:?}", errors);
                }
            }
        }
    }

    /// Test with input tensors (external data).
    #[test]
    fn test_with_input_tensors() {
        let mut g = NanoGraph::new();

        // Register an input tensor.
        let input_base = g.add_input_tensor(GlobalId(1), 100, DType::F32);

        // A group that reads from the input tensor.
        let a = g.push_group(
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

        let b = g.push_group(
            100,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine { base: a, stride: 1 }],
        );

        g.outputs = vec![b];

        let it = &g.input_tensors().to_vec();
        let outputs = g.outputs.clone();
        let phases = plan(&g, 2, it, &outputs);

        assert!(!phases.is_empty());

        // The span that has work should declare the input tensor in its inputs.
        let active_spans: Vec<&Span> = phases
            .iter()
            .flat_map(|p| p.spans.iter())
            .filter(|s| s.graph.num_groups() > 0)
            .collect();

        assert!(!active_spans.is_empty());

        // At least one active span should have inputs (the input tensor).
        let has_inputs = active_spans.iter().any(|s| !s.inputs.is_empty());
        assert!(
            has_inputs,
            "Active spans should declare input tensor as input"
        );

        for span in &active_spans {
            let errors = span.graph.validate();
            assert!(errors.is_empty(), "Span validation errors: {:?}", errors);
        }
    }

    /// Test empty graph.
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

    /// Test single lane (no parallelism).
    #[test]
    fn test_single_lane() {
        let mut g = NanoGraph::new();
        let a = push_literal(&mut g, 50);
        let b = push_literal(&mut g, 50);
        let c = push_add(&mut g, 50, a, b);
        g.outputs = vec![c];

        let phases = plan(&g, 1, &[], &g.outputs.clone());

        assert!(!phases.is_empty());
        for phase in &phases {
            assert_eq!(phase.spans.len(), 1);
        }

        let total_groups: usize = phases
            .iter()
            .flat_map(|p| p.spans.iter())
            .map(|s| s.graph.num_groups())
            .sum();
        assert_eq!(total_groups, 3);
    }

    /// Test that span outputs are correctly identified.
    #[test]
    fn test_span_outputs() {
        let mut g = NanoGraph::new();
        let a = push_literal(&mut g, 100);
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
        g.outputs = vec![b];

        let phases = plan(&g, 1, &[], &g.outputs.clone());

        // The last phase should have spans whose outputs include the model output.
        let last_phase = phases.last().unwrap();
        let output_atoms: Vec<AtomId> = last_phase
            .spans
            .iter()
            .flat_map(|s| s.outputs.iter())
            .map(|r| r.base)
            .collect();

        assert!(
            output_atoms.contains(&b),
            "Model output atom should be in last phase's span outputs"
        );
    }

    /// Simulate a two-layer pattern with a pinch point in between.
    /// Layer 1: weights1 -> mul1 rows -> reduce1 rows -> residual
    /// Layer 2: residual + weights2 -> mul2 rows -> reduce2 rows -> output
    #[test]
    fn test_two_layer_pinch() {
        let mut g = NanoGraph::new();
        let m = 16; // rows
        let k = 32; // reduction dimension
        let hidden = 16; // residual stream width

        // Layer 1: weights.
        let w1 = push_literal(&mut g, k * m);
        let x = push_literal(&mut g, k);

        // Layer 1: M multiply groups.
        let mut mul1 = Vec::new();
        for row in 0..m {
            let m_id = g.push_group(
                k,
                DType::F32,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                },
                vec![],
                vec![
                    InputRef::Affine {
                        base: w1.offset(row * k),
                        stride: 1,
                    },
                    InputRef::Affine { base: x, stride: 1 },
                ],
            );
            mul1.push(m_id);
        }

        // Layer 1: M reduce groups -> hidden-size residual.
        let mut res = Vec::new();
        for row in 0..m {
            let r = push_reduce_sum(&mut g, 1, mul1[row as usize], k, 1);
            res.push(r);
        }

        // Layer 2: weights.
        let w2 = push_literal(&mut g, hidden * m);

        // Layer 2: M multiply groups (reading from residual).
        let mut mul2 = Vec::new();
        for row in 0..m {
            let m_id = g.push_group(
                hidden,
                DType::F32,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                },
                vec![],
                vec![
                    InputRef::Affine {
                        base: w2.offset(row * hidden),
                        stride: 1,
                    },
                    InputRef::Explicit((0..hidden).map(|i| res[i as usize]).collect()),
                ],
            );
            mul2.push(m_id);
        }

        // Layer 2: M reduce groups.
        let mut out = Vec::new();
        for row in 0..m {
            let r = push_reduce_sum(&mut g, 1, mul2[row as usize], hidden, 1);
            out.push(r);
        }

        g.outputs = out.clone();

        let phases = plan(&g, 4, &[], &g.outputs.clone());

        assert!(!phases.is_empty());

        // Verify all groups are covered.
        let total_groups: usize = phases
            .iter()
            .flat_map(|p| p.spans.iter())
            .map(|s| s.graph.num_groups())
            .sum();

        // Count expected: 2 weight groups + 1 input + 2*M mul + 2*M reduce = 3 + 4*M
        let expected = 3 + 4 * m as usize;
        assert!(
            total_groups >= expected,
            "Expected at least {} groups, got {}",
            expected,
            total_groups
        );

        // Validate all spans.
        for (pi, phase) in phases.iter().enumerate() {
            for (li, span) in phase.spans.iter().enumerate() {
                if span.graph.num_groups() > 0 {
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
    }

    /// Test that broadcast inputs are handled correctly.
    #[test]
    fn test_broadcast_input() {
        let mut g = NanoGraph::new();
        let scalar = g.push_group(
            1,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(3.14)),
            vec![],
            vec![],
        );
        let vec_data = push_literal(&mut g, 100);

        let result = g.push_group(
            100,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine {
                    base: vec_data,
                    stride: 1,
                },
                InputRef::Broadcast(scalar),
            ],
        );

        g.outputs = vec![result];

        let phases = plan(&g, 2, &[], &g.outputs.clone());

        // Validate all spans.
        for phase in &phases {
            for span in &phase.spans {
                if span.graph.num_groups() > 0 {
                    let errors = span.graph.validate();
                    assert!(errors.is_empty(), "Span validation errors: {:?}", errors);
                }
            }
        }
    }

    /// Verify invariant: within a phase, spans are independent (no data flow).
    #[test]
    fn test_span_independence() {
        let mut g = NanoGraph::new();

        // Create two independent chains.
        let a1 = push_literal(&mut g, 500);
        let b1 = g.push_group(
            500,
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

        let a2 = push_literal(&mut g, 500);
        let b2 = g.push_group(
            500,
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

        let phases = plan(&g, 2, &[], &g.outputs.clone());

        // For each phase, verify no span reads atoms produced by another span.
        for phase in &phases {
            for (i, span_i) in phase.spans.iter().enumerate() {
                let produced: HashSet<u64> = span_i
                    .graph
                    .groups()
                    .iter()
                    .flat_map(|g| (g.base_id.0..g.base_id.0 + g.count))
                    .collect();

                for (j, span_j) in phase.spans.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    // Check that span_j doesn't reference atoms produced by span_i
                    // (other than through its declared inputs, which come from the store).
                    for group in span_j.graph.groups() {
                        for input_ref in &group.inputs {
                            for k in 0..group.count.min(10) {
                                let src = input_ref.resolve(k + group.atom_offset);
                                if produced.contains(&src.0) {
                                    // This is only a problem if the source isn't
                                    // declared as an input to span_j.
                                    let is_declared = span_j
                                        .inputs
                                        .iter()
                                        .any(|r| src.0 >= r.base.0 && src.0 < r.base.0 + r.count);
                                    // If it's produced by span_i and consumed by span_j
                                    // in the same phase, that's an independence violation
                                    // unless it's a declared input (from a prior phase).
                                    if !is_declared {
                                        // Check if it's produced in a prior phase via the store.
                                        // For now, we just check the graph structure.
                                        let is_input_tensor =
                                            span_j.graph.input_tensors().iter().any(|it| {
                                                src.0 >= it.base_id.0
                                                    && src.0 < it.base_id.0 + it.count
                                            });
                                        assert!(
                                            is_input_tensor,
                                            "Independence violation: span {} produces atom {}, consumed by span {} without declaration",
                                            i, src, j
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
