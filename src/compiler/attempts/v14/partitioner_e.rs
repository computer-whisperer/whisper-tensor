#![allow(
    clippy::all,
    dead_code,
    unreachable_code,
    unreachable_patterns,
    unused_imports,
    unused_variables
)]

//! Simulated Annealing Partitioner (attempt E)
//!
//! Strategy: start from a valid wavefront assignment (B-style) that guarantees
//! zero cross-lane violations, then use simulated annealing to optimize lane
//! assignments for work balance while preserving correctness.
//!
//! The key insight is that the hard problem is lane assignment + balance, not
//! phase detection. Greedy approaches (A, C) achieved 0 violations but
//! catastrophic imbalance. Annealing can escape local optima that greedy
//! approaches get stuck in.
//!
//! Algorithm:
//! 1. Build group-level dependency DAG
//! 2. Compute depth for each group (longest path from any root)
//! 3. Groups at the same depth form a wavefront (provably independent)
//! 4. Merge consecutive wavefronts when safe (conservative criterion)
//! 5. Within each phase, use simulated annealing to optimize lane assignments
//!    - Constraint chains (intra-phase dependencies) must stay on the same lane
//!    - Moves: reassign chains to different lanes, accepted by Metropolis criterion
//!    - Cost: work imbalance across lanes (max/mean ratio)
//! 6. Build span NanoGraphs with preserved atom IDs

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::dtype::DType;
use crate::graph::GlobalId;
use crate::nano_graph::pattern::InputTensor;
use crate::nano_graph::{AtomGroup, AtomId, AtomRange, InputRef, NanoGraph, ScalarOp, SymDim};

use super::types::{Phase, Span};

// ─── Public API ────────────────────────────────────────────────────────────

/// Partition a NanoGraph into phases and spans for parallel execution.
///
/// Uses simulated annealing to optimize lane assignments within each phase
/// for work balance while maintaining correctness (no cross-lane violations).
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

    // Step 2: Compute depth for each group.
    let depths = compute_depths(n, &producers);

    // Step 3: Form wavefronts (groups at the same depth).
    let max_depth = depths.iter().copied().max().unwrap_or(0);
    let mut wavefronts: Vec<Vec<usize>> = vec![vec![]; max_depth + 1];
    for (gi, &depth) in depths.iter().enumerate() {
        wavefronts[depth].push(gi);
    }

    // Step 4: Merge wavefronts into phases.
    let phase_assignments = merge_wavefronts(&wavefronts, &producers, n);

    // Step 5: Collect groups per phase.
    let num_phases = phase_assignments.iter().copied().max().unwrap_or(0) + 1;
    let mut phase_groups: Vec<Vec<usize>> = vec![vec![]; num_phases];
    for (gi, &phase) in phase_assignments.iter().enumerate() {
        phase_groups[phase].push(gi);
    }

    // Step 6: Identify output groups.
    let output_group_set = identify_output_groups(graph, output_atom_ids);

    // Step 7: For each phase, build constraint chains, anneal lane assignments,
    //         and construct spans.
    let mut phases = Vec::with_capacity(num_phases);
    for phase_idx in 0..num_phases {
        let phase = build_phase_with_annealing(
            graph,
            &phase_groups[phase_idx],
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
/// Depth = longest path from any root (a group with no producers).
fn compute_depths(n: usize, producers: &[Vec<usize>]) -> Vec<usize> {
    let mut depths = vec![0usize; n];
    for gi in 0..n {
        let mut max_dep_depth = 0usize;
        for &pi in &producers[gi] {
            max_dep_depth = max_dep_depth.max(depths[pi] + 1);
        }
        depths[gi] = max_dep_depth;
    }
    depths
}

// ─── Wavefront merging ─────────────────────────────────────────────────────

/// Merge consecutive wavefronts into phases to reduce barrier count.
///
/// Two adjacent wavefronts can be merged if every group in the later wavefront
/// depends on at most one group from wavefronts already in the same phase.
/// This ensures each group can be co-located with its single dependency on the
/// same lane, maintaining inter-lane independence.
fn merge_wavefronts(wavefronts: &[Vec<usize>], producers: &[Vec<usize>], n: usize) -> Vec<usize> {
    if wavefronts.is_empty() {
        return vec![];
    }

    let mut phase_assignment = vec![0usize; n];
    let mut group_wavefront = vec![0usize; n];
    for (wi, wavefront) in wavefronts.iter().enumerate() {
        for &gi in wavefront {
            group_wavefront[gi] = wi;
        }
    }

    let mut wavefront_to_phase = vec![0usize; wavefronts.len()];
    let mut next_phase = 0usize;

    for wi in 0..wavefronts.len() {
        if wi == 0 {
            wavefront_to_phase[wi] = 0;
            next_phase = 1;
            continue;
        }

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

    for (wi, wavefront) in wavefronts.iter().enumerate() {
        for &gi in wavefront {
            phase_assignment[gi] = wavefront_to_phase[wi];
        }
    }

    phase_assignment
}

/// Check if wavefront `wi` can be merged into the same phase as wavefront `wi-1`.
fn can_merge_wavefront(
    wavefronts: &[Vec<usize>],
    wi: usize,
    wavefront_to_phase: &[usize],
    producers: &[Vec<usize>],
    group_wavefront: &[usize],
) -> bool {
    let target_phase = wavefront_to_phase[wi - 1];

    for &gi in &wavefronts[wi] {
        let mut deps_in_phase = HashSet::new();
        for &pi in &producers[gi] {
            let pw = group_wavefront[pi];
            if pw < wi && wavefront_to_phase[pw] == target_phase {
                deps_in_phase.insert(pi);
            }
        }
        if deps_in_phase.len() > 1 {
            return false;
        }
    }

    true
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

// ─── Constraint chain computation ──────────────────────────────────────────

/// A constraint chain: a set of groups that must be on the same lane
/// because of intra-phase dependencies.
#[derive(Debug, Clone)]
struct ConstraintChain {
    /// Group indices in this chain (sorted by group index = topo order).
    groups: Vec<usize>,
    /// Total atom count across all groups.
    total_atoms: u64,
}

/// Build constraint chains within a phase using union-find.
/// Groups that depend on each other within the same phase must be
/// on the same lane — they form a constraint chain.
fn build_constraint_chains(
    groups: &[AtomGroup],
    phase_group_indices: &[usize],
    producers: &[Vec<usize>],
) -> Vec<ConstraintChain> {
    let phase_set: HashSet<usize> = phase_group_indices.iter().copied().collect();

    // We need a compact index mapping for union-find.
    // Map group_idx -> compact_idx and back.
    let mut gi_to_compact: HashMap<usize, usize> = HashMap::new();
    let mut compact_to_gi: Vec<usize> = Vec::new();
    for (i, &gi) in phase_group_indices.iter().enumerate() {
        gi_to_compact.insert(gi, i);
        compact_to_gi.push(gi);
    }

    let nc = compact_to_gi.len();
    let mut uf = UnionFind::new(nc);

    for &gi in phase_group_indices {
        let ci = gi_to_compact[&gi];
        for &pi in &producers[gi] {
            if let Some(&cpi) = gi_to_compact.get(&pi) {
                uf.union(ci, cpi);
            }
        }
    }

    // Collect chains by root.
    let mut chain_map: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..nc {
        let root = uf.find(i);
        chain_map.entry(root).or_default().push(compact_to_gi[i]);
    }

    chain_map
        .into_values()
        .map(|mut gis| {
            gis.sort_unstable();
            let total_atoms: u64 = gis.iter().map(|&gi| groups[gi].count).sum();
            ConstraintChain {
                groups: gis,
                total_atoms,
            }
        })
        .collect()
}

// ─── Simulated Annealing ───────────────────────────────────────────────────

/// Simple PRNG (xorshift64) for deterministic annealing.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1), // Avoid 0 state
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Returns a value in [0, bound).
    fn next_usize(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }

    /// Returns a float in [0.0, 1.0).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Cost function for a lane assignment.
/// Lower is better. Captures:
/// - Work imbalance (max_atoms / mean_atoms ratio, or infinite if any lane has 0 atoms
///   while others have work)
fn compute_cost(lane_atoms: &[u64]) -> f64 {
    let total: u64 = lane_atoms.iter().sum();
    if total == 0 {
        return 0.0; // No work in this phase, perfect balance.
    }

    let num_lanes = lane_atoms.len() as f64;
    let mean = total as f64 / num_lanes;
    let max = *lane_atoms.iter().max().unwrap() as f64;

    // Imbalance ratio: 1.0 is perfect, higher is worse.
    let imbalance = max / mean;

    // Also penalize lanes with zero work when there is work to distribute.
    let empty_lanes = lane_atoms.iter().filter(|&&a| a == 0).count();
    let empty_penalty = empty_lanes as f64 * 2.0;

    imbalance + empty_penalty
}

/// Run simulated annealing to optimize lane assignments for a set of chains.
///
/// Returns the lane assignment for each chain (chain_lane[i] = lane for chain i).
fn anneal_lane_assignments(
    chains: &[ConstraintChain],
    num_lanes: usize,
    phase_seed: u64,
) -> Vec<usize> {
    let nc = chains.len();

    if nc == 0 || num_lanes <= 1 {
        return vec![0; nc];
    }

    // Initial assignment: round-robin by descending chain size.
    // This gives a reasonable starting point (similar to B's approach).
    let mut sorted_indices: Vec<usize> = (0..nc).collect();
    sorted_indices.sort_by(|&a, &b| chains[b].total_atoms.cmp(&chains[a].total_atoms));

    let mut chain_lane = vec![0usize; nc];
    let mut lane_atoms = vec![0u64; num_lanes];

    // Assign largest chains first to least-loaded lane (greedy initial).
    for &ci in &sorted_indices {
        let target = lane_atoms
            .iter()
            .enumerate()
            .min_by_key(|&(_, &atoms)| atoms)
            .map(|(i, _)| i)
            .unwrap();
        chain_lane[ci] = target;
        lane_atoms[target] += chains[ci].total_atoms;
    }

    let initial_cost = compute_cost(&lane_atoms);

    // If cost is already near-perfect or there's only 1 chain, skip annealing.
    if initial_cost <= 1.01 || nc <= 1 {
        return chain_lane;
    }

    // Annealing parameters. Scale iterations with problem size but cap it.
    // For GPT-2 scale (~45K groups, but chains are much fewer per phase),
    // we want fast convergence.
    let max_iterations = (nc * 50).min(10_000);
    let t_start = initial_cost * 0.5; // Start temperature relative to initial cost.
    let t_end = 0.001;

    let mut rng = Rng::new(phase_seed);
    let mut current_cost = initial_cost;
    let mut best_cost = current_cost;
    let mut best_assignment = chain_lane.clone();

    for iter in 0..max_iterations {
        // Temperature schedule: exponential decay.
        let progress = iter as f64 / max_iterations as f64;
        let temperature = t_start * (t_end / t_start).powf(progress);

        // Generate a random move.
        let move_type = rng.next_usize(100);

        if move_type < 70 {
            // Move: reassign a random chain to a random different lane.
            let ci = rng.next_usize(nc);
            let old_lane = chain_lane[ci];
            let new_lane = loop {
                let l = rng.next_usize(num_lanes);
                if l != old_lane {
                    break l;
                }
            };

            // Compute delta cost.
            lane_atoms[old_lane] -= chains[ci].total_atoms;
            lane_atoms[new_lane] += chains[ci].total_atoms;
            let new_cost = compute_cost(&lane_atoms);
            let delta = new_cost - current_cost;

            // Metropolis criterion.
            if delta < 0.0 || rng.next_f64() < (-delta / temperature).exp() {
                // Accept.
                chain_lane[ci] = new_lane;
                current_cost = new_cost;
                if current_cost < best_cost {
                    best_cost = current_cost;
                    best_assignment = chain_lane.clone();
                }
            } else {
                // Reject: undo.
                lane_atoms[new_lane] -= chains[ci].total_atoms;
                lane_atoms[old_lane] += chains[ci].total_atoms;
            }
        } else {
            // Swap: exchange two chains between different lanes.
            let ci_a = rng.next_usize(nc);
            let ci_b = rng.next_usize(nc);
            if ci_a == ci_b || chain_lane[ci_a] == chain_lane[ci_b] {
                continue;
            }

            let lane_a = chain_lane[ci_a];
            let lane_b = chain_lane[ci_b];

            // Apply swap.
            lane_atoms[lane_a] =
                lane_atoms[lane_a] - chains[ci_a].total_atoms + chains[ci_b].total_atoms;
            lane_atoms[lane_b] =
                lane_atoms[lane_b] - chains[ci_b].total_atoms + chains[ci_a].total_atoms;

            let new_cost = compute_cost(&lane_atoms);
            let delta = new_cost - current_cost;

            if delta < 0.0 || rng.next_f64() < (-delta / temperature).exp() {
                chain_lane[ci_a] = lane_b;
                chain_lane[ci_b] = lane_a;
                current_cost = new_cost;
                if current_cost < best_cost {
                    best_cost = current_cost;
                    best_assignment = chain_lane.clone();
                }
            } else {
                // Undo.
                lane_atoms[lane_a] =
                    lane_atoms[lane_a] + chains[ci_a].total_atoms - chains[ci_b].total_atoms;
                lane_atoms[lane_b] =
                    lane_atoms[lane_b] + chains[ci_b].total_atoms - chains[ci_a].total_atoms;
            }
        }
    }

    best_assignment
}

// ─── Phase construction with annealing ─────────────────────────────────────

/// A unit of work assigned to a lane.
#[derive(Debug, Clone)]
struct WorkItem {
    group_idx: usize,
    atom_offset: u64,
    atom_count: u64,
}

/// Build a Phase using simulated annealing for lane assignment.
fn build_phase_with_annealing(
    graph: &NanoGraph,
    phase_group_indices: &[usize],
    phase_assignments: &[usize],
    producers: &[Vec<usize>],
    num_lanes: usize,
    input_tensors: &[InputTensor],
    output_group_set: &HashSet<usize>,
    phase_idx: usize,
    num_phases: usize,
) -> Phase {
    let groups = graph.groups();

    let mut sorted_groups = phase_group_indices.to_vec();
    sorted_groups.sort_unstable();

    let phase_group_set: HashSet<usize> = sorted_groups.iter().copied().collect();

    // Build constraint chains.
    let chains = build_constraint_chains(groups, &sorted_groups, producers);

    // Run simulated annealing on the chains.
    let chain_lanes = anneal_lane_assignments(&chains, num_lanes, phase_idx as u64 * 7919 + 42);

    // Convert chain assignments to per-group lane assignments.
    let mut lane_work: Vec<Vec<WorkItem>> = vec![vec![]; num_lanes];

    for (ci, chain) in chains.iter().enumerate() {
        let lane = chain_lanes[ci];
        for &gi in &chain.groups {
            lane_work[lane].push(WorkItem {
                group_idx: gi,
                atom_offset: 0,
                atom_count: groups[gi].count,
            });
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
    lane_work: &[WorkItem],
    phase_assignments: &[usize],
    producers: &[Vec<usize>],
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

    span_graph.sym_dim_names = graph.sym_dim_names.clone();
    span_graph.sym_dim_bounds = graph.sym_dim_bounds.clone();

    // Track which groups are computed in this span.
    let span_compute_set: HashSet<usize> = lane_work.iter().map(|w| w.group_idx).collect();

    // Collect external dependencies via BFS.
    let mut needed_external_atoms: BTreeMap<AtomId, (u64, DType)> = BTreeMap::new();
    let mut needed_internal_literals: BTreeSet<usize> = BTreeSet::new();

    const LITERAL_INLINE_THRESHOLD: u64 = 65536;

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

        // Check input tensor dependencies.
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

    // Check literal dependencies' own inputs.
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

    // Merge external ranges.
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

                span_graph.insert_group_at(
                    base,
                    *atom_count,
                    *atom_offset,
                    group.output_dtype,
                    group.op.clone(),
                    group.sym_dims.clone(),
                    group.inputs.clone(),
                );
                inserted_ranges.push((base.0, *atom_count));

                if output_group_set.contains(gi)
                    || needs_output(*gi, &groups[*gi], &span_compute_set)
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

    span_outputs = merge_atom_ranges(span_outputs);

    Span {
        graph: span_graph,
        inputs: span_inputs,
        outputs: span_outputs,
    }
}

/// Check if a group's output is needed beyond this span.
/// Conservative: output all non-literal compute groups.
fn needs_output(gi: usize, group: &AtomGroup, span_compute_set: &HashSet<usize>) -> bool {
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

    // ─── Test graph builders ───────────────────────────────────────────

    /// Linear chain: a -> b -> c.
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

    /// Diamond: lit -> (b, c) -> d.
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

    /// Two independent parallel chains.
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

    /// Graph with external input tensors.
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

    /// Matmul-like structure: M independent row computations sharing weights.
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

    /// Wider matmul-like for testing annealing balance with more rows.
    fn make_wide_matmul() -> (NanoGraph, Vec<InputTensor>) {
        let mut g = NanoGraph::new();

        let k = 32u64;
        let m = 16u64;

        // M independent chains: each has lit -> unary -> unary.
        // All chains are independent so should be distributed across lanes.
        let mut outputs = vec![];
        for i in 0..m {
            let lit = g.push_group(
                k,
                DType::F32,
                ScalarOp::Literal(NumericScalar::F32(i as f32)),
                vec![],
                vec![],
            );
            let neg = g.push_group(
                k,
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
            let exp = g.push_group(
                k,
                DType::F32,
                ScalarOp::Unary {
                    op: ScalarUnaryOp::Exp,
                    compute_dtype: DType::F32,
                },
                vec![],
                vec![InputRef::Affine {
                    base: neg,
                    stride: 1,
                }],
            );
            outputs.push(exp);
        }

        for out in &outputs {
            g.outputs.push(*out);
        }
        (g, vec![])
    }

    // ─── Verification helpers ──────────────────────────────────────────

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

    /// Compute imbalance ratio for a plan.
    fn compute_imbalance(phases: &[Phase]) -> f64 {
        let mut max_ratio: f64 = 1.0;
        for phase in phases {
            let lane_atoms: Vec<u64> = phase
                .spans
                .iter()
                .map(|s| s.graph.groups().iter().map(|g| g.count).sum::<u64>())
                .collect();
            let total: u64 = lane_atoms.iter().sum();
            if total == 0 {
                continue;
            }
            let mean = total as f64 / lane_atoms.len() as f64;
            let max = *lane_atoms.iter().max().unwrap() as f64;
            let ratio = max / mean;
            if ratio > max_ratio {
                max_ratio = ratio;
            }
        }
        max_ratio
    }

    // ─── Core invariant tests ──────────────────────────────────────────

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

    // ─── Depth and wavefront tests ─────────────────────────────────────

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

    // ─── Annealing-specific tests ──────────────────────────────────────

    #[test]
    fn test_annealing_single_chain() {
        // Single chain should go on lane 0.
        let chains = vec![ConstraintChain {
            groups: vec![0, 1],
            total_atoms: 200,
        }];
        let assignments = anneal_lane_assignments(&chains, 4, 42);
        assert_eq!(assignments.len(), 1);
        // Single chain, any lane is fine (it's just one chain).
    }

    #[test]
    fn test_annealing_perfect_balance() {
        // 4 equal chains, 4 lanes -> perfect balance.
        let chains: Vec<ConstraintChain> = (0..4)
            .map(|i| ConstraintChain {
                groups: vec![i],
                total_atoms: 1000,
            })
            .collect();

        let assignments = anneal_lane_assignments(&chains, 4, 42);
        assert_eq!(assignments.len(), 4);

        // Each lane should have exactly one chain.
        let mut lane_counts = vec![0u32; 4];
        for &lane in &assignments {
            lane_counts[lane] += 1;
        }
        // Every lane should have exactly 1 chain for perfect balance.
        for &count in &lane_counts {
            assert_eq!(
                count, 1,
                "Expected exactly 1 chain per lane for equal-sized chains"
            );
        }
    }

    #[test]
    fn test_annealing_imbalanced_input() {
        // Unequal chains: annealing should still find reasonable balance.
        let chains = vec![
            ConstraintChain {
                groups: vec![0],
                total_atoms: 1000,
            },
            ConstraintChain {
                groups: vec![1],
                total_atoms: 500,
            },
            ConstraintChain {
                groups: vec![2],
                total_atoms: 500,
            },
            ConstraintChain {
                groups: vec![3],
                total_atoms: 200,
            },
            ConstraintChain {
                groups: vec![4],
                total_atoms: 200,
            },
            ConstraintChain {
                groups: vec![5],
                total_atoms: 100,
            },
        ];

        let assignments = anneal_lane_assignments(&chains, 2, 42);
        assert_eq!(assignments.len(), 6);

        // Check balance.
        let mut lane_atoms = vec![0u64; 2];
        for (ci, &lane) in assignments.iter().enumerate() {
            lane_atoms[lane] += chains[ci].total_atoms;
        }
        let total = lane_atoms.iter().sum::<u64>() as f64;
        let max = *lane_atoms.iter().max().unwrap() as f64;
        let mean = total / 2.0;
        let ratio = max / mean;
        // Should achieve reasonable balance (< 1.5x).
        assert!(
            ratio < 1.5,
            "Imbalance ratio {} too high for annealing result",
            ratio
        );
    }

    #[test]
    fn test_annealing_many_small_chains() {
        // 32 small equal chains across 8 lanes.
        let chains: Vec<ConstraintChain> = (0..32)
            .map(|i| ConstraintChain {
                groups: vec![i],
                total_atoms: 100,
            })
            .collect();

        let assignments = anneal_lane_assignments(&chains, 8, 99);
        assert_eq!(assignments.len(), 32);

        let mut lane_atoms = vec![0u64; 8];
        for (ci, &lane) in assignments.iter().enumerate() {
            lane_atoms[lane] += chains[ci].total_atoms;
        }

        // Perfect balance = 400 per lane. Allow up to 500.
        let max = *lane_atoms.iter().max().unwrap();
        assert!(max <= 500, "Max lane load {} too high", max);
    }

    #[test]
    fn test_cost_function() {
        // Perfect balance.
        assert!((compute_cost(&[100, 100, 100, 100]) - 1.0).abs() < 0.01);

        // All on one lane.
        let cost_unbalanced = compute_cost(&[400, 0, 0, 0]);
        assert!(cost_unbalanced > 4.0); // imbalance = 4.0 + 3 * 2.0 empty penalty

        // Moderate imbalance.
        let cost_moderate = compute_cost(&[200, 100, 100, 0]);
        assert!(cost_moderate > 1.5);
        assert!(cost_moderate < cost_unbalanced);
    }

    // ─── Integration tests ─────────────────────────────────────────────

    #[test]
    fn test_parallel_balance_2_lanes() {
        let (g, inputs) = make_parallel();
        let output_ids: Vec<AtomId> = g.outputs.clone();
        let phases = plan(&g, 2, &inputs, &output_ids);

        verify_plan(&g, &phases, 2);
        verify_no_cross_span_reads(&phases);

        // Two independent chains of 1000 atoms each should balance well.
        let imbalance = compute_imbalance(&phases);
        assert!(
            imbalance < 2.0,
            "Parallel chains on 2 lanes should be balanced, got {}x",
            imbalance
        );
    }

    #[test]
    fn test_wide_matmul_balance() {
        let (g, inputs) = make_wide_matmul();
        let output_ids: Vec<AtomId> = g.outputs.clone();
        let phases = plan(&g, 4, &inputs, &output_ids);

        verify_plan(&g, &phases, 4);
        verify_no_cross_span_reads(&phases);

        // 16 independent chains across 4 lanes should balance well.
        let imbalance = compute_imbalance(&phases);
        assert!(
            imbalance < 2.0,
            "Wide matmul on 4 lanes should be balanced, got {}x",
            imbalance
        );
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
    fn test_deterministic() {
        // Same inputs should produce same output (PRNG is seeded deterministically).
        let (g, inputs) = make_matmul_like();
        let output_ids: Vec<AtomId> = g.outputs.clone();

        let phases1 = plan(&g, 4, &inputs, &output_ids);
        let phases2 = plan(&g, 4, &inputs, &output_ids);

        assert_eq!(phases1.len(), phases2.len());
        for (p1, p2) in phases1.iter().zip(phases2.iter()) {
            assert_eq!(p1.spans.len(), p2.spans.len());
            for (s1, s2) in p1.spans.iter().zip(p2.spans.iter()) {
                assert_eq!(s1.graph.num_groups(), s2.graph.num_groups());
                assert_eq!(s1.inputs.len(), s2.inputs.len());
                assert_eq!(s1.outputs.len(), s2.outputs.len());
            }
        }
    }

    #[test]
    fn test_rng_determinism() {
        let mut r1 = Rng::new(42);
        let mut r2 = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }

    #[test]
    fn test_union_find_basic() {
        let mut uf = UnionFind::new(5);
        uf.union(0, 1);
        uf.union(2, 3);
        assert_eq!(uf.find(0), uf.find(1));
        assert_eq!(uf.find(2), uf.find(3));
        assert_ne!(uf.find(0), uf.find(2));
        uf.union(1, 3);
        assert_eq!(uf.find(0), uf.find(3));
    }

    #[test]
    fn test_constraint_chains_independent() {
        // Two independent groups -> two separate chains.
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
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
        );

        let producers = vec![vec![], vec![]];
        let chains = build_constraint_chains(g.groups(), &[0, 1], &producers);
        assert_eq!(chains.len(), 2);
    }

    #[test]
    fn test_constraint_chains_dependent() {
        // Two dependent groups -> one chain.
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

        let producers = vec![vec![], vec![0]];
        let chains = build_constraint_chains(g.groups(), &[0, 1], &producers);
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].groups.len(), 2);
        assert_eq!(chains[0].total_atoms, 200);
    }
}
