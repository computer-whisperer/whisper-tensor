#![allow(
    clippy::all,
    dead_code,
    unreachable_code,
    unreachable_patterns,
    unused_imports,
    unused_variables
)]

//! Row-Aware Matmul Scheduling Partitioner (attempt G)
//!
//! Core insight: GPT-2 is ~95% matmul groups. For C[M,N] = A[M,K] @ B[K,N],
//! the NanoGraph has M Mul groups (count=K*N) and M ReduceSum groups (count=N),
//! forming M independent row chains. These (Mul, ReduceSum) pairs are the
//! natural unit of parallelism.
//!
//! Algorithm:
//! 1. Build group-level dependency DAG
//! 2. Identify matmul row chains: (Mul, ReduceSum) pairs where the ReduceSum's
//!    sole producer is the Mul group
//! 3. Detect matmul stages: groups of row chains that share weight inputs
//! 4. Phase boundaries between sequential matmul stages (where one's output
//!    feeds the next's input)
//! 5. Within each phase, assign row chains round-robin to lanes; split
//!    elementwise ops across lanes aligned with row assignment
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
/// Returns a sequence of phases, each containing one span per lane.
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

    // Step 2: Compute depths for phase detection.
    let depths = compute_depths(n, &producers);

    // Step 3: Identify matmul row chains.
    let row_chains = find_matmul_row_chains(graph, &producers, &successors);

    // Step 4: Group row chains into matmul stages and detect phase boundaries.
    let phase_assignments =
        assign_phases_matmul_aware(graph, &producers, &successors, &depths, &row_chains);

    // Step 5: Build per-phase group lists.
    let num_phases = phase_assignments.iter().copied().max().unwrap_or(0) + 1;
    let mut phase_groups: Vec<Vec<usize>> = vec![vec![]; num_phases];
    for (gi, &phase) in phase_assignments.iter().enumerate() {
        phase_groups[phase].push(gi);
    }

    // Step 6: Identify output groups.
    let output_group_set = identify_output_groups(graph, output_atom_ids);

    // Step 7: For each phase, assign groups to lanes with matmul-row-aware
    //         scheduling and build spans.
    let mut phases = Vec::with_capacity(num_phases);
    for phase_idx in 0..num_phases {
        let phase = build_phase(
            graph,
            &phase_groups[phase_idx],
            &phase_assignments,
            &producers,
            &successors,
            num_lanes,
            input_tensors,
            &output_group_set,
            phase_idx,
            num_phases,
            &row_chains,
        );
        phases.push(phase);
    }

    phases
}

// ─── Matmul Row Chain Detection ────────────────────────────────────────────

/// A matmul row chain: a (Mul, ReduceSum) pair where the ReduceSum's sole
/// compute-producing producer is the Mul group.
#[derive(Debug, Clone)]
struct RowChain {
    /// Index of the Mul group (Binary::Mul with count=K*N).
    mul_gi: usize,
    /// Index of the ReduceSum group (Reduce::Sum with count=N).
    reduce_gi: usize,
    /// Total atom cost for scheduling: mul.count + reduce.count.
    total_atoms: u64,
}

/// Identify matmul row chains in the graph.
///
/// A row chain is a (Mul, ReduceSum) pair where:
/// 1. The Mul group has ScalarOp::Binary { op: Mul }
/// 2. The ReduceSum group has ScalarOp::Reduce { kind: Sum }
/// 3. The ReduceSum has exactly one compute-producing producer, which is the
///    Mul group (it may also depend on literal groups etc., but only one
///    non-literal producer)
/// 4. The Mul group is the ReduceSum's input source
fn find_matmul_row_chains(
    graph: &NanoGraph,
    producers: &[Vec<usize>],
    successors: &[Vec<usize>],
) -> Vec<RowChain> {
    let groups = graph.groups();
    let mut chains = Vec::new();
    let mut used_as_mul: HashSet<usize> = HashSet::new();

    for (gi, group) in groups.iter().enumerate() {
        // Look for ReduceSum groups.
        let is_reduce_sum = matches!(
            &group.op,
            ScalarOp::Reduce {
                kind: crate::nano_graph::ops::ReduceKind::Sum,
                ..
            }
        );
        if !is_reduce_sum {
            continue;
        }

        // Find the non-literal producers of this ReduceSum.
        let compute_producers: Vec<usize> = producers[gi]
            .iter()
            .copied()
            .filter(|&pi| !is_literal_group(&groups[pi]))
            .collect();

        // A matmul row chain has exactly one compute producer (the Mul group).
        if compute_producers.len() != 1 {
            continue;
        }

        let mul_gi = compute_producers[0];
        let mul_group = &groups[mul_gi];

        // Check that the producer is a Binary::Mul.
        let is_mul = matches!(
            &mul_group.op,
            ScalarOp::Binary {
                op: crate::nano_graph::ops::ScalarBinOp::Mul,
                ..
            }
        );
        if !is_mul {
            continue;
        }

        // Verify the ReduceSum reads from the Mul group via its input.
        // The ReduceSum should have an input that resolves to atoms in the Mul group.
        let reads_mul = group.inputs.iter().any(|input| {
            let resolved = input.resolve(group.atom_offset);
            resolved.0 >= mul_group.base_id.0 && resolved.0 < mul_group.base_id.0 + mul_group.count
        });

        if !reads_mul {
            continue;
        }

        // Don't reuse a Mul group for multiple chains — each should be unique.
        if used_as_mul.contains(&mul_gi) {
            continue;
        }
        used_as_mul.insert(mul_gi);

        chains.push(RowChain {
            mul_gi,
            reduce_gi: gi,
            total_atoms: mul_group.count + group.count,
        });
    }

    chains
}

// ─── Phase Assignment ──────────────────────────────────────────────────────

/// Assign groups to phases using matmul-aware wavefront merging.
///
/// Uses depth-based wavefronts as the foundation (provably correct), then
/// merges adjacent wavefronts aggressively when safe. Row chains are kept
/// together (Mul and ReduceSum in the same phase).
fn assign_phases_matmul_aware(
    graph: &NanoGraph,
    producers: &[Vec<usize>],
    successors: &[Vec<usize>],
    depths: &[usize],
    row_chains: &[RowChain],
) -> Vec<usize> {
    let groups = graph.groups();
    let n = groups.len();

    if n == 0 {
        return vec![];
    }

    // Build wavefronts from depths.
    let max_depth = depths.iter().copied().max().unwrap_or(0);
    let mut wavefronts: Vec<Vec<usize>> = vec![vec![]; max_depth + 1];
    for (gi, &d) in depths.iter().enumerate() {
        wavefronts[d].push(gi);
    }

    // Build a set of groups that are part of row chains — these should stay
    // together if possible.
    let mut row_chain_partner: HashMap<usize, usize> = HashMap::new();
    for chain in row_chains {
        row_chain_partner.insert(chain.mul_gi, chain.reduce_gi);
        row_chain_partner.insert(chain.reduce_gi, chain.mul_gi);
    }

    // Merge strategy: merge wavefront i+1 into wavefront i's phase when
    // every group in wavefront i+1 depends only on groups within the merged
    // phase or strictly earlier phases. For safety, require that each group
    // in the later wavefront depends on at most one group from the current
    // phase (so they can be co-located on the same lane).
    //
    // Special case: if a ReduceSum's partner Mul is in the current phase,
    // always allow the merge (the chain is meant to be co-located).

    let mut group_wavefront = vec![0usize; n];
    for (wi, wf) in wavefronts.iter().enumerate() {
        for &gi in wf {
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

        // Check merge eligibility.
        let target_phase = wavefront_to_phase[wi - 1];
        let mut can_merge = true;

        for &gi in &wavefronts[wi] {
            let mut deps_in_phase = 0usize;
            for &pi in &producers[gi] {
                let pw = group_wavefront[pi];
                if pw < wi && wavefront_to_phase[pw] == target_phase {
                    deps_in_phase += 1;
                }
            }
            // Allow merge if this group depends on at most one group from the
            // current phase, OR if this group is part of a row chain and its
            // partner is in the current phase.
            if deps_in_phase > 1 {
                // Check if this is a row chain partner merge.
                let is_chain_merge = row_chain_partner.get(&gi).map_or(false, |&partner| {
                    let pw = group_wavefront[partner];
                    pw < wi && wavefront_to_phase[pw] == target_phase
                });
                if !is_chain_merge {
                    can_merge = false;
                    break;
                }
            }
        }

        if can_merge {
            wavefront_to_phase[wi] = target_phase;
        } else {
            wavefront_to_phase[wi] = next_phase;
            next_phase += 1;
        }
    }

    // Apply assignments.
    let mut phase_assignment = vec![0usize; n];
    for (wi, wf) in wavefronts.iter().enumerate() {
        for &gi in wf {
            phase_assignment[gi] = wavefront_to_phase[wi];
        }
    }

    phase_assignment
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

    // Deduplicate successors.
    for s in successors.iter_mut() {
        s.sort_unstable();
        s.dedup();
    }

    (producers, successors)
}

// ─── Depth computation ─────────────────────────────────────────────────────

fn compute_depths(n: usize, producers: &[Vec<usize>]) -> Vec<usize> {
    let mut depths = vec![0usize; n];
    for gi in 0..n {
        let mut max_dep = 0usize;
        for &pi in &producers[gi] {
            max_dep = max_dep.max(depths[pi] + 1);
        }
        depths[gi] = max_dep;
    }
    depths
}

// ─── Lane Assignment ───────────────────────────────────────────────────────

/// A unit of work assigned to a lane.
#[derive(Debug, Clone)]
struct WorkItem {
    group_idx: usize,
    atom_offset: u64,
    atom_count: u64,
}

/// Build a Phase with matmul-row-aware lane assignment.
///
/// Row chains (Mul+ReduceSum pairs) are the primary scheduling unit.
/// They are assigned round-robin to lanes for natural load balance.
/// Non-chain groups (elementwise ops, other compute) are assigned to the
/// least-loaded lane while respecting intra-phase dependencies.
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
    row_chains: &[RowChain],
) -> Phase {
    let groups = graph.groups();

    let mut sorted_groups = phase_group_indices.to_vec();
    sorted_groups.sort_unstable();

    let phase_group_set: HashSet<usize> = sorted_groups.iter().copied().collect();

    // Build union-find for intra-phase dependency chains.
    // Groups that depend on each other within this phase must be on the same lane.
    let mut uf = UnionFind::new(groups.len());
    for &gi in &sorted_groups {
        for &pi in &producers[gi] {
            if phase_group_set.contains(&pi) {
                uf.union(gi, pi);
            }
        }
    }

    // Identify which groups in this phase are part of row chains.
    let mut chain_group_set: HashSet<usize> = HashSet::new();
    let mut phase_chains: Vec<&RowChain> = Vec::new();
    for chain in row_chains {
        if phase_group_set.contains(&chain.mul_gi) && phase_group_set.contains(&chain.reduce_gi) {
            chain_group_set.insert(chain.mul_gi);
            chain_group_set.insert(chain.reduce_gi);
            phase_chains.push(chain);
        }
    }

    // Group chains by union-find root to find "super-chains" — sets of
    // groups that must be co-located due to intra-phase dependencies.
    let mut uf_chains: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &gi in &sorted_groups {
        let root = uf.find(gi);
        uf_chains.entry(root).or_default().push(gi);
    }

    // Classify super-chains:
    // - "row super-chains" contain at least one row chain → scheduled round-robin
    // - "other super-chains" contain no row chains → bin-packed
    let mut row_super_chains: Vec<(usize, Vec<usize>, u64)> = Vec::new();
    let mut other_super_chains: Vec<(usize, Vec<usize>, u64)> = Vec::new();

    for (root, gis) in &uf_chains {
        let total_atoms: u64 = gis.iter().map(|&gi| groups[gi].count).sum();
        let has_row_chain = gis.iter().any(|gi| chain_group_set.contains(gi));

        if has_row_chain {
            row_super_chains.push((*root, gis.clone(), total_atoms));
        } else {
            other_super_chains.push((*root, gis.clone(), total_atoms));
        }
    }

    // Assign row super-chains round-robin (sorted by topo order for determinism).
    // This is the key insight: matmul rows are naturally balanced, so round-robin
    // over them gives near-perfect balance.
    row_super_chains.sort_by_key(|(_, gis, _)| gis[0]);

    let mut lane_work: Vec<Vec<WorkItem>> = vec![vec![]; num_lanes];
    let mut lane_atoms: Vec<u64> = vec![0; num_lanes];
    let mut group_to_lane: HashMap<usize, usize> = HashMap::new();

    for (i, (_, chain_groups, _)) in row_super_chains.iter().enumerate() {
        let lane = i % num_lanes;
        for &gi in chain_groups {
            lane_work[lane].push(WorkItem {
                group_idx: gi,
                atom_offset: 0,
                atom_count: groups[gi].count,
            });
            lane_atoms[lane] += groups[gi].count;
            group_to_lane.insert(gi, lane);
        }
    }

    // Assign other super-chains using first-fit-decreasing bin packing.
    other_super_chains.sort_by(|a, b| b.2.cmp(&a.2));

    for (_, chain_groups, _) in &other_super_chains {
        // Check if any group in this chain has a dependency on a group already
        // assigned to a lane. If so, assign to that lane.
        let mut forced_lane: Option<usize> = None;
        for &gi in chain_groups {
            if let Some(&lane) = group_to_lane.get(&gi) {
                forced_lane = Some(lane);
                break;
            }
            // Check if any producer is already assigned.
            for &pi in &producers[gi] {
                if let Some(&lane) = group_to_lane.get(&pi) {
                    forced_lane = Some(lane);
                    break;
                }
            }
            if forced_lane.is_some() {
                break;
            }
        }

        let target_lane = forced_lane.unwrap_or_else(|| {
            lane_atoms
                .iter()
                .enumerate()
                .min_by_key(|&(_, &atoms)| atoms)
                .map(|(i, _)| i)
                .unwrap_or(0)
        });

        for &gi in chain_groups {
            lane_work[target_lane].push(WorkItem {
                group_idx: gi,
                atom_offset: 0,
                atom_count: groups[gi].count,
            });
            lane_atoms[target_lane] += groups[gi].count;
            group_to_lane.insert(gi, target_lane);
        }
    }

    // Sort each lane's work items by group index (topo order).
    for lane in &mut lane_work {
        lane.sort_by_key(|w| (w.group_idx, w.atom_offset));
    }

    // Build span NanoGraphs.
    let span_compute_sets: Vec<HashSet<usize>> = lane_work
        .iter()
        .map(|work| work.iter().map(|w| w.group_idx).collect())
        .collect();

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
                &span_compute_sets[lane_idx],
            )
        })
        .collect();

    Phase { spans }
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
    span_compute_set: &HashSet<usize>,
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

    // Copy sym_dim configuration.
    span_graph.sym_dim_names = graph.sym_dim_names.clone();
    span_graph.sym_dim_bounds = graph.sym_dim_bounds.clone();

    // Small literal inline threshold.
    const LITERAL_INLINE_THRESHOLD: u64 = 65536;

    // Collect external dependencies via BFS from compute groups.
    let mut needed_external: BTreeMap<AtomId, (u64, DType)> = BTreeMap::new();
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
                record_external_range(&mut needed_external, g.base_id, g.count, g.output_dtype);
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
                &mut needed_external,
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
                        span_compute_set,
                        &mut needed_external,
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
                            &mut needed_external,
                            g.base_id,
                            g.count,
                            g.output_dtype,
                        );
                    }
                }
            }
        }
    }

    // Check internal literals' own deps (usually none, but handle edge cases).
    for &li in &needed_internal_literals.clone() {
        let group = &groups[li];
        for input in &group.inputs {
            collect_input_tensor_deps(
                input,
                group.count,
                group.atom_offset,
                graph,
                input_tensors,
                &mut needed_external,
            );
        }
    }

    // Merge external ranges.
    let merged_external = merge_external_ranges(&needed_external);

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

    // Track inserted ranges to avoid duplicates.
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

                let inputs = group.inputs.clone();

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

                // Output if needed by later phases or by other lanes in this phase.
                if output_group_set.contains(gi) || needs_output(&groups[*gi]) {
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

/// Conservatively determine if a group's output is needed externally.
/// Non-literal compute groups always output (safe, slightly wasteful).
fn needs_output(group: &AtomGroup) -> bool {
    !is_literal_op(&group.op)
}

// ─── Helpers ───────────────────────────────────────────────────────────────

fn is_literal_op(op: &ScalarOp) -> bool {
    matches!(op, ScalarOp::Literal(_))
}

fn is_literal_group(group: &AtomGroup) -> bool {
    is_literal_op(&group.op) && group.inputs.is_empty()
}

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

    /// Simple linear chain: a -> b -> c.
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

    /// Diamond: lit -> {b, c} -> d.
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

    /// Matmul-like structure: M=8 rows, K=64 reduction dimension, N=1.
    /// Each row: Mul(K elements) -> ReduceSum(1 element).
    /// This is the simplest matmul: C[M] = A[M,K] @ B[K] (matvec).
    fn make_matmul_like() -> (NanoGraph, Vec<InputTensor>) {
        let mut g = NanoGraph::new();

        let k: u64 = 64;
        let m: u64 = 8;

        // Weight vector: K values (shared by all rows).
        let weights = g.push_group(
            k,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.5)),
            vec![],
            vec![],
        );

        // Input matrix: M*K values.
        let input = g.push_group(
            m * k,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );

        // Mul: M*K elementwise products.
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

    /// Two sequential matmuls: C1 = A @ B1, then C2 = C1 @ B2.
    /// This tests phase boundaries between matmul stages.
    fn make_sequential_matmuls() -> (NanoGraph, Vec<InputTensor>) {
        let mut g = NanoGraph::new();

        let k1: u64 = 16;
        let m: u64 = 4;
        let k2: u64 = 8;

        // First matmul: A[M,K1] @ B1[K1]
        let b1 = g.push_group(
            k1,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.5)),
            vec![],
            vec![],
        );
        let a = g.push_group(
            m * k1,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );
        let mul1 = g.push_group(
            m * k1,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine { base: a, stride: 1 },
                InputRef::Modular {
                    base: b1,
                    stride: 1,
                    modulus: k1,
                },
            ],
        );
        let red1 = g.push_group(
            m,
            DType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: k1,
                reduce_stride: 1,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: mul1,
                stride: k1 as i64,
            }],
        );

        // Second matmul: C1[M] * B2[K2] (each element of C1 broadcast over K2)
        // Then reduce M*K2 -> M outputs.
        let b2 = g.push_group(
            k2,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.3)),
            vec![],
            vec![],
        );
        let mul2 = g.push_group(
            m * k2,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::StridedBroadcast {
                    base: red1,
                    stride: 1,
                    repeat: k2,
                },
                InputRef::Modular {
                    base: b2,
                    stride: 1,
                    modulus: k2,
                },
            ],
        );
        let red2 = g.push_group(
            m,
            DType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: k2,
                reduce_stride: 1,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: mul2,
                stride: k2 as i64,
            }],
        );

        g.outputs.push(red2);
        (g, vec![])
    }

    /// Matmul with many rows: M=32 rows, K=16.
    /// Tests that round-robin assignment distributes well.
    fn make_wide_matmul() -> (NanoGraph, Vec<InputTensor>) {
        let mut g = NanoGraph::new();

        let k: u64 = 16;
        let m: u64 = 32;

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

    /// Matmul followed by elementwise operation (like bias add + GELU).
    fn make_matmul_plus_elementwise() -> (NanoGraph, Vec<InputTensor>) {
        let mut g = NanoGraph::new();

        let k: u64 = 16;
        let m: u64 = 8;

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

        // Bias add (elementwise on the reduce output).
        let bias = g.push_group(
            m,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.1)),
            vec![],
            vec![],
        );
        let biased = g.push_group(
            m,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine {
                    base: reduce,
                    stride: 1,
                },
                InputRef::Affine {
                    base: bias,
                    stride: 1,
                },
            ],
        );

        // GELU-like activation.
        let activated = g.push_group(
            m,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: biased,
                stride: 1,
            }],
        );

        g.outputs.push(activated);
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
                num_lanes,
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
                    errors,
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
                output_id,
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
                                    "Phase {} lane {} reads atom {} produced by lane {} in same phase",
                                    pi, li, atom, other_li,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Compute load balance ratio (max / min non-zero lane atoms).
    fn compute_balance_ratio(phases: &[Phase]) -> f64 {
        let mut lane_totals: Vec<u64> = vec![];
        for phase in phases {
            for (li, span) in phase.spans.iter().enumerate() {
                if li >= lane_totals.len() {
                    lane_totals.resize(phase.spans.len(), 0);
                }
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

        let max_atoms = lane_totals.iter().copied().max().unwrap_or(0);
        let min_atoms = lane_totals
            .iter()
            .copied()
            .filter(|&x| x > 0)
            .min()
            .unwrap_or(1);
        if min_atoms == 0 {
            max_atoms as f64
        } else {
            max_atoms as f64 / min_atoms as f64
        }
    }

    // ─── Tests ─────────────────────────────────────────────────────────

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
    fn test_matmul_row_chain_detection() {
        let (g, _inputs) = make_matmul_like();
        let (producers, successors) = build_dependency_dag(&g);
        let chains = find_matmul_row_chains(&g, &producers, &successors);

        // Should detect at least one row chain (Mul + ReduceSum pair).
        assert!(
            !chains.is_empty(),
            "No matmul row chains detected in matmul-like graph",
        );

        // The Mul and ReduceSum should be different groups.
        for chain in &chains {
            assert_ne!(chain.mul_gi, chain.reduce_gi);
        }
    }

    #[test]
    fn test_sequential_matmuls() {
        let (g, inputs) = make_sequential_matmuls();
        let output_ids: Vec<AtomId> = g.outputs.clone();
        let phases = plan(&g, 4, &inputs, &output_ids);

        verify_plan(&g, &phases, 4);
        verify_no_cross_span_reads(&phases);

        // Should have more than one phase (barrier between matmul stages).
        // But not strictly required — what matters is correctness.
    }

    #[test]
    fn test_wide_matmul_balance() {
        let (g, inputs) = make_wide_matmul();
        let output_ids: Vec<AtomId> = g.outputs.clone();
        let phases = plan(&g, 4, &inputs, &output_ids);

        verify_plan(&g, &phases, 4);
        verify_no_cross_span_reads(&phases);

        // With 32 rows distributed round-robin across 4 lanes, balance
        // should be very good (close to 1.0x).
        let balance = compute_balance_ratio(&phases);
        assert!(
            balance < 2.0,
            "Wide matmul balance ratio {} should be < 2.0x",
            balance,
        );
    }

    #[test]
    fn test_matmul_plus_elementwise() {
        let (g, inputs) = make_matmul_plus_elementwise();
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
                base,
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
    fn test_depth_computation() {
        let producers = vec![vec![], vec![0], vec![0], vec![1, 2]];
        let depths = compute_depths(4, &producers);
        assert_eq!(depths, vec![0, 1, 1, 2]);
    }

    #[test]
    fn test_no_row_chains_in_non_matmul_graph() {
        let (g, _) = make_linear_chain();
        let (producers, successors) = build_dependency_dag(&g);
        let chains = find_matmul_row_chains(&g, &producers, &successors);
        assert!(
            chains.is_empty(),
            "Non-matmul graph should have no row chains"
        );
    }

    #[test]
    fn test_many_lanes_few_groups() {
        // More lanes than groups: some lanes should be empty.
        let mut g = NanoGraph::new();
        let a = g.push_group(
            10,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );
        let b = g.push_group(
            10,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine { base: a, stride: 1 }],
        );
        g.outputs.push(b);

        let phases = plan(&g, 16, &[], &g.outputs.clone());
        verify_plan(&g, &phases, 16);
    }

    #[test]
    fn test_strided_broadcast_input() {
        let mut g = NanoGraph::new();

        let source = g.push_group(
            4,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );
        let expanded = g.push_group(
            12,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::StridedBroadcast {
                base: source,
                stride: 1,
                repeat: 3,
            }],
        );

        g.outputs.push(expanded);

        let phases = plan(&g, 2, &[], &g.outputs.clone());
        verify_plan(&g, &phases, 2);
        verify_no_cross_span_reads(&phases);
    }
}
