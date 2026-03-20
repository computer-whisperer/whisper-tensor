#![allow(
    clippy::all,
    dead_code,
    unreachable_patterns,
    unused_imports,
    unused_variables,
    unused_assignments
)]

//! Hierarchical Decomposition Partitioner (variant D).
//!
//! Divides the NanoGraph into phases and lanes using a multi-scale approach:
//! 1. Build a group-level dependency DAG
//! 2. Identify "super-groups" — tightly connected clusters of groups
//!    (found by removing high-fanout broadcast edges and taking connected components)
//! 3. Build a coarse DAG of super-groups and compute depth
//! 4. Schedule super-groups into phases based on dependency depth
//! 5. Within each phase, partition super-groups across lanes for load balance
//! 6. Build span NanoGraphs preserving the main graph's AtomId space

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::dtype::DType;
use crate::graph::GlobalId;
use crate::nano_graph::pattern::InputTensor;
use crate::nano_graph::{AtomGroup, AtomId, AtomRange, InputRef, NanoGraph, ScalarOp, SymDim};

use super::types::{Phase, Span};

// ─── Public API ─────────────────────────────────────────────────────────────

/// Partition a NanoGraph into phases and spans for multi-lane execution.
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

    // Step 1: Build group-level dependency DAG (predecessors for each group).
    let predecessors = build_predecessor_sets(graph);

    // Step 2: Identify super-groups via hierarchical clustering.
    let super_groups = find_super_groups(graph, &predecessors);

    // Step 3: Build coarse DAG of super-groups and compute depth.
    let (_, sg_depth) = build_super_group_dag(graph, &predecessors, &super_groups);

    // Step 4: Assign super-groups to phases based on depth.
    let phase_assignments = assign_phases(&sg_depth);

    // Step 5: Within each phase, assign super-groups to lanes for load balance.
    let num_phases = phase_assignments.iter().copied().max().unwrap_or(0) + 1;
    let lane_assignments = assign_lanes(
        graph,
        &super_groups,
        &phase_assignments,
        num_phases,
        num_lanes,
    );

    // Step 6: Build the final Phase/Span structures.
    build_phases(
        graph,
        input_tensors,
        output_atom_ids,
        &super_groups,
        &predecessors,
        &phase_assignments,
        &lane_assignments,
        num_phases,
        num_lanes,
    )
}

// ─── Step 1: Build predecessor sets ─────────────────────────────────────────

/// For each group index, compute the set of group indices it depends on.
fn build_predecessor_sets(graph: &NanoGraph) -> Vec<HashSet<usize>> {
    let groups = graph.groups();
    let n = groups.len();
    let mut preds = Vec::with_capacity(n);
    for (gi, group) in groups.iter().enumerate() {
        let mut set = HashSet::new();
        graph.collect_all_producer_indices(group, gi, &mut set);
        preds.push(set);
    }
    preds
}

// ─── Step 2: Find super-groups ──────────────────────────────────────────────

/// Identify super-groups: clusters of tightly connected groups.
///
/// Strategy: compute fanout for each group (how many downstream groups consume it).
/// Groups with high fanout are "hubs" (typically weight tensors, shared constants).
/// Remove hub edges and find connected components in the undirected residual graph.
/// This naturally clusters sequential computation chains while separating
/// independent branches that only share weights.
fn find_super_groups(graph: &NanoGraph, predecessors: &[HashSet<usize>]) -> Vec<Vec<usize>> {
    let groups = graph.groups();
    let n = groups.len();

    if n == 0 {
        return vec![];
    }

    // Compute fanout (number of downstream consumers) for each group.
    let mut fanout = vec![0u32; n];
    for preds in predecessors {
        for &pi in preds {
            fanout[pi] += 1;
        }
    }

    // Determine hub threshold: groups consumed by many others are hubs.
    // A reasonable threshold: more than sqrt(n) consumers, or more than 8.
    let hub_threshold = ((n as f64).sqrt() as u32).max(8);

    // A group is treated as a hub node if it has high fanout OR is a Literal
    // (weights/constants are naturally shared and shouldn't glue unrelated
    // computation chains together).
    let is_hub: Vec<bool> = groups
        .iter()
        .enumerate()
        .map(|(gi, g)| fanout[gi] >= hub_threshold || matches!(g.op, ScalarOp::Literal(_)))
        .collect();

    // Build undirected adjacency using only non-hub edges.
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
    for (gi, preds) in predecessors.iter().enumerate() {
        if is_hub[gi] {
            continue;
        }
        for &pi in preds {
            if is_hub[pi] {
                continue;
            }
            adj[gi].push(pi);
            adj[pi].push(gi);
        }
    }

    // Find connected components via BFS.
    let mut component = vec![usize::MAX; n];
    let mut next_component = 0usize;
    for start in 0..n {
        if component[start] != usize::MAX {
            continue;
        }
        let cid = next_component;
        next_component += 1;
        let mut stack = vec![start];
        component[start] = cid;
        while let Some(node) = stack.pop() {
            for &neighbor in &adj[node] {
                if component[neighbor] == usize::MAX {
                    component[neighbor] = cid;
                    stack.push(neighbor);
                }
            }
        }
    }

    // Collect components, preserving group ordering within each.
    let mut sg_map: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (gi, &cid) in component.iter().enumerate() {
        sg_map.entry(cid).or_default().push(gi);
    }

    sg_map.into_values().collect()
}

// ─── Step 3: Build super-group DAG ──────────────────────────────────────────

/// Build the DAG between super-groups and compute depth (longest path from roots).
fn build_super_group_dag(
    graph: &NanoGraph,
    predecessors: &[HashSet<usize>],
    super_groups: &[Vec<usize>],
) -> (Vec<HashSet<usize>>, Vec<usize>) {
    let groups = graph.groups();
    let n_sg = super_groups.len();

    // Map: group_index -> super_group_index.
    let mut group_to_sg = vec![0usize; groups.len()];
    for (sg_idx, members) in super_groups.iter().enumerate() {
        for &gi in members {
            group_to_sg[gi] = sg_idx;
        }
    }

    // Build super-group predecessor sets.
    let mut sg_preds: Vec<HashSet<usize>> = vec![HashSet::new(); n_sg];
    for (sg_idx, members) in super_groups.iter().enumerate() {
        for &gi in members {
            for &pi in &predecessors[gi] {
                let pred_sg = group_to_sg[pi];
                if pred_sg != sg_idx {
                    sg_preds[sg_idx].insert(pred_sg);
                }
            }
        }
    }

    // Compute depth via Kahn's algorithm (topological BFS).
    let mut sg_successors: Vec<Vec<usize>> = vec![vec![]; n_sg];
    let mut in_degree = vec![0usize; n_sg];
    for (sg_idx, preds) in sg_preds.iter().enumerate() {
        for &pred in preds {
            sg_successors[pred].push(sg_idx);
        }
        in_degree[sg_idx] = preds.len();
    }

    let mut depth = vec![0usize; n_sg];
    let mut queue: Vec<usize> = (0..n_sg).filter(|&i| in_degree[i] == 0).collect();
    while let Some(sg) = queue.pop() {
        for &succ in &sg_successors[sg] {
            depth[succ] = depth[succ].max(depth[sg] + 1);
            in_degree[succ] -= 1;
            if in_degree[succ] == 0 {
                queue.push(succ);
            }
        }
    }

    (sg_preds, depth)
}

// ─── Step 4: Assign phases ──────────────────────────────────────────────────

/// Assign each super-group to a phase based on its depth.
/// Each depth level becomes a separate phase.
fn assign_phases(sg_depth: &[usize]) -> Vec<usize> {
    sg_depth.to_vec()
}

// ─── Step 5: Assign lanes ───────────────────────────────────────────────────

/// Within each phase, assign super-groups to lanes for load balance.
/// Uses greedy bin-packing: largest-first, assign to least-loaded lane.
fn assign_lanes(
    graph: &NanoGraph,
    super_groups: &[Vec<usize>],
    phase_assignments: &[usize],
    num_phases: usize,
    num_lanes: usize,
) -> Vec<usize> {
    let groups = graph.groups();
    let n_sg = super_groups.len();
    let mut lane_assignments = vec![0usize; n_sg];

    for phase in 0..num_phases {
        let mut phase_sgs: Vec<(usize, u64)> = (0..n_sg)
            .filter(|&sg| phase_assignments[sg] == phase)
            .map(|sg| {
                let atom_count: u64 = super_groups[sg].iter().map(|&gi| groups[gi].count).sum();
                (sg, atom_count)
            })
            .collect();

        // Sort largest first for better bin-packing.
        phase_sgs.sort_by(|a, b| b.1.cmp(&a.1));

        let mut lane_load = vec![0u64; num_lanes];
        for (sg, atom_count) in phase_sgs {
            let best_lane = lane_load
                .iter()
                .enumerate()
                .min_by_key(|&(_, &load)| load)
                .map(|(i, _)| i)
                .unwrap_or(0);
            lane_assignments[sg] = best_lane;
            lane_load[best_lane] += atom_count;
        }
    }

    lane_assignments
}

// ─── Step 6: Build phases ───────────────────────────────────────────────────

/// Build the final Phase/Span structures.
fn build_phases(
    graph: &NanoGraph,
    input_tensors: &[InputTensor],
    output_atom_ids: &[AtomId],
    super_groups: &[Vec<usize>],
    predecessors: &[HashSet<usize>],
    phase_assignments: &[usize],
    lane_assignments: &[usize],
    num_phases: usize,
    num_lanes: usize,
) -> Vec<Phase> {
    let groups = graph.groups();

    // For each (phase, lane), collect group indices (sorted for topo order).
    let mut phase_lane_groups: Vec<Vec<Vec<usize>>> = vec![vec![vec![]; num_lanes]; num_phases];
    for (sg_idx, members) in super_groups.iter().enumerate() {
        let phase = phase_assignments[sg_idx];
        let lane = lane_assignments[sg_idx];
        phase_lane_groups[phase][lane].extend(members);
    }
    // Sort within each (phase, lane) for topological order.
    for phase_lanes in &mut phase_lane_groups {
        for lane_groups in phase_lanes {
            lane_groups.sort();
        }
    }

    // Pre-compute: for each group, which other groups consume it (successors).
    // This is used to decide which groups need to be in span outputs.
    let mut successors: Vec<Vec<usize>> = vec![vec![]; groups.len()];
    for (gi, preds) in predecessors.iter().enumerate() {
        for &pi in preds {
            successors[pi].push(gi);
        }
    }

    // Build output atom set for quick lookup.
    let output_atom_set: HashSet<u64> = output_atom_ids.iter().map(|a| a.0).collect();

    let mut phases = Vec::with_capacity(num_phases);

    for phase_idx in 0..num_phases {
        let mut spans = Vec::with_capacity(num_lanes);

        for lane_idx in 0..num_lanes {
            let lane_group_indices = &phase_lane_groups[phase_idx][lane_idx];

            if lane_group_indices.is_empty() {
                spans.push(empty_span());
                continue;
            }

            let lane_group_set: HashSet<usize> = lane_group_indices.iter().copied().collect();

            // Determine external group dependencies (predecessors not in this span).
            let mut external_inputs: BTreeMap<u64, AtomRange> = BTreeMap::new();
            let mut needed_input_tensors: BTreeSet<usize> = BTreeSet::new();

            for &gi in lane_group_indices {
                let group = &groups[gi];

                // Collect producer groups that are external to this span.
                for &pi in &predecessors[gi] {
                    if !lane_group_set.contains(&pi) {
                        let pg = &groups[pi];
                        external_inputs
                            .entry(pg.base_id.0)
                            .or_insert_with(|| AtomRange {
                                base: pg.base_id,
                                count: pg.count,
                                dtype: pg.output_dtype,
                            });
                    }
                }

                // Collect input tensor references.
                for input_ref in &group.inputs {
                    collect_input_tensor_refs(
                        graph,
                        input_ref,
                        group.count,
                        group.atom_offset,
                        &mut needed_input_tensors,
                    );
                }

                // Check reduce stride ranges for input tensor references.
                if let ScalarOp::Reduce {
                    reduce_count,
                    reduce_stride,
                    ..
                } = &group.op
                {
                    if *reduce_count > 1 && *reduce_stride != 0 {
                        for input_ref in &group.inputs {
                            check_reduce_input_tensor_refs(
                                input_ref,
                                group.atom_offset,
                                group.count,
                                *reduce_count,
                                *reduce_stride,
                                input_tensors,
                                &mut needed_input_tensors,
                            );
                        }
                    }
                }

                // Check IndirectLoad table references.
                if let ScalarOp::IndirectLoad { table_base } = &group.op {
                    for (it_idx, it) in input_tensors.iter().enumerate() {
                        if table_base.0 >= it.base_id.0 && table_base.0 < it.base_id.0 + it.count {
                            needed_input_tensors.insert(it_idx);
                        }
                    }
                }
            }

            // Build span input ranges: input tensors + external groups.
            let mut span_input_ranges: Vec<AtomRange> = Vec::new();
            for &it_idx in &needed_input_tensors {
                let it = &input_tensors[it_idx];
                span_input_ranges.push(AtomRange {
                    base: it.base_id,
                    count: it.count,
                    dtype: it.dtype,
                });
            }
            for (_, range) in &external_inputs {
                span_input_ranges.push(range.clone());
            }

            // Determine span outputs: groups needed by other spans or model outputs.
            let mut span_output_ranges: Vec<AtomRange> = Vec::new();
            for &gi in lane_group_indices {
                let g = &groups[gi];
                let is_model_output =
                    (0..g.count).any(|i| output_atom_set.contains(&(g.base_id.0 + i)));
                let needed_externally = successors[gi]
                    .iter()
                    .any(|&succ| !lane_group_set.contains(&succ));
                if is_model_output || needed_externally {
                    span_output_ranges.push(AtomRange {
                        base: g.base_id,
                        count: g.count,
                        dtype: g.output_dtype,
                    });
                }
            }

            // Build the span NanoGraph.
            let span_graph = build_span_nanograph(
                graph,
                input_tensors,
                lane_group_indices,
                &external_inputs,
                &needed_input_tensors,
            );

            spans.push(Span {
                graph: span_graph,
                inputs: span_input_ranges,
                outputs: span_output_ranges,
            });
        }

        phases.push(Phase { spans });
    }

    // Remove empty trailing phases.
    while phases.len() > 1
        && phases
            .last()
            .map_or(false, |p| p.spans.iter().all(|s| s.graph.num_groups() == 0))
    {
        phases.pop();
    }

    phases
}

// ─── Span NanoGraph construction ────────────────────────────────────────────

/// Build a NanoGraph for a span that preserves the main graph's AtomId space.
///
/// Collects all atom ID ranges this span needs (input tensors + external group
/// ranges + the span's own groups), sorts by base_id, and adds them sequentially
/// to a new NanoGraph. Gaps are filled with dummy input tensor entries to advance
/// the internal ID counter.
fn build_span_nanograph(
    graph: &NanoGraph,
    input_tensors: &[InputTensor],
    group_indices: &[usize],
    external_inputs: &BTreeMap<u64, AtomRange>,
    needed_input_tensors: &BTreeSet<usize>,
) -> NanoGraph {
    let groups = graph.groups();

    // Collect all ranges to place, tagged by kind.
    enum RangeKind {
        InputTensor(usize),
        ExternalGroup,
        OwnGroup(usize),
    }

    let mut ranges: Vec<(u64, u64, RangeKind)> = Vec::new();

    for &it_idx in needed_input_tensors {
        let it = &input_tensors[it_idx];
        ranges.push((it.base_id.0, it.count, RangeKind::InputTensor(it_idx)));
    }

    for (_, range) in external_inputs {
        ranges.push((range.base.0, range.count, RangeKind::ExternalGroup));
    }

    for &gi in group_indices {
        let g = &groups[gi];
        ranges.push((g.base_id.0, g.count, RangeKind::OwnGroup(gi)));
    }

    // Sort by base_id.
    ranges.sort_by_key(|r| r.0);

    // Build span NanoGraph.
    let mut span_graph = NanoGraph::new();
    span_graph.sym_dim_names = graph.sym_dim_names.clone();
    span_graph.sym_dim_bounds = graph.sym_dim_bounds.clone();

    let mut current_id: u64 = 0;

    for (base, count, kind) in &ranges {
        let base = *base;
        let count = *count;

        // Skip ranges fully behind current_id (overlap).
        if base + count <= current_id {
            continue;
        }
        // Handle partial overlap: advance base past current_id.
        if base < current_id {
            continue;
        }

        // Fill gap with a dummy input tensor.
        if base > current_id {
            let gap = base - current_id;
            span_graph.add_input_tensor(GlobalId(u64::MAX - current_id), gap, DType::F32);
            current_id = base;
        }

        match kind {
            RangeKind::InputTensor(it_idx) => {
                let it = &input_tensors[*it_idx];
                span_graph.add_input_tensor(it.tensor_id, count, it.dtype);
            }
            RangeKind::ExternalGroup => {
                // External groups become input tensors in the span.
                // Use a synthetic GlobalId based on the base atom ID.
                span_graph.add_input_tensor(
                    GlobalId(base),
                    count,
                    external_inputs.get(&base).map_or(DType::F32, |r| r.dtype),
                );
            }
            RangeKind::OwnGroup(gi) => {
                let g = &groups[*gi];
                let allocated = span_graph.alloc_placeholder(count, g.output_dtype);
                debug_assert_eq!(
                    allocated.0, base,
                    "AtomId mismatch: expected {}, got {}",
                    base, allocated.0
                );
                span_graph.fill_placeholder(
                    allocated,
                    count,
                    g.output_dtype,
                    g.op.clone(),
                    g.sym_dims.clone(),
                    g.inputs.clone(),
                );
            }
        }

        current_id = base + count;
    }

    // Copy outputs from main graph that belong to this span.
    for &output_id in &graph.outputs {
        for &gi in group_indices {
            let g = &groups[gi];
            if g.contains(output_id) {
                span_graph.outputs.push(output_id);
            }
        }
    }

    span_graph
}

// ─── Input tensor reference helpers ─────────────────────────────────────────

/// Check which input tensors an InputRef references.
fn collect_input_tensor_refs(
    graph: &NanoGraph,
    input_ref: &InputRef,
    count: u64,
    atom_offset: u64,
    needed: &mut BTreeSet<usize>,
) {
    let input_tensors = graph.input_tensors();
    if input_tensors.is_empty() || count == 0 {
        return;
    }
    match input_ref {
        InputRef::Broadcast(base) => {
            for (it_idx, it) in input_tensors.iter().enumerate() {
                if base.0 >= it.base_id.0 && base.0 < it.base_id.0 + it.count {
                    needed.insert(it_idx);
                }
            }
        }
        InputRef::Affine { .. } | InputRef::StridedBroadcast { .. } => {
            let first = input_ref.resolve(atom_offset);
            let last = input_ref.resolve(atom_offset + count - 1);
            let lo = first.0.min(last.0);
            let hi = first.0.max(last.0);
            for (it_idx, it) in input_tensors.iter().enumerate() {
                let it_end = it.base_id.0 + it.count;
                if lo < it_end && hi >= it.base_id.0 {
                    needed.insert(it_idx);
                }
            }
        }
        InputRef::Modular {
            base,
            stride,
            modulus,
        } => {
            let a = base.0;
            let b = (base.0 as i64 + *stride * (*modulus as i64 - 1)) as u64;
            let lo = a.min(b);
            let hi = a.max(b);
            for (it_idx, it) in input_tensors.iter().enumerate() {
                let it_end = it.base_id.0 + it.count;
                if lo < it_end && hi >= it.base_id.0 {
                    needed.insert(it_idx);
                }
            }
        }
        InputRef::Explicit(ids) => {
            for id in ids.iter().skip(atom_offset as usize).take(count as usize) {
                for (it_idx, it) in input_tensors.iter().enumerate() {
                    if id.0 >= it.base_id.0 && id.0 < it.base_id.0 + it.count {
                        needed.insert(it_idx);
                    }
                }
            }
        }
    }
}

/// Check if reduce stride ranges reference input tensors.
fn check_reduce_input_tensor_refs(
    input_ref: &InputRef,
    atom_offset: u64,
    count: u64,
    reduce_count: u64,
    reduce_stride: i64,
    input_tensors: &[InputTensor],
    needed: &mut BTreeSet<usize>,
) {
    let first = input_ref.resolve(atom_offset);
    let last = input_ref.resolve(atom_offset + count - 1);
    let end_off = (reduce_count as i64 - 1) * reduce_stride;
    let endpoints = [
        first.0,
        (first.0 as i64 + end_off) as u64,
        last.0,
        (last.0 as i64 + end_off) as u64,
    ];
    let lo = *endpoints.iter().min().unwrap();
    let hi = *endpoints.iter().max().unwrap();
    for (it_idx, it) in input_tensors.iter().enumerate() {
        let it_end = it.base_id.0 + it.count;
        if lo < it_end && hi >= it.base_id.0 {
            needed.insert(it_idx);
        }
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn empty_span() -> Span {
    Span {
        graph: NanoGraph::new(),
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
    use crate::numeric_scalar::NumericScalar;

    /// Helper: create a simple linear chain graph.
    /// a (literal) -> b (neg) -> c (neg) -> d (neg)
    fn make_chain_graph() -> (NanoGraph, Vec<InputTensor>) {
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
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine { base: b, stride: 1 }],
        );
        let d = g.push_group(
            100,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine { base: c, stride: 1 }],
        );
        g.outputs.push(d);
        (g, vec![])
    }

    /// Helper: create a parallel diamond graph.
    /// input -> [branch_a, branch_b] -> merge
    fn make_diamond_graph() -> (NanoGraph, Vec<InputTensor>) {
        let mut g = NanoGraph::new();
        let input = g.push_group(
            200,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );
        let branch_a = g.push_group(
            200,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: input,
                stride: 1,
            }],
        );
        let branch_b = g.push_group(
            200,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: input,
                stride: 1,
            }],
        );
        let merge = g.push_group(
            200,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine {
                    base: branch_a,
                    stride: 1,
                },
                InputRef::Affine {
                    base: branch_b,
                    stride: 1,
                },
            ],
        );
        g.outputs.push(merge);
        (g, vec![])
    }

    /// Helper: create a graph with input tensors (simulating weights).
    fn make_weighted_graph() -> (NanoGraph, Vec<InputTensor>) {
        let mut g = NanoGraph::new();
        let w = g.add_input_tensor(GlobalId(0x1000), 768, DType::F32);
        let x = g.add_input_tensor(GlobalId(0x2000), 768, DType::F32);
        let y = g.push_group(
            768,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine { base: w, stride: 1 },
                InputRef::Affine { base: x, stride: 1 },
            ],
        );
        g.outputs.push(y);
        let inputs = g.input_tensors().to_vec();
        (g, inputs)
    }

    /// Helper: create a graph mimicking a small matmul: y = x * W reduced.
    /// M=4 independent rows, K=8.
    fn make_matmul_graph() -> (NanoGraph, Vec<InputTensor>) {
        let mut g = NanoGraph::new();
        let m = 4u64;
        let k = 8u64;

        // Weight matrix (K elements).
        let w = g.push_group(
            k,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.5)),
            vec![],
            vec![],
        );

        // Input vector (K elements).
        let x = g.push_group(
            k,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );

        // M*K Mul groups (each row broadcasts W).
        let mul = g.push_group(
            m * k,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Modular {
                    base: w,
                    stride: 1,
                    modulus: k,
                },
                InputRef::StridedBroadcast {
                    base: x,
                    stride: 1,
                    repeat: k,
                },
            ],
        );

        // M ReduceSum groups.
        let red = g.push_group(
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

        g.outputs.push(red);
        (g, vec![])
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
    fn test_single_lane_chain() {
        let (g, inputs) = make_chain_graph();
        let output_ids: Vec<AtomId> = g.outputs.clone();
        let phases = plan(&g, 1, &inputs, &output_ids);

        assert!(!phases.is_empty());
        for phase in &phases {
            assert_eq!(phase.spans.len(), 1);
        }

        // All groups should appear across spans.
        let total_groups: usize = phases
            .iter()
            .flat_map(|p| p.spans.iter())
            .map(|s| s.graph.num_groups())
            .sum();
        assert_eq!(total_groups, g.num_groups());

        // Validate all span nanographs.
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

    #[test]
    fn test_multi_lane_diamond() {
        let (g, inputs) = make_diamond_graph();
        let output_ids: Vec<AtomId> = g.outputs.clone();
        let phases = plan(&g, 2, &inputs, &output_ids);

        assert!(!phases.is_empty());

        for (pi, phase) in phases.iter().enumerate() {
            assert_eq!(phase.spans.len(), 2);
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

        let total_groups: usize = phases
            .iter()
            .flat_map(|p| p.spans.iter())
            .map(|s| s.graph.num_groups())
            .sum();
        assert_eq!(total_groups, g.num_groups());
    }

    #[test]
    fn test_weighted_graph() {
        let (g, inputs) = make_weighted_graph();
        let output_ids: Vec<AtomId> = g.outputs.clone();
        let phases = plan(&g, 2, &inputs, &output_ids);

        assert!(!phases.is_empty());

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

    #[test]
    fn test_matmul_graph() {
        let (g, inputs) = make_matmul_graph();
        let output_ids: Vec<AtomId> = g.outputs.clone();
        let phases = plan(&g, 4, &inputs, &output_ids);

        assert!(!phases.is_empty());

        for (pi, phase) in phases.iter().enumerate() {
            assert_eq!(phase.spans.len(), 4);
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

        let total_groups: usize = phases
            .iter()
            .flat_map(|p| p.spans.iter())
            .map(|s| s.graph.num_groups())
            .sum();
        assert_eq!(total_groups, g.num_groups());
    }

    #[test]
    fn test_span_independence() {
        // Within each phase, no span should consume atoms produced by
        // another span in the same phase.
        let (g, inputs) = make_diamond_graph();
        let output_ids: Vec<AtomId> = g.outputs.clone();
        let phases = plan(&g, 2, &inputs, &output_ids);

        for (pi, phase) in phases.iter().enumerate() {
            let mut produced_by_span: Vec<HashSet<u64>> = Vec::new();
            for span in &phase.spans {
                let mut produced = HashSet::new();
                for group in span.graph.groups() {
                    for i in 0..group.count {
                        produced.insert(group.base_id.0 + i);
                    }
                }
                produced_by_span.push(produced);
            }

            for (li, span) in phase.spans.iter().enumerate() {
                for input_range in &span.inputs {
                    for i in 0..input_range.count {
                        let atom_id = input_range.base.0 + i;
                        for (other_li, other_produced) in produced_by_span.iter().enumerate() {
                            if other_li != li {
                                assert!(
                                    !other_produced.contains(&atom_id),
                                    "Phase {} span {} reads atom {} produced by span {}",
                                    pi,
                                    li,
                                    atom_id,
                                    other_li
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_output_coverage() {
        let (g, inputs) = make_matmul_graph();
        let output_ids: Vec<AtomId> = g.outputs.clone();
        let phases = plan(&g, 4, &inputs, &output_ids);

        let mut all_produced: HashSet<u64> = HashSet::new();
        for phase in &phases {
            for span in &phase.spans {
                for group in span.graph.groups() {
                    for i in 0..group.count {
                        all_produced.insert(group.base_id.0 + i);
                    }
                }
            }
        }

        for &output_id in &output_ids {
            if let Some(group) = g.group_of(output_id) {
                for i in 0..group.count {
                    assert!(
                        all_produced.contains(&(group.base_id.0 + i)),
                        "Output atom {} not produced by any span",
                        group.base_id.0 + i
                    );
                }
            }
        }
    }

    #[test]
    fn test_predecessor_sets() {
        let (g, _) = make_chain_graph();
        let preds = build_predecessor_sets(&g);
        assert!(preds[0].is_empty());
        assert_eq!(preds[1], HashSet::from([0]));
        assert_eq!(preds[2], HashSet::from([1]));
        assert_eq!(preds[3], HashSet::from([2]));
    }

    #[test]
    fn test_super_groups_chain() {
        let (g, _) = make_chain_graph();
        let preds = build_predecessor_sets(&g);
        let sgs = find_super_groups(&g, &preds);
        assert!(sgs.len() >= 1);
        let total: usize = sgs.iter().map(|sg| sg.len()).sum();
        assert_eq!(total, 4);
    }

    #[test]
    fn test_super_groups_diamond() {
        let (g, _) = make_diamond_graph();
        let preds = build_predecessor_sets(&g);
        let sgs = find_super_groups(&g, &preds);
        let total: usize = sgs.iter().map(|sg| sg.len()).sum();
        assert_eq!(total, g.num_groups());
    }

    #[test]
    fn test_multi_independent_chains() {
        let mut g = NanoGraph::new();
        let a1 = g.push_group(
            500,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );
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
        let a2 = g.push_group(
            500,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
        );
        let b2 = g.push_group(
            500,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: a2,
                stride: 1,
            }],
        );
        g.outputs.push(b1);
        g.outputs.push(b2);

        let phases = plan(&g, 2, &[], &g.outputs.clone());
        assert!(!phases.is_empty());

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

        let total_groups: usize = phases
            .iter()
            .flat_map(|p| p.spans.iter())
            .map(|s| s.graph.num_groups())
            .sum();
        assert_eq!(total_groups, g.num_groups());
    }

    #[test]
    fn test_broadcast_shared_weight() {
        let mut g = NanoGraph::new();
        let weight = g.push_group(
            1,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.5)),
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
            vec![InputRef::Broadcast(weight)],
        );
        let b = g.push_group(
            100,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Broadcast(weight)],
        );
        g.outputs.push(a);
        g.outputs.push(b);

        let phases = plan(&g, 2, &[], &g.outputs.clone());

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

    #[test]
    fn test_single_group_graph() {
        let mut g = NanoGraph::new();
        let a = g.push_group(
            50,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(42.0)),
            vec![],
            vec![],
        );
        g.outputs.push(a);

        let phases = plan(&g, 4, &[], &g.outputs.clone());
        assert!(!phases.is_empty());

        let total_groups: usize = phases
            .iter()
            .flat_map(|p| p.spans.iter())
            .map(|s| s.graph.num_groups())
            .sum();
        assert_eq!(total_groups, 1);

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

    #[test]
    fn test_deep_dag_phases() {
        // Build a deep DAG and verify that the partitioner creates
        // multiple phases corresponding to the dependency depth.
        let mut g = NanoGraph::new();
        let lit = g.push_group(
            10,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );
        let mut prev = lit;
        for _ in 0..5 {
            prev = g.push_group(
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
        }
        g.outputs.push(prev);

        let phases = plan(&g, 2, &[], &g.outputs.clone());
        // Should have multiple phases for the sequential chain.
        assert!(phases.len() >= 2);

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

    #[test]
    fn test_hierarchical_decomposition() {
        // Build a graph with clear hierarchical structure:
        // Two independent "layers", each with internal parallelism.
        let mut g = NanoGraph::new();

        // Layer 1: weight -> [branch_a, branch_b] -> merge
        let w1 = g.push_group(
            50,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );
        let l1a = g.push_group(
            50,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: w1,
                stride: 1,
            }],
        );
        let l1b = g.push_group(
            50,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: w1,
                stride: 1,
            }],
        );
        let l1_merge = g.push_group(
            50,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine {
                    base: l1a,
                    stride: 1,
                },
                InputRef::Affine {
                    base: l1b,
                    stride: 1,
                },
            ],
        );

        // Layer 2: l1_merge -> [branch_a, branch_b] -> output
        let l2a = g.push_group(
            50,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: l1_merge,
                stride: 1,
            }],
        );
        let l2b = g.push_group(
            50,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: l1_merge,
                stride: 1,
            }],
        );
        let output = g.push_group(
            50,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine {
                    base: l2a,
                    stride: 1,
                },
                InputRef::Affine {
                    base: l2b,
                    stride: 1,
                },
            ],
        );
        g.outputs.push(output);

        let phases = plan(&g, 2, &[], &g.outputs.clone());

        // Should use multiple phases.
        assert!(phases.len() >= 2);

        // All groups covered.
        let total_groups: usize = phases
            .iter()
            .flat_map(|p| p.spans.iter())
            .map(|s| s.graph.num_groups())
            .sum();
        assert_eq!(total_groups, g.num_groups());

        // All spans valid.
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
}
