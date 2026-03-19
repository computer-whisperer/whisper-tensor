#![allow(clippy::all, dead_code, unreachable_code, unreachable_patterns, unused_imports, unused_variables)]

//! Recursive-Bisection Partitioner with Structural Pattern Detection (attempt F)
//!
//! # Core insight
//!
//! The NanoGraph for a transformer model has massive structural regularity:
//! matmul operations decompose into M independent row computations (Mul+ReduceSum
//! chains) that all share weight data. The graph has a very specific shape:
//! sequential transformer layers connected by narrow residual streams.
//!
//! This partitioner exploits that structure directly:
//!
//! 1. **Phase detection via dominator cuts**: Instead of liveness profiles or
//!    depth wavefronts, we find "narrow waists" in the dependency DAG — points
//!    where the set of live cross-group data dependencies is minimal. These
//!    correspond to inter-layer boundaries (residual streams of ~768 values
//!    between layers of ~millions of atoms). We detect these by tracking the
//!    "frontier width" — how many distinct groups have unsatisfied consumers
//!    at each point in the topological order.
//!
//! 2. **Row-bundle detection**: Within each phase, we identify structurally
//!    regular patterns: groups that share the same producer set via broadcast/
//!    modular access (matmul rows sharing weights). These "row bundles" are
//!    the natural unit of work distribution — each row is independent and
//!    produces the same amount of work.
//!
//! 3. **Balanced distribution via bundle-aware bin-packing**: Row bundles are
//!    split across lanes for perfect balance. Singleton groups (elementwise ops,
//!    small utility computations) are assigned to lanes using least-loaded-first.
//!    Literal groups (weights/constants) are duplicated into every lane that
//!    needs them.
//!
//! # Complexity
//!
//! All algorithms operate at group granularity: O(G^2) worst case for dependency
//! DAG construction, O(G log G) for sorting and bin-packing. No per-atom
//! allocation.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::dtype::DType;
use crate::graph::GlobalId;
use crate::nano_graph::pattern::InputTensor;
use crate::nano_graph::{AtomGroup, AtomId, AtomRange, InputRef, NanoGraph, ScalarOp, SymDim};

use super::types::{Phase, Span};

// ─── Public API ────────────────────────────────────────────────────────────

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
    let (producers, consumers) = build_dependency_dag(graph);

    // Step 2: Identify output groups.
    let output_group_set = identify_output_groups(graph, output_atom_ids);

    // Step 3: Find phase boundaries using frontier-width analysis.
    let phase_boundaries =
        find_phase_boundaries(graph, &producers, &consumers, &output_group_set);

    // Step 4: Assign groups to phases.
    let phase_assignments = assign_to_phases(n, &phase_boundaries);
    let num_phases = phase_assignments.iter().copied().max().unwrap_or(0) + 1;

    // Step 5: Within each phase, assign groups to lanes using structural
    // pattern detection and balanced bin-packing.
    let lane_assignments = assign_lanes_per_phase(
        graph,
        &producers,
        &consumers,
        &phase_assignments,
        num_phases,
        num_lanes,
    );

    // Step 6: Build span NanoGraphs.
    build_all_phases(
        graph,
        input_tensors,
        &producers,
        &consumers,
        &phase_assignments,
        &lane_assignments,
        &output_group_set,
        num_phases,
        num_lanes,
    )
}

// ─── Dependency DAG ────────────────────────────────────────────────────────

fn build_dependency_dag(graph: &NanoGraph) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let groups = graph.groups();
    let n = groups.len();
    let mut producers: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut consumers: Vec<Vec<usize>> = vec![vec![]; n];

    for (gi, group) in groups.iter().enumerate() {
        let mut seen = HashSet::new();
        graph.collect_all_producer_indices(group, gi, &mut seen);
        let deps: Vec<usize> = seen.into_iter().collect();
        for &pi in &deps {
            consumers[pi].push(gi);
        }
        producers.push(deps);
    }

    // Deduplicate consumers.
    for c in consumers.iter_mut() {
        c.sort_unstable();
        c.dedup();
    }

    (producers, consumers)
}

// ─── Phase boundary detection ──────────────────────────────────────────────

/// Find phase boundaries by tracking the "frontier width" — the number of
/// distinct group outputs that are still "live" (have unsatisfied consumers)
/// at each point in the topological traversal.
///
/// Narrow waists in this frontier correspond to inter-layer boundaries in
/// transformers. We place barriers at the deepest local minima.
fn find_phase_boundaries(
    graph: &NanoGraph,
    producers: &[Vec<usize>],
    consumers: &[Vec<usize>],
    output_group_set: &HashSet<usize>,
) -> Vec<usize> {
    let groups = graph.groups();
    let n = groups.len();

    if n < 6 {
        return vec![];
    }

    // Track remaining consumer count for each group.
    let mut remaining: Vec<u32> = consumers
        .iter()
        .enumerate()
        .map(|(gi, c)| {
            let base = c.len() as u32;
            // Output groups never die (they must survive to the end).
            if output_group_set.contains(&gi) {
                base + 1
            } else {
                base
            }
        })
        .collect();

    // Traverse in topo order, tracking frontier: atom-weighted live set.
    let mut live_atoms: i64 = 0;
    let mut profile = Vec::with_capacity(n);

    for gi in 0..n {
        let g = &groups[gi];
        live_atoms += g.count as i64;

        for &pi in &producers[gi] {
            remaining[pi] -= 1;
            if remaining[pi] == 0 {
                live_atoms -= groups[pi].count as i64;
            }
        }

        profile.push(live_atoms);
    }

    // Find the global maximum to calibrate the threshold.
    let max_live = *profile.iter().max().unwrap_or(&1);
    if max_live <= 0 {
        return vec![];
    }

    // Adaptive threshold: we want narrow waists that are significantly below
    // the peaks. Use 10% of max as the threshold — transformer inter-layer
    // boundaries have ~768 live values vs millions within a layer.
    let threshold = (max_live as f64 * 0.10).max(1.0) as i64;

    // Scan for "valley regions" — contiguous stretches below threshold.
    // Pick the deepest point in each valley as the barrier position.
    let margin = 2.min(n / 4);
    let mut valleys: Vec<(usize, i64)> = Vec::new();
    let mut in_valley = false;
    let mut valley_best_idx = 0usize;
    let mut valley_best_val = i64::MAX;

    for i in margin..(n.saturating_sub(margin)) {
        if profile[i] <= threshold {
            if !in_valley {
                in_valley = true;
                valley_best_idx = i;
                valley_best_val = profile[i];
            } else if profile[i] < valley_best_val {
                valley_best_idx = i;
                valley_best_val = profile[i];
            }
        } else {
            if in_valley {
                valleys.push((valley_best_idx, valley_best_val));
                in_valley = false;
                valley_best_val = i64::MAX;
            }
        }
    }
    if in_valley {
        valleys.push((valley_best_idx, valley_best_val));
    }

    // Deduplicate valleys that are very close (keep deepest).
    let min_gap = 4.max(n / 100);
    let mut boundaries: Vec<usize> = Vec::new();
    let mut i = 0;
    while i < valleys.len() {
        let mut best_idx = valleys[i].0;
        let mut best_val = valleys[i].1;
        let mut j = i + 1;
        while j < valleys.len() && valleys[j].0.saturating_sub(valleys[i].0) < min_gap {
            if valleys[j].1 < best_val {
                best_val = valleys[j].1;
                best_idx = valleys[j].0;
            }
            j += 1;
        }
        boundaries.push(best_idx);
        i = j;
    }

    // Validate boundaries: ensure no group in a later phase depends on a group
    // in an earlier phase that *isn't* in the live set at the boundary. This
    // is guaranteed by construction (the boundary is at a point where only
    // boundary-crossing groups are live), but let's verify.
    // Actually the topo-order assignment handles this — groups before boundary
    // are in phase P, groups after are in phase P+1.

    boundaries
}

fn identify_output_groups(graph: &NanoGraph, output_atom_ids: &[AtomId]) -> HashSet<usize> {
    let mut out = HashSet::new();
    for &id in output_atom_ids {
        if let Some(gi) = graph.find_group_idx(id) {
            out.insert(gi);
        }
    }
    out
}

fn assign_to_phases(n: usize, boundaries: &[usize]) -> Vec<usize> {
    let mut assignments = vec![0usize; n];
    let mut phase = 0;
    let mut bi = 0;
    for gi in 0..n {
        if bi < boundaries.len() && gi > boundaries[bi] {
            phase += 1;
            bi += 1;
        }
        assignments[gi] = phase;
    }
    assignments
}

// ─── Lane assignment ───────────────────────────────────────────────────────

/// Per-group lane assignment info.
struct LaneInfo {
    /// lane[gi] = which lane group gi is assigned to.
    lane: Vec<usize>,
    /// Groups that should be duplicated (inlined) into every lane that needs them.
    duplicated: HashSet<usize>,
}

/// Assign groups to lanes within each phase.
///
/// The key insight: we detect "row bundles" — sets of groups that are
/// structurally identical independent computations sharing common inputs
/// (like matmul rows sharing weights). These bundles are distributed
/// round-robin across lanes for near-perfect balance.
fn assign_lanes_per_phase(
    graph: &NanoGraph,
    producers: &[Vec<usize>],
    consumers: &[Vec<usize>],
    phase_assignments: &[usize],
    num_phases: usize,
    num_lanes: usize,
) -> LaneInfo {
    let groups = graph.groups();
    let n = groups.len();
    let mut lane_assignments = vec![0usize; n];
    let mut duplicated = HashSet::new();

    if num_lanes <= 1 {
        return LaneInfo {
            lane: lane_assignments,
            duplicated,
        };
    }

    for phase in 0..num_phases {
        // Collect groups in this phase (in topo order = index order).
        let phase_groups: Vec<usize> = (0..n)
            .filter(|&gi| phase_assignments[gi] == phase)
            .collect();

        if phase_groups.is_empty() {
            continue;
        }

        let phase_set: HashSet<usize> = phase_groups.iter().copied().collect();

        // Count intra-phase consumers for each group.
        let mut intra_consumer_count: HashMap<usize, usize> = HashMap::new();
        for &gi in &phase_groups {
            for &pi in &producers[gi] {
                if phase_set.contains(&pi) {
                    *intra_consumer_count.entry(pi).or_default() += 1;
                }
            }
        }

        // Identify literal groups that are shared (consumed by >1 group in phase).
        // These get duplicated into every lane that needs them.
        for &gi in &phase_groups {
            let count = intra_consumer_count.get(&gi).copied().unwrap_or(0);
            if count > 1 && is_literal_group(&groups[gi]) {
                duplicated.insert(gi);
            }
        }

        // Build connected components of non-duplicated groups.
        // Two groups are connected if they share a private producer-consumer edge
        // (the producer has exactly 1 intra-phase consumer) or if they're
        // connected through a shared non-literal producer.
        let mut uf = UnionFind::new(n);

        for &gi in &phase_groups {
            if duplicated.contains(&gi) {
                continue;
            }
            for &pi in &producers[gi] {
                if !phase_set.contains(&pi) || duplicated.contains(&pi) {
                    continue;
                }
                let pi_consumers = intra_consumer_count.get(&pi).copied().unwrap_or(0);
                if pi_consumers <= 1 {
                    // Private edge: always union.
                    uf.union(gi, pi);
                } else if !is_literal_group(&groups[pi]) {
                    // Shared non-literal producer: consumers must be on same lane.
                    uf.union(gi, pi);
                }
                // Shared literal: consumers are independent, skip.
            }
        }

        // Collect connected components.
        let mut components: HashMap<usize, Vec<usize>> = HashMap::new();
        for &gi in &phase_groups {
            if duplicated.contains(&gi) {
                continue;
            }
            let rep = uf.find(gi);
            components.entry(rep).or_default().push(gi);
        }

        // Detect row bundles: components that share the same set of literal
        // producer groups (i.e., matmul rows sharing the same weights).
        // These can be distributed round-robin.
        let mut bundle_key_to_components: HashMap<Vec<usize>, Vec<(usize, Vec<usize>, u64)>> =
            HashMap::new();

        let mut unbundled: Vec<(Vec<usize>, u64)> = Vec::new();

        for (rep, comp_groups) in &components {
            let total_atoms: u64 = comp_groups.iter().map(|&gi| groups[gi].count).sum();

            // Compute the "signature" of this component: sorted set of
            // shared literal producers from outside the component.
            let comp_set: HashSet<usize> = comp_groups.iter().copied().collect();
            let mut literal_producers: Vec<usize> = Vec::new();
            for &gi in comp_groups {
                for &pi in &producers[gi] {
                    if duplicated.contains(&pi) && !comp_set.contains(&pi) {
                        literal_producers.push(pi);
                    }
                }
            }
            literal_producers.sort_unstable();
            literal_producers.dedup();

            // Also consider the op-type signature: what kinds of ops in this component.
            // For bundling, we want components with the same literal-producer set AND
            // the same op composition. This ensures we're grouping true "row clones."
            let mut op_sig = Vec::new();
            for &gi in comp_groups {
                op_sig.push(op_discriminant(&groups[gi].op));
            }
            op_sig.sort_unstable();

            // Bundle key: literal producers + op signature.
            // But we only bundle if there are shared literal producers (weight sharing).
            if literal_producers.is_empty() || comp_groups.len() > 20 {
                // No weight sharing or already large — treat as standalone.
                unbundled.push((comp_groups.clone(), total_atoms));
            } else {
                let mut key = literal_producers.clone();
                key.extend(op_sig.iter().map(|&x| x + 1_000_000)); // namespace separation
                bundle_key_to_components
                    .entry(key)
                    .or_default()
                    .push((*rep, comp_groups.clone(), total_atoms));
            }
        }

        // Distribute bundles round-robin across lanes.
        let mut lane_loads = vec![0u64; num_lanes];

        for (key, mut bundle) in bundle_key_to_components {
            if bundle.len() < 2 {
                // Single component, not a real bundle.
                for (_, comp_groups, total_atoms) in bundle {
                    unbundled.push((comp_groups, total_atoms));
                }
                continue;
            }

            // Sort components within this bundle by total atoms descending
            // for better balance during round-robin.
            bundle.sort_by(|a, b| b.2.cmp(&a.2));

            // Assign each component in the bundle to the least-loaded lane.
            for (_, comp_groups, total_atoms) in &bundle {
                let target = lane_loads
                    .iter()
                    .enumerate()
                    .min_by_key(|&(_, load)| *load)
                    .unwrap()
                    .0;

                for &gi in comp_groups {
                    lane_assignments[gi] = target;
                }
                lane_loads[target] += total_atoms;
            }
        }

        // Sort unbundled components by total atoms descending (first-fit-decreasing).
        unbundled.sort_by(|a, b| b.1.cmp(&a.1));

        for (comp_groups, total_atoms) in &unbundled {
            let target = lane_loads
                .iter()
                .enumerate()
                .min_by_key(|&(_, load)| *load)
                .unwrap()
                .0;

            for &gi in comp_groups {
                lane_assignments[gi] = target;
            }
            lane_loads[target] += total_atoms;
        }

        // Duplicated groups default to lane 0 (they get inlined per-lane).
        for &gi in &phase_groups {
            if duplicated.contains(&gi) {
                lane_assignments[gi] = 0;
            }
        }
    }

    LaneInfo {
        lane: lane_assignments,
        duplicated,
    }
}

/// Classify a ScalarOp into a small number of discriminant categories for bundling.
fn op_discriminant(op: &ScalarOp) -> usize {
    match op {
        ScalarOp::Literal(_) => 0,
        ScalarOp::Identity => 1,
        ScalarOp::Binary { .. } => 2,
        ScalarOp::Unary { .. } => 3,
        ScalarOp::Select => 4,
        ScalarOp::Reduce { .. } => 5,
        ScalarOp::IndirectLoad { .. } => 6,
    }
}

fn is_literal_group(group: &AtomGroup) -> bool {
    matches!(group.op, ScalarOp::Literal(_)) && group.inputs.is_empty()
}

fn is_literal_op(op: &ScalarOp) -> bool {
    matches!(op, ScalarOp::Literal(_))
}

// ─── Union-Find ────────────────────────────────────────────────────────────

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

// ─── Phase & Span construction ─────────────────────────────────────────────

fn build_all_phases(
    graph: &NanoGraph,
    input_tensors: &[InputTensor],
    producers: &[Vec<usize>],
    consumers: &[Vec<usize>],
    phase_assignments: &[usize],
    lane_info: &LaneInfo,
    output_group_set: &HashSet<usize>,
    num_phases: usize,
    num_lanes: usize,
) -> Vec<Phase> {
    let groups = graph.groups();
    let n = groups.len();

    // Build per-phase-per-lane group lists.
    let mut phase_lane_groups: Vec<Vec<Vec<usize>>> =
        vec![vec![Vec::new(); num_lanes]; num_phases];

    // Place non-duplicated groups.
    for gi in 0..n {
        if lane_info.duplicated.contains(&gi) {
            continue;
        }
        let phase = phase_assignments[gi];
        let lane = lane_info.lane[gi];
        phase_lane_groups[phase][lane].push(gi);
    }

    // Place duplicated groups into every lane that has a consumer of them.
    for &gi in &lane_info.duplicated {
        let phase = phase_assignments[gi];
        let mut needed_lanes: HashSet<usize> = HashSet::new();

        for &ci in &consumers[gi] {
            if phase_assignments[ci] == phase && !lane_info.duplicated.contains(&ci) {
                needed_lanes.insert(lane_info.lane[ci]);
            }
        }

        if needed_lanes.is_empty() {
            needed_lanes.insert(0);
        }

        for lane in needed_lanes {
            phase_lane_groups[phase][lane].push(gi);
        }
    }

    // Sort each lane's groups by index (topo order).
    for phase_lanes in phase_lane_groups.iter_mut() {
        for lane_groups in phase_lanes.iter_mut() {
            lane_groups.sort_unstable();
        }
    }

    // Determine which groups produce outputs that cross phase boundaries.
    let mut crosses_boundary: HashSet<usize> = HashSet::new();
    for gi in 0..n {
        let my_phase = phase_assignments[gi];
        if output_group_set.contains(&gi) {
            crosses_boundary.insert(gi);
            continue;
        }
        for &ci in &consumers[gi] {
            if phase_assignments[ci] > my_phase {
                crosses_boundary.insert(gi);
                break;
            }
        }
    }

    // Build each phase.
    let mut phases: Vec<Phase> = Vec::with_capacity(num_phases);

    for phase in 0..num_phases {
        let mut spans = Vec::with_capacity(num_lanes);
        let phase_set: HashSet<usize> = (0..n)
            .filter(|&gi| phase_assignments[gi] == phase)
            .collect();

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

            let my_set: HashSet<usize> = my_groups.iter().copied().collect();
            let span = build_span(
                graph,
                input_tensors,
                my_groups,
                &my_set,
                producers,
                &crosses_boundary,
                &lane_info.duplicated,
                &phase_set,
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
    my_groups: &[usize],
    my_set: &HashSet<usize>,
    producers: &[Vec<usize>],
    crosses_boundary: &HashSet<usize>,
    duplicated: &HashSet<usize>,
    phase_set: &HashSet<usize>,
) -> Span {
    let groups = graph.groups();
    let mut span_graph = NanoGraph::new();
    span_graph.sym_dim_names = graph.sym_dim_names.clone();
    span_graph.sym_dim_bounds = graph.sym_dim_bounds.clone();

    // Collect external dependencies: groups we depend on that are not in our set.
    let mut external_ranges: BTreeMap<u64, (AtomId, u64, DType)> = BTreeMap::new();

    // Also inline small literals that we depend on and are not in our set.
    let mut inlined_literals: BTreeSet<usize> = BTreeSet::new();
    const LITERAL_INLINE_THRESHOLD: u64 = 65536;

    for &gi in my_groups {
        let group = &groups[gi];

        // Check producer dependencies.
        for &pi in &producers[gi] {
            if my_set.contains(&pi) {
                continue; // Internal.
            }
            let pg = &groups[pi];
            if is_literal_group(pg) && pg.count < LITERAL_INLINE_THRESHOLD {
                inlined_literals.insert(pi);
            } else {
                record_external(&mut external_ranges, pg.base_id, pg.count, pg.output_dtype);
            }
        }

        // Check InputRefs for input tensor references.
        for input_ref in &group.inputs {
            collect_input_tensor_external(
                graph,
                input_ref,
                group.count,
                group.atom_offset,
                &mut external_ranges,
            );
        }

        // Reduce stride: extended range.
        if let ScalarOp::Reduce {
            reduce_count,
            reduce_stride,
            ..
        } = &group.op
        {
            if *reduce_count > 1 && *reduce_stride != 0 {
                for input_ref in &group.inputs {
                    collect_reduce_external(
                        graph,
                        input_ref,
                        group,
                        *reduce_count,
                        *reduce_stride,
                        my_set,
                        &mut external_ranges,
                        &mut inlined_literals,
                    );
                }
            }
        }

        // IndirectLoad table reference.
        if let ScalarOp::IndirectLoad { table_base } = &group.op {
            if let Some(pi) = graph.find_group_idx(*table_base) {
                if !my_set.contains(&pi) {
                    let pg = &groups[pi];
                    if is_literal_group(pg) && pg.count < LITERAL_INLINE_THRESHOLD {
                        inlined_literals.insert(pi);
                    } else {
                        record_external(
                            &mut external_ranges,
                            pg.base_id,
                            pg.count,
                            pg.output_dtype,
                        );
                    }
                }
            }
            // Also check if table_base is in an input tensor.
            for it in graph.input_tensors() {
                if table_base.0 >= it.base_id.0 && table_base.0 < it.base_id.0 + it.count {
                    record_external(&mut external_ranges, it.base_id, it.count, it.dtype);
                    break;
                }
            }
        }
    }

    // Also check inlined literals' own input tensor dependencies.
    for &li in &inlined_literals {
        let group = &groups[li];
        for input_ref in &group.inputs {
            collect_input_tensor_external(
                graph,
                input_ref,
                group.count,
                group.atom_offset,
                &mut external_ranges,
            );
        }
    }

    // Merge overlapping/adjacent external ranges.
    let merged_external = merge_ranges(&external_ranges);

    // Now insert everything into the span graph in atom ID order.
    #[derive(Clone)]
    enum InsertItem {
        External { base: AtomId, count: u64, dtype: DType },
        InlineLiteral { gi: usize },
        ComputeGroup { gi: usize },
    }

    let mut items: Vec<(u64, InsertItem)> = Vec::new();

    for &(base, count, dtype) in &merged_external {
        items.push((base.0, InsertItem::External { base, count, dtype }));
    }

    for &li in &inlined_literals {
        items.push((groups[li].base_id.0, InsertItem::InlineLiteral { gi: li }));
    }

    for &gi in my_groups {
        items.push((groups[gi].base_id.0, InsertItem::ComputeGroup { gi }));
    }

    items.sort_by_key(|(pos, _)| *pos);

    // Track inserted ranges to avoid duplicates.
    let mut inserted: Vec<(u64, u64)> = Vec::new();
    let mut span_inputs: Vec<AtomRange> = Vec::new();
    let mut span_outputs: Vec<AtomRange> = Vec::new();

    for (_, item) in &items {
        match item {
            InsertItem::External { base, count, dtype } => {
                if would_overlap(&inserted, base.0, *count) {
                    continue;
                }
                span_graph.insert_input_tensor_at(*base, GlobalId(0), *count, *dtype);
                span_inputs.push(AtomRange {
                    base: *base,
                    count: *count,
                    dtype: *dtype,
                });
                inserted.push((base.0, *count));
            }
            InsertItem::InlineLiteral { gi } => {
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
            InsertItem::ComputeGroup { gi } => {
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

                // Output if this group crosses a phase boundary or is a model output.
                if crosses_boundary.contains(gi) {
                    span_outputs.push(AtomRange {
                        base: g.base_id,
                        count: g.count,
                        dtype: g.output_dtype,
                    });
                }
            }
        }
    }

    // Merge adjacent output ranges.
    span_outputs = merge_atom_ranges(span_outputs);

    Span {
        graph: span_graph,
        inputs: span_inputs,
        outputs: span_outputs,
    }
}

// ─── External dependency helpers ───────────────────────────────────────────

fn record_external(
    ranges: &mut BTreeMap<u64, (AtomId, u64, DType)>,
    base: AtomId,
    count: u64,
    dtype: DType,
) {
    ranges
        .entry(base.0)
        .and_modify(|(_, existing_count, _)| {
            *existing_count = (*existing_count).max(count);
        })
        .or_insert((base, count, dtype));
}

fn collect_input_tensor_external(
    graph: &NanoGraph,
    input_ref: &InputRef,
    count: u64,
    atom_offset: u64,
    external: &mut BTreeMap<u64, (AtomId, u64, DType)>,
) {
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
        InputRef::Modular { base, stride, modulus } => {
            let a = *base;
            let b = AtomId((base.0 as i64 + *stride * (*modulus as i64 - 1)) as u64);
            vec![a, b]
        }
        InputRef::Explicit(ids) => {
            let mut samples = Vec::new();
            if !ids.is_empty() {
                let start = atom_offset as usize;
                let end = (atom_offset + count) as usize;
                if start < ids.len() {
                    samples.push(ids[start]);
                }
                if end > 0 && end - 1 < ids.len() && end - 1 > start {
                    samples.push(ids[end - 1]);
                }
            }
            samples
        }
    };

    for id in sample_ids {
        for it in graph.input_tensors() {
            if id.0 >= it.base_id.0 && id.0 < it.base_id.0 + it.count {
                record_external(external, it.base_id, it.count, it.dtype);
                break;
            }
        }
    }
}

fn collect_reduce_external(
    graph: &NanoGraph,
    input_ref: &InputRef,
    group: &AtomGroup,
    reduce_count: u64,
    reduce_stride: i64,
    my_set: &HashSet<usize>,
    external: &mut BTreeMap<u64, (AtomId, u64, DType)>,
    inlined_literals: &mut BTreeSet<usize>,
) {
    let first = input_ref.resolve(group.atom_offset);
    let last = input_ref.resolve(group.atom_offset + group.count - 1);
    let end_off = (reduce_count as i64 - 1) * reduce_stride;
    let endpoints = [
        first.0,
        (first.0 as i64 + end_off) as u64,
        last.0,
        (last.0 as i64 + end_off) as u64,
    ];
    let lo = *endpoints.iter().min().unwrap();
    let hi = *endpoints.iter().max().unwrap();

    let all_groups = graph.groups();
    for (gi, g) in all_groups.iter().enumerate() {
        let g_lo = g.base_id.0;
        let g_hi = g_lo + g.count - 1;
        if g_lo > hi {
            break;
        }
        if g_hi >= lo && g_lo <= hi && !my_set.contains(&gi) {
            if is_literal_group(g) && g.count < 65536 {
                inlined_literals.insert(gi);
            } else {
                record_external(external, g.base_id, g.count, g.output_dtype);
            }
        }
    }

    // Also check input tensors.
    for it in graph.input_tensors() {
        let it_lo = it.base_id.0;
        let it_hi = it_lo + it.count - 1;
        if lo <= it_hi && hi >= it_lo {
            record_external(external, it.base_id, it.count, it.dtype);
        }
    }
}

// ─── Range merging ─────────────────────────────────────────────────────────

fn merge_ranges(
    ranges: &BTreeMap<u64, (AtomId, u64, DType)>,
) -> Vec<(AtomId, u64, DType)> {
    if ranges.is_empty() {
        return vec![];
    }

    let sorted: Vec<(AtomId, u64, DType)> = ranges
        .values()
        .map(|&(base, count, dtype)| (base, count, dtype))
        .collect();

    let mut merged: Vec<(AtomId, u64, DType)> = Vec::new();
    for (base, count, dtype) in sorted {
        if let Some(last) = merged.last_mut() {
            let last_end = last.0 .0 + last.1;
            if base.0 <= last_end && dtype == last.2 {
                let new_end = (base.0 + count).max(last_end);
                last.1 = new_end - last.0 .0;
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

fn would_overlap(inserted: &[(u64, u64)], start: u64, count: u64) -> bool {
    let end = start + count;
    for &(es, ec) in inserted {
        let ee = es + ec;
        if start < ee && end > es {
            return true;
        }
    }
    false
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::nano_graph::ops::{ReduceKind, ScalarBinOp, ScalarUnaryOp};
    use crate::numeric_scalar::NumericScalar;

    fn push_literal(g: &mut NanoGraph, count: u64) -> AtomId {
        g.push_group(
            count,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        )
    }

    fn push_neg(g: &mut NanoGraph, count: u64, input: AtomId) -> AtomId {
        g.push_group(
            count,
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
        )
    }

    fn push_exp(g: &mut NanoGraph, count: u64, input: AtomId) -> AtomId {
        g.push_group(
            count,
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
        )
    }

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

    fn push_mul(
        g: &mut NanoGraph,
        count: u64,
        a: AtomId,
        a_stride: i64,
        b: AtomId,
        b_ref: InputRef,
    ) -> AtomId {
        g.push_group(
            count,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine { base: a, stride: a_stride }, b_ref],
        )
    }

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
                stride: reduce_count as i64,
            }],
        )
    }

    /// Verify basic structural invariants.
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
                    pi, li, errors
                );
            }
        }

        // All output atoms are produced somewhere.
        let mut produced: HashSet<u64> = HashSet::new();
        for phase in phases {
            for span in &phase.spans {
                for output in &span.outputs {
                    for i in 0..output.count {
                        produced.insert(output.base.0 + i);
                    }
                }
            }
        }
        for &out_id in &graph.outputs {
            assert!(
                produced.contains(&out_id.0),
                "Output atom {} not produced by any span",
                out_id
            );
        }
    }

    /// Verify no cross-span dependencies within a phase.
    fn verify_independence(phases: &[Phase]) {
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
                        for (other_li, other_prod) in span_produces.iter().enumerate() {
                            if other_li != li && other_prod.contains(&atom) {
                                panic!(
                                    "Phase {} lane {} reads atom {} produced by lane {} in same phase",
                                    pi, li, atom, other_li
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Compute imbalance ratio across all phases.
    fn compute_imbalance(phases: &[Phase]) -> f64 {
        let mut max_ratio = 1.0f64;
        for phase in phases {
            let loads: Vec<u64> = phase
                .spans
                .iter()
                .map(|s| s.graph.groups().iter().map(|g| g.count).sum::<u64>())
                .collect();
            let max_load = loads.iter().copied().max().unwrap_or(0);
            let min_load = loads.iter().copied().filter(|&l| l > 0).min().unwrap_or(1).max(1);
            let ratio = max_load as f64 / min_load as f64;
            if ratio > max_ratio {
                max_ratio = ratio;
            }
        }
        max_ratio
    }

    // ─── Basic tests ───────────────────────────────────────────────────────

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
    fn test_single_group() {
        let mut g = NanoGraph::new();
        let a = push_literal(&mut g, 100);
        g.outputs.push(a);

        let phases = plan(&g, 4, &[], &g.outputs.clone());
        verify_plan(&g, &phases, 4);
    }

    #[test]
    fn test_single_lane() {
        let mut g = NanoGraph::new();
        let a = push_literal(&mut g, 100);
        let b = push_neg(&mut g, 100, a);
        let c = push_exp(&mut g, 100, b);
        g.outputs.push(c);

        let phases = plan(&g, 1, &[], &g.outputs.clone());
        verify_plan(&g, &phases, 1);
    }

    #[test]
    fn test_linear_chain() {
        let mut g = NanoGraph::new();
        let a = push_literal(&mut g, 100);
        let b = push_neg(&mut g, 100, a);
        let c = push_exp(&mut g, 100, b);
        g.outputs.push(c);

        let phases = plan(&g, 4, &[], &g.outputs.clone());
        verify_plan(&g, &phases, 4);
        verify_independence(&phases);
    }

    #[test]
    fn test_diamond() {
        let mut g = NanoGraph::new();
        let a = push_literal(&mut g, 100);
        let b = push_neg(&mut g, 100, a);
        let c = push_exp(&mut g, 100, a);
        let d = push_add(&mut g, 100, b, c);
        g.outputs.push(d);

        let phases = plan(&g, 2, &[], &g.outputs.clone());
        verify_plan(&g, &phases, 2);
        verify_independence(&phases);
    }

    #[test]
    fn test_parallel_chains() {
        let mut g = NanoGraph::new();
        let a1 = push_literal(&mut g, 1000);
        let b1 = push_neg(&mut g, 1000, a1);
        let a2 = push_literal(&mut g, 1000);
        let b2 = push_exp(&mut g, 1000, a2);
        g.outputs.push(b1);
        g.outputs.push(b2);

        let phases = plan(&g, 2, &[], &g.outputs.clone());
        verify_plan(&g, &phases, 2);
        verify_independence(&phases);

        // Should have parallel execution.
        let has_parallel = phases.iter().any(|p| {
            p.spans.iter().filter(|s| s.graph.num_groups() > 0).count() > 1
        });
        assert!(has_parallel, "Expected parallel execution");
    }

    #[test]
    fn test_with_input_tensors() {
        let mut g = NanoGraph::new();
        let input_base = g.add_input_tensor(GlobalId(1), 100, DType::F32);
        let inputs = vec![InputTensor {
            tensor_id: GlobalId(1),
            base_id: input_base,
            count: 100,
            dtype: DType::F32,
        }];

        let b = push_neg(&mut g, 100, input_base);
        g.outputs.push(b);

        let phases = plan(&g, 2, &inputs, &g.outputs.clone());
        verify_plan(&g, &phases, 2);
        verify_independence(&phases);

        let has_input = phases
            .iter()
            .any(|p| p.spans.iter().any(|s| !s.inputs.is_empty()));
        assert!(has_input, "No span declares the input tensor");
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
        verify_independence(&phases);
    }

    #[test]
    fn test_modular_input_ref() {
        let mut g = NanoGraph::new();
        let weights = push_literal(&mut g, 10);
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
        verify_independence(&phases);
    }

    // ─── Matmul pattern tests ──────────────────────────────────────────────

    #[test]
    fn test_matmul_like() {
        let mut g = NanoGraph::new();
        let k: u64 = 64;
        let m: u64 = 8;

        let weights = push_literal(&mut g, k);
        let input = push_literal(&mut g, m * k);

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

        let phases = plan(&g, 4, &[], &g.outputs.clone());
        verify_plan(&g, &phases, 4);
        verify_independence(&phases);
    }

    #[test]
    fn test_matmul_row_decomposed() {
        // Matmul decomposed into M separate Mul+Reduce row groups (as GPT-2 has).
        let mut g = NanoGraph::new();
        let k: u64 = 64;
        let m: u64 = 16;

        let weights = push_literal(&mut g, k * m);
        let input = push_literal(&mut g, k);

        let mut reduce_ids = Vec::new();
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
            let red = g.push_group(
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
                    base: mul,
                    stride: 1,
                }],
            );
            reduce_ids.push(red);
        }

        for &r in &reduce_ids {
            g.outputs.push(r);
        }

        let phases = plan(&g, 4, &[], &g.outputs.clone());
        verify_plan(&g, &phases, 4);
        verify_independence(&phases);

        // Check that work is distributed across lanes.
        for phase in &phases {
            let active_lanes = phase
                .spans
                .iter()
                .filter(|s| s.graph.num_groups() > 0)
                .count();
            if active_lanes > 1 {
                // Compute balance for this phase.
                let loads: Vec<u64> = phase
                    .spans
                    .iter()
                    .map(|s| {
                        s.graph
                            .groups()
                            .iter()
                            .filter(|g| !is_literal_group(g))
                            .map(|g| g.count)
                            .sum::<u64>()
                    })
                    .collect();
                let max_load = loads.iter().copied().max().unwrap_or(0);
                let min_load = loads.iter().copied().filter(|&l| l > 0).min().unwrap_or(1);
                if min_load > 0 {
                    let ratio = max_load as f64 / min_load as f64;
                    assert!(
                        ratio < 4.0,
                        "Row-decomposed matmul imbalance too high: {:.1}x",
                        ratio
                    );
                }
            }
        }
    }

    #[test]
    fn test_reduce_with_literals() {
        let mut g = NanoGraph::new();
        let data = push_literal(&mut g, 64);
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

    // ─── Two-layer pinch point test ────────────────────────────────────────

    #[test]
    fn test_two_layer_pinch() {
        let mut g = NanoGraph::new();
        let m: u64 = 16;
        let k: u64 = 32;
        let hidden: u64 = 16;

        // Layer 1: weights + input.
        let w1 = push_literal(&mut g, k * m);
        let x = push_literal(&mut g, k);

        // Layer 1: M multiply groups.
        let mut mul1 = Vec::new();
        for row in 0..m {
            let mid = g.push_group(
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
                    InputRef::Affine {
                        base: x,
                        stride: 1,
                    },
                ],
            );
            mul1.push(mid);
        }

        // Layer 1: M reduce groups -> hidden-size residual.
        let mut res = Vec::new();
        for row in 0..m {
            let r = g.push_group(
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
                    base: mul1[row as usize],
                    stride: 1,
                }],
            );
            res.push(r);
        }

        // Layer 2: weights.
        let w2 = push_literal(&mut g, hidden * m);

        // Layer 2: M multiply groups (reading from residual).
        let mut mul2 = Vec::new();
        for row in 0..m {
            let mid = g.push_group(
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
                    InputRef::Explicit(
                        (0..hidden).map(|i| res[i as usize]).collect(),
                    ),
                ],
            );
            mul2.push(mid);
        }

        // Layer 2: M reduce groups.
        let mut out = Vec::new();
        for row in 0..m {
            let r = g.push_group(
                1,
                DType::F32,
                ScalarOp::Reduce {
                    kind: ReduceKind::Sum,
                    reduce_count: hidden,
                    reduce_stride: 1,
                    compute_dtype: DType::F32,
                },
                vec![],
                vec![InputRef::Affine {
                    base: mul2[row as usize],
                    stride: 1,
                }],
            );
            out.push(r);
        }

        g.outputs = out;

        let phases = plan(&g, 4, &[], &g.outputs.clone());
        verify_plan(&g, &phases, 4);
        verify_independence(&phases);

        // Verify all compute groups are covered.
        let total_compute: usize = phases
            .iter()
            .flat_map(|p| p.spans.iter())
            .flat_map(|s| s.graph.groups().iter())
            .filter(|g| !is_literal_group(g))
            .count();
        // 2*M mul + 2*M reduce = 4*M = 64 compute groups
        assert!(
            total_compute >= (4 * m) as usize,
            "Expected at least {} compute groups, got {}",
            4 * m,
            total_compute
        );
    }

    #[test]
    fn test_plan_preserves_all_groups() {
        let mut g = NanoGraph::new();
        let a = push_literal(&mut g, 100);
        let b = push_neg(&mut g, 100, a);
        let c = push_exp(&mut g, 100, a);
        let d = push_add(&mut g, 100, b, c);
        g.outputs.push(d);

        let phases = plan(&g, 2, &[], &g.outputs.clone());

        let main_groups: HashSet<u64> = g
            .groups()
            .iter()
            .map(|gr| gr.base_id.0)
            .collect();

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
    fn test_large_groups() {
        let mut g = NanoGraph::new();
        let a = push_literal(&mut g, 10000);
        let b = push_neg(&mut g, 10000, a);
        g.outputs.push(b);

        let phases = plan(&g, 4, &[], &g.outputs.clone());
        verify_plan(&g, &phases, 4);
    }

    // ─── Balance-focused tests ─────────────────────────────────────────────

    #[test]
    fn test_many_independent_chains_balance() {
        // 16 independent chains, 4 lanes — should achieve good balance.
        let mut g = NanoGraph::new();
        let mut outputs = Vec::new();

        for _ in 0..16 {
            let a = push_literal(&mut g, 500);
            let b = push_neg(&mut g, 500, a);
            outputs.push(b);
        }
        g.outputs = outputs;

        let phases = plan(&g, 4, &[], &g.outputs.clone());
        verify_plan(&g, &phases, 4);
        verify_independence(&phases);

        let imbalance = compute_imbalance(&phases);
        assert!(
            imbalance < 3.0,
            "16 independent chains on 4 lanes should be well-balanced, got {:.1}x",
            imbalance
        );
    }

    #[test]
    fn test_matmul_rows_balance() {
        // 32 matmul rows (Mul+Reduce chains), 8 lanes.
        // Should distribute 4 rows per lane.
        let mut g = NanoGraph::new();
        let k: u64 = 128;
        let m: u64 = 32;

        let weights = push_literal(&mut g, k * m);
        let input = push_literal(&mut g, k);

        let mut reduce_ids = Vec::new();
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
            let red = g.push_group(
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
                    base: mul,
                    stride: 1,
                }],
            );
            reduce_ids.push(red);
        }

        for &r in &reduce_ids {
            g.outputs.push(r);
        }

        let phases = plan(&g, 8, &[], &g.outputs.clone());
        verify_plan(&g, &phases, 8);
        verify_independence(&phases);

        // Check that multiple lanes have work.
        for phase in &phases {
            let active = phase
                .spans
                .iter()
                .filter(|s| s.graph.num_groups() > 0)
                .count();
            // With 32 rows and 8 lanes, we should use most lanes.
            if active < 2 {
                // This phase might just be literals; check total compute.
                let total_compute: u64 = phase
                    .spans
                    .iter()
                    .flat_map(|s| s.graph.groups())
                    .filter(|g| !is_literal_group(g))
                    .map(|g| g.count)
                    .sum();
                if total_compute > 100 {
                    panic!("Phase with {} compute atoms only uses {} lanes", total_compute, active);
                }
            }
        }
    }

    #[test]
    fn test_span_outputs_include_model_outputs() {
        let mut g = NanoGraph::new();
        let a = push_literal(&mut g, 100);
        let b = push_neg(&mut g, 100, a);
        g.outputs.push(b);

        let phases = plan(&g, 1, &[], &g.outputs.clone());
        let last_phase = phases.last().unwrap();

        let output_atoms: HashSet<u64> = last_phase
            .spans
            .iter()
            .flat_map(|s| s.outputs.iter())
            .flat_map(|r| r.base.0..r.base.0 + r.count)
            .collect();

        assert!(
            output_atoms.contains(&b.0),
            "Model output atom should be in last phase's span outputs"
        );
    }

    #[test]
    fn test_depth_computation_basic() {
        // Verify the frontier-width approach produces valid phase boundaries.
        let mut g = NanoGraph::new();
        // Simple chain: should be one phase (no pinch points).
        let a = push_literal(&mut g, 100);
        let b = push_neg(&mut g, 100, a);
        let c = push_exp(&mut g, 100, b);
        g.outputs.push(c);

        let (producers, consumers) = build_dependency_dag(&g);
        let output_set = identify_output_groups(&g, &g.outputs);
        let boundaries = find_phase_boundaries(&g, &producers, &consumers, &output_set);

        // A simple linear chain has no narrow waists — should be 0 or very few boundaries.
        assert!(
            boundaries.len() <= 1,
            "Simple linear chain should have at most 1 boundary, got {}",
            boundaries.len()
        );
    }
}
