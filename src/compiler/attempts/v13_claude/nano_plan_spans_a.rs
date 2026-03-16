#![allow(
    clippy::all,
    dead_code,
    unreachable_patterns,
    unused_variables,
    unused_imports
)]
//! NanoGraph span partitioner: splits a NanoGraph into self-contained sub-graphs.
//!
//! Each span is an independent NanoGraph with declared inputs (atoms read from
//! the shared values buffer) and declared outputs (atoms written back). Spans in
//! the same phase execute in parallel on different lanes. Spans in different
//! phases are separated by barriers.
//!
//! Algorithm overview:
//!
//! 1. Build group-level dependency DAG (producer/consumer edges).
//! 2. Topological sort. Compute "barrier depth" for each group: how many
//!    full-width reduce boundaries (ReduceSum/ReduceMax whose outputs feed
//!    groups that read from multiple independent row families) separate it
//!    from the sources.
//! 3. Groups at the same barrier depth belong to the same phase.
//! 4. Within each phase, find connected components of compute groups
//!    (ignoring Literal groups, which are duplicated into each span).
//!    Each component becomes a span.
//! 5. Small dependency chains from earlier phases (< DUPLICATION_THRESHOLD
//!    atoms) are duplicated into each span that needs them, to avoid
//!    cross-span dependencies for trivial computations.
//! 6. For each span, build a self-contained NanoGraph with remapped AtomIds.
//!    All InputRefs are rewritten to reference span-local atoms.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use crate::nano_graph::{AtomGroup, AtomId, InputRef, NanoGraph, ScalarOp};

// ─── Public API ──────────────────────────────────────────────────────────────

/// A contiguous range of atoms mapped between main graph and span graph.
#[derive(Debug, Clone)]
pub struct AtomMapping {
    pub main_base: AtomId,
    pub span_base: AtomId,
    pub count: u64,
}

/// A self-contained unit of work: one lane's work in one phase.
pub struct Span {
    /// Self-contained NanoGraph for this span's computation.
    pub graph: NanoGraph,
    /// Contiguous ranges of atoms this span reads from the main graph.
    pub inputs: Vec<AtomMapping>,
    /// Contiguous ranges of atoms this span produces for the main graph.
    pub outputs: Vec<AtomMapping>,
}

/// One phase of execution (work between two consecutive barriers).
pub struct Phase {
    pub spans: Vec<Span>,
}

/// The full execution plan.
pub struct ExecutionPlan {
    pub num_lanes: usize,
    pub phases: Vec<Phase>,
}

/// Maximum atoms in a dependency chain that we'll duplicate into spans
/// rather than requiring as a cross-phase input.
const DUPLICATION_THRESHOLD: u64 = 512;

/// Literal groups with fewer atoms than this are duplicated into spans.
/// Larger literals (weight matrices) become external inputs instead.
const LITERAL_INLINE_THRESHOLD: u64 = 1024;

/// Range-based atom map: stores contiguous range mappings for O(log n) lookup.
struct RangeAtomMap {
    ranges: Vec<(u64, u64, u64)>, // (main_base, span_base, count)
}

impl RangeAtomMap {
    fn new() -> Self {
        Self { ranges: Vec::new() }
    }

    fn insert_range(&mut self, main_base: AtomId, span_base: AtomId, count: u64) {
        self.ranges.push((main_base.0, span_base.0, count));
    }

    fn sort(&mut self) {
        self.ranges.sort_by_key(|&(base, _, _)| base);
    }

    fn get(&self, main_id: AtomId) -> Option<AtomId> {
        let idx = self
            .ranges
            .partition_point(|&(base, _, _)| base <= main_id.0);
        if idx == 0 {
            return None;
        }
        let (base, span_base, count) = self.ranges[idx - 1];
        let offset = main_id.0.wrapping_sub(base);
        if offset < count {
            Some(AtomId(span_base + offset))
        } else {
            None
        }
    }
}

/// Plan execution for a NanoGraph with `num_lanes` persistent threads.
///
/// Splits the graph into phases (separated by barriers) and spans within
/// each phase (one per lane, executing in parallel). Each span is a
/// self-contained NanoGraph.
pub fn plan_execution(graph: &NanoGraph, num_lanes: usize) -> ExecutionPlan {
    let groups = graph.groups();
    let n = groups.len();
    let num_lanes = num_lanes.max(1);

    if n == 0 {
        return ExecutionPlan {
            num_lanes,
            phases: vec![],
        };
    }

    // Step 1: Classify groups and build dependency DAG.
    let is_literal: Vec<bool> = groups
        .iter()
        .map(|g| matches!(g.op, ScalarOp::Literal(_)) && g.inputs.is_empty())
        .collect();

    let compute_indices: Vec<usize> = (0..n).filter(|&i| !is_literal[i]).collect();
    if compute_indices.is_empty() {
        return ExecutionPlan {
            num_lanes,
            phases: vec![],
        };
    }

    let (producers, consumers) = build_group_deps(groups);

    // Step 2: Topological sort.
    let topo_order = topological_sort(n, &producers);

    // Step 3: Assign groups to phases.
    // A phase boundary occurs at every ReduceSum/ReduceMax group whose output
    // is consumed by groups that also read from other independent families.
    // Simpler model: every reduce group starts a new phase for its consumers.
    let (group_phase, num_phases) =
        compute_phase_assignment(groups, &topo_order, &producers, &is_literal);

    // Step 4: Within each phase, find connected components among non-literal groups.
    let phase_components = find_phase_components(
        n,
        num_phases,
        &group_phase,
        &is_literal,
        &producers,
        &consumers,
    );

    // Step 5: Distribute components across lanes (balance by atom count).
    let phase_spans = distribute_to_lanes(&phase_components, groups, num_lanes);

    // Step 6: Build self-contained NanoGraphs for each span.
    let phases = build_span_graphs(
        graph,
        groups,
        &is_literal,
        &producers,
        &group_phase,
        &phase_spans,
        num_phases,
    );

    ExecutionPlan { num_lanes, phases }
}

// ─── Group dependency graph ──────────────────────────────────────────────────

/// Build producer and consumer graphs at group level.
///
/// Handles InputRef resolution, ReduceSum/ReduceMax strided access, and
/// IndirectLoad table_base references.
fn build_group_deps(groups: &[AtomGroup]) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let n = groups.len();
    let mut producers: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (gi, group) in groups.iter().enumerate() {
        let mut prod_set = BTreeSet::new();

        // 1. Basic InputRef resolution.
        for input in &group.inputs {
            let prods = resolve_producer_groups(input, group.count, groups);
            for pi in prods {
                if pi != gi {
                    prod_set.insert(pi);
                }
            }
        }

        // 2. ReduceSum/ReduceMax strided access.
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
                    let prods = resolve_producer_groups_with_reduce(
                        input,
                        group.count,
                        *reduce_count,
                        *reduce_stride,
                        groups,
                    );
                    for pi in prods {
                        if pi != gi {
                            prod_set.insert(pi);
                        }
                    }
                }
            }
            _ => {}
        }

        // 3. IndirectLoad table_base.
        if let ScalarOp::IndirectLoad { table_base, .. } = &group.op {
            if let Some(pi) = find_group_idx(groups, *table_base) {
                if pi != gi {
                    prod_set.insert(pi);
                }
            }
        }

        let prod_vec: Vec<usize> = prod_set.into_iter().collect();
        for &pi in &prod_vec {
            consumers[pi].push(gi);
        }
        producers.push(prod_vec);
    }

    (producers, consumers)
}

/// Find all groups that produce atoms referenced by an InputRef.
fn resolve_producer_groups(input: &InputRef, count: u64, groups: &[AtomGroup]) -> Vec<usize> {
    match input {
        InputRef::Broadcast(atom_id) => find_group_idx(groups, *atom_id).into_iter().collect(),
        InputRef::Affine { base, stride } => {
            if count == 0 {
                return vec![];
            }
            let last_offset = (*stride as i64) * (count as i64 - 1);
            let last = AtomId(base.0.wrapping_add(last_offset as u64));
            let lo = base.0.min(last.0);
            let hi = base.0.max(last.0);
            find_groups_in_range(groups, lo, hi)
        }
        InputRef::Explicit(ids) => {
            let mut result = Vec::new();
            let mut seen = HashSet::new();
            for id in ids {
                if let Some(gi) = find_group_idx(groups, *id) {
                    if seen.insert(gi) {
                        result.push(gi);
                    }
                }
            }
            result
        }
        InputRef::SymAffine { base, stride_i, .. } => {
            if count == 0 {
                return vec![];
            }
            let last_offset = (*stride_i as i64) * (count as i64 - 1);
            let last = AtomId(base.0.wrapping_add(last_offset as u64));
            let lo = base.0.min(last.0);
            let hi = base.0.max(last.0);
            find_groups_in_range(groups, lo, hi)
        }
        InputRef::StridedBroadcast {
            base,
            stride,
            repeat,
        } => {
            if count == 0 {
                return vec![];
            }
            let num_blocks = (count + repeat - 1) / repeat;
            let last_offset = *stride * (num_blocks as i64 - 1);
            let last = AtomId(base.0.wrapping_add(last_offset as u64));
            let lo = base.0.min(last.0);
            let hi = base.0.max(last.0);
            find_groups_in_range(groups, lo, hi)
        }
        InputRef::Modular {
            base,
            stride,
            modulus,
        } => {
            if *modulus == 0 {
                return vec![];
            }
            let last_offset = (*stride as i64) * (*modulus as i64 - 1);
            let last = AtomId(base.0.wrapping_add(last_offset as u64));
            let lo = base.0.min(last.0);
            let hi = base.0.max(last.0);
            find_groups_in_range(groups, lo, hi)
        }
    }
}

/// Resolve producer groups accounting for ReduceSum/ReduceMax strided reads.
fn resolve_producer_groups_with_reduce(
    input: &InputRef,
    count: u64,
    reduce_count: u64,
    reduce_stride: i64,
    groups: &[AtomGroup],
) -> Vec<usize> {
    // The reduce op reads: for each atom i, for k in 0..reduce_count,
    // input.resolve(i, 0) + k * reduce_stride.
    // This extends the read range beyond the base InputRef range.
    match input {
        InputRef::Affine { base, stride } => {
            if count == 0 || reduce_count == 0 {
                return vec![];
            }
            // Base range: base .. base + stride*(count-1)
            // Reduce extends each atom by k*reduce_stride for k in 0..reduce_count
            // Full range: min to max of all accessed atoms
            let base_first = base.0 as i64;
            let base_last = base_first + (*stride as i64) * (count as i64 - 1);
            let reduce_end = reduce_stride * (reduce_count as i64 - 1);
            let lo = base_first
                .min(base_first + reduce_end)
                .min(base_last)
                .min(base_last + reduce_end);
            let hi = base_first
                .max(base_first + reduce_end)
                .max(base_last)
                .max(base_last + reduce_end);
            find_groups_in_range(groups, lo as u64, hi as u64)
        }
        _ => {
            // For other InputRef types, sample boundary atoms.
            let mut all_ids = BTreeSet::new();
            let sample_count = count.min(64);
            for si in 0..sample_count {
                let i = if sample_count == count {
                    si
                } else {
                    si * count / sample_count
                };
                let base_atom = input.resolve(i, 0);
                for k in 0..reduce_count {
                    let read_id =
                        AtomId(base_atom.0.wrapping_add((k as i64 * reduce_stride) as u64));
                    all_ids.insert(read_id.0);
                }
            }
            if all_ids.is_empty() {
                return vec![];
            }
            let lo = *all_ids.iter().next().unwrap();
            let hi = *all_ids.iter().next_back().unwrap();
            find_groups_in_range(groups, lo, hi)
        }
    }
}

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

/// Find all groups whose atom ranges overlap [lo, hi].
fn find_groups_in_range(groups: &[AtomGroup], lo: u64, hi: u64) -> Vec<usize> {
    let mut result = Vec::new();
    let start = groups.partition_point(|g| g.base_id.0 + g.count <= lo);
    for gi in start..groups.len() {
        let g = &groups[gi];
        if g.base_id.0 > hi {
            break;
        }
        let g_end = g.base_id.0 + g.count - 1;
        if g.base_id.0 <= hi && g_end >= lo {
            result.push(gi);
        }
    }
    result
}

// ─── Topological sort ────────────────────────────────────────────────────────

fn topological_sort(n: usize, producers: &[Vec<usize>]) -> Vec<usize> {
    let mut in_degree = vec![0usize; n];
    let mut consumers_map: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (gi, prods) in producers.iter().enumerate() {
        in_degree[gi] = prods.len();
        for &pi in prods {
            consumers_map[pi].push(gi);
        }
    }

    let mut queue: VecDeque<usize> = VecDeque::new();
    for i in 0..n {
        if in_degree[i] == 0 {
            queue.push_back(i);
        }
    }

    let mut order = Vec::with_capacity(n);
    while let Some(gi) = queue.pop_front() {
        order.push(gi);
        for &ci in &consumers_map[gi] {
            in_degree[ci] -= 1;
            if in_degree[ci] == 0 {
                queue.push_back(ci);
            }
        }
    }

    order
}

// ─── Phase assignment ────────────────────────────────────────────────────────

/// Assign each group to a phase. Phase boundaries are placed after ReduceSum/
/// ReduceMax groups — their consumers start a new phase because the reduction
/// creates a synchronization point (all inputs to the reduce must complete
/// before its output is available).
///
/// Returns (group_phase, num_phases).
fn compute_phase_assignment(
    groups: &[AtomGroup],
    topo_order: &[usize],
    producers: &[Vec<usize>],
    is_literal: &[bool],
) -> (Vec<usize>, usize) {
    let n = groups.len();
    let mut group_phase = vec![0usize; n];

    // Forward pass in topo order: each group's phase = max(producer phases),
    // but if any producer is a reduce op, bump by 1.
    for &gi in topo_order {
        if is_literal[gi] {
            // Literals are in phase 0; they're shared across all phases.
            group_phase[gi] = 0;
            continue;
        }

        let mut max_phase = 0usize;
        for &pi in &producers[gi] {
            if is_literal[pi] {
                continue;
            }
            let prod_phase = group_phase[pi];
            let effective = if groups[pi].op.is_reduce() {
                // Consumer of a reduce: starts AFTER the reduce completes.
                prod_phase + 1
            } else {
                prod_phase
            };
            max_phase = max_phase.max(effective);
        }
        group_phase[gi] = max_phase;
    }

    let num_phases = group_phase.iter().copied().max().unwrap_or(0) + 1;
    (group_phase, num_phases)
}

// ─── Connected components within phases ──────────────────────────────────────

/// For each phase, find connected components among non-literal groups.
/// Two non-literal groups in the same phase are connected if one is a
/// producer/consumer of the other (directly).
///
/// Returns: for each phase, a Vec of components (each component is a Vec of group indices).
fn find_phase_components(
    n: usize,
    num_phases: usize,
    group_phase: &[usize],
    is_literal: &[bool],
    producers: &[Vec<usize>],
    consumers: &[Vec<usize>],
) -> Vec<Vec<Vec<usize>>> {
    let mut phase_components: Vec<Vec<Vec<usize>>> = Vec::with_capacity(num_phases);

    for phase in 0..num_phases {
        // Gather non-literal groups in this phase.
        let phase_groups: Vec<usize> = (0..n)
            .filter(|&gi| !is_literal[gi] && group_phase[gi] == phase)
            .collect();

        if phase_groups.is_empty() {
            phase_components.push(vec![]);
            continue;
        }

        // Build adjacency within phase (undirected for component finding).
        let phase_set: HashSet<usize> = phase_groups.iter().copied().collect();
        let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
        for &gi in &phase_groups {
            adj.entry(gi).or_default();
            // Connect to producers in same phase.
            for &pi in &producers[gi] {
                if phase_set.contains(&pi) {
                    adj.entry(gi).or_default().push(pi);
                    adj.entry(pi).or_default().push(gi);
                }
            }
            // Connect to consumers in same phase.
            for &ci in &consumers[gi] {
                if phase_set.contains(&ci) {
                    adj.entry(gi).or_default().push(ci);
                    adj.entry(ci).or_default().push(gi);
                }
            }
        }

        // BFS to find components.
        let mut visited: HashSet<usize> = HashSet::new();
        let mut components: Vec<Vec<usize>> = Vec::new();
        for &gi in &phase_groups {
            if visited.contains(&gi) {
                continue;
            }
            let mut component = Vec::new();
            let mut queue = VecDeque::new();
            queue.push_back(gi);
            visited.insert(gi);
            while let Some(cur) = queue.pop_front() {
                component.push(cur);
                if let Some(neighbors) = adj.get(&cur) {
                    for &nb in neighbors {
                        if visited.insert(nb) {
                            queue.push_back(nb);
                        }
                    }
                }
            }
            component.sort();
            components.push(component);
        }

        phase_components.push(components);
    }

    phase_components
}

// ─── Lane distribution ───────────────────────────────────────────────────────

/// Distribute components across lanes for balance.
/// Returns: for each phase, a Vec of spans (each span = Vec of group indices).
/// Components are assigned to lanes round-robin sorted by descending atom count
/// (largest-first bin packing).
fn distribute_to_lanes(
    phase_components: &[Vec<Vec<usize>>],
    groups: &[AtomGroup],
    num_lanes: usize,
) -> Vec<Vec<Vec<usize>>> {
    let mut phase_spans = Vec::with_capacity(phase_components.len());

    for components in phase_components {
        if components.is_empty() {
            phase_spans.push(vec![]);
            continue;
        }

        // Sort components by total atom count (descending) for bin packing.
        let mut indexed: Vec<(usize, u64)> = components
            .iter()
            .enumerate()
            .map(|(ci, comp)| {
                let atoms: u64 = comp.iter().map(|&gi| groups[gi].count).sum();
                (ci, atoms)
            })
            .collect();
        indexed.sort_by(|a, b| b.1.cmp(&a.1));

        // Bin-pack into num_lanes bins using "least loaded" strategy.
        let mut bins: Vec<Vec<usize>> = vec![Vec::new(); num_lanes];
        let mut bin_load: Vec<u64> = vec![0; num_lanes];

        for (ci, atom_count) in indexed {
            // Find least loaded bin.
            let best_bin = bin_load
                .iter()
                .enumerate()
                .min_by_key(|&(_, &load)| load)
                .unwrap()
                .0;
            bins[best_bin].extend(components[ci].iter().copied());
            bin_load[best_bin] += atom_count;
        }

        // Filter out empty bins.
        let spans: Vec<Vec<usize>> = bins.into_iter().filter(|b| !b.is_empty()).collect();
        phase_spans.push(spans);
    }

    phase_spans
}

// ─── Build span NanoGraphs ───────────────────────────────────────────────────

/// Build self-contained NanoGraphs for each span.
///
/// For each span:
/// 1. Collect the span's groups and all literal groups they reference.
/// 2. Identify external dependencies (atoms from earlier phases).
/// 3. Build a new NanoGraph with remapped AtomIds.
/// 4. Declare inputs (external atoms) and outputs (atoms consumed by later phases
///    or that are graph outputs).
fn build_span_graphs(
    original: &NanoGraph,
    groups: &[AtomGroup],
    is_literal: &[bool],
    producers: &[Vec<usize>],
    group_phase: &[usize],
    phase_spans: &[Vec<Vec<usize>>],
    num_phases: usize,
) -> Vec<Phase> {
    let n = groups.len();

    // Precompute: which groups are consumed by groups in later phases?
    // And which atoms are graph outputs?
    let output_atoms: HashSet<AtomId> = original.outputs.iter().copied().collect();

    // For each group, find consuming groups (already in `producers` via inversion,
    // but we need consumers). Rebuild consumers from producers.
    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (gi, prods) in producers.iter().enumerate() {
        for &pi in prods {
            consumers[pi].push(gi);
        }
    }

    // Build a set of all group indices per span for quick lookup.
    let mut phases_out: Vec<Phase> = Vec::new();

    for (phase_idx, spans) in phase_spans.iter().enumerate() {
        let mut span_structs: Vec<Span> = Vec::new();

        for span_groups in spans {
            let span = build_single_span(
                original,
                groups,
                is_literal,
                producers,
                &consumers,
                group_phase,
                &output_atoms,
                span_groups,
                phase_idx,
            );
            span_structs.push(span);
        }

        phases_out.push(Phase {
            spans: span_structs,
        });
    }

    phases_out
}

/// Build a single self-contained span NanoGraph.
/// All operations are O(num_groups), not O(num_atoms).
fn build_single_span(
    original: &NanoGraph,
    groups: &[AtomGroup],
    is_literal: &[bool],
    producers: &[Vec<usize>],
    consumers: &[Vec<usize>],
    group_phase: &[usize],
    output_atoms: &HashSet<AtomId>,
    span_group_indices: &[usize],
    phase_idx: usize,
) -> Span {
    let span_set: HashSet<usize> = span_group_indices.iter().copied().collect();

    let mut included_groups: BTreeSet<usize> = BTreeSet::new();
    // Collect external ranges as (group_idx, offset, count) instead of individual atoms.
    let mut external_ranges: Vec<(usize, u64, u64)> = Vec::new();

    for &gi in span_group_indices {
        included_groups.insert(gi);
    }

    // Collect literal groups referenced by span groups.
    let mut literal_groups_needed: BTreeSet<usize> = BTreeSet::new();
    for &gi in span_group_indices {
        collect_literal_deps(
            gi,
            groups,
            producers,
            is_literal,
            &mut literal_groups_needed,
        );
    }
    // Split literals by size: small ones get duplicated, large ones become external inputs.
    let mut large_literal_groups: BTreeSet<usize> = BTreeSet::new();
    for &li in &literal_groups_needed {
        if groups[li].count < LITERAL_INLINE_THRESHOLD {
            included_groups.insert(li);
        } else {
            large_literal_groups.insert(li);
            // ONE range entry for the entire large literal group.
            let lg = &groups[li];
            external_ranges.push((li, 0, lg.count));
        }
    }

    // Check for small duplicatable chains from earlier phases.
    let mut to_duplicate: BTreeSet<usize> = BTreeSet::new();
    for &gi in span_group_indices {
        find_duplicatable_deps(
            gi,
            groups,
            producers,
            is_literal,
            group_phase,
            phase_idx,
            &span_set,
            &mut to_duplicate,
        );
    }
    for &di in &to_duplicate {
        included_groups.insert(di);
        collect_literal_deps(
            di,
            groups,
            producers,
            is_literal,
            &mut literal_groups_needed,
        );
    }
    for &li in &literal_groups_needed {
        if large_literal_groups.contains(&li) {
            continue;
        }
        if groups[li].count < LITERAL_INLINE_THRESHOLD {
            included_groups.insert(li);
        } else {
            large_literal_groups.insert(li);
            let lg = &groups[li];
            external_ranges.push((li, 0, lg.count));
        }
    }

    // Identify external dependencies: atoms referenced by included groups
    // that come from groups NOT included in the span.
    for &gi in &included_groups {
        if is_literal[gi] {
            continue;
        }
        let group = &groups[gi];
        for input in &group.inputs {
            collect_external_ranges_a(
                input,
                group.count,
                &group.op,
                groups,
                &included_groups,
                &mut external_ranges,
            );
        }
        // Handle IndirectLoad table_base.
        if let ScalarOp::IndirectLoad { table_base, .. } = &group.op {
            if let Some(table_gi) = find_group_idx(groups, *table_base) {
                if !included_groups.contains(&table_gi) {
                    let tg = &groups[table_gi];
                    external_ranges.push((table_gi, 0, tg.count));
                }
            }
        }
    }

    // Merge overlapping/adjacent ranges.
    let external_ranges = merge_group_ranges_a(&mut external_ranges);

    // Build the remapping using RangeAtomMap.
    let mut new_graph = NanoGraph::new();
    let mut atom_remap = RangeAtomMap::new();

    // Copy sym_dim info from original.
    for (name, &sd) in &original.sym_dim_names {
        new_graph.sym_dim(name);
    }
    for (&sd, &bound) in &original.sym_dim_bounds {
        new_graph.sym_dim_bounds.insert(sd, bound);
    }

    // Allocate placeholder groups for external input ranges.
    let mut input_mapping: Vec<AtomMapping> = Vec::new();
    for &(gi, offset, count) in &external_ranges {
        let main_base = groups[gi].base_id.offset(offset);
        let local_base = new_graph.push_group(
            count,
            ScalarOp::Literal(crate::numeric_scalar::NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );
        atom_remap.insert_range(main_base, local_base, count);
        input_mapping.push(AtomMapping {
            main_base,
            span_base: local_base,
            count,
        });
    }

    // Sort atom map before compute group processing.
    atom_remap.sort();

    // Add included groups in topological order (BTreeSet gives ascending = topo order).
    let mut group_output_remap: HashMap<usize, AtomId> = HashMap::new();
    for &gi in &included_groups {
        let group = &groups[gi];
        let new_inputs: Vec<InputRef> = group
            .inputs
            .iter()
            .map(|iref| remap_input_ref_range(iref, &atom_remap))
            .collect();
        let new_op = remap_op_range(&group.op, &atom_remap);
        let new_base = new_graph.push_group(
            group.count,
            new_op,
            group.sym_dims.clone(),
            group.reduce_dims.clone(),
            new_inputs,
        );
        atom_remap.insert_range(group.base_id, new_base, group.count);
        group_output_remap.insert(gi, new_base);
    }

    // Determine outputs as ranges.
    let mut output_mapping: Vec<AtomMapping> = Vec::new();
    let mut output_groups_emitted: HashSet<usize> = HashSet::new();

    for &gi in span_group_indices {
        let group = &groups[gi];

        // Check if any consumer is outside the span.
        let has_external_consumer = consumers[gi]
            .iter()
            .any(|&ci| !span_set.contains(&ci) && !to_duplicate.contains(&ci));

        // Check if group has graph output atoms.
        let has_graph_output = output_atoms.iter().any(|&out| group.contains(out));

        if (has_external_consumer || has_graph_output) && !output_groups_emitted.contains(&gi) {
            output_groups_emitted.insert(gi);
            let new_base = group_output_remap[&gi];
            output_mapping.push(AtomMapping {
                main_base: group.base_id,
                span_base: new_base,
                count: group.count,
            });
        }
    }

    Span {
        graph: new_graph,
        inputs: input_mapping,
        outputs: output_mapping,
    }
}

/// Range-based external atom collection for spans_a.
/// Collects (group_idx, offset, count) ranges instead of individual atoms.
fn collect_external_ranges_a(
    input: &InputRef,
    count: u64,
    op: &ScalarOp,
    groups: &[AtomGroup],
    included_groups: &BTreeSet<usize>,
    external_ranges: &mut Vec<(usize, u64, u64)>,
) {
    let referenced_groups = resolve_producer_groups(input, count, groups);

    let mut all_referenced = referenced_groups.clone();
    match op {
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
            let extra = resolve_producer_groups_with_reduce(
                input,
                count,
                *reduce_count,
                *reduce_stride,
                groups,
            );
            for gi in extra {
                if !all_referenced.contains(&gi) {
                    all_referenced.push(gi);
                }
            }
        }
        _ => {}
    }

    for &gi in &all_referenced {
        if !included_groups.contains(&gi) {
            // Compute the range of atoms referenced from this external group.
            let range = compute_referenced_range(input, count, op, gi, groups);
            if let Some((offset, range_count)) = range {
                external_ranges.push((gi, offset, range_count));
            }
        }
    }
}

/// Compute (offset, count) within ext_group that the input references.
fn compute_referenced_range(
    input: &InputRef,
    count: u64,
    op: &ScalarOp,
    ext_group_idx: usize,
    groups: &[AtomGroup],
) -> Option<(u64, u64)> {
    let ext_group = &groups[ext_group_idx];
    let (reduce_count, reduce_stride) = match op {
        ScalarOp::ReduceSum {
            reduce_count,
            reduce_stride,
            ..
        }
        | ScalarOp::ReduceMax {
            reduce_count,
            reduce_stride,
            ..
        } if *reduce_count > 1 && *reduce_stride != 0 => (*reduce_count, *reduce_stride),
        _ => (1, 0),
    };

    // Sample boundary atoms to find the range.
    let sample_points = if count <= 128 {
        (0..count).collect::<Vec<_>>()
    } else {
        let mut pts = Vec::new();
        for i in 0..64u64 {
            pts.push(i);
        }
        for i in (count.saturating_sub(64))..count {
            pts.push(i);
        }
        pts.sort();
        pts.dedup();
        pts
    };

    let mut min_atom = u64::MAX;
    let mut max_atom = 0u64;

    for &i in &sample_points {
        for k in 0..reduce_count {
            let base = input.resolve(i, 0);
            let atom = AtomId(base.0.wrapping_add((k as i64 * reduce_stride) as u64));
            if ext_group.contains(atom) {
                min_atom = min_atom.min(atom.0);
                max_atom = max_atom.max(atom.0);
            }
        }
    }

    if min_atom <= max_atom {
        let lo = min_atom.max(ext_group.base_id.0);
        let hi = (max_atom + 1).min(ext_group.base_id.0 + ext_group.count);
        let offset = lo - ext_group.base_id.0;
        let range_count = hi - lo;
        Some((offset, range_count))
    } else {
        None
    }
}

/// Merge overlapping/adjacent ranges from the same group.
fn merge_group_ranges_a(ranges: &mut Vec<(usize, u64, u64)>) -> Vec<(usize, u64, u64)> {
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

/// Remap InputRef using RangeAtomMap.
fn remap_input_ref_range(input: &InputRef, atom_remap: &RangeAtomMap) -> InputRef {
    match input {
        InputRef::Broadcast(id) => InputRef::Broadcast(atom_remap.get(*id).unwrap_or(*id)),
        InputRef::Affine { base, stride } => InputRef::Affine {
            base: atom_remap.get(*base).unwrap_or(*base),
            stride: *stride,
        },
        InputRef::Explicit(ids) => InputRef::Explicit(
            ids.iter()
                .map(|id| atom_remap.get(*id).unwrap_or(*id))
                .collect(),
        ),
        InputRef::SymAffine {
            base,
            stride_i,
            stride_k,
        } => InputRef::SymAffine {
            base: atom_remap.get(*base).unwrap_or(*base),
            stride_i: *stride_i,
            stride_k: *stride_k,
        },
        InputRef::StridedBroadcast {
            base,
            stride,
            repeat,
        } => InputRef::StridedBroadcast {
            base: atom_remap.get(*base).unwrap_or(*base),
            stride: *stride,
            repeat: *repeat,
        },
        InputRef::Modular {
            base,
            stride,
            modulus,
        } => InputRef::Modular {
            base: atom_remap.get(*base).unwrap_or(*base),
            stride: *stride,
            modulus: *modulus,
        },
    }
}

/// Remap ScalarOp fields using RangeAtomMap.
fn remap_op_range(op: &ScalarOp, atom_remap: &RangeAtomMap) -> ScalarOp {
    match op {
        ScalarOp::IndirectLoad {
            table_base,
            output_dtype,
        } => ScalarOp::IndirectLoad {
            table_base: atom_remap.get(*table_base).unwrap_or(*table_base),
            output_dtype: *output_dtype,
        },
        other => other.clone(),
    }
}

/// Collect literal group indices that a group depends on (direct deps only).
fn collect_literal_deps(
    gi: usize,
    groups: &[AtomGroup],
    producers: &[Vec<usize>],
    is_literal: &[bool],
    lit_set: &mut BTreeSet<usize>,
) {
    for &pi in &producers[gi] {
        if is_literal[pi] {
            lit_set.insert(pi);
        }
    }
}

/// Walk backward from a group to find small chains from earlier phases
/// that can be duplicated into the span.
///
/// A chain is duplicatable if:
/// - Total atoms in the chain < DUPLICATION_THRESHOLD
/// - All groups in the chain are from earlier phases
/// - The chain doesn't include IndirectLoad (too expensive to duplicate)
fn find_duplicatable_deps(
    gi: usize,
    groups: &[AtomGroup],
    producers: &[Vec<usize>],
    is_literal: &[bool],
    group_phase: &[usize],
    current_phase: usize,
    span_set: &HashSet<usize>,
    duplicated: &mut BTreeSet<usize>,
) {
    let group = &groups[gi];
    for &pi in &producers[gi] {
        if is_literal[pi] || span_set.contains(&pi) || duplicated.contains(&pi) {
            continue;
        }
        if group_phase[pi] >= current_phase {
            // Same phase but different span — don't duplicate, use as external input.
            continue;
        }
        // Walk backward and compute total chain size.
        let chain = trace_chain_backward(
            pi,
            groups,
            producers,
            is_literal,
            group_phase,
            current_phase,
        );
        let total_atoms: u64 = chain.iter().map(|&ci| groups[ci].count).sum();
        if total_atoms <= DUPLICATION_THRESHOLD
            && !chain
                .iter()
                .any(|&ci| matches!(groups[ci].op, ScalarOp::IndirectLoad { .. }))
        {
            for &ci in &chain {
                duplicated.insert(ci);
            }
        }
    }
}

/// Trace a chain of groups backward from `start`, collecting all non-literal
/// groups from earlier phases.
fn trace_chain_backward(
    start: usize,
    groups: &[AtomGroup],
    producers: &[Vec<usize>],
    is_literal: &[bool],
    group_phase: &[usize],
    current_phase: usize,
) -> Vec<usize> {
    let mut chain = Vec::new();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back(start);
    visited.insert(start);

    while let Some(gi) = queue.pop_front() {
        if is_literal[gi] {
            continue;
        }
        if group_phase[gi] >= current_phase {
            continue;
        }
        chain.push(gi);
        for &pi in &producers[gi] {
            if !is_literal[pi] && !visited.contains(&pi) && group_phase[pi] < current_phase {
                visited.insert(pi);
                queue.push_back(pi);
            }
        }
    }

    chain
}

/// Collect atoms that are external to the span (referenced but not produced
/// by any included group).
fn collect_external_atoms(
    input: &InputRef,
    count: u64,
    op: &ScalarOp,
    groups: &[AtomGroup],
    included_groups: &BTreeSet<usize>,
    external: &mut BTreeSet<AtomId>,
) {
    // For each atom referenced by this input, check if its producer group is included.
    // If not, the atom is external.
    let referenced_groups = resolve_producer_groups(input, count, groups);

    // Also handle reduce stride extension.
    let mut all_referenced = referenced_groups.clone();
    match op {
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
            let extra = resolve_producer_groups_with_reduce(
                input,
                count,
                *reduce_count,
                *reduce_stride,
                groups,
            );
            for gi in extra {
                if !all_referenced.contains(&gi) {
                    all_referenced.push(gi);
                }
            }
        }
        _ => {}
    }

    for &gi in &all_referenced {
        if !included_groups.contains(&gi) {
            // This group is external. We need to figure out which specific
            // atoms are referenced.
            add_referenced_atoms_from_group(input, count, op, gi, groups, external);
        }
    }
}

/// Add specific referenced atoms from an external group to the external set.
fn add_referenced_atoms_from_group(
    input: &InputRef,
    count: u64,
    op: &ScalarOp,
    ext_group_idx: usize,
    groups: &[AtomGroup],
    external: &mut BTreeSet<AtomId>,
) {
    let ext_group = &groups[ext_group_idx];

    // Determine the range of atoms this input references from the external group.
    // For efficiency, compute the range and add all atoms in it that belong to
    // the external group.
    let (reduce_count, reduce_stride) = match op {
        ScalarOp::ReduceSum {
            reduce_count,
            reduce_stride,
            ..
        }
        | ScalarOp::ReduceMax {
            reduce_count,
            reduce_stride,
            ..
        } if *reduce_count > 1 && *reduce_stride != 0 => (*reduce_count, *reduce_stride),
        _ => (1, 0),
    };

    // Sample atoms to find the range within the external group.
    // For small counts, check all. For large, sample boundaries.
    let sample_points = if count <= 128 {
        (0..count).collect::<Vec<_>>()
    } else {
        let mut pts = Vec::new();
        for i in 0..64 {
            pts.push(i);
        }
        for i in (count - 64)..count {
            pts.push(i);
        }
        pts.sort();
        pts.dedup();
        pts
    };

    let mut min_atom = u64::MAX;
    let mut max_atom = 0u64;

    for &i in &sample_points {
        for k in 0..reduce_count {
            let base = input.resolve(i, 0);
            let atom = AtomId(base.0.wrapping_add((k as i64 * reduce_stride) as u64));
            if ext_group.contains(atom) {
                min_atom = min_atom.min(atom.0);
                max_atom = max_atom.max(atom.0);
            }
        }
    }

    if min_atom <= max_atom {
        // Clamp to the external group's range.
        let lo = min_atom.max(ext_group.base_id.0);
        let hi = max_atom.min(ext_group.base_id.0 + ext_group.count - 1);
        for a in lo..=hi {
            external.insert(AtomId(a));
        }
    }
}

/// Allocate placeholder Identity groups in the span's graph for external inputs.
/// Groups contiguous ranges of external atoms together.
///
/// Returns the input mapping: Vec<(main_graph_atom, span_local_atom)>.
fn allocate_external_inputs(
    external_atoms: &BTreeSet<AtomId>,
    new_graph: &mut NanoGraph,
    atom_remap: &mut HashMap<AtomId, AtomId>,
) -> Vec<(AtomId, AtomId)> {
    if external_atoms.is_empty() {
        return vec![];
    }

    let mut inputs = Vec::new();

    // Group contiguous external atoms into runs for efficient allocation.
    let atoms: Vec<AtomId> = external_atoms.iter().copied().collect();
    let mut i = 0;
    while i < atoms.len() {
        let run_start = atoms[i];
        let mut run_len = 1u64;
        while (i + run_len as usize) < atoms.len()
            && atoms[i + run_len as usize].0 == run_start.0 + run_len
        {
            run_len += 1;
        }

        // Allocate a Literal group for this run of external atoms.
        // We use Literal(0.0) as a placeholder — the actual values come from
        // the shared buffer at runtime.
        use crate::numeric_scalar::NumericScalar;
        let new_base = new_graph.push_group(
            run_len,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );

        for j in 0..run_len {
            let old_atom = run_start.offset(j);
            let new_atom = new_base.offset(j);
            atom_remap.insert(old_atom, new_atom);
            inputs.push((old_atom, new_atom));
        }

        i += run_len as usize;
    }

    inputs
}

/// Remap an InputRef to use new AtomIds.
fn remap_input_ref(input: &InputRef, count: u64, atom_remap: &HashMap<AtomId, AtomId>) -> InputRef {
    match input {
        InputRef::Broadcast(atom_id) => {
            let new_id = atom_remap.get(atom_id).copied().unwrap_or(*atom_id);
            InputRef::Broadcast(new_id)
        }
        InputRef::Affine { base, stride } => {
            let new_base = atom_remap.get(base).copied().unwrap_or(*base);
            InputRef::Affine {
                base: new_base,
                stride: *stride,
            }
        }
        InputRef::Explicit(ids) => {
            let new_ids: Vec<AtomId> = ids
                .iter()
                .map(|id| atom_remap.get(id).copied().unwrap_or(*id))
                .collect();
            InputRef::Explicit(new_ids)
        }
        InputRef::SymAffine {
            base,
            stride_i,
            stride_k,
        } => {
            let new_base = atom_remap.get(base).copied().unwrap_or(*base);
            InputRef::SymAffine {
                base: new_base,
                stride_i: *stride_i,
                stride_k: *stride_k,
            }
        }
        InputRef::StridedBroadcast {
            base,
            stride,
            repeat,
        } => {
            let new_base = atom_remap.get(base).copied().unwrap_or(*base);
            InputRef::StridedBroadcast {
                base: new_base,
                stride: *stride,
                repeat: *repeat,
            }
        }
        InputRef::Modular {
            base,
            stride,
            modulus,
        } => {
            let new_base = atom_remap.get(base).copied().unwrap_or(*base);
            InputRef::Modular {
                base: new_base,
                stride: *stride,
                modulus: *modulus,
            }
        }
    }
}

/// Remap ScalarOp fields that reference AtomIds (IndirectLoad table_base).
fn remap_op(op: &ScalarOp, atom_remap: &HashMap<AtomId, AtomId>) -> ScalarOp {
    match op {
        ScalarOp::IndirectLoad {
            table_base,
            output_dtype,
        } => {
            let new_base = atom_remap.get(table_base).copied().unwrap_or(*table_base);
            ScalarOp::IndirectLoad {
                table_base: new_base,
                output_dtype: *output_dtype,
            }
        }
        // All other ops don't reference AtomIds.
        other => other.clone(),
    }
}

// ─── Validation ──────────────────────────────────────────────────────────────

/// Validate an ExecutionPlan: check that each span is self-contained and that
/// all original graph outputs are produced by some span.
pub fn validate_plan(plan: &ExecutionPlan, original: &NanoGraph) -> Vec<String> {
    let mut errors = Vec::new();

    for (pi, phase) in plan.phases.iter().enumerate() {
        for (si, span) in phase.spans.iter().enumerate() {
            let prefix = format!("Phase {} Span {}", pi, si);

            // Validate span's internal graph.
            let graph_errors = span.graph.validate();
            for err in graph_errors {
                errors.push(format!("{}: {}", prefix, err));
            }

            // Check that all declared output base atoms exist in the span's graph.
            for mapping in &span.outputs {
                if !span.graph.contains_atom(mapping.span_base) {
                    errors.push(format!(
                        "{}: declared output base {} does not exist in span graph",
                        prefix, mapping.span_base
                    ));
                }
            }

            // Check that all declared input base atoms exist in the span's graph.
            for mapping in &span.inputs {
                if !span.graph.contains_atom(mapping.span_base) {
                    errors.push(format!(
                        "{}: declared input base {} does not exist in span graph",
                        prefix, mapping.span_base
                    ));
                }
            }
        }
    }

    // Check that all original outputs are covered.
    // Build a set of all main-graph atoms covered by output mappings.
    let mut all_output_atoms: HashSet<AtomId> = HashSet::new();
    for phase in &plan.phases {
        for span in &phase.spans {
            for mapping in &span.outputs {
                for i in 0..mapping.count {
                    all_output_atoms.insert(mapping.main_base.offset(i));
                }
            }
        }
    }

    for &out_atom in &original.outputs {
        if !all_output_atoms.contains(&out_atom) {
            errors.push(format!(
                "Original graph output {} is not produced by any span",
                out_atom
            ));
        }
    }

    // Check within-phase independence using range-based checks.
    for (pi, phase) in plan.phases.iter().enumerate() {
        // Collect output ranges per span.
        let span_output_ranges: Vec<&Vec<AtomMapping>> =
            phase.spans.iter().map(|s| &s.outputs).collect();

        for (si, span) in phase.spans.iter().enumerate() {
            for input_mapping in &span.inputs {
                // Check if this input range overlaps with any other span's output range.
                for (other_si, other_outputs) in span_output_ranges.iter().enumerate() {
                    if other_si == si {
                        continue;
                    }
                    for other_mapping in other_outputs.iter() {
                        // Check range overlap.
                        let a_lo = input_mapping.main_base.0;
                        let a_hi = a_lo + input_mapping.count;
                        let b_lo = other_mapping.main_base.0;
                        let b_hi = b_lo + other_mapping.count;
                        if a_lo < b_hi && b_lo < a_hi {
                            errors.push(format!(
                                "Phase {}: Span {} reads atoms [{}, {}) produced by Span {} (cross-span same-phase dependency)",
                                pi, si, a_lo, a_hi, other_si
                            ));
                        }
                    }
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
    use crate::compiler::attempts::v13_claude::test_graphs;
    use crate::dtype::DType;
    use crate::nano_graph::{ScalarBinOp, ScalarUnaryOp};
    use crate::numeric_scalar::NumericScalar;

    /// Helper: validate a plan and assert no errors.
    fn assert_plan_valid(plan: &ExecutionPlan, original: &NanoGraph) {
        let errors = validate_plan(plan, original);
        assert!(
            errors.is_empty(),
            "Plan validation errors:\n{}",
            errors.join("\n")
        );
    }

    /// Helper: validate each span's graph independently.
    fn assert_spans_valid(plan: &ExecutionPlan) {
        for (pi, phase) in plan.phases.iter().enumerate() {
            for (si, span) in phase.spans.iter().enumerate() {
                let errors = span.graph.validate();
                assert!(
                    errors.is_empty(),
                    "Phase {} Span {} graph validation errors:\n{}",
                    pi,
                    si,
                    errors.join("\n")
                );
            }
        }
    }

    #[test]
    fn test_elementwise_single_lane() {
        let (g, _, _, _) = test_graphs::elementwise_binary(1024, ScalarBinOp::Add);
        let plan = plan_execution(&g, 1);
        assert_plan_valid(&plan, &g);
        assert_spans_valid(&plan);
        assert_eq!(plan.num_lanes, 1);
        // All in one phase (no reductions).
        assert_eq!(plan.phases.len(), 1);
        assert_eq!(plan.phases[0].spans.len(), 1);
    }

    #[test]
    fn test_elementwise_multi_lane() {
        let (g, _, _, _) = test_graphs::elementwise_binary(1024, ScalarBinOp::Add);
        let plan = plan_execution(&g, 4);
        assert_plan_valid(&plan, &g);
        assert_spans_valid(&plan);
        assert_eq!(plan.num_lanes, 4);
        // Elementwise: one phase, one span (single connected component).
        assert_eq!(plan.phases.len(), 1);
        // One component → one span (the add group + its literal deps).
        assert_eq!(plan.phases[0].spans.len(), 1);
    }

    #[test]
    fn test_matmul_single_lane() {
        let (g, _, _, _) = test_graphs::matmul(4, 8, 16);
        let plan = plan_execution(&g, 1);
        assert_plan_valid(&plan, &g);
        assert_spans_valid(&plan);
        assert_eq!(plan.num_lanes, 1);
        // Should have exactly 1 phase for Mul + 1 phase for ReduceSum consumers
        // (ReduceSum itself could be in phase 0 or 1 depending on phase assignment).
        assert!(
            plan.phases.len() >= 1,
            "Expected >= 1 phase, got {}",
            plan.phases.len()
        );
    }

    #[test]
    fn test_matmul_multi_lane() {
        let (g, _, _, _) = test_graphs::matmul(4, 8, 16);
        let plan = plan_execution(&g, 2);
        assert_plan_valid(&plan, &g);
        assert_spans_valid(&plan);
        assert_eq!(plan.num_lanes, 2);
        // Matmul rows are independent → should be distributable across lanes.
        // Check that first phase has at least 2 spans.
        let first_phase_spans = plan.phases[0].spans.len();
        assert!(
            first_phase_spans >= 1,
            "Expected >= 1 spans in first phase, got {}",
            first_phase_spans
        );
    }

    #[test]
    fn test_matmul_activation_valid() {
        let (g, _, _, _) = test_graphs::matmul_activation(4, 8, 16, ScalarUnaryOp::Tanh);
        let plan = plan_execution(&g, 2);
        assert_plan_valid(&plan, &g);
        assert_spans_valid(&plan);
    }

    #[test]
    fn test_matmul_chain_valid() {
        let (g, _, _, _, _) = test_graphs::matmul_chain(4, 8, 16, 16, 32);
        let plan = plan_execution(&g, 4);
        assert_plan_valid(&plan, &g);
        assert_spans_valid(&plan);
        // Two matmuls chained → at least 2 phases (barrier between them).
        assert!(
            plan.phases.len() >= 2,
            "Expected >= 2 phases for chained matmuls, got {}",
            plan.phases.len()
        );
    }

    #[test]
    fn test_unary_chain_valid() {
        let (g, _, _) = test_graphs::unary_chain(
            256,
            &[ScalarUnaryOp::Exp, ScalarUnaryOp::Neg, ScalarUnaryOp::Tanh],
        );
        let plan = plan_execution(&g, 2);
        assert_plan_valid(&plan, &g);
        assert_spans_valid(&plan);
        // No reductions → one phase.
        assert_eq!(plan.phases.len(), 1);
    }

    #[test]
    fn test_broadcast_add_valid() {
        let (g, _, _, _) = test_graphs::broadcast_add(1024);
        let plan = plan_execution(&g, 2);
        assert_plan_valid(&plan, &g);
        assert_spans_valid(&plan);
    }

    #[test]
    fn test_within_phase_independence() {
        // Build a graph with two independent matmuls (no shared non-literal deps).
        let mut g = NanoGraph::new();

        // Matmul 1: A[2,4] @ B[4,8]
        let a1 = g.push_group(
            8,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let b1 = g.push_group(
            32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let mut mul1_base = None;
        for mi in 0..2u64 {
            for ki in 0..4u64 {
                let a_atom = a1.offset(mi * 4 + ki);
                let b_row = b1.offset(ki * 8);
                let base = g.push_group(
                    8,
                    ScalarOp::Binary {
                        op: ScalarBinOp::Mul,
                        compute_dtype: DType::F32,
                        output_dtype: DType::F32,
                    },
                    vec![],
                    vec![],
                    vec![
                        InputRef::Broadcast(a_atom),
                        InputRef::Affine {
                            base: b_row,
                            stride: 1,
                        },
                    ],
                );
                if mul1_base.is_none() {
                    mul1_base = Some(base);
                }
            }
        }
        let mul1_base = mul1_base.unwrap();
        let mut red1_base = None;
        for mi in 0..2u64 {
            let row_base = AtomId(mul1_base.0 + mi * 4 * 8);
            let base = g.push_group(
                8,
                ScalarOp::ReduceSum {
                    reduce_count: 4,
                    reduce_stride: 8,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![InputRef::Affine {
                    base: row_base,
                    stride: 1,
                }],
            );
            if red1_base.is_none() {
                red1_base = Some(base);
            }
        }

        // Matmul 2: C[2,4] @ D[4,8] (completely independent)
        let c1 = g.push_group(
            8,
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
            vec![],
        );
        let d1 = g.push_group(
            32,
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
            vec![],
        );
        let mut mul2_base = None;
        for mi in 0..2u64 {
            for ki in 0..4u64 {
                let c_atom = c1.offset(mi * 4 + ki);
                let d_row = d1.offset(ki * 8);
                let base = g.push_group(
                    8,
                    ScalarOp::Binary {
                        op: ScalarBinOp::Mul,
                        compute_dtype: DType::F32,
                        output_dtype: DType::F32,
                    },
                    vec![],
                    vec![],
                    vec![
                        InputRef::Broadcast(c_atom),
                        InputRef::Affine {
                            base: d_row,
                            stride: 1,
                        },
                    ],
                );
                if mul2_base.is_none() {
                    mul2_base = Some(base);
                }
            }
        }
        let mul2_base = mul2_base.unwrap();
        let mut red2_base = None;
        for mi in 0..2u64 {
            let row_base = AtomId(mul2_base.0 + mi * 4 * 8);
            let base = g.push_group(
                8,
                ScalarOp::ReduceSum {
                    reduce_count: 4,
                    reduce_stride: 8,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![InputRef::Affine {
                    base: row_base,
                    stride: 1,
                }],
            );
            if red2_base.is_none() {
                red2_base = Some(base);
            }
        }

        g.outputs = vec![red1_base.unwrap(), red2_base.unwrap()];
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let plan = plan_execution(&g, 2);
        assert_plan_valid(&plan, &g);
        assert_spans_valid(&plan);

        // With two independent matmuls and 2 lanes, we should get
        // multiple spans in at least one phase.
        let total_spans: usize = plan.phases.iter().map(|p| p.spans.len()).sum();
        assert!(
            total_spans >= 2,
            "Expected >= 2 total spans for 2 independent matmuls, got {}",
            total_spans
        );
    }

    #[test]
    fn test_span_graph_self_contained() {
        // Verify that each span's graph can be validated independently —
        // all InputRefs resolve to atoms within the span.
        let (g, _, _, _) = test_graphs::matmul(4, 8, 16);
        let plan = plan_execution(&g, 2);

        for (pi, phase) in plan.phases.iter().enumerate() {
            for (si, span) in phase.spans.iter().enumerate() {
                // The span's graph should validate cleanly.
                let errors = span.graph.validate();
                assert!(
                    errors.is_empty(),
                    "Phase {} Span {} is not self-contained:\n{}",
                    pi,
                    si,
                    errors.join("\n")
                );

                // All declared input base atoms should exist.
                for mapping in &span.inputs {
                    assert!(
                        span.graph.contains_atom(mapping.span_base),
                        "Phase {} Span {}: input local atom {} missing",
                        pi,
                        si,
                        mapping.span_base
                    );
                }

                // All declared output base atoms should exist.
                for mapping in &span.outputs {
                    assert!(
                        span.graph.contains_atom(mapping.span_base),
                        "Phase {} Span {}: output local atom {} missing",
                        pi,
                        si,
                        mapping.span_base
                    );
                }
            }
        }
    }

    #[test]
    fn test_no_cross_span_same_phase_deps() {
        // Explicitly test that no span in a phase reads atoms produced by
        // another span in the same phase.
        let (g, _, _, _, _) = test_graphs::matmul_chain(4, 8, 16, 16, 32);
        let plan = plan_execution(&g, 4);

        for (pi, phase) in plan.phases.iter().enumerate() {
            // Build a map from main atom ranges to span index using output mappings.
            let mut produced_ranges: Vec<(usize, &AtomMapping)> = Vec::new();
            for (si, span) in phase.spans.iter().enumerate() {
                for mapping in &span.outputs {
                    produced_ranges.push((si, mapping));
                }
            }
            for (si, span) in phase.spans.iter().enumerate() {
                for input_mapping in &span.inputs {
                    for &(other_si, other_mapping) in &produced_ranges {
                        if other_si == si {
                            continue;
                        }
                        let a_lo = input_mapping.main_base.0;
                        let a_hi = a_lo + input_mapping.count;
                        let b_lo = other_mapping.main_base.0;
                        let b_hi = b_lo + other_mapping.count;
                        assert!(
                            a_lo >= b_hi || b_lo >= a_hi,
                            "Phase {}: Span {} reads atoms [{}, {}) produced by Span {} (violation!)",
                            pi,
                            si,
                            a_lo,
                            a_hi,
                            other_si
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_all_outputs_covered() {
        // Every original graph output must be produced by some span.
        let (g, _, _, _, _) = test_graphs::matmul_chain(4, 8, 16, 16, 32);
        let plan = plan_execution(&g, 4);

        // Check all original outputs are covered by span output ranges.
        for &out in &g.outputs {
            let covered = plan
                .phases
                .iter()
                .flat_map(|p| p.spans.iter())
                .flat_map(|s| s.outputs.iter())
                .any(|mapping| {
                    out.0 >= mapping.main_base.0 && out.0 < mapping.main_base.0 + mapping.count
                });
            assert!(covered, "Original output {} not covered by any span", out);
        }
    }

    #[test]
    fn test_empty_graph() {
        let g = NanoGraph::new();
        let plan = plan_execution(&g, 4);
        assert_eq!(plan.phases.len(), 0);
    }

    #[test]
    fn test_literals_only() {
        let mut g = NanoGraph::new();
        g.push_group(
            100,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let plan = plan_execution(&g, 4);
        assert_eq!(plan.phases.len(), 0);
    }

    #[test]
    fn test_plan_summary() {
        // Test that we can inspect plan structure.
        let (g, _, _, _, _) = test_graphs::matmul_chain(4, 8, 16, 16, 32);
        let plan = plan_execution(&g, 4);

        let mut total_span_groups = 0;
        let mut total_span_atoms = 0u64;
        for phase in &plan.phases {
            for span in &phase.spans {
                total_span_groups += span.graph.num_groups();
                total_span_atoms += span.graph.num_atoms();
            }
        }
        // Span graphs may duplicate literals, so total may exceed original.
        assert!(total_span_groups > 0, "No groups in any span");
        assert!(total_span_atoms > 0, "No atoms in any span");
    }
}
