#![allow(clippy::all, dead_code, unreachable_patterns, unused_variables, unused_imports)]
//! Span-based partitioner: emits self-contained NanoGraphs for parallel execution.
//!
//! Instead of tracking sub-ranges of the original graph, this partitioner emits
//! N separate NanoGraphs (one per span). Each span is self-contained: it has its
//! own groups, declares what atoms it reads (inputs) and produces (outputs).
//! Small shared computations (like index arithmetic) get duplicated into each
//! span rather than synchronized.
//!
//! ## Algorithm
//!
//! 1. **Build group dependency DAG** — same approach as v2c.
//! 2. **Classify groups** — Literal/Shared, Row(root), or AllRows.
//! 3. **Assign phases** — barrier when cross-lane dependencies are detected.
//! 4. **Assign groups to (phase, lane) spans** — rows go to their lane,
//!    AllRows groups get split across lanes.
//! 5. **Extract self-contained NanoGraphs** — for each span:
//!    - Collect the span's groups (full or sub-range).
//!    - Walk inputs to find external dependencies (atoms from other spans).
//!    - Create a new NanoGraph with fresh AtomIds and remapped InputRefs.
//!    - Declare inputs (external reads) and outputs (atoms consumed by later spans).
//!    - Duplicate Literal groups read by the span (cheap, avoids synchronization).

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

use crate::nano_graph::{AtomGroup, AtomId, InputRef, NanoGraph, ScalarBinOp, ScalarOp};

/// Literal groups with fewer atoms than this are duplicated into spans.
/// Larger literals (weight matrices) become external inputs instead.
const LITERAL_INLINE_THRESHOLD: u64 = 1024;

// ─── Public types ────────────────────────────────────────────────────────────

/// A self-contained execution span with its own NanoGraph.
pub struct Span {
    /// Self-contained NanoGraph for this span's computation.
    pub graph: NanoGraph,
    /// External atoms this span reads: (main graph AtomId, span-local AtomId).
    /// These must be provided by earlier phases via the shared values buffer.
    pub inputs: Vec<(AtomId, AtomId)>,
    /// Atoms this span produces that other spans need:
    /// (span-local AtomId, main graph AtomId).
    pub outputs: Vec<(AtomId, AtomId)>,
    /// Mapping from main-graph AtomId to span-local AtomId for duplicated
    /// Literal groups. Used by the runtime to inject weight/input values
    /// into the span's Literal placeholders.
    pub literal_map: Vec<(AtomId, AtomId)>,
}

/// One phase of execution (work between two consecutive barriers).
pub struct Phase {
    /// One span per lane. Empty spans have zero groups and no I/O.
    pub spans: Vec<Span>,
}

/// The full execution plan with self-contained span NanoGraphs.
pub struct SpanPlan {
    pub num_lanes: usize,
    pub phases: Vec<Phase>,
}

/// A unit of work: a group (or sub-range of a group) assigned to a span.
#[derive(Debug, Clone)]
struct SpanWork {
    group_idx: usize,
    atom_offset: u64,
    atom_count: u64,
}

// ─── Entry point ─────────────────────────────────────────────────────────────

/// Partition a NanoGraph into self-contained spans for parallel execution.
pub fn plan_spans(graph: &NanoGraph, num_lanes: usize) -> SpanPlan {
    let groups = graph.groups();
    let n = groups.len();
    let num_lanes = num_lanes.max(1);

    if n == 0 {
        return SpanPlan {
            num_lanes,
            phases: vec![Phase {
                spans: (0..num_lanes)
                    .map(|_| Span {
                        graph: NanoGraph::new(),
                        inputs: vec![],
                        outputs: vec![],
                        literal_map: vec![],
                    })
                    .collect(),
            }],
        };
    }

    // Step 1: Build dependency DAG.
    let is_literal: Vec<bool> = groups
        .iter()
        .map(|g| matches!(g.op, ScalarOp::Literal(_)) && g.inputs.is_empty())
        .collect();

    let (producers, consumers) = build_group_deps(groups);
    let topo_order = topological_sort(n, &producers);

    // Step 2: Row family classification.
    let row_families = identify_row_families(groups, &producers, &consumers, &is_literal);

    // Step 3: Phase assignment (barrier detection).
    let (group_phase, num_phases) = compute_phase_assignment(
        groups,
        &topo_order,
        &producers,
        &consumers,
        &is_literal,
        &row_families,
    );

    // Step 4: Assign groups to (phase, lane) spans.
    let root_to_lane = compute_root_lane_map(groups, num_lanes, &is_literal, &row_families, &producers);
    let span_assignments = assign_spans(
        groups,
        num_lanes,
        num_phases,
        &group_phase,
        &is_literal,
        &row_families,
        &root_to_lane,
    );

    // Step 5: Determine which atoms each span needs from other spans
    // (cross-span data flow).
    let span_outputs = compute_span_outputs(
        graph,
        groups,
        &is_literal,
        &span_assignments,
        &producers,
        &consumers,
        &group_phase,
        num_lanes,
        num_phases,
    );

    // Step 6: Extract self-contained NanoGraphs for each span.
    let phases = extract_span_graphs(
        graph,
        groups,
        num_lanes,
        num_phases,
        &is_literal,
        &span_assignments,
        &span_outputs,
        &group_phase,
    );

    SpanPlan { num_lanes, phases }
}

// ─── Group dependency graph ──────────────────────────────────────────────────

fn build_group_deps(groups: &[AtomGroup]) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let n = groups.len();
    let mut producers: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (gi, group) in groups.iter().enumerate() {
        let mut prod_set = BTreeSet::new();

        for input in &group.inputs {
            for pi in resolve_producer_groups(input, group.count, groups) {
                if pi != gi {
                    prod_set.insert(pi);
                }
            }
        }

        // ReduceSum/ReduceMax strided access extends the read range.
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
                    for pi in resolve_producer_groups_with_reduce(
                        input,
                        group.count,
                        *reduce_count,
                        *reduce_stride,
                        groups,
                    ) {
                        if pi != gi {
                            prod_set.insert(pi);
                        }
                    }
                }
            }
            _ => {}
        }

        // IndirectLoad table_base dependency.
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

fn resolve_producer_groups_with_reduce(
    input: &InputRef,
    count: u64,
    reduce_count: u64,
    reduce_stride: i64,
    groups: &[AtomGroup],
) -> Vec<usize> {
    if count == 0 {
        return vec![];
    }
    let min_reduce_ext = 0i64.min(reduce_stride * (reduce_count as i64 - 1));
    let max_reduce_ext = 0i64.max(reduce_stride * (reduce_count as i64 - 1));

    match input {
        InputRef::Affine { base, stride } => {
            let first_base = base.0 as i64;
            let last_base = base.0 as i64 + *stride as i64 * (count as i64 - 1);
            let lo = first_base.min(last_base) + min_reduce_ext;
            let hi = first_base.max(last_base) + max_reduce_ext;
            find_groups_in_range(groups, lo as u64, hi as u64)
        }
        InputRef::StridedBroadcast {
            base,
            stride,
            repeat,
        } => {
            let first_read = base.0 as i64;
            let last_block = ((count - 1) / repeat) as i64;
            let last_read = base.0 as i64 + stride * last_block;
            let lo = first_read.min(last_read) + min_reduce_ext;
            let hi = first_read.max(last_read) + max_reduce_ext;
            find_groups_in_range(groups, lo as u64, hi as u64)
        }
        InputRef::Broadcast(atom_id) => {
            let base = atom_id.0 as i64;
            let lo = base + min_reduce_ext;
            let hi = base + max_reduce_ext;
            find_groups_in_range(groups, lo as u64, hi as u64)
        }
        InputRef::SymAffine { base, stride_i, .. } => {
            let first_base = base.0 as i64;
            let last_base = base.0 as i64 + *stride_i as i64 * (count as i64 - 1);
            let lo = first_base.min(last_base) + min_reduce_ext;
            let hi = first_base.max(last_base) + max_reduce_ext;
            find_groups_in_range(groups, lo as u64, hi as u64)
        }
        _ => resolve_producer_groups(input, count, groups),
    }
}

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

// ─── Row family identification ───────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum RowFamily {
    Row(usize),
    AllRows,
    Shared,
}

fn identify_row_families(
    groups: &[AtomGroup],
    producers: &[Vec<usize>],
    consumers: &[Vec<usize>],
    is_literal: &[bool],
) -> Vec<RowFamily> {
    let n = groups.len();
    let mut families = vec![RowFamily::AllRows; n];

    for gi in 0..n {
        if is_literal[gi] {
            families[gi] = RowFamily::Shared;
        }
    }

    // Classify root compute groups by literal producer signature.
    let mut sig_to_roots: HashMap<Vec<usize>, Vec<usize>> = HashMap::new();
    for gi in 0..n {
        if is_literal[gi] {
            continue;
        }
        let all_producers_literal = producers[gi].iter().all(|&pi| is_literal[pi]);
        if all_producers_literal {
            let mut lit_sig: Vec<usize> = producers[gi]
                .iter()
                .filter(|&&pi| is_literal[pi])
                .copied()
                .collect();
            lit_sig.sort();
            lit_sig.dedup();
            sig_to_roots.entry(lit_sig).or_default().push(gi);
        }
    }

    for (_, roots) in &sig_to_roots {
        if roots.len() >= 2 {
            for &gi in roots {
                families[gi] = RowFamily::Row(gi);
            }
        }
    }

    // Forward-propagate in topological order.
    let topo_order = topological_sort(n, producers);

    for &gi in &topo_order {
        if is_literal[gi] || producers[gi].iter().all(|&pi| is_literal[pi]) {
            continue;
        }

        let mut producer_families: HashSet<RowFamily> = HashSet::new();
        for &pi in &producers[gi] {
            if !is_literal[pi] {
                producer_families.insert(families[pi]);
            }
        }
        producer_families.remove(&RowFamily::Shared);

        if producer_families.is_empty() {
            continue;
        }

        if producer_families.len() == 1 {
            let single_family = *producer_families.iter().next().unwrap();
            match single_family {
                RowFamily::Row(_) => {
                    families[gi] = single_family;
                }
                RowFamily::AllRows => {
                    // Check if this is one of multiple peer consumers splitting
                    // an AllRows producer into independent rows.
                    let allrows_producers: Vec<usize> = producers[gi]
                        .iter()
                        .filter(|&&pi| !is_literal[pi] && families[pi] == RowFamily::AllRows)
                        .copied()
                        .collect();

                    let mut is_row_of_allrows = false;
                    for &api in &allrows_producers {
                        let siblings: Vec<usize> = consumers[api]
                            .iter()
                            .filter(|&&ci| !is_literal[ci])
                            .copied()
                            .collect();
                        if siblings.len() >= 2 {
                            let my_op = std::mem::discriminant(&groups[gi].op);
                            let my_count = groups[gi].count;
                            let similar_count = siblings
                                .iter()
                                .filter(|&&ci| {
                                    std::mem::discriminant(&groups[ci].op) == my_op
                                        && groups[ci].count == my_count
                                })
                                .count();
                            if similar_count >= 2 {
                                is_row_of_allrows = true;
                            }
                        }
                    }

                    if is_row_of_allrows {
                        families[gi] = RowFamily::Row(gi);
                    }
                    // else: stays AllRows
                }
                RowFamily::Shared => unreachable!(),
            }
        }
        // else: multiple families -> stays AllRows
    }

    families
}

// ─── Lane-locality check ─────────────────────────────────────────────────────

fn is_allrows_chain_lane_local(
    gi: usize,
    groups: &[AtomGroup],
    row_families: &[RowFamily],
    is_literal: &[bool],
    group_phase: &[usize],
    max_prod_phase: usize,
) -> bool {
    let consumer = &groups[gi];
    let consumer_count = consumer.count;
    if consumer_count == 0 {
        return true;
    }

    for input in &consumer.inputs {
        let prod_groups = resolve_producer_groups(input, consumer_count, groups);

        let same_phase_allrows: Vec<usize> = prod_groups
            .iter()
            .copied()
            .filter(|&pi| {
                !is_literal[pi]
                    && row_families[pi] == RowFamily::AllRows
                    && group_phase[pi] == max_prod_phase
            })
            .collect();

        if same_phase_allrows.is_empty() {
            continue;
        }

        match input {
            InputRef::Affine { stride, .. } => {
                if *stride != 1 {
                    return false;
                }
                for &pi in &same_phase_allrows {
                    if groups[pi].count != consumer_count {
                        return false;
                    }
                }
            }
            InputRef::Broadcast(_) => {
                return false;
            }
            InputRef::StridedBroadcast { repeat, .. } => {
                for &pi in &same_phase_allrows {
                    let expected = (consumer_count + repeat - 1) / repeat;
                    if groups[pi].count != expected {
                        return false;
                    }
                }
            }
            InputRef::Modular { .. } | InputRef::Explicit(_) | InputRef::SymAffine { .. } => {
                return false;
            }
        }
    }

    // Also check ReduceSum/ReduceMax strided access.
    match &consumer.op {
        ScalarOp::ReduceSum {
            reduce_count,
            reduce_stride,
            ..
        }
        | ScalarOp::ReduceMax {
            reduce_count,
            reduce_stride,
            ..
        } => {
            if *reduce_count > 1 && *reduce_stride != 0 {
                let has_same_phase_allrows_input = consumer.inputs.iter().any(|inp| {
                    let prods = resolve_producer_groups_with_reduce(
                        inp,
                        consumer_count,
                        *reduce_count,
                        *reduce_stride,
                        groups,
                    );
                    prods.iter().any(|&pi| {
                        !is_literal[pi]
                            && row_families[pi] == RowFamily::AllRows
                            && group_phase[pi] == max_prod_phase
                    })
                });
                if has_same_phase_allrows_input {
                    return false;
                }
            }
        }
        _ => {}
    }

    true
}

// ─── Phase assignment ────────────────────────────────────────────────────────

fn compute_phase_assignment(
    groups: &[AtomGroup],
    topo_order: &[usize],
    producers: &[Vec<usize>],
    consumers: &[Vec<usize>],
    is_literal: &[bool],
    row_families: &[RowFamily],
) -> (Vec<usize>, usize) {
    let n = groups.len();
    let mut group_phase = vec![0usize; n];

    for &gi in topo_order {
        if is_literal[gi] {
            group_phase[gi] = 0;
            continue;
        }

        let mut max_prod_phase = 0usize;
        for &pi in &producers[gi] {
            if is_literal[pi] {
                continue;
            }
            max_prod_phase = max_prod_phase.max(group_phase[pi]);
        }

        let mut same_phase_families: HashSet<RowFamily> = HashSet::new();
        for &pi in &producers[gi] {
            if is_literal[pi] {
                continue;
            }
            if group_phase[pi] == max_prod_phase {
                same_phase_families.insert(row_families[pi]);
            }
        }

        let has_all_rows = same_phase_families.remove(&RowFamily::AllRows);

        let distinct_row_families = same_phase_families
            .iter()
            .filter(|f| matches!(f, RowFamily::Row(_)))
            .count();

        let allrows_to_allrows_needs_barrier = if has_all_rows
            && same_phase_families.is_empty()
            && row_families[gi] == RowFamily::AllRows
        {
            !is_allrows_chain_lane_local(
                gi,
                groups,
                row_families,
                is_literal,
                &group_phase,
                max_prod_phase,
            )
        } else if has_all_rows && same_phase_families.is_empty() {
            // Consumer is not AllRows reading from AllRows at same phase
            true
        } else {
            false
        };

        let allrows_consumer_reads_row =
            row_families[gi] == RowFamily::AllRows && distinct_row_families >= 1;

        let needs_barrier = distinct_row_families > 1
            || (has_all_rows && distinct_row_families >= 1)
            || allrows_to_allrows_needs_barrier
            || allrows_consumer_reads_row;

        if needs_barrier {
            group_phase[gi] = max_prod_phase + 1;
        } else {
            group_phase[gi] = max_prod_phase;
        }
    }

    let num_phases = group_phase.iter().copied().max().map(|m| m + 1).unwrap_or(1);
    (group_phase, num_phases)
}

// ─── Root-to-lane mapping ────────────────────────────────────────────────────

fn compute_root_lane_map(
    groups: &[AtomGroup],
    num_lanes: usize,
    is_literal: &[bool],
    row_families: &[RowFamily],
    producers: &[Vec<usize>],
) -> HashMap<usize, usize> {
    let n = groups.len();
    let mut root_to_lane: HashMap<usize, usize> = HashMap::new();

    let mut root_members: HashMap<usize, Vec<usize>> = HashMap::new();
    for gi in 0..n {
        if let RowFamily::Row(root) = row_families[gi] {
            root_members.entry(root).or_default().push(gi);
        }
    }

    let mut sig_to_roots: HashMap<Vec<usize>, Vec<usize>> = HashMap::new();
    for (&root, members) in &root_members {
        let mut lit_sig: BTreeSet<usize> = BTreeSet::new();
        for &gi in members {
            for &pi in &producers[gi] {
                if is_literal[pi] {
                    lit_sig.insert(pi);
                }
            }
        }
        let sig: Vec<usize> = lit_sig.into_iter().collect();
        sig_to_roots.entry(sig).or_default().push(root);
    }

    for (_, roots) in &mut sig_to_roots {
        roots.sort();
        for (i, &root) in roots.iter().enumerate() {
            root_to_lane.insert(root, i % num_lanes);
        }
    }

    root_to_lane
}

// ─── Span assignment ─────────────────────────────────────────────────────────

/// For each (phase, lane), collect the SpanWork items.
/// Returns: span_assignments[phase][lane] = Vec<SpanWork>
fn assign_spans(
    groups: &[AtomGroup],
    num_lanes: usize,
    num_phases: usize,
    group_phase: &[usize],
    is_literal: &[bool],
    row_families: &[RowFamily],
    root_to_lane: &HashMap<usize, usize>,
) -> Vec<Vec<Vec<SpanWork>>> {
    let n = groups.len();
    let mut assignments: Vec<Vec<Vec<SpanWork>>> =
        vec![vec![Vec::new(); num_lanes]; num_phases];

    for phase_idx in 0..num_phases {
        let phase_groups: Vec<usize> = (0..n)
            .filter(|&gi| !is_literal[gi] && group_phase[gi] == phase_idx)
            .collect();

        for &gi in &phase_groups {
            match row_families[gi] {
                RowFamily::Shared => {}
                RowFamily::Row(root) => {
                    let lane = root_to_lane.get(&root).copied().unwrap_or(0);
                    assignments[phase_idx][lane].push(SpanWork {
                        group_idx: gi,
                        atom_offset: 0,
                        atom_count: groups[gi].count,
                    });
                }
                RowFamily::AllRows => {
                    let count = groups[gi].count;
                    let is_reduce = groups[gi].op.is_reduce();

                    if num_lanes == 1 || count == 0 || is_reduce {
                        // ReduceSum/ReduceMax cannot be split across lanes because
                        // the reduce_stride in the op assumes a specific layout of
                        // input atoms. Splitting changes the local atom layout and
                        // breaks the stride relationship.
                        //
                        // Assign the whole group to lane 0. (A more sophisticated
                        // strategy could round-robin across lanes for balance.)
                        assignments[phase_idx][0].push(SpanWork {
                            group_idx: gi,
                            atom_offset: 0,
                            atom_count: count,
                        });
                    } else {
                        let chunk = (count + num_lanes as u64 - 1) / num_lanes as u64;
                        let mut offset = 0u64;
                        for lane in 0..num_lanes {
                            if offset >= count {
                                break;
                            }
                            let remaining = count - offset;
                            let this_chunk = if lane < num_lanes - 1 {
                                chunk.min(remaining)
                            } else {
                                remaining
                            };
                            if this_chunk > 0 {
                                assignments[phase_idx][lane].push(SpanWork {
                                    group_idx: gi,
                                    atom_offset: offset,
                                    atom_count: this_chunk,
                                });
                            }
                            offset += this_chunk;
                        }
                    }
                }
            }
        }

        // Sort each lane's work by (group_idx, offset) for determinism.
        for lane in 0..num_lanes {
            assignments[phase_idx][lane]
                .sort_by_key(|w| (w.group_idx, w.atom_offset));
        }
    }

    assignments
}

// ─── Output computation ──────────────────────────────────────────────────────

/// Determine which main-graph atoms each span writes to the shared buffer.
///
/// Every non-literal atom produced by a span is an output. This is the
/// conservative strategy: the codegen can later prune outputs that are
/// only consumed within the same span, but the partitioner marks everything
/// so the inter-span protocol is complete.
///
/// Returns: span_outputs[phase][lane] = set of main-graph AtomIds this span produces.
fn compute_span_outputs(
    source_graph: &NanoGraph,
    groups: &[AtomGroup],
    is_literal: &[bool],
    span_assignments: &[Vec<Vec<SpanWork>>],
    producers: &[Vec<usize>],
    consumers: &[Vec<usize>],
    group_phase: &[usize],
    num_lanes: usize,
    num_phases: usize,
) -> Vec<Vec<HashSet<u64>>> {
    let mut span_outputs: Vec<Vec<HashSet<u64>>> =
        vec![vec![HashSet::new(); num_lanes]; num_phases];

    // Mark every atom produced by each span as an output.
    for (phase_idx, phase_lanes) in span_assignments.iter().enumerate() {
        for (lane_idx, works) in phase_lanes.iter().enumerate() {
            for work in works {
                let group = &groups[work.group_idx];
                let base = group.base_id.0 + work.atom_offset;
                for i in 0..work.atom_count {
                    span_outputs[phase_idx][lane_idx].insert(base + i);
                }
            }
        }
    }

    span_outputs
}


// ─── NanoGraph extraction ────────────────────────────────────────────────────

/// Extract self-contained NanoGraphs for each span.
///
/// For each (phase, lane), we build a fresh NanoGraph containing:
/// 1. Input placeholder groups for external atoms this span reads.
/// 2. Duplicated Literal groups for constants this span needs.
/// 3. The compute groups assigned to this span (with remapped InputRefs).
///
/// The key insight: every atom the span reads must either be:
/// - Produced by a group within the span, or
/// - Declared as an external input.
///
/// Literals are always duplicated (cheap, avoids synchronization).
fn extract_span_graphs(
    source_graph: &NanoGraph,
    groups: &[AtomGroup],
    num_lanes: usize,
    num_phases: usize,
    is_literal: &[bool],
    span_assignments: &[Vec<Vec<SpanWork>>],
    span_outputs: &[Vec<HashSet<u64>>],
    group_phase: &[usize],
) -> Vec<Phase> {
    let mut phases = Vec::with_capacity(num_phases);

    for phase_idx in 0..num_phases {
        let mut spans = Vec::with_capacity(num_lanes);

        for lane_idx in 0..num_lanes {
            let works = &span_assignments[phase_idx][lane_idx];
            let outputs_needed = &span_outputs[phase_idx][lane_idx];

            let span = extract_single_span(
                source_graph,
                groups,
                is_literal,
                works,
                outputs_needed,
                group_phase,
                phase_idx,
            );
            spans.push(span);
        }

        phases.push(Phase { spans });
    }

    phases
}

/// Extract a single span's self-contained NanoGraph.
fn extract_single_span(
    source_graph: &NanoGraph,
    groups: &[AtomGroup],
    is_literal: &[bool],
    works: &[SpanWork],
    outputs_needed: &HashSet<u64>,
    group_phase: &[usize],
    phase_idx: usize,
) -> Span {
    if works.is_empty() {
        return Span {
            graph: NanoGraph::new(),
            inputs: vec![],
            outputs: vec![],
            literal_map: vec![],
        };
    }

    let mut span_graph = NanoGraph::new();

    // Copy sym_dim metadata from source graph.
    for (name, &sd) in &source_graph.sym_dim_names {
        let new_sd = span_graph.sym_dim(name);
        if let Some(&bound) = source_graph.sym_dim_bounds.get(&sd) {
            span_graph.sym_dim_bounds.insert(new_sd, bound);
        }
    }

    // Mapping: main-graph AtomId -> span-local AtomId.
    // This is built incrementally as we add groups to the span.
    let mut atom_map: HashMap<u64, AtomId> = HashMap::new();

    // Track which main-graph groups we're including (for fast lookup).
    // For split groups, we need to know the sub-range.
    let mut included_groups: HashMap<usize, (u64, u64)> = HashMap::new(); // gi -> (offset, count)
    for work in works {
        included_groups.insert(work.group_idx, (work.atom_offset, work.atom_count));
    }

    // Collect all atoms this span produces.
    let mut produced_atoms: HashSet<u64> = HashSet::new();
    for work in works {
        let group = &groups[work.group_idx];
        let base = group.base_id.0 + work.atom_offset;
        for i in 0..work.atom_count {
            produced_atoms.insert(base + i);
        }
    }

    // Phase 1: Identify ALL external atoms this span needs.
    // Walk each work item's inputs and find atoms not produced by this span.
    let mut external_atoms: BTreeSet<u64> = BTreeSet::new(); // sorted for determinism
    let mut needed_literals: BTreeSet<usize> = BTreeSet::new(); // literal group indices

    for work in works {
        let group = &groups[work.group_idx];
        let needed = collect_all_needed_atoms(groups, work, is_literal);
        for atom_raw in needed {
            if produced_atoms.contains(&atom_raw) {
                continue; // Produced within this span.
            }
            // Check if it's from a literal group.
            if let Some(gi) = find_group_idx(groups, AtomId(atom_raw)) {
                if is_literal[gi] {
                    needed_literals.insert(gi);
                    continue;
                }
            }
            external_atoms.insert(atom_raw);
        }
    }

    // Phase 2: Add duplicated Literal groups (small ones only).
    // Large literals become external inputs to avoid memory explosion.
    for &lit_gi in &needed_literals {
        let lit_group = &groups[lit_gi];
        if lit_group.count < LITERAL_INLINE_THRESHOLD {
            // Small literal: duplicate into span.
            let local_base = span_graph.push_group(
                lit_group.count,
                lit_group.op.clone(),
                lit_group.sym_dims.clone(),
                lit_group.reduce_dims.clone(),
                vec![], // Literals have no inputs
            );
            for i in 0..lit_group.count {
                atom_map.insert(lit_group.base_id.0 + i, AtomId(local_base.0 + i));
            }
        } else {
            // Large literal (weight matrix): treat as external input.
            for i in 0..lit_group.count {
                external_atoms.insert(lit_group.base_id.0 + i);
            }
        }
    }

    // Phase 3: Add input placeholder groups for external atoms.
    // Each external atom becomes a single-atom Literal(0.0) placeholder in the span
    // graph. The runtime will overwrite these with actual values.
    let mut span_inputs: Vec<(AtomId, AtomId)> = Vec::new();
    for &atom_raw in &external_atoms {
        // Determine the dtype from the source group's output_dtype.
        let local_id = span_graph.push_atom(
            ScalarOp::Literal(crate::numeric_scalar::NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );
        atom_map.insert(atom_raw, local_id);
        span_inputs.push((AtomId(atom_raw), local_id));
    }

    // Phase 4: Add the span's compute groups with remapped InputRefs.
    // Process in group_idx order (which is topological order in the source graph).
    let mut sorted_works: Vec<&SpanWork> = works.iter().collect();
    sorted_works.sort_by_key(|w| (w.group_idx, w.atom_offset));

    for work in &sorted_works {
        let group = &groups[work.group_idx];

        if work.atom_offset == 0 && work.atom_count == group.count {
            // Full group — remap inputs and add.
            let remapped_inputs = remap_inputs(&group.inputs, group.count, 0, &atom_map);
            let local_base = span_graph.push_group(
                group.count,
                group.op.clone(),
                group.sym_dims.clone(),
                group.reduce_dims.clone(),
                remapped_inputs,
            );
            for i in 0..group.count {
                atom_map.insert(group.base_id.0 + i, AtomId(local_base.0 + i));
            }
        } else {
            // Sub-range of a group — create a new group for the slice.
            let remapped_inputs =
                remap_inputs(&group.inputs, work.atom_count, work.atom_offset, &atom_map);

            // For ReduceSum/ReduceMax, the op parameters stay the same (reduce_count
            // and reduce_stride are per-atom, not per-group).
            let local_base = span_graph.push_group(
                work.atom_count,
                group.op.clone(),
                group.sym_dims.clone(),
                group.reduce_dims.clone(),
                remapped_inputs,
            );
            for i in 0..work.atom_count {
                atom_map.insert(
                    group.base_id.0 + work.atom_offset + i,
                    AtomId(local_base.0 + i),
                );
            }
        }
    }

    // Phase 5: Compute outputs — atoms produced by this span that are needed
    // by other spans.
    let mut span_outputs_vec: Vec<(AtomId, AtomId)> = Vec::new();
    for &atom_raw in outputs_needed {
        if let Some(&local_id) = atom_map.get(&atom_raw) {
            span_outputs_vec.push((local_id, AtomId(atom_raw)));
        }
    }
    span_outputs_vec.sort_by_key(|&(_, main_id)| main_id.0);

    // Phase 6: Build literal_map — mapping from main-graph AtomId to local
    // AtomId for all duplicated Literal atoms.
    let mut literal_map: Vec<(AtomId, AtomId)> = Vec::new();
    for &lit_gi in &needed_literals {
        let lit_group = &groups[lit_gi];
        for i in 0..lit_group.count {
            let main_atom = lit_group.base_id.0 + i;
            if let Some(&local_id) = atom_map.get(&main_atom) {
                literal_map.push((AtomId(main_atom), local_id));
            }
        }
    }
    literal_map.sort_by_key(|&(main_id, _)| main_id.0);

    Span {
        graph: span_graph,
        inputs: span_inputs,
        outputs: span_outputs_vec,
        literal_map,
    }
}

/// Collect ALL atoms needed by a work item (from all inputs, including reduce strides).
fn collect_all_needed_atoms(
    groups: &[AtomGroup],
    work: &SpanWork,
    is_literal: &[bool],
) -> HashSet<u64> {
    let group = &groups[work.group_idx];
    let offset = work.atom_offset;
    let count = work.atom_count;
    let mut needed = HashSet::new();

    for input in &group.inputs {
        match input {
            InputRef::Broadcast(atom_id) => {
                needed.insert(atom_id.0);
            }
            InputRef::Affine { base, stride } => {
                for i in 0..count {
                    let atom = input.resolve(offset + i, 0).0;
                    needed.insert(atom);
                }
            }
            InputRef::StridedBroadcast {
                base,
                stride,
                repeat,
            } => {
                for i in 0..count {
                    let atom = input.resolve(offset + i, 0).0;
                    needed.insert(atom);
                }
            }
            InputRef::Modular {
                base,
                stride,
                modulus,
            } => {
                for i in 0..count {
                    let atom = input.resolve(offset + i, 0).0;
                    needed.insert(atom);
                }
            }
            InputRef::Explicit(ids) => {
                for i in 0..count {
                    let idx = (offset + i) as usize;
                    if idx < ids.len() {
                        needed.insert(ids[idx].0);
                    }
                }
            }
            InputRef::SymAffine { .. } => {
                for i in 0..count {
                    let atom = input.resolve(offset + i, 0).0;
                    needed.insert(atom);
                }
            }
        }
    }

    // ReduceSum/ReduceMax: strided access extends reads.
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
                for i in 0..count {
                    let base_atom = input.resolve(offset + i, 0).0 as i64;
                    for k in 0..(*reduce_count) {
                        let atom = (base_atom + k as i64 * reduce_stride) as u64;
                        needed.insert(atom);
                    }
                }
            }
        }
        _ => {}
    }

    // IndirectLoad: table_base dependency.
    if let ScalarOp::IndirectLoad { table_base, .. } = &group.op {
        // We can't know which atoms will be loaded at runtime, but we need the
        // table group to be available. For now, include the table_base atom.
        // The runtime must ensure the entire table is accessible.
        needed.insert(table_base.0);
    }

    needed
}

/// Remap InputRefs from main-graph AtomIds to span-local AtomIds.
///
/// `count` is the number of atoms in the (possibly sliced) group.
/// `offset` is the starting offset within the original group.
fn remap_inputs(
    inputs: &[InputRef],
    count: u64,
    offset: u64,
    atom_map: &HashMap<u64, AtomId>,
) -> Vec<InputRef> {
    inputs
        .iter()
        .map(|input| remap_single_input(input, count, offset, atom_map))
        .collect()
}

fn remap_single_input(
    input: &InputRef,
    count: u64,
    offset: u64,
    atom_map: &HashMap<u64, AtomId>,
) -> InputRef {
    match input {
        InputRef::Broadcast(atom_id) => {
            let local = atom_map
                .get(&atom_id.0)
                .copied()
                .unwrap_or_else(|| panic!("Unmapped atom {} in Broadcast", atom_id.0));
            InputRef::Broadcast(local)
        }
        InputRef::Affine { base, stride } => {
            // Original: atom i in group reads base + stride * i.
            // Sliced: atom j in sub-group reads base + stride * (offset + j).
            // = (base + stride * offset) + stride * j.
            let new_base_raw =
                base.0.wrapping_add((*stride as i64 * offset as i64) as u64);
            let local_base = atom_map
                .get(&new_base_raw)
                .copied()
                .unwrap_or_else(|| {
                    panic!(
                        "Unmapped atom {} in Affine (base={}, stride={}, offset={})",
                        new_base_raw, base.0, stride, offset
                    )
                });

            // Check if stride=1 and all atoms are contiguous in the local graph.
            // This is the common case for elementwise chains.
            if *stride == 1 && count > 1 {
                // Verify the local atoms are contiguous.
                let second_raw = new_base_raw + 1;
                if let Some(&second_local) = atom_map.get(&second_raw) {
                    if second_local.0 == local_base.0 + 1 {
                        return InputRef::Affine {
                            base: local_base,
                            stride: 1,
                        };
                    }
                }
            }

            // For stride != 1 or non-contiguous local mapping, we need to check
            // if the stride relationship is preserved in the local graph.
            if count <= 1 {
                return InputRef::Affine {
                    base: local_base,
                    stride: *stride,
                };
            }

            // Check if the stride is preserved: local(base + stride) - local(base) == stride
            let next_raw = (new_base_raw as i64 + *stride as i64) as u64;
            if let Some(&next_local) = atom_map.get(&next_raw) {
                let local_stride = next_local.0 as i64 - local_base.0 as i64;
                // Verify consistent stride across a few more samples.
                let stride_ok = if count > 2 {
                    let third_raw = (new_base_raw as i64 + 2 * *stride as i64) as u64;
                    atom_map
                        .get(&third_raw)
                        .map(|&third_local| {
                            (third_local.0 as i64 - next_local.0 as i64) == local_stride
                        })
                        .unwrap_or(false)
                } else {
                    true
                };
                if stride_ok {
                    return InputRef::Affine {
                        base: local_base,
                        stride: local_stride as i32,
                    };
                }
            }

            // Fallback: Explicit mapping.
            let ids: Vec<AtomId> = (0..count)
                .map(|j| {
                    let raw = (new_base_raw as i64 + *stride as i64 * j as i64) as u64;
                    atom_map
                        .get(&raw)
                        .copied()
                        .unwrap_or_else(|| panic!("Unmapped atom {} in Affine explicit fallback", raw))
                })
                .collect();
            InputRef::Explicit(ids)
        }
        InputRef::StridedBroadcast {
            base,
            stride,
            repeat,
        } => {
            // Original: atom i reads base + stride * (i / repeat).
            // Sliced: atom j reads base + stride * ((offset + j) / repeat).
            //
            // If offset is aligned to repeat, this simplifies:
            // = base + stride * (offset/repeat + j/repeat)
            // = (base + stride * offset/repeat) + stride * (j/repeat)
            // which is a StridedBroadcast with adjusted base.

            if offset % repeat == 0 {
                let block_offset = offset / repeat;
                let new_base_raw =
                    (base.0 as i64 + stride * block_offset as i64) as u64;
                let local_base = atom_map.get(&new_base_raw).copied().unwrap_or_else(|| {
                    panic!(
                        "Unmapped atom {} in StridedBroadcast aligned (base={}, stride={}, offset={}, repeat={})",
                        new_base_raw, base.0, stride, offset, repeat
                    )
                });

                // Check if the stride relationship is preserved.
                let num_blocks = (count + repeat - 1) / repeat;
                if num_blocks > 1 {
                    let second_raw = (new_base_raw as i64 + stride) as u64;
                    if let Some(&second_local) = atom_map.get(&second_raw) {
                        let local_stride = second_local.0 as i64 - local_base.0 as i64;
                        return InputRef::StridedBroadcast {
                            base: local_base,
                            stride: local_stride,
                            repeat: *repeat,
                        };
                    }
                }
                // Single block or can't verify stride.
                return InputRef::StridedBroadcast {
                    base: local_base,
                    stride: *stride,
                    repeat: *repeat,
                };
            }

            // Unaligned offset: fall back to Explicit.
            let ids: Vec<AtomId> = (0..count)
                .map(|j| {
                    let raw = input.resolve(offset + j, 0).0;
                    atom_map
                        .get(&raw)
                        .copied()
                        .unwrap_or_else(|| panic!("Unmapped atom {} in StridedBroadcast explicit fallback", raw))
                })
                .collect();
            InputRef::Explicit(ids)
        }
        InputRef::Modular {
            base,
            stride,
            modulus,
        } => {
            // Modular: atom i reads base + stride * ((offset + i) % modulus).
            // If offset is 0, this preserves as Modular with same params.
            // Otherwise, fall back to Explicit.
            if offset == 0 {
                let local_base = atom_map
                    .get(&base.0)
                    .copied()
                    .unwrap_or_else(|| panic!("Unmapped atom {} in Modular", base.0));

                // Check stride preservation.
                if *modulus > 1 {
                    let second_raw = (base.0 as i64 + *stride as i64) as u64;
                    if let Some(&second_local) = atom_map.get(&second_raw) {
                        let local_stride = (second_local.0 as i64 - local_base.0 as i64) as i32;
                        return InputRef::Modular {
                            base: local_base,
                            stride: local_stride,
                            modulus: *modulus,
                        };
                    }
                }

                return InputRef::Modular {
                    base: local_base,
                    stride: *stride,
                    modulus: *modulus,
                };
            }

            // Non-zero offset with Modular: Explicit fallback.
            let ids: Vec<AtomId> = (0..count)
                .map(|j| {
                    let raw = input.resolve(offset + j, 0).0;
                    atom_map
                        .get(&raw)
                        .copied()
                        .unwrap_or_else(|| panic!("Unmapped atom {} in Modular explicit fallback", raw))
                })
                .collect();
            InputRef::Explicit(ids)
        }
        InputRef::Explicit(ids) => {
            let local_ids: Vec<AtomId> = (0..count)
                .map(|j| {
                    let idx = (offset + j) as usize;
                    let raw = ids[idx].0;
                    atom_map
                        .get(&raw)
                        .copied()
                        .unwrap_or_else(|| panic!("Unmapped atom {} in Explicit", raw))
                })
                .collect();
            InputRef::Explicit(local_ids)
        }
        InputRef::SymAffine {
            base,
            stride_i,
            stride_k,
        } => {
            // SymAffine: atom i reads base + stride_i * i + stride_k * k.
            // Sliced: atom j reads base + stride_i * (offset + j) + stride_k * k
            // = (base + stride_i * offset) + stride_i * j + stride_k * k.
            let new_base_raw =
                base.0.wrapping_add((*stride_i as i64 * offset as i64) as u64);
            let local_base = atom_map
                .get(&new_base_raw)
                .copied()
                .unwrap_or_else(|| {
                    panic!("Unmapped atom {} in SymAffine", new_base_raw)
                });

            // Check stride_i preservation.
            if count > 1 {
                let next_raw =
                    (new_base_raw as i64 + *stride_i as i64) as u64;
                if let Some(&next_local) = atom_map.get(&next_raw) {
                    let local_stride_i = (next_local.0 as i64 - local_base.0 as i64) as i32;
                    return InputRef::SymAffine {
                        base: local_base,
                        stride_i: local_stride_i,
                        stride_k: *stride_k,
                    };
                }
            }

            InputRef::SymAffine {
                base: local_base,
                stride_i: *stride_i,
                stride_k: *stride_k,
            }
        }
    }
}

// ─── Diagnostics ─────────────────────────────────────────────────────────────

impl SpanPlan {
    /// Print summary statistics.
    pub fn print_summary(&self) {
        println!(
            "SpanPlan: {} lanes, {} phases",
            self.num_lanes,
            self.phases.len()
        );

        for (pi, phase) in self.phases.iter().enumerate() {
            let lane_groups: Vec<usize> = phase
                .spans
                .iter()
                .map(|s| s.graph.num_groups())
                .collect();
            let lane_atoms: Vec<u64> = phase
                .spans
                .iter()
                .map(|s| s.graph.num_atoms())
                .collect();
            let total_groups: usize = lane_groups.iter().sum();
            let total_atoms: u64 = lane_atoms.iter().sum();
            let total_inputs: usize = phase.spans.iter().map(|s| s.inputs.len()).sum();
            let total_outputs: usize = phase.spans.iter().map(|s| s.outputs.len()).sum();

            println!(
                "  Phase {}: {} groups, {} atoms, {} span inputs, {} span outputs",
                pi, total_groups, total_atoms, total_inputs, total_outputs
            );
            for (li, span) in phase.spans.iter().enumerate() {
                if span.graph.num_groups() > 0 {
                    println!(
                        "    Lane {}: {} groups, {} atoms, {} inputs, {} outputs",
                        li,
                        span.graph.num_groups(),
                        span.graph.num_atoms(),
                        span.inputs.len(),
                        span.outputs.len(),
                    );
                }
            }
        }
    }

    /// Validate all spans: each span's NanoGraph should pass validate(),
    /// and all declared inputs/outputs should reference valid atoms.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();

        for (pi, phase) in self.phases.iter().enumerate() {
            for (li, span) in phase.spans.iter().enumerate() {
                let prefix = format!("Phase {} Lane {}", pi, li);

                // Validate the span's NanoGraph.
                let graph_errors = span.graph.validate();
                for err in graph_errors {
                    errors.push(format!("{}: {}", prefix, err));
                }

                // Validate inputs: each local AtomId should exist in the span graph.
                for &(main_id, local_id) in &span.inputs {
                    if !span.graph.contains_atom(local_id) {
                        errors.push(format!(
                            "{}: input (main={}, local={}) — local atom not in span graph",
                            prefix, main_id, local_id
                        ));
                    }
                }

                // Validate outputs: each local AtomId should exist in the span graph.
                for &(local_id, main_id) in &span.outputs {
                    if !span.graph.contains_atom(local_id) {
                        errors.push(format!(
                            "{}: output (local={}, main={}) — local atom not in span graph",
                            prefix, local_id, main_id
                        ));
                    }
                }
            }
        }

        errors
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::attempts::v13_claude::test_graphs;
    use crate::dtype::DType;
    use crate::nano_graph::{ScalarBinOp, ScalarUnaryOp};
    use crate::numeric_scalar::NumericScalar;

    /// Validate that all spans are self-contained: every InputRef in a span's
    /// NanoGraph resolves to an atom within that NanoGraph.
    fn assert_spans_self_contained(plan: &SpanPlan) {
        let errors = plan.validate();
        assert!(errors.is_empty(), "Span validation errors:\n{}", errors.join("\n"));
    }



    /// Validate independence: within a phase, no span reads atoms produced
    /// by another span in the same phase.
    fn assert_phase_independence(plan: &SpanPlan) {
        for (pi, phase) in plan.phases.iter().enumerate() {
            // Collect outputs of each span in this phase.
            let mut span_output_mains: Vec<HashSet<u64>> = Vec::new();
            for span in &phase.spans {
                let mains: HashSet<u64> = span.outputs.iter().map(|&(_, m)| m.0).collect();
                span_output_mains.push(mains);
            }

            // Check: no span's inputs include another span's outputs from this phase.
            for (li, span) in phase.spans.iter().enumerate() {
                let my_input_mains: HashSet<u64> =
                    span.inputs.iter().map(|&(m, _)| m.0).collect();

                for (other_li, other_outputs) in span_output_mains.iter().enumerate() {
                    if other_li == li {
                        continue;
                    }
                    let overlap: Vec<u64> = my_input_mains
                        .intersection(other_outputs)
                        .copied()
                        .collect();
                    assert!(
                        overlap.is_empty(),
                        "Phase {} Lane {} reads atoms from Lane {} in same phase: {:?}",
                        pi, li, other_li, overlap
                    );
                }
            }
        }
    }

    // ─── Basic tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_empty_graph() {
        let g = NanoGraph::new();
        let plan = plan_spans(&g, 2);
        assert_eq!(plan.num_lanes, 2);
        assert_spans_self_contained(&plan);
    }

    #[test]
    fn test_single_literal() {
        let mut g = NanoGraph::new();
        g.push_group(
            100,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let plan = plan_spans(&g, 2);
        assert_spans_self_contained(&plan);
    }

    #[test]
    fn test_elementwise_add_1_lane() {
        let (g, _, _, _) = test_graphs::elementwise_binary(1024, ScalarBinOp::Add);
        let plan = plan_spans(&g, 1);
        assert_spans_self_contained(&plan);
        assert_phase_independence(&plan);

        // 1 lane: everything in one span.
        assert_eq!(plan.phases.len(), 1);
        assert!(plan.phases[0].spans[0].graph.num_groups() > 0);
    }

    #[test]
    fn test_elementwise_add_2_lanes() {
        let (g, _, _, _) = test_graphs::elementwise_binary(1024, ScalarBinOp::Add);
        let plan = plan_spans(&g, 2);
        assert_spans_self_contained(&plan);
        assert_phase_independence(&plan);

        // Elementwise ops should be split across lanes (AllRows groups).
        // Total compute atoms should match original.
        let total_compute_atoms: u64 = plan
            .phases
            .iter()
            .flat_map(|p| p.spans.iter())
            .map(|s| {
                s.graph
                    .groups()
                    .iter()
                    .filter(|g| !matches!(g.op, ScalarOp::Literal(_)))
                    .map(|g| g.count)
                    .sum::<u64>()
            })
            .sum();
        // Original has 1024 Add atoms.
        assert_eq!(total_compute_atoms, 1024);
    }

    #[test]
    fn test_broadcast_add_2_lanes() {
        let (g, _, _, _) = test_graphs::broadcast_add(1024);
        let plan = plan_spans(&g, 2);
        assert_spans_self_contained(&plan);
        assert_phase_independence(&plan);
    }

    #[test]
    fn test_unary_chain_2_lanes() {
        let (g, _, _) = test_graphs::unary_chain(
            256,
            &[ScalarUnaryOp::Exp, ScalarUnaryOp::Neg, ScalarUnaryOp::Tanh],
        );
        let plan = plan_spans(&g, 2);
        assert_spans_self_contained(&plan);
        assert_phase_independence(&plan);
    }

    // ─── Matmul tests ────────────────────────────────────────────────────────

    #[test]
    fn test_matmul_1_lane() {
        let (g, _, _, _) = test_graphs::matmul(4, 8, 16);
        let plan = plan_spans(&g, 1);
        assert_spans_self_contained(&plan);
        assert_phase_independence(&plan);
    }

    #[test]
    fn test_matmul_2_lanes() {
        let (g, _, _, _) = test_graphs::matmul(4, 8, 16);
        let plan = plan_spans(&g, 2);
        assert_spans_self_contained(&plan);
        assert_phase_independence(&plan);

        // With M=4 rows and 2 lanes, rows should distribute:
        // lane 0 gets rows 0,2 and lane 1 gets rows 1,3 (or similar).
        // Each lane should have work.
        for phase in &plan.phases {
            let active_lanes = phase
                .spans
                .iter()
                .filter(|s| s.graph.num_groups() > 0)
                .count();
            // At least one lane should be active per phase.
            assert!(active_lanes >= 1);
        }
    }

    #[test]
    fn test_matmul_4_lanes() {
        let (g, _, _, _) = test_graphs::matmul(8, 4, 16);
        let plan = plan_spans(&g, 4);
        assert_spans_self_contained(&plan);
        assert_phase_independence(&plan);
    }

    #[test]
    fn test_matmul_activation_2_lanes() {
        let (g, _, _, _) = test_graphs::matmul_activation(4, 8, 16, ScalarUnaryOp::Tanh);
        let plan = plan_spans(&g, 2);
        assert_spans_self_contained(&plan);
        assert_phase_independence(&plan);
    }

    // ─── Matmul chain tests ──────────────────────────────────────────────────

    #[test]
    fn test_matmul_chain_1_lane() {
        let (g, _, _, _, _) = test_graphs::matmul_chain(4, 8, 16, 16, 32);
        let plan = plan_spans(&g, 1);
        assert_spans_self_contained(&plan);
        assert_phase_independence(&plan);

        // Chain should produce multiple phases (barriers between matmuls).
        // With 1 lane, at least 1 phase.
        assert!(plan.phases.len() >= 1);
    }

    #[test]
    fn test_matmul_chain_2_lanes() {
        let (g, _, _, _, _) = test_graphs::matmul_chain(4, 8, 16, 16, 32);
        let plan = plan_spans(&g, 2);
        assert_spans_self_contained(&plan);
        assert_phase_independence(&plan);

        // Chain of 2 matmuls should have at least 2 phases
        // (barrier between first reduce and second mul).
        assert!(
            plan.phases.len() >= 2,
            "Expected >= 2 phases for matmul chain, got {}",
            plan.phases.len()
        );
    }

    #[test]
    fn test_matmul_chain_4_lanes() {
        let (g, _, _, _, _) = test_graphs::matmul_chain(8, 4, 16, 16, 32);
        let plan = plan_spans(&g, 4);
        assert_spans_self_contained(&plan);
        assert_phase_independence(&plan);
    }

    // ─── Correctness test: compare span execution with naive ─────────────────

    /// Execute a SpanPlan by running each span's NanoGraph through the naive
    /// executor, propagating values through the shared buffer between phases.
    fn execute_span_plan(
        plan: &SpanPlan,
        original_graph: &NanoGraph,
        original_inputs: &HashMap<u64, NumericScalar>,
    ) -> HashMap<u64, NumericScalar> {
        use crate::compiler::attempts::v13_claude::nano_execute::execute_nanograph_naive;

        // Shared values buffer: main-graph AtomId -> value.
        let mut shared: HashMap<u64, NumericScalar> = HashMap::new();

        // Pre-fill with original inputs.
        for (&k, v) in original_inputs {
            shared.insert(k, v.clone());
        }

        // Also pre-fill Literal atoms from the original graph.
        for group in original_graph.groups() {
            if let ScalarOp::Literal(scalar) = &group.op {
                for i in 0..group.count {
                    shared.insert(group.base_id.0 + i, scalar.clone());
                }
            }
        }

        // Apply original_inputs overrides after literals.
        for (&k, v) in original_inputs {
            shared.insert(k, v.clone());
        }

        for (pi, phase) in plan.phases.iter().enumerate() {
            // Execute each span independently, then merge outputs.
            let mut phase_outputs: Vec<Vec<(u64, NumericScalar)>> = Vec::new();

            for (li, span) in phase.spans.iter().enumerate() {
                if span.graph.num_groups() == 0 {
                    phase_outputs.push(vec![]);
                    continue;
                }

                // Build inputs for this span: overwrite placeholder atoms
                // with values from the shared buffer.
                let mut span_inputs: HashMap<u64, NumericScalar> = HashMap::new();
                for &(main_id, local_id) in &span.inputs {
                    if let Some(val) = shared.get(&main_id.0) {
                        span_inputs.insert(local_id.0, val.clone());
                    }
                }
                // Also inject original input overrides into duplicated literals.
                for &(main_id, local_id) in &span.literal_map {
                    if let Some(val) = shared.get(&main_id.0) {
                        span_inputs.insert(local_id.0, val.clone());
                    }
                }

                // Execute the span's NanoGraph.
                let span_values = execute_nanograph_naive(&span.graph, &span_inputs);

                // Collect outputs.
                let mut outputs = Vec::new();
                for &(local_id, main_id) in &span.outputs {
                    if let Some(val) = span_values.get(&local_id.0) {
                        outputs.push((main_id.0, val.clone()));
                    }
                }
                phase_outputs.push(outputs);
            }

            // Merge all span outputs into the shared buffer.
            for outputs in phase_outputs {
                for (main_atom, val) in outputs {
                    shared.insert(main_atom, val);
                }
            }
        }

        shared
    }

    #[test]
    fn test_correctness_elementwise_add() {
        let (g, a_base, b_base, c_base) =
            test_graphs::elementwise_binary(64, ScalarBinOp::Add);

        // Set up inputs: A[i] = i, B[i] = 100+i.
        let mut inputs = HashMap::new();
        for i in 0..64u64 {
            inputs.insert(a_base.0 + i, NumericScalar::F32(i as f32));
            inputs.insert(b_base.0 + i, NumericScalar::F32(100.0 + i as f32));
        }

        // Reference: naive execution.
        use crate::compiler::attempts::v13_claude::nano_execute::execute_nanograph_naive;
        let reference = execute_nanograph_naive(&g, &inputs);

        // Span execution with 2 lanes.
        let plan = plan_spans(&g, 2);
        assert_spans_self_contained(&plan);
        let span_result = execute_span_plan(&plan, &g, &inputs);

        // Compare outputs.
        for i in 0..64u64 {
            let atom = c_base.0 + i;
            let ref_val = reference.get(&atom).unwrap().to_f64();
            let span_val = span_result.get(&atom).unwrap().to_f64();
            assert!(
                (ref_val - span_val).abs() < 1e-6,
                "Mismatch at atom {}: ref={}, span={}",
                atom,
                ref_val,
                span_val
            );
        }
    }

    #[test]
    fn test_correctness_matmul_small() {
        // Small matmul: C[2,4] = A[2,3] @ B[3,4]
        let (g, a_base, b_base, reduce_base) = test_graphs::matmul(2, 3, 4);

        let mut inputs = HashMap::new();
        // A = [[1,2,3],[4,5,6]]
        let a_vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        for (i, &v) in a_vals.iter().enumerate() {
            inputs.insert(a_base.0 + i as u64, NumericScalar::F32(v));
        }
        // B = [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
        let b_vals = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        for (i, &v) in b_vals.iter().enumerate() {
            inputs.insert(b_base.0 + i as u64, NumericScalar::F32(v));
        }

        use crate::compiler::attempts::v13_claude::nano_execute::execute_nanograph_naive;
        let reference = execute_nanograph_naive(&g, &inputs);

        for num_lanes in [1, 2] {
            let plan = plan_spans(&g, num_lanes);
            assert_spans_self_contained(&plan);
            assert_phase_independence(&plan);
            let span_result = execute_span_plan(&plan, &g, &inputs);

            // C should be: [[1,2,3,0],[4,5,6,0]]
            for i in 0..8u64 {
                let atom = reduce_base.0 + i;
                let ref_val = reference.get(&atom).unwrap().to_f64();
                let span_val = span_result.get(&atom).unwrap().to_f64();
                assert!(
                    (ref_val - span_val).abs() < 1e-5,
                    "lanes={} atom {}: ref={}, span={}",
                    num_lanes,
                    atom,
                    ref_val,
                    span_val
                );
            }
        }
    }

    #[test]
    fn test_correctness_matmul_chain() {
        // Chain: D[2,4] = (A[2,3] @ B[3,3]) @ C[3,4]
        let (g, a_base, b_base, c_base, out_base) = test_graphs::matmul_chain(2, 3, 3, 3, 4);

        let mut inputs = HashMap::new();
        // A = [[1,0,0],[0,1,0]] (identity-like)
        let a_vals = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        for (i, &v) in a_vals.iter().enumerate() {
            inputs.insert(a_base.0 + i as u64, NumericScalar::F32(v));
        }
        // B = identity 3x3
        let b_vals = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        for (i, &v) in b_vals.iter().enumerate() {
            inputs.insert(b_base.0 + i as u64, NumericScalar::F32(v));
        }
        // C = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
        let c_vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        for (i, &v) in c_vals.iter().enumerate() {
            inputs.insert(c_base.0 + i as u64, NumericScalar::F32(v));
        }

        use crate::compiler::attempts::v13_claude::nano_execute::execute_nanograph_naive;
        let reference = execute_nanograph_naive(&g, &inputs);

        for num_lanes in [1, 2] {
            let plan = plan_spans(&g, num_lanes);
            assert_spans_self_contained(&plan);
            assert_phase_independence(&plan);
            let span_result = execute_span_plan(&plan, &g, &inputs);

            // Result should be: [[1,2,3,4],[5,6,7,8]] (A@I = A, then A@C = first 2 rows of C).
            for i in 0..8u64 {
                let atom = out_base.0 + i;
                let ref_val = reference.get(&atom).unwrap().to_f64();
                let span_val = span_result.get(&atom).unwrap().to_f64();
                assert!(
                    (ref_val - span_val).abs() < 1e-4,
                    "lanes={} atom {}: ref={}, span={}",
                    num_lanes,
                    atom,
                    ref_val,
                    span_val
                );
            }
        }
    }

    /// Test with a StridedBroadcast matmul (the real lowering format).
    /// Builds a matmul with merged Mul groups using StridedBroadcast.
    #[test]
    fn test_correctness_strided_broadcast_matmul() {
        // C[M,N] = A[M,K] @ B[K,N] with merged Mul groups.
        // M=4, K=3, N=4.
        let m = 4u64;
        let k = 3u64;
        let n = 4u64;

        let mut g = NanoGraph::new();

        let a = g.push_group(
            m * k,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );
        let b = g.push_group(
            k * n,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );

        // M merged Mul groups, each of count K*N with StridedBroadcast.
        let mut mul_bases = Vec::new();
        for mi in 0..m {
            let a_row_base = a.offset(mi * k);
            let mul_base = g.push_group(
                k * n,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![
                    InputRef::StridedBroadcast {
                        base: a_row_base,
                        stride: 1,
                        repeat: n,
                    },
                    InputRef::Affine {
                        base: b,
                        stride: 1,
                    },
                ],
            );
            mul_bases.push(mul_base);
        }

        // M ReduceSum groups, each of count N.
        let mut reduce_bases = Vec::new();
        for mi in 0..m {
            let reduce_base = g.push_group(
                n,
                ScalarOp::ReduceSum {
                    reduce_count: k,
                    reduce_stride: n as i64,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![InputRef::Affine {
                    base: mul_bases[mi as usize],
                    stride: 1,
                }],
            );
            reduce_bases.push(reduce_base);
        }
        g.outputs = vec![reduce_bases[0]];

        assert!(g.validate().is_empty(), "{:?}", g.validate());

        // Set up inputs.
        let mut inputs = HashMap::new();
        // A = [[1,0,0],[0,1,0],[0,0,1],[1,1,1]]
        let a_vals = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        for (i, &v) in a_vals.iter().enumerate() {
            inputs.insert(a.0 + i as u64, NumericScalar::F32(v));
        }
        // B = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
        let b_vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        for (i, &v) in b_vals.iter().enumerate() {
            inputs.insert(b.0 + i as u64, NumericScalar::F32(v));
        }

        use crate::compiler::attempts::v13_claude::nano_execute::execute_nanograph_naive;
        let reference = execute_nanograph_naive(&g, &inputs);

        for num_lanes in [1, 2, 4] {
            let plan = plan_spans(&g, num_lanes);
            assert_spans_self_contained(&plan);
            assert_phase_independence(&plan);
            let span_result = execute_span_plan(&plan, &g, &inputs);

            // Check all reduce output atoms.
            for mi in 0..m {
                for ni in 0..n {
                    let atom = reduce_bases[mi as usize].0 + ni;
                    let ref_val = reference.get(&atom).unwrap().to_f64();
                    let span_val = span_result.get(&atom).unwrap().to_f64();
                    assert!(
                        (ref_val - span_val).abs() < 1e-4,
                        "lanes={} C[{},{}] (atom {}): ref={}, span={}",
                        num_lanes,
                        mi,
                        ni,
                        atom,
                        ref_val,
                        span_val
                    );
                }
            }
        }
    }

    /// Test that span count grows with lane count.
    #[test]
    fn test_parallelism_scaling() {
        let (g, _, _, _) = test_graphs::matmul(8, 4, 16);

        for num_lanes in [1, 2, 4, 8] {
            let plan = plan_spans(&g, num_lanes);
            assert_spans_self_contained(&plan);
            assert_phase_independence(&plan);

            // Count active spans across all phases.
            let active_spans: usize = plan
                .phases
                .iter()
                .flat_map(|p| p.spans.iter())
                .filter(|s| s.graph.num_groups() > 0)
                .count();

            // Should have at least min(num_lanes, M) active spans.
            assert!(
                active_spans >= num_lanes.min(8),
                "Expected >= {} active spans with {} lanes, got {}",
                num_lanes.min(8),
                num_lanes,
                active_spans
            );
        }
    }

    /// Test that splitting a large elementwise group produces balanced spans.
    #[test]
    fn test_balance_elementwise() {
        let (g, _, _, _) = test_graphs::elementwise_binary(1024, ScalarBinOp::Mul);
        let plan = plan_spans(&g, 4);
        assert_spans_self_contained(&plan);
        assert_phase_independence(&plan);

        // Check balance: each lane should handle roughly 1024/4 = 256 compute atoms.
        for phase in &plan.phases {
            let lane_atoms: Vec<u64> = phase
                .spans
                .iter()
                .map(|s| {
                    s.graph
                        .groups()
                        .iter()
                        .filter(|g| !matches!(g.op, ScalarOp::Literal(_)))
                        .map(|g| g.count)
                        .sum::<u64>()
                })
                .collect();

            let max = lane_atoms.iter().copied().max().unwrap_or(0);
            let min = lane_atoms.iter().copied().filter(|&a| a > 0).min().unwrap_or(0);

            if max > 0 {
                let ratio = max as f64 / min.max(1) as f64;
                assert!(
                    ratio <= 2.0,
                    "Imbalanced lanes: {:?} (ratio {:.1}x)",
                    lane_atoms,
                    ratio
                );
            }
        }
    }
}
