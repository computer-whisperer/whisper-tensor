//! Lane+barrier execution planner v2c: Row-Slice Lane Planner with group splitting.
//!
//! Key innovations over previous attempts:
//!
//! 1. **AtomGroups are splittable.** A monolithic 49,152-atom elementwise group
//!    gets divided into sub-ranges assigned to different lanes. `LaneWork` tracks
//!    which slice of a group each lane handles.
//!
//! 2. **Barriers are structural, not positional.** A barrier is only placed when
//!    a group reads atoms produced by MULTIPLE lanes. Internal reduces (where
//!    input and output are on the same lane) do NOT generate barriers.
//!
//! 3. **Row-consistent lane assignment.** The "row dimension" (M in matmul
//!    M×K×N) is the natural split axis. Lane i consistently gets the same row
//!    slice across all phases, keeping data cache-hot in L1/L2.
//!
//! Algorithm overview:
//!
//! 1. Build group-level dependency DAG.
//! 2. Topological sort and compute "reduce depth" for each group (how many
//!    ReduceSum/ReduceMax layers separate it from sources).
//! 3. Identify mandatory barrier points: groups that consume atoms from
//!    multiple "row families" (where each row family is an independent
//!    row's worth of work through a matmul).
//! 4. Compute phases from barrier points.
//! 5. Within each phase, assign work to lanes with group splitting.
//!    Row-indexed groups (matmul Mul/ReduceSum with M independent rows)
//!    get row i assigned to lane i % num_lanes.
//!    Monolithic groups get split into num_lanes equal chunks.
//! 6. Verify correctness: no cross-lane reads within a phase.

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

use crate::nano_graph::{AtomGroup, AtomId, InputRef, NanoGraph, ScalarOp};

// ─── Public API ──────────────────────────────────────────────────────────────

/// A unit of work for one lane: a contiguous slice of an atom group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaneWork {
    /// Index into the NanoGraph's groups array.
    pub group_idx: usize,
    /// Offset within the group (0 for full group).
    pub atom_offset: u64,
    /// Number of atoms this lane handles (group.count for full group).
    pub atom_count: u64,
}

/// The full execution plan.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub num_lanes: usize,
    pub phases: Vec<Phase>,
}

/// One phase of execution (work between two consecutive barriers).
#[derive(Debug, Clone)]
pub struct Phase {
    /// `lane_work[lane_idx]` = work items assigned to that lane in this phase.
    pub lane_work: Vec<Vec<LaneWork>>,
}

/// Plan execution for a NanoGraph with `num_lanes` persistent threads.
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

    // Step 3: Identify row families.
    // A "row family" is a set of groups that form an independent row of computation
    // through a matmul. Groups in different row families are independent.
    let row_families = identify_row_families(groups, &producers, &consumers, &is_literal);

    // Step 4: Compute "reduce generation" for each group.
    // This counts how many full-width reduce boundaries separate a group from the source.
    // A full-width reduce is one that creates a cross-family sync point.
    let (group_phase, num_phases) = compute_phase_assignment(
        groups,
        &topo_order,
        &producers,
        &consumers,
        &is_literal,
        &row_families,
    );

    // Validate: every non-literal group's producers must be in an earlier or equal phase.
    validate_phase_ordering(groups, &producers, &is_literal, &group_phase, &row_families);

    // Step 5: Within each phase, assign work to lanes with group splitting.
    let phases = assign_lanes_with_splitting(
        groups,
        num_lanes,
        num_phases,
        &group_phase,
        &is_literal,
        &row_families,
        &producers,
    );

    ExecutionPlan { num_lanes, phases }
}

// ─── Group dependency graph ──────────────────────────────────────────────────

/// Build producer and consumer graphs at group level.
///
/// This accounts for THREE kinds of data dependencies:
/// 1. InputRef resolution (the basic case)
/// 2. ReduceSum/ReduceMax strided access (extends the read range beyond the InputRef)
/// 3. IndirectLoad table_base (reads from a table group embedded in the ScalarOp)
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

        // 2. ReduceSum/ReduceMax: the strided access extends beyond the InputRef range.
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

        // 3. IndirectLoad: table_base references a group whose atoms are read at runtime.
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

// ─── Row family identification ───────────────────────────────────────────────

/// A RowFamily identifies a set of groups that form an independent "row" of
/// computation. Groups within the same family share no data dependencies
/// (except through shared Literal groups).
///
/// For matmul M=64, K=768, N=768: each row m has one Mul group (K*N atoms)
/// and one ReduceSum group (N atoms). There are 64 independent row families.
///
/// For monolithic elementwise groups (count=49,152 = 64*768), there's one
/// "family" that spans all rows. These are the groups that need splitting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum RowFamily {
    /// This group is part of a specific independent row.
    /// The u64 is the group index of the "root" of this row family
    /// (typically the Mul group in a matmul row).
    Row(usize),
    /// This group spans all rows (monolithic). Needs splitting.
    AllRows,
    /// This is a data/Literal group (shared, not assigned to lanes).
    Shared,
}

/// Identify row families in the graph using forward propagation.
///
/// Algorithm:
/// 1. Find "root compute groups" — groups whose only producers are literals.
///    Group them by literal signature. Peer groups (>= 2 sharing a sig) become
///    separate Row families. Singletons become AllRows.
///
/// 2. Forward-propagate in topological order:
///    - If a group's non-literal producers are ALL from the SAME Row family,
///      it inherits that family (exclusive chain continuation).
///    - If producers span multiple Row families, or include AllRows producers,
///      the group is AllRows.
///    - If a group has NO non-literal producers, use the root classification
///      from step 1.
fn identify_row_families(
    groups: &[AtomGroup],
    producers: &[Vec<usize>],
    consumers: &[Vec<usize>],
    is_literal: &[bool],
) -> Vec<RowFamily> {
    let n = groups.len();
    let mut families = vec![RowFamily::AllRows; n];

    // Mark literals as Shared.
    for gi in 0..n {
        if is_literal[gi] {
            families[gi] = RowFamily::Shared;
        }
    }

    // Step 1: Classify root compute groups.
    // A root compute group has only literal (Shared) producers.
    let mut root_groups: Vec<usize> = Vec::new();
    for gi in 0..n {
        if is_literal[gi] {
            continue;
        }
        let all_producers_literal = producers[gi].iter().all(|&pi| is_literal[pi]);
        if all_producers_literal {
            root_groups.push(gi);
        }
    }

    // Group root compute groups by their literal producer signature.
    let mut sig_to_roots: HashMap<Vec<usize>, Vec<usize>> = HashMap::new();
    for &gi in &root_groups {
        let mut lit_sig: Vec<usize> = producers[gi]
            .iter()
            .filter(|&&pi| is_literal[pi])
            .copied()
            .collect();
        lit_sig.sort();
        lit_sig.dedup();
        sig_to_roots.entry(lit_sig).or_default().push(gi);
    }

    // Root groups with >= 2 peers sharing the same literal signature are Rows.
    for (_, roots) in &sig_to_roots {
        if roots.len() >= 2 {
            for &gi in roots {
                families[gi] = RowFamily::Row(gi); // Each root is its own Row family.
            }
        }
        // else: stays AllRows
    }

    // Step 2: Forward-propagate in topological order.
    let topo_order = topological_sort(n, producers);

    for &gi in &topo_order {
        if is_literal[gi] || producers[gi].iter().all(|&pi| is_literal[pi]) {
            // Already classified (literal or root).
            continue;
        }

        // Collect the row families of non-literal producers.
        let mut producer_families: HashSet<RowFamily> = HashSet::new();
        for &pi in &producers[gi] {
            if !is_literal[pi] {
                producer_families.insert(families[pi]);
            }
        }

        // Remove Shared (shouldn't appear for non-literals, but be safe).
        producer_families.remove(&RowFamily::Shared);

        if producer_families.is_empty() {
            // All producers are Shared/literal — this is a root group.
            // Should have been handled above, but leave as AllRows.
            continue;
        }

        if producer_families.len() == 1 {
            let single_family = *producer_families.iter().next().unwrap();
            match single_family {
                RowFamily::Row(_) => {
                    // All non-literal producers are from the same Row family.
                    // This group inherits that family.
                    families[gi] = single_family;
                }
                RowFamily::AllRows => {
                    // Producer is AllRows (monolithic). This group inherits AllRows
                    // UNLESS it's one of multiple peer consumers that each read
                    // a different slice of the AllRows producer.
                    //
                    // Detect this: if the AllRows producer has multiple compute
                    // consumers, and those consumers all have the same (op, count,
                    // lit-sig) — they're parallel rows reading from the split.
                    //
                    // For now, check if the producer's fan-out > 1 and all siblings
                    // are structurally similar.
                    let allrows_producers: Vec<usize> = producers[gi]
                        .iter()
                        .filter(|&&pi| !is_literal[pi] && families[pi] == RowFamily::AllRows)
                        .copied()
                        .collect();

                    // Check if this group is one of multiple peer consumers of the
                    // AllRows producer.
                    let mut is_row_of_allrows = false;
                    for &api in &allrows_producers {
                        let siblings: Vec<usize> = consumers[api]
                            .iter()
                            .filter(|&&ci| !is_literal[ci])
                            .copied()
                            .collect();
                        if siblings.len() >= 2 {
                            // Check structural similarity: same op discriminant, same count.
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
        // else: multiple different families among producers -> AllRows (default).
    }

    families
}

// ─── Lane-locality check for AllRows chains ─────────────────────────────────

/// Check whether an AllRows consumer group's inputs from same-phase AllRows
/// producers are all lane-local when both consumer and producers are split
/// evenly across lanes.
///
/// Only checks inputs that reference AllRows producers at `max_prod_phase`.
/// Inputs from earlier-phase producers are already behind a barrier and
/// don't need cross-lane checking.
///
/// An input is lane-local if lane j's slice of the consumer only reads
/// from lane j's slice of each same-phase AllRows producer it references.
///
/// Lane-local patterns:
/// - `Affine{stride=1}` from an AllRows producer with the same count:
///   lane j reads `[j*chunk..(j+1)*chunk)` from producer = lane j's chunk.
/// - `StridedBroadcast{stride, repeat}` from an AllRows producer where
///   the mapping is proportional (monotonically maps each lane's chunk).
/// - `Broadcast` from an earlier-phase producer: OK (already synchronized).
///
/// NOT lane-local:
/// - `Broadcast` from a same-phase AllRows producer (one lane owns the
///   atom, all lanes read it).
/// - `Affine` with stride != 1 between same-phase AllRows of different counts.
/// - `Modular` (wraps around, generally crosses lanes).
/// - `Explicit` (arbitrary mapping).
fn is_allrows_chain_lane_local_samephase(
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
        // Resolve which groups this input references.
        let prod_groups = resolve_producer_groups(input, consumer_count, groups);

        // Filter to non-literal AllRows producers IN THE SAME PHASE.
        // Earlier-phase producers are already behind a barrier.
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
            // All producers for this input are either literal, non-AllRows,
            // or in an earlier phase. No cross-lane issue from this input.
            continue;
        }

        // Check the access pattern for lane-locality.
        match input {
            InputRef::Affine { stride, .. } => {
                if *stride != 1 {
                    return false; // Non-unit stride can cross lane boundaries.
                }
                // For stride=1, check that each same-phase AllRows producer
                // has the same count as the consumer. Then lane j's chunk
                // maps 1:1.
                for &pi in &same_phase_allrows {
                    if groups[pi].count != consumer_count {
                        return false;
                    }
                }
            }
            InputRef::Broadcast(_) => {
                // Broadcast from a same-phase AllRows producer: the broadcast
                // atom lives in one lane, but ALL lanes read it → cross-lane.
                return false;
            }
            InputRef::StridedBroadcast {
                base: _,
                stride: _,
                repeat,
            } => {
                // atom i reads base + stride * (i / repeat).
                // The mapping i → i/repeat is monotonically non-decreasing,
                // so lane j's consumer range maps proportionally to the
                // producer's range. Lane-local if each same-phase AllRows
                // producer has count = ceil(consumer_count / repeat).
                for &pi in &same_phase_allrows {
                    let expected_prod_atoms = (consumer_count + repeat - 1) / repeat;
                    if groups[pi].count != expected_prod_atoms {
                        return false;
                    }
                }
            }
            InputRef::Modular { .. } => {
                // Modular wraps around — not lane-local in general.
                return false;
            }
            InputRef::Explicit(_) => {
                // Arbitrary mapping — can't prove lane-locality.
                return false;
            }
            InputRef::SymAffine { .. } => {
                // SymAffine is used for contractions (matmul inner loop).
                // Not lane-local.
                return false;
            }
        }
    }

    // Also check ReduceSum/ReduceMax: the reduce itself must be lane-local.
    // For ReduceSum with reduce_stride != 0, atom i accumulates over
    // resolved_base(i) + k * reduce_stride for k in 0..reduce_count.
    // This reads across stride boundaries in the input, potentially
    // spanning multiple lanes AND multiple producer groups.
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
                // Check if any input (including extended reduce stride range)
                // references a same-phase AllRows producer. If so, the strided
                // reads would cross lane boundaries.
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
                // If the reduce input is from an earlier phase, it's already
                // synchronized — the reduce can proceed lane-locally if the
                // InputRef check above passed.
            }
        }
        _ => {}
    }

    true
}

// ─── Phase assignment ────────────────────────────────────────────────────────

/// Compute phase assignments based on structural barrier detection.
///
/// A barrier is needed when a group reads atoms from groups assigned to
/// multiple different row families. This happens when:
/// - An elementwise group reads the output of a matmul (all M row families
///   feed into one monolithic group)
/// - A group reads from a shared intermediate that multiple rows produced
///
/// The algorithm walks in topo order and increments the phase counter
/// when it encounters a cross-family dependency.
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

    // For each group in topo order, its phase = max(producer phases) + barrier_penalty.
    //
    // barrier_penalty = 1 if the group reads from producers on different lanes
    // (different row families or AllRows producers split across lanes).
    // barrier_penalty = 0 if all non-literal producers are in the same row family.

    for &gi in topo_order {
        if is_literal[gi] {
            group_phase[gi] = 0;
            continue;
        }

        // Collect the phases and families of non-literal producers.
        let mut max_prod_phase = 0usize;

        for &pi in &producers[gi] {
            if is_literal[pi] {
                continue;
            }
            max_prod_phase = max_prod_phase.max(group_phase[pi]);
        }

        // Only consider producers at max_prod_phase for cross-lane analysis.
        // Producers in earlier phases are already behind a barrier — no concern.
        let mut same_phase_families: HashSet<RowFamily> = HashSet::new();
        for &pi in &producers[gi] {
            if is_literal[pi] {
                continue;
            }
            if group_phase[pi] == max_prod_phase {
                same_phase_families.insert(row_families[pi]);
            }
        }

        // Remove AllRows — an AllRows producer will be split across lanes,
        // so it acts as if it's in multiple families.
        let has_all_rows = same_phase_families.remove(&RowFamily::AllRows);

        // Count distinct Row families among same-phase producers.
        let distinct_row_families = same_phase_families
            .iter()
            .filter(|f| matches!(f, RowFamily::Row(_)))
            .count();

        // Need barrier if same-phase producers create cross-lane dependencies:
        // - Multiple distinct row families at max_prod_phase (e.g., all M rows
        //   of matmul feed into one elementwise)
        // - An AllRows producer at max_prod_phase alongside Row producers
        //   (AllRows producer was split across lanes, so we need all lanes
        //   to finish before reading it)
        // - An AllRows producer at max_prod_phase feeds an AllRows consumer
        //   BUT the access pattern is NOT lane-local
        //
        // Key insight: producers in earlier phases (< max_prod_phase) are
        // already synchronized by the barrier that created their phase
        // boundary. Only same-phase producers matter for cross-lane checks.
        let allrows_to_allrows_needs_barrier = if has_all_rows
            && same_phase_families.is_empty()
            && row_families[gi] == RowFamily::AllRows
        {
            // All same-phase non-literal producers are AllRows, consumer is AllRows.
            // Check if the access pattern from same-phase AllRows producers is lane-local.
            !is_allrows_chain_lane_local_samephase(
                gi,
                groups,
                row_families,
                is_literal,
                &group_phase,
                max_prod_phase,
            )
        } else if has_all_rows && same_phase_families.is_empty() {
            // Consumer is not AllRows (e.g. Row) reading from AllRows at same phase
            true
        } else {
            false
        };

        // Also: if consumer is AllRows (split across all lanes) but reads
        // from any Row producer at same phase, it's cross-lane: the Row
        // producer is on one specific lane, but other lanes of the consumer
        // also need that data.
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

    let num_phases = group_phase
        .iter()
        .copied()
        .max()
        .map(|m| m + 1)
        .unwrap_or(1);
    (group_phase, num_phases)
}

/// Validate that phase ordering respects data dependencies: for every non-literal
/// group, all atoms it reads must come from groups in earlier or equal phases.
///
/// This validation is MORE THOROUGH than just checking the `producers` list --
/// it independently discovers all dependencies by examining InputRefs AND
/// ReduceSum/ReduceMax strided access AND IndirectLoad table_base references.
fn validate_phase_ordering(
    groups: &[AtomGroup],
    producers: &[Vec<usize>],
    is_literal: &[bool],
    group_phase: &[usize],
    row_families: &[RowFamily],
) {
    let n = groups.len();
    for gi in 0..n {
        if is_literal[gi] {
            continue;
        }
        let my_phase = group_phase[gi];
        let group = &groups[gi];

        // Check 1: All producer groups from the InputRef resolution.
        for input in &group.inputs {
            let prods = resolve_producer_groups(input, group.count, groups);
            for pi in prods {
                if pi == gi || is_literal[pi] {
                    continue;
                }
                if group_phase[pi] > my_phase {
                    panic!(
                        "Phase ordering violation (InputRef): group {} (phase {}, family {:?}, op {:?}) \
                         reads from producer group {} (phase {}, family {:?}, op {:?}), \
                         but producer is in a LATER phase! \
                         Group {} base_id={}, count={}; Producer {} base_id={}, count={}",
                        gi,
                        my_phase,
                        row_families[gi],
                        std::mem::discriminant(&groups[gi].op),
                        pi,
                        group_phase[pi],
                        row_families[pi],
                        std::mem::discriminant(&groups[pi].op),
                        gi,
                        groups[gi].base_id.0,
                        groups[gi].count,
                        pi,
                        groups[pi].base_id.0,
                        groups[pi].count,
                    );
                }
            }
        }

        // Check 2: ReduceSum/ReduceMax strided access extends the read range.
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
                    // Compute the FULL accessed range including reduce strides.
                    let extended_prods = resolve_producer_groups_with_reduce(
                        input,
                        group.count,
                        *reduce_count,
                        *reduce_stride,
                        groups,
                    );
                    for pi in extended_prods {
                        if pi == gi || is_literal[pi] {
                            continue;
                        }
                        if group_phase[pi] > my_phase {
                            panic!(
                                "Phase ordering violation (ReduceStride): group {} (phase {}) \
                                 reads from group {} (phase {}) via reduce stride access. \
                                 reduce_count={}, reduce_stride={}. \
                                 Group {} base_id={}, count={}; Producer {} base_id={}, count={}",
                                gi,
                                my_phase,
                                pi,
                                group_phase[pi],
                                reduce_count,
                                reduce_stride,
                                gi,
                                groups[gi].base_id.0,
                                groups[gi].count,
                                pi,
                                groups[pi].base_id.0,
                                groups[pi].count,
                            );
                        }
                    }
                }
            }
            _ => {}
        }

        // Check 3: IndirectLoad table_base dependency.
        if let ScalarOp::IndirectLoad { table_base, .. } = &group.op {
            // The table_base points to a group whose atoms are read at runtime.
            // We don't know the exact range (data-dependent), but the table group
            // must be in an earlier or equal phase.
            if let Some(pi) = find_group_idx(groups, *table_base) {
                if !is_literal[pi] && group_phase[pi] > my_phase {
                    panic!(
                        "Phase ordering violation (IndirectLoad table_base): group {} (phase {}) \
                         reads from table group {} (phase {}) via IndirectLoad. \
                         Group {} base_id={}, count={}; Table group {} base_id={}, count={}",
                        gi,
                        my_phase,
                        pi,
                        group_phase[pi],
                        gi,
                        groups[gi].base_id.0,
                        groups[gi].count,
                        pi,
                        groups[pi].base_id.0,
                        groups[pi].count,
                    );
                }
            }
        }
    }
}

/// Resolve producer groups considering ReduceSum/ReduceMax strided access.
/// The reduce operation reads: resolved_input(i) + k * reduce_stride for k in 0..reduce_count.
/// This extends the accessed range beyond what the InputRef alone specifies.
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
            let first_block = 0i64;
            let last_block = ((count - 1) / repeat) as i64;
            let first_read = base.0 as i64 + stride * first_block;
            let last_read = base.0 as i64 + stride * last_block;
            let lo = first_read.min(last_read) + min_reduce_ext;
            let hi = first_read.max(last_read) + max_reduce_ext;
            find_groups_in_range(groups, lo as u64, hi as u64)
        }
        InputRef::Broadcast(atom_id) => {
            // Broadcast: all atoms read from the same base. Extend by reduce stride.
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
        _ => {
            // Modular, Explicit: fall back to the basic resolution.
            // These are conservative enough for most cases.
            resolve_producer_groups(input, count, groups)
        }
    }
}

// ─── Lane assignment with group splitting ────────────────────────────────────

/// Assign work to lanes within each phase, splitting groups as needed.
///
/// Strategy:
/// - Row(root) groups: assign to lane `row_index % num_lanes` where
///   row_index is determined by the root's position among its peers.
/// - AllRows groups: split into num_lanes equal chunks.
/// - Groups that read from AllRows producers (after a barrier): these
///   are typically the next set of row groups. They inherit their lane
///   from their row family.
fn assign_lanes_with_splitting(
    groups: &[AtomGroup],
    num_lanes: usize,
    num_phases: usize,
    group_phase: &[usize],
    is_literal: &[bool],
    row_families: &[RowFamily],
    producers: &[Vec<usize>],
) -> Vec<Phase> {
    let n = groups.len();

    // For Row families, we need to know the lane assignment for each root.
    // Group rows by their literal signature, then assign lanes round-robin.
    let mut root_to_lane: HashMap<usize, usize> = HashMap::new();
    {
        // For each Row root, collect all members and compute literal signature.
        let mut root_members: HashMap<usize, Vec<usize>> = HashMap::new();
        for gi in 0..n {
            if let RowFamily::Row(root) = row_families[gi] {
                root_members.entry(root).or_default().push(gi);
            }
        }

        let mut sig_to_roots2: HashMap<Vec<usize>, Vec<usize>> = HashMap::new();
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
            sig_to_roots2.entry(sig).or_default().push(root);
        }

        // Assign lanes round-robin within each signature group.
        // Sort roots by their group index for determinism.
        for (_, roots) in &mut sig_to_roots2 {
            roots.sort();
            for (i, &root) in roots.iter().enumerate() {
                root_to_lane.insert(root, i % num_lanes);
            }
        }
    }

    // Build phases.
    let mut phases = Vec::with_capacity(num_phases);

    for phase_idx in 0..num_phases {
        let mut lane_work: Vec<Vec<LaneWork>> = vec![Vec::new(); num_lanes];

        // Collect all compute groups in this phase.
        let phase_groups: Vec<usize> = (0..n)
            .filter(|&gi| !is_literal[gi] && group_phase[gi] == phase_idx)
            .collect();

        for &gi in &phase_groups {
            match row_families[gi] {
                RowFamily::Shared => {
                    // Literals are not assigned. This shouldn't happen since
                    // we filtered is_literal above.
                }
                RowFamily::Row(root) => {
                    // Assign to the lane determined by this row family.
                    let lane = root_to_lane.get(&root).copied().unwrap_or(0);
                    lane_work[lane].push(LaneWork {
                        group_idx: gi,
                        atom_offset: 0,
                        atom_count: groups[gi].count,
                    });
                }
                RowFamily::AllRows => {
                    // Split this group across lanes.
                    let count = groups[gi].count;
                    if num_lanes == 1 || count == 0 {
                        lane_work[0].push(LaneWork {
                            group_idx: gi,
                            atom_offset: 0,
                            atom_count: count,
                        });
                    } else {
                        // Try to find a natural split granularity.
                        // If count is divisible by some factor that relates to
                        // the row dimension, use that.
                        let chunk = split_count(count, num_lanes);
                        let mut offset = 0u64;
                        for lane in 0..num_lanes {
                            if offset >= count {
                                break; // No more atoms to assign.
                            }
                            let remaining = count - offset;
                            let this_chunk = if lane < num_lanes - 1 {
                                chunk.min(remaining)
                            } else {
                                remaining
                            };
                            if this_chunk > 0 {
                                lane_work[lane].push(LaneWork {
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

        // Sort work items within each lane by group index for determinism.
        for lane in &mut lane_work {
            lane.sort_by_key(|w| (w.group_idx, w.atom_offset));
        }

        phases.push(Phase { lane_work });
    }

    phases
}

/// Compute chunk size for splitting `count` atoms across `num_lanes` lanes.
/// Returns the size of each chunk (last lane gets the remainder).
fn split_count(count: u64, num_lanes: usize) -> u64 {
    // Divide as evenly as possible.
    (count + num_lanes as u64 - 1) / num_lanes as u64
}

// ─── Union-Find ──────────────────────────────────────────────────────────────

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) -> usize {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return ra;
        }
        let (big, small) = if self.rank[ra] >= self.rank[rb] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.parent[small] = big;
        if self.rank[big] == self.rank[small] {
            self.rank[big] += 1;
        }
        big
    }
}

// ─── Diagnostics ─────────────────────────────────────────────────────────────

impl ExecutionPlan {
    /// Print summary statistics about the plan.
    pub fn print_summary(&self, groups: &[AtomGroup]) {
        println!(
            "ExecutionPlan: {} lanes, {} phases",
            self.num_lanes,
            self.phases.len()
        );

        let is_literal: Vec<bool> = groups
            .iter()
            .map(|g| matches!(g.op, ScalarOp::Literal(_)) && g.inputs.is_empty())
            .collect();

        let total_compute_groups = (0..groups.len()).filter(|&i| !is_literal[i]).count();
        let mut assigned_groups: HashSet<usize> = HashSet::new();
        let mut total_assigned_atoms = 0u64;

        for (phase_idx, phase) in self.phases.iter().enumerate() {
            let lane_atoms: Vec<u64> = phase
                .lane_work
                .iter()
                .map(|lane| lane.iter().map(|w| w.atom_count).sum::<u64>())
                .collect();

            let total_phase_atoms: u64 = lane_atoms.iter().sum();
            let max_atoms = lane_atoms.iter().copied().max().unwrap_or(0);
            let min_atoms = lane_atoms
                .iter()
                .copied()
                .filter(|&a| a > 0)
                .min()
                .unwrap_or(0);

            let groups_in_phase: usize = phase.lane_work.iter().map(|lane| lane.len()).sum();

            for lane in &phase.lane_work {
                for w in lane {
                    assigned_groups.insert(w.group_idx);
                    total_assigned_atoms += w.atom_count;
                }
            }

            let balance = if min_atoms > 0 {
                format!("{:.1}x", max_atoms as f64 / min_atoms as f64)
            } else {
                "N/A".to_string()
            };

            let active_lanes = lane_atoms.iter().filter(|&&a| a > 0).count();

            println!(
                "  Phase {}: {} work items, {} atoms, {} active lanes, balance {}",
                phase_idx, groups_in_phase, total_phase_atoms, active_lanes, balance
            );
        }

        println!(
            "  Coverage: {}/{} compute groups assigned, {} total atoms in plan",
            assigned_groups.len(),
            total_compute_groups,
            total_assigned_atoms
        );

        // Overall balance: worst imbalance across all phases.
        let mut worst_ratio = 1.0f64;
        let mut worst_phase = 0;
        for (phase_idx, phase) in self.phases.iter().enumerate() {
            let lane_atoms: Vec<u64> = phase
                .lane_work
                .iter()
                .map(|lane| lane.iter().map(|w| w.atom_count).sum::<u64>())
                .collect();
            let non_zero: Vec<u64> = lane_atoms.iter().copied().filter(|&a| a > 0).collect();
            if non_zero.len() > 1 {
                let max_a = *non_zero.iter().max().unwrap();
                let min_a = *non_zero.iter().min().unwrap();
                if min_a > 0 {
                    let ratio = max_a as f64 / min_a as f64;
                    if ratio > worst_ratio {
                        worst_ratio = ratio;
                        worst_phase = phase_idx;
                    }
                }
            }
        }
        println!(
            "  Worst balance: {:.1}x in phase {}",
            worst_ratio, worst_phase
        );
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::nano_graph::{InputRef, NanoGraph, ScalarBinOp, ScalarOp, ScalarUnaryOp};
    use crate::numeric_scalar::NumericScalar;

    // ─── Test graph builders ──────────────────────────────────────────────

    /// Build a simple matmul C[M,N] = A[M,K] @ B[K,N].
    fn build_matmul(m: u64, k: u64, n: u64) -> NanoGraph {
        let mut g = NanoGraph::new();

        let a_base = g.push_group(
            m * k,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        let b_base = g.push_group(
            k * n,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        let mut mul_bases = Vec::new();
        for row in 0..m {
            let a_row = AtomId(a_base.0 + row * k);
            let mul = g.push_group(
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
                        base: a_row,
                        stride: 1,
                        repeat: n,
                    },
                    InputRef::Affine {
                        base: b_base,
                        stride: 1,
                    },
                ],
            );
            mul_bases.push(mul);
        }

        let mut reduce_bases = Vec::new();
        for row in 0..m {
            let red = g.push_group(
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
                    base: mul_bases[row as usize],
                    stride: 1,
                }],
            );
            reduce_bases.push(red);
        }

        for &rb in &reduce_bases {
            for i in 0..n {
                g.outputs.push(AtomId(rb.0 + i));
            }
        }

        g
    }

    /// Build a chain: matmul1 -> elementwise activation -> matmul2.
    /// This is the critical test case: the elementwise activation is a
    /// monolithic group (M*N1 atoms) that needs splitting.
    fn build_matmul_chain(m: u64, k1: u64, n1: u64, k2: u64, n2: u64) -> NanoGraph {
        let mut g = NanoGraph::new();

        // Weights for matmul 1.
        let a1 = g.push_group(
            m * k1,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let b1 = g.push_group(
            k1 * n1,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        // Matmul 1: M independent rows.
        let mut mul1_bases = Vec::new();
        for row in 0..m {
            let a_row = AtomId(a1.0 + row * k1);
            let mul = g.push_group(
                k1 * n1,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![
                    InputRef::StridedBroadcast {
                        base: a_row,
                        stride: 1,
                        repeat: n1,
                    },
                    InputRef::Affine {
                        base: b1,
                        stride: 1,
                    },
                ],
            );
            mul1_bases.push(mul);
        }

        let mut red1_bases = Vec::new();
        for row in 0..m {
            let red = g.push_group(
                n1,
                ScalarOp::ReduceSum {
                    reduce_count: k1,
                    reduce_stride: n1 as i64,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![InputRef::Affine {
                    base: mul1_bases[row as usize],
                    stride: 1,
                }],
            );
            red1_bases.push(red);
        }

        // Elementwise activation (Tanh) on matmul1 output.
        // In the real GPT-2 lowering, this is ONE monolithic group of M*N1 atoms.
        // That's the pathological case: it reads from all M ReduceSum outputs.
        let act = g.push_group(
            m * n1,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Tanh,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: red1_bases[0],
                stride: 1,
            }],
        );

        // Weights for matmul 2.
        let b2 = g.push_group(
            n1 * n2,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        // Matmul 2: reads from activation output (the monolithic group).
        let mut mul2_bases = Vec::new();
        for row in 0..m {
            let mul = g.push_group(
                n1 * n2,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![
                    InputRef::StridedBroadcast {
                        base: AtomId(act.0 + row * n1),
                        stride: 1,
                        repeat: n2,
                    },
                    InputRef::Affine {
                        base: b2,
                        stride: 1,
                    },
                ],
            );
            mul2_bases.push(mul);
        }

        let mut red2_bases = Vec::new();
        for row in 0..m {
            let red = g.push_group(
                n2,
                ScalarOp::ReduceSum {
                    reduce_count: n1,
                    reduce_stride: n2 as i64,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![InputRef::Affine {
                    base: mul2_bases[row as usize],
                    stride: 1,
                }],
            );
            red2_bases.push(red);
        }

        for &rb in &red2_bases {
            for i in 0..n2 {
                g.outputs.push(AtomId(rb.0 + i));
            }
        }

        g
    }

    /// Build a chain with per-row elementwise (NOT monolithic).
    /// This is the ideal case: lowering already split the activation per row.
    fn build_matmul_chain_per_row(m: u64, k1: u64, n1: u64, k2: u64, n2: u64) -> NanoGraph {
        let mut g = NanoGraph::new();

        let a1 = g.push_group(
            m * k1,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let b1 = g.push_group(
            k1 * n1,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        let mut mul1_bases = Vec::new();
        for row in 0..m {
            let a_row = AtomId(a1.0 + row * k1);
            let mul = g.push_group(
                k1 * n1,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![
                    InputRef::StridedBroadcast {
                        base: a_row,
                        stride: 1,
                        repeat: n1,
                    },
                    InputRef::Affine {
                        base: b1,
                        stride: 1,
                    },
                ],
            );
            mul1_bases.push(mul);
        }

        let mut red1_bases = Vec::new();
        for row in 0..m {
            let red = g.push_group(
                n1,
                ScalarOp::ReduceSum {
                    reduce_count: k1,
                    reduce_stride: n1 as i64,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![InputRef::Affine {
                    base: mul1_bases[row as usize],
                    stride: 1,
                }],
            );
            red1_bases.push(red);
        }

        // Per-row activation: M separate groups of N1 atoms each.
        let mut act_bases = Vec::new();
        for row in 0..m {
            let act = g.push_group(
                n1,
                ScalarOp::Unary {
                    op: ScalarUnaryOp::Tanh,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![InputRef::Affine {
                    base: red1_bases[row as usize],
                    stride: 1,
                }],
            );
            act_bases.push(act);
        }

        let b2 = g.push_group(
            n1 * n2,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        let mut mul2_bases = Vec::new();
        for row in 0..m {
            let mul = g.push_group(
                n1 * n2,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![
                    InputRef::StridedBroadcast {
                        base: act_bases[row as usize],
                        stride: 1,
                        repeat: n2,
                    },
                    InputRef::Affine {
                        base: b2,
                        stride: 1,
                    },
                ],
            );
            mul2_bases.push(mul);
        }

        let mut red2_bases = Vec::new();
        for row in 0..m {
            let red = g.push_group(
                n2,
                ScalarOp::ReduceSum {
                    reduce_count: n1,
                    reduce_stride: n2 as i64,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![InputRef::Affine {
                    base: mul2_bases[row as usize],
                    stride: 1,
                }],
            );
            red2_bases.push(red);
        }

        for &rb in &red2_bases {
            for i in 0..n2 {
                g.outputs.push(AtomId(rb.0 + i));
            }
        }

        g
    }

    /// Build two parallel matmuls from the same input (like Q/K projections).
    fn build_parallel_matmuls(m: u64, k: u64, n: u64) -> NanoGraph {
        let mut g = NanoGraph::new();

        let a = g.push_group(
            m * k,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        let b1 = g.push_group(
            k * n,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        let b2 = g.push_group(
            k * n,
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
            vec![],
        );

        let mut red1_bases = Vec::new();
        for row in 0..m {
            let a_row = AtomId(a.0 + row * k);
            let mul = g.push_group(
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
                        base: a_row,
                        stride: 1,
                        repeat: n,
                    },
                    InputRef::Affine {
                        base: b1,
                        stride: 1,
                    },
                ],
            );
            let red = g.push_group(
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
                    base: mul,
                    stride: 1,
                }],
            );
            red1_bases.push(red);
        }

        let mut red2_bases = Vec::new();
        for row in 0..m {
            let a_row = AtomId(a.0 + row * k);
            let mul = g.push_group(
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
                        base: a_row,
                        stride: 1,
                        repeat: n,
                    },
                    InputRef::Affine {
                        base: b2,
                        stride: 1,
                    },
                ],
            );
            let red = g.push_group(
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
                    base: mul,
                    stride: 1,
                }],
            );
            red2_bases.push(red);
        }

        for &rb in &red1_bases {
            for i in 0..n {
                g.outputs.push(AtomId(rb.0 + i));
            }
        }
        for &rb in &red2_bases {
            for i in 0..n {
                g.outputs.push(AtomId(rb.0 + i));
            }
        }

        g
    }

    /// Build a simple elementwise graph.
    fn build_elementwise(count: u64) -> NanoGraph {
        let mut g = NanoGraph::new();
        let a = g.push_group(
            count,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let b = g.push_group(
            count,
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
            vec![],
        );
        let c = g.push_group(
            count,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine { base: a, stride: 1 },
                InputRef::Affine { base: b, stride: 1 },
            ],
        );
        for i in 0..count {
            g.outputs.push(AtomId(c.0 + i));
        }
        g
    }

    // ─── Validation helpers ──────────────────────────────────────────────

    /// Verify every compute group's atoms appear exactly once across all lanes/phases.
    fn verify_coverage(graph: &NanoGraph, plan: &ExecutionPlan) {
        let groups = graph.groups();
        let is_literal: Vec<bool> = groups
            .iter()
            .map(|g| matches!(g.op, ScalarOp::Literal(_)) && g.inputs.is_empty())
            .collect();

        // Track atom coverage per group.
        let mut group_atoms_assigned: HashMap<usize, Vec<(u64, u64)>> = HashMap::new();

        for (phase_idx, phase) in plan.phases.iter().enumerate() {
            for (lane_idx, lane_groups) in phase.lane_work.iter().enumerate() {
                for work in lane_groups {
                    group_atoms_assigned
                        .entry(work.group_idx)
                        .or_default()
                        .push((work.atom_offset, work.atom_count));
                }
            }
        }

        // Every compute group must be fully covered.
        for gi in 0..groups.len() {
            if is_literal[gi] {
                continue;
            }

            let assigned = group_atoms_assigned.get(&gi);
            if assigned.is_none() {
                panic!("Compute group {} not assigned to any lane/phase", gi);
            }

            let ranges = assigned.unwrap();
            // Check that ranges cover [0, count) without overlap.
            let mut sorted_ranges: Vec<(u64, u64)> = ranges.clone();
            sorted_ranges.sort();

            let mut covered = 0u64;
            for &(offset, count) in &sorted_ranges {
                if offset != covered {
                    panic!(
                        "Group {}: gap in atom coverage at offset {} (expected {})",
                        gi, offset, covered
                    );
                }
                covered += count;
            }
            if covered != groups[gi].count {
                panic!(
                    "Group {}: covered {} atoms but group has {}",
                    gi, covered, groups[gi].count
                );
            }
        }
    }

    /// Verify within-phase independence: no lane reads atoms produced by
    /// another lane in the same phase.
    ///
    /// This version is split-aware: it checks at the atom level by computing
    /// the actual atom range each consumer slice reads from each producer,
    /// and verifying those atoms belong to the same lane.
    fn verify_phase_independence(graph: &NanoGraph, plan: &ExecutionPlan) {
        let groups = graph.groups();

        for (phase_idx, phase) in plan.phases.iter().enumerate() {
            // Build map: group_idx -> [(atom_offset, atom_count, lane_idx)]
            // for all work items produced in this phase.
            let mut atom_to_lane: HashMap<usize, Vec<(u64, u64, usize)>> = HashMap::new();

            for (lane_idx, lane_work) in phase.lane_work.iter().enumerate() {
                for work in lane_work {
                    atom_to_lane.entry(work.group_idx).or_default().push((
                        work.atom_offset,
                        work.atom_count,
                        lane_idx,
                    ));
                }
            }

            // For each work item, check that all atoms it reads from
            // same-phase producers belong to the same lane.
            for (lane_idx, lane_work) in phase.lane_work.iter().enumerate() {
                for work in lane_work {
                    let group = &groups[work.group_idx];
                    // For this work item's slice [atom_offset..atom_offset+atom_count),
                    // compute the range of atoms it reads from each input.
                    for input in &group.inputs {
                        // Compute the actual atom IDs accessed by this work item's slice.
                        let accessed_ranges = compute_accessed_ranges(
                            input,
                            work.atom_offset,
                            work.atom_count,
                            group.count,
                            groups,
                        );

                        for (prod_gi, prod_lo, prod_hi) in &accessed_ranges {
                            if let Some(lane_entries) = atom_to_lane.get(prod_gi) {
                                for &(prod_offset, prod_count, prod_lane) in lane_entries {
                                    // Check if the accessed range overlaps with this
                                    // producer lane's range.
                                    let prod_end = prod_offset + prod_count;
                                    if *prod_lo < prod_end && *prod_hi > prod_offset {
                                        // There's overlap — check same lane.
                                        if prod_lane != lane_idx {
                                            panic!(
                                                "Phase {}: group {} slice [{},+{}) (lane {}) reads atoms [{},{}) from group {} (lane {}, slice [{},+{})) - cross-lane dependency!",
                                                phase_idx,
                                                work.group_idx,
                                                work.atom_offset,
                                                work.atom_count,
                                                lane_idx,
                                                prod_lo,
                                                prod_hi,
                                                prod_gi,
                                                prod_lane,
                                                prod_offset,
                                                prod_count,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // For ReduceSum/ReduceMax, also check the strided access.
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
                            // Each atom i in [atom_offset..atom_offset+atom_count)
                            // reads from input at resolved_base(i) + k*reduce_stride
                            // for k in 0..reduce_count.
                            // Compute the full span for this slice.
                            for input in &group.inputs {
                                let accessed = compute_accessed_ranges_with_reduce(
                                    input,
                                    work.atom_offset,
                                    work.atom_count,
                                    group.count,
                                    *reduce_count,
                                    *reduce_stride,
                                    groups,
                                );
                                for (prod_gi, prod_lo, prod_hi) in &accessed {
                                    if let Some(lane_entries) = atom_to_lane.get(prod_gi) {
                                        for &(prod_offset, prod_count, prod_lane) in lane_entries {
                                            let prod_end = prod_offset + prod_count;
                                            if *prod_lo < prod_end && *prod_hi > prod_offset {
                                                if prod_lane != lane_idx {
                                                    panic!(
                                                        "Phase {}: group {} (ReduceSum) slice [{},+{}) (lane {}) reads atoms [{},{}) from group {} (lane {}) - cross-lane dependency!",
                                                        phase_idx,
                                                        work.group_idx,
                                                        work.atom_offset,
                                                        work.atom_count,
                                                        lane_idx,
                                                        prod_lo,
                                                        prod_hi,
                                                        prod_gi,
                                                        prod_lane,
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    /// Compute the range of atoms accessed by a work item's slice from an input.
    /// Returns list of (producer_group_idx, lo_offset_in_group, hi_offset_in_group).
    /// Offsets are relative to the producer group's base.
    fn compute_accessed_ranges(
        input: &InputRef,
        atom_offset: u64,
        atom_count: u64,
        group_count: u64,
        groups: &[AtomGroup],
    ) -> Vec<(usize, u64, u64)> {
        if atom_count == 0 {
            return vec![];
        }
        let first_i = atom_offset;
        let last_i = atom_offset + atom_count - 1;

        match input {
            InputRef::Broadcast(atom_id) => {
                // All atoms read the same source.
                if let Some(gi) = find_group_idx(groups, *atom_id) {
                    let off = atom_id.0 - groups[gi].base_id.0;
                    vec![(gi, off, off + 1)]
                } else {
                    vec![]
                }
            }
            InputRef::Affine { base, stride } => {
                let first_read = base.0 as i64 + *stride as i64 * first_i as i64;
                let last_read = base.0 as i64 + *stride as i64 * last_i as i64;
                let lo = first_read.min(last_read) as u64;
                let hi = first_read.max(last_read) as u64 + 1;
                // Find producer groups in this range.
                let prods = find_groups_in_range(groups, lo, hi);
                prods
                    .into_iter()
                    .map(|gi| {
                        let g = &groups[gi];
                        let g_lo = g.base_id.0;
                        let g_hi = g_lo + g.count;
                        let overlap_lo = lo.max(g_lo) - g_lo;
                        let overlap_hi = hi.min(g_hi) - g_lo;
                        (gi, overlap_lo, overlap_hi)
                    })
                    .collect()
            }
            InputRef::StridedBroadcast {
                base,
                stride,
                repeat,
            } => {
                let first_block = first_i / repeat;
                let last_block = last_i / repeat;
                let first_read = base.0 as i64 + *stride * first_block as i64;
                let last_read = base.0 as i64 + *stride * last_block as i64;
                let lo = first_read.min(last_read) as u64;
                let hi = first_read.max(last_read) as u64 + 1;
                let prods = find_groups_in_range(groups, lo, hi);
                prods
                    .into_iter()
                    .map(|gi| {
                        let g = &groups[gi];
                        let g_lo = g.base_id.0;
                        let g_hi = g_lo + g.count;
                        let overlap_lo = lo.max(g_lo) - g_lo;
                        let overlap_hi = hi.min(g_hi) - g_lo;
                        (gi, overlap_lo, overlap_hi)
                    })
                    .collect()
            }
            InputRef::Modular {
                base,
                stride,
                modulus,
            } => {
                // Modular wraps, so the full range of the modulus is accessed.
                let lo = base.0;
                let span = (*stride as i64).unsigned_abs() * (*modulus - 1);
                let hi = lo + span + 1;
                let prods = find_groups_in_range(groups, lo, hi);
                prods
                    .into_iter()
                    .map(|gi| {
                        let g = &groups[gi];
                        let g_lo = g.base_id.0;
                        let g_hi = g_lo + g.count;
                        let overlap_lo = lo.max(g_lo) - g_lo;
                        let overlap_hi = hi.min(g_hi) - g_lo;
                        (gi, overlap_lo, overlap_hi)
                    })
                    .collect()
            }
            InputRef::Explicit(ids) => {
                // Check only the atoms in our slice.
                let mut result: HashMap<usize, (u64, u64)> = HashMap::new();
                for i in first_i..=last_i {
                    if (i as usize) < ids.len() {
                        let atom_id = ids[i as usize];
                        if let Some(gi) = find_group_idx(groups, atom_id) {
                            let off = atom_id.0 - groups[gi].base_id.0;
                            let entry = result.entry(gi).or_insert((off, off + 1));
                            entry.0 = entry.0.min(off);
                            entry.1 = entry.1.max(off + 1);
                        }
                    }
                }
                result
                    .into_iter()
                    .map(|(gi, (lo, hi))| (gi, lo, hi))
                    .collect()
            }
            InputRef::SymAffine {
                base,
                stride_i,
                stride_k: _,
            } => {
                // SymAffine for sym_dim k: we don't know k at plan time.
                // Conservative: compute range for stride_i only.
                let first_read = base.0 as i64 + *stride_i as i64 * first_i as i64;
                let last_read = base.0 as i64 + *stride_i as i64 * last_i as i64;
                let lo = first_read.min(last_read) as u64;
                let hi = first_read.max(last_read) as u64 + 1;
                let prods = find_groups_in_range(groups, lo, hi);
                prods
                    .into_iter()
                    .map(|gi| {
                        let g = &groups[gi];
                        let g_lo = g.base_id.0;
                        let g_hi = g_lo + g.count;
                        let overlap_lo = lo.max(g_lo) - g_lo;
                        let overlap_hi = hi.min(g_hi) - g_lo;
                        (gi, overlap_lo, overlap_hi)
                    })
                    .collect()
            }
        }
    }

    /// Compute accessed ranges including ReduceSum/ReduceMax strided access.
    fn compute_accessed_ranges_with_reduce(
        input: &InputRef,
        atom_offset: u64,
        atom_count: u64,
        group_count: u64,
        reduce_count: u64,
        reduce_stride: i64,
        groups: &[AtomGroup],
    ) -> Vec<(usize, u64, u64)> {
        if atom_count == 0 {
            return vec![];
        }

        // For each atom i in [atom_offset..atom_offset+atom_count),
        // the reduce reads: resolved_input(i) + k*reduce_stride for k in 0..reduce_count.
        // The min/max offsets across all i and k determine the accessed range.
        match input {
            InputRef::Affine { base, stride } => {
                let first_i = atom_offset;
                let last_i = atom_offset + atom_count - 1;
                let first_base = base.0 as i64 + *stride as i64 * first_i as i64;
                let last_base = base.0 as i64 + *stride as i64 * last_i as i64;
                let min_base = first_base.min(last_base);
                let max_base = first_base.max(last_base);
                let min_stride_offset = 0i64.min(reduce_stride * (reduce_count as i64 - 1));
                let max_stride_offset = 0i64.max(reduce_stride * (reduce_count as i64 - 1));
                let lo = (min_base + min_stride_offset) as u64;
                let hi = (max_base + max_stride_offset) as u64 + 1;
                let prods = find_groups_in_range(groups, lo, hi);
                prods
                    .into_iter()
                    .map(|gi| {
                        let g = &groups[gi];
                        let g_lo = g.base_id.0;
                        let g_hi = g_lo + g.count;
                        let overlap_lo = lo.max(g_lo) - g_lo;
                        let overlap_hi = hi.min(g_hi) - g_lo;
                        (gi, overlap_lo, overlap_hi)
                    })
                    .collect()
            }
            _ => {
                // For non-Affine inputs with ReduceSum, be conservative:
                // check the full group range.
                compute_accessed_ranges(input, atom_offset, atom_count, group_count, groups)
            }
        }
    }

    /// Verify work balance across lanes within each phase.
    fn verify_balance(plan: &ExecutionPlan, threshold: f64) {
        for (phase_idx, phase) in plan.phases.iter().enumerate() {
            let lane_atoms: Vec<u64> = phase
                .lane_work
                .iter()
                .map(|lane| lane.iter().map(|w| w.atom_count).sum::<u64>())
                .collect();

            let non_zero: Vec<u64> = lane_atoms.iter().copied().filter(|&a| a > 0).collect();
            if non_zero.len() <= 1 {
                continue;
            }

            let max_work = *non_zero.iter().max().unwrap();
            let min_work = *non_zero.iter().min().unwrap();
            if min_work > 0 {
                let ratio = max_work as f64 / min_work as f64;
                assert!(
                    ratio < threshold,
                    "Phase {}: work imbalance {:.1}x (max={}, min={}) exceeds {}x threshold",
                    phase_idx,
                    ratio,
                    max_work,
                    min_work,
                    threshold
                );
            }
        }
    }

    // ─── Test cases ──────────────────────────────────────────────────────

    #[test]
    fn test_empty_graph() {
        let g = NanoGraph::new();
        let plan = plan_execution(&g, 4);
        assert_eq!(plan.num_lanes, 4);
        assert!(plan.phases.is_empty());
    }

    #[test]
    fn test_single_lane_trivial() {
        let g = build_matmul(4, 4, 4);
        let plan = plan_execution(&g, 1);
        assert_eq!(plan.num_lanes, 1);
        verify_coverage(&g, &plan);
        verify_phase_independence(&g, &plan);
    }

    #[test]
    fn test_single_matmul_2lanes() {
        let g = build_matmul(4, 8, 4);
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let plan = plan_execution(&g, 2);
        assert_eq!(plan.num_lanes, 2);
        assert!(!plan.phases.is_empty());

        verify_coverage(&g, &plan);
        verify_phase_independence(&g, &plan);

        // All rows are independent, so should be 1 phase.
        assert_eq!(plan.phases.len(), 1, "Single matmul should be 1 phase");

        println!("Single matmul 2 lanes:");
        plan.print_summary(g.groups());
    }

    #[test]
    fn test_single_matmul_8lanes() {
        let g = build_matmul(8, 4, 4);
        assert!(g.validate().is_empty());

        let plan = plan_execution(&g, 8);
        verify_coverage(&g, &plan);
        verify_phase_independence(&g, &plan);

        // 8 rows across 8 lanes = 1 row per lane, 1 phase.
        assert_eq!(plan.phases.len(), 1);

        // Perfect balance: each lane gets exactly 1 row.
        verify_balance(&plan, 1.1);

        println!("Single matmul 8 lanes:");
        plan.print_summary(g.groups());
    }

    #[test]
    fn test_matmul_chain_with_monolithic_activation() {
        // The critical test: matmul1 -> monolithic activation -> matmul2.
        // The activation group (M*N1 atoms) must be SPLIT across lanes.
        // This should produce 3 phases:
        //   Phase 0: matmul1 rows (distributed across lanes)
        //   Phase 1: activation (split across lanes) — needs barrier because
        //            it reads from all matmul1 rows
        //   Phase 2: matmul2 rows (distributed across lanes) — needs barrier
        //            because it reads from activation split across lanes
        let g = build_matmul_chain(4, 4, 4, 4, 4);
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let plan = plan_execution(&g, 2);
        verify_coverage(&g, &plan);
        verify_phase_independence(&g, &plan);

        println!("Matmul chain (monolithic activation) 2 lanes:");
        plan.print_summary(g.groups());

        // Should have multiple phases (barrier between matmul1 and activation,
        // and between activation and matmul2).
        assert!(
            plan.phases.len() >= 2,
            "Matmul chain should have >= 2 phases, got {}",
            plan.phases.len()
        );
    }

    #[test]
    fn test_matmul_chain_per_row_no_extra_barriers() {
        // When activation is per-row (not monolithic), the activation stays
        // in the same row family and NO extra barrier is needed.
        let g = build_matmul_chain_per_row(4, 4, 4, 4, 4);
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let plan = plan_execution(&g, 2);
        verify_coverage(&g, &plan);
        verify_phase_independence(&g, &plan);

        println!("Matmul chain (per-row activation) 2 lanes:");
        plan.print_summary(g.groups());

        // Per-row activation: each row's chain (mul->reduce->act->mul->reduce)
        // is independent. Should be 1 phase.
        assert_eq!(
            plan.phases.len(),
            1,
            "Per-row matmul chain should be 1 phase, got {}",
            plan.phases.len()
        );
    }

    #[test]
    fn test_parallel_matmuls_same_phase() {
        let g = build_parallel_matmuls(4, 4, 4);
        assert!(g.validate().is_empty());

        let plan = plan_execution(&g, 4);
        verify_coverage(&g, &plan);
        verify_phase_independence(&g, &plan);

        println!("Parallel matmuls 4 lanes:");
        plan.print_summary(g.groups());

        // Two independent matmuls from shared input: should be 1 phase.
        assert_eq!(plan.phases.len(), 1);
    }

    #[test]
    fn test_elementwise_single_phase_with_splitting() {
        let g = build_elementwise(1024);
        assert!(g.validate().is_empty());

        let plan = plan_execution(&g, 4);
        assert_eq!(plan.phases.len(), 1);

        verify_coverage(&g, &plan);
        verify_phase_independence(&g, &plan);

        // The elementwise group should be split across 4 lanes.
        let phase = &plan.phases[0];
        let active_lanes = phase
            .lane_work
            .iter()
            .filter(|lane| !lane.is_empty())
            .count();
        assert!(
            active_lanes >= 2,
            "Elementwise should use multiple lanes, got {}",
            active_lanes
        );

        println!("Elementwise 4 lanes:");
        plan.print_summary(g.groups());
    }

    #[test]
    fn test_balance_single_matmul() {
        let g = build_matmul(8, 4, 4);
        assert!(g.validate().is_empty());

        let plan = plan_execution(&g, 4);
        verify_coverage(&g, &plan);
        verify_phase_independence(&g, &plan);
        // Perfect balance: 8 rows / 4 lanes = 2 rows per lane.
        verify_balance(&plan, 1.1);
    }

    #[test]
    fn test_balance_monolithic_split() {
        // A monolithic group split across lanes should be well-balanced.
        let g = build_elementwise(8192);
        let plan = plan_execution(&g, 8);
        verify_coverage(&g, &plan);
        verify_phase_independence(&g, &plan);
        verify_balance(&plan, 1.1);
    }

    #[test]
    fn test_coverage_exhaustive() {
        let configs: Vec<(&str, NanoGraph, usize)> = vec![
            ("matmul_2x2_2lanes", build_matmul(2, 2, 2), 2),
            ("matmul_4x8x4_2lanes", build_matmul(4, 8, 4), 2),
            ("matmul_4x8x4_4lanes", build_matmul(4, 8, 4), 4),
            ("matmul_8x4x4_8lanes", build_matmul(8, 4, 4), 8),
            ("chain_mono_2lanes", build_matmul_chain(2, 4, 4, 4, 2), 2),
            ("chain_mono_4lanes", build_matmul_chain(4, 4, 4, 4, 4), 4),
            (
                "chain_perrow_2lanes",
                build_matmul_chain_per_row(2, 4, 4, 4, 2),
                2,
            ),
            (
                "chain_perrow_4lanes",
                build_matmul_chain_per_row(4, 4, 4, 4, 4),
                4,
            ),
            ("parallel_2lanes", build_parallel_matmuls(4, 4, 4), 2),
            ("parallel_4lanes", build_parallel_matmuls(4, 4, 4), 4),
            ("elementwise_2lanes", build_elementwise(256), 2),
            ("elementwise_4lanes", build_elementwise(1024), 4),
            ("elementwise_8lanes", build_elementwise(8192), 8),
        ];

        for (name, graph, lanes) in configs {
            assert!(graph.validate().is_empty(), "{}: validation failed", name);
            let plan = plan_execution(&graph, lanes);
            verify_coverage(&graph, &plan);
            verify_phase_independence(&graph, &plan);
            println!(
                "  {}: {} phases, {} lanes",
                name,
                plan.phases.len(),
                plan.num_lanes
            );
        }
    }

    #[test]
    fn test_matmul_chain_larger() {
        // Larger matmul chain: M=8, simulating more realistic sizes.
        let g = build_matmul_chain(8, 16, 16, 16, 16);
        assert!(g.validate().is_empty());

        let plan = plan_execution(&g, 4);
        verify_coverage(&g, &plan);
        verify_phase_independence(&g, &plan);

        println!("Larger matmul chain 4 lanes:");
        plan.print_summary(g.groups());

        // Should have reasonable number of phases (not hundreds).
        assert!(
            plan.phases.len() <= 10,
            "Should have <= 10 phases, got {}",
            plan.phases.len()
        );
    }

    #[test]
    fn test_three_matmul_chain() {
        // Three matmuls in sequence with monolithic elementwise between each.
        // Key: push all Muls first, then all Reds, so ReduceSum outputs
        // are contiguous in AtomId space (required for Affine input to
        // the monolithic activation).
        let mut g = NanoGraph::new();
        let m = 4u64;
        let k = 4u64;
        let n = 4u64;

        // Matmul 1: push all Muls, then all Reds.
        let a1 = g.push_group(
            m * k,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let b1 = g.push_group(
            k * n,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        let mut mul1_bases = Vec::new();
        for row in 0..m {
            let mul = g.push_group(
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
                        base: AtomId(a1.0 + row * k),
                        stride: 1,
                        repeat: n,
                    },
                    InputRef::Affine {
                        base: b1,
                        stride: 1,
                    },
                ],
            );
            mul1_bases.push(mul);
        }
        let mut prev_reduce_bases = Vec::new();
        for row in 0..m {
            let red = g.push_group(
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
                    base: mul1_bases[row as usize],
                    stride: 1,
                }],
            );
            prev_reduce_bases.push(red);
        }

        // Two more matmuls with monolithic elementwise between.
        for _matmul_idx in 0..2 {
            // Monolithic elementwise (reads all M ReduceSum outputs contiguously).
            let act = g.push_group(
                m * n,
                ScalarOp::Unary {
                    op: ScalarUnaryOp::Tanh,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![InputRef::Affine {
                    base: prev_reduce_bases[0],
                    stride: 1,
                }],
            );

            let b = g.push_group(
                n * n,
                ScalarOp::Literal(NumericScalar::F32(1.0)),
                vec![],
                vec![],
                vec![],
            );

            // Push all Muls, then all Reds.
            let mut mul_bases = Vec::new();
            for row in 0..m {
                let mul = g.push_group(
                    n * n,
                    ScalarOp::Binary {
                        op: ScalarBinOp::Mul,
                        compute_dtype: DType::F32,
                        output_dtype: DType::F32,
                    },
                    vec![],
                    vec![],
                    vec![
                        InputRef::StridedBroadcast {
                            base: AtomId(act.0 + row * n),
                            stride: 1,
                            repeat: n,
                        },
                        InputRef::Affine { base: b, stride: 1 },
                    ],
                );
                mul_bases.push(mul);
            }
            let mut new_reduce_bases = Vec::new();
            for row in 0..m {
                let red = g.push_group(
                    n,
                    ScalarOp::ReduceSum {
                        reduce_count: n,
                        reduce_stride: n as i64,
                        compute_dtype: DType::F32,
                        output_dtype: DType::F32,
                    },
                    vec![],
                    vec![],
                    vec![InputRef::Affine {
                        base: mul_bases[row as usize],
                        stride: 1,
                    }],
                );
                new_reduce_bases.push(red);
            }
            prev_reduce_bases = new_reduce_bases;
        }

        for &rb in &prev_reduce_bases {
            for i in 0..n {
                g.outputs.push(AtomId(rb.0 + i));
            }
        }

        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let plan = plan_execution(&g, 4);
        verify_coverage(&g, &plan);
        verify_phase_independence(&g, &plan);

        println!("Three matmul chain 4 lanes:");
        plan.print_summary(g.groups());

        // 3 matmuls with 2 monolithic elementwise between them:
        // Each elementwise creates a barrier on each side = 5 potential phases.
        assert!(
            plan.phases.len() >= 3,
            "Three matmul chain should have >= 3 phases"
        );
        assert!(
            plan.phases.len() <= 10,
            "Should have <= 10 phases, got {}",
            plan.phases.len()
        );
    }

    #[test]
    fn test_realistic_matmul_dimensions() {
        // GPT-2-like dimensions: M=64, K=32, N=32 (scaled down for test speed).
        let g = build_matmul(64, 32, 32);
        assert!(g.validate().is_empty());

        let plan = plan_execution(&g, 8);
        verify_coverage(&g, &plan);
        verify_phase_independence(&g, &plan);

        // 64 rows / 8 lanes = 8 rows per lane, perfect balance.
        assert_eq!(plan.phases.len(), 1, "Single matmul should be 1 phase");
        verify_balance(&plan, 1.1);
    }

    #[test]
    fn test_realistic_matmul_chain() {
        // Two matmuls with monolithic activation between them.
        // GPT-2-like: M=16, K=32, N=32.
        let g = build_matmul_chain(16, 32, 32, 32, 32);
        assert!(g.validate().is_empty());

        let plan = plan_execution(&g, 8);
        verify_coverage(&g, &plan);
        verify_phase_independence(&g, &plan);

        // Should have 3 phases: matmul1, activation, matmul2.
        assert!(
            plan.phases.len() >= 2,
            "Should have >= 2 phases, got {}",
            plan.phases.len()
        );
        assert!(
            plan.phases.len() <= 5,
            "Should have <= 5 phases, got {}",
            plan.phases.len()
        );
    }

    #[test]
    fn test_single_group_various_sizes() {
        // Monolithic groups of various sizes should split cleanly.
        for count in [1, 7, 8, 15, 16, 100, 1000] {
            let g = build_elementwise(count);
            let plan = plan_execution(&g, 4);
            verify_coverage(&g, &plan);
            verify_phase_independence(&g, &plan);
        }
    }

    #[test]
    fn test_uneven_row_count() {
        // 7 rows across 4 lanes: some lanes get 2 rows, some get 1.
        let g = build_matmul(7, 4, 4);
        assert!(g.validate().is_empty());

        let plan = plan_execution(&g, 4);
        verify_coverage(&g, &plan);
        verify_phase_independence(&g, &plan);

        // With 7 rows and 4 lanes: max 2 rows/lane, min 1 row/lane = 2x ratio.
        verify_balance(&plan, 2.5);
    }

    /// Build an AllRows→AllRows chain: A → B → C, all same count,
    /// connected by Affine{stride=1}. Should be 1 phase (no barriers).
    fn build_allrows_chain(count: u64) -> NanoGraph {
        let mut g = NanoGraph::new();
        let lit = g.push_group(
            count,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let a = g.push_group(
            count,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Tanh,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: lit,
                stride: 1,
            }],
        );
        let b = g.push_group(
            count,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Tanh,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: a, stride: 1 }],
        );
        let c = g.push_group(
            count,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Tanh,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: b, stride: 1 }],
        );
        for i in 0..count {
            g.outputs.push(AtomId(c.0 + i));
        }
        g
    }

    #[test]
    fn test_allrows_chain_single_phase() {
        // Three AllRows groups in a chain, all same count.
        // Should be 1 phase: each lane's slice is independent.
        let g = build_allrows_chain(1024);
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let plan = plan_execution(&g, 8);
        verify_coverage(&g, &plan);
        verify_phase_independence(&g, &plan);

        assert_eq!(
            plan.phases.len(),
            1,
            "AllRows chain should be 1 phase, got {}",
            plan.phases.len()
        );
        verify_balance(&plan, 1.1);

        println!("AllRows chain 8 lanes:");
        plan.print_summary(g.groups());
    }

    #[test]
    fn test_allrows_chain_with_broadcast() {
        // AllRows chain where one group reads via Broadcast from
        // a previous AllRows group. This needs a barrier.
        let mut g = NanoGraph::new();
        let count = 1024u64;
        let lit = g.push_group(
            count,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let a = g.push_group(
            count,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Tanh,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: lit,
                stride: 1,
            }],
        );
        // Broadcast from first element of a (which is AllRows) → needs barrier
        let b = g.push_group(
            count,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine { base: a, stride: 1 },
                InputRef::Broadcast(a), // Broadcast from AllRows
            ],
        );
        for i in 0..count {
            g.outputs.push(AtomId(b.0 + i));
        }
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let plan = plan_execution(&g, 4);
        verify_coverage(&g, &plan);
        verify_phase_independence(&g, &plan);

        // Should have >= 2 phases because of the Broadcast cross-lane read
        assert!(
            plan.phases.len() >= 2,
            "AllRows chain with Broadcast should have >= 2 phases, got {}",
            plan.phases.len()
        );
    }

    #[test]
    fn test_allrows_chain_with_earlier_phase_broadcast() {
        // AllRows group C reads from:
        //   1. AllRows group B via Affine{stride=1} (same phase, lane-local)
        //   2. AllRows group A via Broadcast (A is in EARLIER phase due to
        //      a barrier between A and B)
        // Since A is in an earlier phase, the Broadcast is fine — no new barrier.
        let mut g = NanoGraph::new();
        let count = 1024u64;
        let small_count = 10u64;

        // Literal for count
        let lit = g.push_group(
            count,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        // A: AllRows group (count) → phase 0
        let a = g.push_group(
            count,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Tanh,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: lit,
                stride: 1,
            }],
        );

        // mean: AllRows ReduceSum (small_count, reduce_count=count/small_count)
        // This creates a barrier (AllRows→AllRows with different count)
        let mean = g.push_group(
            small_count,
            ScalarOp::ReduceSum {
                reduce_count: count / small_count,
                reduce_stride: small_count as i64,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: a, stride: 1 }],
        );

        // B: AllRows binary that reads from mean via Broadcast
        // This creates another barrier (Broadcast from AllRows)
        let b = g.push_group(
            count,
            ScalarOp::Binary {
                op: ScalarBinOp::Sub,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine { base: a, stride: 1 },
                InputRef::Broadcast(mean),
            ],
        );

        // C: AllRows unary reading from B via Affine{stride=1}
        // B is in same phase, lane-local → NO barrier
        let c = g.push_group(
            count,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Tanh,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: b, stride: 1 }],
        );

        for i in 0..count {
            g.outputs.push(AtomId(c.0 + i));
        }
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let plan = plan_execution(&g, 4);
        verify_coverage(&g, &plan);
        verify_phase_independence(&g, &plan);

        println!("AllRows chain with earlier-phase broadcast, 4 lanes:");
        plan.print_summary(g.groups());

        // B and C should be in the same phase (no barrier between them).
        // Total: at least 3 phases: A, mean, B+C
        assert!(
            plan.phases.len() >= 3,
            "Should have >= 3 phases, got {}",
            plan.phases.len()
        );
        // But B and C should be fused, so not more than ~4 phases
        assert!(
            plan.phases.len() <= 4,
            "Should have <= 4 phases (B and C fused), got {}",
            plan.phases.len()
        );
    }

    #[test]
    fn test_more_lanes_than_rows() {
        // 2 rows across 8 lanes: only 2 lanes should be active.
        let g = build_matmul(2, 4, 4);
        let plan = plan_execution(&g, 8);
        verify_coverage(&g, &plan);
        verify_phase_independence(&g, &plan);

        // Should still work correctly, with most lanes idle.
        let phase = &plan.phases[0];
        let active_lanes = phase
            .lane_work
            .iter()
            .filter(|lane| !lane.is_empty())
            .count();
        assert!(
            active_lanes <= 2,
            "Should have at most 2 active lanes, got {}",
            active_lanes
        );
    }

    /// Test that ReduceSum strided access is properly tracked for dependencies.
    ///
    /// Constructs a graph where a ReduceSum's InputRef points to group A, but its
    /// strided access reads atoms from group B (which is in a later phase due to
    /// a barrier). The planner MUST recognize B as a dependency of the ReduceSum
    /// and place it in a phase >= B's phase.
    ///
    /// Graph structure:
    ///   lit_a (10 atoms) → row_mul_0, row_mul_1 (each K*N atoms, Row families)
    ///                     → row_red_0, row_red_1 (each N atoms, Row families)
    ///   lit_b (10 atoms)  ↗
    ///   The reduces feed into a monolithic elementwise group (AllRows, phase 1).
    ///   Then we have a second monolithic group that depends on the first.
    ///   Finally, a ReduceSum whose InputRef base points into the first monolithic
    ///   group but whose stride extends into the second — the missed dependency.
    #[test]
    fn test_reducesum_cross_group_stride_dependency() {
        let mut g = NanoGraph::new();

        // Phase 0: two Row families from a simple "matmul"
        let lit_a = g.push_group(
            8,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let lit_b = g.push_group(
            8,
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
            vec![],
        );

        // Two "rows" of Mul (same literal signature → Row families)
        let mul0 = g.push_group(
            8,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::StridedBroadcast {
                    base: AtomId(lit_a.0),
                    stride: 1,
                    repeat: 4,
                },
                InputRef::Affine {
                    base: lit_b,
                    stride: 1,
                },
            ],
        );
        let mul1 = g.push_group(
            8,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::StridedBroadcast {
                    base: AtomId(lit_a.0 + 2),
                    stride: 1,
                    repeat: 4,
                },
                InputRef::Affine {
                    base: lit_b,
                    stride: 1,
                },
            ],
        );

        // Two Row ReduceSums
        let red0 = g.push_group(
            4,
            ScalarOp::ReduceSum {
                reduce_count: 2,
                reduce_stride: 4,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: mul0,
                stride: 1,
            }],
        );
        let red1 = g.push_group(
            4,
            ScalarOp::ReduceSum {
                reduce_count: 2,
                reduce_stride: 4,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: mul1,
                stride: 1,
            }],
        );

        // Phase 1: monolithic elementwise that reads from BOTH row reduces (AllRows).
        // This creates a barrier because it reads from multiple Row families.
        // Note: red0 and red1 are contiguous in AtomId space.
        let mono_a = g.push_group(
            8, // 4 from red0 + 4 from red1
            ScalarOp::Unary {
                op: ScalarUnaryOp::Tanh,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: red0,
                stride: 1,
            }],
        );

        // mono_b: depends on mono_a via Broadcast (reads ONE atom from mono_a).
        // Broadcast from AllRows → needs barrier → mono_b is in phase 2 (or later).
        let mono_b = g.push_group(
            8,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine {
                    base: mono_a,
                    stride: 1,
                },
                InputRef::Broadcast(mono_a), // Broadcast from AllRows → barrier
            ],
        );

        // The critical group: ReduceSum whose InputRef base is mono_a,
        // but reduce_stride=8 extends the access into mono_b.
        // atom i reads: mono_a+i, mono_a+i+8
        // mono_a+i is in mono_a (atoms 0-7)
        // mono_a+i+8 is in mono_b (atoms 0-7, since mono_b.base = mono_a.base + 8)
        //
        // BUG: resolve_producer_groups only finds mono_a (range [mono_a, mono_a+3]),
        // so max_prod_phase = phase_of_mono_a. But mono_b is in a LATER phase,
        // and the ReduceSum actually reads from it.
        let reduce_cross = g.push_group(
            4,
            ScalarOp::ReduceSum {
                reduce_count: 2,
                reduce_stride: 8,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: mono_a,
                stride: 1,
            }],
        );

        for i in 0..4 {
            g.outputs.push(AtomId(reduce_cross.0 + i));
        }

        assert!(g.validate().is_empty(), "{:?}", g.validate());

        // This should NOT panic from validate_phase_ordering.
        // If the ReduceSum's strided dependency on mono_b is missed,
        // reduce_cross could be placed in the same phase as mono_a
        // (because its only known producer via InputRef is mono_a),
        // but mono_b is also in that phase and has higher group_idx,
        // so within-phase ordering saves us. The validation should still pass.
        //
        // The REAL test is: does the validation function catch the dependency?
        let plan = plan_execution(&g, 1);
        verify_coverage(&g, &plan);

        // Extract phase assignments from plan
        let groups = g.groups();
        let mut group_phase_from_plan: HashMap<usize, usize> = HashMap::new();
        for (phase_idx, phase) in plan.phases.iter().enumerate() {
            for lane in &phase.lane_work {
                for w in lane {
                    group_phase_from_plan.insert(w.group_idx, phase_idx);
                }
            }
        }

        let mono_b_gi = groups.iter().position(|grp| grp.base_id == mono_b).unwrap();
        let reduce_gi = groups
            .iter()
            .position(|grp| grp.base_id == reduce_cross)
            .unwrap();

        let mono_b_phase = group_phase_from_plan[&mono_b_gi];
        let reduce_phase = group_phase_from_plan[&reduce_gi];

        // reduce_cross MUST be in a phase >= mono_b's phase, since the
        // ReduceSum's strided access reads from mono_b.
        assert!(
            reduce_phase >= mono_b_phase,
            "ReduceSum group (gi={}, phase={}) reads atoms from mono_b (gi={}, phase={}) \
             via strided reduce access, but is placed in an earlier phase!",
            reduce_gi,
            reduce_phase,
            mono_b_gi,
            mono_b_phase,
        );
    }
}
