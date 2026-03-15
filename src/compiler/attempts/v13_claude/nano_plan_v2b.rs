#![allow(clippy::all, dead_code, unreachable_patterns, unused_variables, unused_imports)]
//! Lane+barrier execution planner v2b: lanes first, barriers second.
//!
//! Previous attempts placed barriers first, then tried to distribute monolithic
//! groups across lanes. This failed because:
//! - Barriers at every ReduceSum created too many tiny phases
//! - Monolithic groups (49,152 atoms) couldn't be split across lanes
//!
//! This approach reverses the order:
//! 1. Build the group DAG
//! 2. Compute a "row dimension" for each group (how many independent rows)
//! 3. Assign row ranges to lanes (consistent slices across all groups)
//! 4. Split groups into per-lane sub-ranges
//! 5. Detect barriers by checking cross-lane reads
//! 6. Collect phases from barrier positions
//!
//! The key insight: if lanes have consistent row slices, most elementwise ops
//! are lane-local. Barriers only appear where a downstream op genuinely needs
//! ALL lanes' results (e.g., a matmul that reads the full output vector from
//! a previous matmul, requiring all rows to be complete).

use std::collections::{HashMap, HashSet, VecDeque};

use crate::nano_graph::{AtomGroup, AtomId, InputRef, NanoGraph, ScalarOp};

// ─── Public API ──────────────────────────────────────────────────────────────

/// A unit of work assigned to a lane within a phase.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaneWork {
    /// Index of the group in the NanoGraph this work comes from.
    pub group_idx: usize,
    /// Offset within the group (first atom = base_id + atom_offset).
    pub atom_offset: u64,
    /// Number of atoms this lane handles from this group.
    pub atom_count: u64,
}

/// The execution plan: lanes and barrier-separated phases.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Number of lanes (persistent threads).
    pub num_lanes: usize,
    /// Phases separated by barriers. Each phase contains per-lane work.
    pub phases: Vec<Phase>,
}

/// One phase of execution (work between two consecutive barriers).
#[derive(Debug, Clone)]
pub struct Phase {
    /// For each lane index, the work items assigned to that lane in this phase.
    /// `lane_work[lane_idx]` = list of LaneWork items.
    pub lane_work: Vec<Vec<LaneWork>>,
}

/// Plan execution for a NanoGraph.
///
/// Assigns rows to lanes first, then detects where barriers are needed based
/// on cross-lane data dependencies.
pub fn plan_execution(graph: &NanoGraph, num_lanes: usize) -> ExecutionPlan {
    let groups = graph.groups();
    let num_groups = groups.len();

    if num_groups == 0 || num_lanes == 0 {
        return ExecutionPlan {
            num_lanes: num_lanes.max(1),
            phases: vec![],
        };
    }

    // Step 1: Build group-level dependency DAG.
    let (predecessors, successors) = build_group_dag(graph);

    // Step 2: Topological sort.
    let topo_order = topo_sort(num_groups, &predecessors, &successors);

    // Step 3: Classify groups and compute row structure.
    let group_info = compute_group_info(graph, &predecessors);

    // Step 4: Compute per-lane assignments (which atoms each lane owns).
    let lane_assignments = assign_lanes(graph, &group_info, &topo_order, &predecessors, num_lanes);

    // Step 5: Detect barriers by checking cross-lane reads.
    let phase_ids = detect_barriers(
        graph,
        &lane_assignments,
        &topo_order,
        &predecessors,
        &group_info,
    );

    // Step 6: Collect phases.
    let num_phases = phase_ids.iter().copied().max().map(|m| m + 1).unwrap_or(1);
    let mut phases: Vec<Phase> = (0..num_phases)
        .map(|_| Phase {
            lane_work: vec![vec![]; num_lanes],
        })
        .collect();

    for &gi in &topo_order {
        let phase = phase_ids[gi];
        for lane in 0..num_lanes {
            let la = &lane_assignments[gi];
            if lane < la.lane_ranges.len() {
                let (offset, count) = la.lane_ranges[lane];
                if count > 0 {
                    phases[phase].lane_work[lane].push(LaneWork {
                        group_idx: gi,
                        atom_offset: offset,
                        atom_count: count,
                    });
                }
            }
        }
    }

    ExecutionPlan { num_lanes, phases }
}

// ─── Internal types ──────────────────────────────────────────────────────────

/// Classification of a group for lane assignment purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GroupKind {
    /// Literal/weight data — shared by all lanes, never assigned.
    Literal,
    /// Matmul Mul group — already one row, assign whole group to one lane.
    MatmulMul,
    /// Matmul ReduceSum — already one row, assign whole group to one lane.
    MatmulReduce,
    /// Elementwise/unary/other compute — can be split across lanes.
    Elementwise,
    /// Non-matmul reduce (e.g. LayerNorm mean/variance) — needs analysis.
    OtherReduce,
    /// IndirectLoad — typically not splittable, assign to one lane.
    IndirectLoad,
}

/// Row structure information for a group.
#[derive(Debug, Clone)]
struct GroupInfo {
    kind: GroupKind,
    /// Number of independent rows in this group.
    num_rows: u64,
    /// Number of atoms per row.
    atoms_per_row: u64,
    /// Which "row family" this group belongs to (groups in the same family
    /// should get consistent lane assignments). Identified by the base_id of
    /// the first matmul ReduceSum group in the family.
    row_family: Option<u64>,
    /// For matmul Mul/ReduceSum groups: which row index (0..M-1) within the matmul.
    row_index: Option<u64>,
}

/// Per-group lane assignment: which atoms go to which lane.
#[derive(Debug, Clone)]
struct LaneAssignment {
    /// For each lane, (atom_offset, atom_count) within this group.
    lane_ranges: Vec<(u64, u64)>,
}

// ─── Group DAG construction ──────────────────────────────────────────────────

/// Build predecessor and successor lists for the group DAG.
fn build_group_dag(graph: &NanoGraph) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let groups = graph.groups();
    let num_groups = groups.len();

    let mut predecessors: Vec<Vec<usize>> = vec![vec![]; num_groups];
    let mut successors: Vec<Vec<usize>> = vec![vec![]; num_groups];

    for (gi, group) in groups.iter().enumerate() {
        let mut pred_set = HashSet::new();
        for input in &group.inputs {
            let source_ids = sample_source_atoms(input, group.count);
            for src_id in source_ids {
                if let Some(src_gi) = find_group_index(groups, src_id) {
                    if src_gi != gi {
                        pred_set.insert(src_gi);
                    }
                }
            }
        }
        let preds: Vec<usize> = pred_set.into_iter().collect();
        for &p in &preds {
            successors[p].push(gi);
        }
        predecessors[gi] = preds;
    }

    for s in &mut successors {
        s.sort();
        s.dedup();
    }

    (predecessors, successors)
}

/// Sample source atoms from an InputRef to determine which groups it references.
fn sample_source_atoms(input: &InputRef, count: u64) -> Vec<AtomId> {
    match input {
        InputRef::Broadcast(id) => vec![*id],
        InputRef::Affine { base, stride } => {
            let mut ids = vec![*base];
            if count > 1 {
                ids.push(AtomId(
                    base.0
                        .wrapping_add((*stride as i64 * (count as i64 - 1)) as u64),
                ));
            }
            if count > 2 {
                let mid = count / 2;
                ids.push(AtomId(
                    base.0
                        .wrapping_add((*stride as i64 * mid as i64) as u64),
                ));
            }
            ids
        }
        InputRef::Explicit(ids) => {
            let mut sampled = vec![ids[0]];
            if ids.len() > 1 {
                sampled.push(*ids.last().unwrap());
            }
            if ids.len() > 2 {
                sampled.push(ids[ids.len() / 2]);
            }
            sampled
        }
        InputRef::SymAffine { base, .. } => vec![*base],
        InputRef::StridedBroadcast {
            base,
            stride,
            repeat,
        } => {
            let mut ids = vec![*base];
            if count > *repeat {
                let last_block = (count - 1) / repeat;
                ids.push(AtomId(
                    base.0.wrapping_add((*stride * last_block as i64) as u64),
                ));
            }
            ids
        }
        InputRef::Modular {
            base,
            stride,
            modulus,
        } => {
            let mut ids = vec![*base];
            if *modulus > 1 {
                ids.push(AtomId(
                    base.0
                        .wrapping_add((*stride as i64 * (*modulus as i64 - 1)) as u64),
                ));
            }
            ids
        }
    }
}

/// Find the group index containing a given AtomId via binary search.
fn find_group_index(groups: &[AtomGroup], id: AtomId) -> Option<usize> {
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

/// Topological sort via Kahn's algorithm.
fn topo_sort(
    num_groups: usize,
    predecessors: &[Vec<usize>],
    successors: &[Vec<usize>],
) -> Vec<usize> {
    let mut in_degree: Vec<usize> = predecessors.iter().map(|p| p.len()).collect();
    let mut queue: VecDeque<usize> = VecDeque::new();
    for gi in 0..num_groups {
        if in_degree[gi] == 0 {
            queue.push_back(gi);
        }
    }

    let mut order = Vec::with_capacity(num_groups);
    while let Some(gi) = queue.pop_front() {
        order.push(gi);
        for &succ in &successors[gi] {
            in_degree[succ] -= 1;
            if in_degree[succ] == 0 {
                queue.push_back(succ);
            }
        }
    }

    order
}

// ─── Group classification ────────────────────────────────────────────────────

/// Classify each group and compute its row structure.
fn compute_group_info(graph: &NanoGraph, predecessors: &[Vec<usize>]) -> Vec<GroupInfo> {
    let groups = graph.groups();
    let num_groups = groups.len();
    let mut infos: Vec<GroupInfo> = Vec::with_capacity(num_groups);

    // First pass: classify each group.
    for (gi, group) in groups.iter().enumerate() {
        let info = classify_group(gi, group, groups, predecessors);
        infos.push(info);
    }

    // Second pass: propagate row families from matmul reduces to their consumers.
    // Groups that read exclusively from groups with a common row family inherit it.
    propagate_row_families(&mut infos, groups, predecessors);

    infos
}

/// Classify a single group.
fn classify_group(
    gi: usize,
    group: &AtomGroup,
    groups: &[AtomGroup],
    predecessors: &[Vec<usize>],
) -> GroupInfo {
    match &group.op {
        ScalarOp::Literal(_) => GroupInfo {
            kind: GroupKind::Literal,
            num_rows: 1,
            atoms_per_row: group.count,
            row_family: None,
            row_index: None,
        },
        ScalarOp::IndirectLoad { .. } => GroupInfo {
            kind: GroupKind::IndirectLoad,
            num_rows: 1,
            atoms_per_row: group.count,
            row_family: None,
            row_index: None,
        },
        ScalarOp::Binary { op, .. } if *op == crate::nano_graph::ScalarBinOp::Mul => {
            // Check if this is a matmul Mul group (has StridedBroadcast input).
            let has_strided_broadcast = group
                .inputs
                .iter()
                .any(|inp| matches!(inp, InputRef::StridedBroadcast { .. }));
            if has_strided_broadcast {
                GroupInfo {
                    kind: GroupKind::MatmulMul,
                    num_rows: 1, // Each merged Mul group IS one row
                    atoms_per_row: group.count,
                    row_family: None, // Will be linked via ReduceSum
                    row_index: None,  // Will be set when we find siblings
                }
            } else {
                // Regular elementwise Mul
                GroupInfo {
                    kind: GroupKind::Elementwise,
                    num_rows: 1,
                    atoms_per_row: group.count,
                    row_family: None,
                    row_index: None,
                }
            }
        }
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
            // Check if this reduce reads from a matmul Mul group.
            let reads_matmul_mul = predecessors[gi].iter().any(|&pred_gi| {
                matches!(groups[pred_gi].op, ScalarOp::Binary { op: crate::nano_graph::ScalarBinOp::Mul, .. })
                    && groups[pred_gi]
                        .inputs
                        .iter()
                        .any(|inp| matches!(inp, InputRef::StridedBroadcast { .. }))
            });

            if reads_matmul_mul {
                GroupInfo {
                    kind: GroupKind::MatmulReduce,
                    num_rows: 1, // Each reduce group IS one row
                    atoms_per_row: group.count,
                    row_family: Some(group.base_id.0), // Self-identify
                    row_index: None,
                }
            } else {
                GroupInfo {
                    kind: GroupKind::OtherReduce,
                    num_rows: 1,
                    atoms_per_row: group.count,
                    row_family: None,
                    row_index: None,
                }
            }
        }
        _ => {
            // Elementwise, unary, identity, select, etc.
            GroupInfo {
                kind: GroupKind::Elementwise,
                num_rows: 1,
                atoms_per_row: group.count,
                row_family: None,
                row_index: None,
            }
        }
    }
}

/// Identify matmul row families: groups of M ReduceSum groups that are siblings
/// (produced by M Mul groups reading from the same weight matrix).
///
/// Then propagate row structure to elementwise consumers.
fn propagate_row_families(
    infos: &mut [GroupInfo],
    groups: &[AtomGroup],
    predecessors: &[Vec<usize>],
) {
    let num_groups = infos.len();

    // Find matmul families: sets of matmul ReduceSum groups that share the same
    // weight matrix (their Mul predecessors share the same Affine input source).
    //
    // For a matmul C[M,N] = A[M,K] @ B[K,N]:
    // - M Mul groups, each with StridedBroadcast(A[m,*]) + Affine(B)
    // - M ReduceSum groups, each reading from one Mul group
    // The Mul groups all share the same Affine base (B[0,0]).
    // So ReduceSum groups whose Mul predecessors share the same Affine base are siblings.

    // Map: weight_base_id -> list of (reduce_group_idx, mul_group_idx)
    let mut weight_families: HashMap<u64, Vec<(usize, usize)>> = HashMap::new();

    for gi in 0..num_groups {
        if infos[gi].kind != GroupKind::MatmulReduce {
            continue;
        }

        // Find the Mul predecessor
        for &pred_gi in &predecessors[gi] {
            if infos[pred_gi].kind != GroupKind::MatmulMul {
                continue;
            }

            // Find the Affine input on the Mul group (the weight matrix).
            // In a matmul Mul, one input is StridedBroadcast (the activation row)
            // and the other is Affine (the weight matrix).
            for inp in &groups[pred_gi].inputs {
                if let InputRef::Affine { base, stride } = inp {
                    if *stride == 1 {
                        weight_families
                            .entry(base.0)
                            .or_default()
                            .push((gi, pred_gi));
                    }
                }
            }
        }
    }

    // For each family, assign consistent row indices and a shared family id.
    for (_weight_base, members) in &weight_families {
        if members.len() <= 1 {
            continue;
        }

        // Sort by group index to get consistent ordering.
        let mut sorted: Vec<(usize, usize)> = members.clone();
        sorted.sort_by_key(|&(reduce_gi, _)| reduce_gi);

        // Use the first ReduceSum's base_id as the family identifier.
        let family_id = groups[sorted[0].0].base_id.0;

        for (row_idx, &(reduce_gi, mul_gi)) in sorted.iter().enumerate() {
            infos[reduce_gi].row_family = Some(family_id);
            infos[reduce_gi].row_index = Some(row_idx as u64);
            infos[mul_gi].row_family = Some(family_id);
            infos[mul_gi].row_index = Some(row_idx as u64);
        }
    }

    // Now propagate row structure to elementwise consumers.
    // If all non-Literal predecessors of an elementwise group belong to the same
    // row family AND the group's count equals the number of family members times
    // the per-row atom count, then this group can be split across rows.
    //
    // We need multiple passes because elementwise ops chain: reduce -> add -> sub -> ...
    // and each layer inherits from its predecessors.

    // First, build a map: family_id -> (num_rows, atoms_per_row)
    let mut family_shape: HashMap<u64, (u64, u64)> = HashMap::new();
    for (_weight_base, members) in &weight_families {
        if members.is_empty() {
            continue;
        }
        let mut sorted: Vec<(usize, usize)> = members.clone();
        sorted.sort_by_key(|&(reduce_gi, _)| reduce_gi);
        let family_id = groups[sorted[0].0].base_id.0;
        let m = sorted.len() as u64;
        let n = groups[sorted[0].0].count;
        family_shape.insert(family_id, (m, n));
    }

    // Propagate: iterate until stable.
    let mut changed = true;
    while changed {
        changed = false;
        for gi in 0..num_groups {
            if infos[gi].kind != GroupKind::Elementwise
                && infos[gi].kind != GroupKind::OtherReduce
            {
                continue;
            }
            if infos[gi].row_family.is_some() {
                continue; // Already assigned
            }

            // Find non-Literal predecessors with row families.
            let non_lit_preds: Vec<usize> = predecessors[gi]
                .iter()
                .copied()
                .filter(|&p| infos[p].kind != GroupKind::Literal)
                .collect();

            if non_lit_preds.is_empty() {
                continue;
            }

            // Check if all non-Literal predecessors share the same row family.
            let families: HashSet<Option<u64>> = non_lit_preds
                .iter()
                .map(|&p| infos[p].row_family)
                .collect();

            // Remove None — predecessors without a family.
            let known_families: HashSet<u64> = families.iter().filter_map(|f| *f).collect();

            if known_families.len() != 1 {
                continue; // Multiple families or no family — can't propagate
            }

            let family_id = *known_families.iter().next().unwrap();
            let (m, n) = match family_shape.get(&family_id) {
                Some(shape) => *shape,
                None => continue,
            };

            // Check if this group's count is compatible with the family.
            // An elementwise group over the full matmul output has count = M*N.
            // It should split into M rows of N atoms each.
            let count = groups[gi].count;
            if count == m * n {
                infos[gi].row_family = Some(family_id);
                infos[gi].num_rows = m;
                infos[gi].atoms_per_row = n;
                changed = true;
            } else if count == n {
                // Same size as one row — this is already row-level.
                // Check: does this group read from exactly one row of the family?
                // If so, it inherits that row's assignment.
                // For now, mark it as same family, single row.
                infos[gi].row_family = Some(family_id);
                infos[gi].num_rows = 1;
                infos[gi].atoms_per_row = n;
                // Try to inherit row_index from a predecessor.
                for &p in &non_lit_preds {
                    if infos[p].row_family == Some(family_id) && infos[p].row_index.is_some() {
                        infos[gi].row_index = infos[p].row_index;
                        break;
                    }
                }
                changed = true;
            }
            // If count doesn't match, we can't split — leave it unassigned.
        }
    }
}

// ─── Lane assignment ─────────────────────────────────────────────────────────

/// Assign atoms to lanes for each group based on row structure.
fn assign_lanes(
    graph: &NanoGraph,
    group_info: &[GroupInfo],
    topo_order: &[usize],
    predecessors: &[Vec<usize>],
    num_lanes: usize,
) -> Vec<LaneAssignment> {
    let groups = graph.groups();
    let num_groups = groups.len();

    // Build row-to-lane mapping for each family.
    // For family with M rows, assign rows to lanes evenly.
    let mut family_row_lanes: HashMap<u64, Vec<usize>> = HashMap::new();

    // Collect all families and their row counts.
    let mut family_rows: HashMap<u64, u64> = HashMap::new();
    for info in group_info.iter() {
        if let Some(fam) = info.row_family {
            if info.num_rows > 1 {
                family_rows.entry(fam).or_insert(info.num_rows);
            }
            // For single-row groups in the family, find the family's M from siblings
        }
    }

    // Also get M from matmul groups that have row_index set.
    for info in group_info.iter() {
        if let (Some(fam), Some(row_idx)) = (info.row_family, info.row_index) {
            let entry = family_rows.entry(fam).or_insert(0);
            *entry = (*entry).max(row_idx + 1);
        }
    }

    // Build row-to-lane maps.
    for (&fam, &m) in &family_rows {
        let mut row_lanes = Vec::with_capacity(m as usize);
        for row in 0..m {
            let lane = (row as usize * num_lanes) / m as usize;
            row_lanes.push(lane.min(num_lanes - 1));
        }
        family_row_lanes.insert(fam, row_lanes);
    }

    // Now assign each group.
    let mut assignments: Vec<LaneAssignment> = Vec::with_capacity(num_groups);

    for (gi, group) in groups.iter().enumerate() {
        let info = &group_info[gi];
        let la = match info.kind {
            GroupKind::Literal => {
                // Literals are shared — all lanes get the full range.
                // But we mark them as lane 0 with full count since they're
                // not actually executed, just read.
                let ranges = vec![(0u64, 0u64); num_lanes];
                // Literals aren't assigned to any lane for execution.
                LaneAssignment { lane_ranges: ranges }
            }
            GroupKind::MatmulMul | GroupKind::MatmulReduce => {
                // Single-row matmul group — assign to the lane owning this row.
                let mut ranges = vec![(0u64, 0u64); num_lanes];
                if let (Some(fam), Some(row_idx)) = (info.row_family, info.row_index) {
                    if let Some(row_lanes) = family_row_lanes.get(&fam) {
                        let lane = row_lanes[row_idx as usize];
                        ranges[lane] = (0, group.count);
                    } else {
                        // No family map — assign to lane based on row_index.
                        let lane = (row_idx as usize) % num_lanes;
                        ranges[lane] = (0, group.count);
                    }
                } else {
                    // No family/row info — round-robin by group index.
                    let lane = gi % num_lanes;
                    ranges[lane] = (0, group.count);
                }
                LaneAssignment { lane_ranges: ranges }
            }
            GroupKind::Elementwise | GroupKind::OtherReduce => {
                if let Some(fam) = info.row_family {
                    if info.num_rows > 1 {
                        // Multi-row group — split across lanes.
                        let m = info.num_rows;
                        let n = info.atoms_per_row;
                        let row_lanes = family_row_lanes
                            .get(&fam)
                            .cloned()
                            .unwrap_or_else(|| (0..m as usize).map(|r| r % num_lanes).collect());

                        // Count atoms per lane: consecutive rows assigned to the
                        // same lane form a contiguous block.
                        // Group atoms into per-lane ranges.
                        compute_multi_row_lane_ranges(m, n, &row_lanes, num_lanes)
                    } else if info.row_index.is_some() {
                        // Single-row group with known row index.
                        let mut ranges = vec![(0u64, 0u64); num_lanes];
                        if let Some(row_lanes) = family_row_lanes.get(&fam) {
                            let row_idx = info.row_index.unwrap() as usize;
                            if row_idx < row_lanes.len() {
                                let lane = row_lanes[row_idx];
                                ranges[lane] = (0, group.count);
                            } else {
                                ranges[0] = (0, group.count);
                            }
                        } else {
                            ranges[0] = (0, group.count);
                        }
                        LaneAssignment { lane_ranges: ranges }
                    } else {
                        // In a family but no multi-row structure or row index.
                        // Split evenly.
                        split_evenly(group.count, num_lanes)
                    }
                } else {
                    // No family — split evenly across lanes.
                    split_evenly(group.count, num_lanes)
                }

            }
            GroupKind::IndirectLoad => {
                // IndirectLoad is typically small, assign to lane 0.
                let mut ranges = vec![(0u64, 0u64); num_lanes];
                ranges[0] = (0, group.count);
                LaneAssignment { lane_ranges: ranges }
            }
        };
        assignments.push(la);
    }

    assignments
}

/// Compute lane ranges for a multi-row group where rows are assigned to lanes.
/// Rows assigned to the same lane must be contiguous because the group's atoms
/// are laid out as [row0_atoms..., row1_atoms..., ...].
fn compute_multi_row_lane_ranges(
    num_rows: u64,
    atoms_per_row: u64,
    row_lanes: &[usize],
    num_lanes: usize,
) -> LaneAssignment {
    // For each lane, find the range of rows assigned to it.
    // Because our row-to-lane mapping is monotonic (row i -> lane floor(i*L/M)),
    // rows for each lane are contiguous.
    let mut lane_first_row: Vec<Option<u64>> = vec![None; num_lanes];
    let mut lane_last_row: Vec<Option<u64>> = vec![None; num_lanes];

    for (row, &lane) in row_lanes.iter().enumerate() {
        let row = row as u64;
        if lane_first_row[lane].is_none() {
            lane_first_row[lane] = Some(row);
        }
        lane_last_row[lane] = Some(row);
    }

    let mut ranges = vec![(0u64, 0u64); num_lanes];
    for lane in 0..num_lanes {
        if let (Some(first), Some(last)) = (lane_first_row[lane], lane_last_row[lane]) {
            let offset = first * atoms_per_row;
            let count = (last - first + 1) * atoms_per_row;
            ranges[lane] = (offset, count);
        }
    }

    LaneAssignment { lane_ranges: ranges }
}

/// Split a group evenly across lanes (for groups with no row structure).
fn split_evenly(count: u64, num_lanes: usize) -> LaneAssignment {
    let mut ranges = vec![(0u64, 0u64); num_lanes];
    if count == 0 {
        return LaneAssignment { lane_ranges: ranges };
    }

    let base_chunk = count / num_lanes as u64;
    let remainder = count % num_lanes as u64;
    let mut offset = 0u64;

    for lane in 0..num_lanes {
        let chunk = base_chunk + if (lane as u64) < remainder { 1 } else { 0 };
        ranges[lane] = (offset, chunk);
        offset += chunk;
    }

    LaneAssignment { lane_ranges: ranges }
}

// ─── Barrier detection ───────────────────────────────────────────────────────

/// Detect where barriers are needed by checking cross-lane data dependencies.
///
/// A barrier is needed before group G if G reads atoms produced by a different
/// lane than G's own lane assignment. Literals (shared data) don't trigger
/// barriers.
///
/// Returns a phase ID for each group. Groups between consecutive barriers
/// share a phase ID.
fn detect_barriers(
    graph: &NanoGraph,
    lane_assignments: &[LaneAssignment],
    topo_order: &[usize],
    predecessors: &[Vec<usize>],
    group_info: &[GroupInfo],
) -> Vec<usize> {
    let groups = graph.groups();
    let num_groups = groups.len();
    let mut phase_ids = vec![0usize; num_groups];

    // For each group in topo order, check if any non-Literal predecessor
    // is assigned to a different lane.
    //
    // But the concept of "different lane" is subtle when groups are split
    // across lanes. The real question is: does this group, at any lane,
    // need to read atoms that were produced by a different lane?
    //
    // For a multi-row elementwise group split as [rows 0-7 -> lane 0, rows 8-15 -> lane 1]:
    // - If its predecessor is also split the same way, each lane reads its own slice -> no barrier
    // - If its predecessor is a Literal -> no barrier (shared data)
    // - If its predecessor is split differently -> barrier
    //
    // For simplicity and correctness, we check: for each predecessor P of group G,
    // do the producing lanes of P overlap with the consuming lanes of G in a
    // compatible way? Specifically, every atom that G reads from P must be
    // produced by the same lane that G's reading lane is assigned to.
    //
    // Approximation: if a predecessor is assigned to lanes that are a SUBSET of
    // the consumer's lanes, no barrier needed. If there's any mismatch, barrier.
    //
    // More precisely: a group needs a barrier if it reads from a predecessor
    // where the predecessor's atom ranges are produced by different lanes than
    // the consumer expects.

    // Track the "latest phase" of each group's predecessors.
    // A group must be in a phase >= max(predecessor phases).
    // A barrier (phase increment) happens when cross-lane reads are detected.

    for &gi in topo_order {
        if group_info[gi].kind == GroupKind::Literal {
            phase_ids[gi] = 0; // Literals are always phase 0 (available from start)
            continue;
        }

        // The earliest phase this group can be in: max of predecessor phases.
        let min_phase = predecessors[gi]
            .iter()
            .map(|&p| phase_ids[p])
            .max()
            .unwrap_or(0);

        // Check if this group needs a barrier (cross-lane read).
        let needs_barrier = has_cross_lane_dependency(
            gi,
            graph,
            lane_assignments,
            predecessors,
            group_info,
        );

        if needs_barrier {
            // This group needs its predecessors' outputs to be visible.
            // Place it in the phase after the latest predecessor.
            phase_ids[gi] = min_phase + 1;
        } else {
            // No cross-lane dependency — same phase as predecessors.
            phase_ids[gi] = min_phase;
        }
    }

    phase_ids
}

/// Check if a group has cross-lane dependencies (needs a barrier before it).
fn has_cross_lane_dependency(
    gi: usize,
    graph: &NanoGraph,
    lane_assignments: &[LaneAssignment],
    predecessors: &[Vec<usize>],
    group_info: &[GroupInfo],
) -> bool {
    let groups = graph.groups();
    let group = &groups[gi];
    let my_assignment = &lane_assignments[gi];

    // Find which lanes own this group's atoms.
    let my_lanes: HashSet<usize> = (0..my_assignment.lane_ranges.len())
        .filter(|&lane| my_assignment.lane_ranges[lane].1 > 0)
        .collect();

    if my_lanes.is_empty() {
        return false;
    }

    for &pred_gi in &predecessors[gi] {
        if group_info[pred_gi].kind == GroupKind::Literal {
            continue; // Literals are shared, never trigger barriers.
        }

        let pred_assignment = &lane_assignments[pred_gi];

        // Check if the predecessor produces atoms that this group needs,
        // and whether those atoms are on different lanes.
        //
        // For aligned row splits: if both groups have the same row family
        // and the same row-to-lane mapping, the reads are lane-local.
        if let (Some(my_fam), Some(pred_fam)) = (
            group_info[gi].row_family,
            group_info[pred_gi].row_family,
        ) {
            if my_fam == pred_fam {
                // Same family — check if lane ranges are compatible.
                // If the predecessor's lane ranges are a subset of ours,
                // or if for each lane, the predecessor's atoms on that lane
                // are only read by that same lane's atoms, no barrier.
                if are_lane_compatible(my_assignment, pred_assignment) {
                    continue;
                }
            }
        }

        // For groups not in the same family or with incompatible splits:
        // check if the predecessor's producing lanes are all within our lanes.
        let pred_lanes: HashSet<usize> = (0..pred_assignment.lane_ranges.len())
            .filter(|&lane| pred_assignment.lane_ranges[lane].1 > 0)
            .collect();

        // If the predecessor produces on lanes that overlap with ours,
        // but also produces on lanes that DON'T overlap, that's a cross-lane read.
        //
        // However, we need to check more carefully: does THIS group actually
        // read atoms from the predecessor that are on a different lane?
        //
        // Simple conservative check: if any predecessor lane is not in my lanes,
        // and this group reads from the predecessor, it's a cross-lane dependency.
        //
        // But actually, a group might be split across ALL lanes, and the predecessor
        // is also split across all lanes. If they're split compatibly (same rows
        // to same lanes), no barrier is needed. The family check above handles this.
        //
        // For non-family cases: if the predecessor is on a single lane and that
        // lane is one of ours, no barrier. If the predecessor is on multiple lanes
        // and we're on multiple lanes, it depends on the access pattern.

        // Conservative approach: check if every lane that owns part of this group
        // only reads from atoms produced by that same lane.
        if !check_lane_local_reads(gi, pred_gi, graph, lane_assignments, group_info) {
            return true;
        }
    }

    false
}

/// Check if two lane assignments are compatible (same rows go to same lanes).
fn are_lane_compatible(a: &LaneAssignment, b: &LaneAssignment) -> bool {
    // Two assignments are compatible if for each lane, the atom ranges overlap
    // in the expected way (both have atoms for the same row indices on the same lanes).
    //
    // Simple check: if both have the same set of active lanes with proportional
    // atom counts, they're compatible.
    let a_active: Vec<usize> = (0..a.lane_ranges.len())
        .filter(|&l| a.lane_ranges[l].1 > 0)
        .collect();
    let b_active: Vec<usize> = (0..b.lane_ranges.len())
        .filter(|&l| b.lane_ranges[l].1 > 0)
        .collect();

    if a_active != b_active {
        return false;
    }

    // Check proportional: each lane should have the same fraction of atoms.
    let a_total: u64 = a.lane_ranges.iter().map(|r| r.1).sum();
    let b_total: u64 = b.lane_ranges.iter().map(|r| r.1).sum();

    if a_total == 0 || b_total == 0 {
        return true;
    }

    for &lane in &a_active {
        let a_frac = a.lane_ranges[lane].1 as f64 / a_total as f64;
        let b_frac = b.lane_ranges[lane].1 as f64 / b_total as f64;
        if (a_frac - b_frac).abs() > 0.01 {
            return false;
        }
    }

    true
}

/// Check if all reads from group `gi` to predecessor `pred_gi` are lane-local.
///
/// For each lane that owns part of `gi`, check that all atoms it reads from
/// `pred_gi` are produced by that same lane.
fn check_lane_local_reads(
    gi: usize,
    pred_gi: usize,
    graph: &NanoGraph,
    lane_assignments: &[LaneAssignment],
    group_info: &[GroupInfo],
) -> bool {
    let groups = graph.groups();
    let group = &groups[gi];
    let pred_group = &groups[pred_gi];
    let my_la = &lane_assignments[gi];
    let pred_la = &lane_assignments[pred_gi];
    let num_lanes = my_la.lane_ranges.len();

    // For each input ref that could reference pred_gi's atoms:
    for input in &group.inputs {
        // Quick check: does this input ref even reference pred_gi?
        let sample = sample_source_atoms(input, group.count);
        let refs_pred = sample.iter().any(|id| pred_group.contains(*id));
        if !refs_pred {
            continue;
        }

        // For each lane that owns part of this group:
        for lane in 0..num_lanes {
            let (my_offset, my_count) = my_la.lane_ranges[lane];
            if my_count == 0 {
                continue;
            }

            // Sample a few atoms from this lane's range and check which
            // lane they map to in the predecessor.
            let check_offsets = if my_count <= 8 {
                (0..my_count).collect::<Vec<_>>()
            } else {
                vec![
                    0,
                    1,
                    my_count / 4,
                    my_count / 2,
                    3 * my_count / 4,
                    my_count - 2,
                    my_count - 1,
                ]
            };

            for &local_off in &check_offsets {
                let global_off = my_offset + local_off;
                if global_off >= group.count {
                    continue;
                }
                let src_id = input.resolve(global_off, 0);
                if !pred_group.contains(src_id) {
                    continue; // This atom reads from a different predecessor
                }

                // Find which lane owns this source atom in pred.
                let src_offset_in_pred = src_id.0 - pred_group.base_id.0;
                let src_lane = find_owning_lane(pred_la, src_offset_in_pred);

                if let Some(sl) = src_lane {
                    if sl != lane {
                        return false; // Cross-lane read!
                    }
                }
            }
        }
    }

    true
}

/// Find which lane owns a given atom offset within a group.
fn find_owning_lane(la: &LaneAssignment, atom_offset: u64) -> Option<usize> {
    for (lane, &(off, count)) in la.lane_ranges.iter().enumerate() {
        if count > 0 && atom_offset >= off && atom_offset < off + count {
            return Some(lane);
        }
    }
    None
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::nano_graph::{ScalarBinOp, ScalarUnaryOp};
    use crate::numeric_scalar::NumericScalar;

    /// Build a merged-matmul NanoGraph matching real lowering structure.
    /// C[M,N] = A[M,K] @ B[K,N] with M merged Mul groups (StridedBroadcast).
    fn build_merged_matmul(m: u64, k: u64, n: u64) -> NanoGraph {
        let mut g = NanoGraph::new();

        // A[M, K]
        let a_base = g.push_group(
            m * k,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );

        // B[K, N]
        let b_base = g.push_group(
            k * n,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );

        // M Mul groups, each count=K*N
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

        // M ReduceSum groups, each count=N
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

    /// Build a chain: matmul1 -> elementwise op (monolithic) -> matmul2.
    /// This is the challenging case: the elementwise group has count=M*N
    /// and must be split across lanes.
    fn build_matmul_elementwise_matmul(
        m: u64,
        k1: u64,
        n1: u64,
        k2: u64,
        n2: u64,
    ) -> NanoGraph {
        assert_eq!(n1, k2, "inner dimensions must match");
        let mut g = NanoGraph::new();

        // Weights for matmul 1
        let a1 = g.push_group(
            m * k1,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );
        let b1 = g.push_group(
            k1 * n1,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );

        // Matmul 1
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

        // Monolithic elementwise op: count = M*N1, reading from all reduces.
        // This models a residual add or activation applied to the full output.
        let elem = g.push_group(
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

        // Weights for matmul 2
        let b2 = g.push_group(
            k2 * n2,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );

        // Matmul 2 reads from the elementwise output
        let mut mul2_bases = Vec::new();
        for row in 0..m {
            let mul = g.push_group(
                k2 * n2,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![
                    InputRef::StridedBroadcast {
                        base: AtomId(elem.0 + row * n1),
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
                    reduce_count: k2,
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

    // ─── Coverage test ───────────────────────────────────────────────────

    /// Every non-Literal atom must be assigned to exactly one lane.
    #[test]
    fn test_coverage_single_matmul() {
        let g = build_merged_matmul(8, 16, 32);
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let plan = plan_execution(&g, 4);

        // Collect all assigned atoms.
        let groups = g.groups();
        let mut assigned: HashMap<(usize, u64), usize> = HashMap::new(); // (group_idx, offset) -> lane

        for phase in &plan.phases {
            for (lane, work) in phase.lane_work.iter().enumerate() {
                for lw in work {
                    for off in 0..lw.atom_count {
                        let key = (lw.group_idx, lw.atom_offset + off);
                        let prev = assigned.insert(key, lane);
                        assert!(
                            prev.is_none(),
                            "Atom (group={}, offset={}) assigned to both lane {} and {}",
                            key.0,
                            key.1,
                            prev.unwrap(),
                            lane
                        );
                    }
                }
            }
        }

        // Check that all non-Literal atoms are covered.
        for (gi, group) in groups.iter().enumerate() {
            if matches!(group.op, ScalarOp::Literal(_)) {
                continue;
            }
            for off in 0..group.count {
                assert!(
                    assigned.contains_key(&(gi, off)),
                    "Atom (group={}, offset={}) not assigned to any lane",
                    gi,
                    off
                );
            }
        }
    }

    /// Coverage check for the matmul-elementwise-matmul chain.
    #[test]
    fn test_coverage_matmul_chain() {
        let g = build_matmul_elementwise_matmul(8, 16, 32, 32, 16);
        assert!(g.validate().is_empty(), "{:?}", g.validate());

        let plan = plan_execution(&g, 4);

        let groups = g.groups();
        let mut assigned: HashMap<(usize, u64), usize> = HashMap::new();

        for phase in &plan.phases {
            for (lane, work) in phase.lane_work.iter().enumerate() {
                for lw in work {
                    for off in 0..lw.atom_count {
                        let key = (lw.group_idx, lw.atom_offset + off);
                        let prev = assigned.insert(key, lane);
                        assert!(
                            prev.is_none(),
                            "Atom (group={}, offset={}) assigned to both lane {} and {}",
                            key.0,
                            key.1,
                            prev.unwrap(),
                            lane
                        );
                    }
                }
            }
        }

        for (gi, group) in groups.iter().enumerate() {
            if matches!(group.op, ScalarOp::Literal(_)) {
                continue;
            }
            for off in 0..group.count {
                assert!(
                    assigned.contains_key(&(gi, off)),
                    "Atom (group={}, offset={}) not assigned to any lane",
                    gi,
                    off
                );
            }
        }
    }

    // ─── Balance test ────────────────────────────────────────────────────

    /// Work should be roughly balanced across lanes (within 2x).
    #[test]
    fn test_balance_single_matmul() {
        let g = build_merged_matmul(64, 768, 768);
        let plan = plan_execution(&g, 8);

        for (phase_idx, phase) in plan.phases.iter().enumerate() {
            let lane_atoms: Vec<u64> = phase
                .lane_work
                .iter()
                .map(|work| work.iter().map(|lw| lw.atom_count).sum())
                .collect();

            let max_atoms = *lane_atoms.iter().max().unwrap_or(&0);
            let min_atoms = *lane_atoms.iter().filter(|&&a| a > 0).min().unwrap_or(&0);

            if max_atoms > 0 && min_atoms > 0 {
                let ratio = max_atoms as f64 / min_atoms as f64;
                assert!(
                    ratio < 2.0,
                    "Phase {}: imbalance ratio {:.1}x (max={}, min={}). Lane atoms: {:?}",
                    phase_idx,
                    ratio,
                    max_atoms,
                    min_atoms,
                    lane_atoms,
                );
            }
        }
    }

    /// Balance check for the chain case.
    #[test]
    fn test_balance_matmul_chain() {
        let g = build_matmul_elementwise_matmul(64, 768, 768, 768, 768);
        let plan = plan_execution(&g, 8);

        for (phase_idx, phase) in plan.phases.iter().enumerate() {
            let lane_atoms: Vec<u64> = phase
                .lane_work
                .iter()
                .map(|work| work.iter().map(|lw| lw.atom_count).sum())
                .collect();

            let max_atoms = *lane_atoms.iter().max().unwrap_or(&0);
            let min_atoms = *lane_atoms.iter().filter(|&&a| a > 0).min().unwrap_or(&0);

            if max_atoms > 0 && min_atoms > 0 {
                let ratio = max_atoms as f64 / min_atoms as f64;
                assert!(
                    ratio < 2.0,
                    "Phase {}: imbalance ratio {:.1}x (max={}, min={}). Lane atoms: {:?}",
                    phase_idx,
                    ratio,
                    max_atoms,
                    min_atoms,
                    lane_atoms,
                );
            }
        }
    }

    // ─── Independence test ───────────────────────────────────────────────

    /// Within each phase, no lane should read atoms produced by another lane
    /// in the same phase (that would violate the barrier model).
    #[test]
    fn test_independence_single_matmul() {
        let g = build_merged_matmul(16, 32, 64);
        assert!(g.validate().is_empty());
        let plan = plan_execution(&g, 4);
        verify_phase_independence(&g, &plan);
    }

    #[test]
    fn test_independence_matmul_chain() {
        let g = build_matmul_elementwise_matmul(16, 32, 64, 64, 32);
        assert!(g.validate().is_empty());
        let plan = plan_execution(&g, 4);
        verify_phase_independence(&g, &plan);
    }

    /// Verify that within each phase, lanes are independent: no lane reads
    /// atoms produced by another lane in the same phase.
    fn verify_phase_independence(graph: &NanoGraph, plan: &ExecutionPlan) {
        let groups = graph.groups();

        for (phase_idx, phase) in plan.phases.iter().enumerate() {
            // Build a map: (group_idx, atom_offset) -> producing lane for this phase.
            let mut producers: HashMap<(usize, u64), usize> = HashMap::new();

            for (lane, work) in phase.lane_work.iter().enumerate() {
                for lw in work {
                    for off in 0..lw.atom_count {
                        producers.insert((lw.group_idx, lw.atom_offset + off), lane);
                    }
                }
            }

            // For each lane's work, check that every atom it reads was either:
            // 1. Produced by the same lane in this phase, OR
            // 2. Produced in a previous phase (not in `producers`), OR
            // 3. A Literal (shared data).
            for (lane, work) in phase.lane_work.iter().enumerate() {
                for lw in work {
                    let group = &groups[lw.group_idx];
                    if matches!(group.op, ScalarOp::Literal(_)) {
                        continue;
                    }

                    for input in &group.inputs {
                        // Check a sample of atoms in this lane's range.
                        let check_count = lw.atom_count.min(16);
                        let step = if lw.atom_count > check_count {
                            lw.atom_count / check_count
                        } else {
                            1
                        };

                        for i in (0..lw.atom_count).step_by(step as usize) {
                            let global_offset = lw.atom_offset + i;
                            if global_offset >= group.count {
                                continue;
                            }
                            let src_id = input.resolve(global_offset, 0);

                            // Find which group this source belongs to.
                            if let Some(src_gi) = find_group_index(groups, src_id) {
                                let src_group = &groups[src_gi];
                                if matches!(src_group.op, ScalarOp::Literal(_)) {
                                    continue;
                                }

                                let src_offset = src_id.0 - src_group.base_id.0;

                                // Check if this source was produced in this phase.
                                if let Some(&producer_lane) =
                                    producers.get(&(src_gi, src_offset))
                                {
                                    assert_eq!(
                                        producer_lane, lane,
                                        "Phase {}: lane {} reads atom (group={}, offset={}) \
                                         produced by lane {} in the same phase. \
                                         Consumer: group={}, offset={}",
                                        phase_idx,
                                        lane,
                                        src_gi,
                                        src_offset,
                                        producer_lane,
                                        lw.group_idx,
                                        global_offset,
                                    );
                                }
                                // If not in producers, it was produced in a previous phase — OK.
                            }
                        }
                    }
                }
            }
        }
    }

    // ─── Barrier count test ──────────────────────────────────────────────

    /// A single matmul should need 0 barriers (all rows are independent).
    #[test]
    fn test_single_matmul_no_barriers() {
        let g = build_merged_matmul(8, 16, 32);
        let plan = plan_execution(&g, 4);

        // All work should be in a single phase (or very few phases).
        assert!(
            plan.phases.len() <= 2,
            "Single matmul should need at most 1-2 phases, got {}",
            plan.phases.len()
        );
    }

    /// A matmul chain (matmul -> elementwise -> matmul) where each row of
    /// matmul2 reads only its own row's elementwise output should NOT need
    /// a barrier — all work is lane-local.
    #[test]
    fn test_matmul_chain_no_barrier_when_row_local() {
        let g = build_matmul_elementwise_matmul(8, 16, 32, 32, 16);
        let plan = plan_execution(&g, 4);

        // Row-local chains should be 1 phase (no cross-lane deps).
        assert!(
            plan.phases.len() <= 2,
            "Row-local matmul chain should need at most 2 phases, got {}",
            plan.phases.len()
        );
    }

    /// Build a graph with genuine cross-lane dependency: a "global reduce"
    /// that reads ALL rows' outputs, then broadcasts back.
    /// matmul -> global_reduce(all M*N atoms to 1) -> broadcast_add(all M*N atoms)
    /// The global reduce requires all lanes' outputs, so it needs a barrier.
    fn build_matmul_with_global_reduce(m: u64, k: u64, n: u64) -> NanoGraph {
        let mut g = NanoGraph::new();

        let a = g.push_group(
            m * k,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![], vec![], vec![],
        );
        let b = g.push_group(
            k * n,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![], vec![], vec![],
        );

        let mut mul_bases = Vec::new();
        for row in 0..m {
            let a_row = AtomId(a.0 + row * k);
            let mul = g.push_group(
                k * n,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![], vec![],
                vec![
                    InputRef::StridedBroadcast {
                        base: a_row,
                        stride: 1,
                        repeat: n,
                    },
                    InputRef::Affine {
                        base: b,
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
                vec![], vec![],
                vec![InputRef::Affine {
                    base: mul_bases[row as usize],
                    stride: 1,
                }],
            );
            reduce_bases.push(red);
        }

        // Global reduce: sum ALL M*N matmul outputs down to 1 atom.
        // This reads across all lanes.
        let global_reduce = g.push_group(
            1,
            ScalarOp::ReduceSum {
                reduce_count: m * n,
                reduce_stride: 1,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![], vec![],
            vec![InputRef::Affine {
                base: reduce_bases[0],
                stride: 1,
            }],
        );

        // Broadcast the global result back to M*N atoms and add to matmul output.
        let out = g.push_group(
            m * n,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![], vec![],
            vec![
                InputRef::Affine {
                    base: reduce_bases[0],
                    stride: 1,
                },
                InputRef::Broadcast(global_reduce),
            ],
        );

        for i in 0..(m * n) {
            g.outputs.push(AtomId(out.0 + i));
        }
        g
    }

    /// A graph with a global reduce (cross-lane dependency) should have barriers.
    #[test]
    fn test_global_reduce_needs_barrier() {
        let g = build_matmul_with_global_reduce(8, 16, 32);
        assert!(g.validate().is_empty(), "{:?}", g.validate());
        let plan = plan_execution(&g, 4);

        // The global reduce reads all lanes' outputs, so we need at least 2 phases.
        assert!(
            plan.phases.len() >= 2,
            "Global reduce should need at least 2 phases, got {}",
            plan.phases.len()
        );
    }

    /// Independence must hold even with global reduces.
    #[test]
    fn test_independence_global_reduce() {
        let g = build_matmul_with_global_reduce(8, 16, 32);
        assert!(g.validate().is_empty());
        let plan = plan_execution(&g, 4);
        verify_phase_independence(&g, &plan);
    }

    /// Coverage must hold with global reduces.
    #[test]
    fn test_coverage_global_reduce() {
        let g = build_matmul_with_global_reduce(8, 16, 32);
        assert!(g.validate().is_empty());
        let plan = plan_execution(&g, 4);

        let groups = g.groups();
        let mut assigned: HashSet<(usize, u64)> = HashSet::new();
        for phase in &plan.phases {
            for (lane, work) in phase.lane_work.iter().enumerate() {
                for lw in work {
                    for off in 0..lw.atom_count {
                        assigned.insert((lw.group_idx, lw.atom_offset + off));
                    }
                }
            }
        }

        for (gi, group) in groups.iter().enumerate() {
            if matches!(group.op, ScalarOp::Literal(_)) {
                continue;
            }
            for off in 0..group.count {
                assert!(
                    assigned.contains(&(gi, off)),
                    "Atom (group={}, offset={}) not assigned",
                    gi,
                    off
                );
            }
        }
    }

    // ─── Consistency test ────────────────────────────────────────────────

    /// Lane assignments should be consistent across matmuls: the same lane
    /// should handle the same row indices in both matmuls.
    #[test]
    fn test_lane_consistency() {
        let g = build_matmul_elementwise_matmul(8, 16, 32, 32, 16);
        let plan = plan_execution(&g, 4);
        let groups = g.groups();

        // Find ReduceSum groups (matmul outputs).
        let reduce_groups: Vec<usize> = (0..groups.len())
            .filter(|&gi| matches!(groups[gi].op, ScalarOp::ReduceSum { .. }))
            .collect();

        // For each phase, check which lanes own which reduce groups.
        let mut lane_of_reduce: HashMap<usize, usize> = HashMap::new();
        for phase in &plan.phases {
            for (lane, work) in phase.lane_work.iter().enumerate() {
                for lw in work {
                    if reduce_groups.contains(&lw.group_idx) && lw.atom_count > 0 {
                        lane_of_reduce.insert(lw.group_idx, lane);
                    }
                }
            }
        }

        // Matmul 1's reduce groups should have a consistent pattern with matmul 2's.
        // Specifically, if M=8 and num_lanes=4, rows should be assigned:
        // Lane 0: rows 0,1  Lane 1: rows 2,3  Lane 2: rows 4,5  Lane 3: rows 6,7
        // And this should hold for BOTH matmuls.
        // (We verify this by checking that the lane assignment is the same function
        // of row index for both matmuls.)

        // Group the reduce groups by matmul (by checking their predecessors' structure).
        // For simplicity, just check that row N in the first matmul and row N in the
        // second matmul are on the same lane.
        let m = 8usize;
        let matmul1_reduces: Vec<usize> = reduce_groups[..m].to_vec();
        let matmul2_reduces: Vec<usize> = reduce_groups[m..].to_vec();

        if matmul1_reduces.len() == matmul2_reduces.len() {
            for row in 0..m {
                let lane1 = lane_of_reduce.get(&matmul1_reduces[row]);
                let lane2 = lane_of_reduce.get(&matmul2_reduces[row]);
                if let (Some(&l1), Some(&l2)) = (lane1, lane2) {
                    assert_eq!(
                        l1, l2,
                        "Row {} is on lane {} in matmul1 but lane {} in matmul2",
                        row, l1, l2
                    );
                }
            }
        }
    }

    // ─── Diagnostic test ─────────────────────────────────────────────────

    /// Print plan summary for manual inspection.
    #[test]
    fn test_plan_summary() {
        let g = build_matmul_elementwise_matmul(64, 768, 768, 768, 768);
        let plan = plan_execution(&g, 8);

        let groups = g.groups();
        let total_compute_atoms: u64 = groups
            .iter()
            .filter(|g| !matches!(g.op, ScalarOp::Literal(_)))
            .map(|g| g.count)
            .sum();

        println!("Plan: {} phases, {} lanes", plan.phases.len(), plan.num_lanes);
        println!("Total compute atoms: {}", total_compute_atoms);

        let mut total_assigned: u64 = 0;
        for (phase_idx, phase) in plan.phases.iter().enumerate() {
            let lane_atoms: Vec<u64> = phase
                .lane_work
                .iter()
                .map(|work| work.iter().map(|lw| lw.atom_count).sum())
                .collect();
            let phase_total: u64 = lane_atoms.iter().sum();
            total_assigned += phase_total;

            let max_a = *lane_atoms.iter().max().unwrap_or(&0);
            let min_a = *lane_atoms.iter().filter(|&&a| a > 0).min().unwrap_or(&0);
            let ratio = if min_a > 0 {
                max_a as f64 / min_a as f64
            } else {
                f64::INFINITY
            };

            let work_items: Vec<usize> = phase.lane_work.iter().map(|w| w.len()).collect();

            println!(
                "  Phase {}: {:>12} atoms, ratio {:.2}x, lane atoms {:?}, work items {:?}",
                phase_idx, phase_total, ratio, lane_atoms, work_items,
            );
        }

        println!("Total assigned: {} / {}", total_assigned, total_compute_atoms);
        assert_eq!(total_assigned, total_compute_atoms, "Not all atoms assigned!");
    }

    // ─── Edge case tests ─────────────────────────────────────────────────

    #[test]
    fn test_empty_graph() {
        let g = NanoGraph::new();
        let plan = plan_execution(&g, 4);
        assert_eq!(plan.phases.len(), 0);
    }

    #[test]
    fn test_single_lane() {
        let g = build_merged_matmul(8, 16, 32);
        let plan = plan_execution(&g, 1);

        // With 1 lane, everything should be in that lane.
        let mut total_atoms = 0u64;
        for phase in &plan.phases {
            assert_eq!(phase.lane_work.len(), 1);
            total_atoms += phase.lane_work[0]
                .iter()
                .map(|lw| lw.atom_count)
                .sum::<u64>();
        }

        let groups = g.groups();
        let expected: u64 = groups
            .iter()
            .filter(|g| !matches!(g.op, ScalarOp::Literal(_)))
            .map(|g| g.count)
            .sum();
        assert_eq!(total_atoms, expected);
    }

    #[test]
    fn test_more_lanes_than_rows() {
        // 4 rows, 8 lanes — some lanes will be empty.
        let g = build_merged_matmul(4, 16, 32);
        let plan = plan_execution(&g, 8);

        // Should still produce a valid plan.
        let groups = g.groups();
        let mut assigned: HashSet<(usize, u64)> = HashSet::new();
        for phase in &plan.phases {
            for (lane, work) in phase.lane_work.iter().enumerate() {
                for lw in work {
                    for off in 0..lw.atom_count {
                        assigned.insert((lw.group_idx, lw.atom_offset + off));
                    }
                }
            }
        }

        for (gi, group) in groups.iter().enumerate() {
            if matches!(group.op, ScalarOp::Literal(_)) {
                continue;
            }
            for off in 0..group.count {
                assert!(
                    assigned.contains(&(gi, off)),
                    "Atom (group={}, offset={}) not assigned",
                    gi,
                    off
                );
            }
        }
    }
}
