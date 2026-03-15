#![allow(clippy::all, dead_code, unreachable_patterns, unused_variables, unused_imports)]
//! Lane+barrier execution plan v2a: group-splitting planner.
//!
//! Previous attempts treated AtomGroups as indivisible. GPT-2's elementwise ops
//! produce ONE group of 49,152 atoms. With 8 lanes, that's 49,152 atoms on one
//! lane and 0 on the others. This planner splits groups into sub-ranges.
//!
//! ## Algorithm
//!
//! 1. **Build group DAG** — resolve InputRefs to find producer groups.
//!
//! 2. **Topological sort + reduce depth** — each group gets a "reduce depth"
//!    counting how many ReduceSum/ReduceMax groups lie on its longest path to
//!    a root. This gives the coarse phase structure.
//!
//! 3. **Intra-phase leveling** — within each reduce-depth phase, topologically
//!    sort the groups and compute levels. Level 0 has no intra-phase producers.
//!    Level 1 depends on level 0, etc.
//!
//! 4. **Split + assign per level** — at each level, split large groups across
//!    lanes. Then check: does the next level's groups read from split groups?
//!    If so, a barrier is needed between these levels (they become separate
//!    phases). If all reads are from unsplit groups (or same-lane), no barrier.
//!
//! 5. **Greedy load balancing** — largest-first assignment to least-loaded lane.

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

use crate::nano_graph::{AtomGroup, AtomId, InputRef, NanoGraph, ScalarOp};

// ─── Public API ──────────────────────────────────────────────────────────────

/// An execution plan: lanes running through phases separated by barriers.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub num_lanes: usize,
    pub phases: Vec<Phase>,
}

/// One phase of execution. All lanes work independently within a phase.
#[derive(Debug, Clone)]
pub struct Phase {
    /// lane_idx -> work items for this lane in this phase.
    pub lane_work: Vec<Vec<LaneWork>>,
}

/// A unit of work assigned to a lane: a (possibly partial) AtomGroup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaneWork {
    /// Index into the NanoGraph's group list.
    pub group_idx: usize,
    /// Start offset within the group (0 for full group).
    pub atom_offset: u64,
    /// Number of atoms this work item covers (group.count for full group).
    pub atom_count: u64,
}

impl LaneWork {
    fn full(group_idx: usize, count: u64) -> Self {
        LaneWork {
            group_idx,
            atom_offset: 0,
            atom_count: count,
        }
    }

    fn sub_range(group_idx: usize, offset: u64, count: u64) -> Self {
        LaneWork {
            group_idx,
            atom_offset: offset,
            atom_count: count,
        }
    }
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

    // Step 1: Classify groups and build DAG.
    let is_literal: Vec<bool> = groups
        .iter()
        .map(|g| matches!(g.op, ScalarOp::Literal(_)))
        .collect();

    let compute_indices: Vec<usize> = (0..n).filter(|&i| !is_literal[i]).collect();
    if compute_indices.is_empty() {
        return ExecutionPlan {
            num_lanes,
            phases: vec![],
        };
    }

    let (producers, consumers) = build_group_deps(groups);

    // Step 2: Topological sort + reduce depth.
    let topo_order = topological_sort(n, &producers);
    let reduce_depth = compute_reduce_depth(groups, &topo_order, &producers, &is_literal);

    let max_depth = reduce_depth.iter().copied().max().unwrap_or(0);

    // Step 3: Group by reduce depth.
    let num_coarse_phases = max_depth + 1;
    let mut coarse_phase_groups: Vec<Vec<usize>> = vec![vec![]; num_coarse_phases];
    for &gi in &compute_indices {
        let depth = reduce_depth[gi];
        coarse_phase_groups[depth].push(gi);
    }

    // Step 4: For each coarse phase, level the groups and produce fine-grained
    // phases where within-phase independence is guaranteed.
    let mut all_phases: Vec<Phase> = Vec::new();

    for coarse_idx in 0..num_coarse_phases {
        let phase_groups = &coarse_phase_groups[coarse_idx];
        if phase_groups.is_empty() {
            continue;
        }

        let sub_phases = build_leveled_phases(
            phase_groups,
            groups,
            num_lanes,
            &producers,
            &is_literal,
        );
        all_phases.extend(sub_phases);
    }

    // Remove empty phases.
    all_phases.retain(|p| p.lane_work.iter().any(|lane| !lane.is_empty()));

    ExecutionPlan {
        num_lanes,
        phases: all_phases,
    }
}

// ─── Group DAG ───────────────────────────────────────────────────────────────

/// Build producer and consumer graphs at group level.
fn build_group_deps(groups: &[AtomGroup]) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let n = groups.len();
    let mut producers: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut consumers: Vec<Vec<usize>> = vec![vec![]; n];

    for (gi, group) in groups.iter().enumerate() {
        let mut prod_set = BTreeSet::new();
        for input in &group.inputs {
            for pi in resolve_producer_groups(input, group.count, groups) {
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

/// Find the group index containing an AtomId via binary search.
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

/// Find all groups overlapping an AtomId range [lo, hi].
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

/// Resolve which groups produce atoms referenced by an InputRef.
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

        InputRef::StridedBroadcast {
            base,
            stride,
            repeat,
        } => {
            if count == 0 {
                return vec![];
            }
            let num_blocks = (count + repeat - 1) / repeat;
            let last_block_offset = (*stride) * (num_blocks as i64 - 1);
            let last = AtomId(base.0.wrapping_add(last_block_offset as u64));
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

        InputRef::SymAffine {
            base,
            stride_i,
            stride_k,
        } => {
            if count == 0 {
                return vec![];
            }
            let last_offset = (*stride_i as i64) * (count as i64 - 1);
            let last = AtomId(base.0.wrapping_add(last_offset as u64));
            let lo = base.0.min(last.0);
            let hi = base.0.max(last.0);
            find_groups_in_range(groups, lo, hi)
        }
    }
}

// ─── Topological sort + depth ────────────────────────────────────────────────

/// Kahn's algorithm topological sort.
fn topological_sort(n: usize, producers: &[Vec<usize>]) -> Vec<usize> {
    let mut in_degree = vec![0u32; n];
    for gi in 0..n {
        in_degree[gi] = producers[gi].len() as u32;
    }

    let mut forward: Vec<Vec<usize>> = vec![vec![]; n];
    for gi in 0..n {
        for &pi in &producers[gi] {
            forward[pi].push(gi);
        }
    }

    let mut queue: VecDeque<usize> = VecDeque::new();
    for gi in 0..n {
        if in_degree[gi] == 0 {
            queue.push_back(gi);
        }
    }

    let mut order = Vec::with_capacity(n);
    while let Some(gi) = queue.pop_front() {
        order.push(gi);
        for &ci in &forward[gi] {
            in_degree[ci] -= 1;
            if in_degree[ci] == 0 {
                queue.push_back(ci);
            }
        }
    }

    order
}

/// Compute the "reduce depth" of each group — the number of reduce operations
/// on the longest path from any root to this group.
fn compute_reduce_depth(
    groups: &[AtomGroup],
    topo_order: &[usize],
    producers: &[Vec<usize>],
    is_literal: &[bool],
) -> Vec<usize> {
    let n = groups.len();
    let mut depth = vec![0usize; n];

    for &gi in topo_order {
        if is_literal[gi] {
            depth[gi] = 0;
            continue;
        }

        let mut max_dep = 0usize;
        for &pi in &producers[gi] {
            let mut d = depth[pi];
            if !is_literal[pi] && is_reduce_op(&groups[pi].op) {
                d += 1;
            }
            max_dep = max_dep.max(d);
        }
        depth[gi] = max_dep;
    }

    depth
}

fn is_reduce_op(op: &ScalarOp) -> bool {
    matches!(op, ScalarOp::ReduceSum { .. } | ScalarOp::ReduceMax { .. })
}

// ─── Phase construction with intra-phase leveling ────────────────────────────

/// Build phases for a set of groups within one reduce-depth level.
///
/// This handles the key challenge: groups within the same reduce depth can
/// have internal dependencies (e.g., elementwise Tanh reads from ReduceSum,
/// both at reduce depth 1). We level the intra-phase DAG, split groups at
/// each level, and insert barriers between levels when cross-lane reads exist.
fn build_leveled_phases(
    phase_group_indices: &[usize],
    groups: &[AtomGroup],
    num_lanes: usize,
    producers: &[Vec<usize>],
    is_literal: &[bool],
) -> Vec<Phase> {
    if phase_group_indices.is_empty() {
        return vec![];
    }

    let phase_set: HashSet<usize> = phase_group_indices.iter().copied().collect();

    // Compute intra-phase levels: within this phase, what's the longest path
    // of intra-phase dependencies to reach each group?
    let mut intra_level: HashMap<usize, usize> = HashMap::new();

    // Process groups in a topological order restricted to this phase.
    // We use a simple iterative approach since the subset is small relative
    // to the full graph.
    let mut remaining: HashSet<usize> = phase_set.clone();
    let mut order: Vec<usize> = Vec::new();

    // Build intra-phase forward edges.
    let mut intra_forward: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut intra_in_degree: HashMap<usize, usize> = HashMap::new();

    for &gi in phase_group_indices {
        let mut deg = 0;
        for &pi in &producers[gi] {
            if phase_set.contains(&pi) {
                intra_forward.entry(pi).or_default().push(gi);
                deg += 1;
            }
        }
        intra_in_degree.insert(gi, deg);
    }

    // Kahn's on the phase subset.
    let mut queue: VecDeque<usize> = VecDeque::new();
    for &gi in phase_group_indices {
        if intra_in_degree[&gi] == 0 {
            queue.push_back(gi);
        }
    }
    while let Some(gi) = queue.pop_front() {
        order.push(gi);
        if let Some(fwd) = intra_forward.get(&gi) {
            for &ci in fwd {
                let deg = intra_in_degree.get_mut(&ci).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    queue.push_back(ci);
                }
            }
        }
    }

    // Compute levels.
    for &gi in &order {
        let mut max_level = 0usize;
        for &pi in &producers[gi] {
            if let Some(&pl) = intra_level.get(&pi) {
                max_level = max_level.max(pl + 1);
            }
        }
        intra_level.insert(gi, max_level);
    }

    let max_level = intra_level.values().copied().max().unwrap_or(0);

    // Group by level.
    let mut level_groups: Vec<Vec<usize>> = vec![vec![]; max_level + 1];
    for &gi in phase_group_indices {
        let level = intra_level[&gi];
        level_groups[level].push(gi);
    }

    // Build phases. At each level, we handle two cases:
    //
    // 1. **Split producer** — a group at this level reads from a group that's
    //    been split across multiple lanes. This requires a barrier: flush the
    //    current phase, start a new one.
    //
    // 2. **Unsplit producer on a specific lane** — the consumer must go on
    //    that same lane (lane constraint). No barrier needed.
    //
    // 3. **No intra-phase producer** — free assignment (greedy load balance).

    let mut phases: Vec<Phase> = Vec::new();
    let mut current_lane_work: Vec<Vec<LaneWork>> = vec![vec![]; num_lanes];

    // Track which lane(s) each group is on in the current phase.
    let mut group_lanes: HashMap<usize, HashSet<usize>> = HashMap::new();

    for level in 0..=max_level {
        let level_gis = &level_groups[level];
        if level_gis.is_empty() {
            continue;
        }

        // Check if any group at this level has a split intra-phase producer.
        let has_split_producer = level > 0
            && level_gis.iter().any(|&gi| {
                producers[gi].iter().any(|&pi| {
                    if is_literal[pi] {
                        return false;
                    }
                    if let Some(lanes) = group_lanes.get(&pi) {
                        lanes.len() > 1
                    } else {
                        false
                    }
                })
            });

        if has_split_producer && current_lane_work.iter().any(|l| !l.is_empty()) {
            // Flush current work as a phase (barrier).
            phases.push(Phase {
                lane_work: std::mem::replace(&mut current_lane_work, vec![vec![]; num_lanes]),
            });
            group_lanes.clear();
        }

        // For each group, determine its lane constraint from intra-phase producers.
        // If a group has an unsplit intra-phase producer, it must go on the same lane.
        let mut lane_constraints: HashMap<usize, Option<usize>> = HashMap::new();
        for &gi in level_gis {
            let mut constraint: Option<usize> = None;
            for &pi in &producers[gi] {
                if is_literal[pi] {
                    continue;
                }
                if let Some(lanes) = group_lanes.get(&pi) {
                    if lanes.len() == 1 {
                        let prod_lane = *lanes.iter().next().unwrap();
                        if let Some(existing) = constraint {
                            if existing != prod_lane {
                                // Conflicting constraints from two unsplit producers
                                // on different lanes. This shouldn't happen if levels
                                // are computed correctly, but if it does, we lose the
                                // constraint (will be caught by independence verification).
                                constraint = None;
                                break;
                            }
                        } else {
                            constraint = Some(prod_lane);
                        }
                    }
                    // Split producers: barrier was already inserted above.
                }
            }
            lane_constraints.insert(gi, constraint);
        }

        // Split groups and generate work items.
        let total_atoms: u64 = level_gis.iter().map(|&gi| groups[gi].count).sum();
        let target_per_lane = (total_atoms + num_lanes as u64 - 1) / num_lanes as u64;

        // Separate constrained and unconstrained groups.
        let mut constrained_items: Vec<(LaneWork, usize)> = Vec::new(); // (work, target_lane)
        let mut free_items: Vec<LaneWork> = Vec::new();

        for &gi in level_gis {
            let g = &groups[gi];
            let count = g.count;
            let constraint = lane_constraints[&gi];

            if let Some(target_lane) = constraint {
                // Constrained to a specific lane — don't split, assign whole.
                constrained_items.push((LaneWork::full(gi, count), target_lane));
            } else if count <= 1 || !is_splittable(&g.op, &g.inputs) {
                free_items.push(LaneWork::full(gi, count));
            } else {
                // Free to split across lanes.
                let max_chunk = target_per_lane.max(1);
                let num_chunks = ((count + max_chunk - 1) / max_chunk).max(1) as usize;
                let num_chunks = num_chunks.min(num_lanes);

                let chunk_base = count / num_chunks as u64;
                let remainder = count % num_chunks as u64;

                let mut offset = 0u64;
                for chunk_idx in 0..num_chunks {
                    let this_chunk =
                        chunk_base + if (chunk_idx as u64) < remainder { 1 } else { 0 };
                    if this_chunk > 0 {
                        free_items.push(LaneWork::sub_range(gi, offset, this_chunk));
                        offset += this_chunk;
                    }
                }
            }
        }

        // First, assign constrained items to their required lanes.
        let mut lane_load: Vec<u64> = current_lane_work
            .iter()
            .map(|lane| lane.iter().map(|w| w.atom_count).sum())
            .collect();

        for (item, target_lane) in constrained_items {
            lane_load[target_lane] += item.atom_count;
            current_lane_work[target_lane].push(item);
            group_lanes
                .entry(item.group_idx)
                .or_default()
                .insert(target_lane);
        }

        // Then, assign free items with greedy load balancing.
        free_items.sort_by(|a, b| b.atom_count.cmp(&a.atom_count));

        for item in free_items {
            let min_lane = lane_load
                .iter()
                .enumerate()
                .min_by_key(|(_, load)| **load)
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            lane_load[min_lane] += item.atom_count;
            current_lane_work[min_lane].push(item);
            group_lanes
                .entry(item.group_idx)
                .or_default()
                .insert(min_lane);
        }
    }

    // Flush remaining work.
    if current_lane_work.iter().any(|l| !l.is_empty()) {
        phases.push(Phase {
            lane_work: current_lane_work,
        });
    }

    phases
}

/// Determine if a group can be split into sub-ranges.
fn is_splittable(op: &ScalarOp, inputs: &[InputRef]) -> bool {
    for input in inputs {
        if matches!(input, InputRef::SymAffine { .. }) {
            return false;
        }
    }
    true
}

// ─── Diagnostics ─────────────────────────────────────────────────────────────

impl ExecutionPlan {
    /// Compute balance metrics for the plan.
    pub fn balance_report(&self) -> BalanceReport {
        let mut phase_reports = Vec::new();
        let mut total_atoms = 0u64;

        for (pi, phase) in self.phases.iter().enumerate() {
            let lane_atoms: Vec<u64> = phase
                .lane_work
                .iter()
                .map(|lane| lane.iter().map(|w| w.atom_count).sum())
                .collect();

            let max_load = lane_atoms.iter().copied().max().unwrap_or(0);
            let min_load = lane_atoms
                .iter()
                .copied()
                .filter(|&x| x > 0)
                .min()
                .unwrap_or(0);
            let phase_total: u64 = lane_atoms.iter().sum();
            total_atoms += phase_total;

            let imbalance = if min_load > 0 {
                max_load as f64 / min_load as f64
            } else if max_load > 0 {
                f64::INFINITY
            } else {
                1.0
            };

            phase_reports.push(PhaseBalance {
                phase_idx: pi,
                lane_atoms,
                max_load,
                min_load,
                imbalance,
            });
        }

        let worst_imbalance = phase_reports
            .iter()
            .map(|p| p.imbalance)
            .fold(1.0f64, f64::max);

        BalanceReport {
            num_phases: self.phases.len(),
            num_lanes: self.num_lanes,
            total_atoms,
            worst_imbalance,
            phase_reports,
        }
    }

    /// Verify that within each phase, no lane reads another lane's current-phase
    /// output. Returns a list of violations.
    pub fn verify_independence(
        &self,
        groups: &[AtomGroup],
        producers: &[Vec<usize>],
        is_literal: &[bool],
    ) -> Vec<String> {
        let mut errors = Vec::new();

        for (phase_idx, phase) in self.phases.iter().enumerate() {
            // Build: for each group in this phase, which lanes have work items for it.
            let mut group_to_lanes: HashMap<usize, HashSet<usize>> = HashMap::new();
            for (lane_idx, lane) in phase.lane_work.iter().enumerate() {
                for work in lane {
                    group_to_lanes
                        .entry(work.group_idx)
                        .or_default()
                        .insert(lane_idx);
                }
            }

            // For each group in this phase, check its producers.
            for (lane_idx, lane) in phase.lane_work.iter().enumerate() {
                for work in lane {
                    let gi = work.group_idx;
                    for &pi in &producers[gi] {
                        if is_literal[pi] {
                            continue; // Literals are shared, always OK.
                        }
                        if let Some(prod_lanes) = group_to_lanes.get(&pi) {
                            // Producer is in this phase. Check if it's on a different lane.
                            if prod_lanes.len() > 1 {
                                // Producer is split across multiple lanes — this is a
                                // cross-lane dependency within the phase.
                                errors.push(format!(
                                    "Phase {}: group {} on lane {} reads from split group {} (on lanes {:?})",
                                    phase_idx, gi, lane_idx, pi, prod_lanes
                                ));
                            } else if !prod_lanes.contains(&lane_idx) {
                                // Producer is on a single different lane.
                                errors.push(format!(
                                    "Phase {}: group {} on lane {} reads from group {} on lane {:?}",
                                    phase_idx, gi, lane_idx, pi, prod_lanes
                                ));
                            }
                        }
                        // Producer not in this phase — it's in an earlier phase, OK.
                    }
                }
            }
        }

        errors
    }
}

/// Balance metrics for the execution plan.
#[derive(Debug)]
pub struct BalanceReport {
    pub num_phases: usize,
    pub num_lanes: usize,
    pub total_atoms: u64,
    pub worst_imbalance: f64,
    pub phase_reports: Vec<PhaseBalance>,
}

/// Per-phase balance metrics.
#[derive(Debug)]
pub struct PhaseBalance {
    pub phase_idx: usize,
    pub lane_atoms: Vec<u64>,
    pub max_load: u64,
    pub min_load: u64,
    pub imbalance: f64,
}

impl std::fmt::Display for BalanceReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "ExecutionPlan: {} phases, {} lanes, {} total atoms",
            self.num_phases, self.num_lanes, self.total_atoms
        )?;
        writeln!(f, "Worst imbalance: {:.2}x", self.worst_imbalance)?;

        let mut sorted: Vec<&PhaseBalance> = self.phase_reports.iter().collect();
        sorted.sort_by(|a, b| b.imbalance.partial_cmp(&a.imbalance).unwrap());
        for pb in sorted.iter().take(5) {
            writeln!(
                f,
                "  Phase {}: max={} min={} imbalance={:.2}x lanes={:?}",
                pb.phase_idx, pb.max_load, pb.min_load, pb.imbalance, pb.lane_atoms
            )?;
        }
        Ok(())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::nano_graph::{InputRef, NanoGraph, ScalarBinOp, ScalarOp, ScalarUnaryOp};
    use crate::numeric_scalar::NumericScalar;

    /// Build a simple elementwise graph: one big group that should be split.
    fn build_elementwise_graph(count: u64) -> NanoGraph {
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
        let _c = g.push_group(
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
        g
    }

    /// Build a mini matmul graph: M rows, each with a Mul (count=K*N) and ReduceSum (count=N).
    fn build_matmul_graph(m: u64, k: u64, n: u64) -> NanoGraph {
        let mut g = NanoGraph::new();

        let b = g.push_group(
            k * n,
            ScalarOp::Literal(NumericScalar::F32(0.5)),
            vec![],
            vec![],
            vec![],
        );
        let a = g.push_group(
            m * k,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        for row in 0..m {
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
                        base: AtomId(a.0 + row * k),
                        stride: 1,
                        repeat: n,
                    },
                    InputRef::Affine { base: b, stride: 1 },
                ],
            );

            let _reduce = g.push_group(
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
                    base: mul_base,
                    stride: 1,
                }],
            );
        }
        g
    }

    /// Build a two-matmul chain: matmul1 → elementwise → matmul2.
    fn build_two_matmul_chain(m: u64, k: u64, n: u64) -> NanoGraph {
        let mut g = NanoGraph::new();

        // Matmul 1
        let b1 = g.push_group(
            k * n,
            ScalarOp::Literal(NumericScalar::F32(0.5)),
            vec![],
            vec![],
            vec![],
        );
        let a1 = g.push_group(
            m * k,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        let mut reduce1_bases = Vec::new();
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

            let reduce = g.push_group(
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
            reduce1_bases.push(reduce);
        }

        // Elementwise activation on matmul1 output — ONE big group.
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
                base: reduce1_bases[0],
                stride: 1,
            }],
        );

        // Matmul 2
        let b2 = g.push_group(
            n * n,
            ScalarOp::Literal(NumericScalar::F32(0.3)),
            vec![],
            vec![],
            vec![],
        );

        for row in 0..m {
            let mul2 = g.push_group(
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
                    InputRef::Affine {
                        base: b2,
                        stride: 1,
                    },
                ],
            );

            let _reduce2 = g.push_group(
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
                    base: mul2,
                    stride: 1,
                }],
            );
        }

        g
    }

    // ── Helper to verify plan invariants ──

    fn assert_plan_valid(graph: &NanoGraph, plan: &ExecutionPlan) {
        let groups = graph.groups();
        let (producers, _) = build_group_deps(groups);
        let is_literal: Vec<bool> = groups
            .iter()
            .map(|g| matches!(g.op, ScalarOp::Literal(_)))
            .collect();

        // 1. Coverage: every non-literal group has all atoms assigned.
        let mut group_atoms: HashMap<usize, u64> = HashMap::new();
        for phase in &plan.phases {
            for lane in &phase.lane_work {
                for work in lane {
                    *group_atoms.entry(work.group_idx).or_default() += work.atom_count;
                }
            }
        }
        for (gi, group) in groups.iter().enumerate() {
            if is_literal[gi] {
                continue;
            }
            let assigned = group_atoms.get(&gi).copied().unwrap_or(0);
            assert_eq!(
                assigned, group.count,
                "Group {} (count={}) has {} atoms in plan",
                gi, group.count, assigned
            );
        }

        // 2. No overlap: for each group, sub-ranges don't overlap.
        let mut group_ranges: HashMap<usize, Vec<(u64, u64)>> = HashMap::new();
        for phase in &plan.phases {
            for lane in &phase.lane_work {
                for work in lane {
                    group_ranges
                        .entry(work.group_idx)
                        .or_default()
                        .push((work.atom_offset, work.atom_offset + work.atom_count));
                }
            }
        }
        for (gi, ranges) in &group_ranges {
            let mut sorted = ranges.clone();
            sorted.sort();
            for i in 1..sorted.len() {
                assert!(
                    sorted[i].0 >= sorted[i - 1].1,
                    "Group {}: overlapping ranges {:?} and {:?}",
                    gi,
                    sorted[i - 1],
                    sorted[i]
                );
            }
        }

        // 3. Within-phase independence.
        let errors = plan.verify_independence(groups, &producers, &is_literal);
        assert!(
            errors.is_empty(),
            "Within-phase independence violations:\n{}",
            errors.join("\n")
        );
    }

    // ── Test: single elementwise group gets split ──

    #[test]
    fn test_elementwise_split_balance() {
        let g = build_elementwise_graph(49_152);
        let plan = plan_execution(&g, 8);

        assert_plan_valid(&g, &plan);

        let report = plan.balance_report();
        assert_eq!(
            report.num_phases, 1,
            "Expected 1 phase, got {}",
            report.num_phases
        );
        assert!(
            report.worst_imbalance < 1.01,
            "Imbalance {:.2}x too high (expected ~1.0x for evenly split group)",
            report.worst_imbalance
        );

        // Every lane should have work.
        let phase0 = &plan.phases[0];
        for (lane_idx, lane) in phase0.lane_work.iter().enumerate() {
            let atoms: u64 = lane.iter().map(|w| w.atom_count).sum();
            assert!(atoms > 0, "Lane {} has no work", lane_idx);
        }
    }

    // ── Test: coverage — all atoms are accounted for ──

    #[test]
    fn test_coverage() {
        let g = build_elementwise_graph(49_152);
        let plan = plan_execution(&g, 8);

        let total: u64 = plan
            .phases
            .iter()
            .flat_map(|p| p.lane_work.iter())
            .flat_map(|lane| lane.iter())
            .map(|w| w.atom_count)
            .sum();
        assert_eq!(total, 49_152);
    }

    // ── Test: matmul row distribution ──

    #[test]
    fn test_matmul_balance() {
        let g = build_matmul_graph(64, 32, 32);
        let plan = plan_execution(&g, 8);

        assert_plan_valid(&g, &plan);

        let report = plan.balance_report();
        for pb in &report.phase_reports {
            if pb.max_load > 0 && pb.min_load > 0 {
                assert!(
                    pb.imbalance < 2.0,
                    "Phase {} imbalance {:.2}x too high",
                    pb.phase_idx,
                    pb.imbalance
                );
            }
        }
    }

    // ── Test: two-matmul chain produces multiple phases and is balanced ──

    #[test]
    fn test_two_matmul_phases() {
        let g = build_two_matmul_chain(8, 16, 16);
        let plan = plan_execution(&g, 4);

        assert_plan_valid(&g, &plan);

        let report = plan.balance_report();
        assert!(
            report.num_phases >= 2,
            "Expected >= 2 phases for two-matmul chain, got {}",
            report.num_phases
        );

        for pb in &report.phase_reports {
            if pb.max_load > 0 && pb.min_load > 0 {
                assert!(
                    pb.imbalance < 2.0,
                    "Phase {} imbalance {:.2}x too high",
                    pb.phase_idx,
                    pb.imbalance
                );
            }
        }
    }

    // ── Test: within-phase independence verified via verify_independence ──

    #[test]
    fn test_within_phase_independence() {
        let g = build_two_matmul_chain(8, 16, 16);
        let plan = plan_execution(&g, 4);
        assert_plan_valid(&g, &plan);
    }

    // ── Test: no atoms lost ──

    #[test]
    fn test_no_atoms_lost_matmul() {
        let g = build_matmul_graph(8, 16, 16);
        let plan = plan_execution(&g, 4);
        assert_plan_valid(&g, &plan);
    }

    // ── Test: split sub-ranges don't overlap and cover the full group ──

    #[test]
    fn test_split_coverage() {
        let g = build_elementwise_graph(49_152);
        let plan = plan_execution(&g, 8);

        // Find all work items for the compute group (group index 2).
        let mut work_items: Vec<LaneWork> = Vec::new();
        for phase in &plan.phases {
            for lane in &phase.lane_work {
                for work in lane {
                    if work.group_idx == 2 {
                        work_items.push(*work);
                    }
                }
            }
        }

        work_items.sort_by_key(|w| w.atom_offset);

        let mut expected_offset = 0u64;
        for work in &work_items {
            assert_eq!(
                work.atom_offset, expected_offset,
                "Gap or overlap at offset {} (expected {})",
                work.atom_offset, expected_offset
            );
            expected_offset += work.atom_count;
        }
        assert_eq!(
            expected_offset, 49_152,
            "Total coverage {} != 49,152",
            expected_offset
        );
    }

    #[test]
    fn test_empty_graph() {
        let g = NanoGraph::new();
        let plan = plan_execution(&g, 8);
        assert!(plan.phases.is_empty());
    }

    #[test]
    fn test_all_literals() {
        let mut g = NanoGraph::new();
        g.push_group(
            1024,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let plan = plan_execution(&g, 4);
        assert!(plan.phases.is_empty());
    }

    #[test]
    fn test_reduce_split() {
        let mut g = NanoGraph::new();
        let input = g.push_group(
            1024 * 768,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        let _reduce = g.push_group(
            1024,
            ScalarOp::ReduceSum {
                reduce_count: 768,
                reduce_stride: 1024,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: input,
                stride: 1,
            }],
        );

        let plan = plan_execution(&g, 8);
        assert_plan_valid(&g, &plan);

        let report = plan.balance_report();
        assert!(
            report.worst_imbalance < 1.01,
            "ReduceSum split imbalance {:.2}x too high",
            report.worst_imbalance
        );
    }

    // ── Test: the critical two-matmul-with-elementwise case ──
    // This is the case that killed previous attempts: elementwise ops between
    // matmuls produce one big group that must be split, but downstream matmul
    // rows read from the split group.

    #[test]
    fn test_elementwise_between_matmuls_split() {
        // M=64 rows, K=32, N=32 — moderate size.
        let g = build_two_matmul_chain(64, 32, 32);
        let plan = plan_execution(&g, 8);

        assert_plan_valid(&g, &plan);

        let report = plan.balance_report();

        // With group splitting, the worst imbalance should be bounded.
        for pb in &report.phase_reports {
            if pb.max_load > 0 && pb.min_load > 0 {
                assert!(
                    pb.imbalance < 2.0,
                    "Phase {} imbalance {:.2}x too high. Lanes: {:?}",
                    pb.phase_idx,
                    pb.imbalance,
                    pb.lane_atoms
                );
            }
        }

        // The plan should have enough phases for two matmuls.
        assert!(
            report.num_phases >= 2,
            "Expected >= 2 phases, got {}",
            report.num_phases
        );
    }

    // ── Test: single-lane plan should work ──

    #[test]
    fn test_single_lane() {
        let g = build_matmul_graph(8, 16, 16);
        let plan = plan_execution(&g, 1);
        assert_plan_valid(&g, &plan);

        let report = plan.balance_report();
        assert_eq!(report.num_lanes, 1);
        // Single lane: imbalance is always 1.0.
        assert!(report.worst_imbalance <= 1.0);
    }

    // ── Test: chain of elementwise ops ──

    #[test]
    fn test_elementwise_chain() {
        // A → B → C, all elementwise, single large groups.
        let mut g = NanoGraph::new();
        let lit = g.push_group(
            10_000,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let neg = g.push_group(
            10_000,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
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
        let _exp = g.push_group(
            10_000,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: neg,
                stride: 1,
            }],
        );

        let plan = plan_execution(&g, 4);
        assert_plan_valid(&g, &plan);

        let report = plan.balance_report();
        // No reduces, so chains of elementwise. The key question: are Neg and
        // Exp in the same phase? They CAN be if Neg is split and Exp is split
        // the same way (same lane gets the same sub-range). But our algorithm
        // will detect that Exp reads from split Neg and insert a barrier.
        // That's correct — it trades one extra barrier for guaranteed independence.
        assert!(
            report.worst_imbalance < 2.0,
            "Imbalance {:.2}x too high",
            report.worst_imbalance
        );
    }

    // ── Test: matmul Mul and ReduceSum co-located on same lane ──

    #[test]
    fn test_matmul_mul_reduce_same_lane() {
        // 8 rows, each Mul -> ReduceSum. The ReduceSum must be on the same
        // lane as its Mul producer.
        let g = build_matmul_graph(8, 16, 16);
        let plan = plan_execution(&g, 4);
        assert_plan_valid(&g, &plan);

        // Build group-to-lane mapping.
        let groups = g.groups();
        let mut group_to_lane: HashMap<usize, usize> = HashMap::new();
        for phase in &plan.phases {
            for (lane_idx, lane) in phase.lane_work.iter().enumerate() {
                for work in lane {
                    group_to_lane.insert(work.group_idx, lane_idx);
                }
            }
        }

        // For each ReduceSum group, verify its Mul producer is on the same lane.
        let (producers, _) = build_group_deps(groups);
        for (gi, group) in groups.iter().enumerate() {
            if matches!(group.op, ScalarOp::ReduceSum { .. }) {
                let reduce_lane = group_to_lane.get(&gi);
                for &pi in &producers[gi] {
                    if matches!(groups[pi].op, ScalarOp::Literal(_)) {
                        continue;
                    }
                    let prod_lane = group_to_lane.get(&pi);
                    if let (Some(&rl), Some(&pl)) = (reduce_lane, prod_lane) {
                        assert_eq!(
                            rl, pl,
                            "ReduceSum group {} on lane {} but its producer {} on lane {}",
                            gi, rl, pi, pl
                        );
                    }
                }
            }
        }
    }

    // ── Test: many lanes, few groups ──

    #[test]
    fn test_more_lanes_than_groups() {
        // 2 compute groups, 16 lanes — should still work.
        let mut g = NanoGraph::new();
        let lit = g.push_group(
            100,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let _neg = g.push_group(
            100,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
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

        let plan = plan_execution(&g, 16);
        assert_plan_valid(&g, &plan);
    }

    // ── Test: Broadcast input ref doesn't prevent splitting ──

    #[test]
    fn test_broadcast_input_split() {
        let mut g = NanoGraph::new();
        let scalar = g.push_group(
            1,
            ScalarOp::Literal(NumericScalar::F32(42.0)),
            vec![],
            vec![],
            vec![],
        );
        let big = g.push_group(
            10_000,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let _add = g.push_group(
            10_000,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine {
                    base: big,
                    stride: 1,
                },
                InputRef::Broadcast(scalar),
            ],
        );

        let plan = plan_execution(&g, 4);
        assert_plan_valid(&g, &plan);

        let report = plan.balance_report();
        assert!(
            report.worst_imbalance < 1.01,
            "Broadcast input should not prevent balanced splitting: {:.2}x",
            report.worst_imbalance
        );
    }

    // ── Test: Modular input ref can be split ──

    #[test]
    fn test_modular_input_split() {
        let mut g = NanoGraph::new();
        let bias = g.push_group(
            768,
            ScalarOp::Literal(NumericScalar::F32(0.1)),
            vec![],
            vec![],
            vec![],
        );
        let input = g.push_group(
            49_152,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        // Add bias: atom i reads bias[i % 768].
        let _add = g.push_group(
            49_152,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine {
                    base: input,
                    stride: 1,
                },
                InputRef::Modular {
                    base: bias,
                    stride: 1,
                    modulus: 768,
                },
            ],
        );

        let plan = plan_execution(&g, 8);
        assert_plan_valid(&g, &plan);

        let report = plan.balance_report();
        assert!(
            report.worst_imbalance < 1.01,
            "Modular input should not prevent balanced splitting: {:.2}x",
            report.worst_imbalance
        );
    }
}
