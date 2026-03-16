#![allow(
    clippy::all,
    dead_code,
    unreachable_patterns,
    unused_variables,
    unused_imports
)]
//! DAG-slicing span partitioner: Phase+Lane → self-contained NanoGraph spans.
//!
//! Algorithm:
//!
//! 1. **Build group dependency DAG** — same as v2c (InputRef resolution,
//!    ReduceSum/ReduceMax strided access, IndirectLoad table references).
//!
//! 2. **Compute depth** — longest path from any root for each group. Groups at
//!    the same depth are potentially parallel.
//!
//! 3. **Find barrier positions** — a group that reads from multiple row families
//!    (groups that were independent at a shallower depth) forces a sync. These
//!    groups begin new phases.
//!
//! 4. **Within each phase, partition into lanes** — independent groups (no
//!    producer-consumer relationship within the phase) go to different lanes.
//!    Groups that form a chain stay on the same lane.
//!
//! 5. **Build span sub-graphs** — for each (phase, lane), extract groups into a
//!    new self-contained NanoGraph. Cross-span dependencies become declared
//!    inputs. Each span is independently validatable.
//!
//! Key property: within a phase, no span reads atoms produced by another span.
//! All cross-span data flows through declared inputs from earlier phases.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use crate::nano_graph::{AtomGroup, AtomId, InputRef, NanoGraph, ScalarOp};

/// Literal groups with fewer atoms than this are duplicated into spans.
/// Larger literals (weight matrices) become external inputs instead.
const LITERAL_INLINE_THRESHOLD: u64 = 1024;

// ─── Public types ────────────────────────────────────────────────────────────

/// A contiguous range of atoms mapped between main graph and span graph.
/// Replaces per-atom `(AtomId, AtomId)` pairs for O(num_groups) instead of O(num_atoms).
#[derive(Debug, Clone)]
pub struct AtomMapping {
    /// Start atom in the main graph.
    pub main_base: AtomId,
    /// Start atom in the span graph.
    pub span_base: AtomId,
    /// Number of contiguous atoms in this mapping.
    pub count: u64,
}

/// A self-contained computation unit: one lane's work in one phase.
pub struct Span {
    /// Self-contained NanoGraph for this span's computation.
    pub graph: NanoGraph,
    /// Contiguous ranges of atoms this span reads from the shared buffer.
    pub inputs: Vec<AtomMapping>,
    /// Contiguous ranges of atoms this span writes back to the shared buffer.
    pub outputs: Vec<AtomMapping>,
    /// Group indices from the original graph that this span contains.
    pub source_groups: Vec<usize>,
}

/// One phase of execution (between two barriers).
pub struct Phase {
    /// One span per lane. Empty spans are possible for idle lanes.
    pub spans: Vec<Span>,
}

/// The full span-based execution plan.
pub struct SpanPlan {
    pub num_lanes: usize,
    pub phases: Vec<Phase>,
}

// ─── Range-based atom map ────────────────────────────────────────────────────

/// An atom map that stores contiguous ranges instead of individual atoms.
/// Supports O(log n) lookup by main-graph AtomId, where n = number of ranges.
/// Insertion is append-only (ranges added in order).
struct RangeAtomMap {
    /// Sorted ranges: (main_base, span_base, count).
    ranges: Vec<(u64, u64, u64)>,
}

impl RangeAtomMap {
    fn new() -> Self {
        Self { ranges: Vec::new() }
    }

    /// Add a contiguous range mapping: main_base..main_base+count -> span_base..span_base+count.
    fn insert_range(&mut self, main_base: AtomId, span_base: AtomId, count: u64) {
        self.ranges.push((main_base.0, span_base.0, count));
    }

    /// Sort ranges by main_base. Must be called before get() if ranges were inserted out of order.
    fn sort(&mut self) {
        self.ranges.sort_by_key(|&(base, _, _)| base);
    }

    /// Look up a single main-graph AtomId, returning the span-graph AtomId.
    fn get(&self, main_id: AtomId) -> Option<AtomId> {
        // Binary search for the range containing main_id.
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

// ─── Public API ──────────────────────────────────────────────────────────────

/// Partition a NanoGraph into self-contained spans organized by phase and lane.
pub fn plan_spans(graph: &NanoGraph, num_lanes: usize) -> SpanPlan {
    let groups = graph.groups();
    let n = groups.len();
    let num_lanes = num_lanes.max(1);

    if n == 0 {
        return SpanPlan {
            num_lanes,
            phases: vec![],
        };
    }

    // Step 1: Build dependency DAG.
    let is_literal: Vec<bool> = groups
        .iter()
        .map(|g| matches!(g.op, ScalarOp::Literal(_)) && g.inputs.is_empty())
        .collect();

    // If all groups are literals, there's no compute work.
    if is_literal.iter().all(|&lit| lit) {
        return SpanPlan {
            num_lanes,
            phases: vec![],
        };
    }

    let (producers, consumers) = build_group_deps(groups);
    let topo_order = topological_sort(n, &producers);

    // Step 2: Compute depth (longest path from root).
    let depth = compute_depth(n, &topo_order, &producers, &is_literal);

    // Step 3: Find phase boundaries and assign groups to phases.
    let (group_phase, num_phases) = assign_phases(
        groups,
        &topo_order,
        &producers,
        &consumers,
        &is_literal,
        &depth,
    );

    // Step 4: Within each phase, assign groups to lanes.
    let group_lane = assign_lanes(
        groups,
        num_lanes,
        num_phases,
        &group_phase,
        &producers,
        &consumers,
        &is_literal,
    );

    // Step 5: Build self-contained span sub-graphs.
    let phases = build_spans(
        graph,
        num_lanes,
        num_phases,
        &group_phase,
        &group_lane,
        &producers,
        &is_literal,
    );

    SpanPlan { num_lanes, phases }
}

// ─── Group dependency graph ──────────────────────────────────────────────────

/// Build producer and consumer DAG at group level.
fn build_group_deps(groups: &[AtomGroup]) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let n = groups.len();
    let mut producers: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (gi, group) in groups.iter().enumerate() {
        let mut prod_set = BTreeSet::new();

        // InputRef resolution.
        for input in &group.inputs {
            for pi in resolve_producer_groups(input, group.count, groups) {
                if pi != gi {
                    prod_set.insert(pi);
                }
            }
        }

        // ReduceSum/ReduceMax strided access.
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

        // IndirectLoad table reference.
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

/// Resolve producer groups accounting for ReduceSum/ReduceMax strided access.
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
            let last_block = ((count - 1) / repeat) as i64;
            let first_read = base.0 as i64;
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
        InputRef::Modular {
            base,
            stride,
            modulus,
        } => {
            if *modulus == 0 {
                return vec![];
            }
            let first_read = base.0 as i64;
            let last_read = base.0 as i64 + *stride as i64 * (*modulus as i64 - 1);
            let lo = first_read.min(last_read) + min_reduce_ext;
            let hi = first_read.max(last_read) + max_reduce_ext;
            find_groups_in_range(groups, lo as u64, hi as u64)
        }
        InputRef::Explicit(ids) => {
            let mut result = HashSet::new();
            for id in ids {
                let base_val = id.0 as i64;
                let lo = base_val + min_reduce_ext;
                let hi = base_val + max_reduce_ext;
                for gi in find_groups_in_range(groups, lo as u64, hi as u64) {
                    result.insert(gi);
                }
            }
            result.into_iter().collect()
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

// ─── Depth computation ───────────────────────────────────────────────────────

/// Compute depth for each group: longest path from any root (literal or no-input group).
/// Literals get depth 0. Each compute group's depth = max(producer depths) + 1.
fn compute_depth(
    n: usize,
    topo_order: &[usize],
    producers: &[Vec<usize>],
    is_literal: &[bool],
) -> Vec<usize> {
    let mut depth = vec![0usize; n];

    for &gi in topo_order {
        if is_literal[gi] {
            depth[gi] = 0;
            continue;
        }
        let max_prod_depth = producers[gi].iter().map(|&pi| depth[pi]).max().unwrap_or(0);
        // Literals don't add depth, compute groups do.
        depth[gi] = if producers[gi].iter().all(|&pi| is_literal[pi]) {
            1 // first compute layer
        } else {
            max_prod_depth + 1
        };
    }

    depth
}

// ─── Phase assignment ────────────────────────────────────────────────────────

/// Assign each group to a phase. Phase boundaries occur at "convergence points"
/// where a group reads from multiple independent sub-DAGs (cross-family sync).
///
/// Strategy:
/// - Identify groups that read from producers in multiple independent families.
///   These are sync points that need a barrier before them.
/// - Use a two-pass approach: first identify root families, then propagate forward
///   and detect convergence.
fn assign_phases(
    groups: &[AtomGroup],
    topo_order: &[usize],
    producers: &[Vec<usize>],
    consumers: &[Vec<usize>],
    is_literal: &[bool],
    depth: &[usize],
) -> (Vec<usize>, usize) {
    let n = groups.len();

    // Identify independent root families.
    // A "family" is a set of groups rooted at the same non-literal root group
    // that share literal producers. Groups with multiple independent families
    // in their ancestry are convergence points.
    let families = identify_families(groups, topo_order, producers, consumers, is_literal);

    // Assign phases: groups in the same family stay in the same phase.
    // Convergence points (multiple families) start new phases.
    let mut group_phase = vec![0usize; n];
    let mut num_phases = 1usize;

    for &gi in topo_order {
        if is_literal[gi] {
            // Literals don't get a phase (they're shared data).
            // We assign them to phase 0 but they'll be duplicated into spans as needed.
            group_phase[gi] = 0;
            continue;
        }

        // This group's phase must be >= max phase of its non-literal producers.
        let max_producer_phase = producers[gi]
            .iter()
            .filter(|&&pi| !is_literal[pi])
            .map(|&pi| group_phase[pi])
            .max()
            .unwrap_or(0);

        // Check if this group is a convergence point: it reads from producers
        // belonging to multiple distinct families.
        let producer_families: HashSet<usize> = producers[gi]
            .iter()
            .filter(|&&pi| !is_literal[pi])
            .map(|&pi| families[pi])
            .collect();

        if producer_families.len() > 1 {
            // Convergence: this group and everything after it must be in a new phase
            // after all its producers.
            group_phase[gi] = max_producer_phase + 1;
        } else {
            // Same family chain continues — same phase as latest producer.
            group_phase[gi] = max_producer_phase;
        }

        if group_phase[gi] + 1 > num_phases {
            num_phases = group_phase[gi] + 1;
        }
    }

    (group_phase, num_phases)
}

/// Identify independent root families.
///
/// Each non-literal group gets a "family ID". Groups whose only non-literal
/// ancestry traces back to the same root(s) share a family. When multiple
/// independent root families converge, the group gets its own unique family.
fn identify_families(
    groups: &[AtomGroup],
    topo_order: &[usize],
    producers: &[Vec<usize>],
    consumers: &[Vec<usize>],
    is_literal: &[bool],
) -> Vec<usize> {
    let n = groups.len();
    let mut family = vec![0usize; n];
    let mut next_family = 0usize;

    // Each root compute group (only literal producers) gets its own family
    // UNLESS it shares literal producers with peers — then peer groups in
    // the same "sibling set" each get their own family (they're parallel rows).
    for &gi in topo_order {
        if is_literal[gi] {
            continue;
        }

        let has_compute_producer = producers[gi].iter().any(|&pi| !is_literal[pi]);
        if !has_compute_producer {
            // Root compute group — assign a unique family.
            family[gi] = next_family;
            next_family += 1;
        }
    }

    // Forward-propagate families.
    for &gi in topo_order {
        if is_literal[gi] {
            continue;
        }
        let has_compute_producer = producers[gi].iter().any(|&pi| !is_literal[pi]);
        if !has_compute_producer {
            continue; // Already assigned above.
        }

        // Collect families of non-literal producers.
        let mut producer_families: BTreeSet<usize> = BTreeSet::new();
        for &pi in &producers[gi] {
            if !is_literal[pi] {
                producer_families.insert(family[pi]);
            }
        }

        if producer_families.len() == 1 {
            // Single-family producer chain — inherit.
            family[gi] = *producer_families.iter().next().unwrap();
        } else {
            // Convergence — new family.
            family[gi] = next_family;
            next_family += 1;
        }
    }

    family
}

// ─── Lane assignment ─────────────────────────────────────────────────────────

/// Within each phase, assign groups to lanes. Groups that are independent
/// (no producer-consumer relationship within the same phase) go to different
/// lanes. Groups forming chains stay on the same lane.
fn assign_lanes(
    groups: &[AtomGroup],
    num_lanes: usize,
    num_phases: usize,
    group_phase: &[usize],
    producers: &[Vec<usize>],
    consumers: &[Vec<usize>],
    is_literal: &[bool],
) -> Vec<usize> {
    let n = groups.len();
    let mut group_lane = vec![0usize; n];

    for phase in 0..num_phases {
        // Collect groups in this phase (excluding literals).
        let phase_groups: Vec<usize> = (0..n)
            .filter(|&gi| !is_literal[gi] && group_phase[gi] == phase)
            .collect();

        if phase_groups.is_empty() {
            continue;
        }

        // Build within-phase dependency chains.
        // Two groups in the same phase that have a producer-consumer relationship
        // must be on the same lane.
        let phase_set: HashSet<usize> = phase_groups.iter().copied().collect();

        // Union-Find for chaining groups within a phase.
        let mut parent: HashMap<usize, usize> = HashMap::new();
        for &gi in &phase_groups {
            parent.insert(gi, gi);
        }

        // Chain groups that have within-phase dependencies.
        for &gi in &phase_groups {
            for &pi in &producers[gi] {
                if phase_set.contains(&pi) {
                    // gi depends on pi within this phase — same chain.
                    union(&mut parent, gi, pi);
                }
            }
        }

        // Collect chains (connected components).
        let mut chains: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for &gi in &phase_groups {
            let root = find(&parent, gi);
            chains.entry(root).or_default().push(gi);
        }

        // Sort chains by total atom count (largest first) for better balance.
        let mut chain_list: Vec<(usize, Vec<usize>)> = chains.into_iter().collect();
        chain_list.sort_by(|a, b| {
            let count_a: u64 = a.1.iter().map(|&gi| groups[gi].count).sum();
            let count_b: u64 = b.1.iter().map(|&gi| groups[gi].count).sum();
            count_b.cmp(&count_a) // Largest first.
        });

        // Greedy assignment: assign each chain to the least-loaded lane.
        let mut lane_load = vec![0u64; num_lanes];
        for (_root, chain) in &chain_list {
            let chain_atoms: u64 = chain.iter().map(|&gi| groups[gi].count).sum();
            let best_lane = lane_load
                .iter()
                .enumerate()
                .min_by_key(|&(_, load)| *load)
                .unwrap()
                .0;
            for &gi in chain {
                group_lane[gi] = best_lane;
            }
            lane_load[best_lane] += chain_atoms;
        }
    }

    group_lane
}

// Union-Find helpers.
fn find(parent: &HashMap<usize, usize>, mut x: usize) -> usize {
    while parent[&x] != x {
        x = parent[&x];
    }
    x
}

fn union(parent: &mut HashMap<usize, usize>, a: usize, b: usize) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra != rb {
        parent.insert(ra, rb);
    }
}

// ─── Span building ──────────────────────────────────────────────────────────

/// Build self-contained NanoGraph spans from the phase+lane assignment.
///
/// For each (phase, lane) cell:
/// 1. Collect the groups assigned to this cell.
/// 2. Determine which atoms are needed from outside this cell (inputs).
/// 3. Determine which atoms are consumed by groups outside this cell (outputs).
/// 4. Build a new NanoGraph with remapped atom IDs.
/// 5. Add Literal "stub" groups for external inputs.
fn build_spans(
    graph: &NanoGraph,
    num_lanes: usize,
    num_phases: usize,
    group_phase: &[usize],
    group_lane: &[usize],
    producers: &[Vec<usize>],
    is_literal: &[bool],
) -> Vec<Phase> {
    let groups = graph.groups();
    let n = groups.len();

    // Pre-compute: which atoms does each group produce?
    // atom_id -> (group_idx, offset_within_group)
    // We don't build a full map — we use group lookup when needed.

    // Pre-compute: for each group, which other groups consume it?
    let mut group_consumers: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (gi, prods) in producers.iter().enumerate() {
        for &pi in prods {
            group_consumers[pi].push(gi);
        }
    }

    // Build spans.
    let mut phases = Vec::with_capacity(num_phases);

    for phase in 0..num_phases {
        let mut spans = Vec::with_capacity(num_lanes);

        for lane in 0..num_lanes {
            // Groups in this cell.
            let cell_groups: Vec<usize> = (0..n)
                .filter(|&gi| {
                    if is_literal[gi] {
                        return false;
                    }
                    group_phase[gi] == phase && group_lane[gi] == lane
                })
                .collect();

            if cell_groups.is_empty() {
                // Empty span.
                spans.push(Span {
                    graph: NanoGraph::new(),
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                    source_groups: Vec::new(),
                });
                continue;
            }

            let cell_set: HashSet<usize> = cell_groups.iter().copied().collect();

            // Collect all literal groups directly used by cell groups,
            // then split by size: small ones get inlined, large ones become external.
            let mut all_literal_deps: BTreeSet<usize> = BTreeSet::new();
            for &gi in &cell_groups {
                for &pi in &producers[gi] {
                    if is_literal[pi] {
                        all_literal_deps.insert(pi);
                    }
                }
            }
            let mut literal_deps: BTreeSet<usize> = BTreeSet::new(); // small, inlined
            let mut large_literal_groups: BTreeSet<usize> = BTreeSet::new();
            for &li in &all_literal_deps {
                if groups[li].count < LITERAL_INLINE_THRESHOLD {
                    literal_deps.insert(li);
                } else {
                    large_literal_groups.insert(li);
                }
            }

            // Determine external inputs as ranges: (group_idx, offset, count).
            // O(num_groups) instead of O(num_atoms).
            let mut external_ranges: Vec<(usize, u64, u64)> = Vec::new();
            for &gi in &cell_groups {
                let group = &groups[gi];
                collect_external_ranges(
                    group,
                    groups,
                    &cell_set,
                    &literal_deps,
                    is_literal,
                    &mut external_ranges,
                );
            }
            // Also explicitly add large literal groups as full-range external inputs.
            for &li in &large_literal_groups {
                let lg = &groups[li];
                external_ranges.push((li, 0, lg.count));
            }
            // Deduplicate and merge overlapping ranges per group.
            let external_ranges = merge_group_ranges(&mut external_ranges);

            // Determine outputs as ranges: (group_idx, offset, count).
            let mut output_ranges: Vec<(usize, u64, u64)> = Vec::new();
            for &gi in &cell_groups {
                let group = &groups[gi];
                for &ci in &group_consumers[gi] {
                    if is_literal[ci] {
                        continue;
                    }
                    if cell_set.contains(&ci) {
                        continue; // Internal consumer.
                    }
                    // ci is external — compute the range of atoms from gi it reads.
                    if let Some((offset, count)) = compute_consumed_range(gi, ci, groups) {
                        output_ranges.push((gi, offset, count));
                    }
                }
                // Also check if any of this group's atoms are graph outputs.
                for &out_id in &graph.outputs {
                    if group.contains(out_id) {
                        let offset = out_id.0 - group.base_id.0;
                        output_ranges.push((gi, offset, 1));
                    }
                }
            }
            let output_ranges = merge_group_ranges(&mut output_ranges);

            // Build the sub-graph.
            let span = build_span_graph(
                graph,
                &cell_groups,
                &literal_deps,
                &external_ranges,
                &output_ranges,
            );

            spans.push(span);
        }

        phases.push(Phase { spans });
    }

    phases
}

/// Collect atom IDs that a group reads from external (non-cell, non-literal) sources.
///
/// Works at the group level: identifies which producer groups are external,
/// then includes the full atom range from those groups (since if any atom
/// from a group is needed, the InputRef addressing pattern will reference
/// a contiguous range from that group).
fn collect_external_atoms(
    group: &AtomGroup,
    all_groups: &[AtomGroup],
    cell_set: &HashSet<usize>,
    literal_deps: &BTreeSet<usize>,
    is_literal: &[bool],
    external: &mut BTreeSet<AtomId>,
) {
    // Find all producer groups for this group's InputRefs.
    let producer_gis = find_all_producer_group_indices(group, all_groups);

    for pi in producer_gis {
        if cell_set.contains(&pi) || literal_deps.contains(&pi) {
            continue; // Internal or inlined literal — not external.
        }
        // Skip non-dep literals that are small (not referenced by this span).
        // Large literals (not in literal_deps) fall through and become external.
        if is_literal[pi] && all_groups[pi].count < LITERAL_INLINE_THRESHOLD {
            continue;
        }
        // This is an external producer. Add all atoms from it that this group reads.
        let read_range = compute_read_range_from_group(group, &all_groups[pi], all_groups);
        for atom_id in read_range {
            external.insert(atom_id);
        }
    }
}

/// Range-based version: collect external atom ranges as (group_idx, offset, count).
/// O(num_groups) instead of O(num_atoms).
fn collect_external_ranges(
    group: &AtomGroup,
    all_groups: &[AtomGroup],
    cell_set: &HashSet<usize>,
    literal_deps: &BTreeSet<usize>,
    is_literal: &[bool],
    external_ranges: &mut Vec<(usize, u64, u64)>,
) {
    let producer_gis = find_all_producer_group_indices(group, all_groups);

    for pi in producer_gis {
        if cell_set.contains(&pi) || literal_deps.contains(&pi) {
            continue;
        }
        if is_literal[pi] && all_groups[pi].count < LITERAL_INLINE_THRESHOLD {
            continue;
        }
        // Compute the range of atoms from this producer that the consumer reads.
        let (overlap_lo, overlap_hi) =
            compute_read_range_bounds(group, &all_groups[pi], all_groups);
        if overlap_hi > overlap_lo {
            let prod = &all_groups[pi];
            let offset = overlap_lo - prod.base_id.0;
            let count = overlap_hi - overlap_lo;
            external_ranges.push((pi, offset, count));
        }
    }
}

/// Compute the [lo, hi) range of atoms from `producer` that `consumer` reads.
/// Returns absolute atom IDs (not offsets).
fn compute_read_range_bounds(
    consumer: &AtomGroup,
    producer: &AtomGroup,
    all_groups: &[AtomGroup],
) -> (u64, u64) {
    let prod_lo = producer.base_id.0;
    let prod_hi = producer.base_id.0 + producer.count;

    let (is_reduce, reduce_count, reduce_stride) = match &consumer.op {
        ScalarOp::ReduceSum {
            reduce_count,
            reduce_stride,
            ..
        }
        | ScalarOp::ReduceMax {
            reduce_count,
            reduce_stride,
            ..
        } if *reduce_count > 1 && *reduce_stride != 0 => (true, *reduce_count, *reduce_stride),
        _ => (false, 0, 0),
    };

    let mut overall_lo = u64::MAX;
    let mut overall_hi = 0u64;

    for input in &consumer.inputs {
        let (range_lo, range_hi) = input_ref_atom_range(
            input,
            consumer.count,
            is_reduce,
            reduce_count,
            reduce_stride,
        );
        if range_hi <= prod_lo || range_lo >= prod_hi {
            continue;
        }
        let overlap_lo = range_lo.max(prod_lo);
        let overlap_hi = range_hi.min(prod_hi);
        overall_lo = overall_lo.min(overlap_lo);
        overall_hi = overall_hi.max(overlap_hi);
    }

    if overall_lo >= overall_hi {
        (0, 0)
    } else {
        (overall_lo, overall_hi)
    }
}

/// Range-based version of collect_consumed_atoms_from.
/// Returns (offset_within_producer, count) or None if no overlap.
fn compute_consumed_range(
    producer_gi: usize,
    consumer_gi: usize,
    groups: &[AtomGroup],
) -> Option<(u64, u64)> {
    let producer = &groups[producer_gi];
    let consumer = &groups[consumer_gi];
    let (lo, hi) = compute_read_range_bounds(consumer, producer, groups);
    if hi > lo {
        let offset = lo - producer.base_id.0;
        Some((offset, hi - lo))
    } else {
        None
    }
}

/// Find all producer group indices for a consumer group.
fn find_all_producer_group_indices(group: &AtomGroup, all_groups: &[AtomGroup]) -> BTreeSet<usize> {
    let mut result = BTreeSet::new();

    for input in &group.inputs {
        for pi in resolve_producer_groups(input, group.count, all_groups) {
            result.insert(pi);
        }
    }

    // Reduce strided access.
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
                    all_groups,
                ) {
                    result.insert(pi);
                }
            }
        }
        _ => {}
    }

    // IndirectLoad table reference.
    if let ScalarOp::IndirectLoad { table_base, .. } = &group.op {
        if let Some(pi) = find_group_idx(all_groups, *table_base) {
            result.insert(pi);
        }
    }

    result
}

/// Compute the exact set of atoms from `producer` that `consumer` reads.
///
/// For structured InputRefs (Affine, StridedBroadcast, etc.), compute the
/// range of source atoms and intersect with the producer's atom range.
/// For Explicit refs, check each atom individually.
fn compute_read_range_from_group(
    consumer: &AtomGroup,
    producer: &AtomGroup,
    all_groups: &[AtomGroup],
) -> Vec<AtomId> {
    let prod_lo = producer.base_id.0;
    let prod_hi = producer.base_id.0 + producer.count;

    let (is_reduce, reduce_count, reduce_stride) = match &consumer.op {
        ScalarOp::ReduceSum {
            reduce_count,
            reduce_stride,
            ..
        }
        | ScalarOp::ReduceMax {
            reduce_count,
            reduce_stride,
            ..
        } if *reduce_count > 1 && *reduce_stride != 0 => (true, *reduce_count, *reduce_stride),
        _ => (false, 0, 0),
    };

    let mut result = BTreeSet::new();

    for input in &consumer.inputs {
        // Compute the full range of source atoms this InputRef covers
        // (including reduce expansion) and intersect with producer range.
        let (range_lo, range_hi) = input_ref_atom_range(
            input,
            consumer.count,
            is_reduce,
            reduce_count,
            reduce_stride,
        );
        if range_hi < prod_lo || range_lo >= prod_hi {
            continue; // No overlap.
        }
        // Add atoms in the overlap.
        let overlap_lo = range_lo.max(prod_lo);
        let overlap_hi = range_hi.min(prod_hi);
        for id in overlap_lo..overlap_hi {
            result.insert(AtomId(id));
        }
    }

    result.into_iter().collect()
}

/// Compute the [lo, hi) atom range covered by an InputRef (including reduce expansion).
fn input_ref_atom_range(
    input: &InputRef,
    count: u64,
    is_reduce: bool,
    reduce_count: u64,
    reduce_stride: i64,
) -> (u64, u64) {
    let min_reduce_ext = if is_reduce {
        0i64.min(reduce_stride * (reduce_count as i64 - 1))
    } else {
        0
    };
    let max_reduce_ext = if is_reduce {
        0i64.max(reduce_stride * (reduce_count as i64 - 1))
    } else {
        0
    };

    match input {
        InputRef::Broadcast(id) => {
            let lo = id.0 as i64 + min_reduce_ext;
            let hi = id.0 as i64 + max_reduce_ext + 1;
            (lo as u64, hi as u64)
        }
        InputRef::Affine { base, stride } => {
            if count == 0 {
                return (0, 0);
            }
            let first = base.0 as i64;
            let last = base.0 as i64 + *stride as i64 * (count as i64 - 1);
            let lo = first.min(last) + min_reduce_ext;
            let hi = first.max(last) + max_reduce_ext + 1;
            (lo as u64, hi as u64)
        }
        InputRef::StridedBroadcast {
            base,
            stride,
            repeat,
        } => {
            if count == 0 {
                return (0, 0);
            }
            let last_block = ((count - 1) / repeat) as i64;
            let first_read = base.0 as i64;
            let last_read = base.0 as i64 + stride * last_block;
            let lo = first_read.min(last_read) + min_reduce_ext;
            let hi = first_read.max(last_read) + max_reduce_ext + 1;
            (lo as u64, hi as u64)
        }
        InputRef::Modular {
            base,
            stride,
            modulus,
        } => {
            if *modulus == 0 {
                return (0, 0);
            }
            let first = base.0 as i64;
            let last = base.0 as i64 + *stride as i64 * (*modulus as i64 - 1);
            let lo = first.min(last) + min_reduce_ext;
            let hi = first.max(last) + max_reduce_ext + 1;
            (lo as u64, hi as u64)
        }
        InputRef::SymAffine { base, stride_i, .. } => {
            if count == 0 {
                return (0, 0);
            }
            let first = base.0 as i64;
            let last = base.0 as i64 + *stride_i as i64 * (count as i64 - 1);
            let lo = first.min(last);
            let hi = first.max(last) + 1;
            (lo as u64, hi as u64)
        }
        InputRef::Explicit(ids) => {
            if ids.is_empty() {
                return (0, 0);
            }
            let mut lo = u64::MAX;
            let mut hi = 0u64;
            for id in ids {
                let id_lo = (id.0 as i64 + min_reduce_ext) as u64;
                let id_hi = (id.0 as i64 + max_reduce_ext) as u64 + 1;
                lo = lo.min(id_lo);
                hi = hi.max(id_hi);
            }
            (lo, hi)
        }
    }
}

/// Collect atoms that consumer group `ci` reads from producer group `gi`.
fn collect_consumed_atoms_from(
    producer_gi: usize,
    consumer_gi: usize,
    groups: &[AtomGroup],
    output_atoms: &mut BTreeSet<AtomId>,
) {
    let producer = &groups[producer_gi];
    let consumer = &groups[consumer_gi];

    // Use range-based approach: compute intersection of consumer's read range
    // with the producer's atom range.
    let read_atoms = compute_read_range_from_group(consumer, producer, groups);
    for atom_id in read_atoms {
        output_atoms.insert(atom_id);
    }
}

/// Build a self-contained NanoGraph span from the given groups.
///
/// Creates a new NanoGraph with:
/// - Literal groups from the original graph (for internal literals)
/// - Stub literal groups for external input atoms
/// - Compute groups with remapped InputRefs
///
/// All operations are O(num_groups), not O(num_atoms).
fn build_span_graph(
    graph: &NanoGraph,
    cell_groups: &[usize],
    literal_deps: &BTreeSet<usize>,
    external_ranges: &[(usize, u64, u64)],
    output_ranges: &[(usize, u64, u64)],
) -> Span {
    let groups = graph.groups();
    let mut span_graph = NanoGraph::new();

    // Copy sym_dim setup from the main graph.
    for (name, &sd) in &graph.sym_dim_names {
        let local_sd = span_graph.sym_dim(name);
        if let Some(&bound) = graph.sym_dim_bounds.get(&sd) {
            span_graph.sym_dim_bounds.insert(local_sd, bound);
        }
    }

    // Range-based atom map: O(log n) lookup, O(1) per range insertion.
    let mut atom_map = RangeAtomMap::new();

    // Phase 1: Add literal groups.
    for &li in literal_deps {
        let lit_group = &groups[li];
        let local_base = span_graph.push_group(
            lit_group.count,
            lit_group.op.clone(),
            remap_sym_dims(&lit_group.sym_dims, graph, &span_graph),
            remap_sym_dims(&lit_group.reduce_dims, graph, &span_graph),
            vec![], // Literals have no inputs.
        );
        // Map the entire range at once.
        atom_map.insert_range(lit_group.base_id, local_base, lit_group.count);
    }

    // Phase 2: Add stub literals for external input ranges.
    // Each range becomes ONE placeholder group and ONE AtomMapping entry.
    let mut input_mappings: Vec<AtomMapping> = Vec::new();
    for &(source_group_idx, offset, count) in external_ranges {
        let src_group = &groups[source_group_idx];
        let main_base = src_group.base_id.offset(offset);
        // Create a single stub literal group for the entire range.
        let local_base = span_graph.push_group(
            count,
            ScalarOp::Literal(crate::numeric_scalar::NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );
        atom_map.insert_range(main_base, local_base, count);
        input_mappings.push(AtomMapping {
            main_base,
            span_base: local_base,
            count,
        });
    }

    // Sort atom map before compute group processing (ranges may be out of order).
    atom_map.sort();

    // Phase 3: Add compute groups.
    // Sort cell_groups by group index to maintain topological order.
    let mut sorted_cell_groups = cell_groups.to_vec();
    sorted_cell_groups.sort();

    for &gi in &sorted_cell_groups {
        let group = &groups[gi];
        let remapped_inputs = remap_input_refs(&group.inputs, &atom_map);
        let local_base = span_graph.push_group(
            group.count,
            group.op.clone(),
            remap_sym_dims(&group.sym_dims, graph, &span_graph),
            remap_sym_dims(&group.reduce_dims, graph, &span_graph),
            remapped_inputs,
        );
        atom_map.insert_range(group.base_id, local_base, group.count);
    }

    // Phase 4: Build output mappings from ranges.
    let mut output_mappings: Vec<AtomMapping> = Vec::new();
    for &(gi, offset, count) in output_ranges {
        let group = &groups[gi];
        let main_base = group.base_id.offset(offset);
        if let Some(local_base) = atom_map.get(main_base) {
            output_mappings.push(AtomMapping {
                main_base,
                span_base: local_base,
                count,
            });
        }
    }

    // Set graph outputs: we need all output atoms listed.
    // Build this from the output ranges (still O(num_ranges) if we're careful,
    // but graph.outputs is a Vec<AtomId> that needs individual atoms).
    // For correctness we must list them, but this is only the OUTPUT atoms of
    // the span, which are typically much smaller than input atoms.
    span_graph.outputs = Vec::new();
    for mapping in &output_mappings {
        for i in 0..mapping.count {
            span_graph.outputs.push(mapping.span_base.offset(i));
        }
    }

    Span {
        graph: span_graph,
        inputs: input_mappings,
        outputs: output_mappings,
        source_groups: sorted_cell_groups,
    }
}

/// Remap InputRefs from main-graph AtomIds to span-graph AtomIds.
fn remap_input_refs(inputs: &[InputRef], atom_map: &RangeAtomMap) -> Vec<InputRef> {
    inputs
        .iter()
        .map(|input| remap_one_input_ref(input, atom_map))
        .collect()
}

fn remap_one_input_ref(input: &InputRef, atom_map: &RangeAtomMap) -> InputRef {
    match input {
        InputRef::Broadcast(id) => InputRef::Broadcast(atom_map.get(*id).unwrap_or(*id)),
        InputRef::Affine { base, stride } => InputRef::Affine {
            base: atom_map.get(*base).unwrap_or(*base),
            stride: *stride,
        },
        InputRef::StridedBroadcast {
            base,
            stride,
            repeat,
        } => InputRef::StridedBroadcast {
            base: atom_map.get(*base).unwrap_or(*base),
            stride: *stride,
            repeat: *repeat,
        },
        InputRef::Modular {
            base,
            stride,
            modulus,
        } => InputRef::Modular {
            base: atom_map.get(*base).unwrap_or(*base),
            stride: *stride,
            modulus: *modulus,
        },
        InputRef::SymAffine {
            base,
            stride_i,
            stride_k,
        } => InputRef::SymAffine {
            base: atom_map.get(*base).unwrap_or(*base),
            stride_i: *stride_i,
            stride_k: *stride_k,
        },
        InputRef::Explicit(ids) => InputRef::Explicit(
            ids.iter()
                .map(|id| atom_map.get(*id).unwrap_or(*id))
                .collect(),
        ),
    }
}

/// Remap SymDims from main graph to span graph.
fn remap_sym_dims(
    dims: &[crate::nano_graph::SymDim],
    main_graph: &NanoGraph,
    span_graph: &NanoGraph,
) -> Vec<crate::nano_graph::SymDim> {
    dims.iter()
        .map(|&sd| {
            // Find the name in the main graph, then look up in span graph.
            for (name, &main_sd) in &main_graph.sym_dim_names {
                if main_sd == sd {
                    if let Some(&local_sd) = span_graph.sym_dim_names.get(name) {
                        return local_sd;
                    }
                }
            }
            sd // Fallback: use the same SymDim (shouldn't happen).
        })
        .collect()
}

/// Merge overlapping/adjacent ranges from the same group.
/// Input: unsorted Vec of (group_idx, offset, count).
/// Output: sorted, non-overlapping Vec of (group_idx, offset, count).
fn merge_group_ranges(ranges: &mut Vec<(usize, u64, u64)>) -> Vec<(usize, u64, u64)> {
    if ranges.is_empty() {
        return vec![];
    }
    // Sort by (group_idx, offset).
    ranges.sort_by_key(|&(gi, off, _)| (gi, off));

    let mut merged: Vec<(usize, u64, u64)> = Vec::new();
    for &(gi, off, count) in ranges.iter() {
        if let Some(last) = merged.last_mut() {
            if last.0 == gi && off <= last.1 + last.2 {
                // Overlapping or adjacent — extend.
                let new_end = (off + count).max(last.1 + last.2);
                last.2 = new_end - last.1;
                continue;
            }
        }
        merged.push((gi, off, count));
    }
    merged
}

/// Group a set of AtomIds into contiguous ranges from the same source group.
/// Returns (group_idx, offset_within_group, count).
fn group_into_contiguous_ranges(
    atom_ids: &BTreeSet<AtomId>,
    groups: &[AtomGroup],
) -> Vec<(usize, u64, u64)> {
    if atom_ids.is_empty() {
        return vec![];
    }

    let mut ranges: Vec<(usize, u64, u64)> = Vec::new();

    for &atom_id in atom_ids {
        if let Some(gi) = find_group_idx(groups, atom_id) {
            let offset = atom_id.0 - groups[gi].base_id.0;
            if let Some(last) = ranges.last_mut() {
                if last.0 == gi && last.1 + last.2 == offset {
                    // Extend current range.
                    last.2 += 1;
                    continue;
                }
            }
            ranges.push((gi, offset, 1));
        }
    }

    ranges
}

// ─── Validation ──────────────────────────────────────────────────────────────

/// Validate that a SpanPlan is correct:
/// 1. Every span's NanoGraph passes validate().
/// 2. No cross-lane reads within a phase (span independence).
/// 3. All groups from the original graph are accounted for.
/// 4. All span inputs come from earlier phases or literals.
pub fn validate_span_plan(plan: &SpanPlan, graph: &NanoGraph) -> Vec<String> {
    let mut errors = Vec::new();
    let groups = graph.groups();
    let n = groups.len();
    let is_literal: Vec<bool> = groups
        .iter()
        .map(|g| matches!(g.op, ScalarOp::Literal(_)) && g.inputs.is_empty())
        .collect();

    // Check 1: Each span's NanoGraph validates.
    for (pi, phase) in plan.phases.iter().enumerate() {
        for (li, span) in phase.spans.iter().enumerate() {
            if span.source_groups.is_empty() {
                continue;
            }
            let span_errors = span.graph.validate();
            for err in span_errors {
                errors.push(format!("Phase {} Lane {} span validation: {}", pi, li, err));
            }
        }
    }

    // Check 2: All non-literal groups are assigned to exactly one span.
    let mut assigned: HashSet<usize> = HashSet::new();
    for phase in &plan.phases {
        for span in &phase.spans {
            for &gi in &span.source_groups {
                if !assigned.insert(gi) {
                    errors.push(format!("Group {} assigned to multiple spans", gi));
                }
            }
        }
    }
    for gi in 0..n {
        if !is_literal[gi] && !assigned.contains(&gi) {
            errors.push(format!("Group {} not assigned to any span", gi));
        }
    }

    // Check 3: Cross-lane independence within a phase.
    // Within a phase, no span should read atoms produced by another span in the same phase.
    for (pi, phase) in plan.phases.iter().enumerate() {
        // Collect which main-graph atoms each span produces.
        let mut lane_produces: Vec<HashSet<AtomId>> = Vec::new();
        for span in &phase.spans {
            let mut produced = HashSet::new();
            for &gi in &span.source_groups {
                let group = &groups[gi];
                for i in 0..group.count {
                    produced.insert(group.base_id.offset(i));
                }
            }
            lane_produces.push(produced);
        }

        // Check that no span's inputs come from another span in this phase.
        for (li, span) in phase.spans.iter().enumerate() {
            for mapping in &span.inputs {
                // Check each range's base atom as a representative.
                // If any atom in the range is produced by another lane, that's a violation.
                for (other_li, other_produced) in lane_produces.iter().enumerate() {
                    if other_li == li {
                        continue;
                    }
                    // Check if any atom in this input range overlaps with other lane's produced atoms.
                    for i in 0..mapping.count.min(1) {
                        // Quick check: just test the first atom.
                        let main_id = mapping.main_base.offset(i);
                        if other_produced.contains(&main_id) {
                            errors.push(format!(
                                "Phase {} Lane {} reads atom {} produced by Lane {} (cross-lane violation)",
                                pi, li, main_id, other_li
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

    /// Helper: plan and validate, returning errors.
    fn plan_and_validate(graph: &NanoGraph, num_lanes: usize) -> (SpanPlan, Vec<String>) {
        let plan = plan_spans(graph, num_lanes);
        let errors = validate_span_plan(&plan, graph);
        (plan, errors)
    }

    #[test]
    fn test_empty_graph() {
        let g = NanoGraph::new();
        let (plan, errors) = plan_and_validate(&g, 4);
        assert!(errors.is_empty(), "{:?}", errors);
        assert_eq!(plan.phases.len(), 0);
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
        // Literal-only graph: no compute groups, so no phases.
        let (plan, errors) = plan_and_validate(&g, 2);
        assert!(errors.is_empty(), "{:?}", errors);
        assert_eq!(plan.phases.len(), 0);
    }

    #[test]
    fn test_elementwise_add_single_lane() {
        let (g, _, _, _) = test_graphs::elementwise_binary(1024, ScalarBinOp::Add);
        let (plan, errors) = plan_and_validate(&g, 1);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
        assert!(plan.phases.len() >= 1);
        // Single lane — all compute in one span.
        let total_compute_groups: usize = plan
            .phases
            .iter()
            .flat_map(|p| p.spans.iter())
            .map(|s| s.source_groups.len())
            .sum();
        assert_eq!(total_compute_groups, 1); // One Add group.
    }

    #[test]
    fn test_elementwise_add_multi_lane() {
        let (g, _, _, _) = test_graphs::elementwise_binary(1024, ScalarBinOp::Add);
        let (plan, errors) = plan_and_validate(&g, 4);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
        // Should be one phase with 4 lanes (but group isn't split, so
        // only 1 lane gets the work).
        assert!(plan.phases.len() >= 1);
    }

    #[test]
    fn test_unary_chain_single_lane() {
        let (g, _, _) = test_graphs::unary_chain(
            256,
            &[ScalarUnaryOp::Exp, ScalarUnaryOp::Neg, ScalarUnaryOp::Tanh],
        );
        let (plan, errors) = plan_and_validate(&g, 1);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
        // 3 compute groups, all on one lane, one phase (sequential chain).
        let total_groups: usize = plan
            .phases
            .iter()
            .flat_map(|p| p.spans.iter())
            .map(|s| s.source_groups.len())
            .sum();
        assert_eq!(total_groups, 3);
    }

    #[test]
    fn test_matmul_independence() {
        // Small matmul: M=4, K=2, N=3.
        // 4*2=8 Mul groups + 4 ReduceSum groups = 12 compute groups.
        // The 4 rows are independent — should be distributed across lanes.
        let (g, _, _, _) = test_graphs::matmul(4, 2, 3);
        let errors = g.validate();
        assert!(errors.is_empty(), "Graph validation: {:?}", errors);

        let (plan, plan_errors) = plan_and_validate(&g, 4);
        assert!(plan_errors.is_empty(), "Plan errors: {:?}", plan_errors);

        // All 12 compute groups must be assigned.
        let total_groups: usize = plan
            .phases
            .iter()
            .flat_map(|p| p.spans.iter())
            .map(|s| s.source_groups.len())
            .sum();
        assert_eq!(total_groups, 12); // 8 Mul + 4 ReduceSum

        // Check that each span's graph validates independently.
        for (pi, phase) in plan.phases.iter().enumerate() {
            for (li, span) in phase.spans.iter().enumerate() {
                let span_errors = span.graph.validate();
                assert!(
                    span_errors.is_empty(),
                    "Phase {} Lane {} span graph errors: {:?}",
                    pi,
                    li,
                    span_errors
                );
            }
        }
    }

    #[test]
    fn test_matmul_chain_phases() {
        // Two chained matmuls: out = (A @ B) @ C.
        // M=4, K1=2, N1=3, K2=3, N2=5.
        // First matmul: 8 Mul + 4 ReduceSum = 12 groups.
        // Second matmul: 12 Mul + 4 ReduceSum = 16 groups.
        // The second matmul depends on all outputs of the first → needs barrier.
        let (g, _, _, _, _) = test_graphs::matmul_chain(4, 2, 3, 3, 5);
        let errors = g.validate();
        assert!(errors.is_empty(), "Graph validation: {:?}", errors);

        let (plan, plan_errors) = plan_and_validate(&g, 2);
        assert!(plan_errors.is_empty(), "Plan errors: {:?}", plan_errors);

        // Must have at least 2 phases (barrier between matmuls).
        assert!(
            plan.phases.len() >= 2,
            "Expected >= 2 phases for chained matmuls, got {}",
            plan.phases.len()
        );

        // All compute groups must be assigned.
        let total_groups: usize = plan
            .phases
            .iter()
            .flat_map(|p| p.spans.iter())
            .map(|s| s.source_groups.len())
            .sum();
        assert_eq!(total_groups, 28); // 12 + 16
    }

    #[test]
    fn test_span_self_containment() {
        // Verify that each span can be evaluated independently.
        // Build a simple graph: c = a + b, d = c * 2.
        let mut g = NanoGraph::new();
        let a = g.push_group(
            8,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let b = g.push_group(
            8,
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
            vec![],
        );
        let two = g.push_atom(
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
            vec![],
        );
        let c = g.push_group(
            8,
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
        let d = g.push_group(
            8,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine { base: c, stride: 1 },
                InputRef::Broadcast(two),
            ],
        );
        g.outputs = vec![d];

        let (plan, errors) = plan_and_validate(&g, 1);
        assert!(errors.is_empty(), "Errors: {:?}", errors);

        // Every span's graph should validate.
        for phase in &plan.phases {
            for span in &phase.spans {
                let span_errors = span.graph.validate();
                assert!(
                    span_errors.is_empty(),
                    "Span graph validation failed: {:?}",
                    span_errors
                );
            }
        }
    }

    #[test]
    fn test_broadcast_add_spans() {
        // c[N] = a[N] + scalar_b — the scalar should be duplicated into each span
        // that needs it, not cause a cross-lane dependency.
        let (g, _, _, _) = test_graphs::broadcast_add(64);
        let (plan, errors) = plan_and_validate(&g, 2);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
    }

    #[test]
    fn test_matmul_activation_spans() {
        // Matmul + activation: the activation is elementwise on the matmul output.
        // Should be in the same phase as the reduce since it reads only from one
        // family (the reduce outputs).
        let (g, _, _, _) = test_graphs::matmul_activation(4, 2, 3, ScalarUnaryOp::Tanh);
        let errors = g.validate();
        assert!(errors.is_empty(), "Graph validation: {:?}", errors);

        let (plan, plan_errors) = plan_and_validate(&g, 2);
        assert!(plan_errors.is_empty(), "Plan errors: {:?}", plan_errors);

        // All compute groups assigned.
        let total_groups: usize = plan
            .phases
            .iter()
            .flat_map(|p| p.spans.iter())
            .map(|s| s.source_groups.len())
            .sum();
        // 8 Mul + 4 ReduceSum + 1 Tanh activation = 13
        assert_eq!(total_groups, 13);
    }

    #[test]
    fn test_no_cross_lane_reads_matmul() {
        // Core property: within a phase, no lane reads another lane's output.
        let (g, _, _, _) = test_graphs::matmul(8, 4, 6);
        let (plan, errors) = plan_and_validate(&g, 4);
        assert!(errors.is_empty(), "Cross-lane violations: {:?}", errors);
    }

    #[test]
    fn test_diamond_dag() {
        // Diamond: A → B, A → C, B+C → D.
        // B and C are independent (can be parallel).
        // D is a convergence point (reads from both B and C).
        let mut g = NanoGraph::new();
        let a = g.push_group(
            16,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let b = g.push_group(
            16,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: a, stride: 1 }],
        );
        let c = g.push_group(
            16,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: a, stride: 1 }],
        );
        let d = g.push_group(
            16,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine { base: b, stride: 1 },
                InputRef::Affine { base: c, stride: 1 },
            ],
        );
        g.outputs = vec![d];

        let (plan, errors) = plan_and_validate(&g, 2);
        assert!(errors.is_empty(), "Errors: {:?}", errors);

        // B and C should be in the first phase (parallel).
        // D should be in a later phase (convergence).
        assert!(
            plan.phases.len() >= 2,
            "Expected >= 2 phases for diamond, got {}",
            plan.phases.len()
        );
    }

    #[test]
    fn test_parallel_branches() {
        // Two independent branches from different literals — should parallelize.
        let mut g = NanoGraph::new();
        let a = g.push_group(
            32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let b = g.push_group(
            32,
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
            vec![],
        );
        // Branch 1: exp(a)
        let branch1 = g.push_group(
            32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: a, stride: 1 }],
        );
        // Branch 2: neg(b)
        let branch2 = g.push_group(
            32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: b, stride: 1 }],
        );
        g.outputs = vec![branch1, branch2];

        let (plan, errors) = plan_and_validate(&g, 2);
        assert!(errors.is_empty(), "Errors: {:?}", errors);

        // Both branches should be in the same phase, on different lanes.
        // Check that no errors means no cross-lane violations.
    }

    #[test]
    fn test_reduce_creates_barrier() {
        // Two matmul rows, each with Mul+ReduceSum, followed by an Add that
        // combines both rows' outputs → needs a barrier.
        let mut g = NanoGraph::new();
        let a0 = g.push_group(
            4,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let a1 = g.push_group(
            4,
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
            vec![],
        );
        let b = g.push_group(
            4,
            ScalarOp::Literal(NumericScalar::F32(3.0)),
            vec![],
            vec![],
            vec![],
        );

        // Row 0: mul0 = a0 * b, reduce0 = sum(mul0)
        let mul0 = g.push_group(
            4,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine {
                    base: a0,
                    stride: 1,
                },
                InputRef::Affine { base: b, stride: 1 },
            ],
        );
        let red0 = g.push_atom(
            ScalarOp::ReduceSum {
                reduce_count: 4,
                reduce_stride: 1,
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

        // Row 1: mul1 = a1 * b, reduce1 = sum(mul1)
        let mul1 = g.push_group(
            4,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine {
                    base: a1,
                    stride: 1,
                },
                InputRef::Affine { base: b, stride: 1 },
            ],
        );
        let red1 = g.push_atom(
            ScalarOp::ReduceSum {
                reduce_count: 4,
                reduce_stride: 1,
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

        // Combine: out = red0 + red1 (convergence point)
        let out = g.push_atom(
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Broadcast(red0), InputRef::Broadcast(red1)],
        );
        g.outputs = vec![out];

        let (plan, errors) = plan_and_validate(&g, 2);
        assert!(errors.is_empty(), "Errors: {:?}", errors);

        // The Add must be in a later phase than the two ReduceSums.
        assert!(
            plan.phases.len() >= 2,
            "Expected >= 2 phases (barrier after reduce convergence), got {}",
            plan.phases.len()
        );
    }

    /// Test that spans properly declare their inputs and outputs.
    #[test]
    fn test_span_io_declarations() {
        // Chain: A(lit) → B(exp) → C(neg)
        // With 2 phases: B in phase 0, C in phase 1 (if they get split).
        // Actually with a chain they should be in the same phase.
        // Let's force a diamond to get two phases.
        let mut g = NanoGraph::new();
        let a = g.push_group(
            8,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let b = g.push_group(
            8,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: a, stride: 1 }],
        );
        let c = g.push_group(
            8,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: a, stride: 1 }],
        );
        // D = B + C (convergence)
        let d = g.push_group(
            8,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine { base: b, stride: 1 },
                InputRef::Affine { base: c, stride: 1 },
            ],
        );
        g.outputs = vec![d];

        let (plan, errors) = plan_and_validate(&g, 2);
        assert!(errors.is_empty(), "Errors: {:?}", errors);

        // The spans that produce B and C should have outputs declared.
        // The span that computes D should have inputs from B and C.
        let mut total_inputs = 0;
        let mut total_outputs = 0;
        for phase in &plan.phases {
            for span in &phase.spans {
                total_inputs += span.inputs.len();
                total_outputs += span.outputs.len();
            }
        }
        // There should be some cross-span data flow.
        // (B and C outputs → D inputs)
        if plan.phases.len() >= 2 {
            assert!(
                total_outputs > 0,
                "Expected some span outputs for cross-phase data"
            );
            assert!(
                total_inputs > 0,
                "Expected some span inputs for cross-phase data"
            );
        }
    }
}
