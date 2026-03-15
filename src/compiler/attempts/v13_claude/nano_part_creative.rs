#![allow(clippy::all, dead_code, unreachable_patterns)]
//! NanoGraph partitioner: Chain-based DAG partitioning with phase detection.
//!
//! Algorithm:
//!
//! 1. **Build group DAG**: For each group, find producer groups via InputRef
//!    resolution. Literal groups are "data" (shared across kernels).
//!
//! 2. **Extract independent chains**: Walk backward from each "sink" group
//!    (groups with no consumers within the phase) following exclusive
//!    producer-consumer relationships. A chain is a maximal set of groups
//!    connected by exclusive (fan-out=1) edges. Chains are the atomic units
//!    of parallelism: each chain's groups must stay together.
//!
//! 3. **Phase detection via live-set pinch points**: Sweep topo order tracking
//!    live intermediate atom counts. Positions where liveness is minimal are
//!    natural phase boundaries (e.g., between transformer layers).
//!
//! 4. **Within-phase parallel distribution**: Within each phase, chains that
//!    don't depend on each other are distributed round-robin across parallel
//!    kernels. Chains that DO have dependencies are placed together.
//!
//! 5. **Acyclicity enforcement**: Build the kernel dependency graph and verify
//!    it's a DAG. Merge any kernels involved in cycles.
//!
//! Key properties:
//! - Kernels do NOT require contiguous group index ranges
//! - Parallel kernels have interleaved group indices
//! - Groups connected by exclusive edges stay together (chains)
//! - Phase boundaries correspond to natural computation boundaries
//! - Matmul rows are independent chains -> parallel kernels

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

use crate::nano_graph::{AtomGroup, AtomId, InputRef, NanoGraph, ScalarOp};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Result of partitioning a NanoGraph into kernels.
#[derive(Debug)]
pub struct NanoPartitionResult {
    /// kernel_idx -> group indices (into NanoGraph::groups()).
    pub kernel_groups: Vec<Vec<usize>>,
    pub num_kernels: usize,
}

/// Partition a NanoGraph into roughly `target_kernels` kernels.
pub fn partition_nanograph(graph: &NanoGraph, target_kernels: usize) -> NanoPartitionResult {
    let groups = graph.groups();
    let n = groups.len();

    if n == 0 {
        return NanoPartitionResult {
            kernel_groups: vec![],
            num_kernels: 0,
        };
    }

    let target_kernels = target_kernels.max(1);

    if n <= target_kernels {
        let kernel_groups: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        return NanoPartitionResult {
            num_kernels: kernel_groups.len(),
            kernel_groups,
        };
    }

    // Step 1: Classify groups and build dependency graph.
    let is_data: Vec<bool> = groups
        .iter()
        .map(|g| matches!(g.op, ScalarOp::Literal(_)) && g.inputs.is_empty())
        .collect();

    let producers = build_producer_graph(groups, &is_data);
    let consumers = build_consumer_graph(n, &producers);

    // Compute fan-out for each group: how many distinct compute groups consume it.
    let fan_out: Vec<usize> = consumers
        .iter()
        .enumerate()
        .map(|(gi, cons)| {
            cons.iter()
                .filter(|&&ci| !is_data[ci])
                .count()
        })
        .collect();

    // Step 2: Extract chains.
    let chains = extract_chains(n, &producers, &consumers, &fan_out, &is_data);

    // Step 3: Build chain dependency graph.
    // chain_deps[i] = set of chain indices that chain i depends on.
    let group_to_chain = build_group_to_chain_map(n, &chains);
    let chain_deps = build_chain_deps(&chains, &producers, &group_to_chain, &is_data);

    // Step 4: Detect phase boundaries.
    let topo_order = topological_sort(n, &producers);
    let phases = detect_and_split_phases(
        &topo_order,
        groups,
        &consumers,
        &is_data,
        target_kernels,
    );

    // Step 5: Assign each chain to exactly one phase (by its sink group's position).
    // Then distribute chains across parallel kernels within each phase.
    let mut kernel_groups: Vec<Vec<usize>> = Vec::new();

    // Build topo_pos for phase membership.
    let mut topo_pos = vec![0usize; n];
    for (pos, &gi) in topo_order.iter().enumerate() {
        topo_pos[gi] = pos;
    }

    // Determine which phase each chain belongs to: use the topo position of
    // the chain's last (sink) group.
    let phase_boundaries: Vec<usize> = {
        let mut bounds = Vec::new();
        let mut end = 0;
        for phase in &phases {
            end += phase.len();
            bounds.push(end);
        }
        bounds
    };

    let chain_phase: Vec<usize> = chains
        .iter()
        .map(|chain| {
            // Use the maximum topo_pos of any group in the chain.
            let max_pos = chain.iter().map(|&gi| topo_pos[gi]).max().unwrap_or(0);
            // Find which phase this position falls in.
            phase_boundaries
                .iter()
                .position(|&end| max_pos < end)
                .unwrap_or(phases.len().saturating_sub(1))
        })
        .collect();

    // For each phase, gather its chains and distribute them.
    for (phase_idx, phase) in phases.iter().enumerate() {
        let phase_chain_ids: Vec<usize> = (0..chains.len())
            .filter(|&ci| chain_phase[ci] == phase_idx)
            .collect();

        if phase_chain_ids.is_empty() {
            // Phase has only data groups.
            let data_in_phase: Vec<usize> = phase
                .iter()
                .copied()
                .filter(|&gi| is_data[gi])
                .collect();
            if !data_in_phase.is_empty() {
                kernel_groups.push(data_in_phase);
            }
            continue;
        }

        let local_chain_set: HashSet<usize> = phase_chain_ids.iter().copied().collect();

        let mut chain_local_deps: HashMap<usize, HashSet<usize>> = HashMap::new();
        for &ci in &phase_chain_ids {
            let deps: HashSet<usize> = chain_deps[ci]
                .iter()
                .copied()
                .filter(|d| local_chain_set.contains(d))
                .collect();
            chain_local_deps.insert(ci, deps);
        }

        let chain_topo = topo_sort_chains(&phase_chain_ids, &chain_local_deps);

        let parallel_slots = compute_parallel_slots(
            &chain_topo,
            &chain_local_deps,
            target_kernels,
        );

        let num_slots = parallel_slots.iter().copied().max().unwrap_or(0) + 1;
        let mut slot_groups: Vec<Vec<usize>> = vec![Vec::new(); num_slots];

        for (&ci, &slot) in chain_topo.iter().zip(parallel_slots.iter()) {
            // Include ALL groups from this chain (not filtered by phase_set).
            for &gi in &chains[ci] {
                slot_groups[slot].push(gi);
            }
        }

        // Assign data groups in this phase to the slot that consumes them most.
        let phase_set: HashSet<usize> = phase.iter().copied().collect();
        let data_in_phase: Vec<usize> = phase
            .iter()
            .copied()
            .filter(|&gi| is_data[gi] && group_to_chain.get(&gi).is_none())
            .collect();

        for &di in &data_in_phase {
            let mut slot_votes: HashMap<usize, usize> = HashMap::new();
            for &ci in &consumers[di] {
                if let Some(&chain_id) = group_to_chain.get(&ci) {
                    if let Some(pos) = chain_topo.iter().position(|&c| c == chain_id) {
                        *slot_votes.entry(parallel_slots[pos]).or_default() += 1;
                    }
                }
            }
            let best_slot = slot_votes
                .iter()
                .max_by_key(|&(_, &count)| count)
                .map(|(&slot, _)| slot)
                .unwrap_or(0);
            if best_slot < slot_groups.len() {
                slot_groups[best_slot].push(di);
            } else if !slot_groups.is_empty() {
                slot_groups[0].push(di);
            }
        }

        for mut sg in slot_groups {
            if !sg.is_empty() {
                sg.sort();
                sg.dedup();
                kernel_groups.push(sg);
            }
        }
    }

    // Handle any groups not yet assigned (shouldn't happen, but defensive).
    let mut assigned: HashSet<usize> = HashSet::new();
    for kg in &kernel_groups {
        for &gi in kg {
            assigned.insert(gi);
        }
    }
    let unassigned: Vec<usize> = (0..n).filter(|gi| !assigned.contains(gi)).collect();
    if !unassigned.is_empty() {
        kernel_groups.push(unassigned);
    }

    // Step 6: Enforce acyclicity.
    enforce_acyclicity(&mut kernel_groups, groups);

    // Balance: split oversized kernels.
    balance_kernels(&mut kernel_groups, groups, target_kernels);

    // Final acyclicity check after balancing.
    enforce_acyclicity(&mut kernel_groups, groups);

    // Remove empty kernels.
    kernel_groups.retain(|k| !k.is_empty());

    NanoPartitionResult {
        num_kernels: kernel_groups.len(),
        kernel_groups,
    }
}

// ---------------------------------------------------------------------------
// Dependency graph construction
// ---------------------------------------------------------------------------

/// For each group, find the set of producer group indices (all, including data).
fn build_producer_graph(groups: &[AtomGroup], _is_data: &[bool]) -> Vec<Vec<usize>> {
    let n = groups.len();
    let mut producers: Vec<Vec<usize>> = Vec::with_capacity(n);

    for (gi, group) in groups.iter().enumerate() {
        let mut prod_set = BTreeSet::new();
        for input in &group.inputs {
            let prod_indices = resolve_producer_groups(input, group.count, groups);
            for pi in prod_indices {
                if pi != gi {
                    prod_set.insert(pi);
                }
            }
        }
        producers.push(prod_set.into_iter().collect());
    }

    producers
}

/// Build reverse mapping: for each group, which groups consume it.
fn build_consumer_graph(n: usize, producers: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (gi, prods) in producers.iter().enumerate() {
        for &pi in prods {
            consumers[pi].push(gi);
        }
    }
    consumers
}

// ---------------------------------------------------------------------------
// Chain extraction
// ---------------------------------------------------------------------------

/// Extract chains: maximal sets of groups connected by exclusive edges.
///
/// A chain starts at a "sink" (group with no compute consumers, or whose
/// consumers all have multiple producers) and walks backward through
/// exclusive producer relationships (producer has fan-out of exactly 1
/// to compute groups).
///
/// Each group belongs to exactly one chain. Shared groups (fan-out > 1)
/// start their own chain.
fn extract_chains(
    n: usize,
    producers: &[Vec<usize>],
    consumers: &[Vec<usize>],
    fan_out: &[usize],
    is_data: &[bool],
) -> Vec<Vec<usize>> {
    let mut chain_of: Vec<Option<usize>> = vec![None; n];
    let mut chains: Vec<Vec<usize>> = Vec::new();

    // Process groups in reverse topo order (sinks first).
    // We use reverse insertion order as a heuristic (groups are inserted
    // in topological order by construction).
    for gi in (0..n).rev() {
        if is_data[gi] || chain_of[gi].is_some() {
            continue;
        }

        // Start a new chain from this group.
        let chain_id = chains.len();
        let mut chain = vec![gi];
        chain_of[gi] = Some(chain_id);

        // Walk backward through exclusive producers.
        let mut current = gi;
        loop {
            // Find producers that are exclusive to this consumer
            // (fan_out == 1 means only one compute consumer).
            let exclusive_prods: Vec<usize> = producers[current]
                .iter()
                .copied()
                .filter(|&pi| {
                    !is_data[pi]
                        && chain_of[pi].is_none()
                        && fan_out[pi] == 1
                })
                .collect();

            if exclusive_prods.len() == 1 {
                // Single exclusive producer: extend the chain.
                let pi = exclusive_prods[0];
                chain_of[pi] = Some(chain_id);
                chain.push(pi);
                current = pi;
            } else {
                // Zero or multiple exclusive producers: stop.
                break;
            }
        }

        chain.reverse(); // Put in dependency order (producers first).
        chains.push(chain);
    }

    // Any remaining unassigned compute groups get their own chains.
    for gi in 0..n {
        if !is_data[gi] && chain_of[gi].is_none() {
            let chain_id = chains.len();
            chain_of[gi] = Some(chain_id);
            chains.push(vec![gi]);
        }
    }

    chains
}

/// Map group index -> chain index.
fn build_group_to_chain_map(n: usize, chains: &[Vec<usize>]) -> HashMap<usize, usize> {
    let mut m = HashMap::new();
    for (ci, chain) in chains.iter().enumerate() {
        for &gi in chain {
            m.insert(gi, ci);
        }
    }
    m
}

/// Build chain-level dependency graph.
/// chain_deps[i] = set of chain indices that chain i depends on.
fn build_chain_deps(
    chains: &[Vec<usize>],
    producers: &[Vec<usize>],
    group_to_chain: &HashMap<usize, usize>,
    is_data: &[bool],
) -> Vec<HashSet<usize>> {
    let nc = chains.len();
    let mut deps: Vec<HashSet<usize>> = vec![HashSet::new(); nc];

    for (ci, chain) in chains.iter().enumerate() {
        for &gi in chain {
            for &pi in &producers[gi] {
                if is_data[pi] {
                    continue;
                }
                if let Some(&pci) = group_to_chain.get(&pi) {
                    if pci != ci {
                        deps[ci].insert(pci);
                    }
                }
            }
        }
    }

    deps
}

// ---------------------------------------------------------------------------
// Topological sort
// ---------------------------------------------------------------------------

/// Topological sort of all groups (Kahn's algorithm).
fn topological_sort(n: usize, producers: &[Vec<usize>]) -> Vec<usize> {
    let mut in_degree = vec![0usize; n];
    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (gi, prods) in producers.iter().enumerate() {
        for &pi in prods {
            in_degree[gi] += 1;
            consumers[pi].push(gi);
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
        for &ci in &consumers[gi] {
            in_degree[ci] -= 1;
            if in_degree[ci] == 0 {
                queue.push_back(ci);
            }
        }
    }

    // Append any remaining (shouldn't happen in valid DAG).
    if order.len() < n {
        let in_order: HashSet<usize> = order.iter().copied().collect();
        for gi in 0..n {
            if !in_order.contains(&gi) {
                order.push(gi);
            }
        }
    }

    order
}

// ---------------------------------------------------------------------------
// Phase detection
// ---------------------------------------------------------------------------

/// Detect phase boundaries and split topo_order into phases.
fn detect_and_split_phases(
    topo_order: &[usize],
    groups: &[AtomGroup],
    consumers: &[Vec<usize>],
    is_data: &[bool],
    target_kernels: usize,
) -> Vec<Vec<usize>> {
    let n = topo_order.len();
    if n <= 1 || target_kernels <= 1 {
        return vec![topo_order.to_vec()];
    }

    // topo_pos[gi] = position in topo_order.
    let mut topo_pos = vec![0usize; groups.len()];
    for (pos, &gi) in topo_order.iter().enumerate() {
        topo_pos[gi] = pos;
    }

    // Compute last_consumer_pos for each group.
    let mut last_consumer_pos = vec![0usize; groups.len()];
    for (gi, cons) in consumers.iter().enumerate() {
        last_consumer_pos[gi] = topo_pos[gi];
        for &ci in cons {
            let cpos = topo_pos[ci];
            if cpos > last_consumer_pos[gi] {
                last_consumer_pos[gi] = cpos;
            }
        }
    }

    // Sweep through topo order computing live set size at each position.
    let mut expire_at: Vec<Vec<usize>> = vec![vec![]; n];
    for (gi, &lc_pos) in last_consumer_pos.iter().enumerate() {
        if !is_data[gi] && topo_pos[gi] < lc_pos {
            expire_at[lc_pos].push(gi);
        }
    }

    let mut live_cost = vec![0u64; n];
    let mut current_live: u64 = 0;

    for (pos, &gi) in topo_order.iter().enumerate() {
        for &expired_gi in &expire_at[pos] {
            current_live = current_live.saturating_sub(groups[expired_gi].count);
        }
        if !is_data[gi] && last_consumer_pos[gi] > pos {
            current_live += groups[gi].count;
        }
        live_cost[pos] = current_live;
    }

    // Find phase boundaries: positions with lowest live cost.
    // We want sqrt(target_kernels) phases, with within-phase splitting for the rest.
    let num_phase_cuts = ((target_kernels as f64).sqrt().ceil() as usize)
        .max(1)
        .min(target_kernels - 1);

    let min_phase_size = (n / 50).max(2);
    let boundaries = find_best_cuts(&live_cost, num_phase_cuts, n, min_phase_size);

    // Split into phases.
    if boundaries.is_empty() {
        return vec![topo_order.to_vec()];
    }

    let mut phases = Vec::new();
    let mut start = 0;
    for &boundary in &boundaries {
        let end = boundary + 1;
        if end > start && end <= n {
            phases.push(topo_order[start..end].to_vec());
            start = end;
        }
    }
    if start < n {
        phases.push(topo_order[start..].to_vec());
    }

    phases
}

/// Find the `num_cuts` positions with lowest cost, respecting minimum spacing.
fn find_best_cuts(costs: &[u64], num_cuts: usize, total: usize, min_spacing: usize) -> Vec<usize> {
    if num_cuts == 0 || costs.is_empty() {
        return vec![];
    }

    let mut indexed: Vec<(usize, u64)> = costs.iter().copied().enumerate().collect();
    indexed.retain(|&(pos, _)| pos > 0 && pos < total - 1);
    indexed.sort_by_key(|&(_, cost)| cost);

    let mut cuts: Vec<usize> = Vec::with_capacity(num_cuts);

    for &(pos, _) in &indexed {
        if cuts.len() >= num_cuts {
            break;
        }
        let valid = cuts.iter().all(|&c| {
            let dist = if pos > c { pos - c } else { c - pos };
            dist >= min_spacing
        });
        let from_start = pos;
        let from_end = total - 1 - pos;

        if valid && from_start >= min_spacing && from_end >= min_spacing {
            cuts.push(pos);
        }
    }

    cuts.sort();
    cuts
}

// ---------------------------------------------------------------------------
// Within-phase chain distribution
// ---------------------------------------------------------------------------

/// Topological sort of chains within a phase.
fn topo_sort_chains(
    chain_ids: &[usize],
    chain_deps: &HashMap<usize, HashSet<usize>>,
) -> Vec<usize> {
    let id_set: HashSet<usize> = chain_ids.iter().copied().collect();

    let mut in_degree: HashMap<usize, usize> = HashMap::new();
    let mut rev: HashMap<usize, Vec<usize>> = HashMap::new();

    for &ci in chain_ids {
        let local_deps = chain_deps
            .get(&ci)
            .map(|d| d.iter().filter(|x| id_set.contains(x)).count())
            .unwrap_or(0);
        in_degree.insert(ci, local_deps);

        if let Some(deps) = chain_deps.get(&ci) {
            for &dep in deps {
                if id_set.contains(&dep) {
                    rev.entry(dep).or_default().push(ci);
                }
            }
        }
    }

    let mut queue: VecDeque<usize> = VecDeque::new();
    for &ci in chain_ids {
        if *in_degree.get(&ci).unwrap_or(&0) == 0 {
            queue.push_back(ci);
        }
    }

    let mut order = Vec::with_capacity(chain_ids.len());
    while let Some(ci) = queue.pop_front() {
        order.push(ci);
        if let Some(consumers) = rev.get(&ci) {
            for &consumer in consumers {
                if let Some(deg) = in_degree.get_mut(&consumer) {
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(consumer);
                    }
                }
            }
        }
    }

    // Append any remaining (cycles, shouldn't happen).
    if order.len() < chain_ids.len() {
        let in_order: HashSet<usize> = order.iter().copied().collect();
        for &ci in chain_ids {
            if !in_order.contains(&ci) {
                order.push(ci);
            }
        }
    }

    order
}

/// Distribute chains across parallel slots using a greedy approach.
///
/// Strategy: process chains in topological order. For each chain, assign it
/// to the slot with the least total work among slots that don't violate
/// dependencies. Chains that depend on each other go to the same slot to
/// minimize cross-kernel traffic.
fn compute_parallel_slots(
    chain_topo: &[usize],
    chain_deps: &HashMap<usize, HashSet<usize>>,
    target_kernels: usize,
) -> Vec<usize> {
    if chain_topo.is_empty() {
        return vec![];
    }

    let num_slots = target_kernels.min(chain_topo.len()).max(1);
    let mut slot_work = vec![0usize; num_slots]; // approximate work per slot
    let mut chain_to_slot: HashMap<usize, usize> = HashMap::new();
    let mut result = vec![0usize; chain_topo.len()];

    for (idx, &ci) in chain_topo.iter().enumerate() {
        // Check if any dependency of this chain is in a specific slot.
        // If so, prefer that slot (to keep dependent chains together and
        // minimize traffic).
        let dep_slots: HashSet<usize> = chain_deps
            .get(&ci)
            .map(|deps| {
                deps.iter()
                    .filter_map(|d| chain_to_slot.get(d).copied())
                    .collect()
            })
            .unwrap_or_default();

        let chosen_slot = if dep_slots.len() == 1 {
            // Single dependency slot: put in the same slot to minimize traffic.
            // But only if it won't over-pack.
            let dep_slot = *dep_slots.iter().next().unwrap();
            let avg_work = slot_work.iter().sum::<usize>() / num_slots.max(1);
            if slot_work[dep_slot] < avg_work * 3 {
                dep_slot
            } else {
                // Slot is overloaded, pick the lightest.
                slot_work
                    .iter()
                    .enumerate()
                    .min_by_key(|&(_, &w)| w)
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            }
        } else {
            // No dependencies or multiple dependency slots: pick lightest slot.
            slot_work
                .iter()
                .enumerate()
                .min_by_key(|&(_, &w)| w)
                .map(|(i, _)| i)
                .unwrap_or(0)
        };

        chain_to_slot.insert(ci, chosen_slot);
        slot_work[chosen_slot] += 1; // Use chain count as work approximation.
        result[idx] = chosen_slot;
    }

    result
}

// ---------------------------------------------------------------------------
// Kernel dependency graph and acyclicity
// ---------------------------------------------------------------------------

/// Build kernel dependency graph.
fn build_kernel_dep_graph(
    kernel_groups: &[Vec<usize>],
    groups: &[AtomGroup],
) -> Vec<HashSet<usize>> {
    let num_kernels = kernel_groups.len();

    let mut group_to_kernel: HashMap<usize, usize> = HashMap::new();
    for (ki, kg) in kernel_groups.iter().enumerate() {
        for &gi in kg {
            group_to_kernel.insert(gi, ki);
        }
    }

    let mut deps: Vec<HashSet<usize>> = vec![HashSet::new(); num_kernels];

    for (ki, kg) in kernel_groups.iter().enumerate() {
        for &gi in kg {
            let group = &groups[gi];
            for input in &group.inputs {
                let prod_indices = resolve_producer_groups(input, group.count, groups);
                for pi in prod_indices {
                    if let Some(&pk) = group_to_kernel.get(&pi) {
                        if pk != ki {
                            deps[ki].insert(pk);
                        }
                    }
                }
            }
        }
    }

    deps
}

/// Find cycles in the kernel dependency graph.
fn find_cycles(deps: &[HashSet<usize>]) -> Vec<(usize, usize)> {
    let n = deps.len();
    let mut cycles = Vec::new();

    // Check for direct mutual dependencies.
    for ki in 0..n {
        for &dep in &deps[ki] {
            if dep > ki && deps[dep].contains(&ki) {
                cycles.push((ki, dep));
            }
        }
    }

    if !cycles.is_empty() {
        return cycles;
    }

    // Full cycle detection using DFS coloring.
    let mut color = vec![0u8; n]; // 0=white, 1=gray, 2=black

    fn dfs(
        node: usize,
        deps: &[HashSet<usize>],
        color: &mut [u8],
        cycles: &mut Vec<(usize, usize)>,
    ) {
        color[node] = 1;
        for &next in &deps[node] {
            if color[next] == 1 {
                cycles.push((node, next));
            } else if color[next] == 0 {
                dfs(next, deps, color, cycles);
            }
        }
        color[node] = 2;
    }

    for start in 0..n {
        if color[start] == 0 {
            dfs(start, deps, &mut color, &mut cycles);
        }
    }

    cycles
}

/// Enforce acyclicity by merging kernels involved in cycles.
fn enforce_acyclicity(kernel_groups: &mut Vec<Vec<usize>>, groups: &[AtomGroup]) {
    for _ in 0..100 {
        let deps = build_kernel_dep_graph(kernel_groups, groups);
        let cycles = find_cycles(&deps);

        if cycles.is_empty() {
            break;
        }

        let (a, b) = cycles[0];
        let (keep, remove) = if a < b { (a, b) } else { (b, a) };

        let removed_groups = kernel_groups[remove].clone();
        kernel_groups[keep].extend(removed_groups);
        kernel_groups[keep].sort();
        kernel_groups[keep].dedup();
        kernel_groups.remove(remove);
    }
}

// ---------------------------------------------------------------------------
// Balancing
// ---------------------------------------------------------------------------

/// Balance kernels: split oversized ones, merge tiny ones.
fn balance_kernels(
    kernel_groups: &mut Vec<Vec<usize>>,
    groups: &[AtomGroup],
    _target_kernels: usize,
) {
    let total_work: u64 = groups.iter().map(|g| g.count).sum();
    let max_work_per_kernel = total_work / 5; // 20% threshold

    // Split oversized kernels.
    let mut i = 0;
    while i < kernel_groups.len() {
        let work: u64 = kernel_groups[i].iter().map(|&gi| groups[gi].count).sum();
        if work > max_work_per_kernel && kernel_groups[i].len() > 1 {
            let mid = kernel_groups[i].len() / 2;
            let second_half: Vec<usize> = kernel_groups[i][mid..].to_vec();
            kernel_groups[i].truncate(mid);
            kernel_groups.push(second_half);
            // Don't increment i, re-check the first half.
        } else {
            i += 1;
        }
    }

    // Merge tiny kernels.
    let min_work = total_work / 1000;
    let mut changed = true;
    while changed {
        changed = false;
        let mut i = 0;
        while i < kernel_groups.len() && kernel_groups.len() > 1 {
            let work: u64 = kernel_groups[i].iter().map(|&gi| groups[gi].count).sum();
            if work < min_work {
                // Find neighbor to merge with.
                let deps = build_kernel_dep_graph(kernel_groups, groups);
                let target = find_merge_target(i, &deps, kernel_groups, groups);

                if let Some(target) = target {
                    let (keep, remove) = if target < i {
                        (target, i)
                    } else {
                        (i, target)
                    };
                    let removed = kernel_groups[remove].clone();
                    kernel_groups[keep].extend(removed);
                    kernel_groups[keep].sort();
                    kernel_groups[keep].dedup();
                    kernel_groups.remove(remove);
                    changed = true;
                    continue;
                }
            }
            i += 1;
        }
    }
}

/// Find best kernel to merge a tiny kernel with.
fn find_merge_target(
    ki: usize,
    deps: &[HashSet<usize>],
    kernel_groups: &[Vec<usize>],
    groups: &[AtomGroup],
) -> Option<usize> {
    let mut candidates: Vec<usize> = Vec::new();

    // Kernels we depend on.
    for &dep in &deps[ki] {
        candidates.push(dep);
    }
    // Kernels that depend on us.
    for (other_ki, other_deps) in deps.iter().enumerate() {
        if other_ki != ki && other_deps.contains(&ki) {
            candidates.push(other_ki);
        }
    }

    if candidates.is_empty() {
        return if ki > 0 {
            Some(ki - 1)
        } else if kernel_groups.len() > 1 {
            Some(1)
        } else {
            None
        };
    }

    candidates.dedup();
    candidates
        .iter()
        .min_by_key(|&&k| {
            kernel_groups[k]
                .iter()
                .map(|&gi| groups[gi].count)
                .sum::<u64>()
        })
        .copied()
}

// ---------------------------------------------------------------------------
// Producer group resolution
// ---------------------------------------------------------------------------

fn resolve_producer_groups(input: &InputRef, count: u64, groups: &[AtomGroup]) -> Vec<usize> {
    match input {
        InputRef::Broadcast(atom_id) => {
            find_group_idx(groups, *atom_id).into_iter().collect()
        }

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

        InputRef::SymAffine {
            base, stride_i, ..
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
    if groups.is_empty() || lo > hi {
        return vec![];
    }
    let start_idx = groups.partition_point(|g| g.base_id.0 + g.count <= lo);
    let mut result = Vec::new();
    for gi in start_idx..groups.len() {
        let g = &groups[gi];
        let g_lo = g.base_id.0;
        let g_hi = g.base_id.0 + g.count - 1;
        if g_lo > hi {
            break;
        }
        if g_hi >= lo && g_lo <= hi {
            result.push(gi);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::nano_graph::{ScalarBinOp, ScalarOp, ScalarUnaryOp};
    use crate::nano_graph::NanoGraph;
    use crate::numeric_scalar::NumericScalar;

    // --- Test graph builders ---

    fn make_linear_chain(n: usize) -> NanoGraph {
        let mut g = NanoGraph::new();
        let mut prev = g.push_group(
            100,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        for _ in 1..n {
            prev = g.push_group(
                100,
                ScalarOp::Unary {
                    op: ScalarUnaryOp::Neg,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![InputRef::Affine {
                    base: prev,
                    stride: 1,
                }],
            );
        }
        g
    }

    fn make_diamond() -> NanoGraph {
        let mut g = NanoGraph::new();
        let a = g.push_group(
            100,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let b = g.push_group(
            100,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: a,
                stride: 1,
            }],
        );
        let c = g.push_group(
            100,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: a,
                stride: 1,
            }],
        );
        let _d = g.push_group(
            100,
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
        g
    }

    fn make_pipeline_pinch() -> NanoGraph {
        let mut g = NanoGraph::new();

        let input = g.push_group(
            1000,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        let mut stage1_outputs = Vec::new();
        for i in 0..10 {
            let out = g.push_group(
                100,
                ScalarOp::Unary {
                    op: ScalarUnaryOp::Neg,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![InputRef::Affine {
                    base: AtomId(input.0 + i * 100),
                    stride: 1,
                }],
            );
            stage1_outputs.push(out);
        }

        let pinch = g.push_group(
            100,
            ScalarOp::ReduceSum {
                reduce_count: 10,
                reduce_stride: 100,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: stage1_outputs[0],
                stride: 1,
            }],
        );

        for _ in 0..10 {
            g.push_group(
                100,
                ScalarOp::Unary {
                    op: ScalarUnaryOp::Exp,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![InputRef::Broadcast(pinch)],
            );
        }

        g
    }

    /// Matmul-like structure: C[M,N] = A[M,K] * B[K,N] with ReduceSum.
    fn make_matmul(m: usize, k: usize, nn: usize) -> NanoGraph {
        let mut g = NanoGraph::new();

        let a_base = g.push_group(
            (m * k) as u64,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        let b_base = g.push_group(
            (k * nn) as u64,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        let mut mul_bases = Vec::new();
        for row in 0..m {
            let mul = g.push_group(
                (k * nn) as u64,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![
                    InputRef::StridedBroadcast {
                        base: AtomId(a_base.0 + (row * k) as u64),
                        stride: 1,
                        repeat: nn as u64,
                    },
                    InputRef::Affine {
                        base: b_base,
                        stride: 1,
                    },
                ],
            );
            mul_bases.push(mul);
        }

        for row in 0..m {
            g.push_group(
                nn as u64,
                ScalarOp::ReduceSum {
                    reduce_count: k as u64,
                    reduce_stride: nn as i64,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![InputRef::Affine {
                    base: mul_bases[row],
                    stride: 1,
                }],
            );
        }

        g
    }

    /// Two sequential matmuls: A@B then result@C.
    fn make_matmul_chain(m: usize, k1: usize, k2: usize, nn: usize) -> NanoGraph {
        let mut g = NanoGraph::new();

        let a_base = g.push_group(
            (m * k1) as u64,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let b_base = g.push_group(
            (k1 * k2) as u64,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        let mut mul1_bases = Vec::new();
        for row in 0..m {
            let mul = g.push_group(
                (k1 * k2) as u64,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![
                    InputRef::StridedBroadcast {
                        base: AtomId(a_base.0 + (row * k1) as u64),
                        stride: 1,
                        repeat: k2 as u64,
                    },
                    InputRef::Affine {
                        base: b_base,
                        stride: 1,
                    },
                ],
            );
            mul1_bases.push(mul);
        }

        let mut reduce1_bases = Vec::new();
        for row in 0..m {
            let red = g.push_group(
                k2 as u64,
                ScalarOp::ReduceSum {
                    reduce_count: k1 as u64,
                    reduce_stride: k2 as i64,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![InputRef::Affine {
                    base: mul1_bases[row],
                    stride: 1,
                }],
            );
            reduce1_bases.push(red);
        }

        let c_base = g.push_group(
            (k2 * nn) as u64,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        for row in 0..m {
            let mul = g.push_group(
                (k2 * nn) as u64,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![
                    InputRef::StridedBroadcast {
                        base: reduce1_bases[row],
                        stride: 1,
                        repeat: nn as u64,
                    },
                    InputRef::Affine {
                        base: c_base,
                        stride: 1,
                    },
                ],
            );

            g.push_group(
                nn as u64,
                ScalarOp::ReduceSum {
                    reduce_count: k2 as u64,
                    reduce_stride: nn as i64,
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
        }

        g
    }

    /// Parallel matmuls sharing X input: X@W1 and X@W2.
    fn make_parallel_matmuls(m: usize, k: usize, n1: usize, n2: usize) -> NanoGraph {
        let mut g = NanoGraph::new();

        let x_base = g.push_group(
            (m * k) as u64,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let w1_base = g.push_group(
            (k * n1) as u64,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let w2_base = g.push_group(
            (k * n2) as u64,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );

        // Matmul 1: X @ W1
        for row in 0..m {
            let mul = g.push_group(
                (k * n1) as u64,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![
                    InputRef::StridedBroadcast {
                        base: AtomId(x_base.0 + (row * k) as u64),
                        stride: 1,
                        repeat: n1 as u64,
                    },
                    InputRef::Affine {
                        base: w1_base,
                        stride: 1,
                    },
                ],
            );
            g.push_group(
                n1 as u64,
                ScalarOp::ReduceSum {
                    reduce_count: k as u64,
                    reduce_stride: n1 as i64,
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
        }

        // Matmul 2: X @ W2
        for row in 0..m {
            let mul = g.push_group(
                (k * n2) as u64,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![],
                vec![],
                vec![
                    InputRef::StridedBroadcast {
                        base: AtomId(x_base.0 + (row * k) as u64),
                        stride: 1,
                        repeat: n2 as u64,
                    },
                    InputRef::Affine {
                        base: w2_base,
                        stride: 1,
                    },
                ],
            );
            g.push_group(
                n2 as u64,
                ScalarOp::ReduceSum {
                    reduce_count: k as u64,
                    reduce_stride: n2 as i64,
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
        }

        g
    }

    // --- Validation helpers ---

    fn check_coverage(result: &NanoPartitionResult, total_groups: usize) {
        let mut assigned: Vec<usize> = result
            .kernel_groups
            .iter()
            .flat_map(|k| k.iter().copied())
            .collect();
        let total_assigned = assigned.len();
        assigned.sort();
        assigned.dedup();
        assert_eq!(
            assigned.len(),
            total_assigned,
            "Duplicate group assignments detected: {} assigned, {} unique",
            total_assigned,
            assigned.len()
        );
        assert_eq!(
            assigned.len(),
            total_groups,
            "Expected {} groups, got {}",
            total_groups,
            assigned.len()
        );
        let expected: Vec<usize> = (0..total_groups).collect();
        assert_eq!(assigned, expected, "Not all groups assigned");
    }

    fn check_acyclicity(result: &NanoPartitionResult, groups: &[AtomGroup]) {
        let deps = build_kernel_dep_graph(&result.kernel_groups, groups);
        let cycles = find_cycles(&deps);
        assert!(
            cycles.is_empty(),
            "Found {} cycles: {:?}",
            cycles.len(),
            &cycles[..cycles.len().min(5)]
        );

        // Verify topological sort succeeds.
        let n = result.kernel_groups.len();
        let mut in_deg: Vec<usize> = deps.iter().map(|d| d.len()).collect();
        let mut rev_deps: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (ki, ki_deps) in deps.iter().enumerate() {
            for &dep in ki_deps {
                rev_deps[dep].push(ki);
            }
        }

        let mut queue: VecDeque<usize> = VecDeque::new();
        for ki in 0..n {
            if in_deg[ki] == 0 {
                queue.push_back(ki);
            }
        }

        let mut sorted = 0;
        while let Some(ki) = queue.pop_front() {
            sorted += 1;
            for &consumer in &rev_deps[ki] {
                in_deg[consumer] -= 1;
                if in_deg[consumer] == 0 {
                    queue.push_back(consumer);
                }
            }
        }

        assert_eq!(sorted, n, "Topo sort visited {} of {} kernels", sorted, n);
    }

    fn check_balance(result: &NanoPartitionResult, groups: &[AtomGroup], max_fraction: f64) {
        let total_atoms: u64 = groups.iter().map(|g| g.count).sum();
        let max_allowed = (total_atoms as f64 * max_fraction) as u64 + 1;

        for (ki, kernel) in result.kernel_groups.iter().enumerate() {
            let kernel_atoms: u64 = kernel.iter().map(|&gi| groups[gi].count).sum();
            assert!(
                kernel_atoms <= max_allowed,
                "Kernel {} has {} atoms ({:.1}% of {}), exceeds {:.0}%",
                ki,
                kernel_atoms,
                kernel_atoms as f64 / total_atoms as f64 * 100.0,
                total_atoms,
                max_fraction * 100.0,
            );
        }
    }

    fn check_parallelism(result: &NanoPartitionResult, groups: &[AtomGroup]) -> usize {
        let deps = build_kernel_dep_graph(&result.kernel_groups, groups);
        let n = result.kernel_groups.len();
        let mut independent_pairs = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                if !deps[i].contains(&j) && !deps[j].contains(&i) {
                    independent_pairs += 1;
                }
            }
        }
        independent_pairs
    }

    // --- Tests ---

    #[test]
    fn test_empty_graph() {
        let g = NanoGraph::new();
        let result = partition_nanograph(&g, 4);
        assert_eq!(result.num_kernels, 0);
    }

    #[test]
    fn test_single_group() {
        let mut g = NanoGraph::new();
        g.push_group(
            100,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
            vec![],
        );
        let result = partition_nanograph(&g, 4);
        check_coverage(&result, 1);
    }

    #[test]
    fn test_linear_chain() {
        let g = make_linear_chain(20);
        let result = partition_nanograph(&g, 4);
        check_coverage(&result, g.num_groups());
        check_acyclicity(&result, g.groups());
        println!(
            "Linear chain: {} kernels from {} groups",
            result.num_kernels,
            g.num_groups()
        );
    }

    #[test]
    fn test_diamond() {
        let g = make_diamond();
        let result = partition_nanograph(&g, 4);
        check_coverage(&result, g.num_groups());
        check_acyclicity(&result, g.groups());
        let parallel = check_parallelism(&result, g.groups());
        println!("Diamond: {} kernels, {} independent pairs", result.num_kernels, parallel);
    }

    #[test]
    fn test_pipeline_pinch() {
        let g = make_pipeline_pinch();
        let result = partition_nanograph(&g, 4);
        check_coverage(&result, g.num_groups());
        check_acyclicity(&result, g.groups());
        println!(
            "Pipeline pinch: {} kernels from {} groups",
            result.num_kernels,
            g.num_groups()
        );
        assert!(result.num_kernels >= 2, "Expected >=2, got {}", result.num_kernels);
    }

    #[test]
    fn test_matmul_parallelism() {
        let g = make_matmul(4, 8, 16);
        let result = partition_nanograph(&g, 4);
        check_coverage(&result, g.num_groups());
        check_acyclicity(&result, g.groups());
        let parallel = check_parallelism(&result, g.groups());
        println!(
            "Matmul 4x8x16: {} kernels, {} independent pairs",
            result.num_kernels, parallel
        );
        assert!(result.num_kernels >= 2, "Expected >=2 kernels, got {}", result.num_kernels);
    }

    #[test]
    fn test_matmul_rows_are_parallel() {
        // Key test: matmul rows should end up in different kernels.
        let g = make_matmul(4, 8, 16);
        let result = partition_nanograph(&g, 4);
        check_coverage(&result, g.num_groups());
        check_acyclicity(&result, g.groups());

        // Mul groups are at indices 2..6 (after 2 data groups).
        // ReduceSum groups are at indices 6..10.
        // Each row: (Mul[i], ReduceSum[i]) should be a chain.
        // Different rows should be in different kernels.

        let mul_groups: Vec<usize> = (2..6).collect();
        let reduce_groups: Vec<usize> = (6..10).collect();

        // Check that mul and its reduce are in the same kernel.
        for row in 0..4 {
            let mul_gi = mul_groups[row];
            let red_gi = reduce_groups[row];

            let mul_kernel = result
                .kernel_groups
                .iter()
                .position(|k| k.contains(&mul_gi));
            let red_kernel = result
                .kernel_groups
                .iter()
                .position(|k| k.contains(&red_gi));

            assert_eq!(
                mul_kernel, red_kernel,
                "Row {}: Mul in kernel {:?}, Reduce in kernel {:?}",
                row, mul_kernel, red_kernel
            );
        }

        // Check that at least 2 different rows are in different kernels.
        let row_kernels: Vec<usize> = mul_groups
            .iter()
            .map(|&gi| {
                result
                    .kernel_groups
                    .iter()
                    .position(|k| k.contains(&gi))
                    .unwrap()
            })
            .collect();
        let unique_kernels: HashSet<usize> = row_kernels.iter().copied().collect();
        assert!(
            unique_kernels.len() >= 2,
            "Expected rows in >=2 kernels, all in {:?}",
            unique_kernels
        );

        println!(
            "Row kernel assignments: {:?} ({} unique)",
            row_kernels,
            unique_kernels.len()
        );
    }

    #[test]
    fn test_matmul_balance() {
        let g = make_matmul(8, 16, 8);
        let result = partition_nanograph(&g, 8);
        check_coverage(&result, g.num_groups());
        check_acyclicity(&result, g.groups());
        check_balance(&result, g.groups(), 0.5);
        println!("Matmul 8x16x8: {} kernels", result.num_kernels);
    }

    #[test]
    fn test_matmul_chain() {
        let g = make_matmul_chain(4, 8, 8, 4);
        let result = partition_nanograph(&g, 8);
        check_coverage(&result, g.num_groups());
        check_acyclicity(&result, g.groups());
        println!(
            "Matmul chain: {} kernels from {} groups",
            result.num_kernels,
            g.num_groups()
        );
        assert!(result.num_kernels >= 2, "Expected >=2, got {}", result.num_kernels);
    }

    #[test]
    fn test_parallel_matmuls() {
        let g = make_parallel_matmuls(4, 8, 16, 8);
        let result = partition_nanograph(&g, 8);
        check_coverage(&result, g.num_groups());
        check_acyclicity(&result, g.groups());
        let parallel = check_parallelism(&result, g.groups());
        println!(
            "Parallel matmuls: {} kernels, {} independent pairs",
            result.num_kernels, parallel
        );
        assert!(parallel > 0, "Expected independent kernel pairs, got 0");
    }

    #[test]
    fn test_large_matmul_no_mega_kernel() {
        let g = make_matmul(16, 32, 64);
        let result = partition_nanograph(&g, 16);
        check_coverage(&result, g.num_groups());
        check_acyclicity(&result, g.groups());
        check_balance(&result, g.groups(), 0.30);
        let parallel = check_parallelism(&result, g.groups());
        println!(
            "Large matmul 16x32x64: {} kernels, {} independent pairs",
            result.num_kernels, parallel
        );
    }

    #[test]
    fn test_comprehensive_validity() {
        let cases: Vec<(&str, NanoGraph, usize)> = vec![
            ("linear_10", make_linear_chain(10), 4),
            ("linear_50", make_linear_chain(50), 8),
            ("diamond", make_diamond(), 4),
            ("pinch", make_pipeline_pinch(), 4),
            ("matmul_4x8x16", make_matmul(4, 8, 16), 4),
            ("matmul_8x16x8", make_matmul(8, 16, 8), 8),
            ("chain_4x8x8x4", make_matmul_chain(4, 8, 8, 4), 8),
            ("par_4x8x16x8", make_parallel_matmuls(4, 8, 16, 8), 8),
        ];

        for (name, graph, target) in &cases {
            let result = partition_nanograph(&graph, *target);
            check_coverage(&result, graph.num_groups());
            check_acyclicity(&result, graph.groups());
            let parallel = check_parallelism(&result, graph.groups());
            println!(
                "{}: {} groups -> {} kernels, {} independent pairs",
                name,
                graph.num_groups(),
                result.num_kernels,
                parallel
            );
        }
    }

    #[test]
    fn test_interleaved_group_indices() {
        let g = make_matmul(4, 8, 16);
        let result = partition_nanograph(&g, 4);

        let mut has_interleaved = false;
        for kernel in &result.kernel_groups {
            if kernel.len() >= 2 {
                for w in kernel.windows(2) {
                    if w[1] != w[0] + 1 {
                        has_interleaved = true;
                        break;
                    }
                }
            }
            if has_interleaved {
                break;
            }
        }

        println!("Has interleaved indices: {}", has_interleaved);
        // For matmul, chains (Mul+ReduceSum) are non-contiguous, so kernels
        // should have interleaved indices.
        assert!(
            has_interleaved,
            "Matmul should produce interleaved group indices in kernels"
        );
    }
}
