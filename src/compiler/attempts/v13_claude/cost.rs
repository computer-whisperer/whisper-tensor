//! Kernel cost estimation from structural analysis of addressing modes.
//!
//! Estimates memory traffic for a candidate kernel WITHOUT fully scheduling it.
//! The key insight: addressing modes (Broadcast, Affine, SymAffine) tell us
//! which values are shared, which are streamed sequentially, and which
//! dimensions are reduced (streamed one step at a time). This is enough to
//! estimate peak liveness and therefore cache pressure.
//!
//! See `src/compiler/problem_shape.md` § "Kernel Cost Estimation" for rationale.

use std::collections::{HashMap, HashSet};

use crate::nano_graph::{AtomGroup, AtomId, InputRef, NanoGraph, SymDim};

use super::partition::{AtomRange, CacheConfig};

/// Cost estimate for a candidate kernel.
#[derive(Debug, Clone)]
pub struct KernelCost {
    /// Minimum unavoidable I/O: values that must cross the kernel boundary.
    /// Each external input loaded once + each external output stored once.
    pub external_io: u64,

    /// Estimated peak number of simultaneously live values during execution.
    /// This determines whether the kernel fits in cache or needs to spill.
    pub peak_liveness: u64,

    /// Total estimated memory traffic (loads + stores across cache boundary).
    /// Equal to external_io when peak_liveness <= cache, higher when spilling.
    pub estimated_traffic: u64,

    /// Total compute ops in this kernel (number of atoms computed).
    pub compute_ops: u64,

    /// Arithmetic intensity: compute_ops / estimated_traffic.
    /// Higher is better. Memory-bound kernels have low intensity.
    pub arithmetic_intensity: f64,
}

impl std::fmt::Display for KernelCost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "cost(io={}, peak={}, traffic={}, ops={}, intensity={:.2})",
            self.external_io,
            self.peak_liveness,
            self.estimated_traffic,
            self.compute_ops,
            self.arithmetic_intensity,
        )
    }
}

/// Estimate the cost of a kernel defined by a set of atom ranges.
///
/// This walks the groups that overlap the kernel's atom ranges, classifies
/// each input by addressing mode, and estimates peak liveness based on
/// streaming patterns.
pub fn estimate_kernel_cost(
    graph: &NanoGraph,
    kernel_ranges: &[AtomRange],
    config: &CacheConfig,
) -> KernelCost {
    let cache_capacity = config.l1_capacity() as u64;

    // Build a set of atoms in this kernel for fast lookup.
    let kernel_atoms = KernelAtomSet::new(kernel_ranges);

    // Collect the groups (or sub-groups) that overlap this kernel.
    let overlapping = find_overlapping_groups(graph, &kernel_atoms);

    // For each group in the kernel, analyze its inputs to determine:
    // - external inputs (sources outside the kernel)
    // - internal streaming structure (which inputs are streamed vs resident)
    let mut external_inputs: HashSet<AtomId> = HashSet::new();
    let mut compute_ops: u64 = 0;

    // Track per-group liveness contributions.
    // We'll walk groups in order and estimate how they compose.
    let mut group_infos: Vec<GroupLivenessInfo> = Vec::new();

    for overlap in &overlapping {
        let group = overlap.group;
        let atom_count = overlap.count;
        compute_ops += atom_count;

        let mut info = GroupLivenessInfo {
            atom_count,
            resident_inputs: 0,
            streaming_inputs: 0,
            streaming_step_size: 0,
            is_reduction: !group.reduce_dims.is_empty(),
            reduce_k: resolve_k_bound(graph, &group.reduce_dims),
        };

        for input_ref in &group.inputs {
            match input_ref {
                InputRef::Broadcast(src) => {
                    // One value shared by all atoms in the group.
                    if !kernel_atoms.contains(*src) {
                        external_inputs.insert(*src);
                    }
                    // Adds 1 to resident set (stays live while group executes).
                    info.resident_inputs += 1;
                }
                InputRef::Affine { base, stride } => {
                    // Each atom reads a different source. Count how many are external.
                    let mut ext_count = 0u64;
                    // Sample to avoid O(n) per atom for large groups.
                    // For stride=1 contiguous access, we can compute directly.
                    if *stride == 1 {
                        let range_start = base.0;
                        let range_end = base.0 + overlap.count;
                        for id in range_start..range_end {
                            let atom = AtomId(id);
                            if !kernel_atoms.contains(atom) {
                                external_inputs.insert(atom);
                                ext_count += 1;
                            }
                        }
                    } else {
                        for i in 0..overlap.count {
                            let src = input_ref.resolve(i, 0);
                            if !kernel_atoms.contains(src) {
                                external_inputs.insert(src);
                                ext_count += 1;
                            }
                        }
                    }
                    // Affine stride-1 is streamable — only a chunk is live at once.
                    // The chunk size depends on the consumer's processing rate.
                    // For estimation, we treat the full extent as streaming input.
                    info.streaming_inputs += ext_count;
                    info.streaming_step_size += 1; // one element per atom step
                }
                InputRef::SymAffine { base, stride_i, stride_k } => {
                    // 2D access: varies with both atom offset and reduction step k.
                    // The k dimension is STREAMED — only one k-step is live at a time.
                    let k_bound = info.reduce_k;

                    // At each k step, the group reads `count` source atoms
                    // (one per atom in the group, at stride_i spacing).
                    // Across k steps, these come from different groups.
                    // Sample k=0 to find per-step external inputs.
                    let mut per_step_external = 0u64;
                    for i in 0..overlap.count {
                        let src = input_ref.resolve(i, 0);
                        if !kernel_atoms.contains(src) {
                            per_step_external += 1;
                        }
                    }

                    // Total external inputs across all k steps.
                    // Each step potentially references different source atoms.
                    for k in 0..k_bound {
                        for i in 0..overlap.count {
                            let src = input_ref.resolve(i, k);
                            if !kernel_atoms.contains(src) {
                                external_inputs.insert(src);
                            }
                        }
                    }

                    // Streaming: only one k-step's worth of sources is live at a time.
                    // Per step: `per_step_external` external loads.
                    // Plus the accumulator (count atoms) is resident across all k steps.
                    info.streaming_inputs += per_step_external;
                    info.streaming_step_size += overlap.count;
                }
                InputRef::Explicit(ids) => {
                    // Arbitrary access — count external refs.
                    for &src in ids {
                        if !kernel_atoms.contains(src) {
                            external_inputs.insert(src);
                        }
                    }
                    // Conservative: treat all as resident (no streaming pattern).
                    info.resident_inputs += ids.iter()
                        .filter(|id| !kernel_atoms.contains(**id))
                        .collect::<HashSet<_>>()
                        .len() as u64;
                }
                InputRef::StridedBroadcast { repeat, .. } => {
                    // Each block of `repeat` atoms shares one source.
                    // The distinct sources are the block-boundary atoms.
                    let num_blocks = (overlap.count + repeat - 1) / repeat;
                    let mut ext_count = 0u64;
                    for block in 0..num_blocks {
                        let src = input_ref.resolve(block * repeat, 0);
                        if !kernel_atoms.contains(src) {
                            external_inputs.insert(src);
                            ext_count += 1;
                        }
                    }
                    // Treat like broadcast — each source stays live while its block executes.
                    info.resident_inputs += ext_count;
                }
            }
        }

        group_infos.push(info);
    }

    // Count external outputs: atoms in this kernel consumed by groups outside it.
    let external_outputs = count_external_outputs(graph, &kernel_atoms);

    let external_io = external_inputs.len() as u64 + external_outputs;

    // Estimate peak liveness.
    // Walk the groups in order, tracking peak simultaneous live values.
    let peak_liveness = estimate_peak_liveness(&group_infos);

    // Estimate total traffic.
    let estimated_traffic = if peak_liveness <= cache_capacity {
        // Everything fits — traffic is just the external I/O.
        external_io
    } else {
        // Cache pressure — excess values must be evicted and potentially reloaded.
        // Simple model: the excess contributes additional traffic proportional
        // to how much we exceed cache.
        let excess = peak_liveness - cache_capacity;
        // Each excess value is evicted once and reloaded once (pessimistic).
        // In practice some evictions are to dead values (no reload needed),
        // so we use a 1.5x factor rather than 2x.
        external_io + (excess as f64 * 1.5) as u64
    };

    let arithmetic_intensity = if estimated_traffic > 0 {
        compute_ops as f64 / estimated_traffic as f64
    } else {
        f64::INFINITY
    };

    KernelCost {
        external_io,
        peak_liveness,
        estimated_traffic,
        compute_ops,
        arithmetic_intensity,
    }
}

/// Liveness analysis info for one group within a kernel.
#[derive(Debug)]
struct GroupLivenessInfo {
    /// Number of atoms this group contributes to the kernel.
    atom_count: u64,
    /// Number of values that stay resident while this group executes
    /// (broadcast inputs, explicit inputs).
    resident_inputs: u64,
    /// Number of external values streamed in per step (affine, symaffine per-k).
    streaming_inputs: u64,
    /// How many values are loaded per streaming step.
    streaming_step_size: u64,
    /// Whether this group is a reduction (has reduce_dims).
    is_reduction: bool,
    /// Bound on the reduction dimension (K).
    reduce_k: u64,
}

/// Estimate peak liveness from per-group analysis.
///
/// The idea: walk groups in topological order (which for a NanoGraph is
/// insertion order, since later groups reference earlier ones). At each
/// group, the live set includes:
/// - The group's output atoms (produced, not yet consumed by later groups)
/// - The group's resident inputs (broadcast sources)
/// - One streaming step's worth of streaming inputs
///
/// For reductions, the accumulators (output atoms) stay resident across
/// all K steps, but only one step's intermediate inputs are live at a time.
fn estimate_peak_liveness(groups: &[GroupLivenessInfo]) -> u64 {
    if groups.is_empty() {
        return 0;
    }

    let mut peak: u64 = 0;

    for info in groups {
        // During this group's execution:
        let live = if info.is_reduction {
            // Reduction: accumulators + one k-step's inputs + broadcast sources.
            // Accumulators = atom_count (they persist across k steps).
            // Per k-step: streaming_inputs values loaded, used, discarded.
            info.atom_count            // accumulators
                + info.resident_inputs // broadcast sources
                + info.streaming_inputs // one k-step's sources
        } else {
            // Non-reduction: outputs + all inputs.
            // Streaming inputs are processed in chunks, but we estimate
            // the chunk as the per-step size.
            info.atom_count
                + info.resident_inputs
                + info.streaming_inputs
        };

        peak = peak.max(live);
    }

    peak
}

/// Compact set for fast atom membership testing.
struct KernelAtomSet {
    ranges: Vec<(u64, u64)>, // sorted (start, end) pairs
}

impl KernelAtomSet {
    fn new(ranges: &[AtomRange]) -> Self {
        let mut sorted: Vec<(u64, u64)> = ranges
            .iter()
            .map(|r| (r.base.0, r.base.0 + r.count))
            .collect();
        sorted.sort_by_key(|&(s, _)| s);
        // Merge overlapping/adjacent ranges.
        let mut merged: Vec<(u64, u64)> = Vec::new();
        for (s, e) in sorted {
            if let Some(last) = merged.last_mut() {
                if s <= last.1 {
                    last.1 = last.1.max(e);
                    continue;
                }
            }
            merged.push((s, e));
        }
        Self { ranges: merged }
    }

    fn contains(&self, id: AtomId) -> bool {
        let val = id.0;
        match self.ranges.binary_search_by(|&(s, e)| {
            if val < s {
                std::cmp::Ordering::Greater
            } else if val >= e {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        }) {
            Ok(_) => true,
            Err(_) => false,
        }
    }
}

/// Info about which portion of a group overlaps the kernel.
struct GroupOverlap<'a> {
    group: &'a AtomGroup,
    /// How many atoms from this group are in the kernel.
    count: u64,
}

/// Find groups that overlap the kernel's atom ranges.
fn find_overlapping_groups<'a>(
    graph: &'a NanoGraph,
    kernel_atoms: &KernelAtomSet,
) -> Vec<GroupOverlap<'a>> {
    let mut result = Vec::new();
    for group in graph.groups() {
        // Quick check: does the group's range overlap any kernel range?
        let g_start = group.base_id.0;
        let g_end = g_start + group.count;

        let mut overlap_count = 0u64;
        for &(r_start, r_end) in &kernel_atoms.ranges {
            if r_start >= g_end || r_end <= g_start {
                continue;
            }
            let overlap_start = g_start.max(r_start);
            let overlap_end = g_end.min(r_end);
            overlap_count += overlap_end - overlap_start;
        }

        if overlap_count > 0 {
            result.push(GroupOverlap {
                group,
                count: overlap_count,
            });
        }
    }
    result
}

/// Count how many atoms in this kernel are consumed by groups outside it.
fn count_external_outputs(graph: &NanoGraph, kernel_atoms: &KernelAtomSet) -> u64 {
    let mut output_atoms: HashSet<AtomId> = HashSet::new();

    for group in graph.groups() {
        // Skip groups entirely inside the kernel — their consumers might be outside.
        // Skip groups entirely outside — they can't produce external outputs.
        // We need to check groups OUTSIDE the kernel that reference atoms INSIDE.
        let g_start = group.base_id.0;
        let g_end = g_start + group.count;

        // Check if any of this group's inputs reference kernel atoms.
        for input_ref in &group.inputs {
            match input_ref {
                InputRef::Broadcast(src) => {
                    if kernel_atoms.contains(*src) && !is_group_in_kernel(g_start, g_end, kernel_atoms) {
                        output_atoms.insert(*src);
                    }
                }
                InputRef::Affine { base, stride } => {
                    if is_group_in_kernel(g_start, g_end, kernel_atoms) {
                        continue; // consumer is internal
                    }
                    if *stride == 1 {
                        let src_start = base.0;
                        let src_end = base.0 + group.count;
                        for &(r_start, r_end) in &kernel_atoms.ranges {
                            if r_start >= src_end || r_end <= src_start {
                                continue;
                            }
                            let os = src_start.max(r_start);
                            let oe = src_end.min(r_end);
                            for id in os..oe {
                                output_atoms.insert(AtomId(id));
                            }
                        }
                    } else {
                        for i in 0..group.count {
                            let src = input_ref.resolve(i, 0);
                            if kernel_atoms.contains(src) {
                                output_atoms.insert(src);
                            }
                        }
                    }
                }
                InputRef::SymAffine { .. } => {
                    if is_group_in_kernel(g_start, g_end, kernel_atoms) {
                        continue;
                    }
                    let k_bound = resolve_k_bound(graph, &group.reduce_dims);
                    for k in 0..k_bound {
                        for i in 0..group.count {
                            let src = input_ref.resolve(i, k);
                            if kernel_atoms.contains(src) {
                                output_atoms.insert(src);
                            }
                        }
                    }
                }
                InputRef::Explicit(ids) => {
                    if is_group_in_kernel(g_start, g_end, kernel_atoms) {
                        continue;
                    }
                    for &src in ids {
                        if kernel_atoms.contains(src) {
                            output_atoms.insert(src);
                        }
                    }
                }
                InputRef::StridedBroadcast { repeat, .. } => {
                    if is_group_in_kernel(g_start, g_end, kernel_atoms) {
                        continue;
                    }
                    let num_blocks = (group.count + repeat - 1) / repeat;
                    for block in 0..num_blocks {
                        let src = input_ref.resolve(block * repeat, 0);
                        if kernel_atoms.contains(src) {
                            output_atoms.insert(src);
                        }
                    }
                }
            }
        }
    }

    output_atoms.len() as u64
}

/// Check if a group is fully inside the kernel.
fn is_group_in_kernel(g_start: u64, g_end: u64, kernel_atoms: &KernelAtomSet) -> bool {
    // A group is "in the kernel" if all its atoms are in the kernel.
    // Quick check: is there a single kernel range that covers the entire group?
    for &(r_start, r_end) in &kernel_atoms.ranges {
        if r_start <= g_start && r_end >= g_end {
            return true;
        }
    }
    false
}

/// Resolve the K bound for a set of reduce dims.
fn resolve_k_bound(graph: &NanoGraph, reduce_dims: &[SymDim]) -> u64 {
    reduce_dims
        .iter()
        .filter_map(|sd| graph.sym_dim_bounds.get(sd))
        .next()
        .copied()
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::attempts::v13_claude::partition::CacheConfig;
    use crate::compiler::attempts::v13_claude::test_graphs;
    use crate::nano_graph::ScalarBinOp;

    fn default_config() -> CacheConfig {
        CacheConfig {
            l1_bytes: 32 * 1024,
            value_size: 4,
        }
    }

    #[test]
    fn test_elementwise_single_kernel() {
        // C = A + B, 1024 elements. Put everything in one kernel.
        let (g, a, _, c) = test_graphs::elementwise_binary(1024, ScalarBinOp::Add);
        let config = default_config();

        // One kernel containing all atoms.
        let ranges = vec![AtomRange::new(a, g.num_atoms())];
        let cost = estimate_kernel_cost(&g, &ranges, &config);

        println!("elementwise single kernel: {}", cost);

        // No external I/O (everything is in the kernel).
        assert_eq!(cost.external_io, 0);
        // Peak liveness: the Add group needs 1024 outputs + 1024 + 1024 inputs.
        // But Literal groups have no inputs, so their liveness is just their atoms.
        // The Add group's peak = 1024 (outputs) + 1024 + 1024 (affine inputs) = 3072.
        // But the Literal inputs are internal, so streaming_inputs = 0 for them
        // because they're in the kernel. Actually the Literal groups produce values,
        // and the Add group consumes them with Affine stride=1 from inside the kernel.
        assert!(cost.peak_liveness <= 3072);
        assert_eq!(cost.compute_ops, 3 * 1024);
    }

    #[test]
    fn test_elementwise_split_kernels() {
        // C = A + B, 1024 elements. Split: {A, B} and {C} in separate kernels.
        let (g, a, _, c) = test_graphs::elementwise_binary(1024, ScalarBinOp::Add);
        let config = default_config();

        // Kernel 1: just the inputs (A and B literals).
        let k1_ranges = vec![AtomRange::new(a, 2 * 1024)];
        let cost1 = estimate_kernel_cost(&g, &k1_ranges, &config);

        // Kernel 2: just the Add op.
        let k2_ranges = vec![AtomRange::new(c, 1024)];
        let cost2 = estimate_kernel_cost(&g, &k2_ranges, &config);

        println!("split kernel 1 (inputs): {}", cost1);
        println!("split kernel 2 (add):    {}", cost2);

        // Kernel 2 has 2048 external inputs (A and B atoms are in kernel 1).
        // The Add result isn't consumed by any other group, so external_outputs = 0.
        assert_eq!(cost2.external_io, 2048);

        // Kernel 1 has 2048 external outputs (A and B atoms consumed by the Add in kernel 2).
        assert_eq!(cost1.external_io, 2048);

        // Total traffic for the split is worse than single kernel (which has 0).
        assert!(cost1.estimated_traffic + cost2.estimated_traffic > 0);
    }

    #[test]
    fn test_matmul_single_kernel() {
        // Small matmul: 4x8x16. Put everything in one kernel.
        let (g, a, _, _) = test_graphs::matmul(4, 8, 16);
        let config = default_config();

        let ranges = vec![AtomRange::new(a, g.num_atoms())];
        let cost = estimate_kernel_cost(&g, &ranges, &config);

        println!("matmul 4x8x16 single kernel: {}", cost);

        // Everything internal, no external I/O.
        assert_eq!(cost.external_io, 0);
        // Should fit in cache (736 atoms < 8192 capacity).
        assert!(cost.peak_liveness <= config.l1_capacity() as u64);
        assert_eq!(cost.estimated_traffic, 0);
    }

    #[test]
    fn test_matmul_reduction_streaming() {
        // Medium matmul: 8x32x64. Check that reductions are recognized as streaming.
        let (g, a, _, _) = test_graphs::matmul(8, 32, 64);
        let config = default_config();

        let ranges = vec![AtomRange::new(a, g.num_atoms())];
        let cost = estimate_kernel_cost(&g, &ranges, &config);

        println!("matmul 8x32x64 single kernel: {}", cost);

        // Total atoms: 8*32 + 32*64 + 8*32*64 + 8*64 = 256 + 2048 + 16384 + 512 = 19200
        assert_eq!(cost.compute_ops, 19200);

        // Peak liveness should be much less than total atoms because:
        // - ReduceSum groups stream over K (only one k-step live at a time)
        // - Mul groups are intermediates consumed by ReduceSum
        // The ReduceSum peak per group: 64 accumulators + 1 broadcast + 64 streaming = 129
        println!("  peak liveness: {} (vs {} total atoms)", cost.peak_liveness, g.num_atoms());
        assert!(cost.peak_liveness < cost.compute_ops);
    }

    #[test]
    fn test_matmul_large_cost() {
        // Realistic matmul: 24x96x192. Check traffic estimate.
        let (g, a, _, _) = test_graphs::matmul(24, 96, 192);
        let config = default_config();

        let ranges = vec![AtomRange::new(a, g.num_atoms())];
        let cost = estimate_kernel_cost(&g, &ranges, &config);

        println!("matmul 24x96x192 single kernel: {}", cost);
        println!("  peak_liveness: {} vs cache_capacity: {}", cost.peak_liveness, config.l1_capacity());

        // This should show nonzero traffic if peak exceeds cache.
        // But all atoms are internal, so external_io = 0.
        // Traffic comes entirely from spill penalty.
        assert_eq!(cost.external_io, 0);
    }
}
