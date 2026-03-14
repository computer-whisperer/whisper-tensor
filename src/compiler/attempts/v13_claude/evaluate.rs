//! Evaluation metrics for partition quality.
//!
//! Given a NanoGraph and a PartitionResult, compute metrics that let us
//! compare different partitioning strategies on an apples-to-apples basis.
//! The primary metric is total cross-kernel data transfer (values that
//! cross a kernel boundary = a load + store).

use std::collections::{HashMap, HashSet};

use crate::nano_graph::{AtomId, InputRef, NanoGraph};

use super::partition::{CacheConfig, Kernel, PartitionResult};

/// Quality metrics for a partitioning.
#[derive(Debug, Clone)]
pub struct PartitionMetrics {
    /// Total number of kernels.
    pub num_kernels: usize,
    /// Total number of atoms in the graph.
    pub total_atoms: u32,
    /// Total cross-kernel data transfers (each value that crosses a boundary
    /// counts once per consuming kernel — i.e., if a value is read by 3
    /// different kernels, it counts as 3 transfers).
    pub total_transfers: u64,
    /// Total unique values that must be materialized to memory (produced in
    /// one kernel, consumed in another).
    pub unique_materialized: u64,
    /// Maximum working set across all kernels (in values, not bytes).
    pub max_working_set: usize,
    /// Per-kernel working set sizes.
    pub kernel_working_sets: Vec<usize>,
    /// Number of kernels whose working set exceeds the cache budget.
    pub kernels_over_budget: usize,
    /// Cache budget in values (for reference).
    pub cache_capacity: usize,
}

impl std::fmt::Display for PartitionMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Partition Metrics:")?;
        writeln!(f, "  {} kernels, {} atoms", self.num_kernels, self.total_atoms)?;
        writeln!(f, "  {} total transfers ({} unique materialized values)",
            self.total_transfers, self.unique_materialized)?;
        writeln!(f, "  max working set: {} values (budget: {})",
            self.max_working_set, self.cache_capacity)?;
        if self.kernels_over_budget > 0 {
            writeln!(f, "  WARNING: {} kernels exceed cache budget!", self.kernels_over_budget)?;
        }
        // Arithmetic intensity: total compute ops / total memory ops
        if self.total_transfers > 0 {
            let intensity = self.total_atoms as f64 / self.total_transfers as f64;
            writeln!(f, "  arithmetic intensity: {:.2} ops/transfer", intensity)?;
        }
        Ok(())
    }
}

/// Compute partition quality metrics.
///
/// This walks the NanoGraph, resolves all input references, and counts
/// how many values must cross kernel boundaries.
pub fn evaluate(graph: &NanoGraph, result: &PartitionResult) -> PartitionMetrics {
    let cache_capacity = result.config.l1_capacity();

    // Build atom → kernel index lookup.
    let mut atom_kernel: HashMap<AtomId, usize> = HashMap::new();
    for (ki, kernel) in result.kernels.iter().enumerate() {
        for range in &kernel.atom_ranges {
            for id in range.iter() {
                atom_kernel.insert(id, ki);
            }
        }
    }

    // For each kernel, compute:
    // - external inputs (atoms read from other kernels)
    // - internal atoms (produced and consumed within the kernel)
    // - external outputs (atoms produced here, consumed elsewhere)
    let num_kernels = result.kernels.len();
    let mut kernel_ext_inputs: Vec<HashSet<AtomId>> = vec![HashSet::new(); num_kernels];
    let mut materialized: HashSet<AtomId> = HashSet::new();
    let mut total_transfers: u64 = 0;

    for group in graph.groups() {
        for i in 0..group.count {
            let atom_id = group.base_id.offset(i);
            let my_kernel = match atom_kernel.get(&atom_id) {
                Some(&k) => k,
                None => continue, // shouldn't happen if partition is valid
            };

            // Resolve all inputs for this atom.
            for input_ref in &group.inputs {
                // For SymAffine, we need to check all k values.
                let source_atoms: Vec<AtomId> = match input_ref {
                    InputRef::SymAffine { .. } => {
                        // Find the reduce_dim bound to know k range.
                        let k_max = group.reduce_dims.iter()
                            .filter_map(|sd| graph.sym_dim_bounds.get(sd))
                            .next()
                            .copied()
                            .unwrap_or(1) as u32;
                        (0..k_max).map(|k| input_ref.resolve(i, k)).collect()
                    }
                    _ => vec![input_ref.resolve(i, 0)],
                };

                for source_id in source_atoms {
                    if let Some(&source_kernel) = atom_kernel.get(&source_id) {
                        if source_kernel != my_kernel {
                            // Cross-kernel transfer.
                            kernel_ext_inputs[my_kernel].insert(source_id);
                            materialized.insert(source_id);
                            total_transfers += 1;
                        }
                    }
                    // Source not in any kernel = external input (graph boundary).
                    // These always require a load regardless of partitioning.
                }
            }
        }
    }

    // Compute working set per kernel.
    // Working set = atoms produced in this kernel + external inputs needed.
    let kernel_working_sets: Vec<usize> = result.kernels.iter().enumerate().map(|(ki, kernel)| {
        let internal: usize = kernel.atom_ranges.iter().map(|r| r.count as usize).sum();
        let external = kernel_ext_inputs[ki].len();
        internal + external
    }).collect();

    let max_working_set = kernel_working_sets.iter().copied().max().unwrap_or(0);
    let kernels_over_budget = kernel_working_sets.iter().filter(|&&ws| ws > cache_capacity).count();

    PartitionMetrics {
        num_kernels,
        total_atoms: graph.num_atoms(),
        total_transfers,
        unique_materialized: materialized.len() as u64,
        max_working_set,
        kernel_working_sets,
        kernels_over_budget,
        cache_capacity,
    }
}
