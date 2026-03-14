//! Common data structures for kernel partitioning.
//!
//! A partition assigns atoms from a NanoGraph into kernels. Each kernel
//! represents a chunk of work whose cost is measured by estimated memory
//! traffic. Cache is a cost, not a hard constraint — a kernel that exceeds
//! cache pays a streaming penalty, but fragmenting into many tiny kernels
//! can be worse due to forced materialization at boundaries.
//!
//! The quality of a partitioning is measured by total estimated memory
//! traffic across all kernels.

use std::collections::{HashMap, HashSet};

use crate::nano_graph::{AtomId, NanoGraph};

use super::cost::{self, KernelCost};

/// Configuration for the target cache hierarchy.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Size of the tightest cache bucket in bytes (L1, warp cache, SRAM tile).
    /// This determines the streaming penalty — a kernel whose peak liveness
    /// exceeds this will pay extra traffic from cache evictions.
    pub l1_bytes: usize,
    /// Bytes per scalar value (typically 4 for f32).
    pub value_size: usize,
}

impl CacheConfig {
    /// How many scalar values fit in the L1 cache.
    pub fn l1_capacity(&self) -> usize {
        self.l1_bytes / self.value_size
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            l1_bytes: 32 * 1024, // 32 KB L1
            value_size: 4,       // f32
        }
    }
}

/// A single kernel in a partitioning result.
#[derive(Debug, Clone)]
pub struct Kernel {
    /// Which atoms this kernel computes. Stored as sorted (base, count) ranges
    /// for compactness — these can span across AtomGroup boundaries.
    pub atom_ranges: Vec<AtomRange>,

    /// Atoms that must be loaded from memory (produced by other kernels or external inputs).
    pub external_inputs: HashSet<AtomId>,

    /// Atoms computed by this kernel that are consumed by other kernels (must be stored).
    pub external_outputs: HashSet<AtomId>,
}

/// A contiguous range of AtomIds assigned to a kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AtomRange {
    pub base: AtomId,
    pub count: u32,
}

impl AtomRange {
    pub fn new(base: AtomId, count: u32) -> Self {
        Self { base, count }
    }

    pub fn contains(&self, id: AtomId) -> bool {
        id.0 >= self.base.0 && id.0 < self.base.0 + self.count
    }

    pub fn iter(&self) -> impl Iterator<Item = AtomId> {
        let base = self.base.0;
        let count = self.count;
        (0..count).map(move |i| AtomId(base + i))
    }

    pub fn end(&self) -> AtomId {
        AtomId(self.base.0 + self.count)
    }
}

/// The result of partitioning a NanoGraph into kernels.
#[derive(Debug, Clone)]
pub struct PartitionResult {
    pub kernels: Vec<Kernel>,
    pub config: CacheConfig,
}

impl PartitionResult {
    /// Check that every atom in the graph is assigned to exactly one kernel.
    pub fn validate(&self, graph: &NanoGraph) -> Vec<String> {
        let mut errors = Vec::new();
        let mut atom_to_kernel: HashMap<AtomId, usize> = HashMap::new();

        for (ki, kernel) in self.kernels.iter().enumerate() {
            for range in &kernel.atom_ranges {
                for id in range.iter() {
                    if let Some(prev) = atom_to_kernel.insert(id, ki) {
                        errors.push(format!(
                            "Atom {} assigned to both kernel {} and kernel {}",
                            id, prev, ki
                        ));
                    }
                }
            }
        }

        // Check all graph atoms are covered.
        for group in graph.groups() {
            for id in group.atom_ids() {
                if !atom_to_kernel.contains_key(&id) {
                    errors.push(format!("Atom {} not assigned to any kernel", id));
                }
            }
        }

        errors
    }

    /// Compute total estimated cost across all kernels.
    pub fn total_cost(&self, graph: &NanoGraph) -> PartitionCost {
        let mut kernel_costs = Vec::with_capacity(self.kernels.len());
        let mut total_traffic: u64 = 0;
        let mut total_compute: u64 = 0;

        for kernel in &self.kernels {
            let kc = cost::estimate_kernel_cost(graph, &kernel.atom_ranges, &self.config);
            total_traffic += kc.estimated_traffic;
            total_compute += kc.compute_ops;
            kernel_costs.push(kc);
        }

        let overall_intensity = if total_traffic > 0 {
            total_compute as f64 / total_traffic as f64
        } else {
            f64::INFINITY
        };

        PartitionCost {
            num_kernels: self.kernels.len(),
            total_traffic,
            total_compute,
            overall_intensity,
            kernel_costs,
        }
    }
}

/// Aggregate cost for an entire partitioning.
#[derive(Debug, Clone)]
pub struct PartitionCost {
    pub num_kernels: usize,
    pub total_traffic: u64,
    pub total_compute: u64,
    /// Total compute ops / total memory traffic. Higher is better.
    pub overall_intensity: f64,
    pub kernel_costs: Vec<KernelCost>,
}

impl std::fmt::Display for PartitionCost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Partition Cost:")?;
        writeln!(f, "  {} kernels", self.num_kernels)?;
        writeln!(f, "  total traffic: {} values", self.total_traffic)?;
        writeln!(f, "  total compute: {} ops", self.total_compute)?;
        writeln!(f, "  overall intensity: {:.2} ops/transfer", self.overall_intensity)?;

        // Summarize kernel cost distribution.
        if !self.kernel_costs.is_empty() {
            let max_traffic = self.kernel_costs.iter().map(|k| k.estimated_traffic).max().unwrap();
            let max_peak = self.kernel_costs.iter().map(|k| k.peak_liveness).max().unwrap();
            let min_intensity = self.kernel_costs.iter()
                .map(|k| k.arithmetic_intensity)
                .filter(|i| i.is_finite())
                .fold(f64::INFINITY, f64::min);
            writeln!(f, "  max kernel traffic: {}", max_traffic)?;
            writeln!(f, "  max kernel peak liveness: {}", max_peak)?;
            if min_intensity.is_finite() {
                writeln!(f, "  min kernel intensity: {:.2}", min_intensity)?;
            }
        }
        Ok(())
    }
}

/// Trait that all partitioning algorithms implement.
pub trait Partitioner {
    /// Partition a NanoGraph into kernels for the given cache configuration.
    fn partition(&self, graph: &NanoGraph, config: &CacheConfig) -> PartitionResult;
}
