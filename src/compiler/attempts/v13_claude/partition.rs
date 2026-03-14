//! Common data structures for kernel partitioning.
//!
//! A partition assigns atoms from a NanoGraph into kernels. Each kernel
//! represents a chunk of work whose intermediates fit in a target cache.
//! The quality of a partitioning is measured by total cross-kernel data
//! movement — values that must be loaded from or stored to memory because
//! their producer and consumer are in different kernels.

use std::collections::{HashMap, HashSet};

use crate::nano_graph::{AtomId, NanoGraph};

/// Configuration for the target cache hierarchy.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Size of the tightest cache bucket in bytes (L1, warp cache, SRAM tile).
    /// This is the primary constraint — each kernel's working set should fit here.
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
}

/// Trait that all partitioning algorithms implement.
pub trait Partitioner {
    /// Partition a NanoGraph into kernels for the given cache configuration.
    fn partition(&self, graph: &NanoGraph, config: &CacheConfig) -> PartitionResult;
}
