//! Optimization passes on the NanoGraph.
//!
//! TODO: These passes need to be rewritten for the v2 AtomGroup-based
//! representation. The old PatternBlock-based passes have been removed.

use crate::nano_graph::pattern::NanoGraph;

/// Result of running an optimization pass.
pub struct OptResult {
    pub graph: NanoGraph,
    /// Number of groups/atoms eliminated by this pass.
    pub eliminated: usize,
}

/// Remove atoms not reachable backward from outputs.
pub fn dce(graph: NanoGraph) -> OptResult {
    // TODO: implement for v2
    OptResult {
        graph,
        eliminated: 0,
    }
}

/// Algebraic simplifications on binary ops with literal operands.
pub fn strength_reduce(graph: NanoGraph) -> OptResult {
    // TODO: implement for v2
    OptResult {
        graph,
        eliminated: 0,
    }
}

/// Deduplicate groups that have the same structure and inputs.
pub fn cse(graph: NanoGraph) -> OptResult {
    // TODO: implement for v2
    OptResult {
        graph,
        eliminated: 0,
    }
}
