//! Core data types for the execution plan.
//!
//! # Pipeline
//!
//! ```text
//! MilliOpGraph ─── lower() ──→ NanoGraph ─── plan() ──→ ExecutionPlan
//!                                                            │
//!                                                       compile()
//!                                                            │
//!                                                            ▼
//!                                                      CompiledPlan
//! ```
//!
//! # Execution model
//!
//! An ExecutionPlan partitions a NanoGraph into **phases** separated by
//! barriers. Within each phase, independent **spans** execute in parallel
//! across **lanes**. Each span has its own NanoGraph — a fragment of the
//! main graph that the planner may have split or rearranged, but which
//! preserves atom IDs from the main graph.
//!
//! Data flows between spans through a shared **value store** keyed by AtomId.
//! Before phase 0, weights and user inputs are loaded into the store.
//! After each phase, spans' outputs are available in the store for
//! subsequent phases to read.
//!
//! # Atom ID invariant
//!
//! Span NanoGraphs use the **same atom ID space** as the main graph.
//! An atom at id X in a span is the same atom as id X in the main graph.
//! This means:
//! - No remapping tables between span and main graph.
//! - Verification is direct: evaluate both and compare values at matching IDs.
//! - The planner can split groups (changing group boundaries) without
//!   inventing new IDs — the atoms keep their original identity.
//!
//! # Structural invariants
//!
//! - Spans within a phase are independent: no data flow between them.
//! - Every atom ID referenced by a span's InputRefs is either produced by
//!   a group in that span, or covered by an entry in `inputs`.
//! - Every atom in `outputs` is produced by a group in the span.
//! - The union of all spans' groups covers every group in the main graph
//!   (nothing is dropped).

use std::collections::HashMap;

use crate::graph::GlobalId;
pub use crate::nano_graph::{AtomId, AtomRange, NanoGraph, SymDim};

// ─── Execution plan ─────────────────────────────────────────────────────────

/// Complete execution plan for a lowered model.
///
/// Owns the main NanoGraph (the source of truth) and the partitioned
/// span NanoGraphs that will be compiled and executed.
pub struct ExecutionPlan {
    /// The original lowered NanoGraph, kept for validation and metadata.
    /// Span NanoGraphs are derived from this but may have different group
    /// boundaries (splits, rearrangements).
    pub graph: NanoGraph,

    /// How model-level tensors map to atom ranges in the graph.
    /// The executor uses this to know which atom ranges to fill from
    /// the TensorStore (weights) or from user-provided data (inputs).
    pub tensor_map: HashMap<GlobalId, TensorMapping>,

    /// Sequential phases of execution. A barrier separates each phase.
    pub phases: Vec<Phase>,

    /// Final model outputs to extract after all phases complete.
    pub model_outputs: Vec<OutputMapping>,
}

// ─── Tensor metadata ────────────────────────────────────────────────────────

/// How a model-level tensor (identified by GlobalId) maps to atoms.
///
/// This is the bridge between the MilliOpGraph world (named tensors with
/// shapes) and the NanoGraph world (anonymous atom IDs). The executor
/// uses this to load weight data into the right atom ranges and to
/// extract output tensors from atom ranges after execution.
pub struct TensorMapping {
    /// Atom range for this tensor's data.
    pub range: AtomRange,
    /// Symbolic dimensions this tensor varies over (e.g. batch, seq_len).
    /// Empty for fixed-size tensors like weights.
    pub sym_dims: Vec<SymDim>,
    /// Where the data comes from.
    pub kind: TensorKind,
}

/// Where a tensor's data comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorKind {
    /// Weight: loaded from TensorStore once before execution.
    Weight,
    /// Input: provided by the caller for each inference.
    Input,
    /// Computed: produced by groups during execution.
    Computed,
}

/// A model output to return to the caller after execution.
pub struct OutputMapping {
    pub tensor_id: GlobalId,
    pub range: AtomRange,
    pub sym_dims: Vec<SymDim>,
}

// ─── Phases and spans ───────────────────────────────────────────────────────

/// A phase of execution.
///
/// All spans in a phase are independent and may execute in parallel.
/// The barrier after each phase ensures all spans complete before the
/// next phase begins.
pub struct Phase {
    /// One span per lane. `spans[i]` is lane i's work in this phase.
    /// A span with an empty NanoGraph is valid (idle lane).
    pub spans: Vec<Span>,
}

/// A span: one lane's work within one phase.
///
/// Contains a NanoGraph fragment — a subset of the main graph's atoms
/// with the same atom IDs. The planner may split or rearrange groups
/// compared to the main graph, but atom semantics are preserved.
///
/// Self-contained: its data dependencies are fully described by its
/// groups (internal) and `inputs` (external). The NanoGraph can be
/// compiled or evaluated independently given the input data.
pub struct Span {
    /// NanoGraph fragment for this span's computation.
    ///
    /// Uses the main graph's atom ID space (no remapping).
    /// Groups may differ from the main graph: the planner can split
    /// a main-graph group into smaller groups, duplicate shared
    /// computation, or reorder groups for better locality.
    ///
    /// Atom IDs referenced by InputRefs that are not produced by any
    /// group in this graph are external — declared in `inputs`.
    pub graph: NanoGraph,

    /// Atom ranges this span reads from the shared value store.
    /// Includes weight data, user inputs, and prior-phase outputs.
    /// The executor resolves the source of each range by consulting
    /// the ExecutionPlan's tensor_map.
    pub inputs: Vec<AtomRange>,

    /// Atom ranges this span produces that are needed after this phase.
    /// Only includes atoms consumed by later phases or model outputs.
    pub outputs: Vec<AtomRange>,
}
