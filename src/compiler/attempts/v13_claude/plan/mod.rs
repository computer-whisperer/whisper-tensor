#![allow(clippy::all, dead_code, unreachable_patterns, unused_variables, unused_imports)]
//! Span-based execution plan types and planner implementations.
//!
//! A `SpanPlan` splits a NanoGraph into phases (separated by barriers) and
//! lanes (parallel work units). Each lane in each phase gets a self-contained
//! `Span` with its own NanoGraph, input/output mappings, and literal mappings.

use crate::nano_graph::{AtomId, NanoGraph};

pub mod spans_c;
pub mod v2c;
pub mod v4c;

/// A contiguous range of atoms mapped between main graph and span graph.
#[derive(Debug, Clone)]
pub struct AtomMapping {
    pub main_base: AtomId,
    pub span_base: AtomId,
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
    /// Main↔span mapping for inlined literal groups (needed to feed
    /// tensor overrides into NanoEval during execution).
    pub literal_mappings: Vec<AtomMapping>,
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
