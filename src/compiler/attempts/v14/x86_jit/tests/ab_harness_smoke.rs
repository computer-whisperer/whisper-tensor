//! Phase 0 smoke test: an empty NanoGraph compiles via X86JitSpan,
//! the harness can also run it via PoolEvalSpan, and both produce the
//! same (empty) outputs. The harness existing means subsequent phases
//! land into a working test fixture, not a fresh one.

use super::ab_harness::ab_test_bytes;
use crate::compiler::attempts::v14::executor::CompiledSpanFn;
use crate::compiler::attempts::v14::placer::run_placer;
use crate::compiler::attempts::v14::types::{Phase, Span};
use crate::compiler::attempts::v14::x86_jit::X86JitSpan;
use crate::nano_graph::pattern::NanoGraph;
use crate::pool::SystemPool;

#[test]
fn empty_span_compiles_and_executes() {
    // An empty graph should compile to a prologue+epilogue-only
    // function and execute as a no-op without crashing.
    let graph: NanoGraph<'static, SystemPool> = NanoGraph::new();
    let phases = vec![Phase {
        spans: vec![Span {
            graph: graph.clone(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }],
    }];
    let placement = run_placer(&graph, &phases, &[], &std::collections::HashMap::new())
        .expect("empty placement");
    let span = X86JitSpan::compile(&graph, &[], &placement).expect("empty span compile");

    // Build a minimal buffer_ptrs sized to the scratch slot. Every
    // slot is null/empty since the empty graph needs no data.
    let len = (placement.scratch_buffer_id as usize) + 1;
    let buffer_ptrs: Vec<*mut u8> = vec![std::ptr::null_mut(); len];
    span.execute(&buffer_ptrs, &std::collections::HashMap::new());
}

#[test]
fn empty_span_ab_against_pool_eval() {
    // Both backends agree on the empty graph (vacuously: zero
    // outputs, zero comparisons). Establishes that the harness wiring
    // is sound before later phases bring in real op coverage.
    let graph: NanoGraph<'static, SystemPool> = NanoGraph::new();
    let inputs = Vec::new();
    let outputs = Vec::new();
    let agreed = ab_test_bytes(&graph, &inputs, &outputs);
    assert!(agreed.is_empty());
}
