//! Phase 0 smoke test: an empty NanoGraph compiles via X86JitSpan,
//! the harness can also run it via PoolEvalSpan, and both produce the
//! same (empty) outputs. This is the only thing the new x86_jit can
//! do in phase 0; the harness existing means subsequent phases land
//! into a working test fixture, not a fresh one.

use super::ab_harness::ab_test_bytes;
use crate::compiler::attempts::v14::executor::{CompiledSpanFn, SpanOutput, StoreSlice};
use crate::compiler::attempts::v14::x86_jit::X86JitSpan;
use crate::nano_graph::pattern::NanoGraph;
use crate::pool::SystemPool;

#[test]
fn empty_span_compiles_and_executes() {
    // An empty graph should compile to a `ret`-only function and
    // execute as a no-op without crashing.
    let graph: NanoGraph<'static, SystemPool> = NanoGraph::new();
    let span = X86JitSpan::compile(&graph, &[]).expect("empty span compile");
    let inputs: &[StoreSlice<'_>] = &[];
    let mut outputs: Vec<SpanOutput<'_>> = Vec::new();
    span.execute(inputs, &mut outputs);
}

#[test]
fn empty_span_ab_against_pool_eval() {
    // Both backends agree on the empty graph (vacuously: zero
    // outputs, zero comparisons). Establishes that the harness wiring
    // is sound before phase 2 brings in real op coverage.
    let graph: NanoGraph<'static, SystemPool> = NanoGraph::new();
    let inputs = Vec::new();
    let outputs = Vec::new();
    let agreed = ab_test_bytes(&graph, &inputs, &outputs);
    assert!(agreed.is_empty());
}
