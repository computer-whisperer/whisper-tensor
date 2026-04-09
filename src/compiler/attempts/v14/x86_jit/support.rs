//! `check_supported`: the reject-list filter that decides whether
//! `X86JitSpan::compile` accepts a graph or returns Err for the
//! cranelift fallback.
//!
//! Phase 2.B.4 gate: zero-group graphs **or** all-Identity graphs.
//! All InputRef shapes (Broadcast, Strided N-d, Explicit multi) are
//! now accepted — the address layer handles them all.
//!
//! As phases land, the reject list shrinks. Phase 4's gate is
//! "rejects nothing"; once that holds, the cranelift fallback can be
//! deleted (phase 5).

use crate::nano_graph::ScalarOp;
use crate::nano_graph::pattern::NanoGraph;
use crate::pool::SystemPool;

/// Returns `Ok(())` if `X86JitSpan::compile` should accept this graph,
/// or `Err(reason)` if the caller should fall back to cranelift.
///
/// The reason string is forwarded to the cranelift fallback so it
/// shows up in compile-error logs without losing context.
pub fn check_supported(graph: &NanoGraph<'static, SystemPool>) -> Result<(), String> {
    if graph.num_groups() == 0 {
        return Ok(());
    }

    for (gi, group) in graph.groups().iter().enumerate() {
        // Only Identity bodies are emitted so far.
        if !matches!(group.op, ScalarOp::Identity) {
            return Err(format!(
                "x86_jit: group {gi} op {:?} not yet supported (Identity only)",
                group.op
            ));
        }

        // Identity expects exactly one input.
        if group.inputs.len() != 1 {
            return Err(format!(
                "x86_jit: group {gi} Identity has {} inputs, expected 1",
                group.inputs.len()
            ));
        }
    }

    Ok(())
}
