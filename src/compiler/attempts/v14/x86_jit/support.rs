//! `check_supported`: the reject-list filter that decides whether
//! `X86JitSpan::compile` accepts a graph or returns Err for the
//! cranelift fallback.
//!
//! Phase 2.B.1: still rejects everything except zero-group graphs.
//! The pipeline machinery is in place; the next sub-phases (P2.B.3+)
//! widen this gate as ops come online.
//!
//! As phases land, the reject list shrinks. Phase 4's gate is
//! "rejects nothing"; once that holds, the cranelift fallback can be
//! deleted (phase 5).

use crate::nano_graph::pattern::NanoGraph;
use crate::pool::SystemPool;

/// Returns `Ok(())` if `X86JitSpan::compile` should accept this graph,
/// or `Err(reason)` if the caller should fall back to cranelift.
///
/// The reason string is forwarded to the cranelift fallback so it
/// shows up in compile-error logs without losing context.
pub fn check_supported(graph: &NanoGraph<'static, SystemPool>) -> Result<(), String> {
    if graph.num_groups() != 0 {
        // Phase 2.B.1 gate: only empty graphs compile via the new
        // x86_jit. Identity/Cast land in P2.B.3.
        return Err("x86_jit: rewrite in progress".to_string());
    }
    Ok(())
}
