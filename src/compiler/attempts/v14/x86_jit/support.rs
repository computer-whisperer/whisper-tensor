//! `check_supported`: the reject-list filter that decides whether
//! `X86JitSpan::compile` accepts a graph or returns Err for the
//! cranelift fallback.
//!
//! Phase 2.B.3 gate: zero-group graphs **or** all-Identity graphs
//! whose inputs use only the InputRef shapes
//! [`super::orch::address`] currently handles (Broadcast, 1D Strided,
//! single-element Explicit) and whose source dtype matches the output
//! dtype.
//!
//! As phases land, the reject list shrinks. Phase 4's gate is
//! "rejects nothing"; once that holds, the cranelift fallback can be
//! deleted (phase 5).

use crate::nano_graph::ScalarOp;
use crate::nano_graph::pattern::{InputRef, NanoGraph};
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
        // P2.B.3 only knows how to emit Identity bodies.
        if !matches!(group.op, ScalarOp::Identity) {
            return Err(format!(
                "x86_jit: group {gi} op {:?} not yet supported (P2.B.3 = Identity only)",
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

        // Restrict the InputRef to shapes orch::address can resolve.
        match &group.inputs[0] {
            InputRef::Broadcast(_) => {}
            InputRef::Strided { dim_strides, .. } if dim_strides.len() == 1 => {}
            InputRef::Strided { dim_strides, .. } => {
                return Err(format!(
                    "x86_jit: group {gi} Strided n-d (nd={}) not yet supported (P2.B.4)",
                    dim_strides.len()
                ));
            }
            InputRef::Explicit(ids) if ids.len() == 1 => {}
            InputRef::Explicit(ids) => {
                return Err(format!(
                    "x86_jit: group {gi} Explicit({}) not yet supported (P2.B.4)",
                    ids.len()
                ));
            }
        }
    }

    Ok(())
}
