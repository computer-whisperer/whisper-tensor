//! `check_supported`: the reject-list filter that decides whether
//! `X86JitSpan::compile` accepts a graph or returns Err for the
//! cranelift fallback.
//!
//! Phase 2.B.5 gate: zero-group graphs **or** graphs containing only
//! Identity, Cast, Literal, and LiteralSpan ops. All InputRef shapes
//! are accepted. Cross-compute-repr Cast (float↔int) is rejected
//! at emit time in `orch::group` and falls back to cranelift.
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
        // Phase 4 sym support: Identity, Cast, Binary, Unary, Select,
        // Reduce, and IndirectLoad groups accept sym_dims. SymReduce
        // still falls back (semantically distinct). Reduce + inline
        // producer with sym_dims returns Err at emit time (cranelift
        // fallback) — the outer-loop-only version of Reduce sym is
        // handled here. IndirectLoad's table lookup is sym-independent
        // by design (the table IS the gather table, not per-sym) —
        // only the index load and output store carry sym_ctx.
        if !group.sym_dims.is_empty() {
            if !matches!(
                group.op,
                ScalarOp::Identity
                    | ScalarOp::Cast { .. }
                    | ScalarOp::Binary { .. }
                    | ScalarOp::Unary { .. }
                    | ScalarOp::Select
                    | ScalarOp::Reduce { .. }
                    | ScalarOp::IndirectLoad { .. },
            ) {
                return Err(format!(
                    "x86_jit: group {gi} op {:?} with sym_dims not yet supported \
                     (Phase 4: Identity+Cast+Binary+Unary+Select+Reduce+IndirectLoad only)",
                    group.op
                ));
            }
            // Require identity sym_dim_map on every input: consumer
            // axis j reads from producer axis j. Producer and consumer
            // sym_dims match, so sym_flat indexes the same byte across
            // both. (Cast has a single input like Identity; loop here
            // anyway to stay general for future ops.)
            let expected_len = group.sym_dims.len();
            for (ii, inp) in group.inputs.iter().enumerate() {
                let map = &inp.sym_dim_map;
                if map.len() != expected_len
                    || !map.iter().enumerate().all(|(j, m)| {
                        matches!(m, crate::nano_graph::pattern::SymDimMap::Identity(k) if *k == j)
                    })
                {
                    return Err(format!(
                        "x86_jit: group {gi} {:?} input {ii} with non-identity sym_dim_map \
                         not yet supported (Phase 2a: identity mapping only)",
                        group.op
                    ));
                }
            }
        }
        match &group.op {
            ScalarOp::Identity | ScalarOp::Cast { .. } => {
                if group.inputs.len() != 1 {
                    return Err(format!(
                        "x86_jit: group {gi} {:?} has {} inputs, expected 1",
                        group.op,
                        group.inputs.len()
                    ));
                }
            }
            ScalarOp::Binary { .. } => {
                if group.inputs.len() != 2 {
                    return Err(format!(
                        "x86_jit: group {gi} Binary has {} inputs, expected 2",
                        group.inputs.len()
                    ));
                }
            }
            ScalarOp::Unary { .. } => {
                if group.inputs.len() != 1 {
                    return Err(format!(
                        "x86_jit: group {gi} Unary has {} inputs, expected 1",
                        group.inputs.len()
                    ));
                }
            }
            ScalarOp::Select => {
                if group.inputs.len() != 3 {
                    return Err(format!(
                        "x86_jit: group {gi} Select has {} inputs, expected 3",
                        group.inputs.len()
                    ));
                }
            }
            ScalarOp::Reduce { .. } => {
                if group.inputs.len() != 1 {
                    return Err(format!(
                        "x86_jit: group {gi} Reduce has {} inputs, expected 1",
                        group.inputs.len()
                    ));
                }
            }
            ScalarOp::IndirectLoad { .. } => {
                if group.inputs.len() != 1 {
                    return Err(format!(
                        "x86_jit: group {gi} IndirectLoad has {} inputs, expected 1",
                        group.inputs.len()
                    ));
                }
            }
            ScalarOp::Literal(_) | ScalarOp::LiteralSpan(_) => {}
            op => {
                return Err(format!("x86_jit: group {gi} op {op:?} not yet supported"));
            }
        }
    }

    Ok(())
}
