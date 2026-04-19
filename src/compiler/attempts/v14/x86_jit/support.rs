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
//!
//! SymReduce is admitted only in the narrow "1 sym → 0 sym" shape
//! (producer has exactly one sym axis, consumer has none, axis=0).
//! The N → N-1 general form still routes to pool_eval.

use crate::nano_graph::ScalarOp;
use crate::nano_graph::pattern::{InputRef, NanoGraph};
use crate::pool::SystemPool;

/// Returns `Ok(())` if `X86JitSpan::compile` should accept this graph,
/// or `Err(reason)` if the caller should fall back to cranelift.
///
/// The reason string is forwarded to the cranelift fallback so it
/// shows up in compile-error logs without losing context.
pub fn check_supported(
    graph: &NanoGraph<'static, SystemPool>,
    external_input_sym_dims: &crate::range_map::RangeMap<
        Vec<crate::nano_graph::pattern::GraphConstantId>,
    >,
) -> Result<(), String> {
    if graph.num_groups() == 0 {
        return Ok(());
    }

    for (gi, group) in graph.groups().iter().enumerate() {
        // Sym-bearing groups accept: Identity, Cast, Binary, Unary,
        // Select, Reduce, IndirectLoad, GcLiteral (all with the
        // "consumer sym axes align 1:1 with producer's" pattern), and
        // SymReduce (with the "skip-axis Identity" pattern since the
        // consumer has one fewer sym axis than the producer).
        //
        // Reduce + inline producer with sym_dims returns Err at emit
        // time (cranelift fallback) — the outer-loop-only version of
        // Reduce sym is handled here. IndirectLoad's table lookup is
        // sym-independent by design (the table IS the gather table,
        // not per-sym) — only the index load and output store carry
        // sym_ctx. GcLiteral writes a runtime-resolved scalar that is
        // constant across every (atom, sym_flat) slot in the group;
        // only the output store is sym-aware.
        if !group.sym_dims.is_empty() {
            if !matches!(
                group.op,
                ScalarOp::Identity
                    | ScalarOp::Cast { .. }
                    | ScalarOp::Binary { .. }
                    | ScalarOp::Unary { .. }
                    | ScalarOp::Select
                    | ScalarOp::Reduce { .. }
                    | ScalarOp::IndirectLoad { .. }
                    | ScalarOp::GcLiteral(_)
                    | ScalarOp::SymReduce { .. },
            ) {
                return Err(format!(
                    "x86_jit: group {gi} op {:?} with sym_dims not yet supported",
                    group.op
                ));
            }
            // Diagnostic gates — set WT_SYMJIT_REJECT=Bin,Id,Ind,Un,Sel,Red,Cast,Gc,SRed
            // to route specific sym ops to pool_eval fallback. Used to
            // bisect which sym codegen is producing wrong addresses.
            if let Ok(rej) = std::env::var("WT_SYMJIT_REJECT") {
                for tag in rej.split(',') {
                    let t = tag.trim();
                    let matches_op = match t {
                        "Bin" => matches!(group.op, ScalarOp::Binary { .. }),
                        "Un" => matches!(group.op, ScalarOp::Unary { .. }),
                        "Sel" => matches!(group.op, ScalarOp::Select),
                        "Red" => matches!(group.op, ScalarOp::Reduce { .. }),
                        "Cast" => matches!(group.op, ScalarOp::Cast { .. }),
                        "Id" => matches!(group.op, ScalarOp::Identity),
                        "Ind" => matches!(group.op, ScalarOp::IndirectLoad { .. }),
                        "Gc" => matches!(group.op, ScalarOp::GcLiteral(_)),
                        "SRed" => matches!(group.op, ScalarOp::SymReduce { .. }),
                        _ => false,
                    };
                    if matches_op {
                        return Err(format!(
                            "x86_jit: sym op {:?} rejected by WT_SYMJIT_REJECT={tag}",
                            group.op
                        ));
                    }
                }
            }
            // Elementwise-style ops (everything except SymReduce) require
            // each input's sym_dim_map to be either:
            //   (a) empty — no sym component, input is scalar.
            //   (b) len == consumer.sym_dims.len(), and every Identity(k)
            //       entry has k in bounds for the producer.
            //
            // The address layer handles three shapes:
            //   - All-Broadcast: skipped (producer atom is sym-invariant).
            //   - All-Identity(j == j): fast path, producer flat == sym_i.
            //   - Other (mixed Broadcast + Identity, or non-same-index
            //     Identity): general remap in `apply_sym_offset_pub`.
            //
            // SymReduce is a separate pattern — the consumer has
            // exactly one fewer sym axis than the producer, so the
            // map uses `Identity(p(j))` where `p(j) = j if j < axis
            // else j + 1`. Checked in the SymReduce arm below.
            if !matches!(group.op, ScalarOp::SymReduce { .. }) {
                let expected_len = group.sym_dims.len();
                for (ii, inp) in group.inputs.iter().enumerate() {
                    let map = &inp.sym_dim_map;
                    if map.is_empty() {
                        continue;
                    }
                    if map.len() != expected_len {
                        return Err(format!(
                            "x86_jit: group {gi} {:?} input {ii} sym_dim_map has {} entries, \
                             expected {expected_len} (or empty)",
                            group.op,
                            map.len()
                        ));
                    }
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
            ScalarOp::SymReduce { axis, .. } => {
                // Consumer has N-1 sym_dims; producer has N. `axis`
                // indexes the producer's sym_dims list. Map entries
                // follow the skip-axis Identity pattern:
                //   map[j_c] == Identity(j_c)       for j_c < axis
                //   map[j_c] == Identity(j_c + 1)   for j_c >= axis
                if group.inputs.len() != 1 {
                    return Err(format!(
                        "x86_jit: group {gi} SymReduce has {} inputs, expected 1",
                        group.inputs.len()
                    ));
                }
                let n_consumer = group.sym_dims.len();
                let prod_sym_len = producer_sym_dim_count(
                    graph,
                    external_input_sym_dims,
                    &group.inputs[0].input_ref,
                )?;
                if prod_sym_len != n_consumer + 1 {
                    return Err(format!(
                        "x86_jit: group {gi} SymReduce producer sym_dims {prod_sym_len} \
                         != consumer sym_dims {n_consumer} + 1"
                    ));
                }
                if *axis >= prod_sym_len {
                    return Err(format!(
                        "x86_jit: group {gi} SymReduce axis={axis} out of range \
                         for producer with {prod_sym_len} sym_dims"
                    ));
                }
                let map = &group.inputs[0].sym_dim_map;
                if map.len() != n_consumer {
                    return Err(format!(
                        "x86_jit: group {gi} SymReduce input sym_dim_map has {} \
                         entries, expected {n_consumer}",
                        map.len()
                    ));
                }
                for (j_c, entry) in map.iter().enumerate() {
                    let expected_p = if j_c < *axis { j_c } else { j_c + 1 };
                    match entry {
                        crate::nano_graph::pattern::SymDimMap::Identity(k) if *k == expected_p => {}
                        _ => {
                            return Err(format!(
                                "x86_jit: group {gi} SymReduce input sym_dim_map[{j_c}] = \
                                 {entry:?}, expected Identity({expected_p})"
                            ));
                        }
                    }
                }
            }
            ScalarOp::Literal(_) | ScalarOp::LiteralSpan(_) => {}
            ScalarOp::GcLiteral(_) => {
                if !group.inputs.is_empty() {
                    return Err(format!(
                        "x86_jit: group {gi} GcLiteral has {} inputs, expected 0",
                        group.inputs.len()
                    ));
                }
            }
            op => {
                return Err(format!("x86_jit: group {gi} op {op:?} not yet supported"));
            }
        }
    }

    Ok(())
}

/// Resolve a SymReduce input's producer and return how many sym_dims
/// the producer carries. Producers may be a NanoGraph group (read
/// sym_dims directly) or an external input tensor (look it up in the
/// external sym_dims RangeMap).
fn producer_sym_dim_count(
    graph: &NanoGraph<'static, SystemPool>,
    external_input_sym_dims: &crate::range_map::RangeMap<
        Vec<crate::nano_graph::pattern::GraphConstantId>,
    >,
    input_ref: &InputRef,
) -> Result<usize, String> {
    let probe = match input_ref {
        InputRef::Broadcast(id) => *id,
        InputRef::Strided { base, .. } => *base,
        InputRef::Explicit(ids) => *ids
            .first()
            .ok_or_else(|| "x86_jit: SymReduce Explicit input is empty".to_string())?,
    };
    if let Some(gi) = graph.find_group_idx(probe) {
        return Ok(graph.groups()[gi].sym_dims.len());
    }
    if let Some((sym_dims, _)) = external_input_sym_dims.get(probe.0) {
        return Ok(sym_dims.len());
    }
    Ok(0)
}
