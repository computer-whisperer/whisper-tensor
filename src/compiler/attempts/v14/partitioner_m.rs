#![allow(
    clippy::all,
    dead_code,
    unreachable_code,
    unreachable_patterns,
    unused_imports,
    unused_variables
)]

//! Spatial-tiling partitioner (attempt M)
//!
//! Core idea: lane assignment IS tiling. Lane k always gets the k-th slice of
//! every splittable group. This means a serial chain of elementwise ops needs
//! ZERO barriers — each lane processes its slice of the whole chain locally.
//!
//! Algorithm:
//! 1. Build group-level dependency DAG
//! 2. Classify each group: Splittable / Duplicate / Whole
//! 3. Process groups in topological order, assigning them to phases.
//!    - A group stays in the current phase if all its dependencies are either:
//!      (a) from a prior phase (already available via value store), or
//!      (b) from the current phase AND on the same lane(s)
//!    - A new phase (barrier) is only needed when a group reads cross-lane
//!      data produced in the current phase.
//! 4. Build span NanoGraphs: split groups get fragment-per-lane, duplicate
//!    groups get full copies in each consuming lane, whole groups go to lane 0.
//!
//! The key invariant: within a phase, no span reads atoms produced by another
//! span. Splittable groups are split identically (by atom range), so each
//! lane's fragment is self-contained. Barriers only appear at reduce boundaries
//! where partial results from multiple lanes must be gathered.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::graph::GlobalId;
use crate::nano_graph::pattern::InputTensor;
use crate::nano_graph::{AtomGroup, AtomId, AtomRange, InputRef, NanoGraph, ScalarOp, SymDim};
use crate::numeric_dtype::NumericDType;

use super::types::{Phase, Span};

type DeclaredInputKey = (u64, u64, NumericDType, Option<GlobalId>);

// ─── Matmul relayout ──────────────────────────────────────────────────────

fn is_strided_broadcast_repeat(input: &InputRef, repeat: u64) -> bool {
    match input {
        InputRef::Strided {
            dim_strides,
            dim_shape,
            ..
        } => {
            dim_strides.len() == 2
                && dim_shape.len() == 2
                && dim_strides[1] == 0
                && dim_shape[1] == repeat
        }
        _ => false,
    }
}

fn is_affine_strided(input: &InputRef) -> Option<i64> {
    match input {
        InputRef::Strided {
            dim_strides,
            dim_shape,
            ..
        } if dim_strides.len() == 1 && dim_shape.len() == 1 && dim_shape[0] == u64::MAX => {
            Some(dim_strides[0])
        }
        _ => None,
    }
}

fn is_strided_transposed_nk(input: &InputRef, n: u64, k: u64) -> Option<i64> {
    match input {
        InputRef::Strided {
            dim_strides,
            dim_shape,
            ..
        } if dim_strides.len() == 2 && dim_shape.len() == 2 && dim_shape[1] == n => {
            let outer = dim_strides[0];
            let Some(expected_inner) = outer.checked_mul(k as i64) else {
                return None;
            };
            if dim_strides[1] == expected_inner {
                Some(outer)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Detect matmul Mul→Reduce pairs in [K,N] layout and transpose to [N,K].
///
/// Pattern detected (per pair):
///   Mul group:   count = K*N, op = Binary{Mul}
///     input_a:   StridedBroadcast(A_base, a_stride, repeat=N)  [A element per K-block]
///     input_b:   Affine(B_base, stride=1)                      [B row-major K×N]
///   Reduce group: count = N, op = Reduce{count=K, stride=N}
///     input:     Affine(Mul_base, stride=1)
///
/// Transformed to:
///   Mul group:   count = K*N (unchanged), op = Binary{Mul} (unchanged)
///     input_a:   Modular(A_base, a_stride, modulus=K)
///     input_b:   Strided(B_base, dim_strides=[1, N], dim_shape=[MAX, K])
///   Reduce group: count = N (unchanged), op = Reduce{count=K, stride=1}
///     input:     Affine(Mul_base, stride=K)
///
/// Returns the number of pairs relayouted.
fn relayout_matmul_groups(graph: &mut NanoGraph<'static, crate::pool::SystemPool>) -> usize {
    let mut count = 0;

    // First pass: find Reduce groups and their Mul producers.
    // Collect (mul_group_idx, reduce_group_idx, K, N) tuples.
    let mut pairs: Vec<(usize, usize, u64, u64)> = Vec::new();

    let groups = graph.groups();
    for (ri, rgroup) in groups.iter().enumerate() {
        let (k, n_stride) = match &rgroup.op {
            ScalarOp::Reduce {
                reduce_count,
                reduce_stride,
                ..
            } if *reduce_count > 1 && *reduce_stride > 0 => (*reduce_count, *reduce_stride),
            _ => continue,
        };

        let n = rgroup.count;
        if n_stride != n as i64 {
            continue; // reduce_stride must equal N for the [K,N] pattern
        }

        // Check the Reduce's input: must be Affine(Mul_base, stride=1).
        if rgroup.inputs.len() != 1 {
            continue;
        }
        let (mul_base, mul_stride) = match &rgroup.inputs[0].input_ref {
            InputRef::Strided {
                base,
                dim_strides,
                dim_shape,
            } if dim_strides.len() == 1 && dim_strides[0] == 1 && dim_shape[0] == u64::MAX => {
                (*base, dim_strides[0])
            }
            _ => continue,
        };
        if mul_stride != 1 {
            continue;
        }

        // Find the Mul group at this base.
        let Some(mi) = graph.find_group_idx(mul_base) else {
            continue;
        };
        let mgroup = &groups[mi];

        // Verify it's a Binary{Mul} with count = K*N.
        let is_mul = matches!(
            &mgroup.op,
            ScalarOp::Binary {
                op: crate::nano_graph::ScalarBinOp::Mul,
                ..
            }
        );
        if !is_mul || mgroup.count != k * n {
            continue;
        }

        // Verify the Mul's inputs match a [K,N]-ordered matmul expansion:
        // One input is StridedBroadcast(_, _, repeat=N) — the A input.
        // The other input is either:
        //   - Affine(base, stride=s), or
        //   - Strided(base, [s, s*K], [MAX, N]) (equivalent transposed view).
        if mgroup.inputs.len() != 2 {
            continue;
        }
        let sb_idx = mgroup
            .inputs
            .iter()
            .position(|inp| is_strided_broadcast_repeat(&inp.input_ref, n));
        let Some(sb_idx) = sb_idx else {
            continue;
        };
        let other_idx = if sb_idx == 0 { 1 } else { 0 };
        let other = &mgroup.inputs[other_idx].input_ref;
        let other_ok =
            is_affine_strided(other).is_some() || is_strided_transposed_nk(other, n, k).is_some();
        if !other_ok {
            continue;
        }

        pairs.push((mi, ri, k, n));
    }

    // Second pass: apply transformations.
    let groups = graph.groups_mut();
    for (mi, ri, k, n) in pairs {
        let mgroup = &mut groups[mi];

        // Transform Mul inputs from [K,N] to [N,K] layout.
        for inp in &mut mgroup.inputs {
            match &mut inp.input_ref {
                // StridedBroadcast(A_base, a_stride, repeat=N)
                //   → Modular(A_base, a_stride, modulus=K)
                InputRef::Strided {
                    dim_strides,
                    dim_shape,
                    ..
                } if dim_strides.len() == 2 && dim_strides[1] == 0 && dim_shape[1] == n => {
                    // Was: dim_strides=[a_stride, 0], dim_shape=[MAX, N] (StridedBroadcast)
                    // New: dim_strides=[0, a_stride], dim_shape=[MAX, K] (Modular)
                    let a_stride = dim_strides[0];
                    dim_strides[0] = 0;
                    dim_strides[1] = a_stride;
                    dim_shape[1] = k;
                }
                // Affine(B_base, stride=s)
                //   → Strided(B_base, dim_strides=[s, s*N], dim_shape=[MAX, K])
                InputRef::Strided {
                    dim_strides,
                    dim_shape,
                    ..
                } if dim_strides.len() == 1 && dim_shape[0] == u64::MAX => {
                    let s = dim_strides[0];
                    // Was: 1D affine stride=s
                    // New: 2D with dim_strides=[s, s*N], dim_shape=[MAX, K]
                    if let Some(inner) = s.checked_mul(n as i64) {
                        *dim_strides = vec![s, inner];
                        *dim_shape = vec![u64::MAX, k];
                    }
                }
                // Strided(B_base, [s, s*K], [MAX, N])
                //   → Affine(B_base, stride=s)
                InputRef::Strided {
                    dim_strides,
                    dim_shape,
                    ..
                } if dim_strides.len() == 2
                    && dim_shape.len() == 2
                    && dim_shape[1] == n
                    && dim_strides[0]
                        .checked_mul(k as i64)
                        .is_some_and(|expected_inner| dim_strides[1] == expected_inner) =>
                {
                    let s = dim_strides[0];
                    *dim_strides = vec![s];
                    *dim_shape = vec![u64::MAX];
                }
                _ => {}
            }
        }

        // Transform Reduce: stride=1→stride=K, reduce_stride=N→1.
        let rgroup = &mut groups[ri];
        if let Some(inp) = rgroup.inputs.first_mut() {
            match &mut inp.input_ref {
                InputRef::Strided {
                    dim_strides,
                    dim_shape,
                    ..
                } if dim_strides.len() == 1 && dim_strides[0] == 1 && dim_shape[0] == u64::MAX => {
                    dim_strides[0] = k as i64;
                }
                _ => {}
            }
        }
        if let ScalarOp::Reduce { reduce_stride, .. } = &mut rgroup.op {
            *reduce_stride = 1;
        }

        count += 1;
    }

    count
}

// ─── Group classification ──────────────────────────────────────────────────

/// How a group should be handled during partitioning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GroupKind {
    /// Split across lanes (elementwise, matmul components, etc.).
    /// Each lane gets count/num_lanes atoms.
    Split,
    /// Duplicate into every lane that needs it (literals, small scalars).
    Duplicate,
    /// Keep whole on a single lane (unsplittable reduces, tiny groups, explicit-input groups).
    Whole,
}

/// Classify a group based on its op and structure.
fn classify_group(
    group: &AtomGroup<'static, crate::pool::SystemPool>,
    num_lanes: usize,
) -> GroupKind {
    // Literals are always duplicated — every lane may need the values.
    if matches!(group.op, ScalarOp::Literal(_) | ScalarOp::LiteralSpan(_)) {
        return GroupKind::Duplicate;
    }

    // Very small groups aren't worth splitting.
    if group.count < num_lanes as u64 {
        if group.count <= 1 {
            return GroupKind::Duplicate; // scalar — just duplicate
        }
        return GroupKind::Whole;
    }

    match &group.op {
        ScalarOp::Literal(_) | ScalarOp::LiteralSpan(_) => unreachable!(), // handled above

        // Elementwise ops: embarrassingly parallel, always split.
        ScalarOp::Binary { .. }
        | ScalarOp::Unary { .. }
        | ScalarOp::Select
        | ScalarOp::Identity
        | ScalarOp::Cast { .. } => {
            // Groups with Explicit inputs can't be trivially split because
            // InputRef::Explicit stores a Vec<AtomId> that must have exactly
            // group.count entries, and the atom_offset mechanism doesn't compose
            // with sliced Explicit vectors. Keep these whole.
            let has_explicit = group
                .inputs
                .iter()
                .any(|inp| matches!(inp.input_ref, InputRef::Explicit(_)));
            if has_explicit {
                GroupKind::Whole
            } else {
                GroupKind::Split
            }
        }

        // IndirectLoad: each lookup is independent, split freely.
        ScalarOp::IndirectLoad { .. } => GroupKind::Split,

        // Opaque ops: can't be split — the eval function sees the whole tensor.
        ScalarOp::OpaqueOutput { .. } => GroupKind::Whole,

        ScalarOp::Reduce {
            reduce_count,
            reduce_stride,
            ..
        } => {
            // Each output element of a reduce is an independent accumulation.
            // If we have M output elements (group.count = M), we can split
            // those M independent reductions across lanes.
            //
            // The key question: does the reduce's INPUT span cross lane
            // boundaries? If the reduce reads from a group that was split
            // across lanes, and the reduce_stride causes it to read atoms
            // from different lanes, then we need a barrier first.
            //
            // But that's a *scheduling* concern (phase assignment), not a
            // *splitting* concern. The reduce itself is always splittable
            // by its output dimension — each of the M reductions is independent.
            if group.count >= num_lanes as u64 {
                GroupKind::Split
            } else if group.count > 1 {
                GroupKind::Whole
            } else {
                // Single-element reduce (scalar output) — duplicate downstream.
                GroupKind::Whole
            }
        }

        ScalarOp::SymReduce { .. } => {
            // SymReduce: treat like Reduce for now.
            if group.count >= num_lanes as u64 {
                GroupKind::Split
            } else {
                GroupKind::Whole
            }
        }
    }
}

fn input_is_modular_like(input: &InputRef) -> bool {
    match input {
        InputRef::Strided { dim_strides, .. } => dim_strides.len() >= 2 && dim_strides[0] == 0,
        _ => false,
    }
}

fn consumer_has_direct_ref_to_producer(
    consumer: &AtomGroup<'static, crate::pool::SystemPool>,
    producer: &AtomGroup<'static, crate::pool::SystemPool>,
) -> bool {
    consumer.inputs.iter().any(|inp| {
        input_refs_group(
            &inp.input_ref,
            consumer.count,
            consumer.atom_offset,
            producer,
        )
    })
}

fn consumer_reads_producer_modularly(
    consumer: &AtomGroup<'static, crate::pool::SystemPool>,
    producer: &AtomGroup<'static, crate::pool::SystemPool>,
) -> bool {
    consumer.inputs.iter().any(|inp| {
        input_is_modular_like(&inp.input_ref)
            && input_refs_group(
                &inp.input_ref,
                consumer.count,
                consumer.atom_offset,
                producer,
            )
    })
}

/// Promote small split groups to Duplicate when they are consumed only through
/// modular/tiled expansion by large split consumers.
///
/// This avoids creating a phase barrier just to gather a tiny producer range
/// across lanes for a huge modular consumer.
fn promote_small_modular_sources_to_duplicate(
    groups: &[AtomGroup<'static, crate::pool::SystemPool>],
    successors: &[Vec<usize>],
    kinds: &mut [GroupKind],
) {
    let dup_max = std::env::var("WT_PARTITIONER_M_MODULAR_DUP_MAX")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(4096);
    let min_expand = std::env::var("WT_PARTITIONER_M_MODULAR_DUP_MIN_EXPAND")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(16384);
    let trace_dup = std::env::var("WT_PARTITIONER_M_TRACE_DUP_PROMOTE")
        .ok()
        .is_some_and(|v| v != "0");

    let mut promoted = 0usize;
    for pi in 0..groups.len() {
        if kinds[pi] != GroupKind::Split {
            continue;
        }
        let prod = &groups[pi];
        if prod.count == 0 || prod.count > dup_max {
            continue;
        }

        let mut saw_direct = false;
        let mut saw_large_modular = false;
        let mut all_direct_are_large_modular = true;

        for &ci in &successors[pi] {
            let cons = &groups[ci];
            if !consumer_has_direct_ref_to_producer(cons, prod) {
                continue;
            }
            saw_direct = true;

            let is_large_modular = kinds[ci] == GroupKind::Split
                && cons.count >= prod.count.saturating_mul(min_expand)
                && consumer_reads_producer_modularly(cons, prod);

            if is_large_modular {
                saw_large_modular = true;
            } else {
                all_direct_are_large_modular = false;
                break;
            }
        }

        if saw_direct && saw_large_modular && all_direct_are_large_modular {
            kinds[pi] = GroupKind::Duplicate;
            promoted += 1;
            if trace_dup {
                eprintln!(
                    "  [partitioner_m] promoted g{} to Duplicate for modular fanout (count={})",
                    pi, prod.count
                );
            }
        }
    }

    if trace_dup && promoted > 0 {
        eprintln!("  [partitioner_m] promoted {promoted} groups to Duplicate");
    }
}

/// Promote tiny split sources to Duplicate when they would otherwise force a
/// split->split non-lane-local barrier into very large split consumers.
///
/// This is opt-in (disabled by default) and is intended for experimentation
/// on over-partitioned tiny producer groups.
fn promote_small_nonlocal_sources_to_duplicate(
    groups: &[AtomGroup<'static, crate::pool::SystemPool>],
    successors: &[Vec<usize>],
    kinds: &mut [GroupKind],
    num_lanes: usize,
) {
    let dup_max = std::env::var("WT_PARTITIONER_M_DUP_NONLOCAL_MAX")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);
    if dup_max == 0 {
        return;
    }

    let min_expand = std::env::var("WT_PARTITIONER_M_DUP_NONLOCAL_MIN_EXPAND")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(256);
    let min_consumer_atoms = std::env::var("WT_PARTITIONER_M_DUP_NONLOCAL_MIN_CONSUMER")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(1_000_000);
    let reduce_only = std::env::var("WT_PARTITIONER_M_DUP_NONLOCAL_REDUCE_ONLY")
        .ok()
        .map(|v| v != "0")
        .unwrap_or(true);
    let trace_dup = std::env::var("WT_PARTITIONER_M_TRACE_DUP_PROMOTE")
        .ok()
        .is_some_and(|v| v != "0");

    let mut promoted = 0usize;
    for pi in 0..groups.len() {
        if kinds[pi] != GroupKind::Split {
            continue;
        }
        let prod = &groups[pi];
        if prod.count == 0 || prod.count > dup_max {
            continue;
        }
        if reduce_only && !matches!(prod.op, ScalarOp::Reduce { .. }) {
            continue;
        }

        let mut saw_direct = false;
        let mut saw_candidate = false;
        let mut all_direct_are_candidates = true;

        for &ci in &successors[pi] {
            let cons = &groups[ci];
            if !consumer_has_direct_ref_to_producer(cons, prod) {
                continue;
            }
            saw_direct = true;

            let is_candidate = kinds[ci] == GroupKind::Split
                && cons.count >= min_consumer_atoms
                && cons.count >= prod.count.saturating_mul(min_expand)
                && lane_local_access_check(cons, prod, num_lanes, false).is_err();

            if is_candidate {
                saw_candidate = true;
            } else {
                all_direct_are_candidates = false;
                break;
            }
        }

        if saw_direct && saw_candidate && all_direct_are_candidates {
            kinds[pi] = GroupKind::Duplicate;
            promoted += 1;
            if trace_dup {
                eprintln!(
                    "  [partitioner_m] promoted g{} to Duplicate for nonlocal fanout (count={})",
                    pi, prod.count
                );
            }
        }
    }

    if trace_dup && promoted > 0 {
        eprintln!(
            "  [partitioner_m] promoted {promoted} groups to Duplicate via nonlocal heuristic"
        );
    }
}

// ─── Dependency analysis ───────────────────────────────────────────────────

/// Build group-level dependency DAG.
/// Returns (producers[gi], successors[gi]) as sorted vecs.
fn build_dependency_dag(
    graph: &NanoGraph<'static, crate::pool::SystemPool>,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let groups = graph.groups();
    let n = groups.len();
    let mut producers: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut successors: Vec<Vec<usize>> = vec![vec![]; n];

    for (gi, group) in groups.iter().enumerate() {
        let mut seen = HashSet::new();
        graph.collect_all_producer_indices(group, gi, &mut seen);
        let mut deps: Vec<usize> = seen.into_iter().collect();
        deps.sort_unstable();
        for &pi in &deps {
            successors[pi].push(gi);
        }
        producers.push(deps);
    }

    (producers, successors)
}

// ─── Phase assignment ──────────────────────────────────────────────────────

/// Determines which phase each group belongs to.
///
/// The core insight: a splittable group that reads only from (a) input tensors,
/// (b) duplicated groups in the same phase, or (c) split groups in the same
/// phase with compatible lane tiling — needs NO barrier. It stays in the
/// current phase.
///
/// A barrier (new phase) is needed when:
/// - A reduce reads from atoms that were split across lanes (needs all lanes'
///   partial results gathered first)
/// - A whole/single-lane group reads from atoms produced in the current phase
///   on a different lane
///
/// Returns phase_of[gi] for each group.
fn assign_phases(
    groups: &[AtomGroup<'static, crate::pool::SystemPool>],
    kinds: &[GroupKind],
    producers: &[Vec<usize>],
    successors: &[Vec<usize>],
    graph: &NanoGraph<'static, crate::pool::SystemPool>,
    num_lanes: usize,
) -> (Vec<usize>, Vec<bool>) {
    let n = groups.len();
    let mut phase_of = vec![0usize; n];
    // `can_be_aligned[gi]` = true if gi can be safely aligned-split to fit
    // a downstream perfect-tile Reduce consumer. The safety conditions are
    // (a) gi's only consumer is the Reduce (nothing else sees gi's layout)
    // and (b) gi has no same-phase group producers (so aligning gi's own
    // per-lane fragments never crosses a producer lane boundary). The per-
    // consumer part is checked at use time via `successors[gi].len()`; the
    // producer-side part is tracked here because it requires phase info
    // that only becomes available as we walk groups in topo order.
    let mut has_same_phase_producer = vec![false; n];
    let trace_barriers = std::env::var("WT_PARTITIONER_M_TRACE_BARRIERS")
        .ok()
        .is_some_and(|v| v != "0");
    let trace_min_atoms = std::env::var("WT_PARTITIONER_M_TRACE_MIN_ATOMS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);
    let trace_top = std::env::var("WT_PARTITIONER_M_TRACE_TOP")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(64);
    let mut barrier_events: Vec<(usize, usize, BarrierReason)> = Vec::new();

    // For each group, determine the earliest phase it can be in.
    // Groups are in topological order (NanoGraph invariant).
    for gi in 0..n {
        let mut min_phase = 0usize;

        for &pi in &producers[gi] {
            let prod_phase = phase_of[pi];
            let prod_kind = kinds[pi];
            let cons_kind = kinds[gi];
            // The perfect-tile fast path in `lane_local_access_check` is
            // only safe when the producer P can actually be aligned-split:
            // (1) P's only consumer is the Reduce that triggered the check
            // (nothing else sees P's layout), and (2) P has no same-phase
            // group producers (so aligning P's own per-lane fragments never
            // crosses a producer lane boundary upstream — see the
            // LayerNorm `(x-mean)^2 → variance` case).
            let producer_can_align = successors[pi].len() == 1 && !has_same_phase_producer[pi];

            // Determine if we need a barrier between producer and consumer.
            let barrier_reason = barrier_reason_between(
                pi,
                gi,
                prod_kind,
                cons_kind,
                groups,
                graph,
                num_lanes,
                producer_can_align,
            );

            if let Some(reason) = barrier_reason {
                min_phase = min_phase.max(prod_phase + 1);
                if trace_barriers {
                    let prod_atoms = groups[pi].count;
                    let cons_atoms = groups[gi].count;
                    if prod_atoms.max(cons_atoms) >= trace_min_atoms {
                        barrier_events.push((pi, gi, reason));
                    }
                }
            } else {
                // Same phase is fine — no cross-lane dependency.
                min_phase = min_phase.max(prod_phase);
            }
        }

        phase_of[gi] = min_phase;

        // Now that phase_of[gi] is known, record whether gi has any
        // lane-boundary-imposing group producer sharing its phase. Read by
        // the fast-path gate when gi appears as `pi` for a later consumer.
        // Duplicate producers are exempt: they broadcast the full tensor to
        // every lane, so aligning gi's per-lane split never crosses a
        // producer lane boundary against them.
        has_same_phase_producer[gi] = producers[gi]
            .iter()
            .any(|&pp| phase_of[pp] == min_phase && kinds[pp] != GroupKind::Duplicate);
    }

    if trace_barriers && !barrier_events.is_empty() {
        fn op_tag(op: &ScalarOp) -> &'static str {
            match op {
                ScalarOp::Binary { .. } => "Binary",
                ScalarOp::Unary { .. } => "Unary",
                ScalarOp::Select => "Select",
                ScalarOp::Identity => "Identity",
                ScalarOp::Cast { .. } => "Cast",
                ScalarOp::Literal(_) => "Literal",
                ScalarOp::Reduce { .. } => "Reduce",
                ScalarOp::IndirectLoad { .. } => "IndirectLoad",
                ScalarOp::OpaqueOutput { .. } => "OpaqueOutput",
                ScalarOp::LiteralSpan(_) => "LiteralSpan",
                ScalarOp::SymReduce { .. } => "SymReduce",
            }
        }

        barrier_events
            .sort_by_key(|(pi, gi, _)| std::cmp::Reverse(groups[*pi].count.max(groups[*gi].count)));
        eprintln!(
            "  [partitioner_m] barrier trace: {} edges (min_atoms={}, top={})",
            barrier_events.len(),
            trace_min_atoms,
            trace_top
        );
        for (idx, (pi, gi, reason)) in barrier_events.iter().enumerate() {
            if idx >= trace_top {
                break;
            }
            let prod = &groups[*pi];
            let cons = &groups[*gi];
            eprintln!(
                "    g{} [{} {} base={} count={} phase={}] -> g{} [{} {} base={} count={} phase={}]  reason={}",
                pi,
                op_tag(&prod.op),
                kind_tag(kinds[*pi]),
                prod.base_id.0,
                prod.count,
                phase_of[*pi],
                gi,
                op_tag(&cons.op),
                kind_tag(kinds[*gi]),
                cons.base_id.0,
                cons.count,
                phase_of[*gi],
                reason.as_str()
            );
            let access = describe_consumer_access_to_producer(cons, prod);
            if !access.is_empty() {
                eprintln!("      access: {access}");
            }
            let prod_inputs = describe_group_inputs(prod, 2);
            if !prod_inputs.is_empty() {
                eprintln!("      producer_inputs: {prod_inputs}");
            }
            let cons_inputs = describe_group_inputs(cons, 2);
            if !cons_inputs.is_empty() {
                eprintln!("      consumer_inputs: {cons_inputs}");
            }
        }
        if barrier_events.len() > trace_top {
            eprintln!(
                "    ... {} additional barrier edges omitted",
                barrier_events.len() - trace_top
            );
        }
    }

    (phase_of, has_same_phase_producer)
}

fn describe_group_inputs(
    group: &AtomGroup<'static, crate::pool::SystemPool>,
    max_inputs: usize,
) -> String {
    let mut parts: Vec<String> = Vec::new();
    for (idx, input) in group.inputs.iter().take(max_inputs).enumerate() {
        let s = match &input.input_ref {
            InputRef::Broadcast(id) => format!("in{idx}=Broadcast({})", id.0),
            InputRef::Explicit(ids) => format!("in{idx}=Explicit(len={})", ids.len()),
            InputRef::Strided {
                base,
                dim_strides,
                dim_shape,
            } => format!(
                "in{idx}=Strided(base={} strides={:?} shape={:?})",
                base.0, dim_strides, dim_shape
            ),
        };
        parts.push(s);
    }
    parts.join(", ")
}

fn describe_consumer_access_to_producer(
    consumer: &AtomGroup<'static, crate::pool::SystemPool>,
    producer: &AtomGroup<'static, crate::pool::SystemPool>,
) -> String {
    for input in &consumer.inputs {
        if !input_refs_group(
            &input.input_ref,
            consumer.count,
            consumer.atom_offset,
            producer,
        ) {
            continue;
        }

        match &input.input_ref {
            InputRef::Broadcast(id) => {
                return format!(
                    "broadcast(id={}) cons_offset={} cons_count={} prod_base={} prod_count={}",
                    id.0, consumer.atom_offset, consumer.count, producer.base_id.0, producer.count
                );
            }
            InputRef::Strided {
                base,
                dim_strides,
                dim_shape,
            } => {
                let nd = dim_strides.len();
                let first = input.input_ref.resolve(consumer.atom_offset).0;
                let last = input
                    .input_ref
                    .resolve(consumer.atom_offset + consumer.count - 1)
                    .0;
                let stride_hint = if nd == 0 {
                    0
                } else if nd == 1 {
                    dim_strides[0]
                } else {
                    dim_strides[nd - 1]
                };
                let stride_mul = stride_hint.unsigned_abs().saturating_mul(consumer.count);
                let reduce_meta = if let ScalarOp::Reduce {
                    reduce_count,
                    reduce_stride,
                    ..
                } = &consumer.op
                {
                    format!(
                        " reduce_count={} reduce_stride={}",
                        reduce_count, reduce_stride
                    )
                } else {
                    String::new()
                };
                return format!(
                    "strided(base={} nd={} strides={:?} shape={:?} cons_offset={} cons_count={} first={} last={} stride_hint={} |stride|*count={} prod_base={} prod_count={}{}",
                    base.0,
                    nd,
                    dim_strides,
                    dim_shape,
                    consumer.atom_offset,
                    consumer.count,
                    first,
                    last,
                    stride_hint,
                    stride_mul,
                    producer.base_id.0,
                    producer.count,
                    reduce_meta
                );
            }
            InputRef::Explicit(ids) => {
                return format!(
                    "explicit(len={} cons_offset={} cons_count={} prod_base={} prod_count={})",
                    ids.len(),
                    consumer.atom_offset,
                    consumer.count,
                    producer.base_id.0,
                    producer.count
                );
            }
        }
    }

    String::new()
}

fn kind_tag(kind: GroupKind) -> &'static str {
    match kind {
        GroupKind::Split => "Split",
        GroupKind::Duplicate => "Duplicate",
        GroupKind::Whole => "Whole",
    }
}

#[derive(Debug, Clone)]
enum BarrierReason {
    SplitSplitNonLocal(&'static str),
    SplitToDuplicate,
    SplitToWhole,
    WholeToSplit,
    WholeToDuplicate,
}

impl BarrierReason {
    fn as_str(&self) -> &'static str {
        match self {
            BarrierReason::SplitSplitNonLocal(msg) => msg,
            BarrierReason::SplitToDuplicate => "split->duplicate",
            BarrierReason::SplitToWhole => "split->whole",
            BarrierReason::WholeToSplit => "whole->split",
            BarrierReason::WholeToDuplicate => "whole->duplicate",
        }
    }
}

/// Determine whether a barrier is needed between producer/consumer groups.
///
/// Returns `Some(reason)` when the consumer cannot safely execute in the same
/// phase as the producer due to cross-lane dependencies.
fn barrier_reason_between(
    pi: usize,
    ci: usize,
    prod_kind: GroupKind,
    cons_kind: GroupKind,
    groups: &[AtomGroup<'static, crate::pool::SystemPool>],
    graph: &NanoGraph<'static, crate::pool::SystemPool>,
    num_lanes: usize,
    producer_has_unique_consumer: bool,
) -> Option<BarrierReason> {
    let prod = &groups[pi];
    let cons = &groups[ci];

    match (prod_kind, cons_kind) {
        // Split → Split: no barrier if the consumer reads only from its own
        // lane's slice. This is true when the input pattern is Affine with
        // stride 1 (elementwise chain) or StridedBroadcast with aligned blocks.
        (GroupKind::Split, GroupKind::Split) => {
            lane_local_access_check(cons, prod, num_lanes, producer_has_unique_consumer)
                .err()
                .map(BarrierReason::SplitSplitNonLocal)
        }

        // Duplicate → anything: no barrier. Duplicated data is in every lane.
        (GroupKind::Duplicate, _) => None,

        // Split → Duplicate consumer: the consumer is duplicated into every
        // lane, and it reads from the split producer. If the consumer needs
        // all atoms of the producer (e.g., a reduce over the full range),
        // that's a cross-lane dependency → barrier needed.
        // If the consumer only reads a broadcast (single atom), no barrier
        // if that atom is available.
        (GroupKind::Split, GroupKind::Duplicate) => Some(BarrierReason::SplitToDuplicate),

        // Split → Whole: the whole group sits on one lane but may need
        // atoms from all lanes of the split producer → barrier.
        (GroupKind::Split, GroupKind::Whole) => Some(BarrierReason::SplitToWhole),

        // Whole → Split: the whole group is on lane 0 only. Other lanes
        // can't read its output within the same phase. Barrier needed.
        (GroupKind::Whole, GroupKind::Split) => Some(BarrierReason::WholeToSplit),

        // Whole → Whole: both on lane 0 (or same lane) → no barrier if
        // same phase ordering works.
        (GroupKind::Whole, GroupKind::Whole) => None,

        // Whole → Duplicate: the duplicate runs on all lanes but Whole is only
        // on lane 0. If in the same phase, other lanes can't access the Whole
        // output. Need a barrier so the Whole group's output is in the value store.
        (GroupKind::Whole, GroupKind::Duplicate) => Some(BarrierReason::WholeToDuplicate),

        // Duplicate → anything already handled above.
        (GroupKind::Duplicate, _) => None,
    }
}

/// Check if consumer's access to producer is lane-local when both are split
/// across lanes by simple chunking.
///
/// Lane-local means: if we split producer [0..P) into chunks of P/L per lane,
/// and split consumer [0..C) into chunks of C/L per lane, then lane k's
/// consumer chunk only reads from lane k's producer chunk.
///
/// This is true for:
/// - Affine(base=prod.base_id, stride=1) when both have the same count
/// - Affine where the access pattern tiles identically
/// - Broadcast (reads single atom — but from which lane?)
/// - StridedBroadcast with repeat aligned to chunk boundaries
///
/// `producer_has_unique_consumer` enables the perfect-tile fast path: when
/// the producer feeds only this one consumer, we're free to aligned-split
/// the producer to fit the consumer's access pattern without breaking other
/// consumers. Matches the matmul Mul→Reduce pair exactly.
#[allow(dead_code)]
fn is_lane_local_access(
    consumer: &AtomGroup<'static, crate::pool::SystemPool>,
    producer: &AtomGroup<'static, crate::pool::SystemPool>,
    num_lanes: usize,
) -> bool {
    lane_local_access_check(consumer, producer, num_lanes, false).is_ok()
}

fn lane_local_access_check(
    consumer: &AtomGroup<'static, crate::pool::SystemPool>,
    producer: &AtomGroup<'static, crate::pool::SystemPool>,
    num_lanes: usize,
    producer_has_unique_consumer: bool,
) -> Result<(), &'static str> {
    let prod_base = producer.base_id.0;
    let prod_end = prod_base + producer.count;

    for input in &consumer.inputs {
        // Check if this input references the producer at all.
        let refs_producer = input_refs_group(
            &input.input_ref,
            consumer.count,
            consumer.atom_offset,
            producer,
        );
        if !refs_producer {
            continue;
        }

        match &input.input_ref {
            InputRef::Broadcast(_) => {
                // A broadcast reads a single atom. If the producer is split,
                // that atom lives on exactly one lane. Other lanes won't have it.
                // → Not lane-local (need barrier or duplication).
                return Err("split->split non-lane-local: broadcast input");
            }
            InputRef::Strided {
                base,
                dim_strides,
                dim_shape,
            } => {
                let nd = dim_strides.len();

                // StridedBroadcast: dim_strides=[stride, 0], dim_shape=[MAX, repeat]
                if nd == 2 && dim_strides[1] == 0 {
                    let stride = dim_strides[0];
                    let repeat = dim_shape[1];
                    // StridedBroadcast: atom i reads base + stride * (i / repeat).
                    // This is used in matmul Mul groups where chunks of K atoms
                    // share the same A element.
                    //
                    // For lane-local access when chunked:
                    // Lane k's atoms [k*C/L .. (k+1)*C/L) read source atoms at
                    // base + stride * (i/repeat) for i in that range.
                    //
                    // For this to be lane-local:
                    // 1. Consumer chunk size C/L must be a multiple of repeat.
                    //    Otherwise, lane boundaries don't align with repeat boundaries,
                    //    and some lanes read atoms from adjacent lanes' producer chunks.
                    // 2. The number of distinct reads per chunk (C/(L*repeat)) must equal
                    //    the producer chunk size (P/L), ensuring each lane reads exactly
                    //    its own producer fragment.
                    // 3. Base alignment: first read of lane 0 must hit producer base.
                    let chunk = consumer.count / num_lanes as u64;
                    let prod_chunk = producer.count / num_lanes as u64;
                    let abs_stride = stride.unsigned_abs();

                    // chunk must be a multiple of repeat for alignment.
                    if chunk % repeat != 0 {
                        return Err(
                            "split->split non-lane-local: strided-broadcast chunk not aligned",
                        );
                    }

                    let distinct_per_chunk = chunk / repeat;

                    if abs_stride > 0
                        && distinct_per_chunk == prod_chunk
                        && consumer.count % num_lanes as u64 == 0
                        && producer.count % num_lanes as u64 == 0
                    {
                        // Check base alignment: first consumer atom reads from producer base.
                        let first_read = base
                            .0
                            .wrapping_add((abs_stride * (consumer.atom_offset / repeat)) as u64);
                        if first_read == prod_base {
                            continue; // lane-local
                        }
                    }

                    return Err("split->split non-lane-local: strided-broadcast mismatch");
                }

                // Modular: dim_strides=[0, stride], dim_shape=[MAX, modulus]
                if nd == 2 && dim_strides[0] == 0 {
                    // Modular: atom i reads base + stride * (i % modulus).
                    // This tiles/repeats — every lane needs the same modulus-sized
                    // range. NOT lane-local unless the producer is duplicated.
                    return Err("split->split non-lane-local: modular input");
                }

                // Affine or general: use innermost stride
                let stride = if nd == 1 {
                    dim_strides[0]
                } else {
                    dim_strides[1]
                };

                // For lane-local access with chunked splitting:
                // Consumer atom i (at offset i + atom_offset) reads producer atom at
                // base + stride * (i + atom_offset).
                // After splitting consumer into chunks of C/L, lane k's atoms
                // are [k*C/L .. (k+1)*C/L). They read producer atoms at
                // base + stride * (k*C/L + offset) through base + stride * ((k+1)*C/L - 1 + offset).
                //
                // For this to land in lane k's producer chunk [prod_base + k*P/L .. prod_base + (k+1)*P/L),
                // we need |stride| * C/L == P/L, i.e., |stride| * C == P, AND base alignment.
                //
                // For reduces: the Affine gives the base atom per output element,
                // and the reduce_stride accesses the next reduce_count atoms.
                // The full footprint per element is stride (= K for matmul) atoms wide,
                // so |stride| * C = K * M = P (the full Mul group size). This satisfies
                // the lane-local condition.
                let abs_stride = stride.unsigned_abs();

                // General check: does stride * consumer_count == producer_count?
                if abs_stride > 0 && abs_stride * consumer.count == producer.count {
                    let first_read = if stride >= 0 {
                        base.0
                            .wrapping_add((abs_stride * consumer.atom_offset) as u64)
                    } else {
                        base.0
                            .wrapping_sub((abs_stride * consumer.atom_offset) as u64)
                    };

                    if first_read == prod_base {
                        // For Reduce consumers, the InputRef stride gives the
                        // distance between output bases, but each output also
                        // reads reduce_count atoms via reduce_stride.  The
                        // per-lane read span must fit within one producer chunk.
                        if let ScalarOp::Reduce {
                            reduce_count,
                            reduce_stride,
                            ..
                        } = &consumer.op
                        {
                            let reduce_extent =
                                reduce_stride.unsigned_abs() * (*reduce_count - 1) + 1;
                            // Perfect-tiling fast path: we already know
                            // abs_stride * consumer.count == producer.count.
                            // If each output's reduce footprint fits within
                            // one stride unit of the producer AND the
                            // producer has no other consumers, we can
                            // safely align the producer's per-lane split to
                            // the consumer's per-lane split regardless of
                            // whether counts divide evenly by num_lanes.
                            // This matches the post-relayout matmul Mul →
                            // Reduce pair, where the Mul is only consumed by
                            // its paired Reduce. If the producer has other
                            // consumers (e.g. a LayerNorm source feeding
                            // mean, variance, and (x-mean) simultaneously),
                            // aligning the producer for the Reduce would
                            // break the others — fall back to the tight
                            // per-chunk check against the default split.
                            let safe_aligned =
                                producer_has_unique_consumer && reduce_extent <= abs_stride;
                            if !safe_aligned {
                                let max_chunk =
                                    (consumer.count + num_lanes as u64 - 1) / num_lanes as u64;
                                let prod_chunk = producer.count / num_lanes as u64;
                                let span = abs_stride * (max_chunk - 1) + reduce_extent;
                                if span > prod_chunk {
                                    return Err(
                                        "split->split non-lane-local: reduce footprint exceeds producer chunk",
                                    );
                                }
                            }
                        }
                        // Lane-local with aligned split.
                        continue;
                    }
                }

                // Special case: stride=1, same count (elementwise 1:1).
                if stride == 1 && consumer.count == producer.count {
                    let first_read = base.0.wrapping_add(consumer.atom_offset);
                    let last_read = base
                        .0
                        .wrapping_add(consumer.atom_offset + consumer.count - 1);
                    if first_read == prod_base && last_read == prod_end - 1 {
                        continue; // lane-local
                    }
                    if first_read >= prod_base && last_read < prod_end {
                        continue; // lane-local
                    }
                }

                // Not lane-local.
                return Err("split->split non-lane-local: affine/general stride mismatch");
            }
            InputRef::Explicit(_) => {
                // Arbitrary mapping — not lane-local in general.
                return Err("split->split non-lane-local: explicit mapping");
            }
        }
    }

    Ok(())
}

/// Check if any of the consumer's inputs broadcast a single atom from the producer.
fn is_broadcast_access(
    consumer: &AtomGroup<'static, crate::pool::SystemPool>,
    producer: &AtomGroup<'static, crate::pool::SystemPool>,
) -> Option<AtomId> {
    for input in &consumer.inputs {
        if let InputRef::Broadcast(id) = &input.input_ref {
            if producer.contains(*id) {
                return Some(*id);
            }
        }
    }
    None
}

/// Check if an InputRef references any atom in the given producer group.
fn input_refs_group(
    input: &InputRef,
    consumer_count: u64,
    consumer_offset: u64,
    producer: &AtomGroup<'static, crate::pool::SystemPool>,
) -> bool {
    if consumer_count == 0 {
        return false;
    }
    let pb = producer.base_id.0;
    let pe = pb + producer.count;
    input_access_segments(input, consumer_offset, consumer_count)
        .into_iter()
        .any(|(lo, hi)| lo < pe && hi > pb)
}

// ─── Span building ─────────────────────────────────────────────────────────

/// For a split group, compute the atom range for a given lane.
fn split_range(
    group: &AtomGroup<'static, crate::pool::SystemPool>,
    lane: usize,
    num_lanes: usize,
) -> (u64, u64) {
    let cl_atoms = cache_line_atoms(group.output_dtype);
    split_count_cl(group.count, lane, num_lanes, cl_atoms)
}

fn normalize_num_lanes(requested: usize) -> usize {
    requested.max(1)
}

/// Atoms per cache line for a given dtype.
fn cache_line_atoms(dtype: crate::numeric_dtype::NumericDType) -> u64 {
    const CACHE_LINE_BYTES: u64 = 64;
    let bpe = dtype.bytes_per_element() as u64;
    if bpe == 0 {
        return 1;
    }
    CACHE_LINE_BYTES / bpe
}

/// Split `count` items evenly across lanes (no alignment).
fn split_count(count: u64, lane: usize, num_lanes: usize) -> (u64, u64) {
    let chunk = count / num_lanes as u64;
    let remainder = count % num_lanes as u64;
    // Distribute remainder: first `remainder` lanes get one extra item.
    let start = chunk * lane as u64 + (lane as u64).min(remainder);
    let lane_count = chunk + if (lane as u64) < remainder { 1 } else { 0 };
    (start, lane_count)
}

/// Split `count` items across lanes with cache-line-aligned boundaries.
///
/// Every inter-lane boundary falls on a `cl_atoms` multiple so that
/// adjacent lanes never share a cache line. When `count` is too small
/// to give every lane a full cache-line-sized chunk, the effective lane
/// count is reduced — surplus lanes receive `(start_past_end, 0)`.
fn split_count_cl(count: u64, lane: usize, num_lanes: usize, cl_atoms: u64) -> (u64, u64) {
    if cl_atoms <= 1 || num_lanes <= 1 {
        return split_count(count, lane, num_lanes);
    }

    let n_blocks = count / cl_atoms;
    let tail = count - n_blocks * cl_atoms;

    // Effective lanes: no more than available blocks.
    let eff = (n_blocks as usize).min(num_lanes).max(1);
    if eff <= 1 {
        // Whole-lane fallback: lane 0 gets everything.
        return if lane == 0 { (0, count) } else { (count, 0) };
    }

    if lane >= eff {
        // This lane is inactive.
        return (count, 0);
    }

    // Distribute cache-line blocks among effective lanes.
    let blocks_per = n_blocks / eff as u64;
    let block_rem = n_blocks % eff as u64;
    let my_blocks = blocks_per + if (lane as u64) < block_rem { 1 } else { 0 };
    let start_block = blocks_per * lane as u64 + (lane as u64).min(block_rem);
    let start = start_block * cl_atoms;
    let mut my_count = my_blocks * cl_atoms;

    // Last active lane absorbs the sub-cache-line tail.
    if lane == eff - 1 {
        my_count += tail;
    }

    (start, my_count)
}

/// Split a producer group aligned to a consumer's boundaries.
///
/// When a Mul group feeds a Reduce with stride K, the producer must be split
/// as K × consumer_chunk per lane (not even splits of the producer count).
/// This ensures each lane's Reduce fragment reads exactly from its lane's
/// Mul fragment, even when the consumer count doesn't divide evenly by num_lanes.
fn split_range_aligned(
    group: &AtomGroup<'static, crate::pool::SystemPool>,
    lane: usize,
    num_lanes: usize,
    consumer_count: u64,
    stride: u64,
) -> (u64, u64) {
    // Split the consumer with cache-line alignment, then scale by
    // stride for the producer. The consumer dtype matches the
    // producer's output_dtype (the reduce reads from the producer).
    let cl_atoms = cache_line_atoms(group.output_dtype);
    let (cons_start, cons_lane_count) = split_count_cl(consumer_count, lane, num_lanes, cl_atoms);
    let start = cons_start * stride;
    let count = cons_lane_count * stride;
    debug_assert!(
        start + count <= group.count,
        "aligned split overflow: start={} count={} group.count={} consumer_count={} stride={}",
        start,
        count,
        group.count,
        consumer_count,
        stride
    );
    (start, count)
}

/// Collect all atom ranges that a group's inputs reference, including
/// reduce stride ranges and indirect load tables.
fn collect_input_atom_ranges(
    group: &AtomGroup<'static, crate::pool::SystemPool>,
    graph: &NanoGraph<'static, crate::pool::SystemPool>,
) -> Vec<(AtomId, u64, NumericDType)> {
    let mut ranges = Vec::new();
    let mut seen_bases = HashSet::new();

    // Collect from InputRefs
    let mut prod_indices = HashSet::new();
    graph.collect_all_producer_indices(group, usize::MAX, &mut prod_indices);

    // For each producer, we need its full range.
    let all_groups = graph.groups();
    for &pi in &prod_indices {
        if pi < all_groups.len() {
            let pg = &all_groups[pi];
            if seen_bases.insert(pg.base_id.0) {
                ranges.push((pg.base_id, pg.count, pg.output_dtype));
            }
        }
    }

    // Also check input tensors
    for input in &group.inputs {
        match &input.input_ref {
            InputRef::Broadcast(id) | InputRef::Strided { base: id, .. } => {
                for (idx, _) in graph.find_input_idxs(*id) {
                    let it = &graph.input_tensors()[idx];
                    if seen_bases.insert(it.base_id.0) {
                        ranges.push((it.base_id, it.count, it.dtype));
                    }
                }
            }
            InputRef::Explicit(ids) => {
                for id in ids {
                    for (idx, _) in graph.find_input_idxs(*id) {
                        let it = &graph.input_tensors()[idx];
                        if seen_bases.insert(it.base_id.0) {
                            ranges.push((it.base_id, it.count, it.dtype));
                        }
                    }
                }
            }
        }
    }

    // IndirectLoad table — may be a group (inlined LiteralSpan) or an
    // InputTensor (large constant that wasn't inlined during lowering).
    if let ScalarOp::IndirectLoad { table_base, .. } = &group.op {
        if let Some(tg) = graph.group_of(*table_base) {
            if seen_bases.insert(tg.base_id.0) {
                ranges.push((tg.base_id, tg.count, tg.output_dtype));
            }
        } else {
            for (ti, _) in graph.find_input_idxs(*table_base) {
                let it = &graph.input_tensors()[ti];
                if seen_bases.insert(it.base_id.0) {
                    ranges.push((it.base_id, it.count, it.dtype));
                }
            }
        }
    }

    ranges
}

// ─── Public API ────────────────────────────────────────────────────────────

/// Partition a NanoGraph into phases and spans for parallel execution.
///
/// Returns a sequence of phases, each containing one span per lane.
pub fn plan(
    graph: &NanoGraph<'static, crate::pool::SystemPool>,
    num_lanes: usize,
    input_tensors: &[InputTensor],
    output_atom_ids: &[AtomRange],
) -> Vec<Phase> {
    let requested_lanes = num_lanes.max(1);
    let num_lanes = normalize_num_lanes(requested_lanes);
    if num_lanes != requested_lanes {
        eprintln!(
            "  [partitioner_m] requested {} lanes is not supported; using {} lanes",
            requested_lanes, num_lanes
        );
    }

    if graph.groups().is_empty() {
        return vec![Phase {
            spans: (0..num_lanes)
                .map(|_| Span {
                    graph: NanoGraph::new(),
                    inputs: vec![],
                    outputs: vec![],
                })
                .collect(),
        }];
    }

    // Step 0: Relayout matmul Mul→Reduce pairs from [K,N] to [N,K] order.
    //
    // The lowering produces Mul groups in [K,N] layout where each output atom's
    // reduction footprint spans the entire group (stride=1, reduce_stride=N).
    // This prevents lane-local splitting because every consumer lane needs all
    // producer atoms.
    //
    // Transposing to [N,K] layout makes the reduce access contiguous (stride=K,
    // reduce_stride=1), so each lane's N/L output atoms access a contiguous
    // K*(N/L) chunk of the producer — exactly lane-local.
    let mut graph = graph.clone();
    let relayouted = relayout_matmul_groups(&mut graph);
    if relayouted > 0 {
        eprintln!(
            "  Relayouted {} matmul Mul→Reduce pairs from [K,N] to [N,K]",
            relayouted
        );
    }
    let graph = &graph;
    let groups = graph.groups();
    let n = groups.len();

    // Step 1: Classify groups.
    let mut kinds: Vec<GroupKind> = groups
        .iter()
        .map(|g| classify_group(g, num_lanes))
        .collect();

    // Step 2: Build dependency DAG.
    let (producers, successors) = build_dependency_dag(graph);

    // Step 2b: Duplicate tiny modular sources so large modular expansion
    // consumers don't force a cross-lane barrier.
    promote_small_modular_sources_to_duplicate(groups, &successors, &mut kinds);
    // Step 2c: Optional heuristic for tiny non-lane-local sources feeding
    // very large split consumers.
    promote_small_nonlocal_sources_to_duplicate(groups, &successors, &mut kinds, num_lanes);

    // Step 2d: Cache-line discipline — demote remaining Split groups
    // whose per-lane fragment would be smaller than a cache line to
    // Whole. This runs AFTER promotion passes so that groups eligible
    // for Duplicate have already been promoted; only groups that
    // stayed Split are considered. The demotion to Whole causes
    // assign_phases to insert phase barriers for downstream Split
    // consumers that can no longer read lane-locally.
    for gi in 0..groups.len() {
        if kinds[gi] != GroupKind::Split {
            continue;
        }
        let cl_atoms = cache_line_atoms(groups[gi].output_dtype);
        if cl_atoms > 1 && groups[gi].count < cl_atoms * num_lanes as u64 {
            kinds[gi] = GroupKind::Whole;
        }
    }

    // Step 3: Assign phases.
    let (phase_of, has_same_phase_producer) =
        assign_phases(groups, &kinds, &producers, &successors, graph, num_lanes);

    // Step 4: Collect groups by phase.
    let num_phases = phase_of.iter().copied().max().unwrap_or(0) + 1;
    let mut phase_groups: Vec<Vec<usize>> = vec![vec![]; num_phases];
    for (gi, &ph) in phase_of.iter().enumerate() {
        phase_groups[ph].push(gi);
    }

    // Step 5: Collect output group indices for determining span outputs.
    // Each output range may span multiple groups (e.g., pad output =
    // literal + identity + literal), so we walk through all groups
    // overlapping each range, not just the group at range.base.
    let output_group_set: HashSet<usize> = {
        let groups = graph.groups();
        let mut set = HashSet::new();
        for range in output_atom_ids {
            let range_lo = range.base.0;
            let range_hi = range_lo + range.count;
            // Walk groups that overlap [range_lo, range_hi).
            for (gi, group) in groups.iter().enumerate() {
                let g_lo = group.base_id.0;
                let g_hi = g_lo + group.count;
                if g_lo < range_hi && g_hi > range_lo {
                    set.insert(gi);
                }
            }
        }
        set
    };

    // Track which groups are consumed by groups in later phases.
    let mut cross_phase_consumed: HashSet<usize> = HashSet::new();
    for gi in 0..n {
        for &pi in &producers[gi] {
            if phase_of[pi] < phase_of[gi] {
                cross_phase_consumed.insert(pi);
            }
        }
    }

    // Step 6: Build phases.
    let mut input_tensor_set: HashMap<u64, &InputTensor> = HashMap::new();
    for it in input_tensors {
        input_tensor_set.insert(it.base_id.0, it);
    }

    // Also index graph's own input tensors
    for it in graph.input_tensors() {
        input_tensor_set.insert(it.base_id.0, it);
    }

    let mut phases = Vec::with_capacity(num_phases);

    for phase_idx in 0..num_phases {
        let phase = build_phase(
            graph,
            &phase_groups[phase_idx],
            &kinds,
            &phase_of,
            &producers,
            &successors,
            &has_same_phase_producer,
            &cross_phase_consumed,
            &output_group_set,
            &input_tensor_set,
            num_lanes,
            phase_idx,
            num_phases,
        );
        phases.push(phase);
    }

    phases
}

/// Build a single phase: create one span per lane with the appropriate
/// group fragments.
fn build_phase(
    graph: &NanoGraph<'static, crate::pool::SystemPool>,
    phase_group_indices: &[usize],
    kinds: &[GroupKind],
    phase_of: &[usize],
    producers: &[Vec<usize>],
    successors: &[Vec<usize>],
    has_same_phase_producer: &[bool],
    cross_phase_consumed: &HashSet<usize>,
    output_group_set: &HashSet<usize>,
    input_tensor_set: &HashMap<u64, &InputTensor>,
    num_lanes: usize,
    phase_idx: usize,
    num_phases: usize,
) -> Phase {
    let all_groups = graph.groups();

    // Initialize span builders — one per lane.
    let mut span_graphs: Vec<NanoGraph> = (0..num_lanes).map(|_| NanoGraph::new()).collect();
    let mut span_inputs: Vec<Vec<AtomRange>> = (0..num_lanes).map(|_| Vec::new()).collect();
    let mut span_outputs: Vec<Vec<AtomRange>> = (0..num_lanes).map(|_| Vec::new()).collect();

    // Copy metadata into each span graph.
    for sg in &mut span_graphs {
        sg.graph_constants = graph.graph_constants.clone();
        sg.set_opaque_ops(graph.opaque_ops().to_vec());
    }

    // Track which atom ranges have been declared as inputs in each span.
    let mut declared_inputs: Vec<HashSet<DeclaredInputKey>> =
        (0..num_lanes).map(|_| HashSet::new()).collect();
    // Track which atom ranges are produced in each span (for intra-phase deps).
    let mut produced_in_span: Vec<HashSet<u64>> = (0..num_lanes).map(|_| HashSet::new()).collect();

    // Determine which groups in this phase need their output to be in span outputs.
    let groups_needing_output: HashSet<usize> = phase_group_indices
        .iter()
        .filter(|&&gi| output_group_set.contains(&gi) || cross_phase_consumed.contains(&gi))
        .copied()
        .collect();

    // Pre-compute aligned splits: for each Split group in this phase that
    // feeds a Reduce consumer in the same phase via Affine with stride K,
    // record (consumer_count, K) so emit_split_group uses aligned splitting.
    // This ensures the Mul→Reduce pair stays lane-local even when the Reduce
    // count doesn't divide evenly by num_lanes — `split_range_aligned` uses
    // `split_count` which handles the remainder by giving the first few lanes
    // one extra atom, and the stride multiplier propagates that to the
    // producer's chunk sizes so the partition stays exact.
    //
    // For the non-divisible case, only lift the alignment when the producer
    // has a single consumer. An aligned per-lane split of the producer fits
    // exactly one consumer's access pattern; if other consumers existed they
    // would see the producer's atoms at the wrong lane boundaries. This
    // matches the relayouted matmul Mul→Reduce pair, where the Mul has only
    // the Reduce as a consumer (structural).
    let mut aligned_splits: HashMap<usize, (u64, u64)> = HashMap::new(); // gi → (consumer_count, stride)
    let phase_set: HashSet<usize> = phase_group_indices.iter().copied().collect();
    for &gi in phase_group_indices {
        if kinds[gi] != GroupKind::Split {
            continue;
        }
        let group = &all_groups[gi];
        // Same gate as the `lane_local_access_check` fast path: we can only
        // non-divisibly align a group whose layout is not observed by any
        // other consumer AND whose own upstream reads do not depend on a
        // same-phase lane layout. See the LayerNorm `(x-mean)^2 → variance`
        // discussion in `assign_phases`.
        let producer_can_align = successors[gi].len() == 1 && !has_same_phase_producer[gi];
        for &ci in &successors[gi] {
            if !phase_set.contains(&ci) || kinds[ci] != GroupKind::Split {
                continue;
            }
            let cons = &all_groups[ci];
            if let ScalarOp::Reduce { .. } = &cons.op {
                for inp in &cons.inputs {
                    if let InputRef::Strided {
                        base, dim_strides, ..
                    } = &inp.input_ref
                    {
                        // Use innermost stride for the affine-like pattern
                        let stride = dim_strides.last().copied().unwrap_or(0);
                        let abs_stride = stride.unsigned_abs();
                        let divisible = cons.count % num_lanes as u64 == 0
                            && group.count % num_lanes as u64 == 0;
                        if abs_stride > 0
                            && abs_stride * cons.count == group.count
                            && (divisible || producer_can_align)
                            && input_refs_group(&inp.input_ref, cons.count, cons.atom_offset, group)
                        {
                            aligned_splits.insert(gi, (cons.count, abs_stride));
                        }
                    }
                }
            }
        }
    }

    // Process groups in topological order (they're already sorted by index
    // which is topological order in NanoGraph).
    let mut sorted_indices = phase_group_indices.to_vec();
    sorted_indices.sort_unstable();

    for &gi in &sorted_indices {
        let group = &all_groups[gi];
        let kind = kinds[gi];

        match kind {
            GroupKind::Split => {
                emit_split_group(
                    graph,
                    group,
                    gi,
                    num_lanes,
                    aligned_splits.get(&gi).copied(),
                    &mut span_graphs,
                    &mut span_inputs,
                    &mut span_outputs,
                    &mut declared_inputs,
                    &mut produced_in_span,
                    phase_of,
                    kinds,
                    all_groups,
                    groups_needing_output.contains(&gi),
                    input_tensor_set,
                );
            }
            GroupKind::Duplicate => {
                emit_duplicate_group(
                    graph,
                    group,
                    gi,
                    num_lanes,
                    &mut span_graphs,
                    &mut span_inputs,
                    &mut span_outputs,
                    &mut declared_inputs,
                    &mut produced_in_span,
                    phase_of,
                    kinds,
                    all_groups,
                    groups_needing_output.contains(&gi),
                    input_tensor_set,
                    phase_idx,
                );
            }
            GroupKind::Whole => {
                emit_whole_group(
                    graph,
                    group,
                    gi,
                    num_lanes,
                    &mut span_graphs,
                    &mut span_inputs,
                    &mut span_outputs,
                    &mut declared_inputs,
                    &mut produced_in_span,
                    phase_of,
                    kinds,
                    all_groups,
                    groups_needing_output.contains(&gi),
                    input_tensor_set,
                );
            }
        }
    }

    Phase {
        spans: span_graphs
            .into_iter()
            .zip(span_inputs)
            .zip(span_outputs)
            .map(|((g, inp), out)| Span {
                graph: g,
                inputs: inp,
                outputs: out,
            })
            .collect(),
    }
}

/// Emit a split group: fragment into N lanes, each getting count/N atoms.
///
/// `aligned_split`: if Some((consumer_count, stride)), split as
/// stride × consumer_chunks instead of even splits. Used for Mul groups
/// paired with a Reduce consumer to maintain lane alignment.
fn emit_split_group(
    graph: &NanoGraph<'static, crate::pool::SystemPool>,
    group: &AtomGroup<'static, crate::pool::SystemPool>,
    gi: usize,
    num_lanes: usize,
    aligned_split: Option<(u64, u64)>,
    span_graphs: &mut [NanoGraph<'static, crate::pool::SystemPool>],
    span_inputs: &mut [Vec<AtomRange>],
    span_outputs: &mut [Vec<AtomRange>],
    declared_inputs: &mut [HashSet<DeclaredInputKey>],
    produced_in_span: &mut [HashSet<u64>],
    phase_of: &[usize],
    kinds: &[GroupKind],
    all_groups: &[AtomGroup<'static, crate::pool::SystemPool>],
    needs_output: bool,
    input_tensor_set: &HashMap<u64, &InputTensor>,
) {
    let base = group.base_id.0;

    for lane in 0..num_lanes {
        let (start, count) = if let Some((cons_count, stride)) = aligned_split {
            split_range_aligned(group, lane, num_lanes, cons_count, stride)
        } else {
            split_range(group, lane, num_lanes)
        };
        if count == 0 {
            continue;
        }

        let frag_base = AtomId(base + start);
        let atom_offset = group.atom_offset + start;

        // Clone inputs — atom_offset handles correct resolution for all
        // non-Explicit InputRef types. Groups with Explicit inputs are
        // classified as Whole, so we should never see them here.
        let inputs: Vec<crate::nano_graph::pattern::GroupInput> = group
            .inputs
            .iter()
            .map(|inp| clone_input_for_split(inp))
            .collect();

        // Ensure all dependencies are declared as inputs to this span.
        ensure_inputs_declared(
            graph,
            &inputs,
            &group.op,
            atom_offset,
            count,
            lane,
            span_graphs,
            span_inputs,
            declared_inputs,
            produced_in_span,
            input_tensor_set,
        );

        span_graphs[lane].insert_group_at(
            frag_base,
            count,
            atom_offset,
            group.output_dtype,
            group.op.clone(),
            group.sym_dims.clone(),
            inputs,
        );

        produced_in_span[lane].insert(frag_base.0);

        if needs_output {
            span_outputs[lane].push(AtomRange {
                base: frag_base,
                count,
                dtype: group.output_dtype,
            });
        }
    }
}

/// Clone an InputRef for a split fragment.
///
/// Per the execution model: "The inputs vector is the SAME for all fragments.
/// The atom_offset parameter tells the eval/codegen to resolve
/// input.resolve(i + atom_offset) instead of input.resolve(i)."
///
/// For Affine, Broadcast, StridedBroadcast, Modular: the input stays unchanged
/// and atom_offset handles correct resolution.
///
/// Explicit InputRefs should never reach here — groups with Explicit inputs
/// are classified as Whole (not Split) to avoid the vector-slicing issue.
fn clone_input_for_split(
    input: &crate::nano_graph::pattern::GroupInput,
) -> crate::nano_graph::pattern::GroupInput {
    debug_assert!(
        !matches!(input.input_ref, InputRef::Explicit(_)),
        "Explicit InputRef should not appear in a split group"
    );
    input.clone()
}

/// Emit a duplicate group: full copy in every lane that needs it.
/// For simplicity (and correctness), duplicate into all lanes.
fn emit_duplicate_group(
    graph: &NanoGraph<'static, crate::pool::SystemPool>,
    group: &AtomGroup<'static, crate::pool::SystemPool>,
    gi: usize,
    num_lanes: usize,
    span_graphs: &mut [NanoGraph<'static, crate::pool::SystemPool>],
    span_inputs: &mut [Vec<AtomRange>],
    span_outputs: &mut [Vec<AtomRange>],
    declared_inputs: &mut [HashSet<DeclaredInputKey>],
    produced_in_span: &mut [HashSet<u64>],
    phase_of: &[usize],
    kinds: &[GroupKind],
    all_groups: &[AtomGroup<'static, crate::pool::SystemPool>],
    needs_output: bool,
    input_tensor_set: &HashMap<u64, &InputTensor>,
    phase_idx: usize,
) {
    for lane in 0..num_lanes {
        // Ensure all dependencies are declared.
        ensure_inputs_declared(
            graph,
            &group.inputs,
            &group.op,
            group.atom_offset,
            group.count,
            lane,
            span_graphs,
            span_inputs,
            declared_inputs,
            produced_in_span,
            input_tensor_set,
        );

        span_graphs[lane].insert_group_at(
            group.base_id,
            group.count,
            group.atom_offset,
            group.output_dtype,
            group.op.clone(),
            group.sym_dims.clone(),
            group.inputs.clone(),
        );

        produced_in_span[lane].insert(group.base_id.0);
    }

    // Duplicated groups: output from lane 0 (all lanes produce identical data,
    // but we only need to export once).
    if needs_output {
        span_outputs[0].push(AtomRange {
            base: group.base_id,
            count: group.count,
            dtype: group.output_dtype,
        });
    }
}

/// Emit a whole (unsplittable) group: put on lane 0.
fn emit_whole_group(
    graph: &NanoGraph<'static, crate::pool::SystemPool>,
    group: &AtomGroup<'static, crate::pool::SystemPool>,
    gi: usize,
    num_lanes: usize,
    span_graphs: &mut [NanoGraph<'static, crate::pool::SystemPool>],
    span_inputs: &mut [Vec<AtomRange>],
    span_outputs: &mut [Vec<AtomRange>],
    declared_inputs: &mut [HashSet<DeclaredInputKey>],
    produced_in_span: &mut [HashSet<u64>],
    phase_of: &[usize],
    kinds: &[GroupKind],
    all_groups: &[AtomGroup<'static, crate::pool::SystemPool>],
    needs_output: bool,
    input_tensor_set: &HashMap<u64, &InputTensor>,
) {
    let lane = 0;

    ensure_inputs_declared(
        graph,
        &group.inputs,
        &group.op,
        group.atom_offset,
        group.count,
        lane,
        span_graphs,
        span_inputs,
        declared_inputs,
        produced_in_span,
        input_tensor_set,
    );

    span_graphs[lane].insert_group_at(
        group.base_id,
        group.count,
        group.atom_offset,
        group.output_dtype,
        group.op.clone(),
        group.sym_dims.clone(),
        group.inputs.clone(),
    );

    produced_in_span[lane].insert(group.base_id.0);

    if needs_output {
        span_outputs[lane].push(AtomRange {
            base: group.base_id,
            count: group.count,
            dtype: group.output_dtype,
        });
    }
}

fn normalize_half_open_segments(mut segments: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    segments.retain(|(lo, hi)| lo < hi);
    segments.sort_unstable_by_key(|&(lo, hi)| (lo, hi));
    if segments.is_empty() {
        return segments;
    }
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(segments.len());
    for (lo, hi) in segments {
        if let Some((last_lo, last_hi)) = merged.last_mut()
            && lo <= *last_hi
        {
            *last_hi = (*last_hi).max(hi);
            continue;
        }
        merged.push((lo, hi));
    }
    merged
}

fn explicit_access_segments(ids: &[AtomId], atom_offset: u64, count: u64) -> Vec<(u64, u64)> {
    if count == 0 || ids.is_empty() {
        return Vec::new();
    }
    let ids_len = ids.len() as u64;
    let start = atom_offset.min(ids_len) as usize;
    let end = atom_offset.saturating_add(count).min(ids_len) as usize;
    if start >= end {
        return Vec::new();
    }

    let mut points: Vec<u64> = ids[start..end].iter().map(|id| id.0).collect();
    points.sort_unstable();
    points.dedup();

    let mut out = Vec::new();
    let mut run_lo = points[0];
    let mut prev = points[0];
    for &p in points.iter().skip(1) {
        if p == prev.saturating_add(1) {
            prev = p;
        } else {
            out.push((run_lo, prev.saturating_add(1)));
            run_lo = p;
            prev = p;
        }
    }
    out.push((run_lo, prev.saturating_add(1)));
    out
}

fn clamp_i128_to_u64(x: i128) -> u64 {
    if x <= 0 {
        0
    } else if x >= u64::MAX as i128 {
        u64::MAX
    } else {
        x as u64
    }
}

fn strided_linear_segment(base: u64, stride: i64, idx_lo: u64, idx_hi: u64) -> Option<(u64, u64)> {
    if idx_lo > idx_hi {
        return None;
    }
    let a = (base as i128).saturating_add((stride as i128).saturating_mul(idx_lo as i128));
    let b = (base as i128).saturating_add((stride as i128).saturating_mul(idx_hi as i128));
    let lo = a.min(b);
    let hi = a.max(b).saturating_add(1);
    let lo_u = clamp_i128_to_u64(lo);
    let hi_u = clamp_i128_to_u64(hi);
    if lo_u < hi_u {
        Some((lo_u, hi_u))
    } else {
        None
    }
}

fn generic_nd_strided_hull(
    base: u64,
    dim_strides: &[i64],
    dim_shape: &[u64],
    atom_offset: u64,
    count: u64,
) -> Option<(u64, u64)> {
    if count == 0 || dim_strides.is_empty() || dim_strides.len() != dim_shape.len() {
        return None;
    }
    if dim_shape.iter().skip(1).any(|&d| d == 0) {
        return None;
    }

    let mut inner_volume: u128 = 1;
    for &d in dim_shape.iter().skip(1) {
        inner_volume = inner_volume.saturating_mul(d as u128);
    }
    if inner_volume == 0 {
        return None;
    }

    let start = atom_offset as u128;
    let end = start.saturating_add(count as u128).saturating_sub(1);
    let outer_lo = start / inner_volume;
    let outer_hi = end / inner_volume;

    let mut min_off: i128 = 0;
    let mut max_off: i128 = 0;
    for (d, &stride) in dim_strides.iter().enumerate() {
        let (coord_lo, coord_hi): (u128, u128) = if d == 0 {
            (outer_lo, outer_hi)
        } else {
            (0, dim_shape[d].saturating_sub(1) as u128)
        };
        let a = (stride as i128).saturating_mul(coord_lo as i128);
        let b = (stride as i128).saturating_mul(coord_hi as i128);
        min_off = min_off.saturating_add(a.min(b));
        max_off = max_off.saturating_add(a.max(b));
    }

    let lo = (base as i128).saturating_add(min_off);
    let hi = (base as i128).saturating_add(max_off).saturating_add(1);
    let lo_u = clamp_i128_to_u64(lo);
    let hi_u = clamp_i128_to_u64(hi);
    if lo_u < hi_u {
        Some((lo_u, hi_u))
    } else {
        None
    }
}

pub(crate) fn input_access_segments(
    input: &InputRef,
    atom_offset: u64,
    count: u64,
) -> Vec<(u64, u64)> {
    if count == 0 {
        return Vec::new();
    }
    match input {
        InputRef::Broadcast(id) => vec![(id.0, id.0.saturating_add(1))],
        InputRef::Explicit(ids) => explicit_access_segments(ids, atom_offset, count),
        InputRef::Strided {
            base,
            dim_strides,
            dim_shape,
        } => {
            let nd = dim_strides.len();
            if nd == 0 {
                return Vec::new();
            }

            // 1D affine.
            if nd == 1 && dim_shape.len() == 1 && dim_shape[0] == u64::MAX {
                if let Some(seg) = strided_linear_segment(
                    base.0,
                    dim_strides[0],
                    atom_offset,
                    atom_offset + count - 1,
                ) {
                    return vec![seg];
                } else {
                    return Vec::new();
                }
            }

            // StridedBroadcast: base + stride * (i / repeat).
            if nd == 2 && dim_shape.len() == 2 && dim_strides[1] == 0 {
                let repeat = dim_shape[1].max(1);
                let first_block = atom_offset / repeat;
                let last_block = (atom_offset + count - 1) / repeat;
                if let Some(seg) =
                    strided_linear_segment(base.0, dim_strides[0], first_block, last_block)
                {
                    return vec![seg];
                } else {
                    return Vec::new();
                }
            }

            // Modular: base + stride * (i % modulus).
            // The residue interval may wrap; emit one or two conservative
            // segments over the covered residue window.
            if nd == 2 && dim_shape.len() == 2 && dim_strides[0] == 0 {
                let modulus = dim_shape[1].max(1);
                let covered = count.min(modulus);
                if covered == 0 {
                    return Vec::new();
                }
                let start = atom_offset % modulus;
                let end_excl = start + covered;
                let mut segs = Vec::with_capacity(2);
                if end_excl <= modulus {
                    if let Some(seg) =
                        strided_linear_segment(base.0, dim_strides[1], start, end_excl - 1)
                    {
                        segs.push(seg);
                    }
                } else {
                    if let Some(seg) =
                        strided_linear_segment(base.0, dim_strides[1], start, modulus - 1)
                    {
                        segs.push(seg);
                    }
                    let wrap_end = end_excl % modulus;
                    if wrap_end > 0
                        && let Some(seg) =
                            strided_linear_segment(base.0, dim_strides[1], 0, wrap_end - 1)
                    {
                        segs.push(seg);
                    }
                }
                return normalize_half_open_segments(segs);
            }

            // Generic ND conservative hull: bound outer index range for this
            // lane fragment and allow full inner-dimension variation.
            if let Some(seg) =
                generic_nd_strided_hull(base.0, dim_strides, dim_shape, atom_offset, count)
            {
                vec![seg]
            } else {
                Vec::new()
            }
        }
    }
}

fn expand_segments_for_reduce(
    segments: &[(u64, u64)],
    reduce_count: u64,
    reduce_stride: i64,
) -> Vec<(u64, u64)> {
    if reduce_count <= 1 || reduce_stride == 0 {
        return segments.to_vec();
    }
    let end_off = (reduce_count as i128 - 1) * reduce_stride as i128;
    let mut out = Vec::with_capacity(segments.len());
    for &(lo, hi) in segments {
        if lo >= hi {
            continue;
        }
        let last = hi - 1;
        let a = lo as i128;
        let b = last as i128;
        let lo_i = a.min(b).min(a + end_off).min(b + end_off).max(0);
        let hi_i = a.max(b).max(a + end_off).max(b + end_off).max(0);
        out.push((lo_i as u64, (hi_i as u64).saturating_add(1)));
    }
    normalize_half_open_segments(out)
}

/// Ensure that all atoms referenced by a group's inputs (and reduce strides,
/// indirect load tables) are either produced in this span or declared as
/// span inputs.
///
/// Uses the main graph's producer analysis to deterministically find all
/// producer groups, avoiding sampling-based approaches that can miss producers.
fn ensure_inputs_declared(
    graph: &NanoGraph<'static, crate::pool::SystemPool>,
    inputs: &[crate::nano_graph::pattern::GroupInput],
    op: &ScalarOp,
    atom_offset: u64,
    count: u64,
    lane: usize,
    span_graphs: &mut [NanoGraph<'static, crate::pool::SystemPool>],
    span_inputs: &mut [Vec<AtomRange>],
    declared_inputs: &mut [HashSet<DeclaredInputKey>],
    produced_in_span: &[HashSet<u64>],
    input_tensor_set: &HashMap<u64, &InputTensor>,
) {
    // Helper: declare a source range intersection as a span input if needed.
    // `access_lo..access_hi` is the actually accessed half-open segment.
    let mut declare_range = |base: AtomId,
                             cnt: u64,
                             dtype: NumericDType,
                             tensor_id: Option<GlobalId>,
                             access_lo: u64,
                             access_hi: u64| {
        if access_lo >= access_hi {
            return;
        }
        let source_hi = base.0.saturating_add(cnt);
        let overlap_lo = base.0.max(access_lo);
        let overlap_hi = source_hi.min(access_hi);
        if overlap_lo >= overlap_hi {
            return;
        }

        // Sample the overlap region.
        let mid = overlap_lo + (overlap_hi - overlap_lo) / 2;
        if span_graphs[lane].contains_atom(AtomId(overlap_lo))
            && span_graphs[lane].contains_atom(AtomId(overlap_hi - 1))
            && span_graphs[lane].contains_atom(AtomId(mid))
        {
            return; // Already in span (produced by a same-phase, same-lane group).
        }

        let key = (overlap_lo, overlap_hi, dtype, tensor_id);
        if !declared_inputs[lane].insert(key) {
            return;
        }

        let overlap_count = overlap_hi - overlap_lo;
        span_inputs[lane].push(AtomRange {
            base: AtomId(overlap_lo),
            count: overlap_count,
            dtype,
        });
        span_graphs[lane].insert_input_tensor_at_allow_overlap(
            AtomId(overlap_lo),
            tensor_id.unwrap_or(GlobalId(0)),
            overlap_count,
            dtype,
        );
    };

    // Build precise access segments when possible (half-open [lo, hi)).
    let reduce_cfg = match op {
        ScalarOp::Reduce {
            reduce_count,
            reduce_stride,
            ..
        } if *reduce_count > 1 && *reduce_stride != 0 => Some((*reduce_count, *reduce_stride)),
        _ => None,
    };

    let mut segments: Vec<(u64, u64)> = Vec::new();
    for input in inputs {
        let base_segments = input_access_segments(&input.input_ref, atom_offset, count);
        if let Some((reduce_count, reduce_stride)) = reduce_cfg {
            segments.extend(expand_segments_for_reduce(
                &base_segments,
                reduce_count,
                reduce_stride,
            ));
        } else {
            segments.extend(base_segments);
        }
    }

    // IndirectLoad table: need the full table range.
    // May be a group (inlined) or an InputTensor (not inlined).
    if let ScalarOp::IndirectLoad { table_base, .. } = op {
        if let Some(tg) = graph.group_of(*table_base) {
            segments.push((tg.base_id.0, tg.base_id.0.saturating_add(tg.count)));
        } else {
            for (ti, _) in graph.find_input_idxs(*table_base) {
                let it = &graph.input_tensors()[ti];
                segments.push((it.base_id.0, it.base_id.0.saturating_add(it.count)));
            }
        }
    }

    let segments = normalize_half_open_segments(segments);

    // For each accessed segment, find main-graph groups and input tensors
    // that intersect it.
    let all_groups = graph.groups();
    let all_inputs = graph.input_tensors();
    for (lo, hi) in &segments {
        // Check input tensors (typically few, linear scan is fine).
        for it in all_inputs {
            let it_lo = it.base_id.0;
            let it_hi = it_lo.saturating_add(it.count);
            if *lo < it_hi && it_lo < *hi {
                declare_range(it.base_id, it.count, it.dtype, Some(it.tensor_id), *lo, *hi);
            }
        }

        // Check producer groups using binary search.
        // Groups are sorted by base_id. Find first group that could overlap [lo, hi).
        let start_idx = all_groups.partition_point(|g| g.base_id.0 + g.count <= *lo);
        for pg in &all_groups[start_idx..] {
            if pg.base_id.0 >= *hi {
                break;
            }
            let pg_end = pg.base_id.0.saturating_add(pg.count);
            if *lo < pg_end && pg.base_id.0 < *hi {
                declare_range(pg.base_id, pg.count, pg.output_dtype, None, *lo, *hi);
            }
        }
    }
}

/// Find the input tensor covering an atom ID.
fn find_covering_input_tensor<'a>(
    atom: &AtomId,
    input_tensor_set: &HashMap<u64, &'a InputTensor>,
    graph: &NanoGraph<'static, crate::pool::SystemPool>,
) -> Option<InputTensor> {
    // Check graph's input tensors via find_input_idx
    if let Some((idx, _)) = graph.find_input_idxs(*atom).next() {
        let it = &graph.input_tensors()[idx];
        return Some(it.clone());
    }
    None
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GlobalId;
    use crate::nano_graph::ops::{ReduceKind, ScalarBinOp, ScalarOp, ScalarUnaryOp};
    use crate::nano_graph::pattern::{GroupInput, InputTensor};
    use crate::numeric_dtype::NumericDType;
    use crate::numeric_scalar::NumericScalar;

    fn gi(ir: InputRef) -> GroupInput {
        GroupInput::scalar(ir)
    }

    /// Helper: count total atoms across all lanes in a phase.
    fn phase_total_atoms(phase: &Phase) -> u64 {
        phase
            .spans
            .iter()
            .map(|s| s.graph.groups().iter().map(|g| g.count).sum::<u64>())
            .sum()
    }

    /// Helper: count atoms per lane in a phase.
    fn atoms_per_lane(phase: &Phase) -> Vec<u64> {
        phase
            .spans
            .iter()
            .map(|s| {
                s.graph
                    .groups()
                    .iter()
                    .filter(|g| !matches!(g.op, ScalarOp::Literal(_) | ScalarOp::LiteralSpan(_)))
                    .map(|g| g.count)
                    .sum::<u64>()
            })
            .collect()
    }

    /// Helper: count how many lanes have non-zero work in a phase.
    fn active_lanes(phase: &Phase) -> usize {
        phase
            .spans
            .iter()
            .filter(|s| s.graph.num_groups() > 0)
            .count()
    }

    #[test]
    fn test_normalize_num_lanes() {
        assert_eq!(normalize_num_lanes(0), 1);
        assert_eq!(normalize_num_lanes(1), 1);
        assert_eq!(normalize_num_lanes(2), 2);
        assert_eq!(normalize_num_lanes(3), 3);
        assert_eq!(normalize_num_lanes(4), 4);
        assert_eq!(normalize_num_lanes(5), 5);
        assert_eq!(normalize_num_lanes(6), 6);
        assert_eq!(normalize_num_lanes(7), 7);
        assert_eq!(normalize_num_lanes(8), 8);
    }

    fn segments_contain(segs: &[(u64, u64)], atom: u64) -> bool {
        segs.iter().any(|(lo, hi)| *lo <= atom && atom < *hi)
    }

    #[test]
    fn test_input_access_segments_modular_wrap() {
        let input = InputRef::modular(AtomId(1_000), 1, 64);
        let segs = input_access_segments(&input, 60, 16);

        // Covers residues [60..64) U [0..12).
        assert!(segments_contain(&segs, 1_060));
        assert!(segments_contain(&segs, 1_063));
        assert!(segments_contain(&segs, 1_000));
        assert!(segments_contain(&segs, 1_011));
        assert!(!segments_contain(&segs, 1_050));
    }

    #[test]
    fn test_input_access_segments_nd_not_misclassified_as_strided_broadcast() {
        // 3D strided pattern seen in RWKV runs.
        let input = InputRef::Strided {
            base: AtomId(10_000),
            dim_strides: vec![64, 0, 1],
            dim_shape: vec![12, 64, 64],
        };
        let atom_offset = 8_192;
        let count = 8_192;
        let segs = input_access_segments(&input, atom_offset, count);

        let first = input.resolve(atom_offset).0;
        let mid = input.resolve(atom_offset + count / 2).0;
        let last = input.resolve(atom_offset + count - 1).0;

        assert!(segments_contain(&segs, first));
        assert!(segments_contain(&segs, mid));
        assert!(segments_contain(&segs, last));
    }

    /// Helper: validate all span nanographs.
    fn validate_all_spans(phases: &[Phase]) -> Vec<String> {
        let mut errors = Vec::new();
        for (pi, phase) in phases.iter().enumerate() {
            for (li, span) in phase.spans.iter().enumerate() {
                let span_errors = span.graph.validate();
                for err in span_errors {
                    errors.push(format!("Phase {} Lane {}: {}", pi, li, err));
                }
            }
        }
        errors
    }

    /// Verify no cross-span reads within a phase.
    /// Returns a list of violations.
    fn check_cross_lane_reads(phases: &[Phase]) -> Vec<String> {
        let mut violations = Vec::new();
        for (pi, phase) in phases.iter().enumerate() {
            // Collect atom ranges produced by each span's groups.
            let mut span_produces: Vec<Vec<(u64, u64)>> = Vec::new(); // (base, end)
            for span in &phase.spans {
                let mut ranges = Vec::new();
                for group in span.graph.groups() {
                    ranges.push((group.base_id.0, group.base_id.0 + group.count));
                }
                span_produces.push(ranges);
            }

            // Check: no span's declared inputs overlap with another span's produced atoms.
            for (li, span) in phase.spans.iter().enumerate() {
                for input in &span.inputs {
                    let inp_base = input.base.0;
                    let inp_end = inp_base + input.count;
                    for (other_li, other_ranges) in span_produces.iter().enumerate() {
                        if other_li == li {
                            continue;
                        }
                        for &(prod_base, prod_end) in other_ranges {
                            if inp_base < prod_end && prod_base < inp_end {
                                violations.push(format!(
                                    "Phase {} lane {} input [{}, {}) overlaps lane {} produced [{}, {})",
                                    pi, li, inp_base, inp_end, other_li, prod_base, prod_end,
                                ));
                            }
                        }
                    }
                }
            }
        }
        violations
    }

    /// Full plan verification: span validation + cross-lane checks.
    fn verify_plan_full(phases: &[Phase]) {
        let errors = validate_all_spans(phases);
        assert!(
            errors.is_empty(),
            "Validation errors:\n{}",
            errors.join("\n")
        );
        let violations = check_cross_lane_reads(phases);
        assert!(
            violations.is_empty(),
            "Cross-lane violations ({}):\n{}",
            violations.len(),
            violations
                .iter()
                .take(10)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    // ─── Test: Linear chain splitting ────────────────────────────────────

    /// A chain: Lit(8000) → Sub(8000) → Pow(8000) → Add(8000)
    /// All elementwise with stride=1. Should be ONE phase with all 4 lanes
    /// each getting 2000 atoms of Sub, Pow, Add (Lit duplicated).
    #[test]
    fn test_linear_chain_split() {
        let mut g = NanoGraph::new();
        let num_lanes = 4;

        let lit = g.push_group(
            8000,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(1.0)),
            vec![],
            vec![],
        );

        let sub = g.push_group(
            8000,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Sub,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(lit, 1)), gi(InputRef::affine(lit, 1))],
        );

        let pow = g.push_group(
            8000,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Pow,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![
                gi(InputRef::affine(sub, 1)),
                gi(InputRef::Broadcast(lit)), // broadcast a literal atom
            ],
        );

        let add = g.push_group(
            8000,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(pow, 1)), gi(InputRef::affine(sub, 1))],
        );

        g.outputs = vec![g.atom_to_range(add)];

        let input_tensors = vec![];
        let phases = plan(&g, num_lanes, &input_tensors, &g.outputs.clone());

        // Should be exactly 1 phase (no barriers needed).
        assert_eq!(
            phases.len(),
            1,
            "Linear chain should need only 1 phase, got {}",
            phases.len()
        );

        // All lanes should be active.
        assert_eq!(
            active_lanes(&phases[0]),
            num_lanes,
            "All {} lanes should be active",
            num_lanes,
        );

        // Each lane should have roughly 2000 atoms of Sub + 2000 of Pow + 2000 of Add.
        // Plus 8000 literal atoms (duplicated).
        let per_lane = atoms_per_lane(&phases[0]);
        for (lane, &atoms) in per_lane.iter().enumerate() {
            assert!(
                atoms >= 5000 && atoms <= 7000,
                "Lane {} has {} non-literal atoms, expected ~6000",
                lane,
                atoms,
            );
        }

        // Validate all span NanoGraphs.
        let errors = validate_all_spans(&phases);
        assert!(errors.is_empty(), "Validation errors: {:?}", errors);
    }

    // ─── Test: Diamond graph ─────────────────────────────────────────────

    /// Diamond: A(8000) → B(8000) and A(8000) → C(8000), then D = B + C (8000).
    /// All elementwise. Should be 1 phase, split across lanes.
    #[test]
    fn test_diamond_graph() {
        let mut g = NanoGraph::new();
        let num_lanes = 4;

        let a = g.push_group(
            8000,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(1.0)),
            vec![],
            vec![],
        );

        let b = g.push_group(
            8000,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(a, 1))],
        );

        let c = g.push_group(
            8000,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(a, 1))],
        );

        let d = g.push_group(
            8000,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(b, 1)), gi(InputRef::affine(c, 1))],
        );

        g.outputs = vec![g.atom_to_range(d)];

        let phases = plan(&g, num_lanes, &[], &g.outputs.clone());

        // Should be 1 phase — all lane-local.
        assert_eq!(
            phases.len(),
            1,
            "Diamond should be 1 phase, got {}",
            phases.len()
        );

        // All lanes active.
        assert_eq!(active_lanes(&phases[0]), num_lanes);

        // Each lane should have B + C + D fragments = 3 * 2000 = 6000 non-lit atoms.
        let per_lane = atoms_per_lane(&phases[0]);
        for (lane, &atoms) in per_lane.iter().enumerate() {
            assert!(
                atoms >= 5000 && atoms <= 7000,
                "Lane {} has {} non-literal atoms, expected ~6000",
                lane,
                atoms,
            );
        }

        let errors = validate_all_spans(&phases);
        assert!(errors.is_empty(), "Validation errors: {:?}", errors);
    }

    // ─── Test: MatMul-like structure with split ──────────────────────────

    /// Simulates a simplified matmul: M=32 output rows, K=64 reduction dim.
    /// Structure:
    ///   weights: Literal(32*64 = 2048) — the weight matrix
    ///   input: InputTensor(64) — input vector
    ///   mul: Binary::Mul(2048) with StridedBroadcast — M rows of K products
    ///   reduce: Reduce(32) with reduce_count=64, reduce_stride=1 — sum each row
    ///   bias: Literal(32) — bias
    ///   add: Binary::Add(32) — output + bias
    ///
    /// Expected: mul split across 4 lanes (512 atoms each), reduce split (8 each),
    /// add split (8 each). May need barrier between mul and reduce if reduce
    /// reads cross-lane data.
    #[test]
    fn test_matmul_structure() {
        let mut g = NanoGraph::new();
        let num_lanes = 4;
        let m = 32u64;
        let k = 64u64;

        // Input vector (external).
        let inp = g.add_input_tensor(GlobalId(0), k, NumericDType::F32);

        // Weight literal.
        let weights = g.push_group(
            m * k,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(0.5)),
            vec![],
            vec![],
        );

        // Mul: each of M*K atoms computes weight[i] * input[i % K].
        // Input pattern: Modular(base=inp, stride=1, modulus=K)
        // Weight pattern: Affine(base=weights, stride=1)
        let mul = g.push_group(
            m * k,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![
                gi(InputRef::affine(weights, 1)),
                gi(InputRef::modular(inp, 1, k)),
            ],
        );

        // Reduce: M output atoms, each sums K consecutive mul outputs.
        let reduce = g.push_group(
            m,
            NumericDType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: k,
                reduce_stride: 1,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(mul, k as i64))],
        );

        // Bias literal.
        let bias = g.push_group(
            m,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(0.1)),
            vec![],
            vec![],
        );

        // Add: reduce + bias.
        let add = g.push_group(
            m,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![
                gi(InputRef::affine(reduce, 1)),
                gi(InputRef::affine(bias, 1)),
            ],
        );

        g.outputs = vec![g.atom_to_range(add)];

        let it = vec![InputTensor {
            tensor_id: GlobalId(0),
            base_id: inp,
            count: k,
            dtype: NumericDType::F32,
        }];

        let phases = plan(&g, num_lanes, &it, &g.outputs.clone());

        // Validate.
        let errors = validate_all_spans(&phases);
        assert!(errors.is_empty(), "Validation errors: {:?}", errors);

        // The mul group has Modular input (reads all K elements from input vector),
        // which is NOT lane-local. So mul needs the input vector duplicated/declared.
        // But mul itself is split by its M*K atoms.

        // Check that groups ARE actually split across lanes.
        let mut lanes_with_mul = 0;
        for phase in &phases {
            for span in &phase.spans {
                let mut has_mul = false;
                for g in span.graph.groups() {
                    if matches!(
                        g.op,
                        ScalarOp::Binary {
                            op: ScalarBinOp::Mul,
                            ..
                        }
                    ) {
                        has_mul = true;
                    }
                }
                if has_mul {
                    lanes_with_mul += 1;
                }
            }
        }

        // The mul group (2048 atoms) should be split across all 4 lanes.
        assert!(
            lanes_with_mul >= 2,
            "Mul group should be split across multiple lanes, only found on {} lanes",
            lanes_with_mul,
        );
    }

    // ─── Test: Literal duplication ───────────────────────────────────────

    /// Literals should be duplicated into every lane, not split.
    #[test]
    fn test_literal_duplication() {
        let mut g = NanoGraph::new();
        let num_lanes = 4;

        let lit = g.push_group(
            1000,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(42.0)),
            vec![],
            vec![],
        );

        let neg = g.push_group(
            1000,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(lit, 1))],
        );

        g.outputs = vec![g.atom_to_range(neg)];

        let phases = plan(&g, num_lanes, &[], &g.outputs.clone());

        // Each lane should have the full literal (1000 atoms).
        for (lane, span) in phases[0].spans.iter().enumerate() {
            let lit_atoms: u64 = span
                .graph
                .groups()
                .iter()
                .filter(|g| matches!(g.op, ScalarOp::Literal(_) | ScalarOp::LiteralSpan(_)))
                .map(|g| g.count)
                .sum();
            assert_eq!(
                lit_atoms, 1000,
                "Lane {} should have 1000 literal atoms, got {}",
                lane, lit_atoms,
            );
        }

        // Total neg atoms across all lanes should sum to 1000. Per-lane
        // counts may differ from 250 due to cache-line-aligned splitting
        // (boundaries rounded to 16-atom multiples for F32).
        let total_neg: u64 = phases[0]
            .spans
            .iter()
            .flat_map(|s| s.graph.groups().iter())
            .filter(|g| matches!(g.op, ScalarOp::Unary { .. }))
            .map(|g| g.count)
            .sum();
        assert_eq!(
            total_neg, 1000,
            "Total neg atoms across lanes should be 1000, got {}",
            total_neg,
        );
        // Every lane should have a non-zero share.
        for (lane, span) in phases[0].spans.iter().enumerate() {
            let neg_atoms: u64 = span
                .graph
                .groups()
                .iter()
                .filter(|g| matches!(g.op, ScalarOp::Unary { .. }))
                .map(|g| g.count)
                .sum();
            assert!(
                neg_atoms > 0,
                "Lane {} should have non-zero neg atoms",
                lane,
            );
        }

        let errors = validate_all_spans(&phases);
        assert!(errors.is_empty(), "Validation errors: {:?}", errors);
    }

    // ─── Test: Reduce handling ───────────────────────────────────────────

    /// ReduceSum over a large group, then downstream elementwise.
    /// Structure: Lit(1024) → ReduceSum(1) → Neg (broadcast from reduce output)
    ///
    /// The reduce is a scalar output (count=1), so it can't be split.
    /// It should be Whole on lane 0, and its output duplicated for downstream use.
    #[test]
    fn test_reduce_to_scalar() {
        let mut g = NanoGraph::new();
        let num_lanes = 4;

        let data = g.push_group(
            1024,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(1.0)),
            vec![],
            vec![],
        );

        let reduced = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: 1024,
                reduce_stride: 1,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(data, 1))],
        );

        // Use the scalar result in a large group (broadcast).
        let output = g.push_group(
            8000,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::Broadcast(reduced))],
        );

        g.outputs = vec![g.atom_to_range(output)];

        let phases = plan(&g, num_lanes, &[], &g.outputs.clone());

        let errors = validate_all_spans(&phases);
        assert!(errors.is_empty(), "Validation errors: {:?}", errors);

        // The output group (8000 atoms) should be split across lanes.
        let last_phase = phases.last().unwrap();
        let output_lanes_active = last_phase
            .spans
            .iter()
            .filter(|s| {
                s.graph.groups().iter().any(|g| {
                    matches!(
                        g.op,
                        ScalarOp::Unary {
                            op: ScalarUnaryOp::Neg,
                            ..
                        }
                    )
                })
            })
            .count();
        assert!(
            output_lanes_active >= 2,
            "Output Neg should be split across lanes, only on {} lanes",
            output_lanes_active,
        );
    }

    // ─── Test: Reduce with splittable output dimension ───────────────────

    /// M independent reductions: Lit(M*K) → ReduceSum(M, reduce_count=K).
    /// The M reductions are independent and should be split across lanes.
    #[test]
    fn test_reduce_split_by_output() {
        let mut g = NanoGraph::new();
        let num_lanes = 4;
        let m = 64u64;
        let k = 16u64;

        let data = g.push_group(
            m * k,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(1.0)),
            vec![],
            vec![],
        );

        let reduced = g.push_group(
            m,
            NumericDType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: k,
                reduce_stride: 1,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(data, k as i64))],
        );

        g.outputs = vec![g.atom_to_range(reduced)];

        let phases = plan(&g, num_lanes, &[], &g.outputs.clone());

        let errors = validate_all_spans(&phases);
        assert!(errors.is_empty(), "Validation errors: {:?}", errors);

        // The reduce group (64 elements) should be split across 4 lanes (16 each).
        let mut lanes_with_reduce = 0;
        for phase in &phases {
            for span in &phase.spans {
                if span.graph.groups().iter().any(|g| g.op.is_reduce()) {
                    lanes_with_reduce += 1;
                }
            }
        }
        assert_eq!(
            lanes_with_reduce, num_lanes,
            "Reduce should be split across all {} lanes, found on {}",
            num_lanes, lanes_with_reduce,
        );
    }

    // ─── Test: Groups are ACTUALLY split (not just shuffled) ─────────────

    /// Verify that a single large group becomes multiple smaller fragments
    /// across lanes, and the fragments' atom ranges are disjoint and
    /// cover the full original range.
    #[test]
    fn test_split_coverage() {
        let mut g = NanoGraph::new();
        let num_lanes = 4;

        let lit = g.push_group(
            400,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(1.0)),
            vec![],
            vec![],
        );

        let op = g.push_group(
            400,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(lit, 1))],
        );

        g.outputs = vec![g.atom_to_range(op)];

        let phases = plan(&g, num_lanes, &[], &g.outputs.clone());

        // Collect all unary group fragments across all lanes.
        let mut fragments: Vec<(u64, u64)> = Vec::new(); // (base, count)
        for phase in &phases {
            for span in &phase.spans {
                for group in span.graph.groups() {
                    if matches!(group.op, ScalarOp::Unary { .. }) {
                        fragments.push((group.base_id.0, group.count));
                    }
                }
            }
        }

        // Should have exactly num_lanes fragments.
        assert_eq!(
            fragments.len(),
            num_lanes,
            "Expected {} fragments, got {}",
            num_lanes,
            fragments.len(),
        );

        // Sort by base and verify contiguous coverage.
        fragments.sort_by_key(|f| f.0);
        let op_base = lit.0 + 400; // op starts after lit
        assert_eq!(
            fragments[0].0, op_base,
            "First fragment should start at op base"
        );

        let mut total = 0u64;
        let mut prev_end = op_base;
        for (base, count) in &fragments {
            assert_eq!(*base, prev_end, "Fragments should be contiguous");
            total += count;
            prev_end = base + count;
        }
        assert_eq!(
            total, 400,
            "Total fragment atoms should equal original group count"
        );

        let errors = validate_all_spans(&phases);
        assert!(errors.is_empty(), "Validation errors: {:?}", errors);
    }

    // ─── Test: Input tensor handling ─────────────────────────────────────

    /// External input tensor is consumed by a split group.
    #[test]
    fn test_input_tensor_consumed() {
        let mut g = NanoGraph::new();
        let num_lanes = 4;

        let inp = g.add_input_tensor(GlobalId(0), 800, NumericDType::F32);

        let neg = g.push_group(
            800,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(inp, 1))],
        );

        g.outputs = vec![g.atom_to_range(neg)];

        let it = vec![InputTensor {
            tensor_id: GlobalId(0),
            base_id: inp,
            count: 800,
            dtype: NumericDType::F32,
        }];

        let phases = plan(&g, num_lanes, &it, &g.outputs.clone());

        // Each lane should declare the input tensor.
        for (lane, span) in phases[0].spans.iter().enumerate() {
            assert!(
                !span.inputs.is_empty(),
                "Lane {} should have input declarations",
                lane,
            );
        }

        // Neg should be split across all 4 lanes.
        let mut neg_lanes = 0;
        for span in &phases[0].spans {
            if span
                .graph
                .groups()
                .iter()
                .any(|g| matches!(g.op, ScalarOp::Unary { .. }))
            {
                neg_lanes += 1;
            }
        }
        assert_eq!(neg_lanes, num_lanes);

        let errors = validate_all_spans(&phases);
        assert!(errors.is_empty(), "Validation errors: {:?}", errors);
    }

    // ─── Test: Empty graph ───────────────────────────────────────────────

    #[test]
    fn test_empty_graph() {
        let g = NanoGraph::new();
        let phases = plan(&g, 4, &[], &[]);
        assert_eq!(phases.len(), 1);
        assert_eq!(phases[0].spans.len(), 4);
    }

    // ─── Test: Single atom group ─────────────────────────────────────────

    #[test]
    fn test_single_atom() {
        let mut g = NanoGraph::new();

        let lit = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(5.0)),
            vec![],
            vec![],
        );

        g.outputs = vec![g.atom_to_range(lit)];

        let phases = plan(&g, 4, &[], &g.outputs.clone());

        let errors = validate_all_spans(&phases);
        assert!(errors.is_empty(), "Validation errors: {:?}", errors);

        // Single atom literal should be duplicated, at least lane 0 has it.
        assert!(phases[0].spans[0].graph.num_groups() > 0);
    }

    // ─── Test: Cross-lane violation checks on existing tests ─────────────

    #[test]
    fn test_linear_chain_no_cross_lane() {
        let mut g = NanoGraph::new();
        let lit = g.push_group(
            8000,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(1.0)),
            vec![],
            vec![],
        );
        let sub = g.push_group(
            8000,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Sub,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(lit, 1)), gi(InputRef::affine(lit, 1))],
        );
        let pow = g.push_group(
            8000,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Pow,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(sub, 1)), gi(InputRef::Broadcast(lit))],
        );
        let add = g.push_group(
            8000,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(pow, 1)), gi(InputRef::affine(sub, 1))],
        );
        g.outputs = vec![g.atom_to_range(add)];
        let phases = plan(&g, 4, &[], &g.outputs.clone());
        verify_plan_full(&phases);
    }

    #[test]
    fn test_matmul_no_cross_lane() {
        let mut g = NanoGraph::new();
        let num_lanes = 4;
        let m = 32u64;
        let k = 64u64;
        let inp = g.add_input_tensor(GlobalId(0), k, NumericDType::F32);
        let weights = g.push_group(
            m * k,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(0.5)),
            vec![],
            vec![],
        );
        let mul = g.push_group(
            m * k,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![
                gi(InputRef::affine(weights, 1)),
                gi(InputRef::modular(inp, 1, k)),
            ],
        );
        let reduce = g.push_group(
            m,
            NumericDType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: k,
                reduce_stride: 1,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(mul, k as i64))],
        );
        let bias = g.push_group(
            m,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(0.1)),
            vec![],
            vec![],
        );
        let add = g.push_group(
            m,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![
                gi(InputRef::affine(reduce, 1)),
                gi(InputRef::affine(bias, 1)),
            ],
        );
        g.outputs = vec![g.atom_to_range(add)];
        let it = vec![InputTensor {
            tensor_id: GlobalId(0),
            base_id: inp,
            count: k,
            dtype: NumericDType::F32,
        }];
        let phases = plan(&g, num_lanes, &it, &g.outputs.clone());
        verify_plan_full(&phases);
    }

    /// Relayout should also catch the variant where the second Mul input is
    /// already a 2D strided view [s, s*K] with shape [MAX, N].
    #[test]
    fn test_matmul_relayout_for_strided_nk_variant() {
        let mut g = NanoGraph::new();
        let num_lanes = 4;
        let n = 8u64;
        let k = 4u64;

        let a = g.push_group(
            k,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(1.0)),
            vec![],
            vec![],
        );
        let b = g.push_group(
            n * k,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(2.0)),
            vec![],
            vec![],
        );

        let mul = g.push_group(
            n * k,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![
                gi(InputRef::strided_broadcast(a, 1, n)),
                gi(InputRef::Strided {
                    base: b,
                    dim_strides: vec![1, k as i64],
                    dim_shape: vec![u64::MAX, n],
                }),
            ],
        );

        let reduce = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: k,
                reduce_stride: n as i64,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(mul, 1))],
        );
        g.outputs = vec![g.atom_to_range(reduce)];

        let phases = plan(&g, num_lanes, &[], &g.outputs.clone());
        verify_plan_full(&phases);

        assert_eq!(
            phases.len(),
            1,
            "relayouted [K,N] matmul variant should stay in one phase"
        );

        for span in &phases[0].spans {
            let Some(rg) = span.graph.groups().iter().find(|g| g.op.is_reduce()) else {
                continue;
            };
            if let ScalarOp::Reduce { reduce_stride, .. } = &rg.op {
                assert_eq!(
                    *reduce_stride, 1,
                    "reduce_stride should be relayouted to 1 for lane-local splits"
                );
            }
            if let Some(InputRef::Strided { dim_strides, .. }) =
                rg.inputs.first().map(|gi| &gi.input_ref)
            {
                assert_eq!(
                    dim_strides.len(),
                    1,
                    "reduce input should remain affine after relayout"
                );
                assert_eq!(
                    dim_strides[0], k as i64,
                    "reduce input affine stride should be K after relayout"
                );
            } else {
                panic!("expected affine reduce input after relayout");
            }
        }
    }

    #[test]
    fn test_small_modular_source_promoted_to_duplicate() {
        let mut g = NanoGraph::new();
        let num_lanes = 4;

        let a = g.push_group(
            8,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(1.0)),
            vec![],
            vec![],
        );
        let b = g.push_group(
            8,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(2.0)),
            vec![],
            vec![],
        );
        let src = g.push_group(
            8,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(a, 1)), gi(InputRef::affine(b, 1))],
        );

        let expanded = 131_072u64; // 8 * 16_384
        let w = g.push_group(
            expanded,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(3.0)),
            vec![],
            vec![],
        );
        let out = g.push_group(
            expanded,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::modular(src, 1, 8)), gi(InputRef::affine(w, 1))],
        );
        g.outputs = vec![g.atom_to_range(out)];

        let phases = plan(&g, num_lanes, &[], &g.outputs.clone());
        verify_plan_full(&phases);

        assert_eq!(
            phases.len(),
            1,
            "small modular source should be duplicated instead of forcing a barrier"
        );

        let mut lanes_with_src = 0usize;
        for span in &phases[0].spans {
            if span
                .graph
                .groups()
                .iter()
                .any(|grp| grp.base_id == src && grp.count == 8)
            {
                lanes_with_src += 1;
            }
        }
        assert_eq!(
            lanes_with_src, num_lanes,
            "modular source should be duplicated to all lanes"
        );
    }

    // ─── Test: LayerNorm-like structure ──────────────────────────────────

    /// Simulates LayerNorm: x(N) → mean(N/D) → x-mean(N) → var(N/D) → rsqrt(N/D) → normalize(N)
    /// N=4096, D=768 (so N/D = 64/12 = ~5.3.. use nice numbers: N=3072, D=768, M=4)
    /// This exercises the broadcast + StridedBroadcast patterns.
    #[test]
    fn test_layernorm_pattern() {
        let mut g = NanoGraph::new();
        let num_lanes = 8;
        let d = 768u64; // hidden dim
        let m = 8u64; // batch*seq (must be divisible by num_lanes)
        let n = m * d; // total elements

        // Input: external tensor of size N.
        let x = g.add_input_tensor(GlobalId(0), n, NumericDType::F32);

        // Step 1: ReduceSum over D elements → M outputs.
        // reduce(M) reads x via Affine(stride=D).
        let sum = g.push_group(
            m,
            NumericDType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: d,
                reduce_stride: 1,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(x, d as i64))],
        );

        // Step 2: Divide by D to get mean. Broadcast a literal 1/D.
        let inv_d = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(1.0 / d as f32)),
            vec![],
            vec![],
        );
        let mean = g.push_group(
            m,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(sum, 1)), gi(InputRef::Broadcast(inv_d))],
        );

        // Step 3: x - mean. Uses StridedBroadcast to broadcast each mean value across D elements.
        let x_centered = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Sub,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![
                gi(InputRef::affine(x, 1)),
                gi(InputRef::strided_broadcast(mean, 1, d)),
            ],
        );

        // Step 4: x_centered^2
        let two = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(2.0)),
            vec![],
            vec![],
        );
        let pow2 = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Pow,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![
                gi(InputRef::affine(x_centered, 1)),
                gi(InputRef::Broadcast(two)),
            ],
        );

        // Step 5: ReduceSum of pow2 → M variance values.
        let var_sum = g.push_group(
            m,
            NumericDType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: d,
                reduce_stride: 1,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(pow2, d as i64))],
        );

        // Step 6: Divide by D and add epsilon, then rsqrt.
        let var_mean = g.push_group(
            m,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![
                gi(InputRef::affine(var_sum, 1)),
                gi(InputRef::Broadcast(inv_d)),
            ],
        );
        let eps = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(1e-5)),
            vec![],
            vec![],
        );
        let var_eps = g.push_group(
            m,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![
                gi(InputRef::affine(var_mean, 1)),
                gi(InputRef::Broadcast(eps)),
            ],
        );
        let sqrt_var = g.push_group(
            m,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Sqrt,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(var_eps, 1))],
        );
        let rsqrt = g.push_group(
            m,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Reciprocal,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(sqrt_var, 1))],
        );

        // Step 7: Normalize: x_centered * rsqrt (StridedBroadcast).
        let normalized = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![
                gi(InputRef::affine(x_centered, 1)),
                gi(InputRef::strided_broadcast(rsqrt, 1, d)),
            ],
        );

        g.outputs = vec![g.atom_to_range(normalized)];
        let it = vec![InputTensor {
            tensor_id: GlobalId(0),
            base_id: x,
            count: n,
            dtype: NumericDType::F32,
        }];
        let phases = plan(&g, num_lanes, &it, &g.outputs.clone());

        // Check no cross-lane violations.
        verify_plan_full(&phases);

        // The normalized output should be split across lanes.
        let mut lanes_with_norm = 0;
        for phase in &phases {
            for span in &phase.spans {
                if span.graph.groups().iter().any(|g| {
                    matches!(
                        g.op,
                        ScalarOp::Binary {
                            op: ScalarBinOp::Mul,
                            ..
                        }
                    ) && g.count > 1
                        && g.count < n
                }) {
                    lanes_with_norm += 1;
                }
            }
        }
        assert!(
            lanes_with_norm >= 2,
            "Normalized output should be split across lanes, found on {} lanes",
            lanes_with_norm
        );
    }

    // ─── Test: StridedBroadcast cross-lane edge case ─────────────────────

    /// Tests that StridedBroadcast access from a split producer is correctly
    /// identified as lane-local when chunk size is a multiple of repeat.
    #[test]
    fn test_strided_broadcast_lane_local() {
        let mut g = NanoGraph::new();
        let num_lanes = 4;
        let m = 16u64;
        let d = 8u64; // repeat value
        let n = m * d; // 128 total elements

        // Source: split group of M elements.
        let source = g.push_group(
            m,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(1.0)),
            vec![],
            vec![],
        );

        // Consumer: N elements reading source via StridedBroadcast(repeat=D).
        // Each chunk of D consecutive consumer atoms reads the same source atom.
        let consumer = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::strided_broadcast(source, 1, d))],
        );

        g.outputs = vec![g.atom_to_range(consumer)];
        let phases = plan(&g, num_lanes, &[], &g.outputs.clone());
        verify_plan_full(&phases);
    }

    // ─── Test: StridedBroadcast NON-aligned (should trigger barrier) ─────

    /// Tests that StridedBroadcast access where chunk is NOT a multiple of
    /// repeat correctly gets a barrier (not falsely identified as lane-local).
    #[test]
    fn test_strided_broadcast_non_aligned() {
        let mut g = NanoGraph::new();
        let num_lanes = 4;
        let m = 12u64;
        let d = 7u64; // repeat that doesn't divide chunk
        let n = m * d; // 84 total elements

        // Source: split group of M elements.
        let source = g.push_group(
            m,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(1.0)),
            vec![],
            vec![],
        );

        // Consumer: N elements reading source via StridedBroadcast(repeat=D).
        let consumer = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::strided_broadcast(source, 1, d))],
        );

        g.outputs = vec![g.atom_to_range(consumer)];
        let phases = plan(&g, num_lanes, &[], &g.outputs.clone());
        verify_plan_full(&phases);
    }

    // ─── Test: Multiple matmul chain (GPT-2-like) ────────────────────────

    /// Tests a chain of two matmuls: input → matmul1 → matmul2 → output.
    /// This exercises cross-phase data flow with split groups.
    #[test]
    fn test_matmul_chain() {
        let mut g = NanoGraph::new();
        let num_lanes = 4;
        let m = 16u64;
        let k1 = 32u64;
        let k2 = 16u64;

        // Input vector.
        let inp = g.add_input_tensor(GlobalId(0), k1, NumericDType::F32);

        // MatMul 1: [M, K1] @ input[K1] → output[M]
        let w1 = g.push_group(
            m * k1,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(0.1)),
            vec![],
            vec![],
        );
        let mul1 = g.push_group(
            m * k1,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![
                gi(InputRef::affine(w1, 1)),
                gi(InputRef::modular(inp, 1, k1)),
            ],
        );
        let red1 = g.push_group(
            m,
            NumericDType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: k1,
                reduce_stride: 1,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(mul1, k1 as i64))],
        );

        // Activation (elementwise).
        let act = g.push_group(
            m,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Tanh,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(red1, 1))],
        );

        // MatMul 2: [K2, M] @ act[M] → output[K2]
        let w2 = g.push_group(
            k2 * m,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(0.2)),
            vec![],
            vec![],
        );
        let mul2 = g.push_group(
            k2 * m,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![
                gi(InputRef::affine(w2, 1)),
                gi(InputRef::modular(act, 1, m)),
            ],
        );
        let red2 = g.push_group(
            k2,
            NumericDType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: m,
                reduce_stride: 1,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(mul2, m as i64))],
        );

        g.outputs = vec![g.atom_to_range(red2)];
        let it = vec![InputTensor {
            tensor_id: GlobalId(0),
            base_id: inp,
            count: k1,
            dtype: NumericDType::F32,
        }];
        let phases = plan(&g, num_lanes, &it, &g.outputs.clone());
        verify_plan_full(&phases);
    }

    // ─── Test: Reduce from split source needs barrier ────────────────────

    /// A scalar reduce over a split source: needs barrier because the
    /// reduce must see all lanes' partial results.
    #[test]
    fn test_scalar_reduce_from_split() {
        let mut g = NanoGraph::new();
        let num_lanes = 4;

        let data = g.push_group(
            1000,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(1.0)),
            vec![],
            vec![],
        );
        let neg = g.push_group(
            1000,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(data, 1))],
        );
        // Scalar reduce: reads ALL 1000 atoms of neg.
        let reduced = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: 1000,
                reduce_stride: 1,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(neg, 1))],
        );
        // Broadcast reduced to large output.
        let output = g.push_group(
            1000,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::Broadcast(reduced))],
        );

        g.outputs = vec![g.atom_to_range(output)];
        let phases = plan(&g, num_lanes, &[], &g.outputs.clone());
        verify_plan_full(&phases);

        // Should need multiple phases: neg split, then barrier, then reduce, then output.
        assert!(
            phases.len() >= 2,
            "Should need at least 2 phases for reduce from split source, got {}",
            phases.len()
        );
    }

    // ─── Test: StridedBroadcast FALSE lane-local (distinct fits but misaligned) ──

    /// This tests a case where distinct_per_chunk <= prod_chunk (so the count
    /// of reads fits per lane) but the actual reads cross lane boundaries.
    /// Consumer: count=100, StridedBroadcast(stride=1, repeat=10), 4 lanes.
    /// Producer: count=40, split 4 ways (10 each).
    /// Lane 0 consumer chunk=[0..25), reads producer at base+i/10 → [base+0..base+2].
    /// Lane 1 consumer chunk=[25..50), reads base+25/10=base+2 through base+49/10=base+4.
    /// Lane 1 producer chunk=[base+10..base+20). But lane 1 reads base+2..base+4 → in lane 0!
    #[test]
    fn test_strided_broadcast_false_lane_local() {
        let mut g = NanoGraph::new();
        let num_lanes = 4;

        // Producer: 40 computed atoms.
        let lit = g.push_group(
            40,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(1.0)),
            vec![],
            vec![],
        );
        let source = g.push_group(
            40,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(lit, 1))],
        );

        // Consumer: 100 atoms reading source via StridedBroadcast(repeat=10).
        // chunk=25, distinct_per_chunk=ceil(25/10)=3, prod_chunk=10.
        // 3 <= 10 → would pass the simple check. But the access is NOT lane-local!
        let consumer = g.push_group(
            100,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::strided_broadcast(source, 1, 10))],
        );

        g.outputs = vec![g.atom_to_range(consumer)];
        let phases = plan(&g, num_lanes, &[], &g.outputs.clone());

        // This MUST have no cross-lane violations.
        verify_plan_full(&phases);
    }

    // ─── Test: Remainder in split (uneven division) ───────────────────────

    /// Tests groups where count % num_lanes != 0, exercising remainder handling.
    #[test]
    fn test_uneven_split() {
        let mut g = NanoGraph::new();
        let num_lanes = 4;

        // count=103 doesn't divide evenly by 4 (25+25+25+28 or 26+26+26+25).
        let lit = g.push_group(
            103,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(1.0)),
            vec![],
            vec![],
        );
        let neg = g.push_group(
            103,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(lit, 1))],
        );
        let exp = g.push_group(
            103,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(neg, 1))],
        );

        g.outputs = vec![g.atom_to_range(exp)];
        let phases = plan(&g, num_lanes, &[], &g.outputs.clone());
        verify_plan_full(&phases);
    }

    // ─── Test: Large matmul with attention-like dimensions ───────────────

    /// Tests matmul with dimensions that match GPT-2 attention.
    #[test]
    fn test_attention_matmul() {
        let mut g = NanoGraph::new();
        let num_lanes = 8;
        let m = 64u64; // seq_len
        let k = 64u64; // head_dim

        let inp = g.add_input_tensor(GlobalId(0), k, NumericDType::F32);
        let weights = g.push_group(
            m * k,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(0.1)),
            vec![],
            vec![],
        );
        let mul = g.push_group(
            m * k,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![
                gi(InputRef::affine(weights, 1)),
                gi(InputRef::modular(inp, 1, k)),
            ],
        );
        let reduce = g.push_group(
            m,
            NumericDType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: k,
                reduce_stride: 1,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(mul, k as i64))],
        );

        g.outputs = vec![g.atom_to_range(reduce)];
        let it = vec![InputTensor {
            tensor_id: GlobalId(0),
            base_id: inp,
            count: k,
            dtype: NumericDType::F32,
        }];
        let phases = plan(&g, num_lanes, &it, &g.outputs.clone());
        verify_plan_full(&phases);
    }

    // ─── Test: Multiple StridedBroadcast patterns ────────────────────────

    /// Tests a chain where StridedBroadcast is used with different repeat values.
    #[test]
    fn test_multi_strided_broadcast() {
        let mut g = NanoGraph::new();
        let num_lanes = 8;
        let m = 8u64;
        let d1 = 768u64;
        let d2 = 64u64;
        let n1 = m * d1; // 6144
        let n2 = m * d2; // 512

        // Two source vectors of different sizes.
        let src1 = g.push_group(
            m,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(1.0)),
            vec![],
            vec![],
        );
        let src2 = g.push_group(
            m,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(2.0)),
            vec![],
            vec![],
        );

        // Apply StridedBroadcast with repeat=768.
        let expanded1 = g.push_group(
            n1,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::strided_broadcast(src1, 1, d1))],
        );

        // Apply StridedBroadcast with repeat=64.
        let expanded2 = g.push_group(
            n2,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::strided_broadcast(src2, 1, d2))],
        );

        g.outputs = vec![g.atom_to_range(expanded1), g.atom_to_range(expanded2)];
        let phases = plan(&g, num_lanes, &[], &g.outputs.clone());
        verify_plan_full(&phases);
    }

    // ─── Test: Whole group between split groups ──────────────────────────

    /// A Whole group (explicit inputs) between two split groups.
    /// The whole group is on lane 0, needs barriers on both sides.
    #[test]
    fn test_whole_group_sandwich() {
        let mut g = NanoGraph::new();
        let num_lanes = 4;
        let n = 100u64;

        // Split input.
        let data = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(1.0)),
            vec![],
            vec![],
        );
        let neg = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(data, 1))],
        );

        // Whole group with Explicit input (reverse order).
        let explicit_ids: Vec<AtomId> = (0..n).rev().map(|i| AtomId(neg.0 + i)).collect();
        let reversed = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Identity,
            vec![],
            vec![gi(InputRef::Explicit(explicit_ids))],
        );

        // Split output reading from whole.
        let output = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(reversed, 1))],
        );

        g.outputs = vec![g.atom_to_range(output)];
        let phases = plan(&g, num_lanes, &[], &g.outputs.clone());
        verify_plan_full(&phases);
    }

    /// Explicit sparse input refs should only declare the actually used
    /// input atoms, not the full input tensor hull.
    #[test]
    fn test_explicit_sparse_inputs_are_clipped() {
        let mut g = NanoGraph::new();
        let num_lanes = 1;

        let input = g.add_input_tensor(GlobalId(0), 1024, NumericDType::F32);
        let explicit_ids = vec![input.offset(0), input.offset(511), input.offset(1023)];

        let out = g.push_group(
            explicit_ids.len() as u64,
            NumericDType::F32,
            ScalarOp::Identity,
            vec![],
            vec![gi(InputRef::Explicit(explicit_ids))],
        );

        g.outputs = vec![g.atom_to_range(out)];
        let it = vec![InputTensor {
            tensor_id: GlobalId(0),
            base_id: input,
            count: 1024,
            dtype: NumericDType::F32,
        }];

        let phases = plan(&g, num_lanes, &it, &g.outputs.clone());
        verify_plan_full(&phases);
        assert_eq!(phases.len(), 1);

        let span = &phases[0].spans[0];
        let total_input_atoms: u64 = span.inputs.iter().map(|r| r.count).sum();
        assert_eq!(
            total_input_atoms, 3,
            "expected sparse explicit access to declare only 3 atoms, got {total_input_atoms}"
        );

        let mut bases: Vec<u64> = span.inputs.iter().map(|r| r.base.0).collect();
        bases.sort_unstable();
        assert_eq!(bases, vec![input.0, input.0 + 511, input.0 + 1023]);
    }

    // ─── Test: GroupNorm-like reduce stays lane-local with aligned splits ──

    /// Regression test for the RWKV7 GroupNorm pattern:
    /// 768 atoms (12 heads × 64 head_dim), ReduceSum with stride=64 and
    /// reduce_count=64. When split across 8 lanes, the consumer count (12)
    /// doesn't divide evenly, but the Mul→Reduce pair is a perfect tile
    /// (`reduce_extent == abs_stride == 64`), so aligned splitting gives
    /// each lane a matching producer/consumer chunk and no barrier is
    /// needed.
    #[test]
    fn test_groupnorm_reduce_needs_barrier() {
        let mut g = NanoGraph::new();
        let num_lanes = 8;
        let n_heads: u64 = 12;
        let head_dim: u64 = 64;
        let total = n_heads * head_dim; // 768

        // Input: 768 elements (simulating GroupNorm input after reshape).
        let data = g.push_group(
            total,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(1.0)),
            vec![],
            vec![],
        );

        // Elementwise op on the data (stands in for "x - mean").
        let processed = g.push_group(
            total,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(data, 1))],
        );

        // ReduceSum: 12 outputs, each summing 64 elements with stride=1.
        // Input ref: Affine(processed, stride=64) — output k reads
        // atoms [processed + 64*k .. processed + 64*k + 63].
        let reduced = g.push_group(
            n_heads,
            NumericDType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: head_dim,
                reduce_stride: 1,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(processed, head_dim as i64))],
        );

        // Output uses the reduced values.
        let output = g.push_group(
            n_heads,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![gi(InputRef::affine(reduced, 1))],
        );

        g.outputs = vec![g.atom_to_range(output)];
        let phases = plan(&g, num_lanes, &[], &g.outputs.clone());
        verify_plan_full(&phases);

        // The reduce has 12 output atoms — below the cache-line
        // threshold (16 F32 atoms) for safe multi-lane splitting.
        // Cache-line discipline demotes it to Whole, requiring a
        // barrier before downstream consumers. Production-scale
        // models have reduce counts >> 128 and remain single-phase.
        assert!(
            phases.len() <= 3,
            "GroupNorm reduce should need at most 3 phases, got {}",
            phases.len(),
        );
    }
}
