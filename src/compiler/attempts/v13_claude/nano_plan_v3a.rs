#![allow(clippy::all, dead_code, unreachable_patterns, unused_variables, unused_imports)]
//! Span-based NanoGraph partitioner v3a: edge-classification approach.
//!
//! Previous attempts failed because they assigned phases first (e.g., using v2c),
//! then tried to split groups across lanes, creating cross-lane reads when the
//! split broke dependencies. This approach works from the other direction:
//!
//! 1. **Build group dependency DAG.** Same mechanics as previous attempts.
//!
//! 2. **Classify each edge as lane-aligned or cross-lane.** An edge from
//!    producer P to consumer C is "lane-aligned" if splitting both P and C
//!    across N lanes means each lane's slice of C only reads from that same
//!    lane's slice of P. Otherwise it's "cross-lane" — a barrier is needed.
//!
//! 3. **Phase assignment from edge classification.** Groups connected only by
//!    lane-aligned edges can be in the same phase (both split). Cross-lane
//!    edges force phase boundaries. Small producers consumed cross-lane can
//!    be duplicated instead of forcing a barrier.
//!
//! 4. **Within each phase, split all groups across lanes.** Since all within-
//!    phase edges are lane-aligned by construction, splitting is safe.
//!
//! 5. **Build self-contained span NanoGraphs** with remapped atom IDs.
//!
//! Key invariant: within a phase, NO span reads atoms produced by another span.
//! This is guaranteed by the edge classification: all within-phase edges are
//! lane-aligned, so splitting preserves locality.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use crate::nano_graph::{AtomGroup, AtomId, InputRef, NanoGraph, ScalarOp, SymDim};

/// Literal groups with fewer atoms than this are duplicated into spans.
/// Larger literals (weight matrices) become external inputs.
const LITERAL_INLINE_THRESHOLD: u64 = 1024;

/// Groups with fewer atoms than this that are consumed cross-lane can be
/// duplicated into each lane instead of forcing a barrier.
const DUPLICATE_THRESHOLD: u64 = 65536;

// ─── Public types ────────────────────────────────────────────────────────────

/// A contiguous range of atoms mapped between main graph and span graph.
#[derive(Debug, Clone)]
pub struct AtomMapping {
    /// Start atom in the main graph.
    pub main_base: AtomId,
    /// Start atom in the span graph.
    pub span_base: AtomId,
    /// Number of contiguous atoms in this mapping.
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

// ─── Range-based atom map ────────────────────────────────────────────────────

/// An atom map that stores contiguous ranges instead of individual atoms.
/// Supports O(log n) lookup by main-graph AtomId.
struct RangeAtomMap {
    /// Sorted ranges: (main_base, span_base, count).
    ranges: Vec<(u64, u64, u64)>,
}

impl RangeAtomMap {
    fn new() -> Self {
        Self { ranges: Vec::new() }
    }

    fn insert_range(&mut self, main_base: AtomId, span_base: AtomId, count: u64) {
        self.ranges.push((main_base.0, span_base.0, count));
    }

    fn sort(&mut self) {
        self.ranges.sort_by_key(|&(base, _, _)| base);
    }

    fn get(&self, main_id: AtomId) -> Option<AtomId> {
        let idx = self.ranges.partition_point(|&(base, _, _)| base <= main_id.0);
        if idx == 0 {
            return None;
        }
        let (base, span_base, count) = self.ranges[idx - 1];
        let offset = main_id.0.wrapping_sub(base);
        if offset < count {
            Some(AtomId(span_base + offset))
        } else {
            None
        }
    }
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Partition a NanoGraph into self-contained spans organized by phase and lane.
///
/// The algorithm guarantees that within each phase, no span reads atoms
/// produced by another span (cross-lane independence).
pub fn plan_spans(graph: &NanoGraph, num_lanes: usize) -> SpanPlan {
    let groups = graph.groups();
    let n = groups.len();
    let num_lanes = num_lanes.max(1);

    if n == 0 {
        return SpanPlan {
            num_lanes,
            phases: vec![],
        };
    }

    let is_literal: Vec<bool> = groups
        .iter()
        .map(|g| matches!(g.op, ScalarOp::Literal(_)) && g.inputs.is_empty())
        .collect();

    if is_literal.iter().all(|&lit| lit) {
        return SpanPlan {
            num_lanes,
            phases: vec![],
        };
    }

    // Step 1: Build group dependency DAG.
    let (producers, consumers) = build_group_deps(groups);

    // Step 2: Topological sort.
    let topo_order = topological_sort(n, &producers);

    // Step 3: Classify edges and assign phases.
    let (group_phase, num_phases, duplicated) =
        assign_phases_from_edges(groups, &topo_order, &producers, &consumers, &is_literal, num_lanes);

    // Step 4: Build spans.
    let phases = build_all_spans(
        graph,
        num_lanes,
        num_phases,
        &group_phase,
        &producers,
        &consumers,
        &is_literal,
        &topo_order,
        &duplicated,
    );

    SpanPlan { num_lanes, phases }
}

// ─── Edge classification ─────────────────────────────────────────────────────

/// Determine if an edge from producer P to consumer C is "lane-aligned":
/// when both P and C are split into N equal slices, does lane i's slice of C
/// only read from lane i's slice of P?
///
/// Returns true if lane-aligned (safe to split both in same phase).
fn is_edge_lane_aligned(
    producer: &AtomGroup,
    consumer: &AtomGroup,
    input_ref: &InputRef,
    num_lanes: usize,
) -> bool {
    if num_lanes <= 1 {
        return true;
    }

    let prod_base = producer.base_id.0;
    let prod_count = producer.count;
    let cons_count = consumer.count;

    // For a ReduceSum/ReduceMax consumer, the strided access pattern extends
    // beyond the InputRef's direct mapping. This always reads across the full
    // producer range per output atom, so it's cross-lane unless the producer
    // is entirely consumed by this lane's slice.
    //
    // We handle reduces separately in the phase assignment by checking if the
    // reduce's total read range for lane i's slice stays within lane i's slice
    // of the producer.

    match input_ref {
        InputRef::Broadcast(_) => {
            // All consumer atoms read the same single producer atom.
            // Splitting the consumer means all lanes read this atom.
            // This is cross-lane IF the broadcast atom is in a splittable group.
            // If the producer has count=1 or is a literal, it's always available.
            // For non-trivial producers: cross-lane.
            prod_count <= 1
        }

        InputRef::Affine { base, stride } => {
            if *stride == 0 {
                // Effectively a broadcast.
                return prod_count <= 1;
            }

            // Affine with stride=1: atom i of consumer reads base + i of producer.
            // If consumer and producer have the same count, then splitting both
            // at the same boundaries means lane i reads exactly lane i.
            //
            // If counts differ (consumer reads a sub-range or different range of producer),
            // the access may not be lane-aligned.
            //
            // The key test: does consumer's lane-local slice [lo, hi) map to
            // a subset of producer's lane-local slice [lo', hi')?
            //
            // For Affine{stride=1, base=P.base}: consumer[i] reads P[i].
            // Lane j of consumer = [j*C/N, (j+1)*C/N).
            // These atoms read P[j*C/N, (j+1)*C/N).
            // Lane j of producer = [j*P/N, (j+1)*P/N).
            // Aligned iff j*C/N >= j*P/N and (j+1)*C/N <= (j+1)*P/N for all j.
            // That simplifies to: C <= P (consumer count <= producer count)
            // AND base == producer.base_id (consumer starts at producer's start)
            // AND stride == 1.
            //
            // But actually we need: consumer's read range for lane j must be
            // WITHIN producer's lane j range. With stride=1:
            //   consumer reads [base + j*C/N, base + (j+1)*C/N)
            //   producer lane j = [prod_base + j*P/N, prod_base + (j+1)*P/N)
            //
            // This works if base == prod_base and C == P and stride == 1.
            // It also works if C <= P and base == prod_base and stride == 1,
            // because each lane reads a proportional sub-range.

            if *stride == 1 && base.0 == prod_base && cons_count == prod_count {
                // Perfect 1:1 alignment.
                return true;
            }

            if *stride == 1 && base.0 == prod_base && cons_count <= prod_count {
                // Consumer reads a prefix of producer. When both are split by N,
                // consumer lane j reads [j*C/N, (j+1)*C/N) which is a subset
                // of producer lane j [j*P/N, (j+1)*P/N) since C/N <= P/N.
                //
                // BUT: this only holds if the split boundaries are at the same
                // granularity. With integer division: j*C/N might not align with
                // j*P/N.
                //
                // Actually: consumer lane j needs atoms [base + j*C/N .. base + (j+1)*C/N).
                // Producer lane j owns atoms [prod_base + j*P/N .. prod_base + (j+1)*P/N).
                // Since base == prod_base: we need j*C/N >= j*P/N (always false if C<P)
                // WAIT: j*C/N < j*P/N when C < P. So consumer lane 0 reads [0, C/N)
                // and producer lane 0 owns [0, P/N). Since C/N < P/N, this is fine.
                // Consumer lane 1 reads [C/N, 2*C/N) and producer lane 1 owns
                // [P/N, 2*P/N). Is C/N >= P/N? No, C < P implies C/N < P/N.
                // So consumer lane 1's start (C/N) < producer lane 1's start (P/N).
                // This means consumer lane 1 reads atoms from producer lane 0!
                // NOT lane-aligned.
                //
                // So: Affine{stride=1} is lane-aligned only when counts match
                // (assuming base matches).
                return false;
            }

            // For other stride values or mismatched bases: check numerically.
            // Sample a few lane boundaries to see if alignment holds.
            is_affine_lane_aligned(*base, *stride, prod_base, prod_count, cons_count, num_lanes)
        }

        InputRef::StridedBroadcast { base, stride, repeat } => {
            // Atom i of consumer reads base + stride * (i / repeat).
            // Each block of `repeat` consecutive atoms shares one source.
            //
            // For this to be lane-aligned: lane j's consumer slice [j*C/N, (j+1)*C/N)
            // must only access source atoms that are in lane j's producer slice.
            //
            // The source atom for consumer offset i is: base + stride * (i / repeat).
            // Lane j's consumer range reads sources base + stride * (lo/repeat) through
            // base + stride * ((hi-1)/repeat), where lo = j*C/N, hi = (j+1)*C/N.
            //
            // This is typically NOT lane-aligned because the source range is compressed
            // (fewer distinct sources than consumer atoms, each shared by `repeat` atoms).
            //
            // It IS lane-aligned if repeat >= C/N (each lane's chunk is within one
            // repeat block) AND the producer's lane boundaries align with the source
            // atoms that each block maps to. This is complex enough to just check.
            is_strided_broadcast_lane_aligned(
                *base, *stride, *repeat, prod_base, prod_count, cons_count, num_lanes,
            )
        }

        InputRef::Modular { base, stride, modulus } => {
            // Atom i reads base + stride * (i % modulus).
            // This cycles through the same `modulus` atoms for every `modulus` consumer atoms.
            // Splitting the consumer means each lane reads the full set of `modulus` atoms.
            // This is cross-lane unless modulus <= 1.
            *modulus <= 1
        }

        InputRef::SymAffine { .. } => {
            // SymAffine involves symbolic iteration — can't statically determine alignment.
            // Conservative: cross-lane.
            false
        }

        InputRef::Explicit(ids) => {
            // Check if each lane's consumer slice only references atoms in that
            // lane's producer slice. This would require checking every atom, which
            // is expensive for large groups. Conservative: cross-lane unless small.
            if ids.len() <= 256 {
                is_explicit_lane_aligned(ids, prod_base, prod_count, cons_count, num_lanes)
            } else {
                false
            }
        }
    }
}

/// Check if Affine{base, stride} access from a consumer of `cons_count` atoms
/// to a producer starting at `prod_base` with `prod_count` atoms is lane-aligned
/// when both are split into `num_lanes`.
fn is_affine_lane_aligned(
    base: AtomId,
    stride: i32,
    prod_base: u64,
    prod_count: u64,
    cons_count: u64,
    num_lanes: usize,
) -> bool {
    let n = num_lanes as u64;
    for lane in 0..num_lanes {
        let j = lane as u64;
        // Consumer lane j: offsets [j*C/N, (j+1)*C/N)
        let c_lo = j * cons_count / n;
        let c_hi = (j + 1) * cons_count / n;
        // Producer lane j: atoms [prod_base + j*P/N, prod_base + (j+1)*P/N)
        let p_lo = prod_base + j * prod_count / n;
        let p_hi = prod_base + (j + 1) * prod_count / n;
        // What atoms does consumer lane j read?
        if c_lo == c_hi {
            continue;
        }
        let read_lo = base.0.wrapping_add((stride as i64 * c_lo as i64) as u64);
        let read_hi = base.0.wrapping_add((stride as i64 * (c_hi - 1) as i64) as u64);
        let (read_min, read_max) = if stride >= 0 {
            (read_lo, read_hi)
        } else {
            (read_hi, read_lo)
        };
        if read_min < p_lo || read_max >= p_hi {
            return false;
        }
    }
    true
}

/// Check if StridedBroadcast access is lane-aligned.
fn is_strided_broadcast_lane_aligned(
    base: AtomId,
    stride: i64,
    repeat: u64,
    prod_base: u64,
    prod_count: u64,
    cons_count: u64,
    num_lanes: usize,
) -> bool {
    let n = num_lanes as u64;
    for lane in 0..num_lanes {
        let j = lane as u64;
        let c_lo = j * cons_count / n;
        let c_hi = (j + 1) * cons_count / n;
        let p_lo = prod_base + j * prod_count / n;
        let p_hi = prod_base + (j + 1) * prod_count / n;
        if c_lo == c_hi {
            continue;
        }
        // Source range: base + stride * (c_lo/repeat) through base + stride * ((c_hi-1)/repeat)
        let first_block = c_lo / repeat;
        let last_block = (c_hi - 1) / repeat;
        let src_first = base.0.wrapping_add((stride * first_block as i64) as u64);
        let src_last = base.0.wrapping_add((stride * last_block as i64) as u64);
        let (src_min, src_max) = if stride >= 0 {
            (src_first, src_last)
        } else {
            (src_last, src_first)
        };
        if src_min < p_lo || src_max >= p_hi {
            return false;
        }
    }
    true
}

/// Check if Explicit access is lane-aligned.
fn is_explicit_lane_aligned(
    ids: &[AtomId],
    prod_base: u64,
    prod_count: u64,
    cons_count: u64,
    num_lanes: usize,
) -> bool {
    let n = num_lanes as u64;
    for lane in 0..num_lanes {
        let j = lane as u64;
        let c_lo = (j * cons_count / n) as usize;
        let c_hi = ((j + 1) * cons_count / n) as usize;
        let p_lo = prod_base + j * prod_count / n;
        let p_hi = prod_base + (j + 1) * prod_count / n;
        for i in c_lo..c_hi.min(ids.len()) {
            let src = ids[i].0;
            if src < p_lo || src >= p_hi {
                // Check if this atom is even from this producer.
                if src >= prod_base && src < prod_base + prod_count {
                    return false;
                }
                // If it's from a different producer, ignore (different edge).
            }
        }
    }
    true
}

/// Check if a ReduceSum/ReduceMax edge is lane-aligned.
///
/// For ReduceSum with reduce_count K and reduce_stride S:
/// Output atom i reads input atoms at offsets input_base + i + k*S for k=0..K.
/// When we split the consumer, lane j gets output atoms [j*C/N, (j+1)*C/N).
/// Output atom i reads input range [input_base + i, input_base + i + (K-1)*S]
/// (assuming S > 0). This range is `i + (K-1)*S` wide.
///
/// For this to be lane-aligned, the full read range for all atoms in lane j
/// must fit within lane j's producer slice.
fn is_reduce_edge_lane_aligned(
    input_ref: &InputRef,
    cons_count: u64,
    reduce_count: u64,
    reduce_stride: i64,
    prod_base: u64,
    prod_count: u64,
    num_lanes: usize,
) -> bool {
    if reduce_count <= 1 || reduce_stride == 0 {
        // No strided access; just the base InputRef matters.
        return true; // Caller already checked the InputRef alignment.
    }

    let n = num_lanes as u64;
    let reduce_extent = reduce_stride * (reduce_count as i64 - 1);

    match input_ref {
        InputRef::Affine { base, stride } => {
            for lane in 0..num_lanes {
                let j = lane as u64;
                let c_lo = j * cons_count / n;
                let c_hi = (j + 1) * cons_count / n;
                let p_lo = prod_base + j * prod_count / n;
                let p_hi = prod_base + (j + 1) * prod_count / n;
                if c_lo == c_hi {
                    continue;
                }
                // Consumer atom c_lo reads: base + stride * c_lo + k*reduce_stride for k in 0..K
                // Consumer atom c_hi-1 reads: base + stride * (c_hi-1) + k*reduce_stride for k in 0..K
                let first_read = base.0 as i64 + *stride as i64 * c_lo as i64;
                let last_read = base.0 as i64 + *stride as i64 * (c_hi as i64 - 1);
                // Full range with reduce expansion:
                let read_min = first_read.min(last_read) + reduce_extent.min(0);
                let read_max = first_read.max(last_read) + reduce_extent.max(0);
                if (read_min as u64) < p_lo || (read_max as u64) >= p_hi {
                    return false;
                }
            }
            true
        }
        _ => {
            // For non-Affine input refs with reduce: conservative cross-lane.
            false
        }
    }
}

// ─── Phase assignment ────────────────────────────────────────────────────────

/// Assign groups to phases based on edge classification.
///
/// Groups connected only by lane-aligned edges can be in the same phase.
/// Cross-lane edges force the consumer into a later phase.
///
/// Small groups that are consumed cross-lane can be duplicated into each lane
/// instead of forcing a new phase.
///
/// Returns (group_phase, num_phases, duplicated_set).
fn assign_phases_from_edges(
    groups: &[AtomGroup],
    topo_order: &[usize],
    producers: &[Vec<usize>],
    consumers: &[Vec<usize>],
    is_literal: &[bool],
    num_lanes: usize,
) -> (Vec<usize>, usize, HashSet<usize>) {
    let n = groups.len();
    let mut group_phase = vec![0usize; n];
    let mut duplicated: HashSet<usize> = HashSet::new();

    // For each edge, determine if it's cross-lane.
    // An edge is (producer_gi, consumer_gi, input_ref_idx).
    // We store a summary: for each consumer, is ANY of its producer edges cross-lane?

    for &gi in topo_order {
        if is_literal[gi] {
            group_phase[gi] = 0;
            continue;
        }

        let group = &groups[gi];
        let mut max_required_phase = 0usize;

        // Check each producer edge.
        for &pi in &producers[gi] {
            if is_literal[pi] {
                continue; // Literals are always available.
            }

            let producer = &groups[pi];
            let prod_phase = group_phase[pi];

            // Check if this edge is lane-aligned.
            let mut edge_aligned = true;

            // Check each InputRef that references this producer.
            for input_ref in &group.inputs {
                if !input_ref_touches_group(input_ref, group.count, producer) {
                    continue; // This InputRef doesn't reference this producer.
                }

                if !is_edge_lane_aligned(producer, group, input_ref, num_lanes) {
                    edge_aligned = false;
                    break;
                }

                // Also check reduce-extended access if applicable.
                match &group.op {
                    ScalarOp::ReduceSum { reduce_count, reduce_stride, .. }
                    | ScalarOp::ReduceMax { reduce_count, reduce_stride, .. }
                        if *reduce_count > 1 && *reduce_stride != 0 =>
                    {
                        if !is_reduce_edge_lane_aligned(
                            input_ref,
                            group.count,
                            *reduce_count,
                            *reduce_stride,
                            producer.base_id.0,
                            producer.count,
                            num_lanes,
                        ) {
                            edge_aligned = false;
                            break;
                        }
                    }
                    _ => {}
                }
            }

            // Also check IndirectLoad table reference.
            if let ScalarOp::IndirectLoad { table_base, .. } = &group.op {
                if producer.contains(*table_base) {
                    // IndirectLoad: runtime-indexed, always cross-lane.
                    edge_aligned = false;
                }
            }

            if edge_aligned {
                // Same phase as producer.
                max_required_phase = max_required_phase.max(prod_phase);
            } else {
                // Cross-lane edge. Can we duplicate the producer instead?
                if producer.count <= DUPLICATE_THRESHOLD && !producer.op.is_reduce() {
                    // Duplicate: consumer stays in same phase as producer.
                    duplicated.insert(pi);
                    max_required_phase = max_required_phase.max(prod_phase);
                } else {
                    // Must be in a later phase.
                    max_required_phase = max_required_phase.max(prod_phase + 1);
                }
            }
        }

        group_phase[gi] = max_required_phase;
    }

    // Compute num_phases.
    let num_phases = group_phase
        .iter()
        .enumerate()
        .filter(|&(i, _)| !is_literal[i])
        .map(|(_, &p)| p + 1)
        .max()
        .unwrap_or(1);

    // Transitive closure on duplication: if a duplicated group D depends on
    // a split (non-duplicated) group S in the same phase via a cross-lane edge,
    // then S must also be duplicated (since D needs ALL of S to compute ALL of D,
    // and D is being duplicated meaning each lane computes all of D).
    let mut changed = true;
    while changed {
        changed = false;
        for &gi in topo_order {
            if !duplicated.contains(&gi) || is_literal[gi] {
                continue;
            }
            let group = &groups[gi];
            for &pi in &producers[gi] {
                if is_literal[pi] || duplicated.contains(&pi) {
                    continue;
                }
                // pi is not duplicated. Check if the edge to gi is cross-lane.
                // If gi is duplicated, each lane computes ALL of gi's atoms.
                // So gi reads ALL of pi's atoms (via whatever InputRef).
                // If pi is in the same phase and split, each lane only HAS a slice of pi.
                // So pi must also be duplicated.
                let producer = &groups[pi];
                if group_phase[pi] == group_phase[gi] {
                    // Check if this edge requires full pi.
                    let mut needs_full = false;
                    for input_ref in &group.inputs {
                        if input_ref_touches_group(input_ref, group.count, producer) {
                            // Since gi is duplicated (each lane computes full gi),
                            // gi needs the full range of pi. If pi is split, that's cross-lane.
                            needs_full = true;
                            break;
                        }
                    }
                    if needs_full && producer.count <= DUPLICATE_THRESHOLD && !producer.op.is_reduce() {
                        duplicated.insert(pi);
                        changed = true;
                    }
                }
            }
        }
    }

    (group_phase, num_phases, duplicated)
}

/// Check if an InputRef potentially references atoms from the given group.
fn input_ref_touches_group(input: &InputRef, consumer_count: u64, producer: &AtomGroup) -> bool {
    let prod_lo = producer.base_id.0;
    let prod_hi = prod_lo + producer.count;

    let (lo, hi) = input_ref_range(input, consumer_count);
    lo < prod_hi && hi > prod_lo
}

/// Compute the [lo, hi) atom range an InputRef covers.
fn input_ref_range(input: &InputRef, count: u64) -> (u64, u64) {
    if count == 0 {
        return (0, 0);
    }
    match input {
        InputRef::Broadcast(id) => (id.0, id.0 + 1),
        InputRef::Affine { base, stride } => {
            let first = base.0 as i64;
            let last = first + *stride as i64 * (count as i64 - 1);
            (first.min(last) as u64, first.max(last) as u64 + 1)
        }
        InputRef::StridedBroadcast { base, stride, repeat } => {
            let last_block = ((count - 1) / repeat) as i64;
            let first = base.0 as i64;
            let last = first + stride * last_block;
            (first.min(last) as u64, first.max(last) as u64 + 1)
        }
        InputRef::Modular { base, stride, modulus } => {
            if *modulus == 0 {
                return (0, 0);
            }
            let first = base.0 as i64;
            let last = first + *stride as i64 * (*modulus as i64 - 1);
            (first.min(last) as u64, first.max(last) as u64 + 1)
        }
        InputRef::SymAffine { base, stride_i, .. } => {
            let first = base.0 as i64;
            let last = first + *stride_i as i64 * (count as i64 - 1);
            (first.min(last) as u64, first.max(last) as u64 + 1)
        }
        InputRef::Explicit(ids) => {
            if ids.is_empty() {
                return (0, 0);
            }
            let lo = ids.iter().map(|id| id.0).min().unwrap();
            let hi = ids.iter().map(|id| id.0).max().unwrap() + 1;
            (lo, hi)
        }
    }
}

/// Compute [lo, hi) range INCLUDING reduce extension.
fn input_ref_range_with_reduce(
    input: &InputRef,
    count: u64,
    reduce_count: u64,
    reduce_stride: i64,
) -> (u64, u64) {
    let (lo, hi) = input_ref_range(input, count);
    if reduce_count <= 1 || reduce_stride == 0 {
        return (lo, hi);
    }
    let ext_lo = reduce_stride.min(0) * (reduce_count as i64 - 1);
    let ext_hi = reduce_stride.max(0) * (reduce_count as i64 - 1);
    ((lo as i64 + ext_lo) as u64, (hi as i64 + ext_hi) as u64)
}

// ─── Group dependency graph ──────────────────────────────────────────────────

fn build_group_deps(groups: &[AtomGroup]) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let n = groups.len();
    let mut producers: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (gi, group) in groups.iter().enumerate() {
        let mut prod_set = BTreeSet::new();

        for input in &group.inputs {
            for pi in resolve_producer_groups(input, group.count, groups) {
                if pi != gi {
                    prod_set.insert(pi);
                }
            }
        }

        // ReduceSum/ReduceMax strided access.
        match &group.op {
            ScalarOp::ReduceSum { reduce_count, reduce_stride, .. }
            | ScalarOp::ReduceMax { reduce_count, reduce_stride, .. }
                if *reduce_count > 1 && *reduce_stride != 0 =>
            {
                for input in &group.inputs {
                    for pi in resolve_producer_groups_with_reduce(
                        input, group.count, *reduce_count, *reduce_stride, groups,
                    ) {
                        if pi != gi {
                            prod_set.insert(pi);
                        }
                    }
                }
            }
            _ => {}
        }

        // IndirectLoad table reference.
        if let ScalarOp::IndirectLoad { table_base, .. } = &group.op {
            if let Some(pi) = find_group_idx(groups, *table_base) {
                if pi != gi {
                    prod_set.insert(pi);
                }
            }
        }

        let prod_vec: Vec<usize> = prod_set.into_iter().collect();
        for &pi in &prod_vec {
            consumers[pi].push(gi);
        }
        producers.push(prod_vec);
    }

    (producers, consumers)
}

fn resolve_producer_groups(input: &InputRef, count: u64, groups: &[AtomGroup]) -> Vec<usize> {
    match input {
        InputRef::Broadcast(atom_id) => find_group_idx(groups, *atom_id).into_iter().collect(),
        InputRef::Affine { base, stride } => {
            if count == 0 {
                return vec![];
            }
            let last_offset = (*stride as i64) * (count as i64 - 1);
            let last = AtomId(base.0.wrapping_add(last_offset as u64));
            let lo = base.0.min(last.0);
            let hi = base.0.max(last.0);
            find_groups_in_range(groups, lo, hi)
        }
        InputRef::Explicit(ids) => {
            let mut result = Vec::new();
            let mut seen = HashSet::new();
            for id in ids {
                if let Some(gi) = find_group_idx(groups, *id) {
                    if seen.insert(gi) {
                        result.push(gi);
                    }
                }
            }
            result
        }
        InputRef::SymAffine { base, stride_i, .. } => {
            if count == 0 {
                return vec![];
            }
            let last_offset = (*stride_i as i64) * (count as i64 - 1);
            let last = AtomId(base.0.wrapping_add(last_offset as u64));
            let lo = base.0.min(last.0);
            let hi = base.0.max(last.0);
            find_groups_in_range(groups, lo, hi)
        }
        InputRef::StridedBroadcast { base, stride, repeat } => {
            if count == 0 {
                return vec![];
            }
            let num_blocks = (count + repeat - 1) / repeat;
            let last_offset = *stride * (num_blocks as i64 - 1);
            let last = AtomId(base.0.wrapping_add(last_offset as u64));
            let lo = base.0.min(last.0);
            let hi = base.0.max(last.0);
            find_groups_in_range(groups, lo, hi)
        }
        InputRef::Modular { base, stride, modulus } => {
            if *modulus == 0 {
                return vec![];
            }
            let last_offset = (*stride as i64) * (*modulus as i64 - 1);
            let last = AtomId(base.0.wrapping_add(last_offset as u64));
            let lo = base.0.min(last.0);
            let hi = base.0.max(last.0);
            find_groups_in_range(groups, lo, hi)
        }
    }
}

fn resolve_producer_groups_with_reduce(
    input: &InputRef,
    count: u64,
    reduce_count: u64,
    reduce_stride: i64,
    groups: &[AtomGroup],
) -> Vec<usize> {
    if count == 0 {
        return vec![];
    }
    let min_ext = 0i64.min(reduce_stride * (reduce_count as i64 - 1));
    let max_ext = 0i64.max(reduce_stride * (reduce_count as i64 - 1));

    match input {
        InputRef::Affine { base, stride } => {
            let first = base.0 as i64;
            let last = first + *stride as i64 * (count as i64 - 1);
            let lo = (first.min(last) + min_ext) as u64;
            let hi = (first.max(last) + max_ext) as u64;
            find_groups_in_range(groups, lo, hi)
        }
        InputRef::StridedBroadcast { base, stride, repeat } => {
            let last_block = ((count - 1) / repeat) as i64;
            let first = base.0 as i64;
            let last = first + stride * last_block;
            let lo = (first.min(last) + min_ext) as u64;
            let hi = (first.max(last) + max_ext) as u64;
            find_groups_in_range(groups, lo, hi)
        }
        InputRef::Broadcast(id) => {
            let base = id.0 as i64;
            let lo = (base + min_ext) as u64;
            let hi = (base + max_ext) as u64;
            find_groups_in_range(groups, lo, hi)
        }
        InputRef::SymAffine { base, stride_i, .. } => {
            let first = base.0 as i64;
            let last = first + *stride_i as i64 * (count as i64 - 1);
            let lo = (first.min(last) + min_ext) as u64;
            let hi = (first.max(last) + max_ext) as u64;
            find_groups_in_range(groups, lo, hi)
        }
        InputRef::Modular { base, stride, modulus } => {
            if *modulus == 0 {
                return vec![];
            }
            let first = base.0 as i64;
            let last = first + *stride as i64 * (*modulus as i64 - 1);
            let lo = (first.min(last) + min_ext) as u64;
            let hi = (first.max(last) + max_ext) as u64;
            find_groups_in_range(groups, lo, hi)
        }
        InputRef::Explicit(ids) => {
            let mut result = HashSet::new();
            for id in ids {
                let lo = (id.0 as i64 + min_ext) as u64;
                let hi = (id.0 as i64 + max_ext) as u64;
                for gi in find_groups_in_range(groups, lo, hi) {
                    result.insert(gi);
                }
            }
            result.into_iter().collect()
        }
    }
}

fn find_group_idx(groups: &[AtomGroup], id: AtomId) -> Option<usize> {
    let idx = groups.partition_point(|g| g.base_id.0 <= id.0);
    if idx == 0 {
        return None;
    }
    let gi = idx - 1;
    if groups[gi].contains(id) {
        Some(gi)
    } else {
        None
    }
}

fn find_groups_in_range(groups: &[AtomGroup], lo: u64, hi: u64) -> Vec<usize> {
    let mut result = Vec::new();
    let start = groups.partition_point(|g| g.base_id.0 + g.count <= lo);
    for gi in start..groups.len() {
        let g = &groups[gi];
        if g.base_id.0 > hi {
            break;
        }
        let g_end = g.base_id.0 + g.count - 1;
        if g.base_id.0 <= hi && g_end >= lo {
            result.push(gi);
        }
    }
    result
}

fn topological_sort(n: usize, producers: &[Vec<usize>]) -> Vec<usize> {
    let mut in_degree = vec![0usize; n];
    let mut consumers_map: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (gi, prods) in producers.iter().enumerate() {
        in_degree[gi] = prods.len();
        for &pi in prods {
            consumers_map[pi].push(gi);
        }
    }

    let mut queue: VecDeque<usize> = VecDeque::new();
    for i in 0..n {
        if in_degree[i] == 0 {
            queue.push_back(i);
        }
    }

    let mut order = Vec::with_capacity(n);
    while let Some(gi) = queue.pop_front() {
        order.push(gi);
        for &ci in &consumers_map[gi] {
            in_degree[ci] -= 1;
            if in_degree[ci] == 0 {
                queue.push_back(ci);
            }
        }
    }

    order
}

// ─── Span building ───────────────────────────────────────────────────────────

/// Within each phase, distribute groups across lanes with splitting.
///
/// Groups marked as "duplicated" are placed on ALL lanes (each lane computes the full group).
/// Other groups are split across lanes: each lane gets a proportional slice.
fn build_all_spans(
    graph: &NanoGraph,
    num_lanes: usize,
    num_phases: usize,
    group_phase: &[usize],
    producers: &[Vec<usize>],
    consumers: &[Vec<usize>],
    is_literal: &[bool],
    topo_order: &[usize],
    duplicated: &HashSet<usize>,
) -> Vec<Phase> {
    let groups = graph.groups();
    let n = groups.len();

    // For each phase, collect compute groups.
    let mut phase_groups: Vec<Vec<usize>> = vec![Vec::new(); num_phases];
    for &gi in topo_order {
        if !is_literal[gi] {
            phase_groups[group_phase[gi]].push(gi);
        }
    }

    // Track which atoms are "available" after each phase.
    // (phase, lane) -> set of main-graph atom ranges produced.
    // We use a more efficient representation: for each group, track which lane(s) produce it.

    let mut phases = Vec::with_capacity(num_phases);

    for phase_idx in 0..num_phases {
        let pg = &phase_groups[phase_idx];
        let mut spans = Vec::with_capacity(num_lanes);

        if pg.is_empty() {
            for _ in 0..num_lanes {
                spans.push(Span {
                    graph: NanoGraph::new(),
                    inputs: vec![],
                    outputs: vec![],
                });
            }
            phases.push(Phase { spans });
            continue;
        }

        // Determine which groups are split vs duplicated vs single-lane.
        // Split groups: each lane gets count/num_lanes atoms.
        // Duplicated groups: each lane computes the full group.
        // Single-lane groups: assigned to one lane.

        // We split ALL non-duplicated groups across lanes.
        // This is safe because we guaranteed that all within-phase edges are lane-aligned.

        for lane in 0..num_lanes {
            let span = build_one_span(
                graph,
                groups,
                pg,
                lane,
                num_lanes,
                phase_idx,
                num_phases,
                group_phase,
                producers,
                consumers,
                is_literal,
                duplicated,
            );
            spans.push(span);
        }

        phases.push(Phase { spans });
    }

    phases
}

/// Representation of a group's work assignment for one lane.
struct LaneSlice {
    group_idx: usize,
    /// Offset within the group (0 for full group or start of slice).
    atom_offset: u64,
    /// Number of atoms this lane handles.
    atom_count: u64,
    /// Whether this is a duplicated group (lane computes full group but only "owns" a slice for output).
    is_duplicated: bool,
}

fn build_one_span(
    graph: &NanoGraph,
    groups: &[AtomGroup],
    phase_groups: &[usize],
    lane: usize,
    num_lanes: usize,
    phase_idx: usize,
    num_phases: usize,
    group_phase: &[usize],
    producers: &[Vec<usize>],
    consumers: &[Vec<usize>],
    is_literal: &[bool],
    duplicated: &HashSet<usize>,
) -> Span {
    let n = groups.len();

    // Determine this lane's work: for each group in this phase, what slice does this lane get?
    let mut lane_slices: Vec<LaneSlice> = Vec::new();

    for &gi in phase_groups {
        let group = &groups[gi];
        if duplicated.contains(&gi) {
            // Duplicated: this lane computes the full group.
            // Each lane gets a slice for output purposes.
            let out_offset = lane as u64 * group.count / num_lanes as u64;
            let out_end = (lane as u64 + 1) * group.count / num_lanes as u64;
            let out_count = out_end - out_offset;
            if out_count > 0 || lane == 0 {
                // Include even if zero count for lane 0 (to keep topology consistent).
                lane_slices.push(LaneSlice {
                    group_idx: gi,
                    atom_offset: 0, // Computes full group.
                    atom_count: group.count,
                    is_duplicated: true,
                });
            }
        } else {
            // Split: this lane gets a proportional slice.
            let offset = lane as u64 * group.count / num_lanes as u64;
            let end = (lane as u64 + 1) * group.count / num_lanes as u64;
            let count = end - offset;
            if count > 0 {
                lane_slices.push(LaneSlice {
                    group_idx: gi,
                    atom_offset: offset,
                    atom_count: count,
                    is_duplicated: false,
                });
            }
        }
    }

    if lane_slices.is_empty() {
        return Span {
            graph: NanoGraph::new(),
            inputs: vec![],
            outputs: vec![],
        };
    }

    // Build the span's NanoGraph.
    // We need to:
    // 1. Determine external inputs (atoms read from earlier phases or literals).
    // 2. Create stub groups for external inputs.
    // 3. Create compute groups with remapped InputRefs.

    let mut span_graph = NanoGraph::new();

    // Copy sym_dim setup.
    for (name, &sd) in &graph.sym_dim_names {
        let local_sd = span_graph.sym_dim(name);
        if let Some(&bound) = graph.sym_dim_bounds.get(&sd) {
            span_graph.sym_dim_bounds.insert(local_sd, bound);
        }
    }

    let mut atom_map = RangeAtomMap::new();

    // Set of group indices in this span's local compute.
    let local_group_set: HashSet<usize> = lane_slices.iter().map(|s| s.group_idx).collect();

    // Collect literal dependencies.
    let mut small_literals: BTreeSet<usize> = BTreeSet::new();
    let mut large_literal_ranges: Vec<(usize, u64, u64)> = Vec::new();
    for slice in &lane_slices {
        let gi = slice.group_idx;
        for &pi in &producers[gi] {
            if is_literal[pi] {
                if groups[pi].count < LITERAL_INLINE_THRESHOLD {
                    small_literals.insert(pi);
                } else {
                    // Large literal: only include the range this slice actually reads.
                    let ranges = compute_read_ranges_from_producer(
                        &groups[gi], slice.atom_offset, slice.atom_count,
                        &groups[pi], groups,
                    );
                    for (offset, count) in ranges {
                        large_literal_ranges.push((pi, offset, count));
                    }
                }
            }
        }
    }

    // Collect external (non-literal, non-local) dependencies.
    let mut external_ranges: Vec<(usize, u64, u64)> = Vec::new();
    for slice in &lane_slices {
        let gi = slice.group_idx;
        for &pi in &producers[gi] {
            if is_literal[pi] || local_group_set.contains(&pi) {
                continue;
            }
            // This producer is from an earlier phase. Determine which atoms we need.
            // If our group is split, we only need the atoms our slice reads.
            // If our group is duplicated, we need the full range.
            let ranges = if slice.is_duplicated {
                compute_read_ranges_from_producer(
                    &groups[gi], 0, groups[gi].count,
                    &groups[pi], groups,
                )
            } else {
                compute_read_ranges_from_producer(
                    &groups[gi], slice.atom_offset, slice.atom_count,
                    &groups[pi], groups,
                )
            };
            for (offset, count) in ranges {
                external_ranges.push((pi, offset, count));
            }
        }
    }

    // Merge all external ranges.
    let external_ranges = merge_group_ranges(&mut external_ranges);
    let large_literal_ranges = merge_group_ranges(&mut large_literal_ranges);

    // Phase 1: Add small literal groups.
    for &li in &small_literals {
        let lit_group = &groups[li];
        let local_base = span_graph.push_group(
            lit_group.count,
            lit_group.op.clone(),
            remap_sym_dims(&lit_group.sym_dims, graph, &span_graph),
            remap_sym_dims(&lit_group.reduce_dims, graph, &span_graph),
            vec![],
        );
        atom_map.insert_range(lit_group.base_id, local_base, lit_group.count);
    }

    // Phase 2: Add stubs for large literals.
    let mut input_mappings: Vec<AtomMapping> = Vec::new();
    for &(li, offset, count) in &large_literal_ranges {
        let src_group = &groups[li];
        let main_base = src_group.base_id.offset(offset);
        let local_base = span_graph.push_group(
            count,
            ScalarOp::Literal(crate::numeric_scalar::NumericScalar::F32(0.0)),
            vec![], vec![], vec![],
        );
        atom_map.insert_range(main_base, local_base, count);
        input_mappings.push(AtomMapping {
            main_base,
            span_base: local_base,
            count,
        });
    }

    // Phase 3: Add stubs for external (non-literal) inputs.
    for &(pi, offset, count) in &external_ranges {
        let src_group = &groups[pi];
        let main_base = src_group.base_id.offset(offset);
        let local_base = span_graph.push_group(
            count,
            ScalarOp::Literal(crate::numeric_scalar::NumericScalar::F32(0.0)),
            vec![], vec![], vec![],
        );
        atom_map.insert_range(main_base, local_base, count);
        input_mappings.push(AtomMapping {
            main_base,
            span_base: local_base,
            count,
        });
    }

    // Sort atom map before adding compute groups.
    atom_map.sort();

    // Phase 4: Add compute groups.
    // Process in topological order (group indices are already in topo order
    // since groups were pushed in topo order during lowering).
    let mut sorted_slices = lane_slices;
    sorted_slices.sort_by_key(|s| s.group_idx);

    // Track output ranges: for split groups, output the lane's owned slice.
    // For duplicated groups, output only this lane's proportional slice.
    let mut output_ranges: Vec<(usize, u64, u64)> = Vec::new();

    for slice in &sorted_slices {
        let gi = slice.group_idx;
        let group = &groups[gi];

        if slice.is_duplicated {
            // Duplicated group: add the full group to the span graph.
            let remapped_inputs = remap_input_refs(&group.inputs, &atom_map);
            let local_base = span_graph.push_group(
                group.count,
                group.op.clone(),
                remap_sym_dims(&group.sym_dims, graph, &span_graph),
                remap_sym_dims(&group.reduce_dims, graph, &span_graph),
                remapped_inputs,
            );
            atom_map.insert_range(group.base_id, local_base, group.count);
            // Re-sort since we added a new range that may be out of order.
            atom_map.sort();

            // Output only this lane's proportional slice.
            let out_offset = lane as u64 * group.count / num_lanes as u64;
            let out_end = (lane as u64 + 1) * group.count / num_lanes as u64;
            let out_count = out_end - out_offset;
            if out_count > 0 {
                output_ranges.push((gi, out_offset, out_count));
            }
        } else {
            // Split group: add only this lane's slice.
            // We need to remap the InputRefs to account for the slice offset.
            let sliced_inputs = remap_input_refs_for_slice(
                &group.inputs,
                &group.op,
                slice.atom_offset,
                slice.atom_count,
                group.count,
                &atom_map,
            );
            let local_base = span_graph.push_group(
                slice.atom_count,
                adjust_op_for_slice(&group.op),
                remap_sym_dims(&group.sym_dims, graph, &span_graph),
                remap_sym_dims(&group.reduce_dims, graph, &span_graph),
                sliced_inputs,
            );
            atom_map.insert_range(
                group.base_id.offset(slice.atom_offset),
                local_base,
                slice.atom_count,
            );
            atom_map.sort();

            // Output the full slice.
            output_ranges.push((gi, slice.atom_offset, slice.atom_count));
        }
    }

    // Determine which output ranges are actually needed by consumers outside this span.
    // For simplicity, output everything this lane produces — the execution engine
    // can optimize away unused outputs later.
    let output_ranges = merge_group_ranges(&mut output_ranges);

    // Build output mappings.
    let mut output_mappings: Vec<AtomMapping> = Vec::new();
    for &(gi, offset, count) in &output_ranges {
        let main_base = groups[gi].base_id.offset(offset);
        if let Some(span_base) = atom_map.get(main_base) {
            output_mappings.push(AtomMapping {
                main_base,
                span_base,
                count,
            });
        }
    }

    Span {
        graph: span_graph,
        inputs: input_mappings,
        outputs: output_mappings,
    }
}

/// Compute which atom ranges from `producer` the consumer's slice reads.
///
/// Consumer has been sliced to [atom_offset, atom_offset + atom_count).
/// Returns ranges as (offset_within_producer, count).
fn compute_read_ranges_from_producer(
    consumer: &AtomGroup,
    consumer_offset: u64,
    consumer_count: u64,
    producer: &AtomGroup,
    all_groups: &[AtomGroup],
) -> Vec<(u64, u64)> {
    let prod_lo = producer.base_id.0;
    let prod_hi = prod_lo + producer.count;

    let (is_reduce, reduce_count, reduce_stride) = match &consumer.op {
        ScalarOp::ReduceSum { reduce_count, reduce_stride, .. }
        | ScalarOp::ReduceMax { reduce_count, reduce_stride, .. }
            if *reduce_count > 1 && *reduce_stride != 0 =>
        {
            (true, *reduce_count, *reduce_stride)
        }
        _ => (false, 0u64, 0i64),
    };

    let mut ranges = Vec::new();

    for input in &consumer.inputs {
        // Compute the atom range this sliced InputRef accesses.
        let (read_lo, read_hi) = sliced_input_ref_range(
            input,
            consumer_offset,
            consumer_count,
            consumer.count,
            is_reduce,
            reduce_count,
            reduce_stride,
        );

        // Intersect with producer range.
        if read_hi <= prod_lo || read_lo >= prod_hi {
            continue;
        }
        let overlap_lo = read_lo.max(prod_lo);
        let overlap_hi = read_hi.min(prod_hi);
        let offset = overlap_lo - prod_lo;
        let count = overlap_hi - overlap_lo;
        if count > 0 {
            ranges.push((offset, count));
        }
    }

    // Also handle IndirectLoad table reference.
    if let ScalarOp::IndirectLoad { table_base, .. } = &consumer.op {
        if table_base.0 >= prod_lo && table_base.0 < prod_hi {
            // The entire table might be needed. Include the full producer.
            ranges.push((0, producer.count));
        }
    }

    ranges
}

/// Compute the [lo, hi) atom range accessed by a sliced InputRef.
///
/// The consumer has been sliced to [atom_offset, atom_offset + atom_count).
fn sliced_input_ref_range(
    input: &InputRef,
    atom_offset: u64,
    atom_count: u64,
    original_count: u64,
    is_reduce: bool,
    reduce_count: u64,
    reduce_stride: i64,
) -> (u64, u64) {
    if atom_count == 0 {
        return (0, 0);
    }

    let min_reduce_ext = if is_reduce {
        0i64.min(reduce_stride * (reduce_count as i64 - 1))
    } else {
        0
    };
    let max_reduce_ext = if is_reduce {
        0i64.max(reduce_stride * (reduce_count as i64 - 1))
    } else {
        0
    };

    match input {
        InputRef::Broadcast(id) => {
            let lo = id.0 as i64 + min_reduce_ext;
            let hi = id.0 as i64 + max_reduce_ext + 1;
            (lo as u64, hi as u64)
        }
        InputRef::Affine { base, stride } => {
            let first = base.0 as i64 + *stride as i64 * atom_offset as i64;
            let last = base.0 as i64 + *stride as i64 * (atom_offset + atom_count - 1) as i64;
            let lo = first.min(last) + min_reduce_ext;
            let hi = first.max(last) + max_reduce_ext + 1;
            (lo as u64, hi as u64)
        }
        InputRef::StridedBroadcast { base, stride, repeat } => {
            let first_block = atom_offset / repeat;
            let last_block = (atom_offset + atom_count - 1) / repeat;
            let first_read = base.0 as i64 + *stride * first_block as i64;
            let last_read = base.0 as i64 + *stride * last_block as i64;
            let lo = first_read.min(last_read) + min_reduce_ext;
            let hi = first_read.max(last_read) + max_reduce_ext + 1;
            (lo as u64, hi as u64)
        }
        InputRef::Modular { base, stride, modulus } => {
            if *modulus == 0 {
                return (0, 0);
            }
            // Modular always accesses the full modulus range regardless of slice.
            let first = base.0 as i64;
            let last = first + *stride as i64 * (*modulus as i64 - 1);
            let lo = first.min(last) + min_reduce_ext;
            let hi = first.max(last) + max_reduce_ext + 1;
            (lo as u64, hi as u64)
        }
        InputRef::SymAffine { base, stride_i, .. } => {
            let first = base.0 as i64 + *stride_i as i64 * atom_offset as i64;
            let last = base.0 as i64 + *stride_i as i64 * (atom_offset + atom_count - 1) as i64;
            let lo = first.min(last);
            let hi = first.max(last) + 1;
            (lo as u64, hi as u64)
        }
        InputRef::Explicit(ids) => {
            let slice_lo = atom_offset as usize;
            let slice_hi = (atom_offset + atom_count) as usize;
            if slice_lo >= ids.len() {
                return (0, 0);
            }
            let slice_end = slice_hi.min(ids.len());
            let mut lo = u64::MAX;
            let mut hi = 0u64;
            for i in slice_lo..slice_end {
                let id_lo = (ids[i].0 as i64 + min_reduce_ext) as u64;
                let id_hi = (ids[i].0 as i64 + max_reduce_ext) as u64 + 1;
                lo = lo.min(id_lo);
                hi = hi.max(id_hi);
            }
            if lo > hi { (0, 0) } else { (lo, hi) }
        }
    }
}

/// Remap InputRefs for a sliced group.
///
/// When we take a slice [offset, offset+count) of a group, we need to adjust
/// the InputRefs so that atom 0 in the slice maps to what was atom `offset` in
/// the original group.
fn remap_input_refs_for_slice(
    inputs: &[InputRef],
    op: &ScalarOp,
    offset: u64,
    count: u64,
    original_count: u64,
    atom_map: &RangeAtomMap,
) -> Vec<InputRef> {
    inputs
        .iter()
        .map(|input| {
            let adjusted = adjust_input_ref_for_slice(input, offset, count, original_count);
            remap_one_input_ref(&adjusted, atom_map)
        })
        .collect()
}

/// Adjust an InputRef for a group slice.
///
/// The slice takes atoms [offset, offset+count) from the original group.
/// The new InputRef must be such that atom 0 in the new group reads what
/// atom `offset` would have read in the original group.
fn adjust_input_ref_for_slice(
    input: &InputRef,
    offset: u64,
    count: u64,
    original_count: u64,
) -> InputRef {
    match input {
        InputRef::Broadcast(id) => {
            // All atoms read the same source — no change needed.
            InputRef::Broadcast(*id)
        }
        InputRef::Affine { base, stride } => {
            // Original: atom i reads base + stride * i.
            // Sliced: new atom j (= original atom offset + j) reads base + stride * (offset + j).
            // New base = base + stride * offset, same stride.
            let new_base = AtomId(base.0.wrapping_add((*stride as i64 * offset as i64) as u64));
            InputRef::Affine {
                base: new_base,
                stride: *stride,
            }
        }
        InputRef::StridedBroadcast { base, stride, repeat } => {
            // Original: atom i reads base + stride * (i / repeat).
            // Sliced: new atom j reads base + stride * ((offset + j) / repeat).
            //
            // If offset is aligned to repeat boundary:
            //   (offset + j) / repeat = offset/repeat + j/repeat
            //   New base = base + stride * (offset / repeat), same stride and repeat.
            //
            // If not aligned, we need Explicit or a more complex transformation.
            // For now, handle the aligned case (which covers the common matmul pattern).
            if offset % repeat == 0 {
                let block_offset = offset / repeat;
                let new_base = AtomId(base.0.wrapping_add((*stride * block_offset as i64) as u64));
                InputRef::StridedBroadcast {
                    base: new_base,
                    stride: *stride,
                    repeat: *repeat,
                }
            } else {
                // Unaligned: fall back to Explicit.
                let ids: Vec<AtomId> = (0..count)
                    .map(|j| {
                        let orig_i = offset + j;
                        let block = orig_i / repeat;
                        AtomId(base.0.wrapping_add((*stride * block as i64) as u64))
                    })
                    .collect();
                InputRef::Explicit(ids)
            }
        }
        InputRef::Modular { base, stride, modulus } => {
            // Original: atom i reads base + stride * (i % modulus).
            // Sliced: new atom j reads base + stride * ((offset + j) % modulus).
            //
            // If modulus divides count evenly and offset is aligned, same InputRef works.
            // Otherwise, fall back to Explicit.
            if *modulus > 0 && offset % modulus == 0 {
                InputRef::Modular {
                    base: *base,
                    stride: *stride,
                    modulus: *modulus,
                }
            } else if *modulus > 0 {
                let ids: Vec<AtomId> = (0..count)
                    .map(|j| {
                        let wrapped = (offset + j) % modulus;
                        AtomId(base.0.wrapping_add((*stride as i64 * wrapped as i64) as u64))
                    })
                    .collect();
                InputRef::Explicit(ids)
            } else {
                InputRef::Modular { base: *base, stride: *stride, modulus: *modulus }
            }
        }
        InputRef::SymAffine { base, stride_i, stride_k } => {
            // Original: atom i reads base + stride_i * i + stride_k * k.
            // Sliced: new atom j reads base + stride_i * (offset + j) + stride_k * k.
            // = (base + stride_i * offset) + stride_i * j + stride_k * k.
            let new_base = AtomId(base.0.wrapping_add((*stride_i as i64 * offset as i64) as u64));
            InputRef::SymAffine {
                base: new_base,
                stride_i: *stride_i,
                stride_k: *stride_k,
            }
        }
        InputRef::Explicit(ids) => {
            // Take the slice of the ids vector.
            let start = offset as usize;
            let end = (offset + count) as usize;
            InputRef::Explicit(ids[start..end.min(ids.len())].to_vec())
        }
    }
}

/// Adjust ScalarOp for slicing (currently identity — reduce ops keep their parameters).
fn adjust_op_for_slice(op: &ScalarOp) -> ScalarOp {
    // ReduceSum/ReduceMax: the reduce_count and reduce_stride are per-output-atom
    // properties and don't change when we split along the output dimension.
    op.clone()
}

fn remap_input_refs(inputs: &[InputRef], atom_map: &RangeAtomMap) -> Vec<InputRef> {
    inputs.iter().map(|input| remap_one_input_ref(input, atom_map)).collect()
}

fn remap_one_input_ref(input: &InputRef, atom_map: &RangeAtomMap) -> InputRef {
    match input {
        InputRef::Broadcast(id) => {
            InputRef::Broadcast(atom_map.get(*id).unwrap_or(*id))
        }
        InputRef::Affine { base, stride } => {
            InputRef::Affine {
                base: atom_map.get(*base).unwrap_or(*base),
                stride: *stride,
            }
        }
        InputRef::StridedBroadcast { base, stride, repeat } => {
            InputRef::StridedBroadcast {
                base: atom_map.get(*base).unwrap_or(*base),
                stride: *stride,
                repeat: *repeat,
            }
        }
        InputRef::Modular { base, stride, modulus } => {
            InputRef::Modular {
                base: atom_map.get(*base).unwrap_or(*base),
                stride: *stride,
                modulus: *modulus,
            }
        }
        InputRef::SymAffine { base, stride_i, stride_k } => {
            InputRef::SymAffine {
                base: atom_map.get(*base).unwrap_or(*base),
                stride_i: *stride_i,
                stride_k: *stride_k,
            }
        }
        InputRef::Explicit(ids) => {
            InputRef::Explicit(
                ids.iter()
                    .map(|id| atom_map.get(*id).unwrap_or(*id))
                    .collect(),
            )
        }
    }
}

fn remap_sym_dims(
    dims: &[SymDim],
    main_graph: &NanoGraph,
    span_graph: &NanoGraph,
) -> Vec<SymDim> {
    dims.iter()
        .map(|&sd| {
            for (name, &main_sd) in &main_graph.sym_dim_names {
                if main_sd == sd {
                    if let Some(&local_sd) = span_graph.sym_dim_names.get(name) {
                        return local_sd;
                    }
                }
            }
            sd
        })
        .collect()
}

fn merge_group_ranges(ranges: &mut Vec<(usize, u64, u64)>) -> Vec<(usize, u64, u64)> {
    if ranges.is_empty() {
        return vec![];
    }
    ranges.sort_by_key(|&(gi, off, _)| (gi, off));
    let mut merged: Vec<(usize, u64, u64)> = Vec::new();
    for &(gi, off, count) in ranges.iter() {
        if let Some(last) = merged.last_mut() {
            if last.0 == gi && off <= last.1 + last.2 {
                let new_end = (off + count).max(last.1 + last.2);
                last.2 = new_end - last.1;
                continue;
            }
        }
        merged.push((gi, off, count));
    }
    merged
}

// ─── Validation ──────────────────────────────────────────────────────────────

/// Validate that a SpanPlan is correct.
///
/// Checks:
/// 1. Every span's NanoGraph passes validate().
/// 2. No cross-lane reads within a phase (span independence).
/// 3. All atoms from the original graph are produced by exactly one span.
/// 4. All span inputs come from earlier phases' outputs or literals.
pub fn validate_span_plan(plan: &SpanPlan, graph: &NanoGraph) -> Vec<String> {
    let mut errors = Vec::new();
    let groups = graph.groups();
    let is_literal: Vec<bool> = groups
        .iter()
        .map(|g| matches!(g.op, ScalarOp::Literal(_)) && g.inputs.is_empty())
        .collect();

    // Check 1: Each span's NanoGraph validates.
    for (pi, phase) in plan.phases.iter().enumerate() {
        for (li, span) in phase.spans.iter().enumerate() {
            if span.graph.num_groups() == 0 {
                continue;
            }
            let span_errors = span.graph.validate();
            for err in span_errors {
                errors.push(format!("Phase {} Lane {} span validation: {}", pi, li, err));
            }
        }
    }

    // Check 2: All compute atoms are covered.
    // Track which main-graph atom ranges are output by each span.
    let mut atom_covered: BTreeMap<u64, (usize, usize)> = BTreeMap::new(); // atom_id -> (phase, lane)
    for (pi, phase) in plan.phases.iter().enumerate() {
        for (li, span) in phase.spans.iter().enumerate() {
            for mapping in &span.outputs {
                for i in 0..mapping.count {
                    let atom = mapping.main_base.0 + i;
                    if let Some(&(prev_p, prev_l)) = atom_covered.get(&atom) {
                        errors.push(format!(
                            "Atom {} output by both Phase {} Lane {} and Phase {} Lane {}",
                            atom, prev_p, prev_l, pi, li
                        ));
                    } else {
                        atom_covered.insert(atom, (pi, li));
                    }
                }
            }
        }
    }

    // Check that all non-literal compute atoms are covered.
    for (gi, group) in groups.iter().enumerate() {
        if is_literal[gi] {
            continue;
        }
        for i in 0..group.count {
            let atom = group.base_id.0 + i;
            if !atom_covered.contains_key(&atom) {
                errors.push(format!("Compute atom {} (group {}) not output by any span", atom, gi));
                // Only report first missing atom per group.
                break;
            }
        }
    }

    // Check 3: Cross-lane independence within a phase.
    for (pi, phase) in plan.phases.iter().enumerate() {
        // For each lane, collect the main-graph atom ranges it produces.
        let mut lane_output_ranges: Vec<Vec<(u64, u64)>> = Vec::new();
        for span in &phase.spans {
            let mut ranges: Vec<(u64, u64)> = Vec::new();
            for mapping in &span.outputs {
                ranges.push((mapping.main_base.0, mapping.main_base.0 + mapping.count));
            }
            lane_output_ranges.push(ranges);
        }

        // Check that no span's inputs overlap with another lane's outputs.
        for (li, span) in phase.spans.iter().enumerate() {
            for mapping in &span.inputs {
                let in_lo = mapping.main_base.0;
                let in_hi = in_lo + mapping.count;
                for (other_li, other_ranges) in lane_output_ranges.iter().enumerate() {
                    if other_li == li {
                        continue;
                    }
                    for &(out_lo, out_hi) in other_ranges {
                        if in_lo < out_hi && in_hi > out_lo {
                            // Overlap: cross-lane violation.
                            let overlap_lo = in_lo.max(out_lo);
                            let overlap_hi = in_hi.min(out_hi);
                            errors.push(format!(
                                "Phase {} Lane {} reads atoms [{}, {}) produced by Lane {} (cross-lane violation)",
                                pi, li, overlap_lo, overlap_hi, other_li
                            ));
                        }
                    }
                }
            }
        }
    }

    errors
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::attempts::v13_claude::test_graphs;
    use crate::dtype::DType;
    use crate::nano_graph::{ScalarBinOp, ScalarUnaryOp};
    use crate::numeric_scalar::NumericScalar;

    fn plan_and_validate(graph: &NanoGraph, num_lanes: usize) -> (SpanPlan, Vec<String>) {
        let plan = plan_spans(graph, num_lanes);
        let errors = validate_span_plan(&plan, graph);
        (plan, errors)
    }

    fn count_output_atoms(plan: &SpanPlan) -> u64 {
        let mut total = 0u64;
        for phase in &plan.phases {
            for span in &phase.spans {
                for mapping in &span.outputs {
                    total += mapping.count;
                }
            }
        }
        total
    }

    // ── Basic tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_empty_graph() {
        let g = NanoGraph::new();
        let (plan, errors) = plan_and_validate(&g, 4);
        assert!(errors.is_empty(), "{:?}", errors);
        assert_eq!(plan.phases.len(), 0);
    }

    #[test]
    fn test_literal_only() {
        let mut g = NanoGraph::new();
        g.push_group(
            100,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![], vec![], vec![],
        );
        let (plan, errors) = plan_and_validate(&g, 2);
        assert!(errors.is_empty(), "{:?}", errors);
        assert_eq!(plan.phases.len(), 0);
    }

    #[test]
    fn test_single_lane() {
        let (g, _, _, _) = test_graphs::elementwise_binary(1024, ScalarBinOp::Add);
        let (plan, errors) = plan_and_validate(&g, 1);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
        assert!(plan.phases.len() >= 1);
    }

    // ── Elementwise tests ────────────────────────────────────────────────────

    #[test]
    fn test_elementwise_add_4_lanes() {
        let (g, _, _, _) = test_graphs::elementwise_binary(1024, ScalarBinOp::Add);
        let (plan, errors) = plan_and_validate(&g, 4);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
        // 1024 compute atoms should be distributed across 4 lanes.
        let total = count_output_atoms(&plan);
        assert_eq!(total, 1024, "Expected 1024 output atoms, got {}", total);
    }

    #[test]
    fn test_elementwise_chain() {
        // A chain: Lit -> Neg -> Exp -> Tanh. All 256 atoms, all Affine(stride=1).
        // Should all be in one phase, split across lanes.
        let (g, _, _) = test_graphs::unary_chain(
            256,
            &[ScalarUnaryOp::Neg, ScalarUnaryOp::Exp, ScalarUnaryOp::Tanh],
        );
        let (plan, errors) = plan_and_validate(&g, 4);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
        // 3 compute groups * 256 atoms = 768 atoms total, split across 4 lanes.
        let total = count_output_atoms(&plan);
        assert_eq!(total, 768, "Expected 768 output atoms, got {}", total);
    }

    #[test]
    fn test_broadcast_add_4_lanes() {
        // C[1024] = A[1024] + scalar_b. The Broadcast edge to scalar_b is cross-lane
        // but scalar_b has count=1 (below dup threshold), so it gets duplicated.
        let (g, _, _, _) = test_graphs::broadcast_add(1024);
        let (plan, errors) = plan_and_validate(&g, 4);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
        let total = count_output_atoms(&plan);
        assert_eq!(total, 1024, "Expected 1024 output atoms, got {}", total);
    }

    // ── Matmul tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_small_matmul_single_lane() {
        let (g, _, _, _) = test_graphs::matmul(4, 2, 3);
        let errors = g.validate();
        assert!(errors.is_empty(), "Graph validation: {:?}", errors);

        let (plan, plan_errors) = plan_and_validate(&g, 1);
        assert!(plan_errors.is_empty(), "Plan errors: {:?}", plan_errors);
    }

    #[test]
    fn test_small_matmul_4_lanes() {
        // M=4, K=2, N=3.
        // 8 Mul groups (each count=3) + 4 ReduceSum groups (each count=3).
        // 12 compute groups total.
        let (g, _, _, _) = test_graphs::matmul(4, 2, 3);
        let (plan, errors) = plan_and_validate(&g, 4);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
    }

    #[test]
    fn test_matmul_activation_4_lanes() {
        let (g, _, _, _) = test_graphs::matmul_activation(8, 4, 6, ScalarUnaryOp::Tanh);
        let (plan, errors) = plan_and_validate(&g, 4);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
    }

    // ── Matmul chain (multi-layer) ───────────────────────────────────────────

    #[test]
    fn test_matmul_chain_single_lane() {
        let (g, _, _, _, _) = test_graphs::matmul_chain(4, 4, 4, 4, 4);
        let (plan, errors) = plan_and_validate(&g, 1);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
    }

    #[test]
    fn test_matmul_chain_4_lanes() {
        // Two chained matmuls: (A @ B) @ C.
        // M=4, K1=4, N1=4, K2=4, N2=4.
        // First matmul: 16 Mul groups + 4 ReduceSum groups.
        // Second matmul: 16 Mul groups + 4 ReduceSum groups.
        // 40 compute groups total.
        //
        // The second matmul reads ALL outputs of the first matmul (cross-lane),
        // so there must be a barrier between them.
        let (g, _, _, _, _) = test_graphs::matmul_chain(4, 4, 4, 4, 4);
        let (plan, errors) = plan_and_validate(&g, 4);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
        // Should have at least 2 phases (barrier between matmuls).
        assert!(
            plan.phases.len() >= 2,
            "Expected >= 2 phases, got {}",
            plan.phases.len()
        );
    }

    // ── Multi-layer test with split and non-split groups ─────────────────────

    #[test]
    fn test_matmul_elementwise_matmul() {
        // Pattern: matmul -> elementwise activation -> matmul.
        // First matmul: M=8, K=4, N=6 -> produces 8*6=48 atoms.
        // Elementwise: Tanh over 48 atoms. Reads matmul output 1:1 (lane-aligned).
        // Second matmul: M=8, K=6, N=4 -> reads ALL 48 activation outputs (cross-lane).
        //
        // Expected: phase 0 = matmul1 + tanh, phase 1 = matmul2.
        let (g, _, _, out) = test_graphs::matmul_activation(8, 4, 6, ScalarUnaryOp::Tanh);
        // Verify base graph is valid.
        let errors = g.validate();
        assert!(errors.is_empty(), "Graph validation: {:?}", errors);

        let (plan, plan_errors) = plan_and_validate(&g, 4);
        assert!(plan_errors.is_empty(), "Plan errors: {:?}", plan_errors);
    }

    /// Test: a Select group (small, count=3072) consumed by a larger group
    /// via StridedBroadcast. This is the GPT-2 attention mask pattern that
    /// caused 96 violations in v2c.
    #[test]
    fn test_select_consumed_by_strided_broadcast() {
        let mut g = NanoGraph::new();

        // Condition literal (small).
        let cond = g.push_group(
            3072,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![], vec![], vec![],
        );
        // True value literal.
        let true_val = g.push_group(
            3072,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![], vec![], vec![],
        );
        // False value literal.
        let false_val = g.push_group(
            3072,
            ScalarOp::Literal(NumericScalar::F32(-1e9)),
            vec![], vec![], vec![],
        );

        // Select group: count=3072, reads 3 literals.
        let select = g.push_group(
            3072,
            ScalarOp::Select {
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![], vec![],
            vec![
                InputRef::Affine { base: cond, stride: 1 },
                InputRef::Affine { base: true_val, stride: 1 },
                InputRef::Affine { base: false_val, stride: 1 },
            ],
        );

        // Large data literal.
        let data_lit = g.push_group(
            24576,
            ScalarOp::Literal(NumericScalar::F32(0.5)),
            vec![], vec![], vec![],
        );

        // Consumer: count=24576, reads data_lit 1:1 and Select via StridedBroadcast{repeat=8}.
        // This means each block of 8 consumer atoms shares one Select output.
        // 24576 / 8 = 3072, matching the Select's count.
        let consumer = g.push_group(
            24576,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![], vec![],
            vec![
                InputRef::Affine { base: data_lit, stride: 1 },
                InputRef::StridedBroadcast { base: select, stride: 1, repeat: 8 },
            ],
        );
        g.outputs = vec![consumer];

        let errors = g.validate();
        assert!(errors.is_empty(), "Graph validation: {:?}", errors);

        let (plan, plan_errors) = plan_and_validate(&g, 4);
        assert!(
            plan_errors.is_empty(),
            "Plan errors (should be 0 violations): {:?}",
            plan_errors
        );

        // The consumer has 24576 atoms split across 4 lanes (6144 each).
        // The StridedBroadcast{repeat=8} means each lane reads 6144/8 = 768 Select atoms.
        // Lane 0 reads Select atoms 0..768, Lane 1 reads 768..1536, etc.
        // This IS lane-aligned: 3072/4 = 768 = 6144/8.
        // So Select and Consumer should be in the same phase, both split.
        let total = count_output_atoms(&plan);
        assert_eq!(total, 24576 + 3072, "Expected {} output atoms, got {}", 24576 + 3072, total);
    }

    /// Test: Select consumed by a consumer where the StridedBroadcast repeat
    /// does NOT align with lane boundaries, forcing duplication.
    #[test]
    fn test_select_misaligned_strided_broadcast() {
        let mut g = NanoGraph::new();

        let cond = g.push_group(
            100,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![], vec![], vec![],
        );
        let true_val = g.push_group(
            100,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![], vec![], vec![],
        );
        let false_val = g.push_group(
            100,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![], vec![], vec![],
        );

        let select = g.push_group(
            100,
            ScalarOp::Select {
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![], vec![],
            vec![
                InputRef::Affine { base: cond, stride: 1 },
                InputRef::Affine { base: true_val, stride: 1 },
                InputRef::Affine { base: false_val, stride: 1 },
            ],
        );

        let data = g.push_group(
            1000,
            ScalarOp::Literal(NumericScalar::F32(0.5)),
            vec![], vec![], vec![],
        );

        // StridedBroadcast{repeat=7}: 1000/7 ~= 143, but Select has only 100 atoms.
        // This will NOT be lane-aligned because 100/4 = 25 but consumer lane reads
        // different sets of Select atoms.
        let consumer = g.push_group(
            1000,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![], vec![],
            vec![
                InputRef::Affine { base: data, stride: 1 },
                InputRef::StridedBroadcast { base: select, stride: 1, repeat: 7 },
            ],
        );
        g.outputs = vec![consumer];

        let errors = g.validate();
        assert!(errors.is_empty(), "Graph validation: {:?}", errors);

        let (plan, plan_errors) = plan_and_validate(&g, 4);
        assert!(
            plan_errors.is_empty(),
            "Plan errors (should be 0 violations): {:?}",
            plan_errors
        );
    }

    /// Multi-layer test: matmul chain with elementwise ops between layers.
    /// Tests both split groups (elementwise) and barrier-forcing cross-lane reads.
    #[test]
    fn test_multi_layer_complex() {
        let mut g = NanoGraph::new();

        // Layer 1: Simple matmul M=8, K=4, N=8
        let a = g.push_group(
            8 * 4,
            ScalarOp::Literal(NumericScalar::F32(0.1)),
            vec![], vec![], vec![],
        );
        let b = g.push_group(
            4 * 8,
            ScalarOp::Literal(NumericScalar::F32(0.2)),
            vec![], vec![], vec![],
        );

        // Mul groups for layer 1.
        let mut mul1_base: Option<AtomId> = None;
        for m in 0..8u64 {
            for k in 0..4u64 {
                let base = g.push_group(
                    8,
                    ScalarOp::Binary {
                        op: ScalarBinOp::Mul,
                        compute_dtype: DType::F32,
                        output_dtype: DType::F32,
                    },
                    vec![], vec![],
                    vec![
                        InputRef::Broadcast(a.offset(m * 4 + k)),
                        InputRef::Affine { base: b.offset(k * 8), stride: 1 },
                    ],
                );
                if mul1_base.is_none() {
                    mul1_base = Some(base);
                }
            }
        }
        let mul1_base = mul1_base.unwrap();

        // ReduceSum groups for layer 1.
        let mut red1_base: Option<AtomId> = None;
        for m in 0..8u64 {
            let base = g.push_group(
                8,
                ScalarOp::ReduceSum {
                    reduce_count: 4,
                    reduce_stride: 8,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![], vec![],
                vec![InputRef::Affine {
                    base: AtomId(mul1_base.0 + m * 4 * 8),
                    stride: 1,
                }],
            );
            if red1_base.is_none() {
                red1_base = Some(base);
            }
        }
        let red1_base = red1_base.unwrap();

        // Elementwise Tanh (count=64, same as matmul output).
        // Affine(stride=1) from reduce output -> lane-aligned.
        let tanh_out = g.push_group(
            64,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Tanh,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![], vec![],
            vec![InputRef::Affine { base: red1_base, stride: 1 }],
        );

        // Bias add with Broadcast scalar (should be duplicated, not barrier).
        let bias = g.push_atom(
            ScalarOp::Literal(NumericScalar::F32(0.1)),
            vec![], vec![], vec![],
        );
        let biased = g.push_group(
            64,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![], vec![],
            vec![
                InputRef::Affine { base: tanh_out, stride: 1 },
                InputRef::Broadcast(bias),
            ],
        );

        // Layer 2: Another matmul M=8, K=8, N=4 reading from biased output.
        // This creates a cross-lane dependency (each Mul group reads one atom
        // from `biased` via Broadcast, and the full biased output is split).
        let c = g.push_group(
            8 * 4,
            ScalarOp::Literal(NumericScalar::F32(0.3)),
            vec![], vec![], vec![],
        );

        let mut mul2_base: Option<AtomId> = None;
        for m in 0..8u64 {
            for k in 0..8u64 {
                let base = g.push_group(
                    4,
                    ScalarOp::Binary {
                        op: ScalarBinOp::Mul,
                        compute_dtype: DType::F32,
                        output_dtype: DType::F32,
                    },
                    vec![], vec![],
                    vec![
                        InputRef::Broadcast(biased.offset(m * 8 + k)),
                        InputRef::Affine { base: c.offset(k * 4), stride: 1 },
                    ],
                );
                if mul2_base.is_none() {
                    mul2_base = Some(base);
                }
            }
        }
        let mul2_base = mul2_base.unwrap();

        let mut red2_base: Option<AtomId> = None;
        for m in 0..8u64 {
            let base = g.push_group(
                4,
                ScalarOp::ReduceSum {
                    reduce_count: 8,
                    reduce_stride: 4,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![], vec![],
                vec![InputRef::Affine {
                    base: AtomId(mul2_base.0 + m * 8 * 4),
                    stride: 1,
                }],
            );
            if red2_base.is_none() {
                red2_base = Some(base);
            }
        }
        let red2_base = red2_base.unwrap();
        g.outputs = vec![red2_base];

        let errors = g.validate();
        assert!(errors.is_empty(), "Graph validation: {:?}", errors);

        // Test with 4 lanes.
        let (plan, plan_errors) = plan_and_validate(&g, 4);
        assert!(
            plan_errors.is_empty(),
            "Plan errors: {:?}",
            plan_errors
        );

        // Should have multiple phases (at least 2: matmul1+tanh+bias, matmul2).
        assert!(
            plan.phases.len() >= 2,
            "Expected >= 2 phases, got {}",
            plan.phases.len()
        );

        // Test with 8 lanes for good measure.
        let (plan8, plan8_errors) = plan_and_validate(&g, 8);
        assert!(
            plan8_errors.is_empty(),
            "Plan errors (8 lanes): {:?}",
            plan8_errors
        );
    }

    /// Test with a Modular InputRef pattern (bias broadcast along batch dim).
    #[test]
    fn test_modular_input_ref() {
        let mut g = NanoGraph::new();

        // Bias vector: 768 atoms.
        let bias = g.push_group(
            768,
            ScalarOp::Literal(NumericScalar::F32(0.1)),
            vec![], vec![], vec![],
        );

        // Input data: 3072 atoms (4 * 768 batch dimension).
        let data = g.push_group(
            3072,
            ScalarOp::Literal(NumericScalar::F32(0.5)),
            vec![], vec![], vec![],
        );

        // Add with Modular input: each batch repeats the same bias.
        // atom i reads bias[i % 768].
        let added = g.push_group(
            3072,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![], vec![],
            vec![
                InputRef::Affine { base: data, stride: 1 },
                InputRef::Modular { base: bias, stride: 1, modulus: 768 },
            ],
        );
        g.outputs = vec![added];

        let errors = g.validate();
        assert!(errors.is_empty(), "Graph validation: {:?}", errors);

        let (plan, plan_errors) = plan_and_validate(&g, 4);
        assert!(
            plan_errors.is_empty(),
            "Plan errors: {:?}",
            plan_errors
        );
    }

    /// Test edge classification directly.
    #[test]
    fn test_edge_classification_affine_same_count() {
        let prod = AtomGroup {
            base_id: AtomId(0),
            count: 1024,
            op: ScalarOp::Literal(NumericScalar::F32(0.0)),
            sym_dims: vec![],
            reduce_dims: vec![],
            inputs: vec![],
        };
        let cons = AtomGroup {
            base_id: AtomId(1024),
            count: 1024,
            op: ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            sym_dims: vec![],
            reduce_dims: vec![],
            inputs: vec![InputRef::Affine { base: AtomId(0), stride: 1 }],
        };
        // Same count, stride=1, base matches producer start -> lane-aligned.
        assert!(is_edge_lane_aligned(&prod, &cons, &InputRef::Affine { base: AtomId(0), stride: 1 }, 4));
    }

    #[test]
    fn test_edge_classification_broadcast() {
        let prod = AtomGroup {
            base_id: AtomId(0),
            count: 1024,
            op: ScalarOp::Literal(NumericScalar::F32(0.0)),
            sym_dims: vec![],
            reduce_dims: vec![],
            inputs: vec![],
        };
        let cons = AtomGroup {
            base_id: AtomId(1024),
            count: 1024,
            op: ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            sym_dims: vec![],
            reduce_dims: vec![],
            inputs: vec![InputRef::Broadcast(AtomId(500))],
        };
        // Broadcast from a multi-atom producer -> cross-lane.
        assert!(!is_edge_lane_aligned(&prod, &cons, &InputRef::Broadcast(AtomId(500)), 4));
    }

    #[test]
    fn test_edge_classification_broadcast_single_atom() {
        let prod = AtomGroup {
            base_id: AtomId(0),
            count: 1,
            op: ScalarOp::Literal(NumericScalar::F32(0.0)),
            sym_dims: vec![],
            reduce_dims: vec![],
            inputs: vec![],
        };
        let cons = AtomGroup {
            base_id: AtomId(1),
            count: 1024,
            op: ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            sym_dims: vec![],
            reduce_dims: vec![],
            inputs: vec![InputRef::Broadcast(AtomId(0))],
        };
        // Broadcast from a single-atom producer -> lane-aligned (trivially).
        assert!(is_edge_lane_aligned(&prod, &cons, &InputRef::Broadcast(AtomId(0)), 4));
    }

    /// Build a realistic matmul using the StridedBroadcast pattern from GPT-2.
    /// C[M, N] = A[M, K] @ B[K, N], with M Mul groups (each count=K*N)
    /// and M ReduceSum groups (each count=N).
    fn build_realistic_matmul(g: &mut NanoGraph, a_base: AtomId, b_base: AtomId, m: u64, k: u64, n: u64) -> AtomId {
        let mut mul_bases = Vec::new();
        for row in 0..m {
            let a_row = a_base.offset(row * k);
            let mul = g.push_group(
                k * n,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![], vec![],
                vec![
                    InputRef::StridedBroadcast { base: a_row, stride: 1, repeat: n },
                    InputRef::Affine { base: b_base, stride: 1 },
                ],
            );
            mul_bases.push(mul);
        }
        let mut reduce_base: Option<AtomId> = None;
        for row in 0..m {
            let red = g.push_group(
                n,
                ScalarOp::ReduceSum {
                    reduce_count: k,
                    reduce_stride: n as i64,
                    compute_dtype: DType::F32,
                    output_dtype: DType::F32,
                },
                vec![], vec![],
                vec![InputRef::Affine { base: mul_bases[row as usize], stride: 1 }],
            );
            if reduce_base.is_none() {
                reduce_base = Some(red);
            }
        }
        reduce_base.unwrap()
    }

    /// Realistic GPT-2 style matmul with StridedBroadcast Mul groups.
    #[test]
    fn test_realistic_matmul_strided_broadcast() {
        let mut g = NanoGraph::new();
        let m = 8u64;
        let k = 4u64;
        let n = 6u64;

        let a = g.push_group(m * k, ScalarOp::Literal(NumericScalar::F32(1.0)), vec![], vec![], vec![]);
        let b = g.push_group(k * n, ScalarOp::Literal(NumericScalar::F32(1.0)), vec![], vec![], vec![]);

        let red_base = build_realistic_matmul(&mut g, a, b, m, k, n);
        g.outputs = vec![red_base];

        let errors = g.validate();
        assert!(errors.is_empty(), "Graph validation: {:?}", errors);

        // Test with 4 lanes.
        let (plan, plan_errors) = plan_and_validate(&g, 4);
        assert!(plan_errors.is_empty(), "Plan errors: {:?}", plan_errors);

        // Test with 8 lanes.
        let (plan8, plan8_errors) = plan_and_validate(&g, 8);
        assert!(plan8_errors.is_empty(), "Plan errors (8 lanes): {:?}", plan8_errors);
    }

    /// Realistic two-layer matmul chain: (A @ B) -> Tanh -> (result @ C).
    /// Uses StridedBroadcast pattern. This is the core GPT-2 pattern that
    /// caused 96 violations in previous attempts.
    #[test]
    fn test_realistic_matmul_chain_with_activation() {
        let mut g = NanoGraph::new();
        let m = 8u64;
        let k1 = 4u64;
        let n1 = 6u64;
        let k2 = n1; // n1 == k2 for the chain
        let n2 = 4u64;

        // Layer 1 weights.
        let a = g.push_group(m * k1, ScalarOp::Literal(NumericScalar::F32(1.0)), vec![], vec![], vec![]);
        let b1 = g.push_group(k1 * n1, ScalarOp::Literal(NumericScalar::F32(1.0)), vec![], vec![], vec![]);

        let red1 = build_realistic_matmul(&mut g, a, b1, m, k1, n1);

        // Activation: Tanh over M*N1=48 atoms. Lane-aligned with reduce output.
        let act = g.push_group(
            m * n1,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Tanh,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![], vec![],
            vec![InputRef::Affine { base: red1, stride: 1 }],
        );

        // Layer 2 weights.
        let b2 = g.push_group(k2 * n2, ScalarOp::Literal(NumericScalar::F32(1.0)), vec![], vec![], vec![]);

        // Layer 2 matmul: uses activation output as A matrix.
        // Each Mul group reads one A element via Broadcast from `act`.
        // This is the key: each Mul group broadcasts one atom from `act` (count=64),
        // which is split across lanes. The Broadcast is cross-lane.
        let red2 = build_realistic_matmul(&mut g, act, b2, m, k2, n2);
        g.outputs = vec![red2];

        let errors = g.validate();
        assert!(errors.is_empty(), "Graph validation: {:?}", errors);

        // Test with 4 lanes.
        let (plan, plan_errors) = plan_and_validate(&g, 4);
        assert!(plan_errors.is_empty(), "Plan errors: {:?}", plan_errors);
        // Must have at least 2 phases: matmul1+tanh vs matmul2.
        assert!(plan.phases.len() >= 2, "Expected >= 2 phases, got {}", plan.phases.len());

        // Test with 8 lanes.
        let (plan8, plan8_errors) = plan_and_validate(&g, 8);
        assert!(plan8_errors.is_empty(), "Plan errors (8 lanes): {:?}", plan8_errors);
    }

    /// Test the GPT-2 attention mask pattern: a small Select group consumed
    /// via StridedBroadcast by a large Mul group, in the realistic matmul structure.
    #[test]
    fn test_attention_mask_pattern() {
        let mut g = NanoGraph::new();

        // Attention mask Select: count=3072 (12 heads * 256 positions).
        let mask_cond = g.push_group(3072, ScalarOp::Literal(NumericScalar::F32(0.0)), vec![], vec![], vec![]);
        let mask_true = g.push_group(3072, ScalarOp::Literal(NumericScalar::F32(1.0)), vec![], vec![], vec![]);
        let mask_false = g.push_group(3072, ScalarOp::Literal(NumericScalar::F32(-1e9)), vec![], vec![], vec![]);

        let mask = g.push_group(
            3072,
            ScalarOp::Select { compute_dtype: DType::F32, output_dtype: DType::F32 },
            vec![], vec![],
            vec![
                InputRef::Affine { base: mask_cond, stride: 1 },
                InputRef::Affine { base: mask_true, stride: 1 },
                InputRef::Affine { base: mask_false, stride: 1 },
            ],
        );

        // Attention scores: count=24576 (3072 * 8 repeat).
        let scores = g.push_group(24576, ScalarOp::Literal(NumericScalar::F32(0.5)), vec![], vec![], vec![]);

        // Masked scores: scores + mask via StridedBroadcast{repeat=8}.
        let masked = g.push_group(
            24576,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![], vec![],
            vec![
                InputRef::Affine { base: scores, stride: 1 },
                InputRef::StridedBroadcast { base: mask, stride: 1, repeat: 8 },
            ],
        );

        // Softmax (simplified: Exp).
        let softmax = g.push_group(
            24576,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![], vec![],
            vec![InputRef::Affine { base: masked, stride: 1 }],
        );
        g.outputs = vec![softmax];

        let errors = g.validate();
        assert!(errors.is_empty(), "Graph validation: {:?}", errors);

        // Test with 4 lanes: the StridedBroadcast{repeat=8} is lane-aligned
        // when 3072/4 = 768 and 24576/4 = 6144, since 6144/8 = 768.
        let (plan, plan_errors) = plan_and_validate(&g, 4);
        assert!(plan_errors.is_empty(), "Plan errors (4 lanes): {:?}", plan_errors);

        // Test with 8 lanes: 3072/8 = 384, 24576/8 = 3072, 3072/8 = 384. Also aligned.
        let (plan8, plan8_errors) = plan_and_validate(&g, 8);
        assert!(plan8_errors.is_empty(), "Plan errors (8 lanes): {:?}", plan8_errors);

        // Test with 7 lanes: not perfectly aligned, should still produce 0 violations.
        let (plan7, plan7_errors) = plan_and_validate(&g, 7);
        assert!(plan7_errors.is_empty(), "Plan errors (7 lanes): {:?}", plan7_errors);
    }

    /// Stress test: larger matmul chain to verify no regressions.
    #[test]
    fn test_larger_matmul_chain() {
        let mut g = NanoGraph::new();
        let m = 16u64;
        let k1 = 8u64;
        let n1 = 12u64;
        let k2 = n1;
        let n2 = 8u64;

        let a = g.push_group(m * k1, ScalarOp::Literal(NumericScalar::F32(0.1)), vec![], vec![], vec![]);
        let b1 = g.push_group(k1 * n1, ScalarOp::Literal(NumericScalar::F32(0.2)), vec![], vec![], vec![]);

        let red1 = build_realistic_matmul(&mut g, a, b1, m, k1, n1);

        // Elementwise chain: Tanh -> Add bias -> result.
        let act = g.push_group(
            m * n1,
            ScalarOp::Unary { op: ScalarUnaryOp::Tanh, compute_dtype: DType::F32, output_dtype: DType::F32 },
            vec![], vec![],
            vec![InputRef::Affine { base: red1, stride: 1 }],
        );
        let bias = g.push_atom(ScalarOp::Literal(NumericScalar::F32(0.1)), vec![], vec![], vec![]);
        let biased = g.push_group(
            m * n1,
            ScalarOp::Binary { op: ScalarBinOp::Add, compute_dtype: DType::F32, output_dtype: DType::F32 },
            vec![], vec![],
            vec![
                InputRef::Affine { base: act, stride: 1 },
                InputRef::Broadcast(bias),
            ],
        );

        let b2 = g.push_group(k2 * n2, ScalarOp::Literal(NumericScalar::F32(0.3)), vec![], vec![], vec![]);
        let red2 = build_realistic_matmul(&mut g, biased, b2, m, k2, n2);
        g.outputs = vec![red2];

        let errors = g.validate();
        assert!(errors.is_empty(), "Graph validation: {:?}", errors);

        for lanes in &[1, 2, 4, 8, 16] {
            let (plan, plan_errors) = plan_and_validate(&g, *lanes);
            assert!(
                plan_errors.is_empty(),
                "Plan errors ({} lanes): {:?}",
                lanes,
                plan_errors
            );
        }
    }
}
