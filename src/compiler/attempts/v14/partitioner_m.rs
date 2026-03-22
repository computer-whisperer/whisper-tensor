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

use crate::dtype::DType;
use crate::graph::GlobalId;
use crate::nano_graph::pattern::InputTensor;
use crate::nano_graph::{AtomGroup, AtomId, AtomRange, InputRef, NanoGraph, ScalarOp, SymDim};

use super::types::{Phase, Span};

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
fn classify_group(group: &AtomGroup, num_lanes: usize) -> GroupKind {
    // Literals are always duplicated — every lane may need the values.
    if matches!(group.op, ScalarOp::Literal(_)) {
        return GroupKind::Duplicate;
    }

    // Very small groups aren't worth splitting.
    if group.count < num_lanes as u64 {
        // But if count == 1 and it's not a literal, it might be a scalar
        // intermediate. Still not worth splitting.
        if group.count <= 1 {
            return GroupKind::Duplicate; // scalar — just duplicate
        }
        return GroupKind::Whole;
    }

    match &group.op {
        ScalarOp::Literal(_) => unreachable!(), // handled above

        // Elementwise ops: embarrassingly parallel, always split.
        ScalarOp::Binary { .. }
        | ScalarOp::Unary { .. }
        | ScalarOp::Select
        | ScalarOp::Identity => {
            // Groups with Explicit inputs can't be trivially split because
            // InputRef::Explicit stores a Vec<AtomId> that must have exactly
            // group.count entries, and the atom_offset mechanism doesn't compose
            // with sliced Explicit vectors. Keep these whole.
            let has_explicit = group
                .inputs
                .iter()
                .any(|inp| matches!(inp, InputRef::Explicit(_)));
            if has_explicit {
                GroupKind::Whole
            } else {
                GroupKind::Split
            }
        }

        // IndirectLoad: each lookup is independent, split freely.
        ScalarOp::IndirectLoad { .. } => GroupKind::Split,

        // Reduce: depends on what it's reducing over.
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
    }
}

// ─── Dependency analysis ───────────────────────────────────────────────────

/// Build group-level dependency DAG.
/// Returns (producers[gi], successors[gi]) as sorted vecs.
fn build_dependency_dag(graph: &NanoGraph) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
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
    groups: &[AtomGroup],
    kinds: &[GroupKind],
    producers: &[Vec<usize>],
    graph: &NanoGraph,
    num_lanes: usize,
) -> Vec<usize> {
    let n = groups.len();
    let mut phase_of = vec![0usize; n];

    // For each group, determine the earliest phase it can be in.
    // Groups are in topological order (NanoGraph invariant).
    for gi in 0..n {
        let mut min_phase = 0usize;

        for &pi in &producers[gi] {
            let prod_phase = phase_of[pi];
            let prod_kind = kinds[pi];
            let cons_kind = kinds[gi];

            // Determine if we need a barrier between producer and consumer.
            let needs_barrier =
                needs_barrier_between(pi, gi, prod_kind, cons_kind, groups, graph, num_lanes);

            if needs_barrier {
                min_phase = min_phase.max(prod_phase + 1);
            } else {
                // Same phase is fine — no cross-lane dependency.
                min_phase = min_phase.max(prod_phase);
            }
        }

        phase_of[gi] = min_phase;
    }

    phase_of
}

/// Determine if a barrier is needed between a producer and consumer group.
///
/// Returns true if the consumer cannot safely execute in the same phase
/// as the producer due to cross-lane data dependencies.
fn needs_barrier_between(
    pi: usize,
    ci: usize,
    prod_kind: GroupKind,
    cons_kind: GroupKind,
    groups: &[AtomGroup],
    graph: &NanoGraph,
    num_lanes: usize,
) -> bool {
    let prod = &groups[pi];
    let cons = &groups[ci];

    match (prod_kind, cons_kind) {
        // Split → Split: no barrier if the consumer reads only from its own
        // lane's slice. This is true when the input pattern is Affine with
        // stride 1 (elementwise chain) or StridedBroadcast with aligned blocks.
        (GroupKind::Split, GroupKind::Split) => {
            // Check if the consumer's input refs to this producer are
            // lane-local (each lane's fragment only reads from the same
            // lane's fragment of the producer).
            !is_lane_local_access(cons, prod, num_lanes)
        }

        // Duplicate → anything: no barrier. Duplicated data is in every lane.
        (GroupKind::Duplicate, _) => false,

        // Split → Duplicate consumer: the consumer is duplicated into every
        // lane, and it reads from the split producer. If the consumer needs
        // all atoms of the producer (e.g., a reduce over the full range),
        // that's a cross-lane dependency → barrier needed.
        // If the consumer only reads a broadcast (single atom), no barrier
        // if that atom is available.
        (GroupKind::Split, GroupKind::Duplicate) => {
            // A duplicated consumer that reads from a split producer:
            // if the consumer broadcasts a single atom from the producer,
            // we need a barrier because we don't know which lane has it.
            // Actually: duplicated groups are small (literals, scalars).
            // If a non-literal duplicated group reads from a split group,
            // we need a barrier.
            true
        }

        // Split → Whole: the whole group sits on one lane but may need
        // atoms from all lanes of the split producer → barrier.
        (GroupKind::Split, GroupKind::Whole) => true,

        // Whole → Split: the whole group is on lane 0 only. Other lanes
        // can't read its output within the same phase. Barrier needed.
        (GroupKind::Whole, GroupKind::Split) => true,

        // Whole → Whole: both on lane 0 (or same lane) → no barrier if
        // same phase ordering works.
        (GroupKind::Whole, GroupKind::Whole) => false,

        // Whole → Duplicate: the duplicate runs on all lanes but Whole is only
        // on lane 0. If in the same phase, other lanes can't access the Whole
        // output. Need a barrier so the Whole group's output is in the value store.
        (GroupKind::Whole, GroupKind::Duplicate) => true,

        // Duplicate → anything already handled above.
        (GroupKind::Duplicate, _) => false,
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
fn is_lane_local_access(consumer: &AtomGroup, producer: &AtomGroup, num_lanes: usize) -> bool {
    let prod_base = producer.base_id.0;
    let prod_end = prod_base + producer.count;

    for input in &consumer.inputs {
        // Check if this input references the producer at all.
        let refs_producer = input_refs_group(input, consumer.count, consumer.atom_offset, producer);
        if !refs_producer {
            continue;
        }

        match input {
            InputRef::Broadcast(_) => {
                // A broadcast reads a single atom. If the producer is split,
                // that atom lives on exactly one lane. Other lanes won't have it.
                // → Not lane-local (need barrier or duplication).
                return false;
            }
            InputRef::Affine { base, stride } => {
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
                let abs_stride = (*stride).unsigned_abs();

                // General check: does stride * consumer_count == producer_count?
                // This means each lane's consumer chunk maps to exactly one lane's
                // producer chunk.
                if abs_stride > 0
                    && abs_stride * consumer.count == producer.count
                    && producer.count % num_lanes as u64 == 0
                    && consumer.count % num_lanes as u64 == 0
                {
                    // Verify base alignment: first consumer atom should read from
                    // producer start (or start of producer + some lane-aligned offset).
                    let first_read = if *stride >= 0 {
                        base.0
                            .wrapping_add((abs_stride * consumer.atom_offset) as u64)
                    } else {
                        base.0
                            .wrapping_sub((abs_stride * consumer.atom_offset) as u64)
                    };

                    if first_read == prod_base {
                        // Perfect alignment: lane k's consumer chunk
                        // reads exactly lane k's producer chunk.
                        continue; // lane-local
                    }
                }

                // Special case: stride=1, same count (elementwise 1:1).
                if *stride == 1 && consumer.count == producer.count {
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
                return false;
            }
            InputRef::StridedBroadcast {
                base,
                stride,
                repeat,
            } => {
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
                let abs_stride = (*stride).unsigned_abs();

                // chunk must be a multiple of repeat for alignment.
                if chunk % *repeat != 0 {
                    return false;
                }

                let distinct_per_chunk = chunk / *repeat;

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

                return false;
            }
            InputRef::Modular {
                base,
                stride,
                modulus,
            } => {
                // Modular: atom i reads base + stride * (i % modulus).
                // This tiles/repeats — every lane needs the same modulus-sized
                // range. NOT lane-local unless the producer is duplicated.
                return false;
            }
            InputRef::Explicit(_) => {
                // Arbitrary mapping — not lane-local in general.
                return false;
            }
        }
    }

    true
}

/// Check if any of the consumer's inputs broadcast a single atom from the producer.
fn is_broadcast_access(consumer: &AtomGroup, producer: &AtomGroup) -> Option<AtomId> {
    for input in &consumer.inputs {
        if let InputRef::Broadcast(id) = input {
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
    producer: &AtomGroup,
) -> bool {
    if consumer_count == 0 {
        return false;
    }
    let pb = producer.base_id.0;
    let pe = pb + producer.count;

    match input {
        InputRef::Broadcast(id) => id.0 >= pb && id.0 < pe,
        InputRef::Affine { base, stride } => {
            let first = base
                .0
                .wrapping_add((*stride * consumer_offset as i64) as u64);
            let last = base
                .0
                .wrapping_add((*stride * (consumer_offset + consumer_count - 1) as i64) as u64);
            let lo = first.min(last);
            let hi = first.max(last);
            lo < pe && hi >= pb
        }
        InputRef::StridedBroadcast {
            base,
            stride,
            repeat,
        } => {
            let first_block = consumer_offset / repeat;
            let last_block = (consumer_offset + consumer_count - 1) / repeat;
            let first = base.0.wrapping_add((*stride * first_block as i64) as u64);
            let last = base.0.wrapping_add((*stride * last_block as i64) as u64);
            let lo = first.min(last);
            let hi = first.max(last);
            lo < pe && hi >= pb
        }
        InputRef::Modular {
            base,
            stride,
            modulus,
        } => {
            let a = base.0;
            let b = base
                .0
                .wrapping_add((*stride * (*modulus as i64 - 1)) as u64);
            let lo = a.min(b);
            let hi = a.max(b);
            lo < pe && hi >= pb
        }
        InputRef::Explicit(ids) => ids
            .iter()
            .skip(consumer_offset as usize)
            .take(consumer_count as usize)
            .any(|id| id.0 >= pb && id.0 < pe),
    }
}

// ─── Span building ─────────────────────────────────────────────────────────

/// For a split group, compute the atom range for a given lane.
fn split_range(group: &AtomGroup, lane: usize, num_lanes: usize) -> (u64, u64) {
    let chunk = group.count / num_lanes as u64;
    let remainder = group.count % num_lanes as u64;
    // Distribute remainder: first `remainder` lanes get one extra atom.
    let start = chunk * lane as u64 + (lane as u64).min(remainder);
    let count = chunk + if (lane as u64) < remainder { 1 } else { 0 };
    (start, count)
}

/// Collect all atom ranges that a group's inputs reference, including
/// reduce stride ranges and indirect load tables.
fn collect_input_atom_ranges(group: &AtomGroup, graph: &NanoGraph) -> Vec<(AtomId, u64, DType)> {
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
        match input {
            InputRef::Broadcast(id)
            | InputRef::Affine { base: id, .. }
            | InputRef::StridedBroadcast { base: id, .. }
            | InputRef::Modular { base: id, .. } => {
                if let Some((idx, _)) = graph.find_input_idx(*id) {
                    let it = &graph.input_tensors()[idx];
                    if seen_bases.insert(it.base_id.0) {
                        ranges.push((it.base_id, it.count, it.dtype));
                    }
                }
            }
            InputRef::Explicit(ids) => {
                for id in ids {
                    if let Some((idx, _)) = graph.find_input_idx(*id) {
                        let it = &graph.input_tensors()[idx];
                        if seen_bases.insert(it.base_id.0) {
                            ranges.push((it.base_id, it.count, it.dtype));
                        }
                    }
                }
            }
        }
    }

    // IndirectLoad table
    if let ScalarOp::IndirectLoad { table_base } = &group.op {
        if let Some(tg) = graph.group_of(*table_base) {
            if seen_bases.insert(tg.base_id.0) {
                ranges.push((tg.base_id, tg.count, tg.output_dtype));
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
    graph: &NanoGraph,
    num_lanes: usize,
    input_tensors: &[InputTensor],
    output_atom_ids: &[AtomId],
) -> Vec<Phase> {
    let num_lanes = num_lanes.max(1);
    let groups = graph.groups();
    let n = groups.len();

    if n == 0 {
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

    // Step 1: Classify groups.
    let kinds: Vec<GroupKind> = groups
        .iter()
        .map(|g| classify_group(g, num_lanes))
        .collect();

    // Step 2: Build dependency DAG.
    let (producers, successors) = build_dependency_dag(graph);

    // Step 3: Assign phases.
    let phase_of = assign_phases(groups, &kinds, &producers, graph, num_lanes);

    // Step 4: Collect groups by phase.
    let num_phases = phase_of.iter().copied().max().unwrap_or(0) + 1;
    let mut phase_groups: Vec<Vec<usize>> = vec![vec![]; num_phases];
    for (gi, &ph) in phase_of.iter().enumerate() {
        phase_groups[ph].push(gi);
    }

    // Step 5: Collect output group indices for determining span outputs.
    let output_group_set: HashSet<usize> = output_atom_ids
        .iter()
        .filter_map(|id| graph.find_group_idx(*id))
        .collect();

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
    graph: &NanoGraph,
    phase_group_indices: &[usize],
    kinds: &[GroupKind],
    phase_of: &[usize],
    producers: &[Vec<usize>],
    successors: &[Vec<usize>],
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

    // Copy sym_dim metadata into each span graph.
    for sg in &mut span_graphs {
        sg.sym_dim_names = graph.sym_dim_names.clone();
        sg.sym_dim_bounds = graph.sym_dim_bounds.clone();
    }

    // Track which atom ranges have been declared as inputs in each span.
    let mut declared_inputs: Vec<HashSet<u64>> = (0..num_lanes).map(|_| HashSet::new()).collect();
    // Track which atom ranges are produced in each span (for intra-phase deps).
    let mut produced_in_span: Vec<HashSet<u64>> = (0..num_lanes).map(|_| HashSet::new()).collect();

    // Determine which groups in this phase need their output to be in span outputs.
    let groups_needing_output: HashSet<usize> = phase_group_indices
        .iter()
        .filter(|&&gi| output_group_set.contains(&gi) || cross_phase_consumed.contains(&gi))
        .copied()
        .collect();

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
fn emit_split_group(
    graph: &NanoGraph,
    group: &AtomGroup,
    gi: usize,
    num_lanes: usize,
    span_graphs: &mut [NanoGraph],
    span_inputs: &mut [Vec<AtomRange>],
    span_outputs: &mut [Vec<AtomRange>],
    declared_inputs: &mut [HashSet<u64>],
    produced_in_span: &mut [HashSet<u64>],
    phase_of: &[usize],
    kinds: &[GroupKind],
    all_groups: &[AtomGroup],
    needs_output: bool,
    input_tensor_set: &HashMap<u64, &InputTensor>,
) {
    let base = group.base_id.0;

    for lane in 0..num_lanes {
        let (start, count) = split_range(group, lane, num_lanes);
        if count == 0 {
            continue;
        }

        let frag_base = AtomId(base + start);
        let atom_offset = group.atom_offset + start;

        // Clone inputs — atom_offset handles correct resolution for all
        // non-Explicit InputRef types. Groups with Explicit inputs are
        // classified as Whole, so we should never see them here.
        let inputs: Vec<InputRef> = group
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
fn clone_input_for_split(input: &InputRef) -> InputRef {
    debug_assert!(
        !matches!(input, InputRef::Explicit(_)),
        "Explicit InputRef should not appear in a split group"
    );
    input.clone()
}

/// Emit a duplicate group: full copy in every lane that needs it.
/// For simplicity (and correctness), duplicate into all lanes.
fn emit_duplicate_group(
    graph: &NanoGraph,
    group: &AtomGroup,
    gi: usize,
    num_lanes: usize,
    span_graphs: &mut [NanoGraph],
    span_inputs: &mut [Vec<AtomRange>],
    span_outputs: &mut [Vec<AtomRange>],
    declared_inputs: &mut [HashSet<u64>],
    produced_in_span: &mut [HashSet<u64>],
    phase_of: &[usize],
    kinds: &[GroupKind],
    all_groups: &[AtomGroup],
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
    graph: &NanoGraph,
    group: &AtomGroup,
    gi: usize,
    num_lanes: usize,
    span_graphs: &mut [NanoGraph],
    span_inputs: &mut [Vec<AtomRange>],
    span_outputs: &mut [Vec<AtomRange>],
    declared_inputs: &mut [HashSet<u64>],
    produced_in_span: &mut [HashSet<u64>],
    phase_of: &[usize],
    kinds: &[GroupKind],
    all_groups: &[AtomGroup],
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

/// Ensure that all atoms referenced by a group's inputs (and reduce strides,
/// indirect load tables) are either produced in this span or declared as
/// span inputs.
///
/// Uses the main graph's producer analysis to deterministically find all
/// producer groups, avoiding sampling-based approaches that can miss producers.
fn ensure_inputs_declared(
    graph: &NanoGraph,
    inputs: &[InputRef],
    op: &ScalarOp,
    atom_offset: u64,
    count: u64,
    lane: usize,
    span_graphs: &mut [NanoGraph],
    span_inputs: &mut [Vec<AtomRange>],
    declared_inputs: &mut [HashSet<u64>],
    produced_in_span: &[HashSet<u64>],
    input_tensor_set: &HashMap<u64, &InputTensor>,
) {
    // Strategy: collect all producer groups from the main graph, then check
    // which ones need to be declared as span inputs (not already in span).
    //
    // For each InputRef, find the range of atom IDs it can resolve to,
    // then look up which main-graph groups/input-tensors cover that range.

    // Helper: declare a range as a span input if not already present.
    // `access_lo` and `access_hi` describe the actual atom range accessed
    // (may be a subset of the full group range).
    let mut declare_range = |base: AtomId,
                             cnt: u64,
                             dtype: DType,
                             tensor_id: Option<GlobalId>,
                             access_lo: u64,
                             access_hi: u64| {
        let key = base.0;
        if declared_inputs[lane].contains(&key) {
            return;
        }
        // Check if the accessed portion of this range is already in the span.
        // We sample a few points in the overlap region [max(base, access_lo), min(base+cnt, access_hi+1)).
        let overlap_lo = base.0.max(access_lo);
        let overlap_hi = (base.0 + cnt).min(access_hi + 1);
        if overlap_lo < overlap_hi {
            // Sample the overlap region.
            let mid = overlap_lo + (overlap_hi - overlap_lo) / 2;
            if span_graphs[lane].contains_atom(AtomId(overlap_lo))
                && span_graphs[lane].contains_atom(AtomId(overlap_hi - 1))
                && span_graphs[lane].contains_atom(AtomId(mid))
            {
                return; // Already in span (produced by a same-phase, same-lane group).
            }
        } else if span_graphs[lane].contains_atom(base) {
            return;
        }
        declared_inputs[lane].insert(key);
        span_inputs[lane].push(AtomRange {
            base,
            count: cnt,
            dtype,
        });
        span_graphs[lane].insert_input_tensor_at(
            base,
            tensor_id.unwrap_or(GlobalId(0)),
            cnt,
            dtype,
        );
    };

    // Collect the full atom range accessed by each InputRef for this fragment.
    // We compute [lo, hi] bounds for the resolved atoms, then find all
    // main-graph groups and input tensors in that range.
    let mut ranges_to_cover: Vec<(u64, u64)> = Vec::new(); // (lo, hi) inclusive

    for input in inputs {
        if count == 0 {
            continue;
        }
        match input {
            InputRef::Broadcast(id) => {
                ranges_to_cover.push((id.0, id.0));
            }
            InputRef::Affine { base, stride } => {
                let first = input.resolve(atom_offset).0;
                let last = input.resolve(atom_offset + count - 1).0;
                ranges_to_cover.push((first.min(last), first.max(last)));
            }
            InputRef::StridedBroadcast {
                base,
                stride,
                repeat,
            } => {
                let first = input.resolve(atom_offset).0;
                let last = input.resolve(atom_offset + count - 1).0;
                ranges_to_cover.push((first.min(last), first.max(last)));
            }
            InputRef::Modular {
                base,
                stride,
                modulus,
            } => {
                let a = base.0;
                let b = (base.0 as i64 + *stride * (*modulus as i64 - 1)) as u64;
                ranges_to_cover.push((a.min(b), a.max(b)));
            }
            InputRef::Explicit(ids) => {
                let start = atom_offset as usize;
                let end = (atom_offset + count) as usize;
                let slice = &ids[start.min(ids.len())..end.min(ids.len())];
                if !slice.is_empty() {
                    let lo = slice.iter().map(|id| id.0).min().unwrap();
                    let hi = slice.iter().map(|id| id.0).max().unwrap();
                    ranges_to_cover.push((lo, hi));
                }
            }
        }
    }

    // Reduce stride: extends the accessed range beyond what the InputRef alone covers.
    if let ScalarOp::Reduce {
        reduce_count,
        reduce_stride,
        ..
    } = op
    {
        if *reduce_count > 1 && *reduce_stride != 0 {
            for input in inputs {
                if count == 0 {
                    continue;
                }
                let first = input.resolve(atom_offset).0;
                let last_base = input.resolve(atom_offset + count - 1).0;
                let stride_extent = (*reduce_count as i64 - 1) * reduce_stride;
                let endpoints = [
                    first,
                    (first as i64 + stride_extent) as u64,
                    last_base,
                    (last_base as i64 + stride_extent) as u64,
                ];
                let lo = *endpoints.iter().min().unwrap();
                let hi = *endpoints.iter().max().unwrap();
                ranges_to_cover.push((lo, hi));
            }
        }
    }

    // IndirectLoad table: need the full table range.
    if let ScalarOp::IndirectLoad { table_base } = op {
        // Look up the full table group.
        if let Some(tg) = graph.group_of(*table_base) {
            ranges_to_cover.push((tg.base_id.0, tg.base_id.0 + tg.count - 1));
        }
    }

    // For each range, find all main-graph groups and input tensors that intersect it.
    let all_groups = graph.groups();
    let all_inputs = graph.input_tensors();
    for (lo, hi) in &ranges_to_cover {
        // Check input tensors (typically few, linear scan is fine).
        for it in all_inputs {
            let it_end = it.base_id.0 + it.count;
            if *lo < it_end && it.base_id.0 <= *hi {
                declare_range(it.base_id, it.count, it.dtype, Some(it.tensor_id), *lo, *hi);
            }
        }

        // Check producer groups using binary search.
        // Groups are sorted by base_id. Find first group that could overlap with [lo, hi].
        // A group overlaps if group.base_id <= hi AND group.base_id + group.count > lo.
        // Start scanning from the first group whose base_id could overlap.
        // The first group that could overlap has base_id such that base_id + count > lo.
        // Conservative: find first group with base_id >= lo, then back up one.
        let start_idx = all_groups.partition_point(|g| g.base_id.0 + g.count <= *lo);
        for pg in &all_groups[start_idx..] {
            if pg.base_id.0 > *hi {
                break;
            }
            let pg_end = pg.base_id.0 + pg.count;
            if *lo < pg_end && pg.base_id.0 <= *hi {
                declare_range(pg.base_id, pg.count, pg.output_dtype, None, *lo, *hi);
            }
        }
    }
}

/// Find the input tensor covering an atom ID.
fn find_covering_input_tensor<'a>(
    atom: &AtomId,
    input_tensor_set: &HashMap<u64, &'a InputTensor>,
    graph: &NanoGraph,
) -> Option<InputTensor> {
    // Check graph's input tensors via find_input_idx
    if let Some((idx, _)) = graph.find_input_idx(*atom) {
        let it = &graph.input_tensors()[idx];
        return Some(it.clone());
    }
    None
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::graph::GlobalId;
    use crate::nano_graph::ops::{ReduceKind, ScalarBinOp, ScalarOp, ScalarUnaryOp};
    use crate::nano_graph::pattern::InputTensor;
    use crate::numeric_scalar::NumericScalar;

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
                    .filter(|g| !matches!(g.op, ScalarOp::Literal(_)))
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
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );

        let sub = g.push_group(
            8000,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Sub,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine {
                    base: lit,
                    stride: 1,
                },
                InputRef::Affine {
                    base: lit,
                    stride: 1,
                },
            ],
        );

        let pow = g.push_group(
            8000,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Pow,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine {
                    base: sub,
                    stride: 1,
                },
                InputRef::Broadcast(lit), // broadcast a literal atom
            ],
        );

        let add = g.push_group(
            8000,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine {
                    base: pow,
                    stride: 1,
                },
                InputRef::Affine {
                    base: sub,
                    stride: 1,
                },
            ],
        );

        g.outputs = vec![add];

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
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );

        let b = g.push_group(
            8000,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine { base: a, stride: 1 }],
        );

        let c = g.push_group(
            8000,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine { base: a, stride: 1 }],
        );

        let d = g.push_group(
            8000,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine { base: b, stride: 1 },
                InputRef::Affine { base: c, stride: 1 },
            ],
        );

        g.outputs = vec![d];

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
        let inp = g.add_input_tensor(GlobalId(0), k, DType::F32);

        // Weight literal.
        let weights = g.push_group(
            m * k,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.5)),
            vec![],
            vec![],
        );

        // Mul: each of M*K atoms computes weight[i] * input[i % K].
        // Input pattern: Modular(base=inp, stride=1, modulus=K)
        // Weight pattern: Affine(base=weights, stride=1)
        let mul = g.push_group(
            m * k,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine {
                    base: weights,
                    stride: 1,
                },
                InputRef::Modular {
                    base: inp,
                    stride: 1,
                    modulus: k,
                },
            ],
        );

        // Reduce: M output atoms, each sums K consecutive mul outputs.
        let reduce = g.push_group(
            m,
            DType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: k,
                reduce_stride: 1,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: mul,
                stride: k as i64,
            }],
        );

        // Bias literal.
        let bias = g.push_group(
            m,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.1)),
            vec![],
            vec![],
        );

        // Add: reduce + bias.
        let add = g.push_group(
            m,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine {
                    base: reduce,
                    stride: 1,
                },
                InputRef::Affine {
                    base: bias,
                    stride: 1,
                },
            ],
        );

        g.outputs = vec![add];

        let it = vec![InputTensor {
            tensor_id: GlobalId(0),
            base_id: inp,
            count: k,
            dtype: DType::F32,
        }];

        let phases = plan(&g, num_lanes, &it, &g.outputs.clone());

        // Validate.
        let errors = validate_all_spans(&phases);
        assert!(errors.is_empty(), "Validation errors: {:?}", errors);

        // The mul group has Modular input (reads all K elements from input vector),
        // which is NOT lane-local. So mul needs the input vector duplicated/declared.
        // But mul itself is split by its M*K atoms.

        // Check that groups ARE actually split across lanes.
        // Count total non-literal atoms across all spans.
        let mut total_non_lit = 0u64;
        let mut lanes_with_mul = 0;
        for phase in &phases {
            for (lane, span) in phase.spans.iter().enumerate() {
                let mut has_mul = false;
                for g in span.graph.groups() {
                    if !matches!(g.op, ScalarOp::Literal(_)) {
                        total_non_lit += g.count;
                    }
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
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(42.0)),
            vec![],
            vec![],
        );

        let neg = g.push_group(
            1000,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: lit,
                stride: 1,
            }],
        );

        g.outputs = vec![neg];

        let phases = plan(&g, num_lanes, &[], &g.outputs.clone());

        // Each lane should have the full literal (1000 atoms).
        for (lane, span) in phases[0].spans.iter().enumerate() {
            let lit_atoms: u64 = span
                .graph
                .groups()
                .iter()
                .filter(|g| matches!(g.op, ScalarOp::Literal(_)))
                .map(|g| g.count)
                .sum();
            assert_eq!(
                lit_atoms, 1000,
                "Lane {} should have 1000 literal atoms, got {}",
                lane, lit_atoms,
            );
        }

        // Each lane should have 250 neg atoms (1000 / 4).
        for (lane, span) in phases[0].spans.iter().enumerate() {
            let neg_atoms: u64 = span
                .graph
                .groups()
                .iter()
                .filter(|g| matches!(g.op, ScalarOp::Unary { .. }))
                .map(|g| g.count)
                .sum();
            assert_eq!(
                neg_atoms, 250,
                "Lane {} should have 250 neg atoms, got {}",
                lane, neg_atoms,
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
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );

        let reduced = g.push_group(
            1,
            DType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: 1024,
                reduce_stride: 1,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: data,
                stride: 1,
            }],
        );

        // Use the scalar result in a large group (broadcast).
        let output = g.push_group(
            8000,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Broadcast(reduced)],
        );

        g.outputs = vec![output];

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
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );

        let reduced = g.push_group(
            m,
            DType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: k,
                reduce_stride: 1,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: data,
                stride: k as i64,
            }],
        );

        g.outputs = vec![reduced];

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
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );

        let op = g.push_group(
            400,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: lit,
                stride: 1,
            }],
        );

        g.outputs = vec![op];

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

        let inp = g.add_input_tensor(GlobalId(0), 800, DType::F32);

        let neg = g.push_group(
            800,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: inp,
                stride: 1,
            }],
        );

        g.outputs = vec![neg];

        let it = vec![InputTensor {
            tensor_id: GlobalId(0),
            base_id: inp,
            count: 800,
            dtype: DType::F32,
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
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(5.0)),
            vec![],
            vec![],
        );

        g.outputs = vec![lit];

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
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );
        let sub = g.push_group(
            8000,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Sub,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine {
                    base: lit,
                    stride: 1,
                },
                InputRef::Affine {
                    base: lit,
                    stride: 1,
                },
            ],
        );
        let pow = g.push_group(
            8000,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Pow,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine {
                    base: sub,
                    stride: 1,
                },
                InputRef::Broadcast(lit),
            ],
        );
        let add = g.push_group(
            8000,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine {
                    base: pow,
                    stride: 1,
                },
                InputRef::Affine {
                    base: sub,
                    stride: 1,
                },
            ],
        );
        g.outputs = vec![add];
        let phases = plan(&g, 4, &[], &g.outputs.clone());
        verify_plan_full(&phases);
    }

    #[test]
    fn test_matmul_no_cross_lane() {
        let mut g = NanoGraph::new();
        let num_lanes = 4;
        let m = 32u64;
        let k = 64u64;
        let inp = g.add_input_tensor(GlobalId(0), k, DType::F32);
        let weights = g.push_group(
            m * k,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.5)),
            vec![],
            vec![],
        );
        let mul = g.push_group(
            m * k,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine {
                    base: weights,
                    stride: 1,
                },
                InputRef::Modular {
                    base: inp,
                    stride: 1,
                    modulus: k,
                },
            ],
        );
        let reduce = g.push_group(
            m,
            DType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: k,
                reduce_stride: 1,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: mul,
                stride: k as i64,
            }],
        );
        let bias = g.push_group(
            m,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.1)),
            vec![],
            vec![],
        );
        let add = g.push_group(
            m,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine {
                    base: reduce,
                    stride: 1,
                },
                InputRef::Affine {
                    base: bias,
                    stride: 1,
                },
            ],
        );
        g.outputs = vec![add];
        let it = vec![InputTensor {
            tensor_id: GlobalId(0),
            base_id: inp,
            count: k,
            dtype: DType::F32,
        }];
        let phases = plan(&g, num_lanes, &it, &g.outputs.clone());
        verify_plan_full(&phases);
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
        let x = g.add_input_tensor(GlobalId(0), n, DType::F32);

        // Step 1: ReduceSum over D elements → M outputs.
        // reduce(M) reads x via Affine(stride=D).
        let sum = g.push_group(
            m,
            DType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: d,
                reduce_stride: 1,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: x,
                stride: d as i64,
            }],
        );

        // Step 2: Divide by D to get mean. Broadcast a literal 1/D.
        let inv_d = g.push_group(
            1,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0 / d as f32)),
            vec![],
            vec![],
        );
        let mean = g.push_group(
            m,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine {
                    base: sum,
                    stride: 1,
                },
                InputRef::Broadcast(inv_d),
            ],
        );

        // Step 3: x - mean. Uses StridedBroadcast to broadcast each mean value across D elements.
        let x_centered = g.push_group(
            n,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Sub,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine { base: x, stride: 1 },
                InputRef::StridedBroadcast {
                    base: mean,
                    stride: 1,
                    repeat: d,
                },
            ],
        );

        // Step 4: x_centered^2
        let two = g.push_group(
            1,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
        );
        let pow2 = g.push_group(
            n,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Pow,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine {
                    base: x_centered,
                    stride: 1,
                },
                InputRef::Broadcast(two),
            ],
        );

        // Step 5: ReduceSum of pow2 → M variance values.
        let var_sum = g.push_group(
            m,
            DType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: d,
                reduce_stride: 1,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: pow2,
                stride: d as i64,
            }],
        );

        // Step 6: Divide by D and add epsilon, then rsqrt.
        let var_mean = g.push_group(
            m,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine {
                    base: var_sum,
                    stride: 1,
                },
                InputRef::Broadcast(inv_d),
            ],
        );
        let eps = g.push_group(
            1,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1e-5)),
            vec![],
            vec![],
        );
        let var_eps = g.push_group(
            m,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine {
                    base: var_mean,
                    stride: 1,
                },
                InputRef::Broadcast(eps),
            ],
        );
        let sqrt_var = g.push_group(
            m,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Sqrt,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: var_eps,
                stride: 1,
            }],
        );
        let rsqrt = g.push_group(
            m,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Reciprocal,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: sqrt_var,
                stride: 1,
            }],
        );

        // Step 7: Normalize: x_centered * rsqrt (StridedBroadcast).
        let normalized = g.push_group(
            n,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine {
                    base: x_centered,
                    stride: 1,
                },
                InputRef::StridedBroadcast {
                    base: rsqrt,
                    stride: 1,
                    repeat: d,
                },
            ],
        );

        g.outputs = vec![normalized];
        let it = vec![InputTensor {
            tensor_id: GlobalId(0),
            base_id: x,
            count: n,
            dtype: DType::F32,
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
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );

        // Consumer: N elements reading source via StridedBroadcast(repeat=D).
        // Each chunk of D consecutive consumer atoms reads the same source atom.
        let consumer = g.push_group(
            n,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::StridedBroadcast {
                base: source,
                stride: 1,
                repeat: d,
            }],
        );

        g.outputs = vec![consumer];
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
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );

        // Consumer: N elements reading source via StridedBroadcast(repeat=D).
        let consumer = g.push_group(
            n,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::StridedBroadcast {
                base: source,
                stride: 1,
                repeat: d,
            }],
        );

        g.outputs = vec![consumer];
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
        let inp = g.add_input_tensor(GlobalId(0), k1, DType::F32);

        // MatMul 1: [M, K1] @ input[K1] → output[M]
        let w1 = g.push_group(
            m * k1,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.1)),
            vec![],
            vec![],
        );
        let mul1 = g.push_group(
            m * k1,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine {
                    base: w1,
                    stride: 1,
                },
                InputRef::Modular {
                    base: inp,
                    stride: 1,
                    modulus: k1,
                },
            ],
        );
        let red1 = g.push_group(
            m,
            DType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: k1,
                reduce_stride: 1,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: mul1,
                stride: k1 as i64,
            }],
        );

        // Activation (elementwise).
        let act = g.push_group(
            m,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Tanh,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: red1,
                stride: 1,
            }],
        );

        // MatMul 2: [K2, M] @ act[M] → output[K2]
        let w2 = g.push_group(
            k2 * m,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.2)),
            vec![],
            vec![],
        );
        let mul2 = g.push_group(
            k2 * m,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine {
                    base: w2,
                    stride: 1,
                },
                InputRef::Modular {
                    base: act,
                    stride: 1,
                    modulus: m,
                },
            ],
        );
        let red2 = g.push_group(
            k2,
            DType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: m,
                reduce_stride: 1,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: mul2,
                stride: m as i64,
            }],
        );

        g.outputs = vec![red2];
        let it = vec![InputTensor {
            tensor_id: GlobalId(0),
            base_id: inp,
            count: k1,
            dtype: DType::F32,
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
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );
        let neg = g.push_group(
            1000,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: data,
                stride: 1,
            }],
        );
        // Scalar reduce: reads ALL 1000 atoms of neg.
        let reduced = g.push_group(
            1,
            DType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: 1000,
                reduce_stride: 1,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: neg,
                stride: 1,
            }],
        );
        // Broadcast reduced to large output.
        let output = g.push_group(
            1000,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Broadcast(reduced)],
        );

        g.outputs = vec![output];
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
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );
        let source = g.push_group(
            40,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: lit,
                stride: 1,
            }],
        );

        // Consumer: 100 atoms reading source via StridedBroadcast(repeat=10).
        // chunk=25, distinct_per_chunk=ceil(25/10)=3, prod_chunk=10.
        // 3 <= 10 → would pass the simple check. But the access is NOT lane-local!
        let consumer = g.push_group(
            100,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::StridedBroadcast {
                base: source,
                stride: 1,
                repeat: 10,
            }],
        );

        g.outputs = vec![consumer];
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
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );
        let neg = g.push_group(
            103,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: lit,
                stride: 1,
            }],
        );
        let exp = g.push_group(
            103,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: neg,
                stride: 1,
            }],
        );

        g.outputs = vec![exp];
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

        let inp = g.add_input_tensor(GlobalId(0), k, DType::F32);
        let weights = g.push_group(
            m * k,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.1)),
            vec![],
            vec![],
        );
        let mul = g.push_group(
            m * k,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine {
                    base: weights,
                    stride: 1,
                },
                InputRef::Modular {
                    base: inp,
                    stride: 1,
                    modulus: k,
                },
            ],
        );
        let reduce = g.push_group(
            m,
            DType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: k,
                reduce_stride: 1,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: mul,
                stride: k as i64,
            }],
        );

        g.outputs = vec![reduce];
        let it = vec![InputTensor {
            tensor_id: GlobalId(0),
            base_id: inp,
            count: k,
            dtype: DType::F32,
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
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );
        let src2 = g.push_group(
            m,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
        );

        // Apply StridedBroadcast with repeat=768.
        let expanded1 = g.push_group(
            n1,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::StridedBroadcast {
                base: src1,
                stride: 1,
                repeat: d1,
            }],
        );

        // Apply StridedBroadcast with repeat=64.
        let expanded2 = g.push_group(
            n2,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::StridedBroadcast {
                base: src2,
                stride: 1,
                repeat: d2,
            }],
        );

        g.outputs = vec![expanded1, expanded2];
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
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );
        let neg = g.push_group(
            n,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: data,
                stride: 1,
            }],
        );

        // Whole group with Explicit input (reverse order).
        let explicit_ids: Vec<AtomId> = (0..n).rev().map(|i| AtomId(neg.0 + i)).collect();
        let reversed = g.push_group(
            n,
            DType::F32,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::Explicit(explicit_ids)],
        );

        // Split output reading from whole.
        let output = g.push_group(
            n,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: reversed,
                stride: 1,
            }],
        );

        g.outputs = vec![output];
        let phases = plan(&g, num_lanes, &[], &g.outputs.clone());
        verify_plan_full(&phases);
    }
}
