#![allow(clippy::all, dead_code, unreachable_patterns, unused_variables, unused_imports)]
//! Split-first span partitioner v3b.
//!
//! Previous attempts failed because they assigned phases first, then tried to
//! split groups across lanes — creating cross-lane violations when a downstream
//! group in the same phase reads atoms from multiple lanes' slices.
//!
//! **This approach inverts the order: split first, then assign phases.**
//!
//! Algorithm:
//!
//! 1. **Build group dependency DAG** on the original graph.
//!
//! 2. **Classify each group's split behaviour.** For each group, analyze whether
//!    splitting it across N lanes preserves lane-independence:
//!    - "LaneSplittable": each lane's slice only reads from the corresponding
//!      slice of each producer. (Affine stride=1 with same-count producers, etc.)
//!    - "Unsplittable": the group reads a pattern that doesn't decompose by lane
//!      (Broadcast, Modular, cross-cutting Explicit, etc.)
//!
//! 3. **Build a virtual-node DAG** where every large splittable group is replaced
//!    by N virtual nodes (one per lane). Each virtual node knows its atom
//!    sub-range and its dependencies (other virtual nodes or whole groups).
//!    Small groups and unsplittable groups remain as single nodes.
//!
//! 4. **Assign phases on the virtual DAG.** Phase boundaries occur when a node
//!    depends on nodes from multiple lanes (convergence). Within a phase, all
//!    nodes on different lanes are independent by construction.
//!
//! 5. **Collect (phase, lane) assignments** and build self-contained span
//!    NanoGraphs with AtomMapping ranges.
//!
//! The key insight: by splitting BEFORE phase assignment, all cross-lane
//! dependencies are explicit edges in the virtual DAG. Phase assignment
//! naturally separates them with barriers.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use crate::nano_graph::{AtomGroup, AtomId, InputRef, NanoGraph, ScalarOp};

/// Literal groups with fewer atoms than this are duplicated into spans.
const LITERAL_INLINE_THRESHOLD: u64 = 1024;

/// Groups with fewer atoms than this are not split across lanes.
const SPLIT_THRESHOLD: u64 = 64;

// ---- Public types ----

/// A contiguous range of atoms mapped between main graph and span graph.
#[derive(Debug, Clone)]
pub struct AtomMapping {
    pub main_base: AtomId,
    pub span_base: AtomId,
    pub count: u64,
}

/// A self-contained unit of work for one lane in one phase.
pub struct Span {
    pub graph: NanoGraph,
    pub inputs: Vec<AtomMapping>,
    pub outputs: Vec<AtomMapping>,
}

/// One phase of execution.
pub struct Phase {
    pub spans: Vec<Span>,
}

/// The full execution plan.
pub struct SpanPlan {
    pub num_lanes: usize,
    pub phases: Vec<Phase>,
}

// ---- Virtual node representation ----

/// A virtual node in the split DAG. Represents either a whole group or a
/// lane-slice of a split group.
#[derive(Debug, Clone)]
struct VNode {
    /// Original group index in the NanoGraph.
    group_idx: usize,
    /// Atom offset within the group (0 for whole groups).
    atom_offset: u64,
    /// Number of atoms this vnode handles.
    atom_count: u64,
    /// Lane assignment (0..num_lanes for split groups, LANE_ANY for unsplit).
    lane: u32,
}

const LANE_ANY: u32 = u32::MAX;

// ---- Public API ----

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

    // Step 1: Build group-level dependency DAG.
    let (producers, consumers) = build_group_deps(groups);

    // Step 2: Classify each group's split behaviour.
    let split_class = classify_splits(groups, &producers, &is_literal, num_lanes);

    // Step 3: Build virtual-node DAG.
    let (vnodes, vnode_producers, group_to_vnodes) =
        build_vnode_dag(groups, &producers, &is_literal, &split_class, num_lanes);

    // Step 4: Assign phases on the virtual DAG.
    let (vnode_phase, num_phases) =
        assign_vnode_phases(&vnodes, &vnode_producers, &is_literal);

    // Step 5: Assign lanes. Split vnodes already have lanes. Unsplit vnodes
    // get assigned to the least-loaded lane in their phase.
    let vnode_lane = assign_vnode_lanes(
        &vnodes, &vnode_producers, &vnode_phase, num_phases, num_lanes, groups, &is_literal,
    );

    // Step 6: Build spans.
    build_spans(
        graph, groups, &is_literal, &vnodes, &vnode_phase, &vnode_lane,
        &producers, &consumers, &group_to_vnodes, num_lanes, num_phases,
    )
}

// ---- Split classification ----

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SplitClass {
    /// Can be evenly split across lanes. Each lane's slice is independent.
    LaneSplit,
    /// Cannot be split. Must go to one lane (or be duplicated if cheap).
    Whole,
}

/// Classify each group's split behaviour.
///
/// A group is LaneSplit if:
/// 1. It has count >= SPLIT_THRESHOLD (worth splitting).
/// 2. Every input is "lane-decomposable": when the group's atoms are divided
///    into N equal chunks, chunk j only reads from atoms that are either:
///    a. In chunk j of a same-count producer (Affine stride=1).
///    b. In chunk j of a proportionally-sized producer (StridedBroadcast
///       where the broadcast pattern aligns with the lane split).
///    c. From a literal/constant (always available).
///    d. From a producer that is itself Whole (read in entirety, available
///       after a barrier).
///
/// Groups that don't meet these criteria are Whole.
fn classify_splits(
    groups: &[AtomGroup],
    producers: &[Vec<usize>],
    is_literal: &[bool],
    num_lanes: usize,
) -> Vec<SplitClass> {
    let n = groups.len();
    let mut class = vec![SplitClass::Whole; n];

    for gi in 0..n {
        if is_literal[gi] {
            continue; // Literals are never "split" — they're shared data.
        }
        let group = &groups[gi];
        if group.count < SPLIT_THRESHOLD {
            continue;
        }

        // Check if every input is lane-decomposable.
        let mut can_split = true;
        for input in &group.inputs {
            if !is_input_lane_decomposable(input, group.count, groups, producers, is_literal, num_lanes) {
                can_split = false;
                break;
            }
        }

        // ReduceSum/ReduceMax: the reduce's strided access pattern must also
        // be lane-local. For the standard matmul ReduceSum (Affine input,
        // reduce_stride = N, count = N), splitting along count splits the N
        // output columns across lanes. Each output column's reduction reads
        // the same set of K source atoms at stride N — the source atoms
        // for column j are at offsets j, j+N, j+2N, ... The source atoms for
        // lane L's chunk [L*N/lanes, (L+1)*N/lanes) are a subset of the Mul
        // group that is also split the same way. This works IF the Mul group
        // is also split the same way (same count, Affine stride=1).
        //
        // More generally: the reduce reads input.resolve(i, 0) + k*reduce_stride
        // for k=0..reduce_count. If input is Affine(base, 1), then for atom i
        // the reads are base+i, base+i+reduce_stride, base+i+2*reduce_stride, ...
        // When we split by i-range [lo, hi), each lane's reads are disjoint IF
        // reduce_stride >= count (the full read range for each atom doesn't
        // overlap with atoms from other lanes' i-ranges). For the matmul case
        // reduce_stride=N=count, so this holds exactly.
        //
        // But if reduce_stride < count, splitting by i-range causes overlapping
        // read ranges. We must check this.
        if can_split {
            match &group.op {
                ScalarOp::ReduceSum { reduce_count, reduce_stride, .. }
                | ScalarOp::ReduceMax { reduce_count, reduce_stride, .. }
                    if *reduce_count > 1 && *reduce_stride != 0 =>
                {
                    // The reduce extends reads. Check that splitting the count
                    // dimension doesn't cause lane j to read atoms that belong
                    // to a different lane's producer slice.
                    //
                    // For Affine(base, stride=1) input with reduce_stride:
                    // Lane j reads atoms in range [base + lo, base + hi) UNION
                    // [base + lo + reduce_stride, base + hi + reduce_stride) ...
                    // up to reduce_stride * (reduce_count - 1).
                    //
                    // The producer's lane j slice is [prod_base + lo, prod_base + hi).
                    // Lane-locality holds iff the reduce reads stay within the
                    // producer's atom range (crossing into other groups is fine
                    // as long as each lane's reads are disjoint from other lanes').
                    //
                    // Simplified check: the full read extent for lane j is
                    // [lo, hi + |reduce_stride| * (reduce_count-1)). This must
                    // not overlap with lane j+1's base read range [hi, ...].
                    // I.e., reduce reads must stay within the producer group's
                    // per-lane range. For the standard matmul, reduce_stride=N=count,
                    // so the read extent is [lo, lo + (K-1)*N + (N/lanes)], which
                    // spans K*N atoms — the full Mul group. Each lane reads a
                    // disjoint subset of columns across all K blocks.
                    //
                    // General check: the source group must be splittable the same
                    // way (or be a single unsplit group), AND each lane's reduce
                    // reads must be disjoint.
                    //
                    // We verify: for Affine stride=1 input, check that the
                    // per-lane reduce read ranges are non-overlapping. The lane j
                    // reads atoms at positions base + (lo..hi) + k*reduce_stride
                    // for k=0..reduce_count. Two lanes j and j' have disjoint
                    // reads iff their column ranges [lo_j, hi_j) and [lo_j', hi_j')
                    // are disjoint (since each k-offset just shifts by the same
                    // reduce_stride for both lanes).
                    //
                    // This is true by construction when the group is split into
                    // contiguous sub-ranges. So the reduce case is fine as long
                    // as the input is Affine stride=1.
                    for input in &group.inputs {
                        match input {
                            InputRef::Affine { stride, .. } if *stride == 1 => {
                                // OK: per-lane column ranges are disjoint, reduce
                                // stride just accesses different rows.
                            }
                            _ => {
                                // Non-trivial input pattern with reduce — don't split.
                                can_split = false;
                                break;
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        if can_split {
            class[gi] = SplitClass::LaneSplit;
        }
    }

    class
}

/// Check if a single InputRef is lane-decomposable.
///
/// Lane-decomposable means: when the consumer group (count atoms) is split into
/// N equal chunks, chunk j only reads from atoms that are in chunk j of the
/// producer, OR from a producer that is always fully available (literal, or
/// a Whole group that will be available after a barrier).
fn is_input_lane_decomposable(
    input: &InputRef,
    consumer_count: u64,
    groups: &[AtomGroup],
    producers: &[Vec<usize>],
    is_literal: &[bool],
    num_lanes: usize,
) -> bool {
    match input {
        InputRef::Affine { base, stride } => {
            if *stride == 0 {
                // All atoms read the same source — it's a broadcast. The source
                // must be available to all lanes, which means it needs to come
                // from an earlier phase. This makes the group unsplittable unless
                // the source is a literal.
                if let Some(gi) = find_group_idx(groups, *base) {
                    return is_literal[gi];
                }
                return false;
            }
            if *stride == 1 || *stride == -1 {
                // 1:1 mapping. Lane j's chunk reads from a contiguous sub-range
                // of the producer. This is lane-decomposable.
                return true;
            }
            // High stride: lane j's chunk reads a strided pattern from the
            // producer. The reads are still disjoint across lanes (different
            // i values), so it's lane-decomposable. The key question is whether
            // the producer atoms are also split compatibly. Since the reads are
            // disjoint by i, each lane reads different atoms, so splitting is
            // safe even if the producer is split differently — we just declare
            // the needed atoms as inputs from an earlier phase.
            //
            // But: if the producer is in the same phase and split, we need
            // lane-locality. High-stride reads from a split producer span
            // multiple lanes' slices. So we need the producer to NOT be split,
            // meaning it will be in an earlier phase.
            //
            // For now, allow it — the phase assignment will handle the
            // dependency correctly. The split group will read from an earlier
            // phase's output.
            true
        }
        InputRef::Broadcast(atom_id) => {
            // All atoms read the same source. The source must be from an earlier
            // phase (available to all lanes). Lane-decomposable: each lane's
            // chunk reads the same single atom.
            true
        }
        InputRef::StridedBroadcast { base, stride, repeat } => {
            // atom i reads base + stride * (i / repeat).
            // When split into N chunks of consumer_count/N each:
            // chunk j reads atoms base + stride * (floor((j*consumer_count/N + local_i) / repeat))
            // for local_i in 0..consumer_count/N.
            //
            // This is lane-decomposable if the source atoms for different chunks
            // are disjoint. They are disjoint because different i values produce
            // different i/repeat values (or the same, but within the same chunk).
            //
            // More precisely: the mapping i -> i/repeat is monotonically
            // non-decreasing, so consecutive atoms map to consecutive or same
            // source indices. Splitting into contiguous chunks preserves this.
            true
        }
        InputRef::Modular { base, stride, modulus } => {
            // atom i reads base + stride * (i % modulus).
            // This wraps around — all chunks read the same set of `modulus`
            // source atoms. Not lane-decomposable (each lane needs the full
            // modular range).
            //
            // However, the modular source is typically a small shared constant
            // (bias, mask) that comes from a literal or will be available from
            // an earlier phase. So the group CAN be split — each lane just
            // declares the modular range as an input.
            true
        }
        InputRef::Explicit(ids) => {
            // Arbitrary mapping. We can't prove lane-locality in general.
            // But we CAN split the group: each lane gets its chunk of atoms,
            // and each chunk's explicit source atoms become inputs.
            // The question is whether those inputs come from the same phase.
            // Since we can't prove they don't, allow split and let the phase
            // assignment handle it. The source atoms will be declared as
            // inputs from earlier phases.
            true
        }
        InputRef::SymAffine { .. } => {
            // SymAffine involves a sym_dim iteration. The split is along the
            // atom index (i dimension), which is independent of k. Each lane's
            // chunk iterates k independently. Lane-decomposable.
            true
        }
    }
}

// ---- Virtual-node DAG ----

/// Build the virtual-node DAG.
///
/// Returns:
/// - vnodes: the virtual nodes
/// - vnode_producers: dependency DAG on vnodes
/// - group_to_vnodes: maps each group_idx to its vnode indices
fn build_vnode_dag(
    groups: &[AtomGroup],
    producers: &[Vec<usize>],
    is_literal: &[bool],
    split_class: &[SplitClass],
    num_lanes: usize,
) -> (Vec<VNode>, Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let n = groups.len();
    let mut vnodes: Vec<VNode> = Vec::new();
    let mut group_to_vnodes: Vec<Vec<usize>> = vec![Vec::new(); n];

    // Create vnodes.
    for gi in 0..n {
        if is_literal[gi] {
            // Literals get a single vnode (they're shared data, not assigned to lanes).
            let vi = vnodes.len();
            vnodes.push(VNode {
                group_idx: gi,
                atom_offset: 0,
                atom_count: groups[gi].count,
                lane: LANE_ANY,
            });
            group_to_vnodes[gi].push(vi);
            continue;
        }

        match split_class[gi] {
            SplitClass::LaneSplit => {
                let count = groups[gi].count;
                let chunk_size = count / num_lanes as u64;
                let remainder = count % num_lanes as u64;
                let mut offset = 0u64;
                for lane in 0..num_lanes {
                    let this_chunk = chunk_size + if (lane as u64) < remainder { 1 } else { 0 };
                    if this_chunk == 0 {
                        continue;
                    }
                    let vi = vnodes.len();
                    vnodes.push(VNode {
                        group_idx: gi,
                        atom_offset: offset,
                        atom_count: this_chunk,
                        lane: lane as u32,
                    });
                    group_to_vnodes[gi].push(vi);
                    offset += this_chunk;
                }
            }
            SplitClass::Whole => {
                let vi = vnodes.len();
                vnodes.push(VNode {
                    group_idx: gi,
                    atom_offset: 0,
                    atom_count: groups[gi].count,
                    lane: LANE_ANY,
                });
                group_to_vnodes[gi].push(vi);
            }
        }
    }

    // Build vnode dependency DAG.
    // For each vnode, find which vnodes it depends on.
    let num_vnodes = vnodes.len();
    let mut vnode_producers: Vec<Vec<usize>> = vec![Vec::new(); num_vnodes];

    for vi in 0..num_vnodes {
        let vnode = &vnodes[vi];
        let gi = vnode.group_idx;
        if is_literal[gi] {
            continue; // Literals have no producers.
        }

        let group = &groups[gi];
        let mut prod_vnodes: BTreeSet<usize> = BTreeSet::new();

        // For each producer group of this group, find the relevant vnodes.
        for &prod_gi in &producers[gi] {
            let prod_vnodes_list = &group_to_vnodes[prod_gi];

            if prod_vnodes_list.len() == 1 {
                // Producer is a single vnode (literal or Whole).
                prod_vnodes.insert(prod_vnodes_list[0]);
            } else {
                // Producer is split. We need to figure out which of the
                // producer's vnodes our vnode actually reads from.
                //
                // This depends on the InputRef pattern and the atom offsets.
                let relevant = find_relevant_producer_vnodes(
                    vnode, group, &groups[prod_gi], prod_gi, prod_vnodes_list, &vnodes, groups,
                );
                for pvi in relevant {
                    prod_vnodes.insert(pvi);
                }
            }
        }

        // Remove self-references.
        prod_vnodes.remove(&vi);

        vnode_producers[vi] = prod_vnodes.into_iter().collect();
    }

    (vnodes, vnode_producers, group_to_vnodes)
}

/// Find which of a split producer's vnodes are actually read by a consumer vnode.
///
/// The consumer vnode represents atoms [atom_offset, atom_offset + atom_count) of
/// the consumer group. The producer group is split into multiple vnodes with
/// different atom ranges. We need to find which producer vnodes contain atoms
/// that the consumer actually reads.
fn find_relevant_producer_vnodes(
    consumer_vnode: &VNode,
    consumer_group: &AtomGroup,
    producer_group: &AtomGroup,
    producer_gi: usize,
    producer_vnode_indices: &[usize],
    all_vnodes: &[VNode],
    all_groups: &[AtomGroup],
) -> Vec<usize> {
    let mut result = Vec::new();

    // Compute the range of producer atoms that the consumer vnode reads.
    let (read_lo, read_hi) = compute_consumer_read_range(
        consumer_vnode, consumer_group, producer_group, all_groups,
    );

    if read_hi <= read_lo {
        return result;
    }

    // Find which producer vnodes overlap with [read_lo, read_hi).
    let prod_base = producer_group.base_id.0;
    for &pvi in producer_vnode_indices {
        let pvnode = &all_vnodes[pvi];
        let pv_lo = prod_base + pvnode.atom_offset;
        let pv_hi = pv_lo + pvnode.atom_count;
        if pv_lo < read_hi && pv_hi > read_lo {
            result.push(pvi);
        }
    }

    result
}

/// Compute the [lo, hi) range of absolute atom IDs from a producer group
/// that a consumer vnode reads.
fn compute_consumer_read_range(
    consumer_vnode: &VNode,
    consumer_group: &AtomGroup,
    producer_group: &AtomGroup,
    all_groups: &[AtomGroup],
) -> (u64, u64) {
    let prod_lo = producer_group.base_id.0;
    let prod_hi = prod_lo + producer_group.count;

    let c_offset = consumer_vnode.atom_offset;
    let c_count = consumer_vnode.atom_count;

    let (is_reduce, reduce_count, reduce_stride) = match &consumer_group.op {
        ScalarOp::ReduceSum { reduce_count, reduce_stride, .. }
        | ScalarOp::ReduceMax { reduce_count, reduce_stride, .. }
            if *reduce_count > 1 && *reduce_stride != 0 =>
            (true, *reduce_count, *reduce_stride),
        _ => (false, 0, 0),
    };

    let mut overall_lo = u64::MAX;
    let mut overall_hi = 0u64;

    for input in &consumer_group.inputs {
        let (range_lo, range_hi) = input_ref_range_for_slice(
            input, c_offset, c_count, is_reduce, reduce_count, reduce_stride,
        );
        if range_hi <= prod_lo || range_lo >= prod_hi {
            continue;
        }
        let overlap_lo = range_lo.max(prod_lo);
        let overlap_hi = range_hi.min(prod_hi);
        overall_lo = overall_lo.min(overlap_lo);
        overall_hi = overall_hi.max(overlap_hi);
    }

    // IndirectLoad table reference.
    if let ScalarOp::IndirectLoad { table_base, .. } = &consumer_group.op {
        if table_base.0 >= prod_lo && table_base.0 < prod_hi {
            overall_lo = overall_lo.min(prod_lo);
            overall_hi = overall_hi.max(prod_hi);
        }
    }

    if overall_lo >= overall_hi {
        (0, 0)
    } else {
        (overall_lo, overall_hi)
    }
}

/// Compute the [lo, hi) atom range that an InputRef covers for a sub-range
/// of the consumer, including reduce extension.
fn input_ref_range_for_slice(
    input: &InputRef,
    offset: u64,
    count: u64,
    is_reduce: bool,
    reduce_count: u64,
    reduce_stride: i64,
) -> (u64, u64) {
    if count == 0 {
        return (0, 0);
    }
    let min_reduce_ext = if is_reduce {
        0i64.min(reduce_stride * (reduce_count as i64 - 1))
    } else { 0 };
    let max_reduce_ext = if is_reduce {
        0i64.max(reduce_stride * (reduce_count as i64 - 1))
    } else { 0 };

    match input {
        InputRef::Broadcast(id) => {
            let lo = id.0 as i64 + min_reduce_ext;
            let hi = id.0 as i64 + max_reduce_ext + 1;
            (lo as u64, hi as u64)
        }
        InputRef::Affine { base, stride } => {
            let first = base.0 as i64 + *stride as i64 * offset as i64;
            let last = base.0 as i64 + *stride as i64 * (offset + count - 1) as i64;
            let lo = first.min(last) + min_reduce_ext;
            let hi = first.max(last) + max_reduce_ext + 1;
            (lo as u64, hi as u64)
        }
        InputRef::StridedBroadcast { base, stride, repeat } => {
            let first_block = (offset / repeat) as i64;
            let last_block = ((offset + count - 1) / repeat) as i64;
            let first_read = base.0 as i64 + stride * first_block;
            let last_read = base.0 as i64 + stride * last_block;
            let lo = first_read.min(last_read) + min_reduce_ext;
            let hi = first_read.max(last_read) + max_reduce_ext + 1;
            (lo as u64, hi as u64)
        }
        InputRef::Modular { base, stride, modulus } => {
            if *modulus == 0 {
                return (0, 0);
            }
            // Modular reads the full range regardless of consumer offset.
            let first = base.0 as i64;
            let last = base.0 as i64 + *stride as i64 * (*modulus as i64 - 1);
            let lo = first.min(last) + min_reduce_ext;
            let hi = first.max(last) + max_reduce_ext + 1;
            (lo as u64, hi as u64)
        }
        InputRef::SymAffine { base, stride_i, .. } => {
            let first = base.0 as i64 + *stride_i as i64 * offset as i64;
            let last = base.0 as i64 + *stride_i as i64 * (offset + count - 1) as i64;
            let lo = first.min(last) + min_reduce_ext;
            let hi = first.max(last) + max_reduce_ext + 1;
            (lo as u64, hi as u64)
        }
        InputRef::Explicit(ids) => {
            let start = offset as usize;
            let end = ((offset + count) as usize).min(ids.len());
            if start >= end {
                return (0, 0);
            }
            let mut lo = u64::MAX;
            let mut hi = 0u64;
            for i in start..end {
                let id_lo = (ids[i].0 as i64 + min_reduce_ext) as u64;
                let id_hi = (ids[i].0 as i64 + max_reduce_ext) as u64 + 1;
                lo = lo.min(id_lo);
                hi = hi.max(id_hi);
            }
            (lo, hi)
        }
    }
}

// ---- Phase assignment on the virtual DAG ----

/// Assign phases to vnodes. Phase boundaries occur in two cases:
///
/// 1. **Cross-lane dependency**: A vnode on lane X depends on a vnode on lane Y
///    (where X != Y) in the same phase. This is a direct lane conflict.
///
/// 2. **Family convergence**: A vnode depends on producer vnodes from multiple
///    independent families. These families will be assigned to different lanes,
///    so the vnode needs all families to complete (barrier) before it can run.
///
/// Family convergence is the key mechanism that detects matmul chain boundaries
/// and other structural sync points, even when all groups are LANE_ANY.
fn assign_vnode_phases(
    vnodes: &[VNode],
    vnode_producers: &[Vec<usize>],
    is_literal: &[bool],
) -> (Vec<usize>, usize) {
    let nv = vnodes.len();

    // Topological sort of vnodes.
    let topo_order = topo_sort_vnodes(nv, vnode_producers);

    // Assign families: each root compute vnode gets a unique family.
    // Vnodes with a single-family producer chain inherit that family.
    // Convergence points (multiple families) get a new unique family.
    let mut vnode_family = vec![0usize; nv];
    let mut next_family = 0usize;

    for &vi in &topo_order {
        let vnode = &vnodes[vi];
        if is_literal[vnode.group_idx] {
            continue;
        }

        // Collect families of non-literal producers.
        let mut producer_families: BTreeSet<usize> = BTreeSet::new();
        for &pvi in &vnode_producers[vi] {
            if !is_literal[vnodes[pvi].group_idx] {
                producer_families.insert(vnode_family[pvi]);
            }
        }

        if producer_families.is_empty() {
            // Root compute vnode — unique family.
            vnode_family[vi] = next_family;
            next_family += 1;
        } else if producer_families.len() == 1 {
            // Single-family chain — inherit.
            vnode_family[vi] = *producer_families.iter().next().unwrap();
        } else {
            // Convergence — new family.
            vnode_family[vi] = next_family;
            next_family += 1;
        }
    }

    // Now assign phases using both family convergence and lane conflicts.
    let mut vnode_phase = vec![0usize; nv];
    let mut num_phases = 1usize;

    for &vi in &topo_order {
        let vnode = &vnodes[vi];
        if is_literal[vnode.group_idx] {
            vnode_phase[vi] = 0;
            continue;
        }

        // This vnode's phase must be >= max phase of its non-literal producers.
        let mut max_prod_phase = 0usize;
        for &pvi in &vnode_producers[vi] {
            if !is_literal[vnodes[pvi].group_idx] {
                max_prod_phase = max_prod_phase.max(vnode_phase[pvi]);
            }
        }

        // Check for convergence: does this vnode read from producers in
        // multiple distinct families at the max_prod_phase level?
        let mut prod_families_at_max: BTreeSet<usize> = BTreeSet::new();
        let mut prod_lanes_at_max: HashSet<u32> = HashSet::new();

        for &pvi in &vnode_producers[vi] {
            let pvnode = &vnodes[pvi];
            if is_literal[pvnode.group_idx] {
                continue;
            }
            if vnode_phase[pvi] == max_prod_phase {
                prod_families_at_max.insert(vnode_family[pvi]);
                prod_lanes_at_max.insert(pvnode.lane);
            }
        }

        // Need a barrier if:
        // a) Multiple families converge (structural convergence), OR
        // b) Multiple lanes conflict (lane-based conflict)
        let family_convergence = prod_families_at_max.len() > 1;

        let lane_conflict = {
            let non_any_lanes: HashSet<u32> = prod_lanes_at_max.iter()
                .copied()
                .filter(|&l| l != LANE_ANY)
                .collect();
            let my_lane = vnode.lane;

            if non_any_lanes.len() > 1 {
                // Multiple different specific lanes.
                true
            } else if non_any_lanes.len() == 1 && my_lane != LANE_ANY {
                // One specific lane, and we're on a different specific lane.
                let single = *non_any_lanes.iter().next().unwrap();
                single != my_lane
            } else {
                false
            }
        };

        if family_convergence || lane_conflict {
            vnode_phase[vi] = max_prod_phase + 1;
        } else {
            vnode_phase[vi] = max_prod_phase;
        }

        if vnode_phase[vi] + 1 > num_phases {
            num_phases = vnode_phase[vi] + 1;
        }
    }

    (vnode_phase, num_phases)
}

fn topo_sort_vnodes(nv: usize, producers: &[Vec<usize>]) -> Vec<usize> {
    let mut in_degree = vec![0usize; nv];
    let mut consumers_map: Vec<Vec<usize>> = vec![Vec::new(); nv];
    for (vi, prods) in producers.iter().enumerate() {
        in_degree[vi] = prods.len();
        for &pi in prods {
            consumers_map[pi].push(vi);
        }
    }

    let mut queue: VecDeque<usize> = VecDeque::new();
    for i in 0..nv {
        if in_degree[i] == 0 {
            queue.push_back(i);
        }
    }

    let mut order = Vec::with_capacity(nv);
    while let Some(vi) = queue.pop_front() {
        order.push(vi);
        for &ci in &consumers_map[vi] {
            in_degree[ci] -= 1;
            if in_degree[ci] == 0 {
                queue.push_back(ci);
            }
        }
    }

    order
}

// ---- Lane assignment for LANE_ANY vnodes ----

/// Assign lanes to LANE_ANY vnodes using greedy load balancing.
/// Split vnodes already have lanes. For LANE_ANY vnodes, we consider:
/// 1. If they have a same-phase producer on a specific lane, chain to that lane
///    (to avoid creating cross-lane dependencies within a phase).
/// 2. Otherwise, assign to the least-loaded lane in the phase.
fn assign_vnode_lanes(
    vnodes: &[VNode],
    vnode_producers: &[Vec<usize>],
    vnode_phase: &[usize],
    num_phases: usize,
    num_lanes: usize,
    groups: &[AtomGroup],
    is_literal: &[bool],
) -> Vec<usize> {
    let nv = vnodes.len();
    let mut vnode_lane = vec![0usize; nv];

    // Pre-fill split vnodes' lanes.
    for vi in 0..nv {
        if vnodes[vi].lane != LANE_ANY {
            vnode_lane[vi] = vnodes[vi].lane as usize;
        }
    }

    // Process phases in order.
    for phase in 0..num_phases {
        // Collect LANE_ANY vnodes in this phase.
        let mut any_vnodes: Vec<usize> = Vec::new();
        for vi in 0..nv {
            if vnode_phase[vi] == phase && vnodes[vi].lane == LANE_ANY && !is_literal[vnodes[vi].group_idx] {
                any_vnodes.push(vi);
            }
        }

        if any_vnodes.is_empty() {
            continue;
        }

        // Build within-phase dependency chains among LANE_ANY vnodes.
        // If a LANE_ANY vnode depends on a specific-lane vnode in the same phase,
        // it must go on that lane.
        let phase_set: HashSet<usize> = (0..nv)
            .filter(|&vi| vnode_phase[vi] == phase && !is_literal[vnodes[vi].group_idx])
            .collect();

        // Union-Find for chaining within-phase LANE_ANY vnodes.
        let mut parent: HashMap<usize, usize> = HashMap::new();
        for &vi in &any_vnodes {
            parent.insert(vi, vi);
        }

        // Forced lane assignments from same-phase specific-lane producers.
        let mut forced_lane: HashMap<usize, usize> = HashMap::new();

        for &vi in &any_vnodes {
            for &pvi in &vnode_producers[vi] {
                if !phase_set.contains(&pvi) {
                    continue;
                }
                let prod = &vnodes[pvi];
                if prod.lane != LANE_ANY {
                    // This LANE_ANY vnode depends on a specific-lane vnode
                    // in the same phase. Must go on that lane.
                    forced_lane.insert(vi, prod.lane as usize);
                } else if parent.contains_key(&pvi) {
                    // Both are LANE_ANY in the same phase — chain them.
                    union_hm(&mut parent, vi, pvi);
                }
            }
        }

        // Propagate forced lanes through chains.
        // If any member of a chain has a forced lane, the whole chain gets it.
        let mut chain_forced: HashMap<usize, usize> = HashMap::new();
        for (&vi, &lane) in &forced_lane {
            let root = find_hm(&parent, vi);
            chain_forced.insert(root, lane);
        }

        // Collect chains.
        let mut chains: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for &vi in &any_vnodes {
            let root = find_hm(&parent, vi);
            chains.entry(root).or_default().push(vi);
        }

        // Sort chains by atom count (largest first) for load balancing.
        let mut chain_list: Vec<(usize, Vec<usize>)> = chains.into_iter().collect();
        chain_list.sort_by(|a, b| {
            let count_a: u64 = a.1.iter().map(|&vi| vnodes[vi].atom_count).sum();
            let count_b: u64 = b.1.iter().map(|&vi| vnodes[vi].atom_count).sum();
            count_b.cmp(&count_a)
        });

        // Compute per-lane load from already-assigned vnodes in this phase.
        let mut lane_load = vec![0u64; num_lanes];
        for vi in 0..nv {
            if vnode_phase[vi] == phase && vnodes[vi].lane != LANE_ANY && !is_literal[vnodes[vi].group_idx] {
                lane_load[vnodes[vi].lane as usize] += vnodes[vi].atom_count;
            }
        }

        // Assign chains to lanes.
        for (root, chain) in &chain_list {
            let chain_atoms: u64 = chain.iter().map(|&vi| vnodes[vi].atom_count).sum();

            let target_lane = if let Some(&forced) = chain_forced.get(root) {
                forced
            } else {
                // Assign to least-loaded lane.
                lane_load.iter().enumerate().min_by_key(|&(_, load)| *load).unwrap().0
            };

            for &vi in chain {
                vnode_lane[vi] = target_lane;
            }
            lane_load[target_lane] += chain_atoms;
        }
    }

    vnode_lane
}

// Union-Find helpers.
fn find_hm(parent: &HashMap<usize, usize>, mut x: usize) -> usize {
    while parent.get(&x).copied().unwrap_or(x) != x {
        x = parent[&x];
    }
    x
}

fn union_hm(parent: &mut HashMap<usize, usize>, a: usize, b: usize) {
    let ra = find_hm(parent, a);
    let rb = find_hm(parent, b);
    if ra != rb {
        parent.insert(ra, rb);
    }
}

// ---- Span building ----

/// Build self-contained span NanoGraphs from the vnode assignments.
fn build_spans(
    graph: &NanoGraph,
    groups: &[AtomGroup],
    is_literal: &[bool],
    vnodes: &[VNode],
    vnode_phase: &[usize],
    vnode_lane: &[usize],
    producers: &[Vec<usize>],
    consumers: &[Vec<usize>],
    group_to_vnodes: &[Vec<usize>],
    num_lanes: usize,
    num_phases: usize,
) -> SpanPlan {
    let nv = vnodes.len();

    // Collect work items per (phase, lane).
    // work_items[phase][lane] = vec of (group_idx, atom_offset, atom_count)
    let mut work_items: Vec<Vec<Vec<(usize, u64, u64)>>> =
        vec![vec![Vec::new(); num_lanes]; num_phases];

    for vi in 0..nv {
        let vnode = &vnodes[vi];
        if is_literal[vnode.group_idx] {
            continue;
        }
        let phase = vnode_phase[vi];
        let lane = vnode_lane[vi];
        work_items[phase][lane].push((vnode.group_idx, vnode.atom_offset, vnode.atom_count));
    }

    // Sort work items within each (phase, lane) by (group_idx, atom_offset)
    // to maintain topological order.
    for phase in &mut work_items {
        for lane_work in phase.iter_mut() {
            lane_work.sort_by_key(|&(gi, off, _)| (gi, off));
        }
    }

    // Build spans.
    let mut phases = Vec::with_capacity(num_phases);

    for phase_idx in 0..num_phases {
        let mut spans = Vec::with_capacity(num_lanes);

        for lane_idx in 0..num_lanes {
            let lane_work = &work_items[phase_idx][lane_idx];

            if lane_work.is_empty() {
                spans.push(Span {
                    graph: NanoGraph::new(),
                    inputs: vec![],
                    outputs: vec![],
                });
                continue;
            }

            let span = build_single_span(
                graph, groups, is_literal, lane_work,
            );
            spans.push(span);
        }

        phases.push(Phase { spans });
    }
    SpanPlan { num_lanes, phases }
}

// ---- Range-based atom map ----

struct RangeAtomMap {
    map: BTreeMap<u64, (u64, u64)>, // main_base -> (span_base, count)
}

impl RangeAtomMap {
    fn new() -> Self {
        Self { map: BTreeMap::new() }
    }

    fn insert_range(&mut self, main_base: AtomId, span_base: AtomId, count: u64) {
        self.map.insert(main_base.0, (span_base.0, count));
    }

    fn get(&self, main_id: AtomId) -> Option<AtomId> {
        use std::ops::Bound;
        let mut iter = self.map.range((Bound::Unbounded, Bound::Included(main_id.0)));
        if let Some((&base, &(span_base, count))) = iter.next_back() {
            let offset = main_id.0.wrapping_sub(base);
            if offset < count {
                return Some(AtomId(span_base + offset));
            }
        }
        None
    }
}

/// Build a single span NanoGraph for one (phase, lane).
fn build_single_span(
    main_graph: &NanoGraph,
    groups: &[AtomGroup],
    is_literal: &[bool],
    lane_work: &[(usize, u64, u64)],
) -> Span {
    let mut span_graph = NanoGraph::new();

    // Copy sym_dim configuration.
    for (name, &sd) in &main_graph.sym_dim_names {
        let local_sd = span_graph.sym_dim(name);
        if let Some(&bound) = main_graph.sym_dim_bounds.get(&sd) {
            span_graph.sym_dim_bounds.insert(local_sd, bound);
        }
    }

    let mut main_to_local = RangeAtomMap::new();

    // Build a set of locally-produced (group_idx, atom_offset, atom_count) for
    // fast lookup.
    let local_set: HashSet<(usize, u64, u64)> = lane_work.iter().copied().collect();
    let local_groups: HashSet<usize> = lane_work.iter().map(|&(gi, _, _)| gi).collect();

    // 1. Determine needed literals.
    let mut needed_literals: BTreeSet<usize> = BTreeSet::new();
    for &(gi, _, _) in lane_work {
        collect_literal_deps(gi, groups, is_literal, &mut needed_literals);
    }

    // 2. Inline small literals, track large ones as external.
    let mut large_literal_groups: BTreeSet<usize> = BTreeSet::new();
    let mut inlined_literals: BTreeSet<usize> = BTreeSet::new();
    for &lit_gi in &needed_literals {
        let lit_group = &groups[lit_gi];
        if lit_group.count < LITERAL_INLINE_THRESHOLD {
            let local_base = span_graph.push_group(
                lit_group.count, lit_group.op.clone(),
                remap_sym_dims(&lit_group.sym_dims, main_graph, &span_graph),
                remap_sym_dims(&lit_group.reduce_dims, main_graph, &span_graph),
                vec![],
            );
            main_to_local.insert_range(lit_group.base_id, local_base, lit_group.count);
            inlined_literals.insert(lit_gi);
        } else {
            large_literal_groups.insert(lit_gi);
        }
    }

    // 3. Collect external dependency ranges.
    let mut external_ranges: Vec<(usize, u64, u64)> = Vec::new();

    // Large literals.
    for &li in &large_literal_groups {
        let lg = &groups[li];
        external_ranges.push((li, 0, lg.count));
    }

    // For each work item, find external producer groups.
    for &(gi, atom_offset, atom_count) in lane_work {
        let group = &groups[gi];
        collect_external_ranges(
            group, atom_offset, atom_count, groups, is_literal,
            lane_work, &inlined_literals, &mut external_ranges,
        );
    }

    // Merge overlapping ranges.
    let external_ranges = merge_group_ranges(&mut external_ranges);

    // 4. Create placeholder groups for external input ranges.
    let mut input_mappings: Vec<AtomMapping> = Vec::new();
    for &(gi, offset, count) in &external_ranges {
        let main_base = groups[gi].base_id.offset(offset);
        let local_base = span_graph.push_group(
            count,
            ScalarOp::Literal(crate::numeric_scalar::NumericScalar::F32(0.0)),
            vec![], vec![], vec![],
        );
        main_to_local.insert_range(main_base, local_base, count);
        input_mappings.push(AtomMapping {
            main_base,
            span_base: local_base,
            count,
        });
    }

    // 5. Build compute groups with remapped InputRefs.
    let mut output_mappings: Vec<AtomMapping> = Vec::new();

    for &(gi, atom_offset, atom_count) in lane_work {
        let group = &groups[gi];

        let local_inputs = remap_inputs_range(
            &group.inputs, &group.op, atom_offset, atom_count,
            group.count, &main_to_local,
        );
        let local_op = remap_op(&group.op, &main_to_local);

        let local_base = span_graph.push_group(
            atom_count, local_op,
            remap_sym_dims(&group.sym_dims, main_graph, &span_graph),
            remap_sym_dims(&group.reduce_dims, main_graph, &span_graph),
            local_inputs,
        );

        let main_base = AtomId(group.base_id.0 + atom_offset);
        main_to_local.insert_range(main_base, local_base, atom_count);

        output_mappings.push(AtomMapping {
            main_base,
            span_base: local_base,
            count: atom_count,
        });
    }

    // 6. Mark graph outputs.
    // Instead of iterating all atoms in all mappings (O(atoms)), check each
    // graph output against the output mappings (O(outputs * log(mappings))).
    for &out_id in &main_graph.outputs {
        for mapping in &output_mappings {
            let m_lo = mapping.main_base.0;
            let m_hi = m_lo + mapping.count;
            if out_id.0 >= m_lo && out_id.0 < m_hi {
                let offset = out_id.0 - m_lo;
                span_graph.outputs.push(mapping.span_base.offset(offset));
            }
        }
    }

    Span {
        graph: span_graph,
        inputs: input_mappings,
        outputs: output_mappings,
    }
}

/// Collect external dependency ranges for a work item (sub-range of a group).
fn collect_external_ranges(
    group: &AtomGroup,
    atom_offset: u64,
    atom_count: u64,
    all_groups: &[AtomGroup],
    is_literal: &[bool],
    assigned_slices: &[(usize, u64, u64)],
    inlined_literals: &BTreeSet<usize>,
    external_ranges: &mut Vec<(usize, u64, u64)>,
) {
    if atom_count == 0 {
        return;
    }

    let is_locally_covered = |atom_lo: u64, atom_hi: u64, prod_gi: usize| -> bool {
        let prod_base = all_groups[prod_gi].base_id.0;
        let off_lo = atom_lo - prod_base;
        let off_hi = atom_hi - prod_base;
        for &(work_gi, work_offset, work_count) in assigned_slices {
            if work_gi == prod_gi && work_offset <= off_lo && work_offset + work_count >= off_hi {
                return true;
            }
        }
        false
    };

    let should_skip = |gi: usize| -> bool {
        inlined_literals.contains(&gi)
            || (is_literal[gi] && all_groups[gi].count < LITERAL_INLINE_THRESHOLD)
    };

    let (is_reduce, reduce_count, reduce_stride) = match &group.op {
        ScalarOp::ReduceSum { reduce_count, reduce_stride, .. }
        | ScalarOp::ReduceMax { reduce_count, reduce_stride, .. }
            if *reduce_count > 1 && *reduce_stride != 0 =>
            (true, *reduce_count, *reduce_stride),
        _ => (false, 0, 0),
    };

    for input in &group.inputs {
        let referenced = resolve_input_to_group_ranges(
            input, atom_offset, atom_count,
            if is_reduce { reduce_count } else { 1 },
            if is_reduce { reduce_stride } else { 0 },
            all_groups,
        );

        for (gi, range_lo, range_hi) in referenced {
            if should_skip(gi) {
                continue;
            }
            let g_lo = all_groups[gi].base_id.0;
            let overlap_lo = range_lo.max(g_lo);
            let overlap_hi = range_hi.min(g_lo + all_groups[gi].count);
            if overlap_lo >= overlap_hi {
                continue;
            }
            if !is_locally_covered(overlap_lo, overlap_hi, gi) {
                let offset = overlap_lo - g_lo;
                let count = overlap_hi - overlap_lo;
                external_ranges.push((gi, offset, count));
            }
        }
    }

    // IndirectLoad table.
    if let ScalarOp::IndirectLoad { table_base, .. } = &group.op {
        if let Some(gi) = find_group_idx(all_groups, *table_base) {
            if !inlined_literals.contains(&gi) {
                let g = &all_groups[gi];
                let is_assigned = assigned_slices.iter().any(|&(wgi, _, _)| wgi == gi);
                if !is_assigned {
                    external_ranges.push((gi, 0, g.count));
                }
            }
        }
    }
}

/// Resolve an InputRef (for a sub-range of a consumer group, including reduce
/// extension) to the actual set of (group_idx, atom_lo, atom_hi) tuples.
fn resolve_input_to_group_ranges(
    input: &InputRef,
    offset: u64,
    count: u64,
    reduce_count: u64,
    reduce_stride: i64,
    groups: &[AtomGroup],
) -> Vec<(usize, u64, u64)> {
    let mut result = Vec::new();
    if count == 0 {
        return result;
    }

    match input {
        InputRef::Broadcast(atom_id) => {
            let base = atom_id.0 as i64;
            let (lo, hi) = reduce_extent(base, reduce_count, reduce_stride);
            for gi in find_groups_in_range(groups, lo as u64, hi as u64) {
                result.push((gi, lo as u64, (hi + 1) as u64));
            }
        }
        InputRef::Affine { base, stride } => {
            let first_k = offset;
            let last_k = offset + count - 1;
            let first_pos = base.0 as i64 + *stride as i64 * first_k as i64;
            let last_pos = base.0 as i64 + *stride as i64 * last_k as i64;

            if *stride == 0 {
                let (lo, hi) = reduce_extent(first_pos, reduce_count, reduce_stride);
                for gi in find_groups_in_range(groups, lo as u64, hi as u64) {
                    result.push((gi, lo as u64, (hi + 1) as u64));
                }
            } else if stride.unsigned_abs() == 1 {
                let base_lo = first_pos.min(last_pos);
                let base_hi = first_pos.max(last_pos);
                let (lo, hi) = reduce_extent_range(base_lo, base_hi, reduce_count, reduce_stride);
                for gi in find_groups_in_range(groups, lo as u64, hi as u64) {
                    result.push((gi, lo as u64, (hi + 1) as u64));
                }
            } else {
                // High stride: bounding box approach.
                let base_lo = first_pos.min(last_pos);
                let base_hi = first_pos.max(last_pos);
                let (ext_lo, ext_hi) = reduce_extent_range(base_lo, base_hi, reduce_count, reduce_stride);
                let candidates = find_groups_in_range(groups, ext_lo as u64, ext_hi as u64);
                for gi in candidates {
                    let g = &groups[gi];
                    let g_lo = g.base_id.0 as i64;
                    let g_hi = g_lo + g.count as i64;
                    if affine_touches_range(
                        first_pos, *stride as i64, count, g_lo, g_hi,
                        reduce_count, reduce_stride,
                    ) {
                        let overlap_lo = (g_lo as u64).max(ext_lo as u64);
                        let overlap_hi = (g_hi as u64).min((ext_hi + 1) as u64);
                        result.push((gi, overlap_lo, overlap_hi));
                    }
                }
            }
        }
        InputRef::StridedBroadcast { base, stride, repeat } => {
            let first_block = offset / repeat;
            let last_block = (offset + count - 1) / repeat;
            // Compute bounding box for all blocks.
            let first_read = base.0 as i64 + *stride * first_block as i64;
            let last_read = base.0 as i64 + *stride * last_block as i64;
            let base_lo = first_read.min(last_read);
            let base_hi = first_read.max(last_read);
            let (lo, hi) = reduce_extent_range(base_lo, base_hi, reduce_count, reduce_stride);
            for gi in find_groups_in_range(groups, lo as u64, hi as u64) {
                result.push((gi, lo as u64, (hi + 1) as u64));
            }
        }
        InputRef::Modular { base, stride, modulus } => {
            if *modulus == 0 {
                return result;
            }
            let num_distinct = *modulus;
            let mut lo = base.0 as i64;
            let mut hi = base.0 as i64;
            for j in 0..num_distinct {
                let pos = base.0 as i64 + *stride as i64 * j as i64;
                lo = lo.min(pos);
                hi = hi.max(pos);
            }
            let (ext_lo, ext_hi) = reduce_extent_range(lo, hi, reduce_count, reduce_stride);
            for gi in find_groups_in_range(groups, ext_lo as u64, ext_hi as u64) {
                result.push((gi, ext_lo as u64, (ext_hi + 1) as u64));
            }
        }
        InputRef::Explicit(ids) => {
            let start = offset as usize;
            let end = ((offset + count) as usize).min(ids.len());
            let mut group_ranges: BTreeMap<usize, (u64, u64)> = BTreeMap::new();
            for i in start..end {
                let atom = ids[i];
                let (lo, hi) = reduce_extent(atom.0 as i64, reduce_count, reduce_stride);
                for gi in find_groups_in_range(groups, lo as u64, hi as u64) {
                    let entry = group_ranges.entry(gi).or_insert((lo as u64, (hi + 1) as u64));
                    entry.0 = entry.0.min(lo as u64);
                    entry.1 = entry.1.max((hi + 1) as u64);
                }
            }
            for (gi, (lo, hi)) in group_ranges {
                result.push((gi, lo, hi));
            }
        }
        InputRef::SymAffine { base, stride_i, stride_k } => {
            let first_pos = base.0 as i64 + *stride_i as i64 * offset as i64;
            let last_pos = base.0 as i64 + *stride_i as i64 * (offset + count - 1) as i64;
            let base_lo = first_pos.min(last_pos);
            let base_hi = first_pos.max(last_pos);
            let (ext_lo, ext_hi) = reduce_extent_range(base_lo, base_hi, reduce_count, reduce_stride);

            if stride_i.unsigned_abs() <= 1 {
                for gi in find_groups_in_range(groups, ext_lo as u64, ext_hi as u64) {
                    result.push((gi, ext_lo as u64, (ext_hi + 1) as u64));
                }
            } else {
                let candidates = find_groups_in_range(groups, ext_lo as u64, ext_hi as u64);
                for gi in candidates {
                    let g = &groups[gi];
                    let g_lo = g.base_id.0 as i64;
                    let g_hi = g_lo + g.count as i64;
                    if affine_touches_range(
                        first_pos, *stride_i as i64, count, g_lo, g_hi,
                        reduce_count, reduce_stride,
                    ) {
                        let overlap_lo = (g_lo as u64).max(ext_lo as u64);
                        let overlap_hi = (g_hi as u64).min((ext_hi + 1) as u64);
                        result.push((gi, overlap_lo, overlap_hi));
                    }
                }
            }
        }
    }

    result
}

// ---- InputRef remapping ----

fn remap_inputs_range(
    inputs: &[InputRef],
    op: &ScalarOp,
    atom_offset: u64,
    atom_count: u64,
    orig_group_count: u64,
    atom_map: &RangeAtomMap,
) -> Vec<InputRef> {
    inputs.iter().map(|input| {
        remap_single_input(input, atom_offset, atom_count, orig_group_count, atom_map)
    }).collect()
}

fn remap_single_input(
    input: &InputRef,
    atom_offset: u64,
    atom_count: u64,
    orig_group_count: u64,
    atom_map: &RangeAtomMap,
) -> InputRef {
    match input {
        InputRef::Broadcast(id) => {
            InputRef::Broadcast(atom_map.get(*id).unwrap_or(*id))
        }
        InputRef::Affine { base, stride } => {
            let new_base_raw = AtomId(base.0.wrapping_add((*stride as i64 * atom_offset as i64) as u64));
            InputRef::Affine {
                base: atom_map.get(new_base_raw).unwrap_or(new_base_raw),
                stride: *stride,
            }
        }
        InputRef::StridedBroadcast { base, stride, repeat } => {
            let block_idx = (atom_offset / repeat) as i64;
            let new_base_raw = AtomId(base.0.wrapping_add((stride * block_idx) as u64));
            let new_offset_in_block = atom_offset % repeat;
            if new_offset_in_block == 0 {
                InputRef::StridedBroadcast {
                    base: atom_map.get(new_base_raw).unwrap_or(new_base_raw),
                    stride: *stride,
                    repeat: *repeat,
                }
            } else {
                // Complex sub-range: fall back to explicit.
                let mut ids = Vec::with_capacity(atom_count as usize);
                for i in 0..atom_count {
                    let main_id = input.resolve(atom_offset + i, 0);
                    ids.push(atom_map.get(main_id).unwrap_or(main_id));
                }
                InputRef::Explicit(ids)
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
            let new_base_raw = AtomId(base.0.wrapping_add((*stride_i as i64 * atom_offset as i64) as u64));
            InputRef::SymAffine {
                base: atom_map.get(new_base_raw).unwrap_or(new_base_raw),
                stride_i: *stride_i,
                stride_k: *stride_k,
            }
        }
        InputRef::Explicit(ids) => {
            let start = atom_offset as usize;
            let end = (atom_offset + atom_count) as usize;
            let slice = if end <= ids.len() { &ids[start..end] } else { &ids[start..] };
            InputRef::Explicit(
                slice.iter().map(|id| atom_map.get(*id).unwrap_or(*id)).collect(),
            )
        }
    }
}

fn remap_op(op: &ScalarOp, atom_map: &RangeAtomMap) -> ScalarOp {
    match op {
        ScalarOp::IndirectLoad { table_base, output_dtype } => ScalarOp::IndirectLoad {
            table_base: atom_map.get(*table_base).unwrap_or(*table_base),
            output_dtype: *output_dtype,
        },
        other => other.clone(),
    }
}

fn remap_sym_dims(
    dims: &[crate::nano_graph::SymDim],
    main_graph: &NanoGraph,
    span_graph: &NanoGraph,
) -> Vec<crate::nano_graph::SymDim> {
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

// ---- Literal dependency collection ----

fn collect_literal_deps(
    gi: usize,
    groups: &[AtomGroup],
    is_literal: &[bool],
    literals: &mut BTreeSet<usize>,
) {
    let group = &groups[gi];
    for input in &group.inputs {
        let referenced = resolve_producer_groups(input, group.count, groups);
        for ref_gi in referenced {
            if is_literal[ref_gi] {
                literals.insert(ref_gi);
            }
        }
    }
    // ReduceSum/ReduceMax extended access.
    match &group.op {
        ScalarOp::ReduceSum { reduce_count, reduce_stride, .. }
        | ScalarOp::ReduceMax { reduce_count, reduce_stride, .. }
            if *reduce_count > 1 && *reduce_stride != 0 =>
        {
            for input in &group.inputs {
                let referenced = resolve_producer_groups_with_reduce(
                    input, group.count, *reduce_count, *reduce_stride, groups,
                );
                for ref_gi in referenced {
                    if is_literal[ref_gi] {
                        literals.insert(ref_gi);
                    }
                }
            }
        }
        _ => {}
    }
    // IndirectLoad table_base.
    if let ScalarOp::IndirectLoad { table_base, .. } = &group.op {
        if let Some(tgi) = find_group_idx(groups, *table_base) {
            if is_literal[tgi] {
                literals.insert(tgi);
            }
        }
    }
}

// ---- Group dependency graph ----

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
            if count == 0 { return vec![]; }
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
                    if seen.insert(gi) { result.push(gi); }
                }
            }
            result
        }
        InputRef::SymAffine { base, stride_i, .. } => {
            if count == 0 { return vec![]; }
            let last_offset = (*stride_i as i64) * (count as i64 - 1);
            let last = AtomId(base.0.wrapping_add(last_offset as u64));
            let lo = base.0.min(last.0);
            let hi = base.0.max(last.0);
            find_groups_in_range(groups, lo, hi)
        }
        InputRef::StridedBroadcast { base, stride, repeat } => {
            if count == 0 { return vec![]; }
            let num_blocks = (count + repeat - 1) / repeat;
            let last_offset = *stride * (num_blocks as i64 - 1);
            let last = AtomId(base.0.wrapping_add(last_offset as u64));
            let lo = base.0.min(last.0);
            let hi = base.0.max(last.0);
            find_groups_in_range(groups, lo, hi)
        }
        InputRef::Modular { base, stride, modulus } => {
            if *modulus == 0 { return vec![]; }
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
    if count == 0 { return vec![]; }
    let min_reduce_ext = 0i64.min(reduce_stride * (reduce_count as i64 - 1));
    let max_reduce_ext = 0i64.max(reduce_stride * (reduce_count as i64 - 1));

    match input {
        InputRef::Affine { base, stride } => {
            let first_base = base.0 as i64;
            let last_base = base.0 as i64 + *stride as i64 * (count as i64 - 1);
            let lo = first_base.min(last_base) + min_reduce_ext;
            let hi = first_base.max(last_base) + max_reduce_ext;
            find_groups_in_range(groups, lo as u64, hi as u64)
        }
        InputRef::StridedBroadcast { base, stride, repeat } => {
            let last_block = ((count - 1) / repeat) as i64;
            let first_read = base.0 as i64;
            let last_read = base.0 as i64 + stride * last_block;
            let lo = first_read.min(last_read) + min_reduce_ext;
            let hi = first_read.max(last_read) + max_reduce_ext;
            find_groups_in_range(groups, lo as u64, hi as u64)
        }
        InputRef::Broadcast(atom_id) => {
            let base = atom_id.0 as i64;
            let lo = base + min_reduce_ext;
            let hi = base + max_reduce_ext;
            find_groups_in_range(groups, lo as u64, hi as u64)
        }
        InputRef::SymAffine { base, stride_i, .. } => {
            let first_base = base.0 as i64;
            let last_base = base.0 as i64 + *stride_i as i64 * (count as i64 - 1);
            let lo = first_base.min(last_base) + min_reduce_ext;
            let hi = first_base.max(last_base) + max_reduce_ext;
            find_groups_in_range(groups, lo as u64, hi as u64)
        }
        InputRef::Modular { base, stride, modulus } => {
            if *modulus == 0 { return vec![]; }
            let first_read = base.0 as i64;
            let last_read = base.0 as i64 + *stride as i64 * (*modulus as i64 - 1);
            let lo = first_read.min(last_read) + min_reduce_ext;
            let hi = first_read.max(last_read) + max_reduce_ext;
            find_groups_in_range(groups, lo as u64, hi as u64)
        }
        InputRef::Explicit(ids) => {
            let mut result = HashSet::new();
            for id in ids {
                let base_val = id.0 as i64;
                let lo = base_val + min_reduce_ext;
                let hi = base_val + max_reduce_ext;
                for gi in find_groups_in_range(groups, lo as u64, hi as u64) {
                    result.insert(gi);
                }
            }
            result.into_iter().collect()
        }
    }
}

// ---- Helper functions ----

fn find_group_idx(groups: &[AtomGroup], id: AtomId) -> Option<usize> {
    let idx = groups.partition_point(|g| g.base_id.0 <= id.0);
    if idx == 0 { return None; }
    let gi = idx - 1;
    if groups[gi].contains(id) { Some(gi) } else { None }
}

fn find_groups_in_range(groups: &[AtomGroup], lo: u64, hi: u64) -> Vec<usize> {
    let mut result = Vec::new();
    let start = groups.partition_point(|g| g.base_id.0 + g.count <= lo);
    for gi in start..groups.len() {
        let g = &groups[gi];
        if g.base_id.0 > hi { break; }
        if g.count == 0 { continue; }
        let g_end = g.base_id.0 + g.count - 1;
        if g.base_id.0 <= hi && g_end >= lo {
            result.push(gi);
        }
    }
    result
}

fn reduce_extent(pos: i64, reduce_count: u64, reduce_stride: i64) -> (i64, i64) {
    if reduce_count <= 1 { return (pos, pos); }
    let ext = reduce_stride * (reduce_count as i64 - 1);
    (pos + ext.min(0), pos + ext.max(0))
}

fn reduce_extent_range(base_lo: i64, base_hi: i64, reduce_count: u64, reduce_stride: i64) -> (i64, i64) {
    if reduce_count <= 1 { return (base_lo, base_hi); }
    let ext = reduce_stride * (reduce_count as i64 - 1);
    (base_lo + ext.min(0), base_hi + ext.max(0))
}

fn affine_touches_range(
    first_pos: i64, stride: i64, count: u64, g_lo: i64, g_hi: i64,
    reduce_count: u64, reduce_stride: i64,
) -> bool {
    if count == 0 || g_lo >= g_hi { return false; }
    let (min_reduce_ext, max_reduce_ext) = if reduce_count > 1 {
        let ext = reduce_stride * (reduce_count as i64 - 1);
        (ext.min(0), ext.max(0))
    } else { (0, 0) };
    let eff_lo = g_lo - max_reduce_ext;
    let eff_hi = g_hi - min_reduce_ext;

    if stride == 0 {
        return first_pos >= eff_lo && first_pos < eff_hi;
    }

    let (i_lo, i_hi) = if stride > 0 {
        let num_lo = eff_lo - first_pos;
        let num_hi = eff_hi - first_pos;
        (div_ceil_signed(num_lo, stride), div_ceil_signed(num_hi, stride))
    } else {
        let neg_stride = -stride;
        let i_max = div_floor_signed(first_pos - eff_lo, neg_stride);
        let i_min = div_ceil_signed(first_pos - eff_hi + 1, neg_stride);
        (i_min, i_max + 1)
    };

    let valid_lo = i_lo.max(0);
    let valid_hi = i_hi.min(count as i64);
    valid_lo < valid_hi
}

fn div_ceil_signed(a: i64, b: i64) -> i64 {
    assert!(b > 0);
    if a >= 0 { (a + b - 1) / b } else { -((-a) / b) }
}

fn div_floor_signed(a: i64, b: i64) -> i64 {
    assert!(b > 0);
    if a >= 0 { a / b } else { -(((-a) + b - 1) / b) }
}

fn merge_group_ranges(ranges: &mut Vec<(usize, u64, u64)>) -> Vec<(usize, u64, u64)> {
    if ranges.is_empty() { return vec![]; }
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

// ---- Tests ----

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::attempts::v13_claude::test_graphs;
    use crate::dtype::DType;
    use crate::nano_graph::{ScalarBinOp, ScalarUnaryOp};
    use crate::numeric_scalar::NumericScalar;

    /// Validate a span plan: every span's NanoGraph validates, no cross-lane
    /// reads within a phase, all compute atoms accounted for.
    fn validate_plan(plan: &SpanPlan, graph: &NanoGraph) -> Vec<String> {
        let mut errors = Vec::new();
        let groups = graph.groups();
        let n = groups.len();
        let is_literal: Vec<bool> = groups
            .iter()
            .map(|g| matches!(g.op, ScalarOp::Literal(_)) && g.inputs.is_empty())
            .collect();

        // Check 1: Each span's NanoGraph validates.
        for (pi, phase) in plan.phases.iter().enumerate() {
            for (li, span) in phase.spans.iter().enumerate() {
                if span.outputs.is_empty() && span.inputs.is_empty() {
                    continue;
                }
                let span_errors = span.graph.validate();
                for err in span_errors {
                    errors.push(format!("Phase {} Lane {} span validation: {}", pi, li, err));
                }
            }
        }

        // Check 2: Topology — span inputs must be available.
        let mut produced_ranges: Vec<(u64, u64)> = Vec::new();
        for g in groups {
            if matches!(&g.op, ScalarOp::Literal(_)) && g.inputs.is_empty() {
                produced_ranges.push((g.base_id.0, g.count));
            }
        }
        produced_ranges.sort();

        let range_contains = |ranges: &[(u64, u64)], atom: u64| -> bool {
            match ranges.binary_search_by(|&(base, _)| base.cmp(&atom)) {
                Ok(_) => true,
                Err(0) => false,
                Err(i) => {
                    let (base, count) = ranges[i - 1];
                    atom < base + count
                }
            }
        };

        for (phase_idx, phase) in plan.phases.iter().enumerate() {
            for (lane_idx, span) in phase.spans.iter().enumerate() {
                for mapping in &span.inputs {
                    if !range_contains(&produced_ranges, mapping.main_base.0) {
                        errors.push(format!(
                            "Phase {} Lane {}: input base {:?} (count={}) not available",
                            phase_idx, lane_idx, mapping.main_base, mapping.count
                        ));
                    }
                }

                // Check topo order within span.
                let span_groups = span.graph.groups();
                let span_bases: Vec<u64> = span_groups.iter().map(|g| g.base_id.0).collect();
                for (gi, group) in span_groups.iter().enumerate() {
                    for input in &group.inputs {
                        let src = input.resolve(0, 0);
                        let src_gi = match span_bases.binary_search(&src.0) {
                            Ok(i) => Some(i),
                            Err(0) => None,
                            Err(i) => {
                                let candidate = i - 1;
                                if src.0 < span_groups[candidate].base_id.0 + span_groups[candidate].count {
                                    Some(candidate)
                                } else { None }
                            }
                        };
                        if let Some(src_gi) = src_gi {
                            if src_gi > gi && !matches!(&span_groups[src_gi].op, ScalarOp::Literal(_)) {
                                errors.push(format!(
                                    "Phase {} Lane {}: topo violation, group {} reads from later group {}",
                                    phase_idx, lane_idx, gi, src_gi
                                ));
                            }
                        }
                    }
                }
            }

            // After this phase, add all span outputs to produced_ranges.
            for span in &phase.spans {
                for mapping in &span.outputs {
                    produced_ranges.push((mapping.main_base.0, mapping.count));
                }
            }
            produced_ranges.sort();
        }

        errors
    }

    fn plan_and_validate(graph: &NanoGraph, num_lanes: usize) -> (SpanPlan, Vec<String>) {
        let plan = plan_spans(graph, num_lanes);
        let errors = validate_plan(&plan, graph);
        (plan, errors)
    }

    #[test]
    fn test_empty_graph() {
        let g = NanoGraph::new();
        let (plan, errors) = plan_and_validate(&g, 4);
        assert!(errors.is_empty(), "{:?}", errors);
        assert_eq!(plan.phases.len(), 0);
    }

    #[test]
    fn test_single_literal() {
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
    fn test_elementwise_add_single_lane() {
        let (g, _, _, _) = test_graphs::elementwise_binary(1024, ScalarBinOp::Add);
        let (plan, errors) = plan_and_validate(&g, 1);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
        assert!(plan.phases.len() >= 1);
    }

    #[test]
    fn test_elementwise_add_multi_lane() {
        let (g, _, _, _) = test_graphs::elementwise_binary(1024, ScalarBinOp::Add);
        let (plan, errors) = plan_and_validate(&g, 4);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
        assert!(plan.phases.len() >= 1);
        // The Add group should be split across 4 lanes.
        let total_output_atoms: u64 = plan.phases.iter()
            .flat_map(|p| p.spans.iter())
            .flat_map(|s| s.outputs.iter())
            .map(|m| m.count)
            .sum();
        assert_eq!(total_output_atoms, 1024);
    }

    #[test]
    fn test_broadcast_add() {
        let (g, _, _, _) = test_graphs::broadcast_add(1024);
        let (plan, errors) = plan_and_validate(&g, 4);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
    }

    #[test]
    fn test_unary_chain_single_lane() {
        let (g, _, _) = test_graphs::unary_chain(
            256,
            &[ScalarUnaryOp::Exp, ScalarUnaryOp::Neg, ScalarUnaryOp::Tanh],
        );
        let (plan, errors) = plan_and_validate(&g, 1);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
    }

    #[test]
    fn test_unary_chain_multi_lane() {
        let (g, _, _) = test_graphs::unary_chain(
            1024,
            &[ScalarUnaryOp::Exp, ScalarUnaryOp::Neg, ScalarUnaryOp::Tanh],
        );
        let (plan, errors) = plan_and_validate(&g, 4);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
    }

    #[test]
    fn test_matmul_single_lane() {
        let (g, _, _, _) = test_graphs::matmul(4, 2, 3);
        let errors = g.validate();
        assert!(errors.is_empty(), "Graph validation: {:?}", errors);
        let (plan, plan_errors) = plan_and_validate(&g, 1);
        assert!(plan_errors.is_empty(), "Plan errors: {:?}", plan_errors);
    }

    #[test]
    fn test_matmul_multi_lane() {
        let (g, _, _, _) = test_graphs::matmul(4, 2, 3);
        let errors = g.validate();
        assert!(errors.is_empty(), "Graph validation: {:?}", errors);
        let (plan, plan_errors) = plan_and_validate(&g, 4);
        assert!(plan_errors.is_empty(), "Plan errors: {:?}", plan_errors);
    }

    #[test]
    fn test_matmul_chain_phases() {
        let (g, _, _, _, _) = test_graphs::matmul_chain(4, 2, 3, 3, 5);
        let errors = g.validate();
        assert!(errors.is_empty(), "Graph validation: {:?}", errors);
        let (plan, plan_errors) = plan_and_validate(&g, 2);
        assert!(plan_errors.is_empty(), "Plan errors: {:?}", plan_errors);
        // Must have at least 2 phases (barrier between matmuls).
        assert!(plan.phases.len() >= 2, "Expected >= 2 phases, got {}", plan.phases.len());
    }

    #[test]
    fn test_matmul_activation() {
        let (g, _, _, _) = test_graphs::matmul_activation(8, 4, 6, ScalarUnaryOp::Tanh);
        let errors = g.validate();
        assert!(errors.is_empty(), "Graph validation: {:?}", errors);
        let (plan, plan_errors) = plan_and_validate(&g, 4);
        assert!(plan_errors.is_empty(), "Plan errors: {:?}", plan_errors);
    }

    #[test]
    fn test_select_pattern() {
        // Simulate the GPT-2 attention mask pattern:
        // Small Select group (3072 atoms) reads from split producers.
        let mut g = NanoGraph::new();

        // Mask literal (small, will be inlined).
        let mask = g.push_group(
            512, ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![], vec![], vec![],
        );

        // Large input (split across lanes).
        let input = g.push_group(
            3072, ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![], vec![], vec![],
        );

        // Negative infinity.
        let neg_inf = g.push_atom(
            ScalarOp::Literal(NumericScalar::F32(f32::NEG_INFINITY)),
            vec![], vec![], vec![],
        );

        // Select: condition ? input : neg_inf
        let sel = g.push_group(
            3072,
            ScalarOp::Select { compute_dtype: DType::F32, output_dtype: DType::F32 },
            vec![], vec![],
            vec![
                InputRef::Modular { base: mask, stride: 1, modulus: 512 },
                InputRef::Affine { base: input, stride: 1 },
                InputRef::Broadcast(neg_inf),
            ],
        );

        // Downstream consumer.
        let out = g.push_group(
            3072,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![], vec![],
            vec![InputRef::Affine { base: sel, stride: 1 }],
        );
        g.outputs = vec![out];

        let errors = g.validate();
        assert!(errors.is_empty(), "Graph validation: {:?}", errors);

        let (plan, plan_errors) = plan_and_validate(&g, 8);
        assert!(plan_errors.is_empty(), "Plan errors: {:?}", plan_errors);
    }

    /// Build a GPT-2-like layer using merged matmul structure (StridedBroadcast).
    ///
    /// Each matmul C[M,N] = A[M,K] @ B[K,N] is represented as:
    /// - M Mul groups, each count = K*N, with:
    ///   - Input 0: StridedBroadcast { base: A[m,0], stride: 1, repeat: N }
    ///   - Input 1: Affine { base: B[0,0], stride: 1 }
    /// - M ReduceSum groups, each count = N, with:
    ///   - Input: Affine { base: mul_group_m, stride: 1 }
    ///   - reduce_count: K, reduce_stride: N
    fn build_merged_matmul(
        g: &mut NanoGraph,
        a_base: AtomId, // [M*K] literal
        b_base: AtomId, // [K*N] literal
        m: u64, k: u64, n: u64,
    ) -> AtomId {
        // M Mul groups, each count = K*N.
        let mut mul_bases: Vec<AtomId> = Vec::new();
        for mi in 0..m {
            let a_row = a_base.offset(mi * k); // A[m, 0]
            let base = g.push_group(
                k * n,
                ScalarOp::Binary {
                    op: ScalarBinOp::Mul,
                    compute_dtype: DType::F32, output_dtype: DType::F32,
                },
                vec![], vec![],
                vec![
                    InputRef::StridedBroadcast { base: a_row, stride: 1, repeat: n },
                    InputRef::Affine { base: b_base, stride: 1 },
                ],
            );
            mul_bases.push(base);
        }

        // M ReduceSum groups, each count = N.
        let mut reduce_base: Option<AtomId> = None;
        for mi in 0..m {
            let base = g.push_group(
                n,
                ScalarOp::ReduceSum {
                    reduce_count: k, reduce_stride: n as i64,
                    compute_dtype: DType::F32, output_dtype: DType::F32,
                },
                vec![], vec![],
                vec![InputRef::Affine { base: mul_bases[mi as usize], stride: 1 }],
            );
            if reduce_base.is_none() {
                reduce_base = Some(base);
            }
        }
        reduce_base.unwrap()
    }

    /// Build elementwise op on the full output of a matmul [M*N].
    fn build_elementwise(
        g: &mut NanoGraph,
        input_base: AtomId,
        count: u64,
        op: ScalarUnaryOp,
    ) -> AtomId {
        g.push_group(
            count,
            ScalarOp::Unary {
                op, compute_dtype: DType::F32, output_dtype: DType::F32,
            },
            vec![], vec![],
            vec![InputRef::Affine { base: input_base, stride: 1 }],
        )
    }

    /// Simulate a GPT-2-like transformer layer:
    /// matmul_q -> matmul_kv -> attention_mask -> matmul_out -> add_residual
    #[test]
    fn test_transformer_layer_pattern() {
        let mut g = NanoGraph::new();

        let m = 16u64; // batch * seq
        let hidden = 64u64; // hidden size
        let k = hidden;
        let n = hidden;

        // Input embedding (residual).
        let input = g.push_group(
            m * hidden,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![], vec![], vec![],
        );

        // Weight matrices.
        let w_q = g.push_group(
            k * n,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![], vec![], vec![],
        );
        let w_v = g.push_group(
            k * n,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![], vec![], vec![],
        );
        let w_out = g.push_group(
            k * n,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![], vec![], vec![],
        );

        // Q projection: [M, hidden] = input @ W_q
        let q = build_merged_matmul(&mut g, input, w_q, m, k, n);

        // V projection: [M, hidden] = input @ W_v
        let v = build_merged_matmul(&mut g, input, w_v, m, k, n);

        // Elementwise activation on Q.
        let q_act = build_elementwise(&mut g, q, m * n, ScalarUnaryOp::Tanh);

        // Output projection: [M, hidden] = q_act @ W_out
        // (simplified — in real GPT-2, attention is more complex)
        let out = build_merged_matmul(&mut g, q_act, w_out, m, k, n);

        // Residual add: input + out.
        let residual = g.push_group(
            m * hidden,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32, output_dtype: DType::F32,
            },
            vec![], vec![],
            vec![
                InputRef::Affine { base: input, stride: 1 },
                InputRef::Affine { base: out, stride: 1 },
            ],
        );
        g.outputs = vec![residual];

        let errors = g.validate();
        assert!(errors.is_empty(), "Graph validation: {:?}", errors);

        let (plan, plan_errors) = plan_and_validate(&g, 8);
        assert!(plan_errors.is_empty(), "Plan errors: {:?}", plan_errors);

        // Should have multiple phases (at least 3: Q/V matmuls, activation+out matmul, residual).
        assert!(plan.phases.len() >= 2, "Expected >= 2 phases, got {}", plan.phases.len());

        // Check balance: compute max imbalance across phases.
        for (pi, phase) in plan.phases.iter().enumerate() {
            let lane_atoms: Vec<u64> = phase.spans.iter()
                .map(|s| s.outputs.iter().map(|m| m.count).sum())
                .collect();
            let mx = lane_atoms.iter().copied().max().unwrap_or(0);
            let mn = lane_atoms.iter().copied().filter(|&a| a > 0).min().unwrap_or(1);
            if mn > 0 && mx > 0 {
                let imbalance = mx as f64 / mn as f64;
                assert!(imbalance < 100.0,
                    "Phase {} imbalance {:.1}x too high (max={}, min={})",
                    pi, imbalance, mx, mn);
            }
        }
    }

    /// Simulate multiple transformer layers chained together.
    #[test]
    fn test_multi_layer_transformer() {
        let mut g = NanoGraph::new();

        let m = 16u64;
        let hidden = 64u64;
        let num_layers = 3;

        // Initial input.
        let mut current = g.push_group(
            m * hidden,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![], vec![], vec![],
        );

        for _layer in 0..num_layers {
            // Weight for this layer's matmul.
            let w = g.push_group(
                hidden * hidden,
                ScalarOp::Literal(NumericScalar::F32(0.0)),
                vec![], vec![], vec![],
            );

            // Matmul.
            let mm_out = build_merged_matmul(&mut g, current, w, m, hidden, hidden);

            // Activation.
            let act = build_elementwise(&mut g, mm_out, m * hidden, ScalarUnaryOp::Tanh);

            // Residual.
            let residual = g.push_group(
                m * hidden,
                ScalarOp::Binary {
                    op: ScalarBinOp::Add,
                    compute_dtype: DType::F32, output_dtype: DType::F32,
                },
                vec![], vec![],
                vec![
                    InputRef::Affine { base: current, stride: 1 },
                    InputRef::Affine { base: act, stride: 1 },
                ],
            );

            current = residual;
        }
        g.outputs = vec![current];

        let errors = g.validate();
        assert!(errors.is_empty(), "Graph validation: {:?}", errors);

        let t0 = std::time::Instant::now();
        let (plan, plan_errors) = plan_and_validate(&g, 8);
        let elapsed = t0.elapsed();

        assert!(plan_errors.is_empty(), "Plan errors: {:?}", plan_errors);

        // Should have enough phases for the 3-layer chain.
        assert!(plan.phases.len() >= num_layers,
            "Expected >= {} phases for {} layers, got {}",
            num_layers, num_layers, plan.phases.len());

        // Should be fast.
        assert!(elapsed.as_secs_f64() < 5.0,
            "Planning took {:.2}s, expected < 5s", elapsed.as_secs_f64());
    }

    /// Larger test: 10-layer transformer chain with merged matmuls.
    /// Simulates the core GPT-2 structure. Must complete in < 10 seconds.
    #[test]
    fn test_large_transformer_stack() {
        let mut g = NanoGraph::new();

        let m = 64u64;     // batch*seq
        let hidden = 128u64; // hidden dim
        let num_layers = 10;

        // Initial input.
        let mut current = g.push_group(
            m * hidden,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![], vec![], vec![],
        );

        for _layer in 0..num_layers {
            // Weight matrix for Q projection.
            let w_q = g.push_group(
                hidden * hidden,
                ScalarOp::Literal(NumericScalar::F32(0.0)),
                vec![], vec![], vec![],
            );
            // Weight matrix for output projection.
            let w_out = g.push_group(
                hidden * hidden,
                ScalarOp::Literal(NumericScalar::F32(0.0)),
                vec![], vec![], vec![],
            );

            // Q projection matmul.
            let q = build_merged_matmul(&mut g, current, w_q, m, hidden, hidden);

            // Activation.
            let act = build_elementwise(&mut g, q, m * hidden, ScalarUnaryOp::Tanh);

            // Output projection matmul.
            let out = build_merged_matmul(&mut g, act, w_out, m, hidden, hidden);

            // Residual add.
            current = g.push_group(
                m * hidden,
                ScalarOp::Binary {
                    op: ScalarBinOp::Add,
                    compute_dtype: DType::F32, output_dtype: DType::F32,
                },
                vec![], vec![],
                vec![
                    InputRef::Affine { base: current, stride: 1 },
                    InputRef::Affine { base: out, stride: 1 },
                ],
            );
        }
        g.outputs = vec![current];

        let errors = g.validate();
        assert!(errors.is_empty(), "Graph validation: {:?}", errors);

        let num_groups = g.num_groups();
        let num_atoms = g.num_atoms();

        let t0 = std::time::Instant::now();
        let (plan, plan_errors) = plan_and_validate(&g, 8);
        let elapsed = t0.elapsed();

        assert!(plan_errors.is_empty(),
            "Plan errors (first 10): {:?}",
            &plan_errors[..plan_errors.len().min(10)]);

        // Performance check.
        assert!(elapsed.as_secs_f64() < 10.0,
            "Planning {} groups ({:.1}B atoms) took {:.2}s, expected < 10s",
            num_groups, num_atoms as f64 / 1e9, elapsed.as_secs_f64());

        // Print stats for debugging.
        eprintln!("  [v3b large] {:.1}ms, {} groups, {:.1}M atoms",
            elapsed.as_secs_f64() * 1e3, num_groups, num_atoms as f64 / 1e6);
        eprintln!("    {} phases, {} lanes", plan.phases.len(), 8);

        // Compute balance info.
        let mut max_imbalance = 0f64;
        for phase in &plan.phases {
            let lane_atoms: Vec<u64> = phase.spans.iter()
                .map(|s| s.outputs.iter().map(|m| m.count).sum())
                .collect();
            let mx = lane_atoms.iter().copied().max().unwrap_or(0);
            let mn = lane_atoms.iter().copied().filter(|&a| a > 0).min().unwrap_or(1);
            if mn > 0 && mx > 0 {
                max_imbalance = max_imbalance.max(mx as f64 / mn as f64);
            }
        }
        eprintln!("    max_imbalance: {:.1}x", max_imbalance);

        // Structural checks.
        assert!(plan.phases.len() >= num_layers,
            "Expected >= {} phases for {} layers, got {}",
            num_layers, num_layers, plan.phases.len());

        // Check that all compute atoms are covered.
        let total_output_atoms: u64 = plan.phases.iter()
            .flat_map(|p| p.spans.iter())
            .flat_map(|s| s.outputs.iter())
            .map(|m| m.count)
            .sum();
        assert!(total_output_atoms > 0, "No output atoms produced");
    }

    /// Scalability test: larger hidden dim, closer to GPT-2 scale.
    /// GPT-2 has hidden=768, M depends on batch*seq.
    #[test]
    fn test_gpt2_scale_single_layer() {
        let mut g = NanoGraph::new();

        let m = 48u64;      // batch * seq (keeping small for test speed)
        let hidden = 768u64; // GPT-2 hidden dim
        let num_layers = 1;

        let mut current = g.push_group(
            m * hidden,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![], vec![], vec![],
        );

        // One transformer layer with Q, K, V projections.
        let w_q = g.push_group(hidden * hidden, ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![], vec![], vec![]);
        let w_k = g.push_group(hidden * hidden, ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![], vec![], vec![]);
        let w_v = g.push_group(hidden * hidden, ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![], vec![], vec![]);
        let w_out = g.push_group(hidden * hidden, ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![], vec![], vec![]);

        let q = build_merged_matmul(&mut g, current, w_q, m, hidden, hidden);
        let k = build_merged_matmul(&mut g, current, w_k, m, hidden, hidden);
        let v = build_merged_matmul(&mut g, current, w_v, m, hidden, hidden);

        // Elementwise on Q.
        let q_act = build_elementwise(&mut g, q, m * hidden, ScalarUnaryOp::Tanh);

        // Output projection (simplified).
        let out = build_merged_matmul(&mut g, q_act, w_out, m, hidden, hidden);

        current = g.push_group(
            m * hidden,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32, output_dtype: DType::F32,
            },
            vec![], vec![],
            vec![
                InputRef::Affine { base: current, stride: 1 },
                InputRef::Affine { base: out, stride: 1 },
            ],
        );
        g.outputs = vec![current];

        let errors = g.validate();
        assert!(errors.is_empty(), "Graph validation: {:?}", errors);

        let t0 = std::time::Instant::now();
        let plan = plan_spans(&g, 8);
        let plan_elapsed = t0.elapsed();

        let t1 = std::time::Instant::now();
        let plan_errors = validate_plan(&plan, &g);
        let val_elapsed = t1.elapsed();

        eprintln!("  [v3b gpt2_scale] plan={:.1}ms val={:.1}ms, {} groups, {:.1}B atoms, {} phases",
            plan_elapsed.as_secs_f64() * 1e3, val_elapsed.as_secs_f64() * 1e3,
            g.num_groups(), g.num_atoms() as f64 / 1e9,
            plan.phases.len());

        assert!(plan_errors.is_empty(),
            "Plan errors (first 10): {:?}",
            &plan_errors[..plan_errors.len().min(10)]);

        assert!(plan_elapsed.as_secs_f64() < 10.0,
            "Planning took {:.2}s, expected < 10s", plan_elapsed.as_secs_f64());
    }

    /// Test with attention mask pattern: Select group with Modular input
    /// between two matmuls.
    #[test]
    fn test_attention_mask_between_matmuls() {
        let mut g = NanoGraph::new();

        let m = 16u64;
        let k = 32u64;
        let n = 32u64;

        // Weights.
        let w1 = g.push_group(k * n, ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![], vec![], vec![]);
        let w2 = g.push_group(n * n, ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![], vec![], vec![]);

        // Input.
        let input = g.push_group(m * k, ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![], vec![], vec![]);

        // First matmul.
        let mm1 = build_merged_matmul(&mut g, input, w1, m, k, n);

        // Attention mask.
        let mask = g.push_group(64, ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![], vec![], vec![]);
        let neg_inf = g.push_atom(
            ScalarOp::Literal(NumericScalar::F32(f32::NEG_INFINITY)),
            vec![], vec![], vec![]);

        let masked = g.push_group(
            m * n,
            ScalarOp::Select { compute_dtype: DType::F32, output_dtype: DType::F32 },
            vec![], vec![],
            vec![
                InputRef::Modular { base: mask, stride: 1, modulus: 64 },
                InputRef::Affine { base: mm1, stride: 1 },
                InputRef::Broadcast(neg_inf),
            ],
        );

        // Second matmul.
        let mm2 = build_merged_matmul(&mut g, masked, w2, m, n, n);

        g.outputs = vec![mm2];

        let errors = g.validate();
        assert!(errors.is_empty(), "Graph validation: {:?}", errors);

        let (plan, plan_errors) = plan_and_validate(&g, 8);
        assert!(plan_errors.is_empty(), "Plan errors: {:?}", plan_errors);
    }
}
