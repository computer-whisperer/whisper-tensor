#![allow(clippy::all, dead_code, unused)]
//! Global memory placer.
//!
//! Takes the partitioner's `Vec<Phase>` output and assigns every
//! cross-span atom a `(buffer_id, byte_offset)`. See `MEMORY_PLACEMENT.md`
//! for the full design.
//!
//! # What this does (step 3 of the implementation order)
//!
//! 1. Classify every main-graph group as Input / Output / Intermediate
//!    / Scratch based on which buffer it would live in under the new
//!    design.
//! 2. Compute `[first_produce_phase, last_consume_phase]` liveness
//!    intervals for every cross-span (Intermediate) group.
//! 3. Walk every InputRef in every span and detect coalescing
//!    constraints that force cross-span groups into contiguous slabs.
//!    Union-find merges connected components; each component becomes a
//!    slab with the union of its members' liveness.
//! 4. Pack slabs into the intermediate buffer via a first-fit
//!    allocator keyed on liveness intervals. Slot offsets are reused
//!    once every member of a slab is dead.
//! 5. Assign dedicated per-input and per-output buffers.
//! 6. Emit an `AtomPlacementMap` plus per-buffer sizes and a peak
//!    live footprint diagnostic.
//!
//! # What this does NOT do
//!
//! - Per-span scratch sizing stays with per-span codegen (it already
//!   runs today's `compute_layout`).
//! - Nothing here mutates the executor or the JIT ABI yet; this is a
//!   standalone pass whose output is logged by
//!   `compile_nano_graph` under `WT_PRINT_PLACEMENT=1`.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::nano_graph::pattern::{AtomId, AtomRange, InputRef, NanoGraph};
use crate::numeric_dtype::NumericDType;
use crate::pool::SystemPool;

use super::partitioner_m::input_access_segments;
use super::types::Phase;

// ─── Public types ─────────────────────────────────────────────────────

/// Which buffer an atom lives in under the new design.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferKind {
    /// Model input — one dedicated buffer per input tensor, identified
    /// by its slot in `main_graph.input_tensors()`.
    Input,
    /// Model output — one dedicated buffer per output atom range.
    Output,
    /// Cross-span working data — lives in the shared intermediate buffer.
    Intermediate,
    /// Plan-wide immutable literal data. One buffer per plan, holds
    /// every `Literal` / `LiteralSpan` group's bytes.
    Literal,
    /// Per-lane scratch — not placed by the global pass. Every span's
    /// per-lane scratch shares the same `scratch_buffer_id`; each
    /// lane's arena lives at a distinct pointer the executor patches
    /// in at dispatch time.
    Scratch,
}

/// Opaque buffer identifier. Stable within one `AtomPlacementMap`.
///
/// u8 is wide enough for every realistic model (dozens of buffers max)
/// and the value is used directly as an index into the `buffer_ptrs`
/// array the JIT receives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BufferId(pub u8);

/// The shared intermediate buffer always has id 0.
pub const INTERMEDIATE_BUFFER: BufferId = BufferId(0);

/// The plan-wide literal buffer always has id 1.
pub const LITERAL_BUFFER: BufferId = BufferId(1);

/// Per-buffer metadata the executor needs at allocation time.
#[derive(Debug, Clone)]
pub struct BufferInfo {
    pub id: BufferId,
    pub kind: BufferKind,
    pub size_bytes: u64,
    /// Optional debug name — model input name, output name, or
    /// "intermediate".
    pub name: String,
}

/// Placement record for a contiguous range of atoms.
///
/// The placer emits one entry per cross-span group (or slab member).
/// Lookup is by `atom_base` with a binary search; atom `i` within the
/// group lives at `byte_offset + (i - atom_base) * bytes_per_element`.
#[derive(Debug, Clone)]
pub struct PlacementEntry {
    pub atom_base: AtomId,
    pub count: u64,
    pub dtype: NumericDType,
    pub buffer_id: BufferId,
    pub byte_offset: u64,
}

/// Output of the placer.
#[derive(Debug)]
pub struct AtomPlacementMap {
    /// All placement entries sorted by `atom_base` for binary search.
    entries: Vec<PlacementEntry>,
    /// Per-buffer metadata.
    pub buffers: Vec<BufferInfo>,
    /// Global scratch buffer_id. Not a real entry in `buffers` — it is
    /// the first id above every input/output/intermediate/literal
    /// buffer. Per-lane scratch arenas all share this id; each lane's
    /// distinct pointer is patched into `buffer_ptrs[scratch_buffer_id]`
    /// at dispatch time.
    pub scratch_buffer_id: u8,
    /// Peak live byte footprint in the intermediate buffer during
    /// interval packing. The intermediate buffer is sized to this
    /// high-water mark.
    pub intermediate_peak_bytes: u64,
    /// Total bytes the plan-wide literal buffer needs. The executor
    /// allocates this `Box<[u8]>` at plan-build time and walks the
    /// main graph once to populate it from `Literal`/`LiteralSpan`
    /// group bytes.
    pub literal_buffer_size: u64,
    /// Per-literal-group byte offset in the literal buffer where that
    /// group's source bytes live. **Every** `Literal`/`LiteralSpan`
    /// group has an entry here — including groups whose primary
    /// placement is an output buffer (because their atoms overlap a
    /// model output range). The executor reads from here when
    /// populating the literal buffer at plan-build; the JIT reads from
    /// here when it needs to emit a copy from the literal buffer to a
    /// non-literal destination. Keyed by the group's `base_id`.
    pub literal_sources: HashMap<AtomId, u64>,
    /// Number of coalescing slabs the placer built for the intermediate
    /// buffer.
    pub intermediate_slab_count: usize,
    /// Groups the placer considered but classified as span-local; these
    /// fall through to per-span scratch.
    pub scratch_group_count: usize,
    /// Top slabs by size, for diagnostics.
    pub top_slabs: Vec<SlabDiag>,
}

/// Diagnostic info about one slab, for reporting.
#[derive(Debug, Clone)]
pub struct SlabDiag {
    pub size_bytes: u64,
    pub first_phase: usize,
    pub last_phase: usize,
    pub member_count: usize,
    pub sample_op: String,
    pub sample_atom_base: u64,
    pub sample_count: u64,
}

impl AtomPlacementMap {
    /// Find the placement entry covering `atom`, if any.
    pub fn find(&self, atom: AtomId) -> Option<&PlacementEntry> {
        let idx = self.entries.partition_point(|e| e.atom_base.0 <= atom.0);
        if idx == 0 {
            return None;
        }
        let e = &self.entries[idx - 1];
        if atom.0 < e.atom_base.0 + e.count {
            Some(e)
        } else {
            None
        }
    }

    /// Byte offset (within the entry's buffer) of a specific atom.
    pub fn byte_offset_of(&self, atom: AtomId) -> Option<(BufferId, u64)> {
        let e = self.find(atom)?;
        let idx = atom.0 - e.atom_base.0;
        let off = e.byte_offset + idx * e.dtype.bytes_per_element() as u64;
        Some((e.buffer_id, off))
    }

    /// The placement entries (read-only). Callers that need to walk
    /// every entry — e.g. to reverse-map `buffer_id → atom_base` —
    /// use this accessor.
    pub fn entries(&self) -> &[PlacementEntry] {
        &self.entries
    }

    /// Print a human-readable summary to stderr. Useful under
    /// `WT_PRINT_PLACEMENT=1`.
    pub fn print_summary(&self) {
        let n_in = self
            .buffers
            .iter()
            .filter(|b| b.kind == BufferKind::Input)
            .count();
        let n_out = self
            .buffers
            .iter()
            .filter(|b| b.kind == BufferKind::Output)
            .count();
        let input_bytes: u64 = self
            .buffers
            .iter()
            .filter(|b| b.kind == BufferKind::Input)
            .map(|b| b.size_bytes)
            .sum();
        let output_bytes: u64 = self
            .buffers
            .iter()
            .filter(|b| b.kind == BufferKind::Output)
            .map(|b| b.size_bytes)
            .sum();

        eprintln!("[placer] AtomPlacementMap summary:");
        eprintln!("  buffers: {} total", self.buffers.len());
        eprintln!("    input buffers:  {n_in} ({} bytes total)", input_bytes);
        eprintln!("    output buffers: {n_out} ({} bytes total)", output_bytes);
        eprintln!(
            "    intermediate buffer peak: {} bytes ({:.1} MB)",
            self.intermediate_peak_bytes,
            self.intermediate_peak_bytes as f64 / (1024.0 * 1024.0),
        );
        eprintln!(
            "    literal buffer:           {} bytes ({:.1} MB)",
            self.literal_buffer_size,
            self.literal_buffer_size as f64 / (1024.0 * 1024.0),
        );
        eprintln!("    scratch buffer_id:        {}", self.scratch_buffer_id);
        eprintln!("  placement entries:          {}", self.entries.len());
        eprintln!(
            "  intermediate slabs:         {}",
            self.intermediate_slab_count
        );
        eprintln!(
            "  span-local groups (scratch): {}",
            self.scratch_group_count
        );
        if !self.top_slabs.is_empty() {
            eprintln!("  top intermediate slabs by size:");
            for (i, s) in self.top_slabs.iter().take(10).enumerate() {
                eprintln!(
                    "    {i:2}. {:>12} bytes ({:.1} MB)  phases [{}..{}]  {} members  op={} base={} count={}",
                    s.size_bytes,
                    s.size_bytes as f64 / (1024.0 * 1024.0),
                    s.first_phase,
                    s.last_phase,
                    s.member_count,
                    s.sample_op,
                    s.sample_atom_base,
                    s.sample_count,
                );
            }
        }
    }
}

// ─── Internal: union-find ─────────────────────────────────────────────

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }
    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }
    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] += 1;
        }
    }
}

// ─── Slab representation ──────────────────────────────────────────────

#[derive(Debug, Clone)]
struct SlabInterval {
    /// Member group indices (into main_graph.groups()).
    members: Vec<usize>,
    /// Contiguous atom-id range covered by all members.
    atom_lo: u64,
    atom_hi: u64,
    /// Element size — the slab's bytes-per-atom. All members must agree.
    elem_bytes: u64,
    /// Dtype — kept for the PlacementEntry emission.
    dtype: NumericDType,
    /// Liveness interval (inclusive both ends).
    first_phase: usize,
    last_phase: usize,
    /// Set if any member would be written by multiple lanes — we cache-
    /// line-align the start.
    split_written: bool,
}

// ─── Entry point ──────────────────────────────────────────────────────

const CACHE_LINE_BYTES: u64 = 64;

/// Run the placer against a partitioned model.
pub fn run_placer(
    main_graph: &NanoGraph<'static, SystemPool>,
    phases: &[Phase],
    all_output_atom_ranges: &[AtomRange],
) -> Result<AtomPlacementMap, String> {
    use crate::nano_graph::ops::ScalarOp;

    let groups = main_graph.groups();
    let input_tensors = main_graph.input_tensors();

    // ── Step 1: classify every group by buffer kind ──
    //
    // Literal/LiteralSpan groups always go in the plan-wide literal
    // buffer regardless of their phase position. Output groups (those
    // that overlap a model output range) go in output buffers.
    // Everything else starts as Scratch and may be promoted to
    // Intermediate in step 2 if it flows across spans. Input tensors
    // always go in their own buffers.

    let mut group_kind = vec![BufferKind::Scratch; groups.len()];

    // Literal groups take precedence — they're immutable plan data,
    // unaffected by phase structure.
    for (gi, g) in groups.iter().enumerate() {
        if matches!(&g.op, ScalarOp::Literal(_) | ScalarOp::LiteralSpan(_)) {
            group_kind[gi] = BufferKind::Literal;
        }
    }

    // Mark output groups. A model output range may straddle multiple
    // groups (Pad-style lowering); every such group is Output. If a
    // Literal group overlaps an output range (test-only pattern: a
    // graph that constant-folds an output), keep it as Output so the
    // JIT emits stores for it into the output buffer on every execute
    // — literals written to the literal buffer are immutable and
    // can't be routed into a per-execute output allocation.
    let mut group_output_range: Vec<Option<usize>> = vec![None; groups.len()];
    for (oi, range) in all_output_atom_ranges.iter().enumerate() {
        let r_lo = range.base.0;
        let r_hi = r_lo + range.count;
        for (gi, g) in groups.iter().enumerate() {
            let g_lo = g.base_id.0;
            let g_hi = g_lo + g.count;
            if g_lo < r_hi && r_lo < g_hi {
                if group_kind[gi] == BufferKind::Output && group_output_range[gi] != Some(oi) {
                    return Err(format!(
                        "placer: group {gi} (base={}) straddles multiple output ranges",
                        g.base_id.0
                    ));
                }
                group_kind[gi] = BufferKind::Output;
                group_output_range[gi] = Some(oi);
            }
        }
    }

    // Mark intermediate groups. A group that appears in any span's
    // outputs flows between spans. Skip if already marked Output or
    // Literal (Literal groups live in the plan-wide literal buffer
    // regardless of phase position).
    //
    // Also track `declaring_phases[gi]` for diagnostics so we can see
    // which groups got marked Intermediate and why — this is the
    // subset of groups the placer routes into the shared buffer.
    let mut declaring_phases: HashMap<usize, Vec<usize>> = HashMap::new();
    for (pi, phase) in phases.iter().enumerate() {
        for span in &phase.spans {
            for range in &span.outputs {
                if let Some(gi) = main_graph.find_group_idx(range.base) {
                    if group_kind[gi] == BufferKind::Scratch {
                        group_kind[gi] = BufferKind::Intermediate;
                    }
                    declaring_phases.entry(gi).or_default().push(pi);
                }
            }
        }
    }

    // ── Step 2: liveness intervals for cross-span groups ──
    //
    // Intermediate and Output groups need liveness; Input lives for
    // the whole call; Scratch is per-span so it doesn't need tracking.
    // Literal groups are tracked here too, in case step 3 promotes
    // any of them into an Intermediate slab (Pad-style: a Literal
    // pad zero group that a consumer's Strided InputRef coalesces
    // with adjacent Identity groups). A promoted Literal's slab
    // needs a first_phase/last_phase interval like any other
    // intermediate-buffer resident.

    let last_phase = phases.len().saturating_sub(1);
    let mut liveness: HashMap<usize, (usize, usize)> = HashMap::new();

    for (pi, phase) in phases.iter().enumerate() {
        for span in &phase.spans {
            // Writer side: `outputs` marks when a group is produced.
            for range in &span.outputs {
                if let Some(gi) = main_graph.find_group_idx(range.base) {
                    if matches!(
                        group_kind[gi],
                        BufferKind::Intermediate | BufferKind::Output | BufferKind::Literal
                    ) {
                        let entry = liveness.entry(gi).or_insert((pi, pi));
                        entry.0 = entry.0.min(pi);
                        entry.1 = entry.1.max(pi);
                    }
                }
            }
            // Reader side: `inputs` extends the group's lifetime.
            for range in &span.inputs {
                if let Some(gi) = main_graph.find_group_idx(range.base) {
                    if matches!(
                        group_kind[gi],
                        BufferKind::Intermediate | BufferKind::Output | BufferKind::Literal
                    ) {
                        let entry = liveness.entry(gi).or_insert((pi, pi));
                        entry.1 = entry.1.max(pi);
                    }
                }
            }
        }
    }

    // Pin model outputs to survive to extraction.
    for range in all_output_atom_ranges {
        if let Some(gi) = main_graph.find_group_idx(range.base) {
            if group_kind[gi] == BufferKind::Output {
                let entry = liveness.entry(gi).or_insert((0, last_phase));
                entry.1 = entry.1.max(last_phase);
            }
        }
    }

    // ── Step 3: coalescing constraints via union-find ──
    //
    // Unions any two groups that a consumer's Strided InputRef (or
    // Reduce k-stride) requires to be laid out contiguously in
    // memory. Both `Intermediate` and `Literal` groups participate:
    //
    // - `Intermediate` members are ordinary cross-span values that
    //   live in the shared intermediate buffer; coalescing packs
    //   them into slabs.
    //
    // - `Literal` members are pulled into a slab only when a
    //   consumer's access interleaves them with Intermediate atoms
    //   (Pad-style lowering where pad zeros and copied input rows
    //   alternate). A Literal that lands in an intermediate slab
    //   gets its **primary** placement moved from the plan-wide
    //   literal buffer to the slab position in the intermediate
    //   buffer; its source bytes still live in the literal buffer
    //   at `literal_sources[base]`, and the per-span JIT emits an
    //   ordinary copy from that source into the intermediate slot
    //   at every execute (same mechanism compute_layout already
    //   uses for the intra-span Pad case).
    //
    // `Output` members are pinned to their placer-assigned output
    // buffer and can't move; `Scratch` members are span-local and
    // never reach the placer's coalescer. Input tensors are
    // similarly Fixed. Any coalescing constraint that pulls in
    // such a member is still rejected as cross-buffer (the audit
    // said it never happens on real models).

    let mut uf = UnionFind::new(groups.len());

    // Pre-sort items by start atom-id for binary search of access
    // segment overlaps. Items = groups only (input tensors live in
    // their own buffers and by the audit never coalesce).
    let mut sorted_groups: Vec<(u64, u64, usize)> = groups
        .iter()
        .enumerate()
        .map(|(gi, g)| (g.base_id.0, g.base_id.0 + g.count, gi))
        .collect();
    sorted_groups.sort_unstable_by_key(|&(lo, _, _)| lo);

    // Track which groups were "split-written" for cache-line alignment.
    // A group is split-written if more than one span declares it as an
    // output (typically one per lane for a Split group).
    let mut split_written = vec![0u32; groups.len()];
    for phase in phases {
        for span in &phase.spans {
            for range in &span.outputs {
                if let Some(gi) = main_graph.find_group_idx(range.base) {
                    split_written[gi] = split_written[gi].saturating_add(1);
                }
            }
        }
    }
    let is_split_written = |gi: usize| split_written[gi] > 1;

    for phase in phases {
        for span in &phase.spans {
            for group in span.graph.groups() {
                for input_ref in &group.inputs {
                    let mut segments =
                        input_access_segments(&input_ref.input_ref, group.atom_offset, group.count);

                    // Reduce groups iterate an inner k-loop that reads
                    // `k * reduce_stride` atoms beyond each nominal
                    // atom. `input_access_segments` only sees the
                    // InputRef's own stride pattern and has no knowledge
                    // of the ScalarOp, so the base segments cover only
                    // `k == 0`. Replicate the base segments at each
                    // `k * reduce_stride` offset so the coalescing pass
                    // can see every producer group the reduce actually
                    // reads and union them into one contiguous slab.
                    // Without this, a multi-group reduce (e.g. Conv's
                    // 9 kernel-position Mul groups feeding one Reduce
                    // with reduce_stride == spatial_size) lands with
                    // its producers at arbitrary first-fit offsets,
                    // and the JIT's `base + k*reduce_stride*bit_stride`
                    // formula reads the wrong bytes.
                    if let ScalarOp::Reduce {
                        reduce_count,
                        reduce_stride,
                        ..
                    } = &group.op
                    {
                        if *reduce_count > 1 && *reduce_stride != 0 {
                            let base_segments: Vec<(u64, u64)> = segments.clone();
                            for k in 1..*reduce_count {
                                let off = k as i64 * *reduce_stride;
                                for &(seg_lo, seg_hi) in &base_segments {
                                    let new_lo = seg_lo as i64 + off;
                                    let new_hi = seg_hi as i64 + off;
                                    if new_lo >= 0 && new_hi > new_lo {
                                        segments.push((new_lo as u64, new_hi as u64));
                                    }
                                }
                            }
                        }
                    }

                    // Gather the set of main-graph groups overlapping any segment.
                    let mut items: Vec<usize> = Vec::new();
                    let mut any_input_tensor = false;
                    let mut any_scratch = false;
                    for &(seg_lo, seg_hi) in &segments {
                        if seg_lo >= seg_hi {
                            continue;
                        }
                        // Check input tensors.
                        for it in input_tensors {
                            let it_lo = it.base_id.0;
                            let it_hi = it_lo + it.count;
                            if seg_lo < it_hi && it_lo < seg_hi {
                                any_input_tensor = true;
                            }
                        }
                        // Check groups via binary search.
                        let start = sorted_groups.partition_point(|&(_, hi, _)| hi <= seg_lo);
                        for &(g_lo, g_hi, gi) in &sorted_groups[start..] {
                            if g_lo >= seg_hi {
                                break;
                            }
                            if g_hi > seg_lo && g_lo < seg_hi {
                                if !items.contains(&gi) {
                                    items.push(gi);
                                }
                                if group_kind[gi] == BufferKind::Scratch {
                                    any_scratch = true;
                                }
                            }
                        }
                    }

                    if items.len() < 2 {
                        continue;
                    }

                    // Members eligible for coalescing: Intermediate
                    // (natural slab citizen) and Literal (promoted
                    // into the slab via a plan-build copy). Any
                    // other member kind (Scratch / Output / Input)
                    // being pulled into the access range is a
                    // cross-buffer design violation.
                    let coalescable_members: Vec<usize> = items
                        .iter()
                        .copied()
                        .filter(|&gi| {
                            matches!(
                                group_kind[gi],
                                BufferKind::Intermediate | BufferKind::Literal
                            )
                        })
                        .collect();

                    let has_non_coalescable = items.iter().any(|&gi| {
                        !matches!(
                            group_kind[gi],
                            BufferKind::Intermediate | BufferKind::Literal
                        )
                    });
                    if has_non_coalescable && !coalescable_members.is_empty() {
                        // Diagnostic: report the first such case and continue.
                        // (The audit said this never happens — if it does
                        // now, we want to know.)
                        eprintln!(
                            "[placer] WARN: cross-buffer coalescing constraint \
                             (expected none per audit). Consumer group base={} \
                             coalescable={} other={}",
                            group.base_id.0,
                            coalescable_members.len(),
                            items.len() - coalescable_members.len(),
                        );
                        continue;
                    }

                    // Union all coalescable members together.
                    for i in 1..coalescable_members.len() {
                        uf.union(coalescable_members[0], coalescable_members[i]);
                    }
                }
            }
        }
    }

    // ── Step 4: build slabs from connected components ──
    //
    // Components are built from both Intermediate and Literal groups
    // — any Literal that got unioned with an Intermediate member
    // during step 3 shows up in the same component. A component is
    // promoted to an intermediate slab iff it has at least one
    // Intermediate member; all-Literal components (which happen
    // when a consumer's Strided access spans only consecutive
    // Literal groups) stay at their placer-assigned literal-buffer
    // offsets and don't need a slab. `literal_promoted[gi]` tracks
    // which Literal groups got pulled into an intermediate slab so
    // the literal-entry emission below knows to skip them.

    let mut literal_promoted: Vec<bool> = vec![false; groups.len()];

    // BTreeMap for deterministic iteration — slab ordering affects
    // first-fit packing. Even though a correct layout must produce
    // identical outputs regardless of ordering, keeping this
    // deterministic removes one variable when debugging.
    let mut component_members: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for gi in 0..groups.len() {
        if !matches!(
            group_kind[gi],
            BufferKind::Intermediate | BufferKind::Literal
        ) {
            continue;
        }
        let root = uf.find(gi);
        component_members.entry(root).or_default().push(gi);
    }

    let mut slabs: Vec<SlabInterval> = Vec::new();
    for (_, members) in component_members {
        // Skip all-Literal components: those groups stay in the
        // literal buffer, laid out contiguously by step 6.
        let has_intermediate = members
            .iter()
            .any(|&gi| group_kind[gi] == BufferKind::Intermediate);
        if !has_intermediate {
            continue;
        }
        // Mark Literal members of this slab as promoted so they
        // skip the LITERAL_BUFFER primary-entry emission below.
        for &gi in &members {
            if group_kind[gi] == BufferKind::Literal {
                literal_promoted[gi] = true;
            }
        }
        // Compute slab extent, dtype, liveness, split-written flag.
        let mut lo = u64::MAX;
        let mut hi = 0u64;
        let mut dtype = NumericDType::F32;
        let mut elem_bytes = 0u64;
        let mut first_phase = usize::MAX;
        let mut last_p = 0usize;
        let mut split = false;
        let mut any_live = false;

        for &gi in &members {
            let g = &groups[gi];
            lo = lo.min(g.base_id.0);
            hi = hi.max(g.base_id.0 + g.count);

            // Elem bytes — all members must agree (same dtype) for a
            // single stride formula to work.
            let eb = g.output_dtype.bytes_per_element() as u64;
            if elem_bytes == 0 {
                elem_bytes = eb;
                dtype = g.output_dtype;
            } else if elem_bytes != eb {
                return Err(format!(
                    "placer: slab members disagree on elem_bytes \
                     (member {gi} dtype={:?})",
                    g.output_dtype
                ));
            }

            if let Some(&(fp, lp)) = liveness.get(&gi) {
                any_live = true;
                first_phase = first_phase.min(fp);
                last_p = last_p.max(lp);
            }

            if is_split_written(gi) {
                split = true;
            }
        }

        if !any_live {
            // Dead group (in span outputs but no reader). Skip.
            continue;
        }

        if std::env::var("WT_DUMP_PLACER_SLABS")
            .ok()
            .is_some_and(|v| v != "0")
        {
            let member_info: Vec<String> = members
                .iter()
                .map(|&mi| {
                    let g = &groups[mi];
                    format!("{}+{}:{:?}", g.base_id.0, g.count, group_kind[mi])
                })
                .collect();
            eprintln!(
                "[placer slab] atom_range=[{lo}..{hi}) eb={elem_bytes} first={first_phase} last={last_p} members=[{}]",
                member_info.join(",")
            );
        }
        slabs.push(SlabInterval {
            members,
            atom_lo: lo,
            atom_hi: hi,
            elem_bytes,
            dtype,
            first_phase,
            last_phase: last_p,
            split_written: split,
        });
    }

    // ── Step 5: pack slabs into the intermediate buffer ──
    //
    // First-fit allocator over a free list, plus a running set of
    // allocations ordered by end-phase. Before placing a slab, free
    // any entries whose `last_phase < slab.first_phase`.

    // Sort by first_phase ascending, breaking ties by descending size
    // (bigger slabs get choice placement first).
    slabs.sort_by(|a, b| {
        a.first_phase
            .cmp(&b.first_phase)
            .then_with(|| (b.atom_hi - b.atom_lo).cmp(&(a.atom_hi - a.atom_lo)))
    });

    // Placement state.
    #[derive(Debug, Clone)]
    struct Live {
        off: u64,
        size: u64,
        last_phase: usize,
    }
    let mut live: Vec<Live> = Vec::new();
    let mut free_list: Vec<(u64, u64)> = Vec::new(); // (offset, size)
    let mut watermark: u64 = 0;
    let mut peak: u64 = 0;
    let mut slab_placements: Vec<(usize, u64)> = Vec::with_capacity(slabs.len()); // (slab_idx, offset)

    for (si, slab) in slabs.iter().enumerate() {
        // Free any dead entries.
        let mut kept: Vec<Live> = Vec::with_capacity(live.len());
        for l in live.drain(..) {
            if l.last_phase < slab.first_phase {
                free_list.push((l.off, l.size));
            } else {
                kept.push(l);
            }
        }
        live = kept;

        // Coalesce the free list before search.
        free_list.sort_unstable_by_key(|&(off, _)| off);
        let mut merged: Vec<(u64, u64)> = Vec::with_capacity(free_list.len());
        for (off, size) in free_list.drain(..) {
            if let Some(last) = merged.last_mut() {
                if last.0 + last.1 == off {
                    last.1 += size;
                    continue;
                }
            }
            merged.push((off, size));
        }
        free_list = merged;

        let size = (slab.atom_hi - slab.atom_lo) * slab.elem_bytes;
        let align = if slab.split_written {
            CACHE_LINE_BYTES.max(slab.elem_bytes)
        } else {
            slab.elem_bytes.max(1)
        };

        // First-fit over free list.
        let mut chosen: Option<(usize, u64)> = None; // (free_list_idx, aligned_offset)
        for (idx, &(off, sz)) in free_list.iter().enumerate() {
            let aligned = align_up(off, align);
            let pad = aligned - off;
            if sz >= size + pad {
                chosen = Some((idx, aligned));
                break;
            }
        }

        let placement_off = if let Some((idx, aligned)) = chosen {
            let (off, sz) = free_list.swap_remove(idx);
            let pad = aligned - off;
            let remaining = sz - size - pad;
            if pad > 0 {
                free_list.push((off, pad));
            }
            if remaining > 0 {
                free_list.push((aligned + size, remaining));
            }
            aligned
        } else {
            // Bump from watermark.
            let aligned = align_up(watermark, align);
            let pad = aligned - watermark;
            if pad > 0 {
                free_list.push((watermark, pad));
            }
            watermark = aligned + size;
            aligned
        };

        live.push(Live {
            off: placement_off,
            size,
            last_phase: slab.last_phase,
        });
        slab_placements.push((si, placement_off));
        peak = peak.max(watermark);
    }

    // ── Step 6: build literal buffer layout ──
    //
    // Every `Literal`/`LiteralSpan` group — regardless of its final
    // classification — gets a slot in the plan-wide literal buffer at
    // a sequential offset aligned to its element size. The buffer is
    // immutable at runtime: the executor pre-populates it at
    // plan-build and links it in as a read-only source on every
    // execute call. Literal groups whose atoms overlap a model output
    // range were promoted to `BufferKind::Output` in step 1, but we
    // still reserve them a literal-buffer slot here so the JIT can
    // emit a normal copy from the literal buffer to the output buffer
    // (with no execute-time literal-handling logic). Groups whose
    // classification stayed `Literal` use the literal-buffer slot as
    // their primary placement entry; Output-classified literal groups
    // stash the offset in `literal_sources` and keep their primary
    // entry in the output buffer (emitted in the output-entry loop
    // below).
    let mut literal_offsets: Vec<Option<u64>> = vec![None; groups.len()];
    let mut literal_sources: HashMap<AtomId, u64> = HashMap::new();
    let mut literal_watermark: u64 = 0;
    for (gi, g) in groups.iter().enumerate() {
        let is_literal_op = matches!(&g.op, ScalarOp::Literal(_) | ScalarOp::LiteralSpan(_));
        if !is_literal_op {
            continue;
        }
        let eb = g.output_dtype.bytes_per_element() as u64;
        let aligned = align_up(literal_watermark, eb.max(1));
        literal_watermark = aligned + g.count * eb;
        literal_sources.insert(g.base_id, aligned);
        // Primary `LITERAL_BUFFER` entry only for groups whose
        // classification stayed `Literal` AND weren't pulled into an
        // intermediate slab in step 4. Promoted Literals get their
        // primary entry in the intermediate slab emission loop
        // below; the JIT then reads their source bytes from
        // `literal_sources[base]` via an emit-time copy.
        if group_kind[gi] == BufferKind::Literal && !literal_promoted[gi] {
            literal_offsets[gi] = Some(aligned);
        }
    }
    let literal_buffer_size = literal_watermark;

    // ── Step 7: build the placement map entries ──
    //
    // Buffer id assignment is u8-bounded:
    //   0       = intermediate
    //   1       = literal
    //   2..M    = inputs (one per input tensor)
    //   M..N    = outputs (one per output range)
    //   N       = scratch_buffer_id (not a real BufferInfo entry)
    //
    // We check u8 overflow before finalizing.
    let next_id = |count: usize, what: &str| -> Result<u8, String> {
        u8::try_from(count).map_err(|_| format!("placer: {what} exceeds u8 buffer id range"))
    };

    let mut entries: Vec<PlacementEntry> = Vec::new();
    let mut buffers: Vec<BufferInfo> = Vec::new();

    // Slot 0: intermediate.
    buffers.push(BufferInfo {
        id: INTERMEDIATE_BUFFER,
        kind: BufferKind::Intermediate,
        size_bytes: peak,
        name: "intermediate".to_string(),
    });

    // Slot 1: literal.
    buffers.push(BufferInfo {
        id: LITERAL_BUFFER,
        kind: BufferKind::Literal,
        size_bytes: literal_buffer_size,
        name: "literal".to_string(),
    });

    // Input buffers.
    for (ii, it) in input_tensors.iter().enumerate() {
        let bid = BufferId(next_id(buffers.len(), "buffers (input)")?);
        let size = it.count * it.dtype.bytes_per_element() as u64;
        buffers.push(BufferInfo {
            id: bid,
            kind: BufferKind::Input,
            size_bytes: size,
            name: format!("input[{ii}]"),
        });
        entries.push(PlacementEntry {
            atom_base: it.base_id,
            count: it.count,
            dtype: it.dtype,
            buffer_id: bid,
            byte_offset: 0,
        });
    }

    // Output buffers.
    for (oi, range) in all_output_atom_ranges.iter().enumerate() {
        let bid = BufferId(next_id(buffers.len(), "buffers (output)")?);
        let size = range.count * range.dtype.bytes_per_element() as u64;
        buffers.push(BufferInfo {
            id: bid,
            kind: BufferKind::Output,
            size_bytes: size,
            name: format!("output[{oi}]"),
        });
        entries.push(PlacementEntry {
            atom_base: range.base,
            count: range.count,
            dtype: range.dtype,
            buffer_id: bid,
            byte_offset: 0,
        });
    }

    // scratch_buffer_id sits above every real buffer.
    let scratch_buffer_id = next_id(buffers.len(), "buffers (scratch slot)")?;

    // Literal entries.
    for (gi, offset_opt) in literal_offsets.iter().enumerate() {
        if let Some(offset) = *offset_opt {
            let g = &groups[gi];
            entries.push(PlacementEntry {
                atom_base: g.base_id,
                count: g.count,
                dtype: g.output_dtype,
                buffer_id: LITERAL_BUFFER,
                byte_offset: offset,
            });
        }
    }

    // Intermediate entries — one per group in each slab.
    for (si, slab_off) in slab_placements {
        let slab = &slabs[si];
        for &gi in &slab.members {
            let g = &groups[gi];
            let off_in_slab = (g.base_id.0 - slab.atom_lo) * slab.elem_bytes;
            entries.push(PlacementEntry {
                atom_base: g.base_id,
                count: g.count,
                dtype: g.output_dtype,
                buffer_id: INTERMEDIATE_BUFFER,
                byte_offset: slab_off + off_in_slab,
            });
        }
    }

    // Sort entries by atom_base for binary-search lookup.
    entries.sort_by_key(|e| e.atom_base.0);

    // Detect overlapping entries — a bug in the placement map.
    for w in entries.windows(2) {
        let a = &w[0];
        let b = &w[1];
        let a_end = a.atom_base.0 + a.count;
        if a_end > b.atom_base.0 {
            return Err(format!(
                "placer: overlapping entries at atoms {} and {}",
                a.atom_base.0, b.atom_base.0
            ));
        }
    }

    let scratch_count = group_kind
        .iter()
        .filter(|&&k| k == BufferKind::Scratch)
        .count();

    // Diagnostic: top slabs by size.
    let mut diag_slabs: Vec<SlabDiag> = slabs
        .iter()
        .map(|s| {
            let sample_gi = s.members[0];
            let g = &groups[sample_gi];
            SlabDiag {
                size_bytes: (s.atom_hi - s.atom_lo) * s.elem_bytes,
                first_phase: s.first_phase,
                last_phase: s.last_phase,
                member_count: s.members.len(),
                sample_op: op_tag(&g.op).to_string(),
                sample_atom_base: g.base_id.0,
                sample_count: g.count,
            }
        })
        .collect();
    diag_slabs.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));
    diag_slabs.truncate(20);

    Ok(AtomPlacementMap {
        entries,
        buffers,
        scratch_buffer_id,
        intermediate_peak_bytes: peak,
        literal_buffer_size,
        literal_sources,
        intermediate_slab_count: slabs.len(),
        scratch_group_count: scratch_count,
        top_slabs: diag_slabs,
    })
}

fn op_tag(op: &crate::nano_graph::ops::ScalarOp<'_, SystemPool>) -> &'static str {
    use crate::nano_graph::ops::ScalarOp;
    match op {
        ScalarOp::Literal(_) => "Literal",
        ScalarOp::GcLiteral(_) => "GcLiteral",
        ScalarOp::LiteralSpan(_) => "LiteralSpan",
        ScalarOp::Identity => "Identity",
        ScalarOp::Cast { .. } => "Cast",
        ScalarOp::Binary { .. } => "Binary",
        ScalarOp::Unary { .. } => "Unary",
        ScalarOp::Select => "Select",
        ScalarOp::Reduce { .. } => "Reduce",
        ScalarOp::IndirectLoad { .. } => "IndirectLoad",
        ScalarOp::OpaqueOutput { .. } => "OpaqueOutput",
        ScalarOp::SymReduce { .. } => "SymReduce",
    }
}

fn align_up(v: u64, align: u64) -> u64 {
    if align <= 1 {
        return v;
    }
    (v + align - 1) & !(align - 1)
}
