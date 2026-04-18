#![allow(clippy::all, dead_code, unreachable_patterns, unused)]
//! Buffer layout, slot allocation, and validation.
//!
//! Defines [`SlotInfo`] and [`BufferLayout`] — the typed-region map a
//! compiled span uses to find each atom's bytes — plus [`compute_layout`],
//! the liveness-aware slot allocator. Also home to [`EmbeddedTables`]
//! (for `Explicit` InputRef lookup tables baked into the JIT) and
//! [`validate_layout`].
//!
//! Under the memory-placement rewrite, `compute_layout` consumes a global
//! [`AtomPlacementMap`] from the placer: atoms the placer has already
//! assigned to a buffer (input, output, intermediate, literal) are
//! emitted verbatim; everything else lives in lane-private scratch via
//! the per-span `FreeList` allocator. Each span's `BufferBases` table
//! maps every `buffer_id` the span actually touches to a callee-saved
//! GPR the prologue loads from the `buffer_ptrs` argument array.

use std::collections::{BTreeMap, HashMap, HashSet};

use super::placer::{AtomPlacementMap, INTERMEDIATE_BUFFER, LITERAL_BUFFER};
use crate::nano_graph::{AtomGroup, AtomId, AtomRange, InputRef, NanoGraph, ScalarOp};
use crate::numeric_dtype::NumericDType;

// ─── Buffer layout types ────────────────────────────────────────────────────

/// A typed region in the working buffer, addressed at bit granularity.
///
/// Phase 1.B switched the address fields from byte-based to bit-based so
/// that future phases can pack sub-byte values without changing the API.
/// The byte-based helpers ([`SlotInfo::byte_offset`], [`SlotInfo::elem_bytes`])
/// remain available for backends that only handle byte-aligned slots; they
/// debug-assert that the slot actually is byte-aligned.
///
/// Phase 1 invariants (relaxed in phase 6 with bit-packing):
/// - `bit_offset` is a multiple of 8 (slot starts on a byte).
/// - `bit_stride` is a multiple of 8 (sub-byte elements are byte-padded
///   inside a slot, matching the pre-rewrite memory layout exactly).
/// - `elem_bits = dtype.total_bits()` is the *semantic* element width;
///   for sub-byte dtypes (e.g. Bool = 1, I4 = 4) it is **less than**
///   `bit_stride` because phase 1 still pads to a byte boundary.
#[derive(Debug, Clone)]
pub struct SlotInfo {
    /// First AtomId mapped to this slot.
    pub atom_base: AtomId,
    /// Number of elements.
    pub count: u64,
    /// Which buffer the slot lives in. Under the single-buffer per-span
    /// layout (pre memory-placement rework) every slot has
    /// `buffer_id == 0`. Step 4 of the memory-placement rework adds this
    /// field so `emit_compute_bit_offset` can start reporting it, and
    /// step 5 starts populating non-zero values from the global
    /// placement map. See `MEMORY_PLACEMENT.md` §"JIT ABI".
    pub buffer_id: u8,
    /// Bit offset of the first element from the start of the slot's
    /// buffer. Always a multiple of 8 in phase 1 (byte-aligned slot
    /// starts).
    pub bit_offset: u64,
    /// Bits between consecutive elements.
    /// In phase 1, equals `dtype.bytes_per_element() * 8` — sub-byte
    /// types are byte-padded. Phase 6 may pack sub-byte elements by
    /// setting `bit_stride = elem_bits`.
    pub bit_stride: u64,
    /// Semantic element width in bits = `dtype.total_bits()`.
    /// For sub-byte dtypes this is less than `bit_stride` in phase 1.
    pub elem_bits: u64,
    /// Storage dtype.
    pub dtype: NumericDType,
}

impl SlotInfo {
    /// Whether the slot's start AND stride are both byte-multiples.
    /// Phase 1 always returns `true`; phase 6 may return `false` for
    /// bit-packed sub-byte slots.
    #[inline]
    pub fn is_byte_aligned(&self) -> bool {
        self.bit_offset.is_multiple_of(8) && self.bit_stride.is_multiple_of(8)
    }

    /// Byte offset of the first element. Debug-asserts byte alignment.
    /// Backends that don't yet handle bit-packed slots use this and
    /// fail loudly if the invariant breaks.
    #[inline]
    pub fn byte_offset(&self) -> usize {
        debug_assert!(
            self.bit_offset.is_multiple_of(8),
            "SlotInfo::byte_offset called on non-byte-aligned slot \
             (bit_offset={})",
            self.bit_offset
        );
        (self.bit_offset / 8) as usize
    }

    /// Bytes between consecutive elements. Debug-asserts that
    /// `bit_stride` is a multiple of 8.
    #[inline]
    pub fn elem_bytes(&self) -> usize {
        debug_assert!(
            self.bit_stride.is_multiple_of(8),
            "SlotInfo::elem_bytes called with non-byte-multiple bit_stride={}",
            self.bit_stride
        );
        (self.bit_stride / 8) as usize
    }

    /// Bit position of the first element within its starting byte.
    /// Returns 0 for byte-aligned slots (the only kind in phase 1).
    #[inline]
    pub fn bit_in_byte(&self) -> u8 {
        (self.bit_offset % 8) as u8
    }
}

/// Per-span buffer-base register allocation.
///
/// The JIT dedicates `r14` (`BUFFER_PTRS_REG`) to holding the
/// `buffer_ptrs` array pointer for the whole call, so any buffer's
/// base is one indirect load away: `mov <scratch>, QWORD [r14 +
/// id*8]`. For performance, the first few distinct buffer_ids a span
/// touches get **persistent** callee-saved GPRs from
/// [`BUFFER_BASE_REG_POOL`] — the prologue loads each once and the
/// loop bodies address them directly. Buffer_ids beyond the fast
/// pool are **overflow**: the orch layer's `materialize_buffer_base`
/// helper emits a per-access load into a caller-chosen scratch
/// register.
///
/// There is no hard upper bound on the number of buffer_ids a span
/// may reference — the fast pool is a best-effort accelerator, not a
/// correctness constraint.
#[derive(Debug, Clone)]
pub struct BufferBases {
    /// `buffer_id_to_reg[buf_id as usize]` = `Some(reg_code)` when the
    /// span has that `buffer_id` pinned to a fast-pool register,
    /// `None` for overflow buffers (which still count as "seen" but
    /// are fetched per-access via `[r14 + id*8]`). The vec is sized
    /// to `max_buffer_id + 1`; entries outside the seen set are
    /// `None` as well — callers must only query `buffer_id`s the
    /// span actually touches.
    buffer_id_to_reg: Vec<Option<u8>>,
    /// Ordered `(buffer_id, reg_code)` list the prologue emits
    /// `mov reg, [r14 + buf_id*8]` for — the fast-pool assignments
    /// in slot-index order.
    loads: Vec<(u8, u8)>,
}

/// Size of the fast-path buffer base pool. The first `MAX_BUFFER_BASES`
/// distinct `buffer_id`s a span touches get persistent callee-saved
/// GPRs; further buffers go through the overflow path.
pub const MAX_BUFFER_BASES: usize = 4;

/// Callee-saved GPR pool available to hold buffer base pointers.
/// `r12, r15, rbx, rbp` — all callee-saved under System V AMD64 so
/// they survive libm trampolines.
pub const BUFFER_BASE_REG_POOL: [u8; MAX_BUFFER_BASES] = [12, 15, 3, 5];

impl BufferBases {
    /// Build a table from a sorted+deduped set of buffer_ids the span
    /// actually uses. The first [`MAX_BUFFER_BASES`] entries land in
    /// the fast pool; anything beyond is marked as overflow (its
    /// `reg_for_opt` returns `None` so the orch layer knows to emit
    /// a per-access load).
    pub fn assign(buffer_ids: &[u8]) -> Result<Self, String> {
        let max_id = buffer_ids.iter().copied().max().unwrap_or(0) as usize;
        let mut buffer_id_to_reg = vec![None; max_id + 1];
        let mut loads = Vec::with_capacity(buffer_ids.len().min(MAX_BUFFER_BASES));
        for (slot_idx, &buf_id) in buffer_ids.iter().enumerate() {
            if slot_idx < MAX_BUFFER_BASES {
                let reg = BUFFER_BASE_REG_POOL[slot_idx];
                buffer_id_to_reg[buf_id as usize] = Some(reg);
                loads.push((buf_id, reg));
            }
            // Beyond the fast pool: buffer_id_to_reg stays None, the
            // orch layer loads it on demand via r14.
        }
        Ok(BufferBases {
            buffer_id_to_reg,
            loads,
        })
    }

    /// Empty table for spans that emit no body (zero-group graphs).
    pub fn empty() -> Self {
        BufferBases {
            buffer_id_to_reg: Vec::new(),
            loads: Vec::new(),
        }
    }

    /// Look up the fast-pool GPR holding `buffer_id`'s base, if any.
    /// Returns `None` for overflow buffers — callers must emit an
    /// on-demand load via `[r14 + buffer_id*8]` in that case.
    #[inline]
    pub fn reg_for_opt(&self, buffer_id: u8) -> Option<u8> {
        self.buffer_id_to_reg
            .get(buffer_id as usize)
            .copied()
            .flatten()
    }

    /// Look up the fast-pool GPR holding `buffer_id`'s base. Panics
    /// on overflow — this is the legacy API for call sites that
    /// require fast-pool residence (e.g. the reduce inner k-loop,
    /// which can't afford an overflow load per iteration). Such
    /// call sites are responsible for handling the overflow case
    /// upstream (typically by rejecting the span so the caller
    /// falls back to pool_eval).
    #[inline]
    pub fn reg_for_fast(&self, buffer_id: u8) -> u8 {
        self.reg_for_opt(buffer_id).unwrap_or_else(|| {
            panic!(
                "BufferBases::reg_for_fast({buffer_id}): buffer not in fast pool \
                 (overflow buffer — callers that require fast-pool residence \
                 must check reg_for_opt first). loads = {:?}",
                self.loads
            )
        })
    }

    /// Ordered `(buffer_id, reg_code)` pairs the prologue loads into
    /// the fast pool.
    pub fn loads(&self) -> &[(u8, u8)] {
        &self.loads
    }
}

/// Memory layout for a compiled span.
///
/// Maps each atom in the span's graph to a `SlotInfo`. Slots whose
/// `buffer_id` names a placer-owned buffer (input, output, intermediate,
/// literal) carry the placer's offset verbatim; slots with
/// `buffer_id == scratch_buffer_id` live in the lane's per-execute
/// scratch arena at an offset the per-span `FreeList` chose.
pub struct BufferLayout {
    /// All slots sorted by `atom_base` for binary-search lookup.
    slots: Vec<SlotInfo>,
    /// Scratch high-water mark in bytes. Lane scratch arenas are
    /// sized to the max `total_bytes` across spans on the lane.
    pub total_bytes: usize,
    /// Per-buffer-id → callee-saved register assignment, with the
    /// ordered load list the prologue uses.
    pub buffer_bases: BufferBases,
    /// Per-group use counts from liveness analysis. Groups with use_count=0
    /// are dead — their slots may be reused, so the JIT must NOT emit code
    /// for them (their writes would corrupt the new slot occupant).
    pub group_use_counts: Vec<u32>,
    /// Per-group: true if this group's expression should be inlined into its
    /// (single) consumer's loop body instead of being emitted as a standalone
    /// loop. Inlinable groups have NO slot allocated and NO loop emitted —
    /// their consumer is responsible for evaluating their body inline.
    pub inlinable: Vec<bool>,
    /// Per-group: if this group is a Reduce that consumes an inlinable
    /// producer, this is `Some(producer_index)`. The producer's expression is
    /// evaluated in the reduce's inner k-loop instead of loaded from memory.
    pub inlines_producer: Vec<Option<usize>>,
    /// Per-group sym_dims lists (clone of `group.sym_dims`). The JIT
    /// execute path walks this to build the per-group `sym_prods`
    /// array passed to the compiled function as its second argument.
    /// Separate from `AtomGroup::sym_dims` so `BufferLayout` stays
    /// self-contained (no borrow back into the graph at execute
    /// time).
    pub group_sym_dims: Vec<Vec<crate::nano_graph::pattern::GraphConstantId>>,
}

impl BufferLayout {
    /// Empty layout placeholder for spans with no compute (no groups).
    pub(crate) fn empty() -> Self {
        BufferLayout {
            slots: Vec::new(),
            total_bytes: 0,
            buffer_bases: BufferBases::empty(),
            group_use_counts: Vec::new(),
            inlinable: Vec::new(),
            inlines_producer: Vec::new(),
            group_sym_dims: Vec::new(),
        }
    }

    /// Find the slot containing `atom`.
    ///
    /// Returns `(slot, element_index_within_slot)`.
    pub fn find(&self, atom: AtomId) -> Option<(&SlotInfo, u64)> {
        let idx = self.slots.partition_point(|s| s.atom_base.0 <= atom.0);
        if idx == 0 {
            return None;
        }
        let slot = &self.slots[idx - 1];
        if atom.0 < slot.atom_base.0 + slot.count {
            Some((slot, atom.0 - slot.atom_base.0))
        } else {
            None
        }
    }

    /// Byte offset for a specific atom.
    pub fn byte_offset_of(&self, atom: AtomId) -> Option<usize> {
        self.find(atom)
            .map(|(slot, idx)| slot.byte_offset() + idx as usize * slot.elem_bytes())
    }

    /// Write f32 input data into the buffer. Handles ranges spanning multiple slots.
    /// Used by unit tests (F32-only). Production code uses write_scalar_input.
    pub fn write_f32_input(&self, base: AtomId, data: &[f32], buffer: &mut [u8]) {
        use crate::numeric_scalar::NumericScalar;
        let mut written = 0usize;
        let mut atom = base.0;
        while written < data.len() {
            if let Some((slot, elem_start)) = self.find(AtomId(atom)) {
                let available = (slot.count - elem_start) as usize;
                let to_write = available.min(data.len() - written);
                for i in 0..to_write {
                    let off = slot.byte_offset() + (elem_start as usize + i) * slot.elem_bytes();
                    if off + slot.elem_bytes() <= buffer.len() {
                        let scalar = NumericScalar::from_f32(data[written + i]).cast_to(slot.dtype);
                        write_scalar(buffer, off, &scalar);
                    }
                }
                written += to_write;
                atom += to_write as u64;
            } else {
                written += 1;
                atom += 1;
            }
        }
    }

    /// Read output data as f32 from the buffer. Handles ranges spanning multiple slots.
    /// Used by unit tests and diagnostics. Production extraction uses extract_outputs.
    pub fn read_f32_output(&self, range: &AtomRange, buffer: &[u8]) -> Vec<f32> {
        use crate::numeric_scalar::NumericScalar;
        let mut result = Vec::with_capacity(range.count as usize);
        let mut remaining = range.count;
        let mut atom = range.base.0;
        while remaining > 0 {
            if let Some((slot, elem_start)) = self.find(AtomId(atom)) {
                let available = slot.count - elem_start;
                let to_read = remaining.min(available);
                for i in 0..to_read {
                    let off = slot.byte_offset() + (elem_start + i) as usize * slot.elem_bytes();
                    if off + slot.elem_bytes() <= buffer.len() {
                        let scalar = read_scalar(buffer, off, slot.dtype);
                        result.push(scalar.to_f64() as f32);
                    } else {
                        result.push(0.0);
                    }
                }
                atom += to_read;
                remaining -= to_read;
            } else {
                result.push(0.0);
                atom += 1;
                remaining -= 1;
            }
        }
        result
    }
}

/// Write a NumericScalar to `buffer[off..]` in its native byte format.
fn write_scalar(buffer: &mut [u8], off: usize, val: &crate::numeric_scalar::NumericScalar) {
    let bytes = val.as_le_bytes();
    buffer[off..off + bytes.len()].copy_from_slice(bytes);
}

/// Read a NumericScalar from `buffer[off..]` in the given storage dtype.
fn read_scalar(
    buffer: &[u8],
    off: usize,
    dtype: NumericDType,
) -> crate::numeric_scalar::NumericScalar {
    let n = dtype.bytes_per_element();
    let mut bits = [0u8; 8];
    bits[..n].copy_from_slice(&buffer[off..off + n]);
    crate::numeric_scalar::NumericScalar { bits, dtype }
}

/// Legacy: read an f32 from buffer (used by diagnostics only).
fn read_f32_from(buffer: &[u8], off: usize, dtype: NumericDType) -> f32 {
    read_scalar(buffer, off, dtype).to_f64() as f32
}

fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn dtype_elem_bytes(dtype: NumericDType) -> usize {
    dtype.bytes_per_element()
}

// ─── Free-list allocator ────────────────────────────────────────────────────

struct FreeList {
    watermark: usize,
    free_regions: Vec<(usize, usize)>, // (byte_offset, byte_size)
}

impl FreeList {
    fn new() -> Self {
        FreeList {
            watermark: 0,
            free_regions: Vec::new(),
        }
    }

    fn alloc(&mut self, size: usize, align: usize) -> usize {
        // First-fit from free list.
        for i in 0..self.free_regions.len() {
            let (off, sz) = self.free_regions[i];
            let aligned = align_up(off, align);
            let padding = aligned - off;
            if sz >= size + padding {
                self.free_regions.swap_remove(i);
                let remaining = sz - size - padding;
                if remaining > 0 {
                    self.free_regions.push((aligned + size, remaining));
                }
                if padding > 0 {
                    self.free_regions.push((off, padding));
                }
                return aligned;
            }
        }
        // Bump-allocate from watermark.
        let aligned = align_up(self.watermark, align);
        self.watermark = aligned + size;
        aligned
    }

    fn free(&mut self, offset: usize, size: usize) {
        if size > 0 {
            self.free_regions.push((offset, size));
        }
    }
}

fn align_up(v: usize, align: usize) -> usize {
    if align == 0 {
        return v;
    }
    (v + align - 1) & !(align - 1)
}
// ─── Layout computation ─────────────────────────────────────────────────────

/// Compute a slot layout for a span's NanoGraph, combining the placer's
/// global assignments with per-span FreeList scratch.
///
/// Atoms the placer has already placed (input tensors, model outputs,
/// cross-span intermediates, literals) are emitted verbatim with the
/// `(buffer_id, byte_offset)` the placer chose. Everything else runs
/// through the per-span `FreeList` allocator with
/// `buffer_id = placement.scratch_buffer_id`, preserving liveness-aware
/// reuse for span-local intermediates.
///
/// The returned layout's `total_bytes` is the scratch high-water mark
/// for this span — the size the executor must hand this span as a
/// scratch region at dispatch. Cross-span, input, output, and literal
/// atoms contribute nothing to that number: they live in buffers the
/// executor allocates once per execute call (or once per plan, in the
/// literal case).
///
/// Slab coalescing for stride-compatible groups still runs, but only
/// over **scratch** items. Cross-span atoms that need to share a slab
/// have already been coalesced by the global placer.
pub fn compute_layout(
    graph: &NanoGraph<'static, crate::pool::SystemPool>,
    output_ranges: &[AtomRange],
    allow_inline: bool,
    placement: &AtomPlacementMap,
    gc_max_overrides: &std::collections::HashMap<crate::nano_graph::pattern::GraphConstantId, u64>,
) -> Result<BufferLayout, String> {
    let groups = graph.groups();

    // Per-group max_sym_prod. Mirrors `placer::group_max_sym_prod`: for
    // each sym gc take the override if present, else the graph's
    // declared max_value. Used to size span-local slots for sym groups
    // that don't get a placer entry (Scratch groups) so the atom slot
    // has room for the full `max_sym_prod * bpe` footprint the JIT's
    // sym inner loop writes.
    let group_max_syms: Vec<u64> = groups
        .iter()
        .map(|g| super::placer::group_max_sym_prod(graph, g, gc_max_overrides).max(1))
        .collect();

    let n = groups.len();
    let num_inputs = graph.input_tensors().len();
    let total_items = num_inputs + n; // inputs then groups

    let scratch_buffer_id = placement.scratch_buffer_id;

    // Does this span declare `(base..base+count)` as one of its outputs?
    // Split fragments of a Split group overlap one output range each;
    // Duplicate groups are only declared as outputs by lane 0's span.
    let range_overlaps_outputs = |base_id: AtomId, count: u64| -> bool {
        let g_lo = base_id.0;
        let g_hi = g_lo + count;
        output_ranges.iter().any(|r| {
            let r_lo = r.base.0;
            let r_hi = r_lo + r.count;
            g_lo < r_hi && r_lo < g_hi
        })
    };

    // Classify each item (input tensor or group) by how it's laid
    // out in memory:
    //
    // - `Fixed`: input tensors, and group writers that own a
    //   placer-assigned slot and can't be moved (cross-span outputs,
    //   canonical writers for duplicate groups). Fixed items do NOT
    //   participate in slab coalescing — they sit at their placer
    //   offsets.
    //
    // - `Scratch`: ordinary span-local groups. Either get pulled
    //   into a coalesced slab (driven by some consumer's Strided
    //   InputRef) or allocated standalone via the FreeList.
    //
    // - `Literal`: `Literal`/`LiteralSpan` groups. Default to living
    //   in the plan-wide literal buffer at their placer-assigned
    //   offset, with the JIT emitting no store. If a consumer's
    //   Strided InputRef pulls a Literal group into a slab with
    //   non-Literal members, the Literal gets **promoted** to a
    //   scratch slab slot and the JIT emits an ordinary copy from
    //   `placement.literal_sources[base]` into that slot — same
    //   mechanism the Pad-output-overlap path already uses. This
    //   keeps the stride-contiguity guarantee the consumer's
    //   InputRef relies on without carving up the nano graph.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum ItemKind {
        Fixed,
        Scratch,
        Literal,
    }
    let mut item_kinds: Vec<ItemKind> = vec![ItemKind::Fixed; total_items];
    for (gi, group) in groups.iter().enumerate() {
        let placed = placement.byte_offset_of(group.base_id);
        let is_literal = matches!(&group.op, ScalarOp::Literal(_) | ScalarOp::LiteralSpan(_));
        let canonical_writer = range_overlaps_outputs(group.base_id, group.count);
        let kind = if is_literal {
            ItemKind::Literal
        } else if placed.is_some() && canonical_writer {
            ItemKind::Fixed
        } else {
            ItemKind::Scratch
        };
        item_kinds[num_inputs + gi] = kind;
    }

    // ── Step 1: Find contiguity constraints ──
    //
    // For each stride-based InputRef (Affine, StridedBroadcast, Modular)
    // and Reduce stride, compute the accessed atom range. If it spans
    // multiple items (groups/inputs), those items must share a contiguous
    // slab. Union-find tracks connected components.

    let mut parent: Vec<usize> = (0..total_items).collect();
    fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    fn uf_union(parent: &mut [usize], a: usize, b: usize) {
        let ra = uf_find(parent, a);
        let rb = uf_find(parent, b);
        if ra != rb {
            parent[rb] = ra;
        }
    }

    // Map atom → item index. Build a sorted list for binary search.
    // Each entry: (atom_start, atom_end, item_index).
    let mut atom_items: Vec<(u64, u64, usize)> = Vec::new();
    for (ii, it) in graph.input_tensors().iter().enumerate() {
        atom_items.push((it.base_id.0, it.base_id.0 + it.count, ii));
    }
    for (gi, group) in groups.iter().enumerate() {
        atom_items.push((
            group.base_id.0,
            group.base_id.0 + group.count,
            num_inputs + gi,
        ));
    }
    atom_items.sort_by_key(|e| e.0);

    // Find all items overlapping [lo, hi).
    let items_in_range = |lo: u64, hi: u64| -> Vec<usize> {
        let mut result = Vec::new();
        for &(start, end, idx) in &atom_items {
            if start >= hi {
                break;
            }
            if end > lo {
                result.push(idx);
            }
        }
        result
    };

    // Compute the bounding atom range accessed by an InputRef over [atom_offset, atom_offset+count).
    let input_ref_range = |ir: &InputRef, count: u64, atom_offset: u64| -> Option<(u64, u64)> {
        match ir {
            InputRef::Broadcast(_) | InputRef::Explicit(_) => None,
            InputRef::Strided {
                base,
                dim_strides,
                dim_shape,
            } => {
                // Use the closed-form bounds helper, then shift by atom_offset's
                // contribution. Since the inner dims fully cycle for any fragment
                // count > inner_product, we conservatively bound the access over
                // [0, atom_offset + count) and clamp by the atom_offset start.
                let total = atom_offset + count;
                let (lo_full, hi_full) = strided_offset_bounds(dim_strides, dim_shape, total);
                let first = strided_resolve_offset(dim_strides, dim_shape, atom_offset);
                let last = strided_resolve_offset(dim_strides, dim_shape, total - 1);
                // The fragment touches at least [first, last]; combine with the
                // conservative full-range hull and use the union.
                let lo = lo_full.min(first).min(last);
                let hi = hi_full.max(first).max(last);
                let a = base.0 as i64 + lo;
                let b = base.0 as i64 + hi;
                Some((a as u64, b as u64 + 1))
            }
        }
    };

    // Slab coalescing unions items that a Strided consumer requires
    // to be contiguous in memory. `Fixed` items (inputs, cross-span
    // outputs, canonical writers) are excluded — their byte offsets
    // are authoritative and can't be moved into a slab. `Scratch`
    // and `Literal` items are both eligible; if a component ends up
    // with any non-Literal member the whole thing is promoted to a
    // scratch slab, and Literal members of that slab get an
    // on-the-fly copy from the literal buffer emitted by the JIT.
    let coalescable = |items: Vec<usize>| -> Vec<usize> {
        items
            .into_iter()
            .filter(|&i| item_kinds[i] != ItemKind::Fixed)
            .collect()
    };

    for group in groups {
        for ir in &group.inputs {
            if let Some((lo, hi)) = input_ref_range(&ir.input_ref, group.count, group.atom_offset) {
                let items = coalescable(items_in_range(lo, hi));
                if items.len() > 1 {
                    for i in 1..items.len() {
                        uf_union(&mut parent, items[0], items[i]);
                    }
                }
            }
        }

        // Reduce stride extends the access range.
        if let ScalarOp::Reduce {
            reduce_count,
            reduce_stride,
            ..
        } = &group.op
        {
            if *reduce_count > 1 && *reduce_stride != 0 {
                if let Some(InputRef::Strided {
                    base, dim_strides, ..
                }) = group.inputs.first().map(|gi| &gi.input_ref)
                {
                    // Use the innermost stride (last element) for affine-like access
                    let stride = dim_strides.last().copied().unwrap_or(0);
                    let first_i = base.0 as i64 + stride * group.atom_offset as i64;
                    let last_i =
                        base.0 as i64 + stride * (group.atom_offset + group.count - 1) as i64;
                    let end_off = (*reduce_count as i64 - 1) * reduce_stride;
                    let endpoints = [first_i, first_i + end_off, last_i, last_i + end_off];
                    let lo = *endpoints.iter().min().unwrap() as u64;
                    let hi = *endpoints.iter().max().unwrap() as u64 + 1;
                    let items = coalescable(items_in_range(lo, hi));
                    if items.len() > 1 {
                        for i in 1..items.len() {
                            uf_union(&mut parent, items[0], items[i]);
                        }
                    }
                }
            }
        }
    }

    // ── Step 2: Build contiguous slabs for each component ──

    // Collect components: root → list of item indices. BTreeMap for
    // deterministic iteration; per-span slab index ends up in the
    // per-span layout and affects register allocation / first-fit
    // order downstream.
    let mut components: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..total_items {
        let root = uf_find(&mut parent, i);
        components.entry(root).or_default().push(i);
    }

    // For each component with >1 items, build a slab.
    // slab_assignment[item_idx] = Some((slab_idx, offset_within_slab))
    let mut slab_assignment: Vec<Option<(usize, usize)>> = vec![None; total_items];

    struct Slab {
        atom_lo: u64,
        atom_hi: u64, // exclusive
        elem_bytes: usize,
        /// Per-atom byte stride = `elem_bytes * max_sym_prod`. Sym-free
        /// slabs keep this equal to `elem_bytes`; sym slabs (all
        /// members share the same `sym_dims`) reserve full max-bound
        /// footprint so the JIT's sym inner loop has room to write.
        atom_byte_stride: usize,
        byte_offset: usize, // filled during allocation
    }
    let mut slabs: Vec<Slab> = Vec::new();

    for (_, members) in &components {
        if members.len() <= 1 {
            continue;
        }
        // Skip all-Literal components: they're already laid out
        // contiguously in the plan-wide literal buffer (the placer's
        // `literal_sources` walks groups in declaration order, which
        // matches their atom-id order for any consecutive Literal
        // range). Promoting them to scratch just adds a useless copy
        // and another buffer base register. Any component that has
        // at least one non-Literal member is promoted to a scratch
        // slab and Literal members in it get an on-the-fly copy
        // from the literal buffer at JIT emit time.
        let all_literal = members.iter().all(|&m| item_kinds[m] == ItemKind::Literal);
        if all_literal {
            continue;
        }
        // Find atom range, elem_bytes, and max_sym_prod for this
        // component. Sym/non-sym can't mix in one slab (their atom
        // strides differ); sym members with different sym_dims also
        // can't share a slab safely — mirroring the placer's step-3
        // gate. When the component violates these, bail and leave
        // the items to be allocated standalone via the FreeList.
        let mut lo = u64::MAX;
        let mut hi = 0u64;
        let mut eb = 4usize;
        let mut max_sym: u64 = 1;
        let mut first_sym: Option<Vec<crate::nano_graph::pattern::GraphConstantId>> = None;
        let mut component_ok = true;
        for &idx in members {
            let (atom_base, count, dtype, sym_dims_opt) = if idx < num_inputs {
                let it = &graph.input_tensors()[idx];
                (it.base_id.0, it.count, it.dtype, None)
            } else {
                let g = &groups[idx - num_inputs];
                (
                    g.base_id.0,
                    g.count,
                    g.output_dtype,
                    Some(g.sym_dims.clone()),
                )
            };
            lo = lo.min(atom_base);
            hi = hi.max(atom_base + count);
            eb = eb.max(dtype_elem_bytes(dtype));
            if let Some(sd) = sym_dims_opt {
                if !sd.is_empty() {
                    match &first_sym {
                        None => {
                            first_sym = Some(sd.clone());
                            max_sym = group_max_syms[idx - num_inputs].max(1);
                        }
                        Some(prev) => {
                            if prev != &sd {
                                component_ok = false;
                                break;
                            }
                        }
                    }
                } else if first_sym.is_some() {
                    // Mixing sym with non-sym in one slab is unsafe
                    // (different atom strides). Skip and fall back.
                    component_ok = false;
                    break;
                }
            }
        }
        if !component_ok {
            continue;
        }
        let atom_byte_stride = eb * max_sym as usize;
        let slab_idx = slabs.len();
        slabs.push(Slab {
            atom_lo: lo,
            atom_hi: hi,
            elem_bytes: eb,
            atom_byte_stride,
            byte_offset: 0,
        });
        for &idx in members {
            let atom_base = if idx < num_inputs {
                graph.input_tensors()[idx].base_id.0
            } else {
                groups[idx - num_inputs].base_id.0
            };
            let off_in_slab = (atom_base - lo) as usize * atom_byte_stride;
            slab_assignment[idx] = Some((slab_idx, off_in_slab));
        }
    }

    // ── Step 3: Compute liveness + producer indices ──

    let mut use_counts = vec![0u32; n];
    let mut producer_lists: Vec<Vec<usize>> = Vec::with_capacity(n);
    // unique_consumer[pi] = Some(ci) if pi has exactly one intra-span consumer ci.
    // Set to None if 0 consumers, or as soon as a 2nd distinct consumer is seen.
    let mut unique_consumer: Vec<Option<usize>> = vec![None; n];
    let mut multi_consumer = vec![false; n];
    for (gi, group) in groups.iter().enumerate() {
        let mut producers = HashSet::new();
        graph.collect_all_producer_indices(group, gi, &mut producers);
        for &pi in &producers {
            use_counts[pi] += 1;
            if !multi_consumer[pi] {
                match unique_consumer[pi] {
                    None => unique_consumer[pi] = Some(gi),
                    Some(prev) if prev == gi => {}
                    Some(_) => {
                        multi_consumer[pi] = true;
                        unique_consumer[pi] = None;
                    }
                }
            }
        }
        producer_lists.push(producers.into_iter().collect());
    }

    // Pin output groups: bump use_count for any group whose atom range overlaps
    // an output range. The previous implementation built a HashSet of every
    // individual output atom and looked up each group atom — O(N_atoms) per
    // span. For phase 233 (LM-head Mul, 6.3M atoms × 8 lanes) that was the
    // dominant cost in compile_phases. Range overlap is O(groups × output_ranges)
    // which is tiny by comparison.
    for (gi, group) in groups.iter().enumerate() {
        let g_lo = group.base_id.0;
        let g_hi = g_lo + group.count;
        for r in output_ranges {
            let r_lo = r.base.0;
            let r_hi = r_lo + r.count;
            if g_lo < r_hi && r_lo < g_hi {
                use_counts[gi] += 1;
                break;
            }
        }
    }

    // ── Step 3b: Identify inlinable groups (reduce-fold inlining) ──
    //
    // A group is inlinable iff its expression can be folded directly into its
    // single consumer's loop body, eliminating the materialized intermediate.
    // Stage 1 only inlines pure-scalar producers into Reduce consumers where
    // the consumer reads the producer via a stride-K affine InputRef and
    // K matches the consumer's reduce_count (so each producer atom is read
    // exactly once). The resulting loop is the textbook "reduction in
    // registers" form: no buffer roundtrip for the intermediate.
    //
    // Stage 1 restriction: the producer's own inputs must NOT reference other
    // inlinable groups (no recursive inlining yet — handled in a follow-up).
    let inline_disabled = !allow_inline;
    let mut inlinable = vec![false; n];
    let mut inlines_producer: Vec<Option<usize>> = vec![None; n];
    if !inline_disabled {
        for (gi, group) in groups.iter().enumerate() {
            // Single intra-span consumer, not output-pinned.
            if use_counts[gi] != 1 {
                continue;
            }
            let Some(ci) = unique_consumer[gi] else {
                continue;
            };

            // Producer must be pure scalar (no Reduce, IndirectLoad, Literal,
            // OpaqueOutput) — its body must be a closed-form expression we can
            // re-emit at any iteration index.
            if !matches!(
                &group.op,
                ScalarOp::Binary { .. }
                    | ScalarOp::Unary { .. }
                    | ScalarOp::Select
                    | ScalarOp::Identity
                    | ScalarOp::Cast { .. }
            ) {
                continue;
            }

            // Consumer must be a Reduce with reduce_stride == 1 (the relayouted
            // matmul shape and other "natural" reductions over a contiguous
            // producer range).
            let consumer = &groups[ci];
            let (k_count, _kind) = match &consumer.op {
                ScalarOp::Reduce {
                    reduce_count,
                    reduce_stride,
                    kind,
                    ..
                } if *reduce_stride == 1 && *reduce_count > 0 => (*reduce_count, *kind),
                _ => continue,
            };

            // Consumer's reduce input must be a 1D Strided ref into this
            // producer with dim_strides=[k_count] (each consumer output reads
            // a contiguous K-block of producer atoms).
            if consumer.inputs.len() != 1 {
                continue;
            }
            let InputRef::Strided {
                base,
                dim_strides,
                dim_shape,
            } = &consumer.inputs[0].input_ref
            else {
                continue;
            };
            if *base != group.base_id {
                continue;
            }
            if dim_strides.len() != 1 || dim_strides[0] != k_count as i64 {
                continue;
            }
            if dim_shape.len() != 1 || dim_shape[0] != u64::MAX {
                continue;
            }

            // Producer's atom range must be exactly K * consumer's range.
            if group.count != k_count * consumer.count {
                continue;
            }
            if group.atom_offset != k_count * consumer.atom_offset {
                continue;
            }

            // Stage 1 restriction: producer's inputs must not reference any
            // other inlinable group. Because we process groups in topological
            // order, all upstream inlinables have already been marked.
            let mut all_buffer_inputs = true;
            'inputs: for inp in &group.inputs {
                let bases: &[AtomId] = match &inp.input_ref {
                    InputRef::Broadcast(a) => std::slice::from_ref(a),
                    InputRef::Strided { base, .. } => std::slice::from_ref(base),
                    InputRef::Explicit(ids) => ids.as_slice(),
                };
                for b in bases {
                    // Look up which group (if any) contains this base atom.
                    // O(n) scan is fine — n is small per span.
                    for (qi, q) in groups.iter().enumerate() {
                        if b.0 >= q.base_id.0 && b.0 < q.base_id.0 + q.count && inlinable[qi] {
                            all_buffer_inputs = false;
                            break 'inputs;
                        }
                    }
                }
            }
            if !all_buffer_inputs {
                continue;
            }

            inlinable[gi] = true;
            inlines_producer[ci] = Some(gi);
        }
    }

    // ── Step 4: Allocate slots ──

    let trace_byte = std::env::var("TRACE_BYTE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());

    let mut allocator = FreeList::new();
    let mut all_slots = Vec::new();
    let mut slab_allocated = vec![false; slabs.len()];

    // Helper: ensure a slab is allocated, return its byte_offset.
    fn ensure_slab(
        slabs: &mut [Slab],
        slab_allocated: &mut [bool],
        allocator: &mut FreeList,
        slab_idx: usize,
    ) -> usize {
        if !slab_allocated[slab_idx] {
            let slab = &slabs[slab_idx];
            let size = (slab.atom_hi - slab.atom_lo) as usize * slab.atom_byte_stride;
            let offset = allocator.alloc(size, slab.atom_byte_stride.max(slab.elem_bytes));
            slabs[slab_idx].byte_offset = offset;
            slab_allocated[slab_idx] = true;
        }
        slabs[slab_idx].byte_offset
    }

    // Input tensor slots. The placer owns every input tensor's layout
    // unconditionally, so we read the assigned (buffer_id, byte_offset)
    // directly and skip the FreeList. Slab assignment from step 2
    // cannot apply because input items are `ItemKind::Fixed` and were
    // excluded from the coalescing filter.
    for it in graph.input_tensors().iter() {
        let elem_bytes = dtype_elem_bytes(it.dtype);
        let elem_bits_semantic = it.dtype.total_bits() as u64;
        let (buf_id, byte_off) = placement.byte_offset_of(it.base_id).ok_or_else(|| {
            format!(
                "compute_layout: input tensor base={} not in placement map",
                it.base_id.0
            )
        })?;
        // Atom-to-atom stride must follow the placer — when the input
        // points into a max-stride intermediate slab (e.g. a main-graph
        // sym group that this JIT span reads via its input tensor),
        // atoms sit at `max_sym_prod * bpe` bytes apart, not `bpe`.
        let atom_stride_bytes = placement
            .atom_byte_stride_of(it.base_id)
            .unwrap_or(elem_bytes as u64);
        all_slots.push(SlotInfo {
            atom_base: it.base_id,
            count: it.count,
            buffer_id: buf_id.0,
            bit_offset: byte_off * 8,
            bit_stride: atom_stride_bytes * 8,
            elem_bits: elem_bits_semantic,
            dtype: it.dtype,
        });
    }

    // Group slots.
    let mut group_slot_indices = vec![0usize; n];
    let mut remaining = use_counts.clone();

    for (gi, group) in groups.iter().enumerate() {
        // Inlinable groups have no slot — their value lives in registers
        // inside the consumer's loop body. Skip allocation entirely.
        if inlinable[gi] {
            group_slot_indices[gi] = usize::MAX;
            continue;
        }

        let elem_bytes = dtype_elem_bytes(group.output_dtype);
        let elem_bits_semantic = group.output_dtype.total_bits() as u64;
        let item_idx = num_inputs + gi;
        let slot_idx = all_slots.len();

        // Slot allocation, ordered by precedence:
        //
        // 1. **Slab** — if the item was pulled into a scratch slab
        //    during coalescing, the slab slot wins unconditionally.
        //    This includes Literal groups that got promoted: their
        //    primary slot becomes the slab offset in scratch, and
        //    `emit_group`'s Literal path will emit an on-the-fly
        //    copy from `placement.literal_sources[base]` into that
        //    slot at execute time.
        //
        // 2. **Placer slot** — for Literal groups that stayed out
        //    of any slab (they live in the plan-wide literal buffer)
        //    and for Fixed items (cross-span outputs / canonical
        //    writers that own placer-assigned offsets).
        //
        // 3. **Standalone FreeList scratch** — for ordinary Scratch
        //    items that didn't get coalesced, and for non-canonical
        //    writers of duplicate groups (which recompute the value
        //    locally without touching the placer's shared slot).
        if let Some((slab_idx, off_in_slab)) = slab_assignment[item_idx] {
            let slab_base = ensure_slab(&mut slabs, &mut slab_allocated, &mut allocator, slab_idx);
            all_slots.push(SlotInfo {
                atom_base: group.base_id,
                count: group.count,
                buffer_id: scratch_buffer_id,
                bit_offset: ((slab_base + off_in_slab) as u64) * 8,
                bit_stride: (slabs[slab_idx].atom_byte_stride as u64) * 8,
                elem_bits: elem_bits_semantic,
                dtype: group.output_dtype,
            });
        } else if matches!(item_kinds[item_idx], ItemKind::Fixed | ItemKind::Literal) {
            let (buf_id, byte_off) = placement.byte_offset_of(group.base_id).ok_or_else(|| {
                format!(
                    "compute_layout: group base={} (kind={:?}) missing from placement map",
                    group.base_id, item_kinds[item_idx]
                )
            })?;
            // Respect the placer's atom stride (see matching note on
            // the input_tensors loop above).
            let atom_stride_bytes = placement
                .atom_byte_stride_of(group.base_id)
                .unwrap_or(elem_bytes as u64);
            all_slots.push(SlotInfo {
                atom_base: group.base_id,
                count: group.count,
                buffer_id: buf_id.0,
                bit_offset: byte_off * 8,
                bit_stride: atom_stride_bytes * 8,
                elem_bits: elem_bits_semantic,
                dtype: group.output_dtype,
            });
        } else {
            // Sym groups need room for `max_sym_prod` elements per atom
            // (matching the placer's intermediate-slab stride). Sym-free
            // groups keep the old `count * bpe` tight packing.
            let max_sym_prod = group_max_syms[gi] as usize;
            let atom_stride_bytes = elem_bytes * max_sym_prod;
            let size = group.count as usize * atom_stride_bytes;
            let offset = allocator.alloc(size, atom_stride_bytes.max(elem_bytes));
            all_slots.push(SlotInfo {
                atom_base: group.base_id,
                count: group.count,
                buffer_id: scratch_buffer_id,
                bit_offset: (offset as u64) * 8,
                bit_stride: (atom_stride_bytes as u64) * 8,
                elem_bits: elem_bits_semantic,
                dtype: group.output_dtype,
            });
        }

        group_slot_indices[gi] = slot_idx;

        if let Some(tb) = trace_byte {
            let s = &all_slots[slot_idx];
            let end = s.byte_offset() + s.count as usize * s.elem_bytes();
            if s.byte_offset() <= tb && end > tb {
                eprintln!(
                    "  ALLOC group {} base={} at [{}-{}) eb={} dtype={:?} op={:?}",
                    gi,
                    group.base_id,
                    s.byte_offset(),
                    end,
                    s.elem_bytes(),
                    s.dtype,
                    op_name_short(&group.op)
                );
            }
        }

        // Slot reuse: disabled pending investigation of slab/freelist
        // interaction that causes segfaults. The producer_lists and remaining
        // arrays are computed but not used for freeing. TODO: fix this properly.
    }

    all_slots.sort_by_key(|s| s.atom_base.0);

    // Build the per-span BufferBases table from the distinct buffer_ids
    // the slots actually use.
    //
    // If this span contains any `Literal`/`LiteralSpan` group whose
    // primary slot is a non-literal buffer (Pad-style: the literal's
    // atoms overlap a model output range), the JIT emits an ordinary
    // copy from the literal buffer to that slot and needs
    // `LITERAL_BUFFER` in the used-set. The prologue will place it
    // in the fast pool if there's room, or the orch layer will fetch
    // it via the overflow path at each copy iteration.
    let mut used_ids: Vec<u8> = all_slots.iter().map(|s| s.buffer_id).collect();
    let needs_literal_base = groups.iter().any(|g| {
        if !matches!(&g.op, ScalarOp::Literal(_) | ScalarOp::LiteralSpan(_)) {
            return false;
        }
        match placement.byte_offset_of(g.base_id) {
            Some((buf, _)) => buf != LITERAL_BUFFER,
            None => false,
        }
    });
    if needs_literal_base {
        used_ids.push(LITERAL_BUFFER.0);
    }
    used_ids.sort_unstable();
    used_ids.dedup();
    let buffer_bases = BufferBases::assign(&used_ids)?;

    let group_sym_dims: Vec<Vec<crate::nano_graph::pattern::GraphConstantId>> =
        graph.groups().iter().map(|g| g.sym_dims.clone()).collect();

    Ok(BufferLayout {
        total_bytes: allocator.watermark,
        slots: all_slots,
        buffer_bases,
        group_use_counts: use_counts,
        inlinable,
        inlines_producer,
        group_sym_dims,
    })
}
/// Read a NumericScalar from raw bytes in a given dtype.
fn read_scalar_raw(data: &[u8], dtype: NumericDType) -> crate::numeric_scalar::NumericScalar {
    let n = dtype.bytes_per_element();
    let mut bits = [0u8; 8];
    bits[..n].copy_from_slice(&data[..n]);
    crate::numeric_scalar::NumericScalar { bits, dtype }
}
pub fn validate_layout(
    graph: &NanoGraph<'static, crate::pool::SystemPool>,
    layout: &BufferLayout,
) -> Vec<String> {
    let mut errors = Vec::new();
    for (gi, group) in graph.groups().iter().enumerate() {
        for (ii, ir) in group.inputs.iter().enumerate() {
            match &ir.input_ref {
                InputRef::Strided {
                    base, dim_strides, ..
                // Only validate 1D Affine patterns. N-D Strided InputRefs use
                // a non-linear address function (per-dim modulus + accumulate)
                // and `dim_strides.last()` is not the access stride; the layout
                // construction handles them via `input_ref_range`/strided_offset_bounds.
                } if dim_strides.len() == 1 => {
                    let stride = dim_strides[0];
                    if stride != 0 {
                        let first_atom = (base.0 as i64 + stride * group.atom_offset as i64) as u64;
                        let last_atom = (base.0 as i64
                            + stride * (group.atom_offset + group.count - 1) as i64)
                            as u64;
                        let lo = first_atom.min(last_atom);
                        let hi = first_atom.max(last_atom);
                        if let (Some((slot_lo, _)), Some((slot_hi, _))) =
                            (layout.find(AtomId(lo)), layout.find(AtomId(hi)))
                        {
                            if slot_lo.atom_base != slot_hi.atom_base
                                && slot_lo.elem_bytes() == slot_hi.elem_bytes()
                            {
                                let atom_delta =
                                    slot_hi.atom_base.0 as i64 - slot_lo.atom_base.0 as i64;
                                let byte_delta =
                                    slot_hi.byte_offset() as i64 - slot_lo.byte_offset() as i64;
                                let expected_byte_delta = atom_delta * slot_lo.elem_bytes() as i64;
                                if byte_delta != expected_byte_delta {
                                    errors.push(format!(
                                        "group {} input {} Affine(base={},stride={},count={},off={}): atoms {}..{} slots {} and {} byte_delta={} expected={}",
                                        gi, ii, base, stride, group.count, group.atom_offset,
                                        lo, hi, slot_lo.atom_base, slot_hi.atom_base,
                                        byte_delta, expected_byte_delta,
                                    ));
                                }
                            }
                        }
                    }
                }
                _ => {}
            }

            // Also check Reduce stride.
            if let ScalarOp::Reduce {
                reduce_count,
                reduce_stride,
                ..
            } = &group.op
            {
                if *reduce_count > 1 && *reduce_stride != 0 {
                    if let InputRef::Strided {
                        base, dim_strides, ..
                    } = &ir.input_ref
                    {
                        let stride = dim_strides.last().copied().unwrap_or(0);
                        let first = (base.0 as i64 + stride * group.atom_offset as i64) as u64;
                        let last = (base.0 as i64
                            + stride * (group.atom_offset + group.count - 1) as i64)
                            as u64;
                        let end_off = (*reduce_count as i64 - 1) * reduce_stride;
                        let endpoints = [
                            first,
                            (first as i64 + end_off) as u64,
                            last,
                            (last as i64 + end_off) as u64,
                        ];
                        let lo = *endpoints.iter().min().unwrap();
                        let hi = *endpoints.iter().max().unwrap();
                        if let (Some((slot_lo, _)), Some((slot_hi, _))) =
                            (layout.find(AtomId(lo)), layout.find(AtomId(hi)))
                        {
                            if slot_lo.atom_base != slot_hi.atom_base
                                && slot_lo.elem_bytes() == slot_hi.elem_bytes()
                            {
                                let atom_delta =
                                    slot_hi.atom_base.0 as i64 - slot_lo.atom_base.0 as i64;
                                let byte_delta =
                                    slot_hi.byte_offset() as i64 - slot_lo.byte_offset() as i64;
                                let expected = atom_delta * slot_lo.elem_bytes() as i64;
                                if byte_delta != expected {
                                    errors.push(format!(
                                        "group {} Reduce stride: atoms {}..{} slots {} and {} byte_delta={} expected={}",
                                        gi, lo, hi, slot_lo.atom_base, slot_hi.atom_base, byte_delta, expected
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    errors
}
pub(crate) fn strided_resolve_offset(dim_strides: &[i64], dim_shape: &[u64], i: u64) -> i64 {
    let nd = dim_strides.len();
    let mut offset = 0i64;
    let mut remaining = i;
    for d in (0..nd).rev() {
        let coord = if d == 0 {
            remaining
        } else {
            let c = remaining % dim_shape[d];
            remaining /= dim_shape[d];
            c
        };
        offset += coord as i64 * dim_strides[d];
    }
    offset
}

/// Compute conservative offset bounds for an N-D Strided InputRef accessed
/// over `count` consecutive consumer atoms.
///
/// Treats inner dims (`d > 0`) as fully cycled (coord ∈ [0, dim_shape[d])) and
/// the outermost dim as ranging over `ceil(count / inner_product)` coords. The
/// resulting `[lo, hi]` is a safe over-estimate of the actual atom-offset
/// range, which is fine for the buffer-layout slab union — it never under-
/// counts an access.
fn strided_offset_bounds(dim_strides: &[i64], dim_shape: &[u64], count: u64) -> (i64, i64) {
    let nd = dim_strides.len();
    let mut lo: i64 = 0;
    let mut hi: i64 = 0;
    let mut inner_product: u64 = 1;
    for d in (1..nd).rev() {
        let s = dim_strides[d];
        let max_coord = dim_shape[d].saturating_sub(1) as i64;
        let contrib = s.saturating_mul(max_coord);
        if contrib > 0 {
            hi = hi.saturating_add(contrib);
        } else if contrib < 0 {
            lo = lo.saturating_add(contrib);
        }
        inner_product = inner_product.saturating_mul(dim_shape[d]);
    }
    if nd >= 1 {
        let max_outer = if inner_product == 0 || count == 0 {
            0
        } else {
            count.div_ceil(inner_product).saturating_sub(1)
        };
        let s = dim_strides[0];
        let contrib = s.saturating_mul(max_outer as i64);
        if contrib > 0 {
            hi = hi.saturating_add(contrib);
        } else if contrib < 0 {
            lo = lo.saturating_add(contrib);
        }
    }
    (lo, hi)
}
pub struct EmbeddedTables {
    /// Next available byte offset in the buffer (starts at layout.total_bytes).
    watermark: usize,
    /// Table entries to append to the literal template.
    entries: Vec<EmbeddedTableEntry>,
}

struct EmbeddedTableEntry {
    byte_offset: usize,
    data: Vec<u8>,
}

impl EmbeddedTables {
    pub(crate) fn new(initial_watermark: usize) -> Self {
        EmbeddedTables {
            watermark: initial_watermark,
            entries: Vec::new(),
        }
    }

    /// Allocate space for a table and return its byte offset in the buffer.
    pub(crate) fn alloc(&mut self, data: Vec<u8>) -> usize {
        // Align to 8 bytes for i64 entries.
        let aligned = (self.watermark + 7) & !7;
        let offset = aligned;
        self.watermark = aligned + data.len();
        self.entries.push(EmbeddedTableEntry {
            byte_offset: offset,
            data,
        });
        offset
    }

    /// Total buffer size including all embedded tables.
    pub(crate) fn total_bytes(&self) -> usize {
        self.watermark
    }

    /// Write all table data into the buffer.
    pub(crate) fn populate(&self, buffer: &mut [u8]) {
        for entry in &self.entries {
            let end = entry.byte_offset + entry.data.len();
            if end <= buffer.len() {
                buffer[entry.byte_offset..end].copy_from_slice(&entry.data);
            }
        }
    }
}
pub(crate) fn op_name_short(op: &ScalarOp) -> &'static str {
    match op {
        ScalarOp::Literal(_) => "Lit",
        ScalarOp::GcLiteral(_) => "GcLit",
        ScalarOp::Identity | ScalarOp::Cast { .. } => "Id",
        ScalarOp::Binary { .. } => "Bin",
        ScalarOp::Unary { .. } => "Un",
        ScalarOp::Select => "Sel",
        ScalarOp::Reduce { .. } => "Red",
        ScalarOp::IndirectLoad { .. } => "Ind",
        ScalarOp::OpaqueOutput { .. } => "Opq",
        ScalarOp::LiteralSpan(_) => "LitS",
        ScalarOp::SymReduce { .. } => "SRed",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::attempts::v14::placer::run_placer;
    use crate::compiler::attempts::v14::types::{Phase, Span};
    use crate::graph::GlobalId;
    use crate::nano_graph::ops::ScalarOp;
    use crate::nano_graph::pattern::GroupInput;
    use crate::numeric_scalar::NumericScalar;

    /// Build a trivial placement (one phase, one span containing the
    /// whole graph) so compute_layout can run in unit tests.
    pub(crate) fn test_placement(
        graph: &NanoGraph<'static, crate::pool::SystemPool>,
        outputs: &[AtomRange],
    ) -> AtomPlacementMap {
        let span_inputs: Vec<AtomRange> = graph
            .input_tensors()
            .iter()
            .map(|it| AtomRange {
                base: it.base_id,
                count: it.count,
                dtype: it.dtype,
            })
            .collect();
        let phases = vec![Phase {
            spans: vec![Span {
                graph: graph.clone(),
                inputs: span_inputs,
                outputs: outputs.to_vec(),
            }],
        }];
        run_placer(graph, &phases, outputs, &std::collections::HashMap::new())
            .expect("placer failed in test helper")
    }

    // ─── SlotInfo helper invariants ─────────────────────────────────────────

    #[test]
    fn slot_info_helpers_byte_aligned_native_dtypes() {
        for dtype in [
            NumericDType::F32,
            NumericDType::F64,
            NumericDType::I32,
            NumericDType::I64,
            NumericDType::U8,
            NumericDType::BF16,
            NumericDType::F16,
            NumericDType::BOOL, // 1-bit semantic, byte-padded
        ] {
            let slot = SlotInfo {
                atom_base: AtomId(0),
                count: 4,
                buffer_id: 0,
                bit_offset: 16,
                bit_stride: (dtype.bytes_per_element() as u64) * 8,
                elem_bits: dtype.total_bits() as u64,
                dtype,
            };
            assert!(slot.is_byte_aligned(), "{dtype:?}");
            assert_eq!(slot.byte_offset(), 2);
            assert_eq!(slot.elem_bytes(), dtype.bytes_per_element());
            assert_eq!(slot.bit_in_byte(), 0);
        }
    }

    #[test]
    fn slot_info_bool_semantic_vs_padded_widths() {
        let slot = SlotInfo {
            atom_base: AtomId(7),
            count: 3,
            buffer_id: 0,
            bit_offset: 24,
            bit_stride: 8,
            elem_bits: 1,
            dtype: NumericDType::BOOL,
        };
        assert!(slot.is_byte_aligned());
        assert_eq!(slot.byte_offset(), 3);
        assert_eq!(slot.elem_bytes(), 1);
        assert_eq!(slot.elem_bits, 1);
        assert_eq!(slot.bit_stride, 8);
    }

    #[test]
    fn slot_info_bit_packed_phase6_shape() {
        let slot = SlotInfo {
            atom_base: AtomId(0),
            count: 16,
            buffer_id: 0,
            bit_offset: 0,
            bit_stride: 1,
            elem_bits: 1,
            dtype: NumericDType::BOOL,
        };
        assert!(!slot.is_byte_aligned());
    }

    // ─── compute_layout shape with Bool input ───────────────────────────────

    #[test]
    fn compute_layout_bool_input_slot_shape() {
        let mut g = NanoGraph::new();
        let bool_input = g.add_input_tensor(GlobalId(0), 5, NumericDType::BOOL);
        let identity = g.push_group(
            5,
            NumericDType::BOOL,
            ScalarOp::Identity,
            vec![],
            vec![GroupInput::scalar(InputRef::affine(bool_input, 1))],
        );
        let outputs = vec![AtomRange {
            base: identity,
            count: 5,
            dtype: NumericDType::BOOL,
        }];
        let placement = test_placement(&g, &outputs);
        let layout = compute_layout(
            &g,
            &outputs,
            true,
            &placement,
            &std::collections::HashMap::new(),
        )
        .expect("layout");

        let (input_slot, _) = layout.find(bool_input).expect("bool input slot present");
        assert_eq!(input_slot.dtype, NumericDType::BOOL);
        assert_eq!(input_slot.elem_bits, 1);
        assert_eq!(input_slot.bit_stride, 8);
        assert!(input_slot.is_byte_aligned());

        let (out_slot, _) = layout.find(identity).expect("identity slot present");
        assert_eq!(out_slot.dtype, NumericDType::BOOL);
        assert_eq!(out_slot.elem_bits, 1);
        assert_eq!(out_slot.bit_stride, 8);
    }
}
