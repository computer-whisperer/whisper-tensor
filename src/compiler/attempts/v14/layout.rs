#![allow(clippy::all, dead_code, unreachable_patterns, unused)]
//! Buffer layout, slot allocation, marshalling, and validation.
//!
//! Defines [`SlotInfo`] and [`BufferLayout`] — the typed-region map a
//! compiled span uses to find each atom's bytes inside the working
//! buffer — plus [`compute_layout`], the liveness-aware slot allocator.
//! Also home to [`EmbeddedTables`] (for `Explicit` InputRef lookup
//! tables baked into the JIT buffer), the marshalling functions
//! [`write_store_slice_to_buffer`] / [`read_buffer_to_output`], and
//! [`validate_layout`].
//!
//! This module is **backend-agnostic**. Both `codegen.rs` (the
//! Cranelift backend) and the new `x86_jit/` backend import from here.
//! Phase 1.A is a pure mechanical extraction from `codegen.rs` — no
//! behavioral changes. Phase 1.B rewrites [`SlotInfo`] to be
//! bit-addressed; phases 2+ build the new x86_jit on top.

use std::collections::{HashMap, HashSet};

use super::executor::{SpanOutput, StoreSlice};
use crate::nano_graph::{AtomGroup, AtomId, AtomRange, InputRef, NanoGraph, ScalarOp};
use crate::numeric_dtype::NumericDType;
use crate::numeric_scalar::{NumericScalarView, NumericScalarViewMut};

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

/// Memory layout for a compiled span.
///
/// Maps each group output and input tensor to a `SlotInfo` in the working
/// buffer. Computed with liveness-aware reuse: group output slots are freed
/// when all downstream consumers have been emitted.
pub struct BufferLayout {
    /// All slots sorted by `atom_base` for binary-search lookup.
    slots: Vec<SlotInfo>,
    /// Total buffer size in bytes (high-water mark of the allocator).
    pub total_bytes: usize,
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
}

impl BufferLayout {
    /// Empty layout placeholder for spans with no compute (no groups).
    pub(crate) fn empty() -> Self {
        BufferLayout {
            slots: Vec::new(),
            total_bytes: 0,
            group_use_counts: Vec::new(),
            inlinable: Vec::new(),
            inlines_producer: Vec::new(),
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

    /// Write literal group values into the buffer.
    pub fn populate_literals(
        &self,
        graph: &NanoGraph<'static, crate::pool::SystemPool>,
        buffer: &mut [u8],
    ) {
        use crate::numeric_scalar::NumericScalar;
        for group in graph.groups() {
            match &group.op {
                ScalarOp::Literal(scalar) => {
                    if let Some((slot, _)) = self.find(group.base_id) {
                        // Cast the literal to the slot's storage dtype, then write raw bytes.
                        let stored = scalar.cast_to(slot.dtype);
                        for i in 0..group.count {
                            let off = slot.byte_offset() + i as usize * slot.elem_bytes();
                            write_scalar(buffer, off, &stored);
                        }
                    }
                }
                ScalarOp::LiteralSpan(tensor) => {
                    if let Some((slot, _)) = self.find(group.base_id) {
                        for i in 0..group.count {
                            let scalar = tensor.read_element(i as usize);
                            let stored = scalar.cast_to(slot.dtype);
                            let off = slot.byte_offset() + i as usize * slot.elem_bytes();
                            write_scalar(buffer, off, &stored);
                        }
                    }
                }
                _ => {}
            }
        }
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

/// Compute a liveness-aware buffer layout for a span's NanoGraph.
///
/// Groups/inputs whose atoms are accessed by stride-based InputRefs spanning
/// multiple groups are coalesced into contiguous slabs (atom-ID-proportional
/// offsets within the slab). This guarantees that stride arithmetic works.
/// Other groups use liveness-based slot reuse for memory efficiency.
pub fn compute_layout(
    graph: &NanoGraph<'static, crate::pool::SystemPool>,
    output_ranges: &[AtomRange],
    allow_inline: bool,
) -> BufferLayout {
    let groups = graph.groups();
    let n = groups.len();
    let num_inputs = graph.input_tensors().len();
    let total_items = num_inputs + n; // inputs then groups

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

    for group in groups {
        for ir in &group.inputs {
            if let Some((lo, hi)) = input_ref_range(ir, group.count, group.atom_offset) {
                let items = items_in_range(lo, hi);
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
                }) = group.inputs.first()
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
                    let items = items_in_range(lo, hi);
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

    // Collect components: root → list of item indices.
    let mut components: HashMap<usize, Vec<usize>> = HashMap::new();
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
        byte_offset: usize, // filled during allocation
    }
    let mut slabs: Vec<Slab> = Vec::new();

    for (_, members) in &components {
        if members.len() <= 1 {
            continue;
        }
        // Find atom range and elem_bytes for this component.
        let mut lo = u64::MAX;
        let mut hi = 0u64;
        let mut eb = 4usize;
        for &idx in members {
            let (atom_base, count, dtype) = if idx < num_inputs {
                let it = &graph.input_tensors()[idx];
                (it.base_id.0, it.count, it.dtype)
            } else {
                let g = &groups[idx - num_inputs];
                (g.base_id.0, g.count, g.output_dtype)
            };
            lo = lo.min(atom_base);
            hi = hi.max(atom_base + count);
            eb = eb.max(dtype_elem_bytes(dtype));
        }
        let slab_idx = slabs.len();
        slabs.push(Slab {
            atom_lo: lo,
            atom_hi: hi,
            elem_bytes: eb,
            byte_offset: 0,
        });
        for &idx in members {
            let atom_base = if idx < num_inputs {
                graph.input_tensors()[idx].base_id.0
            } else {
                groups[idx - num_inputs].base_id.0
            };
            let off_in_slab = (atom_base - lo) as usize * eb;
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
            } = &consumer.inputs[0]
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
                let bases: &[AtomId] = match inp {
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
            let size = (slab.atom_hi - slab.atom_lo) as usize * slab.elem_bytes;
            let offset = allocator.alloc(size, slab.elem_bytes);
            slabs[slab_idx].byte_offset = offset;
            slab_allocated[slab_idx] = true;
        }
        slabs[slab_idx].byte_offset
    }

    // Input tensor slots.
    for (ii, it) in graph.input_tensors().iter().enumerate() {
        let elem_bytes = dtype_elem_bytes(it.dtype);
        let elem_bits_semantic = it.dtype.total_bits() as u64;
        if let Some((slab_idx, off_in_slab)) = slab_assignment[ii] {
            let slab_base = ensure_slab(&mut slabs, &mut slab_allocated, &mut allocator, slab_idx);
            all_slots.push(SlotInfo {
                atom_base: it.base_id,
                count: it.count,
                buffer_id: 0,
                bit_offset: ((slab_base + off_in_slab) as u64) * 8,
                bit_stride: (slabs[slab_idx].elem_bytes as u64) * 8,
                elem_bits: elem_bits_semantic,
                dtype: it.dtype,
            });
        } else {
            let size = it.count as usize * elem_bytes;
            let offset = allocator.alloc(size, elem_bytes);
            all_slots.push(SlotInfo {
                atom_base: it.base_id,
                count: it.count,
                buffer_id: 0,
                bit_offset: (offset as u64) * 8,
                bit_stride: (elem_bytes as u64) * 8,
                elem_bits: elem_bits_semantic,
                dtype: it.dtype,
            });
        }
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

        if let Some((slab_idx, off_in_slab)) = slab_assignment[item_idx] {
            let slab_base = ensure_slab(&mut slabs, &mut slab_allocated, &mut allocator, slab_idx);
            all_slots.push(SlotInfo {
                atom_base: group.base_id,
                count: group.count,
                buffer_id: 0,
                bit_offset: ((slab_base + off_in_slab) as u64) * 8,
                bit_stride: (slabs[slab_idx].elem_bytes as u64) * 8,
                elem_bits: elem_bits_semantic,
                dtype: group.output_dtype,
            });
        } else {
            let size = group.count as usize * elem_bytes;
            let offset = allocator.alloc(size, elem_bytes);
            all_slots.push(SlotInfo {
                atom_base: group.base_id,
                count: group.count,
                buffer_id: 0,
                bit_offset: (offset as u64) * 8,
                bit_stride: (elem_bytes as u64) * 8,
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

    BufferLayout {
        total_bytes: allocator.watermark,
        slots: all_slots,
        group_use_counts: use_counts,
        inlinable,
        inlines_producer,
    }
}
pub(crate) fn write_store_slice_to_buffer(
    slice: &StoreSlice<'_>,
    layout: &BufferLayout,
    buffer: &mut [u8],
) {
    let mut written: u64 = 0;
    let mut atom = slice.base.0;
    let total = slice.count;

    while written < total {
        let Some((slot, elem_start)) = layout.find(AtomId(atom)) else {
            written += 1;
            atom += 1;
            continue;
        };
        let available = slot.count - elem_start;
        let to_write = available.min(total - written);

        // Fast path: byte-natural source AND byte-aligned destination AND
        // matching dtype → direct memcpy. Phase 1 destinations are always
        // byte-aligned and most sources are byte-natural, so this fires
        // for the conventional case.
        let elem_bytes = slice.dtype.bytes_per_element();
        if slot.dtype == slice.dtype && slice.is_byte_natural() && slot.is_byte_aligned() {
            let src_byte_start = slice.src_byte_offset() + (written as usize) * elem_bytes;
            let src_byte_end = src_byte_start + (to_write as usize) * elem_bytes;
            let dst_byte_start = slot.byte_offset() + (elem_start as usize) * slot.elem_bytes();
            let dst_byte_end = dst_byte_start + (to_write as usize) * slot.elem_bytes();
            if src_byte_end <= slice.data.len() && dst_byte_end <= buffer.len() {
                buffer[dst_byte_start..dst_byte_end]
                    .copy_from_slice(&slice.data[src_byte_start..src_byte_end]);
            }
        } else {
            // General path: per-element via NumericScalarView. Handles
            // bit-strided sources (sub-byte dtype, non-zero offset_bits)
            // and dtype conversion.
            for i in 0..to_write {
                let src_bit = slice.src_bit_offset + (written + i) * slice.src_bit_stride;
                let src_view = NumericScalarView {
                    data: slice.data,
                    bit_offset: src_bit as usize,
                    dtype: slice.dtype,
                };
                let scalar = src_view.to_owned_scalar();
                let converted = if slot.dtype == slice.dtype {
                    scalar
                } else {
                    scalar.cast_to(slot.dtype)
                };

                let dst_bit = slot.bit_offset + (elem_start + i) * slot.bit_stride;
                if slot.is_byte_aligned() {
                    let dst_off = (dst_bit / 8) as usize;
                    if dst_off + slot.elem_bytes() <= buffer.len() {
                        write_scalar(buffer, dst_off, &converted);
                    }
                } else {
                    // Bit-aligned destination (phase 6); use the bit-aware
                    // write path. Caller bears the responsibility of
                    // ensuring the buffer is large enough.
                    let mut view = NumericScalarViewMut {
                        data: buffer,
                        bit_offset: dst_bit as usize,
                        dtype: slot.dtype,
                    };
                    view.write_scalar(&converted);
                }
            }
        }

        written += to_write;
        atom += to_write;
    }
}

/// Read output range from buffer into a SpanOutput.
pub(crate) fn read_buffer_to_output(
    range: &AtomRange,
    layout: &BufferLayout,
    buffer: &[u8],
    out: &mut SpanOutput<'_>,
) {
    let elem_bytes = range.dtype.bytes_per_element();
    let mut read: u64 = 0;
    let mut atom = range.base.0;
    let total = range.count;

    while read < total {
        let Some((slot, elem_start)) = layout.find(AtomId(atom)) else {
            // Gap: write zeros.
            let dst_off = (read as usize) * elem_bytes;
            if dst_off + elem_bytes <= out.data.len() {
                for b in &mut out.data[dst_off..dst_off + elem_bytes] {
                    *b = 0;
                }
            }
            read += 1;
            atom += 1;
            continue;
        };

        let available = slot.count - elem_start;
        let to_read = available.min(total - read);

        // Fast path: matching dtype AND byte-aligned slot AND
        // byte-natural slot stride → direct memcpy. SpanOutput is always
        // byte-natural by construction.
        if slot.dtype == range.dtype && slot.is_byte_aligned() {
            let src_byte_start = slot.byte_offset() + (elem_start as usize) * slot.elem_bytes();
            let src_byte_end = src_byte_start + (to_read as usize) * slot.elem_bytes();
            let dst_byte_start = (read as usize) * elem_bytes;
            let dst_byte_end = dst_byte_start + (to_read as usize) * elem_bytes;
            if src_byte_end <= buffer.len() && dst_byte_end <= out.data.len() {
                out.data[dst_byte_start..dst_byte_end]
                    .copy_from_slice(&buffer[src_byte_start..src_byte_end]);
            }
        } else {
            // General path: per-element via NumericScalarView.
            for i in 0..to_read {
                let src_bit = slot.bit_offset + (elem_start + i) * slot.bit_stride;
                let src_view = NumericScalarView {
                    data: buffer,
                    bit_offset: src_bit as usize,
                    dtype: slot.dtype,
                };
                let scalar = src_view.to_owned_scalar();
                let converted = if slot.dtype == range.dtype {
                    scalar
                } else {
                    scalar.cast_to(range.dtype)
                };
                let dst_off = ((read + i) as usize) * elem_bytes;
                if dst_off + elem_bytes <= out.data.len() {
                    write_scalar(&mut out.data[..], dst_off, &converted);
                }
            }
        }

        read += to_read;
        atom += to_read;
    }
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
            match ir {
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
                    } = ir
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
        ScalarOp::Identity | ScalarOp::Cast { .. } => "Id",
        ScalarOp::Binary { .. } => "Bin",
        ScalarOp::Unary { .. } => "Un",
        ScalarOp::Select => "Sel",
        ScalarOp::Reduce { .. } => "Red",
        ScalarOp::IndirectLoad { .. } => "Ind",
        ScalarOp::OpaqueOutput { .. } => "Opq",
        ScalarOp::LiteralSpan(_) => "LitS",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GlobalId;
    use crate::nano_graph::ops::ScalarOp;
    use crate::numeric_scalar::NumericScalar;

    // ─── SlotInfo helper invariants ─────────────────────────────────────────

    #[test]
    fn slot_info_helpers_byte_aligned_native_dtypes() {
        // Phase 1 invariant: every slot compute_layout produces is
        // byte-aligned and the helpers return the byte view directly.
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
        // Bool: elem_bits=1 (semantic) but bit_stride=8 (phase 1 byte-padded).
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
        // Phase 6 will allow bit-packed slots: bit_stride == elem_bits.
        // is_byte_aligned() returns false because the stride isn't a
        // multiple of 8. Phase 1 doesn't produce these but the helper
        // shape is forward-compatible.
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
        // A graph with a Bool input + an Identity group consuming it.
        // Phase 1 produces byte-padded Bool slots:
        //   bit_stride = 8, elem_bits = 1.
        let mut g = NanoGraph::new();
        let bool_input = g.add_input_tensor(GlobalId(0), 5, NumericDType::BOOL);
        let identity = g.push_group(
            5,
            NumericDType::BOOL,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::affine(bool_input, 1)],
        );
        let outputs = vec![AtomRange {
            base: identity,
            count: 5,
            dtype: NumericDType::BOOL,
        }];
        let layout = compute_layout(&g, &outputs, true);

        let (input_slot, _) = layout.find(bool_input).expect("bool input slot present");
        assert_eq!(input_slot.dtype, NumericDType::BOOL);
        assert_eq!(input_slot.elem_bits, 1, "Bool semantic width is 1 bit");
        assert_eq!(input_slot.bit_stride, 8, "phase 1 pads sub-byte to a byte");
        assert!(input_slot.is_byte_aligned());

        let (out_slot, _) = layout.find(identity).expect("identity slot present");
        assert_eq!(out_slot.dtype, NumericDType::BOOL);
        assert_eq!(out_slot.elem_bits, 1);
        assert_eq!(out_slot.bit_stride, 8);
    }

    // ─── populate_literals for a sub-byte dtype ─────────────────────────────

    #[test]
    fn populate_literals_bool_writes_byte_padded() {
        // A graph with a Bool literal scalar — populate_literals should
        // write a single 0x01 byte at the slot's byte offset.
        let mut g = NanoGraph::new();
        let lit = g.push_group(
            1,
            NumericDType::BOOL,
            ScalarOp::Literal(NumericScalar::from_bool(true)),
            vec![],
            vec![],
        );
        let outputs = vec![AtomRange {
            base: lit,
            count: 1,
            dtype: NumericDType::BOOL,
        }];
        let layout = compute_layout(&g, &outputs, true);

        let mut buffer = vec![0u8; layout.total_bytes];
        layout.populate_literals(&g, &mut buffer);

        let (slot, _) = layout.find(lit).expect("literal slot present");
        assert_eq!(slot.elem_bytes(), 1);
        let byte = buffer[slot.byte_offset()];
        assert_eq!(byte, 1, "Bool true literal stored as 0x01");

        // Round-trip via read_buffer_to_output: extract back into a
        // SpanOutput and confirm we read 1 byte = 1 (true).
        let mut out_data = vec![0u8; 1];
        let mut out = SpanOutput {
            data: &mut out_data,
            dtype: NumericDType::BOOL,
            count: 1,
        };
        read_buffer_to_output(&outputs[0], &layout, &buffer, &mut out);
        assert_eq!(out_data[0], 1);
    }

    // ─── Bit-strided source round-trip via marshalling ──────────────────────

    /// Drive a `StoreSlice` whose `data` is a bit-packed source buffer
    /// (1-bit Bool elements at consecutive bit positions, no byte padding)
    /// through `write_store_slice_to_buffer`, then read it back via
    /// `read_buffer_to_output`. The destination slot is byte-padded as
    /// always in phase 1, so the round-trip exercises the bit-aware read
    /// path on the source side.
    #[test]
    fn bit_strided_source_round_trip_bool_packed() {
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), 9, NumericDType::BOOL);
        // Identity group so the input has a slot in the layout.
        let out = g.push_group(
            9,
            NumericDType::BOOL,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::affine(inp, 1)],
        );
        let outputs = vec![AtomRange {
            base: out,
            count: 9,
            dtype: NumericDType::BOOL,
        }];
        let layout = compute_layout(&g, &outputs, true);

        // Source: 9 Bool values packed at 1 bit per element starting at
        // bit offset 3 (so element 0 lives at bit position 3 within byte 0).
        // Pattern: 1, 0, 1, 0, 1, 1, 0, 0, 1
        let src_pattern: [bool; 9] = [true, false, true, false, true, true, false, false, true];
        // Encode the pattern starting at bit_offset=3 in a 3-byte buffer.
        let mut packed = vec![0u8; 3];
        for (i, &bit) in src_pattern.iter().enumerate() {
            if bit {
                let bit_pos = 3 + i;
                let byte = bit_pos / 8;
                let in_byte = bit_pos % 8;
                packed[byte] |= 1 << in_byte;
            }
        }

        let slice = StoreSlice {
            base: inp,
            data: &packed,
            dtype: NumericDType::BOOL,
            count: 9,
            src_bit_offset: 3,
            src_bit_stride: 1, // bit-packed, no byte padding
        };
        assert!(
            !slice.is_byte_natural(),
            "bit-packed source must take the slow path"
        );

        let mut buffer = vec![0u8; layout.total_bytes];
        write_store_slice_to_buffer(&slice, &layout, &mut buffer);

        // Now extract the input slot's bytes — phase 1 pads Bool to one
        // byte per element, so we should see 0x00 / 0x01 per element in
        // the layout's byte-padded format.
        let (input_slot, _) = layout.find(inp).expect("bool input slot");
        for (i, &expected) in src_pattern.iter().enumerate() {
            let off = input_slot.byte_offset() + i * input_slot.elem_bytes();
            let got = buffer[off];
            assert_eq!(
                got, expected as u8,
                "element {i}: byte-padded bool expected {expected}, got 0x{got:02x}"
            );
        }

        // Round-trip out via the identity group's output range. The
        // identity copy is a no-op layout-wise (input == output) so the
        // output bytes should match the input bytes.
        //
        // We can't actually run the JIT here (no compiler in this test),
        // but we can directly read the input slot via read_buffer_to_output
        // pointed at the input range. That exercises the byte-fast path
        // on the read side and confirms the round-trip is bit-equal.
        let input_range = AtomRange {
            base: inp,
            count: 9,
            dtype: NumericDType::BOOL,
        };
        let mut out_data = vec![0u8; 9];
        let mut out = SpanOutput {
            data: &mut out_data,
            dtype: NumericDType::BOOL,
            count: 9,
        };
        read_buffer_to_output(&input_range, &layout, &buffer, &mut out);
        for (i, &expected) in src_pattern.iter().enumerate() {
            assert_eq!(out_data[i], expected as u8, "round-trip element {i}");
        }
    }

    /// Bit-strided source with a non-zero `src_bit_offset` and a
    /// byte-natural stride (= 8 for Bool). This is the "offset_bits != 0"
    /// case from `TensorLayout::ElementStrided`. The source byte buffer
    /// has 3 unrelated bytes of padding before element 0.
    #[test]
    fn bit_strided_source_offset_bits_byte_stride() {
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), 4, NumericDType::U8);
        let _ident = g.push_group(
            4,
            NumericDType::U8,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::affine(inp, 1)],
        );
        let outputs = vec![AtomRange {
            base: inp,
            count: 4,
            dtype: NumericDType::U8,
        }];
        let layout = compute_layout(&g, &outputs, true);

        // Source data: 3 bytes of padding (0xff each), then the actual U8
        // values [10, 20, 30, 40]. src_bit_offset = 24 (3 bytes).
        let src_data: Vec<u8> = vec![0xff, 0xff, 0xff, 10, 20, 30, 40];
        let slice = StoreSlice {
            base: inp,
            data: &src_data,
            dtype: NumericDType::U8,
            count: 4,
            src_bit_offset: 24, // byte-aligned but non-zero
            src_bit_stride: 8,  // U8 byte-natural
        };
        // Both fields are byte-multiples, so the source IS byte-natural and
        // the fast path memcpy handles the offset via `src_byte_offset()`.
        // This test exercises the offset arithmetic on the fast path.
        assert!(slice.is_byte_natural());
        assert_eq!(slice.src_byte_offset(), 3);

        let mut buffer = vec![0u8; layout.total_bytes];
        write_store_slice_to_buffer(&slice, &layout, &mut buffer);

        let (input_slot, _) = layout.find(inp).expect("u8 input slot");
        for (i, &expected) in [10u8, 20, 30, 40].iter().enumerate() {
            let off = input_slot.byte_offset() + i * input_slot.elem_bytes();
            assert_eq!(buffer[off], expected, "element {i}");
        }
    }

    /// Symmetric case: byte-natural source (the conventional executor
    /// gather output) takes the memcpy fast path. This is what every
    /// existing test exercises and serves as a baseline.
    #[test]
    fn byte_natural_source_round_trip_f32() {
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), 4, NumericDType::F32);
        let _ident = g.push_group(
            4,
            NumericDType::F32,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::affine(inp, 1)],
        );
        let outputs = vec![AtomRange {
            base: inp,
            count: 4,
            dtype: NumericDType::F32,
        }];
        let layout = compute_layout(&g, &outputs, true);

        let values = [1.5f32, -2.5, 3.0, 4.25];
        let bytes: Vec<u8> = values.iter().flat_map(|f| f.to_le_bytes()).collect();
        let slice = StoreSlice {
            base: inp,
            data: &bytes,
            dtype: NumericDType::F32,
            count: 4,
            src_bit_offset: 0,
            src_bit_stride: 32,
        };
        assert!(slice.is_byte_natural());

        let mut buffer = vec![0u8; layout.total_bytes];
        write_store_slice_to_buffer(&slice, &layout, &mut buffer);

        let mut out_data = vec![0u8; 16];
        let mut out = SpanOutput {
            data: &mut out_data,
            dtype: NumericDType::F32,
            count: 4,
        };
        read_buffer_to_output(&outputs[0], &layout, &buffer, &mut out);
        let recovered: Vec<f32> = out_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(recovered, values);
    }
}
