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
    /// Bit offset of the first element from the start of the buffer.
    /// Always a multiple of 8 in phase 1 (byte-aligned slot starts).
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
    let inline_disabled = std::env::var("INLINE").as_deref() == Ok("0");
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
    let elem_bytes = slice.dtype.bytes_per_element();
    let mut written = 0usize;
    let mut atom = slice.base.0;
    let total = slice.count as usize;

    while written < total {
        let Some((slot, elem_start)) = layout.find(AtomId(atom)) else {
            written += 1;
            atom += 1;
            continue;
        };
        let available = (slot.count - elem_start) as usize;
        let to_write = available.min(total - written);

        let src_start = written * elem_bytes;
        let src_end = src_start + to_write * elem_bytes;

        if slot.dtype == slice.dtype && src_end <= slice.data.len() {
            // Fast path: dtypes match, direct memcpy.
            let dst_start = slot.byte_offset() + elem_start as usize * slot.elem_bytes();
            let dst_end = dst_start + to_write * slot.elem_bytes();
            if dst_end <= buffer.len() {
                buffer[dst_start..dst_end].copy_from_slice(&slice.data[src_start..src_end]);
            }
        } else if src_end <= slice.data.len() {
            // Slow path: per-element with dtype conversion.
            for i in 0..to_write {
                let src_off = (written + i) * elem_bytes;
                let dst_off = slot.byte_offset() + (elem_start as usize + i) * slot.elem_bytes();
                if src_off + elem_bytes <= slice.data.len()
                    && dst_off + slot.elem_bytes() <= buffer.len()
                {
                    let scalar = read_scalar_raw(&slice.data[src_off..], slice.dtype);
                    let converted = scalar.cast_to(slot.dtype);
                    write_scalar(buffer, dst_off, &converted);
                }
            }
        }

        written += to_write;
        atom += to_write as u64;
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
    let mut read = 0usize;
    let mut atom = range.base.0;
    let total = range.count as usize;

    while read < total {
        let Some((slot, elem_start)) = layout.find(AtomId(atom)) else {
            // Gap: write zeros.
            let dst_off = read * elem_bytes;
            if dst_off + elem_bytes <= out.data.len() {
                for b in &mut out.data[dst_off..dst_off + elem_bytes] {
                    *b = 0;
                }
            }
            read += 1;
            atom += 1;
            continue;
        };

        let available = (slot.count - elem_start) as usize;
        let to_read = available.min(total - read);

        if slot.dtype == range.dtype {
            // Fast path: direct memcpy.
            let src_start = slot.byte_offset() + elem_start as usize * slot.elem_bytes();
            let src_end = src_start + to_read * slot.elem_bytes();
            let dst_start = read * elem_bytes;
            let dst_end = dst_start + to_read * elem_bytes;
            if src_end <= buffer.len() && dst_end <= out.data.len() {
                out.data[dst_start..dst_end].copy_from_slice(&buffer[src_start..src_end]);
            }
        } else {
            // Slow path: per-element dtype conversion.
            for i in 0..to_read {
                let src_off = slot.byte_offset() + (elem_start as usize + i) * slot.elem_bytes();
                let dst_off = (read + i) * elem_bytes;
                if src_off + slot.elem_bytes() <= buffer.len()
                    && dst_off + elem_bytes <= out.data.len()
                {
                    let scalar = read_scalar(buffer, src_off, slot.dtype);
                    let converted = scalar.cast_to(range.dtype);
                    write_scalar(&mut out.data[..], dst_off, &converted);
                }
            }
        }

        read += to_read;
        atom += to_read as u64;
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
