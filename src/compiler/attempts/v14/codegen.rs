#![allow(clippy::all, dead_code, unreachable_patterns, unused)]
//! Cranelift JIT codegen with multi-dtype buffer scheduling.
//!
//! Compiles a span's NanoGraph into a native function operating on a typed
//! byte buffer. Memory is scheduled with liveness-aware slot reuse: when a
//! group's output is consumed by all its downstream dependents, its buffer
//! space is reclaimed and may be assigned to later groups.
//!
//! # Function signature
//!
//! `fn(buffer: *mut u8) -> ()`
//!
//! # Buffer model
//!
//! Working memory is a flat byte buffer. Each group's output and each external
//! input occupies a typed `SlotInfo` (byte offset + element count + dtype).
//! Elements are stored at their native width (8 bytes for I64, 4 for F32,
//! 2 for BF16, 1 for BOOL/U8).
//!
//! # Type discipline
//!
//! Values are loaded in their native storage type, cast to the op's
//! `compute_dtype` for arithmetic, then cast to `output_dtype` for storage.
//! Wider compute types are acceptable (e.g. BF16 computed via f32+truncation)
//! but the output must be representative of the specified dtype.
//!
//! Two Cranelift "representation kinds" carry values through the IR:
//! - `types::F32` for float dtypes (F32, BF16, F16, F64)
//! - `types::I64` for integer dtypes (I64, I32, BOOL, U8, I8)

use std::collections::{HashMap, HashSet};

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, Value};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use crate::dtype::DType;
use crate::nano_graph::{
    AtomGroup, AtomId, AtomRange, InputRef, NanoGraph, ReduceKind, ScalarBinOp, ScalarOp,
    ScalarUnaryOp,
};

// ─── Buffer layout types ────────────────────────────────────────────────────

/// A typed region in the working buffer.
#[derive(Debug, Clone)]
pub struct SlotInfo {
    /// First AtomId mapped to this slot.
    pub atom_base: AtomId,
    /// Number of elements.
    pub count: u64,
    /// Byte offset in the buffer.
    pub byte_offset: usize,
    /// Bytes per element (derived from dtype).
    pub elem_bytes: usize,
    /// Storage dtype.
    pub dtype: DType,
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
}

impl BufferLayout {
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
            .map(|(slot, idx)| slot.byte_offset + idx as usize * slot.elem_bytes)
    }

    /// Write literal group values into the buffer.
    pub fn populate_literals(&self, graph: &NanoGraph, buffer: &mut [u8]) {
        use crate::numeric_scalar::NumericScalar;
        for group in graph.groups() {
            if let ScalarOp::Literal(scalar) = &group.op {
                if let Some((slot, _)) = self.find(group.base_id) {
                    // Cast the literal to the slot's storage dtype, then write raw bytes.
                    let stored = scalar.cast_to(slot.dtype);
                    for i in 0..group.count {
                        let off = slot.byte_offset + i as usize * slot.elem_bytes;
                        write_scalar(buffer, off, &stored);
                    }
                }
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
                    let off = slot.byte_offset + (elem_start as usize + i) * slot.elem_bytes;
                    if off + slot.elem_bytes <= buffer.len() {
                        let scalar = NumericScalar::F32(data[written + i]).cast_to(slot.dtype);
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
                    let off = slot.byte_offset + (elem_start + i) as usize * slot.elem_bytes;
                    if off + slot.elem_bytes <= buffer.len() {
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
    use crate::numeric_scalar::NumericScalar;
    match val {
        NumericScalar::F32(v) => buffer[off..off + 4].copy_from_slice(&v.to_le_bytes()),
        NumericScalar::F64(v) => buffer[off..off + 8].copy_from_slice(&v.to_le_bytes()),
        NumericScalar::BF16(v) => buffer[off..off + 2].copy_from_slice(&v.to_bits().to_le_bytes()),
        NumericScalar::F16(v) => buffer[off..off + 2].copy_from_slice(&v.to_bits().to_le_bytes()),
        NumericScalar::I64(v) => buffer[off..off + 8].copy_from_slice(&v.to_le_bytes()),
        NumericScalar::U64(v) => buffer[off..off + 8].copy_from_slice(&v.to_le_bytes()),
        NumericScalar::I32(v) => buffer[off..off + 4].copy_from_slice(&v.to_le_bytes()),
        NumericScalar::U32(v) => buffer[off..off + 4].copy_from_slice(&v.to_le_bytes()),
        NumericScalar::I16(v) => buffer[off..off + 2].copy_from_slice(&v.to_le_bytes()),
        NumericScalar::U16(v) => buffer[off..off + 2].copy_from_slice(&v.to_le_bytes()),
        NumericScalar::I8(v) => buffer[off] = *v as u8,
        NumericScalar::U8(v) => buffer[off] = *v,
        NumericScalar::BOOL(v) => buffer[off] = if *v { 1 } else { 0 },
        _ => {
            // Fallback: cast to F32.
            let f = val.to_f64() as f32;
            buffer[off..off + 4].copy_from_slice(&f.to_le_bytes());
        }
    }
}

/// Read a NumericScalar from `buffer[off..]` in the given storage dtype.
fn read_scalar(buffer: &[u8], off: usize, dtype: DType) -> crate::numeric_scalar::NumericScalar {
    use crate::numeric_scalar::NumericScalar;
    match dtype {
        DType::F32 => NumericScalar::F32(f32::from_le_bytes(buffer[off..off + 4].try_into().unwrap())),
        DType::F64 => NumericScalar::F64(f64::from_le_bytes(buffer[off..off + 8].try_into().unwrap())),
        DType::BF16 => {
            let bits = u16::from_le_bytes(buffer[off..off + 2].try_into().unwrap());
            NumericScalar::BF16(half::bf16::from_bits(bits))
        }
        DType::F16 => {
            let bits = u16::from_le_bytes(buffer[off..off + 2].try_into().unwrap());
            NumericScalar::F16(half::f16::from_bits(bits))
        }
        DType::I64 => NumericScalar::I64(i64::from_le_bytes(buffer[off..off + 8].try_into().unwrap())),
        DType::U64 => NumericScalar::U64(u64::from_le_bytes(buffer[off..off + 8].try_into().unwrap())),
        DType::I32 => NumericScalar::I32(i32::from_le_bytes(buffer[off..off + 4].try_into().unwrap())),
        DType::U32 => NumericScalar::U32(u32::from_le_bytes(buffer[off..off + 4].try_into().unwrap())),
        DType::I16 => NumericScalar::I16(i16::from_le_bytes(buffer[off..off + 2].try_into().unwrap())),
        DType::U16 => NumericScalar::U16(u16::from_le_bytes(buffer[off..off + 2].try_into().unwrap())),
        DType::I8 => NumericScalar::I8(buffer[off] as i8),
        DType::U8 => NumericScalar::U8(buffer[off]),
        DType::BOOL => NumericScalar::BOOL(buffer[off] != 0),
        _ => NumericScalar::F32(f32::from_le_bytes(buffer[off..off + 4].try_into().unwrap())),
    }
}

/// Legacy: read an f32 from buffer (used by diagnostics only).
fn read_f32_from(buffer: &[u8], off: usize, dtype: DType) -> f32 {
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

fn dtype_elem_bytes(dtype: DType) -> usize {
    dtype.size().unwrap_or(4)
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
pub fn compute_layout(graph: &NanoGraph, output_ranges: &[AtomRange]) -> BufferLayout {
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
        atom_items.push((group.base_id.0, group.base_id.0 + group.count, num_inputs + gi));
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

    // Compute accessed atom range for an InputRef applied to a group.
    let input_ref_range = |ir: &InputRef, count: u64, atom_offset: u64| -> Option<(u64, u64)> {
        match ir {
            InputRef::Broadcast(_) | InputRef::Explicit(_) => None, // no stride, no constraint
            InputRef::Affine { base, stride } => {
                let a = base.0 as i64 + *stride * atom_offset as i64;
                let b = base.0 as i64 + *stride * (atom_offset + count - 1) as i64;
                Some((a.min(b) as u64, a.max(b) as u64 + 1))
            }
            InputRef::StridedBroadcast { base, stride, repeat } => {
                let max_block = (atom_offset + count - 1) / repeat;
                let a = base.0 as i64;
                let b = base.0 as i64 + *stride * max_block as i64;
                Some((a.min(b) as u64, a.max(b) as u64 + 1))
            }
            InputRef::Modular { base, stride, modulus } => {
                let a = base.0 as i64;
                let b = base.0 as i64 + *stride as i64 * (*modulus as i64 - 1);
                Some((a.min(b) as u64, a.max(b) as u64 + 1))
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
                if let Some(InputRef::Affine { base, stride }) = group.inputs.first() {
                    let first_i = base.0 as i64 + *stride * group.atom_offset as i64;
                    let last_i =
                        base.0 as i64 + *stride * (group.atom_offset + group.count - 1) as i64;
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

    // ── Step 3: Compute liveness ──

    let mut use_counts = vec![0u32; n];
    for (gi, group) in groups.iter().enumerate() {
        let mut producers = HashSet::new();
        graph.collect_all_producer_indices(group, gi, &mut producers);
        for pi in producers {
            use_counts[pi] += 1;
        }
    }

    // Pin output groups.
    let output_atoms: HashSet<u64> = output_ranges
        .iter()
        .flat_map(|r| r.base.0..r.base.0 + r.count)
        .collect();
    for (gi, group) in groups.iter().enumerate() {
        let group_end = group.base_id.0 + group.count;
        if (group.base_id.0..group_end).any(|a| output_atoms.contains(&a)) {
            use_counts[gi] += 1;
        }
    }

    // ── Step 4: Allocate slots ──

    let trace_byte = std::env::var("TRACE_BYTE").ok().and_then(|s| s.parse::<usize>().ok());

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
        if let Some((slab_idx, off_in_slab)) = slab_assignment[ii] {
            let slab_base = ensure_slab(&mut slabs, &mut slab_allocated, &mut allocator, slab_idx);
            all_slots.push(SlotInfo {
                atom_base: it.base_id,
                count: it.count,
                byte_offset: slab_base + off_in_slab,
                elem_bytes: slabs[slab_idx].elem_bytes,
                dtype: it.dtype,
            });
        } else {
            let size = it.count as usize * elem_bytes;
            let offset = allocator.alloc(size, elem_bytes);
            all_slots.push(SlotInfo {
                atom_base: it.base_id,
                count: it.count,
                byte_offset: offset,
                elem_bytes,
                dtype: it.dtype,
            });
        }
    }

    // Group slots.
    let mut group_slot_indices = vec![0usize; n];
    let mut remaining = use_counts.clone();

    for (gi, group) in groups.iter().enumerate() {
        let elem_bytes = dtype_elem_bytes(group.output_dtype);
        let item_idx = num_inputs + gi;
        let slot_idx = all_slots.len();

        if let Some((slab_idx, off_in_slab)) = slab_assignment[item_idx] {
            let slab_base = ensure_slab(&mut slabs, &mut slab_allocated, &mut allocator, slab_idx);
            all_slots.push(SlotInfo {
                atom_base: group.base_id,
                count: group.count,
                byte_offset: slab_base + off_in_slab,
                elem_bytes: slabs[slab_idx].elem_bytes,
                dtype: group.output_dtype,
            });
        } else {
            let size = group.count as usize * elem_bytes;
            let offset = allocator.alloc(size, elem_bytes);
            all_slots.push(SlotInfo {
                atom_base: group.base_id,
                count: group.count,
                byte_offset: offset,
                elem_bytes,
                dtype: group.output_dtype,
            });
        }

        group_slot_indices[gi] = slot_idx;

        if let Some(tb) = trace_byte {
            let s = &all_slots[slot_idx];
            let end = s.byte_offset + s.count as usize * s.elem_bytes;
            if s.byte_offset <= tb && end > tb {
                eprintln!(
                    "  ALLOC group {} base={} at [{}-{}) eb={} dtype={:?} op={:?}",
                    gi, group.base_id, s.byte_offset, end, s.elem_bytes, s.dtype,
                    op_name_short(&group.op)
                );
            }
        }

        // Slot reuse disabled: we were getting overlapping allocations between
        // slab members and FreeList allocations. TODO: fix the allocator to
        // properly handle mixed-dtype slabs before re-enabling.
        //
        // The old code freed producer slots here, but with mixed I64/F32 slabs
        // the freed regions can overlap with slab-allocated literal slots,
        // causing the JIT to corrupt pre-populated literal values.
    }

    all_slots.sort_by_key(|s| s.atom_base.0);

    BufferLayout {
        total_bytes: allocator.watermark,
        slots: all_slots,
        group_use_counts: use_counts,
    }
}

// ─── Math function trampolines ──────────────────────────────────────────────
//
// Cranelift doesn't have intrinsics for transcendental functions. We
// declare extern "C" wrappers and link them via JITBuilder::symbol().

extern "C" fn jit_expf(x: f32) -> f32 {
    x.exp()
}
extern "C" fn jit_logf(x: f32) -> f32 {
    x.ln()
}
extern "C" fn jit_tanhf(x: f32) -> f32 {
    x.tanh()
}
extern "C" fn jit_sqrtf(x: f32) -> f32 {
    x.sqrt()
}
extern "C" fn jit_floorf(x: f32) -> f32 {
    x.floor()
}
extern "C" fn jit_ceilf(x: f32) -> f32 {
    x.ceil()
}
extern "C" fn jit_fabsf(x: f32) -> f32 {
    x.abs()
}
extern "C" fn jit_powf(x: f32, y: f32) -> f32 {
    x.powf(y)
}
extern "C" fn jit_fmodf(x: f32, y: f32) -> f32 {
    x % y
}

struct MathFuncs {
    expf: cranelift_module::FuncId,
    logf: cranelift_module::FuncId,
    tanhf: cranelift_module::FuncId,
    sqrtf: cranelift_module::FuncId,
    floorf: cranelift_module::FuncId,
    ceilf: cranelift_module::FuncId,
    fabsf: cranelift_module::FuncId,
    powf: cranelift_module::FuncId,
    fmodf: cranelift_module::FuncId,
}

fn register_math_symbols(jit_builder: &mut JITBuilder) {
    jit_builder.symbol("jit_expf", jit_expf as *const u8);
    jit_builder.symbol("jit_logf", jit_logf as *const u8);
    jit_builder.symbol("jit_tanhf", jit_tanhf as *const u8);
    jit_builder.symbol("jit_sqrtf", jit_sqrtf as *const u8);
    jit_builder.symbol("jit_floorf", jit_floorf as *const u8);
    jit_builder.symbol("jit_ceilf", jit_ceilf as *const u8);
    jit_builder.symbol("jit_fabsf", jit_fabsf as *const u8);
    jit_builder.symbol("jit_powf", jit_powf as *const u8);
    jit_builder.symbol("jit_fmodf", jit_fmodf as *const u8);
}

fn declare_math_funcs(module: &mut JITModule) -> Result<MathFuncs, String> {
    let mut sig1 = module.make_signature();
    sig1.params.push(AbiParam::new(types::F32));
    sig1.returns.push(AbiParam::new(types::F32));

    let mut sig2 = module.make_signature();
    sig2.params.push(AbiParam::new(types::F32));
    sig2.params.push(AbiParam::new(types::F32));
    sig2.returns.push(AbiParam::new(types::F32));

    let decl = |m: &mut JITModule, name: &str, sig: &cranelift_codegen::ir::Signature| {
        m.declare_function(name, Linkage::Import, sig)
            .map_err(|e| format!("declare {}: {}", name, e))
    };

    Ok(MathFuncs {
        expf: decl(module, "jit_expf", &sig1)?,
        logf: decl(module, "jit_logf", &sig1)?,
        tanhf: decl(module, "jit_tanhf", &sig1)?,
        sqrtf: decl(module, "jit_sqrtf", &sig1)?,
        floorf: decl(module, "jit_floorf", &sig1)?,
        ceilf: decl(module, "jit_ceilf", &sig1)?,
        fabsf: decl(module, "jit_fabsf", &sig1)?,
        powf: decl(module, "jit_powf", &sig2)?,
        fmodf: decl(module, "jit_fmodf", &sig2)?,
    })
}

// ─── Variable counter ───────────────────────────────────────────────────────

struct VarCounter(u32);

impl VarCounter {
    fn new() -> Self {
        VarCounter(0)
    }
    fn next(&mut self) -> Variable {
        let v = Variable::from_u32(self.0);
        self.0 += 1;
        v
    }
}

// ─── Compiled span ──────────────────────────────────────────────────────────

/// A JIT-compiled span function.
pub struct CompiledSpan {
    func_ptr: *const u8,
    _module: JITModule,
}

unsafe impl Send for CompiledSpan {}
unsafe impl Sync for CompiledSpan {}

impl CompiledSpan {
    /// Run the compiled function on a buffer.
    ///
    /// # Safety
    ///
    /// Buffer must be at least `layout.total_bytes` bytes and properly
    /// initialized (literals and inputs populated).
    pub fn execute(&self, buffer: &mut [u8]) {
        let func: unsafe extern "C" fn(*mut u8) = unsafe { std::mem::transmute(self.func_ptr) };
        unsafe { func(buffer.as_mut_ptr()) };
    }
}

// ─── Span compilation ───────────────────────────────────────────────────────

/// Validate that all stride-based InputRefs access atoms within single slots.
/// Returns a list of errors (empty = ok).
pub fn validate_layout(graph: &NanoGraph, layout: &BufferLayout) -> Vec<String> {
    let mut errors = Vec::new();
    for (gi, group) in graph.groups().iter().enumerate() {
        for (ii, ir) in group.inputs.iter().enumerate() {
            match ir {
                InputRef::Affine { base, stride } if *stride != 0 => {
                    let first_atom =
                        (base.0 as i64 + *stride * group.atom_offset as i64) as u64;
                    let last_atom = (base.0 as i64
                        + *stride * (group.atom_offset + group.count - 1) as i64)
                        as u64;
                    let lo = first_atom.min(last_atom);
                    let hi = first_atom.max(last_atom);
                    if let (Some((slot_lo, _)), Some((slot_hi, _))) =
                        (layout.find(AtomId(lo)), layout.find(AtomId(hi)))
                    {
                        if slot_lo.atom_base != slot_hi.atom_base
                            && slot_lo.elem_bytes == slot_hi.elem_bytes
                        {
                            // Check proportionality: byte_offset difference should match
                            // atom_base difference * elem_bytes (slab-compatible layout).
                            let atom_delta = slot_hi.atom_base.0 as i64 - slot_lo.atom_base.0 as i64;
                            let byte_delta = slot_hi.byte_offset as i64 - slot_lo.byte_offset as i64;
                            let expected_byte_delta = atom_delta * slot_lo.elem_bytes as i64;
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
                    if let InputRef::Affine { base, stride } = ir {
                        let first =
                            (base.0 as i64 + *stride * group.atom_offset as i64) as u64;
                        let last = (base.0 as i64
                            + *stride * (group.atom_offset + group.count - 1) as i64)
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
                                && slot_lo.elem_bytes == slot_hi.elem_bytes
                            {
                                let atom_delta = slot_hi.atom_base.0 as i64 - slot_lo.atom_base.0 as i64;
                                let byte_delta = slot_hi.byte_offset as i64 - slot_lo.byte_offset as i64;
                                let expected = atom_delta * slot_lo.elem_bytes as i64;
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

/// Compile a span's NanoGraph into native code using the given buffer layout.
pub fn compile_span(graph: &NanoGraph, layout: &BufferLayout) -> Result<CompiledSpan, String> {
    let mut flag_builder = settings::builder();
    flag_builder.set("opt_level", "speed").unwrap();
    let isa_builder =
        cranelift_native::builder().map_err(|e| format!("cranelift native ISA: {}", e))?;
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .map_err(|e| format!("ISA finish: {}", e))?;

    let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    register_math_symbols(&mut jit_builder);

    let mut module = JITModule::new(jit_builder);
    let math = declare_math_funcs(&mut module)?;

    let mut ctx = module.make_context();
    ctx.func.signature.params.push(AbiParam::new(types::I64)); // buffer ptr

    let func_id = module
        .declare_function("span_main", Linkage::Local, &ctx.func.signature)
        .map_err(|e| format!("declare: {}", e))?;

    {
        let mut func_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        builder.seal_block(entry);

        let buffer_ptr = builder.block_params(entry)[0];
        let mut var_counter = VarCounter::new();
        let mut table_counter = 0usize;

        for (gi, group) in graph.groups().iter().enumerate() {
            if matches!(&group.op, ScalarOp::Literal(_)) {
                continue; // Pre-filled by caller.
            }
            // Skip dead groups: their slots may have been reused by later groups,
            // so emitting a write would corrupt the new occupant.
            if gi < layout.group_use_counts.len() && layout.group_use_counts[gi] == 0 {
                continue;
            }
            emit_group(
                &mut builder,
                &mut module,
                group,
                layout,
                buffer_ptr,
                &math,
                &mut var_counter,
                &mut table_counter,
            )?;
        }

        builder.ins().return_(&[]);
        builder.finalize();
    }

    module
        .define_function(func_id, &mut ctx)
        .map_err(|e| format!("define: {}", e))?;
    module
        .finalize_definitions()
        .map_err(|e| format!("finalize: {}", e))?;

    let func_ptr = module.get_finalized_function(func_id);

    Ok(CompiledSpan {
        func_ptr,
        _module: module,
    })
}

// ─── Group emission ─────────────────────────────────────────────────────────

/// Emit Cranelift IR for one group. Generates a loop over the group's atom
/// range, or inline code for single-element groups.
fn emit_group(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    group: &AtomGroup,
    layout: &BufferLayout,
    buffer_ptr: Value,
    math: &MathFuncs,
    var_counter: &mut VarCounter,
    table_counter: &mut usize,
) -> Result<(), String> {
    let count = group.count;
    let atom_offset = group.atom_offset;

    if count == 0 {
        return Ok(());
    }

    // Single element: no loop.
    if count == 1 {
        return emit_group_body(
            builder,
            module,
            group,
            layout,
            buffer_ptr,
            None,
            atom_offset,
            math,
            var_counter,
            table_counter,
        );
    }

    // Loop: for i in atom_offset..atom_offset+count
    let loop_header = builder.create_block();
    let loop_body = builder.create_block();
    let loop_exit = builder.create_block();

    let start = builder.ins().iconst(types::I64, atom_offset as i64);
    let end = builder
        .ins()
        .iconst(types::I64, (atom_offset + count) as i64);

    builder.ins().jump(loop_header, &[start]);

    builder.switch_to_block(loop_header);
    builder.append_block_param(loop_header, types::I64);
    let i_val = builder.block_params(loop_header)[0];

    let cmp = builder
        .ins()
        .icmp(IntCC::SignedLessThan, i_val, end);
    builder.ins().brif(cmp, loop_body, &[], loop_exit, &[]);

    builder.switch_to_block(loop_body);

    emit_group_body(
        builder,
        module,
        group,
        layout,
        buffer_ptr,
        Some(i_val),
        0,
        math,
        var_counter,
        table_counter,
    )?;

    let i_next = builder.ins().iadd_imm(i_val, 1);
    builder.ins().jump(loop_header, &[i_next]);

    builder.switch_to_block(loop_exit);
    builder.seal_block(loop_header);
    builder.seal_block(loop_body);
    builder.seal_block(loop_exit);

    Ok(())
}

// ─── Group body emission ────────────────────────────────────────────────────

/// Emit one iteration of a group's computation.
///
/// `i_val` is the loop variable (atom_offset..atom_offset+count), or None
/// for single-element groups (use `i_const`).
fn emit_group_body(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    group: &AtomGroup,
    layout: &BufferLayout,
    buffer_ptr: Value,
    i_val: Option<Value>,
    i_const: u64,
    math: &MathFuncs,
    var_counter: &mut VarCounter,
    table_counter: &mut usize,
) -> Result<(), String> {
    let (out_slot, _) = layout
        .find(group.base_id)
        .ok_or_else(|| format!("no slot for group base={}", group.base_id))?;
    let out_slot = out_slot.clone();

    let output_dtype = group.output_dtype;
    let output_repr = repr_of(output_dtype);

    match &group.op {
        ScalarOp::Literal(_) => Ok(()),

        ScalarOp::Identity => {
            // Identity: cast input to output_dtype.
            let src = load_input(
                builder, module, &group.inputs[0], layout, buffer_ptr, i_val, i_const,
                table_counter,
            )?;
            let src_repr = input_slot_dtype(&group.inputs[0], layout)
                .map(repr_of).unwrap_or(ReprKind::Float);
            let result = emit_cast_to_output(builder, src, src_repr, output_dtype);
            store_result(builder, buffer_ptr, &out_slot, group.atom_offset, i_val, i_const, result);
            Ok(())
        }

        ScalarOp::Binary { op, compute_dtype } => {
            let compute_repr = repr_of(*compute_dtype);
            let a_raw = load_input(
                builder, module, &group.inputs[0], layout, buffer_ptr, i_val, i_const,
                table_counter,
            )?;
            let a_repr = input_slot_dtype(&group.inputs[0], layout)
                .map(repr_of).unwrap_or(compute_repr);
            let a = emit_repr_cast(builder, a_raw, a_repr, compute_repr);

            let b_raw = load_input(
                builder, module, &group.inputs[1], layout, buffer_ptr, i_val, i_const,
                table_counter,
            )?;
            let b_repr = input_slot_dtype(&group.inputs[1], layout)
                .map(repr_of).unwrap_or(compute_repr);
            let b = emit_repr_cast(builder, b_raw, b_repr, compute_repr);

            let result = emit_binop(builder, module, math, *op, a, b, compute_repr)?;
            let output_val = emit_cast_to_output(builder, result, compute_repr, output_dtype);
            store_result(
                builder, buffer_ptr, &out_slot, group.atom_offset, i_val, i_const, output_val,
            );
            Ok(())
        }

        ScalarOp::Unary { op, compute_dtype } => {
            let compute_repr = repr_of(*compute_dtype);
            let x_raw = load_input(
                builder, module, &group.inputs[0], layout, buffer_ptr, i_val, i_const,
                table_counter,
            )?;
            let x_repr = input_slot_dtype(&group.inputs[0], layout)
                .map(repr_of).unwrap_or(compute_repr);
            let x = emit_repr_cast(builder, x_raw, x_repr, compute_repr);

            let result = emit_unop(builder, module, math, *op, x, compute_repr)?;
            let output_val = emit_cast_to_output(builder, result, compute_repr, output_dtype);
            store_result(
                builder, buffer_ptr, &out_slot, group.atom_offset, i_val, i_const, output_val,
            );
            Ok(())
        }

        ScalarOp::Select => {
            // Select: truthiness test on cond, then cast selected value to output_dtype.
            let cond = load_input(
                builder, module, &group.inputs[0], layout, buffer_ptr, i_val, i_const,
                table_counter,
            )?;
            let cond_repr = input_slot_dtype(&group.inputs[0], layout)
                .map(repr_of).unwrap_or(ReprKind::Float);

            let is_nonzero = match cond_repr {
                ReprKind::Float => {
                    let zero = builder.ins().f32const(0.0);
                    builder.ins().fcmp(FloatCC::NotEqual, cond, zero)
                }
                ReprKind::Int => {
                    let zero = builder.ins().iconst(types::I64, 0);
                    builder.ins().icmp(IntCC::NotEqual, cond, zero)
                }
            };

            // Load x and y, cast both to output_dtype.
            let x_raw = load_input(
                builder, module, &group.inputs[1], layout, buffer_ptr, i_val, i_const,
                table_counter,
            )?;
            let x_repr = input_slot_dtype(&group.inputs[1], layout)
                .map(repr_of).unwrap_or(output_repr);
            let x = emit_cast_to_output(builder, x_raw, x_repr, output_dtype);

            let y_raw = load_input(
                builder, module, &group.inputs[2], layout, buffer_ptr, i_val, i_const,
                table_counter,
            )?;
            let y_repr = input_slot_dtype(&group.inputs[2], layout)
                .map(repr_of).unwrap_or(output_repr);
            let y = emit_cast_to_output(builder, y_raw, y_repr, output_dtype);

            let result = builder.ins().select(is_nonzero, x, y);
            store_result(
                builder, buffer_ptr, &out_slot, group.atom_offset, i_val, i_const, result,
            );
            Ok(())
        }

        ScalarOp::Reduce {
            kind,
            reduce_count,
            reduce_stride,
            compute_dtype,
        } => emit_reduce(
            builder,
            module,
            group,
            layout,
            buffer_ptr,
            i_val,
            i_const,
            math,
            var_counter,
            table_counter,
            *kind,
            *reduce_count,
            *reduce_stride,
            *compute_dtype,
            &out_slot,
        ),

        ScalarOp::IndirectLoad { table_base } => {
            // Index: load as integer regardless of source dtype.
            let idx_raw = load_input(
                builder, module, &group.inputs[0], layout, buffer_ptr, i_val, i_const,
                table_counter,
            )?;
            let idx_repr = input_slot_dtype(&group.inputs[0], layout)
                .map(repr_of).unwrap_or(ReprKind::Int);
            let idx_i64 = emit_repr_cast(builder, idx_raw, idx_repr, ReprKind::Int);

            let (table_slot, _) = layout
                .find(*table_base)
                .ok_or_else(|| format!("no slot for IndirectLoad table_base={}", table_base))?;

            // address = buffer_ptr + table_slot.byte_offset + idx * elem_bytes
            let base = builder
                .ins()
                .iconst(types::I64, table_slot.byte_offset as i64);
            let idx_bytes = builder
                .ins()
                .imul_imm(idx_i64, table_slot.elem_bytes as i64);
            let offset = builder.ins().iadd(base, idx_bytes);
            let addr = builder.ins().iadd(buffer_ptr, offset);
            let loaded = emit_typed_load(builder, addr, table_slot.dtype);
            let loaded_repr = repr_of(table_slot.dtype);

            // Cast to output_dtype.
            let result = emit_cast_to_output(builder, loaded, loaded_repr, output_dtype);
            store_result(
                builder, buffer_ptr, &out_slot, group.atom_offset, i_val, i_const, result,
            );
            Ok(())
        }
    }
}

// ─── Input loading ──────────────────────────────────────────────────────────

/// Determine the storage dtype that `load_input` will load from for a given InputRef.
fn input_slot_dtype(input: &InputRef, layout: &BufferLayout) -> Option<DType> {
    match input {
        InputRef::Broadcast(atom_id) => layout.find(*atom_id).map(|(s, _)| s.dtype),
        InputRef::Affine { base, .. } => layout.find(*base).map(|(s, _)| s.dtype),
        InputRef::StridedBroadcast { base, .. } => layout.find(*base).map(|(s, _)| s.dtype),
        InputRef::Modular { base, .. } => layout.find(*base).map(|(s, _)| s.dtype),
        InputRef::Explicit(ids) if !ids.is_empty() => layout.find(ids[0]).map(|(s, _)| s.dtype),
        _ => None,
    }
}

/// Load a value from an InputRef, resolving to a buffer byte address.
///
/// Returns a Cranelift Value in the storage dtype's representation kind
/// (types::F32 for float dtypes, types::I64 for integer dtypes).
fn load_input(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    input: &InputRef,
    layout: &BufferLayout,
    buffer_ptr: Value,
    i_val: Option<Value>,
    i_const: u64,
    table_counter: &mut usize,
) -> Result<Value, String> {
    match input {
        InputRef::Broadcast(atom_id) => {
            let (slot, elem_idx) = layout
                .find(*atom_id)
                .ok_or_else(|| format!("no slot for Broadcast atom={}", atom_id))?;
            let byte_off = slot.byte_offset as i64 + elem_idx as i64 * slot.elem_bytes as i64;
            let addr = addr_const(builder, buffer_ptr, byte_off);
            Ok(emit_typed_load(builder, addr, slot.dtype))
        }

        InputRef::Affine { base, stride } => {
            let (slot, slot_elem_base) = layout
                .find(*base)
                .ok_or_else(|| format!("no slot for Affine base={}", base))?;
            let base_byte =
                slot.byte_offset as i64 + slot_elem_base as i64 * slot.elem_bytes as i64;
            let byte_stride = *stride * slot.elem_bytes as i64;
            let load_dtype = slot.dtype;

            let addr = match i_val {
                Some(iv) => {
                    let i_bytes = builder.ins().imul_imm(iv, byte_stride);
                    let base_val = builder.ins().iconst(types::I64, base_byte);
                    let off = builder.ins().iadd(base_val, i_bytes);
                    builder.ins().iadd(buffer_ptr, off)
                }
                None => {
                    let byte_off = base_byte + byte_stride * i_const as i64;
                    addr_const(builder, buffer_ptr, byte_off)
                }
            };
            Ok(emit_typed_load(builder, addr, load_dtype))
        }

        InputRef::StridedBroadcast {
            base,
            stride,
            repeat,
        } => {
            let (slot, slot_elem_base) = layout
                .find(*base)
                .ok_or_else(|| format!("no slot for StridedBroadcast base={}", base))?;
            let base_byte =
                slot.byte_offset as i64 + slot_elem_base as i64 * slot.elem_bytes as i64;
            let byte_stride = *stride * slot.elem_bytes as i64;
            let load_dtype = slot.dtype;

            let addr = match i_val {
                Some(iv) => {
                    let block_idx = if repeat.is_power_of_two() {
                        let shift = repeat.trailing_zeros() as i64;
                        builder.ins().ushr_imm(iv, shift)
                    } else {
                        let rep = builder.ins().iconst(types::I64, *repeat as i64);
                        builder.ins().udiv(iv, rep)
                    };
                    let elem_off = builder.ins().imul_imm(block_idx, byte_stride);
                    let base_val = builder.ins().iconst(types::I64, base_byte);
                    let off = builder.ins().iadd(base_val, elem_off);
                    builder.ins().iadd(buffer_ptr, off)
                }
                None => {
                    let block = i_const / repeat;
                    let byte_off = base_byte + byte_stride * block as i64;
                    addr_const(builder, buffer_ptr, byte_off)
                }
            };
            Ok(emit_typed_load(builder, addr, load_dtype))
        }

        InputRef::Modular {
            base,
            stride,
            modulus,
        } => {
            let (slot, slot_elem_base) = layout
                .find(*base)
                .ok_or_else(|| format!("no slot for Modular base={}", base))?;
            let base_byte =
                slot.byte_offset as i64 + slot_elem_base as i64 * slot.elem_bytes as i64;
            let byte_stride = *stride as i64 * slot.elem_bytes as i64;
            let load_dtype = slot.dtype;

            let addr = match i_val {
                Some(iv) => {
                    let modval = builder.ins().iconst(types::I64, *modulus as i64);
                    let wrapped = builder.ins().urem(iv, modval);
                    let elem_off = builder.ins().imul_imm(wrapped, byte_stride);
                    let base_val = builder.ins().iconst(types::I64, base_byte);
                    let off = builder.ins().iadd(base_val, elem_off);
                    builder.ins().iadd(buffer_ptr, off)
                }
                None => {
                    let wrapped = i_const % modulus;
                    let byte_off = base_byte + byte_stride * wrapped as i64;
                    addr_const(builder, buffer_ptr, byte_off)
                }
            };
            Ok(emit_typed_load(builder, addr, load_dtype))
        }

        InputRef::Explicit(ids) => {
            if ids.len() == 1 {
                let (slot, elem_idx) = layout
                    .find(ids[0])
                    .ok_or_else(|| format!("no slot for Explicit[0] atom={}", ids[0]))?;
                let byte_off =
                    slot.byte_offset as i64 + elem_idx as i64 * slot.elem_bytes as i64;
                let addr = addr_const(builder, buffer_ptr, byte_off);
                return Ok(emit_typed_load(builder, addr, slot.dtype));
            }

            // Determine load dtype from first entry.
            let (first_slot, _) = layout
                .find(ids[0])
                .ok_or_else(|| format!("no slot for Explicit[0] atom={}", ids[0]))?;
            let load_dtype = first_slot.dtype;

            // Build lookup table of byte offsets (resolved at compile time).
            let byte_offsets: Vec<i64> = ids
                .iter()
                .map(|id| {
                    let (slot, elem_idx) = layout.find(*id).expect("Explicit atom not in layout");
                    slot.byte_offset as i64 + elem_idx as i64 * slot.elem_bytes as i64
                })
                .collect();

            let data_name = format!("explicit_{}", *table_counter);
            *table_counter += 1;

            let data_id = module
                .declare_data(&data_name, Linkage::Local, false, false)
                .map_err(|e| format!("declare explicit table: {}", e))?;
            let mut data_desc = cranelift_module::DataDescription::new();
            let bytes: Vec<u8> = byte_offsets
                .iter()
                .flat_map(|off| off.to_le_bytes())
                .collect();
            data_desc.define(bytes.into_boxed_slice());
            module
                .define_data(data_id, &data_desc)
                .map_err(|e| format!("define explicit table: {}", e))?;

            let gv = module.declare_data_in_func(data_id, builder.func);
            let table_ptr = builder.ins().global_value(types::I64, gv);

            let i = i_val.unwrap_or_else(|| builder.ins().iconst(types::I64, i_const as i64));
            let idx_byte_off = builder.ins().imul_imm(i, 8); // 8 bytes per i64 entry
            let idx_addr = builder.ins().iadd(table_ptr, idx_byte_off);
            let byte_off = builder
                .ins()
                .load(types::I64, MemFlags::new(), idx_addr, 0);
            let addr = builder.ins().iadd(buffer_ptr, byte_off);
            Ok(emit_typed_load(builder, addr, load_dtype))
        }
    }
}

/// Helper: buffer_ptr + constant byte offset.
fn addr_const(builder: &mut FunctionBuilder, buffer_ptr: Value, byte_off: i64) -> Value {
    if byte_off == 0 {
        buffer_ptr
    } else {
        let off = builder.ins().iconst(types::I64, byte_off);
        builder.ins().iadd(buffer_ptr, off)
    }
}

// ─── Representation kinds ───────────────────────────────────────────────────

/// Whether a Cranelift value is floating-point or integer.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ReprKind {
    Float, // types::F32
    Int,   // types::I64
}

/// Map a DType to its Cranelift representation kind.
fn repr_of(dtype: DType) -> ReprKind {
    match dtype {
        DType::F32 | DType::BF16 | DType::F16 | DType::F64 => ReprKind::Float,
        _ => ReprKind::Int,
    }
}

/// Emit a cast between Cranelift representation kinds.
/// If from == to, returns val unchanged. Otherwise converts f32↔i64.
fn emit_repr_cast(builder: &mut FunctionBuilder, val: Value, from: ReprKind, to: ReprKind) -> Value {
    match (from, to) {
        (ReprKind::Float, ReprKind::Float) | (ReprKind::Int, ReprKind::Int) => val,
        (ReprKind::Float, ReprKind::Int) => builder.ins().fcvt_to_sint_sat(types::I64, val),
        (ReprKind::Int, ReprKind::Float) => builder.ins().fcvt_from_sint(types::F32, val),
    }
}

/// Emit a cast from a compute-repr value to the output dtype's storage repr.
/// Handles narrowing (e.g., i64 → i8 for BOOL, f32 → bf16 bits for BF16).
/// Returns a value ready for `emit_typed_store`.
fn emit_cast_to_output(builder: &mut FunctionBuilder, val: Value, compute_repr: ReprKind, output_dtype: DType) -> Value {
    let target_repr = repr_of(output_dtype);
    let val = emit_repr_cast(builder, val, compute_repr, target_repr);

    // Further narrowing for sub-word output types.
    match output_dtype {
        DType::BOOL => {
            // Nonzero test → 0 or 1 as i8.
            match target_repr {
                ReprKind::Int => {
                    let zero = builder.ins().iconst(types::I64, 0);
                    let is_nz = builder.ins().icmp(IntCC::NotEqual, val, zero);
                    let one = builder.ins().iconst(types::I8, 1);
                    let zero8 = builder.ins().iconst(types::I8, 0);
                    builder.ins().select(is_nz, one, zero8)
                }
                ReprKind::Float => {
                    // Shouldn't happen after repr_cast, but handle defensively.
                    let zero = builder.ins().f32const(0.0);
                    let is_nz = builder.ins().fcmp(FloatCC::NotEqual, val, zero);
                    let one = builder.ins().iconst(types::I8, 1);
                    let zero8 = builder.ins().iconst(types::I8, 0);
                    builder.ins().select(is_nz, one, zero8)
                }
            }
        }
        DType::U8 | DType::I8 => {
            // Already Int repr (i64), narrow to i8.
            builder.ins().ireduce(types::I8, val)
        }
        DType::I32 | DType::U32 => {
            // Already Int repr (i64), narrow to i32.
            builder.ins().ireduce(types::I32, val)
        }
        // F32, BF16, I64 — val is already in the right repr, store handles format.
        _ => val,
    }
}

// ─── Typed load/store ───────────────────────────────────────────────────────

/// Load from a buffer address in native storage format.
///
/// Returns a value in the dtype's representation kind:
/// - Float dtypes → types::F32 (BF16 widened to f32)
/// - Integer dtypes → types::I64 (smaller ints zero/sign-extended)
fn emit_typed_load(builder: &mut FunctionBuilder, addr: Value, dtype: DType) -> Value {
    match dtype {
        DType::F32 => builder.ins().load(types::F32, MemFlags::trusted(), addr, 0),
        DType::BF16 => {
            let raw = builder.ins().load(types::I16, MemFlags::trusted(), addr, 0);
            let wide = builder.ins().uextend(types::I32, raw);
            let shifted = builder.ins().ishl_imm(wide, 16);
            builder.ins().bitcast(types::F32, MemFlags::new(), shifted)
        }
        DType::F64 => {
            let raw = builder.ins().load(types::F64, MemFlags::trusted(), addr, 0);
            builder.ins().fdemote(types::F32, raw)
        }
        DType::I64 => builder.ins().load(types::I64, MemFlags::trusted(), addr, 0),
        DType::I32 => {
            let raw = builder.ins().load(types::I32, MemFlags::trusted(), addr, 0);
            builder.ins().sextend(types::I64, raw)
        }
        DType::U32 => {
            let raw = builder.ins().load(types::I32, MemFlags::trusted(), addr, 0);
            builder.ins().uextend(types::I64, raw)
        }
        DType::BOOL | DType::U8 => {
            let raw = builder.ins().load(types::I8, MemFlags::trusted(), addr, 0);
            builder.ins().uextend(types::I64, raw)
        }
        DType::I8 => {
            let raw = builder.ins().load(types::I8, MemFlags::trusted(), addr, 0);
            builder.ins().sextend(types::I64, raw)
        }
        _ => {
            // Fallback: assume 4-byte float-like.
            builder.ins().load(types::F32, MemFlags::trusted(), addr, 0)
        }
    }
}

/// Store a value to a buffer address in native storage format.
///
/// `val` must be in the correct Cranelift type for the dtype:
/// - F32: types::F32
/// - BF16: types::F32 (will be rounded and narrowed)
/// - I64: types::I64
/// - BOOL/U8/I8: types::I8
/// - I32/U32: types::I32
fn emit_typed_store(builder: &mut FunctionBuilder, addr: Value, val: Value, dtype: DType) {
    match dtype {
        DType::F32 => {
            builder.ins().store(MemFlags::trusted(), val, addr, 0);
        }
        DType::BF16 => {
            // Round f32 → BF16: bitcast to i32, round-to-nearest-even, take top 16 bits.
            let bits = builder.ins().bitcast(types::I32, MemFlags::new(), val);
            let shifted16 = builder.ins().ushr_imm(bits, 16);
            let lsb = builder.ins().band_imm(shifted16, 1);
            let bias = builder.ins().iadd_imm(lsb, 0x7FFF);
            let rounded = builder.ins().iadd(bits, bias);
            let top = builder.ins().ushr_imm(rounded, 16);
            let narrow = builder.ins().ireduce(types::I16, top);
            builder.ins().store(MemFlags::trusted(), narrow, addr, 0);
        }
        DType::I64 | DType::U64 => {
            builder.ins().store(MemFlags::trusted(), val, addr, 0); // val is i64
        }
        DType::I32 | DType::U32 => {
            builder.ins().store(MemFlags::trusted(), val, addr, 0); // val is i32
        }
        DType::BOOL | DType::U8 | DType::I8 => {
            builder.ins().store(MemFlags::trusted(), val, addr, 0); // val is i8
        }
        _ => {
            builder.ins().store(MemFlags::trusted(), val, addr, 0);
        }
    }
}

// ─── Output store ───────────────────────────────────────────────────────────

/// Store a computed f32 value to the group's output slot.
///
/// Loop variable `i_val` ranges from `atom_offset` to `atom_offset + count`.
/// Slot element index = `i - atom_offset`.
fn store_result(
    builder: &mut FunctionBuilder,
    buffer_ptr: Value,
    slot: &SlotInfo,
    atom_offset: u64,
    i_val: Option<Value>,
    i_const: u64,
    val: Value,
) {
    // store_base = slot.byte_offset - atom_offset * elem_bytes
    // addr = buffer_ptr + store_base + i * elem_bytes
    let store_base = slot.byte_offset as i64 - atom_offset as i64 * slot.elem_bytes as i64;

    let addr = match i_val {
        Some(iv) => {
            let i_bytes = builder.ins().imul_imm(iv, slot.elem_bytes as i64);
            let base_val = builder.ins().iconst(types::I64, store_base);
            let off = builder.ins().iadd(base_val, i_bytes);
            builder.ins().iadd(buffer_ptr, off)
        }
        None => {
            let byte_off = store_base + i_const as i64 * slot.elem_bytes as i64;
            addr_const(builder, buffer_ptr, byte_off)
        }
    };

    emit_typed_store(builder, addr, val, slot.dtype);
}

// ─── Reduce emission ────────────────────────────────────────────────────────

fn emit_reduce(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    group: &AtomGroup,
    layout: &BufferLayout,
    buffer_ptr: Value,
    i_val: Option<Value>,
    i_const: u64,
    math: &MathFuncs,
    var_counter: &mut VarCounter,
    table_counter: &mut usize,
    kind: ReduceKind,
    reduce_count: u64,
    reduce_stride: i64,
    compute_dtype: DType,
    out_slot: &SlotInfo,
) -> Result<(), String> {
    let is_sum = matches!(kind, ReduceKind::Sum);
    let compute_repr = repr_of(compute_dtype);
    let output_dtype = group.output_dtype;

    // Resolve the source slot. Reduce input must be Affine.
    let (base_byte, input_byte_stride, reduce_byte_stride, src_dtype) = match &group.inputs[0] {
        InputRef::Affine { base, stride } => {
            let (slot, elem) = layout
                .find(*base)
                .ok_or_else(|| format!("no slot for reduce input base={}", base))?;
            let base_byte = slot.byte_offset as i64 + elem as i64 * slot.elem_bytes as i64;
            let byte_stride = *stride * slot.elem_bytes as i64;
            let red_stride = reduce_stride * slot.elem_bytes as i64;
            (base_byte, byte_stride, red_stride, slot.dtype)
        }
        other => {
            return Err(format!(
                "Reduce input must be Affine, got {:?}",
                std::mem::discriminant(other)
            ))
        }
    };
    let src_repr = repr_of(src_dtype);

    // Accumulator variable — type depends on compute_repr.
    let acc_var = var_counter.next();
    let acc_cl_type = match compute_repr {
        ReprKind::Float => types::F32,
        ReprKind::Int => types::I64,
    };
    builder.declare_var(acc_var, acc_cl_type);
    let init = match (is_sum, compute_repr) {
        (true, ReprKind::Float) => builder.ins().f32const(0.0),
        (false, ReprKind::Float) => builder.ins().f32const(f32::NEG_INFINITY),
        (true, ReprKind::Int) => builder.ins().iconst(types::I64, 0),
        (false, ReprKind::Int) => builder.ins().iconst(types::I64, i64::MIN),
    };
    builder.def_var(acc_var, init);

    // Base address for this iteration's reduce input at k=0.
    let base_addr_off = match i_val {
        Some(iv) => {
            let i_bytes = builder.ins().imul_imm(iv, input_byte_stride);
            let base_val = builder.ins().iconst(types::I64, base_byte);
            builder.ins().iadd(base_val, i_bytes)
        }
        None => {
            let off = base_byte + input_byte_stride * i_const as i64;
            builder.ins().iconst(types::I64, off)
        }
    };

    // Inner loop: for k in 0..reduce_count
    let k_var = var_counter.next();
    builder.declare_var(k_var, types::I64);
    let k_init = builder.ins().iconst(types::I64, 0);
    builder.def_var(k_var, k_init);

    let red_header = builder.create_block();
    let red_body = builder.create_block();
    let red_exit = builder.create_block();

    let bound = builder.ins().iconst(types::I64, reduce_count as i64);
    builder.ins().jump(red_header, &[]);

    builder.switch_to_block(red_header);
    let k_val = builder.use_var(k_var);
    let k_cmp = builder
        .ins()
        .icmp(IntCC::SignedLessThan, k_val, bound);
    builder.ins().brif(k_cmp, red_body, &[], red_exit, &[]);

    // Body: load from base_addr_off + k * reduce_byte_stride, accumulate.
    builder.switch_to_block(red_body);
    let k_val = builder.use_var(k_var);
    let k_offset = builder.ins().imul_imm(k_val, reduce_byte_stride);
    let src_off = builder.ins().iadd(base_addr_off, k_offset);
    let src_addr = builder.ins().iadd(buffer_ptr, src_off);
    let loaded = emit_typed_load(builder, src_addr, src_dtype);
    // Cast loaded value to compute repr.
    let src_val = emit_repr_cast(builder, loaded, src_repr, compute_repr);

    let acc = builder.use_var(acc_var);
    let new_acc = match (is_sum, compute_repr) {
        (true, ReprKind::Float) => builder.ins().fadd(acc, src_val),
        (false, ReprKind::Float) => {
            let cmp = builder.ins().fcmp(FloatCC::GreaterThan, src_val, acc);
            builder.ins().select(cmp, src_val, acc)
        }
        (true, ReprKind::Int) => builder.ins().iadd(acc, src_val),
        (false, ReprKind::Int) => {
            let cmp = builder.ins().icmp(IntCC::SignedGreaterThan, src_val, acc);
            builder.ins().select(cmp, src_val, acc)
        }
    };
    builder.def_var(acc_var, new_acc);

    let k_next = builder.ins().iadd_imm(k_val, 1);
    builder.def_var(k_var, k_next);
    builder.ins().jump(red_header, &[]);

    builder.switch_to_block(red_exit);
    builder.seal_block(red_header);
    builder.seal_block(red_body);
    builder.seal_block(red_exit);

    let final_acc = builder.use_var(acc_var);
    let output_val = emit_cast_to_output(builder, final_acc, compute_repr, output_dtype);
    store_result(
        builder,
        buffer_ptr,
        out_slot,
        group.atom_offset,
        i_val,
        i_const,
        output_val,
    );

    Ok(())
}

// ─── Scalar op emission ─────────────────────────────────────────────────────

fn emit_binop(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    math: &MathFuncs,
    op: ScalarBinOp,
    a: Value,
    b: Value,
    compute_repr: ReprKind,
) -> Result<Value, String> {
    // Comparison results: return in the compute repr (f32 1.0/0.0 or i64 1/0).
    macro_rules! cmp_result {
        ($cmp:expr) => {
            match compute_repr {
                ReprKind::Float => {
                    let one = builder.ins().f32const(1.0);
                    let zero = builder.ins().f32const(0.0);
                    builder.ins().select($cmp, one, zero)
                }
                ReprKind::Int => {
                    let one = builder.ins().iconst(types::I64, 1);
                    let zero = builder.ins().iconst(types::I64, 0);
                    builder.ins().select($cmp, one, zero)
                }
            }
        };
    }

    Ok(match (op, compute_repr) {
        // ── Arithmetic ──
        (ScalarBinOp::Add, ReprKind::Float) => builder.ins().fadd(a, b),
        (ScalarBinOp::Add, ReprKind::Int) => builder.ins().iadd(a, b),
        (ScalarBinOp::Sub, ReprKind::Float) => builder.ins().fsub(a, b),
        (ScalarBinOp::Sub, ReprKind::Int) => builder.ins().isub(a, b),
        (ScalarBinOp::Mul, ReprKind::Float) => builder.ins().fmul(a, b),
        (ScalarBinOp::Mul, ReprKind::Int) => builder.ins().imul(a, b),
        (ScalarBinOp::Div, ReprKind::Float) => builder.ins().fdiv(a, b),
        (ScalarBinOp::Div, ReprKind::Int) => {
            // Guard against division by zero (which traps on x86).
            // If b == 0, result is 0 (matches NumericScalar behavior).
            let zero = builder.ins().iconst(types::I64, 0);
            let one = builder.ins().iconst(types::I64, 1);
            let is_zero = builder.ins().icmp(IntCC::Equal, b, zero);
            let safe_b = builder.ins().select(is_zero, one, b);
            let quot = builder.ins().sdiv(a, safe_b);
            builder.ins().select(is_zero, zero, quot)
        }
        (ScalarBinOp::Mod, ReprKind::Float) => {
            let func_ref = module.declare_func_in_func(math.fmodf, builder.func);
            let call = builder.ins().call(func_ref, &[a, b]);
            builder.inst_results(call)[0]
        }
        (ScalarBinOp::Mod, ReprKind::Int) => {
            let zero = builder.ins().iconst(types::I64, 0);
            let one = builder.ins().iconst(types::I64, 1);
            let is_zero = builder.ins().icmp(IntCC::Equal, b, zero);
            let safe_b = builder.ins().select(is_zero, one, b);
            let rem = builder.ins().srem(a, safe_b);
            builder.ins().select(is_zero, zero, rem)
        }

        // ── Min/Max ──
        (ScalarBinOp::Max, ReprKind::Float) => {
            let cmp = builder.ins().fcmp(FloatCC::GreaterThan, a, b);
            builder.ins().select(cmp, a, b)
        }
        (ScalarBinOp::Max, ReprKind::Int) => {
            let cmp = builder.ins().icmp(IntCC::SignedGreaterThan, a, b);
            builder.ins().select(cmp, a, b)
        }
        (ScalarBinOp::Min, ReprKind::Float) => {
            let cmp = builder.ins().fcmp(FloatCC::LessThan, a, b);
            builder.ins().select(cmp, a, b)
        }
        (ScalarBinOp::Min, ReprKind::Int) => {
            let cmp = builder.ins().icmp(IntCC::SignedLessThan, a, b);
            builder.ins().select(cmp, a, b)
        }

        // ── Pow (float only; integer pow not needed for now) ──
        (ScalarBinOp::Pow, ReprKind::Float) => {
            let func_ref = module.declare_func_in_func(math.powf, builder.func);
            let call = builder.ins().call(func_ref, &[a, b]);
            builder.inst_results(call)[0]
        }
        (ScalarBinOp::Pow, ReprKind::Int) => {
            return Err("integer Pow not implemented".into());
        }

        // ── Comparisons ──
        (ScalarBinOp::Equal, ReprKind::Float) => {
            let cmp = builder.ins().fcmp(FloatCC::Equal, a, b);
            cmp_result!(cmp)
        }
        (ScalarBinOp::Equal, ReprKind::Int) => {
            let cmp = builder.ins().icmp(IntCC::Equal, a, b);
            cmp_result!(cmp)
        }
        (ScalarBinOp::Greater, ReprKind::Float) => {
            let cmp = builder.ins().fcmp(FloatCC::GreaterThan, a, b);
            cmp_result!(cmp)
        }
        (ScalarBinOp::Greater, ReprKind::Int) => {
            let cmp = builder.ins().icmp(IntCC::SignedGreaterThan, a, b);
            cmp_result!(cmp)
        }
        (ScalarBinOp::GreaterOrEqual, ReprKind::Float) => {
            let cmp = builder.ins().fcmp(FloatCC::GreaterThanOrEqual, a, b);
            cmp_result!(cmp)
        }
        (ScalarBinOp::GreaterOrEqual, ReprKind::Int) => {
            let cmp = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, a, b);
            cmp_result!(cmp)
        }
        (ScalarBinOp::Less, ReprKind::Float) => {
            let cmp = builder.ins().fcmp(FloatCC::LessThan, a, b);
            cmp_result!(cmp)
        }
        (ScalarBinOp::Less, ReprKind::Int) => {
            let cmp = builder.ins().icmp(IntCC::SignedLessThan, a, b);
            cmp_result!(cmp)
        }
        (ScalarBinOp::LessOrEqual, ReprKind::Float) => {
            let cmp = builder.ins().fcmp(FloatCC::LessThanOrEqual, a, b);
            cmp_result!(cmp)
        }
        (ScalarBinOp::LessOrEqual, ReprKind::Int) => {
            let cmp = builder.ins().icmp(IntCC::SignedLessThanOrEqual, a, b);
            cmp_result!(cmp)
        }

        // ── Logical (truthiness-based) ──
        (ScalarBinOp::And, ReprKind::Float) => {
            let zero = builder.ins().f32const(0.0);
            let a_nz = builder.ins().fcmp(FloatCC::NotEqual, a, zero);
            let b_nz = builder.ins().fcmp(FloatCC::NotEqual, b, zero);
            let both = builder.ins().band(a_nz, b_nz);
            cmp_result!(both)
        }
        (ScalarBinOp::And, ReprKind::Int) => {
            let zero = builder.ins().iconst(types::I64, 0);
            let a_nz = builder.ins().icmp(IntCC::NotEqual, a, zero);
            let b_nz = builder.ins().icmp(IntCC::NotEqual, b, zero);
            let both = builder.ins().band(a_nz, b_nz);
            cmp_result!(both)
        }
        (ScalarBinOp::Or, ReprKind::Float) => {
            let zero = builder.ins().f32const(0.0);
            let a_nz = builder.ins().fcmp(FloatCC::NotEqual, a, zero);
            let b_nz = builder.ins().fcmp(FloatCC::NotEqual, b, zero);
            let either = builder.ins().bor(a_nz, b_nz);
            cmp_result!(either)
        }
        (ScalarBinOp::Or, ReprKind::Int) => {
            let zero = builder.ins().iconst(types::I64, 0);
            let a_nz = builder.ins().icmp(IntCC::NotEqual, a, zero);
            let b_nz = builder.ins().icmp(IntCC::NotEqual, b, zero);
            let either = builder.ins().bor(a_nz, b_nz);
            cmp_result!(either)
        }
        (ScalarBinOp::Xor, ReprKind::Float) => {
            let zero = builder.ins().f32const(0.0);
            let a_nz = builder.ins().fcmp(FloatCC::NotEqual, a, zero);
            let b_nz = builder.ins().fcmp(FloatCC::NotEqual, b, zero);
            let x = builder.ins().bxor(a_nz, b_nz);
            cmp_result!(x)
        }
        (ScalarBinOp::Xor, ReprKind::Int) => {
            let zero = builder.ins().iconst(types::I64, 0);
            let a_nz = builder.ins().icmp(IntCC::NotEqual, a, zero);
            let b_nz = builder.ins().icmp(IntCC::NotEqual, b, zero);
            let x = builder.ins().bxor(a_nz, b_nz);
            cmp_result!(x)
        }
    })
}

fn emit_unop(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    math: &MathFuncs,
    op: ScalarUnaryOp,
    x: Value,
    compute_repr: ReprKind,
) -> Result<Value, String> {
    Ok(match (op, compute_repr) {
        (ScalarUnaryOp::Neg, ReprKind::Float) => builder.ins().fneg(x),
        (ScalarUnaryOp::Neg, ReprKind::Int) => builder.ins().ineg(x),
        (ScalarUnaryOp::Abs, ReprKind::Float) => builder.ins().fabs(x),
        (ScalarUnaryOp::Abs, ReprKind::Int) => {
            // abs(x) = x < 0 ? -x : x
            let zero = builder.ins().iconst(types::I64, 0);
            let neg = builder.ins().icmp(IntCC::SignedLessThan, x, zero);
            let negated = builder.ins().ineg(x);
            builder.ins().select(neg, negated, x)
        }
        (ScalarUnaryOp::Floor, ReprKind::Float) => builder.ins().floor(x),
        (ScalarUnaryOp::Floor, ReprKind::Int) => x, // no-op for integers
        (ScalarUnaryOp::Ceil, ReprKind::Float) => builder.ins().ceil(x),
        (ScalarUnaryOp::Ceil, ReprKind::Int) => x, // no-op for integers
        // Transcendentals — float only.
        (ScalarUnaryOp::Exp, ReprKind::Float) => {
            let func_ref = module.declare_func_in_func(math.expf, builder.func);
            let call = builder.ins().call(func_ref, &[x]);
            builder.inst_results(call)[0]
        }
        (ScalarUnaryOp::Ln, ReprKind::Float) => {
            let func_ref = module.declare_func_in_func(math.logf, builder.func);
            let call = builder.ins().call(func_ref, &[x]);
            builder.inst_results(call)[0]
        }
        (ScalarUnaryOp::Sqrt, ReprKind::Float) => builder.ins().sqrt(x),
        (ScalarUnaryOp::Reciprocal, ReprKind::Float) => {
            let one = builder.ins().f32const(1.0);
            builder.ins().fdiv(one, x)
        }
        (ScalarUnaryOp::Tanh, ReprKind::Float) => {
            let func_ref = module.declare_func_in_func(math.tanhf, builder.func);
            let call = builder.ins().call(func_ref, &[x]);
            builder.inst_results(call)[0]
        }
        // Integer transcendentals — not meaningful, but cast through f32 if needed.
        (_, ReprKind::Int) => {
            return Err(format!("integer {:?} not implemented", op));
        }
    })
}

fn op_name_short(op: &ScalarOp) -> &'static str {
    match op {
        ScalarOp::Literal(_) => "Lit",
        ScalarOp::Identity => "Id",
        ScalarOp::Binary { .. } => "Bin",
        ScalarOp::Unary { .. } => "Un",
        ScalarOp::Select => "Sel",
        ScalarOp::Reduce { .. } => "Red",
        ScalarOp::IndirectLoad { .. } => "Ind",
    }
}

// ─── Compiled execution plan ─────────────────────────────────────────────────

use ndarray::{ArcArray, IxDyn};

use crate::backends::ndarray_backend::numeric_tensor::NDArrayNumericTensor;
use crate::DynRank;

use super::types::*;

/// Pre-compiled execution plan: all spans JIT-compiled, ready to execute.
pub struct CompiledPlan {
    phases: Vec<CompiledPhase>,
}

struct CompiledPhase {
    spans: Vec<CompiledSpanEntry>,
}

struct CompiledSpanEntry {
    layout: BufferLayout,
    compiled: CompiledSpan,
    graph: NanoGraph,
    inputs: Vec<AtomRange>,
    outputs: Vec<AtomRange>,
    /// Literal-populated buffer template. Cloned per execution to avoid
    /// re-populating literals each time.
    literal_buffer: Vec<u8>,
}

impl CompiledPlan {
    /// Compile all spans in an execution plan.
    pub fn compile(plan: &ExecutionPlan) -> Result<Self, String> {
        let mut phases = Vec::with_capacity(plan.phases.len());
        for (pi, phase) in plan.phases.iter().enumerate() {
            let mut compiled_spans = Vec::with_capacity(phase.spans.len());
            for (si, span) in phase.spans.iter().enumerate() {
                if span.graph.num_groups() == 0 {
                    // Empty span: no-op.
                    compiled_spans.push(CompiledSpanEntry {
                        layout: BufferLayout {
                            slots: vec![],
                            total_bytes: 0,
                            group_use_counts: vec![],
                        },
                        compiled: compile_empty_span()?,
                        graph: span.graph.clone(),
                        inputs: span.inputs.clone(),
                        outputs: span.outputs.clone(),
                        literal_buffer: vec![],
                    });
                    continue;
                }

                let layout = compute_layout(&span.graph, &span.outputs);
                let compiled = compile_span(&span.graph, &layout).map_err(|e| {
                    format!("phase {} span {}: {}", pi, si, e)
                })?;

                // Pre-populate literals into a template buffer.
                let mut lit_buf = vec![0u8; layout.total_bytes];
                layout.populate_literals(&span.graph, &mut lit_buf);

                compiled_spans.push(CompiledSpanEntry {
                    layout,
                    compiled,
                    graph: span.graph.clone(),
                    inputs: span.inputs.clone(),
                    outputs: span.outputs.clone(),
                    literal_buffer: lit_buf,
                });
            }
            phases.push(CompiledPhase {
                spans: compiled_spans,
            });
        }
        Ok(CompiledPlan { phases })
    }

    /// Execute the compiled plan.
    ///
    /// `inputs` provides all external data (weights + user inputs) as tensors
    /// keyed by base AtomId. Returns the value store after all phases.
    pub fn execute(
        &self,
        inputs: Vec<(AtomId, NDArrayNumericTensor<DynRank>)>,
    ) -> HashMap<AtomId, NDArrayNumericTensor<DynRank>> {
        use rayon::prelude::*;

        let mut store: HashMap<AtomId, NDArrayNumericTensor<DynRank>> = HashMap::new();
        for (base, tensor) in inputs {
            store.insert(base, tensor);
        }

        for phase in &self.phases {
            let phase_outputs: Vec<Vec<(AtomId, NDArrayNumericTensor<DynRank>)>> = phase
                .spans
                .par_iter()
                .map(|entry| execute_compiled_span(entry, &store))
                .collect();

            for span_outputs in phase_outputs {
                for (base, tensor) in span_outputs {
                    store.insert(base, tensor);
                }
            }
        }

        store
    }

    /// Execute with per-phase diagnostic output (NaN detection + eval comparison
    /// for first divergent span).
    pub fn execute_with_diagnostics(
        &self,
        inputs: Vec<(AtomId, NDArrayNumericTensor<DynRank>)>,
    ) -> HashMap<AtomId, NDArrayNumericTensor<DynRank>> {
        use crate::nano_graph::eval;
        use rayon::prelude::*;

        let mut store: HashMap<AtomId, NDArrayNumericTensor<DynRank>> = HashMap::new();
        for (base, tensor) in inputs {
            store.insert(base, tensor);
        }

        let mut found_divergence = false;

        for (pi, phase) in self.phases.iter().enumerate() {
            let phase_outputs: Vec<Vec<(AtomId, NDArrayNumericTensor<DynRank>)>> = phase
                .spans
                .par_iter()
                .map(|entry| execute_compiled_span(entry, &store))
                .collect();

            // Check each span's outputs for NaN.
            for (si, span_out) in phase_outputs.iter().enumerate() {
                let mut span_nans = 0u64;
                for (_, tensor) in span_out {
                    if let NDArrayNumericTensor::F32(arr) = tensor {
                        span_nans += arr.iter().filter(|v| v.is_nan()).count() as u64;
                    }
                }
                if span_nans > 0 && !found_divergence {
                    found_divergence = true;
                    let entry = &phase.spans[si];

                    // Re-run JIT to get the buffer for inspection.
                    let mut dbg_buf = entry.literal_buffer.clone();
                    populate_buffer_from_store(&entry.inputs, &store, &entry.layout, &mut dbg_buf);
                    entry.compiled.execute(&mut dbg_buf);

                    // Scan span OUTPUT atoms for the first NaN.
                    let mut first_nan_group: Option<(&AtomGroup, u64)> = None;
                    for out_range in &entry.outputs {
                        let data = entry.layout.read_f32_output(out_range, &dbg_buf);
                        for (i, &v) in data.iter().enumerate() {
                            if v.is_nan() {
                                let atom = AtomId(out_range.base.0 + i as u64);
                                if let Some(group) = entry.graph.group_of(atom) {
                                    first_nan_group = Some((group, atom.0 - group.base_id.0));
                                }
                                break;
                            }
                        }
                        if first_nan_group.is_some() { break; }
                    }
                    if let Some((g, elem)) = first_nan_group {
                        let op_detail = match &g.op {
                            ScalarOp::Binary { op, compute_dtype } => format!("Binary({:?}, {:?})", op, compute_dtype),
                            ScalarOp::Unary { op, compute_dtype } => format!("Unary({:?}, {:?})", op, compute_dtype),
                            other => format!("{:?}", op_name_short(other)),
                        };
                        eprintln!(
                            "  ROOT NaN: group base={} op={} count={} atom_offset={} elem={} output_dtype={:?}",
                            g.base_id, op_detail, g.count, g.atom_offset, elem, g.output_dtype
                        );
                        for (ii, ir) in g.inputs.iter().enumerate() {
                            eprintln!("    input[{}]: {:?}", ii, ir);
                        }
                        if let ScalarOp::Reduce { reduce_count, reduce_stride, kind, .. } = &g.op {
                            eprintln!("    reduce: kind={:?} count={} stride={}", kind, reduce_count, reduce_stride);
                        }
                        // Read a window around the NaN element.
                        if let Some((slot, _)) = entry.layout.find(g.base_id) {
                            let start = if elem > 3 { elem - 3 } else { 0 };
                            let end = (elem + 5).min(g.count);
                            let mut window = Vec::new();
                            for j in start..end {
                                let off = slot.byte_offset + j as usize * slot.elem_bytes;
                                if off + 4 <= dbg_buf.len() {
                                    let v = f32::from_le_bytes(dbg_buf[off..off+4].try_into().unwrap());
                                    window.push((j, v));
                                }
                            }
                            eprintln!("    output window: {:?}", window);
                        }
                        // Read the input values for this group's NaN element.
                        for (ii, ir) in g.inputs.iter().enumerate() {
                            let src_atom = ir.resolve(elem + g.atom_offset);
                            let src_val = entry.layout.byte_offset_of(src_atom)
                                .map(|off| read_f32_from(&dbg_buf, off, DType::F32));
                            let src_owner = entry.graph.group_of(src_atom)
                                .map(|sg| format!("group base={} op={:?}", sg.base_id, op_name_short(&sg.op)))
                                .or_else(|| entry.graph.find_input_idx(src_atom).map(|(idx, _)| format!("input[{}]", idx)));
                            eprintln!("    input[{}] → atom {} val={:?} owner={:?}", ii, src_atom, src_val, src_owner);
                        }
                    }

                    // Find NaN outputs and trace source atoms.
                    for out_range in &entry.outputs {
                        let data = entry.layout.read_f32_output(out_range, &dbg_buf);
                        for (i, &v) in data.iter().enumerate() {
                            if v.is_nan() {
                                let atom = AtomId(out_range.base.0 + i as u64);
                                if let Some(group) = entry.graph.group_of(atom) {
                                    // For Explicit InputRef, trace each source atom.
                                    if let Some(InputRef::Explicit(ids)) = group.inputs.first() {
                                        let elem = (atom.0 - group.base_id.0) as usize;
                                        if elem < ids.len() {
                                            let src_atom = ids[elem];
                                            let src_val = entry.layout.byte_offset_of(src_atom)
                                                .map(|off| read_f32_from(&dbg_buf, off, DType::F32));
                                            // Also check what group produces the source atom.
                                            let src_group = entry.graph.group_of(src_atom);
                                            let src_input = entry.graph.find_input_idx(src_atom);
                                            let src_info = src_group.map(|g| format!(
                                                "group base={} op={:?} count={}",
                                                g.base_id, op_name_short(&g.op), g.count
                                            )).or_else(|| src_input.map(|(idx, off)| {
                                                let it = &entry.graph.input_tensors()[idx];
                                                format!("input_tensor idx={} base={} count={} off={}", idx, it.base_id, it.count, off)
                                            }));
                                            // Also check if the byte offset makes sense.
                                            let byte_off = entry.layout.byte_offset_of(src_atom);
                                            eprintln!(
                                                "  NaN trace: output atom {} elem {} ← src atom {} val={:?} byte_off={:?} src={:?}",
                                                atom, elem, src_atom, src_val, byte_off, src_info
                                            );
                                        }
                                    }
                                    // For other ops, trace the group details and its first source values.
                                    else {
                                        // For Reduce, show parameters and first few source values.
                                        if let ScalarOp::Reduce { kind, reduce_count, reduce_stride, .. } = &group.op {
                                            if let Some(InputRef::Affine { base, stride }) = group.inputs.first() {
                                                let elem = (atom.0 - group.base_id.0) as u64;
                                                let src_base = (base.0 as i64 + *stride * (elem + group.atom_offset) as i64) as u64;
                                                // Read first few k values from buffer.
                                                let mut k_vals = Vec::new();
                                                for k in 0..(*reduce_count).min(8) {
                                                    let src_atom = (src_base as i64 + k as i64 * reduce_stride) as u64;
                                                    let val = entry.layout.byte_offset_of(AtomId(src_atom))
                                                        .map(|off| read_f32_from(&dbg_buf, off, DType::F32));
                                                    k_vals.push((src_atom, val));
                                                }
                                                eprintln!(
                                                    "  NaN Reduce: atom {} base={} op={:?} count={} rc={} rs={} input=Affine(base={},stride={})",
                                                    atom, group.base_id, kind, group.count, reduce_count, reduce_stride, base, stride,
                                                );
                                                eprintln!("    first k values: {:?}", k_vals);
                                                break; // Only trace once per NaN group.
                                            }
                                        } else {
                                            eprintln!(
                                                "  NaN at atom {} in group base={} op={:?}",
                                                atom, group.base_id, op_name_short(&group.op)
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Also run eval for comparison.
                    let interp_inputs = super::execute::gather_inputs_pub(
                        &entry.inputs, &entry.outputs, &store,
                    );
                    let refs: Vec<_> = interp_inputs.iter().map(|(b, t)| (*b, t)).collect();
                    let eval_out = eval::eval(&entry.graph, &refs, &entry.outputs);
                    let eval_nans: u64 = eval_out.iter().map(|t| {
                        if let NDArrayNumericTensor::F32(a) = t {
                            a.iter().filter(|v| v.is_nan()).count() as u64
                        } else { 0 }
                    }).sum();

                    eprintln!(
                        "  [phase {} span {}] JIT nans={}, eval nans={}, groups={}",
                        pi, si, span_nans, eval_nans, entry.graph.num_groups(),
                    );
                }
            }

            for span_outputs in phase_outputs {
                for (base, tensor) in span_outputs {
                    store.insert(base, tensor);
                }
            }
        }

        store
    }
}

/// Serial per-group JIT vs eval comparison for a compiled plan.
///
/// Runs each phase sequentially. Within each phase, runs each span's JIT
/// and eval independently (same store inputs), then compares every group's
/// output. Stops and reports at the first diverging group.
pub fn diagnose_first_divergence(plan: &CompiledPlan, exec_plan: &ExecutionPlan, inputs: Vec<(AtomId, NDArrayNumericTensor<DynRank>)>) {
    use crate::nano_graph::eval;

    let mut store: HashMap<AtomId, NDArrayNumericTensor<DynRank>> = HashMap::new();
    for (base, tensor) in inputs {
        store.insert(base, tensor);
    }

    for (pi, (compiled_phase, plan_phase)) in plan.phases.iter().zip(exec_plan.phases.iter()).enumerate() {
        let mut phase_outputs: Vec<Vec<(AtomId, NDArrayNumericTensor<DynRank>)>> = Vec::new();

        for (si, (entry, span)) in compiled_phase.spans.iter().zip(plan_phase.spans.iter()).enumerate() {
            if entry.layout.total_bytes == 0 {
                phase_outputs.push(Vec::new());
                continue;
            }

            // Run JIT.
            let mut buffer = entry.literal_buffer.clone();
            populate_buffer_from_store(&entry.inputs, &store, &entry.layout, &mut buffer);

            // Snapshot literal values before JIT for corruption detection.
            let pre_jit_literals: Vec<(usize, f32, AtomId)> = entry.graph.groups().iter()
                .filter(|g| matches!(&g.op, ScalarOp::Literal(_)))
                .filter_map(|g| {
                    let range = AtomRange { base: g.base_id, count: g.count, dtype: g.output_dtype };
                    let vals = entry.layout.read_f32_output(&range, &buffer);
                    let gi = entry.graph.find_group_idx(g.base_id)?;
                    Some((gi, vals[0], g.base_id))
                })
                .collect();

            entry.compiled.execute(&mut buffer);

            // Check if any literal was corrupted by JIT execution.
            for &(gi, pre_val, base) in &pre_jit_literals {
                let group = &entry.graph.groups()[gi];
                let range = AtomRange { base: group.base_id, count: group.count, dtype: group.output_dtype };
                let post_vals = entry.layout.read_f32_output(&range, &buffer);
                if (post_vals[0] - pre_val).abs() > 1e-10 {
                    eprintln!(
                        "  LITERAL CORRUPTED by JIT: phase {} span {} group {} base={} pre={} post={} op={:?}",
                        pi, si, gi, base, pre_val, post_vals[0], group.op
                    );
                    // Show the slot info for this literal.
                    if let Some((slot, _)) = entry.layout.find(base) {
                        let lit_off = slot.byte_offset;
                        eprintln!(
                            "    slot: byte_offset={} count={} elem_bytes={} dtype={:?}",
                            lit_off, slot.count, slot.elem_bytes, slot.dtype
                        );
                        // Show raw bytes at that location.
                        if lit_off + 8 <= buffer.len() {
                            let bytes = &buffer[lit_off..lit_off + 8];
                            eprintln!("    raw bytes: {:02x?}", bytes);
                        }
                        // Find neighbor slots that end at or near this offset.
                        for other_group in entry.graph.groups() {
                            if let Some((os, _)) = entry.layout.find(other_group.base_id) {
                                let end = os.byte_offset + os.count as usize * os.elem_bytes;
                                if end > lit_off && end <= lit_off + 8 && os.atom_base != slot.atom_base {
                                    eprintln!(
                                        "    NEIGHBOR: base={} byte_offset={} count={} elem_bytes={} dtype={:?} end={} op={:?}",
                                        os.atom_base, os.byte_offset, os.count, os.elem_bytes, os.dtype,
                                        end, op_name_short(&other_group.op)
                                    );
                                }
                                // Also check if any slot CONTAINS byte lit_off.
                                if os.byte_offset <= lit_off && end > lit_off && os.atom_base != slot.atom_base {
                                    eprintln!(
                                        "    OVERLAPPING: base={} byte_offset={} count={} elem_bytes={} dtype={:?} end={} op={:?}",
                                        os.atom_base, os.byte_offset, os.count, os.elem_bytes, os.dtype,
                                        end, op_name_short(&other_group.op)
                                    );
                                }
                            }
                        }
                    }
                    break;
                }
            }

            // Run eval with same inputs.
            let interp_inputs = super::execute::gather_inputs_pub(
                &entry.inputs, &entry.outputs, &store,
            );
            let refs: Vec<_> = interp_inputs.iter().map(|(b, t)| (*b, t)).collect();

            // Build per-group output ranges so eval computes every group.
            let group_ranges: Vec<AtomRange> = entry.graph.groups().iter().map(|g| AtomRange {
                base: g.base_id,
                count: g.count,
                dtype: g.output_dtype,
            }).collect();
            let eval_results = eval::eval(&entry.graph, &refs, &group_ranges);

            // Compare group by group (skip dead groups — JIT doesn't emit code for them).
            for (gi, (group, eval_tensor)) in entry.graph.groups().iter().zip(eval_results.iter()).enumerate() {
                if gi < entry.layout.group_use_counts.len() && entry.layout.group_use_counts[gi] == 0 {
                    continue;
                }
                let jit_data = entry.layout.read_f32_output(
                    &AtomRange { base: group.base_id, count: group.count, dtype: group.output_dtype },
                    &buffer,
                );

                let eval_flat = eval_tensor.flatten();
                let n = jit_data.len().min(eval_flat.num_elements());
                let mut max_diff = 0.0f64;
                let mut first_bad: Option<(usize, f32, f64)> = None;
                for j in 0..n {
                    let jv = jit_data[j] as f64;
                    let ev = eval_flat.get(&[j as u64]).unwrap().to_f64();
                    if jv.is_nan() != ev.is_nan() {
                        first_bad.get_or_insert((j, jit_data[j], ev));
                    }
                    if !jv.is_nan() && !ev.is_nan() {
                        let d = (jv - ev).abs();
                        max_diff = max_diff.max(d);
                        if d > 1e-4 && first_bad.is_none() {
                            first_bad = Some((j, jit_data[j], ev));
                        }
                    }
                }

                if let Some((elem, jv, ev)) = first_bad {
                    let op_desc = match &group.op {
                        ScalarOp::Binary { op, compute_dtype } => format!("Binary({:?}, {:?})", op, compute_dtype),
                        ScalarOp::Unary { op, compute_dtype } => format!("Unary({:?}, {:?})", op, compute_dtype),
                        ScalarOp::Reduce { kind, reduce_count, reduce_stride, compute_dtype } =>
                            format!("Reduce({:?}, rc={}, rs={}, {:?})", kind, reduce_count, reduce_stride, compute_dtype),
                        ScalarOp::Identity => "Identity".to_string(),
                        ScalarOp::Select => "Select".to_string(),
                        ScalarOp::IndirectLoad { table_base } => format!("IndirectLoad(table={})", table_base),
                        ScalarOp::Literal(s) => format!("Literal({:?})", s),
                    };
                    eprintln!(
                        "\n  DIVERGENCE: phase {} span {} group {} (of {})",
                        pi, si, gi, entry.graph.num_groups()
                    );
                    eprintln!(
                        "    base={} count={} atom_offset={} output_dtype={:?}",
                        group.base_id, group.count, group.atom_offset, group.output_dtype
                    );
                    eprintln!("    op: {}", op_desc);
                    let is_dead = gi < entry.layout.group_use_counts.len()
                        && entry.layout.group_use_counts[gi] == 0;
                    eprintln!(
                        "    elem={}: jit={} eval={} max_diff={:.6} dead={}",
                        elem, jv, ev, max_diff, is_dead
                    );
                    for (ii, ir) in group.inputs.iter().enumerate() {
                        eprintln!("    input[{}]: {:?}", ii, ir);
                        // Read input values at the diverging element from both JIT buffer and eval.
                        let src_atom = ir.resolve(elem as u64 + group.atom_offset);
                        let jit_src = entry.layout.find(src_atom).map(|(slot, idx)| {
                            let off = slot.byte_offset + idx as usize * slot.elem_bytes;
                            let s = read_scalar(&buffer, off, slot.dtype);
                            format!("{:?} (dtype={:?})", s, slot.dtype)
                        });
                        // Find eval value for same atom.
                        let eval_src = entry.graph.find_group_idx(src_atom).and_then(|sgi| {
                            let sg = &entry.graph.groups()[sgi];
                            let off = (src_atom.0 - sg.base_id.0) as usize;
                            eval_results.get(sgi).map(|t| {
                                let f = t.flatten();
                                if off < f.num_elements() { f.get(&[off as u64]).unwrap().to_f64() }
                                else { f64::NAN }
                            })
                        });
                        let src_dead = entry.graph.find_group_idx(src_atom).map(|sgi| {
                            sgi < entry.layout.group_use_counts.len()
                                && entry.layout.group_use_counts[sgi] == 0
                        });
                        // Check store for this atom.
                        let store_val = {
                            let mut found = None;
                            for (base, tensor) in store.iter() {
                                let t_lo = base.0;
                                let t_hi = t_lo + tensor.num_elements() as u64;
                                if src_atom.0 >= t_lo && src_atom.0 < t_hi {
                                    let off = (src_atom.0 - t_lo) as usize;
                                    let scalars = tensor_slice_to_scalars(&tensor, off, 1);
                                    found = Some(format!("{:?} (tensor dtype={:?}, base={}, n={})",
                                        scalars[0], tensor.dtype(), base, tensor.num_elements()));
                                    break;
                                }
                            }
                            found
                        };
                        eprintln!("      src atom={} jit_buf={:?} eval={:?} src_dead={:?} store={:?}",
                            src_atom, jit_src, eval_src, src_dead, store_val);
                    }

                    // Show a window of JIT vs eval around the diverging element.
                    let start = if elem > 3 { elem - 3 } else { 0 };
                    let end = (elem + 5).min(n);
                    let mut window = Vec::new();
                    for j in start..end {
                        let jv = jit_data[j];
                        let ev = eval_flat.get(&[j as u64]).unwrap().to_f64();
                        window.push((j, jv, ev as f32));
                    }
                    eprintln!("    window (elem, jit, eval): {:?}", window);
                    return;
                }
            }

            // Extract JIT outputs for the store.
            let jit_outputs = extract_outputs(&entry.outputs, &entry.layout, &buffer);
            phase_outputs.push(jit_outputs);
        }

        // Commit phase outputs to store.
        for span_outputs in phase_outputs {
            for (base, tensor) in span_outputs {
                store.insert(base, tensor);
            }
        }
        eprintln!("  phase {}: all {} spans match", pi, compiled_phase.spans.len());
    }
    eprintln!("  All phases match — no divergence found.");
}

/// Execute a single compiled span: populate buffer from store, run JIT, extract outputs.
fn execute_compiled_span(
    entry: &CompiledSpanEntry,
    store: &HashMap<AtomId, NDArrayNumericTensor<DynRank>>,
) -> Vec<(AtomId, NDArrayNumericTensor<DynRank>)> {
    if entry.layout.total_bytes == 0 {
        return Vec::new();
    }

    // Start from the literal-populated template.
    let mut buffer = entry.literal_buffer.clone();

    // Gather inputs from store into buffer.
    populate_buffer_from_store(&entry.inputs, store, &entry.layout, &mut buffer);

    // Run the JIT function.
    entry.compiled.execute(&mut buffer);

    // Extract outputs as NDArray tensors.
    extract_outputs(&entry.outputs, &entry.layout, &buffer)
}

/// Copy data from the store into the buffer for each declared input range.
/// Writes tensor elements in their native dtype, matching the slot's storage format.
fn populate_buffer_from_store(
    input_ranges: &[AtomRange],
    store: &HashMap<AtomId, NDArrayNumericTensor<DynRank>>,
    layout: &BufferLayout,
    buffer: &mut [u8],
) {
    use crate::numeric_scalar::NumericScalar;

    for range in input_ranges {
        let range_lo = range.base.0;
        let range_hi = range_lo + range.count;

        for (&base, tensor) in store {
            let t_lo = base.0;
            let t_hi = t_lo + tensor.num_elements() as u64;
            if t_lo < range_hi && t_hi > range_lo {
                let overlap_start = t_lo.max(range_lo);
                let overlap_end = t_hi.min(range_hi);
                let skip = (overlap_start - t_lo) as usize;
                let count = (overlap_end - overlap_start) as usize;

                let scalars = tensor_slice_to_scalars(tensor, skip, count);
                // Bulk-write within slots: find the slot for the first atom,
                // then write contiguously until we exhaust this slot or the data.
                let mut written = 0usize;
                let mut atom = overlap_start;
                while written < scalars.len() {
                    if let Some((slot, elem_start)) = layout.find(AtomId(atom)) {
                        let available = (slot.count - elem_start) as usize;
                        let to_write = available.min(scalars.len() - written);
                        for i in 0..to_write {
                            let off = slot.byte_offset + (elem_start as usize + i) * slot.elem_bytes;
                            if off + slot.elem_bytes <= buffer.len() {
                                let stored = scalars[written + i].cast_to(slot.dtype);
                                write_scalar(buffer, off, &stored);
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
        }
    }
}

/// Extract elements from a tensor at [skip..skip+count] as NumericScalars.
fn tensor_slice_to_scalars(
    tensor: &NDArrayNumericTensor<DynRank>,
    skip: usize,
    count: usize,
) -> Vec<crate::numeric_scalar::NumericScalar> {
    use crate::numeric_scalar::NumericScalar;
    macro_rules! extract {
        ($arr:expr, $variant:ident) => {{
            $arr.iter()
                .skip(skip)
                .take(count)
                .map(|v| NumericScalar::$variant(*v))
                .collect()
        }};
    }
    match tensor {
        NDArrayNumericTensor::F32(a) => extract!(a, F32),
        NDArrayNumericTensor::F64(a) => extract!(a, F64),
        NDArrayNumericTensor::I64(a) => extract!(a, I64),
        NDArrayNumericTensor::I32(a) => extract!(a, I32),
        NDArrayNumericTensor::U64(a) => extract!(a, U64),
        NDArrayNumericTensor::U32(a) => extract!(a, U32),
        NDArrayNumericTensor::BF16(a) => extract!(a, BF16),
        NDArrayNumericTensor::F16(a) => extract!(a, F16),
        NDArrayNumericTensor::U8(a) => extract!(a, U8),
        NDArrayNumericTensor::I8(a) => extract!(a, I8),
        NDArrayNumericTensor::BOOL(a) => extract!(a, BOOL),
        _ => vec![NumericScalar::F32(0.0); count],
    }
}

/// Extract output ranges from the buffer as NDArray tensors in the correct dtype.
fn extract_outputs(
    output_ranges: &[AtomRange],
    layout: &BufferLayout,
    buffer: &[u8],
) -> Vec<(AtomId, NDArrayNumericTensor<DynRank>)> {
    use crate::numeric_scalar::{NumericScalar, NumericScalarType};

    output_ranges
        .iter()
        .map(|range| {
            // Read scalars from buffer in their native slot dtype.
            let mut scalars = Vec::with_capacity(range.count as usize);
            let mut remaining = range.count;
            let mut atom = range.base.0;
            while remaining > 0 {
                if let Some((slot, elem_start)) = layout.find(AtomId(atom)) {
                    let available = slot.count - elem_start;
                    let to_read = remaining.min(available);
                    for i in 0..to_read {
                        let off = slot.byte_offset + (elem_start + i) as usize * slot.elem_bytes;
                        if off + slot.elem_bytes <= buffer.len() {
                            scalars.push(read_scalar(buffer, off, slot.dtype));
                        } else {
                            scalars.push(NumericScalar::F32(0.0));
                        }
                    }
                    atom += to_read;
                    remaining -= to_read;
                } else {
                    scalars.push(NumericScalar::F32(0.0));
                    atom += 1;
                    remaining -= 1;
                }
            }

            // Build NDArray tensor in the output range's dtype.
            let len = scalars.len();
            let tensor = match range.dtype {
                DType::F32 => {
                    let data: Vec<f32> = scalars.iter().map(|s| s.to_f64() as f32).collect();
                    NDArrayNumericTensor::F32(ArcArray::from_shape_vec(IxDyn(&[len]), data).unwrap())
                }
                DType::I64 => {
                    let data: Vec<i64> = scalars.iter().map(|s| i64::cast_from_numeric_scalar(s)).collect();
                    NDArrayNumericTensor::I64(ArcArray::from_shape_vec(IxDyn(&[len]), data).unwrap())
                }
                DType::BF16 => {
                    let data: Vec<half::bf16> = scalars.iter().map(|s| half::bf16::cast_from_numeric_scalar(s)).collect();
                    NDArrayNumericTensor::BF16(ArcArray::from_shape_vec(IxDyn(&[len]), data).unwrap())
                }
                DType::U8 => {
                    let data: Vec<u8> = scalars.iter().map(|s| u8::cast_from_numeric_scalar(s)).collect();
                    NDArrayNumericTensor::U8(ArcArray::from_shape_vec(IxDyn(&[len]), data).unwrap())
                }
                DType::BOOL => {
                    let data: Vec<bool> = scalars.iter().map(|s| bool::cast_from_numeric_scalar(s)).collect();
                    NDArrayNumericTensor::BOOL(ArcArray::from_shape_vec(IxDyn(&[len]), data).unwrap())
                }
                _ => {
                    // Fallback to f32.
                    let data: Vec<f32> = scalars.iter().map(|s| s.to_f64() as f32).collect();
                    NDArrayNumericTensor::F32(ArcArray::from_shape_vec(IxDyn(&[len]), data).unwrap())
                }
            };
            (range.base, tensor)
        })
        .collect()
}

/// Compile an empty span (no groups). Returns a no-op CompiledSpan.
fn compile_empty_span() -> Result<CompiledSpan, String> {
    let mut flag_builder = settings::builder();
    flag_builder.set("opt_level", "speed").unwrap();
    let isa_builder =
        cranelift_native::builder().map_err(|e| format!("cranelift native ISA: {}", e))?;
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .map_err(|e| format!("ISA finish: {}", e))?;
    let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    register_math_symbols(&mut jit_builder);
    let mut module = JITModule::new(jit_builder);
    let _ = declare_math_funcs(&mut module)?;

    let mut ctx = module.make_context();
    ctx.func.signature.params.push(AbiParam::new(types::I64));
    let func_id = module
        .declare_function("noop", Linkage::Local, &ctx.func.signature)
        .map_err(|e| format!("declare: {}", e))?;
    {
        let mut func_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        builder.seal_block(entry);
        builder.ins().return_(&[]);
        builder.finalize();
    }
    module.define_function(func_id, &mut ctx).map_err(|e| format!("define: {}", e))?;
    module.finalize_definitions().map_err(|e| format!("finalize: {}", e))?;
    let func_ptr = module.get_finalized_function(func_id);
    Ok(CompiledSpan {
        func_ptr,
        _module: module,
    })
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GlobalId;
    use crate::nano_graph::ops::{ReduceKind, ScalarBinOp, ScalarOp, ScalarUnaryOp};
    use crate::numeric_scalar::NumericScalar;

    /// Build a graph: input[4] + broadcast(2.0) → output[4]
    #[test]
    fn test_add_broadcast_f32() {
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), 4, DType::F32);
        let lit = g.push_group(
            1,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
        );
        let add = g.push_group(
            4,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine {
                    base: inp,
                    stride: 1,
                },
                InputRef::Broadcast(lit),
            ],
        );

        let outputs = vec![AtomRange {
            base: add,
            count: 4,
            dtype: DType::F32,
        }];
        let layout = compute_layout(&g, &outputs);
        let compiled = compile_span(&g, &layout).unwrap();

        let mut buffer = vec![0u8; layout.total_bytes];
        layout.populate_literals(&g, &mut buffer);
        layout.write_f32_input(inp, &[1.0, 2.0, 3.0, 4.0], &mut buffer);

        compiled.execute(&mut buffer);

        let result = layout.read_f32_output(&outputs[0], &buffer);
        assert_eq!(result, vec![3.0, 4.0, 5.0, 6.0]);
    }

    /// Unary neg: -input[3]
    #[test]
    fn test_unary_neg() {
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), 3, DType::F32);
        let neg = g.push_group(
            3,
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

        let outputs = vec![AtomRange {
            base: neg,
            count: 3,
            dtype: DType::F32,
        }];
        let layout = compute_layout(&g, &outputs);
        let compiled = compile_span(&g, &layout).unwrap();

        let mut buffer = vec![0u8; layout.total_bytes];
        layout.write_f32_input(inp, &[1.0, -2.5, 3.0], &mut buffer);

        compiled.execute(&mut buffer);

        let result = layout.read_f32_output(&outputs[0], &buffer);
        assert_eq!(result, vec![-1.0, 2.5, -3.0]);
    }

    /// Reduce sum: sum of 4 elements.
    #[test]
    fn test_reduce_sum() {
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), 4, DType::F32);
        let sum = g.push_group(
            1,
            DType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: 4,
                reduce_stride: 1,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: inp,
                stride: 1,
            }],
        );

        let outputs = vec![AtomRange {
            base: sum,
            count: 1,
            dtype: DType::F32,
        }];
        let layout = compute_layout(&g, &outputs);
        let compiled = compile_span(&g, &layout).unwrap();

        let mut buffer = vec![0u8; layout.total_bytes];
        layout.write_f32_input(inp, &[1.0, 2.0, 3.0, 4.0], &mut buffer);

        compiled.execute(&mut buffer);

        let result = layout.read_f32_output(&outputs[0], &buffer);
        assert_eq!(result, vec![10.0]);
    }

    /// Chain: input → mul(3.0) → exp → output. Tests multi-group pipelines.
    #[test]
    fn test_chain_mul_exp() {
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), 2, DType::F32);
        let lit3 = g.push_group(
            1,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(3.0)),
            vec![],
            vec![],
        );
        let mul = g.push_group(
            2,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine {
                    base: inp,
                    stride: 1,
                },
                InputRef::Broadcast(lit3),
            ],
        );
        let exp = g.push_group(
            2,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: mul,
                stride: 1,
            }],
        );

        let outputs = vec![AtomRange {
            base: exp,
            count: 2,
            dtype: DType::F32,
        }];
        let layout = compute_layout(&g, &outputs);
        let compiled = compile_span(&g, &layout).unwrap();

        let mut buffer = vec![0u8; layout.total_bytes];
        layout.populate_literals(&g, &mut buffer);
        layout.write_f32_input(inp, &[0.0, 1.0], &mut buffer);

        compiled.execute(&mut buffer);

        let result = layout.read_f32_output(&outputs[0], &buffer);
        let expected = vec![(0.0f32 * 3.0).exp(), (1.0f32 * 3.0).exp()];
        for (a, b) in result.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "got {}, expected {}", a, b);
        }
    }

    /// Liveness reuse: chain A → B → C where A's slot can be reused by C.
    #[test]
    fn test_liveness_reuse() {
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), 100, DType::F32);
        let a = g.push_group(
            100,
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
        let b = g.push_group(
            100,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: a,
                stride: 1,
            }],
        );
        let c = g.push_group(
            100,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine {
                base: b,
                stride: 1,
            }],
        );

        let outputs = vec![AtomRange {
            base: c,
            count: 100,
            dtype: DType::F32,
        }];
        let layout = compute_layout(&g, &outputs);

        // Without reuse: input(400) + A(400) + B(400) + C(400) = 1600 bytes.
        // With reuse: A freed after B allocated, C reuses A's space = 1200 bytes.
        // (Neg has stride=1 Affine, but each references only one source group,
        // so no slab constraint — liveness reuse is active.)
        assert!(
            layout.total_bytes <= 1200,
            "expected liveness reuse, got {} bytes",
            layout.total_bytes
        );

        // Verify correctness: neg(neg(neg(x))) = -x
        let compiled = compile_span(&g, &layout).unwrap();
        let mut buffer = vec![0u8; layout.total_bytes];
        let input_data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        layout.write_f32_input(inp, &input_data, &mut buffer);

        compiled.execute(&mut buffer);

        let result = layout.read_f32_output(&outputs[0], &buffer);
        for (i, &val) in result.iter().enumerate() {
            let expected = -(i as f32);
            assert!(
                (val - expected).abs() < 1e-6,
                "elem {}: got {}, expected {}",
                i,
                val,
                expected
            );
        }
    }

    /// Select (ternary): condition ? x : y
    #[test]
    fn test_select() {
        let mut g = NanoGraph::new();
        let cond = g.add_input_tensor(GlobalId(0), 3, DType::F32);
        let x = g.add_input_tensor(GlobalId(1), 3, DType::F32);
        let y = g.add_input_tensor(GlobalId(2), 3, DType::F32);
        let sel = g.push_group(
            3,
            DType::F32,
            ScalarOp::Select,
            vec![],
            vec![
                InputRef::Affine {
                    base: cond,
                    stride: 1,
                },
                InputRef::Affine {
                    base: x,
                    stride: 1,
                },
                InputRef::Affine {
                    base: y,
                    stride: 1,
                },
            ],
        );

        let outputs = vec![AtomRange {
            base: sel,
            count: 3,
            dtype: DType::F32,
        }];
        let layout = compute_layout(&g, &outputs);
        let compiled = compile_span(&g, &layout).unwrap();

        let mut buffer = vec![0u8; layout.total_bytes];
        layout.write_f32_input(cond, &[1.0, 0.0, 5.0], &mut buffer);
        layout.write_f32_input(x, &[10.0, 20.0, 30.0], &mut buffer);
        layout.write_f32_input(y, &[100.0, 200.0, 300.0], &mut buffer);

        compiled.execute(&mut buffer);

        let result = layout.read_f32_output(&outputs[0], &buffer);
        // cond[0]=1.0 (nonzero) → x=10.0, cond[1]=0.0 → y=200.0, cond[2]=5.0 → x=30.0
        assert_eq!(result, vec![10.0, 200.0, 30.0]);
    }

    /// Identity with Explicit InputRef gathering from 4 source groups.
    /// Reproduces the phase-0 NaN bug pattern.
    #[test]
    fn test_explicit_gather() {
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), 16, DType::F32);
        // 4 source groups, each 4 atoms, that read from different input regions.
        let mut sources = Vec::new();
        for i in 0..4u64 {
            let src = g.push_group(
                4,
                DType::F32,
                ScalarOp::Identity,
                vec![],
                vec![InputRef::Affine {
                    base: AtomId(inp.0 + i * 4),
                    stride: 1,
                }],
            );
            sources.push(src);
        }
        // Gather: pick one atom from each of the 4 source groups, interleaved.
        let explicit_ids: Vec<AtomId> = (0..16)
            .map(|i| {
                let group = i / 4;
                let elem = i % 4;
                AtomId(sources[group].0 + elem as u64)
            })
            .collect();
        let gather = g.push_group(
            16,
            DType::F32,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::Explicit(explicit_ids)],
        );

        let outputs = vec![AtomRange {
            base: gather,
            count: 16,
            dtype: DType::F32,
        }];
        let layout = compute_layout(&g, &outputs);
        let compiled = compile_span(&g, &layout).unwrap();

        let mut buffer = vec![0u8; layout.total_bytes];
        let input_data: Vec<f32> = (0..16).map(|i| (i + 1) as f32).collect();
        layout.write_f32_input(inp, &input_data, &mut buffer);
        layout.populate_literals(&g, &mut buffer);

        compiled.execute(&mut buffer);

        let result = layout.read_f32_output(&outputs[0], &buffer);
        // Sources copy input directly. Gather picks elements from sources.
        // Source 0 = [1,2,3,4], Source 1 = [5,6,7,8], etc.
        // Gather picks: src0[0..3], src1[0..3], src2[0..3], src3[0..3]
        let expected: Vec<f32> = (0..16).map(|i| (i + 1) as f32).collect();
        assert_eq!(result, expected, "Explicit gather mismatch");
    }

    /// Cross-group Affine: identity reads across two source groups.
    /// This is the pattern that caused the contiguity bug.
    #[test]
    fn test_cross_group_affine() {
        let mut g = NanoGraph::new();
        // Two source groups: A (3 atoms) then B (3 atoms), contiguous in atom space.
        let a = g.push_group(
            3,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(1.0)),
            vec![],
            vec![],
        );
        let b = g.push_group(
            3,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(2.0)),
            vec![],
            vec![],
        );
        // Consumer reads 6 atoms starting from A's base, crossing into B.
        let out = g.push_group(
            6,
            DType::F32,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::Affine {
                base: a,
                stride: 1,
            }],
        );

        let outputs = vec![AtomRange {
            base: out,
            count: 6,
            dtype: DType::F32,
        }];
        let layout = compute_layout(&g, &outputs);

        // Validate no cross-slot errors.
        let errors = validate_layout(&g, &layout);
        assert!(errors.is_empty(), "layout errors: {:?}", errors);

        let compiled = compile_span(&g, &layout).unwrap();
        let mut buffer = vec![0u8; layout.total_bytes];
        layout.populate_literals(&g, &mut buffer);
        compiled.execute(&mut buffer);

        let result = layout.read_f32_output(&outputs[0], &buffer);
        assert_eq!(result, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    }
}
