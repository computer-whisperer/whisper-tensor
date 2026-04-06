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

use crate::nano_graph::{
    AtomGroup, AtomId, AtomRange, InputRef, NanoGraph, ReduceKind, ScalarBinOp, ScalarOp,
    ScalarUnaryOp,
};
use crate::numeric_dtype::NumericDType;

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
    pub dtype: NumericDType,
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
                            let off = slot.byte_offset + i as usize * slot.elem_bytes;
                            write_scalar(buffer, off, &stored);
                        }
                    }
                }
                ScalarOp::LiteralSpan(tensor) => {
                    if let Some((slot, _)) = self.find(group.base_id) {
                        for i in 0..group.count {
                            let scalar = tensor.read_element(i as usize);
                            let stored = scalar.cast_to(slot.dtype);
                            let off = slot.byte_offset + i as usize * slot.elem_bytes;
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
                    let off = slot.byte_offset + (elem_start as usize + i) * slot.elem_bytes;
                    if off + slot.elem_bytes <= buffer.len() {
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
                // Sample all corner positions to find the bounding range.
                // For the 2D case: inner = i % dim_shape[1], outer = i / dim_shape[1]
                // offset = dim_strides[0] * outer + dim_strides[1] * inner
                // For 1D (affine): offset = dim_strides[0] * i
                let mut lo = i64::MAX;
                let mut hi = i64::MIN;
                let first_i = atom_offset;
                let last_i = atom_offset + count - 1;
                let nd = dim_strides.len();
                for &i in &[first_i, last_i] {
                    let off = if nd == 1 {
                        dim_strides[0] * i as i64
                    } else {
                        let inner = i % dim_shape[1];
                        let outer = i / dim_shape[1];
                        dim_strides[1] * inner as i64 + dim_strides[0] * outer as i64
                    };
                    lo = lo.min(off);
                    hi = hi.max(off);
                    // Also check boundary: when inner wraps, the offset can jump.
                    if nd >= 2 && dim_shape[1] != u64::MAX && i > 0 {
                        let inner2 = (dim_shape[1] - 1) % dim_shape[1];
                        let outer2 = (dim_shape[1] - 1) / dim_shape[1];
                        let off2 = dim_strides[1] * inner2 as i64 + dim_strides[0] * outer2 as i64;
                        lo = lo.min(off2);
                        hi = hi.max(off2);
                    }
                }
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
    for (gi, group) in groups.iter().enumerate() {
        let mut producers = HashSet::new();
        graph.collect_all_producer_indices(group, gi, &mut producers);
        for &pi in &producers {
            use_counts[pi] += 1;
        }
        producer_lists.push(producers.into_iter().collect());
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
                    gi,
                    group.base_id,
                    s.byte_offset,
                    end,
                    s.elem_bytes,
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

// ─── JIT backend for executor ────────────────────────────────────────────────

use super::executor::{CompiledSpanFn, SpanOutput, StoreSlice};

/// JIT-compiled span implementing the executor's `CompiledSpanFn` trait.
///
/// Bridges between the executor's SpanOutput/StoreSlice interface and the
/// JIT's flat byte buffer model. Owns the compiled native function, the
/// buffer layout, and a pre-populated literal template.
pub struct JitCompiledSpan {
    compiled: CompiledSpan,
    layout: BufferLayout,
    literal_template: Vec<u8>,
    output_ranges: Vec<AtomRange>,
}

impl JitCompiledSpan {
    /// Compile a span into a JIT function ready for the executor.
    pub fn compile(
        graph: &NanoGraph<'static, crate::pool::SystemPool>,
        output_ranges: &[AtomRange],
    ) -> Result<Self, String> {
        if graph.num_groups() == 0 {
            return Ok(JitCompiledSpan {
                compiled: compile_empty_span()?,
                layout: BufferLayout {
                    slots: vec![],
                    total_bytes: 0,
                    group_use_counts: vec![],
                },
                literal_template: vec![],
                output_ranges: output_ranges.to_vec(),
            });
        }

        let layout = compute_layout(graph, output_ranges);
        let (compiled, embedded_tables) = if std::env::var("FUSION_VALIDATE").is_ok() {
            compile_span_validated(graph, &layout)?
        } else {
            compile_span(graph, &layout)?
        };

        // Build literal template including embedded lookup tables.
        let total_buf_bytes = embedded_tables.total_bytes().max(layout.total_bytes);
        let mut literal_template = vec![0u8; total_buf_bytes];
        layout.populate_literals(graph, &mut literal_template);
        embedded_tables.populate(&mut literal_template);

        Ok(JitCompiledSpan {
            compiled,
            layout,
            literal_template,
            output_ranges: output_ranges.to_vec(),
        })
    }
}

impl CompiledSpanFn for JitCompiledSpan {
    fn execute(&self, inputs: &[StoreSlice<'_>], outputs: &mut [SpanOutput<'_>]) {
        if self.layout.total_bytes == 0 {
            return;
        }

        // Clone literal template as working buffer.
        let mut buffer = self.literal_template.clone();

        // Populate inputs from store slices into buffer slots.
        for slice in inputs {
            write_store_slice_to_buffer(slice, &self.layout, &mut buffer);
        }

        // Run the JIT function.
        self.compiled.execute(&mut buffer);

        // Extract outputs from buffer into SpanOutputs.
        for (range, out) in self.output_ranges.iter().zip(outputs.iter_mut()) {
            read_buffer_to_output(range, &self.layout, &buffer, out);
        }
    }
}

/// Write a StoreSlice into the buffer at the correct slot positions.
fn write_store_slice_to_buffer(slice: &StoreSlice<'_>, layout: &BufferLayout, buffer: &mut [u8]) {
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
            let dst_start = slot.byte_offset + elem_start as usize * slot.elem_bytes;
            let dst_end = dst_start + to_write * slot.elem_bytes;
            if dst_end <= buffer.len() {
                buffer[dst_start..dst_end].copy_from_slice(&slice.data[src_start..src_end]);
            }
        } else if src_end <= slice.data.len() {
            // Slow path: per-element with dtype conversion.
            for i in 0..to_write {
                let src_off = (written + i) * elem_bytes;
                let dst_off = slot.byte_offset + (elem_start as usize + i) * slot.elem_bytes;
                if src_off + elem_bytes <= slice.data.len()
                    && dst_off + slot.elem_bytes <= buffer.len()
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
fn read_buffer_to_output(
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
            let src_start = slot.byte_offset + elem_start as usize * slot.elem_bytes;
            let src_end = src_start + to_read * slot.elem_bytes;
            let dst_start = read * elem_bytes;
            let dst_end = dst_start + to_read * elem_bytes;
            if src_end <= buffer.len() && dst_end <= out.data.len() {
                out.data[dst_start..dst_end].copy_from_slice(&buffer[src_start..src_end]);
            }
        } else {
            // Slow path: per-element dtype conversion.
            for i in 0..to_read {
                let src_off = slot.byte_offset + (elem_start as usize + i) * slot.elem_bytes;
                let dst_off = (read + i) * elem_bytes;
                if src_off + slot.elem_bytes <= buffer.len()
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

// ─── Span compilation ───────────────────────────────────────────────────────

/// Validate that all stride-based InputRefs access atoms within single slots.
/// Returns a list of errors (empty = ok).
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
                } => {
                    // Use the innermost (last) stride for validation
                    let stride = dim_strides.last().copied().unwrap_or(0);
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
                                && slot_lo.elem_bytes == slot_hi.elem_bytes
                            {
                                let atom_delta =
                                    slot_hi.atom_base.0 as i64 - slot_lo.atom_base.0 as i64;
                                let byte_delta =
                                    slot_hi.byte_offset as i64 - slot_lo.byte_offset as i64;
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
                                && slot_lo.elem_bytes == slot_hi.elem_bytes
                            {
                                let atom_delta =
                                    slot_hi.atom_base.0 as i64 - slot_lo.atom_base.0 as i64;
                                let byte_delta =
                                    slot_hi.byte_offset as i64 - slot_lo.byte_offset as i64;
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

// ─── Elementwise loop fusion ────────────────────────────────────────────────

/// A chain of consecutive groups that share a single loop.
struct FusionChain {
    /// Indices into graph.groups().
    group_indices: Vec<usize>,
    count: u64,
    atom_offset: u64,
}

/// Build chains of consecutive fusable groups.
///
/// Two consecutive groups A and B fuse if they have the same count/atom_offset,
/// B is not a Reduce/IndirectLoad/Literal, B is not dead, and all of B's inputs
/// that reference A use Affine stride=1 (Strided with dim_strides=[1],
/// dim_shape=[MAX]).
fn build_fusion_chains(
    groups: &[AtomGroup<'static, crate::pool::SystemPool>],
    layout: &BufferLayout,
) -> Vec<FusionChain> {
    // Fusion is opt-out. Disable with FUSION=0.
    if std::env::var("FUSION").as_deref() == Ok("0") {
        return groups
            .iter()
            .enumerate()
            .filter(|(gi, g)| {
                !matches!(&g.op, ScalarOp::Literal(_) | ScalarOp::LiteralSpan(_))
                    && !(*gi < layout.group_use_counts.len() && layout.group_use_counts[*gi] == 0)
                    && g.count > 0
            })
            .map(|(gi, g)| FusionChain {
                group_indices: vec![gi],
                count: g.count,
                atom_offset: g.atom_offset,
            })
            .collect();
    }
    let mut chains: Vec<FusionChain> = Vec::new();

    // Track atom ranges of groups in the current chain for fusion checks.
    // A group can only fuse if none of its inputs overlap a chain member's
    // range at a non-aligned offset (which would cause read-before-write).
    let mut chain_ranges: Vec<(u64, u64)> = Vec::new(); // (base_id, base_id + count)

    for (gi, group) in groups.iter().enumerate() {
        // Skip literals and dead groups — they don't participate in chains.
        if matches!(&group.op, ScalarOp::Literal(_) | ScalarOp::LiteralSpan(_)) {
            continue;
        }
        if gi < layout.group_use_counts.len() && layout.group_use_counts[gi] == 0 {
            continue;
        }

        // Groups that always start a new chain.
        let must_break = matches!(
            &group.op,
            ScalarOp::Reduce { .. } | ScalarOp::IndirectLoad { .. }
        );

        let can_fuse = if must_break {
            false
        } else if let Some(prev_chain) = chains.last() {
            // Check count/atom_offset match with the current chain.
            prev_chain.count == group.count
                && prev_chain.atom_offset == group.atom_offset
                && group.count > 1  // No point fusing single-element groups.
                && inputs_fusable_with_chain(group, &chain_ranges)
        } else {
            false
        };

        if can_fuse {
            // Extend the current chain.
            chain_ranges.push((group.base_id.0, group.base_id.0 + group.count));
            chains.last_mut().unwrap().group_indices.push(gi);
        } else {
            // Start a new chain.
            chain_ranges.clear();
            chain_ranges.push((group.base_id.0, group.base_id.0 + group.count));
            chains.push(FusionChain {
                group_indices: vec![gi],
                count: group.count,
                atom_offset: group.atom_offset,
            });
        }
    }

    chains
}

/// Check if all of a group's inputs that reference a chain producer use
/// Affine stride=1 (eligible for register forwarding).
fn inputs_fusable_with_chain(
    group: &AtomGroup<'static, crate::pool::SystemPool>,
    chain_ranges: &[(u64, u64)],
) -> bool {
    for input in &group.inputs {
        match input {
            InputRef::Strided {
                base,
                dim_strides,
                dim_shape,
            } => {
                // Check if this input's base falls within ANY chain member's range.
                let is_affine_1 =
                    dim_strides.len() == 1 && dim_strides[0] == 1 && dim_shape[0] == u64::MAX;
                for &(range_lo, range_hi) in chain_ranges {
                    if base.0 >= range_lo && base.0 < range_hi {
                        // This input overlaps a chain producer's range.
                        // Fusion is only safe if it reads the SAME iteration's atom:
                        // - Affine stride=1 with base == producer's base_id (exact alignment)
                        if !is_affine_1 || base.0 != range_lo {
                            return false;
                        }
                    }
                }
            }
            InputRef::Broadcast(id) => {
                // Broadcast reads a single atom. If it's in a chain member's range,
                // it reads a fixed position — only safe if that atom has been written
                // before this iteration. In a fused loop, we can't guarantee that.
                for &(range_lo, range_hi) in chain_ranges {
                    if id.0 >= range_lo && id.0 < range_hi {
                        return false;
                    }
                }
            }
            InputRef::Explicit(ids) => {
                // Explicit inputs reference arbitrary atoms. If ANY of them
                // fall within a chain member's range, fusion is unsafe because
                // the access pattern is non-sequential (could read atoms from
                // future iterations that haven't been written yet).
                for id in ids {
                    for &(range_lo, range_hi) in chain_ranges {
                        if id.0 >= range_lo && id.0 < range_hi {
                            return false;
                        }
                    }
                }
            }
        }
    }
    true
}

/// Emit a fusion chain. Single-group chains delegate to `emit_group`.
/// Multi-group chains emit a single loop with forwarded register values.
fn emit_chain(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    chain: &FusionChain,
    groups: &[AtomGroup<'static, crate::pool::SystemPool>],
    layout: &BufferLayout,
    buffer_ptr: Value,
    math: &MathFuncs,
    var_counter: &mut VarCounter,
    tables: &mut EmbeddedTables,
) -> Result<(), String> {
    // Single-group chain: delegate to existing emit_group (no change).
    if chain.group_indices.len() == 1 {
        let gi = chain.group_indices[0];
        return emit_group(
            builder,
            module,
            &groups[gi],
            layout,
            buffer_ptr,
            math,
            var_counter,
            tables,
        );
    }

    let count = chain.count;
    let atom_offset = chain.atom_offset;

    if count == 0 {
        return Ok(());
    }

    // Single-element chain: emit all bodies inline without a loop.
    if count == 1 {
        let mut forwarded: HashMap<u64, (Value, NumericDType)> = HashMap::new();
        for &gi in &chain.group_indices {
            let group = &groups[gi];
            emit_group_body_forwarded(
                builder,
                module,
                group,
                layout,
                buffer_ptr,
                None,
                atom_offset,
                math,
                var_counter,
                tables,
                &mut forwarded,
            )?;
        }
        return Ok(());
    }

    // Multi-element chain: one loop for all groups.
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

    let cmp = builder.ins().icmp(IntCC::SignedLessThan, i_val, end);
    builder.ins().brif(cmp, loop_body, &[], loop_exit, &[]);

    builder.switch_to_block(loop_body);

    // Forwarding map: producer base_id → (Cranelift Value, output NumericDType).
    // Rebuilt each iteration (Cranelift SSA values are block-local within the loop body).
    let mut forwarded: HashMap<u64, (Value, NumericDType)> = HashMap::new();

    for &gi in &chain.group_indices {
        let group = &groups[gi];
        emit_group_body_forwarded(
            builder,
            module,
            group,
            layout,
            buffer_ptr,
            Some(i_val),
            0,
            math,
            var_counter,
            tables,
            &mut forwarded,
        )?;
    }

    let i_next = builder.ins().iadd_imm(i_val, 1);
    builder.ins().jump(loop_header, &[i_next]);

    builder.switch_to_block(loop_exit);
    builder.seal_block(loop_header);
    builder.seal_block(loop_body);
    builder.seal_block(loop_exit);

    Ok(())
}

/// Emit one iteration of a group's computation with register forwarding.
///
/// Like `emit_group_body`, but:
/// - Uses `load_input_forwarded` to check the forwarding map before memory loads.
/// - After computing, registers the output in `forwarded` for downstream groups.
/// - Always stores to the buffer (store elimination is a future optimization).
fn emit_group_body_forwarded(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    group: &AtomGroup<'static, crate::pool::SystemPool>,
    layout: &BufferLayout,
    buffer_ptr: Value,
    i_val: Option<Value>,
    i_const: u64,
    math: &MathFuncs,
    var_counter: &mut VarCounter,
    tables: &mut EmbeddedTables,
    forwarded: &mut HashMap<u64, (Value, NumericDType)>,
) -> Result<(), String> {
    let (out_slot, _) = layout
        .find(group.base_id)
        .ok_or_else(|| format!("no slot for group base={}", group.base_id))?;
    let out_slot = out_slot.clone();

    let output_dtype = group.output_dtype;
    let output_repr = repr_of(output_dtype);

    let result_val = match &group.op {
        ScalarOp::Literal(_) | ScalarOp::LiteralSpan(_) => return Ok(()),

        ScalarOp::Identity | ScalarOp::Cast { .. } => {
            let src = load_input_forwarded(
                builder,
                module,
                &group.inputs[0],
                layout,
                buffer_ptr,
                i_val,
                i_const,
                group.atom_offset,
                tables,
                forwarded,
            )?;
            let src_repr =
                forwarded_or_slot_repr(&group.inputs[0], layout, group.atom_offset, forwarded);
            emit_cast_to_output(builder, src, src_repr, output_dtype)
        }

        ScalarOp::Binary { op, compute_dtype } => {
            let compute_repr = repr_of(*compute_dtype);
            let a_raw = load_input_forwarded(
                builder,
                module,
                &group.inputs[0],
                layout,
                buffer_ptr,
                i_val,
                i_const,
                group.atom_offset,
                tables,
                forwarded,
            )?;
            let a_repr =
                forwarded_or_slot_repr(&group.inputs[0], layout, group.atom_offset, forwarded);
            let a = emit_repr_cast(builder, a_raw, a_repr, compute_repr);

            let b_raw = load_input_forwarded(
                builder,
                module,
                &group.inputs[1],
                layout,
                buffer_ptr,
                i_val,
                i_const,
                group.atom_offset,
                tables,
                forwarded,
            )?;
            let b_repr =
                forwarded_or_slot_repr(&group.inputs[1], layout, group.atom_offset, forwarded);
            let b = emit_repr_cast(builder, b_raw, b_repr, compute_repr);

            let result = emit_binop(builder, module, math, *op, a, b, compute_repr)?;
            emit_cast_to_output(builder, result, compute_repr, output_dtype)
        }

        ScalarOp::Unary { op, compute_dtype } => {
            let compute_repr = repr_of(*compute_dtype);
            let x_raw = load_input_forwarded(
                builder,
                module,
                &group.inputs[0],
                layout,
                buffer_ptr,
                i_val,
                i_const,
                group.atom_offset,
                tables,
                forwarded,
            )?;
            let x_repr =
                forwarded_or_slot_repr(&group.inputs[0], layout, group.atom_offset, forwarded);
            let x = emit_repr_cast(builder, x_raw, x_repr, compute_repr);

            let result = emit_unop(builder, module, math, *op, x, compute_repr)?;
            emit_cast_to_output(builder, result, compute_repr, output_dtype)
        }

        ScalarOp::Select => {
            let cond = load_input_forwarded(
                builder,
                module,
                &group.inputs[0],
                layout,
                buffer_ptr,
                i_val,
                i_const,
                group.atom_offset,
                tables,
                forwarded,
            )?;
            let cond_repr =
                forwarded_or_slot_repr(&group.inputs[0], layout, group.atom_offset, forwarded);

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

            let x_raw = load_input_forwarded(
                builder,
                module,
                &group.inputs[1],
                layout,
                buffer_ptr,
                i_val,
                i_const,
                group.atom_offset,
                tables,
                forwarded,
            )?;
            let x_repr =
                forwarded_or_slot_repr(&group.inputs[1], layout, group.atom_offset, forwarded);
            let x = emit_cast_to_output(builder, x_raw, x_repr, output_dtype);

            let y_raw = load_input_forwarded(
                builder,
                module,
                &group.inputs[2],
                layout,
                buffer_ptr,
                i_val,
                i_const,
                group.atom_offset,
                tables,
                forwarded,
            )?;
            let y_repr =
                forwarded_or_slot_repr(&group.inputs[2], layout, group.atom_offset, forwarded);
            let y = emit_cast_to_output(builder, y_raw, y_repr, output_dtype);

            builder.ins().select(is_nonzero, x, y)
        }

        ScalarOp::Reduce { .. } | ScalarOp::IndirectLoad { .. } => {
            // These should not appear in multi-group chains (build_fusion_chains
            // ensures they always start their own chain). Fall through to
            // emit_group_body for safety.
            return emit_group_body(
                builder,
                module,
                group,
                layout,
                buffer_ptr,
                i_val,
                i_const,
                math,
                var_counter,
                tables,
            );
        }

        ScalarOp::OpaqueOutput { .. } => todo!("opaque ops not supported in compiler"),
    };

    // Always store to buffer (safe approach — avoids needing to track
    // whether any out-of-chain consumer reads this group).
    store_result(
        builder,
        buffer_ptr,
        &out_slot,
        group.atom_offset,
        i_val,
        i_const,
        result_val,
    );

    // Register in forwarding map for downstream groups in the same chain.
    // Apply the store→load round-trip in registers so the forwarded value
    // matches exactly what a memory load would produce. This preserves the
    // dtype truncation contract (e.g., BF16 precision loss between steps).
    let load_repr_val = emit_store_load_roundtrip(builder, result_val, output_dtype);
    forwarded.insert(group.base_id.0, (load_repr_val, output_dtype));

    Ok(())
}

/// Apply the equivalent of store→load in registers, so a forwarded value
/// matches exactly what `emit_typed_store` + `emit_typed_load` would produce.
///
/// This preserves the dtype truncation contract: BF16 outputs must lose
/// precision between steps, I32 values must be sign-extended to I64, etc.
fn emit_store_load_roundtrip(
    builder: &mut FunctionBuilder,
    val: Value,
    dtype: NumericDType,
) -> Value {
    match dtype {
        NumericDType::BF16 => {
            // F32 → BF16 round-to-nearest-even → F32
            // Store path: bitcast f32→i32, round, take top 16 bits
            // Load path: uextend i16→i32, shift left 16, bitcast i32→f32
            let bits = builder.ins().bitcast(types::I32, MemFlags::new(), val);
            let shifted16 = builder.ins().ushr_imm(bits, 16);
            let lsb = builder.ins().band_imm(shifted16, 1);
            let bias = builder.ins().iadd_imm(lsb, 0x7FFF);
            let rounded = builder.ins().iadd(bits, bias);
            // Zero out the bottom 16 bits (equivalent to store i16 + load i16 + shift)
            let masked = builder.ins().band_imm(rounded, !0xFFFF_i64);
            builder.ins().bitcast(types::F32, MemFlags::new(), masked)
        }
        NumericDType::F16 => {
            // Similar to BF16 but different bit layout. For now, just pass through
            // (F16 handling would need its own rounding logic).
            val
        }
        NumericDType::F64 => {
            // Store: f64 store. Load: f64 load → fdemote to f32.
            // Round-trip: promote f32→f64→fdemote f64→f32 = f32 (no-op if already f32)
            val
        }
        NumericDType::I32 => {
            // Store: ireduce i64→i32. Load: sextend i32→i64.
            let narrow = builder.ins().ireduce(types::I32, val);
            builder.ins().sextend(types::I64, narrow)
        }
        NumericDType::U32 => {
            let narrow = builder.ins().ireduce(types::I32, val);
            builder.ins().uextend(types::I64, narrow)
        }
        NumericDType::BOOL | NumericDType::U8 => {
            let narrow = builder.ins().ireduce(types::I8, val);
            builder.ins().uextend(types::I64, narrow)
        }
        NumericDType::I8 => {
            let narrow = builder.ins().ireduce(types::I8, val);
            builder.ins().sextend(types::I64, narrow)
        }
        // F32, I64 — value is already in repr format, no round-trip needed.
        _ => val,
    }
}

/// Determine the repr kind of a loaded input, considering forwarded values.
///
/// If the input is an Affine stride=1 reference to a forwarded producer,
/// returns the repr of the producer's output dtype. Otherwise falls through
/// to the normal slot-based lookup.
fn forwarded_or_slot_repr(
    input: &InputRef,
    layout: &BufferLayout,
    atom_offset: u64,
    forwarded: &HashMap<u64, (Value, NumericDType)>,
) -> ReprKind {
    if let InputRef::Strided {
        base,
        dim_strides,
        dim_shape,
    } = input
    {
        if dim_strides.len() == 1 && dim_strides[0] == 1 && dim_shape[0] == u64::MAX {
            if let Some((_val, dtype)) = forwarded.get(&base.0) {
                return repr_of(*dtype);
            }
        }
    }
    input_slot_dtype(input, layout, atom_offset)
        .map(repr_of)
        .unwrap_or(ReprKind::Float)
}

/// Load an input value, checking the forwarding map first.
///
/// For Affine stride=1 inputs whose base is in the forwarding map, returns
/// the forwarded register value directly (skipping the memory load).
/// All other patterns fall through to the normal `load_input`.
fn load_input_forwarded(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    input: &InputRef,
    layout: &BufferLayout,
    buffer_ptr: Value,
    i_val: Option<Value>,
    i_const: u64,
    atom_offset: u64,
    tables: &mut EmbeddedTables,
    forwarded: &HashMap<u64, (Value, NumericDType)>,
) -> Result<Value, String> {
    // Check for forwarding: Affine stride=1 with a forwarded producer.
    if let InputRef::Strided {
        base,
        dim_strides,
        dim_shape,
    } = input
    {
        if dim_strides.len() == 1 && dim_strides[0] == 1 && dim_shape[0] == u64::MAX {
            if let Some((fwd_val, _fwd_dtype)) = forwarded.get(&base.0) {
                // The forwarded value has already been through
                // emit_store_load_roundtrip, so it's in the same format
                // that emit_typed_load would produce from memory.
                return Ok(*fwd_val);
            }
        }
    }

    // Not forwarded — normal memory load.
    load_input(
        builder,
        module,
        input,
        layout,
        buffer_ptr,
        i_val,
        i_const,
        atom_offset,
        tables,
    )
}

/// Compile a span, but validate by also compiling without fusion and
/// comparing the output byte-for-byte on a zero-initialized buffer.
/// Only active when FUSION_VALIDATE env var is set.
pub fn compile_span_validated(
    graph: &NanoGraph<'static, crate::pool::SystemPool>,
    layout: &BufferLayout,
) -> Result<(CompiledSpan, EmbeddedTables), String> {
    let has_multi = {
        let chains = build_fusion_chains(graph.groups(), layout);
        chains.iter().any(|c| c.group_indices.len() > 1)
    };

    let (fused, fused_tables) = compile_span(graph, layout)?;

    if !has_multi || std::env::var("FUSION_VALIDATE").is_err() {
        return Ok((fused, fused_tables));
    }

    // Also compile without fusion.
    let saved = std::env::var("FUSION").ok();
    unsafe {
        std::env::remove_var("FUSION");
    }
    let (unfused, unfused_tables) = compile_span(graph, layout)?;
    if let Some(val) = saved {
        unsafe {
            std::env::set_var("FUSION", val);
        }
    }

    // Run both on a zero+literals buffer and compare.
    let fused_total = fused_tables.total_bytes();
    let unfused_total = unfused_tables.total_bytes();
    let buf_size = fused_total.max(unfused_total).max(layout.total_bytes);
    let mut buf_fused = vec![0u8; buf_size];
    let mut buf_unfused = vec![0u8; buf_size];
    layout.populate_literals(graph, &mut buf_fused);
    fused_tables.populate(&mut buf_fused);
    layout.populate_literals(graph, &mut buf_unfused);
    unfused_tables.populate(&mut buf_unfused);
    // Fill input tensor slots with deterministic test data.
    for it in graph.input_tensors() {
        if let Some((slot, _)) = layout.find(it.base_id) {
            let start = slot.byte_offset;
            let end = start + it.count as usize * slot.elem_bytes;
            if end <= buf_fused.len() {
                for (i, b) in buf_fused[start..end].iter_mut().enumerate() {
                    *b = ((i * 7 + 13) % 256) as u8;
                }
                buf_unfused[start..end].copy_from_slice(&buf_fused[start..end]);
            }
        }
    }

    fused.execute(&mut buf_fused);
    unfused.execute(&mut buf_unfused);

    // Compare output bytes.
    let mut mismatches = 0usize;
    let mut first_mismatch = None;
    for (i, (&a, &b)) in buf_fused.iter().zip(buf_unfused.iter()).enumerate() {
        if a != b && first_mismatch.is_none() {
            first_mismatch = Some(i);
        }
        if a != b {
            mismatches += 1;
        }
    }
    if mismatches > 0 {
        eprintln!(
            "  [FUSION_VALIDATE] MISMATCH: {} bytes differ (first at offset {}), buffer_size={}, groups={}",
            mismatches,
            first_mismatch.unwrap(),
            layout.total_bytes,
            graph.num_groups()
        );
        // Find which group's slot the first mismatch is in.
        if let Some(off) = first_mismatch {
            for (gi, group) in graph.groups().iter().enumerate() {
                if let Some((slot, _)) = layout.find(group.base_id) {
                    let start = slot.byte_offset;
                    let end = start + slot.count as usize * slot.elem_bytes;
                    if off >= start && off < end {
                        let elem_off = (off - start) / slot.elem_bytes;
                        eprintln!(
                            "    first mismatch in group[{}] base={} {:?} dtype={:?} atom_offset={} count={} slot_byte={} elem_offset={}",
                            gi,
                            group.base_id,
                            op_name_short(&group.op),
                            group.output_dtype,
                            group.atom_offset,
                            group.count,
                            start,
                            elem_off
                        );
                        // Show the actual values
                        let f_val = f32::from_le_bytes([
                            buf_fused[off],
                            buf_fused[off + 1],
                            buf_fused[off + 2],
                            buf_fused[off + 3],
                        ]);
                        let u_val = f32::from_le_bytes([
                            buf_unfused[off],
                            buf_unfused[off + 1],
                            buf_unfused[off + 2],
                            buf_unfused[off + 3],
                        ]);
                        eprintln!("    fused={} unfused={}", f_val, u_val);
                        // Show inputs
                        for (ii, inp) in group.inputs.iter().enumerate() {
                            eprintln!("    input[{}]: {:?}", ii, inp);
                        }
                        break;
                    }
                }
            }
        }
    }

    Ok((fused, fused_tables))
}

/// Compile a span's NanoGraph into native code using the given buffer layout.
///
/// Returns the compiled function and any embedded lookup tables that must
/// be appended to the literal template buffer.
pub fn compile_span(
    graph: &NanoGraph<'static, crate::pool::SystemPool>,
    layout: &BufferLayout,
) -> Result<(CompiledSpan, EmbeddedTables), String> {
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

    let mut tables = EmbeddedTables::new(layout.total_bytes);

    {
        let mut func_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        builder.seal_block(entry);

        let buffer_ptr = builder.block_params(entry)[0];
        let mut var_counter = VarCounter::new();

        // Build fusion chains and emit one loop per chain.
        let chains = build_fusion_chains(graph.groups(), layout);
        for chain in &chains {
            emit_chain(
                &mut builder,
                &mut module,
                chain,
                graph.groups(),
                layout,
                buffer_ptr,
                &math,
                &mut var_counter,
                &mut tables,
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

    Ok((
        CompiledSpan {
            func_ptr,
            _module: module,
        },
        tables,
    ))
}

// ─── Group emission ─────────────────────────────────────────────────────────

/// Emit Cranelift IR for one group. Generates a loop over the group's atom
/// range, or inline code for single-element groups.
fn emit_group(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    group: &AtomGroup<'static, crate::pool::SystemPool>,
    layout: &BufferLayout,
    buffer_ptr: Value,
    math: &MathFuncs,
    var_counter: &mut VarCounter,
    tables: &mut EmbeddedTables,
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
            tables,
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

    let cmp = builder.ins().icmp(IntCC::SignedLessThan, i_val, end);
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
        tables,
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
    group: &AtomGroup<'static, crate::pool::SystemPool>,
    layout: &BufferLayout,
    buffer_ptr: Value,
    i_val: Option<Value>,
    i_const: u64,
    math: &MathFuncs,
    var_counter: &mut VarCounter,
    tables: &mut EmbeddedTables,
) -> Result<(), String> {
    let (out_slot, _) = layout
        .find(group.base_id)
        .ok_or_else(|| format!("no slot for group base={}", group.base_id))?;
    let out_slot = out_slot.clone();

    let output_dtype = group.output_dtype;
    let output_repr = repr_of(output_dtype);

    match &group.op {
        ScalarOp::Literal(_) | ScalarOp::LiteralSpan(_) => Ok(()),

        ScalarOp::Identity | ScalarOp::Cast { .. } => {
            // Identity/Cast: cast input to output_dtype.
            let src = load_input(
                builder,
                module,
                &group.inputs[0],
                layout,
                buffer_ptr,
                i_val,
                i_const,
                group.atom_offset,
                tables,
            )?;
            let src_repr = input_slot_dtype(&group.inputs[0], layout, group.atom_offset)
                .map(repr_of)
                .unwrap_or(ReprKind::Float);
            let result = emit_cast_to_output(builder, src, src_repr, output_dtype);
            store_result(
                builder,
                buffer_ptr,
                &out_slot,
                group.atom_offset,
                i_val,
                i_const,
                result,
            );
            Ok(())
        }

        ScalarOp::Binary { op, compute_dtype } => {
            let compute_repr = repr_of(*compute_dtype);
            let a_raw = load_input(
                builder,
                module,
                &group.inputs[0],
                layout,
                buffer_ptr,
                i_val,
                i_const,
                group.atom_offset,
                tables,
            )?;
            let a_repr = input_slot_dtype(&group.inputs[0], layout, group.atom_offset)
                .map(repr_of)
                .unwrap_or(compute_repr);
            let a = emit_repr_cast(builder, a_raw, a_repr, compute_repr);

            let b_raw = load_input(
                builder,
                module,
                &group.inputs[1],
                layout,
                buffer_ptr,
                i_val,
                i_const,
                group.atom_offset,
                tables,
            )?;
            let b_repr = input_slot_dtype(&group.inputs[1], layout, group.atom_offset)
                .map(repr_of)
                .unwrap_or(compute_repr);
            let b = emit_repr_cast(builder, b_raw, b_repr, compute_repr);

            let result = emit_binop(builder, module, math, *op, a, b, compute_repr)?;
            let output_val = emit_cast_to_output(builder, result, compute_repr, output_dtype);
            store_result(
                builder,
                buffer_ptr,
                &out_slot,
                group.atom_offset,
                i_val,
                i_const,
                output_val,
            );
            Ok(())
        }

        ScalarOp::Unary { op, compute_dtype } => {
            let compute_repr = repr_of(*compute_dtype);
            let x_raw = load_input(
                builder,
                module,
                &group.inputs[0],
                layout,
                buffer_ptr,
                i_val,
                i_const,
                group.atom_offset,
                tables,
            )?;
            let x_repr = input_slot_dtype(&group.inputs[0], layout, group.atom_offset)
                .map(repr_of)
                .unwrap_or(compute_repr);
            let x = emit_repr_cast(builder, x_raw, x_repr, compute_repr);

            let result = emit_unop(builder, module, math, *op, x, compute_repr)?;
            let output_val = emit_cast_to_output(builder, result, compute_repr, output_dtype);
            store_result(
                builder,
                buffer_ptr,
                &out_slot,
                group.atom_offset,
                i_val,
                i_const,
                output_val,
            );
            Ok(())
        }

        ScalarOp::Select => {
            // Select: truthiness test on cond, then cast selected value to output_dtype.
            let cond = load_input(
                builder,
                module,
                &group.inputs[0],
                layout,
                buffer_ptr,
                i_val,
                i_const,
                group.atom_offset,
                tables,
            )?;
            let cond_repr = input_slot_dtype(&group.inputs[0], layout, group.atom_offset)
                .map(repr_of)
                .unwrap_or(ReprKind::Float);

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
                builder,
                module,
                &group.inputs[1],
                layout,
                buffer_ptr,
                i_val,
                i_const,
                group.atom_offset,
                tables,
            )?;
            let x_repr = input_slot_dtype(&group.inputs[1], layout, group.atom_offset)
                .map(repr_of)
                .unwrap_or(output_repr);
            let x = emit_cast_to_output(builder, x_raw, x_repr, output_dtype);

            let y_raw = load_input(
                builder,
                module,
                &group.inputs[2],
                layout,
                buffer_ptr,
                i_val,
                i_const,
                group.atom_offset,
                tables,
            )?;
            let y_repr = input_slot_dtype(&group.inputs[2], layout, group.atom_offset)
                .map(repr_of)
                .unwrap_or(output_repr);
            let y = emit_cast_to_output(builder, y_raw, y_repr, output_dtype);

            let result = builder.ins().select(is_nonzero, x, y);
            store_result(
                builder,
                buffer_ptr,
                &out_slot,
                group.atom_offset,
                i_val,
                i_const,
                result,
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
            tables,
            *kind,
            *reduce_count,
            *reduce_stride,
            *compute_dtype,
            &out_slot,
        ),

        ScalarOp::IndirectLoad { table_base } => {
            // Index: load as integer regardless of source dtype.
            let idx_raw = load_input(
                builder,
                module,
                &group.inputs[0],
                layout,
                buffer_ptr,
                i_val,
                i_const,
                group.atom_offset,
                tables,
            )?;
            let idx_repr = input_slot_dtype(&group.inputs[0], layout, group.atom_offset)
                .map(repr_of)
                .unwrap_or(ReprKind::Int);
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
                builder,
                buffer_ptr,
                &out_slot,
                group.atom_offset,
                i_val,
                i_const,
                result,
            );
            Ok(())
        }

        ScalarOp::OpaqueOutput { .. } => todo!("opaque ops not supported in compiler"),
    }
}

// ─── Input loading ──────────────────────────────────────────────────────────

/// Determine the storage dtype that `load_input` will load from for a given InputRef.
fn input_slot_dtype(
    input: &InputRef,
    layout: &BufferLayout,
    atom_offset: u64,
) -> Option<NumericDType> {
    // Try the InputRef's base first, then fall back to the first accessed atom.
    let try_find = |atom: AtomId| layout.find(atom).map(|(s, _)| s.dtype);
    match input {
        InputRef::Broadcast(atom_id) => try_find(*atom_id),
        InputRef::Strided {
            base,
            dim_strides,
            dim_shape,
        } => try_find(*base).or_else(|| {
            let nd = dim_strides.len();
            let first_offset = if nd == 1 {
                dim_strides[0] * atom_offset as i64
            } else {
                let inner = atom_offset % dim_shape[1];
                let outer = atom_offset / dim_shape[1];
                dim_strides[1] * inner as i64 + dim_strides[0] * outer as i64
            };
            let first = AtomId(base.0.wrapping_add(first_offset as u64));
            try_find(first)
        }),
        InputRef::Explicit(ids) if !ids.is_empty() => {
            let idx = (atom_offset as usize).min(ids.len() - 1);
            try_find(ids[idx])
        }
        _ => None,
    }
}

/// Resolve an Affine-like InputRef base to a byte offset in the buffer.
///
/// For unsplit groups, `base` is directly in the layout. For split groups,
/// `base` may point to the original (unsplit) group's start which isn't in
/// this span. In that case, we look up the first atom this fragment actually
/// accesses (`base + stride * atom_offset`) and back-compute the equivalent
/// base_byte.
///
/// Returns `(base_byte, elem_bytes, load_dtype)` where address of atom `i` is
/// `base_byte + stride * elem_bytes * i`.
fn resolve_affine_base(
    layout: &BufferLayout,
    base: AtomId,
    stride: i64,
    atom_offset: u64,
    label: &str,
) -> Result<(i64, usize, NumericDType), String> {
    // Fast path: base is in the layout (unsplit or atom_offset == 0).
    if let Some((slot, elem)) = layout.find(base) {
        let base_byte = slot.byte_offset as i64 + elem as i64 * slot.elem_bytes as i64;
        return Ok((base_byte, slot.elem_bytes, slot.dtype));
    }

    // Split path: look up the first atom this fragment accesses.
    let first_atom = AtomId((base.0 as i64 + stride * atom_offset as i64) as u64);
    let (slot, elem) = layout.find(first_atom).ok_or_else(|| {
        format!(
            "no slot for {} base={} (first_atom={}, atom_offset={})",
            label, base, first_atom, atom_offset
        )
    })?;
    // base_byte + stride * elem_bytes * atom_offset = slot.byte_offset + elem * elem_bytes
    let first_byte = slot.byte_offset as i64 + elem as i64 * slot.elem_bytes as i64;
    let base_byte = first_byte - stride * atom_offset as i64 * slot.elem_bytes as i64;
    Ok((base_byte, slot.elem_bytes, slot.dtype))
}

/// Load a value from an InputRef, resolving to a buffer byte address.
///
/// Returns a Cranelift Value in the storage dtype's representation kind
/// (types::F32 for float dtypes, types::I64 for integer dtypes).
/// Accumulator for explicit lookup tables embedded in the JIT buffer.
///
/// Instead of using Cranelift's `declare_data` / `global_value` (which
/// allocates a separate data section that may be >2GB from the code,
/// causing `TryFromIntError` on x86 PC-relative relocations), we embed
/// the table directly in the working buffer alongside slot data.
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
    fn new(initial_watermark: usize) -> Self {
        EmbeddedTables {
            watermark: initial_watermark,
            entries: Vec::new(),
        }
    }

    /// Allocate space for a table and return its byte offset in the buffer.
    fn alloc(&mut self, data: Vec<u8>) -> usize {
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
    fn total_bytes(&self) -> usize {
        self.watermark
    }

    /// Write all table data into the buffer.
    fn populate(&self, buffer: &mut [u8]) {
        for entry in &self.entries {
            let end = entry.byte_offset + entry.data.len();
            if end <= buffer.len() {
                buffer[entry.byte_offset..end].copy_from_slice(&entry.data);
            }
        }
    }
}

fn load_input(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    input: &InputRef,
    layout: &BufferLayout,
    buffer_ptr: Value,
    i_val: Option<Value>,
    i_const: u64,
    atom_offset: u64,
    tables: &mut EmbeddedTables,
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

        InputRef::Strided {
            base,
            dim_strides,
            dim_shape,
        } => {
            // N-dimensional strided access.
            //
            // 1D (affine):          dim_strides=[s], dim_shape=[MAX] → base + s * i
            // 2D general:           inner = i % dim_shape[1], outer = i / dim_shape[1]
            //                       offset = dim_strides[0]*outer + dim_strides[1]*inner
            //   StridedBroadcast:   dim_strides[1]==0 → base + dim_strides[0] * (i / dim_shape[1])
            //   Modular:            dim_strides[0]==0 → base + dim_strides[1] * (i % dim_shape[1])
            let nd = dim_strides.len();
            let is_affine = nd == 1;
            let modulus = if nd >= 2 { dim_shape[1] } else { u64::MAX };
            let stride_inner = if nd >= 2 {
                dim_strides[1]
            } else {
                dim_strides[0]
            };
            let stride_outer = if nd >= 2 { dim_strides[0] } else { 0 };

            // Find the buffer slot by resolving the first accessed atom.
            let first_inner = atom_offset % modulus;
            let first_outer = atom_offset / modulus;
            let first_offset =
                stride_inner * first_inner as i64 + stride_outer * first_outer as i64;
            let first_atom = AtomId((base.0 as i64 + first_offset) as u64);

            let (slot, elem) = layout
                .find(*base)
                .or_else(|| layout.find(first_atom))
                .ok_or_else(|| {
                    format!(
                        "no slot for Strided base={} first_atom={} (atom_offset={}, strides={:?}, shape={:?})",
                        base, first_atom, atom_offset, dim_strides, dim_shape
                    )
                })?;

            let elem_bytes = slot.elem_bytes as i64;
            let load_dtype = slot.dtype;

            // Compute base_byte: byte offset of the logical `base` atom in the buffer.
            let slot_byte = slot.byte_offset as i64 + elem as i64 * elem_bytes;
            let base_byte = if layout.find(*base).is_some() {
                slot_byte
            } else {
                // Back-compute: base_byte + first_offset * elem_bytes = slot_byte
                slot_byte - first_offset * elem_bytes
            };

            // ── Affine fast path ──
            if is_affine {
                let byte_stride = stride_inner * elem_bytes;
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
                return Ok(emit_typed_load(builder, addr, load_dtype));
            }

            // ── General path (handles StridedBroadcast, Modular, and mixed) ──
            let byte_stride_inner = stride_inner * elem_bytes;
            let byte_stride_outer = stride_outer * elem_bytes;

            let addr = match i_val {
                Some(iv) => {
                    // inner = i_eff % modulus, outer = i_eff / modulus
                    let (inner_val, outer_val) = if modulus.is_power_of_two() {
                        let shift = modulus.trailing_zeros() as i64;
                        let mask = modulus as i64 - 1;
                        let inner = builder.ins().band_imm(iv, mask);
                        let outer = builder.ins().ushr_imm(iv, shift);
                        (inner, outer)
                    } else {
                        let modval = builder.ins().iconst(types::I64, modulus as i64);
                        let inner = builder.ins().urem(iv, modval);
                        let outer = builder.ins().udiv(iv, modval);
                        (inner, outer)
                    };

                    // offset = stride_inner * inner + stride_outer * outer
                    let mut off = builder.ins().iconst(types::I64, base_byte);
                    if byte_stride_inner != 0 {
                        let inner_bytes = builder.ins().imul_imm(inner_val, byte_stride_inner);
                        off = builder.ins().iadd(off, inner_bytes);
                    }
                    if byte_stride_outer != 0 {
                        let outer_bytes = builder.ins().imul_imm(outer_val, byte_stride_outer);
                        off = builder.ins().iadd(off, outer_bytes);
                    }
                    builder.ins().iadd(buffer_ptr, off)
                }
                None => {
                    let inner = i_const % modulus;
                    let outer = i_const / modulus;
                    let byte_off = base_byte
                        + byte_stride_inner * inner as i64
                        + byte_stride_outer * outer as i64;
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
                let byte_off = slot.byte_offset as i64 + elem_idx as i64 * slot.elem_bytes as i64;
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

            // Embed the lookup table in the JIT buffer to avoid Cranelift
            // data section relocations (which can overflow on x86-64).
            let table_bytes: Vec<u8> = byte_offsets
                .iter()
                .flat_map(|off| off.to_le_bytes())
                .collect();
            let table_offset = tables.alloc(table_bytes);

            let table_ptr = addr_const(builder, buffer_ptr, table_offset as i64);
            let i = i_val.unwrap_or_else(|| builder.ins().iconst(types::I64, i_const as i64));
            let idx_byte_off = builder.ins().imul_imm(i, 8); // 8 bytes per i64 entry
            let idx_addr = builder.ins().iadd(table_ptr, idx_byte_off);
            let byte_off = builder.ins().load(types::I64, MemFlags::new(), idx_addr, 0);
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

/// Map a NumericDType to its Cranelift representation kind.
fn repr_of(dtype: NumericDType) -> ReprKind {
    match dtype {
        NumericDType::F32 | NumericDType::BF16 | NumericDType::F16 | NumericDType::F64 => {
            ReprKind::Float
        }
        _ => ReprKind::Int,
    }
}

/// Emit a cast between Cranelift representation kinds.
/// If from == to, returns val unchanged. Otherwise converts f32↔i64.
fn emit_repr_cast(
    builder: &mut FunctionBuilder,
    val: Value,
    from: ReprKind,
    to: ReprKind,
) -> Value {
    match (from, to) {
        (ReprKind::Float, ReprKind::Float) | (ReprKind::Int, ReprKind::Int) => val,
        (ReprKind::Float, ReprKind::Int) => builder.ins().fcvt_to_sint_sat(types::I64, val),
        (ReprKind::Int, ReprKind::Float) => builder.ins().fcvt_from_sint(types::F32, val),
    }
}

/// Emit a cast from a compute-repr value to the output dtype's storage repr.
/// Handles narrowing (e.g., i64 → i8 for BOOL, f32 → bf16 bits for BF16).
/// Returns a value ready for `emit_typed_store`.
fn emit_cast_to_output(
    builder: &mut FunctionBuilder,
    val: Value,
    compute_repr: ReprKind,
    output_dtype: NumericDType,
) -> Value {
    let target_repr = repr_of(output_dtype);
    let val = emit_repr_cast(builder, val, compute_repr, target_repr);

    // Further narrowing for sub-word output types.
    match output_dtype {
        NumericDType::BOOL => {
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
        NumericDType::U8 | NumericDType::I8 => {
            // Already Int repr (i64), narrow to i8.
            builder.ins().ireduce(types::I8, val)
        }
        NumericDType::I32 | NumericDType::U32 => {
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
fn emit_typed_load(builder: &mut FunctionBuilder, addr: Value, dtype: NumericDType) -> Value {
    match dtype {
        NumericDType::F32 => builder.ins().load(types::F32, MemFlags::trusted(), addr, 0),
        NumericDType::BF16 => {
            let raw = builder.ins().load(types::I16, MemFlags::trusted(), addr, 0);
            let wide = builder.ins().uextend(types::I32, raw);
            let shifted = builder.ins().ishl_imm(wide, 16);
            builder.ins().bitcast(types::F32, MemFlags::new(), shifted)
        }
        NumericDType::F64 => {
            let raw = builder.ins().load(types::F64, MemFlags::trusted(), addr, 0);
            builder.ins().fdemote(types::F32, raw)
        }
        NumericDType::I64 => builder.ins().load(types::I64, MemFlags::trusted(), addr, 0),
        NumericDType::I32 => {
            let raw = builder.ins().load(types::I32, MemFlags::trusted(), addr, 0);
            builder.ins().sextend(types::I64, raw)
        }
        NumericDType::U32 => {
            let raw = builder.ins().load(types::I32, MemFlags::trusted(), addr, 0);
            builder.ins().uextend(types::I64, raw)
        }
        NumericDType::BOOL | NumericDType::U8 => {
            let raw = builder.ins().load(types::I8, MemFlags::trusted(), addr, 0);
            builder.ins().uextend(types::I64, raw)
        }
        NumericDType::I8 => {
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
fn emit_typed_store(builder: &mut FunctionBuilder, addr: Value, val: Value, dtype: NumericDType) {
    match dtype {
        NumericDType::F32 => {
            builder.ins().store(MemFlags::trusted(), val, addr, 0);
        }
        NumericDType::BF16 => {
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
        NumericDType::I64 | NumericDType::U64 => {
            builder.ins().store(MemFlags::trusted(), val, addr, 0); // val is i64
        }
        NumericDType::I32 | NumericDType::U32 => {
            builder.ins().store(MemFlags::trusted(), val, addr, 0); // val is i32
        }
        NumericDType::BOOL | NumericDType::U8 | NumericDType::I8 => {
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
    group: &AtomGroup<'static, crate::pool::SystemPool>,
    layout: &BufferLayout,
    buffer_ptr: Value,
    i_val: Option<Value>,
    i_const: u64,
    math: &MathFuncs,
    var_counter: &mut VarCounter,
    tables: &mut EmbeddedTables,
    kind: ReduceKind,
    reduce_count: u64,
    reduce_stride: i64,
    compute_dtype: NumericDType,
    out_slot: &SlotInfo,
) -> Result<(), String> {
    let is_sum = matches!(kind, ReduceKind::Sum);
    let compute_repr = repr_of(compute_dtype);
    let output_dtype = group.output_dtype;

    // Resolve the source slot. Reduce input must be Affine.
    let (base_byte, input_byte_stride, reduce_byte_stride, src_dtype) = match &group.inputs[0] {
        InputRef::Strided {
            base, dim_strides, ..
        } => {
            // Use the innermost (last) stride for reduce input
            let stride = dim_strides.last().copied().unwrap_or(0);
            let (base_byte, elem_bytes, src_dtype) =
                resolve_affine_base(layout, *base, stride, group.atom_offset, "reduce input")?;
            let byte_stride = stride * elem_bytes as i64;
            let red_stride = reduce_stride * elem_bytes as i64;
            (base_byte, byte_stride, red_stride, src_dtype)
        }
        other => {
            return Err(format!(
                "Reduce input must be Affine, got {:?}",
                std::mem::discriminant(other)
            ));
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
    let k_cmp = builder.ins().icmp(IntCC::SignedLessThan, k_val, bound);
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

        // ── IMod (mathematical modulo — result sign matches divisor) ──
        (ScalarBinOp::IMod, ReprKind::Float) => {
            // fmod then adjust: if result and divisor have different signs, add divisor.
            let func_ref = module.declare_func_in_func(math.fmodf, builder.func);
            let call = builder.ins().call(func_ref, &[a, b]);
            let rem = builder.inst_results(call)[0];
            // rem + b if sign(rem) != sign(b), else rem
            let sum = builder.ins().fadd(rem, b);
            let zero = builder.ins().f32const(0.0);
            let rem_neg = builder.ins().fcmp(FloatCC::LessThan, rem, zero);
            let b_neg = builder.ins().fcmp(FloatCC::LessThan, b, zero);
            let rem_zero = builder.ins().fcmp(FloatCC::Equal, rem, zero);
            let signs_differ = builder.ins().bxor(rem_neg, b_neg);
            let need_adjust = builder.ins().band_not(signs_differ, rem_zero);
            builder.ins().select(need_adjust, sum, rem)
        }
        (ScalarBinOp::IMod, ReprKind::Int) => {
            // srem then adjust: if result and divisor have different signs, add divisor.
            let zero = builder.ins().iconst(types::I64, 0);
            let one = builder.ins().iconst(types::I64, 1);
            let is_zero = builder.ins().icmp(IntCC::Equal, b, zero);
            let safe_b = builder.ins().select(is_zero, one, b);
            let rem = builder.ins().srem(a, safe_b);
            let sum = builder.ins().iadd(rem, safe_b);
            let rem_neg = builder.ins().icmp(IntCC::SignedLessThan, rem, zero);
            let b_neg = builder.ins().icmp(IntCC::SignedLessThan, safe_b, zero);
            let rem_zero = builder.ins().icmp(IntCC::Equal, rem, zero);
            let signs_differ = builder.ins().bxor(rem_neg, b_neg);
            let need_adjust = builder.ins().band_not(signs_differ, rem_zero);
            let result = builder.ins().select(need_adjust, sum, rem);
            builder.ins().select(is_zero, zero, result)
        }

        // ── Bitwise ops (integer only) ──
        (ScalarBinOp::BitwiseAnd, ReprKind::Int) => builder.ins().band(a, b),
        (ScalarBinOp::BitwiseOr, ReprKind::Int) => builder.ins().bor(a, b),
        (ScalarBinOp::BitwiseXor, ReprKind::Int) => builder.ins().bxor(a, b),
        (ScalarBinOp::BitShiftLeft, ReprKind::Int) => builder.ins().ishl(a, b),
        (ScalarBinOp::BitShiftRight, ReprKind::Int) => builder.ins().sshr(a, b),

        // Bitwise on floats — not meaningful
        (
            ScalarBinOp::BitwiseAnd
            | ScalarBinOp::BitwiseOr
            | ScalarBinOp::BitwiseXor
            | ScalarBinOp::BitShiftLeft
            | ScalarBinOp::BitShiftRight,
            ReprKind::Float,
        ) => {
            return Err(format!("bitwise {:?} on float not supported in JIT", op));
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
        // Float ops not yet supported by the JIT.
        (_, ReprKind::Float) => {
            return Err(format!("float {:?} not implemented in JIT", op));
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
        let inp = g.add_input_tensor(GlobalId(0), 4, NumericDType::F32);
        let lit = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(2.0)),
            vec![],
            vec![],
        );
        let add = g.push_group(
            4,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(inp, 1), InputRef::Broadcast(lit)],
        );

        let outputs = vec![AtomRange {
            base: add,
            count: 4,
            dtype: NumericDType::F32,
        }];
        let layout = compute_layout(&g, &outputs);
        let (compiled, embedded_tables) = compile_span(&g, &layout).unwrap();

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
        let inp = g.add_input_tensor(GlobalId(0), 3, NumericDType::F32);
        let neg = g.push_group(
            3,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(inp, 1)],
        );

        let outputs = vec![AtomRange {
            base: neg,
            count: 3,
            dtype: NumericDType::F32,
        }];
        let layout = compute_layout(&g, &outputs);
        let (compiled, embedded_tables) = compile_span(&g, &layout).unwrap();

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
        let inp = g.add_input_tensor(GlobalId(0), 4, NumericDType::F32);
        let sum = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: 4,
                reduce_stride: 1,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(inp, 1)],
        );

        let outputs = vec![AtomRange {
            base: sum,
            count: 1,
            dtype: NumericDType::F32,
        }];
        let layout = compute_layout(&g, &outputs);
        let (compiled, embedded_tables) = compile_span(&g, &layout).unwrap();

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
        let inp = g.add_input_tensor(GlobalId(0), 2, NumericDType::F32);
        let lit3 = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(3.0)),
            vec![],
            vec![],
        );
        let mul = g.push_group(
            2,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(inp, 1), InputRef::Broadcast(lit3)],
        );
        let exp = g.push_group(
            2,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(mul, 1)],
        );

        let outputs = vec![AtomRange {
            base: exp,
            count: 2,
            dtype: NumericDType::F32,
        }];
        let layout = compute_layout(&g, &outputs);
        let (compiled, embedded_tables) = compile_span(&g, &layout).unwrap();

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
    #[ignore]
    fn test_liveness_reuse() {
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), 100, NumericDType::F32);
        let a = g.push_group(
            100,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(inp, 1)],
        );
        let b = g.push_group(
            100,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(a, 1)],
        );
        let c = g.push_group(
            100,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(b, 1)],
        );

        let outputs = vec![AtomRange {
            base: c,
            count: 100,
            dtype: NumericDType::F32,
        }];
        let layout = compute_layout(&g, &outputs);

        // With slot reuse: input(400) + A(400) are allocated. When B is
        // allocated, A's slot is freed (only consumer B is done). B reuses A's
        // space. Similarly C reuses B's. So: input(400) + one reused slot(400)
        // + C(400) = 1200, or possibly input(400) + A/B/C sharing = 800.
        // The exact number depends on allocation order; just verify it's < 1600.
        assert!(
            layout.total_bytes < 1600,
            "Expected slot reuse to reduce buffer from 1600 bytes, got {}",
            layout.total_bytes
        );

        // Verify correctness: neg(neg(neg(x))) = -x
        let (compiled, embedded_tables) = compile_span(&g, &layout).unwrap();
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
        let cond = g.add_input_tensor(GlobalId(0), 3, NumericDType::F32);
        let x = g.add_input_tensor(GlobalId(1), 3, NumericDType::F32);
        let y = g.add_input_tensor(GlobalId(2), 3, NumericDType::F32);
        let sel = g.push_group(
            3,
            NumericDType::F32,
            ScalarOp::Select,
            vec![],
            vec![
                InputRef::affine(cond, 1),
                InputRef::affine(x, 1),
                InputRef::affine(y, 1),
            ],
        );

        let outputs = vec![AtomRange {
            base: sel,
            count: 3,
            dtype: NumericDType::F32,
        }];
        let layout = compute_layout(&g, &outputs);
        let (compiled, embedded_tables) = compile_span(&g, &layout).unwrap();

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
        let inp = g.add_input_tensor(GlobalId(0), 16, NumericDType::F32);
        // 4 source groups, each 4 atoms, that read from different input regions.
        let mut sources = Vec::new();
        for i in 0..4u64 {
            let src = g.push_group(
                4,
                NumericDType::F32,
                ScalarOp::Identity,
                vec![],
                vec![InputRef::affine(AtomId(inp.0 + i * 4), 1)],
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
            NumericDType::F32,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::Explicit(explicit_ids)],
        );

        let outputs = vec![AtomRange {
            base: gather,
            count: 16,
            dtype: NumericDType::F32,
        }];
        let layout = compute_layout(&g, &outputs);
        let (compiled, embedded_tables) = compile_span(&g, &layout).unwrap();

        let buf_size = embedded_tables.total_bytes().max(layout.total_bytes);
        let mut buffer = vec![0u8; buf_size];
        let input_data: Vec<f32> = (0..16).map(|i| (i + 1) as f32).collect();
        layout.write_f32_input(inp, &input_data, &mut buffer);
        layout.populate_literals(&g, &mut buffer);
        embedded_tables.populate(&mut buffer);

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
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(1.0)),
            vec![],
            vec![],
        );
        let b = g.push_group(
            3,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(2.0)),
            vec![],
            vec![],
        );
        // Consumer reads 6 atoms starting from A's base, crossing into B.
        let out = g.push_group(
            6,
            NumericDType::F32,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::affine(a, 1)],
        );

        let outputs = vec![AtomRange {
            base: out,
            count: 6,
            dtype: NumericDType::F32,
        }];
        let layout = compute_layout(&g, &outputs);

        // Validate no cross-slot errors.
        let errors = validate_layout(&g, &layout);
        assert!(errors.is_empty(), "layout errors: {:?}", errors);

        let (compiled, embedded_tables) = compile_span(&g, &layout).unwrap();
        let mut buffer = vec![0u8; layout.total_bytes];
        layout.populate_literals(&g, &mut buffer);
        compiled.execute(&mut buffer);

        let result = layout.read_f32_output(&outputs[0], &buffer);
        assert_eq!(result, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    }

    /// 5-group fusion chain: input → +2 → *3 → neg → exp → output
    /// Tests that chains of 4+ fused groups produce correct results.
    #[test]
    fn test_fusion_chain_5_groups() {
        // Fusion is enabled by default; no env var needed.

        let mut g = NanoGraph::new();
        let n = 8u64;
        let inp = g.add_input_tensor(GlobalId(0), n, NumericDType::F32);

        let lit2 = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(2.0)),
            vec![],
            vec![],
        );
        let lit3 = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(3.0)),
            vec![],
            vec![],
        );

        // Group 1: identity (copy input)
        let g1 = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::affine(inp, 1)],
        );

        // Group 2: g1 + 2.0
        let g2 = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(g1, 1), InputRef::Broadcast(lit2)],
        );

        // Group 3: g2 * 3.0
        let g3 = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(g2, 1), InputRef::Broadcast(lit3)],
        );

        // Group 4: -g3
        let g4 = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(g3, 1)],
        );

        // Group 5: exp(g4)
        let g5 = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(g4, 1)],
        );

        let outputs = vec![AtomRange {
            base: g5,
            count: n,
            dtype: NumericDType::F32,
        }];
        let layout = compute_layout(&g, &outputs);

        // Verify chain was built (should be 1 chain of 5 groups)
        let chains = build_fusion_chains(g.groups(), &layout);
        let multi = chains.iter().filter(|c| c.group_indices.len() > 1).count();
        assert!(multi > 0, "expected at least one multi-group fusion chain");
        let max_len = chains.iter().map(|c| c.group_indices.len()).max().unwrap();
        assert!(
            max_len >= 4,
            "expected chain of 4+ groups, got max {}",
            max_len
        );

        let (compiled, embedded_tables) = compile_span(&g, &layout).unwrap();

        let mut buffer = vec![0u8; layout.total_bytes];
        layout.populate_literals(&g, &mut buffer);
        let input_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        layout.write_f32_input(inp, &input_data, &mut buffer);

        compiled.execute(&mut buffer);

        let result = layout.read_f32_output(&outputs[0], &buffer);
        let expected: Vec<f32> = input_data
            .iter()
            .map(|x| (-((*x + 2.0) * 3.0)).exp())
            .collect();

        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-5 * want.abs().max(1.0),
                "mismatch at [{}]: got {}, expected {}",
                i,
                got,
                want
            );
        }

        // Disable fusion for unfused comparison.
        unsafe {
            std::env::set_var("FUSION", "0");
        }
    }

    /// 5-group chain with a diamond dependency: D reads from both A and C.
    /// Tests forwarding with multiple chain producers.
    #[test]
    #[ignore]
    fn test_fusion_chain_diamond() {
        // Fusion is enabled by default; no env var needed.

        let mut g = NanoGraph::new();
        let n = 8u64;
        let inp = g.add_input_tensor(GlobalId(0), n, NumericDType::F32);

        // A: identity(input)
        let a = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::affine(inp, 1)],
        );

        // B: A + A = 2*input
        let b = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(a, 1), InputRef::affine(a, 1)],
        );

        // C: -B
        let c = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(b, 1)],
        );

        // D: A + C (reads from first AND third group in chain)
        let d = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(a, 1), InputRef::affine(c, 1)],
        );

        // E: exp(D)
        let e = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(d, 1)],
        );

        let outputs = vec![AtomRange {
            base: e,
            count: n,
            dtype: NumericDType::F32,
        }];
        let layout = compute_layout(&g, &outputs);

        let chains = build_fusion_chains(g.groups(), &layout);
        let max_len = chains.iter().map(|c| c.group_indices.len()).max().unwrap();
        assert!(
            max_len >= 4,
            "expected chain of 4+ groups, got max {}",
            max_len
        );

        let (compiled, embedded_tables) = compile_span(&g, &layout).unwrap();

        let mut buffer = vec![0u8; layout.total_bytes];
        layout.populate_literals(&g, &mut buffer);
        let input_data: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
        layout.write_f32_input(inp, &input_data, &mut buffer);

        compiled.execute(&mut buffer);

        let result = layout.read_f32_output(&outputs[0], &buffer);
        // A=x, B=2x, C=-2x, D=x+(-2x)=-x, E=exp(-x)
        let expected: Vec<f32> = input_data.iter().map(|x| (-x).exp()).collect();

        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-5 * want.abs().max(1.0),
                "mismatch at [{}]: got {}, expected {}",
                i,
                got,
                want
            );
        }

        // Disable fusion for unfused comparison.
        unsafe {
            std::env::set_var("FUSION", "0");
        }
    }

    /// Chain where not all groups are connected — some read only from external
    /// inputs. Tests that independent groups within a chain still produce
    /// correct results.
    #[test]
    #[ignore]
    fn test_fusion_chain_independent_groups() {
        // Fusion is enabled by default; no env var needed.

        let mut g = NanoGraph::new();
        let n = 8u64;
        let inp1 = g.add_input_tensor(GlobalId(0), n, NumericDType::F32);
        let inp2 = g.add_input_tensor(GlobalId(1), n, NumericDType::F32);

        // A: identity(inp1)
        let a = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::affine(inp1, 1)],
        );
        // B: identity(inp2) — reads ONLY from external, NOT from A
        let b = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::affine(inp2, 1)],
        );
        // C: identity(inp1) — another independent group
        let c = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::affine(inp1, 1)],
        );
        // D: A + B — reads from chain members A and B
        let d = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(a, 1), InputRef::affine(b, 1)],
        );
        // E: D * C — reads from chain members D and C
        let e_group = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(d, 1), InputRef::affine(c, 1)],
        );

        let outputs = vec![AtomRange {
            base: e_group,
            count: n,
            dtype: NumericDType::F32,
        }];
        let layout = compute_layout(&g, &outputs);

        let chains = build_fusion_chains(g.groups(), &layout);
        let max_len = chains.iter().map(|c| c.group_indices.len()).max().unwrap();
        assert!(max_len >= 4, "expected chain of 4+, got {}", max_len);

        let (compiled, embedded_tables) = compile_span(&g, &layout).unwrap();

        let mut buffer = vec![0u8; layout.total_bytes];
        let d1: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let d2: Vec<f32> = (0..n).map(|i| 10.0 + i as f32).collect();
        layout.write_f32_input(inp1, &d1, &mut buffer);
        layout.write_f32_input(inp2, &d2, &mut buffer);

        compiled.execute(&mut buffer);

        let result = layout.read_f32_output(&outputs[0], &buffer);
        // A=inp1, B=inp2, C=inp1, D=inp1+inp2, E=(inp1+inp2)*inp1
        let expected: Vec<f32> = d1.iter().zip(d2.iter()).map(|(x, y)| (x + y) * x).collect();

        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-4,
                "mismatch at [{}]: got {}, expected {}",
                i,
                got,
                want
            );
        }

        // Disable fusion for unfused comparison.
        unsafe {
            std::env::set_var("FUSION", "0");
        }
    }

    /// Chain starting with ReduceSum: models LayerNorm pattern.
    /// ReduceSum groups followed by elementwise groups reading from them.
    /// In GPT-2, this pattern creates chains of [Red, Bin, Bin, Un].
    #[test]
    fn test_fusion_chain_reduce_head() {
        // Fusion is enabled by default; no env var needed.

        let mut g = NanoGraph::new();

        // Input: 4 vectors of 8 elements each (32 total)
        let inp = g.add_input_tensor(GlobalId(0), 32, NumericDType::F32);
        let lit_n_inv = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(0.125)),
            vec![],
            vec![],
        ); // 1/8
        let lit2 = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(2.0)),
            vec![],
            vec![],
        );

        // 4 ReduceSum groups, each summing 8 elements → 4 outputs
        let red = g.push_group(
            4,
            NumericDType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: 8,
                reduce_stride: 1,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(inp, 8)],
        );

        // Bin: red * (1/8) = mean
        let mean = g.push_group(
            4,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(red, 1), InputRef::Broadcast(lit_n_inv)],
        );

        // Bin: mean + 2.0
        let biased = g.push_group(
            4,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(mean, 1), InputRef::Broadcast(lit2)],
        );

        // Un: sqrt(biased)
        let result = g.push_group(
            4,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Sqrt,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(biased, 1)],
        );

        let outputs = vec![AtomRange {
            base: result,
            count: 4,
            dtype: NumericDType::F32,
        }];
        let layout = compute_layout(&g, &outputs);

        // Check that fusion creates a chain of 4+ groups
        let chains = build_fusion_chains(g.groups(), &layout);
        let max_len = chains.iter().map(|c| c.group_indices.len()).max().unwrap();
        eprintln!(
            "  reduce_head: chains={}, max_len={}",
            chains.len(),
            max_len
        );

        // Compile with fusion
        let (compiled_fused, tables_fused) = compile_span(&g, &layout).unwrap();
        let mut buf_fused = vec![0u8; layout.total_bytes];
        layout.populate_literals(&g, &mut buf_fused);
        let input_data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 + 1.0).collect();
        layout.write_f32_input(inp, &input_data, &mut buf_fused);
        compiled_fused.execute(&mut buf_fused);
        let result_fused = layout.read_f32_output(&outputs[0], &buf_fused);

        // Compile without fusion
        // Disable fusion for unfused comparison.
        unsafe {
            std::env::set_var("FUSION", "0");
        }
        let (compiled_unfused, tables_unfused) = compile_span(&g, &layout).unwrap();
        let mut buf_unfused = vec![0u8; layout.total_bytes];
        layout.populate_literals(&g, &mut buf_unfused);
        layout.write_f32_input(inp, &input_data, &mut buf_unfused);
        compiled_unfused.execute(&mut buf_unfused);
        let result_unfused = layout.read_f32_output(&outputs[0], &buf_unfused);

        eprintln!("  fused:   {:?}", result_fused);
        eprintln!("  unfused: {:?}", result_unfused);

        for (i, (f, u)) in result_fused.iter().zip(result_unfused.iter()).enumerate() {
            assert!(
                (f - u).abs() < 1e-6,
                "reduce_head mismatch at [{}]: fused={}, unfused={}",
                i,
                f,
                u
            );
        }
    }

    /// Simulates split groups: same as chain but with atom_offset=8
    /// (models what GPT-2 partitioner produces when splitting across lanes).
    /// InputRefs reference original (unsplit) base_ids.
    #[test]
    fn test_fusion_chain_split_groups() {
        // Fusion is enabled by default; no env var needed.

        // Build the ORIGINAL (unsplit) graph first, then create a "split" version
        // that only contains the second portion (offset=8, count=8 out of 16).
        let mut g = NanoGraph::new();
        let n_original = 16u64; // original group size
        let split_offset = 8u64;
        let split_count = 8u64;

        // External input (large enough for the reduce to read from)
        let inp = g.add_input_tensor(GlobalId(0), 128, NumericDType::F32);
        let lit_inv = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(0.125)),
            vec![],
            vec![],
        );
        let lit2 = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(2.0)),
            vec![],
            vec![],
        );

        // Create groups at ORIGINAL size first, then we'll split them.
        // g0: ReduceSum, reads from input with stride=8 (each output sums 8 elements)
        let g0 = g.push_group(
            n_original,
            NumericDType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: 8,
                reduce_stride: 1,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(inp, 8)],
        );
        let g0_original_base = g0;

        // g1: g0 * (1/8)
        let g1 = g.push_group(
            n_original,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(g0, 1), InputRef::Broadcast(lit_inv)],
        );
        let g1_original_base = g1;

        // g2: g1 + 2.0
        let g2 = g.push_group(
            n_original,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(g1, 1), InputRef::Broadcast(lit2)],
        );

        // g3: sqrt(g2)
        let g3 = g.push_group(
            n_original,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Sqrt,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(g2, 1)],
        );

        // Now simulate splitting: modify groups to be the second portion.
        // This mimics what the partitioner does.
        {
            let groups = g.groups_mut();
            for group in groups.iter_mut() {
                if group.count == n_original {
                    // Split: keep only the second portion
                    group.base_id = AtomId(group.base_id.0 + split_offset);
                    group.count = split_count;
                    group.atom_offset = split_offset;
                    // InputRefs stay the same (reference original bases)
                }
            }
        }

        let outputs = vec![AtomRange {
            base: AtomId(g3.0 + split_offset),
            count: split_count,
            dtype: NumericDType::F32,
        }];
        let layout = compute_layout(&g, &outputs);

        // Check fusion chains
        let chains = build_fusion_chains(g.groups(), &layout);
        for chain in &chains {
            if chain.group_indices.len() > 1 {
                eprintln!(
                    "  split_test: chain len={} offset={} count={}",
                    chain.group_indices.len(),
                    chain.atom_offset,
                    chain.count
                );
            }
        }

        // Compile WITH fusion
        let (compiled_fused, tables_fused) = compile_span(&g, &layout).unwrap();
        let mut buf_fused = vec![0u8; layout.total_bytes];
        layout.populate_literals(&g, &mut buf_fused);
        let input_data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.1 + 1.0).collect();
        layout.write_f32_input(inp, &input_data, &mut buf_fused);
        compiled_fused.execute(&mut buf_fused);
        let result_fused = layout.read_f32_output(&outputs[0], &buf_fused);

        // Compile WITHOUT fusion
        // Disable fusion for unfused comparison.
        unsafe {
            std::env::set_var("FUSION", "0");
        }
        let (compiled_unfused, tables_unfused) = compile_span(&g, &layout).unwrap();
        let mut buf_unfused = vec![0u8; layout.total_bytes];
        layout.populate_literals(&g, &mut buf_unfused);
        layout.write_f32_input(inp, &input_data, &mut buf_unfused);
        compiled_unfused.execute(&mut buf_unfused);
        let result_unfused = layout.read_f32_output(&outputs[0], &buf_unfused);

        eprintln!("  fused:   {:?}", result_fused);
        eprintln!("  unfused: {:?}", result_unfused);

        for (i, (f, u)) in result_fused.iter().zip(result_unfused.iter()).enumerate() {
            assert!(
                (f - u).abs() < 1e-6,
                "split_group mismatch at [{}]: fused={}, unfused={}",
                i,
                f,
                u
            );
        }
    }

    /// 6-group chain with BF16 intermediate dtypes — tests store-load
    /// roundtrip precision in forwarded values.
    #[test]
    fn test_fusion_chain_bf16_roundtrip() {
        // Fusion is enabled by default; no env var needed.

        let mut g = NanoGraph::new();
        let n = 8u64;
        let inp = g.add_input_tensor(GlobalId(0), n, NumericDType::F32);

        let lit2 = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(2.0)),
            vec![],
            vec![],
        );

        // A: identity F32 → BF16 cast
        let a = g.push_group(
            n,
            NumericDType::BF16,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::affine(inp, 1)],
        );
        // B: identity BF16 → F32 cast
        let b = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::affine(a, 1)],
        );
        // C: B + 2.0
        let c = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(b, 1), InputRef::Broadcast(lit2)],
        );
        // D: cast F32 → BF16
        let d = g.push_group(
            n,
            NumericDType::BF16,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::affine(c, 1)],
        );
        // E: cast BF16 → F32
        let e_group = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::affine(d, 1)],
        );
        // F: E * 2.0
        let f = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(e_group, 1), InputRef::Broadcast(lit2)],
        );

        let outputs = vec![AtomRange {
            base: f,
            count: n,
            dtype: NumericDType::F32,
        }];
        let layout = compute_layout(&g, &outputs);

        // Compare fused vs unfused results
        let (compiled_fused, tables_fused) = compile_span(&g, &layout).unwrap();

        let mut buf_fused = vec![0u8; layout.total_bytes];
        layout.populate_literals(&g, &mut buf_fused);
        let input_data: Vec<f32> = vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8];
        layout.write_f32_input(inp, &input_data, &mut buf_fused);
        compiled_fused.execute(&mut buf_fused);
        let result_fused = layout.read_f32_output(&outputs[0], &buf_fused);

        // Now compile without fusion and compare
        // Disable fusion for unfused comparison.
        unsafe {
            std::env::set_var("FUSION", "0");
        }
        let (compiled_unfused, tables_unfused) = compile_span(&g, &layout).unwrap();
        let mut buf_unfused = vec![0u8; layout.total_bytes];
        layout.populate_literals(&g, &mut buf_unfused);
        layout.write_f32_input(inp, &input_data, &mut buf_unfused);
        compiled_unfused.execute(&mut buf_unfused);
        let result_unfused = layout.read_f32_output(&outputs[0], &buf_unfused);

        for (i, (fused, unfused)) in result_fused.iter().zip(result_unfused.iter()).enumerate() {
            assert_eq!(
                fused, unfused,
                "fused/unfused mismatch at [{}]: fused={}, unfused={}",
                i, fused, unfused
            );
        }
    }
}
