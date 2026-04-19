//! Address computation: `InputRef` → bit offset register.
//!
//! Translates a NanoGraph `InputRef` (Broadcast, Strided, Explicit)
//! into the bit offset within the compiled span's buffer where the
//! referenced atom's bits live. Used by [`super::group`] to feed
//! [`super::super::codec::bit_io`] reads.
//!
//! # Output
//!
//! `emit_compute_bit_offset` emits assembly that materializes the bit
//! offset into a caller-supplied destination register, and returns
//! the storage dtype + bit width the caller should pass to the codec
//! `emit_load_bits` / `emit_decode` calls.
//!
//! The bit offset is **relative to the buffer base**, not a pointer.
//! `bit_io` adds it to the buffer base register (`r12`) internally.
//!
//! # InputRef coverage
//!
//! All three InputRef variants are handled:
//!
//! - `Broadcast` — single source atom, always a constant offset.
//! - `Strided` 1D affine — `base + dim_strides[0] * i`.
//! - `Strided` N-d — decomposes the flat index into per-dimension
//!   coordinates (innermost first), then accumulates
//!   `coord[d] * dim_strides[d] * bit_stride`. Power-of-2 moduli
//!   use `and` + `shr`; general moduli use x86 `div`.
//! - `Explicit` single — equivalent to `Broadcast`.
//! - `Explicit` multi — builds a heap-resident bit-offset lookup
//!   table whose pointer is embedded as imm64 in the JIT code.
//!
//! # Extra register clobbering (N-d Strided, `IterVar::Reg`)
//!
//! The N-d Strided path with a runtime iteration variable uses `rax`
//! (0) and `rdx` (2) for the x86 `div` instruction in addition to
//! `dst_bit_reg` and `scratch_reg`. All four plus `iter_reg` must be
//! pairwise distinct. The 1D path and all `IterVar::Const` paths do
//! **not** clobber `rax` / `rdx`.

use dynasmrt::x64::Assembler;
use dynasmrt::{DynasmApi, dynasm};

use crate::compiler::attempts::v14::layout::{BufferLayout, SlotInfo, strided_resolve_offset};
use crate::nano_graph::pattern::{AtomId, GraphConstantId, InputRef, SymDimMap};
use crate::numeric_dtype::NumericDType;

/// Whether a slot qualifies for the byte-aligned fast path.
///
/// True when `bit_offset` and `bit_stride` are multiples of 8 and
/// `elem_bits` is a power-of-2 byte width (8, 16, 32, 64). In this
/// case, the address layer emits **byte** offsets and the codec layer
/// can use direct `mov` instructions instead of bit extraction.
#[inline]
fn slot_is_byte_fast(slot: &SlotInfo) -> bool {
    slot.is_byte_aligned() && matches!(slot.elem_bits, 8 | 16 | 32 | 64)
}

/// Owns lookup tables for multi-entry `Explicit` InputRefs.
///
/// Each table is a boxed slice of `i64` bit offsets — one per Explicit
/// entry. The JIT code embeds each table's raw pointer as an imm64,
/// so the table memory must stay at the same address for the lifetime
/// of the compiled function.
pub struct AddressTables {
    tables: Vec<Box<[i64]>>,
}

impl AddressTables {
    pub fn new() -> Self {
        Self { tables: Vec::new() }
    }

    /// Allocate a bit-offset lookup table and return a raw pointer the
    /// JIT can embed. The caller must keep this `AddressTables` alive
    /// for as long as the JIT function executes.
    pub fn alloc_bit_offset_table(&mut self, bit_offsets: Vec<i64>) -> *const i64 {
        let boxed: Box<[i64]> = bit_offsets.into_boxed_slice();
        let ptr = boxed.as_ptr();
        self.tables.push(boxed);
        ptr
    }
}

/// Where the per-element iteration variable comes from when computing
/// an address.
#[derive(Clone, Copy)]
pub enum IterVar {
    /// The iteration variable is a compile-time constant. Used for
    /// `count == 1` groups and any unrolled access.
    Const(u64),
    /// The iteration variable is held in a runtime GP register.
    /// Used inside a `count > 1` loop.
    Reg(u8),
}

/// Per-input sym-dim remap descriptor.
///
/// Present in [`SymCtx`] when the input's `sym_dim_map` is not the
/// trivial all-`Identity(j==j)` case — i.e., when the producer's
/// sym flat index differs from the consumer's `sym_i` and we must
/// decompose + reassemble at runtime.
///
/// The consumer_sym_dims slice tells the decomposer which runtime
/// extents (via `gc_values`) to divide by. The sym_dim_map tells the
/// reassembler which producer axis each consumer coord contributes
/// to (`Identity(p)`) or skips (`Broadcast`).
#[derive(Clone, Copy, Debug)]
pub struct SymRemap<'a> {
    pub consumer_sym_dims: &'a [GraphConstantId],
    pub sym_dim_map: &'a [SymDimMap],
}

/// Per-call sym-loop context threaded through the address layer.
///
/// When [`emit_compute_bit_offset`] receives `Some(SymCtx)`, it
/// adds `sym_i * info.elem_bits` (or `info.elem_bits / 8` for the
/// byte-aligned fast path) to the atom bit offset after the atom
/// term is computed. `sym_i` is loaded from `[rsp + sym_i_rsp_off]`
/// — the caller (the atom-body loop skeleton in `orch::group`)
/// reserves that stack slot and increments it once per inner sym
/// iteration. `scratch_reg` is reused for the `mov`/`imul`
/// sequence, so it must still be dead at the post-dispatch point
/// (it is, at every existing call site).
///
/// When `remap` is `Some`, the producer's sym flat index is computed
/// from consumer sym_i via coord decomposition + producer-axis
/// reassembly (see [`SymRemap`]). Otherwise the producer flat index
/// equals consumer sym_i directly (all-`Identity(j==j)` case).
///
/// Sym-free groups pass `None` — no sym term is emitted and the
/// address layer behaves identically to the pre-sym version.
#[derive(Clone, Copy, Debug)]
pub struct SymCtx<'a> {
    /// Stack offset (relative to the current `rsp`) where the inner
    /// sym loop stores its `sym_i` counter as a `u64`.
    pub sym_i_rsp_off: i32,
    /// When false, [`apply_sym_offset`] skips emitting the
    /// `sym_i * elem_bits` addend. Used by SymReduce where the
    /// caller steps through producer sym axes explicitly and the
    /// automatic "consumer sym_i → producer offset" addition would be
    /// wrong. `emit_runtime_sym_prod`'s stack-offset math still honors
    /// `sym_i_rsp_off` so gc_values lookups land correctly even when
    /// the caller is inside `emit_atom_body_loop`'s reserve.
    pub emit_sym_term: bool,
    /// Optional non-identity sym_dim_map remap. When `Some`, the
    /// sym term computation walks consumer coords + producer axes
    /// at runtime instead of using `sym_i` directly.
    pub remap: Option<SymRemap<'a>>,
}

impl<'a> SymCtx<'a> {
    /// Standard consumer sym_ctx: both the sym_i address term AND
    /// the gc_values rsp-delta lookup use `sym_i_rsp_off`. No remap —
    /// producer's sym flat equals consumer sym_i directly.
    pub fn new(sym_i_rsp_off: i32) -> Self {
        Self {
            sym_i_rsp_off,
            emit_sym_term: true,
            remap: None,
        }
    }

    /// Address-only sym_ctx: propagates the rsp-delta for gc_values
    /// lookups without auto-applying `sym_i * elem_bits` at the end
    /// of address compute. Callers that handle sym stepping manually
    /// (e.g. SymReduce) use this when computing the producer's
    /// sym=0 atom base.
    pub fn address_only(sym_i_rsp_off: i32) -> Self {
        Self {
            sym_i_rsp_off,
            emit_sym_term: false,
            remap: None,
        }
    }

    /// Attach a non-trivial `sym_dim_map` remap. Used by
    /// [`input_sym_ctx`] when the consumer/producer sym axes aren't
    /// 1:1 aligned.
    pub fn with_remap(mut self, remap: SymRemap<'a>) -> Self {
        self.remap = Some(remap);
        self
    }
}

/// Return the effective sym_ctx to use at a specific input's
/// address-compute site, given the outer consumer sym_ctx (from the
/// atom-body loop), the consumer group's sym_dims, and the input's
/// `sym_dim_map`.
///
/// - Empty `sym_dim_map` → input has no sym component at all; return
///   `None`.
/// - All-`Broadcast` map → producer has no sym axes that vary with
///   the consumer; the same producer atom is read for every
///   `(atom_i, sym_flat)` slot; return `None` so the address layer
///   doesn't add a `sym_i * elem_bits` term.
/// - All-`Identity(j == j)` map → return `outer` unchanged (fast path:
///   producer sym flat equals consumer sym_i).
/// - Any other shape (mixed Identity + Broadcast, or non-trivial
///   Identity(k) indices) → return `outer` with a [`SymRemap`]
///   attached so `apply_sym_offset_pub` emits a runtime decompose +
///   reassemble sequence.
pub fn input_sym_ctx<'a>(
    outer: Option<SymCtx<'a>>,
    consumer_sym_dims: &'a [GraphConstantId],
    sym_dim_map: &'a [SymDimMap],
) -> Option<SymCtx<'a>> {
    if sym_dim_map.is_empty() {
        return None;
    }
    if sym_dim_map
        .iter()
        .all(|m| matches!(m, SymDimMap::Broadcast))
    {
        return None;
    }
    let all_identity_same_index = sym_dim_map
        .iter()
        .enumerate()
        .all(|(j, m)| matches!(m, SymDimMap::Identity(k) if *k == j));
    let outer = outer?;
    if all_identity_same_index {
        return Some(outer);
    }
    Some(outer.with_remap(SymRemap {
        consumer_sym_dims,
        sym_dim_map,
    }))
}

/// Information returned by [`emit_compute_bit_offset`] so the caller
/// knows how to read the bits the offset addresses.
#[derive(Clone, Copy, Debug)]
pub struct AddressInfo<'a> {
    /// Storage dtype of the slot the offset addresses.
    pub dtype: NumericDType,
    /// Number of bits per element, equal to `slot.elem_bits`.
    pub n_bits: u32,
    /// Which buffer the resolved slot lives in.
    pub buffer_id: u8,
    /// When true, the emitted offset register holds a **byte** offset
    /// instead of a bit offset, and the element width is a power-of-2
    /// number of bytes (1, 2, 4, or 8). The caller can use the
    /// byte-aligned load/store fast path in `bit_io`.
    pub byte_aligned: bool,
    /// Producer's sym axes (from `slot.sym_dims`). Empty for sym-free
    /// slots. Needed by [`apply_sym_offset_pub`] to emit the general
    /// `sym_dim_map` remap when the consumer/producer sym axes aren't
    /// 1:1 aligned.
    pub producer_sym_dims: &'a [GraphConstantId],
}

/// Emit code that materializes the bit offset of `input.resolve(i)`
/// (relative to the buffer base) into `dst_bit_reg`.
///
/// # Register usage
///
/// - `dst_bit_reg`: GP register that receives the bit offset.
///   **Written.**
/// - `scratch_reg`: GP register used as temporary. **Clobbered**
///   for runtime arithmetic; left untouched if the result is a
///   constant.
/// - For [`IterVar::Reg`]: the iter register is **read** but not
///   modified.
///
/// `dst_bit_reg`, `scratch_reg`, and the iter register (if `Reg`)
/// must be pairwise distinct.
///
/// For N-d Strided with [`IterVar::Reg`], `rax` (0) and `rdx` (2)
/// are additionally clobbered — see the module-level doc comment.
/// `dst_bit_reg`, `scratch_reg`, and `iter_reg` must all be distinct
/// from `rax` and `rdx` in that case.
///
/// # Errors
///
/// Returns `Err` for:
/// - Missing slot in the layout (typically a layout bug)
/// - Register-aliasing violations
/// - Negative absolute bit offsets (typically an InputRef stride bug)
#[allow(clippy::too_many_arguments)]
pub fn emit_compute_bit_offset<'a>(
    asm: &mut Assembler,
    layout: &'a BufferLayout,
    input: &InputRef,
    iter: IterVar,
    atom_offset: u64,
    dst_bit_reg: u8,
    scratch_reg: u8,
    sym_ctx: Option<SymCtx>,
    tables: &mut AddressTables,
) -> Result<AddressInfo<'a>, String> {
    let info = match input {
        InputRef::Broadcast(atom_id) => {
            emit_constant_atom(asm, layout, *atom_id, dst_bit_reg, scratch_reg, sym_ctx)?
        }

        InputRef::Explicit(ids) if ids.len() <= 1 => {
            if ids.is_empty() {
                return Err("address: empty Explicit InputRef".to_string());
            }
            emit_constant_atom(asm, layout, ids[0], dst_bit_reg, scratch_reg, sym_ctx)?
        }

        InputRef::Explicit(ids) => emit_explicit_multi(
            asm,
            layout,
            ids,
            iter,
            dst_bit_reg,
            scratch_reg,
            tables,
            sym_ctx,
        )?,

        InputRef::Strided {
            base,
            dim_strides,
            dim_shape,
        } if dim_strides.len() == 1 => emit_strided_1d(
            asm,
            layout,
            *base,
            dim_strides[0],
            iter,
            atom_offset,
            dst_bit_reg,
            scratch_reg,
            sym_ctx,
        )?,

        InputRef::Strided {
            base,
            dim_strides,
            dim_shape,
        } => emit_strided_nd(
            asm,
            layout,
            *base,
            dim_strides,
            dim_shape,
            iter,
            atom_offset,
            dst_bit_reg,
            scratch_reg,
            sym_ctx,
        )?,
    };

    apply_sym_offset(
        asm,
        sym_ctx,
        &info,
        dst_bit_reg,
        scratch_reg,
        &layout.buffer_bases,
    )?;
    Ok(info)
}

/// Add `sym_i * step` to `dst_bit_reg` when a sym loop is active.
///
/// `step` is `info.n_bits` (bit-mode offsets) or `info.n_bits / 8`
/// (byte-aligned fast-path offsets). `sym_i` is loaded from
/// `[rsp + ctx.sym_i_rsp_off]`; `scratch_reg` is clobbered but was
/// already live as a scratch in the preceding address computation
/// and is dead at every caller's post-return point.
///
/// No-op when `sym_ctx` is `None` (sym-free groups); this keeps the
/// historical sym-free code path identical at asm level.
fn apply_sym_offset(
    asm: &mut Assembler,
    sym_ctx: Option<SymCtx>,
    info: &AddressInfo,
    dst_bit_reg: u8,
    scratch_reg: u8,
    bases: &crate::compiler::attempts::v14::layout::BufferBases,
) -> Result<(), String> {
    apply_sym_offset_pub(asm, sym_ctx, info, dst_bit_reg, scratch_reg, bases)
}

/// Module-visible wrapper so `orch::group::emit_output_bit_offset`
/// (which lives in a sibling module but computes an address outside
/// the `emit_compute_bit_offset` dispatcher) can reuse the same
/// sym-offset emission without duplicating the code.
pub(super) fn apply_sym_offset_pub(
    asm: &mut Assembler,
    sym_ctx: Option<SymCtx>,
    info: &AddressInfo,
    dst_bit_reg: u8,
    scratch_reg: u8,
    bases: &crate::compiler::attempts::v14::layout::BufferBases,
) -> Result<(), String> {
    let Some(ctx) = sym_ctx else {
        return Ok(());
    };
    if !ctx.emit_sym_term {
        return Ok(());
    }
    let step: i32 = if info.byte_aligned {
        if info.n_bits % 8 != 0 {
            return Err(format!(
                "address: byte_aligned slot with n_bits={} not /8 — sym offset \
                 would desync against caller's byte-offset expectation",
                info.n_bits
            ));
        }
        (info.n_bits / 8) as i32
    } else {
        info.n_bits as i32
    };
    if step == 0 {
        return Ok(());
    }

    // Non-identity sym_dim_map: compute producer's sym flat index from
    // consumer sym_i by decomposing into consumer coords and reassembling
    // along producer axes. See [`emit_producer_sym_flat_remap`] for the
    // asm shape. Trivial (all-Identity-same-index) maps skip this path.
    if let Some(remap) = ctx.remap {
        return emit_producer_sym_flat_remap(
            asm,
            ctx,
            remap,
            info.producer_sym_dims,
            step,
            dst_bit_reg,
            scratch_reg,
            bases,
        );
    }

    dynasm!(asm
        ; .arch x64
        ; mov Rq(scratch_reg), QWORD [rsp + ctx.sym_i_rsp_off]
        ; imul Rq(scratch_reg), Rq(scratch_reg), step
        ; add Rq(dst_bit_reg), Rq(scratch_reg)
    );
    Ok(())
}

/// Emit the general `sym_dim_map` remap: compute
/// `producer_sym_flat * step` and add to `dst_bit_reg`.
///
/// # Algorithm
///
/// Walks consumer sym axes innermost→outermost. At each axis:
///
/// 1. **Decompose** consumer `sym_i` into a coord: for `d > 0`, `div`
///    by the runtime extent `gc_values[consumer_sym_dims[d]]`; for the
///    outermost axis (`d == 0`) the remaining register IS the coord.
/// 2. **Reassemble** when `sym_dim_map[d] == Identity(p)`: fold the
///    stride for any producer axes strictly between `p` and the
///    previously processed producer index into the running stride,
///    then `accum += coord * stride` and `stride *= producer_extents[p]`.
///
/// At the end, `accum = producer_sym_flat`. Multiply by `step` and
/// add to `dst_bit_reg`.
///
/// # Register clobbers
///
/// - `rax`, `rdx` — x86 `div` dividend/remainder. Callers must not
///   have live values in these registers across the call. (The input
///   address compute site is inside a per-input sequence where `rax`
///   is either about to be overwritten by a load or was already
///   stashed to xmm via the int-compute pattern in
///   `emit_binary_compute`, matching the existing N-d Strided
///   clobber discipline.)
/// - `scratch_reg` — consumed as a general scratch.
/// - `dst_bit_reg` — preserved across decomposition, only updated at
///   the final `add`.
///
/// # Stack
///
/// Reserves 16 bytes at entry, releases at exit:
/// - `[rsp + 0]` — producer_sym_flat accumulator
/// - `[rsp + 8]` — producer stride (running fold over producer
///   extents from innermost upward)
///
/// The 16-byte push shifts all existing rsp-based offsets (sym_i,
/// gc_values) by +16 for the duration of this call.
#[allow(clippy::too_many_arguments)]
fn emit_producer_sym_flat_remap(
    asm: &mut Assembler,
    ctx: SymCtx,
    remap: SymRemap,
    producer_sym_dims: &[GraphConstantId],
    step: i32,
    dst_bit_reg: u8,
    scratch_reg: u8,
    bases: &crate::compiler::attempts::v14::layout::BufferBases,
) -> Result<(), String> {
    const RAX: u8 = 0;
    const RDX: u8 = 2;

    if dst_bit_reg == RAX || dst_bit_reg == RDX {
        return Err(format!(
            "address: emit_producer_sym_flat_remap dst_bit_reg={dst_bit_reg} \
             aliases rax/rdx (used by div)"
        ));
    }
    if scratch_reg == RAX || scratch_reg == RDX {
        return Err(format!(
            "address: emit_producer_sym_flat_remap scratch_reg={scratch_reg} \
             aliases rax/rdx (used by div)"
        ));
    }
    if remap.sym_dim_map.len() != remap.consumer_sym_dims.len() {
        return Err(format!(
            "address: sym_dim_map.len()={} != consumer_sym_dims.len()={}",
            remap.sym_dim_map.len(),
            remap.consumer_sym_dims.len()
        ));
    }

    let gc_elem_off = |gc: GraphConstantId| -> Result<i32, String> {
        (gc.0 as i64)
            .checked_mul(8)
            .and_then(|v| i32::try_from(v).ok())
            .ok_or_else(|| format!("address: gc index {} overflow", gc.0))
    };

    // rsp shifts by -16 during this sequence. Pre-compute the adjusted
    // stack offsets from the shifted rsp:
    // - sym_i_off: ctx.sym_i_rsp_off measured from the pre-shift rsp,
    //   so after our `sub rsp, 16` the offset grows by 16.
    // - gc_values_off: the `emit_runtime_sym_prod` formula —
    //   `gc_values_stack_offset + (total rsp shift from post-prologue)`.
    //   Post-prologue → body-loop → ours adds two 16-byte shifts on top
    //   of any ctx.sym_i_rsp_off the caller already saw.
    let sym_i_off = ctx
        .sym_i_rsp_off
        .checked_add(16)
        .ok_or_else(|| format!("address: sym_i_rsp_off {} + 16 overflow", ctx.sym_i_rsp_off))?;
    let gc_values_off = super::super::prologue::gc_values_stack_offset(bases)
        .checked_add(ctx.sym_i_rsp_off)
        .and_then(|v| v.checked_add(32))
        .ok_or_else(|| "address: gc_values offset overflow in remap".to_string())?;

    // Reserve: [rsp+0] = accumulator, [rsp+8] = stride
    dynasm!(asm
        ; .arch x64
        ; sub rsp, 16
        ; mov QWORD [rsp + 0], 0
        ; mov QWORD [rsp + 8], 1
        // rax = remaining (starts as consumer sym_i)
        ; mov Rq(RAX), QWORD [rsp + sym_i_off]
    );

    // Walk consumer axes innermost → outermost. `next_p` tracks how far
    // we've folded the producer stride from the innermost axis upward.
    let n = remap.consumer_sym_dims.len();
    let m = producer_sym_dims.len();
    let mut next_p = m;

    for d in (0..n).rev() {
        // Get the consumer coord for axis d into rdx (or copy from rax
        // for d=0).
        // For d > 0: div by consumer extent.
        //   xor rdx, rdx  (via Rq(2), Rq(2))
        //   mov scratch, [rsp + gc_values_off]
        //   mov scratch, [scratch + consumer_sym_dims[d] * 8]
        //   div scratch
        //   (rdx = coord = remaining % extent, rax = new remaining)
        // For d == 0: coord = rax (no div since remaining < outermost_extent).
        //   mov rdx, rax
        let coord_in_rdx = d > 0;
        if coord_in_rdx {
            let off = gc_elem_off(remap.consumer_sym_dims[d])?;
            dynasm!(asm
                ; .arch x64
                ; xor Rq(RDX), Rq(RDX)
                ; mov Rq(scratch_reg), QWORD [rsp + gc_values_off]
                ; mov Rq(scratch_reg), QWORD [Rq(scratch_reg) + off]
                ; div Rq(scratch_reg)
            );
        } else {
            dynasm!(asm; .arch x64; mov Rq(RDX), Rq(RAX));
        }

        let SymDimMap::Identity(p) = remap.sym_dim_map[d] else {
            continue;
        };
        if p >= m {
            return Err(format!(
                "address: sym_dim_map[{d}] = Identity({p}) out of range \
                 for producer with {m} sym_dims"
            ));
        }

        // Fold producer_extents[q] into the stride for q in (p..next_p-1],
        // i.e. producer axes strictly between p and our previous position
        // that were NOT mapped by any consumer axis. (Unmapped producer
        // axes contribute 0 to producer_coords but still contribute to
        // stride for outer axes.)
        while next_p > p + 1 {
            next_p -= 1;
            let off = gc_elem_off(producer_sym_dims[next_p])?;
            dynasm!(asm
                ; .arch x64
                ; mov Rq(scratch_reg), QWORD [rsp + gc_values_off]
                ; mov Rq(scratch_reg), QWORD [Rq(scratch_reg) + off]
                ; imul Rq(scratch_reg), QWORD [rsp + 8]
                ; mov QWORD [rsp + 8], Rq(scratch_reg)
            );
        }
        // accum += coord * stride. Coord is in rdx (copied from rax for d=0).
        dynasm!(asm
            ; .arch x64
            ; imul Rq(RDX), QWORD [rsp + 8]
            ; add QWORD [rsp + 0], Rq(RDX)
        );
        // stride *= producer_extents[p]
        let off = gc_elem_off(producer_sym_dims[p])?;
        dynasm!(asm
            ; .arch x64
            ; mov Rq(scratch_reg), QWORD [rsp + gc_values_off]
            ; mov Rq(scratch_reg), QWORD [Rq(scratch_reg) + off]
            ; imul Rq(scratch_reg), QWORD [rsp + 8]
            ; mov QWORD [rsp + 8], Rq(scratch_reg)
        );
        next_p = p;
    }

    // accum (= producer_sym_flat) is at [rsp+0]. Multiply by step and
    // fold into dst_bit_reg.
    dynasm!(asm
        ; .arch x64
        ; mov Rq(scratch_reg), QWORD [rsp + 0]
        ; add rsp, 16
        ; imul Rq(scratch_reg), Rq(scratch_reg), step
        ; add Rq(dst_bit_reg), Rq(scratch_reg)
    );
    Ok(())
}

/// Emit a constant bit-offset materialization for a single atom.
/// Used by `Broadcast` and single-element `Explicit`.
fn emit_constant_atom<'a>(
    asm: &mut Assembler,
    layout: &'a BufferLayout,
    atom_id: AtomId,
    dst_bit_reg: u8,
    scratch_reg: u8,
    sym_ctx: Option<SymCtx>,
) -> Result<AddressInfo<'a>, String> {
    let (slot, elem_idx) = layout
        .find(atom_id)
        .ok_or_else(|| format!("address: no slot for atom={atom_id}"))?;
    let byte_fast = slot_is_byte_fast(slot);
    let bit_off = slot.bit_offset + elem_idx * slot.bit_stride;
    let compile_off = if byte_fast { bit_off / 8 } else { bit_off };

    if !slot.sym_dims.is_empty() {
        // TAMI-native runtime-tight sym slot: the caller writes at
        // `sym_prod * bpe` per atom, so the compile-time offset
        // (computed assuming tight `bpe` stride) must be multiplied
        // by the runtime sym_prod. Load sym_prod into dst, then
        // `imul dst, dst, compile_off` so the result is the real
        // byte/bit offset.
        emit_runtime_sym_prod(
            asm,
            &layout.buffer_bases,
            &slot.sym_dims,
            dst_bit_reg,
            scratch_reg,
            sym_ctx,
        )?;
        let compile_signed = compile_off as i64;
        if (i32::MIN as i64..=i32::MAX as i64).contains(&compile_signed) {
            dynasm!(asm
                ; .arch x64
                ; imul Rq(dst_bit_reg), Rq(dst_bit_reg), compile_signed as i32
            );
        } else {
            emit_mov_imm64(asm, scratch_reg, compile_off);
            dynasm!(asm; .arch x64; imul Rq(dst_bit_reg), Rq(scratch_reg));
        }
    } else {
        emit_mov_imm64(asm, dst_bit_reg, compile_off);
    }

    Ok(AddressInfo {
        dtype: slot.dtype,
        n_bits: slot.elem_bits as u32,
        buffer_id: slot.buffer_id,
        byte_aligned: byte_fast,
        producer_sym_dims: &slot.sym_dims,
    })
}

/// 1D affine path: `bit_offset(i) = base_bit + bit_stride * i` where
/// `base_bit` is the bit offset of the logical `base` atom in the
/// buffer (back-computed for split groups whose base is outside this
/// span's slot map) and `bit_stride = dim_strides[0] * slot.bit_stride`.
///
/// **Runtime-stride inputs**: when `slot.sym_dims` is non-empty the
/// atom stride is `bit_stride * (∏ gc_values[gc.0])` — the TAMI-native
/// packing for input tensors with sym dims (caller writes
/// atom-major-sym-innermost at runtime `sym_prod * bpe` per atom).
/// The per-atom multiplier is resolved at execute time from the
/// `gc_values` pointer the prologue saved on the stack; the compile
/// path emits a `mov`/`imul` sequence instead of baking `bit_stride`
/// into an `imm32`.
#[allow(clippy::too_many_arguments)]
fn emit_strided_1d<'a>(
    asm: &mut Assembler,
    layout: &'a BufferLayout,
    base: AtomId,
    stride_atoms: i64,
    iter: IterVar,
    atom_offset: u64,
    dst_bit_reg: u8,
    scratch_reg: u8,
    sym_ctx: Option<SymCtx>,
) -> Result<AddressInfo<'a>, String> {
    // Resolve the slot via `base` directly, or fall back to the first
    // accessed atom if `base` itself isn't in the layout (split groups).
    let first_offset_atoms = stride_atoms * atom_offset as i64;
    let first_atom = AtomId(((base.0 as i64) + first_offset_atoms) as u64);

    let (slot, elem_idx) = layout
        .find(base)
        .or_else(|| layout.find(first_atom))
        .ok_or_else(|| {
            format!(
                "address: no slot for Strided base={base} \
                 first_atom={first_atom} (atom_offset={atom_offset}, \
                 stride={stride_atoms})"
            )
        })?;

    // bit offset of the LOGICAL base atom (not the slot's first
    // atom — they differ for split groups).
    let slot_bit = slot.bit_offset + elem_idx * slot.bit_stride;
    let base_bit_signed = if layout.find(base).is_some() {
        slot_bit as i64
    } else {
        // Back-compute: base_bit + first_offset_atoms * slot.bit_stride
        // = slot_bit
        slot_bit as i64 - first_offset_atoms * slot.bit_stride as i64
    };

    // Per-iteration bit stride between consecutive consumer atoms.
    // For TAMI-runtime-stride slots this is the *compile-time* base
    // stride; at emit time the code below multiplies by the runtime
    // sym_prod so the effective atom stride is
    // `stride_atoms * slot.bit_stride * sym_prod`.
    let bit_stride_signed = stride_atoms * slot.bit_stride as i64;

    let byte_fast = slot_is_byte_fast(slot);
    let info = AddressInfo {
        dtype: slot.dtype,
        n_bits: slot.elem_bits as u32,
        buffer_id: slot.buffer_id,
        byte_aligned: byte_fast,
        producer_sym_dims: &slot.sym_dims,
    };

    // When byte-aligned, emit byte offsets (divide by 8 at JIT-build
    // time) so the codec can use direct `mov` instructions.
    let (eff_base, eff_stride) = if byte_fast {
        (base_bit_signed / 8, bit_stride_signed / 8)
    } else {
        (base_bit_signed, bit_stride_signed)
    };

    let runtime_sym = !slot.sym_dims.is_empty();

    match iter {
        IterVar::Const(c) => {
            if runtime_sym {
                // Full runtime byte offset for TAMI-native tight sym
                // layout: sym_prod * (eff_base + c * eff_stride) + sym_i*bpe.
                // The caller writes at runtime stride `sym_prod * bpe`
                // per atom, so the slot's compile-time `eff_base`
                // (computed assuming tight `bpe` stride) and the
                // per-iter `eff_stride` both need the sym_prod factor.
                // Factor sym_prod out so we only need one runtime
                // multiply; eff_base + c*eff_stride is compile-time.
                let c_times_stride = (c as i64).checked_mul(eff_stride).ok_or_else(|| {
                    format!("address: Strided Const overflow c={c} stride={eff_stride}")
                })?;
                let pre_sym = eff_base.checked_add(c_times_stride).ok_or_else(|| {
                    format!(
                        "address: Strided Const eff_base+c*stride overflow \
                         (eff_base={eff_base} c_times_stride={c_times_stride})"
                    )
                })?;
                emit_runtime_sym_prod(
                    asm,
                    &layout.buffer_bases,
                    &slot.sym_dims,
                    dst_bit_reg,
                    scratch_reg,
                    sym_ctx,
                )?;
                if (i32::MIN as i64..=i32::MAX as i64).contains(&pre_sym) {
                    dynasm!(asm
                        ; .arch x64
                        ; imul Rq(dst_bit_reg), Rq(dst_bit_reg), pre_sym as i32
                    );
                } else {
                    emit_mov_imm64(asm, scratch_reg, pre_sym as u64);
                    dynasm!(asm; .arch x64; imul Rq(dst_bit_reg), Rq(scratch_reg));
                }
            } else {
                let abs = eff_base + eff_stride * c as i64;
                if abs < 0 {
                    return Err(format!(
                        "address: negative offset {abs} for Strided base={base} \
                         atom_offset={atom_offset} stride={stride_atoms} c={c}"
                    ));
                }
                emit_mov_imm64(asm, dst_bit_reg, abs as u64);
            }
        }
        IterVar::Reg(iter_reg) => {
            assert_distinct(dst_bit_reg, scratch_reg, iter_reg)?;

            if runtime_sym {
                // dst = sym_prod * (iter_reg * eff_stride + eff_base).
                // See the Const comment above; factored form needs only
                // one runtime multiply and one scratch register.
                // 1. dst = iter_reg
                dynasm!(asm; .arch x64; mov Rq(dst_bit_reg), Rq(iter_reg));
                // 2. dst *= eff_stride (compile-time tight stride)
                if (i32::MIN as i64..=i32::MAX as i64).contains(&eff_stride) {
                    dynasm!(asm
                        ; .arch x64
                        ; imul Rq(dst_bit_reg), Rq(dst_bit_reg), eff_stride as i32
                    );
                } else {
                    emit_mov_imm64(asm, scratch_reg, eff_stride as u64);
                    dynasm!(asm; .arch x64; imul Rq(dst_bit_reg), Rq(scratch_reg));
                }
                // 3. dst += eff_base (compile-time, assumes tight pack)
                if (i32::MIN as i64..=i32::MAX as i64).contains(&eff_base) {
                    dynasm!(asm; .arch x64; add Rq(dst_bit_reg), eff_base as i32);
                } else {
                    emit_mov_imm64(asm, scratch_reg, eff_base as u64);
                    dynasm!(asm; .arch x64; add Rq(dst_bit_reg), Rq(scratch_reg));
                }
                // 4. scratch = runtime sym_prod. For single-sym the
                //    fold uses only `scratch` internally (the mov
                //    sequence consumes and replaces the ptr). For
                //    multi-sym we need a second register to hold the
                //    gc_values ptr across the imul fold — push the
                //    iter*stride+base accumulator to the stack and
                //    reuse dst_bit_reg as the ptr scratch, then
                //    restore it. The `push` shifts rsp by -8, so the
                //    sym_ctx we forward to emit_runtime_sym_prod
                //    advertises `sym_i_rsp_off + 8` to compensate.
                if slot.sym_dims.len() > 1 {
                    dynasm!(asm; .arch x64; push Rq(dst_bit_reg));
                    let shifted_sym_ctx = match sym_ctx {
                        Some(ctx) => Some(SymCtx {
                            sym_i_rsp_off: ctx.sym_i_rsp_off.checked_add(8).ok_or_else(|| {
                                format!(
                                    "address: sym_i_rsp_off {} + 8 overflow in multi-sym Strided Reg push",
                                    ctx.sym_i_rsp_off
                                )
                            })?,
                            ..ctx
                        }),
                        None => None,
                    };
                    emit_runtime_sym_prod(
                        asm,
                        &layout.buffer_bases,
                        &slot.sym_dims,
                        scratch_reg,
                        dst_bit_reg,
                        shifted_sym_ctx,
                    )?;
                    dynasm!(asm; .arch x64; pop Rq(dst_bit_reg));
                } else {
                    emit_runtime_sym_prod(
                        asm,
                        &layout.buffer_bases,
                        &slot.sym_dims,
                        scratch_reg,
                        dst_bit_reg,
                        sym_ctx,
                    )?;
                }
                dynasm!(asm; .arch x64; imul Rq(dst_bit_reg), Rq(scratch_reg));
            } else {
                // dst = iter_reg
                dynasm!(asm
                    ; .arch x64
                    ; mov Rq(dst_bit_reg), Rq(iter_reg)
                );

                // dst *= stride
                if (i32::MIN as i64..=i32::MAX as i64).contains(&eff_stride) {
                    let imm = eff_stride as i32;
                    dynasm!(asm
                        ; imul Rq(dst_bit_reg), Rq(dst_bit_reg), imm
                    );
                } else {
                    emit_mov_imm64(asm, scratch_reg, eff_stride as u64);
                    dynasm!(asm
                        ; imul Rq(dst_bit_reg), Rq(scratch_reg)
                    );
                }

                // dst += base
                if (i32::MIN as i64..=i32::MAX as i64).contains(&eff_base) {
                    let imm = eff_base as i32;
                    dynasm!(asm
                        ; add Rq(dst_bit_reg), imm
                    );
                } else {
                    emit_mov_imm64(asm, scratch_reg, eff_base as u64);
                    dynasm!(asm
                        ; add Rq(dst_bit_reg), Rq(scratch_reg)
                    );
                }
            }
        }
    }

    Ok(info)
}

/// Emit code that loads `∏ gc_values[gc.0]` for `sym_dims` into
/// `dst_reg`. `ptr_scratch_reg` holds the gc_values pointer across the
/// multi-sym fold (ignored when `sym_dims.len() == 1`; the single-sym
/// fast path reuses `dst_reg` as its own ptr scratch since the final
/// `mov` discards it).
///
/// Multi-sym fold pattern:
///
/// ```text
/// mov  ptr_scratch, [rsp + gc_values_off]
/// mov  dst,   [ptr_scratch + sym[0].0 * 8]
/// imul dst,   [ptr_scratch + sym[1].0 * 8]
/// imul dst,   [ptr_scratch + sym[2].0 * 8]
/// ...
/// ```
///
/// `dst_reg` and `ptr_scratch_reg` must be pairwise distinct when
/// `sym_dims.len() > 1`. For single-sym, `ptr_scratch_reg` is unused.
pub(super) fn emit_runtime_sym_prod_pub(
    asm: &mut Assembler,
    bases: &crate::compiler::attempts::v14::layout::BufferBases,
    sym_dims: &[crate::nano_graph::pattern::GraphConstantId],
    dst_reg: u8,
    ptr_scratch_reg: u8,
    sym_ctx: Option<SymCtx>,
) -> Result<(), String> {
    emit_runtime_sym_prod(asm, bases, sym_dims, dst_reg, ptr_scratch_reg, sym_ctx)
}

fn emit_runtime_sym_prod(
    asm: &mut Assembler,
    bases: &crate::compiler::attempts::v14::layout::BufferBases,
    sym_dims: &[crate::nano_graph::pattern::GraphConstantId],
    dst_reg: u8,
    ptr_scratch_reg: u8,
    sym_ctx: Option<SymCtx>,
) -> Result<(), String> {
    if sym_dims.is_empty() {
        return Err(
            "address: emit_runtime_sym_prod called with empty sym_dims (caller bug)".to_string(),
        );
    }
    let rsp_delta: i32 = match sym_ctx {
        Some(ctx) => ctx
            .sym_i_rsp_off
            .checked_add(16)
            .ok_or_else(|| format!("address: sym_i_rsp_off {} + 16 overflow", ctx.sym_i_rsp_off))?,
        None => 0,
    };
    let gc_values_off = super::super::prologue::gc_values_stack_offset(bases)
        .checked_add(rsp_delta)
        .ok_or_else(|| "address: gc_values offset overflow".to_string())?;

    let gc_elem_off = |gc: crate::nano_graph::pattern::GraphConstantId| -> Result<i32, String> {
        (gc.0 as i64)
            .checked_mul(8)
            .and_then(|v| i32::try_from(v).ok())
            .ok_or_else(|| format!("address: gc index {} overflow", gc.0))
    };

    if sym_dims.len() == 1 {
        // Single-sym fast path: no separate ptr scratch needed since
        // we consume the ptr with the first and only indirection.
        let first_off = gc_elem_off(sym_dims[0])?;
        dynasm!(asm
            ; .arch x64
            ; mov Rq(dst_reg), QWORD [rsp + gc_values_off]
            ; mov Rq(dst_reg), QWORD [Rq(dst_reg) + first_off]
        );
        return Ok(());
    }

    if dst_reg == ptr_scratch_reg {
        return Err(format!(
            "address: emit_runtime_sym_prod multi-sym requires distinct \
             dst_reg={dst_reg} and ptr_scratch_reg={ptr_scratch_reg}"
        ));
    }

    let first_off = gc_elem_off(sym_dims[0])?;
    dynasm!(asm
        ; .arch x64
        ; mov Rq(ptr_scratch_reg), QWORD [rsp + gc_values_off]
        ; mov Rq(dst_reg), QWORD [Rq(ptr_scratch_reg) + first_off]
    );
    for gc in &sym_dims[1..] {
        let off = gc_elem_off(*gc)?;
        dynasm!(asm
            ; .arch x64
            ; imul Rq(dst_reg), QWORD [Rq(ptr_scratch_reg) + off]
        );
    }
    Ok(())
}

// ─── N-d Strided ────────────────────────────────────────────────────

/// N-d Strided address computation.
///
/// Decomposes the flat iteration index into per-dimension coordinates
/// (innermost first, matching [`InputRef::resolve`]) and sums
/// `coord[d] * dim_strides[d] * bit_stride` into a total bit offset.
#[allow(clippy::too_many_arguments)]
fn emit_strided_nd<'a>(
    asm: &mut Assembler,
    layout: &'a BufferLayout,
    base: AtomId,
    dim_strides: &[i64],
    dim_shape: &[u64],
    iter: IterVar,
    atom_offset: u64,
    dst_bit_reg: u8,
    scratch_reg: u8,
    _sym_ctx: Option<SymCtx>,
) -> Result<AddressInfo<'a>, String> {
    let nd = dim_strides.len();
    assert!(nd >= 2, "emit_strided_nd called with nd < 2");
    assert_eq!(nd, dim_shape.len(), "dim_strides/dim_shape length mismatch");

    // Resolve the slot via `base` or the first accessed atom.
    let first_offset = strided_resolve_offset(dim_strides, dim_shape, atom_offset);
    let first_atom = AtomId(((base.0 as i64) + first_offset) as u64);

    let (slot, elem_idx) = layout
        .find(base)
        .or_else(|| layout.find(first_atom))
        .ok_or_else(|| {
            format!(
                "address: no slot for Strided n-d base={base} \
                 first_atom={first_atom} (atom_offset={atom_offset})"
            )
        })?;

    // Runtime-stride slots (sym inputs under TAMI-native packing) would
    // need per-dim sym_prod multiplication in the N-d decomposition —
    // not yet wired. Falls back to cranelift for now; if a real case
    // surfaces we'll extend the N-d path similarly to emit_strided_1d.
    if !slot.sym_dims.is_empty() {
        return Err(format!(
            "address: N-d Strided on runtime-sym input slot (sym_dims={:?}) \
             not yet supported",
            slot.sym_dims
        ));
    }

    let slot_bit = slot.bit_offset + elem_idx * slot.bit_stride;
    let base_bit_signed = if layout.find(base).is_some() {
        slot_bit as i64
    } else {
        slot_bit as i64 - first_offset * slot.bit_stride as i64
    };

    let byte_fast = slot_is_byte_fast(slot);
    let info = AddressInfo {
        dtype: slot.dtype,
        n_bits: slot.elem_bits as u32,
        buffer_id: slot.buffer_id,
        byte_aligned: byte_fast,
        producer_sym_dims: &slot.sym_dims,
    };

    let (eff_base, eff_stride) = if byte_fast {
        (base_bit_signed / 8, slot.bit_stride / 8)
    } else {
        (base_bit_signed, slot.bit_stride)
    };

    match iter {
        IterVar::Const(c) => {
            let atom_off = strided_resolve_offset(dim_strides, dim_shape, c);
            let abs = eff_base + atom_off * eff_stride as i64;
            if abs < 0 {
                return Err(format!(
                    "address: negative offset {abs} for n-d Strided \
                     base={base} c={c}"
                ));
            }
            emit_mov_imm64(asm, dst_bit_reg, abs as u64);
        }
        IterVar::Reg(iter_reg) => {
            emit_strided_nd_reg(
                asm,
                eff_base,
                dim_strides,
                dim_shape,
                eff_stride,
                iter_reg,
                dst_bit_reg,
                scratch_reg,
            )?;
        }
    }
    Ok(info)
}

/// `rax` — used as the remaining/quotient register by x86 `div`.
const ND_REMAINING: u8 = 0;
/// `rdx` — used as the remainder/coordinate register by x86 `div`.
const ND_COORD: u8 = 2;

/// Emit the runtime n-d coordinate decomposition loop.
///
/// Uses `rax` (0) as the remaining register and `rdx` (2) as the
/// coordinate register (matching x86 `div rdx:rax / r/m → rax, rdx`).
/// Both are clobbered. `dst_bit_reg`, `scratch_reg`, and `iter_reg`
/// must all be distinct from each other and from `rax`/`rdx`.
#[allow(clippy::too_many_arguments)]
fn emit_strided_nd_reg(
    asm: &mut Assembler,
    base_bit_signed: i64,
    dim_strides: &[i64],
    dim_shape: &[u64],
    bit_stride: u64,
    iter_reg: u8,
    dst_bit_reg: u8,
    scratch_reg: u8,
) -> Result<(), String> {
    let nd = dim_strides.len();

    // Validate: none of the caller's registers may alias rax or rdx.
    for (name, reg) in [
        ("dst_bit_reg", dst_bit_reg),
        ("scratch_reg", scratch_reg),
        ("iter_reg", iter_reg),
    ] {
        if reg == ND_REMAINING {
            return Err(format!(
                "address n-d: {name} ({reg}) aliases rax (used by div)"
            ));
        }
        if reg == ND_COORD {
            return Err(format!(
                "address n-d: {name} ({reg}) aliases rdx (used by div)"
            ));
        }
    }
    assert_distinct(dst_bit_reg, scratch_reg, iter_reg)?;

    // dst = base_bit
    emit_mov_imm64(asm, dst_bit_reg, base_bit_signed as u64);
    // remaining = iter_reg
    dynasm!(asm
        ; .arch x64
        ; mov Rq(ND_REMAINING), Rq(iter_reg)
    );

    // Inner-to-outer decomposition, mirroring InputRef::resolve.
    for d in (0..nd).rev() {
        let stride_bits = dim_strides[d] * bit_stride as i64;

        if d == 0 {
            // Outermost: coord = remaining. No modulus.
            if stride_bits != 0 {
                emit_imul_accum(asm, ND_REMAINING, stride_bits, dst_bit_reg, scratch_reg);
            }
        } else {
            let modulus = dim_shape[d];
            if modulus <= 1 {
                // coord is always 0, remaining unchanged. No-op.
                continue;
            }

            // Decompose: coord = remaining % modulus, remaining /= modulus.
            if modulus.is_power_of_two() {
                let shift = modulus.trailing_zeros() as i8;
                let mask = modulus as i64 - 1;
                // coord = remaining & mask
                dynasm!(asm
                    ; .arch x64
                    ; mov Rq(ND_COORD), Rq(ND_REMAINING)
                );
                if mask <= i32::MAX as i64 {
                    dynasm!(asm
                        ; and Rq(ND_COORD), DWORD mask as i32
                    );
                } else {
                    emit_mov_imm64(asm, scratch_reg, mask as u64);
                    dynasm!(asm
                        ; and Rq(ND_COORD), Rq(scratch_reg)
                    );
                }
                // remaining >>= shift
                dynasm!(asm
                    ; shr Rq(ND_REMAINING), shift
                );
            } else {
                // General: div. rdx:rax / scratch → rax = quot, rdx = rem.
                dynasm!(asm
                    ; .arch x64
                    ; xor Rq(ND_COORD), Rq(ND_COORD)
                );
                emit_mov_imm64(asm, scratch_reg, modulus);
                dynasm!(asm
                    ; div Rq(scratch_reg)
                );
            }

            // Accumulate: dst += coord * stride_bits.
            if stride_bits != 0 {
                emit_imul_accum(asm, ND_COORD, stride_bits, dst_bit_reg, scratch_reg);
            }
        }
    }
    Ok(())
}

/// Emit `dst += src_reg * imm`, using `scratch` for large immediates.
fn emit_imul_accum(asm: &mut Assembler, src_reg: u8, imm: i64, dst_reg: u8, scratch: u8) {
    if imm == 1 {
        dynasm!(asm
            ; .arch x64
            ; add Rq(dst_reg), Rq(src_reg)
        );
    } else if imm == -1 {
        dynasm!(asm
            ; .arch x64
            ; sub Rq(dst_reg), Rq(src_reg)
        );
    } else if (i32::MIN as i64..=i32::MAX as i64).contains(&imm) {
        dynasm!(asm
            ; .arch x64
            ; imul Rq(scratch), Rq(src_reg), imm as i32
            ; add Rq(dst_reg), Rq(scratch)
        );
    } else {
        emit_mov_imm64(asm, scratch, imm as u64);
        dynasm!(asm
            ; .arch x64
            ; imul Rq(scratch), Rq(src_reg)
            ; add Rq(dst_reg), Rq(scratch)
        );
    }
}

// ─── Multi-entry Explicit ───────────────────────────────────────────

/// Multi-entry Explicit address: build a heap-resident lookup table of
/// bit offsets (one `i64` per entry), embed the table pointer as imm64
/// in the JIT, and load `table[i]` at runtime.
#[allow(clippy::too_many_arguments)]
fn emit_explicit_multi<'a>(
    asm: &mut Assembler,
    layout: &'a BufferLayout,
    ids: &[AtomId],
    iter: IterVar,
    dst_bit_reg: u8,
    scratch_reg: u8,
    tables: &mut AddressTables,
    sym_ctx: Option<SymCtx>,
) -> Result<AddressInfo<'a>, String> {
    debug_assert!(ids.len() >= 2);

    // Resolve dtype from the first entry. All entries are expected to
    // share the same dtype (the graph construction enforces this).
    let (first_slot, _) = layout
        .find(ids[0])
        .ok_or_else(|| format!("address: no slot for Explicit[0] atom={}", ids[0]))?;
    let byte_fast = slot_is_byte_fast(first_slot);
    let info = AddressInfo {
        dtype: first_slot.dtype,
        n_bits: first_slot.elem_bits as u32,
        buffer_id: first_slot.buffer_id,
        byte_aligned: byte_fast,
        producer_sym_dims: &first_slot.sym_dims,
    };

    match iter {
        IterVar::Const(c) => {
            let idx = c as usize;
            if idx >= ids.len() {
                return Err(format!(
                    "address: Explicit const index {c} out of bounds (len={})",
                    ids.len()
                ));
            }
            // Delegate to the single-atom path — just a constant mov.
            // Override info with what emit_constant_atom returns (it
            // sets byte_aligned consistently).
            return emit_constant_atom(asm, layout, ids[idx], dst_bit_reg, scratch_reg, sym_ctx);
        }
        IterVar::Reg(iter_reg) => {
            // Build the offset lookup table. When byte-aligned, store
            // byte offsets; otherwise bit offsets. Compile-time offsets
            // assume tight `bpe` stride; for runtime-sym slots we apply
            // an `imul dst, sym_prod` after the table load so the real
            // byte offset follows the caller's `sym_prod * bpe` packing.
            let mut shared_sym_dims: Option<Vec<crate::nano_graph::pattern::GraphConstantId>> =
                None;
            let offsets: Vec<i64> = ids
                .iter()
                .enumerate()
                .map(|(i, id)| {
                    let (slot, elem_idx) = layout
                        .find(*id)
                        .ok_or_else(|| format!("address: no slot for Explicit[{i}] atom={id}"))?;
                    if i == 0 {
                        shared_sym_dims = Some(slot.sym_dims.clone());
                    } else if shared_sym_dims.as_ref() != Some(&slot.sym_dims) {
                        return Err(format!(
                            "address: Explicit multi atoms span slots with \
                             heterogeneous sym_dims (first={:?}, [{i}]={:?}) — \
                             runtime sym_prod mul would apply inconsistently",
                            shared_sym_dims, slot.sym_dims
                        ));
                    }
                    let bit_off = (slot.bit_offset + elem_idx * slot.bit_stride) as i64;
                    Ok(if byte_fast { bit_off / 8 } else { bit_off })
                })
                .collect::<Result<Vec<_>, String>>()?;

            let table_ptr = tables.alloc_bit_offset_table(offsets);

            // dst = table[iter_reg]
            dynasm!(asm
                ; .arch x64
                ; mov Rq(scratch_reg), QWORD table_ptr as i64
                ; mov Rq(dst_bit_reg), QWORD [Rq(scratch_reg) + Rq(iter_reg) * 8]
            );

            if let Some(sd) = shared_sym_dims.as_ref() {
                if !sd.is_empty() {
                    if sd.len() > 1 {
                        // Multi-sym fold: dst_bit_reg holds the
                        // table-loaded offset and must survive the
                        // mul. Spill to stack, reuse dst_bit_reg as
                        // ptr scratch inside emit_runtime_sym_prod,
                        // restore.
                        dynasm!(asm; .arch x64; push Rq(dst_bit_reg));
                        let shifted_sym_ctx = match sym_ctx {
                            Some(ctx) => Some(SymCtx {
                                sym_i_rsp_off: ctx.sym_i_rsp_off.checked_add(8).ok_or_else(
                                    || {
                                        format!(
                                            "address: sym_i_rsp_off {} + 8 overflow in multi-sym Explicit-multi push",
                                            ctx.sym_i_rsp_off
                                        )
                                    },
                                )?,
                                ..ctx
                            }),
                            None => None,
                        };
                        emit_runtime_sym_prod(
                            asm,
                            &layout.buffer_bases,
                            sd,
                            scratch_reg,
                            dst_bit_reg,
                            shifted_sym_ctx,
                        )?;
                        dynasm!(asm; .arch x64; pop Rq(dst_bit_reg));
                    } else {
                        emit_runtime_sym_prod(
                            asm,
                            &layout.buffer_bases,
                            sd,
                            scratch_reg,
                            dst_bit_reg,
                            sym_ctx,
                        )?;
                    }
                    dynasm!(asm; .arch x64; imul Rq(dst_bit_reg), Rq(scratch_reg));
                }
            }
        }
    }
    Ok(info)
}

// ─── Helpers ────────────────────────────────────────────────────────

/// Emit `mov dst_reg, imm64`. Always emits the full 10-byte form
/// — codegen optimization (xor for zero, mov-imm32 for small
/// constants) is not worth the complexity at this layer; the linker
/// has nothing to do with our bytes anyway.
fn emit_mov_imm64(asm: &mut Assembler, dst_reg: u8, imm: u64) {
    dynasm!(asm
        ; .arch x64
        ; mov Rq(dst_reg), QWORD imm as i64
    );
}

/// Three-way distinctness check used by the variable-iter Strided
/// path. Returns a typed `Err` describing the alias on failure.
fn assert_distinct(a: u8, b: u8, c: u8) -> Result<(), String> {
    if a == b {
        return Err(format!(
            "address: dst_bit_reg ({a}) aliases scratch_reg ({b})"
        ));
    }
    if a == c {
        return Err(format!("address: dst_bit_reg ({a}) aliases iter_reg ({c})"));
    }
    if b == c {
        return Err(format!("address: scratch_reg ({b}) aliases iter_reg ({c})"));
    }
    Ok(())
}
