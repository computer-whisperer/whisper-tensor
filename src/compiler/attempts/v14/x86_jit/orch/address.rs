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
use crate::nano_graph::pattern::{AtomId, InputRef};
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
/// Sym-free groups pass `None` — no sym term is emitted and the
/// address layer behaves identically to the pre-sym version.
#[derive(Clone, Copy, Debug)]
pub struct SymCtx {
    /// Stack offset (relative to the current `rsp`) where the inner
    /// sym loop stores its `sym_i` counter as a `u64`.
    pub sym_i_rsp_off: i32,
}

/// Return the effective sym_ctx to use at a specific input's
/// address-compute site, given the outer consumer sym_ctx (from the
/// atom-body loop) and the input's `sym_dim_map`.
///
/// - Empty `sym_dim_map` → input has no sym component at all; return
///   `None`.
/// - All-`Broadcast` map → producer has no sym axes that vary with
///   the consumer; the same producer atom is read for every
///   `(atom_i, sym_flat)` slot; return `None` so the address layer
///   doesn't add a `sym_i * elem_bits` term.
/// - Any other shape (all-`Identity(j == index)` — the only other
///   variant `support::check_supported` admits today) → return
///   `outer` unchanged.
pub fn input_sym_ctx(
    outer: Option<SymCtx>,
    sym_dim_map: &[crate::nano_graph::pattern::SymDimMap],
) -> Option<SymCtx> {
    if sym_dim_map.is_empty() {
        return None;
    }
    if sym_dim_map
        .iter()
        .all(|m| matches!(m, crate::nano_graph::pattern::SymDimMap::Broadcast))
    {
        return None;
    }
    outer
}

/// Information returned by [`emit_compute_bit_offset`] so the caller
/// knows how to read the bits the offset addresses.
#[derive(Clone, Copy, Debug)]
pub struct AddressInfo {
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
pub fn emit_compute_bit_offset(
    asm: &mut Assembler,
    layout: &BufferLayout,
    input: &InputRef,
    iter: IterVar,
    atom_offset: u64,
    dst_bit_reg: u8,
    scratch_reg: u8,
    sym_ctx: Option<SymCtx>,
    tables: &mut AddressTables,
) -> Result<AddressInfo, String> {
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

    apply_sym_offset(asm, sym_ctx, &info, dst_bit_reg, scratch_reg)?;
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
) -> Result<(), String> {
    apply_sym_offset_pub(asm, sym_ctx, info, dst_bit_reg, scratch_reg)
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
) -> Result<(), String> {
    let Some(ctx) = sym_ctx else {
        return Ok(());
    };
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
    dynasm!(asm
        ; .arch x64
        ; mov Rq(scratch_reg), QWORD [rsp + ctx.sym_i_rsp_off]
        ; imul Rq(scratch_reg), Rq(scratch_reg), step
        ; add Rq(dst_bit_reg), Rq(scratch_reg)
    );
    Ok(())
}

/// Emit a constant bit-offset materialization for a single atom.
/// Used by `Broadcast` and single-element `Explicit`.
fn emit_constant_atom(
    asm: &mut Assembler,
    layout: &BufferLayout,
    atom_id: AtomId,
    dst_bit_reg: u8,
    scratch_reg: u8,
    sym_ctx: Option<SymCtx>,
) -> Result<AddressInfo, String> {
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
fn emit_strided_1d(
    asm: &mut Assembler,
    layout: &BufferLayout,
    base: AtomId,
    stride_atoms: i64,
    iter: IterVar,
    atom_offset: u64,
    dst_bit_reg: u8,
    scratch_reg: u8,
    sym_ctx: Option<SymCtx>,
) -> Result<AddressInfo, String> {
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
                // 4. scratch = runtime sym_prod, then dst *= scratch.
                emit_runtime_sym_prod(
                    asm,
                    &layout.buffer_bases,
                    &slot.sym_dims,
                    scratch_reg,
                    sym_ctx,
                )?;
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
/// `dst_reg`. Uses `dst_reg` itself as the accumulator and one
/// additional stack-loaded pointer to the gc_values array.
///
/// Single sym dim (the overwhelming majority — RWKV's batch, etc.):
/// one `mov` from the gc_values array, no multiplies. Multi sym dim:
/// subsequent `imul` ops against each gc's slot. The gc_values
/// pointer sits at `[rsp + gc_values_stack_offset(bases) + rsp_delta]`
/// where `rsp_delta` accounts for `emit_atom_body_loop`'s `sub rsp, 16`
/// (and any inner `push`es threaded via `sym_ctx.sym_i_rsp_off`).
pub(super) fn emit_runtime_sym_prod_pub(
    asm: &mut Assembler,
    bases: &crate::compiler::attempts::v14::layout::BufferBases,
    sym_dims: &[crate::nano_graph::pattern::GraphConstantId],
    dst_reg: u8,
    sym_ctx: Option<SymCtx>,
) -> Result<(), String> {
    emit_runtime_sym_prod(asm, bases, sym_dims, dst_reg, sym_ctx)
}

fn emit_runtime_sym_prod(
    asm: &mut Assembler,
    bases: &crate::compiler::attempts::v14::layout::BufferBases,
    sym_dims: &[crate::nano_graph::pattern::GraphConstantId],
    dst_reg: u8,
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

    // dst = gc_values_ptr
    dynasm!(asm
        ; .arch x64
        ; mov Rq(dst_reg), QWORD [rsp + gc_values_off]
    );
    // dst = gc_values_ptr[sym_dims[0].0]
    let first_off: i32 = (sym_dims[0].0 as i64)
        .checked_mul(8)
        .and_then(|v| i32::try_from(v).ok())
        .ok_or_else(|| format!("address: gc index {} overflow", sym_dims[0].0))?;
    dynasm!(asm
        ; .arch x64
        ; mov Rq(dst_reg), QWORD [Rq(dst_reg) + first_off]
    );
    // Multi-sym fold-in: imul dst, gc_values_ptr[sym_dims[k].0]. We'd
    // need another scratch register to reload the gc_values_ptr after
    // the first mov clobbered dst; reject until a real case shows up.
    if sym_dims.len() > 1 {
        return Err(format!(
            "address: multi-sym-dim runtime stride not yet supported \
             (slot has {} sym dims); add second scratch reg first",
            sym_dims.len()
        ));
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
fn emit_strided_nd(
    asm: &mut Assembler,
    layout: &BufferLayout,
    base: AtomId,
    dim_strides: &[i64],
    dim_shape: &[u64],
    iter: IterVar,
    atom_offset: u64,
    dst_bit_reg: u8,
    scratch_reg: u8,
    _sym_ctx: Option<SymCtx>,
) -> Result<AddressInfo, String> {
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
fn emit_explicit_multi(
    asm: &mut Assembler,
    layout: &BufferLayout,
    ids: &[AtomId],
    iter: IterVar,
    dst_bit_reg: u8,
    scratch_reg: u8,
    tables: &mut AddressTables,
    sym_ctx: Option<SymCtx>,
) -> Result<AddressInfo, String> {
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
                    emit_runtime_sym_prod(asm, &layout.buffer_bases, sd, scratch_reg, sym_ctx)?;
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
