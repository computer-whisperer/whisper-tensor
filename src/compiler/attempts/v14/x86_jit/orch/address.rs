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

/// Information returned by [`emit_compute_bit_offset`] so the caller
/// knows how to read the bits the offset addresses.
#[derive(Clone, Copy, Debug)]
pub struct AddressInfo {
    /// Storage dtype of the slot the bit offset addresses.
    pub dtype: NumericDType,
    /// Number of bits to load from that bit offset, equal to
    /// `slot.elem_bits` (the dtype's semantic width — sub-byte
    /// dtypes have `n_bits < 8`).
    pub n_bits: u32,
    /// Which buffer the resolved slot lives in. The caller is
    /// responsible for selecting the base register that holds this
    /// buffer's pointer and adding the bit offset to it. Under the
    /// single-buffer per-span layout (pre memory-placement rework)
    /// this is always 0. See `MEMORY_PLACEMENT.md` §"JIT ABI".
    pub buffer_id: u8,
    /// When true, the emitted offset register holds a **byte** offset
    /// instead of a bit offset, and the element width is a power-of-2
    /// number of bytes (1, 2, 4, or 8). The caller can use the
    /// byte-aligned load/store fast path in `bit_io` — a single `mov`
    /// instruction instead of the general bit-extraction sequence.
    ///
    /// Set when the slot's `bit_offset` and `bit_stride` are both
    /// multiples of 8 and `elem_bits` is 8, 16, 32, or 64.
    pub byte_aligned: bool,
    /// When `Some`, the address was NOT emitted to a register — the
    /// caller should fold it into the load/store instruction as a SIB
    /// addressing mode: `[base_reg + index_reg * scale + disp]`.
    ///
    /// `index_reg` is the loop iteration register. `scale` is 1, 2,
    /// 4, or 8. `disp` fits in i32.
    ///
    /// When `None`, the offset was materialized into `dst_bit_reg` as
    /// before, and the caller uses `[base_reg + dst_bit_reg]`.
    pub sib: Option<SibMode>,
}

/// SIB addressing mode: `[base_reg + index_reg * scale + disp]`.
///
/// The address layer emits **no code** when it returns a SibMode — the
/// caller is responsible for encoding the SIB form directly in the
/// load/store instruction.
#[derive(Clone, Copy, Debug)]
pub struct SibMode {
    /// Register holding the loop iteration variable.
    pub index_reg: u8,
    /// SIB scale factor (1, 2, 4, or 8).
    pub scale: u8,
    /// Signed 32-bit displacement added to `base + index*scale`.
    pub disp: i32,
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
pub fn emit_compute_bit_offset(
    asm: &mut Assembler,
    layout: &BufferLayout,
    input: &InputRef,
    iter: IterVar,
    atom_offset: u64,
    dst_bit_reg: u8,
    scratch_reg: u8,
    tables: &mut AddressTables,
) -> Result<AddressInfo, String> {
    match input {
        InputRef::Broadcast(atom_id) => emit_constant_atom(asm, layout, *atom_id, dst_bit_reg),

        InputRef::Explicit(ids) if ids.len() <= 1 => {
            if ids.is_empty() {
                return Err("address: empty Explicit InputRef".to_string());
            }
            emit_constant_atom(asm, layout, ids[0], dst_bit_reg)
        }

        InputRef::Explicit(ids) => {
            emit_explicit_multi(asm, layout, ids, iter, dst_bit_reg, scratch_reg, tables)
        }

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
        ),

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
        ),
    }
}

/// Emit a constant bit-offset materialization for a single atom.
/// Used by `Broadcast` and single-element `Explicit`.
fn emit_constant_atom(
    asm: &mut Assembler,
    layout: &BufferLayout,
    atom_id: AtomId,
    dst_bit_reg: u8,
) -> Result<AddressInfo, String> {
    let (slot, elem_idx) = layout
        .find(atom_id)
        .ok_or_else(|| format!("address: no slot for atom={atom_id}"))?;
    let byte_fast = slot_is_byte_fast(slot);
    let bit_off = slot.bit_offset + elem_idx * slot.bit_stride;
    if byte_fast {
        emit_mov_imm64(asm, dst_bit_reg, bit_off / 8);
    } else {
        emit_mov_imm64(asm, dst_bit_reg, bit_off);
    }
    Ok(AddressInfo {
        dtype: slot.dtype,
        n_bits: slot.elem_bits as u32,
        buffer_id: slot.buffer_id,
        byte_aligned: byte_fast,
        sib: None,
    })
}

/// 1D affine path: `bit_offset(i) = base_bit + bit_stride * i` where
/// `base_bit` is the bit offset of the logical `base` atom in the
/// buffer (back-computed for split groups whose base is outside this
/// span's slot map) and `bit_stride = dim_strides[0] * slot.bit_stride`.
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
    let bit_stride_signed = stride_atoms * slot.bit_stride as i64;

    let byte_fast = slot_is_byte_fast(slot);
    let info = AddressInfo {
        dtype: slot.dtype,
        n_bits: slot.elem_bits as u32,
        buffer_id: slot.buffer_id,
        byte_aligned: byte_fast,
        sib: None,
    };

    // When byte-aligned, emit byte offsets (divide by 8 at JIT-build
    // time) so the codec can use direct `mov` instructions.
    let (eff_base, eff_stride) = if byte_fast {
        (base_bit_signed / 8, bit_stride_signed / 8)
    } else {
        (base_bit_signed, bit_stride_signed)
    };

    match iter {
        IterVar::Const(c) => {
            let abs = eff_base + eff_stride * c as i64;
            if abs < 0 {
                return Err(format!(
                    "address: negative offset {abs} for Strided base={base} \
                     atom_offset={atom_offset} stride={stride_atoms} c={c}"
                ));
            }
            emit_mov_imm64(asm, dst_bit_reg, abs as u64);
        }
        IterVar::Reg(iter_reg) => {
            assert_distinct(dst_bit_reg, scratch_reg, iter_reg)?;

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

    Ok(info)
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
        sib: None,
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
fn emit_explicit_multi(
    asm: &mut Assembler,
    layout: &BufferLayout,
    ids: &[AtomId],
    iter: IterVar,
    dst_bit_reg: u8,
    scratch_reg: u8,
    tables: &mut AddressTables,
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
        sib: None,
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
            return emit_constant_atom(asm, layout, ids[idx], dst_bit_reg);
        }
        IterVar::Reg(iter_reg) => {
            // Build the offset lookup table. When byte-aligned, store
            // byte offsets; otherwise bit offsets.
            let offsets: Vec<i64> = ids
                .iter()
                .enumerate()
                .map(|(i, id)| {
                    let (slot, elem_idx) = layout
                        .find(*id)
                        .ok_or_else(|| format!("address: no slot for Explicit[{i}] atom={id}"))?;
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
