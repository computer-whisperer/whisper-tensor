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
//! # Phase 2.B.2 scope
//!
//! Only the simplest InputRef shapes are wired here, just enough for
//! Identity / Cast over byte-aligned slots:
//!
//! - `Broadcast` — single source atom, always a constant offset
//! - `Strided` 1D affine — `base + dim_strides[0] * i`, where `i`
//!   may be a compile-time constant or a runtime register
//! - `Explicit` with exactly one entry — equivalent to `Broadcast`
//!
//! N-d Strided and multi-entry Explicit return `Err` for now and are
//! widened in [P2.B.4]. The `support::check_supported` gate refuses
//! anything that uses an unsupported InputRef shape.

use dynasmrt::x64::Assembler;
use dynasmrt::{DynasmApi, dynasm};

use crate::compiler::attempts::v14::layout::BufferLayout;
use crate::nano_graph::pattern::{AtomId, InputRef};
use crate::numeric_dtype::NumericDType;

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
/// # Errors
///
/// Returns `Err` for:
/// - InputRef shapes not yet supported in P2.B.2 (n-d Strided,
///   multi-entry Explicit)
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
) -> Result<AddressInfo, String> {
    match input {
        InputRef::Broadcast(atom_id) => emit_constant_atom(asm, layout, *atom_id, dst_bit_reg),

        InputRef::Explicit(ids) => {
            if ids.len() != 1 {
                return Err(format!(
                    "address: Explicit with {} entries not yet supported (P2.B.4)",
                    ids.len()
                ));
            }
            emit_constant_atom(asm, layout, ids[0], dst_bit_reg)
        }

        InputRef::Strided {
            base,
            dim_strides,
            dim_shape: _,
        } => {
            if dim_strides.len() != 1 {
                return Err(format!(
                    "address: Strided n-d (nd={}) not yet supported (P2.B.4)",
                    dim_strides.len()
                ));
            }
            emit_strided_1d(
                asm,
                layout,
                *base,
                dim_strides[0],
                iter,
                atom_offset,
                dst_bit_reg,
                scratch_reg,
            )
        }
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
    let bit_off = slot.bit_offset + elem_idx * slot.bit_stride;
    emit_mov_imm64(asm, dst_bit_reg, bit_off);
    Ok(AddressInfo {
        dtype: slot.dtype,
        n_bits: slot.elem_bits as u32,
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

    let info = AddressInfo {
        dtype: slot.dtype,
        n_bits: slot.elem_bits as u32,
    };

    match iter {
        IterVar::Const(c) => {
            let abs_bit = base_bit_signed + bit_stride_signed * c as i64;
            if abs_bit < 0 {
                return Err(format!(
                    "address: negative bit offset {abs_bit} for Strided base={base} \
                     atom_offset={atom_offset} stride={stride_atoms} c={c}"
                ));
            }
            emit_mov_imm64(asm, dst_bit_reg, abs_bit as u64);
        }
        IterVar::Reg(iter_reg) => {
            assert_distinct(dst_bit_reg, scratch_reg, iter_reg)?;

            // dst = iter_reg
            dynasm!(asm
                ; .arch x64
                ; mov Rq(dst_bit_reg), Rq(iter_reg)
            );

            // dst *= bit_stride
            // Use the imm32 form when possible — saves a `mov scratch,
            // imm64` and an extra register dependency.
            if (i32::MIN as i64..=i32::MAX as i64).contains(&bit_stride_signed) {
                let imm = bit_stride_signed as i32;
                dynasm!(asm
                    ; imul Rq(dst_bit_reg), Rq(dst_bit_reg), imm
                );
            } else {
                emit_mov_imm64(asm, scratch_reg, bit_stride_signed as u64);
                dynasm!(asm
                    ; imul Rq(dst_bit_reg), Rq(scratch_reg)
                );
            }

            // dst += base_bit
            // base_bit is a full 64-bit signed value; the imm32 add
            // form (`add r/m64, imm8/imm32`) sign-extends, so we can
            // use it for small bases. For larger bases, route through
            // scratch_reg.
            if (i32::MIN as i64..=i32::MAX as i64).contains(&base_bit_signed) {
                let imm = base_bit_signed as i32;
                dynasm!(asm
                    ; add Rq(dst_bit_reg), imm
                );
            } else {
                emit_mov_imm64(asm, scratch_reg, base_bit_signed as u64);
                dynasm!(asm
                    ; add Rq(dst_bit_reg), Rq(scratch_reg)
                );
            }
        }
    }

    Ok(info)
}

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
