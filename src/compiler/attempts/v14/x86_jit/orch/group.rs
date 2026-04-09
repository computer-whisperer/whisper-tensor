//! Emit a single `AtomGroup` as a loop or unrolled sequence.
//!
//! The base case for non-fused, non-reduce groups: load operands via
//! [`super::super::codec`], call into [`super::super::ops`], store the
//! result, and loop. Knows nothing about op semantics — just sequences
//! the codec/op calls per atom.
//!
//! # Phase 2.B.3 scope
//!
//! Only [`ScalarOp::Identity`] with `src dtype == output dtype` is
//! emitted. The body for each iteration is a bit-for-bit copy via
//! [`super::super::codec::bit_io::emit_load_bits`] and
//! [`super::super::codec::bit_io::emit_store_bits`] — no decode /
//! encode round-trip is needed since both ends share the same dtype.
//!
//! Cast (different src/dst dtypes) lands in P2.B.5 alongside
//! Literal / LiteralSpan, which adds the format-codec wiring.
//!
//! # Register layout
//!
//! Each iteration uses a fixed register set. The layout is documented
//! here so it stays in sync with [`super::super::prologue`] and
//! [`super::super::codec::bit_io`]:
//!
//! - `r12` (`BUFFER_REG`)        — buffer base, set up by the prologue.
//! - `r13` (`LOOP_VAR_REG`)      — loop induction variable (intra-slab
//!                                 absolute index, starting at
//!                                 `group.atom_offset`).
//! - `r14` (`LOOP_END_REG`)      — exclusive loop upper bound.
//! - `r10` (`BIT_OFF_REG`)       — output of [`address::emit_compute_bit_offset`],
//!                                 input to bit_io. Reused for both src
//!                                 and dst bit offsets within an iter.
//! - `r11` (`ADDR_SCRATCH`)      — scratch for address arithmetic AND
//!                                 the bit_io load `scratch_reg` AND the
//!                                 bit_io store `tmp3_reg`. All three
//!                                 uses are dead at different points in
//!                                 the iteration so there's no clash.
//! - `rax` (`RAW_REG`)           — holds the loaded raw bits across the
//!                                 src-load → dst-store sequence.
//! - `r8`  (`BIT_IO_TMP1`)       — store tmp1.
//! - `r9`  (`BIT_IO_TMP2`)       — store tmp2.
//! - `rcx`                       — clobbered by every bit_io call.

use dynasmrt::x64::Assembler;
use dynasmrt::{DynasmApi, DynasmLabelApi, dynasm};

use crate::compiler::attempts::v14::layout::BufferLayout;
use crate::nano_graph::ScalarOp;
use crate::nano_graph::pattern::{AtomGroup, AtomId, InputRef};
use crate::pool::SystemPool;

use super::super::codec::bit_io::{emit_load_bits, emit_store_bits};
use super::super::codec::format::{CodecSlot, CodecTables, ComputeRepr, emit_decode, emit_encode};
use super::super::prologue::{BUFFER_REG, LOOP_END_REG, LOOP_VAR_REG};
use super::address::{AddressInfo, AddressTables, IterVar, emit_compute_bit_offset};

/// Address output / bit_io input.
const BIT_OFF_REG: u8 = 10;
/// Address scratch + bit_io load scratch + bit_io store tmp3.
const ADDR_SCRATCH: u8 = 11;
/// Holds the loaded raw bits across the load → store sequence.
const RAW_REG: u8 = 0;
/// Bit_io store tmp1. Also codec scratch_gp1 between load and store.
const BIT_IO_TMP1: u8 = 8;
/// Bit_io store tmp2. Also codec scratch_gp2 between load and store.
const BIT_IO_TMP2: u8 = 9;
/// XMM register used as the float compute slot A.
const FLT_SLOT: u8 = 0;
/// XMM register used as codec scratch.
const FLT_SCRATCH: u8 = 1;

/// Emit one `AtomGroup` body. Either inlined (for `count == 1`) or
/// wrapped in a loop over the group's atoms.
///
/// Returns `Err` for ops that aren't yet supported. The error string
/// flows up to `X86JitSpan::compile`, which falls back to cranelift.
pub fn emit_group(
    asm: &mut Assembler,
    layout: &BufferLayout,
    group: &AtomGroup<'static, SystemPool>,
    addr_tables: &mut AddressTables,
    codec_tables: &mut CodecTables,
) -> Result<(), String> {
    match &group.op {
        ScalarOp::Identity => emit_identity_group(asm, layout, group, addr_tables),
        ScalarOp::Cast { .. } => emit_cast_group(asm, layout, group, addr_tables, codec_tables),
        ScalarOp::Literal(_) | ScalarOp::LiteralSpan(_) => {
            // Values are pre-populated in the buffer template by
            // `BufferLayout::populate_literals`. No code emission.
            Ok(())
        }
        op => Err(format!(
            "x86_jit emit_group: unsupported op {op:?} (P2.B.5 = Identity/Cast/Literal only)"
        )),
    }
}

/// Emit an Identity group: per-atom bit copy from
/// `group.inputs[0]` into the group's own slot.
fn emit_identity_group(
    asm: &mut Assembler,
    layout: &BufferLayout,
    group: &AtomGroup<'static, SystemPool>,
    tables: &mut AddressTables,
) -> Result<(), String> {
    if group.inputs.len() != 1 {
        return Err(format!(
            "Identity group has {} inputs, expected 1",
            group.inputs.len()
        ));
    }

    // The output is the group's own slot, accessed via the same
    // 1D-affine pattern as a Strided source. Reusing the address
    // primitive keeps the math in one place.
    let output_ref = InputRef::affine(group.base_id, 1);

    // P2.B.3 only handles same-dtype Identity. Verify by peeking at
    // the slot dtypes via the InputRef base lookups (cheap).
    let src_dtype = lookup_input_dtype(layout, &group.inputs[0], group.atom_offset)?;
    if src_dtype != group.output_dtype {
        return Err(format!(
            "x86_jit emit_group Identity: src dtype {:?} != output dtype {:?} \
             (P2.B.5 will add Cast)",
            src_dtype, group.output_dtype
        ));
    }

    if group.count == 1 {
        // Single-element fast path: no loop, both addresses are
        // compile-time constants from the iter point of view.
        emit_identity_iter(
            asm,
            layout,
            &group.inputs[0],
            &output_ref,
            IterVar::Const(group.atom_offset),
            group.atom_offset,
            tables,
        )
    } else {
        emit_identity_loop(
            asm,
            layout,
            &group.inputs[0],
            &output_ref,
            group.atom_offset,
            group.count,
            tables,
        )
    }
}

/// Emit one iteration of the Identity body: load raw bits from the
/// source bit offset, then store them at the destination bit offset.
fn emit_identity_iter(
    asm: &mut Assembler,
    layout: &BufferLayout,
    src_input: &InputRef,
    dst_input: &InputRef,
    iter: IterVar,
    atom_offset: u64,
    tables: &mut AddressTables,
) -> Result<(), String> {
    // 1. Compute src bit offset → r10.
    let src_info = emit_compute_bit_offset(
        asm,
        layout,
        src_input,
        iter,
        atom_offset,
        BIT_OFF_REG,
        ADDR_SCRATCH,
        tables,
    )?;

    // 2. Load src bits → rax.
    emit_load_bits(
        asm,
        BUFFER_REG,
        BIT_OFF_REG,
        src_info.n_bits,
        RAW_REG,
        ADDR_SCRATCH,
    );

    // 3. Compute dst bit offset → r10 (overwriting the src offset,
    // which we no longer need).
    let dst_info = emit_compute_bit_offset(
        asm,
        layout,
        dst_input,
        iter,
        atom_offset,
        BIT_OFF_REG,
        ADDR_SCRATCH,
        tables,
    )?;

    if src_info.n_bits != dst_info.n_bits {
        return Err(format!(
            "x86_jit emit_group Identity: src n_bits ({}) != dst n_bits ({})",
            src_info.n_bits, dst_info.n_bits
        ));
    }

    // 4. Store rax at dst bit offset. ADDR_SCRATCH (r11) is dead at
    // this point and can be reused as bit_io's tmp3.
    emit_store_bits(
        asm,
        BUFFER_REG,
        BIT_OFF_REG,
        dst_info.n_bits,
        RAW_REG,
        BIT_IO_TMP1,
        BIT_IO_TMP2,
        ADDR_SCRATCH,
    );

    Ok(())
}

/// Emit a count-loop around `emit_identity_iter`.
///
/// Loop shape (using r13 = `LOOP_VAR_REG`, r14 = `LOOP_END_REG`):
///
/// ```text
///     mov r13, atom_offset
///     mov r14, atom_offset + count
/// loop_top:
///     cmp r13, r14
///     jge loop_end
///     <emit_identity_iter with IterVar::Reg(r13)>
///     add r13, 1
///     jmp loop_top
/// loop_end:
/// ```
fn emit_identity_loop(
    asm: &mut Assembler,
    layout: &BufferLayout,
    src_input: &InputRef,
    dst_input: &InputRef,
    atom_offset: u64,
    count: u64,
    tables: &mut AddressTables,
) -> Result<(), String> {
    let start = atom_offset as i64;
    let end = (atom_offset + count) as i64;

    dynasm!(asm
        ; .arch x64
        ; mov Rq(LOOP_VAR_REG), QWORD start
        ; mov Rq(LOOP_END_REG), QWORD end
    );

    let loop_top = asm.new_dynamic_label();
    let loop_exit = asm.new_dynamic_label();

    dynasm!(asm
        ; =>loop_top
        ; cmp Rq(LOOP_VAR_REG), Rq(LOOP_END_REG)
        ; jge =>loop_exit
    );

    // Inside the loop, the iter register IS the absolute intra-slab
    // index — pass `atom_offset = 0` so address.rs doesn't double-shift.
    emit_identity_iter(
        asm,
        layout,
        src_input,
        dst_input,
        IterVar::Reg(LOOP_VAR_REG),
        0,
        tables,
    )?;

    dynasm!(asm
        ; add Rq(LOOP_VAR_REG), 1
        ; jmp =>loop_top
        ; =>loop_exit
    );

    Ok(())
}

// ─── Cast emission ──────────────────────────────────────────────────

/// Emit a Cast group: load raw bits, decode via src dtype, encode via
/// dst dtype, store raw bits.
///
/// Currently handles same-compute-repr casts (float↔float, int↔int).
/// Cross-repr casts (float↔int) return `Err` and fall back to
/// cranelift.
fn emit_cast_group(
    asm: &mut Assembler,
    layout: &BufferLayout,
    group: &AtomGroup<'static, SystemPool>,
    addr_tables: &mut AddressTables,
    codec_tables: &mut CodecTables,
) -> Result<(), String> {
    if group.inputs.len() != 1 {
        return Err(format!(
            "Cast group has {} inputs, expected 1",
            group.inputs.len()
        ));
    }

    let src_dtype = lookup_input_dtype(layout, &group.inputs[0], group.atom_offset)?;
    let dst_dtype = group.output_dtype;
    let src_repr = ComputeRepr::for_dtype(src_dtype);
    let dst_repr = ComputeRepr::for_dtype(dst_dtype);

    if src_repr != dst_repr {
        return Err(format!(
            "x86_jit Cast: cross-repr {src_dtype} ({src_repr:?}) → \
             {dst_dtype} ({dst_repr:?}) not yet supported (P2.B.6)"
        ));
    }

    let output_ref = InputRef::affine(group.base_id, 1);

    if group.count == 1 {
        emit_cast_iter(
            asm,
            layout,
            &group.inputs[0],
            &output_ref,
            src_dtype,
            dst_dtype,
            IterVar::Const(group.atom_offset),
            group.atom_offset,
            addr_tables,
            codec_tables,
        )
    } else {
        emit_cast_loop(
            asm,
            layout,
            &group.inputs[0],
            &output_ref,
            src_dtype,
            dst_dtype,
            group.atom_offset,
            group.count,
            addr_tables,
            codec_tables,
        )
    }
}

/// Emit one iteration of the Cast body: load raw bits, decode to
/// compute repr via src_dtype, encode from compute repr via dst_dtype,
/// store raw bits.
#[allow(clippy::too_many_arguments)]
fn emit_cast_iter(
    asm: &mut Assembler,
    layout: &BufferLayout,
    src_input: &InputRef,
    dst_input: &InputRef,
    src_dtype: crate::numeric_dtype::NumericDType,
    dst_dtype: crate::numeric_dtype::NumericDType,
    iter: IterVar,
    atom_offset: u64,
    addr_tables: &mut AddressTables,
    codec_tables: &mut CodecTables,
) -> Result<(), String> {
    // 1. Compute src bit offset → r10.
    let src_info = emit_compute_bit_offset(
        asm,
        layout,
        src_input,
        iter,
        atom_offset,
        BIT_OFF_REG,
        ADDR_SCRATCH,
        addr_tables,
    )?;

    // 2. Load src raw bits → rax.
    emit_load_bits(
        asm,
        BUFFER_REG,
        BIT_OFF_REG,
        src_info.n_bits,
        RAW_REG,
        ADDR_SCRATCH,
    );

    // 3. Decode raw bits to compute repr.
    let compute_repr = ComputeRepr::for_dtype(src_dtype);
    let slot = match compute_repr {
        ComputeRepr::F32 | ComputeRepr::F64 => CodecSlot::Xmm(FLT_SLOT),
        ComputeRepr::Int => CodecSlot::Gp(RAW_REG),
    };
    emit_decode(
        asm,
        src_dtype,
        RAW_REG,
        slot,
        BIT_IO_TMP1,  // scratch_gp
        FLT_SCRATCH,   // scratch_xmm
        codec_tables,
    )?;

    // 4. Encode from compute repr to dst raw bits → rax.
    emit_encode(
        asm,
        dst_dtype,
        slot,
        RAW_REG,
        BIT_IO_TMP1,  // scratch_gp1
        BIT_IO_TMP2,  // scratch_gp2
        FLT_SCRATCH,   // scratch_xmm
    )?;

    // 5. Compute dst bit offset → r10.
    let dst_info = emit_compute_bit_offset(
        asm,
        layout,
        dst_input,
        iter,
        atom_offset,
        BIT_OFF_REG,
        ADDR_SCRATCH,
        addr_tables,
    )?;

    // 6. Store raw bits.
    emit_store_bits(
        asm,
        BUFFER_REG,
        BIT_OFF_REG,
        dst_info.n_bits,
        RAW_REG,
        BIT_IO_TMP1,
        BIT_IO_TMP2,
        ADDR_SCRATCH,
    );

    Ok(())
}

/// Emit a count-loop around `emit_cast_iter`.
#[allow(clippy::too_many_arguments)]
fn emit_cast_loop(
    asm: &mut Assembler,
    layout: &BufferLayout,
    src_input: &InputRef,
    dst_input: &InputRef,
    src_dtype: crate::numeric_dtype::NumericDType,
    dst_dtype: crate::numeric_dtype::NumericDType,
    atom_offset: u64,
    count: u64,
    addr_tables: &mut AddressTables,
    codec_tables: &mut CodecTables,
) -> Result<(), String> {
    let start = atom_offset as i64;
    let end = (atom_offset + count) as i64;

    dynasm!(asm
        ; .arch x64
        ; mov Rq(LOOP_VAR_REG), QWORD start
        ; mov Rq(LOOP_END_REG), QWORD end
    );

    let loop_top = asm.new_dynamic_label();
    let loop_exit = asm.new_dynamic_label();

    dynasm!(asm
        ; =>loop_top
        ; cmp Rq(LOOP_VAR_REG), Rq(LOOP_END_REG)
        ; jge =>loop_exit
    );

    emit_cast_iter(
        asm,
        layout,
        src_input,
        dst_input,
        src_dtype,
        dst_dtype,
        IterVar::Reg(LOOP_VAR_REG),
        0,
        addr_tables,
        codec_tables,
    )?;

    dynasm!(asm
        ; add Rq(LOOP_VAR_REG), 1
        ; jmp =>loop_top
        ; =>loop_exit
    );

    Ok(())
}

/// Look up the storage dtype the codec will read for an `InputRef`.
/// Used by `emit_identity_group` to enforce the
/// "src dtype == output dtype" precondition before emitting any code.
///
/// Mirrors the slot-resolution logic in
/// [`super::address::emit_compute_bit_offset`]: probe the InputRef's
/// declared base atom first, then fall back to the first accessed
/// atom for split groups whose base lives outside this span.
fn lookup_input_dtype(
    layout: &BufferLayout,
    input: &InputRef,
    atom_offset: u64,
) -> Result<crate::numeric_dtype::NumericDType, String> {
    match input {
        InputRef::Broadcast(a) => layout
            .find(*a)
            .map(|(slot, _)| slot.dtype)
            .ok_or_else(|| format!("emit_group: no slot for Broadcast atom={a}")),

        InputRef::Strided {
            base,
            dim_strides,
            dim_shape,
        } => {
            if let Some((slot, _)) = layout.find(*base) {
                return Ok(slot.dtype);
            }
            // Split group fallback: resolve the first accessed atom.
            let first_offset =
                crate::compiler::attempts::v14::layout::strided_resolve_offset(
                    dim_strides, dim_shape, atom_offset,
                );
            let first_atom = AtomId(((base.0 as i64) + first_offset) as u64);
            layout
                .find(first_atom)
                .map(|(slot, _)| slot.dtype)
                .ok_or_else(|| {
                    format!(
                        "emit_group: no slot for Strided base={base} first={first_atom}"
                    )
                })
        }

        InputRef::Explicit(ids) if !ids.is_empty() => layout
            .find(ids[0])
            .map(|(slot, _)| slot.dtype)
            .ok_or_else(|| format!("emit_group: no slot for Explicit[0] atom={}", ids[0])),
        InputRef::Explicit(_) => Err("emit_group: empty Explicit InputRef".to_string()),
    }
}
