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
use crate::nano_graph::ops::{ScalarBinOp, ScalarUnaryOp};
use crate::nano_graph::pattern::{AtomGroup, AtomId, InputRef};
use crate::numeric_dtype::NumericDType;
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
        ScalarOp::Identity => emit_identity_group(asm, layout, group, addr_tables, codec_tables),
        ScalarOp::Cast { .. } => emit_cast_group(asm, layout, group, addr_tables, codec_tables),
        ScalarOp::Literal(_) | ScalarOp::LiteralSpan(_) => {
            // Values are pre-populated in the buffer template by
            // `BufferLayout::populate_literals`. No code emission.
            Ok(())
        }
        ScalarOp::Binary { op, compute_dtype } => {
            emit_binary_group(asm, layout, group, *op, *compute_dtype, addr_tables, codec_tables)
        }
        ScalarOp::Unary { op, compute_dtype } => {
            emit_unary_group(asm, layout, group, *op, *compute_dtype, addr_tables, codec_tables)
        }
        ScalarOp::Select => {
            emit_select_group(asm, layout, group, addr_tables, codec_tables)
        }
        ScalarOp::IndirectLoad { table_base } => {
            emit_indirect_load_group(asm, layout, group, *table_base, addr_tables, codec_tables)
        }
        ScalarOp::Reduce {
            kind,
            reduce_count,
            reduce_stride,
            compute_dtype,
        } => super::reduce::emit_reduce_group(
            asm, layout, group, *kind, *reduce_count, *reduce_stride, *compute_dtype,
            addr_tables, codec_tables,
        ),
        op => Err(format!(
            "x86_jit emit_group: unsupported op {op:?} not yet supported"
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
    codec_tables: &mut CodecTables,
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

    // If src and output dtypes differ, this is effectively a Cast —
    // route through the Cast path which does decode(src) → encode(dst).
    let src_dtype = lookup_input_dtype(layout, &group.inputs[0], group.atom_offset)?;
    if src_dtype != group.output_dtype {
        return emit_cast_group(asm, layout, group, tables, codec_tables);
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

/// Emit a Cast group: load raw bits, decode via src dtype, optionally
/// convert between compute reprs, encode via dst dtype, store raw bits.
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

    // 3. Decode raw bits to src compute repr.
    let src_repr = ComputeRepr::for_dtype(src_dtype);
    let src_slot = match src_repr {
        ComputeRepr::F32 | ComputeRepr::F64 => CodecSlot::Xmm(FLT_SLOT),
        ComputeRepr::Int => CodecSlot::Gp(RAW_REG),
    };
    emit_decode(
        asm,
        src_dtype,
        RAW_REG,
        src_slot,
        BIT_IO_TMP1,  // scratch_gp
        FLT_SCRATCH,   // scratch_xmm
        codec_tables,
    )?;

    // 3b. If src and dst compute reprs differ, convert.
    let dst_repr = ComputeRepr::for_dtype(dst_dtype);
    let dst_slot = match dst_repr {
        ComputeRepr::F32 | ComputeRepr::F64 => CodecSlot::Xmm(FLT_SLOT),
        ComputeRepr::Int => CodecSlot::Gp(RAW_REG),
    };
    if src_repr != dst_repr {
        emit_repr_convert(asm, src_repr, dst_repr)?;
    }

    // 4. Encode from dst compute repr to dst raw bits → rax.
    emit_encode(
        asm,
        dst_dtype,
        dst_slot,
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

// ─── Binary op emission ─────────────────────────────────────────────

/// Emit a Binary group: load two inputs, apply the op, store result.
#[allow(clippy::too_many_arguments)]
fn emit_binary_group(
    asm: &mut Assembler,
    layout: &BufferLayout,
    group: &AtomGroup<'static, SystemPool>,
    op: ScalarBinOp,
    compute_dtype: NumericDType,
    addr_tables: &mut AddressTables,
    codec_tables: &mut CodecTables,
) -> Result<(), String> {
    if group.inputs.len() != 2 {
        return Err(format!(
            "Binary group has {} inputs, expected 2",
            group.inputs.len()
        ));
    }
    let output_ref = InputRef::affine(group.base_id, 1);

    if group.count == 1 {
        emit_binary_iter(
            asm, layout, &group.inputs[0], &group.inputs[1], &output_ref,
            op, compute_dtype, group.output_dtype,
            IterVar::Const(group.atom_offset), group.atom_offset,
            addr_tables, codec_tables,
        )
    } else {
        emit_binary_loop(
            asm, layout, &group.inputs[0], &group.inputs[1], &output_ref,
            op, compute_dtype, group.output_dtype,
            group.atom_offset, group.count,
            addr_tables, codec_tables,
        )
    }
}

/// Emit one iteration of a Binary body.
///
/// Register flow for float compute repr:
///   1. Load+decode A → xmm0 (FLT_SLOT_A)
///   2. Load+decode B → xmm1 (FLT_SLOT_B)  [rax free, xmm0 holds A]
///   3. Op → xmm2 (FLT_SLOT_C)
///   4. Encode xmm2 → rax, store
///
/// Register flow for int compute repr:
///   1. Load+decode A → rax (INT_SLOT_A)
///   2. Stash A: movq xmm0, rax
///   3. Load+decode B → rcx (INT_SLOT_B)
///   4. Restore A: movq rax, xmm0
///   5. Op → rdx (INT_SLOT_C)
///   6. Encode rdx → rax, store
#[allow(clippy::too_many_arguments)]
fn emit_binary_iter(
    asm: &mut Assembler,
    layout: &BufferLayout,
    input_a: &InputRef,
    input_b: &InputRef,
    output: &InputRef,
    op: ScalarBinOp,
    compute_dtype: NumericDType,
    output_dtype: NumericDType,
    iter: IterVar,
    atom_offset: u64,
    addr_tables: &mut AddressTables,
    codec_tables: &mut CodecTables,
) -> Result<(), String> {
    use super::super::ops::float::{emit_binop_f32, emit_binop_f64};
    use super::super::prologue::{FLT_SLOT_A, FLT_SLOT_B, INT_SLOT_A, INT_SLOT_B};

    let repr = ComputeRepr::for_dtype(compute_dtype);

    // ── Load + decode input A ──
    let _a_info = emit_load_decode_input(
        asm, layout, input_a, iter, atom_offset,
        addr_tables, codec_tables,
        match repr {
            ComputeRepr::F32 | ComputeRepr::F64 => CodecSlot::Xmm(FLT_SLOT_A),
            ComputeRepr::Int => CodecSlot::Gp(INT_SLOT_A),
        },
    )?;

    // For int ops: stash A in xmm0 before loading B (which clobbers rax).
    if repr == ComputeRepr::Int {
        dynasm!(asm; .arch x64; movq Rx(FLT_SLOT), Rq(INT_SLOT_A));
    }

    // ── Load + decode input B ──
    let b_slot = match repr {
        ComputeRepr::F32 | ComputeRepr::F64 => CodecSlot::Xmm(FLT_SLOT_B),
        ComputeRepr::Int => CodecSlot::Gp(INT_SLOT_B),
    };
    let _b_info = emit_load_decode_input(
        asm, layout, input_b, iter, atom_offset,
        addr_tables, codec_tables, b_slot,
    )?;

    // For int ops: restore A from xmm stash.
    if repr == ComputeRepr::Int {
        dynasm!(asm; .arch x64; movq Rq(INT_SLOT_A), Rx(FLT_SLOT));
    }

    // ── Apply the op ──
    match repr {
        ComputeRepr::F32 => emit_binop_f32(asm, op, BIT_IO_TMP1)?,
        ComputeRepr::F64 => emit_binop_f64(asm, op, BIT_IO_TMP1)?,
        ComputeRepr::Int => {
            use super::super::ops::int::emit_binop_int;
            let (signed, bits) = int_dtype_info(compute_dtype)?;
            emit_binop_int(asm, op, signed, bits, BIT_IO_TMP1)?;
            // Apply wrapping: mask to compute width + sign-extend.
            emit_int_wrap(asm, bits, signed, super::super::prologue::INT_SLOT_C);
        }
    }

    // ── Encode result + store ──
    let result_slot = match repr {
        ComputeRepr::F32 | ComputeRepr::F64 => {
            CodecSlot::Xmm(super::super::prologue::FLT_SLOT_C)
        }
        ComputeRepr::Int => CodecSlot::Gp(super::super::prologue::INT_SLOT_C),
    };
    emit_encode(
        asm, output_dtype, result_slot, RAW_REG,
        BIT_IO_TMP1, BIT_IO_TMP2, FLT_SCRATCH,
    )?;

    let dst_info = emit_compute_bit_offset(
        asm, layout, output, iter, atom_offset,
        BIT_OFF_REG, ADDR_SCRATCH, addr_tables,
    )?;

    emit_store_bits(
        asm, BUFFER_REG, BIT_OFF_REG, dst_info.n_bits, RAW_REG,
        BIT_IO_TMP1, BIT_IO_TMP2, ADDR_SCRATCH,
    );

    Ok(())
}

/// Helper: compute bit offset, load raw bits, decode to compute repr.
///
/// The decode uses the **storage dtype** from the layout slot (via
/// `AddressInfo::dtype`), not the op's `compute_dtype`. This is
/// correct because the codec converts from the storage format to
/// the dtype's natural compute repr (e.g., BF16 → F32).
#[allow(clippy::too_many_arguments)]
fn emit_load_decode_input(
    asm: &mut Assembler,
    layout: &BufferLayout,
    input: &InputRef,
    iter: IterVar,
    atom_offset: u64,
    addr_tables: &mut AddressTables,
    codec_tables: &mut CodecTables,
    slot: CodecSlot,
) -> Result<AddressInfo, String> {
    let info = emit_compute_bit_offset(
        asm, layout, input, iter, atom_offset,
        BIT_OFF_REG, ADDR_SCRATCH, addr_tables,
    )?;
    emit_load_bits(asm, BUFFER_REG, BIT_OFF_REG, info.n_bits, RAW_REG, ADDR_SCRATCH);
    emit_decode(asm, info.dtype, RAW_REG, slot, BIT_IO_TMP1, FLT_SCRATCH, codec_tables)?;
    Ok(info)
}

/// Emit a count-loop around `emit_binary_iter`.
#[allow(clippy::too_many_arguments)]
fn emit_binary_loop(
    asm: &mut Assembler,
    layout: &BufferLayout,
    input_a: &InputRef,
    input_b: &InputRef,
    output: &InputRef,
    op: ScalarBinOp,
    compute_dtype: NumericDType,
    output_dtype: NumericDType,
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

    emit_binary_iter(
        asm, layout, input_a, input_b, output,
        op, compute_dtype, output_dtype,
        IterVar::Reg(LOOP_VAR_REG), 0,
        addr_tables, codec_tables,
    )?;

    dynasm!(asm
        ; add Rq(LOOP_VAR_REG), 1
        ; jmp =>loop_top
        ; =>loop_exit
    );

    Ok(())
}

// ─── Unary op emission ──────────────────────────────────────────────

/// Emit a Unary group: load one input, apply the op, store result.
#[allow(clippy::too_many_arguments)]
fn emit_unary_group(
    asm: &mut Assembler,
    layout: &BufferLayout,
    group: &AtomGroup<'static, SystemPool>,
    op: ScalarUnaryOp,
    compute_dtype: NumericDType,
    addr_tables: &mut AddressTables,
    codec_tables: &mut CodecTables,
) -> Result<(), String> {
    if group.inputs.len() != 1 {
        return Err(format!(
            "Unary group has {} inputs, expected 1",
            group.inputs.len()
        ));
    }
    let output_ref = InputRef::affine(group.base_id, 1);

    if group.count == 1 {
        emit_unary_iter(
            asm, layout, &group.inputs[0], &output_ref,
            op, compute_dtype, group.output_dtype,
            IterVar::Const(group.atom_offset), group.atom_offset,
            addr_tables, codec_tables,
        )
    } else {
        emit_unary_loop(
            asm, layout, &group.inputs[0], &output_ref,
            op, compute_dtype, group.output_dtype,
            group.atom_offset, group.count,
            addr_tables, codec_tables,
        )
    }
}

/// Emit one iteration of a Unary body.
#[allow(clippy::too_many_arguments)]
fn emit_unary_iter(
    asm: &mut Assembler,
    layout: &BufferLayout,
    input: &InputRef,
    output: &InputRef,
    op: ScalarUnaryOp,
    compute_dtype: NumericDType,
    output_dtype: NumericDType,
    iter: IterVar,
    atom_offset: u64,
    addr_tables: &mut AddressTables,
    codec_tables: &mut CodecTables,
) -> Result<(), String> {
    use super::super::ops::float::{emit_unop_f32, emit_unop_f64};
    use super::super::prologue::FLT_SLOT_A;

    let repr = ComputeRepr::for_dtype(compute_dtype);

    // Load + decode input → slot A.
    let _info = emit_load_decode_input(
        asm, layout, input, iter, atom_offset,
        addr_tables, codec_tables,
        match repr {
            ComputeRepr::F32 | ComputeRepr::F64 => CodecSlot::Xmm(FLT_SLOT_A),
            ComputeRepr::Int => CodecSlot::Gp(super::super::prologue::INT_SLOT_A),
        },
    )?;

    // Apply the op.
    match repr {
        ComputeRepr::F32 => emit_unop_f32(asm, op, BIT_IO_TMP1)?,
        ComputeRepr::F64 => emit_unop_f64(asm, op, BIT_IO_TMP1)?,
        ComputeRepr::Int => {
            use super::super::ops::int::emit_unop_int;
            let (signed, bits) = int_dtype_info(compute_dtype)?;
            emit_unop_int(asm, op, signed)?;
            emit_int_wrap(asm, bits, signed, super::super::prologue::INT_SLOT_C);
        }
    }

    // Encode result + store.
    let result_slot = match repr {
        ComputeRepr::F32 | ComputeRepr::F64 => {
            CodecSlot::Xmm(super::super::prologue::FLT_SLOT_C)
        }
        ComputeRepr::Int => CodecSlot::Gp(super::super::prologue::INT_SLOT_C),
    };
    emit_encode(
        asm, output_dtype, result_slot, RAW_REG,
        BIT_IO_TMP1, BIT_IO_TMP2, FLT_SCRATCH,
    )?;

    let dst_info = emit_compute_bit_offset(
        asm, layout, output, iter, atom_offset,
        BIT_OFF_REG, ADDR_SCRATCH, addr_tables,
    )?;

    emit_store_bits(
        asm, BUFFER_REG, BIT_OFF_REG, dst_info.n_bits, RAW_REG,
        BIT_IO_TMP1, BIT_IO_TMP2, ADDR_SCRATCH,
    );

    Ok(())
}

/// Emit a count-loop around `emit_unary_iter`.
#[allow(clippy::too_many_arguments)]
fn emit_unary_loop(
    asm: &mut Assembler,
    layout: &BufferLayout,
    input: &InputRef,
    output: &InputRef,
    op: ScalarUnaryOp,
    compute_dtype: NumericDType,
    output_dtype: NumericDType,
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

    emit_unary_iter(
        asm, layout, input, output,
        op, compute_dtype, output_dtype,
        IterVar::Reg(LOOP_VAR_REG), 0,
        addr_tables, codec_tables,
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
// ─── Cross-repr conversion ──────────────────────────────────────────

/// Emit the conversion between compute representations.
///
/// After decoding the source value into `src_repr`, this function
/// converts it to `dst_repr` so the subsequent encode can operate
/// on the correct slot type. The conversion matches `cast_raw`
/// semantics (which goes through f64 intermediate).
///
/// Register contract:
/// - Float→Int: reads xmm0 (FLT_SLOT), writes rax (RAW_REG).
///   Clobbers xmm1 (FLT_SCRATCH), r8 (BIT_IO_TMP1).
/// - Int→Float: reads rax (RAW_REG), writes xmm0 (FLT_SLOT).
fn emit_repr_convert(
    asm: &mut Assembler,
    src: ComputeRepr,
    dst: ComputeRepr,
) -> Result<(), String> {
    match (src, dst) {
        // Float (F32 in xmm0) → Int (i64 in rax)
        //
        // cast_raw goes through f64: decode → f64 → encode.
        // cvttss2si truncates toward zero and returns 0x8000..00
        // for NaN, ±inf, and out-of-range values. We fix up:
        // - NaN → 0 (matching IntType::encode_f64 for NaN)
        // - The int encode's saturation handles overflow.
        (ComputeRepr::F32, ComputeRepr::Int) => {
            dynasm!(asm
                ; .arch x64
                // Check for NaN: ucomiss sets PF on unordered (NaN).
                ; ucomiss Rx(FLT_SLOT), Rx(FLT_SLOT)
                ; cvttss2si Rq(RAW_REG), Rx(FLT_SLOT)
                // If NaN (PF set), zero out rax.
                ; mov Rq(BIT_IO_TMP1), 0
                ; cmovp Rq(RAW_REG), Rq(BIT_IO_TMP1)
            );
            Ok(())
        }

        // Float (F64 in xmm0) → Int (i64 in rax)
        (ComputeRepr::F64, ComputeRepr::Int) => {
            dynasm!(asm
                ; .arch x64
                ; ucomisd Rx(FLT_SLOT), Rx(FLT_SLOT)
                ; cvttsd2si Rq(RAW_REG), Rx(FLT_SLOT)
                ; mov Rq(BIT_IO_TMP1), 0
                ; cmovp Rq(RAW_REG), Rq(BIT_IO_TMP1)
            );
            Ok(())
        }

        // Int (i64 in rax) → Float (F32 in xmm0)
        //
        // cvtsi2ss converts signed i64 to F32. For large i64 values
        // (> 2^24) there's precision loss, which matches cast_raw's
        // f64 intermediate (f64 → F32 rounds the same way).
        (ComputeRepr::Int, ComputeRepr::F32) => {
            dynasm!(asm
                ; .arch x64
                ; cvtsi2ss Rx(FLT_SLOT), Rq(RAW_REG)
            );
            Ok(())
        }

        // Int (i64 in rax) → Float (F64 in xmm0)
        (ComputeRepr::Int, ComputeRepr::F64) => {
            dynasm!(asm
                ; .arch x64
                ; cvtsi2sd Rx(FLT_SLOT), Rq(RAW_REG)
            );
            Ok(())
        }

        // F32 ↔ F64: widen or narrow the float.
        (ComputeRepr::F32, ComputeRepr::F64) => {
            dynasm!(asm
                ; .arch x64
                ; cvtss2sd Rx(FLT_SLOT), Rx(FLT_SLOT)
            );
            Ok(())
        }
        (ComputeRepr::F64, ComputeRepr::F32) => {
            dynasm!(asm
                ; .arch x64
                ; cvtsd2ss Rx(FLT_SLOT), Rx(FLT_SLOT)
            );
            Ok(())
        }

        // Same repr — should not be called.
        _ => Ok(()),
    }
}

// ─── Select emission ────────────────────────────────────────────────

/// Register used to hold the truthiness boolean across loads.
/// r9 (BIT_IO_TMP2) is not clobbered by address/load/decode.
const COND_REG: u8 = 9;

fn emit_select_group(
    asm: &mut Assembler,
    layout: &BufferLayout,
    group: &AtomGroup<'static, SystemPool>,
    addr_tables: &mut AddressTables,
    codec_tables: &mut CodecTables,
) -> Result<(), String> {
    if group.inputs.len() != 3 {
        return Err(format!(
            "Select group has {} inputs, expected 3",
            group.inputs.len()
        ));
    }
    let output_ref = InputRef::affine(group.base_id, 1);

    if group.count == 1 {
        emit_select_iter(
            asm, layout, group, &output_ref,
            IterVar::Const(group.atom_offset), group.atom_offset,
            addr_tables, codec_tables,
        )
    } else {
        emit_select_loop(
            asm, layout, group, &output_ref,
            group.atom_offset, group.count,
            addr_tables, codec_tables,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_select_iter(
    asm: &mut Assembler,
    layout: &BufferLayout,
    group: &AtomGroup<'static, SystemPool>,
    output: &InputRef,
    iter: IterVar,
    atom_offset: u64,
    addr_tables: &mut AddressTables,
    codec_tables: &mut CodecTables,
) -> Result<(), String> {
    use super::super::prologue::{FLT_SLOT_A, FLT_SLOT_B, FLT_SLOT_C, INT_SLOT_A, INT_SLOT_B};

    let output_dtype = group.output_dtype;
    let out_repr = ComputeRepr::for_dtype(output_dtype);

    // 1. Load condition, test truthiness → COND_REG (r9).
    let cond_info = emit_compute_bit_offset(
        asm, layout, &group.inputs[0], iter, atom_offset,
        BIT_OFF_REG, ADDR_SCRATCH, addr_tables,
    )?;
    emit_load_bits(asm, BUFFER_REG, BIT_OFF_REG, cond_info.n_bits, RAW_REG, ADDR_SCRATCH);
    let cond_repr = ComputeRepr::for_dtype(cond_info.dtype);
    let cond_slot = match cond_repr {
        ComputeRepr::F32 | ComputeRepr::F64 => CodecSlot::Xmm(FLT_SLOT_A),
        ComputeRepr::Int => CodecSlot::Gp(RAW_REG),
    };
    emit_decode(asm, cond_info.dtype, RAW_REG, cond_slot, BIT_IO_TMP1, FLT_SCRATCH, codec_tables)?;

    // Extract truthiness to COND_REG: nonzero → 1, zero → 0.
    match cond_repr {
        ComputeRepr::F32 => {
            // Float truthiness: (raw & 0x7fffffff) != 0.
            dynasm!(asm
                ; .arch x64
                ; vmovd Rd(COND_REG), Rx(FLT_SLOT_A)
                ; and Rd(COND_REG), 0x7fffffff
                ; test Rd(COND_REG), Rd(COND_REG)
                ; setne Rb(COND_REG)
                ; movzx Rd(COND_REG), Rb(COND_REG)
            );
        }
        ComputeRepr::F64 => {
            dynasm!(asm
                ; .arch x64
                ; vmovq Rq(COND_REG), Rx(FLT_SLOT_A)
                ; mov Rq(BIT_IO_TMP1), QWORD 0x7fffffffffffffff_u64 as i64
                ; and Rq(COND_REG), Rq(BIT_IO_TMP1)
                ; test Rq(COND_REG), Rq(COND_REG)
                ; setne Rb(COND_REG)
                ; movzx Rd(COND_REG), Rb(COND_REG)
            );
        }
        ComputeRepr::Int => {
            dynasm!(asm
                ; .arch x64
                ; test Rq(RAW_REG), Rq(RAW_REG)
                ; setne Rb(COND_REG)
                ; movzx Rd(COND_REG), Rb(COND_REG)
            );
        }
    }
    // COND_REG now holds 0 or 1. It survives all subsequent loads.

    // 2. Load x (true branch) → slot A.
    let _x_info = emit_load_decode_input(
        asm, layout, &group.inputs[1], iter, atom_offset,
        addr_tables, codec_tables,
        match out_repr {
            ComputeRepr::F32 | ComputeRepr::F64 => CodecSlot::Xmm(FLT_SLOT_A),
            ComputeRepr::Int => CodecSlot::Gp(INT_SLOT_A),
        },
    )?;

    // For int: stash x before loading y.
    if out_repr == ComputeRepr::Int {
        dynasm!(asm; .arch x64; movq Rx(FLT_SLOT), Rq(INT_SLOT_A));
    }

    // 3. Load y (false branch) → slot B.
    let _y_info = emit_load_decode_input(
        asm, layout, &group.inputs[2], iter, atom_offset,
        addr_tables, codec_tables,
        match out_repr {
            ComputeRepr::F32 | ComputeRepr::F64 => CodecSlot::Xmm(FLT_SLOT_B),
            ComputeRepr::Int => CodecSlot::Gp(INT_SLOT_B),
        },
    )?;

    // For int: restore x.
    if out_repr == ComputeRepr::Int {
        dynasm!(asm; .arch x64; movq Rq(INT_SLOT_A), Rx(FLT_SLOT));
    }

    // 4. Select: result = cond ? x : y → slot C.
    let done = asm.new_dynamic_label();
    match out_repr {
        ComputeRepr::F32 | ComputeRepr::F64 => {
            // Default to y, overwrite with x if cond is truthy.
            dynasm!(asm
                ; .arch x64
                ; vmovaps Rx(FLT_SLOT_C), Rx(FLT_SLOT_B)
                ; test Rd(COND_REG), Rd(COND_REG)
                ; jz =>done
                ; vmovaps Rx(FLT_SLOT_C), Rx(FLT_SLOT_A)
                ; =>done
            );
        }
        ComputeRepr::Int => {
            // cmovnz: if cond != 0, C = x; else C = y.
            dynasm!(asm
                ; .arch x64
                ; mov Rq(super::super::prologue::INT_SLOT_C), Rq(INT_SLOT_B)
                ; test Rd(COND_REG), Rd(COND_REG)
                ; cmovnz Rq(super::super::prologue::INT_SLOT_C), Rq(INT_SLOT_A)
            );
        }
    }

    // 5. Encode + store.
    let result_slot = match out_repr {
        ComputeRepr::F32 | ComputeRepr::F64 => CodecSlot::Xmm(FLT_SLOT_C),
        ComputeRepr::Int => CodecSlot::Gp(super::super::prologue::INT_SLOT_C),
    };
    emit_encode(
        asm, output_dtype, result_slot, RAW_REG,
        BIT_IO_TMP1, BIT_IO_TMP2, FLT_SCRATCH,
    )?;
    let dst_info = emit_compute_bit_offset(
        asm, layout, output, iter, atom_offset,
        BIT_OFF_REG, ADDR_SCRATCH, addr_tables,
    )?;
    emit_store_bits(
        asm, BUFFER_REG, BIT_OFF_REG, dst_info.n_bits, RAW_REG,
        BIT_IO_TMP1, BIT_IO_TMP2, ADDR_SCRATCH,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn emit_select_loop(
    asm: &mut Assembler,
    layout: &BufferLayout,
    group: &AtomGroup<'static, SystemPool>,
    output: &InputRef,
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
    dynasm!(asm; =>loop_top; cmp Rq(LOOP_VAR_REG), Rq(LOOP_END_REG); jge =>loop_exit);
    emit_select_iter(
        asm, layout, group, output,
        IterVar::Reg(LOOP_VAR_REG), 0,
        addr_tables, codec_tables,
    )?;
    dynasm!(asm; add Rq(LOOP_VAR_REG), 1; jmp =>loop_top; =>loop_exit);
    Ok(())
}

// ─── IndirectLoad emission ──────────────────────────────────────────

/// Emit an IndirectLoad group: load a runtime index, look up a value
/// from the table at `table_base + index`, store.
fn emit_indirect_load_group(
    asm: &mut Assembler,
    layout: &BufferLayout,
    group: &AtomGroup<'static, SystemPool>,
    table_base: crate::nano_graph::pattern::AtomId,
    addr_tables: &mut AddressTables,
    codec_tables: &mut CodecTables,
) -> Result<(), String> {
    if group.inputs.len() != 1 {
        return Err(format!(
            "IndirectLoad group has {} inputs, expected 1",
            group.inputs.len()
        ));
    }

    // Look up the table slot at JIT-build time.
    let (table_slot, _) = layout
        .find(table_base)
        .ok_or_else(|| format!("IndirectLoad: no slot for table_base={table_base}"))?;
    let table_bit_offset = table_slot.bit_offset;
    let table_bit_stride = table_slot.bit_stride;
    let table_n_bits = table_slot.elem_bits as u32;
    let table_dtype = table_slot.dtype;

    let output_ref = InputRef::affine(group.base_id, 1);

    if group.count == 1 {
        emit_indirect_load_iter(
            asm, layout, &group.inputs[0], &output_ref,
            table_bit_offset, table_bit_stride, table_n_bits, table_dtype,
            group.output_dtype,
            IterVar::Const(group.atom_offset), group.atom_offset,
            addr_tables, codec_tables,
        )
    } else {
        let start = group.atom_offset as i64;
        let end = (group.atom_offset + group.count) as i64;
        dynasm!(asm
            ; .arch x64
            ; mov Rq(LOOP_VAR_REG), QWORD start
            ; mov Rq(LOOP_END_REG), QWORD end
        );
        let loop_top = asm.new_dynamic_label();
        let loop_exit = asm.new_dynamic_label();
        dynasm!(asm; =>loop_top; cmp Rq(LOOP_VAR_REG), Rq(LOOP_END_REG); jge =>loop_exit);

        emit_indirect_load_iter(
            asm, layout, &group.inputs[0], &output_ref,
            table_bit_offset, table_bit_stride, table_n_bits, table_dtype,
            group.output_dtype,
            IterVar::Reg(LOOP_VAR_REG), 0,
            addr_tables, codec_tables,
        )?;

        dynasm!(asm; add Rq(LOOP_VAR_REG), 1; jmp =>loop_top; =>loop_exit);
        Ok(())
    }
}

/// Emit one iteration of IndirectLoad.
#[allow(clippy::too_many_arguments)]
fn emit_indirect_load_iter(
    asm: &mut Assembler,
    layout: &BufferLayout,
    index_input: &InputRef,
    output: &InputRef,
    table_bit_offset: u64,
    table_bit_stride: u64,
    table_n_bits: u32,
    table_dtype: NumericDType,
    output_dtype: NumericDType,
    iter: IterVar,
    atom_offset: u64,
    addr_tables: &mut AddressTables,
    codec_tables: &mut CodecTables,
) -> Result<(), String> {
    // 1. Load the index value.
    let idx_info = emit_compute_bit_offset(
        asm, layout, index_input, iter, atom_offset,
        BIT_OFF_REG, ADDR_SCRATCH, addr_tables,
    )?;
    emit_load_bits(asm, BUFFER_REG, BIT_OFF_REG, idx_info.n_bits, RAW_REG, ADDR_SCRATCH);

    // 2. Decode index to its compute repr, then extract as u64.
    let idx_repr = ComputeRepr::for_dtype(idx_info.dtype);
    match idx_repr {
        ComputeRepr::F32 => {
            emit_decode(
                asm, idx_info.dtype, RAW_REG,
                CodecSlot::Xmm(FLT_SLOT), BIT_IO_TMP1, FLT_SCRATCH, codec_tables,
            )?;
            // Convert F32 → u64 (truncate).
            dynasm!(asm; .arch x64; vcvttss2si Rq(RAW_REG), Rx(FLT_SLOT));
        }
        ComputeRepr::F64 => {
            emit_decode(
                asm, idx_info.dtype, RAW_REG,
                CodecSlot::Xmm(FLT_SLOT), BIT_IO_TMP1, FLT_SCRATCH, codec_tables,
            )?;
            dynasm!(asm; .arch x64; vcvttsd2si Rq(RAW_REG), Rx(FLT_SLOT));
        }
        ComputeRepr::Int => {
            emit_decode(
                asm, idx_info.dtype, RAW_REG,
                CodecSlot::Gp(RAW_REG), BIT_IO_TMP1, FLT_SCRATCH, codec_tables,
            )?;
            // Value already in rax as i64. Treat as u64.
        }
    }
    // rax now holds the index as u64.

    // 3. Compute table bit offset: r10 = table_bit_offset + index * table_bit_stride.
    let stride = table_bit_stride as i64;
    if (i32::MIN as i64..=i32::MAX as i64).contains(&stride) {
        dynasm!(asm; .arch x64; imul Rq(BIT_OFF_REG), Rq(RAW_REG), stride as i32);
    } else {
        dynasm!(asm
            ; .arch x64
            ; mov Rq(BIT_OFF_REG), QWORD stride
            ; imul Rq(BIT_OFF_REG), Rq(RAW_REG)
        );
    }
    if (i32::MIN as i64..=i32::MAX as i64).contains(&(table_bit_offset as i64)) {
        dynasm!(asm; .arch x64; add Rq(BIT_OFF_REG), table_bit_offset as i32);
    } else {
        dynasm!(asm
            ; .arch x64
            ; mov Rq(ADDR_SCRATCH), QWORD table_bit_offset as i64
            ; add Rq(BIT_OFF_REG), Rq(ADDR_SCRATCH)
        );
    }

    // 4. Load table value → rax.
    emit_load_bits(asm, BUFFER_REG, BIT_OFF_REG, table_n_bits, RAW_REG, ADDR_SCRATCH);

    // 5. Decode table value, encode to output dtype.
    let out_repr = ComputeRepr::for_dtype(output_dtype);
    let val_slot = match out_repr {
        ComputeRepr::F32 | ComputeRepr::F64 => CodecSlot::Xmm(FLT_SLOT),
        ComputeRepr::Int => CodecSlot::Gp(RAW_REG),
    };
    emit_decode(asm, table_dtype, RAW_REG, val_slot, BIT_IO_TMP1, FLT_SCRATCH, codec_tables)?;

    // If table_dtype's repr != output_dtype's repr, convert.
    let table_repr = ComputeRepr::for_dtype(table_dtype);
    if table_repr != out_repr {
        emit_repr_convert(asm, table_repr, out_repr)?;
    }

    let encode_slot = match out_repr {
        ComputeRepr::F32 | ComputeRepr::F64 => CodecSlot::Xmm(FLT_SLOT),
        ComputeRepr::Int => CodecSlot::Gp(RAW_REG),
    };
    emit_encode(
        asm, output_dtype, encode_slot, RAW_REG,
        BIT_IO_TMP1, BIT_IO_TMP2, FLT_SCRATCH,
    )?;

    // 6. Compute output address + store.
    let dst_info = emit_compute_bit_offset(
        asm, layout, output, iter, atom_offset,
        BIT_OFF_REG, ADDR_SCRATCH, addr_tables,
    )?;
    emit_store_bits(
        asm, BUFFER_REG, BIT_OFF_REG, dst_info.n_bits, RAW_REG,
        BIT_IO_TMP1, BIT_IO_TMP2, ADDR_SCRATCH,
    );

    Ok(())
}

// ─── Int wrapping helpers ────────────────────────────────────────────

/// Extract (signed, bits) from a NumericDType for the int op layer.
fn int_dtype_info(dtype: NumericDType) -> Result<(bool, u8), String> {
    match dtype {
        NumericDType::SignedInt(it) => Ok((true, it.bits)),
        NumericDType::UnsignedInt(it) => Ok((false, it.bits)),
        NumericDType::Bool => Ok((false, 1)),
        _ => Err(format!("int_dtype_info: {dtype} is not an integer type")),
    }
}

/// Emit the wrapping mask + sign-extension for integer ops.
///
/// After a 64-bit op, this reduces the result to `bits` width:
/// - Mask to keep only the low `bits` bits
/// - For signed: sign-extend from `bits` back to 64
///
/// No-op for 64-bit types.
fn emit_int_wrap(asm: &mut Assembler, bits: u8, signed: bool, reg: u8) {
    if bits >= 64 {
        return;
    }
    let mask = if bits < 32 {
        (1u64 << bits) - 1
    } else {
        (1u64 << bits) - 1
    };
    if mask <= i32::MAX as u64 {
        dynasm!(asm; .arch x64; and Rq(reg), DWORD mask as i32);
    } else {
        // bits 33..63: need a wide mask. Use a scratch (r8).
        dynasm!(asm
            ; .arch x64
            ; mov Rq(BIT_IO_TMP1), QWORD mask as i64
            ; and Rq(reg), Rq(BIT_IO_TMP1)
        );
    }
    if signed {
        let shift = 64 - bits;
        dynasm!(asm
            ; .arch x64
            ; shl Rq(reg), BYTE shift as i8
            ; sar Rq(reg), BYTE shift as i8
        );
    }
}

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
