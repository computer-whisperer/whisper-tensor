//! Emit a Reduce group as outer + inner loops.
//!
//! The inner loop accumulates `reduce_count` values into the
//! compute-repr accumulator. Per dtype contract §4.7, the accumulator
//! must be at the *full compute_dtype precision* after every iteration
//! — for narrow compute_dtypes (BF16, F16, etc.) this means an
//! explicit `narrow_to` call from [`super::super::codec::precision`]
//! after each op.
//!
//! # Register layout (in addition to the standard outer-loop regs)
//!
//! - `r9` (REDUCE_K): inner loop counter `k`
//! - `rsi` (6, REDUCE_K_END): inner loop end (`reduce_count`)
//! - `rdi` (7, REDUCE_SRC_BIT): current source bit offset, incremented
//!   by `k_bit_stride` each inner iteration
//! - Float accumulator: `xmm2` (FLT_SLOT_C)
//! - Int accumulator: `rdx` (INT_SLOT_C)

use dynasmrt::x64::Assembler;
use dynasmrt::{DynasmApi, DynasmLabelApi, dynasm};

use crate::compiler::attempts::v14::layout::BufferLayout;
use crate::nano_graph::ops::ReduceKind;
use crate::nano_graph::pattern::{AtomGroup, InputRef};
use crate::numeric_dtype::NumericDType;
use crate::pool::SystemPool;

use super::super::codec::bit_io::{emit_load_bits, emit_store_bits};
use super::super::codec::format::{CodecSlot, CodecTables, ComputeRepr, emit_decode, emit_encode};
use super::super::codec::precision::emit_narrow_to;
use super::super::prologue::{
    BUFFER_REG, FLT_SLOT_A, FLT_SLOT_C, INT_SLOT_C, LOOP_END_REG, LOOP_VAR_REG,
};
use super::address::{AddressTables, IterVar, emit_compute_bit_offset};

/// Inner loop counter `k`.
const REDUCE_K: u8 = 9; // r9
/// Inner loop end (reduce_count).
const REDUCE_K_END: u8 = 6; // rsi
/// Current source bit offset (incremented each inner iteration).
const REDUCE_SRC_BIT: u8 = 7; // rdi
/// Scratch for address computation + bit_io.
const SCRATCH: u8 = 11; // r11
/// Bit offset register for output address.
const BIT_OFF: u8 = 10; // r10
/// Raw bits register.
const RAW: u8 = 0; // rax
/// Scratch GP for codec.
const CODEC_SCRATCH: u8 = 8; // r8
/// Scratch XMM for codec.
const CODEC_XMM_SCRATCH: u8 = 1; // xmm1

pub fn emit_reduce_group(
    asm: &mut Assembler,
    layout: &BufferLayout,
    group: &AtomGroup<'static, SystemPool>,
    kind: ReduceKind,
    reduce_count: u64,
    reduce_stride: i64,
    compute_dtype: NumericDType,
    addr_tables: &mut AddressTables,
    codec_tables: &mut CodecTables,
) -> Result<(), String> {
    if group.inputs.len() != 1 {
        return Err(format!(
            "Reduce group has {} inputs, expected 1",
            group.inputs.len()
        ));
    }

    let repr = ComputeRepr::for_dtype(compute_dtype);
    let output_ref = InputRef::affine(group.base_id, 1);

    // Look up the source slot to determine n_bits and k_bit_stride.
    let src_info = resolve_reduce_source_info(layout, &group.inputs[0], group.atom_offset)?;
    let n_bits = src_info.n_bits;
    let k_bit_stride = reduce_stride * src_info.bit_stride as i64;

    if group.count == 1 {
        emit_reduce_body(
            asm, layout, &group.inputs[0], &output_ref,
            kind, reduce_count, k_bit_stride, compute_dtype, group.output_dtype,
            repr, n_bits, src_info.src_dtype,
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

        emit_reduce_body(
            asm, layout, &group.inputs[0], &output_ref,
            kind, reduce_count, k_bit_stride, compute_dtype, group.output_dtype,
            repr, n_bits, src_info.src_dtype,
            IterVar::Reg(LOOP_VAR_REG), 0,
            addr_tables, codec_tables,
        )?;

        dynasm!(asm; add Rq(LOOP_VAR_REG), 1; jmp =>loop_top; =>loop_exit);
        Ok(())
    }
}

struct ReduceSourceInfo {
    n_bits: u32,
    bit_stride: u64,
    src_dtype: NumericDType,
}

fn resolve_reduce_source_info(
    layout: &BufferLayout,
    input: &InputRef,
    atom_offset: u64,
) -> Result<ReduceSourceInfo, String> {
    // Resolve the input's base atom to get the slot info.
    match input {
        InputRef::Broadcast(id) => {
            let (slot, _) = layout.find(*id)
                .ok_or_else(|| format!("reduce: no slot for Broadcast atom={id}"))?;
            Ok(ReduceSourceInfo {
                n_bits: slot.elem_bits as u32,
                bit_stride: slot.bit_stride,
                src_dtype: slot.dtype,
            })
        }
        InputRef::Strided { base, dim_strides, dim_shape } => {
            let first_offset = crate::compiler::attempts::v14::layout::strided_resolve_offset(
                dim_strides, dim_shape, atom_offset,
            );
            let first_atom = crate::nano_graph::pattern::AtomId(
                ((base.0 as i64) + first_offset) as u64,
            );
            let (slot, _) = layout.find(*base)
                .or_else(|| layout.find(first_atom))
                .ok_or_else(|| format!("reduce: no slot for Strided base={base}"))?;
            Ok(ReduceSourceInfo {
                n_bits: slot.elem_bits as u32,
                bit_stride: slot.bit_stride,
                src_dtype: slot.dtype,
            })
        }
        InputRef::Explicit(ids) if !ids.is_empty() => {
            let (slot, _) = layout.find(ids[0])
                .ok_or_else(|| format!("reduce: no slot for Explicit[0]={}", ids[0]))?;
            Ok(ReduceSourceInfo {
                n_bits: slot.elem_bits as u32,
                bit_stride: slot.bit_stride,
                src_dtype: slot.dtype,
            })
        }
        _ => Err("reduce: empty Explicit InputRef".to_string()),
    }
}

/// Emit one outer iteration of the Reduce body.
#[allow(clippy::too_many_arguments)]
fn emit_reduce_body(
    asm: &mut Assembler,
    layout: &BufferLayout,
    input: &InputRef,
    output: &InputRef,
    kind: ReduceKind,
    reduce_count: u64,
    k_bit_stride: i64,
    compute_dtype: NumericDType,
    output_dtype: NumericDType,
    repr: ComputeRepr,
    n_bits: u32,
    src_dtype: NumericDType,
    iter: IterVar,
    atom_offset: u64,
    addr_tables: &mut AddressTables,
    codec_tables: &mut CodecTables,
) -> Result<(), String> {
    // 1. Compute base address for k=0 → r10, copy to rdi.
    emit_compute_bit_offset(
        asm, layout, input, iter, atom_offset,
        BIT_OFF, SCRATCH, addr_tables,
    )?;
    dynasm!(asm; .arch x64; mov Rq(REDUCE_SRC_BIT), Rq(BIT_OFF));

    // 2. Initialize accumulator.
    emit_reduce_init(asm, kind, compute_dtype, repr)?;

    // 3. Inner loop.
    dynasm!(asm
        ; .arch x64
        ; xor Rq(REDUCE_K), Rq(REDUCE_K)
        ; mov Rq(REDUCE_K_END), QWORD reduce_count as i64
    );
    let inner_top = asm.new_dynamic_label();
    let inner_done = asm.new_dynamic_label();
    dynasm!(asm
        ; =>inner_top
        ; cmp Rq(REDUCE_K), Rq(REDUCE_K_END)
        ; jge =>inner_done
    );

    // 3a. Load source bits from rdi, decode to slot A.
    emit_load_bits(asm, BUFFER_REG, REDUCE_SRC_BIT, n_bits, RAW, SCRATCH);
    let slot_a = match repr {
        ComputeRepr::F32 | ComputeRepr::F64 => CodecSlot::Xmm(FLT_SLOT_A),
        ComputeRepr::Int => CodecSlot::Gp(RAW),
    };
    emit_decode(asm, src_dtype, RAW, slot_a, CODEC_SCRATCH, CODEC_XMM_SCRATCH, codec_tables)?;

    // 3b. Accumulate: acc = acc op val.
    emit_reduce_accum(asm, kind, repr, compute_dtype)?;

    // 3c. Per-step quantization: narrow_to(compute_dtype).
    let acc_slot = match repr {
        ComputeRepr::F32 | ComputeRepr::F64 => CodecSlot::Xmm(FLT_SLOT_C),
        ComputeRepr::Int => CodecSlot::Gp(INT_SLOT_C),
    };
    emit_narrow_to(
        asm, compute_dtype, acc_slot,
        RAW, CODEC_SCRATCH, BIT_OFF, CODEC_XMM_SCRATCH,
        codec_tables,
    )?;

    // 3d. Advance source pointer, increment k.
    if (i32::MIN as i64..=i32::MAX as i64).contains(&k_bit_stride) {
        dynasm!(asm; .arch x64; add Rq(REDUCE_SRC_BIT), k_bit_stride as i32);
    } else {
        dynasm!(asm
            ; .arch x64
            ; mov Rq(SCRATCH), QWORD k_bit_stride
            ; add Rq(REDUCE_SRC_BIT), Rq(SCRATCH)
        );
    }
    dynasm!(asm
        ; add Rq(REDUCE_K), 1
        ; jmp =>inner_top
        ; =>inner_done
    );

    // 4. Encode accumulator → rax, store.
    emit_encode(
        asm, output_dtype, acc_slot, RAW,
        CODEC_SCRATCH, BIT_OFF, CODEC_XMM_SCRATCH,
    )?;
    let dst_info = emit_compute_bit_offset(
        asm, layout, output, iter, atom_offset,
        BIT_OFF, SCRATCH, addr_tables,
    )?;
    emit_store_bits(
        asm, BUFFER_REG, BIT_OFF, dst_info.n_bits, RAW,
        CODEC_SCRATCH, REDUCE_K, SCRATCH,
    );

    Ok(())
}

/// Initialize the reduce accumulator.
fn emit_reduce_init(
    asm: &mut Assembler,
    kind: ReduceKind,
    compute_dtype: NumericDType,
    repr: ComputeRepr,
) -> Result<(), String> {
    let c_xmm = FLT_SLOT_C;
    let c_gp = INT_SLOT_C;

    match (repr, kind) {
        // Float Sum: 0.0
        (ComputeRepr::F32, ReduceKind::Sum) | (ComputeRepr::F64, ReduceKind::Sum) => {
            dynasm!(asm; .arch x64; vxorps Rx(c_xmm), Rx(c_xmm), Rx(c_xmm));
        }
        // Float Prod: 1.0
        (ComputeRepr::F32, ReduceKind::Prod) => {
            dynasm!(asm; .arch x64; mov eax, DWORD 0x3f800000_u32 as i32; vmovd Rx(c_xmm), eax);
        }
        (ComputeRepr::F64, ReduceKind::Prod) => {
            dynasm!(asm; .arch x64; mov rax, QWORD 0x3ff0000000000000_u64 as i64; vmovq Rx(c_xmm), rax);
        }
        // Float Max: -inf
        (ComputeRepr::F32, ReduceKind::Max) => {
            dynasm!(asm; .arch x64; mov eax, DWORD 0xff800000_u32 as i32; vmovd Rx(c_xmm), eax);
        }
        (ComputeRepr::F64, ReduceKind::Max) => {
            dynasm!(asm; .arch x64; mov rax, QWORD 0xfff0000000000000_u64 as i64; vmovq Rx(c_xmm), rax);
        }
        // Float Min: +inf
        (ComputeRepr::F32, ReduceKind::Min) => {
            dynasm!(asm; .arch x64; mov eax, DWORD 0x7f800000_u32 as i32; vmovd Rx(c_xmm), eax);
        }
        (ComputeRepr::F64, ReduceKind::Min) => {
            dynasm!(asm; .arch x64; mov rax, QWORD 0x7ff0000000000000_u64 as i64; vmovq Rx(c_xmm), rax);
        }
        // Int Sum: 0
        (ComputeRepr::Int, ReduceKind::Sum) => {
            dynasm!(asm; .arch x64; xor Rq(c_gp), Rq(c_gp));
        }
        // Int Prod: 1
        (ComputeRepr::Int, ReduceKind::Prod) => {
            dynasm!(asm; .arch x64; mov Rq(c_gp), 1);
        }
        // Int Max: minimum for the dtype
        (ComputeRepr::Int, ReduceKind::Max) => {
            let init = compute_dtype.encode_from_f64(f64::NEG_INFINITY);
            dynasm!(asm; .arch x64; mov Rq(c_gp), QWORD init as i64);
        }
        // Int Min: maximum for the dtype
        (ComputeRepr::Int, ReduceKind::Min) => {
            let init = compute_dtype.encode_from_f64(f64::INFINITY);
            dynasm!(asm; .arch x64; mov Rq(c_gp), QWORD init as i64);
        }
    }
    Ok(())
}

/// Emit the accumulation: acc = acc op val.
///
/// Float: acc in xmm2 (FLT_SLOT_C), val in xmm0 (FLT_SLOT_A).
/// Int: acc in rdx (INT_SLOT_C), val in rax (RAW = INT_SLOT_A).
fn emit_reduce_accum(
    asm: &mut Assembler,
    kind: ReduceKind,
    repr: ComputeRepr,
    compute_dtype: NumericDType,
) -> Result<(), String> {
    let a = FLT_SLOT_A;
    let c = FLT_SLOT_C;

    match (repr, kind) {
        (ComputeRepr::F32, ReduceKind::Sum) => {
            dynasm!(asm; .arch x64; vaddss Rx(c), Rx(c), Rx(a));
        }
        (ComputeRepr::F32, ReduceKind::Prod) => {
            dynasm!(asm; .arch x64; vmulss Rx(c), Rx(c), Rx(a));
        }
        (ComputeRepr::F32, ReduceKind::Max) => {
            // IEEE minNum/maxNum with NaN handling (same as binop Max).
            let nan_b = asm.new_dynamic_label();
            let done = asm.new_dynamic_label();
            dynasm!(asm
                ; .arch x64
                ; vucomiss Rx(a), Rx(a)
                ; jp =>nan_b
                ; vmaxss Rx(c), Rx(c), Rx(a)
                ; jmp =>done
                ; =>nan_b
                // val is NaN → keep acc unchanged.
                ; =>done
            );
        }
        (ComputeRepr::F32, ReduceKind::Min) => {
            let nan_b = asm.new_dynamic_label();
            let done = asm.new_dynamic_label();
            dynasm!(asm
                ; .arch x64
                ; vucomiss Rx(a), Rx(a)
                ; jp =>nan_b
                ; vminss Rx(c), Rx(c), Rx(a)
                ; jmp =>done
                ; =>nan_b
                ; =>done
            );
        }
        (ComputeRepr::F64, ReduceKind::Sum) => {
            dynasm!(asm; .arch x64; vaddsd Rx(c), Rx(c), Rx(a));
        }
        (ComputeRepr::F64, ReduceKind::Prod) => {
            dynasm!(asm; .arch x64; vmulsd Rx(c), Rx(c), Rx(a));
        }
        (ComputeRepr::F64, ReduceKind::Max) => {
            let nan_b = asm.new_dynamic_label();
            let done = asm.new_dynamic_label();
            dynasm!(asm
                ; .arch x64
                ; vucomisd Rx(a), Rx(a)
                ; jp =>nan_b
                ; vmaxsd Rx(c), Rx(c), Rx(a)
                ; jmp =>done
                ; =>nan_b
                ; =>done
            );
        }
        (ComputeRepr::F64, ReduceKind::Min) => {
            let nan_b = asm.new_dynamic_label();
            let done = asm.new_dynamic_label();
            dynasm!(asm
                ; .arch x64
                ; vucomisd Rx(a), Rx(a)
                ; jp =>nan_b
                ; vminsd Rx(c), Rx(c), Rx(a)
                ; jmp =>done
                ; =>nan_b
                ; =>done
            );
        }
        (ComputeRepr::Int, ReduceKind::Sum) => {
            dynasm!(asm; .arch x64; add Rq(INT_SLOT_C), Rq(RAW));
        }
        (ComputeRepr::Int, ReduceKind::Prod) => {
            dynasm!(asm; .arch x64; imul Rq(INT_SLOT_C), Rq(RAW));
        }
        (ComputeRepr::Int, ReduceKind::Max) => {
            let signed = matches!(compute_dtype, NumericDType::SignedInt(_));
            if signed {
                dynasm!(asm; .arch x64; cmp Rq(INT_SLOT_C), Rq(RAW); cmovl Rq(INT_SLOT_C), Rq(RAW));
            } else {
                dynasm!(asm; .arch x64; cmp Rq(INT_SLOT_C), Rq(RAW); cmovb Rq(INT_SLOT_C), Rq(RAW));
            }
        }
        (ComputeRepr::Int, ReduceKind::Min) => {
            let signed = matches!(compute_dtype, NumericDType::SignedInt(_));
            if signed {
                dynasm!(asm; .arch x64; cmp Rq(INT_SLOT_C), Rq(RAW); cmovg Rq(INT_SLOT_C), Rq(RAW));
            } else {
                dynasm!(asm; .arch x64; cmp Rq(INT_SLOT_C), Rq(RAW); cmova Rq(INT_SLOT_C), Rq(RAW));
            }
        }
    }
    Ok(())
}
