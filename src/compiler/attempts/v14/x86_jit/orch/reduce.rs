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
use crate::nano_graph::pattern::{AtomGroup, AtomId, InputRef, NanoGraph};
use crate::numeric_dtype::NumericDType;
use crate::pool::SystemPool;

use super::super::codec::bit_io::{
    emit_load_aligned, emit_load_bits, emit_store_aligned, emit_store_bits,
};
use super::super::codec::format::{CodecSlot, CodecTables, ComputeRepr, emit_decode, emit_encode};
use super::super::codec::precision::emit_narrow_to;
use super::super::prologue::{FLT_SLOT_A, FLT_SLOT_C, INT_SLOT_C, LOOP_VAR_REG};
use super::address::{AddressTables, IterVar, emit_compute_bit_offset};
use super::group::{emit_output_bit_offset, materialize_buffer_base};

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
    graph: &NanoGraph<'static, SystemPool>,
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

    // Check if this reduce inlines a producer (reduce-fold inlining).
    let reduce_gi = graph
        .find_group_idx(group.base_id)
        .expect("reduce group must be in graph");
    let inline_producer = if reduce_gi < layout.inlines_producer.len() {
        layout.inlines_producer[reduce_gi].map(|pi| &graph.groups()[pi])
    } else {
        None
    };

    // When inlining, we don't need the source slot info (no buffer load).
    let src_info_opt = if inline_producer.is_some() {
        None
    } else {
        Some(resolve_reduce_source_info(
            layout,
            &group.inputs[0].input_ref,
            group.atom_offset,
        )?)
    };
    let n_bits = src_info_opt.as_ref().map_or(0, |s| s.n_bits);
    let k_bit_stride = src_info_opt.as_ref().map_or(0, |s| {
        let stride = if s.byte_aligned {
            (s.bit_stride / 8) as i64
        } else {
            s.bit_stride as i64
        };
        reduce_stride * stride
    });
    let src_dtype = src_info_opt.as_ref().map_or(compute_dtype, |s| s.src_dtype);

    if group.count == 1 {
        emit_reduce_body(
            asm,
            layout,
            &group.inputs[0].input_ref,
            group.base_id,
            group.atom_offset,
            kind,
            reduce_count,
            k_bit_stride,
            compute_dtype,
            group.output_dtype,
            repr,
            n_bits,
            src_dtype,
            inline_producer,
            IterVar::Const(group.atom_offset),
            group.atom_offset,
            addr_tables,
            codec_tables,
        )
    } else {
        let start = group.atom_offset as i64;
        let end = (group.atom_offset + group.count) as i64;
        if !(i32::MIN as i64..=i32::MAX as i64).contains(&end) {
            return Err(format!(
                "x86_jit reduce: loop end {end} doesn't fit in i32 — \
                 falling back to pool_eval"
            ));
        }
        dynasm!(asm
            ; .arch x64
            ; mov Rq(LOOP_VAR_REG), QWORD start
        );
        let loop_top = asm.new_dynamic_label();
        let loop_exit = asm.new_dynamic_label();
        dynasm!(asm
            ; =>loop_top
            ; cmp Rq(LOOP_VAR_REG), DWORD end as i32
            ; jge =>loop_exit
        );

        emit_reduce_body(
            asm,
            layout,
            &group.inputs[0].input_ref,
            group.base_id,
            group.atom_offset,
            kind,
            reduce_count,
            k_bit_stride,
            compute_dtype,
            group.output_dtype,
            repr,
            n_bits,
            src_dtype,
            inline_producer,
            IterVar::Reg(LOOP_VAR_REG),
            group.atom_offset,
            addr_tables,
            codec_tables,
        )?;

        dynasm!(asm; add Rq(LOOP_VAR_REG), 1; jmp =>loop_top; =>loop_exit);
        Ok(())
    }
}

struct ReduceSourceInfo {
    n_bits: u32,
    bit_stride: u64,
    src_dtype: NumericDType,
    byte_aligned: bool,
}

fn resolve_reduce_source_info(
    layout: &BufferLayout,
    input: &InputRef,
    atom_offset: u64,
) -> Result<ReduceSourceInfo, String> {
    // Resolve the input's base atom to get the slot info.
    match input {
        InputRef::Broadcast(id) => {
            let (slot, _) = layout
                .find(*id)
                .ok_or_else(|| format!("reduce: no slot for Broadcast atom={id}"))?;
            Ok(ReduceSourceInfo {
                n_bits: slot.elem_bits as u32,
                bit_stride: slot.bit_stride,
                src_dtype: slot.dtype,
                byte_aligned: slot.is_byte_aligned() && matches!(slot.elem_bits, 8 | 16 | 32 | 64),
            })
        }
        InputRef::Strided {
            base,
            dim_strides,
            dim_shape,
        } => {
            let first_offset = crate::compiler::attempts::v14::layout::strided_resolve_offset(
                dim_strides,
                dim_shape,
                atom_offset,
            );
            let first_atom =
                crate::nano_graph::pattern::AtomId(((base.0 as i64) + first_offset) as u64);
            let (slot, _) = layout
                .find(*base)
                .or_else(|| layout.find(first_atom))
                .ok_or_else(|| format!("reduce: no slot for Strided base={base}"))?;
            Ok(ReduceSourceInfo {
                n_bits: slot.elem_bits as u32,
                bit_stride: slot.bit_stride,
                src_dtype: slot.dtype,
                byte_aligned: slot.is_byte_aligned() && matches!(slot.elem_bits, 8 | 16 | 32 | 64),
            })
        }
        InputRef::Explicit(ids) if !ids.is_empty() => {
            let (slot, _) = layout
                .find(ids[0])
                .ok_or_else(|| format!("reduce: no slot for Explicit[0]={}", ids[0]))?;
            Ok(ReduceSourceInfo {
                n_bits: slot.elem_bits as u32,
                bit_stride: slot.bit_stride,
                src_dtype: slot.dtype,
                byte_aligned: slot.is_byte_aligned() && matches!(slot.elem_bits, 8 | 16 | 32 | 64),
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
    output_base: AtomId,
    output_atom_offset: u64,
    kind: ReduceKind,
    reduce_count: u64,
    k_bit_stride: i64,
    compute_dtype: NumericDType,
    output_dtype: NumericDType,
    repr: ComputeRepr,
    n_bits: u32,
    src_dtype: NumericDType,
    inline_producer: Option<&AtomGroup<'static, SystemPool>>,
    iter: IterVar,
    atom_offset: u64,
    addr_tables: &mut AddressTables,
    codec_tables: &mut CodecTables,
) -> Result<(), String> {
    let acc_slot = match repr {
        ComputeRepr::F32 | ComputeRepr::F64 => CodecSlot::Xmm(FLT_SLOT_C),
        ComputeRepr::Int => CodecSlot::Gp(INT_SLOT_C),
    };

    if let Some(producer) = inline_producer {
        // ── Reduce-fold inlining: re-evaluate producer per k-iteration ──
        //
        // Instead of loading from a pre-computed buffer, we compute the
        // producer's expression inline for each k. REDUCE_SRC_BIT (rdi)
        // holds the flat producer atom index (outer * reduce_count + k),
        // used as the IterVar for emit_op_compute.

        // 1. Initialize flat index: rsi = outer_iter * reduce_count.
        //
        // We use REDUCE_K_END (rsi=6) for the flat producer index
        // instead of REDUCE_SRC_BIT (rdi=7) because rdi is clobbered
        // by OVERFLOW_BASE_SCRATCH in bbase() during emit_op_compute.
        // rsi is not touched by any codec/address function. We compare
        // k against reduce_count via an immediate instead of the
        // register.
        let rc = reduce_count as i64;
        match iter {
            IterVar::Const(c) => {
                let base_idx = c as i64 * rc;
                dynasm!(asm; .arch x64; mov Rq(REDUCE_K_END), QWORD base_idx);
            }
            IterVar::Reg(r) => {
                if (i32::MIN as i64..=i32::MAX as i64).contains(&rc) {
                    dynasm!(asm; .arch x64; imul Rq(REDUCE_K_END), Rq(r), rc as i32);
                } else {
                    dynasm!(asm; .arch x64
                        ; mov Rq(REDUCE_K_END), QWORD rc
                        ; imul Rq(REDUCE_K_END), Rq(r)
                    );
                }
            }
        }

        // 2. Initialize accumulator.
        emit_reduce_init(asm, kind, compute_dtype, repr)?;

        // 3. Inner k-loop. Compare k against immediate reduce_count
        // since REDUCE_K_END holds the flat index now.
        dynasm!(asm
            ; .arch x64
            ; xor Rq(REDUCE_K), Rq(REDUCE_K)
        );
        let inner_top = asm.new_dynamic_label();
        let inner_done = asm.new_dynamic_label();
        let rc_i32 = if (i32::MIN as i64..=i32::MAX as i64).contains(&rc) {
            rc as i32
        } else {
            return Err(format!(
                "x86_jit reduce-inline: reduce_count {rc} doesn't fit in i32"
            ));
        };
        dynasm!(asm
            ; =>inner_top
            ; cmp Rq(REDUCE_K), DWORD rc_i32
            ; jge =>inner_done
        );

        // 3a. Evaluate producer at flat index rsi.
        //
        // emit_op_compute puts its result in FLT_SLOT_C (xmm2),
        // which is also the reduce accumulator. Save the accumulator
        // to xmm3 (ACC_SAVE_XMM) before the op compute. xmm3 is not
        // used by any codec/ops function (they use xmm0/1/2 only).
        const ACC_SAVE_XMM: u8 = 3;
        match repr {
            ComputeRepr::F32 | ComputeRepr::F64 => {
                dynasm!(asm; .arch x64; vmovaps Rx(ACC_SAVE_XMM), Rx(FLT_SLOT_C));
            }
            ComputeRepr::Int => {
                dynasm!(asm; .arch x64; push Rq(INT_SLOT_C));
            }
        }
        let result_slot = super::group::emit_op_compute(
            asm,
            layout,
            producer,
            IterVar::Reg(REDUCE_K_END),
            producer.atom_offset,
            None,
            addr_tables,
            codec_tables,
        )?;

        // Narrow the producer's result to its output dtype. This
        // matches the precision loss that would occur if the value
        // went through a buffer store+load round-trip (encode to
        // output_dtype, decode back). Without this, the fused path
        // would silently use full compute-repr precision for the
        // intermediate, violating the dtype contract.
        emit_narrow_to(
            asm,
            producer.output_dtype,
            result_slot,
            RAW,
            CODEC_SCRATCH,
            BIT_OFF,
            CODEC_XMM_SCRATCH,
            codec_tables,
        )?;

        // Move result to slot A, then restore the accumulator.
        // The result is in FLT_SLOT_C (same as acc) — move to A first,
        // then restore acc from the saved copy.
        match repr {
            ComputeRepr::F32 | ComputeRepr::F64 => {
                // xmm0 = result (from xmm2)
                dynasm!(asm; .arch x64; vmovaps Rx(FLT_SLOT_A), Rx(FLT_SLOT_C));
                // xmm2 = saved accumulator (from xmm3)
                dynasm!(asm; .arch x64; vmovaps Rx(FLT_SLOT_C), Rx(ACC_SAVE_XMM));
            }
            ComputeRepr::Int => {
                dynasm!(asm; .arch x64
                    ; mov Rq(RAW), Rq(INT_SLOT_C)
                    ; pop Rq(INT_SLOT_C)
                );
            }
        }

        // 3b. Accumulate: acc = acc op val.
        emit_reduce_accum(asm, kind, repr, compute_dtype)?;

        // 3c. Per-step quantization.
        emit_narrow_to(
            asm,
            compute_dtype,
            acc_slot,
            RAW,
            CODEC_SCRATCH,
            BIT_OFF,
            CODEC_XMM_SCRATCH,
            codec_tables,
        )?;

        // 3d. Advance flat index (rsi), increment k.
        dynasm!(asm
            ; add Rq(REDUCE_K_END), 1
            ; add Rq(REDUCE_K), 1
            ; jmp =>inner_top
            ; =>inner_done
        );
    } else {
        // ── Standard path: load from pre-computed buffer ──

        // 1. Compute base address for k=0 → r10, copy to rdi.
        let src_info = emit_compute_bit_offset(
            asm,
            layout,
            input,
            iter,
            atom_offset,
            BIT_OFF,
            SCRATCH,
            None,
            addr_tables,
        )?;
        let src_fast_reg = layout.buffer_bases.reg_for_opt(src_info.buffer_id);
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

        // 3a. Load source bits, decode to slot A.
        let src_buffer_reg = match src_fast_reg {
            Some(reg) => reg,
            None => {
                dynasm!(asm
                    ; .arch x64
                    ; mov Rq(CODEC_SCRATCH), QWORD [Rq(super::super::prologue::BUFFER_PTRS_REG)
                        + (src_info.buffer_id as i32) * 8]
                );
                CODEC_SCRATCH
            }
        };
        if src_info.byte_aligned {
            emit_load_aligned(asm, src_buffer_reg, REDUCE_SRC_BIT, n_bits, RAW);
        } else {
            emit_load_bits(asm, src_buffer_reg, REDUCE_SRC_BIT, n_bits, RAW, SCRATCH);
        }
        let slot_a = match repr {
            ComputeRepr::F32 | ComputeRepr::F64 => CodecSlot::Xmm(FLT_SLOT_A),
            ComputeRepr::Int => CodecSlot::Gp(RAW),
        };
        emit_decode(
            asm,
            src_dtype,
            RAW,
            slot_a,
            CODEC_SCRATCH,
            CODEC_XMM_SCRATCH,
            codec_tables,
        )?;

        // 3b. Accumulate.
        emit_reduce_accum(asm, kind, repr, compute_dtype)?;

        // 3c. Per-step quantization.
        emit_narrow_to(
            asm,
            compute_dtype,
            acc_slot,
            RAW,
            CODEC_SCRATCH,
            BIT_OFF,
            CODEC_XMM_SCRATCH,
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
    }

    // 4. Encode accumulator → rax, store.
    let dst_info = emit_output_bit_offset(
        asm,
        layout,
        output_base,
        output_atom_offset,
        iter,
        BIT_OFF,
        SCRATCH,
        None,
    )?;
    let dst_base = materialize_buffer_base(
        asm,
        layout,
        dst_info.buffer_id,
        super::group::OVERFLOW_BASE_SCRATCH,
    );

    if dst_info.byte_aligned {
        // Direct XMM store for F32/F64 — skip encode + store_bits.
        if output_dtype == NumericDType::F32 {
            if let CodecSlot::Xmm(xmm) = acc_slot {
                dynasm!(asm; .arch x64
                    ; movd DWORD [Rq(dst_base) + Rq(BIT_OFF)], Rx(xmm)
                );
                return Ok(());
            }
        }
        if output_dtype == NumericDType::F64 {
            if let CodecSlot::Xmm(xmm) = acc_slot {
                dynasm!(asm; .arch x64
                    ; movq QWORD [Rq(dst_base) + Rq(BIT_OFF)], Rx(xmm)
                );
                return Ok(());
            }
        }
        // Other byte-aligned: encode to GP, then direct store.
        emit_encode(
            asm,
            output_dtype,
            acc_slot,
            RAW,
            CODEC_SCRATCH,
            BIT_OFF,
            CODEC_XMM_SCRATCH,
        )?;
        emit_store_aligned(asm, dst_base, BIT_OFF, dst_info.n_bits, RAW);
    } else {
        emit_encode(
            asm,
            output_dtype,
            acc_slot,
            RAW,
            CODEC_SCRATCH,
            BIT_OFF,
            CODEC_XMM_SCRATCH,
        )?;
        emit_store_bits(
            asm,
            dst_base,
            BIT_OFF,
            dst_info.n_bits,
            RAW,
            CODEC_SCRATCH,
            REDUCE_K,
            SCRATCH,
        );
    }

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
