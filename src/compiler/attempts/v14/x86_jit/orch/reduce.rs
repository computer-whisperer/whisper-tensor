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
use crate::nano_graph::pattern::{AtomGroup, AtomId, GraphConstantId, InputRef, NanoGraph};
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

#[allow(clippy::too_many_arguments)]
pub fn emit_reduce_group(
    asm: &mut Assembler,
    layout: &BufferLayout,
    graph: &NanoGraph<'static, SystemPool>,
    group: &AtomGroup<'static, SystemPool>,
    gi: usize,
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

    super::group::emit_atom_body_loop(
        asm,
        &layout.buffer_bases,
        group.atom_offset,
        group.count,
        gi,
        &layout.group_sym_dims[gi],
        |asm, iter, sym_ctx| {
            let sym_ctx_in = super::address::input_sym_ctx(
                sym_ctx,
                &group.sym_dims,
                &group.inputs[0].sym_dim_map,
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
                iter,
                group.atom_offset,
                sym_ctx_in,
                sym_ctx,
                addr_tables,
                codec_tables,
            )
        },
    )
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
///
/// When `sym_ctx` is `Some`, every input/output address in this body
/// is shifted by `sym_i * elem_bits` via the address layer, so the
/// caller's sym inner loop visits each `sym_flat` slot of the atom's
/// max-stride region.
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
    sym_ctx_in: Option<super::address::SymCtx>,
    sym_ctx_out: Option<super::address::SymCtx>,
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
        // For Int repr the `push INT_SLOT_C` above shifted rsp by -8,
        // so any [rsp + sym_i_rsp_off] access inside emit_op_compute
        // must add 8 to find the real sym_i slot. Float repr saves
        // the accumulator in xmm3 without touching rsp.
        let producer_sym_ctx = match (repr, sym_ctx_out) {
            (ComputeRepr::Int, Some(ctx)) => Some(super::address::SymCtx {
                sym_i_rsp_off: ctx.sym_i_rsp_off + 8,
                ..ctx
            }),
            _ => sym_ctx_out,
        };
        let result_slot = super::group::emit_op_compute(
            asm,
            layout,
            producer,
            IterVar::Reg(REDUCE_K_END),
            producer.atom_offset,
            producer_sym_ctx,
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

        // `emit_op_compute` always lands its result in the C slot.
        // Move it to slot A for the accumulator step, then restore
        // the saved accumulator back into C.
        let _ = result_slot; // slot is fixed by emit_op_compute's contract
        match repr {
            ComputeRepr::F32 | ComputeRepr::F64 => {
                dynasm!(asm; .arch x64; vmovaps Rx(FLT_SLOT_A), Rx(FLT_SLOT_C));
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
            sym_ctx_in,
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
        sym_ctx_out,
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

/// Emit a SymReduce group — reduces over one of the producer's sym
/// axes. Supports any N ≥ 1 producer sym axes and any axis
/// 0..=N-1; the consumer has N-1 sym axes (the reduced one removed).
///
/// The reduce count (= `gc_values[producer_sym_dims[axis]]`) and the
/// per-k step within the producer atom are both runtime values. The
/// S-decomposition (consumer_sym_flat S → producer_sym_flat at k=0)
/// specializes by axis position:
///
/// - **axis = 0**: producer_sym_flat(S, k) = k · inner_extent_prod + S.
///   No division. Per-k step = `inner_extent_prod * elem_bits` (runtime).
/// - **axis = N−1**: producer_sym_flat(S, k) = S · e_axis + k.
///   No division. Per-k step = `elem_bits` (compile-time).
/// - **axis middle**: S_outer = S / inner_extent_prod, S_inner = S %
///   inner_extent_prod. producer_sym_flat(S, k) = (S_outer · e_axis + k)
///   · inner_extent_prod + S_inner. One runtime `div`.
#[allow(clippy::too_many_arguments)]
pub fn emit_sym_reduce_group(
    asm: &mut Assembler,
    layout: &BufferLayout,
    graph: &NanoGraph<'static, SystemPool>,
    group: &AtomGroup<'static, SystemPool>,
    gi: usize,
    kind: ReduceKind,
    axis: usize,
    compute_dtype: NumericDType,
    addr_tables: &mut AddressTables,
    codec_tables: &mut CodecTables,
) -> Result<(), String> {
    if group.inputs.len() != 1 {
        return Err(format!(
            "SymReduce group has {} inputs, expected 1",
            group.inputs.len()
        ));
    }

    let producer_sym_dims = resolve_producer_sym_dims(graph, layout, &group.inputs[0].input_ref)?;
    if axis >= producer_sym_dims.len() {
        return Err(format!(
            "SymReduce: axis {axis} out of range for producer with {} sym_dims",
            producer_sym_dims.len()
        ));
    }
    if producer_sym_dims.len() != group.sym_dims.len() + 1 {
        return Err(format!(
            "SymReduce: producer has {} sym_dims, consumer has {}, expected producer = consumer + 1",
            producer_sym_dims.len(),
            group.sym_dims.len()
        ));
    }

    let src_info =
        resolve_reduce_source_info(layout, &group.inputs[0].input_ref, group.atom_offset)?;
    let repr = ComputeRepr::for_dtype(compute_dtype);
    let n_bits = src_info.n_bits;
    let elem_step_bits: i32 = if src_info.byte_aligned {
        (n_bits / 8) as i32
    } else {
        n_bits as i32
    };

    super::group::emit_atom_body_loop(
        asm,
        &layout.buffer_bases,
        group.atom_offset,
        group.count,
        gi,
        &layout.group_sym_dims[gi],
        |asm, iter, sym_ctx| {
            emit_sym_reduce_body(
                asm,
                layout,
                &group.inputs[0].input_ref,
                group.base_id,
                group.atom_offset,
                kind,
                &producer_sym_dims,
                axis,
                elem_step_bits,
                src_info.byte_aligned,
                compute_dtype,
                group.output_dtype,
                repr,
                n_bits,
                src_info.src_dtype,
                iter,
                group.atom_offset,
                sym_ctx,
                addr_tables,
                codec_tables,
            )
        },
    )
}

/// Resolve the producer's full `sym_dims` list from the input's
/// referenced atom. Producer may be a NanoGraph group or an external
/// input tensor (whose sym_dims were populated onto the slot at layout
/// time).
fn resolve_producer_sym_dims(
    graph: &NanoGraph<'static, SystemPool>,
    layout: &BufferLayout,
    input: &InputRef,
) -> Result<Vec<GraphConstantId>, String> {
    let probe = match input {
        InputRef::Broadcast(id) => *id,
        InputRef::Strided { base, .. } => *base,
        InputRef::Explicit(ids) => *ids
            .first()
            .ok_or_else(|| "SymReduce: empty Explicit input".to_string())?,
    };
    if let Some(gi) = graph.find_group_idx(probe) {
        return Ok(graph.groups()[gi].sym_dims.clone());
    }
    if let Some((slot, _)) = layout.find(probe) {
        if !slot.sym_dims.is_empty() {
            return Ok(slot.sym_dims.clone());
        }
    }
    Err(format!(
        "SymReduce: producer atom {probe} has no resolvable sym_dims"
    ))
}

/// Return the stack offset to `gc_values_ptr` from the current `rsp`,
/// accounting for `emit_atom_body_loop`'s 16-byte reserve when
/// `sym_ctx` is `Some`, and any additional `sym_i_rsp_off` shift.
fn gc_values_off(
    bases: &crate::compiler::attempts::v14::layout::BufferBases,
    sym_ctx: Option<super::address::SymCtx>,
) -> Result<i32, String> {
    let base = super::super::prologue::gc_values_stack_offset(bases);
    match sym_ctx {
        Some(ctx) => base
            .checked_add(ctx.sym_i_rsp_off)
            .and_then(|v| v.checked_add(16))
            .ok_or_else(|| {
                format!(
                    "SymReduce: gc_values offset overflow (base={base}, shift={})",
                    ctx.sym_i_rsp_off
                )
            }),
        None => Ok(base),
    }
}

/// Byte offset of a GraphConstantId slot within the gc_values array.
fn gc_elem_byte_off(gc: GraphConstantId) -> Result<i32, String> {
    (gc.0 as i64)
        .checked_mul(8)
        .and_then(|v| i32::try_from(v).ok())
        .ok_or_else(|| format!("SymReduce: gc index {} overflow", gc.0))
}

/// Emit `dst = ∏ gc_values[gc] for gc in dims` using `ptr_reg` as the
/// gc_values_ptr cache. Caller must set up `ptr_reg` with the gc
/// values pointer beforehand. Rejects on empty `dims` (callers guard).
fn emit_fold_extent_prod(
    asm: &mut Assembler,
    ptr_reg: u8,
    dst_reg: u8,
    dims: &[GraphConstantId],
) -> Result<(), String> {
    if dims.is_empty() {
        return Err("SymReduce: emit_fold_extent_prod called with empty dims".to_string());
    }
    let first_off = gc_elem_byte_off(dims[0])?;
    dynasm!(asm
        ; .arch x64
        ; mov Rq(dst_reg), QWORD [Rq(ptr_reg) + first_off]
    );
    for gc in &dims[1..] {
        let off = gc_elem_byte_off(*gc)?;
        dynasm!(asm
            ; .arch x64
            ; imul Rq(dst_reg), QWORD [Rq(ptr_reg) + off]
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn emit_sym_reduce_body(
    asm: &mut Assembler,
    layout: &BufferLayout,
    input: &InputRef,
    output_base: AtomId,
    output_atom_offset: u64,
    kind: ReduceKind,
    producer_sym_dims: &[GraphConstantId],
    axis: usize,
    elem_step_bits: i32,
    byte_aligned: bool,
    compute_dtype: NumericDType,
    output_dtype: NumericDType,
    repr: ComputeRepr,
    n_bits: u32,
    src_dtype: NumericDType,
    iter: IterVar,
    atom_offset: u64,
    sym_ctx: Option<super::address::SymCtx>,
    addr_tables: &mut AddressTables,
    codec_tables: &mut CodecTables,
) -> Result<(), String> {
    let acc_slot = match repr {
        ComputeRepr::F32 | ComputeRepr::F64 => CodecSlot::Xmm(FLT_SLOT_C),
        ComputeRepr::Int => CodecSlot::Gp(INT_SLOT_C),
    };
    let axis_gc = producer_sym_dims[axis];
    let axis_gc_off = gc_elem_byte_off(axis_gc)?;
    let inner_dims = &producer_sym_dims[axis + 1..];
    let inner_is_runtime = !inner_dims.is_empty();
    let is_axis_zero = axis == 0;
    let is_axis_last = axis == producer_sym_dims.len() - 1;

    // 1. Compute producer atom_i's sym=0 base address into BIT_OFF.
    //    Use `SymCtx::address_only` so the address layer:
    //      - forwards the consumer's rsp_delta when emit_runtime_sym_prod
    //        needs to reach gc_values (critical when we're inside
    //        emit_atom_body_loop's 16-byte reserve),
    //      - but skips adding `sym_i * elem_bits` — we step through
    //        producer sym axes manually below (the consumer's sym_i
    //        maps to a producer coord at `axis`, not at the end).
    let addr_sym_ctx = sym_ctx.map(|c| super::address::SymCtx::address_only(c.sym_i_rsp_off));
    let src_info = emit_compute_bit_offset(
        asm,
        layout,
        input,
        iter,
        atom_offset,
        BIT_OFF,
        SCRATCH,
        addr_sym_ctx,
        addr_tables,
    )?;
    let src_fast_reg = layout.buffer_bases.reg_for_opt(src_info.buffer_id);

    // 2. S-decomposition + load e_axis into REDUCE_K_END. If the
    //    consumer has no sym axes, S = 0 and no per-sym shift is
    //    needed. If `inner_dims` is empty (axis=N-1) the per-k step
    //    is `elem_step_bits` (compile-time); otherwise we compute
    //    `inner_extent_prod * elem_step_bits` at runtime and stash
    //    it on the stack.
    let gc_off = gc_values_off(&layout.buffer_bases, sym_ctx)?;
    let sym_i_off: Option<i32> = sym_ctx.map(|ctx| ctx.sym_i_rsp_off);

    // SCRATCH will hold R_inner on entry to the k-loop when runtime.
    let r_inner_on_stack = inner_is_runtime;

    match sym_i_off {
        None => {
            // Consumer has no sym axes (N=1 axis=0 narrow case).
            // No S-dependent shift. Just load e_axis.
            dynasm!(asm
                ; .arch x64
                ; mov Rq(CODEC_SCRATCH), QWORD [rsp + gc_off]
                ; mov Rq(REDUCE_K_END), QWORD [Rq(CODEC_SCRATCH) + axis_gc_off]
            );
            // For N=1 axis=0, inner_dims is empty → r_inner_on_stack = false.
        }
        Some(s_off) if is_axis_zero => {
            // axis=0 with consumer sym axes: inner_dims non-empty.
            // S_outer = 0. offset_elems = S. per-k stride = inner_extent_prod.
            dynasm!(asm
                ; .arch x64
                ; mov Rq(CODEC_SCRATCH), QWORD [rsp + gc_off]
            );
            // SCRATCH = inner_extent_prod (fold multi-sym).
            emit_fold_extent_prod(asm, CODEC_SCRATCH, SCRATCH, inner_dims)?;
            // REDUCE_K_END = e_axis.
            dynasm!(asm
                ; .arch x64
                ; mov Rq(REDUCE_K_END), QWORD [Rq(CODEC_SCRATCH) + axis_gc_off]
            );
            // RAW = S (consumer sym_flat), offset_bits = S * elem_step_bits.
            dynasm!(asm
                ; .arch x64
                ; mov Rq(RAW), QWORD [rsp + s_off]
                ; imul Rq(RAW), Rq(RAW), elem_step_bits
                ; add Rq(BIT_OFF), Rq(RAW)
            );
            // R_inner = inner_extent_prod * elem_step_bits.
            dynasm!(asm
                ; .arch x64
                ; imul Rq(SCRATCH), Rq(SCRATCH), elem_step_bits
            );
        }
        Some(s_off) if is_axis_last => {
            // axis = N-1: inner_extent_prod = 1 compile-time. S_inner = 0,
            // S_outer = S. offset_elems = S * e_axis. per-k stride =
            // elem_step_bits compile-time.
            dynasm!(asm
                ; .arch x64
                ; mov Rq(CODEC_SCRATCH), QWORD [rsp + gc_off]
                ; mov Rq(REDUCE_K_END), QWORD [Rq(CODEC_SCRATCH) + axis_gc_off]
                // RAW = S; RAW *= e_axis; RAW *= elem_step_bits.
                ; mov Rq(RAW), QWORD [rsp + s_off]
                ; imul Rq(RAW), Rq(REDUCE_K_END)
                ; imul Rq(RAW), Rq(RAW), elem_step_bits
                ; add Rq(BIT_OFF), Rq(RAW)
            );
        }
        Some(s_off) => {
            // Middle axis: runtime div for S_outer / S_inner.
            dynasm!(asm
                ; .arch x64
                ; mov Rq(CODEC_SCRATCH), QWORD [rsp + gc_off]
            );
            // SCRATCH = inner_extent_prod.
            emit_fold_extent_prod(asm, CODEC_SCRATCH, SCRATCH, inner_dims)?;
            // Divide S by inner_extent_prod:
            //   rax = S, rdx = 0, div SCRATCH → rax = S_outer, rdx = S_inner.
            dynasm!(asm
                ; .arch x64
                ; mov rax, QWORD [rsp + s_off]
                ; xor rdx, rdx
                ; div Rq(SCRATCH)
            );
            // REDUCE_K_END = e_axis.
            dynasm!(asm
                ; .arch x64
                ; mov Rq(REDUCE_K_END), QWORD [Rq(CODEC_SCRATCH) + axis_gc_off]
                // offset_elems = S_outer * e_axis * inner_extent_prod + S_inner
                //              = rax * REDUCE_K_END * SCRATCH + rdx
                ; imul rax, Rq(REDUCE_K_END)
                ; imul rax, Rq(SCRATCH)
                ; add rax, rdx
                // offset_bits = offset_elems * elem_step_bits
                ; imul rax, rax, elem_step_bits
                ; add Rq(BIT_OFF), rax
            );
            // R_inner = inner_extent_prod * elem_step_bits.
            dynasm!(asm
                ; .arch x64
                ; imul Rq(SCRATCH), Rq(SCRATCH), elem_step_bits
            );
        }
    }

    // 3. Copy final source bit offset into REDUCE_SRC_BIT; stash
    //    R_inner on the stack when the per-k step is runtime. After
    //    the sub, any prior rsp-relative access (sym_i, gc_values)
    //    would shift by 16 — we're done with them.
    dynasm!(asm; .arch x64; mov Rq(REDUCE_SRC_BIT), Rq(BIT_OFF));
    if r_inner_on_stack {
        dynasm!(asm
            ; .arch x64
            ; sub rsp, 16
            ; mov QWORD [rsp + 0], Rq(SCRATCH)
        );
    }

    // 4. Initialize accumulator.
    emit_reduce_init(asm, kind, compute_dtype, repr)?;

    // 5. Inner k-loop. Empty reduce (REDUCE_K_END == 0) is skipped
    //    by the leading cmp/jge.
    dynasm!(asm
        ; .arch x64
        ; xor Rq(REDUCE_K), Rq(REDUCE_K)
    );
    let inner_top = asm.new_dynamic_label();
    let inner_done = asm.new_dynamic_label();
    dynasm!(asm
        ; =>inner_top
        ; cmp Rq(REDUCE_K), Rq(REDUCE_K_END)
        ; jge =>inner_done
    );

    // 5a. Load source bits, decode to slot A.
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
    if byte_aligned {
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

    // 5b. Accumulate.
    emit_reduce_accum(asm, kind, repr, compute_dtype)?;

    // 5c. Per-step quantization to match the dtype contract.
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

    // 5d. Advance source pointer by one producer-sym-axis step,
    //     increment k.
    if r_inner_on_stack {
        dynasm!(asm; .arch x64; add Rq(REDUCE_SRC_BIT), QWORD [rsp + 0]);
    } else {
        dynasm!(asm; .arch x64; add Rq(REDUCE_SRC_BIT), elem_step_bits);
    }
    dynasm!(asm
        ; add Rq(REDUCE_K), 1
        ; jmp =>inner_top
        ; =>inner_done
    );

    // 6. Release R_inner's stack slot before computing the output
    //    address (emit_output_bit_offset uses rsp-relative offsets
    //    into the consumer sym frame).
    if r_inner_on_stack {
        dynasm!(asm; .arch x64; add rsp, 16);
    }

    // 7. Encode accumulator and store at the consumer's (atom, sym_i)
    //    slot. sym_ctx here is the consumer's outer sym_ctx (from
    //    `emit_atom_body_loop`), so the output address steps correctly
    //    through consumer sym slots.
    let dst_info = emit_output_bit_offset(
        asm,
        layout,
        output_base,
        output_atom_offset,
        iter,
        BIT_OFF,
        SCRATCH,
        sym_ctx,
    )?;
    let dst_base = materialize_buffer_base(
        asm,
        layout,
        dst_info.buffer_id,
        super::group::OVERFLOW_BASE_SCRATCH,
    );

    if dst_info.byte_aligned {
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
