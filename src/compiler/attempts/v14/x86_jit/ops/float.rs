//! Float arithmetic on compute slots.
//!
//! Binary ops (Add, Sub, Mul, Div, Min, Max, comparisons, logical)
//! and unary ops on XMM registers. All use the VEX-encoded 3-operand
//! forms so the source operands are preserved.
//!
//! Per dtype contract §5.3, Min/Max use IEEE 754-2008 minNum/maxNum
//! semantics — the non-NaN operand wins when exactly one is NaN. This
//! requires extra emission compared to bare `vminss`/`vmaxss` (which
//! propagates src2 when either operand is NaN on x86).
//!
//! Comparisons return 1.0 for true, 0.0 for false (ONNX convention),
//! emitted as `ucomiss` + `set*` + `cvtsi2ss`.
//!
//! Logical ops (And, Or, Xor, Not) treat nonzero float as truthy
//! (including -0.0 as falsy, NaN as truthy per dtype contract §5.4).

use dynasmrt::x64::Assembler;
use dynasmrt::{DynasmApi, DynasmLabelApi, dynasm};

use super::super::prologue::{FLT_SLOT_A, FLT_SLOT_B, FLT_SLOT_C};
use crate::nano_graph::ops::ScalarBinOp;

/// Emit a float binary op in F32 compute repr.
///
/// Operands in `xmm(FLT_SLOT_A)` and `xmm(FLT_SLOT_B)`.
/// Result written to `xmm(FLT_SLOT_C)`.
///
/// `scratch_gp` is a free GP register, clobbered by comparison /
/// logical paths (used for the `setcc` → `cvtsi2ss` sequence).
pub fn emit_binop_f32(
    asm: &mut Assembler,
    op: ScalarBinOp,
    scratch_gp: u8,
) -> Result<(), String> {
    let a = FLT_SLOT_A;
    let b = FLT_SLOT_B;
    let c = FLT_SLOT_C;
    match op {
        ScalarBinOp::Add => dynasm!(asm; .arch x64; vaddss Rx(c), Rx(a), Rx(b)),
        ScalarBinOp::Sub => dynasm!(asm; .arch x64; vsubss Rx(c), Rx(a), Rx(b)),
        ScalarBinOp::Mul => dynasm!(asm; .arch x64; vmulss Rx(c), Rx(a), Rx(b)),
        ScalarBinOp::Div => dynasm!(asm; .arch x64; vdivss Rx(c), Rx(a), Rx(b)),

        ScalarBinOp::Max => emit_minmax_f32(asm, true),
        ScalarBinOp::Min => emit_minmax_f32(asm, false),

        ScalarBinOp::Equal => emit_cmp_f32(asm, CmpKind::Equal, scratch_gp),
        ScalarBinOp::Greater => emit_cmp_f32(asm, CmpKind::Greater, scratch_gp),
        ScalarBinOp::GreaterOrEqual => emit_cmp_f32(asm, CmpKind::GreaterOrEqual, scratch_gp),
        ScalarBinOp::Less => emit_cmp_f32(asm, CmpKind::Less, scratch_gp),
        ScalarBinOp::LessOrEqual => emit_cmp_f32(asm, CmpKind::LessOrEqual, scratch_gp),

        ScalarBinOp::And => emit_logical_f32(asm, LogicalKind::And, scratch_gp),
        ScalarBinOp::Or => emit_logical_f32(asm, LogicalKind::Or, scratch_gp),
        ScalarBinOp::Xor => emit_logical_f32(asm, LogicalKind::Xor, scratch_gp),

        ScalarBinOp::Mod => emit_fmod_f32(asm),
        ScalarBinOp::IMod => emit_fimod_f32(asm),
        ScalarBinOp::Pow => emit_fpow_f32(asm),

        ScalarBinOp::BitwiseAnd
        | ScalarBinOp::BitwiseOr
        | ScalarBinOp::BitwiseXor
        | ScalarBinOp::BitShiftLeft
        | ScalarBinOp::BitShiftRight => {
            return Err(format!(
                "emit_binop_f32: bitwise op {op:?} not valid for float compute repr"
            ));
        }
    }
    Ok(())
}

/// Emit a float binary op in F64 compute repr.
pub fn emit_binop_f64(
    asm: &mut Assembler,
    op: ScalarBinOp,
    scratch_gp: u8,
) -> Result<(), String> {
    let a = FLT_SLOT_A;
    let b = FLT_SLOT_B;
    let c = FLT_SLOT_C;
    match op {
        ScalarBinOp::Add => dynasm!(asm; .arch x64; vaddsd Rx(c), Rx(a), Rx(b)),
        ScalarBinOp::Sub => dynasm!(asm; .arch x64; vsubsd Rx(c), Rx(a), Rx(b)),
        ScalarBinOp::Mul => dynasm!(asm; .arch x64; vmulsd Rx(c), Rx(a), Rx(b)),
        ScalarBinOp::Div => dynasm!(asm; .arch x64; vdivsd Rx(c), Rx(a), Rx(b)),

        ScalarBinOp::Max => emit_minmax_f64(asm, true),
        ScalarBinOp::Min => emit_minmax_f64(asm, false),

        ScalarBinOp::Equal => emit_cmp_f64(asm, CmpKind::Equal, scratch_gp),
        ScalarBinOp::Greater => emit_cmp_f64(asm, CmpKind::Greater, scratch_gp),
        ScalarBinOp::GreaterOrEqual => emit_cmp_f64(asm, CmpKind::GreaterOrEqual, scratch_gp),
        ScalarBinOp::Less => emit_cmp_f64(asm, CmpKind::Less, scratch_gp),
        ScalarBinOp::LessOrEqual => emit_cmp_f64(asm, CmpKind::LessOrEqual, scratch_gp),

        ScalarBinOp::And => emit_logical_f64(asm, LogicalKind::And, scratch_gp),
        ScalarBinOp::Or => emit_logical_f64(asm, LogicalKind::Or, scratch_gp),
        ScalarBinOp::Xor => emit_logical_f64(asm, LogicalKind::Xor, scratch_gp),

        ScalarBinOp::Mod => emit_fmod_f64(asm),
        ScalarBinOp::IMod => emit_fimod_f64(asm),
        ScalarBinOp::Pow => emit_fpow_f64(asm),

        ScalarBinOp::BitwiseAnd
        | ScalarBinOp::BitwiseOr
        | ScalarBinOp::BitwiseXor
        | ScalarBinOp::BitShiftLeft
        | ScalarBinOp::BitShiftRight => {
            return Err(format!(
                "emit_binop_f64: bitwise op {op:?} not valid for float compute repr"
            ));
        }
    }
    Ok(())
}

// ─── Min/Max with NaN handling ──────────────────────────────────────

/// IEEE 754-2008 minNum/maxNum: the non-NaN operand wins when exactly
/// one is NaN. Both NaN → NaN.
///
/// x86 `vminss`/`vmaxss` propagate src2 when *either* is NaN, so we
/// emit two instructions with swapped operands and combine:
///
///     vmaxss tmp, A, B   ; tmp = max(A,B), but if A is NaN → B
///     vmaxss C, B, A     ; C   = max(B,A), but if B is NaN → A
///     vmaxss C, tmp, C   ; picks the non-NaN result (or NaN if both)
///
/// Wait, that's 3 instructions. Simpler: vmaxss with both orderings
/// gives `max(A,B)` when neither is NaN, `B` when `A` is NaN (first),
/// `A` when `B` is NaN (second). We want the non-NaN, so:
///
///     vmaxss C, A, B     ; if A NaN → B, if B NaN → B(!), if neither → max
///     vmaxss tmp, B, A   ; if B NaN → A, if A NaN → A(!), if neither → max
///
/// Hmm, this doesn't work simply. The canonical approach is:
///
///     vmaxss C, A, B      ; C = vmaxss(A,B)
///     vcmpunordss mask, A, A  ; mask = A is NaN
///     vblendvps C, C, B, mask ; if A was NaN, use B
///
/// But vblendvps is SSE4.1 and operates on packed. Let me use a simpler
/// two-vmax approach which is correct for minNum/maxNum:
///
///     vmaxss C, A, B      ; if A NaN: C=B; if B NaN: C=B (wrong!); else max
///     vmaxss tmp, B, A    ; if B NaN: tmp=A; if A NaN: tmp=A (wrong!); else max
///     ; Need: if both NaN → NaN, if A NaN → B, if B NaN → A, else max
///     ; vmaxss(A,B) gives: A NaN→B, B NaN→B, both NaN→NaN(?), neither→max
///     ; vmaxss(B,A) gives: B NaN→A, A NaN→A, both NaN→NaN(?), neither→max
///     ; So: vmaxss(A,B) is correct when B is not NaN (gives B if A NaN, max if neither)
///     ;     vmaxss(B,A) is correct when A is not NaN (gives A if B NaN, max if neither)
///     ; Combine: use vmaxss(A,B) unless B is NaN → use vmaxss(B,A) instead.
///     ; Detect B NaN: if B NaN, vmaxss(A,B) != vmaxss(B,A) only when A is not NaN.
///     ; Simplest correct: vmaxss(A,B), then vmaxss(result, B, A) — nope.
///
/// Actually the simplest correct approach for minNum:
///     vminss C, B, A    ; C = min(B, A): if B NaN → A (correct!)
///     vminss C, A, C    ; C = min(A, C): if A NaN → C (which was A's correct value)
///                       ;                if C NaN (from both NaN) → C = NaN (correct)
///                       ;                else → min(A, prev_result)
///
/// No wait. Let me just implement the documented two-instruction idiom:
///     vminss C, A, B   ; first operand NaN → returns B
///     vminss C, B, C   ; first operand (B) NaN → returns C (which was correct when A not NaN)
///                      ; if B NaN and A not NaN: first gives B→wait, that's wrong
///
/// I'm going in circles. Let me use the well-known correct approach:
fn emit_minmax_f32(asm: &mut Assembler, is_max: bool) {
    let a = FLT_SLOT_A;
    let b = FLT_SLOT_B;
    let c = FLT_SLOT_C;
    // C = op(A, B): if A is NaN → returns B
    // tmp = op(B, A): if B is NaN → returns A
    // If neither NaN: C = tmp = correct result
    // If A NaN: C = B (correct), tmp = A (NaN, wrong)
    // If B NaN: C = B (NaN, wrong), tmp = A (correct)
    // If both NaN: C = B (NaN), tmp = A (NaN) — both NaN, correct
    //
    // So: pick C when A is not NaN, pick tmp when A is NaN.
    // = vblendvps(C, tmp, A_is_nan_mask)
    //
    // Simpler: C = op(A, B); tmp = op(B, A); final = op(C, tmp)
    // because op(NaN, x) = x, op(x, NaN) = NaN means:
    // - A NaN only: C=B, tmp=NaN → op(B, NaN) = NaN?? No, op returns second when first NaN.
    //   Hmm, vminss(src1, src2) returns src2 when src1 is NaN. So:
    //   vminss(C=B, tmp=NaN_A) → returns NaN_A. Wrong.
    //
    // OK let me just use the one-branch approach.
    if is_max {
        let nan_b = asm.new_dynamic_label();
        let done = asm.new_dynamic_label();
        dynasm!(asm
            ; .arch x64
            // Check if B is NaN (unordered with itself).
            ; vucomiss Rx(b), Rx(b)
            ; jp =>nan_b
            // B is not NaN: vmaxss(A, B) gives correct result
            // (if A NaN → B which is correct, else max).
            ; vmaxss Rx(c), Rx(a), Rx(b)
            ; jmp =>done
            ; =>nan_b
            // B is NaN: result is A (whether A is NaN or not).
            ; vmovss Rx(c), Rx(c), Rx(a)
            ; =>done
        );
    } else {
        let nan_b = asm.new_dynamic_label();
        let done = asm.new_dynamic_label();
        dynasm!(asm
            ; .arch x64
            ; vucomiss Rx(b), Rx(b)
            ; jp =>nan_b
            ; vminss Rx(c), Rx(a), Rx(b)
            ; jmp =>done
            ; =>nan_b
            ; vmovss Rx(c), Rx(c), Rx(a)
            ; =>done
        );
    }
}

fn emit_minmax_f64(asm: &mut Assembler, is_max: bool) {
    let a = FLT_SLOT_A;
    let b = FLT_SLOT_B;
    let c = FLT_SLOT_C;
    if is_max {
        let nan_b = asm.new_dynamic_label();
        let done = asm.new_dynamic_label();
        dynasm!(asm
            ; .arch x64
            ; vucomisd Rx(b), Rx(b)
            ; jp =>nan_b
            ; vmaxsd Rx(c), Rx(a), Rx(b)
            ; jmp =>done
            ; =>nan_b
            ; vmovsd Rx(c), Rx(c), Rx(a)
            ; =>done
        );
    } else {
        let nan_b = asm.new_dynamic_label();
        let done = asm.new_dynamic_label();
        dynasm!(asm
            ; .arch x64
            ; vucomisd Rx(b), Rx(b)
            ; jp =>nan_b
            ; vminsd Rx(c), Rx(a), Rx(b)
            ; jmp =>done
            ; =>nan_b
            ; vmovsd Rx(c), Rx(c), Rx(a)
            ; =>done
        );
    }
}

// ─── Comparisons ────────────────────────────────────────────────────

enum CmpKind {
    Equal,
    Greater,
    GreaterOrEqual,
    Less,
    LessOrEqual,
}

/// Float comparison: result is 1.0f32 for true, 0.0f32 for false.
/// Uses `vucomiss` to set flags, `set*` + `movzx` + `cvtsi2ss`.
fn emit_cmp_f32(asm: &mut Assembler, kind: CmpKind, scratch: u8) {
    let a = FLT_SLOT_A;
    let b = FLT_SLOT_B;
    let c = FLT_SLOT_C;
    dynasm!(asm; .arch x64; vucomiss Rx(a), Rx(b));
    emit_setcc(asm, kind, scratch);
    dynasm!(asm
        ; .arch x64
        ; vcvtsi2ss Rx(c), Rx(c), Rd(scratch)
    );
}

fn emit_cmp_f64(asm: &mut Assembler, kind: CmpKind, scratch: u8) {
    let a = FLT_SLOT_A;
    let b = FLT_SLOT_B;
    let c = FLT_SLOT_C;
    dynasm!(asm; .arch x64; vucomisd Rx(a), Rx(b));
    emit_setcc(asm, kind, scratch);
    dynasm!(asm
        ; .arch x64
        ; vcvtsi2sd Rx(c), Rx(c), Rd(scratch)
    );
}

/// Set `scratch` to 0 or 1 based on the flags left by `vucomiss`/`vucomisd`.
/// Equal uses `sete + setnp` (both ZF=1 and PF=0 for ordered equal).
/// Greater/Less use the unsigned condition codes (CF-based) which
/// correspond to the IEEE ordering after `ucomiss`.
fn emit_setcc(asm: &mut Assembler, kind: CmpKind, scratch: u8) {
    match kind {
        CmpKind::Equal => {
            // Ordered equal: ZF=1 AND PF=0. NaN comparisons set PF.
            dynasm!(asm
                ; .arch x64
                ; sete Rb(scratch)
                ; setnp cl
                ; and Rb(scratch), cl
                ; movzx Rd(scratch), Rb(scratch)
            );
        }
        CmpKind::Greater => {
            // A > B (ordered): CF=0, ZF=0.
            dynasm!(asm; .arch x64; seta Rb(scratch); movzx Rd(scratch), Rb(scratch));
        }
        CmpKind::GreaterOrEqual => {
            // A >= B (ordered): CF=0.
            dynasm!(asm; .arch x64; setae Rb(scratch); movzx Rd(scratch), Rb(scratch));
        }
        CmpKind::Less => {
            // A < B (ordered). ucomiss sets CF=1 when A < B.
            dynasm!(asm; .arch x64; setb Rb(scratch); movzx Rd(scratch), Rb(scratch));
        }
        CmpKind::LessOrEqual => {
            // A <= B (ordered): CF=1 or ZF=1.
            dynasm!(asm; .arch x64; setbe Rb(scratch); movzx Rd(scratch), Rb(scratch));
        }
    }
}

// ─── Logical ops ────────────────────────────────────────────────────

enum LogicalKind {
    And,
    Or,
    Xor,
}

/// Float logical: `is_truthy(A) LOGOP is_truthy(B)` → 1.0 or 0.0.
/// Truthiness: `decode_to_f64(raw) != 0.0` — this means -0.0 is
/// falsy and NaN is truthy.
fn emit_logical_f32(asm: &mut Assembler, kind: LogicalKind, scratch: u8) {
    let a = FLT_SLOT_A;
    let b = FLT_SLOT_B;
    let c = FLT_SLOT_C;
    // Truthiness of A: compare with 0.0. Non-zero (including NaN) is truthy.
    // ucomiss A, 0 → ZF=1 only for ±0. PF=1 for NaN.
    // truthy = !(ZF=1 && PF=0) = (ZF=0 || PF=1)
    // = setne OR setp. Simplest: xorps zero, zero; ucomiss A, zero; setne; also setp; or them.
    //
    // Actually: simpler to use the raw bits. A float is truthy iff
    // its raw bits are not all-zero or 0x80000000 (-0). That's
    // `(raw & 0x7fffffff) != 0`. Let's use that.
    dynasm!(asm
        ; .arch x64
        // scratch = is_truthy(A)
        ; vmovd Rd(scratch), Rx(a)
        ; and Rd(scratch), 0x7fffffff_u32 as i32
        ; test Rd(scratch), Rd(scratch)
        ; setne Rb(scratch)
        ; movzx Rd(scratch), Rb(scratch)
        // cl = is_truthy(B) — rcx is caller-saved scratch
        ; vmovd ecx, Rx(b)
        ; and ecx, 0x7fffffff_u32 as i32
        ; test ecx, ecx
        ; setne cl
        ; movzx ecx, cl
    );
    // Combine.
    match kind {
        LogicalKind::And => dynasm!(asm; .arch x64; and Rd(scratch), ecx),
        LogicalKind::Or => dynasm!(asm; .arch x64; or Rd(scratch), ecx),
        LogicalKind::Xor => dynasm!(asm; .arch x64; xor Rd(scratch), ecx),
    }
    // Convert 0/1 → 0.0/1.0.
    dynasm!(asm
        ; .arch x64
        ; vcvtsi2ss Rx(c), Rx(c), Rd(scratch)
    );
}

fn emit_logical_f64(asm: &mut Assembler, kind: LogicalKind, scratch: u8) {
    let a = FLT_SLOT_A;
    let b = FLT_SLOT_B;
    let c = FLT_SLOT_C;
    // Truthiness of F64: (raw & 0x7fff_ffff_ffff_ffff) != 0.
    dynasm!(asm
        ; .arch x64
        ; vmovq Rq(scratch), Rx(a)
        ; mov rcx, QWORD 0x7fffffffffffffff_u64 as i64
        ; and Rq(scratch), rcx
        ; test Rq(scratch), Rq(scratch)
        ; setne Rb(scratch)
        ; movzx Rd(scratch), Rb(scratch)
        ; vmovq rcx, Rx(b)
        ; mov Rq(BIT_IO_TMP1_REG), QWORD 0x7fffffffffffffff_u64 as i64
        ; and rcx, Rq(BIT_IO_TMP1_REG)
        ; test rcx, rcx
        ; setne cl
        ; movzx ecx, cl
    );
    match kind {
        LogicalKind::And => dynasm!(asm; .arch x64; and Rd(scratch), ecx),
        LogicalKind::Or => dynasm!(asm; .arch x64; or Rd(scratch), ecx),
        LogicalKind::Xor => dynasm!(asm; .arch x64; xor Rd(scratch), ecx),
    }
    dynasm!(asm
        ; .arch x64
        ; vcvtsi2sd Rx(c), Rx(c), Rd(scratch)
    );
}

// Need r8 for the F64 logical path mask.
const BIT_IO_TMP1_REG: u8 = 8;

// ─── Mod / IMod / Pow stubs ─────────────────────────────────────────
// These need libm trampolines. Stubs for now.

fn emit_fmod_f32(_asm: &mut Assembler) {
    todo!("emit_fmod_f32: needs libm fmodf trampoline (P3.E)")
}

fn emit_fmod_f64(_asm: &mut Assembler) {
    todo!("emit_fmod_f64: needs libm fmod trampoline (P3.E)")
}

fn emit_fimod_f32(_asm: &mut Assembler) {
    todo!("emit_fimod_f32: needs libm remainderf trampoline (P3.E)")
}

fn emit_fimod_f64(_asm: &mut Assembler) {
    todo!("emit_fimod_f64: needs libm remainder trampoline (P3.E)")
}

fn emit_fpow_f32(_asm: &mut Assembler) {
    todo!("emit_fpow_f32: needs libm powf trampoline (P3.E)")
}

fn emit_fpow_f64(_asm: &mut Assembler) {
    todo!("emit_fpow_f64: needs libm pow trampoline (P3.E)")
}
