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
use crate::nano_graph::ops::{ScalarBinOp, ScalarUnaryOp};

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

// ─── Binary libm trampolines ────────────────────────────────────────

/// Emit a call to a binary libm function `fn(f32, f32) -> f32`.
/// Inputs are already in xmm0 (slot A) and xmm1 (slot B) — the
/// System V first two float arguments. Result lands in xmm0, then
/// is copied to slot C (xmm2).
fn emit_libm2_f32(asm: &mut Assembler, fn_ptr: *const ()) {
    dynasm!(asm
        ; .arch x64
        ; mov rax, QWORD fn_ptr as i64
        ; call rax
        ; vmovaps Rx(FLT_SLOT_C), xmm0
    );
}

fn emit_libm2_f64(asm: &mut Assembler, fn_ptr: *const ()) {
    dynasm!(asm
        ; .arch x64
        ; mov rax, QWORD fn_ptr as i64
        ; call rax
        ; vmovaps Rx(FLT_SLOT_C), xmm0
    );
}

unsafe extern "C" {
    fn fmodf(a: f32, b: f32) -> f32;
    fn fmod(a: f64, b: f64) -> f64;
    fn powf(a: f32, b: f32) -> f32;
    fn pow(a: f64, b: f64) -> f64;
}

fn emit_fmod_f32(asm: &mut Assembler) {
    emit_libm2_f32(asm, fmodf as *const ());
}
fn emit_fmod_f64(asm: &mut Assembler) {
    emit_libm2_f64(asm, fmod as *const ());
}
fn emit_fpow_f32(asm: &mut Assembler) {
    emit_libm2_f32(asm, powf as *const ());
}
fn emit_fpow_f64(asm: &mut Assembler) {
    emit_libm2_f64(asm, pow as *const ());
}

/// IMod (Euclidean modulo): `a - floor(a/b) * b`.
/// For floats, this differs from fmod when signs differ.
fn emit_fimod_f32(asm: &mut Assembler) {
    // Compute via: result = a - floor(a/b) * b
    let a = FLT_SLOT_A;
    let b = FLT_SLOT_B;
    let c = FLT_SLOT_C;
    dynasm!(asm
        ; .arch x64
        ; vdivss Rx(c), Rx(a), Rx(b)       // c = a / b
        ; vroundss Rx(c), Rx(c), Rx(c), 1   // c = floor(a / b)
        ; vmulss Rx(c), Rx(c), Rx(b)        // c = floor(a / b) * b
        ; vsubss Rx(c), Rx(a), Rx(c)        // c = a - floor(a / b) * b
    );
}

fn emit_fimod_f64(asm: &mut Assembler) {
    let a = FLT_SLOT_A;
    let b = FLT_SLOT_B;
    let c = FLT_SLOT_C;
    dynasm!(asm
        ; .arch x64
        ; vdivsd Rx(c), Rx(a), Rx(b)
        ; vroundsd Rx(c), Rx(c), Rx(c), 1
        ; vmulsd Rx(c), Rx(c), Rx(b)
        ; vsubsd Rx(c), Rx(a), Rx(c)
    );
}

// ─── Float unary ops ────────────────────────────────────────────────

/// Emit a float unary op in F32 compute repr.
///
/// Input in `xmm(FLT_SLOT_A)`. Result in `xmm(FLT_SLOT_C)`.
/// `scratch_gp` is a free GP register, clobbered by some paths.
///
/// **Libm calls** clobber all caller-saved registers (GP and XMM).
/// The orchestration layer must ensure callee-saved registers (r12–r14)
/// hold the only live state across a unary op call.
pub fn emit_unop_f32(
    asm: &mut Assembler,
    op: ScalarUnaryOp,
    scratch_gp: u8,
) -> Result<(), String> {
    let a = FLT_SLOT_A;
    let c = FLT_SLOT_C;
    match op {
        ScalarUnaryOp::Neg => {
            // Flip sign bit via XOR with 0x80000000.
            dynasm!(asm
                ; .arch x64
                ; mov Rd(scratch_gp), DWORD 0x80000000_u32 as i32
                ; vmovd Rx(c), Rd(scratch_gp)
                ; vxorps Rx(c), Rx(a), Rx(c)
            );
        }
        ScalarUnaryOp::Abs => {
            // Clear sign bit via AND with 0x7fffffff.
            dynasm!(asm
                ; .arch x64
                ; mov Rd(scratch_gp), 0x7fffffff
                ; vmovd Rx(c), Rd(scratch_gp)
                ; vandps Rx(c), Rx(a), Rx(c)
            );
        }
        ScalarUnaryOp::Sqrt => {
            dynasm!(asm; .arch x64; vsqrtss Rx(c), Rx(c), Rx(a));
        }
        ScalarUnaryOp::Reciprocal => {
            // 1.0 / A
            dynasm!(asm
                ; .arch x64
                ; mov Rd(scratch_gp), DWORD 0x3f800000_u32 as i32  // 1.0f
                ; vmovd Rx(c), Rd(scratch_gp)
                ; vdivss Rx(c), Rx(c), Rx(a)
            );
        }
        ScalarUnaryOp::Floor => {
            dynasm!(asm; .arch x64; vroundss Rx(c), Rx(c), Rx(a), 1);
        }
        ScalarUnaryOp::Ceil => {
            dynasm!(asm; .arch x64; vroundss Rx(c), Rx(c), Rx(a), 2);
        }
        ScalarUnaryOp::Round => {
            dynasm!(asm; .arch x64; vroundss Rx(c), Rx(c), Rx(a), 0);
        }
        ScalarUnaryOp::Not => {
            // Float Not: truthy(A) → 0.0, falsy(A) → 1.0.
            // Truthy = (raw & 0x7fffffff) != 0.
            dynasm!(asm
                ; .arch x64
                ; vmovd Rd(scratch_gp), Rx(a)
                ; and Rd(scratch_gp), 0x7fffffff
                ; test Rd(scratch_gp), Rd(scratch_gp)
                ; sete Rb(scratch_gp)
                ; movzx Rd(scratch_gp), Rb(scratch_gp)
                ; vcvtsi2ss Rx(c), Rx(c), Rd(scratch_gp)
            );
        }
        ScalarUnaryOp::IsNan => {
            // 1.0 if NaN, 0.0 otherwise.
            dynasm!(asm
                ; .arch x64
                ; vucomiss Rx(a), Rx(a)
                ; setp Rb(scratch_gp)
                ; movzx Rd(scratch_gp), Rb(scratch_gp)
                ; vcvtsi2ss Rx(c), Rx(c), Rd(scratch_gp)
            );
        }
        ScalarUnaryOp::IsInf { detect_positive, detect_negative } => {
            // Check if raw bits match ±inf pattern.
            let inf_pos: u32 = 0x7f800000;
            let inf_neg: u32 = 0xff800000;
            dynasm!(asm
                ; .arch x64
                ; vmovd Rd(scratch_gp), Rx(a)
                ; xor ecx, ecx  // result = 0
            );
            if detect_positive {
                dynasm!(asm
                    ; cmp Rd(scratch_gp), DWORD inf_pos as i32
                    ; sete cl
                );
            }
            if detect_negative {
                dynasm!(asm
                    ; cmp Rd(scratch_gp), DWORD inf_neg as i32
                    ; sete Rb(scratch_gp)
                    ; or cl, Rb(scratch_gp)
                );
            }
            dynasm!(asm
                ; .arch x64
                ; movzx Rd(scratch_gp), cl
                ; vcvtsi2ss Rx(c), Rx(c), Rd(scratch_gp)
            );
        }
        ScalarUnaryOp::Sign => {
            // -1.0 if A < 0, 0.0 if A is ±0 or NaN, 1.0 if A > 0.
            let neg = asm.new_dynamic_label();
            let zero_or_nan = asm.new_dynamic_label();
            let done = asm.new_dynamic_label();
            dynasm!(asm
                ; .arch x64
                ; vxorps Rx(c), Rx(c), Rx(c)  // 0.0
                ; vucomiss Rx(a), Rx(c)
                ; jp =>zero_or_nan             // NaN
                ; je =>zero_or_nan             // ±0
                ; jb =>neg                     // A < 0
                // A > 0
                ; mov Rd(scratch_gp), DWORD 0x3f800000_u32 as i32
                ; vmovd Rx(c), Rd(scratch_gp)
                ; jmp =>done
                ; =>neg
                ; mov Rd(scratch_gp), DWORD 0xbf800000_u32 as i32
                ; vmovd Rx(c), Rd(scratch_gp)
                ; jmp =>done
                ; =>zero_or_nan
                ; vxorps Rx(c), Rx(c), Rx(c)
                ; =>done
            );
        }
        ScalarUnaryOp::BitwiseNot => {
            return Err("emit_unop_f32: BitwiseNot not valid for float".to_string());
        }
        // Libm trampolines — input already in xmm0 (slot A = System V arg0).
        ScalarUnaryOp::Exp => emit_libm1_f32(asm, expf as *const ()),
        ScalarUnaryOp::Ln => emit_libm1_f32(asm, logf as *const ()),
        ScalarUnaryOp::Tanh => emit_libm1_f32(asm, tanhf as *const ()),
        ScalarUnaryOp::Erf => emit_libm1_f32(asm, erff as *const ()),
        ScalarUnaryOp::Sin => emit_libm1_f32(asm, sinf as *const ()),
        ScalarUnaryOp::Cos => emit_libm1_f32(asm, cosf as *const ()),
        ScalarUnaryOp::Tan => emit_libm1_f32(asm, tanf as *const ()),
        ScalarUnaryOp::Asin => emit_libm1_f32(asm, asinf as *const ()),
        ScalarUnaryOp::Acos => emit_libm1_f32(asm, acosf as *const ()),
        ScalarUnaryOp::Atan => emit_libm1_f32(asm, atanf as *const ()),
        ScalarUnaryOp::Sinh => emit_libm1_f32(asm, sinhf as *const ()),
        ScalarUnaryOp::Cosh => emit_libm1_f32(asm, coshf as *const ()),
        ScalarUnaryOp::Asinh => emit_libm1_f32(asm, asinhf as *const ()),
        ScalarUnaryOp::Acosh => emit_libm1_f32(asm, acoshf as *const ()),
        ScalarUnaryOp::Atanh => emit_libm1_f32(asm, atanhf as *const ()),
        ScalarUnaryOp::Log1p => emit_libm1_f32(asm, log1pf as *const ()),
    }
    Ok(())
}

/// Emit a float unary op in F64 compute repr.
pub fn emit_unop_f64(
    asm: &mut Assembler,
    op: ScalarUnaryOp,
    scratch_gp: u8,
) -> Result<(), String> {
    let a = FLT_SLOT_A;
    let c = FLT_SLOT_C;
    match op {
        ScalarUnaryOp::Neg => {
            dynasm!(asm
                ; .arch x64
                ; mov Rq(scratch_gp), QWORD 0x8000000000000000_u64 as i64
                ; vmovq Rx(c), Rq(scratch_gp)
                ; vxorpd Rx(c), Rx(a), Rx(c)
            );
        }
        ScalarUnaryOp::Abs => {
            dynasm!(asm
                ; .arch x64
                ; mov Rq(scratch_gp), QWORD 0x7fffffffffffffff_u64 as i64
                ; vmovq Rx(c), Rq(scratch_gp)
                ; vandpd Rx(c), Rx(a), Rx(c)
            );
        }
        ScalarUnaryOp::Sqrt => {
            dynasm!(asm; .arch x64; vsqrtsd Rx(c), Rx(c), Rx(a));
        }
        ScalarUnaryOp::Reciprocal => {
            dynasm!(asm
                ; .arch x64
                ; mov Rq(scratch_gp), QWORD 0x3ff0000000000000_u64 as i64
                ; vmovq Rx(c), Rq(scratch_gp)
                ; vdivsd Rx(c), Rx(c), Rx(a)
            );
        }
        ScalarUnaryOp::Floor => {
            dynasm!(asm; .arch x64; vroundsd Rx(c), Rx(c), Rx(a), 1);
        }
        ScalarUnaryOp::Ceil => {
            dynasm!(asm; .arch x64; vroundsd Rx(c), Rx(c), Rx(a), 2);
        }
        ScalarUnaryOp::Round => {
            dynasm!(asm; .arch x64; vroundsd Rx(c), Rx(c), Rx(a), 0);
        }
        ScalarUnaryOp::Not => {
            dynasm!(asm
                ; .arch x64
                ; vmovq Rq(scratch_gp), Rx(a)
                ; mov rcx, QWORD 0x7fffffffffffffff_u64 as i64
                ; and Rq(scratch_gp), rcx
                ; test Rq(scratch_gp), Rq(scratch_gp)
                ; sete Rb(scratch_gp)
                ; movzx Rd(scratch_gp), Rb(scratch_gp)
                ; vcvtsi2sd Rx(c), Rx(c), Rd(scratch_gp)
            );
        }
        ScalarUnaryOp::IsNan => {
            dynasm!(asm
                ; .arch x64
                ; vucomisd Rx(a), Rx(a)
                ; setp Rb(scratch_gp)
                ; movzx Rd(scratch_gp), Rb(scratch_gp)
                ; vcvtsi2sd Rx(c), Rx(c), Rd(scratch_gp)
            );
        }
        ScalarUnaryOp::IsInf { detect_positive, detect_negative } => {
            let inf_pos: u64 = 0x7ff0000000000000;
            let inf_neg: u64 = 0xfff0000000000000;
            dynasm!(asm
                ; .arch x64
                ; vmovq Rq(scratch_gp), Rx(a)
                ; xor ecx, ecx
            );
            if detect_positive {
                dynasm!(asm
                    ; mov rax, QWORD inf_pos as i64
                    ; cmp Rq(scratch_gp), rax
                    ; sete cl
                );
            }
            if detect_negative {
                dynasm!(asm
                    ; mov rax, QWORD inf_neg as i64
                    ; cmp Rq(scratch_gp), rax
                    ; sete Rb(scratch_gp)
                    ; or cl, Rb(scratch_gp)
                );
            }
            dynasm!(asm
                ; .arch x64
                ; movzx Rd(scratch_gp), cl
                ; vcvtsi2sd Rx(c), Rx(c), Rd(scratch_gp)
            );
        }
        ScalarUnaryOp::Sign => {
            let neg = asm.new_dynamic_label();
            let zero_or_nan = asm.new_dynamic_label();
            let done = asm.new_dynamic_label();
            dynasm!(asm
                ; .arch x64
                ; vxorpd Rx(c), Rx(c), Rx(c)
                ; vucomisd Rx(a), Rx(c)
                ; jp =>zero_or_nan
                ; je =>zero_or_nan
                ; jb =>neg
                ; mov Rq(scratch_gp), QWORD 0x3ff0000000000000_u64 as i64
                ; vmovq Rx(c), Rq(scratch_gp)
                ; jmp =>done
                ; =>neg
                ; mov Rq(scratch_gp), QWORD 0xbff0000000000000_u64 as i64
                ; vmovq Rx(c), Rq(scratch_gp)
                ; jmp =>done
                ; =>zero_or_nan
                ; vxorpd Rx(c), Rx(c), Rx(c)
                ; =>done
            );
        }
        ScalarUnaryOp::BitwiseNot => {
            return Err("emit_unop_f64: BitwiseNot not valid for float".to_string());
        }
        ScalarUnaryOp::Exp => emit_libm1_f64(asm, exp as *const ()),
        ScalarUnaryOp::Ln => emit_libm1_f64(asm, log as *const ()),
        ScalarUnaryOp::Tanh => emit_libm1_f64(asm, tanh as *const ()),
        ScalarUnaryOp::Erf => emit_libm1_f64(asm, erf as *const ()),
        ScalarUnaryOp::Sin => emit_libm1_f64(asm, sin as *const ()),
        ScalarUnaryOp::Cos => emit_libm1_f64(asm, cos as *const ()),
        ScalarUnaryOp::Tan => emit_libm1_f64(asm, tan as *const ()),
        ScalarUnaryOp::Asin => emit_libm1_f64(asm, asin as *const ()),
        ScalarUnaryOp::Acos => emit_libm1_f64(asm, acos as *const ()),
        ScalarUnaryOp::Atan => emit_libm1_f64(asm, atan as *const ()),
        ScalarUnaryOp::Sinh => emit_libm1_f64(asm, sinh as *const ()),
        ScalarUnaryOp::Cosh => emit_libm1_f64(asm, cosh as *const ()),
        ScalarUnaryOp::Asinh => emit_libm1_f64(asm, asinh as *const ()),
        ScalarUnaryOp::Acosh => emit_libm1_f64(asm, acosh as *const ()),
        ScalarUnaryOp::Atanh => emit_libm1_f64(asm, atanh as *const ()),
        ScalarUnaryOp::Log1p => emit_libm1_f64(asm, log1p as *const ()),
    }
    Ok(())
}

// ─── Unary libm trampolines ────────────────────────────────────────

/// Emit a call to a unary libm function `fn(f32) -> f32`.
/// Input already in xmm0 (slot A = System V first float arg).
/// Result lands in xmm0, copied to slot C (xmm2).
///
/// Clobbers all caller-saved registers (GP + XMM).
fn emit_libm1_f32(asm: &mut Assembler, fn_ptr: *const ()) {
    dynasm!(asm
        ; .arch x64
        ; mov rax, QWORD fn_ptr as i64
        ; call rax
        ; vmovaps Rx(FLT_SLOT_C), xmm0
    );
}

fn emit_libm1_f64(asm: &mut Assembler, fn_ptr: *const ()) {
    dynasm!(asm
        ; .arch x64
        ; mov rax, QWORD fn_ptr as i64
        ; call rax
        ; vmovaps Rx(FLT_SLOT_C), xmm0
    );
}

unsafe extern "C" {
    fn expf(x: f32) -> f32;
    fn logf(x: f32) -> f32;
    fn tanhf(x: f32) -> f32;
    fn erff(x: f32) -> f32;
    fn sinf(x: f32) -> f32;
    fn cosf(x: f32) -> f32;
    fn tanf(x: f32) -> f32;
    fn asinf(x: f32) -> f32;
    fn acosf(x: f32) -> f32;
    fn atanf(x: f32) -> f32;
    fn sinhf(x: f32) -> f32;
    fn coshf(x: f32) -> f32;
    fn asinhf(x: f32) -> f32;
    fn acoshf(x: f32) -> f32;
    fn atanhf(x: f32) -> f32;
    fn log1pf(x: f32) -> f32;

    fn exp(x: f64) -> f64;
    fn log(x: f64) -> f64;
    fn tanh(x: f64) -> f64;
    fn erf(x: f64) -> f64;
    fn sin(x: f64) -> f64;
    fn cos(x: f64) -> f64;
    fn tan(x: f64) -> f64;
    fn asin(x: f64) -> f64;
    fn acos(x: f64) -> f64;
    fn atan(x: f64) -> f64;
    fn sinh(x: f64) -> f64;
    fn cosh(x: f64) -> f64;
    fn asinh(x: f64) -> f64;
    fn acosh(x: f64) -> f64;
    fn atanh(x: f64) -> f64;
    fn log1p(x: f64) -> f64;
}
