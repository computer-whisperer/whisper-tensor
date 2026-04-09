//! Integer arithmetic on compute slots.
//!
//! Wrapping Add/Sub/Mul, signed and unsigned Div (with div-by-zero → 0
//! and signed MIN/-1 → MIN), Mod/IMod, comparisons, bitwise ops, and
//! shifts. Results land in `INT_SLOT_C` (rdx = 2).
//!
//! Per dtype contract §5.2, all integer arithmetic is **wrapping**: the
//! op produces the mathematical result reduced modulo 2^bits. The
//! caller (`orch::group`) applies the width mask + sign-extension
//! after the op returns.
//!
//! # Performance debt: Pow trampoline
//!
//! Int Pow calls a custom Rust `extern "C"` function via absolute
//! pointer (same mechanism as the float libm trampolines). The
//! saturating i128 exponentiation loop could be implemented inline
//! as x86 assembly to eliminate the call boundary.
//!
//! # Register layout
//!
//! - Slot A: `rax` (0) — first operand
//! - Slot B: `rcx` (1) — second operand
//! - Slot C: `rdx` (2) — result
//!
//! `idiv` / `div` use `rdx:rax` as the dividend, so Div/Mod clobber
//! all three slots. The code handles this by sequencing carefully.

use dynasmrt::x64::Assembler;
use dynasmrt::{DynasmApi, DynasmLabelApi, dynasm};

use super::super::prologue::{INT_SLOT_A, INT_SLOT_B, INT_SLOT_C};
use crate::nano_graph::ops::{ScalarBinOp, ScalarUnaryOp};

/// Emit an integer binary op.
///
/// Operands in `INT_SLOT_A` (rax) and `INT_SLOT_B` (rcx).
/// Result written to `INT_SLOT_C` (rdx).
///
/// `signed` and `bits` describe the compute dtype — used for signed
/// vs unsigned division/comparison/shift selection. The caller
/// applies the wrapping mask after this returns.
///
/// `scratch_gp` is a free GP register for temporaries. Must not be
/// rax, rcx, or rdx.
pub fn emit_binop_int(
    asm: &mut Assembler,
    op: ScalarBinOp,
    signed: bool,
    bits: u8,
    scratch_gp: u8,
) -> Result<(), String> {
    let a = INT_SLOT_A; // rax = 0
    let b = INT_SLOT_B; // rcx = 1
    let c = INT_SLOT_C; // rdx = 2

    match op {
        // ── Arithmetic ──────────────────────────────────────────
        ScalarBinOp::Add => {
            dynasm!(asm; .arch x64; mov Rq(c), Rq(a); add Rq(c), Rq(b));
        }
        ScalarBinOp::Sub => {
            dynasm!(asm; .arch x64; mov Rq(c), Rq(a); sub Rq(c), Rq(b));
        }
        ScalarBinOp::Mul => {
            // imul r64, r64 gives the low 64 bits (wrapping).
            dynasm!(asm; .arch x64; mov Rq(c), Rq(a); imul Rq(c), Rq(b));
        }
        ScalarBinOp::Div => emit_int_div(asm, signed, bits, scratch_gp),
        ScalarBinOp::Mod => emit_int_mod(asm, signed, bits, scratch_gp),
        ScalarBinOp::IMod => emit_int_imod(asm, signed, bits, scratch_gp),
        ScalarBinOp::Pow => emit_int_pow(asm, signed, bits),

        // ── Min / Max ───────────────────────────────────────────
        ScalarBinOp::Max => {
            dynasm!(asm; .arch x64; cmp Rq(a), Rq(b));
            if signed {
                dynasm!(asm; .arch x64; mov Rq(c), Rq(a); cmovl Rq(c), Rq(b));
            } else {
                dynasm!(asm; .arch x64; mov Rq(c), Rq(a); cmovb Rq(c), Rq(b));
            }
        }
        ScalarBinOp::Min => {
            dynasm!(asm; .arch x64; cmp Rq(a), Rq(b));
            if signed {
                dynasm!(asm; .arch x64; mov Rq(c), Rq(a); cmovg Rq(c), Rq(b));
            } else {
                dynasm!(asm; .arch x64; mov Rq(c), Rq(a); cmova Rq(c), Rq(b));
            }
        }

        // ── Comparisons → 0 / 1 ────────────────────────────────
        ScalarBinOp::Equal => {
            dynasm!(asm; .arch x64; cmp Rq(a), Rq(b); sete Rb(c); movzx Rq(c), Rb(c));
        }
        ScalarBinOp::Greater => {
            dynasm!(asm; .arch x64; cmp Rq(a), Rq(b));
            if signed {
                dynasm!(asm; .arch x64; setg Rb(c); movzx Rq(c), Rb(c));
            } else {
                dynasm!(asm; .arch x64; seta Rb(c); movzx Rq(c), Rb(c));
            }
        }
        ScalarBinOp::GreaterOrEqual => {
            dynasm!(asm; .arch x64; cmp Rq(a), Rq(b));
            if signed {
                dynasm!(asm; .arch x64; setge Rb(c); movzx Rq(c), Rb(c));
            } else {
                dynasm!(asm; .arch x64; setae Rb(c); movzx Rq(c), Rb(c));
            }
        }
        ScalarBinOp::Less => {
            dynasm!(asm; .arch x64; cmp Rq(a), Rq(b));
            if signed {
                dynasm!(asm; .arch x64; setl Rb(c); movzx Rq(c), Rb(c));
            } else {
                dynasm!(asm; .arch x64; setb Rb(c); movzx Rq(c), Rb(c));
            }
        }
        ScalarBinOp::LessOrEqual => {
            dynasm!(asm; .arch x64; cmp Rq(a), Rq(b));
            if signed {
                dynasm!(asm; .arch x64; setle Rb(c); movzx Rq(c), Rb(c));
            } else {
                dynasm!(asm; .arch x64; setbe Rb(c); movzx Rq(c), Rb(c));
            }
        }

        // ── Logical (truthiness) → 0 / 1 ───────────────────────
        ScalarBinOp::And => {
            // truthy(A) && truthy(B) → 0/1.
            dynasm!(asm
                ; .arch x64
                ; test Rq(a), Rq(a)
                ; setne Rb(c)
                ; test Rq(b), Rq(b)
                ; setne Rb(scratch_gp)
                ; and Rb(c), Rb(scratch_gp)
                ; movzx Rq(c), Rb(c)
            );
        }
        ScalarBinOp::Or => {
            dynasm!(asm
                ; .arch x64
                ; test Rq(a), Rq(a)
                ; setne Rb(c)
                ; test Rq(b), Rq(b)
                ; setne Rb(scratch_gp)
                ; or Rb(c), Rb(scratch_gp)
                ; movzx Rq(c), Rb(c)
            );
        }
        ScalarBinOp::Xor => {
            dynasm!(asm
                ; .arch x64
                ; test Rq(a), Rq(a)
                ; setne Rb(c)
                ; test Rq(b), Rq(b)
                ; setne Rb(scratch_gp)
                ; xor Rb(c), Rb(scratch_gp)
                ; movzx Rq(c), Rb(c)
            );
        }

        // ── Bitwise (raw bits) ──────────────────────────────────
        ScalarBinOp::BitwiseAnd => {
            dynasm!(asm; .arch x64; mov Rq(c), Rq(a); and Rq(c), Rq(b));
        }
        ScalarBinOp::BitwiseOr => {
            dynasm!(asm; .arch x64; mov Rq(c), Rq(a); or Rq(c), Rq(b));
        }
        ScalarBinOp::BitwiseXor => {
            dynasm!(asm; .arch x64; mov Rq(c), Rq(a); xor Rq(c), Rq(b));
        }
        ScalarBinOp::BitShiftLeft => {
            // Shift amount from B (in cl). Already in rcx.
            // Shift amount is modulo width per §5.8.
            if bits < 64 {
                dynasm!(asm; .arch x64; and cl, BYTE (bits - 1) as i8);
            }
            dynasm!(asm; .arch x64; mov Rq(c), Rq(a); shl Rq(c), cl);
        }
        ScalarBinOp::BitShiftRight => {
            if bits < 64 {
                dynasm!(asm; .arch x64; and cl, BYTE (bits - 1) as i8);
            }
            if signed {
                // Arithmetic shift right (preserves sign).
                dynasm!(asm; .arch x64; mov Rq(c), Rq(a); sar Rq(c), cl);
            } else {
                // Logical shift right (zeros from left).
                dynasm!(asm; .arch x64; mov Rq(c), Rq(a); shr Rq(c), cl);
            }
        }
    }
    Ok(())
}

// ─── Division helpers ───────────────────────────────────────────────

/// Emit integer division: A / B → C (quotient).
///
/// - Division by zero returns 0.
/// - Signed I64 MIN / -1 returns MIN (wraps).
/// - Uses `idiv` (signed) or `div` (unsigned).
fn emit_int_div(asm: &mut Assembler, signed: bool, bits: u8, _scratch: u8) {
    let a = INT_SLOT_A;
    let b = INT_SLOT_B;
    let c = INT_SLOT_C;

    let div_zero = asm.new_dynamic_label();
    let done = asm.new_dynamic_label();

    // Check B == 0 → result = 0.
    dynasm!(asm
        ; .arch x64
        ; test Rq(b), Rq(b)
        ; jz =>div_zero
    );

    if signed {
        // For I64 signed: check MIN / -1 (would trap).
        if bits == 64 {
            let normal = asm.new_dynamic_label();
            dynasm!(asm
                ; .arch x64
                ; cmp Rq(b), -1
                ; jne =>normal
                ; mov Rq(c), QWORD i64::MIN
                ; cmp Rq(a), Rq(c)
                ; jne =>normal
                // MIN / -1: result = MIN.
                ; jmp =>done
                ; =>normal
            );
        }
        // cqo sign-extends rax into rdx.
        dynasm!(asm
            ; .arch x64
            ; cqo
            ; idiv Rq(b)
        );
    } else {
        dynasm!(asm
            ; .arch x64
            ; xor Rq(c), Rq(c)
            ; div Rq(b)
        );
    }
    // Quotient is in rax; move to rdx (slot C).
    dynasm!(asm
        ; .arch x64
        ; mov Rq(c), Rq(a)
        ; jmp =>done
        ; =>div_zero
        ; xor Rq(c), Rq(c)
        ; =>done
    );
}

/// Emit integer modulo (C-style remainder): A % B → C.
fn emit_int_mod(asm: &mut Assembler, signed: bool, bits: u8, _scratch: u8) {
    let a = INT_SLOT_A;
    let b = INT_SLOT_B;
    let c = INT_SLOT_C;

    let div_zero = asm.new_dynamic_label();
    let done = asm.new_dynamic_label();

    dynasm!(asm; .arch x64; test Rq(b), Rq(b); jz =>div_zero);

    if signed {
        if bits == 64 {
            let normal = asm.new_dynamic_label();
            dynasm!(asm
                ; .arch x64
                ; cmp Rq(b), -1
                ; jne =>normal
                // MIN % -1 = 0 (no remainder for MIN / -1).
                ; xor Rq(c), Rq(c)
                ; jmp =>done
                ; =>normal
            );
        }
        dynasm!(asm; .arch x64; cqo; idiv Rq(b));
    } else {
        dynasm!(asm; .arch x64; xor Rq(c), Rq(c); div Rq(b));
    }
    // Remainder is already in rdx (slot C). Done.
    dynasm!(asm
        ; .arch x64
        ; jmp =>done
        ; =>div_zero
        ; xor Rq(c), Rq(c)
        ; =>done
    );
}

// ─── Int unary ops ──────────────────────────────────────────────────

/// Emit an integer unary op.
///
/// Input in `INT_SLOT_A` (rax). Result in `INT_SLOT_C` (rdx).
/// The caller applies wrapping after this returns.
pub fn emit_unop_int(asm: &mut Assembler, op: ScalarUnaryOp, signed: bool) -> Result<(), String> {
    let a = INT_SLOT_A;
    let c = INT_SLOT_C;
    match op {
        ScalarUnaryOp::Neg => {
            dynasm!(asm; .arch x64; xor Rq(c), Rq(c); sub Rq(c), Rq(a));
        }
        ScalarUnaryOp::Abs => {
            if signed {
                // Branchless abs: negate, then cmov if the negation
                // went negative (i.e., original was positive).
                dynasm!(asm
                    ; .arch x64
                    ; mov Rq(c), Rq(a)
                    ; neg Rq(c)
                    ; cmovs Rq(c), Rq(a)
                );
            } else {
                // Unsigned abs is identity.
                dynasm!(asm; .arch x64; mov Rq(c), Rq(a));
            }
        }
        ScalarUnaryOp::Not => {
            // Logical not: truthy → 0, falsy → 1.
            dynasm!(asm
                ; .arch x64
                ; test Rq(a), Rq(a)
                ; sete Rb(c)
                ; movzx Rq(c), Rb(c)
            );
        }
        ScalarUnaryOp::BitwiseNot => {
            dynasm!(asm; .arch x64; mov Rq(c), Rq(a); not Rq(c));
        }
        ScalarUnaryOp::Sign => {
            if signed {
                let neg = asm.new_dynamic_label();
                let done = asm.new_dynamic_label();
                dynasm!(asm
                    ; .arch x64
                    ; xor Rq(c), Rq(c)
                    ; test Rq(a), Rq(a)
                    ; jz =>done
                    ; js =>neg
                    ; mov Rq(c), 1
                    ; jmp =>done
                    ; =>neg
                    ; mov Rq(c), -1
                    ; =>done
                );
            } else {
                // Unsigned sign: 0 if zero, 1 if nonzero.
                dynasm!(asm
                    ; .arch x64
                    ; test Rq(a), Rq(a)
                    ; setne Rb(c)
                    ; movzx Rq(c), Rb(c)
                );
            }
        }
        ScalarUnaryOp::IsNan => {
            // Integers are never NaN → always 0.
            dynasm!(asm; .arch x64; xor Rq(c), Rq(c));
        }
        ScalarUnaryOp::IsInf { .. } => {
            // Integers are never infinite → always 0.
            dynasm!(asm; .arch x64; xor Rq(c), Rq(c));
        }
        // Float-only ops: not valid for int compute repr.
        ScalarUnaryOp::Sqrt
        | ScalarUnaryOp::Reciprocal
        | ScalarUnaryOp::Floor
        | ScalarUnaryOp::Ceil
        | ScalarUnaryOp::Round
        | ScalarUnaryOp::Exp
        | ScalarUnaryOp::Ln
        | ScalarUnaryOp::Tanh
        | ScalarUnaryOp::Erf
        | ScalarUnaryOp::Sin
        | ScalarUnaryOp::Cos
        | ScalarUnaryOp::Tan
        | ScalarUnaryOp::Asin
        | ScalarUnaryOp::Acos
        | ScalarUnaryOp::Atan
        | ScalarUnaryOp::Sinh
        | ScalarUnaryOp::Cosh
        | ScalarUnaryOp::Asinh
        | ScalarUnaryOp::Acosh
        | ScalarUnaryOp::Atanh
        | ScalarUnaryOp::Log1p => {
            return Err(format!(
                "emit_unop_int: float-only op {op:?} not valid for int compute repr"
            ));
        }
    }
    Ok(())
}

// ─── Pow via trampoline ─────────────────────────────────────────────

/// Emit integer Pow via a call to a Rust trampoline function.
///
/// Pow uses **saturating** semantics (not wrapping) per dtype contract
/// §5.5: the mathematical result is computed in i128/u128 and clamped
/// to the target type's range. Negative exponents return 0 (signed).
///
/// The trampoline takes (a: i64, b: i64, bits: u32) and returns i64.
/// System V ABI: rdi=a, rsi=b, rdx=bits. Return in rax.
fn emit_int_pow(asm: &mut Assembler, signed: bool, bits: u8) {
    let fn_ptr = if signed {
        jit_signed_pow as *const ()
    } else {
        jit_unsigned_pow as *const ()
    };
    dynasm!(asm
        ; .arch x64
        ; mov rdi, Rq(INT_SLOT_A)        // a → first arg
        ; mov rsi, Rq(INT_SLOT_B)        // b → second arg
        ; mov edx, bits as i32            // bits → third arg
        ; mov rax, QWORD fn_ptr as i64
        ; call rax
        ; mov Rq(INT_SLOT_C), rax         // result → slot C
    );
}

/// Signed integer power: saturating, negative exponent → 0.
unsafe extern "C" fn jit_signed_pow(a: i64, b: i64, bits: u32) -> i64 {
    if b < 0 {
        return 0;
    }
    if b == 0 {
        return 1;
    }
    let exp = (b as u64).min(u32::MAX as u64) as u32;
    let mut result: i128 = 1;
    let mut base: i128 = a as i128;
    let mut e = exp;
    loop {
        if e & 1 == 1 {
            result = result.saturating_mul(base);
        }
        e >>= 1;
        if e == 0 {
            break;
        }
        base = base.saturating_mul(base);
    }
    // Clamp to signed range.
    let max = if bits >= 64 {
        i64::MAX as i128
    } else {
        (1i128 << (bits - 1)) - 1
    };
    let min = if bits >= 64 {
        i64::MIN as i128
    } else {
        -(1i128 << (bits - 1))
    };
    result.clamp(min, max) as i64
}

/// Unsigned integer power: saturating, no negative exponent case.
unsafe extern "C" fn jit_unsigned_pow(a: i64, b: i64, bits: u32) -> i64 {
    let a = a as u64;
    let b = b as u64;
    if b == 0 {
        return 1;
    }
    let exp = b.min(u32::MAX as u64) as u32;
    let mut result: u128 = 1;
    let mut base: u128 = a as u128;
    let mut e = exp;
    loop {
        if e & 1 == 1 {
            result = result.saturating_mul(base);
        }
        e >>= 1;
        if e == 0 {
            break;
        }
        base = base.saturating_mul(base);
    }
    let max: u128 = if bits >= 64 {
        u64::MAX as u128
    } else {
        (1u128 << bits) - 1
    };
    result.min(max) as i64
}

/// Emit Euclidean modulo (IMod): result sign matches divisor.
fn emit_int_imod(asm: &mut Assembler, signed: bool, bits: u8, scratch: u8) {
    if !signed {
        // For unsigned, IMod == Mod.
        emit_int_mod(asm, false, bits, scratch);
        return;
    }
    // Signed: compute C-style remainder, then adjust if signs differ.
    emit_int_mod(asm, true, bits, scratch);

    let c = INT_SLOT_C;
    let b = INT_SLOT_B;
    let done = asm.new_dynamic_label();
    dynasm!(asm
        ; .arch x64
        // If remainder is zero, no adjustment.
        ; test Rq(c), Rq(c)
        ; jz =>done
        // Check if rem and b have different signs.
        ; mov Rq(scratch), Rq(c)
        ; xor Rq(scratch), Rq(b)
        ; test Rq(scratch), Rq(scratch)
        ; jns =>done       // same sign (high bit 0) → no adjustment
        ; add Rq(c), Rq(b) // adjust: rem += divisor
        ; =>done
    );
}
