//! Function prologue, epilogue, and register allocation map.
//!
//! ABI: `extern "C" fn(buffer: *mut u8) -> ()`, System V AMD64. The
//! `buffer` argument arrives in `rdi`. The compiled function is
//! responsible for saving/restoring all callee-saved registers it
//! clobbers (per System V).
//!
//! # Register layout
//!
//! Slot register assignment (compute slots A / B / C from the codec
//! design):
//!
//! - **Float compute repr** (XMM): A = `xmm0`, B = `xmm1`, C = `xmm2`.
//! - **Int compute repr** (GP, 64-bit, sign/zero-extended): A = `rax`,
//!   B = `rcx`, C = `rdx`.
//! - **Buffer pointer**: `r12` (callee-saved across libm trampolines).
//! - **Loop variable**: `r13` (callee-saved).
//! - **Loop end**: `r14` (callee-saved).
//!
//! # Codec scratch budget
//!
//! The bit-I/O / format / precision primitives in [`super::codec`] use a
//! small set of caller-saved scratch registers. The codec functions never
//! touch the slot register conventions above (A/B/C); the orch layer
//! allocates which slot the codec output lands in by passing register
//! parameters to the codec emit functions.
//!
//! - `rcx` is reserved as the shift-count register (`cl` is the only
//!   register x86's variable shifts can use). Codec primitives clobber
//!   `rcx` whenever they need a runtime shift. Slot B (Int) lives in
//!   `rcx` by convention; orch must arrange that codec calls do not
//!   collide with a live slot-B value.
//! - `rdi`, `rsi`, `r8`, `r9`, `r10`, `r11` are caller-saved general
//!   scratch — codec primitives may use these freely without saving.
//!
//! # Stack frame
//!
//! At function entry, System V guarantees `rsp % 16 == 8` (the call
//! pushed an 8-byte return address onto a 16-byte aligned stack). The
//! prologue pushes three callee-saved registers (`r12`, `r13`, `r14`),
//! growing the stack by 24 bytes. Combined with the return address,
//! that leaves `rsp` at `rsp_entry - 24`, i.e. 16-byte aligned again —
//! ready for any libm-style call the body may issue.

#![allow(dead_code)]

use dynasmrt::x64::Assembler;
use dynasmrt::{DynasmApi, dynasm};

/// Register codes for the System V x64 ABI argument registers.
///
/// dynasm-rs takes the numeric register code (the same value as the
/// hardware encoding) when using `Rq(n)` / `Rd(n)` / `Rb(n)` /
/// `Rx(n)` dynamic register references. These constants name the
/// codes so call sites read like `Rq(SYSV_ARG0)` instead of `Rq(7)`.
pub const SYSV_ARG0: u8 = 7; // rdi
pub const SYSV_ARG1: u8 = 6; // rsi
pub const SYSV_ARG2: u8 = 2; // rdx
pub const SYSV_ARG3: u8 = 1; // rcx
pub const SYSV_ARG4: u8 = 8; // r8
pub const SYSV_ARG5: u8 = 9; // r9

/// Slot A in the Int compute repr — `rax`. Per System V, `rax` is also
/// the integer return register.
pub const INT_SLOT_A: u8 = 0;
/// Slot B in the Int compute repr — `rcx`.
pub const INT_SLOT_B: u8 = 1;
/// Slot C in the Int compute repr — `rdx`.
pub const INT_SLOT_C: u8 = 2;

/// Slot A in the Float compute repr — `xmm0`. Per System V, `xmm0` is
/// also the float return register.
pub const FLT_SLOT_A: u8 = 0;
/// Slot B in the Float compute repr — `xmm1`.
pub const FLT_SLOT_B: u8 = 1;
/// Slot C in the Float compute repr — `xmm2`.
pub const FLT_SLOT_C: u8 = 2;

/// The shift-count register code (`rcx`). x86's variable shifts (`shl`,
/// `shr`, `shrd`, `shld`, ...) require the count in `cl`. The codec
/// reserves `rcx` for this purpose; callers that need a live value in
/// `rcx` across a codec call must spill it.
pub const RCX_REG: u8 = 1;

/// Buffer pointer register code (`r12`). The prologue copies the
/// `*mut u8` buffer argument from `rdi` here so it survives any libm
/// trampoline (`r12`–`r15` are callee-saved under System V).
pub const BUFFER_REG: u8 = 12;

/// Loop induction variable register code (`r13`). Used by `emit_group`
/// when emitting `count > 1` loops.
pub const LOOP_VAR_REG: u8 = 13;

/// Loop end register code (`r14`). Holds the exclusive upper bound the
/// loop checks against `r13`.
pub const LOOP_END_REG: u8 = 14;

/// Emit the function prologue.
///
/// Saves the callee-saved scratch registers we use (`r12`, `r13`,
/// `r14`) and copies the buffer pointer from the System V argument
/// register (`rdi`) into `r12`. After this returns, the stack is
/// 16-byte aligned (see the §Stack frame doc-comment above) and any
/// emit_group code can call into libm without further alignment work.
pub fn emit_prologue(asm: &mut Assembler) {
    dynasm!(asm
        ; .arch x64
        ; push r12
        ; push r13
        ; push r14
        ; mov r12, rdi
    );
}

/// Emit the function epilogue.
///
/// Restores the callee-saved registers the prologue pushed and
/// returns. Must be paired one-for-one with [`emit_prologue`].
pub fn emit_epilogue(asm: &mut Assembler) {
    dynasm!(asm
        ; .arch x64
        ; pop r14
        ; pop r13
        ; pop r12
        ; ret
    );
}
