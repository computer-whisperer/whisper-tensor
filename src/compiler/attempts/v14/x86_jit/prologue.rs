//! Function prologue, epilogue, and register allocation map.
//!
//! ABI: `extern "C" fn(buffer_ptrs: *const *mut u8) -> ()`, System V
//! AMD64. `rdi` holds a pointer to an array of buffer base pointers;
//! `buffer_ptrs[buffer_id]` is the base of the buffer the span's
//! `buffer_id` addresses. The compiled function saves/restores all
//! callee-saved registers it clobbers.
//!
//! # Buffer base layout
//!
//! The prologue dedicates one callee-saved register — `r14`
//! ([`BUFFER_PTRS_REG`]) — to permanently hold the `buffer_ptrs` array
//! pointer for the lifetime of the call. After entry, `r14 = rdi`.
//! Because `r14` is callee-saved, the array pointer survives every
//! libm trampoline the body might issue; `rdi` itself is free to be
//! clobbered as a general-purpose scratch register thereafter.
//!
//! Each span's [`BufferBases`](super::layout::BufferBases) table names
//! the distinct `buffer_id`s the body touches. The first
//! [`MAX_BUFFER_BASES`](super::layout::MAX_BUFFER_BASES) of them get
//! persistent callee-saved GPRs from
//! [`BUFFER_BASE_REG_POOL`](super::layout::BUFFER_BASE_REG_POOL) —
//! the prologue loads each via `mov <reg>, QWORD [r14 + id * 8]` —
//! and the rest land in the **overflow path**: the address layer's
//! [`materialize_buffer_base`] helper emits a `mov scratch,
//! QWORD [r14 + id * 8]` on demand at each access. One extra mov
//! per access for overflow buffers; the buffer_ptrs array is a few
//! words and typically hot in L1, so the cost is negligible.
//!
//! The epilogue restores every pushed register in reverse order.
//!
//! # Register layout
//!
//! - **Float compute repr** (XMM): A = `xmm0`, B = `xmm1`, C = `xmm2`.
//! - **Int compute repr** (GP, 64-bit): A = `rax`, B = `rcx`, C = `rdx`.
//! - **Loop variable**: `r13` (callee-saved).
//! - **Loop end**: embedded as an `i32` immediate in the `cmp`
//!   instruction at loop header emit time. No dedicated register. If
//!   a span's loop count exceeds `i32::MAX`, compile fails and the
//!   caller falls back to pool_eval.
//! - **buffer_ptrs array pointer**: `r14` (`BUFFER_PTRS_REG`,
//!   callee-saved). Loaded once from `rdi` at prologue entry.
//! - **Fast-pool buffer bases**: one of `r12/r15/rbx/rbp` per distinct
//!   in-pool buffer (up to 4 per span).
//! - **Overflow buffer bases**: fetched via `[r14 + id*8]` into a
//!   caller-chosen scratch register at each access site.
//!
//! # Codec scratch budget
//!
//! The bit-I/O / format / precision primitives in [`super::codec`]
//! use a small set of caller-saved scratch registers. The codec
//! functions never touch the slot register conventions (A/B/C); the
//! orch layer chooses which slot the codec output lands in by
//! passing register parameters to the codec emit functions.
//!
//! - `rcx` is reserved as the shift-count register (`cl` is the only
//!   register x86's variable shifts can use). Slot B (Int) also lives
//!   in `rcx`; orch arranges that codec calls do not collide with a
//!   live slot-B value.
//! - `rdi`, `rsi`, `r8`, `r9`, `r10`, `r11` are caller-saved general
//!   scratch — codec primitives may use these freely without saving.
//!
//! # Stack frame
//!
//! At function entry System V guarantees `rsp % 16 == 8` (the CALL
//! pushed an 8-byte return address onto a 16-byte aligned stack). We
//! push an even number of 8-byte values so post-prologue `rsp` is
//! 16-byte aligned for any libm trampoline the body issues. r13 and
//! r14 contribute two pushes; each buffer base register contributes
//! one more. If `(2 + num_buf_pushes)` is odd we pad with
//! `sub rsp, 8`.

#![allow(dead_code)]

use dynasmrt::x64::Assembler;
use dynasmrt::{DynasmApi, dynasm};

use super::super::layout::BufferBases;

/// Register codes for the System V x64 ABI argument registers.
pub const SYSV_ARG0: u8 = 7; // rdi
pub const SYSV_ARG1: u8 = 6; // rsi
pub const SYSV_ARG2: u8 = 2; // rdx
pub const SYSV_ARG3: u8 = 1; // rcx
pub const SYSV_ARG4: u8 = 8; // r8
pub const SYSV_ARG5: u8 = 9; // r9

/// Slot A in the Int compute repr — `rax` (also the SysV int return).
pub const INT_SLOT_A: u8 = 0;
/// Slot B in the Int compute repr — `rcx`.
pub const INT_SLOT_B: u8 = 1;
/// Slot C in the Int compute repr — `rdx`.
pub const INT_SLOT_C: u8 = 2;

/// Slot A in the Float compute repr — `xmm0` (also the SysV float return).
pub const FLT_SLOT_A: u8 = 0;
/// Slot B in the Float compute repr — `xmm1`.
pub const FLT_SLOT_B: u8 = 1;
/// Slot C in the Float compute repr — `xmm2`.
pub const FLT_SLOT_C: u8 = 2;

/// The shift-count register code (`rcx`).
pub const RCX_REG: u8 = 1;

/// Loop induction variable register code (`r13`).
pub const LOOP_VAR_REG: u8 = 13;

/// Register dedicated to holding the `buffer_ptrs` array pointer for
/// the lifetime of the compiled call. Loaded from `rdi` at the top
/// of [`emit_prologue`]; callee-saved so it survives every libm
/// trampoline the body might issue. The overflow path of
/// [`super::orch::group::materialize_buffer_base`] addresses buffer
/// bases via `[r14 + buffer_id*8]`.
pub const BUFFER_PTRS_REG: u8 = 14;

/// Emit the function prologue.
///
/// Saves the loop induction register (`r13`), the `buffer_ptrs`
/// register (`r14`), and every callee-saved buffer-base register the
/// span's body will use. Then copies `rdi` (the `buffer_ptrs` arg)
/// into `r14` and loads each fast-pool buffer base from
/// `[r14 + id*8]`. After this returns the stack is 16-byte aligned.
pub fn emit_prologue(asm: &mut Assembler, bases: &BufferBases) {
    let buffer_loads = bases.loads();
    let num_buf_pushes = buffer_loads.len();

    // Push the registers we'll clobber. r13 holds the loop induction
    // var; r14 holds the buffer_ptrs array pointer.
    dynasm!(asm
        ; .arch x64
        ; push r13
        ; push r14
    );
    // Push each fast-pool buffer base register in the order the
    // table assigns. The epilogue pops them in reverse.
    for &(_buf_id, reg_code) in buffer_loads {
        dynasm!(asm
            ; .arch x64
            ; push Rq(reg_code)
        );
    }
    // Pad to 16-byte alignment. Entry rsp ≡ 8 (mod 16); each push
    // toggles by 8. After `2 + num_buf_pushes` pushes we need an
    // even total count for rsp to return to 0 (mod 16).
    let frame_misaligned = ((2 + num_buf_pushes) % 2) != 0;
    if frame_misaligned {
        dynasm!(asm
            ; .arch x64
            ; sub rsp, 8
        );
    }
    // Copy the buffer_ptrs arg from rdi into r14 (BUFFER_PTRS_REG).
    // After this line, rdi is free as a general-purpose scratch —
    // any access to a buffer base (fast-pool or overflow) goes
    // through r14.
    dynasm!(asm
        ; .arch x64
        ; mov r14, rdi
    );
    // Load each fast-pool buffer base from the array via r14.
    for &(buf_id, reg_code) in buffer_loads {
        dynasm!(asm
            ; .arch x64
            ; mov Rq(reg_code), QWORD [r14 + (buf_id as i32) * 8]
        );
    }
}

/// Emit the function epilogue.
///
/// Restores the callee-saved registers the prologue pushed and
/// returns. Must be paired one-for-one with [`emit_prologue`].
pub fn emit_epilogue(asm: &mut Assembler, bases: &BufferBases) {
    let buffer_loads = bases.loads();
    let num_buf_pushes = buffer_loads.len();

    let frame_misaligned = ((2 + num_buf_pushes) % 2) != 0;
    if frame_misaligned {
        dynasm!(asm
            ; .arch x64
            ; add rsp, 8
        );
    }
    // Pop fast-pool buffer base registers in reverse.
    for &(_buf_id, reg_code) in buffer_loads.iter().rev() {
        dynasm!(asm
            ; .arch x64
            ; pop Rq(reg_code)
        );
    }
    dynasm!(asm
        ; .arch x64
        ; pop r14
        ; pop r13
        ; ret
    );
}
