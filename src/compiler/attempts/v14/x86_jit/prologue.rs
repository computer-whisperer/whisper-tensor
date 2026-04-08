//! Function prologue, epilogue, and register allocation map.
//!
//! ABI: `extern "C" fn(buffer: *mut u8) -> ()`, System V AMD64. The
//! `buffer` argument arrives in `rdi`. The compiled function is
//! responsible for saving/restoring all callee-saved registers it
//! clobbers (per System V).
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
//! Phase 0 stub. The actual prologue/epilogue emission lands in
//! phase 2 once the first non-empty group needs to be compiled.
