//! JIT-wrapping helper for codec unit tests.
//!
//! Each codec primitive is an `emit_*` function that takes `&mut
//! Assembler` and emits assembly with no prologue/epilogue and a fixed
//! register convention. To unit test those primitives in isolation
//! (before phases 2.B / 3 / 4 wire them into a real compiled span),
//! we wrap them in a tiny `extern "C"` function with a
//! test-harness ABI.
//!
//! [`JitFn`] owns the executable buffer and yields a function pointer.
//! The buffer must outlive any function call, so the typical pattern
//! is to keep the `JitFn` in a local binding for the duration of the
//! test.

use dynasmrt::x64::Assembler;
use dynasmrt::{AssemblyOffset, DynasmApi, ExecutableBuffer, dynasm};

/// A JITed function held alive by its owning `ExecutableBuffer`. The
/// `ptr` accessor returns a raw pointer that is valid for the lifetime
/// of the [`JitFn`] borrow.
pub struct JitFn {
    code: ExecutableBuffer,
    entry: AssemblyOffset,
}

impl JitFn {
    /// Build a JitFn whose body is whatever the closure emits, framed
    /// by a single trailing `ret`. The closure may emit any
    /// caller-saved scratch use; callee-saved registers (`rbx`, `rbp`,
    /// `r12`–`r15`) must be preserved by the body if it touches them.
    pub fn build<F>(emit: F) -> Self
    where
        F: FnOnce(&mut Assembler),
    {
        let mut asm = Assembler::new().expect("assembler init");
        let entry = asm.offset();
        emit(&mut asm);
        dynasm!(asm
            ; .arch x64
            ; ret
        );
        let code = asm.finalize().expect("assembler finalize");
        Self { code, entry }
    }

    /// Raw entry pointer. Caller transmutes to the appropriate
    /// `extern "C"` signature and invokes.
    pub fn ptr(&self) -> *const u8 {
        self.code.ptr(self.entry)
    }
}
