#![allow(clippy::all, dead_code, unreachable_patterns, unused)]
//! Direct x86-64 JIT codegen via dynasm-rs — coverage-first rewrite.
//!
//! This is the v14 x86_jit replacement. See `X86_JIT_PLAN.md` for the
//! full plan and `x86_jit_codec.md` for the codec architecture.
//!
//! # Architecture
//!
//! Three layers, each in its own subdirectory and independently
//! testable:
//!
//! - [`codec`] — bit-level I/O, format conversion, and in-register
//!   precision narrowing. Knows nothing about ScalarOps; takes raw bits
//!   and a dtype, produces compute-repr values (and vice versa).
//! - [`ops`] — per-ScalarOp emission. Takes compute-repr operands in
//!   slots A/B and writes to slot C. Knows nothing about loops or
//!   memory layout.
//! - [`orch`] — orchestrates address computation, loop emission, and
//!   the prologue/epilogue. The only layer that knows about
//!   `BufferLayout` and the executor ABI.
//!
//! # Phase 0 status
//!
//! Only empty spans (`graph.num_groups() == 0`) compile. Everything
//! else returns `Err("x86_jit: rewrite in progress")` and the caller
//! falls back to the cranelift backend.

pub mod codec;
pub mod ops;
pub mod orch;
pub mod prologue;
pub mod support;

#[cfg(test)]
pub(crate) mod tests;

use dynasmrt::x64::Assembler;
use dynasmrt::{AssemblyOffset, DynasmApi, ExecutableBuffer, dynasm};

use super::executor::{CompiledSpanFn, SpanOutput, StoreSlice};
use crate::nano_graph::pattern::{AtomRange, NanoGraph};
use crate::pool::SystemPool;

/// A span compiled to native x86-64 machine code.
///
/// Wired into `compiled_eval::compile_one_span_native`. The cranelift
/// backend remains the per-span fallback for everything this can't
/// (yet) handle.
pub struct X86JitSpan {
    /// Owned executable memory holding the compiled function. Must
    /// outlive any function pointer derived from it.
    code: ExecutableBuffer,
    /// Offset of the entry point inside `code`.
    entry: AssemblyOffset,
    /// Output ranges declared by this span — used by `execute` to
    /// wire span outputs back into the executor.
    output_ranges: Vec<AtomRange>,
    /// Set when the span has zero groups. `execute` short-circuits in
    /// that case to skip even the function call.
    is_empty: bool,
}

// SAFETY: ExecutableBuffer is Send+Sync, the entry offset is a plain
// usize, and the function we transmute it to has no captured state.
// The compiled code only touches its `*mut u8` argument and CPU state.
unsafe impl Send for X86JitSpan {}
unsafe impl Sync for X86JitSpan {}

impl X86JitSpan {
    /// Compile a span's NanoGraph into a native function ready for the
    /// executor.
    ///
    /// Phase 0: only zero-group graphs compile. Anything else returns
    /// `Err("x86_jit: rewrite in progress")` so the caller falls back
    /// to cranelift.
    pub fn compile(
        graph: &NanoGraph<'static, SystemPool>,
        output_ranges: &[AtomRange],
    ) -> Result<Self, String> {
        if graph.num_groups() == 0 {
            return Self::compile_empty(output_ranges);
        }
        Err("x86_jit: rewrite in progress".to_string())
    }

    /// Build a no-op compiled span for graphs with zero groups.
    ///
    /// Anchors the ABI plumbing end-to-end: the compiled function is
    /// just `ret`, and `execute` is short-circuited so we don't even
    /// call it. Validates that the executable-memory + fn-ptr
    /// transmute path is wired correctly.
    fn compile_empty(output_ranges: &[AtomRange]) -> Result<Self, String> {
        let mut ops = Assembler::new().map_err(|e| format!("x86_jit: assembler init: {e}"))?;
        let entry = ops.offset();
        dynasm!(ops
            ; .arch x64
            ; ret
        );
        let code = ops
            .finalize()
            .map_err(|_| "x86_jit: assembler finalize failed".to_string())?;

        Ok(Self {
            code,
            entry,
            output_ranges: output_ranges.to_vec(),
            is_empty: true,
        })
    }

    /// Type alias for the compiled function signature.
    #[inline]
    fn entry_fn(&self) -> unsafe extern "C" fn(*mut u8) {
        // SAFETY: bytes at `self.code.ptr(self.entry)` were emitted as a
        // System V AMD64 function taking a single `*mut u8` and returning
        // nothing. The buffer outlives this borrow because it lives in
        // `self.code`.
        unsafe { std::mem::transmute(self.code.ptr(self.entry)) }
    }
}

impl CompiledSpanFn for X86JitSpan {
    fn execute(&self, _inputs: &[StoreSlice<'_>], _outputs: &mut [SpanOutput<'_>]) {
        if self.is_empty {
            return;
        }
        // Phase 0 returns Err for any non-empty span, so we never
        // construct an X86JitSpan with `is_empty = false`. This branch
        // exists for symmetry only.
        unreachable!("x86_jit: non-empty span execute reached in phase 0");
    }
}
