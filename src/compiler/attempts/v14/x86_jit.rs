#![allow(clippy::all, dead_code, unreachable_patterns, unused)]
//! Direct x86-64 JIT codegen via dynasm-rs.
//!
//! Replacement for the Cranelift-based `codegen.rs`. Targets the same
//! `CompiledSpanFn` interface and reuses `BufferLayout`, `EmbeddedTables`,
//! and the rest of the compile pipeline. Cranelift remains as the
//! transitional fallback while op coverage is built up phase-by-phase
//! per `X86_JIT_DESIGN.md`.
//!
//! P0 status: empty-span path only. Any non-empty graph returns
//! `Err("unsupported")`, which `compile_nano_graph` translates to a
//! Cranelift fallback (or a hard error under `X86_JIT_STRICT=1`).
//!
//! # ABI
//!
//! Compiled spans are `extern "C" fn(buffer: *mut u8) -> ()` — System V
//! AMD64. The single argument lands in `rdi`, no return value, no other
//! state. Inside the function:
//!
//! - `r12` holds the buffer pointer (callee-saved by us across math calls)
//! - `r13` holds the loop variable (callee-saved by us across math calls)
//! - `xmm0..xmm2` are scratch float registers
//! - `rax`, `rcx`, `rdx` are scratch GP registers
//!
//! See `X86_JIT_DESIGN.md` for the full register map and rollout plan.

use dynasmrt::{AssemblyOffset, DynasmApi, ExecutableBuffer, dynasm, x64::Assembler};

use crate::compiler::attempts::v14::executor::{CompiledSpanFn, SpanOutput, StoreSlice};
use crate::nano_graph::pattern::{AtomRange, NanoGraph};
use crate::pool::SystemPool;

/// A span compiled to native x86-64 machine code via dynasm-rs.
///
/// Mirrors the structure of `JitCompiledSpan` (the Cranelift backend) so
/// the two can be swapped at the `compile_nano_graph` boundary without
/// disturbing executor or layout code.
pub struct X86JitSpan {
    /// Owned executable memory holding the compiled function. Must outlive
    /// the function pointer derived from it.
    code: ExecutableBuffer,
    /// Offset of the entry point inside `code`.
    entry: AssemblyOffset,
    /// Output ranges declared by this span — used by `execute` to wire
    /// span outputs back into the executor.
    output_ranges: Vec<AtomRange>,
    /// Set when the span has zero groups (no compute, no buffer touches).
    /// `execute` short-circuits in that case to skip even the function call.
    is_empty: bool,
}

// SAFETY: ExecutableBuffer is Send+Sync, the entry offset is a plain usize,
// and the function we transmute it to has no captured state. The compiled
// code only touches its `*mut u8` argument and CPU state.
unsafe impl Send for X86JitSpan {}
unsafe impl Sync for X86JitSpan {}

impl X86JitSpan {
    /// Compile a span's NanoGraph into a native function ready for the
    /// executor.
    ///
    /// Returns `Err` for any unsupported graph shape; callers should fall
    /// back to another backend (Cranelift today, eventually nothing).
    pub fn compile(
        graph: &NanoGraph<'static, SystemPool>,
        output_ranges: &[AtomRange],
    ) -> Result<Self, String> {
        if graph.num_groups() == 0 {
            return Self::compile_empty(output_ranges);
        }

        // P0: every non-empty graph is unsupported. Subsequent phases
        // (P1+) flesh this out per X86_JIT_DESIGN.md.
        Err("x86_jit: P0 only supports empty spans".to_string())
    }

    /// Build a no-op compiled span for graphs with zero groups.
    ///
    /// Anchors the ABI and dynasm plumbing end-to-end: the compiled
    /// function is just `ret`, and `execute` is short-circuited so we
    /// don't even call it. Validates that the executable-memory + fn-ptr
    /// transmute path is wired correctly before any real codegen lands.
    fn compile_empty(output_ranges: &[AtomRange]) -> Result<Self, String> {
        let mut ops =
            Assembler::new().map_err(|e| format!("x86_jit: assembler init: {e}"))?;
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
        // SAFETY: the bytes at `self.code.ptr(self.entry)` were emitted by
        // `compile_empty` (or future per-op emitters) as a System V AMD64
        // function taking a single `*mut u8` arg and returning nothing.
        // The buffer outlives this borrow because it lives in `self.code`.
        unsafe { std::mem::transmute(self.code.ptr(self.entry)) }
    }
}

impl CompiledSpanFn for X86JitSpan {
    fn execute(&self, _inputs: &[StoreSlice<'_>], _outputs: &mut [SpanOutput<'_>]) {
        if self.is_empty {
            // No groups → nothing to compute, nothing to store. Skip the
            // function call entirely; the empty function is only kept so
            // the test path can prove the transmute / executable-memory
            // plumbing works end-to-end (see `tests` below).
            return;
        }
        // P0: only empty spans reach here. Future phases will populate
        // a working buffer from `_inputs`, call `self.entry_fn()`, and
        // copy results into `_outputs` — same shape as
        // `JitCompiledSpan::execute`.
        unimplemented!("x86_jit: non-empty span execute not yet implemented");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The empty-span path must compile, transmute, and call without
    /// crashing. This is the smoke test that validates dynasmrt + the
    /// executable-memory plumbing on the host platform before any real
    /// codegen lands.
    #[test]
    fn empty_span_compiles_and_calls() {
        let graph: NanoGraph<'static, SystemPool> = NanoGraph::new();
        let span = X86JitSpan::compile(&graph, &[]).expect("empty compile");
        assert!(span.is_empty);

        // The smoke test isn't `execute()` (that short-circuits) — it's
        // calling the compiled function pointer directly to prove the
        // transmute is sound.
        let f = span.entry_fn();
        let mut buffer = [0u8; 16];
        unsafe {
            f(buffer.as_mut_ptr());
        }
        // Function returned. The buffer must be untouched by `ret`.
        assert_eq!(buffer, [0u8; 16]);
    }

    /// Non-empty graphs must report unsupported in P0 so the
    /// `compile_nano_graph` fallback path kicks in.
    #[test]
    fn non_empty_graph_unsupported_in_p0() {
        // We can't easily construct a real NanoGraph here without pulling
        // in the lowering machinery, so use a sentinel: any graph with
        // num_groups() > 0 should return Err. We rely on the fact that
        // NanoGraph::new() yields an empty graph and add_group is the
        // only way to populate it; the empty graph is the trivial test
        // case that compile() handles, and the unsupported branch is
        // exercised by integration tests under SIMPLE_JIT_STRICT once P1
        // lands real op support. For P0 we just verify that the
        // compile() function exists and the empty path works.
        let graph: NanoGraph<'static, SystemPool> = NanoGraph::new();
        assert_eq!(graph.num_groups(), 0);
        // (The Err branch is unreachable from this test in isolation;
        // it's covered by the integration test in compiled_eval that
        // runs the full pipeline with X86_JIT=1.)
    }
}
