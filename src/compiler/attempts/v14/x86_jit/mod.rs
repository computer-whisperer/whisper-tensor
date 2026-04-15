#![allow(clippy::all, dead_code, unreachable_patterns, unused)]
//! Direct x86-64 JIT codegen via dynasm-rs — coverage-first rewrite.
//!
//! # Architecture
//!
//! Three layers, each in its own subdirectory and independently
//! testable:
//!
//! - [`codec`] — bit-level I/O, format conversion, and in-register
//!   precision narrowing. Knows nothing about ScalarOps.
//! - [`ops`] — per-ScalarOp emission. Takes compute-repr operands in
//!   slots A/B and writes to slot C. Knows nothing about loops or
//!   memory layout.
//! - [`orch`] — orchestrates address computation, loop emission, and
//!   the prologue/epilogue. The only layer that knows about
//!   `BufferLayout` and the executor ABI.
//!
//! Under the memory-placement rewrite each compiled span reads and
//! writes through a `buffer_ptrs: *const *mut u8` argument array. The
//! prologue loads each distinct `buffer_id` it will touch into a
//! callee-saved GPR and every load/store indexes that register via
//! the slot's placer-assigned offset.

pub mod codec;
pub mod ops;
pub mod orch;
pub mod prologue;
pub mod support;

#[cfg(test)]
pub(crate) mod tests;

use dynasmrt::x64::Assembler;
use dynasmrt::{AssemblyOffset, DynasmApi, ExecutableBuffer};

use super::executor::CompiledSpanFn;
use super::layout::{BufferLayout, compute_layout};
use super::placer::AtomPlacementMap;
use crate::nano_graph::pattern::{AtomRange, NanoGraph};
use crate::pool::SystemPool;

use codec::format::CodecTables;
use orch::address::AddressTables;
use orch::group::emit_group;

/// A span compiled to native x86-64 machine code.
///
/// Thin wrapper: `code` is the JIT'd function, `layout` tells the
/// executor how much scratch the span needs, and the codec/address
/// tables are pinned until the code is dropped. Literals live in the
/// plan-wide literal buffer the executor owns — the JIT just emits
/// loads against `buffer_ptrs[literal_buf_id]` like any other buffer.
pub struct X86JitSpan {
    /// Owned executable memory holding the compiled function. Must
    /// outlive any function pointer derived from it.
    code: ExecutableBuffer,
    /// Offset of the entry point inside `code`.
    entry: AssemblyOffset,
    /// Layout metadata — `total_bytes` is the lane scratch size the
    /// executor must hand this span at dispatch. Also carries the
    /// per-buffer-id register assignment the prologue already loaded.
    layout: BufferLayout,
    /// Codec lookup tables held alive for the lifetime of the JIT.
    /// The compiled code embeds raw pointers into these slabs, so
    /// they must not be dropped or relocated until `code` is.
    _tables: CodecTables,
    /// Address lookup tables for multi-entry Explicit InputRefs.
    _addr_tables: AddressTables,
}

// SAFETY: ExecutableBuffer is Send+Sync, the entry offset is a plain
// usize, and the function we transmute it to has no captured state
// beyond the caller-supplied `buffer_ptrs` array. All other fields
// are themselves Send+Sync.
unsafe impl Send for X86JitSpan {}
unsafe impl Sync for X86JitSpan {}

impl X86JitSpan {
    /// Compile a span's NanoGraph into a native function.
    ///
    /// Reject unsupported ops up front so the caller falls back to
    /// pool_eval. Literal / LiteralSpan groups generate **no** code —
    /// their bytes live in the plan-wide literal buffer the executor
    /// owns, populated once at plan-build from the main graph. The
    /// JIT just emits loads against `buffer_ptrs[literal_buf_id]`
    /// when a consumer reads them.
    pub fn compile(
        graph: &NanoGraph<'static, SystemPool>,
        output_ranges: &[AtomRange],
        placement: &AtomPlacementMap,
    ) -> Result<Self, String> {
        support::check_supported(graph)?;

        let layout = compute_layout(graph, output_ranges, true, placement)?;

        let mut tables = CodecTables::new();
        let mut addr_tables = AddressTables::new();
        let mut asm = Assembler::new().map_err(|e| format!("x86_jit: assembler init: {e}"))?;
        let entry = asm.offset();
        prologue::emit_prologue(&mut asm, &layout.buffer_bases);
        for group in graph.groups() {
            // Skip dead groups (use_count == 0).
            let gi = graph
                .find_group_idx(group.base_id)
                .expect("group must be present");
            if layout.group_use_counts[gi] == 0 {
                continue;
            }
            // Skip inlinable groups — consumer's loop will re-emit
            // the producer's expression inline.
            if gi < layout.inlinable.len() && layout.inlinable[gi] {
                continue;
            }
            // `emit_group` handles `Literal`/`LiteralSpan` groups
            // itself: when their destination slot lives in the
            // literal buffer it's a no-op (source == destination,
            // bytes pre-populated at plan-build); when the
            // destination is an output buffer it emits an ordinary
            // copy from the literal buffer. No pre-dispatch skip
            // here.
            emit_group(
                &mut asm,
                &layout,
                graph,
                group,
                placement,
                &mut addr_tables,
                &mut tables,
            )
            .map_err(|e| {
                format!(
                    "group[{gi}] base={} op={:?} count={} offset={}: {e}",
                    group.base_id,
                    crate::compiler::attempts::v14::layout::op_name_short(&group.op),
                    group.count,
                    group.atom_offset
                )
            })?;
        }
        prologue::emit_epilogue(&mut asm, &layout.buffer_bases);
        let code = asm
            .finalize()
            .map_err(|_| "x86_jit: assembler finalize failed".to_string())?;

        Ok(Self {
            code,
            entry,
            layout,
            _tables: tables,
            _addr_tables: addr_tables,
        })
    }

    /// Expose the span's layout for tests and diagnostics.
    pub fn layout(&self) -> &BufferLayout {
        &self.layout
    }
}

impl CompiledSpanFn for X86JitSpan {
    fn scratch_bytes(&self) -> usize {
        // Codec bit_io primitives may read/write up to 16 bytes
        // starting at any byte position, so we pad the reported
        // scratch requirement by a constant tail slack.
        const CODEC_TAIL_SLACK: usize = 16;
        self.layout.total_bytes + CODEC_TAIL_SLACK
    }

    fn execute(
        &self,
        buffer_ptrs: &[*mut u8],
        _bindings: &std::collections::HashMap<crate::nano_graph::pattern::GraphConstantId, u64>,
    ) {
        // Zero-body spans: the prologue/epilogue is the entire
        // function and the argument array is unused.
        //
        // Sym-dim bindings are ignored here: the JIT path only runs
        // on sym-free spans (the partitioner routes sym-touching
        // spans to PoolEvalSpan). Phase 2 will emit dynamic loops
        // keyed off `_bindings`.
        let func: unsafe extern "C" fn(*const *mut u8) =
            unsafe { std::mem::transmute(self.code.ptr(self.entry)) };
        // SAFETY: bytes at `code.ptr(entry)` were emitted as a
        // System V AMD64 function taking a `*const *mut u8` — a
        // pointer to an array of buffer base pointers — and
        // returning nothing. Each pointer in `buffer_ptrs` is valid
        // for this call's duration per the executor's contract.
        unsafe { func(buffer_ptrs.as_ptr()) };
    }
}
