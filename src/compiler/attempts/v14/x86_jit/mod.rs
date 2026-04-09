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
//! # Phase 2.B.5 status
//!
//! The pipeline emits Identity, same-repr Cast, Literal, and
//! LiteralSpan with **all InputRef shapes**. Cross-compute-repr
//! Cast (float↔int) falls back to cranelift. Everything else is
//! rejected by [`support::check_supported`].

pub mod codec;
pub mod ops;
pub mod orch;
pub mod prologue;
pub mod support;

#[cfg(test)]
pub(crate) mod tests;

use dynasmrt::x64::Assembler;
use dynasmrt::{AssemblyOffset, DynasmApi, ExecutableBuffer};

use super::executor::{CompiledSpanFn, SpanOutput, StoreSlice};
use super::layout::{
    BufferLayout, compute_layout, read_buffer_to_output, write_store_slice_to_buffer,
};
use crate::nano_graph::pattern::{AtomRange, NanoGraph};
use crate::pool::SystemPool;

use codec::format::CodecTables;
use orch::address::AddressTables;
use orch::group::emit_group;

/// Tail padding (in bytes) added to the working buffer beyond
/// `BufferLayout::total_bytes`. The bit_io codec primitives may read
/// up to 16 bytes starting at any byte offset they touch, so the
/// last few atoms in the buffer can spill past `total_bytes` by up
/// to 8 bytes. We zero-pad those bytes once in the literal template
/// and the read-modify-write store path preserves them as zero.
const CODEC_TAIL_SLACK: usize = 8;

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
    /// Buffer layout describing where each atom lives. Empty for
    /// zero-group graphs.
    layout: BufferLayout,
    /// Pre-populated working buffer template — literals and lookup
    /// tables already baked in. `execute` clones this, marshals
    /// inputs over it, runs the JIT, then reads outputs back.
    literal_template: Vec<u8>,
    /// Output ranges declared by this span — used by `execute` to
    /// wire span outputs back into the executor.
    output_ranges: Vec<AtomRange>,
    /// Codec lookup tables held alive for the lifetime of the JIT.
    /// The compiled code embeds raw pointers into these slabs, so
    /// they must not be dropped or relocated until `code` is.
    _tables: CodecTables,
    /// Address lookup tables for multi-entry Explicit InputRefs.
    /// Same lifetime contract as `_tables`.
    _addr_tables: AddressTables,
}

// SAFETY: ExecutableBuffer is Send+Sync, the entry offset is a plain
// usize, and the function we transmute it to has no captured state
// beyond the buffer pointer it receives as `rdi`. The compiled code
// reads/writes only the buffer it's handed plus its callee-saved
// stack slots, and the embedded codec tables (which are pinned by
// _tables for the JIT's lifetime). All fields are themselves Send/Sync.
unsafe impl Send for X86JitSpan {}
unsafe impl Sync for X86JitSpan {}

impl X86JitSpan {
    /// Compile a span's NanoGraph into a native function ready for the
    /// executor.
    ///
    /// Phase 2.B.4: zero-group graphs and Identity-only graphs with
    /// any InputRef shape. Anything else is rejected by
    /// [`support::check_supported`] and the caller falls back to
    /// cranelift.
    pub fn compile(
        graph: &NanoGraph<'static, SystemPool>,
        output_ranges: &[AtomRange],
    ) -> Result<Self, String> {
        // Reject anything we can't (yet) handle. Caller falls back to
        // the cranelift backend for the Err case.
        support::check_supported(graph)?;

        // Layout is computed even for empty graphs — `compute_layout`
        // returns a zero-byte layout in that case, but going through
        // the same code path anchors the marshalling pipeline.
        let layout = compute_layout(graph, output_ranges);

        // Pre-populate the working-buffer template with literals.
        // For empty / Identity-only graphs there are no embedded
        // codec tables yet, so the template is just `total_bytes` of
        // literal-filled bytes plus the codec tail-slack.
        let template_bytes = layout.total_bytes + CODEC_TAIL_SLACK;
        let mut literal_template = vec![0u8; template_bytes];
        layout.populate_literals(graph, &mut literal_template);

        // Build the JIT. Empty graphs still go through the prologue
        // and epilogue so the function shape matches what later
        // phases will produce.
        let mut tables = CodecTables::new();
        let mut addr_tables = AddressTables::new();
        let mut asm = Assembler::new().map_err(|e| format!("x86_jit: assembler init: {e}"))?;
        let entry = asm.offset();
        prologue::emit_prologue(&mut asm);
        for group in graph.groups() {
            // Skip dead groups (use_count == 0). The layout still
            // allocates them, but their consumers have been deleted
            // so the JIT must not write into the (potentially
            // reused) slot. Mirrors the cranelift backend.
            let gi = graph
                .find_group_idx(group.base_id)
                .expect("group must be present");
            if layout.group_use_counts[gi] == 0 {
                continue;
            }
            emit_group(&mut asm, &layout, group, &mut addr_tables, &mut tables)?;
        }
        prologue::emit_epilogue(&mut asm);
        let code = asm
            .finalize()
            .map_err(|_| "x86_jit: assembler finalize failed".to_string())?;

        Ok(Self {
            code,
            entry,
            layout,
            literal_template,
            output_ranges: output_ranges.to_vec(),
            _tables: tables,
            _addr_tables: addr_tables,
        })
    }
}

impl CompiledSpanFn for X86JitSpan {
    fn execute(&self, inputs: &[StoreSlice<'_>], outputs: &mut [SpanOutput<'_>]) {
        // Empty layout → nothing to compute, nothing to marshal.
        // Skip the buffer alloc + JIT call entirely (matches what
        // `JitCompiledSpan::execute` does for the empty-graph case).
        if self.layout.total_bytes == 0 {
            return;
        }

        // Working buffer = literal template + marshalled inputs.
        let mut buffer = self.literal_template.clone();
        for slice in inputs {
            write_store_slice_to_buffer(slice, &self.layout, &mut buffer);
        }

        // Run the JIT.
        // SAFETY: bytes at `self.code.ptr(self.entry)` were emitted
        // as a System V AMD64 function taking a single `*mut u8` and
        // returning nothing. The buffer outlives this call.
        let func: unsafe extern "C" fn(*mut u8) =
            unsafe { std::mem::transmute(self.code.ptr(self.entry)) };
        unsafe { func(buffer.as_mut_ptr()) };

        // Marshal declared outputs back into SpanOutputs.
        for (range, out) in self.output_ranges.iter().zip(outputs.iter_mut()) {
            read_buffer_to_output(range, &self.layout, &buffer, out);
        }
    }
}
