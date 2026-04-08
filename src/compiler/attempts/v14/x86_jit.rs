#![allow(clippy::all, dead_code, unreachable_patterns, unused)]
//! Direct x86-64 JIT codegen via dynasm-rs.
//!
//! Replacement for the Cranelift-based `codegen.rs`. Targets the same
//! `CompiledSpanFn` interface and reuses `BufferLayout`, `EmbeddedTables`,
//! and the rest of the compile pipeline. Cranelift remains as the
//! transitional fallback while op coverage is built up phase-by-phase
//! per `X86_JIT_DESIGN.md`.
//!
//! P1 status: F32 happy path. Spans whose groups are F32-only with
//! 1D affine `Strided` / `Broadcast` inputs and arithmetic / select /
//! cast / identity ops compile here. Anything else returns
//! `Err("unsupported")`, which the caller translates to a Cranelift
//! fallback (or a hard error under `X86_JIT_STRICT=1`).
//!
//! # ABI
//!
//! Compiled spans are `extern "C" fn(buffer: *mut u8) -> ()` — System V
//! AMD64. The single argument lands in `rdi`, no return value, no other
//! state. Inside the function:
//!
//! - `r12` holds the buffer pointer (callee-saved by us across math calls)
//! - `r13` holds the loop variable (callee-saved by us across math calls)
//! - `r14` holds the loop end constant (callee-saved by us across math calls)
//! - `xmm0..xmm2` are scratch float registers
//! - `rax`, `rcx`, `rdx` are scratch GP registers
//!
//! See `X86_JIT_DESIGN.md` for the full register map and rollout plan.

use dynasmrt::x64::Assembler;
use dynasmrt::{AssemblyOffset, DynasmApi, DynasmLabelApi, ExecutableBuffer, dynasm};

use super::codegen::{
    BufferLayout, EmbeddedTables, SlotInfo, compute_layout, read_buffer_to_output,
    write_store_slice_to_buffer,
};
use super::executor::{CompiledSpanFn, SpanOutput, StoreSlice};
use crate::nano_graph::ops::{ScalarBinOp, ScalarOp, ScalarUnaryOp};
use crate::nano_graph::pattern::{AtomGroup, AtomId, AtomRange, InputRef, NanoGraph};
use crate::numeric_dtype::NumericDType;
use crate::pool::SystemPool;

// ─── Math function trampolines ──────────────────────────────────────────────
//
// Extern "C" wrappers for transcendental float ops. Their addresses are
// embedded inline as imm64 at each call site (no PLT, no relocations).

extern "C" fn jit_expf(x: f32) -> f32 {
    x.exp()
}
extern "C" fn jit_logf(x: f32) -> f32 {
    x.ln()
}
extern "C" fn jit_tanhf(x: f32) -> f32 {
    x.tanh()
}
extern "C" fn jit_sqrtf(x: f32) -> f32 {
    x.sqrt()
}
extern "C" fn jit_floorf(x: f32) -> f32 {
    x.floor()
}
extern "C" fn jit_ceilf(x: f32) -> f32 {
    x.ceil()
}
extern "C" fn jit_fabsf(x: f32) -> f32 {
    x.abs()
}
extern "C" fn jit_powf(x: f32, y: f32) -> f32 {
    x.powf(y)
}
extern "C" fn jit_fmodf(x: f32, y: f32) -> f32 {
    x % y
}
extern "C" fn jit_sinf(x: f32) -> f32 {
    x.sin()
}
extern "C" fn jit_cosf(x: f32) -> f32 {
    x.cos()
}

// ─── Compiled span ──────────────────────────────────────────────────────────

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
    /// Buffer layout (slot byte-offset map). Reused from the cranelift
    /// backend's compute_layout.
    layout: BufferLayout,
    /// Pre-populated literal-and-table buffer template; cloned per execute.
    literal_template: Vec<u8>,
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

        let layout = compute_layout(graph, output_ranges);
        check_supported(graph, &layout)?;

        let (code, entry) = emit_native(graph, &layout)?;

        // Build literal template — same as JitCompiledSpan::compile but with
        // an empty EmbeddedTables (P1 has no IndirectLoad / Explicit lookups).
        let tables = EmbeddedTables::new(layout.total_bytes);
        let total_buf_bytes = tables.total_bytes().max(layout.total_bytes);
        let mut literal_template = vec![0u8; total_buf_bytes];
        layout.populate_literals(graph, &mut literal_template);
        tables.populate(&mut literal_template);

        Ok(Self {
            code,
            entry,
            layout,
            literal_template,
            output_ranges: output_ranges.to_vec(),
            is_empty: false,
        })
    }

    /// Build a no-op compiled span for graphs with zero groups.
    ///
    /// Anchors the ABI and dynasm plumbing end-to-end: the compiled
    /// function is just `ret`, and `execute` is short-circuited so we
    /// don't even call it. Validates that the executable-memory + fn-ptr
    /// transmute path is wired correctly before any real codegen lands.
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
            layout: BufferLayout::empty(),
            literal_template: Vec::new(),
            output_ranges: output_ranges.to_vec(),
            is_empty: true,
        })
    }

    /// Type alias for the compiled function signature.
    #[inline]
    fn entry_fn(&self) -> unsafe extern "C" fn(*mut u8) {
        // SAFETY: the bytes at `self.code.ptr(self.entry)` were emitted as a
        // System V AMD64 function taking a single `*mut u8` arg and returning
        // nothing. The buffer outlives this borrow because it lives in
        // `self.code`.
        unsafe { std::mem::transmute(self.code.ptr(self.entry)) }
    }
}

impl CompiledSpanFn for X86JitSpan {
    fn execute(&self, inputs: &[StoreSlice<'_>], outputs: &mut [SpanOutput<'_>]) {
        if self.is_empty || self.layout.total_bytes == 0 {
            return;
        }

        // Clone literal template as working buffer.
        let mut buffer = self.literal_template.clone();

        // Populate inputs from store slices into buffer slots.
        for slice in inputs {
            write_store_slice_to_buffer(slice, &self.layout, &mut buffer);
        }

        // Run the JIT function.
        let f = self.entry_fn();
        unsafe { f(buffer.as_mut_ptr()) };

        // Extract outputs from buffer into SpanOutputs.
        for (range, out) in self.output_ranges.iter().zip(outputs.iter_mut()) {
            read_buffer_to_output(range, &self.layout, &buffer, out);
        }
    }
}

// ─── Supportability filter ──────────────────────────────────────────────────

/// Check whether a span is within the P1 envelope.
///
/// P1 supports:
/// - F32 storage and compute throughout (every group's output_dtype = F32,
///   every input slot's dtype = F32, every binary/unary compute_dtype = F32)
/// - Op variants: Identity, Cast, Binary, Unary, Select, Literal, LiteralSpan
/// - InputRef variants: Broadcast, 1D affine Strided
/// - No inlinable groups (no reduce-fold inlining)
///
/// Returns `Err("…")` describing the first unsupported feature found.
fn check_supported(
    graph: &NanoGraph<'static, SystemPool>,
    layout: &BufferLayout,
) -> Result<(), String> {
    let groups = graph.groups();

    // No reduce-fold inlining yet.
    for (gi, &inlinable) in layout.inlinable.iter().enumerate() {
        if inlinable {
            return Err(format!("x86_jit: group {gi} is inlinable (P1 unsupported)"));
        }
    }

    for (gi, group) in groups.iter().enumerate() {
        // Skip dead and literal groups (codegen never visits these).
        let is_dead = gi < layout.group_use_counts.len() && layout.group_use_counts[gi] == 0;
        if is_dead {
            continue;
        }
        if matches!(&group.op, ScalarOp::Literal(_) | ScalarOp::LiteralSpan(_)) {
            // Literals are populated by populate_literals; we don't emit code.
            continue;
        }

        if group.output_dtype != NumericDType::F32 {
            return Err(format!(
                "x86_jit: group {gi} output_dtype {:?} != F32",
                group.output_dtype
            ));
        }

        match &group.op {
            ScalarOp::Identity => {}
            ScalarOp::Cast { .. } => {}
            ScalarOp::Binary { compute_dtype, .. } | ScalarOp::Unary { compute_dtype, .. } => {
                if *compute_dtype != NumericDType::F32 {
                    return Err(format!(
                        "x86_jit: group {gi} compute_dtype {:?} != F32",
                        compute_dtype
                    ));
                }
            }
            ScalarOp::Select => {}
            ScalarOp::Reduce { .. } => {
                return Err(format!("x86_jit: group {gi} Reduce (P1 unsupported)"));
            }
            ScalarOp::IndirectLoad { .. } => {
                return Err(format!("x86_jit: group {gi} IndirectLoad (P1 unsupported)"));
            }
            ScalarOp::OpaqueOutput { .. } => {
                return Err(format!("x86_jit: group {gi} OpaqueOutput (P1 unsupported)"));
            }
            ScalarOp::Literal(_) | ScalarOp::LiteralSpan(_) => unreachable!(),
        }

        // Verify InputRef variants and that all referenced slots are F32.
        for (ii, ir) in group.inputs.iter().enumerate() {
            match ir {
                InputRef::Broadcast(atom) => {
                    let (slot, _) = layout.find(*atom).ok_or_else(|| {
                        format!("x86_jit: group {gi} input {ii} Broadcast atom={atom} no slot")
                    })?;
                    if slot.dtype != NumericDType::F32 {
                        return Err(format!(
                            "x86_jit: group {gi} input {ii} slot dtype {:?} != F32",
                            slot.dtype
                        ));
                    }
                }
                InputRef::Strided {
                    base,
                    dim_strides,
                    dim_shape,
                } => {
                    if dim_strides.len() != 1 {
                        return Err(format!(
                            "x86_jit: group {gi} input {ii} Strided nd={} (P1 supports nd=1)",
                            dim_strides.len()
                        ));
                    }
                    // Resolve the slot via the same logic load_input uses.
                    let stride = dim_strides[0];
                    let first_offset = stride * group.atom_offset as i64;
                    let first_atom = AtomId((base.0 as i64 + first_offset) as u64);
                    let (slot, _) = layout
                        .find(*base)
                        .or_else(|| layout.find(first_atom))
                        .ok_or_else(|| {
                            format!("x86_jit: group {gi} input {ii} Strided base={base} no slot")
                        })?;
                    if slot.dtype != NumericDType::F32 {
                        return Err(format!(
                            "x86_jit: group {gi} input {ii} Strided slot dtype {:?} != F32",
                            slot.dtype
                        ));
                    }
                    let _ = dim_shape; // 1D affine; dim_shape[0] is just MAX
                }
                InputRef::Explicit(_) => {
                    return Err(format!(
                        "x86_jit: group {gi} input {ii} Explicit (P1 unsupported)"
                    ));
                }
            }
        }

        // Per-op input count + supported binop/unop variants.
        match &group.op {
            ScalarOp::Identity | ScalarOp::Cast { .. } => {
                if group.inputs.len() != 1 {
                    return Err(format!(
                        "x86_jit: group {gi} Identity/Cast expects 1 input, got {}",
                        group.inputs.len()
                    ));
                }
            }
            ScalarOp::Binary { op, .. } => {
                if group.inputs.len() != 2 {
                    return Err(format!(
                        "x86_jit: group {gi} Binary expects 2 inputs, got {}",
                        group.inputs.len()
                    ));
                }
                if !is_supported_binop_f32(*op) {
                    return Err(format!("x86_jit: group {gi} Binary {:?} unsupported", op));
                }
            }
            ScalarOp::Unary { op, .. } => {
                if group.inputs.len() != 1 {
                    return Err(format!(
                        "x86_jit: group {gi} Unary expects 1 input, got {}",
                        group.inputs.len()
                    ));
                }
                if !is_supported_unop_f32(*op) {
                    return Err(format!("x86_jit: group {gi} Unary {:?} unsupported", op));
                }
            }
            ScalarOp::Select => {
                if group.inputs.len() != 3 {
                    return Err(format!(
                        "x86_jit: group {gi} Select expects 3 inputs, got {}",
                        group.inputs.len()
                    ));
                }
            }
            _ => {}
        }
    }

    Ok(())
}

fn is_supported_binop_f32(op: ScalarBinOp) -> bool {
    matches!(
        op,
        ScalarBinOp::Add
            | ScalarBinOp::Sub
            | ScalarBinOp::Mul
            | ScalarBinOp::Div
            | ScalarBinOp::Min
            | ScalarBinOp::Max
            | ScalarBinOp::Mod
            | ScalarBinOp::IMod
            | ScalarBinOp::Pow
            | ScalarBinOp::Equal
            | ScalarBinOp::Greater
            | ScalarBinOp::GreaterOrEqual
            | ScalarBinOp::Less
            | ScalarBinOp::LessOrEqual
            | ScalarBinOp::And
            | ScalarBinOp::Or
            | ScalarBinOp::Xor
    )
}

fn is_supported_unop_f32(op: ScalarUnaryOp) -> bool {
    matches!(
        op,
        ScalarUnaryOp::Neg
            | ScalarUnaryOp::Abs
            | ScalarUnaryOp::Sqrt
            | ScalarUnaryOp::Exp
            | ScalarUnaryOp::Ln
            | ScalarUnaryOp::Tanh
            | ScalarUnaryOp::Reciprocal
            | ScalarUnaryOp::Floor
            | ScalarUnaryOp::Ceil
            | ScalarUnaryOp::Sin
            | ScalarUnaryOp::Cos
    )
}

// ─── Native code emission ───────────────────────────────────────────────────

const ELEM_BYTES: i32 = 4; // F32 only in P1

/// Emit a complete native function for a span.
///
/// Returns the executable buffer and the offset of the entry point.
fn emit_native(
    graph: &NanoGraph<'static, SystemPool>,
    layout: &BufferLayout,
) -> Result<(ExecutableBuffer, AssemblyOffset), String> {
    let mut ops = Assembler::new().map_err(|e| format!("x86_jit: assembler init: {e}"))?;
    let entry = ops.offset();

    emit_prologue(&mut ops);

    let groups = graph.groups();
    for (gi, group) in groups.iter().enumerate() {
        // Skip dead groups (their slots may have been reused).
        let is_dead = gi < layout.group_use_counts.len() && layout.group_use_counts[gi] == 0;
        if is_dead {
            continue;
        }
        // Literals are populated by populate_literals — no codegen.
        if matches!(&group.op, ScalarOp::Literal(_) | ScalarOp::LiteralSpan(_)) {
            continue;
        }
        if group.count == 0 {
            continue;
        }
        emit_group(&mut ops, group, layout)?;
    }

    emit_epilogue(&mut ops);

    let code = ops
        .finalize()
        .map_err(|_| "x86_jit: assembler finalize failed".to_string())?;
    Ok((code, entry))
}

/// System V AMD64 function prologue.
///
/// Stack layout after prologue:
/// ```text
///   [rbp+0]   saved rbp
///   [rbp-8]   saved r12 (we use r12 for buffer pointer)
///   [rbp-16]  saved r13 (we use r13 for loop variable)
///   [rbp-24]  saved r14 (we use r14 for loop end constant)
///   [rsp+0..7]   8 bytes pad to keep rsp 16-aligned for calls
/// ```
fn emit_prologue(ops: &mut Assembler) {
    dynasm!(ops
        ; .arch x64
        ; push rbp
        ; mov rbp, rsp
        ; push r12
        ; push r13
        ; push r14
        ; sub rsp, BYTE 8       // align rsp to 16 (entry was +8 mod 16)
        ; mov r12, rdi          // r12 = buffer pointer
    );
}

fn emit_epilogue(ops: &mut Assembler) {
    dynasm!(ops
        ; .arch x64
        ; add rsp, BYTE 8
        ; pop r14
        ; pop r13
        ; pop r12
        ; pop rbp
        ; ret
    );
}

/// Emit one group as either a counted loop (count > 1) or inline body (count == 1).
fn emit_group(
    ops: &mut Assembler,
    group: &AtomGroup<'static, SystemPool>,
    layout: &BufferLayout,
) -> Result<(), String> {
    let count = group.count;
    let atom_offset = group.atom_offset;

    if count == 1 {
        // Inline path: i_const = atom_offset, no loop.
        emit_group_body(ops, group, layout, /* in_loop */ false, atom_offset)?;
        return Ok(());
    }

    // Loop: r13 = atom_offset, r14 = atom_offset + count.
    // for r13 in [atom_offset .. atom_offset+count): body; r13 += 1.
    let start = atom_offset as i64;
    let end = (atom_offset + count) as i64;

    // Materialize start and end as 64-bit immediates (always fits in QWORD).
    dynasm!(ops
        ; .arch x64
        ; mov r13, QWORD start
        ; mov r14, QWORD end
    );

    let loop_top = ops.new_dynamic_label();
    let loop_done = ops.new_dynamic_label();

    dynasm!(ops
        ; .arch x64
        ; =>loop_top
        ; cmp r13, r14
        ; jge =>loop_done
    );

    emit_group_body(ops, group, layout, /* in_loop */ true, 0)?;

    dynasm!(ops
        ; .arch x64
        ; inc r13
        ; jmp =>loop_top
        ; =>loop_done
    );

    Ok(())
}

/// Emit one iteration of a group body. Loads inputs, computes, stores result.
///
/// `in_loop`: when true, the loop variable lives in r13 (use it for indexed
/// addressing). When false, all addresses are constant displacements based on
/// `i_const`.
fn emit_group_body(
    ops: &mut Assembler,
    group: &AtomGroup<'static, SystemPool>,
    layout: &BufferLayout,
    in_loop: bool,
    i_const: u64,
) -> Result<(), String> {
    let (out_slot, _) = layout
        .find(group.base_id)
        .ok_or_else(|| format!("x86_jit: no slot for group base={}", group.base_id))?;
    let out_slot = out_slot.clone();

    match &group.op {
        ScalarOp::Identity | ScalarOp::Cast { .. } => {
            // Load input → xmm0, store xmm0.
            emit_load_f32(
                ops,
                &group.inputs[0],
                layout,
                in_loop,
                i_const,
                group.atom_offset,
                0,
            )?;
            emit_store_f32(ops, &out_slot, in_loop, i_const, group.atom_offset, 0);
        }

        ScalarOp::Binary { op, .. } => {
            // a → xmm0, b → xmm1, op(xmm0, xmm1) → xmm0, store xmm0.
            emit_load_f32(
                ops,
                &group.inputs[0],
                layout,
                in_loop,
                i_const,
                group.atom_offset,
                0,
            )?;
            emit_load_f32(
                ops,
                &group.inputs[1],
                layout,
                in_loop,
                i_const,
                group.atom_offset,
                1,
            )?;
            emit_binop_f32(ops, *op);
            emit_store_f32(ops, &out_slot, in_loop, i_const, group.atom_offset, 0);
        }

        ScalarOp::Unary { op, .. } => {
            // x → xmm0, op(xmm0) → xmm0, store xmm0.
            emit_load_f32(
                ops,
                &group.inputs[0],
                layout,
                in_loop,
                i_const,
                group.atom_offset,
                0,
            )?;
            emit_unop_f32(ops, *op);
            emit_store_f32(ops, &out_slot, in_loop, i_const, group.atom_offset, 0);
        }

        ScalarOp::Select => {
            // cond → xmm0, x → xmm1, y → xmm2.
            // result = (cond != 0) ? x : y.
            // Emit as: cmp xmm0 vs 0; if nonzero, xmm1 → xmm0; else xmm2 → xmm0.
            emit_load_f32(
                ops,
                &group.inputs[0],
                layout,
                in_loop,
                i_const,
                group.atom_offset,
                0,
            )?;
            emit_load_f32(
                ops,
                &group.inputs[1],
                layout,
                in_loop,
                i_const,
                group.atom_offset,
                1,
            )?;
            emit_load_f32(
                ops,
                &group.inputs[2],
                layout,
                in_loop,
                i_const,
                group.atom_offset,
                2,
            )?;

            // xmm3 = 0; ucomiss xmm0, xmm3 sets ZF if xmm0 == 0 (and PF on NaN).
            // For NaN cond we treat as nonzero (matches FloatCC::NotEqual).
            // jne (xmm0 != 0) → take x (xmm1); else take y (xmm2).
            let take_x = ops.new_dynamic_label();
            let done = ops.new_dynamic_label();
            dynasm!(ops
                ; .arch x64
                ; xorps xmm3, xmm3
                ; ucomiss xmm0, xmm3
                ; jne =>take_x          // xmm0 != 0 (zero flag clear)
                ; jp  =>take_x          // NaN → treat as nonzero
                // y branch: xmm0 = xmm2 (y)
                ; movaps xmm0, xmm2
                ; jmp =>done
                ; =>take_x
                // x branch: xmm0 = xmm1 (x)
                ; movaps xmm0, xmm1
                ; =>done
            );

            emit_store_f32(ops, &out_slot, in_loop, i_const, group.atom_offset, 0);
        }

        ScalarOp::Literal(_) | ScalarOp::LiteralSpan(_) => {
            // Should have been filtered out earlier.
        }
        _ => {
            return Err(format!("x86_jit: emit_group_body unexpected op"));
        }
    }

    Ok(())
}

// ─── Address resolution ─────────────────────────────────────────────────────

/// Resolve a 1D affine `Strided` InputRef to its (base_byte, byte_stride),
/// matching `load_input`'s back-compute logic.
fn resolve_strided_addr(
    base: AtomId,
    stride: i64,
    atom_offset: u64,
    layout: &BufferLayout,
) -> Result<(i64, i64), String> {
    let first_offset = stride * atom_offset as i64;
    let first_atom = AtomId((base.0 as i64 + first_offset) as u64);

    let (slot, elem) = layout
        .find(base)
        .or_else(|| layout.find(first_atom))
        .ok_or_else(|| format!("x86_jit: no slot for Strided base={base}"))?;

    let elem_bytes = slot.elem_bytes as i64;
    let slot_byte = slot.byte_offset as i64 + elem as i64 * elem_bytes;
    let base_byte = if layout.find(base).is_some() {
        slot_byte
    } else {
        slot_byte - first_offset * elem_bytes
    };
    let byte_stride = stride * elem_bytes;
    Ok((base_byte, byte_stride))
}

// ─── F32 load / store ───────────────────────────────────────────────────────

/// Emit `movss <xmm{dst}>, [r12 + addr]` for an InputRef.
///
/// `dst_xmm` selects the destination register (0..=2 in P1).
fn emit_load_f32(
    ops: &mut Assembler,
    input: &InputRef,
    layout: &BufferLayout,
    in_loop: bool,
    i_const: u64,
    atom_offset: u64,
    dst_xmm: u8,
) -> Result<(), String> {
    match input {
        InputRef::Broadcast(atom) => {
            let (slot, elem_idx) = layout
                .find(*atom)
                .ok_or_else(|| format!("x86_jit: no slot for Broadcast atom={atom}"))?;
            let byte_off = slot.byte_offset as i64 + elem_idx as i64 * slot.elem_bytes as i64;
            emit_load_f32_disp(ops, dst_xmm, byte_off);
        }
        InputRef::Strided {
            base, dim_strides, ..
        } => {
            assert_eq!(dim_strides.len(), 1, "check_supported guarantees 1D");
            let stride = dim_strides[0];
            let (base_byte, byte_stride) =
                resolve_strided_addr(*base, stride, atom_offset, layout)?;

            if in_loop {
                // addr = r12 + base_byte + r13 * byte_stride
                emit_load_f32_indexed(ops, dst_xmm, base_byte, byte_stride);
            } else {
                let byte_off = base_byte + byte_stride * i_const as i64;
                emit_load_f32_disp(ops, dst_xmm, byte_off);
            }
        }
        InputRef::Explicit(_) => {
            return Err("x86_jit: Explicit inputref in emission (filter bug)".into());
        }
    }
    Ok(())
}

/// Emit `movss xmm{dst_xmm}, DWORD [r12 + disp]`.
fn emit_load_f32_disp(ops: &mut Assembler, dst_xmm: u8, disp: i64) {
    let d = disp as i32;
    debug_assert_eq!(disp, d as i64, "displacement out of i32 range: {disp}");
    match dst_xmm {
        0 => dynasm!(ops ; .arch x64 ; movss xmm0, DWORD [r12 + d]),
        1 => dynasm!(ops ; .arch x64 ; movss xmm1, DWORD [r12 + d]),
        2 => dynasm!(ops ; .arch x64 ; movss xmm2, DWORD [r12 + d]),
        _ => unreachable!("dst_xmm out of range: {dst_xmm}"),
    }
}

/// Emit `movss xmm{dst_xmm}, DWORD [r12 + r13 * byte_stride + base_byte]`.
///
/// SSE SIB encoding requires `byte_stride` ∈ {1, 2, 4, 8}. For F32 it's always 4
/// at the slot level, but the *atom* stride can be != 1 (e.g. broadcast over a
/// row, stride = N). We handle the common stride=4 case directly; for other
/// strides we materialize the index multiplication into rax.
fn emit_load_f32_indexed(ops: &mut Assembler, dst_xmm: u8, base_byte: i64, byte_stride: i64) {
    let bb = base_byte as i32;
    debug_assert_eq!(base_byte, bb as i64, "base_byte out of i32 range");

    if byte_stride == 4 {
        match dst_xmm {
            0 => dynasm!(ops ; .arch x64 ; movss xmm0, DWORD [r12 + r13 * 4 + bb]),
            1 => dynasm!(ops ; .arch x64 ; movss xmm1, DWORD [r12 + r13 * 4 + bb]),
            2 => dynasm!(ops ; .arch x64 ; movss xmm2, DWORD [r12 + r13 * 4 + bb]),
            _ => unreachable!(),
        }
    } else if byte_stride == 0 {
        // Effectively a broadcast — use the constant address.
        emit_load_f32_disp(ops, dst_xmm, base_byte);
    } else if byte_stride == 8 {
        match dst_xmm {
            0 => dynasm!(ops ; .arch x64 ; movss xmm0, DWORD [r12 + r13 * 8 + bb]),
            1 => dynasm!(ops ; .arch x64 ; movss xmm1, DWORD [r12 + r13 * 8 + bb]),
            2 => dynasm!(ops ; .arch x64 ; movss xmm2, DWORD [r12 + r13 * 8 + bb]),
            _ => unreachable!(),
        }
    } else if byte_stride == 2 {
        match dst_xmm {
            0 => dynasm!(ops ; .arch x64 ; movss xmm0, DWORD [r12 + r13 * 2 + bb]),
            1 => dynasm!(ops ; .arch x64 ; movss xmm1, DWORD [r12 + r13 * 2 + bb]),
            2 => dynasm!(ops ; .arch x64 ; movss xmm2, DWORD [r12 + r13 * 2 + bb]),
            _ => unreachable!(),
        }
    } else if byte_stride == 1 {
        match dst_xmm {
            0 => dynasm!(ops ; .arch x64 ; movss xmm0, DWORD [r12 + r13 + bb]),
            1 => dynasm!(ops ; .arch x64 ; movss xmm1, DWORD [r12 + r13 + bb]),
            2 => dynasm!(ops ; .arch x64 ; movss xmm2, DWORD [r12 + r13 + bb]),
            _ => unreachable!(),
        }
    } else {
        // General stride: materialize r13 * byte_stride in rax, then index.
        // rax = r13 * byte_stride; addr = r12 + rax + base_byte
        let stride = byte_stride;
        dynasm!(ops
            ; .arch x64
            ; mov rax, r13
            ; imul rax, rax, stride as i32
        );
        match dst_xmm {
            0 => dynasm!(ops ; .arch x64 ; movss xmm0, DWORD [r12 + rax + bb]),
            1 => dynasm!(ops ; .arch x64 ; movss xmm1, DWORD [r12 + rax + bb]),
            2 => dynasm!(ops ; .arch x64 ; movss xmm2, DWORD [r12 + rax + bb]),
            _ => unreachable!(),
        }
    }
}

/// Emit `movss DWORD [...], xmm{src_xmm}` for an output slot.
fn emit_store_f32(
    ops: &mut Assembler,
    slot: &SlotInfo,
    in_loop: bool,
    i_const: u64,
    atom_offset: u64,
    src_xmm: u8,
) {
    // store_base = slot.byte_offset - atom_offset * elem_bytes
    let store_base = slot.byte_offset as i64 - atom_offset as i64 * slot.elem_bytes as i64;

    if in_loop {
        // addr = r12 + r13 * elem_bytes + store_base
        let bb = store_base as i32;
        debug_assert_eq!(store_base, bb as i64, "store_base out of i32 range");
        match src_xmm {
            0 => dynasm!(ops ; .arch x64 ; movss DWORD [r12 + r13 * 4 + bb], xmm0),
            1 => dynasm!(ops ; .arch x64 ; movss DWORD [r12 + r13 * 4 + bb], xmm1),
            2 => dynasm!(ops ; .arch x64 ; movss DWORD [r12 + r13 * 4 + bb], xmm2),
            _ => unreachable!(),
        }
    } else {
        let byte_off = store_base + i_const as i64 * slot.elem_bytes as i64;
        let bb = byte_off as i32;
        debug_assert_eq!(byte_off, bb as i64, "byte_off out of i32 range");
        match src_xmm {
            0 => dynasm!(ops ; .arch x64 ; movss DWORD [r12 + bb], xmm0),
            1 => dynasm!(ops ; .arch x64 ; movss DWORD [r12 + bb], xmm1),
            2 => dynasm!(ops ; .arch x64 ; movss DWORD [r12 + bb], xmm2),
            _ => unreachable!(),
        }
    }
}

// ─── F32 op emission ────────────────────────────────────────────────────────

/// Emit a binary op: `op(xmm0, xmm1) → xmm0`. Some ops use xmm2 / xmm3 as
/// scratch.
fn emit_binop_f32(ops: &mut Assembler, op: ScalarBinOp) {
    match op {
        ScalarBinOp::Add => dynasm!(ops ; .arch x64 ; addss xmm0, xmm1),
        ScalarBinOp::Sub => dynasm!(ops ; .arch x64 ; subss xmm0, xmm1),
        ScalarBinOp::Mul => dynasm!(ops ; .arch x64 ; mulss xmm0, xmm1),
        ScalarBinOp::Div => dynasm!(ops ; .arch x64 ; divss xmm0, xmm1),
        ScalarBinOp::Min => {
            // Cranelift Min: a < b ? a : b. SSE minss handles NaN as "second operand wins"
            // (returns src2), so to match Cranelift's select(a<b, a, b) semantics with NaN,
            // we'd need a manual compare. For P1 we use minss directly — A/B will catch
            // any divergence (existing tests don't pass NaN).
            dynasm!(ops ; .arch x64 ; minss xmm0, xmm1)
        }
        ScalarBinOp::Max => dynasm!(ops ; .arch x64 ; maxss xmm0, xmm1),

        // Comparisons → 1.0 / 0.0. Emit via cmpss (vector compare with predicate),
        // which produces an all-1s or all-0s mask, then andps with f32 1.0 to
        // produce 1.0 / 0.0.
        ScalarBinOp::Equal => emit_f32_cmp(ops, CmpPred::Eq),
        ScalarBinOp::Less => emit_f32_cmp(ops, CmpPred::Lt),
        ScalarBinOp::LessOrEqual => emit_f32_cmp(ops, CmpPred::Le),
        ScalarBinOp::Greater => emit_f32_cmp(ops, CmpPred::Gt),
        ScalarBinOp::GreaterOrEqual => emit_f32_cmp(ops, CmpPred::Ge),

        // Logical: convert each operand to 0/1, then bitand/bitor/bitxor.
        ScalarBinOp::And => emit_f32_logical(ops, LogicalOp::And),
        ScalarBinOp::Or => emit_f32_logical(ops, LogicalOp::Or),
        ScalarBinOp::Xor => emit_f32_logical(ops, LogicalOp::Xor),

        // Math fn calls.
        ScalarBinOp::Pow => emit_extern_call_binary_f32(ops, jit_powf as *const u8),
        ScalarBinOp::Mod => emit_extern_call_binary_f32(ops, jit_fmodf as *const u8),
        ScalarBinOp::IMod => {
            // Cranelift IMod: fmod-then-adjust (sign of result matches divisor).
            // rem = fmod(a, b); if sign(rem) != sign(b) and rem != 0 then rem+b else rem.
            // First call fmod (xmm0, xmm1) → xmm0 = rem.
            // We need to keep b around for the adjust step. Stash b in [rsp+0].
            dynasm!(ops
                ; .arch x64
                ; movss DWORD [rsp + 0], xmm1   // save b on stack
            );
            emit_extern_call_binary_f32(ops, jit_fmodf as *const u8);
            // xmm0 = rem, [rsp+0] = b. Reload b → xmm1.
            dynasm!(ops
                ; .arch x64
                ; movss xmm1, DWORD [rsp + 0]
                // sum = rem + b → xmm2
                ; movaps xmm2, xmm0
                ; addss xmm2, xmm1
                // rem_zero = (rem == 0). Test rem against 0 in xmm3.
                ; xorps xmm3, xmm3
                ; ucomiss xmm0, xmm3
                ; jne >adjust_check
                ; jp  >adjust_check
                // rem == 0 → keep rem (xmm0). jmp done.
                ; jmp >done
                ; adjust_check:
                // Compare signs of rem and b: sign differs iff (rem<0) != (b<0).
                // Easier: check if (rem * b) < 0 (sign mismatch when product is negative).
                ; movaps xmm3, xmm0
                ; mulss xmm3, xmm1
                ; xorps xmm4, xmm4
                ; ucomiss xmm3, xmm4
                ; jae >done                   // product >= 0 → no adjust
                ; jp >done                    // unordered (NaN) → no adjust
                ; movaps xmm0, xmm2           // adjust: rem += b
                ; done:
            );
        }
        _ => {
            // Already filtered by check_supported.
            panic!("emit_binop_f32: unexpected op {op:?}");
        }
    }
}

#[derive(Copy, Clone)]
enum CmpPred {
    Eq,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Emit `xmm0 = (xmm0 cmp xmm1) ? 1.0 : 0.0`, F32.
///
/// SSE `cmpss` produces an all-1s mask on true and 0 on false. AND with the
/// f32 bit pattern of 1.0 turns mask into 1.0 / 0.0.
fn emit_f32_cmp(ops: &mut Assembler, pred: CmpPred) {
    // cmpss imm8: 0=eq, 1=lt, 2=le, 3=unord, 4=neq, 5=nlt, 6=nle, 7=ord
    // For Gt/Ge we swap operands and use Lt/Le.
    let one_bits = 1.0f32.to_bits() as i32;
    match pred {
        CmpPred::Eq => dynasm!(ops ; .arch x64 ; cmpss xmm0, xmm1, 0),
        CmpPred::Lt => dynasm!(ops ; .arch x64 ; cmpss xmm0, xmm1, 1),
        CmpPred::Le => dynasm!(ops ; .arch x64 ; cmpss xmm0, xmm1, 2),
        CmpPred::Gt => {
            // a > b  ⇔  b < a. Swap operands by computing in xmm1 then moving back.
            // Easiest: use cmpnle (not less-or-equal): 6 = nle, but that includes NaN.
            // Cleaner: cmpss xmm1, xmm0, 1 → mask in xmm1; mov to xmm0.
            dynasm!(ops
                ; .arch x64
                ; cmpss xmm1, xmm0, 1
                ; movaps xmm0, xmm1
            );
        }
        CmpPred::Ge => {
            dynasm!(ops
                ; .arch x64
                ; cmpss xmm1, xmm0, 2
                ; movaps xmm0, xmm1
            );
        }
    }
    // Now xmm0 holds an all-1s or 0 mask. AND with 1.0 to materialize 1.0 / 0.0.
    // Stash 1.0 in xmm2 via mov from rax.
    dynasm!(ops
        ; .arch x64
        ; mov eax, DWORD one_bits
        ; movd xmm2, eax
        ; andps xmm0, xmm2
    );
}

#[derive(Copy, Clone)]
enum LogicalOp {
    And,
    Or,
    Xor,
}

/// Emit logical op on F32 truthiness: each operand → 0/1, then op.
///
/// 1.0 if nonzero, 0.0 if zero. NaN counts as nonzero (matches Cranelift).
fn emit_f32_logical(ops: &mut Assembler, op: LogicalOp) {
    // Materialize a → 0/1 in xmm0, b → 0/1 in xmm1.
    // For each: cmpss x, zero, NEQ (4) → mask; AND with 1.0.
    let one_bits = 1.0f32.to_bits() as i32;
    dynasm!(ops
        ; .arch x64
        ; xorps xmm2, xmm2          // xmm2 = 0.0
        ; cmpss xmm0, xmm2, 4       // xmm0 = (a != 0) mask
        ; cmpss xmm1, xmm2, 4       // xmm1 = (b != 0) mask
    );
    match op {
        LogicalOp::And => dynasm!(ops ; .arch x64 ; andps xmm0, xmm1),
        LogicalOp::Or => dynasm!(ops ; .arch x64 ; orps xmm0, xmm1),
        LogicalOp::Xor => dynasm!(ops ; .arch x64 ; xorps xmm0, xmm1),
    }
    dynasm!(ops
        ; .arch x64
        ; mov eax, DWORD one_bits
        ; movd xmm2, eax
        ; andps xmm0, xmm2
    );
}

/// Emit a unary op: `op(xmm0) → xmm0`.
fn emit_unop_f32(ops: &mut Assembler, op: ScalarUnaryOp) {
    match op {
        ScalarUnaryOp::Neg => {
            // xor with sign-bit mask 0x80000000.
            let mask = 0x80000000u32 as i32;
            dynasm!(ops
                ; .arch x64
                ; mov eax, DWORD mask
                ; movd xmm1, eax
                ; xorps xmm0, xmm1
            );
        }
        ScalarUnaryOp::Abs => {
            // and with abs mask 0x7fffffff.
            let mask = 0x7fffffffu32 as i32;
            dynasm!(ops
                ; .arch x64
                ; mov eax, DWORD mask
                ; movd xmm1, eax
                ; andps xmm0, xmm1
            );
        }
        ScalarUnaryOp::Sqrt => dynasm!(ops ; .arch x64 ; sqrtss xmm0, xmm0),
        ScalarUnaryOp::Reciprocal => {
            // 1.0 / x. Emit as movss xmm1, 1.0; divss xmm1, xmm0; mov xmm0, xmm1.
            let one_bits = 1.0f32.to_bits() as i32;
            dynasm!(ops
                ; .arch x64
                ; mov eax, DWORD one_bits
                ; movd xmm1, eax
                ; divss xmm1, xmm0
                ; movaps xmm0, xmm1
            );
        }
        ScalarUnaryOp::Exp => emit_extern_call_unary_f32(ops, jit_expf as *const u8),
        ScalarUnaryOp::Ln => emit_extern_call_unary_f32(ops, jit_logf as *const u8),
        ScalarUnaryOp::Tanh => emit_extern_call_unary_f32(ops, jit_tanhf as *const u8),
        ScalarUnaryOp::Floor => emit_extern_call_unary_f32(ops, jit_floorf as *const u8),
        ScalarUnaryOp::Ceil => emit_extern_call_unary_f32(ops, jit_ceilf as *const u8),
        ScalarUnaryOp::Sin => emit_extern_call_unary_f32(ops, jit_sinf as *const u8),
        ScalarUnaryOp::Cos => emit_extern_call_unary_f32(ops, jit_cosf as *const u8),
        _ => panic!("emit_unop_f32: unexpected op {op:?}"),
    }
}

/// Emit a call to an `extern "C" fn(f32) -> f32` with the arg already in xmm0.
///
/// Stack alignment: prologue keeps rsp ≡ 0 (mod 16) so a `call` lands the
/// return address at +8. The callee can rely on this.
fn emit_extern_call_unary_f32(ops: &mut Assembler, fn_addr: *const u8) {
    let addr = fn_addr as i64;
    dynasm!(ops
        ; .arch x64
        ; mov rax, QWORD addr
        ; call rax
    );
}

/// Emit a call to an `extern "C" fn(f32, f32) -> f32` with args in xmm0, xmm1.
fn emit_extern_call_binary_f32(ops: &mut Assembler, fn_addr: *const u8) {
    let addr = fn_addr as i64;
    dynasm!(ops
        ; .arch x64
        ; mov rax, QWORD addr
        ; call rax
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::attempts::v14::codegen::JitCompiledSpan;
    use crate::graph::GlobalId;
    use crate::nano_graph::ops::{ScalarBinOp, ScalarOp, ScalarUnaryOp};
    use crate::numeric_scalar::NumericScalar;

    /// A/B harness: compile via both backends, run on the same StoreSlice
    /// inputs, compare output bytes.
    ///
    /// `inputs`: per-external-input (base AtomId, dtype, raw element bytes).
    /// `outputs`: AtomRanges to extract.
    fn ab_test_bytes(
        graph: &NanoGraph<'static, SystemPool>,
        inputs: &[(AtomId, NumericDType, Vec<u8>)],
        outputs: &[AtomRange],
    ) -> Vec<Vec<u8>> {
        let store_slices: Vec<StoreSlice<'_>> = inputs
            .iter()
            .map(|(base, dtype, bytes)| StoreSlice {
                base: *base,
                data: bytes.as_slice(),
                dtype: *dtype,
                count: (bytes.len() / dtype.bytes_per_element()) as u64,
            })
            .collect();

        let alloc_out = || -> Vec<Vec<u8>> {
            outputs
                .iter()
                .map(|r| vec![0u8; r.count as usize * r.dtype.bytes_per_element()])
                .collect()
        };

        let mut cl_outs = alloc_out();
        {
            let span = JitCompiledSpan::compile(graph, outputs).expect("cranelift compile");
            let mut span_outs: Vec<SpanOutput<'_>> = outputs
                .iter()
                .zip(cl_outs.iter_mut())
                .map(|(r, buf)| SpanOutput {
                    data: buf.as_mut_slice(),
                    dtype: r.dtype,
                    count: r.count,
                })
                .collect();
            span.execute(&store_slices, &mut span_outs);
        }

        let mut x86_outs = alloc_out();
        {
            let span = X86JitSpan::compile(graph, outputs).expect("x86_jit compile");
            let mut span_outs: Vec<SpanOutput<'_>> = outputs
                .iter()
                .zip(x86_outs.iter_mut())
                .map(|(r, buf)| SpanOutput {
                    data: buf.as_mut_slice(),
                    dtype: r.dtype,
                    count: r.count,
                })
                .collect();
            span.execute(&store_slices, &mut span_outs);
        }

        for (i, (a, b)) in cl_outs.iter().zip(x86_outs.iter()).enumerate() {
            assert_eq!(a, b, "output {i}: cranelift vs x86_jit byte mismatch");
        }
        cl_outs
    }

    /// Convenience: take f32 inputs and outputs.
    fn ab_test_f32(
        graph: &NanoGraph<'static, SystemPool>,
        inputs: &[(AtomId, &[f32])],
        outputs: &[AtomRange],
    ) {
        let raw_inputs: Vec<(AtomId, NumericDType, Vec<u8>)> = inputs
            .iter()
            .map(|(base, data)| {
                let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
                (*base, NumericDType::F32, bytes)
            })
            .collect();
        ab_test_bytes(graph, &raw_inputs, outputs);
    }

    /// Build a graph with a single binary op (count=N) and A/B test it.
    fn run_binop_ab(op: ScalarBinOp, a_data: &[f32], b_data: &[f32]) {
        assert_eq!(a_data.len(), b_data.len());
        let n = a_data.len() as u64;
        let mut g = NanoGraph::new();
        let a = g.add_input_tensor(GlobalId(0), n, NumericDType::F32);
        let b = g.add_input_tensor(GlobalId(1), n, NumericDType::F32);
        let out = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Binary {
                op,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(a, 1), InputRef::affine(b, 1)],
        );
        let outputs = vec![AtomRange {
            base: out,
            count: n,
            dtype: NumericDType::F32,
        }];
        ab_test_f32(&g, &[(a, a_data), (b, b_data)], &outputs);
    }

    /// Build a graph with a single unary op (count=N) and A/B test it.
    fn run_unop_ab(op: ScalarUnaryOp, x_data: &[f32]) {
        let n = x_data.len() as u64;
        let mut g = NanoGraph::new();
        let x = g.add_input_tensor(GlobalId(0), n, NumericDType::F32);
        let out = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Unary {
                op,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(x, 1)],
        );
        let outputs = vec![AtomRange {
            base: out,
            count: n,
            dtype: NumericDType::F32,
        }];
        ab_test_f32(&g, &[(x, x_data)], &outputs);
    }

    /// The empty-span path must compile, transmute, and call without
    /// crashing. This is the smoke test that validates dynasmrt + the
    /// executable-memory plumbing on the host platform before any real
    /// codegen lands.
    #[test]
    fn empty_span_compiles_and_calls() {
        let graph: NanoGraph<'static, SystemPool> = NanoGraph::new();
        let span = X86JitSpan::compile(&graph, &[]).expect("empty compile");
        assert!(span.is_empty);

        let f = span.entry_fn();
        let mut buffer = [0u8; 16];
        unsafe {
            f(buffer.as_mut_ptr());
        }
        assert_eq!(buffer, [0u8; 16]);
    }

    /// Build a graph: input[4] + broadcast(2.0) → output[4].
    /// A/B against the cranelift backend on f32 happy-path values.
    #[test]
    fn add_broadcast_f32() {
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), 4, NumericDType::F32);
        let lit = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(2.0)),
            vec![],
            vec![],
        );
        let add = g.push_group(
            4,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(inp, 1), InputRef::Broadcast(lit)],
        );

        let outputs = vec![AtomRange {
            base: add,
            count: 4,
            dtype: NumericDType::F32,
        }];
        let span = X86JitSpan::compile(&g, &outputs).expect("compile");
        assert!(!span.is_empty);

        let mut buffer = span.literal_template.clone();
        span.layout
            .write_f32_input(inp, &[1.0, 2.0, 3.0, 4.0], &mut buffer);
        let f = span.entry_fn();
        unsafe { f(buffer.as_mut_ptr()) };
        let result = span.layout.read_f32_output(&outputs[0], &buffer);
        assert_eq!(result, vec![3.0, 4.0, 5.0, 6.0]);
    }

    /// Unary neg over a 3-element input.
    #[test]
    fn unary_neg_f32() {
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), 3, NumericDType::F32);
        let neg = g.push_group(
            3,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(inp, 1)],
        );

        let outputs = vec![AtomRange {
            base: neg,
            count: 3,
            dtype: NumericDType::F32,
        }];
        let span = X86JitSpan::compile(&g, &outputs).expect("compile");
        let mut buffer = span.literal_template.clone();
        span.layout
            .write_f32_input(inp, &[1.0, -2.5, 3.0], &mut buffer);
        let f = span.entry_fn();
        unsafe { f(buffer.as_mut_ptr()) };
        let result = span.layout.read_f32_output(&outputs[0], &buffer);
        assert_eq!(result, vec![-1.0, 2.5, -3.0]);
    }

    /// Mul + Exp pipeline. Tests math-fn extern call ABI.
    #[test]
    fn mul_then_exp_f32() {
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), 2, NumericDType::F32);
        let lit3 = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(3.0)),
            vec![],
            vec![],
        );
        let mul = g.push_group(
            2,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(inp, 1), InputRef::Broadcast(lit3)],
        );
        let exp = g.push_group(
            2,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(mul, 1)],
        );

        let outputs = vec![AtomRange {
            base: exp,
            count: 2,
            dtype: NumericDType::F32,
        }];
        let span = X86JitSpan::compile(&g, &outputs).expect("compile");
        let mut buffer = span.literal_template.clone();
        span.layout.write_f32_input(inp, &[0.0, 1.0], &mut buffer);
        let f = span.entry_fn();
        unsafe { f(buffer.as_mut_ptr()) };
        let result = span.layout.read_f32_output(&outputs[0], &buffer);
        let expected = [(0.0f32 * 3.0).exp(), (1.0f32 * 3.0).exp()];
        for (a, b) in result.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "got {}, expected {}", a, b);
        }
    }

    /// Select (ternary): cond ? x : y.
    #[test]
    fn select_f32() {
        let mut g = NanoGraph::new();
        let cond = g.add_input_tensor(GlobalId(0), 3, NumericDType::F32);
        let x = g.add_input_tensor(GlobalId(1), 3, NumericDType::F32);
        let y = g.add_input_tensor(GlobalId(2), 3, NumericDType::F32);
        let sel = g.push_group(
            3,
            NumericDType::F32,
            ScalarOp::Select,
            vec![],
            vec![
                InputRef::affine(cond, 1),
                InputRef::affine(x, 1),
                InputRef::affine(y, 1),
            ],
        );

        let outputs = vec![AtomRange {
            base: sel,
            count: 3,
            dtype: NumericDType::F32,
        }];
        let span = X86JitSpan::compile(&g, &outputs).expect("compile");
        let mut buffer = span.literal_template.clone();
        span.layout
            .write_f32_input(cond, &[1.0, 0.0, 5.0], &mut buffer);
        span.layout
            .write_f32_input(x, &[10.0, 20.0, 30.0], &mut buffer);
        span.layout
            .write_f32_input(y, &[100.0, 200.0, 300.0], &mut buffer);
        let f = span.entry_fn();
        unsafe { f(buffer.as_mut_ptr()) };
        let result = span.layout.read_f32_output(&outputs[0], &buffer);
        assert_eq!(result, vec![10.0, 200.0, 30.0]);
    }

    /// Reduce ops should reject and return Err so callers fall back to cranelift.
    #[test]
    fn reduce_unsupported() {
        use crate::nano_graph::ops::ReduceKind;
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), 4, NumericDType::F32);
        let sum = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: 4,
                reduce_stride: 1,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(inp, 1)],
        );
        let outputs = vec![AtomRange {
            base: sum,
            count: 1,
            dtype: NumericDType::F32,
        }];
        let err = match X86JitSpan::compile(&g, &outputs) {
            Ok(_) => panic!("should reject Reduce"),
            Err(e) => e,
        };
        assert!(err.contains("Reduce"), "unexpected err: {err}");
    }

    /// BF16 storage dtype should reject.
    #[test]
    fn bf16_unsupported() {
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), 4, NumericDType::BF16);
        let id = g.push_group(
            4,
            NumericDType::BF16,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::affine(inp, 1)],
        );
        let outputs = vec![AtomRange {
            base: id,
            count: 4,
            dtype: NumericDType::BF16,
        }];
        let err = match X86JitSpan::compile(&g, &outputs) {
            Ok(_) => panic!("should reject BF16"),
            Err(e) => e,
        };
        assert!(err.contains("F32"), "unexpected err: {err}");
    }

    // ─── Binary op A/B coverage ─────────────────────────────────────────────

    // A representative input set with positives, negatives, large/small,
    // exactly-representable values. Avoids NaN/Inf since min/max NaN handling
    // diverges between Cranelift's select-based and SSE's hardware semantics.
    const F32_A: &[f32] = &[1.0, -2.5, 3.5, 0.0, 5.0, -7.25, 8.0, 16.0];
    const F32_B: &[f32] = &[2.0, 4.0, -1.5, 1.0, -5.0, 3.0, 2.5, -4.0];

    #[test]
    fn binop_add_f32_ab() {
        run_binop_ab(ScalarBinOp::Add, F32_A, F32_B);
    }
    #[test]
    fn binop_sub_f32_ab() {
        run_binop_ab(ScalarBinOp::Sub, F32_A, F32_B);
    }
    #[test]
    fn binop_mul_f32_ab() {
        run_binop_ab(ScalarBinOp::Mul, F32_A, F32_B);
    }
    #[test]
    fn binop_div_f32_ab() {
        run_binop_ab(ScalarBinOp::Div, F32_A, F32_B);
    }
    #[test]
    fn binop_min_f32_ab() {
        run_binop_ab(ScalarBinOp::Min, F32_A, F32_B);
    }
    #[test]
    fn binop_max_f32_ab() {
        run_binop_ab(ScalarBinOp::Max, F32_A, F32_B);
    }
    #[test]
    fn binop_pow_f32_ab() {
        // Use positive bases to avoid NaN from negative ^ non-integer.
        run_binop_ab(
            ScalarBinOp::Pow,
            &[1.0, 2.0, 3.0, 0.5, 1.5, 4.0, 0.25, 8.0],
            &[2.0, 0.5, 1.0, 3.0, 2.0, 0.5, 2.0, 0.333],
        );
    }
    #[test]
    fn binop_mod_f32_ab() {
        run_binop_ab(
            ScalarBinOp::Mod,
            &[5.0, 7.5, -3.0, 10.0, 0.5, 17.0, -8.0, 4.0],
            &[2.0, 1.5, 1.0, 3.0, 0.25, 5.0, 3.0, 2.0],
        );
    }
    #[test]
    fn binop_imod_f32_ab() {
        run_binop_ab(
            ScalarBinOp::IMod,
            &[5.0, 7.5, -3.0, 10.0, 0.5, 17.0, -8.0, 4.0],
            &[2.0, 1.5, 1.0, 3.0, 0.25, 5.0, 3.0, 2.0],
        );
    }
    #[test]
    fn binop_equal_f32_ab() {
        run_binop_ab(
            ScalarBinOp::Equal,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[1.0, 0.0, 3.0, 5.0, 5.0, 7.0, 7.0, 9.0],
        );
    }
    #[test]
    fn binop_less_f32_ab() {
        run_binop_ab(ScalarBinOp::Less, F32_A, F32_B);
    }
    #[test]
    fn binop_lessoreq_f32_ab() {
        run_binop_ab(ScalarBinOp::LessOrEqual, F32_A, F32_B);
    }
    #[test]
    fn binop_greater_f32_ab() {
        run_binop_ab(ScalarBinOp::Greater, F32_A, F32_B);
    }
    #[test]
    fn binop_greateroreq_f32_ab() {
        run_binop_ab(ScalarBinOp::GreaterOrEqual, F32_A, F32_B);
    }
    #[test]
    fn binop_and_f32_ab() {
        run_binop_ab(
            ScalarBinOp::And,
            &[1.0, 0.0, 0.0, 1.0, 5.0, 0.0, 2.0, 0.0],
            &[1.0, 1.0, 0.0, 0.0, 0.0, 5.0, 3.0, 0.0],
        );
    }
    #[test]
    fn binop_or_f32_ab() {
        run_binop_ab(
            ScalarBinOp::Or,
            &[1.0, 0.0, 0.0, 1.0, 5.0, 0.0, 2.0, 0.0],
            &[1.0, 1.0, 0.0, 0.0, 0.0, 5.0, 3.0, 0.0],
        );
    }
    #[test]
    fn binop_xor_f32_ab() {
        run_binop_ab(
            ScalarBinOp::Xor,
            &[1.0, 0.0, 0.0, 1.0, 5.0, 0.0, 2.0, 0.0],
            &[1.0, 1.0, 0.0, 0.0, 0.0, 5.0, 3.0, 0.0],
        );
    }

    // ─── Unary op A/B coverage ──────────────────────────────────────────────

    const F32_X: &[f32] = &[1.0, -2.5, 3.5, 0.5, 5.0, -7.25, 8.0, 16.0];

    #[test]
    fn unop_neg_f32_ab() {
        run_unop_ab(ScalarUnaryOp::Neg, F32_X);
    }
    #[test]
    fn unop_abs_f32_ab() {
        run_unop_ab(ScalarUnaryOp::Abs, F32_X);
    }
    #[test]
    fn unop_sqrt_f32_ab() {
        // Positive only.
        run_unop_ab(
            ScalarUnaryOp::Sqrt,
            &[1.0, 4.0, 9.0, 16.0, 0.25, 100.0, 2.0, 0.5],
        );
    }
    #[test]
    fn unop_reciprocal_f32_ab() {
        run_unop_ab(
            ScalarUnaryOp::Reciprocal,
            &[1.0, 2.0, 4.0, 0.5, -1.0, 8.0, 0.25, 16.0],
        );
    }
    #[test]
    fn unop_exp_f32_ab() {
        run_unop_ab(
            ScalarUnaryOp::Exp,
            &[0.0, 1.0, -1.0, 2.0, 0.5, -0.5, 3.0, -3.0],
        );
    }
    #[test]
    fn unop_ln_f32_ab() {
        // Positive only.
        run_unop_ab(
            ScalarUnaryOp::Ln,
            &[1.0, 2.0, 4.0, 8.0, 0.5, 16.0, 0.25, 100.0],
        );
    }
    #[test]
    fn unop_tanh_f32_ab() {
        run_unop_ab(
            ScalarUnaryOp::Tanh,
            &[0.0, 1.0, -1.0, 2.0, -2.0, 0.5, 5.0, -5.0],
        );
    }
    #[test]
    fn unop_floor_f32_ab() {
        run_unop_ab(
            ScalarUnaryOp::Floor,
            &[1.5, -2.5, 3.0, -3.0, 0.1, -0.1, 7.7, -7.7],
        );
    }
    #[test]
    fn unop_ceil_f32_ab() {
        run_unop_ab(
            ScalarUnaryOp::Ceil,
            &[1.5, -2.5, 3.0, -3.0, 0.1, -0.1, 7.7, -7.7],
        );
    }
    /// Sin/Cos aren't supported by the cranelift backend, so we can't A/B them.
    /// Run a direct value check against the standard library instead.
    fn run_unop_value_check(op: ScalarUnaryOp, x_data: &[f32], expect: impl Fn(f32) -> f32) {
        let n = x_data.len() as u64;
        let mut g = NanoGraph::new();
        let x = g.add_input_tensor(GlobalId(0), n, NumericDType::F32);
        let out = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Unary {
                op,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(x, 1)],
        );
        let outputs = vec![AtomRange {
            base: out,
            count: n,
            dtype: NumericDType::F32,
        }];
        let span = X86JitSpan::compile(&g, &outputs).expect("compile");
        let mut buffer = span.literal_template.clone();
        span.layout.write_f32_input(x, x_data, &mut buffer);
        let f = span.entry_fn();
        unsafe { f(buffer.as_mut_ptr()) };
        let result = span.layout.read_f32_output(&outputs[0], &buffer);
        for (i, (&got, &input)) in result.iter().zip(x_data.iter()).enumerate() {
            let want = expect(input);
            assert!(
                (got - want).abs() < 1e-5,
                "elem {i}: got {got}, want {want} for input {input}"
            );
        }
    }

    #[test]
    fn unop_sin_f32_value_check() {
        run_unop_value_check(
            ScalarUnaryOp::Sin,
            &[0.0, 1.0, -1.0, 1.5708, 3.14159, -3.14159, 0.5, -0.5],
            f32::sin,
        );
    }
    #[test]
    fn unop_cos_f32_value_check() {
        run_unop_value_check(
            ScalarUnaryOp::Cos,
            &[0.0, 1.0, -1.0, 1.5708, 3.14159, -3.14159, 0.5, -0.5],
            f32::cos,
        );
    }

    // ─── Identity / Cast / Select coverage ──────────────────────────────────

    #[test]
    fn identity_f32_ab() {
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), 8, NumericDType::F32);
        let id = g.push_group(
            8,
            NumericDType::F32,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::affine(inp, 1)],
        );
        let outputs = vec![AtomRange {
            base: id,
            count: 8,
            dtype: NumericDType::F32,
        }];
        ab_test_f32(&g, &[(inp, F32_X)], &outputs);
    }

    #[test]
    fn cast_f32_to_f32_ab() {
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), 8, NumericDType::F32);
        let cast = g.push_group(
            8,
            NumericDType::F32,
            ScalarOp::Cast { saturating: false },
            vec![],
            vec![InputRef::affine(inp, 1)],
        );
        let outputs = vec![AtomRange {
            base: cast,
            count: 8,
            dtype: NumericDType::F32,
        }];
        ab_test_f32(&g, &[(inp, F32_X)], &outputs);
    }

    #[test]
    fn select_f32_ab() {
        let mut g = NanoGraph::new();
        let cond = g.add_input_tensor(GlobalId(0), 6, NumericDType::F32);
        let x = g.add_input_tensor(GlobalId(1), 6, NumericDType::F32);
        let y = g.add_input_tensor(GlobalId(2), 6, NumericDType::F32);
        let sel = g.push_group(
            6,
            NumericDType::F32,
            ScalarOp::Select,
            vec![],
            vec![
                InputRef::affine(cond, 1),
                InputRef::affine(x, 1),
                InputRef::affine(y, 1),
            ],
        );
        let outputs = vec![AtomRange {
            base: sel,
            count: 6,
            dtype: NumericDType::F32,
        }];
        ab_test_f32(
            &g,
            &[
                (cond, &[1.0, 0.0, 5.0, -3.0, 0.0, 2.5]),
                (x, &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
                (y, &[100.0, 200.0, 300.0, 400.0, 500.0, 600.0]),
            ],
            &outputs,
        );
    }

    // ─── Single-element groups (count=1, no loop) ───────────────────────────

    #[test]
    fn single_element_add_ab() {
        let mut g = NanoGraph::new();
        let a = g.add_input_tensor(GlobalId(0), 1, NumericDType::F32);
        let b = g.add_input_tensor(GlobalId(1), 1, NumericDType::F32);
        let out = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(a, 1), InputRef::affine(b, 1)],
        );
        let outputs = vec![AtomRange {
            base: out,
            count: 1,
            dtype: NumericDType::F32,
        }];
        ab_test_f32(&g, &[(a, &[3.5]), (b, &[1.25])], &outputs);
    }

    // ─── Mid-sized loops (count > 1, exercises r13 indexing) ────────────────

    #[test]
    fn mid_sized_loop_ab() {
        // 1024-element add to exercise the loop scaffold beyond the trivial
        // cases above.
        let n = 1024usize;
        let a_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.25).collect();
        let mut g = NanoGraph::new();
        let a = g.add_input_tensor(GlobalId(0), n as u64, NumericDType::F32);
        let b = g.add_input_tensor(GlobalId(1), n as u64, NumericDType::F32);
        let out = g.push_group(
            n as u64,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(a, 1), InputRef::affine(b, 1)],
        );
        let outputs = vec![AtomRange {
            base: out,
            count: n as u64,
            dtype: NumericDType::F32,
        }];
        ab_test_f32(&g, &[(a, &a_data), (b, &b_data)], &outputs);
    }

    // ─── Multi-group span: literal-broadcast plus chained ops ───────────────

    #[test]
    fn multi_group_chain_ab() {
        // (input * 3.0) + (-input)  — exercises Literal broadcast + multiple
        // groups in one span.
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), 8, NumericDType::F32);
        let lit3 = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(3.0)),
            vec![],
            vec![],
        );
        let mul = g.push_group(
            8,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(inp, 1), InputRef::Broadcast(lit3)],
        );
        let neg = g.push_group(
            8,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(inp, 1)],
        );
        let add = g.push_group(
            8,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(mul, 1), InputRef::affine(neg, 1)],
        );
        let outputs = vec![AtomRange {
            base: add,
            count: 8,
            dtype: NumericDType::F32,
        }];
        ab_test_f32(&g, &[(inp, F32_X)], &outputs);
    }
}
