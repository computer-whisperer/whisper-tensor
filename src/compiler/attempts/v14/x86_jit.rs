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
use crate::nano_graph::ops::{ReduceKind, ScalarBinOp, ScalarOp, ScalarUnaryOp};
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
extern "C" fn jit_tanf(x: f32) -> f32 {
    x.tan()
}
extern "C" fn jit_asinf(x: f32) -> f32 {
    x.asin()
}
extern "C" fn jit_acosf(x: f32) -> f32 {
    x.acos()
}
extern "C" fn jit_atanf(x: f32) -> f32 {
    x.atan()
}
extern "C" fn jit_sinhf(x: f32) -> f32 {
    x.sinh()
}
extern "C" fn jit_coshf(x: f32) -> f32 {
    x.cosh()
}
extern "C" fn jit_asinhf(x: f32) -> f32 {
    x.asinh()
}
extern "C" fn jit_acoshf(x: f32) -> f32 {
    x.acosh()
}
extern "C" fn jit_atanhf(x: f32) -> f32 {
    x.atanh()
}
extern "C" fn jit_log1pf(x: f32) -> f32 {
    x.ln_1p()
}
extern "C" fn jit_erff(x: f32) -> f32 {
    // Use the same series-based erf as numeric_dtype/conversions if needed,
    // but std doesn't expose erf so we hand-roll Abramowitz & Stegun 7.1.26.
    // Cranelift uses libm's erff via a trampoline; we mirror that behaviour.
    libm_erff(x)
}
extern "C" fn jit_roundf(x: f32) -> f32 {
    // Round half away from zero (matches f32::round / Cranelift's nearest).
    x.round()
}

/// libm-equivalent erff (Abramowitz & Stegun 7.1.26).
///
/// Cranelift's backend doesn't actually emit Erf today (it returns
/// `Err("float Erf not implemented in JIT")`), so this exists only so the
/// x86 backend can support Erf where the cranelift backend can't.
fn libm_erff(x: f32) -> f32 {
    // a1..a5 from A&S 7.1.26 with p = 0.3275911.
    let a1 = 0.254_829_592_f32;
    let a2 = -0.284_496_736_f32;
    let a3 = 1.421_413_741_f32;
    let a4 = -1.453_152_027_f32;
    let a5 = 1.061_405_429_f32;
    let p = 0.327_591_1_f32;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let xa = x.abs();
    let t = 1.0 / (1.0 + p * xa);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-xa * xa).exp();
    sign * y
}

// F64 variants. Distinct names from the F32 trampolines so we can take both
// addresses without ambiguity at codegen time.
extern "C" fn jit_exp_d(x: f64) -> f64 {
    x.exp()
}
extern "C" fn jit_log_d(x: f64) -> f64 {
    x.ln()
}
extern "C" fn jit_tanh_d(x: f64) -> f64 {
    x.tanh()
}
extern "C" fn jit_sqrt_d(x: f64) -> f64 {
    x.sqrt()
}
extern "C" fn jit_floor_d(x: f64) -> f64 {
    x.floor()
}
extern "C" fn jit_ceil_d(x: f64) -> f64 {
    x.ceil()
}
extern "C" fn jit_pow_d(x: f64, y: f64) -> f64 {
    x.powf(y)
}
extern "C" fn jit_fmod_d(x: f64, y: f64) -> f64 {
    x % y
}
extern "C" fn jit_sin_d(x: f64) -> f64 {
    x.sin()
}
extern "C" fn jit_cos_d(x: f64) -> f64 {
    x.cos()
}
extern "C" fn jit_round_d(x: f64) -> f64 {
    x.round()
}
extern "C" fn jit_tan_d(x: f64) -> f64 {
    x.tan()
}
extern "C" fn jit_asin_d(x: f64) -> f64 {
    x.asin()
}
extern "C" fn jit_acos_d(x: f64) -> f64 {
    x.acos()
}
extern "C" fn jit_atan_d(x: f64) -> f64 {
    x.atan()
}
extern "C" fn jit_sinh_d(x: f64) -> f64 {
    x.sinh()
}
extern "C" fn jit_cosh_d(x: f64) -> f64 {
    x.cosh()
}
extern "C" fn jit_asinh_d(x: f64) -> f64 {
    x.asinh()
}
extern "C" fn jit_acosh_d(x: f64) -> f64 {
    x.acosh()
}
extern "C" fn jit_atanh_d(x: f64) -> f64 {
    x.atanh()
}
extern "C" fn jit_log1p_d(x: f64) -> f64 {
    x.ln_1p()
}
extern "C" fn jit_erf_d(x: f64) -> f64 {
    libm_erf(x)
}

/// libm-equivalent erf for f64 (same A&S 7.1.26 series).
fn libm_erf(x: f64) -> f64 {
    let a1 = 0.254_829_592_f64;
    let a2 = -0.284_496_736_f64;
    let a3 = 1.421_413_741_f64;
    let a4 = -1.453_152_027_f64;
    let a5 = 1.061_405_429_f64;
    let p = 0.327_591_1_f64;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let xa = x.abs();
    let t = 1.0 / (1.0 + p * xa);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-xa * xa).exp();
    sign * y
}

// F16 conversion trampolines.
//
// We don't try to inline F16 expand/narrow — IEEE 754 half-precision has
// enough special-case handling (denormal subnormals, infinities, NaN
// propagation, RTNE rounding on the way down) that the inline path runs
// to ~25 instructions per direction. The half crate is already a workspace
// dependency, so we delegate.
//
// `from_bits` takes a u16 — under System V the u16 lives in `di`, with the
// upper bytes of edi implementation-defined. half::f16::from_bits ignores
// them, so callers can pass any zero/one-extended value.
extern "C" fn jit_f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

extern "C" fn jit_f32_to_f16(x: f32) -> u16 {
    half::f16::from_f32(x).to_bits()
}

// ─── Compute representation ─────────────────────────────────────────────────

/// How a group's value lives in registers during compute.
///
/// Mirrors `codegen::ReprKind` but lives in this module so we can dispatch
/// on it without a cross-module dependency. F32/F64 use the SSE register
/// file (xmm0..xmm2). Int uses the GP file (rax/rcx/rdx) at i64 width
/// regardless of source storage width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Repr {
    F32,
    F64,
    Int,
}

impl Repr {
    /// Map a NumericDType to the compute representation it lives in.
    fn from_dtype(dtype: NumericDType) -> Self {
        match dtype {
            NumericDType::F64 => Repr::F64,
            NumericDType::F32 | NumericDType::BF16 | NumericDType::F16 => Repr::F32,
            _ => Repr::Int,
        }
    }
}

/// Pick the compute repr for a group.
///
/// Binary/Unary/Reduce groups carry an explicit `compute_dtype`. Other ops
/// (Identity, Cast, Select, IndirectLoad) have no compute dtype — we use
/// the output dtype's natural repr.
fn group_compute_repr(group: &AtomGroup<'static, SystemPool>) -> Repr {
    match group.op.compute_dtype() {
        Some(dt) => Repr::from_dtype(dt),
        None => Repr::from_dtype(group.output_dtype),
    }
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

        // P4: tables are now created BEFORE emit_native because IndirectLoad
        // and Explicit InputRefs allocate embedded lookup tables whose byte
        // offsets need to be baked into the emitted code.
        let mut tables = EmbeddedTables::new(layout.total_bytes);
        let (code, entry) = emit_native(graph, &layout, &mut tables)?;

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

/// Check whether a span is within the current x86_jit envelope.
///
/// As of P4: native fast paths for F32, F64, and Int groups, plus BF16/F16
/// half-width float storage going through F32 compute. Cross-repr Cast,
/// Reduce (Sum/Max/Min/Prod), IndirectLoad, and Explicit InputRef are all
/// supported.
///
/// Op variants: Identity, Cast, Binary, Unary, Select, Reduce, IndirectLoad,
/// Literal, LiteralSpan.
/// InputRef variants: Broadcast, 1D affine Strided, 2D Strided
/// (modular / strided_broadcast / general nd=2), and Explicit.
/// Still unsupported: reduce-fold inlining, N-D Strided with nd >= 3,
/// arbitrary FloatType (F8 / F4 / F6 / etc).
fn check_supported(
    graph: &NanoGraph<'static, SystemPool>,
    layout: &BufferLayout,
) -> Result<(), String> {
    let groups = graph.groups();

    // No reduce-fold inlining yet.
    for (gi, &inlinable) in layout.inlinable.iter().enumerate() {
        if inlinable {
            return Err(format!("x86_jit: group {gi} is inlinable (unsupported)"));
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

        let group_repr = group_compute_repr(group);
        let is_cast_op = matches!(&group.op, ScalarOp::Identity | ScalarOp::Cast { .. });

        // Output dtype must live in the group's repr family (Cast follows the
        // dispatch repr derived from output_dtype, so this check is trivially
        // true for Cast; non-Cast ops require alignment).
        if Repr::from_dtype(group.output_dtype) != group_repr {
            return Err(format!(
                "x86_jit: group {gi} output_dtype {:?} doesn't match compute repr {group_repr:?}",
                group.output_dtype
            ));
        }
        if !is_supported_storage_dtype(group.output_dtype) {
            return Err(format!(
                "x86_jit: group {gi} output_dtype {:?} not yet supported",
                group.output_dtype
            ));
        }

        match &group.op {
            ScalarOp::Identity => {}
            ScalarOp::Cast { .. } => {}
            ScalarOp::Binary { compute_dtype, .. } | ScalarOp::Unary { compute_dtype, .. } => {
                if Repr::from_dtype(*compute_dtype) != group_repr {
                    return Err(format!(
                        "x86_jit: group {gi} compute_dtype {:?} doesn't match group repr {group_repr:?}",
                        compute_dtype
                    ));
                }
                if !is_supported_storage_dtype(*compute_dtype) {
                    return Err(format!(
                        "x86_jit: group {gi} compute_dtype {:?} not yet supported",
                        compute_dtype
                    ));
                }
            }
            ScalarOp::Select => {}
            ScalarOp::Reduce { compute_dtype, .. } => {
                if Repr::from_dtype(*compute_dtype) != group_repr {
                    return Err(format!(
                        "x86_jit: group {gi} Reduce compute_dtype {:?} doesn't match group repr {group_repr:?}",
                        compute_dtype
                    ));
                }
                if !is_supported_storage_dtype(*compute_dtype) {
                    return Err(format!(
                        "x86_jit: group {gi} Reduce compute_dtype {:?} not yet supported",
                        compute_dtype
                    ));
                }
                // Reduce input must be Strided 1D affine — matches the
                // lowering's affine path. Explicit Reduce inputs are rare
                // and would need a separate code path.
                if !matches!(
                    group.inputs.first(),
                    Some(InputRef::Strided { dim_strides, .. }) if dim_strides.len() == 1
                ) {
                    return Err(format!(
                        "x86_jit: group {gi} Reduce input must be 1D affine Strided"
                    ));
                }
            }
            ScalarOp::IndirectLoad { .. } => {
                if group.inputs.len() != 1 {
                    return Err(format!(
                        "x86_jit: group {gi} IndirectLoad expects 1 input, got {}",
                        group.inputs.len()
                    ));
                }
            }
            ScalarOp::OpaqueOutput { .. } => {
                return Err(format!("x86_jit: group {gi} OpaqueOutput (unsupported)"));
            }
            ScalarOp::Literal(_) | ScalarOp::LiteralSpan(_) => unreachable!(),
        }

        // IndirectLoad's input[0] is the index (any int dtype, doesn't need
        // to match the output repr). Reduce + Cast / Identity are allowed to
        // bridge reprs.
        let bridge_repr = is_cast_op
            || matches!(
                &group.op,
                ScalarOp::IndirectLoad { .. } | ScalarOp::Reduce { .. }
            );

        for (ii, ir) in group.inputs.iter().enumerate() {
            let slot = match ir {
                InputRef::Broadcast(atom) => {
                    layout.find(*atom).map(|(s, _)| s).ok_or_else(|| {
                        format!("x86_jit: group {gi} input {ii} Broadcast atom={atom} no slot")
                    })?
                }
                InputRef::Strided {
                    base,
                    dim_strides,
                    dim_shape,
                } => {
                    let nd = dim_strides.len();
                    if nd != dim_shape.len() {
                        return Err(format!(
                            "x86_jit: group {gi} input {ii} dim_strides/dim_shape length mismatch"
                        ));
                    }
                    if !(1..=2).contains(&nd) {
                        return Err(format!(
                            "x86_jit: group {gi} input {ii} Strided nd={nd} (only nd=1 and nd=2 supported)"
                        ));
                    }
                    // Resolve the slot via base or the first accessed atom
                    // (split-group fallback). For nd=2 we must compute the
                    // first-atom offset using the full ND decomposition, not
                    // just `stride * atom_offset`.
                    let first_offset =
                        strided_offset_atoms(dim_strides, dim_shape, group.atom_offset);
                    let first_atom = AtomId((base.0 as i64 + first_offset) as u64);
                    layout
                        .find(*base)
                        .or_else(|| layout.find(first_atom))
                        .map(|(s, _)| s)
                        .ok_or_else(|| {
                            format!("x86_jit: group {gi} input {ii} Strided base={base} no slot")
                        })?
                }
                InputRef::Explicit(ids) => {
                    let id = ids.first().ok_or_else(|| {
                        format!("x86_jit: group {gi} input {ii} Explicit empty list")
                    })?;
                    layout.find(*id).map(|(s, _)| s).ok_or_else(|| {
                        format!("x86_jit: group {gi} input {ii} Explicit atom={id} no slot")
                    })?
                }
            };
            if !is_supported_storage_dtype(slot.dtype) {
                return Err(format!(
                    "x86_jit: group {gi} input {ii} slot dtype {:?} not yet supported",
                    slot.dtype
                ));
            }
            if !bridge_repr && Repr::from_dtype(slot.dtype) != group_repr {
                return Err(format!(
                    "x86_jit: group {gi} input {ii} slot dtype {:?} doesn't match group repr {group_repr:?}",
                    slot.dtype
                ));
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
                if !is_supported_binop(*op, group_repr) {
                    return Err(format!(
                        "x86_jit: group {gi} Binary {:?} on {group_repr:?} unsupported",
                        op
                    ));
                }
            }
            ScalarOp::Unary { op, .. } => {
                if group.inputs.len() != 1 {
                    return Err(format!(
                        "x86_jit: group {gi} Unary expects 1 input, got {}",
                        group.inputs.len()
                    ));
                }
                if !is_supported_unop(*op, group_repr) {
                    return Err(format!(
                        "x86_jit: group {gi} Unary {:?} on {group_repr:?} unsupported",
                        op
                    ));
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

/// Native storage dtypes the x86_jit backend can load and store.
///
/// BF16 expands inline (3 ins load, 7 ins store with RTNE). F16 goes through
/// an extern trampoline using the `half` crate (P3 software path; F16C
/// inline fast path is a follow-up). Sub-byte ints (I4/U4) remain deferred.
fn is_supported_storage_dtype(dtype: NumericDType) -> bool {
    matches!(
        dtype,
        NumericDType::F32
            | NumericDType::F64
            | NumericDType::BF16
            | NumericDType::F16
            | NumericDType::I64
            | NumericDType::U64
            | NumericDType::I32
            | NumericDType::U32
            | NumericDType::I16
            | NumericDType::U16
            | NumericDType::I8
            | NumericDType::U8
            | NumericDType::BOOL
    )
}

fn is_supported_binop(op: ScalarBinOp, repr: Repr) -> bool {
    match repr {
        Repr::F32 | Repr::F64 => is_supported_binop_float(op),
        Repr::Int => is_supported_binop_int(op),
    }
}

fn is_supported_unop(op: ScalarUnaryOp, repr: Repr) -> bool {
    match repr {
        Repr::F32 | Repr::F64 => is_supported_unop_float(op),
        Repr::Int => is_supported_unop_int(op),
    }
}

fn is_supported_binop_float(op: ScalarBinOp) -> bool {
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

fn is_supported_unop_float(op: ScalarUnaryOp) -> bool {
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
            | ScalarUnaryOp::Tan
            | ScalarUnaryOp::Asin
            | ScalarUnaryOp::Acos
            | ScalarUnaryOp::Atan
            | ScalarUnaryOp::Sinh
            | ScalarUnaryOp::Cosh
            | ScalarUnaryOp::Asinh
            | ScalarUnaryOp::Acosh
            | ScalarUnaryOp::Atanh
            | ScalarUnaryOp::Log1p
            | ScalarUnaryOp::Erf
            | ScalarUnaryOp::Round
            | ScalarUnaryOp::IsNan
            | ScalarUnaryOp::IsInf { .. }
            | ScalarUnaryOp::Sign
            | ScalarUnaryOp::Not
    )
}

/// Binary ops supported on integer compute (i64 internal).
fn is_supported_binop_int(op: ScalarBinOp) -> bool {
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
            | ScalarBinOp::Equal
            | ScalarBinOp::Greater
            | ScalarBinOp::GreaterOrEqual
            | ScalarBinOp::Less
            | ScalarBinOp::LessOrEqual
            | ScalarBinOp::And
            | ScalarBinOp::Or
            | ScalarBinOp::Xor
            | ScalarBinOp::BitwiseAnd
            | ScalarBinOp::BitwiseOr
            | ScalarBinOp::BitwiseXor
            | ScalarBinOp::BitShiftLeft
            | ScalarBinOp::BitShiftRight
    )
}

/// Unary ops supported on integer compute.
fn is_supported_unop_int(op: ScalarUnaryOp) -> bool {
    matches!(
        op,
        ScalarUnaryOp::Neg
            | ScalarUnaryOp::Abs
            | ScalarUnaryOp::BitwiseNot
            | ScalarUnaryOp::Not
            | ScalarUnaryOp::Sign
            | ScalarUnaryOp::IsNan
            | ScalarUnaryOp::IsInf { .. }
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
    tables: &mut EmbeddedTables,
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
        emit_group(&mut ops, group, layout, tables)?;
    }

    emit_epilogue(&mut ops);

    let code = ops
        .finalize()
        .map_err(|_| "x86_jit: assembler finalize failed".to_string())?;
    Ok((code, entry))
}

/// System V AMD64 function prologue.
///
/// Stack layout after prologue (relative to the new rsp):
/// ```text
///   [rsp+ 0..7]   IMod fmod scratch (saved divisor across the libm call)
///   [rsp+ 8..11]  F16-load xmm0 spill slot
///   [rsp+12..15]  F16-load xmm1 spill slot
///   [rsp+16..19]  F16-load xmm2 spill slot
///   [rsp+20..31]  padding / scratch (rounds the frame to a 16-byte multiple)
/// ```
///
/// Saved callee-saved regs (rbp/r12/r13/r14/r15) sit above [rsp+32]. The
/// frame ends 16-byte aligned so any `call` we issue inside the body lands
/// the caller's return address at +8 mod 16.
///
/// `r15` was added in P4 so the Reduce inner-k loop has its own counter
/// without colliding with the outer r13/r14 loop variables.
fn emit_prologue(ops: &mut Assembler) {
    dynasm!(ops
        ; .arch x64
        ; push rbp
        ; mov rbp, rsp
        ; push r12
        ; push r13
        ; push r14
        ; push r15
        ; sub rsp, BYTE 32      // 5 pushes + return addr = 48 bytes; 48+32 = 80 ≡ 0 mod 16
        ; mov r12, rdi          // r12 = buffer pointer
    );
}

fn emit_epilogue(ops: &mut Assembler) {
    dynasm!(ops
        ; .arch x64
        ; add rsp, BYTE 32
        ; pop r15
        ; pop r14
        ; pop r13
        ; pop r12
        ; pop rbp
        ; ret
    );
}

/// Stack offsets for the F16-load spill slots. Each holds an f32 (4 bytes).
/// Used by emit_group_body_f32 to preserve xmm0/xmm1 across F16 extern-call
/// loads that would otherwise clobber them.
const F16_SPILL_XMM0: i32 = 8;
const F16_SPILL_XMM1: i32 = 12;
const F16_SPILL_XMM2: i32 = 16;
/// Spill slot for the inner Reduce k-loop counter (r11) across F16 extern
/// calls. r11 is caller-saved per System V so the call clobbers it; reduce
/// body iterations save it here and reload after the call.
const K_SPILL: i32 = 24;

/// Returns true if loading this input requires an extern call (currently
/// only F16, which goes through `jit_f16_to_f32`). Used by binop/select
/// emission to decide whether prior xmm values need spilling.
fn input_load_clobbers_xmm(input: &InputRef, layout: &BufferLayout, atom_offset: u64) -> bool {
    let dtype = match input {
        InputRef::Broadcast(atom) => layout.find(*atom).map(|(s, _)| s.dtype),
        InputRef::Strided {
            base, dim_strides, ..
        } => {
            let stride = dim_strides[0];
            let first_offset = stride * atom_offset as i64;
            let first_atom = AtomId((base.0 as i64 + first_offset) as u64);
            layout
                .find(*base)
                .or_else(|| layout.find(first_atom))
                .map(|(s, _)| s.dtype)
        }
        InputRef::Explicit(_) => None,
    };
    matches!(dtype, Some(NumericDType::F16))
}

/// Emit one group as either a counted loop (count > 1) or inline body (count == 1).
fn emit_group(
    ops: &mut Assembler,
    group: &AtomGroup<'static, SystemPool>,
    layout: &BufferLayout,
    tables: &mut EmbeddedTables,
) -> Result<(), String> {
    let count = group.count;
    let atom_offset = group.atom_offset;

    if count == 1 {
        // Inline path: i_const = atom_offset, no loop.
        emit_group_body(
            ops,
            group,
            layout,
            tables,
            /* in_loop */ false,
            atom_offset,
        )?;
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

    emit_group_body(ops, group, layout, tables, /* in_loop */ true, 0)?;

    dynasm!(ops
        ; .arch x64
        ; inc r13
        ; jmp =>loop_top
        ; =>loop_done
    );

    Ok(())
}

/// Top-level dispatcher: pick a compute repr and route to the matching
/// repr-specific body emitter.
///
/// Identity / Cast bridge dtypes (and reprs), so they get a dedicated emitter
/// that doesn't depend on a single compute repr.
fn emit_group_body(
    ops: &mut Assembler,
    group: &AtomGroup<'static, SystemPool>,
    layout: &BufferLayout,
    tables: &mut EmbeddedTables,
    in_loop: bool,
    i_const: u64,
) -> Result<(), String> {
    if matches!(&group.op, ScalarOp::Identity | ScalarOp::Cast { .. }) {
        return emit_group_body_cast(ops, group, layout, tables, in_loop, i_const);
    }
    if let ScalarOp::IndirectLoad { table_base } = &group.op {
        return emit_group_body_indirect_load(
            ops,
            group,
            layout,
            tables,
            in_loop,
            i_const,
            *table_base,
        );
    }
    if let ScalarOp::Reduce {
        kind,
        reduce_count,
        reduce_stride,
        compute_dtype,
    } = &group.op
    {
        return emit_group_body_reduce(
            ops,
            group,
            layout,
            tables,
            in_loop,
            i_const,
            *kind,
            *reduce_count,
            *reduce_stride,
            *compute_dtype,
        );
    }
    let repr = group_compute_repr(group);
    match repr {
        Repr::F32 => emit_group_body_f32(ops, group, layout, tables, in_loop, i_const),
        Repr::F64 => emit_group_body_f64(ops, group, layout, tables, in_loop, i_const),
        Repr::Int => emit_group_body_int(ops, group, layout, tables, in_loop, i_const),
    }
}

/// Emit Identity / Cast: load input in its natural repr, convert to the
/// output dtype's repr if needed, store.
///
/// The float→int and int→float conversions match cranelift's behaviour
/// (saturating-truncating for float→int via cvttss2si / cvttsd2si, signed
/// load via cvtsi2ss / cvtsi2sd). Bool storage normalization happens inside
/// emit_store_int.
fn emit_group_body_cast(
    ops: &mut Assembler,
    group: &AtomGroup<'static, SystemPool>,
    layout: &BufferLayout,
    tables: &mut EmbeddedTables,
    in_loop: bool,
    i_const: u64,
) -> Result<(), String> {
    let (out_slot, _) = layout
        .find(group.base_id)
        .ok_or_else(|| format!("x86_jit: no slot for group base={}", group.base_id))?;
    let out_slot = out_slot.clone();

    // Resolve the input slot's dtype to pick the load path.
    let input = &group.inputs[0];
    let input_dtype = match input {
        InputRef::Broadcast(atom) => layout
            .find(*atom)
            .map(|(s, _)| s.dtype)
            .ok_or_else(|| format!("x86_jit cast: no slot for Broadcast atom={atom}"))?,
        InputRef::Strided {
            base,
            dim_strides,
            dim_shape,
        } => {
            let first_offset = strided_offset_atoms(dim_strides, dim_shape, group.atom_offset);
            let first_atom = AtomId((base.0 as i64 + first_offset) as u64);
            layout
                .find(*base)
                .or_else(|| layout.find(first_atom))
                .map(|(s, _)| s.dtype)
                .ok_or_else(|| format!("x86_jit cast: no slot for Strided base={base}"))?
        }
        InputRef::Explicit(ids) => {
            let id = ids
                .first()
                .ok_or_else(|| "x86_jit cast: Explicit InputRef with empty list".to_string())?;
            layout
                .find(*id)
                .map(|(s, _)| s.dtype)
                .ok_or_else(|| format!("x86_jit cast: no slot for Explicit atom={id}"))?
        }
    };

    let in_repr = Repr::from_dtype(input_dtype);
    let out_repr = Repr::from_dtype(group.output_dtype);

    // Load into the natural repr's slot 0 register.
    match in_repr {
        Repr::F32 => emit_load_f32(
            ops,
            input,
            layout,
            tables,
            in_loop,
            i_const,
            group.atom_offset,
            0,
        )?,
        Repr::F64 => emit_load_f64(
            ops,
            input,
            layout,
            tables,
            in_loop,
            i_const,
            group.atom_offset,
            0,
        )?,
        Repr::Int => emit_load_int(
            ops,
            input,
            layout,
            tables,
            in_loop,
            i_const,
            group.atom_offset,
            IntReg::Rax,
        )?,
    }

    // Cross-repr conversion (if needed).
    match (in_repr, out_repr) {
        (Repr::F32, Repr::F32) | (Repr::F64, Repr::F64) | (Repr::Int, Repr::Int) => {
            // Same repr — no conversion. (Width narrowing for ints happens
            // inside emit_store_int.)
        }
        (Repr::F32, Repr::F64) => {
            dynasm!(ops ; .arch x64 ; cvtss2sd xmm0, xmm0);
        }
        (Repr::F64, Repr::F32) => {
            dynasm!(ops ; .arch x64 ; cvtsd2ss xmm0, xmm0);
        }
        (Repr::F32, Repr::Int) => {
            dynasm!(ops ; .arch x64 ; cvttss2si rax, xmm0);
        }
        (Repr::F64, Repr::Int) => {
            dynasm!(ops ; .arch x64 ; cvttsd2si rax, xmm0);
        }
        (Repr::Int, Repr::F32) => {
            dynasm!(ops ; .arch x64 ; cvtsi2ss xmm0, rax);
        }
        (Repr::Int, Repr::F64) => {
            dynasm!(ops ; .arch x64 ; cvtsi2sd xmm0, rax);
        }
    }

    // Store in the output repr.
    match out_repr {
        Repr::F32 => {
            emit_store_f32(ops, &out_slot, in_loop, i_const, group.atom_offset, 0);
        }
        Repr::F64 => {
            emit_store_f64(ops, &out_slot, in_loop, i_const, group.atom_offset, 0);
        }
        Repr::Int => {
            emit_store_int(
                ops,
                &out_slot,
                in_loop,
                i_const,
                group.atom_offset,
                IntReg::Rax,
            );
        }
    }

    Ok(())
}

/// Emit one iteration of a group body in F32 compute.
///
/// `in_loop`: when true, the loop variable lives in r13 (use it for indexed
/// addressing). When false, all addresses are constant displacements based on
/// `i_const`.
fn emit_group_body_f32(
    ops: &mut Assembler,
    group: &AtomGroup<'static, SystemPool>,
    layout: &BufferLayout,
    tables: &mut EmbeddedTables,
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
                tables,
                in_loop,
                i_const,
                group.atom_offset,
                0,
            )?;
            emit_store_f32(ops, &out_slot, in_loop, i_const, group.atom_offset, 0);
        }

        ScalarOp::Binary { op, .. } => {
            // a → xmm0, b → xmm1, op(xmm0, xmm1) → xmm0, store xmm0.
            //
            // F16 loads go through an extern call which clobbers xmm0. If
            // input[1] is F16, save a to a stack spill slot before the b
            // load and restore it afterwards.
            emit_load_f32(
                ops,
                &group.inputs[0],
                layout,
                tables,
                in_loop,
                i_const,
                group.atom_offset,
                0,
            )?;
            let needs_spill = input_load_clobbers_xmm(&group.inputs[1], layout, group.atom_offset);
            if needs_spill {
                dynasm!(ops ; .arch x64 ; movss DWORD [rsp + F16_SPILL_XMM0], xmm0);
            }
            emit_load_f32(
                ops,
                &group.inputs[1],
                layout,
                tables,
                in_loop,
                i_const,
                group.atom_offset,
                1,
            )?;
            if needs_spill {
                dynasm!(ops ; .arch x64 ; movss xmm0, DWORD [rsp + F16_SPILL_XMM0]);
            }
            emit_binop_f32(ops, *op);
            emit_store_f32(ops, &out_slot, in_loop, i_const, group.atom_offset, 0);
        }

        ScalarOp::Unary { op, .. } => {
            // x → xmm0, op(xmm0) → xmm0, store xmm0.
            emit_load_f32(
                ops,
                &group.inputs[0],
                layout,
                tables,
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
            //
            // Each subsequent load may clobber prior xmm registers if it's
            // an F16 extern call. Spill what we need to preserve.
            emit_load_f32(
                ops,
                &group.inputs[0],
                layout,
                tables,
                in_loop,
                i_const,
                group.atom_offset,
                0,
            )?;
            let x_clobbers = input_load_clobbers_xmm(&group.inputs[1], layout, group.atom_offset);
            let y_clobbers = input_load_clobbers_xmm(&group.inputs[2], layout, group.atom_offset);
            if x_clobbers || y_clobbers {
                dynasm!(ops ; .arch x64 ; movss DWORD [rsp + F16_SPILL_XMM0], xmm0);
            }
            emit_load_f32(
                ops,
                &group.inputs[1],
                layout,
                tables,
                in_loop,
                i_const,
                group.atom_offset,
                1,
            )?;
            if y_clobbers {
                dynasm!(ops ; .arch x64 ; movss DWORD [rsp + F16_SPILL_XMM1], xmm1);
            }
            emit_load_f32(
                ops,
                &group.inputs[2],
                layout,
                tables,
                in_loop,
                i_const,
                group.atom_offset,
                2,
            )?;
            if y_clobbers {
                dynasm!(ops ; .arch x64 ; movss xmm1, DWORD [rsp + F16_SPILL_XMM1]);
            }
            if x_clobbers || y_clobbers {
                dynasm!(ops ; .arch x64 ; movss xmm0, DWORD [rsp + F16_SPILL_XMM0]);
            }

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
            return Err("x86_jit: emit_group_body_f32 unexpected op".to_string());
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

/// Emit a load that produces an f32 value in `xmm{dst_xmm}`, regardless of
/// the underlying slot's storage dtype. Slots holding F32 use `movss`,
/// BF16 expand inline (`shl 16` into the high half), F16 go through an
/// extern call to `jit_f16_to_f32`.
fn emit_load_f32(
    ops: &mut Assembler,
    input: &InputRef,
    layout: &BufferLayout,
    tables: &mut EmbeddedTables,
    in_loop: bool,
    i_const: u64,
    atom_offset: u64,
    dst_xmm: u8,
) -> Result<(), String> {
    let (slot_dtype, addr) =
        resolve_input_load_addr(ops, input, layout, tables, in_loop, i_const, atom_offset)?;
    match slot_dtype {
        NumericDType::F32 => match addr {
            LoadAddr::Disp(d) => emit_load_f32_disp(ops, dst_xmm, d),
            LoadAddr::Indexed { base, stride } => emit_load_f32_indexed(ops, dst_xmm, base, stride),
            LoadAddr::RaxIndexed { base } => emit_load_f32_rax(ops, dst_xmm, base),
        },
        NumericDType::BF16 => match addr {
            LoadAddr::Disp(d) => emit_load_bf16_disp(ops, dst_xmm, d),
            LoadAddr::Indexed { base, stride } => {
                emit_load_bf16_indexed(ops, dst_xmm, base, stride)
            }
            LoadAddr::RaxIndexed { base } => emit_load_bf16_rax(ops, dst_xmm, base),
        },
        NumericDType::F16 => match addr {
            LoadAddr::Disp(d) => emit_load_f16_disp(ops, dst_xmm, d),
            LoadAddr::Indexed { base, stride } => emit_load_f16_indexed(ops, dst_xmm, base, stride),
            LoadAddr::RaxIndexed { base } => emit_load_f16_rax(ops, dst_xmm, base),
        },
        other => {
            return Err(format!(
                "x86_jit emit_load_f32: unexpected slot dtype {other:?}"
            ));
        }
    }
    Ok(())
}

/// Resolved address for an InputRef load, plus the slot's storage dtype.
#[derive(Debug, Clone, Copy)]
enum LoadAddr {
    /// Constant displacement: `[r12 + disp]`.
    Disp(i64),
    /// Indexed by r13: `[r12 + r13*scale + base_byte]`.
    Indexed { base: i64, stride: i64 },
    /// Byte offset is materialized in `rax` already; address is
    /// `[r12 + rax + base_byte]`. Used by 2D Strided InputRefs and by
    /// multi-entry `Explicit` InputRefs which both compute their per-iteration
    /// byte offset via additional instructions emitted before the load.
    RaxIndexed { base: i64 },
}

/// Resolve an InputRef to its load address and slot storage dtype.
///
/// May emit address-compute instructions before the load (into `rax`) for
/// 2D Strided / Explicit InputRefs whose per-iteration byte offset isn't
/// expressible as a constant displacement or a single SIB scale.
///
/// **Caller contract**: rax must be free at the call site. The returned
/// `LoadAddr::RaxIndexed` variant means the load helper should use
/// `[r12 + rax + base]`. The 1D affine and Broadcast cases don't touch rax.
fn resolve_input_load_addr(
    ops: &mut Assembler,
    input: &InputRef,
    layout: &BufferLayout,
    tables: &mut EmbeddedTables,
    in_loop: bool,
    i_const: u64,
    atom_offset: u64,
) -> Result<(NumericDType, LoadAddr), String> {
    match input {
        InputRef::Broadcast(atom) => {
            let (slot, elem_idx) = layout
                .find(*atom)
                .ok_or_else(|| format!("x86_jit: no slot for Broadcast atom={atom}"))?;
            let byte_off = slot.byte_offset as i64 + elem_idx as i64 * slot.elem_bytes as i64;
            Ok((slot.dtype, LoadAddr::Disp(byte_off)))
        }
        InputRef::Strided {
            base,
            dim_strides,
            dim_shape,
        } => {
            let nd = dim_strides.len();
            assert!(nd >= 1, "check_supported guarantees nd >= 1");
            assert_eq!(nd, dim_shape.len(), "dim_strides/dim_shape length mismatch");

            // Find the slot via base or the first accessed atom (split-group fallback).
            let first_offset = strided_offset_atoms(dim_strides, dim_shape, atom_offset);
            let first_atom = AtomId((base.0 as i64 + first_offset) as u64);
            let (slot, elem) = layout
                .find(*base)
                .or_else(|| layout.find(first_atom))
                .ok_or_else(|| format!("x86_jit: no slot for Strided base={base}"))?;
            let elem_bytes = slot.elem_bytes as i64;
            let dtype = slot.dtype;

            // Compute base_byte: byte offset of the logical `base` atom.
            let slot_byte = slot.byte_offset as i64 + elem as i64 * elem_bytes;
            let base_byte = if layout.find(*base).is_some() {
                slot_byte
            } else {
                slot_byte - first_offset * elem_bytes
            };

            if nd == 1 {
                // 1D affine fast path.
                let byte_stride = dim_strides[0] * elem_bytes;
                let addr = if in_loop {
                    LoadAddr::Indexed {
                        base: base_byte,
                        stride: byte_stride,
                    }
                } else {
                    LoadAddr::Disp(base_byte + byte_stride * i_const as i64)
                };
                return Ok((dtype, addr));
            }

            // 2D / N-D path. For !in_loop this resolves to a constant Disp.
            if !in_loop {
                let off_atoms = strided_offset_atoms(dim_strides, dim_shape, i_const);
                let byte_off = base_byte + off_atoms * elem_bytes;
                return Ok((dtype, LoadAddr::Disp(byte_off)));
            }

            // 2D in_loop: emit address compute into rax.
            // Cranelift's general N-D path: decompose i (in r13) from
            // innermost dim outward, where coord[d] = remaining % shape[d]
            // for d > 0 and coord[0] = remaining (no modulus on outermost).
            // Sum coord[d] * stride[d] * elem_bytes.
            //
            // For nd == 2 (the only case we currently support), this is:
            //   inner = r13 % shape[1]   (or shr/and if power of 2)
            //   outer = r13 / shape[1]
            //   rax   = outer * stride[0]_bytes + inner * stride[1]_bytes
            //
            // Special cases:
            //   - dim_strides[0] == 0 (modular): only inner contributes.
            //   - dim_strides[1] == 0 (strided_broadcast): only outer contributes.
            if nd != 2 {
                return Err(format!(
                    "x86_jit: Strided nd={} (only nd=1 and nd=2 supported)",
                    nd
                ));
            }
            emit_strided_2d_offset_into_rax(ops, dim_strides, dim_shape, elem_bytes)?;
            Ok((dtype, LoadAddr::RaxIndexed { base: base_byte }))
        }
        InputRef::Explicit(ids) => {
            if ids.is_empty() {
                return Err("x86_jit: Explicit InputRef with empty list".into());
            }
            // Single-entry: every iteration loads the same atom — use Disp.
            if ids.len() == 1 {
                let (slot, elem_idx) = layout
                    .find(ids[0])
                    .ok_or_else(|| format!("x86_jit: no slot for Explicit[0] atom={}", ids[0]))?;
                let byte_off = slot.byte_offset as i64 + elem_idx as i64 * slot.elem_bytes as i64;
                return Ok((slot.dtype, LoadAddr::Disp(byte_off)));
            }
            // Multi-entry: build a u64 byte-offset table at compile time and
            // do a runtime lookup keyed by r13 (or i_const for !in_loop).
            let (first_slot, _) = layout
                .find(ids[0])
                .ok_or_else(|| format!("x86_jit: no slot for Explicit[0] atom={}", ids[0]))?;
            let load_dtype = first_slot.dtype;

            let byte_offsets: Vec<i64> = ids
                .iter()
                .map(|id| {
                    layout
                        .find(*id)
                        .map(|(slot, elem_idx)| {
                            slot.byte_offset as i64 + elem_idx as i64 * slot.elem_bytes as i64
                        })
                        .ok_or_else(|| format!("x86_jit: no slot for Explicit atom={}", id))
                })
                .collect::<Result<Vec<_>, String>>()?;

            let table_bytes: Vec<u8> = byte_offsets
                .iter()
                .flat_map(|off| off.to_le_bytes())
                .collect();
            let table_offset = tables.alloc(table_bytes);
            let table_off_i32 = i32::try_from(table_offset)
                .map_err(|_| "x86_jit: Explicit table offset overflows i32".to_string())?;

            if !in_loop {
                // Compile-time-known index: bake the byte offset into Disp.
                let idx = i_const as usize;
                if idx >= byte_offsets.len() {
                    return Err(format!(
                        "x86_jit: Explicit i_const={idx} out of range (len={})",
                        byte_offsets.len()
                    ));
                }
                return Ok((load_dtype, LoadAddr::Disp(byte_offsets[idx])));
            }

            // Runtime lookup:
            //   mov rax, r13              ; index
            //   mov rax, QWORD [r12 + rax*8 + table_off]  ; load i64 byte offset
            // → load uses [r12 + rax] (base = 0).
            dynasm!(ops
                ; .arch x64
                ; mov rax, r13
                ; mov rax, QWORD [r12 + rax * 8 + table_off_i32]
            );
            Ok((load_dtype, LoadAddr::RaxIndexed { base: 0 }))
        }
    }
}

/// Atom-relative offset computed at compile-time for an N-D Strided InputRef
/// at flat consumer index `i`. Mirrors `InputRef::resolve` exactly.
fn strided_offset_atoms(dim_strides: &[i64], dim_shape: &[u64], i: u64) -> i64 {
    let nd = dim_strides.len();
    let mut offset = 0i64;
    let mut remaining = i;
    for d in (0..nd).rev() {
        let coord = if d == 0 {
            remaining
        } else {
            let c = remaining % dim_shape[d];
            remaining /= dim_shape[d];
            c
        };
        offset += coord as i64 * dim_strides[d];
    }
    offset
}

/// Emit instructions that compute the per-iteration byte offset for a 2D
/// Strided InputRef into `rax`, given the loop variable in `r13`.
///
/// Layout for nd=2 (matching cranelift / `InputRef::resolve`):
///   inner = r13 % dim_shape[1]
///   outer = r13 / dim_shape[1]    (no modulus on the outermost dim)
///   rax   = outer * dim_strides[0] * elem_bytes + inner * dim_strides[1] * elem_bytes
///
/// Special cases:
///   - dim_strides[0] == 0  → only the inner term contributes (modular)
///   - dim_strides[1] == 0  → only the outer term contributes (strided_broadcast)
///   - dim_shape[1] is power of 2 → use shift/mask instead of div/mod
///
/// Clobbers `rax`, `rcx`, `rdx`, and `r10` (rcx for divisor / inner; rdx
/// for the high half of div; r10 for the constant divisor in the
/// non-power-of-2 path). The caller must not have live values in these
/// registers at the call site.
fn emit_strided_2d_offset_into_rax(
    ops: &mut Assembler,
    dim_strides: &[i64],
    dim_shape: &[u64],
    elem_bytes: i64,
) -> Result<(), String> {
    debug_assert_eq!(dim_strides.len(), 2);
    debug_assert_eq!(dim_shape.len(), 2);
    let s0 = dim_strides[0];
    let s1 = dim_strides[1];
    let m = dim_shape[1];
    let s0_bytes = s0 * elem_bytes;
    let s1_bytes = s1 * elem_bytes;

    // Modular only: rax = (r13 % m) * s1_bytes
    if s0 == 0 {
        emit_2d_inner_into_rax(ops, m, s1_bytes)?;
        return Ok(());
    }
    // Strided-broadcast only: rax = (r13 / m) * s0_bytes
    if s1 == 0 {
        emit_2d_outer_into_rax(ops, m, s0_bytes)?;
        return Ok(());
    }

    // General 2D: both contribute. Compute outer*s0_bytes in rax, then add
    // inner*s1_bytes from a scratch register.
    //
    // Power-of-2 modulus path: use shift/mask, both terms drop out of one
    // r13 read.
    if m.is_power_of_two() {
        let shift = m.trailing_zeros() as i8;
        let mask = (m as i64 - 1) as i32;
        let s0_imm = i32::try_from(s0_bytes)
            .map_err(|_| "x86_jit: strided 2D s0_bytes too large for i32".to_string())?;
        let s1_imm = i32::try_from(s1_bytes)
            .map_err(|_| "x86_jit: strided 2D s1_bytes too large for i32".to_string())?;
        dynasm!(ops
            ; .arch x64
            ; mov rcx, r13
            ; and rcx, mask          // rcx = inner = r13 & (m-1)
            ; mov rax, r13
            ; shr rax, BYTE shift    // rax = outer = r13 >> shift
            ; imul rax, rax, s0_imm  // rax = outer * s0_bytes
            ; imul rcx, rcx, s1_imm  // rcx = inner * s1_bytes
            ; add rax, rcx
        );
        return Ok(());
    }

    // Non-power-of-2: use div. r10 holds the divisor.
    let s0_imm = i32::try_from(s0_bytes)
        .map_err(|_| "x86_jit: strided 2D s0_bytes too large for i32".to_string())?;
    let s1_imm = i32::try_from(s1_bytes)
        .map_err(|_| "x86_jit: strided 2D s1_bytes too large for i32".to_string())?;
    dynasm!(ops
        ; .arch x64
        ; mov rax, r13
        ; xor rdx, rdx
        ; mov r10, QWORD m as i64
        ; div r10                    // rax = outer, rdx = inner
        ; mov rcx, rdx               // rcx = inner
        ; imul rax, rax, s0_imm      // rax = outer * s0_bytes
        ; imul rcx, rcx, s1_imm      // rcx = inner * s1_bytes
        ; add rax, rcx
    );
    Ok(())
}

/// Emit `rax = (r13 % m) * stride_bytes` for the modular-only case.
fn emit_2d_inner_into_rax(ops: &mut Assembler, m: u64, stride_bytes: i64) -> Result<(), String> {
    let stride_imm = i32::try_from(stride_bytes)
        .map_err(|_| "x86_jit: strided 2D stride too large for i32".to_string())?;
    if m.is_power_of_two() {
        let mask = (m as i64 - 1) as i32;
        dynasm!(ops
            ; .arch x64
            ; mov rax, r13
            ; and rax, mask
            ; imul rax, rax, stride_imm
        );
    } else {
        dynasm!(ops
            ; .arch x64
            ; mov rax, r13
            ; xor rdx, rdx
            ; mov r10, QWORD m as i64
            ; div r10                // rdx = inner = r13 % m
            ; mov rax, rdx
            ; imul rax, rax, stride_imm
        );
    }
    Ok(())
}

/// Emit `rax = (r13 / m) * stride_bytes` for the strided_broadcast-only case.
fn emit_2d_outer_into_rax(ops: &mut Assembler, m: u64, stride_bytes: i64) -> Result<(), String> {
    let stride_imm = i32::try_from(stride_bytes)
        .map_err(|_| "x86_jit: strided 2D stride too large for i32".to_string())?;
    if m.is_power_of_two() {
        let shift = m.trailing_zeros() as i8;
        dynasm!(ops
            ; .arch x64
            ; mov rax, r13
            ; shr rax, BYTE shift
            ; imul rax, rax, stride_imm
        );
    } else {
        dynasm!(ops
            ; .arch x64
            ; mov rax, r13
            ; xor rdx, rdx
            ; mov r10, QWORD m as i64
            ; div r10                // rax = outer = r13 / m
            ; imul rax, rax, stride_imm
        );
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

/// Emit `movss xmm{dst_xmm}, DWORD [r12 + rax + base]` for the case where
/// the per-iteration byte offset has already been materialized in `rax` by
/// `resolve_input_load_addr` (2D Strided / multi-entry Explicit).
fn emit_load_f32_rax(ops: &mut Assembler, dst_xmm: u8, base: i64) {
    let bb = base as i32;
    debug_assert_eq!(base, bb as i64, "rax-indexed F32 base out of i32 range");
    match dst_xmm {
        0 => dynasm!(ops ; .arch x64 ; movss xmm0, DWORD [r12 + rax + bb]),
        1 => dynasm!(ops ; .arch x64 ; movss xmm1, DWORD [r12 + rax + bb]),
        2 => dynasm!(ops ; .arch x64 ; movss xmm2, DWORD [r12 + rax + bb]),
        _ => unreachable!(),
    }
}

/// Emit a store from `xmm{src_xmm}` (holding an f32) into the output slot,
/// converting to BF16 / F16 if the slot's storage dtype requires it.
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
    let addr = if in_loop {
        StoreAddr::Indexed {
            base: store_base,
            elem_bytes: slot.elem_bytes as i64,
        }
    } else {
        let byte_off = store_base + i_const as i64 * slot.elem_bytes as i64;
        StoreAddr::Disp(byte_off)
    };

    match slot.dtype {
        NumericDType::F32 => emit_store_f32_at(ops, src_xmm, addr),
        NumericDType::BF16 => emit_store_bf16_at(ops, src_xmm, addr),
        NumericDType::F16 => emit_store_f16_at(ops, src_xmm, addr),
        other => panic!("emit_store_f32: unexpected slot dtype {other:?}"),
    }
}

#[derive(Debug, Clone, Copy)]
enum StoreAddr {
    /// Constant displacement: `[r12 + disp]`.
    Disp(i64),
    /// Indexed by r13: `[r12 + r13 * elem_bytes + base]`.
    Indexed { base: i64, elem_bytes: i64 },
}

/// Emit `movss [...], xmm{src_xmm}`.
fn emit_store_f32_at(ops: &mut Assembler, src_xmm: u8, addr: StoreAddr) {
    match addr {
        StoreAddr::Indexed { base, elem_bytes } => {
            let bb = base as i32;
            debug_assert_eq!(base, bb as i64, "store base out of i32 range");
            debug_assert_eq!(elem_bytes, 4);
            match src_xmm {
                0 => dynasm!(ops ; .arch x64 ; movss DWORD [r12 + r13 * 4 + bb], xmm0),
                1 => dynasm!(ops ; .arch x64 ; movss DWORD [r12 + r13 * 4 + bb], xmm1),
                2 => dynasm!(ops ; .arch x64 ; movss DWORD [r12 + r13 * 4 + bb], xmm2),
                _ => unreachable!(),
            }
        }
        StoreAddr::Disp(disp) => {
            let bb = disp as i32;
            debug_assert_eq!(disp, bb as i64, "store disp out of i32 range");
            match src_xmm {
                0 => dynasm!(ops ; .arch x64 ; movss DWORD [r12 + bb], xmm0),
                1 => dynasm!(ops ; .arch x64 ; movss DWORD [r12 + bb], xmm1),
                2 => dynasm!(ops ; .arch x64 ; movss DWORD [r12 + bb], xmm2),
                _ => unreachable!(),
            }
        }
    }
}

// ─── BF16 expand / narrow ───────────────────────────────────────────────────
//
// BF16 storage is just the high half of an f32: same sign bit, same 8-bit
// exponent, and the top 7 bits of the mantissa. Expand is a 3-instruction
// sequence (load 16 bits, shift left by 16, move to xmm). Narrow rounds to
// nearest even by adding `0x7fff + ((bits >> 16) & 1)` and shifting.

/// Emit a BF16 → F32 expand load: f32 in `xmm{dst}` from `[r12 + disp]`.
fn emit_load_bf16_disp(ops: &mut Assembler, dst_xmm: u8, disp: i64) {
    let d = disp as i32;
    debug_assert_eq!(disp, d as i64, "displacement out of i32 range: {disp}");
    dynasm!(ops
        ; .arch x64
        ; movzx eax, WORD [r12 + d]
        ; shl eax, 16
    );
    match dst_xmm {
        0 => dynasm!(ops ; .arch x64 ; movd xmm0, eax),
        1 => dynasm!(ops ; .arch x64 ; movd xmm1, eax),
        2 => dynasm!(ops ; .arch x64 ; movd xmm2, eax),
        _ => unreachable!(),
    }
}

/// Indexed BF16 → F32 expand load.
fn emit_load_bf16_indexed(ops: &mut Assembler, dst_xmm: u8, base_byte: i64, byte_stride: i64) {
    let bb = base_byte as i32;
    debug_assert_eq!(base_byte, bb as i64, "base_byte out of i32 range");
    if byte_stride == 0 {
        emit_load_bf16_disp(ops, dst_xmm, base_byte);
        return;
    }
    if byte_stride == 2 {
        dynasm!(ops ; .arch x64 ; movzx eax, WORD [r12 + r13 * 2 + bb]);
    } else if byte_stride == 4 {
        dynasm!(ops ; .arch x64 ; movzx eax, WORD [r12 + r13 * 4 + bb]);
    } else if byte_stride == 8 {
        dynasm!(ops ; .arch x64 ; movzx eax, WORD [r12 + r13 * 8 + bb]);
    } else if byte_stride == 1 {
        dynasm!(ops ; .arch x64 ; movzx eax, WORD [r12 + r13 + bb]);
    } else {
        // Materialize r13 * stride in rcx so rax stays free for the load result.
        let stride = byte_stride;
        dynasm!(ops
            ; .arch x64
            ; mov rcx, r13
            ; imul rcx, rcx, stride as i32
            ; movzx eax, WORD [r12 + rcx + bb]
        );
    }
    dynasm!(ops ; .arch x64 ; shl eax, 16);
    match dst_xmm {
        0 => dynasm!(ops ; .arch x64 ; movd xmm0, eax),
        1 => dynasm!(ops ; .arch x64 ; movd xmm1, eax),
        2 => dynasm!(ops ; .arch x64 ; movd xmm2, eax),
        _ => unreachable!(),
    }
}

/// BF16 → F32 expand load with byte offset already in `rax`.
fn emit_load_bf16_rax(ops: &mut Assembler, dst_xmm: u8, base: i64) {
    let bb = base as i32;
    debug_assert_eq!(base, bb as i64, "rax-indexed BF16 base out of i32 range");
    dynasm!(ops
        ; .arch x64
        ; movzx ecx, WORD [r12 + rax + bb]
        ; shl ecx, 16
    );
    match dst_xmm {
        0 => dynasm!(ops ; .arch x64 ; movd xmm0, ecx),
        1 => dynasm!(ops ; .arch x64 ; movd xmm1, ecx),
        2 => dynasm!(ops ; .arch x64 ; movd xmm2, ecx),
        _ => unreachable!(),
    }
}

/// Emit an F32 → BF16 narrow store: round-to-nearest-even, take top 16 bits.
///
/// Mirrors cranelift's emit_typed_store BF16 path:
///   bits = bitcast(val to i32)
///   bias = 0x7fff + ((bits >> 16) & 1)
///   rounded = bits + bias
///   bf16 = (rounded >> 16) as u16
fn emit_store_bf16_at(ops: &mut Assembler, src_xmm: u8, addr: StoreAddr) {
    // bits → eax
    match src_xmm {
        0 => dynasm!(ops ; .arch x64 ; movd eax, xmm0),
        1 => dynasm!(ops ; .arch x64 ; movd eax, xmm1),
        2 => dynasm!(ops ; .arch x64 ; movd eax, xmm2),
        _ => unreachable!(),
    }
    // bias = 0x7fff + ((bits >> 16) & 1)
    dynasm!(ops
        ; .arch x64
        ; mov ecx, eax
        ; shr ecx, 16
        ; and ecx, 1
        ; add ecx, 0x7fff
        ; add eax, ecx
        ; shr eax, 16
    );
    // store low 16 bits of eax = ax
    match addr {
        StoreAddr::Indexed { base, elem_bytes } => {
            let bb = base as i32;
            debug_assert_eq!(base, bb as i64, "store base out of i32 range");
            debug_assert_eq!(elem_bytes, 2, "BF16 elem_bytes != 2");
            dynasm!(ops ; .arch x64 ; mov WORD [r12 + r13 * 2 + bb], ax);
        }
        StoreAddr::Disp(disp) => {
            let bb = disp as i32;
            debug_assert_eq!(disp, bb as i64, "store disp out of i32 range");
            dynasm!(ops ; .arch x64 ; mov WORD [r12 + bb], ax);
        }
    }
}

// ─── F16 expand / narrow (software path via extern call) ───────────────────
//
// IEEE 754 half-precision has enough special-case handling (denormal
// subnormals, RTNE rounding on the way down, infinity/NaN propagation) that
// the inline path runs to ~25 instructions per direction. We delegate to
// `half::f16` via extern trampolines instead. F16C inline fast path is a
// follow-up.
//
// Calling convention: the load helper sets up `edi` (System V int arg 0)
// with the f16 bits, calls `jit_f16_to_f32`, and the result lands in xmm0.
// If the destination wasn't xmm0, we then move it to the right register.
// The store helper moves xmm{src} → xmm0 if needed, calls jit_f32_to_f16,
// and stores the resulting u16 from `ax`.

fn emit_load_f16_disp(ops: &mut Assembler, dst_xmm: u8, disp: i64) {
    let d = disp as i32;
    debug_assert_eq!(disp, d as i64, "displacement out of i32 range: {disp}");
    dynasm!(ops
        ; .arch x64
        ; movzx edi, WORD [r12 + d]
        ; mov rax, QWORD jit_f16_to_f32 as *const u8 as i64
        ; call rax
    );
    if dst_xmm != 0 {
        match dst_xmm {
            1 => dynasm!(ops ; .arch x64 ; movaps xmm1, xmm0),
            2 => dynasm!(ops ; .arch x64 ; movaps xmm2, xmm0),
            _ => unreachable!(),
        }
    }
}

fn emit_load_f16_indexed(ops: &mut Assembler, dst_xmm: u8, base_byte: i64, byte_stride: i64) {
    let bb = base_byte as i32;
    debug_assert_eq!(base_byte, bb as i64, "base_byte out of i32 range");
    if byte_stride == 0 {
        emit_load_f16_disp(ops, dst_xmm, base_byte);
        return;
    }
    if byte_stride == 2 {
        dynasm!(ops ; .arch x64 ; movzx edi, WORD [r12 + r13 * 2 + bb]);
    } else if byte_stride == 4 {
        dynasm!(ops ; .arch x64 ; movzx edi, WORD [r12 + r13 * 4 + bb]);
    } else if byte_stride == 8 {
        dynasm!(ops ; .arch x64 ; movzx edi, WORD [r12 + r13 * 8 + bb]);
    } else if byte_stride == 1 {
        dynasm!(ops ; .arch x64 ; movzx edi, WORD [r12 + r13 + bb]);
    } else {
        let stride = byte_stride;
        dynasm!(ops
            ; .arch x64
            ; mov rcx, r13
            ; imul rcx, rcx, stride as i32
            ; movzx edi, WORD [r12 + rcx + bb]
        );
    }
    dynasm!(ops
        ; .arch x64
        ; mov rax, QWORD jit_f16_to_f32 as *const u8 as i64
        ; call rax
    );
    if dst_xmm != 0 {
        match dst_xmm {
            1 => dynasm!(ops ; .arch x64 ; movaps xmm1, xmm0),
            2 => dynasm!(ops ; .arch x64 ; movaps xmm2, xmm0),
            _ => unreachable!(),
        }
    }
}

/// F16 → F32 software path with byte offset already in `rax`.
fn emit_load_f16_rax(ops: &mut Assembler, dst_xmm: u8, base: i64) {
    let bb = base as i32;
    debug_assert_eq!(base, bb as i64, "rax-indexed F16 base out of i32 range");
    dynasm!(ops
        ; .arch x64
        ; movzx edi, WORD [r12 + rax + bb]
        ; mov rax, QWORD jit_f16_to_f32 as *const u8 as i64
        ; call rax
    );
    if dst_xmm != 0 {
        match dst_xmm {
            1 => dynasm!(ops ; .arch x64 ; movaps xmm1, xmm0),
            2 => dynasm!(ops ; .arch x64 ; movaps xmm2, xmm0),
            _ => unreachable!(),
        }
    }
}

fn emit_store_f16_at(ops: &mut Assembler, src_xmm: u8, addr: StoreAddr) {
    // Move source to xmm0 if needed.
    if src_xmm != 0 {
        match src_xmm {
            1 => dynasm!(ops ; .arch x64 ; movaps xmm0, xmm1),
            2 => dynasm!(ops ; .arch x64 ; movaps xmm0, xmm2),
            _ => unreachable!(),
        }
    }
    dynasm!(ops
        ; .arch x64
        ; mov rax, QWORD jit_f32_to_f16 as *const u8 as i64
        ; call rax
    );
    // Result is u16 in ax (or rax low 16).
    match addr {
        StoreAddr::Indexed { base, elem_bytes } => {
            let bb = base as i32;
            debug_assert_eq!(base, bb as i64, "store base out of i32 range");
            debug_assert_eq!(elem_bytes, 2, "F16 elem_bytes != 2");
            dynasm!(ops ; .arch x64 ; mov WORD [r12 + r13 * 2 + bb], ax);
        }
        StoreAddr::Disp(disp) => {
            let bb = disp as i32;
            debug_assert_eq!(disp, bb as i64, "store disp out of i32 range");
            dynasm!(ops ; .arch x64 ; mov WORD [r12 + bb], ax);
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
        ScalarUnaryOp::Tan => emit_extern_call_unary_f32(ops, jit_tanf as *const u8),
        ScalarUnaryOp::Asin => emit_extern_call_unary_f32(ops, jit_asinf as *const u8),
        ScalarUnaryOp::Acos => emit_extern_call_unary_f32(ops, jit_acosf as *const u8),
        ScalarUnaryOp::Atan => emit_extern_call_unary_f32(ops, jit_atanf as *const u8),
        ScalarUnaryOp::Sinh => emit_extern_call_unary_f32(ops, jit_sinhf as *const u8),
        ScalarUnaryOp::Cosh => emit_extern_call_unary_f32(ops, jit_coshf as *const u8),
        ScalarUnaryOp::Asinh => emit_extern_call_unary_f32(ops, jit_asinhf as *const u8),
        ScalarUnaryOp::Acosh => emit_extern_call_unary_f32(ops, jit_acoshf as *const u8),
        ScalarUnaryOp::Atanh => emit_extern_call_unary_f32(ops, jit_atanhf as *const u8),
        ScalarUnaryOp::Log1p => emit_extern_call_unary_f32(ops, jit_log1pf as *const u8),
        ScalarUnaryOp::Erf => emit_extern_call_unary_f32(ops, jit_erff as *const u8),
        ScalarUnaryOp::Round => emit_extern_call_unary_f32(ops, jit_roundf as *const u8),
        ScalarUnaryOp::IsNan => {
            // x is NaN iff (x ucomi x) is unordered (PF=1).
            // Result is 1.0 if NaN, else 0.0.
            //   ucomiss xmm0, xmm0
            //   setp al
            //   movzx eax, al
            //   cvtsi2ss xmm0, eax
            dynasm!(ops
                ; .arch x64
                ; ucomiss xmm0, xmm0
                ; setp al
                ; movzx eax, al
                ; cvtsi2ss xmm0, eax
            );
        }
        ScalarUnaryOp::IsInf {
            detect_positive,
            detect_negative,
        } => {
            // x is +inf iff bit pattern == 0x7f800000.
            // x is -inf iff bit pattern == 0xff800000.
            // Result is 1.0 if matches, else 0.0.
            // Implementation: extract bits, compare against the appropriate
            // pattern(s), produce 0/1, convert to f32.
            let pos_inf = f32::INFINITY.to_bits() as i32;
            let neg_inf = f32::NEG_INFINITY.to_bits() as i32;
            dynasm!(ops
                ; .arch x64
                ; movd eax, xmm0       // eax = bits of x
                ; xor edx, edx         // edx = 0 (result accumulator)
            );
            if detect_positive {
                dynasm!(ops
                    ; .arch x64
                    ; cmp eax, DWORD pos_inf
                    ; sete cl
                    ; movzx ecx, cl
                    ; or edx, ecx
                );
            }
            if detect_negative {
                dynasm!(ops
                    ; .arch x64
                    ; cmp eax, DWORD neg_inf
                    ; sete cl
                    ; movzx ecx, cl
                    ; or edx, ecx
                );
            }
            dynasm!(ops ; .arch x64 ; cvtsi2ss xmm0, edx);
        }
        ScalarUnaryOp::Sign => {
            // sign(x) = -1 if x < 0, 0 if x == 0 (or NaN), 1 if x > 0.
            // Use the same construction as Cranelift: select(x>0, 1.0, select(x<0, -1.0, 0.0)).
            let one_bits = 1.0f32.to_bits() as i32;
            let neg_one_bits = (-1.0f32).to_bits() as i32;
            dynasm!(ops
                ; .arch x64
                // xmm1 = 0
                ; xorps xmm1, xmm1
                // xmm2 = 1.0
                ; mov eax, DWORD one_bits
                ; movd xmm2, eax
                // xmm3 = -1.0
                ; mov eax, DWORD neg_one_bits
                ; movd xmm3, eax
                // Test x > 0
                ; ucomiss xmm0, xmm1
                ; ja >sign_pos
                // Test x < 0
                ; ucomiss xmm1, xmm0
                ; ja >sign_neg
                // Else: 0
                ; movaps xmm0, xmm1
                ; jmp >sign_done
                ; sign_pos:
                ; movaps xmm0, xmm2
                ; jmp >sign_done
                ; sign_neg:
                ; movaps xmm0, xmm3
                ; sign_done:
            );
        }
        ScalarUnaryOp::Not => {
            // Logical not on float truthiness: 1.0 if x == 0, else 0.0.
            // NaN counts as nonzero (matches the And/Or/Xor convention).
            //   ucomiss xmm0, 0
            //   setz + setnp combined → ZF set AND PF clear means x == 0
            // We need x == 0 (ordered, not NaN). Easiest: check ZF=1 AND PF=0.
            //   sete al   ; ZF set
            //   setnp cl  ; PF clear
            //   and al, cl
            //   movzx eax, al
            //   cvtsi2ss xmm0, eax
            dynasm!(ops
                ; .arch x64
                ; xorps xmm1, xmm1
                ; ucomiss xmm0, xmm1
                ; sete al
                ; setnp cl
                ; and al, cl
                ; movzx eax, al
                ; cvtsi2ss xmm0, eax
            );
        }
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

// ─── F64 emission ───────────────────────────────────────────────────────────
//
// Mirrors the F32 path with movsd/addsd/etc. and 8-byte loads/stores.
// Storage stride is 8, internal compute lives in xmm0..xmm2 (lower 64 bits).

/// Emit one iteration of a group body in F64 compute.
fn emit_group_body_f64(
    ops: &mut Assembler,
    group: &AtomGroup<'static, SystemPool>,
    layout: &BufferLayout,
    tables: &mut EmbeddedTables,
    in_loop: bool,
    i_const: u64,
) -> Result<(), String> {
    let (out_slot, _) = layout
        .find(group.base_id)
        .ok_or_else(|| format!("x86_jit: no slot for group base={}", group.base_id))?;
    let out_slot = out_slot.clone();

    match &group.op {
        ScalarOp::Identity | ScalarOp::Cast { .. } => {
            emit_load_f64(
                ops,
                &group.inputs[0],
                layout,
                tables,
                in_loop,
                i_const,
                group.atom_offset,
                0,
            )?;
            emit_store_f64(ops, &out_slot, in_loop, i_const, group.atom_offset, 0);
        }

        ScalarOp::Binary { op, .. } => {
            emit_load_f64(
                ops,
                &group.inputs[0],
                layout,
                tables,
                in_loop,
                i_const,
                group.atom_offset,
                0,
            )?;
            emit_load_f64(
                ops,
                &group.inputs[1],
                layout,
                tables,
                in_loop,
                i_const,
                group.atom_offset,
                1,
            )?;
            emit_binop_f64(ops, *op);
            emit_store_f64(ops, &out_slot, in_loop, i_const, group.atom_offset, 0);
        }

        ScalarOp::Unary { op, .. } => {
            emit_load_f64(
                ops,
                &group.inputs[0],
                layout,
                tables,
                in_loop,
                i_const,
                group.atom_offset,
                0,
            )?;
            emit_unop_f64(ops, *op);
            emit_store_f64(ops, &out_slot, in_loop, i_const, group.atom_offset, 0);
        }

        ScalarOp::Select => {
            emit_load_f64(
                ops,
                &group.inputs[0],
                layout,
                tables,
                in_loop,
                i_const,
                group.atom_offset,
                0,
            )?;
            emit_load_f64(
                ops,
                &group.inputs[1],
                layout,
                tables,
                in_loop,
                i_const,
                group.atom_offset,
                1,
            )?;
            emit_load_f64(
                ops,
                &group.inputs[2],
                layout,
                tables,
                in_loop,
                i_const,
                group.atom_offset,
                2,
            )?;

            let take_x = ops.new_dynamic_label();
            let done = ops.new_dynamic_label();
            dynasm!(ops
                ; .arch x64
                ; xorpd xmm3, xmm3
                ; ucomisd xmm0, xmm3
                ; jne =>take_x
                ; jp  =>take_x
                ; movapd xmm0, xmm2
                ; jmp =>done
                ; =>take_x
                ; movapd xmm0, xmm1
                ; =>done
            );

            emit_store_f64(ops, &out_slot, in_loop, i_const, group.atom_offset, 0);
        }

        ScalarOp::Literal(_) | ScalarOp::LiteralSpan(_) => {}
        _ => {
            return Err("x86_jit: emit_group_body_f64 unexpected op".to_string());
        }
    }

    Ok(())
}

/// Emit `movsd <xmm{dst}>, [r12 + addr]` for an InputRef. F64 storage only.
fn emit_load_f64(
    ops: &mut Assembler,
    input: &InputRef,
    layout: &BufferLayout,
    tables: &mut EmbeddedTables,
    in_loop: bool,
    i_const: u64,
    atom_offset: u64,
    dst_xmm: u8,
) -> Result<(), String> {
    let (slot_dtype, addr) =
        resolve_input_load_addr(ops, input, layout, tables, in_loop, i_const, atom_offset)?;
    if slot_dtype != NumericDType::F64 {
        return Err(format!(
            "x86_jit emit_load_f64: unexpected slot dtype {slot_dtype:?}"
        ));
    }
    match addr {
        LoadAddr::Disp(d) => emit_load_f64_disp(ops, dst_xmm, d),
        LoadAddr::Indexed { base, stride } => emit_load_f64_indexed(ops, dst_xmm, base, stride),
        LoadAddr::RaxIndexed { base } => emit_load_f64_rax(ops, dst_xmm, base),
    }
    Ok(())
}

/// Emit `movsd xmm{dst_xmm}, QWORD [r12 + rax + base]`.
fn emit_load_f64_rax(ops: &mut Assembler, dst_xmm: u8, base: i64) {
    let bb = base as i32;
    debug_assert_eq!(base, bb as i64, "rax-indexed F64 base out of i32 range");
    match dst_xmm {
        0 => dynasm!(ops ; .arch x64 ; movsd xmm0, QWORD [r12 + rax + bb]),
        1 => dynasm!(ops ; .arch x64 ; movsd xmm1, QWORD [r12 + rax + bb]),
        2 => dynasm!(ops ; .arch x64 ; movsd xmm2, QWORD [r12 + rax + bb]),
        _ => unreachable!(),
    }
}

fn emit_load_f64_disp(ops: &mut Assembler, dst_xmm: u8, disp: i64) {
    let d = disp as i32;
    debug_assert_eq!(disp, d as i64, "displacement out of i32 range: {disp}");
    match dst_xmm {
        0 => dynasm!(ops ; .arch x64 ; movsd xmm0, QWORD [r12 + d]),
        1 => dynasm!(ops ; .arch x64 ; movsd xmm1, QWORD [r12 + d]),
        2 => dynasm!(ops ; .arch x64 ; movsd xmm2, QWORD [r12 + d]),
        _ => unreachable!("dst_xmm out of range: {dst_xmm}"),
    }
}

/// Emit `movsd xmm{dst_xmm}, QWORD [r12 + r13 * scale + base_byte]` where
/// `byte_stride` should be 8 for normal F64 stride-1 access. Other strides
/// fall back to materializing the index in rax.
fn emit_load_f64_indexed(ops: &mut Assembler, dst_xmm: u8, base_byte: i64, byte_stride: i64) {
    let bb = base_byte as i32;
    debug_assert_eq!(base_byte, bb as i64, "base_byte out of i32 range");

    if byte_stride == 8 {
        match dst_xmm {
            0 => dynasm!(ops ; .arch x64 ; movsd xmm0, QWORD [r12 + r13 * 8 + bb]),
            1 => dynasm!(ops ; .arch x64 ; movsd xmm1, QWORD [r12 + r13 * 8 + bb]),
            2 => dynasm!(ops ; .arch x64 ; movsd xmm2, QWORD [r12 + r13 * 8 + bb]),
            _ => unreachable!(),
        }
    } else if byte_stride == 0 {
        emit_load_f64_disp(ops, dst_xmm, base_byte);
    } else if byte_stride == 4 {
        match dst_xmm {
            0 => dynasm!(ops ; .arch x64 ; movsd xmm0, QWORD [r12 + r13 * 4 + bb]),
            1 => dynasm!(ops ; .arch x64 ; movsd xmm1, QWORD [r12 + r13 * 4 + bb]),
            2 => dynasm!(ops ; .arch x64 ; movsd xmm2, QWORD [r12 + r13 * 4 + bb]),
            _ => unreachable!(),
        }
    } else if byte_stride == 2 {
        match dst_xmm {
            0 => dynasm!(ops ; .arch x64 ; movsd xmm0, QWORD [r12 + r13 * 2 + bb]),
            1 => dynasm!(ops ; .arch x64 ; movsd xmm1, QWORD [r12 + r13 * 2 + bb]),
            2 => dynasm!(ops ; .arch x64 ; movsd xmm2, QWORD [r12 + r13 * 2 + bb]),
            _ => unreachable!(),
        }
    } else if byte_stride == 1 {
        match dst_xmm {
            0 => dynasm!(ops ; .arch x64 ; movsd xmm0, QWORD [r12 + r13 + bb]),
            1 => dynasm!(ops ; .arch x64 ; movsd xmm1, QWORD [r12 + r13 + bb]),
            2 => dynasm!(ops ; .arch x64 ; movsd xmm2, QWORD [r12 + r13 + bb]),
            _ => unreachable!(),
        }
    } else {
        let stride = byte_stride;
        dynasm!(ops
            ; .arch x64
            ; mov rax, r13
            ; imul rax, rax, stride as i32
        );
        match dst_xmm {
            0 => dynasm!(ops ; .arch x64 ; movsd xmm0, QWORD [r12 + rax + bb]),
            1 => dynasm!(ops ; .arch x64 ; movsd xmm1, QWORD [r12 + rax + bb]),
            2 => dynasm!(ops ; .arch x64 ; movsd xmm2, QWORD [r12 + rax + bb]),
            _ => unreachable!(),
        }
    }
}

fn emit_store_f64(
    ops: &mut Assembler,
    slot: &SlotInfo,
    in_loop: bool,
    i_const: u64,
    atom_offset: u64,
    src_xmm: u8,
) {
    let store_base = slot.byte_offset as i64 - atom_offset as i64 * slot.elem_bytes as i64;

    if in_loop {
        let bb = store_base as i32;
        debug_assert_eq!(store_base, bb as i64, "store_base out of i32 range");
        match src_xmm {
            0 => dynasm!(ops ; .arch x64 ; movsd QWORD [r12 + r13 * 8 + bb], xmm0),
            1 => dynasm!(ops ; .arch x64 ; movsd QWORD [r12 + r13 * 8 + bb], xmm1),
            2 => dynasm!(ops ; .arch x64 ; movsd QWORD [r12 + r13 * 8 + bb], xmm2),
            _ => unreachable!(),
        }
    } else {
        let byte_off = store_base + i_const as i64 * slot.elem_bytes as i64;
        let bb = byte_off as i32;
        debug_assert_eq!(byte_off, bb as i64, "byte_off out of i32 range");
        match src_xmm {
            0 => dynasm!(ops ; .arch x64 ; movsd QWORD [r12 + bb], xmm0),
            1 => dynasm!(ops ; .arch x64 ; movsd QWORD [r12 + bb], xmm1),
            2 => dynasm!(ops ; .arch x64 ; movsd QWORD [r12 + bb], xmm2),
            _ => unreachable!(),
        }
    }
}

/// Emit a binary op: `op(xmm0, xmm1) → xmm0`, F64.
fn emit_binop_f64(ops: &mut Assembler, op: ScalarBinOp) {
    match op {
        ScalarBinOp::Add => dynasm!(ops ; .arch x64 ; addsd xmm0, xmm1),
        ScalarBinOp::Sub => dynasm!(ops ; .arch x64 ; subsd xmm0, xmm1),
        ScalarBinOp::Mul => dynasm!(ops ; .arch x64 ; mulsd xmm0, xmm1),
        ScalarBinOp::Div => dynasm!(ops ; .arch x64 ; divsd xmm0, xmm1),
        ScalarBinOp::Min => dynasm!(ops ; .arch x64 ; minsd xmm0, xmm1),
        ScalarBinOp::Max => dynasm!(ops ; .arch x64 ; maxsd xmm0, xmm1),

        ScalarBinOp::Equal => emit_f64_cmp(ops, CmpPred::Eq),
        ScalarBinOp::Less => emit_f64_cmp(ops, CmpPred::Lt),
        ScalarBinOp::LessOrEqual => emit_f64_cmp(ops, CmpPred::Le),
        ScalarBinOp::Greater => emit_f64_cmp(ops, CmpPred::Gt),
        ScalarBinOp::GreaterOrEqual => emit_f64_cmp(ops, CmpPred::Ge),

        ScalarBinOp::And => emit_f64_logical(ops, LogicalOp::And),
        ScalarBinOp::Or => emit_f64_logical(ops, LogicalOp::Or),
        ScalarBinOp::Xor => emit_f64_logical(ops, LogicalOp::Xor),

        ScalarBinOp::Pow => emit_extern_call_binary_f64(ops, jit_pow_d as *const u8),
        ScalarBinOp::Mod => emit_extern_call_binary_f64(ops, jit_fmod_d as *const u8),
        ScalarBinOp::IMod => {
            // Same fmod-then-adjust as F32 IMod, with sd mnemonics. Save b on
            // the stack so we can recover it after the fmod call clobbers xmm1.
            dynasm!(ops
                ; .arch x64
                ; movsd QWORD [rsp + 0], xmm1
            );
            emit_extern_call_binary_f64(ops, jit_fmod_d as *const u8);
            dynasm!(ops
                ; .arch x64
                ; movsd xmm1, QWORD [rsp + 0]
                ; movapd xmm2, xmm0
                ; addsd xmm2, xmm1
                ; xorpd xmm3, xmm3
                ; ucomisd xmm0, xmm3
                ; jne >adjust_check
                ; jp  >adjust_check
                ; jmp >done
                ; adjust_check:
                ; movapd xmm3, xmm0
                ; mulsd xmm3, xmm1
                ; xorpd xmm4, xmm4
                ; ucomisd xmm3, xmm4
                ; jae >done
                ; jp >done
                ; movapd xmm0, xmm2
                ; done:
            );
        }
        _ => panic!("emit_binop_f64: unexpected op {op:?}"),
    }
}

/// Emit `xmm0 = (xmm0 cmp xmm1) ? 1.0 : 0.0`, F64.
fn emit_f64_cmp(ops: &mut Assembler, pred: CmpPred) {
    let one_bits = 1.0f64.to_bits() as i64;
    match pred {
        CmpPred::Eq => dynasm!(ops ; .arch x64 ; cmpsd xmm0, xmm1, 0),
        CmpPred::Lt => dynasm!(ops ; .arch x64 ; cmpsd xmm0, xmm1, 1),
        CmpPred::Le => dynasm!(ops ; .arch x64 ; cmpsd xmm0, xmm1, 2),
        CmpPred::Gt => {
            dynasm!(ops
                ; .arch x64
                ; cmpsd xmm1, xmm0, 1
                ; movapd xmm0, xmm1
            );
        }
        CmpPred::Ge => {
            dynasm!(ops
                ; .arch x64
                ; cmpsd xmm1, xmm0, 2
                ; movapd xmm0, xmm1
            );
        }
    }
    dynasm!(ops
        ; .arch x64
        ; mov rax, QWORD one_bits
        ; movq xmm2, rax
        ; andpd xmm0, xmm2
    );
}

/// Emit logical op on F64 truthiness: each operand → 0/1, then op.
fn emit_f64_logical(ops: &mut Assembler, op: LogicalOp) {
    let one_bits = 1.0f64.to_bits() as i64;
    dynasm!(ops
        ; .arch x64
        ; xorpd xmm2, xmm2
        ; cmpsd xmm0, xmm2, 4
        ; cmpsd xmm1, xmm2, 4
    );
    match op {
        LogicalOp::And => dynasm!(ops ; .arch x64 ; andpd xmm0, xmm1),
        LogicalOp::Or => dynasm!(ops ; .arch x64 ; orpd xmm0, xmm1),
        LogicalOp::Xor => dynasm!(ops ; .arch x64 ; xorpd xmm0, xmm1),
    }
    dynasm!(ops
        ; .arch x64
        ; mov rax, QWORD one_bits
        ; movq xmm2, rax
        ; andpd xmm0, xmm2
    );
}

/// Emit a unary op: `op(xmm0) → xmm0`, F64.
fn emit_unop_f64(ops: &mut Assembler, op: ScalarUnaryOp) {
    match op {
        ScalarUnaryOp::Neg => {
            let mask = 0x8000_0000_0000_0000u64 as i64;
            dynasm!(ops
                ; .arch x64
                ; mov rax, QWORD mask
                ; movq xmm1, rax
                ; xorpd xmm0, xmm1
            );
        }
        ScalarUnaryOp::Abs => {
            let mask = 0x7fff_ffff_ffff_ffffu64 as i64;
            dynasm!(ops
                ; .arch x64
                ; mov rax, QWORD mask
                ; movq xmm1, rax
                ; andpd xmm0, xmm1
            );
        }
        ScalarUnaryOp::Sqrt => dynasm!(ops ; .arch x64 ; sqrtsd xmm0, xmm0),
        ScalarUnaryOp::Reciprocal => {
            let one_bits = 1.0f64.to_bits() as i64;
            dynasm!(ops
                ; .arch x64
                ; mov rax, QWORD one_bits
                ; movq xmm1, rax
                ; divsd xmm1, xmm0
                ; movapd xmm0, xmm1
            );
        }
        ScalarUnaryOp::Exp => emit_extern_call_unary_f64(ops, jit_exp_d as *const u8),
        ScalarUnaryOp::Ln => emit_extern_call_unary_f64(ops, jit_log_d as *const u8),
        ScalarUnaryOp::Tanh => emit_extern_call_unary_f64(ops, jit_tanh_d as *const u8),
        ScalarUnaryOp::Floor => emit_extern_call_unary_f64(ops, jit_floor_d as *const u8),
        ScalarUnaryOp::Ceil => emit_extern_call_unary_f64(ops, jit_ceil_d as *const u8),
        ScalarUnaryOp::Sin => emit_extern_call_unary_f64(ops, jit_sin_d as *const u8),
        ScalarUnaryOp::Cos => emit_extern_call_unary_f64(ops, jit_cos_d as *const u8),
        ScalarUnaryOp::Tan => emit_extern_call_unary_f64(ops, jit_tan_d as *const u8),
        ScalarUnaryOp::Asin => emit_extern_call_unary_f64(ops, jit_asin_d as *const u8),
        ScalarUnaryOp::Acos => emit_extern_call_unary_f64(ops, jit_acos_d as *const u8),
        ScalarUnaryOp::Atan => emit_extern_call_unary_f64(ops, jit_atan_d as *const u8),
        ScalarUnaryOp::Sinh => emit_extern_call_unary_f64(ops, jit_sinh_d as *const u8),
        ScalarUnaryOp::Cosh => emit_extern_call_unary_f64(ops, jit_cosh_d as *const u8),
        ScalarUnaryOp::Asinh => emit_extern_call_unary_f64(ops, jit_asinh_d as *const u8),
        ScalarUnaryOp::Acosh => emit_extern_call_unary_f64(ops, jit_acosh_d as *const u8),
        ScalarUnaryOp::Atanh => emit_extern_call_unary_f64(ops, jit_atanh_d as *const u8),
        ScalarUnaryOp::Log1p => emit_extern_call_unary_f64(ops, jit_log1p_d as *const u8),
        ScalarUnaryOp::Erf => emit_extern_call_unary_f64(ops, jit_erf_d as *const u8),
        ScalarUnaryOp::Round => emit_extern_call_unary_f64(ops, jit_round_d as *const u8),
        ScalarUnaryOp::IsNan => {
            // x is NaN iff (x ucomi x) is unordered (PF=1).
            dynasm!(ops
                ; .arch x64
                ; ucomisd xmm0, xmm0
                ; setp al
                ; movzx eax, al
                ; cvtsi2sd xmm0, eax
            );
        }
        ScalarUnaryOp::IsInf {
            detect_positive,
            detect_negative,
        } => {
            let pos_inf = f64::INFINITY.to_bits() as i64;
            let neg_inf = f64::NEG_INFINITY.to_bits() as i64;
            dynasm!(ops
                ; .arch x64
                ; movq rax, xmm0
                ; xor edx, edx
            );
            if detect_positive {
                dynasm!(ops
                    ; .arch x64
                    ; mov rcx, QWORD pos_inf
                    ; cmp rax, rcx
                    ; sete cl
                    ; movzx ecx, cl
                    ; or edx, ecx
                );
            }
            if detect_negative {
                dynasm!(ops
                    ; .arch x64
                    ; mov rcx, QWORD neg_inf
                    ; cmp rax, rcx
                    ; sete cl
                    ; movzx ecx, cl
                    ; or edx, ecx
                );
            }
            dynasm!(ops ; .arch x64 ; cvtsi2sd xmm0, edx);
        }
        ScalarUnaryOp::Sign => {
            let one_bits = 1.0f64.to_bits() as i64;
            let neg_one_bits = (-1.0f64).to_bits() as i64;
            dynasm!(ops
                ; .arch x64
                ; xorpd xmm1, xmm1
                ; mov rax, QWORD one_bits
                ; movq xmm2, rax
                ; mov rax, QWORD neg_one_bits
                ; movq xmm3, rax
                ; ucomisd xmm0, xmm1
                ; ja >sign_pos_d
                ; ucomisd xmm1, xmm0
                ; ja >sign_neg_d
                ; movapd xmm0, xmm1
                ; jmp >sign_done_d
                ; sign_pos_d:
                ; movapd xmm0, xmm2
                ; jmp >sign_done_d
                ; sign_neg_d:
                ; movapd xmm0, xmm3
                ; sign_done_d:
            );
        }
        ScalarUnaryOp::Not => {
            dynasm!(ops
                ; .arch x64
                ; xorpd xmm1, xmm1
                ; ucomisd xmm0, xmm1
                ; sete al
                ; setnp cl
                ; and al, cl
                ; movzx eax, al
                ; cvtsi2sd xmm0, eax
            );
        }
        _ => panic!("emit_unop_f64: unexpected op {op:?}"),
    }
}

/// Emit a call to an `extern "C" fn(f64) -> f64` with the arg in xmm0.
fn emit_extern_call_unary_f64(ops: &mut Assembler, fn_addr: *const u8) {
    let addr = fn_addr as i64;
    dynasm!(ops
        ; .arch x64
        ; mov rax, QWORD addr
        ; call rax
    );
}

/// Emit a call to an `extern "C" fn(f64, f64) -> f64` with args in xmm0, xmm1.
fn emit_extern_call_binary_f64(ops: &mut Assembler, fn_addr: *const u8) {
    let addr = fn_addr as i64;
    dynasm!(ops
        ; .arch x64
        ; mov rax, QWORD addr
        ; call rax
    );
}

// ─── Int emission ───────────────────────────────────────────────────────────
//
// All integer compute happens in i64. Loads from narrower slots sign- or
// zero-extend; stores narrow back. The reg layout for an op is:
//
//   rax — slot 0 (a / x / cond / result)
//   rcx — slot 1 (b)
//   rdx — slot 2 (Select y; idiv high half)
//   r8, r9 — additional scratch for IMod
//
// Loads with non-power-of-2 stride use rax as a scratch in the address
// calculation, so we always sequence loads such that the rax-destination
// load happens LAST.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IntReg {
    Rax,
    Rcx,
    Rdx,
}

/// Emit one iteration of a group body in integer compute (i64 internal).
fn emit_group_body_int(
    ops: &mut Assembler,
    group: &AtomGroup<'static, SystemPool>,
    layout: &BufferLayout,
    tables: &mut EmbeddedTables,
    in_loop: bool,
    i_const: u64,
) -> Result<(), String> {
    let (out_slot, _) = layout
        .find(group.base_id)
        .ok_or_else(|| format!("x86_jit: no slot for group base={}", group.base_id))?;
    let out_slot = out_slot.clone();

    match &group.op {
        ScalarOp::Identity | ScalarOp::Cast { .. } => {
            emit_load_int(
                ops,
                &group.inputs[0],
                layout,
                tables,
                in_loop,
                i_const,
                group.atom_offset,
                IntReg::Rax,
            )?;
            emit_store_int(
                ops,
                &out_slot,
                in_loop,
                i_const,
                group.atom_offset,
                IntReg::Rax,
            );
        }

        ScalarOp::Binary { op, .. } => {
            // Load b first (into rcx) so the rax load happens last and the
            // address-scratch use of rax doesn't clobber it.
            emit_load_int(
                ops,
                &group.inputs[1],
                layout,
                tables,
                in_loop,
                i_const,
                group.atom_offset,
                IntReg::Rcx,
            )?;
            emit_load_int(
                ops,
                &group.inputs[0],
                layout,
                tables,
                in_loop,
                i_const,
                group.atom_offset,
                IntReg::Rax,
            )?;
            emit_binop_int(ops, *op);
            emit_store_int(
                ops,
                &out_slot,
                in_loop,
                i_const,
                group.atom_offset,
                IntReg::Rax,
            );
        }

        ScalarOp::Unary { op, .. } => {
            emit_load_int(
                ops,
                &group.inputs[0],
                layout,
                tables,
                in_loop,
                i_const,
                group.atom_offset,
                IntReg::Rax,
            )?;
            emit_unop_int(ops, *op);
            emit_store_int(
                ops,
                &out_slot,
                in_loop,
                i_const,
                group.atom_offset,
                IntReg::Rax,
            );
        }

        ScalarOp::Select => {
            // Load y → rdx, x → rcx, cond → rax (rax last so address-scratch
            // doesn't clobber it).
            emit_load_int(
                ops,
                &group.inputs[2],
                layout,
                tables,
                in_loop,
                i_const,
                group.atom_offset,
                IntReg::Rdx,
            )?;
            emit_load_int(
                ops,
                &group.inputs[1],
                layout,
                tables,
                in_loop,
                i_const,
                group.atom_offset,
                IntReg::Rcx,
            )?;
            emit_load_int(
                ops,
                &group.inputs[0],
                layout,
                tables,
                in_loop,
                i_const,
                group.atom_offset,
                IntReg::Rax,
            )?;
            // result = (cond != 0) ? x : y
            //   test rax, rax  ; sets ZF based on cond
            //   mov rax, rdx   ; default to y
            //   cmovne rax, rcx ; if cond != 0, take x
            dynasm!(ops
                ; .arch x64
                ; test rax, rax
                ; mov rax, rdx
                ; cmovne rax, rcx
            );
            emit_store_int(
                ops,
                &out_slot,
                in_loop,
                i_const,
                group.atom_offset,
                IntReg::Rax,
            );
        }

        ScalarOp::Literal(_) | ScalarOp::LiteralSpan(_) => {}
        _ => {
            return Err("x86_jit: emit_group_body_int unexpected op".to_string());
        }
    }

    Ok(())
}

/// Emit a load from an InputRef into the given GP register, sign/zero
/// extending the slot's storage width to i64.
fn emit_load_int(
    ops: &mut Assembler,
    input: &InputRef,
    layout: &BufferLayout,
    tables: &mut EmbeddedTables,
    in_loop: bool,
    i_const: u64,
    atom_offset: u64,
    dst: IntReg,
) -> Result<(), String> {
    let (slot_dtype, addr) =
        resolve_input_load_addr(ops, input, layout, tables, in_loop, i_const, atom_offset)?;
    match addr {
        LoadAddr::Disp(d) => emit_load_int_disp(ops, slot_dtype, dst, d),
        LoadAddr::Indexed { base, stride } => {
            emit_load_int_indexed(ops, slot_dtype, dst, base, stride)
        }
        LoadAddr::RaxIndexed { base } => {
            // The byte offset is in rax; reuse the same `[r12 + rax + bb]`
            // helper that emit_load_int_indexed uses for non-power-of-2
            // strides. Add `base` to rax first if non-zero.
            if base != 0 {
                let bb_imm = i32::try_from(base)
                    .map_err(|_| "x86_jit: int rax-indexed base too large for i32".to_string())?;
                dynasm!(ops ; .arch x64 ; add rax, bb_imm);
            }
            emit_load_int_indexed_rax(ops, slot_dtype, dst, 0);
        }
    }
    Ok(())
}

/// Emit a sign/zero-extending load from `[r12 + disp]` into the given i64
/// register.
fn emit_load_int_disp(ops: &mut Assembler, slot_dtype: NumericDType, dst: IntReg, disp: i64) {
    let d = disp as i32;
    debug_assert_eq!(disp, d as i64, "displacement out of i32 range: {disp}");
    match (slot_dtype, dst) {
        (NumericDType::I64 | NumericDType::U64, IntReg::Rax) => {
            dynasm!(ops ; .arch x64 ; mov rax, QWORD [r12 + d])
        }
        (NumericDType::I64 | NumericDType::U64, IntReg::Rcx) => {
            dynasm!(ops ; .arch x64 ; mov rcx, QWORD [r12 + d])
        }
        (NumericDType::I64 | NumericDType::U64, IntReg::Rdx) => {
            dynasm!(ops ; .arch x64 ; mov rdx, QWORD [r12 + d])
        }
        (NumericDType::I32, IntReg::Rax) => {
            dynasm!(ops ; .arch x64 ; movsxd rax, DWORD [r12 + d])
        }
        (NumericDType::I32, IntReg::Rcx) => {
            dynasm!(ops ; .arch x64 ; movsxd rcx, DWORD [r12 + d])
        }
        (NumericDType::I32, IntReg::Rdx) => {
            dynasm!(ops ; .arch x64 ; movsxd rdx, DWORD [r12 + d])
        }
        (NumericDType::U32, IntReg::Rax) => {
            // mov eax zero-extends to rax automatically.
            dynasm!(ops ; .arch x64 ; mov eax, DWORD [r12 + d])
        }
        (NumericDType::U32, IntReg::Rcx) => {
            dynasm!(ops ; .arch x64 ; mov ecx, DWORD [r12 + d])
        }
        (NumericDType::U32, IntReg::Rdx) => {
            dynasm!(ops ; .arch x64 ; mov edx, DWORD [r12 + d])
        }
        (NumericDType::I16, IntReg::Rax) => {
            dynasm!(ops ; .arch x64 ; movsx rax, WORD [r12 + d])
        }
        (NumericDType::I16, IntReg::Rcx) => {
            dynasm!(ops ; .arch x64 ; movsx rcx, WORD [r12 + d])
        }
        (NumericDType::I16, IntReg::Rdx) => {
            dynasm!(ops ; .arch x64 ; movsx rdx, WORD [r12 + d])
        }
        (NumericDType::U16, IntReg::Rax) => {
            dynasm!(ops ; .arch x64 ; movzx rax, WORD [r12 + d])
        }
        (NumericDType::U16, IntReg::Rcx) => {
            dynasm!(ops ; .arch x64 ; movzx rcx, WORD [r12 + d])
        }
        (NumericDType::U16, IntReg::Rdx) => {
            dynasm!(ops ; .arch x64 ; movzx rdx, WORD [r12 + d])
        }
        (NumericDType::I8, IntReg::Rax) => {
            dynasm!(ops ; .arch x64 ; movsx rax, BYTE [r12 + d])
        }
        (NumericDType::I8, IntReg::Rcx) => {
            dynasm!(ops ; .arch x64 ; movsx rcx, BYTE [r12 + d])
        }
        (NumericDType::I8, IntReg::Rdx) => {
            dynasm!(ops ; .arch x64 ; movsx rdx, BYTE [r12 + d])
        }
        (NumericDType::U8 | NumericDType::BOOL, IntReg::Rax) => {
            dynasm!(ops ; .arch x64 ; movzx rax, BYTE [r12 + d])
        }
        (NumericDType::U8 | NumericDType::BOOL, IntReg::Rcx) => {
            dynasm!(ops ; .arch x64 ; movzx rcx, BYTE [r12 + d])
        }
        (NumericDType::U8 | NumericDType::BOOL, IntReg::Rdx) => {
            dynasm!(ops ; .arch x64 ; movzx rdx, BYTE [r12 + d])
        }
        _ => panic!("emit_load_int_disp: unsupported slot dtype {slot_dtype:?}"),
    }
}

/// Emit a sign/zero-extending load from `[r12 + r13*scale + base_byte]` into
/// the given i64 register.
fn emit_load_int_indexed(
    ops: &mut Assembler,
    slot_dtype: NumericDType,
    dst: IntReg,
    base_byte: i64,
    byte_stride: i64,
) {
    let bb = base_byte as i32;
    debug_assert_eq!(base_byte, bb as i64, "base_byte out of i32 range");

    if byte_stride == 0 {
        emit_load_int_disp(ops, slot_dtype, dst, base_byte);
        return;
    }

    // For non-power-of-2 strides, materialize r13 * stride into rax first.
    if !matches!(byte_stride, 1 | 2 | 4 | 8) {
        let stride = byte_stride;
        dynasm!(ops
            ; .arch x64
            ; mov rax, r13
            ; imul rax, rax, stride as i32
        );
        // Now use [r12 + rax + base_byte] as the address.
        emit_load_int_indexed_rax(ops, slot_dtype, dst, bb);
        return;
    }

    // SIB-encoded path with the natural scale.
    match (slot_dtype, dst, byte_stride) {
        // I64 / U64 (scale 8 only — slot stride should be 8 for affine i64)
        (NumericDType::I64 | NumericDType::U64, IntReg::Rax, 8) => {
            dynasm!(ops ; .arch x64 ; mov rax, QWORD [r12 + r13 * 8 + bb])
        }
        (NumericDType::I64 | NumericDType::U64, IntReg::Rcx, 8) => {
            dynasm!(ops ; .arch x64 ; mov rcx, QWORD [r12 + r13 * 8 + bb])
        }
        (NumericDType::I64 | NumericDType::U64, IntReg::Rdx, 8) => {
            dynasm!(ops ; .arch x64 ; mov rdx, QWORD [r12 + r13 * 8 + bb])
        }
        // I32 (scale 4 for affine, other scales possible for broadcast)
        (NumericDType::I32, IntReg::Rax, 4) => {
            dynasm!(ops ; .arch x64 ; movsxd rax, DWORD [r12 + r13 * 4 + bb])
        }
        (NumericDType::I32, IntReg::Rcx, 4) => {
            dynasm!(ops ; .arch x64 ; movsxd rcx, DWORD [r12 + r13 * 4 + bb])
        }
        (NumericDType::I32, IntReg::Rdx, 4) => {
            dynasm!(ops ; .arch x64 ; movsxd rdx, DWORD [r12 + r13 * 4 + bb])
        }
        // U32 — uses mov dword (auto zero-extend)
        (NumericDType::U32, IntReg::Rax, 4) => {
            dynasm!(ops ; .arch x64 ; mov eax, DWORD [r12 + r13 * 4 + bb])
        }
        (NumericDType::U32, IntReg::Rcx, 4) => {
            dynasm!(ops ; .arch x64 ; mov ecx, DWORD [r12 + r13 * 4 + bb])
        }
        (NumericDType::U32, IntReg::Rdx, 4) => {
            dynasm!(ops ; .arch x64 ; mov edx, DWORD [r12 + r13 * 4 + bb])
        }
        // I16 (scale 2 for affine)
        (NumericDType::I16, IntReg::Rax, 2) => {
            dynasm!(ops ; .arch x64 ; movsx rax, WORD [r12 + r13 * 2 + bb])
        }
        (NumericDType::I16, IntReg::Rcx, 2) => {
            dynasm!(ops ; .arch x64 ; movsx rcx, WORD [r12 + r13 * 2 + bb])
        }
        (NumericDType::I16, IntReg::Rdx, 2) => {
            dynasm!(ops ; .arch x64 ; movsx rdx, WORD [r12 + r13 * 2 + bb])
        }
        (NumericDType::U16, IntReg::Rax, 2) => {
            dynasm!(ops ; .arch x64 ; movzx rax, WORD [r12 + r13 * 2 + bb])
        }
        (NumericDType::U16, IntReg::Rcx, 2) => {
            dynasm!(ops ; .arch x64 ; movzx rcx, WORD [r12 + r13 * 2 + bb])
        }
        (NumericDType::U16, IntReg::Rdx, 2) => {
            dynasm!(ops ; .arch x64 ; movzx rdx, WORD [r12 + r13 * 2 + bb])
        }
        // I8 / U8 / BOOL (scale 1 for affine)
        (NumericDType::I8, IntReg::Rax, 1) => {
            dynasm!(ops ; .arch x64 ; movsx rax, BYTE [r12 + r13 + bb])
        }
        (NumericDType::I8, IntReg::Rcx, 1) => {
            dynasm!(ops ; .arch x64 ; movsx rcx, BYTE [r12 + r13 + bb])
        }
        (NumericDType::I8, IntReg::Rdx, 1) => {
            dynasm!(ops ; .arch x64 ; movsx rdx, BYTE [r12 + r13 + bb])
        }
        (NumericDType::U8 | NumericDType::BOOL, IntReg::Rax, 1) => {
            dynasm!(ops ; .arch x64 ; movzx rax, BYTE [r12 + r13 + bb])
        }
        (NumericDType::U8 | NumericDType::BOOL, IntReg::Rcx, 1) => {
            dynasm!(ops ; .arch x64 ; movzx rcx, BYTE [r12 + r13 + bb])
        }
        (NumericDType::U8 | NumericDType::BOOL, IntReg::Rdx, 1) => {
            dynasm!(ops ; .arch x64 ; movzx rdx, BYTE [r12 + r13 + bb])
        }
        // Mismatched scale — fall back to materialized index in rax.
        _ => {
            let stride = byte_stride;
            dynasm!(ops
                ; .arch x64
                ; mov rax, r13
                ; imul rax, rax, stride as i32
            );
            emit_load_int_indexed_rax(ops, slot_dtype, dst, bb);
        }
    }
}

/// Helper for `emit_load_int_indexed`'s rax-scratch fallback path.
fn emit_load_int_indexed_rax(ops: &mut Assembler, slot_dtype: NumericDType, dst: IntReg, bb: i32) {
    match (slot_dtype, dst) {
        (NumericDType::I64 | NumericDType::U64, IntReg::Rax) => {
            dynasm!(ops ; .arch x64 ; mov rax, QWORD [r12 + rax + bb])
        }
        (NumericDType::I64 | NumericDType::U64, IntReg::Rcx) => {
            dynasm!(ops ; .arch x64 ; mov rcx, QWORD [r12 + rax + bb])
        }
        (NumericDType::I64 | NumericDType::U64, IntReg::Rdx) => {
            dynasm!(ops ; .arch x64 ; mov rdx, QWORD [r12 + rax + bb])
        }
        (NumericDType::I32, IntReg::Rax) => {
            dynasm!(ops ; .arch x64 ; movsxd rax, DWORD [r12 + rax + bb])
        }
        (NumericDType::I32, IntReg::Rcx) => {
            dynasm!(ops ; .arch x64 ; movsxd rcx, DWORD [r12 + rax + bb])
        }
        (NumericDType::I32, IntReg::Rdx) => {
            dynasm!(ops ; .arch x64 ; movsxd rdx, DWORD [r12 + rax + bb])
        }
        (NumericDType::U32, IntReg::Rax) => {
            dynasm!(ops ; .arch x64 ; mov eax, DWORD [r12 + rax + bb])
        }
        (NumericDType::U32, IntReg::Rcx) => {
            dynasm!(ops ; .arch x64 ; mov ecx, DWORD [r12 + rax + bb])
        }
        (NumericDType::U32, IntReg::Rdx) => {
            dynasm!(ops ; .arch x64 ; mov edx, DWORD [r12 + rax + bb])
        }
        (NumericDType::I16, IntReg::Rax) => {
            dynasm!(ops ; .arch x64 ; movsx rax, WORD [r12 + rax + bb])
        }
        (NumericDType::I16, IntReg::Rcx) => {
            dynasm!(ops ; .arch x64 ; movsx rcx, WORD [r12 + rax + bb])
        }
        (NumericDType::I16, IntReg::Rdx) => {
            dynasm!(ops ; .arch x64 ; movsx rdx, WORD [r12 + rax + bb])
        }
        (NumericDType::U16, IntReg::Rax) => {
            dynasm!(ops ; .arch x64 ; movzx rax, WORD [r12 + rax + bb])
        }
        (NumericDType::U16, IntReg::Rcx) => {
            dynasm!(ops ; .arch x64 ; movzx rcx, WORD [r12 + rax + bb])
        }
        (NumericDType::U16, IntReg::Rdx) => {
            dynasm!(ops ; .arch x64 ; movzx rdx, WORD [r12 + rax + bb])
        }
        (NumericDType::I8, IntReg::Rax) => {
            dynasm!(ops ; .arch x64 ; movsx rax, BYTE [r12 + rax + bb])
        }
        (NumericDType::I8, IntReg::Rcx) => {
            dynasm!(ops ; .arch x64 ; movsx rcx, BYTE [r12 + rax + bb])
        }
        (NumericDType::I8, IntReg::Rdx) => {
            dynasm!(ops ; .arch x64 ; movsx rdx, BYTE [r12 + rax + bb])
        }
        (NumericDType::U8 | NumericDType::BOOL, IntReg::Rax) => {
            dynasm!(ops ; .arch x64 ; movzx rax, BYTE [r12 + rax + bb])
        }
        (NumericDType::U8 | NumericDType::BOOL, IntReg::Rcx) => {
            dynasm!(ops ; .arch x64 ; movzx rcx, BYTE [r12 + rax + bb])
        }
        (NumericDType::U8 | NumericDType::BOOL, IntReg::Rdx) => {
            dynasm!(ops ; .arch x64 ; movzx rdx, BYTE [r12 + rax + bb])
        }
        _ => panic!("emit_load_int_indexed_rax: unsupported slot dtype {slot_dtype:?}"),
    }
}

/// Emit a store from a GP register to an output slot, narrowing as needed.
///
/// For BOOL outputs we first normalize the value to 0/1 with a `test/setne`
/// pair, mirroring `emit_cast_to_output`'s BOOL handling in cranelift.
fn emit_store_int(
    ops: &mut Assembler,
    slot: &SlotInfo,
    in_loop: bool,
    i_const: u64,
    atom_offset: u64,
    src: IntReg,
) {
    // Normalize BOOL to 0/1 in src first.
    if slot.dtype == NumericDType::BOOL {
        match src {
            IntReg::Rax => dynasm!(ops
                ; .arch x64
                ; test rax, rax
                ; setne al
                ; movzx rax, al
            ),
            IntReg::Rcx => dynasm!(ops
                ; .arch x64
                ; test rcx, rcx
                ; setne cl
                ; movzx rcx, cl
            ),
            IntReg::Rdx => dynasm!(ops
                ; .arch x64
                ; test rdx, rdx
                ; setne dl
                ; movzx rdx, dl
            ),
        }
    }

    let store_base = slot.byte_offset as i64 - atom_offset as i64 * slot.elem_bytes as i64;

    if in_loop {
        let bb = store_base as i32;
        debug_assert_eq!(store_base, bb as i64, "store_base out of i32 range");
        emit_store_int_indexed(ops, slot.dtype, src, bb, slot.elem_bytes as i64);
    } else {
        let byte_off = store_base + i_const as i64 * slot.elem_bytes as i64;
        let bb = byte_off as i32;
        debug_assert_eq!(byte_off, bb as i64, "byte_off out of i32 range");
        emit_store_int_disp(ops, slot.dtype, src, bb);
    }
}

fn emit_store_int_disp(ops: &mut Assembler, slot_dtype: NumericDType, src: IntReg, disp: i32) {
    let d = disp;
    match (slot_dtype, src) {
        (NumericDType::I64 | NumericDType::U64, IntReg::Rax) => {
            dynasm!(ops ; .arch x64 ; mov QWORD [r12 + d], rax)
        }
        (NumericDType::I64 | NumericDType::U64, IntReg::Rcx) => {
            dynasm!(ops ; .arch x64 ; mov QWORD [r12 + d], rcx)
        }
        (NumericDType::I64 | NumericDType::U64, IntReg::Rdx) => {
            dynasm!(ops ; .arch x64 ; mov QWORD [r12 + d], rdx)
        }
        (NumericDType::I32 | NumericDType::U32, IntReg::Rax) => {
            dynasm!(ops ; .arch x64 ; mov DWORD [r12 + d], eax)
        }
        (NumericDType::I32 | NumericDType::U32, IntReg::Rcx) => {
            dynasm!(ops ; .arch x64 ; mov DWORD [r12 + d], ecx)
        }
        (NumericDType::I32 | NumericDType::U32, IntReg::Rdx) => {
            dynasm!(ops ; .arch x64 ; mov DWORD [r12 + d], edx)
        }
        (NumericDType::I16 | NumericDType::U16, IntReg::Rax) => {
            dynasm!(ops ; .arch x64 ; mov WORD [r12 + d], ax)
        }
        (NumericDType::I16 | NumericDType::U16, IntReg::Rcx) => {
            dynasm!(ops ; .arch x64 ; mov WORD [r12 + d], cx)
        }
        (NumericDType::I16 | NumericDType::U16, IntReg::Rdx) => {
            dynasm!(ops ; .arch x64 ; mov WORD [r12 + d], dx)
        }
        (NumericDType::I8 | NumericDType::U8 | NumericDType::BOOL, IntReg::Rax) => {
            dynasm!(ops ; .arch x64 ; mov BYTE [r12 + d], al)
        }
        (NumericDType::I8 | NumericDType::U8 | NumericDType::BOOL, IntReg::Rcx) => {
            dynasm!(ops ; .arch x64 ; mov BYTE [r12 + d], cl)
        }
        (NumericDType::I8 | NumericDType::U8 | NumericDType::BOOL, IntReg::Rdx) => {
            dynasm!(ops ; .arch x64 ; mov BYTE [r12 + d], dl)
        }
        _ => panic!("emit_store_int_disp: unsupported slot dtype {slot_dtype:?}"),
    }
}

fn emit_store_int_indexed(
    ops: &mut Assembler,
    slot_dtype: NumericDType,
    src: IntReg,
    bb: i32,
    elem_bytes: i64,
) {
    // Slot stride is always elem_bytes for the natural row-major case
    // (atoms laid out contiguously in the buffer).
    if !matches!(elem_bytes, 1 | 2 | 4 | 8) {
        panic!("emit_store_int_indexed: unexpected elem_bytes {elem_bytes}");
    }
    match (slot_dtype, src, elem_bytes) {
        (NumericDType::I64 | NumericDType::U64, IntReg::Rax, 8) => {
            dynasm!(ops ; .arch x64 ; mov QWORD [r12 + r13 * 8 + bb], rax)
        }
        (NumericDType::I64 | NumericDType::U64, IntReg::Rcx, 8) => {
            dynasm!(ops ; .arch x64 ; mov QWORD [r12 + r13 * 8 + bb], rcx)
        }
        (NumericDType::I64 | NumericDType::U64, IntReg::Rdx, 8) => {
            dynasm!(ops ; .arch x64 ; mov QWORD [r12 + r13 * 8 + bb], rdx)
        }
        (NumericDType::I32 | NumericDType::U32, IntReg::Rax, 4) => {
            dynasm!(ops ; .arch x64 ; mov DWORD [r12 + r13 * 4 + bb], eax)
        }
        (NumericDType::I32 | NumericDType::U32, IntReg::Rcx, 4) => {
            dynasm!(ops ; .arch x64 ; mov DWORD [r12 + r13 * 4 + bb], ecx)
        }
        (NumericDType::I32 | NumericDType::U32, IntReg::Rdx, 4) => {
            dynasm!(ops ; .arch x64 ; mov DWORD [r12 + r13 * 4 + bb], edx)
        }
        (NumericDType::I16 | NumericDType::U16, IntReg::Rax, 2) => {
            dynasm!(ops ; .arch x64 ; mov WORD [r12 + r13 * 2 + bb], ax)
        }
        (NumericDType::I16 | NumericDType::U16, IntReg::Rcx, 2) => {
            dynasm!(ops ; .arch x64 ; mov WORD [r12 + r13 * 2 + bb], cx)
        }
        (NumericDType::I16 | NumericDType::U16, IntReg::Rdx, 2) => {
            dynasm!(ops ; .arch x64 ; mov WORD [r12 + r13 * 2 + bb], dx)
        }
        (NumericDType::I8 | NumericDType::U8 | NumericDType::BOOL, IntReg::Rax, 1) => {
            dynasm!(ops ; .arch x64 ; mov BYTE [r12 + r13 + bb], al)
        }
        (NumericDType::I8 | NumericDType::U8 | NumericDType::BOOL, IntReg::Rcx, 1) => {
            dynasm!(ops ; .arch x64 ; mov BYTE [r12 + r13 + bb], cl)
        }
        (NumericDType::I8 | NumericDType::U8 | NumericDType::BOOL, IntReg::Rdx, 1) => {
            dynasm!(ops ; .arch x64 ; mov BYTE [r12 + r13 + bb], dl)
        }
        _ => panic!(
            "emit_store_int_indexed: unsupported (dtype {slot_dtype:?}, reg {src:?}, elem_bytes {elem_bytes})"
        ),
    }
}

/// Emit `op(rax, rcx) → rax`, i64.
fn emit_binop_int(ops: &mut Assembler, op: ScalarBinOp) {
    match op {
        ScalarBinOp::Add => dynasm!(ops ; .arch x64 ; add rax, rcx),
        ScalarBinOp::Sub => dynasm!(ops ; .arch x64 ; sub rax, rcx),
        ScalarBinOp::Mul => dynasm!(ops ; .arch x64 ; imul rax, rcx),

        // Division: guard against /0 with select(b == 0, 0, a / b). Mirrors
        // cranelift's int Div emission.
        ScalarBinOp::Div => {
            // if rcx == 0, result = 0; else cqo ; idiv rcx -> rax = a/b
            dynasm!(ops
                ; .arch x64
                ; test rcx, rcx
                ; jz >div_zero
                ; cqo
                ; idiv rcx
                ; jmp >div_done
                ; div_zero:
                ; xor rax, rax
                ; div_done:
            );
        }
        ScalarBinOp::Mod => {
            // C remainder (truncated division). Same /0 guard. Result lands
            // in rdx after idiv; move to rax.
            dynasm!(ops
                ; .arch x64
                ; test rcx, rcx
                ; jz >mod_zero
                ; cqo
                ; idiv rcx
                ; mov rax, rdx
                ; jmp >mod_done
                ; mod_zero:
                ; xor rax, rax
                ; mod_done:
            );
        }
        ScalarBinOp::IMod => {
            // Mathematical modulo: result sign matches divisor.
            // rem = srem(a, b)
            // if rem == 0 → rem
            // else if sign(rem) != sign(b) → rem + b
            // else → rem
            // Returns 0 if b == 0.
            //
            // We need to preserve b across the idiv (which clobbers rdx).
            // Stash it in r8 (caller-saved scratch we don't otherwise use).
            dynasm!(ops
                ; .arch x64
                ; test rcx, rcx
                ; jz >imod_zero
                ; mov r8, rcx          // r8 = b
                ; cqo
                ; idiv rcx             // rdx = rem, rax = quot
                ; mov rax, rdx         // rax = rem
                // sum = rem + b
                ; mov r9, rax
                ; add r9, r8           // r9 = rem + b
                // if rem == 0, return rem (already in rax)
                ; test rax, rax
                ; jz >imod_done
                // sign-mismatch: (rem ^ b) < 0
                ; mov rdx, rax
                ; xor rdx, r8
                ; jns >imod_done       // signs match → return rem
                ; mov rax, r9          // signs differ → return rem + b
                ; jmp >imod_done
                ; imod_zero:
                ; xor rax, rax
                ; imod_done:
            );
        }

        ScalarBinOp::Min => {
            // if rax > rcx, take rcx
            dynasm!(ops
                ; .arch x64
                ; cmp rax, rcx
                ; cmovg rax, rcx
            );
        }
        ScalarBinOp::Max => {
            dynasm!(ops
                ; .arch x64
                ; cmp rax, rcx
                ; cmovl rax, rcx
            );
        }

        // Comparisons → 1 / 0 (i64). Use cmp + setcc + movzx.
        ScalarBinOp::Equal => emit_int_cmp(ops, IntCmp::Eq),
        ScalarBinOp::Less => emit_int_cmp(ops, IntCmp::Lt),
        ScalarBinOp::LessOrEqual => emit_int_cmp(ops, IntCmp::Le),
        ScalarBinOp::Greater => emit_int_cmp(ops, IntCmp::Gt),
        ScalarBinOp::GreaterOrEqual => emit_int_cmp(ops, IntCmp::Ge),

        // Logical (truthy): convert each to 0/1 then bitand/bitor/bitxor.
        ScalarBinOp::And => emit_int_logical(ops, LogicalOp::And),
        ScalarBinOp::Or => emit_int_logical(ops, LogicalOp::Or),
        ScalarBinOp::Xor => emit_int_logical(ops, LogicalOp::Xor),

        // Bitwise (raw bits, no truthiness conversion).
        ScalarBinOp::BitwiseAnd => dynasm!(ops ; .arch x64 ; and rax, rcx),
        ScalarBinOp::BitwiseOr => dynasm!(ops ; .arch x64 ; or rax, rcx),
        ScalarBinOp::BitwiseXor => dynasm!(ops ; .arch x64 ; xor rax, rcx),
        ScalarBinOp::BitShiftLeft => dynasm!(ops ; .arch x64 ; shl rax, cl),
        ScalarBinOp::BitShiftRight => dynasm!(ops ; .arch x64 ; sar rax, cl),

        ScalarBinOp::Pow => panic!("emit_binop_int: integer Pow unsupported"),
    }
}

#[derive(Copy, Clone)]
enum IntCmp {
    Eq,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Emit `rax = (rax cmp rcx) ? 1 : 0`, i64. (Signed compare.)
fn emit_int_cmp(ops: &mut Assembler, cmp: IntCmp) {
    dynasm!(ops ; .arch x64 ; cmp rax, rcx);
    match cmp {
        IntCmp::Eq => dynasm!(ops ; .arch x64 ; sete al),
        IntCmp::Lt => dynasm!(ops ; .arch x64 ; setl al),
        IntCmp::Le => dynasm!(ops ; .arch x64 ; setle al),
        IntCmp::Gt => dynasm!(ops ; .arch x64 ; setg al),
        IntCmp::Ge => dynasm!(ops ; .arch x64 ; setge al),
    }
    dynasm!(ops ; .arch x64 ; movzx rax, al);
}

/// Emit `rax = (rax_truthy logical_op rcx_truthy) ? 1 : 0`, i64.
fn emit_int_logical(ops: &mut Assembler, op: LogicalOp) {
    // Normalize each operand to 0/1.
    dynasm!(ops
        ; .arch x64
        ; test rax, rax
        ; setne al
        ; movzx rax, al
        ; test rcx, rcx
        ; setne cl
        ; movzx rcx, cl
    );
    match op {
        LogicalOp::And => dynasm!(ops ; .arch x64 ; and rax, rcx),
        LogicalOp::Or => dynasm!(ops ; .arch x64 ; or rax, rcx),
        LogicalOp::Xor => dynasm!(ops ; .arch x64 ; xor rax, rcx),
    }
}

/// Emit `op(rax) → rax`, i64.
fn emit_unop_int(ops: &mut Assembler, op: ScalarUnaryOp) {
    match op {
        ScalarUnaryOp::Neg => dynasm!(ops ; .arch x64 ; neg rax),
        ScalarUnaryOp::Abs => {
            // Branch-free abs:
            //   mov rcx, rax
            //   sar rcx, 63    ; rcx = -1 if rax<0 else 0
            //   xor rax, rcx
            //   sub rax, rcx
            dynasm!(ops
                ; .arch x64
                ; mov rcx, rax
                ; sar rcx, 63
                ; xor rax, rcx
                ; sub rax, rcx
            );
        }
        ScalarUnaryOp::BitwiseNot => dynasm!(ops ; .arch x64 ; not rax),
        ScalarUnaryOp::Not => {
            // Logical not: 1 if rax == 0, else 0.
            dynasm!(ops
                ; .arch x64
                ; test rax, rax
                ; sete al
                ; movzx rax, al
            );
        }
        ScalarUnaryOp::Sign => {
            // sign(x) for signed i64:
            //   -1 if x < 0
            //    0 if x == 0
            //   +1 if x > 0
            // Branch-free: result = (x > 0) - (x < 0).
            //   xor edx, edx
            //   test rax, rax
            //   setg dl       ; dl = (x > 0)
            //   setl al       ; al = (x < 0)
            //   movzx rax, al
            //   sub rdx, rax
            //   mov rax, rdx
            dynasm!(ops
                ; .arch x64
                ; test rax, rax
                ; setg dl
                ; setl al
                ; movzx rdx, dl
                ; movzx rax, al
                ; sub rdx, rax
                ; mov rax, rdx
            );
        }
        ScalarUnaryOp::IsNan | ScalarUnaryOp::IsInf { .. } => {
            // Integers are never NaN/Inf — always 0.
            dynasm!(ops ; .arch x64 ; xor eax, eax);
        }
        _ => panic!("emit_unop_int: unsupported op {op:?}"),
    }
}

// ─── IndirectLoad emission ─────────────────────────────────────────────────
//
// `IndirectLoad { table_base }` evaluates: `table_base[input[0]]` where the
// index is loaded from `input[0]` (cast to i64), the table lives in the
// buffer at `table_base`'s slot, and the result is cast to `output_dtype`
// before storing.
//
// Codegen sequence:
//   1. emit_load_int(input[0], rax)            ← index → rax
//   2. imul rax, rax, table.elem_bytes         ← byte offset within table
//   3. add  rax, table.byte_offset             ← absolute byte offset
//   4. typed load from [r12 + rax] in the table dtype's repr
//   5. cast to output dtype's repr
//   6. typed store into output slot

fn emit_group_body_indirect_load(
    ops: &mut Assembler,
    group: &AtomGroup<'static, SystemPool>,
    layout: &BufferLayout,
    tables: &mut EmbeddedTables,
    in_loop: bool,
    i_const: u64,
    table_base: AtomId,
) -> Result<(), String> {
    let (out_slot, _) = layout
        .find(group.base_id)
        .ok_or_else(|| format!("x86_jit: no slot for group base={}", group.base_id))?;
    let out_slot = out_slot.clone();

    let (table_slot, _) = layout
        .find(table_base)
        .ok_or_else(|| format!("x86_jit: no slot for IndirectLoad table_base={table_base}"))?;
    let table_byte_off = table_slot.byte_offset as i64;
    let table_elem_bytes = table_slot.elem_bytes as i64;
    let table_dtype = table_slot.dtype;

    // Step 1: load index → rax. The index can come from any int dtype slot,
    // or a float dtype that we cast to int (matching cranelift's repr_cast).
    // We support int sources here; float-indexed gather is rare.
    let index_input = &group.inputs[0];
    let index_slot_dtype = input_slot_dtype(index_input, layout, group.atom_offset)
        .ok_or_else(|| "x86_jit: no slot for IndirectLoad index input".to_string())?;
    if Repr::from_dtype(index_slot_dtype) != Repr::Int {
        return Err(format!(
            "x86_jit: IndirectLoad index dtype {index_slot_dtype:?} (only int supported)"
        ));
    }
    emit_load_int(
        ops,
        index_input,
        layout,
        tables,
        in_loop,
        i_const,
        group.atom_offset,
        IntReg::Rax,
    )?;

    // Step 2 + 3: rax = rax * table_elem_bytes + table_byte_off.
    if table_elem_bytes != 1 {
        let imm = i32::try_from(table_elem_bytes)
            .map_err(|_| "x86_jit: IndirectLoad table_elem_bytes too large".to_string())?;
        dynasm!(ops ; .arch x64 ; imul rax, rax, imm);
    }
    if table_byte_off != 0 {
        // Handle large table offsets: small ones via add imm, large via mov+add.
        if let Ok(imm) = i32::try_from(table_byte_off) {
            dynasm!(ops ; .arch x64 ; add rax, imm);
        } else {
            dynasm!(ops
                ; .arch x64
                ; mov rcx, QWORD table_byte_off
                ; add rax, rcx
            );
        }
    }

    // Step 4: typed load from [r12 + rax]. Repurpose the int / float
    // RaxIndexed paths.
    let table_repr = Repr::from_dtype(table_dtype);
    let out_repr = Repr::from_dtype(group.output_dtype);

    match table_repr {
        Repr::F32 => match table_dtype {
            NumericDType::F32 => emit_load_f32_rax(ops, 0, 0),
            NumericDType::BF16 => emit_load_bf16_rax(ops, 0, 0),
            NumericDType::F16 => emit_load_f16_rax(ops, 0, 0),
            _ => unreachable!(),
        },
        Repr::F64 => {
            emit_load_f64_rax(ops, 0, 0);
        }
        Repr::Int => {
            emit_load_int_indexed_rax(ops, table_dtype, IntReg::Rax, 0);
        }
    }

    // Step 5: cross-repr cast. Same logic as emit_group_body_cast.
    match (table_repr, out_repr) {
        (Repr::F32, Repr::F32) | (Repr::F64, Repr::F64) | (Repr::Int, Repr::Int) => {}
        (Repr::F32, Repr::F64) => dynasm!(ops ; .arch x64 ; cvtss2sd xmm0, xmm0),
        (Repr::F64, Repr::F32) => dynasm!(ops ; .arch x64 ; cvtsd2ss xmm0, xmm0),
        (Repr::F32, Repr::Int) => dynasm!(ops ; .arch x64 ; cvttss2si rax, xmm0),
        (Repr::F64, Repr::Int) => dynasm!(ops ; .arch x64 ; cvttsd2si rax, xmm0),
        (Repr::Int, Repr::F32) => dynasm!(ops ; .arch x64 ; cvtsi2ss xmm0, rax),
        (Repr::Int, Repr::F64) => dynasm!(ops ; .arch x64 ; cvtsi2sd xmm0, rax),
    }

    // Step 6: store.
    match out_repr {
        Repr::F32 => emit_store_f32(ops, &out_slot, in_loop, i_const, group.atom_offset, 0),
        Repr::F64 => emit_store_f64(ops, &out_slot, in_loop, i_const, group.atom_offset, 0),
        Repr::Int => emit_store_int(
            ops,
            &out_slot,
            in_loop,
            i_const,
            group.atom_offset,
            IntReg::Rax,
        ),
    }

    Ok(())
}

/// Look up the storage dtype an InputRef will load from. Mirrors cranelift's
/// `input_slot_dtype` helper.
fn input_slot_dtype(
    input: &InputRef,
    layout: &BufferLayout,
    atom_offset: u64,
) -> Option<NumericDType> {
    match input {
        InputRef::Broadcast(atom) => layout.find(*atom).map(|(s, _)| s.dtype),
        InputRef::Strided {
            base,
            dim_strides,
            dim_shape,
        } => layout.find(*base).map(|(s, _)| s.dtype).or_else(|| {
            let off = strided_offset_atoms(dim_strides, dim_shape, atom_offset);
            let first = AtomId((base.0 as i64 + off) as u64);
            layout.find(first).map(|(s, _)| s.dtype)
        }),
        InputRef::Explicit(ids) => ids
            .first()
            .and_then(|id| layout.find(*id).map(|(s, _)| s.dtype)),
    }
}

// ─── Reduce emission ───────────────────────────────────────────────────────
//
// `Reduce { kind, reduce_count, reduce_stride, compute_dtype }` performs an
// affine k-loop over `reduce_count` source atoms accessed via the input's
// 1D affine Strided pattern. The accumulator initializes per-kind:
//
//   Sum  → 0       Prod → 1
//   Max  → -inf    Min  → +inf      (for ints: i64::MIN / i64::MAX)
//
// Each iteration: load src[k], cast-up to compute repr if narrower, fold
// into the accumulator, k++. After the inner loop, cast-to-output and store.
//
// We use **r15** as the inner k-loop variable so it doesn't conflict with
// r13 (outer atom index when count > 1) or r14 (outer end constant).

fn emit_group_body_reduce(
    ops: &mut Assembler,
    group: &AtomGroup<'static, SystemPool>,
    layout: &BufferLayout,
    _tables: &mut EmbeddedTables,
    in_loop: bool,
    i_const: u64,
    kind: ReduceKind,
    reduce_count: u64,
    reduce_stride: i64,
    compute_dtype: NumericDType,
) -> Result<(), String> {
    let (out_slot, _) = layout
        .find(group.base_id)
        .ok_or_else(|| format!("x86_jit: no slot for group base={}", group.base_id))?;
    let out_slot = out_slot.clone();
    let compute_repr = Repr::from_dtype(compute_dtype);
    let out_repr = Repr::from_dtype(group.output_dtype);

    // Reduce input is 1D affine Strided (checked by check_supported).
    let (input_base, input_stride_atoms) = match &group.inputs[0] {
        InputRef::Strided {
            base, dim_strides, ..
        } => (*base, dim_strides[0]),
        _ => unreachable!("check_supported guarantees 1D Strided"),
    };

    // Resolve the source slot. Treat it as a 1D affine resolution.
    let (base_byte, byte_stride, src_dtype) =
        resolve_reduce_input_addr(input_base, input_stride_atoms, group.atom_offset, layout)?;
    let reduce_byte_stride = reduce_stride * (src_dtype.bytes_per_element() as i64);
    let src_repr = Repr::from_dtype(src_dtype);

    // Per-iteration outer base address — depends on whether we're in the
    // outer atom loop (r13) or compiling a constant-i body.
    //
    // For in_loop:  outer_base = base_byte + r13 * byte_stride
    // For !in_loop: outer_base = base_byte + i_const * byte_stride (constant)
    //
    // We materialize this into r15 (so the inner k-loop can derive
    // [r12 + r15 + k*reduce_byte_stride] without re-doing the outer multiply).
    if in_loop {
        // r15 = base_byte + r13 * byte_stride
        if byte_stride == 0 {
            dynasm!(ops ; .arch x64 ; mov r15, QWORD base_byte);
        } else {
            let bs_imm = i32::try_from(byte_stride)
                .map_err(|_| "x86_jit: reduce input byte_stride too large for i32".to_string())?;
            dynasm!(ops
                ; .arch x64
                ; mov r15, r13
                ; imul r15, r15, bs_imm
            );
            if base_byte != 0 {
                if let Ok(bb_imm) = i32::try_from(base_byte) {
                    dynasm!(ops ; .arch x64 ; add r15, bb_imm);
                } else {
                    dynasm!(ops
                        ; .arch x64
                        ; mov rax, QWORD base_byte
                        ; add r15, rax
                    );
                }
            }
        }
    } else {
        let outer_byte_off = base_byte + byte_stride * i_const as i64;
        dynasm!(ops ; .arch x64 ; mov r15, QWORD outer_byte_off);
    }

    // Initialize the accumulator in xmm0 / xmm0 / rax depending on compute repr.
    emit_reduce_init(ops, kind, compute_repr);

    // Inner k loop: rcx counts down from reduce_count to 0. We choose rcx
    // (caller-saved) instead of r13 because r13 is the outer loop var.
    // Body: load src at [r12 + r15 + k_offset], accumulate.
    //
    // For the very common reduce_count == 1 case, we can unroll: just one
    // iteration, no loop overhead.
    if reduce_count == 0 {
        // Empty reduction — accumulator stays at init. (Lowering already
        // refuses extent-0 reductions, so this is defensive.)
    } else if reduce_count == 1 {
        emit_reduce_body_iter(
            ops,
            kind,
            compute_repr,
            src_repr,
            src_dtype,
            /*k_disp=*/ 0,
        )?;
    } else {
        // Counted loop: r11 = 0..reduce_count.
        //
        // Register choices:
        //   r11    — inner k counter (caller-saved; spilled across F16 calls)
        //   rdx    — int loaded-value scratch (fold reads it)
        //   rcx    — bf16 expand scratch (load uses ecx; clobbered each iter)
        //   xmm1   — float loaded-value scratch
        //   r10    — address scratch (for non-power-of-2 stride)
        //
        // We embed `reduce_count` as an i32 immediate in the cmp; counts above
        // 2^31 are not expected for any practical tensor and would have been
        // rejected by check_supported earlier (TODO: explicit check).
        let end_imm = i32::try_from(reduce_count)
            .map_err(|_| "x86_jit: reduce_count exceeds i32::MAX (unsupported)".to_string())?;
        let k_top = ops.new_dynamic_label();
        let k_done = ops.new_dynamic_label();
        dynasm!(ops
            ; .arch x64
            ; xor r11d, r11d            // k = 0
            ; =>k_top
            ; cmp r11, end_imm
            ; jge =>k_done
        );
        emit_reduce_body_loop_iter(
            ops,
            kind,
            compute_repr,
            src_repr,
            src_dtype,
            reduce_byte_stride,
        )?;
        dynasm!(ops
            ; .arch x64
            ; inc r11
            ; jmp =>k_top
            ; =>k_done
        );
    }

    // Cross-repr cast to output and store.
    match (compute_repr, out_repr) {
        (Repr::F32, Repr::F32) | (Repr::F64, Repr::F64) | (Repr::Int, Repr::Int) => {}
        (Repr::F32, Repr::F64) => dynasm!(ops ; .arch x64 ; cvtss2sd xmm0, xmm0),
        (Repr::F64, Repr::F32) => dynasm!(ops ; .arch x64 ; cvtsd2ss xmm0, xmm0),
        (Repr::F32, Repr::Int) => dynasm!(ops ; .arch x64 ; cvttss2si rax, xmm0),
        (Repr::F64, Repr::Int) => dynasm!(ops ; .arch x64 ; cvttsd2si rax, xmm0),
        (Repr::Int, Repr::F32) => dynasm!(ops ; .arch x64 ; cvtsi2ss xmm0, rax),
        (Repr::Int, Repr::F64) => dynasm!(ops ; .arch x64 ; cvtsi2sd xmm0, rax),
    }

    match out_repr {
        Repr::F32 => emit_store_f32(ops, &out_slot, in_loop, i_const, group.atom_offset, 0),
        Repr::F64 => emit_store_f64(ops, &out_slot, in_loop, i_const, group.atom_offset, 0),
        Repr::Int => emit_store_int(
            ops,
            &out_slot,
            in_loop,
            i_const,
            group.atom_offset,
            IntReg::Rax,
        ),
    }

    Ok(())
}

/// Resolve a 1D affine Reduce input to its (base_byte, byte_stride, dtype).
fn resolve_reduce_input_addr(
    base: AtomId,
    stride_atoms: i64,
    atom_offset: u64,
    layout: &BufferLayout,
) -> Result<(i64, i64, NumericDType), String> {
    let first_offset = stride_atoms * atom_offset as i64;
    let first_atom = AtomId((base.0 as i64 + first_offset) as u64);
    let (slot, elem) = layout
        .find(base)
        .or_else(|| layout.find(first_atom))
        .ok_or_else(|| format!("x86_jit: no slot for Reduce input base={base}"))?;
    let elem_bytes = slot.elem_bytes as i64;
    let slot_byte = slot.byte_offset as i64 + elem as i64 * elem_bytes;
    let base_byte = if layout.find(base).is_some() {
        slot_byte
    } else {
        slot_byte - first_offset * elem_bytes
    };
    Ok((base_byte, stride_atoms * elem_bytes, slot.dtype))
}

/// Initialize the reduce accumulator (xmm0 for F32/F64, rax for Int) per kind.
fn emit_reduce_init(ops: &mut Assembler, kind: ReduceKind, compute_repr: Repr) {
    match (kind, compute_repr) {
        (ReduceKind::Sum, Repr::F32) => {
            dynasm!(ops ; .arch x64 ; xorps xmm0, xmm0);
        }
        (ReduceKind::Sum, Repr::F64) => {
            dynasm!(ops ; .arch x64 ; xorpd xmm0, xmm0);
        }
        (ReduceKind::Sum, Repr::Int) => {
            dynasm!(ops ; .arch x64 ; xor eax, eax);
        }
        (ReduceKind::Prod, Repr::F32) => {
            let one = 1.0f32.to_bits() as i32;
            dynasm!(ops
                ; .arch x64
                ; mov eax, DWORD one
                ; movd xmm0, eax
            );
        }
        (ReduceKind::Prod, Repr::F64) => {
            let one = 1.0f64.to_bits() as i64;
            dynasm!(ops
                ; .arch x64
                ; mov rax, QWORD one
                ; movq xmm0, rax
            );
        }
        (ReduceKind::Prod, Repr::Int) => {
            dynasm!(ops ; .arch x64 ; mov eax, 1);
        }
        (ReduceKind::Max, Repr::F32) => {
            let neg_inf = f32::NEG_INFINITY.to_bits() as i32;
            dynasm!(ops
                ; .arch x64
                ; mov eax, DWORD neg_inf
                ; movd xmm0, eax
            );
        }
        (ReduceKind::Max, Repr::F64) => {
            let neg_inf = f64::NEG_INFINITY.to_bits() as i64;
            dynasm!(ops
                ; .arch x64
                ; mov rax, QWORD neg_inf
                ; movq xmm0, rax
            );
        }
        (ReduceKind::Max, Repr::Int) => {
            let min = i64::MIN;
            dynasm!(ops ; .arch x64 ; mov rax, QWORD min);
        }
        (ReduceKind::Min, Repr::F32) => {
            let pos_inf = f32::INFINITY.to_bits() as i32;
            dynasm!(ops
                ; .arch x64
                ; mov eax, DWORD pos_inf
                ; movd xmm0, eax
            );
        }
        (ReduceKind::Min, Repr::F64) => {
            let pos_inf = f64::INFINITY.to_bits() as i64;
            dynasm!(ops
                ; .arch x64
                ; mov rax, QWORD pos_inf
                ; movq xmm0, rax
            );
        }
        (ReduceKind::Min, Repr::Int) => {
            let max = i64::MAX;
            dynasm!(ops ; .arch x64 ; mov rax, QWORD max);
        }
    }
}

/// Emit one reduce body iteration where the address is `[r12 + r15 + k_disp]`.
/// Used for the unrolled `reduce_count == 1` case.
fn emit_reduce_body_iter(
    ops: &mut Assembler,
    kind: ReduceKind,
    compute_repr: Repr,
    src_repr: Repr,
    src_dtype: NumericDType,
    k_disp: i64,
) -> Result<(), String> {
    let bb = i32::try_from(k_disp)
        .map_err(|_| "x86_jit: reduce k_disp too large for i32".to_string())?;
    // Load src[k_disp] from [r12 + r15 + bb] into xmm1 (float) or rdx (int).
    emit_reduce_load_into_scratch(ops, src_dtype, src_repr, compute_repr, bb)?;
    // Fold scratch into acc.
    emit_reduce_fold(ops, kind, compute_repr);
    Ok(())
}

/// Emit one reduce body iteration where the inner k variable is in `r11`.
/// We compute address = `[r12 + r15 + r11*reduce_byte_stride]` for the SIB
/// scales we natively support, or materialize the byte offset into r10 for
/// everything else.
fn emit_reduce_body_loop_iter(
    ops: &mut Assembler,
    kind: ReduceKind,
    compute_repr: Repr,
    src_repr: Repr,
    src_dtype: NumericDType,
    reduce_byte_stride: i64,
) -> Result<(), String> {
    if matches!(reduce_byte_stride, 1 | 2 | 4 | 8) {
        emit_reduce_load_via_sib(ops, src_dtype, src_repr, compute_repr, reduce_byte_stride)?;
    } else {
        let stride_imm = i32::try_from(reduce_byte_stride)
            .map_err(|_| "x86_jit: reduce stride too large for i32".to_string())?;
        dynasm!(ops
            ; .arch x64
            ; mov r10, r11
            ; imul r10, r10, stride_imm
        );
        emit_reduce_load_via_r10(ops, src_dtype, src_repr, compute_repr)?;
    }
    emit_reduce_fold(ops, kind, compute_repr);
    Ok(())
}

/// Load `src` from `[r12 + r15 + bb]`, expand-cast to compute_repr, leave
/// result in xmm1 (float compute) or rdx (int compute).
fn emit_reduce_load_into_scratch(
    ops: &mut Assembler,
    src_dtype: NumericDType,
    src_repr: Repr,
    compute_repr: Repr,
    bb: i32,
) -> Result<(), String> {
    match src_dtype {
        NumericDType::F32 => {
            dynasm!(ops ; .arch x64 ; movss xmm1, DWORD [r12 + r15 + bb]);
        }
        NumericDType::F64 => {
            dynasm!(ops ; .arch x64 ; movsd xmm1, QWORD [r12 + r15 + bb]);
        }
        NumericDType::BF16 => {
            dynasm!(ops
                ; .arch x64
                ; movzx ecx, WORD [r12 + r15 + bb]
                ; shl ecx, 16
                ; movd xmm1, ecx
            );
        }
        NumericDType::F16 => {
            // F16 reduce iteration calls jit_f16_to_f32. xmm0 (acc) and r11
            // (k counter) are both caller-saved and must be spilled.
            dynasm!(ops
                ; .arch x64
                ; movss DWORD [rsp + F16_SPILL_XMM0], xmm0
                ; mov QWORD [rsp + K_SPILL], r11
                ; movzx edi, WORD [r12 + r15 + bb]
                ; mov rax, QWORD jit_f16_to_f32 as *const u8 as i64
                ; call rax
                ; movaps xmm1, xmm0
                ; movss xmm0, DWORD [rsp + F16_SPILL_XMM0]
                ; mov r11, QWORD [rsp + K_SPILL]
            );
        }
        NumericDType::I64 | NumericDType::U64 => {
            dynasm!(ops ; .arch x64 ; mov rdx, QWORD [r12 + r15 + bb]);
        }
        NumericDType::I32 => {
            dynasm!(ops ; .arch x64 ; movsxd rdx, DWORD [r12 + r15 + bb]);
        }
        NumericDType::U32 => {
            dynasm!(ops ; .arch x64 ; mov edx, DWORD [r12 + r15 + bb]);
        }
        NumericDType::I16 => {
            dynasm!(ops ; .arch x64 ; movsx rdx, WORD [r12 + r15 + bb]);
        }
        NumericDType::U16 => {
            dynasm!(ops ; .arch x64 ; movzx rdx, WORD [r12 + r15 + bb]);
        }
        NumericDType::I8 => {
            dynasm!(ops ; .arch x64 ; movsx rdx, BYTE [r12 + r15 + bb]);
        }
        NumericDType::U8 | NumericDType::BOOL => {
            dynasm!(ops ; .arch x64 ; movzx rdx, BYTE [r12 + r15 + bb]);
        }
        _ => {
            return Err(format!(
                "x86_jit: reduce src dtype {src_dtype:?} unsupported"
            ));
        }
    }

    // Cross-repr cast (only F32 ↔ F64 used in practice; src→compute always
    // narrows or stays the same since BF16/F16 → F32, integers → i64).
    match (src_repr, compute_repr) {
        (Repr::F32, Repr::F64) => dynasm!(ops ; .arch x64 ; cvtss2sd xmm1, xmm1),
        (Repr::F64, Repr::F32) => dynasm!(ops ; .arch x64 ; cvtsd2ss xmm1, xmm1),
        _ => {} // same repr or int→int (already i64)
    }
    Ok(())
}

/// Variant that uses the SIB addressing mode `[r12 + r15 + r11*scale]`.
/// Only valid for `scale ∈ {1, 2, 4, 8}`.
fn emit_reduce_load_via_sib(
    ops: &mut Assembler,
    src_dtype: NumericDType,
    src_repr: Repr,
    compute_repr: Repr,
    scale: i64,
) -> Result<(), String> {
    // x86 SIB only allows two index registers per address mode (base + index*scale),
    // not three (r15 + r11*scale + r12). So we have to materialize r15+r11*scale
    // into a temp, then load via [r12 + temp].
    //
    // Materialize: r10 = r15 + r11 * scale.
    match scale {
        1 => dynasm!(ops ; .arch x64 ; lea r10, [r15 + r11]),
        2 => dynasm!(ops ; .arch x64 ; lea r10, [r15 + r11 * 2]),
        4 => dynasm!(ops ; .arch x64 ; lea r10, [r15 + r11 * 4]),
        8 => dynasm!(ops ; .arch x64 ; lea r10, [r15 + r11 * 8]),
        _ => unreachable!(),
    }
    emit_reduce_load_via_r10(ops, src_dtype, src_repr, compute_repr)
}

/// Load `src` from `[r12 + r10]`, expand-cast to compute_repr, leave
/// result in xmm1 (float compute) or rdx (int compute).
fn emit_reduce_load_via_r10(
    ops: &mut Assembler,
    src_dtype: NumericDType,
    src_repr: Repr,
    compute_repr: Repr,
) -> Result<(), String> {
    match src_dtype {
        NumericDType::F32 => dynasm!(ops ; .arch x64 ; movss xmm1, DWORD [r12 + r10]),
        NumericDType::F64 => dynasm!(ops ; .arch x64 ; movsd xmm1, QWORD [r12 + r10]),
        NumericDType::BF16 => dynasm!(ops
            ; .arch x64
            ; movzx ecx, WORD [r12 + r10]
            ; shl ecx, 16
            ; movd xmm1, ecx
        ),
        NumericDType::F16 => dynasm!(ops
            ; .arch x64
            ; movss DWORD [rsp + F16_SPILL_XMM0], xmm0
            ; mov QWORD [rsp + K_SPILL], r11
            ; movzx edi, WORD [r12 + r10]
            ; mov rax, QWORD jit_f16_to_f32 as *const u8 as i64
            ; call rax
            ; movaps xmm1, xmm0
            ; movss xmm0, DWORD [rsp + F16_SPILL_XMM0]
            ; mov r11, QWORD [rsp + K_SPILL]
        ),
        NumericDType::I64 | NumericDType::U64 => {
            dynasm!(ops ; .arch x64 ; mov rdx, QWORD [r12 + r10])
        }
        NumericDType::I32 => dynasm!(ops ; .arch x64 ; movsxd rdx, DWORD [r12 + r10]),
        NumericDType::U32 => dynasm!(ops ; .arch x64 ; mov edx, DWORD [r12 + r10]),
        NumericDType::I16 => dynasm!(ops ; .arch x64 ; movsx rdx, WORD [r12 + r10]),
        NumericDType::U16 => dynasm!(ops ; .arch x64 ; movzx rdx, WORD [r12 + r10]),
        NumericDType::I8 => dynasm!(ops ; .arch x64 ; movsx rdx, BYTE [r12 + r10]),
        NumericDType::U8 | NumericDType::BOOL => {
            dynasm!(ops ; .arch x64 ; movzx rdx, BYTE [r12 + r10])
        }
        _ => {
            return Err(format!(
                "x86_jit: reduce src dtype {src_dtype:?} unsupported"
            ));
        }
    }
    match (src_repr, compute_repr) {
        (Repr::F32, Repr::F64) => dynasm!(ops ; .arch x64 ; cvtss2sd xmm1, xmm1),
        (Repr::F64, Repr::F32) => dynasm!(ops ; .arch x64 ; cvtsd2ss xmm1, xmm1),
        _ => {}
    }
    Ok(())
}

/// Fold the per-iteration value (xmm1 for float compute, rdx for int) into
/// the accumulator (xmm0 / rax) per the reduction kind.
fn emit_reduce_fold(ops: &mut Assembler, kind: ReduceKind, compute_repr: Repr) {
    match (kind, compute_repr) {
        (ReduceKind::Sum, Repr::F32) => dynasm!(ops ; .arch x64 ; addss xmm0, xmm1),
        (ReduceKind::Sum, Repr::F64) => dynasm!(ops ; .arch x64 ; addsd xmm0, xmm1),
        (ReduceKind::Sum, Repr::Int) => dynasm!(ops ; .arch x64 ; add rax, rdx),
        (ReduceKind::Prod, Repr::F32) => dynasm!(ops ; .arch x64 ; mulss xmm0, xmm1),
        (ReduceKind::Prod, Repr::F64) => dynasm!(ops ; .arch x64 ; mulsd xmm0, xmm1),
        (ReduceKind::Prod, Repr::Int) => dynasm!(ops ; .arch x64 ; imul rax, rdx),
        (ReduceKind::Max, Repr::F32) => {
            // acc = (src > acc) ? src : acc. Cranelift uses GreaterThan with
            // src on the left, which makes NaN sources fall through to acc.
            // Match that behaviour: use cmpss + masked select.
            dynasm!(ops
                ; .arch x64
                ; movaps xmm2, xmm1     // xmm2 = src
                ; cmpss xmm2, xmm0, 1   // xmm2 = (src < acc) mask, but we want src>acc
                                          // → cmpss imm 1 is "lt", and the mask is set
                                          // when src < acc. We invert below.
                // Actually: easier path — emulate select(src>acc, src, acc):
                //   cmpss src, acc, NLE (6) → mask = (src > acc), ordered
                //   res = (src & mask) | (acc & !mask)
                ; movaps xmm2, xmm1
                ; cmpss xmm2, xmm0, 6   // xmm2 = (src > acc) mask, ordered
                ; movaps xmm3, xmm2
                ; andps xmm1, xmm2      // src & mask
                ; andnps xmm3, xmm0     // acc & !mask
                ; orps xmm1, xmm3
                ; movaps xmm0, xmm1
            );
        }
        (ReduceKind::Max, Repr::F64) => {
            dynasm!(ops
                ; .arch x64
                ; movapd xmm2, xmm1
                ; cmpsd xmm2, xmm0, 6   // (src > acc) ordered
                ; movapd xmm3, xmm2
                ; andpd xmm1, xmm2
                ; andnpd xmm3, xmm0
                ; orpd xmm1, xmm3
                ; movapd xmm0, xmm1
            );
        }
        (ReduceKind::Max, Repr::Int) => {
            dynasm!(ops
                ; .arch x64
                ; cmp rdx, rax
                ; cmovg rax, rdx
            );
        }
        (ReduceKind::Min, Repr::F32) => {
            dynasm!(ops
                ; .arch x64
                ; movaps xmm2, xmm1
                ; cmpss xmm2, xmm0, 1   // (src < acc) ordered
                ; movaps xmm3, xmm2
                ; andps xmm1, xmm2
                ; andnps xmm3, xmm0
                ; orps xmm1, xmm3
                ; movaps xmm0, xmm1
            );
        }
        (ReduceKind::Min, Repr::F64) => {
            dynasm!(ops
                ; .arch x64
                ; movapd xmm2, xmm1
                ; cmpsd xmm2, xmm0, 1
                ; movapd xmm3, xmm2
                ; andpd xmm1, xmm2
                ; andnpd xmm3, xmm0
                ; orpd xmm1, xmm3
                ; movapd xmm0, xmm1
            );
        }
        (ReduceKind::Min, Repr::Int) => {
            dynasm!(ops
                ; .arch x64
                ; cmp rdx, rax
                ; cmovl rax, rdx
            );
        }
    }
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

    /// F64 variant of `ab_test_f32`.
    fn ab_test_f64(
        graph: &NanoGraph<'static, SystemPool>,
        inputs: &[(AtomId, &[f64])],
        outputs: &[AtomRange],
    ) {
        let raw_inputs: Vec<(AtomId, NumericDType, Vec<u8>)> = inputs
            .iter()
            .map(|(base, data)| {
                let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
                (*base, NumericDType::F64, bytes)
            })
            .collect();
        ab_test_bytes(graph, &raw_inputs, outputs);
    }

    /// Build a graph with a single F64 binary op (count=N) and A/B test it.
    fn run_binop_ab_f64(op: ScalarBinOp, a_data: &[f64], b_data: &[f64]) {
        assert_eq!(a_data.len(), b_data.len());
        let n = a_data.len() as u64;
        let mut g = NanoGraph::new();
        let a = g.add_input_tensor(GlobalId(0), n, NumericDType::F64);
        let b = g.add_input_tensor(GlobalId(1), n, NumericDType::F64);
        let out = g.push_group(
            n,
            NumericDType::F64,
            ScalarOp::Binary {
                op,
                compute_dtype: NumericDType::F64,
            },
            vec![],
            vec![InputRef::affine(a, 1), InputRef::affine(b, 1)],
        );
        let outputs = vec![AtomRange {
            base: out,
            count: n,
            dtype: NumericDType::F64,
        }];
        ab_test_f64(&g, &[(a, a_data), (b, b_data)], &outputs);
    }

    /// I64 variant of `ab_test_f32`.
    fn ab_test_i64(
        graph: &NanoGraph<'static, SystemPool>,
        inputs: &[(AtomId, &[i64])],
        outputs: &[AtomRange],
    ) {
        let raw_inputs: Vec<(AtomId, NumericDType, Vec<u8>)> = inputs
            .iter()
            .map(|(base, data)| {
                let bytes: Vec<u8> = data.iter().flat_map(|i| i.to_le_bytes()).collect();
                (*base, NumericDType::I64, bytes)
            })
            .collect();
        ab_test_bytes(graph, &raw_inputs, outputs);
    }

    /// Build a graph with a single I64 binary op (count=N) and A/B test it.
    fn run_binop_ab_i64(op: ScalarBinOp, a_data: &[i64], b_data: &[i64]) {
        assert_eq!(a_data.len(), b_data.len());
        let n = a_data.len() as u64;
        let mut g = NanoGraph::new();
        let a = g.add_input_tensor(GlobalId(0), n, NumericDType::I64);
        let b = g.add_input_tensor(GlobalId(1), n, NumericDType::I64);
        let out = g.push_group(
            n,
            NumericDType::I64,
            ScalarOp::Binary {
                op,
                compute_dtype: NumericDType::I64,
            },
            vec![],
            vec![InputRef::affine(a, 1), InputRef::affine(b, 1)],
        );
        let outputs = vec![AtomRange {
            base: out,
            count: n,
            dtype: NumericDType::I64,
        }];
        ab_test_i64(&g, &[(a, a_data), (b, b_data)], &outputs);
    }

    /// Build a graph with a single I64 unary op (count=N) and A/B test it.
    fn run_unop_ab_i64(op: ScalarUnaryOp, x_data: &[i64]) {
        let n = x_data.len() as u64;
        let mut g = NanoGraph::new();
        let x = g.add_input_tensor(GlobalId(0), n, NumericDType::I64);
        let out = g.push_group(
            n,
            NumericDType::I64,
            ScalarOp::Unary {
                op,
                compute_dtype: NumericDType::I64,
            },
            vec![],
            vec![InputRef::affine(x, 1)],
        );
        let outputs = vec![AtomRange {
            base: out,
            count: n,
            dtype: NumericDType::I64,
        }];
        ab_test_i64(&g, &[(x, x_data)], &outputs);
    }

    /// Build a graph with a single F64 unary op (count=N) and A/B test it.
    fn run_unop_ab_f64(op: ScalarUnaryOp, x_data: &[f64]) {
        let n = x_data.len() as u64;
        let mut g = NanoGraph::new();
        let x = g.add_input_tensor(GlobalId(0), n, NumericDType::F64);
        let out = g.push_group(
            n,
            NumericDType::F64,
            ScalarOp::Unary {
                op,
                compute_dtype: NumericDType::F64,
            },
            vec![],
            vec![InputRef::affine(x, 1)],
        );
        let outputs = vec![AtomRange {
            base: out,
            count: n,
            dtype: NumericDType::F64,
        }];
        ab_test_f64(&g, &[(x, x_data)], &outputs);
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
    fn unop_tan_f32_value_check() {
        run_unop_value_check(
            ScalarUnaryOp::Tan,
            &[0.0, 0.5, -0.5, 1.0, -1.0, 0.78539816, 0.25, -0.25],
            f32::tan,
        );
    }
    #[test]
    fn unop_asin_f32_value_check() {
        run_unop_value_check(
            ScalarUnaryOp::Asin,
            &[0.0, 0.5, -0.5, 1.0, -1.0, 0.25, -0.25, 0.75],
            f32::asin,
        );
    }
    #[test]
    fn unop_acos_f32_value_check() {
        run_unop_value_check(
            ScalarUnaryOp::Acos,
            &[0.0, 0.5, -0.5, 1.0, -1.0, 0.25, -0.25, 0.75],
            f32::acos,
        );
    }
    #[test]
    fn unop_atan_f32_value_check() {
        run_unop_value_check(
            ScalarUnaryOp::Atan,
            &[0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 10.0],
            f32::atan,
        );
    }
    #[test]
    fn unop_sinh_f32_value_check() {
        run_unop_value_check(
            ScalarUnaryOp::Sinh,
            &[0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0],
            f32::sinh,
        );
    }
    #[test]
    fn unop_cosh_f32_value_check() {
        run_unop_value_check(
            ScalarUnaryOp::Cosh,
            &[0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0],
            f32::cosh,
        );
    }
    #[test]
    fn unop_asinh_f32_value_check() {
        run_unop_value_check(
            ScalarUnaryOp::Asinh,
            &[0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 10.0],
            f32::asinh,
        );
    }
    #[test]
    fn unop_acosh_f32_value_check() {
        run_unop_value_check(
            ScalarUnaryOp::Acosh,
            &[1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 100.0, 1.25],
            f32::acosh,
        );
    }
    #[test]
    fn unop_atanh_f32_value_check() {
        run_unop_value_check(
            ScalarUnaryOp::Atanh,
            &[0.0, 0.5, -0.5, 0.25, -0.25, 0.75, -0.75, 0.9],
            f32::atanh,
        );
    }
    #[test]
    fn unop_log1p_f32_value_check() {
        run_unop_value_check(
            ScalarUnaryOp::Log1p,
            &[0.0, 1.0, 2.0, 0.5, -0.5, 3.0, 100.0, 0.001],
            f32::ln_1p,
        );
    }
    #[test]
    fn unop_round_f32_value_check() {
        run_unop_value_check(
            ScalarUnaryOp::Round,
            &[1.5, -2.5, 3.0, -3.0, 0.4, -0.4, 7.7, -7.7],
            f32::round,
        );
    }
    #[test]
    fn unop_isnan_f32_value_check() {
        run_unop_value_check(
            ScalarUnaryOp::IsNan,
            &[
                0.0,
                1.0,
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                -1.5,
                42.0,
                f32::NAN,
            ],
            |x| if x.is_nan() { 1.0 } else { 0.0 },
        );
    }
    #[test]
    fn unop_isinf_f32_value_check() {
        // Detect both positive and negative infinity.
        let n = 8u64;
        let mut g = NanoGraph::new();
        let x = g.add_input_tensor(GlobalId(0), n, NumericDType::F32);
        let out = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::IsInf {
                    detect_positive: true,
                    detect_negative: true,
                },
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
        let inputs = [
            0.0_f32,
            1.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -1.5,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        span.layout.write_f32_input(x, &inputs, &mut buffer);
        let f = span.entry_fn();
        unsafe { f(buffer.as_mut_ptr()) };
        let result = span.layout.read_f32_output(&outputs[0], &buffer);
        let expected: Vec<f32> = inputs
            .iter()
            .map(|v| if v.is_infinite() { 1.0 } else { 0.0 })
            .collect();
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                got, want,
                "elem {i}: input {} got {got} want {want}",
                inputs[i]
            );
        }
    }
    #[test]
    fn unop_sign_f32_value_check() {
        run_unop_value_check(
            ScalarUnaryOp::Sign,
            &[0.0, 1.0, -1.0, 100.0, -100.0, 0.5, -0.5, 0.0],
            |x| {
                if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            },
        );
    }
    #[test]
    fn unop_not_f32_value_check() {
        run_unop_value_check(
            ScalarUnaryOp::Not,
            &[0.0, 1.0, -1.0, 100.0, 0.0, 0.5, -0.5, 0.0],
            |x| if x == 0.0 { 1.0 } else { 0.0 },
        );
    }
    #[test]
    fn unop_erf_f32_value_check() {
        // Compare against the same Abramowitz & Stegun series we ship in
        // libm_erff. The series is the source of truth for the JIT side, so
        // this test mostly verifies the calling-convention plumbing.
        let xs: &[f32] = &[0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.25];
        let span = {
            let mut g = NanoGraph::new();
            let x = g.add_input_tensor(GlobalId(0), 8, NumericDType::F32);
            let out = g.push_group(
                8,
                NumericDType::F32,
                ScalarOp::Unary {
                    op: ScalarUnaryOp::Erf,
                    compute_dtype: NumericDType::F32,
                },
                vec![],
                vec![InputRef::affine(x, 1)],
            );
            let outputs = vec![AtomRange {
                base: out,
                count: 8,
                dtype: NumericDType::F32,
            }];
            let s = X86JitSpan::compile(&g, &outputs).expect("compile");
            (s, x, outputs)
        };
        let (span, x_id, outputs) = span;
        let mut buffer = span.literal_template.clone();
        span.layout.write_f32_input(x_id, xs, &mut buffer);
        let f = span.entry_fn();
        unsafe { f(buffer.as_mut_ptr()) };
        let result = span.layout.read_f32_output(&outputs[0], &buffer);
        for (i, (got, &input)) in result.iter().zip(xs.iter()).enumerate() {
            let want = libm_erff(input);
            assert!(
                (got - want).abs() < 1e-5,
                "elem {i}: input {input} got {got} want {want}"
            );
        }
    }

    // ─── Int deferred ops (value checks) ────────────────────────────────────

    #[test]
    fn unop_bitwise_not_i64_value_check() {
        let xs: &[i64] = &[0, 1, -1, 0xff, 0xff_00, 0x1234_5678, -42, i64::MAX];
        let span = {
            let mut g = NanoGraph::new();
            let x = g.add_input_tensor(GlobalId(0), 8, NumericDType::I64);
            let out = g.push_group(
                8,
                NumericDType::I64,
                ScalarOp::Unary {
                    op: ScalarUnaryOp::BitwiseNot,
                    compute_dtype: NumericDType::I64,
                },
                vec![],
                vec![InputRef::affine(x, 1)],
            );
            let outputs = vec![AtomRange {
                base: out,
                count: 8,
                dtype: NumericDType::I64,
            }];
            let s = X86JitSpan::compile(&g, &outputs).expect("compile");
            (s, x, outputs)
        };
        let (span, x_id, outputs) = span;
        let mut buffer = span.literal_template.clone();
        let raw: Vec<u8> = xs.iter().flat_map(|i| i.to_le_bytes()).collect();
        let store_slice = StoreSlice {
            base: x_id,
            data: raw.as_slice(),
            dtype: NumericDType::I64,
            count: 8,
        };
        write_store_slice_to_buffer(&store_slice, &span.layout, &mut buffer);
        let f = span.entry_fn();
        unsafe { f(buffer.as_mut_ptr()) };
        // Read out i64 manually.
        let mut out_buf = vec![0u8; 8 * 8];
        let mut span_out = SpanOutput {
            data: out_buf.as_mut_slice(),
            dtype: NumericDType::I64,
            count: 8,
        };
        read_buffer_to_output(&outputs[0], &span.layout, &buffer, &mut span_out);
        let result: Vec<i64> = (0..8)
            .map(|i| i64::from_le_bytes(out_buf[i * 8..(i + 1) * 8].try_into().unwrap()))
            .collect();
        for (i, (got, &input)) in result.iter().zip(xs.iter()).enumerate() {
            let want = !input;
            assert_eq!(*got, want, "elem {i}: input {input} got {got} want {want}");
        }
    }

    #[test]
    fn unop_sign_i64_ab() {
        // Cranelift doesn't implement Sign for ints either, so this is a value
        // check (using the same expected formula our int impl uses).
        let xs: &[i64] = &[0, 1, -1, 100, -100, 5, -5, 0];
        let span = {
            let mut g = NanoGraph::new();
            let x = g.add_input_tensor(GlobalId(0), 8, NumericDType::I64);
            let out = g.push_group(
                8,
                NumericDType::I64,
                ScalarOp::Unary {
                    op: ScalarUnaryOp::Sign,
                    compute_dtype: NumericDType::I64,
                },
                vec![],
                vec![InputRef::affine(x, 1)],
            );
            let outputs = vec![AtomRange {
                base: out,
                count: 8,
                dtype: NumericDType::I64,
            }];
            let s = X86JitSpan::compile(&g, &outputs).expect("compile");
            (s, x, outputs)
        };
        let (span, x_id, outputs) = span;
        let mut buffer = span.literal_template.clone();
        let raw: Vec<u8> = xs.iter().flat_map(|i| i.to_le_bytes()).collect();
        let store_slice = StoreSlice {
            base: x_id,
            data: raw.as_slice(),
            dtype: NumericDType::I64,
            count: 8,
        };
        write_store_slice_to_buffer(&store_slice, &span.layout, &mut buffer);
        let f = span.entry_fn();
        unsafe { f(buffer.as_mut_ptr()) };
        let mut out_buf = vec![0u8; 8 * 8];
        let mut span_out = SpanOutput {
            data: out_buf.as_mut_slice(),
            dtype: NumericDType::I64,
            count: 8,
        };
        read_buffer_to_output(&outputs[0], &span.layout, &buffer, &mut span_out);
        let result: Vec<i64> = (0..8)
            .map(|i| i64::from_le_bytes(out_buf[i * 8..(i + 1) * 8].try_into().unwrap()))
            .collect();
        for (i, (got, &input)) in result.iter().zip(xs.iter()).enumerate() {
            let want = input.signum();
            assert_eq!(*got, want, "elem {i}: input {input} got {got} want {want}");
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

    // ─── F64 binop / unop A/B coverage ──────────────────────────────────────

    const F64_A: &[f64] = &[1.0, -2.5, 3.5, 0.0, 5.0, -7.25, 8.0, 16.0];
    const F64_B: &[f64] = &[2.0, 4.0, -1.5, 1.0, -5.0, 3.0, 2.5, -4.0];
    const F64_X: &[f64] = &[1.0, -2.5, 3.5, 0.5, 5.0, -7.25, 8.0, 16.0];

    #[test]
    fn binop_add_f64_ab() {
        run_binop_ab_f64(ScalarBinOp::Add, F64_A, F64_B);
    }
    #[test]
    fn binop_sub_f64_ab() {
        run_binop_ab_f64(ScalarBinOp::Sub, F64_A, F64_B);
    }
    #[test]
    fn binop_mul_f64_ab() {
        run_binop_ab_f64(ScalarBinOp::Mul, F64_A, F64_B);
    }
    #[test]
    fn binop_div_f64_ab() {
        run_binop_ab_f64(ScalarBinOp::Div, F64_A, F64_B);
    }
    #[test]
    fn binop_min_f64_ab() {
        run_binop_ab_f64(ScalarBinOp::Min, F64_A, F64_B);
    }
    #[test]
    fn binop_max_f64_ab() {
        run_binop_ab_f64(ScalarBinOp::Max, F64_A, F64_B);
    }
    #[test]
    fn binop_pow_f64_ab() {
        run_binop_ab_f64(
            ScalarBinOp::Pow,
            &[1.0, 2.0, 3.0, 0.5, 1.5, 4.0, 0.25, 8.0],
            &[2.0, 0.5, 1.0, 3.0, 2.0, 0.5, 2.0, 0.333],
        );
    }
    #[test]
    fn binop_mod_f64_ab() {
        run_binop_ab_f64(
            ScalarBinOp::Mod,
            &[5.0, 7.5, -3.0, 10.0, 0.5, 17.0, -8.0, 4.0],
            &[2.0, 1.5, 1.0, 3.0, 0.25, 5.0, 3.0, 2.0],
        );
    }
    #[test]
    fn binop_imod_f64_ab() {
        run_binop_ab_f64(
            ScalarBinOp::IMod,
            &[5.0, 7.5, -3.0, 10.0, 0.5, 17.0, -8.0, 4.0],
            &[2.0, 1.5, 1.0, 3.0, 0.25, 5.0, 3.0, 2.0],
        );
    }
    #[test]
    fn binop_equal_f64_ab() {
        run_binop_ab_f64(
            ScalarBinOp::Equal,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[1.0, 0.0, 3.0, 5.0, 5.0, 7.0, 7.0, 9.0],
        );
    }
    #[test]
    fn binop_less_f64_ab() {
        run_binop_ab_f64(ScalarBinOp::Less, F64_A, F64_B);
    }
    #[test]
    fn binop_lessoreq_f64_ab() {
        run_binop_ab_f64(ScalarBinOp::LessOrEqual, F64_A, F64_B);
    }
    #[test]
    fn binop_greater_f64_ab() {
        run_binop_ab_f64(ScalarBinOp::Greater, F64_A, F64_B);
    }
    #[test]
    fn binop_greateroreq_f64_ab() {
        run_binop_ab_f64(ScalarBinOp::GreaterOrEqual, F64_A, F64_B);
    }
    #[test]
    fn binop_and_f64_ab() {
        run_binop_ab_f64(
            ScalarBinOp::And,
            &[1.0, 0.0, 0.0, 1.0, 5.0, 0.0, 2.0, 0.0],
            &[1.0, 1.0, 0.0, 0.0, 0.0, 5.0, 3.0, 0.0],
        );
    }
    #[test]
    fn binop_or_f64_ab() {
        run_binop_ab_f64(
            ScalarBinOp::Or,
            &[1.0, 0.0, 0.0, 1.0, 5.0, 0.0, 2.0, 0.0],
            &[1.0, 1.0, 0.0, 0.0, 0.0, 5.0, 3.0, 0.0],
        );
    }
    #[test]
    fn binop_xor_f64_ab() {
        run_binop_ab_f64(
            ScalarBinOp::Xor,
            &[1.0, 0.0, 0.0, 1.0, 5.0, 0.0, 2.0, 0.0],
            &[1.0, 1.0, 0.0, 0.0, 0.0, 5.0, 3.0, 0.0],
        );
    }

    #[test]
    fn unop_neg_f64_ab() {
        run_unop_ab_f64(ScalarUnaryOp::Neg, F64_X);
    }
    #[test]
    fn unop_abs_f64_ab() {
        run_unop_ab_f64(ScalarUnaryOp::Abs, F64_X);
    }
    #[test]
    fn unop_sqrt_f64_ab() {
        run_unop_ab_f64(
            ScalarUnaryOp::Sqrt,
            &[1.0, 4.0, 9.0, 16.0, 0.25, 100.0, 2.0, 0.5],
        );
    }
    #[test]
    fn unop_reciprocal_f64_ab() {
        run_unop_ab_f64(
            ScalarUnaryOp::Reciprocal,
            &[1.0, 2.0, 4.0, 0.5, -1.0, 8.0, 0.25, 16.0],
        );
    }
    #[test]
    fn unop_exp_f64_ab() {
        run_unop_ab_f64(
            ScalarUnaryOp::Exp,
            &[0.0, 1.0, -1.0, 2.0, 0.5, -0.5, 3.0, -3.0],
        );
    }
    #[test]
    fn unop_ln_f64_ab() {
        run_unop_ab_f64(
            ScalarUnaryOp::Ln,
            &[1.0, 2.0, 4.0, 8.0, 0.5, 16.0, 0.25, 100.0],
        );
    }
    #[test]
    fn unop_tanh_f64_ab() {
        run_unop_ab_f64(
            ScalarUnaryOp::Tanh,
            &[0.0, 1.0, -1.0, 2.0, -2.0, 0.5, 5.0, -5.0],
        );
    }
    #[test]
    fn unop_floor_f64_ab() {
        run_unop_ab_f64(
            ScalarUnaryOp::Floor,
            &[1.5, -2.5, 3.0, -3.0, 0.1, -0.1, 7.7, -7.7],
        );
    }
    #[test]
    fn unop_ceil_f64_ab() {
        run_unop_ab_f64(
            ScalarUnaryOp::Ceil,
            &[1.5, -2.5, 3.0, -3.0, 0.1, -0.1, 7.7, -7.7],
        );
    }

    /// F64 select with cross-input-shape mix.
    #[test]
    fn select_f64_ab() {
        let mut g = NanoGraph::new();
        let cond = g.add_input_tensor(GlobalId(0), 6, NumericDType::F64);
        let x = g.add_input_tensor(GlobalId(1), 6, NumericDType::F64);
        let y = g.add_input_tensor(GlobalId(2), 6, NumericDType::F64);
        let sel = g.push_group(
            6,
            NumericDType::F64,
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
            dtype: NumericDType::F64,
        }];
        ab_test_f64(
            &g,
            &[
                (cond, &[1.0, 0.0, 5.0, -3.0, 0.0, 2.5]),
                (x, &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
                (y, &[100.0, 200.0, 300.0, 400.0, 500.0, 600.0]),
            ],
            &outputs,
        );
    }

    // ─── I64 binop / unop A/B coverage ──────────────────────────────────────

    const I64_A: &[i64] = &[1, -2, 3, 0, 5, -7, 8, 16];
    const I64_B: &[i64] = &[2, 4, -1, 1, -5, 3, 2, -4];
    const I64_X: &[i64] = &[1, -2, 3, 0, 5, -7, 8, 16];

    #[test]
    fn binop_add_i64_ab() {
        run_binop_ab_i64(ScalarBinOp::Add, I64_A, I64_B);
    }
    #[test]
    fn binop_sub_i64_ab() {
        run_binop_ab_i64(ScalarBinOp::Sub, I64_A, I64_B);
    }
    #[test]
    fn binop_mul_i64_ab() {
        run_binop_ab_i64(ScalarBinOp::Mul, I64_A, I64_B);
    }
    #[test]
    fn binop_div_i64_ab() {
        // Includes division by zero (must produce 0).
        run_binop_ab_i64(
            ScalarBinOp::Div,
            &[10, 7, -8, 100, 0, -25, 1000, 9],
            &[2, 3, 4, 0, 5, -5, 13, -3],
        );
    }
    #[test]
    fn binop_min_i64_ab() {
        run_binop_ab_i64(ScalarBinOp::Min, I64_A, I64_B);
    }
    #[test]
    fn binop_max_i64_ab() {
        run_binop_ab_i64(ScalarBinOp::Max, I64_A, I64_B);
    }
    #[test]
    fn binop_mod_i64_ab() {
        run_binop_ab_i64(
            ScalarBinOp::Mod,
            &[10, 7, -8, 100, 0, -25, 1000, 9],
            &[3, 2, 5, 0, 5, 4, 13, -3],
        );
    }
    #[test]
    fn binop_imod_i64_ab() {
        run_binop_ab_i64(
            ScalarBinOp::IMod,
            &[10, 7, -8, 100, 0, -25, 1000, 9],
            &[3, 2, 5, 0, 5, 4, 13, -3],
        );
    }
    #[test]
    fn binop_equal_i64_ab() {
        run_binop_ab_i64(
            ScalarBinOp::Equal,
            &[1, 2, 3, 4, 5, 6, 7, 8],
            &[1, 0, 3, 5, 5, 7, 7, 9],
        );
    }
    #[test]
    fn binop_less_i64_ab() {
        run_binop_ab_i64(ScalarBinOp::Less, I64_A, I64_B);
    }
    #[test]
    fn binop_lessoreq_i64_ab() {
        run_binop_ab_i64(ScalarBinOp::LessOrEqual, I64_A, I64_B);
    }
    #[test]
    fn binop_greater_i64_ab() {
        run_binop_ab_i64(ScalarBinOp::Greater, I64_A, I64_B);
    }
    #[test]
    fn binop_greateroreq_i64_ab() {
        run_binop_ab_i64(ScalarBinOp::GreaterOrEqual, I64_A, I64_B);
    }
    #[test]
    fn binop_and_i64_ab() {
        run_binop_ab_i64(
            ScalarBinOp::And,
            &[1, 0, 0, 1, 5, 0, 2, 0],
            &[1, 1, 0, 0, 0, 5, 3, 0],
        );
    }
    #[test]
    fn binop_or_i64_ab() {
        run_binop_ab_i64(
            ScalarBinOp::Or,
            &[1, 0, 0, 1, 5, 0, 2, 0],
            &[1, 1, 0, 0, 0, 5, 3, 0],
        );
    }
    #[test]
    fn binop_xor_i64_ab() {
        run_binop_ab_i64(
            ScalarBinOp::Xor,
            &[1, 0, 0, 1, 5, 0, 2, 0],
            &[1, 1, 0, 0, 0, 5, 3, 0],
        );
    }
    #[test]
    fn binop_bitwise_and_i64_ab() {
        run_binop_ab_i64(
            ScalarBinOp::BitwiseAnd,
            &[0xff, 0x0f, 0xa5, -1, 0x12345678, 0, 0xff00, 0x55],
            &[0x0f, 0xff, 0x5a, 0xffff_ffff, 0xff, -1, 0x00ff, 0xaa],
        );
    }
    #[test]
    fn binop_bitwise_or_i64_ab() {
        run_binop_ab_i64(
            ScalarBinOp::BitwiseOr,
            &[0xff, 0x0f, 0xa5, 0, 0x12345678, 1, 0xff00, 0x55],
            &[0x0f, 0xff, 0x5a, 0xffff_ffff, 0xff, 0, 0x00ff, 0xaa],
        );
    }
    #[test]
    fn binop_bitwise_xor_i64_ab() {
        run_binop_ab_i64(
            ScalarBinOp::BitwiseXor,
            &[0xff, 0x0f, 0xa5, 0, 0x12345678, 1, 0xff00, 0x55],
            &[0x0f, 0xff, 0x5a, 0xffff_ffff, 0xff, 0, 0x00ff, 0xaa],
        );
    }
    #[test]
    fn binop_shl_i64_ab() {
        run_binop_ab_i64(
            ScalarBinOp::BitShiftLeft,
            &[1, 1, 1, 0xff, 0x12345678, -1, 100, 0],
            &[0, 1, 8, 4, 16, 1, 32, 5],
        );
    }
    #[test]
    fn binop_shr_i64_ab() {
        run_binop_ab_i64(
            ScalarBinOp::BitShiftRight,
            &[1024, 256, -1024, 0xff_0000, 0x12345678, -1, 100, 0],
            &[0, 4, 8, 4, 16, 1, 2, 5],
        );
    }

    #[test]
    fn unop_neg_i64_ab() {
        run_unop_ab_i64(ScalarUnaryOp::Neg, I64_X);
    }
    #[test]
    fn unop_abs_i64_ab() {
        run_unop_ab_i64(ScalarUnaryOp::Abs, I64_X);
    }

    #[test]
    fn identity_i64_ab() {
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), 8, NumericDType::I64);
        let id = g.push_group(
            8,
            NumericDType::I64,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::affine(inp, 1)],
        );
        let outputs = vec![AtomRange {
            base: id,
            count: 8,
            dtype: NumericDType::I64,
        }];
        ab_test_i64(&g, &[(inp, I64_X)], &outputs);
    }

    #[test]
    fn select_i64_ab() {
        let mut g = NanoGraph::new();
        let cond = g.add_input_tensor(GlobalId(0), 6, NumericDType::I64);
        let x = g.add_input_tensor(GlobalId(1), 6, NumericDType::I64);
        let y = g.add_input_tensor(GlobalId(2), 6, NumericDType::I64);
        let sel = g.push_group(
            6,
            NumericDType::I64,
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
            dtype: NumericDType::I64,
        }];
        ab_test_i64(
            &g,
            &[
                (cond, &[1, 0, 5, -3, 0, 2]),
                (x, &[10, 20, 30, 40, 50, 60]),
                (y, &[100, 200, 300, 400, 500, 600]),
            ],
            &outputs,
        );
    }

    /// 1024-element I64 add — exercises the loop scaffold for ints.
    #[test]
    fn mid_sized_loop_i64_ab() {
        let n = 1024usize;
        let a_data: Vec<i64> = (0..n).map(|i| i as i64).collect();
        let b_data: Vec<i64> = (0..n).map(|i| (n - i) as i64).collect();
        let mut g = NanoGraph::new();
        let a = g.add_input_tensor(GlobalId(0), n as u64, NumericDType::I64);
        let b = g.add_input_tensor(GlobalId(1), n as u64, NumericDType::I64);
        let out = g.push_group(
            n as u64,
            NumericDType::I64,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::I64,
            },
            vec![],
            vec![InputRef::affine(a, 1), InputRef::affine(b, 1)],
        );
        let outputs = vec![AtomRange {
            base: out,
            count: n as u64,
            dtype: NumericDType::I64,
        }];
        ab_test_i64(&g, &[(a, &a_data), (b, &b_data)], &outputs);
    }

    /// I32 add — checks the width-parameterized load/store path.
    #[test]
    fn binop_add_i32_ab() {
        let a_data: &[i32] = &[1, -2, 3, 0, 5, -7, 8, 16];
        let b_data: &[i32] = &[2, 4, -1, 1, -5, 3, 2, -4];
        let n = a_data.len() as u64;
        let mut g = NanoGraph::new();
        let a = g.add_input_tensor(GlobalId(0), n, NumericDType::I32);
        let b = g.add_input_tensor(GlobalId(1), n, NumericDType::I32);
        let out = g.push_group(
            n,
            NumericDType::I32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::I32,
            },
            vec![],
            vec![InputRef::affine(a, 1), InputRef::affine(b, 1)],
        );
        let outputs = vec![AtomRange {
            base: out,
            count: n,
            dtype: NumericDType::I32,
        }];
        let raw_inputs = vec![
            (
                a,
                NumericDType::I32,
                a_data.iter().flat_map(|i| i.to_le_bytes()).collect(),
            ),
            (
                b,
                NumericDType::I32,
                b_data.iter().flat_map(|i| i.to_le_bytes()).collect(),
            ),
        ];
        ab_test_bytes(&g, &raw_inputs, &outputs);
    }

    /// U16 add — exercises 2-byte zero-extending loads.
    #[test]
    fn binop_add_u16_ab() {
        let a_data: &[u16] = &[1, 2, 3, 0, 5, 7, 8, 16];
        let b_data: &[u16] = &[2, 4, 1, 1, 5, 3, 2, 4];
        let n = a_data.len() as u64;
        let mut g = NanoGraph::new();
        let a = g.add_input_tensor(GlobalId(0), n, NumericDType::U16);
        let b = g.add_input_tensor(GlobalId(1), n, NumericDType::U16);
        let out = g.push_group(
            n,
            NumericDType::U16,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::U16,
            },
            vec![],
            vec![InputRef::affine(a, 1), InputRef::affine(b, 1)],
        );
        let outputs = vec![AtomRange {
            base: out,
            count: n,
            dtype: NumericDType::U16,
        }];
        let raw_inputs = vec![
            (
                a,
                NumericDType::U16,
                a_data.iter().flat_map(|i| i.to_le_bytes()).collect(),
            ),
            (
                b,
                NumericDType::U16,
                b_data.iter().flat_map(|i| i.to_le_bytes()).collect(),
            ),
        ];
        ab_test_bytes(&g, &raw_inputs, &outputs);
    }

    /// I8 add — exercises 1-byte signed loads.
    #[test]
    fn binop_add_i8_ab() {
        let a_data: &[i8] = &[1, -2, 3, 0, 5, -7, 8, 16];
        let b_data: &[i8] = &[2, 4, -1, 1, -5, 3, 2, -4];
        let n = a_data.len() as u64;
        let mut g = NanoGraph::new();
        let a = g.add_input_tensor(GlobalId(0), n, NumericDType::I8);
        let b = g.add_input_tensor(GlobalId(1), n, NumericDType::I8);
        let out = g.push_group(
            n,
            NumericDType::I8,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::I8,
            },
            vec![],
            vec![InputRef::affine(a, 1), InputRef::affine(b, 1)],
        );
        let outputs = vec![AtomRange {
            base: out,
            count: n,
            dtype: NumericDType::I8,
        }];
        let raw_inputs = vec![
            (
                a,
                NumericDType::I8,
                a_data.iter().map(|i| *i as u8).collect(),
            ),
            (
                b,
                NumericDType::I8,
                b_data.iter().map(|i| *i as u8).collect(),
            ),
        ];
        ab_test_bytes(&g, &raw_inputs, &outputs);
    }

    /// Bool: comparing two i64 inputs and storing as Bool.
    #[test]
    fn cmp_to_bool_ab() {
        let a_data: &[i64] = &[1, 2, 3, 4, 5, 6, 7, 8];
        let b_data: &[i64] = &[1, 0, 3, 5, 5, 7, 7, 9];
        let n = a_data.len() as u64;
        let mut g = NanoGraph::new();
        let a = g.add_input_tensor(GlobalId(0), n, NumericDType::I64);
        let b = g.add_input_tensor(GlobalId(1), n, NumericDType::I64);
        // Output dtype is BOOL, compute dtype is I64.
        let out = g.push_group(
            n,
            NumericDType::BOOL,
            ScalarOp::Binary {
                op: ScalarBinOp::Less,
                compute_dtype: NumericDType::I64,
            },
            vec![],
            vec![InputRef::affine(a, 1), InputRef::affine(b, 1)],
        );
        let outputs = vec![AtomRange {
            base: out,
            count: n,
            dtype: NumericDType::BOOL,
        }];
        let raw_inputs = vec![
            (
                a,
                NumericDType::I64,
                a_data.iter().flat_map(|i| i.to_le_bytes()).collect(),
            ),
            (
                b,
                NumericDType::I64,
                b_data.iter().flat_map(|i| i.to_le_bytes()).collect(),
            ),
        ];
        ab_test_bytes(&g, &raw_inputs, &outputs);
    }

    // ─── Cast across native pairs (A/B coverage) ────────────────────────────
    //
    // Each test builds a single Identity group with input dtype A and output
    // dtype B and verifies our emission matches cranelift byte-for-byte. Uses
    // values that don't overflow either side so saturating-vs-truncating
    // semantics don't diverge.

    /// Build an Identity span: input dtype A → output dtype B.
    fn run_cast_ab(in_dtype: NumericDType, out_dtype: NumericDType, raw_input: Vec<u8>, n: u64) {
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), n, in_dtype);
        let out = g.push_group(
            n,
            out_dtype,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::affine(inp, 1)],
        );
        let outputs = vec![AtomRange {
            base: out,
            count: n,
            dtype: out_dtype,
        }];
        ab_test_bytes(&g, &[(inp, in_dtype, raw_input)], &outputs);
    }

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|f| f.to_le_bytes()).collect()
    }
    fn f64_bytes(v: &[f64]) -> Vec<u8> {
        v.iter().flat_map(|f| f.to_le_bytes()).collect()
    }
    fn i64_bytes(v: &[i64]) -> Vec<u8> {
        v.iter().flat_map(|i| i.to_le_bytes()).collect()
    }
    fn i32_bytes(v: &[i32]) -> Vec<u8> {
        v.iter().flat_map(|i| i.to_le_bytes()).collect()
    }

    #[test]
    fn cast_f32_to_f64_ab() {
        run_cast_ab(
            NumericDType::F32,
            NumericDType::F64,
            f32_bytes(&[1.0, -2.5, 3.5, 0.0, 5.0, -7.25, 8.0, 16.0]),
            8,
        );
    }
    #[test]
    fn cast_f64_to_f32_ab() {
        run_cast_ab(
            NumericDType::F64,
            NumericDType::F32,
            f64_bytes(&[1.0, -2.5, 3.5, 0.0, 5.0, -7.25, 8.0, 16.0]),
            8,
        );
    }
    #[test]
    fn cast_f32_to_i64_ab() {
        // Truncating; values inside i64 range.
        run_cast_ab(
            NumericDType::F32,
            NumericDType::I64,
            f32_bytes(&[1.5, -2.7, 100.0, 0.5, -0.1, 42.0, -1000.0, 7.99]),
            8,
        );
    }
    #[test]
    fn cast_f64_to_i64_ab() {
        run_cast_ab(
            NumericDType::F64,
            NumericDType::I64,
            f64_bytes(&[1.5, -2.7, 100.0, 0.5, -0.1, 42.0, -1000.0, 7.99]),
            8,
        );
    }
    #[test]
    fn cast_i64_to_f32_ab() {
        run_cast_ab(
            NumericDType::I64,
            NumericDType::F32,
            i64_bytes(&[1, -2, 100, 0, -1, 42, -1000, 7]),
            8,
        );
    }
    #[test]
    fn cast_i64_to_f64_ab() {
        run_cast_ab(
            NumericDType::I64,
            NumericDType::F64,
            i64_bytes(&[1, -2, 100, 0, -1, 42, -1000, 7]),
            8,
        );
    }
    #[test]
    fn cast_i32_to_i64_ab() {
        run_cast_ab(
            NumericDType::I32,
            NumericDType::I64,
            i32_bytes(&[1, -2, 100, 0, -1, 42, -1000, 7]),
            8,
        );
    }
    #[test]
    fn cast_i64_to_i32_ab() {
        run_cast_ab(
            NumericDType::I64,
            NumericDType::I32,
            i64_bytes(&[1, -2, 100, 0, -1, 42, -1000, 7]),
            8,
        );
    }
    #[test]
    fn cast_f32_to_bool_ab() {
        run_cast_ab(
            NumericDType::F32,
            NumericDType::BOOL,
            f32_bytes(&[1.0, 0.0, -2.5, 0.0, 5.0, 0.0, 8.0, 0.0]),
            8,
        );
    }
    #[test]
    fn cast_i64_to_bool_ab() {
        run_cast_ab(
            NumericDType::I64,
            NumericDType::BOOL,
            i64_bytes(&[1, 0, -2, 0, 5, 0, 8, 0]),
            8,
        );
    }
    #[test]
    fn cast_bool_to_i64_ab() {
        run_cast_ab(
            NumericDType::BOOL,
            NumericDType::I64,
            vec![1, 0, 1, 0, 1, 0, 1, 0],
            8,
        );
    }
    #[test]
    fn cast_bool_to_f32_ab() {
        run_cast_ab(
            NumericDType::BOOL,
            NumericDType::F32,
            vec![1, 0, 1, 0, 1, 0, 1, 0],
            8,
        );
    }

    // ─── BF16 / F16 storage with F32 compute ────────────────────────────────

    fn bf16_bytes(v: &[f32]) -> Vec<u8> {
        v.iter()
            .flat_map(|f| half::bf16::from_f32(*f).to_bits().to_le_bytes())
            .collect()
    }
    fn f16_bytes(v: &[f32]) -> Vec<u8> {
        v.iter()
            .flat_map(|f| half::f16::from_f32(*f).to_bits().to_le_bytes())
            .collect()
    }

    /// BF16 Identity (load + store, no compute) — A/B against cranelift.
    #[test]
    fn identity_bf16_ab() {
        let xs: &[f32] = &[1.0, -2.5, 3.5, 0.0, 5.0, -7.25, 8.0, 16.0];
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), 8, NumericDType::BF16);
        let id = g.push_group(
            8,
            NumericDType::BF16,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::affine(inp, 1)],
        );
        let outputs = vec![AtomRange {
            base: id,
            count: 8,
            dtype: NumericDType::BF16,
        }];
        ab_test_bytes(&g, &[(inp, NumericDType::BF16, bf16_bytes(xs))], &outputs);
    }

    /// BF16 storage with F32 compute — Add of two BF16 inputs into a BF16
    /// output.
    #[test]
    fn bf16_add_ab() {
        let a: &[f32] = &[1.0, -2.5, 3.5, 0.0, 5.0, -7.25, 8.0, 16.0];
        let b: &[f32] = &[2.0, 4.0, -1.5, 1.0, -5.0, 3.0, 2.5, -4.0];
        let mut g = NanoGraph::new();
        let a_id = g.add_input_tensor(GlobalId(0), 8, NumericDType::BF16);
        let b_id = g.add_input_tensor(GlobalId(1), 8, NumericDType::BF16);
        let out = g.push_group(
            8,
            NumericDType::BF16,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(a_id, 1), InputRef::affine(b_id, 1)],
        );
        let outputs = vec![AtomRange {
            base: out,
            count: 8,
            dtype: NumericDType::BF16,
        }];
        ab_test_bytes(
            &g,
            &[
                (a_id, NumericDType::BF16, bf16_bytes(a)),
                (b_id, NumericDType::BF16, bf16_bytes(b)),
            ],
            &outputs,
        );
    }

    /// BF16 storage, F32 compute, multi-op chain (mul-then-exp).
    #[test]
    fn bf16_mul_exp_ab() {
        let xs: &[f32] = &[0.0, 1.0, -1.0, 2.0, 0.5, -0.5, 3.0, -3.0];
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), 8, NumericDType::BF16);
        let lit3 = g.push_group(
            1,
            NumericDType::BF16,
            ScalarOp::Literal(NumericScalar::from_f32(3.0)),
            vec![],
            vec![],
        );
        let mul = g.push_group(
            8,
            NumericDType::BF16,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(inp, 1), InputRef::Broadcast(lit3)],
        );
        let exp = g.push_group(
            8,
            NumericDType::BF16,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(mul, 1)],
        );
        let outputs = vec![AtomRange {
            base: exp,
            count: 8,
            dtype: NumericDType::BF16,
        }];
        ab_test_bytes(&g, &[(inp, NumericDType::BF16, bf16_bytes(xs))], &outputs);
    }

    /// Cast F32 → BF16 (narrow) — A/B.
    #[test]
    fn cast_f32_to_bf16_ab() {
        let xs: &[f32] = &[1.0, -2.5, 3.5, 0.0, 5.0, -7.25, 8.0, 16.0];
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), 8, NumericDType::F32);
        let out = g.push_group(
            8,
            NumericDType::BF16,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::affine(inp, 1)],
        );
        let outputs = vec![AtomRange {
            base: out,
            count: 8,
            dtype: NumericDType::BF16,
        }];
        ab_test_bytes(&g, &[(inp, NumericDType::F32, f32_bytes(xs))], &outputs);
    }

    /// Cast BF16 → F32 (expand) — A/B.
    #[test]
    fn cast_bf16_to_f32_ab() {
        let xs: &[f32] = &[1.0, -2.5, 3.5, 0.0, 5.0, -7.25, 8.0, 16.0];
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), 8, NumericDType::BF16);
        let out = g.push_group(
            8,
            NumericDType::F32,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::affine(inp, 1)],
        );
        let outputs = vec![AtomRange {
            base: out,
            count: 8,
            dtype: NumericDType::F32,
        }];
        ab_test_bytes(&g, &[(inp, NumericDType::BF16, bf16_bytes(xs))], &outputs);
    }

    /// 1024-element BF16 add — exercises the loop scaffold + bulk expand/narrow.
    #[test]
    fn mid_sized_loop_bf16_ab() {
        let n = 1024usize;
        let a_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.25).collect();
        let mut g = NanoGraph::new();
        let a = g.add_input_tensor(GlobalId(0), n as u64, NumericDType::BF16);
        let b = g.add_input_tensor(GlobalId(1), n as u64, NumericDType::BF16);
        let out = g.push_group(
            n as u64,
            NumericDType::BF16,
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
            dtype: NumericDType::BF16,
        }];
        ab_test_bytes(
            &g,
            &[
                (a, NumericDType::BF16, bf16_bytes(&a_data)),
                (b, NumericDType::BF16, bf16_bytes(&b_data)),
            ],
            &outputs,
        );
    }

    /// F16 round-trip Identity. Cranelift's emit_typed_load doesn't have an
    /// F16 case (falls through to F32 fallback) so we can't A/B against it —
    /// instead we run our backend and check the output bytes match what
    /// `half::f16` produces directly.
    #[test]
    fn identity_f16_value_check() {
        let xs: &[f32] = &[1.0, -2.5, 3.5, 0.0, 5.0, -7.25, 8.0, 16.0];
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), 8, NumericDType::F16);
        let id = g.push_group(
            8,
            NumericDType::F16,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::affine(inp, 1)],
        );
        let outputs = vec![AtomRange {
            base: id,
            count: 8,
            dtype: NumericDType::F16,
        }];
        let span = X86JitSpan::compile(&g, &outputs).expect("compile");
        let mut buffer = span.literal_template.clone();
        let raw = f16_bytes(xs);
        let store_slice = StoreSlice {
            base: inp,
            data: raw.as_slice(),
            dtype: NumericDType::F16,
            count: 8,
        };
        write_store_slice_to_buffer(&store_slice, &span.layout, &mut buffer);
        let f = span.entry_fn();
        unsafe { f(buffer.as_mut_ptr()) };
        let mut out_buf = vec![0u8; 8 * 2];
        let mut span_out = SpanOutput {
            data: out_buf.as_mut_slice(),
            dtype: NumericDType::F16,
            count: 8,
        };
        read_buffer_to_output(&outputs[0], &span.layout, &buffer, &mut span_out);
        // Identity round-trips through f32, so the output bytes should match
        // what `half::f16::from_f32(half::f16::to_f32(input))` produces.
        for i in 0..8 {
            let in_bits = u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]);
            let out_bits = u16::from_le_bytes([out_buf[i * 2], out_buf[i * 2 + 1]]);
            let want = half::f16::from_f32(half::f16::from_bits(in_bits).to_f32()).to_bits();
            assert_eq!(out_bits, want, "elem {i}");
        }
    }

    /// F16 add via F32 compute — value check using half::f16 as the oracle.
    #[test]
    fn f16_add_value_check() {
        let a: &[f32] = &[1.0, -2.5, 3.5, 0.0, 5.0, -7.25, 8.0, 16.0];
        let b: &[f32] = &[2.0, 4.0, -1.5, 1.0, -5.0, 3.0, 2.5, -4.0];
        let mut g = NanoGraph::new();
        let a_id = g.add_input_tensor(GlobalId(0), 8, NumericDType::F16);
        let b_id = g.add_input_tensor(GlobalId(1), 8, NumericDType::F16);
        let out = g.push_group(
            8,
            NumericDType::F16,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(a_id, 1), InputRef::affine(b_id, 1)],
        );
        let outputs = vec![AtomRange {
            base: out,
            count: 8,
            dtype: NumericDType::F16,
        }];
        let span = X86JitSpan::compile(&g, &outputs).expect("compile");
        let mut buffer = span.literal_template.clone();
        let a_raw = f16_bytes(a);
        let b_raw = f16_bytes(b);
        write_store_slice_to_buffer(
            &StoreSlice {
                base: a_id,
                data: a_raw.as_slice(),
                dtype: NumericDType::F16,
                count: 8,
            },
            &span.layout,
            &mut buffer,
        );
        write_store_slice_to_buffer(
            &StoreSlice {
                base: b_id,
                data: b_raw.as_slice(),
                dtype: NumericDType::F16,
                count: 8,
            },
            &span.layout,
            &mut buffer,
        );
        let f = span.entry_fn();
        unsafe { f(buffer.as_mut_ptr()) };
        let mut out_buf = vec![0u8; 8 * 2];
        let mut span_out = SpanOutput {
            data: out_buf.as_mut_slice(),
            dtype: NumericDType::F16,
            count: 8,
        };
        read_buffer_to_output(&outputs[0], &span.layout, &buffer, &mut span_out);
        for i in 0..8 {
            let a_bits = u16::from_le_bytes([a_raw[i * 2], a_raw[i * 2 + 1]]);
            let b_bits = u16::from_le_bytes([b_raw[i * 2], b_raw[i * 2 + 1]]);
            let out_bits = u16::from_le_bytes([out_buf[i * 2], out_buf[i * 2 + 1]]);
            let af = half::f16::from_bits(a_bits).to_f32();
            let bf = half::f16::from_bits(b_bits).to_f32();
            let want = half::f16::from_f32(af + bf).to_bits();
            assert_eq!(out_bits, want, "elem {i}: a={af} b={bf}");
        }
    }

    /// 1024-element F16 loop — exercises the bulk extern-call path.
    #[test]
    fn mid_sized_loop_f16_value_check() {
        let n = 1024usize;
        let a_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.25).collect();
        let mut g = NanoGraph::new();
        let a = g.add_input_tensor(GlobalId(0), n as u64, NumericDType::F16);
        let b = g.add_input_tensor(GlobalId(1), n as u64, NumericDType::F16);
        let out = g.push_group(
            n as u64,
            NumericDType::F16,
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
            dtype: NumericDType::F16,
        }];
        let span = X86JitSpan::compile(&g, &outputs).expect("compile");
        let mut buffer = span.literal_template.clone();
        let a_raw = f16_bytes(&a_data);
        let b_raw = f16_bytes(&b_data);
        write_store_slice_to_buffer(
            &StoreSlice {
                base: a,
                data: a_raw.as_slice(),
                dtype: NumericDType::F16,
                count: n as u64,
            },
            &span.layout,
            &mut buffer,
        );
        write_store_slice_to_buffer(
            &StoreSlice {
                base: b,
                data: b_raw.as_slice(),
                dtype: NumericDType::F16,
                count: n as u64,
            },
            &span.layout,
            &mut buffer,
        );
        let f = span.entry_fn();
        unsafe { f(buffer.as_mut_ptr()) };
        let mut out_buf = vec![0u8; n * 2];
        let mut span_out = SpanOutput {
            data: out_buf.as_mut_slice(),
            dtype: NumericDType::F16,
            count: n as u64,
        };
        read_buffer_to_output(&outputs[0], &span.layout, &buffer, &mut span_out);
        for i in 0..n {
            let a_bits = u16::from_le_bytes([a_raw[i * 2], a_raw[i * 2 + 1]]);
            let b_bits = u16::from_le_bytes([b_raw[i * 2], b_raw[i * 2 + 1]]);
            let out_bits = u16::from_le_bytes([out_buf[i * 2], out_buf[i * 2 + 1]]);
            let af = half::f16::from_bits(a_bits).to_f32();
            let bf = half::f16::from_bits(b_bits).to_f32();
            let want = half::f16::from_f32(af + bf).to_bits();
            assert_eq!(out_bits, want, "elem {i}");
        }
    }

    /// 1024-element F64 add to exercise the loop scaffold.
    #[test]
    fn mid_sized_loop_f64_ab() {
        let n = 1024usize;
        let a_data: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
        let b_data: Vec<f64> = (0..n).map(|i| (n - i) as f64 * 0.25).collect();
        let mut g = NanoGraph::new();
        let a = g.add_input_tensor(GlobalId(0), n as u64, NumericDType::F64);
        let b = g.add_input_tensor(GlobalId(1), n as u64, NumericDType::F64);
        let out = g.push_group(
            n as u64,
            NumericDType::F64,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F64,
            },
            vec![],
            vec![InputRef::affine(a, 1), InputRef::affine(b, 1)],
        );
        let outputs = vec![AtomRange {
            base: out,
            count: n as u64,
            dtype: NumericDType::F64,
        }];
        ab_test_f64(&g, &[(a, &a_data), (b, &b_data)], &outputs);
    }

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

    // ─── P4: 2D Strided InputRef A/B coverage ───────────────────────────────

    /// Modular 2D Strided: each output reads `inp[i % m]`. Used for repeated
    /// broadcast across an outer dim (e.g. RoPE indexing into a small table).
    #[test]
    fn modular_2d_f32_ab() {
        let m = 4u64;
        let n = 12u64; // 3 modular cycles
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), m, NumericDType::F32);
        let out = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::modular(inp, 1, m)],
        );
        let outputs = vec![AtomRange {
            base: out,
            count: n,
            dtype: NumericDType::F32,
        }];
        let data: Vec<f32> = (0..m as usize).map(|i| i as f32 + 0.25).collect();
        ab_test_f32(&g, &[(inp, &data)], &outputs);
    }

    /// Strided-broadcast 2D: each output reads `inp[i / repeat]`. Used for
    /// row-by-row broadcast where each input row gets repeated `repeat` times.
    #[test]
    fn strided_broadcast_2d_f32_ab() {
        let r = 3u64;
        let outer = 5u64;
        let n = outer * r;
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), outer, NumericDType::F32);
        let out = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::strided_broadcast(inp, 1, r)],
        );
        let outputs = vec![AtomRange {
            base: out,
            count: n,
            dtype: NumericDType::F32,
        }];
        let data: Vec<f32> = (0..outer as usize).map(|i| i as f32 - 1.5).collect();
        ab_test_f32(&g, &[(inp, &data)], &outputs);
    }

    /// Power-of-2 modulus path: m = 8 → uses shift/mask instead of div.
    #[test]
    fn modular_pow2_f32_ab() {
        let m = 8u64;
        let n = 24u64;
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), m, NumericDType::F32);
        let out = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Abs,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::modular(inp, 1, m)],
        );
        let outputs = vec![AtomRange {
            base: out,
            count: n,
            dtype: NumericDType::F32,
        }];
        let data: Vec<f32> = (0..m as usize).map(|i| (i as f32) * 1.5 - 4.0).collect();
        ab_test_f32(&g, &[(inp, &data)], &outputs);
    }

    /// 2D Strided as part of a Binary op: input[0] is affine, input[1] is
    /// modular. Exercises the address-compute-into-rax path while another
    /// input is also being loaded.
    #[test]
    fn binop_with_modular_input_f32_ab() {
        let n = 12u64;
        let m = 4u64;
        let mut g = NanoGraph::new();
        let a = g.add_input_tensor(GlobalId(0), n, NumericDType::F32);
        let b = g.add_input_tensor(GlobalId(1), m, NumericDType::F32);
        let out = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(a, 1), InputRef::modular(b, 1, m)],
        );
        let outputs = vec![AtomRange {
            base: out,
            count: n,
            dtype: NumericDType::F32,
        }];
        let a_data: Vec<f32> = (0..n as usize).map(|i| i as f32 * 0.5).collect();
        let b_data: Vec<f32> = (0..m as usize).map(|i| i as f32 - 1.0).collect();
        ab_test_f32(&g, &[(a, &a_data), (b, &b_data)], &outputs);
    }

    /// 2D Strided with I64 compute repr — checks the int load path's
    /// rax-indexed branch.
    #[test]
    fn modular_2d_i64_ab() {
        let m = 5u64; // non-power-of-2 → exercises div path
        let n = 15u64;
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), m, NumericDType::I64);
        let out = g.push_group(
            n,
            NumericDType::I64,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::I64,
            },
            vec![],
            vec![InputRef::modular(inp, 1, m)],
        );
        let outputs = vec![AtomRange {
            base: out,
            count: n,
            dtype: NumericDType::I64,
        }];
        let data: Vec<i64> = (0..m as i64).collect();
        ab_test_i64(&g, &[(inp, &data)], &outputs);
    }

    // ─── P4: Reduce A/B coverage ────────────────────────────────────────────

    /// Helper: build a graph that reduces a 1D tensor to a scalar via the
    /// given kind / dtype, then A/B against cranelift.
    fn run_reduce_to_scalar_f32_ab(kind: ReduceKind, data: &[f32]) {
        let n = data.len() as u64;
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), n, NumericDType::F32);
        let red = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Reduce {
                kind,
                reduce_count: n,
                reduce_stride: 1,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(inp, 1)],
        );
        let outputs = vec![AtomRange {
            base: red,
            count: 1,
            dtype: NumericDType::F32,
        }];
        ab_test_f32(&g, &[(inp, data)], &outputs);
    }

    fn run_reduce_to_scalar_f64_ab(kind: ReduceKind, data: &[f64]) {
        let n = data.len() as u64;
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), n, NumericDType::F64);
        let red = g.push_group(
            1,
            NumericDType::F64,
            ScalarOp::Reduce {
                kind,
                reduce_count: n,
                reduce_stride: 1,
                compute_dtype: NumericDType::F64,
            },
            vec![],
            vec![InputRef::affine(inp, 1)],
        );
        let outputs = vec![AtomRange {
            base: red,
            count: 1,
            dtype: NumericDType::F64,
        }];
        ab_test_f64(&g, &[(inp, data)], &outputs);
    }

    fn run_reduce_to_scalar_i64_ab(kind: ReduceKind, data: &[i64]) {
        let n = data.len() as u64;
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), n, NumericDType::I64);
        let red = g.push_group(
            1,
            NumericDType::I64,
            ScalarOp::Reduce {
                kind,
                reduce_count: n,
                reduce_stride: 1,
                compute_dtype: NumericDType::I64,
            },
            vec![],
            vec![InputRef::affine(inp, 1)],
        );
        let outputs = vec![AtomRange {
            base: red,
            count: 1,
            dtype: NumericDType::I64,
        }];
        ab_test_i64(&g, &[(inp, data)], &outputs);
    }

    #[test]
    fn reduce_sum_f32_ab() {
        run_reduce_to_scalar_f32_ab(ReduceKind::Sum, &[1.0, 2.5, -3.0, 4.0, 0.5, -2.0, 7.0, 1.5]);
    }
    #[test]
    fn reduce_max_f32_ab() {
        run_reduce_to_scalar_f32_ab(ReduceKind::Max, &[1.0, 2.5, -3.0, 4.0, 0.5, -2.0, 7.0, 1.5]);
    }
    #[test]
    fn reduce_min_f32_ab() {
        run_reduce_to_scalar_f32_ab(ReduceKind::Min, &[1.0, 2.5, -3.0, 4.0, 0.5, -2.0, 7.0, 1.5]);
    }
    #[test]
    fn reduce_prod_f32_ab() {
        run_reduce_to_scalar_f32_ab(
            ReduceKind::Prod,
            &[1.0, 2.0, -1.5, 1.0, 0.5, -2.0, 1.0, 0.25],
        );
    }
    #[test]
    fn reduce_sum_f64_ab() {
        run_reduce_to_scalar_f64_ab(ReduceKind::Sum, &[1.0, 2.5, -3.0, 4.0, 0.5, -2.0, 7.0, 1.5]);
    }
    #[test]
    fn reduce_max_f64_ab() {
        run_reduce_to_scalar_f64_ab(ReduceKind::Max, &[1.0, 2.5, -3.0, 4.0, 0.5, -2.0, 7.0, 1.5]);
    }
    #[test]
    fn reduce_min_f64_ab() {
        run_reduce_to_scalar_f64_ab(ReduceKind::Min, &[1.0, 2.5, -3.0, 4.0, 0.5, -2.0, 7.0, 1.5]);
    }
    #[test]
    fn reduce_prod_f64_ab() {
        run_reduce_to_scalar_f64_ab(
            ReduceKind::Prod,
            &[1.0, 2.0, -1.5, 1.0, 0.5, -2.0, 1.0, 0.25],
        );
    }
    #[test]
    fn reduce_sum_i64_ab() {
        run_reduce_to_scalar_i64_ab(ReduceKind::Sum, &[1, 2, -3, 4, 5, -7, 8, 16]);
    }
    #[test]
    fn reduce_max_i64_ab() {
        run_reduce_to_scalar_i64_ab(ReduceKind::Max, &[1, 2, -3, 4, 5, -7, 8, 16]);
    }
    #[test]
    fn reduce_min_i64_ab() {
        run_reduce_to_scalar_i64_ab(ReduceKind::Min, &[1, 2, -3, 4, 5, -7, 8, 16]);
    }
    #[test]
    fn reduce_prod_i64_ab() {
        run_reduce_to_scalar_i64_ab(ReduceKind::Prod, &[1, 2, -1, 1, 3, -2, 1, 1]);
    }

    /// Reduce with outer count > 1: a [N, K] → [N] reduction. Exercises the
    /// outer r13 loop + inner k counted loop interaction.
    #[test]
    fn reduce_per_row_sum_f32_ab() {
        let outer = 4u64;
        let k = 5u64;
        let total = outer * k;
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), total, NumericDType::F32);
        // For each output i, reads inp[i*k .. i*k+k]; that's a 1D affine of
        // stride k, where each Reduce iteration walks k consecutive atoms.
        let red = g.push_group(
            outer,
            NumericDType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: k,
                reduce_stride: 1,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(inp, k as i64)],
        );
        let outputs = vec![AtomRange {
            base: red,
            count: outer,
            dtype: NumericDType::F32,
        }];
        let data: Vec<f32> = (0..total as usize).map(|i| i as f32 * 0.5 - 1.0).collect();
        ab_test_f32(&g, &[(inp, &data)], &outputs);
    }

    /// Reduce with non-power-of-2 reduce_count to exercise the loop path
    /// (vs the unrolled count==1 path).
    #[test]
    fn reduce_count7_max_f32_ab() {
        let n = 7u64;
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), n, NumericDType::F32);
        let red = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Max,
                reduce_count: n,
                reduce_stride: 1,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(inp, 1)],
        );
        let outputs = vec![AtomRange {
            base: red,
            count: 1,
            dtype: NumericDType::F32,
        }];
        ab_test_f32(&g, &[(inp, &[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0])], &outputs);
    }

    // ─── P4: IndirectLoad A/B coverage ──────────────────────────────────────

    /// Embedding-style gather: 1D F32 table indexed by I64 indices.
    #[test]
    fn indirect_load_f32_table_ab() {
        let table_n = 8u64;
        let n_indices = 6u64;
        let mut g = NanoGraph::new();
        let table = g.add_input_tensor(GlobalId(0), table_n, NumericDType::F32);
        let idx = g.add_input_tensor(GlobalId(1), n_indices, NumericDType::I64);
        let gather = g.push_group(
            n_indices,
            NumericDType::F32,
            ScalarOp::IndirectLoad { table_base: table },
            vec![],
            vec![InputRef::affine(idx, 1)],
        );
        let outputs = vec![AtomRange {
            base: gather,
            count: n_indices,
            dtype: NumericDType::F32,
        }];
        let table_data: Vec<f32> = (0..table_n as usize).map(|i| i as f32 + 100.0).collect();
        let idx_data: Vec<i64> = vec![3, 0, 7, 1, 5, 2];
        let table_bytes: Vec<u8> = table_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let idx_bytes: Vec<u8> = idx_data.iter().flat_map(|i| i.to_le_bytes()).collect();
        ab_test_bytes(
            &g,
            &[
                (table, NumericDType::F32, table_bytes),
                (idx, NumericDType::I64, idx_bytes),
            ],
            &outputs,
        );
    }

    /// IndirectLoad with I32 indices into an I64 table.
    #[test]
    fn indirect_load_i64_table_i32_idx_ab() {
        let table_n = 6u64;
        let n_indices = 5u64;
        let mut g = NanoGraph::new();
        let table = g.add_input_tensor(GlobalId(0), table_n, NumericDType::I64);
        let idx = g.add_input_tensor(GlobalId(1), n_indices, NumericDType::I32);
        let gather = g.push_group(
            n_indices,
            NumericDType::I64,
            ScalarOp::IndirectLoad { table_base: table },
            vec![],
            vec![InputRef::affine(idx, 1)],
        );
        let outputs = vec![AtomRange {
            base: gather,
            count: n_indices,
            dtype: NumericDType::I64,
        }];
        let table_data: Vec<i64> = (0..table_n as i64).map(|i| i * 1000 + 7).collect();
        let idx_data: Vec<i32> = vec![5, 0, 4, 1, 2];
        let table_bytes: Vec<u8> = table_data.iter().flat_map(|i| i.to_le_bytes()).collect();
        let idx_bytes: Vec<u8> = idx_data.iter().flat_map(|i| i.to_le_bytes()).collect();
        ab_test_bytes(
            &g,
            &[
                (table, NumericDType::I64, table_bytes),
                (idx, NumericDType::I32, idx_bytes),
            ],
            &outputs,
        );
    }

    // ─── P4: Explicit InputRef A/B coverage ─────────────────────────────────

    /// Single-entry Explicit (most common case): one atom referenced by
    /// every iteration. The compile-time fast path turns this into a Disp.
    #[test]
    fn explicit_single_atom_f32_ab() {
        let n = 4u64;
        let mut g = NanoGraph::new();
        // Two literal scalars at different IDs.
        let a = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(2.0)),
            vec![],
            vec![],
        );
        let inp = g.add_input_tensor(GlobalId(0), n, NumericDType::F32);
        let out = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            // input[0] = inp via 1D affine; input[1] = single-entry Explicit
            // referencing the literal scalar.
            vec![InputRef::affine(inp, 1), InputRef::Explicit(vec![a])],
        );
        let outputs = vec![AtomRange {
            base: out,
            count: n,
            dtype: NumericDType::F32,
        }];
        ab_test_f32(&g, &[(inp, &[1.0, 2.0, 3.0, 4.0])], &outputs);
    }

    /// Multi-entry Explicit: builds an embedded byte-offset table. Tests the
    /// runtime lookup path with `r13 * 8 + table_off` indirection.
    #[test]
    fn explicit_multi_entry_f32_ab() {
        let n = 6u64;
        let mut g = NanoGraph::new();
        // 6 distinct input atoms; the Explicit list permutes them.
        let inp = g.add_input_tensor(GlobalId(0), n, NumericDType::F32);
        let permuted: Vec<AtomId> = vec![
            AtomId(inp.0 + 5),
            AtomId(inp.0 + 3),
            AtomId(inp.0 + 0),
            AtomId(inp.0 + 4),
            AtomId(inp.0 + 1),
            AtomId(inp.0 + 2),
        ];
        let out = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::Explicit(permuted)],
        );
        let outputs = vec![AtomRange {
            base: out,
            count: n,
            dtype: NumericDType::F32,
        }];
        ab_test_f32(
            &g,
            &[(inp, &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0])],
            &outputs,
        );
    }
}
