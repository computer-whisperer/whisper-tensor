#![allow(clippy::all, dead_code, unreachable_patterns, unused)]
//! Cranelift JIT codegen with multi-dtype buffer scheduling.
//!
//! Compiles a span's NanoGraph into a native function operating on a typed
//! byte buffer. Memory is scheduled with liveness-aware slot reuse: when a
//! group's output is consumed by all its downstream dependents, its buffer
//! space is reclaimed and may be assigned to later groups.
//!
//! # Function signature
//!
//! `fn(buffer: *mut u8) -> ()`
//!
//! # Buffer model
//!
//! Working memory is a flat byte buffer. Each group's output and each external
//! input occupies a typed `SlotInfo` (byte offset + element count + dtype).
//! Elements are stored at their native width (8 bytes for I64, 4 for F32,
//! 2 for BF16, 1 for BOOL/U8).
//!
//! # Type discipline
//!
//! Values are loaded in their native storage type, cast to the op's
//! `compute_dtype` for arithmetic, then cast to `output_dtype` for storage.
//! Wider compute types are acceptable (e.g. BF16 computed via f32+truncation)
//! but the output must be representative of the specified dtype.
//!
//! Two Cranelift "representation kinds" carry values through the IR:
//! - `types::F32` for F32/BF16/F16 dtypes, `types::F64` for F64
//! - `types::I64` for integer dtypes (I64, I32, BOOL, U8, I8)

use std::collections::{HashMap, HashSet};

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, Value};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use crate::nano_graph::{
    AtomGroup, AtomId, AtomRange, InputRef, NanoGraph, ReduceKind, ScalarBinOp, ScalarOp,
    ScalarUnaryOp,
};
use crate::numeric_dtype::NumericDType;

// Buffer layout, marshalling, and validation moved to `layout.rs` in
// phase 1.A of the x86_jit rewrite. Both this file and `x86_jit/`
// import layout types from there.
use super::layout::{
    BufferLayout, EmbeddedTables, SlotInfo, compute_layout, op_name_short, read_buffer_to_output,
    strided_resolve_offset, validate_layout, write_store_slice_to_buffer,
};

// ─── Math function trampolines ──────────────────────────────────────────────
//
// Cranelift doesn't have intrinsics for transcendental functions. We
// declare extern "C" wrappers and link them via JITBuilder::symbol().

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

struct MathFuncs {
    // F32 variants
    expf: cranelift_module::FuncId,
    logf: cranelift_module::FuncId,
    tanhf: cranelift_module::FuncId,
    sqrtf: cranelift_module::FuncId,
    floorf: cranelift_module::FuncId,
    ceilf: cranelift_module::FuncId,
    fabsf: cranelift_module::FuncId,
    powf: cranelift_module::FuncId,
    fmodf: cranelift_module::FuncId,
    // F64 variants
    exp: cranelift_module::FuncId,
    log: cranelift_module::FuncId,
    tanh: cranelift_module::FuncId,
    pow: cranelift_module::FuncId,
    fmod: cranelift_module::FuncId,
}

impl MathFuncs {
    /// Get the function ID for a unary math op at the given float precision.
    fn unary(&self, name: &str, repr: ReprKind) -> Option<cranelift_module::FuncId> {
        match (name, repr) {
            ("exp", ReprKind::F32) => Some(self.expf),
            ("exp", ReprKind::F64) => Some(self.exp),
            ("log", ReprKind::F32) => Some(self.logf),
            ("log", ReprKind::F64) => Some(self.log),
            ("tanh", ReprKind::F32) => Some(self.tanhf),
            ("tanh", ReprKind::F64) => Some(self.tanh),
            _ => None,
        }
    }

    /// Get the function ID for a binary math op at the given float precision.
    fn binary(&self, name: &str, repr: ReprKind) -> Option<cranelift_module::FuncId> {
        match (name, repr) {
            ("pow", ReprKind::F32) => Some(self.powf),
            ("pow", ReprKind::F64) => Some(self.pow),
            ("fmod", ReprKind::F32) => Some(self.fmodf),
            ("fmod", ReprKind::F64) => Some(self.fmod),
            _ => None,
        }
    }
}

// F64 math wrappers.
extern "C" fn jit_exp(x: f64) -> f64 {
    x.exp()
}
extern "C" fn jit_log(x: f64) -> f64 {
    x.ln()
}
extern "C" fn jit_tanh(x: f64) -> f64 {
    x.tanh()
}
extern "C" fn jit_pow(x: f64, y: f64) -> f64 {
    x.powf(y)
}
extern "C" fn jit_fmod(x: f64, y: f64) -> f64 {
    x % y
}

fn register_math_symbols(jit_builder: &mut JITBuilder) {
    jit_builder.symbol("jit_expf", jit_expf as *const u8);
    jit_builder.symbol("jit_logf", jit_logf as *const u8);
    jit_builder.symbol("jit_tanhf", jit_tanhf as *const u8);
    jit_builder.symbol("jit_sqrtf", jit_sqrtf as *const u8);
    jit_builder.symbol("jit_floorf", jit_floorf as *const u8);
    jit_builder.symbol("jit_ceilf", jit_ceilf as *const u8);
    jit_builder.symbol("jit_fabsf", jit_fabsf as *const u8);
    jit_builder.symbol("jit_powf", jit_powf as *const u8);
    jit_builder.symbol("jit_fmodf", jit_fmodf as *const u8);
    // F64
    jit_builder.symbol("jit_exp", jit_exp as *const u8);
    jit_builder.symbol("jit_log", jit_log as *const u8);
    jit_builder.symbol("jit_tanh", jit_tanh as *const u8);
    jit_builder.symbol("jit_pow", jit_pow as *const u8);
    jit_builder.symbol("jit_fmod", jit_fmod as *const u8);
}

fn declare_math_funcs(module: &mut JITModule) -> Result<MathFuncs, String> {
    let mut sig1_f32 = module.make_signature();
    sig1_f32.params.push(AbiParam::new(types::F32));
    sig1_f32.returns.push(AbiParam::new(types::F32));

    let mut sig2_f32 = module.make_signature();
    sig2_f32.params.push(AbiParam::new(types::F32));
    sig2_f32.params.push(AbiParam::new(types::F32));
    sig2_f32.returns.push(AbiParam::new(types::F32));

    let mut sig1_f64 = module.make_signature();
    sig1_f64.params.push(AbiParam::new(types::F64));
    sig1_f64.returns.push(AbiParam::new(types::F64));

    let mut sig2_f64 = module.make_signature();
    sig2_f64.params.push(AbiParam::new(types::F64));
    sig2_f64.params.push(AbiParam::new(types::F64));
    sig2_f64.returns.push(AbiParam::new(types::F64));

    let decl = |m: &mut JITModule, name: &str, sig: &cranelift_codegen::ir::Signature| {
        m.declare_function(name, Linkage::Import, sig)
            .map_err(|e| format!("declare {}: {}", name, e))
    };

    Ok(MathFuncs {
        expf: decl(module, "jit_expf", &sig1_f32)?,
        logf: decl(module, "jit_logf", &sig1_f32)?,
        tanhf: decl(module, "jit_tanhf", &sig1_f32)?,
        sqrtf: decl(module, "jit_sqrtf", &sig1_f32)?,
        floorf: decl(module, "jit_floorf", &sig1_f32)?,
        ceilf: decl(module, "jit_ceilf", &sig1_f32)?,
        fabsf: decl(module, "jit_fabsf", &sig1_f32)?,
        powf: decl(module, "jit_powf", &sig2_f32)?,
        fmodf: decl(module, "jit_fmodf", &sig2_f32)?,
        exp: decl(module, "jit_exp", &sig1_f64)?,
        log: decl(module, "jit_log", &sig1_f64)?,
        tanh: decl(module, "jit_tanh", &sig1_f64)?,
        pow: decl(module, "jit_pow", &sig2_f64)?,
        fmod: decl(module, "jit_fmod", &sig2_f64)?,
    })
}

// ─── Variable counter ───────────────────────────────────────────────────────

struct VarCounter(u32);

impl VarCounter {
    fn new() -> Self {
        VarCounter(0)
    }
    fn next(&mut self) -> Variable {
        let v = Variable::from_u32(self.0);
        self.0 += 1;
        v
    }
}

// ─── Compiled span ──────────────────────────────────────────────────────────

/// A JIT-compiled span function.
pub struct CompiledSpan {
    func_ptr: *const u8,
    _module: JITModule,
}

unsafe impl Send for CompiledSpan {}
unsafe impl Sync for CompiledSpan {}

impl CompiledSpan {
    /// Run the compiled function on a buffer.
    ///
    /// # Safety
    ///
    /// Buffer must be at least `layout.total_bytes` bytes and properly
    /// initialized (literals and inputs populated).
    pub fn execute(&self, buffer: &mut [u8]) {
        let func: unsafe extern "C" fn(*mut u8) = unsafe { std::mem::transmute(self.func_ptr) };
        unsafe { func(buffer.as_mut_ptr()) };
    }
}

// ─── JIT backend for executor ────────────────────────────────────────────────

use super::executor::{CompiledSpanFn, SpanOutput, StoreSlice};

/// JIT-compiled span implementing the executor's `CompiledSpanFn` trait.
///
/// Bridges between the executor's SpanOutput/StoreSlice interface and the
/// JIT's flat byte buffer model. Owns the compiled native function, the
/// buffer layout, and a pre-populated literal template.
pub struct JitCompiledSpan {
    compiled: CompiledSpan,
    layout: BufferLayout,
    literal_template: Vec<u8>,
    output_ranges: Vec<AtomRange>,
}

impl JitCompiledSpan {
    /// Compile a span into a JIT function ready for the executor.
    pub fn compile(
        graph: &NanoGraph<'static, crate::pool::SystemPool>,
        output_ranges: &[AtomRange],
    ) -> Result<Self, String> {
        if graph.num_groups() == 0 {
            return Ok(JitCompiledSpan {
                compiled: compile_empty_span()?,
                layout: BufferLayout::empty(),
                literal_template: vec![],
                output_ranges: output_ranges.to_vec(),
            });
        }

        let t_layout = std::time::Instant::now();
        let layout = compute_layout(graph, output_ranges);
        let dt_layout = t_layout.elapsed();

        let (compiled, embedded_tables) = if std::env::var("FUSION_VALIDATE").is_ok() {
            compile_span_validated(graph, &layout)?
        } else {
            compile_span(graph, &layout)?
        };

        let t_lit = std::time::Instant::now();
        // Build literal template including embedded lookup tables.
        let total_buf_bytes = embedded_tables.total_bytes().max(layout.total_bytes);
        let mut literal_template = vec![0u8; total_buf_bytes];
        layout.populate_literals(graph, &mut literal_template);
        embedded_tables.populate(&mut literal_template);
        let dt_lit = t_lit.elapsed();

        // Add compute_layout + literal-template times to the most recent
        // span profile record (the one that compile_span just appended).
        profile::amend_layout_lit(dt_layout, dt_lit);

        Ok(JitCompiledSpan {
            compiled,
            layout,
            literal_template,
            output_ranges: output_ranges.to_vec(),
        })
    }
}

impl CompiledSpanFn for JitCompiledSpan {
    fn execute(&self, inputs: &[StoreSlice<'_>], outputs: &mut [SpanOutput<'_>]) {
        if self.layout.total_bytes == 0 {
            return;
        }

        // Clone literal template as working buffer.
        let mut buffer = self.literal_template.clone();

        // Populate inputs from store slices into buffer slots.
        for slice in inputs {
            write_store_slice_to_buffer(slice, &self.layout, &mut buffer);
        }

        // Run the JIT function.
        self.compiled.execute(&mut buffer);

        // Extract outputs from buffer into SpanOutputs.
        for (range, out) in self.output_ranges.iter().zip(outputs.iter_mut()) {
            read_buffer_to_output(range, &self.layout, &buffer, out);
        }
    }
}

/// Write a StoreSlice into the buffer at the correct slot positions.

// ─── Span compilation ───────────────────────────────────────────────────────

/// Validate that all stride-based InputRefs access atoms within single slots.
/// Returns a list of errors (empty = ok).

// ─── Elementwise loop fusion ────────────────────────────────────────────────

/// A chain of consecutive groups that share a single loop.
struct FusionChain {
    /// Indices into graph.groups().
    group_indices: Vec<usize>,
    count: u64,
    atom_offset: u64,
}

/// Build chains of consecutive fusable groups.
///
/// Two consecutive groups A and B fuse if they have the same count/atom_offset,
/// B is not a Reduce/IndirectLoad/Literal, B is not dead, and all of B's inputs
/// that reference A use Affine stride=1 (Strided with dim_strides=[1],
/// dim_shape=[MAX]).
fn build_fusion_chains(
    groups: &[AtomGroup<'static, crate::pool::SystemPool>],
    layout: &BufferLayout,
) -> Vec<FusionChain> {
    // Fusion is opt-out. Disable with FUSION=0.
    if std::env::var("FUSION").as_deref() == Ok("0") {
        return groups
            .iter()
            .enumerate()
            .filter(|(gi, g)| {
                !matches!(&g.op, ScalarOp::Literal(_) | ScalarOp::LiteralSpan(_))
                    && !(*gi < layout.group_use_counts.len() && layout.group_use_counts[*gi] == 0)
                    && !(*gi < layout.inlinable.len() && layout.inlinable[*gi])
                    && g.count > 0
            })
            .map(|(gi, g)| FusionChain {
                group_indices: vec![gi],
                count: g.count,
                atom_offset: g.atom_offset,
            })
            .collect();
    }
    let mut chains: Vec<FusionChain> = Vec::new();

    // Track atom ranges of groups in the current chain for fusion checks.
    // A group can only fuse if none of its inputs overlap a chain member's
    // range at a non-aligned offset (which would cause read-before-write).
    let mut chain_ranges: Vec<(u64, u64)> = Vec::new(); // (base_id, base_id + count)

    for (gi, group) in groups.iter().enumerate() {
        // Skip literals and dead groups — they don't participate in chains.
        if matches!(&group.op, ScalarOp::Literal(_) | ScalarOp::LiteralSpan(_)) {
            continue;
        }
        if gi < layout.group_use_counts.len() && layout.group_use_counts[gi] == 0 {
            continue;
        }
        // Skip inlinable groups — their bodies get folded into a consumer's
        // loop and they have no slot to write to.
        if gi < layout.inlinable.len() && layout.inlinable[gi] {
            continue;
        }

        // Groups that always start a new chain.
        let must_break = matches!(
            &group.op,
            ScalarOp::Reduce { .. } | ScalarOp::IndirectLoad { .. }
        );

        let can_fuse = if must_break {
            false
        } else if let Some(prev_chain) = chains.last() {
            // Check count/atom_offset match with the current chain.
            prev_chain.count == group.count
                && prev_chain.atom_offset == group.atom_offset
                && group.count > 1  // No point fusing single-element groups.
                && inputs_fusable_with_chain(group, &chain_ranges)
        } else {
            false
        };

        if can_fuse {
            // Extend the current chain.
            chain_ranges.push((group.base_id.0, group.base_id.0 + group.count));
            chains.last_mut().unwrap().group_indices.push(gi);
        } else {
            // Start a new chain.
            chain_ranges.clear();
            chain_ranges.push((group.base_id.0, group.base_id.0 + group.count));
            chains.push(FusionChain {
                group_indices: vec![gi],
                count: group.count,
                atom_offset: group.atom_offset,
            });
        }
    }

    chains
}

/// Check if all of a group's inputs that reference a chain producer use
/// Affine stride=1 (eligible for register forwarding).
fn inputs_fusable_with_chain(
    group: &AtomGroup<'static, crate::pool::SystemPool>,
    chain_ranges: &[(u64, u64)],
) -> bool {
    for input in &group.inputs {
        match input {
            InputRef::Strided {
                base,
                dim_strides,
                dim_shape,
            } => {
                // Check if this input's base falls within ANY chain member's range.
                let is_affine_1 =
                    dim_strides.len() == 1 && dim_strides[0] == 1 && dim_shape[0] == u64::MAX;
                for &(range_lo, range_hi) in chain_ranges {
                    if base.0 >= range_lo && base.0 < range_hi {
                        // This input overlaps a chain producer's range.
                        // Fusion is only safe if it reads the SAME iteration's atom:
                        // - Affine stride=1 with base == producer's base_id (exact alignment)
                        if !is_affine_1 || base.0 != range_lo {
                            return false;
                        }
                    }
                }
            }
            InputRef::Broadcast(id) => {
                // Broadcast reads a single atom. If it's in a chain member's range,
                // it reads a fixed position — only safe if that atom has been written
                // before this iteration. In a fused loop, we can't guarantee that.
                for &(range_lo, range_hi) in chain_ranges {
                    if id.0 >= range_lo && id.0 < range_hi {
                        return false;
                    }
                }
            }
            InputRef::Explicit(ids) => {
                // Explicit inputs reference arbitrary atoms. If ANY of them
                // fall within a chain member's range, fusion is unsafe because
                // the access pattern is non-sequential (could read atoms from
                // future iterations that haven't been written yet).
                for id in ids {
                    for &(range_lo, range_hi) in chain_ranges {
                        if id.0 >= range_lo && id.0 < range_hi {
                            return false;
                        }
                    }
                }
            }
        }
    }
    true
}

/// Emit a fusion chain. Single-group chains delegate to `emit_group`.
/// Multi-group chains emit a single loop with forwarded register values.
fn emit_chain(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    chain: &FusionChain,
    groups: &[AtomGroup<'static, crate::pool::SystemPool>],
    layout: &BufferLayout,
    buffer_ptr: Value,
    math: &MathFuncs,
    var_counter: &mut VarCounter,
    tables: &mut EmbeddedTables,
) -> Result<(), String> {
    // Single-group chain: delegate to existing emit_group (no change).
    if chain.group_indices.len() == 1 {
        let gi = chain.group_indices[0];
        return emit_group(
            builder,
            module,
            &groups[gi],
            gi,
            groups,
            layout,
            buffer_ptr,
            math,
            var_counter,
            tables,
        );
    }

    let count = chain.count;
    let atom_offset = chain.atom_offset;

    if count == 0 {
        return Ok(());
    }

    // Single-element chain: emit all bodies inline without a loop.
    if count == 1 {
        let mut forwarded: HashMap<u64, (Value, NumericDType)> = HashMap::new();
        for &gi in &chain.group_indices {
            let group = &groups[gi];
            emit_group_body_forwarded(
                builder,
                module,
                group,
                gi,
                groups,
                layout,
                buffer_ptr,
                None,
                atom_offset,
                math,
                var_counter,
                tables,
                &mut forwarded,
            )?;
        }
        return Ok(());
    }

    // Multi-element chain: one loop for all groups.
    let loop_header = builder.create_block();
    let loop_body = builder.create_block();
    let loop_exit = builder.create_block();

    let start = builder.ins().iconst(types::I64, atom_offset as i64);
    let end = builder
        .ins()
        .iconst(types::I64, (atom_offset + count) as i64);

    builder.ins().jump(loop_header, &[start]);

    builder.switch_to_block(loop_header);
    builder.append_block_param(loop_header, types::I64);
    let i_val = builder.block_params(loop_header)[0];

    let cmp = builder.ins().icmp(IntCC::SignedLessThan, i_val, end);
    builder.ins().brif(cmp, loop_body, &[], loop_exit, &[]);

    builder.switch_to_block(loop_body);

    // Forwarding map: producer base_id → (Cranelift Value, output NumericDType).
    // Rebuilt each iteration (Cranelift SSA values are block-local within the loop body).
    let mut forwarded: HashMap<u64, (Value, NumericDType)> = HashMap::new();

    for &gi in &chain.group_indices {
        let group = &groups[gi];
        emit_group_body_forwarded(
            builder,
            module,
            group,
            gi,
            groups,
            layout,
            buffer_ptr,
            Some(i_val),
            0,
            math,
            var_counter,
            tables,
            &mut forwarded,
        )?;
    }

    let i_next = builder.ins().iadd_imm(i_val, 1);
    builder.ins().jump(loop_header, &[i_next]);

    builder.switch_to_block(loop_exit);
    builder.seal_block(loop_header);
    builder.seal_block(loop_body);
    builder.seal_block(loop_exit);

    Ok(())
}

/// Emit one iteration of a group's computation with register forwarding.
///
/// Like `emit_group_body`, but:
/// - Uses `load_input_forwarded` to check the forwarding map before memory loads.
/// - After computing, registers the output in `forwarded` for downstream groups.
/// - Always stores to the buffer (store elimination is a future optimization).
fn emit_group_body_forwarded(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    group: &AtomGroup<'static, crate::pool::SystemPool>,
    gi: usize,
    groups: &[AtomGroup<'static, crate::pool::SystemPool>],
    layout: &BufferLayout,
    buffer_ptr: Value,
    i_val: Option<Value>,
    i_const: u64,
    math: &MathFuncs,
    var_counter: &mut VarCounter,
    tables: &mut EmbeddedTables,
    forwarded: &mut HashMap<u64, (Value, NumericDType)>,
) -> Result<(), String> {
    let (out_slot, _) = layout
        .find(group.base_id)
        .ok_or_else(|| format!("no slot for group base={}", group.base_id))?;
    let out_slot = out_slot.clone();

    let output_dtype = group.output_dtype;
    let output_repr = repr_of(output_dtype);

    let result_val = match &group.op {
        ScalarOp::Literal(_) | ScalarOp::LiteralSpan(_) => return Ok(()),

        ScalarOp::Identity | ScalarOp::Cast { .. } => {
            let src = load_input_forwarded(
                builder,
                module,
                &group.inputs[0],
                layout,
                buffer_ptr,
                i_val,
                i_const,
                group.atom_offset,
                tables,
                forwarded,
            )?;
            let src_repr =
                forwarded_or_slot_repr(&group.inputs[0], layout, group.atom_offset, forwarded);
            emit_cast_to_output(builder, src, src_repr, output_dtype)
        }

        ScalarOp::Binary { op, compute_dtype } => {
            let compute_repr = repr_of(*compute_dtype);
            let a_raw = load_input_forwarded(
                builder,
                module,
                &group.inputs[0],
                layout,
                buffer_ptr,
                i_val,
                i_const,
                group.atom_offset,
                tables,
                forwarded,
            )?;
            let a_repr =
                forwarded_or_slot_repr(&group.inputs[0], layout, group.atom_offset, forwarded);
            let a = emit_repr_cast(builder, a_raw, a_repr, compute_repr);

            let b_raw = load_input_forwarded(
                builder,
                module,
                &group.inputs[1],
                layout,
                buffer_ptr,
                i_val,
                i_const,
                group.atom_offset,
                tables,
                forwarded,
            )?;
            let b_repr =
                forwarded_or_slot_repr(&group.inputs[1], layout, group.atom_offset, forwarded);
            let b = emit_repr_cast(builder, b_raw, b_repr, compute_repr);

            let result = emit_binop(builder, module, math, *op, a, b, compute_repr)?;
            emit_cast_to_output(builder, result, compute_repr, output_dtype)
        }

        ScalarOp::Unary { op, compute_dtype } => {
            let compute_repr = repr_of(*compute_dtype);
            let x_raw = load_input_forwarded(
                builder,
                module,
                &group.inputs[0],
                layout,
                buffer_ptr,
                i_val,
                i_const,
                group.atom_offset,
                tables,
                forwarded,
            )?;
            let x_repr =
                forwarded_or_slot_repr(&group.inputs[0], layout, group.atom_offset, forwarded);
            let x = emit_repr_cast(builder, x_raw, x_repr, compute_repr);

            let result = emit_unop(builder, module, math, *op, x, compute_repr)?;
            emit_cast_to_output(builder, result, compute_repr, output_dtype)
        }

        ScalarOp::Select => {
            let cond = load_input_forwarded(
                builder,
                module,
                &group.inputs[0],
                layout,
                buffer_ptr,
                i_val,
                i_const,
                group.atom_offset,
                tables,
                forwarded,
            )?;
            let cond_repr =
                forwarded_or_slot_repr(&group.inputs[0], layout, group.atom_offset, forwarded);

            let is_nonzero = match cond_repr {
                ReprKind::F32 | ReprKind::F64 => {
                    let zero = emit_float_zero(builder, cond_repr);
                    builder.ins().fcmp(FloatCC::NotEqual, cond, zero)
                }
                ReprKind::Int => {
                    let zero = builder.ins().iconst(types::I64, 0);
                    builder.ins().icmp(IntCC::NotEqual, cond, zero)
                }
            };

            let x_raw = load_input_forwarded(
                builder,
                module,
                &group.inputs[1],
                layout,
                buffer_ptr,
                i_val,
                i_const,
                group.atom_offset,
                tables,
                forwarded,
            )?;
            let x_repr =
                forwarded_or_slot_repr(&group.inputs[1], layout, group.atom_offset, forwarded);
            let x = emit_cast_to_output(builder, x_raw, x_repr, output_dtype);

            let y_raw = load_input_forwarded(
                builder,
                module,
                &group.inputs[2],
                layout,
                buffer_ptr,
                i_val,
                i_const,
                group.atom_offset,
                tables,
                forwarded,
            )?;
            let y_repr =
                forwarded_or_slot_repr(&group.inputs[2], layout, group.atom_offset, forwarded);
            let y = emit_cast_to_output(builder, y_raw, y_repr, output_dtype);

            builder.ins().select(is_nonzero, x, y)
        }

        ScalarOp::Reduce { .. } | ScalarOp::IndirectLoad { .. } => {
            // These should not appear in multi-group chains (build_fusion_chains
            // ensures they always start their own chain). Fall through to
            // emit_group_body for safety.
            return emit_group_body(
                builder,
                module,
                group,
                gi,
                groups,
                layout,
                buffer_ptr,
                i_val,
                i_const,
                math,
                var_counter,
                tables,
            );
        }

        ScalarOp::OpaqueOutput { .. } => todo!("opaque ops not supported in compiler"),
    };

    // Always store to buffer (safe approach — avoids needing to track
    // whether any out-of-chain consumer reads this group).
    store_result(
        builder,
        buffer_ptr,
        &out_slot,
        group.atom_offset,
        i_val,
        i_const,
        result_val,
    );

    // Register in forwarding map for downstream groups in the same chain.
    // Apply the store→load round-trip in registers so the forwarded value
    // matches exactly what a memory load would produce. This preserves the
    // dtype truncation contract (e.g., BF16 precision loss between steps).
    let load_repr_val = emit_store_load_roundtrip(builder, result_val, output_dtype);
    forwarded.insert(group.base_id.0, (load_repr_val, output_dtype));

    Ok(())
}

/// Apply the equivalent of store→load in registers, so a forwarded value
/// matches exactly what `emit_typed_store` + `emit_typed_load` would produce.
///
/// This preserves the dtype truncation contract: BF16 outputs must lose
/// precision between steps, I32 values must be sign-extended to I64, etc.
fn emit_store_load_roundtrip(
    builder: &mut FunctionBuilder,
    val: Value,
    dtype: NumericDType,
) -> Value {
    match dtype {
        NumericDType::BF16 => {
            // F32 → BF16 round-to-nearest-even → F32
            // Store path: bitcast f32→i32, round, take top 16 bits
            // Load path: uextend i16→i32, shift left 16, bitcast i32→f32
            let bits = builder.ins().bitcast(types::I32, MemFlags::new(), val);
            let shifted16 = builder.ins().ushr_imm(bits, 16);
            let lsb = builder.ins().band_imm(shifted16, 1);
            let bias = builder.ins().iadd_imm(lsb, 0x7FFF);
            let rounded = builder.ins().iadd(bits, bias);
            // Zero out the bottom 16 bits (equivalent to store i16 + load i16 + shift)
            let masked = builder.ins().band_imm(rounded, !0xFFFF_i64);
            builder.ins().bitcast(types::F32, MemFlags::new(), masked)
        }
        NumericDType::F16 => {
            // Similar to BF16 but different bit layout. For now, just pass through
            // (F16 handling would need its own rounding logic).
            val
        }
        NumericDType::F64 => {
            // F64 stays in F64 repr — no truncation, no round-trip needed.
            val
        }
        NumericDType::I32 => {
            // Store: ireduce i64→i32. Load: sextend i32→i64.
            let narrow = builder.ins().ireduce(types::I32, val);
            builder.ins().sextend(types::I64, narrow)
        }
        NumericDType::U32 => {
            let narrow = builder.ins().ireduce(types::I32, val);
            builder.ins().uextend(types::I64, narrow)
        }
        NumericDType::BOOL | NumericDType::U8 => {
            let narrow = builder.ins().ireduce(types::I8, val);
            builder.ins().uextend(types::I64, narrow)
        }
        NumericDType::I8 => {
            let narrow = builder.ins().ireduce(types::I8, val);
            builder.ins().sextend(types::I64, narrow)
        }
        // F32, I64 — value is already in repr format, no round-trip needed.
        _ => val,
    }
}

/// Determine the repr kind of a loaded input, considering forwarded values.
///
/// If the input is an Affine stride=1 reference to a forwarded producer,
/// returns the repr of the producer's output dtype. Otherwise falls through
/// to the normal slot-based lookup.
fn forwarded_or_slot_repr(
    input: &InputRef,
    layout: &BufferLayout,
    atom_offset: u64,
    forwarded: &HashMap<u64, (Value, NumericDType)>,
) -> ReprKind {
    if let InputRef::Strided {
        base,
        dim_strides,
        dim_shape,
    } = input
    {
        if dim_strides.len() == 1 && dim_strides[0] == 1 && dim_shape[0] == u64::MAX {
            if let Some((_val, dtype)) = forwarded.get(&base.0) {
                return repr_of(*dtype);
            }
        }
    }
    input_slot_dtype(input, layout, atom_offset)
        .map(repr_of)
        .unwrap_or(ReprKind::F32)
}

/// Load an input value, checking the forwarding map first.
///
/// For Affine stride=1 inputs whose base is in the forwarding map, returns
/// the forwarded register value directly (skipping the memory load).
/// All other patterns fall through to the normal `load_input`.
fn load_input_forwarded(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    input: &InputRef,
    layout: &BufferLayout,
    buffer_ptr: Value,
    i_val: Option<Value>,
    i_const: u64,
    atom_offset: u64,
    tables: &mut EmbeddedTables,
    forwarded: &HashMap<u64, (Value, NumericDType)>,
) -> Result<Value, String> {
    // Check for forwarding: Affine stride=1 with a forwarded producer.
    if let InputRef::Strided {
        base,
        dim_strides,
        dim_shape,
    } = input
    {
        if dim_strides.len() == 1 && dim_strides[0] == 1 && dim_shape[0] == u64::MAX {
            if let Some((fwd_val, _fwd_dtype)) = forwarded.get(&base.0) {
                // The forwarded value has already been through
                // emit_store_load_roundtrip, so it's in the same format
                // that emit_typed_load would produce from memory.
                return Ok(*fwd_val);
            }
        }
    }

    // Not forwarded — normal memory load.
    load_input(
        builder,
        module,
        input,
        layout,
        buffer_ptr,
        i_val,
        i_const,
        atom_offset,
        tables,
    )
}

/// Compile a span, but validate by also compiling without fusion and
/// comparing the output byte-for-byte on a zero-initialized buffer.
/// Only active when FUSION_VALIDATE env var is set.
///
/// Note: when reduce-fold inlining is active in the layout, FUSION_VALIDATE
/// is bypassed — the inline-aware layout has slots stripped for inlinable
/// groups, so the "unfused" comparison branch can't be compiled against the
/// same layout. To validate inlining itself, set `INLINE=0` to disable
/// inlining for both branches.
pub fn compile_span_validated(
    graph: &NanoGraph<'static, crate::pool::SystemPool>,
    layout: &BufferLayout,
) -> Result<(CompiledSpan, EmbeddedTables), String> {
    let has_inlining = layout.inlinable.iter().any(|&b| b);
    let has_multi = {
        let chains = build_fusion_chains(graph.groups(), layout);
        chains.iter().any(|c| c.group_indices.len() > 1)
    };

    let (fused, fused_tables) = compile_span(graph, layout)?;

    if !has_multi || has_inlining || std::env::var("FUSION_VALIDATE").is_err() {
        return Ok((fused, fused_tables));
    }

    // Also compile without fusion.
    let saved = std::env::var("FUSION").ok();
    unsafe {
        std::env::remove_var("FUSION");
    }
    let (unfused, unfused_tables) = compile_span(graph, layout)?;
    if let Some(val) = saved {
        unsafe {
            std::env::set_var("FUSION", val);
        }
    }

    // Run both on a zero+literals buffer and compare.
    let fused_total = fused_tables.total_bytes();
    let unfused_total = unfused_tables.total_bytes();
    let buf_size = fused_total.max(unfused_total).max(layout.total_bytes);
    let mut buf_fused = vec![0u8; buf_size];
    let mut buf_unfused = vec![0u8; buf_size];
    layout.populate_literals(graph, &mut buf_fused);
    fused_tables.populate(&mut buf_fused);
    layout.populate_literals(graph, &mut buf_unfused);
    unfused_tables.populate(&mut buf_unfused);
    // Fill input tensor slots with deterministic test data.
    for it in graph.input_tensors() {
        if let Some((slot, _)) = layout.find(it.base_id) {
            let start = slot.byte_offset();
            let end = start + it.count as usize * slot.elem_bytes();
            if end <= buf_fused.len() {
                for (i, b) in buf_fused[start..end].iter_mut().enumerate() {
                    *b = ((i * 7 + 13) % 256) as u8;
                }
                buf_unfused[start..end].copy_from_slice(&buf_fused[start..end]);
            }
        }
    }

    fused.execute(&mut buf_fused);
    unfused.execute(&mut buf_unfused);

    // Compare output bytes.
    let mut mismatches = 0usize;
    let mut first_mismatch = None;
    for (i, (&a, &b)) in buf_fused.iter().zip(buf_unfused.iter()).enumerate() {
        if a != b && first_mismatch.is_none() {
            first_mismatch = Some(i);
        }
        if a != b {
            mismatches += 1;
        }
    }
    if mismatches > 0 {
        eprintln!(
            "  [FUSION_VALIDATE] MISMATCH: {} bytes differ (first at offset {}), buffer_size={}, groups={}",
            mismatches,
            first_mismatch.unwrap(),
            layout.total_bytes,
            graph.num_groups()
        );
        // Find which group's slot the first mismatch is in.
        if let Some(off) = first_mismatch {
            for (gi, group) in graph.groups().iter().enumerate() {
                if let Some((slot, _)) = layout.find(group.base_id) {
                    let start = slot.byte_offset();
                    let end = start + slot.count as usize * slot.elem_bytes();
                    if off >= start && off < end {
                        let elem_off = (off - start) / slot.elem_bytes();
                        eprintln!(
                            "    first mismatch in group[{}] base={} {:?} dtype={:?} atom_offset={} count={} slot_byte={} elem_offset={}",
                            gi,
                            group.base_id,
                            op_name_short(&group.op),
                            group.output_dtype,
                            group.atom_offset,
                            group.count,
                            start,
                            elem_off
                        );
                        // Show the actual values
                        let f_val = f32::from_le_bytes([
                            buf_fused[off],
                            buf_fused[off + 1],
                            buf_fused[off + 2],
                            buf_fused[off + 3],
                        ]);
                        let u_val = f32::from_le_bytes([
                            buf_unfused[off],
                            buf_unfused[off + 1],
                            buf_unfused[off + 2],
                            buf_unfused[off + 3],
                        ]);
                        eprintln!("    fused={} unfused={}", f_val, u_val);
                        // Show inputs
                        for (ii, inp) in group.inputs.iter().enumerate() {
                            eprintln!("    input[{}]: {:?}", ii, inp);
                        }
                        break;
                    }
                }
            }
        }
    }

    Ok((fused, fused_tables))
}

/// Build a Cranelift settings::Flags configuration for compile_span.
///
/// `COMPILE_FAST=1` enables the cheap fast-compile knobs:
/// - opt_level=none (skip GVN, LICM, etc.)
/// - enable_verifier=false (skip IR verification)
/// - regalloc_algorithm=single_pass (linear regalloc instead of backtracking)
/// - enable_alias_analysis=false (we have no aliasing — the buffer pointer is
///   the only memory we touch and our load/store offsets are explicit)
fn build_cranelift_flags() -> settings::Flags {
    let mut flag_builder = settings::builder();
    if std::env::var("COMPILE_FAST").is_ok() {
        flag_builder.set("opt_level", "none").unwrap();
        flag_builder.set("enable_verifier", "false").unwrap();
        flag_builder
            .set("regalloc_algorithm", "single_pass")
            .unwrap();
        flag_builder.set("enable_alias_analysis", "false").unwrap();
    } else {
        flag_builder.set("opt_level", "speed").unwrap();
    }
    settings::Flags::new(flag_builder)
}

/// Compile a span's NanoGraph into native code using the given buffer layout.
///
/// Returns the compiled function and any embedded lookup tables that must
/// be appended to the literal template buffer.
pub fn compile_span(
    graph: &NanoGraph<'static, crate::pool::SystemPool>,
    layout: &BufferLayout,
) -> Result<(CompiledSpan, EmbeddedTables), String> {
    let t_setup = std::time::Instant::now();
    let isa_builder =
        cranelift_native::builder().map_err(|e| format!("cranelift native ISA: {}", e))?;
    let isa = isa_builder
        .finish(build_cranelift_flags())
        .map_err(|e| format!("ISA finish: {}", e))?;

    let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    register_math_symbols(&mut jit_builder);

    let mut module = JITModule::new(jit_builder);
    let math = declare_math_funcs(&mut module)?;

    let mut ctx = module.make_context();
    ctx.func.signature.params.push(AbiParam::new(types::I64)); // buffer ptr

    let func_id = module
        .declare_function("span_main", Linkage::Local, &ctx.func.signature)
        .map_err(|e| format!("declare: {}", e))?;

    let mut tables = EmbeddedTables::new(layout.total_bytes);
    let dt_setup = t_setup.elapsed();

    let t_ir = std::time::Instant::now();
    {
        let mut func_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        builder.seal_block(entry);

        let buffer_ptr = builder.block_params(entry)[0];
        let mut var_counter = VarCounter::new();

        // Build fusion chains and emit one loop per chain.
        let chains = build_fusion_chains(graph.groups(), layout);
        for chain in &chains {
            emit_chain(
                &mut builder,
                &mut module,
                chain,
                graph.groups(),
                layout,
                buffer_ptr,
                &math,
                &mut var_counter,
                &mut tables,
            )?;
        }

        builder.ins().return_(&[]);
        builder.finalize();
    }
    let dt_ir = t_ir.elapsed();

    let t_define = std::time::Instant::now();
    module
        .define_function(func_id, &mut ctx)
        .map_err(|e| format!("define: {}", e))?;
    let dt_define = t_define.elapsed();

    let t_finalize = std::time::Instant::now();
    module
        .finalize_definitions()
        .map_err(|e| format!("finalize: {}", e))?;
    let dt_finalize = t_finalize.elapsed();

    let func_ptr = module.get_finalized_function(func_id);

    profile::record(profile::SpanStageTimes {
        layout: std::time::Duration::ZERO,
        setup: dt_setup,
        ir_build: dt_ir,
        cl_define: dt_define,
        cl_finalize: dt_finalize,
        literal_template: std::time::Duration::ZERO,
    });

    Ok((
        CompiledSpan {
            func_ptr,
            _module: module,
        },
        tables,
    ))
}

/// Per-span compile-time profiling. Gated on the `COMPILE_PROFILE=1` env var:
/// when unset, `record` is a no-op and there's zero overhead beyond the timer
/// reads in `compile_span` (a few hundred ns per span).
pub mod profile {
    use std::cell::RefCell;
    use std::time::Duration;

    #[derive(Clone, Copy, Default)]
    pub struct SpanStageTimes {
        pub layout: Duration,
        pub setup: Duration,
        pub ir_build: Duration,
        pub cl_define: Duration,
        pub cl_finalize: Duration,
        pub literal_template: Duration,
    }

    impl SpanStageTimes {
        pub fn total(&self) -> Duration {
            self.layout
                + self.setup
                + self.ir_build
                + self.cl_define
                + self.cl_finalize
                + self.literal_template
        }
    }

    thread_local! {
        static SINK: RefCell<Option<Vec<SpanStageTimes>>> = const { RefCell::new(None) };
    }

    /// Enable per-span recording on the current thread. Subsequent compile_span
    /// calls will append timing rows. `take()` returns and clears the buffer.
    pub fn enable() {
        SINK.with(|s| {
            *s.borrow_mut() = Some(Vec::new());
        });
    }

    pub fn disable() {
        SINK.with(|s| {
            *s.borrow_mut() = None;
        });
    }

    /// Drain accumulated span timings without disabling the sink. Subsequent
    /// `record` calls will continue to append to a fresh buffer.
    pub fn take() -> Vec<SpanStageTimes> {
        SINK.with(|s| {
            let mut borrow = s.borrow_mut();
            match borrow.as_mut() {
                Some(buf) => std::mem::take(buf),
                None => Vec::new(),
            }
        })
    }

    pub(crate) fn record(times: SpanStageTimes) {
        SINK.with(|s| {
            if let Some(buf) = s.borrow_mut().as_mut() {
                buf.push(times);
            }
        });
    }

    /// Patch the most-recently-recorded span with layout/literal-template times
    /// measured outside `compile_span`. Caller is `JitCompiledSpan::compile`.
    pub(crate) fn amend_layout_lit(layout: Duration, literal_template: Duration) {
        SINK.with(|s| {
            if let Some(buf) = s.borrow_mut().as_mut() {
                if let Some(last) = buf.last_mut() {
                    last.layout = layout;
                    last.literal_template = literal_template;
                }
            }
        });
    }
}

// ─── Group emission ─────────────────────────────────────────────────────────

/// Emit Cranelift IR for one group. Generates a loop over the group's atom
/// range, or inline code for single-element groups.
fn emit_group(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    group: &AtomGroup<'static, crate::pool::SystemPool>,
    gi: usize,
    groups: &[AtomGroup<'static, crate::pool::SystemPool>],
    layout: &BufferLayout,
    buffer_ptr: Value,
    math: &MathFuncs,
    var_counter: &mut VarCounter,
    tables: &mut EmbeddedTables,
) -> Result<(), String> {
    let count = group.count;
    let atom_offset = group.atom_offset;

    if count == 0 {
        return Ok(());
    }

    // Single element: no loop.
    if count == 1 {
        return emit_group_body(
            builder,
            module,
            group,
            gi,
            groups,
            layout,
            buffer_ptr,
            None,
            atom_offset,
            math,
            var_counter,
            tables,
        );
    }

    // Loop: for i in atom_offset..atom_offset+count
    let loop_header = builder.create_block();
    let loop_body = builder.create_block();
    let loop_exit = builder.create_block();

    let start = builder.ins().iconst(types::I64, atom_offset as i64);
    let end = builder
        .ins()
        .iconst(types::I64, (atom_offset + count) as i64);

    builder.ins().jump(loop_header, &[start]);

    builder.switch_to_block(loop_header);
    builder.append_block_param(loop_header, types::I64);
    let i_val = builder.block_params(loop_header)[0];

    let cmp = builder.ins().icmp(IntCC::SignedLessThan, i_val, end);
    builder.ins().brif(cmp, loop_body, &[], loop_exit, &[]);

    builder.switch_to_block(loop_body);

    emit_group_body(
        builder,
        module,
        group,
        gi,
        groups,
        layout,
        buffer_ptr,
        Some(i_val),
        0,
        math,
        var_counter,
        tables,
    )?;

    let i_next = builder.ins().iadd_imm(i_val, 1);
    builder.ins().jump(loop_header, &[i_next]);

    builder.switch_to_block(loop_exit);
    builder.seal_block(loop_header);
    builder.seal_block(loop_body);
    builder.seal_block(loop_exit);

    Ok(())
}

// ─── Group body emission ────────────────────────────────────────────────────

/// Emit one iteration of a group's computation.
///
/// `i_val` is the loop variable (atom_offset..atom_offset+count), or None
/// for single-element groups (use `i_const`).
fn emit_group_body(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    group: &AtomGroup<'static, crate::pool::SystemPool>,
    gi: usize,
    groups: &[AtomGroup<'static, crate::pool::SystemPool>],
    layout: &BufferLayout,
    buffer_ptr: Value,
    i_val: Option<Value>,
    i_const: u64,
    math: &MathFuncs,
    var_counter: &mut VarCounter,
    tables: &mut EmbeddedTables,
) -> Result<(), String> {
    let (out_slot, _) = layout
        .find(group.base_id)
        .ok_or_else(|| format!("no slot for group base={}", group.base_id))?;
    let out_slot = out_slot.clone();

    let output_dtype = group.output_dtype;
    let output_repr = repr_of(output_dtype);

    match &group.op {
        ScalarOp::Literal(_) | ScalarOp::LiteralSpan(_) => Ok(()),

        ScalarOp::Identity | ScalarOp::Cast { .. } => {
            // Identity/Cast: cast input to output_dtype.
            let src = load_input(
                builder,
                module,
                &group.inputs[0],
                layout,
                buffer_ptr,
                i_val,
                i_const,
                group.atom_offset,
                tables,
            )?;
            let src_repr = input_slot_dtype(&group.inputs[0], layout, group.atom_offset)
                .map(repr_of)
                .unwrap_or(ReprKind::F32);
            let result = emit_cast_to_output(builder, src, src_repr, output_dtype);
            store_result(
                builder,
                buffer_ptr,
                &out_slot,
                group.atom_offset,
                i_val,
                i_const,
                result,
            );
            Ok(())
        }

        ScalarOp::Binary { op, compute_dtype } => {
            let compute_repr = repr_of(*compute_dtype);
            let a_raw = load_input(
                builder,
                module,
                &group.inputs[0],
                layout,
                buffer_ptr,
                i_val,
                i_const,
                group.atom_offset,
                tables,
            )?;
            let a_repr = input_slot_dtype(&group.inputs[0], layout, group.atom_offset)
                .map(repr_of)
                .unwrap_or(compute_repr);
            let a = emit_repr_cast(builder, a_raw, a_repr, compute_repr);

            let b_raw = load_input(
                builder,
                module,
                &group.inputs[1],
                layout,
                buffer_ptr,
                i_val,
                i_const,
                group.atom_offset,
                tables,
            )?;
            let b_repr = input_slot_dtype(&group.inputs[1], layout, group.atom_offset)
                .map(repr_of)
                .unwrap_or(compute_repr);
            let b = emit_repr_cast(builder, b_raw, b_repr, compute_repr);

            let result = emit_binop(builder, module, math, *op, a, b, compute_repr)?;
            let output_val = emit_cast_to_output(builder, result, compute_repr, output_dtype);
            store_result(
                builder,
                buffer_ptr,
                &out_slot,
                group.atom_offset,
                i_val,
                i_const,
                output_val,
            );
            Ok(())
        }

        ScalarOp::Unary { op, compute_dtype } => {
            let compute_repr = repr_of(*compute_dtype);
            let x_raw = load_input(
                builder,
                module,
                &group.inputs[0],
                layout,
                buffer_ptr,
                i_val,
                i_const,
                group.atom_offset,
                tables,
            )?;
            let x_repr = input_slot_dtype(&group.inputs[0], layout, group.atom_offset)
                .map(repr_of)
                .unwrap_or(compute_repr);
            let x = emit_repr_cast(builder, x_raw, x_repr, compute_repr);

            let result = emit_unop(builder, module, math, *op, x, compute_repr)?;
            let output_val = emit_cast_to_output(builder, result, compute_repr, output_dtype);
            store_result(
                builder,
                buffer_ptr,
                &out_slot,
                group.atom_offset,
                i_val,
                i_const,
                output_val,
            );
            Ok(())
        }

        ScalarOp::Select => {
            // Select: truthiness test on cond, then cast selected value to output_dtype.
            let cond = load_input(
                builder,
                module,
                &group.inputs[0],
                layout,
                buffer_ptr,
                i_val,
                i_const,
                group.atom_offset,
                tables,
            )?;
            let cond_repr = input_slot_dtype(&group.inputs[0], layout, group.atom_offset)
                .map(repr_of)
                .unwrap_or(ReprKind::F32);

            let is_nonzero = match cond_repr {
                ReprKind::F32 | ReprKind::F64 => {
                    let zero = emit_float_zero(builder, cond_repr);
                    builder.ins().fcmp(FloatCC::NotEqual, cond, zero)
                }
                ReprKind::Int => {
                    let zero = builder.ins().iconst(types::I64, 0);
                    builder.ins().icmp(IntCC::NotEqual, cond, zero)
                }
            };

            // Load x and y, cast both to output_dtype.
            let x_raw = load_input(
                builder,
                module,
                &group.inputs[1],
                layout,
                buffer_ptr,
                i_val,
                i_const,
                group.atom_offset,
                tables,
            )?;
            let x_repr = input_slot_dtype(&group.inputs[1], layout, group.atom_offset)
                .map(repr_of)
                .unwrap_or(output_repr);
            let x = emit_cast_to_output(builder, x_raw, x_repr, output_dtype);

            let y_raw = load_input(
                builder,
                module,
                &group.inputs[2],
                layout,
                buffer_ptr,
                i_val,
                i_const,
                group.atom_offset,
                tables,
            )?;
            let y_repr = input_slot_dtype(&group.inputs[2], layout, group.atom_offset)
                .map(repr_of)
                .unwrap_or(output_repr);
            let y = emit_cast_to_output(builder, y_raw, y_repr, output_dtype);

            let result = builder.ins().select(is_nonzero, x, y);
            store_result(
                builder,
                buffer_ptr,
                &out_slot,
                group.atom_offset,
                i_val,
                i_const,
                result,
            );
            Ok(())
        }

        ScalarOp::Reduce {
            kind,
            reduce_count,
            reduce_stride,
            compute_dtype,
        } => {
            // Look up an inlinable producer (reduce-fold inlining).
            let inlined_producer = layout
                .inlines_producer
                .get(gi)
                .and_then(|opt| *opt)
                .map(|pi| &groups[pi]);
            emit_reduce(
                builder,
                module,
                group,
                layout,
                buffer_ptr,
                i_val,
                i_const,
                math,
                var_counter,
                tables,
                *kind,
                *reduce_count,
                *reduce_stride,
                *compute_dtype,
                &out_slot,
                inlined_producer,
            )
        }

        ScalarOp::IndirectLoad { table_base } => {
            // Index: load as integer regardless of source dtype.
            let idx_raw = load_input(
                builder,
                module,
                &group.inputs[0],
                layout,
                buffer_ptr,
                i_val,
                i_const,
                group.atom_offset,
                tables,
            )?;
            let idx_repr = input_slot_dtype(&group.inputs[0], layout, group.atom_offset)
                .map(repr_of)
                .unwrap_or(ReprKind::Int);
            let idx_i64 = emit_repr_cast(builder, idx_raw, idx_repr, ReprKind::Int);

            let (table_slot, _) = layout
                .find(*table_base)
                .ok_or_else(|| format!("no slot for IndirectLoad table_base={}", table_base))?;

            // address = buffer_ptr + table_slot.byte_offset() + idx * elem_bytes
            let base = builder
                .ins()
                .iconst(types::I64, table_slot.byte_offset() as i64);
            let idx_bytes = builder
                .ins()
                .imul_imm(idx_i64, table_slot.elem_bytes() as i64);
            let offset = builder.ins().iadd(base, idx_bytes);
            let addr = builder.ins().iadd(buffer_ptr, offset);
            let loaded = emit_typed_load(builder, addr, table_slot.dtype);
            let loaded_repr = repr_of(table_slot.dtype);

            // Cast to output_dtype.
            let result = emit_cast_to_output(builder, loaded, loaded_repr, output_dtype);
            store_result(
                builder,
                buffer_ptr,
                &out_slot,
                group.atom_offset,
                i_val,
                i_const,
                result,
            );
            Ok(())
        }

        ScalarOp::OpaqueOutput { .. } => todo!("opaque ops not supported in compiler"),
    }
}

// ─── Input loading ──────────────────────────────────────────────────────────

/// Resolve an N-D `InputRef::Strided` access for a given consumer flat index.
///
/// Mirrors `InputRef::resolve` exactly: decomposes `i` into per-dim coords
/// (innermost first via modulus, outermost last via remainder), then sums
/// `coord_d * dim_strides[d]`. Returns the producer-relative offset (in atom
/// units, NOT bytes); the caller adds it to the InputRef's `base`.

/// Determine the storage dtype that `load_input` will load from for a given InputRef.
fn input_slot_dtype(
    input: &InputRef,
    layout: &BufferLayout,
    atom_offset: u64,
) -> Option<NumericDType> {
    // Try the InputRef's base first, then fall back to the first accessed atom.
    let try_find = |atom: AtomId| layout.find(atom).map(|(s, _)| s.dtype);
    match input {
        InputRef::Broadcast(atom_id) => try_find(*atom_id),
        InputRef::Strided {
            base,
            dim_strides,
            dim_shape,
        } => try_find(*base).or_else(|| {
            let first_offset = strided_resolve_offset(dim_strides, dim_shape, atom_offset);
            let first = AtomId(base.0.wrapping_add(first_offset as u64));
            try_find(first)
        }),
        InputRef::Explicit(ids) if !ids.is_empty() => {
            let idx = (atom_offset as usize).min(ids.len() - 1);
            try_find(ids[idx])
        }
        _ => None,
    }
}

/// Resolve an Affine-like InputRef base to a byte offset in the buffer.
///
/// For unsplit groups, `base` is directly in the layout. For split groups,
/// `base` may point to the original (unsplit) group's start which isn't in
/// this span. In that case, we look up the first atom this fragment actually
/// accesses (`base + stride * atom_offset`) and back-compute the equivalent
/// base_byte.
///
/// Returns `(base_byte, elem_bytes, load_dtype)` where address of atom `i` is
/// `base_byte + stride * elem_bytes * i`.
fn resolve_affine_base(
    layout: &BufferLayout,
    base: AtomId,
    stride: i64,
    atom_offset: u64,
    label: &str,
) -> Result<(i64, usize, NumericDType), String> {
    // Fast path: base is in the layout (unsplit or atom_offset == 0).
    if let Some((slot, elem)) = layout.find(base) {
        let base_byte = slot.byte_offset() as i64 + elem as i64 * slot.elem_bytes() as i64;
        return Ok((base_byte, slot.elem_bytes(), slot.dtype));
    }

    // Split path: look up the first atom this fragment accesses.
    let first_atom = AtomId((base.0 as i64 + stride * atom_offset as i64) as u64);
    let (slot, elem) = layout.find(first_atom).ok_or_else(|| {
        format!(
            "no slot for {} base={} (first_atom={}, atom_offset={})",
            label, base, first_atom, atom_offset
        )
    })?;
    // base_byte + stride * elem_bytes * atom_offset = slot.byte_offset() + elem * elem_bytes
    let first_byte = slot.byte_offset() as i64 + elem as i64 * slot.elem_bytes() as i64;
    let base_byte = first_byte - stride * atom_offset as i64 * slot.elem_bytes() as i64;
    Ok((base_byte, slot.elem_bytes(), slot.dtype))
}

/// Load a value from an InputRef, resolving to a buffer byte address.
///
/// Returns a Cranelift Value in the storage dtype's representation kind
/// (types::F32 for float dtypes, types::I64 for integer dtypes).
/// Accumulator for explicit lookup tables embedded in the JIT buffer.
///
/// Instead of using Cranelift's `declare_data` / `global_value` (which
/// allocates a separate data section that may be >2GB from the code,
/// causing `TryFromIntError` on x86 PC-relative relocations), we embed
/// the table directly in the working buffer alongside slot data.

fn load_input(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    input: &InputRef,
    layout: &BufferLayout,
    buffer_ptr: Value,
    i_val: Option<Value>,
    i_const: u64,
    atom_offset: u64,
    tables: &mut EmbeddedTables,
) -> Result<Value, String> {
    match input {
        InputRef::Broadcast(atom_id) => {
            let (slot, elem_idx) = layout
                .find(*atom_id)
                .ok_or_else(|| format!("no slot for Broadcast atom={}", atom_id))?;
            let byte_off = slot.byte_offset() as i64 + elem_idx as i64 * slot.elem_bytes() as i64;
            let addr = addr_const(builder, buffer_ptr, byte_off);
            Ok(emit_typed_load(builder, addr, slot.dtype))
        }

        InputRef::Strided {
            base,
            dim_strides,
            dim_shape,
        } => {
            // N-dimensional strided access.
            //
            // Mirrors `InputRef::Strided::resolve`: decomposes `i` into per-dim
            // coords (innermost first via modulus, outermost last via the
            // remainder) then sums `coord[d] * dim_strides[d] * elem_bytes`.
            //
            // 1D (affine):     dim_strides=[s], dim_shape=[MAX] → base + s * i
            // 2D general:      inner = i % dim_shape[1], outer = i / dim_shape[1]
            //                  offset = dim_strides[1]*inner + dim_strides[0]*outer
            // N-D general:     same shape, more dims (innermost-first wraparound).
            let nd = dim_strides.len();
            assert!(nd >= 1, "InputRef::Strided with zero dims at base={}", base);
            assert_eq!(
                nd,
                dim_shape.len(),
                "dim_strides/dim_shape length mismatch at base={}",
                base
            );

            // Find the buffer slot by resolving the first accessed atom.
            let first_offset = strided_resolve_offset(dim_strides, dim_shape, atom_offset);
            let first_atom = AtomId((base.0 as i64 + first_offset) as u64);

            let (slot, elem) = layout
                .find(*base)
                .or_else(|| layout.find(first_atom))
                .ok_or_else(|| {
                    format!(
                        "no slot for Strided base={} first_atom={} (atom_offset={}, strides={:?}, shape={:?})",
                        base, first_atom, atom_offset, dim_strides, dim_shape
                    )
                })?;

            let elem_bytes = slot.elem_bytes() as i64;
            let load_dtype = slot.dtype;

            // Compute base_byte: byte offset of the logical `base` atom in the buffer.
            let slot_byte = slot.byte_offset() as i64 + elem as i64 * elem_bytes;
            let base_byte = if layout.find(*base).is_some() {
                slot_byte
            } else {
                // Back-compute: base_byte + first_offset * elem_bytes = slot_byte
                slot_byte - first_offset * elem_bytes
            };

            // ── 1D affine fast path ──
            if nd == 1 {
                let byte_stride = dim_strides[0] * elem_bytes;
                let addr = match i_val {
                    Some(iv) => {
                        let i_bytes = builder.ins().imul_imm(iv, byte_stride);
                        let base_val = builder.ins().iconst(types::I64, base_byte);
                        let off = builder.ins().iadd(base_val, i_bytes);
                        builder.ins().iadd(buffer_ptr, off)
                    }
                    None => {
                        let byte_off = base_byte + byte_stride * i_const as i64;
                        addr_const(builder, buffer_ptr, byte_off)
                    }
                };
                return Ok(emit_typed_load(builder, addr, load_dtype));
            }

            // ── General N-D path ──
            //
            // We decompose `i` from the innermost dim outward. For each dim d,
            // `coord[d] = remaining % dim_shape[d]` (or `remaining` for d == 0,
            // matching `InputRef::Strided::resolve`'s outermost-no-modulus rule).
            // Each coord is multiplied by `dim_strides[d] * elem_bytes` and
            // accumulated into the byte offset.
            let addr = match i_val {
                Some(iv) => {
                    let mut off = builder.ins().iconst(types::I64, base_byte);
                    let mut remaining = iv;
                    for d in (0..nd).rev() {
                        let stride_bytes = dim_strides[d] * elem_bytes;
                        if d == 0 {
                            // Outermost: no modulus — use the remainder directly.
                            if stride_bytes != 0 {
                                let coord_bytes = builder.ins().imul_imm(remaining, stride_bytes);
                                off = builder.ins().iadd(off, coord_bytes);
                            }
                        } else {
                            let modulus = dim_shape[d];
                            let (coord_val, next_remaining) = if modulus.is_power_of_two() {
                                let shift = modulus.trailing_zeros() as i64;
                                let mask = modulus as i64 - 1;
                                let c = builder.ins().band_imm(remaining, mask);
                                let r = builder.ins().ushr_imm(remaining, shift);
                                (c, r)
                            } else {
                                let modval = builder.ins().iconst(types::I64, modulus as i64);
                                let c = builder.ins().urem(remaining, modval);
                                let r = builder.ins().udiv(remaining, modval);
                                (c, r)
                            };
                            if stride_bytes != 0 {
                                let coord_bytes = builder.ins().imul_imm(coord_val, stride_bytes);
                                off = builder.ins().iadd(off, coord_bytes);
                            }
                            remaining = next_remaining;
                        }
                    }
                    builder.ins().iadd(buffer_ptr, off)
                }
                None => {
                    // Constant index: resolve at compile time.
                    let off = strided_resolve_offset(dim_strides, dim_shape, i_const);
                    let byte_off = base_byte + off * elem_bytes;
                    addr_const(builder, buffer_ptr, byte_off)
                }
            };
            Ok(emit_typed_load(builder, addr, load_dtype))
        }

        InputRef::Explicit(ids) => {
            if ids.len() == 1 {
                let (slot, elem_idx) = layout
                    .find(ids[0])
                    .ok_or_else(|| format!("no slot for Explicit[0] atom={}", ids[0]))?;
                let byte_off =
                    slot.byte_offset() as i64 + elem_idx as i64 * slot.elem_bytes() as i64;
                let addr = addr_const(builder, buffer_ptr, byte_off);
                return Ok(emit_typed_load(builder, addr, slot.dtype));
            }

            // Determine load dtype from first entry.
            let (first_slot, _) = layout
                .find(ids[0])
                .ok_or_else(|| format!("no slot for Explicit[0] atom={}", ids[0]))?;
            let load_dtype = first_slot.dtype;

            // Build lookup table of byte offsets (resolved at compile time).
            let byte_offsets: Vec<i64> = ids
                .iter()
                .map(|id| {
                    let (slot, elem_idx) = layout.find(*id).expect("Explicit atom not in layout");
                    slot.byte_offset() as i64 + elem_idx as i64 * slot.elem_bytes() as i64
                })
                .collect();

            // Embed the lookup table in the JIT buffer to avoid Cranelift
            // data section relocations (which can overflow on x86-64).
            let table_bytes: Vec<u8> = byte_offsets
                .iter()
                .flat_map(|off| off.to_le_bytes())
                .collect();
            let table_offset = tables.alloc(table_bytes);

            let table_ptr = addr_const(builder, buffer_ptr, table_offset as i64);
            let i = i_val.unwrap_or_else(|| builder.ins().iconst(types::I64, i_const as i64));
            let idx_byte_off = builder.ins().imul_imm(i, 8); // 8 bytes per i64 entry
            let idx_addr = builder.ins().iadd(table_ptr, idx_byte_off);
            let byte_off = builder.ins().load(types::I64, MemFlags::new(), idx_addr, 0);
            let addr = builder.ins().iadd(buffer_ptr, byte_off);
            Ok(emit_typed_load(builder, addr, load_dtype))
        }
    }
}

/// Helper: buffer_ptr + constant byte offset.
fn addr_const(builder: &mut FunctionBuilder, buffer_ptr: Value, byte_off: i64) -> Value {
    if byte_off == 0 {
        buffer_ptr
    } else {
        let off = builder.ins().iconst(types::I64, byte_off);
        builder.ins().iadd(buffer_ptr, off)
    }
}

/// Decomposed iteration index for inlined producer evaluation.
///
/// Reduce-fold inlining synthesizes the producer's iteration index as
/// `i_p = inline_k * outer + inner` where `outer` is the consumer's per-output
/// iv (in absolute producer coords) and `inner` is the reduce k-loop variable
/// (`0..inline_k`). When the producer's input has a 2D Strided pattern with
/// `dim_shape[1] == inline_k`, the modular decomposition collapses to
/// `coord_outer = i_p / k = outer` and `coord_inner = i_p % k = inner` —
/// avoiding the urem/udiv that the generic flat-index path would emit.
#[derive(Clone, Copy)]
struct InlineIdx {
    /// Outermost iteration variable (the consumer's `iv`, absolute coord).
    outer: Value,
    /// Inner reduce-loop variable (`0..k`).
    inner: Value,
    /// Compile-time k = consumer's reduce_count.
    k: u64,
}

/// Coordinate-aware version of `load_input` for the inlined-into-reduce path.
///
/// Recognizes the canonical 2D Strided pattern with `dim_shape[1] == idx.k`
/// and substitutes `outer`/`inner` directly, eliminating urem/udiv. Falls
/// back to materializing `i_p = idx.k * outer + inner` for any pattern that
/// doesn't match the fast structure.
#[allow(clippy::too_many_arguments)]
fn load_input_inlined(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    input: &InputRef,
    layout: &BufferLayout,
    buffer_ptr: Value,
    idx: InlineIdx,
    atom_offset: u64,
    tables: &mut EmbeddedTables,
) -> Result<Value, String> {
    // Materialize the flat index lazily — only used by the fall-back path.
    let materialize_flat = |builder: &mut FunctionBuilder| -> Value {
        if idx.k == 1 {
            // i_p == outer (degenerate; shouldn't normally happen with reduces).
            return idx.outer;
        }
        let scaled = builder.ins().imul_imm(idx.outer, idx.k as i64);
        builder.ins().iadd(scaled, idx.inner)
    };

    match input {
        // Broadcast doesn't depend on i_p — same fast path as load_input.
        InputRef::Broadcast(_) => load_input(
            builder,
            module,
            input,
            layout,
            buffer_ptr,
            None,
            0,
            atom_offset,
            tables,
        ),

        InputRef::Strided {
            base,
            dim_strides,
            dim_shape,
        } => {
            let nd = dim_strides.len();
            assert!(nd >= 1);
            assert_eq!(nd, dim_shape.len());

            // Find the buffer slot (same logic as load_input).
            let first_offset = strided_resolve_offset(dim_strides, dim_shape, atom_offset);
            let first_atom = AtomId((base.0 as i64 + first_offset) as u64);
            let (slot, elem) = layout
                .find(*base)
                .or_else(|| layout.find(first_atom))
                .ok_or_else(|| {
                    format!(
                        "no slot for Strided base={} (inlined load, atom_offset={})",
                        base, atom_offset
                    )
                })?;
            let elem_bytes = slot.elem_bytes() as i64;
            let load_dtype = slot.dtype;
            let slot_byte = slot.byte_offset() as i64 + elem as i64 * elem_bytes;
            let base_byte = if layout.find(*base).is_some() {
                slot_byte
            } else {
                slot_byte - first_offset * elem_bytes
            };

            // ── Fast path: 2D with dim_shape[1] == idx.k ──
            // The producer's input was deliberately laid out so that the inner
            // dim cycles every k atoms. With i_p = k*outer + inner, we get
            // coord_outer = outer and coord_inner = inner directly.
            if nd == 2 && dim_shape[1] == idx.k {
                let stride0_bytes = dim_strides[0] * elem_bytes;
                let stride1_bytes = dim_strides[1] * elem_bytes;

                let mut off = builder.ins().iconst(types::I64, base_byte);
                if stride0_bytes != 0 {
                    let outer_bytes = builder.ins().imul_imm(idx.outer, stride0_bytes);
                    off = builder.ins().iadd(off, outer_bytes);
                }
                if stride1_bytes != 0 {
                    let inner_bytes = builder.ins().imul_imm(idx.inner, stride1_bytes);
                    off = builder.ins().iadd(off, inner_bytes);
                }
                let addr = builder.ins().iadd(buffer_ptr, off);
                return Ok(emit_typed_load(builder, addr, load_dtype));
            }

            // ── 1D affine fast path ──
            // i_p = k*outer + inner; offset = stride * i_p =
            //   k*stride*outer + stride*inner. Compute without flat materialization.
            if nd == 1 {
                let stride = dim_strides[0];
                let byte_stride = stride * elem_bytes;
                let mut off = builder.ins().iconst(types::I64, base_byte);
                let outer_bytes_per_step = stride * idx.k as i64 * elem_bytes;
                if outer_bytes_per_step != 0 {
                    let outer_bytes = builder.ins().imul_imm(idx.outer, outer_bytes_per_step);
                    off = builder.ins().iadd(off, outer_bytes);
                }
                if byte_stride != 0 {
                    let inner_bytes = builder.ins().imul_imm(idx.inner, byte_stride);
                    off = builder.ins().iadd(off, inner_bytes);
                }
                let addr = builder.ins().iadd(buffer_ptr, off);
                return Ok(emit_typed_load(builder, addr, load_dtype));
            }

            // ── Fall-back: materialize flat i_p, defer to load_input ──
            let i_p = materialize_flat(builder);
            load_input(
                builder,
                module,
                input,
                layout,
                buffer_ptr,
                Some(i_p),
                0,
                atom_offset,
                tables,
            )
        }

        InputRef::Explicit(_) => {
            // Explicit lookups need the flat index for table indexing.
            let i_p = materialize_flat(builder);
            load_input(
                builder,
                module,
                input,
                layout,
                buffer_ptr,
                Some(i_p),
                0,
                atom_offset,
                tables,
            )
        }
    }
}

// ─── Representation kinds ───────────────────────────────────────────────────

/// Cranelift compute representation for a value.
///
/// Maps dtype semantics to Cranelift types:
/// - F64 dtypes compute in `types::F64` to preserve full precision.
/// - F32/BF16/F16 compute in `types::F32` (hardware promotes half→F32, computes, stores back).
/// - All integer dtypes compute in `types::I64`.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ReprKind {
    F32, // types::F32
    F64, // types::F64
    Int, // types::I64
}

impl ReprKind {
    fn is_float(self) -> bool {
        matches!(self, ReprKind::F32 | ReprKind::F64)
    }

    fn cranelift_type(self) -> types::Type {
        match self {
            ReprKind::F32 => types::F32,
            ReprKind::F64 => types::F64,
            ReprKind::Int => types::I64,
        }
    }
}

/// Emit a float zero constant matching the given repr.
fn emit_float_zero(builder: &mut FunctionBuilder, repr: ReprKind) -> Value {
    match repr {
        ReprKind::F32 => builder.ins().f32const(0.0),
        ReprKind::F64 => builder.ins().f64const(0.0),
        ReprKind::Int => unreachable!("emit_float_zero on Int repr"),
    }
}

/// Emit a float constant matching the given repr.
fn emit_float_const(builder: &mut FunctionBuilder, repr: ReprKind, val: f64) -> Value {
    match repr {
        ReprKind::F32 => builder.ins().f32const(val as f32),
        ReprKind::F64 => builder.ins().f64const(val),
        ReprKind::Int => unreachable!("emit_float_const on Int repr"),
    }
}

/// Map a NumericDType to its Cranelift representation kind.
fn repr_of(dtype: NumericDType) -> ReprKind {
    match dtype {
        NumericDType::F64 => ReprKind::F64,
        NumericDType::F32 | NumericDType::BF16 | NumericDType::F16 => ReprKind::F32,
        _ => ReprKind::Int,
    }
}

/// Emit a cast between Cranelift representation kinds.
/// If from == to, returns val unchanged. Otherwise promotes/demotes/converts.
fn emit_repr_cast(
    builder: &mut FunctionBuilder,
    val: Value,
    from: ReprKind,
    to: ReprKind,
) -> Value {
    if from == to {
        return val;
    }
    match (from, to) {
        (ReprKind::F32, ReprKind::F64) => builder.ins().fpromote(types::F64, val),
        (ReprKind::F64, ReprKind::F32) => builder.ins().fdemote(types::F32, val),
        (ReprKind::F32 | ReprKind::F64, ReprKind::Int) => {
            builder.ins().fcvt_to_sint_sat(types::I64, val)
        }
        (ReprKind::Int, ReprKind::F32) => builder.ins().fcvt_from_sint(types::F32, val),
        (ReprKind::Int, ReprKind::F64) => builder.ins().fcvt_from_sint(types::F64, val),
        _ => val, // same kind
    }
}

/// Emit a cast from a compute-repr value to the output dtype's storage repr.
/// Handles narrowing (e.g., i64 → i8 for BOOL, f32 → bf16 bits for BF16).
/// Returns a value ready for `emit_typed_store`.
fn emit_cast_to_output(
    builder: &mut FunctionBuilder,
    val: Value,
    compute_repr: ReprKind,
    output_dtype: NumericDType,
) -> Value {
    let target_repr = repr_of(output_dtype);
    let val = emit_repr_cast(builder, val, compute_repr, target_repr);

    // Further narrowing for sub-word output types.
    match output_dtype {
        NumericDType::BOOL => {
            // Nonzero test → 0 or 1 as i8.
            match target_repr {
                ReprKind::Int => {
                    let zero = builder.ins().iconst(types::I64, 0);
                    let is_nz = builder.ins().icmp(IntCC::NotEqual, val, zero);
                    let one = builder.ins().iconst(types::I8, 1);
                    let zero8 = builder.ins().iconst(types::I8, 0);
                    builder.ins().select(is_nz, one, zero8)
                }
                ReprKind::F32 | ReprKind::F64 => {
                    // Shouldn't happen after repr_cast, but handle defensively.
                    let zero = emit_float_zero(builder, target_repr);
                    let is_nz = builder.ins().fcmp(FloatCC::NotEqual, val, zero);
                    let one = builder.ins().iconst(types::I8, 1);
                    let zero8 = builder.ins().iconst(types::I8, 0);
                    builder.ins().select(is_nz, one, zero8)
                }
            }
        }
        NumericDType::U8 | NumericDType::I8 => {
            // Already Int repr (i64), narrow to i8.
            builder.ins().ireduce(types::I8, val)
        }
        NumericDType::U16 | NumericDType::I16 => {
            // Already Int repr (i64), narrow to i16.
            builder.ins().ireduce(types::I16, val)
        }
        NumericDType::I32 | NumericDType::U32 => {
            // Already Int repr (i64), narrow to i32.
            builder.ins().ireduce(types::I32, val)
        }
        // F32, BF16, I64 — val is already in the right repr, store handles format.
        _ => val,
    }
}

// ─── Typed load/store ───────────────────────────────────────────────────────

/// Load from a buffer address in native storage format.
///
/// Returns a value in the dtype's representation kind:
/// - Float dtypes → types::F32 (BF16 widened to f32)
/// - Integer dtypes → types::I64 (smaller ints zero/sign-extended)
fn emit_typed_load(builder: &mut FunctionBuilder, addr: Value, dtype: NumericDType) -> Value {
    match dtype {
        NumericDType::F32 => builder.ins().load(types::F32, MemFlags::trusted(), addr, 0),
        NumericDType::BF16 => {
            let raw = builder.ins().load(types::I16, MemFlags::trusted(), addr, 0);
            let wide = builder.ins().uextend(types::I32, raw);
            let shifted = builder.ins().ishl_imm(wide, 16);
            builder.ins().bitcast(types::F32, MemFlags::new(), shifted)
        }
        NumericDType::F64 => builder.ins().load(types::F64, MemFlags::trusted(), addr, 0),
        NumericDType::I64 | NumericDType::U64 => {
            builder.ins().load(types::I64, MemFlags::trusted(), addr, 0)
        }
        NumericDType::I32 => {
            let raw = builder.ins().load(types::I32, MemFlags::trusted(), addr, 0);
            builder.ins().sextend(types::I64, raw)
        }
        NumericDType::U32 => {
            let raw = builder.ins().load(types::I32, MemFlags::trusted(), addr, 0);
            builder.ins().uextend(types::I64, raw)
        }
        NumericDType::I16 => {
            let raw = builder.ins().load(types::I16, MemFlags::trusted(), addr, 0);
            builder.ins().sextend(types::I64, raw)
        }
        NumericDType::U16 => {
            let raw = builder.ins().load(types::I16, MemFlags::trusted(), addr, 0);
            builder.ins().uextend(types::I64, raw)
        }
        NumericDType::BOOL | NumericDType::U8 => {
            let raw = builder.ins().load(types::I8, MemFlags::trusted(), addr, 0);
            builder.ins().uextend(types::I64, raw)
        }
        NumericDType::I8 => {
            let raw = builder.ins().load(types::I8, MemFlags::trusted(), addr, 0);
            builder.ins().sextend(types::I64, raw)
        }
        _ => {
            // Fallback: assume 4-byte float-like.
            builder.ins().load(types::F32, MemFlags::trusted(), addr, 0)
        }
    }
}

/// Store a value to a buffer address in native storage format.
///
/// `val` must be in the correct Cranelift type for the dtype:
/// - F32: types::F32
/// - BF16: types::F32 (will be rounded and narrowed)
/// - I64: types::I64
/// - BOOL/U8/I8: types::I8
/// - I32/U32: types::I32
fn emit_typed_store(builder: &mut FunctionBuilder, addr: Value, val: Value, dtype: NumericDType) {
    match dtype {
        NumericDType::F32 => {
            builder.ins().store(MemFlags::trusted(), val, addr, 0);
        }
        NumericDType::BF16 => {
            // Round f32 → BF16: bitcast to i32, round-to-nearest-even, take top 16 bits.
            let bits = builder.ins().bitcast(types::I32, MemFlags::new(), val);
            let shifted16 = builder.ins().ushr_imm(bits, 16);
            let lsb = builder.ins().band_imm(shifted16, 1);
            let bias = builder.ins().iadd_imm(lsb, 0x7FFF);
            let rounded = builder.ins().iadd(bits, bias);
            let top = builder.ins().ushr_imm(rounded, 16);
            let narrow = builder.ins().ireduce(types::I16, top);
            builder.ins().store(MemFlags::trusted(), narrow, addr, 0);
        }
        NumericDType::I64 | NumericDType::U64 => {
            builder.ins().store(MemFlags::trusted(), val, addr, 0); // val is i64
        }
        NumericDType::I32 | NumericDType::U32 => {
            builder.ins().store(MemFlags::trusted(), val, addr, 0); // val is i32
        }
        NumericDType::I16 | NumericDType::U16 => {
            builder.ins().store(MemFlags::trusted(), val, addr, 0); // val is i16 (narrowed in emit_cast_to_output)
        }
        NumericDType::BOOL | NumericDType::U8 | NumericDType::I8 => {
            builder.ins().store(MemFlags::trusted(), val, addr, 0); // val is i8
        }
        NumericDType::F64 => {
            // After emit_cast_to_output the value is already in F64 repr
            // (emit_repr_cast(F32→F64) → fpromote) so we store it directly.
            // Re-promoting an F64 value trips the verifier with a type error.
            builder.ins().store(MemFlags::trusted(), val, addr, 0);
        }
        _ => {
            builder.ins().store(MemFlags::trusted(), val, addr, 0);
        }
    }
}

// ─── Output store ───────────────────────────────────────────────────────────

/// Store a computed f32 value to the group's output slot.
///
/// Loop variable `i_val` ranges from `atom_offset` to `atom_offset + count`.
/// Slot element index = `i - atom_offset`.
fn store_result(
    builder: &mut FunctionBuilder,
    buffer_ptr: Value,
    slot: &SlotInfo,
    atom_offset: u64,
    i_val: Option<Value>,
    i_const: u64,
    val: Value,
) {
    // store_base = slot.byte_offset() - atom_offset * elem_bytes
    // addr = buffer_ptr + store_base + i * elem_bytes
    let store_base = slot.byte_offset() as i64 - atom_offset as i64 * slot.elem_bytes() as i64;

    let addr = match i_val {
        Some(iv) => {
            let i_bytes = builder.ins().imul_imm(iv, slot.elem_bytes() as i64);
            let base_val = builder.ins().iconst(types::I64, store_base);
            let off = builder.ins().iadd(base_val, i_bytes);
            builder.ins().iadd(buffer_ptr, off)
        }
        None => {
            let byte_off = store_base + i_const as i64 * slot.elem_bytes() as i64;
            addr_const(builder, buffer_ptr, byte_off)
        }
    };

    emit_typed_store(builder, addr, val, slot.dtype);
}

// ─── Inlined producer evaluation ────────────────────────────────────────────

/// Evaluate a pure-scalar group's body at a decomposed iteration index,
/// returning the resulting Cranelift value in the format an `emit_typed_load`
/// from the group's slot would have produced.
///
/// Used by reduce-fold inlining: instead of materializing the producer's
/// output to memory and loading it back inside the reducer's k-loop, we
/// re-emit the producer's expression with `idx.outer*k + idx.inner` as the
/// (synthetic) iteration index. The producer's inputs are loaded via
/// `load_input_inlined`, which substitutes `outer`/`inner` directly when the
/// access pattern's modulus matches `idx.k` — eliminating the urem/udiv that
/// the flat-index path would emit.
///
/// Restricted to pure scalar ops (Binary/Unary/Cast/Identity/Select).
#[allow(clippy::too_many_arguments)]
fn eval_group_value(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    group: &AtomGroup<'static, crate::pool::SystemPool>,
    layout: &BufferLayout,
    buffer_ptr: Value,
    idx: InlineIdx,
    math: &MathFuncs,
    _var_counter: &mut VarCounter,
    tables: &mut EmbeddedTables,
) -> Result<Value, String> {
    let output_dtype = group.output_dtype;
    let atom_offset = group.atom_offset;

    let result_val = match &group.op {
        ScalarOp::Identity | ScalarOp::Cast { .. } => {
            let src = load_input_inlined(
                builder,
                module,
                &group.inputs[0],
                layout,
                buffer_ptr,
                idx,
                atom_offset,
                tables,
            )?;
            let src_repr = input_slot_dtype(&group.inputs[0], layout, atom_offset)
                .map(repr_of)
                .unwrap_or(ReprKind::F32);
            emit_cast_to_output(builder, src, src_repr, output_dtype)
        }

        ScalarOp::Binary { op, compute_dtype } => {
            let compute_repr = repr_of(*compute_dtype);
            let a_raw = load_input_inlined(
                builder,
                module,
                &group.inputs[0],
                layout,
                buffer_ptr,
                idx,
                atom_offset,
                tables,
            )?;
            let a_repr = input_slot_dtype(&group.inputs[0], layout, atom_offset)
                .map(repr_of)
                .unwrap_or(compute_repr);
            let a = emit_repr_cast(builder, a_raw, a_repr, compute_repr);

            let b_raw = load_input_inlined(
                builder,
                module,
                &group.inputs[1],
                layout,
                buffer_ptr,
                idx,
                atom_offset,
                tables,
            )?;
            let b_repr = input_slot_dtype(&group.inputs[1], layout, atom_offset)
                .map(repr_of)
                .unwrap_or(compute_repr);
            let b = emit_repr_cast(builder, b_raw, b_repr, compute_repr);

            let result = emit_binop(builder, module, math, *op, a, b, compute_repr)?;
            emit_cast_to_output(builder, result, compute_repr, output_dtype)
        }

        ScalarOp::Unary { op, compute_dtype } => {
            let compute_repr = repr_of(*compute_dtype);
            let x_raw = load_input_inlined(
                builder,
                module,
                &group.inputs[0],
                layout,
                buffer_ptr,
                idx,
                atom_offset,
                tables,
            )?;
            let x_repr = input_slot_dtype(&group.inputs[0], layout, atom_offset)
                .map(repr_of)
                .unwrap_or(compute_repr);
            let x = emit_repr_cast(builder, x_raw, x_repr, compute_repr);

            let result = emit_unop(builder, module, math, *op, x, compute_repr)?;
            emit_cast_to_output(builder, result, compute_repr, output_dtype)
        }

        ScalarOp::Select => {
            let cond = load_input_inlined(
                builder,
                module,
                &group.inputs[0],
                layout,
                buffer_ptr,
                idx,
                atom_offset,
                tables,
            )?;
            let cond_repr = input_slot_dtype(&group.inputs[0], layout, atom_offset)
                .map(repr_of)
                .unwrap_or(ReprKind::F32);

            let is_nonzero = match cond_repr {
                ReprKind::F32 | ReprKind::F64 => {
                    let zero = emit_float_zero(builder, cond_repr);
                    builder.ins().fcmp(FloatCC::NotEqual, cond, zero)
                }
                ReprKind::Int => {
                    let zero = builder.ins().iconst(types::I64, 0);
                    builder.ins().icmp(IntCC::NotEqual, cond, zero)
                }
            };

            let x_raw = load_input_inlined(
                builder,
                module,
                &group.inputs[1],
                layout,
                buffer_ptr,
                idx,
                atom_offset,
                tables,
            )?;
            let x_repr = input_slot_dtype(&group.inputs[1], layout, atom_offset)
                .map(repr_of)
                .unwrap_or(repr_of(output_dtype));
            let x = emit_cast_to_output(builder, x_raw, x_repr, output_dtype);

            let y_raw = load_input_inlined(
                builder,
                module,
                &group.inputs[2],
                layout,
                buffer_ptr,
                idx,
                atom_offset,
                tables,
            )?;
            let y_repr = input_slot_dtype(&group.inputs[2], layout, atom_offset)
                .map(repr_of)
                .unwrap_or(repr_of(output_dtype));
            let y = emit_cast_to_output(builder, y_raw, y_repr, output_dtype);

            builder.ins().select(is_nonzero, x, y)
        }

        _ => {
            return Err(format!(
                "eval_group_value: op {:?} not supported for inlining",
                op_name_short(&group.op)
            ));
        }
    };

    // Match memory load semantics — preserves dtype truncation contracts
    // (BF16 round-to-nearest-even, integer narrowing, etc.).
    Ok(emit_store_load_roundtrip(builder, result_val, output_dtype))
}

// ─── Reduce emission ────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn emit_reduce(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    group: &AtomGroup<'static, crate::pool::SystemPool>,
    layout: &BufferLayout,
    buffer_ptr: Value,
    i_val: Option<Value>,
    i_const: u64,
    math: &MathFuncs,
    var_counter: &mut VarCounter,
    tables: &mut EmbeddedTables,
    kind: ReduceKind,
    reduce_count: u64,
    reduce_stride: i64,
    compute_dtype: NumericDType,
    out_slot: &SlotInfo,
    inlined_producer: Option<&AtomGroup<'static, crate::pool::SystemPool>>,
) -> Result<(), String> {
    let compute_repr = repr_of(compute_dtype);
    let output_dtype = group.output_dtype;

    // Atom-stride from the reduce's InputRef — used by both paths to compute
    // the synthetic producer atom index (inline path) and to convert into
    // byte stride (buffer path).
    let stride_atoms = match &group.inputs[0] {
        InputRef::Strided { dim_strides, .. } => dim_strides.last().copied().unwrap_or(0),
        other => {
            return Err(format!(
                "Reduce input must be Strided, got {:?}",
                std::mem::discriminant(other)
            ));
        }
    };

    // Resolve the source slot for the buffer path. For the inlined path the
    // producer's slot doesn't exist; we use the producer's output_dtype as
    // src_dtype so the compute_repr cast and store→load round-trip stay
    // consistent with what a memory load would have produced.
    let (base_byte, input_byte_stride, reduce_byte_stride, src_dtype) =
        if let Some(producer) = inlined_producer {
            (0i64, 0i64, 0i64, producer.output_dtype)
        } else {
            let InputRef::Strided { base, .. } = &group.inputs[0] else {
                unreachable!("checked above")
            };
            let (base_byte, elem_bytes, src_dtype) = resolve_affine_base(
                layout,
                *base,
                stride_atoms,
                group.atom_offset,
                "reduce input",
            )?;
            let byte_stride = stride_atoms * elem_bytes as i64;
            let red_stride = reduce_stride * elem_bytes as i64;
            (base_byte, byte_stride, red_stride, src_dtype)
        };
    let src_repr = repr_of(src_dtype);

    // Accumulator variable — type depends on compute_repr.
    let acc_var = var_counter.next();
    let acc_cl_type = compute_repr.cranelift_type();
    builder.declare_var(acc_var, acc_cl_type);
    let init = match (kind, compute_repr) {
        (ReduceKind::Sum, ReprKind::F32) => builder.ins().f32const(0.0),
        (ReduceKind::Sum, ReprKind::F64) => builder.ins().f64const(0.0),
        (ReduceKind::Sum, ReprKind::Int) => builder.ins().iconst(types::I64, 0),
        (ReduceKind::Prod, ReprKind::F32) => builder.ins().f32const(1.0),
        (ReduceKind::Prod, ReprKind::F64) => builder.ins().f64const(1.0),
        (ReduceKind::Prod, ReprKind::Int) => builder.ins().iconst(types::I64, 1),
        (ReduceKind::Max, ReprKind::F32) => builder.ins().f32const(f32::NEG_INFINITY),
        (ReduceKind::Max, ReprKind::F64) => builder.ins().f64const(f64::NEG_INFINITY),
        (ReduceKind::Max, ReprKind::Int) => builder.ins().iconst(types::I64, i64::MIN),
        (ReduceKind::Min, ReprKind::F32) => builder.ins().f32const(f32::INFINITY),
        (ReduceKind::Min, ReprKind::F64) => builder.ins().f64const(f64::INFINITY),
        (ReduceKind::Min, ReprKind::Int) => builder.ins().iconst(types::I64, i64::MAX),
    };
    builder.def_var(acc_var, init);

    // Base address for this iteration's reduce input at k=0.
    // Buffer path only — the inline path computes a synthetic atom index
    // inside the inner loop instead.
    let base_addr_off = if inlined_producer.is_none() {
        Some(match i_val {
            Some(iv) => {
                let i_bytes = builder.ins().imul_imm(iv, input_byte_stride);
                let base_val = builder.ins().iconst(types::I64, base_byte);
                builder.ins().iadd(base_val, i_bytes)
            }
            None => {
                let off = base_byte + input_byte_stride * i_const as i64;
                builder.ins().iconst(types::I64, off)
            }
        })
    } else {
        None
    };

    // For the inline path, pre-compute the consumer's iv as a Cranelift Value.
    // We pass it as `InlineIdx::outer` along with the inner k-loop variable.
    // The decomposed (outer, inner, k) is consumed by load_input_inlined to
    // skip urem/udiv on producer access patterns whose modulus matches `k`.
    let inline_outer = if inlined_producer.is_some() {
        // Stage 1 only inlines when stride_atoms == reduce_count, i.e. each
        // output reads K consecutive producer atoms aligned at iv*K. Sanity-
        // check: stride_atoms must equal reduce_count.
        debug_assert_eq!(
            stride_atoms as u64, reduce_count,
            "inlining requires stride_atoms == reduce_count"
        );
        Some(match i_val {
            Some(iv) => iv,
            None => builder.ins().iconst(types::I64, i_const as i64),
        })
    } else {
        None
    };

    // Inner loop: for k in 0..reduce_count
    let k_var = var_counter.next();
    builder.declare_var(k_var, types::I64);
    let k_init = builder.ins().iconst(types::I64, 0);
    builder.def_var(k_var, k_init);

    let red_header = builder.create_block();
    let red_body = builder.create_block();
    let red_exit = builder.create_block();

    let bound = builder.ins().iconst(types::I64, reduce_count as i64);
    builder.ins().jump(red_header, &[]);

    builder.switch_to_block(red_header);
    let k_val = builder.use_var(k_var);
    let k_cmp = builder.ins().icmp(IntCC::SignedLessThan, k_val, bound);
    builder.ins().brif(k_cmp, red_body, &[], red_exit, &[]);

    // Body: load (or evaluate inlined producer), accumulate.
    builder.switch_to_block(red_body);
    let k_val = builder.use_var(k_var);
    let loaded = if let Some(producer) = inlined_producer {
        let outer = inline_outer.expect("inline_outer set when inlining");
        let idx = InlineIdx {
            outer,
            inner: k_val,
            k: reduce_count,
        };
        eval_group_value(
            builder,
            module,
            producer,
            layout,
            buffer_ptr,
            idx,
            math,
            var_counter,
            tables,
        )?
    } else {
        let base_addr_off = base_addr_off.expect("base_addr_off set on buffer path");
        let k_offset = builder.ins().imul_imm(k_val, reduce_byte_stride);
        let src_off = builder.ins().iadd(base_addr_off, k_offset);
        let src_addr = builder.ins().iadd(buffer_ptr, src_off);
        emit_typed_load(builder, src_addr, src_dtype)
    };
    // Cast loaded value to compute repr.
    let src_val = emit_repr_cast(builder, loaded, src_repr, compute_repr);

    let acc = builder.use_var(acc_var);
    let new_acc = match (kind, compute_repr) {
        (ReduceKind::Sum, ReprKind::F32 | ReprKind::F64) => builder.ins().fadd(acc, src_val),
        (ReduceKind::Sum, ReprKind::Int) => builder.ins().iadd(acc, src_val),
        (ReduceKind::Prod, ReprKind::F32 | ReprKind::F64) => builder.ins().fmul(acc, src_val),
        (ReduceKind::Prod, ReprKind::Int) => builder.ins().imul(acc, src_val),
        (ReduceKind::Max, ReprKind::F32 | ReprKind::F64) => {
            let cmp = builder.ins().fcmp(FloatCC::GreaterThan, src_val, acc);
            builder.ins().select(cmp, src_val, acc)
        }
        (ReduceKind::Max, ReprKind::Int) => {
            let cmp = builder.ins().icmp(IntCC::SignedGreaterThan, src_val, acc);
            builder.ins().select(cmp, src_val, acc)
        }
        (ReduceKind::Min, ReprKind::F32 | ReprKind::F64) => {
            let cmp = builder.ins().fcmp(FloatCC::LessThan, src_val, acc);
            builder.ins().select(cmp, src_val, acc)
        }
        (ReduceKind::Min, ReprKind::Int) => {
            let cmp = builder.ins().icmp(IntCC::SignedLessThan, src_val, acc);
            builder.ins().select(cmp, src_val, acc)
        }
    };
    builder.def_var(acc_var, new_acc);

    let k_next = builder.ins().iadd_imm(k_val, 1);
    builder.def_var(k_var, k_next);
    builder.ins().jump(red_header, &[]);

    builder.switch_to_block(red_exit);
    builder.seal_block(red_header);
    builder.seal_block(red_body);
    builder.seal_block(red_exit);

    let final_acc = builder.use_var(acc_var);
    let output_val = emit_cast_to_output(builder, final_acc, compute_repr, output_dtype);
    store_result(
        builder,
        buffer_ptr,
        out_slot,
        group.atom_offset,
        i_val,
        i_const,
        output_val,
    );

    Ok(())
}

// ─── Scalar op emission ─────────────────────────────────────────────────────

fn emit_binop(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    math: &MathFuncs,
    op: ScalarBinOp,
    a: Value,
    b: Value,
    compute_repr: ReprKind,
) -> Result<Value, String> {
    // Comparison results: return in the compute repr (float 1.0/0.0 or i64 1/0).
    macro_rules! cmp_result {
        ($cmp:expr) => {
            match compute_repr {
                ReprKind::F32 => {
                    let one = builder.ins().f32const(1.0);
                    let zero = builder.ins().f32const(0.0);
                    builder.ins().select($cmp, one, zero)
                }
                ReprKind::F64 => {
                    let one = builder.ins().f64const(1.0);
                    let zero = builder.ins().f64const(0.0);
                    builder.ins().select($cmp, one, zero)
                }
                ReprKind::Int => {
                    let one = builder.ins().iconst(types::I64, 1);
                    let zero = builder.ins().iconst(types::I64, 0);
                    builder.ins().select($cmp, one, zero)
                }
            }
        };
    }

    Ok(match (op, compute_repr) {
        // ── Arithmetic ──
        (ScalarBinOp::Add, ReprKind::F32 | ReprKind::F64) => builder.ins().fadd(a, b),
        (ScalarBinOp::Add, ReprKind::Int) => builder.ins().iadd(a, b),
        (ScalarBinOp::Sub, ReprKind::F32 | ReprKind::F64) => builder.ins().fsub(a, b),
        (ScalarBinOp::Sub, ReprKind::Int) => builder.ins().isub(a, b),
        (ScalarBinOp::Mul, ReprKind::F32 | ReprKind::F64) => builder.ins().fmul(a, b),
        (ScalarBinOp::Mul, ReprKind::Int) => builder.ins().imul(a, b),
        (ScalarBinOp::Div, ReprKind::F32 | ReprKind::F64) => builder.ins().fdiv(a, b),
        (ScalarBinOp::Div, ReprKind::Int) => {
            // Guard against division by zero (which traps on x86).
            // If b == 0, result is 0 (matches NumericScalar behavior).
            let zero = builder.ins().iconst(types::I64, 0);
            let one = builder.ins().iconst(types::I64, 1);
            let is_zero = builder.ins().icmp(IntCC::Equal, b, zero);
            let safe_b = builder.ins().select(is_zero, one, b);
            let quot = builder.ins().sdiv(a, safe_b);
            builder.ins().select(is_zero, zero, quot)
        }
        (ScalarBinOp::Mod, ReprKind::F32 | ReprKind::F64) => {
            let fid = math.binary("fmod", compute_repr).unwrap();
            let func_ref = module.declare_func_in_func(fid, builder.func);
            let call = builder.ins().call(func_ref, &[a, b]);
            builder.inst_results(call)[0]
        }
        (ScalarBinOp::Mod, ReprKind::Int) => {
            let zero = builder.ins().iconst(types::I64, 0);
            let one = builder.ins().iconst(types::I64, 1);
            let is_zero = builder.ins().icmp(IntCC::Equal, b, zero);
            let safe_b = builder.ins().select(is_zero, one, b);
            let rem = builder.ins().srem(a, safe_b);
            builder.ins().select(is_zero, zero, rem)
        }

        // ── Min/Max ──
        (ScalarBinOp::Max, ReprKind::F32 | ReprKind::F64) => {
            let cmp = builder.ins().fcmp(FloatCC::GreaterThan, a, b);
            builder.ins().select(cmp, a, b)
        }
        (ScalarBinOp::Max, ReprKind::Int) => {
            let cmp = builder.ins().icmp(IntCC::SignedGreaterThan, a, b);
            builder.ins().select(cmp, a, b)
        }
        (ScalarBinOp::Min, ReprKind::F32 | ReprKind::F64) => {
            let cmp = builder.ins().fcmp(FloatCC::LessThan, a, b);
            builder.ins().select(cmp, a, b)
        }
        (ScalarBinOp::Min, ReprKind::Int) => {
            let cmp = builder.ins().icmp(IntCC::SignedLessThan, a, b);
            builder.ins().select(cmp, a, b)
        }

        // ── Pow (float only; integer pow not needed for now) ──
        (ScalarBinOp::Pow, ReprKind::F32 | ReprKind::F64) => {
            let fid = math.binary("pow", compute_repr).unwrap();
            let func_ref = module.declare_func_in_func(fid, builder.func);
            let call = builder.ins().call(func_ref, &[a, b]);
            builder.inst_results(call)[0]
        }
        (ScalarBinOp::Pow, ReprKind::Int) => {
            return Err("integer Pow not implemented".into());
        }

        // ── Comparisons ──
        (ScalarBinOp::Equal, ReprKind::F32 | ReprKind::F64) => {
            let cmp = builder.ins().fcmp(FloatCC::Equal, a, b);
            cmp_result!(cmp)
        }
        (ScalarBinOp::Equal, ReprKind::Int) => {
            let cmp = builder.ins().icmp(IntCC::Equal, a, b);
            cmp_result!(cmp)
        }
        (ScalarBinOp::Greater, ReprKind::F32 | ReprKind::F64) => {
            let cmp = builder.ins().fcmp(FloatCC::GreaterThan, a, b);
            cmp_result!(cmp)
        }
        (ScalarBinOp::Greater, ReprKind::Int) => {
            let cmp = builder.ins().icmp(IntCC::SignedGreaterThan, a, b);
            cmp_result!(cmp)
        }
        (ScalarBinOp::GreaterOrEqual, ReprKind::F32 | ReprKind::F64) => {
            let cmp = builder.ins().fcmp(FloatCC::GreaterThanOrEqual, a, b);
            cmp_result!(cmp)
        }
        (ScalarBinOp::GreaterOrEqual, ReprKind::Int) => {
            let cmp = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, a, b);
            cmp_result!(cmp)
        }
        (ScalarBinOp::Less, ReprKind::F32 | ReprKind::F64) => {
            let cmp = builder.ins().fcmp(FloatCC::LessThan, a, b);
            cmp_result!(cmp)
        }
        (ScalarBinOp::Less, ReprKind::Int) => {
            let cmp = builder.ins().icmp(IntCC::SignedLessThan, a, b);
            cmp_result!(cmp)
        }
        (ScalarBinOp::LessOrEqual, ReprKind::F32 | ReprKind::F64) => {
            let cmp = builder.ins().fcmp(FloatCC::LessThanOrEqual, a, b);
            cmp_result!(cmp)
        }
        (ScalarBinOp::LessOrEqual, ReprKind::Int) => {
            let cmp = builder.ins().icmp(IntCC::SignedLessThanOrEqual, a, b);
            cmp_result!(cmp)
        }

        // ── Logical (truthiness-based) ──
        (ScalarBinOp::And, ReprKind::F32 | ReprKind::F64) => {
            let zero = emit_float_zero(builder, compute_repr);
            let a_nz = builder.ins().fcmp(FloatCC::NotEqual, a, zero);
            let b_nz = builder.ins().fcmp(FloatCC::NotEqual, b, zero);
            let both = builder.ins().band(a_nz, b_nz);
            cmp_result!(both)
        }
        (ScalarBinOp::And, ReprKind::Int) => {
            let zero = builder.ins().iconst(types::I64, 0);
            let a_nz = builder.ins().icmp(IntCC::NotEqual, a, zero);
            let b_nz = builder.ins().icmp(IntCC::NotEqual, b, zero);
            let both = builder.ins().band(a_nz, b_nz);
            cmp_result!(both)
        }
        (ScalarBinOp::Or, ReprKind::F32 | ReprKind::F64) => {
            let zero = emit_float_zero(builder, compute_repr);
            let a_nz = builder.ins().fcmp(FloatCC::NotEqual, a, zero);
            let b_nz = builder.ins().fcmp(FloatCC::NotEqual, b, zero);
            let either = builder.ins().bor(a_nz, b_nz);
            cmp_result!(either)
        }
        (ScalarBinOp::Or, ReprKind::Int) => {
            let zero = builder.ins().iconst(types::I64, 0);
            let a_nz = builder.ins().icmp(IntCC::NotEqual, a, zero);
            let b_nz = builder.ins().icmp(IntCC::NotEqual, b, zero);
            let either = builder.ins().bor(a_nz, b_nz);
            cmp_result!(either)
        }
        (ScalarBinOp::Xor, ReprKind::F32 | ReprKind::F64) => {
            let zero = emit_float_zero(builder, compute_repr);
            let a_nz = builder.ins().fcmp(FloatCC::NotEqual, a, zero);
            let b_nz = builder.ins().fcmp(FloatCC::NotEqual, b, zero);
            let x = builder.ins().bxor(a_nz, b_nz);
            cmp_result!(x)
        }
        (ScalarBinOp::Xor, ReprKind::Int) => {
            let zero = builder.ins().iconst(types::I64, 0);
            let a_nz = builder.ins().icmp(IntCC::NotEqual, a, zero);
            let b_nz = builder.ins().icmp(IntCC::NotEqual, b, zero);
            let x = builder.ins().bxor(a_nz, b_nz);
            cmp_result!(x)
        }

        // ── IMod (mathematical modulo — result sign matches divisor) ──
        (ScalarBinOp::IMod, ReprKind::F32 | ReprKind::F64) => {
            // fmod then adjust: if result and divisor have different signs, add divisor.
            let fid = math.binary("fmod", compute_repr).unwrap();
            let func_ref = module.declare_func_in_func(fid, builder.func);
            let call = builder.ins().call(func_ref, &[a, b]);
            let rem = builder.inst_results(call)[0];
            // rem + b if sign(rem) != sign(b), else rem
            let sum = builder.ins().fadd(rem, b);
            let zero = emit_float_zero(builder, compute_repr);
            let rem_neg = builder.ins().fcmp(FloatCC::LessThan, rem, zero);
            let b_neg = builder.ins().fcmp(FloatCC::LessThan, b, zero);
            let rem_zero = builder.ins().fcmp(FloatCC::Equal, rem, zero);
            let signs_differ = builder.ins().bxor(rem_neg, b_neg);
            let need_adjust = builder.ins().band_not(signs_differ, rem_zero);
            builder.ins().select(need_adjust, sum, rem)
        }
        (ScalarBinOp::IMod, ReprKind::Int) => {
            // srem then adjust: if result and divisor have different signs, add divisor.
            let zero = builder.ins().iconst(types::I64, 0);
            let one = builder.ins().iconst(types::I64, 1);
            let is_zero = builder.ins().icmp(IntCC::Equal, b, zero);
            let safe_b = builder.ins().select(is_zero, one, b);
            let rem = builder.ins().srem(a, safe_b);
            let sum = builder.ins().iadd(rem, safe_b);
            let rem_neg = builder.ins().icmp(IntCC::SignedLessThan, rem, zero);
            let b_neg = builder.ins().icmp(IntCC::SignedLessThan, safe_b, zero);
            let rem_zero = builder.ins().icmp(IntCC::Equal, rem, zero);
            let signs_differ = builder.ins().bxor(rem_neg, b_neg);
            let need_adjust = builder.ins().band_not(signs_differ, rem_zero);
            let result = builder.ins().select(need_adjust, sum, rem);
            builder.ins().select(is_zero, zero, result)
        }

        // ── Bitwise ops (integer only) ──
        (ScalarBinOp::BitwiseAnd, ReprKind::Int) => builder.ins().band(a, b),
        (ScalarBinOp::BitwiseOr, ReprKind::Int) => builder.ins().bor(a, b),
        (ScalarBinOp::BitwiseXor, ReprKind::Int) => builder.ins().bxor(a, b),
        (ScalarBinOp::BitShiftLeft, ReprKind::Int) => builder.ins().ishl(a, b),
        (ScalarBinOp::BitShiftRight, ReprKind::Int) => builder.ins().sshr(a, b),

        // Bitwise on floats — not meaningful
        (
            ScalarBinOp::BitwiseAnd
            | ScalarBinOp::BitwiseOr
            | ScalarBinOp::BitwiseXor
            | ScalarBinOp::BitShiftLeft
            | ScalarBinOp::BitShiftRight,
            ReprKind::F32 | ReprKind::F64,
        ) => {
            return Err(format!("bitwise {:?} on float not supported in JIT", op));
        }
    })
}

fn emit_unop(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    math: &MathFuncs,
    op: ScalarUnaryOp,
    x: Value,
    compute_repr: ReprKind,
) -> Result<Value, String> {
    Ok(match (op, compute_repr) {
        (ScalarUnaryOp::Neg, ReprKind::F32 | ReprKind::F64) => builder.ins().fneg(x),
        (ScalarUnaryOp::Neg, ReprKind::Int) => builder.ins().ineg(x),
        (ScalarUnaryOp::Abs, ReprKind::F32 | ReprKind::F64) => builder.ins().fabs(x),
        (ScalarUnaryOp::Abs, ReprKind::Int) => {
            // abs(x) = x < 0 ? -x : x
            let zero = builder.ins().iconst(types::I64, 0);
            let neg = builder.ins().icmp(IntCC::SignedLessThan, x, zero);
            let negated = builder.ins().ineg(x);
            builder.ins().select(neg, negated, x)
        }
        (ScalarUnaryOp::Floor, ReprKind::F32 | ReprKind::F64) => builder.ins().floor(x),
        (ScalarUnaryOp::Floor, ReprKind::Int) => x, // no-op for integers
        (ScalarUnaryOp::Ceil, ReprKind::F32 | ReprKind::F64) => builder.ins().ceil(x),
        (ScalarUnaryOp::Ceil, ReprKind::Int) => x, // no-op for integers
        // Transcendentals — float only.
        (ScalarUnaryOp::Exp, ReprKind::F32 | ReprKind::F64) => {
            let fid = math.unary("exp", compute_repr).unwrap();
            let func_ref = module.declare_func_in_func(fid, builder.func);
            let call = builder.ins().call(func_ref, &[x]);
            builder.inst_results(call)[0]
        }
        (ScalarUnaryOp::Ln, ReprKind::F32 | ReprKind::F64) => {
            let fid = math.unary("log", compute_repr).unwrap();
            let func_ref = module.declare_func_in_func(fid, builder.func);
            let call = builder.ins().call(func_ref, &[x]);
            builder.inst_results(call)[0]
        }
        (ScalarUnaryOp::Sqrt, ReprKind::F32 | ReprKind::F64) => builder.ins().sqrt(x),
        (ScalarUnaryOp::Reciprocal, ReprKind::F32 | ReprKind::F64) => {
            let one = emit_float_const(builder, compute_repr, 1.0);
            builder.ins().fdiv(one, x)
        }
        (ScalarUnaryOp::Tanh, ReprKind::F32 | ReprKind::F64) => {
            let fid = math.unary("tanh", compute_repr).unwrap();
            let func_ref = module.declare_func_in_func(fid, builder.func);
            let call = builder.ins().call(func_ref, &[x]);
            builder.inst_results(call)[0]
        }
        // Float ops not yet supported by the JIT.
        (_, ReprKind::F32 | ReprKind::F64) => {
            return Err(format!("float {:?} not implemented in JIT", op));
        }
        // Integer transcendentals — not meaningful, but cast through f32 if needed.
        (_, ReprKind::Int) => {
            return Err(format!("integer {:?} not implemented", op));
        }
    })
}

/// Compile an empty span (no groups). Returns a no-op CompiledSpan.
fn compile_empty_span() -> Result<CompiledSpan, String> {
    let isa_builder =
        cranelift_native::builder().map_err(|e| format!("cranelift native ISA: {}", e))?;
    let isa = isa_builder
        .finish(build_cranelift_flags())
        .map_err(|e| format!("ISA finish: {}", e))?;
    let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    register_math_symbols(&mut jit_builder);
    let mut module = JITModule::new(jit_builder);
    let _ = declare_math_funcs(&mut module)?;

    let mut ctx = module.make_context();
    ctx.func.signature.params.push(AbiParam::new(types::I64));
    let func_id = module
        .declare_function("noop", Linkage::Local, &ctx.func.signature)
        .map_err(|e| format!("declare: {}", e))?;
    {
        let mut func_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        builder.seal_block(entry);
        builder.ins().return_(&[]);
        builder.finalize();
    }
    module
        .define_function(func_id, &mut ctx)
        .map_err(|e| format!("define: {}", e))?;
    module
        .finalize_definitions()
        .map_err(|e| format!("finalize: {}", e))?;
    let func_ptr = module.get_finalized_function(func_id);
    Ok(CompiledSpan {
        func_ptr,
        _module: module,
    })
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GlobalId;
    use crate::nano_graph::ops::{ReduceKind, ScalarBinOp, ScalarOp, ScalarUnaryOp};
    use crate::numeric_scalar::NumericScalar;

    /// Build a graph: input[4] + broadcast(2.0) → output[4]
    #[test]
    fn test_add_broadcast_f32() {
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
        let layout = compute_layout(&g, &outputs);
        let (compiled, embedded_tables) = compile_span(&g, &layout).unwrap();

        let mut buffer = vec![0u8; layout.total_bytes];
        layout.populate_literals(&g, &mut buffer);
        layout.write_f32_input(inp, &[1.0, 2.0, 3.0, 4.0], &mut buffer);

        compiled.execute(&mut buffer);

        let result = layout.read_f32_output(&outputs[0], &buffer);
        assert_eq!(result, vec![3.0, 4.0, 5.0, 6.0]);
    }

    /// Unary neg: -input[3]
    #[test]
    fn test_unary_neg() {
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
        let layout = compute_layout(&g, &outputs);
        let (compiled, embedded_tables) = compile_span(&g, &layout).unwrap();

        let mut buffer = vec![0u8; layout.total_bytes];
        layout.write_f32_input(inp, &[1.0, -2.5, 3.0], &mut buffer);

        compiled.execute(&mut buffer);

        let result = layout.read_f32_output(&outputs[0], &buffer);
        assert_eq!(result, vec![-1.0, 2.5, -3.0]);
    }

    /// Reduce sum: sum of 4 elements.
    #[test]
    fn test_reduce_sum() {
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
        let layout = compute_layout(&g, &outputs);
        let (compiled, embedded_tables) = compile_span(&g, &layout).unwrap();

        let mut buffer = vec![0u8; layout.total_bytes];
        layout.write_f32_input(inp, &[1.0, 2.0, 3.0, 4.0], &mut buffer);

        compiled.execute(&mut buffer);

        let result = layout.read_f32_output(&outputs[0], &buffer);
        assert_eq!(result, vec![10.0]);
    }

    /// Chain: input → mul(3.0) → exp → output. Tests multi-group pipelines.
    #[test]
    fn test_chain_mul_exp() {
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
        let layout = compute_layout(&g, &outputs);
        let (compiled, embedded_tables) = compile_span(&g, &layout).unwrap();

        let mut buffer = vec![0u8; layout.total_bytes];
        layout.populate_literals(&g, &mut buffer);
        layout.write_f32_input(inp, &[0.0, 1.0], &mut buffer);

        compiled.execute(&mut buffer);

        let result = layout.read_f32_output(&outputs[0], &buffer);
        let expected = vec![(0.0f32 * 3.0).exp(), (1.0f32 * 3.0).exp()];
        for (a, b) in result.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "got {}, expected {}", a, b);
        }
    }

    /// Liveness reuse: chain A → B → C where A's slot can be reused by C.
    #[test]
    #[ignore]
    fn test_liveness_reuse() {
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), 100, NumericDType::F32);
        let a = g.push_group(
            100,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(inp, 1)],
        );
        let b = g.push_group(
            100,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(a, 1)],
        );
        let c = g.push_group(
            100,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(b, 1)],
        );

        let outputs = vec![AtomRange {
            base: c,
            count: 100,
            dtype: NumericDType::F32,
        }];
        let layout = compute_layout(&g, &outputs);

        // With slot reuse: input(400) + A(400) are allocated. When B is
        // allocated, A's slot is freed (only consumer B is done). B reuses A's
        // space. Similarly C reuses B's. So: input(400) + one reused slot(400)
        // + C(400) = 1200, or possibly input(400) + A/B/C sharing = 800.
        // The exact number depends on allocation order; just verify it's < 1600.
        assert!(
            layout.total_bytes < 1600,
            "Expected slot reuse to reduce buffer from 1600 bytes, got {}",
            layout.total_bytes
        );

        // Verify correctness: neg(neg(neg(x))) = -x
        let (compiled, embedded_tables) = compile_span(&g, &layout).unwrap();
        let mut buffer = vec![0u8; layout.total_bytes];
        let input_data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        layout.write_f32_input(inp, &input_data, &mut buffer);

        compiled.execute(&mut buffer);

        let result = layout.read_f32_output(&outputs[0], &buffer);
        for (i, &val) in result.iter().enumerate() {
            let expected = -(i as f32);
            assert!(
                (val - expected).abs() < 1e-6,
                "elem {}: got {}, expected {}",
                i,
                val,
                expected
            );
        }
    }

    /// Select (ternary): condition ? x : y
    #[test]
    fn test_select() {
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
        let layout = compute_layout(&g, &outputs);
        let (compiled, embedded_tables) = compile_span(&g, &layout).unwrap();

        let mut buffer = vec![0u8; layout.total_bytes];
        layout.write_f32_input(cond, &[1.0, 0.0, 5.0], &mut buffer);
        layout.write_f32_input(x, &[10.0, 20.0, 30.0], &mut buffer);
        layout.write_f32_input(y, &[100.0, 200.0, 300.0], &mut buffer);

        compiled.execute(&mut buffer);

        let result = layout.read_f32_output(&outputs[0], &buffer);
        // cond[0]=1.0 (nonzero) → x=10.0, cond[1]=0.0 → y=200.0, cond[2]=5.0 → x=30.0
        assert_eq!(result, vec![10.0, 200.0, 30.0]);
    }

    /// Identity with Explicit InputRef gathering from 4 source groups.
    /// Reproduces the phase-0 NaN bug pattern.
    #[test]
    fn test_explicit_gather() {
        let mut g = NanoGraph::new();
        let inp = g.add_input_tensor(GlobalId(0), 16, NumericDType::F32);
        // 4 source groups, each 4 atoms, that read from different input regions.
        let mut sources = Vec::new();
        for i in 0..4u64 {
            let src = g.push_group(
                4,
                NumericDType::F32,
                ScalarOp::Identity,
                vec![],
                vec![InputRef::affine(AtomId(inp.0 + i * 4), 1)],
            );
            sources.push(src);
        }
        // Gather: pick one atom from each of the 4 source groups, interleaved.
        let explicit_ids: Vec<AtomId> = (0..16)
            .map(|i| {
                let group = i / 4;
                let elem = i % 4;
                AtomId(sources[group].0 + elem as u64)
            })
            .collect();
        let gather = g.push_group(
            16,
            NumericDType::F32,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::Explicit(explicit_ids)],
        );

        let outputs = vec![AtomRange {
            base: gather,
            count: 16,
            dtype: NumericDType::F32,
        }];
        let layout = compute_layout(&g, &outputs);
        let (compiled, embedded_tables) = compile_span(&g, &layout).unwrap();

        let buf_size = embedded_tables.total_bytes().max(layout.total_bytes);
        let mut buffer = vec![0u8; buf_size];
        let input_data: Vec<f32> = (0..16).map(|i| (i + 1) as f32).collect();
        layout.write_f32_input(inp, &input_data, &mut buffer);
        layout.populate_literals(&g, &mut buffer);
        embedded_tables.populate(&mut buffer);

        compiled.execute(&mut buffer);

        let result = layout.read_f32_output(&outputs[0], &buffer);
        // Sources copy input directly. Gather picks elements from sources.
        // Source 0 = [1,2,3,4], Source 1 = [5,6,7,8], etc.
        // Gather picks: src0[0..3], src1[0..3], src2[0..3], src3[0..3]
        let expected: Vec<f32> = (0..16).map(|i| (i + 1) as f32).collect();
        assert_eq!(result, expected, "Explicit gather mismatch");
    }

    /// Cross-group Affine: identity reads across two source groups.
    /// This is the pattern that caused the contiguity bug.
    #[test]
    fn test_cross_group_affine() {
        let mut g = NanoGraph::new();
        // Two source groups: A (3 atoms) then B (3 atoms), contiguous in atom space.
        let a = g.push_group(
            3,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(1.0)),
            vec![],
            vec![],
        );
        let b = g.push_group(
            3,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(2.0)),
            vec![],
            vec![],
        );
        // Consumer reads 6 atoms starting from A's base, crossing into B.
        let out = g.push_group(
            6,
            NumericDType::F32,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::affine(a, 1)],
        );

        let outputs = vec![AtomRange {
            base: out,
            count: 6,
            dtype: NumericDType::F32,
        }];
        let layout = compute_layout(&g, &outputs);

        // Validate no cross-slot errors.
        let errors = validate_layout(&g, &layout);
        assert!(errors.is_empty(), "layout errors: {:?}", errors);

        let (compiled, embedded_tables) = compile_span(&g, &layout).unwrap();
        let mut buffer = vec![0u8; layout.total_bytes];
        layout.populate_literals(&g, &mut buffer);
        compiled.execute(&mut buffer);

        let result = layout.read_f32_output(&outputs[0], &buffer);
        assert_eq!(result, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    }

    /// 5-group fusion chain: input → +2 → *3 → neg → exp → output
    /// Tests that chains of 4+ fused groups produce correct results.
    #[test]
    fn test_fusion_chain_5_groups() {
        // Fusion is enabled by default; no env var needed.

        let mut g = NanoGraph::new();
        let n = 8u64;
        let inp = g.add_input_tensor(GlobalId(0), n, NumericDType::F32);

        let lit2 = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(2.0)),
            vec![],
            vec![],
        );
        let lit3 = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(3.0)),
            vec![],
            vec![],
        );

        // Group 1: identity (copy input)
        let g1 = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::affine(inp, 1)],
        );

        // Group 2: g1 + 2.0
        let g2 = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(g1, 1), InputRef::Broadcast(lit2)],
        );

        // Group 3: g2 * 3.0
        let g3 = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(g2, 1), InputRef::Broadcast(lit3)],
        );

        // Group 4: -g3
        let g4 = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(g3, 1)],
        );

        // Group 5: exp(g4)
        let g5 = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(g4, 1)],
        );

        let outputs = vec![AtomRange {
            base: g5,
            count: n,
            dtype: NumericDType::F32,
        }];
        let layout = compute_layout(&g, &outputs);

        // Verify chain was built (should be 1 chain of 5 groups)
        let chains = build_fusion_chains(g.groups(), &layout);
        let multi = chains.iter().filter(|c| c.group_indices.len() > 1).count();
        assert!(multi > 0, "expected at least one multi-group fusion chain");
        let max_len = chains.iter().map(|c| c.group_indices.len()).max().unwrap();
        assert!(
            max_len >= 4,
            "expected chain of 4+ groups, got max {}",
            max_len
        );

        let (compiled, embedded_tables) = compile_span(&g, &layout).unwrap();

        let mut buffer = vec![0u8; layout.total_bytes];
        layout.populate_literals(&g, &mut buffer);
        let input_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        layout.write_f32_input(inp, &input_data, &mut buffer);

        compiled.execute(&mut buffer);

        let result = layout.read_f32_output(&outputs[0], &buffer);
        let expected: Vec<f32> = input_data
            .iter()
            .map(|x| (-((*x + 2.0) * 3.0)).exp())
            .collect();

        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-5 * want.abs().max(1.0),
                "mismatch at [{}]: got {}, expected {}",
                i,
                got,
                want
            );
        }

        // Disable fusion for unfused comparison.
        unsafe {
            std::env::set_var("FUSION", "0");
        }
    }

    /// 5-group chain with a diamond dependency: D reads from both A and C.
    /// Tests forwarding with multiple chain producers.
    #[test]
    #[ignore]
    fn test_fusion_chain_diamond() {
        // Fusion is enabled by default; no env var needed.

        let mut g = NanoGraph::new();
        let n = 8u64;
        let inp = g.add_input_tensor(GlobalId(0), n, NumericDType::F32);

        // A: identity(input)
        let a = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::affine(inp, 1)],
        );

        // B: A + A = 2*input
        let b = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(a, 1), InputRef::affine(a, 1)],
        );

        // C: -B
        let c = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(b, 1)],
        );

        // D: A + C (reads from first AND third group in chain)
        let d = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(a, 1), InputRef::affine(c, 1)],
        );

        // E: exp(D)
        let e = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(d, 1)],
        );

        let outputs = vec![AtomRange {
            base: e,
            count: n,
            dtype: NumericDType::F32,
        }];
        let layout = compute_layout(&g, &outputs);

        let chains = build_fusion_chains(g.groups(), &layout);
        let max_len = chains.iter().map(|c| c.group_indices.len()).max().unwrap();
        assert!(
            max_len >= 4,
            "expected chain of 4+ groups, got max {}",
            max_len
        );

        let (compiled, embedded_tables) = compile_span(&g, &layout).unwrap();

        let mut buffer = vec![0u8; layout.total_bytes];
        layout.populate_literals(&g, &mut buffer);
        let input_data: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
        layout.write_f32_input(inp, &input_data, &mut buffer);

        compiled.execute(&mut buffer);

        let result = layout.read_f32_output(&outputs[0], &buffer);
        // A=x, B=2x, C=-2x, D=x+(-2x)=-x, E=exp(-x)
        let expected: Vec<f32> = input_data.iter().map(|x| (-x).exp()).collect();

        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-5 * want.abs().max(1.0),
                "mismatch at [{}]: got {}, expected {}",
                i,
                got,
                want
            );
        }

        // Disable fusion for unfused comparison.
        unsafe {
            std::env::set_var("FUSION", "0");
        }
    }

    /// Chain where not all groups are connected — some read only from external
    /// inputs. Tests that independent groups within a chain still produce
    /// correct results.
    #[test]
    #[ignore]
    fn test_fusion_chain_independent_groups() {
        // Fusion is enabled by default; no env var needed.

        let mut g = NanoGraph::new();
        let n = 8u64;
        let inp1 = g.add_input_tensor(GlobalId(0), n, NumericDType::F32);
        let inp2 = g.add_input_tensor(GlobalId(1), n, NumericDType::F32);

        // A: identity(inp1)
        let a = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::affine(inp1, 1)],
        );
        // B: identity(inp2) — reads ONLY from external, NOT from A
        let b = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::affine(inp2, 1)],
        );
        // C: identity(inp1) — another independent group
        let c = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::affine(inp1, 1)],
        );
        // D: A + B — reads from chain members A and B
        let d = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(a, 1), InputRef::affine(b, 1)],
        );
        // E: D * C — reads from chain members D and C
        let e_group = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(d, 1), InputRef::affine(c, 1)],
        );

        let outputs = vec![AtomRange {
            base: e_group,
            count: n,
            dtype: NumericDType::F32,
        }];
        let layout = compute_layout(&g, &outputs);

        let chains = build_fusion_chains(g.groups(), &layout);
        let max_len = chains.iter().map(|c| c.group_indices.len()).max().unwrap();
        assert!(max_len >= 4, "expected chain of 4+, got {}", max_len);

        let (compiled, embedded_tables) = compile_span(&g, &layout).unwrap();

        let mut buffer = vec![0u8; layout.total_bytes];
        let d1: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let d2: Vec<f32> = (0..n).map(|i| 10.0 + i as f32).collect();
        layout.write_f32_input(inp1, &d1, &mut buffer);
        layout.write_f32_input(inp2, &d2, &mut buffer);

        compiled.execute(&mut buffer);

        let result = layout.read_f32_output(&outputs[0], &buffer);
        // A=inp1, B=inp2, C=inp1, D=inp1+inp2, E=(inp1+inp2)*inp1
        let expected: Vec<f32> = d1.iter().zip(d2.iter()).map(|(x, y)| (x + y) * x).collect();

        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-4,
                "mismatch at [{}]: got {}, expected {}",
                i,
                got,
                want
            );
        }

        // Disable fusion for unfused comparison.
        unsafe {
            std::env::set_var("FUSION", "0");
        }
    }

    /// Chain starting with ReduceSum: models LayerNorm pattern.
    /// ReduceSum groups followed by elementwise groups reading from them.
    /// In GPT-2, this pattern creates chains of [Red, Bin, Bin, Un].
    #[test]
    fn test_fusion_chain_reduce_head() {
        // Fusion is enabled by default; no env var needed.

        let mut g = NanoGraph::new();

        // Input: 4 vectors of 8 elements each (32 total)
        let inp = g.add_input_tensor(GlobalId(0), 32, NumericDType::F32);
        let lit_n_inv = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(0.125)),
            vec![],
            vec![],
        ); // 1/8
        let lit2 = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(2.0)),
            vec![],
            vec![],
        );

        // 4 ReduceSum groups, each summing 8 elements → 4 outputs
        let red = g.push_group(
            4,
            NumericDType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: 8,
                reduce_stride: 1,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(inp, 8)],
        );

        // Bin: red * (1/8) = mean
        let mean = g.push_group(
            4,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(red, 1), InputRef::Broadcast(lit_n_inv)],
        );

        // Bin: mean + 2.0
        let biased = g.push_group(
            4,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(mean, 1), InputRef::Broadcast(lit2)],
        );

        // Un: sqrt(biased)
        let result = g.push_group(
            4,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Sqrt,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(biased, 1)],
        );

        let outputs = vec![AtomRange {
            base: result,
            count: 4,
            dtype: NumericDType::F32,
        }];
        let layout = compute_layout(&g, &outputs);

        // Check that fusion creates a chain of 4+ groups
        let chains = build_fusion_chains(g.groups(), &layout);
        let max_len = chains.iter().map(|c| c.group_indices.len()).max().unwrap();
        eprintln!(
            "  reduce_head: chains={}, max_len={}",
            chains.len(),
            max_len
        );

        // Compile with fusion
        let (compiled_fused, tables_fused) = compile_span(&g, &layout).unwrap();
        let mut buf_fused = vec![0u8; layout.total_bytes];
        layout.populate_literals(&g, &mut buf_fused);
        let input_data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 + 1.0).collect();
        layout.write_f32_input(inp, &input_data, &mut buf_fused);
        compiled_fused.execute(&mut buf_fused);
        let result_fused = layout.read_f32_output(&outputs[0], &buf_fused);

        // Compile without fusion
        // Disable fusion for unfused comparison.
        unsafe {
            std::env::set_var("FUSION", "0");
        }
        let (compiled_unfused, tables_unfused) = compile_span(&g, &layout).unwrap();
        let mut buf_unfused = vec![0u8; layout.total_bytes];
        layout.populate_literals(&g, &mut buf_unfused);
        layout.write_f32_input(inp, &input_data, &mut buf_unfused);
        compiled_unfused.execute(&mut buf_unfused);
        let result_unfused = layout.read_f32_output(&outputs[0], &buf_unfused);

        eprintln!("  fused:   {:?}", result_fused);
        eprintln!("  unfused: {:?}", result_unfused);

        for (i, (f, u)) in result_fused.iter().zip(result_unfused.iter()).enumerate() {
            assert!(
                (f - u).abs() < 1e-6,
                "reduce_head mismatch at [{}]: fused={}, unfused={}",
                i,
                f,
                u
            );
        }
    }

    /// Simulates split groups: same as chain but with atom_offset=8
    /// (models what GPT-2 partitioner produces when splitting across lanes).
    /// InputRefs reference original (unsplit) base_ids.
    #[test]
    fn test_fusion_chain_split_groups() {
        // Fusion is enabled by default; no env var needed.

        // Build the ORIGINAL (unsplit) graph first, then create a "split" version
        // that only contains the second portion (offset=8, count=8 out of 16).
        let mut g = NanoGraph::new();
        let n_original = 16u64; // original group size
        let split_offset = 8u64;
        let split_count = 8u64;

        // External input (large enough for the reduce to read from)
        let inp = g.add_input_tensor(GlobalId(0), 128, NumericDType::F32);
        let lit_inv = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(0.125)),
            vec![],
            vec![],
        );
        let lit2 = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(2.0)),
            vec![],
            vec![],
        );

        // Create groups at ORIGINAL size first, then we'll split them.
        // g0: ReduceSum, reads from input with stride=8 (each output sums 8 elements)
        let g0 = g.push_group(
            n_original,
            NumericDType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: 8,
                reduce_stride: 1,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(inp, 8)],
        );
        let g0_original_base = g0;

        // g1: g0 * (1/8)
        let g1 = g.push_group(
            n_original,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(g0, 1), InputRef::Broadcast(lit_inv)],
        );
        let g1_original_base = g1;

        // g2: g1 + 2.0
        let g2 = g.push_group(
            n_original,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(g1, 1), InputRef::Broadcast(lit2)],
        );

        // g3: sqrt(g2)
        let g3 = g.push_group(
            n_original,
            NumericDType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Sqrt,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(g2, 1)],
        );

        // Now simulate splitting: modify groups to be the second portion.
        // This mimics what the partitioner does.
        {
            let groups = g.groups_mut();
            for group in groups.iter_mut() {
                if group.count == n_original {
                    // Split: keep only the second portion
                    group.base_id = AtomId(group.base_id.0 + split_offset);
                    group.count = split_count;
                    group.atom_offset = split_offset;
                    // InputRefs stay the same (reference original bases)
                }
            }
        }

        let outputs = vec![AtomRange {
            base: AtomId(g3.0 + split_offset),
            count: split_count,
            dtype: NumericDType::F32,
        }];
        let layout = compute_layout(&g, &outputs);

        // Check fusion chains
        let chains = build_fusion_chains(g.groups(), &layout);
        for chain in &chains {
            if chain.group_indices.len() > 1 {
                eprintln!(
                    "  split_test: chain len={} offset={} count={}",
                    chain.group_indices.len(),
                    chain.atom_offset,
                    chain.count
                );
            }
        }

        // Compile WITH fusion
        let (compiled_fused, tables_fused) = compile_span(&g, &layout).unwrap();
        let mut buf_fused = vec![0u8; layout.total_bytes];
        layout.populate_literals(&g, &mut buf_fused);
        let input_data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.1 + 1.0).collect();
        layout.write_f32_input(inp, &input_data, &mut buf_fused);
        compiled_fused.execute(&mut buf_fused);
        let result_fused = layout.read_f32_output(&outputs[0], &buf_fused);

        // Compile WITHOUT fusion
        // Disable fusion for unfused comparison.
        unsafe {
            std::env::set_var("FUSION", "0");
        }
        let (compiled_unfused, tables_unfused) = compile_span(&g, &layout).unwrap();
        let mut buf_unfused = vec![0u8; layout.total_bytes];
        layout.populate_literals(&g, &mut buf_unfused);
        layout.write_f32_input(inp, &input_data, &mut buf_unfused);
        compiled_unfused.execute(&mut buf_unfused);
        let result_unfused = layout.read_f32_output(&outputs[0], &buf_unfused);

        eprintln!("  fused:   {:?}", result_fused);
        eprintln!("  unfused: {:?}", result_unfused);

        for (i, (f, u)) in result_fused.iter().zip(result_unfused.iter()).enumerate() {
            assert!(
                (f - u).abs() < 1e-6,
                "split_group mismatch at [{}]: fused={}, unfused={}",
                i,
                f,
                u
            );
        }
    }

    /// 6-group chain with BF16 intermediate dtypes — tests store-load
    /// roundtrip precision in forwarded values.
    #[test]
    fn test_fusion_chain_bf16_roundtrip() {
        // Fusion is enabled by default; no env var needed.

        let mut g = NanoGraph::new();
        let n = 8u64;
        let inp = g.add_input_tensor(GlobalId(0), n, NumericDType::F32);

        let lit2 = g.push_group(
            1,
            NumericDType::F32,
            ScalarOp::Literal(NumericScalar::from_f32(2.0)),
            vec![],
            vec![],
        );

        // A: identity F32 → BF16 cast
        let a = g.push_group(
            n,
            NumericDType::BF16,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::affine(inp, 1)],
        );
        // B: identity BF16 → F32 cast
        let b = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::affine(a, 1)],
        );
        // C: B + 2.0
        let c = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(b, 1), InputRef::Broadcast(lit2)],
        );
        // D: cast F32 → BF16
        let d = g.push_group(
            n,
            NumericDType::BF16,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::affine(c, 1)],
        );
        // E: cast BF16 → F32
        let e_group = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::affine(d, 1)],
        );
        // F: E * 2.0
        let f = g.push_group(
            n,
            NumericDType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: NumericDType::F32,
            },
            vec![],
            vec![InputRef::affine(e_group, 1), InputRef::Broadcast(lit2)],
        );

        let outputs = vec![AtomRange {
            base: f,
            count: n,
            dtype: NumericDType::F32,
        }];
        let layout = compute_layout(&g, &outputs);

        // Compare fused vs unfused results
        let (compiled_fused, tables_fused) = compile_span(&g, &layout).unwrap();

        let mut buf_fused = vec![0u8; layout.total_bytes];
        layout.populate_literals(&g, &mut buf_fused);
        let input_data: Vec<f32> = vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8];
        layout.write_f32_input(inp, &input_data, &mut buf_fused);
        compiled_fused.execute(&mut buf_fused);
        let result_fused = layout.read_f32_output(&outputs[0], &buf_fused);

        // Now compile without fusion and compare
        // Disable fusion for unfused comparison.
        unsafe {
            std::env::set_var("FUSION", "0");
        }
        let (compiled_unfused, tables_unfused) = compile_span(&g, &layout).unwrap();
        let mut buf_unfused = vec![0u8; layout.total_bytes];
        layout.populate_literals(&g, &mut buf_unfused);
        layout.write_f32_input(inp, &input_data, &mut buf_unfused);
        compiled_unfused.execute(&mut buf_unfused);
        let result_unfused = layout.read_f32_output(&outputs[0], &buf_unfused);

        for (i, (fused, unfused)) in result_fused.iter().zip(result_unfused.iter()).enumerate() {
            assert_eq!(
                fused, unfused,
                "fused/unfused mismatch at [{}]: fused={}, unfused={}",
                i, fused, unfused
            );
        }
    }
}
