#![allow(clippy::all, dead_code, unreachable_patterns, unused)]
//! Cranelift JIT codegen for NanoGraph with proper dtype support.
//!
//! Compiles a NanoGraph into native code. Each "kernel" is a set of groups
//! that are executed together. For this first pass, each kernel is simply
//! the full set of groups executed in topological (insertion) order.
//!
//! Function signature: `fn(values: *mut f32) -> ()`
//! The `values` pointer addresses a flat f32 array indexed by AtomId.0.
//!
//! ## Dtype strategy (Option C)
//!
//! The buffer remains `*mut f32` — every atom occupies one f32 slot. Dtype
//! semantics are enforced by applying precision-rounding at store boundaries:
//!
//! - When `output_dtype` is BF16, the f32 result is rounded to BF16 precision
//!   (7 mantissa bits) before storing. The stored value is still a valid f32
//!   that happens to be exactly representable in BF16.
//! - When `output_dtype` is F16, the f32 result is rounded to F16 precision
//!   (10 mantissa bits) via the `half` crate.
//! - For integer dtypes (I32, I64), values are stored as f32 casts. This is
//!   lossy for large integers but correct for typical index values.
//! - For Bool, nonzero is stored as 1.0, zero as 0.0.
//!
//! This gives correct numerical results (matching the reference evaluator's
//! precision) without changing the buffer layout.

use std::collections::HashMap;

use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, Value};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use crate::dtype::DType;
use crate::nano_graph::{
    AtomGroup, AtomId, InputRef, NanoGraph, ScalarBinOp, ScalarOp, ScalarUnaryOp, SymDim,
};
use crate::numeric_scalar::NumericScalar;

// ---- Math function wrappers (extern "C" for Cranelift calls) ----

extern "C" fn v13_expf(x: f32) -> f32 {
    x.exp()
}
extern "C" fn v13_logf(x: f32) -> f32 {
    x.ln()
}
extern "C" fn v13_tanhf(x: f32) -> f32 {
    x.tanh()
}
extern "C" fn v13_sqrtf(x: f32) -> f32 {
    x.sqrt()
}
extern "C" fn v13_floorf(x: f32) -> f32 {
    x.floor()
}
extern "C" fn v13_ceilf(x: f32) -> f32 {
    x.ceil()
}
extern "C" fn v13_fabsf(x: f32) -> f32 {
    x.abs()
}
extern "C" fn v13_powf(x: f32, y: f32) -> f32 {
    x.powf(y)
}
extern "C" fn v13_fmodf(x: f32, y: f32) -> f32 {
    x % y
}

// ---- Dtype rounding helpers (called from JIT via extern "C") ----

/// Round an f32 value to BF16 precision (round-to-nearest-even), return as f32.
///
/// BF16 has 1 sign + 8 exponent + 7 mantissa bits. This truncates the lower
/// 16 mantissa bits with proper rounding.
extern "C" fn v13_round_bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    // Round to nearest even: add bias that depends on the truncation boundary bit.
    // The 0x7FFF is the half-way point for the truncated bits.
    // Adding ((bits >> 16) & 1) implements round-to-nearest-even.
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    f32::from_bits(rounded & 0xFFFF0000)
}

/// Round an f32 value to F16 precision, return as f32.
extern "C" fn v13_round_f16(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

// ---- Math function declarations in the JIT module ----

struct MathFuncs {
    expf: cranelift_module::FuncId,
    logf: cranelift_module::FuncId,
    tanhf: cranelift_module::FuncId,
    sqrtf: cranelift_module::FuncId,
    floorf: cranelift_module::FuncId,
    ceilf: cranelift_module::FuncId,
    fabsf: cranelift_module::FuncId,
    powf: cranelift_module::FuncId,
    fmodf: cranelift_module::FuncId,
    round_bf16: cranelift_module::FuncId,
    round_f16: cranelift_module::FuncId,
}

fn declare_math_funcs(module: &mut JITModule) -> Result<MathFuncs, String> {
    let mut sig1 = module.make_signature();
    sig1.params.push(AbiParam::new(types::F32));
    sig1.returns.push(AbiParam::new(types::F32));

    let mut sig2 = module.make_signature();
    sig2.params.push(AbiParam::new(types::F32));
    sig2.params.push(AbiParam::new(types::F32));
    sig2.returns.push(AbiParam::new(types::F32));

    let decl = |module: &mut JITModule, name: &str, sig: &cranelift_codegen::ir::Signature| {
        module
            .declare_function(name, Linkage::Import, sig)
            .map_err(|e| format!("declare {}: {}", name, e))
    };

    Ok(MathFuncs {
        expf: decl(module, "v13_expf", &sig1)?,
        logf: decl(module, "v13_logf", &sig1)?,
        tanhf: decl(module, "v13_tanhf", &sig1)?,
        sqrtf: decl(module, "v13_sqrtf", &sig1)?,
        floorf: decl(module, "v13_floorf", &sig1)?,
        ceilf: decl(module, "v13_ceilf", &sig1)?,
        fabsf: decl(module, "v13_fabsf", &sig1)?,
        powf: decl(module, "v13_powf", &sig2)?,
        fmodf: decl(module, "v13_fmodf", &sig2)?,
        round_bf16: decl(module, "v13_round_bf16", &sig1)?,
        round_f16: decl(module, "v13_round_f16", &sig1)?,
    })
}

// ---- Compiled pipeline ----

/// A compiled NanoGraph ready for execution.
pub struct CompiledPipeline {
    _modules: Vec<JITModule>,
    /// One native function pointer per kernel.
    kernel_ptrs: Vec<*const u8>,
    /// Total number of atoms (size of the values buffer).
    num_atoms: usize,
    /// Literal values to pre-fill before execution.
    /// Maps atom index -> f32 value.
    literals: Vec<(u64, f32)>,
    /// Output atom IDs.
    output_atoms: Vec<AtomId>,
}

unsafe impl Send for CompiledPipeline {}
unsafe impl Sync for CompiledPipeline {}

/// Variable counter for Cranelift Variables within a single function.
/// We need unique Variable indices across all groups in a kernel.
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

impl CompiledPipeline {
    /// Compile a NanoGraph into a JIT pipeline.
    ///
    /// For this naive first pass, we emit a single kernel containing all groups
    /// in topological order (group index order).
    pub fn compile(graph: &NanoGraph) -> Result<Self, String> {
        let num_atoms = graph.num_atoms() as usize;
        let groups = graph.groups();

        // Collect literals for pre-fill.
        let mut literals = Vec::new();
        for group in groups {
            if let ScalarOp::Literal(scalar) = &group.op {
                let val = scalar.to_f64() as f32;
                for i in 0..group.count {
                    literals.push((group.base_id.0 + i, val));
                }
            }
        }

        // Set up Cranelift.
        let mut flag_builder = settings::builder();
        flag_builder.set("opt_level", "speed").unwrap();
        let isa_builder =
            cranelift_native::builder().map_err(|e| format!("cranelift native ISA: {}", e))?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| format!("ISA finish: {}", e))?;

        let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        // Register math function symbols.
        jit_builder.symbol("v13_expf", v13_expf as *const u8);
        jit_builder.symbol("v13_logf", v13_logf as *const u8);
        jit_builder.symbol("v13_tanhf", v13_tanhf as *const u8);
        jit_builder.symbol("v13_sqrtf", v13_sqrtf as *const u8);
        jit_builder.symbol("v13_floorf", v13_floorf as *const u8);
        jit_builder.symbol("v13_ceilf", v13_ceilf as *const u8);
        jit_builder.symbol("v13_fabsf", v13_fabsf as *const u8);
        jit_builder.symbol("v13_powf", v13_powf as *const u8);
        jit_builder.symbol("v13_fmodf", v13_fmodf as *const u8);
        jit_builder.symbol("v13_round_bf16", v13_round_bf16 as *const u8);
        jit_builder.symbol("v13_round_f16", v13_round_f16 as *const u8);

        let mut module = JITModule::new(jit_builder);
        let mut ctx = module.make_context();
        let mut func_ctx = FunctionBuilderContext::new();

        let math_funcs = declare_math_funcs(&mut module)?;

        // Emit a single kernel function for the entire graph.
        // Signature: fn(values_ptr: *mut f32)
        ctx.func.signature.params.push(AbiParam::new(types::I64));

        let func_id = module
            .declare_function("v13_kernel_0", Linkage::Local, &ctx.func.signature)
            .map_err(|e| format!("declare: {}", e))?;

        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);

            let values_ptr = builder.block_params(entry)[0];

            let mut var_counter = VarCounter::new();
            let mut table_counter: usize = 0;

            // Emit each group in order (topological = insertion order).
            for (_gi, group) in groups.iter().enumerate() {
                emit_group(
                    &mut builder,
                    &mut module,
                    group,
                    graph,
                    values_ptr,
                    &math_funcs,
                    &mut var_counter,
                    &mut table_counter,
                )?;
            }

            builder.ins().return_(&[]);
            builder.finalize();
        }

        module
            .define_function(func_id, &mut ctx)
            .map_err(|e| format!("define: {}", e))?;
        ctx.clear();

        module
            .finalize_definitions()
            .map_err(|e| format!("finalize: {}", e))?;

        let func_ptr = module.get_finalized_function(func_id);

        Ok(CompiledPipeline {
            _modules: vec![module],
            kernel_ptrs: vec![func_ptr],
            num_atoms,
            literals,
            output_atoms: graph.outputs.clone(),
        })
    }

    /// Execute the compiled pipeline.
    ///
    /// `overrides` maps atom index -> f32 value for input atoms.
    /// Returns a map of output atom ids -> f32 values.
    pub fn execute(&self, overrides: &HashMap<u64, f32>) -> HashMap<u64, f32> {
        let mut values = vec![0.0f32; self.num_atoms];

        // Pre-fill literals.
        for &(idx, val) in &self.literals {
            values[idx as usize] = val;
        }

        // Apply overrides (input values).
        for (&idx, &val) in overrides {
            values[idx as usize] = val;
        }

        // Execute kernels.
        let values_ptr = values.as_mut_ptr();
        for &func_ptr in &self.kernel_ptrs {
            let func: unsafe extern "C" fn(*mut f32) = unsafe { std::mem::transmute(func_ptr) };
            unsafe { func(values_ptr) };
        }

        // Collect outputs.
        let mut result = HashMap::new();
        for &atom_id in &self.output_atoms {
            result.insert(atom_id.0, values[atom_id.0 as usize]);
        }
        result
    }

    /// Execute and return the entire values buffer (for testing).
    pub fn execute_full(&self, overrides: &HashMap<u64, f32>) -> Vec<f32> {
        let mut values = vec![0.0f32; self.num_atoms];

        // Pre-fill literals.
        for &(idx, val) in &self.literals {
            values[idx as usize] = val;
        }

        // Apply overrides.
        for (&idx, &val) in overrides {
            values[idx as usize] = val;
        }

        // Execute.
        let values_ptr = values.as_mut_ptr();
        for &func_ptr in &self.kernel_ptrs {
            let func: unsafe extern "C" fn(*mut f32) = unsafe { std::mem::transmute(func_ptr) };
            unsafe { func(values_ptr) };
        }

        values
    }

    /// Execute kernels on a pre-filled buffer. The caller is responsible for
    /// allocating the buffer (`vec![0.0f32; pipeline.num_atoms()]`) and filling
    /// in literal/input values before calling. This avoids intermediate HashMaps.
    pub fn execute_on_buffer(&self, values: &mut [f32]) {
        assert!(values.len() >= self.num_atoms);
        let values_ptr = values.as_mut_ptr();
        for &func_ptr in &self.kernel_ptrs {
            let func: unsafe extern "C" fn(*mut f32) = unsafe { std::mem::transmute(func_ptr) };
            unsafe { func(values_ptr) };
        }
    }

    /// Pre-fill literal values into a buffer.
    pub fn fill_literals(&self, values: &mut [f32]) {
        for &(idx, val) in &self.literals {
            values[idx as usize] = val;
        }
    }

    /// Compile a NanoGraph with partitioned kernels.
    ///
    /// Each kernel is a set of group indices compiled into its own native function.
    /// The shared `values` buffer is the communication layer between kernels.
    /// Groups within each kernel are sorted by index (topological order).
    pub fn compile_partitioned(
        graph: &NanoGraph,
        partition: &super::nano_part_creative::NanoPartitionResult,
    ) -> Result<Self, String> {
        let num_atoms = graph.num_atoms() as usize;
        let groups = graph.groups();

        // Collect literals for pre-fill.
        let mut literals = Vec::new();
        for group in groups {
            if let ScalarOp::Literal(scalar) = &group.op {
                let val = scalar.to_f64() as f32;
                for i in 0..group.count {
                    literals.push((group.base_id.0 + i, val));
                }
            }
        }

        let mut modules = Vec::new();
        let mut kernel_ptrs = Vec::new();
        let mut table_counter: usize = 0;

        // Compile each kernel in its own JIT module to avoid relocation overflow.
        for (ki, kernel_group_indices) in partition.kernel_groups.iter().enumerate() {
            let mut sorted_indices = kernel_group_indices.clone();
            sorted_indices.sort();

            let mut flag_builder = settings::builder();
            flag_builder.set("opt_level", "speed").unwrap();
            let isa_builder =
                cranelift_native::builder().map_err(|e| format!("cranelift native ISA: {}", e))?;
            let isa = isa_builder
                .finish(settings::Flags::new(flag_builder))
                .map_err(|e| format!("ISA finish: {}", e))?;

            let mut jit_builder =
                JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

            jit_builder.symbol("v13_expf", v13_expf as *const u8);
            jit_builder.symbol("v13_logf", v13_logf as *const u8);
            jit_builder.symbol("v13_tanhf", v13_tanhf as *const u8);
            jit_builder.symbol("v13_sqrtf", v13_sqrtf as *const u8);
            jit_builder.symbol("v13_floorf", v13_floorf as *const u8);
            jit_builder.symbol("v13_ceilf", v13_ceilf as *const u8);
            jit_builder.symbol("v13_fabsf", v13_fabsf as *const u8);
            jit_builder.symbol("v13_powf", v13_powf as *const u8);
            jit_builder.symbol("v13_fmodf", v13_fmodf as *const u8);
            jit_builder.symbol("v13_round_bf16", v13_round_bf16 as *const u8);
            jit_builder.symbol("v13_round_f16", v13_round_f16 as *const u8);

            let mut module = JITModule::new(jit_builder);
            let mut func_ctx = FunctionBuilderContext::new();
            let math_funcs = declare_math_funcs(&mut module)?;

            let mut ctx = module.make_context();
            ctx.func.signature.params.push(AbiParam::new(types::I64));

            let func_name = format!("v13_kernel_{}", ki);
            let func_id = module
                .declare_function(&func_name, Linkage::Local, &ctx.func.signature)
                .map_err(|e| format!("declare kernel {}: {}", ki, e))?;

            {
                let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
                let entry = builder.create_block();
                builder.append_block_params_for_function_params(entry);
                builder.switch_to_block(entry);
                builder.seal_block(entry);

                let values_ptr = builder.block_params(entry)[0];
                let mut var_counter = VarCounter::new();

                for &gi in &sorted_indices {
                    let group = &groups[gi];
                    emit_group(
                        &mut builder,
                        &mut module,
                        group,
                        graph,
                        values_ptr,
                        &math_funcs,
                        &mut var_counter,
                        &mut table_counter,
                    )?;
                }

                builder.ins().return_(&[]);
                builder.finalize();
            }

            module
                .define_function(func_id, &mut ctx)
                .map_err(|e| format!("define kernel {}: {}", ki, e))?;

            module
                .finalize_definitions()
                .map_err(|e| format!("finalize kernel {}: {}", ki, e))?;

            let func_ptr = module.get_finalized_function(func_id);
            kernel_ptrs.push(func_ptr);
            modules.push(module);
        }

        Ok(CompiledPipeline {
            _modules: modules,
            kernel_ptrs,
            num_atoms,
            literals,
            output_atoms: graph.outputs.clone(),
        })
    }

    /// Number of atoms in the graph.
    pub fn num_atoms(&self) -> usize {
        self.num_atoms
    }
}

// ---- Dtype rounding emission ----

/// Emit Cranelift IR to round an f32 value to the precision of `output_dtype`.
///
/// For F32: identity (no rounding needed).
/// For BF16: call v13_round_bf16 (round-to-nearest-even, 7 mantissa bits).
/// For F16: call v13_round_f16 (via half crate).
/// For integer types: identity (lossy but acceptable for indices).
/// For Bool: identity (caller ensures 0.0/1.0).
///
/// Returns the (possibly rounded) value.
fn emit_output_round(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    math: &MathFuncs,
    val: Value,
    output_dtype: DType,
) -> Value {
    match output_dtype {
        DType::BF16 => {
            let func_ref = module.declare_func_in_func(math.round_bf16, builder.func);
            let call = builder.ins().call(func_ref, &[val]);
            builder.inst_results(call)[0]
        }
        DType::F16 => {
            let func_ref = module.declare_func_in_func(math.round_f16, builder.func);
            let call = builder.ins().call(func_ref, &[val]);
            builder.inst_results(call)[0]
        }
        // F32 and integer types: no rounding needed (values are already f32).
        _ => val,
    }
}

// ---- Group emission ----

/// Emit Cranelift IR for a single AtomGroup.
///
/// For groups with count > 0, emits a loop over i in 0..count.
/// For each iteration, resolves inputs, computes the op, and stores the result.
fn emit_group(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    group: &AtomGroup,
    graph: &NanoGraph,
    values_ptr: Value,
    math: &MathFuncs,
    var_counter: &mut VarCounter,
    table_counter: &mut usize,
) -> Result<(), String> {
    // Skip Literal groups -- their values are pre-filled by the caller.
    if matches!(&group.op, ScalarOp::Literal(_)) {
        return Ok(());
    }

    if group.count == 0 {
        return Ok(());
    }

    // For count == 1, emit inline (no loop).
    if group.count == 1 {
        emit_group_body(
            builder,
            module,
            group,
            graph,
            values_ptr,
            None, // i = 0 (constant)
            0,
            math,
            var_counter,
            table_counter,
        )?;
        return Ok(());
    }

    // Emit loop: for i in 0..count
    let loop_header = builder.create_block();
    let loop_body = builder.create_block();
    let loop_exit = builder.create_block();

    let zero = builder.ins().iconst(types::I64, 0);
    let count = builder.ins().iconst(types::I64, group.count as i64);

    builder.ins().jump(loop_header, &[zero]);

    // Loop header: i = phi(0, i+1)
    builder.switch_to_block(loop_header);
    builder.append_block_param(loop_header, types::I64);
    let i_val = builder.block_params(loop_header)[0];

    let cmp = builder.ins().icmp(
        cranelift_codegen::ir::condcodes::IntCC::SignedLessThan,
        i_val,
        count,
    );
    builder.ins().brif(cmp, loop_body, &[], loop_exit, &[]);

    builder.switch_to_block(loop_body);

    emit_group_body(
        builder,
        module,
        group,
        graph,
        values_ptr,
        Some(i_val),
        0,
        math,
        var_counter,
        table_counter,
    )?;

    let i_next = builder.ins().iadd_imm(i_val, 1);
    builder.ins().jump(loop_header, &[i_next]);

    builder.switch_to_block(loop_exit);
    builder.seal_block(loop_header);
    builder.seal_block(loop_body);
    builder.seal_block(loop_exit);

    Ok(())
}

/// Emit the body of a group iteration (for one value of i).
///
/// `i_val` is None when the group has count == 1 (i is implicitly 0).
fn emit_group_body(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    group: &AtomGroup,
    graph: &NanoGraph,
    values_ptr: Value,
    i_val: Option<Value>,
    i_const: u64,
    math: &MathFuncs,
    var_counter: &mut VarCounter,
    table_counter: &mut usize,
) -> Result<(), String> {
    let base_id = group.base_id.0;

    match &group.op {
        ScalarOp::Literal(_) => {
            // Already handled by pre-fill.
            Ok(())
        }

        ScalarOp::Identity {
            compute_dtype,
            output_dtype,
        } => {
            // Load source, apply output rounding, store.
            let src_val = load_input_ref(
                builder,
                module,
                &group.inputs[0],
                values_ptr,
                i_val,
                i_const,
                table_counter,
            )?;
            let rounded = emit_output_round(builder, module, math, src_val, *output_dtype);
            store_atom(builder, values_ptr, base_id, i_val, i_const, rounded);
            Ok(())
        }

        ScalarOp::Binary {
            op,
            compute_dtype,
            output_dtype,
        } => {
            let a = load_input_ref(
                builder,
                module,
                &group.inputs[0],
                values_ptr,
                i_val,
                i_const,
                table_counter,
            )?;
            let b = load_input_ref(
                builder,
                module,
                &group.inputs[1],
                values_ptr,
                i_val,
                i_const,
                table_counter,
            )?;
            let result = emit_binop(builder, module, math, *op, a, b)?;
            let rounded = emit_output_round(builder, module, math, result, *output_dtype);
            store_atom(builder, values_ptr, base_id, i_val, i_const, rounded);
            Ok(())
        }

        ScalarOp::Unary {
            op,
            compute_dtype,
            output_dtype,
        } => {
            let x = load_input_ref(
                builder,
                module,
                &group.inputs[0],
                values_ptr,
                i_val,
                i_const,
                table_counter,
            )?;
            let result = emit_unop(builder, module, math, *op, x)?;
            let rounded = emit_output_round(builder, module, math, result, *output_dtype);
            store_atom(builder, values_ptr, base_id, i_val, i_const, rounded);
            Ok(())
        }

        ScalarOp::Select {
            compute_dtype,
            output_dtype,
        } => {
            let cond = load_input_ref(
                builder,
                module,
                &group.inputs[0],
                values_ptr,
                i_val,
                i_const,
                table_counter,
            )?;
            let x = load_input_ref(
                builder,
                module,
                &group.inputs[1],
                values_ptr,
                i_val,
                i_const,
                table_counter,
            )?;
            let y = load_input_ref(
                builder,
                module,
                &group.inputs[2],
                values_ptr,
                i_val,
                i_const,
                table_counter,
            )?;
            let zero = builder.ins().f32const(0.0);
            let is_nonzero = builder.ins().fcmp(
                cranelift_codegen::ir::condcodes::FloatCC::NotEqual,
                cond,
                zero,
            );
            let result = builder.ins().select(is_nonzero, x, y);
            let rounded = emit_output_round(builder, module, math, result, *output_dtype);
            store_atom(builder, values_ptr, base_id, i_val, i_const, rounded);
            Ok(())
        }

        ScalarOp::ReduceSum { output_dtype, .. } => emit_reduce(
            builder,
            module,
            group,
            graph,
            values_ptr,
            i_val,
            i_const,
            math,
            var_counter,
            table_counter,
            true,
            *output_dtype,
        ),

        ScalarOp::ReduceMax { output_dtype, .. } => emit_reduce(
            builder,
            module,
            group,
            graph,
            values_ptr,
            i_val,
            i_const,
            math,
            var_counter,
            table_counter,
            false,
            *output_dtype,
        ),

        ScalarOp::IndirectLoad {
            table_base,
            output_dtype,
        } => {
            let idx_f32 = load_input_ref(
                builder,
                module,
                &group.inputs[0],
                values_ptr,
                i_val,
                i_const,
                table_counter,
            )?;
            let idx_i64 = builder.ins().fcvt_to_sint(types::I64, idx_f32);
            let table_base_val = builder.ins().iconst(types::I64, table_base.0 as i64);
            let atom_idx = builder.ins().iadd(table_base_val, idx_i64);
            let byte_offset = builder.ins().ishl_imm(atom_idx, 2);
            let addr = builder.ins().iadd(values_ptr, byte_offset);
            let result = builder.ins().load(types::F32, MemFlags::trusted(), addr, 0);
            let rounded = emit_output_round(builder, module, math, result, *output_dtype);
            store_atom(builder, values_ptr, base_id, i_val, i_const, rounded);
            Ok(())
        }
    }
}

// ---- Reduction emission ----

fn emit_reduce(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    group: &AtomGroup,
    graph: &NanoGraph,
    values_ptr: Value,
    i_val: Option<Value>,
    i_const: u64,
    math: &MathFuncs,
    var_counter: &mut VarCounter,
    table_counter: &mut usize,
    is_sum: bool,
    output_dtype: DType,
) -> Result<(), String> {
    // Extract reduce_count and reduce_stride from the op.
    let (reduce_count, reduce_stride) = match &group.op {
        ScalarOp::ReduceSum {
            reduce_count,
            reduce_stride,
            ..
        } => (*reduce_count, *reduce_stride),
        ScalarOp::ReduceMax {
            reduce_count,
            reduce_stride,
            ..
        } => (*reduce_count, *reduce_stride),
        _ => return Err("emit_reduce called on non-reduce op".to_string()),
    };

    let base_id = group.base_id.0;

    // Accumulator variable.
    let acc_var = var_counter.next();
    builder.declare_var(acc_var, types::F32);
    let init_val = if is_sum {
        builder.ins().f32const(0.0)
    } else {
        builder.ins().f32const(f32::NEG_INFINITY)
    };
    builder.def_var(acc_var, init_val);

    // Compute the base index for this atom's input (at k=0).
    let base_idx = match &group.inputs[0] {
        InputRef::Affine {
            base: input_base,
            stride,
        } => match i_val {
            Some(iv) => {
                let si = builder.ins().imul_imm(iv, *stride as i64);
                builder.ins().iadd_imm(si, input_base.0 as i64)
            }
            None => builder.ins().iconst(
                types::I64,
                input_base.0 as i64 + (*stride as i64) * (i_const as i64),
            ),
        },
        _ => {
            return Err(format!(
                "Reduce input must be Affine, got {:?}",
                group.inputs[0]
            ));
        }
    };

    // K loop variable.
    let k_var = var_counter.next();
    builder.declare_var(k_var, types::I64);
    let k_init = builder.ins().iconst(types::I64, 0);
    builder.def_var(k_var, k_init);

    let red_header = builder.create_block();
    let red_body = builder.create_block();
    let red_exit = builder.create_block();

    let bound_val = builder.ins().iconst(types::I64, reduce_count as i64);

    builder.ins().jump(red_header, &[]);

    // Reduction loop header.
    builder.switch_to_block(red_header);
    let k_val = builder.use_var(k_var);
    let k_cmp = builder.ins().icmp(
        cranelift_codegen::ir::condcodes::IntCC::SignedLessThan,
        k_val,
        bound_val,
    );
    builder.ins().brif(k_cmp, red_body, &[], red_exit, &[]);

    // Reduction loop body: load source at base_idx + k * reduce_stride.
    builder.switch_to_block(red_body);
    let k_val = builder.use_var(k_var);

    // src_idx = base_idx + k * reduce_stride
    let k_offset = builder.ins().imul_imm(k_val, reduce_stride);
    let src_idx = builder.ins().iadd(base_idx, k_offset);
    let byte_offset = builder.ins().imul_imm(src_idx, 4);
    let addr = builder.ins().iadd(values_ptr, byte_offset);
    let src_val = builder.ins().load(types::F32, MemFlags::new(), addr, 0);

    let acc = builder.use_var(acc_var);
    let new_acc = if is_sum {
        builder.ins().fadd(acc, src_val)
    } else {
        // max(acc, src)
        let cmp = builder.ins().fcmp(
            cranelift_codegen::ir::condcodes::FloatCC::GreaterThan,
            src_val,
            acc,
        );
        builder.ins().select(cmp, src_val, acc)
    };
    builder.def_var(acc_var, new_acc);

    let k_next = builder.ins().iadd_imm(k_val, 1);
    builder.def_var(k_var, k_next);

    builder.ins().jump(red_header, &[]);

    // Exit: store the accumulated result with output dtype rounding.
    builder.switch_to_block(red_exit);
    builder.seal_block(red_header);
    builder.seal_block(red_body);
    builder.seal_block(red_exit);

    let final_acc = builder.use_var(acc_var);
    let rounded = emit_output_round(builder, module, math, final_acc, output_dtype);
    store_atom(builder, values_ptr, base_id, i_val, i_const, rounded);

    Ok(())
}

// ---- Input ref resolution ----

/// Load a value from the values array according to an InputRef, for atom offset i.
/// k is not used (set to 0 for non-SymAffine).
fn load_input_ref(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    input: &InputRef,
    values_ptr: Value,
    i_val: Option<Value>,
    i_const: u64,
    table_counter: &mut usize,
) -> Result<Value, String> {
    match input {
        InputRef::Broadcast(atom_id) => {
            // Fixed offset, same for all i.
            let byte_offset = (atom_id.0 as i64) * 4;
            if byte_offset <= i32::MAX as i64 {
                Ok(builder
                    .ins()
                    .load(types::F32, MemFlags::new(), values_ptr, byte_offset as i32))
            } else {
                let offset = builder.ins().iconst(types::I64, byte_offset);
                let addr = builder.ins().iadd(values_ptr, offset);
                Ok(builder.ins().load(types::F32, MemFlags::new(), addr, 0))
            }
        }

        InputRef::Affine { base, stride } => {
            // addr = values_ptr + (base.0 + stride * i) * 4
            let atom_idx = match i_val {
                Some(iv) => {
                    if *stride == 1 {
                        builder.ins().iadd_imm(iv, base.0 as i64)
                    } else if *stride == -1 {
                        let neg_i = builder.ins().ineg(iv);
                        builder.ins().iadd_imm(neg_i, base.0 as i64)
                    } else {
                        let si = builder.ins().imul_imm(iv, *stride as i64);
                        builder.ins().iadd_imm(si, base.0 as i64)
                    }
                }
                None => {
                    // i = i_const
                    let idx = (base.0 as i64) + (*stride as i64) * (i_const as i64);
                    builder.ins().iconst(types::I64, idx)
                }
            };
            let byte_offset = builder.ins().imul_imm(atom_idx, 4);
            let addr = builder.ins().iadd(values_ptr, byte_offset);
            Ok(builder.ins().load(types::F32, MemFlags::new(), addr, 0))
        }

        InputRef::Explicit(ids) => {
            if ids.len() == 1 {
                // Only one element -- treat like broadcast.
                let byte_offset = (ids[0].0 as i64) * 4;
                if byte_offset <= i32::MAX as i64 {
                    return Ok(builder.ins().load(
                        types::F32,
                        MemFlags::new(),
                        values_ptr,
                        byte_offset as i32,
                    ));
                }
                let offset = builder.ins().iconst(types::I64, byte_offset);
                let addr = builder.ins().iadd(values_ptr, offset);
                return Ok(builder.ins().load(types::F32, MemFlags::new(), addr, 0));
            }

            // Embed the atom ID array as a data section.
            let data_name = format!("v13_explicit_{}", *table_counter);
            *table_counter += 1;
            let data_id = module
                .declare_data(&data_name, Linkage::Local, false, false)
                .map_err(|e| format!("declare explicit table: {}", e))?;
            let mut data_desc = cranelift_module::DataDescription::new();
            let bytes: Vec<u8> = ids.iter().flat_map(|id| id.0.to_le_bytes()).collect();
            data_desc.define(bytes.into_boxed_slice());
            module
                .define_data(data_id, &data_desc)
                .map_err(|e| format!("define explicit table: {}", e))?;

            let gv = module.declare_data_in_func(data_id, builder.func);
            let table_ptr = builder.ins().global_value(types::I64, gv);

            // Load atom_id = table[i] (table entries are u64 = 8 bytes each)
            let i = i_val.unwrap_or_else(|| builder.ins().iconst(types::I64, i_const as i64));
            let idx_byte_off = builder.ins().imul_imm(i, 8);
            let idx_addr = builder.ins().iadd(table_ptr, idx_byte_off);
            let atom_id_i64 = builder.ins().load(types::I64, MemFlags::new(), idx_addr, 0);

            // Load values[atom_id]
            let data_byte_off = builder.ins().imul_imm(atom_id_i64, 4);
            let data_addr = builder.ins().iadd(values_ptr, data_byte_off);
            Ok(builder
                .ins()
                .load(types::F32, MemFlags::new(), data_addr, 0))
        }

        InputRef::SymAffine {
            base,
            stride_i,
            stride_k,
        } => {
            // For non-reduce context, k=0 so this reduces to base + stride_i * i.
            let atom_idx = match i_val {
                Some(iv) => {
                    let si = builder.ins().imul_imm(iv, *stride_i as i64);
                    builder.ins().iadd_imm(si, base.0 as i64)
                }
                None => {
                    let idx = (base.0 as i64) + (*stride_i as i64) * (i_const as i64);
                    builder.ins().iconst(types::I64, idx)
                }
            };
            let byte_offset = builder.ins().imul_imm(atom_idx, 4);
            let addr = builder.ins().iadd(values_ptr, byte_offset);
            Ok(builder.ins().load(types::F32, MemFlags::new(), addr, 0))
        }

        InputRef::StridedBroadcast {
            base,
            stride,
            repeat,
        } => {
            // atom_idx = base + stride * (i / repeat)
            let atom_idx = match i_val {
                Some(iv) => {
                    // block_idx = i / repeat
                    let block_idx = if repeat.is_power_of_two() {
                        let shift = repeat.trailing_zeros() as i64;
                        builder.ins().ushr_imm(iv, shift)
                    } else {
                        let rep = builder.ins().iconst(types::I64, *repeat as i64);
                        builder.ins().udiv(iv, rep)
                    };
                    // atom_idx = base + stride * block_idx
                    let offset = builder.ins().imul_imm(block_idx, *stride);
                    builder.ins().iadd_imm(offset, base.0 as i64)
                }
                None => {
                    let block = i_const / repeat;
                    let idx = (base.0 as i64) + (*stride * block as i64);
                    builder.ins().iconst(types::I64, idx)
                }
            };
            let byte_offset = builder.ins().imul_imm(atom_idx, 4);
            let addr = builder.ins().iadd(values_ptr, byte_offset);
            Ok(builder.ins().load(types::F32, MemFlags::new(), addr, 0))
        }

        InputRef::Modular {
            base,
            stride,
            modulus,
        } => {
            // atom_idx = base + stride * (i % modulus)
            let atom_idx = match i_val {
                Some(iv) => {
                    let modval = builder.ins().iconst(types::I64, *modulus as i64);
                    let wrapped = builder.ins().urem(iv, modval);
                    let offset = builder.ins().imul_imm(wrapped, *stride as i64);
                    builder.ins().iadd_imm(offset, base.0 as i64)
                }
                None => {
                    let wrapped = i_const % modulus;
                    let idx = (base.0 as i64) + (*stride as i64) * (wrapped as i64);
                    builder.ins().iconst(types::I64, idx)
                }
            };
            let byte_offset = builder.ins().imul_imm(atom_idx, 4);
            let addr = builder.ins().iadd(values_ptr, byte_offset);
            Ok(builder.ins().load(types::F32, MemFlags::new(), addr, 0))
        }
    }
}

/// Load a value from the values array according to an InputRef, for atom offset i
/// and reduction iteration k.
fn load_input_ref_with_k(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    input: &InputRef,
    values_ptr: Value,
    i_val: Option<Value>,
    i_const: u64,
    k_val: Value,
    table_counter: &mut usize,
) -> Result<Value, String> {
    match input {
        InputRef::SymAffine {
            base,
            stride_i,
            stride_k,
        } => {
            // atom_idx = base + stride_i * i + stride_k * k
            let si = match i_val {
                Some(iv) => builder.ins().imul_imm(iv, *stride_i as i64),
                None => builder
                    .ins()
                    .iconst(types::I64, (*stride_i as i64) * (i_const as i64)),
            };
            let sk = builder.ins().imul_imm(k_val, *stride_k as i64);
            let idx = builder.ins().iadd(si, sk);
            let idx = builder.ins().iadd_imm(idx, base.0 as i64);
            let byte_offset = builder.ins().imul_imm(idx, 4);
            let addr = builder.ins().iadd(values_ptr, byte_offset);
            Ok(builder.ins().load(types::F32, MemFlags::new(), addr, 0))
        }

        // For non-SymAffine inputs in a reduce context, k doesn't affect the address.
        InputRef::Broadcast(_)
        | InputRef::Affine { .. }
        | InputRef::Explicit(_)
        | InputRef::StridedBroadcast { .. }
        | InputRef::Modular { .. } => load_input_ref(
            builder,
            module,
            input,
            values_ptr,
            i_val,
            i_const,
            table_counter,
        ),
    }
}

// ---- Atom store helper ----

/// Store a value into values[base_id + i].
fn store_atom(
    builder: &mut FunctionBuilder,
    values_ptr: Value,
    base_id: u64,
    i_val: Option<Value>,
    i_const: u64,
    val: Value,
) {
    let atom_idx = match i_val {
        Some(iv) => builder.ins().iadd_imm(iv, base_id as i64),
        None => builder.ins().iconst(types::I64, (base_id + i_const) as i64),
    };
    let byte_offset = builder.ins().imul_imm(atom_idx, 4);
    let addr = builder.ins().iadd(values_ptr, byte_offset);
    builder.ins().store(MemFlags::new(), val, addr, 0);
}

// ---- Scalar op emission ----

fn emit_binop(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    math: &MathFuncs,
    op: ScalarBinOp,
    a: Value,
    b: Value,
) -> Result<Value, String> {
    Ok(match op {
        ScalarBinOp::Add => builder.ins().fadd(a, b),
        ScalarBinOp::Sub => builder.ins().fsub(a, b),
        ScalarBinOp::Mul => builder.ins().fmul(a, b),
        ScalarBinOp::Div => builder.ins().fdiv(a, b),
        ScalarBinOp::Max => {
            let cmp =
                builder
                    .ins()
                    .fcmp(cranelift_codegen::ir::condcodes::FloatCC::GreaterThan, a, b);
            builder.ins().select(cmp, a, b)
        }
        ScalarBinOp::Min => {
            let cmp = builder
                .ins()
                .fcmp(cranelift_codegen::ir::condcodes::FloatCC::LessThan, a, b);
            builder.ins().select(cmp, a, b)
        }
        ScalarBinOp::Pow => {
            let func_ref = module.declare_func_in_func(math.powf, builder.func);
            let call = builder.ins().call(func_ref, &[a, b]);
            builder.inst_results(call)[0]
        }
        ScalarBinOp::Mod => {
            let func_ref = module.declare_func_in_func(math.fmodf, builder.func);
            let call = builder.ins().call(func_ref, &[a, b]);
            builder.inst_results(call)[0]
        }
        ScalarBinOp::Equal => {
            let cmp = builder
                .ins()
                .fcmp(cranelift_codegen::ir::condcodes::FloatCC::Equal, a, b);
            let one = builder.ins().f32const(1.0);
            let zero = builder.ins().f32const(0.0);
            builder.ins().select(cmp, one, zero)
        }
        ScalarBinOp::Greater => {
            let cmp =
                builder
                    .ins()
                    .fcmp(cranelift_codegen::ir::condcodes::FloatCC::GreaterThan, a, b);
            let one = builder.ins().f32const(1.0);
            let zero = builder.ins().f32const(0.0);
            builder.ins().select(cmp, one, zero)
        }
        ScalarBinOp::GreaterOrEqual => {
            let cmp = builder.ins().fcmp(
                cranelift_codegen::ir::condcodes::FloatCC::GreaterThanOrEqual,
                a,
                b,
            );
            let one = builder.ins().f32const(1.0);
            let zero = builder.ins().f32const(0.0);
            builder.ins().select(cmp, one, zero)
        }
        ScalarBinOp::Less => {
            let cmp = builder
                .ins()
                .fcmp(cranelift_codegen::ir::condcodes::FloatCC::LessThan, a, b);
            let one = builder.ins().f32const(1.0);
            let zero = builder.ins().f32const(0.0);
            builder.ins().select(cmp, one, zero)
        }
        ScalarBinOp::LessOrEqual => {
            let cmp = builder.ins().fcmp(
                cranelift_codegen::ir::condcodes::FloatCC::LessThanOrEqual,
                a,
                b,
            );
            let one = builder.ins().f32const(1.0);
            let zero = builder.ins().f32const(0.0);
            builder.ins().select(cmp, one, zero)
        }
        ScalarBinOp::And => {
            let zero = builder.ins().f32const(0.0);
            let a_nz =
                builder
                    .ins()
                    .fcmp(cranelift_codegen::ir::condcodes::FloatCC::NotEqual, a, zero);
            let b_nz =
                builder
                    .ins()
                    .fcmp(cranelift_codegen::ir::condcodes::FloatCC::NotEqual, b, zero);
            let both = builder.ins().band(a_nz, b_nz);
            let one = builder.ins().f32const(1.0);
            builder.ins().select(both, one, zero)
        }
        ScalarBinOp::Or => {
            let zero = builder.ins().f32const(0.0);
            let a_nz =
                builder
                    .ins()
                    .fcmp(cranelift_codegen::ir::condcodes::FloatCC::NotEqual, a, zero);
            let b_nz =
                builder
                    .ins()
                    .fcmp(cranelift_codegen::ir::condcodes::FloatCC::NotEqual, b, zero);
            let either = builder.ins().bor(a_nz, b_nz);
            let one = builder.ins().f32const(1.0);
            builder.ins().select(either, one, zero)
        }
        ScalarBinOp::Xor => {
            let zero = builder.ins().f32const(0.0);
            let a_nz =
                builder
                    .ins()
                    .fcmp(cranelift_codegen::ir::condcodes::FloatCC::NotEqual, a, zero);
            let b_nz =
                builder
                    .ins()
                    .fcmp(cranelift_codegen::ir::condcodes::FloatCC::NotEqual, b, zero);
            let x = builder.ins().bxor(a_nz, b_nz);
            let one = builder.ins().f32const(1.0);
            builder.ins().select(x, one, zero)
        }
    })
}

fn emit_unop(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    math: &MathFuncs,
    op: ScalarUnaryOp,
    x: Value,
) -> Result<Value, String> {
    Ok(match op {
        ScalarUnaryOp::Neg => builder.ins().fneg(x),
        ScalarUnaryOp::Abs => builder.ins().fabs(x),
        ScalarUnaryOp::Exp => {
            let func_ref = module.declare_func_in_func(math.expf, builder.func);
            let call = builder.ins().call(func_ref, &[x]);
            builder.inst_results(call)[0]
        }
        ScalarUnaryOp::Ln => {
            let func_ref = module.declare_func_in_func(math.logf, builder.func);
            let call = builder.ins().call(func_ref, &[x]);
            builder.inst_results(call)[0]
        }
        ScalarUnaryOp::Sqrt => builder.ins().sqrt(x),
        ScalarUnaryOp::Reciprocal => {
            let one = builder.ins().f32const(1.0);
            builder.ins().fdiv(one, x)
        }
        ScalarUnaryOp::Tanh => {
            let func_ref = module.declare_func_in_func(math.tanhf, builder.func);
            let call = builder.ins().call(func_ref, &[x]);
            builder.inst_results(call)[0]
        }
        ScalarUnaryOp::Floor => builder.ins().floor(x),
        ScalarUnaryOp::Ceil => builder.ins().ceil(x),
    })
}

// ---- Tests ----

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::nano_graph::eval::NanoEval;
    use crate::nano_graph::{AtomId, InputRef, NanoGraph, ScalarBinOp, ScalarOp, ScalarUnaryOp};
    use crate::numeric_scalar::NumericScalar;
    use std::collections::HashMap;

    /// Helper: compare JIT output to interpreter output.
    fn compare_jit_vs_interp(graph: &NanoGraph, overrides_f32: &HashMap<u64, f32>, label: &str) {
        // Build NumericScalar overrides for interpreter.
        let overrides_ns: HashMap<u64, NumericScalar> = overrides_f32
            .iter()
            .map(|(&k, &v)| (k, NumericScalar::F32(v)))
            .collect();

        // Interpreter.
        let t0 = std::time::Instant::now();
        let interp = NanoEval::eval(graph, &overrides_ns);
        let interp_time = t0.elapsed();

        // JIT compile.
        let t0 = std::time::Instant::now();
        let pipeline = CompiledPipeline::compile(graph).expect("compile failed");
        let compile_time = t0.elapsed();

        // JIT execute.
        let t0 = std::time::Instant::now();
        let jit_values = pipeline.execute_full(overrides_f32);
        let jit_time = t0.elapsed();

        eprintln!(
            "[{}] interp: {:?}, compile: {:?}, jit_exec: {:?}, atoms: {}",
            label,
            interp_time,
            compile_time,
            jit_time,
            graph.num_atoms()
        );

        // Compare all atoms.
        let mut max_abs_err: f64 = 0.0;
        let mut max_err_atom: u64 = 0;
        for i in 0..graph.num_atoms() {
            let interp_val = interp.get(AtomId(i));
            let jit_val = jit_values[i as usize] as f64;
            let diff = (interp_val - jit_val).abs();
            if diff > max_abs_err {
                max_abs_err = diff;
                max_err_atom = i;
            }
            // Allow tolerance relative to magnitude.
            let tol = 1e-4 * interp_val.abs().max(1.0);
            assert!(
                diff < tol || (!interp_val.is_finite() && !jit_val.is_finite()),
                "[{}] atom {}: interp={} jit={} diff={} (tol={})",
                label,
                i,
                interp_val,
                jit_val,
                diff,
                tol,
            );
        }
        eprintln!(
            "[{}] max_abs_error={:.2e} at atom {}",
            label, max_abs_err, max_err_atom
        );
    }

    #[test]
    fn test_elementwise_add() {
        let mut g = NanoGraph::new();

        // a: 4 literals
        let a = g.push_group(
            4,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );
        // b: 4 literals
        let b = g.push_group(
            4,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );
        // c = a + b
        let c = g.push_group(
            4,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine { base: a, stride: 1 },
                InputRef::Affine { base: b, stride: 1 },
            ],
        );

        // Mark outputs.
        for i in 0..4 {
            g.outputs.push(c.offset(i));
        }

        let mut overrides = HashMap::new();
        for i in 0..4u64 {
            overrides.insert(a.0 + i, (i as f32 + 1.0) * 1.0); // a = [1,2,3,4]
            overrides.insert(b.0 + i, (i as f32 + 1.0) * 10.0); // b = [10,20,30,40]
        }

        compare_jit_vs_interp(&g, &overrides, "elementwise_add");
    }

    #[test]
    fn test_unary_chain() {
        let mut g = NanoGraph::new();

        let a = g.push_group(
            4,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );
        // neg(a)
        let b = g.push_group(
            4,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: a, stride: 1 }],
        );
        // exp(neg(a))
        let c = g.push_group(
            4,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Exp,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: b, stride: 1 }],
        );

        for i in 0..4 {
            g.outputs.push(c.offset(i));
        }

        let mut overrides = HashMap::new();
        for i in 0..4u64 {
            overrides.insert(a.0 + i, i as f32 * 0.5); // [0, 0.5, 1.0, 1.5]
        }

        compare_jit_vs_interp(&g, &overrides, "unary_chain_neg_exp");
    }

    #[test]
    fn test_broadcast_add() {
        let mut g = NanoGraph::new();

        // a: 4 elements
        let a = g.push_group(
            4,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );
        // b: 1 scalar
        let b = g.push_atom(
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );
        // c = a + broadcast(b)
        let c = g.push_group(
            4,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine { base: a, stride: 1 },
                InputRef::Broadcast(b),
            ],
        );

        for i in 0..4 {
            g.outputs.push(c.offset(i));
        }

        let mut overrides = HashMap::new();
        for i in 0..4u64 {
            overrides.insert(a.0 + i, i as f32 + 1.0);
        }
        overrides.insert(b.0, 100.0);

        compare_jit_vs_interp(&g, &overrides, "broadcast_add");
    }

    #[test]
    fn test_matmul_4x8x16() {
        let m = 4u64;
        let k_dim = 8u64;
        let n = 16u64;

        let mut g = NanoGraph::new();

        let a = g.push_group(
            m * k_dim,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );
        let b_base = g.push_group(
            k_dim * n,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );

        let total_prods = m * n * k_dim;
        let mut a_refs = Vec::with_capacity(total_prods as usize);
        let mut b_refs = Vec::with_capacity(total_prods as usize);
        for row in 0..m {
            for col in 0..n {
                for k in 0..k_dim {
                    a_refs.push(AtomId(a.0 + row * k_dim + k));
                    b_refs.push(AtomId(b_base.0 + k * n + col));
                }
            }
        }

        let prods = g.push_group(
            total_prods,
            ScalarOp::Binary {
                op: ScalarBinOp::Mul,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Explicit(a_refs), InputRef::Explicit(b_refs)],
        );

        let c = g.push_group(
            m * n,
            ScalarOp::ReduceSum {
                reduce_count: k_dim,
                reduce_stride: 1,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine {
                base: prods,
                stride: k_dim as i32,
            }],
        );

        for i in 0..(m * n) {
            g.outputs.push(c.offset(i));
        }

        let mut overrides = HashMap::new();
        for i in 0..(m * k_dim) {
            overrides.insert(a.0 + i, (i as f32) + 1.0);
        }
        for i in 0..(k_dim * n) {
            let row = i / n;
            let col = i % n;
            overrides.insert(b_base.0 + i, if row == col % k_dim { 1.0 } else { 0.0 });
        }

        assert!(
            g.validate().is_empty(),
            "validation errors: {:?}",
            g.validate()
        );

        compare_jit_vs_interp(&g, &overrides, "matmul_4x8x16");
    }

    #[test]
    fn test_identity_passthrough() {
        let mut g = NanoGraph::new();

        let a = g.push_group(
            4,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );
        let b = g.push_group(
            4,
            ScalarOp::Identity {
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: a, stride: 1 }],
        );

        for i in 0..4 {
            g.outputs.push(b.offset(i));
        }

        let mut overrides = HashMap::new();
        for i in 0..4u64 {
            overrides.insert(a.0 + i, (i as f32 + 1.0) * 3.14);
        }

        compare_jit_vs_interp(&g, &overrides, "identity_passthrough");
    }

    #[test]
    fn test_select() {
        let mut g = NanoGraph::new();

        let cond = g.push_group(
            4,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );
        let x = g.push_group(
            4,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );
        let y = g.push_group(
            4,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );
        let result = g.push_group(
            4,
            ScalarOp::Select {
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine {
                    base: cond,
                    stride: 1,
                },
                InputRef::Affine { base: x, stride: 1 },
                InputRef::Affine { base: y, stride: 1 },
            ],
        );

        for i in 0..4 {
            g.outputs.push(result.offset(i));
        }

        let mut overrides = HashMap::new();
        overrides.insert(cond.0, 1.0f32);
        overrides.insert(cond.0 + 1, 0.0);
        overrides.insert(cond.0 + 2, 5.0);
        overrides.insert(cond.0 + 3, 0.0);
        for i in 0..4u64 {
            overrides.insert(x.0 + i, (i as f32 + 1.0) * 10.0);
        }
        for i in 0..4u64 {
            overrides.insert(y.0 + i, (i as f32 + 1.0) * 100.0);
        }

        compare_jit_vs_interp(&g, &overrides, "select");
    }

    #[test]
    fn test_reduce_max() {
        let mut g = NanoGraph::new();

        let a = g.push_group(
            8,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );

        let c = g.push_group(
            1,
            ScalarOp::ReduceMax {
                reduce_count: 8,
                reduce_stride: 1,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: a, stride: 0 }],
        );

        g.outputs.push(c);

        let mut overrides = HashMap::new();
        let test_vals = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        for (i, &v) in test_vals.iter().enumerate() {
            overrides.insert(a.0 + i as u64, v);
        }

        compare_jit_vs_interp(&g, &overrides, "reduce_max");
    }

    #[test]
    fn test_explicit_input_ref() {
        let mut g = NanoGraph::new();

        let a = g.push_group(
            4,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );

        let b = g.push_group(
            3,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Explicit(vec![
                a.offset(2),
                a.offset(0),
                a.offset(3),
            ])],
        );

        for i in 0..3 {
            g.outputs.push(b.offset(i));
        }

        let mut overrides = HashMap::new();
        overrides.insert(a.0, 10.0f32);
        overrides.insert(a.0 + 1, 20.0);
        overrides.insert(a.0 + 2, 30.0);
        overrides.insert(a.0 + 3, 40.0);

        compare_jit_vs_interp(&g, &overrides, "explicit_gather_neg");
    }

    // ---- BF16 rounding tests ----

    #[test]
    fn test_bf16_rounding_add() {
        // Test that output_dtype: BF16 applies BF16 rounding.
        // BF16 has 7 mantissa bits = precision of ~3 decimal digits.
        // 1.0 + 1e-4 = 1.0001 in f32, which BF16 rounds to 1.0 (or 1.0078125).
        let mut g = NanoGraph::new();

        let a = g.push_group(
            4,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );
        let b = g.push_group(
            4,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );
        // Add with BF16 output: should round the result to BF16 precision.
        let c = g.push_group(
            4,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::BF16,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine { base: a, stride: 1 },
                InputRef::Affine { base: b, stride: 1 },
            ],
        );

        for i in 0..4 {
            g.outputs.push(c.offset(i));
        }

        // Values chosen so BF16 rounding is visible:
        // a = [1.0, 256.0, 0.1, 1000.0]
        // b = [1e-4, 0.001, 0.0001, 0.5]
        let a_vals = [1.0f32, 256.0, 0.1, 1000.0];
        let b_vals = [1e-4f32, 0.001, 0.0001, 0.5];
        let mut overrides = HashMap::new();
        for i in 0..4u64 {
            overrides.insert(a.0 + i, a_vals[i as usize]);
            overrides.insert(b.0 + i, b_vals[i as usize]);
        }

        compare_jit_vs_interp(&g, &overrides, "bf16_rounding_add");
    }

    #[test]
    fn test_bf16_rounding_chain() {
        // Test BF16 rounding through a chain: add(BF16) -> neg(F32).
        // The neg op loads a BF16-rounded value and computes at F32.
        let mut g = NanoGraph::new();

        let a = g.push_group(
            4,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );
        let b = g.push_group(
            4,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );
        // Step 1: add with BF16 output.
        let c = g.push_group(
            4,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::BF16,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine { base: a, stride: 1 },
                InputRef::Affine { base: b, stride: 1 },
            ],
        );
        // Step 2: neg with F32 compute and output (reads the BF16-rounded value).
        let d = g.push_group(
            4,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: c, stride: 1 }],
        );

        for i in 0..4 {
            g.outputs.push(d.offset(i));
        }

        let a_vals = [1.0f32, 256.0, 0.1, 1000.0];
        let b_vals = [1e-4f32, 0.001, 0.0001, 0.5];
        let mut overrides = HashMap::new();
        for i in 0..4u64 {
            overrides.insert(a.0 + i, a_vals[i as usize]);
            overrides.insert(b.0 + i, b_vals[i as usize]);
        }

        compare_jit_vs_interp(&g, &overrides, "bf16_rounding_chain");
    }

    #[test]
    fn test_f16_rounding_add() {
        // Test that output_dtype: F16 applies F16 rounding.
        let mut g = NanoGraph::new();

        let a = g.push_group(
            4,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );
        let b = g.push_group(
            4,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );
        let c = g.push_group(
            4,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
                output_dtype: DType::F16,
            },
            vec![],
            vec![],
            vec![
                InputRef::Affine { base: a, stride: 1 },
                InputRef::Affine { base: b, stride: 1 },
            ],
        );

        for i in 0..4 {
            g.outputs.push(c.offset(i));
        }

        // F16 has ~3.3 decimal digits of precision. Values that expose rounding:
        let a_vals = [1.0f32, 100.0, 0.1, 2048.0];
        let b_vals = [1e-5f32, 0.01, 1e-5, 0.25];
        let mut overrides = HashMap::new();
        for i in 0..4u64 {
            overrides.insert(a.0 + i, a_vals[i as usize]);
            overrides.insert(b.0 + i, b_vals[i as usize]);
        }

        compare_jit_vs_interp(&g, &overrides, "f16_rounding_add");
    }

    #[test]
    fn test_bf16_reduce_sum() {
        // Test that ReduceSum with output_dtype: BF16 rounds the accumulated result.
        let mut g = NanoGraph::new();

        let a = g.push_group(
            8,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );

        let c = g.push_group(
            1,
            ScalarOp::ReduceSum {
                reduce_count: 8,
                reduce_stride: 1,
                compute_dtype: DType::F32,
                output_dtype: DType::BF16,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: a, stride: 0 }],
        );

        g.outputs.push(c);

        let mut overrides = HashMap::new();
        // Values that when summed produce something with low-bit detail that BF16 drops.
        let vals = [1.0f32, 0.001, 0.0001, 2.0, 0.00001, 3.0, 0.000001, 4.0];
        for (i, &v) in vals.iter().enumerate() {
            overrides.insert(a.0 + i as u64, v);
        }

        compare_jit_vs_interp(&g, &overrides, "bf16_reduce_sum");
    }

    #[test]
    fn test_bf16_identity_cast() {
        // Test Identity with BF16 output (used as a dtype cast op).
        let mut g = NanoGraph::new();

        let a = g.push_group(
            4,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );
        let b = g.push_group(
            4,
            ScalarOp::Identity {
                compute_dtype: DType::F32,
                output_dtype: DType::BF16,
            },
            vec![],
            vec![],
            vec![InputRef::Affine { base: a, stride: 1 }],
        );

        for i in 0..4 {
            g.outputs.push(b.offset(i));
        }

        let mut overrides = HashMap::new();
        // Values with fine detail that BF16 will round.
        overrides.insert(a.0, 1.0001f32);
        overrides.insert(a.0 + 1, 3.14159f32);
        overrides.insert(a.0 + 2, 0.123456f32);
        overrides.insert(a.0 + 3, 65504.0f32); // near F16 max, BF16 can represent this

        compare_jit_vs_interp(&g, &overrides, "bf16_identity_cast");
    }

    #[test]
    fn test_bf16_rounding_function_correctness() {
        // Verify that our v13_round_bf16 matches half::bf16::from_f32().to_f32()
        // for various test values.
        let test_values: Vec<f32> = vec![
            0.0,
            1.0,
            -1.0,
            0.5,
            0.1,
            0.123456789,
            1.0001,
            256.001,
            65504.0,
            1e-7,
            -3.14159,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1.0 + 1e-4,
            1000.5,
        ];

        for &v in &test_values {
            let our_result = v13_round_bf16(v);
            let half_result = half::bf16::from_f32(v).to_f32();
            assert_eq!(
                our_result.to_bits(),
                half_result.to_bits(),
                "BF16 rounding mismatch for {}: ours={} (bits {:08x}), half={} (bits {:08x})",
                v,
                our_result,
                our_result.to_bits(),
                half_result,
                half_result.to_bits()
            );
        }
    }

    /// Integration test: build via lowering (milli -> nano), compile, compare.
    #[test]
    fn test_lowered_add() {
        use crate::DynRank;
        use crate::backends::eval_backend::EvalBackend;
        use crate::graph::{GlobalId, Graph};
        use crate::milli_graph::MilliOpGraph;
        use crate::nano_graph::lower::lower_with_info;
        use crate::numeric_tensor::NumericTensor;
        use crate::tensor_info::TensorInfo;

        let mut rng = rand::rng();
        let (mut milli, _) = MilliOpGraph::new(std::iter::empty(), &mut rng);
        let a = milli.add_input(&mut rng);
        let b = milli.add_input(&mut rng);
        let c = crate::milli_graph::ops::SimpleBinary::add(&mut milli, a, b, &mut rng);

        let a_tensor = NumericTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let b_tensor =
            NumericTensor::from_vec_shape(vec![10.0f32, 20.0, 30.0, 40.0], vec![4]).unwrap();

        let mut info_inputs = HashMap::new();
        info_inputs.insert(a, TensorInfo::from(a_tensor.clone()));
        info_inputs.insert(b, TensorInfo::from(b_tensor.clone()));

        let result = lower_with_info(&milli, &info_inputs).unwrap();
        assert!(result.unsupported.is_empty());

        // Build overrides: constants + inputs.
        let mut overrides_f32 = HashMap::new();
        for (&atom_idx, scalar) in &result.numeric_overrides {
            overrides_f32.insert(atom_idx, scalar.to_f64() as f32);
        }
        // Add input tensor values.
        for (id, tensor) in [(a, &a_tensor), (b, &b_tensor)] {
            if let Some(tam) = result.tensor_map.get(&id) {
                let mut backend = EvalBackend::NDArray;
                let f32_t = tensor.cast(DType::F32, &mut backend).unwrap();
                let flat = f32_t.flatten().unwrap();
                let v: Vec<f32> = flat.to_ndarray().unwrap().try_into().unwrap();
                for (i, &val) in v.iter().enumerate() {
                    overrides_f32.insert(tam.base_id.0 + i as u64, val);
                }
            }
        }

        compare_jit_vs_interp(&result.graph, &overrides_f32, "lowered_add");
    }

    /// Integration test: lowered matmul through the full pipeline.
    #[test]
    fn test_lowered_matmul() {
        use crate::DynRank;
        use crate::backends::eval_backend::EvalBackend;
        use crate::graph::{GlobalId, Graph};
        use crate::milli_graph::MilliOpGraph;
        use crate::nano_graph::lower::lower_with_info;
        use crate::numeric_tensor::NumericTensor;
        use crate::tensor_info::TensorInfo;

        let mut rng = rand::rng();
        let (mut milli, _) = MilliOpGraph::new(std::iter::empty(), &mut rng);
        let a = milli.add_input(&mut rng);
        let b = milli.add_input(&mut rng);
        let c = crate::milli_graph::ops::MatMul::push_new_default_precision(
            &mut milli,
            a,
            b,
            DType::F32,
            &mut rng,
        );

        let a_tensor =
            NumericTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])
                .unwrap();
        let b_tensor =
            NumericTensor::from_vec_shape(vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2])
                .unwrap();

        let mut info_inputs = HashMap::new();
        info_inputs.insert(a, TensorInfo::from(a_tensor.clone()));
        info_inputs.insert(b, TensorInfo::from(b_tensor.clone()));

        let result = lower_with_info(&milli, &info_inputs).unwrap();
        assert!(
            result.unsupported.is_empty(),
            "unsupported: {:?}",
            result.unsupported_details
        );

        let mut overrides_f32 = HashMap::new();
        for (&atom_idx, scalar) in &result.numeric_overrides {
            overrides_f32.insert(atom_idx, scalar.to_f64() as f32);
        }
        for (id, tensor) in [(a, &a_tensor), (b, &b_tensor)] {
            if let Some(tam) = result.tensor_map.get(&id) {
                let mut backend = EvalBackend::NDArray;
                let f32_t = tensor.cast(DType::F32, &mut backend).unwrap();
                let flat = f32_t.flatten().unwrap();
                let v: Vec<f32> = flat.to_ndarray().unwrap().try_into().unwrap();
                for (i, &val) in v.iter().enumerate() {
                    overrides_f32.insert(tam.base_id.0 + i as u64, val);
                }
            }
        }

        compare_jit_vs_interp(&result.graph, &overrides_f32, "lowered_matmul_2x3_3x2");
    }

    #[test]
    fn test_partitioned_matmul() {
        use crate::backends::eval_backend::EvalBackend;
        use crate::compiler::attempts::v13_claude::nano_part_creative::partition_nanograph;
        use crate::milli_graph::MilliOpGraph;
        use crate::nano_graph::lower::lower_with_info;
        use crate::numeric_tensor::NumericTensor;
        use crate::tensor_info::TensorInfo;

        let mut rng = rand::rng();
        let (mut milli, _) = MilliOpGraph::new(std::iter::empty(), &mut rng);
        let a = milli.add_input(&mut rng);
        let b = milli.add_input(&mut rng);
        let _c = crate::milli_graph::ops::MatMul::push_new_default_precision(
            &mut milli,
            a,
            b,
            DType::F32,
            &mut rng,
        );

        let a_data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.1 + 0.5).collect();
        let b_data: Vec<f32> = (0..48).map(|i| (i as f32) * 0.05 - 1.0).collect();
        let a_tensor = NumericTensor::from_vec_shape(a_data, vec![4, 6]).unwrap();
        let b_tensor = NumericTensor::from_vec_shape(b_data, vec![6, 8]).unwrap();

        let mut info_inputs = HashMap::new();
        info_inputs.insert(a, TensorInfo::from(a_tensor.clone()));
        info_inputs.insert(b, TensorInfo::from(b_tensor.clone()));

        let result = lower_with_info(&milli, &info_inputs).unwrap();
        assert!(result.unsupported.is_empty());

        // Build f32 overrides
        let mut overrides = HashMap::new();
        for (&atom_idx, scalar) in &result.numeric_overrides {
            overrides.insert(atom_idx, scalar.to_f64() as f32);
        }
        for (id, tensor) in [(a, &a_tensor), (b, &b_tensor)] {
            if let Some(tam) = result.tensor_map.get(&id) {
                let mut backend = EvalBackend::NDArray;
                let f32_t = tensor.cast(DType::F32, &mut backend).unwrap();
                let flat = f32_t.flatten().unwrap();
                let v: Vec<f32> = flat.to_ndarray().unwrap().try_into().unwrap();
                for (i, &val) in v.iter().enumerate() {
                    overrides.insert(tam.base_id.0 + i as u64, val);
                }
            }
        }

        // Compile whole graph (reference)
        let whole = CompiledPipeline::compile(&result.graph).unwrap();
        let whole_vals = whole.execute_full(&overrides);

        // Partition and compile per-kernel
        let partition = partition_nanograph(&result.graph, 4);
        println!("Partitioned into {} kernels", partition.num_kernels);
        for (ki, kg) in partition.kernel_groups.iter().enumerate() {
            println!("  kernel {}: {} groups", ki, kg.len());
        }

        let partitioned = CompiledPipeline::compile_partitioned(&result.graph, &partition).unwrap();
        let part_vals = partitioned.execute_full(&overrides);

        // Compare
        let mut max_err: f32 = 0.0;
        let mut max_err_atom = 0u64;
        for i in 0..whole_vals.len() {
            let err = (whole_vals[i] - part_vals[i]).abs();
            if err > max_err {
                max_err = err;
                max_err_atom = i as u64;
            }
        }
        println!(
            "Max abs error (partitioned vs whole): {} at atom {}",
            max_err, max_err_atom
        );
        assert!(
            max_err < 1e-5,
            "Partitioned execution diverged: max_err={}",
            max_err
        );
    }
}
