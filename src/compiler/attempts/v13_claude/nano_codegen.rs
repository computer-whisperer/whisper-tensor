#![allow(clippy::all, dead_code, unreachable_patterns, unused)]
//! Cranelift JIT codegen for NanoGraph.
//!
//! Compiles a NanoGraph into native code. Each "kernel" is a set of groups
//! that are executed together. For this first pass, each kernel is simply
//! the full set of groups executed in topological (insertion) order.
//!
//! Function signature: `fn(values: *mut f32) -> ()`
//! The `values` pointer addresses a flat f32 array indexed by AtomId.0.

use std::collections::HashMap;

use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, Value};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use crate::nano_graph::{AtomGroup, AtomId, InputRef, NanoGraph, ScalarBinOp, ScalarOp, ScalarUnaryOp, SymDim};
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
    })
}

// ---- Compiled pipeline ----

/// A compiled NanoGraph ready for execution.
pub struct CompiledPipeline {
    _module: JITModule,
    /// One native function pointer per kernel.
    kernel_ptrs: Vec<*const u8>,
    /// Total number of atoms (size of the values buffer).
    num_atoms: usize,
    /// Literal values to pre-fill before execution.
    /// Maps atom index → f32 value.
    literals: Vec<(u32, f32)>,
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
        let isa_builder = cranelift_native::builder()
            .map_err(|e| format!("cranelift native ISA: {}", e))?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| format!("ISA finish: {}", e))?;

        let mut jit_builder =
            JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

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
            for (gi, group) in groups.iter().enumerate() {
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
            _module: module,
            kernel_ptrs: vec![func_ptr],
            num_atoms,
            literals,
            output_atoms: graph.outputs.clone(),
        })
    }

    /// Execute the compiled pipeline.
    ///
    /// `overrides` maps atom index → f32 value for input atoms.
    /// Returns a map of output atom ids → f32 values.
    pub fn execute(&self, overrides: &HashMap<u32, f32>) -> HashMap<u32, f32> {
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
            let func: unsafe extern "C" fn(*mut f32) =
                unsafe { std::mem::transmute(func_ptr) };
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
    pub fn execute_full(&self, overrides: &HashMap<u32, f32>) -> Vec<f32> {
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
            let func: unsafe extern "C" fn(*mut f32) =
                unsafe { std::mem::transmute(func_ptr) };
            unsafe { func(values_ptr) };
        }

        values
    }

    /// Number of atoms in the graph.
    pub fn num_atoms(&self) -> usize {
        self.num_atoms
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
    // Skip Literal groups — their values are pre-filled by the caller.
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
    i_const: u32,
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
            compute_dtype: _,
            output_dtype: _,
        } => {
            // Load source, store to dest. (All f32 for now.)
            let src_val = load_input_ref(
                builder, module, &group.inputs[0], values_ptr, i_val, i_const, table_counter,
            )?;
            store_atom(builder, values_ptr, base_id, i_val, i_const, src_val);
            Ok(())
        }

        ScalarOp::Binary {
            op,
            compute_dtype: _,
            output_dtype: _,
        } => {
            let a = load_input_ref(
                builder, module, &group.inputs[0], values_ptr, i_val, i_const, table_counter,
            )?;
            let b = load_input_ref(
                builder, module, &group.inputs[1], values_ptr, i_val, i_const, table_counter,
            )?;
            let result = emit_binop(builder, module, math, *op, a, b)?;
            store_atom(builder, values_ptr, base_id, i_val, i_const, result);
            Ok(())
        }

        ScalarOp::Unary {
            op,
            compute_dtype: _,
            output_dtype: _,
        } => {
            let x = load_input_ref(
                builder, module, &group.inputs[0], values_ptr, i_val, i_const, table_counter,
            )?;
            let result = emit_unop(builder, module, math, *op, x)?;
            store_atom(builder, values_ptr, base_id, i_val, i_const, result);
            Ok(())
        }

        ScalarOp::Select {
            compute_dtype: _,
            output_dtype: _,
        } => {
            let cond = load_input_ref(
                builder, module, &group.inputs[0], values_ptr, i_val, i_const, table_counter,
            )?;
            let x = load_input_ref(
                builder, module, &group.inputs[1], values_ptr, i_val, i_const, table_counter,
            )?;
            let y = load_input_ref(
                builder, module, &group.inputs[2], values_ptr, i_val, i_const, table_counter,
            )?;
            let zero = builder.ins().f32const(0.0);
            let is_nonzero = builder.ins().fcmp(
                cranelift_codegen::ir::condcodes::FloatCC::NotEqual,
                cond,
                zero,
            );
            let result = builder.ins().select(is_nonzero, x, y);
            store_atom(builder, values_ptr, base_id, i_val, i_const, result);
            Ok(())
        }

        ScalarOp::ReduceSum {
            compute_dtype: _,
            output_dtype: _,
        } => {
            emit_reduce(builder, module, group, graph, values_ptr, i_val, i_const, math, var_counter, table_counter, true)
        }

        ScalarOp::ReduceMax {
            compute_dtype: _,
            output_dtype: _,
        } => {
            emit_reduce(builder, module, group, graph, values_ptr, i_val, i_const, math, var_counter, table_counter, false)
        }

        ScalarOp::IndirectLoad { table_base, output_dtype: _ } => {
            // Load the index from input[0], cast f32→i64, compute address, load value.
            let idx_f32 = load_input_ref(builder, module, &group.inputs[0], values_ptr, i_val, i_const, table_counter)?;
            let idx_i64 = builder.ins().fcvt_to_sint(types::I64, idx_f32);
            let table_base_val = builder.ins().iconst(types::I64, table_base.0 as i64);
            let atom_idx = builder.ins().iadd(table_base_val, idx_i64);
            let byte_offset = builder.ins().ishl_imm(atom_idx, 2); // * 4
            let addr = builder.ins().iadd(values_ptr, byte_offset);
            let result = builder.ins().load(types::F32, MemFlags::trusted(), addr, 0);
            store_atom(builder, values_ptr, base_id, i_val, i_const, result);
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
    i_const: u32,
    math: &MathFuncs,
    var_counter: &mut VarCounter,
    table_counter: &mut usize,
    is_sum: bool,
) -> Result<(), String> {
    assert_eq!(group.reduce_dims.len(), 1, "Only single reduce_dim supported");
    let rd = group.reduce_dims[0];
    let bound = graph
        .sym_dim_bounds
        .get(&rd)
        .copied()
        .ok_or_else(|| format!("Reduce dim {:?} has no bound", rd))?;

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

    // K loop variable.
    let k_var = var_counter.next();
    builder.declare_var(k_var, types::I64);
    let k_init = builder.ins().iconst(types::I64, 0);
    builder.def_var(k_var, k_init);

    let red_header = builder.create_block();
    let red_body = builder.create_block();
    let red_exit = builder.create_block();

    let bound_val = builder.ins().iconst(types::I64, bound as i64);

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

    // Reduction loop body: load source at (i, k), accumulate.
    builder.switch_to_block(red_body);
    let k_val = builder.use_var(k_var);

    // The single input ref for a reduce. For ReduceSum/ReduceMax, inputs[0]
    // should be a SymAffine that depends on both i and k.
    let src_val = load_input_ref_with_k(
        builder,
        module,
        &group.inputs[0],
        values_ptr,
        i_val,
        i_const,
        k_val,
        table_counter,
    )?;

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

    // Exit: store the accumulated result.
    builder.switch_to_block(red_exit);
    builder.seal_block(red_header);
    builder.seal_block(red_body);
    builder.seal_block(red_exit);

    let final_acc = builder.use_var(acc_var);
    store_atom(builder, values_ptr, base_id, i_val, i_const, final_acc);

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
    i_const: u32,
    table_counter: &mut usize,
) -> Result<Value, String> {
    match input {
        InputRef::Broadcast(atom_id) => {
            // Fixed offset, same for all i.
            let byte_offset = (atom_id.0 as i64) * 4;
            if byte_offset <= i32::MAX as i64 {
                Ok(builder.ins().load(
                    types::F32,
                    MemFlags::new(),
                    values_ptr,
                    byte_offset as i32,
                ))
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
                // Only one element — treat like broadcast.
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
            let bytes: Vec<u8> = ids
                .iter()
                .flat_map(|id| id.0.to_le_bytes())
                .collect();
            data_desc.define(bytes.into_boxed_slice());
            module
                .define_data(data_id, &data_desc)
                .map_err(|e| format!("define explicit table: {}", e))?;

            let gv = module.declare_data_in_func(data_id, builder.func);
            let table_ptr = builder.ins().global_value(types::I64, gv);

            // Load atom_id = table[i]
            let i = i_val.unwrap_or_else(|| builder.ins().iconst(types::I64, i_const as i64));
            let idx_byte_off = builder.ins().imul_imm(i, 4);
            let idx_addr = builder.ins().iadd(table_ptr, idx_byte_off);
            let raw_atom_id = builder.ins().load(types::I32, MemFlags::new(), idx_addr, 0);
            let atom_id_i64 = builder.ins().uextend(types::I64, raw_atom_id);

            // Load values[atom_id]
            let data_byte_off = builder.ins().imul_imm(atom_id_i64, 4);
            let data_addr = builder.ins().iadd(values_ptr, data_byte_off);
            Ok(builder.ins().load(types::F32, MemFlags::new(), data_addr, 0))
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
    i_const: u32,
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
                None => builder.ins().iconst(types::I64, (*stride_i as i64) * (i_const as i64)),
            };
            let sk = builder.ins().imul_imm(k_val, *stride_k as i64);
            let idx = builder.ins().iadd(si, sk);
            let idx = builder.ins().iadd_imm(idx, base.0 as i64);
            let byte_offset = builder.ins().imul_imm(idx, 4);
            let addr = builder.ins().iadd(values_ptr, byte_offset);
            Ok(builder.ins().load(types::F32, MemFlags::new(), addr, 0))
        }

        // For non-SymAffine inputs in a reduce context, k doesn't affect the address.
        InputRef::Broadcast(_) | InputRef::Affine { .. } | InputRef::Explicit(_) => {
            load_input_ref(builder, module, input, values_ptr, i_val, i_const, table_counter)
        }
    }
}

// ---- Atom store helper ----

/// Store a value into values[base_id + i].
fn store_atom(
    builder: &mut FunctionBuilder,
    values_ptr: Value,
    base_id: u32,
    i_val: Option<Value>,
    i_const: u32,
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
            let cmp = builder.ins().fcmp(
                cranelift_codegen::ir::condcodes::FloatCC::GreaterThan,
                a,
                b,
            );
            builder.ins().select(cmp, a, b)
        }
        ScalarBinOp::Min => {
            let cmp = builder.ins().fcmp(
                cranelift_codegen::ir::condcodes::FloatCC::LessThan,
                a,
                b,
            );
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
        ScalarUnaryOp::Abs => {
            let func_ref = module.declare_func_in_func(math.fabsf, builder.func);
            let call = builder.ins().call(func_ref, &[x]);
            builder.inst_results(call)[0]
        }
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
        ScalarUnaryOp::Sqrt => {
            let func_ref = module.declare_func_in_func(math.sqrtf, builder.func);
            let call = builder.ins().call(func_ref, &[x]);
            builder.inst_results(call)[0]
        }
        ScalarUnaryOp::Reciprocal => {
            let one = builder.ins().f32const(1.0);
            builder.ins().fdiv(one, x)
        }
        ScalarUnaryOp::Tanh => {
            let func_ref = module.declare_func_in_func(math.tanhf, builder.func);
            let call = builder.ins().call(func_ref, &[x]);
            builder.inst_results(call)[0]
        }
        ScalarUnaryOp::Floor => {
            let func_ref = module.declare_func_in_func(math.floorf, builder.func);
            let call = builder.ins().call(func_ref, &[x]);
            builder.inst_results(call)[0]
        }
        ScalarUnaryOp::Ceil => {
            let func_ref = module.declare_func_in_func(math.ceilf, builder.func);
            let call = builder.ins().call(func_ref, &[x]);
            builder.inst_results(call)[0]
        }
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
    fn compare_jit_vs_interp(
        graph: &NanoGraph,
        overrides_f32: &HashMap<u32, f32>,
        label: &str,
    ) {
        // Build NumericScalar overrides for interpreter.
        let overrides_ns: HashMap<u32, NumericScalar> = overrides_f32
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
            label, interp_time, compile_time, jit_time, graph.num_atoms()
        );

        // Compare all atoms.
        let mut max_abs_err: f64 = 0.0;
        let mut max_err_atom: u32 = 0;
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
        for i in 0..4u32 {
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
        for i in 0..4u32 {
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
        for i in 0..4u32 {
            overrides.insert(a.0 + i, i as f32 + 1.0);
        }
        overrides.insert(b.0, 100.0);

        compare_jit_vs_interp(&g, &overrides, "broadcast_add");
    }

    #[test]
    fn test_matmul_4x8x16() {
        // MatMul: C[i,j] = sum_k A[i,k] * B[k,j]
        // A is 4x8, B is 8x16, C is 4x16 = 64 elements.
        //
        // Structure: Products group (m*n*k atoms with Explicit refs for non-linear
        // index mapping) followed by ReduceSum (m*n atoms with SymAffine).
        let m = 4u32;
        let k_dim = 8u32;
        let n = 16u32;

        let mut g = NanoGraph::new();
        let k_sym = g.bounded_sym_dim("k", k_dim as u64);

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

        // Products group: count = m * n * k_dim
        // atom at offset p, where p = row*(n*k_dim) + col*k_dim + k:
        //   A input: A[row*k_dim + k] = Explicit[p]
        //   B input: B[k*n + col] = Explicit[p]
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
            vec![
                InputRef::Explicit(a_refs),
                InputRef::Explicit(b_refs),
            ],
        );

        // ReduceSum group: count = m * n
        // For output atom idx (0..m*n):
        //   sum over k of prods[idx * k_dim + k]
        //   SymAffine(base=prods, stride_i=k_dim, stride_k=1)
        let c = g.push_group(
            m * n,
            ScalarOp::ReduceSum {
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![k_sym],
            vec![InputRef::SymAffine {
                base: prods,
                stride_i: k_dim as i32,
                stride_k: 1,
            }],
        );

        for i in 0..(m * n) {
            g.outputs.push(c.offset(i));
        }

        // Fill A with sequential data.
        let mut overrides = HashMap::new();
        for i in 0..(m * k_dim) {
            overrides.insert(a.0 + i, (i as f32) + 1.0);
        }
        // Fill B with identity-ish pattern.
        for i in 0..(k_dim * n) {
            let row = i / n;
            let col = i % n;
            overrides.insert(b_base.0 + i, if row == col % k_dim { 1.0 } else { 0.0 });
        }

        assert!(g.validate().is_empty(), "validation errors: {:?}", g.validate());

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
        for i in 0..4u32 {
            overrides.insert(a.0 + i, (i as f32 + 1.0) * 3.14);
        }

        compare_jit_vs_interp(&g, &overrides, "identity_passthrough");
    }

    #[test]
    fn test_select() {
        let mut g = NanoGraph::new();

        // cond, x, y: 4 elements each
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
                InputRef::Affine { base: cond, stride: 1 },
                InputRef::Affine { base: x, stride: 1 },
                InputRef::Affine { base: y, stride: 1 },
            ],
        );

        for i in 0..4 {
            g.outputs.push(result.offset(i));
        }

        let mut overrides = HashMap::new();
        // cond: [1, 0, 1, 0] (nonzero = true)
        overrides.insert(cond.0, 1.0f32);
        overrides.insert(cond.0 + 1, 0.0);
        overrides.insert(cond.0 + 2, 5.0);
        overrides.insert(cond.0 + 3, 0.0);
        // x: [10, 20, 30, 40]
        for i in 0..4u32 {
            overrides.insert(x.0 + i, (i as f32 + 1.0) * 10.0);
        }
        // y: [100, 200, 300, 400]
        for i in 0..4u32 {
            overrides.insert(y.0 + i, (i as f32 + 1.0) * 100.0);
        }

        compare_jit_vs_interp(&g, &overrides, "select");
    }

    #[test]
    fn test_reduce_max() {
        let mut g = NanoGraph::new();
        let k_sym = g.bounded_sym_dim("k", 8);

        // 8 elements to reduce.
        let a = g.push_group(
            8,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
            vec![],
        );

        // ReduceMax over k: 1 output atom.
        let c = g.push_group(
            1,
            ScalarOp::ReduceMax {
                compute_dtype: DType::F32,
                output_dtype: DType::F32,
            },
            vec![],
            vec![k_sym],
            vec![InputRef::SymAffine {
                base: a,
                stride_i: 0,
                stride_k: 1,
            }],
        );

        g.outputs.push(c);

        let mut overrides = HashMap::new();
        let test_vals = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        for (i, &v) in test_vals.iter().enumerate() {
            overrides.insert(a.0 + i as u32, v);
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

        // Gather: pick elements [2, 0, 3] from a.
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

    /// Integration test: build via lowering (milli → nano), compile, compare.
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

        let a_tensor =
            NumericTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
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
                    overrides_f32.insert(tam.base_id.0 + i as u32, val);
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

        let a_tensor = NumericTensor::from_vec_shape(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        )
        .unwrap();
        let b_tensor = NumericTensor::from_vec_shape(
            vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![3, 2],
        )
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
                    overrides_f32.insert(tam.base_id.0 + i as u32, val);
                }
            }
        }

        compare_jit_vs_interp(&result.graph, &overrides_f32, "lowered_matmul_2x3_3x2");
    }
}
