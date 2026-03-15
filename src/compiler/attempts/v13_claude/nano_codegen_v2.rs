#![allow(clippy::all, dead_code, unreachable_patterns, unused)]
//! Cranelift JIT codegen for ExecutionPlan (v2c lane planner).
//!
//! Compiles each lane's work in each phase as a separate Cranelift function.
//! Each phase gets its own JITModule to avoid relocation overflow.
//!
//! Function signature: `fn(values: *mut f32) -> ()`
//!
//! Key difference from nano_codegen.rs: instead of iterating all atoms in a
//! group (0..group.count), each function iterates a SUB-RANGE:
//! `atom_offset..atom_offset+atom_count` from its LaneWork item.

use std::collections::HashMap;

use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, Value};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use crate::dtype::DType;
use crate::nano_graph::{
    AtomGroup, AtomId, InputRef, NanoGraph, ScalarBinOp, ScalarOp, ScalarUnaryOp,
};

use super::nano_plan_v2c::{ExecutionPlan, LaneWork, Phase};

// ---- Math function wrappers (extern "C" for Cranelift calls) ----
// Reuse the same external functions as nano_codegen.rs. We re-declare them here
// so this module is self-contained (no cross-module function pointer leakage).

extern "C" fn v2_expf(x: f32) -> f32 {
    x.exp()
}
extern "C" fn v2_logf(x: f32) -> f32 {
    x.ln()
}
extern "C" fn v2_tanhf(x: f32) -> f32 {
    x.tanh()
}
extern "C" fn v2_sqrtf(x: f32) -> f32 {
    x.sqrt()
}
extern "C" fn v2_floorf(x: f32) -> f32 {
    x.floor()
}
extern "C" fn v2_ceilf(x: f32) -> f32 {
    x.ceil()
}
extern "C" fn v2_fabsf(x: f32) -> f32 {
    x.abs()
}
extern "C" fn v2_powf(x: f32, y: f32) -> f32 {
    x.powf(y)
}
extern "C" fn v2_fmodf(x: f32, y: f32) -> f32 {
    x % y
}
extern "C" fn v2_round_bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    f32::from_bits(rounded & 0xFFFF0000)
}
extern "C" fn v2_round_f16(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

// ---- Math function declarations ----

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
        expf: decl(module, "v2_expf", &sig1)?,
        logf: decl(module, "v2_logf", &sig1)?,
        tanhf: decl(module, "v2_tanhf", &sig1)?,
        sqrtf: decl(module, "v2_sqrtf", &sig1)?,
        floorf: decl(module, "v2_floorf", &sig1)?,
        ceilf: decl(module, "v2_ceilf", &sig1)?,
        fabsf: decl(module, "v2_fabsf", &sig1)?,
        powf: decl(module, "v2_powf", &sig2)?,
        fmodf: decl(module, "v2_fmodf", &sig2)?,
        round_bf16: decl(module, "v2_round_bf16", &sig1)?,
        round_f16: decl(module, "v2_round_f16", &sig1)?,
    })
}

fn register_math_symbols(jit_builder: &mut JITBuilder) {
    jit_builder.symbol("v2_expf", v2_expf as *const u8);
    jit_builder.symbol("v2_logf", v2_logf as *const u8);
    jit_builder.symbol("v2_tanhf", v2_tanhf as *const u8);
    jit_builder.symbol("v2_sqrtf", v2_sqrtf as *const u8);
    jit_builder.symbol("v2_floorf", v2_floorf as *const u8);
    jit_builder.symbol("v2_ceilf", v2_ceilf as *const u8);
    jit_builder.symbol("v2_fabsf", v2_fabsf as *const u8);
    jit_builder.symbol("v2_powf", v2_powf as *const u8);
    jit_builder.symbol("v2_fmodf", v2_fmodf as *const u8);
    jit_builder.symbol("v2_round_bf16", v2_round_bf16 as *const u8);
    jit_builder.symbol("v2_round_f16", v2_round_f16 as *const u8);
}

// ---- Variable counter ----

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

// ---- Compiled plan ----

/// A compiled ExecutionPlan ready for single-threaded execution.
pub struct CompiledPlan {
    /// phase_idx -> list of function pointers (one per lane-work-item in that phase).
    phase_funcs: Vec<Vec<*const u8>>,
    /// Keep JIT modules alive so function pointers remain valid.
    modules: Vec<JITModule>,
    /// Total number of atoms (size of the values buffer).
    num_atoms: usize,
    /// Literal values to pre-fill before execution.
    literals: Vec<(u64, f32)>,
}

unsafe impl Send for CompiledPlan {}
unsafe impl Sync for CompiledPlan {}

impl CompiledPlan {
    /// Compile an ExecutionPlan into native code.
    ///
    /// Each phase is compiled into its own JITModule. Within each phase,
    /// every LaneWork item across all lanes becomes a separate function.
    pub fn compile(graph: &NanoGraph, plan: &ExecutionPlan) -> Result<Self, String> {
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

        let mut all_modules = Vec::new();
        let mut phase_funcs: Vec<Vec<*const u8>> = Vec::new();
        let mut global_table_counter: usize = 0;

        for (phase_idx, phase) in plan.phases.iter().enumerate() {
            // Collect all work items for this phase across all lanes.
            let mut work_items: Vec<&LaneWork> = Vec::new();
            for lane_work in &phase.lane_work {
                for item in lane_work {
                    work_items.push(item);
                }
            }

            if work_items.is_empty() {
                phase_funcs.push(Vec::new());
                continue;
            }

            // Create a JIT module for this phase.
            let mut flag_builder = settings::builder();
            flag_builder.set("opt_level", "speed").unwrap();
            let isa_builder = cranelift_native::builder()
                .map_err(|e| format!("cranelift native ISA: {}", e))?;
            let isa = isa_builder
                .finish(settings::Flags::new(flag_builder))
                .map_err(|e| format!("ISA finish: {}", e))?;

            let mut jit_builder =
                JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
            register_math_symbols(&mut jit_builder);

            let mut module = JITModule::new(jit_builder);
            let math_funcs = declare_math_funcs(&mut module)?;

            let mut func_ptrs_this_phase = Vec::new();
            let mut func_ids = Vec::new();

            // Compile each work item as a separate function within this module.
            for (wi, work) in work_items.iter().enumerate() {
                let group = &groups[work.group_idx];

                // Skip literal groups -- pre-filled.
                if matches!(&group.op, ScalarOp::Literal(_)) {
                    continue;
                }

                if work.atom_count == 0 {
                    continue;
                }

                let mut func_ctx = FunctionBuilderContext::new();
                let mut ctx = module.make_context();
                ctx.func.signature.params.push(AbiParam::new(types::I64));

                let func_name = format!("v2_p{}_w{}", phase_idx, wi);
                let func_id = module
                    .declare_function(&func_name, Linkage::Local, &ctx.func.signature)
                    .map_err(|e| format!("declare {}: {}", func_name, e))?;

                {
                    let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
                    let entry = builder.create_block();
                    builder.append_block_params_for_function_params(entry);
                    builder.switch_to_block(entry);
                    builder.seal_block(entry);

                    let values_ptr = builder.block_params(entry)[0];
                    let mut var_counter = VarCounter::new();

                    emit_lane_work(
                        &mut builder,
                        &mut module,
                        group,
                        graph,
                        values_ptr,
                        work.atom_offset,
                        work.atom_count,
                        &math_funcs,
                        &mut var_counter,
                        &mut global_table_counter,
                    )?;

                    builder.ins().return_(&[]);
                    builder.finalize();
                }

                module
                    .define_function(func_id, &mut ctx)
                    .map_err(|e| format!("define {}: {}", func_name, e))?;

                func_ids.push(func_id);
            }

            module
                .finalize_definitions()
                .map_err(|e| format!("finalize phase {}: {}", phase_idx, e))?;

            for &fid in &func_ids {
                func_ptrs_this_phase.push(module.get_finalized_function(fid));
            }

            phase_funcs.push(func_ptrs_this_phase);
            all_modules.push(module);
        }

        Ok(CompiledPlan {
            phase_funcs,
            modules: all_modules,
            num_atoms,
            literals,
        })
    }

    /// Execute the compiled plan, returning the full values buffer.
    pub fn execute(&self, overrides: &HashMap<u64, f32>) -> Vec<f32> {
        let mut values = vec![0.0f32; self.num_atoms];

        // Pre-fill literals.
        for &(idx, val) in &self.literals {
            values[idx as usize] = val;
        }

        // Apply overrides.
        for (&idx, &val) in overrides {
            values[idx as usize] = val;
        }

        // Execute phases sequentially; within each phase, execute all funcs.
        let values_ptr = values.as_mut_ptr();
        for phase_funcs in &self.phase_funcs {
            for &func_ptr in phase_funcs {
                let func: unsafe extern "C" fn(*mut f32) =
                    unsafe { std::mem::transmute(func_ptr) };
                unsafe { func(values_ptr) };
            }
        }

        values
    }

    /// Number of atoms in the graph.
    pub fn num_atoms(&self) -> usize {
        self.num_atoms
    }
}

// ---- Lane work emission ----

/// Emit Cranelift IR for a LaneWork item: a sub-range of an AtomGroup.
///
/// Iterates i in atom_offset..atom_offset+atom_count (where i is the
/// offset within the group, NOT within the lane work).
fn emit_lane_work(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    group: &AtomGroup,
    graph: &NanoGraph,
    values_ptr: Value,
    atom_offset: u64,
    atom_count: u64,
    math: &MathFuncs,
    var_counter: &mut VarCounter,
    table_counter: &mut usize,
) -> Result<(), String> {
    // Skip literals (pre-filled).
    if matches!(&group.op, ScalarOp::Literal(_)) {
        return Ok(());
    }

    if atom_count == 0 {
        return Ok(());
    }

    // For count == 1, emit inline (no loop), i = atom_offset (constant).
    if atom_count == 1 {
        emit_group_body(
            builder,
            module,
            group,
            graph,
            values_ptr,
            None, // no loop variable
            atom_offset,
            math,
            var_counter,
            table_counter,
        )?;
        return Ok(());
    }

    // Emit loop: for i in atom_offset..atom_offset+atom_count
    let loop_header = builder.create_block();
    let loop_body = builder.create_block();
    let loop_exit = builder.create_block();

    let start = builder.ins().iconst(types::I64, atom_offset as i64);
    let end = builder
        .ins()
        .iconst(types::I64, (atom_offset + atom_count) as i64);

    builder.ins().jump(loop_header, &[start]);

    // Loop header: i = phi(start, i+1)
    builder.switch_to_block(loop_header);
    builder.append_block_param(loop_header, types::I64);
    let i_val = builder.block_params(loop_header)[0];

    let cmp = builder.ins().icmp(
        cranelift_codegen::ir::condcodes::IntCC::SignedLessThan,
        i_val,
        end,
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
        0, // i_const unused when i_val is Some
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

// ---- Group body emission (per-iteration) ----

/// Emit the body of a group iteration for one value of i.
///
/// `i_val` is None when atom_count == 1 (i is the constant `i_const`).
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
        ScalarOp::Literal(_) => Ok(()),

        ScalarOp::Identity { output_dtype, .. } => {
            let src = load_input_ref(
                builder, module, &group.inputs[0], values_ptr, i_val, i_const, table_counter,
            )?;
            let rounded = emit_output_round(builder, module, math, src, *output_dtype);
            store_atom(builder, values_ptr, base_id, i_val, i_const, rounded);
            Ok(())
        }

        ScalarOp::Binary {
            op, output_dtype, ..
        } => {
            let a = load_input_ref(
                builder, module, &group.inputs[0], values_ptr, i_val, i_const, table_counter,
            )?;
            let b = load_input_ref(
                builder, module, &group.inputs[1], values_ptr, i_val, i_const, table_counter,
            )?;
            let result = emit_binop(builder, module, math, *op, a, b)?;
            let rounded = emit_output_round(builder, module, math, result, *output_dtype);
            store_atom(builder, values_ptr, base_id, i_val, i_const, rounded);
            Ok(())
        }

        ScalarOp::Unary {
            op, output_dtype, ..
        } => {
            let x = load_input_ref(
                builder, module, &group.inputs[0], values_ptr, i_val, i_const, table_counter,
            )?;
            let result = emit_unop(builder, module, math, *op, x)?;
            let rounded = emit_output_round(builder, module, math, result, *output_dtype);
            store_atom(builder, values_ptr, base_id, i_val, i_const, rounded);
            Ok(())
        }

        ScalarOp::Select { output_dtype, .. } => {
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
            let rounded = emit_output_round(builder, module, math, result, *output_dtype);
            store_atom(builder, values_ptr, base_id, i_val, i_const, rounded);
            Ok(())
        }

        ScalarOp::ReduceSum { output_dtype, .. } => emit_reduce(
            builder, module, group, graph, values_ptr, i_val, i_const, math, var_counter,
            table_counter, true, *output_dtype,
        ),

        ScalarOp::ReduceMax { output_dtype, .. } => emit_reduce(
            builder, module, group, graph, values_ptr, i_val, i_const, math, var_counter,
            table_counter, false, *output_dtype,
        ),

        ScalarOp::IndirectLoad {
            table_base,
            output_dtype,
        } => {
            let idx_f32 = load_input_ref(
                builder, module, &group.inputs[0], values_ptr, i_val, i_const, table_counter,
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
            ))
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

    let k_offset = builder.ins().imul_imm(k_val, reduce_stride);
    let src_idx = builder.ins().iadd(base_idx, k_offset);
    let byte_offset = builder.ins().imul_imm(src_idx, 4);
    let addr = builder.ins().iadd(values_ptr, byte_offset);
    let src_val = builder.ins().load(types::F32, MemFlags::new(), addr, 0);

    let acc = builder.use_var(acc_var);
    let new_acc = if is_sum {
        builder.ins().fadd(acc, src_val)
    } else {
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
    let rounded = emit_output_round(builder, module, math, final_acc, output_dtype);
    store_atom(builder, values_ptr, base_id, i_val, i_const, rounded);

    Ok(())
}

// ---- Input ref resolution ----

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

            let data_name = format!("v2_explicit_{}", *table_counter);
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

            let i = i_val.unwrap_or_else(|| builder.ins().iconst(types::I64, i_const as i64));
            let idx_byte_off = builder.ins().imul_imm(i, 8);
            let idx_addr = builder.ins().iadd(table_ptr, idx_byte_off);
            let atom_id_i64 = builder.ins().load(types::I64, MemFlags::new(), idx_addr, 0);

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
            // In non-reduce context, k=0, so this reduces to base + stride_i * i.
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
            let atom_idx = match i_val {
                Some(iv) => {
                    let block_idx = if repeat.is_power_of_two() {
                        let shift = repeat.trailing_zeros() as i64;
                        builder.ins().ushr_imm(iv, shift)
                    } else {
                        let rep = builder.ins().iconst(types::I64, *repeat as i64);
                        builder.ins().udiv(iv, rep)
                    };
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

// ---- Atom store ----

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
        None => builder
            .ins()
            .iconst(types::I64, (base_id + i_const) as i64),
    };
    let byte_offset = builder.ins().imul_imm(atom_idx, 4);
    let addr = builder.ins().iadd(values_ptr, byte_offset);
    builder.ins().store(MemFlags::new(), val, addr, 0);
}

// ---- Dtype rounding ----

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
        _ => val,
    }
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
        ScalarBinOp::Equal => {
            let cmp = builder
                .ins()
                .fcmp(cranelift_codegen::ir::condcodes::FloatCC::Equal, a, b);
            let one = builder.ins().f32const(1.0);
            let zero = builder.ins().f32const(0.0);
            builder.ins().select(cmp, one, zero)
        }
        ScalarBinOp::Greater => {
            let cmp = builder.ins().fcmp(
                cranelift_codegen::ir::condcodes::FloatCC::GreaterThan,
                a,
                b,
            );
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
            let cmp = builder.ins().fcmp(
                cranelift_codegen::ir::condcodes::FloatCC::LessThan,
                a,
                b,
            );
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
            let a_nz = builder.ins().fcmp(
                cranelift_codegen::ir::condcodes::FloatCC::NotEqual,
                a,
                zero,
            );
            let b_nz = builder.ins().fcmp(
                cranelift_codegen::ir::condcodes::FloatCC::NotEqual,
                b,
                zero,
            );
            let both = builder.ins().band(a_nz, b_nz);
            let one = builder.ins().f32const(1.0);
            builder.ins().select(both, one, zero)
        }
        ScalarBinOp::Or => {
            let zero = builder.ins().f32const(0.0);
            let a_nz = builder.ins().fcmp(
                cranelift_codegen::ir::condcodes::FloatCC::NotEqual,
                a,
                zero,
            );
            let b_nz = builder.ins().fcmp(
                cranelift_codegen::ir::condcodes::FloatCC::NotEqual,
                b,
                zero,
            );
            let either = builder.ins().bor(a_nz, b_nz);
            let one = builder.ins().f32const(1.0);
            builder.ins().select(either, one, zero)
        }
        ScalarBinOp::Xor => {
            let zero = builder.ins().f32const(0.0);
            let a_nz = builder.ins().fcmp(
                cranelift_codegen::ir::condcodes::FloatCC::NotEqual,
                a,
                zero,
            );
            let b_nz = builder.ins().fcmp(
                cranelift_codegen::ir::condcodes::FloatCC::NotEqual,
                b,
                zero,
            );
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
    use crate::nano_graph::eval::NanoEval;
    use crate::nano_graph::{AtomId, InputRef, NanoGraph, ScalarOp};
    use crate::numeric_scalar::NumericScalar;

    /// Compare CompiledPlan output against the NanoEval interpreter.
    fn compare_plan_vs_interp(
        graph: &NanoGraph,
        plan: &ExecutionPlan,
        overrides_f32: &HashMap<u64, f32>,
        label: &str,
    ) {
        // Interpreter reference.
        let overrides_ns: HashMap<u64, NumericScalar> = overrides_f32
            .iter()
            .map(|(&k, &v)| (k, NumericScalar::F32(v)))
            .collect();
        let t0 = std::time::Instant::now();
        let interp = NanoEval::eval(graph, &overrides_ns);
        let interp_time = t0.elapsed();

        // Compile the plan.
        let t0 = std::time::Instant::now();
        let compiled = CompiledPlan::compile(graph, plan).expect("compile failed");
        let compile_time = t0.elapsed();

        // Execute.
        let t0 = std::time::Instant::now();
        let jit_values = compiled.execute(overrides_f32);
        let exec_time = t0.elapsed();

        eprintln!(
            "[{}] interp: {:?}, compile: {:?}, exec: {:?}, atoms: {}, phases: {}, lanes: {}",
            label,
            interp_time,
            compile_time,
            exec_time,
            graph.num_atoms(),
            plan.phases.len(),
            plan.num_lanes,
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
    fn test_plan_lowered_matmul() {
        use crate::backends::eval_backend::EvalBackend;
        use crate::milli_graph::MilliOpGraph;
        use crate::nano_graph::lower::lower_with_info;
        use crate::numeric_tensor::NumericTensor;
        use crate::tensor_info::TensorInfo;

        use super::super::nano_plan_v2c::plan_execution;

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

        // 4x6 * 6x8 = 4x8 matmul
        let a_data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.1 + 0.5).collect();
        let b_data: Vec<f32> = (0..48).map(|i| (i as f32) * 0.05 - 1.0).collect();
        let a_tensor = NumericTensor::from_vec_shape(a_data, vec![4, 6]).unwrap();
        let b_tensor = NumericTensor::from_vec_shape(b_data, vec![6, 8]).unwrap();

        let mut info_inputs = HashMap::new();
        info_inputs.insert(a, TensorInfo::from(a_tensor.clone()));
        info_inputs.insert(b, TensorInfo::from(b_tensor.clone()));

        let result = lower_with_info(&milli, &info_inputs).unwrap();
        assert!(
            result.unsupported.is_empty(),
            "unsupported: {:?}",
            result.unsupported_details
        );

        // Build f32 overrides.
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

        // Plan with 2 lanes.
        let plan = plan_execution(&result.graph, 2);
        eprintln!(
            "Plan: {} phases, {} lanes",
            plan.phases.len(),
            plan.num_lanes
        );
        for (pi, phase) in plan.phases.iter().enumerate() {
            for (li, lane) in phase.lane_work.iter().enumerate() {
                eprintln!("  phase {} lane {}: {} items", pi, li, lane.len());
            }
        }

        compare_plan_vs_interp(&result.graph, &plan, &overrides, "plan_matmul_4x6x8_2lanes");

        // Plan with 4 lanes.
        let plan4 = plan_execution(&result.graph, 4);
        compare_plan_vs_interp(&result.graph, &plan4, &overrides, "plan_matmul_4x6x8_4lanes");
    }

    #[test]
    fn test_plan_small_matmul() {
        use crate::backends::eval_backend::EvalBackend;
        use crate::milli_graph::MilliOpGraph;
        use crate::nano_graph::lower::lower_with_info;
        use crate::numeric_tensor::NumericTensor;
        use crate::tensor_info::TensorInfo;

        use super::super::nano_plan_v2c::plan_execution;

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

        // 2x3 * 3x2 = 2x2 matmul
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
        assert!(result.unsupported.is_empty());

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

        // Single-lane plan (should behave identically to whole-graph codegen).
        let plan1 = plan_execution(&result.graph, 1);
        compare_plan_vs_interp(&result.graph, &plan1, &overrides, "plan_matmul_2x3x2_1lane");

        // Two lanes.
        let plan2 = plan_execution(&result.graph, 2);
        compare_plan_vs_interp(&result.graph, &plan2, &overrides, "plan_matmul_2x3x2_2lanes");
    }
}
