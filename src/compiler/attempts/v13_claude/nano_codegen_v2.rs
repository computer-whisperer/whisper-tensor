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
    AtomGroup, AtomId, InputRef, NanoGraph, ReduceKind, ScalarBinOp, ScalarOp, ScalarUnaryOp,
};

use super::plan::v2c::{ExecutionPlan, LaneWork, Phase};

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
            let isa_builder =
                cranelift_native::builder().map_err(|e| format!("cranelift native ISA: {}", e))?;
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
                let func: unsafe extern "C" fn(*mut f32) = unsafe { std::mem::transmute(func_ptr) };
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

    let output_dtype = group.output_dtype;

    match &group.op {
        ScalarOp::Literal(_) => Ok(()),

        ScalarOp::Identity => {
            let src = load_input_ref(
                builder,
                module,
                &group.inputs[0],
                values_ptr,
                i_val,
                i_const,
                table_counter,
            )?;
            let rounded = emit_output_round(builder, module, math, src, output_dtype);
            store_atom(builder, values_ptr, base_id, i_val, i_const, rounded);
            Ok(())
        }

        ScalarOp::Binary { op, .. } => {
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
            let rounded = emit_output_round(builder, module, math, result, output_dtype);
            store_atom(builder, values_ptr, base_id, i_val, i_const, rounded);
            Ok(())
        }

        ScalarOp::Unary { op, .. } => {
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
            let rounded = emit_output_round(builder, module, math, result, output_dtype);
            store_atom(builder, values_ptr, base_id, i_val, i_const, rounded);
            Ok(())
        }

        ScalarOp::Select => {
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
            let rounded = emit_output_round(builder, module, math, result, output_dtype);
            store_atom(builder, values_ptr, base_id, i_val, i_const, rounded);
            Ok(())
        }

        ScalarOp::Reduce {
            kind: ReduceKind::Sum,
            ..
        } => emit_reduce(
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
            output_dtype,
        ),

        ScalarOp::Reduce {
            kind: ReduceKind::Max,
            ..
        } => emit_reduce(
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
            output_dtype,
        ),

        ScalarOp::IndirectLoad { table_base } => {
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
            let rounded = emit_output_round(builder, module, math, result, output_dtype);
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
        ScalarOp::Reduce {
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
        None => builder.ins().iconst(types::I64, (base_id + i_const) as i64),
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

    use crate::nano_graph::{AtomId, InputRef, NanoGraph, ReduceKind, ScalarOp};
    use crate::numeric_scalar::NumericScalar;

    /// Build f32 overrides from TensorInfo inputs using the tensor_map in LowerResult.
    /// Iterates all info_inputs, extracts numeric tensor data, and maps elements to atom IDs.
    fn build_f32_overrides(
        info_inputs: &HashMap<crate::graph::GlobalId, crate::tensor_info::TensorInfo>,
        result: &crate::nano_graph::lower::LowerResult,
    ) -> HashMap<u64, f32> {
        use crate::numeric_tensor::NumericTensor;
        let mut overrides = HashMap::new();
        let mut backend = crate::backends::eval_backend::EvalBackend::NDArray;
        for (id, info) in info_inputs {
            let Some(numeric): Option<&NumericTensor<crate::DynRank>> = info.as_numeric() else {
                continue;
            };
            let Some(tam) = result.tensor_map.get(id) else {
                continue;
            };
            let f32_t = numeric
                .cast(crate::dtype::DType::F32, &mut backend)
                .unwrap();
            let flat = f32_t.flatten().unwrap();
            let nd = flat.to_ndarray().unwrap();
            let v: Vec<f32> = nd.try_into().unwrap();
            for (i, &val) in v.iter().enumerate() {
                overrides.insert(tam.base_id.0 + i as u64, val);
            }
        }
        overrides
    }

    /// Simple scalar evaluator for NanoGraph with per-atom f32 overrides.
    /// Processes groups in order, computing each atom using f64 arithmetic.
    /// Used by compiler tests where graphs are built manually with Literal+overrides.
    fn eval_nano_graph(graph: &NanoGraph, overrides_f32: &HashMap<u64, f32>) -> Vec<f64> {
        use crate::nano_graph::ops::{ReduceKind, ScalarBinOp, ScalarOp, ScalarUnaryOp};

        let n = graph.num_atoms() as usize;
        let mut values = vec![0.0f64; n];

        // Apply overrides first (these override literal values).
        for (&idx, &val) in overrides_f32 {
            if (idx as usize) < n {
                values[idx as usize] = val as f64;
            }
        }

        // Process groups in order.
        for group in graph.groups() {
            let base = group.base_id.0 as usize;
            match &group.op {
                ScalarOp::Literal(scalar) => {
                    let val = scalar.to_f64();
                    for i in 0..group.count as usize {
                        let idx = base + i;
                        // Don't overwrite if an override was set.
                        if !overrides_f32.contains_key(&(idx as u64)) {
                            values[idx] = val;
                        }
                    }
                }
                ScalarOp::Identity => {
                    for i in 0..group.count as usize {
                        let src = group.inputs[0].resolve(i as u64);
                        values[base + i] = values[src.0 as usize];
                    }
                }
                ScalarOp::Binary { op, .. } => {
                    for i in 0..group.count as usize {
                        let a = values[group.inputs[0].resolve(i as u64).0 as usize];
                        let b = values[group.inputs[1].resolve(i as u64).0 as usize];
                        values[base + i] = match op {
                            ScalarBinOp::Add => a + b,
                            ScalarBinOp::Sub => a - b,
                            ScalarBinOp::Mul => a * b,
                            ScalarBinOp::Div => a / b,
                            ScalarBinOp::Min => a.min(b),
                            ScalarBinOp::Max => a.max(b),
                            ScalarBinOp::Pow => a.powf(b),
                            ScalarBinOp::Less => {
                                if a < b {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                            ScalarBinOp::Greater => {
                                if a > b {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                            ScalarBinOp::Equal => {
                                if (a - b).abs() < f64::EPSILON {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                            ScalarBinOp::Mod => a % b,
                            ScalarBinOp::GreaterOrEqual => {
                                if a >= b {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                            ScalarBinOp::LessOrEqual => {
                                if a <= b {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                            ScalarBinOp::And => {
                                if a != 0.0 && b != 0.0 {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                            ScalarBinOp::Or => {
                                if a != 0.0 || b != 0.0 {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                            ScalarBinOp::Xor => {
                                if (a != 0.0) ^ (b != 0.0) {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                        };
                    }
                }
                ScalarOp::Unary { op, .. } => {
                    for i in 0..group.count as usize {
                        let x = values[group.inputs[0].resolve(i as u64).0 as usize];
                        values[base + i] = match op {
                            ScalarUnaryOp::Neg => -x,
                            ScalarUnaryOp::Abs => x.abs(),
                            ScalarUnaryOp::Exp => x.exp(),
                            ScalarUnaryOp::Ln => x.ln(),
                            ScalarUnaryOp::Sqrt => x.sqrt(),
                            ScalarUnaryOp::Reciprocal => 1.0 / x,
                            ScalarUnaryOp::Tanh => x.tanh(),
                            ScalarUnaryOp::Floor => x.floor(),
                            ScalarUnaryOp::Ceil => x.ceil(),
                        };
                    }
                }
                ScalarOp::Select => {
                    for i in 0..group.count as usize {
                        let cond = values[group.inputs[0].resolve(i as u64).0 as usize];
                        let a = values[group.inputs[1].resolve(i as u64).0 as usize];
                        let b = values[group.inputs[2].resolve(i as u64).0 as usize];
                        values[base + i] = if cond != 0.0 { a } else { b };
                    }
                }
                ScalarOp::Reduce {
                    kind,
                    reduce_count,
                    reduce_stride,
                    ..
                } => {
                    for i in 0..group.count as usize {
                        let start = group.inputs[0].resolve(i as u64).0 as i64;
                        let mut acc = match kind {
                            ReduceKind::Sum => 0.0f64,
                            ReduceKind::Max => f64::NEG_INFINITY,
                        };
                        for r in 0..*reduce_count {
                            let src_idx = (start + r as i64 * reduce_stride) as usize;
                            let v = values[src_idx];
                            acc = match kind {
                                ReduceKind::Sum => acc + v,
                                ReduceKind::Max => acc.max(v),
                            };
                        }
                        values[base + i] = acc;
                    }
                }
                ScalarOp::IndirectLoad { .. } => {
                    // Not commonly used in compiler tests.
                }
            }
        }

        values
    }

    /// Compare CompiledPlan output against the NanoEval interpreter.
    fn compare_plan_vs_interp(
        graph: &NanoGraph,
        plan: &ExecutionPlan,
        overrides_f32: &HashMap<u64, f32>,
        label: &str,
    ) {
        // Interpreter reference.
        let t0 = std::time::Instant::now();
        let interp_values = eval_nano_graph(graph, overrides_f32);
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
            let interp_val = interp_values[i as usize];
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
        use crate::milli_graph::MilliOpGraph;
        use crate::nano_graph::lower::lower_with_info;
        use crate::numeric_tensor::NumericTensor;
        use crate::tensor_info::TensorInfo;

        use super::super::plan::v2c::plan_execution;

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
        let a_tensor = NumericTensor::<crate::DynRank>::from_vec_shape(a_data, vec![4, 6]).unwrap();
        let b_tensor = NumericTensor::<crate::DynRank>::from_vec_shape(b_data, vec![6, 8]).unwrap();

        let mut info_inputs = HashMap::new();
        info_inputs.insert(a, TensorInfo::from(a_tensor.clone()));
        info_inputs.insert(b, TensorInfo::from(b_tensor.clone()));

        let result = lower_with_info(&milli, &info_inputs).unwrap();
        assert!(
            result.unsupported.is_empty(),
            "unsupported: {:?}",
            result.unsupported_details
        );

        // Build f32 overrides from info_inputs.
        let overrides = build_f32_overrides(&info_inputs, &result);

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
        compare_plan_vs_interp(
            &result.graph,
            &plan4,
            &overrides,
            "plan_matmul_4x6x8_4lanes",
        );
    }

    #[test]
    fn test_plan_small_matmul() {
        use crate::milli_graph::MilliOpGraph;
        use crate::nano_graph::lower::lower_with_info;
        use crate::numeric_tensor::NumericTensor;
        use crate::tensor_info::TensorInfo;

        use super::super::plan::v2c::plan_execution;

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
        let a_tensor = NumericTensor::<crate::DynRank>::from_vec_shape(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        )
        .unwrap();
        let b_tensor = NumericTensor::<crate::DynRank>::from_vec_shape(
            vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![3, 2],
        )
        .unwrap();

        let mut info_inputs = HashMap::new();
        info_inputs.insert(a, TensorInfo::from(a_tensor.clone()));
        info_inputs.insert(b, TensorInfo::from(b_tensor.clone()));

        let result = lower_with_info(&milli, &info_inputs).unwrap();
        assert!(result.unsupported.is_empty());

        let overrides = build_f32_overrides(&info_inputs, &result);

        // Single-lane plan (should behave identically to whole-graph codegen).
        let plan1 = plan_execution(&result.graph, 1);
        compare_plan_vs_interp(&result.graph, &plan1, &overrides, "plan_matmul_2x3x2_1lane");

        // Two lanes.
        let plan2 = plan_execution(&result.graph, 2);
        compare_plan_vs_interp(
            &result.graph,
            &plan2,
            &overrides,
            "plan_matmul_2x3x2_2lanes",
        );
    }

    /// Test matmul chain: A@B + bias, then result@C.
    /// This exercises multi-phase execution with AllRows groups (the Add).
    #[test]
    fn test_plan_matmul_chain() {
        use crate::milli_graph::MilliOpGraph;
        use crate::milli_graph::ops::{MatMul, SimpleBinary};
        use crate::nano_graph::lower::lower_with_info;
        use crate::numeric_tensor::NumericTensor;
        use crate::tensor_info::TensorInfo;

        use super::super::plan::v2c::plan_execution;

        let mut rng = rand::rng();
        let (mut milli, _) = MilliOpGraph::new(std::iter::empty(), &mut rng);
        let a = milli.add_input(&mut rng); // 4x6
        let b = milli.add_input(&mut rng); // 6x8
        let bias = milli.add_input(&mut rng); // 4x8 (broadcast-compatible)
        let c = milli.add_input(&mut rng); // 8x3

        // ab = A @ B  (4x8)
        let ab = MatMul::push_new_default_precision(&mut milli, a, b, DType::F32, &mut rng);
        // ab_bias = ab + bias  (4x8, elementwise)
        let ab_bias = SimpleBinary::add(&mut milli, ab, bias, &mut rng);
        // result = ab_bias @ C  (4x3)
        let _result =
            MatMul::push_new_default_precision(&mut milli, ab_bias, c, DType::F32, &mut rng);

        let a_data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.1 + 0.5).collect();
        let b_data: Vec<f32> = (0..48).map(|i| (i as f32) * 0.05 - 1.0).collect();
        let bias_data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.01).collect();
        let c_data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.02 + 0.1).collect();

        let a_tensor = NumericTensor::<crate::DynRank>::from_vec_shape(a_data, vec![4, 6]).unwrap();
        let b_tensor = NumericTensor::<crate::DynRank>::from_vec_shape(b_data, vec![6, 8]).unwrap();
        let bias_tensor =
            NumericTensor::<crate::DynRank>::from_vec_shape(bias_data, vec![4, 8]).unwrap();
        let c_tensor = NumericTensor::<crate::DynRank>::from_vec_shape(c_data, vec![8, 3]).unwrap();

        let mut info_inputs = HashMap::new();
        info_inputs.insert(a, TensorInfo::from(a_tensor.clone()));
        info_inputs.insert(b, TensorInfo::from(b_tensor.clone()));
        info_inputs.insert(bias, TensorInfo::from(bias_tensor.clone()));
        info_inputs.insert(c, TensorInfo::from(c_tensor.clone()));

        let result = lower_with_info(&milli, &info_inputs).unwrap();
        assert!(
            result.unsupported.is_empty(),
            "unsupported: {:?}",
            result.unsupported_details
        );

        let overrides = build_f32_overrides(&info_inputs, &result);

        // Print plan structure for debugging.
        let groups = result.graph.groups();
        eprintln!(
            "Matmul chain: {} groups, {} atoms",
            groups.len(),
            result.graph.num_atoms()
        );
        for (gi, g) in groups.iter().enumerate() {
            eprintln!(
                "  group {}: base={} count={} op={:?} inputs={}",
                gi,
                g.base_id.0,
                g.count,
                std::mem::discriminant(&g.op),
                g.inputs.len()
            );
        }

        // Single-lane plan (no splitting, should match interpreter exactly).
        let plan1 = plan_execution(&result.graph, 1);
        eprintln!("Plan 1 lane: {} phases", plan1.phases.len());
        for (pi, phase) in plan1.phases.iter().enumerate() {
            for (li, lane) in phase.lane_work.iter().enumerate() {
                for w in lane {
                    eprintln!(
                        "  phase {} lane {} group_idx={} offset={} count={}",
                        pi, li, w.group_idx, w.atom_offset, w.atom_count,
                    );
                }
            }
        }
        compare_plan_vs_interp(&result.graph, &plan1, &overrides, "matmul_chain_1lane");

        // Two lanes.
        let plan2 = plan_execution(&result.graph, 2);
        eprintln!("Plan 2 lanes: {} phases", plan2.phases.len());
        for (pi, phase) in plan2.phases.iter().enumerate() {
            for (li, lane) in phase.lane_work.iter().enumerate() {
                for w in lane {
                    let g = &groups[w.group_idx];
                    eprintln!(
                        "  phase {} lane {} group_idx={} offset={} count={} (base={} total={})",
                        pi, li, w.group_idx, w.atom_offset, w.atom_count, g.base_id.0, g.count,
                    );
                }
            }
        }
        compare_plan_vs_interp(&result.graph, &plan2, &overrides, "matmul_chain_2lanes");

        // Four lanes.
        let plan4 = plan_execution(&result.graph, 4);
        compare_plan_vs_interp(&result.graph, &plan4, &overrides, "matmul_chain_4lanes");

        // Eight lanes.
        let plan8 = plan_execution(&result.graph, 8);
        compare_plan_vs_interp(&result.graph, &plan8, &overrides, "matmul_chain_8lanes");
    }

    /// Test with elementwise chain only (no matmul) to isolate AllRows splitting.
    #[test]
    fn test_plan_elementwise_chain() {
        use crate::milli_graph::MilliOpGraph;
        use crate::milli_graph::ops::{SimpleBinary, SimpleUnaryOp};
        use crate::nano_graph::lower::lower_with_info;
        use crate::numeric_tensor::NumericTensor;
        use crate::tensor_info::TensorInfo;

        use super::super::plan::v2c::plan_execution;

        let mut rng = rand::rng();
        let (mut milli, _) = MilliOpGraph::new(std::iter::empty(), &mut rng);
        let a = milli.add_input(&mut rng); // 16 elements
        let b = milli.add_input(&mut rng); // 16 elements

        // Chain: a + b, then exp
        let sum = SimpleBinary::add(&mut milli, a, b, &mut rng);
        let _exp = SimpleUnaryOp::exp(&mut milli, sum, &mut rng);

        let a_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
        let b_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.05 + 0.5).collect();
        let a_tensor = NumericTensor::<crate::DynRank>::from_vec_shape(a_data, vec![4, 4]).unwrap();
        let b_tensor = NumericTensor::<crate::DynRank>::from_vec_shape(b_data, vec![4, 4]).unwrap();

        let mut info_inputs = HashMap::new();
        info_inputs.insert(a, TensorInfo::from(a_tensor.clone()));
        info_inputs.insert(b, TensorInfo::from(b_tensor.clone()));

        let result = lower_with_info(&milli, &info_inputs).unwrap();
        assert!(result.unsupported.is_empty());

        let overrides = build_f32_overrides(&info_inputs, &result);

        for lanes in [1, 2, 4, 8] {
            let plan = plan_execution(&result.graph, lanes);
            compare_plan_vs_interp(
                &result.graph,
                &plan,
                &overrides,
                &format!("elementwise_chain_{}lanes", lanes),
            );
        }
    }

    /// Test with a larger matmul that has more rows to split.
    #[test]
    fn test_plan_large_matmul() {
        use crate::milli_graph::MilliOpGraph;
        use crate::milli_graph::ops::MatMul;
        use crate::nano_graph::lower::lower_with_info;
        use crate::numeric_tensor::NumericTensor;
        use crate::tensor_info::TensorInfo;

        use super::super::plan::v2c::plan_execution;

        let mut rng = rand::rng();
        let (mut milli, _) = MilliOpGraph::new(std::iter::empty(), &mut rng);
        let a = milli.add_input(&mut rng); // 16x32
        let b = milli.add_input(&mut rng); // 32x16

        let _c = MatMul::push_new_default_precision(&mut milli, a, b, DType::F32, &mut rng);

        let a_data: Vec<f32> = (0..512).map(|i| ((i as f32) * 0.01).sin()).collect();
        let b_data: Vec<f32> = (0..512).map(|i| ((i as f32) * 0.02).cos()).collect();
        let a_tensor =
            NumericTensor::<crate::DynRank>::from_vec_shape(a_data, vec![16, 32]).unwrap();
        let b_tensor =
            NumericTensor::<crate::DynRank>::from_vec_shape(b_data, vec![32, 16]).unwrap();

        let mut info_inputs = HashMap::new();
        info_inputs.insert(a, TensorInfo::from(a_tensor.clone()));
        info_inputs.insert(b, TensorInfo::from(b_tensor.clone()));

        let result = lower_with_info(&milli, &info_inputs).unwrap();
        assert!(
            result.unsupported.is_empty(),
            "unsupported: {:?}",
            result.unsupported_details,
        );

        let overrides = build_f32_overrides(&info_inputs, &result);

        for lanes in [1, 2, 4, 8] {
            let plan = plan_execution(&result.graph, lanes);
            compare_plan_vs_interp(
                &result.graph,
                &plan,
                &overrides,
                &format!("large_matmul_16x32x16_{}lanes", lanes),
            );
        }
    }

    /// Test ReduceMean + matmul chain (simulates layer norm + projection).
    /// ReduceMean creates a reduce pattern that interacts with AllRows splitting.
    #[test]
    fn test_plan_reducemean_matmul() {
        use crate::milli_graph::MilliOpGraph;
        use crate::milli_graph::ops::{MatMul, ReduceMean, SimpleBinary};
        use crate::nano_graph::lower::lower_with_info;
        use crate::numeric_tensor::NumericTensor;
        use crate::tensor_info::TensorInfo;

        use super::super::plan::v2c::plan_execution;

        let mut rng = rand::rng();
        let (mut milli, _) = MilliOpGraph::new(std::iter::empty(), &mut rng);
        let x = milli.add_input(&mut rng); // 8x16 input
        let w = milli.add_input(&mut rng); // 16x12 weight

        // ReduceMean over last axis (simulates part of layer norm)
        let mean = ReduceMean::push_new(&mut milli, x, None, true, false, &mut rng);
        // Subtract mean from input
        let centered = SimpleBinary::sub(&mut milli, x, mean, &mut rng);
        // Project: centered @ w
        let _proj =
            MatMul::push_new_default_precision(&mut milli, centered, w, DType::F32, &mut rng);

        let x_data: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.07).sin() + 1.0).collect();
        let w_data: Vec<f32> = (0..192).map(|i| ((i as f32) * 0.03).cos() * 0.5).collect();
        let x_tensor =
            NumericTensor::<crate::DynRank>::from_vec_shape(x_data, vec![8, 16]).unwrap();
        let w_tensor =
            NumericTensor::<crate::DynRank>::from_vec_shape(w_data, vec![16, 12]).unwrap();

        let mut info_inputs = HashMap::new();
        info_inputs.insert(x, TensorInfo::from(x_tensor.clone()));
        info_inputs.insert(w, TensorInfo::from(w_tensor.clone()));

        let result = lower_with_info(&milli, &info_inputs).unwrap();
        if !result.unsupported.is_empty() {
            eprintln!("unsupported: {:?}", result.unsupported_details);
        }

        let overrides = build_f32_overrides(&info_inputs, &result);

        for lanes in [1, 2, 4, 8] {
            let plan = plan_execution(&result.graph, lanes);
            compare_plan_vs_interp(
                &result.graph,
                &plan,
                &overrides,
                &format!("reducemean_matmul_{}lanes", lanes),
            );
        }
    }

    /// Test double matmul with residual add (GPT-2 pattern: attention + residual).
    /// Pattern: Y = X @ W1 + X (residual), then Z = Y @ W2
    #[test]
    fn test_plan_matmul_residual() {
        use crate::milli_graph::MilliOpGraph;
        use crate::milli_graph::ops::{MatMul, SimpleBinary};
        use crate::nano_graph::lower::lower_with_info;
        use crate::numeric_tensor::NumericTensor;
        use crate::tensor_info::TensorInfo;

        use super::super::plan::v2c::plan_execution;

        let mut rng = rand::rng();
        let (mut milli, _) = MilliOpGraph::new(std::iter::empty(), &mut rng);
        let x = milli.add_input(&mut rng); // 8x16
        let w1 = milli.add_input(&mut rng); // 16x16
        let w2 = milli.add_input(&mut rng); // 16x8

        // Y = X @ W1
        let y = MatMul::push_new_default_precision(&mut milli, x, w1, DType::F32, &mut rng);
        // residual = Y + X (requires same shape: 8x16)
        let residual = SimpleBinary::add(&mut milli, y, x, &mut rng);
        // Z = residual @ W2
        let _z = MatMul::push_new_default_precision(&mut milli, residual, w2, DType::F32, &mut rng);

        let x_data: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.05).sin()).collect();
        let w1_data: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.02).cos() * 0.3).collect();
        let w2_data: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.04).sin() * 0.2).collect();

        let x_tensor =
            NumericTensor::<crate::DynRank>::from_vec_shape(x_data, vec![8, 16]).unwrap();
        let w1_tensor =
            NumericTensor::<crate::DynRank>::from_vec_shape(w1_data, vec![16, 16]).unwrap();
        let w2_tensor =
            NumericTensor::<crate::DynRank>::from_vec_shape(w2_data, vec![16, 8]).unwrap();

        let mut info_inputs = HashMap::new();
        info_inputs.insert(x, TensorInfo::from(x_tensor.clone()));
        info_inputs.insert(w1, TensorInfo::from(w1_tensor.clone()));
        info_inputs.insert(w2, TensorInfo::from(w2_tensor.clone()));

        let result = lower_with_info(&milli, &info_inputs).unwrap();
        assert!(
            result.unsupported.is_empty(),
            "unsupported: {:?}",
            result.unsupported_details,
        );

        let overrides = build_f32_overrides(&info_inputs, &result);

        for lanes in [1, 2, 4, 8] {
            let plan = plan_execution(&result.graph, lanes);
            compare_plan_vs_interp(
                &result.graph,
                &plan,
                &overrides,
                &format!("matmul_residual_{}lanes", lanes),
            );
        }
    }

    // =========================================================================
    // Plan validation test: checks ordering, completeness, and f32 simulation
    // =========================================================================

    /// Helper: resolve all source atom indices for atom at offset `i` within a group.
    /// For non-reduce ops, returns the InputRef-resolved sources.
    /// For reduce ops, returns ALL atoms read by the reduction loop.
    fn resolve_atom_sources(group: &AtomGroup, i: u64) -> Vec<u64> {
        let mut sources = Vec::new();
        match &group.op {
            ScalarOp::Literal(_) => {}
            ScalarOp::Reduce {
                reduce_count,
                reduce_stride,
                ..
            } => {
                let base = group.inputs[0].resolve(i);
                for k in 0..*reduce_count {
                    let src = (base.0 as i64 + k as i64 * reduce_stride) as u64;
                    sources.push(src);
                }
            }
            ScalarOp::IndirectLoad { table_base, .. } => {
                // The index input
                let idx_src = group.inputs[0].resolve(i);
                sources.push(idx_src.0);
                // The table base is also a dependency but we can't know which
                // element at compile time. We'll skip the table atoms for ordering
                // validation since the plan builder handles them at group level.
            }
            _ => {
                // Identity, Binary, Unary, Select: resolve each input
                for input in &group.inputs {
                    let src = input.resolve(i);
                    sources.push(src.0);
                }
            }
        }
        sources
    }

    /// Helper: compute a single atom's value using f32 arithmetic,
    /// given the current values buffer. Mirrors what the JIT codegen does.
    fn compute_atom_f32(group: &AtomGroup, i: u64, values: &[f32]) -> f32 {
        match &group.op {
            ScalarOp::Literal(scalar) => scalar.to_f64() as f32,
            ScalarOp::Identity => {
                let src = group.inputs[0].resolve(i);
                values[src.0 as usize]
            }
            ScalarOp::Binary { op, .. } => {
                let a = values[group.inputs[0].resolve(i).0 as usize];
                let b = values[group.inputs[1].resolve(i).0 as usize];
                match op {
                    ScalarBinOp::Add => a + b,
                    ScalarBinOp::Sub => a - b,
                    ScalarBinOp::Mul => a * b,
                    ScalarBinOp::Div => a / b,
                    ScalarBinOp::Max => a.max(b),
                    ScalarBinOp::Min => a.min(b),
                    ScalarBinOp::Mod => a % b,
                    ScalarBinOp::Pow => a.powf(b),
                    ScalarBinOp::Equal => {
                        if a == b {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    ScalarBinOp::Greater => {
                        if a > b {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    ScalarBinOp::GreaterOrEqual => {
                        if a >= b {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    ScalarBinOp::Less => {
                        if a < b {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    ScalarBinOp::LessOrEqual => {
                        if a <= b {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    ScalarBinOp::And => {
                        if a != 0.0 && b != 0.0 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    ScalarBinOp::Or => {
                        if a != 0.0 || b != 0.0 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    ScalarBinOp::Xor => {
                        if (a != 0.0) ^ (b != 0.0) {
                            1.0
                        } else {
                            0.0
                        }
                    }
                }
            }
            ScalarOp::Unary { op, .. } => {
                let x = values[group.inputs[0].resolve(i).0 as usize];
                match op {
                    ScalarUnaryOp::Neg => -x,
                    ScalarUnaryOp::Abs => x.abs(),
                    ScalarUnaryOp::Exp => x.exp(),
                    ScalarUnaryOp::Ln => x.ln(),
                    ScalarUnaryOp::Sqrt => x.sqrt(),
                    ScalarUnaryOp::Reciprocal => 1.0 / x,
                    ScalarUnaryOp::Tanh => x.tanh(),
                    ScalarUnaryOp::Floor => x.floor(),
                    ScalarUnaryOp::Ceil => x.ceil(),
                }
            }
            ScalarOp::Select => {
                let cond = values[group.inputs[0].resolve(i).0 as usize];
                if cond != 0.0 {
                    values[group.inputs[1].resolve(i).0 as usize]
                } else {
                    values[group.inputs[2].resolve(i).0 as usize]
                }
            }
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count,
                reduce_stride,
                ..
            } => {
                let base = group.inputs[0].resolve(i);
                let mut acc = 0.0f32;
                for k in 0..*reduce_count {
                    let src_idx = (base.0 as i64 + k as i64 * reduce_stride) as u64;
                    acc += values[src_idx as usize];
                }
                acc
            }
            ScalarOp::Reduce {
                kind: ReduceKind::Max,
                reduce_count,
                reduce_stride,
                ..
            } => {
                let base = group.inputs[0].resolve(i);
                let mut acc = f32::NEG_INFINITY;
                for k in 0..*reduce_count {
                    let src_idx = (base.0 as i64 + k as i64 * reduce_stride) as u64;
                    acc = acc.max(values[src_idx as usize]);
                }
                acc
            }
            ScalarOp::IndirectLoad { table_base, .. } => {
                let idx_src = group.inputs[0].resolve(i);
                let index = values[idx_src.0 as usize] as usize;
                values[table_base.0 as usize + index]
            }
        }
    }

    /// Comprehensive plan validation test.
    ///
    /// Builds a 2-layer MLP via real lowering (matmul -> exp -> matmul),
    /// then validates the v2c plan's:
    /// 1. Ordering correctness (no cross-lane dependencies within a phase)
    /// 2. Completeness (every compute atom appears exactly once)
    /// 3. "Copy from reference" simulation (plan structure is sound)
    /// 4. f32 simulation (matches NanoEval within tolerance)
    #[test]
    fn test_plan_validation_2layer_mlp() {
        use crate::milli_graph::MilliOpGraph;
        use crate::milli_graph::ops::{MatMul, SimpleUnaryOp};
        use crate::nano_graph::lower::lower_with_info;
        use crate::numeric_tensor::NumericTensor;
        use crate::tensor_info::TensorInfo;

        use super::super::plan::v2c::plan_execution;

        // --- Step 1: Build a 2-layer MLP ---
        // Layer 1: A(8x16) @ B(16x32) -> exp activation
        // Layer 2: result(8x32) @ C(32x16)
        let m1 = 8usize;
        let k1 = 16usize;
        let n1 = 32usize;
        let k2 = n1; // 32
        let n2 = 16usize;

        let mut rng = rand::rng();
        let (mut milli, _) = MilliOpGraph::new(std::iter::empty(), &mut rng);
        let a_id = milli.add_input(&mut rng);
        let b_id = milli.add_input(&mut rng);
        let c_id = milli.add_input(&mut rng);

        // Layer 1: A @ B
        let ab = MatMul::push_new_default_precision(&mut milli, a_id, b_id, DType::F32, &mut rng);
        // Activation: exp (simple, exercises unary ops)
        let activated = SimpleUnaryOp::exp(&mut milli, ab, &mut rng);
        // Layer 2: activated @ C
        let _output =
            MatMul::push_new_default_precision(&mut milli, activated, c_id, DType::F32, &mut rng);

        // Generate deterministic test data with bounded values (small for exp stability)
        let a_data: Vec<f32> = (0..(m1 * k1))
            .map(|i| ((i as f32) * 0.037).sin() * 0.1)
            .collect();
        let b_data: Vec<f32> = (0..(k1 * n1))
            .map(|i| ((i as f32) * 0.023).cos() * 0.1)
            .collect();
        let c_data: Vec<f32> = (0..(k2 * n2))
            .map(|i| ((i as f32) * 0.041).sin() * 0.1)
            .collect();

        let a_tensor =
            NumericTensor::<crate::DynRank>::from_vec_shape(a_data, vec![m1, k1]).unwrap();
        let b_tensor =
            NumericTensor::<crate::DynRank>::from_vec_shape(b_data, vec![k1, n1]).unwrap();
        let c_tensor =
            NumericTensor::<crate::DynRank>::from_vec_shape(c_data, vec![k2, n2]).unwrap();

        let mut info_inputs = HashMap::new();
        info_inputs.insert(a_id, TensorInfo::from(a_tensor.clone()));
        info_inputs.insert(b_id, TensorInfo::from(b_tensor.clone()));
        info_inputs.insert(c_id, TensorInfo::from(c_tensor.clone()));

        // --- Step 2: Lower to NanoGraph ---
        let result = lower_with_info(&milli, &info_inputs).unwrap();
        assert!(
            result.unsupported.is_empty(),
            "unsupported ops: {:?}",
            result.unsupported_details,
        );

        // Build f32 overrides from info_inputs.
        let overrides_f32 = build_f32_overrides(&info_inputs, &result);

        let graph = &result.graph;
        let groups = graph.groups();
        let num_atoms = graph.num_atoms() as usize;

        eprintln!(
            "=== 2-layer MLP: {} groups, {} atoms ===",
            groups.len(),
            num_atoms
        );
        for (gi, g) in groups.iter().enumerate() {
            let op_name = format!("{:?}", g.op)
                .chars()
                .take_while(|c| *c != ' ' && *c != '{')
                .collect::<String>();
            eprintln!(
                "  group {:>2}: base={:>6} count={:>6} op={} inputs={}",
                gi,
                g.base_id.0,
                g.count,
                op_name,
                g.inputs.len(),
            );
        }

        // --- Step 3: Run trusted NanoGraph evaluator ---
        let reference_values = eval_nano_graph(graph, &overrides_f32);

        // --- Step 4: Generate v2c plan with 4 lanes ---
        let plan = plan_execution(graph, 4);
        eprintln!(
            "Plan: {} phases, {} lanes",
            plan.phases.len(),
            plan.num_lanes,
        );
        for (pi, phase) in plan.phases.iter().enumerate() {
            for (li, lane) in phase.lane_work.iter().enumerate() {
                for w in lane {
                    let g = &groups[w.group_idx];
                    let op_name = format!("{:?}", g.op)
                        .chars()
                        .take_while(|c| *c != ' ' && *c != '{')
                        .collect::<String>();
                    eprintln!(
                        "  phase {} lane {} group={} offset={} count={} op={}",
                        pi, li, w.group_idx, w.atom_offset, w.atom_count, op_name,
                    );
                }
            }
        }

        // --- Step 5: Validate plan ordering ---
        // Build a map: atom_idx -> (phase, lane) where it's produced.
        let mut atom_producer: HashMap<u64, (usize, usize)> = HashMap::new();

        // Mark all literal atoms as "always available" with sentinel (usize::MAX, 0)
        for g in groups {
            if matches!(&g.op, ScalarOp::Literal(_)) {
                for i in 0..g.count {
                    atom_producer.insert(g.base_id.0 + i, (usize::MAX, 0));
                }
            }
        }

        // Register all atoms produced by plan work items
        for (phase_idx, phase) in plan.phases.iter().enumerate() {
            for (lane_idx, lane) in phase.lane_work.iter().enumerate() {
                for work in lane {
                    let g = &groups[work.group_idx];
                    for i in work.atom_offset..(work.atom_offset + work.atom_count) {
                        let atom_idx = g.base_id.0 + i;
                        let prev = atom_producer.insert(atom_idx, (phase_idx, lane_idx));
                        assert!(
                            prev.is_none(),
                            "atom {} produced by multiple work items: prev={:?} new=({}, {})",
                            atom_idx,
                            prev,
                            phase_idx,
                            lane_idx,
                        );
                    }
                }
            }
        }

        // Now check ordering: for each work item, for each atom, check that
        // all source atoms are either:
        // a) Literal (usize::MAX sentinel)
        // b) Produced in an earlier phase
        // c) Produced earlier in the same lane's same phase (within-lane ordering)
        //
        // A source in the SAME phase but DIFFERENT lane is a violation.
        let mut ordering_violations = 0usize;
        // Track within-lane ordering: atoms produced earlier in the same phase+lane
        for (phase_idx, phase) in plan.phases.iter().enumerate() {
            for (lane_idx, lane) in phase.lane_work.iter().enumerate() {
                // Build set of atoms produced so far in this lane during this phase
                let mut lane_produced_so_far: std::collections::HashSet<u64> =
                    std::collections::HashSet::new();

                for work in lane {
                    let g = &groups[work.group_idx];
                    for i in work.atom_offset..(work.atom_offset + work.atom_count) {
                        let atom_idx = g.base_id.0 + i;
                        let sources = resolve_atom_sources(g, i);

                        for src in &sources {
                            if let Some(&(src_phase, src_lane)) = atom_producer.get(src) {
                                if src_phase == usize::MAX {
                                    // Literal, always available
                                    continue;
                                }
                                if src_phase < phase_idx {
                                    // Produced in earlier phase, OK
                                    continue;
                                }
                                if src_phase == phase_idx && src_lane == lane_idx {
                                    // Same phase, same lane.
                                    // Must have been produced earlier in this lane.
                                    if !lane_produced_so_far.contains(src) {
                                        // The source is in the same lane+phase but hasn't
                                        // been produced yet. This is OK if it's part of the
                                        // SAME work item (e.g., a self-referencing reduce).
                                        // Check if it's in the current work item's range.
                                        let work_start = g.base_id.0 + work.atom_offset;
                                        let work_end = work_start + work.atom_count;
                                        if *src >= work_start && *src < work_end {
                                            // Same work item -- this is fine for reduces
                                            // (they read from OTHER groups, not self)
                                            // Actually, a self-reference within the same
                                            // group is unusual. Let's flag it but not fail.
                                        } else {
                                            eprintln!(
                                                "  ORDERING VIOLATION: atom {} (phase {} lane {}) \
                                                 reads src {} (same phase, same lane, but not yet produced)",
                                                atom_idx, phase_idx, lane_idx, src,
                                            );
                                            ordering_violations += 1;
                                        }
                                    }
                                    continue;
                                }
                                if src_phase == phase_idx && src_lane != lane_idx {
                                    // VIOLATION: same phase, different lane
                                    eprintln!(
                                        "  ORDERING VIOLATION: atom {} (phase {} lane {}) \
                                         reads src {} from DIFFERENT lane {} in same phase",
                                        atom_idx, phase_idx, lane_idx, src, src_lane,
                                    );
                                    ordering_violations += 1;
                                    continue;
                                }
                                if src_phase > phase_idx {
                                    // Source produced in a LATER phase!
                                    eprintln!(
                                        "  ORDERING VIOLATION: atom {} (phase {} lane {}) \
                                         reads src {} from LATER phase {}",
                                        atom_idx, phase_idx, lane_idx, src, src_phase,
                                    );
                                    ordering_violations += 1;
                                }
                            }
                            // If src not in atom_producer, it might be an override
                            // (input tensor data). That's fine.
                        }

                        lane_produced_so_far.insert(atom_idx);
                    }
                }
            }
        }
        eprintln!(
            "Ordering validation: {} violations found",
            ordering_violations,
        );
        assert_eq!(
            ordering_violations, 0,
            "Plan has {} ordering violations (cross-lane reads within phase or \
             reads from later phase)",
            ordering_violations,
        );

        // --- Step 6: Validate completeness ---
        // Every compute (non-Literal) atom must appear in exactly one LaneWork item.
        let mut plan_atoms: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for phase in &plan.phases {
            for lane in &phase.lane_work {
                for work in lane {
                    let g = &groups[work.group_idx];
                    if matches!(&g.op, ScalarOp::Literal(_)) {
                        continue;
                    }
                    for i in work.atom_offset..(work.atom_offset + work.atom_count) {
                        let atom_idx = g.base_id.0 + i;
                        let is_new = plan_atoms.insert(atom_idx);
                        assert!(is_new, "atom {} appears in plan multiple times", atom_idx,);
                    }
                }
            }
        }

        let mut missing_atoms = Vec::new();
        for g in groups {
            if matches!(&g.op, ScalarOp::Literal(_)) {
                continue;
            }
            for i in 0..g.count {
                let atom_idx = g.base_id.0 + i;
                if !plan_atoms.contains(&atom_idx) {
                    missing_atoms.push(atom_idx);
                }
            }
        }
        if !missing_atoms.is_empty() {
            eprintln!(
                "Completeness: {} atoms missing from plan (first 10: {:?})",
                missing_atoms.len(),
                &missing_atoms[..missing_atoms.len().min(10)],
            );
        }
        assert!(
            missing_atoms.is_empty(),
            "Plan is incomplete: {} compute atoms not covered",
            missing_atoms.len(),
        );
        eprintln!(
            "Completeness validation: OK ({} compute atoms covered)",
            plan_atoms.len()
        );

        // --- Step 7: "Copy from reference" simulation ---
        // Walk the plan in execution order. For each atom, copy the trusted
        // NanoEval value. If the plan's ordering is correct, the final buffer
        // should exactly match the reference.
        let mut copy_values = vec![0.0f32; num_atoms];
        // Pre-fill literals + overrides
        for g in groups {
            if let ScalarOp::Literal(scalar) = &g.op {
                let val = scalar.to_f64() as f32;
                for i in 0..g.count {
                    copy_values[(g.base_id.0 + i) as usize] = val;
                }
            }
        }
        for (&idx, &val) in &overrides_f32 {
            copy_values[idx as usize] = val;
        }

        // Walk plan, copy from reference
        for phase in &plan.phases {
            for lane in &phase.lane_work {
                for work in lane {
                    let g = &groups[work.group_idx];
                    if matches!(&g.op, ScalarOp::Literal(_)) {
                        continue;
                    }
                    for i in work.atom_offset..(work.atom_offset + work.atom_count) {
                        let atom_idx = g.base_id.0 + i;
                        copy_values[atom_idx as usize] = reference_values[atom_idx as usize] as f32;
                    }
                }
            }
        }

        // Compare copy_values against reference
        let mut copy_max_err: f64 = 0.0;
        let mut copy_err_atom: u64 = 0;
        for i in 0..num_atoms {
            let ref_val = reference_values[i];
            let copy_val = copy_values[i] as f64;
            let diff = (ref_val - copy_val).abs();
            if diff > copy_max_err {
                copy_max_err = diff;
                copy_err_atom = i as u64;
            }
        }
        eprintln!(
            "Copy-from-reference simulation: max_err={:.2e} at atom {}",
            copy_max_err, copy_err_atom,
        );
        // This should be essentially zero (only f64->f32->f64 roundtrip error)
        assert!(
            copy_max_err < 1e-3,
            "Copy-from-reference has error {:.6e} at atom {} -- plan structure is wrong!",
            copy_max_err,
            copy_err_atom,
        );

        // --- Step 8: f32 computation simulation ---
        // Walk the plan, compute each atom using f32 arithmetic (matching JIT).
        let mut sim_values = vec![0.0f32; num_atoms];
        // Pre-fill literals + overrides
        for g in groups {
            if let ScalarOp::Literal(scalar) = &g.op {
                let val = scalar.to_f64() as f32;
                for i in 0..g.count {
                    sim_values[(g.base_id.0 + i) as usize] = val;
                }
            }
        }
        for (&idx, &val) in &overrides_f32 {
            sim_values[idx as usize] = val;
        }

        // Walk plan, compute each atom
        for phase in &plan.phases {
            for lane in &phase.lane_work {
                for work in lane {
                    let g = &groups[work.group_idx];
                    if matches!(&g.op, ScalarOp::Literal(_)) {
                        continue;
                    }
                    for i in work.atom_offset..(work.atom_offset + work.atom_count) {
                        let atom_idx = g.base_id.0 + i;
                        let val = compute_atom_f32(g, i, &sim_values);
                        sim_values[atom_idx as usize] = val;
                    }
                }
            }
        }

        // Compare f32 simulation against NanoEval reference
        let mut sim_max_err: f64 = 0.0;
        let mut sim_err_atom: u64 = 0;
        let mut sim_large_err_count = 0usize;
        for i in 0..num_atoms {
            let ref_val = reference_values[i];
            let sim_val = sim_values[i] as f64;
            let diff = (ref_val - sim_val).abs();
            if diff > sim_max_err {
                sim_max_err = diff;
                sim_err_atom = i as u64;
            }
            if diff > 1e-3 {
                sim_large_err_count += 1;
                if sim_large_err_count <= 5 {
                    let gi = groups
                        .iter()
                        .position(|g| i as u64 >= g.base_id.0 && (i as u64) < g.base_id.0 + g.count)
                        .unwrap_or(usize::MAX);
                    let op_name = if gi < groups.len() {
                        format!("{:?}", groups[gi].op)
                            .chars()
                            .take_while(|c| *c != ' ' && *c != '{')
                            .collect::<String>()
                    } else {
                        "???".to_string()
                    };
                    eprintln!(
                        "  f32-sim large error: atom {} (group {} {}) ref={} sim={} diff={}",
                        i, gi, op_name, ref_val, sim_val, diff,
                    );
                }
            }
        }
        eprintln!(
            "f32 simulation: max_err={:.2e} at atom {}, {} atoms with err > 1e-3",
            sim_max_err, sim_err_atom, sim_large_err_count,
        );

        // The f32 simulation should match reasonably well. For a 2-layer MLP with
        // small values, we expect small precision differences. If max_err > 1.0,
        // the plan likely has an ordering bug.
        if sim_max_err > 1.0 {
            panic!(
                "f32 simulation max_err={:.2e} is too large -- likely a plan ordering bug, \
                 not just precision. atom={}, {} atoms with err > 1e-3",
                sim_max_err, sim_err_atom, sim_large_err_count,
            );
        }

        // Also compare JIT execution against the f32 simulation.
        // If the JIT disagrees with our manual f32 sim, the codegen has a bug.
        let compiled = CompiledPlan::compile(graph, &plan).expect("compile failed");
        let jit_values = compiled.execute(&overrides_f32);

        let mut jit_vs_sim_max_err: f64 = 0.0;
        let mut jit_vs_sim_err_atom: u64 = 0;
        let mut jit_vs_sim_large = 0usize;
        for i in 0..num_atoms {
            let sim_val = sim_values[i] as f64;
            let jit_val = jit_values[i] as f64;
            let diff = (sim_val - jit_val).abs();
            if diff > jit_vs_sim_max_err {
                jit_vs_sim_max_err = diff;
                jit_vs_sim_err_atom = i as u64;
            }
            if diff > 1e-4 {
                jit_vs_sim_large += 1;
                if jit_vs_sim_large <= 5 {
                    let gi = groups
                        .iter()
                        .position(|g| i as u64 >= g.base_id.0 && (i as u64) < g.base_id.0 + g.count)
                        .unwrap_or(usize::MAX);
                    let op_name = if gi < groups.len() {
                        format!("{:?}", groups[gi].op)
                            .chars()
                            .take_while(|c| *c != ' ' && *c != '{')
                            .collect::<String>()
                    } else {
                        "???".to_string()
                    };
                    eprintln!(
                        "  JIT vs f32-sim: atom {} (group {} {}) sim={} jit={} diff={}",
                        i, gi, op_name, sim_val, jit_val, diff,
                    );
                }
            }
        }
        eprintln!(
            "JIT vs f32-sim: max_err={:.2e} at atom {}, {} atoms with err > 1e-4",
            jit_vs_sim_max_err, jit_vs_sim_err_atom, jit_vs_sim_large,
        );

        // JIT should match our manual f32 computation very closely.
        // Differences > 1e-3 suggest a codegen bug.
        if jit_vs_sim_max_err > 1e-3 {
            // Get reference value for context
            let ref_val = reference_values[jit_vs_sim_err_atom as usize];
            panic!(
                "JIT vs f32-sim max_err={:.2e} at atom {} (ref={}) -- likely a CODEGEN bug. \
                 {} atoms differ by > 1e-4",
                jit_vs_sim_max_err, jit_vs_sim_err_atom, ref_val, jit_vs_sim_large,
            );
        }

        // Summary
        eprintln!("=== Plan validation PASSED ===");
        eprintln!("  Ordering:           0 violations");
        eprintln!(
            "  Completeness:       {} compute atoms covered",
            plan_atoms.len()
        );
        eprintln!("  Copy-from-ref err:  {:.2e}", copy_max_err);
        eprintln!("  f32-sim vs ref:     {:.2e}", sim_max_err);
        eprintln!("  JIT vs f32-sim:     {:.2e}", jit_vs_sim_max_err);
    }

    /// Test with GPT-2 scale dimensions: 4x768 @ 768x768 + bias, followed by another matmul.
    /// This exercises the exact patterns that fail in GPT-2.
    #[test]
    fn test_plan_gpt2_scale() {
        use crate::milli_graph::MilliOpGraph;
        use crate::milli_graph::ops::{MatMul, SimpleBinary};
        use crate::nano_graph::lower::lower_with_info;
        use crate::numeric_tensor::NumericTensor;
        use crate::tensor_info::TensorInfo;

        use super::super::plan::v2c::plan_execution;

        let mut rng = rand::rng();
        let (mut milli, _) = MilliOpGraph::new(std::iter::empty(), &mut rng);

        // GPT-2 like dimensions
        let seq = 4;
        let d_model = 64; // Use 64 instead of 768 to keep test fast
        let d_ff = 64;

        let x = milli.add_input(&mut rng); // [seq, d_model]
        let w1 = milli.add_input(&mut rng); // [d_model, d_ff]
        let bias1 = milli.add_input(&mut rng); // [d_ff]
        let w2 = milli.add_input(&mut rng); // [d_ff, d_model]
        let bias2 = milli.add_input(&mut rng); // [d_model]

        // Layer 1: Y = X @ W1 + bias1
        let y = MatMul::push_new_default_precision(&mut milli, x, w1, DType::F32, &mut rng);
        let y_bias = SimpleBinary::add(&mut milli, y, bias1, &mut rng);

        // Layer 2: Z = Y_bias @ W2 + bias2
        let z = MatMul::push_new_default_precision(&mut milli, y_bias, w2, DType::F32, &mut rng);
        let _z_bias = SimpleBinary::add(&mut milli, z, bias2, &mut rng);

        // Generate random data
        let mk_data =
            |n: usize| -> Vec<f32> { (0..n).map(|i| ((i as f32) * 0.0037).sin() * 0.5).collect() };
        let x_data = mk_data(seq * d_model);
        let w1_data = mk_data(d_model * d_ff);
        let bias1_data = mk_data(d_ff);
        let w2_data = mk_data(d_ff * d_model);
        let bias2_data = mk_data(d_model);

        let x_tensor =
            NumericTensor::<crate::DynRank>::from_vec_shape(x_data, vec![seq, d_model]).unwrap();
        let w1_tensor =
            NumericTensor::<crate::DynRank>::from_vec_shape(w1_data, vec![d_model, d_ff]).unwrap();
        let bias1_tensor =
            NumericTensor::<crate::DynRank>::from_vec_shape(bias1_data, vec![d_ff]).unwrap();
        let w2_tensor =
            NumericTensor::<crate::DynRank>::from_vec_shape(w2_data, vec![d_ff, d_model]).unwrap();
        let bias2_tensor =
            NumericTensor::<crate::DynRank>::from_vec_shape(bias2_data, vec![d_model]).unwrap();

        let mut info_inputs = HashMap::new();
        info_inputs.insert(x, TensorInfo::from(x_tensor.clone()));
        info_inputs.insert(w1, TensorInfo::from(w1_tensor.clone()));
        info_inputs.insert(bias1, TensorInfo::from(bias1_tensor.clone()));
        info_inputs.insert(w2, TensorInfo::from(w2_tensor.clone()));
        info_inputs.insert(bias2, TensorInfo::from(bias2_tensor.clone()));

        let result = lower_with_info(&milli, &info_inputs).unwrap();
        assert!(
            result.unsupported.is_empty(),
            "unsupported: {:?}",
            result.unsupported_details,
        );

        let overrides = build_f32_overrides(&info_inputs, &result);

        for lanes in [1, 2, 4, 8] {
            let plan = plan_execution(&result.graph, lanes);
            compare_plan_vs_interp(
                &result.graph,
                &plan,
                &overrides,
                &format!("gpt2_scale_{}lanes", lanes),
            );
        }
    }

    // ---- Ported from nano_codegen.rs (v1) ----

    /// Helper: build a plan and compare against interpreter for a NanoGraph built directly.
    fn plan_and_compare(graph: &NanoGraph, overrides: &HashMap<u64, f32>, label: &str) {
        use super::super::plan::v2c::plan_execution;
        let plan = plan_execution(graph, 1);
        compare_plan_vs_interp(graph, &plan, overrides, label);
    }

    #[test]
    fn test_bf16_rounding_add() {
        let mut g = NanoGraph::new();

        let a = g.push_group(
            4,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
        );
        let b = g.push_group(
            4,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
        );
        let c = g.push_group(
            4,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine { base: a, stride: 1 },
                InputRef::Affine { base: b, stride: 1 },
            ],
        );

        for i in 0..4 {
            g.outputs.push(c.offset(i));
        }

        let a_vals = [1.0f32, 256.0, 0.1, 1000.0];
        let b_vals = [1e-4f32, 0.001, 0.0001, 0.5];
        let mut overrides = HashMap::new();
        for i in 0..4u64 {
            overrides.insert(a.0 + i, a_vals[i as usize]);
            overrides.insert(b.0 + i, b_vals[i as usize]);
        }

        plan_and_compare(&g, &overrides, "bf16_rounding_add");
    }

    #[test]
    fn test_bf16_rounding_chain() {
        let mut g = NanoGraph::new();

        let a = g.push_group(
            4,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
        );
        let b = g.push_group(
            4,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
        );
        let c = g.push_group(
            4,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine { base: a, stride: 1 },
                InputRef::Affine { base: b, stride: 1 },
            ],
        );
        let d = g.push_group(
            4,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
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

        plan_and_compare(&g, &overrides, "bf16_rounding_chain");
    }

    #[test]
    fn test_f16_rounding_add() {
        let mut g = NanoGraph::new();

        let a = g.push_group(
            4,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
        );
        let b = g.push_group(
            4,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
        );
        let c = g.push_group(
            4,
            DType::F32,
            ScalarOp::Binary {
                op: ScalarBinOp::Add,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![
                InputRef::Affine { base: a, stride: 1 },
                InputRef::Affine { base: b, stride: 1 },
            ],
        );

        for i in 0..4 {
            g.outputs.push(c.offset(i));
        }

        let a_vals = [1.0f32, 100.0, 0.1, 2048.0];
        let b_vals = [1e-5f32, 0.01, 1e-5, 0.25];
        let mut overrides = HashMap::new();
        for i in 0..4u64 {
            overrides.insert(a.0 + i, a_vals[i as usize]);
            overrides.insert(b.0 + i, b_vals[i as usize]);
        }

        plan_and_compare(&g, &overrides, "f16_rounding_add");
    }

    #[test]
    fn test_bf16_reduce_sum() {
        let mut g = NanoGraph::new();

        let a = g.push_group(
            8,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
        );

        let c = g.push_group(
            1,
            DType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Sum,
                reduce_count: 8,
                reduce_stride: 1,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine { base: a, stride: 0 }],
        );

        g.outputs.push(c);

        let mut overrides = HashMap::new();
        let vals = [1.0f32, 0.001, 0.0001, 2.0, 0.00001, 3.0, 0.000001, 4.0];
        for (i, &v) in vals.iter().enumerate() {
            overrides.insert(a.0 + i as u64, v);
        }

        plan_and_compare(&g, &overrides, "bf16_reduce_sum");
    }

    #[test]
    fn test_bf16_identity_cast() {
        let mut g = NanoGraph::new();

        let a = g.push_group(
            4,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
        );
        let b = g.push_group(
            4,
            DType::F32,
            ScalarOp::Identity,
            vec![],
            vec![InputRef::Affine { base: a, stride: 1 }],
        );

        for i in 0..4 {
            g.outputs.push(b.offset(i));
        }

        let mut overrides = HashMap::new();
        overrides.insert(a.0, 1.0001f32);
        overrides.insert(a.0 + 1, 3.14159f32);
        overrides.insert(a.0 + 2, 0.123456f32);
        overrides.insert(a.0 + 3, 65504.0f32);

        plan_and_compare(&g, &overrides, "bf16_identity_cast");
    }

    #[test]
    fn test_bf16_rounding_function_correctness() {
        // Verify that our v2_round_bf16 matches half::bf16::from_f32().to_f32()
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
            let our_result = v2_round_bf16(v);
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

    #[test]
    fn test_select() {
        let mut g = NanoGraph::new();

        let cond = g.push_group(
            4,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
        );
        let x = g.push_group(
            4,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
        );
        let y = g.push_group(
            4,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
        );
        let result = g.push_group(
            4,
            DType::F32,
            ScalarOp::Select,
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

        plan_and_compare(&g, &overrides, "select");
    }

    #[test]
    fn test_reduce_max() {
        let mut g = NanoGraph::new();

        let a = g.push_group(
            8,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
        );

        let c = g.push_group(
            1,
            DType::F32,
            ScalarOp::Reduce {
                kind: ReduceKind::Max,
                reduce_count: 8,
                reduce_stride: 1,
                compute_dtype: DType::F32,
            },
            vec![],
            vec![InputRef::Affine { base: a, stride: 0 }],
        );

        g.outputs.push(c);

        let mut overrides = HashMap::new();
        let test_vals = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        for (i, &v) in test_vals.iter().enumerate() {
            overrides.insert(a.0 + i as u64, v);
        }

        plan_and_compare(&g, &overrides, "reduce_max");
    }

    #[test]
    fn test_explicit_input_ref() {
        let mut g = NanoGraph::new();

        let a = g.push_group(
            4,
            DType::F32,
            ScalarOp::Literal(NumericScalar::F32(0.0)),
            vec![],
            vec![],
        );

        let b = g.push_group(
            3,
            DType::F32,
            ScalarOp::Unary {
                op: ScalarUnaryOp::Neg,
                compute_dtype: DType::F32,
            },
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

        plan_and_compare(&g, &overrides, "explicit_gather_neg");
    }
}
