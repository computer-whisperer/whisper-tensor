#![allow(clippy::all, dead_code, unreachable_patterns)]
//! Cranelift codegen for v6 recovered schedules.
//!
//! Current implementation focuses on additive-reduction loops recovered from
//! whitewashed nano-op pools. Unsupported schedules transparently fall back
//! to v1 scalar nano-op codegen.

use super::synthesis::{
    AccessDimRole, LoopDim, LoopIntent, LoopScheduleCandidate, PipelineArtifacts, PipelineError,
    RecoveredLoop, RecoveredTensorAccess, ReductionBinOp, ReductionIntent, ReductionTermPattern,
    ReductionUnaryOp, build_from_graph,
};
use crate::compiler::attempts::v1_scalar_crystal::codegen as v1_codegen;
use crate::graph::GlobalId;
use crate::milli_graph::MilliOpGraph;
use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use std::collections::HashMap;

pub use v1_codegen::{CodegenError, TensorLayout};

#[derive(Debug, thiserror::Error)]
pub enum V6CodegenError {
    #[error(transparent)]
    Synthesis(#[from] PipelineError),
    #[error(transparent)]
    Codegen(#[from] CodegenError),
}

pub struct NativeCompiledGraph {
    _module: JITModule,
    func_ptr: *const u8,
    pub layout: TensorLayout,
}

unsafe impl Send for NativeCompiledGraph {}

impl NativeCompiledGraph {
    pub unsafe fn execute(&self, buffers: &mut [*mut f32]) {
        assert!(
            buffers.len() >= self.layout.num_buffers,
            "Expected at least {} buffers, got {}",
            self.layout.num_buffers,
            buffers.len()
        );
        let func: unsafe extern "C" fn(*const *mut f32) =
            unsafe { std::mem::transmute(self.func_ptr) };
        unsafe { func(buffers.as_ptr()) };
    }
}

#[derive(Debug, Clone)]
struct ReductionKernelSpec {
    output_tensor: GlobalId,
    output_shape: Vec<usize>,
    schedule: LoopScheduleCandidate,
    terms: usize,
    canonical_term: ReductionTermPattern,
    accesses: Vec<RecoveredTensorAccess>,
}

#[derive(Debug, Clone)]
struct Rank2AffineAccess {
    constant: i64,
    row_stride: i64,
    col_stride: i64,
    reduction_stride: i64,
}

pub fn compile_graph(
    graph: &MilliOpGraph,
    shapes: &HashMap<GlobalId, Vec<usize>>,
) -> Result<(NativeCompiledGraph, PipelineArtifacts), V6CodegenError> {
    let artifacts = build_from_graph(graph, shapes)?;
    let spec = select_single_reduction_kernel(&artifacts.schedule.loops).ok_or_else(|| {
        V6CodegenError::Codegen(CodegenError::CodegenError(
            "v6: no supported single additive-reduction loop recovered for codegen".to_string(),
        ))
    })?;

    let layout = TensorLayout::from_shapes(shapes);
    let compiled = compile_reduction_kernel(&spec, &layout)?;
    Ok((compiled, artifacts))
}

fn select_single_reduction_kernel(loops: &[RecoveredLoop]) -> Option<ReductionKernelSpec> {
    if loops.len() != 1 {
        return None;
    }
    let loop0 = &loops[0];
    let selected_idx = loop0.selected_schedule?;
    let schedule = loop0.schedule_candidates.get(selected_idx)?.clone();

    let LoopIntent::AdditiveReduction(ReductionIntent {
        terms,
        canonical_term,
        accesses,
        ..
    }) = &loop0.intent
    else {
        return None;
    };

    if !is_supported_term(canonical_term) {
        return None;
    }
    if accesses.iter().any(|a| !is_supported_access(a)) {
        return None;
    }

    Some(ReductionKernelSpec {
        output_tensor: loop0.output_tensor,
        output_shape: loop0.output_shape.clone(),
        schedule,
        terms: *terms,
        canonical_term: canonical_term.clone(),
        accesses: accesses.clone(),
    })
}

fn is_supported_access(access: &RecoveredTensorAccess) -> bool {
    access
        .dim_roles
        .iter()
        .all(|role| !matches!(role, AccessDimRole::Unknown))
}

fn is_supported_term(term: &ReductionTermPattern) -> bool {
    match term {
        ReductionTermPattern::Load(_) | ReductionTermPattern::Literal(_) => true,
        ReductionTermPattern::Bin { a, b, .. } => is_supported_term(a) && is_supported_term(b),
        ReductionTermPattern::Unary { input, .. } => is_supported_term(input),
    }
}

fn compile_reduction_kernel(
    spec: &ReductionKernelSpec,
    layout: &TensorLayout,
) -> Result<NativeCompiledGraph, CodegenError> {
    let (mut module, math_func_ids) = setup_jit_module()?;
    let ptr_type = module.isa().pointer_type();

    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(ptr_type));
    let func_id = module.declare_function("wt_v6_reduction_kernel", Linkage::Local, &sig)?;

    let mut ctx = module.make_context();
    ctx.func.signature = sig;

    let math_refs: HashMap<&str, cranelift_codegen::ir::FuncRef> = math_func_ids
        .iter()
        .map(|(name, fid)| {
            let fref = module.declare_func_in_func(*fid, &mut ctx.func);
            (*name, fref)
        })
        .collect();

    let mut fn_builder_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);

    let entry = builder.create_block();
    builder.append_block_params_for_function_params(entry);
    builder.switch_to_block(entry);
    builder.seal_block(entry);
    let ptr_table = builder.block_params(entry)[0];

    let output_ptr = load_tensor_base_ptr(
        &mut builder,
        layout,
        ptr_table,
        ptr_type,
        spec.output_tensor,
    )?;

    let mut input_ptrs = HashMap::new();
    for access in &spec.accesses {
        let base = load_tensor_base_ptr(&mut builder, layout, ptr_table, ptr_type, access.tensor)?;
        input_ptrs.insert(access.tensor, base);
    }

    let mut next_var_index = 0u32;
    if can_emit_rank2_microkernel(spec) {
        emit_rank2_microkernel(
            &mut builder,
            &mut next_var_index,
            spec,
            &input_ptrs,
            output_ptr,
            &math_refs,
        )?;
    } else {
        let output_elements = spec.output_shape.iter().product::<usize>() as i64;
        let iv_idx = alloc_loop_var(&mut next_var_index);
        emit_for_loop(
            &mut builder,
            iv_idx,
            0,
            output_elements,
            1,
            |builder, flat| {
                let zero = builder.ins().f32const(0.0);
                store_f32_at_flat(builder, output_ptr, flat, zero);
                Ok(())
            },
        )?;

        let mut output_axis_vars: Vec<Option<cranelift_codegen::ir::Value>> =
            vec![None; spec.output_shape.len()];
        let mut reduction_var = None;
        emit_nested_loops(
            &mut builder,
            &mut next_var_index,
            spec,
            0,
            &mut output_axis_vars,
            &mut reduction_var,
            &input_ptrs,
            output_ptr,
            &math_refs,
        )?;
    }

    builder.ins().return_(&[]);
    builder.finalize();

    module.define_function(func_id, &mut ctx)?;
    module.clear_context(&mut ctx);
    module.finalize_definitions().expect("finalize JIT");
    let code_ptr = module.get_finalized_function(func_id);

    Ok(NativeCompiledGraph {
        _module: module,
        func_ptr: code_ptr,
        layout: layout.clone(),
    })
}

fn can_emit_rank2_microkernel(spec: &ReductionKernelSpec) -> bool {
    if spec.output_shape.len() != 2 {
        return false;
    }
    let has_row = spec.schedule.loop_order.contains(&LoopDim::OutputAxis(0));
    let has_col = spec.schedule.loop_order.contains(&LoopDim::OutputAxis(1));
    let has_reduction = spec.schedule.loop_order.contains(&LoopDim::ReductionAxis);
    has_row && has_col && has_reduction && spec.terms > 0
}

#[allow(clippy::too_many_arguments)]
fn emit_rank2_microkernel(
    builder: &mut FunctionBuilder,
    next_var_index: &mut u32,
    spec: &ReductionKernelSpec,
    input_ptrs: &HashMap<GlobalId, cranelift_codegen::ir::Value>,
    output_ptr: cranelift_codegen::ir::Value,
    math_refs: &HashMap<&str, cranelift_codegen::ir::FuncRef>,
) -> Result<(), CodegenError> {
    let m = spec.output_shape[0] as i64;
    let n = spec.output_shape[1] as i64;
    let mut nr = spec
        .schedule
        .vectorize
        .as_ref()
        .map(|v| v.width)
        .or_else(|| spec.schedule.output_tiles.get(1).copied())
        .unwrap_or(4)
        .clamp(1, 4);
    if nr > spec.output_shape[1].max(1) {
        nr = spec.output_shape[1].max(1);
    }
    let affine_accesses: Option<Vec<Rank2AffineAccess>> = spec
        .accesses
        .iter()
        .map(derive_rank2_affine_access)
        .collect();
    let access_base_ptrs = spec
        .accesses
        .iter()
        .map(|access| {
            input_ptrs
                .get(&access.tensor)
                .copied()
                .ok_or(CodegenError::UnknownTensor(access.tensor))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let n_main = (spec.output_shape[1] / nr) * nr;
    let k_unroll = spec
        .schedule
        .reduction_unroll
        .clamp(1, 8)
        .min(spec.terms.max(1));
    let k_main = (spec.terms / k_unroll) * k_unroll;

    let iv_i = alloc_loop_var(next_var_index);
    let iv_j0 = alloc_loop_var(next_var_index);
    let iv_k_blk_vec = alloc_loop_var(next_var_index);
    let iv_k_tail_vec = alloc_loop_var(next_var_index);
    let iv_j_tail = alloc_loop_var(next_var_index);
    let iv_k_blk_tail = alloc_loop_var(next_var_index);
    let iv_k_tail_tail = alloc_loop_var(next_var_index);
    let acc_vec_vars: Vec<Variable> = (0..nr)
        .map(|_| {
            let var = Variable::from_u32(alloc_loop_var(next_var_index));
            builder.declare_var(var, types::F32);
            var
        })
        .collect();
    let acc_tail_var = Variable::from_u32(alloc_loop_var(next_var_index));
    builder.declare_var(acc_tail_var, types::F32);

    emit_for_loop(builder, iv_i, 0, m, 1, |builder, row_i| {
        if n_main > 0 {
            emit_for_loop(
                builder,
                iv_j0,
                0,
                n_main as i64,
                nr as i64,
                |builder, col_j0| {
                    for &var in &acc_vec_vars {
                        let zero = builder.ins().f32const(0.0);
                        builder.def_var(var, zero);
                    }

                    let lane_affine_bases = if let Some(affine) = affine_accesses.as_ref() {
                        let mut lane_bases = Vec::with_capacity(nr);
                        for lane in 0..nr {
                            let col_j = if lane == 0 {
                                col_j0
                            } else {
                                builder.ins().iadd_imm(col_j0, lane as i64)
                            };
                            let mut access_bases = Vec::with_capacity(affine.len());
                            for access in affine {
                                access_bases
                                    .push(emit_rank2_affine_base(builder, access, row_i, col_j));
                            }
                            lane_bases.push(access_bases);
                        }
                        Some(lane_bases)
                    } else {
                        None
                    };

                    if k_main > 0 {
                        emit_for_loop(
                            builder,
                            iv_k_blk_vec,
                            0,
                            k_main as i64,
                            k_unroll as i64,
                            |builder, red_k_base| {
                                for u in 0..k_unroll {
                                    let red_k = if u == 0 {
                                        red_k_base
                                    } else {
                                        builder.ins().iadd_imm(red_k_base, u as i64)
                                    };
                                    let shared_loads = if let Some(affine) =
                                        affine_accesses.as_ref()
                                    {
                                        let mut shared = vec![None; affine.len()];
                                        let lane0_bases = lane_affine_bases
                                            .as_ref()
                                            .map(|bases| bases[0].as_slice());
                                        for (access_idx, access) in affine.iter().enumerate() {
                                            if access.col_stride != 0 {
                                                continue;
                                            }
                                            let flat_base = if let Some(bases) = lane0_bases {
                                                *bases.get(access_idx).ok_or_else(|| {
                                                    CodegenError::CodegenError(
                                                        "v6: missing lane0 affine base".to_string(),
                                                    )
                                                })?
                                            } else {
                                                emit_rank2_affine_base(
                                                    builder, access, row_i, col_j0,
                                                )
                                            };
                                            let flat = add_scaled_i64(
                                                builder,
                                                flat_base,
                                                red_k,
                                                access.reduction_stride,
                                            );
                                            let base = *access_base_ptrs
                                                .get(access_idx)
                                                .ok_or_else(|| {
                                                    CodegenError::CodegenError(
                                                        "v6: missing base ptr for shared load"
                                                            .to_string(),
                                                    )
                                                })?;
                                            shared[access_idx] =
                                                Some(load_f32_at_flat(builder, base, flat));
                                        }
                                        Some(shared)
                                    } else {
                                        None
                                    };
                                    for lane in 0..nr {
                                        let col_j = if lane == 0 {
                                            col_j0
                                        } else {
                                            builder.ins().iadd_imm(col_j0, lane as i64)
                                        };
                                        let lane_bases = lane_affine_bases
                                            .as_ref()
                                            .map(|bases| bases[lane].as_slice());
                                        let load_vals = build_rank2_load_values(
                                            builder,
                                            spec,
                                            &access_base_ptrs,
                                            affine_accesses.as_deref(),
                                            lane_bases,
                                            shared_loads.as_deref(),
                                            row_i,
                                            col_j,
                                            red_k,
                                        )?;
                                        let var = acc_vec_vars[lane];
                                        let cur = builder.use_var(var);
                                        let next = emit_accumulate_term(
                                            builder,
                                            cur,
                                            &spec.canonical_term,
                                            &load_vals,
                                            math_refs,
                                        )?;
                                        builder.def_var(var, next);
                                    }
                                }
                                Ok(())
                            },
                        )?;
                    }

                    if k_main < spec.terms {
                        emit_for_loop(
                            builder,
                            iv_k_tail_vec,
                            k_main as i64,
                            spec.terms as i64,
                            1,
                            |builder, red_k| {
                                let shared_loads = if let Some(affine) = affine_accesses.as_ref() {
                                    let mut shared = vec![None; affine.len()];
                                    let lane0_bases =
                                        lane_affine_bases.as_ref().map(|bases| bases[0].as_slice());
                                    for (access_idx, access) in affine.iter().enumerate() {
                                        if access.col_stride != 0 {
                                            continue;
                                        }
                                        let flat_base = if let Some(bases) = lane0_bases {
                                            *bases.get(access_idx).ok_or_else(|| {
                                                CodegenError::CodegenError(
                                                    "v6: missing lane0 affine base".to_string(),
                                                )
                                            })?
                                        } else {
                                            emit_rank2_affine_base(builder, access, row_i, col_j0)
                                        };
                                        let flat = add_scaled_i64(
                                            builder,
                                            flat_base,
                                            red_k,
                                            access.reduction_stride,
                                        );
                                        let base =
                                            *access_base_ptrs.get(access_idx).ok_or_else(|| {
                                                CodegenError::CodegenError(
                                                    "v6: missing base ptr for shared load"
                                                        .to_string(),
                                                )
                                            })?;
                                        shared[access_idx] =
                                            Some(load_f32_at_flat(builder, base, flat));
                                    }
                                    Some(shared)
                                } else {
                                    None
                                };
                                for lane in 0..nr {
                                    let col_j = if lane == 0 {
                                        col_j0
                                    } else {
                                        builder.ins().iadd_imm(col_j0, lane as i64)
                                    };
                                    let lane_bases = lane_affine_bases
                                        .as_ref()
                                        .map(|bases| bases[lane].as_slice());
                                    let load_vals = build_rank2_load_values(
                                        builder,
                                        spec,
                                        &access_base_ptrs,
                                        affine_accesses.as_deref(),
                                        lane_bases,
                                        shared_loads.as_deref(),
                                        row_i,
                                        col_j,
                                        red_k,
                                    )?;
                                    let var = acc_vec_vars[lane];
                                    let cur = builder.use_var(var);
                                    let next = emit_accumulate_term(
                                        builder,
                                        cur,
                                        &spec.canonical_term,
                                        &load_vals,
                                        math_refs,
                                    )?;
                                    builder.def_var(var, next);
                                }
                                Ok(())
                            },
                        )?;
                    }

                    for (lane, &var) in acc_vec_vars.iter().enumerate() {
                        let col_j = if lane == 0 {
                            col_j0
                        } else {
                            builder.ins().iadd_imm(col_j0, lane as i64)
                        };
                        let out_flat = output_flat_2d(builder, row_i, col_j, n);
                        let acc_lane = builder.use_var(var);
                        store_f32_at_flat(builder, output_ptr, out_flat, acc_lane);
                    }
                    Ok(())
                },
            )?;
        }

        if n_main < spec.output_shape[1] {
            emit_for_loop(builder, iv_j_tail, n_main as i64, n, 1, |builder, col_j| {
                let zero = builder.ins().f32const(0.0);
                builder.def_var(acc_tail_var, zero);
                let tail_affine_bases = affine_accesses.as_ref().map(|affine| {
                    affine
                        .iter()
                        .map(|access| emit_rank2_affine_base(builder, access, row_i, col_j))
                        .collect::<Vec<_>>()
                });
                if k_main > 0 {
                    emit_for_loop(
                        builder,
                        iv_k_blk_tail,
                        0,
                        k_main as i64,
                        k_unroll as i64,
                        |builder, red_k_base| {
                            for u in 0..k_unroll {
                                let red_k = if u == 0 {
                                    red_k_base
                                } else {
                                    builder.ins().iadd_imm(red_k_base, u as i64)
                                };
                                let load_vals = build_rank2_load_values(
                                    builder,
                                    spec,
                                    &access_base_ptrs,
                                    affine_accesses.as_deref(),
                                    tail_affine_bases.as_deref(),
                                    None,
                                    row_i,
                                    col_j,
                                    red_k,
                                )?;
                                let cur = builder.use_var(acc_tail_var);
                                let next = emit_accumulate_term(
                                    builder,
                                    cur,
                                    &spec.canonical_term,
                                    &load_vals,
                                    math_refs,
                                )?;
                                builder.def_var(acc_tail_var, next);
                            }
                            Ok(())
                        },
                    )?;
                }

                if k_main < spec.terms {
                    emit_for_loop(
                        builder,
                        iv_k_tail_tail,
                        k_main as i64,
                        spec.terms as i64,
                        1,
                        |builder, red_k| {
                            let load_vals = build_rank2_load_values(
                                builder,
                                spec,
                                &access_base_ptrs,
                                affine_accesses.as_deref(),
                                tail_affine_bases.as_deref(),
                                None,
                                row_i,
                                col_j,
                                red_k,
                            )?;
                            let cur = builder.use_var(acc_tail_var);
                            let next = emit_accumulate_term(
                                builder,
                                cur,
                                &spec.canonical_term,
                                &load_vals,
                                math_refs,
                            )?;
                            builder.def_var(acc_tail_var, next);
                            Ok(())
                        },
                    )?;
                }
                let out_flat = output_flat_2d(builder, row_i, col_j, n);
                let acc = builder.use_var(acc_tail_var);
                store_f32_at_flat(builder, output_ptr, out_flat, acc);
                Ok(())
            })?;
        }

        Ok(())
    })
}

#[allow(clippy::too_many_arguments)]
fn emit_nested_loops(
    builder: &mut FunctionBuilder,
    next_var_index: &mut u32,
    spec: &ReductionKernelSpec,
    depth: usize,
    output_axis_vars: &mut [Option<cranelift_codegen::ir::Value>],
    reduction_var: &mut Option<cranelift_codegen::ir::Value>,
    input_ptrs: &HashMap<GlobalId, cranelift_codegen::ir::Value>,
    output_ptr: cranelift_codegen::ir::Value,
    math_refs: &HashMap<&str, cranelift_codegen::ir::FuncRef>,
) -> Result<(), CodegenError> {
    if depth == spec.schedule.loop_order.len() {
        let k = reduction_var.ok_or_else(|| {
            CodegenError::CodegenError("v6: missing reduction loop variable".to_string())
        })?;
        emit_reduction_update(
            builder,
            spec,
            output_axis_vars,
            k,
            input_ptrs,
            output_ptr,
            math_refs,
        )?;
        return Ok(());
    }

    match spec.schedule.loop_order[depth] {
        LoopDim::OutputAxis(axis) => {
            if axis >= spec.output_shape.len() {
                return Err(CodegenError::CodegenError(format!(
                    "v6: output axis {} out of range for rank {}",
                    axis,
                    spec.output_shape.len()
                )));
            }
            let end = spec.output_shape[axis] as i64;
            let iv_idx = alloc_loop_var(next_var_index);
            emit_for_loop(builder, iv_idx, 0, end, 1, |builder, iv| {
                output_axis_vars[axis] = Some(iv);
                emit_nested_loops(
                    builder,
                    next_var_index,
                    spec,
                    depth + 1,
                    output_axis_vars,
                    reduction_var,
                    input_ptrs,
                    output_ptr,
                    math_refs,
                )?;
                output_axis_vars[axis] = None;
                Ok(())
            })
        }
        LoopDim::ReductionAxis => {
            let end = spec.terms as i64;
            let iv_idx = alloc_loop_var(next_var_index);
            emit_for_loop(builder, iv_idx, 0, end, 1, |builder, iv| {
                *reduction_var = Some(iv);
                emit_nested_loops(
                    builder,
                    next_var_index,
                    spec,
                    depth + 1,
                    output_axis_vars,
                    reduction_var,
                    input_ptrs,
                    output_ptr,
                    math_refs,
                )?;
                *reduction_var = None;
                Ok(())
            })
        }
    }
}

fn emit_reduction_update(
    builder: &mut FunctionBuilder,
    spec: &ReductionKernelSpec,
    output_axis_vars: &[Option<cranelift_codegen::ir::Value>],
    reduction_k: cranelift_codegen::ir::Value,
    input_ptrs: &HashMap<GlobalId, cranelift_codegen::ir::Value>,
    output_ptr: cranelift_codegen::ir::Value,
    math_refs: &HashMap<&str, cranelift_codegen::ir::FuncRef>,
) -> Result<(), CodegenError> {
    let mut load_vals = Vec::with_capacity(spec.accesses.len());
    for access in &spec.accesses {
        let base = *input_ptrs
            .get(&access.tensor)
            .ok_or(CodegenError::UnknownTensor(access.tensor))?;
        let flat = emit_flat_index_from_access(builder, access, output_axis_vars, reduction_k)?;
        let val = load_f32_at_flat(builder, base, flat);
        load_vals.push(val);
    }

    let out_flat = emit_output_flat_index(builder, &spec.output_shape, output_axis_vars)?;
    let cur = load_f32_at_flat(builder, output_ptr, out_flat);
    let next = emit_accumulate_term(builder, cur, &spec.canonical_term, &load_vals, math_refs)?;
    store_f32_at_flat(builder, output_ptr, out_flat, next);
    Ok(())
}

fn output_flat_2d(
    builder: &mut FunctionBuilder,
    row: cranelift_codegen::ir::Value,
    col: cranelift_codegen::ir::Value,
    n: i64,
) -> cranelift_codegen::ir::Value {
    let row_off = builder.ins().imul_imm(row, n);
    builder.ins().iadd(row_off, col)
}

fn emit_output_flat_index(
    builder: &mut FunctionBuilder,
    shape: &[usize],
    output_axis_vars: &[Option<cranelift_codegen::ir::Value>],
) -> Result<cranelift_codegen::ir::Value, CodegenError> {
    let mut flat = builder.ins().iconst(types::I64, 0);
    for (axis, &dim) in shape.iter().enumerate() {
        let idx = output_axis_vars.get(axis).and_then(|v| *v).ok_or_else(|| {
            CodegenError::CodegenError(format!("v6: missing output axis value {}", axis))
        })?;
        flat = builder.ins().imul_imm(flat, dim as i64);
        flat = builder.ins().iadd(flat, idx);
    }
    Ok(flat)
}

fn emit_flat_index_from_access(
    builder: &mut FunctionBuilder,
    access: &RecoveredTensorAccess,
    output_axis_vars: &[Option<cranelift_codegen::ir::Value>],
    reduction_k: cranelift_codegen::ir::Value,
) -> Result<cranelift_codegen::ir::Value, CodegenError> {
    let mut flat = builder.ins().iconst(types::I64, 0);

    for (&dim_extent, role) in access.tensor_shape.iter().zip(access.dim_roles.iter()) {
        let coord = emit_coord_from_role(builder, role, output_axis_vars, reduction_k)?;
        flat = builder.ins().imul_imm(flat, dim_extent as i64);
        flat = builder.ins().iadd(flat, coord);
    }

    Ok(flat)
}

fn build_rank2_load_values(
    builder: &mut FunctionBuilder,
    spec: &ReductionKernelSpec,
    access_base_ptrs: &[cranelift_codegen::ir::Value],
    affine_accesses: Option<&[Rank2AffineAccess]>,
    affine_bases: Option<&[cranelift_codegen::ir::Value]>,
    shared_loads: Option<&[Option<cranelift_codegen::ir::Value>]>,
    row_i: cranelift_codegen::ir::Value,
    col_j: cranelift_codegen::ir::Value,
    reduction_k: cranelift_codegen::ir::Value,
) -> Result<Vec<cranelift_codegen::ir::Value>, CodegenError> {
    let mut load_vals = Vec::with_capacity(spec.accesses.len());
    if let Some(affine) = affine_accesses {
        for (idx, access) in affine.iter().enumerate() {
            if let Some(Some(shared)) = shared_loads.and_then(|loads| loads.get(idx)) {
                load_vals.push(*shared);
                continue;
            }

            let base = *access_base_ptrs.get(idx).ok_or_else(|| {
                CodegenError::CodegenError(
                    "v6: missing base ptr for affine rank2 access".to_string(),
                )
            })?;
            let flat_base = if let Some(bases) = affine_bases {
                *bases.get(idx).ok_or_else(|| {
                    CodegenError::CodegenError(
                        "v6: missing precomputed affine base for rank2 access".to_string(),
                    )
                })?
            } else {
                emit_rank2_affine_base(builder, access, row_i, col_j)
            };
            let flat = add_scaled_i64(builder, flat_base, reduction_k, access.reduction_stride);
            let val = load_f32_at_flat(builder, base, flat);
            load_vals.push(val);
        }
        return Ok(load_vals);
    }

    let output_axis_vars = [Some(row_i), Some(col_j)];
    for (idx, access) in spec.accesses.iter().enumerate() {
        if let Some(Some(shared)) = shared_loads.and_then(|loads| loads.get(idx)) {
            load_vals.push(*shared);
            continue;
        }

        let base = *access_base_ptrs.get(idx).ok_or_else(|| {
            CodegenError::CodegenError(
                "v6: missing base ptr for non-affine rank2 access".to_string(),
            )
        })?;
        let flat = emit_flat_index_from_access(builder, access, &output_axis_vars, reduction_k)?;
        let val = load_f32_at_flat(builder, base, flat);
        load_vals.push(val);
    }
    Ok(load_vals)
}

fn emit_rank2_affine_base(
    builder: &mut FunctionBuilder,
    access: &Rank2AffineAccess,
    row_i: cranelift_codegen::ir::Value,
    col_j: cranelift_codegen::ir::Value,
) -> cranelift_codegen::ir::Value {
    let base = builder.ins().iconst(types::I64, access.constant);
    let base = add_scaled_i64(builder, base, row_i, access.row_stride);
    add_scaled_i64(builder, base, col_j, access.col_stride)
}

fn derive_rank2_affine_access(access: &RecoveredTensorAccess) -> Option<Rank2AffineAccess> {
    if access.tensor_shape.len() != access.dim_roles.len() {
        return None;
    }
    let rank = access.tensor_shape.len();
    let mut dim_strides = vec![1i64; rank];
    let mut running_stride = 1i64;
    for dim in (0..rank).rev() {
        dim_strides[dim] = running_stride;
        running_stride *= access.tensor_shape[dim].max(1) as i64;
    }

    let mut constant = 0i64;
    let mut row_stride = 0i64;
    let mut col_stride = 0i64;
    let mut reduction_stride = 0i64;
    for (dim, role) in access.dim_roles.iter().enumerate() {
        let dim_stride = dim_strides[dim];
        match role {
            AccessDimRole::Constant { value } => {
                constant += dim_stride * (*value as i64);
            }
            AccessDimRole::OutputAxis {
                axis,
                stride,
                offset,
            } => {
                constant += dim_stride * (*offset as i64);
                match axis {
                    0 => row_stride += dim_stride * (*stride as i64),
                    1 => col_stride += dim_stride * (*stride as i64),
                    _ => return None,
                }
            }
            AccessDimRole::ReductionAxis { stride, offset } => {
                constant += dim_stride * (*offset as i64);
                reduction_stride += dim_stride * (*stride as i64);
            }
            AccessDimRole::AffineMixed {
                output_strides,
                reduction_stride: red_stride,
                offset,
            } => {
                constant += dim_stride * (*offset as i64);
                reduction_stride += dim_stride * (*red_stride as i64);
                for (axis, stride) in output_strides {
                    match axis {
                        0 => row_stride += dim_stride * (*stride as i64),
                        1 => col_stride += dim_stride * (*stride as i64),
                        _ => return None,
                    }
                }
            }
            AccessDimRole::Unknown => return None,
        }
    }

    Some(Rank2AffineAccess {
        constant,
        row_stride,
        col_stride,
        reduction_stride,
    })
}

fn emit_coord_from_role(
    builder: &mut FunctionBuilder,
    role: &AccessDimRole,
    output_axis_vars: &[Option<cranelift_codegen::ir::Value>],
    reduction_k: cranelift_codegen::ir::Value,
) -> Result<cranelift_codegen::ir::Value, CodegenError> {
    let mut coord = match role {
        AccessDimRole::Constant { value } => builder.ins().iconst(types::I64, *value as i64),
        AccessDimRole::OutputAxis { offset, .. }
        | AccessDimRole::ReductionAxis { offset, .. }
        | AccessDimRole::AffineMixed { offset, .. } => {
            builder.ins().iconst(types::I64, *offset as i64)
        }
        AccessDimRole::Unknown => {
            return Err(CodegenError::CodegenError(
                "v6: unsupported Unknown access role in codegen".to_string(),
            ));
        }
    };

    match role {
        AccessDimRole::Constant { .. } => {}
        AccessDimRole::OutputAxis { axis, stride, .. } => {
            let var = output_axis_vars
                .get(*axis)
                .and_then(|v| *v)
                .ok_or_else(|| {
                    CodegenError::CodegenError(format!(
                        "v6: missing output variable for axis {}",
                        axis
                    ))
                })?;
            coord = add_scaled(builder, coord, var, *stride);
        }
        AccessDimRole::ReductionAxis { stride, .. } => {
            coord = add_scaled(builder, coord, reduction_k, *stride);
        }
        AccessDimRole::AffineMixed {
            output_strides,
            reduction_stride,
            ..
        } => {
            for (axis, stride) in output_strides {
                let var = output_axis_vars
                    .get(*axis)
                    .and_then(|v| *v)
                    .ok_or_else(|| {
                        CodegenError::CodegenError(format!(
                            "v6: missing output variable for axis {}",
                            axis
                        ))
                    })?;
                coord = add_scaled(builder, coord, var, *stride);
            }
            coord = add_scaled(builder, coord, reduction_k, *reduction_stride);
        }
        AccessDimRole::Unknown => {}
    }

    Ok(coord)
}

fn add_scaled(
    builder: &mut FunctionBuilder,
    base: cranelift_codegen::ir::Value,
    var: cranelift_codegen::ir::Value,
    stride: isize,
) -> cranelift_codegen::ir::Value {
    if stride == 0 {
        return base;
    }
    if stride == 1 {
        return builder.ins().iadd(base, var);
    }
    let stride_val = builder.ins().iconst(types::I64, stride as i64);
    let scaled = builder.ins().imul(var, stride_val);
    builder.ins().iadd(base, scaled)
}

fn add_scaled_i64(
    builder: &mut FunctionBuilder,
    base: cranelift_codegen::ir::Value,
    var: cranelift_codegen::ir::Value,
    stride: i64,
) -> cranelift_codegen::ir::Value {
    if stride == 0 {
        return base;
    }
    if stride == 1 {
        return builder.ins().iadd(base, var);
    }
    if stride == -1 {
        return builder.ins().isub(base, var);
    }
    let stride_val = builder.ins().iconst(types::I64, stride);
    let scaled = builder.ins().imul(var, stride_val);
    builder.ins().iadd(base, scaled)
}

fn term_load_index(term: &ReductionTermPattern) -> Option<usize> {
    match term {
        ReductionTermPattern::Load(i) => Some(*i),
        _ => None,
    }
}

fn emit_accumulate_term(
    builder: &mut FunctionBuilder,
    acc: cranelift_codegen::ir::Value,
    term: &ReductionTermPattern,
    load_vals: &[cranelift_codegen::ir::Value],
    math_refs: &HashMap<&str, cranelift_codegen::ir::FuncRef>,
) -> Result<cranelift_codegen::ir::Value, CodegenError> {
    if let ReductionTermPattern::Bin {
        op: ReductionBinOp::Mul,
        a,
        b,
    } = term
    {
        if let (Some(ai), Some(bi)) = (term_load_index(a), term_load_index(b)) {
            let va = *load_vals.get(ai).ok_or_else(|| {
                CodegenError::CodegenError(format!("v6: term load index {} out of range", ai))
            })?;
            let vb = *load_vals.get(bi).ok_or_else(|| {
                CodegenError::CodegenError(format!("v6: term load index {} out of range", bi))
            })?;
            return Ok(builder.ins().fma(va, vb, acc));
        }
    }

    let term_val = emit_term_expr(builder, term, load_vals, math_refs)?;
    Ok(builder.ins().fadd(acc, term_val))
}

fn emit_term_expr(
    builder: &mut FunctionBuilder,
    term: &ReductionTermPattern,
    load_vals: &[cranelift_codegen::ir::Value],
    math_refs: &HashMap<&str, cranelift_codegen::ir::FuncRef>,
) -> Result<cranelift_codegen::ir::Value, CodegenError> {
    Ok(match term {
        ReductionTermPattern::Load(i) => *load_vals.get(*i).ok_or_else(|| {
            CodegenError::CodegenError(format!("v6: term load index {} out of range", i))
        })?,
        ReductionTermPattern::Literal(bits) => {
            let v = f64::from_bits(*bits) as f32;
            builder.ins().f32const(v)
        }
        ReductionTermPattern::Bin { op, a, b } => {
            let va = emit_term_expr(builder, a, load_vals, math_refs)?;
            let vb = emit_term_expr(builder, b, load_vals, math_refs)?;
            match op {
                ReductionBinOp::Add => builder.ins().fadd(va, vb),
                ReductionBinOp::Sub => builder.ins().fsub(va, vb),
                ReductionBinOp::Mul => builder.ins().fmul(va, vb),
                ReductionBinOp::Div => builder.ins().fdiv(va, vb),
                ReductionBinOp::Max => builder.ins().fmax(va, vb),
                ReductionBinOp::Min => builder.ins().fmin(va, vb),
            }
        }
        ReductionTermPattern::Unary { op, input } => {
            let vin = emit_term_expr(builder, input, load_vals, math_refs)?;
            match op {
                ReductionUnaryOp::Neg => builder.ins().fneg(vin),
                ReductionUnaryOp::Abs => builder.ins().fabs(vin),
                ReductionUnaryOp::Sqrt => builder.ins().sqrt(vin),
                ReductionUnaryOp::Floor => builder.ins().floor(vin),
                ReductionUnaryOp::Ceil => builder.ins().ceil(vin),
                ReductionUnaryOp::Reciprocal => {
                    let one = builder.ins().f32const(1.0);
                    builder.ins().fdiv(one, vin)
                }
                ReductionUnaryOp::Exp => {
                    let fref = math_refs["wt_expf"];
                    let call = builder.ins().call(fref, &[vin]);
                    builder.inst_results(call)[0]
                }
                ReductionUnaryOp::Ln => {
                    let fref = math_refs["wt_logf"];
                    let call = builder.ins().call(fref, &[vin]);
                    builder.inst_results(call)[0]
                }
                ReductionUnaryOp::Tanh => {
                    let fref = math_refs["wt_tanhf"];
                    let call = builder.ins().call(fref, &[vin]);
                    builder.inst_results(call)[0]
                }
            }
        }
    })
}

fn emit_for_loop<F>(
    builder: &mut FunctionBuilder,
    iv_index: u32,
    start: i64,
    end: i64,
    step: i64,
    mut body: F,
) -> Result<(), CodegenError>
where
    F: FnMut(&mut FunctionBuilder, cranelift_codegen::ir::Value) -> Result<(), CodegenError>,
{
    let iv = Variable::from_u32(iv_index);
    builder.declare_var(iv, types::I64);
    let start_val = builder.ins().iconst(types::I64, start);
    builder.def_var(iv, start_val);

    let loop_header = builder.create_block();
    let loop_body = builder.create_block();
    let loop_exit = builder.create_block();

    builder.ins().jump(loop_header, &[]);
    builder.switch_to_block(loop_header);

    let i = builder.use_var(iv);
    let end_val = builder.ins().iconst(types::I64, end);
    let cmp = builder.ins().icmp(
        cranelift_codegen::ir::condcodes::IntCC::SignedLessThan,
        i,
        end_val,
    );
    builder.ins().brif(cmp, loop_body, &[], loop_exit, &[]);

    builder.switch_to_block(loop_body);
    body(builder, i)?;
    let i_next = builder.ins().iadd_imm(i, step);
    builder.def_var(iv, i_next);
    builder.ins().jump(loop_header, &[]);
    builder.seal_block(loop_body);
    builder.seal_block(loop_header);

    builder.switch_to_block(loop_exit);
    builder.seal_block(loop_exit);
    Ok(())
}

fn alloc_loop_var(next_var_index: &mut u32) -> u32 {
    let idx = *next_var_index;
    *next_var_index += 1;
    idx
}

fn load_tensor_base_ptr(
    builder: &mut FunctionBuilder,
    layout: &TensorLayout,
    ptr_table: cranelift_codegen::ir::Value,
    ptr_type: cranelift_codegen::ir::Type,
    tensor: GlobalId,
) -> Result<cranelift_codegen::ir::Value, CodegenError> {
    let tidx = layout
        .tensor_index
        .get(&tensor)
        .ok_or(CodegenError::UnknownTensor(tensor))?;
    let ptr_offset = (*tidx as i64) * (ptr_type.bytes() as i64);
    let base_ptr_addr = builder.ins().iadd_imm(ptr_table, ptr_offset);
    Ok(builder
        .ins()
        .load(ptr_type, MemFlags::trusted(), base_ptr_addr, 0))
}

fn load_f32_at_flat(
    builder: &mut FunctionBuilder,
    base_ptr: cranelift_codegen::ir::Value,
    flat_idx: cranelift_codegen::ir::Value,
) -> cranelift_codegen::ir::Value {
    let byte_offset = builder.ins().ishl_imm(flat_idx, 2);
    let addr = builder.ins().iadd(base_ptr, byte_offset);
    builder.ins().load(types::F32, MemFlags::trusted(), addr, 0)
}

fn store_f32_at_flat(
    builder: &mut FunctionBuilder,
    base_ptr: cranelift_codegen::ir::Value,
    flat_idx: cranelift_codegen::ir::Value,
    val: cranelift_codegen::ir::Value,
) {
    let byte_offset = builder.ins().ishl_imm(flat_idx, 2);
    let addr = builder.ins().iadd(base_ptr, byte_offset);
    builder.ins().store(MemFlags::trusted(), val, addr, 0);
}

extern "C" fn wt_expf(x: f32) -> f32 {
    x.exp()
}
extern "C" fn wt_logf(x: f32) -> f32 {
    x.ln()
}
extern "C" fn wt_tanhf(x: f32) -> f32 {
    x.tanh()
}

fn setup_jit_module()
-> Result<(JITModule, HashMap<&'static str, cranelift_module::FuncId>), CodegenError> {
    let mut flag_builder = settings::builder();
    flag_builder
        .set("use_colocated_libcalls", "false")
        .map_err(|e| CodegenError::CodegenError(e.to_string()))?;
    flag_builder
        .set("is_pic", "false")
        .map_err(|e| CodegenError::CodegenError(e.to_string()))?;
    let flags = settings::Flags::new(flag_builder);
    let isa = cranelift_native::builder()
        .map_err(|msg| CodegenError::CodegenError(msg.to_string()))?
        .finish(flags)
        .map_err(|e| CodegenError::CodegenError(e.to_string()))?;

    let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    jit_builder.symbol("wt_expf", wt_expf as *const u8);
    jit_builder.symbol("wt_logf", wt_logf as *const u8);
    jit_builder.symbol("wt_tanhf", wt_tanhf as *const u8);

    let mut module = JITModule::new(jit_builder);
    let math_func_ids = declare_math_functions(&mut module)?;
    Ok((module, math_func_ids))
}

fn declare_math_functions(
    module: &mut JITModule,
) -> Result<HashMap<&'static str, cranelift_module::FuncId>, CodegenError> {
    let mut ids = HashMap::new();
    for name in &["wt_expf", "wt_logf", "wt_tanhf"] {
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(types::F32));
        sig.returns.push(AbiParam::new(types::F32));
        let fid = module.declare_function(name, Linkage::Import, &sig)?;
        ids.insert(*name, fid);
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::milli_graph::ops::MatMul;
    use rand::RngCore;

    fn run_compiled_rank2_matmul(
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), V6CodegenError> {
        let mut rng = wyrand::WyRand::new(77 + (m as u64) + (n as u64));
        let ext_a = GlobalId::new(&mut rng);
        let ext_b = GlobalId::new(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let a = input_map[&ext_a];
        let b = input_map[&ext_b];
        let c = MatMul::push_new(&mut graph, a, b, &mut rng);

        let mut shapes = HashMap::new();
        shapes.insert(a, vec![m, k]);
        shapes.insert(b, vec![k, n]);
        shapes.insert(c, vec![m, n]);

        let (compiled, artifacts) = compile_graph(&graph, &shapes)?;
        assert_eq!(artifacts.schedule.loops.len(), 1);
        assert_eq!(artifacts.schedule.stats.additive_reduction_families, 1);

        let layout = &compiled.layout;
        let mut bufs: Vec<Vec<f32>> = (0..layout.num_buffers).map(|_| Vec::new()).collect();
        let mut rng_a = wyrand::WyRand::new(5001 + m as u64 + k as u64);
        let mut rng_b = wyrand::WyRand::new(6001 + k as u64 + n as u64);
        bufs[layout.tensor_index[&a]] = (0..(m * k))
            .map(|_| (rng_a.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0)
            .collect();
        bufs[layout.tensor_index[&b]] = (0..(k * n))
            .map(|_| (rng_b.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0)
            .collect();
        bufs[layout.tensor_index[&c]] = vec![0.0; m * n];
        let mut ptrs: Vec<*mut f32> = bufs.iter_mut().map(|v| v.as_mut_ptr()).collect();
        unsafe { compiled.execute(&mut ptrs) };

        let mut ref_out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += bufs[layout.tensor_index[&a]][i * k + kk]
                        * bufs[layout.tensor_index[&b]][kk * n + j];
                }
                ref_out[i * n + j] = acc;
            }
        }

        let got = bufs[layout.tensor_index[&c]].clone();
        Ok((got, ref_out))
    }

    #[test]
    fn test_compile_recovered_matmul_kernel() {
        let (got, ref_out) = run_compiled_rank2_matmul(4, 8, 3).expect("v6 compile");
        for i in 0..got.len() {
            assert!((got[i] - ref_out[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn test_compile_recovered_matmul_kernel_larger() {
        let (got, ref_out) = run_compiled_rank2_matmul(80, 96, 96).expect("v6 compile");
        for i in 0..got.len() {
            assert!((got[i] - ref_out[i]).abs() < 2e-3);
        }
    }
}
