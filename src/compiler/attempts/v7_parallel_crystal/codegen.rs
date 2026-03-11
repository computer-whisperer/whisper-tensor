#![allow(clippy::all, dead_code, unreachable_patterns)]
//! Cranelift codegen for v7 planned loop tasks.
//!
//! Current scope:
//! - compile planned additive-reduction loops into JIT code
//! - compile recovered pointwise loops directly from v6 schedule intents
//! - support rank-2 `sum_k(load(a)*load(b))` kernels
//! - emit task-tiled loops directly in Cranelift (no Rust executor hot path)

use super::planner::{ReductionTileTask, TaskPlan, TaskPlannerConfig, plan_from_graph};
use crate::compiler::attempts::v1_scalar_crystal::codegen::{CodegenError, TensorLayout};
use crate::compiler::attempts::v6_schedule_synthesis::synthesis::{
    AccessDimRole, LoopIntent, PipelineArtifacts, RecoveredTensorAccess, ReductionBinOp,
    ReductionTermPattern, ReductionUnaryOp,
};
use crate::graph::GlobalId;
use crate::milli_graph::MilliOpGraph;
use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, StackSlotData, StackSlotKind};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use std::collections::{BTreeSet, HashMap, HashSet};

const REDUCTION_NR: i64 = 8;

#[derive(Debug, thiserror::Error)]
pub enum V7CodegenError {
    #[error(transparent)]
    Plan(#[from] super::planner::V7PlanError),
    #[error(transparent)]
    Codegen(#[from] CodegenError),
    #[error("v7 codegen: {0}")]
    Unsupported(String),
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
struct CompiledAccess {
    tensor: GlobalId,
    output_coeffs: Vec<i64>,
    reduction_coeff: i64,
    base: i64,
    len: usize,
}

#[derive(Debug, Clone)]
struct KernelTask {
    i_start: i64,
    i_end: i64,
    j_start: i64,
    j_end: i64,
    k_start: i64,
    k_end: i64,
}

#[derive(Debug, Clone)]
struct ReductionKernelSpec {
    output_tensor: GlobalId,
    output_shape: [usize; 2],
    access_a: CompiledAccess,
    access_b: CompiledAccess,
    tasks: Vec<KernelTask>,
}

#[derive(Debug, Clone)]
struct PointwiseKernelSpec {
    output_tensor: GlobalId,
    output_shape: Vec<usize>,
    canonical_term: ReductionTermPattern,
    accesses: Vec<CompiledAccess>,
}

#[derive(Debug, Clone)]
enum KernelSpec {
    Reduction(ReductionKernelSpec),
    Pointwise(PointwiseKernelSpec),
}

#[derive(Debug, Clone, Default)]
pub struct CompileCoverage {
    pub schedule_loops: usize,
    pub planned_reduction_loops: usize,
    pub compiled_reduction_loops: usize,
    pub pointwise_loops: usize,
    pub unknown_loops: usize,
}

pub fn compile_graph(
    graph: &MilliOpGraph,
    shapes: &HashMap<GlobalId, Vec<usize>>,
    config: TaskPlannerConfig,
) -> Result<(NativeCompiledGraph, PipelineArtifacts, TaskPlan), V7CodegenError> {
    let (artifacts, plan) = plan_from_graph(graph, shapes, config)?;
    let compiled = compile_plan(&artifacts, &plan, shapes)?;
    Ok((compiled, artifacts, plan))
}

pub fn compile_plan(
    artifacts: &PipelineArtifacts,
    plan: &TaskPlan,
    shapes: &HashMap<GlobalId, Vec<usize>>,
) -> Result<NativeCompiledGraph, V7CodegenError> {
    let coverage = analyze_coverage(artifacts, plan);
    if coverage.unknown_loops > 0 {
        return Err(V7CodegenError::Unsupported(format!(
            "full-graph compile requires unknown-loop support (unknown={})",
            coverage.unknown_loops
        )));
    }
    let specs = build_full_kernel_specs(artifacts, plan)?;
    let layout = TensorLayout::from_shapes(shapes);
    compile_kernel(&specs, &layout).map_err(V7CodegenError::from)
}

pub fn compile_plan_reduction_only(
    artifacts: &PipelineArtifacts,
    plan: &TaskPlan,
    shapes: &HashMap<GlobalId, Vec<usize>>,
) -> Result<(NativeCompiledGraph, CompileCoverage), V7CodegenError> {
    let specs = build_reduction_kernel_specs(artifacts, plan)?
        .into_iter()
        .map(KernelSpec::Reduction)
        .collect::<Vec<_>>();
    let coverage = analyze_coverage(artifacts, plan);
    let layout = TensorLayout::from_shapes(shapes);
    let compiled = compile_kernel(&specs, &layout).map_err(V7CodegenError::from)?;
    Ok((compiled, coverage))
}

fn analyze_coverage(artifacts: &PipelineArtifacts, plan: &TaskPlan) -> CompileCoverage {
    let mut out = CompileCoverage {
        schedule_loops: artifacts.schedule.loops.len(),
        ..CompileCoverage::default()
    };
    for recovered in &artifacts.schedule.loops {
        match recovered.intent {
            LoopIntent::Pointwise(_) => out.pointwise_loops += 1,
            LoopIntent::AdditiveReduction(_) => {}
            LoopIntent::Unknown => out.unknown_loops += 1,
        }
    }

    let mut seen = std::collections::HashSet::new();
    for task in &plan.tasks {
        if seen.insert(task.loop_index) {
            out.planned_reduction_loops += 1;
        }
    }
    out.compiled_reduction_loops = out.planned_reduction_loops;
    out
}

fn build_full_kernel_specs(
    artifacts: &PipelineArtifacts,
    plan: &TaskPlan,
) -> Result<Vec<KernelSpec>, V7CodegenError> {
    let mut grouped_reduction_tasks = group_reduction_tasks_by_loop(plan);
    let loop_order = derive_loop_execution_order(artifacts)?;
    let mut specs = Vec::with_capacity(loop_order.len());

    for loop_index in loop_order {
        let recovered = artifacts.schedule.loops.get(loop_index).ok_or_else(|| {
            V7CodegenError::Unsupported(format!("missing recovered loop {loop_index}"))
        })?;
        match &recovered.intent {
            LoopIntent::AdditiveReduction(_) => {
                let loop_tasks = grouped_reduction_tasks
                    .remove(&loop_index)
                    .unwrap_or_default();
                if loop_tasks.is_empty() {
                    return Err(V7CodegenError::Unsupported(format!(
                        "missing planned tasks for additive-reduction loop {loop_index}"
                    )));
                }
                let reduction =
                    build_reduction_kernel_spec_for_loop(artifacts, loop_index, &loop_tasks)?;
                specs.push(KernelSpec::Reduction(reduction));
            }
            LoopIntent::Pointwise(_) => {
                let pointwise = build_pointwise_kernel_spec_for_loop(artifacts, loop_index)?;
                specs.push(KernelSpec::Pointwise(pointwise));
            }
            LoopIntent::Unknown => {
                return Err(V7CodegenError::Unsupported(format!(
                    "unknown loop {loop_index} is not compilable"
                )));
            }
        }
    }

    if let Some((leftover_loop, _)) = grouped_reduction_tasks.into_iter().next() {
        return Err(V7CodegenError::Unsupported(format!(
            "planned tasks reference missing loop index {leftover_loop}"
        )));
    }

    Ok(specs)
}

fn group_reduction_tasks_by_loop(plan: &TaskPlan) -> HashMap<usize, Vec<&ReductionTileTask>> {
    let mut by_loop = HashMap::<usize, Vec<&ReductionTileTask>>::new();
    for task in &plan.tasks {
        by_loop.entry(task.loop_index).or_default().push(task);
    }
    by_loop
}

fn derive_loop_execution_order(
    artifacts: &PipelineArtifacts,
) -> Result<Vec<usize>, V7CodegenError> {
    let loop_count = artifacts.schedule.loops.len();
    let mut output_to_loop = HashMap::<GlobalId, usize>::new();
    for (loop_index, recovered) in artifacts.schedule.loops.iter().enumerate() {
        if let Some(prev_index) = output_to_loop.insert(recovered.output_tensor, loop_index) {
            return Err(V7CodegenError::Unsupported(format!(
                "multiple recovered loops write tensor {} (loop {} and {})",
                recovered.output_tensor, prev_index, loop_index
            )));
        }
    }

    let mut indegree = vec![0usize; loop_count];
    let mut consumers = vec![Vec::<usize>::new(); loop_count];
    for (loop_index, recovered) in artifacts.schedule.loops.iter().enumerate() {
        let mut deps = HashSet::new();
        for tensor in &recovered.load_tensors {
            if let Some(&producer_loop) = output_to_loop.get(tensor) {
                if producer_loop != loop_index && deps.insert(producer_loop) {
                    indegree[loop_index] = indegree[loop_index].saturating_add(1);
                    consumers[producer_loop].push(loop_index);
                }
            }
        }
    }

    let mut ready = BTreeSet::new();
    for (loop_index, &deg) in indegree.iter().enumerate() {
        if deg == 0 {
            ready.insert(loop_index);
        }
    }

    let mut order = Vec::with_capacity(loop_count);
    while let Some(&loop_index) = ready.first() {
        ready.remove(&loop_index);
        order.push(loop_index);
        for &consumer in &consumers[loop_index] {
            if indegree[consumer] == 0 {
                continue;
            }
            indegree[consumer] -= 1;
            if indegree[consumer] == 0 {
                ready.insert(consumer);
            }
        }
    }

    if order.len() != loop_count {
        return Err(V7CodegenError::Unsupported(
            "recovered loop dependency cycle detected".to_string(),
        ));
    }

    Ok(order)
}

fn build_reduction_kernel_specs(
    artifacts: &PipelineArtifacts,
    plan: &TaskPlan,
) -> Result<Vec<ReductionKernelSpec>, V7CodegenError> {
    let first_task = plan
        .tasks
        .first()
        .ok_or_else(|| V7CodegenError::Unsupported("empty task plan".to_string()))?;

    let mut loop_order = Vec::<usize>::new();
    let mut by_loop = HashMap::<usize, Vec<&ReductionTileTask>>::new();
    for task in &plan.tasks {
        if !by_loop.contains_key(&task.loop_index) {
            loop_order.push(task.loop_index);
        }
        by_loop.entry(task.loop_index).or_default().push(task);
    }

    if loop_order.is_empty() {
        return Err(V7CodegenError::Unsupported(format!(
            "no tasks found for first loop index {}",
            first_task.loop_index
        )));
    }

    let mut specs = Vec::with_capacity(loop_order.len());
    for loop_index in loop_order {
        let loop_tasks = by_loop.remove(&loop_index).unwrap_or_default();
        let spec = build_reduction_kernel_spec_for_loop(artifacts, loop_index, &loop_tasks)?;
        specs.push(spec);
    }
    Ok(specs)
}

fn build_reduction_kernel_spec_for_loop(
    artifacts: &PipelineArtifacts,
    loop_index: usize,
    loop_tasks: &[&ReductionTileTask],
) -> Result<ReductionKernelSpec, V7CodegenError> {
    let recovered = artifacts.schedule.loops.get(loop_index).ok_or_else(|| {
        V7CodegenError::Unsupported(format!("missing recovered loop {loop_index}"))
    })?;
    let LoopIntent::AdditiveReduction(reduction) = &recovered.intent else {
        return Err(V7CodegenError::Unsupported(
            "only additive-reduction loops are supported".to_string(),
        ));
    };

    if recovered.output_shape.len() != 2 {
        return Err(V7CodegenError::Unsupported(format!(
            "only rank-2 outputs are supported (got rank {})",
            recovered.output_shape.len()
        )));
    }
    let output_shape = [recovered.output_shape[0], recovered.output_shape[1]];

    let (slot0, slot1) = extract_mul2_slots(&reduction.canonical_term).ok_or_else(|| {
        V7CodegenError::Unsupported(
            "only Mul(Load(_), Load(_)) canonical terms are supported".to_string(),
        )
    })?;

    let compiled_accesses = reduction
        .accesses
        .iter()
        .map(|access| compile_access(access, recovered.output_shape.len(), loop_index))
        .collect::<Result<Vec<_>, _>>()?;
    if slot0 >= compiled_accesses.len() || slot1 >= compiled_accesses.len() {
        return Err(V7CodegenError::Unsupported(
            "canonical term load slot out of range".to_string(),
        ));
    }

    let access0 = compiled_accesses[slot0].clone();
    let access1 = compiled_accesses[slot1].clone();
    let (access_a, access_b) = if is_row_accum_a(&access0) && is_row_accum_b(&access1) {
        (access0, access1)
    } else if is_row_accum_a(&access1) && is_row_accum_b(&access0) {
        (access1, access0)
    } else {
        return Err(V7CodegenError::Unsupported(
            "unsupported affine layout for row-accumulation rank2 kernel".to_string(),
        ));
    };

    let mut tasks = Vec::new();
    for task in loop_tasks {
        if task.output_tensor != recovered.output_tensor
            || task.output_shape != recovered.output_shape
        {
            return Err(V7CodegenError::Unsupported(
                "task output tensor/shape mismatch with recovered loop".to_string(),
            ));
        }
        if task.output_start.len() != 2 || task.output_end.len() != 2 {
            return Err(V7CodegenError::Unsupported(
                "only rank-2 tasks are supported".to_string(),
            ));
        }
        validate_task_ranges(task, &output_shape)?;
        validate_task_affine_ranges(task, &access_a)?;
        validate_task_affine_ranges(task, &access_b)?;
        tasks.push(KernelTask {
            i_start: task.output_start[0] as i64,
            i_end: task.output_end[0] as i64,
            j_start: task.output_start[1] as i64,
            j_end: task.output_end[1] as i64,
            k_start: task.reduction_start as i64,
            k_end: task.reduction_end as i64,
        });
    }
    let tasks = coalesce_kernel_tasks(tasks);

    Ok(ReductionKernelSpec {
        output_tensor: recovered.output_tensor,
        output_shape,
        access_a,
        access_b,
        tasks,
    })
}

fn coalesce_kernel_tasks(tasks: Vec<KernelTask>) -> Vec<KernelTask> {
    tasks
}

fn build_pointwise_kernel_spec_for_loop(
    artifacts: &PipelineArtifacts,
    loop_index: usize,
) -> Result<PointwiseKernelSpec, V7CodegenError> {
    let recovered = artifacts.schedule.loops.get(loop_index).ok_or_else(|| {
        V7CodegenError::Unsupported(format!("missing recovered loop {loop_index}"))
    })?;
    let LoopIntent::Pointwise(pointwise) = &recovered.intent else {
        return Err(V7CodegenError::Unsupported(
            "only pointwise loops are supported in pointwise builder".to_string(),
        ));
    };

    let output_rank = recovered.output_shape.len();
    let accesses = pointwise
        .accesses
        .iter()
        .map(|access| compile_access(access, output_rank, loop_index))
        .collect::<Result<Vec<_>, _>>()?;

    for access in &accesses {
        validate_pointwise_access_bounds(&recovered.output_shape, access)?;
    }

    Ok(PointwiseKernelSpec {
        output_tensor: recovered.output_tensor,
        output_shape: recovered.output_shape.clone(),
        canonical_term: pointwise.canonical_term.clone(),
        accesses,
    })
}

fn is_row_accum_a(access: &CompiledAccess) -> bool {
    access.output_coeffs.len() >= 2 && access.output_coeffs[1] == 0
}

fn is_row_accum_b(access: &CompiledAccess) -> bool {
    access.output_coeffs.len() >= 2 && access.output_coeffs[0] == 0 && access.output_coeffs[1] == 1
}

fn extract_mul2_slots(term: &ReductionTermPattern) -> Option<(usize, usize)> {
    let ReductionTermPattern::Bin { op, a, b } = term else {
        return None;
    };
    if *op != ReductionBinOp::Mul {
        return None;
    }
    match (&**a, &**b) {
        (ReductionTermPattern::Load(slot_a), ReductionTermPattern::Load(slot_b)) => {
            Some((*slot_a, *slot_b))
        }
        _ => None,
    }
}

fn validate_task_ranges(
    task: &ReductionTileTask,
    output_shape: &[usize; 2],
) -> Result<(), V7CodegenError> {
    for (axis, dim) in output_shape.iter().enumerate() {
        let start = task.output_start[axis];
        let end = task.output_end[axis];
        if start >= end || end > *dim {
            return Err(V7CodegenError::Unsupported(format!(
                "invalid task output range axis {axis}: [{start}, {end}) for dim {dim}"
            )));
        }
    }
    if task.reduction_start >= task.reduction_end {
        return Err(V7CodegenError::Unsupported(
            "empty reduction range is not supported".to_string(),
        ));
    }
    Ok(())
}

fn validate_task_affine_ranges(
    task: &ReductionTileTask,
    access: &CompiledAccess,
) -> Result<(), V7CodegenError> {
    let (min_flat, max_flat) = affine_range_for_task(task, access);
    if min_flat < 0 || max_flat >= access.len as i64 {
        return Err(V7CodegenError::Unsupported(format!(
            "access out of bounds for tensor {} in task range: [{min_flat}, {max_flat}] len={}",
            access.tensor, access.len
        )));
    }
    Ok(())
}

fn validate_pointwise_access_bounds(
    output_shape: &[usize],
    access: &CompiledAccess,
) -> Result<(), V7CodegenError> {
    if access.reduction_coeff != 0 {
        return Err(V7CodegenError::Unsupported(format!(
            "pointwise access for tensor {} unexpectedly uses reduction stride {}",
            access.tensor, access.reduction_coeff
        )));
    }

    let mut min = access.base;
    let mut max = access.base;
    for (axis, coeff) in access.output_coeffs.iter().enumerate() {
        let lo = 0i64;
        let hi = output_shape
            .get(axis)
            .copied()
            .unwrap_or(1)
            .saturating_sub(1) as i64;
        if *coeff >= 0 {
            min = min.saturating_add(coeff.saturating_mul(lo));
            max = max.saturating_add(coeff.saturating_mul(hi));
        } else {
            min = min.saturating_add(coeff.saturating_mul(hi));
            max = max.saturating_add(coeff.saturating_mul(lo));
        }
    }

    if min < 0 || max >= access.len as i64 {
        return Err(V7CodegenError::Unsupported(format!(
            "pointwise access out of bounds for tensor {}: [{min}, {max}] len={}",
            access.tensor, access.len
        )));
    }
    Ok(())
}

fn affine_range_for_task(task: &ReductionTileTask, access: &CompiledAccess) -> (i64, i64) {
    let mut min = access.base;
    let mut max = access.base;
    for (axis, coeff) in access.output_coeffs.iter().enumerate() {
        let lo = task.output_start.get(axis).copied().unwrap_or(0) as i64;
        let hi = task
            .output_end
            .get(axis)
            .copied()
            .unwrap_or(1)
            .saturating_sub(1) as i64;
        if *coeff >= 0 {
            min = min.saturating_add(coeff.saturating_mul(lo));
            max = max.saturating_add(coeff.saturating_mul(hi));
        } else {
            min = min.saturating_add(coeff.saturating_mul(hi));
            max = max.saturating_add(coeff.saturating_mul(lo));
        }
    }
    let k_lo = task.reduction_start as i64;
    let k_hi = task.reduction_end.saturating_sub(1) as i64;
    if access.reduction_coeff >= 0 {
        min = min.saturating_add(access.reduction_coeff.saturating_mul(k_lo));
        max = max.saturating_add(access.reduction_coeff.saturating_mul(k_hi));
    } else {
        min = min.saturating_add(access.reduction_coeff.saturating_mul(k_hi));
        max = max.saturating_add(access.reduction_coeff.saturating_mul(k_lo));
    }
    (min, max)
}

fn compile_access(
    access: &RecoveredTensorAccess,
    output_rank: usize,
    loop_index: usize,
) -> Result<CompiledAccess, V7CodegenError> {
    let mut dim_strides = vec![0i64; access.tensor_shape.len()];
    let mut running = 1i64;
    for axis in (0..access.tensor_shape.len()).rev() {
        dim_strides[axis] = running;
        running = running.saturating_mul(access.tensor_shape[axis] as i64);
    }

    let mut output_coeffs = vec![0i64; output_rank];
    let mut reduction_coeff = 0i64;
    let mut base = 0i64;

    for (axis, role) in access.dim_roles.iter().enumerate() {
        let dim_stride = *dim_strides.get(axis).unwrap_or(&0);
        match role {
            AccessDimRole::Constant { value } => {
                base = base.saturating_add((*value as i64).saturating_mul(dim_stride));
            }
            AccessDimRole::OutputAxis {
                axis,
                stride,
                offset,
            } => {
                if *axis < output_coeffs.len() {
                    output_coeffs[*axis] = output_coeffs[*axis]
                        .saturating_add((*stride as i64).saturating_mul(dim_stride));
                }
                base = base.saturating_add((*offset as i64).saturating_mul(dim_stride));
            }
            AccessDimRole::ReductionAxis { stride, offset } => {
                reduction_coeff =
                    reduction_coeff.saturating_add((*stride as i64).saturating_mul(dim_stride));
                base = base.saturating_add((*offset as i64).saturating_mul(dim_stride));
            }
            AccessDimRole::AffineMixed {
                output_strides,
                reduction_stride,
                offset,
            } => {
                for (axis, stride) in output_strides {
                    if *axis < output_coeffs.len() {
                        output_coeffs[*axis] = output_coeffs[*axis]
                            .saturating_add((*stride as i64).saturating_mul(dim_stride));
                    }
                }
                reduction_coeff = reduction_coeff
                    .saturating_add((*reduction_stride as i64).saturating_mul(dim_stride));
                base = base.saturating_add((*offset as i64).saturating_mul(dim_stride));
            }
            AccessDimRole::Unknown => {
                return Err(V7CodegenError::Unsupported(format!(
                    "unknown access role in loop {loop_index}"
                )));
            }
        }
    }

    Ok(CompiledAccess {
        tensor: access.tensor,
        output_coeffs,
        reduction_coeff,
        base,
        len: shape_elements(&access.tensor_shape),
    })
}

fn shape_elements(shape: &[usize]) -> usize {
    shape.iter().copied().fold(1usize, usize::saturating_mul)
}

fn compile_kernel(
    specs: &[KernelSpec],
    layout: &TensorLayout,
) -> Result<NativeCompiledGraph, CodegenError> {
    let (mut module, math_func_ids) = setup_jit_module()?;
    let ptr_type = module.isa().pointer_type();

    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(ptr_type));
    let func_id = module.declare_function("wt_v7_task_kernel", Linkage::Local, &sig)?;

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

    let mut next_var_index = 0u32;
    for spec in specs {
        match spec {
            KernelSpec::Reduction(spec) => {
                let output_ptr = load_tensor_base_ptr(
                    &mut builder,
                    layout,
                    ptr_table,
                    ptr_type,
                    spec.output_tensor,
                )?;
                let a_ptr = load_tensor_base_ptr(
                    &mut builder,
                    layout,
                    ptr_table,
                    ptr_type,
                    spec.access_a.tensor,
                )?;
                let b_ptr = load_tensor_base_ptr(
                    &mut builder,
                    layout,
                    ptr_table,
                    ptr_type,
                    spec.access_b.tensor,
                )?;

                let max_k_span = spec
                    .tasks
                    .iter()
                    .map(|task| {
                        if task.k_end <= task.k_start {
                            0usize
                        } else {
                            (task.k_end - task.k_start) as usize
                        }
                    })
                    .max()
                    .unwrap_or(0);
                if max_k_span == 0 {
                    return Err(CodegenError::CodegenError(
                        "v7 reduction task has empty reduction span".to_string(),
                    ));
                }
                let panel_elems =
                    max_k_span
                        .checked_mul(REDUCTION_NR as usize)
                        .ok_or_else(|| {
                            CodegenError::CodegenError(
                                "v7 packed panel element count overflow".to_string(),
                            )
                        })?;
                let panel_bytes = panel_elems.checked_mul(4).ok_or_else(|| {
                    CodegenError::CodegenError("v7 packed panel byte count overflow".to_string())
                })?;
                let panel_bytes_u32 = u32::try_from(panel_bytes).map_err(|_| {
                    CodegenError::CodegenError(format!(
                        "v7 packed panel is too large for a stack slot: {panel_bytes} bytes"
                    ))
                })?;
                let panel_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    panel_bytes_u32,
                    0,
                ));
                let b_panel_ptr = builder.ins().stack_addr(ptr_type, panel_slot, 0);

                for task in &spec.tasks {
                    emit_reduction_task(
                        &mut builder,
                        &mut next_var_index,
                        spec,
                        task,
                        output_ptr,
                        a_ptr,
                        b_ptr,
                        b_panel_ptr,
                    )?;
                }
            }
            KernelSpec::Pointwise(spec) => {
                let output_ptr = load_tensor_base_ptr(
                    &mut builder,
                    layout,
                    ptr_table,
                    ptr_type,
                    spec.output_tensor,
                )?;
                let mut input_ptrs = Vec::with_capacity(spec.accesses.len());
                for access in &spec.accesses {
                    let ptr = load_tensor_base_ptr(
                        &mut builder,
                        layout,
                        ptr_table,
                        ptr_type,
                        access.tensor,
                    )?;
                    input_ptrs.push(ptr);
                }

                emit_pointwise_kernel(
                    &mut builder,
                    &mut next_var_index,
                    spec,
                    output_ptr,
                    &input_ptrs,
                    &math_refs,
                )?;
            }
        }
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

#[allow(clippy::too_many_arguments)]
fn emit_reduction_task(
    builder: &mut FunctionBuilder,
    next_var_index: &mut u32,
    spec: &ReductionKernelSpec,
    task: &KernelTask,
    output_ptr: cranelift_codegen::ir::Value,
    a_ptr: cranelift_codegen::ir::Value,
    b_ptr: cranelift_codegen::ir::Value,
    b_panel_ptr: cranelift_codegen::ir::Value,
) -> Result<(), CodegenError> {
    const NR: i64 = REDUCTION_NR;
    let n = spec.output_shape[1] as i64;
    let k_start = task.k_start;
    let j_main = task.j_start + ((task.j_end - task.j_start) / NR) * NR;
    let a_row = spec.access_a.output_coeffs.first().copied().unwrap_or(0);
    let b_row = spec.access_b.output_coeffs.first().copied().unwrap_or(0);
    if b_row != 0 {
        return Err(CodegenError::CodegenError(format!(
            "v7 reduction expects B row coeff 0, got {b_row}"
        )));
    }

    let iv_i = alloc_loop_var(next_var_index);
    let iv_i_tail = alloc_loop_var(next_var_index);
    let iv_j_main = alloc_loop_var(next_var_index);
    let iv_pack_k = alloc_loop_var(next_var_index);
    let iv_k_main = alloc_loop_var(next_var_index);
    let iv_j_tail = alloc_loop_var(next_var_index);
    let iv_k_tail = alloc_loop_var(next_var_index);

    let acc_vars: Vec<Variable> = (0..NR)
        .map(|_| {
            let var = Variable::from_u32(alloc_loop_var(next_var_index));
            builder.declare_var(var, types::F32);
            var
        })
        .collect();
    let acc_tail = Variable::from_u32(alloc_loop_var(next_var_index));
    builder.declare_var(acc_tail, types::F32);

    if j_main > task.j_start {
        emit_for_loop(
            builder,
            iv_j_main,
            task.j_start,
            j_main,
            NR,
            |builder, col_j0| {
                // Pack B[k_start..k_end, col_j0..col_j0+NR) into a contiguous panel.
                emit_for_loop(
                    builder,
                    iv_pack_k,
                    k_start,
                    task.k_end,
                    1,
                    |builder, red_k| {
                        let b_base_const = builder.ins().iconst(types::I64, spec.access_b.base);
                        let b_k_base = add_scaled_i64(
                            builder,
                            b_base_const,
                            red_k,
                            spec.access_b.reduction_coeff,
                        );
                        let b_base = builder.ins().iadd(b_k_base, col_j0);
                        let panel_k = builder.ins().iadd_imm(red_k, -k_start);
                        let panel_base = if NR == 1 {
                            panel_k
                        } else {
                            builder.ins().imul_imm(panel_k, NR)
                        };

                        for lane in 0..NR {
                            let b_idx = if lane == 0 {
                                b_base
                            } else {
                                builder.ins().iadd_imm(b_base, lane)
                            };
                            let bv = load_f32_at_flat(builder, b_ptr, b_idx);
                            let panel_idx = if lane == 0 {
                                panel_base
                            } else {
                                builder.ins().iadd_imm(panel_base, lane)
                            };
                            store_f32_at_flat(builder, b_panel_ptr, panel_idx, bv);
                        }
                        Ok(())
                    },
                )?;

                emit_for_loop(
                    builder,
                    iv_i,
                    task.i_start,
                    task.i_end,
                    1,
                    |builder, row_i| {
                        let mut a_row_base = builder.ins().iconst(types::I64, spec.access_a.base);
                        a_row_base = add_scaled_i64(builder, a_row_base, row_i, a_row);
                        let zero = builder.ins().f32const(0.0);
                        for &var in &acc_vars {
                            builder.def_var(var, zero);
                        }

                        emit_for_loop(
                            builder,
                            iv_k_main,
                            k_start,
                            task.k_end,
                            1,
                            |builder, red_k| {
                                let idx_a = add_scaled_i64(
                                    builder,
                                    a_row_base,
                                    red_k,
                                    spec.access_a.reduction_coeff,
                                );
                                let av = load_f32_at_flat(builder, a_ptr, idx_a);
                                let panel_k = builder.ins().iadd_imm(red_k, -k_start);
                                let panel_base = if NR == 1 {
                                    panel_k
                                } else {
                                    builder.ins().imul_imm(panel_k, NR)
                                };

                                for (lane, &var) in acc_vars.iter().enumerate() {
                                    let panel_idx = if lane == 0 {
                                        panel_base
                                    } else {
                                        builder.ins().iadd_imm(panel_base, lane as i64)
                                    };
                                    let bv = load_f32_at_flat(builder, b_panel_ptr, panel_idx);
                                    let cur = builder.use_var(var);
                                    let prod = builder.ins().fmul(av, bv);
                                    let next = builder.ins().fadd(cur, prod);
                                    builder.def_var(var, next);
                                }
                                Ok(())
                            },
                        )?;

                        for (lane, &var) in acc_vars.iter().enumerate() {
                            let col_j = if lane == 0 {
                                col_j0
                            } else {
                                builder.ins().iadd_imm(col_j0, lane as i64)
                            };
                            let out_flat = output_flat_2d(builder, row_i, col_j, n);
                            let acc = builder.use_var(var);
                            store_f32_at_flat(builder, output_ptr, out_flat, acc);
                        }
                        Ok(())
                    },
                )?;
                Ok(())
            },
        )?;
    }

    if j_main < task.j_end {
        emit_for_loop(
            builder,
            iv_i_tail,
            task.i_start,
            task.i_end,
            1,
            |builder, row_i| {
                let mut a_row_base = builder.ins().iconst(types::I64, spec.access_a.base);
                a_row_base = add_scaled_i64(builder, a_row_base, row_i, a_row);
                let mut b_row_base = builder.ins().iconst(types::I64, spec.access_b.base);
                b_row_base = add_scaled_i64(builder, b_row_base, row_i, b_row);

                emit_for_loop(
                    builder,
                    iv_j_tail,
                    j_main,
                    task.j_end,
                    1,
                    |builder, col_j| {
                        let zero = builder.ins().f32const(0.0);
                        builder.def_var(acc_tail, zero);

                        emit_for_loop(
                            builder,
                            iv_k_tail,
                            task.k_start,
                            task.k_end,
                            1,
                            |builder, red_k| {
                                let idx_a = add_scaled_i64(
                                    builder,
                                    a_row_base,
                                    red_k,
                                    spec.access_a.reduction_coeff,
                                );
                                let av = load_f32_at_flat(builder, a_ptr, idx_a);

                                let b_k_base = add_scaled_i64(
                                    builder,
                                    b_row_base,
                                    red_k,
                                    spec.access_b.reduction_coeff,
                                );
                                let b_idx = builder.ins().iadd(b_k_base, col_j);
                                let bv = load_f32_at_flat(builder, b_ptr, b_idx);

                                let cur = builder.use_var(acc_tail);
                                let prod = builder.ins().fmul(av, bv);
                                let next = builder.ins().fadd(cur, prod);
                                builder.def_var(acc_tail, next);
                                Ok(())
                            },
                        )?;

                        let out_flat = output_flat_2d(builder, row_i, col_j, n);
                        let acc = builder.use_var(acc_tail);
                        store_f32_at_flat(builder, output_ptr, out_flat, acc);
                        Ok(())
                    },
                )?;
                Ok(())
            },
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn emit_pointwise_kernel(
    builder: &mut FunctionBuilder,
    next_var_index: &mut u32,
    spec: &PointwiseKernelSpec,
    output_ptr: cranelift_codegen::ir::Value,
    input_ptrs: &[cranelift_codegen::ir::Value],
    math_refs: &HashMap<&str, cranelift_codegen::ir::FuncRef>,
) -> Result<(), CodegenError> {
    let rank = spec.output_shape.len();
    let mut output_axis_vars = vec![None; rank];
    emit_pointwise_axes(
        builder,
        next_var_index,
        spec,
        0,
        &mut output_axis_vars,
        output_ptr,
        input_ptrs,
        math_refs,
    )
}

#[allow(clippy::too_many_arguments)]
fn emit_pointwise_axes(
    builder: &mut FunctionBuilder,
    next_var_index: &mut u32,
    spec: &PointwiseKernelSpec,
    depth: usize,
    output_axis_vars: &mut [Option<cranelift_codegen::ir::Value>],
    output_ptr: cranelift_codegen::ir::Value,
    input_ptrs: &[cranelift_codegen::ir::Value],
    math_refs: &HashMap<&str, cranelift_codegen::ir::FuncRef>,
) -> Result<(), CodegenError> {
    if depth == spec.output_shape.len() {
        return emit_pointwise_store(
            builder,
            spec,
            output_axis_vars,
            output_ptr,
            input_ptrs,
            math_refs,
        );
    }

    let end = spec.output_shape[depth] as i64;
    let iv_idx = alloc_loop_var(next_var_index);
    emit_for_loop(builder, iv_idx, 0, end, 1, |builder, iv| {
        output_axis_vars[depth] = Some(iv);
        emit_pointwise_axes(
            builder,
            next_var_index,
            spec,
            depth + 1,
            output_axis_vars,
            output_ptr,
            input_ptrs,
            math_refs,
        )?;
        output_axis_vars[depth] = None;
        Ok(())
    })
}

fn emit_pointwise_store(
    builder: &mut FunctionBuilder,
    spec: &PointwiseKernelSpec,
    output_axis_vars: &[Option<cranelift_codegen::ir::Value>],
    output_ptr: cranelift_codegen::ir::Value,
    input_ptrs: &[cranelift_codegen::ir::Value],
    math_refs: &HashMap<&str, cranelift_codegen::ir::FuncRef>,
) -> Result<(), CodegenError> {
    let mut load_vals = Vec::with_capacity(spec.accesses.len());
    for (access, &base_ptr) in spec.accesses.iter().zip(input_ptrs.iter()) {
        let flat = emit_flat_index_from_compiled_access(builder, access, output_axis_vars)?;
        let val = load_f32_at_flat(builder, base_ptr, flat);
        load_vals.push(val);
    }

    let out_flat = emit_output_flat_index(builder, &spec.output_shape, output_axis_vars)?;
    let out_val = emit_term_expr(builder, &spec.canonical_term, &load_vals, math_refs)?;
    store_f32_at_flat(builder, output_ptr, out_flat, out_val);
    Ok(())
}

fn output_flat_2d(
    builder: &mut FunctionBuilder,
    row: cranelift_codegen::ir::Value,
    col: cranelift_codegen::ir::Value,
    n: i64,
) -> cranelift_codegen::ir::Value {
    let row_base = if n == 1 {
        row
    } else {
        builder.ins().imul_imm(row, n)
    };
    builder.ins().iadd(row_base, col)
}

fn add_scaled_i64(
    builder: &mut FunctionBuilder,
    base: cranelift_codegen::ir::Value,
    iv: cranelift_codegen::ir::Value,
    stride: i64,
) -> cranelift_codegen::ir::Value {
    match stride {
        0 => base,
        1 => builder.ins().iadd(base, iv),
        -1 => {
            let neg = builder.ins().ineg(iv);
            builder.ins().iadd(base, neg)
        }
        _ => {
            let scaled = builder.ins().imul_imm(iv, stride);
            builder.ins().iadd(base, scaled)
        }
    }
}

fn emit_output_flat_index(
    builder: &mut FunctionBuilder,
    shape: &[usize],
    output_axis_vars: &[Option<cranelift_codegen::ir::Value>],
) -> Result<cranelift_codegen::ir::Value, CodegenError> {
    let mut flat = builder.ins().iconst(types::I64, 0);
    for (axis, &dim) in shape.iter().enumerate() {
        let idx = output_axis_vars.get(axis).and_then(|v| *v).ok_or_else(|| {
            CodegenError::CodegenError(format!("v7: missing output axis value {}", axis))
        })?;
        flat = if dim == 1 {
            flat
        } else {
            builder.ins().imul_imm(flat, dim as i64)
        };
        flat = builder.ins().iadd(flat, idx);
    }
    Ok(flat)
}

fn emit_flat_index_from_compiled_access(
    builder: &mut FunctionBuilder,
    access: &CompiledAccess,
    output_axis_vars: &[Option<cranelift_codegen::ir::Value>],
) -> Result<cranelift_codegen::ir::Value, CodegenError> {
    if access.reduction_coeff != 0 {
        return Err(CodegenError::CodegenError(format!(
            "v7 pointwise: access tensor {} has non-zero reduction coeff {}",
            access.tensor, access.reduction_coeff
        )));
    }

    let mut flat = builder.ins().iconst(types::I64, access.base);
    for (axis, &coeff) in access.output_coeffs.iter().enumerate() {
        let axis_val = output_axis_vars.get(axis).and_then(|v| *v).ok_or_else(|| {
            CodegenError::CodegenError(format!("v7: missing output axis value {}", axis))
        })?;
        flat = add_scaled_i64(builder, flat, axis_val, coeff);
    }
    Ok(flat)
}

fn emit_term_expr(
    builder: &mut FunctionBuilder,
    term: &ReductionTermPattern,
    load_vals: &[cranelift_codegen::ir::Value],
    math_refs: &HashMap<&str, cranelift_codegen::ir::FuncRef>,
) -> Result<cranelift_codegen::ir::Value, CodegenError> {
    Ok(match term {
        ReductionTermPattern::Load(i) => *load_vals.get(*i).ok_or_else(|| {
            CodegenError::CodegenError(format!("v7: term load index {} out of range", i))
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
    use crate::milli_graph::ops::{MatMul, SimpleBinary, SimpleUnaryOp};
    use rand::RngCore;

    fn run_compiled_rank2_matmul(
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), V7CodegenError> {
        let mut rng = wyrand::WyRand::new(9100 + m as u64 + n as u64);
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

        let (compiled, _artifacts, plan) = compile_graph(
            &graph,
            &shapes,
            TaskPlannerConfig {
                min_tile_elements: 512,
                max_tile_elements: 4096,
            },
        )?;
        assert!(plan.stats.total_tasks > 0);

        let layout = &compiled.layout;
        let mut bufs: Vec<Vec<f32>> = (0..layout.num_buffers).map(|_| Vec::new()).collect();
        let mut rng_a = wyrand::WyRand::new(9200 + m as u64 + k as u64);
        let mut rng_b = wyrand::WyRand::new(9300 + k as u64 + n as u64);

        for (id, &size) in &layout.tensor_sizes {
            let idx = layout.tensor_index[id];
            let data = if *id == a {
                (0..size)
                    .map(|_| (rng_a.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0)
                    .collect::<Vec<_>>()
            } else if *id == b {
                (0..size)
                    .map(|_| (rng_b.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0)
                    .collect::<Vec<_>>()
            } else {
                vec![0.0f32; size]
            };
            bufs[idx] = data;
        }

        let a_data = bufs[layout.tensor_index[&a]].clone();
        let b_data = bufs[layout.tensor_index[&b]].clone();
        let mut ptrs: Vec<*mut f32> = bufs.iter_mut().map(|v| v.as_mut_ptr()).collect();
        unsafe { compiled.execute(&mut ptrs) };
        let got = bufs[layout.tensor_index[&c]].clone();

        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += a_data[i * k + kk] * b_data[kk * n + j];
                }
                expected[i * n + j] = acc;
            }
        }

        Ok((got, expected))
    }

    fn run_compiled_two_rank2_matmuls(
        m: usize,
        k: usize,
        n1: usize,
        n2: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), V7CodegenError> {
        let mut rng = wyrand::WyRand::new(9800 + m as u64 + n1 as u64 + n2 as u64);
        let ext_a = GlobalId::new(&mut rng);
        let ext_b1 = GlobalId::new(&mut rng);
        let ext_b2 = GlobalId::new(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b1, ext_b2], &mut rng);
        let a = input_map[&ext_a];
        let b1 = input_map[&ext_b1];
        let b2 = input_map[&ext_b2];
        let c1 = MatMul::push_new(&mut graph, a, b1, &mut rng);
        let c2 = MatMul::push_new(&mut graph, a, b2, &mut rng);
        let ext_out = GlobalId::new(&mut rng);
        graph.set_output_map([(c2, ext_out)]);

        let mut shapes = HashMap::new();
        shapes.insert(a, vec![m, k]);
        shapes.insert(b1, vec![k, n1]);
        shapes.insert(b2, vec![k, n2]);
        shapes.insert(c1, vec![m, n1]);
        shapes.insert(c2, vec![m, n2]);

        let (artifacts, plan) = plan_from_graph(
            &graph,
            &shapes,
            TaskPlannerConfig {
                min_tile_elements: 256,
                max_tile_elements: 4096,
            },
        )?;
        assert!(plan.stats.planned_loops >= 2);

        let coverage = analyze_coverage(&artifacts, &plan);
        assert_eq!(coverage.pointwise_loops, 0);
        assert_eq!(coverage.unknown_loops, 0);
        assert!(coverage.planned_reduction_loops >= 2);

        let compiled = compile_plan(&artifacts, &plan, &shapes)?;
        let layout = &compiled.layout;
        let mut bufs: Vec<Vec<f32>> = (0..layout.num_buffers).map(|_| Vec::new()).collect();
        let mut rng_a = wyrand::WyRand::new(9811 + m as u64 + k as u64);
        let mut rng_b1 = wyrand::WyRand::new(9821 + k as u64 + n1 as u64);
        let mut rng_b2 = wyrand::WyRand::new(9831 + k as u64 + n2 as u64);

        for (id, &size) in &layout.tensor_sizes {
            let idx = layout.tensor_index[id];
            let data = if *id == a {
                (0..size)
                    .map(|_| (rng_a.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0)
                    .collect::<Vec<_>>()
            } else if *id == b1 {
                (0..size)
                    .map(|_| (rng_b1.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0)
                    .collect::<Vec<_>>()
            } else if *id == b2 {
                (0..size)
                    .map(|_| (rng_b2.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0)
                    .collect::<Vec<_>>()
            } else {
                vec![0.0f32; size]
            };
            bufs[idx] = data;
        }

        let a_data = bufs[layout.tensor_index[&a]].clone();
        let b2_data = bufs[layout.tensor_index[&b2]].clone();
        let mut ptrs: Vec<*mut f32> = bufs.iter_mut().map(|v| v.as_mut_ptr()).collect();
        unsafe { compiled.execute(&mut ptrs) };
        let got = bufs[layout.tensor_index[&c2]].clone();

        let mut expected = vec![0.0f32; m * n2];
        for i in 0..m {
            for j in 0..n2 {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += a_data[i * k + kk] * b2_data[kk * n2 + j];
                }
                expected[i * n2 + j] = acc;
            }
        }

        Ok((got, expected))
    }

    fn run_compiled_rank2_matmul_sigmoid_bias(
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), V7CodegenError> {
        let mut rng = wyrand::WyRand::new(9900 + m as u64 + n as u64);
        let ext_a = GlobalId::new(&mut rng);
        let ext_b = GlobalId::new(&mut rng);
        let ext_bias = GlobalId::new(&mut rng);
        let ext_ones = GlobalId::new(&mut rng);
        let (mut graph, input_map) =
            MilliOpGraph::new([ext_a, ext_b, ext_bias, ext_ones], &mut rng);
        let a = input_map[&ext_a];
        let b = input_map[&ext_b];
        let bias = input_map[&ext_bias];
        let ones = input_map[&ext_ones];
        let mm = MatMul::push_new(&mut graph, a, b, &mut rng);
        let bias_add = SimpleBinary::add(&mut graph, mm, bias, &mut rng);
        let neg = SimpleUnaryOp::neg(&mut graph, bias_add, &mut rng);
        let exp = SimpleUnaryOp::exp(&mut graph, neg, &mut rng);
        let denom = SimpleBinary::add(&mut graph, exp, ones, &mut rng);
        let out = SimpleUnaryOp::reciprocal(&mut graph, denom, &mut rng);
        let ext_out = GlobalId::new(&mut rng);
        graph.set_output_map([(out, ext_out)]);

        let mut shapes = HashMap::new();
        shapes.insert(a, vec![m, k]);
        shapes.insert(b, vec![k, n]);
        shapes.insert(bias, vec![1, n]);
        shapes.insert(ones, vec![1, n]);
        shapes.insert(mm, vec![m, n]);
        shapes.insert(bias_add, vec![m, n]);
        shapes.insert(neg, vec![m, n]);
        shapes.insert(exp, vec![m, n]);
        shapes.insert(denom, vec![m, n]);
        shapes.insert(out, vec![m, n]);

        let (artifacts, plan) = plan_from_graph(
            &graph,
            &shapes,
            TaskPlannerConfig {
                min_tile_elements: 256,
                max_tile_elements: 4096,
            },
        )?;
        let coverage = analyze_coverage(&artifacts, &plan);
        assert!(coverage.planned_reduction_loops >= 1);
        assert!(coverage.pointwise_loops >= 1);
        assert_eq!(coverage.unknown_loops, 0);

        let compiled = compile_plan(&artifacts, &plan, &shapes)?;
        let layout = &compiled.layout;
        let mut bufs: Vec<Vec<f32>> = (0..layout.num_buffers).map(|_| Vec::new()).collect();
        let mut rng_a = wyrand::WyRand::new(9911 + m as u64 + k as u64);
        let mut rng_b = wyrand::WyRand::new(9921 + k as u64 + n as u64);
        let mut rng_bias = wyrand::WyRand::new(9931 + n as u64);

        for (id, &size) in &layout.tensor_sizes {
            let idx = layout.tensor_index[id];
            let data = if *id == a {
                (0..size)
                    .map(|_| (rng_a.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0)
                    .collect::<Vec<_>>()
            } else if *id == b {
                (0..size)
                    .map(|_| (rng_b.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0)
                    .collect::<Vec<_>>()
            } else if *id == bias {
                (0..size)
                    .map(|_| (rng_bias.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0)
                    .collect::<Vec<_>>()
            } else if *id == ones {
                vec![1.0f32; size]
            } else {
                vec![0.0f32; size]
            };
            bufs[idx] = data;
        }

        let a_data = bufs[layout.tensor_index[&a]].clone();
        let b_data = bufs[layout.tensor_index[&b]].clone();
        let bias_data = bufs[layout.tensor_index[&bias]].clone();
        let mut ptrs: Vec<*mut f32> = bufs.iter_mut().map(|v| v.as_mut_ptr()).collect();
        unsafe { compiled.execute(&mut ptrs) };
        let got = bufs[layout.tensor_index[&out]].clone();

        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += a_data[i * k + kk] * b_data[kk * n + j];
                }
                let z = acc + bias_data[j];
                expected[i * n + j] = 1.0 / (1.0 + (-z).exp());
            }
        }

        Ok((got, expected))
    }

    #[test]
    fn test_compile_rank2_matmul_small() {
        let (got, expected) = run_compiled_rank2_matmul(8, 12, 10).expect("v7 compile");
        for i in 0..got.len() {
            assert!((got[i] - expected[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn test_compile_rank2_matmul_medium() {
        let (got, expected) = run_compiled_rank2_matmul(64, 96, 80).expect("v7 compile");
        for i in 0..got.len() {
            assert!((got[i] - expected[i]).abs() < 2e-3);
        }
    }

    #[test]
    fn test_compile_two_rank2_matmuls() {
        let (got, expected) = run_compiled_two_rank2_matmuls(32, 48, 24, 20).expect("v7 compile");
        for i in 0..got.len() {
            assert!((got[i] - expected[i]).abs() < 2e-3);
        }
    }

    #[test]
    fn test_compile_reduction_plus_pointwise_math_chain() {
        let (got, expected) =
            run_compiled_rank2_matmul_sigmoid_bias(16, 24, 20).expect("v7 compile");
        for i in 0..got.len() {
            assert!((got[i] - expected[i]).abs() < 2e-3);
        }
    }
}
