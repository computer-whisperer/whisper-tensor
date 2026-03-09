//! Task-plan execution for v7 parallel crystal scheduling.
//!
//! Current scope:
//! - execute additive-reduction `TaskPlan` tasks on CPU
//! - evaluate recovered reduction terms directly from v6 synthesis metadata
//! - provide serial + loop-parallel execution paths with per-loop barriers

use super::planner::{TaskPlan, TaskPlannerConfig, plan_from_graph};
use crate::compiler::attempts::v6_schedule_synthesis::synthesis::{
    AccessDimRole, LoopIntent, PipelineArtifacts, ReductionBinOp, ReductionTermPattern,
    ReductionUnaryOp,
};
use crate::graph::GlobalId;
use crate::milli_graph::MilliOpGraph;
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Clone)]
pub struct ExecuteConfig {
    pub num_threads: usize,
    pub min_parallel_tasks: usize,
    pub min_parallel_fmas: usize,
}

impl ExecuteConfig {
    pub fn single_threaded() -> Self {
        Self {
            num_threads: 1,
            min_parallel_tasks: usize::MAX,
            min_parallel_fmas: usize::MAX,
        }
    }

    pub fn auto() -> Self {
        Self {
            num_threads: std::thread::available_parallelism()
                .map(|x| x.get())
                .unwrap_or(1),
            min_parallel_tasks: 8,
            min_parallel_fmas: 64_000,
        }
    }
}

impl Default for ExecuteConfig {
    fn default() -> Self {
        Self::single_threaded()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum V7ExecuteError {
    #[error(transparent)]
    Plan(#[from] super::planner::V7PlanError),
    #[error("v7 execute: loop index {0} missing from schedule")]
    MissingLoop(usize),
    #[error("v7 execute: loop {0} is not additive-reduction")]
    UnsupportedLoopIntent(usize),
    #[error("v7 execute: missing buffer for tensor {0}")]
    MissingTensorBuffer(GlobalId),
    #[error("v7 execute: buffer for tensor {tensor} too small (have {have}, need {need})")]
    BufferTooSmall {
        tensor: GlobalId,
        have: usize,
        need: usize,
    },
    #[error("v7 execute: task output tensor mismatch for loop {loop_index}")]
    TaskOutputMismatch { loop_index: usize },
    #[error("v7 execute: task rank mismatch for loop {loop_index}")]
    TaskRankMismatch { loop_index: usize },
    #[error("v7 execute: invalid task output range on axis {axis} for loop {loop_index}")]
    InvalidTaskRange { loop_index: usize, axis: usize },
    #[error("v7 execute: output tensor used as reduction input in loop {0} (aliasing unsupported)")]
    OutputAliasingUnsupported(usize),
    #[error(
        "v7 execute: access out of bounds for tensor {tensor}, axis {axis}, coord {coord}, dim {dim}"
    )]
    AccessOutOfBounds {
        tensor: GlobalId,
        axis: usize,
        coord: i64,
        dim: usize,
    },
    #[error("v7 execute: reduction slot {slot} out of range (have {len})")]
    SlotOutOfRange { slot: usize, len: usize },
    #[error("v7 execute: unknown access role in loop {loop_index}")]
    UnknownAccessRole { loop_index: usize },
    #[error("v7 execute: worker thread panicked in loop {loop_index}")]
    WorkerPanic { loop_index: usize },
}

#[derive(Debug, Clone)]
struct CompiledAccess {
    tensor: GlobalId,
    output_coeffs: Vec<i64>,
    reduction_coeff: i64,
    base: i64,
    len: usize,
}

#[derive(Debug, Clone, Copy)]
enum FastTermPattern {
    Mul2 { a_slot: usize, b_slot: usize },
}

pub fn execute_graph_f32(
    graph: &MilliOpGraph,
    shapes: &HashMap<GlobalId, Vec<usize>>,
    buffers: &mut HashMap<GlobalId, Vec<f32>>,
    config: TaskPlannerConfig,
) -> Result<(PipelineArtifacts, TaskPlan), V7ExecuteError> {
    execute_graph_f32_with_config(graph, shapes, buffers, config, ExecuteConfig::default())
}

pub fn execute_graph_f32_with_config(
    graph: &MilliOpGraph,
    shapes: &HashMap<GlobalId, Vec<usize>>,
    buffers: &mut HashMap<GlobalId, Vec<f32>>,
    config: TaskPlannerConfig,
    exec: ExecuteConfig,
) -> Result<(PipelineArtifacts, TaskPlan), V7ExecuteError> {
    let (artifacts, plan) = plan_from_graph(graph, shapes, config)?;
    execute_plan_f32_with_config(&artifacts, &plan, buffers, exec)?;
    Ok((artifacts, plan))
}

pub fn execute_plan_f32(
    artifacts: &PipelineArtifacts,
    plan: &TaskPlan,
    buffers: &mut HashMap<GlobalId, Vec<f32>>,
) -> Result<(), V7ExecuteError> {
    execute_plan_f32_with_config(artifacts, plan, buffers, ExecuteConfig::default())
}

pub fn execute_plan_f32_with_config(
    artifacts: &PipelineArtifacts,
    plan: &TaskPlan,
    buffers: &mut HashMap<GlobalId, Vec<f32>>,
    exec: ExecuteConfig,
) -> Result<(), V7ExecuteError> {
    let mut task_idx = 0usize;
    while task_idx < plan.tasks.len() {
        let loop_index = plan.tasks[task_idx].loop_index;
        let recovered = artifacts
            .schedule
            .loops
            .get(loop_index)
            .ok_or(V7ExecuteError::MissingLoop(loop_index))?;
        let LoopIntent::AdditiveReduction(reduction) = &recovered.intent else {
            return Err(V7ExecuteError::UnsupportedLoopIntent(loop_index));
        };
        let output_tensor = recovered.output_tensor;
        let compiled_accesses = reduction
            .accesses
            .iter()
            .map(|access| compile_access(access, recovered.output_shape.len(), loop_index))
            .collect::<Result<Vec<_>, _>>()?;
        let fast_term = try_compile_fast_term(&reduction.canonical_term);

        if reduction.accesses.iter().any(|a| a.tensor == output_tensor) {
            return Err(V7ExecuteError::OutputAliasingUnsupported(loop_index));
        }

        let output_elems = shape_elements(&recovered.output_shape);
        let mut output_buffer = buffers
            .remove(&output_tensor)
            .ok_or(V7ExecuteError::MissingTensorBuffer(output_tensor))?;
        if output_buffer.len() < output_elems {
            return Err(V7ExecuteError::BufferTooSmall {
                tensor: output_tensor,
                have: output_buffer.len(),
                need: output_elems,
            });
        }

        let mut access_slices = Vec::with_capacity(reduction.accesses.len());
        for access in &reduction.accesses {
            let required = shape_elements(&access.tensor_shape);
            let buf = buffers
                .get(&access.tensor)
                .ok_or(V7ExecuteError::MissingTensorBuffer(access.tensor))?;
            if buf.len() < required {
                return Err(V7ExecuteError::BufferTooSmall {
                    tensor: access.tensor,
                    have: buf.len(),
                    need: required,
                });
            }
            access_slices.push(buf.as_slice());
        }

        let task_start = task_idx;
        while task_idx < plan.tasks.len() && plan.tasks[task_idx].loop_index == loop_index {
            task_idx += 1;
        }
        let loop_tasks = &plan.tasks[task_start..task_idx];

        if !try_execute_loop_parallel_rank2(
            loop_index,
            loop_tasks,
            &recovered.output_shape,
            recovered.output_tensor,
            &mut output_buffer,
            reduction,
            &access_slices,
            &compiled_accesses,
            fast_term,
            &exec,
        )? {
            execute_loop_serial(
                loop_tasks,
                loop_index,
                recovered.output_tensor,
                &recovered.output_shape,
                &mut output_buffer,
                reduction,
                &access_slices,
                &compiled_accesses,
                fast_term,
            )?;
        }

        buffers.insert(output_tensor, output_buffer);
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_loop_serial(
    loop_tasks: &[super::planner::ReductionTileTask],
    loop_index: usize,
    output_tensor: GlobalId,
    output_shape: &[usize],
    output_buffer: &mut [f32],
    reduction: &crate::compiler::attempts::v6_schedule_synthesis::synthesis::ReductionIntent,
    access_slices: &[&[f32]],
    compiled_accesses: &[CompiledAccess],
    fast_term: Option<FastTermPattern>,
) -> Result<(), V7ExecuteError> {
    for task in loop_tasks {
        execute_task(
            task,
            loop_index,
            output_tensor,
            output_shape,
            output_buffer,
            0,
            reduction,
            access_slices,
            compiled_accesses,
            fast_term,
        )?;
    }
    Ok(())
}

#[derive(Debug)]
struct RowTaskBlock<'a> {
    row_start: usize,
    row_end: usize,
    estimated_fmas: usize,
    tasks: Vec<&'a super::planner::ReductionTileTask>,
}

#[allow(clippy::too_many_arguments)]
fn try_execute_loop_parallel_rank2(
    loop_index: usize,
    loop_tasks: &[super::planner::ReductionTileTask],
    output_shape: &[usize],
    output_tensor: GlobalId,
    output_buffer: &mut [f32],
    reduction: &crate::compiler::attempts::v6_schedule_synthesis::synthesis::ReductionIntent,
    access_slices: &[&[f32]],
    compiled_accesses: &[CompiledAccess],
    fast_term: Option<FastTermPattern>,
    exec: &ExecuteConfig,
) -> Result<bool, V7ExecuteError> {
    if exec.num_threads <= 1 || loop_tasks.len() < exec.min_parallel_tasks {
        return Ok(false);
    }
    let total_fmas: usize = loop_tasks.iter().map(|t| t.estimated_fmas).sum();
    if total_fmas < exec.min_parallel_fmas {
        return Ok(false);
    }
    if output_shape.len() != 2 {
        return Ok(false);
    }
    let n = output_shape[1];
    if n == 0 {
        return Ok(false);
    }

    let mut grouped = BTreeMap::<(usize, usize), RowTaskBlock<'_>>::new();
    for task in loop_tasks {
        if task.output_shape != output_shape || task.output_start.len() != 2 || task.output_end.len() != 2 {
            return Ok(false);
        }
        let row_start = task.output_start[0];
        let row_end = task.output_end[0];
        if row_start >= row_end || row_end > output_shape[0] {
            return Ok(false);
        }
        let key = (row_start, row_end);
        grouped
            .entry(key)
            .or_insert_with(|| RowTaskBlock {
                row_start,
                row_end,
                estimated_fmas: 0,
                tasks: Vec::new(),
            })
            .tasks
            .push(task);
        if let Some(block) = grouped.get_mut(&key) {
            block.estimated_fmas = block.estimated_fmas.saturating_add(task.estimated_fmas);
        }
    }

    let mut row_blocks = grouped.into_values().collect::<Vec<_>>();
    if row_blocks.len() < 2 {
        return Ok(false);
    }
    row_blocks.sort_by_key(|b| (b.row_start, b.row_end));

    let mut prev_end = 0usize;
    for block in &row_blocks {
        if block.row_start < prev_end {
            return Ok(false);
        }
        prev_end = block.row_end;
    }

    let thread_count = exec.num_threads.min(row_blocks.len());
    if thread_count <= 1 {
        return Ok(false);
    }

    let total_block_work: usize = row_blocks.iter().map(|b| b.estimated_fmas).sum();
    let target_work = total_block_work.div_ceil(thread_count).max(1);

    let mut thread_blocks = Vec::<Vec<RowTaskBlock<'_>>>::new();
    let mut current = Vec::<RowTaskBlock<'_>>::new();
    let mut current_work = 0usize;
    for block in row_blocks {
        let block_work = block.estimated_fmas.max(1);
        if !current.is_empty()
            && current_work >= target_work
            && thread_blocks.len() + 1 < thread_count
        {
            thread_blocks.push(current);
            current = Vec::new();
            current_work = 0;
        }
        current_work = current_work.saturating_add(block_work);
        current.push(block);
    }
    if !current.is_empty() {
        thread_blocks.push(current);
    }
    if thread_blocks.len() <= 1 {
        return Ok(false);
    }

    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(thread_blocks.len());
        let mut remaining = output_buffer;
        let mut consumed_rows = 0usize;

        for blocks in thread_blocks {
            let row_start = blocks.first().map(|b| b.row_start).unwrap_or(0);
            let row_end = blocks.last().map(|b| b.row_end).unwrap_or(row_start);
            if row_start < consumed_rows || row_end < row_start {
                return Err(V7ExecuteError::TaskRankMismatch { loop_index });
            }

            let skip_rows = row_start.saturating_sub(consumed_rows);
            let skip_elems = skip_rows.saturating_mul(n);
            if skip_elems > remaining.len() {
                return Err(V7ExecuteError::TaskRankMismatch { loop_index });
            }
            let (_skip, tail) = remaining.split_at_mut(skip_elems);

            let block_rows = row_end.saturating_sub(row_start);
            let block_elems = block_rows.saturating_mul(n);
            if block_elems > tail.len() {
                return Err(V7ExecuteError::TaskRankMismatch { loop_index });
            }
            let (thread_output, tail_after) = tail.split_at_mut(block_elems);
            remaining = tail_after;
            consumed_rows = row_end;

            handles.push(scope.spawn(move || -> Result<(), V7ExecuteError> {
                for block in blocks {
                    for task in block.tasks {
                        execute_task(
                            task,
                            loop_index,
                            output_tensor,
                            output_shape,
                            thread_output,
                            row_start,
                            reduction,
                            access_slices,
                            compiled_accesses,
                            fast_term,
                        )?;
                    }
                }
                Ok(())
            }));
        }

        for handle in handles {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(err)) => return Err(err),
                Err(_) => return Err(V7ExecuteError::WorkerPanic { loop_index }),
            }
        }
        Ok(true)
    })
}

#[allow(clippy::too_many_arguments)]
fn execute_task(
    task: &super::planner::ReductionTileTask,
    loop_index: usize,
    output_tensor: GlobalId,
    output_shape: &[usize],
    output_buffer: &mut [f32],
    output_row_offset: usize,
    reduction: &crate::compiler::attempts::v6_schedule_synthesis::synthesis::ReductionIntent,
    access_slices: &[&[f32]],
    compiled_accesses: &[CompiledAccess],
    fast_term: Option<FastTermPattern>,
) -> Result<(), V7ExecuteError> {
    if task.output_tensor != output_tensor {
        return Err(V7ExecuteError::TaskOutputMismatch { loop_index });
    }
    if task.output_shape != output_shape {
        return Err(V7ExecuteError::TaskRankMismatch { loop_index });
    }
    if task.output_start.len() != output_shape.len() || task.output_end.len() != output_shape.len()
    {
        return Err(V7ExecuteError::TaskRankMismatch { loop_index });
    }
    for axis in 0..output_shape.len() {
        let start = task.output_start[axis];
        let end = task.output_end[axis];
        if start >= end || end > output_shape[axis] {
            return Err(V7ExecuteError::InvalidTaskRange { loop_index, axis });
        }
    }

    validate_task_affine_ranges(task, compiled_accesses, loop_index)?;
    let output_elem_offset = output_row_offset.saturating_mul(shape_elements(&output_shape[1..]));

    if let Some(FastTermPattern::Mul2 { a_slot, b_slot }) = fast_term {
        if output_shape.len() == 2
            && a_slot < compiled_accesses.len()
            && b_slot < compiled_accesses.len()
        {
            return execute_task_fast_mul2_rank2(
                task,
                output_shape,
                output_buffer,
                output_row_offset,
                &compiled_accesses[a_slot],
                &compiled_accesses[b_slot],
                access_slices[a_slot],
                access_slices[b_slot],
            );
        }
    }

    let mut coords = task.output_start.clone();
    let mut load_vals = vec![0.0f32; reduction.accesses.len()];
    for_each_coord_in_range(
        &task.output_start,
        &task.output_end,
        0,
        &mut coords,
        &mut |coord| {
            let mut acc = 0.0f32;
            for kk in task.reduction_start..task.reduction_end {
                for slot in 0..reduction.accesses.len() {
                    let flat = affine_index(&compiled_accesses[slot], coord, kk as i64);
                    debug_assert!(flat < compiled_accesses[slot].len);
                    load_vals[slot] = access_slices[slot][flat];
                }
                acc += eval_reduction_term_f32(&reduction.canonical_term, &load_vals)?;
            }
            let out_flat = flatten_index(output_shape, coord);
            if out_flat < output_elem_offset || out_flat >= output_elem_offset + output_buffer.len() {
                return Err(V7ExecuteError::TaskRankMismatch { loop_index });
            }
            output_buffer[out_flat - output_elem_offset] = acc;
            Ok(())
        },
    )
}

fn shape_elements(shape: &[usize]) -> usize {
    shape.iter().copied().fold(1usize, usize::saturating_mul)
}

fn flatten_index(shape: &[usize], coord: &[usize]) -> usize {
    let mut flat = 0usize;
    for (d, c) in shape.iter().zip(coord.iter()) {
        flat = flat * *d + *c;
    }
    flat
}

fn for_each_coord_in_range(
    start: &[usize],
    end: &[usize],
    axis: usize,
    coord: &mut [usize],
    f: &mut impl FnMut(&[usize]) -> Result<(), V7ExecuteError>,
) -> Result<(), V7ExecuteError> {
    if axis == start.len() {
        return f(coord);
    }
    for v in start[axis]..end[axis] {
        coord[axis] = v;
        for_each_coord_in_range(start, end, axis + 1, coord, f)?;
    }
    Ok(())
}

fn compile_access(
    access: &crate::compiler::attempts::v6_schedule_synthesis::synthesis::RecoveredTensorAccess,
    output_rank: usize,
    loop_index: usize,
) -> Result<CompiledAccess, V7ExecuteError> {
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
            AccessDimRole::Unknown => return Err(V7ExecuteError::UnknownAccessRole { loop_index }),
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

fn try_compile_fast_term(term: &ReductionTermPattern) -> Option<FastTermPattern> {
    let ReductionTermPattern::Bin { op, a, b } = term else {
        return None;
    };
    if *op != ReductionBinOp::Mul {
        return None;
    }
    match (&**a, &**b) {
        (ReductionTermPattern::Load(a_slot), ReductionTermPattern::Load(b_slot)) => {
            Some(FastTermPattern::Mul2 {
                a_slot: *a_slot,
                b_slot: *b_slot,
            })
        }
        _ => None,
    }
}

fn affine_index(access: &CompiledAccess, output_coord: &[usize], reduction_k: i64) -> usize {
    let mut flat = access
        .base
        .saturating_add(access.reduction_coeff.saturating_mul(reduction_k));
    for (axis, coeff) in access.output_coeffs.iter().enumerate() {
        let coord = output_coord.get(axis).copied().unwrap_or(0) as i64;
        flat = flat.saturating_add(coeff.saturating_mul(coord));
    }
    debug_assert!(flat >= 0);
    flat as usize
}

fn validate_task_affine_ranges(
    task: &super::planner::ReductionTileTask,
    accesses: &[CompiledAccess],
    loop_index: usize,
) -> Result<(), V7ExecuteError> {
    for (slot, access) in accesses.iter().enumerate() {
        let (min_flat, max_flat) = affine_range_for_task(task, access);
        if min_flat < 0 || max_flat >= access.len as i64 {
            return Err(V7ExecuteError::AccessOutOfBounds {
                tensor: access.tensor,
                axis: slot,
                coord: if min_flat < 0 { min_flat } else { max_flat },
                dim: access.len,
            });
        }
        if access.output_coeffs.len() < task.output_start.len() {
            return Err(V7ExecuteError::TaskRankMismatch { loop_index });
        }
    }
    Ok(())
}

fn affine_range_for_task(
    task: &super::planner::ReductionTileTask,
    access: &CompiledAccess,
) -> (i64, i64) {
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
    if task.reduction_end > task.reduction_start {
        let k_lo = task.reduction_start as i64;
        let k_hi = task.reduction_end.saturating_sub(1) as i64;
        if access.reduction_coeff >= 0 {
            min = min.saturating_add(access.reduction_coeff.saturating_mul(k_lo));
            max = max.saturating_add(access.reduction_coeff.saturating_mul(k_hi));
        } else {
            min = min.saturating_add(access.reduction_coeff.saturating_mul(k_hi));
            max = max.saturating_add(access.reduction_coeff.saturating_mul(k_lo));
        }
    }
    (min, max)
}

#[allow(clippy::too_many_arguments)]
fn execute_task_fast_mul2_rank2(
    task: &super::planner::ReductionTileTask,
    output_shape: &[usize],
    output_buffer: &mut [f32],
    output_row_offset: usize,
    access_a: &CompiledAccess,
    access_b: &CompiledAccess,
    buf_a: &[f32],
    buf_b: &[f32],
) -> Result<(), V7ExecuteError> {
    let n = output_shape[1];
    let a_i = access_a.output_coeffs.first().copied().unwrap_or(0);
    let a_j = access_a.output_coeffs.get(1).copied().unwrap_or(0);
    let a_k = access_a.reduction_coeff;
    let b_i = access_b.output_coeffs.first().copied().unwrap_or(0);
    let b_j = access_b.output_coeffs.get(1).copied().unwrap_or(0);
    let b_k = access_b.reduction_coeff;
    let red_start = task.reduction_start as i64;

    if a_j == 0 && b_j == 1 {
        return execute_task_fast_mul2_rank2_row_accum(
            task,
            output_shape,
            output_buffer,
            output_row_offset,
            access_a,
            access_b,
            buf_a,
            buf_b,
            a_i,
            a_k,
            b_i,
            b_k,
            red_start,
        );
    }

    for i in task.output_start[0]..task.output_end[0] {
        let i64_i = i as i64;
        let base_ai = access_a.base + a_i * i64_i;
        let base_bi = access_b.base + b_i * i64_i;
        for j in task.output_start[1]..task.output_end[1] {
            let i64_j = j as i64;
            let mut idx_a = base_ai + a_j * i64_j + a_k * red_start;
            let mut idx_b = base_bi + b_j * i64_j + b_k * red_start;
            let mut acc = 0.0f32;
            for _ in task.reduction_start..task.reduction_end {
                debug_assert!(idx_a >= 0 && (idx_a as usize) < access_a.len);
                debug_assert!(idx_b >= 0 && (idx_b as usize) < access_b.len);
                acc += buf_a[idx_a as usize] * buf_b[idx_b as usize];
                idx_a += a_k;
                idx_b += b_k;
            }
            if i < output_row_offset {
                return Err(V7ExecuteError::TaskRankMismatch { loop_index: task.loop_index });
            }
            output_buffer[(i - output_row_offset) * n + j] = acc;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_task_fast_mul2_rank2_row_accum(
    task: &super::planner::ReductionTileTask,
    output_shape: &[usize],
    output_buffer: &mut [f32],
    output_row_offset: usize,
    access_a: &CompiledAccess,
    access_b: &CompiledAccess,
    buf_a: &[f32],
    buf_b: &[f32],
    a_i: i64,
    a_k: i64,
    b_i: i64,
    b_k: i64,
    red_start: i64,
) -> Result<(), V7ExecuteError> {
    let n = output_shape[1];
    let j_start = task.output_start[1];
    let j_end = task.output_end[1];
    let row_len = j_end.saturating_sub(j_start);

    for i in task.output_start[0]..task.output_end[0] {
        if i < output_row_offset {
            return Err(V7ExecuteError::TaskRankMismatch { loop_index: task.loop_index });
        }
        let i64_i = i as i64;
        let out_start = (i - output_row_offset) * n + j_start;
        let out_end = out_start + row_len;
        let out_row = &mut output_buffer[out_start..out_end];
        out_row.fill(0.0);

        let mut idx_a = access_a.base + a_i * i64_i + a_k * red_start;
        let mut base_b = access_b.base + b_i * i64_i + (j_start as i64) + b_k * red_start;
        for _ in task.reduction_start..task.reduction_end {
            debug_assert!(idx_a >= 0 && (idx_a as usize) < access_a.len);
            let av = buf_a[idx_a as usize];
            let b_start = base_b as usize;
            let b_end = b_start + row_len;
            debug_assert!(b_end <= access_b.len);
            let b_row = &buf_b[b_start..b_end];
            for x in 0..row_len {
                out_row[x] += av * b_row[x];
            }
            idx_a += a_k;
            base_b += b_k;
        }
    }
    Ok(())
}

fn eval_reduction_term_f32(
    term: &ReductionTermPattern,
    loads: &[f32],
) -> Result<f32, V7ExecuteError> {
    match term {
        ReductionTermPattern::Load(slot) => {
            loads
                .get(*slot)
                .copied()
                .ok_or(V7ExecuteError::SlotOutOfRange {
                    slot: *slot,
                    len: loads.len(),
                })
        }
        ReductionTermPattern::Literal(bits) => Ok(f64::from_bits(*bits) as f32),
        ReductionTermPattern::Bin { op, a, b } => {
            let av = eval_reduction_term_f32(a, loads)?;
            let bv = eval_reduction_term_f32(b, loads)?;
            let out = match op {
                ReductionBinOp::Add => av + bv,
                ReductionBinOp::Sub => av - bv,
                ReductionBinOp::Mul => av * bv,
                ReductionBinOp::Div => av / bv,
                ReductionBinOp::Max => av.max(bv),
                ReductionBinOp::Min => av.min(bv),
            };
            Ok(out)
        }
        ReductionTermPattern::Unary { op, input } => {
            let v = eval_reduction_term_f32(input, loads)?;
            let out = match op {
                ReductionUnaryOp::Neg => -v,
                ReductionUnaryOp::Abs => v.abs(),
                ReductionUnaryOp::Exp => v.exp(),
                ReductionUnaryOp::Ln => v.ln(),
                ReductionUnaryOp::Sqrt => v.sqrt(),
                ReductionUnaryOp::Reciprocal => 1.0 / v,
                ReductionUnaryOp::Tanh => v.tanh(),
                ReductionUnaryOp::Floor => v.floor(),
                ReductionUnaryOp::Ceil => v.ceil(),
            };
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::milli_graph::ops::MatMul;
    use rand::RngCore;

    fn build_matmul(
        m: usize,
        k: usize,
        n: usize,
        seed: u64,
    ) -> (
        MilliOpGraph,
        HashMap<GlobalId, Vec<usize>>,
        GlobalId,
        GlobalId,
        GlobalId,
    ) {
        let mut rng = wyrand::WyRand::new(seed);
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
        (graph, shapes, a, b, c)
    }

    fn random_vec(len: usize, seed: u64) -> Vec<f32> {
        let mut rng = wyrand::WyRand::new(seed);
        (0..len)
            .map(|_| (rng.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0)
            .collect()
    }

    fn reference_matmul(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for kk in 0..k {
                    acc += a[i * k + kk] * b[kk * n + j];
                }
                out[i * n + j] = acc;
            }
        }
        out
    }

    #[test]
    fn test_execute_plan_small_matmul() {
        let (graph, shapes, a, b, c) = build_matmul(8, 12, 10, 4101);
        let (artifacts, plan) =
            plan_from_graph(&graph, &shapes, TaskPlannerConfig::default()).expect("v7 plan");
        let a_data = random_vec(8 * 12, 7001);
        let b_data = random_vec(12 * 10, 7002);
        let mut bufs = HashMap::new();
        bufs.insert(a, a_data.clone());
        bufs.insert(b, b_data.clone());
        bufs.insert(c, vec![0.0; 8 * 10]);

        execute_plan_f32(&artifacts, &plan, &mut bufs).expect("v7 execute");
        let got = bufs.get(&c).unwrap();
        let exp = reference_matmul(8, 12, 10, &a_data, &b_data);
        for i in 0..got.len() {
            assert!((got[i] - exp[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn test_execute_plan_larger_matmul() {
        let (graph, shapes, a, b, c) = build_matmul(80, 96, 96, 4102);
        let (artifacts, plan) = plan_from_graph(
            &graph,
            &shapes,
            TaskPlannerConfig {
                min_tile_elements: 512,
                max_tile_elements: 4096,
            },
        )
        .expect("v7 plan");

        let a_data = random_vec(80 * 96, 7101);
        let b_data = random_vec(96 * 96, 7102);
        let mut bufs = HashMap::new();
        bufs.insert(a, a_data.clone());
        bufs.insert(b, b_data.clone());
        bufs.insert(c, vec![0.0; 80 * 96]);

        execute_plan_f32(&artifacts, &plan, &mut bufs).expect("v7 execute");
        let got = bufs.get(&c).unwrap();
        let exp = reference_matmul(80, 96, 96, &a_data, &b_data);
        for i in 0..got.len() {
            assert!((got[i] - exp[i]).abs() < 2e-3);
        }
    }

    #[test]
    fn test_execute_plan_parallel_matches_single_thread() {
        let (graph, shapes, a, b, c) = build_matmul(112, 128, 112, 4103);
        let (artifacts, plan) = plan_from_graph(
            &graph,
            &shapes,
            TaskPlannerConfig {
                min_tile_elements: 512,
                max_tile_elements: 4096,
            },
        )
        .expect("v7 plan");

        let a_data = random_vec(112 * 128, 7201);
        let b_data = random_vec(128 * 112, 7202);

        let mut serial = HashMap::new();
        serial.insert(a, a_data.clone());
        serial.insert(b, b_data.clone());
        serial.insert(c, vec![0.0; 112 * 112]);
        execute_plan_f32(&artifacts, &plan, &mut serial).expect("v7 serial execute");
        let serial_out = serial.get(&c).unwrap().clone();

        let mut parallel = HashMap::new();
        parallel.insert(a, a_data);
        parallel.insert(b, b_data);
        parallel.insert(c, vec![0.0; 112 * 112]);
        execute_plan_f32_with_config(
            &artifacts,
            &plan,
            &mut parallel,
            ExecuteConfig {
                num_threads: 4,
                min_parallel_tasks: 1,
                min_parallel_fmas: 1,
            },
        )
        .expect("v7 parallel execute");
        let parallel_out = parallel.get(&c).unwrap();

        assert_eq!(parallel_out.len(), serial_out.len());
        for i in 0..parallel_out.len() {
            assert!((parallel_out[i] - serial_out[i]).abs() < 2e-3);
        }
    }
}
