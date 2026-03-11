//! Parallel-ready task planner on top of v6 recovered schedules.
//!
//! Current scope:
//! - consume v6 recovered loop/schedule artifacts for smaller graphs
//! - bypass v6 synthesis with a direct large-graph fallback for rank-2 matmul
//! - produce non-overlapping output tile tasks for additive reductions
//! - keep reduction full-range per tile (no split-k yet)

use crate::compiler::attempts::v6_schedule_synthesis::synthesis::{
    AccessDimRole, LoopDim, LoopIntent, LoopScheduleCandidate, PipelineArtifacts, PipelineError,
    RecoveredLoop, RecoveredTensorAccess, ReductionBinOp, ReductionIntent, ReductionTermPattern,
    ScheduleIR, ScheduleStats, VectorizePlan, build_from_graph,
};
use crate::graph::{GlobalId, Graph, Node};
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::AnyMilliOp;
use std::collections::HashMap;

const DIRECT_FALLBACK_ESTIMATE_DEFAULT: usize = 64_000_000;
const DIRECT_FALLBACK_ESTIMATE_ENV: &str = "WT_V7_DIRECT_FALLBACK_ESTIMATE";

#[derive(Debug, thiserror::Error)]
pub enum V7PlanError {
    #[error(transparent)]
    Pipeline(#[from] PipelineError),
    #[error("v7 planner: {0}")]
    Planner(String),
}

#[derive(Debug, Clone)]
pub struct TaskPlannerConfig {
    pub min_tile_elements: usize,
    pub max_tile_elements: usize,
}

impl Default for TaskPlannerConfig {
    fn default() -> Self {
        Self {
            min_tile_elements: 128,
            max_tile_elements: 4096,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct TaskPlanStats {
    pub loop_count: usize,
    pub additive_reduction_loops: usize,
    pub planned_loops: usize,
    pub total_tasks: usize,
    pub total_estimated_fmas: usize,
}

#[derive(Debug, Clone)]
pub struct ReductionTileTask {
    pub loop_index: usize,
    pub output_tensor: GlobalId,
    pub output_shape: Vec<usize>,
    pub output_start: Vec<usize>,
    pub output_end: Vec<usize>,
    pub reduction_start: usize,
    pub reduction_end: usize,
    pub estimated_fmas: usize,
}

#[derive(Debug, Clone, Default)]
pub struct TaskPlan {
    pub tasks: Vec<ReductionTileTask>,
    pub stats: TaskPlanStats,
}

pub fn plan_from_graph(
    graph: &MilliOpGraph,
    shapes: &HashMap<GlobalId, Vec<usize>>,
    config: TaskPlannerConfig,
) -> Result<(PipelineArtifacts, TaskPlan), V7PlanError> {
    let fallback_estimate = read_direct_fallback_estimate_from_env()?;
    let artifacts = build_artifacts(graph, shapes, fallback_estimate)?;
    let plan = plan_from_artifacts(&artifacts, config)?;
    Ok((artifacts, plan))
}

fn build_artifacts(
    graph: &MilliOpGraph,
    shapes: &HashMap<GlobalId, Vec<usize>>,
    fallback_estimate: usize,
) -> Result<PipelineArtifacts, V7PlanError> {
    let estimated_nano_ops = estimate_graph_nano_ops(graph, shapes)?;
    if estimated_nano_ops > fallback_estimate {
        build_direct_artifacts_for_large_graph(graph, shapes, estimated_nano_ops)
    } else {
        Ok(build_from_graph(graph, shapes)?)
    }
}

fn read_direct_fallback_estimate_from_env() -> Result<usize, V7PlanError> {
    match std::env::var(DIRECT_FALLBACK_ESTIMATE_ENV) {
        Ok(raw) => {
            let parsed = raw.parse::<usize>().map_err(|_| {
                V7PlanError::Planner(format!(
                    "{DIRECT_FALLBACK_ESTIMATE_ENV} must be a positive integer, got '{raw}'"
                ))
            })?;
            if parsed == 0 {
                return Err(V7PlanError::Planner(format!(
                    "{DIRECT_FALLBACK_ESTIMATE_ENV} must be > 0"
                )));
            }
            Ok(parsed)
        }
        Err(std::env::VarError::NotPresent) => Ok(DIRECT_FALLBACK_ESTIMATE_DEFAULT),
        Err(std::env::VarError::NotUnicode(_)) => Err(V7PlanError::Planner(format!(
            "{DIRECT_FALLBACK_ESTIMATE_ENV} must be valid unicode"
        ))),
    }
}

fn build_direct_artifacts_for_large_graph(
    graph: &MilliOpGraph,
    shapes: &HashMap<GlobalId, Vec<usize>>,
    estimated_nano_ops: usize,
) -> Result<PipelineArtifacts, V7PlanError> {
    let mut loops = Vec::<RecoveredLoop>::new();
    let mut total_stores = 0usize;

    for op_id in graph.op_ordering() {
        let op = graph.get_node_by_id(op_id).unwrap();
        match op.op_kind().as_str() {
            "Constant" | "ConstantOfShape" => {}
            "MatMul" => {
                let recovered = build_rank2_matmul_loop(op, shapes)?;
                total_stores = checked_add_usize(
                    total_stores,
                    recovered.covered_outputs,
                    "direct fallback total stores",
                )?;
                loops.push(recovered);
            }
            kind => {
                return Err(V7PlanError::Planner(format!(
                    "direct fallback only supports MatMul when nano-op estimate is large; encountered unsupported op '{kind}'"
                )));
            }
        }
    }

    let stats = ScheduleStats {
        total_nano_ops: estimated_nano_ops,
        total_stores,
        grouped_families: loops.len(),
        pointwise_families: 0,
        additive_reduction_families: loops.len(),
        unknown_families: 0,
    };

    Ok(PipelineArtifacts {
        ordered_nano_ops: Vec::new(),
        whitewashed_nano_ops: Vec::new(),
        schedule: ScheduleIR { loops, stats },
    })
}

fn build_rank2_matmul_loop(
    op: &AnyMilliOp,
    shapes: &HashMap<GlobalId, Vec<usize>>,
) -> Result<RecoveredLoop, V7PlanError> {
    let inputs: Vec<GlobalId> = op.inputs().collect();
    let outputs: Vec<GlobalId> = op.outputs().collect();
    if inputs.len() < 2 || outputs.is_empty() {
        return Err(V7PlanError::Planner(
            "MatMul op is missing expected inputs/outputs".to_string(),
        ));
    }

    let a = inputs[0];
    let b = inputs[1];
    let c = outputs[0];

    let a_shape = shapes
        .get(&a)
        .ok_or(PipelineError::MissingShape(a))?
        .clone();
    let b_shape = shapes
        .get(&b)
        .ok_or(PipelineError::MissingShape(b))?
        .clone();
    let c_shape = shapes
        .get(&c)
        .ok_or(PipelineError::MissingShape(c))?
        .clone();

    if a_shape.len() != 2 || b_shape.len() != 2 || c_shape.len() != 2 {
        return Err(V7PlanError::Planner(format!(
            "direct fallback only supports rank-2 MatMul (got A{:?}, B{:?}, C{:?})",
            a_shape, b_shape, c_shape
        )));
    }

    let m = a_shape[0];
    let k = a_shape[1];
    let kb = b_shape[0];
    let n = b_shape[1];
    if kb != k || c_shape[0] != m || c_shape[1] != n {
        return Err(V7PlanError::Planner(format!(
            "incompatible MatMul shapes in direct fallback: A{:?}, B{:?}, C{:?}",
            a_shape, b_shape, c_shape
        )));
    }

    let covered_outputs = checked_mul_usize(m, n, "direct fallback covered outputs")?;
    let schedule = direct_rank2_matmul_schedule(m, n, k);
    let reduction = ReductionIntent {
        terms: k,
        canonical_term: ReductionTermPattern::Bin {
            op: ReductionBinOp::Mul,
            a: Box::new(ReductionTermPattern::Load(0)),
            b: Box::new(ReductionTermPattern::Load(1)),
        },
        term_load_tensors: vec![a, b],
        accesses: vec![
            RecoveredTensorAccess {
                tensor: a,
                tensor_shape: a_shape,
                dim_roles: vec![
                    AccessDimRole::OutputAxis {
                        axis: 0,
                        stride: 1,
                        offset: 0,
                    },
                    AccessDimRole::ReductionAxis {
                        stride: 1,
                        offset: 0,
                    },
                ],
            },
            RecoveredTensorAccess {
                tensor: b,
                tensor_shape: b_shape,
                dim_roles: vec![
                    AccessDimRole::ReductionAxis {
                        stride: 1,
                        offset: 0,
                    },
                    AccessDimRole::OutputAxis {
                        axis: 1,
                        stride: 1,
                        offset: 0,
                    },
                ],
            },
        ],
    };

    Ok(RecoveredLoop {
        output_tensor: c,
        output_shape: c_shape,
        covered_outputs,
        load_tensors: vec![a, b],
        intent: LoopIntent::AdditiveReduction(reduction),
        schedule_candidates: vec![schedule],
        selected_schedule: Some(0),
    })
}

fn direct_rank2_matmul_schedule(m: usize, n: usize, k: usize) -> LoopScheduleCandidate {
    let output_i_tile = if m >= 64 { 64 } else { m.max(1) };
    let output_j_tile = if n >= 64 { 64 } else { n.max(1) };
    let reduction_tile = if k >= 256 {
        128
    } else if k >= 64 {
        64
    } else {
        k.max(1)
    };
    let reduction_unroll = if k >= 16 {
        8
    } else if k >= 8 {
        4
    } else {
        1
    };
    let vector_width = if n.is_multiple_of(8) {
        8
    } else if n.is_multiple_of(4) {
        4
    } else {
        1
    };
    let vectorize = (vector_width > 1).then_some(VectorizePlan {
        axis: 1,
        width: vector_width,
    });

    LoopScheduleCandidate {
        loop_order: vec![
            LoopDim::OutputAxis(0),
            LoopDim::OutputAxis(1),
            LoopDim::ReductionAxis,
        ],
        output_tiles: vec![output_i_tile, output_j_tile],
        reduction_tile,
        reduction_unroll,
        vectorize,
        score: 100,
        rationale: "v7 direct fallback rank2 matmul schedule".to_string(),
    }
}

fn estimate_graph_nano_ops(
    graph: &MilliOpGraph,
    shapes: &HashMap<GlobalId, Vec<usize>>,
) -> Result<usize, V7PlanError> {
    let mut total = 0usize;
    for op_id in graph.op_ordering() {
        let op = graph.get_node_by_id(op_id).unwrap();
        let est = estimate_op_nano_count(op, shapes)?;
        total = checked_add_usize(total, est, "total nano-op estimate accumulation")?;
    }
    Ok(total)
}

fn estimate_op_nano_count(
    op: &AnyMilliOp,
    shapes: &HashMap<GlobalId, Vec<usize>>,
) -> Result<usize, V7PlanError> {
    let kind = op.op_kind();
    let outputs: Vec<GlobalId> = op.outputs().collect();
    let inputs: Vec<GlobalId> = op.inputs().collect();

    let scalar_elems = |tensor: GlobalId| -> Result<usize, V7PlanError> {
        let shape = shapes
            .get(&tensor)
            .ok_or(PipelineError::MissingShape(tensor))?;
        shape.iter().try_fold(1usize, |acc, &dim| {
            checked_mul_usize(
                acc,
                dim,
                &format!("shape product for tensor {tensor} in op {kind}"),
            )
        })
    };

    match kind.as_str() {
        "Constant" | "ConstantOfShape" => Ok(0),
        "Add" | "Sub" | "Mul" | "Div" | "Max" | "Min" => {
            let out = outputs
                .first()
                .copied()
                .ok_or_else(|| V7PlanError::Planner(format!("missing output for op {kind}")))?;
            let elems = scalar_elems(out)?;
            checked_mul_usize(elems, 4, &format!("binary op count for {kind}"))
        }
        "Neg" | "Abs" | "Exp" | "Ln" | "Sqrt" | "Reciprocal" | "Floor" | "Ceil" | "Tanh" => {
            let out = outputs
                .first()
                .copied()
                .ok_or_else(|| V7PlanError::Planner(format!("missing output for op {kind}")))?;
            let elems = scalar_elems(out)?;
            checked_mul_usize(elems, 3, &format!("unary op count for {kind}"))
        }
        "MatMul" => {
            if inputs.len() < 2 || outputs.is_empty() {
                return Err(V7PlanError::Planner(
                    "matmul inputs/outputs missing".to_string(),
                ));
            }
            let a_shape = shapes
                .get(&inputs[0])
                .ok_or(PipelineError::MissingShape(inputs[0]))?;
            let b_shape = shapes
                .get(&inputs[1])
                .ok_or(PipelineError::MissingShape(inputs[1]))?;
            let out_shape = shapes
                .get(&outputs[0])
                .ok_or(PipelineError::MissingShape(outputs[0]))?;
            if a_shape.len() != 2 || b_shape.len() != 2 || out_shape.len() != 2 {
                return Ok(0);
            }
            let m = out_shape[0];
            let n = out_shape[1];
            if m == 0 || n == 0 {
                return Ok(0);
            }
            let k = a_shape[1];
            let out_elems = checked_mul_usize(m, n, "matmul output elements")?;
            let per_out = checked_add_usize(
                checked_mul_usize(4, k, "matmul per-output k terms")?,
                2,
                "matmul per-output fixed terms",
            )?;
            checked_mul_usize(out_elems, per_out, "matmul nano-op estimate")
        }
        _ => Ok(0),
    }
}

fn checked_mul_usize(a: usize, b: usize, context: &str) -> Result<usize, V7PlanError> {
    a.checked_mul(b)
        .ok_or_else(|| V7PlanError::Planner(format!("overflow while computing {context}")))
}

fn checked_add_usize(a: usize, b: usize, context: &str) -> Result<usize, V7PlanError> {
    a.checked_add(b)
        .ok_or_else(|| V7PlanError::Planner(format!("overflow while computing {context}")))
}

pub fn plan_from_artifacts(
    artifacts: &PipelineArtifacts,
    config: TaskPlannerConfig,
) -> Result<TaskPlan, V7PlanError> {
    if config.min_tile_elements == 0 || config.max_tile_elements == 0 {
        return Err(V7PlanError::Planner(
            "tile element bounds must be non-zero".to_string(),
        ));
    }
    if config.min_tile_elements > config.max_tile_elements {
        return Err(V7PlanError::Planner(
            "min_tile_elements cannot exceed max_tile_elements".to_string(),
        ));
    }

    let mut out = TaskPlan::default();
    out.stats.loop_count = artifacts.schedule.loops.len();

    for (loop_index, recovered) in artifacts.schedule.loops.iter().enumerate() {
        let LoopIntent::AdditiveReduction(reduction) = &recovered.intent else {
            continue;
        };
        out.stats.additive_reduction_loops += 1;

        let selected = recovered
            .selected_schedule
            .and_then(|idx| recovered.schedule_candidates.get(idx))
            .or_else(|| recovered.schedule_candidates.first());
        let Some(schedule) = selected else {
            continue;
        };

        let tile_shape = choose_tile_shape(&recovered.output_shape, schedule, &config);
        let axis_order = derive_output_axis_order(recovered.output_shape.len(), schedule);
        let mut starts = vec![0usize; recovered.output_shape.len()];
        let mut tasks_for_loop = Vec::new();
        emit_tile_starts(
            &recovered.output_shape,
            &tile_shape,
            &axis_order,
            0,
            &mut starts,
            &mut tasks_for_loop,
        );

        for start in tasks_for_loop {
            let end = start
                .iter()
                .enumerate()
                .map(|(axis, s)| (*s + tile_shape[axis]).min(recovered.output_shape[axis]))
                .collect::<Vec<_>>();

            let tile_elements = start
                .iter()
                .enumerate()
                .map(|(axis, s)| end[axis] - s)
                .product::<usize>();

            out.tasks.push(ReductionTileTask {
                loop_index,
                output_tensor: recovered.output_tensor,
                output_shape: recovered.output_shape.clone(),
                output_start: start,
                output_end: end,
                reduction_start: 0,
                reduction_end: reduction.terms,
                estimated_fmas: tile_elements.saturating_mul(reduction.terms),
            });
        }

        out.stats.planned_loops += 1;
    }

    out.stats.total_tasks = out.tasks.len();
    out.stats.total_estimated_fmas = out.tasks.iter().map(|t| t.estimated_fmas).sum();
    Ok(out)
}

fn derive_output_axis_order(rank: usize, schedule: &LoopScheduleCandidate) -> Vec<usize> {
    if rank == 0 {
        return Vec::new();
    }

    let mut seen = vec![false; rank];
    let mut order = Vec::with_capacity(rank);

    for dim in &schedule.loop_order {
        let LoopDim::OutputAxis(axis) = dim else {
            continue;
        };
        if *axis < rank && !seen[*axis] {
            seen[*axis] = true;
            order.push(*axis);
        }
    }

    for (axis, axis_seen) in seen.iter().enumerate().take(rank) {
        if !axis_seen {
            order.push(axis);
        }
    }

    order
}

fn choose_tile_shape(
    output_shape: &[usize],
    schedule: &LoopScheduleCandidate,
    config: &TaskPlannerConfig,
) -> Vec<usize> {
    if output_shape.is_empty() {
        return Vec::new();
    }

    let mut tile = (0..output_shape.len())
        .map(|axis| {
            schedule
                .output_tiles
                .get(axis)
                .copied()
                .unwrap_or(1)
                .clamp(1, output_shape[axis].max(1))
        })
        .collect::<Vec<_>>();

    let mut tile_elems = tile.iter().product::<usize>();
    if tile_elems < config.min_tile_elements {
        for axis in (0..tile.len()).rev() {
            while tile[axis] < output_shape[axis]
                && tile_elems < config.min_tile_elements
                && tile_elems < config.max_tile_elements
            {
                let next = (tile[axis] * 2).min(output_shape[axis]);
                if next == tile[axis] {
                    break;
                }
                tile_elems = tile_elems / tile[axis] * next;
                tile[axis] = next;
            }
        }
    }

    while tile_elems > config.max_tile_elements {
        let mut shrunk = false;
        for t in &mut tile {
            if *t > 1 {
                let next = t.div_ceil(2);
                tile_elems = tile_elems / *t * next;
                *t = next;
                shrunk = true;
                if tile_elems <= config.max_tile_elements {
                    break;
                }
            }
        }
        if !shrunk {
            break;
        }
    }

    tile
}

fn emit_tile_starts(
    output_shape: &[usize],
    tile_shape: &[usize],
    axis_order: &[usize],
    depth: usize,
    start: &mut [usize],
    out: &mut Vec<Vec<usize>>,
) {
    if depth == axis_order.len() {
        out.push(start.to_vec());
        return;
    }

    let axis = axis_order[depth];
    let step = tile_shape[axis].max(1);
    for s in (0..output_shape[axis]).step_by(step) {
        start[axis] = s;
        emit_tile_starts(output_shape, tile_shape, axis_order, depth + 1, start, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GlobalId;
    use crate::milli_graph::ops::MatMul;

    fn build_rank2_matmul_graph(
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

    #[test]
    fn test_plan_matmul_reduction_tiles_cover_output_without_overlap() {
        let m = 64usize;
        let k = 96usize;
        let n = 80usize;
        let (graph, shapes, _a, _b, c) = build_rank2_matmul_graph(m, k, n, 301);

        let (_artifacts, plan) = plan_from_graph(&graph, &shapes, TaskPlannerConfig::default())
            .expect("v7 task plan from graph");
        assert!(!plan.tasks.is_empty());
        assert_eq!(plan.stats.planned_loops, 1);

        let mut covered = vec![false; m * n];
        for task in &plan.tasks {
            assert_eq!(task.output_tensor, c);
            assert_eq!(task.reduction_start, 0);
            assert_eq!(task.reduction_end, k);
            assert_eq!(task.output_shape, vec![m, n]);
            assert!(task.output_start[0] < task.output_end[0]);
            assert!(task.output_start[1] < task.output_end[1]);
            for i in task.output_start[0]..task.output_end[0] {
                for j in task.output_start[1]..task.output_end[1] {
                    let flat = i * n + j;
                    assert!(!covered[flat], "overlapping task coverage at ({i},{j})");
                    covered[flat] = true;
                }
            }
        }

        assert!(covered.iter().all(|v| *v), "not all outputs covered");
    }

    #[test]
    fn test_direct_fallback_builds_rank2_matmul_artifacts() {
        let m = 1024usize;
        let k = 1024usize;
        let n = 1024usize;
        let (graph, shapes, _a, _b, c) = build_rank2_matmul_graph(m, k, n, 311);

        let artifacts = build_artifacts(&graph, &shapes, 1).expect("direct fallback artifacts");
        assert_eq!(artifacts.schedule.stats.grouped_families, 1);
        assert_eq!(artifacts.schedule.stats.additive_reduction_families, 1);
        assert_eq!(artifacts.schedule.stats.unknown_families, 0);
        assert!(artifacts.schedule.stats.total_nano_ops > 1_000_000_000);
        assert_eq!(artifacts.schedule.loops[0].output_tensor, c);
        assert_eq!(artifacts.schedule.loops[0].output_shape, vec![m, n]);
        assert_eq!(artifacts.schedule.loops[0].covered_outputs, m * n);
        assert_eq!(
            artifacts.schedule.loops[0].schedule_candidates[0].rationale,
            "v7 direct fallback rank2 matmul schedule"
        );

        let plan = plan_from_artifacts(
            &artifacts,
            TaskPlannerConfig {
                min_tile_elements: 512,
                max_tile_elements: 4096,
            },
        )
        .expect("plan from fallback artifacts");
        assert!(!plan.tasks.is_empty());
        assert!(plan.stats.total_estimated_fmas >= m * n * k);
    }

    #[test]
    fn test_plan_rejects_invalid_tile_bounds() {
        let (graph, shapes, _a, _b, _c) = build_rank2_matmul_graph(4, 8, 3, 302);

        let artifacts = build_from_graph(&graph, &shapes).expect("v6 artifacts");
        let err = plan_from_artifacts(
            &artifacts,
            TaskPlannerConfig {
                min_tile_elements: 512,
                max_tile_elements: 64,
            },
        )
        .expect_err("expected invalid bounds error");

        match err {
            V7PlanError::Planner(msg) => assert!(msg.contains("min_tile_elements")),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_derive_output_axis_order_follows_schedule_order() {
        let schedule = LoopScheduleCandidate {
            loop_order: vec![
                LoopDim::OutputAxis(1),
                LoopDim::ReductionAxis,
                LoopDim::OutputAxis(0),
            ],
            output_tiles: vec![4, 8],
            reduction_tile: 16,
            reduction_unroll: 4,
            vectorize: None,
            score: 7,
            rationale: "test".to_string(),
        };
        assert_eq!(derive_output_axis_order(2, &schedule), vec![1, 0]);
    }

    #[test]
    fn test_emit_tile_starts_respects_axis_order() {
        let output_shape = vec![4usize, 4usize];
        let tile_shape = vec![2usize, 2usize];
        let axis_order = vec![1usize, 0usize];
        let mut start = vec![0usize; 2];
        let mut starts = Vec::new();
        emit_tile_starts(
            &output_shape,
            &tile_shape,
            &axis_order,
            0,
            &mut start,
            &mut starts,
        );

        assert_eq!(starts, vec![vec![0, 0], vec![2, 0], vec![0, 2], vec![2, 2]]);
    }

    #[test]
    fn test_plan_large_matmul_fma_and_coverage() {
        let m = 80usize;
        let k = 96usize;
        let n = 96usize;
        let (graph, shapes, _a, _b, c) = build_rank2_matmul_graph(m, k, n, 303);

        let (_artifacts, plan) = plan_from_graph(
            &graph,
            &shapes,
            TaskPlannerConfig {
                min_tile_elements: 512,
                max_tile_elements: 4096,
            },
        )
        .expect("v7 task plan for large matmul");

        assert_eq!(plan.stats.planned_loops, 1);
        assert!(plan.stats.total_tasks > 1);
        assert_eq!(plan.stats.total_estimated_fmas, m * n * k);

        let mut covered = vec![0u8; m * n];
        for task in &plan.tasks {
            assert_eq!(task.output_tensor, c);
            assert_eq!(task.reduction_start, 0);
            assert_eq!(task.reduction_end, k);
            for i in task.output_start[0]..task.output_end[0] {
                for j in task.output_start[1]..task.output_end[1] {
                    covered[i * n + j] = covered[i * n + j].saturating_add(1);
                }
            }
        }
        assert!(
            covered.iter().all(|&c| c == 1),
            "coverage must be exact-once"
        );
    }
}
