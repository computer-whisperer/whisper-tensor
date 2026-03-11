//! Generic schedule-intent recovery from whitewashed nano-op pools.
//!
//! This module is intentionally pattern-agnostic at the op level. It groups
//! stores by reconstructed expression families and infers high-level intent
//! (pointwise vs additive reduction) from structure and index behavior.

use crate::compiler::common::v1_frontend::nano_op::{
    NanoExpandError, NanoOp, NanoOpExpander, NanoValue, ScalarBinOp, ScalarUnaryOp,
};
use crate::graph::{GlobalId, Graph, Node};
use crate::milli_graph::MilliOpGraph;
use crate::milli_graph::ops::AnyMilliOp;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const DEFAULT_MAX_MATERIALIZED_NANO_OPS: usize = 8_000_000;
const MAX_MATERIALIZED_NANO_OPS_ENV: &str = "WT_V6_MAX_MATERIALIZED_NANO_OPS";

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error(transparent)]
    Expand(#[from] NanoExpandError),
    #[error("Duplicate definition for nano value {0:?}")]
    DuplicateDefinition(NanoValue),
    #[error("Missing definition for nano value {0:?}")]
    MissingDefinition(NanoValue),
    #[error("Missing shape for tensor {0}")]
    MissingShape(GlobalId),
    #[error("Invalid flat index {flat_index} for tensor {tensor} with shape {shape:?}")]
    InvalidFlatIndex {
        tensor: GlobalId,
        flat_index: usize,
        shape: Vec<usize>,
    },
    #[error(
        "v6 synthesis estimated {estimated_ops} nano ops exceeds materialization limit {limit_ops}; set {env_var} to override"
    )]
    NanoOpBudgetExceeded {
        estimated_ops: usize,
        limit_ops: usize,
        env_var: &'static str,
    },
    #[error("v6 synthesis nano-op estimate overflow ({context})")]
    NanoOpEstimateOverflow { context: String },
    #[error("Invalid value for {env_var}: {value}")]
    InvalidNanoOpLimit {
        env_var: &'static str,
        value: String,
    },
}

#[derive(Debug, Clone, Default)]
pub struct ScheduleStats {
    pub total_nano_ops: usize,
    pub total_stores: usize,
    pub grouped_families: usize,
    pub pointwise_families: usize,
    pub additive_reduction_families: usize,
    pub unknown_families: usize,
}

#[derive(Debug, Clone)]
pub struct ScheduleIR {
    pub loops: Vec<RecoveredLoop>,
    pub stats: ScheduleStats,
}

#[derive(Debug, Clone)]
pub struct RecoveredLoop {
    pub output_tensor: GlobalId,
    pub output_shape: Vec<usize>,
    pub covered_outputs: usize,
    pub load_tensors: Vec<GlobalId>,
    pub intent: LoopIntent,
    pub schedule_candidates: Vec<LoopScheduleCandidate>,
    pub selected_schedule: Option<usize>,
}

#[derive(Debug, Clone)]
pub enum LoopIntent {
    Pointwise(PointwiseIntent),
    AdditiveReduction(ReductionIntent),
    Unknown,
}

#[derive(Debug, Clone)]
pub struct PointwiseIntent {
    pub contiguous_output: bool,
    pub canonical_term: ReductionTermPattern,
    pub accesses: Vec<RecoveredTensorAccess>,
}

#[derive(Debug, Clone)]
pub struct ReductionIntent {
    pub terms: usize,
    pub canonical_term: ReductionTermPattern,
    pub term_load_tensors: Vec<GlobalId>,
    pub accesses: Vec<RecoveredTensorAccess>,
}

#[derive(Debug, Clone)]
pub struct RecoveredTensorAccess {
    pub tensor: GlobalId,
    pub tensor_shape: Vec<usize>,
    pub dim_roles: Vec<AccessDimRole>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccessDimRole {
    Constant {
        value: usize,
    },
    OutputAxis {
        axis: usize,
        stride: isize,
        offset: isize,
    },
    ReductionAxis {
        stride: isize,
        offset: isize,
    },
    AffineMixed {
        output_strides: Vec<(usize, isize)>,
        reduction_stride: isize,
        offset: isize,
    },
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoopDim {
    OutputAxis(usize),
    ReductionAxis,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorizePlan {
    pub axis: usize,
    pub width: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopScheduleCandidate {
    pub loop_order: Vec<LoopDim>,
    pub output_tiles: Vec<usize>,
    pub reduction_tile: usize,
    pub reduction_unroll: usize,
    pub vectorize: Option<VectorizePlan>,
    pub score: i64,
    pub rationale: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ReductionTermPattern {
    Load(usize),
    Literal(u64),
    Bin {
        op: ReductionBinOp,
        a: Box<ReductionTermPattern>,
        b: Box<ReductionTermPattern>,
    },
    Unary {
        op: ReductionUnaryOp,
        input: Box<ReductionTermPattern>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReductionBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReductionUnaryOp {
    Neg,
    Abs,
    Exp,
    Ln,
    Sqrt,
    Reciprocal,
    Tanh,
    Floor,
    Ceil,
}

#[derive(Debug, Clone)]
pub struct PipelineArtifacts {
    pub ordered_nano_ops: Vec<NanoOp>,
    pub whitewashed_nano_ops: Vec<NanoOp>,
    pub schedule: ScheduleIR,
}

pub fn build_from_graph(
    graph: &MilliOpGraph,
    shapes: &HashMap<GlobalId, Vec<usize>>,
) -> Result<PipelineArtifacts, PipelineError> {
    let nano_op_limit = read_nano_op_limit_from_env()?;
    build_from_graph_with_limit(graph, shapes, nano_op_limit)
}

fn build_from_graph_with_limit(
    graph: &MilliOpGraph,
    shapes: &HashMap<GlobalId, Vec<usize>>,
    nano_op_limit: usize,
) -> Result<PipelineArtifacts, PipelineError> {
    let estimated = estimate_nano_op_materialization(graph, shapes)?;
    if estimated > nano_op_limit {
        return Err(PipelineError::NanoOpBudgetExceeded {
            estimated_ops: estimated,
            limit_ops: nano_op_limit,
            env_var: MAX_MATERIALIZED_NANO_OPS_ENV,
        });
    }

    let schedule = analyze_from_graph_streaming(graph, shapes)?;

    Ok(PipelineArtifacts {
        // v6 now streams synthesis directly from the nano-op iterator.
        // Keep these vectors for API compatibility with existing callsites.
        ordered_nano_ops: Vec::new(),
        whitewashed_nano_ops: Vec::new(),
        schedule,
    })
}

/// Memory-bounded synthesis using sampled nano-op expansion.
///
/// Instead of materializing all nano-ops (which is O(M×N×K) for matmul),
/// this function:
/// 1. Expands only a sampled subset of output elements per op (~256),
///    preserving full reduction depth so access patterns are correct.
/// 2. Uses a single pass with `.remove()` instead of `.get().cloned()`,
///    eliminating the need for a global use-count vector.
/// 3. Stores at most `RESERVOIR_SIZE` instances per group, bounding memory
///    regardless of output tensor size.
///
/// For a [4096, 8192] × [8192, 22528] matmul, the full stream would be
/// ~3 trillion nano-ops and ~12 TB of StoreInstances. This function
/// processes ~256 × 32K = 8M nano-ops and stores ~256 × 131KB = 33 MB.
pub fn build_from_graph_sampled(
    graph: &MilliOpGraph,
    shapes: &HashMap<GlobalId, Vec<usize>>,
) -> Result<PipelineArtifacts, PipelineError> {
    let schedule = analyze_from_graph_sampled(graph, shapes)?;
    Ok(PipelineArtifacts {
        ordered_nano_ops: Vec::new(),
        whitewashed_nano_ops: Vec::new(),
        schedule,
    })
}

/// Target number of output elements to sample per op.
const SAMPLED_OUTPUT_TARGET: usize = 256;

/// Maximum instances kept per store group (reservoir size).
const RESERVOIR_SIZE: usize = 512;

fn analyze_from_graph_sampled(
    graph: &MilliOpGraph,
    shapes: &HashMap<GlobalId, Vec<usize>>,
) -> Result<ScheduleIR, PipelineError> {
    let mut expander = NanoOpExpander::new_sampled(shapes.clone(), SAMPLED_OUTPUT_TARGET);
    let mut value_exprs: HashMap<NanoValue, Expr> = HashMap::new();
    let mut raw_groups: HashMap<GroupKey, ReservoirStoreGroup> = HashMap::new();
    let mut total_stores = 0usize;
    let mut total_nano_ops = 0usize;

    for op in expander.expand_iter(graph) {
        let op = op?;
        total_nano_ops += 1;
        match op {
            NanoOp::Load {
                dst,
                tensor,
                flat_index,
            } => {
                value_exprs.insert(dst, Expr::Load { tensor, flat_index });
            }
            NanoOp::Literal { dst, value } => {
                value_exprs.insert(
                    dst,
                    Expr::Literal {
                        value_bits: value.to_bits(),
                    },
                );
            }
            NanoOp::UnaryOp { dst, op, input } => {
                // Move instead of clone — all current emitters produce single-use values.
                let input_expr = value_exprs
                    .remove(&input)
                    .ok_or(PipelineError::MissingDefinition(input))?;
                value_exprs.insert(
                    dst,
                    Expr::Unary {
                        op,
                        input: Box::new(input_expr),
                    },
                );
            }
            NanoOp::BinOp { dst, op, a, b } => {
                let a_expr = value_exprs
                    .remove(&a)
                    .ok_or(PipelineError::MissingDefinition(a))?;
                let b_expr = value_exprs
                    .remove(&b)
                    .ok_or(PipelineError::MissingDefinition(b))?;
                value_exprs.insert(
                    dst,
                    Expr::Bin {
                        op,
                        a: Box::new(a_expr),
                        b: Box::new(b_expr),
                    },
                );
            }
            NanoOp::Store {
                tensor,
                flat_index,
                src,
            } => {
                total_stores += 1;
                let src_expr = value_exprs
                    .remove(&src)
                    .ok_or(PipelineError::MissingDefinition(src))?;
                absorb_store_expr_reservoir(&mut raw_groups, tensor, flat_index, &src_expr);
            }
        }
    }

    finalize_grouped_schedule_sampled(raw_groups, total_nano_ops, total_stores, shapes)
}

fn absorb_store_expr_reservoir(
    raw_groups: &mut HashMap<GroupKey, ReservoirStoreGroup>,
    output_tensor: GlobalId,
    out_flat_index: usize,
    expr: &Expr,
) {
    let mut load_tensors = Vec::new();
    let pattern = pattern_expr(expr, &mut load_tensors);
    let mut load_indices = Vec::new();
    collect_load_indices(expr, &mut load_indices);

    let key = GroupKey {
        output_tensor,
        pattern_hash: stable_hash(&pattern),
        load_tensors_hash: stable_hash(&load_tensors),
    };

    let instance = StoreInstance {
        out_flat_index,
        load_indices,
    };

    let entry = raw_groups
        .entry(key)
        .or_insert_with(|| ReservoirStoreGroup {
            output_tensor,
            pattern: pattern.clone(),
            load_tensors: load_tensors.clone(),
            reservoir: Vec::with_capacity(RESERVOIR_SIZE),
            total_count: 0,
            rng_state: 0xDEAD_BEEF_CAFE_BABEu64,
        });

    // Hash collision fallback.
    if entry.pattern != pattern || entry.load_tensors != load_tensors {
        let mut alt_key = key;
        alt_key.pattern_hash ^= 0x9E37_79B9_7F4A_7C15;
        let alt_entry = raw_groups
            .entry(alt_key)
            .or_insert_with(|| ReservoirStoreGroup {
                output_tensor,
                pattern,
                load_tensors,
                reservoir: Vec::with_capacity(RESERVOIR_SIZE),
                total_count: 0,
                rng_state: 0xDEAD_BEEF_CAFE_BABEu64,
            });
        reservoir_insert(alt_entry, instance);
        return;
    }

    reservoir_insert(entry, instance);
}

/// Algorithm R reservoir sampling: keep first RESERVOIR_SIZE items,
/// then replace with probability RESERVOIR_SIZE / total_count.
fn reservoir_insert(group: &mut ReservoirStoreGroup, instance: StoreInstance) {
    group.total_count += 1;
    if group.reservoir.len() < RESERVOIR_SIZE {
        group.reservoir.push(instance);
    } else {
        // Simple xorshift64 PRNG for replacement decisions.
        group.rng_state ^= group.rng_state << 13;
        group.rng_state ^= group.rng_state >> 7;
        group.rng_state ^= group.rng_state << 17;
        let idx = (group.rng_state as usize) % group.total_count;
        if idx < RESERVOIR_SIZE {
            group.reservoir[idx] = instance;
        }
    }
}

fn finalize_grouped_schedule_sampled(
    raw_groups: HashMap<GroupKey, ReservoirStoreGroup>,
    total_nano_ops: usize,
    total_stores: usize,
    shapes: &HashMap<GlobalId, Vec<usize>>,
) -> Result<ScheduleIR, PipelineError> {
    // Convert ReservoirStoreGroups to StoreGroups for classify_group reuse.
    let mut groups: Vec<StoreGroup> = raw_groups
        .into_values()
        .map(|rg| StoreGroup {
            output_tensor: rg.output_tensor,
            pattern: rg.pattern,
            load_tensors: rg.load_tensors,
            instances: rg.reservoir,
        })
        .collect();
    groups.sort_by_key(|g| g.output_tensor);

    let mut loops = Vec::new();
    let mut pointwise = 0usize;
    let mut reductions = 0usize;
    let mut unknown = 0usize;

    for group in groups {
        let recovered = classify_group(&group, shapes)?;
        match &recovered.intent {
            LoopIntent::Pointwise(_) => pointwise += 1,
            LoopIntent::AdditiveReduction(_) => reductions += 1,
            LoopIntent::Unknown => unknown += 1,
        }
        loops.push(recovered);
    }

    let stats = ScheduleStats {
        total_nano_ops,
        total_stores,
        grouped_families: loops.len(),
        pointwise_families: pointwise,
        additive_reduction_families: reductions,
        unknown_families: unknown,
    };

    Ok(ScheduleIR { loops, stats })
}

#[derive(Debug, Clone)]
struct ReservoirStoreGroup {
    output_tensor: GlobalId,
    pattern: PatternExpr,
    load_tensors: Vec<GlobalId>,
    reservoir: Vec<StoreInstance>,
    total_count: usize,
    rng_state: u64,
}

fn analyze_from_graph_streaming(
    graph: &MilliOpGraph,
    shapes: &HashMap<GlobalId, Vec<usize>>,
) -> Result<ScheduleIR, PipelineError> {
    let mut pass1_expander = NanoOpExpander::new(shapes.clone());
    let (mut remaining_uses, total_nano_ops) =
        count_nano_value_uses(pass1_expander.expand_iter(graph))?;

    let mut pass2_expander = NanoOpExpander::new(shapes.clone());
    let mut value_exprs: HashMap<NanoValue, Expr> = HashMap::new();
    let mut raw_groups: HashMap<GroupKey, StoreGroup> = HashMap::new();
    let mut total_stores = 0usize;

    for op in pass2_expander.expand_iter(graph) {
        let op = op?;
        match op {
            NanoOp::Load {
                dst,
                tensor,
                flat_index,
            } => {
                insert_value_expr(&mut value_exprs, dst, Expr::Load { tensor, flat_index })?;
                maybe_drop_dead_value(&mut value_exprs, &remaining_uses, dst);
            }
            NanoOp::Literal { dst, value } => {
                insert_value_expr(
                    &mut value_exprs,
                    dst,
                    Expr::Literal {
                        value_bits: value.to_bits(),
                    },
                )?;
                maybe_drop_dead_value(&mut value_exprs, &remaining_uses, dst);
            }
            NanoOp::UnaryOp { dst, op, input } => {
                let input_expr = value_exprs
                    .get(&input)
                    .cloned()
                    .ok_or(PipelineError::MissingDefinition(input))?;
                insert_value_expr(
                    &mut value_exprs,
                    dst,
                    Expr::Unary {
                        op,
                        input: Box::new(input_expr),
                    },
                )?;
                consume_value_use(&mut remaining_uses, &mut value_exprs, input)?;
                maybe_drop_dead_value(&mut value_exprs, &remaining_uses, dst);
            }
            NanoOp::BinOp { dst, op, a, b } => {
                let a_expr = value_exprs
                    .get(&a)
                    .cloned()
                    .ok_or(PipelineError::MissingDefinition(a))?;
                let b_expr = value_exprs
                    .get(&b)
                    .cloned()
                    .ok_or(PipelineError::MissingDefinition(b))?;
                insert_value_expr(
                    &mut value_exprs,
                    dst,
                    Expr::Bin {
                        op,
                        a: Box::new(a_expr),
                        b: Box::new(b_expr),
                    },
                )?;
                consume_value_use(&mut remaining_uses, &mut value_exprs, a)?;
                consume_value_use(&mut remaining_uses, &mut value_exprs, b)?;
                maybe_drop_dead_value(&mut value_exprs, &remaining_uses, dst);
            }
            NanoOp::Store {
                tensor,
                flat_index,
                src,
            } => {
                total_stores = checked_add_usize(total_stores, 1, "total stores")?;
                let src_expr = value_exprs
                    .get(&src)
                    .ok_or(PipelineError::MissingDefinition(src))?;
                absorb_store_expr(&mut raw_groups, tensor, flat_index, src_expr);
                consume_value_use(&mut remaining_uses, &mut value_exprs, src)?;
            }
        }
    }

    finalize_grouped_schedule(raw_groups, total_nano_ops, total_stores, shapes)
}

fn count_nano_value_uses<I>(iter: I) -> Result<(Vec<usize>, usize), PipelineError>
where
    I: IntoIterator<Item = Result<NanoOp, NanoExpandError>>,
{
    let mut use_counts = Vec::<usize>::new();
    let mut defined = Vec::<bool>::new();
    let mut total_nano_ops = 0usize;

    for op in iter {
        let op = op?;
        total_nano_ops = checked_add_usize(total_nano_ops, 1, "total nano ops")?;

        if let Some(dst) = op.dst() {
            let dst_i = dst.0 as usize;
            if dst_i >= defined.len() {
                defined.resize(dst_i + 1, false);
            }
            if defined[dst_i] {
                return Err(PipelineError::DuplicateDefinition(dst));
            }
            defined[dst_i] = true;
            if dst_i >= use_counts.len() {
                use_counts.resize(dst_i + 1, 0);
            }
        }

        match op {
            NanoOp::Store { src, .. } => increment_use_count(&mut use_counts, src)?,
            NanoOp::BinOp { a, b, .. } => {
                increment_use_count(&mut use_counts, a)?;
                increment_use_count(&mut use_counts, b)?;
            }
            NanoOp::UnaryOp { input, .. } => increment_use_count(&mut use_counts, input)?,
            NanoOp::Load { .. } | NanoOp::Literal { .. } => {}
        }
    }

    Ok((use_counts, total_nano_ops))
}

fn increment_use_count(use_counts: &mut Vec<usize>, value: NanoValue) -> Result<(), PipelineError> {
    let idx = value.0 as usize;
    if idx >= use_counts.len() {
        use_counts.resize(idx + 1, 0);
    }
    use_counts[idx] = checked_add_usize(use_counts[idx], 1, "value use count")?;
    Ok(())
}

fn insert_value_expr(
    value_exprs: &mut HashMap<NanoValue, Expr>,
    dst: NanoValue,
    expr: Expr,
) -> Result<(), PipelineError> {
    if value_exprs.insert(dst, expr).is_some() {
        return Err(PipelineError::DuplicateDefinition(dst));
    }
    Ok(())
}

fn consume_value_use(
    remaining_uses: &mut [usize],
    value_exprs: &mut HashMap<NanoValue, Expr>,
    value: NanoValue,
) -> Result<(), PipelineError> {
    let idx = value.0 as usize;
    let Some(rem) = remaining_uses.get_mut(idx) else {
        return Err(PipelineError::MissingDefinition(value));
    };
    if *rem == 0 {
        return Err(PipelineError::MissingDefinition(value));
    }
    *rem -= 1;
    if *rem == 0 {
        value_exprs.remove(&value);
    }
    Ok(())
}

fn maybe_drop_dead_value(
    value_exprs: &mut HashMap<NanoValue, Expr>,
    remaining_uses: &[usize],
    value: NanoValue,
) {
    let idx = value.0 as usize;
    if remaining_uses.get(idx).copied().unwrap_or(0) == 0 {
        value_exprs.remove(&value);
    }
}

fn read_nano_op_limit_from_env() -> Result<usize, PipelineError> {
    match std::env::var(MAX_MATERIALIZED_NANO_OPS_ENV) {
        Ok(raw) => {
            let parsed = raw
                .parse::<usize>()
                .map_err(|_| PipelineError::InvalidNanoOpLimit {
                    env_var: MAX_MATERIALIZED_NANO_OPS_ENV,
                    value: raw.clone(),
                })?;
            if parsed == 0 {
                return Err(PipelineError::InvalidNanoOpLimit {
                    env_var: MAX_MATERIALIZED_NANO_OPS_ENV,
                    value: raw,
                });
            }
            Ok(parsed)
        }
        Err(std::env::VarError::NotPresent) => Ok(DEFAULT_MAX_MATERIALIZED_NANO_OPS),
        Err(std::env::VarError::NotUnicode(_)) => Err(PipelineError::InvalidNanoOpLimit {
            env_var: MAX_MATERIALIZED_NANO_OPS_ENV,
            value: "<non-unicode>".to_string(),
        }),
    }
}

fn estimate_nano_op_materialization(
    graph: &MilliOpGraph,
    shapes: &HashMap<GlobalId, Vec<usize>>,
) -> Result<usize, PipelineError> {
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
) -> Result<usize, PipelineError> {
    let kind = op.op_kind();
    let outputs: Vec<GlobalId> = op.outputs().collect();
    let inputs: Vec<GlobalId> = op.inputs().collect();

    let scalar_elems = |tensor: GlobalId| -> Result<usize, PipelineError> {
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
            let out =
                outputs
                    .first()
                    .copied()
                    .ok_or_else(|| PipelineError::NanoOpEstimateOverflow {
                        context: format!("missing output for op {kind}"),
                    })?;
            let elems = scalar_elems(out)?;
            checked_mul_usize(elems, 4, &format!("binary op count for {kind}"))
        }
        "Neg" | "Abs" | "Exp" | "Ln" | "Sqrt" | "Reciprocal" | "Floor" | "Ceil" | "Tanh" => {
            let out =
                outputs
                    .first()
                    .copied()
                    .ok_or_else(|| PipelineError::NanoOpEstimateOverflow {
                        context: format!("missing output for op {kind}"),
                    })?;
            let elems = scalar_elems(out)?;
            checked_mul_usize(elems, 3, &format!("unary op count for {kind}"))
        }
        "MatMul" => {
            if inputs.len() < 2 || outputs.is_empty() {
                return Err(PipelineError::NanoOpEstimateOverflow {
                    context: "matmul inputs/outputs missing".to_string(),
                });
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

fn checked_mul_usize(a: usize, b: usize, context: &str) -> Result<usize, PipelineError> {
    a.checked_mul(b)
        .ok_or_else(|| PipelineError::NanoOpEstimateOverflow {
            context: context.to_string(),
        })
}

fn checked_add_usize(a: usize, b: usize, context: &str) -> Result<usize, PipelineError> {
    a.checked_add(b)
        .ok_or_else(|| PipelineError::NanoOpEstimateOverflow {
            context: context.to_string(),
        })
}

pub fn analyze_from_pool(
    nano_ops: &[NanoOp],
    shapes: &HashMap<GlobalId, Vec<usize>>,
) -> Result<ScheduleIR, PipelineError> {
    let defs = build_defs(nano_ops)?;
    let mut expr_cache = HashMap::new();
    let mut raw_groups: HashMap<GroupKey, StoreGroup> = HashMap::new();
    let mut total_stores = 0usize;

    for op in nano_ops {
        let (output_tensor, out_flat_index, src) = match op {
            NanoOp::Store {
                tensor,
                flat_index,
                src,
            } => (*tensor, *flat_index, *src),
            _ => continue,
        };

        total_stores += 1;
        let expr = build_expr(src, &defs, &mut expr_cache)?;

        absorb_store_expr(&mut raw_groups, output_tensor, out_flat_index, &expr);
    }

    finalize_grouped_schedule(raw_groups, nano_ops.len(), total_stores, shapes)
}

fn absorb_store_expr(
    raw_groups: &mut HashMap<GroupKey, StoreGroup>,
    output_tensor: GlobalId,
    out_flat_index: usize,
    expr: &Expr,
) {
    let mut load_tensors = Vec::new();
    let pattern = pattern_expr(expr, &mut load_tensors);
    let mut load_indices = Vec::new();
    collect_load_indices(expr, &mut load_indices);

    let key = GroupKey {
        output_tensor,
        pattern_hash: stable_hash(&pattern),
        load_tensors_hash: stable_hash(&load_tensors),
    };

    let entry = raw_groups.entry(key).or_insert_with(|| StoreGroup {
        output_tensor,
        pattern: pattern.clone(),
        load_tensors: load_tensors.clone(),
        instances: Vec::new(),
    });

    // Hash collision fallback.
    if entry.pattern != pattern || entry.load_tensors != load_tensors {
        let mut alt_key = key;
        alt_key.pattern_hash ^= 0x9E37_79B9_7F4A_7C15;
        raw_groups
            .entry(alt_key)
            .or_insert_with(|| StoreGroup {
                output_tensor,
                pattern,
                load_tensors,
                instances: Vec::new(),
            })
            .instances
            .push(StoreInstance {
                out_flat_index,
                load_indices,
            });
        return;
    }

    entry.instances.push(StoreInstance {
        out_flat_index,
        load_indices,
    });
}

fn finalize_grouped_schedule(
    raw_groups: HashMap<GroupKey, StoreGroup>,
    total_nano_ops: usize,
    total_stores: usize,
    shapes: &HashMap<GlobalId, Vec<usize>>,
) -> Result<ScheduleIR, PipelineError> {
    let mut loops = Vec::new();
    let mut grouped: Vec<_> = raw_groups.into_values().collect();
    grouped.sort_by_key(|g| g.output_tensor);

    let mut pointwise = 0usize;
    let mut reductions = 0usize;
    let mut unknown = 0usize;

    for group in grouped {
        let recovered = classify_group(&group, shapes)?;
        match &recovered.intent {
            LoopIntent::Pointwise(_) => pointwise += 1,
            LoopIntent::AdditiveReduction(_) => reductions += 1,
            LoopIntent::Unknown => unknown += 1,
        }
        loops.push(recovered);
    }

    let stats = ScheduleStats {
        total_nano_ops,
        total_stores,
        grouped_families: loops.len(),
        pointwise_families: pointwise,
        additive_reduction_families: reductions,
        unknown_families: unknown,
    };

    Ok(ScheduleIR { loops, stats })
}

#[derive(Debug, Clone)]
enum Expr {
    Load {
        tensor: GlobalId,
        flat_index: usize,
    },
    Literal {
        value_bits: u64,
    },
    Bin {
        op: ScalarBinOp,
        a: Box<Expr>,
        b: Box<Expr>,
    },
    Unary {
        op: ScalarUnaryOp,
        input: Box<Expr>,
    },
}

#[derive(Debug, Clone)]
struct StoreInstance {
    out_flat_index: usize,
    load_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
struct StoreGroup {
    output_tensor: GlobalId,
    pattern: PatternExpr,
    load_tensors: Vec<GlobalId>,
    instances: Vec<StoreInstance>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum PatternExpr {
    Load(usize),
    Literal(u64),
    Bin {
        op: BinTag,
        a: Box<PatternExpr>,
        b: Box<PatternExpr>,
    },
    Unary {
        op: UnaryTag,
        input: Box<PatternExpr>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum BinTag {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum UnaryTag {
    Neg,
    Abs,
    Exp,
    Ln,
    Sqrt,
    Reciprocal,
    Tanh,
    Floor,
    Ceil,
}

impl From<ScalarBinOp> for BinTag {
    fn from(value: ScalarBinOp) -> Self {
        match value {
            ScalarBinOp::Add => BinTag::Add,
            ScalarBinOp::Sub => BinTag::Sub,
            ScalarBinOp::Mul => BinTag::Mul,
            ScalarBinOp::Div => BinTag::Div,
            ScalarBinOp::Max => BinTag::Max,
            ScalarBinOp::Min => BinTag::Min,
        }
    }
}

impl From<ScalarUnaryOp> for UnaryTag {
    fn from(value: ScalarUnaryOp) -> Self {
        match value {
            ScalarUnaryOp::Neg => UnaryTag::Neg,
            ScalarUnaryOp::Abs => UnaryTag::Abs,
            ScalarUnaryOp::Exp => UnaryTag::Exp,
            ScalarUnaryOp::Ln => UnaryTag::Ln,
            ScalarUnaryOp::Sqrt => UnaryTag::Sqrt,
            ScalarUnaryOp::Reciprocal => UnaryTag::Reciprocal,
            ScalarUnaryOp::Tanh => UnaryTag::Tanh,
            ScalarUnaryOp::Floor => UnaryTag::Floor,
            ScalarUnaryOp::Ceil => UnaryTag::Ceil,
        }
    }
}

impl From<BinTag> for ReductionBinOp {
    fn from(value: BinTag) -> Self {
        match value {
            BinTag::Add => ReductionBinOp::Add,
            BinTag::Sub => ReductionBinOp::Sub,
            BinTag::Mul => ReductionBinOp::Mul,
            BinTag::Div => ReductionBinOp::Div,
            BinTag::Max => ReductionBinOp::Max,
            BinTag::Min => ReductionBinOp::Min,
        }
    }
}

impl From<UnaryTag> for ReductionUnaryOp {
    fn from(value: UnaryTag) -> Self {
        match value {
            UnaryTag::Neg => ReductionUnaryOp::Neg,
            UnaryTag::Abs => ReductionUnaryOp::Abs,
            UnaryTag::Exp => ReductionUnaryOp::Exp,
            UnaryTag::Ln => ReductionUnaryOp::Ln,
            UnaryTag::Sqrt => ReductionUnaryOp::Sqrt,
            UnaryTag::Reciprocal => ReductionUnaryOp::Reciprocal,
            UnaryTag::Tanh => ReductionUnaryOp::Tanh,
            UnaryTag::Floor => ReductionUnaryOp::Floor,
            UnaryTag::Ceil => ReductionUnaryOp::Ceil,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GroupKey {
    output_tensor: GlobalId,
    pattern_hash: u64,
    load_tensors_hash: u64,
}

#[derive(Debug, Clone)]
struct ReductionMatch {
    terms: usize,
    canonical_term: ReductionTermPattern,
    term_load_tensors: Vec<GlobalId>,
    term_slots: Vec<Vec<usize>>,
}

fn stable_hash<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

fn build_defs(nano_ops: &[NanoOp]) -> Result<HashMap<NanoValue, NanoOp>, PipelineError> {
    let mut defs = HashMap::new();
    for op in nano_ops {
        if let Some(dst) = op.dst() {
            if defs.contains_key(&dst) {
                return Err(PipelineError::DuplicateDefinition(dst));
            }
            defs.insert(dst, op.clone());
        }
    }
    Ok(defs)
}

fn build_expr(
    value: NanoValue,
    defs: &HashMap<NanoValue, NanoOp>,
    cache: &mut HashMap<NanoValue, Expr>,
) -> Result<Expr, PipelineError> {
    if let Some(expr) = cache.get(&value) {
        return Ok(expr.clone());
    }

    let op = defs
        .get(&value)
        .ok_or(PipelineError::MissingDefinition(value))?;

    let expr = match op {
        NanoOp::Load {
            tensor, flat_index, ..
        } => Expr::Load {
            tensor: *tensor,
            flat_index: *flat_index,
        },
        NanoOp::Literal { value, .. } => Expr::Literal {
            value_bits: value.to_bits(),
        },
        NanoOp::BinOp { op, a, b, .. } => {
            let a_expr = build_expr(*a, defs, cache)?;
            let b_expr = build_expr(*b, defs, cache)?;
            Expr::Bin {
                op: *op,
                a: Box::new(a_expr),
                b: Box::new(b_expr),
            }
        }
        NanoOp::UnaryOp { op, input, .. } => {
            let input_expr = build_expr(*input, defs, cache)?;
            Expr::Unary {
                op: *op,
                input: Box::new(input_expr),
            }
        }
        NanoOp::Store { .. } => return Err(PipelineError::MissingDefinition(value)),
    };

    cache.insert(value, expr.clone());
    Ok(expr)
}

fn pattern_expr(expr: &Expr, load_tensors: &mut Vec<GlobalId>) -> PatternExpr {
    match expr {
        Expr::Load { tensor, .. } => {
            let slot = load_tensors.len();
            load_tensors.push(*tensor);
            PatternExpr::Load(slot)
        }
        Expr::Literal { value_bits } => PatternExpr::Literal(*value_bits),
        Expr::Bin { op, a, b } => PatternExpr::Bin {
            op: (*op).into(),
            a: Box::new(pattern_expr(a, load_tensors)),
            b: Box::new(pattern_expr(b, load_tensors)),
        },
        Expr::Unary { op, input } => PatternExpr::Unary {
            op: (*op).into(),
            input: Box::new(pattern_expr(input, load_tensors)),
        },
    }
}

fn collect_load_indices(expr: &Expr, out: &mut Vec<usize>) {
    match expr {
        Expr::Load { flat_index, .. } => out.push(*flat_index),
        Expr::Literal { .. } => {}
        Expr::Bin { a, b, .. } => {
            collect_load_indices(a, out);
            collect_load_indices(b, out);
        }
        Expr::Unary { input, .. } => collect_load_indices(input, out),
    }
}

fn classify_group(
    group: &StoreGroup,
    shapes: &HashMap<GlobalId, Vec<usize>>,
) -> Result<RecoveredLoop, PipelineError> {
    let output_shape = shapes
        .get(&group.output_tensor)
        .cloned()
        .ok_or(PipelineError::MissingShape(group.output_tensor))?;
    let output_elements = output_shape.iter().product::<usize>();

    let intent = if let Some(reduction_match) = try_classify_additive_reduction(group) {
        let accesses = infer_reduction_accesses(group, &output_shape, shapes, &reduction_match)?;
        LoopIntent::AdditiveReduction(ReductionIntent {
            terms: reduction_match.terms,
            canonical_term: reduction_match.canonical_term,
            term_load_tensors: reduction_match.term_load_tensors,
            accesses,
        })
    } else if let Some(pointwise) =
        try_classify_pointwise(group, &output_shape, output_elements, shapes)?
    {
        LoopIntent::Pointwise(pointwise)
    } else {
        LoopIntent::Unknown
    };
    let schedule_candidates = synthesize_schedule_candidates(&intent, &output_shape);
    let selected_schedule = choose_best_schedule(&schedule_candidates);

    Ok(RecoveredLoop {
        output_tensor: group.output_tensor,
        output_shape,
        covered_outputs: group.instances.len(),
        load_tensors: group.load_tensors.clone(),
        intent,
        schedule_candidates,
        selected_schedule,
    })
}

fn try_classify_pointwise(
    group: &StoreGroup,
    output_shape: &[usize],
    output_elements: usize,
    shapes: &HashMap<GlobalId, Vec<usize>>,
) -> Result<Option<PointwiseIntent>, PipelineError> {
    let mut inst = group.instances.clone();
    if inst.is_empty() {
        return Ok(None);
    }
    inst.sort_by_key(|i| i.out_flat_index);
    if inst
        .windows(2)
        .any(|w| w[0].out_flat_index == w[1].out_flat_index)
    {
        return Ok(None);
    }
    let contiguous_output = is_contiguous_output(&inst, output_elements);

    let mut output_coords = Vec::with_capacity(inst.len());
    for sample in &inst {
        let coords = unflatten_index(sample.out_flat_index, output_shape).ok_or(
            PipelineError::InvalidFlatIndex {
                tensor: group.output_tensor,
                flat_index: sample.out_flat_index,
                shape: output_shape.to_vec(),
            },
        )?;
        output_coords.push(coords);
    }

    let mut term_tensors = Vec::new();
    let mut slot_map = HashMap::new();
    let canonical_term = if let Some(term) =
        canonicalize_term(&group.pattern, group, &mut slot_map, &mut term_tensors)
    {
        term
    } else {
        return Ok(None);
    };
    if term_tensors.is_empty() {
        return Ok(None);
    }

    let mut orig_slot_for_canonical = vec![usize::MAX; term_tensors.len()];
    for (orig_slot, canonical_slot) in slot_map {
        if canonical_slot >= orig_slot_for_canonical.len() {
            return Ok(None);
        }
        orig_slot_for_canonical[canonical_slot] = orig_slot;
    }
    if orig_slot_for_canonical.contains(&usize::MAX) {
        return Ok(None);
    }

    let output_rank = output_shape.len();
    let output_step = sampling_step(inst.len(), 128);
    let mut accesses = Vec::with_capacity(term_tensors.len());
    for (canonical_slot, &tensor) in term_tensors.iter().enumerate() {
        let tensor_shape = shapes
            .get(&tensor)
            .cloned()
            .ok_or(PipelineError::MissingShape(tensor))?;
        let orig_slot = orig_slot_for_canonical[canonical_slot];
        let var_count = output_rank;
        let mut sample_vars_flat = Vec::new();
        let mut sample_dim_values = vec![Vec::new(); tensor_shape.len()];
        let mut samples_valid = true;

        for inst_i in (0..inst.len()).step_by(output_step) {
            let sample = &inst[inst_i];
            let Some(&flat_index) = sample.load_indices.get(orig_slot) else {
                samples_valid = false;
                break;
            };
            let coords = unflatten_index(flat_index, &tensor_shape).ok_or(
                PipelineError::InvalidFlatIndex {
                    tensor,
                    flat_index,
                    shape: tensor_shape.clone(),
                },
            )?;
            sample_vars_flat.extend(output_coords[inst_i].iter().map(|&v| v as isize));
            for dim in 0..coords.len() {
                sample_dim_values[dim].push(coords[dim] as isize);
            }
        }

        let sample_count = if var_count == 0 {
            sample_dim_values.first().map(|x| x.len()).unwrap_or(0)
        } else {
            sample_vars_flat.len() / var_count
        };
        let dim_roles = if !samples_valid || sample_count == 0 {
            vec![AccessDimRole::Unknown; tensor_shape.len()]
        } else {
            let mut roles = Vec::with_capacity(tensor_shape.len());
            for dim_values in sample_dim_values.iter().take(tensor_shape.len()) {
                let role = infer_affine_dim_role_pointwise(
                    &sample_vars_flat,
                    var_count,
                    dim_values,
                    output_rank,
                );
                roles.push(role);
            }
            roles
        };
        if dim_roles
            .iter()
            .any(|r| matches!(r, AccessDimRole::Unknown))
        {
            return Ok(None);
        }

        accesses.push(RecoveredTensorAccess {
            tensor,
            tensor_shape,
            dim_roles,
        });
    }

    Ok(Some(PointwiseIntent {
        contiguous_output,
        canonical_term,
        accesses,
    }))
}

fn is_contiguous_output(instances: &[StoreInstance], output_elements: usize) -> bool {
    !instances.is_empty()
        && instances[0].out_flat_index == 0
        && instances.len() == output_elements
        && instances
            .iter()
            .enumerate()
            .all(|(i, x)| x.out_flat_index == i)
}

fn try_classify_additive_reduction(group: &StoreGroup) -> Option<ReductionMatch> {
    let mut terms = Vec::new();
    collect_add_terms(&group.pattern, &mut terms);
    if terms.len() < 2 {
        return None;
    }

    let mut term_slots = Vec::with_capacity(terms.len());
    for term in &terms {
        let mut slots = Vec::new();
        collect_term_slots(term, &mut slots);
        term_slots.push(slots);
    }

    let mut first_tensors = Vec::new();
    let mut first_slot_map = HashMap::new();
    let first_pattern =
        canonicalize_term(terms[0], group, &mut first_slot_map, &mut first_tensors)?;
    if term_slots[0].len() != first_tensors.len() {
        return None;
    }

    for (idx, term) in terms.iter().enumerate().skip(1) {
        let mut tensors = Vec::new();
        let mut slot_map = HashMap::new();
        let pattern = canonicalize_term(term, group, &mut slot_map, &mut tensors)?;
        if pattern != first_pattern
            || tensors != first_tensors
            || term_slots[idx].len() != first_tensors.len()
        {
            return None;
        }
    }

    Some(ReductionMatch {
        terms: terms.len(),
        canonical_term: first_pattern,
        term_load_tensors: first_tensors,
        term_slots,
    })
}

fn synthesize_schedule_candidates(
    intent: &LoopIntent,
    output_shape: &[usize],
) -> Vec<LoopScheduleCandidate> {
    match intent {
        LoopIntent::Pointwise(pointwise) => {
            synthesize_pointwise_candidates(output_shape, pointwise.contiguous_output)
        }
        LoopIntent::AdditiveReduction(reduction) => {
            synthesize_additive_reduction_candidates(output_shape, reduction)
        }
        LoopIntent::Unknown => vec![LoopScheduleCandidate {
            loop_order: (0..output_shape.len())
                .map(LoopDim::OutputAxis)
                .collect::<Vec<_>>(),
            output_tiles: vec![1; output_shape.len()],
            reduction_tile: 1,
            reduction_unroll: 1,
            vectorize: None,
            score: 1,
            rationale: "Fallback schedule for unknown intent".to_string(),
        }],
    }
}

fn infer_affine_dim_role_pointwise(
    vars_flat: &[isize],
    var_count: usize,
    values: &[isize],
    output_rank: usize,
) -> AccessDimRole {
    let Some((offset, coeffs)) = infer_affine_coefficients(vars_flat, var_count, values) else {
        return AccessDimRole::Unknown;
    };

    if coeffs.len() != output_rank {
        return AccessDimRole::Unknown;
    }

    let output_strides: Vec<(usize, isize)> = coeffs
        .iter()
        .enumerate()
        .filter_map(|(axis, &stride)| (stride != 0).then_some((axis, stride)))
        .collect();

    if output_strides.is_empty() {
        if offset >= 0 {
            return AccessDimRole::Constant {
                value: offset as usize,
            };
        }
        return AccessDimRole::Unknown;
    }

    if output_strides.len() == 1 {
        let (axis, stride) = output_strides[0];
        return AccessDimRole::OutputAxis {
            axis,
            stride,
            offset,
        };
    }

    AccessDimRole::AffineMixed {
        output_strides,
        reduction_stride: 0,
        offset,
    }
}

fn choose_best_schedule(candidates: &[LoopScheduleCandidate]) -> Option<usize> {
    candidates
        .iter()
        .enumerate()
        .max_by_key(|(_, candidate)| candidate.score)
        .map(|(idx, _)| idx)
}

fn synthesize_pointwise_candidates(
    output_shape: &[usize],
    contiguous_output: bool,
) -> Vec<LoopScheduleCandidate> {
    let rank = output_shape.len();
    let natural_order: Vec<LoopDim> = (0..rank).map(LoopDim::OutputAxis).collect();
    let mut output_tiles = vec![1usize; rank];
    let mut candidates = Vec::new();

    let vectorize = rank
        .checked_sub(1)
        .and_then(|axis| {
            choose_vector_width(*output_shape.get(axis).unwrap_or(&1)).map(|width| (axis, width))
        })
        .map(|(axis, width)| {
            if rank > 0 {
                output_tiles[axis] = (width * 4).min(output_shape[axis].max(1));
            }
            VectorizePlan { axis, width }
        });

    let mut score = 70i64;
    if contiguous_output {
        score += 30;
    }
    if vectorize.is_some() {
        score += 20;
    }

    candidates.push(LoopScheduleCandidate {
        loop_order: natural_order.clone(),
        output_tiles: output_tiles.clone(),
        reduction_tile: 1,
        reduction_unroll: 1,
        vectorize: vectorize.clone(),
        score,
        rationale: if contiguous_output {
            "Natural output order with contiguous-store vectorization".to_string()
        } else {
            "Natural output order without contiguous guarantee".to_string()
        },
    });

    if rank >= 2 {
        let mut swapped = (0..rank).collect::<Vec<_>>();
        swapped.swap(rank - 2, rank - 1);
        candidates.push(LoopScheduleCandidate {
            loop_order: swapped.into_iter().map(LoopDim::OutputAxis).collect(),
            output_tiles: vec![1; rank],
            reduction_tile: 1,
            reduction_unroll: 1,
            vectorize: None,
            score: if contiguous_output { 40 } else { 55 },
            rationale: "Alternative output-axis order for locality exploration".to_string(),
        });
    }

    candidates
}

fn synthesize_additive_reduction_candidates(
    output_shape: &[usize],
    reduction: &ReductionIntent,
) -> Vec<LoopScheduleCandidate> {
    let output_rank = output_shape.len();
    let reduction_tile = choose_reduction_tile(reduction.terms);
    let reduction_unroll = reduction_tile.min(4);
    let vectorize = pick_vector_axis(output_shape, &reduction.accesses);
    let contraction_like = is_contraction_like(reduction, output_rank);

    let mut candidates = Vec::new();

    let mut natural_order: Vec<LoopDim> = (0..output_rank).map(LoopDim::OutputAxis).collect();
    natural_order.push(LoopDim::ReductionAxis);
    let mut score_natural = 80i64;
    if reduction_tile > 1 {
        score_natural += 10;
    }
    if vectorize.is_some() {
        score_natural += 10;
    }
    if contraction_like {
        score_natural += 5;
    }

    candidates.push(LoopScheduleCandidate {
        loop_order: natural_order,
        output_tiles: default_output_tiles(output_shape, vectorize.clone()),
        reduction_tile,
        reduction_unroll,
        vectorize: vectorize.clone(),
        score: score_natural,
        rationale: "Output-major with reduction innermost".to_string(),
    });

    if let Some(v) = vectorize.clone() {
        let mut order = Vec::with_capacity(output_rank + 1);
        for axis in 0..output_rank {
            if axis != v.axis {
                order.push(LoopDim::OutputAxis(axis));
            }
        }
        order.push(LoopDim::ReductionAxis);
        order.push(LoopDim::OutputAxis(v.axis));

        let mut score = 90i64;
        if contraction_like {
            score += 25;
        }
        if reduction_tile > 1 {
            score += 8;
        }
        if v.width >= 4 {
            score += 12;
        }

        candidates.push(LoopScheduleCandidate {
            loop_order: order,
            output_tiles: default_output_tiles(output_shape, Some(v.clone())),
            reduction_tile,
            reduction_unroll,
            vectorize: Some(v),
            score,
            rationale: "Reduction-interleaved order for vectorized output micro-kernel".to_string(),
        });
    }

    if contraction_like && output_rank >= 2 {
        let mut score = 95i64;
        if reduction_tile > 1 {
            score += 12;
        }
        if vectorize.is_some() {
            score += 10;
        }

        let mut output_tiles = vec![1usize; output_rank];
        output_tiles[0] = output_shape[0].clamp(1, 8);
        if output_rank > 1 {
            output_tiles[1] = output_shape[1].clamp(1, 8);
        }
        if let Some(v) = vectorize.clone() {
            output_tiles[v.axis] = (v.width * 4).min(output_shape[v.axis].max(1));
        }

        let mut order = Vec::with_capacity(output_rank + 1);
        order.push(LoopDim::OutputAxis(0));
        order.push(LoopDim::ReductionAxis);
        for axis in 1..output_rank {
            order.push(LoopDim::OutputAxis(axis));
        }

        candidates.push(LoopScheduleCandidate {
            loop_order: order,
            output_tiles,
            reduction_tile,
            reduction_unroll,
            vectorize,
            score,
            rationale: "Contraction-like blocked schedule (output tiles + reduction block)"
                .to_string(),
        });
    }

    dedup_candidates(candidates)
}

fn dedup_candidates(candidates: Vec<LoopScheduleCandidate>) -> Vec<LoopScheduleCandidate> {
    let mut out = Vec::new();
    for candidate in candidates {
        let is_dup = out.iter().any(|existing: &LoopScheduleCandidate| {
            existing.loop_order == candidate.loop_order
                && existing.output_tiles == candidate.output_tiles
                && existing.reduction_tile == candidate.reduction_tile
                && existing.reduction_unroll == candidate.reduction_unroll
                && existing.vectorize == candidate.vectorize
        });
        if !is_dup {
            out.push(candidate);
        }
    }
    out
}

fn default_output_tiles(output_shape: &[usize], vectorize: Option<VectorizePlan>) -> Vec<usize> {
    let mut tiles = vec![1usize; output_shape.len()];
    if let Some(v) = vectorize
        && let Some(extent) = output_shape.get(v.axis)
    {
        tiles[v.axis] = (v.width * 4).min((*extent).max(1));
    }
    tiles
}

fn choose_reduction_tile(terms: usize) -> usize {
    for tile in [64usize, 32, 16, 8, 4, 2] {
        if terms >= tile {
            return tile;
        }
    }
    terms.max(1)
}

fn choose_vector_width(extent: usize) -> Option<usize> {
    for width in [16usize, 8, 4, 2] {
        if extent >= width && extent.is_multiple_of(width) {
            return Some(width);
        }
    }
    [8usize, 4, 2].into_iter().find(|&width| extent >= width)
}

fn pick_vector_axis(
    output_shape: &[usize],
    accesses: &[RecoveredTensorAccess],
) -> Option<VectorizePlan> {
    for axis in (0..output_shape.len()).rev() {
        let extent = output_shape[axis];
        let Some(width) = choose_vector_width(extent) else {
            continue;
        };
        if accesses
            .iter()
            .all(|access| axis_vector_friendly(access, axis))
        {
            return Some(VectorizePlan { axis, width });
        }
    }
    None
}

fn axis_vector_friendly(access: &RecoveredTensorAccess, axis: usize) -> bool {
    for role in &access.dim_roles {
        match role {
            AccessDimRole::Constant { .. } => {}
            AccessDimRole::OutputAxis {
                axis: a, stride, ..
            } if *a == axis => {
                if stride.abs() != 1 {
                    return false;
                }
            }
            AccessDimRole::OutputAxis { .. } => {}
            AccessDimRole::ReductionAxis { .. } => {}
            AccessDimRole::AffineMixed {
                output_strides,
                reduction_stride,
                ..
            } => {
                if *reduction_stride != 0 {
                    return false;
                }
                for (a, stride) in output_strides {
                    if *a == axis && stride.abs() != 1 {
                        return false;
                    }
                }
            }
            AccessDimRole::Unknown => return false,
        }
    }
    true
}

fn is_contraction_like(reduction: &ReductionIntent, output_rank: usize) -> bool {
    if output_rank < 2 {
        return false;
    }
    let mut dependent = Vec::new();
    for access in &reduction.accesses {
        let mut axes = vec![false; output_rank];
        let mut has_reduction = false;
        for role in &access.dim_roles {
            match role {
                AccessDimRole::OutputAxis { axis, stride, .. } => {
                    if *axis < output_rank && *stride != 0 {
                        axes[*axis] = true;
                    }
                }
                AccessDimRole::ReductionAxis { stride, .. } => {
                    if *stride != 0 {
                        has_reduction = true;
                    }
                }
                AccessDimRole::AffineMixed {
                    output_strides,
                    reduction_stride,
                    ..
                } => {
                    if *reduction_stride != 0 {
                        has_reduction = true;
                    }
                    for (axis, stride) in output_strides {
                        if *axis < output_rank && *stride != 0 {
                            axes[*axis] = true;
                        }
                    }
                }
                AccessDimRole::Constant { .. } | AccessDimRole::Unknown => {}
            }
        }
        if has_reduction && axes.iter().any(|v| *v) {
            dependent.push(axes);
        }
    }

    for i in 0..dependent.len() {
        for j in i + 1..dependent.len() {
            let mut disjoint = true;
            let mut covered = 0usize;
            for (&di, &dj) in dependent[i].iter().zip(&dependent[j]).take(output_rank) {
                if di && dj {
                    disjoint = false;
                    break;
                }
                if di || dj {
                    covered += 1;
                }
            }
            if disjoint && covered >= 2 {
                return true;
            }
        }
    }
    false
}

fn collect_add_terms<'a>(expr: &'a PatternExpr, out: &mut Vec<&'a PatternExpr>) {
    match expr {
        PatternExpr::Bin {
            op: BinTag::Add,
            a,
            b,
        } => {
            collect_add_terms(a, out);
            collect_add_terms(b, out);
        }
        PatternExpr::Literal(v) if *v == 0f64.to_bits() => {}
        _ => out.push(expr),
    }
}

fn collect_term_slots(expr: &PatternExpr, out: &mut Vec<usize>) {
    match expr {
        PatternExpr::Load(slot) => out.push(*slot),
        PatternExpr::Literal(_) => {}
        PatternExpr::Bin { a, b, .. } => {
            collect_term_slots(a, out);
            collect_term_slots(b, out);
        }
        PatternExpr::Unary { input, .. } => collect_term_slots(input, out),
    }
}

fn canonicalize_term(
    expr: &PatternExpr,
    group: &StoreGroup,
    slot_map: &mut HashMap<usize, usize>,
    tensors: &mut Vec<GlobalId>,
) -> Option<ReductionTermPattern> {
    match expr {
        PatternExpr::Load(slot) => {
            let idx = if let Some(&idx) = slot_map.get(slot) {
                idx
            } else {
                let idx = tensors.len();
                tensors.push(*group.load_tensors.get(*slot)?);
                slot_map.insert(*slot, idx);
                idx
            };
            Some(ReductionTermPattern::Load(idx))
        }
        PatternExpr::Literal(bits) => Some(ReductionTermPattern::Literal(*bits)),
        PatternExpr::Bin { op, a, b } => Some(ReductionTermPattern::Bin {
            op: (*op).into(),
            a: Box::new(canonicalize_term(a, group, slot_map, tensors)?),
            b: Box::new(canonicalize_term(b, group, slot_map, tensors)?),
        }),
        PatternExpr::Unary { op, input } => Some(ReductionTermPattern::Unary {
            op: (*op).into(),
            input: Box::new(canonicalize_term(input, group, slot_map, tensors)?),
        }),
    }
}

fn infer_reduction_accesses(
    group: &StoreGroup,
    output_shape: &[usize],
    shapes: &HashMap<GlobalId, Vec<usize>>,
    reduction: &ReductionMatch,
) -> Result<Vec<RecoveredTensorAccess>, PipelineError> {
    let mut instances = group.instances.clone();
    instances.sort_by_key(|i| i.out_flat_index);
    let output_rank = output_shape.len();
    let output_step = sampling_step(instances.len(), 128);
    let term_step = sampling_step(reduction.terms, 128);

    let mut output_coords = Vec::with_capacity(instances.len());
    for inst in &instances {
        let coords = unflatten_index(inst.out_flat_index, output_shape).ok_or(
            PipelineError::InvalidFlatIndex {
                tensor: group.output_tensor,
                flat_index: inst.out_flat_index,
                shape: output_shape.to_vec(),
            },
        )?;
        output_coords.push(coords);
    }

    let mut accesses = Vec::with_capacity(reduction.term_load_tensors.len());

    for (load_idx, &tensor) in reduction.term_load_tensors.iter().enumerate() {
        let tensor_shape = shapes
            .get(&tensor)
            .cloned()
            .ok_or(PipelineError::MissingShape(tensor))?;
        let var_count = output_rank + 1;
        let mut sample_vars_flat: Vec<isize> = Vec::new();
        let mut sample_dim_values: Vec<Vec<isize>> = vec![Vec::new(); tensor_shape.len()];
        let mut samples_valid = true;

        for inst_i in (0..instances.len()).step_by(output_step) {
            let inst = &instances[inst_i];
            for term in (0..reduction.terms).step_by(term_step) {
                let Some(slot) = reduction
                    .term_slots
                    .get(term)
                    .and_then(|slots| slots.get(load_idx))
                    .copied()
                else {
                    samples_valid = false;
                    break;
                };

                let Some(&flat_index) = inst.load_indices.get(slot) else {
                    samples_valid = false;
                    break;
                };

                let coords = unflatten_index(flat_index, &tensor_shape).ok_or(
                    PipelineError::InvalidFlatIndex {
                        tensor,
                        flat_index,
                        shape: tensor_shape.clone(),
                    },
                )?;

                sample_vars_flat.extend(output_coords[inst_i].iter().map(|&v| v as isize));
                sample_vars_flat.push(term as isize);
                for dim in 0..coords.len() {
                    sample_dim_values[dim].push(coords[dim] as isize);
                }
            }
            if !samples_valid {
                break;
            }
        }

        let sample_count = sample_vars_flat.len() / var_count;
        let dim_roles = if !samples_valid || sample_count == 0 {
            vec![AccessDimRole::Unknown; tensor_shape.len()]
        } else {
            let mut roles = Vec::with_capacity(tensor_shape.len());
            for dim_values in sample_dim_values.iter().take(tensor_shape.len()) {
                let role =
                    infer_affine_dim_role(&sample_vars_flat, var_count, dim_values, output_rank);
                roles.push(role);
            }
            roles
        };

        accesses.push(RecoveredTensorAccess {
            tensor,
            tensor_shape,
            dim_roles,
        });
    }

    Ok(accesses)
}

fn infer_affine_dim_role(
    vars_flat: &[isize],
    var_count: usize,
    values: &[isize],
    output_rank: usize,
) -> AccessDimRole {
    let Some((offset, coeffs)) = infer_affine_coefficients(vars_flat, var_count, values) else {
        return AccessDimRole::Unknown;
    };

    if coeffs.len() != output_rank + 1 {
        return AccessDimRole::Unknown;
    }

    let reduction_stride = coeffs[output_rank];
    let output_strides: Vec<(usize, isize)> = coeffs[..output_rank]
        .iter()
        .enumerate()
        .filter_map(|(axis, &stride)| (stride != 0).then_some((axis, stride)))
        .collect();

    if output_strides.is_empty() && reduction_stride == 0 {
        if offset >= 0 {
            return AccessDimRole::Constant {
                value: offset as usize,
            };
        }
        return AccessDimRole::Unknown;
    }

    if output_strides.len() == 1 && reduction_stride == 0 {
        let (axis, stride) = output_strides[0];
        return AccessDimRole::OutputAxis {
            axis,
            stride,
            offset,
        };
    }

    if output_strides.is_empty() && reduction_stride != 0 {
        return AccessDimRole::ReductionAxis {
            stride: reduction_stride,
            offset,
        };
    }

    AccessDimRole::AffineMixed {
        output_strides,
        reduction_stride,
        offset,
    }
}

fn infer_affine_coefficients(
    vars_flat: &[isize],
    var_count: usize,
    values: &[isize],
) -> Option<(isize, Vec<isize>)> {
    if values.is_empty() || var_count == 0 || vars_flat.len() != values.len() * var_count {
        return None;
    }

    let sample_count = values.len();

    let mut coeffs = vec![0isize; var_count];

    for axis in 0..var_count {
        let mut maybe_coeff = None;
        let mut seen: HashMap<Vec<isize>, (isize, isize)> = HashMap::new();

        for i in 0..sample_count {
            let row = &vars_flat[i * var_count..(i + 1) * var_count];
            let mut key = Vec::with_capacity(var_count - 1);
            key.extend_from_slice(&row[..axis]);
            key.extend_from_slice(&row[axis + 1..]);

            if let Some((prev_axis_val, prev_value)) = seen.get(&key).copied() {
                let dv = row[axis] - prev_axis_val;
                if dv == 0 {
                    continue;
                }
                let dy = values[i] - prev_value;
                if dy % dv != 0 {
                    return None;
                }
                let candidate = dy / dv;
                match maybe_coeff {
                    Some(prev) if prev != candidate => return None,
                    Some(_) => {}
                    None => maybe_coeff = Some(candidate),
                }
            } else {
                seen.insert(key, (row[axis], values[i]));
            }
        }

        coeffs[axis] = maybe_coeff.unwrap_or(0);
    }

    let row0 = &vars_flat[..var_count];
    let bias = values[0]
        - row0
            .iter()
            .zip(coeffs.iter())
            .map(|(v, c)| v * c)
            .sum::<isize>();

    for i in 0..sample_count {
        let row = &vars_flat[i * var_count..(i + 1) * var_count];
        let value = values[i];
        let predicted = bias
            + row
                .iter()
                .zip(coeffs.iter())
                .map(|(v, c)| v * c)
                .sum::<isize>();
        if predicted != value {
            return None;
        }
    }

    Some((bias, coeffs))
}

fn sampling_step(total: usize, target_samples: usize) -> usize {
    if total <= target_samples {
        1
    } else {
        (total / target_samples).max(1)
    }
}

fn unflatten_index(flat_index: usize, shape: &[usize]) -> Option<Vec<usize>> {
    if shape.is_empty() {
        return (flat_index == 0).then(Vec::new);
    }

    let total = shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))?;
    if total == 0 || flat_index >= total {
        return None;
    }

    let mut rem = flat_index;
    let mut coords = vec![0usize; shape.len()];
    for axis in (0..shape.len()).rev() {
        let dim = shape[axis];
        coords[axis] = rem % dim;
        rem /= dim;
    }
    Some(coords)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::common::v1_frontend::nano_op::NanoOpExpander;
    use crate::graph::GlobalId;
    use crate::milli_graph::ops::{MatMul, SimpleBinary};

    #[test]
    fn test_classifies_pointwise_add_as_pointwise() {
        let mut rng = wyrand::WyRand::new(11);
        let ext_a = GlobalId::new(&mut rng);
        let ext_b = GlobalId::new(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let a = input_map[&ext_a];
        let b = input_map[&ext_b];
        let c = SimpleBinary::add(&mut graph, a, b, &mut rng);

        let mut shapes = HashMap::new();
        shapes.insert(a, vec![16]);
        shapes.insert(b, vec![16]);
        shapes.insert(c, vec![16]);

        let artifacts = build_from_graph(&graph, &shapes).unwrap();
        assert_eq!(artifacts.schedule.loops.len(), 1);
        assert_eq!(artifacts.schedule.stats.pointwise_families, 1);
        assert_eq!(artifacts.schedule.stats.additive_reduction_families, 0);

        match &artifacts.schedule.loops[0].intent {
            LoopIntent::Pointwise(pointwise) => assert!(pointwise.contiguous_output),
            _ => panic!("expected pointwise intent"),
        }
        let loop0 = &artifacts.schedule.loops[0];
        assert!(!loop0.schedule_candidates.is_empty());
        assert!(loop0.selected_schedule.is_some());
        let best = &loop0.schedule_candidates[loop0.selected_schedule.unwrap()];
        assert!(
            best.loop_order
                .iter()
                .all(|d| matches!(d, LoopDim::OutputAxis(_)))
        );
        assert!(best.score > 0);
    }

    #[test]
    fn test_classifies_matmul_as_additive_reduction_without_matmul_match() {
        let mut rng = wyrand::WyRand::new(12);
        let ext_a = GlobalId::new(&mut rng);
        let ext_b = GlobalId::new(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let a = input_map[&ext_a];
        let b = input_map[&ext_b];
        let c = MatMul::push_new(&mut graph, a, b, &mut rng);

        let mut shapes = HashMap::new();
        shapes.insert(a, vec![4, 8]);
        shapes.insert(b, vec![8, 3]);
        shapes.insert(c, vec![4, 3]);

        let artifacts = build_from_graph(&graph, &shapes).unwrap();
        assert_eq!(artifacts.schedule.loops.len(), 1);
        assert_eq!(artifacts.schedule.stats.additive_reduction_families, 1);
        assert_eq!(artifacts.schedule.stats.pointwise_families, 0);

        let loop0 = &artifacts.schedule.loops[0];
        match &loop0.intent {
            LoopIntent::AdditiveReduction(red) => {
                assert_eq!(red.terms, 8);
                assert_eq!(red.term_load_tensors.len(), 2);
                assert_ne!(red.term_load_tensors[0], red.term_load_tensors[1]);
                assert_eq!(red.accesses.len(), 2);

                let a_access = &red.accesses[0];
                assert_eq!(a_access.tensor, a);
                assert_eq!(a_access.tensor_shape, vec![4, 8]);
                assert_eq!(
                    a_access.dim_roles,
                    vec![
                        AccessDimRole::OutputAxis {
                            axis: 0,
                            stride: 1,
                            offset: 0
                        },
                        AccessDimRole::ReductionAxis {
                            stride: 1,
                            offset: 0
                        }
                    ]
                );

                let b_access = &red.accesses[1];
                assert_eq!(b_access.tensor, b);
                assert_eq!(b_access.tensor_shape, vec![8, 3]);
                assert_eq!(
                    b_access.dim_roles,
                    vec![
                        AccessDimRole::ReductionAxis {
                            stride: 1,
                            offset: 0
                        },
                        AccessDimRole::OutputAxis {
                            axis: 1,
                            stride: 1,
                            offset: 0
                        }
                    ]
                );
            }
            _ => panic!("expected additive reduction intent"),
        }

        assert!(!loop0.schedule_candidates.is_empty());
        assert!(loop0.selected_schedule.is_some());
        let best = &loop0.schedule_candidates[loop0.selected_schedule.unwrap()];
        assert!(best.loop_order.contains(&LoopDim::ReductionAxis));
        assert!(best.vectorize.is_some());
        assert!(best.reduction_tile >= 1);
        assert!(best.score >= 1);
    }

    #[test]
    fn test_rejects_oversized_nano_materialization() {
        let mut rng = wyrand::WyRand::new(13);
        let ext_a = GlobalId::new(&mut rng);
        let ext_b = GlobalId::new(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let a = input_map[&ext_a];
        let b = input_map[&ext_b];
        let c = MatMul::push_new(&mut graph, a, b, &mut rng);

        let mut shapes = HashMap::new();
        shapes.insert(a, vec![256, 384]);
        shapes.insert(b, vec![384, 320]);
        shapes.insert(c, vec![256, 320]);

        let err =
            build_from_graph_with_limit(&graph, &shapes, 8_000_000).expect_err("expected guard");
        match err {
            PipelineError::NanoOpBudgetExceeded {
                estimated_ops,
                limit_ops,
                ..
            } => {
                assert_eq!(limit_ops, 8_000_000);
                assert!(estimated_ops > limit_ops);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_streaming_build_matches_pool_analysis() {
        let mut rng = wyrand::WyRand::new(14);
        let ext_a = GlobalId::new(&mut rng);
        let ext_b = GlobalId::new(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let a = input_map[&ext_a];
        let b = input_map[&ext_b];
        let c = MatMul::push_new(&mut graph, a, b, &mut rng);

        let mut shapes = HashMap::new();
        shapes.insert(a, vec![8, 16]);
        shapes.insert(b, vec![16, 6]);
        shapes.insert(c, vec![8, 6]);

        let mut expander = NanoOpExpander::new(shapes.clone());
        let ordered = expander
            .expand_iter(&graph)
            .collect::<Result<Vec<_>, _>>()
            .expect("expand nano stream");
        let pool_schedule = analyze_from_pool(&ordered, &shapes).expect("pool schedule");

        let stream_artifacts = build_from_graph(&graph, &shapes).expect("stream build");
        assert!(stream_artifacts.ordered_nano_ops.is_empty());
        assert!(stream_artifacts.whitewashed_nano_ops.is_empty());
        let stream_schedule = stream_artifacts.schedule;

        assert_eq!(
            stream_schedule.stats.total_nano_ops,
            pool_schedule.stats.total_nano_ops
        );
        assert_eq!(
            stream_schedule.stats.total_stores,
            pool_schedule.stats.total_stores
        );
        assert_eq!(
            stream_schedule.stats.additive_reduction_families,
            pool_schedule.stats.additive_reduction_families
        );
        assert_eq!(
            stream_schedule.stats.pointwise_families,
            pool_schedule.stats.pointwise_families
        );
        assert_eq!(stream_schedule.loops.len(), pool_schedule.loops.len());
        assert_eq!(
            stream_schedule.loops[0].selected_schedule,
            pool_schedule.loops[0].selected_schedule
        );
    }
}
